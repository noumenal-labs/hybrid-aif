"""Typed factor declarations: the node layer.

Declaration objects whose type is the family tag; ``SystemParams`` fields are unions over these
and the tabular families, dispatched by ``isinstance`` at build time (trace-legal, since pytree
structure is static). :class:`Deterministic` is a pure-map node; :class:`Recognition` and
:class:`DerivedInitial` are the structural declarations of the amortized path.

Design: docs/design/hybrid-system.md D1; docs/design/parameter-learning.md D3, D7, D8;
docs/design/amortization.md A1, A2, A10, A11
"""

import itertools
import math
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.linalg import cho_solve, solve_triangular
from jax.scipy.special import gammaln

from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalDirichlet,
    ConditionalGaussian,
    Dirichlet,
    InverseGamma,
    InverseWishart,
    MatrixNormal,
)
from hybrid_aif.distributions.neural import (
    GRUCell,
    LearnedVector,
    MLPCategorical,
    MLPConditional,
    MLPPredictor,
    PredictorHead,
    apply_gradient,
    init_opt_state,
)
from hybrid_aif.distributions.noise_priors import _anchored, _check_nu_floor

_LOG_PI = math.log(math.pi)


def _student_t_log_density(y: Array, loc: Array, scale: Array, df: Array) -> Array:
    """The multivariate Student-t log-density of ``y`` (``(..., d)``) at location ``loc``, scale
    matrix ``scale`` (``(..., d, d)``, positive definite) and ``df`` degrees of freedom."""
    d = y.shape[-1]
    chol = jnp.linalg.cholesky(scale)
    z = solve_triangular(chol, (y - loc)[..., None], lower=True)[..., 0]
    maha = jnp.sum(z**2, axis=-1)
    half_log_det = jnp.sum(jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)
    return (
        gammaln(0.5 * (df + d))
        - gammaln(0.5 * df)
        - 0.5 * d * (jnp.log(df) + _LOG_PI)
        - half_log_det
        - 0.5 * (df + d) * jnp.log1p(maha / df)
    )


# Named tabular unions: each slot accepts its point form or the parameter
# distribution over it — the same tabular family with an uncertain table, not a
# new family. The classifiers (`System._is_hybrid`,
# `objectives.hybrid.factor_families`, the observation classifier) dispatch on
# these unions, never on special-cased Dirichlet branches.
TabularObservation = ConditionalCategorical | ConditionalDirichlet
TabularTransition = ConditionalCategorical | ConditionalDirichlet
TabularPrior = Categorical | ConditionalCategorical | Dirichlet | ConditionalDirichlet


class LinearGaussianTransition(eqx.Module):
    """``x' = F_j x + G_j u + b_j + eps_j``: linear map(s) with conditioned innovations.

    ``innovations`` is a :class:`ConditionalGaussian` whose parent axes are the factor's discrete
    parents in declared (``B_deps``) order: its discrete state parents and any discrete control
    node, each a mode index ``j``; a factor with no discrete parent declares the trivial parent
    shape ``(1,)`` (always index 0). ``F`` is ``(d, d)``, shared across modes, or
    ``(*parent_shape, d, d)`` matching ``innovations.parent_shape``, a per-mode map (the SLDS
    pattern). ``G`` is the control input map of the factor's continuous control nodes, ``u``
    their values concatenated in declared order: ``(d, du)`` shared, or ``(*parent_shape, d, du)``
    per mode on the rule ``F`` follows; ``None`` when the factor reads no continuous node.
    """

    F: Array  # (d, d) shared, or (*innovations.parent_shape, d, d) per-mode.
    innovations: ConditionalGaussian
    G: Array | None = None  # (d, du) shared, or (*innovations.parent_shape, d, du) per-mode.

    @property
    def dim(self) -> int:
        return self.F.shape[-1]


#: The moment propagations a :class:`NonlinearGaussianTransition` declares.
PROPAGATIONS = ("first_order", "unscented")


class NonlinearGaussianTransition(eqx.Module):
    """``x' = f(x, controls) + eps``: a user-written mean map with conditioned innovations.

    ``f`` is an :class:`equinox.Module` called as ``f(x, controls)``: ``x`` the factor's
    ``(d,)`` state, ``controls`` a tuple with one entry per control parent of the factor in
    declared order (a discrete node's integer level, a continuous node's float vector); it
    returns the ``(d,)`` mean of the next state. Its numeric fields are pytree leaves and its
    code is static structure. ``innovations`` is a :class:`ConditionalGaussian` added to
    ``f``'s output, ``eps ~ N(loc_j, chol_j chol_jᵀ)``, whose parent axes are the factor's
    discrete control nodes in declared order (the trivial ``(1,)`` for none); its Cholesky
    factor's diagonal must be positive for beliefs to be propagated.

    ``propagation`` (static) is how a belief is pushed through ``f``: ``"first_order"``
    linearises ``f`` at the belief's mean, ``"unscented"`` pushes ``2n + 1`` sigma points through
    it (``n`` the state's dimension plus that of any control carried as a belief). Both are
    exact for ``f`` affine in its continuous inputs. A factor declaring this transition has no
    discrete state parents; it is Gaussian-carried, or mixture-carried when its prior is a
    :class:`~hybrid_aif.distributions.MixtureGaussian`. ``f`` is not learned.
    Design: docs/design/nonlinear-transitions.md N1–N5.
    """

    f: eqx.Module
    innovations: ConditionalGaussian
    propagation: str = eqx.field(static=True, default="first_order")

    def __check_init__(self) -> None:
        if not isinstance(self.f, eqx.Module) or not callable(self.f):
            raise TypeError(
                "NonlinearGaussianTransition: f must be a callable equinox.Module, "
                f"f(x, controls) -> mean, got a {type(self.f).__name__}; write the map as a "
                "module whose numeric fields are its parameters"
            )
        if self.propagation not in PROPAGATIONS:
            raise ValueError(
                f"NonlinearGaussianTransition: propagation={self.propagation!r}; pass one of "
                f"{list(PROPAGATIONS)}"
            )

    @property
    def dim(self) -> int:
        return self.innovations.dim


class NoiseMap(eqx.Module):
    """An observation noise that varies over the observed state: a module that gives
    ``R_chol`` at a mean. Subclass it to declare one; the consumers read a map at the belief's
    mean (design: planning-with-observations.md D3)."""

    def __call__(self, loc: Array) -> Array:
        """``R_chol`` at ``loc`` (``(..., d)``): ``(..., d_obs, d_obs)``."""
        raise NotImplementedError


def _is_noise_map(noise: object) -> bool:
    return isinstance(noise, NoiseMap)


class GridNoiseMap(NoiseMap):
    """An observation noise that varies over the observed state: ``R_chol`` at the nodes of a
    rectangular grid over some of the state's coordinates, read between nodes by multilinear
    interpolation (design: planning-with-observations.md D3).

    ``values`` is ``(*grid_shape, d_obs, d_obs)``, a lower-triangular Cholesky factor with a
    positive diagonal at every node; interpolating such factors keeps both. ``axes`` names the
    state coordinates the grid spans, in order; ``low`` and ``high`` are the grid's extent per
    axis, one number each, and a mean outside the extent reads the nearest edge. The map is
    differentiable in the mean almost everywhere, so the gradient planner sees it; a grid
    blurred before declaration gives it a smooth gradient. ``values``, ``low`` and ``high`` are
    leaves and only ``axes`` is static, so a new grid or a new extent is a new params, never a
    new trace: a map in a frame that moves with the agent is resampled onto a fixed body-frame
    grid each step, the grid's own axes staying aligned with the state's. A point map: no
    prior and no learning.
    """

    values: Array
    low: Array
    high: Array
    axes: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, values: Array, low, high, axes: tuple[int, ...]):
        self.values = jnp.asarray(values)
        self.axes = tuple(int(a) for a in axes)
        k = len(self.axes)
        self.low = jnp.broadcast_to(jnp.asarray(low, self.values.dtype), (k,))
        self.high = jnp.broadcast_to(jnp.asarray(high, self.values.dtype), (k,))

    def __check_init__(self) -> None:
        k = len(self.axes)
        if self.values.ndim != k + 2:
            raise ValueError(
                f"GridNoiseMap: values has {self.values.ndim} axes, expected {k + 2}: one grid "
                f"axis per entry of axes={self.axes} and the (d_obs, d_obs) factor"
            )
        if self.values.shape[-1] != self.values.shape[-2]:
            raise ValueError(
                f"GridNoiseMap: values' trailing shape {self.values.shape[-2:]} is not square"
            )
        if any(n < 2 for n in self.values.shape[:k]):
            raise ValueError(
                "GridNoiseMap: every grid axis needs at least two nodes, got "
                f"{self.values.shape[:k]}"
            )
        if len(set(self.axes)) != k:
            raise ValueError(f"GridNoiseMap: axes={self.axes} names a coordinate twice")
        if any(a < 0 for a in self.axes):
            raise ValueError(f"GridNoiseMap: axes={self.axes} must be non-negative coordinates")
        if not isinstance(self.low, jax.core.Tracer) and not bool(jnp.all(self.high > self.low)):
            raise ValueError("GridNoiseMap: every axis needs low < high")
        if not isinstance(self.values, jax.core.Tracer):
            v = np.asarray(self.values)
            upper = np.triu(v, 1)
            diag = np.diagonal(v, axis1=-2, axis2=-1)
            if not (np.all(upper == 0.0) and np.all(diag > 0.0)):
                raise ValueError(
                    "GridNoiseMap: values must be lower-triangular Cholesky factors with a "
                    "positive diagonal at every node"
                )

    @property
    def obs_dim(self) -> int:
        return self.values.shape[-1]

    def __call__(self, loc: Array) -> Array:
        """``R_chol`` at ``loc`` (``(..., d)``): ``(..., d_obs, d_obs)``."""
        loc = jnp.asarray(loc)
        k = len(self.axes)
        sizes = jnp.asarray(self.values.shape[:k], loc.dtype)
        coords = jnp.stack([loc[..., a] for a in self.axes], axis=-1)
        coords = (coords - self.low) / (self.high - self.low) * (sizes - 1.0)
        coords = jnp.clip(coords, 0.0, sizes - 1.0)
        base = jnp.minimum(jnp.floor(coords), sizes - 2.0)
        frac = coords - base
        base = base.astype(jnp.int32)
        out = jnp.zeros((*loc.shape[:-1], *self.values.shape[-2:]), self.values.dtype)
        for corner in itertools.product((0, 1), repeat=k):
            index = tuple(base[..., j] + corner[j] for j in range(k))
            weight = jnp.ones(loc.shape[:-1], self.values.dtype)
            for j in range(k):
                weight = weight * (frac[..., j] if corner[j] else 1.0 - frac[..., j])
            out = out + weight[..., None, None] * self.values[index]
        return out


class LinearGaussianObservation(eqx.Module):
    """``y = H x + c + eps``: an LG modality over one Gaussian factor.

    ``head`` is the augmented ``[H | c]``, shape ``(*parent_shape, d_obs, d + 1)`` (or a
    ``MatrixNormal`` over it); construct with ``H=``/``c=`` to have it assembled.
    ``H``, ``c`` and the noise are each either shared (``(d_obs, d)`` / ``(d_obs,)`` /
    ``(d_obs, d_obs)``) or per-mode, with leading parent axes ``parent_shape`` (the categorical
    parents named after the observed factor in ``A_deps``), broadcast like the per-mode ``F_j`` of
    :class:`LinearGaussianTransition`. ``noise`` holds a point ``R_chol`` or a posterior over it
    (:class:`~hybrid_aif.distributions.InverseGamma`,
    :class:`~hybrid_aif.distributions.InverseWishart`), which makes it learnable and samplable.
    Every inference consumer reads the :attr:`R_chol` property, which lowers a distributional noise
    by its expected precision.
    """

    head: Array | MatrixNormal  # the augmented [H | c]: (d_obs, dx+1), or
    #   (*parent_shape, d_obs, dx+1) per-mode; point value or the posterior
    #   over it (never separate H/c posteriors — they are coupled)
    noise: Array | InverseGamma | InverseWishart  # point R_chol (lower-triangular
    #   (d_obs, d_obs), or (*parent_shape, d_obs, d_obs) per-mode), or the
    #   conjugate posterior over R

    def __init__(self, head=None, noise=None, *, H=None, c=None):
        if head is None:
            if H is None or c is None:
                raise ValueError("pass head=[H | c] augmented, or both H= and c=")
            H, c = jnp.asarray(H), jnp.asarray(c)
            batch = jnp.broadcast_shapes(H.shape[:-2], c.shape[:-1])
            head = jnp.concatenate(
                [
                    jnp.broadcast_to(H, (*batch, *H.shape[-2:])),
                    jnp.broadcast_to(c, (*batch, c.shape[-1]))[..., None],
                ],
                axis=-1,
            )
        elif H is not None or c is not None:
            raise ValueError("pass head= or (H=, c=), not both")
        self.head = head
        self.noise = noise

    @property
    def _head_value(self) -> Array:
        return self.head.point() if isinstance(self.head, MatrixNormal) else self.head

    @property
    def H(self) -> Array:
        """The map part of the head (point value, or the posterior mean)."""
        return self._head_value[..., :, :-1]

    @property
    def c(self) -> Array:
        """The offset part of the head (point value, or the posterior mean)."""
        return self._head_value[..., :, -1]

    @property
    def learnable_head(self) -> bool:
        """Capability, per-field: the head slot holds a posterior."""
        return isinstance(self.head, MatrixNormal)

    @property
    def R_chol(self) -> Array:
        """The noise as an ``R_chol`` array: the point value itself, or the
        expected-precision lowering of a distributional noise. A noise map has no one value:
        read it at a mean with :meth:`R_chol_at`."""
        if isinstance(self.noise, InverseGamma | InverseWishart):
            return self.noise.point()
        if _is_noise_map(self.noise):
            raise TypeError(
                "this observation's noise is a map over the observed state, so it has no one "
                "R_chol; read it at a belief's mean with R_chol_at(loc)"
            )
        return self.noise

    @property
    def learnable_noise(self) -> bool:
        """Capability, per-field: the slot holds a posterior, not a value."""
        return isinstance(self.noise, InverseGamma | InverseWishart)

    @property
    def noise_map(self) -> bool:
        """Capability, per-field: the noise is a :class:`NoiseMap` over the observed state, not
        a value or a posterior."""
        return _is_noise_map(self.noise)

    def R_chol_at(self, loc: Array) -> Array:
        """The noise as an ``R_chol`` array at a belief's mean ``loc``: a point or conjugate
        noise is the same everywhere; a noise map is evaluated there, the extended filter's
        convention (design: planning-with-observations.md D3)."""
        if self.noise_map:
            return self.noise(loc)
        return self.R_chol

    @property
    def obs_dim(self) -> int:
        return self._head_value.shape[-2]


class LinearGaussianObservationMNIW(eqx.Module):
    """Head and noise under one matrix-normal-inverse-Wishart posterior.

    ``R ~ IW(Psi, nu)`` and ``W | R ~ MN(M, R, V)`` over the augmented head ``W = [H | c]``: an
    inverse-Wishart prior on the noise and a matrix-normal prior on the head whose row covariance
    is that noise, so the head is not independent of the noise (a model description writes the
    pair as two priors, the head's naming the noise as its row covariance; design:
    model-description-0.3.md Q18.3). Satisfies ``Learnable`` at declaration level:
    ``point()`` and ``sample(key)`` return point-valued :class:`LinearGaussianObservation` heads.
    Consumers read the same ``H``/``c``/``R_chol`` properties (mean head, expected-precision noise).
    Leading axes of all four parameters are parent (mode) axes.
    """

    M: Array  # (..., d_obs, p) posterior mean head, p = dx + 1
    V_inv: Array  # (..., p, p) column precision
    Psi: Array  # (..., d_obs, d_obs) IW scale
    nu: Array  # (...,) IW degrees of freedom
    M0: Array | None = None  # decay anchors; required iff decay < 1
    V_inv0: Array | None = None
    Psi0: Array | None = None
    nu0: Array | None = None

    def __check_init__(self) -> None:
        d, p = self.M.shape[-2:]
        if self.V_inv.shape[-2:] != (p, p) or self.Psi.shape[-2:] != (d, d):
            raise ValueError(
                f"MNIW shape mismatch: M {self.M.shape}, V_inv {self.V_inv.shape}, "
                f"Psi {self.Psi.shape}"
            )
        _check_nu_floor(self.nu, d)

    @classmethod
    def from_head_noise(
        cls, head: Array, R_chol: Array, strength: float = 1.0, anchor: bool = False
    ):
        """Prior centred on a known head and noise: the ``H``/``c``/``R_chol``
        properties reproduce them exactly; ``strength`` counts
        pseudo-observations on both the column precision and ``nu``."""
        d, p = head.shape[-2:]
        batch = jnp.broadcast_shapes(head.shape[:-2], R_chol.shape[:-2])
        v_inv = strength * jnp.broadcast_to(jnp.eye(p, dtype=head.dtype), (*batch, p, p))
        r = R_chol @ jnp.swapaxes(R_chol, -1, -2)
        nu = jnp.broadcast_to(jnp.asarray(float(d) + 1.0 + strength, head.dtype), batch)
        psi = jnp.broadcast_to(r, (*batch, d, d)) * nu[..., None, None]
        m = jnp.broadcast_to(head, (*batch, d, p))
        return cls(
            M=m,
            V_inv=v_inv,
            Psi=psi,
            nu=nu,
            M0=m if anchor else None,
            V_inv0=v_inv if anchor else None,
            Psi0=psi if anchor else None,
            nu0=nu if anchor else None,
        )

    # ------------------------------------------------- the shared consumer surface
    @property
    def obs_dim(self) -> int:
        return self.M.shape[-2]

    @property
    def head(self) -> Array:
        return self.M

    @property
    def H(self) -> Array:
        return self.M[..., :, :-1]

    @property
    def c(self) -> Array:
        return self.M[..., :, -1]

    @property
    def R_chol(self) -> Array:
        """Expected-precision lowering, exactly as ``InverseWishart.point()``."""
        return jnp.linalg.cholesky(self.Psi / self.nu[..., None, None])

    @property
    def learnable_head(self) -> bool:
        return True

    @property
    def learnable_noise(self) -> bool:
        return True

    @property
    def noise_map(self) -> bool:
        return False

    def R_chol_at(self, loc: Array) -> Array:
        """The expected-precision noise, the same at every mean."""
        del loc
        return self.R_chol

    # ------------------------------------------------------- Learnable protocol
    def point(self) -> LinearGaussianObservation:
        """The posterior-mean point declaration."""
        return LinearGaussianObservation(head=self.M, noise=self.R_chol)

    def sample(self, key: Array) -> LinearGaussianObservation:
        """One joint draw: ``R ~ IW``, then ``W | R ~ MN(M, R, V)``."""
        kr, kw = jax.random.split(key)
        r_chol = InverseWishart(Psi=self.Psi, nu=self.nu).sample(kr)
        w = MatrixNormal(M=self.M, V_inv=self.V_inv).sample(kw, row_chol=r_chol)
        return LinearGaussianObservation(head=w, noise=r_chol)

    # ------------------------------------------------------- posterior predictive
    def predictive(self, x_aug: Array) -> tuple[Array, Array, Array]:
        """The posterior predictive of ``y = W x̃ + e`` at the regressor ``x_aug`` (``(..., p)``,
        broadcast against the parent axes), a multivariate Student-t: ``(location, scale,
        degrees of freedom)``, ``M x̃``, ``Ψ (1 + x̃ᵀ V x̃) / (ν − d + 1)`` and ``ν − d + 1``."""
        d = self.obs_dim
        loc = (self.M @ x_aug[..., None])[..., 0]
        v_chol = jnp.linalg.cholesky(self.V_inv)
        vx = cho_solve((v_chol, True), x_aug[..., None])[..., 0]
        quad = jnp.sum(x_aug * vx, axis=-1)
        df = self.nu - d + 1.0
        scale = self.Psi * ((1.0 + quad) / df)[..., None, None]
        return loc, scale, df

    def predictive_log_prob(self, y: Array, x_aug: Array) -> Array:
        """``log p(y | x̃)`` under the posterior predictive of :meth:`predictive`: the head and
        the noise integrated out, the exact conjugate answer."""
        return _student_t_log_density(y, *self.predictive(x_aug))

    def stats(self, syy: Array, syx: Array, sxx: Array, count: Array):
        """Package one tick's (responsibility-weighted) joint statistic:
        ``syy = Σ y yᵀ``, ``syx = Σ y E[x̃]ᵀ``, ``sxx = Σ E[x̃ x̃ᵀ]``, mass."""
        return syy, syx, sxx, count

    def absorb(self, stats, lr=1.0, decay=1.0) -> "LinearGaussianObservationMNIW":
        """The MNIW conjugate update, accumulated in natural coordinates
        ``(V⁻¹, M V⁻¹, Φ = Ψ + M V⁻¹ Mᵀ, ν)`` so decay anchors apply per
        natural parameter, then converted back."""
        syy, syx, sxx, count = stats
        nat_mean = self.M @ self.V_inv
        phi = self.Psi + nat_mean @ jnp.swapaxes(self.M, -1, -2)
        nat0 = None if self.M0 is None else self.M0 @ self.V_inv0
        phi0 = (
            None
            if self.Psi0 is None
            else self.Psi0 + (self.M0 @ self.V_inv0) @ jnp.swapaxes(self.M0, -1, -2)
        )
        new_vinv = _anchored(self.V_inv, self.V_inv0, sxx, lr, decay, "V_inv0")
        new_nat = _anchored(nat_mean, nat0, syx, lr, decay, "M0")
        new_phi = _anchored(phi, phi0, syy, lr, decay, "Psi0")
        new_nu = _anchored(self.nu, self.nu0, count, lr, decay, "nu0")
        new_m = jnp.swapaxes(
            jnp.linalg.solve(jnp.swapaxes(new_vinv, -1, -2), jnp.swapaxes(new_nat, -1, -2)),
            -1,
            -2,
        )
        new_psi = new_phi - new_nat @ jnp.swapaxes(new_m, -1, -2)
        return LinearGaussianObservationMNIW(
            M=new_m,
            V_inv=new_vinv,
            Psi=new_psi,
            nu=new_nu,
            M0=self.M0,
            V_inv0=self.V_inv0,
            Psi0=self.Psi0,
            nu0=self.nu0,
        )

    def expected_log_params(self) -> Array:
        raise NotImplementedError(
            "LG inference consumes the point head and expected-precision noise; "
            "no E[log] consumer exists (design: parameter-learning.md D8)"
        )

    def novelty(self) -> Array:
        raise NotImplementedError(
            "an MNIW emission's novelty depends on the state belief, so it has no table form: "
            "it is the 'blended_entropy' infogain estimator of hybrid_aif.objectives.registry "
            "(continuous.param_info_gain), mounted by `epistemic` (design: "
            "parameter-learning.md D10)"
        )


# Every consumer that reads H/c/R_chol accepts either declaration; positive
# isinstance filters MUST use this union, never the point class alone (the
# silent-drop hazard; see docs/design/parameter-learning.md D3).
LGObservation = LinearGaussianObservation | LinearGaussianObservationMNIW


def gaussian_emission(
    loc: Array, R_chol: Array, strength: float = 1.0, anchor: bool = False
) -> LinearGaussianObservationMNIW:
    """A Gaussian emission over categorical parents alone, ``y = c_j + noise_j`` with the
    parents' configuration ``j`` selecting the offset and the noise: the matrix-normal
    inverse-Wishart family with an empty regressor block, its head the offset column only.

    ``loc`` is ``(*parent_shape, d_obs)``, one offset per parent configuration (a Gaussian
    mixture's component means, the parent its component label); ``R_chol`` is per
    configuration, ``(*parent_shape, d_obs, d_obs)``, or shared, ``(d_obs, d_obs)``.
    ``strength`` and ``anchor`` are :meth:`LinearGaussianObservationMNIW.from_head_noise`'s. A
    modality declared this way names only categorical factors in ``A_deps``; its message to
    them is the exact categorical message of its predictive density, and it learns by the
    responsibility-weighted conjugate update at zero regressor width.

    Design: docs/design/structure-learning.md S2
    """
    loc = jnp.asarray(loc)
    return LinearGaussianObservationMNIW.from_head_noise(
        loc[..., None], jnp.asarray(R_chol), strength=strength, anchor=anchor
    )


def lg_point(model: LGObservation) -> LinearGaussianObservation:
    """The point-valued form of an LG observation declaration.

    A distributional slot is lowered once (posterior-mean head, expected-precision noise); a point
    declaration is returned as is.
    """
    if isinstance(model, LinearGaussianObservationMNIW):
        return model.point()
    if model.learnable_head or model.learnable_noise:
        noise = model.noise if model.noise_map else model.R_chol
        return LinearGaussianObservation(head=model._head_value, noise=noise)
    return model


#: The message treatments a :class:`Deterministic` node may declare. Only ``"carry"`` is
#: built (the value is computed forward and carried); the rest are named and refused.
_RULES = ("carry",)
_UNBUILT_RULES = {
    "linear_exact": "a linear map's exact pushforward",
    "sigma_point": "sigma-point / unscented forward messages with an RTS-type backward pass",
    "quadrature": "quadrature forward messages with an RTS-type backward pass",
}


class Deterministic(eqx.Module):
    """A deterministic node: a pure map, optionally with trainable parameters.

    ``map`` is any pure callable pytree (an ``equinox.Module`` whose array leaves are its
    parameters, or a bare function with none); :meth:`gru` builds the RSSM carry case. A recurrent
    map takes its carry as an argument and returns the next one; nothing is stashed on the module.
    ``rule`` is the per-node message treatment, a static string from :data:`_RULES`; only
    ``"carry"`` (computed forward, never inferred) is built. ``frozen`` is the trainable marking.
    The gradient statistic arrives from whatever consumes the node's output, and :meth:`absorb`
    applies it through the shared gradient path.

    Design: docs/design/system.md (Factors)
    """

    map: object  # a pure callable; its array leaves are the node's parameters
    rule: str = eqx.field(static=True, default="carry")
    frozen: bool = eqx.field(static=True, default=False)

    def __check_init__(self) -> None:
        if not callable(self.map):
            raise ValueError(
                f"Deterministic.map must be a pure callable (an equinox.Module or a plain "
                f"function), got {type(self.map).__name__}"
            )
        if self.rule in _UNBUILT_RULES:
            raise NotImplementedError(
                f"Deterministic rule {self.rule!r} ({_UNBUILT_RULES[self.rule]}) is designed but "
                "not supported: messages *through* a deterministic map are not computed; the "
                "RSSM's deter is computed forward and carried. Declare rule='carry' (design: "
                "docs/design/system.md 'Factors'; amortization.md A10)."
            )
        if self.rule not in _RULES:
            raise ValueError(
                f"unknown Deterministic rule {self.rule!r}; built rules are {list(_RULES)} and "
                f"the designed-but-unbuilt ones are {sorted(_UNBUILT_RULES)}"
            )

    @classmethod
    def gru(
        cls,
        deter_dim: int,
        input_dims: tuple[int, ...],
        *,
        key: Array,
        hidden: int,
        activation: str = "silu",
        norm: bool = True,
        frozen: bool = False,
        init: str = "default",
    ) -> "Deterministic":
        """An RSSM's carry node: a :class:`~hybrid_aif.distributions.neural.GRUCell`
        over ``(deter, *inputs)``, with ``input_dims`` the widths of the non-carry
        inputs — for an RSSM ``(num_vars·num_classes, action_dim)``, the previous
        flattened one-hot ``stoch`` and the action. ``init`` is the cell's."""
        return cls(
            map=GRUCell.from_shapes(
                deter_dim,
                input_dims,
                key=key,
                hidden=hidden,
                activation=activation,
                norm=norm,
                init=init,
            ),
            rule="carry",
            frozen=frozen,
        )

    def __call__(self, *args: Array) -> Array:
        """Apply the map. For a recurrent node: ``(carry, *inputs) -> carry'``."""
        return self.map(*args)

    @property
    def learnable(self) -> bool:
        """Capability: a non-frozen node's map parameters are a trainable block."""
        return not self.frozen

    def init_opt_state(self):
        """Fresh optimizer state for the map's parameters — one ``LearnState`` entry."""
        return init_opt_state(self)

    def absorb(self, stats, opt_state, lr: float | Array = 1.0, decay: float | Array = 1.0):
        """One optax step from a gradient pytree: ``(node, opt_state)``.

        ``stats`` is ``∇`` of the consumer's loss with respect to this node.
        """
        return apply_gradient(self, stats, opt_state, lr=lr, decay=decay)


class DerivedInitial(eqx.Module):
    """ "This factor's initial belief is its own transition, at the episode start."

    An ``initial_state_prior`` declaration with no parameters of its own: the factor's ``t=0`` value
    is its transition evaluated at its within-step parents' initialized values (an RSSM's initial
    code, read off the prior at the initial carry). ``reduction`` names how the transition's
    distribution is reduced to the carried value; only ``"mode"`` is built.
    """

    reduction: str = eqx.field(static=True, default="mode")

    def __check_init__(self) -> None:
        if self.reduction == "sample":
            raise NotImplementedError(
                "DerivedInitial(reduction='sample') has no key to draw with: the initial-prior "
                "seam is a pure function of the declarations (System.initial_priors takes no "
                "PRNG key). Sample the initial code yourself if you need one, or declare "
                "reduction='mode'."
            )
        if self.reduction != "mode":
            raise ValueError(
                f"unknown DerivedInitial reduction {self.reduction!r}; the built reduction is "
                "'mode'"
            )


class Recognition(NamedTuple):
    """One amortized variable's structural declaration: its recognition slot.

    A q-side factor: ``variable`` is its child and ``parents`` its conditioning set.
    ``System.recognition`` holds these (the structure); ``SystemParams.recognition[variable]``
    holds the module. :func:`hybrid_aif.inference.amortized.validate_structure` enforces that no
    parent is another amortized variable and no two entries share a module object.

    ``variable`` is a factor name (the amortized posterior for that factor) or any other name (an
    auxiliary q-side variable, such as an observation embedding). ``parents`` is explicit, in the
    module's parent order, each entry one of

    - ``"<factor>"``          a factor's value at this tick (the predicted one, for a variable
                              inferred after it),
    - ``"obs:<modality>"``    this tick's observation of that modality,
    - ``"recognition:<name>"`` an earlier recognition variable's value,
    - ``"external:<name>"``   a vector this System does not declare, supplied per call as data and
                              declared (name, width) in ``System.external_inputs``.

    Evaluation is in declaration order, so a parent must be declared earlier.

    Design: docs/design/amortization.md A11
    """

    variable: str
    parents: tuple[str, ...]


#: Declarations that learn by **gradient** rather than by conjugacy: the
#: neural families, the deterministic node, and the parametrized constant. Every
#: member carries ``learnable``/``init_opt_state``/``absorb`` and is frozen by a
#: static ``frozen`` flag, which is what lets ``System`` treat them uniformly (one
#: ``LearnState`` entry per non-frozen slot, no optimizer state for a frozen one).
GradientDeclaration = (
    MLPConditional | MLPPredictor | MLPCategorical | PredictorHead | Deterministic | LearnedVector
)
