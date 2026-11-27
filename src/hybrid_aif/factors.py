"""Typed factor declarations: the node layer.

Declaration objects whose type is the family tag; ``SystemParams`` fields are unions over these
and the tabular families, dispatched by ``isinstance`` at build time (trace-legal, since pytree
structure is static). :class:`Deterministic` is a pure-map node; :class:`Recognition` and
:class:`DerivedInitial` are the structural declarations of the amortized path.

Design: docs/design/hybrid-system.md D1; docs/design/parameter-learning.md D3, D7, D8;
docs/design/amortization.md A1, A2, A10, A11; docs/design/process-noise-learning.md D1, D4
"""

import dataclasses
import itertools
import math
from typing import Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.linalg import cho_solve, solve_triangular
from jax.scipy.special import gammaln

from hybrid_aif.cells import Grid, cell_mass  # noqa: F401  (re-exported)
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
#: The tabular declarations a ``likelihood`` entry accepts: a child-first conditional table
#: (:class:`ConditionalCategorical`) or a Dirichlet over one (:class:`ConditionalDirichlet`).
TabularObservation = ConditionalCategorical | ConditionalDirichlet
#: The tabular declarations a ``transition`` entry accepts: a child-first conditional table
#: (:class:`ConditionalCategorical`) or a Dirichlet over one (:class:`ConditionalDirichlet`).
TabularTransition = ConditionalCategorical | ConditionalDirichlet
#: The tabular declarations an ``initial_state_prior`` entry accepts: an unconditional
#: :class:`Categorical` or :class:`Dirichlet`, or, for a state factor with ``D_deps`` parents, a
#: :class:`ConditionalCategorical` or :class:`ConditionalDirichlet`.
TabularPrior = Categorical | ConditionalCategorical | Dirichlet | ConditionalDirichlet


class Parameter(eqx.Module):
    """A reference to a named parameter variable, in place of a table.

    A state factor's ``transition`` entry or a modality's ``likelihood`` entry may be
    ``Parameter(name)``, which says the state factor or modality reads the declaration held under
    ``name`` in ``SystemParams.parameters``. Every reader of one name reads one table, and ``learn``
    updates it once from the sum of the readers' count statistics. The token carries no array: the
    name is static, so a reference is a pytree node with no leaves.

    The variable is internal to one particle: its readers share it outright, and another particle
    sees it only as a report of counts (by ``stats``, the counts since the last report; ``mode``
    and ``sample`` of a parameter stay deferred). Design: shared-parameters.md;
    nesting-and-coupling.md N6.
    """

    #: The name of the parameter variable, a key of ``SystemParams.parameters`` (static).
    name: str = eqx.field(static=True)

    def __check_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"Parameter takes a non-empty name, got {self.name!r}")


#: The tabular declarations a parameter variable may hold: a conditional (a point table or a
#: Dirichlet over one), child-first as a transition or likelihood table is, read by a whole
#: ``transition`` or ``likelihood`` entry.
TabularHome = ConditionalCategorical | ConditionalDirichlet
#: The noise declarations a parameter variable may hold: a conjugate prior over a Gaussian noise
#: covariance, read by one field of a declaration (a Gaussian transition's ``noise_prior``, a
#: :class:`LinearGaussianObservation`'s ``noise``). Design: process-noise-learning.md D4.
NoiseHome = InverseGamma | InverseWishart
#: Every kind of declaration a parameter variable (``SystemParams.parameters``) may hold.
ParameterHome = TabularHome | NoiseHome


def _noise_batch(prior: InverseGamma | InverseWishart) -> tuple[int, ...]:
    """A noise prior's leading (mode) axes."""
    if isinstance(prior, InverseGamma):
        return tuple(prior.alpha.shape[:-1])
    return tuple(prior.Psi.shape[:-2])


def _check_noise_prior(owner: str, prior: object, innovations: ConditionalGaussian) -> None:
    """Refuse a transition's ``noise_prior`` that is not a prior over its innovations'
    covariance: the family, the dimension, and the leading axes (none, one prior shared by every
    mode, or the innovations' parent shape, one prior per mode)."""
    if prior is None or isinstance(prior, Parameter):
        return
    if not isinstance(prior, InverseGamma | InverseWishart):
        raise TypeError(
            f"{owner}: noise_prior is an InverseGamma or InverseWishart prior over the "
            f"innovations' covariance (or Parameter(name) naming one), got a "
            f"{type(prior).__name__}"
        )
    d = innovations.dim
    if prior.obs_dim != d:
        raise ValueError(
            f"{owner}: noise_prior is over a {prior.obs_dim}-dimensional noise and the "
            f"innovations are {d}-dimensional"
        )
    batch, modes = _noise_batch(prior), tuple(innovations.parent_shape)
    if batch not in ((), modes, tuple(innovations.batch_shape) + modes):
        raise ValueError(
            f"{owner}: noise_prior has leading axes {batch}; give none (one prior shared by "
            f"every mode) or the innovations' parent shape {modes} (one prior per mode)"
        )


def _lowered_innovations(
    innovations: ConditionalGaussian, prior: InverseGamma | InverseWishart
) -> ConditionalGaussian:
    """The innovations with their Cholesky factor lowered from ``prior``: the expected-precision
    point (``chol(Ψ/ν)``, ``diag(sqrt(β/α))``), shared or per mode as the prior is."""
    return ConditionalGaussian(
        loc=innovations.loc, chol=prior.point(), parent_ndims=innovations.parent_ndims
    )


def _unresolved(owner: str, name: str) -> ValueError:
    return ValueError(
        f"{owner} reads the noise of parameter variable {name!r} (Parameter({name!r})), which "
        "only a System's faces resolve, against SystemParams.parameters; read the lowered "
        "declaration through a face, or resolve it with system.to_positional('params', params)"
    )


class LinearGaussianTransition(eqx.Module):
    """``x' = F_j x + G_j u + b_j + eps_j``: linear map(s) with conditioned innovations.

    The mode index ``j`` runs over the state factor's discrete parents (see ``innovations``);
    ``F`` and ``G`` are shared across modes or given per mode (the switching linear dynamical
    system pattern).
    """

    #: The state map: ``(d, d)``, shared across modes, or ``(*parent_shape, d, d)`` matching
    #: ``innovations.parent_shape``, one map per mode.
    F: Array
    #: A :class:`ConditionalGaussian` whose parent axes are the state factor's discrete parents in
    #: declared (``B_deps``) order: its discrete state parents and any discrete control node, each
    #: a mode index ``j``. A state factor with no discrete parent declares the trivial parent shape
    #: ``(1,)`` (always index 0). Its location is the offset ``b_j``.
    innovations: ConditionalGaussian
    #: The control input map of the state factor's continuous control nodes, ``u`` their values
    #: concatenated in declared order: ``(d, du)`` shared, or ``(*parent_shape, d, du)`` per mode
    #: on the rule ``F`` follows; ``None`` when the state factor reads no continuous node.
    G: Array | None = None
    #: A conjugate prior over the innovations' covariance ``Q``
    #: (:class:`~hybrid_aif.distributions.InverseGamma`, per-dimension variances, or
    #: :class:`~hybrid_aif.distributions.InverseWishart`, a full covariance), shared across the
    #: modes or with leading axes ``innovations.parent_shape`` (one prior per mode); or
    #: ``Parameter(name)``, reading a noise held in ``SystemParams.parameters``. When set, every
    #: consumer reads the innovations' Cholesky factor lowered from the prior's expected-precision
    #: point (:func:`transition_point`) and the declared ``innovations.chol`` is ignored (only its
    #: shape is a declaration); ``learn`` updates the prior from the pairwise residual scatter,
    #: under the rate key ``transition:<factor>:noise``. ``None``: ``Q`` is declared, not learned.
    #: Design: process-noise-learning.md D1, D4.
    noise_prior: "InverseGamma | InverseWishart | Parameter | None" = None

    def __check_init__(self) -> None:
        _check_noise_prior("LinearGaussianTransition", self.noise_prior, self.innovations)

    @property
    def dim(self) -> int:
        return self.F.shape[-1]

    @property
    def learnable_noise(self) -> bool:
        """Capability, per-field: the process noise is a posterior (``noise_prior`` holds one)."""
        return isinstance(self.noise_prior, InverseGamma | InverseWishart)


#: The moment propagations a :class:`NonlinearGaussianTransition` declares.
PROPAGATIONS = ("first_order", "unscented")
#: The values a :class:`NonlinearGaussianTransition`'s ``noise_scale`` takes besides ``None``
#: (``f`` returns the mean alone): ``"per_coordinate"``, ``f`` returns the pair ``(mean,
#: scale)`` and the innovations' Cholesky factor ``chol`` is taken as ``diag(scale) chol``.
NOISE_SCALES = ("per_coordinate",)


class NonlinearGaussianTransition(eqx.Module):
    """``x' = f(x, controls) + eps``: a user-written mean map with conditioned innovations.

    Under ``noise_scale="per_coordinate"`` the map also scales the innovations: ``f`` returns
    ``(mean, scale)`` and ``eps ~ N(loc_j, diag(s) chol_j chol_jᵀ diag(s))``, ``s`` the scale at
    the previous state and the controls, so the process noise depends on both.

    A state factor declaring this transition is Gaussian-carried, or mixture-carried when its
    prior is a :class:`~hybrid_aif.distributions.MixtureGaussian` or when it has discrete state
    parents. Discrete state parents in ``B_deps`` switch it, as they switch a
    :class:`LinearGaussianTransition`: the mode reaches ``f`` as an integer level in the
    parent's declared place among ``controls``, and the innovations take one axis for it, so
    ``x' = f(x, (j, ...)) + eps_j``. One ``f`` reads every mode (a per-mode table inside ``f``,
    indexed by the level, writes one map per mode). The predicted next state is a mode-aligned
    mixture: each component is pushed through ``f`` once per joint mode, weighted by the mode's
    belief, and reduced as the linear switching path reduces.
    Design: docs/design/nonlinear-transitions.md N1–N5, the amendment of 2026-11-13, and N4's
    amendment of 2026-11-15 (switching).
    """

    #: An :class:`equinox.Module` called as ``f(x, controls)``: ``x`` the state factor's ``(d,)``
    #: state, ``controls`` a tuple with one entry per parent in declared (``B_deps``) order: a
    #: discrete state parent's integer mode level, a discrete control node's integer level, a
    #: continuous node's float vector (a factor with no state parent has control entries only,
    #: as before switching was built); it returns the ``(d,)`` mean of the next state, or under
    #: ``noise_scale="per_coordinate"`` the tuple ``(mean, scale)`` of two ``(d,)`` float
    #: arrays. A level may be traced (the propagation maps over the modes), so ``f`` indexes by
    #: it with array operations (``table[level]``, ``jnp.take``), not Python control flow. Its
    #: numeric fields are pytree leaves and its code is static structure. ``f`` is not learned.
    f: eqx.Module
    #: A :class:`ConditionalGaussian` added to ``f``'s output, ``eps ~ N(loc_j, chol_j chol_jᵀ)``,
    #: whose parent axes are the state factor's discrete parents in declared (``B_deps``) order,
    #: its discrete state parents (the modes) and its discrete control nodes, the
    #: :class:`LinearGaussianTransition` convention (the trivial ``(1,)`` for none); its
    #: Cholesky factor's diagonal must be positive for beliefs to be propagated. Under
    #: ``noise_scale="per_coordinate"`` its Cholesky factor is ``diag(scale) chol_j`` and its
    #: offset ``loc_j`` is not scaled.
    innovations: ConditionalGaussian
    #: How a belief is pushed through ``f`` (static): ``"first_order"`` linearises ``f`` at the
    #: belief's mean, ``"unscented"`` pushes ``2n + 1`` sigma points through it (``n`` the state's
    #: dimension plus that of any control carried as a belief). Both are exact for ``f`` affine in
    #: its continuous inputs and a scale that does not vary.
    propagation: str = eqx.field(static=True, default="first_order")
    #: What ``f`` returns besides the mean (static): ``None``, nothing (the mean alone);
    #: ``"per_coordinate"``, the pair ``(mean, scale)``, ``scale`` a ``(d,)`` vector, the
    #: innovations' Cholesky factor taken as ``diag(scale) chol``. Validation checks the map's
    #: return against it; no code path reads the return's structure instead.
    #:
    #: The scale must be strictly positive and finite (a softplus or an exponential, say); it is
    #: not checked under ``jit``. A zero on a coordinate whose mean has no spread makes the
    #: predicted covariance singular, and its Cholesky factor NaN; a negative entry flips the
    #: sign of that coordinate's correlations. For the gradient planner and any gradient through
    #: a belief, the scale must also be differentiable wherever it is evaluated, the unscented
    #: centre point included (its zero weight does not cancel a NaN). A scale such as
    #: ``sqrt(1 - rho(d)**2)`` with ``rho(0) = 1`` has an infinite or NaN derivative at ``d = 0``
    #: (zero displacement, a vehicle at rest), and the gradient planner then returns a NaN plan
    #: from a start at rest; guard it as ``sqrt(1 - rho**2 + eps)``, or with a double ``where``
    #: that keeps the root's argument away from zero on the branch not taken.
    noise_scale: Literal["per_coordinate"] | None = eqx.field(static=True, default=None)
    #: A conjugate prior over the innovations' covariance, or ``Parameter(name)``: the
    #: :class:`LinearGaussianTransition` field of the same name. Under
    #: ``noise_scale="per_coordinate"`` it is a prior over the unscaled ``chol_j cholᵀ_j``.
    #: Learned at first order (``f`` linearised at the previous mean); the unscented rule is
    #: refused by ``learn``. Design: process-noise-learning.md D1, D2.
    noise_prior: "InverseGamma | InverseWishart | Parameter | None" = None

    def __check_init__(self) -> None:
        _check_noise_prior("NonlinearGaussianTransition", self.noise_prior, self.innovations)
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
        if self.noise_scale is not None and self.noise_scale not in NOISE_SCALES:
            raise ValueError(
                f"NonlinearGaussianTransition: noise_scale={self.noise_scale!r}; pass None (f "
                f"returns the mean alone) or one of {list(NOISE_SCALES)} (f returns the pair "
                "(mean, scale))"
            )

    @property
    def dim(self) -> int:
        return self.innovations.dim

    @property
    def learnable_noise(self) -> bool:
        """Capability, per-field: the process noise is a posterior (``noise_prior`` holds one)."""
        return isinstance(self.noise_prior, InverseGamma | InverseWishart)

    @property
    def _scaled(self) -> bool:
        """Whether ``f`` scales the innovations (``noise_scale="per_coordinate"``)."""
        return self.noise_scale == "per_coordinate"

    def _mean_and_scale(self, x: Array, controls: tuple) -> tuple[Array, Array | None]:
        """``f`` at ``x`` and ``controls`` as ``(mean, scale)``, ``scale`` ``None`` under
        ``noise_scale=None``. The return's structure is checked against the declaration when
        traced, so a map that returns a pair undeclared (or a mean alone when a pair is
        declared) raises instead of being misread."""
        out = self.f(x, controls)
        pair = isinstance(out, tuple) and len(out) == 2
        if self._scaled and not pair:
            raise TypeError(
                "NonlinearGaussianTransition: noise_scale='per_coordinate' declares that f "
                f"returns the pair (mean, scale), and it returned a {type(out).__name__}"
            )
        if not self._scaled and isinstance(out, tuple | list):
            raise TypeError(
                f"NonlinearGaussianTransition: f returned a {type(out).__name__} of "
                f"{len(out)} entries, and noise_scale=None declares that it returns the mean "
                "alone; declare noise_scale='per_coordinate' for a map that returns "
                "(mean, scale)"
            )
        return (out[0], out[1]) if self._scaled else (out, None)

    def _mean(self, x: Array, controls: tuple) -> Array:
        """The next state's mean, ``f``'s first output under either ``noise_scale``."""
        return self._mean_and_scale(x, controls)[0]


GaussianTransition = LinearGaussianTransition | NonlinearGaussianTransition


def transition_point(decl):
    """The point form of a Gaussian transition: with a ``noise_prior``, the innovations' Cholesky
    factor lowered from the prior's expected-precision point and the prior dropped; any other
    declaration as is. Every consumer that reads the process noise reads it through this
    (prediction, planning, ``as_env``, the closed loop). Design: process-noise-learning.md D1."""
    prior = getattr(decl, "noise_prior", None)
    if prior is None or not isinstance(decl, GaussianTransition):
        return decl
    if isinstance(prior, Parameter):
        raise _unresolved(f"{type(decl).__name__}.noise_prior", prior.name)
    return dataclasses.replace(
        decl, innovations=_lowered_innovations(decl.innovations, prior), noise_prior=None
    )


def map_noise_modes(prior, fn):
    """``prior`` with ``fn(array, core)`` applied to each of its fields, ``core`` the field's
    trailing non-mode axes (an inverse-Gamma's ``alpha``/``beta`` 1, an inverse-Wishart's
    ``Psi`` 2 and ``nu`` 0); a shared prior (no mode axes), ``None`` or a ``Parameter`` as is.
    How a rewrite of a transition's mode axes (slicing, permuting) carries its noise prior."""
    if not isinstance(prior, InverseGamma | InverseWishart) or not _noise_batch(prior):
        return prior
    cores = (
        {"alpha": 1, "beta": 1, "alpha0": 1, "beta0": 1}
        if isinstance(prior, InverseGamma)
        else {"Psi": 2, "Psi0": 2, "nu": 0, "nu0": 0}
    )
    return dataclasses.replace(
        prior,
        **{
            name: None if getattr(prior, name) is None else fn(getattr(prior, name), core)
            for name, core in cores.items()
        },
    )


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

    A map over a modality's categorical parents (an action node counts) carries one leading
    level axis per parent, in ``A_deps`` order: ``values`` is ``(*levels, *grid_shape, d_obs,
    d_obs)``, the leading axes being whatever ``values`` has beyond the grid and the factor
    (:attr:`n_levels`). A level is selected first (:meth:`select`) and the grid read at the mean
    after, so a levelled map at one level is the plain map declared for that level. Called
    directly it returns every level's factor, ``(*levels, *batch, d_obs, d_obs)``.
    """

    #: The Cholesky factor at every grid node, ``(*levels, *grid_shape, d_obs, d_obs)``: lower
    #: triangular with a positive diagonal (the leading level axes only on a levelled map).
    values: Array
    #: The grid's lower extent per axis, one number each.
    low: Array
    #: The grid's upper extent per axis, one number each.
    high: Array
    #: The state coordinates the grid spans, in order (static).
    axes: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, values: Array, low, high, axes: tuple[int, ...]):
        self.values = jnp.asarray(values)
        self.axes = tuple(int(a) for a in axes)
        k = len(self.axes)
        self.low = jnp.broadcast_to(jnp.asarray(low, self.values.dtype), (k,))
        self.high = jnp.broadcast_to(jnp.asarray(high, self.values.dtype), (k,))

    @property
    def n_levels(self) -> int:
        """How many leading level axes ``values`` carries (zero for a plain map)."""
        return self.values.ndim - len(self.axes) - 2

    @property
    def levels(self) -> tuple[int, ...]:
        """The sizes of the leading level axes, in ``A_deps`` order of the parents."""
        return tuple(self.values.shape[: self.n_levels])

    def __check_init__(self) -> None:
        k = len(self.axes)
        if self.values.ndim < k + 2:
            raise ValueError(
                f"GridNoiseMap: values has {self.values.ndim} axes, expected at least {k + 2}: "
                f"one grid axis per entry of axes={self.axes} and the (d_obs, d_obs) factor, "
                "after any leading level axes"
            )
        if self.values.shape[-1] != self.values.shape[-2]:
            raise ValueError(
                f"GridNoiseMap: values' trailing shape {self.values.shape[-2:]} is not square"
            )
        grid = self.values.shape[self.n_levels : self.n_levels + k]
        if any(n < 2 for n in grid):
            raise ValueError(f"GridNoiseMap: every grid axis needs at least two nodes, got {grid}")
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

    def select(self, levels: tuple[Array, ...]) -> "GridNoiseMap":
        """The plain map at one level of each categorical parent: the leading level axes
        indexed at ``levels`` (integer scalars, traced or not), in order."""
        if len(levels) != self.n_levels:
            raise ValueError(
                f"GridNoiseMap: {len(levels)} level(s) given for a map with {self.n_levels} "
                "level axis/es"
            )
        values = self.values
        for level in levels:
            values = jnp.take(values, level, axis=0)
        return eqx.tree_at(lambda m: m.values, self, values)  # no re-validation: may be traced

    def __call__(self, loc: Array) -> Array:
        """``R_chol`` at ``loc`` (``(..., d)``): ``(..., d_obs, d_obs)``, or ``(*levels, ...,
        d_obs, d_obs)`` for a levelled map."""
        loc = jnp.asarray(loc)
        k, nl = len(self.axes), self.n_levels
        sizes = jnp.asarray(self.values.shape[nl : nl + k], loc.dtype)
        coords = jnp.stack([loc[..., a] for a in self.axes], axis=-1)
        coords = (coords - self.low) / (self.high - self.low) * (sizes - 1.0)
        coords = jnp.clip(coords, 0.0, sizes - 1.0)
        base = jnp.minimum(jnp.floor(coords), sizes - 2.0)
        frac = coords - base
        base = base.astype(jnp.int32)
        out = jnp.zeros(
            (*self.values.shape[:nl], *loc.shape[:-1], *self.values.shape[-2:]), self.values.dtype
        )
        whole = (slice(None),) * nl
        for corner in itertools.product((0, 1), repeat=k):
            index = tuple(base[..., j] + corner[j] for j in range(k))
            weight = jnp.ones(loc.shape[:-1], self.values.dtype)
            for j in range(k):
                weight = weight * (frac[..., j] if corner[j] else 1.0 - frac[..., j])
            out = out + weight[..., None, None] * self.values[whole + index]
        return out


class LinearGaussianObservation(eqx.Module):
    """``y = H x + c + eps``: a linear-Gaussian modality over one Gaussian state factor.

    Construct with ``H=``/``c=`` to have ``head`` assembled. ``H``, ``c`` and the noise are each
    either shared (``(d_obs, d)`` / ``(d_obs,)`` / ``(d_obs, d_obs)``) or per-mode, with leading
    parent axes ``parent_shape`` (the categorical parents named after the observed state factor
    in ``A_deps``), broadcast like the per-mode ``F_j`` of :class:`LinearGaussianTransition`.
    Every inference consumer reads the :attr:`R_chol` property, which lowers a distributional
    noise by its expected precision.
    """

    #: The augmented ``[H | c]``: ``(d_obs, d + 1)``, or ``(*parent_shape, d_obs, d + 1)`` per
    #: mode; a point value or a ``MatrixNormal`` posterior over it (never separate posteriors over
    #: ``H`` and ``c``, which are coupled).
    head: Array | MatrixNormal
    #: A point ``R_chol`` (lower triangular ``(d_obs, d_obs)``, or ``(*parent_shape, d_obs,
    #: d_obs)`` per mode), or the conjugate posterior over ``R``
    #: (:class:`~hybrid_aif.distributions.InverseGamma`,
    #: :class:`~hybrid_aif.distributions.InverseWishart`), which makes it learnable and samplable;
    #: or ``Parameter(name)``, reading a noise held in ``SystemParams.parameters`` (one noise
    #: several modalities or transitions read; design: process-noise-learning.md D4).
    noise: "Array | InverseGamma | InverseWishart | Parameter"

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
        if isinstance(self.noise, Parameter):
            raise _unresolved("LinearGaussianObservation.noise", self.noise.name)
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

    @property
    def noise_levels(self) -> int:
        """How many leading level axes the noise map carries, one per categorical parent (zero
        for a plain map, a point or a posterior)."""
        return self.noise.n_levels if self.noise_map and hasattr(self.noise, "n_levels") else 0

    def R_chol_at(self, loc: Array) -> Array:
        """The noise as an ``R_chol`` array at a belief's mean ``loc``: a point or conjugate
        noise is the same everywhere; a noise map is evaluated there, the extended filter's
        convention (design: planning-with-observations.md D3). A levelled map returns every
        level's factor, leading ``(*levels, ...)`` like a per-mode ``R_chol``."""
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
    modality declared this way names only categorical state factors in ``A_deps``; its message to
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

    :meth:`gru` builds the RSSM carry case. A recurrent map takes its carry as an argument and
    returns the next one; nothing is stashed on the module. The gradient statistic arrives from
    whatever consumes the node's output, and :meth:`absorb` applies it through the shared
    gradient path.

    Design: docs/design/system.md (Factors)
    """

    #: Any pure callable pytree: an ``equinox.Module`` whose array leaves are the node's
    #: parameters, or a bare function with none.
    map: object
    #: The per-node message treatment, a static string from :data:`_RULES`; only ``"carry"``
    #: (computed forward, never inferred) is built.
    rule: str = eqx.field(static=True, default="carry")
    #: The trainable marking: ``True`` keeps the parameters fixed.
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
        dtype: object = None,
    ) -> "Deterministic":
        """An RSSM's carry node: a :class:`~hybrid_aif.distributions.neural.GRUCell`
        over ``(deter, *inputs)``, with ``input_dims`` the widths of the non-carry
        inputs — for an RSSM ``(num_vars·num_classes, action_dim)``, the previous
        flattened one-hot ``stoch`` and the action. ``init`` and ``dtype`` are the cell's."""
        return cls(
            map=GRUCell.from_shapes(
                deter_dim,
                input_dims,
                key=key,
                hidden=hidden,
                activation=activation,
                norm=norm,
                init=init,
                dtype=dtype,
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
    """ "This state factor's initial belief is its own transition, at the episode start."

    An ``initial_state_prior`` declaration with no parameters of its own: the state factor's ``t=0``
    value is its transition evaluated at its within-step parents' initialized values (an RSSM's
    initial code, read off the prior at the initial carry). ``reduction`` names how the transition's
    distribution is reduced to the carried value; only ``"mode"`` is built.
    """

    #: How the transition's distribution is reduced to the carried value; only ``"mode"``.
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

    ``variable`` is a state factor name (the amortized posterior for that state factor) or any other
    name (an auxiliary q-side variable, such as an observation embedding). ``parents`` is explicit,
    in the module's parent order, each entry one of

    - ``"<factor>"``          a state factor's value at this tick (the predicted one, for a variable
                              inferred after it),
    - ``"obs:<modality>"``    this tick's observation of that modality,
    - ``"recognition:<name>"`` an earlier recognition variable's value,
    - ``"external:<name>"``   a vector this System does not declare, supplied per call as data and
                              declared (name, width) in ``System.external_inputs``.

    Evaluation is in declaration order, so a parent must be declared earlier.

    Design: docs/design/amortization.md A11
    """

    #: The inferred variable, the q-side factor's child: a state factor name (the amortized
    #: posterior for that state factor) or any other name (an auxiliary q-side variable).
    variable: str
    #: The conditioning set, in the module's parent order (grammar above).
    parents: tuple[str, ...]


#: Declarations that learn by **gradient** rather than by conjugacy: the
#: neural families, the deterministic node, and the parametrized constant. Every
#: member carries ``learnable``/``init_opt_state``/``absorb`` and is frozen by a
#: static ``frozen`` flag, which is what lets ``System`` treat them uniformly (one
#: ``LearnState`` entry per non-frozen slot, no optimizer state for a frozen one).
GradientDeclaration = (
    MLPConditional | MLPPredictor | MLPCategorical | PredictorHead | Deterministic | LearnedVector
)
