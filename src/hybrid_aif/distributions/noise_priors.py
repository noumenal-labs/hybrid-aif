"""Conjugate priors over LG observation noise: inverse-Gamma and inverse-Wishart.

They sit in ``LinearGaussianObservation.noise`` in place of a point ``R_chol``; leading axes are
parent (mode) axes. Conditioning consumes a precision, so ``point()`` returns the Cholesky
factor of the inverse expected precision ``E[R⁻¹]⁻¹`` (``β/α`` per dimension; ``Ψ/ν``), not the
mean covariance (``mean()``, a diagnostic). ``expected_log_params`` and ``novelty`` raise:
nothing consumes ``E[log R]``, and continuous parameter novelty is deferred (D6). These families
have no canonical decay anchor (the improper limits are the floors decay must not cross), so
``decay < 1`` requires explicit anchors and raises otherwise.

Design: docs/design/parameter-learning.md D7, D8
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.linalg import cho_solve, solve_triangular

from hybrid_aif.distributions._deferrals import deferred_novelty
from hybrid_aif.distributions.gaussian import cholesky_of_inverse


def _anchored(x: Array, anchor: Array | None, stats: Array, lr, decay, what: str) -> Array:
    """Decayed-count kernel with a required anchor.

    A static ``decay == 1.0`` takes the pure-conjugate branch ``x + lr·stats`` (bitwise); any other
    decay, traced included, needs the anchor.
    """
    if isinstance(decay, int | float) and float(decay) == 1.0:
        return x + lr * stats
    if anchor is None:
        raise ValueError(
            f"decay < 1 on an LG noise prior requires an explicit {what} anchor: "
            "the improper limit is exactly the floor decay must not cross (D7)"
        )
    return decay * x + (1.0 - decay) * anchor + lr * stats


def _check_positive(name: str, value: Array | None) -> None:
    if value is None or isinstance(value, jax.core.Tracer):
        return
    v = np.asarray(value)
    if not np.all(np.isfinite(v) & (v > 0.0)):
        raise ValueError(f"{name} must be finite and strictly positive")


def _check_nu_floor(nu: Array, d: int) -> None:
    """Raise unless a concrete ``nu`` exceeds the inverse-Wishart propriety floor ``d − 1``."""
    if isinstance(nu, jax.core.Tracer):
        return
    v = np.asarray(nu)
    if not np.all(np.isfinite(v) & (v > d - 1.0)):
        raise ValueError(f"nu must exceed the propriety floor d - 1 = {d - 1}")


class InverseGamma(eqx.Module):
    """Independent inverse-Gamma posteriors over per-dimension noise variances.

    ``alpha``/``beta`` are ``(*parent_shape, d_obs)``: one (shape, rate) pair per observation
    dimension per parent configuration. Isotropic noise is the ``d_obs = 1`` case.
    """

    alpha: Array  # (..., d_obs) shape parameters
    beta: Array  # (..., d_obs) rate parameters
    alpha0: Array | None = None  # decay anchors (D7); required iff decay < 1
    beta0: Array | None = None

    def __check_init__(self) -> None:
        if self.alpha.shape != self.beta.shape:
            raise ValueError(f"alpha shape {self.alpha.shape} != beta shape {self.beta.shape}")
        for name, v in (("alpha", self.alpha), ("beta", self.beta)):
            _check_positive(name, v)
        for name, v in (("alpha0", self.alpha0), ("beta0", self.beta0)):
            if v is not None and v.shape != self.alpha.shape:
                raise ValueError(f"{name} shape {v.shape} != parameter shape {self.alpha.shape}")
            _check_positive(name, v)

    @classmethod
    def from_R_chol(cls, R_chol: Array, strength: float = 1.0, anchor: bool = False):
        """Prior centred on a known noise level: ``point()`` reproduces ``R_chol``
        exactly, and ``strength`` counts pseudo-observations (``α = strength/2``).
        ``anchor=True`` stores the initial parameters as the decay anchors."""
        sigma2 = jnp.diagonal(R_chol @ jnp.swapaxes(R_chol, -1, -2), axis1=-2, axis2=-1)
        alpha = jnp.full_like(sigma2, strength / 2.0)
        beta = alpha * sigma2
        return cls(
            alpha=alpha,
            beta=beta,
            alpha0=alpha if anchor else None,
            beta0=beta if anchor else None,
        )

    @property
    def obs_dim(self) -> int:
        return self.alpha.shape[-1]

    def expected_precision(self) -> Array:
        """``E[1/σ²] = α/β`` per dimension — what mean-field conditioning consumes."""
        return self.alpha / self.beta

    def point(self) -> Array:
        """``R_chol`` under the expected-precision substitution: ``diag(sqrt(β/α))``.

        Not the mean covariance (:meth:`mean`): conditioning consumes a precision.
        """
        return _diag_chol(jnp.sqrt(self.beta / self.alpha))

    def mean(self) -> Array:
        """``E[σ²] = β/(α−1)`` per dimension (α > 1) — diagnostic, not consumed."""
        return self.beta / (self.alpha - 1.0)

    def stats(self, scatter: Array, count: Array) -> tuple[Array, Array]:
        """Package one tick's statistic: the responsibility-weighted residual scatter
        diagonal ``(..., d_obs)`` and the responsibility mass ``(...,)`` (the degrees-of-freedom
        increment).
        """
        return scatter, count

    def absorb(self, stats: tuple[Array, Array], lr=1.0, decay=1.0) -> "InverseGamma":
        scatter, count = stats
        return InverseGamma(
            alpha=_anchored(self.alpha, self.alpha0, count[..., None] / 2.0, lr, decay, "alpha0"),
            beta=_anchored(self.beta, self.beta0, scatter / 2.0, lr, decay, "beta0"),
            alpha0=self.alpha0,
            beta0=self.beta0,
        )

    def sample(self, key: Array) -> Array:
        """One noise draw as an ``R_chol`` array: ``σ² ~ InvGamma(α, β)`` per dim."""
        return _diag_chol(jnp.sqrt(self.beta / jax.random.gamma(key, self.alpha)))

    def expected_log_params(self) -> Array:
        raise NotImplementedError(
            "LG inference consumes the expected precision (point()), not an E[log R] "
            "table; no consumer exists (design: parameter-learning.md D8)"
        )

    def novelty(self) -> Array:
        raise deferred_novelty("an LG noise posterior")


class InverseWishart(eqx.Module):
    """Inverse-Wishart posterior over a full noise covariance.

    ``Psi`` is ``(*parent_shape, d, d)`` (positive definite), ``nu`` is ``(*parent_shape,)`` with
    the propriety floor ``ν > d − 1`` (the mean exists from ``ν > d + 1``).
    """

    Psi: Array  # (..., d, d) scale matrices
    nu: Array  # (...,) degrees of freedom
    Psi0: Array | None = None  # decay anchors (D7); required iff decay < 1
    nu0: Array | None = None

    def __check_init__(self) -> None:
        d = self.Psi.shape[-1]
        if self.Psi.shape[-2] != d:
            raise ValueError(f"Psi must be square, got {self.Psi.shape}")
        if self.nu.shape != self.Psi.shape[:-2]:
            raise ValueError(f"nu shape {self.nu.shape} != Psi batch shape {self.Psi.shape[:-2]}")
        _check_nu_floor(self.nu, d)
        for name, v in (("Psi0", self.Psi0), ("nu0", self.nu0)):
            if v is not None:
                _check_positive(name if name == "nu0" else f"{name} diagonal", v)

    @property
    def obs_dim(self) -> int:
        return self.Psi.shape[-1]

    def expected_precision(self) -> Array:
        """``E[R⁻¹] = ν Ψ⁻¹`` — what mean-field conditioning consumes."""
        eye = jnp.broadcast_to(jnp.eye(self.obs_dim, dtype=self.Psi.dtype), self.Psi.shape)
        return self.nu[..., None, None] * cho_solve((jnp.linalg.cholesky(self.Psi), True), eye)

    def point(self) -> Array:
        """``R_chol`` under the expected-precision substitution: ``chol(Ψ/ν)``."""
        return jnp.linalg.cholesky(self.Psi / self.nu[..., None, None])

    def mean(self) -> Array:
        """``E[R] = Ψ/(ν − d − 1)`` (ν > d+1) — diagnostic, not consumed."""
        return self.Psi / (self.nu - self.obs_dim - 1.0)[..., None, None]

    def stats(self, scatter: Array, count: Array) -> tuple[Array, Array]:
        """Package one tick's statistic: full residual scatter ``(..., d, d)``
        and responsibility mass ``(...,)``."""
        return scatter, count

    def absorb(self, stats: tuple[Array, Array], lr=1.0, decay=1.0) -> "InverseWishart":
        scatter, count = stats
        return InverseWishart(
            Psi=_anchored(self.Psi, self.Psi0, scatter, lr, decay, "Psi0"),
            nu=_anchored(self.nu, self.nu0, count, lr, decay, "nu0"),
            Psi0=self.Psi0,
            nu0=self.nu0,
        )

    def sample(self, key: Array) -> Array:
        """One noise draw as an ``R_chol`` array, via the Bartlett decomposition
        of ``W ~ Wishart(ν, Ψ⁻¹)`` and ``R = W⁻¹``."""
        d = self.obs_dim
        batch = self.Psi.shape[:-2]
        kg, kn = jax.random.split(key)
        i = jnp.arange(d, dtype=self.Psi.dtype)
        # diag_j ~ sqrt(chi2(nu - j)) = sqrt(2 * Gamma((nu - j)/2))
        df = (self.nu[..., None] - i) / 2.0
        diag = jnp.sqrt(2.0 * jax.random.gamma(kg, df))
        lower = jnp.tril(jax.random.normal(kn, (*batch, d, d)), k=-1)
        a = lower + _diag_embed(diag)
        # W = (L A)(L A)^T with L = chol(Psi^{-1}) and L A lower-triangular, so
        # R = W^{-1} = M^T M with M = (L A)^{-1} from one triangular solve.
        la = cholesky_of_inverse(self.Psi) @ a
        eye = jnp.broadcast_to(jnp.eye(d, dtype=la.dtype), la.shape)
        m = solve_triangular(la, eye, lower=True)
        return jnp.linalg.cholesky(jnp.swapaxes(m, -1, -2) @ m)

    def expected_log_params(self) -> Array:
        raise NotImplementedError(
            "LG inference consumes the expected precision (point()), not an E[log R] "
            "table; no consumer exists (design: parameter-learning.md D8)"
        )

    def novelty(self) -> Array:
        raise deferred_novelty("an LG noise posterior")


def _diag_chol(diag: Array) -> Array:
    """Embed per-dimension scales ``(..., d)`` as diagonal ``(..., d, d)`` factors."""
    return _diag_embed(diag)


def _diag_embed(diag: Array) -> Array:
    return diag[..., None] * jnp.eye(diag.shape[-1], dtype=diag.dtype)
