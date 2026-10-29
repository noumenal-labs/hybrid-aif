"""Linear-Gaussian conditioning and prediction kernels over :class:`Gaussian` objects.

Cholesky-based throughout, with no explicit inverse: covariances are sums of Gram factors (SPD
by construction) and re-factored. Everything broadcasts over leading batch axes.
"""

import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import cho_solve, solve_triangular

from hybrid_aif.distributions.gaussian import Gaussian

_LOG_2PI = jnp.log(2.0 * jnp.pi)


def _tr(a: Array) -> Array:
    return jnp.swapaxes(a, -1, -2)


def predict(g: Gaussian, F: Array, b: Array, Q_chol: Array) -> Gaussian:
    """Push ``g`` through ``x' = F x + b + eps``, ``eps ~ N(0, Q_chol Q_chol^T)``.

    ``F`` (..., dim, dim), ``b`` (..., dim), ``Q_chol`` (..., dim, r): any factor with ``Q =
    Q_chol Q_cholᵀ``, read only through that product; batch axes of ``g`` and the system matrices
    broadcast against each other.
    """
    loc = (F @ g.loc[..., None])[..., 0] + b
    FL = F @ g.chol
    cov = FL @ _tr(FL) + Q_chol @ _tr(Q_chol)
    chol = jnp.linalg.cholesky(cov)
    batch = jnp.broadcast_shapes(loc.shape[:-1], chol.shape[:-2])
    dim = loc.shape[-1]
    return Gaussian(
        loc=jnp.broadcast_to(loc, batch + (dim,)),
        chol=jnp.broadcast_to(chol, batch + (dim, dim)),
    )


def _gain_and_factor(g: Gaussian, H: Array, R_chol: Array) -> tuple[Array, Array, Array, Array]:
    """The innovation Cholesky factor, the gain ``K = Σ Hᵀ S⁻¹``, the posterior covariance's
    Cholesky factor (Joseph form from Gram factors) and ``H L``, for a reading ``y = H x + c +
    eps`` with ``eps ~ N(0, R_chol R_cholᵀ)``; none depends on ``y`` or ``c``."""
    HL = H @ g.chol
    S = HL @ _tr(HL) + R_chol @ _tr(R_chol)
    S_chol = jnp.linalg.cholesky(S)
    # Gain K = Sigma H^T S^{-1}, via S X = (Sigma H^T)^T then K = X^T.
    PHt = g.chol @ _tr(HL)
    gain = _tr(cho_solve((S_chol, True), _tr(PHt)))
    AL = g.chol - gain @ HL  # (I - K H) L
    KR = gain @ R_chol
    chol = jnp.linalg.cholesky(AL @ _tr(AL) + KR @ _tr(KR))
    return S_chol, gain, chol, HL


def condition(g: Gaussian, H: Array, c: Array, R_chol: Array, y: Array) -> tuple[Gaussian, Array]:
    """Bayesian update of ``g`` on ``y = H x + c + eps``, ``eps ~ N(0, R_chol R_chol^T)``.

    Returns ``(posterior, log_marginal)``, ``log_marginal = log p(y)`` from the innovation
    Cholesky. The covariance update is the Joseph form from Gram factors, so the posterior stays
    SPD for near-singular gains.
    """
    dim_y = y.shape[-1]
    v = y - ((H @ g.loc[..., None])[..., 0] + c)
    S_chol, gain, chol, _ = _gain_and_factor(g, H, R_chol)
    loc = g.loc + (gain @ v[..., None])[..., 0]
    z = solve_triangular(S_chol, v[..., None], lower=True)[..., 0]
    half_log_det = jnp.sum(jnp.log(jnp.diagonal(S_chol, axis1=-2, axis2=-1)), axis=-1)
    log_marginal = -0.5 * (dim_y * _LOG_2PI + jnp.sum(z**2, axis=-1)) - half_log_det
    batch = jnp.broadcast_shapes(loc.shape[:-1], chol.shape[:-2])
    dim = loc.shape[-1]
    posterior = Gaussian(
        loc=jnp.broadcast_to(loc, batch + (dim,)),
        chol=jnp.broadcast_to(chol, batch + (dim, dim)),
    )
    return posterior, jnp.broadcast_to(log_marginal, batch)


def condition_on_expected(g: Gaussian, H: Array, R_chol: Array) -> Gaussian:
    """``g`` conditioned on the reading ``H x + c + eps`` it is expected to give at its own mean:
    the innovation is zero, so the mean is ``g.loc`` itself and the covariance takes the same
    Joseph-form update as :func:`condition`, from ``R_chol`` (the noise at that mean)."""
    _, _, chol, _ = _gain_and_factor(g, H, R_chol)
    batch = jnp.broadcast_shapes(g.loc.shape[:-1], chol.shape[:-2])
    dim = g.loc.shape[-1]
    return Gaussian(
        loc=jnp.broadcast_to(g.loc, batch + (dim,)),
        chol=jnp.broadcast_to(chol, batch + (dim, dim)),
    )
