"""Full-covariance multivariate Gaussian.

Stored parameterization: ``mean`` and lower-triangular Cholesky factor ``chol``
(numerically safest; log-dets and solves read off the factor). Event shape:
``(dim,)``. Natural view: ``eta1 = Σ⁻¹ μ``, ``eta2 = -0.5 Σ⁻¹``.
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import cho_solve, solve_triangular

from hybrid_aif.distributions.base import ExponentialFamily
from hybrid_aif.interval import Interval

_LOG_2PI = jnp.log(2.0 * jnp.pi)


class Gaussian(ExponentialFamily):
    loc: Array  # (..., dim)
    chol: Array  # (..., dim, dim), lower-triangular

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.loc.shape[:-1]

    @property
    def dim(self) -> int:
        return self.loc.shape[-1]

    @property
    def cov(self) -> Array:
        return self.chol @ jnp.swapaxes(self.chol, -1, -2)

    @classmethod
    def from_cov(cls, loc: Array, cov: Array) -> "Gaussian":
        return cls(loc=loc, chol=jnp.linalg.cholesky(cov))

    def _half_log_det(self) -> Array:
        return jnp.sum(jnp.log(jnp.diagonal(self.chol, axis1=-2, axis2=-1)), axis=-1)

    def log_prob(self, x: Array) -> Array:
        dev = x - self.loc
        z = solve_triangular(self.chol, dev[..., None], lower=True)[..., 0]
        maha = jnp.sum(z**2, axis=-1)
        return -0.5 * (self.dim * _LOG_2PI + maha) - self._half_log_det()

    def sample(self, key: Array, sample_shape: tuple[int, ...] = ()) -> Array:
        eps = jax.random.normal(key, sample_shape + self.batch_shape + (self.dim,))
        return self.loc + (self.chol @ eps[..., None])[..., 0]

    def mean(self) -> Array:
        return self.loc

    def entropy(self) -> Interval:
        return Interval.exact(0.5 * self.dim * (1.0 + _LOG_2PI) + self._half_log_det())

    def kl_to(self, other) -> Interval:
        """Closed-form Gaussian KL, computed via Cholesky solves.

        A ``MixtureGaussian`` ``other`` is handled by promoting ``self`` to a 1-component mixture
        and using the mixture bounds.
        """
        from hybrid_aif.distributions.mixture import MixtureGaussian

        if isinstance(other, MixtureGaussian):
            return MixtureGaussian.from_gaussian(self).kl_to(other)
        dev = other.loc - self.loc
        z = solve_triangular(other.chol, dev[..., None], lower=True)[..., 0]
        maha = jnp.sum(z**2, axis=-1)
        m = solve_triangular(other.chol, self.chol, lower=True)
        trace = jnp.sum(m**2, axis=(-2, -1))
        log_det_ratio = 2.0 * (other._half_log_det() - self._half_log_det())
        return Interval.exact(0.5 * (trace + maha - self.dim + log_det_ratio))

    def natural(self) -> tuple[Array, ...]:
        prec = cho_solve((self.chol, True), jnp.broadcast_to(jnp.eye(self.dim), self.cov.shape))
        eta1 = (prec @ self.loc[..., None])[..., 0]
        return (eta1, -0.5 * prec)

    @classmethod
    def from_natural(cls, eta1: Array, eta2: Array) -> "Gaussian":
        chol = cholesky_of_inverse(-2.0 * eta2)
        loc = (chol @ (jnp.swapaxes(chol, -1, -2) @ eta1[..., None]))[..., 0]
        return cls(loc=loc, chol=chol)


def sigma_points(loc: Array, chol: Array, kappa: float = 1.0) -> tuple[Array, Array]:
    """Unscented-transform sigma points of ``N(loc, chol cholᵀ)``: ``2d+1`` points and weights.

    The points are ``loc`` and ``loc ± √(d+κ) · chol[:, i]`` for each column ``i``; the weights
    are ``κ/(d+κ)`` for the centre and ``1/(2(d+κ))`` for the rest. The weighted sum of ``f`` at
    the points equals ``E[f(x)]`` exactly for every polynomial ``f`` of degree at most three, for
    any ``κ > −d``; ``κ > 0`` makes every weight positive, so the estimate is a convex combination
    of the point values.

    Args:
        loc: ``(..., d)`` mean; leading axes batch.
        chol: ``(..., d, d)`` lower Cholesky factor of the covariance.
        kappa: the centre's share of the spread (``κ`` above).

    Returns:
        ``(points, weights)`` with shapes ``(..., 2d+1, d)`` and ``(2d+1,)``; point 0 is the
        centre, points ``1..d`` the ``+`` side, ``d+1..2d`` the ``−`` side.
    """
    d = loc.shape[-1]
    scaled = jnp.sqrt(d + kappa) * jnp.swapaxes(chol, -1, -2)  # (..., d, d): row i = column i
    centre = loc[..., None, :]
    points = jnp.concatenate([centre, centre + scaled, centre - scaled], axis=-2)
    weights = jnp.concatenate(
        [
            jnp.asarray([kappa / (d + kappa)], dtype=loc.dtype),
            jnp.full((2 * d,), 1.0 / (2.0 * (d + kappa)), dtype=loc.dtype),
        ]
    )
    return points, weights


def cholesky_of_inverse(a: Array) -> Array:
    """Lower Cholesky factor of ``A⁻¹`` for SPD ``A`` (leading batch axes allowed).

    One Cholesky and one triangular solve: with ``J`` the index reversal and ``J A J = L Lᵀ``,
    ``J L⁻ᵀ J`` is the lower-triangular factor of ``A⁻¹``.
    """
    rev = jnp.flip(jnp.flip(a, -1), -2)
    l_rev = jnp.linalg.cholesky(rev)
    eye = jnp.broadcast_to(jnp.eye(a.shape[-1], dtype=a.dtype), a.shape)
    l_rev_inv = solve_triangular(l_rev, eye, lower=True)
    return jnp.flip(jnp.flip(jnp.swapaxes(l_rev_inv, -1, -2), -1), -2)
