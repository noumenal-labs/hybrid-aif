"""Matrix-normal posterior over an augmented LG head ``W = [H | c]``.

One matrix-normal over the augmented head, never separate posteriors over ``H`` and ``c`` (they
are coupled through the state second moment). Two uses: an independent head
(``LinearGaussianObservation.head``, head ⊥ noise, updated on the noise-whitened regression
statistic ``learn`` supplies), and the conditional ``W | R ~ MN(M, R, V)`` inside
:class:`~hybrid_aif.factors.LinearGaussianObservationMNIW`. Leading axes are parent (mode) axes.

Design: docs/design/parameter-learning.md D8
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.distributions._deferrals import deferred_novelty
from hybrid_aif.distributions.gaussian import cholesky_of_inverse
from hybrid_aif.distributions.noise_priors import _anchored, _check_positive


class MatrixNormal(eqx.Module):
    """``W ~ MN(M, U, V)`` with shared column covariance; row structure external.

    ``M`` is ``(..., d_obs, p)`` (``p = dx + 1`` for an augmented head); ``V_inv`` is ``(..., p,
    p)``, the column precision (``V_inv += Σ E[x̃x̃ᵀ]``). The row covariance is the noise, supplied
    by the consumer, and is not stored.
    """

    M: Array  # (..., d_obs, p) posterior mean head
    V_inv: Array  # (..., p, p) column precision
    M0: Array | None = None  # decay anchors (D7); required iff decay < 1
    V_inv0: Array | None = None

    def __check_init__(self) -> None:
        p = self.M.shape[-1]
        if self.V_inv.shape[-2:] != (p, p):
            raise ValueError(f"V_inv trailing shape {self.V_inv.shape[-2:]} != (p, p) with p={p}")
        for name, v, ref in (("M0", self.M0, self.M), ("V_inv0", self.V_inv0, self.V_inv)):
            if v is not None and v.shape != ref.shape:
                raise ValueError(f"{name} shape {v.shape} != parameter shape {ref.shape}")
        if self.V_inv0 is not None:
            _check_positive("V_inv0 diagonal", jnp.diagonal(self.V_inv0, axis1=-2, axis2=-1))

    @classmethod
    def from_head(cls, head: Array, strength: float = 1.0, anchor: bool = False):
        """Prior centred on a known head: ``point()`` reproduces it exactly;
        ``strength`` is the pseudo-observation count on the column precision."""
        p = head.shape[-1]
        v_inv = strength * jnp.broadcast_to(jnp.eye(p, dtype=head.dtype), (*head.shape[:-2], p, p))
        return cls(
            M=head, V_inv=v_inv, M0=head if anchor else None, V_inv0=v_inv if anchor else None
        )

    @property
    def obs_dim(self) -> int:
        return self.M.shape[-2]

    def point(self) -> Array:
        """The posterior-mean head — what every probability-kind consumer takes."""
        return self.M

    def stats(self, syx: Array, sxx: Array, count: Array) -> tuple[Array, Array, Array]:
        """Package one tick's (noise-whitened, responsibility-weighted)
        regression statistic: ``syx = Σ y x̃ᵀ`` ``(..., d_obs, p)``,
        ``sxx = Σ E[x̃x̃ᵀ]`` ``(..., p, p)``, and the responsibility mass."""
        return syx, sxx, count

    def absorb(self, stats, lr=1.0, decay=1.0) -> "MatrixNormal":
        """Conjugate Bayesian-regression update in natural coordinates:
        ``V_inv += sxx`` and ``(M V_inv) += syx``, then solve for ``M``."""
        syx, sxx, _ = stats
        nat_mean = self.M @ self.V_inv
        nat_mean0 = None if self.M0 is None else self.M0 @ self.V_inv0
        new_vinv = _anchored(self.V_inv, self.V_inv0, sxx, lr, decay, "V_inv0")
        new_nat = _anchored(nat_mean, nat_mean0, syx, lr, decay, "M0")
        new_m = jnp.linalg.solve(jnp.swapaxes(new_vinv, -1, -2), jnp.swapaxes(new_nat, -1, -2))
        return MatrixNormal(
            M=jnp.swapaxes(new_m, -1, -2),
            V_inv=new_vinv,
            M0=self.M0,
            V_inv0=self.V_inv0,
        )

    def sample(self, key: Array, row_chol: Array) -> Array:
        """One head draw ``M + L_row Z L_colᵀ``; the row Cholesky is the noise
        factor, supplied by the caller (the class stores no row covariance)."""
        col_chol = cholesky_of_inverse(self.V_inv)
        z = jax.random.normal(key, self.M.shape, dtype=self.M.dtype)
        return self.M + row_chol @ z @ jnp.swapaxes(col_chol, -1, -2)

    def expected_log_params(self) -> Array:
        raise NotImplementedError(
            "LG inference consumes the point head (point()); no E[log W] consumer "
            "exists (design: parameter-learning.md D8)"
        )

    def novelty(self) -> Array:
        raise deferred_novelty("a MatrixNormal head")
