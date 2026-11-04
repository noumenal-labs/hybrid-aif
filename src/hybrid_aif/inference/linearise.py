"""A neural conditional read at a Gaussian belief as a linear-Gaussian observation model.

A neural likelihood ``o = m(x, parents) + eps``, ``eps ~ N(0, diag sigma(x, parents)^2)`` (an
:class:`~hybrid_aif.distributions.MLPConditional`), over a Gaussian belief ``N(loc, chol chol^T)``
on its continuous parent ``x``, is read as the affine model ``o = H x + c + eps'``,
``eps' ~ N(0, R)`` whose moments match the belief's push through ``m`` under ``propagation``:

- ``"first_order"``: ``H`` is the Jacobian of ``m`` at ``loc``, ``c = m(loc) - H loc`` and
  ``R = diag sigma(loc)^2``; the extended Kalman reading, exact for ``m`` affine in ``x``.
- ``"unscented"``: the ``2d + 1`` sigma points of the belief through ``m``; ``H`` is the
  statistical linearisation ``P_xy^T Sigma^-1``, ``c = y_bar - H loc`` and ``R`` the weighted
  covariance of the fit's residuals at the points plus the averaged noise
  ``sum_i w_i diag sigma(x_i)^2`` (equal to ``P_yy - H Sigma H^T``, formed without the
  subtraction), so ``H loc + c`` and ``H Sigma H^T + R`` are the sigma points' mean and
  covariance of ``o``; exact for affine ``m``, moment-matching to degree three otherwise.

The other parents are held fixed (a categorical parent as its one-hot or probability vector).
Design: docs/design/recognition-beside-analytic.md, amendment L1.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.distributions.gaussian import sigma_points

PROPAGATIONS = ("first_order", "unscented")

#: The sigma points' scaling, the transitions' rule (objectives/hybrid.py): ``kappa = 0`` puts
#: the points at ``loc ± sqrt(d) · chol[:, i]`` with weight ``1/(2d)`` each and none on the centre.
UNSCENTED_KAPPA = 0.0

#: Added to the unscented ``R``'s diagonal before its Cholesky factor, against round-off in the
#: Schur complement; far below any declared noise floor.
_JITTER = 1e-9


def linearise(
    decl,
    loc: Array,
    chol: Array,
    parents: Sequence[Array],
    index: int,
    propagation: str = "first_order",
) -> tuple[Array, Array, Array]:
    """``(H, c, R_chol)`` of ``decl`` read at ``N(loc, chol chol^T)`` on its parent ``index``.

    ``parents`` is the full parent tuple in the declaration's order; its entry at ``index`` is
    replaced by ``x`` and the others are held fixed. ``H`` is ``(d_obs, d)``, ``c`` is
    ``(d_obs,)`` and ``R_chol`` is ``(d_obs, d_obs)`` lower triangular. Batch over components
    with ``jax.vmap`` on ``loc`` and ``chol``.
    """

    def head(x: Array) -> tuple[Array, Array]:
        ps = list(parents)
        ps[index] = x
        return decl._head(tuple(ps))

    if propagation == "first_order":
        mean, scale = head(loc)
        jac = jax.jacfwd(lambda x: head(x)[0])(loc)
        return jac, mean - jac @ loc, jnp.diag(scale)
    if propagation == "unscented":
        points, weights = sigma_points(loc, chol, kappa=UNSCENTED_KAPPA)
        means, scales = jax.vmap(head)(points)
        y_bar = weights @ means
        dx, dy = points - loc, means - y_bar
        P_xy = (dx * weights[:, None]).T @ dy
        H = jax.scipy.linalg.cho_solve((chol, True), P_xy).T
        c = y_bar - H @ loc
        # R from the residuals of the affine fit at the points, not as P_yy - H Sigma H^T: the
        # same matrix in exact arithmetic (the points reproduce Sigma), positive semidefinite by
        # construction, and the noise is added outside the subtraction.
        resid = dy - dx @ H.T
        R = (resid * weights[:, None]).T @ resid + jnp.diag(weights @ scales**2)
        R = 0.5 * (R + R.T) + _JITTER * jnp.eye(R.shape[-1], dtype=R.dtype)
        return H, c, jnp.linalg.cholesky(R)
    raise ValueError(
        f"propagation must be one of {list(PROPAGATIONS)}, got {propagation!r} (design: "
        "recognition-beside-analytic.md L1)"
    )
