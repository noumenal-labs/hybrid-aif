"""``inference.linearise``: a neural conditional read at a belief as an LG observation model.

Design: recognition-beside-analytic.md L1.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import MLPConditional
from hybrid_aif.distributions.gaussian import sigma_points
from hybrid_aif.inference.linearise import UNSCENTED_KAPPA, linearise

D, D_OBS = 2, 3
LOC = jnp.array([0.3, -0.4])
CHOL = jnp.array([[0.5, 0.0], [0.1, 0.3]])


KEY_AFFINE, KEY_CURVED = jax.random.PRNGKey(0), jax.random.PRNGKey(1)


def _affine(key=KEY_AFFINE):
    """A depth-0 conditional: affine mean, and a constant scale (its scale rows zeroed)."""
    decl = MLPConditional.from_shapes((D, 3), D_OBS, key=key, depth=0, frozen=True)
    last = decl.layers[-1]
    weight = last.weight.at[D_OBS:].set(0.0)
    return eqx.tree_at(lambda d: d.layers[-1].weight, decl, weight)


def _curved(key=KEY_CURVED):
    return MLPConditional.from_shapes((D, 3), D_OBS, key=key, width=16, depth=2, frozen=True)


PARENTS = (jnp.zeros(D), jnp.array([0.0, 1.0, 0.0]))


@pytest.mark.parametrize("propagation", ["first_order", "unscented"])
def test_an_affine_conditional_is_read_exactly(propagation):
    decl = _affine()
    H, c, R_chol = linearise(decl, LOC, CHOL, PARENTS, 0, propagation)
    for x in (LOC, LOC + 1.0, jnp.array([-2.0, 3.0])):
        mean, scale = decl._head((x, PARENTS[1]))
        np.testing.assert_allclose(H @ x + c, mean, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(R_chol @ R_chol.T, jnp.diag(scale**2), rtol=1e-5, atol=1e-6)


def test_first_order_is_the_jacobian_and_the_scale_at_the_mean():
    decl = _curved()
    H, c, R_chol = linearise(decl, LOC, CHOL, PARENTS, 0, "first_order")
    mean, scale = decl._head((LOC, PARENTS[1]))
    jac = jax.jacfwd(lambda x: decl._head((x, PARENTS[1]))[0])(LOC)
    np.testing.assert_allclose(H, jac, rtol=1e-6)
    np.testing.assert_allclose(H @ LOC + c, mean, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(R_chol, jnp.diag(scale), rtol=1e-6)


def test_unscented_matches_the_sigma_points_moments_and_tends_to_first_order():
    decl = _curved()
    H, c, R_chol = linearise(decl, LOC, CHOL, PARENTS, 0, "unscented")
    points, weights = sigma_points(LOC, CHOL, kappa=UNSCENTED_KAPPA)
    means, scales = jax.vmap(lambda x: decl._head((x, PARENTS[1])))(points)
    y_bar = weights @ means
    dy = means - y_bar
    P_yy = (dy * weights[:, None]).T @ dy + jnp.diag(weights @ scales**2)
    cov = CHOL @ CHOL.T
    np.testing.assert_allclose(H @ LOC + c, y_bar, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(H @ cov @ H.T + R_chol @ R_chol.T, P_yy, rtol=1e-4, atol=1e-6)
    assert np.all(np.diag(R_chol) > 0)
    tight = 1e-3 * CHOL
    H_u, c_u, _ = linearise(decl, LOC, tight, PARENTS, 0, "unscented")
    H_f, c_f, _ = linearise(decl, LOC, tight, PARENTS, 0, "first_order")
    np.testing.assert_allclose(H_u, H_f, rtol=1e-2, atol=1e-3)
    np.testing.assert_allclose(c_u, c_f, rtol=1e-2, atol=1e-3)


def test_the_read_is_differentiable_in_the_belief_and_batches_over_components():
    decl = _curved()
    grad = jax.grad(lambda loc: jnp.sum(linearise(decl, loc, CHOL, PARENTS, 0)[0] ** 2))(LOC)
    assert np.all(np.isfinite(grad))
    locs, chols = jnp.stack([LOC, LOC + 0.5]), jnp.stack([CHOL, 0.5 * CHOL])
    H, c, R = jax.vmap(lambda lo, ch: linearise(decl, lo, ch, PARENTS, 0, "unscented"))(locs, chols)
    assert H.shape == (2, D_OBS, D) and c.shape == (2, D_OBS) and R.shape == (2, D_OBS, D_OBS)


@pytest.mark.parametrize("spread", [1e2, 1e3])
def test_the_unscented_noise_stays_finite_under_a_wide_belief(spread):
    decl = _affine()
    _, _, R_chol = linearise(decl, LOC, spread * CHOL, PARENTS, 0, "unscented")
    _, scale = decl._head((LOC, PARENTS[1]))
    assert np.all(np.isfinite(R_chol))
    np.testing.assert_allclose(np.diag(R_chol), scale, rtol=1e-3)


def test_an_unknown_propagation_is_refused():
    with pytest.raises(ValueError, match="propagation must be one of"):
        linearise(_affine(), LOC, CHOL, PARENTS, 0, "sampled")
