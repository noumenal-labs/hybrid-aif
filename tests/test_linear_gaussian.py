"""Kalman predict/condition vs a dense float64 numpy reference (de-risk report §2)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Gaussian
from hybrid_aif.inference import condition, predict

_LOG_2PI = np.log(2.0 * np.pi)


def f32(x):
    return jnp.asarray(x, jnp.float32)


def random_system(seed, dx, dy):
    rng = np.random.default_rng(seed)

    def spd(d):
        a = rng.standard_normal((d, d))
        return a @ a.T + d * np.eye(d)

    return dict(
        mu=rng.standard_normal(dx),
        P=spd(dx),
        F=0.5 * rng.standard_normal((dx, dx)),
        b=rng.standard_normal(dx),
        Q=spd(dx),
        H=rng.standard_normal((dy, dx)),
        c=rng.standard_normal(dy),
        R=spd(dy),
        y=rng.standard_normal(dy),
    )


def np_predict(s):
    return s["F"] @ s["mu"] + s["b"], s["F"] @ s["P"] @ s["F"].T + s["Q"]


def np_condition(s):
    S = s["H"] @ s["P"] @ s["H"].T + s["R"]
    v = s["y"] - (s["H"] @ s["mu"] + s["c"])
    K = s["P"] @ s["H"].T @ np.linalg.inv(S)
    mu = s["mu"] + K @ v
    P = (np.eye(len(s["mu"])) - K @ s["H"]) @ s["P"]
    ll = -0.5 * (len(v) * _LOG_2PI + v @ np.linalg.solve(S, v) + np.log(np.linalg.det(S)))
    return mu, P, ll


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("dx,dy", [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3)])
def test_predict_matches_numpy(seed, dx, dy):
    s = random_system(seed, dx, dy)
    g = Gaussian.from_cov(f32(s["mu"]), f32(s["P"]))
    out = predict(g, f32(s["F"]), f32(s["b"]), f32(np.linalg.cholesky(s["Q"])))
    mu, P = np_predict(s)
    np.testing.assert_allclose(np.asarray(out.loc), mu, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(np.asarray(out.cov), P, atol=5e-5, rtol=5e-5)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("dx,dy", [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3)])
def test_condition_matches_numpy(seed, dx, dy):
    s = random_system(seed, dx, dy)
    g = Gaussian.from_cov(f32(s["mu"]), f32(s["P"]))
    post, ll = condition(g, f32(s["H"]), f32(s["c"]), f32(np.linalg.cholesky(s["R"])), f32(s["y"]))
    mu, P, ll_np = np_condition(s)
    np.testing.assert_allclose(np.asarray(post.loc), mu, atol=5e-5, rtol=5e-5)
    np.testing.assert_allclose(np.asarray(post.cov), P, atol=1e-4, rtol=1e-4)
    # log marginal likelihood = numpy Gaussian density of the innovation
    np.testing.assert_allclose(float(ll), ll_np, atol=5e-5, rtol=5e-5)


def test_condition_log_marginal_is_innovation_density():
    s = random_system(7, 2, 2)
    g = Gaussian.from_cov(f32(s["mu"]), f32(s["P"]))
    _, ll = condition(g, f32(s["H"]), f32(s["c"]), f32(np.linalg.cholesky(s["R"])), f32(s["y"]))
    innov = Gaussian.from_cov(
        f32(s["H"] @ s["mu"] + s["c"]), f32(s["H"] @ s["P"] @ s["H"].T + s["R"])
    )
    np.testing.assert_allclose(float(ll), float(innov.log_prob(f32(s["y"]))), atol=1e-5)


def test_batch_broadcasting():
    s = random_system(0, 2, 2)
    locs = f32(np.stack([s["mu"], s["mu"] + 1.0, s["mu"] - 1.0]))
    chol = f32(np.linalg.cholesky(s["P"]))
    g = Gaussian(loc=locs, chol=jnp.broadcast_to(chol, (3, 2, 2)))
    out = predict(g, f32(s["F"]), f32(s["b"]), f32(np.linalg.cholesky(s["Q"])))
    assert out.batch_shape == (3,)
    post, ll = condition(g, f32(s["H"]), f32(s["c"]), f32(np.linalg.cholesky(s["R"])), f32(s["y"]))
    assert post.batch_shape == (3,) and ll.shape == (3,)
    # unbatched g against batched observations
    g0 = Gaussian(loc=locs[0], chol=chol)
    ys = f32(np.stack([s["y"], s["y"] + 0.5]))
    post2, ll2 = condition(g0, f32(s["H"]), f32(s["c"]), f32(np.linalg.cholesky(s["R"])), ys)
    assert post2.batch_shape == (2,) and ll2.shape == (2,)


def test_jit_and_vmap_smoke():
    s = random_system(1, 2, 1)
    g = Gaussian.from_cov(f32(s["mu"]), f32(s["P"]))
    args = (f32(s["H"]), f32(s["c"]), f32(np.linalg.cholesky(s["R"])), f32(s["y"]))
    post, ll = jax.jit(condition)(g, *args)
    post_ref, ll_ref = condition(g, *args)
    np.testing.assert_allclose(np.asarray(post.loc), np.asarray(post_ref.loc), atol=1e-6)
    np.testing.assert_allclose(float(ll), float(ll_ref), atol=1e-6)
    stacked = jax.tree_util.tree_map(lambda a: jnp.stack([a, a]), g)
    lls = jax.vmap(lambda d: condition(d, *args)[1])(stacked)
    np.testing.assert_allclose(np.asarray(lls), float(ll_ref), atol=1e-6)
