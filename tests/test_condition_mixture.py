"""Gaussian-sum conditioning kernel (D1) — exactness and transform safety.

``condition_mixture`` is the mixture measurement update: per-component exact
Kalman ``condition`` plus a marginal-likelihood weight update, renormalized,
with the integrated evidence returned. The load-bearing test is exactness
against an independent brute-force Gaussian-sum computed with explicit matrix
inverses (a different numerical path than the kernel's Cholesky/Joseph form),
at tight tolerance on small cases. Plus: ``K`` is unchanged, and the kernel is
jit/vmap-safe.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import MixtureGaussian
from hybrid_aif.inference import condition_mixture


def _gaussian_logpdf(x, mean, cov):
    """log N(x; mean, cov) via explicit inverse + slogdet (independent path)."""
    d = x.shape[0]
    dev = x - mean
    sign, logdet = np.linalg.slogdet(cov)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + dev @ np.linalg.solve(cov, dev))


def _brute_force_condition(w, mus, covs, H, c, R, y):
    """Exact Gaussian-sum posterior by textbook per-component Kalman + Bayes on
    the mixing weights, all with explicit matrix inverses."""
    K = len(w)
    post_mus, post_covs, log_unnorm = [], [], []
    for j in range(K):
        S = H @ covs[j] @ H.T + R
        gain = covs[j] @ H.T @ np.linalg.inv(S)
        post_mus.append(mus[j] + gain @ (y - (H @ mus[j] + c)))
        post_covs.append(covs[j] - gain @ H @ covs[j])
        log_unnorm.append(np.log(w[j]) + _gaussian_logpdf(y, H @ mus[j] + c, S))
    log_unnorm = np.asarray(log_unnorm)
    log_evidence = np.log(np.sum(np.exp(log_unnorm)))
    w_post = np.exp(log_unnorm - log_evidence)
    return w_post, np.asarray(post_mus), np.asarray(post_covs), log_evidence


def _random_case(seed, K, dx, dy):
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(K))
    mus = rng.normal(size=(K, dx))
    A = rng.normal(size=(K, dx, dx))
    covs = A @ np.transpose(A, (0, 2, 1)) + 0.5 * np.eye(dx)
    H = rng.normal(size=(dy, dx))
    c = rng.normal(size=(dy,))
    Rr = rng.normal(size=(dy, dy))
    R = Rr @ Rr.T + 0.3 * np.eye(dy)
    y = rng.normal(size=(dy,))
    return w, mus, covs, H, c, R, y


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("K,dx,dy", [(3, 2, 2), (4, 2, 1), (2, 3, 2), (5, 1, 1)])
def test_condition_mixture_matches_brute_force(seed, K, dx, dy):
    w, mus, covs, H, c, R, y = _random_case(seed, K, dx, dy)
    w_ref, mu_ref, cov_ref, ev_ref = _brute_force_condition(w, mus, covs, H, c, R, y)

    # float32 kernel vs float64 reference: tolerance isolates float precision,
    # not algorithmic error (observed agreement ~1e-7 on these cases).
    m = MixtureGaussian(
        log_weights=jnp.log(jnp.asarray(w, jnp.float32)),
        means=jnp.asarray(mus, jnp.float32),
        chols=jnp.asarray(np.linalg.cholesky(covs), jnp.float32),
    )
    post, log_evidence = condition_mixture(
        m,
        jnp.asarray(H, jnp.float32),
        jnp.asarray(c, jnp.float32),
        jnp.asarray(np.linalg.cholesky(R), jnp.float32),
        jnp.asarray(y, jnp.float32),
    )
    assert post.num_components == K  # K unchanged by conditioning
    post_cov = np.asarray(post.chols @ jnp.swapaxes(post.chols, -1, -2))
    np.testing.assert_allclose(float(log_evidence), ev_ref, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(jax.nn.softmax(post.log_weights)), w_ref, atol=1e-5)
    np.testing.assert_allclose(np.asarray(post.means), mu_ref, atol=1e-5)
    np.testing.assert_allclose(post_cov, cov_ref, atol=1e-5)


def test_condition_mixture_jit_vmap_smoke():
    w, mus, covs, H, c, R, y = _random_case(7, 3, 2, 2)
    m = MixtureGaussian(
        log_weights=jnp.log(jnp.asarray(w, jnp.float32)),
        means=jnp.asarray(mus, jnp.float32),
        chols=jnp.asarray(np.linalg.cholesky(covs), jnp.float32),
    )
    Hj, cj, Rc, yj = (jnp.asarray(a, jnp.float32) for a in (H, c, np.linalg.cholesky(R), y))
    post, ev = condition_mixture(m, Hj, cj, Rc, yj)
    pj, ej = jax.jit(lambda mm: condition_mixture(mm, Hj, cj, Rc, yj))(m)
    np.testing.assert_allclose(np.asarray(ej), np.asarray(ev), atol=1e-5)
    np.testing.assert_allclose(np.asarray(pj.log_weights), np.asarray(post.log_weights), atol=1e-5)

    stacked = jax.tree_util.tree_map(lambda a: jnp.stack([a, a, a]), m)
    v_post, v_ev = jax.vmap(lambda mm: condition_mixture(mm, Hj, cj, Rc, yj))(stacked)
    assert v_ev.shape == (3,)
    np.testing.assert_allclose(np.asarray(v_ev), np.asarray(ev), atol=1e-5)


def test_gauge_invariance_and_evidence():
    """log-weights are gauge-free: an additive constant on the prior leaves the
    posterior and the (true, normalized-prior) log-evidence unchanged."""
    w, mus, covs, H, c, R, y = _random_case(11, 3, 2, 2)
    m = MixtureGaussian(
        log_weights=jnp.log(jnp.asarray(w, jnp.float32)),
        means=jnp.asarray(mus, jnp.float32),
        chols=jnp.asarray(np.linalg.cholesky(covs), jnp.float32),
    )
    shifted = MixtureGaussian(log_weights=m.log_weights + 5.0, means=m.means, chols=m.chols)
    args = (
        jnp.asarray(H, jnp.float32),
        jnp.asarray(c, jnp.float32),
        jnp.asarray(np.linalg.cholesky(R), jnp.float32),
        jnp.asarray(y, jnp.float32),
    )
    p0, e0 = condition_mixture(m, *args)
    p1, e1 = condition_mixture(shifted, *args)
    np.testing.assert_allclose(float(e0), float(e1), atol=1e-5)
    np.testing.assert_allclose(np.asarray(p0.log_weights), np.asarray(p1.log_weights), atol=1e-5)
