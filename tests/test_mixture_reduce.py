"""Mixture grow-then-reduce correctness suite (de-risk report §2).

Covers: grow-without-reduce exactness vs numpy enumeration, moment-matching
invariants of Runnalls reduction, KT bound sandwiches on reduced mixtures and
mixture-mixture KL, policy-ranking fidelity through the reduced pipeline,
jit/vmap/grad smoke, and dead-component handling.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import MixtureGaussian
from hybrid_aif.inference import propagate, rollout_gpb1, rollout_reduced

KEY = jax.random.PRNGKey(0)


def random_mixture(seed, K, dim, spread=3.0):
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    lw = jax.random.normal(k1, (K,))
    means = spread * jax.random.normal(k2, (K, dim))
    a = 0.3 * jax.random.normal(k3, (K, dim, dim))
    covs = a @ jnp.swapaxes(a, -1, -2) + 0.5 * jnp.eye(dim)
    return MixtureGaussian(log_weights=lw, means=means, chols=jnp.linalg.cholesky(covs))


# ------------------------------------------------------- grow exactness


def test_grow_without_reduce_matches_enumeration():
    """1D state, M=2 modes, H=2: 4-component grow vs exact numpy enumeration."""
    mu0, var0 = 0.5, 1.5
    F = np.array([0.9, -0.4])
    b = np.array([-2.0, 3.0])
    q = np.array([0.09, 0.25])
    t = np.array([0.3, 0.7])

    m = MixtureGaussian(
        log_weights=jnp.zeros(1),
        means=jnp.asarray([[mu0]]),
        chols=jnp.asarray([[[np.sqrt(var0)]]], dtype=jnp.float32),
    )
    Fs = jnp.asarray(F, jnp.float32).reshape(2, 1, 1)
    bs = jnp.asarray(b, jnp.float32).reshape(2, 1)
    Qc = jnp.sqrt(jnp.asarray(q, jnp.float32)).reshape(2, 1, 1)
    tlw = jnp.log(jnp.asarray(t, jnp.float32))
    grown = propagate(propagate(m, Fs, bs, Qc, tlw), Fs, bs, Qc, tlw)
    assert grown.num_components == 4

    # enumeration over mode sequences (j1, j2), component-major order j1*2 + j2
    for slot, (j1, j2) in enumerate(itertools.product(range(2), range(2))):
        mu1 = F[j1] * mu0 + b[j1]
        v1 = F[j1] ** 2 * var0 + q[j1]
        mu2 = F[j2] * mu1 + b[j2]
        v2 = F[j2] ** 2 * v1 + q[j2]
        w = t[j1] * t[j2]
        np.testing.assert_allclose(float(grown.means[slot, 0]), mu2, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(float(grown.chols[slot, 0, 0]) ** 2, v2, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(float(jax.nn.softmax(grown.log_weights)[slot]), w, atol=1e-6)

    # density agrees with the enumerated mixture at probe points
    def np_logpdf(x):
        total = 0.0
        for j1, j2 in itertools.product(range(2), range(2)):
            mu1 = F[j1] * mu0 + b[j1]
            v1 = F[j1] ** 2 * var0 + q[j1]
            mu2 = F[j2] * mu1 + b[j2]
            v2 = F[j2] ** 2 * v1 + q[j2]
            total += t[j1] * t[j2] * np.exp(-0.5 * (x - mu2) ** 2 / v2) / np.sqrt(2 * np.pi * v2)
        return np.log(total)

    for x in [-3.0, 0.0, 1.7, 4.0]:
        np.testing.assert_allclose(
            float(grown.log_prob(jnp.asarray([x]))), np_logpdf(x), atol=1e-5, rtol=1e-5
        )


def test_rollout_with_full_budget_reduce_is_noop_exact():
    """k = K*M makes reduce a no-op; rollout must equal raw propagate exactly."""
    m = random_mixture(0, 2, 1)
    Fs = jnp.asarray([[[0.9]], [[0.5]]])
    bs = jnp.asarray([[-1.0], [1.0]])
    Qc = jnp.asarray([[[0.3]], [[0.4]]])
    tlw = jnp.log(jnp.asarray([0.4, 0.6]))
    grown = propagate(m, Fs, bs, Qc, tlw)
    assert grown.reduce(grown.num_components) is grown


# ------------------------------------------------------- moment preservation


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("k", [7, 5, 3])
def test_reduce_preserves_mean_and_cov(seed, k):
    """Moment matching invariant: every merge (so any k) preserves mixture moments."""
    m = random_mixture(seed, 8, 2)
    r = m.reduce(k)
    assert r.num_components == k
    np.testing.assert_allclose(np.asarray(r.mean()), np.asarray(m.mean()), atol=2e-5)
    np.testing.assert_allclose(np.asarray(r.cov()), np.asarray(m.cov()), atol=2e-4)


def test_reduce_output_weights_sorted_and_normalized():
    r = random_mixture(1, 8, 2).reduce(3)
    lw = np.asarray(r.log_weights)
    assert np.all(np.diff(lw) <= 1e-6)  # lax.top_k: descending weight order
    np.testing.assert_allclose(np.exp(lw).sum(), 1.0, atol=1e-6)


# ------------------------------------------------------- entropy sandwich


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reduced_mixture_entropy_bounds_sandwich_mc(seed):
    r = random_mixture(seed, 8, 2).reduce(3)
    iv = r.entropy()
    x = r.sample(jax.random.PRNGKey(100 + seed), (200_000,))
    mc = float(-r.log_prob(x).mean())
    assert float(iv.lower) - 0.02 <= mc <= float(iv.upper) + 0.02
    assert float(iv.width) <= np.log(3) + 1e-6


# ------------------------------------------------------- kl_to


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kl_bounds_sandwich_mc(seed):
    f = random_mixture(seed, 2, 2, spread=2.0)
    g = random_mixture(seed + 10, 2, 2, spread=2.0)
    iv = f.kl_to(g)
    x = f.sample(jax.random.PRNGKey(200 + seed), (200_000,))
    mc = float((f.log_prob(x) - g.log_prob(x)).mean())
    assert float(iv.lower) - 0.02 <= mc <= float(iv.upper) + 0.02
    assert float(iv.lower) >= 0.0


def test_kl_self_interval_contains_zero():
    """KL(f‖f): observed behavior — lower clamps to 0; upper equals the KT
    entropy-bound gap of f (CE upper with g=f is exactly the KT entropy upper
    bound), so the width is tiny only when f's entropy bounds are tight."""
    f = MixtureGaussian(
        log_weights=jnp.log(jnp.asarray([0.5, 0.5])),
        means=jnp.asarray([[-0.5], [0.5]]),
        chols=jnp.ones((2, 1, 1)),
    )
    iv = f.kl_to(f)
    assert float(iv.lower) == 0.0
    assert float(iv.upper) >= 0.0
    np.testing.assert_allclose(float(iv.upper), float(f.entropy().width), atol=1e-6)
    # well-separated f: entropy bounds tight, so the self-KL interval is tiny
    sep = random_mixture(0, 2, 2, spread=30.0)
    iv_sep = sep.kl_to(sep)
    assert float(iv_sep.lower) == 0.0
    assert float(iv_sep.upper) <= 1e-3


# ------------------------------------------------------- policy invariant


def test_reduced_pipeline_preserves_entropy_ordering():
    """Diverging vs near-identical transition sets: reduced-pipeline entropy
    ordering must match full-enumeration ordering (ranking fidelity)."""
    m0 = MixtureGaussian(
        log_weights=jnp.zeros(1), means=jnp.zeros((1, 1)), chols=0.5 * jnp.ones((1, 1, 1))
    )
    Fs = jnp.ones((2, 1, 1)) * 0.9
    Qc = jnp.ones((2, 1, 1)) * 0.3
    tlw = jnp.log(jnp.asarray([0.5, 0.5]))
    bs_diverge = jnp.asarray([[-4.0], [4.0]])
    bs_similar = jnp.asarray([[-0.1], [0.1]])

    def full_entropy(bs):
        m = m0
        for _ in range(3):
            m = propagate(m, Fs, bs, Qc, tlw)
        return float(m.entropy().midpoint)

    def reduced_entropy(bs):
        final, _ = rollout_reduced(m0, Fs, bs, Qc, tlw, horizon=3, k=2)
        return float(final.entropy().midpoint)

    assert full_entropy(bs_diverge) > full_entropy(bs_similar)
    assert reduced_entropy(bs_diverge) > reduced_entropy(bs_similar)


def test_gpb1_rollout_matches_moment_matched_enumeration():
    """GPB1 step = exact moment match of the one-step grown mixture."""
    m0 = random_mixture(2, 2, 1)
    Fs = jnp.asarray([[[0.9]], [[0.4]]])
    bs = jnp.asarray([[-1.0], [2.0]])
    Qc = jnp.asarray([[[0.3]], [[0.5]]])
    tlw = jnp.log(jnp.asarray([0.5, 0.5]))
    final, traj = rollout_gpb1(m0, Fs, bs, Qc, tlw, horizon=1)
    g0 = m0.moment_match()
    ref = propagate(
        MixtureGaussian(log_weights=jnp.zeros(1), means=g0.loc[None], chols=g0.chol[None]),
        Fs,
        bs,
        Qc,
        tlw,
    ).moment_match()
    np.testing.assert_allclose(float(final.loc[0]), float(ref.loc[0]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(final.cov), np.asarray(ref.cov), atol=1e-5)
    assert traj.loc.shape == (1, 1)


# ------------------------------------------------------- jit / vmap / grad


def test_reduce_and_propagate_jit_vmap_smoke():
    m = random_mixture(3, 4, 2)
    Fs = jnp.broadcast_to(0.8 * jnp.eye(2), (2, 2, 2))
    bs = jnp.asarray([[-1.0, 0.0], [1.0, 0.0]])
    Qc = jnp.broadcast_to(0.4 * jnp.eye(2), (2, 2, 2))
    tlw = jnp.log(jnp.asarray([0.5, 0.5]))

    ref = float(m.reduce(2).entropy().midpoint)
    assert np.isclose(float(jax.jit(lambda d: d.reduce(2).entropy().midpoint)(m)), ref, atol=1e-5)
    stacked = jax.tree_util.tree_map(lambda a: jnp.stack([a, a, a]), m)
    vent = jax.vmap(lambda d: d.reduce(2).entropy().midpoint)(stacked)
    np.testing.assert_allclose(np.asarray(vent), ref, atol=1e-5)
    # batched object without explicit vmap must agree too
    bent = stacked.reduce(2).entropy().midpoint
    np.testing.assert_allclose(np.asarray(bent), ref, atol=1e-5)

    ref_lw = propagate(m, Fs, bs, Qc, tlw).log_weights
    jit_lw = jax.jit(lambda d: propagate(d, Fs, bs, Qc, tlw).log_weights)(m)
    np.testing.assert_allclose(np.asarray(jit_lw), np.asarray(ref_lw), atol=1e-6)
    v_lw = jax.vmap(lambda d: propagate(d, Fs, bs, Qc, tlw).log_weights)(stacked)
    assert v_lw.shape == (3, 8)


def test_grad_through_reduce_matches_finite_differences():
    """d(entropy of reduce(propagate(...)))/dF entry, away from merge boundaries."""
    m = MixtureGaussian(
        log_weights=jnp.log(jnp.asarray([0.6, 0.4])),
        means=jnp.asarray([[-4.0], [4.0]]),
        chols=jnp.asarray([[[0.7]], [[1.1]]]),
    )
    bs = jnp.asarray([[-5.0], [5.0]])
    Qc = 0.4 * jnp.ones((2, 1, 1))
    tlw = jnp.log(jnp.asarray([0.5, 0.5]))

    def loss(f_entry):
        Fs = jnp.stack([jnp.full((1, 1), f_entry), jnp.full((1, 1), 0.8)])
        return propagate(m, Fs, bs, Qc, tlw).reduce(2).entropy().midpoint

    grad = float(jax.grad(loss)(0.9))
    eps = 1e-2
    fd = float((loss(0.9 + eps) - loss(0.9 - eps)) / (2 * eps))
    assert np.isfinite(grad)
    np.testing.assert_allclose(grad, fd, rtol=5e-3, atol=1e-4)


# ------------------------------------------------------- dead components


def test_reduce_with_dead_component_no_nans():
    m = random_mixture(4, 4, 2)
    lw = m.log_weights.at[1].set(-jnp.inf)
    md = MixtureGaussian(log_weights=lw, means=m.means, chols=m.chols)
    r = md.reduce(2)
    assert r.num_components == 2
    assert bool(jnp.all(jnp.isfinite(r.means)))
    assert bool(jnp.all(jnp.isfinite(r.chols)))
    assert not bool(jnp.any(jnp.isnan(r.log_weights)))
    # the dead component contributed nothing: moments preserved
    np.testing.assert_allclose(np.asarray(r.mean()), np.asarray(md.mean()), atol=2e-5)
    np.testing.assert_allclose(np.asarray(r.cov()), np.asarray(md.cov()), atol=2e-4)
    assert np.isfinite(float(r.entropy().midpoint))


def test_reduce_with_surplus_dead_components():
    """More dead slots than merges: survivors include dead slots, no NaNs."""
    m = random_mixture(5, 4, 1)
    lw = m.log_weights.at[1].set(-jnp.inf).at[2].set(-jnp.inf).at[3].set(-jnp.inf)
    md = MixtureGaussian(log_weights=lw, means=m.means, chols=m.chols)
    r = md.reduce(3)  # one merge, two dead slots remain among the survivors
    assert not bool(jnp.any(jnp.isnan(r.log_weights)))
    assert bool(jnp.all(jnp.isfinite(r.means)))
    np.testing.assert_allclose(np.asarray(r.mean()), np.asarray(md.mean()), atol=1e-5)
    assert np.isfinite(float(r.log_prob(jnp.zeros(1))))
