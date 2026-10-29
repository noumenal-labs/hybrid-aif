"""v0.2 machinery-gap tests: per-step weight schedules and Gaussian<->mixture KL.

Covers the two conveniences the v0.1 demos worked around: ``rollout_reduced``
accepting a leading-horizon-axis schedule of transition log-weights, and
``kl_to`` between a ``Gaussian`` and a ``MixtureGaussian`` in either direction
(1-component promotion reusing the Kolchinsky-Tracey bound machinery).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions.gaussian import Gaussian
from hybrid_aif.distributions.mixture import MixtureGaussian
from hybrid_aif.inference import rollout_reduced


def random_mixture(seed, K, dim, spread=3.0):
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    lw = jax.random.normal(k1, (K,))
    means = spread * jax.random.normal(k2, (K, dim))
    a = 0.3 * jax.random.normal(k3, (K, dim, dim))
    covs = a @ jnp.swapaxes(a, -1, -2) + 0.5 * jnp.eye(dim)
    return MixtureGaussian(log_weights=lw, means=means, chols=jnp.linalg.cholesky(covs))


def random_gaussian(seed, dim, spread=2.0):
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    a = 0.3 * jax.random.normal(k2, (dim, dim))
    return Gaussian(
        loc=spread * jax.random.normal(k1, (dim,)),
        chol=jnp.linalg.cholesky(a @ a.T + 0.5 * jnp.eye(dim)),
    )


def switching_model(M=2, dim=1):
    Fs = jnp.stack([0.9 * jnp.eye(dim), 0.5 * jnp.eye(dim)])[:M]
    bs = jnp.asarray([[-1.0] * dim, [1.5] * dim])[:M]
    Qc = jnp.stack([0.3 * jnp.eye(dim), 0.4 * jnp.eye(dim)])[:M]
    return Fs, bs, Qc


def assert_trees_equal(a, b, exact=True):
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        if exact:
            assert np.array_equal(np.asarray(x), np.asarray(y))
        else:
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-6, rtol=1e-6)


# ------------------------------------------------- schedule: equivalence


def test_constant_schedule_bitwise_matches_static_shared_weights():
    m = random_mixture(0, 2, 1)
    Fs, bs, Qc = switching_model()
    tlw = jnp.log(jnp.asarray([0.4, 0.6]))
    static = rollout_reduced(m, Fs, bs, Qc, tlw, horizon=4, k=2)
    sched = rollout_reduced(m, Fs, bs, Qc, jnp.tile(tlw, (4, 1)), horizon=4, k=2)
    assert_trees_equal(static, sched, exact=True)


def test_constant_schedule_bitwise_matches_static_per_component_weights():
    m = random_mixture(1, 3, 1)
    Fs, bs, Qc = switching_model()
    tlw = jax.nn.log_softmax(jax.random.normal(jax.random.PRNGKey(7), (3, 2)), axis=-1)
    static = rollout_reduced(m, Fs, bs, Qc, tlw, horizon=4, k=3)
    sched = rollout_reduced(m, Fs, bs, Qc, jnp.tile(tlw, (4, 1, 1)), horizon=4, k=3)
    assert_trees_equal(static, sched, exact=True)


def test_varying_schedule_matches_horizon_one_loop():
    """A genuinely varying (horizon, M) schedule = the v0.1 demo's per-step loop."""
    m = random_mixture(2, 2, 1)
    Fs, bs, Qc = switching_model()
    schedule = jnp.log(jnp.asarray([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5], [0.1, 0.9], [0.7, 0.3]]))
    final, traj = rollout_reduced(m, Fs, bs, Qc, schedule, horizon=5, k=2)
    carry, per_step = m, []
    for t in range(5):
        carry, _ = rollout_reduced(carry, Fs, bs, Qc, schedule[t], horizon=1, k=2)
        per_step.append(carry)
    assert_trees_equal(final, carry, exact=False)
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *per_step)
    assert_trees_equal(traj, stacked, exact=False)


def test_varying_schedule_actually_varies():
    """Sanity: a varying schedule differs from any of its rows applied statically."""
    m = random_mixture(3, 2, 1)
    Fs, bs, Qc = switching_model()
    schedule = jnp.log(jnp.asarray([[0.95, 0.05], [0.05, 0.95], [0.5, 0.5]]))
    final_s, _ = rollout_reduced(m, Fs, bs, Qc, schedule, horizon=3, k=2)
    final_0, _ = rollout_reduced(m, Fs, bs, Qc, schedule[0], horizon=3, k=2)
    assert not np.allclose(np.asarray(final_s.means), np.asarray(final_0.means))


# ------------------------------------------------- schedule: shape handling


def test_two_dim_leading_k_reads_as_per_component_table_when_ambiguous():
    """horizon == k: the 2-D form keeps its v0.1 (K, M) meaning; 3-D disambiguates."""
    m = random_mixture(4, 2, 1)
    Fs, bs, Qc = switching_model()
    table = jax.nn.log_softmax(jax.random.normal(jax.random.PRNGKey(9), (2, 2)), axis=-1)
    ambiguous = rollout_reduced(m, Fs, bs, Qc, table, horizon=2, k=2)
    explicit = rollout_reduced(m, Fs, bs, Qc, jnp.tile(table, (2, 1, 1)), horizon=2, k=2)
    assert_trees_equal(ambiguous, explicit, exact=True)


def test_bad_schedule_shapes_raise():
    m = random_mixture(5, 2, 1)
    Fs, bs, Qc = switching_model()
    with pytest.raises(ValueError):  # 2-D leading axis matches neither k nor horizon
        rollout_reduced(m, Fs, bs, Qc, jnp.zeros((7, 2)), horizon=3, k=2)
    with pytest.raises(ValueError):  # 3-D schedule with wrong horizon length
        rollout_reduced(m, Fs, bs, Qc, jnp.zeros((4, 2, 2)), horizon=3, k=2)
    with pytest.raises(ValueError):  # 4-D is not a recognized form
        rollout_reduced(m, Fs, bs, Qc, jnp.zeros((3, 2, 2, 1)), horizon=3, k=2)


def test_schedule_jit_and_vmap_legal():
    m = random_mixture(6, 2, 1)
    Fs, bs, Qc = switching_model()
    schedule = jnp.log(jnp.asarray([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]]))

    def run(d, s):
        final, traj = rollout_reduced(d, Fs, bs, Qc, s, horizon=3, k=2)
        return final, traj

    eager = run(m, schedule)
    jitted = jax.jit(run)(m, schedule)
    assert_trees_equal(eager, jitted, exact=True)

    stacked = jax.tree_util.tree_map(lambda a: jnp.stack([a, a, a]), m)
    vfinal, vtraj = jax.vmap(lambda d: run(d, schedule))(stacked)
    assert vfinal.log_weights.shape == (3, 2)
    assert vtraj.log_weights.shape == (3, 3, 2)
    for i in range(3):
        row = jax.tree_util.tree_map(lambda a: a[i], vfinal)  # noqa: B023
        assert_trees_equal(row, eager[0], exact=False)


# ------------------------------------------------- KL: exact degradation


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_promoted_gaussian_vs_gaussian_reproduces_closed_form(seed):
    g1, g2 = random_gaussian(seed, 2), random_gaussian(seed + 10, 2)
    for f, g in [(g1, g2), (g2, g1)]:
        exact = float(f.kl_to(g).lower)
        via_mixture_self = MixtureGaussian.from_gaussian(f).kl_to(g)
        via_mixture_other = f.kl_to(MixtureGaussian.from_gaussian(g))
        for iv in (via_mixture_self, via_mixture_other):
            assert float(iv.width) <= 1e-6
            np.testing.assert_allclose(float(iv.lower), exact, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(float(iv.upper), exact, atol=1e-5, rtol=1e-5)


def test_gaussian_gaussian_path_unchanged():
    g1, g2 = random_gaussian(3, 2), random_gaussian(13, 2)
    iv = g1.kl_to(g2)
    assert float(iv.width) == 0.0
    assert float(iv.lower) >= 0.0


# ------------------------------------------------- KL: bounds sandwich MC


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mixture_to_gaussian_kl_sandwiches_mc(seed):
    f = random_mixture(seed, 3, 2, spread=2.0)
    g = random_gaussian(seed + 20, 2)
    iv = f.kl_to(g)
    assert float(iv.lower) <= float(iv.upper)
    assert float(iv.lower) >= 0.0
    x = f.sample(jax.random.PRNGKey(300 + seed), (200_000,))
    mc = float((f.log_prob(x) - g.log_prob(x)).mean())
    assert float(iv.lower) - 0.02 <= mc <= float(iv.upper) + 0.02


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gaussian_to_mixture_kl_sandwiches_mc(seed):
    g = random_gaussian(seed + 30, 2)
    m = random_mixture(seed, 3, 2, spread=2.0)
    iv = g.kl_to(m)
    assert float(iv.lower) <= float(iv.upper)
    assert float(iv.lower) >= 0.0
    x = g.sample(jax.random.PRNGKey(400 + seed), (200_000,))
    mc = float((g.log_prob(x) - m.log_prob(x)).mean())
    assert float(iv.lower) - 0.02 <= mc <= float(iv.upper) + 0.02


# ------------------------------------------------- KL: gradients and jit


def test_cross_family_kl_gradients_finite():
    m = random_mixture(7, 3, 2, spread=2.0)

    def to_mixture(loc):
        return Gaussian(loc=loc, chol=jnp.eye(2)).kl_to(m).midpoint

    def from_mixture(loc):
        return m.kl_to(Gaussian(loc=loc, chol=jnp.eye(2))).midpoint

    loc = jnp.asarray([0.5, -0.3])
    for fn in (to_mixture, from_mixture):
        grad = jax.grad(fn)(loc)
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert float(jnp.abs(grad).max()) > 0.0


def test_cross_family_kl_jits():
    m = random_mixture(8, 2, 2, spread=2.0)
    g = random_gaussian(8, 2)
    for fn in (lambda a, b: a.kl_to(b), lambda a, b: b.kl_to(a)):
        eager = fn(g, m)
        jitted = jax.jit(lambda a, b, f=fn: f(a, b))(g, m)
        np.testing.assert_allclose(float(eager.lower), float(jitted.lower), atol=1e-6)
        np.testing.assert_allclose(float(eager.upper), float(jitted.upper), atol=1e-6)
