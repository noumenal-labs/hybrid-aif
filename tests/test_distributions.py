"""Contract tests for the distribution layer (workstream A).

Two kinds of test:
- generic contract tests (pytree/jit/vmap/scan legality) run over every family;
- per-family correctness against closed forms and seeded Monte Carlo.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Categorical, Gaussian, MixtureGaussian

KEY = jax.random.PRNGKey(0)
_LOG_2PI = np.log(2.0 * np.pi)


def make_categorical():
    return Categorical(logits=jnp.asarray([0.3, -0.5, 1.1]))


def make_gaussian():
    cov = jnp.asarray([[2.0, 0.6], [0.6, 1.0]])
    return Gaussian.from_cov(loc=jnp.asarray([1.0, -2.0]), cov=cov)


def make_mixture():
    return MixtureGaussian(
        log_weights=jnp.log(jnp.asarray([0.4, 0.6])),
        means=jnp.asarray([[-2.0, 0.0], [2.0, 1.0]]),
        chols=jnp.asarray([jnp.eye(2) * 0.8, jnp.eye(2) * 1.2]),
    )


FAMILIES = [make_categorical, make_gaussian, make_mixture]


# ---------------------------------------------------------------- contract


@pytest.mark.parametrize("make", FAMILIES)
def test_pytree_roundtrip(make):
    d = make()
    leaves, treedef = jax.tree_util.tree_flatten(d)
    d2 = jax.tree_util.tree_unflatten(treedef, leaves)
    x = d.sample(KEY)
    assert jnp.allclose(d.log_prob(x), d2.log_prob(x))


@pytest.mark.parametrize("make", FAMILIES)
def test_jit_with_distribution_argument(make):
    d = make()
    x = d.sample(KEY)
    f = jax.jit(lambda dist, y: dist.log_prob(y))
    assert jnp.allclose(f(d, x), d.log_prob(x), atol=1e-6)


@pytest.mark.parametrize("make", FAMILIES)
def test_vmap_lifts_unbatched_object(make):
    d = make()
    stacked = jax.tree_util.tree_map(lambda a: jnp.stack([a, a, a]), d)
    ent = jax.vmap(lambda dist: dist.entropy().lower)(stacked)
    assert ent.shape == (3,)
    assert jnp.allclose(ent, d.entropy().lower, atol=1e-6)


def test_scan_carries_distribution():
    d = make_gaussian()

    def step(carry, _):
        carry = Gaussian(loc=carry.loc + 1.0, chol=carry.chol)
        return carry, carry.entropy().lower

    final, ents = jax.lax.scan(step, d, None, length=5)
    assert jnp.allclose(final.loc, d.loc + 5.0)
    assert ents.shape == (5,)
    assert jnp.allclose(ents, d.entropy().lower)  # mean shifts leave entropy fixed


def test_batched_leading_axes_broadcast():
    means = jnp.zeros((4, 7, 2))
    chols = jnp.broadcast_to(jnp.eye(2), (4, 7, 2, 2))
    d = Gaussian(loc=means, chol=chols)
    assert d.batch_shape == (4, 7)
    assert d.entropy().lower.shape == (4, 7)
    assert d.log_prob(jnp.zeros((4, 7, 2))).shape == (4, 7)


@pytest.mark.parametrize("make", FAMILIES)
def test_batch_shape_property(make):
    d = make()
    assert d.batch_shape == ()
    stacked = jax.tree_util.tree_map(lambda a: jnp.stack([a, a]), d)
    assert stacked.batch_shape == (2,)


# ---------------------------------------------------------------- categorical


def test_categorical_entropy_exact():
    d = make_categorical()
    iv = d.entropy()
    assert iv.width == 0.0
    p = np.asarray(d.probs)
    assert np.isclose(float(iv.lower), -(p * np.log(p)).sum(), atol=1e-6)


def test_categorical_kl():
    d, e = make_categorical(), Categorical(logits=jnp.asarray([0.0, 0.0, 0.0]))
    assert jnp.isclose(d.kl_to(d).lower, 0.0, atol=1e-6)
    p = np.asarray(d.probs)
    expected = (p * (np.log(p) - np.log(1.0 / 3.0))).sum()
    assert np.isclose(float(d.kl_to(e).lower), expected, atol=1e-6)


def test_categorical_natural_roundtrip():
    d = make_categorical()
    d2 = Categorical.from_natural(*d.natural())
    assert jnp.allclose(d.probs, d2.probs, atol=1e-6)


def test_categorical_sample_frequencies():
    d = make_categorical()
    z = d.sample(KEY, (50_000,))
    freqs = jnp.bincount(z, length=3) / z.shape[0]
    assert jnp.allclose(freqs, d.probs, atol=0.01)


# ---------------------------------------------------------------- gaussian


def test_gaussian_log_prob_matches_manual():
    d = make_gaussian()
    x = jnp.asarray([0.5, -1.0])
    cov = np.asarray(d.cov)
    dev = np.asarray(x - d.loc)
    expected = -0.5 * (2 * _LOG_2PI + dev @ np.linalg.solve(cov, dev)) - 0.5 * np.log(
        np.linalg.det(cov)
    )
    assert np.isclose(float(d.log_prob(x)), expected, atol=1e-5)


def test_gaussian_entropy_exact():
    d = make_gaussian()
    iv = d.entropy()
    assert iv.width == 0.0
    expected = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * np.asarray(d.cov)))
    assert np.isclose(float(iv.lower), expected, atol=1e-5)


def test_gaussian_kl_self_zero_and_1d_closed_form():
    d = make_gaussian()
    assert jnp.isclose(d.kl_to(d).lower, 0.0, atol=1e-5)
    a = Gaussian.from_cov(jnp.asarray([0.0]), jnp.asarray([[1.0]]))
    b = Gaussian.from_cov(jnp.asarray([1.0]), jnp.asarray([[4.0]]))
    expected = 0.5 * (1 / 4 + 1 / 4 - 1 + np.log(4.0))
    assert np.isclose(float(a.kl_to(b).lower), expected, atol=1e-6)


def test_gaussian_natural_roundtrip():
    d = make_gaussian()
    d2 = Gaussian.from_natural(*d.natural())
    assert jnp.allclose(d.loc, d2.loc, atol=1e-5)
    assert jnp.allclose(d.cov, d2.cov, atol=1e-5)


def test_gaussian_sample_moments():
    d = make_gaussian()
    x = d.sample(KEY, (200_000,))
    assert jnp.allclose(x.mean(axis=0), d.loc, atol=0.02)
    assert jnp.allclose(jnp.cov(x.T), d.cov, atol=0.03)


# ---------------------------------------------------------------- mixture


def test_mixture_log_prob_matches_manual():
    d = make_mixture()
    x = jnp.asarray([0.0, 0.5])
    comps = [Gaussian(loc=d.means[k], chol=d.chols[k]) for k in range(2)]
    w = np.asarray(jnp.exp(jax.nn.log_softmax(d.log_weights)))
    expected = np.log(sum(w[k] * np.exp(float(comps[k].log_prob(x))) for k in range(2)))
    assert np.isclose(float(d.log_prob(x)), expected, atol=1e-5)


def test_mixture_entropy_bounds_sandwich_monte_carlo():
    d = make_mixture()
    iv = d.entropy()
    x = d.sample(KEY, (200_000,))
    mc_entropy = float(-d.log_prob(x).mean())
    assert float(iv.lower) - 0.02 <= mc_entropy <= float(iv.upper) + 0.02
    assert float(iv.width) <= np.log(2) + 1e-6  # KT gap ≤ H(w) ≤ log K


def test_mixture_bounds_exact_for_single_component():
    g = make_gaussian()
    d = MixtureGaussian(log_weights=jnp.zeros(1), means=g.loc[None], chols=g.chol[None])
    iv = d.entropy()
    assert np.isclose(float(iv.width), 0.0, atol=1e-5)
    assert np.isclose(float(iv.lower), float(g.entropy().lower), atol=1e-5)


def test_mixture_bounds_exact_for_identical_components():
    g = make_gaussian()
    d = MixtureGaussian(
        log_weights=jnp.log(jnp.asarray([0.3, 0.7])),
        means=jnp.stack([g.loc, g.loc]),
        chols=jnp.stack([g.chol, g.chol]),
    )
    iv = d.entropy()
    assert np.isclose(float(iv.lower), float(g.entropy().lower), atol=1e-5)
    assert np.isclose(float(iv.upper), float(g.entropy().lower), atol=1e-5)


def test_mixture_bounds_exact_when_well_separated():
    d = MixtureGaussian(
        log_weights=jnp.log(jnp.asarray([0.4, 0.6])),
        means=jnp.asarray([[-50.0, 0.0], [50.0, 0.0]]),
        chols=jnp.stack([jnp.eye(2), jnp.eye(2)]),
    )
    iv = d.entropy()
    w = np.asarray([0.4, 0.6])
    comp_h = np.asarray(d.components.entropy().lower)
    expected = (w * comp_h).sum() + -(w * np.log(w)).sum()
    assert np.isclose(float(iv.lower), expected, atol=1e-4)
    assert np.isclose(float(iv.upper), expected, atol=1e-4)


def test_moment_match_preserves_mean_and_cov():
    d = make_mixture()
    g = d.moment_match()
    assert jnp.allclose(g.loc, d.mean(), atol=1e-6)
    assert jnp.allclose(g.cov, d.cov(), atol=1e-5)


def test_mixture_entropy_is_differentiable():
    d = make_mixture()

    def loss(means):
        m = MixtureGaussian(log_weights=d.log_weights, means=means, chols=d.chols)
        return m.entropy().midpoint

    grads = jax.grad(loss)(d.means)
    assert jnp.all(jnp.isfinite(grads))
