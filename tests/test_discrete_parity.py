"""Workstream B tests: Dirichlet contract + discrete-engine parity against pymdp.

Contract tests mirror the generic pattern of ``tests/test_distributions.py`` but
live here (parallel, additive) to avoid touching workstream A's file. Parity
tests feed identical inputs to pymdp's JAX reference (``run_factorized_fpi``,
``update_obs_likelihood_dirichlet``) and to our ``fpi`` / Dirichlet updates.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pymdp.algos import run_factorized_fpi
from pymdp.learning import update_obs_likelihood_dirichlet
from pymdp.maths import MINVAL

from hybrid_aif import Categorical, ConditionalCategorical, Dirichlet
from hybrid_aif.inference import fpi
from hybrid_aif.learning import update_likelihood_dirichlet

KEY = jax.random.PRNGKey(0)


def make_dirichlet():
    return Dirichlet(concentration=jnp.asarray([2.0, 0.7, 4.5]))


# ------------------------------------------------------- dirichlet contract


def test_dirichlet_pytree_roundtrip():
    d = make_dirichlet()
    leaves, treedef = jax.tree_util.tree_flatten(d)
    d2 = jax.tree_util.tree_unflatten(treedef, leaves)
    x = d.sample(KEY)
    assert jnp.allclose(d.log_prob(x), d2.log_prob(x))


def test_dirichlet_jit_with_distribution_argument():
    d = make_dirichlet()
    x = d.sample(KEY)
    f = jax.jit(lambda dist, y: dist.log_prob(y))
    assert jnp.allclose(f(d, x), d.log_prob(x), atol=1e-6)


def test_dirichlet_vmap_lifts_unbatched_object():
    d = make_dirichlet()
    stacked = jax.tree_util.tree_map(lambda a: jnp.stack([a, a, a]), d)
    ent = jax.vmap(lambda dist: dist.entropy().lower)(stacked)
    assert ent.shape == (3,)
    assert jnp.allclose(ent, d.entropy().lower, atol=1e-6)


def test_dirichlet_batch_shape_property():
    d = make_dirichlet()
    assert d.batch_shape == ()
    stacked = jax.tree_util.tree_map(lambda a: jnp.stack([a, a]), d)
    assert stacked.batch_shape == (2,)


def test_dirichlet_natural_roundtrip():
    d = make_dirichlet()
    d2 = Dirichlet.from_natural(*d.natural())
    assert jnp.allclose(d.concentration, d2.concentration, atol=1e-6)


def test_dirichlet_mean_mode_expected_log():
    d = Dirichlet(concentration=jnp.asarray([2.0, 3.0, 5.0]))
    a = np.asarray(d.concentration)
    assert np.allclose(np.asarray(d.mean()), a / a.sum(), atol=1e-6)
    assert np.allclose(np.asarray(d.mode()), (a - 1) / (a.sum() - 3), atol=1e-6)
    x = d.sample(KEY, (200_000,))
    assert np.allclose(np.log(x).mean(axis=0), np.asarray(d.expected_log_params()), atol=0.01)


def test_dirichlet_entropy_exact_matches_monte_carlo():
    d = make_dirichlet()
    iv = d.entropy()
    assert float(iv.width) == 0.0
    x = d.sample(KEY, (200_000,))
    assert np.isclose(float(iv.lower), float(-d.log_prob(x).mean()), atol=0.02)


def test_dirichlet_kl_self_zero_and_matches_monte_carlo():
    d, e = make_dirichlet(), Dirichlet(concentration=jnp.asarray([1.0, 1.0, 1.0]))
    assert jnp.isclose(d.kl_to(d).lower, 0.0, atol=1e-6)
    x = d.sample(KEY, (200_000,))
    mc = float((d.log_prob(x) - e.log_prob(x)).mean())
    assert np.isclose(float(d.kl_to(e).lower), mc, atol=0.02)


def test_dirichlet_sample_mean_frequencies():
    d = make_dirichlet()
    x = d.sample(KEY, (100_000,))
    assert jnp.allclose(x.mean(axis=0), d.mean(), atol=0.01)


# ----------------------------------------------------------- fpi vs pymdp


def _random_model(seed):
    rng = np.random.default_rng(seed)
    num_factors = int(rng.integers(2, 4))
    sizes = [int(rng.integers(2, 5)) for _ in range(num_factors)]
    deps_multi = sorted(rng.choice(num_factors, size=min(2, num_factors), replace=False).tolist())
    deps_single = [int(rng.integers(num_factors))]
    A_deps = [deps_multi, deps_single]
    A, num_obs = [], []
    for deps in A_deps:
        n_o = int(rng.integers(2, 5))
        shape = (n_o, *[sizes[f] for f in deps])
        g = rng.gamma(1.0, size=shape)
        A.append(np.float32(g / g.sum(axis=0, keepdims=True)))
        num_obs.append(n_o)
    priors = []
    for s in sizes:
        g = rng.gamma(1.0, size=s)
        priors.append(np.float32(g / g.sum()))
    obs = [int(rng.integers(n_o)) for n_o in num_obs]
    return sizes, A_deps, A, num_obs, priors, obs


def _log_stable(p):
    return np.log(np.clip(p, float(MINVAL), None)).astype(np.float32)


def _run_both(A_deps, A, num_obs, priors, obs, num_iters=16):
    obs_onehot = [jax.nn.one_hot(o, n) for o, n in zip(obs, num_obs, strict=True)]
    qs_ref = run_factorized_fpi(
        [jnp.asarray(a) for a in A],
        obs_onehot,
        [jnp.asarray(p) for p in priors],
        A_deps,
        num_iter=num_iters,
    )
    ours = fpi(
        [
            ConditionalCategorical(log_table=jnp.asarray(_log_stable(a)), parent_ndims=len(deps))
            for a, deps in zip(A, A_deps, strict=True)
        ],
        A_deps,
        [Categorical(logits=jnp.asarray(_log_stable(p))) for p in priors],
        [jnp.asarray(o, dtype=jnp.int32) for o in obs],
        num_iters=num_iters,
    )
    return qs_ref, ours


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_fpi_parity_random_models(seed):
    _, A_deps, A, num_obs, priors, obs = _random_model(seed)
    qs_ref, ours = _run_both(A_deps, A, num_obs, priors, obs)
    for q_ref, q in zip(qs_ref, ours, strict=True):
        assert np.allclose(np.asarray(q.probs), np.asarray(q_ref), atol=1e-5)


def test_fpi_parity_onehot_observations_match_indices():
    _, A_deps, A, num_obs, priors, obs = _random_model(7)
    _, ours_idx = _run_both(A_deps, A, num_obs, priors, obs)
    ours_oh = fpi(
        [jnp.asarray(_log_stable(a)) for a in A],
        A_deps,
        [Categorical(logits=jnp.asarray(_log_stable(p))) for p in priors],
        [jax.nn.one_hot(o, n) for o, n in zip(obs, num_obs, strict=True)],
    )
    for q_idx, q_oh in zip(ours_idx, ours_oh, strict=True):
        assert np.allclose(np.asarray(q_oh.probs), np.asarray(q_idx.probs), atol=1e-6)


def test_fpi_parity_tmaze():
    pytest.importorskip("pymdp.envs")
    from pymdp.envs import TMaze

    env = TMaze(reward_condition=0)
    A = [np.asarray(a) for a in env.A]
    A_deps = env.A_dependencies
    num_obs = [a.shape[0] for a in A]
    # Uniform priors so both location and context posteriors are non-degenerate.
    priors = [np.full(a, 1.0 / a, dtype=np.float32) for a in (5, 2)]
    obs = [3, 0, 1]  # at cue location, no reward, cue says "left"
    qs_ref, ours = _run_both(A_deps, A, num_obs, priors, obs)
    for q_ref, q in zip(qs_ref, ours, strict=True):
        assert np.allclose(np.asarray(q.probs), np.asarray(q_ref), atol=1e-5)


# ------------------------------------------------------ learning vs pymdp


def test_dirichlet_update_parity():
    sizes, A_deps, A, num_obs, priors, obs = _random_model(11)
    rng = np.random.default_rng(101)
    pA = [np.float32(rng.gamma(2.0, size=a.shape) + 0.1) for a in A]
    _, posteriors = _run_both(A_deps, A, num_obs, priors, obs)
    qs = [q.probs for q in posteriors]

    qA_ref, _ = update_obs_likelihood_dirichlet(
        [jnp.asarray(p) for p in pA],
        [jnp.asarray(a) for a in A],
        [jnp.asarray([o]) for o in obs],  # leading time axis of length 1
        [q[None] for q in qs],
        A_dependencies=A_deps,
        categorical_obs=False,
        num_obs=num_obs,
        lr=1.0,
    )
    qA = update_likelihood_dirichlet(
        [jnp.asarray(p) for p in pA],
        [jnp.asarray(o, dtype=jnp.int32) for o in obs],
        posteriors,
        A_deps,
        lr=1.0,
    )
    for ours, ref in zip(qA, qA_ref, strict=True):
        assert np.allclose(np.asarray(ours), np.asarray(ref), atol=1e-5)


def test_dirichlet_update_accepts_onehot_and_lr():
    _, A_deps, A, num_obs, priors, obs = _random_model(13)
    pA = [jnp.asarray(a) + 1.0 for a in A]
    _, posteriors = _run_both(A_deps, A, num_obs, priors, obs)
    obs_idx = [jnp.asarray(o, dtype=jnp.int32) for o in obs]
    obs_oh = [jax.nn.one_hot(o, n) for o, n in zip(obs, num_obs, strict=True)]
    out_idx = update_likelihood_dirichlet(pA, obs_idx, posteriors, A_deps, lr=0.5)
    out_oh = update_likelihood_dirichlet(pA, obs_oh, posteriors, A_deps, lr=0.5)
    for a, b, p in zip(out_idx, out_oh, pA, strict=True):
        assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-6)
        assert np.allclose(np.asarray(a - p).sum(), 0.5, atol=1e-5)  # lr scales unit mass


# --------------------------------------------------------- jit/vmap smoke


def test_fpi_jit_vmap_over_batched_priors():
    _, A_deps, A, num_obs, priors, obs = _random_model(3)
    lls = [jnp.asarray(_log_stable(a)) for a in A]
    obs_arrs = [jnp.asarray(o, dtype=jnp.int32) for o in obs]
    prior_logits = tuple(jnp.asarray(_log_stable(p)) for p in priors)

    def run(logits_tuple):
        cats = [Categorical(logits=lg) for lg in logits_tuple]
        return tuple(q.probs for q in fpi(lls, A_deps, cats, obs_arrs))

    single = run(prior_logits)
    batched = jax.jit(jax.vmap(run))(tuple(jnp.stack([lg] * 3) for lg in prior_logits))
    for q_b, q_s in zip(batched, single, strict=True):
        assert q_b.shape == (3,) + q_s.shape
        assert np.allclose(np.asarray(q_b), np.asarray(q_s)[None], atol=1e-6)
