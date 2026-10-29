"""Workstream E tests: enumeration planner parity vs pymdp + evaluation correctness.

Parity tests exercise OUR machinery against pymdp's reference (policy
construction, posterior composition from their neg-EFE values, first-step
action marginals) without depending on workstream D's term implementations:
step evaluators here are simple inline callbacks.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pymdp.control import construct_policies, get_marginals, update_posterior_policies

from hybrid_aif import Categorical
from hybrid_aif.interval import Interval
from hybrid_aif.objectives.base import total
from hybrid_aif.planning import (
    action_marginals,
    enumerate_policies,
    evaluate_policies,
    policy_posterior,
    sample_action,
)

KEY = jax.random.PRNGKey(0)


def _normalize(x, axis=0):
    return x / x.sum(axis=axis, keepdims=True)


# ------------------------------------------------------------ policy space


@pytest.mark.parametrize(
    "num_controls,policy_len",
    [((2, 3), 1), ((3,), 3), ((2, 2), 2), ((4, 1, 2), 2)],
)
def test_policy_enumeration_matches_pymdp(num_controls, policy_len):
    ours = enumerate_policies(num_controls, policy_len)
    ref = construct_policies(list(num_controls), list(num_controls), policy_len)
    assert ours.dtype == jnp.int32
    assert ours.shape == ref.shape
    assert np.array_equal(np.asarray(ours), np.asarray(ref))


def test_uncontrolled_factor_pinned_to_noop():
    ours = enumerate_policies((3, 4), policy_len=2, controlled=(True, False))
    ref = construct_policies([3, 4], [3, 1], 2)
    assert ours.shape == (9, 2, 2)
    assert np.all(np.asarray(ours)[:, :, 1] == 0)
    assert np.array_equal(np.asarray(ours), np.asarray(ref))


# ------------------------------------------------------- posterior parity


def _random_model(key):
    ks = jax.random.split(key, 8)
    A = [
        _normalize(jax.random.uniform(ks[0], (2, 3))),
        _normalize(jax.random.uniform(ks[1], (3, 3, 2))),
    ]
    B = [
        _normalize(jax.random.uniform(ks[2], (3, 3, 2))),
        _normalize(jax.random.uniform(ks[3], (2, 2, 1))),
    ]
    C = [jax.random.normal(ks[4], (2,)), jax.random.normal(ks[5], (3,))]
    qs = [_normalize(jax.random.uniform(ks[6], (3,))), _normalize(jax.random.uniform(ks[7], (2,)))]
    return A, B, C, qs


@pytest.mark.parametrize("gamma,uniform_E", [(16.0, True), (4.0, False)])
def test_policy_posterior_parity_with_pymdp(gamma, uniform_E):
    A, B, C, qs = _random_model(jax.random.PRNGKey(7))
    policies = enumerate_policies((2, 1), policy_len=2)
    n = policies.shape[0]
    E = jnp.ones(n) / n if uniform_E else _normalize(jax.random.uniform(KEY, (n,)))
    q_pi_ref, neg_efe = update_posterior_policies(
        policies, qs, A, B, C, E, None, None, [[0], [0, 1]], [[0], [1]], gamma=gamma
    )
    ours = policy_posterior(Interval.exact(-neg_efe), gamma, policy_prior=jnp.log(E))
    assert jnp.allclose(ours.probs, q_pi_ref, atol=1e-6)


def test_policy_posterior_accepts_categorical_habit_and_bounds():
    G = Interval(jnp.asarray([0.0, 1.0]), jnp.asarray([2.0, 1.0]))
    E = Categorical.from_probs(jnp.asarray([0.75, 0.25]))
    q = policy_posterior(G, 1.0, policy_prior=E)
    manual = jax.nn.softmax(-G.midpoint + E.log_probs)
    assert jnp.allclose(q.probs, manual, atol=1e-6)
    lo = policy_posterior(G, 1.0, use_bound="lower")
    hi = policy_posterior(G, 1.0, use_bound="upper")
    assert jnp.allclose(lo.probs, jax.nn.softmax(-G.lower), atol=1e-6)
    assert jnp.allclose(hi.probs, jax.nn.softmax(-G.upper), atol=1e-6)


# ------------------------------------------------- evaluate_policies exact


def _two_factor_setup(key):
    ks = jax.random.split(key, 4)
    B0 = _normalize(jax.random.uniform(ks[0], (2, 2, 2)))
    B1 = _normalize(jax.random.uniform(ks[1], (2, 2, 2)))
    A = _normalize(jax.random.uniform(ks[2], (3, 2, 2)))
    C = jax.random.normal(ks[3], (3,))
    q0 = (
        Categorical.from_probs(jnp.asarray([0.6, 0.4])),
        Categorical.from_probs(jnp.asarray([0.3, 0.7])),
    )

    def transition_fn(qs, action):
        p0 = B0[:, :, action[0]] @ qs[0].probs
        p1 = B1[:, :, action[1]] @ qs[1].probs
        return (Categorical.from_probs(p0), Categorical.from_probs(p1))

    def step_terms(qs, action, t):
        qo = jnp.einsum("oij,i,j->o", A, qs[0].probs, qs[1].probs)
        return {"utility": {"obs": Interval.exact(jnp.sum(qo * C))}}

    return B0, B1, A, C, q0, transition_fn, step_terms


def test_evaluate_policies_matches_numpy_enumeration():
    B0, B1, A, C, q0, transition_fn, step_terms = _two_factor_setup(jax.random.PRNGKey(3))
    policy_len = 2
    policies = enumerate_policies((2, 2), policy_len)
    tree = evaluate_policies(policies, q0, transition_fn, step_terms)

    B0n, B1n, An, Cn = (np.asarray(x) for x in (B0, B1, A, C))
    expected = []
    for pol in itertools.product(range(2), range(2), repeat=policy_len):
        acts = np.asarray(pol).reshape(policy_len, 2)
        p0, p1 = np.asarray(q0[0].probs), np.asarray(q0[1].probs)
        util = 0.0
        for t in range(policy_len):
            p0 = B0n[:, :, acts[t, 0]] @ p0
            p1 = B1n[:, :, acts[t, 1]] @ p1
            qo = np.einsum("oij,i,j->o", An, p0, p1)
            util += float(np.sum(qo * Cn))
        expected.append(util)
    expected = np.asarray(expected, dtype=np.float32)

    leaf = tree["utility"]["obs"]
    assert leaf.lower.shape == (policies.shape[0],)
    assert np.allclose(np.asarray(leaf.lower), expected, atol=1e-5)
    assert np.allclose(np.asarray(leaf.upper), expected, atol=1e-5)
    # G folds utility with sign -1
    assert np.allclose(np.asarray(total(tree).midpoint), -expected, atol=1e-5)


def test_interval_bounds_accumulate_across_steps():
    _, _, _, _, q0, transition_fn, _ = _two_factor_setup(jax.random.PRNGKey(5))
    policy_len = 3
    policies = enumerate_policies((2, 2), policy_len)

    def step_terms(qs, action, t):
        v = qs[0].probs[0] + 0.5 * action[0].astype(jnp.float32)
        return {"risk": {"blk": Interval(v - 0.1, v + 0.3)}}

    def terminal_terms(qs_final):
        return {
            "risk": {"blk": Interval.exact(qs_final[1].probs[0])},
            "utility": {"obs": Interval.exact(jnp.asarray(1.0))},
        }

    tree = evaluate_policies(policies, q0, transition_fn, step_terms, terminal_terms)
    risk = tree["risk"]["blk"]
    assert np.allclose(np.asarray(risk.width), policy_len * 0.4, atol=1e-5)
    assert np.all(np.asarray(risk.lower) <= np.asarray(risk.upper))
    # terminal-only term present once, zero width, per-policy axis
    util = tree["utility"]["obs"]
    assert util.lower.shape == (policies.shape[0],)
    assert np.allclose(np.asarray(util.width), 0.0)
    assert np.allclose(np.asarray(util.lower), 1.0)
    # bounds equal step-wise sums plus terminal midpoint contribution
    no_term = evaluate_policies(policies, q0, transition_fn, step_terms)
    step_risk = no_term["risk"]["blk"]
    term_v = np.asarray(risk.lower) - np.asarray(step_risk.lower)
    assert np.allclose(term_v, np.asarray(risk.upper) - np.asarray(step_risk.upper), atol=1e-6)


# ----------------------------------------------------- marginals + sampling


def test_action_marginals_match_pymdp():
    policies = enumerate_policies((3, 2), policy_len=2)
    n = policies.shape[0]
    q_pi = _normalize(jax.random.uniform(KEY, (n,)))
    ours = action_marginals(Categorical.from_probs(q_pi), policies, (3, 2))
    ref = get_marginals(q_pi, policies, [3, 2])
    for m_ours, m_ref in zip(ours, ref, strict=True):
        assert jnp.allclose(m_ours.probs, m_ref, atol=1e-6)


def test_sample_action_modes_and_dtype():
    policies = enumerate_policies((3, 2), policy_len=2)
    n = policies.shape[0]
    q_pi = Categorical.from_probs(_normalize(jax.random.uniform(jax.random.PRNGKey(2), (n,))))
    key = jax.random.PRNGKey(11)

    a1 = sample_action(key, q_pi, policies, (3, 2), mode="marginal")
    a2 = sample_action(key, q_pi, policies, (3, 2), mode="marginal")
    assert a1.dtype == jnp.int32 and a1.shape == (2,)
    assert np.array_equal(np.asarray(a1), np.asarray(a2))

    marginals = action_marginals(q_pi, policies, (3, 2))
    a3 = sample_action(key, marginals, mode="marginal")
    assert np.array_equal(np.asarray(a3), np.asarray(a1))

    a4 = sample_action(key, q_pi, policies, mode="policy")
    assert a4.dtype == jnp.int32 and a4.shape == (2,)
    first_rows = np.asarray(policies[:, 0, :])
    assert any(np.array_equal(np.asarray(a4), row) for row in first_rows)


def test_deterministic_habit_prior_dominates():
    G = Interval.exact(jnp.zeros(4))
    policy_prior = jnp.log(jnp.asarray([1e-8, 1e-8, 1.0, 1e-8]))
    q_pi = policy_posterior(G, 16.0, policy_prior=policy_prior)
    assert int(jnp.argmax(q_pi.probs)) == 2


# ------------------------------------------------------------- jit + vmap


def test_full_pipeline_jits_and_vmaps_over_beliefs():
    B0, B1, A, C, _, transition_fn, step_terms = _two_factor_setup(jax.random.PRNGKey(9))
    policies = enumerate_policies((2, 2), policy_len=2)

    def pipeline(logits0, logits1, key):
        init_qs = (Categorical(logits=logits0), Categorical(logits=logits1))
        tree = evaluate_policies(policies, init_qs, transition_fn, step_terms)
        q_pi = policy_posterior(total(tree), gamma=4.0)
        action = sample_action(key, q_pi, policies, (2, 2), mode="marginal")
        return q_pi.probs, action

    l0 = jax.random.normal(jax.random.PRNGKey(21), (3, 2))
    l1 = jax.random.normal(jax.random.PRNGKey(22), (3, 2))
    keys = jax.random.split(jax.random.PRNGKey(23), 3)

    batched = jax.jit(jax.vmap(pipeline))
    q_probs, actions = batched(l0, l1, keys)
    assert q_probs.shape == (3, policies.shape[0])
    assert jnp.allclose(q_probs.sum(-1), 1.0, atol=1e-5)
    assert actions.shape == (3, 2) and actions.dtype == jnp.int32

    q_single, a_single = jax.jit(pipeline)(l0[0], l1[0], keys[0])
    assert jnp.allclose(q_single, q_probs[0], atol=1e-6)
    assert np.array_equal(np.asarray(a_single), np.asarray(actions[0]))


def test_deterministic_selection_is_argmax_and_key_free():
    marginals = (
        Categorical(logits=jnp.log(jnp.asarray([0.1, 0.7, 0.2]))),
        Categorical(logits=jnp.log(jnp.asarray([0.9, 0.1]))),
    )
    a1 = sample_action(jax.random.PRNGKey(0), marginals, selection="deterministic")
    a2 = sample_action(jax.random.PRNGKey(999), marginals, selection="deterministic")
    assert jnp.array_equal(a1, a2)
    assert jnp.array_equal(a1, jnp.asarray([1, 0], dtype=jnp.int32))
    assert a1.dtype == jnp.int32


def test_alpha_sharpens_marginal_sampling():
    marginals = (Categorical(logits=jnp.log(jnp.asarray([0.4, 0.6]))),)
    keys = jax.random.split(jax.random.PRNGKey(3), 4000)
    draws_sharp = jnp.stack([sample_action(k, marginals, alpha=32.0) for k in keys[:2000]])
    draws_raw = jnp.stack([sample_action(k, marginals) for k in keys[2000:]])
    assert draws_sharp.mean() > 0.95  # alpha=32 makes the 0.6 mode near-certain
    assert 0.5 < draws_raw.mean() < 0.7  # raw sampling tracks the marginal
