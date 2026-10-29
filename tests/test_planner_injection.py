"""Planner injection: default-path parity, injected callables, registry, use_bound.

The regression guard rebuilds the pre-refactor pipeline (the closures that
lived inline in ``System.infer_policies`` composed directly with the
enumeration functions) and requires matching ``q_pi`` and term tree from the
default path on the TMaze fixture. Since the compiled prob-space path
(``objectives/compiled.py``) became the default, parity is to float
tolerance, not bitwise: the compiled path evaluates the same math over raw
probability arrays without the reference's per-step ``softmax(log(...))``
renormalization. Exhaustive leaf-level compiled-vs-reference parity lives in
``test_compiled_parity.py``.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import Categorical
from hybrid_aif.interval import Interval
from hybrid_aif.objectives import total
from hybrid_aif.objectives.discrete import (
    expected_obs,
    expected_states,
    state_info_gain,
    utility,
)
from hybrid_aif.planning import (
    enumerate_policies,
    enumeration_planner,
    evaluate_policies,
    get_planner,
    policy_posterior,
    register_planner,
)
from tests.test_system import _initial_beliefs, tmaze_system

KEY = jax.random.PRNGKey(0)


def _legacy_enumeration(system, params, beliefs, use_bound="midpoint"):
    params = system.to_positional("params", params)  # the reference indexes by position

    def transition(qs, action):
        #  the T-maze by hand: location reads its own control, context none (pymdp layout)
        per_factor = jnp.stack([action[0], jnp.int32(0)])
        return expected_states(qs, params.transition, per_factor, ((0,), (1,)))

    def step_terms(qs, action, t):
        del action, t
        q_obs = expected_obs(qs, params.likelihood, system.A_deps)
        return {
            **utility(q_obs, params.obs_targets, system.modality_names),
            **state_info_gain(qs, params.likelihood, system.A_deps, system.modality_names),
        }

    policies = enumerate_policies((5,), system.policy_len, (True,))  # one column: the location node
    beliefs = system.to_positional("beliefs", beliefs)
    tree = evaluate_policies(policies, beliefs, transition, step_terms, None)
    q_pi = policy_posterior(total(tree), system.gamma, params.policy_prior, use_bound=use_bound)
    return q_pi, tree


def _uniform3(ctx, use_bound="midpoint", metrics=True):
    first = jnp.asarray([1, 2, 3], jnp.int32)
    policies = jnp.zeros((3, ctx.policy_len, len(ctx.num_controls)), jnp.int32)
    policies = policies.at[:, :, 0].set(first[:, None])
    tree = {"risk": {"toy": Interval.exact(jnp.zeros(3))}}
    return Categorical(logits=jnp.zeros(3)), tree, policies


def test_default_path_matches_pre_refactor_pipeline():
    for policy_len in (1, 2):
        system, params = tmaze_system(policy_len=policy_len)
        beliefs = _initial_beliefs(system, params)
        q_pi, tree = system.infer_policies(params, beliefs)
        ref_q_pi, ref_tree = _legacy_enumeration(system, params, beliefs)
        assert jnp.allclose(q_pi.logits, ref_q_pi.logits, atol=1e-5)
        assert {t: set(b) for t, b in tree.items()} == {t: set(b) for t, b in ref_tree.items()}
        for term, blocks in ref_tree.items():
            for block, ref_iv in blocks.items():
                got = tree[term][block]
                assert jnp.allclose(got.lower, ref_iv.lower, atol=1e-5), (term, block)
                assert jnp.allclose(got.upper, ref_iv.upper, atol=1e-5), (term, block)


def test_injected_callable_flows_through_infer_policies_and_act():
    system, params = tmaze_system()
    beliefs = _initial_beliefs(system, params)

    q_pi, tree = system.infer_policies(params, beliefs, planner=_uniform3)
    assert q_pi.probs.shape == (3,)
    assert np.allclose(np.asarray(q_pi.probs), 1 / 3, atol=1e-6)
    assert set(tree) == {"risk"} and set(tree["risk"]) == {"toy"}

    action, q_pi2, tree2 = system.act(KEY, params, beliefs, planner=_uniform3)
    action = system.to_positional("actions", action)
    assert action.dtype == jnp.int32 and action.shape == (1,)
    assert int(action[0]) in (1, 2, 3)
    assert set(tree2["risk"]) == {"toy"}

    det, _, _ = system.act(KEY, params, beliefs, selection="deterministic", planner=_uniform3)
    assert int(det["a_location"]) == 1

    pol, _, _ = system.act(KEY, params, beliefs, mode="policy", planner=_uniform3)
    assert int(pol["a_location"]) in (1, 2, 3)


def test_registry_register_resolve_and_unknown_name_raises():
    assert get_planner("enumeration") is enumeration_planner

    @register_planner("uniform3_registered")
    def toy(ctx, use_bound="midpoint", metrics=True):
        return _uniform3(ctx, use_bound, metrics)

    assert get_planner("uniform3_registered") is toy
    with pytest.raises(KeyError, match="no_such_planner"):
        get_planner("no_such_planner")
    with pytest.raises(KeyError, match="enumeration"):
        get_planner("no_such_planner")


def test_system_planner_name_resolves_via_registry():
    register_planner("uniform3_named")(_uniform3)
    system, params = tmaze_system()
    assert system.planner == "enumeration"

    named = dataclasses.replace(system, planner="uniform3_named")
    beliefs = _initial_beliefs(named, params)
    q_pi, _ = named.infer_policies(params, beliefs)
    assert q_pi.probs.shape == (3,)
    action, _, _ = named.act(KEY, params, beliefs)
    assert int(action["a_location"]) in (1, 2, 3)

    bogus = dataclasses.replace(system, planner="never_registered")
    with pytest.raises(KeyError, match="never_registered"):
        bogus.infer_policies(params, beliefs)


def test_use_bound_threads_through_act_and_infer_policies():
    seen = []

    def recorder(ctx, use_bound="midpoint", metrics=True):
        seen.append(use_bound)
        return _uniform3(ctx, use_bound, metrics)

    system, params = tmaze_system()
    beliefs = _initial_beliefs(system, params)
    system.act(KEY, params, beliefs, planner=recorder, use_bound="upper")
    system.infer_policies(params, beliefs, planner=recorder, use_bound="lower")
    assert seen == ["upper", "lower"]

    _, q_pi, _ = system.act(KEY, params, beliefs, use_bound="lower")
    ref_q_pi, _ = _legacy_enumeration(system, params, beliefs, use_bound="lower")
    assert jnp.allclose(q_pi.logits, ref_q_pi.logits, atol=1e-5)


def test_default_path_jit_smoke():
    system, params = tmaze_system()
    beliefs = _initial_beliefs(system, params)
    eager, _ = system.infer_policies(params, beliefs)

    f = jax.jit(lambda p, b: system.infer_policies(p, b)[0].logits)
    assert jnp.allclose(f(params, beliefs), eager.logits, atol=1e-6)

    g = jax.jit(lambda k, p, b: system.act(k, p, b)[0])
    action = system.to_positional("actions", g(KEY, params, beliefs))
    assert action.dtype == jnp.int32 and action.shape == (1,)
