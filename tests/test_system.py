"""System integration tests: the v0.1 loop on the TMaze.

Parity strategy: every piece System composes (fpi, expected_states/obs, term
functions, policy machinery) is already pymdp-parity-tested at function level;
here we test the *composition* against a hand-rolled evaluation, plus the
classic epistemic behavior: cue-seeking appears with the info-gain term and
disappears without it.
"""

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_aif import Categorical, ConditionalCategorical
from hybrid_aif.objectives import total
from hybrid_aif.objectives.discrete import expected_obs, expected_states
from hybrid_aif.reference.envs.tmaze import CENTRE, CUE, MIDDLE, tmaze
from hybrid_aif.system import System, SystemParams

_EPS = 1e-16
KEY = jax.random.PRNGKey(0)


def _cc(table, parent_ndims):
    return ConditionalCategorical(
        log_table=jnp.log(jnp.clip(table, _EPS)), parent_ndims=parent_ndims
    )


def tmaze_system(policy_len=2, epistemic=True):
    env, p = tmaze(reward_probability=0.98, punishment_probability=0.98, cue_validity=0.98)
    system = System(
        A_deps=env.A_deps,
        B_deps=env.B_deps,
        controlled=(True, False),
        policy_len=policy_len,
        gamma=16.0,
        epistemic=epistemic,
        modality_names=("location_obs", "reward", "cue"),
        factor_names=("location", "context"),
    )
    #  the pymdp-shaped tables enter by position, through the converter
    params = SystemParams(
        likelihood=tuple(_cc(a, len(d)) for a, d in zip(p.likelihood, env.A_deps, strict=True)),
        transition=tuple(_cc(b, len(d) + 1) for b, d in zip(p.transition, env.B_deps, strict=True)),
        initial_state_prior=tuple(Categorical.from_probs(d) for d in p.initial_state_prior),
        obs_targets={"reward": Categorical(logits=jnp.asarray([0.0, 4.0, -6.0]))},
        state_targets={},
        terminal_targets={},
    )
    return system, system.from_positional("params", params)


def _initial_beliefs(system, params):
    obs = (jnp.asarray(CENTRE), jnp.asarray(0), jnp.asarray(0))
    return system.infer_states(params, system.from_positional("observations", obs))[0]


def test_infer_states_localizes_and_keeps_context_uncertain():
    system, params = tmaze_system()
    beliefs = _initial_beliefs(system, params)
    assert jnp.argmax(beliefs["location"].probs) == CENTRE
    assert np.isclose(float(beliefs["location"].probs[CENTRE]), 1.0, atol=1e-4)
    assert np.allclose(np.asarray(beliefs["context"].probs), [0.5, 0.5], atol=1e-4)


def test_cue_seeking_appears_with_info_gain_and_vanishes_without():
    system_ig, params = tmaze_system(epistemic=True)
    beliefs = _initial_beliefs(system_ig, params)
    q_pi, _ = system_ig.infer_policies(params, beliefs)
    best = system_ig.policies(params)[jnp.argmax(q_pi.probs)]
    assert int(best[0, 0]) == CUE  # epistemic drive: resolve context first

    system_no_ig = tmaze_system(epistemic=False)[0]
    q_pi2, _ = system_no_ig.infer_policies(params, beliefs)
    best2 = system_no_ig.policies(params)[jnp.argmax(q_pi2.probs)]
    assert int(best2[0, 0]) != CUE  # utility alone gives no reason to look


def test_infer_policies_matches_hand_rolled_evaluation():
    system, params = tmaze_system(policy_len=2)
    beliefs = _initial_beliefs(system, params)
    _, tree = system.infer_policies(params, beliefs)
    g = total(tree).midpoint

    policies = np.asarray(system.policies(params))
    for idx in (0, 7, len(policies) - 1):
        qs = system.to_positional("beliefs", beliefs)
        positional = system.to_positional("params", params)
        manual = 0.0
        for step in range(system.policy_len):
            action = jnp.asarray(policies[idx, step])
            #  the T-maze by hand: location reads its own control, context none (pymdp layout)
            per_factor = jnp.stack([action[0], jnp.int32(0)])
            qs = expected_states(qs, positional.transition, per_factor, ((0,), (1,)))
            q_obs = expected_obs(qs, positional.likelihood, system.A_deps)
            u = float(jnp.sum(q_obs[1].probs * params.obs_targets["reward"].log_probs))
            ig = 0.0
            for m, deps in enumerate(system.A_deps):
                from jax.scipy.special import xlogy

                probs = jnp.exp(positional.likelihood[m].log_probs)
                h = -xlogy(probs, probs).sum(axis=0)  # axes follow deps order
                for dep in deps:
                    h = jnp.tensordot(qs[dep].probs, h, axes=[[0], [0]])
                q_o = q_obs[m].probs
                h_pred = float(-xlogy(q_o, q_o).sum())
                ig += h_pred - float(h)
            manual += -(u + ig)  # contributions to G
        assert np.isclose(float(g[idx]), manual, atol=2e-3), f"policy {idx}"


def test_as_env_roundtrip_and_step():
    system, params = tmaze_system()
    env, env_params = system.as_env(params)
    obs, state = env.reset(KEY, env_params)
    assert int(obs["location_obs"]) == CENTRE
    obs2, state2 = env.step(KEY, state, {"a_location": jnp.asarray(CUE, jnp.int32)}, env_params)
    assert int(state2["location"]) == CUE  # centre-cue are connected
    assert int(obs2["cue"]) in (1, 2)  # cue modality reveals context at the cue location


def test_act_deterministic_and_jit():
    system, params = tmaze_system()
    beliefs = _initial_beliefs(system, params)
    a1, q_pi, tree = system.act(KEY, params, beliefs, selection="deterministic")
    a2, _, _ = system.act(jax.random.PRNGKey(99), params, beliefs, selection="deterministic")
    assert jnp.array_equal(a1["a_location"], a2["a_location"])
    assert a1["a_location"].dtype == jnp.int32
    assert int(a1["a_location"]) == CUE
    assert np.isclose(float(q_pi.probs.sum()), 1.0, atol=1e-5)

    f = jax.jit(lambda p, b: system.infer_policies(p, b)[0].logits)
    assert jnp.allclose(f(params, beliefs), q_pi.logits, atol=1e-5)


def test_update_empirical_prior_moves_belief():
    system, params = tmaze_system()
    beliefs = _initial_beliefs(system, params)
    priors = system.update_empirical_prior(
        params, beliefs, system.from_positional("actions", jnp.asarray([CUE], jnp.int32))
    )
    assert jnp.argmax(priors["location"].probs) == CUE
    stay = system.update_empirical_prior(
        params, beliefs, system.from_positional("actions", jnp.asarray([MIDDLE], jnp.int32))
    )
    assert jnp.argmax(stay["location"].probs) == MIDDLE


def test_full_tick_loop_reaches_reward():
    """Three ticks of perceive-act-update on the real env: cue, then correct arm."""
    system, params = tmaze_system(policy_len=3)
    env, env_params = system.as_env(params)
    key = jax.random.PRNGKey(7)
    k_reset, key = jax.random.split(key)
    obs, state = env.reset(k_reset, env_params)
    priors = None
    rewarded = False
    for _t in range(4):
        beliefs, _ = system.infer_states(params, obs, priors)
        k_act, k_env, key = tuple(jax.random.split(key, 3))
        action, _, _ = system.act(k_act, params, beliefs, selection="deterministic")
        priors = system.update_empirical_prior(params, beliefs, action)
        obs, state = env.step(k_env, state, action, env_params)
        if int(obs["reward"]) == 1:
            rewarded = True
    assert rewarded  # cue -> centre -> middle -> correct arm within 4 ticks


def test_info_gain_selects_at_variable_granularity():
    """Cue-seeking is driven by epistemic value about the *context* variable.

    The selector speaks variables, not channels (per-variable-infogain.md V1/V6):
    naming ``context`` mounts every channel whose dependency set involves it
    (reward and cue, not location), which is exactly the drive that makes the
    agent look at the cue. Naming ``location`` mounts all three channels here —
    every modality depends on the location factor — but each leaf is the
    channel's information about the location alone, with the context
    marginalised (V9), so the cue buys nothing and the drive to look is gone.
    """
    _, params = tmaze_system()

    def plan(selector):
        system = tmaze_system(epistemic=selector)[0]
        beliefs = _initial_beliefs(system, params)
        q_pi, tree = system.infer_policies(params, beliefs)
        return int(system.policies(params)[jnp.argmax(q_pi.probs)][0, 0]), tree

    first, tree = plan(("context",))
    assert set(tree["infogain"]) == {"reward", "cue"}  # channel-keyed leaves (V3)
    assert first == CUE  # context info gain alone preserves cue-seeking
    first_all, tree_all = plan(("location",))
    assert set(tree_all["infogain"]) == {"location_obs", "reward", "cue"}
    assert first_all != CUE  # location info gain does not pay for context information
    first_off, tree_off = plan(False)
    assert "infogain" not in tree_off
    assert first_off != CUE  # the pragmatic agent has no reason to look


def test_unknown_designation_or_toggle_names_fail_loudly():
    import pytest

    system, params = tmaze_system()
    beliefs = _initial_beliefs(system, params)
    import equinox as eqx

    bad_params = eqx.tree_at(
        lambda p: p.obs_targets,
        params,
        {"rewrad": params.obs_targets["reward"]},
        is_leaf=lambda x: isinstance(x, dict),
    )
    with pytest.raises(ValueError, match="rewrad"):
        system.infer_policies(bad_params, beliefs)

    bad_system = tmaze_system(epistemic=("context", "typo"))[0]
    with pytest.raises(ValueError, match="typo"):
        bad_system.infer_policies(params, beliefs)


def test_metrics_false_returns_none_with_matching_q_pi():
    """metrics=False takes the leaf-free scan-carry path in the enumeration
    planner; same G by linearity, so parity is to float reassociation
    (~1e-7), not bitwise."""
    system, params = tmaze_system()
    beliefs = _initial_beliefs(system, params)
    q1, tree = system.infer_policies(params, beliefs)
    q2, none = system.infer_policies(params, beliefs, metrics=False)
    assert none is None and tree is not None
    assert jnp.allclose(q1.logits, q2.logits, atol=1e-6)
    a, _, none2 = system.act(KEY, params, beliefs, selection="deterministic", metrics=False)
    assert none2 is None and int(a["a_location"]) == CUE


def test_metrics_false_parity_with_all_designation_kinds():
    """The scan-carry path must accumulate every term the tree path scores,
    including terminal targets folded once at the horizon end."""
    import dataclasses

    named_system, named_params = tmaze_system()
    context_target = Categorical(logits=jnp.asarray([3.0, -3.0]))
    params = dataclasses.replace(
        named_params,
        state_targets={"context": context_target},
        terminal_targets={"context": context_target},
    )
    for decomposition in ("utility_infogain", "risk_ambiguity"):
        system = dataclasses.replace(named_system, decomposition=decomposition)
        beliefs = _initial_beliefs(system, params)
        q1, tree = system.infer_policies(params, beliefs)
        q2, none = system.infer_policies(params, beliefs, metrics=False)
        assert none is None and "terminal_risk" in tree
        assert jnp.allclose(q1.logits, q2.logits, atol=1e-6), decomposition


def test_default_named_state_and_terminal_designations_reach_the_tree():
    """Regression: with default block names, ``_validate`` accepts ``factor_i`` keys,
    so the risk/terminal terms must look blocks up under the same prefix — a
    designation that validates must never be silently dropped from the tree."""
    import dataclasses

    named_system, named_params = tmaze_system()
    system = System(  # the same model declared afresh, with every name generated
        A_deps=named_system.A_deps,
        B_deps=((0,), (1,)),  # the T-maze's state parents, by hand
        controlled=(True, False),
        policy_len=named_system.policy_len,
        gamma=named_system.gamma,
        epistemic=named_system.epistemic,
    )
    context_target = Categorical(logits=jnp.asarray([3.0, -3.0]))
    #  the same tables under the generated names
    renamed = system.from_positional("params", named_system.to_positional("params", named_params))
    params = dataclasses.replace(
        renamed,
        obs_targets={"modality_1": named_params.obs_targets["reward"]},
        state_targets={"factor_1": context_target},
        terminal_targets={"factor_1": context_target},
    )
    beliefs = _initial_beliefs(system, params)
    _, tree = system.infer_policies(params, beliefs)
    assert "factor_1" in tree["risk"]
    assert "factor_1" in tree["terminal_risk"]


def test_a_memoryless_factor_ticks_without_a_transition_parent():
    """``B_deps=()``: the control axis is the table's only parent axis, and the slice is
    next tick's distribution. Found by a composer who had to declare the factor as its
    own parent with a flat table to say "no memory" (2026-09-18)."""
    levels = 3
    system = System(
        B_deps=((),),
        controlled=(False,),
        policy_len=2,
        A_deps=((0,),),
        factor_names=("s",),
        modality_names=("o",),
    )
    reactive = jnp.log(jnp.asarray([[0.2], [0.5], [0.3]]))  # (levels, control=1)
    params = SystemParams(
        transition={"s": ConditionalCategorical(reactive, 1)},
        initial_state_prior={"s": Categorical(jnp.zeros(levels))},
        likelihood={"o": ConditionalCategorical(jnp.log(jnp.eye(levels) * 0.9 + 0.05), 1)},
    )
    beliefs, _ = system.infer_states(params, {"o": jnp.asarray(1, jnp.int32)}, None)
    priors = system.update_empirical_prior(
        params, beliefs, system.from_positional("actions", jnp.zeros(0, jnp.int32))
    )
    np.testing.assert_allclose(np.asarray(priors["s"].probs), [0.2, 0.5, 0.3], atol=1e-6)
    (two_steps,) = system.predict_beliefs(params, beliefs, jnp.zeros((1, 0), jnp.int32))
    np.testing.assert_allclose(np.asarray(two_steps["s"].probs), [0.2, 0.5, 0.3], atol=1e-6)
    #  and the whole loop runs, act included
    action, _, _ = system.act(jax.random.PRNGKey(0), params, priors, selection="deterministic")
    assert system.to_positional("actions", action).shape == (0,)  # no control node
