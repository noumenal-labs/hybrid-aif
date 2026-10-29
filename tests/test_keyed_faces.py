"""Names at the faces (control-states.md C10): keyed beliefs, observations and actions.

The converter round-trips, the sentences a missing or unknown key and a positional input raise,
and the name-keyed rollout of an env with a continuous (LG) modality.
"""

import dataclasses
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import InverseGamma
from hybrid_aif.envs import Env, GenerativeEnv, broadcast_params
from hybrid_aif.replay import replay
from hybrid_aif.rollout import random_policy, rollout
from tests.test_cem import lq_system
from tests.test_system import tmaze_system

KEY = jax.random.PRNGKey(0)
OBS = {"location_obs": jnp.asarray(2), "reward": jnp.asarray(0), "cue": jnp.asarray(0)}


def _example(name: str):
    path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_keyed_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------- converters


def test_belief_and_observation_round_trips_keep_values_and_node_order():
    system, params = tmaze_system()
    beliefs, _ = system.infer_states(params, OBS)
    assert [k for k in beliefs if k in system.names.states] == ["location", "context"]
    positional = system.to_positional("beliefs", beliefs)
    assert isinstance(positional, tuple) and positional[0] is beliefs["location"]
    assert system.from_positional("beliefs", positional) == {
        k: beliefs[k] for k in system.names.states
    }
    obs = system.to_positional("observations", OBS)
    assert obs == (OBS["location_obs"], OBS["reward"], OBS["cue"])
    assert list(system.from_positional("observations", obs)) == ["location_obs", "reward", "cue"]


def test_action_round_trip_is_one_column_per_control_node():
    system, _ = tmaze_system()
    positional = jnp.asarray([3], jnp.int32)
    action = system.from_positional("actions", positional)
    assert list(action) == ["a_location"]
    assert int(action["a_location"]) == 3
    back = system.to_positional("actions", action)
    assert back.dtype == jnp.int32 and np.array_equal(back, positional)
    #  leading axes ride along
    batched = system.from_positional("actions", jnp.asarray([[1], [4]], jnp.int32))
    assert np.array_equal(batched["a_location"], [1, 4])
    assert np.array_equal(system.to_positional("actions", batched), [[1], [4]])


def test_from_positional_refuses_a_per_factor_width_action():
    system, _ = tmaze_system()
    with pytest.raises(ValueError, match="one level per control node"):
        system.from_positional("actions", jnp.asarray([3, 1], jnp.int32))


def test_continuous_actions_round_trip_as_their_blocks():
    system, params = lq_system()
    flat = jnp.asarray([0.25], jnp.float32)
    action = system.from_positional("actions", flat, params)
    assert list(action) == ["a_x"]
    assert np.array_equal(action["a_x"], flat)
    assert np.array_equal(system.to_positional("actions", action, params), flat)
    #  without the params the node's box is unknown, and a float value is refused
    with pytest.raises(ValueError, match="pass params="):
        system.from_positional("actions", flat)


def test_act_and_update_empirical_prior_speak_keyed_actions():
    system, params = tmaze_system()
    beliefs, _ = system.infer_states(params, OBS)
    action, _, _ = system.act(KEY, params, beliefs, selection="deterministic")
    assert list(action) == ["a_location"] and action["a_location"].dtype == jnp.int32
    pushed = system.update_empirical_prior(params, beliefs, action)
    assert [k for k in pushed if k in system.names.states] == ["location", "context"]
    marginals = system.action_marginals(params, system.infer_policies(params, beliefs)[0])
    assert list(marginals) == ["u_location"]
    assert marginals["u_location"].probs.shape == (5,)


def test_the_unknown_kind_is_refused():
    system, _ = tmaze_system()
    with pytest.raises(ValueError, match="kind must be one of"):
        system.to_positional("policies", {})


# ---------------------------------------------------------------------------- the sentences


def test_a_tuple_is_refused_with_a_sentence_naming_the_converter():
    system, params = tmaze_system()
    with pytest.raises(TypeError, match=r"from_positional\('observations', \.\.\.\)"):
        system.infer_states(params, (2, 0, 0))
    beliefs, _ = system.infer_states(params, OBS)
    positional = system.to_positional("beliefs", beliefs)
    with pytest.raises(TypeError, match=r"from_positional\('beliefs', \.\.\.\)"):
        system.infer_policies(params, positional)
    with pytest.raises(TypeError, match=r"from_positional\('actions', \.\.\.\)"):
        system.update_empirical_prior(params, beliefs, jnp.asarray([3, 0], jnp.int32))


def test_missing_and_unknown_keys_are_named_with_the_expected_ones():
    system, params = tmaze_system()
    with pytest.raises(ValueError) as info:
        system.infer_states(params, {"location_obs": 2, "reward": 0, "cues": 0})
    assert str(info.value) == (
        "infer_states: observations are missing ['cue'] and have unknown key ['cues']; "
        "expected keys ['location_obs', 'reward', 'cue']"
    )
    beliefs, _ = system.infer_states(params, OBS)
    with pytest.raises(ValueError, match=r"unknown key \['a_context'\].*\['a_location'\]"):
        system.update_empirical_prior(params, beliefs, {"a_location": 1, "a_context": 0})
    with pytest.raises(ValueError, match=r"missing \['context'\]"):
        system.act(KEY, params, {"location": beliefs["location"]})


def test_an_unobserved_modality_is_a_keyed_present_entry():
    system, params = tmaze_system()
    present = {"location_obs": True, "reward": True, "cue": jnp.asarray(False)}
    #  an absent modality's value is ignored, so two different cues give one posterior
    a, _ = system.infer_states(params, {**OBS, "cue": jnp.asarray(1)}, present=present)
    b, _ = system.infer_states(params, {**OBS, "cue": jnp.asarray(2)}, present=present)
    assert np.array_equal(a["context"].probs, b["context"].probs)
    #  partial: a modality left out is present, so marking only the absent one is the
    #  complete spelling of the same thing
    seen = {"location_obs": jnp.asarray(3), "reward": jnp.asarray(0), "cue": jnp.asarray(1)}
    flat = _flat_priors()
    complete, _ = system.infer_states(
        params, seen, flat, present={"location_obs": True, "reward": True, "cue": False}
    )
    partial, _ = system.infer_states(params, seen, flat, present={"cue": False})
    full, _ = system.infer_states(params, seen, flat)
    for name in ("location", "context"):
        assert np.array_equal(partial[name].probs, complete[name].probs)
    assert not np.allclose(partial["context"].probs, full["context"].probs)
    with pytest.raises(ValueError, match=r"present has unknown key \['cues'\]"):
        system.infer_states(params, OBS, present={"cues": False})
    with pytest.raises(TypeError, match=r"from_positional\('observations', \.\.\.\)"):
        system.infer_states(params, OBS, present=(True, True, False))
    #  a complete positional tuple of flags has the observations' structure: it converts so
    converted = system.from_positional("observations", (True, True, jnp.asarray(False)))
    c, _ = system.infer_states(params, {**OBS, "cue": jnp.asarray(1)}, present=converted)
    assert np.array_equal(c["context"].probs, a["context"].probs)


def test_replay_takes_and_returns_keyed_traces():
    system, params = tmaze_system()
    obs = {k: jnp.stack([v, v]) for k, v in OBS.items()}
    actions = {"a_location": jnp.asarray([3, 0], jnp.int32)}
    out = replay(system, params, obs, actions, present={k: jnp.ones(2, bool) for k in OBS})
    assert [k for k in out.beliefs if k in system.names.states] == ["location", "context"]
    assert list(out.chosen) == ["a_location"] and out.chosen["a_location"].shape == (2,)
    assert list(out.exercised) == ["location_obs", "reward", "cue"]
    assert list(out.unsupported) == ["location", "context"]
    with pytest.raises(ValueError, match=r"actions are missing \['a_location'\]"):
        replay(system, params, obs, {})


# ------------------------------------------------------------------------------ LG rollouts


def test_as_env_speaks_names_and_rolls_out_a_continuous_modality():
    system, params = _example("hill_lg").make_agent()
    env, env_params = system.as_env(params)
    assert isinstance(env, GenerativeEnv)
    obs, state = env.reset(KEY, env_params, state={"state": jnp.zeros(2)})
    assert list(obs) == ["position"] and obs["position"].shape == (1,)
    assert list(state) == ["state"]

    B, T, W = 2, 5, 3
    batched = broadcast_params(env_params, B)
    traj = rollout(KEY, env, batched, random_policy(system, params), T, B, window_len=W)
    assert traj.observations["position"].shape == (B, T + 1, 1)
    assert traj.observations["position"].dtype == jnp.float32
    assert traj.states["state"].shape == (B, T + 1, 2)
    assert traj.actions["a_state"].shape == (B, T) and traj.actions["a_state"].dtype == jnp.int32
    hist = traj.history
    assert hist.obs["position"].shape == (B, T, W, 1)
    assert hist.actions["a_state"].shape == (B, T, W)
    #  the window holds each tick's observation, right-aligned, with -1 sentinels to its left
    obs_seq, valid = np.asarray(traj.observations["position"]), np.asarray(hist.valid)
    window = np.asarray(hist.obs["position"])
    for t in range(T):
        for i in range(W):
            src = t - (W - 1 - i)
            if src >= 0:
                assert np.array_equal(window[:, t, i], obs_seq[:, src])
            else:
                assert np.all(window[:, t, i] == -1.0) and not valid[:, t, i].any()


def test_a_mixed_categorical_and_lg_env_rolls_out_with_each_dtype():
    system, params = _example("switching_filter").make_agent()
    env, env_params = system.as_env(params)
    traj = rollout(KEY, env, broadcast_params(env_params, 2), random_policy(system, params), 3, 2)
    assert traj.observations["y"].dtype == jnp.float32 and traj.observations["y"].shape == (2, 4, 1)
    assert traj.states["mode"].dtype == jnp.int32 and traj.states["mode"].shape == (2, 4)
    assert traj.states["x"].shape == (2, 4, 1)
    assert list(traj.actions) == ["a_x"]


def test_random_policy_on_a_system_draws_what_the_positional_one_draws():
    system, params = tmaze_system()
    k = jax.random.PRNGKey(3)
    keyed_draw = random_policy(system, params)(k, None, 0)
    positional = random_policy((5,))(k, None, 0)
    assert np.array_equal(system.to_positional("actions", keyed_draw), positional)
    with pytest.raises(ValueError, match="continuous control nodes"):
        random_policy(lq_system()[0], lq_system()[1])


# ------------------------------------------------------------------------ more refusals


def test_float_levels_are_refused_at_every_face_rather_than_truncated():
    system, params = tmaze_system()
    beliefs, _ = system.infer_states(params, OBS)
    with pytest.raises(
        ValueError, match=r"update_empirical_prior: actions \['a_location'\] are not"
    ):
        system.update_empirical_prior(params, beliefs, {"a_location": 1.7})
    with pytest.raises(ValueError, match=r"to_positional.*actions \['a_location'\] are not"):
        system.to_positional("actions", {"a_location": jnp.asarray(1.0)})
    env, ep = system.as_env(params)
    _, state = env.reset(KEY, ep)
    with pytest.raises(ValueError, match=r"step: actions \['a_location'\] are not integer"):
        env.step(KEY, state, {"a_location": jnp.asarray(2.0)}, ep)


def test_stack_traces_refuses_positional_traces_with_the_converter_sentence():
    from hybrid_aif.replay import stack_traces

    obs = (jnp.zeros(3, jnp.int32),) * 3
    actions = {"a_location": jnp.zeros(3, jnp.int32)}
    with pytest.raises(TypeError, match=r"from_positional\('observations', \.\.\.\)"):
        stack_traces([(obs, actions, None)])
    keyed_obs = {k: jnp.zeros(3, jnp.int32) for k in OBS}
    with pytest.raises(TypeError, match=r"trace 0's actions .*from_positional\('actions'"):
        stack_traces([(keyed_obs, jnp.zeros((3, 2), jnp.int32), None)])
    with pytest.raises(TypeError, match=r"trace 0's present flags"):
        stack_traces([(keyed_obs, actions, jnp.ones((3, 3), bool))])


def test_a_mind_returning_a_misshapen_or_float_level_is_named():
    from hybrid_aif.composition.driver import _check_action

    with pytest.raises(ValueError, match=r"mind 'a' returned actions\['a_x'\] of shape \(2,\)"):
        _check_action({"a_x": jnp.zeros(2, jnp.int32)}, ("a_x",), "a")
    with pytest.raises(ValueError, match=r"mind 'a': actions \['a_x'\] are not integer"):
        _check_action({"a_x": jnp.asarray(1.0)}, ("a_x",), "a")


# ------------------------------------------------------- two continuous controls, out of order


def _two_continuous():
    """Factors named out of alphabetical order, controls 1 and 2 wide: ``z`` then ``a``."""
    from hybrid_aif import Gaussian, MLPConditional, System, SystemParams
    from tests.test_cem import box

    keys = jax.random.split(jax.random.PRNGKey(7), 2)
    system = System(
        A_deps=(),
        B_deps=((), ()),
        controlled=(True, True),
        policy_len=2,
        planner="cem",
        epistemic=False,
        factor_names=("z", "a"),
    )
    params = SystemParams(
        transition={
            "z": MLPConditional.from_shapes((1, 1), 1, key=keys[0], depth=0, init_scale=0.05),
            "a": MLPConditional.from_shapes((1, 2), 1, key=keys[1], depth=0, init_scale=0.05),
        },
        initial_state_prior={
            "z": Gaussian(loc=jnp.zeros(1), chol=0.1 * jnp.eye(1)),
            "a": Gaussian(loc=jnp.zeros(1), chol=0.1 * jnp.eye(1)),
        },
        state_targets={"z": Gaussian(loc=jnp.ones(1), chol=0.3 * jnp.eye(1))},
        control={"u_z": box(dim=1), "u_a": box(dim=2)},
    )
    return system, params


def test_two_continuous_controls_keep_their_blocks_and_refuse_swapped_ones():
    system, params = _two_continuous()
    beliefs = system.initial_priors(params)
    action, _, _ = system.act(KEY, params, beliefs, plan_key=KEY)
    assert set(action) == {"a_z", "a_a"}
    assert action["a_z"].shape == (1,) and action["a_a"].shape == (2,)
    flat = system.to_positional("actions", action, params)
    assert np.array_equal(flat, jnp.concatenate([action["a_z"], action["a_a"]]))
    #  the keyed face pushes exactly what the positional internal pushes
    pushed = system.update_empirical_prior(params, beliefs, action)
    internal = system._update_empirical_prior(
        params, system.to_positional("beliefs", beliefs), flat
    )
    for name, b in zip(system.names.states, internal, strict=True):
        assert np.array_equal(pushed[name].loc, b.loc)

    swapped = {"a_z": jnp.asarray([0.2, 0.3]), "a_a": jnp.asarray([0.1])}
    with pytest.raises(
        ValueError, match=r"actions\['a_z'\] is the control on 'z', a \(\.\.\., 1\)"
    ):
        system.update_empirical_prior(params, beliefs, swapped)
    env, ep = system.as_env(params)
    obs, state = env.reset(KEY, ep)
    assert obs == {} and set(state) == {"z", "a"}
    nxt_obs, nxt = env.step(KEY, state, action, ep)
    positional_ep = system.to_positional("params", ep)
    _, ref = env.inner.step(KEY, system.to_positional("beliefs", state), flat, positional_ep)
    for name, r in zip(system.names.states, ref, strict=True):
        assert np.array_equal(nxt[name], r)
    with pytest.raises(ValueError, match=r"step: actions\['a_z'\]"):
        env.step(KEY, state, swapped, ep)
    assert set(env.null_action()) == {"a_z", "a_a"} and env.null_action()["a_a"].shape == (2,)

    T = 3
    trace = {"a_z": jnp.full((T, 1), 0.1), "a_a": jnp.full((T, 2), -0.2)}
    out = replay(system, params, {}, trace, key=KEY, metrics=False)
    assert out.chosen["a_z"].shape == (T, 1) and out.chosen["a_a"].shape == (T, 2)
    bad = {"a_z": jnp.full((T, 2), 0.1), "a_a": jnp.full((T, 1), -0.2)}
    with pytest.raises(ValueError, match=r"replay: actions\['a_z'\] must be \(T, 1\)"):
        replay(system, params, {}, bad, key=KEY, metrics=False)


# ------------------------------------------- a multi-factor, multi-modality env, under transforms


def test_keyed_env_matches_its_positional_process_by_name_eagerly_and_under_jit_and_scan():
    system, params = tmaze_system()
    assert system.names.states == ("location", "context")  # not alphabetical
    env, ep = system.as_env(params)
    assert env.num_modalities == 3 and env.num_factors == 2
    action = {"a_location": jnp.asarray(3, jnp.int32)}

    def keyed_steps(key):
        obs, state = env.reset(key, ep)

        def body(carry, k):
            _, state = carry
            nxt = env.step(k, state, action, ep)
            return nxt, nxt

        return jax.lax.scan(body, (obs, state), jax.random.split(key, 4))[1]

    positional_ep = system.to_positional("params", ep)

    def positional_steps(key):
        obs, state = env.inner.reset(key, positional_ep)
        flat = system.to_positional("actions", action)

        def body(carry, k):
            _, state = carry
            nxt = env.inner.step(k, state, flat, positional_ep)
            return nxt, nxt

        return jax.lax.scan(body, (obs, state), jax.random.split(key, 4))[1]

    want_obs, want_state = positional_steps(KEY)
    for got_obs, got_state in (keyed_steps(KEY), jax.jit(keyed_steps)(KEY)):
        for m, name in enumerate(system.modality_names):
            assert np.array_equal(got_obs[name], want_obs[m])
        for f, name in enumerate(system.names.states):
            assert np.array_equal(got_state[name], want_state[f])

    B, T = 2, 4
    batched = broadcast_params(ep, B)
    keyed_traj = rollout(KEY, env, batched, random_policy(system, params), T, B)
    flat_batched = broadcast_params(positional_ep, B)
    flat_traj = rollout(KEY, env.inner, flat_batched, random_policy((5,)), T, B)
    for m, name in enumerate(system.modality_names):
        assert np.array_equal(keyed_traj.observations[name], flat_traj.observations[m])
        assert np.array_equal(keyed_traj.history.obs[name], flat_traj.history.obs[m])
    for f, name in enumerate(system.names.states):
        assert np.array_equal(keyed_traj.states[name], flat_traj.states[f])
    assert np.array_equal(keyed_traj.actions["a_location"], flat_traj.actions[..., 0])
    assert np.array_equal(
        keyed_traj.history.actions["a_location"], flat_traj.history.actions[..., 0]
    )


# ---------------------------------------------------------------- the keyed InferenceDetail


def test_detail_is_keyed_by_modality_and_factor_and_round_trips():
    from hybrid_aif.inference.detail import InferenceDetail

    system, params = tmaze_system()  # factors and modalities in non-alphabetical order
    _, detail = system.infer_states(params, OBS, detail=True)
    assert isinstance(detail, InferenceDetail)
    for field in (detail.pre_reduce, detail.mode_log_evidence, detail.log_evidence):
        assert list(field) == ["location_obs", "reward", "cue"]
    assert list(detail.unsupported) == ["location", "context"]
    positional = system.to_positional("detail", detail)
    assert positional.log_evidence == (None, None, None)
    assert positional.unsupported[0] is detail.unsupported["location"]
    back = system.from_positional("detail", positional)
    assert back.unsupported == detail.unsupported and back.pre_reduce == detail.pre_reduce
    with pytest.raises(TypeError, match=r"takes a positional InferenceDetail"):
        system.from_positional("detail", detail)


def test_learn_takes_the_keyed_detail_and_refuses_a_positional_one():
    noise = _example("noise_learning")
    system = noise.make_system()
    params = noise.make_params(jnp.asarray([[[0.3]], [[0.3]]]))
    obs = {"pose": jnp.asarray([0.2]), "cell": jnp.asarray(0, jnp.int32)}  # keys out of order
    beliefs, detail = system.infer_states(params, obs, detail=True)
    keyed_out = system.learn(params, obs, beliefs, detail=detail)
    internal = system._learn(
        params,
        system.to_positional("observations", obs),
        system.to_positional("beliefs", beliefs),
        detail=system.to_positional("detail", detail),
    )
    for a, b in zip(
        jax.tree_util.tree_leaves(keyed_out), jax.tree_util.tree_leaves(internal), strict=True
    ):
        assert np.array_equal(a, b)
    with pytest.raises(TypeError, match=r"from_positional\('detail', \.\.\.\)"):
        system.learn(params, obs, beliefs, detail=system.to_positional("detail", detail))
    reordered = dataclasses.replace(detail, pre_reduce={"pose": detail.pre_reduce["pose"]})
    with pytest.raises(ValueError, match=r"detail.pre_reduce is missing \['cell'\]"):
        system.learn(params, obs, beliefs, detail=reordered)


# ------------------------------------------------------------------- keyed aux_beliefs


@pytest.fixture(scope="module")
def raif_agent():
    from hybrid_aif.reference.raif.agent import raif_mountain_car

    tiny = dict(
        horizon=2,
        head_units=8,
        head_layers=1,
        units=4,
        deter=8,
        num_vars=3,
        num_classes=4,
        embed=5,
        mlp_units=6,
        ensemble_size=4,
    )
    return raif_mountain_car(jax.random.PRNGKey(0), **tiny)


def test_aux_beliefs_are_keyed_by_the_nested_systems_factor_names(raif_agent):
    system, params = raif_agent.system, raif_agent.params
    row = "cosine_sim_to_pref_rollout"
    nested = dict(system.scoring)[row]
    carry = nested.initial_priors(params.scoring[row])
    reversed_order = dict(reversed(list(carry.items())))
    #  the face resolves the carry by the nested System's names, whatever the dict order
    for face in ("planner_context", "score_policies", "imagine", "rollout_returns"):
        converted = system._aux_in({row: reversed_order}, face)[row]
        assert converted == nested.to_positional("beliefs", carry)
    assert nested.from_positional("beliefs", converted) == carry  # the round trip
    beliefs = system.initial_priors(params)
    plan = jnp.zeros((3, 1))
    system.score_policies(params, beliefs, plan, aux_beliefs={row: reversed_order})
    positional = nested.to_positional("beliefs", carry)
    with pytest.raises(
        TypeError,
        match=r"score_policies aux_beliefs\['cosine_sim_to_pref_rollout'\].*"
        r"dict\(system\.scoring\)\['cosine_sim_to_pref_rollout'\]\.from_positional\('beliefs'",
    ):
        system.score_policies(params, beliefs, plan, aux_beliefs={row: positional})
    missing = dict(list(carry.items())[:1])
    with pytest.raises(ValueError, match=r"planner_context aux_beliefs.* are missing"):
        system.planner_context(params, beliefs, aux_beliefs={row: missing})


# ------------------------------------------------------------------ keyed params containers


def test_params_containers_are_keyed_and_convert_by_name():
    system, params = tmaze_system()
    assert list(params.likelihood) == ["location_obs", "reward", "cue"]
    assert list(params.transition) == ["location", "context"]
    positional = system.to_positional("params", params)
    assert positional.likelihood[2] is params.likelihood["cue"]
    assert positional.transition == (params.transition["location"], params.transition["context"])
    back = system.from_positional("params", positional)
    assert list(back.likelihood) == ["location_obs", "reward", "cue"]
    assert back.initial_state_prior["context"] is params.initial_state_prior["context"]
    #  insertion order never matters: containers are read by name
    shuffled = dataclasses.replace(
        params, likelihood=dict(reversed(list(params.likelihood.items())))
    )
    a, _ = system.infer_states(params, OBS)
    b, _ = system.infer_states(shuffled, OBS)
    assert np.array_equal(a["context"].probs, b["context"].probs)
    with pytest.raises(TypeError, match=r"from_positional\('params', \.\.\.\) takes positional"):
        system.from_positional("params", params)


def test_a_positional_params_container_is_refused_with_the_converter_sentence():
    system, params = tmaze_system()
    positional = system.to_positional("params", params)
    with pytest.raises(TypeError) as info:
        system.infer_states(positional, OBS)
    assert str(info.value) == (
        "infer_states takes params.likelihood as a mapping keyed by modality name "
        "(['location_obs', 'reward', 'cue']); convert positional containers with "
        "system.from_positional('params', params)"
    )
    for face in (
        lambda: system.initial_priors(positional),
        lambda: system.as_env(positional),
        lambda: replay(
            system,
            positional,
            {k: v[None] for k, v in OBS.items()},
            {"a_location": jnp.zeros(1, jnp.int32)},
        ),
    ):
        with pytest.raises(TypeError, match=r"from_positional\('params', params\)"):
            face()
    missing = dataclasses.replace(params, transition={"location": params.transition["location"]})
    with pytest.raises(ValueError, match=r"params.transition is missing \['context'\]"):
        system.infer_states(missing, OBS)


def test_learn_returns_keyed_params_and_slot_addresses_agree():
    noise = _example("noise_learning")
    system = noise.make_system()
    params = noise.make_params(
        InverseGamma.from_R_chol(jnp.full((2, 1, 1), 0.6), strength=2.0, anchor=True)
    )
    obs = {"cell": jnp.asarray(0, jnp.int32), "pose": jnp.asarray([0.2])}
    beliefs, detail = system.infer_states(params, obs, detail=True)
    learned = system.learn(params, obs, beliefs, detail=detail)
    assert list(learned.likelihood) == ["cell", "pose"]
    frozen = system.learn(params, obs, beliefs, detail=detail, lr_by_slot={"likelihood:pose": 0.0})
    assert frozen.likelihood["pose"] is params.likelihood["pose"]
    assert learned.likelihood["pose"] is not params.likelihood["pose"]


def test_system_names_is_public_for_unnamed_models():
    from hybrid_aif import System

    system, params = tmaze_system()
    unnamed = System(
        A_deps=system.A_deps,
        B_deps=system._table.B_deps,
        controlled=(True, False),
        policy_len=2,
    )
    assert unnamed.names.states == ("factor_0", "factor_1")
    assert unnamed.names.modalities == ("modality_0", "modality_1", "modality_2")
    assert unnamed.names.controls == ("u_factor_0",)
    keyed_params = unnamed.from_positional("params", system.to_positional("params", params))
    obs = unnamed.from_positional("observations", tuple(OBS.values()))
    beliefs, _ = unnamed.infer_states(keyed_params, obs)
    assert [k for k in beliefs if k in unnamed.names.states] == ["factor_0", "factor_1"]


# ------------------------------------------------------- wrapping an outside positional env


class _Counter(Env):
    """A minimal pymdp-style positional env: one factor counting actions, one observation of it."""

    @property
    def num_factors(self) -> int:
        return 1

    @property
    def num_modalities(self) -> int:
        return 1

    def reset(self, key, params, state=None):
        start = (jnp.asarray(0, jnp.int32),) if state is None else state
        return (start[0] * params,), start

    def step(self, key, state, action, params):
        nxt = (state[0] + action[0],)
        return (nxt[0] * params,), nxt


def test_generative_env_wraps_an_outside_positional_env_given_a_system():
    from hybrid_aif import Categorical, ConditionalCategorical, System, SystemParams

    system = System(
        A_deps=((0,),),
        B_deps=((0,),),
        controlled=(True,),
        policy_len=1,
        factor_names=("count",),
        modality_names=("seen",),
    )
    #  the control's levels live with the tables: three, read by the one transition
    params = SystemParams(
        likelihood={"seen": ConditionalCategorical(log_table=jnp.zeros((2, 2)), parent_ndims=1)},
        transition={
            "count": ConditionalCategorical(log_table=jnp.zeros((2, 2, 3)), parent_ndims=2)
        },
        initial_state_prior={"count": Categorical(logits=jnp.zeros(2))},
    )
    env = GenerativeEnv.wrap(_Counter(), system)
    obs, state = env.reset(KEY, 10)  # an outside env's own params pass through untouched
    assert obs == {"seen": 0} and list(state) == ["count"]
    obs, state = env.step(KEY, state, {"a_count": jnp.asarray(2, jnp.int32)}, 10)
    assert int(obs["seen"]) == 20 and int(state["count"]) == 2
    traj = rollout(KEY, env, jnp.full((2,), 10), random_policy(system, params), 3, 2)
    assert traj.observations["seen"].shape == (2, 4) and list(traj.actions) == ["a_count"]
    with pytest.raises(ValueError, match=r"has 1 factor\(s\), the System 2"):
        GenerativeEnv.wrap(_Counter(), tmaze_system()[0])


def test_beliefs_from_states_keys_filter_states_by_factor():
    from hybrid_aif.reference.raif.agent import raif_mountain_car
    from hybrid_aif.train.world import assemble, belief_tuple

    agent = raif_mountain_car(
        jax.random.PRNGKey(0),
        horizon=2,
        head_units=8,
        head_layers=1,
        units=4,
        deter=8,
        num_vars=3,
        num_classes=4,
        embed=5,
        mlp_units=6,
        ensemble_size=4,
    )
    core, _ = assemble(agent.system, agent.params)
    states = core.initial_state()
    beliefs = agent.system.beliefs_from_states(agent.params, states)
    assert [k for k in beliefs if k in agent.system.names.states] == list(agent.system.names.states)
    positional = belief_tuple(agent.system._amortized_structure(agent.params), states)
    assert agent.system.to_positional("beliefs", beliefs) == positional


def test_names_is_a_view_of_the_systems_name_tuples():
    from hybrid_aif import System

    system, _ = tmaze_system()
    names = system.names
    assert names._fields == ("states", "controls", "modalities", "actions")
    assert names.actions == system.action_names == ("a_location",)
    assert names.states + names.controls == system.factor_names
    assert names.modalities is system.modality_names
    assert names.states == ("location", "context") and names.controls == ("u_location",)
    #  construction fills the generated names in, so the tuples are never absent
    unnamed = System(
        A_deps=system.A_deps,
        B_deps=system._table.B_deps,
        controlled=(True, False),
        policy_len=2,
    )
    assert unnamed.factor_names == ("factor_0", "factor_1", "u_factor_0")
    assert unnamed.names.states == ("factor_0", "factor_1")
    assert unnamed.modality_names == ("modality_0", "modality_1", "modality_2")


def test_generative_env_refuses_positional_params_like_the_system_faces():
    system, params = tmaze_system()
    env, ep = system.as_env(params)
    positional = system.to_positional("params", ep)
    with pytest.raises(TypeError, match=r"from_positional\('params', params\)"):
        env.reset(KEY, positional)
    _, state = env.reset(KEY, ep)
    with pytest.raises(TypeError, match=r"GenerativeEnv takes params.likelihood as a mapping"):
        env.step(KEY, state, {"a_location": jnp.asarray(1, jnp.int32)}, positional)
    #  the positional pymdp-shaped builders convert explicitly, then wrap
    from hybrid_aif.reference.envs.tmaze import tmaze

    process, raw = tmaze(reward_probability=0.98, punishment_probability=0.98, cue_validity=0.98)
    wrapped = GenerativeEnv.wrap(process, system)
    obs, _ = wrapped.reset(KEY, system.from_positional("params", raw))
    assert list(obs) == ["location_obs", "reward", "cue"]


def test_wrap_checks_counts_and_refuses_a_double_wrap():
    from hybrid_aif import System

    system, params = tmaze_system()
    env, _ = system.as_env(params)
    with pytest.raises(TypeError, match="already a GenerativeEnv"):
        GenerativeEnv.wrap(env, system)
    two_factors_one_modality = System(
        A_deps=((0,),),
        B_deps=((0,), (1,)),
        controlled=(True, False),
        policy_len=1,
        factor_names=("location", "context"),
        modality_names=("seen",),
    )
    with pytest.raises(ValueError, match=r"has 3 modality/ies, the System 1 \(\['seen'\]\)"):
        GenerativeEnv.wrap(env.inner, two_factors_one_modality)


def test_an_int_action_from_an_outside_env_cannot_become_a_continuous_nodes_value():
    from hybrid_aif import (
        Gaussian,
        LinearGaussianObservation,
        MLPConditional,
        System,
        SystemParams,
    )
    from tests.test_cem import box

    system = System(  # one continuous control on "x", one modality, as the counter env has
        A_deps=((0,),),
        B_deps=((),),
        controlled=(True,),
        policy_len=1,
        planner="cem",
        factor_names=("x",),
        modality_names=("seen",),
    )
    params = SystemParams(
        transition={"x": MLPConditional.from_shapes((1, 1), 1, key=KEY, depth=0)},
        initial_state_prior={"x": Gaussian(loc=jnp.zeros(1), chol=jnp.eye(1))},
        likelihood={
            "seen": LinearGaussianObservation(H=jnp.eye(1), c=jnp.zeros(1), noise=jnp.eye(1))
        },
        control={"u_x": box()},
    )
    counted = GenerativeEnv.wrap(_Counter(), system, params)
    with pytest.raises(ValueError, match=r"got dtype int32; this System's controls \['a_x'\]"):
        counted.null_action()
    with pytest.raises(ValueError, match="got dtype int32"):
        system.from_positional("actions", jnp.zeros((1,), jnp.int32), params)


# ---------------------------------------------------------------- the name tuples' counts


def _unnamed():
    from hybrid_aif import System

    return System(B_deps=((0,),), controlled=(True,), policy_len=1, A_deps=((0,),))


def test_a_replace_that_grows_the_factors_must_regenerate_or_rename():
    from hybrid_aif import System

    s = _unnamed()
    assert s.factor_names == ("factor_0", "u_factor_0")
    #  a node form that grows a state factor (the control node moves to index 2)
    grow = dict(B_deps=((0, 2), (1,), ()), controls=(2,))
    with pytest.raises(
        ValueError,
        match=r"factor_names has 2 name\(s\) for 3 node\(s\).*construct the System afresh",
    ):
        dataclasses.replace(s, **grow)
    #  regenerating the names of a System that has control nodes would rename them: refused
    with pytest.raises(ValueError, match=r"factor_names=None on a System with control nodes"):
        dataclasses.replace(s, **grow, factor_names=None)
    #  so a grown System is declared afresh from the sugar, which generates every name
    grown = System(
        B_deps=((0,), (1,)),
        controlled=(True, False),
        policy_len=1,
        A_deps=((0,),),
    )
    assert grown.factor_names == ("factor_0", "factor_1", "u_factor_0")
    assert grown.modality_names == ("modality_0",)
    assert dataclasses.replace(s, policy_len=2).factor_names == s.factor_names
    renamed = dataclasses.replace(s, **grow, factor_names=("a", "b", "u_a"), action_names=("a_a",))
    assert renamed.factor_names == ("a", "b", "u_a")
    assert renamed.modality_names == ("modality_0",)


def test_a_wrong_count_of_declared_names_raises():
    from hybrid_aif import System

    with pytest.raises(ValueError, match=r"modality_names has 2 name\(s\) for 1 modality\(s\)"):
        System(
            B_deps=((0,),),
            controlled=(False,),
            policy_len=1,
            A_deps=((0,),),
            modality_names=("a", "b"),
        )
    with pytest.raises(ValueError, match=r"factor_names has 3 name\(s\) for 2 node\(s\)"):
        dataclasses.replace(_unnamed(), factor_names=("a", "b", "c"))


# ------------------------------------------------------- dtype symmetry and env/policy forms


class _FloatCounter(_Counter):
    """A positional env whose no-op action is a float: wrong for an int-level System."""

    def null_action(self):
        return jnp.zeros((1,), jnp.float32)


def test_an_outside_envs_float_action_on_an_int_system_is_refused_before_any_step():
    from hybrid_aif import System

    system = System(
        A_deps=((0,),),
        B_deps=((0,),),
        controlled=(True,),
        policy_len=1,
        factor_names=("count",),
        modality_names=("seen",),
    )
    env = GenerativeEnv.wrap(_FloatCounter(), system)
    with pytest.raises(
        ValueError, match=r"actions \['a_count'\] are not integer \(positional dtype float32\)"
    ):
        env.null_action()
    with pytest.raises(ValueError, match=r"actions \['a_count'\] are not integer"):
        system.from_positional("actions", jnp.zeros((1,), jnp.float32))


def test_a_continuous_control_refuses_a_non_float_block():
    system, params = lq_system()
    beliefs = system.initial_priors(params)
    with pytest.raises(ValueError, match=r"actions\['a_x'\] has dtype int32; .* take float blocks"):
        system.update_empirical_prior(params, beliefs, {"a_x": jnp.zeros((1,), jnp.int32)})


def test_a_keyed_policy_on_a_positional_env_is_refused_with_the_converter_sentence():
    from hybrid_aif.reference.envs.tmaze import tmaze

    system, params = tmaze_system()
    process, raw = tmaze(reward_probability=0.98, punishment_probability=0.98, cue_validity=0.98)
    with pytest.raises(
        TypeError, match=r"policy returned a mapping, but this env takes a positional"
    ):
        rollout(KEY, process, broadcast_params(raw, 2), random_policy(system, params), 2, 2)
    env, ep = system.as_env(tmaze_system()[1])
    with pytest.raises(
        TypeError, match=r"policy returned a positional array, but this env takes a mapping"
    ):
        rollout(KEY, env, broadcast_params(ep, 2), random_policy((5,)), 2, 2)


def _flat_priors():
    """Uniform priors over the T-maze's factors, so every modality moves the posterior."""
    from hybrid_aif.distributions import Categorical

    return {"location": Categorical(jnp.zeros(5)), "context": Categorical(jnp.zeros(2))}


def test_present_is_partial_none_is_all_and_replay_takes_a_flag_per_tick():
    system, params = tmaze_system()
    a, _ = system.infer_states(params, OBS)
    b, _ = system.infer_states(params, OBS, present=None)
    c, _ = system.infer_states(params, OBS, present={})
    for name in ("location", "context"):
        assert np.array_equal(a[name].probs, b[name].probs)
        assert np.array_equal(a[name].probs, c[name].probs)
    seen = {"location_obs": jnp.asarray(3), "reward": jnp.asarray(0), "cue": jnp.asarray(1)}
    obs = {k: jnp.stack([v, v, v]) for k, v in seen.items()}
    actions = {"a_location": jnp.asarray([3, 3, 3], jnp.int32)}
    cue_off = jnp.asarray([False, True, True])
    flat = _flat_priors()
    partial = replay(system, params, obs, actions, present={"cue": cue_off}, priors=flat)
    complete = replay(
        system,
        params,
        obs,
        actions,
        present={"location_obs": jnp.ones(3, bool), "reward": jnp.ones(3, bool), "cue": cue_off},
        priors=flat,
    )
    unflagged = replay(system, params, obs, actions, priors=flat)
    for name in ("location", "context"):
        np.testing.assert_array_equal(
            np.asarray(partial.beliefs[name].probs), np.asarray(complete.beliefs[name].probs)
        )
    changed = [
        not np.allclose(
            np.asarray(partial.beliefs[n].probs), np.asarray(unflagged.beliefs[n].probs)
        )
        for n in ("location", "context")
    ]
    assert any(changed)
    assert bool(partial.exercised["location_obs"]) and bool(partial.exercised["cue"])
    with pytest.raises(ValueError, match=r"present has unknown key \['cues'\]"):
        replay(system, params, obs, actions, present={"cues": cue_off})
    with pytest.raises(ValueError, match=r"present entries are \(3,\) bool"):
        replay(system, params, obs, actions, present={"cue": cue_off[:2]})


def test_wrapping_a_continuous_system_without_its_params_points_at_params():
    """Without ``params=`` the System cannot know its node is continuous (the box lives in
    params), so a float action is refused with the pointer at ``params=``."""
    from hybrid_aif import System

    system = System(
        A_deps=((0,),),
        B_deps=((),),
        controlled=(True,),
        policy_len=1,
        factor_names=("x",),
        modality_names=("seen",),
    )
    env = GenerativeEnv.wrap(_FloatCounter(), system)
    with pytest.raises(ValueError, match=r"pass params= — to the converter, or to GenerativeEnv"):
        env.null_action()
    with pytest.raises(ValueError, match="pass params="):
        env.step(KEY, {"x": jnp.zeros(1)}, {"a_x": jnp.zeros(1)}, 1.0)
