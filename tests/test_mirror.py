"""The mirror of a System, and ``as_env`` over it.

``as_env`` is the mirror (each modality an emission), lowered, run forward by the positional
process, so every environment it returns is the one the forward process always drew. The eight
world models of :mod:`tests.mirror_cases` are recorded in ``tests/data/as_env_<case>.npz`` as that
process produced them, 50 ticks of reset and step each, and must come back bit for bit.
"""

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import (
    Categorical,
    ConditionalCategorical,
    Emission,
    Parameter,
    System,
    SystemParams,
)
from hybrid_aif.composition.merge import merge
from hybrid_aif.distributions import ConditionalDirichlet
from hybrid_aif.envs.generative import CONTROL_TAG
from hybrid_aif.system.mirror import mirror
from tests import mirror_cases
from tests.test_action_node_modalities import maze_with_cue_by_node
from tests.test_action_nodes import pusher
from tests.test_emissions import world as emitting_world
from tests.test_run_parity import as_recorded
from tests.test_system import tmaze_system
from tests.test_terrain_field import _rover

DATA = Path(__file__).parent / "data"


@pytest.mark.parametrize("name", sorted(mirror_cases.cases()))
def test_as_env_over_the_mirror_reproduces_the_recorded_forward_samples(name):
    """Bitwise. Under JAX's 64-bit mode the neural case (its networks built in float32) differs
    from the recording by at most two float32 ulps of the largest recorded value (measured: 3.6e-7
    on the readings, 1.5 ulps there, and 2.4e-7 on the state), the float32 networks compiled under
    that mode; every other case is bitwise there too."""
    case = mirror_cases.cases()[name]
    #  the recordings are float32 draws; under JAX's 64-bit mode the cases are cast to float32
    #  (as_recorded asserts every float leaf is), whose draws are the recorded stream
    env, params = case.system.as_env(as_recorded(case.params), case.param_key)
    got = mirror_cases.record(env, params, case.states)
    want = np.load(DATA / f"as_env_{name}.npz")
    assert {k.replace("/", "__") for k in got} == set(want.files)
    by_ulp = name == "mlp" and jax.config.jax_enable_x64
    for key, value in got.items():
        recorded = want[key.replace("/", "__")]
        if by_ulp and np.issubdtype(recorded.dtype, np.floating):
            assert value.dtype == recorded.dtype == np.float32, key
            ulp = float(np.spacing(np.abs(recorded).max()))  # at the largest recorded value
            np.testing.assert_allclose(value, recorded, rtol=0, atol=2 * ulp, err_msg=key)
        else:
            np.testing.assert_array_equal(value, recorded, err_msg=key)


# ----------------------------------------------------------------------- the transform

KEY = jax.random.PRNGKey(2)


def test_every_modality_becomes_an_emission_of_the_same_name_and_the_dynamics_are_kept():
    system, params = tmaze_system()
    world, wp = mirror(system, params)
    assert world.modality_names == () and world.A_deps == ()
    assert world.names.emissions == system.modality_names
    for m, deps in zip(system.modality_names, system.A_deps, strict=True):
        parents = tuple(system.node_names[g] for g in deps)
        assert dict(world.emissions)[m] == Emission(parents, "sample")
    for name in (
        "B_deps",
        "D_deps",
        "controls",
        "state_names",
        "control_names",
        "action_names",
        "dims",
        "cells",
    ):
        assert getattr(world, name) == getattr(system, name)
    for m in system.modality_names:
        np.testing.assert_array_equal(
            np.asarray(wp.emission[m].log_table), np.asarray(params.likelihood[m].log_table)
        )
    assert wp.likelihood == {} and wp.actuator == {}
    for f in system.names.states:
        assert wp.transition[f] is params.transition[f]
        assert wp.initial_state_prior[f] is params.initial_state_prior[f]


def test_the_planner_the_learner_and_the_posterior_side_are_dropped():
    system, params = tmaze_system()
    system = dataclasses.replace(system, epistemic=("context",))
    world, wp = mirror(system, params)
    assert world.epistemic is False and world.terms == () and world.C_deps == ()
    assert world.H_deps is None and world.observes == () and world.readings == ()
    assert wp.obs_targets == {} and wp.state_targets == {} and wp.terminal_targets == {}
    assert wp.recognition == {} and wp.scoring == {} and wp.readings == {}
    assert world.subsystems is None


def test_an_action_node_a_modality_reads_becomes_the_control_node_an_emission_reads():
    system, params = maze_with_cue_by_node()
    world, wp = mirror(system, params)
    assert dict(world.emissions)["cue"] == Emission(("context", "u_act"), "sample")
    assert dict(world.emissions)["reward"] == Emission(("context", "arm"), "sample")
    assert world.ports(wp).outs["cue"] == ("level", 2)
    assert world.ports(wp).ins == {"u_act": ("level", 3)}
    assert world.names.actions == system.names.actions


def test_an_actuator_is_not_carried_by_the_mirror_but_as_env_applies_it():
    system, params = mirror_cases.cases()["actuator"][:2]
    world, wp = mirror(system, params)
    assert wp.actuator == {}
    _, env_params = system.as_env(params)
    assert set(env_params.actuator) == {"push"}


def _shared_table_system():
    system = System(
        A_deps=((0,), (0,)),
        B_deps=((0,),),
        controlled=(True,),
        policy_len=1,
        modality_names=("a", "b"),
        state_names=("s",),
        epistemic=False,
    )
    home = ConditionalDirichlet(
        concentration=jnp.asarray([[4.0, 1.0, 1.0], [1.0, 4.0, 1.0], [1.0, 1.0, 4.0]]),
        parent_ndims=1,
    )
    eye = ConditionalCategorical(log_table=jnp.log(0.9 * jnp.eye(3) + 0.05), parent_ndims=1)
    params = SystemParams(
        likelihood={"a": Parameter("table"), "b": Parameter("table")},
        transition={
            "s": ConditionalCategorical(log_table=jnp.log(jnp.ones((3, 3, 3)) / 3), parent_ndims=2)
        },
        initial_state_prior={"s": Categorical(logits=jnp.zeros(3))},
        parameters={"table": home},
    )
    return system, params, eye


def test_parameter_variables_are_kept_as_they_are_and_lowering_is_an_option():
    system, params, _ = _shared_table_system()
    world, wp = mirror(system, params)
    assert wp.parameters["table"] is params.parameters["table"]
    assert wp.emission == {"a": Parameter("table"), "b": Parameter("table")}
    lowered_world, lp = mirror(system, params, lower=True)
    assert lp.parameters == {} and lp.references == ()
    point = params.parameters["table"].point()
    for m in ("a", "b"):
        np.testing.assert_array_equal(
            np.asarray(lp.emission[m].log_table), np.asarray(point.log_table)
        )
    _, drawn = mirror(system, params, lower=True, param_key=KEY)
    assert not np.array_equal(
        np.asarray(drawn.emission["a"].log_table), np.asarray(point.log_table)
    )
    #  one draw per parameter variable: its readers share it
    np.testing.assert_array_equal(
        np.asarray(drawn.emission["a"].log_table), np.asarray(drawn.emission["b"].log_table)
    )
    with pytest.raises(ValueError, match="needs lower=True"):
        mirror(system, params, param_key=KEY)


def test_the_mirror_of_a_system_with_emissions_drops_them_as_its_out_ports():
    system, params = emitting_world()
    world, _ = mirror(system, params)
    assert world.names.emissions == system.modality_names
    system.as_env(params)


def test_the_refusals_of_as_env_are_the_mirrors():
    system, params = _rover()
    with pytest.raises(NotImplementedError, match="mirror on a System with cells"):
        mirror(system, params)
    with pytest.raises(NotImplementedError, match="as_env on a System with cells"):
        system.as_env(params)


# --------------------------------------------------------------- sampling forward


def test_sample_step_is_the_env_step_when_every_control_node_is_driven():
    system, params = pusher()
    env, ep = system.as_env(params)
    world, wp = mirror(system, params)
    obs0, state0 = env.reset(KEY, ep)
    obs1, state1 = world.sample_initial(KEY, wp)
    assert {k: int(v) for k, v in obs0.items()} == {k: int(v) for k, v in obs1.items()}
    assert {k: int(v) for k, v in state0.items()} == {k: int(v) for k, v in state1.items()}
    for t in range(12):
        key = jax.random.fold_in(KEY, t)
        level = jnp.asarray(t % 3, jnp.int32)
        want_obs, want_state = env.step(key, state0, {"push": level}, ep)
        got_obs, got_state = world.sample_step(key, wp, state0, {"u_push": level})
        assert {k: int(v) for k, v in want_obs.items()} == {k: int(v) for k, v in got_obs.items()}
        assert {k: int(v) for k, v in want_state.items()} == {
            k: int(v) for k, v in got_state.items()
        }
        state0 = want_state


def test_a_masked_sample_step_keeps_the_factors_it_is_not_told_to_step():
    system, params = pusher()
    world, wp = mirror(system, params)
    _, state = world.sample_initial(KEY, wp)
    _, kept = world.sample_step(KEY, wp, state, {"u_push": jnp.asarray(2, jnp.int32)}, states=())
    assert int(kept["x"]) == int(state["x"])


def test_a_free_control_node_draws_apart_from_the_process_key_stream():
    system, params = pusher()
    env, ep = system.as_env(params)
    world, wp = mirror(system, params)
    _, state = env.reset(KEY, ep)
    key = jax.random.PRNGKey(9)
    drawn = jax.random.categorical(
        jax.random.split(jax.random.fold_in(key, CONTROL_TAG), 1)[0], jnp.log(jnp.ones(3) / 3)
    )
    want_obs, want_state = env.step(key, state, {"push": drawn.astype(jnp.int32)}, ep)
    got_obs, got_state = world.sample_step(key, wp, state)
    assert int(got_state["x"]) == int(want_state["x"])
    assert int(got_obs["see"]) == int(want_obs["see"])
    levels = {int(world.sample_step(k, wp, state)[1]["x"]) for k in jax.random.split(key, 60)}
    assert len(levels) > 1  # the node is not pinned


def test_a_free_node_draws_from_the_habit_when_there_is_one():
    system, params = pusher()
    habit = jnp.log(jnp.asarray([0.0, 1.0, 0.0]) + 1e-30)
    world, wp = mirror(system, dataclasses.replace(params, policy_prior=habit))
    env, ep = system.as_env(params)
    _, state = env.reset(KEY, ep)
    for k in jax.random.split(KEY, 10):
        want_obs, want_state = env.step(k, state, {"push": jnp.asarray(1, jnp.int32)}, ep)
        got_obs, got_state = world.sample_step(k, wp, state)
        assert int(got_state["x"]) == int(want_state["x"])
        assert int(got_obs["see"]) == int(want_obs["see"])


def test_sample_step_names_a_wrong_input():
    system, params = pusher()
    world, wp = mirror(system, params)
    _, state = world.sample_initial(KEY, wp)
    with pytest.raises(ValueError, match="not control nodes"):
        world.sample_step(KEY, wp, state, {"push": 0})


def test_a_world_with_a_mode_emission_steps_forward():
    emitter, eparams = emitting_world()
    emitted, state = emitter.sample_initial(KEY, eparams)
    assert list(emitted) == ["fix", "img", "gauge"]
    stepped, nxt = emitter.sample_step(KEY, eparams, state, {"v": jnp.asarray(2, jnp.int32)})
    want = emitter.emit(
        jax.random.fold_in(KEY, 0x656D6974), eparams, {**nxt, "v": jnp.asarray(2, jnp.int32)}
    )
    assert int(stepped["img"]) == int(want["img"])  # the mode of its kernel at the stepped state
    table = np.exp(np.asarray(eparams.emission["img"].log_table))
    assert int(stepped["img"]) == int(np.argmax(table[:, int(nxt["x"]), int(nxt["terrain"])]))
    assert int(stepped["fix"]) in (0, 1, 2) and int(stepped["gauge"]) in (0, 1)


def test_sampling_forward_dissolves_parameter_variables_into_their_tables():
    system, params, _ = _shared_table_system()
    world, wp = mirror(system, params)
    assert wp.parameters  # the default mirror keeps the variable
    emitted, state = world.sample_initial(KEY, wp)
    emitted, state = world.sample_step(KEY, wp, state, {"u_s": jnp.asarray(1, jnp.int32)})
    assert set(emitted) == {"a", "b"}
    from tests.test_sibling_read import _rover

    rover, rparams = _rover(shared=True)
    world, wp = mirror(rover, rparams)
    assert wp.parameters
    emitted, state = world.sample_initial(KEY, wp)
    emitted, state = world.sample_step(KEY, wp, state)
    assert set(state) == {"x", "map"}


def test_a_habit_becomes_a_distribution_over_the_mirrors_first_step_level_tuples():
    system, params = pusher()
    habit = jnp.log(jnp.asarray([0.2, 0.5, 0.3]))
    long = dataclasses.replace(system, policy_len=2)
    #  nine two-step policies; the first-step marginal of the habit is what a free node draws
    logits = jnp.log(jnp.arange(1.0, 10.0))
    world, wp = mirror(long, dataclasses.replace(params, policy_prior=logits))
    assert world.policy_len == 1
    first = np.asarray(long.policies(dataclasses.replace(params, policy_prior=logits)))[:, 0, 0]
    want = np.bincount(first, weights=np.asarray(jax.nn.softmax(logits)), minlength=3)
    np.testing.assert_allclose(np.exp(np.asarray(wp.policy_prior)), want, rtol=1e-5)
    sophisticated = dataclasses.replace(system, planner="sophisticated")
    world, wp = mirror(sophisticated, dataclasses.replace(params, policy_prior=habit))
    np.testing.assert_allclose(np.exp(np.asarray(wp.policy_prior)), [0.2, 0.5, 0.3], rtol=1e-5)
    _, state = world.sample_initial(KEY, wp)
    moved = {int(world.sample_step(k, wp, state)[1]["x"]) for k in jax.random.split(KEY, 30)}
    assert len(moved) > 1  # the habit is a distribution, not a point


def test_a_habit_that_cannot_be_converted_is_refused_at_mirror_time():
    system, params = pusher()
    with pytest.raises(ValueError, match="enumerates 3 policies"):
        mirror(system, dataclasses.replace(params, policy_prior=jnp.zeros(5)))
    module = type("Module", (), {})()
    with pytest.raises(NotImplementedError, match="policy module"):
        mirror(system, dataclasses.replace(params, policy_prior=module))
    from tests.test_merge_nodes import _lg_mind

    lg, lp = _lg_mind("x", jnp.asarray([[0.3], [0.1]]))
    push = dataclasses.replace(system, planner="gradient")
    mixed, mp = merge({"a": lg, "p": push}, {"a": lp, "p": params})
    world, wp = mirror(mixed, mp)  # without a habit a mixed System mirrors
    _, state = world.sample_initial(KEY, wp)
    world.sample_step(KEY, wp, state)  # a free discrete node beside a continuous one draws
    with pytest.raises(NotImplementedError, match="continuous control nodes"):
        mirror(mixed, dataclasses.replace(mp, policy_prior=jnp.zeros(1)))
    with pytest.raises(NotImplementedError, match="continuous control nodes"):
        world.sample_step(KEY, dataclasses.replace(wp, policy_prior=jnp.zeros(3)), state)
