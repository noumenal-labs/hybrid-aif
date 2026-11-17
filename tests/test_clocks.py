"""Clocks under ``run``: a node that declares durations decides once per option.

Holding follows from declared durations and ``clocks=`` is only the conversion of a unit of
duration into rounds: the cadence of a coupled and of a merged robot, the unit conversion and its
arithmetic, the refusals; what a held round does (no step call, no write-back, the out-ports and
the driven level standing); a recording step that pins which rounds call a step and with which
key; and the duration face ``System.elapsed`` itself. ``tests/test_clock_parity.py`` holds the
comparison against the runs release 0.19.0 recorded.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.composition import (
    Identify,
    MindCarry,
    couple,
    merge,
    plan,
    run,
    runner,
    sample,
    with_mirror,
)
from hybrid_aif.system.mirror import mirror
from tests.host_log import logged, noted
from tests.test_clock_parity import _bump, _fast_rover, _gauge_step, _robots, cm, scheduler_system

KEY = jax.random.PRNGKey(1)
TICKS = 20


def counted(step):
    """``step`` that records the keys it is called with, and whether the call was a reset (on
    the host, each time it runs: ``tests/host_log.py``)."""
    calls = []

    def wrapped(system, key, obs, carry, params):
        reset = obs is None
        jax.debug.callback(lambda k: calls.append((jnp.asarray(k), reset)), key, ordered=True)
        return step(system, key, obs, carry, params)

    wrapped.sliced = getattr(step, "sliced", False)
    return wrapped, calls


def robots(durations, kernel=None):
    """Two robots and the corridor, robot ``a`` declaring ``durations``: the closed System."""
    (sa, pa), (sb, pb) = _robots(durations)[0], cm.robot(2, 2)
    if kernel is not None:
        pa = dataclasses.replace(pa, actuator={"move": kernel})
    return couple(
        {"a": (sa, pa), "b": (sb, pb), "world": mirror(*cm.world_model(), lower=True)},
        [
            Identify("a.move", "world.u_a"),
            Identify("b.move", "world.u_b"),
            Identify("world.a_pos", "a.me_obs"),
            Identify("world.b_pos", "a.other_obs"),
            Identify("world.status", "a.status"),
            Identify("world.b_pos", "b.me_obs"),
            Identify("world.a_pos", "b.other_obs"),
            Identify("world.status", "b.status"),
        ],
    )


def steps_of(slow):
    return {"a": slow, "b": plan(selection="stochastic"), "world": sample()}


def decisions(moves, rounds_of):
    """The decision rounds implied by the held moves: each option lasts its level's rounds."""
    out, t = [], 0
    while t < len(moves):
        out.append(t)
        t += rounds_of[int(moves[t])]
    return out


def test_a_node_with_durations_decides_only_at_option_boundaries_under_every_order():
    system, params = robots([4.0, 2.0, 4.0])
    for order in ("simultaneous", "sequential", "async"):
        slow, calls = counted(plan(selection="stochastic"))
        traj = run(system, params, TICKS, KEY, steps_of(slow), tick_order=order)
        moves = np.asarray(traj["a"].emitted["move"])
        bounds = decisions(moves, {0: 4, 1: 2, 2: 4})
        for t in bounds:
            assert (moves[t : t + {0: 4, 1: 2, 2: 4}[int(moves[t])]] == moves[t]).all()
        assert [t for t, d in enumerate(np.asarray(traj["a"].decided)) if d] == bounds
        # one reset call and one call per decision; the intended action is held as well
        assert len(calls) == 1 + len(bounds) < 1 + TICKS
        np.testing.assert_array_equal(traj["a"].intended["move"], traj["a"].emitted["move"])
        assert bool(traj["b"].decided.all()) and bool(traj["world"].decided.all())


def test_clocks_converts_the_unit_of_duration_into_rounds():
    system, params = robots([1.5, 1.5, 1.5])
    slow, calls = counted(plan(selection="stochastic"))
    traj = run(system, params, 9, KEY, steps_of(slow), clocks={"a": 2.0})
    # decides on rounds 0, 3 and 6 and holds in between
    assert np.asarray(traj["a"].decided).tolist() == [True, False, False] * 3
    moves = np.asarray(traj["a"].emitted["move"])
    for start in (0, 3, 6):
        assert len(set(moves[start : start + 3].tolist())) == 1
    assert len(calls) == 1 + 3


def test_a_duration_that_is_not_a_whole_number_of_rounds_or_is_under_one_is_refused():
    system, params = robots([1.5, 1.5, 1.5])
    with pytest.raises(ValueError, match="control node 'a.u_move' of 'a'.*not a whole number"):
        run(system, params, 3, KEY, steps_of(plan()))
    system, params = robots([0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="shorter than one round"):
        run(system, params, 3, KEY, steps_of(plan()))


def test_an_actuator_kernel_on_the_node_with_durations_is_refused_though_it_may_be_identity():
    kernel = jnp.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    system, params = robots([2.0, 2.0, 2.0], kernel)
    with pytest.raises(ValueError, match="non-identity actuator"):
        run(system, params, 2, KEY, steps_of(plan()))
    permutation = jnp.asarray(np.eye(3)[[1, 2, 0]])
    system, params = robots([2.0, 2.0, 2.0], permutation)
    with pytest.raises(ValueError, match="non-identity actuator"):
        run(system, params, 2, KEY, steps_of(plan()))
    system, params = robots([2.0, 2.0, 2.0], jnp.eye(3))
    run(system, params, 2, KEY, steps_of(plan()))  # an identity kernel changes nothing


@pytest.mark.parametrize("bad", [0, -1.0, float("inf"), float("nan")])
def test_a_clock_that_is_not_positive_and_finite_is_refused(bad):
    system, params = robots([2.0, 2.0, 2.0])
    with pytest.raises(ValueError, match=r"clocks\['a'\].*is not positive and finite"):
        run(system, params, 2, KEY, steps_of(plan()), clocks={"a": bad})


@pytest.mark.parametrize("bad", [True, "2", [2.0], None])
def test_a_clock_that_is_not_a_number_is_refused(bad):
    system, params = robots([2.0, 2.0, 2.0])
    with pytest.raises(ValueError, match=r"clocks\['a'\].*is not a number"):
        run(system, params, 2, KEY, steps_of(plan()), clocks={"a": bad})


def test_a_scalar_array_is_a_clock():
    system, params = robots([1.5, 1.5, 1.5])
    run(system, params, 4, KEY, steps_of(plan()), clocks={"a": jnp.asarray(2.0)})


def test_clocks_names_a_node_with_durations_only():
    system, params = robots([2.0, 2.0, 2.0])
    with pytest.raises(ValueError, match=r"clocks names \['b'\], whose params declare no dur"):
        run(system, params, 2, KEY, steps_of(plan()), clocks={"b": 1.0})
    with pytest.raises(TypeError, match="clocks is a list"):
        run(system, params, 2, KEY, steps_of(plan()), clocks=[("a", 1.0)])


def test_a_carries_seed_decides_on_the_first_round_and_the_count_is_not_in_the_carry():
    system, params = robots([4.0, 2.0, 4.0])
    slow, calls = counted(plan(selection="stochastic"))
    traj = run(
        system,
        params,
        4,
        KEY,
        steps_of(slow),
        carries={"a": MindCarry()},
        record_carries=True,
    )
    assert bool(traj["a"].decided[0])
    assert len(calls) == 2  # the reset call and the first decision
    assert all(isinstance(c, MindCarry) for c in traj["a"].carries)


def test_a_held_round_calls_no_step_and_the_steps_that_decide_are_called_with_the_node_key():
    """The recording step: a clocked node is called at its reset and on decision rounds only,
    each time with key ``j`` of that round's ``2N + 1`` keys, ``N`` the stepped nodes."""
    system, params = robots([4.0, 2.0, 4.0])
    slow, calls = counted(plan(selection="stochastic"))
    ticks = 12
    traj = run(system, params, ticks, KEY, steps_of(slow))
    decided = np.asarray(traj["a"].decided)
    key, k_reset = jax.random.split(KEY)
    expected = [runner._reset_keys(k_reset, 3)[0]]
    for t in range(ticks):
        key, k_tick = jax.random.split(key)
        if decided[t]:
            expected.append(runner._round_keys(k_tick, 3)[0])
    assert [r for _, r in calls] == [True] + [False] * (len(calls) - 1)
    assert len(calls) == len(expected)
    for (got, _), want in zip(calls, expected, strict=True):
        np.testing.assert_array_equal(jax.random.key_data(got), jax.random.key_data(want))


def test_a_held_node_is_not_written_back_and_its_driven_nodes_keep_their_level():
    system, params, ss = scheduler_system()
    seen = []
    inner = plan(selection="deterministic", replace={"policy_len": 1})

    def mission(view, key, obs, carry, vparams):
        if obs is None:
            return inner(view, key, obs, carry, vparams)
        logged(seen, vparams)
        out, carry, _ = inner(view, key, obs, carry, vparams)
        return out, carry, _bump(view, vparams)  # a learner, every round it is called

    steps = {"agent.mission": mission, "agent.rover": _fast_rover(), "world": _gauge_step}
    traj = run(system, params, 10, KEY, steps, clocks={"agent.mission": 3 / ss.SURVEY_TICKS})
    field = lambda p: np.asarray(p.parameters["agent.rover.field"].concentration)  # noqa: E731
    # the mission learned on its four decisions and the rover on every round, the home holding
    # both; no write-back happened on a held round
    assert len(seen) == 4
    np.testing.assert_array_equal(field(traj.params), field(params) + 4 + 10)
    options = np.asarray(traj["agent.mission"].emitted["option"])
    assert (options[0:3] == options[0]).all() and (options[3:6] == options[3]).all()
    # the rover read the option in force on every round, held or fresh
    np.testing.assert_array_equal(traj["agent.rover"].received["u_opt"][:10], options)
    assert not bool(traj["agent.mission"].decided[1])
    np.testing.assert_array_equal(
        traj["agent.mission"].intended["option"], traj["agent.mission"].emitted["option"]
    )
    assert bool(traj["agent.rover"].decided.all())
    assert set(traj["agent"].decided) == {"mission", "rover"}


def test_the_decision_clock_of_each_leaf_is_its_own():
    """Durations differ across leaves freely: each node is held by its own."""
    system, params = robots([2.0, 2.0, 2.0])
    slow, calls = counted(plan(selection="stochastic"))
    traj = run(system, params, 6, KEY, steps_of(slow))
    assert np.asarray(traj["a"].decided).tolist() == [True, False] * 3
    assert bool(traj["b"].decided.all())


# ---------------------------------------- a clocked node with driven control nodes, custom steps


def _leaf():
    """A leaf with two control nodes (levels 3), two action nodes and one emission."""
    from hybrid_aif import Categorical, ConditionalCategorical
    from hybrid_aif.reads import Emission
    from hybrid_aif.system import System, SystemParams

    copy = jnp.zeros((3, 3, 3, 3)).at[jnp.arange(3), :, :, jnp.arange(3)].set(1.0)
    system = System(
        B_deps=((0, 1, 2), (), ()),
        controls=(1, 2),
        action_names=("a1", "a2"),
        state_names=("s",),
        control_names=("u1", "u2"),
        modality_names=(),
        emissions=(("e", Emission(("s",), "mode", "identity")),),
        policy_len=1,
        epistemic=False,
    )
    params = SystemParams(
        transition={"s": ConditionalCategorical(log_table=jnp.log(copy + 1e-12), parent_ndims=3)},
        initial_state_prior={"s": Categorical.from_probs(jnp.eye(3)[0])},
    )
    return system, params


def _counting(name, calls):
    """A custom step that emits ``t % 3`` on every out-port (``t`` the round, from 1) and logs
    its calls."""

    def step(system, key, obs, carry, params):
        if obs is None:
            return {}, 0, params
        t = carry + 1
        noted(calls, name)
        out = {a: jnp.asarray(t % 3, jnp.int32) for a in system.names.actions if a}
        out.update({e: jnp.asarray(t % 3, jnp.int32) for e, _ in system.emissions})
        return out, t, params

    return step


@pytest.mark.parametrize("order", ["simultaneous", "sequential", "async"])
def test_a_driven_node_that_declares_durations_on_its_own_node_is_held_under_a_custom_step(order):
    """``A`` is driven by ``S`` on ``u1`` and declares durations on its own chosen ``u2``: it
    decides every other round, and ``C``, which ``A`` drives, steps every round on A's standing
    action."""
    leaf, leaf_params = _leaf()
    clocked = (leaf, dataclasses.replace(leaf_params, durations={"u2": jnp.full(3, 2.0)}))
    system, params = couple(
        {"C": _leaf(), "A": clocked, "S": _leaf()},
        [Identify("S.a1", "A.u1"), Identify("A.a2", "C.u1")],
    )
    calls: list = []
    steps = {n: _counting(n, calls) for n in "SAC"}
    traj = run(system, params, 6, KEY, steps, tick_order=order)
    assert np.asarray(traj["A"].decided).tolist() == [True, False] * 3
    assert [c for c in calls if c == "A"] == ["A"] * 3 and [c for c in calls if c == "C"] == [
        "C"
    ] * 6
    # C reads the action A decided and holds through the option, and A's emission stands with it
    assert np.asarray(traj["C"].received["u1"][:6]).tolist() == [1, 1, 2, 2, 0, 0]
    assert np.asarray(traj["A"].emitted["e"]).tolist() == [1, 1, 2, 2, 0, 0]


@pytest.mark.parametrize("order", ["simultaneous", "sequential", "async"])
def test_a_reader_receives_a_held_nodes_emission_standing_on_its_held_rounds(order):
    """``A`` is clocked as above and ``C`` reads A's emission ``e``: on A's held rounds the
    emission keeps its value, and C receives it. C is a source listed first (its node is driven
    by an emission), so under ``"simultaneous"`` it reads at the boundary and under
    ``"sequential"`` before A steps: one round behind, zeros (absent) at round 0. Under
    ``"async"`` C reads before or after A as the round's order fell."""
    leaf, leaf_params = _leaf()
    clocked = (leaf, dataclasses.replace(leaf_params, durations={"u2": jnp.full(3, 2.0)}))
    system, params = couple(
        {"C": _leaf(), "A": clocked, "S": _leaf()},
        [Identify("S.a1", "A.u1"), Identify("A.e", "C.u1")],
    )
    calls: list = []
    traj = run(system, params, 6, KEY, {n: _counting(n, calls) for n in "SAC"}, tick_order=order)
    emitted = np.asarray(traj["A"].emitted["e"]).tolist()
    assert np.asarray(traj["A"].decided).tolist() == [True, False] * 3
    assert emitted == [1, 1, 2, 2, 0, 0]
    received = np.asarray(traj["C"].received["u1"][:6]).tolist()
    if order == "async":
        before = [0, *emitted[:-1]]
        assert all(r in (b, e) for r, b, e in zip(received, before, emitted, strict=True))
    else:
        assert received == [0, 1, 1, 2, 2, 0]


@pytest.mark.parametrize("order", ["simultaneous", "sequential", "async"])
def test_a_source_whose_only_driven_control_node_is_emission_driven_is_clocked(order):
    """``R`` is no node an action drives (its ``u1`` reads ``P``'s emission) and declares
    durations on its own ``u2``: it decides every other round."""
    leaf, leaf_params = _leaf()
    clocked = (leaf, dataclasses.replace(leaf_params, durations={"u2": jnp.full(3, 2.0)}))
    system, params = couple({"P": _leaf(), "R": clocked}, [Identify("P.e", "R.u1")])
    calls: list = []
    steps = {n: _counting(n, calls) for n in "PR"}
    traj = run(system, params, 6, KEY, steps, tick_order=order)
    assert np.asarray(traj["R"].decided).tolist() == [True, False] * 3
    assert bool(traj["P"].decided.all())


# --------------------------------------------------- a merged robot, and the arithmetic of a clock


def _counted_plan(**kwargs):
    """``plan(**kwargs)`` counting its calls after the reset."""
    inner, calls = plan(**kwargs), []

    def step(system, key, obs, carry, params):
        if obs is not None:
            noted(calls, 1)
        return inner(system, key, obs, carry, params)

    step.sliced = True
    return step, calls


@pytest.mark.parametrize("order", ["simultaneous", "sequential", "async"])
def test_a_merged_robot_with_durations_redecides_only_at_option_boundaries(order):
    """The cadence of the robot with durations in a merge of the two robots, run against the
    merge's own mirror: the robot's view declares the durations, the world leaf carries none and
    steps every round, so the positions move once per round under the option standing."""
    (sa, pa), (sb, pb) = _robots([4.0, 2.0, 4.0])
    comp, cp = merge({"a": sa, "b": sb}, {"a": pa, "b": pb}, list(cm.LINKS))
    system, params = with_mirror(comp, cp)
    step_a, calls = _counted_plan(selection="stochastic")
    steps = {"model.a": step_a, "model.b": plan(selection="stochastic"), "world": sample()}
    traj = run(system, params, TICKS, KEY, steps, tick_order=order)
    moves = np.asarray(traj["model.a"].emitted["move"])
    rounds_of = {0: 4, 1: 2, 2: 4}
    bounds = decisions(moves, rounds_of)
    for t in bounds:
        assert (moves[t : t + rounds_of[int(moves[t])]] == moves[t]).all()
    assert len(calls) == len(bounds) < len(moves)
    assert bool(traj["world"].decided.all()) and bool(traj["model.b"].decided.all())


def test_a_float32_duration_times_a_clock_holds_a_whole_number_of_rounds():
    """``33.3 * 10`` in float32 is a whole number of rounds within the clock's tolerance."""
    system, params = robots([33.3, 33.3, 33.3])
    traj = run(system, params, 3, KEY, steps_of(plan()), clocks={"a": 10})
    assert np.asarray(traj["a"].decided).tolist() == [True, False, False]
    #  the option lasts exactly 333 rounds, the float32 product rounded to the whole number
    assert runner._option_rounds(float(jnp.float32(33.3)), 10.0, "duration", "clock") == 333


@pytest.mark.parametrize(
    "good", [np.int64(2), np.float32(2.0), jnp.asarray(2), np.asarray(2.0), 2, 2.0]
)
def test_a_clock_is_any_real_scalar(good):
    system, params = robots([1.0, 1.0, 1.0])
    traj = run(system, params, 4, KEY, steps_of(plan()), clocks={"a": good})
    assert np.asarray(traj["a"].decided).tolist() == [True, False, True, False]


def test_elapsed_is_legal_under_jit_and_cond():
    system, params = _robots([4.0, 2.0, 4.0])[0]

    @jax.jit
    def f(p, level):
        return system.elapsed({"move": level}, p)

    assert float(f(params, jnp.asarray(1))) == 2.0
    assert float(f(params, jnp.asarray(2))) == 4.0

    @jax.jit
    def g(level):
        return jax.lax.cond(
            level > 0, lambda: system.elapsed({"move": level}, params), lambda: jnp.asarray(0.0)
        )

    assert float(g(jnp.asarray(1))) == 2.0


@pytest.mark.parametrize("value", [jnp.asarray([2.0, 0.0, 1.0]), [2.0, -1.0, 1.0]])
def test_a_non_positive_duration_is_refused_when_concrete(value):
    system, params = _robots(value)[0]
    with pytest.raises(ValueError, match="non-positive"):
        system._durations(system._params_in(params, "t"), "t")
    with pytest.raises(ValueError, match="non-positive"):
        system.elapsed({"move": jnp.asarray(1)}, params)
