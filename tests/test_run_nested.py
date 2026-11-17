"""A merged composite with one step per view (design: nesting-and-coupling.md N4), run by ``run``
over a closed System: the merge coupled to its own mirror (``with_mirror``, the merge's leaves
under ``model.``), each leaf given its step. The parameter home is written back each round a step
learns and the views are taken again from it; a driving component's realised decision is routed
into the driven view; a view that declares durations is held for its option.

Under the rulings of 2026-11-10 a driven view is handed the level its driver realised in the same
round under every order (design 1.4, 4.4): ``run`` steps a driven node after its driver whatever
the listing, and under ``"simultaneous"`` a control node driven by an action node is read as
realised this round. The composite form handed the level standing when the round began under
``"simultaneous"``, and under ``"sequential"`` the level of the round before to a driven view
listed before its driver; those tests are inverted here. Pins: ``tests/data/run_parity.npz``
(the learner on the shared parameter variable, the same-step zoo), ``tests/data/clock_parity.npz``
(the held scheduler with a fast learner) and ``tests/data/migration_particle.npz`` (a held robot
inside a merge, one mind of the wired form, as release 0.19.0 recorded it).
"""

import dataclasses
import functools
import importlib.util
import pathlib
from unittest import mock

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import ConditionalCategorical, ConditionalDirichlet
from hybrid_aif.composition import (
    Identify,
    Link,
    couple,
    joint_planner,
    merge,
    plan,
    run,
    sample,
    subsystem,
    with_mirror,
    write_back,
)
from hybrid_aif.composition.merge import _inputs_of, retake
from hybrid_aif.system.mirror import mirror
from tests.host_log import logged, unflagged
from tests.test_merge import OBSERVATION_BINDINGS, agent_and_world, merged
from tests.test_merge_gaussian_control import mission as mission_tier
from tests.test_merge_gaussian_control import rover as lg_rover
from tests.test_sibling_read import GRID, READ, SOURCE, _composite, _rover, _scheduler
from tests.test_subsystem_conditional_target import _conditional_target
from tests.test_subsystem_conditional_target import _rover as _target_rover

KEY = jax.random.PRNGKey(0)
NODE = "agent.u_location"
ACTION = "agent.a_location"
SAME_STEP = Link("agent.a_location", "world.u_location")

_PATH = pathlib.Path(__file__).resolve().parents[1] / "examples" / "compose_minds.py"
_SPEC = importlib.util.spec_from_file_location("compose_minds", _PATH)
cm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cm)

M = "model."  # the prefix of the merge's leaves in the closed System ``with_mirror`` makes
LEVELS = (1, 3, 2, 4, 0, 2)  # the driver's levels, one per round, in the routing tests
FIXTURE = pathlib.Path(__file__).parent / "data" / "migration_particle.npz"
PINNED_LEVELS = [3, 4, 1, 1]  # the routed composite's actions under run's own keys
PINNED_LOCATION = [0, 3, 3, 3, 3]


def _bump_any(view, params):
    """The params of ``view`` with the shared field one count higher, wherever it is named. Called
    inside a traced step: a first import there would build the imported modules' arrays on the
    trace, so it is evaluated at compile time."""
    with jax.ensure_compile_time_eval():
        from tests.test_run_parity import _bump

    return _bump(view, params)


def _same(a, b):
    la, lb = jax.tree_util.tree_flatten(a), jax.tree_util.tree_flatten(b)
    return la[1] == lb[1] and all(
        np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la[0], lb[0], strict=True)
    )


# ------------------------------------------------------------- the zoo pair: routing


def _zoo():
    comp, cp = joint_planner(*merged(bindings=[SAME_STEP, *OBSERVATION_BINDINGS]))
    obs, _ = comp.as_env(cp)[0].reset(KEY, comp.as_env(cp)[1])
    return comp, cp, obs


def test_the_view_that_reads_a_partner_s_node_names_it():
    comp, _, _ = _zoo()
    assert tuple(_inputs_of(comp, "world")) == (ACTION,)
    assert tuple(_inputs_of(comp, "agent")) == ()


def _level_view(comp, cp, level):
    return subsystem(comp, cp, "world", inputs={ACTION: level} if level is not None else None)[1]


def _closed_zoo(actuator=None, durations=None, world_first=False):
    """The zoo pair merged with the agent's action node driving the world's control node (the
    same-step identification), coupled to its own mirror: ``(system, params)``."""
    (agent, ap), (world, wp) = agent_and_world(policy_len=1)
    if durations is not None:
        ap = dataclasses.replace(ap, durations={"u_location": jnp.full(5, float(durations))})
    parts = {"agent": (agent, ap), "world": (world, wp)}
    if world_first:
        parts = {"world": parts["world"], "agent": parts["agent"]}
    comp, cp = merge(
        {n: s for n, (s, _) in parts.items()},
        {n: p for n, (_, p) in parts.items()},
        [SAME_STEP, *OBSERVATION_BINDINGS],
    )
    if actuator is not None:
        cp = dataclasses.replace(cp, actuator={ACTION: actuator})
    return with_mirror(comp, cp)


def _view_at(system, params, level):
    """The world view's params sliced at ``level`` of the agent's action node (``None``: before
    it is realised, averaged uniformly)."""
    return subsystem(system, params, f"{M}world", inputs={f"{M}{ACTION}": level})[1]


def _viewer(log):
    """The world model's step: the params of the view it is handed, logged, and nothing chosen;
    marked ``sliced`` so ``run`` hands it the view sliced at its driver's realised level."""

    def step(system, key, obs, carry, params):
        logged(log, params)
        return {}, carry, params

    step.sliced = True
    return step


def _levels(levels):
    """The agent's step: ``levels[t]`` on its action node at round ``t``."""

    def step(system, key, obs, carry, params):
        if obs is None:
            return {}, 0, params
        return {f"{M}{ACTION}": jnp.asarray(levels, jnp.int32)[carry]}, carry + 1, params

    return step


@pytest.mark.parametrize("order", ["simultaneous", "sequential", "async"])
def test_a_driver_listed_first_is_seen_by_the_driven_view_the_same_round(order):
    system, params = _closed_zoo()
    log: list = []
    steps = {
        f"{M}agent": plan(selection="deterministic"),
        f"{M}world": _viewer(log),
        "world": sample(),
    }
    traj = run(system, params, 3, KEY, steps, tick_order=order)
    levels = [int(a) for a in traj[f"{M}agent"].emitted["a_location"]]
    assert _same(log[0], _view_at(system, params, None))  # the reset call: nothing realised
    for t, level in enumerate(levels):
        assert _same(log[t + 1], _view_at(system, params, level))


@pytest.mark.parametrize("order", ["simultaneous", "sequential", "async"])
def test_a_driven_view_listed_before_its_driver_still_sees_the_level_of_the_same_round(order):
    """``run`` steps a driven node after its driver whatever the listing (the derived order), so
    the world model listed first is handed the agent's level of this round. The composite form
    of release 0.19.0 handed a driven view listed first the level of the round before."""
    system, params = _closed_zoo(world_first=True)
    assert [n for n, _ in system.subsystems][1:3] == [f"{M}world", f"{M}agent"]
    log: list = []
    steps = {f"{M}agent": _levels(LEVELS), f"{M}world": _viewer(log), "world": sample()}
    run(system, params, len(LEVELS), KEY, steps, tick_order=order)
    for t, level in enumerate(LEVELS):
        assert _same(log[t + 1], _view_at(system, params, level))


def test_one_compiled_program_serves_every_routed_level():
    comp, cp, _ = _zoo()
    traces: list = []

    @eqx.filter_jit
    def probe(params):
        traces.append(1)
        return jnp.zeros(())

    views = [_level_view(comp, cp, level) for level in (None, *range(5))]
    for view in views:
        probe(view)
    assert len(traces) == 1  # the uniform round-0 view shares the program
    systems = [subsystem(comp, cp, "world", inputs={ACTION: lv})[0] for lv in (0, 3)]
    assert eqx.tree_equal(*systems)


# ------------------------------------------------------ the shared field: write-back


def _bump(view, params):
    """``params`` with the field's concentrations one higher: what a learner returns."""
    held = view._params_in(params, "test")
    field = held.parameters["rover.field"]
    new = ConditionalDirichlet(concentration=field.concentration + 1.0, parent_ndims=1)
    out = dataclasses.replace(held, parameters={**held.parameters, "rover.field": new})
    for slot, i, home in held.references:
        container = list(getattr(out, slot))
        container[i] = out.parameters[home]
        out = dataclasses.replace(out, **{slot: tuple(container)})
    return view._params_out(out)


def _shared():
    comp, cp = _composite(shared=True, bindings=[Identify("rover.field", "mission.field")])
    rover, mission = subsystem(comp, cp, "rover")[0], subsystem(comp, cp, "mission")[0]
    return comp, cp, rover, mission


def _closed_shared(listing=("rover", "mission")):
    """The rover and the scheduler sharing the parameter variable ``field`` (the scheduler reading
    the rover's cell), listed in ``listing``, coupled to their own mirror."""
    parts = {"rover": _rover(True), "mission": _scheduler(True)}
    parts = {n: parts[n] for n in listing}
    comp, cp = merge(
        {n: s for n, (s, _) in parts.items()},
        {n: p for n, (_, p) in parts.items()},
        [Link(SOURCE, READ, realise="mode", kernel=GRID), Identify("rover.field", "mission.field")],
    )
    return with_mirror(comp, cp)


def _home_field(params):
    return np.asarray(params.parameters[f"{M}rover.field"].concentration)


def _learning(log=None):
    """``plan()`` that also returns the shared field one count higher: a learner."""
    inner = plan()

    @functools.wraps(inner)
    def step(system, key, obs, carry, params):
        out, carry, _ = inner(system, key, obs, carry, params)
        if obs is None:
            return out, carry, params
        if log is not None:
            logged(log, params)
        return out, carry, _bump_any(system, params)

    return step


def _watching(seen):
    """``plan()`` that logs the params of the view it is handed."""
    inner = plan()

    @functools.wraps(inner)
    def step(system, key, obs, carry, params):
        if obs is not None:
            logged(seen, params)
        return inner(system, key, obs, carry, params)

    return step


def _shared_steps(rover, mission):
    return {f"{M}rover": rover, f"{M}mission": mission, "world": sample()}


@pytest.mark.parametrize("listing, offset", [(("rover", "mission"), 1), (("mission", "rover"), 0)])
def test_a_learning_view_is_written_back_each_round_and_the_other_view_is_taken_again(
    listing, offset
):
    """Under ``"sequential"`` the field the scheduler's view holds is the home as it stands at
    its turn: this round's learning when the rover steps before it, the round before's when
    after. Under ``"simultaneous"`` every write-back lands at the round's boundary, so the
    scheduler sees the round before's learning whatever the listing (the composite form applied
    write-backs live, as ``"sequential"`` does here)."""
    system, params = _closed_shared(listing)
    for order, shift in (("simultaneous", 0), ("sequential", offset)):
        seen: list = []
        traj = run(
            system, params, 3, KEY, _shared_steps(_learning(), _watching(seen)), tick_order=order
        )
        assert np.array_equal(_home_field(traj.params), _home_field(params) + 3)
        for t in range(3):
            assert np.array_equal(
                np.asarray(seen[t].parameters[f"{M}rover.field"].concentration),
                _home_field(params) + t + shift,
            )


@pytest.mark.parametrize("order", ["simultaneous", "sequential"])
def test_two_learners_of_one_dirichlet_variable_under_simultaneous_sum_their_deltas(order):
    """Both the rover and the scheduler learn the shared field, one count each a round. Under
    ``"simultaneous"`` every write-back lands at the round's boundary: each view is the home as
    the round began, whatever the listing, and the two deltas of a round are summed (Dirichlet
    counts), two counts a round. Under ``"sequential"`` write-backs are live: the node that steps
    second sees the first's count of the round, and the home is the same."""
    homes = []
    for listing in (("rover", "mission"), ("mission", "rover")):
        system, params = _closed_shared(listing)
        rover_saw, mission_saw = [], []
        steps = _shared_steps(_learning(rover_saw), _learning(mission_saw))
        traj = run(system, params, 3, KEY, steps, tick_order=order)
        base = _home_field(params)
        np.testing.assert_array_equal(_home_field(traj.params), base + 6)
        homes.append(_home_field(traj.params))

        def counts(seen, base=base):
            return [float((_home_field(p) - base).ravel()[0]) for p in seen]

        if order == "simultaneous":
            assert counts(rover_saw) == counts(mission_saw) == [0.0, 2.0, 4.0]
        else:
            first, second = (
                (rover_saw, mission_saw)
                if listing[0] == "rover"
                else (
                    mission_saw,
                    rover_saw,
                )
            )
            assert counts(first) == [0.0, 2.0, 4.0] and counts(second) == [1.0, 3.0, 5.0]
    np.testing.assert_array_equal(homes[0], homes[1])


def test_two_learners_of_one_variable_that_is_not_counts_are_refused_under_simultaneous():
    """A shared variable whose form is not Dirichlet counts has no exact combination of two
    write-backs of one round, so ``"simultaneous"`` refuses two nodes that may both learn it,
    before the first round, naming the variable and the nodes; ``"sequential"`` runs them, and
    so does ``"simultaneous"`` with one learner."""
    system, params = _closed_shared()
    name = f"{M}rover.field"
    counts = params.parameters[name].concentration
    table = ConditionalCategorical(
        log_table=jnp.log(counts / counts.sum(axis=0, keepdims=True)), parent_ndims=1
    )
    held = system._params_in(params, "test")
    held = dataclasses.replace(held, parameters={**held.parameters, name: table})
    for slot, i, home in held.references:
        container = list(getattr(held, slot))
        container[i] = held.parameters[home]
        held = dataclasses.replace(held, **{slot: tuple(container)})
    params = system._params_out(held)
    steps = _shared_steps(plan(learn=True), plan(learn=True))
    with pytest.raises(
        ValueError, match=rf"'{name}' \(ConditionalCategorical\).*'{M}rover' and '{M}mission'"
    ):
        run(system, params, 1, KEY, steps)
    run(system, params, 1, KEY, steps, tick_order="sequential")
    run(system, params, 1, KEY, _shared_steps(plan(learn=True), plan()))


def _hand_loop(system, params, rounds):
    """The home after ``rounds`` write-backs of the rover's view, one count each, by hand."""
    home = params
    for _ in range(rounds):
        taken = subsystem(system, home, f"{M}rover")[1]
        view = subsystem(system, home, f"{M}rover")[0]
        home = write_back(system, home, f"{M}rover", taken, _bump_any(view, taken))
    return home


def test_the_home_after_each_round_is_the_hand_loop_of_write_backs_bitwise():
    """``Trajectory.params`` after ``t`` rounds is the home ``t`` hand write-backs give. The
    learner on this composite is pinned against the composite form by
    ``test_run_parity::test_a_learning_view_is_written_back_to_the_home_as_the_composite_run_did``.
    """
    system, params = _closed_shared()
    for rounds in range(4):
        traj = run(system, params, rounds, KEY, _shared_steps(_learning(), plan()))
        assert _same(traj.params, _hand_loop(system, params, rounds))


def test_retake_is_what_subsystem_would_return_now():
    comp, cp, rover, mission = _shared()
    for name, view in (("mission", mission), ("rover", rover)):
        before = subsystem(comp, cp, name)[1]
        taken = subsystem(comp, cp, "rover")[1]
        home = write_back(comp, cp, "rover", taken, _bump(rover, taken))
        assert _same(retake(comp, view, before, home), subsystem(comp, home, name)[1])


def test_write_back_refuses_a_before_taken_since_the_home_moved():
    """The rule ``run`` keeps by construction: it hands each step a view taken from the current
    home, so its ``before`` is never stale."""
    system, params = _closed_shared()
    stale = subsystem(system, params, f"{M}rover")
    traj = run(system, params, 1, KEY, _shared_steps(_learning(), plan()))
    with pytest.raises(ValueError, match="taken before"):
        write_back(system, traj.params, f"{M}rover", stale[1], _bump_any(stale[0], stale[1]))


def test_a_step_that_does_not_learn_writes_nothing_and_a_learner_beside_it_still_learns():
    system, params = _closed_shared()
    seen = []
    inner = plan()

    def plain(system, key, obs, carry, params):
        if obs is not None:
            logged(seen, obs)
        action, carry, same = inner(system, key, obs, carry, params)
        assert same is params
        return action, carry, same

    traj = run(system, params, 1, KEY, _shared_steps(_learning(), plain))
    assert unflagged(seen[0])[f"{M}mission.cell_obs"] is None
    assert np.array_equal(_home_field(traj.params), _home_field(params) + 1)
    idle = run(system, params, 1, KEY, _shared_steps(plan(), plain))
    assert idle.params is params  # nothing learned: the very object handed in


# --------------------------------------------- routing under run's three orders


def _routed(order, ticks=6, world_first=False):
    system, params = _closed_zoo(world_first=world_first)
    log: list = []
    steps = {f"{M}agent": _levels(LEVELS), f"{M}world": _viewer(log), "world": sample()}
    traj = run(system, params, ticks, KEY, steps, tick_order=order)
    return system, params, log, traj


def test_run_simultaneous_hands_the_view_the_level_realised_in_the_same_round():
    """Design 1.4 and 4.4 (flag 4): under ``"simultaneous"`` the driven view is sliced at the
    level its driver realised this round, a control node driven by an action node being the read
    phase two takes as it stands. The composite form handed the level standing when the round
    began."""
    system, params, log, traj = _routed("simultaneous")
    assert _same(log[0], _view_at(system, params, None))  # the reset call: uniform
    for t, level in enumerate(LEVELS):
        assert _same(log[t + 1], _view_at(system, params, level))
    assert traj.params is params  # nothing learned


def test_run_sequential_hands_a_driver_listed_first_its_choice_of_this_round():
    system, params, log, _ = _routed("sequential")
    for t, level in enumerate(LEVELS):
        assert _same(log[t + 1], _view_at(system, params, level))


def test_run_sequential_hands_a_driven_view_listed_first_its_driver_s_choice_of_this_round():
    """The derived order steps the driven world model after the agent though it is listed
    first; the composite form handed it the level of the round before."""
    system, params, log, _ = _routed("sequential", world_first=True)
    for t, level in enumerate(LEVELS):
        assert _same(log[t + 1], _view_at(system, params, level))


def test_run_async_hands_a_routed_view_the_level_of_the_same_round():
    """Async permutes the sources each round (here the agent and the world model's own
    mirror), and the driven world model steps after the agent in every permutation."""
    system, params, log, _ = _routed("async")
    assert len(log) == 1 + len(LEVELS)
    for t, level in enumerate(LEVELS):
        assert _same(log[t + 1], _view_at(system, params, level))


def test_a_composite_with_a_routed_view_pins_its_trajectory():
    """The joint-planned zoo (its world model reading the agent's node as a partner input)
    against its own mirror, the agent planning deterministically: its actions and the world's
    position. Under the composite form's key layout they are what that form recorded (the first
    four rounds of ``same_step_zoo/sequential`` in ``tests/data/run_parity.npz``, which pinned
    ``[3, 4, 1, 1]`` and ``[0, 3, 3, 3, 3]``); under ``run``'s own keys they are pinned below, and
    are the same, since the planner is deterministic and the world's tables exact."""
    from tests.test_run_parity import FIXTURE, _run_old_keys, idle, states

    comp, cp, _ = _zoo()
    system, params = with_mirror(comp, cp)
    steps = {f"{M}agent": plan(selection="deterministic"), f"{M}world": idle, "world": sample()}
    traj = _run_old_keys(
        2, system, params, 4, KEY, steps, tick_order="sequential", record_carries=True
    )
    levels = [int(a) for a in traj[f"{M}agent"].emitted["a_location"]]
    assert levels == FIXTURE["same_step_zoo/sequential/agent.a_location"][:4].tolist()
    assert levels == [3, 4, 1, 1]
    location = states(traj, "world.world.location").tolist()
    assert location == FIXTURE["same_step_zoo/sequential/world.location"][:5].tolist()
    traj = run(system, params, 4, KEY, steps, tick_order="sequential", record_carries=True)
    assert [int(a) for a in traj[f"{M}agent"].emitted["a_location"]] == PINNED_LEVELS
    assert states(traj, "world.world.location").tolist() == PINNED_LOCATION


@pytest.mark.parametrize("tick_order", ["simultaneous", "sequential", "async"])
def test_run_writes_a_learner_back_and_its_sibling_sees_it(tick_order):
    for listing, offset in ((("rover", "mission"), 1), (("mission", "rover"), 0)):
        system, params = _closed_shared(listing)
        seen: list = []
        traj = run(
            system,
            params,
            3,
            KEY,
            _shared_steps(_learning(), _watching(seen)),
            tick_order=tick_order,
        )
        assert np.array_equal(_home_field(traj.params), _home_field(params) + 3)
        for t in range(3):
            got = np.asarray(seen[t].parameters[f"{M}rover.field"].concentration)
            if tick_order == "async":  # the learner first or last, as the round's order drew
                assert got[0, 0] - _home_field(params)[0, 0] in (t, t + 1)
            elif tick_order == "simultaneous":  # write-backs land at the round's boundary
                assert np.array_equal(got, _home_field(params) + t)
            else:
                assert np.array_equal(got, _home_field(params) + t + offset)


def test_run_leaves_the_world_on_the_starting_params_and_its_home_is_the_hand_loop():
    system, params = _closed_shared()
    traj = run(system, params, 4, KEY, _shared_steps(_learning(), plan()))
    assert _same(traj.params, _hand_loop(system, params, 4))
    for field in ("likelihood", "transition", "initial_state_prior", "emission"):
        world = {k: v for k, v in getattr(params, field).items() if k.startswith("world.")}
        got = {k: v for k, v in getattr(traj.params, field).items() if k.startswith("world.")}
        assert _same(got, world), field


def _held_robots():
    """The two robots merged, ``a`` declaring durations ``[4, 2, 4]``, coupled as one component
    ``m`` to the corridor: ``(system, params)``."""
    (sa, pa), (sb, pb) = cm.robot(2, 2), cm.robot(2, 2)
    pa = dataclasses.replace(pa, durations={"u_move": jnp.asarray([4.0, 2.0, 4.0])})
    comp, cp = merge({"a": sa, "b": sb}, {"a": pa, "b": pb}, list(cm.LINKS))
    return couple(
        {"m": (comp, cp), "world": cm.world((2, 2))},
        [
            Identify("m.a.move", "world.u_a"),
            Identify("m.b.move", "world.u_b"),
            Identify("world.a_pos", "m.a.me_obs"),
            Identify("world.b_pos", "m.a.other_obs"),
            Identify("world.status", "m.a.status"),
            Identify("world.b_pos", "m.b.me_obs"),
            Identify("world.a_pos", "m.b.other_obs"),
            Identify("world.status", "m.b.status"),
        ],
    )


def _wired_layout():
    """The keys of the merge as the one mind of the wired form (release 0.19.0), for the leaves
    ``m.a``, ``m.b`` and the world: the wired form split a round key in four, stepped the merge
    with key 0 (its first component with that key, the next with ``fold_in(key, 1)``), realised
    with key 1 (sequential) or key 3 (simultaneous) and stepped the process with key 3, which it
    reset with the reset key."""
    from tests.test_run_parity import runner

    def round_keys(key, count):
        wired = list(jax.random.split(key, 4))
        mind, last = wired[0], wired[3]
        return [mind, jax.random.fold_in(mind, 1), last, wired[1], wired[1], last, last]

    def reset_keys(key, count):
        return [jax.random.fold_in(key, 0), jax.random.fold_in(key, 1), key]

    return (
        mock.patch.object(runner, "_round_keys", round_keys),
        mock.patch.object(runner, "_reset_keys", reset_keys),
    )


@pytest.mark.parametrize("order", ["simultaneous", "sequential"])
def test_a_merged_robot_with_durations_coupled_to_a_world_decides_once_per_option(order):
    """The robots' merge coupled to the corridor, robot ``a`` declaring durations: ``run`` holds
    ``m.a`` for its option, ``m.b`` and the world step every round. Under the key layout of the
    merge as the one mind of the wired form, ``a`` held for its option inside it, the moves, the
    corridor and the number of decisions are what that form recorded."""
    from tests.test_run_parity import states

    system, params = _held_robots()
    steps = {
        "m.a": plan(selection="stochastic"),
        "m.b": plan(selection="stochastic"),
        "world": sample(),
    }
    first, second = _wired_layout()
    with first, second:
        rec = run(system, params, 20, KEY, steps, tick_order=order, record_carries=True)
    moves = np.asarray(rec["m.a"].emitted["move"])
    rounds, t, decisions = {0: 4, 1: 2, 2: 4}, 0, 0
    while t < len(moves):
        decisions += 1
        assert (moves[t : t + rounds[int(moves[t])]] == moves[t]).all()
        t += rounds[int(moves[t])]
    assert int(rec["m.a"].decided.sum()) == decisions < len(moves)
    assert bool(rec["m.b"].decided.all()) and bool(rec["world"].decided.all())
    fixture = np.load(FIXTURE)
    tag = f"held_particle/{order}"
    np.testing.assert_array_equal(moves, fixture[f"{tag}/a.move"])
    np.testing.assert_array_equal(rec["m.b"].emitted["move"], fixture[f"{tag}/b.move"])
    np.testing.assert_array_equal(states(rec, "world.a"), fixture[f"{tag}/world/a"])
    np.testing.assert_array_equal(states(rec, "world.b"), fixture[f"{tag}/world/b"])
    assert decisions == int(fixture[f"{tag}/calls"])


# ----------------------------------------------------------------------------- refusals


def _rewriting(field):
    """The rover's ``plan()`` returning its params with ``field`` changed: the noise of its own
    likelihood ``obs``, or its precision."""
    inner = plan()

    @functools.wraps(inner)
    def step(system, key, obs, carry, params):
        out, carry, _ = inner(system, key, obs, carry, params)
        if obs is None:
            return out, carry, params
        if field == "likelihood":
            lik = params.likelihood[f"{M}rover.obs"]
            noisier = dataclasses.replace(lik, noise=lik.noise * 2.0)
            changed = {"likelihood": {**params.likelihood, f"{M}rover.obs": noisier}}
        else:
            changed = {"precision": jnp.asarray(2.0)}
        return out, carry, dataclasses.replace(params, **changed)

    return step


def test_a_learned_own_table_is_written_back_under_its_name():
    """A table a leaf's view holds of its own (here the rover's likelihood) is written back to
    the home under its namespaced name (``plan(learn=...)``'s rule)."""
    system, params = _closed_shared()
    traj = run(system, params, 1, KEY, _shared_steps(_rewriting("likelihood"), plan()))
    got = traj.params.likelihood[f"{M}rover.obs"].noise
    np.testing.assert_allclose(got, params.likelihood[f"{M}rover.obs"].noise * 2.0)


def test_a_wire_into_a_modality_a_read_supplies_is_refused():
    """A modality the merge already feeds from a component's emission (the scheduler's cell,
    read from the rover) is an in-port identified once: a second identification is refused."""
    comp, cp = _composite()
    world = mirror(*_scheduler(), lower=True)
    with pytest.raises(ValueError, match="'m.mission.cell_obs' already observes a variable"):
        couple(
            {"m": (comp, cp), "src": world},
            [Identify("src.cell_obs", "m.mission.cell_obs")],
        )


def _two_tier_closed(listing=("mission", "rover")):
    """The two tiers merged (the option driving the rover's ``u_opt``, the rover's target on
    ``x`` conditional on it), listed in ``listing``, coupled to their own mirror."""
    parts = {
        "mission": mission_tier(),
        "rover": _target_rover(_conditional_target(), (("x", ("u_opt",)),)),
    }
    parts = {n: parts[n] for n in listing}
    comp, cp = merge(
        {n: s for n, (s, _) in parts.items()},
        {n: p for n, (_, p) in parts.items()},
        [Link("mission.option", "rover.u_opt")],
    )
    return with_mirror(comp, cp)


def _option_step(system, key, obs, carry, params):
    return ({} if obs is None else {f"{M}mission.option": jnp.asarray(1, jnp.int32)}), carry, params


def _rover_step(seen):
    def step(system, key, obs, carry, params):
        if obs is None:
            return {}, carry, params
        logged(seen, obs[f"{M}rover.u_opt"])
        return {f"{M}rover.a_v": jnp.zeros(2, jnp.float32)}, carry, params

    return step


@pytest.mark.parametrize("listing", [("mission", "rover"), ("rover", "mission")])
@pytest.mark.parametrize("order", ["simultaneous", "sequential", "async"])
def test_a_view_with_a_conditional_target_reads_its_chooser_s_level_of_the_round(order, listing):
    """``run`` steps the rover after the scheduler whose option drives it, under every order and
    whatever the listing, so the rover reads the option realised this round. The composite form
    of release 0.19.0 refused this composite under ``"simultaneous"`` and ``"async"``, and under
    ``"sequential"`` with the scheduler listed second."""
    system, params = _two_tier_closed(listing)
    seen: list = []
    steps = {f"{M}mission": _option_step, f"{M}rover": _rover_step(seen), "world": sample()}
    traj = run(system, params, 2, KEY, steps, tick_order=order)
    assert [int(x) for x in traj[f"{M}mission"].emitted["option"]] == [1, 1]
    assert seen == [1, 1]


def _cycling_option(system, key, obs, carry, params):
    """The scheduler's option cycling through levels 0 and 1, one a round."""
    if obs is None:
        return {}, 0, params
    return {f"{M}mission.option": jnp.asarray(carry % 2, jnp.int32)}, carry + 1, params


def _sliced_target_log(inner, log):
    """``inner`` (a sliced step) that also logs the location of the rover's target in the view
    it is handed each round."""

    @functools.wraps(inner)
    def step(system, key, obs, carry, params):
        if obs is not None:
            logged(log, params.state_targets[f"{M}rover.x"].loc)
        return inner(system, key, obs, carry, params)

    return step


def _sliced_idle(system, key, obs, carry, params):
    return ({} if obs is None else {f"{M}rover.a_v": jnp.zeros(2, jnp.float32)}), carry, params


_sliced_idle.sliced = True


@pytest.mark.parametrize("kind", ["plan", "custom"])
@pytest.mark.parametrize("order", ["simultaneous", "sequential", "async"])
def test_a_sliced_step_runs_a_view_whose_target_is_conditional_on_its_driven_node(order, kind):
    """The rover's Gaussian target is conditional on ``u_opt``, which the scheduler's option
    drives (the survey's tiered rover). The view averaged over the option is refused (the target
    would be a mixture of Gaussians), so ``run`` reads durations from the home's ``parts`` and
    hands the reset call the view at the option's first level, which a reset call does not read;
    each round the sliced step sees the target at the option realised that round."""
    system, params = _two_tier_closed()
    log: list = []
    rover = _sliced_target_log(plan() if kind == "plan" else _sliced_idle, log)
    steps = {f"{M}mission": _cycling_option, f"{M}rover": rover, "world": sample()}
    traj = run(system, params, 4, KEY, steps, tick_order=order)
    options = [int(o) for o in traj[f"{M}mission"].emitted["option"]]
    assert options == [0, 1, 0, 1]
    for seen, level in zip(log, options, strict=True):
        view = subsystem(system, params, f"{M}rover", inputs={f"{M}mission.option": level})[1]
        np.testing.assert_array_equal(seen, view.state_targets[f"{M}rover.x"].loc)
    assert not np.array_equal(log[0], log[1])


def _two_tier_lg_closed():
    """The two tiers merged with the rover's linear-Gaussian transition of ``x`` reading the
    driven ``u_opt`` (innovations per option), coupled to their own mirror."""
    parts = {"mission": mission_tier(), "rover": lg_rover(per_level=True, with_float=True)}
    comp, cp = merge(
        {n: s for n, (s, _) in parts.items()},
        {n: p for n, (_, p) in parts.items()},
        [Link("mission.option", "rover.u_opt")],
    )
    return with_mirror(comp, cp)


@pytest.mark.parametrize("order", ["simultaneous", "sequential", "async"])
def test_a_sliced_step_runs_a_view_whose_transition_reads_its_driven_node(order):
    """The rover's linear-Gaussian transition reads ``u_opt``, which the option drives: the view
    averaged over the option is refused (a mixture of Gaussians, ``MixtureRefusal``), so the
    reset call takes the view at the option's first level, and each round the sliced step's view
    is the rover's view at the option realised that round."""
    system, params = _two_tier_lg_closed()
    log: list = []

    def rover(view, key, obs, carry, p):
        if obs is not None:
            logged(log, p)
        return ({} if obs is None else {f"{M}rover.a_v": jnp.zeros(2, jnp.float32)}), carry, p

    rover.sliced = True
    steps = {f"{M}mission": _cycling_option, f"{M}rover": rover, "world": sample()}
    traj = run(system, params, 4, KEY, steps, tick_order=order)
    options = [int(o) for o in traj[f"{M}mission"].emitted["option"]]
    assert options == [0, 1, 0, 1]
    for seen, level in zip(log, options, strict=True):
        assert _same(
            seen, subsystem(system, params, f"{M}rover", inputs={f"{M}mission.option": level})[1]
        )


def test_durations_on_the_systems_own_params_hold_the_node_as_its_view_declares():
    """``run`` reads a node's durations with ``subsystem``'s precedence: the closed System's own
    ``params.durations`` where it holds any, the leaf's ``parts`` otherwise. Durations of 2 and
    3 rounds on the scheduler's option, alternating 0 and 1, hold it for 2, then 3 rounds."""
    system, params = _two_tier_closed()
    params = dataclasses.replace(
        params, durations={f"{M}mission.u_option": jnp.asarray([2.0, 3.0])}
    )
    steps = {f"{M}mission": _cycling_option, f"{M}rover": _sliced_idle, "world": sample()}
    traj = run(system, params, 6, KEY, steps)
    assert [bool(d) for d in traj[f"{M}mission"].decided] == [True, False, True, False, False, True]


def test_a_kernel_on_a_node_another_view_reads_slices_the_view_at_the_realised_level():
    """Design 1.4 and 4.4: a driven view is sliced at the level its driver realised, so an
    actuator kernel on the driver is allowed; the composite form of release 0.19.0 refused it,
    since it handed on the intended level."""
    kernel = 0.4 * jnp.eye(5) + 0.12
    system, params = _closed_zoo(actuator=kernel)
    log: list = []
    steps = {f"{M}agent": plan(), f"{M}world": _viewer(log), "world": sample()}
    traj = run(system, params, 6, KEY, steps)
    realised = [int(a) for a in traj[f"{M}agent"].emitted["a_location"]]
    intended = [int(a) for a in traj[f"{M}agent"].intended["a_location"]]
    assert realised != intended  # the kernel moved some draw
    for t, level in enumerate(realised):
        assert _same(log[t + 1], _view_at(system, params, level))


def test_plan_refuses_an_all_masked_legal_on_its_first_call():
    system, params = _closed_zoo()
    legal = {f"{M}{NODE}": jnp.zeros(5, bool)}
    steps = {f"{M}agent": plan(legal=legal), f"{M}world": _viewer([]), "world": sample()}
    with pytest.raises(ValueError, match="masks every level"):
        run(system, params, 1, KEY, steps)


def test_a_run_refuses_params_that_differ_outside_the_parameter_variables_and_own_tables():
    system, params = _closed_shared()
    with pytest.raises(ValueError, match=r"'model.rover'.*\['precision'\].*Parameter"):
        run(system, params, 1, KEY, _shared_steps(_rewriting("precision"), plan()))


# ------------------------------------------------------------- held driver, survey shape


def test_a_held_drivers_level_reaches_the_driven_view_on_held_rounds():
    """The agent declares three rounds per option: ``run`` holds it, and the world model's view
    is sliced at the standing level on held rounds as on decision rounds."""
    system, params = _closed_zoo(durations=3.0)
    log: list = []
    steps = {
        f"{M}agent": plan(selection="deterministic"),
        f"{M}world": _viewer(log),
        "world": sample(),
    }
    traj = run(system, params, 4, KEY, steps)
    levels = [int(a) for a in traj[f"{M}agent"].emitted["a_location"]]
    assert np.asarray(traj[f"{M}agent"].decided).tolist() == [True, False, False, True]
    assert levels[0] == levels[1] == levels[2]
    for given, level in zip(log[1:], levels, strict=True):
        assert _same(given, _view_at(system, params, level))


def test_a_slow_mind_with_durations_and_a_fast_learner_sharing_a_parameter():
    """N4 without the camera, under ``run``: the scheduler declares durations and is held three
    rounds per option (decisions at rounds 0, 3, 6); the fast rover's write-back is visible in the
    slow view at each decision; the rover's view is handed the option standing, held or fresh;
    the rover's float action node is emitted as a float. Pinned against the run release 0.19.0
    recorded by ``test_clock_parity::test_a_held_scheduler_with_a_fast_learner_is_the_recorded_
    run_bitwise``."""
    from tests.test_clock_parity import _fast_rover, _gauge_step, scheduler_system

    system, params, ss = scheduler_system()
    slow_seen, fast_seen = [], []
    slow = plan(selection="deterministic", replace={"policy_len": 1})
    fast = _fast_rover()

    def mission(view, key, obs, carry, vparams):
        if obs is not None:
            logged(slow_seen, vparams)
        return slow(view, key, obs, carry, vparams)

    def rover(view, key, obs, carry, vparams):
        if obs is not None:
            logged(fast_seen, (obs, vparams))
        return fast(view, key, obs, carry, vparams)

    steps = {"agent.mission": mission, "agent.rover": rover, "world": _gauge_step}
    traj = run(system, params, 7, KEY, steps, clocks={"agent.mission": 3 / ss.SURVEY_TICKS})
    options = [int(o) for o in traj["agent.mission"].emitted["option"]]
    assert np.asarray(traj["agent.mission"].decided).tolist() == [True, False, False] * 2 + [True]
    assert len(slow_seen) == 3
    name = "agent.rover.field"

    def field(p):
        return np.asarray(p.parameters[name].concentration)

    base = field(params)
    for decision, t in enumerate((0, 3, 6)):
        assert np.array_equal(field(slow_seen[decision]), base + t)  # the fast counts so far
    for t, (obs, _) in enumerate(fast_seen):
        assert int(obs["agent.rover.u_opt"]) == options[t]  # the option in force
        assert np.array_equal(field(fast_seen[t][1]), base + t)
    assert options[0] == options[1] == options[2] and options[3] == options[4] == options[5]
    assert traj["agent.rover"].emitted["a_v"].dtype == jnp.float32
    assert np.array_equal(field(traj.params), base + 7)
