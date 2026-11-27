"""``run`` with clocks, and the per-component mirror, against what release 0.19.0 recorded.

``tests/data/clock_parity.npz`` holds the trajectories of the previous drivers on four cases,
replayed here as closed Systems and compared bit for bit:

- ``wired``: two robots of ``compose_minds`` and a corridor process, robot ``a`` held for its
  option (the wired form), with durations ``[4, 2, 4]`` at one round per unit, and ``[2, 1, 2]``
  at two rounds per unit (``clocks={"a": 2.0}`` here). Compared under all
  three orders: both robots' intended and realised moves, the corridor's two states, and the
  number of decisions (the step's calls, ``Trajectory.decided``).
- ``composite``: the same two robots merged, ``a`` held (the composite form), against
  ``with_mirror`` of the merge. Compared under all three orders: both robots' intended and
  realised moves, the positions of the two robots and the number of decisions.
- ``scheduler``: the survey's two-tier scheduler with the tables the recording used, frozen in
  ``tests/survey_frozen.py``, held for three rounds inside the merge run as one mind of the wired
  form, with the rover a fast learner that writes the shared parameter variable back every round.
  Compared under all three orders: the option held each round, the shared field as each decision
  saw it and as the run left it. The recorded side is driven by hand with a fixed gauge
  observation; the closed System has a world leaf with a hand-written step that emits the same.
- ``arena``: two minds whose tokens each read the other's hand through a mindless arena of state
  relays (``tests/test_run_both_ways.py``), a composite run, against
  ``with_mirror(per_component=True)``.
  Under ``"simultaneous"`` and ``"sequential"`` the minds plan (``plan(selection="deterministic")``)
  and both minds' choices and all six world factors are compared. ``run`` derives the sequential
  order (the sources in listing order, each driven world leaf right after its driver), so under
  ``"sequential"`` each mind steps right after the world leaf it drives and the next mind reads the
  world as moved this round, as the composite form's minds did. The arena's tables are exact and
  the selection deterministic, so no key is used and no key patching is needed. Under ``"async"``
  the case is not comparable: the composite form permuted the two minds only and stepped the
  arena's factors at a fixed turn, while ``run``'s async draws a random order of every source (the
  model leaves and the arena leaf) each round and places each world leaf right after its driver;
  the comparison there is of the world given the recorded choices, which the model's steps replay
  (the two hands).

The key layout is the previous drivers' (``_old_layout``): they split a round key in ``2M + 2`` for
``M`` minds, stepped mind ``j`` with key ``j``, realised with key ``M + j`` and stepped the process
with the last key; ``run`` steps every node with ``2N + 1`` keys, so ``_round_keys``,
``_reset_keys`` and the async sweep seed are patched to hand the minds and the world those keys.
"""

import dataclasses
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Categorical, ConditionalCategorical
from hybrid_aif.composition import (
    Identify,
    Link,
    couple,
    merge,
    plan,
    run,
    runner,
    sample,
    with_mirror,
)
from hybrid_aif.distributions import ConditionalGaussian, Gaussian
from hybrid_aif.factors import LinearGaussianTransition
from hybrid_aif.reads import Emission
from hybrid_aif.system import System, SystemParams
from hybrid_aif.system.mirror import mirror
from tests import survey_frozen
from tests.test_run_both_ways import _arena
from tests.test_run_parity import _bump, _run_old_keys, as_recorded, cm, idle, states

FIXTURE = np.load(pathlib.Path(__file__).parent / "data" / "clock_parity.npz")
KEY = jax.random.PRNGKey(1)
ORDERS = ("simultaneous", "sequential", "async")
TICKS = 20


def same(got, name):
    np.testing.assert_array_equal(np.asarray(got), FIXTURE[name], err_msg=name)


def _robots(durations):
    (sa, pa), (sb, pb) = cm.robot(2, 2), cm.robot(2, 2)
    return (sa, dataclasses.replace(pa, durations={"u_move": jnp.asarray(durations)})), (sb, pb)


# ------------------------------------------------------- a held robot, the wired form


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize(
    "tag, durations, unit",
    [("cadence", [4.0, 2.0, 4.0], 1.0), ("unit2", [2.0, 1.0, 2.0], 2.0)],
)
def test_a_held_robot_on_a_mirror_world_is_the_recorded_wired_run_bitwise(
    order, tag, durations, unit
):
    robots = {"a": _robots(durations)[0], "b": cm.robot(2, 2)}
    world = mirror(*cm.world_model(), lower=True)
    system, params = couple(
        {**robots, "world": world},
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
    steps = {
        "a": plan(selection="stochastic"),
        "b": plan(selection="stochastic"),
        "world": sample(),
    }
    traj = _run_old_keys(
        2,
        system,
        params,
        TICKS,
        KEY,
        steps,
        clocks={"a": unit} if unit != 1.0 else None,
        tick_order=order,
        record_carries=True,
    )
    name = f"wired/{order}/{tag}"
    for m in "ab":
        same(traj[m].intended["move"], f"{name}/{m}/intended")
        same(traj[m].emitted["move"], f"{name}/{m}/realised")
    same(states(traj, "world.a"), f"{name}/world/a")
    same(states(traj, "world.b"), f"{name}/world/b")
    assert int(traj["a"].decided.sum()) == int(FIXTURE[f"{name}/calls"])
    assert bool(traj["b"].decided.all())
    assert not bool(traj["a"].decided.all())


# --------------------------------------------------- a held robot, the composite form


@pytest.mark.parametrize("order", ORDERS)
def test_a_held_robot_in_a_merge_against_its_mirror_is_the_recorded_composite_run(order):
    (sa, pa), (sb, pb) = _robots([4.0, 2.0, 4.0])[0], cm.robot(2, 2)
    composite, cp = merge({"a": sa, "b": sb}, {"a": pa, "b": pb}, list(cm.LINKS))
    system, params = with_mirror(composite, cp)
    steps = {
        "model.a": plan(selection="stochastic"),
        "model.b": plan(selection="stochastic"),
        "world": sample(),
    }
    traj = _run_old_keys(
        2, system, params, TICKS, KEY, steps, tick_order=order, record_carries=True
    )
    name = f"composite/{order}"
    for m in "ab":
        same(traj[f"model.{m}"].intended["move"], f"{name}/{m}/intended")
        same(traj[f"model.{m}"].emitted["move"], f"{name}/{m}/realised")
    same(states(traj, "world.a.me"), f"{name}/a.me")
    same(states(traj, "world.b.me"), f"{name}/b.me")
    assert int(traj["model.a"].decided.sum()) == int(FIXTURE[f"{name}/calls"])


# ------------------------------------------------------------------ the held scheduler


def _gauge_world():
    """A world of one categorical factor (the gauge, four levels) and one Gaussian factor (the
    fix, two dimensions), each emitted as it is; a placeholder whose step is hand-written."""
    system = System(
        A_deps=(),
        B_deps=((0,), ()),
        controls=(),
        state_names=("g", "p"),
        modality_names=(),
        emissions=(
            ("gauge", Emission(("g",), "mode", "identity")),
            ("fix", Emission(("p",), "mode", "identity")),
        ),
        mixture_schedule="mode_aligned",
        policy_len=1,
        epistemic=False,
    )
    params = SystemParams(
        transition={
            "g": ConditionalCategorical(log_table=jnp.log(jnp.eye(4) + 1e-9), parent_ndims=1),
            "p": LinearGaussianTransition(
                F=jnp.eye(2),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 2)), chol=0.1 * jnp.eye(2)[None], parent_ndims=1
                ),
            ),
        },
        initial_state_prior={
            "g": Categorical.from_probs(jnp.eye(4)[3]),
            "p": Gaussian(loc=jnp.zeros(2), chol=0.5 * jnp.eye(2)),
        },
        emission={},
    )
    return system, params


def _gauge_step(system, key, obs, carry, params):
    """The world's step: the gauge reads level 3 and the fix the origin, every round."""
    return {"world.gauge": jnp.asarray(3, jnp.int32), "world.fix": jnp.zeros(2)}, carry, params


def scheduler_system():
    """The two-tier survey agent (the scheduler ``mission`` and the rover) coupled to the gauge
    world: ``(system, params, constants)``, ``constants.SURVEY_TICKS`` the option's duration."""
    rs, rp, grid = survey_frozen.rover()
    ms, mp = survey_frozen.scheduler(grid, policy_len=rs.policy_len)
    agent = merge(
        {"mission": ms, "rover": rs},
        {"mission": mp, "rover": rp},
        [
            Link("mission.option", "rover.u_opt"),
            Identify("rover.field", "mission.field"),
            Link("rover.x", "mission.cell_obs", realise="mode", kernel=grid),
        ],
    )
    system, params = couple(
        {"agent": agent, "world": _gauge_world()},
        [
            Identify("world.gauge", "agent.mission.gauge"),
            Identify("world.gauge", "agent.rover.gauge"),
            Identify("world.fix", "agent.rover.fix"),
        ],
    )
    return system, params, survey_frozen.CONSTANTS


def _fast_rover():
    """The rover's step: a zero action, the cell it reports, and the shared field one count
    higher."""

    def step(system, key, obs, carry, params):
        if obs is None:
            return {}, carry, params
        out = {
            "agent.rover.a_v": jnp.zeros(2, jnp.float32),
            "agent.rover.cell_obs": jnp.asarray(0, jnp.int32),
        }
        return out, carry, _bump(system, params)

    return step


@pytest.mark.parametrize("order", ORDERS)
def test_a_held_scheduler_with_a_fast_learner_is_the_recorded_run_bitwise(order):
    system, params, ss = scheduler_system()
    seen = []
    slow = plan(selection="deterministic", replace={"policy_len": 1})

    def mission(view, key, obs, carry, vparams):
        if obs is not None:
            #  the rounds after the first are one compiled scan, so what the step is handed is
            #  logged on the host by a callback (design batched-run.md R8)
            field = vparams.parameters["agent.rover.field"].concentration
            jax.debug.callback(lambda c: seen.append(np.asarray(c)), field, ordered=True)
        return slow(view, key, obs, carry, vparams)

    steps = {"agent.mission": mission, "agent.rover": _fast_rover(), "world": _gauge_step}
    traj = run(
        system,
        as_recorded(params),
        10,
        jax.random.PRNGKey(1),
        steps,
        clocks={"agent.mission": 3 / ss.SURVEY_TICKS},
        tick_order=order,
    )
    same(traj["agent.mission"].emitted["option"], "scheduler/options")
    same(
        np.stack(seen),
        "scheduler/seen_fields",
    )
    same(traj.params.parameters["agent.rover.field"].concentration, "scheduler/final_field")
    decided = np.asarray(traj["agent.mission"].decided)
    assert decided.tolist() == [True, False, False] * 3 + [True]


# ------------------------------------------------------------------ the arena


def _replay(name, levels):
    """The model's step: the recorded action of each round, no inference (the round a traced
    count in the carry, which indexes the recording as an array)."""
    levels = jnp.asarray(levels, jnp.int32)

    def step(system, key, obs, carry, params):
        if obs is None:
            return {}, 0, params
        return {name: levels[carry]}, carry + 1, params

    return step


@pytest.mark.parametrize("order", ["simultaneous", "sequential"])
def test_the_arena_per_component_is_the_composite_run_bitwise(order):
    composite, cp = _arena()
    system, params = with_mirror(composite, cp, per_component=True)
    steps = {
        "model.a": plan(selection="deterministic"),
        "model.b": plan(selection="deterministic"),
        "model.arena": idle,
        "world.a": sample(),
        "world.b": sample(),
        "world.arena": sample(),
    }
    traj = run(system, params, 6, KEY, steps, tick_order=order, record_carries=True)
    for m in "ab":
        same(traj[f"model.{m}"].emitted["a_hand"], f"arena/{order}/{m}.a_hand")
    for f in composite.names.states:
        same(states(traj, f"world.{f}", f"world.{f.split('.')[0]}"), f"arena/{order}/{f}")


def test_the_arena_per_component_under_async_follows_the_recorded_actions():
    order = "async"
    composite, cp = _arena()
    system, params = with_mirror(composite, cp, per_component=True)
    steps = {
        "model.a": _replay("model.a.a_hand", FIXTURE[f"arena/{order}/a.a_hand"]),
        "model.b": _replay("model.b.a_hand", FIXTURE[f"arena/{order}/b.a_hand"]),
        "model.arena": idle,
        "world.a": sample(),
        "world.b": sample(),
        "world.arena": sample(),
    }
    traj = run(system, params, 6, KEY, steps, tick_order=order, record_carries=True)
    for f in ("a.hand", "b.hand"):
        leaf = f.split(".")[0]
        same(states(traj, f"world.{f}", f"world.{leaf}"), f"arena/{order}/{f}")


# ------------------------- the derived sequential order, in the cases of test_run_parity


def _orders(system, params):
    """The listing of the stepped leaves and the order ``"sequential"`` derives."""
    names = [n for n, _ in system.subsystems]
    leaves = [n for n in names if not any(o.startswith(f"{n}.") for o in names)]
    nodes = runner._structure(system, params, dict.fromkeys(leaves, idle))
    second, derive = runner._check_timing(nodes)
    sources = [n for n in nodes if not n.phase_two]
    return [n.name for n in nodes], [n.name for n in derive(sources)]


def test_the_derived_sequential_order_is_the_listing_in_the_cases_of_the_run_parity_tests():
    """In every case of ``test_run_parity`` but two the driven world leaf is last in the listing,
    so the derived order is the listing. The exceptions: the same-step zoo, whose world is driven
    by ``model.agent`` alone, steps right after it, before the idle ``model.world`` (a step that
    reads and chooses nothing); and the two pushers on a direct wire, whose process is driven by
    ``A`` alone, steps right after ``A``, before ``B``, whose own observation is of ``A``'s action
    and not of the process. Neither changes what a node reads, which ``test_run_parity`` checks
    against the recorded runs."""
    from tests.test_action_nodes import pusher
    from tests.test_merge import OBSERVATION_BINDINGS, agent_and_world
    from tests.test_run_nested import _shared
    from tests.test_run_nested import _zoo as same_step_zoo
    from tests.test_run_parity import _coupled_robots

    (agent, ap), (world, wp) = agent_and_world(policy_len=1)
    zoo = merge({"agent": agent, "world": world}, {"agent": ap, "world": wp}, OBSERVATION_BINDINGS)
    pushers = merge({n: pusher()[0] for n in "abc"}, {n: pusher()[1] for n in "abc"}, [])
    direct = couple(
        {"A": pusher(), "B": pusher(), "P1": mirror(*pusher(noise=0.0), lower=True)},
        [Identify("A.push", "P1.u_push"), Identify("A.push", "B.see"), Identify("B.push", "A.see")],
    )
    same_listing = [
        with_mirror(*zoo),
        with_mirror(*pushers),
        _coupled_robots(cm.SYMMETRIC, "deterministic")[:2],
        with_mirror(*_shared()[:2]),
    ]
    for system, params in same_listing:
        listing, derived = _orders(system, params)
        assert listing == derived, (listing, derived)
    listing, derived = _orders(*with_mirror(*same_step_zoo()[:2]))
    assert listing == ["model.agent", "model.world", "world"]
    assert derived == ["model.agent", "world", "model.world"]
    listing, derived = _orders(*direct)
    assert listing == ["A", "B", "P1"] and derived == ["A", "P1", "B"]


def test_the_derived_async_order_is_the_phase_structure_in_the_cases_of_the_run_parity_tests():
    """Async permutes the sources and places each driven node after its last driver. In the cases
    of ``test_run_parity`` that keep their listing under ``"sequential"`` every driven world leaf
    is driven by all the minds, so for every permutation of the sources the derived order is the
    permuted sources and then the driven leaves, the phase structure of the recorded runs."""
    import itertools

    from tests.test_action_nodes import pusher
    from tests.test_merge import OBSERVATION_BINDINGS, agent_and_world
    from tests.test_run_nested import _shared
    from tests.test_run_parity import _coupled_robots

    (agent, ap), (world, wp) = agent_and_world(policy_len=1)
    zoo = merge({"agent": agent, "world": world}, {"agent": ap, "world": wp}, OBSERVATION_BINDINGS)
    pushers = merge({n: pusher()[0] for n in "abc"}, {n: pusher()[1] for n in "abc"}, [])
    systems = [
        with_mirror(*zoo),
        with_mirror(*pushers),
        _coupled_robots(cm.SYMMETRIC, "deterministic")[:2],
        with_mirror(*_shared()[:2]),
    ]
    for system, params in systems:
        names = [n for n, _ in system.subsystems]
        leaves = [n for n in names if not any(o.startswith(f"{n}.") for o in names)]
        nodes = runner._structure(system, params, dict.fromkeys(leaves, idle))
        second, derive = runner._check_timing(nodes)
        sources = [n for n in nodes if not n.phase_two]
        for perm in itertools.permutations(sources):
            assert [n.name for n in derive(perm)] == [n.name for n in (*perm, *second)]
