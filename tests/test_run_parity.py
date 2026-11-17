"""``run`` over a closed System against what the wired and composite forms recorded.

``tests/data/run_parity.npz`` holds the trajectories of the previous drivers on six cases: the zoo
pair and three merged pushers as a composite run, two pushers on a direct wire through a process,
the two robots of ``compose_minds`` and the minds of ``hide_and_seek`` on a world process, and a
learner on a shared parameter variable. Each case is replayed here as a closed System (the
model coupled to the mirror of its own model, or the minds coupled to a mirror world) under the
key layout of the driver that recorded it, and compared bit for bit.

**Key patching.** The driver that recorded a case split each round key into ``2M + 2`` keys for
its ``M`` minds, stepped mind ``j`` with key ``j``, realised sequentially with key ``M + j`` and
simultaneously, and stepped its one process, with the last key, and reset the process with the
reset key. ``run`` steps every node, the process included, with ``2N + 1`` keys.
``_old_layout(M)`` patches ``_round_keys`` and ``_reset_keys`` to hand the minds and the process
those keys, and ``_sweep_key`` to seed the async order with ``M``; nothing else differs, so
the comparison is of the rounds and not of a key schedule.

Every case is compared under all three orders: every node's intended and realised actions, the
recorded inputs, and the world's states, with two exceptions that compare world states under
``"simultaneous"`` only, the three pushers and the learner's rover. The composite run steps each
component's factors at that component's turn with the component's own keys, and the closed System
has one world that steps once at its turn with its own key, so the pushers' independent worlds
draw differently from the third tick of a sequential or async run. The zoo's world states match
under every order (its two components share latents, and the draws land the same); the
same-step zoo is compared under simultaneous and sequential only.

**The async re-pin.** ``run``'s async order is drawn by ``jax.random.permutation`` inside the
scanned round (design batched-run.md R9), where the previous drivers drew it from a host
generator; under the same seed the two direct-wire cases (four of their fourteen arrays) draw
another order, and are compared against ``tests/data/run_parity_async.npz``, recorded from
``run`` on 2026-11-17. Every other async array is still the previous drivers' own.
"""

import dataclasses
import importlib.util
import pathlib
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.composition import (
    Identify,
    couple,
    merge,
    plan,
    run,
    runner,
    sample,
    with_mirror,
)
from hybrid_aif.distributions import Categorical, ConditionalDirichlet
from hybrid_aif.system import keyed
from hybrid_aif.system.mirror import mirror
from tests.test_action_nodes import KERNEL, pusher
from tests.test_couple import KERNEL5
from tests.test_merge import OBSERVATION_BINDINGS, agent_and_world
from tests.test_run_nested import _shared
from tests.test_run_nested import _zoo as _same_step_zoo

FIXTURE = np.load(pathlib.Path(__file__).parent / "data" / "run_parity.npz")
#: The async direct-wire case re-pinned when ``run``'s async order moved from a host generator to
#: ``jax.random.permutation`` (design batched-run.md R9, R13 pin 1): recorded from ``run`` itself,
#: so it pins that run's own draws, not a previous driver's. Every other async array still matches
#: ``run_parity.npz``.
REPINNED = np.load(pathlib.Path(__file__).parent / "data" / "run_parity_async.npz")
KEY = jax.random.PRNGKey(0)
ORDERS = ("simultaneous", "sequential", "async")


def _example(name):
    path = pathlib.Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cm, hs = _example("compose_minds"), _example("hide_and_seek")


def _old_layout(minds: int):
    """The keys of the previous driver for ``minds`` minds and the processes after them."""

    def round_keys(key, count):
        old = list(jax.random.split(key, 2 * minds + 2))
        keys = [None] * (2 * count + 1)
        for j in range(minds):
            keys[j], keys[count + j] = old[j], old[minds + j]
        for p in range(count - minds):
            keys[minds + p] = old[-1] if p == 0 else jax.random.fold_in(old[-1], p)
            keys[count + minds + p] = old[-1]
        keys[2 * count] = old[-1]
        return keys

    def reset_keys(key, count):
        return [jax.random.fold_in(key, j) for j in range(minds)] + [
            key if p == 0 else jax.random.fold_in(key, p) for p in range(count - minds)
        ]

    original = runner._sweep_key
    return (
        mock.patch.object(runner, "_round_keys", round_keys),
        mock.patch.object(runner, "_reset_keys", reset_keys),
        mock.patch.object(runner, "_sweep_key", lambda key, count: original(key, minds)),
    )


def _run_old_keys(minds: int, *args, **kwargs):
    a, b, c = _old_layout(minds)
    with a, b, c:
        return run(*args, **kwargs)


def habit_draw(system, key, params):
    """The action nodes of ``system``'s control nodes drawn from the habit, as the habit step of
    the composite and wired forms of release 0.19.0 drew them with ``key``: the first step of a
    policy drawn from
    ``params.policy_prior`` where one is declared, else one uniform level per node, each from its
    own key of ``split(key, nodes)``."""
    names = system._names()
    if params.policy_prior is not None:
        prior = params.policy_prior
        logits = prior.log_probs if isinstance(prior, Categorical) else jnp.asarray(prior)
        chosen = system.policies(params)[Categorical(logits=logits).sample(key), 0]
        return keyed.actions_out(names, chosen.astype(jnp.int32))
    if system._table.sugar is not None:  # one draw per state factor, as the sugar lays them out
        levels = tuple(
            n if c else 1
            for n, c in zip(
                system._num_controls(params, "habit"), system._ctl("habit"), strict=True
            )
        )
    else:
        levels = system._control_levels(system._params_in(params, "habit"))
    keys = jax.random.split(key, len(levels))
    draws = [jax.random.randint(k, (), 0, n) for k, n in zip(list(keys), levels, strict=True)]
    drawn = jnp.stack(draws).astype(jnp.int32) if draws else jnp.zeros((0,), jnp.int32)
    if system._table.sugar is None:
        return keyed.actions_out(names, drawn)
    return keyed.factor_actions_out(names, drawn)


def ancestral():
    """A step of ``run`` that draws its node's action nodes from the habit with the key use of
    the habit step of release 0.19.0 (:func:`habit_draw`), so a fixture recorded with it replays
    bit for bit;
    ``sample()`` draws a free node with another key use."""

    def step(system, key, obs, carry, params):
        if obs is None:
            return {}, carry, params
        return habit_draw(system, key, params), carry, params

    return step


def idle(system, key, obs, carry, params):
    return {}, carry, params


def same(got, name):
    want = REPINNED[name] if name in REPINNED.files else FIXTURE[name]
    np.testing.assert_array_equal(np.asarray(got), want, err_msg=name)


def states(trajectory, factor, node="world"):
    return np.asarray([c[factor] for c in trajectory[node].carries])


# ------------------------------------------------------------------ the zoo pair, a composite


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("kernel", ["delta", "kernel"])
def test_the_zoo_pair_against_its_own_mirror_is_the_composite_run_bitwise(order, kernel):
    (agent, ap), (world, wp) = agent_and_world(policy_len=1)
    composite, cp = merge(
        {"agent": agent, "world": world}, {"agent": ap, "world": wp}, OBSERVATION_BINDINGS
    )
    if kernel == "kernel":
        cp = dataclasses.replace(cp, actuator={"agent.a_location": KERNEL5})
    system, params = with_mirror(composite, cp)
    steps = {"model.agent": plan(), "model.world": ancestral(), "world": sample()}
    traj = _run_old_keys(2, system, params, 12, KEY, steps, tick_order=order, record_carries=True)
    tag = f"zoo_{order}_{kernel}"
    for leaf, a in (("model.agent", "agent.a_location"), ("model.world", "world.a_location")):
        node = traj[leaf]
        same(node.intended[a.split(".", 1)[1]], f"{tag}/{a}/intended")
        same(node.emitted[a.split(".", 1)[1]], f"{tag}/{a}/realised")
    same(states(traj, "world.world.location"), f"{tag}/world.location")


@pytest.mark.parametrize("order", ["simultaneous", "sequential"])
def test_the_same_step_zoo_against_its_own_mirror_is_the_composite_run_bitwise(order):
    """The zoo whose world node is driven by the agent's action (the collapsed same-step
    composite), run as a closed System against its mirror. Under ``"async"`` the composite stepped
    the world component's factors at its own random turn, before or after the agent's, which one
    world step cannot reproduce, so that order is not compared."""
    composite, cp, _ = _same_step_zoo()
    system, params = with_mirror(composite, cp)
    steps = {"model.agent": plan(selection="deterministic"), "model.world": idle, "world": sample()}
    traj = _run_old_keys(2, system, params, 6, KEY, steps, tick_order=order, record_carries=True)
    tag = f"same_step_zoo/{order}"
    same(traj["model.agent"].emitted["a_location"], f"{tag}/agent.a_location")
    same(states(traj, "world.world.location"), f"{tag}/world.location")
    same(states(traj, "world.agent.location"), f"{tag}/agent.location")


@pytest.mark.parametrize("order", ORDERS)
def test_three_merged_pushers_with_a_kernel_each_against_their_mirror_are_the_composite_run(order):
    parts = {n: pusher() for n in "abc"}
    composite, cp = merge(
        {n: s for n, (s, _) in parts.items()}, {n: p for n, (_, p) in parts.items()}, []
    )
    cp = dataclasses.replace(cp, actuator={f"{n}.push": KERNEL for n in "abc"})
    system, params = with_mirror(composite, cp)
    steps = {"model.a": plan(), "model.b": ancestral(), "model.c": ancestral(), "world": sample()}
    traj = _run_old_keys(
        3, system, params, 8, jax.random.PRNGKey(3), steps, tick_order=order, record_carries=True
    )
    for n in "abc":
        same(traj[f"model.{n}"].intended["push"], f"pushers/{order}/{n}/intended")
        same(traj[f"model.{n}"].emitted["push"], f"pushers/{order}/{n}/realised")
        if order == "simultaneous":
            same(states(traj, f"world.{n}.x"), f"pushers/{order}/{n}/x")
        assert not np.array_equal(
            traj[f"model.{n}"].intended["push"], traj[f"model.{n}"].emitted["push"]
        )


# ------------------------------------------------------------------- pushers on a direct wire


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("kernel", ["delta", "kernel"])
def test_two_pushers_on_a_direct_wire_through_a_process_under_each_order(order, kernel):
    a = pusher(kernel=KERNEL if kernel == "kernel" else None)
    process = mirror(*pusher(noise=0.0), lower=True)
    system, params = couple(
        {"A": a, "B": pusher(), "P1": process},
        [Identify("A.push", "P1.u_push"), Identify("A.push", "B.see"), Identify("B.push", "A.see")],
    )
    steps = {"A": ancestral(), "B": ancestral(), "P1": sample()}
    traj = _run_old_keys(2, system, params, 12, KEY, steps, tick_order=order, record_carries=True)
    tag = f"direct/{order}/{kernel}"
    for n in "AB":
        same(traj[n].intended["push"], f"{tag}/{n}/intended")
        same(traj[n].emitted["push"], f"{tag}/{n}/realised")
        same(traj[n].received["see"], f"{tag}/{n}/received")
    same(states(traj, "P1.x", "P1"), f"{tag}/P1/x")
    if kernel == "kernel":
        assert not np.array_equal(traj["A"].intended["push"], traj["A"].emitted["push"])


# ----------------------------------------------------- compose_minds and hide_and_seek: worlds


def _coupled_robots(start, selection):
    a, b = start
    robots = {"a": cm.robot(a, b, 2), "b": cm.robot(b, a, 2)}
    world = mirror(*cm.world_model(start), lower=True)
    system, params = couple(
        {**robots, "world": world},
        [
            Identify("a.move", "world.u_a"),
            Identify("b.move", "world.u_b"),
            *(Identify(f"world.{p}", f"a.{m}") for p, m in cm_wires("a")),
            *(Identify(f"world.{p}", f"b.{m}") for p, m in cm_wires("b")),
        ],
    )
    steps = {"a": plan(selection=selection), "b": plan(selection=selection), "world": sample()}
    return system, params, steps


def cm_wires(who):
    me, other = ("a_pos", "b_pos") if who == "a" else ("b_pos", "a_pos")
    return ((me, "me_obs"), (other, "other_obs"), ("status", "status"))


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("selection", ["deterministic", "stochastic"])
@pytest.mark.parametrize("start", ["symmetric", "asymmetric"])
def test_compose_minds_two_robots_on_a_mirror_world_under_each_order(order, selection, start):
    system, params, steps = _coupled_robots(
        cm.SYMMETRIC if start == "symmetric" else cm.ASYMMETRIC, selection
    )
    key = jax.random.PRNGKey(3)
    traj = _run_old_keys(2, system, params, 8, key, steps, tick_order=order, record_carries=True)
    tag = f"compose_minds/{order}/{selection}/{start}"
    same(states(traj, "world.a"), f"{tag}/a")
    same(states(traj, "world.b"), f"{tag}/b")
    for m in "ab":
        same(traj[m].emitted["move"], f"{tag}/{m}/move")


@pytest.mark.parametrize("order", ORDERS)
def test_hide_and_seek_prey_and_predator_on_a_mirror_world_under_each_order(order):
    prey, predator = hs.mind("prey", n=3), hs.mind("predator", n=3)
    world = mirror(*hs.world_model(3), lower=True)
    system, params = couple(
        {"prey": prey, "predator": predator, "world": world},
        [
            Identify("prey.move", "world.u_prey"),
            Identify("predator.move", "world.u_pred"),
            Identify("world.prey_pos", "prey.me_obs"),
            Identify("world.pred_pos", "prey.other_obs"),
            Identify("world.distance", "prey.distance"),
            Identify("world.pred_pos", "predator.me_obs"),
            Identify("world.prey_pos", "predator.other_obs"),
            Identify("world.distance", "predator.distance"),
        ],
    )
    steps = {"prey": plan(), "predator": plan(), "world": sample()}
    traj = _run_old_keys(
        2, system, params, 6, jax.random.PRNGKey(0), steps, tick_order=order, record_carries=True
    )
    tag = f"hide_and_seek/{order}"
    same(states(traj, "world.prey"), f"{tag}/prey")
    same(states(traj, "world.predator"), f"{tag}/predator")
    entries = [traj["world"].start["distance"], *traj["world"].emitted["distance"]]
    same(jnp.stack(entries), f"{tag}/distance")


# --------------------------------------------------------------- a learner's write-backs


def _bump(view, params):
    """``params`` with the shared field's concentrations one higher: what a learner returns."""
    held = view._params_in(params, "test")
    name = next(k for k in held.parameters if k.endswith("rover.field"))
    field = held.parameters[name]
    new = ConditionalDirichlet(concentration=field.concentration + 1.0, parent_ndims=1)
    out = dataclasses.replace(held, parameters={**held.parameters, name: new})
    for slot, i, home in held.references:
        container = list(getattr(out, slot))
        container[i] = out.parameters[home]
        out = dataclasses.replace(out, **{slot: tuple(container)})
    return view._params_out(out)


def learner():
    inner = plan()

    def step(system, key, obs, carry, params):
        action, carry, _ = inner(system, key, obs, carry, params)
        return action, carry, params if obs is None else _bump(system, params)

    return step


@pytest.mark.parametrize("order", ORDERS)
def test_a_learning_view_is_written_back_to_the_home_as_the_composite_run_did(order):
    """The rover learns the shared field, one count a round, and the home after the run is the
    composite form's under every order. The composite form applied a write-back live, so the
    scheduler (stepping after the rover) saw this round's count; under ``"simultaneous"`` ``run``
    applies every write-back at the round's boundary, so the scheduler sees the round before's,
    and the live reading is reproduced under ``"sequential"``. What the scheduler saw is not in
    the fixture; the home and the rover's world are, and they are compared."""
    composite, cp, _, _ = _shared()
    system, params = with_mirror(composite, cp)
    steps = {"model.rover": learner(), "model.mission": plan(), "world": sample()}
    traj = _run_old_keys(2, system, params, 4, KEY, steps, tick_order=order, record_carries=True)
    field = traj.params.parameters["model.rover.field"].concentration
    same(field, f"learning/{order}/field")
    if order == "simultaneous":
        same(states(traj, "world.rover.x"), f"learning/{order}/rover.x")
    assert not np.array_equal(field, params.parameters["model.rover.field"].concentration)
    assert traj.params is not params
    assert jnp.allclose(field, params.parameters["model.rover.field"].concentration + 4)
