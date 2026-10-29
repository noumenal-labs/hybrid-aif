"""Coupling through blankets: minds and processes joined by typed wires, run by the driver.

A run's participants are minds (a System with a respond function) and processes (a keyed env);
a wire is an identity from an action node to a process input or a mind's modality, or from a
process observation to a mind's modality. An action node is realised once per step from its
mind's actuator and every wire from it carries that value. A merged composite run by name is the
one-process case, bitwise.

Design: docs/design/control-states.md C12, C13; docs/design/nesting-systems.md principle 6.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from hybrid_aif import Categorical, ConditionalCategorical
from hybrid_aif.composition import (
    MindCarry,
    MindRecord,
    ProcessRecord,
    Wire,
    Wiring,
    agent_respond,
    ancestral_respond,
    couple,
    merge,
    run,
    subsystem,
)
from hybrid_aif.controls import Box
from hybrid_aif.distributions import ConditionalGaussian, Gaussian
from hybrid_aif.envs import GenerativeEnv
from hybrid_aif.envs._process import DiscreteProcess
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianTransition
from hybrid_aif.system import System, SystemParams
from tests.test_action_nodes import KERNEL, pusher
from tests.test_merge import OBSERVATION_BINDINGS, agent_and_world

KEY = jax.random.PRNGKey(0)

#: P(a | u) over the T-maze's five locations: the intended one most often.
KERNEL5 = 0.4 * jnp.eye(5) + 0.12


def watcher():
    """A native System with one state ``x``, one control node (action ``push``), and two
    three-level modalities on ``x``, ``see`` and ``see2``."""
    system = System(
        A_deps=((0,), (0,)),
        B_deps=((0, 1), ()),
        controls=(1,),
        action_names=("push",),
        factor_names=("x", "u_push"),
        modality_names=("see", "see2"),
        policy_len=1,
        epistemic=False,
    )
    table = ConditionalCategorical(log_table=jnp.log(0.8 * jnp.eye(3) + 0.2 / 3), parent_ndims=1)
    _, pp = pusher()
    params = SystemParams(
        likelihood={"see": table, "see2": table},
        transition=pp.transition,
        initial_state_prior={"x": Categorical(logits=jnp.zeros(3))},
    )
    return system, params


def process(system, params):
    """``system`` run forward as a process, its actuator left to the mind."""
    env, ep = system.as_env(params)
    return env, dataclasses.replace(ep, actuator={})


def constant(level):
    """A RespondFn emitting ``level`` on the action node ``push`` every tick."""

    def respond(key, obs, carry):
        return {"push": jnp.asarray(level, jnp.int32)}, carry

    return respond


def equal(a, b):
    return bool(jnp.array_equal(a, b))


# --------------------------------------------------------------------- 1. the composite as sugar


def _two_minds(kernel):
    (agent, ap), (world, wp) = agent_and_world(policy_len=1)
    composite, cp = merge(
        {"agent": agent, "world": world}, {"agent": ap, "world": wp}, OBSERVATION_BINDINGS
    )
    if kernel:
        cp = dataclasses.replace(
            cp, actuator={"agent.a_location": KERNEL5, "world.a_location": KERNEL5}
        )
    views = {name: subsystem(composite, cp, name) for name in ("agent", "world")}
    responds = {
        "agent": agent_respond(*views["agent"]),
        "world": ancestral_respond(*views["world"]),
    }
    return composite, cp, views, responds


def _sugar(composite, cp, views, responds):
    """The composite as one process, the minds on their views, the wires by name."""
    wires = []
    for name, (view, _) in views.items():
        wires += [Wire(f"{name}.{a}", f"env.{a}") for a in view.names.actions]
        wires += [Wire(f"env.{m}", f"{name}.{m}") for m in view.names.modalities]
    minds = {name: (view, vp, responds[name]) for name, (view, vp) in views.items()}
    return couple(minds=minds, processes={"env": process(composite, cp)}, wires=wires)


@pytest.mark.parametrize("kernel", [False, True], ids=["delta", "kernel"])
def test_the_composite_run_is_the_one_process_wiring_bitwise(kernel):
    """Under ``"simultaneous"``; under the other orders the composite steps each component's
    factors at its turn, which a wiring of one process cannot (the driver docstring)."""
    tick_order = "simultaneous"
    composite, cp, views, responds = _two_minds(kernel)
    traj = run(composite, cp, responds, ticks=12, key=KEY, tick_order=tick_order)
    record = run(_sugar(composite, cp, views, responds), ticks=12, key=KEY, tick_order=tick_order)
    env = record["env"]
    assert isinstance(env, ProcessRecord)
    for f, rows in traj.states.items():
        assert equal(env.states[f], rows), f
    for m, rows in traj.observations.items():
        assert equal(env.observations[m], rows), m
    for name in ("agent", "world"):
        mind = record[name]
        assert isinstance(mind, MindRecord)
        for a in views[name][0].names.actions:
            assert equal(mind.actions[a], traj.actions[a]), a
            assert equal(mind.realised[a], traj.realised[a]), a
    if kernel:  # each kernel moved some draw off the intended level
        for a in ("agent.a_location", "world.a_location"):
            assert not equal(traj.realised[a], traj.actions[a]), a
    if tick_order == "simultaneous":  # the agent received the process's rows
        for m in views["agent"][0].names.modalities:
            assert equal(record["agent"].observations[m], traj.observations[m])


def _three_pushers(order):
    """Three merged pushers, a kernel on every action node, minds and wires in ``order``."""
    comps = ("a", "b", "c")
    parts = {n: pusher() for n in comps}
    composite, cp = merge(
        {n: s for n, (s, _) in parts.items()}, {n: p for n, (_, p) in parts.items()}, []
    )
    cp = dataclasses.replace(cp, actuator={f"{n}.push": KERNEL for n in comps})
    views = {n: subsystem(composite, cp, n) for n in order}
    responds = {n: (agent_respond if n == "a" else ancestral_respond)(*views[n]) for n in order}
    return composite, cp, views, responds


def test_three_minds_in_component_order_match_the_composite_bitwise():
    composite, cp, views, responds = _three_pushers(("a", "b", "c"))
    key = jax.random.PRNGKey(3)
    traj = run(composite, cp, responds, ticks=8, key=key)
    record = run(_sugar(composite, cp, views, responds), ticks=8, key=key)
    for f, rows in traj.states.items():
        assert equal(record["env"].states[f], rows), f
    for m, rows in traj.observations.items():
        assert equal(record["env"].observations[m], rows), m
    for name in ("a", "b", "c"):
        a = f"{name}.push"
        assert equal(record[name].actions[a], traj.actions[a]), a
        assert equal(record[name].realised[a], traj.realised[a]), a
        assert not equal(traj.realised[a], traj.actions[a]), a


def test_minds_out_of_component_order_run_and_record_by_participant():
    composite, cp, views, responds = _three_pushers(("c", "a", "b"))
    record = run(_sugar(composite, cp, views, responds), ticks=4, key=KEY, tick_order="async")
    assert list(record) == ["c", "a", "b", "env"]
    for name in ("c", "a", "b"):
        assert list(record[name].actions) == list(record[name].realised) == [f"{name}.push"]
        assert record[name].realised[f"{name}.push"].shape == (4,)
        assert list(record[name].observations) == [f"{name}.see"]
    assert set(record["env"].states) == {"a.x", "b.x", "c.x"}


# ------------------------------------------------------------ 2. two minds, two processes


def test_two_minds_on_two_processes_record_by_participant_and_match_a_hand_loop():
    a_sys, a_par = pusher()
    b_sys, b_par = watcher()
    wiring = couple(
        minds={
            "A": (a_sys, a_par, agent_respond(a_sys, a_par)),
            "B": (b_sys, b_par, ancestral_respond(b_sys, b_par)),
        },
        processes={"P1": process(*pusher()), "P2": process(*pusher())},
        wires=[
            Wire("A.push", "P1.push"),
            Wire("P1.see", "A.see"),
            Wire("P1.see", "B.see"),
            Wire("B.push", "P2.push"),
            Wire("P2.see", "B.see2"),
        ],
    )
    ticks = 5
    record = run(wiring, ticks=ticks, key=KEY)
    assert list(record) == ["A", "B", "P1", "P2"]
    assert set(record["A"].observations) == {"see"}
    assert set(record["B"].observations) == {"see", "see2"}
    for name in ("A", "B"):
        mind = record[name]
        assert all(v.shape == (ticks + 1,) for v in mind.observations.values())
        assert list(mind.actions) == list(mind.realised) == ["push"]
        assert mind.actions["push"].shape == mind.realised["push"].shape == (ticks,)
        assert isinstance(mind.carry, MindCarry)
    for name in ("P1", "P2"):
        proc = record[name]
        assert list(proc.states) == ["x"] and list(proc.observations) == ["see"]
        assert proc.states["x"].shape == proc.observations["see"].shape == (ticks + 1,)

    #  the hand loop: the documented key schedule, the processes stepped on the recorded acts
    (e1, p1), (e2, p2) = wiring.processes["P1"], wiring.processes["P2"]
    key, k_reset = jax.random.split(KEY)
    _, s1 = e1.reset(k_reset, p1)
    _, s2 = e2.reset(jax.random.fold_in(k_reset, 1), p2)
    want1, want2 = [s1["x"]], [s2["x"]]
    for t in range(ticks):
        key, k_tick = jax.random.split(key)
        k = jax.random.split(k_tick, 2 * 2 + 2)[-1]
        _, s1 = e1.step(k, s1, {"push": record["A"].realised["push"][t]}, p1)
        _, s2 = e2.step(jax.random.fold_in(k, 1), s2, {"push": record["B"].realised["push"][t]}, p2)
        want1.append(s1["x"])
        want2.append(s2["x"])
    assert equal(record["P1"].states["x"], jnp.stack(want1))
    assert equal(record["P2"].states["x"], jnp.stack(want2))
    #  what A and B received of P1 is P1's emission, row for row
    assert equal(record["A"].observations["see"], record["P1"].observations["see"])
    assert equal(record["B"].observations["see"], record["P1"].observations["see"])


# ------------------------------------------------------------------------- 3. a direct wire


def _direct(tick_order, kernel=None):
    a_sys, a_par = pusher(kernel=kernel)
    b_sys, b_par = pusher()
    wiring = couple(
        minds={
            "A": (a_sys, a_par, ancestral_respond(a_sys, a_par)),
            "B": (b_sys, b_par, ancestral_respond(b_sys, b_par)),
        },
        processes={"P1": process(*pusher(noise=0.0))},
        wires=[
            Wire("A.push", "P1.push"),
            Wire("A.push", "B.see"),
            Wire("B.push", "A.see"),
        ],
    )
    return run(wiring, ticks=12, key=KEY, tick_order=tick_order)


def test_a_direct_wire_lags_a_tick_when_simultaneous():
    record = _direct("simultaneous")
    a, b = record["A"], record["B"]
    assert int(b.observations["see"][0]) == 0 and int(a.observations["see"][0]) == 0  # null
    assert equal(b.observations["see"][1:], a.realised["push"])
    assert equal(a.observations["see"][1:], b.realised["push"])


def test_a_later_mind_sees_an_earlier_minds_act_this_tick_when_sequential():
    """A mind's row t is what it responded to at tick t: B's row t holds A's act of tick t, and
    A's row t + 1 holds B's act of tick t."""
    record = _direct("sequential")
    a, b = record["A"], record["B"]
    #  B responds after A: what B received at tick t is A's act of tick t
    assert equal(b.observations["see"][:-1], a.actions["push"])
    #  A responds first: what A received at tick t is B's act of tick t - 1, null at the reset
    assert int(a.observations["see"][0]) == 0
    assert equal(a.observations["see"][1:], b.actions["push"])
    assert not equal(a.observations["see"][:-1], b.actions["push"])


def test_an_action_node_is_drawn_once_and_every_wire_carries_the_draw():
    record = _direct("simultaneous", kernel=KERNEL)
    a, b, p1 = record["A"], record["B"], record["P1"]
    #  P1 is x' = x + a (mod 3), noiseless, so its states give the action it stepped on
    x = p1.states["x"]
    stepped = (x[1:] - x[:-1]) % 3
    assert equal(stepped, a.realised["push"])
    assert equal(b.observations["see"][1:], a.realised["push"])
    assert not equal(a.realised["push"], a.actions["push"])  # the kernel moved some draw


def test_under_sequential_a_draw_is_consumed_once_and_carried_to_the_later_mind():
    """A's node, wired to P1 and to B, is drawn once when A chooses: P1's one step in the
    tick moves on that draw, B (after A) receives it, and A.realised records it."""
    record = _direct("sequential", kernel=KERNEL)
    a, b, p1 = record["A"], record["B"], record["P1"]
    x = p1.states["x"]
    #  P1 is x' = x + a (mod 3), stepped once at the end of the tick: one move of one value
    assert equal((x[1:] - x[:-1]) % 3, a.realised["push"])
    assert equal(b.observations["see"][:-1], a.realised["push"])
    assert not equal(a.realised["push"], a.actions["push"])


def test_the_composite_moves_each_component_once_on_its_own_draw_under_sequential():
    """Two merged noiseless pushers, a kernel on ``a.push`` only: each tick a's state steps
    at a's turn on a's one draw, and b's at b's turn on b's choice; neither moves twice."""
    parts = {n: pusher(noise=0.0) for n in ("a", "b")}
    composite, cp = merge(
        {n: s for n, (s, _) in parts.items()}, {n: p for n, (_, p) in parts.items()}, []
    )
    cp = dataclasses.replace(cp, actuator={"a.push": KERNEL})
    minds = {n: ancestral_respond(*subsystem(composite, cp, n)) for n in ("a", "b")}
    traj = run(composite, cp, minds, ticks=12, key=KEY, tick_order="sequential")
    xa, xb = traj.states["a.x"], traj.states["b.x"]
    assert equal((xa[1:] - xa[:-1]) % 3, traj.realised["a.push"])
    assert equal((xb[1:] - xb[:-1]) % 3, traj.realised["b.push"])
    assert not equal(traj.realised["a.push"], traj.actions["a.push"])


def test_a_simultaneous_kernel_run_is_pinned():
    """Under simultaneous every node is drawn at the tick's one step with the env actuator's
    keys, ``split(fold_in(k, ACTUATOR_TAG), n)[i]`` in composite action order; pinned."""
    composite, cp, _, responds = _two_minds(True)
    traj = run(composite, cp, responds, ticks=12, key=KEY)
    assert traj.realised["agent.a_location"].tolist() == [4, 4, 3, 4, 1, 1, 2, 3, 4, 0, 1, 2]
    assert traj.realised["world.a_location"].tolist() == [1, 2, 2, 0, 0, 4, 2, 2, 1, 4, 0, 2]
    assert traj.states["world.location"].tolist() == [0, 0, 0, 0, 0, 0, 4, 2, 2, 2, 4, 0, 0]


# ---------------------------------------------------------------------------------- 4. fixed=


def test_a_fixed_input_advances_the_process_as_an_emitted_one_would():
    ticks = 6
    fixed = run(
        couple(processes={"P": process(*pusher(noise=0.0))}, fixed={"P.push": 1}),
        ticks=ticks,
        key=KEY,
    )
    assert list(fixed) == ["P"]
    #  the hand loop with no minds: two keys per tick, the step on the last
    env, ep = process(*pusher(noise=0.0))
    key, k_reset = jax.random.split(KEY)
    obs, state = env.reset(k_reset, ep)
    states, seen = [state["x"]], [obs["see"]]
    for _ in range(ticks):
        key, k_tick = jax.random.split(key)
        obs, state = env.step(jax.random.split(k_tick, 2)[-1], state, {"push": 1}, ep)
        states.append(state["x"])
        seen.append(obs["see"])
    assert equal(fixed["P"].states["x"], jnp.stack(states))
    assert equal(fixed["P"].observations["see"], jnp.stack(seen))
    assert equal(fixed["P"].states["x"], (fixed["P"].states["x"][0] + jnp.arange(ticks + 1)) % 3)
    #  a mind emitting the same constant moves the noiseless process identically
    m_sys, m_par = pusher()
    emitted = run(
        couple(
            minds={"M": (m_sys, m_par, constant(1))},
            processes={"P": process(*pusher(noise=0.0))},
            wires=[Wire("M.push", "P.push"), Wire("P.see", "M.see")],
        ),
        ticks=ticks,
        key=KEY,
    )
    assert equal(emitted["P"].states["x"], fixed["P"].states["x"])


# ---------------------------------------------------------------------------- 5. continuous


def _continuous(width):
    """A Gaussian state ``x`` (2-d) moved by a continuous node of ``width`` through ``G u``,
    seen through a 2-d linear-Gaussian modality ``pos``."""
    system = System(
        A_deps=((0,),),
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("u",),
        factor_names=("x", "u_u"),
        modality_names=("pos",),
        policy_len=1,
        planner="gradient",
        epistemic=False,
    )
    transition = LinearGaussianTransition(
        F=jnp.eye(2),
        innovations=ConditionalGaussian(
            loc=jnp.zeros((1, 2)), chol=0.1 * jnp.eye(2)[None], parent_ndims=1
        ),
        G=jnp.full((2, width), 0.5),
    )
    params = SystemParams(
        transition={"x": transition},
        initial_state_prior={"x": Gaussian(loc=jnp.zeros(2), chol=0.1 * jnp.eye(2))},
        likelihood={
            "pos": LinearGaussianObservation(H=jnp.eye(2), c=jnp.zeros(2), noise=0.1 * jnp.eye(2))
        },
        control={"u_u": Box(-2.0 * jnp.ones(width), 2.0 * jnp.ones(width))},
    )
    return system, params


def test_a_continuous_action_node_drives_a_continuous_input_of_its_width():
    m_sys, m_par = _continuous(1)

    def push(key, obs, carry):
        return {"u": jnp.asarray([1.0], jnp.float32)}, carry

    wiring = couple(
        minds={"M": (m_sys, m_par, push)},
        processes={"P": process(*_continuous(1))},
        wires=[Wire("M.u", "P.u"), Wire("P.pos", "M.pos")],
    )
    assert wiring.types["M.u"] == wiring.types["P.u"] == ("float", 1)
    assert wiring.types["P.pos"] == wiring.types["M.pos"] == ("float", 2)
    record = run(wiring, ticks=4, key=KEY)
    assert record["M"].actions["u"].shape == (4, 1)
    assert equal(record["M"].realised["u"], record["M"].actions["u"])
    assert record["P"].states["x"].shape == (5, 2)
    assert equal(record["M"].observations["pos"], record["P"].observations["pos"])
    #  the drift is G u per tick on top of the noise: the mean step is (0.5, 0.5)
    drift = jnp.mean(jnp.diff(record["P"].states["x"], axis=0), axis=0)
    assert bool(jnp.all(drift > 0.25))
    w_sys, w_par = _continuous(2)
    with pytest.raises(ValueError, match=r"joins continuous, width 2 to continuous, width 1"):
        couple(
            minds={"M": (w_sys, w_par, push)},
            processes={"P": process(*_continuous(1))},
            wires=[Wire("M.u", "P.u"), Wire("P.pos", "M.pos")],
        )


# ------------------------------------------------------------------------------- 6. refusals


def _parts():
    m_sys, m_par = pusher()
    return {"M": (m_sys, m_par, constant(1))}, {"P": process(*pusher())}


GOOD = [Wire("M.push", "P.push"), Wire("P.see", "M.see")]


def test_an_unknown_participant_or_port_is_refused_with_the_candidates():
    minds, procs = _parts()
    with pytest.raises(ValueError, match=r"names participant 'Q'.*participants are \['M', 'P'\]"):
        couple(minds=minds, processes=procs, wires=[Wire("M.push", "Q.push"), GOOD[1]])
    with pytest.raises(ValueError, match=r"'P.shove' is not an in-port.*in-ports are \['P.push'\]"):
        couple(minds=minds, processes=procs, wires=[Wire("M.push", "P.shove"), GOOD[1]])
    with pytest.raises(ValueError, match=r"'M.see' is not an out-port.*a wire's target"):
        couple(minds=minds, processes=procs, wires=[Wire("M.see", "P.push"), GOOD[1]])
    with pytest.raises(ValueError, match="write a port as 'participant.port'"):
        couple(minds=minds, processes=procs, wires=[Wire("push", "P.push"), GOOD[1]])


def test_a_type_mismatch_is_refused_with_both_types():
    minds, procs = _parts()
    s_sys, s_par = _five()
    with pytest.raises(ValueError, match=r"joins discrete, 5 levels to discrete, 3 levels"):
        couple(
            minds={**minds, "F": (s_sys, s_par, constant(0))},
            processes=procs,
            wires=[*GOOD, Wire("F.push", "M.see")],
        )


def _five():
    """A pusher-shaped System over five levels."""
    table = jnp.stack([jnp.roll(jnp.eye(5), u, axis=0) for u in range(5)], axis=-1)
    system = System(
        A_deps=(),
        B_deps=((0, 1), ()),
        controls=(1,),
        action_names=("push",),
        factor_names=("x", "u_push"),
        policy_len=1,
        epistemic=False,
    )
    params = SystemParams(
        transition={"x": ConditionalCategorical(log_table=jnp.log(table), parent_ndims=2)},
        initial_state_prior={"x": Categorical(logits=jnp.zeros(5))},
    )
    return system, params


def test_two_wires_into_one_in_port_are_refused():
    minds, procs = _parts()
    procs["Q"] = process(*pusher())
    with pytest.raises(ValueError, match=r"'M.see' is the target of two wires.*takes one wire"):
        couple(
            minds=minds,
            processes=procs,
            wires=[*GOOD, Wire("Q.see", "M.see")],
            fixed={"Q.push": 0},
        )
    with pytest.raises(ValueError, match="both wired .* and in fixed="):
        couple(minds=minds, processes=procs, wires=GOOD, fixed={"P.push": 0, "Q.push": 0})


def test_an_unfed_in_port_is_refused_naming_its_fix():
    minds, procs = _parts()
    with pytest.raises(ValueError, match=r"process input 'P.push' is fed by no wire.*fixed="):
        couple(minds=minds, processes=procs, wires=[GOOD[1]])
    with pytest.raises(ValueError, match=r"mind modality 'M.see' is fed by no wire.*drop the"):
        couple(minds=minds, processes=procs, wires=[GOOD[0]])


def test_fixed_holds_only_a_process_input_of_its_type():
    minds, procs = _parts()
    with pytest.raises(ValueError, match=r"fixed= names 'M.see', which is not a process input"):
        couple(minds=minds, processes=procs, wires=GOOD, fixed={"M.see": 0})
    with pytest.raises(ValueError, match=r"fixed= names 'M.push', which is not a process input"):
        couple(processes=procs, fixed={"M.push": 0})
    with pytest.raises(ValueError, match=r"process input 'P.push' is discrete, 3 levels"):
        couple(processes=procs, fixed={"P.push": 3})
    with pytest.raises(ValueError, match=r"process input 'P.push' is discrete, 3 levels"):
        couple(processes=procs, fixed={"P.push": 1.0})


def test_an_actuator_on_a_process_is_refused_naming_the_mind():
    minds, _ = _parts()
    env, ep = pusher(kernel=KERNEL)[0].as_env(pusher(kernel=KERNEL)[1])
    with pytest.raises(ValueError, match=r"actuator belongs to the mind that emits the action"):
        couple(minds=minds, processes={"P": (env, ep)}, wires=GOOD)


def test_a_bare_positional_env_is_refused_naming_wrap():
    minds, _ = _parts()
    bare = DiscreteProcess(A_deps=((0,),), B_deps=((0,),))
    with pytest.raises(TypeError, match=r"GenerativeEnv.wrap"):
        couple(minds=minds, processes={"P": bare}, wires=GOOD)
    with pytest.raises(TypeError, match=r"GenerativeEnv.wrap"):
        couple(minds=minds, processes={"P": (bare, None)}, wires=GOOD)


def test_a_wire_between_two_processes_and_bad_names_are_refused():
    minds, procs = _parts()
    procs["Q"] = process(*pusher())
    with pytest.raises(ValueError, match=r"from a process's observation to a process's input"):
        couple(minds=minds, processes=procs, wires=[*GOOD, Wire("P.see", "Q.push")])
    with pytest.raises(ValueError, match=r"a non-empty string without a dot"):
        couple(minds={"M.1": minds["M"]}, processes=procs, wires=GOOD)
    with pytest.raises(ValueError, match=r"two participants are named 'M'"):
        couple(minds=minds, processes={"M": procs["P"]}, wires=GOOD)


# ----------------------------------------------------------------------------- 7. run's forms


def test_run_on_a_wiring_refuses_the_composite_arguments():
    minds, procs = _parts()
    wiring = couple(minds=minds, processes=procs, wires=GOOD)
    assert isinstance(wiring, Wiring)
    _, cp = pusher()
    sentence = r"run\(wiring, ticks=\.\.\., key=\.\.\.\) takes ticks and key by keyword"
    with pytest.raises(TypeError, match=sentence):
        run(wiring, ticks=2, key=KEY, params=cp)
    with pytest.raises(TypeError, match=sentence):
        run(wiring, ticks=2, key=KEY, minds={})
    with pytest.raises(TypeError, match=sentence):
        run(wiring, 2, KEY)
    with pytest.raises(TypeError, match=sentence):
        run(wiring, cp, minds, 2, KEY)
    with pytest.raises(TypeError, match="needs both ticks and key"):
        run(wiring, ticks=2)
    with pytest.raises(ValueError, match=r"carries names \['Z'\].*minds of this wiring"):
        run(wiring, ticks=2, key=KEY, carries={"Z": MindCarry()})
    with pytest.raises(ValueError, match=r"states names \['M'\].*processes of this wiring"):
        run(wiring, ticks=2, key=KEY, states={"M": {}})
    record = run(wiring, ticks=2, key=KEY, states={"P": {"x": jnp.asarray(2, jnp.int32)}})
    assert int(record["P"].states["x"][0]) == 2


def test_run_keeps_the_composite_signature_by_position_and_by_keyword():
    composite, cp, views, responds = _two_minds(False)
    by_position = run(composite, cp, responds, 2, KEY)
    by_keyword = run(composite=composite, params=cp, minds=responds, ticks=2, key=KEY)
    for f, rows in by_position.states.items():
        assert equal(by_keyword.states[f], rows)
    with pytest.raises(TypeError, match=r"is missing \['key'\]"):
        run(composite, cp, responds, 2)


# ------------------------------------------------------------ typing an outside process


class _Outside(DiscreteProcess):
    """A positional process whose observation is reshaped by ``emit``."""

    emit: object = None

    def reset(self, key, params, state=None):
        obs, s = super().reset(key, params, state)
        return self.emit(obs), s

    def step(self, key, state, action, params):
        obs, s = super().step(key, state, action, params)
        return self.emit(obs), s


class _OwnParams(eqx.Module):
    """An outside env's own params: the positional containers, not GenerativeEnvParams."""

    likelihood: tuple
    transition: tuple
    initial_state_prior: tuple
    actuator: dict


def _outside(emit):
    """The pusher's process as an outside env with its own params, emitting ``emit(obs)``."""
    system, params = pusher()
    env, ep = process(system, params)
    inner = _Outside(
        A_deps=env.inner.A_deps,
        B_deps=env.inner.B_deps,
        D_deps=env.inner.D_deps,
        B_control_deps=env.inner.B_control_deps,
        sensed=env.inner.sensed,
        emit=emit,
    )
    positional = env._params(ep)
    own = _OwnParams(
        positional.likelihood, positional.transition, positional.initial_state_prior, {}
    )
    return GenerativeEnv.wrap(inner, system, params), own


def test_a_port_with_no_level_count_or_a_shapeless_vector_is_refused():
    minds, _ = _parts()
    system, params = pusher()
    env, ep = process(system, params)
    #  wrapped without params, the inputs have no level count
    bare = GenerativeEnv.wrap(env.inner, system)
    with pytest.raises(ValueError, match=r"GenerativeEnv.wrap\(env, system, params\)"):
        couple(minds=minds, processes={"P": (bare, ep)}, wires=GOOD)
    #  with params, they do, and an outside env typed by GenerativeEnvParams couples
    typed = GenerativeEnv.wrap(env.inner, system, params)
    assert couple(minds=minds, processes={"P": (typed, ep)}, wires=GOOD).types["P.push"] == (
        "level",
        3,
    )
    #  an outside env's own params: a level with no level count is refused
    outside = _outside(lambda obs: obs)
    with pytest.raises(ValueError, match=r"emits 'see' as a level, and its params give no level"):
        couple(minds=minds, processes={"P": outside}, wires=GOOD)
    #  a float observation of no vector shape is refused with its shape
    scalar = _outside(lambda obs: tuple(o.astype(jnp.float32) for o in obs))
    with pytest.raises(ValueError, match=r"emits 'see' with shape \(\); a continuous port"):
        couple(minds=minds, processes={"P": scalar}, wires=GOOD)


def test_a_wired_mind_emitting_a_level_out_of_range_is_refused():
    minds, procs = _parts()
    minds["M"] = (*minds["M"][:2], constant(7))
    wiring = couple(minds=minds, processes=procs, wires=GOOD)
    with pytest.raises(ValueError, match=r"actions\['push'\] = 7; action node 'push' has 3 levels"):
        run(wiring, ticks=1, key=KEY)
