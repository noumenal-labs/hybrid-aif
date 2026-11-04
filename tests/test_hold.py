"""``composition.hold``: a mind that decides once per option, under both driver forms."""

import dataclasses
import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.composition import HoldCarry, MindCarry, agent_respond, couple, hold, merge, run
from hybrid_aif.composition.merge import subsystem

_PATH = pathlib.Path(__file__).resolve().parents[1] / "examples" / "compose_minds.py"
_SPEC = importlib.util.spec_from_file_location("compose_minds", _PATH)
cm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cm)

TICKS = 20
KEY = jax.random.PRNGKey(1)


def _slow(durations):
    system, params = cm.robot(2, 2)
    return system, dataclasses.replace(params, durations={"u_move": jnp.asarray(durations)})


def _counted(respond):
    calls = []

    def wrapped(key, obs, carry):
        calls.append(1)
        return respond(key, obs, carry)

    return wrapped, calls


def _boundaries(actions, rounds_of):
    """The decision rounds implied by the held actions: each option lasts its level's rounds."""
    out, t = [], 0
    while t < len(actions):
        out.append(t)
        t += rounds_of[int(actions[t])]
    return out


def _check(actions, calls, rounds_of):
    actions = np.asarray(actions)
    bounds = _boundaries(actions, rounds_of)
    for t in bounds:
        assert (actions[t : t + rounds_of[int(actions[t])]] == actions[t]).all()
    assert len(calls) == len(bounds)
    assert len(bounds) < len(actions)


def _wired(slow):
    (sa, pa), (sb, pb) = slow, cm.robot(2, 2)
    inner, calls = _counted(agent_respond(sa, pa, selection="stochastic"))
    mind_a = hold(sa, pa, inner)
    mind_b = agent_respond(sb, pb, selection="stochastic")
    wiring = couple(
        minds={"a": (sa, pa, mind_a), "b": (sb, pb, mind_b)},
        processes={"world": cm.world((2, 2))},
        wires=list(cm.WIRES),
    )
    return wiring, calls


def test_wired_form_redecides_only_at_option_boundaries():
    wiring, calls = _wired(_slow([4.0, 2.0, 4.0]))
    rec = run(wiring, ticks=TICKS, key=KEY)
    _check(rec["a"].actions["move"], calls, {0: 4, 1: 2, 2: 4})


@pytest.mark.parametrize("order", ["simultaneous", "sequential"])
def test_composite_form_redecides_only_at_option_boundaries(order):
    """The cadence of the held mind under ``run`` over a merged System.

    The slow mind here owns world factors (positions stepped by the option's kernel), the
    configuration the record's R8 calls invalid for a composite, since the composite environment
    is the merged System's own generative process and applies that kernel once per fast round.
    ``merge`` has no separate world-process slot to move those factors into, so a belief-only
    slow mind beside a world process is expressible in the wired form (the test above) and not
    here; this test checks the call cadence only, not the dynamics.
    """
    (sa, pa), (sb, pb) = _slow([4.0, 2.0, 4.0]), cm.robot(2, 2)
    comp, cp = merge({"a": sa, "b": sb}, {"a": pa, "b": pb}, list(cm.LINKS))
    va, vb = subsystem(comp, cp, "a"), subsystem(comp, cp, "b")
    inner, calls = _counted(agent_respond(*va, selection="stochastic"))
    minds = {
        "a": hold(*va, inner),
        "b": agent_respond(*vb, selection="stochastic"),
    }
    tr = run(comp, cp, minds, TICKS, KEY, tick_order=order)
    _check(tr.actions["a.move"], calls, {0: 4, 1: 2, 2: 4})


def test_ticks_per_unit_converts_the_unit_to_rounds():
    system, params = _slow([1.5, 1.5, 1.5])
    inner, calls = _counted(agent_respond(system, params, selection="stochastic"))
    held = hold(system, params, inner, ticks_per_unit=2)
    # drive the wrapper directly: it decides on rounds 0, 3, 6 and holds in between
    carry = MindCarry()
    obs = {m: jnp.zeros((), jnp.int32) for m in system.names.modalities}
    seen = []
    for t in range(9):
        action, carry = held(jax.random.PRNGKey(t), obs, carry)
        seen.append((len(calls), int(action["move"])))
        assert isinstance(carry, HoldCarry)
    assert [n for n, _ in seen] == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    for start in (0, 3, 6):
        assert len({a for _, a in seen[start : start + 3]}) == 1


def test_a_carries_seed_decides_on_the_first_round():
    wiring, calls = _wired(_slow([4.0, 2.0, 4.0]))
    rec = run(wiring, ticks=1, key=KEY, carries={"a": MindCarry()})
    assert len(calls) == 1
    assert rec["a"].actions["move"].shape == (1,)


def test_a_duration_that_is_not_a_whole_number_of_rounds_is_refused():
    system, params = _slow([1.5, 1.5, 1.5])
    held = hold(system, params, agent_respond(system, params))
    obs = {m: jnp.zeros((), jnp.int32) for m in system.names.modalities}
    with pytest.raises(ValueError, match="not a whole number of rounds"):
        held(KEY, obs, MindCarry())


def test_a_stochastic_actuator_on_the_durations_node_is_refused():
    system, params = _slow([2.0, 2.0, 2.0])
    kernel = jnp.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    params = dataclasses.replace(params, actuator={"move": kernel})
    with pytest.raises(ValueError, match="non-identity actuator.*T9"):
        hold(system, params, agent_respond(system, params))


def test_a_permutation_actuator_is_refused_though_it_is_deterministic():
    system, params = _slow([2.0, 2.0, 2.0])
    params = dataclasses.replace(params, actuator={"move": jnp.asarray(np.eye(3)[[1, 2, 0]])})
    with pytest.raises(ValueError, match="non-identity actuator"):
        hold(system, params, agent_respond(system, params))


def test_a_system_with_no_durations_is_refused():
    system, params = cm.robot(2, 2)
    with pytest.raises(ValueError, match="declares no params.durations"):
        hold(system, params, agent_respond(system, params))


def test_elapsed_is_legal_under_jit_and_cond():
    system, params = _slow([4.0, 2.0, 4.0])

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


def test_a_non_positive_duration_is_still_refused_when_concrete():
    system, params = _slow([4.0, 0.0, 4.0])
    with pytest.raises(ValueError, match="non-positive"):
        system.elapsed({"move": jnp.asarray(0)}, params)
