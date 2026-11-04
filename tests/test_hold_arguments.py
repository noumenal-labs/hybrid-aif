"""``hold``'s argument checks, the duration arithmetic, and carry resumption."""

import dataclasses
import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.composition import HoldCarry, MindCarry, hold

_PATH = pathlib.Path(__file__).resolve().parents[1] / "examples" / "compose_minds.py"
_SPEC = importlib.util.spec_from_file_location("compose_minds_args", _PATH)
cm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cm)

KEY = jax.random.PRNGKey(0)


def _timed(durations):
    system, params = cm.robot(2, 2)
    return system, dataclasses.replace(params, durations={"u_move": jnp.asarray(durations)})


def _stub(calls):
    def respond(key, obs, carry):
        calls.append(1)
        return {"move": jnp.asarray(0, jnp.int32), "look": jnp.asarray(0, jnp.int32)}, carry

    return respond


def test_a_hold_carry_seed_resumes_its_count():
    system, params = _timed([4.0, 4.0, 4.0])
    calls = []
    held = hold(system, params, _stub(calls))
    seed = HoldCarry(MindCarry(), 2, {"move": jnp.asarray(1)})
    action, carry = held(KEY, {}, seed)
    assert calls == [] and int(action["move"]) == 1 and carry.remaining == 1


def test_a_float32_duration_times_ticks_per_unit_passes():
    system, params = _timed([33.3, 33.3, 33.3])
    held = hold(system, params, _stub([]), ticks_per_unit=10)
    _, carry = held(KEY, {}, MindCarry())
    assert carry.remaining == 332


@pytest.mark.parametrize(
    "good", [np.int64(2), np.float32(2.0), jnp.asarray(2), np.asarray(2.0), 2, 2.0]
)
def test_ticks_per_unit_is_any_real_scalar(good):
    system, params = _timed([1.0, 1.0, 1.0])
    held = hold(system, params, _stub([]), ticks_per_unit=good)
    _, carry = held(KEY, {}, MindCarry())
    assert carry.remaining == 1


@pytest.mark.parametrize("bad", [0, -1, "2", None, True, float("nan")])
def test_a_bad_ticks_per_unit_is_refused_at_construction(bad):
    system, params = _timed([2.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="ticks_per_unit"):
        hold(system, params, _stub([]), ticks_per_unit=bad)


def test_a_duration_shorter_than_a_round_has_its_own_sentence():
    system, params = _timed([0.25, 0.25, 0.25])
    held = hold(system, params, _stub([]))
    with pytest.raises(ValueError, match="shorter than one round"):
        held(KEY, {}, MindCarry())


def test_an_identity_kernel_is_accepted():
    system, params = _timed([2.0, 2.0, 2.0])
    params = dataclasses.replace(params, actuator={"move": jnp.eye(3)})
    hold(system, params, _stub([]))


@pytest.mark.parametrize("value", [jnp.asarray([2.0, 0.0, 1.0]), [2.0, -1.0, 1.0]])
def test_a_non_positive_duration_is_refused(value):
    system, params = cm.robot(2, 2)
    params = dataclasses.replace(params, durations={"u_move": value})
    with pytest.raises(ValueError, match="non-positive"):
        system._durations(system._params_in(params, "t"), "t")
