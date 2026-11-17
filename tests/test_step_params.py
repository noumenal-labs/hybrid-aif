"""Params flow through the runner: a step is handed its view's params each round and returns them;
a learner's returned tables are written back to the parameter home, the next view is taken from
it, and ``Trajectory.params`` is the home after the last round (the starting object if no step
learned).

The two robots of ``examples/compose_minds.py`` coupled to the corridor, a closed System.
``tests/data/migration_particle.npz`` holds what the wired form of ``run`` recorded for the robots
when neither learns, compared here under that form's key layout.
"""

import dataclasses
import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.composition import couple, plan, run, sample

_PATH = pathlib.Path(__file__).resolve().parents[1] / "examples" / "compose_minds.py"
_SPEC = importlib.util.spec_from_file_location("compose_minds_step_params", _PATH)
cm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cm)

KEY = jax.random.PRNGKey(3)
TICKS = 5
FIXTURE = pathlib.Path(__file__).parent / "data" / "migration_particle.npz"
SLOT = "a.me_obs"  # the table the learner counts rounds in


def _offset(params) -> float:
    """The rounds a learner has counted: the shift of every log entry of robot ``a``'s
    ``me_obs`` table, which leaves the likelihood it encodes as it was."""
    table = params.likelihood[SLOT].log_table
    return float(table.ravel()[0] - _START)


def _counting(inner):
    """``inner`` that also counts rounds in a shift of its own ``me_obs`` table: a learner."""

    def step(system, key, obs, carry, params):
        action, carry, _ = inner(system, key, obs, carry, params)
        if obs is None:
            return action, carry, params
        table = params.likelihood[SLOT]
        moved = dataclasses.replace(table, log_table=table.log_table + 1.0)
        return (
            action,
            carry,
            dataclasses.replace(params, likelihood={**params.likelihood, SLOT: moved}),
        )

    step.sliced = True
    return step


def _robots():
    (sa, pa), (sb, pb) = cm.robot(2, 2), cm.robot(2, 2)
    pa = dataclasses.replace(pa, precision=jnp.asarray(1.0))
    return couple({"a": (sa, pa), "b": (sb, pb), "world": cm.world((2, 2))}, list(cm.BINDINGS))


_START = float(_robots()[1].likelihood[SLOT].log_table.ravel()[0])


def _steps(a=None, b=None):
    return {"a": a or plan(), "b": b or plan(), "world": sample()}


def test_a_learner_s_tables_are_written_back_and_the_trajectory_holds_the_home():
    system, params = _robots()
    traj = run(system, params, TICKS, KEY, _steps(a=_counting(plan())))
    assert _offset(traj.params) == TICKS
    assert traj.params is not params
    assert _offset(params) == 0.0  # the starting params are not mutated


def test_a_learner_is_handed_what_it_returned_the_round_before():
    system, params = _robots()
    seen = []
    inner = _counting(plan())

    def watcher(system, key, obs, carry, params):
        if obs is not None:
            #  logged on the host: the rounds after the first run in one compiled scan
            jax.debug.callback(lambda p: seen.append(_offset(p)), params, ordered=True)
        return inner(system, key, obs, carry, params)

    watcher.sliced = True
    run(system, params, TICKS, KEY, _steps(a=watcher))
    assert seen == [float(t) for t in range(TICKS)]


def test_a_run_in_which_no_step_learns_holds_the_starting_object_and_the_wired_form_s_moves():
    """Neither robot learns: ``Trajectory.params`` is the very object handed in. Under the key
    layout of the wired form, the moves and the corridor are what that form recorded."""
    from tests.test_run_parity import _run_old_keys, states

    system, params = _robots()
    traj = run(system, params, TICKS, KEY, _steps())
    assert traj.params is params
    fixture = np.load(FIXTURE)
    traj = _run_old_keys(2, system, params, TICKS, KEY, _steps(), record_carries=True)
    for m in "ab":
        np.testing.assert_array_equal(traj[m].intended["move"], fixture[f"step_params/{m}/move"])
    np.testing.assert_array_equal(states(traj, "world.a"), fixture["step_params/world/a"])
    np.testing.assert_array_equal(states(traj, "world.b"), fixture["step_params/world/b"])


def test_a_run_of_no_rounds_holds_the_starting_params():
    system, params = _robots()
    traj = run(system, params, 0, KEY, _steps(a=_counting(plan())))
    assert traj.params is params


def test_a_habit_draw_returns_the_params_it_was_given():
    from tests.test_run_parity import ancestral

    system, params = _robots()
    traj = run(system, params, TICKS, KEY, _steps(b=ancestral()))
    assert traj.params is params


def test_a_step_that_returns_something_else_as_params_is_refused_naming_the_node():
    system, params = _robots()
    inner = plan()

    def broken(system, key, obs, carry, params):
        action, carry, _ = inner(system, key, obs, carry, params)
        return action, carry, None if obs is not None else params

    broken.sliced = True
    with pytest.raises(ValueError, match="step of 'a' returned NoneType as its params"):
        run(system, params, 2, KEY, _steps(a=broken))
