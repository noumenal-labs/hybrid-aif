"""The state ``run`` settles from the reset, read off the structure or by a trace of round 0.

``run`` brings the state straight from the reset into the form of every round's state before the
scan (design batched-run.md, Phase C, *Settle without a trace*). Where every form is derivable
(``plan()`` nodes without ``trace=True``, and steps that return each out-port at their reset) it
reads them off the structure; otherwise it evaluates round 0 abstractly. The two paths settle
one state: pinned here on the pusher pair and the one-level survey, with the paths each run
takes (``Trajectory.timing["settled"]``) and the refusal of a reset carry of another form.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.composition import plan, run, runner, sample, with_mirror
from tests.test_action_nodes import pusher
from tests.test_batched_run import pushers
from tests.test_merge import agent_and_world
from tests.test_run import KEY, level, wire
from tests.test_run_nested import _shared
from tests.test_survey_run import STARTS, STEPS, as_recorded, case_setting, example, small_camera


class _Settled(Exception):  # noqa: N818 - the capture's signal
    pass


def settled(monkeypatch, how, call):
    """The state ``call`` (a function running ``run``) settles from the reset, by the path
    ``how``, captured before the scan is compiled: ``"structure"`` asserts it read off the
    structure (its traces read off round 0 of that state), ``"traced"`` forces round 0 of the
    state from the reset."""
    held = {}

    def scan(self, state, strict=False):
        held["state"] = state
        raise _Settled

    settling = runner._settling
    with monkeypatch.context() as patch:
        patch.setattr(runner._Simulation, "_scan", scan)
        if how == "traced":
            patch.setattr(runner, "_settling", lambda engine, state: runner.TRACED)
        else:

            def structure(engine, state):
                path = settling(engine, state)
                assert path != runner.TRACED
                return path

            patch.setattr(runner, "_settling", structure)
        with pytest.raises(_Settled):
            call()
    return held["state"]


def forms(state):
    """The state's tree structure and each leaf's shape and dtype (a leaf that is no array as
    itself)."""
    leaves, treedef = jax.tree_util.tree_flatten(state)
    return treedef, [
        jax.ShapeDtypeStruct(x.shape, x.dtype, weak_type=x.weak_type) if eqx.is_array(x) else x
        for x in leaves
    ]


def survey():
    """The one-level survey, the float32 model its pin was recorded with (in either mode)."""
    module = example("survey_run")
    setting, ticks = case_setting(module, "one-level")
    camera = small_camera(example("survey_world").IMG_PIXELS, dtype=jnp.float32)
    with as_recorded(module):
        return module.run_setting(setting, ticks, camera, STEPS, STARTS)


def pusher_pair(order, batch=None):
    system, params = wire()
    return run(system, params, 3, KEY, pushers(), tick_order=order, batch=batch)


def learners():
    composite, cp, _, _ = _shared()
    system, params = with_mirror(composite, cp)
    steps = {"model.rover": plan(learn=True), "model.mission": plan(), "world": sample()}
    return run(system, params, 2, KEY, steps)


def traced_plan(batch=None):
    (agent, ap), _ = agent_and_world(policy_len=1)
    system, params = with_mirror(agent, ap)
    steps = {"model": plan(trace=True), "world": sample()}
    return run(system, params, 2, KEY, steps, batch=batch)


CASES = {
    "pushers-simultaneous": lambda: pusher_pair("simultaneous"),
    "pushers-sequential": lambda: pusher_pair("sequential"),
    "pushers-async": lambda: pusher_pair("async"),
    "pushers-batch-3": lambda: pusher_pair("simultaneous", batch=3),
    "learners": learners,
    "traced-plan": traced_plan,
    "traced-plan-batch-2": lambda: traced_plan(batch=2),
    "one-level-survey": survey,
}


@pytest.mark.parametrize("case", list(CASES))
def test_the_state_settled_from_the_structure_is_the_state_settled_by_the_trace(monkeypatch, case):
    """One settle contract: the state read off the structure has the tree structure, shapes and
    dtypes of the state round 0's trace gives, a plan() carry's every field included, and holds
    the same values (zeros where round 0 sets one, the reset's elsewhere)."""
    static = settled(monkeypatch, "structure", CASES[case])
    traced = settled(monkeypatch, "traced", CASES[case])
    (static_def, static_leaves), (traced_def, traced_leaves) = forms(static), forms(traced)
    assert static_def == traced_def
    assert static_leaves == traced_leaves
    for x, y in zip(
        jax.tree_util.tree_leaves(static), jax.tree_util.tree_leaves(traced), strict=True
    ):
        if eqx.is_array(x) and not jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_plan_minds_and_steps_returning_every_out_port_settle_without_a_trace():
    """The pusher pair (plan() minds and a sampled process) settles from the structure, in
    every pass of a batch; ``timing["settle"]`` stays the seconds."""
    traj = pusher_pair("sequential")
    assert traj.timing["settled"] == "static"
    assert isinstance(traj.timing["settle"], float)
    system, params = wire()
    traj = run(system, params, 2, KEY, pushers(), batch=2, replicas_per_pass=1)
    assert traj.timing["settled"] == "static"
    assert run(system, params, 0, KEY, pushers()).timing["settled"] is None


def test_a_traced_plan_s_trace_is_read_off_round_0_of_the_state_settled_from_the_structure():
    """A ``plan(trace=True)`` node: everything but its :class:`Trace` from the structure, the
    trace (the form of the planner's search) off an abstract round 0 of that state."""
    assert traced_plan().timing["settled"] == "static+trace"
    assert traced_plan(batch=2).timing["settled"] == "static+trace"


def test_a_step_leaving_an_out_port_without_a_value_at_its_reset_settles_by_the_trace():
    """A custom step that returns nothing at its reset gives its out-port a form of its own in
    round 0: round 0 of the state from the reset is evaluated abstractly."""
    system, params = wire()
    custom = {**pushers(), "A": level(1)}
    assert run(system, params, 2, KEY, custom).timing["settled"] == "traced"


def counting(reset, every):
    """A step on the pusher's model choosing level 1 from its reset on, its carry ``reset`` at
    the reset and ``every(carry)`` each round."""

    def step(system, key, obs, carry, params):
        out = {"model.push": jnp.asarray(1, jnp.int32)}
        if obs is None:
            return out, reset, params
        return out, every(carry), params

    return step


def test_a_reset_carry_of_another_form_than_round_0_s_is_refused_naming_the_step():
    """A custom step whose reset carry differs from round 0's in shape (a Python float, then a
    vector) or in structure is refused naming the step and the fix, as under the trace."""
    system, params = with_mirror(*pusher())
    for reset, every in (
        (0.0, lambda carry: jnp.zeros(3) + carry),
        ({"n": 0.0}, lambda carry: {"n": carry["n"], "m": carry["n"]}),
    ):
        steps = {"model": counting(reset, every), "world": sample()}
        with pytest.raises(ValueError, match="'model' returns at its reset") as error:
            run(system, params, 2, KEY, steps)
        assert "return from the reset a carry of the structure, shapes and dtypes" in str(
            error.value
        )


def test_a_carry_whose_dtype_round_0_changes_is_settled_by_the_trace_as_before():
    """A reset carry the structure gives the wrong dtype (an integer that round 0 makes a
    float) is found on the scan's trace, and the state is settled by round 0's trace instead:
    the carry is round 0's dtype from the reset on, as it was before the static path."""
    system, params = with_mirror(*pusher())
    steps = {"model": counting(0, lambda carry: carry + 0.5), "world": sample()}
    traj = run(system, params, 3, KEY, steps, record_carries=True)
    assert traj.timing["settled"] == "traced"
    carries = [np.asarray(c) for c in traj["model"].carries]
    #  round 0's float is the default float dtype (a Python 0.5 beside an integer)
    default = np.float64 if jax.config.jax_enable_x64 else np.float32
    assert [c.dtype for c in carries] == [default] * 4
    np.testing.assert_array_equal(carries, [0.0, 0.5, 1.0, 1.5])
