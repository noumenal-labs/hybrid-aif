"""``plan()``'s receding-horizon warm start and what ``plan(trace=True)`` records.

``plan(warm_start=True)`` keeps the continuous controls of the plan each round acted on in the
carry (``MindCarry.plan``) and hands the planner that plan shifted one step on (its steps after
the first, the last step repeated) as its ``init``; the first round starts from zeros. A node that
holds an option shifts once per decision. A planner that takes no ``init`` is refused. The trace
holds the planner's policy set, the inference detail and the parameter variables the round planned
with, under ``run`` and under ``jax.jit``.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Categorical, LinearGaussianObservation
from hybrid_aif.composition import Identify, MindCarry, Trace, black_box, couple, plan, run
from hybrid_aif.controls import FLOAT
from hybrid_aif.planning import (
    make_gradient_planner,
    make_mixed_planner,
    planner_takes_init,
)
from hybrid_aif.system import System, SystemParams
from tests.test_continuous_nodes import G2, _box, _gauss, _lg
from tests.test_mixed_planner import toy
from tests.test_run import COUNTS, shared_field

KEY = jax.random.PRNGKey(0)
HORIZON = 3
BUMP = 0.1 * jnp.arange(1, HORIZON + 1, dtype=jnp.float32)[:, None]  # (H, 1)


def shifted(plan):
    return np.concatenate([plan[1:], plan[-1:]], axis=0)


def spy(inner):
    """A planner that takes an ``init`` and returns ``inner``'s plan from ``init + BUMP`` (zero
    descent steps return the start itself), so the plan acted on shows the ``init`` handed in."""

    def planner(ctx, use_bound="midpoint", metrics=True, init=None):
        return inner(ctx, use_bound=use_bound, metrics=metrics, init=init + BUMP)

    planner.needs_key = False
    planner.takes_init = True
    return planner


def steered():
    """A state ``x`` moved by a continuous node ``u`` (box ``[-2, 2]``), observed through ``y``,
    coupled to a world that takes the command and returns a reading: the closed System."""
    mind = System(
        A_deps=((0,),),
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("u",),
        state_names=("x",),
        control_names=("u_u",),
        modality_names=("y",),
        policy_len=HORIZON,
        planner="gradient",
        epistemic=False,
    )
    params = SystemParams(
        likelihood={
            "y": LinearGaussianObservation(H=jnp.eye(2), c=jnp.zeros(2), noise=0.3 * jnp.eye(2))
        },
        transition={"x": _lg(jnp.zeros((1, 2)), G=G2)},
        initial_state_prior={"x": _gauss()},
        state_targets={"x": _gauss((1.0, 0.0), 1.0)},
        control={"u_u": _box()},
    )
    world = black_box(ins={"v": (FLOAT, 1)}, outs={"y": (FLOAT, 2)})
    return couple(
        {"mind": (mind, params), "world": world},
        [Identify("mind.u", "world.v"), Identify("world.y", "mind.y")],
    )


def reading(system, key, obs, carry, params):
    """The world's step: a fixed reading every round."""
    return {"world.y": jnp.asarray([0.5, 0.1], jnp.float32)}, carry, params


def run_mind(step, rounds=4, system_params=None):
    system, params = system_params or steered()
    return run(system, params, rounds, KEY, {"mind": step, "world": reading}, record_carries=True)


def test_the_option_off_keeps_no_plan_and_the_planners_take_an_init():
    rec = run_mind(plan(make_gradient_planner(steps=2), trace=True))
    assert all(c.plan is None for c in rec["mind"].carries)
    assert planner_takes_init(make_gradient_planner()) and planner_takes_init(make_mixed_planner())
    from hybrid_aif.planning import get_planner

    assert planner_takes_init(get_planner("gradient")) and planner_takes_init(get_planner("mixed"))
    assert not planner_takes_init(get_planner("enumeration"))


def test_the_planner_starts_from_zeros_then_from_the_plan_it_acted_on_shifted_one_step():
    step = plan(spy(make_gradient_planner(steps=0)), trace=True, warm_start=True)
    carries = run_mind(step, rounds=4)["mind"].carries[1:]
    expected = np.zeros((HORIZON, 1), np.float32)
    for carry in carries:
        init = expected
        acted = np.clip(init + np.asarray(BUMP), -2.0, 2.0)
        np.testing.assert_allclose(np.asarray(carry.plan), acted, rtol=0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(carry.trace.plans)[0], acted, rtol=0, atol=1e-6)
        expected = shifted(acted)
    # the plan the mind acted on is its first step
    first = np.asarray(carries[-1].action["mind.u"])
    np.testing.assert_allclose(first, np.asarray(carries[-1].plan)[0], atol=1e-6)


def test_the_warm_start_reaches_the_descent_through_the_registered_planner_too():
    """``planner=None`` resolves the view's registered ``"gradient"`` planner, which takes the
    shifted plan as its ``init``: the run is the one of the planner made with the same defaults."""
    registered = run_mind(plan(trace=True, warm_start=True), rounds=3)["mind"].carries
    made = run_mind(plan(make_gradient_planner(), trace=True, warm_start=True), rounds=3)
    assert registered[0].plan is None
    for a, b in zip(registered[1:], made["mind"].carries[1:], strict=True):
        assert a.plan.shape == (HORIZON, 1)
        np.testing.assert_allclose(np.asarray(a.plan), np.asarray(b.plan), atol=1e-6)


def test_a_held_node_shifts_its_plan_once_per_decision():
    """The mixed planner's node with durations of two rounds decides at rounds 0, 2 and 4: each
    decision starts from the plan of the decision before, shifted one step, not one per round."""
    mind, mparams = toy(HORIZON)
    mparams = dataclasses.replace(mparams, durations={"u_d": jnp.full(3, 2.0)})
    world = black_box(ins={"v": (FLOAT, 1)}, outs={"o": (FLOAT, 1)})
    system, params = couple(
        {"mind": (mind, mparams), "world": world}, [Identify("mind.c", "world.v")]
    )

    def idle(system, key, obs, carry, params):
        return {"world.o": jnp.zeros(1, jnp.float32)}, carry, params

    step = plan(spy(make_mixed_planner(steps=0)), trace=True, warm_start=True)
    rec = run(system, params, 6, KEY, {"mind": step, "world": idle}, record_carries=True)
    node = rec["mind"]
    decided = [t for t in range(6) if bool(node.decided[t])]
    assert decided == [0, 2, 4]
    expected = np.zeros((HORIZON, 1), np.float32)
    for t in decided:
        carry = node.carries[t + 1]
        acted = np.clip(expected + np.asarray(BUMP), -2.0, 2.0)
        np.testing.assert_allclose(np.asarray(carry.plan), acted, atol=1e-6)
        #  held: the carry stands (the record of a compiled run holds equal values, not the
        #  same object)
        np.testing.assert_array_equal(carry.plan, node.carries[t + 2].plan)
        expected = shifted(acted)


def test_a_planner_that_takes_no_init_is_refused():
    def plain(ctx, use_bound="midpoint", metrics=True):
        raise AssertionError("never planned")

    with pytest.raises(ValueError, match="takes no init"):
        plan(plain, warm_start=True)
    system, params = shared_field(COUNTS)  # discrete nodes: the enumerating planner
    with pytest.raises(ValueError, match="'enumeration' takes no init"):
        run(system, params, 1, KEY, {"A": plan(warm_start=True), "B": plan()})


def test_the_trace_holds_the_planners_set_the_inference_detail_and_the_parameters():
    rec = run_mind(plan(make_gradient_planner(steps=3, starts=2), trace=True), rounds=2)
    carry = rec["mind"].carries[-1]
    trace = carry.trace
    assert isinstance(trace, Trace) and trace.policies is None
    assert trace.plans.shape == (2, HORIZON, 1)  # every start's final plan
    assert trace.q_pi.probs.shape == (2,)
    np.testing.assert_allclose(
        np.asarray(carry.action["mind.u"]),
        np.asarray(trace.plans[int(jnp.argmax(trace.q_pi.probs)), 0]),
        atol=1e-6,
    )
    assert np.isfinite(float(trace.detail.log_evidence["mind.y"]))
    assert np.isfinite(float(trace.detail.free_energy))
    assert trace.parameters == {}  # the mind declares no parameter variable
    # an enumerated plan: the planner's set is the enumeration itself
    system, params = shared_field(COUNTS)
    target = Categorical(logits=jnp.log(jnp.asarray([0.6, 0.3, 0.1])))
    params = dataclasses.replace(params, state_targets={"A.s1": target, "B.s1": target})
    rec = run(
        system,
        params,
        3,
        KEY,
        {"A": plan(learn=True, trace=True), "B": plan()},
        tick_order="sequential",
        record_carries=True,
    )
    trace = rec["A"].carry.trace
    np.testing.assert_array_equal(np.asarray(trace.plans), np.asarray(trace.policies))
    learned = trace.parameters["A.field"].concentration
    np.testing.assert_allclose(learned, rec.params.parameters["A.field"].concentration)
    assert not np.allclose(learned, COUNTS.concentration)  # what the round learned


def test_the_traced_step_runs_under_jit():
    system, params = steered()
    step = plan(make_gradient_planner(steps=2, starts=2), trace=True, warm_start=True)
    from hybrid_aif.composition import subsystem

    view, vparams = subsystem(system, params, "mind")
    _, carry, _ = step(view, KEY, None, None, vparams)
    obs = {"mind.y": jnp.asarray([0.5, 0.1], jnp.float32)}

    @jax.jit
    def twice(carry):
        out, carry, _ = step(view, KEY, obs, carry, vparams)
        out, carry, _ = step(view, KEY, obs, carry, vparams)
        return out, carry

    out, carry = twice(carry)
    assert isinstance(carry, MindCarry) and carry.plan.shape == (HORIZON, 1)
    assert carry.trace.plans.shape == (2, HORIZON, 1)
    assert isinstance(carry.trace.q_pi, Categorical)
    assert np.isfinite(float(carry.trace.detail.free_energy))
