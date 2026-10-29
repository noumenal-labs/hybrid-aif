"""The gradient planner's normalisation floor and its best-iterate return.

Design: docs/design/gradient-planner-floor.md.

Adam's denominator is floored at ``sqrt(eps(dtype)) · RMS(gradient)`` and the planner returns the
plan with the lowest ``G`` it scored. The previous rule (an absolute guard of ``1e-8`` and the
last iterate returned) is held here as a reference descent, and the tests check

- the iterates are the previous rule's, bitwise, for as long as the floor does not bind;
- a returned ``G`` is never above the warm start's, and is at most the previous rule's returned
  ``G`` up to float rounding, over a battery of models and seeded warm starts;
- a coordinate whose gradient is far below the others' takes no appreciable step;
- the returned plan is the best one scored, with its own ``G``, tree and curvature.

The tests that read a recorded downstream model are skipped unless ``HYBRID_AIF_FLOOR_DOWNSTREAM``
names a directory holding ``e5model.py``, ``cap014.pkl`` and ``cap015.pkl``.
"""

import dataclasses
import functools
import importlib.util
import os
import pathlib
import types
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Box, Interval, NonlinearGaussianTransition
from hybrid_aif.controls import ControlLayout
from hybrid_aif.objectives.base import total
from hybrid_aif.planning import gradient_plan
from hybrid_aif.planning.base import PlannerContext
from hybrid_aif.planning.enumerate import evaluate_policies_total
from hybrid_aif.planning.gradient import _ADAM_B1, _ADAM_B2, DEFAULT_LR_FRACTION
from tests.test_closed_loop import SMALL_PRESENT, lane, small_lg, unicycle
from tests.test_lq_control import lq
from tests.test_noise_map import _chol, _point_mass
from tests.test_nonlinear_transition import (
    FROZEN_F,
    PLAN_H,
    PLAN_Q,
    PLAN_V,
    PLAN_W,
    RateUnicycle,
    _noise,
    _steered,
)
from tests.test_posterior_chain_cost import e5_model
from tests.test_shared_innovation_noise import _shared_lg

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOWNSTREAM = os.environ.get("HYBRID_AIF_FLOOR_DOWNSTREAM", "")


class Case(NamedTuple):
    name: str
    ctx: PlannerContext
    init: jax.Array | None
    steps: int


# ------------------------------------------------------------------------------ the references


class Descent(NamedTuple):
    """One reference descent: each plan scored (``(steps, T, d)``), its ``G`` (the trace), its
    gradient, the final plan, and per step whether the floor binds (some coordinate with a nonzero
    first moment has a root second moment below it)."""

    plans: np.ndarray
    trace: np.ndarray
    grads: np.ndarray
    last: np.ndarray
    binding: np.ndarray


def _score_fn(ctx):
    def score(plan):
        G = evaluate_policies_total(
            plan[None],
            ctx.init_beliefs,
            ctx.transition_fn,
            ctx.step_terms_fn,
            ctx.terminal_terms_fn,
        )
        return G.midpoint[0]

    return score


def g_of(ctx, plan) -> float:
    """``G`` of ``plan`` (the midpoint of its interval) under the planner context."""
    return float(_score_fn(ctx)(jnp.asarray(plan)))


def tolerance(g: float) -> float:
    """``1e-5`` plus four float32 spacings at ``g``: two evaluations of the same ``G`` in
    different compiled programs agree to this."""
    return 1e-5 + 4.0 * float(np.spacing(np.float32(abs(g))))


def reference_descent(ctx, rule, steps, init=None, lr=None) -> Descent:
    """Adam over the plan with every iterate kept. ``rule="old"`` normalises by ``sqrt(v̂) + 1e-8``
    and ``rule="new"`` by ``max(sqrt(v̂), sqrt(eps) · RMS(g)) + 1e-8``."""
    layout = ctx.continuous_controls
    low, high = layout.bounds
    rate = DEFAULT_LR_FRACTION * (high - low) if lr is None else jnp.asarray(lr, low.dtype)
    shape = (ctx.policy_len, layout.action_dim)
    start = 0.5 * (low + high) if init is None else jnp.asarray(init, low.dtype)
    plan0 = jnp.clip(jnp.broadcast_to(start, shape), low, high)
    value_and_grad = jax.value_and_grad(_score_fn(ctx))

    def step(carry, t):
        plan, m, v = carry
        g_value, grad = value_and_grad(plan)
        m = _ADAM_B1 * m + (1.0 - _ADAM_B1) * grad
        v = _ADAM_B2 * v + (1.0 - _ADAM_B2) * grad**2
        m_hat = m / (1.0 - _ADAM_B1 ** (t + 1.0))
        v_hat = v / (1.0 - _ADAM_B2 ** (t + 1.0))
        root = jnp.sqrt(v_hat)
        floor = jnp.sqrt(jnp.finfo(plan.dtype).eps) * jnp.sqrt(jnp.mean(grad**2))
        scale = root + 1e-8 if rule == "old" else jnp.maximum(root, floor) + 1e-8
        update = (1.0 - t / max(steps, 1)) * m_hat / scale
        new = jnp.clip(plan - rate * update, low, high)
        return (new, m, v), (plan, g_value, grad, jnp.any((root < floor) & (m_hat != 0.0)))

    zeros = jnp.zeros_like(plan0)
    (last, _, _), out = jax.lax.scan(
        step, (plan0, zeros, zeros), jnp.arange(steps, dtype=low.dtype)
    )
    plans, trace, grads, binding = out
    return Descent(*(np.asarray(x) for x in (plans, trace, grads, last, binding)))


# ---------------------------------------------------------------------------------- the cases


def _planned(system, params, present=None):
    return system.planner_context(params, system.initial_priors(params), present=present)


def _zeros(system):
    return jnp.zeros((system.policy_len, 1))


def _lq():
    system, params = lq.make_lq()
    return Case("lq_control", _planned(system, params), None, 200)


def _small_lg():
    system, params = small_lg()
    return Case("small_lg", _planned(system, params, SMALL_PRESENT), None, 200)


def _shared():
    system, _, params = _shared_lg()
    return Case("shared_noise_lg", _planned(system, params, SMALL_PRESENT), None, 100)


def _lane():
    system, params, present = lane(0.05, True)
    return Case("lane", _planned(system, params, present), _zeros(system), 150)


def _unicycle():
    system, params = unicycle("first_order")
    return Case("unicycle", _planned(system, params), _zeros(system), 150)


def _rate_turn(kind):
    over = dict(
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("w",),
        factor_names=("pose", "u_w"),
        planner="gradient",
    )
    if kind == "nonlinear":
        decl = NonlinearGaussianTransition(
            RateUnicycle(PLAN_H, PLAN_V), _noise(jnp.zeros((1, 3)), PLAN_Q)
        )
    else:
        decl = _frozen_rate_turn()
    system, params = _steered(decl, **over)
    box = {"u_w": Box(-PLAN_W * jnp.ones(1), PLAN_W * jnp.ones(1))}
    params = dataclasses.replace(params, control=box)
    return Case(f"turn_{kind}", _planned(system, params), None, 100)


def _frozen_rate_turn():
    from hybrid_aif import LinearGaussianTransition

    return LinearGaussianTransition(
        F=FROZEN_F,
        innovations=_noise(jnp.asarray([[PLAN_H * PLAN_V, 0.0, 0.0]]), PLAN_Q),
        G=jnp.asarray([[0.0], [0.0], [PLAN_H]]),
    )


def _point_mass_flat():
    from hybrid_aif.factors import GridNoiseMap

    flat = GridNoiseMap(
        jnp.asarray(np.stack([_chol(0.6)] * 81).reshape(9, 9, 2, 2)), (-1, -1), (1, 1), (0, 1)
    )
    system, params = _point_mass(flat, policy_len=8)
    return Case("noise_map_point_mass", _planned(system, params), None, 300)


def _e5(decomposition):
    ctx, init, *_ = e5_model(decomposition)
    return Case(f"e5_{decomposition}", ctx, init, 100)


@functools.cache
def _detour_module():
    spec = importlib.util.spec_from_file_location(
        "sensor_detour", ROOT / "examples" / "sensor_detour.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _detour(which):
    module = _detour_module()
    system, params = module.agents()[which]
    return Case(f"detour_{which}", module._context(system, params), None, 400)


def _downstream():
    root = pathlib.Path(DOWNSTREAM)
    if not DOWNSTREAM or not (root / "e5model.py").exists():
        pytest.skip("HYBRID_AIF_FLOOR_DOWNSTREAM does not name the downstream step-0 model")
    spec = importlib.util.spec_from_file_location("e5model", root / "e5model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ns = types.SimpleNamespace(
        cap014=str(root / "cap014.pkl"),
        cap015=str(root / "cap015.pkl"),
        noise="map",
        pile="map",
        dims="sub",
        gap="yes",
        belief="posterior",
        fix="no",
    )
    ctx, init, *_ = module.build(ns)
    return Case("downstream_step0", ctx, init, 100)


#: Fixtures whose own scans carry float32 beliefs and fail to type-check under ``x64``.
FLOAT32_ONLY = {"small_lg", "shared_noise_lg", "lane", "unicycle"}

BUILDERS = {
    "lq_control": _lq,
    "small_lg": _small_lg,
    "shared_noise_lg": _shared,
    "lane": _lane,
    "unicycle": _unicycle,
    "turn_nonlinear": lambda: _rate_turn("nonlinear"),
    "turn_frozen": lambda: _rate_turn("frozen"),
    "noise_map_point_mass": _point_mass_flat,
    "e5_utility_infogain": lambda: _e5("utility_infogain"),
    "e5_risk_ambiguity": lambda: _e5("risk_ambiguity"),
    "detour_prior": lambda: _detour("prior"),
    "detour_posterior": lambda: _detour("posterior"),
    "downstream_step0": _downstream,
}


@functools.cache
def case(name: str) -> Case:
    if jax.config.jax_enable_x64 and name in FLOAT32_ONLY:
        pytest.skip("the fixture's float32 parameters do not scan under x64")
    return BUILDERS[name]()


def warm_start(c: Case, seed: int) -> jax.Array:
    """A seeded warm start inside the middle of the box."""
    low, high = (np.asarray(x) for x in c.ctx.continuous_controls.bounds)
    u = np.random.default_rng(seed).uniform(0.2, 0.8, (c.ctx.policy_len, low.shape[0]))
    return jnp.asarray(low + (high - low) * u, low.dtype)


# --------------------------------------------------------------- unchanged where it does not apply

#: Models whose coordinates all carry gradients above the floor throughout the descent.
UNBOUND = [
    "lq_control",
    "small_lg",
    "shared_noise_lg",
    "lane",
    "turn_nonlinear",
    "turn_frozen",
    "e5_utility_infogain",
    "e5_risk_ambiguity",
    "detour_prior",
]
#: Models in which some coordinate with a nonzero first moment falls below the floor at some step.
BOUND = ["unicycle", "noise_map_point_mass", "detour_posterior"]


@pytest.mark.parametrize("name", UNBOUND)
def test_the_descent_is_the_previous_rule_bitwise_where_the_floor_never_binds(name):
    c = case(name)
    old = reference_descent(c.ctx, "old", c.steps, c.init)
    new = reference_descent(c.ctx, "new", c.steps, c.init)
    assert not new.binding.any()
    np.testing.assert_array_equal(new.plans, old.plans)
    np.testing.assert_array_equal(new.trace, old.trace)
    got = gradient_plan(c.ctx, steps=c.steps, init=c.init)
    np.testing.assert_array_equal(np.asarray(got.trace), old.trace)
    # the returned plan is the last iterate when it is the best, else an earlier one of the same
    # descent; both are among the previous rule's iterates
    assert any(np.array_equal(np.asarray(got.plan), p) for p in [*old.plans, old.last])
    assert float(got.G.midpoint[0]) <= g_of(c.ctx, old.last) + tolerance(float(got.trace[-1]))
    if old.trace.min() >= g_of(c.ctx, old.last):
        np.testing.assert_array_equal(np.asarray(got.plan), old.last)


@pytest.mark.parametrize("name", BOUND)
def test_the_descent_is_the_previous_rule_bitwise_until_the_floor_first_binds(name):
    c = case(name)
    old = reference_descent(c.ctx, "old", c.steps, c.init)
    new = reference_descent(c.ctx, "new", c.steps, c.init)
    assert new.binding.any(), "listed as bound: move it to UNBOUND if the floor never binds"
    first = int(np.argmax(new.binding))
    np.testing.assert_array_equal(new.plans[: first + 1], old.plans[: first + 1])
    np.testing.assert_array_equal(new.trace[: first + 1], old.trace[: first + 1])


@pytest.mark.parametrize("name", [*UNBOUND, *BOUND])
def test_the_library_descent_is_the_reference_new_rule(name):
    """``gradient_plan``'s trace is the reference's, and its plan the best iterate of it."""
    c = case(name)
    new = reference_descent(c.ctx, "new", c.steps, c.init)
    got = gradient_plan(c.ctx, steps=c.steps, init=c.init)
    np.testing.assert_allclose(np.asarray(got.trace), new.trace, rtol=1e-5, atol=1e-5)
    scored = [*new.trace, g_of(c.ctx, new.last)]
    assert float(got.G.midpoint[0]) == pytest.approx(min(scored), abs=1e-4)


# ------------------------------------------------------------------- never above the warm start


@pytest.mark.parametrize("name", [n for n in BUILDERS if n != "downstream_step0"])
@pytest.mark.parametrize("seed", [None, 1, 2])
def test_the_returned_g_is_never_above_the_warm_starts(name, seed):
    c = case(name)
    init = c.init if seed is None else warm_start(c, seed)
    got = gradient_plan(c.ctx, steps=c.steps, init=init)
    plan0 = jnp.broadcast_to(
        0.5 * sum(c.ctx.continuous_controls.bounds) if init is None else init,
        (c.ctx.policy_len, c.ctx.continuous_controls.action_dim),
    )
    low, high = c.ctx.continuous_controls.bounds
    warm = g_of(c.ctx, jnp.clip(plan0, low, high))
    assert float(got.trace[0]) == pytest.approx(warm, abs=1e-5)
    assert float(got.G.midpoint[0]) <= warm + tolerance(warm)
    assert float(got.G.midpoint[0]) == pytest.approx(g_of(c.ctx, got.plan), abs=tolerance(warm))


def test_the_returned_g_is_never_above_the_warm_start_of_the_downstream_model():
    c = case("downstream_step0")
    got = gradient_plan(c.ctx, steps=c.steps, init=c.init)
    warm = g_of(c.ctx, c.init)
    assert float(got.G.midpoint[0]) <= warm + tolerance(warm)


# ------------------------------------------------------------------------- no quality regression

BATTERY = [(n, s) for n in BUILDERS if n != "downstream_step0" for s in (11, 12, 13)]


@pytest.mark.parametrize(("name", "seed"), BATTERY)
def test_the_returned_g_is_at_most_the_previous_rules_case_by_case(name, seed):
    c = case(name)
    init = warm_start(c, seed)
    old = reference_descent(c.ctx, "old", c.steps, init)
    got = gradient_plan(c.ctx, steps=c.steps, init=init)
    want = g_of(c.ctx, old.last)
    assert float(got.G.midpoint[0]) <= want + tolerance(want)


def test_the_default_start_of_the_detour_agent_ends_within_a_known_margin_of_the_previous_rule():
    """From the box midpoint over 400 steps the unconverged descent of the detour agent ends at
    -3.187862 under the previous rule and -3.187774 under the floor (measured): above it by
    8.8e-5, inside the 4.8e-4 spread of that descent's end points."""
    c = case("detour_posterior")
    old = reference_descent(c.ctx, "old", 400)
    got = gradient_plan(c.ctx, steps=400)
    want = g_of(c.ctx, old.last)
    assert float(got.G.midpoint[0]) <= want + 2e-4


# -------------------------------------------------------------------------- the flat direction


def _two_coordinate_context(slope: float, horizon=1) -> PlannerContext:
    """``G(u) = (u₀ − 1)² + slope · u₁``: a plan whose second coordinate has a constant gradient
    ``slope`` while the first has one of order one."""

    def transition(qs, action):
        return action

    def step_terms(qs, action, t):
        g = (action[0] - 1.0) ** 2 + slope * action[1]
        return {"risk": {"q": Interval(g, g)}}

    return PlannerContext(
        transition_fn=transition,
        step_terms_fn=step_terms,
        terminal_terms_fn=None,
        controls=None,
        levels=(1,),
        policy_len=horizon,
        gamma=1.0,
        policy_prior=None,
        init_beliefs=jnp.zeros(2, jnp.float32),
        continuous_controls=ControlLayout(
            node_spans=((0, 2),),
            factor_nodes=((0,),),
            action_dim=2,
            low=-5.0 * jnp.ones(2, jnp.float32),
            high=5.0 * jnp.ones(2, jnp.float32),
        ),
    )


def test_a_coordinate_with_a_gradient_far_below_the_others_takes_no_appreciable_step():
    ctx = _two_coordinate_context(1e-7)
    init = jnp.asarray([[0.0, 0.0]])
    old = reference_descent(ctx, "old", 1, init)
    new = reference_descent(ctx, "new", 1, init)
    box_step = DEFAULT_LR_FRACTION * 10.0
    # the previous rule moves the flat coordinate by nearly the full step (its guard of 1e-8
    # is a tenth of this gradient); the floor moves it in proportion to its gradient
    assert abs(float(old.last[0, 1])) == pytest.approx(0.4545, rel=1e-3)
    assert abs(float(new.last[0, 1])) < 1e-3 * box_step
    assert float(new.last[0, 0]) == pytest.approx(float(old.last[0, 0]), rel=1e-3)
    got = gradient_plan(ctx, steps=1, init=init)
    assert abs(float(got.plan[0, 1])) < 1e-3 * box_step


def test_a_flat_coordinate_stays_put_while_the_others_converge():
    """Over twenty steps the previous rule runs the flat coordinate to the box's far side while
    the other converges; the floored descent leaves it where it was."""
    ctx = _two_coordinate_context(1e-7)
    init = jnp.zeros((1, 2))
    old = reference_descent(ctx, "old", 20, init)
    new = reference_descent(ctx, "new", 20, init)
    assert abs(float(old.last[0, 1])) > 4.0
    assert abs(float(new.last[0, 1])) < 1e-2
    np.testing.assert_array_equal(new.last[0, 0], old.last[0, 0])


def test_a_gradient_that_is_exactly_zero_gives_a_zero_step_and_a_finite_plan():
    ctx = _two_coordinate_context(0.0)
    got = gradient_plan(ctx, steps=5, init=jnp.asarray([[1.0, 0.7]]))
    np.testing.assert_array_equal(np.asarray(got.plan), np.asarray([[1.0, 0.7]], np.float32))
    assert np.all(np.isfinite(np.asarray(got.trace)))


def test_the_downstream_turn_rate_barely_moves_at_the_straight_plan_and_the_start_is_kept():
    """At the straight plan the turn-rate gradients are about 1e-9 against 3e-3 for the speed:
    the first step moves the turn rate by under 1e-3 rad/s where the previous rule moved it by
    0.03 to 0.045, and the hundred-step descent, which wanders above the warm start under either
    rule, returns the warm start."""
    c = case("downstream_step0")
    old = reference_descent(c.ctx, "old", 1, c.init)
    new = reference_descent(c.ctx, "new", 1, c.init)
    moved_old = np.abs(old.last - np.asarray(c.init))[:, 1]
    moved_new = np.abs(new.last - np.asarray(c.init))[:, 1]
    assert moved_new.max() < 1e-3
    assert moved_old.min() > 0.03
    got = gradient_plan(c.ctx, steps=c.steps, init=c.init)
    warm = g_of(c.ctx, c.init)
    assert warm == pytest.approx(170.9937, abs=1e-3)
    end_old = g_of(c.ctx, reference_descent(c.ctx, "old", c.steps, c.init).last)
    end_new = g_of(c.ctx, reference_descent(c.ctx, "new", c.steps, c.init).last)
    assert end_old == pytest.approx(171.0848, abs=2e-3)
    assert end_new == pytest.approx(171.0690, abs=2e-3)
    assert float(got.G.midpoint[0]) <= warm + tolerance(warm)
    np.testing.assert_array_equal(np.asarray(got.plan), np.asarray(c.init))


# -------------------------------------------------------------------------------- reproducibility

#: The detour agent's descent from a start perturbed by 1e-6: the spread of ``G`` over twelve
#: starts.
SPREAD_STEPS = 10_000


@functools.cache
def _spread(rule):
    module = _detour_module()
    c = case("detour_posterior")
    T = module.POLICY_LEN
    values = []
    for seed in range(1, 13):
        init = (1e-6 * np.random.default_rng(seed).standard_normal((T, 2))).astype(np.float32)
        if rule == "old":
            plan = reference_descent(c.ctx, "old", SPREAD_STEPS, init).last
        else:
            plan = np.asarray(gradient_plan(c.ctx, steps=SPREAD_STEPS, init=init).plan)
        values.append(module.expected_free_energy(*module.agents()["posterior"], plan))
    return np.asarray(values)


@pytest.mark.slow
@pytest.mark.skipif(jax.config.jax_enable_x64, reason="measured in float32")
def test_the_perturbed_start_spread_of_the_detour_agent_is_not_larger():
    """Twelve starts perturbed by 1e-6, 10 000 steps each: ``G`` of the end points spreads by
    about half as much under the floor (4.8e-4, measured, against 9.6e-4; under x64, where the
    plan stays float32, 4.5e-4 against 4.3e-4)."""
    before, after = _spread("old"), _spread("new")
    spread = lambda v: float(v.max() - v.min())  # noqa: E731
    assert spread(after) <= spread(before)
    assert spread(after) < 1e-3


# --------------------------------------------------------------------- best-iterate semantics


def _quadratic_context() -> PlannerContext:
    """``G(u) = Σ u²`` over a one-step plan of two coordinates in the box ``[-5, 5]``."""

    def step_terms(qs, action, t):
        g = jnp.sum(action**2)
        return {"risk": {"q": Interval(g, g)}}

    ctx = _two_coordinate_context(0.0)
    return ctx._replace(step_terms_fn=step_terms)


def test_the_planner_returns_the_earlier_plan_when_the_last_iterate_is_worse():
    """A plain gradient step of size 1.2 on ``Σ u²`` multiplies each coordinate by −1.4: the
    iterates overshoot and grow, so the warm start is the best plan scored."""
    ctx = _quadratic_context()
    init = jnp.asarray([[1.0, 0.5]])
    got = gradient_plan(ctx, steps=3, optimizer="gradient", lr=1.2, init=init)
    scored = np.asarray(got.trace)
    assert scored[0] < scored[1] < scored[2]  # every step lands above the last
    np.testing.assert_array_equal(np.asarray(got.plan), np.asarray(init))
    assert float(got.G.midpoint[0]) == pytest.approx(1.25)
    assert float(total(got.tree).midpoint[0]) == pytest.approx(1.25)
    # the trace is still G before each step, from the warm start on
    np.testing.assert_allclose(scored, 1.25 * np.asarray([1.0, 1.96, 1.96**2]), rtol=1e-5)


def test_a_middle_iterate_is_returned_when_the_descent_overshoots_after_reaching_it():
    ctx = _quadratic_context()
    init = jnp.asarray([[2.0, 0.0]])
    # a step of 0.9 multiplies each coordinate by −0.8: a shrinking oscillation
    got = gradient_plan(ctx, steps=4, optimizer="gradient", lr=0.9, init=init)
    scored = np.asarray(got.trace)
    best = min([*scored, g_of(ctx, got.plan)])
    assert float(got.G.midpoint[0]) == pytest.approx(best, abs=1e-6)
    assert float(got.G.midpoint[0]) == pytest.approx(g_of(ctx, got.plan), abs=1e-6)


def test_the_laplace_posterior_is_taken_at_the_returned_plan():
    system, params = lq.make_lq()
    ctx = _planned(system, params)
    got = gradient_plan(ctx, steps=50, posterior="laplace")
    np.testing.assert_array_equal(np.asarray(got.curvature.loc), np.asarray(got.plan).reshape(-1))


def test_steps_zero_returns_the_warm_start_and_its_score():
    system, params = lq.make_lq()
    ctx = _planned(system, params)
    init = jnp.full((ctx.policy_len, 1), 0.3, jnp.float32)
    got = gradient_plan(ctx, steps=0, init=init)
    np.testing.assert_array_equal(np.asarray(got.plan), np.asarray(init))
    assert got.trace.shape == (0,)
    assert float(got.G.midpoint[0]) == pytest.approx(g_of(ctx, init), abs=1e-6)


def test_the_metrics_free_planner_returns_the_same_plan_and_g():
    c = case("small_lg")
    with_tree = gradient_plan(c.ctx, steps=60)
    without = gradient_plan(c.ctx, steps=60, metrics=False)
    np.testing.assert_array_equal(np.asarray(with_tree.plan), np.asarray(without.plan))
    assert without.tree is None
    assert float(without.G.midpoint[0]) == pytest.approx(float(with_tree.G.midpoint[0]), abs=1e-5)


def test_the_floor_is_the_precision_of_the_plans_dtype():
    """The floor is ``sqrt(eps)`` of the plan's dtype times the gradient's RMS. The control box
    fixes the plan at float32 (also with ``x64`` enabled), so the floor is 3.4e-4 of the RMS:
    a coordinate whose gradient is 7e-7 of it moves by about a thousandth of its full step.
    The step the library takes equals the one a hand computation of that floor gives."""
    slope = 1e-6
    ctx = _two_coordinate_context(slope)
    got = gradient_plan(ctx, steps=1, init=jnp.zeros((1, 2), jnp.float32))
    assert got.plan.dtype == jnp.float32
    rms = np.sqrt((4.0 + slope**2) / 2.0)
    floor = float(np.sqrt(np.finfo(np.float32).eps)) * rms
    assert floor == pytest.approx(3.45e-4 * rms, rel=1e-2)
    step = DEFAULT_LR_FRACTION * 10.0 * slope / (floor + 1e-8)
    assert float(got.plan[0, 1]) == pytest.approx(-step, rel=1e-3)
    unfloored = reference_descent(ctx, "old", 1, jnp.zeros((1, 2), jnp.float32))
    assert abs(float(unfloored.last[0, 1])) == pytest.approx(0.495, rel=1e-2)


def _with_terms(step_terms):
    return _two_coordinate_context(0.0)._replace(step_terms_fn=step_terms)


def test_a_nan_gradient_surfaces_as_a_nan_plan_and_g():
    def terms(qs, action, t):
        g = (action[0] - 1.0) ** 2 + jnp.sqrt(jnp.abs(action[1]))
        return {"risk": {"q": Interval(g, g)}}

    got = gradient_plan(_with_terms(terms), steps=5, init=jnp.asarray([[0.0, 0.0]]))
    assert np.isnan(np.asarray(got.plan)).any()
    assert np.isnan(float(got.G.midpoint[0]))


def test_a_g_that_turns_nan_mid_descent_returns_the_nan_final_plan():
    def terms(qs, action, t):
        g = jnp.where(action[0] > 1.5, jnp.nan, (action[0] - 3.0) ** 2)
        return {"risk": {"q": Interval(g, g)}}

    got = gradient_plan(_with_terms(terms), steps=8, init=jnp.asarray([[0.0, 0.0]]))
    assert np.isnan(float(got.G.midpoint[0]))
    assert float(got.plan[0, 0]) > 1.5


def test_an_overflowing_gradient_freezes_only_its_own_coordinate():
    """A gradient of 1e20 squares to infinity in float32; the other coordinate still descends."""

    def terms(qs, action, t):
        g = (action[0] - 1.0) ** 2 + 1e20 * action[1]
        return {"risk": {"q": Interval(g, g)}}

    got = gradient_plan(_with_terms(terms), steps=3, init=jnp.asarray([[0.0, 0.0]]))
    assert float(got.plan[0, 0]) > 0.3
    assert np.all(np.isfinite(np.asarray(got.plan)))
    assert float(got.plan[0, 1]) == 0.0


# ------------------------------------------------------------- the gradient trace and best_step


def _rms_grad(ctx, plan) -> float:
    grad = jax.grad(_score_fn(ctx))(jnp.asarray(plan))
    return float(jnp.sqrt(jnp.mean(grad**2)))


def _weighted_quadratic_context() -> PlannerContext:
    """``G(u) = u₀² + 100 u₁²`` over a one-step plan in the box ``[-5, 5]²``."""

    def step_terms(qs, action, t):
        g = action[0] ** 2 + 100.0 * action[1] ** 2
        return {"risk": {"q": Interval(g, g)}}

    return _two_coordinate_context(0.0)._replace(step_terms_fn=step_terms)


def test_the_gradient_trace_is_the_rms_gradient_before_each_step():
    c = case("small_lg")
    steps = 7
    got = gradient_plan(c.ctx, steps=steps, init=warm_start(c, 1))
    assert got.gradient_trace.shape == (steps,)
    assert np.all(np.isfinite(np.asarray(got.gradient_trace)))
    np.testing.assert_allclose(
        float(got.gradient_trace[0]), _rms_grad(c.ctx, warm_start(c, 1)), rtol=1e-5
    )
    # the last entry is the gradient at the final-minus-one iterate, replayed by the reference
    ref = reference_descent(c.ctx, "new", steps, warm_start(c, 1))
    np.testing.assert_allclose(
        float(got.gradient_trace[-1]), _rms_grad(c.ctx, ref.plans[-1]), rtol=1e-4
    )


def test_best_step_is_zero_when_the_warm_start_is_returned():
    ctx = _quadratic_context()
    init = jnp.asarray([[1.0, 0.5]])
    got = gradient_plan(ctx, steps=3, optimizer="gradient", lr=1.2, init=init)
    assert got.best_step.dtype == jnp.int32
    assert int(got.best_step) == 0


def test_best_step_is_steps_when_the_final_plan_is_best():
    c = case("lq_control")
    got = gradient_plan(c.ctx, steps=10)
    assert int(got.best_step) == 10


def test_best_step_names_an_interior_iterate_when_it_is_best():
    ctx = _weighted_quadratic_context()
    init = jnp.asarray([[3.0, 1e-3]])
    got = gradient_plan(ctx, steps=6, optimizer="gradient", lr=0.05, init=init)
    scored = np.asarray(got.trace)
    best = int(got.best_step)
    assert 0 < best < 6
    assert best == int(np.argmin(scored))
    assert float(got.G.midpoint[0]) == pytest.approx(scored[best], abs=1e-5)


def test_best_step_is_steps_for_a_descent_that_ends_non_finite():
    def terms(qs, action, t):
        g = jnp.where(action[0] > 1.5, jnp.nan, (action[0] - 3.0) ** 2)
        return {"risk": {"q": Interval(g, g)}}

    got = gradient_plan(_with_terms(terms), steps=8, init=jnp.asarray([[0.0, 0.0]]))
    assert int(got.best_step) == 8


def test_steps_zero_has_an_empty_gradient_trace_and_best_step_zero():
    system, params = lq.make_lq()
    ctx = _planned(system, params)
    got = gradient_plan(ctx, steps=0)
    assert got.gradient_trace.shape == (0,)
    assert int(got.best_step) == 0


def test_the_plain_gradient_optimizer_reports_the_same_quantities():
    ctx = _quadratic_context()
    init = jnp.asarray([[1.0, 0.5]])
    got = gradient_plan(ctx, steps=3, optimizer="gradient", lr=0.1, init=init)
    assert got.gradient_trace.shape == (3,)
    np.testing.assert_allclose(float(got.gradient_trace[0]), _rms_grad(ctx, init), rtol=1e-6)
    np.testing.assert_allclose(
        float(got.gradient_trace[1]), _rms_grad(ctx, 0.8 * np.asarray(init)), rtol=1e-5
    )


def test_the_planner_face_returns_the_same_tuple():
    from hybrid_aif.planning.gradient import gradient_planner

    c = case("lq_control")
    out = gradient_planner(c.ctx)
    assert len(out) == 3
    got = gradient_plan(c.ctx)
    np.testing.assert_array_equal(np.asarray(out[2]), np.asarray(got.plan[None]))
    assert out[0] is not None and out[1] is not None
