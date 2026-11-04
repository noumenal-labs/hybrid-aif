"""The ``approximate`` status of an Interval and of the leaves a System reports.

``Interval.approximate`` ORs through every combinator (beside ``derived``'s AND). The registry
stamps it per leaf from a static map of the System: a factor is approximate when its transition
is neural or nonlinear-Gaussian or any factor it reads is, and a modality when its likelihood is
neural or it observes an approximate factor.

Design: docs/design/recognition-beside-analytic.md L7.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from hybrid_aif import (
    Gaussian,
    LinearGaussianObservation,
    LinearGaussianTransition,
    MLPConditional,
    NonlinearGaussianTransition,
    System,
    SystemParams,
)
from hybrid_aif.interval import Interval
from hybrid_aif.objectives import registry
from hybrid_aif.objectives.base import merge, total
from hybrid_aif.planning import gradient_plan
from hybrid_aif.planning.enumerate import evaluate_policies
from tests.test_action_node_modalities import MUTE_SD, payoff_model_by_node
from tests.test_neural_likelihood_planning import (
    D,
    M,
    _gaussian_models,
    _mixture_params,
    _mixture_system,
    _unit_params,
    _unit_system,
    curved,
    lg_equivalent,
)
from tests.test_nonlinear_transition import (
    BEHIND,
    LANE,
    PLAN_H,
    PLAN_Q,
    PLAN_V,
    RATES,
    SteeredUnicycle,
    _noise,
)
from tests.test_presence_by_level import PAYOFF_PLANS
from tests.test_presence_by_level import _tree as _presence_tree

KEY = jax.random.PRNGKey(0)


def _iv(lo=0.0, hi=1.0, derived=True, approximate=False) -> Interval:
    return Interval(jnp.asarray(lo), jnp.asarray(hi), derived, approximate)


# ------------------------------------------------------------------- (a) the algebra


def test_approximate_is_ored_and_derived_anded_through_arithmetic():
    a, b = _iv(approximate=True), _iv(derived=False)
    s = a + b
    assert s.approximate and not s.derived
    assert (b + b).approximate is False and (a + a).derived
    assert (a + 1.0).approximate and (b + 1.0).approximate is False
    assert (-a).approximate and (a - b).approximate and (b - a).approximate
    assert (a * 2.0).approximate and (a * -2.0).approximate
    assert (2.0 * a).approximate and (b * 2.0).approximate is False
    assert a.clip_lower(0.0).approximate
    assert Interval.exact(1.0, approximate=True).approximate
    assert Interval.exact(1.0).approximate is False


def test_approximate_survives_merge_and_total():
    tree = {"utility": {"y": _iv(approximate=True)}, "risk": {"x": _iv(), "z": _iv()}}
    merged = merge(tree, {"ambiguity": {"w": _iv(derived=False)}})
    assert merged["utility"]["y"].approximate and not merged["risk"]["x"].approximate
    assert total(tree).approximate
    assert not total({"risk": {"x": _iv(), "z": _iv()}}).approximate
    both = total({**tree, "ambiguity": {"w": _iv(derived=False)}})
    assert both.approximate and not both.derived


def test_approximate_agrees_across_a_scan_carry():
    def body(carry, _):
        return carry + _iv(approximate=True), None

    init = _iv(0.0, 0.0, approximate=True)
    out, _ = jax.lax.scan(body, init, jnp.arange(3))
    assert out.approximate


# ------------------------------------------------------------------- (b) the static map


def _nl():
    return NonlinearGaussianTransition(
        SteeredUnicycle(PLAN_H, PLAN_V, RATES), _noise(jnp.zeros((3, 3)), PLAN_Q)
    )


def _lg(d=1, levels=1):
    return LinearGaussianTransition(
        F=jnp.broadcast_to(jnp.eye(d), (levels, d, d)),
        innovations=_noise(jnp.zeros((levels, d)), 0.1 * jnp.eye(d)),
    )


def _lgo():
    return LinearGaussianObservation(H=jnp.eye(1), c=jnp.zeros(1), noise=jnp.eye(1))


def test_a_nonlinear_factor_marks_its_readers_and_their_modalities():
    amap = registry.approximate_status(
        B=(_nl(), _lg(), _lg()),
        A=(_lgo(), _lgo()),
        B_deps=((), (0,), ()),
        A_deps=((1,), (2,)),
        factor_names=("x", "y", "w"),
        modality_names=("sees_y", "sees_w"),
    )
    assert amap.factors == {"x": True, "y": True, "w": False}
    assert amap.modalities == {"sees_y": True, "sees_w": False}
    assert amap.leaf("sees_y") and amap.leaf("likelihood:sees_y") and amap.leaf("transition:y")
    assert not amap.leaf("sees_w") and not amap.leaf("unrelated") and amap.any


def test_the_status_follows_a_chain_in_any_declaration_order():
    amap = registry.approximate_status(
        B=(_lg(), _lg(), _nl()),
        A=(),
        B_deps=((1,), (2,), ()),
        A_deps=(),
        factor_names=("a", "b", "c"),
        modality_names=(),
    )
    assert amap.factors == {"a": True, "b": True, "c": True}


def test_a_neural_modality_over_a_linear_factor_is_approximate_alone():
    neural = MLPConditional.from_shapes((1,), 1, key=KEY, width=4, depth=1)
    amap = registry.approximate_status(
        B=(_lg(),),
        A=(neural, _lgo()),
        B_deps=((),),
        A_deps=((0,), (0,)),
        factor_names=("x",),
        modality_names=("image", "gps"),
    )
    assert amap.factors == {"x": False}
    assert amap.modalities == {"image": True, "gps": False}


# ------------------------------------------------------- (c), (d) the end-to-end leaves


def _two_factor(pose_decl):
    """A pose (``pose_decl``) and an unrelated scalar ``z``, each observed, targeted and read."""
    system = System(
        B_deps=((), ()),
        controlled=(True, False),
        factor_names=("pose", "z"),
        A_deps=((0,), (1,)),
        modality_names=("gps", "alt"),
        policy_len=2,
        epistemic=True,
    )
    params = SystemParams(
        likelihood={
            "gps": LinearGaussianObservation(
                H=jnp.eye(3)[:2], c=jnp.zeros(2), noise=0.1 * jnp.eye(2)
            ),
            "alt": _lgo(),
        },
        transition={"pose": pose_decl, "z": _lg()},
        initial_state_prior={
            "pose": Gaussian(loc=jnp.zeros(3), chol=0.05 * jnp.eye(3)),
            "z": Gaussian(loc=jnp.zeros(1), chol=jnp.eye(1)),
        },
        state_targets={"pose": LANE, "z": Gaussian(loc=jnp.ones(1), chol=jnp.eye(1))},
        terminal_targets={"pose": BEHIND},
        obs_targets={"alt": Gaussian(loc=jnp.ones(1), chol=jnp.eye(1))},
    )
    return system, params


def _tree(pose_decl):
    system, params = _two_factor(pose_decl)
    beliefs = system.initial_priors(params)
    return system.score_policies(params, beliefs, system.policies(params))


def _flags(tree) -> dict:
    return {(t, k): iv.approximate for t, blocks in tree.items() for k, iv in blocks.items()}


def test_a_nonlinear_transition_marks_the_leaves_about_it_and_the_total():
    flags = _flags(_tree(_nl()))
    assert flags[("infogain", "gps")] and not flags[("infogain", "alt")]
    assert not flags[("utility", "alt")] and not flags[("risk", "z")]
    assert flags[("risk", "pose")] and flags[("terminal_risk", "pose")]
    assert sum(flags.values()) == 3  # exactly the leaves about the pose


def test_the_total_of_a_nonlinear_model_is_approximate():
    assert total(_tree(_nl())).approximate


def test_a_linear_gaussian_model_has_no_approximate_leaf():
    linear = _lg(3, levels=3)
    tree = _tree(linear)
    flags = [iv.approximate for blocks in tree.values() for iv in blocks.values()]
    assert flags and not any(flags)
    assert not total(tree).approximate


# ------------------------------------------------ (c) a neural modality's leaves, total and G


def _flags(tree) -> dict:
    return {(t, k): iv.approximate for t, blocks in tree.items() for k, iv in blocks.items()}


@pytest.mark.parametrize("decomposition", ["utility_infogain", "risk_ambiguity"])
def test_a_neural_modality_over_a_gaussian_factor_marks_its_leaves_and_the_total(decomposition):
    system, neural, lg = _gaussian_models(curved(), decomposition=decomposition)
    _, tree = system.infer_policies(neural, system.initial_priors(neural))
    flags = _flags(tree)
    modality_leaves = {k: v for k, v in flags.items() if k[1] == "position"}
    assert modality_leaves and all(modality_leaves.values())
    assert not flags[("risk", "state")]  # a leaf about the state's own target stays exact
    assert total(tree).approximate
    _, exact = system.infer_policies(lg, system.initial_priors(lg))
    assert not any(_flags(exact).values()) and not total(exact).approximate


@pytest.mark.parametrize("decomposition", ["utility_infogain", "risk_ambiguity"])
def test_a_neural_modality_over_a_mixture_marks_its_leaves_and_the_total(decomposition):
    system = _mixture_system(decomposition=decomposition)
    params = _mixture_params(curved(parent_dims=(D, M)))
    _, tree = system.infer_policies(params, system.initial_priors(params))
    flags = _flags(tree)
    assert flags and all(flags.values())
    assert total(tree).approximate


def test_the_gradient_planners_G_and_a_continuous_plan_are_approximate_under_a_neural_modality():
    system = _unit_system()
    params = _unit_params(curved())
    ctx = system.planner_context(params, system.initial_priors(params))
    plan = jnp.asarray([[0.3], [-0.2], [0.1], [0.4]])
    tree = evaluate_policies(
        plan[None], ctx.init_beliefs, ctx.transition_fn, ctx.step_terms_fn, ctx.terminal_terms_fn
    )
    assert all(_flags(tree).values()) and total(tree).approximate
    assert gradient_plan(ctx, steps=3, init=plan).G.approximate
    exact = _unit_params(lg_equivalent(curved()))
    ctx = system.planner_context(exact, system.initial_priors(exact))
    assert not gradient_plan(ctx, steps=3, init=plan).G.approximate


def test_a_leaf_keyed_by_a_factor_is_approximate_when_a_channel_it_reads_is():
    # the observation_interaction leaf of a factor read by a neural and a linear modality
    system, params = payoff_model_by_node(20.0, 3)
    decl = MLPConditional.from_shapes((2, 3), 1, key=KEY, depth=0, frozen=True)
    sds = jnp.asarray([MUTE_SD, 0.2, MUTE_SD])
    raw = jnp.where(
        sds > 20,
        sds - decl.min_scale,
        jnp.log(jnp.expm1(jnp.minimum(sds, 20.0) - decl.min_scale)),
    )
    last = decl.layers[-1]
    weight = jnp.zeros((2, 5)).at[0, 1].set(1.0).at[1, 2:].set(raw)
    last = eqx.tree_at(lambda layer: (layer.weight, layer.bias), last, (weight, jnp.zeros(2)))
    decl = eqx.tree_at(lambda m: m.layers, decl, (last,))
    neural = dataclasses.replace(params, likelihood={**params.likelihood, "answer": decl})
    flags = _flags(_presence_tree(system, neural, PAYOFF_PLANS, None, controlled=True))
    assert flags[("infogain", "answer")] and not flags[("infogain", "gps")]
    assert flags[("observation_interaction", "lane")]
    exact = _flags(_presence_tree(system, params, PAYOFF_PLANS, None, controlled=True))
    assert not any(exact.values())


def test_a_neural_modality_over_categorical_parents_alone_is_not_approximate():
    from hybrid_aif import ConditionalCategorical

    table = ConditionalCategorical(log_table=jnp.zeros((3, 3)), parent_ndims=1)
    neural = MLPConditional.from_shapes((3,), 2, key=KEY, width=4, depth=1)
    amap = registry.approximate_status(
        B=(table, _lg()),
        A=(neural, MLPConditional.from_shapes((1,), 2, key=KEY, width=4, depth=1)),
        B_deps=((0,), ()),
        A_deps=((0,), (1,)),
        factor_names=("mode", "x"),
        modality_names=("over_mode", "over_x"),
    )
    assert amap.modalities == {"over_mode": False, "over_x": True}
    assert not amap.leaf("over_mode") and amap.leaf("over_x")
