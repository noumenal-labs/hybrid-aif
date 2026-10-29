"""``present=`` under the levels of a control node, and on the tabular path.

An entry of ``present=`` may be a mapping from one discrete control node's name to a bool per
level, read at each plan step at that step's level of the node. These tests pin it against the
per-step bool array of the same plan on the hybrid path (both belief modes, with and without the
closed-loop correction) and the tabular one, the exactly-zero leaves of a designated Gaussian
channel that is absent under some levels, the tabular path's honouring of ``present=``, the
enumerating and tree planners, and the refusals.

Design: docs/design/planning-with-observations.md D4 (the extension under it).
"""

import dataclasses
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import (
    ConditionalGaussian,
    Gaussian,
    LinearGaussianObservation,
    LinearGaussianTransition,
    System,
    SystemParams,
)
from hybrid_aif.controls import ControlStep
from hybrid_aif.objectives import total
from hybrid_aif.planning.enumerate import evaluate_policies
from hybrid_aif.replay import replay
from tests.test_action_node_modalities import maze_with_cue_by_node, payoff_model_by_node

LOOK, WAIT = 0, 1
CUE_BY_LEVEL = (False, True, False)
REWARD_BY_LEVEL = (True, True, False)
GPS_BY_LEVEL = (True, False, True)
ANSWER_BY_LEVEL = (False, True, False)


def _steps_for(levels: np.ndarray, controlled: bool):
    """The batch of plans ``levels`` (``(n, H)`` option levels) as the planner's action steps."""
    lv = jnp.asarray(levels, jnp.int32)
    if controlled:  # the payoff model's second node is continuous
        return ControlStep(entries=(lv, jnp.zeros((*lv.shape, 1))))
    return lv[..., None]


def _tree(system, params, levels, present, controlled=False):
    ctx = system.planner_context(params, system.initial_priors(params), present=present)
    return evaluate_policies(
        _steps_for(np.asarray(levels), controlled),
        ctx.init_beliefs,
        ctx.transition_fn,
        ctx.step_terms_fn,
        ctx.terminal_terms_fn,
    )


def _step_trees(system, params, plan, present, controlled=False):
    """Each step's term tree for one plan, the carry stepped by hand."""
    ctx = system.planner_context(params, system.initial_priors(params), present=present)
    steps = _steps_for(np.asarray([plan]), controlled)
    carry, trees = ctx.init_beliefs, []
    for t in range(len(plan)):
        action = jax.tree_util.tree_map(lambda x, t=t: x[0, t], steps)
        carry = ctx.transition_fn(carry, action)
        trees.append(ctx.step_terms_fn(carry, action, jnp.int32(t)))
    return trees


def _differ(a, b) -> bool:
    return not np.array_equal(np.asarray(total(a).lower), np.asarray(total(b).lower))


def _assert_trees_equal(batched, single, row):
    assert set(batched) == set(single)
    for term, leaves in batched.items():
        assert set(leaves) == set(single[term]), term
        for key, iv in leaves.items():
            for bound in ("lower", "upper"):
                got = np.asarray(getattr(iv, bound))[row]
                want = np.asarray(getattr(single[term][key], bound))[0]
                assert np.array_equal(got, want), (term, key, bound, got, want)


PAYOFF_PLANS = [(0, 0, 0), (1, 2, 2), (2, 1, 2), (2, 2, 1), (1, 1, 0), (0, 2, 1)]


def _payoff(belief, feedback):
    system, params = payoff_model_by_node(20.0, 3)
    return dataclasses.replace(system, planning_belief=belief, feedback=feedback), params


@pytest.mark.parametrize(
    ("belief", "feedback"), [("posterior", "linear_quadratic"), ("prior", None)]
)
def test_a_per_level_entry_scores_each_plan_as_the_per_step_array_of_that_plan_does(
    belief, feedback
):
    """Hybrid path: the whole term trees (feedback corrections included) are bitwise the same."""
    system, params = _payoff(belief, feedback)
    tables = {"gps": GPS_BY_LEVEL, "answer": ANSWER_BY_LEVEL}
    per_level = {m: {"u_option": t} for m, t in tables.items()}
    batched = _tree(system, params, PAYOFF_PLANS, per_level, controlled=True)
    for row, plan in enumerate(PAYOFF_PLANS):
        per_step = {m: jnp.asarray(np.asarray(t)[list(plan)]) for m, t in tables.items()}
        single = _tree(system, params, [plan], per_step, controlled=True)
        _assert_trees_equal(batched, single, row)
    all_present = _tree(system, params, PAYOFF_PLANS, None, controlled=True)
    assert _differ(batched, all_present)
    if feedback is not None:
        # the closed-loop correction is in the compared trees
        open_loop = dataclasses.replace(system, feedback=None)
        open_tree = _tree(open_loop, params, PAYOFF_PLANS, per_level, controlled=True)
        assert _differ(batched, open_tree)


def test_a_per_level_entry_scores_each_plan_as_the_per_step_array_does_on_the_tabular_path():
    system, params = maze_with_cue_by_node()
    plans = list(itertools.product(range(3), repeat=2))
    tables = {"cue": CUE_BY_LEVEL, "reward": REWARD_BY_LEVEL}
    batched = _tree(system, params, plans, {m: {"u_act": t} for m, t in tables.items()})
    for row, plan in enumerate(plans):
        per_step = {m: jnp.asarray(np.asarray(t)[list(plan)]) for m, t in tables.items()}
        _assert_trees_equal(batched, _tree(system, params, [plan], per_step), row)
    all_present = _tree(system, params, plans, None)
    assert _differ(batched, all_present)


# ------------------------------------------------------------------ the designated channel


def reading_model(noise_limit: bool):
    """A scalar state read by one Gaussian channel ``reading`` with an observation target, and a
    two-level control node ``u`` (LOOK, WAIT). With ``noise_limit`` the channel is declared over
    ``(x, a)`` with a wide noise under WAIT; otherwise it is the plain channel over ``x``."""
    system = System(
        A_deps=((0, 2),) if noise_limit else ((0,),),
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("a",),
        factor_names=("x", "u"),
        modality_names=("reading",),
        policy_len=3,
        planning_belief="posterior",
        epistemic=True,
    )
    noise = jnp.asarray([0.2, 100.0]).reshape(2, 1, 1) if noise_limit else 0.2 * jnp.eye(1)
    params = SystemParams(
        transition={
            "x": LinearGaussianTransition(
                F=jnp.eye(1),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((2, 1)), chol=jnp.full((2, 1, 1), 0.1), parent_ndims=1
                ),
            )
        },
        initial_state_prior={"x": Gaussian(loc=jnp.asarray([0.3]), chol=jnp.full((1, 1), 0.4))},
        likelihood={
            "reading": LinearGaussianObservation(H=jnp.eye(1), c=jnp.zeros(1), noise=noise)
        },
        obs_targets={"reading": Gaussian(loc=jnp.asarray([0.3]), chol=jnp.eye(1))},
    )
    return system, params


def test_a_designated_channel_absent_under_wait_scores_zero_there_not_a_wide_predictive_cost():
    """The noise-limit form pays the wide predictive's divergence from the target at every WAIT
    step; the per-level entry zeroes the channel's leaves exactly there.

    Measured (float32, utility_infogain, horizon 3, plans WAIT-WAIT-WAIT / LOOK-LOOK-LOOK /
    LOOK-WAIT-WAIT): noise-limit 15003.025 / 1.6125 / 10002.218; per-level entry -0.0 / 1.6125 /
    0.19482."""
    plans = [(WAIT,) * 3, (LOOK,) * 3, (LOOK, WAIT, WAIT)]
    wide, wide_params = reading_model(noise_limit=True)
    plain, plain_params = reading_model(noise_limit=False)
    spelled = _tree(wide, wide_params, plans, None)
    exact = _tree(plain, plain_params, plans, {"reading": {"u": (True, False)}})
    never, always, once = np.asarray(total(spelled).lower)
    assert never - once > 4000.0
    np.testing.assert_allclose([never, always, once], [15003.025, 1.6125, 10002.218], rtol=1e-4)
    p_never, p_always, p_once = np.asarray(total(exact).lower)
    assert p_never == 0.0
    assert p_never - p_once < 0.0
    np.testing.assert_allclose([p_always, p_once], [1.6125, 0.19482], rtol=1e-3)
    # LOOK-LOOK-LOOK is the same plan under both spellings (the wide noise is never read)
    np.testing.assert_allclose(p_always, always, rtol=1e-5)
    trees = _step_trees(plain, plain_params, (LOOK, WAIT, WAIT), {"reading": {"u": (True, False)}})
    assert float(trees[0]["infogain"]["reading"].lower) > 0.0
    for t in (1, 2):
        assert float(trees[t]["infogain"]["reading"].lower) == 0.0
        assert float(trees[t]["utility"]["reading"].lower) == 0.0


# ------------------------------------------------------------------------- the tabular path


def test_present_takes_effect_on_the_tabular_path():
    system, params = maze_with_cue_by_node()
    plans = list(itertools.product(range(3), repeat=2))
    base = _tree(system, params, plans, None)
    gone = _tree(system, params, plans, {"cue": False})
    assert np.all(np.asarray(gone["infogain"]["cue"].lower) == 0.0)
    assert np.any(np.asarray(base["infogain"]["cue"].lower) != 0.0)
    assert _differ(base, gone)
    plan = (LOOK, LOOK)
    full = _step_trees(system, params, plan, None)
    second = _step_trees(system, params, plan, {"cue": jnp.asarray([False, True])})
    assert float(second[0]["infogain"]["cue"].lower) == 0.0
    assert float(full[0]["infogain"]["cue"].lower) > 0.0
    assert np.array_equal(
        np.asarray(second[1]["infogain"]["cue"].lower), np.asarray(full[1]["infogain"]["cue"].lower)
    )


def test_an_absent_channel_whose_target_rules_out_an_outcome_scores_an_exact_zero():
    """The maze's reward target gives one outcome no mass, so the reward's utility leaf is not
    finite where the channel is read; where it is absent the leaf is an exact zero under both
    spellings, batched and stepped by hand, and the totals are finite."""
    system, params = maze_with_cue_by_node()
    plans = list(itertools.product(range(3), repeat=2))
    per_step = _tree(system, params, plans, {"reward": False})
    per_level = _tree(system, params, plans, {"reward": {"u_act": (False, False, False)}})
    for tree in (per_step, per_level):
        assert np.all(np.asarray(tree["utility"]["reward"].lower) == 0.0)
        assert np.all(np.isfinite(np.asarray(total(tree).lower)))
    for present in ({"reward": False}, {"reward": {"u_act": (False, False, False)}}):
        for step in _step_trees(system, params, (LOOK, LOOK), present):
            assert float(step["utility"]["reward"].lower) == 0.0


# ------------------------------------------------------------------------------ the planners


def test_a_tree_planner_runs_a_per_level_entry_under_jit_and_it_changes_the_plan_scores():
    system, params = maze_with_cue_by_node()
    tree_system = dataclasses.replace(system, planner="sophisticated")
    beliefs = tree_system.initial_priors(params)

    def run(present):
        return jax.jit(lambda p, b: tree_system.infer_policies(p, b, present=present)[0].logits)(
            params, beliefs
        )

    plain = run(None)
    gated = run({"cue": {"u_act": (False, True, True)}})
    assert np.all(np.isfinite(np.asarray(gated)))
    assert not np.allclose(np.asarray(plain), np.asarray(gated))


# -------------------------------------------------------------------------------- the refusals


def test_a_probability_vector_step_is_refused_with_a_sentence():
    system, params = reading_model(noise_limit=False)
    ctx = system.planner_context(
        params, system.initial_priors(params), present={"reading": {"u": (True, False)}}
    )
    soft = ControlStep(entries=(jnp.asarray([0.5, 0.5]),))
    with pytest.raises(ValueError, match=r"present\['reading'\] is read at each step's level of"):
        ctx.step_terms_fn(ctx.init_beliefs, soft, jnp.int32(0))
    with pytest.raises(ValueError, match="needs a clamped level"):
        ctx.transition_fn(ctx.init_beliefs, soft)
        ctx.step_terms_fn(ctx.init_beliefs, soft, jnp.int32(0))


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ({"nope": (True, False)}, r"names 'nope', which is not a control node"),
        ({"u": (True, False), "x": (True, False)}, "one control node per entry"),
        ({}, "one control node per entry"),
        ({"u": (True, False, True)}, r"node 'u' has 2 level\(s\)"),
        ({"u": (0, 1)}, "must be a bool per level"),
    ],
)
def test_a_malformed_per_level_entry_is_refused_naming_the_fault(entry, match):
    system, params = reading_model(noise_limit=False)
    with pytest.raises(ValueError, match=match):
        system.planner_context(params, system.initial_priors(params), present={"reading": entry})


def test_a_continuous_control_node_cannot_carry_a_per_level_entry():
    system, params = payoff_model_by_node(20.0, 3)
    with pytest.raises(ValueError, match=r"continuous control node 'u_steer'"):
        system.planner_context(
            params, system.initial_priors(params), present={"gps": {"u_steer": (True, False)}}
        )


def test_the_inference_and_replay_faces_refuse_the_planning_form_naming_the_face():
    system, params = reading_model(noise_limit=False)
    present = {"reading": {"u": (True, False)}}
    with pytest.raises(ValueError, match=r"infer_states: present\['reading'\] under the levels"):
        system.infer_states(
            params, {"reading": jnp.zeros(1)}, system.initial_priors(params), present=present
        )
    with pytest.raises(ValueError, match=r"replay: present\['reading'\] under the levels"):
        replay(
            system,
            params,
            {"reading": jnp.zeros((4, 1))},
            {"a": jnp.zeros(4, jnp.int32)},
            present=present,
        )
