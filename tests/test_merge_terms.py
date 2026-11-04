"""``System.terms`` selectors under ``merge``: a component's selector carries under its prefix,
on its variable segment; a component declaring scoring is still refused."""

import dataclasses

import jax
import numpy as np
import pytest

from hybrid_aif.composition.merge import merge, namespaced_term, subsystem
from tests.test_action_node_modalities import scalar_model
from tests.test_merge_gaussian_control import mission


def _pair(fast_terms=(), composite_terms=()):
    fast, fast_p = scalar_model(horizon=2)
    fast = dataclasses.replace(fast, terms=fast_terms)
    slow, slow_p = mission()
    slow = dataclasses.replace(
        slow,
        planner=fast.planner,
        num_iters=fast.num_iters,
        mixture_k=fast.mixture_k,
        mixture_schedule=fast.mixture_schedule,
        planning_belief=fast.planning_belief,
        policy_len=fast.policy_len,
        epistemic=fast.epistemic,
    )
    system, params = merge({"slow": slow, "fast": fast}, {"slow": slow_p, "fast": fast_p})
    if composite_terms:
        system = dataclasses.replace(system, terms=composite_terms)
    return system, params


def test_a_selector_is_namespaced_on_its_variable_segment():
    assert namespaced_term("rover", "uncertainty:x") == "uncertainty:rover.x"
    assert namespaced_term("rover", "uncertainty:pose:b1") == "uncertainty:rover.pose:b1"
    assert (
        namespaced_term("rover", "predicted_reward:obs:reward")
        == "predicted_reward:obs:rover.reward"
    )
    assert (
        namespaced_term("rover", "occupancy:transition:stoch") == "occupancy:transition:rover.stoch"
    )
    with pytest.raises(ValueError, match="variable segment"):
        namespaced_term("rover", "uncertainty:*")
    with pytest.raises(ValueError, match="variable segment"):
        namespaced_term("rover", "occupancy:transition:*")


def test_a_trailing_dimension_wildcard_is_kept_on_the_namespaced_variable():
    assert namespaced_term("rover", "uncertainty:x:*") == "uncertainty:rover.x:*"
    assert (
        namespaced_term("rover", "predicted_reward:obs:reward:*")
        == "predicted_reward:obs:rover.reward:*"
    )


def test_a_subsystem_view_carries_the_selectors_on_its_own_variables():
    system, params = _pair(composite_terms=("uncertainty:fast.x", "uncertainty:slow.x"))
    view, _ = subsystem(system, params, "fast")
    assert view.terms == ("uncertainty:fast.x",)


def test_a_components_selector_equals_the_composites_declared_by_hand():
    merged, params = _pair(fast_terms=("uncertainty:x",))
    by_hand, _ = _pair(composite_terms=("uncertainty:fast.x",))
    assert merged.terms == ("uncertainty:fast.x",)
    mounts = merged._row_mounts(params)
    assert [(m.name, m.index) for m in mounts] == [("uncertainty", (1,))]
    beliefs = merged.initial_priors(params)
    _, tree = merged.infer_policies(params, beliefs)
    _, hand = by_hand.infer_policies(params, beliefs)
    assert jax.tree_util.tree_structure(tree) == jax.tree_util.tree_structure(hand)
    for got, want in zip(
        jax.tree_util.tree_leaves(tree), jax.tree_util.tree_leaves(hand), strict=True
    ):
        np.testing.assert_array_equal(got, want)
    _, bare = _pair()[0].infer_policies(params, beliefs)
    assert jax.tree_util.tree_structure(bare) != jax.tree_util.tree_structure(tree)


def test_a_component_declaring_scoring_is_still_refused():
    fast, fast_p = scalar_model(horizon=2)
    fast = dataclasses.replace(fast, terms=("uncertainty:x",), scoring=(("g_tail", object()),))
    with pytest.raises(NotImplementedError, match=r"fast.*scoring"):
        merge({"fast": fast}, {"fast": fast_p})
    fast, fast_p = scalar_model(horizon=2)
    fast_p = dataclasses.replace(fast_p, scoring={"g_tail": 1.0})
    with pytest.raises(NotImplementedError, match=r"fast.*scoring"):
        merge({"fast": dataclasses.replace(fast, terms=("g_tail",))}, {"fast": fast_p})
