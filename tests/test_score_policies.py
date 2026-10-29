"""Acceptance-fix 2: ``System.score_policies`` — the planner-free scoring face.

Scoring a given policy set used to require hand-building a planner around the
injection seam. ``score_policies`` runs the same ``planner_context`` closures
through ``evaluate_policies`` directly and returns the per-policy term tree;
it must agree leaf-for-leaf with a hand-built injected planner on both the
discrete (T-maze) and hybrid (switching) paths, and reproduce the switching
demo's expected G values.
"""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.objectives import total
from hybrid_aif.planning import evaluate_policies, policy_posterior
from tests.test_hybrid_switching_system import (
    COMMIT_RIGHT,
    POLICIES,
    demo_scores,
    initial_beliefs,
)
from tests.test_hybrid_switching_system import make_params as switching_params
from tests.test_hybrid_switching_system import make_system as switching_system
from tests.test_system import _initial_beliefs, tmaze_system


def _switching_beliefs(system):
    """The switching fixture's initial beliefs, keyed by factor (as the faces take them)."""
    beliefs = initial_beliefs()
    return beliefs if isinstance(beliefs, Mapping) else system.from_positional("beliefs", beliefs)


def _assert_trees_equal(got, ref):
    assert {t: set(b) for t, b in got.items()} == {t: set(b) for t, b in ref.items()}
    for term, blocks in ref.items():
        for block, iv in blocks.items():
            assert jnp.array_equal(got[term][block].lower, iv.lower), (term, block)
            assert jnp.array_equal(got[term][block].upper, iv.upper), (term, block)


# --------------------------------------------------------------------- discrete


def test_discrete_matches_hand_built_injected_planner():
    system, params = tmaze_system()
    beliefs = _initial_beliefs(system, params)
    subset = system.policies(params)[jnp.asarray([0, 5, 11])]

    def hand_built(ctx, use_bound="midpoint", metrics=True):
        tree = evaluate_policies(
            subset, ctx.init_beliefs, ctx.transition_fn, ctx.step_terms_fn, ctx.terminal_terms_fn
        )
        q_pi = policy_posterior(total(tree), ctx.gamma, ctx.policy_prior, use_bound=use_bound)
        return q_pi, tree, subset

    _, tree_planner = system.infer_policies(params, beliefs, planner=hand_built)
    tree_face = system.score_policies(params, beliefs, subset)
    _assert_trees_equal(tree_face, tree_planner)


def test_discrete_matches_default_enumeration_on_the_full_set():
    system, params = tmaze_system()
    beliefs = _initial_beliefs(system, params)
    _, tree = system.infer_policies(params, beliefs)
    tree_face = system.score_policies(params, beliefs, system.policies(params))
    _assert_trees_equal(tree_face, tree)


# ----------------------------------------------------------------------- hybrid


def test_hybrid_reproduces_switching_demo_g_intervals():
    system, params = switching_system(), switching_params()
    full = POLICIES[:, :, None]
    tree = system.score_policies(params, _switching_beliefs(system), full)
    g = total(tree)
    g_demo = jax.vmap(demo_scores)(POLICIES)
    np.testing.assert_allclose(np.asarray(g.lower), np.asarray(g_demo.lower), atol=1e-4)
    np.testing.assert_allclose(np.asarray(g.upper), np.asarray(g_demo.upper), atol=1e-4)
    assert int(jnp.argmin(g.midpoint)) == COMMIT_RIGHT
    assert set(tree) == {"risk", "uncertainty"}  # x is unobserved: its entropy is the row
    assert tree["risk"]["x"].lower.shape == (len(POLICIES),)


def test_hybrid_matches_the_fixtures_injected_planner():
    from tests.test_hybrid_switching_system import six_policy_planner

    system, params = switching_system(), switching_params()
    _, tree_planner = system.infer_policies(
        params, _switching_beliefs(system), planner=six_policy_planner
    )
    full = POLICIES[:, :, None]
    tree_face = system.score_policies(params, _switching_beliefs(system), full)
    _assert_trees_equal(tree_face, tree_planner)


# ------------------------------------------------------- single policy + validation


def test_single_policy_squeezes_the_policy_axis():
    system, params = switching_system(), switching_params()
    full = POLICIES[:, :, None]
    tree_all = system.score_policies(params, _switching_beliefs(system), full)
    tree_one = system.score_policies(params, _switching_beliefs(system), full[COMMIT_RIGHT])
    for term, blocks in tree_one.items():
        for block, iv in blocks.items():
            assert iv.lower.shape == ()
            assert jnp.array_equal(iv.lower, tree_all[term][block].lower[COMMIT_RIGHT])
            assert jnp.array_equal(iv.upper, tree_all[term][block].upper[COMMIT_RIGHT])


def test_score_policies_jits():
    system, params = switching_system(), switching_params()
    full = POLICIES[:, :, None]
    eager = total(system.score_policies(params, _switching_beliefs(system), full)).midpoint
    f = jax.jit(lambda p, b: total(system.score_policies(p, b, full)).midpoint)
    np.testing.assert_allclose(
        np.asarray(f(params, _switching_beliefs(system))), np.asarray(eager), atol=1e-6
    )


def test_validation_is_loud():
    system, params = tmaze_system()
    beliefs = _initial_beliefs(system, params)
    good = system.policies(params)
    with pytest.raises(ValueError, match="integer"):
        system.score_policies(params, beliefs, good.astype(jnp.float32))
    with pytest.raises(ValueError, match="shape"):
        system.score_policies(params, beliefs, good[0, 0])  # ndim 1
    with pytest.raises(ValueError, match="shape"):
        system.score_policies(params, beliefs, good[None])  # ndim 4
    with pytest.raises(ValueError, match="one column per control node"):
        # a per-factor-width plan (an extra column for the uncontrolled factor)
        system.score_policies(params, beliefs, jnp.concatenate([good, 0 * good], axis=-1))
