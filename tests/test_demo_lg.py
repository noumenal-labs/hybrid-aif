"""The linear-Gaussian demo (`examples/hill_lg.py`) as a regression fixture.

Closed-loop behavior (reaches the hilltop goal region deterministically),
filter quality against the known-truth simulator, planner sanity (goal-ward
thrust from either side of the hill), exactness of the Gaussian risk
Intervals, metrics-tree structure, a jit smoke check on the tick, and
numerical parity of the System's first tick against the v0.1 hand-built
enclosure (filter + PlannerContext rebuilt inline from the same observation).
"""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import Gaussian
from hybrid_aif.inference.linear_gaussian import condition, predict
from hybrid_aif.objectives import total
from hybrid_aif.objectives.continuous import risk, state_info_gain, uncertainty
from hybrid_aif.planning import get_planner
from hybrid_aif.planning.base import PlannerContext

_SPEC = importlib.util.spec_from_file_location(
    "hill_lg", Path(__file__).resolve().parents[1] / "examples" / "hill_lg.py"
)
hill_lg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hill_lg)

KEY = jax.random.PRNGKey(0)


@pytest.fixture(scope="module")
def rollout():
    system, params = hill_lg.make_agent()
    traj = hill_lg.closed_loop(KEY, system, params, num_steps=60)
    return system, params, traj


def _first_action(system, params, belief) -> int:
    """Argmax-policy first thrust index from a belief, via the System's act face."""
    action, _, _ = system.act(
        jax.random.PRNGKey(0), params, {"state": belief}, selection="deterministic", mode="policy"
    )
    return int(action["a_state"])


def _tight_belief(p: float, v: float) -> Gaussian:
    return Gaussian(loc=jnp.array([p, v]), chol=0.05 * jnp.eye(2))


def test_closed_loop_reaches_goal_deterministically(rollout):
    _, _, traj = rollout
    reached = np.asarray(jax.vmap(hill_lg.in_goal_region)(traj.xs))
    assert reached.any(), "closed loop never entered the goal region"
    assert int(reached.argmax()) <= 40, "took more than 40 ticks to reach the goal"
    assert reached[-1], "left the goal region by the final tick"
    # Deterministic: the same key reproduces the run bitwise.
    system, params = hill_lg.make_agent()
    again = hill_lg.closed_loop(KEY, system, params, num_steps=60)
    assert jnp.array_equal(traj.xs, again.xs)
    assert jnp.array_equal(traj.actions, again.actions)


def test_filter_tracks_true_state(rollout):
    _, _, traj = rollout
    mse_pos = float(jnp.mean((traj.means[:, 0] - traj.xs[:, 0]) ** 2))
    mse_vel = float(jnp.mean((traj.means[:, 1] - traj.xs[:, 1]) ** 2))
    # Position: the posterior must beat the raw observation noise floor (var 2.5e-3).
    assert mse_pos < 2.5e-3
    # Velocity is never observed; it must still be inferred through the dynamics.
    assert mse_vel < 1e-2


def test_planner_prefers_goalward_thrust():
    system, params = hill_lg.make_agent()
    assert _first_action(system, params, _tight_belief(0.0, 0.0)) == 2  # below goal: thrust up
    assert _first_action(system, params, _tight_belief(2.0, 0.0)) == 0  # beyond goal: thrust down


def test_risk_interval_exact():
    system, params = hill_lg.make_agent(include_entropy=False, epistemic=False)
    belief = _tight_belief(0.3, 0.1)
    # Planner-free scoring face: one one-step policy per thrust, in thrust order.
    policies = jnp.arange(3, dtype=jnp.int32)[:, None, None]
    tree = system.score_policies(params, {"state": belief}, policies)
    assert set(tree) == {"risk"}  # the selector ablates the epistemic leaf, terms= the row
    leaf = tree["risk"]["state"]
    assert np.array_equal(np.asarray(leaf.lower), np.asarray(leaf.upper))  # exact Interval
    # Each leaf is the closed-form Gaussian KL of predict-through-that-mode
    # to the goal target.
    B, goal = params.transition["state"], params.state_targets["state"]
    for j in range(3):
        innov = B.innovations.condition_on((j,))
        manual = predict(belief, B.F, innov.loc, innov.chol).kl_to(goal)
        np.testing.assert_allclose(leaf.midpoint[j], manual.midpoint, rtol=1e-5)


def test_metrics_tree_structure(rollout):
    _, _, traj = rollout
    tree = traj.first_tree
    assert set(tree) == {"risk", "uncertainty", "infogain"}
    assert set(tree["risk"]) == set(tree["uncertainty"]) == {"state"}
    assert set(tree["infogain"]) == {"position"}  # channel-keyed
    for blocks in tree.values():
        for leaf in blocks.values():
            assert leaf.lower.shape == (3**4,)  # one value per enumerated policy
    g = total(tree)
    assert bool(jnp.all(jnp.isfinite(g.midpoint)))
    assert bool(jnp.all(g.width == 0.0))  # all-Gaussian pipeline: G is exact


def test_first_tick_matches_v01_enclosure(rollout):
    """Acceptance parity: the System's recorded first tick equals the v0.1
    hand-built enclosure — `condition` filter, predict-through-mode transition,
    risk + belief-entropy step terms (now the `uncertainty` row) — recomputed inline
    from the same y_0, plus the position reading's information gain."""
    system, params, traj = rollout
    A, B = params.likelihood["position"], params.transition["state"]
    goal = params.state_targets["state"]
    posterior, _ = condition(params.initial_state_prior["state"], A.H, A.c, A.R_chol, traj.ys[0])
    np.testing.assert_allclose(np.asarray(traj.means[0]), np.asarray(posterior.loc), atol=1e-5)

    def transition_fn(qs, action):
        innov = B.innovations.condition_on((action[0],))
        return (predict(qs[0], B.F, innov.loc, innov.chol),)

    def step_terms_fn(qs, action, t):
        del action, t
        return {
            **risk(qs, {"state": goal}, ("state",)),
            **uncertainty(qs, ("state",)),
            **state_info_gain(qs, (A,), ("position",)),
        }

    ctx = PlannerContext(
        transition_fn=transition_fn,
        step_terms_fn=step_terms_fn,
        terminal_terms_fn=None,
        controls=system._table,
        levels=(3,),
        policy_len=4,
        gamma=16.0,
        policy_prior=None,
        init_beliefs=(posterior,),
    )
    _, old_tree, _ = get_planner("enumeration")(ctx)
    assert set(old_tree) == set(traj.first_tree)
    for term in old_tree:
        assert set(old_tree[term]) == set(traj.first_tree[term])
        for block in old_tree[term]:
            np.testing.assert_allclose(
                np.asarray(traj.first_tree[term][block].midpoint),
                np.asarray(old_tree[term][block].midpoint),
                atol=1e-5,
            )
    np.testing.assert_allclose(
        np.asarray(total(traj.first_tree).midpoint),
        np.asarray(total(old_tree).midpoint),
        atol=1e-5,
    )


def test_tick_jit_smoke():
    system, params = hill_lg.make_agent()
    obs = {"position": jnp.array([0.02])}
    priors = dict(params.initial_state_prior)  # the priors, keyed by factor, as beliefs
    posteriors, predicted, action, _ = hill_lg.tick(system, params, obs, priors)
    e_post, e_pred, e_action, _ = hill_lg.tick.__wrapped__(system, params, obs, priors)
    positional = system.to_positional("actions", action)
    assert positional.dtype == jnp.int32 and positional.shape == (1,)
    assert jnp.array_equal(action["a_state"], e_action["a_state"])
    np.testing.assert_allclose(posteriors["state"].loc, e_post["state"].loc, rtol=1e-6)
    np.testing.assert_allclose(predicted["state"].loc, e_pred["state"].loc, rtol=1e-6)
    assert bool(jnp.all(jnp.isfinite(predicted["state"].chol)))
