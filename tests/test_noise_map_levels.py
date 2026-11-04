"""A noise map with a level axis: ``GridNoiseMap`` values ``(*levels, *grid, d_obs, d_obs)``.

A modality over a Gaussian state and an action node (a two-level sensor) reads the noise of the
level the node realised, off the grid at the belief's mean. At each level it equals the plain map
declared for that level alone: in ``infer_states``, in planning with observations
(``planning_belief="posterior"``), in the information-gain leaf, under ``jax.grad``, and in the
environment; the description round-trips it.
"""

from __future__ import annotations

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.controls import Box, ControlStep
from hybrid_aif.description import build, describe
from hybrid_aif.description.mdl import description_length
from hybrid_aif.distributions import Categorical, ConditionalGaussian, Gaussian
from hybrid_aif.factors import (
    GridNoiseMap,
    LinearGaussianObservation,
    LinearGaussianTransition,
)
from hybrid_aif.planning.enumerate import evaluate_policies, evaluate_policies_total
from hybrid_aif.system import System, SystemParams
from tests.test_noise_map import F2, G2, _grid_map, _point_mass

KEY = jax.random.PRNGKey(0)
LEVEL_MAPS = (_grid_map(0.5, 0.05), _grid_map(0.15, 0.4))  # coarse-then-precise, and the reverse


def _levelled(maps=LEVEL_MAPS) -> GridNoiseMap:
    first = maps[0]
    return GridNoiseMap(jnp.stack([m.values for m in maps]), first.low, first.high, axes=first.axes)


def _sensed_mass(noise, policy_len=3):
    """The point mass of ``test_noise_map`` with a two-level sensor node its fix reads."""
    system = System(
        A_deps=((0, 4),),
        B_deps=((1, 2), (), ()),
        controls=(1, 2),
        action_names=("a", "sensor"),
        modality_names=("fix",),
        factor_names=("x", "u", "s"),
        policy_len=policy_len,
        epistemic=True,
        planning_belief="posterior",
    )
    params = SystemParams(
        likelihood={"fix": LinearGaussianObservation(H=jnp.eye(2), c=jnp.zeros(2), noise=noise)},
        transition={
            "x": LinearGaussianTransition(
                F=F2,
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((2, 2)),
                    chol=jnp.broadcast_to(0.02 * jnp.eye(2), (2, 2, 2)),
                    parent_ndims=1,
                ),
                G=G2,
            )
        },
        initial_state_prior={"x": Gaussian(loc=jnp.array([0.6, 0.0]), chol=0.1 * jnp.eye(2))},
        control={"u": Box(-2.0 * jnp.ones(2), 2.0 * jnp.ones(2))},
        terminal_targets={"x": Gaussian(loc=jnp.array([1.0, 0.0]), chol=0.1 * jnp.eye(2))},
    )
    return system, params


def _plain(level: int, policy_len=3):
    """The same model with the sensor node gone and the plain map of ``level``."""
    system, params = _point_mass(LEVEL_MAPS[level], policy_len=policy_len)
    start = {"x": Gaussian(loc=jnp.array([0.6, 0.0]), chol=0.1 * jnp.eye(2))}
    params = dataclasses.replace(params, initial_state_prior=start)
    system = dataclasses.replace(system, planner="enumeration", planning_belief="posterior")
    return system, params


def test_a_levelled_map_is_the_plain_map_of_each_level_and_validates():
    both = _levelled()
    assert both.n_levels == 1 and both.levels == (2,)
    loc = jnp.array([[0.9, 0.1], [-0.4, 0.7]])
    for level, plain in enumerate(LEVEL_MAPS):
        np.testing.assert_array_equal(np.asarray(both(loc)[level]), np.asarray(plain(loc)))
        np.testing.assert_array_equal(
            np.asarray(both.select((jnp.asarray(level),))(loc)), np.asarray(plain(loc))
        )
    assert both(jnp.zeros(2)).shape == (2, 2, 2) and LEVEL_MAPS[0].n_levels == 0
    with pytest.raises(ValueError, match="level"):
        both.select(())
    #  the leading axes must be the modality's parent axes
    system, params = _sensed_mass(both)
    bad = GridNoiseMap(jnp.stack([LEVEL_MAPS[0].values] * 3), both.low, both.high, axes=both.axes)
    with pytest.raises(ValueError, match="leading"):
        _sensed_mass(bad)[0].planner_context(_sensed_mass(bad)[1], system.initial_priors(params))
    shared_system, shared_params = _sensed_mass(LEVEL_MAPS[0])  # no level axis: no dependence
    with pytest.raises(ValueError, match="does not depend on the mode"):
        shared_system.planner_context(shared_params, shared_system.initial_priors(shared_params))
    levelled_alone = _point_mass(both)
    with pytest.raises(ValueError, match="leading"):
        levelled_alone[0].planner_context(
            levelled_alone[1], levelled_alone[0].initial_priors(levelled_alone[1])
        )


@pytest.mark.parametrize("level", [0, 1])
def test_inference_reads_the_realised_levels_map(level):
    system, params = _sensed_mass(_levelled())
    plain, plain_params = _plain(level)
    prior = Gaussian(loc=jnp.array([0.9, 0.1]), chol=0.2 * jnp.eye(2))
    y = jnp.array([0.95, 0.0])
    copy = Categorical.from_probs(jax.nn.one_hot(level, 2))
    got, detail = system.infer_states(params, {"fix": y}, {"x": prior, "sensor": copy}, detail=True)
    want, ref = plain.infer_states(plain_params, {"fix": y}, {"x": prior}, detail=True)
    np.testing.assert_allclose(np.asarray(got["x"].loc), np.asarray(want["x"].loc), atol=1e-6)
    np.testing.assert_allclose(np.asarray(got["x"].chol), np.asarray(want["x"].chol), atol=1e-6)
    np.testing.assert_allclose(
        float(detail.log_evidence["fix"]), float(ref.log_evidence["fix"]), atol=1e-5
    )
    assert jnp.array_equal(got["sensor"].probs, copy.probs)


def _policy(plan, level):
    """One policy, a velocity plan and the sensor held at ``level``, as ``(P=1, H, ...)``."""
    horizon = plan.shape[0]
    return ControlStep(
        entries=(plan[None], jnp.full((1, horizon), level, jnp.int32)),
    )


@pytest.mark.parametrize("level", [0, 1])
def test_planning_with_observations_and_the_information_gain_leaf_match_the_plain_map(level):
    system, params = _sensed_mass(_levelled())
    plain, plain_params = _plain(level)
    ctx = system.planner_context(params, system.initial_priors(params))
    plain_ctx = plain.planner_context(plain_params, plain.initial_priors(plain_params))
    plan = jnp.array([[1.5, 0.3], [1.0, -0.2], [0.5, 0.1]])
    tree = evaluate_policies(
        _policy(plan, level),
        ctx.init_beliefs,
        ctx.transition_fn,
        ctx.step_terms_fn,
        ctx.terminal_terms_fn,
    )
    want = evaluate_policies(
        plan[None],
        plain_ctx.init_beliefs,
        plain_ctx.transition_fn,
        plain_ctx.step_terms_fn,
        plain_ctx.terminal_terms_fn,
    )
    got_leaf, want_leaf = tree["infogain"]["fix"].lower, want["infogain"]["fix"].lower
    np.testing.assert_allclose(np.asarray(got_leaf), np.asarray(want_leaf), rtol=1e-5, atol=1e-6)
    total = evaluate_policies_total(
        _policy(plan, level),
        ctx.init_beliefs,
        ctx.transition_fn,
        ctx.step_terms_fn,
        ctx.terminal_terms_fn,
    ).midpoint
    ref = evaluate_policies_total(
        plan[None],
        plain_ctx.init_beliefs,
        plain_ctx.transition_fn,
        plain_ctx.step_terms_fn,
        plain_ctx.terminal_terms_fn,
    ).midpoint
    np.testing.assert_allclose(np.asarray(total), np.asarray(ref), rtol=1e-5, atol=1e-6)
    #  the other level scores differently: the node's level was read, not a default
    other = evaluate_policies(
        _policy(plan, 1 - level),
        ctx.init_beliefs,
        ctx.transition_fn,
        ctx.step_terms_fn,
        ctx.terminal_terms_fn,
    )["infogain"]["fix"].lower
    assert not np.allclose(np.asarray(other), np.asarray(got_leaf))


@pytest.mark.parametrize("level", [0, 1])
def test_the_gradient_through_the_levelled_map_matches_the_plain_map(level):
    system, params = _sensed_mass(_levelled())
    plain, plain_params = _plain(level)
    ctx = system.planner_context(params, system.initial_priors(params))
    plain_ctx = plain.planner_context(plain_params, plain.initial_priors(plain_params))

    def score(plan):
        return evaluate_policies_total(
            _policy(plan, level),
            ctx.init_beliefs,
            ctx.transition_fn,
            ctx.step_terms_fn,
            ctx.terminal_terms_fn,
        ).midpoint[0]

    def reference(plan):
        return evaluate_policies_total(
            plan[None],
            plain_ctx.init_beliefs,
            plain_ctx.transition_fn,
            plain_ctx.step_terms_fn,
            plain_ctx.terminal_terms_fn,
        ).midpoint[0]

    plan = jnp.array([[1.5, 0.3], [1.0, -0.2], [0.5, 0.1]])
    grad = np.asarray(jax.grad(score)(plan))
    assert np.isfinite(grad).all() and np.abs(grad).max() > 0.0
    np.testing.assert_allclose(grad, np.asarray(jax.grad(reference)(plan)), rtol=1e-4, atol=1e-6)
    eps = 1e-3
    for i, j in ((0, 0), (1, 1), (2, 0)):
        bump = jnp.zeros_like(plan).at[i, j].set(eps)
        fd = (float(score(plan + bump)) - float(score(plan - bump))) / (2 * eps)
        assert grad[i, j] == pytest.approx(fd, rel=2e-2, abs=2e-3)


@pytest.mark.parametrize("level", [0, 1])
def test_the_environment_draws_at_the_realised_levels_noise(level):
    system, params = _sensed_mass(_levelled())
    env, eparams = system.as_env(params)
    _, state = env.reset(KEY, eparams)
    keys = jax.random.split(jax.random.PRNGKey(1), 600)
    actions = {"a": jnp.zeros(2), "sensor": jnp.asarray(level)}
    draws = jax.vmap(lambda k: env.step_actuated(k, state, actions, eparams)[0]["fix"])(keys)
    #  the reading is of the next state, which zero velocity leaves near the reset state
    sd_at = float(np.asarray(LEVEL_MAPS[level](jnp.asarray(state["x"]))[0, 0]))
    spread = float(np.asarray(draws)[:, 1].std())
    assert spread == pytest.approx(np.hypot(sd_at, 0.02), rel=0.15)


def test_the_description_round_trips_and_counts_every_level():
    system, params = _sensed_mass(_levelled())
    document, arrays = describe(system, params, arrays="inline")
    document = json.loads(json.dumps(document))
    rebuilt_system, rebuilt = build(document, arrays=arrays)
    noise = rebuilt.likelihood["fix"].noise
    assert isinstance(noise, GridNoiseMap) and noise.n_levels == 1 and noise.axes == (0, 1)
    np.testing.assert_array_equal(
        np.asarray(noise.values), np.asarray(params.likelihood["fix"].noise.values)
    )
    assert rebuilt_system == system
    import jsonschema

    import hybrid_aif as H

    jsonschema.validate(document, H.schema())
    one, one_params = _sensed_mass(LEVEL_MAPS[0])
    two = description_length(system, 100, params=params)
    single = description_length(one, 100, params=one_params)
    assert two.total > single.total  # a second level's grid is a second set of parameters


def test_the_mixed_planner_scores_the_levelled_map_through_infer_policies():
    system, params = _sensed_mass(_levelled(), policy_len=2)
    system = dataclasses.replace(system, planner="mixed", num_iters=5)
    q_pi, tree = system.infer_policies(params, system.initial_priors(params))
    assert np.isfinite(np.asarray(q_pi.probs)).all()
    assert "fix" in tree["infogain"]
