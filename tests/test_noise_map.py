"""A noise that depends on the state (``docs/design/planning-with-observations.md`` D3).

``GridNoiseMap`` reads an observation's ``R_chol`` off a grid over the observed state by
multilinear interpolation; every consumer reads it at the belief's mean, the extended
filter's convention: inference at the prior mean, planning at the predicted mean, the
environment at the true state. Pinned by hand against the interpolation, a Kalman update,
``jax.grad`` against finite differences through the map, and a plan-dependent choice: with
the epistemic chain and information gain on, the gradient plan detours through the region
where the sensor is precise; with information gain off it does not.
"""

from __future__ import annotations

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hybrid_aif as H
from hybrid_aif.controls import Box
from hybrid_aif.description import build, describe
from hybrid_aif.distributions import ConditionalGaussian, Gaussian, MixtureGaussian
from hybrid_aif.factors import (
    GridNoiseMap,
    LinearGaussianObservation,
    LinearGaussianTransition,
    NoiseMap,
)
from hybrid_aif.inference.linear_gaussian import condition
from hybrid_aif.planning.enumerate import evaluate_policies_total
from hybrid_aif.planning.gradient import gradient_plan
from hybrid_aif.system import System, SystemParams

# ------------------------------------------------------------------ the map itself


def _chol(sd: float) -> np.ndarray:
    return np.diag([sd, sd]).astype(np.float32)


def _grid_map(sd_far=0.5, sd_near=0.05, low=(-1.0, -1.0), high=(1.0, 1.0), n=5):
    """A 2-D grid over the state's two coordinates, precise around the node (1, 0)."""
    values = np.stack([_chol(sd_far)] * (n * n)).reshape(n, n, 2, 2)
    values[-1, n // 2] = _chol(sd_near)
    return GridNoiseMap(jnp.asarray(values), low, high, axes=(0, 1))


def test_grid_noise_map_interpolates_between_its_nodes():
    """Exact at a node, the average of two nodes halfway between them, the edge beyond the
    extent, a batch of means at once, and the one-axis case."""
    m = _grid_map()
    np.testing.assert_allclose(np.asarray(m(jnp.array([1.0, 0.0]))), _chol(0.05), atol=1e-6)
    np.testing.assert_allclose(np.asarray(m(jnp.array([-1.0, -1.0]))), _chol(0.5), atol=1e-6)
    halfway = m(jnp.array([0.75, 0.0]))  # between the nodes at x = 0.5 and x = 1.0
    np.testing.assert_allclose(np.asarray(halfway), _chol(0.5 * (0.5 + 0.05)), atol=1e-6)
    beyond = m(jnp.array([3.0, 0.0]))
    np.testing.assert_allclose(np.asarray(beyond), _chol(0.05), atol=1e-6)
    batch = m(jnp.array([[1.0, 0.0], [-1.0, -1.0]]))
    assert batch.shape == (2, 2, 2)
    line = GridNoiseMap(jnp.asarray([_chol(0.1), _chol(0.3)]), -1.0, 1.0, axes=(1,))
    np.testing.assert_allclose(np.asarray(line(jnp.array([9.0, 0.0]))), _chol(0.2), atol=1e-6)


def test_grid_noise_map_is_validated():
    with pytest.raises(ValueError, match="one grid axis per entry"):
        GridNoiseMap(jnp.asarray(np.stack([_chol(0.1)] * 2)), -1.0, 1.0, axes=(0, 1))
    with pytest.raises(ValueError, match="at least two nodes"):
        GridNoiseMap(jnp.asarray(np.stack([_chol(0.1)]).reshape(1, 2, 2)), -1.0, 1.0, axes=(0,))
    with pytest.raises(ValueError, match="low < high"):
        GridNoiseMap(jnp.asarray(np.stack([_chol(0.1)] * 2)), 1.0, -1.0, axes=(0,))
    with pytest.raises(ValueError, match="lower-triangular"):
        GridNoiseMap(jnp.asarray(np.stack([np.ones((2, 2), np.float32)] * 2)), -1.0, 1.0, axes=(0,))
    with pytest.raises(ValueError, match="non-negative"):
        GridNoiseMap(jnp.asarray(np.stack([_chol(0.1)] * 2)), -1.0, 1.0, axes=(-1,))
    with pytest.raises(ValueError, match="names a coordinate twice"):
        GridNoiseMap(jnp.asarray(np.stack([_chol(0.1)] * 4).reshape(2, 2, 2, 2)), -1.0, 1.0, (0, 0))


# ------------------------------------------------------------------ a point mass that senses


DT = 0.25
F2 = jnp.eye(2)
G2 = DT * jnp.eye(2)


def _emission(noise):
    return LinearGaussianObservation(H=jnp.eye(2), c=jnp.zeros(2), noise=noise)


def _point_mass(noise, *, epistemic=True, planning_belief="posterior", policy_len=4):
    """A 2-D position under a continuous velocity control in a box, observed through
    ``noise``, with a goal straight ahead of the start."""
    system = System(
        A_deps=((0,),),
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("a",),
        modality_names=("fix",),
        factor_names=("x", "u"),
        policy_len=policy_len,
        planner="gradient",
        epistemic=epistemic,
        planning_belief=planning_belief,
    )
    params = SystemParams(
        likelihood={"fix": _emission(noise)},
        transition={
            "x": LinearGaussianTransition(
                F=F2,
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 2)),
                    chol=jnp.broadcast_to(0.02 * jnp.eye(2), (1, 2, 2)),
                    parent_ndims=1,
                ),
                G=G2,
            )
        },
        initial_state_prior={"x": Gaussian(loc=jnp.array([-1.0, 0.0]), chol=0.1 * jnp.eye(2))},
        control={"u": Box(-2.0 * jnp.ones(2), 2.0 * jnp.ones(2))},
        terminal_targets={"x": Gaussian(loc=jnp.array([1.0, 0.0]), chol=0.1 * jnp.eye(2))},
    )
    return system, params


def test_inference_reads_the_map_at_the_prior_mean():
    """``infer_states`` with a map equals a hand-written Kalman update with ``R`` read at
    the prior mean, and the tick's evidence likewise."""
    m = _grid_map()
    system, params = _point_mass(m)
    prior = Gaussian(loc=jnp.array([0.9, 0.1]), chol=0.2 * jnp.eye(2))
    y = jnp.array([0.95, 0.0])
    beliefs, detail = system.infer_states(params, {"fix": y}, priors={"x": prior}, detail=True)
    want, log_ev = condition(prior, jnp.eye(2), jnp.zeros(2), m(prior.loc), y)
    np.testing.assert_allclose(np.asarray(beliefs["x"].loc), np.asarray(want.loc), atol=1e-6)
    np.testing.assert_allclose(np.asarray(beliefs["x"].chol), np.asarray(want.chol), atol=1e-6)
    assert float(detail.log_evidence["fix"]) == pytest.approx(float(log_ev), abs=1e-5)
    # the same tick with the constant far noise differs: the map was read, not a default
    elsewhere, _ = condition(prior, jnp.eye(2), jnp.zeros(2), jnp.asarray(_chol(0.5)), y)
    assert not np.allclose(np.asarray(beliefs["x"].chol), np.asarray(elsewhere.chol))


def test_the_environment_draws_at_the_true_state():
    m = _grid_map()
    system, params = _point_mass(m)
    env, env_params = system.as_env(params)
    keys = jax.random.split(jax.random.PRNGKey(0), 4000)

    # reset draws the state from the initial prior at (-1, 0), where the sensor is coarse
    first = env.reset(keys[0], env_params)[0]
    key_or_index = "fix" if isinstance(first, dict) else 0
    obs = jax.vmap(lambda k: env.reset(k, env_params)[0][key_or_index])(keys)
    assert float(jnp.std(obs[:, 1])) == pytest.approx(np.sqrt(0.5**2 + 0.1**2), rel=0.1)


def test_the_map_is_refused_where_it_is_not_built():
    m = _grid_map()
    with pytest.raises(ValueError, match="outside the observed factor"):
        bad = GridNoiseMap(m.values, m.low, m.high, axes=(0, 5))
        system, params = _point_mass(bad)
        system.planner_context(params, system.initial_priors(params))
    # a mixture-carried factor
    system, params = _point_mass(m)
    mixture = dataclasses.replace(
        params,
        initial_state_prior={
            "x": MixtureGaussian(
                log_weights=jnp.log(jnp.array([0.5, 0.5])),
                means=jnp.array([[-1.0, 0.0], [-1.0, 0.2]]),
                chols=jnp.broadcast_to(0.1 * jnp.eye(2), (2, 2, 2)),
            )
        },
    )
    with pytest.raises(NotImplementedError, match="mixture-carried"):
        dataclasses.replace(system, mixture_k=2).planner_context(
            mixture, dataclasses.replace(system, mixture_k=2).initial_priors(mixture)
        )
    with pytest.raises(TypeError, match="R_chol_at"):
        _ = _emission(m).R_chol


def test_the_predictive_observation_reads_the_map_at_the_mean():
    """``q(o)`` through a map has covariance ``H Σ Hᵀ + R(μ)`` at the belief's mean, per
    component for a mixture; the utility and ambiguity leaves move with the mean."""
    from hybrid_aif.objectives.continuous import conditional_obs_entropy, predictive_obs

    m = _grid_map()
    model = _emission(m)
    q = Gaussian(loc=jnp.array([1.0, 0.0]), chol=0.2 * jnp.eye(2))
    pred = predictive_obs(q, model)
    want = 0.04 * np.eye(2) + np.asarray(m(q.loc)) @ np.asarray(m(q.loc)).T
    np.testing.assert_allclose(np.asarray(pred.chols[0] @ pred.chols[0].T), want, atol=1e-6)
    mix = MixtureGaussian(
        log_weights=jnp.log(jnp.array([0.3, 0.7])),
        means=jnp.array([[1.0, 0.0], [-1.0, -1.0]]),
        chols=jnp.broadcast_to(0.2 * jnp.eye(2), (2, 2, 2)),
    )
    pred = predictive_obs(mix, model)
    for j, mu in enumerate(np.asarray(mix.means)):
        r = np.asarray(m(jnp.asarray(mu)))
        np.testing.assert_allclose(
            np.asarray(pred.chols[j] @ pred.chols[j].T), 0.04 * np.eye(2) + r @ r.T, atol=1e-6
        )
    h = float(conditional_obs_entropy(model, q=mix))
    ent = lambda sd: 0.5 * 2 * (1.0 + np.log(2 * np.pi)) + 2 * np.log(sd)  # noqa: E731
    assert h == pytest.approx(0.3 * ent(0.05) + 0.7 * ent(0.5), abs=1e-5)

    system, params = _point_mass(m, policy_len=2)
    params = dataclasses.replace(
        params,
        initial_state_prior={"x": Gaussian(loc=jnp.array([0.5, 0.0]), chol=0.1 * jnp.eye(2))},
        obs_targets={"fix": Gaussian(loc=jnp.zeros(2), chol=0.5 * jnp.eye(2))},
        terminal_targets={},
    )  # from (0.5, 0) one plan reaches the precise node at (1, 0), the other stays put
    # V11: the selector mounts the channel's ambiguity, so select it
    system = dataclasses.replace(system, decomposition="risk_ambiguity", epistemic=("obs:fix",))
    ctx = system.planner_context(params, system.initial_priors(params))
    leaves = []
    for plan in (jnp.array([[2.0, 0.0], [2.0, 0.0]]), jnp.array([[0.0, 0.0], [0.0, 0.0]])):
        carry = ctx.init_beliefs
        for u in plan:
            carry = ctx.transition_fn(carry, u)
        tree = ctx.step_terms_fn(carry, plan[-1], jnp.asarray(1, jnp.int32))
        leaves.append((float(tree["risk"]["fix"].lower), float(tree["ambiguity"]["fix"].lower)))
    assert leaves[0][0] != pytest.approx(leaves[1][0])  # the plan moved the mean across the map
    assert leaves[0][1] != pytest.approx(leaves[1][1])


# ------------------------------------------------------------------ gradients and the detour


def _objective(ctx):
    def score(plan):
        return evaluate_policies_total(
            plan[None],
            ctx.init_beliefs,
            ctx.transition_fn,
            ctx.step_terms_fn,
            ctx.terminal_terms_fn,
        ).midpoint[0]

    return score


def test_the_gradient_through_the_map_matches_finite_differences():
    m = _grid_map()
    system, params = _point_mass(m, policy_len=3)
    ctx = system.planner_context(params, system.initial_priors(params))
    score = _objective(ctx)
    plan = jnp.array([[1.5, 0.3], [1.0, -0.2], [0.5, 0.1]])
    grad = np.asarray(jax.grad(score)(plan))
    eps = 1e-3
    for i in range(plan.shape[0]):
        for j in range(plan.shape[1]):
            bump = jnp.zeros_like(plan).at[i, j].set(eps)
            fd = (float(score(plan + bump)) - float(score(plan - bump))) / (2 * eps)
            assert grad[i, j] == pytest.approx(fd, rel=2e-2, abs=2e-3)
    # the plan moves the information gain: the map is read along the plan
    trees = [score(plan), score(plan.at[:, 1].add(0.5))]
    assert trees[0] != pytest.approx(trees[1])


def _final_position(system, params, plan):
    ctx = system.planner_context(params, system.initial_priors(params))
    carry = ctx.init_beliefs
    track = []
    for u in plan:
        carry = ctx.transition_fn(carry, u)
        q = carry[0]
        q = q.predictive if hasattr(q, "predictive") else q
        track.append(np.asarray(q.loc))
    return np.asarray(track)


def test_the_plan_detours_through_the_precise_region_only_with_information_gain():
    """A goal straight ahead and a precise sensor off the direct path: with the epistemic
    chain and information gain on, the gradient plan bends toward the precise region; with
    information gain off, or with the sensor as coarse there as elsewhere, it goes straight."""
    # a smooth well of precision centred above the straight line from (-1, 0) to (1, 0): a
    # blurred grid, so the planner feels its gradient from the line
    xs = np.linspace(-1.0, 1.0, 9)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    sd = 0.6 - 0.57 * np.exp(-((gx - 0.0) ** 2 + (gy - 0.6) ** 2) / 0.25)
    values = np.zeros((9, 9, 2, 2), np.float32)
    values[..., 0, 0] = sd
    values[..., 1, 1] = sd
    precise = GridNoiseMap(jnp.asarray(values), (-1.0, -1.0), (1.0, 1.0), axes=(0, 1))
    flat = GridNoiseMap(
        jnp.asarray(np.stack([_chol(0.6)] * 81).reshape(9, 9, 2, 2)), (-1, -1), (1, 1), (0, 1)
    )

    def mean_height(noise, epistemic):
        system, params = _point_mass(noise, epistemic=epistemic, policy_len=8)
        ctx = system.planner_context(params, system.initial_priors(params))
        plan = gradient_plan(ctx, steps=300).plan
        track = _final_position(system, params, plan)
        return float(np.max(track[:, 1])), track

    seeking, track = mean_height(precise, True)
    straight, _ = mean_height(precise, False)
    coarse, _ = mean_height(flat, True)
    assert seeking > 0.25, track
    assert abs(straight) < 0.1
    assert abs(coarse) < 0.1
    # and the plan still ends near the goal
    assert np.linalg.norm(track[-1] - np.array([1.0, 0.0])) < 0.35, track


# ------------------------------------------------------------------ the description


def test_a_grid_noise_map_round_trips_and_a_custom_map_is_refused():
    m = _grid_map()
    system, params = _point_mass(m)
    document, arrays = describe(system, params, arrays="inline")
    document = json.loads(json.dumps(document))
    record = document["factors"]["fix"]["likelihood"]
    assert set(record["parameters"]["noise"]) == {"values"}
    assert record["options"]["noise_map"] == {
        "low": [-1.0, -1.0],
        "high": [1.0, 1.0],
        "axes": [0, 1],
    }
    rebuilt_system, rebuilt = build(document, arrays=arrays)
    again = rebuilt.likelihood["fix"].noise
    assert isinstance(again, GridNoiseMap) and again.axes == (0, 1)
    np.testing.assert_array_equal(np.asarray(again.values), np.asarray(m.values))
    assert rebuilt_system.planning_belief == system.planning_belief == "posterior"
    import jsonschema

    jsonschema.validate(document, H.schema())  # the document validates against the schema

    class Custom(NoiseMap):
        sd: float

        def __call__(self, loc):
            return self.sd * jnp.eye(2)

    with pytest.raises(H.DescriptionError, match="GridNoiseMap"):
        describe(system, dataclasses.replace(params, likelihood={"fix": _emission(Custom(0.3))}))
    from hybrid_aif.distributions import MatrixNormal

    learned_head = LinearGaussianObservation(
        head=MatrixNormal.from_head(jnp.concatenate([jnp.eye(2), jnp.zeros((2, 1))], 1), 2.0),
        noise=m,
    )
    with pytest.raises(H.DescriptionError, match="clamp the head"):
        describe(system, dataclasses.replace(params, likelihood={"fix": learned_head}))
    broken = json.loads(json.dumps(document))
    del broken["factors"]["fix"]["likelihood"]["options"]["noise_map"]
    with pytest.raises(H.DescriptionError, match="options.noise_map"):
        build(broken, arrays=arrays)
