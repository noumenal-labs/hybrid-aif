"""A noise map on a modality that observes a mixture-carried factor.

``GridNoiseMap`` is read at the mean of each component of the mixture, the extended filter's
convention of a Gaussian-carried factor (``docs/design/planning-with-observations.md`` D3)
applied per component. The channel's categorical parents are action nodes alone, which select a
level of a levelled map before anything is read, or it has none; a map on a channel whose mode
parent is a state factor stays refused. Pinned by: a one-component mixture equal to the
Gaussian-carried read; a hand-written per-component Kalman update and mode evidence message; the
information-gain leaf's interval against a Monte Carlo estimate; the mixed and gradient planners;
the environment's draw; the description's round trip.
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
from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalGaussian,
    Gaussian,
    MixtureGaussian,
)
from hybrid_aif.factors import (
    GridNoiseMap,
    LinearGaussianObservation,
    LinearGaussianTransition,
)
from hybrid_aif.inference.linear_gaussian import condition
from hybrid_aif.objectives.continuous import state_info_gain
from hybrid_aif.planning.enumerate import evaluate_policies, evaluate_policies_total
from hybrid_aif.system import System, SystemParams
from tests.test_noise_map import _grid_map, _point_mass

KEY = jax.random.PRNGKey(0)
N = 5


def _position_map(scale: float) -> GridNoiseMap:
    """A grid over the two position coordinates, sharper toward +x, scaled by ``scale``."""
    sd = scale * (0.5 - 0.4 * np.linspace(0.0, 1.0, N))[:, None] * np.ones((1, N))
    values = np.zeros((N, N, 2, 2), np.float32)
    values[..., 0, 0] = sd
    values[..., 1, 1] = sd
    return GridNoiseMap(jnp.asarray(values), (-1.0, -1.0), (1.0, 1.0), axes=(0, 1))


LEVELS = (_position_map(1.0), _position_map(0.4))


def _levelled(maps=LEVELS) -> GridNoiseMap:
    first = maps[0]
    return GridNoiseMap(jnp.stack([m.values for m in maps]), first.low, first.high, axes=first.axes)


HEAD = jnp.zeros((2, 4)).at[0, 0].set(1.0).at[1, 1].set(1.0)
MEANS = jnp.array([[0.3, 0.0, 0.0, 0.0], [-0.3, 0.2, 0.0, 0.0]])


def _showcase(noise, *, policy_len=2, planner="enumeration", state_parent=False):
    """A 4-D position and velocity mixture switched by a terrain factor, a velocity control
    ``a`` and a two-level ``sensor`` node, a position fix whose noise ``noise`` reads."""
    system = System(
        A_deps=((1, 0) if state_parent else (1, 5),),
        B_deps=((0,), (2, 3, 0), (), ()),
        controls=(2, 3),
        action_names=("a", "sensor"),
        modality_names=("fix",),
        factor_names=("terrain", "x", "u", "s"),
        policy_len=policy_len,
        epistemic=True,
        mixture_k=2,
        mixture_schedule="mode_aligned",
        planning_belief="posterior",
        planner=planner,
    )
    F = jnp.eye(4).at[0, 2].set(0.25).at[1, 3].set(0.25)
    G = jnp.zeros((4, 2)).at[2, 0].set(0.25).at[3, 1].set(0.25)
    innovations = ConditionalGaussian(
        loc=jnp.zeros((2, 2, 4)),
        chol=jnp.broadcast_to(0.05 * jnp.eye(4), (2, 2, 4, 4)),
        parent_ndims=2,
    )
    params = SystemParams(
        likelihood={"fix": LinearGaussianObservation(H=HEAD, c=jnp.zeros(2), noise=noise)},
        transition={
            "terrain": ConditionalCategorical(
                log_table=jnp.log(jnp.array([[0.9, 0.1], [0.1, 0.9]])), parent_ndims=1
            ),
            "x": LinearGaussianTransition(F=F, innovations=innovations, G=G),
        },
        initial_state_prior={
            "terrain": Categorical(logits=jnp.zeros(2)),
            "x": MixtureGaussian(
                log_weights=jnp.log(jnp.array([0.6, 0.4])),
                means=MEANS,
                chols=jnp.broadcast_to(0.2 * jnp.eye(4), (2, 4, 4)),
            ),
        },
        control={"u": Box(-jnp.ones(2), jnp.ones(2))},
    )
    return system, params


def _sensor(level: int) -> Categorical:
    return Categorical.from_probs(jax.nn.one_hot(level, 2))


# ------------------------------------------------------------------ inference


def test_a_one_component_mixture_reads_the_map_as_the_gaussian_does():
    m = _grid_map()
    system, params = _point_mass(m)
    prior = Gaussian(loc=jnp.array([0.9, 0.1]), chol=0.2 * jnp.eye(2))
    mixture_system = dataclasses.replace(system, mixture_k=1, mixture_schedule="mode_aligned")
    mixture_params = dataclasses.replace(
        params, initial_state_prior={"x": MixtureGaussian.from_gaussian(prior)}
    )
    y = jnp.array([0.95, 0.0])
    want, ref = system.infer_states(params, {"fix": y}, {"x": prior}, detail=True)
    got, detail = mixture_system.infer_states(
        mixture_params, {"fix": y}, {"x": MixtureGaussian.from_gaussian(prior)}, detail=True
    )
    posterior = got["x"]
    np.testing.assert_allclose(np.asarray(posterior.means[0]), np.asarray(want["x"].loc), atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(posterior.chols[0]), np.asarray(want["x"].chol), atol=1e-6
    )
    assert float(detail.log_evidence["fix"]) == pytest.approx(
        float(ref.log_evidence["fix"]), abs=1e-5
    )
    #  and the information-gain leaf: the pair of a one-component mixture is the Gaussian's value
    model = params.likelihood["fix"]
    gain = state_info_gain((prior,), (model,), ("fix",))["infogain"]["fix"]
    mixed = state_info_gain((MixtureGaussian.from_gaussian(prior),), (model,), ("fix",))[
        "infogain"
    ]["fix"]
    np.testing.assert_allclose(np.asarray(mixed.lower), np.asarray(gain.lower), atol=1e-5)
    np.testing.assert_allclose(np.asarray(mixed.upper), np.asarray(gain.upper), atol=1e-5)


@pytest.mark.parametrize("level", [0, 1])
def test_each_component_takes_the_kalman_update_at_its_own_noise(level):
    system, params = _showcase(_levelled())
    beliefs = system.initial_priors(params)
    y = jnp.array([0.2, 0.1])
    posterior, detail = system.infer_states(
        params, {"fix": y}, {**beliefs, "sensor": _sensor(level)}, detail=True
    )
    prior = beliefs["x"]
    chosen = LEVELS[level]
    noises = [chosen(prior.means[k]) for k in range(2)]
    assert not np.allclose(np.asarray(noises[0]), np.asarray(noises[1]))  # read per component
    log_marginals = []
    for k in range(2):
        want, log_marginal = condition(
            Gaussian(loc=prior.means[k], chol=prior.chols[k]), HEAD, jnp.zeros(2), noises[k], y
        )
        np.testing.assert_allclose(
            np.asarray(posterior["x"].means[k]), np.asarray(want.loc), atol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(posterior["x"].chols[k]), np.asarray(want.chol), atol=1e-5
        )
        log_marginals.append(float(log_marginal))
    joint = np.asarray(jax.nn.log_softmax(prior.log_weights)) + np.asarray(log_marginals)
    evidence = float(jax.scipy.special.logsumexp(joint))
    np.testing.assert_allclose(np.asarray(posterior["x"].log_weights), joint - evidence, atol=1e-5)
    #  one bank per terrain level: the mode message is the prior times each bank's evidence
    mode = np.asarray(beliefs["terrain"].log_probs) + np.asarray(log_marginals)
    np.testing.assert_allclose(
        np.asarray(posterior["terrain"].probs), np.asarray(jax.nn.softmax(mode)), atol=1e-5
    )
    assert float(detail.log_evidence["fix"]) == pytest.approx(
        float(
            jax.scipy.special.logsumexp(np.asarray(beliefs["terrain"].log_probs) + log_marginals)
        ),
        abs=1e-5,
    )


def test_a_map_without_nodes_on_a_mixture_is_read_per_component():
    system, params = _showcase(LEVELS[0])
    system = dataclasses.replace(system, A_deps=((1,),))
    system = dataclasses.replace(system, controls=(2, 3))
    beliefs = system.initial_priors(params)
    y = jnp.array([0.2, 0.1])
    posterior, _ = system.infer_states(params, {"fix": y}, beliefs)
    prior = beliefs["x"]
    for k in range(2):
        want, _ = condition(
            Gaussian(loc=prior.means[k], chol=prior.chols[k]),
            HEAD,
            jnp.zeros(2),
            LEVELS[0](prior.means[k]),
            y,
        )
        np.testing.assert_allclose(
            np.asarray(posterior["x"].means[k]), np.asarray(want.loc), atol=1e-5
        )


def test_a_map_on_a_state_mode_parent_is_still_refused():
    system, params = _showcase(_levelled(), state_parent=True)
    with pytest.raises(NotImplementedError, match="mode parents are all action nodes"):
        system.planner_context(params, system.initial_priors(params))


def test_a_soft_node_prior_is_refused_for_a_selecting_channel_on_a_mixture():
    system, params = _showcase(_levelled())
    beliefs = system.initial_priors(params)
    spread = Categorical.from_probs(jnp.array([0.3, 0.7]))
    with pytest.raises(ValueError, match="map over the mixture-carried factor 'x'.*is a point"):
        system.infer_states(params, {"fix": jnp.array([0.2, 0.1])}, {**beliefs, "sensor": spread})


# ------------------------------------------------------------------ planning


def _step(ctx, plan_step, level):
    """One plan step's carry and the step terms, from the initial belief."""
    action = ControlStep(entries=(plan_step, jnp.asarray(level, jnp.int32)))
    carry = ctx.transition_fn(ctx.init_beliefs, action)
    return carry, ctx.step_terms_fn(carry, action, jnp.asarray(1, jnp.int32))


def _predictive_samples(q: MixtureGaussian, noises: jax.Array, n: int, key):
    """Draws of ``y = H x + eps`` and its mixture log-density, the component's noise ``R_k``
    the one read at that component's mean."""
    k_comp, k_x, k_e = jax.random.split(key, 3)
    w = jax.nn.softmax(q.log_weights)
    comp = jax.random.choice(k_comp, w.shape[0], (n,), p=w)
    x = q.means[comp] + jnp.einsum(
        "nij,nj->ni", q.chols[comp], jax.random.normal(k_x, (n, q.means.shape[-1]))
    )
    y = x @ HEAD.T + jnp.einsum("nij,nj->ni", noises[comp], jax.random.normal(k_e, (n, 2)))
    S = jnp.einsum("ij,kjl,ml->kim", HEAD, q.chols @ jnp.swapaxes(q.chols, -1, -2), HEAD)
    S = S + noises @ jnp.swapaxes(noises, -1, -2)
    mu = q.means @ HEAD.T
    logp = jax.vmap(
        lambda m, s: jax.scipy.stats.multivariate_normal.logpdf(y, m, s), in_axes=(0, 0)
    )(mu, S)
    return -jax.scipy.special.logsumexp(
        q.log_weights[:, None] + logp, axis=0
    ) + jax.scipy.special.logsumexp(q.log_weights)


@pytest.mark.parametrize("level", [0, 1])
def test_the_information_gain_interval_contains_the_monte_carlo_gain(level):
    system, params = _showcase(_levelled(), policy_len=1)
    ctx = system.planner_context(params, system.initial_priors(params))
    carry, tree = _step(ctx, jnp.array([0.4, -0.2]), level)
    leaf = tree["infogain"]["fix"]
    q = carry[1]
    q = q.predictive if hasattr(q, "predictive") else q
    noises = LEVELS[level](q.means)
    h_y = _predictive_samples(q, noises, 20000, jax.random.PRNGKey(3))
    h_cond = float(
        jnp.sum(
            jax.nn.softmax(q.log_weights)
            * (
                jnp.log(jnp.diagonal(noises, axis1=-2, axis2=-1)).sum(-1)
                + 1.0
                + jnp.log(2 * jnp.pi)
            )
        )
    )
    estimate = float(h_y.mean()) - h_cond
    se = float(h_y.std()) / np.sqrt(h_y.shape[0])
    lower, upper = float(jnp.ravel(leaf.lower)[0]), float(jnp.ravel(leaf.upper)[0])
    assert lower - 3 * se <= estimate <= upper + 3 * se


def _policy(plan, level):
    horizon = plan.shape[0]
    return ControlStep(entries=(plan[None], jnp.full((1, horizon), level, jnp.int32)))


@pytest.mark.parametrize("level", [0, 1])
def test_the_gradient_through_the_map_on_a_mixture_is_finite_and_matches_differences(level):
    system, params = _showcase(_levelled())
    ctx = system.planner_context(params, system.initial_priors(params))

    def score(plan):
        return evaluate_policies_total(
            _policy(plan, level),
            ctx.init_beliefs,
            ctx.transition_fn,
            ctx.step_terms_fn,
            ctx.terminal_terms_fn,
        ).midpoint[0]

    plan = jnp.array([[0.8, 0.3], [0.6, -0.2]])
    grad = np.asarray(jax.grad(score)(plan))
    assert np.isfinite(grad).all() and np.abs(grad).max() > 0.0
    eps = 1e-3
    live = [tuple(int(v) for v in ij) for ij in np.argwhere(np.abs(grad) > 1e-3)]
    assert live  # an entry whose gradient is not zero, so the comparison is not vacuous
    for i, j in (*live, (1, 1)):
        bump = jnp.zeros_like(plan).at[i, j].set(eps)
        fd = (float(score(plan + bump)) - float(score(plan - bump))) / (2 * eps)
        assert grad[i, j] == pytest.approx(fd, rel=5e-2, abs=5e-3)


def test_the_levels_score_differently_and_the_mixed_planner_runs():
    system, params = _showcase(_levelled())
    ctx = system.planner_context(params, system.initial_priors(params))
    plan = jnp.array([[0.8, 0.3], [0.6, -0.2]])
    leaves = [
        evaluate_policies(
            _policy(plan, level),
            ctx.init_beliefs,
            ctx.transition_fn,
            ctx.step_terms_fn,
            ctx.terminal_terms_fn,
        )["infogain"]["fix"].lower
        for level in (0, 1)
    ]
    assert not np.allclose(np.asarray(leaves[0]), np.asarray(leaves[1]))
    mixed = dataclasses.replace(system, planner="mixed", num_iters=5)
    q_pi, tree = mixed.infer_policies(params, mixed.initial_priors(params))
    assert np.isfinite(np.asarray(q_pi.probs)).all()
    assert "fix" in tree["infogain"]


# ------------------------------------------------------------------ the world and the description


@pytest.mark.parametrize("level", [0, 1])
def test_the_environment_draws_at_the_true_state_and_the_realised_level(level):
    system, params = _showcase(_levelled())
    env, eparams = system.as_env(params)
    _, state = env.reset(KEY, eparams)
    keys = jax.random.split(jax.random.PRNGKey(1), 600)
    actions = {"a": jnp.zeros(2), "sensor": jnp.asarray(level)}
    draws = jax.vmap(lambda k: env.step_actuated(k, state, actions, eparams)[0]["fix"])(keys)
    sd_at = float(np.asarray(LEVELS[level](jnp.asarray(state["x"])[:2])[0, 0]))
    spread = float(np.asarray(draws)[:, 1].std())
    assert spread == pytest.approx(np.hypot(sd_at, 0.05), rel=0.2)


def test_the_description_round_trips_and_prices_the_levels():
    system, params = _showcase(_levelled())
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
    one_system, one_params = _showcase(LEVELS[0])
    two = description_length(system, 100, params=params)
    one = description_length(one_system, 100, params=one_params)
    assert two.total > one.total  # a second level's grid is a second set of parameters


# ------------------------------------------- selection and switching agree on a constant map


def _constant_map_and_array():
    """The same per-level noise as a constant levelled map (selection) and as an array
    (switching over the node's levels)."""
    sds = (0.3, 0.1)
    array = jnp.stack([r * jnp.eye(2) for r in sds])
    values = jnp.stack([jnp.broadcast_to(r * jnp.eye(2), (N, N, 2, 2)) for r in sds])
    return GridNoiseMap(values, (-1.0, -1.0), (1.0, 1.0), axes=(0, 1)), array


@pytest.mark.parametrize("level", [0, 1])
def test_selection_and_switching_agree_on_the_factors_mode_parent_and_the_evidence(level):
    cmap, array = _constant_map_and_array()
    y = jnp.array([0.2, 0.1])
    results = []
    for noise in (cmap, array):
        system, params = _showcase(noise)
        beliefs = system.initial_priors(params)
        results.append(
            system.infer_states(
                params, {"fix": y}, {**beliefs, "sensor": _sensor(level)}, detail=True
            )
        )
    (by_map, d_map), (by_array, d_array) = results
    np.testing.assert_allclose(by_array["terrain"].probs, by_map["terrain"].probs, atol=1e-6)
    np.testing.assert_allclose(d_array.log_evidence["fix"], d_map.log_evidence["fix"], atol=1e-5)
    np.testing.assert_allclose(by_array["x"].mean(), by_map["x"].mean(), atol=1e-5)
    np.testing.assert_allclose(by_array["x"].cov(), by_map["x"].cov(), atol=1e-5)
