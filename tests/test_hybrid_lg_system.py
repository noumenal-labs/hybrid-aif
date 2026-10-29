"""Hill-class fixture: one LG factor + one LG modality as a System (design D2/D3).

Mirrors ``examples/hill_lg.py``'s model: 2-D double-integrator state, discrete
thrust set realized as a control-conditioned :class:`ConditionalGaussian`
innovation family over shared linear dynamics, noisy position-only LG
observation. Asserts the System's closures are exactly the existing kernels:
``transition_fn`` == ``predict`` through the selected mode, ``infer_states``
== ``condition``, ``update_empirical_prior`` == the demo's ``predicted``; and
``act`` runs end to end deterministically.
"""

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_aif.distributions import ConditionalGaussian, Gaussian
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianTransition
from hybrid_aif.inference.linear_gaussian import condition, predict
from hybrid_aif.system import System, SystemParams

DT, SLOPE, GOAL_POS = 0.1, 0.5, 1.0
THRUSTS = (-1.0, 0.0, 1.0)

F = jnp.array([[1.0, DT], [SLOPE * DT, 1.0]])
B_LOC = jnp.stack([jnp.array([0.0, DT * (u - SLOPE * GOAL_POS)]) for u in THRUSTS])
Q_CHOL = jnp.stack([jnp.diag(jnp.array([1e-3, 5e-3 + 5e-3 * abs(u)])) for u in THRUSTS])
INNOVATIONS = ConditionalGaussian(loc=B_LOC, chol=Q_CHOL, parent_ndims=1)
H = jnp.array([[1.0, 0.0]])
C = jnp.zeros(1)
R_CHOL = jnp.array([[0.05]])
GOAL = Gaussian(loc=jnp.array([GOAL_POS, 0.0]), chol=jnp.diag(jnp.array([0.08, 0.25])))


def make_system(**overrides):
    defaults = dict(
        A_deps=((0,),),
        B_deps=((),),
        controlled=(True,),
        policy_len=4,
        modality_names=("position",),
        factor_names=("state",),
    )
    return System(**{**defaults, **overrides})


def make_params():
    return SystemParams(
        likelihood={"position": LinearGaussianObservation(H=H, c=C, noise=R_CHOL)},
        transition={"state": LinearGaussianTransition(F=F, innovations=INNOVATIONS)},
        initial_state_prior={"state": Gaussian(loc=jnp.zeros(2), chol=0.2 * jnp.eye(2))},
        obs_targets={},
        state_targets={"state": GOAL},
        terminal_targets={},
    )


def test_transition_closure_matches_predict():
    system, params = make_system(), make_params()
    belief = Gaussian(loc=jnp.array([0.3, -0.1]), chol=0.15 * jnp.eye(2))
    ctx = system.planner_context(params, {"state": belief})
    for u in range(3):
        (pushed,) = ctx.transition_fn((belief,), jnp.asarray([u], jnp.int32))
        innov = INNOVATIONS.condition_on((u,))
        ref = predict(belief, F, innov.loc, innov.chol)
        np.testing.assert_allclose(np.asarray(pushed.loc), np.asarray(ref.loc), atol=1e-6)
        np.testing.assert_allclose(np.asarray(pushed.chol), np.asarray(ref.chol), atol=1e-6)


def test_infer_states_matches_condition():
    system, params = make_system(), make_params()
    prior = params.initial_state_prior["state"]
    y = jnp.asarray([0.07])
    posterior = system.infer_states(params, {"position": y}, priors={"state": prior})[0]["state"]
    ref, _ = condition(prior, H, C, R_CHOL, y)
    np.testing.assert_allclose(np.asarray(posterior.loc), np.asarray(ref.loc), atol=1e-6)
    np.testing.assert_allclose(np.asarray(posterior.chol), np.asarray(ref.chol), atol=1e-6)
    # priors=None defaults to D
    from_d = system.infer_states(params, {"position": y})[0]["state"]
    np.testing.assert_allclose(np.asarray(from_d.loc), np.asarray(ref.loc), atol=1e-6)


def test_update_empirical_prior_matches_predict():
    system, params = make_system(), make_params()
    belief = Gaussian(loc=jnp.array([0.5, 0.2]), chol=0.1 * jnp.eye(2))
    predicted = system.update_empirical_prior(
        params, {"state": belief}, {"a_state": jnp.asarray(2, jnp.int32)}
    )["state"]
    innov = INNOVATIONS.condition_on((2,))
    ref = predict(belief, F, innov.loc, innov.chol)
    np.testing.assert_allclose(np.asarray(predicted.loc), np.asarray(ref.loc), atol=1e-6)
    np.testing.assert_allclose(np.asarray(predicted.chol), np.asarray(ref.chol), atol=1e-6)


def test_act_end_to_end_deterministic():
    system, params = make_system(), make_params()
    beliefs, _ = system.infer_states(params, {"position": jnp.asarray([0.0])})
    a1, q_pi, tree = system.act(
        jax.random.PRNGKey(0), params, beliefs, selection="deterministic", mode="policy"
    )
    a2, _, _ = system.act(
        jax.random.PRNGKey(99), params, beliefs, selection="deterministic", mode="policy"
    )
    assert jnp.array_equal(a1["a_state"], a2["a_state"]) and a1["a_state"].dtype == jnp.int32
    assert int(a1["a_state"]) == 2  # below the hilltop at rest: full thrust toward the goal
    assert np.isclose(float(q_pi.probs.sum()), 1.0, atol=1e-5)
    assert set(tree) == {"risk", "infogain"}
    # jit the whole planning face
    f = jax.jit(lambda p, b: system.infer_policies(p, b)[0].logits)
    np.testing.assert_allclose(np.asarray(f(params, beliefs)), np.asarray(q_pi.logits), atol=1e-5)


def test_channel_leaf_and_uncertainty_row_present_and_togglable():
    """The position channel carries the information-gain leaf (V11: channel-keyed on the
    Gaussian path too), toggled by ``epistemic``; the belief's entropy is the opt-in
    ``uncertainty`` row, toggled by ``terms``."""
    system, params = make_system(), make_params()
    beliefs = {"state": params.initial_state_prior["state"]}
    _, tree = system.infer_policies(params, beliefs)
    assert set(tree["infogain"]) == {"position"} and "uncertainty" not in tree
    _, tree_off = make_system(epistemic=False).infer_policies(params, beliefs)
    assert "infogain" not in tree_off
    _, tree_row = make_system(terms=("uncertainty:state",)).infer_policies(params, beliefs)
    assert set(tree_row["uncertainty"]) == {"state"}


def test_terminal_target_scores_horizon_end():
    system, params = make_system(), make_params()
    import dataclasses

    params = dataclasses.replace(params, terminal_targets={"state": GOAL})
    _, tree = system.infer_policies(params, {"state": params.initial_state_prior["state"]})
    assert set(tree["terminal_risk"]) == {"state"}
    assert tree["terminal_risk"]["state"].lower.shape == (system.policies(params).shape[0],)


def test_as_env_runs_the_generative_process():
    system, params = make_system(), make_params()
    env, env_params = system.as_env(params)
    key = jax.random.PRNGKey(5)
    obs, state = env.reset(key, env_params)
    assert state["state"].shape == (2,) and obs["position"].shape == (1,)
    action = {"a_state": jnp.asarray(2, jnp.int32)}
    obs2, state2 = env.step(jax.random.PRNGKey(6), state, action, env_params)
    # the step is x' = F x + b_u + eps: subtracting the deterministic part leaves
    # noise on the innovation scale
    resid = np.asarray(state2["state"] - ((F @ state["state"]) + B_LOC[2]))
    assert np.all(np.abs(resid) < 0.1)
    # observation is y = H x' + c + eps on the R scale
    assert abs(float(obs2["position"][0] - state2["state"][0])) < 0.5
    # pure: same key reproduces
    obs2b, state2b = env.step(jax.random.PRNGKey(6), state, action, env_params)
    np.testing.assert_allclose(np.asarray(state2["state"]), np.asarray(state2b["state"]))


def test_closed_loop_tick_reaches_uphill():
    """Filter -> plan -> act -> predict, a few ticks: the true state moves toward
    the hilltop, exercising every hybrid face together. The prior is tightened so
    the env starts at rest in the valley (the demo's setup) and the run is short."""
    import dataclasses

    system, params = make_system(), make_params()
    params = dataclasses.replace(
        params, initial_state_prior={"state": Gaussian(loc=jnp.zeros(2), chol=1e-3 * jnp.eye(2))}
    )
    env, env_params = system.as_env(params)
    key = jax.random.PRNGKey(7)
    k_reset, key = jax.random.split(key)
    obs, state = env.reset(k_reset, env_params)
    priors = None
    for _ in range(15):
        beliefs, _ = system.infer_states(params, obs, priors)
        k_act, k_env, key = tuple(jax.random.split(key, 3))
        action, _, _ = system.act(
            k_act, params, beliefs, selection="deterministic", mode="policy", metrics=False
        )
        priors = system.update_empirical_prior(params, beliefs, action)
        obs, state = env.step(k_env, state, action, env_params)
    assert float(state["state"][0]) > 0.1  # moved off the valley floor toward p=1
