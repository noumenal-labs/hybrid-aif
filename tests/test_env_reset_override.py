"""Acceptance-fix 1: env reset point-state override (the D conflation).

``D`` is both the agent's initial prior and the env's initial-state
distribution; "believe N(0, 0.04 I) but truly start at exactly (0, 0)" was
inexpressible. ``reset(key, params, state=...)`` supplies a point start (the
env holds a point state anyway — no delta distributions) while observations
are still sampled from it; ``state=None`` keeps the sample-from-D behavior
bitwise.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import Categorical, Gaussian
from hybrid_aif.reference.envs.tmaze import CENTRE, CUE, tmaze
from tests.test_hybrid_lg_system import R_CHOL, C, H
from tests.test_hybrid_lg_system import make_params as hill_params
from tests.test_hybrid_lg_system import make_system as hill_system
from tests.test_hybrid_switching_system import make_params as switching_params
from tests.test_hybrid_switching_system import make_system as switching_system

KEY = jax.random.PRNGKey(11)


# ------------------------------------------------------------------- discrete env


def test_discrete_override_state_honored_and_observed_from_it():
    env, params = tmaze(cue_validity=1.0)
    obs, state = env.reset(KEY, params, state=(jnp.asarray(CUE), jnp.asarray(1)))
    assert int(state[0]) == CUE and int(state[1]) == 1
    assert state[0].dtype == jnp.int32 and state[1].dtype == jnp.int32
    assert int(obs[0]) == CUE  # location modality is the identity
    assert int(obs[2]) == 2  # perfectly valid cue reveals context 1 at the cue location


def test_discrete_override_expresses_prior_vs_true_start():
    # believe 50/50 context (D untouched) but truly start in context 0, off-centre
    env, params = tmaze()
    np.testing.assert_allclose(np.asarray(params.initial_state_prior[1]), [0.5, 0.5])
    _, state = env.reset(KEY, params, state=(jnp.asarray(CENTRE), jnp.asarray(0)))
    assert int(state[1]) == 0


def test_discrete_none_path_bitwise():
    """Without an override, reset must reproduce the pre-change sampling exactly:
    split(key, num_factors + 1), factor draws from keys[1:], observations from
    keys[0] split per modality."""
    env, params = tmaze()
    obs, state = env.reset(KEY, params)
    keys = jax.random.split(KEY, env.num_factors + 1)
    state_ref = tuple(
        Categorical.from_probs(d).sample(k).astype(jnp.int32)
        for k, d in zip(list(keys[1:]), params.initial_state_prior, strict=True)
    )
    obs_keys = jax.random.split(keys[0], env.num_modalities)
    obs_ref = tuple(
        Categorical.from_probs(a[(..., *[state_ref[g] for g in deps])]).sample(k).astype(jnp.int32)
        for k, a, deps in zip(list(obs_keys), params.likelihood, env.A_deps, strict=True)
    )
    for got, ref in zip(state + obs, state_ref + obs_ref, strict=True):
        assert jnp.array_equal(got, ref)


def test_discrete_override_wrong_length_raises():
    env, params = tmaze()
    with pytest.raises(ValueError, match="factors"):
        env.reset(KEY, params, state=(jnp.asarray(CENTRE),))


# --------------------------------------------------------------------- hybrid env


def test_hybrid_override_state_honored_and_observed_from_it():
    """Wide Gaussian belief in D, exact point start: the finding's motivating case."""
    system, params = hill_system(), hill_params()
    env, env_params = system.as_env(params)
    start = {"state": jnp.zeros(2)}
    obs, state = env.reset(KEY, env_params, state=start)
    np.testing.assert_array_equal(np.asarray(state["state"]), np.zeros(2))  # exact, not sampled
    # the observation IS sampled from the point state: y = H x + c + eps, small R
    loc = np.asarray((H @ state["state"]) + C)
    assert abs(float(obs["position"][0]) - loc[0]) < 10 * float(R_CHOL[0, 0])
    # pure: same key reproduces; a different key redraws the observation noise
    obs_b, _ = env.reset(KEY, env_params, state=start)
    np.testing.assert_array_equal(np.asarray(obs["position"]), np.asarray(obs_b["position"]))
    obs_c, _ = env.reset(jax.random.PRNGKey(12), env_params, state=start)
    assert float(obs_c["position"][0]) != float(obs["position"][0])


def test_hybrid_override_types_per_factor():
    system, params = switching_system(), switching_params()
    env, env_params = system.as_env(params)
    obs, state = env.reset(KEY, env_params, state={"mode": 1, "x": jnp.asarray([0.3])})
    assert obs == {}  # no modalities declared
    assert state["mode"].dtype == jnp.int32 and int(state["mode"]) == 1
    assert jnp.issubdtype(state["x"].dtype, jnp.floating)
    np.testing.assert_allclose(np.asarray(state["x"]), [0.3])


def test_hybrid_none_path_bitwise():
    system, params = switching_system(), switching_params()
    env, env_params = system.as_env(params)
    _, state = env.reset(KEY, env_params)
    keys = jax.random.split(KEY, env.num_factors + 1)
    priors = [env_params.initial_state_prior[n] for n in system.names.states]  # node order
    state_ref = tuple(d.sample(k) for k, d in zip(list(keys[1:]), priors, strict=True))
    for name, ref in zip(system.names.states, state_ref, strict=True):
        assert jnp.array_equal(state[name], ref)


def test_hybrid_none_path_observation_bitwise():
    system, params = hill_system(), hill_params()
    env, env_params = system.as_env(params)
    obs, state = env.reset(KEY, env_params)
    keys = jax.random.split(KEY, env.num_factors + 1)
    (k_obs,) = tuple(jax.random.split(keys[0], env.num_modalities))
    ref = Gaussian(loc=(H @ state["state"]) + C, chol=R_CHOL).sample(k_obs)
    assert jnp.array_equal(obs["position"], ref)


def test_hybrid_override_missing_factor_raises():
    system, params = switching_system(), switching_params()
    env, env_params = system.as_env(params)
    with pytest.raises(ValueError, match=r"state is missing \['mode'\]"):
        env.reset(KEY, env_params, state={"x": jnp.asarray([0.3])})
