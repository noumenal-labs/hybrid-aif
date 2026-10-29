"""Workstream F tests: TMaze tensor parity vs pymdp + the generative process's sampling laws."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.envs import GenerativeEnvParams, broadcast_params, make
from hybrid_aif.reference.envs.tmaze import tmaze

pymdp_envs = pytest.importorskip("pymdp.envs")

KEY = jax.random.PRNGKey(0)


# ------------------------------------------------------- tmaze parity vs pymdp


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"reward_probability": 0.7, "punishment_probability": 0.6, "cue_validity": 0.8},
        {"reward_condition": 1},
        {"reward_probability": 0.98, "reward_condition": 0},
    ],
)
def test_tmaze_tensor_parity(kwargs):
    ref = pymdp_envs.TMaze(**kwargs)
    env, params = tmaze(**kwargs)
    assert env.A_deps == tuple(tuple(d) for d in ref.A_dependencies)
    assert env.B_deps == tuple(tuple(d) for d in ref.B_dependencies)
    for ours, theirs in zip(params.likelihood, ref.A, strict=True):
        assert np.allclose(np.asarray(ours), np.asarray(theirs), atol=1e-6)
    for ours, theirs in zip(params.transition, ref.B, strict=True):
        assert np.allclose(np.asarray(ours), np.asarray(theirs), atol=1e-6)
    for ours, theirs in zip(params.initial_state_prior, ref.D, strict=True):
        assert np.allclose(np.asarray(ours), np.asarray(theirs), atol=1e-6)


def test_tmaze_columns_normalized():
    _, params = tmaze(reward_probability=0.7, cue_validity=0.8)
    for a in params.likelihood:
        assert np.allclose(np.asarray(a).sum(axis=0), 1.0, atol=1e-6)
    for b in params.transition:
        assert np.allclose(np.asarray(b).sum(axis=0), 1.0, atol=1e-6)
    for d in params.initial_state_prior:
        assert np.allclose(np.asarray(d).sum(), 1.0, atol=1e-6)


# ------------------------------------------------- generative env: determinism


def _small_model(seed=0):
    """One 3-level factor, 2 actions; one 4-level modality."""
    rng = np.random.default_rng(seed)

    def norm(x, axis=0):
        return np.float32(x / x.sum(axis=axis, keepdims=True))

    A = [norm(rng.gamma(1.0, size=(4, 3)))]
    B = [norm(rng.gamma(1.0, size=(3, 3, 2)))]
    D = [norm(rng.gamma(1.0, size=3))]
    return make(A, B, D)


def test_reset_step_deterministic_given_key():
    env, params = _small_model()
    obs_a, state_a = env.reset(KEY, params)
    obs_b, state_b = env.reset(KEY, params)
    assert obs_a == obs_b and state_a == state_b
    action = jnp.array([1], jnp.int32)
    o1, s1 = env.step(jax.random.PRNGKey(7), state_a, action, params)
    o2, s2 = env.step(jax.random.PRNGKey(7), state_a, action, params)
    assert o1 == o2 and s1 == s2
    o3, s3 = jax.jit(env.step)(jax.random.PRNGKey(7), state_a, action, params)
    assert o1 == o3 and s1 == s3


# ------------------------------------- generative env: conditional sampling laws


def _joint_freq(obs, state, num_obs, num_states):
    joint = np.zeros((num_obs, num_states))
    np.add.at(joint, (np.asarray(obs), np.asarray(state)), 1.0)
    return joint / joint.sum()


def test_reset_samples_match_D_and_A():
    env, params = _small_model()
    n = 40_000
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(KEY, n), params)
    d = np.asarray(params.initial_state_prior[0])
    freq = np.bincount(np.asarray(state[0]), minlength=3) / n
    assert np.allclose(freq, d, atol=0.01)
    expected_joint = np.asarray(params.likelihood[0]) * d  # P(o, s) = A[o, s] D[s]
    assert np.allclose(_joint_freq(obs[0], state[0], 4, 3), expected_joint, atol=0.01)


def test_step_samples_match_B_and_A():
    env, params = _small_model()
    n = 40_000
    s0, a0 = 1, 1
    state = (jnp.asarray(s0, jnp.int32),)
    action = jnp.array([a0], jnp.int32)
    obs, state_next = jax.vmap(env.step, in_axes=(0, None, None, None))(
        jax.random.split(KEY, n), state, action, params
    )
    b_col = np.asarray(params.transition[0])[:, s0, a0]
    freq = np.bincount(np.asarray(state_next[0]), minlength=3) / n
    assert np.allclose(freq, b_col, atol=0.01)
    expected_joint = np.asarray(params.likelihood[0]) * b_col  # P(o, s') = A[o, s'] B[s' | s0, a0]
    assert np.allclose(_joint_freq(obs[0], state_next[0], 4, 3), expected_joint, atol=0.01)


def test_tmaze_reward_frequencies_at_arm():
    rp = 0.7
    env, params = tmaze(reward_probability=rp, reward_condition=0)
    n = 20_000
    state = (jnp.asarray(4, jnp.int32), jnp.asarray(0, jnp.int32))  # middle, context left
    action = jnp.array([1, 0], jnp.int32)  # move to left arm (correct)
    obs, state_next = jax.vmap(env.step, in_axes=(0, None, None, None))(
        jax.random.split(KEY, n), state, action, params
    )
    assert np.all(np.asarray(state_next[0]) == 1)  # deterministic move
    assert np.all(np.asarray(state_next[1]) == 0)  # context fixed
    reward_freq = np.bincount(np.asarray(obs[1]), minlength=3) / n
    assert np.allclose(reward_freq, [1 - rp, rp, 0.0], atol=0.01)


# ----------------------------------------------------------- jit + vmap batches


def test_jit_vmap_batch_of_envs_with_different_reward_probs():
    env, _ = tmaze()
    param_sets = [tmaze(reward_probability=rp)[1] for rp in (0.2, 0.5, 0.8, 1.0)]
    batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *param_sets)
    keys = jax.random.split(KEY, 4)
    obs, state = jax.jit(jax.vmap(env.reset))(keys, batched)
    assert all(o.shape == (4,) for o in obs) and all(s.shape == (4,) for s in state)
    actions = jnp.broadcast_to(jnp.array([3, 0], jnp.int32), (4, 2))
    obs2, state2 = jax.jit(jax.vmap(env.step))(keys, state, actions, batched)
    assert all(o.shape == (4,) for o in obs2)
    assert np.all(np.asarray(state2[0]) == 3)  # centre -> cue is deterministic


def test_broadcast_params_adds_batch_axis():
    _, params = tmaze()
    batched = broadcast_params(params, 3)
    assert isinstance(batched, GenerativeEnvParams)
    assert batched.likelihood[1].shape == (3, 3, 5, 2)
    assert np.allclose(np.asarray(batched.likelihood[1][2]), np.asarray(params.likelihood[1]))
