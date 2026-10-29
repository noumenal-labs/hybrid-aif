"""Workstream F tests: scan rollout shapes, masked windows, jit/vmap batching."""

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_aif.envs import broadcast_params
from hybrid_aif.reference.envs.tmaze import tmaze
from hybrid_aif.rollout import random_policy, rollout

KEY = jax.random.PRNGKey(0)
NUM_ACTIONS = (5, 1)  # tmaze: location controlled, context no-op


def _run(num_steps=7, batch_size=2, window_len=4, key=KEY, **tmaze_kwargs):
    env, params = tmaze(**tmaze_kwargs)
    batched = broadcast_params(params, batch_size)
    policy = random_policy(NUM_ACTIONS)
    return rollout(key, env, batched, policy, num_steps, batch_size, window_len=window_len)


def test_trajectory_shapes():
    B, T, W = 2, 7, 4
    traj = _run(num_steps=T, batch_size=B, window_len=W)
    assert len(traj.observations) == 3 and len(traj.states) == 2
    assert all(o.shape == (B, T + 1) for o in traj.observations)
    assert all(s.shape == (B, T + 1) for s in traj.states)
    assert traj.actions.shape == (B, T, 2)
    assert traj.history.valid.shape == (B, T, W)
    assert all(o.shape == (B, T, W) for o in traj.history.obs)
    assert traj.history.actions.shape == (B, T, W, 2)


def test_validity_mask_early_steps_masked_and_right_aligned():
    B, T, W = 2, 7, 4
    traj = _run(num_steps=T, batch_size=B, window_len=W)
    valid = np.asarray(traj.history.valid)
    for t in range(T):
        expected = np.arange(W) >= W - min(t + 1, W)
        assert np.all(valid[:, t] == expected)


def test_window_slides_right_and_sentinels():
    B, T, W = 2, 9, 4
    traj = _run(num_steps=T, batch_size=B, window_len=W)
    valid = np.asarray(traj.history.valid)
    for m, hist_obs in enumerate(traj.history.obs):
        hist_obs = np.asarray(hist_obs)
        obs = np.asarray(traj.observations[m])
        for t in range(T):
            for i in range(W):
                src = t - (W - 1 - i)  # window slot i holds obs at this time
                if src >= 0:
                    assert hist_obs[:, t, i].tolist() == obs[:, src].tolist()
                else:
                    assert np.all(hist_obs[:, t, i] == -1)
                    assert not valid[:, t, i].any()
    # actions[i] is the action that produced obs[i]; the reset slot stays -1
    hist_act = np.asarray(traj.history.actions)
    act = np.asarray(traj.actions)
    for t in range(1, T):
        assert np.all(hist_act[:, t, -1] == act[:, t - 1])
    assert np.all(hist_act[:, 0] == -1)


def test_whole_rollout_jits_and_is_deterministic():
    env, params = tmaze(reward_probability=0.9)
    batched = broadcast_params(params, 3)
    policy = random_policy(NUM_ACTIONS)

    def fn(k):
        return rollout(k, env, batched, policy, num_steps=6, batch_size=3, window_len=5)

    eager_leaves = jax.tree_util.tree_leaves(fn(KEY))
    jitted_leaves = jax.tree_util.tree_leaves(jax.jit(fn)(KEY))
    for a, b in zip(eager_leaves, jitted_leaves, strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_vmap_batch_of_4_envs_with_different_reward_probs():
    env, _ = tmaze()
    param_sets = [tmaze(reward_probability=rp)[1] for rp in (0.2, 0.5, 0.8, 1.0)]
    batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *param_sets)
    policy = random_policy(NUM_ACTIONS)
    traj = jax.jit(
        lambda k: rollout(k, env, batched, policy, num_steps=10, batch_size=4, window_len=6)
    )(KEY)
    assert traj.actions.shape == (4, 10, 2)
    # context factor is fixed by B_ctx = I: constant within each rollout
    ctx = np.asarray(traj.states[1])
    assert np.all(ctx == ctx[:, :1])
    # location transitions obey adjacent connectivity: centre never jumps to an arm
    loc = np.asarray(traj.states[0])
    from_centre = loc[:, :-1] == 0
    assert np.all(np.isin(loc[:, 1:][from_centre], [0, 3, 4]))


def test_random_policy_actions_in_range():
    traj = _run(num_steps=12, batch_size=3, window_len=4)
    act = np.asarray(traj.actions)
    for f, n in enumerate(NUM_ACTIONS):
        assert act[..., f].min() >= 0 and act[..., f].max() < n


def test_different_keys_give_different_trajectories():
    t1 = _run(key=jax.random.PRNGKey(1), num_steps=10)
    t2 = _run(key=jax.random.PRNGKey(2), num_steps=10)
    assert not np.array_equal(np.asarray(t1.actions), np.asarray(t2.actions))


def test_equal_random_policies_share_one_compiled_rollout(monkeypatch):
    import hybrid_aif.rollout as rollout_module

    traces = []
    single = rollout_module._rollout_single

    def counting(*args, **kwargs):
        traces.append(1)
        return single(*args, **kwargs)

    monkeypatch.setattr(rollout_module, "_rollout_single", counting)
    assert random_policy(NUM_ACTIONS) == random_policy(list(NUM_ACTIONS))
    _run(num_steps=5)  # a fresh random_policy per call, as a loop over seeds would build
    _run(num_steps=5, key=jax.random.PRNGKey(3))
    assert len(traces) == 1
