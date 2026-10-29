"""R-AIF's Mountain Car, pinned to the addendum's spec (§1).

The env is a *replication target*, so its tests read like a transcription check:
the physics against hand-computed steps, the sparse reward's three cases (their
``ActionRepeat(2)`` makes a non-terminating agent step worth −2 and the floor
−200, which is how the addendum reconciles their −68.3 with ≈34 agent steps), the
``easy_mode`` termination predicate, and the identity observation P1 needs. Every
number here comes from ``docs/design/amortization-p0-addendum.md`` §1, not from
what the implementation happens to do.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.reference.envs.mountain_car import MountainCar

KEY = jax.random.PRNGKey(0)


def _state(position: float, velocity: float):
    return (jnp.asarray([position, velocity], jnp.float32),)


# ------------------------------------------------------------------------ physics


def test_physics_matches_hand_computed_gymnasium_steps():
    """``v += a·power − gravity·cos(3p)``; ``p += v`` — with their power 0.0015 and
    gravity 0.0025 (addendum §1: gymnasium's MountainCarContinuous dynamics)."""
    env = MountainCar()
    for position, velocity, action in [(-0.5, 0.0, 1.0), (-0.9, 0.03, -1.0), (0.2, -0.02, 0.4)]:
        want_v = velocity + action * 0.0015 - 0.0025 * np.cos(3.0 * position)
        want_v = float(np.clip(want_v, -0.07, 0.07))
        want_p = float(np.clip(position + want_v, -1.2, 0.6))
        got = np.asarray(env.physics_step(_state(position, velocity)[0], jnp.asarray([action])))
        np.testing.assert_allclose(got, [want_p, want_v], atol=1e-6)


def test_action_is_clipped_speed_is_capped_and_the_left_wall_is_inelastic():
    env = MountainCar()
    hard = env.physics_step(_state(-0.5, 0.0)[0], jnp.asarray([5.0]))
    clipped = env.physics_step(_state(-0.5, 0.0)[0], jnp.asarray([1.0]))
    np.testing.assert_allclose(np.asarray(hard), np.asarray(clipped), atol=0.0)

    fast = env.physics_step(_state(-0.9, 0.07)[0], jnp.asarray([1.0]))
    assert float(fast[1]) == pytest.approx(env.max_speed, abs=1e-6)  # capped, not exceeded

    wall = env.physics_step(_state(-1.199, -0.05)[0], jnp.asarray([-1.0]))
    assert float(wall[0]) == pytest.approx(env.min_position)
    assert float(wall[1]) == 0.0  # inward velocity is killed, not reflected


def test_reset_draws_their_start_distribution():
    """``position ~ U(-0.6, -0.4)``, ``velocity = 0``; the observation is the state."""
    env = MountainCar()
    keys = jax.random.split(KEY, 200)
    starts = np.stack([np.asarray(env.reset(k)[1][0]) for k in keys])
    assert starts[:, 0].min() >= -0.6 and starts[:, 0].max() <= -0.4
    assert starts[:, 0].std() > 0.03  # genuinely random, not a constant
    np.testing.assert_array_equal(starts[:, 1], 0.0)
    obs, state = env.reset(KEY)
    np.testing.assert_array_equal(np.asarray(obs[0]), np.asarray(state[0]))


# --------------------------------------------------------- reward and termination


def test_termination_is_position_only_easy_mode():
    """``easy_mode=True``: the flag is ``position >= 0.45``, velocity ignored —
    so approaching it *backwards* still terminates (addendum §1, mtc.py:122)."""
    env = MountainCar()
    assert bool(env.reached_goal(_state(0.45, 0.0)))
    assert bool(env.reached_goal(_state(0.5, -0.06)))  # velocity ignored
    assert not bool(env.reached_goal(_state(0.4499, 0.07)))


def test_sparse_reward_has_exactly_three_cases_per_agent_step():
    """``-1`` per raw step, ``0`` on the terminating step, ``ActionRepeat(2)``
    summing and breaking: −2 for a plain agent step, −1 when the *second* raw step
    reaches the flag, 0 when the *first* does (addendum §1 + its "Consequences")."""
    env = MountainCar()
    _, _, plain, done = env.agent_step(None, _state(-0.5, 0.0), jnp.asarray([1.0]))
    assert float(plain) == -2.0 and not bool(done)

    _, _, second, done = env.agent_step(None, _state(0.40, 0.03), jnp.asarray([1.0]))
    assert float(second) == -1.0 and bool(done)

    _, _, first, done = env.agent_step(None, _state(0.44, 0.05), jnp.asarray([1.0]))
    assert float(first) == 0.0 and bool(done)


def test_a_terminated_state_is_frozen_and_free():
    """Their wrapper breaks out of the repeat, so a terminated episode accrues
    neither motion nor reward — the mask a fixed-length scan needs."""
    env = MountainCar()
    goal = _state(0.5, 0.02)
    obs, state, reward, done = env.agent_step(None, goal, jnp.asarray([1.0]))
    np.testing.assert_array_equal(np.asarray(state[0]), np.asarray(goal[0]))
    assert float(reward) == 0.0 and bool(done)


def test_the_episode_floor_is_minus_200_and_full_throttle_alone_never_solves_it():
    """``TimeLimit(100) x ActionRepeat(2)`` ⇒ the score floor is −200 (which is
    DAIVPG-G's row in their Table 2). Full throttle right is the classic failure:
    the hill is steeper than the engine, so the car never leaves the valley."""
    env = MountainCar()
    _, state = env.reset(KEY)
    total, hit = 0.0, False
    for _ in range(env.time_limit):
        _, state, reward, done = env.agent_step(None, state, jnp.asarray([1.0]))
        total += float(reward)
        hit = hit or bool(done)
    assert not hit
    assert total == -200.0
    assert float(state[0][0]) < 0.45


def test_a_hand_built_swing_solves_it_in_their_step_budget():
    """The task is solvable inside ``TimeLimit(100)``: an energy-pumping bang-bang
    policy (push along the current velocity) reaches the flag in ≈35 agent steps —
    the scale R-AIF's −68.3 ACR implies (addendum §1, ≈34 agent steps)."""
    env = MountainCar()
    _, state = env.reset(KEY)
    total, steps = 0.0, 0
    for _ in range(env.time_limit):
        push = jnp.where(state[0][1] >= 0.0, 1.0, -1.0)
        _, state, reward, done = env.agent_step(None, state, jnp.asarray([push]))
        total += float(reward)
        steps += 1
        if bool(done):
            break
    assert bool(env.reached_goal(state))
    assert steps < 60 and total > -120.0


# ------------------------------------------------------------------- jax hygiene


def test_the_env_is_jittable_scannable_and_static_shaped():
    """A pure-JAX env: an episode is one ``lax.scan`` under jit, and the state
    keeps its shape across the termination mask."""
    env = MountainCar()

    @jax.jit
    def rollout(key, action):
        _, state = env.reset(key)

        def tick(carry, _):
            state, total = carry
            _, state, reward, done = env.agent_step(None, state, action)
            return (state, total + reward), done

        (state, total), dones = jax.lax.scan(
            tick, (state, jnp.zeros(())), None, length=env.time_limit
        )
        return state[0], total, dones

    state, total, dones = rollout(KEY, jnp.asarray([-1.0]))
    assert state.shape == (2,)
    assert dones.shape == (env.time_limit,)
    assert float(total) < 0.0


def test_declaration_errors_are_loud():
    env = MountainCar()
    with pytest.raises(ValueError, match="action_repeat must be at least 1"):
        dataclasses.replace(env, action_repeat=0)
    with pytest.raises(ValueError, match="time_limit must be at least 1"):
        dataclasses.replace(env, time_limit=0)


def test_action_repeat_one_is_the_raw_env():
    """``action_repeat=1`` ⇒ one raw step per agent step and a −100 floor: the
    knob is their wrapper, not a hidden constant."""
    env = dataclasses.replace(MountainCar(), action_repeat=1)
    x = _state(-0.5, 0.0)
    _, state, reward, _ = env.agent_step(None, x, jnp.asarray([1.0]))
    np.testing.assert_allclose(
        np.asarray(state[0]), np.asarray(env.physics_step(x[0], jnp.asarray([1.0]))), atol=0.0
    )
    assert float(reward) == -1.0
