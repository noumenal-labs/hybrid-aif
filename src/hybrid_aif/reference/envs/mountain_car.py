"""R-AIF's state-based Mountain Car: gymnasium's ``MountainCarContinuous`` physics with their
sparse reward and termination.

State and observation are ``(position, velocity)``; the action is ``Box(-1, 1, (1,))``.
Reward is −1 per raw step and 0 on the terminating one; the episode terminates at
``position >= 0.45`` (velocity ignored), which is also the success predicate. One agent
step is ``action_repeat = 2`` raw steps, breaking on termination, so a non-terminating
agent step scores −2 and the floor over ``TimeLimit(100)`` agent steps is −200. The time
limit is the caller's to enforce (:attr:`MountainCar.time_limit`). Reset draws
``position ~ U(-0.6, -0.4)``, ``velocity = 0``. Reward and termination ride on
:meth:`MountainCar.agent_step`, beside the :class:`~hybrid_aif.envs.base.Env` faces.

Design: ``docs/design/amortization-p0-addendum.md`` §1.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.envs.base import Env, Obs, State


class MountainCar(Env):
    """Sparse-reward continuous Mountain Car. The dynamics constants are static fields."""

    power: float = eqx.field(static=True, default=0.0015)
    max_speed: float = eqx.field(static=True, default=0.07)
    min_position: float = eqx.field(static=True, default=-1.2)
    max_position: float = eqx.field(static=True, default=0.6)
    goal_position: float = eqx.field(static=True, default=0.45)
    gravity: float = eqx.field(static=True, default=0.0025)
    action_repeat: int = eqx.field(static=True, default=2)
    time_limit: int = eqx.field(static=True, default=100)
    reset_low: float = eqx.field(static=True, default=-0.6)
    reset_high: float = eqx.field(static=True, default=-0.4)

    def __check_init__(self) -> None:
        if self.action_repeat < 1:
            raise ValueError(f"action_repeat must be at least 1, got {self.action_repeat}")
        if self.time_limit < 1:
            raise ValueError(f"time_limit must be at least 1, got {self.time_limit}")

    @property
    def num_factors(self) -> int:
        """One factor: the ``(position, velocity)`` vector."""
        return 1

    @property
    def action_dim(self) -> int:
        return 1

    # ------------------------------------------------------------------- dynamics

    def physics_step(self, x: Array, action: Array) -> Array:
        """One **raw** env step of gymnasium's MountainCarContinuous dynamics."""
        position, velocity = x[0], x[1]
        force = jnp.clip(jnp.asarray(action).reshape(-1)[0], -1.0, 1.0)
        velocity = velocity + force * self.power - self.gravity * jnp.cos(3.0 * position)
        velocity = jnp.clip(velocity, -self.max_speed, self.max_speed)
        position = jnp.clip(position + velocity, self.min_position, self.max_position)
        # the left wall is inelastic: hitting it kills inward velocity
        velocity = jnp.where((position <= self.min_position) & (velocity < 0.0), 0.0, velocity)
        return jnp.stack([position, velocity])

    def reached_goal(self, state: State) -> Array:
        """``position >= goal_position``: the termination predicate and the success flag."""
        return state[0][0] >= self.goal_position

    # ------------------------------------------------------------------ env faces

    def reset(self, key: Array, params=None) -> tuple[Obs, State]:
        """``position ~ U(-0.6, -0.4)``, ``velocity = 0`` (gymnasium's reset)."""
        position = jax.random.uniform(
            key, (), minval=self.reset_low, maxval=self.reset_high, dtype=jnp.float32
        )
        state = (jnp.stack([position, jnp.zeros((), jnp.float32)]),)
        return self.observe(state), state

    def step(self, key: Array, state: State, action: Array, params=None) -> tuple[Obs, State]:
        """One agent step as the ``Env`` face: :meth:`agent_step` without reward and done."""
        obs, state, _, _ = self.agent_step(key, state, action, params)
        return obs, state

    def agent_step(
        self, key: Array, state: State, action: Array, params=None
    ) -> tuple[Obs, State, Array, Array]:
        """One agent step with the sparse reward: ``(obs, state, reward, done)``.

        Runs ``action_repeat`` raw steps, freezing the state once the goal is reached. ``reward``
        is −1 per raw step taken before the goal; ``done`` is termination at or before this step.
        """
        del key, params  # the dynamics are deterministic; only reset draws
        x = state[0]
        done = self.reached_goal((x,))
        reward = jnp.zeros((), jnp.float32)
        for _ in range(self.action_repeat):
            x_next = self.physics_step(x, action)
            hit = x_next[0] >= self.goal_position
            x = jnp.where(done, x, x_next)
            reward = reward + jnp.where(done, 0.0, jnp.where(hit, 0.0, -1.0))
            done = done | hit
        state = (x,)
        return self.observe(state), state, reward, done

    def observe(self, state: State) -> Obs:
        """The observation: the state itself, without noise."""
        return (state[0],)
