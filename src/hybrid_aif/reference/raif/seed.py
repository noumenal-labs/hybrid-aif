"""Seeding M⁺: episode collection in their driver's row convention, and the bang-bang
fallback controller for fixture-less runs. Seed statistics are
:func:`hybrid_aif.train.buffers.seed_stats`.

Design: ``docs/design/amortization.md`` A12 item 6.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.reference.envs.mountain_car import MountainCar
from hybrid_aif.reference.raif.columns import raif_episode
from hybrid_aif.train.buffers import Episode

#: A policy for :func:`collect_episode`: ``(key, obs) -> action`` of shape ``(1,)``.
Policy = Callable[[Array, Array], Array]


def bang_bang(key: Array, obs: Array) -> Array:
    """``a = sign(velocity)``: full throttle in the direction of travel, deterministic.

    ``sign(0) = 0`` at the reset state costs one step and is not tie-broken.
    """
    del key  # deterministic, by construction — which is A12 item 6's diversity worry
    return jnp.sign(jnp.asarray(obs)[1:2])


def collect_episode(
    env: MountainCar,
    policy: Policy,
    key: Array,
    *,
    can_self_imitate: bool = False,
    steps: int | None = None,
) -> Episode:
    """Run one episode eagerly and record it as an :class:`Episode`.

    ``steps`` (default: the env's ``time_limit``) bounds the applied actions, so the episode is
    at most ``steps + 1`` rows and stops early at the goal. Rows follow their driver, as
    :func:`hybrid_aif.reference.raif.run.make_collect_fn` does: row ``t`` is
    ``(obs_t, reward_t, flags_t, a_t)`` with ``reward_t`` caused by ``a_{t−1}``, opened by a
    reset row at ``reward = 0`` and closed by a terminal row whose action is zeroed and whose
    ``can_self_imitate`` is cleared. ``is_success`` is the goal predicate on the row's
    observation, ``has_succeeded`` its running OR, and ``can_self_imitate`` is the caller's
    flag for the policy (``True`` for expert data).

    Design: ``docs/experiments/raif-mountain-car/execution-ledger-acting.md`` D1–D3.
    """
    steps = int(env.time_limit if steps is None else steps)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    rows: list[tuple] = []
    succeeded, incoming = False, jnp.zeros((), jnp.float32)
    for t in range(steps + 1):
        key, act_key, step_key = jax.random.split(key, 3)
        current = obs[0]
        success = bool(env.reached_goal(state))
        succeeded = succeeded or success
        is_last = success or t >= steps
        action = jnp.reshape(jnp.asarray(policy(act_key, current), jnp.float32), (1,))
        rows.append(
            (
                current,
                jnp.zeros_like(action) if is_last else action,
                incoming,
                success,
                succeeded,
                0.0 if is_last else float(can_self_imitate),
                is_last,
            )
        )
        if is_last:
            break
        obs, state, incoming, _ = env.agent_step(step_key, state, jnp.clip(action, -1.0, 1.0))
    n = len(rows)
    return raif_episode(
        obs=jnp.stack([row[0] for row in rows]),
        action=jnp.stack([row[1] for row in rows]),
        reward=jnp.stack([jnp.asarray(row[2], jnp.float32) for row in rows]),
        is_success=jnp.stack([jnp.asarray(row[3]) for row in rows]),
        has_succeeded=jnp.stack([jnp.asarray(row[4]) for row in rows]),
        can_self_imitate=jnp.asarray([row[5] for row in rows], jnp.float32),
        is_first=jnp.zeros((n,), bool).at[0].set(True),
        is_last=jnp.zeros((n,), bool).at[-1].set(True),
    )


def bang_bang_episodes(
    key: Array, n_episodes: int = 22, env: MountainCar | None = None
) -> list[Episode]:
    """``n_episodes`` seed episodes from :func:`bang_bang`, flagged expert.

    The default 22 is their Mountain Car seed budget (≈1,000 agent steps). Every row but the
    terminal one has ``can_self_imitate=True``.
    """
    env = MountainCar() if env is None else env
    keys = jax.random.split(key, int(n_episodes))
    return [collect_episode(env, bang_bang, k, can_self_imitate=True) for k in keys]
