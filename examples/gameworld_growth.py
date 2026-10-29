"""A Gameworld-style task learned by a model that grows and merges its own levels.

The world is a plane with a player and a few pieces of food, whose state the driver supplies
(object state given). Each tick the player takes one of four moves; the reward says whether
the move brought it closer to the nearest food (+1), farther from it (−1) or neither (0, a
move into the wall). The model is one growable categorical factor ``z``, a recurrent-mixture
style assignment drawn afresh every tick, whose levels emit the tick's tuple: the direction to
the nearest food (a Gaussian emission), the move and the reward (tabular emissions). The model
starts with one live level and a cap; the learn face wakes a level whenever a tuple is
explained worse by every live level than by the prior (structure-learning.md S3), and every
``reduction_period`` learn calls merges the levels whose pooled evidence is no lower (S4).

The driver chooses the move by the model's own reward prediction: for each move, the
posterior over ``z`` given the tuple without its reward, and the reward's expectation under
it; an exploratory draw on top, at a rate falling over the first half of the run. Run as a
script it prints the reward per window and the live level count along the run, with reduction
on and off. On seed 0 a reduction every 250 learn calls reaches the same reward as no reduction
with half the levels, while one every 100 calls merges young levels into a fat early one before
their move and reward tables have sharpened, and the agent never learns; over seeds 0 to 2 the
task is fragile under every schedule (the design doc's acceptance notes give the table).

Design: docs/design/structure-learning.md (Acceptance)
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_aif import (
    ConditionalDirichlet,
    Dirichlet,
    System,
    SystemParams,
    gaussian_emission,
)

MOVES = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
REWARD_VALUES = np.asarray([-1.0, 0.0, 1.0])


class World(NamedTuple):
    player: np.ndarray  # (2,)
    objects: np.ndarray  # (n, 2), the food


def _spawn(rng: np.random.Generator, n: int, size: float) -> World:
    return World(player=rng.uniform(0.0, size, size=2), objects=rng.uniform(0.0, size, size=(n, 2)))


def _nearest(world: World) -> tuple[int, np.ndarray]:
    gaps = world.objects - world.player
    i = int(np.argmin(np.linalg.norm(gaps, axis=-1)))
    return i, gaps[i]


def _step(world: World, move: int, rng: np.random.Generator, size: float) -> tuple[World, int]:
    """The world after ``move``, and the reward level (0: −1, 1: 0, 2: +1)."""
    i, gap = _nearest(world)
    before = float(np.linalg.norm(gap))
    player = np.clip(world.player + MOVES[move], 0.0, size)
    after = float(np.linalg.norm(world.objects[i] - player))
    reward = 1
    if after < before - 1e-6:
        reward = 2
    elif after > before + 1e-6:
        reward = 0
    objects = world.objects
    if after < 0.8:  # contact: the food respawns elsewhere
        objects = objects.copy()
        objects[i] = rng.uniform(0.0, size, size=2)
    return World(player, objects), reward


def make_model(
    k_max: int = 48,
    alpha: float = 1.0,
    reduction_period: int | None = 250,
    scale: float = 1.0,
    sharpness: float = 0.05,
):
    """The growable mixture over the tick's tuple, one level live. ``scale`` is the prior noise
    of the Gaussian emission, the width of a level in the plane of displacements; ``sharpness``
    is the tabular emissions' prior concentration, small so that a level that has seen one move
    or one reward explains another badly and the tuple wakes a level of its own."""
    system = System(
        A_deps=((0,), (0,), (0,)),
        B_deps=((),),
        controlled=(False,),
        policy_len=1,
        modality_names=("motion", "move", "reward"),
        factor_names=("z",),
        epistemic=False,
        reduction_period=reduction_period,
        reduction_pairs=16,
    )
    weights = jnp.zeros((k_max, 1)).at[0, 0].set(1.0)
    growth = jnp.asarray(alpha)

    def table(levels: int) -> ConditionalDirichlet:
        return ConditionalDirichlet(
            concentration=jnp.full((levels, k_max), sharpness), parent_ndims=1
        )

    params = SystemParams(
        likelihood={
            "motion": gaussian_emission(
                jnp.zeros((k_max, 2)), jnp.broadcast_to(scale * jnp.eye(2), (k_max, 2, 2)), 0.1
            ),
            "move": table(4),
            "reward": table(3),
        },
        transition={
            "z": ConditionalDirichlet(concentration=weights, parent_ndims=1, growth=growth)
        },
        initial_state_prior={"z": Dirichlet(concentration=weights[:, 0], growth=growth)},
    )
    return system, params


def _features(world: World) -> np.ndarray:
    """The tuple's continuous part: the direction to the nearest food (its displacement
    normalised; the reward turns on the direction, not the distance)."""
    _, gap = _nearest(world)
    direction = gap / max(float(np.linalg.norm(gap)), 1e-6)
    return direction.astype(np.float32)


def run(
    ticks: int = 2000,
    reduction_period: int | None = 250,
    seed: int = 0,
    window: int = 200,
    epsilon: float = 0.05,
    size: float = 8.0,
) -> dict[str, np.ndarray]:
    """One run: the reward averaged per window, the live level count at each window's end.
    ``reduction_period`` is the learn calls between reductions (``None``: no reduction). The
    move is the model's best by predicted reward, except for an exploratory draw whose rate
    falls from one at the start to ``epsilon`` by the run's midpoint."""
    rng = np.random.default_rng(seed)
    system, params = make_model(reduction_period=reduction_period, scale=0.5)
    prior = params
    state = system.init_learn_state(params)
    world = _spawn(rng, n=4, size=size)
    moves = jnp.arange(4, dtype=jnp.int32)
    reward_values = jnp.asarray(REWARD_VALUES, jnp.float32)

    @jax.jit
    def expected_rewards(params, motion):
        """The reward each move predicts: the posterior over ``z`` for the tuple without its
        reward, and the reward table's mean under it."""

        def one(move):
            beliefs, _ = system.infer_states(
                params,
                {"motion": motion, "move": move, "reward": None},
                None,
                present={"reward": False},
            )
            table = params.likelihood["reward"].mean()  # (3, K)
            return jnp.sum((table @ beliefs["z"].probs) * reward_values)

        return jax.vmap(one)(moves)

    @jax.jit
    def learn(params, state, motion, move, reward):
        obs = {"motion": motion, "move": move, "reward": reward}
        beliefs, _ = system.infer_states(params, obs, None)
        return system.learn(
            params, obs, beliefs, prev_beliefs=beliefs, learn_state=state, prior=prior
        )

    rewards, live = [], []
    total = 0.0
    for t in range(ticks):
        motion = jnp.asarray(_features(world))
        explore = max(epsilon, 1.0 - 2.0 * t / ticks)
        if rng.uniform() < explore:
            move = int(rng.integers(0, 4))
        else:
            predicted = np.asarray(expected_rewards(params, motion))
            best = np.flatnonzero(predicted >= predicted.max() - 1e-6)
            move = int(rng.choice(best))
        world, reward = _step(world, move, rng, size)
        total += REWARD_VALUES[reward]
        params, state = learn(
            params, state, motion, jnp.asarray(move, jnp.int32), jnp.asarray(reward, jnp.int32)
        )
        if (t + 1) % window == 0:
            rewards.append(total / window)
            live.append(int(np.sum(np.asarray(params.transition["z"].concentration[:, 0]) > 0)))
            total = 0.0
    return {"reward": np.asarray(rewards), "live": np.asarray(live)}


def main() -> None:
    ticks, window = 2000, 200
    on, off = run(ticks, 250, window=window), run(ticks, None, window=window)
    head = ("window", "reward (on)", "live (on)", "reward (off)", "live (off)")
    print(f"{head[0]:>8} {head[1]:>12} {head[2]:>10} {head[3]:>13} {head[4]:>11}")
    for i in range(len(on["reward"])):
        print(
            f"{(i + 1) * window:>8} {on['reward'][i]:>12.3f} {on['live'][i]:>10d} "
            f"{off['reward'][i]:>13.3f} {off['live'][i]:>11d}"
        )


if __name__ == "__main__":
    main()
