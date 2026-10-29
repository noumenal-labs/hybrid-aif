"""Hide and seek: a prey and a predator, two minds on one grid, coupled through blankets.

The model behind ``examples/hide_and_seek.ipynb`` (the composition vignette). A ``N × N`` grid
with five moves (stay and the four directions; a move into a wall stays, and any other move
fails with probability ``slip``). Two minds, each a :class:`System` of its own with its own
objective and planner, and one world process holding both positions:

    world       prey, predator   each moved by its own action input
                prey_pos, pred_pos   exact observations of the positions
                distance             Manhattan distance binned {0, 1, 2, 3+}

    mind        me       {cell}    moved by the mind's action node ``move``
                other    {cell}    the other mind, modelled as chasing or fleeing
                me_obs, other_obs, distance   wired from the world's observations

The prey models the predator as chasing (a move that shortens the distance is likelier) and
the predator models the prey as fleeing; neither sees the other's objective, and the wiring
is the only channel between them (``docs/design/control-states.md`` C13: beliefs never route).
Each mind's objective is its own declaration, which is what the vignette turns.

Run the smallest case::

    .venv/bin/python examples/hide_and_seek.py
"""

from __future__ import annotations

import dataclasses
import sys

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_aif import Categorical, ConditionalCategorical
from hybrid_aif.composition import Wire, agent_respond, couple, run
from hybrid_aif.system import System, SystemParams

MOVES = ("stay", "north", "south", "east", "west")
DISTANCES = ("caught", "adjacent", "two away", "three or more")
STEP = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, 1), 4: (0, -1)}


def move(cell: int, action: int, n: int) -> int:
    """The cell a move lands on; a move into a wall stays."""
    r, c = divmod(cell, n)
    dr, dc = STEP[action]
    r2, c2 = r + dr, c + dc
    if not (0 <= r2 < n and 0 <= c2 < n):
        return cell
    return r2 * n + c2


def manhattan(a: int, b: int, n: int) -> int:
    (ra, ca), (rb, cb) = divmod(a, n), divmod(b, n)
    return abs(ra - rb) + abs(ca - cb)


def _cc(table, parent_ndims):
    return ConditionalCategorical(
        log_table=jnp.log(jnp.clip(jnp.asarray(table), 1e-16)), parent_ndims=parent_ndims
    )


def own_moves(n: int, slip: float) -> np.ndarray:
    """``(cell', cell, move)``: the move lands with probability ``1 - slip``, else stays."""
    k = n * n
    table = np.zeros((k, k, len(MOVES)))
    for cell in range(k):
        for a in range(len(MOVES)):
            table[move(cell, a, n), cell, a] += 1.0 - slip
            table[cell, cell, a] += slip
    return table


def pursuit(n: int, toward: bool, weight: float) -> np.ndarray:
    """``(other', other, me, 1)``: the other mind's move as this mind models it.

    Moves that shorten the distance (``toward``) or lengthen it (not ``toward``) share
    ``weight`` of the mass; every move shares the rest uniformly. When no move changes the
    distance the wanted way (the same cell; a flight from a corner at the longest distance),
    uniform over the five moves' landing cells.
    """
    k = n * n
    table = np.zeros((k, k, k, 1))
    for other in range(k):
        for me in range(k):
            d0 = manhattan(other, me, n)
            landing = [move(other, a, n) for a in range(len(MOVES))]
            if toward:
                good = [c for c in landing if manhattan(c, me, n) < d0]
            else:
                good = [c for c in landing if manhattan(c, me, n) > d0]
            for c in landing:
                table[c, other, me, 0] += (1.0 - weight) / len(MOVES)
            for c in good:
                table[c, other, me, 0] += weight / len(good)
            if not good:
                table[:, other, me, 0] = 0.0
                for c in landing:
                    table[c, other, me, 0] += 1.0 / len(MOVES)
    return table


def distance_table(n: int) -> np.ndarray:
    """``(bin, a, b)``: the binned Manhattan distance between two cells, exact."""
    k = n * n
    table = np.zeros((len(DISTANCES), k, k))
    for a in range(k):
        for b in range(k):
            table[min(manhattan(a, b, n), 3), a, b] = 1.0
    return table


def _one_hot(cell: int, k: int) -> Categorical:
    return Categorical.from_probs(jnp.eye(k)[cell])


def mind(
    role: str,
    n: int = 4,
    slip: float = 0.2,
    pursuit_weight: float = 0.6,
    policy_len: int = 2,
    gamma: float = 16.0,
    **objective,
) -> tuple[System, SystemParams]:
    """One mind: ``role`` is ``"prey"`` (the other chases; prefers distance) or ``"predator"``
    (the other flees; prefers contact). ``objective`` overrides the by-the-book declaration
    (``epistemic``, ``terms``). The prey starts in the first cell, the predator in the last."""
    k = n * n
    chasing = role == "prey"
    start = (0, k - 1) if chasing else (k - 1, 0)
    system = System(
        A_deps=((0,), (1,), (0, 1)),
        B_deps=((0, 2), (1, 0), ()),
        controls=(2,),
        action_names=("move",),
        factor_names=("me", "other", "u_move"),
        modality_names=("me_obs", "other_obs", "distance"),
        policy_len=policy_len,
        gamma=gamma,
        **({"epistemic": True} | objective),
    )
    if chasing:
        target = jnp.asarray([-6.0, -3.0, 0.0, 2.0])  # the prey: far is good, caught is bad
    else:
        target = jnp.asarray([4.0, 1.0, -2.0, -4.0])  # the predator: contact is the point
    params = SystemParams(
        likelihood={
            "me_obs": _cc(np.eye(k), 1),
            "other_obs": _cc(np.eye(k), 1),
            "distance": _cc(distance_table(n), 2),
        },
        transition={
            "me": _cc(own_moves(n, slip), 2),
            "other": _cc(pursuit(n, toward=chasing, weight=pursuit_weight), 3),
        },
        initial_state_prior={"me": _one_hot(start[0], k), "other": _one_hot(start[1], k)},
        obs_targets={"distance": Categorical(logits=target)},
        state_targets={},
        terminal_targets={},
    )
    return system, params


def world(n: int = 4, slip: float = 0.2):
    """The arena as a process: both positions, moved by two action inputs, emitting the two
    positions and their binned distance; the prey starts in the first cell, the predator in
    the last."""
    k = n * n
    start = (0, k - 1)
    system = System(
        A_deps=((0,), (1,), (0, 1)),
        B_deps=((0, 2), (1, 3), (), ()),
        controls=(2, 3),
        action_names=("prey_move", "pred_move"),
        factor_names=("prey", "predator", "u_prey", "u_pred"),
        modality_names=("prey_pos", "pred_pos", "distance"),
        policy_len=1,
        epistemic=False,
    )
    params = SystemParams(
        likelihood={
            "prey_pos": _cc(np.eye(k), 1),
            "pred_pos": _cc(np.eye(k), 1),
            "distance": _cc(distance_table(n), 2),
        },
        transition={"prey": _cc(own_moves(n, slip), 2), "predator": _cc(own_moves(n, slip), 2)},
        initial_state_prior={"prey": _one_hot(start[0], k), "predator": _one_hot(start[1], k)},
    )
    env, ep = system.as_env(params)
    return env, dataclasses.replace(ep, actuator={})


WIRES = (
    Wire("prey.move", "world.prey_move"),
    Wire("predator.move", "world.pred_move"),
    Wire("world.prey_pos", "prey.me_obs"),
    Wire("world.pred_pos", "prey.other_obs"),
    Wire("world.distance", "prey.distance"),
    Wire("world.pred_pos", "predator.me_obs"),
    Wire("world.prey_pos", "predator.other_obs"),
    Wire("world.distance", "predator.distance"),
)


def wiring(prey, predator, n: int = 4, slip: float = 0.2):
    """The two minds and the arena, wired: each mind's action node drives its position, and
    both minds see both positions and the distance."""
    return couple(
        minds={
            "prey": (*prey, agent_respond(*prey)),
            "predator": (*predator, agent_respond(*predator)),
        },
        processes={"world": world(n, slip)},
        wires=list(WIRES),
    )


def play(prey, predator, ticks: int, key, tick_order: str = "simultaneous", n: int = 4, slip=0.2):
    """One game: the record keyed by participant (``prey``, ``predator``, ``world``).

    A round is one tick under every order and the world steps once per round, so the minds'
    one-move-per-tick tables hold under all three; ``"simultaneous"`` has both minds read the
    round's boundary, ``"sequential"`` and ``"async"`` let a later mind read an earlier one's
    act on a direct wire, of which this wiring has none.
    """
    return run(wiring(prey, predator, n, slip), ticks=ticks, key=key, tick_order=tick_order)


def tally(record) -> dict:
    """What a game came to: ticks caught, the mean distance, and where the prey spent its time."""
    distance = np.asarray(record["world"].observations["distance"])
    cells = np.asarray(record["world"].observations["prey_pos"])
    return {
        "caught": int((distance == 0).sum()),
        "mean_distance_bin": float(distance.mean()),
        "prey_cells": np.bincount(cells, minlength=cells.max() + 1),
    }


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    ticks = int(args[0]) if args else 6
    record = play(mind("prey", n=3), mind("predator", n=3), ticks, jax.random.PRNGKey(0), n=3)
    t = tally(record)
    print(f"{ticks} ticks: caught {t['caught']}, mean distance bin {t['mean_distance_bin']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
