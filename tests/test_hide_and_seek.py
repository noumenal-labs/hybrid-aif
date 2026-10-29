"""``examples/hide_and_seek.py``, the composition vignette's model, runs and its wiring is honest.

Loaded from ``examples/`` the way the other demo suites load theirs (``importlib`` off the
file path) and driven on the smallest grid, so the seams it reads through (``couple``, the
driver's tick orders, ``as_env`` on a two-input System) cannot drift without this noticing.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import jax
import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "examples" / "hide_and_seek.py"
_spec = importlib.util.spec_from_file_location("hide_and_seek", _PATH)
hs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hs)


def test_the_smallest_game_runs(capsys):
    assert hs.main(["3"]) == 0
    assert "3 ticks: caught" in capsys.readouterr().out


def test_the_world_emits_the_distance_of_the_positions_it_emits():
    #  under "simultaneous" every mind's rows are the world's tick boundaries; under the other
    #  orders a later mind's row already holds an earlier mind's move (driver docstring)
    record = hs.play(
        hs.mind("prey", n=3),
        hs.mind("predator", n=3),
        5,
        jax.random.PRNGKey(2),
        tick_order="simultaneous",
        n=3,
    )
    world = record["world"].observations
    prey, pred, distance = (np.asarray(world[k]) for k in ("prey_pos", "pred_pos", "distance"))
    want = [min(hs.manhattan(int(a), int(b), 3), 3) for a, b in zip(prey, pred, strict=True)]
    assert distance.tolist() == want
    #  both minds received the world's rows, each under its own port names
    assert np.array_equal(record["prey"].observations["me_obs"], prey)
    assert np.array_equal(record["predator"].observations["other_obs"], prey)
    assert np.array_equal(record["prey"].observations["distance"], distance)
    #  one move per tick, as the minds' tables say
    assert all(
        hs.manhattan(int(a), int(b), 3) <= 1 for a, b in zip(prey[:-1], prey[1:], strict=True)
    )


def test_a_minds_objective_is_its_own_declaration():
    system, params = hs.mind("prey", n=3)
    mop = dataclasses.replace(system, epistemic=False, terms=("occupancy:transition:me",))
    params = dataclasses.replace(params, obs_targets={})
    predator = hs.mind("predator", n=3)
    record = hs.play((mop, params), predator, 3, jax.random.PRNGKey(0), n=3)
    assert record["prey"].actions["move"].shape == (3,)
    #  the swapped mind's tree is the one leaf it declared; the other mind's tree is untouched
    flat = {f: hs.Categorical.from_probs(np.full(9, 1 / 9)) for f in ("me", "other")}
    prey_obs = {m: v[0] for m, v in record["prey"].observations.items()}
    beliefs, _ = mop.infer_states(params, prey_obs, flat)
    _, tree = mop.infer_policies(params, beliefs)
    assert set(tree) == {"occupancy"}
    pred_obs = {m: v[0] for m, v in record["predator"].observations.items()}
    beliefs, _ = predator[0].infer_states(predator[1], pred_obs, flat)
    _, tree = predator[0].infer_policies(predator[1], beliefs)
    assert set(tree) == {"utility", "infogain"}
