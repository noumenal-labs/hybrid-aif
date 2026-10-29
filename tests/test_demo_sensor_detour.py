"""``examples/sensor_detour.py``, the claims notebook Section 10 makes of it: under the
objective that expects no observation (the predictive belief's entropy, the ``uncertainty`` row)
the plan goes straight and the objective cannot tell the two plans apart; under the conditioned
belief's information gain the plan detours past the beacon, ends at the dock, leaves a sharper
belief, and its objective prefers it."""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest


@pytest.fixture(scope="module")
def detour():
    path = pathlib.Path(__file__).resolve().parents[1] / "examples" / "sensor_detour.py"
    spec = importlib.util.spec_from_file_location("sensor_detour", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.compare()


def test_the_predictive_belief_plans_straight_and_cannot_prefer_the_detour(detour):
    _, result = detour
    assert np.max(np.abs(result.tracks["prior"][:, 1])) < 0.05
    assert result.G[("prior", "prior")] == pytest.approx(result.G[("prior", "posterior")], abs=1e-3)


def test_the_map_takes_the_values_the_prose_cites(detour):
    module, _ = detour
    sig = module.map_sigmas()
    assert sig["far"] == pytest.approx(0.60, abs=1e-2)
    assert sig["route"] == pytest.approx(0.46, abs=1e-2)
    assert sig["beacon"] == pytest.approx(0.063, abs=1e-2)


def test_the_conditioned_belief_detours_past_the_beacon_and_still_docks(detour):
    module, result = detour
    track = result.tracks["posterior"]
    assert track[:, 1].max() > 0.4
    beside = np.flatnonzero(np.linalg.norm(track - np.array([0.0, 0.5]), axis=1) < 0.02)
    assert beside.tolist() == [2, 3, 4, 5, 6]  # at (0, 0.5) in two steps, held for four
    assert np.linalg.norm(track[-1] - module.DOCK) < 0.1
    assert result.sds["posterior"][-1] < 0.6 * result.sds["prior"][-1]
    assert result.G[("posterior", "posterior")] < result.G[("posterior", "prior")] - 3.0


def test_the_plan_s_printed_score_is_stable_under_a_perturbed_start(detour):
    """A starting plan perturbed by 1e-6 descends to the same score within 2e-3: the descent has
    not stopped, and across a dozen perturbed starts its end points' scores spread by about
    5e-4 (a little under 1e-3 for the previous rule)."""
    module, result = detour
    system, params = module.agents()["posterior"]
    init = 1e-6 * np.random.default_rng(1).standard_normal((module.POLICY_LEN, 2))
    plan = module.choose_plan(system, params, init=init.astype(np.float32))
    G = module.expected_free_energy(system, params, plan)
    assert G == pytest.approx(result.G[("posterior", "posterior")], abs=2e-3)
