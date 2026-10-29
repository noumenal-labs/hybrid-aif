"""The claims of the demo notebook's Section 8 (``examples/model_document.py``): the documents
round-trip bitwise and differ in the cue's parents alone, each episode's summed free energy is
its exact negative log evidence (the location is observed without noise), and the two-part
code prefers the model whose cue reads the context once the record is long enough."""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
from scipy.special import logsumexp

_PATH = pathlib.Path(__file__).resolve().parents[1] / "examples" / "model_document.py"


def _module():
    spec = importlib.util.spec_from_file_location("model_document", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


md = _module()
MODS = ("reward", "cue")


def test_the_documents_round_trip_and_differ_in_the_cue_parents_alone():
    doc_a, _, same_a = md.round_trip(*md.make_model(True))
    doc_b, _, same_b = md.round_trip(*md.make_model(False))
    assert same_a and same_b
    assert doc_a["factors"]["cue"]["likelihood"]["parents"] == ["location", "context"]
    assert doc_b["factors"]["cue"]["likelihood"]["parents"] == ["location"]
    for name in ("location", "context", "location_obs", "reward"):
        assert doc_a["factors"][name] == doc_b["factors"][name]


def test_an_episode_free_energy_is_its_exact_evidence():
    observations, actions, _ = md.record(6, seed=1)
    for reads in (True, False):
        system, params = md.make_model(reads)
        energies = md.free_energies(system, params, observations, actions)
        tables = {m: np.asarray(params.likelihood[m].log_table, np.float64) for m in MODS}
        a_reward, a_cue = np.exp(tables["reward"]), np.exp(tables["cue"])
        for e in range(6):
            joint = np.log([0.5, 0.5])
            for t in range(md.TICKS_PER_EPISODE):
                loc = int(observations["location_obs"][e, t])
                joint = joint + np.log(a_reward[int(observations["reward"][e, t]), loc])
                cue = a_cue[int(observations["cue"][e, t]), loc]
                joint = joint + np.log(cue if reads else np.broadcast_to(cue, (2,)))
            np.testing.assert_allclose(energies[e].sum(), -logsumexp(joint), atol=1e-4)


def test_the_two_part_code_prefers_a_once_the_record_is_long():
    result = md.compare(n_episodes=120)
    assert result.model_nats["A"][-1] > result.model_nats["B"][-1]  # A costs more to state
    assert result.data_nats["A"][-1] < result.data_nats["B"][-1]  # and codes the data better
    first = md.crossover(result)
    assert first is not None and 10 < first < 120
    # the cue tick costs ln 2 under both models, in every episode
    for label in ("A", "B"):
        np.testing.assert_allclose(result.energies[label][:, 1], np.log(2.0), atol=1e-5)
    # where the cue pays: the arm's reward
    assert result.per_tick["A"][4] < 0.5 * result.per_tick["B"][4]
    # the record part falls by about 0.6 nats an episode
    slope = (result.data_nats["A"][-1] - result.data_nats["B"][-1]) / 120
    assert -0.7 < slope < -0.5


def test_a_pays_heavily_only_when_the_arm_surprises_it():
    result = md.compare(n_episodes=120)
    per_episode = {label: e.sum(axis=1) for label, e in result.energies.items()}
    heavy = per_episode["A"] > 2.0
    surprised = result.wrong_cue | result.null_outcome
    assert heavy.any() and np.array_equal(heavy, surprised)
    # the upward jumps of A − B are the wrong cues: B pays as much as A on a null outcome
    jumps = (per_episode["A"] - per_episode["B"]) > 1.0
    assert np.array_equal(jumps, result.wrong_cue)
