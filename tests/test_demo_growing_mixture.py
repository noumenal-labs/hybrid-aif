"""``examples/growing_mixture.py``, the notebook's structure-learning exhibit: the claims its
prose makes, on the stream the notebook draws and, under ``slow``, over ten streams."""

import importlib.util
import pathlib

import numpy as np
import pytest


def _module():
    path = pathlib.Path(__file__).resolve().parents[1] / "examples" / "growing_mixture.py"
    spec = importlib.util.spec_from_file_location("growing_mixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_prose_of_the_structure_learning_section():
    g = _module()
    points, labels = g.stream()
    r = g.compare(points, labels)
    grown, four, five = r["growing"], r["fixed, 4 levels"], r["fixed, 5 levels"]
    # from tick 300: 4.28, 4.72 and 4.27 nats per point
    settled = [round(g.settled_surprise(x), 2) for x in (grown, four, five)]
    assert settled == [4.28, 4.72, 4.27]
    # the start: a level per cluster within six ticks, about 0.8 nats per point more surprised
    # than either fixed model over the first thirty, within 0.04 of each other from 35 to 250
    assert grown.live[5] == 4
    for fixed in (four, five):
        assert 0.7 < grown.surprise[:30].mean() - fixed.surprise[:30].mean() < 0.9
    middle = [x.surprise[35:250].mean() for x in (grown, four, five)]
    assert max(middle) - min(middle) < 0.04
    # the new cluster's first point: about 40 nats, it wakes a level, the second scores 8.6
    new = np.flatnonzero(labels == 4)
    assert new[0] == 254 and round(float(grown.surprise[new[0]])) in (39, 40)
    assert (grown.live[new[0] - 1], grown.live[new[0]]) == (4, 5)
    assert round(float(grown.surprise[new[1]]), 1) == 8.6
    # the fixed four-level model: the level started on the cluster at (0, −7) takes every new
    # point, and the model is more surprised than the growing one on the old clusters too
    assigned = g.assign(four.system, four.learned, points)
    assert set(assigned[new]) == {2} and np.allclose(g.CENTRES[2], [0.0, -7.0])
    stretched = four.learned.likelihood["point"]
    assert float(stretched.c[2, 0]) < -1.0  # the offset slides towards the new cluster
    sd_x = np.sqrt(float(stretched.Psi[2, 0, 0]) / (float(stretched.nu[2]) - 3.0))
    assert sd_x > 2.0  # and the noise stretches across both (the clusters' own is 0.8)
    old_late = (np.arange(len(points)) >= 300) & (labels < 4)
    assert four.surprise[old_late].mean() > grown.surprise[old_late].mean()
    # reduction: nothing to merge on the growing model; α = 20 fills the cap and comes to five
    assert g.reduce_until_stable(grown.system, grown.learned, grown.prior)[1] == [5]
    system, prior = g.growing_model(20.0)
    eager, _, _ = g.run_stream(system, prior, points)
    assert g.live_count(eager) == g.K_MAX
    assert g.reduce_until_stable(system, eager, prior)[1] == [12, 6, 5, 5]


def test_the_description_lengths_of_the_structure_learning_section():
    g = _module()
    points, labels = g.stream()
    t = g.description_lengths(points, labels)
    grown, four, five = t["growing"], t["fixed, 4 levels"], t["fixed, 5 levels"]
    assert all(x.total.learned for x in (grown, four, five))  # every slot learns along the stream
    # the model parts: 64, 168 and 207 nats; the growing model's prior link is its five levels'
    assert [round(x.total.model) for x in (grown, four, five)] == [64, 168, 207]
    assert [round(x.total.parameters) for x in (grown, four, five)] == [47, 151, 190]
    # the totals: 1,732, 1,864 and 1,810; the five-level model codes the data best
    assert [round(x.total.total) for x in (grown, four, five)] == [1732, 1864, 1810]
    assert five.total.data < grown.total.data < four.total.data
    # the growing model's running total is the lowest at every tick; the five-level model is
    # behind the four-level one through the four-cluster phase and ahead by the end
    assert np.all(four.running > grown.running) and np.all(five.running > grown.running)
    assert five.running[249] > four.running[249] and five.running[-1] < four.running[-1]
    assert round(five.running[249] - four.running[249]) == 43
    passed = np.flatnonzero(five.running < four.running)
    assert passed[0] == 271 and np.all(five.running[271:] < four.running[271:])
    # the new cluster's first point: 39 nats to the growing model, about 9 to the five-level one
    assert round(grown.running[254] - grown.running[253]) == 39
    assert round(five.running[254] - five.running[253]) == 9


@pytest.mark.slow
def test_over_ten_streams_growth_beats_the_fixed_four_and_matches_the_fixed_five():
    g = _module()
    over_four, over_five, eager_ends, strong, four, gaps = [], [], [], [], [], []
    for seed in range(10):
        points, labels = g.stream(seed=seed)
        t = {name: x.running for name, x in g.description_lengths(points, labels).items()}
        gaps.append(
            min(
                (t["fixed, 4 levels"] - t["growing"]).min(),
                (t["fixed, 5 levels"] - t["growing"]).min(),
            )
        )
        assert t["fixed, 5 levels"][249] > t["fixed, 4 levels"][249]
        assert t["fixed, 5 levels"][-1] < t["fixed, 4 levels"][-1]
        r = g.compare(points, labels)
        grown = g.settled_surprise(r["growing"])
        four.append(g.settled_surprise(r["fixed, 4 levels"]))
        over_four.append(four[-1] - grown)
        over_five.append(g.settled_surprise(r["fixed, 5 levels"]) - grown)
        system, prior = g.growing_model(strength=1.0)
        strong.append(float(g.run_stream(system, prior, points)[1][300:].mean()))
        system, prior = g.growing_model(20.0)
        eager, _, _ = g.run_stream(system, prior, points)
        assert g.live_count(eager) == g.K_MAX
        eager_ends.append(g.reduce_until_stable(system, eager, prior)[1][-1])
    assert 0.29 < min(over_four) and max(over_four) < 0.55
    assert max(abs(x) for x in over_five) < 0.04
    assert -0.015 < float(np.mean(over_five)) < -0.005  # the five-level model a little ahead
    # the growing model at the fixed ones' prior strength: worse on average, mixed per stream
    assert (round(float(np.mean(strong)), 2), round(float(np.mean(four)), 2)) == (4.77, 4.63)
    assert sum(s > f for s, f in zip(strong, four, strict=True)) == 6
    assert eager_ends == [5] * 10
    # the description length: the growing model's is the lowest at every tick, by at least 42 nats
    assert min(gaps) > 42.0
