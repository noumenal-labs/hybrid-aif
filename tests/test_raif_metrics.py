"""ACR / SR / R-S against hand-computed cases (``amortization-p0-addendum.md`` §6).

The three numbers 2d's acceptance is read off, so each is worked out in the test
rather than read off the implementation — and the R-S offset in particular, which
§6 says "must be replicated exactly or R-S is not comparable".
"""

import numpy as np
import pytest

from hybrid_aif.reference.raif.metrics import (
    MC_OFFSET,
    acr,
    episode_returns,
    evaluate,
    metrics_over_time,
    relative_stability,
    sr,
)


def test_acr_is_the_raw_mean_episode_score_with_no_offset():
    """ "ACR — average cumulative reward = mean episode score, raw summed reward
    (``callbacks.py:64,72-73``). No offset." (§6)"""
    rewards = np.array([[-2.0] * 5, [-2.0] * 5, [0.0] * 5])
    np.testing.assert_allclose(episode_returns(rewards), [-10.0, -10.0, 0.0])
    assert acr(episode_returns(rewards)) == pytest.approx(-20.0 / 3.0)


def test_sr_is_the_fraction_of_the_window_that_reached_the_goal():
    assert sr(np.array([True, False, True, True])) == pytest.approx(0.75)


def test_the_r_s_offset_is_applied_per_step_and_changes_the_number():
    """The offset is a metric-only rescaling summed *per step*, which is why 1.1 on
    Mountain Car is load-bearing rather than cancelling."""
    rewards = np.array([[-2.0] * 4])
    np.testing.assert_allclose(episode_returns(rewards, MC_OFFSET), [4 * (-2.0 + 1.1)])
    np.testing.assert_allclose(episode_returns(rewards, 0.0), [-8.0])


def test_relative_stability_is_zero_when_every_episode_is_the_best_one():
    """R-S measures how far the running mean sits below the best episode ever seen,
    so a constant history is perfectly stable. Lower is better."""
    c = np.full(200, -3.6)
    assert relative_stability(c, offset=0.0) == pytest.approx(0.0, abs=1e-12)


def test_relative_stability_is_the_normalized_gap_to_the_best_episode():
    """Hand-computed: a two-value history whose rolling mean over the last window
    sits a known fraction below ``max(C)``.

    With ``C = [-4, -2] * 50`` (so ``max(C) = -2``) and ``window = 2``, every
    rolling mean is ``-3``, and R-S is ``|−3 − (−2)| / |−2| = 0.5``.

    The ``|max(C)|`` in the denominator is flagged in ``relative_stability``'s
    docstring: §6 writes ``max(C)``, which is negative on Mountain Car and would
    make R-S negative where their Table 2 reports ``0.1``.
    """
    c = np.tile([-4.0, -2.0], 50)
    got = relative_stability(c, offset=0.0, window=2)
    assert got == pytest.approx(0.5, abs=1e-12)


def test_relative_stability_refuses_a_history_whose_best_is_zero():
    """The offset exists to keep ``C`` away from zero; without it the normalized
    deviation is undefined, and guessing would make the number incomparable."""
    with pytest.raises(ValueError, match="max"):
        relative_stability(np.zeros(10), offset=0.0)


def test_evaluate_windows_acr_and_sr_and_reports_the_window_size():
    """A short run's numbers are over fewer than 100 episodes and that must not be
    invisible: comparing a 40-episode window to their 100-episode one is a
    different statistic, not a noisier one."""
    rewards = np.concatenate([np.full((5, 3), -2.0), np.full((5, 3), 0.0)])
    successes = np.array([False] * 5 + [True] * 5)
    all_of_it = evaluate(rewards, successes, offset=MC_OFFSET, window=100)
    assert all_of_it.n == 10
    assert all_of_it.acr == pytest.approx(-3.0)  # mean of five −6 and five 0
    assert all_of_it.sr == pytest.approx(0.5)
    last_five = evaluate(rewards, successes, offset=MC_OFFSET, window=5)
    assert last_five.n == 5 and last_five.acr == pytest.approx(0.0) and last_five.sr == 1.0


def test_metrics_over_time_reports_the_curve_not_the_endpoint():
    """Addendum §7's window-honesty ruling as a function: the self-revision dips are
    characteristic behaviour, so the report is a trajectory."""
    rewards = np.full((30, 3), -2.0)
    successes = np.zeros(30, bool)
    curve = metrics_over_time(rewards, successes, window=10, every=10)
    assert [i for i, _ in curve] == [10, 20, 30]
    assert all(m.acr == pytest.approx(-6.0) for _, m in curve)


def test_evaluate_checks_that_the_two_records_agree():
    with pytest.raises(ValueError, match="success flag"):
        evaluate(np.zeros((3, 2)), np.zeros(2, bool))
