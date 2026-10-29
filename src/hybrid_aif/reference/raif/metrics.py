"""R-AIF's three metrics: ACR, SR and R-S, folded from recorded training episodes.

They are online training statistics over the last :data:`WINDOW` episodes; nothing here
steps an environment. ACR is the mean raw episode score, SR the fraction of episodes that
reached the goal, and R-S (lower is better) how far the rolling mean of ``C`` sits below the
best episode's. R-S depends on the task's per-step ``offset`` (:data:`MC_OFFSET`).

Design: ``docs/design/amortization-p0-addendum.md`` §6.
"""

from typing import NamedTuple

import numpy as np

#: ``r_reward_offset`` per task (``configs.yaml:116,126,137``). R-S only.
MC_OFFSET = 1.1
METAWORLD_OFFSET = 3.1
ROBOSUITE_OFFSET = 1.0

#: Their window: "the last 100 recorded agent training episodes" (paper §4).
WINDOW = 100


class Metrics(NamedTuple):
    """``(acr, sr, rs)`` over one window, plus ``n``, the number of episodes in it.

    A window shorter than :data:`WINDOW` is a different statistic from a full one.
    """

    acr: float
    sr: float
    rs: float
    n: int


def acr(returns: np.ndarray) -> float:
    """Average cumulative reward: the mean episode score, raw and unoffset."""
    return float(np.mean(np.asarray(returns, np.float64)))


def sr(successes: np.ndarray) -> float:
    """Success rate: the fraction of episodes that reached the goal at least once."""
    return float(np.mean(np.asarray(successes, np.float64)))


def relative_stability(
    returns: np.ndarray, offset: float = MC_OFFSET, window: int = WINDOW
) -> float:
    """R-S over per-episode ``C``: how far the rolling mean sits below the best. Lower is better.

    ``R-S = mean_k |rolling_mean_window(C)_k − max(C)| / |max(C)|`` over the last ``window``
    rolling means, with ``max(C)`` over the whole history. With fewer than ``window`` episodes
    the rolling mean is taken over what there is, and the result is not comparable to a full
    run's. Raises ``ValueError`` on no episodes or on ``max(C) == 0``.

    Design: ``docs/experiments/raif-mountain-car/execution-ledger-acting.md`` D9.
    """
    c = np.asarray(returns, np.float64)
    if c.size == 0:
        raise ValueError("relative_stability: no episodes")
    n = min(int(window), c.size)
    best = np.max(c)
    if best == 0.0:
        raise ValueError(
            "relative_stability: max(C) is 0, so the normalized deviation is undefined — the "
            "offset exists to keep C away from zero and must be the task's "
            f"(Mountain Car: {MC_OFFSET})"
        )
    kernel = np.ones(n) / n
    rolling = np.convolve(c, kernel, mode="valid")
    tail = rolling[-int(window) :]
    return float(np.mean(np.abs(tail - best) / np.abs(best)))


def _reward_rows(rewards) -> list[np.ndarray]:
    """One 1-D reward column per episode, from a rectangular array or a ragged list.

    A rectangle must hold whole episodes of equal length, not padding: :func:`stability_c`
    drops the last row's reward.
    """
    if isinstance(rewards, np.ndarray) and rewards.ndim == 2:
        return [np.asarray(row, np.float64) for row in rewards]
    rows = [np.asarray(row, np.float64).reshape(-1) for row in rewards]
    if not rows:
        raise ValueError("no episodes")
    if any(row.ndim != 1 or row.size == 0 for row in rows):
        raise ValueError("each episode is a non-empty 1-D reward column")
    return rows


def episode_returns(rewards, offset: float = 0.0) -> np.ndarray:
    """Per-episode summed reward, with ``offset`` added per step.

    ``rewards`` is ``(episodes, steps)`` or a ragged sequence of reward columns. At
    ``offset=0`` this is the score ACR averages. R-S's ``C`` is :func:`stability_c`, not this.
    """
    return np.asarray([float(np.sum(row + offset)) for row in _reward_rows(rewards)], np.float64)


def stability_c(rewards: np.ndarray, offset: float = MC_OFFSET) -> float:
    """R-S's ``C`` for one episode: ``Σ_{t<T} (reward[t] + offset)`` over its ``T+1`` rows.

    The closing ``is_last`` row contributes nothing and the reset row contributes one
    ``offset``, as their ``MeasureRStability`` accumulates. ``rewards`` is one episode's
    ``(T+1,)`` reward column in row order.

    Design: ``docs/experiments/raif-mountain-car/execution-ledger-acting.md`` D9.
    """
    r = np.asarray(rewards, np.float64).reshape(-1)
    if r.size == 0:
        raise ValueError("stability_c: an episode with no rows")
    return float(np.sum(r[:-1] + offset))


def evaluate(
    rewards,
    successes: np.ndarray,
    *,
    offset: float = MC_OFFSET,
    window: int = WINDOW,
) -> Metrics:
    """The three metrics over the last ``window`` episodes of a training run.

    ``rewards`` is ``(episodes, steps)`` or a sequence of per-episode reward columns, one whole
    unpadded episode each; ``successes`` ``(episodes,)`` is each episode's
    ``has_succeeded[-1]``. ACR and SR are over the window. R-S is over the window's rolling
    means against the best episode in the whole history.
    """
    rows = _reward_rows(rewards)
    successes = np.asarray(successes)
    if len(rows) != successes.shape[0]:
        raise ValueError(
            f"{len(rows)} episode(s) of reward but {successes.shape[0]} success flag(s)"
        )
    n = min(int(window), len(rows))
    c = np.asarray([stability_c(row, offset) for row in rows], np.float64)
    return Metrics(
        acr=acr(episode_returns(rows)[-n:]),
        sr=sr(successes[-n:]),
        rs=relative_stability(c, offset=offset, window=window),
        n=n,
    )


def metrics_over_time(
    rewards,
    successes: np.ndarray,
    *,
    offset: float = MC_OFFSET,
    window: int = WINDOW,
    every: int = 10,
) -> list[tuple[int, Metrics]]:
    """``(episode index, Metrics)`` every ``every`` episodes: the curve, not the endpoint."""
    rows = _reward_rows(rewards)
    out = []
    for i in range(int(every), len(rows) + 1, int(every)):
        out.append((i, evaluate(rows[:i], successes[:i], offset=offset, window=window)))
    return out
