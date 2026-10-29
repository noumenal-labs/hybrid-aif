"""A mixture that grows a component when the world shows it something new.

A stream of 2-D points arrives one per tick from four clusters; part-way through, a fifth
cluster appears. Every model here is one categorical factor ``z``, drawn afresh every tick,
whose levels emit the point through a Gaussian emission (``gaussian_emission``: a Gaussian per
level, its offset and noise under a matrix-normal inverse-Wishart posterior), run through the
same two faces each tick: ``infer_states`` assigns the point and ``learn`` absorbs it.

- The **growing** model has a cap of ``K_MAX`` levels, one live at the start, and its weights
  carry a growth propensity ``α``. ``learn`` wakes a dead level when no live level explains
  the point better than an unused level at its prior would, by a margin of ``log α``, the
  woken level seeded from the point alone. Its starting level is
  declared as if woken: its weight is ``α``, the prior pseudo-count a woken level gets.
- A **fixed** model has ``k`` live levels and no growth. To be generous, its levels start on
  the first point of each of ``k`` distinct clusters, read from the stream's labels; with
  ``k = 5`` that includes the first point of the cluster that has not yet arrived, which no
  model running online could know.

:func:`compare` runs a stream through the growing model and the fixed four- and five-level
models and returns each one's surprise per tick: ``−log p(y)`` of the point under the model's
posterior predictive before it learns from the point (the mixture of the levels' Student-t
predictives under the weights' posterior mean). :func:`description_lengths` gives each model's
description length over the stream, model part plus data part (``description_length`` with
a learning replay: the declaration's graph and prior links, plus the free energy summed
along the replay, which pays for the parameters as they are learned).
:func:`reduce_until_stable` calls ``System.reduce`` until nothing more merges.

Design: docs/design/structure-learning.md
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import logsumexp

from hybrid_aif import (
    ConditionalDirichlet,
    Dirichlet,
    System,
    SystemParams,
    gaussian_emission,
)
from hybrid_aif.description.mdl import DescriptionLength, description_length
from hybrid_aif.replay import replay

K_MAX = 16
OLD = np.asarray([[6.0, 6.0], [-6.0, 6.0], [0.0, -7.0], [7.0, -5.0]])
NEW = np.asarray([[-7.0, -5.0]])
CENTRES = np.vstack([OLD, NEW])  # the fifth row arrives part-way
SPREAD = 0.8
ARRIVAL = 250


def stream(n: int = 400, arrival: int = ARRIVAL, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``n`` points in random order, each one's cluster beside it: from the four old clusters
    before tick ``arrival``, from all five from then on."""
    rng = np.random.default_rng(seed)
    ticks = np.arange(n)
    labels = np.where(ticks < arrival, rng.integers(0, 4, n), rng.integers(0, 5, n))
    points = CENTRES[labels] + SPREAD * rng.normal(size=(n, 2))
    return points.astype(np.float32), labels


def _model(loc, weights, growth=None, scale: float = 1.0, strength: float = 0.1):
    system = System(
        A_deps=((0,),),
        B_deps=((),),
        controlled=(False,),
        policy_len=1,
        modality_names=("point",),
        factor_names=("z",),
        epistemic=False,
    )
    k = len(weights)
    growth = None if growth is None else jnp.asarray(growth)
    weights = jnp.asarray(weights, jnp.float32)[:, None]
    emission = gaussian_emission(
        jnp.asarray(loc, jnp.float32),
        jnp.broadcast_to(scale * jnp.eye(2), (k, 2, 2)),
        strength=strength,
    )
    params = SystemParams(
        likelihood={"point": emission},
        transition={
            "z": ConditionalDirichlet(concentration=weights, parent_ndims=1, growth=growth)
        },
        initial_state_prior={"z": Dirichlet(concentration=weights[:, 0], growth=growth)},
    )
    return system, params


def growing_model(alpha: float = 1.0, strength: float = 0.1):
    """The growing mixture: ``K_MAX`` levels, level 0 live at weight ``α`` (declared as if
    woken), every level's prior centred at the origin with noise scale 1."""
    weights = np.zeros(K_MAX)
    weights[0] = alpha
    return _model(np.zeros((K_MAX, 2)), weights, growth=alpha, strength=strength)


def fixed_model(points: np.ndarray, labels: np.ndarray, k: int, strength: float = 1.0):
    """A fixed mixture of ``k`` live levels, each started on the first point of one of
    clusters ``0 … k−1`` in the stream (with ``k = 5``, the new cluster's first point too).
    ``strength`` 1 is the better of the two prior strengths tried (0.1 and 1) for these."""
    loc = np.stack([points[np.flatnonzero(labels == c)[0]] for c in range(k)])
    return _model(loc, np.ones(k), strength=strength)


def surprise(params: SystemParams, y: Array) -> Array:
    """``−log p(y)`` under the posterior predictive: the live levels' Student-t predictives
    (each level's offset and noise integrated out, at the regressor ``[1]``), mixed by the
    weights' posterior mean. A level not yet woken carries no mass, so the predictive has no
    ``α / (n + α)`` term for a new level and a novel point is charged in full."""
    w = params.transition["z"].concentration[:, 0]
    live = w > 0
    log_w = jnp.where(live, jnp.log(jnp.where(live, w, 1.0)) - jnp.log(jnp.sum(w)), -jnp.inf)
    emission = params.likelihood["point"]
    regressor = jnp.ones((*emission.M.shape[:-2], 1), emission.M.dtype)
    return -logsumexp(log_w + emission.predictive_log_prob(y, regressor))


def run_stream(system: System, params: SystemParams, points: np.ndarray):
    """One tick per point, in a jitted scan: the surprise of the point before learning, then
    ``infer_states`` and ``learn``. Returns the learned params, the surprise per tick and the
    live level count after each tick."""

    @jax.jit
    def run(params, points):
        def tick(p, y):
            s = surprise(p, y)
            obs = {"point": y}
            beliefs, _ = system.infer_states(p, obs, None)
            p = system.learn(p, obs, beliefs, prev_beliefs=beliefs)
            return p, (s, jnp.sum(p.transition["z"].concentration[:, 0] > 0))

        return jax.lax.scan(tick, params, points)

    learned, (s, live) = run(params, jnp.asarray(points))
    return learned, np.asarray(s), np.asarray(live)


class Result(NamedTuple):
    system: System
    prior: SystemParams
    learned: SystemParams
    surprise: np.ndarray  # (n,) −log p(point) before learning from it
    live: np.ndarray  # (n,) live levels after each tick


def compare(points: np.ndarray, labels: np.ndarray, alpha: float = 1.0) -> dict[str, Result]:
    """The growing model and the fixed four- and five-level models on one stream."""
    models = {
        "growing": growing_model(alpha),
        "fixed, 4 levels": fixed_model(points, labels, 4),
        "fixed, 5 levels": fixed_model(points, labels, 5),
    }
    out = {}
    for name, (system, prior) in models.items():
        learned, s, live = run_stream(system, prior, points)
        out[name] = Result(system, prior, learned, s, live)
    return out


class Lengths(NamedTuple):
    total: DescriptionLength  # the whole stream's: model part, data part, their sum
    running: np.ndarray  # (n,) the model part plus the data part up to each tick


def description_lengths(
    points: np.ndarray, labels: np.ndarray, alpha: float = 1.0
) -> dict[str, Lengths]:
    """Each model's description length over the stream: its declaration replayed with learning
    on, priced by ``description_length(..., replay=)``. The running length is the model part (fixed
    once declared: the graph and prior links) plus the free energy summed up to each tick."""
    models = {
        "growing": growing_model(alpha),
        "fixed, 4 levels": fixed_model(points, labels, 4),
        "fixed, 5 levels": fixed_model(points, labels, 5),
    }
    out = {}
    for name, (system, prior) in models.items():
        trace = replay(system, prior, {"point": jnp.asarray(points)}, {}, learn=True, metrics=False)
        total = description_length(system, replay=trace, params=prior)
        running = total.model + np.cumsum(np.asarray(trace.free_energy, np.float64))
        out[name] = Lengths(total, running)
    return out


def rolling(x: np.ndarray, window: int = 25) -> np.ndarray:
    """The trailing mean over ``window`` ticks (shorter at the start)."""
    c = np.cumsum(np.r_[0.0, x])
    lo = np.maximum(np.arange(1, len(x) + 1) - window, 0)
    return (c[1:] - c[lo]) / (np.arange(1, len(x) + 1) - lo)


def live_count(params: SystemParams) -> int:
    """The number of live levels: those whose weight concentration is positive."""
    return int(np.sum(np.asarray(params.transition["z"].concentration[:, 0]) > 0))


def reduce_until_stable(system: System, params: SystemParams, prior: SystemParams, calls: int = 8):
    """``System.reduce`` until a call merges nothing (at most ``calls`` calls): the reduced
    params and the live count after each call. Each call tests the ``reduction_pairs`` best
    candidate pairs greedily, so a model far over-split takes a few calls."""
    counts = []
    for _ in range(calls):
        reduced = system.reduce(params, prior)
        counts.append(live_count(reduced))
        if np.array_equal(
            np.asarray(reduced.transition["z"].concentration),
            np.asarray(params.transition["z"].concentration),
        ):
            break
        params = reduced
    return params, counts


def assign(system: System, params: SystemParams, points: np.ndarray) -> np.ndarray:
    """Each point's most probable level under ``params``."""

    @jax.jit
    def one(y):
        beliefs, _ = system.infer_states(params, {"point": y}, None)
        return jnp.argmax(beliefs["z"].probs)

    return np.asarray(jax.vmap(one)(jnp.asarray(points)))


def levels(params: SystemParams) -> tuple[np.ndarray, np.ndarray]:
    """The live levels' indices and their posterior mean offsets."""
    live = np.flatnonzero(np.asarray(params.transition["z"].concentration[:, 0]) > 0)
    return live, np.asarray(params.likelihood["point"].c)[live]


def settled_surprise(result: Result, after: int = ARRIVAL + 50) -> float:
    """The mean surprise per point from tick ``after`` on."""
    return float(result.surprise[after:].mean())


def main() -> None:
    points, labels = stream()
    results = compare(points, labels)
    for name, r in results.items():
        print(
            f"{name:>16}: surprise {settled_surprise(r):.2f} nats per point from tick "
            f"{ARRIVAL + 50}, {live_count(r.learned)} live levels at the end"
        )
    grown = results["growing"]
    _, counts = reduce_until_stable(grown.system, grown.learned, grown.prior)
    print(f"reduce on the growing model: live after each call {counts}")


if __name__ == "__main__":
    main()
