"""Diagnostic: the two readings of the ensemble's ``mixsigma``, measured side by side.

**What is being decided.** R-AIF computes the ensemble's disagreement from one **sample**
per member (``preds = [dist.sample() for dist in dists]; mixsigma = preds.std(0)``);
``hybrid_aif.objectives.amortized._ensemble_disagreement`` computes it from the members'
**means** (``mixsigma = means.std(0)``). Ours is theirs in the limit of zero member noise.
Closing the gap would make the leaf a Monte-Carlo estimate, which needs a per-step PRNG key
at the scoring seam that the keyless planner faces (``score_policies``,
``evaluate_policies``, CEM's scorer) cannot supply. This script measures whether that engine
cost would buy anything. It changes no seam and is free to use its own key.
Record: ``docs/experiments/raif-mountain-car/divergence-register.md`` (a).

**What it reports**, per seed, on states from a real imagined rollout:

1. the two leaves' **levels** (ours, deterministic; theirs, averaged over ``--keys``
   draws) and theirs' **across-key spread** — how noisy a single-sample leaf is, which
   is the cost side of adopting it;
2. their **ratio**, and the absolute gap in the units G actually sees;
3. how the gap moves as the members' reported sigma is scaled by
   ``--sigma-scales`` — the aleatoric component is exactly what the sampled reading
   adds, so a leaf that is insensitive to it is a leaf for which the two readings are
   the same estimator in practice.

Point 3 is the one that decides it. ``std(mean_i + sigma_i eps_i)`` over ``M`` members
mixes epistemic spread (the means, what we already have) with aleatoric spread (the
sigmas, what sampling adds); if the ensemble's sigmas are small beside the spread of
its means, the two readings agree and (a) is a principled ablation. If they are
comparable, the readings differ and the leaf's *identity* is at stake.

Run::

    python examples/ensemble_leaf_variants.py --run <run dir>   # a checkpoint's params
    python examples/ensemble_leaf_variants.py                   # untrained agents

Without ``--run`` the windows come from the fixture episodes the run builder loads, so
the ensemble is unfit but the rollout states are real.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile

import jax
import jax.numpy as jnp

from hybrid_aif.objectives.amortized import _action_vector, _diag_entropy, features
from hybrid_aif.reference.raif import run as run_mc
from hybrid_aif.reference.raif.agent import to_world_batch
from hybrid_aif.train.buffers import sample_windows
from hybrid_aif.train.imagine import _code_and_carry, _sampled_step
from hybrid_aif.train.world import posterior_states

ENSEMBLE_ROW = "model_uncertainty"

#: ``means``/``scales`` here are ``(rollout, step, member, flat)`` — the row itself sees
#: only ``(member, flat)``, so the member axis is the one *before* the flattened code,
#: whatever leading batch axes a caller stacked on.
MEMBER_AXIS = -2


def _leaf(mixsigma, scales, flat: int, num_vars: int):
    """The row's arithmetic, with ``mixsigma`` left as the argument under study.

    Everything except the one line being adjudicated is shared, so the two numbers
    below differ in exactly the way the two source lines differ and in no other way.
    """
    #  `mixsigma` is (..., flat) and `scales` (..., M, flat); `_diag_entropy` reduces the
    #  trailing flat axis, so the member axis it must then average over is the LAST one.
    #  (In the row itself there are no leading batch axes and that average is `axis=0`.)
    per_dim = 1.0 / float(flat)
    gap = per_dim * (_diag_entropy(mixsigma) - _diag_entropy(scales).mean(axis=-1))
    return gap / math.sqrt(float(num_vars))


def rollout_parents(system, params, batch, *, key, horizon: int):
    """``(means, scales)`` at every state of a sampled imagined rollout.

    The states are the ones the trainer actually scores: start from the posterior over
    a replay batch, then roll the **actor** forward with :func:`_sampled_step`, which
    is ``roll_batch``'s own transition. Private helpers are imported —
    a diagnostic that reimplemented the rollout could differ from the trainer's, and
    then it would be measuring its own arithmetic instead of the row's.
    """
    world = to_world_batch(batch)
    k_states, k_roll = jax.random.split(key)
    states = posterior_states(system, params, world, key=k_states)
    structure = system._amortized_structure(params)
    beliefs = system.to_positional("beliefs", system.beliefs_from_states(params, states))
    n = beliefs[0].shape[0] * beliefs[0].shape[1]
    flat_beliefs = tuple(jnp.reshape(x, (n, *x.shape[2:])) for x in beliefs)

    code, carry = _code_and_carry(structure, system)
    actor = system._habit_module(params)
    layout = system._control_layout(params)
    bounds = None if layout is None else layout.bounds
    members = tuple(params.scoring[ENSEMBLE_ROW])

    def one(start, k):
        def step(bel, sk):
            act_key, code_key = jax.random.split(sk)
            action = actor.condition_on((features(bel),)).sample(act_key)
            if bounds is not None:
                action = jnp.clip(action, *bounds)
            parents = (
                jnp.reshape(bel[carry], (-1,)),
                jnp.reshape(bel[code], (-1,)),
                _action_vector(
                    action, system.controlled, system._num_controls(params, "act"), layout
                ),
            )
            heads = [m.diag_moments(parents) for m in members]
            out = (
                jnp.stack([mu for mu, _ in heads]),
                jnp.stack([sd for _, sd in heads]),
            )
            new, _, _ = _sampled_step(params, system, carry, code, bel, action, code_key)
            return new, out

        _, out = jax.lax.scan(step, start, jax.random.split(k, horizon))
        return out

    return jax.vmap(one)(flat_beliefs, jax.random.split(k_roll, n))


def compare(means, scales, *, flat: int, num_vars: int, key, draws: int, sigma_scale: float):
    """``(ours, theirs_mean, theirs_std, ratio)`` at one scaling of the member sigmas."""
    scaled = scales * sigma_scale
    ours = _leaf(means.std(axis=MEMBER_AXIS), scaled, flat, num_vars)

    def sampled(k):
        preds = means + scaled * jax.random.normal(k, means.shape, means.dtype)
        return _leaf(preds.std(axis=MEMBER_AXIS), scaled, flat, num_vars)

    theirs = jnp.stack([sampled(k) for k in jax.random.split(key, draws)])
    ours_m = float(jnp.mean(ours))
    theirs_m = float(jnp.mean(theirs))
    #  the across-KEY spread of the leaf's own mean: the sampling noise a single-draw
    #  estimator would inject into G at every step
    across_key = float(jnp.std(jnp.mean(theirs, axis=tuple(range(1, theirs.ndim)))))
    return ours_m, theirs_m, across_key, (theirs_m / ours_m if ours_m != 0 else float("nan"))


def _pool(pair, batch: int):
    """The general stream when it holds ``batch`` windows, else the positive one (the
    fixtures' episodes, which an untrained build has and a run always has)."""
    positive, general = pair
    if general.ready(batch):
        return general
    if positive.ready(batch):
        return positive
    raise SystemExit(f"no buffer holds {batch} windows; collect episodes or pass --run")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=None, help="a run dir whose checkpoint holds trained params")
    p.add_argument("--seeds", default="0,1,2,3")
    p.add_argument("--batch", type=int, default=4, help="replay sequences per seed")
    p.add_argument("--keys", type=int, default=32, help="draws averaged for THEIR leaf")
    p.add_argument("--sigma-scales", default="0.0,0.25,0.5,1.0,2.0,4.0")
    args = p.parse_args(argv)

    seeds = tuple(int(s) for s in args.seeds.split(","))
    if args.run:
        stored = json.loads((run_mc.Config(out=args.run).path / "run.json").read_text())
        stored["seeds"] = tuple(stored["seeds"])
        cfg = run_mc.Config(**stored)
        system, crspp, env, state, buffers, _, _, horizon = run_mc.build(cfg)
        state, buffers, _, counters = run_mc.load_checkpoint(cfg, state, buffers)
        print(f"loaded {args.run} at step {counters.step:,} ({counters.chunk} chunks)")
    else:
        cfg = run_mc.Config(out=os.path.join(tempfile.gettempdir(), "_leaf_diag"), seeds=seeds)
        system, crspp, env, state, buffers, _, _, horizon = run_mc.build(cfg)
        print("no --run: untrained agents (the ensemble has not been fit)")

    structure_params = jax.tree_util.tree_map(lambda x: x[0], state.params)
    members = tuple(structure_params.scoring[ENSEMBLE_ROW])
    flat = members[0].child_dim
    num_vars = structure_params.transition[system.names.states[1]].num_vars
    print(
        f"ensemble: {len(members)} members, flat code {flat}, num_vars {num_vars}, "
        f"horizon {horizon}, {args.batch} sequences x {cfg.seq_len} steps per seed, "
        f"{args.keys} draws for theirs"
    )

    scales_grid = [float(x) for x in args.sigma_scales.split(",")]
    for i, seed in enumerate(cfg.seeds):
        params = jax.tree_util.tree_map(lambda x: x[i], state.params)  # noqa: B023
        batch = sample_windows(
            _pool(buffers[i], args.batch), jax.random.PRNGKey(900 + seed), args.batch
        )
        means, scales = rollout_parents(
            system, params, batch, key=jax.random.PRNGKey(1000 + seed), horizon=horizon
        )
        sd = float(jnp.mean(scales))
        spread = float(jnp.mean(means.std(axis=MEMBER_AXIS)))
        print(
            f"\nseed {seed}: mean member sigma {sd:.5f}, mean spread-of-means {spread:.5f}, "
            f"ratio sigma/spread {sd / spread:.2f}"
        )
        print(
            f"  {'sigma x':>8} {'ours':>12} {'theirs':>12} {'theirs sd':>11} "
            f"{'ratio':>8} {'abs gap':>11}"
        )
        for s in scales_grid:
            ours, theirs, across, ratio = compare(
                means,
                scales,
                flat=flat,
                num_vars=num_vars,
                key=jax.random.fold_in(jax.random.PRNGKey(7), int(s * 1000)),
                draws=args.keys,
                sigma_scale=s,
            )
            print(
                f"  {s:>8.2f} {ours:>12.5f} {theirs:>12.5f} {across:>11.6f} "
                f"{ratio:>8.3f} {theirs - ours:>11.5f}"
            )
    print(
        "\nReading: 'sigma x 1.0' is the live configuration. `ratio` ~ 1 and a small "
        "`abs gap` there means the two readings are the same estimator in practice "
        "(divergence (a) is a principled ablation); a ratio far from 1, or a "
        "`theirs sd` comparable to the gap G's other terms span, means they are not."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
