# The preferred rollout, mean-field versus sampled — a read-only probe (ledger row 6.5)

**Status: measurement, not a decision.** *Followed by `pref-rollout-probe-500k.md` (2026-09-20): three hypotheses for why sampling would help, tested on two seeds at 500k; all three fall, and option 2 below (mode) is measured worse than what ships.* Nothing in the library changed. The question of
*whether* to close this divergence — which needs a PRNG key in the scoring seam, an engine
change — is reserved for a deliberate round (Alex, 2026-09-18: "whether to change how the
library works to pass the PRNG key through was a momentous decision … we want to avoid
changing the library just to accommodate this replication"). This file records what the
divergence costs so that round has a number to work with.

**Date:** 2026-09-18. **Checkpoint:** `runs/crspp_init_100k` at step 100,352 (`734665b`,
`winit: normal`), seed 0. **Theirs:** `raif-paper` @ `9b1864e8`.

## The divergence

Their `imagine_preference` (`agent.py:262-279`) rolls the CRSPP forward with
`img_step_no_action` (`nets.py:189-199`), which **samples** the code at every step, so both
sides of their cosine are one-hot trajectories. Our `cosine_sim_to_pref_rollout` row
(`objectives/amortized.py:412-462`) rolls it with the keyless `update_empirical_prior`, so
the preferred side is a **probability** trajectory. Recorded at build time as FLAG (1) of the
chunk-2c note (`amortization.md`): "Sampling the preferred rollout means giving the row a
key, which the planner seam does not carry ('the keyless faces stay as they are'), so the
build left it mean-field and recorded the divergence." `actor-trace-audit.md` §2.3 ranked it
second among the actor's inputs and stated what nobody had measured: whether it is a
*level* shift (absorbed by the critic baseline) or a *shape* shift (not).

## Method

One agent rollout, then two preferred trajectories from the same start, then the leaf.

1. `run_mc.build` + `load_checkpoint` on the run above; seed 0's parameters.
2. One replay batch of 16 × 64 windows from M (`sample_windows`, key 123); posterior states,
   the flat 1024 starts, and the CRSPP posterior over the batch's embeddings via
   `crspp_recognize` — exactly what `train_step` hands `imagine` as `aux_beliefs`.
3. `train.imagine.roll_batch` — the trainer's own arithmetic — for the agent's imagined
   states `s_0 … s_7` and the logged leaf.
4. The preferred trajectory from the same `aux` start, rolled two ways: **mean-field** with
   `crspp.update_empirical_prior(…, None)` (what ships), and **sampled** with
   `reference.crspp.img_step_no_action` (their step, already ported).
5. `cos(features(s_{k+1}), features(pref_{k+1}))` for `k = 0 … 6`, which is what the row's
   `step` computes at index `k`.

**Sanity check, exact:** the recomputed mean-field leaf equals the trainer's own logged
`per_term["cosine_sim_to_pref_rollout"]` to the last printed digit (0.6511045694351196 both),
so the probe is evaluating the leaf the actor's advantage actually sees.

## Result

| rollout step k | mean-field (ships) | sampled (theirs) | Δ |
|---|---|---|---|
| 0 | 0.6512 | 0.5253 | −0.1258 |
| 1 | 0.6576 | 0.5325 | −0.1252 |
| 2 | 0.6588 | 0.5333 | −0.1255 |
| 3 | 0.6567 | 0.5331 | −0.1237 |
| 4 | 0.6523 | 0.5312 | −0.1211 |
| 5 | 0.6453 | 0.5253 | −0.1200 |
| 6 | 0.6357 | 0.5180 | −0.1177 |
| **all** | **0.6511** | **0.5284** | **−0.1227** |

The oracle's own `train/extr_G_crspp_mean` (seed 0) at the matching steps is 0.4678 (60k),
0.4571 (84k), 0.4737 (104k). Our logged series over 60–100k is 0.65–0.68. Sampling the
preferred rollout accounts for **roughly two thirds** of the standing gap (0.651 → 0.528
against 0.47); the residual 0.05–0.06 is not this divergence.

**It is a level shift, almost purely.** The delta is −0.12 at every step, and the per-start
leaf means correlate at **0.979** between the two variants. Per-start s.d. 0.067 vs 0.057;
per-step s.d. across starts 0.072 vs 0.064. By the actor-trace audit's reasoning a constant
added to every `G[k]` is largely absorbed by the critic baseline in `ret − base`.

**Mechanism, confirmed.** Feature vector `[deter(128), flat stoch(1024)]`:

| part | agent (one-hot) | pref mean-field | pref sampled |
|---|---|---|---|
| stoch-part norm | 5.657 (= √32) | **1.228** | 5.657 |
| deter-part norm | 5.71 | 7.58 | 7.54 |
| deter-only cosine | | 0.899 | 0.900 |
| stoch-only cosine | | 0.195 | **0.042** |

Under mean-field the preferred side's stoch part is 4.6× too short, so the cosine is
dominated by `deter` (0.90) and reads high. Sampled, the stoch halves are one-hot codes that
agree at **chance** (0.042 ≈ 1/32 classes): the agent's imagined codes and the preference
model's imagined codes are essentially uncorrelated at this checkpoint, on both their
convention and ours. Their cosine is therefore mostly a `deter` cosine diluted by an
uninformative stoch half; ours is the same `deter` cosine less diluted.

## What this does and does not settle

- It settles the *size* and *kind* of the divergence: ~0.12, level not shape, from the norm
  re-weighting and not from the preference model being differently trained (the data-batch
  similarity already matched theirs from 4k, `series-sweep.md` §4.1).
- It does **not** settle whether behaviour changes. A level shift is absorbed by the
  baseline, but the *gradient* to the actor through the cosine is re-weighted between the
  `deter` and `stoch` paths, and the preferred GRU is fed a one-hot rather than a
  probability, so the two trajectories drift differently over 8 steps. Only a run answers
  that, and only after the engine question is decided.
- It is consistent with `series-sweep.md` §4.2's verdict that this series does not explain
  the *slow start* (the fast and slow arms carry the same value of it).

## Options for the deliberate round (recorded, not proposed)

1. **A key in the scoring seam.** `imagine` already draws a key per start when it builds the
   scoring closures; `registry.Statics` would carry an optional key and the row would sample
   when given one and stay mean-field on the keyless faces. Closes the divergence exactly.
   Amends FLAG (1) and touches `objectives/registry.py`, `system/core.py`,
   `train/imagine.py`. **Engine change.**
2. **A keyless one-hot rollout** (the prior's mode at each step). Restores the norms with no
   key and no seam change, but is a *different* estimator from theirs — the ensemble
   info-gain history (`ensemble-ig-sign-bug`) says not to bless such a substitution on an
   argument; it would need its own measurement. Still a change to an engine row.
3. **Keep mean-field, argue the divergence** on this measurement: a ~0.12 level shift
   absorbed by the baseline. Zero change. The gradient re-weighting above is the honest
   caveat.

## Reproduction

`probe_65.py` in the 2026-09-18 session scratch directory; the whole script is ~90 lines over
`run_mc.build`/`load_checkpoint`, `sample_windows`, `posterior_states`, `crspp_recognize`,
`roll_batch`, `update_empirical_prior` and `img_step_no_action`, with the cosine computed as
the row's `step` does. Runs read-only on CPU in about two minutes.
