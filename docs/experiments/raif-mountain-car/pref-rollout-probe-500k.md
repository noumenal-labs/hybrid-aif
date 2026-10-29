# The preferred rollout at 500k: mean-field, mode, sampled — three probes (ledger row 6.5)

**Status: measurement, not a decision.** Nothing in the library changed. Second probe on
row 6.5, after `pref-rollout-probe.md` (2026-09-18, the 100k checkpoint, one seed) measured
the divergence as a level shift and left open whether behaviour could differ. Alex, on the
500k report: "the evidence for deciding 6.5 is inconclusive. any cheaper synthetic test we
could run to test hypotheses about why it might help here?" Three, on the checkpoint we have.

**Date:** 2026-09-20. **Checkpoint:** `runs/crspp_init_100k` at step 500,736 (resumed at
`bde8659`, `winit: normal`), seeds 0 and 3 (seed 3 is the one that dipped to −104 at
250–300k and recovered). One replay batch of 16 × 64 windows per seed (`sample_windows`,
key 123), posterior states, the flat 1,024 starts, the CRSPP posterior via
`crspp_recognize`, exactly as the first probe. **Theirs:** `raif-paper` @ `9b1864e8`.

## The three hypotheses, and the probe for each

- **H1, the fix is a norm artefact and needs no key.** The first probe found the mean-field
  code vector 4.6× shorter than a one-hot, so the cosine was dominated by `deter`. A keyless
  on-manifold alternative exists: the prior's **mode** (`onehot_mode`). *P1:* roll the
  preferred trajectory three ways from the same starts — mean-field (ships), mode, sampled
  (theirs) — and read the leaf as the actor's G sees it, `cos(agent s_{k+1}, pref s_{k+1})`.
- **H2, the shift could change what the actor learns.** The actor is trained on comparisons
  between imagined trajectories from the *same* start, which a level shift across starts
  says nothing about. *P2:* 256 starts × 8 imagined trajectories each (`roll_batch`, the
  trainer's arithmetic), scored under each arm; within-start Spearman of trajectory scores,
  the fraction of starts whose best trajectory agrees, and the pooled correlation of the
  within-start-centred per-step leaf (what survives the baseline). **Control:** the same
  statistics between two independent sampled draws — the noise floor a single draw sets —
  and against the mean of 16 draws, the expected sampled leaf.
- **H3, the probability vector is off the GRU's manifold.** The world model was trained on
  one-hot codes only; feeding it a probability vector may drift. *P3:* teacher-force the
  world model over the batch, then from every posterior state roll **open loop** for 8
  steps under the recorded actions with the code carried as a sample (expected error over 8
  draws), as the mode, and as the probability vector; compare the decoder against the
  observed sequence (symlog MSE, the decoder's own loss space), the reward head against the
  recorded reward, and the open-loop `deter` against the teacher-forced posterior `deter`.

Sanity, exact, both seeds: the recomputed mean-field leaf equals the trainer's logged
`per_term["cosine_sim_to_pref_rollout"]` to the last digit (0.5911033 / 0.6113523).

## P1 — the leaf under three preferred rollouts

Mean over 1,024 starts and rollout steps k = 0…6:

| seed | mean-field (ships) | mode | sampled (1 draw) | sd of the sampled mean over 4 draws |
|---|---|---|---|---|
| 0 | 0.5911 | 0.5282 | 0.4959 | 0.0001 |
| 3 | 0.6114 | 0.5488 | 0.5221 | 0.0003 |

The gap mean-field → sampled is **−0.095 / −0.089**, down from −0.123 at 100k; the
mean-field stoch norm has grown from 1.23 to 3.02 / 3.16 (the code got sharper). Mode
closes about two thirds of it and **stops short**: mode ≠ sampled, on both seeds. And mode
is not on the sampled trajectory at all — the `deter` of the preferred rollout:

| cos(pref deter, sampled-arm deter) at step k | 1 | 3 | 5 | 7 |
|---|---|---|---|---|
| mean-field, seed 0 / 3 | 0.998 / 0.998 | 0.996 / 0.997 | 0.994 / 0.994 | 0.993 / 0.992 |
| mode, seed 0 / 3 | 0.991 / 0.992 | 0.965 / 0.966 | 0.929 / 0.935 | 0.877 / 0.898 |

The probability vector through the GRU **tracks the sampled trajectory's carry**; the
argmax one-hot drifts from it monotonically. Decomposed, the mean-field/sampled difference
is almost entirely the stoch half (stoch-only cosine 0.345 vs 0.186; deter-only 0.811 vs
0.809), and the stoch-only cosine of the sampled arm is now 0.19 against 0.04 at 100k — the
agent's and the preference model's imagined codes have come to agree partly. Across starts
the arms correlate at 0.985 (mf~sampled) and 0.960 (mode~sampled): a level shift, as before.

**H1 falls.** A keyless mode rollout does not reproduce the sampled leaf, and it is the arm
that leaves the sampled trajectory. If exact fidelity to theirs is wanted, it needs the key.

## P2 — could the arm change what the actor learns?

Seed 0, 256 starts × 8 imagined trajectories; the control rows are the point:

| pair | Spearman within start (sd) | same best trajectory | centred per-step corr |
|---|---|---|---|
| sampled A vs sampled B | 0.332 (0.361) | 0.33 | 0.393 |
| sampled A vs sampled C | 0.291 (0.364) | 0.26 | 0.393 |
| mean-field vs sampled A | 0.503 (0.333) | 0.41 | 0.625 |
| **mean-field vs mean of 16 draws** | **0.883 (0.114)** | **0.75** | **0.953** |
| mode vs mean of 16 draws | 0.715 (0.215) | 0.60 | 0.756 |
| mean-field vs mode | 0.757 (0.192) | 0.61 | 0.786 |

Seed 3 without the control rows: mf vs sampled 0.516 / 0.44 / 0.577; mode vs sampled 0.416
/ 0.38 / 0.446; mf vs mode 0.720 / 0.52 / 0.786 — the same ordering.

Read against the floor: two independent sampled draws agree with each other on the
within-start ordering at **0.29–0.33**. Mean-field agrees with a single draw at 0.50 —
*better than a draw agrees with another draw* — and with the expected sampled leaf at
**0.88** (centred per-step correlation 0.95). Mean-field is the better estimate of what
sampling does in expectation; a single draw is a noisy one; mode is in between.

And how much of the leaf is there to reorder: the within-start share of the leaf's variance
is **1.6%** under mean-field, 4.6% under a single draw, 1.7% under the 16-draw mean. The
leaf varies almost entirely *by start*, which the critic baseline absorbs on every arm.

**H2 falls, in the direction opposite to the hypothesis.** Sampling the preferred rollout
would not give the actor an ordering it lacks; it would hand it a noisier version of the
ordering mean-field already carries.

## P3 — is the probability vector off the world model's manifold?

Open loop for 8 steps from 767 / 743 teacher-forced posterior states, means over all steps:

| seed | arm | decoder symlog-MSE | reward abs error | deter cos to posterior |
|---|---|---|---|---|
| 0 | sampled (E over 8 draws) | 0.0006 | 0.0026 | 0.9935 |
| 0 | mode | 0.0026 | 0.0119 | 0.9704 |
| 0 | **mean-field** | **0.0005** | **0.0022** | **0.9960** |
| 3 | sampled | 0.0005 | 0.0023 | 0.9956 |
| 3 | mode | 0.0009 | 0.0039 | 0.9863 |
| 3 | **mean-field** | **0.0004** | **0.0016** | **0.9974** |

(Teacher-forced decoder MSE at the same targets, the filter's own: 0.00024 / 0.00020.) The
ordering holds at every step of the eight and the gap widens with k; mode's deter drifts to
0.93 / 0.97 by step 8 while mean-field's stays above 0.99.

**H3 falls.** The probability vector is the *best* single input to this GRU, ahead of the
expected error of a sample and well ahead of the mode. Nothing here says the trained world
model is hurt by mean-field rollouts; the arm that is off-manifold in the harmful sense is
the mode.

## What this settles, and what it does not

- All three hypotheses for *why sampling would help* fail on two seeds at 500k. Mean-field
  is the expectation-consistent arm: it tracks the sampled carry, orders imagined
  trajectories the way sampling does in expectation, and predicts better open loop.
- The keyless alternative that *looks* like theirs (mode) is worse than what ships on every
  measure. Option 2 of the first probe is closed by measurement, not argument.
- What sampling would still buy is **exact fidelity** to their estimator — the same
  stochastic signal, draw for draw. By P2's floor that signal is the mean-field one plus
  noise. A run could still differ through the noise itself (exploration-like effects on the
  actor), which no checkpoint probe can see; that is the honest residual.
- These are probes on *trained* checkpoints. A null on all three says sampling has no lever
  on this agent as trained, not that it never mattered during training. The 100k probe,
  read beside this one, shows the divergence shrinking as training proceeds (−0.12 → −0.09).
- Nothing here bears on whether the library should offer two filtering shapes as a general
  matter (`project-brief.md`, "one scan over a record, or two"); it removes the replication
  as a reason.

## Reproduction

`probe_65b.py` (P1–P3) and `probe_65c.py` (the P2 control) in the 2026-09-20 session
scratch directory, durable copies beside the session's memory; both are ~150-line readers
over `run_mc.build`/`load_checkpoint`, `sample_windows`, `posterior_states`,
`crspp_recognize`, `roll_batch`, `update_empirical_prior`, `img_step_no_action`,
`onehot_mode` and `RSSMCore`'s `gru`/`prior`/`decode_symlog`/`predict_reward`. About four
minutes each on CPU, read-only.
