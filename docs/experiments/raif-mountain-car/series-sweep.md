# Series sweep — every logged series, ours against theirs

*Status: **build-time analysis, not adjudication.** Nothing here changes code, config or any
adjudicated decision. Read-only against `runs/` and `raif-paper/logs/`; no training runs were
started and no git state was changed. Written 2026-09-15 in answer to Alex's "look at all the
metrics, compare against the oracle every step, and look for what's significantly different" —
and to his reason for asking it: "without it, I have to suspect something is amiss somewhere in
the library."*

**Target:** `raif-paper` @ `9b1864e8`, four seeds, `logs/oracle_state_mc_seed{0,1,2,3}`.
**Ours:** `runs/armB_init_pair_100k` (4 × 100k, `winit="normal"`, no joint gradient — the
control), `runs/armA_joint_grad_100k` (same + joint CRSPP gradient, complete to 74.7k at time of
writing), `runs/mc_2d` (the live 2M run, read only, at sha `244a3da` which predates the
initializer adoption).

---

## 0. Verdict in brief

The sweep was run hypothesis-free and it nominated something the hypothesis-driven passes could
not have found, because the thing it nominates is **one of our own runs**.

1. **float16 is discharged, measured, not argued.** Across the whole 2M run and all four oracle
   seeds, the actor, critic and info-gain optimizers record **zero** loss-scaler overflows and
   their grad scale never leaves its `1e4` ceiling. The model optimizer's first overflow is at
   step 94k–181k depending on seed, and its lifetime discard rate is 0.10–0.16% of gradient
   steps. In 0–64k — the window in question — **not one gradient step is discarded on any
   optimizer on any seed**. The "their early training discards steps stochastically" mechanism
   does not exist in their logs. §3.

2. **The behavioural gap is invisible in every training-side series we log.** Our own
   `runs/mc_2d` starts at or above oracle speed on instantaneous success rate (0.34 / 0.70 /
   0.74 / 0.97 over the four 4k bins to 16k, against the oracle's 0.20 / 0.50 / 0.82 / 0.93),
   while Arm B starts far below it (0.12 / 0.16 / 0.28 / 0.62). That is a **10 oracle-sd**
   behavioural separation between two of our own runs. On every matched training series the
   same two runs agree to within **2.7 oracle-sd or less** — decoder loss 2.7, imagined policy
   entropy 2.5, imagined prior entropy 2.4, world loss 2.3, everything else below 1.9. Our
   ~27-series telemetry cannot tell the fast run from the slow one. §5.1.

3. **What separates the two runs is the initializer adopted as divergences 17 + 26 on
   2026-09-15**, plus one intervening commit. This is stated as a measured association with a
   named confound, not as a finding against the adjudication. §5.1 gives the confound and the
   4–5 h experiment that removes it.

4. **The top-ranked matched series is real, sustained and almost certainly not the cause.** Our
   `term.cosine_sim_to_pref_rollout` sits outside the oracle's four-seed envelope in **16 of 16**
   bins from 0 to 64k, at 1.25–2.4× their `train/extr_G_crspp_mean` — the single most divergent
   comparable series in the sweep. But `mc_2d` carries the *same* elevated term (0.45 → 0.66)
   and starts at oracle speed, so the term is discriminated *against* as an explanation of the
   slow start by our own data. Same kill applies to the decoder loss. §4.1, §4.2.

5. **Three findings look like defects or gaps in our code rather than divergences from theirs**,
   and one of them corrupted this very analysis for about an hour: our `bc_logprob` is logged as
   an *unmasked* mean while the loss it is meant to instrument uses the masked product, which
   made the self-imitation channel appear 15–30× the oracle's when the derived contribution to
   the actor loss matches it to ±0.03. §6.

6. **Their four seeds are nearly the same run; ours are a lottery.** Their between-seed sd on
   instantaneous SR is 0.02–0.08 in every bin to 32k. Ours is 0.13–0.43 in Arm B and 0.03–0.35
   even in the fast run. This is the most robust "something is different" signal in the sweep and
   it is not a difference of means. §5.3.

---

## 1. Method

### 1.1 What is comparable at what resolution

Their `train/*` block is logged on a **wall-clock** cadence (`run.log_every: 300` seconds), which
on this machine is one row per ~3344 agent steps — **20 rows per seed in 0–64k**, not the 2k or
4k the brief asked for. That is the binding constraint and it is theirs, not ours. Every value in
that block is additionally a `np.nanmean` over the interval (`embodied/core/metrics.py:33`) taken
over the 1-in-10 train steps for which metrics are computed at all (`jax.metrics_every: 10`), so
a logged row is a windowed mean of ~20 samples. Our rows are **instantaneous**: `[i, -1]`, the
last gradient block of a 1024-step chunk (`run_mc.py:1250`). Binning at 4k puts ~1.25 of their
rows and ~4 of ours in a bin, which is the finest honest resolution for the training block; the
episode block is per-episode on both sides and could go finer, but is binned at 4k too so the two
halves of the sweep are read on one x-axis. Coarse pass: 20k bins to 100k.

### 1.2 The divergence measure

The oracle's own between-seed spread is the yardstick, as the brief asked. For each series and
each bin *b*:

    z_b = (mean_over_our_seeds - mean_over_their_seeds) / s_b
    s_b = max( sd_b(their four seeds, ddof=1),  median_b' sd_b'(their four seeds) )

and series are ranked by `median_b |z_b|` over the window. The floor at the *median* per-bin sd
(rather than at a fraction of the series' dynamic range, which was the first attempt) is the one
judgement call: without it a single freakishly tight bin dominates the median, and with a
range-based floor a series the oracle reproduces very tightly is *punished* for being
reproducible — exactly backwards. Reported alongside: `max|z|`, the longest run of consecutive
same-sign bins with `|z| > 2`, and the count of bins in which our four-seed mean falls outside
the oracle's four-seed min–max envelope. That last column is the one to trust when `s_b` is
small; `z` is reported to rank, the envelope count to judge.

Two series are **degenerate on their side** — their four seeds agree to within 1e-5 — so `z` runs
to four and five figures and is meaningless as a ranking number. They are flagged rather than
ranked: `train/extr_G_ig_mean` (constant 0.05775, the closed-form artefact identified in
`plateau-analysis.md` §2.4) and `train/C_dyn_loss_mean`/`C_rep_loss_mean` (pinned at the
free-bits floor).

### 1.3 Four logging conventions that must be corrected before anything is compared

These are not findings about the algorithm; they are the reason a naive sweep produces garbage.

- **Their optimizer losses are scaled.** `train/{actor,model,ig}_opt_loss` and
  `train/extr_critic_critic_opt_loss` are the loss *after* multiplication by the float16 grad
  scale (`jaxutils.py:406`), which is `1e4` throughout this window. Divide by `1e4` before
  comparing to our `actor`, `world`, `ensemble`, `critic`.
- **Their success rate is a trailing mean.** `episode/success_rate` is a 100-episode running mean
  **per environment lane** (`wrappers.py:639-721`). It is not comparable to an instantaneous
  per-bin rate, and for the oracle the two differ by up to 0.45 in 0–16k. The sweep computes
  both, and computes ours under their exact convention by replaying our per-episode `success`
  flags through the same per-lane `deque(maxlen=100)`.
- **Their `C_model_loss_mean` excludes the contrastive term.** `_metrics` receives
  `C_model_loss` *before* the contrast is added (`agent.py:232-234`), so it is
  `mean(gate·(0.5·dyn + 0.1·rep))` only. Our `crspp` *includes* it. The correct partner for
  `C_model_loss_mean` is our `crspp_kl` (Arm A only), not our `crspp`.
- **Their `relative_stability` is a derived ratio; our `stability` is the raw accumulator.**
  Theirs is `mean|rolling_mean(C) − max(C)| / max(C)` over a 100-episode window
  (`wrappers.py:742-763`); ours is `C` itself (`eval.stability_c`). Compared raw they differ by
  two orders of magnitude and a sign, which is a mapping artefact and nothing else. The sweep
  reconstructs theirs from our per-episode `C` with their exact `np.convolve` call.

---

## 2. Correspondence table

Their logs carry **246 distinct keys**; ours carry **27** (30 in Arm A, which adds the three
CRSPP sub-losses). Of their 246, 82 are `timer/*`, 14 are `replay/*` or `positive_replay/*`, and
6 are `prior_episode/*` from the collection phase. The comparison surface is what remains.

### 2.1 Matched pairs (25)

| ours | theirs | note |
|---|---|---|
| `term.cosine_sim_to_pref_rollout` | `train/extr_G_crspp_mean` | |
| `term.predicted_reward` | `train/extr_G_rew_mean` | |
| `term.model_uncertainty` | `train/extr_G_ig_mean` | divergence 20 / the sampled-std leaf; theirs degenerate |
| `value` | `train/extr_G_mean` | |
| `ret` | `train/extr_return_raw_mean` | |
| `advantage` | `train/adv_mean` | |
| `critic_value` | `train/extr_critic_mean` | |
| `critic` | `train/extr_critic_critic_opt_loss` | ÷ grad scale |
| `actor` | `train/actor_opt_loss` | ÷ grad scale |
| `ensemble` | `train/ig_opt_loss` | ÷ grad scale |
| `world` | `train/model_loss_mean` | also `train/model_opt_loss` ÷ scale, which adds the C branch |
| `decoder` | `train/state_loss_mean` | both sum over the two state dims |
| `dyn` | `train/dyn_loss_mean` | |
| `rep` | `train/rep_loss_mean` | |
| `policy_entropy` | `train/policy_entropy_mean` | |
| `prior_entropy` | `train/policy_wm_entropy_mean` | imagined, not the data batch |
| `crspp_kl` (A only) | `train/C_model_loss_mean` | see §1.3 |
| `crspp_contrast` (A only) | `train/C_contrast_loss_mean` | |
| `crspp_sim` (A only) | `train/similarity_mean` | |
| `crspp_positive` | *(derived)* `C_model_loss/(0.5·C_dyn + 0.1·C_rep)` | the gate fraction, inferred |
| `bc_logprob` × `bc_weight` | `train/actor_refresh_mean` | **not comparable as logged** — §6.1 |
| scores `length` | `episode/length` | |
| scores `return` | `episode/score` | |
| scores `success` | `episode/success_rate` | after applying their trailing-100 rule to ours |
| scores `stability` | `episode/relative_stability` | after applying their ratio formula to ours |

### 2.2 Ours-only (6)

`crspp` (their partner excludes the contrast term), `retnorm_low`, `retnorm_high`,
`rollout_weight`, `probe_ret`, `probe_critic`, `probe_value`, `distill_gap`. The retnorm band is
recoverable on their side only as an inferred ratio (`plateau-analysis.md` §2); the four probes
are ours by construction. None of these is a gap in coverage.

### 2.3 Theirs-only — the list that is itself a finding

Every one of these is a quantity we have never compared. Values in Appendix B; the ones that
matter are called out here.

**Act-time channels — we have none at all.** This is the structural gap.
- `stats/mean_log_entropy` — the **act-time** policy entropy, logged from `outs['log_entropy']`
  at `agent.py:73`, i.e. the entropy of the distribution the agent actually samples its
  environment actions from. Theirs: 1.14 → 0.43 → 0.090 → 0.007 → −0.233 over 0–4k / 4–8k /
  8–16k / 16–32k / 32–64k (σ = 0.76 → 0.37 → 0.27 → 0.24 → 0.19). We log the *imagined* policy
  entropy and nothing else.
- `episode/coverage_mean` — the fraction of a 10 × 10 grid over (position, velocity) visited per
  episode (`wrappers.py:765-904`). Theirs: 0.338 → 0.468 → 0.628 → 0.745 → 0.777. This is a
  direct measurement of early exploration, in exactly the window where we are slow, and it is not
  reconstructible from our `scores.jsonl` because we do not log per-step states.
- `episode/successfulness` — 0.0033 → 0.012 → 0.020 → 0.023 → 0.024.

**Training channels with no counterpart.**
- `train/action_{mean,std,min,max,mag}` — the imagined action tensor. Theirs: mean 0.18 → 0.61,
  std 0.99 → 0.72, mag (max |a|) 3.97 → 2.40. Unclipped samples, so the tail is visible. The
  "constant-push attractor" lives in this channel and we do not log it.
- `train/prior_ent_mean`, `train/post_ent_mean` — RSSM entropies on the **data batch** (110.6 /
  110.5 falling to 107.2 / 106.4). We log only the imagined prior entropy, so the plateau
  analysis's finding that ours collapses 7–27 nats below theirs was measured downstream of a
  quantity we never see.
- `train/reward_loss_mean` and the nine `train/reward_*` balance statistics; `train/cont_*`
  likewise. The reward head's loss falls 5.36 → 0.24 over 0–64k on their side and we have no
  per-head decomposition of `world` at all. `cont_neg_acc` is 0.0 at 4–16k and 0.82 at 32–64k —
  their continuation head does not learn to predict termination until after 16k.
- `train/{actor,critic,ig,model}_opt_grad_norm` and `_grad_steps`; the `_grad_scale` and
  `_grad_overflow` pair used in §3.
- Every `_std`, `_min`, `_max`, `_mag` companion of a series we log only as a mean — 60 keys.
  `train/extr_return_rate`, `train/policy_randomness_mean`, `train/policy_logprob_*`.
- `replay/size`, `replay/inserts`, `positive_replay/size`, `positive_replay/inserts`. M+ is
  reconstructible from our scores and **matches**: adding the ~1.85k seeded prior steps to our
  agent-collected successful-episode lengths gives 1.90k / 1.85k / 2.41k / 2.29k at 4k against
  their 1.85–2.07k, and 51.9k / 44.8k / 47.9k / 51.0k at 64k against their 54.3–56.3k. Not a
  divergence.

---

## 3. The float16 measurement

Their compute dtype is float16 with `optax.apply_if_finite` and a dynamic scaler that starts at
`1e4`, doubles after 1000 consecutive good steps, halves on any non-finite gradient, and is
clipped to `[1e-4, 1e4]` (`jaxutils.py:391-449`). Two channels measure it: `*_grad_overflow`,
which is `(~finite)` as a float and therefore an overflow *rate* once `nanmean`ed over the log
window, and `*_grad_scale`, which is a **state variable** and so detects an overflow even on a
train step whose metrics were not sampled. Both were read over the full 2M run on all four seeds.

| optimizer | rows with scale < 1e4 (of ~650) | first such step | min scale | nonzero overflow rows | lifetime discarded steps |
|---|---|---|---|---|---|
| actor | 0 / 0 / 0 / 0 | — | 1e4 | 0 | 0 |
| critic | 0 / 0 / 0 / 0 | — | 1e4 | 0 | 0 |
| info-gain | 0 / 0 / 0 / 0 | — | 1e4 | 0 | 0 |
| model | 388 / 546 / 558 / 451 | 181k / 155k / 107k / 94k | 5000 / 1250 / 1250 / 1250 | 4 / 12 / 7 / 4 | 123 / 197 / 205 / 182 |

Per-window overflow rate over 0–4k, 4–8k, 8–16k, 16–32k, 32–64k, 64–100k is `0.0000` in every
cell of a 4 optimizers × 6 windows × 4 seeds table. The model optimizer's lifetime discard count
is 123–205 gradient steps out of ~124,800, i.e. **0.10–0.16%**, and every one of them falls after
94k. Cross-check: final `grad_steps` is 124,697–124,880 against an expected `step/16` of
124,820–124,985, a 0.08–0.24% shortfall consistent with those counts.

**What this means.** The argument shaped like 17/26 — *their arithmetic is noisier and the noise
is the point* — needs a mechanism, and the mechanism named in the register was stochastic step
discarding plus gradient quantization. The discarding half is now measured at exactly zero in the
window we care about. The quantization half survives: their gradients are still computed in
float16 on a loss pre-multiplied by `1e4`, and the fact that the scaler never fires means those
gradients sit comfortably inside float16's range, not that they are unquantized. But a scaler
pinned at its ceiling for 2M steps is evidence of a *well-conditioned* float16 computation, not a
noisy one. The register's "suspect of last resort" is now a suspect with its named evidence
channel read and found empty; what remains of it is a rounding argument with no log-side support.
Recommend it be recorded as **measured and demoted**, not closed — closing it needs a run.

---

## 4. The ranked table

Full table in Appendix A. Arm B against the oracle, 4k bins; `med|z|` over 0–16k and over 0–64k;
`out` is bins where our four-seed mean is outside their four-seed envelope; `run` is the longest
same-sign run of `|z| > 2`.

| # | series | med\|z\| 0–16k | med\|z\| 0–64k | run | out (0–64k) | ours / theirs, 0–16k |
|---|---|---|---|---|---|---|
| — | `G: info-gain term` | *degenerate* | *degenerate* | 16 | 16/16 | −4.7e−5 / 0.0578 |
| 1 | `G: crspp term` | 16.8 | 8.16 | **16** | **16/16** | 0.615 / 0.450 |
| 2 | `crspp gate fraction` | 26.1 | 2.82 | 4 | 12/16 | 0.336 / 0.373 |
| 3 | `G: reward term` | 11.1 | 3.86 | 14 | 16/16 | −0.0992 / −0.0613 |
| 4 | `world loss (total)` | 9.36 | 5.57 | 13 | 16/16 | 4.14 / 4.84 |
| 5 | `SR (instantaneous)` | 8.11 | 1.58 | 4 | 13/16 | 0.293 / 0.613 |
| 6 | `SR (trailing 100)` | 7.09 | 1.91 | 7 | 14/16 | 0.159 / 0.349 |
| 7 | `episode length` | 6.93 | 4.97 | 8 | 12/16 | 87.6 / 68.8 |
| 8 | `episode score` | 6.63 | 4.60 | 8 | 12/16 | −173 / −137 |
| 9 | `critic loss` | 6.62 | 2.09 | 4 | 14/16 | 9.79 / 9.02 |
| 10 | `G total (per step)` | 5.87 | 1.94 | 4 | 9/16 | 0.516 / 0.447 |
| 11 | `lambda-return (raw)` | 5.82 | 2.29 | 5 | 12/16 | 2.10 / 1.79 |
| 12 | `critic value` | 5.68 | 0.97 | 1 | 13/16 | 0.0319 / 0.0232 |
| 13 | `decoder loss` | 4.41 | 1.39 | 7 | 11/16 | 0.0406 / 0.281 |
| 14 | `ensemble / ig loss` | 4.12 | 0.39 | 1 | 5/16 | 3828 / 3833 |
| 15 | `advantage (normed)` | 3.83 | 2.65 | 7 | 13/16 | 0.796 / 0.676 |
| 16 | `actor loss` | 2.23 | 1.93 | 4 | 12/16 | −0.692 / −0.503 |
| 17 | `relative stability` | 10.4 | 1.41 | 3 | 11/16 | 0.599 / 0.476 |
| 18 | `policy entropy (imag)` | 1.16 | 0.58 | 1 | 5/16 | 0.359 / 0.389 |
| 19 | `dyn KL` / `rep KL` | 0.40 | 2.73 | 6 | 12/16 | 1.0010 / 1.0009 |
| 20 | `wm entropy (imag)` | 0.22 | 0.23 | 0 | 5/16 | 110.6 / 110.6 |

Arm A's ordering is the same to within one rank everywhere, with its three extra CRSPP series
landing at `crspp gated KL` 18.6, `crspp contrast` 6.00, `crspp sim (data)` 1.78 over 0–16k. The
joint CRSPP gradient does not move any series by more than its own seed noise in this window, and
it does not move behaviour: Arm A's instantaneous SR is 0.14 / 0.13 / 0.19 / 0.50 against Arm B's
0.12 / 0.16 / 0.28 / 0.62, i.e. slightly worse and well inside our own seed spread.

### 4.1 The top series, and what each one means

**1. `G: crspp term` — the only series outside their envelope in all 16 bins, same sign.** This
is the cosine similarity between the imagined rollout and the preference model's rollout,
`crsppfn` on their side, and it is the dominant term of G on both: their G total is 0.25–0.55 and
their crspp term is 0.19–0.54 of it. Per-bin `z`: +22.5, +12.7, +15.9, +16.5, +16.3, +10.6,
+5.8, +6.1, +7.3, +7.9, +12.4, +6.1, +14.8, +7.9, +7.4, +7.3. Ours 0.46 → 0.68 against their
0.19 → 0.54. The first bin is the interesting one: at 0–4k, before any behavioural difference
exists (episode length 94.7 / 93.9, instantaneous SR 0.12 / 0.20), our term is already **2.4×**
theirs. So this is not downstream of the behavioural gap. **It is nonetheless discriminated
against as a cause**, see §4.2. Mechanically it says our preference model's rollout sits closer
to the agent's own imagined rollout than theirs does, from initialization onward; the data-batch
similarity (`crspp_sim` vs `similarity_mean`) matches from 4k (0.48/0.50, 0.52/0.51, 0.53/0.52)
while the *imagined* term stays 1.25× high, so the divergence is in the imagined evaluation, not
in how well the preference model is trained. Worth a closed-form probe (§5.2), not a run.

**2. `crspp gate fraction`.** Ours 0.71 / 0.34 / 0.17 / 0.13 against theirs (inferred from
`C_model_loss / (0.5·C_dyn + 0.1·C_rep)`, which is exact while the KLs sit on the free-bits
floor) 0.81 / 0.27 / 0.21 / 0.20. Our gate collapses to 0.11–0.15 by 16–20k while theirs holds
0.17–0.20. This is the fraction of replay rows carrying a positive preference label — i.e. how
much data the preference model and the self-imitation term are allowed to learn from. It is
**downstream of the behavioural gap**: fewer successful episodes, fewer positive labels. The fast
run confirms this — `mc_2d`'s gate is 0.68 / 0.30 / 0.23 / 0.21 / 0.23, tracking the oracle.
Consequence, not cause.

**3. `G: reward term` and 11. `lambda-return`.** Ours more negative and larger respectively.
Both are functions of where the agent is in the episode; longer episodes mean more accumulated
−1s. Consequence of the behavioural gap.

**4. `world loss (total)`.** Ours 4.14 against 4.84 at 0–16k, 13 consecutive bins with `z < −2`.
Our world model's total loss is below theirs for the whole window. Same family as 13.

**5–8. `SR`, `episode length`, `episode score`.** The gap itself, in four presentations. Not
causes.

**9. `critic loss` and 12. `critic value`.** Ours higher. The critic loss is a symlog-discrete
cross-entropy against a λ-return that is itself larger on our side; the critic value tracks it.
Downstream of 11.

**10. `G total`.** Sum of 1, 3 and the info-gain leaf. Driven by 1.

**13. `decoder loss`.** Ours 0.104 / 0.0317 / 0.0157 / 0.0108 against theirs 1.047 / 0.0416 /
0.0220 / 0.0142. The first bin is a **10× difference**: our observation model is essentially
solved by 4k while theirs is still at 1.05. This is the signature the initializer adoption was
meant to close and it does close from 8k (band 1 of the Arm A design, 0.71–1.13), but the 0–4k
bin sits outside that band's window and is the largest single-bin divergence in the sweep after
series 1. Discriminated against as a cause in §4.2.

**15. `advantage` and 16. `actor loss`.** At 0–4k, `z` = +4.24 and −6.51: our advantage is 1.211
against their 0.655 and our actor loss is −0.840 against their −0.038. Both converge by 4–8k.
The actor-loss difference decomposes cleanly (§6.1): −adv accounts for −0.56 of it and the
self-imitation term for +0.25, and the decomposition reproduces their own `actor_refresh_mean` to
±0.06, which is how the §6.1 defect was caught.

**19. `dyn KL` / `rep KL`.** Matched to 4 decimals through 16k (1.00000 / 1.00032 / 1.00123 /
1.00225 against 1.00000 / 1.00007 / 1.00088 / 1.00235) and *below* theirs from 24k (`z` −1.9 to
−3.8). The plateau analysis found ours 4–6× above the floor where theirs was on it; under the
adopted initializer that is fixed and slightly overshot. Recorded because a closed divergence
that changed sign is worth knowing about.

**20. `wm entropy (imag)`.** Within a nat everywhere in 0–64k, `z` never above 1.2, zero bins
outside the envelope in 0–16k. Band 2 of the Arm A design holds.

### 4.2 The discriminator that kills series 1 and 13

`runs/mc_2d` is our own agent at the pre-adoption initializer. Its instantaneous success rate is
0.34 / 0.70 / 0.74 / 0.97 over the four bins to 16k against the oracle's 0.20 / 0.50 / 0.82 /
0.93 — at or above the oracle in three of four bins. And it carries:

| series, 0–4k … 12–16k | `mc_2d` (fast) | Arm B (slow) | oracle |
|---|---|---|---|
| `G: crspp term` | 0.447 0.629 0.662 0.649 | 0.461 0.643 0.675 0.682 | 0.194 0.533 0.536 0.538 |
| `decoder loss` | 0.0579 0.0218 0.0105 0.0063 | 0.104 0.0317 0.0157 0.0108 | 1.047 0.0416 0.0220 0.0142 |
| `world loss` | 5.91 4.83 3.66 2.57 | 5.88 4.73 3.52 2.43 | 7.47 4.93 3.94 3.01 |
| `value` (G total) | 0.440 0.599 0.572 0.403 | 0.454 0.611 0.580 0.418 | 0.248 0.566 0.528 0.445 |
| `advantage` | 1.200 0.771 0.662 0.540 | 1.211 0.761 0.657 0.557 | 0.655 0.772 0.667 0.610 |

The fast run is *further* from the oracle than the slow run on the decoder loss and on the world
loss, and identical to it on the crspp term. A quantity that takes the same value in a run that
matches the oracle's behaviour and a run that does not cannot be what explains the difference.
Both series are **real divergences from the target** and should stay on the register; neither is
the answer to "why do we start slow".

---

## 5. What the sweep nominates

Three candidates, in order of the strength of the evidence that nominates them.

### 5.1 The adopted initializer (divergences 17 + 26) — strongest evidence, named confound

**Evidence.** Two of our runs, same seeds, same fixtures, same four lanes, same hyperparameters,
differ by 10 oracle-sd in instantaneous SR through 16k and by **at most 2.7 oracle-sd on any
matched training series**. Per-seed first success: `mc_2d` 1452 / 2372 / 3972 / 2180; Arm B 1816
/ 17556 / 2180 / 1988; Arm A 1808 / 19720 / 2184 / 2016; oracle 2236 / 1996 / 2628 / 1568. Arm B
and Arm A each lose one seed entirely for ~18k steps; `mc_2d` and the oracle lose none.
Separation between the two runs in oracle-sd units, 0–16k median: SR 10.0; then `dyn` 8.4 (on a
quantity pinned at the free-bits floor where the oracle's sd is 3e−4 and the absolute difference
is under 0.011 nats — a large `z` on a small thing), `decoder` 2.7, `policy_entropy` 2.5,
`prior_entropy` 2.4, `world` 2.3, and everything else below 1.9.

**Confound, stated plainly.** `mc_2d` is at sha `244a3da` and Arm B at `859faf0` + the `winit`
working-tree edits. The one intervening commit is *"imagine and fit_ensemble raise on a missing
adam state instead of re-seeding it"*. If that silent re-seed was firing in `mc_2d`, then
`mc_2d`'s optimizer state was being reset and it started fast **despite** a defect — which would
make the comparison stranger, not weaker, but it is untested either way. The two runs also differ
in length (2M vs 100k), which is irrelevant below 100k.

**Relation to the adjudication.** Divergence 17 + 26 was closed by Alex on 2026-09-15 *on a
trajectory*, against seven pre-stated bands, and the Arm A design already records that "the
initializer change slowed our start while the oracle, with the same initializer, starts fast".
This section adds one thing to that record: the slowdown is now measured against a same-agent
control rather than against the oracle, and it is **not accompanied by a training-side signature**
— which is what makes it worth re-opening as an experiment rather than as an argument. Nothing
here is a re-proposal of the reverted variant; it is a request for the control the adoption never
had.

**Cheapest experiment.** Arm C: Arm B's exact sha and tree with the `winit` edits reverted, 4
seeds × 100k, `fixtures: true`, everything else identical. ~4–5 h of shared CPU — the same cost
as Arm B, and it is the only unconfounded read on the initializer's behavioural cost.

*Pre-registered bands (Arm B is the control; oracle values are 4-seed means at 4k bins):*

| # | series | Arm B | oracle | Arm C passes if |
|---|---|---|---|---|
| 1 | instantaneous SR, 0–4k / 4–8k / 8–16k | 0.12 / 0.16 / 0.45 | 0.20 / 0.50 / 0.87 | ≥ 0.20 / ≥ 0.40 / ≥ 0.75 |
| 2 | first success, per seed | 1.8k, 17.6k, 2.2k, 2.0k | 2.2k, 2.0k, 2.6k, 1.6k | every seed < 8k |
| 3 | between-seed sd of instantaneous SR at 4–8k | 0.16 | 0.04 | ≤ 0.15 |
| 4 | decoder-loss ratio ours/oracle, 8–64k | 0.71–1.13 | 1.0 | reported, not gated — the point is that it need not hold |
| 5 | imagined prior entropy at 60–64k | within a nat | 105.2 | ≥ 103.5 (band 2 must not regress) |

Row 4 is deliberately ungated: if Arm C is fast *and* its decoder ratio is 0.4–0.5, that settles
whether the world-model sharpness signature and the early behaviour are coupled at all — which
is the question band 1 of the Arm A design was built on and could not answer with one arm.

### 5.2 The CRSPP leaf's imagined evaluation — strong series evidence, no run needed

**Evidence.** §4.1 series 1: outside the oracle's envelope in 16 of 16 bins, always the same
sign, 2.4× at 0–4k before any behavioural difference exists, and *not* explained by the
preference model being better trained (the data-batch similarity matches from 4k while the
imagined term does not). §4.2 says it is not the cause of the slow start, which leaves it as an
unexplained standing divergence from the target — the kind that has twice turned out to be a
transcription of the wrong call site.

**Cheapest experiment.** None that costs compute. A read-only probe on any existing checkpoint:
evaluate our `crsppfn` partner and theirs on one identical imagined rollout and diff the leaf
term by term, the way the info-gain leaf was diffed in September. Call it an hour. The specific
thing to check is the argument order and indexing at the *call site* — `crsppfn = lambda s, p:
wm.crspp_traj_sim(s, p)[1:]` takes both the trajectory and the preference trajectory, and the
`[1:]` is divergence 20's slice, which is argued for the info-gain leaf and has never been
separately measured on this one.

### 5.3 Act-time telemetry — the gap the sweep found by exhaustion

**Evidence.** Two negatives, which together are a positive. First, no training-side series
separates our fast run from our slow one by more than 2.7 oracle-sd while behaviour separates
them by 10. Second, the oracle logs two act-time channels — `stats/mean_log_entropy` and
`episode/coverage_mean` — and we log **zero**. Every one of our 27 series is a training-time
quantity computed inside `train_step`. The gap we are chasing is behavioural and concentrated in
the first 16k, and the instrument that would see it directly is the one we do not have. Add to
this that their four seeds have an sd of 0.02–0.08 on SR and ours 0.13–0.43: a seed lottery on
first contact with the goal is an *exploration* phenomenon, and exploration is act-time.

**Cheapest experiment.** No training run. Add two fields to `scores.jsonl` — per-episode mean
act-time policy entropy, and their `coverage_mean` on the same 10 × 10 (position, velocity) grid
with their bin edges — then read them off the *next* run that happens for any other reason. The
coverage computation is a `np.digitize` over the episode's state column in the collection loop;
the entropy is already computed at act time and discarded. This makes the next 100k run answer a
question it would otherwise not, at roughly zero marginal cost, and it is the prerequisite for
ever comparing §5.1's Arm C to the oracle on the axis that matters.

---

## 6. Defects and gaps in our code, as distinct from divergences from theirs

### 6.1 `bc_logprob` is logged with the wrong reduction — measured, and it misled this analysis

`run_mc.py` logs `bc_logprob = mean(start_logprob)` and `bc_weight = mean(flags)` as two separate
means (`imagine.py:500-501`). The loss those two instrument is `mean(flags · start_logprob)`, and
their partner `train/actor_refresh_mean` is `mean(−logpi2 · can_self_imitate) · 0.5`. A product of
means is not the mean of a product, and here the two disagree by an order of magnitude: naively
`−0.5 · bc_weight · bc_logprob` gives 0.52 / 0.66 / 0.52 / 0.56 through 16k against their
`actor_refresh_mean` of 0.587 / 0.091 / 0.034 / 0.020, which ranks the self-imitation channel
second in the whole sweep at med|z| 385 and reads as "our BC term is 15–30× theirs from 8k on".

It is not. Decomposing the actor loss on both sides with the same formula —
`BC = actor_loss + adv_mean + actent·policy_entropy + wment·wm_entropy` — recovers their
`actor_refresh_mean` to ±0.06 on their side (0.651 vs 0.587, 0.102 vs 0.091, 0.044 vs 0.034,
0.028 vs 0.020, and within 0.01 in every bin from 20k), and gives ours as 0.404 / 0.106 / 0.030 /
0.012, then negative from 16k just as theirs is. **Ours matches theirs to ±0.03 from 4k on and is
0.25 *lower* at 0–4k.** The self-imitation channel is not a divergence; the logging is a defect.

Why it matters beyond this document: `bc_logprob` reading −6 to −10 nats is the kind of number
that invites a mechanistic story ("our actor is 4.6σ away from its own stored expert actions"),
and that story is false. The fix is one line — log `mean(flags · start_logprob)` alongside, or
instead of, the unmasked mean — and it makes the single most suspicious-looking channel in our
telemetry directly comparable to theirs. This is the fourth investigation in a row to find a
defect, and the third to find it in the *instrumentation* rather than in the algorithm.

### 6.2 Two mappings that look like defects and are not, but are undocumented

`scores.jsonl`'s `stability` is their `C` accumulator, not their `relative_stability`; compared
raw the two differ by 0.9 versus −82. And our `crspp` includes the contrastive term while their
`C_model_loss_mean` excludes it. Both are correct on our side and both produce a spurious
top-of-table divergence if paired naively. Reconstructed properly, relative stability is 0.918 /
0.726 / 0.497 / 0.257 against theirs 0.907 / 0.620 / 0.304 / 0.075 — a real gap, but one entirely
accounted for by episode length, and with our between-seed sd at 0.115–0.266 against theirs at
0.002–0.023. Recording the mapping here so the next sweep does not re-find it.

### 6.3 Coverage gaps, ranked by what they would have told us

No act-time channel of any kind (§5.3). No per-head decomposition of `world` — their
`reward_loss_mean` falls 5.36 → 0.24 over 0–64k and we cannot say what ours does. No gradient
norms for any of the four optimizers, which is the cheapest possible check on whether a
parameter group is being trained at all. No `std`/`min`/`max` companions for any series, so every
comparison in this document is a comparison of means and a distribution change that preserves the
mean is invisible to us. No RSSM entropies on the data batch.

None of these is a bug. All of them are reasons a sweep of this kind has less to work with on our
side than on theirs, and the first two are cheap.

---

## 7. What the sweep did *not* nominate

Recorded because a hypothesis-free sweep's negatives are worth as much as its positives, and
because these were the things a hypothesis-driven pass would have gone looking for.

- **float16.** §3. Zero overflows in the window; the named mechanism is absent.
- **The self-imitation / M+ channel.** §6.1 for the loss term (matches to ±0.03 from 4k),
  §2.3 for the buffer (matches to within the seeded prior at every bin).
- **The imagined world-model entropy.** Zero bins outside their envelope in 0–16k.
- **The free-bits floor.** Matched to four decimals through 16k under the adopted initializer.
- **The return normalizer.** `retnorm_low`/`retnorm_high` have no direct partner, but the
  advantage they produce matches theirs to `z` < 1 from 4–8k onward; the 0–4k `z` of +4.2 is one
  bin and is gone by the next.
- **The joint CRSPP→encoder gradient (Arm A).** No matched series moves by more than its own seed
  noise in 0–64k, and behaviour is unchanged within our seed spread.
- **Everything below rank 16.** Eight matched series sit at `med|z|` under 2.3 over 0–64k with no
  sustained sign, which is the sweep saying "nothing here" about the entire value channel, the
  actor's entropy, and the world model's KL structure.

---

## Appendix A — full ranked table

Arm B against the oracle, all 25 matched pairs, 4k bins, sorted by `med|z|` over 0–64k. `sd_them`
is the median per-bin between-seed sd on their side; `sd_us` is ours over 0–16k. The last two
columns are the whole §5.3 argument in numbers.

| series | class | med\|z\| 0–64k | max\|z\| | run | outside | ours 0–16k | theirs 0–16k | sd_them | sd_us |
|---|---|---|---|---|---|---|---|---|---|
| `G: info-gain term` | train | *degenerate* | — | 16 | 16/16 | −4.74e−5 | 0.05775 | 5.0e−6 | 3.1e−6 |
| `G: crspp term` | train | 8.16 | 20.47 | 16 | 16/16 | 0.6150 | 0.4501 | 0.0131 | 0.0260 |
| `world loss (total)` | train | 5.57 | 14.24 | 13 | 16/16 | 4.142 | 4.838 | 0.0070 | 0.0360 |
| `episode length` | episode | 4.97 | 14.28 | 8 | 12/16 | 87.56 | 68.77 | 2.13 | 11.7 |
| `episode score` | episode | 4.60 | 13.88 | 8 | 12/16 | −172.7 | −136.6 | 4.30 | 23.7 |
| `G: reward term` | train | 3.86 | 14.56 | 14 | 16/16 | −0.0992 | −0.0613 | 0.0077 | 0.0031 |
| `crspp gate fraction` | train | 2.82 | 11.99 | 4 | 12/16 | 0.3359 | 0.3727 | 0.0080 | 0.0346 |
| `dyn KL` | train | 2.73 | 4.64 | 6 | 12/16 | 1.00100 | 1.00090 | 0.0033 | 0.00062 |
| `rep KL` | train | 2.73 | 4.64 | 6 | 12/16 | 1.00100 | 1.00090 | 0.0033 | 0.00062 |
| `advantage (normed)` | train | 2.65 | 12.24 | 7 | 13/16 | 0.7964 | 0.6761 | 0.0169 | 0.0197 |
| `lambda-return (raw)` | train | 2.29 | 12.42 | 5 | 12/16 | 2.095 | 1.794 | 0.0738 | 0.105 |
| `critic loss` | train | 2.09 | 4.74 | 4 | 14/16 | 9.793 | 9.015 | 0.0487 | 0.0297 |
| `G total (per step)` | train | 1.94 | 14.41 | 4 | 9/16 | 0.5157 | 0.4466 | 0.0143 | 0.0255 |
| `actor loss` | train | 1.93 | 42.71 | 4 | 12/16 | −0.6916 | −0.5031 | 0.0182 | 0.0272 |
| `SR (trailing 100)` | episode | 1.91 | 17.48 | 7 | 14/16 | 0.1589 | 0.3488 | 0.0149 | 0.135 |
| `SR (instantaneous)` | episode | 1.58 | 17.19 | 4 | 13/16 | 0.2934 | 0.6125 | 0.0180 | 0.244 |
| `relative stability` | episode | 1.41 | 7.38 | 3 | 11/16 | 0.5994 | 0.4764 | 0.0261 | 0.148 |
| `decoder loss` | train | 1.39 | 5.48 | 7 | 11/16 | 0.0406 | 0.2813 | 0.00067 | 0.0093 |
| `critic value` | train | 0.97 | 2.79 | 1 | 13/16 | 0.0319 | 0.0232 | 0.0120 | 0.0016 |
| `policy entropy (imag)` | train | 0.58 | 2.45 | 1 | 5/16 | 0.3586 | 0.3894 | 0.0643 | 0.152 |
| `ensemble / ig loss` | train | 0.39 | 44.69 | 1 | 5/16 | 3828 | 3833 | 0.0917 | 0.106 |
| `wm entropy (imag)` | train | 0.23 | 1.20 | 0 | 5/16 | 110.6 | 110.6 | 0.329 | 0.0246 |

Arm A only (the three CRSPP sub-losses, 0–64k): `crspp gated KL` 7.33 (max 21.4, 15/16 outside),
`crspp contrast` 5.12 (16.5, 14/16), `crspp sim (data)` 1.37 (9.66, 12/16). The
`bc_logprob`/`actor_refresh` pair is excluded from the ranking for the reason in §6.1; its
derived replacement matches to ±0.03 and would rank last.

## Appendix B — theirs-only inventory, early values

Four-seed means over 0–4k / 4–8k / 8–16k / 16–32k / 32–64k, with the 0–16k between-seed sd.

| series | 0–4k | 4–8k | 8–16k | 16–32k | 32–64k | sd |
|---|---|---|---|---|---|---|
| `episode/coverage_mean` | 0.338 | 0.468 | 0.628 | 0.745 | 0.777 | 0.025 |
| `episode/successfulness` | 0.0033 | 0.0120 | 0.0196 | 0.0230 | 0.0241 | 0.0010 |
| `stats/mean_log_entropy` | 1.140 | 0.428 | 0.0897 | 0.0073 | −0.233 | 0.061 |
| `train/action_mean` | 0.183 | 0.525 | 0.611 | 0.538 | 0.422 | 0.065 |
| `train/action_std` | 0.993 | 0.794 | 0.718 | 0.776 | 0.856 | 0.024 |
| `train/action_mag` | 3.97 | 3.24 | 2.99 | 2.67 | 2.40 | 0.081 |
| `train/policy_randomness_mean` | 0.910 | 0.574 | 0.364 | 0.322 | 0.247 | 0.030 |
| `train/policy_logprob_mean` | −1.213 | −0.438 | 0.047 | 0.143 | 0.314 | 0.070 |
| `train/prior_ent_mean` | 110.6 | 110.7 | 110.5 | 110.1 | 107.2 | 0.033 |
| `train/post_ent_mean` | 110.5 | 110.0 | 109.8 | 109.4 | 106.4 | 0.093 |
| `train/C_prior_ent_mean` | 110.6 | 110.7 | 110.6 | 110.4 | 109.8 | 0.031 |
| `train/C_post_ent_mean` | 110.5 | 109.7 | 109.4 | 109.7 | 109.2 | 0.078 |
| `train/reward_loss_mean` | 5.357 | 4.239 | 2.813 | 0.841 | 0.242 | 0.021 |
| `train/reward_pred` | −0.0026 | −0.024 | −0.108 | −0.943 | −1.765 | 0.0016 |
| `train/reward_avg` | −1.927 | −1.960 | −1.958 | −1.946 | −1.935 | 0.0018 |
| `train/cont_loss_mean` | 0.461 | 0.052 | 0.045 | 0.032 | 0.015 | 0.039 |
| `train/cont_pos_acc` | 0.744 | 1.000 | 1.000 | 0.998 | 0.997 | 0.041 |
| `train/cont_neg_acc` | 0.297 | 0.000 | 0.000 | 0.310 | 0.822 | 0.045 |
| `train/cont_pred` | 0.703 | 0.979 | 0.985 | 0.983 | 0.982 | 0.019 |
| `train/extr_return_normed_mean` | 0.590 | 0.602 | 0.472 | 0.279 | 0.464 | 0.020 |
| `train/extr_return_rate` | 0.457 | 1.000 | 0.967 | 0.786 | 0.904 | 0.015 |
| `train/extr_G_std` | 0.042 | 0.051 | 0.051 | 0.099 | 0.527 | 0.0026 |
| `train/adv_std` | 0.325 | 0.359 | 0.302 | 0.346 | 0.318 | 0.011 |
| `train/state_loss_std` | 0.994 | 0.061 | 0.029 | 0.020 | 0.016 | 0.052 |
| `train/actor_opt_grad_norm` | 1.214 | 0.613 | 0.494 | 0.355 | 0.208 | 0.209 |
| `train/extr_critic_critic_opt_grad_norm` | 1.987 | 3.731 | 3.718 | 3.634 | 3.869 | 0.135 |
| `train/ig_opt_grad_norm` | 17.14 | 1.638 | 1.023 | 0.959 | 0.976 | 0.236 |
| `train/model_opt_grad_norm` | 10.09 | 4.207 | 4.588 | 2.420 | 2.160 | 0.602 |
| `replay/size` | 3113 | 7890 | 12900 | 24700 | 50690 | 47.6 |
| `positive_replay/size` | 1908 | 2839 | 5776 | 16450 | 41040 | 187 |
| `fps` | 11.08 | 11.09 | 11.12 | 11.17 | 10.56 | 0.036 |

The 82 `timer/*` keys and the `_min`/`_max`/`_mag` companions not listed above were enumerated and
compared for completeness; none carries algorithmic content, and no `timer/*` series is comparable
across two machines.

## Appendix C — reproduction

Scripts live in this session's scratch directory and are not checked in; each reads only
`runs/*/metrics.jsonl`, `runs/*/scores.jsonl` and
`raif-paper/logs/oracle_state_mc_seed{0..3}/metrics.jsonl`.

- `extract.py` — streams both sides into one pickle, capped at 120k steps.
- `enum_keys.py` — the 246-key enumeration with per-key row counts and step ranges.
- `f16_full.py` — §3, the loss-scaler audit over the full 2M run.
- `rank2.py` — §4 and Appendix A, the ranked tables at both windows.
- `derived.py` — §6.1, the actor-loss decomposition on both sides.
- `episodes.py`, `seeds.py` — the episode channels under matched conventions, per-seed
  trajectories, M+ reconstruction, relative-stability reconstruction.
- `theirsonly.py` — Appendix B.
