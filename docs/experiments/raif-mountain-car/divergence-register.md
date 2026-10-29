# Divergence register — R-AIF state-based Mountain Car

*The front door. One stable ID per divergence from the target, its current
status, and a pointer to wherever the detail lives. Created 2026-09-07 because
the record had grown into three files under three numbering schemes and "what
happened to 17?" had no single place to look.*

**Target:** `raif-paper` @ `9b1864e8fab9d1becd6dd248f2c590d06ff22848`.
**Doctrine:** the oracle is the code; where their paper and their code disagree,
the code wins (which is how the M+ alternation ruling was settled).
**Acceptance:** `amortization-p0-addendum.md` §7 — converged ACR −68.1…−69.5 on
best-200k windows, judged window-honestly, with the trajectory *shape* part of
the target.

**Close-out (2026-10-10).** The 8-seed comparison against the oracle ended inconclusive by
its pre-registered stopping rule: final difference −1.01 return points, 95% CI [−3.19, +1.04],
with the §7 windows within half a point on both sides. The arc is closed; the table, the
stopping-rule checks, the trajectory shape and the open hypotheses (act-time sampling,
float16) are in `acceptance-report.md`.

## Numbering, and why it is what it is

Three schemes accumulated, and renumbering would break every citation in
the code comments and the design records. So the IDs stay as issued and
this table is the index:

| IDs | issued by | detail lives in |
|---|---|---|
| **1–13** | chunk 2d's divergence hunt (2026-09-01/02) | `docs/design/amortization.md`, chunk-2d note |
| **14–19** | the same hunt's open flags | same, and their dispositions inline |
| **20–28** | the training execution-trace audit (2026-09-06) | `execution-ledger-training.md` |
| **D1–D9** | the acting execution-trace audit (2026-09-06) | `execution-ledger-acting.md` |
| §(e) rows | the resolved-config audit (2026-09-02) | `amortization.md` §(e), 54 keys, every one dispositioned |

New divergences get the next free number in the 20+ range and are recorded here
first. **29** is the first issued this way (below).

## Open and argued — the only rows that need reading

Everything else is closed; the closed history is indexed above rather than
duplicated here.

| ID | what | status |
|---|---|---|
| **17 + 26** | their `winit: normal` / `fan: avg` initializer and biasless normed layers on the six non-`MLPConditional` blocks | **ADOPTED 2026-09-15, THEN MEASURED AS A TRADE 2026-09-16** — the default is under review (`ctrl_winit_default_100k` vs `armB_init_pair_100k`). It **causes our slow start**: seed-mean SR over the first three windows 0.12/0.16/0.45 with it against 0.34/0.71/0.86 without (oracle 0.20/0.50/0.82), isolated at one commit with the adam-raise confound measured inert. Mechanism, from the new act-time telemetry: it leaves the actor wide and uncommitted (entropy 0.320 at 12–16k vs the oracle's 0.098) and covers less state in every bin. It also **fixes** the 24–32k length drift (45.2, the oracle's 45.3, against 55.3 without) and cuts 16–100k collapses 3→1 of 4. The oracle beats us in every window under both settings. Original adoption note: **CLOSED 2026-09-15 (Alex), on a trajectory.** Reverted on 2026-09-07 because it regressed an open-loop world-model probe 2.6×; `plateau-analysis.md` then showed our world model was learning *faster and sharper* than theirs (decoder loss 0.42–0.52×, KL above free-bits 4–6×, prior entropy 7–27 nats lower) and that the sharpness fed the actor's constant-push attractor — the probe was measuring the property that hurt. Arm B (4 × 100k, seven pre-stated bands): signature closed (decoder ratio 0.71–1.13, KL ≈ 1×, prior entropy within a nat), early collapses 3 → 1 of 4, length at 24–32k identical to the oracle's; drift delayed not removed. The parity recipe opts in (`raif_mountain_car(winit="normal")`); engine defaults unchanged, so the probe tests still measure the default. |
| **20** | their info-gain leaf is read one rollout step later than ours (both their `rewfn` and `igfn` slice `[1:]`) | **ARGUED, KEEP OURS** (Alex, 2026-09-07). Their indexing needs `a_{k+1}` at index k, so expressing it requires either a non-causal per-step term context or a rollout special case the term-tree grammar cannot express; our contract's guarantee that every row at index k is a functional of the *same* event is what makes the fold to G coherent. **Measured** (the lesson from the sampled-std mistake): 12.7% of the leaf's magnitude, but 0.39% of the normalized advantage's p5–p95 spread, peaking at 1.64% when member spread ≈ σ. Not a pure relabelling — theirs drops `L[0]` and introduces `L[8]`. |
| **float16** | their compute dtype; ours is uniformly float32 | **MEASURED AND DEMOTED 2026-09-15** (`series-sweep.md`): across 2M steps and all four of their seeds, the actor, critic and info-gain loss scalers record **zero** overflows and never leave the `1e4` ceiling; the model optimizer's first overflow is at 94k–181k, lifetime discard rate 0.10–0.16%; in 0–64k not one gradient step is discarded on any optimizer on any seed. The "their training is stochastic where ours is not" half of the argument is dead; the quantization half survives, weakened. Closing it outright would need a run. Previously: **ARGUED.** A divergence that makes our arithmetic *more* accurate, and the most expensive to close. Named a suspect of last resort; both earlier suspects in that ordering have since been discharged. `amortization.md` §(c). |
| **6.5** | their `img_step_no_action` **samples** the preferred code at every step; our `cosine_sim_to_pref_rollout` rolls the CRSPP's preferred trajectory mean-field (keyless `predict_beliefs`) | **BUILT (c68c6d3, `sample_scoring`), RUN FROM SCRATCH (`runs/crspp_sampled_500k`), EXCLUDED AT RUN LEVEL 2026-09-30** (`actor-entropy-diagnostic.md`, §28 addendum: the double-pump phase matches the mean-field control seed for seed). Earlier: **RECORDED (chunk 2c, FLAG (1)); MEASURED 2026-09-18** (`pref-rollout-probe.md`): on the `crspp_init_100k` checkpoint at 100k, same agent rollout, the leaf reads 0.651 mean-field against 0.528 sampled, the oracle's own series 0.46–0.48 at that step — a near-pure **level** shift (−0.12 at every step, per-start correlation 0.98), about two thirds of the standing 16-of-16-bins gap. Mechanism: the mean-field stoch norm is 1.23 against the one-hot's 5.66, so the cosine is 90% `deter`. Closing it needed a key in the scoring seam, built on 2026-09-21 (c68c6d3). **PROBED AGAIN 2026-09-20 at 500k, two seeds** (`pref-rollout-probe-500k.md`): the gap is −0.09 and shrinking; a keyless mode rollout does not reproduce the sampled leaf and drifts from its carry; mean-field orders imagined trajectories like the *expected* sampled leaf (Spearman 0.88) where one draw agrees with another at 0.3; and mean-field predicts best open loop, mode worst. The three hypotheses for why sampling would help all fall; what remains is exact fidelity to their estimator, which by measurement is the mean-field signal plus noise. |
| **21** | their shared forward pass is reused post-step; ours recomputes | **RECORDED 2026-09-07 (statistically benign; measured); CLOSED 2026-09-25** by the performance round: `fit_world(return_states=True)` and `fit_crspp(return_states=True)` hand back the states of the pass the gradient was taken through, and `train_step` reads those — as theirs do. Two `observe` scans fewer per training step. |
| **23, 24, 25, 27, 28, D7** | their critic target is a second `score` sample; their actor runs nine times; our habit rolls a full 8-step plan per acting step (8 actor evaluations and 16 predict applications vs their 1 and 1); three more inert | **COSMETIC.** Recorded with a line each; D7 is behaviourally equivalent but ~10× the work in a loop that is 1.4% of the cycle. |

## 29 — the dropped `moments` carry (found 2026-09-13, closed same day)

Not a divergence from their code but a defect in ours that *masqueraded* as one,
and the cause of the mid-training collapse that made the third 2M run fail §7's
trajectory-shape criterion while passing its windows. `fit_world` returned
`LearnState(opt_states=opt)`, dropping `moments` (default `{}`); it runs first in
`train_step`, so `imagine`'s defensive backfill restarted the return normalizer's
percentile EMA from zeros on every one of 125,056 gradient steps. Signature: the
checkpointed band `[−0.586, +0.007]` was `0.01 × [−57.85, +0.71]` — the true
percentiles — to four significant figures, i.e. exactly `1 − decay`, one step
from zeros. Consequences, measured: `invscale` clamped to 1.0 so the actor's
advantage was **58.6×** the oracle's and never normalized in any of 7,816 logged
rows; the advantage-to-self-imitation ratio swept 1.8 → 221 across the collapse
windows where the oracle's stays 0.25–0.86; and `moments[SLOW_KEY]` was wiped too,
so the slow critic **hard-copied the critic on every step** (bit-identical across
185,980 parameters at the final checkpoint) instead of EMA-ing at 0.02 — which
**re-opens 13**, closed in code but never reaching the trainer. Both fixed in one
line each (`train/world.py`, `System.learn`) plus a regression test at the
trainer seam. The `runs/mc_2d` result (−69.1 ± 2.3) converged *despite* this and is
not a valid parity result; re-run pending.

**Method note, third instance.** Both transcription tests over this machinery
passed throughout: the `perc_ema` port threads the EMA state by hand in a Python
loop, and the actor-loss port takes the advantage as a given input. A port proves a
formula and inherits its author's assumptions about the caller; no test in the
suite ran two `train_step`s and asserted anything about `moments`. The regression
test now does, and catches the fault by its signature (a live carry grows the band
~6.8× over seven steps; a dropped one does not).

## 30 — the CRSPP filter not restarted at episode boundaries (found 2026-10-01, fixed same day)

A defect in ours, not a recorded divergence: their `obs_step_no_action` masks the
CRSPP's carried state at `is_first` like the world model's step does, and our
`observe_no_action` had no `is_first` at all, so the preference model's recognition
scan carried across the ~1.7% of rows where an episode starts inside a 64-step
window while our world model's `observe` restarted there. Training ledger rows 4.2
and 4.4 covered the step's layers and 4.1 the window-start initial; the mask between
them had no row. Found at the end of the actor-entropy diagnostic's mechanism trace
(§28–§29): with everything else swapped between the sides, this one convention sets
whether the CRSPP learns a start-blind preference rollout that rewards a more
saturated imagined trajectory (ours, the double-pump transient's source) or their
start-conditioned, goal-ward one; one line removing their reset flips their sign,
and adding ours flips ours, to either initial. Fixed by threading `is_first` through
the three CRSPP functions, restarting at the CRSPP's own initial. Confirmation:
`runs/fix1k_key0` (the 16k stage resumed to 28k with the fix) for the slope, then the
500k phase read.

## 31 — the act-time episode start read the prior's probabilities, not its mode (found 2026-10-01, fixed same day)

A second boundary defect of the same kind as 30, between acting and training rather than
between two models. Training restarts the world model's filter at `is_first` from the
learned carry and the prior's mode (a one-hot); the collection lane's
`episode_start_beliefs` started from `initial_priors`, whose code is the prior's
probability tensor, so the first act-time state of every episode differed from every
state training filters from. Acting-ledger D6 had recorded the step as matching their
`obs_step`, which steps from the mode. The actor, never trained on that state, read it on
some seeds as "push right" and began against gravity; that start is what the 50k–250k
double-pump phase and the 101-step timeouts are made of. Found by the 65k trace
(diagnostic §30): with parameters frozen at 68.6k (restart-fixed run) and 70.7k (unfixed),
starting from the mode takes the double-pump share on the six pumping seeds from
.66–.98 to .02–.07 and success to .97–1.00, and leaves the oracle-like seed unchanged.
Fixed: `episode_start_beliefs` reads `RSSMCore.initial_state()`; one test pins the acting
start to `observe`'s first filtered step bitwise.

## Closed, by round — what each round of hunting was worth

| round | method | closed |
|---|---|---|
| 2d hunt (2026-09-01/02) | reading their code against ours | entropy mean-not-sum (1024× at `tiny`), member dropout (provably inert), member-loss reduction, and the divergences 1–13 group |
| resolved-config audit (2026-09-02) | their own dumped `config.yaml` as ground truth | critic and ensemble at 2×32 not 3×256, per-trainer Adam `eps`, ig grad clip, four env lanes per seed |
| round 3 (2026-09-02) | the same, at head level | 16 (bounded actor head, tanh mean + `minstd + range·sigmoid(raw + 2.0)`), 14 (LayerNorm), 15 (fixed σ ≡ 1.0 ig head), 17 for `MLPConditional`, 18 (M+ alternation is dead code in *their* repo, so parity matches the code), 19 (2M agent steps) |
| the info-gain fix (2026-09-05) | cross-run comparison against their own `metrics.jsonl` | the leaf dropped `E[σ²]` from the law of total variance, so it went large negative and paid the agent to linger — the defect that cost the first 2M run |
| execution-trace audits (2026-09-06) | running both with jit off and diffing call sequences | 15 new divergences from 229 audited rows, 140 matched |
| consolidated round (2026-09-07) | fixing them | D1–D6, D9, 22, 26-adjacent; the episode-column cluster and the actor's gradient |
| shape investigation (2026-09-13) | fine-resolution binning + the actor channel + their own logs | 29 (and 13, re-opened): the dropped `moments` carry |
| plateau analysis + Arm B (2026-09-15) | 44 matched series vs their logs, checkpoint probes, a pre-registered 100k run | 17+26 adopted; leading remaining suspect is the CRSPP→encoder gradient seam (ledger 5, adjudicated) — `arm-a-design.md` |
| Arm A (2026-09-15) | a 100k run with one joint gradient over world + CRSPP parts | the CRSPP→encoder seam is **behaviourally inert**: bands 4–9 unchanged, while band 3 *passes* (the arm reproduces the oracle's own similarity trajectory), so the null is about the hypothesis, not the implementation. Ledger 5's fence costs nothing measurable; no engine change needed. |
| series sweep + isolation control (2026-09-15/16) | all 246 of their keys vs our 27; a pre-registered control at one commit | float16 measured and demoted; the `bc_logprob` instrumentation defect; 17+26 identified as the cause of the slow start and characterised as a trade; **the 10× seed variance isolated as a separate, unexplained defect** — the standing open question |
| actor-entropy diagnostic + 16-key sweep (2026-09-25) | checkpoint probes under the checkpoint's own commit, both sides' logs, their 2M checkpoints under their code, twelve more init keys at 16k | the 0.320-vs-0.098 residual is not an actor divergence: three seeds on the oracle's curve and one in an early one-sided trap; the trap's mechanism is the cosine-to-preferred-rollout leaf (the only live leaf at 16k on either side) against a preferred rollout that is a fixed point near the CRSPP's attractor, a regime their own logs put both sides in through 600k; rate 3 in 16 under their initializer, their 0 in 4 consistent with it — `actor-entropy-diagnostic.md` |
| act-time sample (2026-09-26/28) | 1k-step replays of trapped keys, the actor read on both feature paths, three interventions and a re-roll control; their code on twenty seeds | the trap is the act-time feature path (acting-ledger D6, second half): the actor trains on sampled codes and carries and acted on mean-field ones; sampling the act-time code for actor, entropy and carry lifts both trapped keys, re-rolls do not, their agents trap 0 of 20 — closed in the reference recipe through a new engine face, `infer_states(key=)` (`act_sampling`); `actor-entropy-diagnostic.md` §8 |

## Method notes — what actually finds divergences

Recorded because the yield differs by an order of magnitude between methods.

- **Their dumped resolved config is the only honest inventory of values.** Every
  divergence in rounds 2 and 3 came from reading it rather than the paper.
- **Their own training logs are the best control available.** Comparing traces
  against their `metrics.jsonl` found in one afternoon what two rounds of config
  auditing had missed.
- **Execution-trace audits find what the other methods structurally cannot.**
  Config audits catch values, transcription tests catch formulas, log comparison
  catches drift — none looks at *control flow and sequencing*, which is the class
  most of the late findings belonged to. Their `debug` config sets
  `jax: {jit: False}`, so their code can be run and traced directly.
- **A transcription test inherits its author's assumptions about the call site.**
  Two of fourteen ports certified a misreading: one wrote `means.std(0)` where
  their source says `preds.std(0)`, and one asserted `winit: normal` for layers
  their code draws uniform. Build a port from their *call site*, with the
  caller's indexing and argument order transcribed too — not just the expression.
  This has cost us twice. See `amortization.md` §(a″).
- **A blessed divergence must still be measured.** The sampled-std leaf was
  argued as principled on a limit argument that did not hold in the target's own
  configuration, and it cost 42 hours of compute. Divergence 20 above is argued
  *and* measured for that reason.
- **Audit the seams, not only the steps.** Divergences 30 and 31 passed every
  per-step audit, because each code path matched its reference on its own terms. One
  sat between two models reading one batch, the other between the acting path and the
  training path. Three checks target that class:
  1. *One source of truth for every boundary state.* The state an episode starts from,
     and the rule for crossing an episode boundary inside a window, should each be one
     function that every path calls. A test then requires every path's first step to
     equal it bitwise, as `test_the_acting_start_is_the_training_restart` does. Searching
     for constructors of initial states and readers of `is_first` gives the inventory.
  2. *Differential execution at the boundaries.* Run a short window with an episode
     start in its middle through both codes, on the act path and on the train path, and
     compare the states on either side of the boundary. The step-level traces compared
     intermediate tensors inside a training step and never at an episode's first
     act-time step.
  3. *Typed paraphrase at the seams.* Where a value crosses between paths or modules,
     write one sentence naming the operation and the value's family and form (one-hot,
     probabilities, mean). Acting ledger D6 said "the initial belief, one GRU step on a
     zero action", true on both sides; the typed sentence would have read "a one-hot
     code" on theirs and "a probability tensor" on ours.
- **Bisect a learned difference by moving the trainer, not the weights.** A weight
  transplant between the two sides passed forward equivalence and still read nothing,
  because a learned latent only means something in the space it was trained against.
  Running each side's trainer inside the other side's latent space, with everything
  else held, is what isolated divergence 30.

## Also here

- `execution-ledger-acting.md` — 81 rows, one full agent step.
- `execution-ledger-training.md` — 148 rows, one full train step.
- The oracle's own record (its command, seeds, per-seed ACR/SR, and the
  landmarks) remains at `amortization-p0-addendum.md` §7, indexed here rather
  than moved: history entries cite it in place.
