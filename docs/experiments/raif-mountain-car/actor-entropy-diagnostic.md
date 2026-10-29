# The actor-entropy residual at 12–16k: one seed in five, one leaf (2026-09-25, amended the same evening)

## Summary (2026-10-01)

The document began as a question about one number, act-time actor entropy at 12–16k, and
became the trace of every behavioural difference between our replication and the oracle
that the earlier audits left. It found three defects. Read the sections below for the
evidence, in order.

- **§1–§10: the early trap.** One seed in five fell into a one-sided collapse whose fixed
  point was the preference model's rollout. Acting on a sampled code rather than the
  mean-field posterior (acting ledger D6, second half) removed it.
- **§11–§27: the double pump.** Seeds drifted from a 42-step single pump to a 66-step double
  pump, a phase over roughly 50k–250k that the oracle never shows. Counterfactual replays,
  both sides' series at 1k resolution, the actor's update statistics and objective
  decompositions narrowed it to one object: the preference model's cosine leaf, whose
  response to a more saturated imagined trajectory was positive on every seed of ours and
  turned negative on theirs from about 21k. The cross-data cell showed the sign follows
  the code, not the data.
- **§28–§29: the first defect behind it.** Our CRSPP's recognition scan did not restart
  at an episode boundary inside a training window. With that one convention swapped, the
  sign swaps on both sides (register 30, training ledger 4.15, fixed in dd47e4a). That fix
  is real, but four seeds to 60k showed it is not what carries the long phase (§29's
  addenda).
- **§30: the driver.** The act-time episode start stepped from the prior's probabilities
  where training restarts from its mode, so every episode began from a state training
  never filters from. On some seeds the actor read it as "push right", and that start
  became the double pump or a timeout (register 31, acting ledger D6, fixed in f8d9fa5).
  With both fixes, four seeds from scratch sit in the oracle's band through 125k (§30's
  addenda). The 2M acceptance run is in progress.

The general lesson, recorded in history entries 096 and 098: paths and modules that read
one stream must agree on its boundaries, meaning where an episode starts and what state it
starts from, as strictly as they agree on its tensors.

**Evidence.** The write-ups behind §21–§30 are committed in `evidence/` next to this
file and cited by path; each states its method, checkpoints and numbers. The probe scripts
were session scratch and are not part of the repository; where a write-up names one, the name
labels the method described beside it. §1–§20 have no separate write-ups: their text and
tables are the record.

**Status: measurement, not a decision.** Nothing in the library changed. Every number
below is a read of existing logs or a read-only probe on an existing checkpoint; the
readings that go beyond the numbers are marked as readings.

**Question.** Under their initializer (rows 17+26) our act-time actor entropy at 12–16k
was recorded as 0.320 against the oracle's 0.098 (`plateau-analysis.md`, `actor-trace-audit.md` §2.3 "Q1"). Is that an actor property, and
if so, where does it come from?

**Checkpoint.** `runs/armB_telemetry_16k` at step 16,384 — four seeds, `winit: normal`,
fixtures, code `a21b091` (dirty) — read under its own commit's source (`8e321d8`, the first
with the `winit` flag), because the CRSPP transition's leaves changed shape since and the
checkpoint no longer deserialises on current code. **Logs:** `runs/*/scores.jsonl`,
`runs/*/metrics.jsonl`; theirs `raif-paper/logs/oracle_state_mc_seed{0..3}/metrics.jsonl`
(`stats/mean_log_entropy`, `train/policy_entropy_mean`, `episode/success_rate`).
**Theirs:** `raif-paper` @ `9b1864e8`.

## 1. The number is a pooled mean over four seeds, three of which are on the oracle's curve

Act-time entropy (nats), mean over episodes in each 4k bin:

| | 0–4k | 4–8k | 8–12k | 12–16k | first success | success rate 12–16k |
|---|---|---|---|---|---|---|
| theirs, seeds 0–3 | 1.08–1.19 | 0.32–0.55 | −0.00–0.14 | 0.06–0.12 | 1.6k–2.6k | 0.59–0.65 |
| ours seed 0 | 0.95 | 0.60 | 0.35 | **0.06** | 1,816 | 1.00 |
| ours seed 2 | 1.06 | 0.47 | 0.35 | 0.25 | 2,180 | 0.80 |
| ours seed 3 | 0.98 | 1.21 | 0.50 | 0.26 | 1,988 | 0.78 |
| ours seed 1 | 1.13 | 1.29 | 1.15 | **1.06** | none by 16k | 0.00 |

The pooled 12–16k mean is 0.318 (281 episodes; seed 1's 44 episodes at 1.06 carry it).
Seed 0 tracks the oracle exactly; seeds 2–3 end 0.15 nats above it with success rates
above any oracle seed's. **Training-side** policy entropy at 12–16k agrees between the
sides for every seed (theirs −0.15 / −0.04 / −0.02 / −0.13; ours −0.23 / +0.19 / 0.00 /
−0.16), so the actor as trained is as committed as theirs.

The same four fates recur to the digit in `crspp_init_100k` and in `crspp_sampled_500k`
(first successes 1,812 / 17,556 / 2,180 / 1,996 in all three runs; entropies equal to two
decimals). The two runs' training metrics differ from their first row (`--sample-scoring`
did reach the trainer: actor loss −1.44 vs −0.85 at step 1,024, seed 0), so the identical
fates are a property of the *collection* stream, not a failed flag. *Reading:* 59% of
recorded actions sit at the clip (|a| > 0.9) and the exploration keys are the same, so
early trajectories are set by the noise stream more than by the objective variant.

## 2. Seed 1 is not an uncommitted actor; it is a one-sided collapse

Read at act-time features on the 16k checkpoint (2,048 of its own replay states):

| seed | σ mean | share σ ≥ 0.95·max | \|μ\| mean | share \|μ\| > 0.9 |
|---|---|---|---|---|
| 0 | 0.27 | 0.04 | 0.90 | 0.77 |
| 1 | **0.61** | 0.44 | **0.95** | **0.92** |
| 2 | 0.39 | 0.29 | 0.88 | 0.74 |
| 3 | 0.41 | 0.03 | 0.87 | 0.70 |

And in its replay stream: action mean −0.96 over its last 4k steps, lag-1 autocorrelation
+0.02 (the oscillating seeds: +0.75 to +0.85), position maximum −0.30, every episode at the
101-step limit, coverage flat at 0.353 from 4k on. It set in within its first ~500
collected steps (action mean −0.67, lag-1 +0.17; by the next 2k bin −0.87, +0.09) and in
the 100k run of the same seed it lifted at 17,556, after which the seed converged
normally (success 1.00 from 24k). The wide σ is the `actent` bonus against a flat
advantage; the entropy is the symptom.

**Invisible training-side.** Every training series of the collapsed seed matches the
other three's (0–4k: advantage 1.11 vs 1.10–1.17, actor loss −0.84 vs −0.78 to −0.86,
cosine leaf 0.48 vs 0.46–0.57, value, return normalisation). The one series that moves is
`policy_entropy` (0.36 vs 0.12–0.40 at 4–8k), mildly. This is `series-sweep.md` §5.3's
finding again, at the seed level.

## 3. Three things ruled out

- **D6's mean-field act-time features** (`execution-ledger-acting.md` D6, second half,
  "stands"). The same states filtered the training way (`posterior_states`, one-hot codes)
  give −0.06 / 0.49 / 0.24 / 0.17 against −0.06 / 0.85 / 0.21 / 0.30 act-time: ~0.1 nats
  pooled, the audit's estimate, and nothing on seeds 0 and 2.
- **A saturated initial actor.** Twenty init keys under their initializer, untrained, read
  at the same features: none saturated (max share |μ| > 0.9 is 0.47 at key 17; seed 1's is
  0.00, |μ| 0.65, μ mean −0.30). Four keys under ours: none.
- **Fixtures.** Identical per seed: 48 episodes, success rate 1.00, mean return −76.

## 4. The objective prefers "left" for seed 1, and the actor follows it

Open-loop 8-step plans scored through `score_policies` (lower G is better) with the CRSPP
posterior as the cosine row's start, at each seed's own replay states:

| states | G(left) | G(right) | share G(right) < G(left) | actor μ |
|---|---|---|---|---|
| seed 1, all (n=2,048) | −3.92 | −3.57 | **0.09** | −0.17 |
| seed 1, x<−0.5 & v>0 (dynamics: push right) | −3.99 | −3.70 | **0.05** | −0.10 |
| seed 1, x>−0.5 (fixture-like states) | −3.78 | −3.09 | 0.06 | +0.24 |
| seed 0, all | −3.68 | −3.51 | 0.59 | +0.20 |
| seed 0, x<−0.5 & v<0 (dynamics: push left) | −3.69 | −3.91 | 0.97 | −0.44 |

Every left–right difference is in `cosine_sim_to_pref_rollout` (max |L−R| 2.5).
`predicted_reward` and `g_tail` are constant across plans *and* across states (sd 0.0000
over 2,048 states), `model_uncertainty` is 0 — `series-sweep.md` already records theirs as
a degenerate constant (0.0578) and ours as ~0, so on both sides the plan-discriminating
objective at 16k is the cosine leaf alone. Seed 0's leaf is not physically aligned either
(it prefers right in 97% of leftward-moving left-half states, where energy pumping says
left), yet seed 0's actor learnt the oscillation. *Reading:* the leaf is not what teaches
the oscillation on the good seeds; success discovery, self-imitation and the critic must
be.

## 5. Why "left": the preferred rollout is a fixed point, and seed 1's sits at the left wall

From 384 left-half start states per seed (mean start −0.68 to −0.77):

- **The agent's world model is fine on every seed.** Decoded position after 8 imagined
  steps: left −0.72 to −0.89, right −0.10 to −0.44 (seed 1: −0.72 / −0.10). "Push right"
  is imagined as moving right.
- **The CRSPP's preferred rollout does not move.** cos(preferred_t, preferred_1) ≥ 0.993
  over the 8 steps on every seed; it is not the start state (cos to the CRSPP posterior at
  the start: 0.78) but the no-action dynamics' attractor. A *sampled* rollout
  (`img_step_no_action` with a key) jumps once (cos 0.63–0.66 to its first step) and then
  sits too.
- **Where the attractor sits, decoded by nearest agent posterior feature in the replay
  bank:** seed 0 −0.57, seed 2 −0.55, seed 3 −0.55, **seed 1 −0.88** (the left wall).
  The sampled rollout's: −0.62, −0.54, −0.57, **−0.72**.
- **The leaf then rewards staying near the attractor.** Per-step cosine of the imagined
  trajectory against the preferred one, seed 1: left flat at 0.947; right 0.948 decaying
  to 0.809 as the imagined car leaves the wall. Seeds 0/2/3: right slightly above left
  throughout (means 0.932 vs 0.923, 0.949 vs 0.942, 0.955 vs 0.940). The same ordering and
  decay appear against the *start* CRSPP posterior alone (seed 1: 0.74 flat vs 0.75 →
  0.64), so the leaf's behaviour here is "keep the features near the CRSPP's current
  attractor".
- The CRSPP posterior's alignment with the agent's posterior at the same state is
  *higher* on ρ = −1 rows than on ρ = +1 rows on every seed (0.53–0.57 vs 0.47–0.51) —
  the opposite of the contrast term's intent. Recorded; small; the negatives are the
  dominant, left-half rows, so this may be the state distribution rather than the term.

**The leaf as training sees it.** The numbers above went through the keyless faces —
mean-field imagined trajectories. `imagine` scores *sampled* agent rollouts, so the same
measurement was repeated with keyed `predict_beliefs` on the current-code sweep checkpoints
(§7), against both preferred-rollout forms, left-half states:

| key | fate | sim (sampled vs sampled pref) | share of states where the leaf prefers right | mean(right − left) |
|---|---|---|---|---|
| 8 | healthy | 0.57 | 0.58–0.59 | −0.001 |
| 14 | healthy | 0.58 | 0.70 | +0.002 |
| 11 | **trapped** | 0.58 | **0.00** | −0.024…−0.029 |
| 15 | **trapped** | 0.58 | **0.04** | −0.018…−0.022 |

Healthy keys are indifferent; trapped keys prefer left at essentially every state (key 15,
indifferent under mean-field imagination at 0.47, is 0.04 here — the sampled form is the
sharper one). The sim level under sampled-vs-sampled, 0.57–0.59, is the level their own
agents show (§6); the mean-field-vs-mean-field 0.92–0.96 of the table above is the
probability-vector geometry (register row 6.5), not the training signal. Feature norms
are what a one-hot code and a tanh carry give (code part 5.66 = √32, deter part ≈ 6).

*Reading, not measured:* a seed whose CRSPP attractor lands at the wall its own failures
occupy is pinned there by the only live leaf — the leaf says "the future should look like
here", the actor obliges, the buffer fills with ρ = −1 wall states, and nothing in the
objective pays for leaving. Escape (17.5k in the 100k run) comes from outside the
objective. Why seed 1's attractor lands there and the others' at −0.55 is not measured.

## 6. Their trained agents, and ours at 500k: the attractor regime is a phase

Their only checkpoints are at 2M steps (`logs/oracle_state_mc_seed{0..3}/checkpoint.ckpt`), so
the comparison is at unmatched stages; it was run by a subagent under their own code
(`raif-jepa/.venv`, `agt.Agent.inner` on the pickled variables, `nj.pure(create=False)`,
verified against their logged `similarity_mean` and `extr_G_crspp_mean`; the session script
that ran it was not kept). Their preferred rollout is `imagine_preference` →
`img_step_no_action` (agent.py:262-280, nets.py:189-201), **sampled**, H = 8.

| | preferred rollout cos(t, 1) at t = 8 | decoded position of the rollout | cos(C_post, post) | sim, left starts |
|---|---|---|---|---|
| theirs, 2M, seeds 0–3 | 0.05–0.12 (moves) | **+0.3 … +0.47, the goal region**, from every start | 0.02 – 0.04 | ≤ 0.35; +1 beats −1 in 67–100% of states |
| ours, 500k sampled run, seed 1 | 0.35 sampled / 0.63 mean-field | **+0.40 / +0.47** | −0.03 | negative; right preferred in 83% |
| ours, 500k, seeds 0, 2, 3 | 0.52–0.62 / 0.78–0.85 | −0.56 … −0.68 | 0.44 – 0.54 | 0.6–0.9, saturated; right preferred in 33–49% |
| ours, 16k, seeds 0–3 (§5) | ≥ 0.993 | −0.55 (×3), −0.88 | ~0.5 | 0.92–0.95, saturated |

**Their 16k checkpoints (matched phase).** Their own code run to 16,384 steps on seeds
4–11 (their recipe, `run_remaining_seeds.sh`'s collect-then-train, `logs/probe16k_state_mc_seed*`)
leaves a checkpoint each; the same probe on all eight (decoder-on-posterior error ≈ 0.1 at
this stage):

| | preferred rollout | cos(C_post, post) | attractor position (decoder / NN) | plan preference, left starts |
|---|---|---|---|---|
| theirs, 16k, seeds 4–11 | **fixed point**: mode cos(8, 1) 0.91–0.97, consecutive 0.99; sampled path flat at its resampling floor 0.63–0.66 | 0.50–0.53 | −0.35 … −0.50 / −0.4 … −0.6, never the wall | indifferent: sim 0.51–0.59 for both actions, |Δ| ≤ 0.02, share preferring +1 0.27–0.83 |
| ours, 16k (§5, §7) | fixed point, ≥ 0.993 (exact on current code) | 0.44–0.56 | healthy −0.35 … −0.57; trapped −0.56 … −0.88 | healthy indifferent (0.58–0.70); trapped 0.00–0.04 |

Same structure, same stage, and the same geometry: their feature norms at 16k are deter
5.6–5.9 and stoch 5.66 (= √32, always a one-hot on their side) against our 5.9–6.1 and 5.66,
and on both sides the 16k cosine is ~98% deter — the 0.5–0.65 similarity floor is a shared
carry direction, not code agreement (at 2M their deter norm halves and de-correlates, and
the plan-separating signal becomes half deter, half code overlap). Their actors already push
right at 16k in every seed (imagined action +0.35 … +0.86), which the leaf is not what drives.

Their logs put the same transition on a clock: `train/similarity_mean` is 0.37 (0–16k), 0.46,
0.50, 0.46 (262–600k) and **0.03** at 600k–2M, seed by seed. So through half a million steps
their CRSPP branch is in the regime ours is in at 16k and 500k — aligned to the posterior,
its rollout near the current state — and detaches late; our seed 1 detached before 500k,
our other three had not. Our 2M runs' cosine leaf drops late too (`mc_2d`: 0.6 → 0.02–0.2).
*Reading:* the fixed-point-near-the-current-state structure of §5 is the early phase on
both sides, not a difference of code; whether their early attractor ever lands at the wall
cannot be read from a 2M checkpoint. Our leaf's *level* is higher than theirs in the same
phase (0–16k: mean-field 0.65, sampled 0.53, theirs 0.40), which the register's 6.5 row
already records.

## 7. The trap's rate: 3 in 16

Twelve more agent-init keys under their initializer, fixtures 0–3, current code (`de8fccf`),
16,384 steps each (`runs/sweep16k_off{4,8,12}`, keys `seed_offset + seed`):

| key | first success | act-time entropy 12–16k | success 12–16k | last-4k action mean / lag-1 / position max |
|---|---|---|---|---|
| 4, 5, 6, 7 | 2,016 / 1,776 / 1,544 / 1,132 | 0.10 / −0.01 / 0.17 / 0.00 | 1.00 / 0.86 / 0.95 / 1.00 | oscillating |
| 8, 9, 10 | 1,448 / 1,760 / 1,780 | 0.20 / 0.13 / −0.05 | 0.97 / 0.89 / 0.77 | −0.08…+0.02 / +0.79…+0.90 / +0.49 |
| **11** | none | 0.92 | 0.00 | **−0.95 / 0.00 / −0.27** |
| 12, 13, 14 | 1,788 / 1,616 / 1,776 | 0.19 / 0.12 / 0.10 | 0.92 / 0.75 / 1.00 | −0.05…−0.12 / +0.79…+0.83 / +0.49 |
| **15** | none | 1.11 | 0.00 | **−0.89 / +0.14 / +0.05** |

With seeds 0–3 (§1): **3 of 16 keys trap** (1, 11, 15), all with the same signature — a
saturated leftward mean, no action coherence, no success, 160 time-limit episodes, coverage
0.32–0.38 against 0.52–0.62.

**Their rate.** Their code, their recipe, seeds 4–19 to 16,384 steps (this machine, ~26 min
a seed, `logs/probe16k_state_mc_seed{4..19}`): first successes 1,120 … 4,228 (median ~1,960);
act-time entropy at 12–16k −0.07 … 0.08 for fifteen seeds and 0.29 for seed 6 (their slow
tail: first success 3,584, entropy 1.05 at 4–8k); success rate 0.38–0.72. **0 of 20** trapped
(with their original four). Against our 3 of 16, Fisher's exact test gives p ≈ 0.08 — the
point estimates differ (0% vs 19%) and the sample nearly separates them; the mechanism in §8
is what separates them.

The trapped keys' leaf, on the current code (the mean-field rollout is an exact fixed point,
cos 1.00 at every step): key 11's attractor decodes at −0.61 with imagined-right's sim decaying
0.97 → 0.83 (right preferred in 20% of its states); key 15's at −0.56, indifferent (47%). The
healthy keys 8 and 14: −0.45 and −0.35, right preferred in 77% and 91%. The relation between
attractor position and fate holds in order but not cleanly (key 15 is the weak case).

**Not the initial actor.** Untrained μ means under their initializer: keys 7 and 10 start more
leftward (−0.49, −0.42) than keys 1, 11 (−0.30, −0.28) and succeed by 1.8k; key 15 starts at
−0.54 and its first 500 own steps lean *right* (+0.15). The trap is decided in the first
~2k steps of interaction.

## 8. Causal order, and the intervention that lifts the trap (2026-09-26)

Our runs are deterministic per key, so a trapped key can be replayed. Keys 11 and 15 (and
the healthy key 8 as control) were replayed for 4,096 steps in 1,024-step resumes with every
stage's checkpoint kept (`runs/causal4k_key*_s*`), and read at each stage.

**The actor as trained never saturates; the actor as acted does, from the first 1k.** On the
same left-half replay states, the actor's mean read at *training* features (`posterior_states`,
sampled one-hot codes) against *act-time* features (collection's mean-field filter):

| key | stage | stream action mean (last 1k) | μ at act-time features (share < −0.9) | μ at training features (share < −0.9) | corr |
|---|---|---|---|---|---|
| 11 | 1k | −0.42 | **−0.81 (0.63)** | −0.15 (0.03) | 0.41 |
| 11 | 2k | −0.99 | **−0.88 (0.85)** | −0.18 (0.10) | 0.43 |
| 11 | 16k | −0.95 | **−0.96 (0.98)** | −0.53 (0.24) | 0.35 |
| 15 | 1k | +0.15 | −0.51 (0.34) | −0.16 (0.11) | 0.69 |
| 15 | 2k | −0.31 | **−0.72 (0.60)** | −0.11 (0.09) | 0.65 |
| 15 | 16k | −0.89 | **−0.91 (0.84)** | −0.39 (0.17) | 0.48 |
| 8 (healthy) | 1k | −0.21 | −0.08 (0.08) | −0.18 (0.07) | 0.69 |
| 8 (healthy) | 16k | −0.43 | −0.30 (0.39) | −0.06 (0.11) | 0.85 |

The recorded actions track the act-time column. The two feature paths differ in both parts:
the act-time code is a probability vector (norm ≈ 1.0 against the one-hot's 5.66) and the
act-time carry is the GRU driven by those probability vectors rather than by samples;
cos(f_act, f_train) ≈ 0.6–0.7. The leaf's tilt toward "left" (§5) comes after the stream for
key 11 (0.46 → 0.20 → 0.09 over 1k–3k while the stream collapsed by 2k) and before it for
key 15 (0.00 from 1k while the stream still leaned right) — secondary either way, since the
actor that the leaf trains stays unsaturated.

**Interventions**, each a monkeypatched copy of `run.py`'s collection function, replayed from
the same init for 4,096 steps:

| replay of key 11 | first success | success 3–4k | coverage 3–4k | stream at 4k: action mean / lag-1 / position max |
|---|---|---|---|---|
| as shipped (mean-field act-time path) | none | 0.00 | 0.26 | −0.98 / −0.03 / −0.29 |
| v1: actor reads argmax-one-hot codes; carry unchanged | none | 0.00 | 0.27 | −0.97 / −0.02 / −0.27 |
| **v2: code sampled at act time; the sample read by the actor, the logged entropy and the carry** (their `obs_step`) | **3,020** | **0.25** | **0.40** | **+0.55 / +0.56 / +0.47** |
| control: as shipped, exploration stream re-rolled (three different fold-ins) | none / none / none | 0.00 ×3 | 0.25–0.27 | trapped ×3 |

Key 15 under v2: first success 1,940, successes at 1–3k, stream at 4k an oscillation
(+0.55 / +0.61 / reaching +0.47; as shipped −0.95 / +0.06 / −0.42). The healthy key 8 under v2:
unchanged (first success 1,556 vs 1,448; success 0.46 vs 0.54 at 3–4k). v1 on key 8 *hurt*
(success 0.00 at 3–4k, entropy back to 0.88): a one-hot code on a mean-field carry is a mixed
state neither side has, so v1 is discarded as a test rather than read.

The re-rolls did change the stream: against the as-shipped replay, mean |Δaction| ≈ 1.0 over the
first 400 own steps (they lean further left, −0.57 … −0.70; v2 leans right, +0.21, from the start).

*Reading:* the trap is a property of the act-time path, not of the exploration stream — three
re-rolls of the stream stay in it, one change of the path leaves it. Their agent acts on the
sampled codes it trains on, which is the state v2 reproduces; it has not trapped in 20 seeds.

**This sits on a standing ruling.** `execution-ledger-acting.md` D6, second half: *"the
mean-field face … stays, as adjudicated — the keyless-scoring-seam decision"*;
`amortization.md` §(f) records "mean-field beliefs throughout" at act time. The ruling is not
re-proposed here; what is new is a measured behavioural consequence at a rate of ~1 key in 5
under their initializer, and a six-line change to the *reference recipe's* collection lane
that removes it without touching the engine's keyless faces.

## 9. What this settles and what it does not

- The "0.320 vs 0.098" residual is **not an actor divergence**: three seeds commit on the
  oracle's schedule, one seed is in an early one-sided trap that the pooled number
  reports as width. Q1 of `actor-trace-audit.md` folds into Q2 (the seed lottery).
- The trap's **mechanism** at 16k is in the cosine-to-preferred-rollout leaf, the only
  plan-discriminating leaf at that stage on either side: a stationary preferred rollout
  whose attractor, for that seed, is the wall.
- The trap's **rate under their initializer is 3 in 16** on our side; **0 in 12** on theirs
  at matched conditions (§7), p ≈ 0.17. The structure the trap runs on — a fixed-point
  preferred rollout, an aligned CRSPP posterior, a plan-indifferent leaf on healthy seeds —
  is the same on their 16k checkpoints (§6).
- The one measured discriminator: on a trapped key the leaf prefers left at every state
  under the training form; on healthy keys, ours and theirs, it is indifferent (§5, §6).
- **The cause, on our side, is the act-time feature path** (§8): the actor and the carry
  run on mean-field codes at act time and on sampled codes in training; on a trapped key the
  actor's extrapolation onto the act-time path is a saturated one-sided push from the first
  1k steps, and sampling the code at act time (their `obs_step`) lifts both trapped keys
  while three re-rolls of the exploration stream do not.
- **Not settled:** whether their rate is truly zero (0 of 20); whether the amended
  recipe changes the converged window (a run); what lifts a trapped seed unaided (17.5k for
  seed 1); and the decision on D6's second half, which is adjudicated and is Alex's.

## Method notes

The probes for §1–§9 were session scripts and were not kept; the sections' text and tables
are their record. Ours ran with `PYTHONPATH=<git archive 8e321d8>/src` against
`runs/armB_telemetry_16k` (a separate script read current-code checkpoints, and another loaded
their checkpoints under their code; §7's sweep was launched the same way). States are
`sample_windows` draws (32 × 64) from each seed's general buffer, so they cover 0–16k, not
the 12–16k window; the act-time path reproduces collection's filter (`infer_states`,
`episode_start_beliefs` at `is_first`, `update_empirical_prior` on the recorded action).
Nearest-neighbour decoding uses cosine similarity to the 2,048 agent posterior features of
the same draw, whose positions are known; it agrees with the decoder head to within 0.1 on
the agent's own imagined trajectories (seed 1, right, t = 8: −0.10 decoded, +0.00 by NN).

## 10. Confirmation under the amended lane (2026-09-28)

The reference lane now acts on the keyed recognition face (`infer_states(key=)`,
`Config.act_sampling`, default on — `execution-ledger-acting.md` D6). Same keys, same
fixtures, 16,384 steps (`runs/confirm16k_*`):

| key | as shipped: first success / success 12–16k | amended lane: first success / entropy 12–16k / success 12–16k / coverage |
|---|---|---|
| 1 (trapped) | 17,556 (in the 100k run) / 0.00 at 16k | **2,228** / −0.16 / 0.65 / 0.66 |
| 11 (trapped) | none / 0.00 | **1,180** / 0.19 / 0.98 / 0.74 |
| 15 (trapped) | none / 0.00 | **3,216** / 0.06 / 0.91 / 0.70 |
| 4, 6, 7 (healthy) | 2,016 / 1,544 / 1,132; 1.00 / 0.95 / 1.00 | 1,812 / 1,564 / 1,136; 0.11 / 0.07 / 0.15; 1.00 / 0.94 / 1.00; 0.70–0.73 |
| 5 (healthy) | 1,776; 0.86 | 2,996; −0.22; 0.60; 0.63 |

All three trapped keys are ordinary under the amended lane; the healthy keys are unchanged
within the spread their own twenty seeds show (0.38–0.72 success at 12–16k). What the
amended recipe does to the converged window is not measured here.

## 11. The shape that remains: the double pump (2026-09-28, open)

Under the amended lane the 500k run's early phase matches the oracle (first successes 1,872 /
2,228 / 2,340 / 2,372; act-time entropy on theirs from 4–8k: −0.24 … −0.39 against their
−0.22 … −0.35 at 48–64k). What remains is the "creep" `plateau-analysis.md` §0 named: from
16–32k on, three of four seeds' *successful* episodes lengthen from ~44 steps to 60–70 and
stay there for hundreds of k, while the oracle's fall 43 → 39. The as-shipped runs under both
initializers show the same regression, so it is not the lane and not the initializer.

**What it is.** A second pump. Their replay (read from `logs/oracle_state_mc_seed*/replay/`,
last 500k steps, 53k successful episodes): a 35-step single pump — one velocity reversal — in
76% of successes, an initial wobble in 19%, a genuine second pump (≥ 3 reversals, ~66 steps)
in 4%; at 12–16k on their sixteen probe runs the second pump is 15% and the median is 40–44.
Ours at 60k (`runs/actsample_500k/snapshots_step60416`): seeds 0–2 median 61–67 steps with
the extra reversal, seed 3 median 40 with one. Same action amplitude (|a| 0.89–0.91, 61%
saturated; theirs 0.94, 79% converged, 0.80, 55% early), same σ (0.19–0.21), same act-time
entropy; the slow seeds' *mean* pushes against the velocity on 6–16% of mid-swing states
(seed 3: 4%; their recorded actions: 9% converged) — marginal for a categorical difference.

**What it is not** (all on the 60k snapshot, seeds 0 and 3):
- Not the world model's accuracy: open-loop velocity-sign accuracy 0.90–0.93 (0.83–0.91 from
  |v| < 0.01) on both seeds; the actor's own imagined rollouts contain the same mid-swing
  disagreement as its real ones (0.81 vs 0.84; 0.87 vs 0.93).
- Not under-optimization: through `imagine`'s own rollout and λ-return from the same real
  states, a bang-bang energy-pumping stand-in (mean = sign of the decoded velocity, the
  actor's σ) scores the same as the actor (−5.15 vs −5.12; bang-bang higher in 49% of starts;
  per-step value, continuation and critic tail within 0.01) on both seeds. The objective the
  actor is handed is blind to the difference at this stage.
- Not the critic — on either side. Ours at 60k is constant across states (V mean −0.50, SD
  0.002–0.007 over 2,048 real states, all seeds) and lags the λ-return by 4.4; at 500k it
  spans 12 units and correlates 0.70 with mechanical energy. Theirs at 16k is constant too
  (SD 3–4 × 10⁻⁴, corr with energy 0.05–0.14; read under their code on
  `probe16k_state_mc_seed{4,5}`) and at 2M discriminates (SD 18, corr 0.66–0.72); their logged
  critic mean tracks ours through 256k and its spread opens at 48–96k. Their agents single-pump
  by 32k with a critic as flat as ours.
- Not the self-imitation term's strength (weight mean 0.03), though its targets mirror each
  seed's own path (slow seeds: 61–67-step episodes; seed 3: 40).

**Open.** Where their single pump comes from at 16–32k with a flat critic — inside the 8-step
horizon: the continuation head, the cosine leaf's shape along the actor's own trajectories,
or the imitation term — and why ours does not. Decidable by the same return probe on their
side (their actor against a bang-bang stand-in through their λ-return, per term) at 16k and
64k; their 64k checkpoints are queued behind the 500k run. The probes behind this section
(episode shape, imagined rollouts, plans, returns and the critic, on both sides) were session
scripts and were not kept.

## 12. The 500k run under the amended recipe (2026-09-29)

`runs/actsample_500k`: four seeds, their initializer, fixtures, mean-field preferred rollout,
act-time sampling on, code `57235fb` (clean), 17.6 h. **It differs from `crspp_init_100k`
in two recipe changes at once** — the act-time sample and the divergence-21 closure (training
on the pre-step states, closed as "statistically benign" on argument) — which was a design
error of the run; the converged-window difference below is not yet attributable to either.

| | first success | successful-episode length p50 by 100k tile | 400–500k tile (mean return) | share of episodes < −100 at 400–500k | act-time entropy 400–500k |
|---|---|---|---|---|---|
| seed 0 | 1,872 | 61 → 51 → 42 → 37 → 36 | −72.4 | 0.04 | −0.52 |
| seed 1 | 2,228 | 53 → 45 → 40 → 37 → 36 | −72.0 | 0.04 | −0.54 |
| seed 2 | 2,340 | 63 → 51 → 41 → 36 → 36 | −68.9 | 0.00 | −0.53 |
| seed 3 | 2,372 | 41 → 41 → 41 → 37 → 37 | −79.9 | 0.14 | −0.55 |
| as shipped (`crspp_init_100k`) | 1,812 / 17,556 / 2,180 / 1,996 | — | −67.9 / −67.8 / −67.5 / −69.4 | 0.00–0.02 | −0.60 … −0.67 |
| oracle | 1,120–4,228 (20 seeds) | 43 → 39 by 32k | last 500k: −70.7 … −73.2; best-200k −68.1 … −69.5 | 0.03–0.06 | −0.67 … −0.72 |

The early phase is the oracle's: every seed succeeds by 2.4k and the double pump of §11
resolves by 300k; over the last 16k the policy is theirs (35–36-step single pumps, 76–77%
saturated, action/velocity sign agreement 0.87–0.89; theirs 35, 79%, 0.89). The converged
window is not on the acceptance band by the best-200k metric (−73.3 ± 4.0 against
−68.1 … −69.5): the medians are (−68 … −70), the means carry a tail of long-but-successful
episodes — recurring double pumps, 4% / 4% / 0% / 14% — that the as-shipped run did not
have (0–2%) and that the oracle's own late window does (3–6%). The actor at the end is less
committed than either (−0.52 … −0.55 against −0.60 … −0.72). Whether that tail is the
act-time sample's price or divergence 21's is the ablation to run (`--no-act-sampling` on
the current code against `crspp_init_100k`).

Instrument note: this run predates the two-path probe columns (commit e2e058d); the next
run carries them.

## 13. The onset, at 1k resolution: the leaf lets go first (2026-09-29)

Seed 0 of the 500k run replayed deterministically in 4k-step resumes to 32k and then in
1k-step resumes from 16k to 28k, every stage kept (`runs/causal32k_key0_s*`,
`runs/causal1k_key0_s*`), and read at each stage on the same 512 real posterior states: the
training-side preference between the actor and a bang-bang pumping stand-in (share of starts
where bang-bang's λ-return is higher), the cosine term's per-plan preference through
`score_policies` (A − B; every other term is 0), the CRSPP attractor's decoded position, and
the successful-episode length in the 1k bin that follows.

| stage | bang-bang preferred (λ-return) | cosine A − B | attractor position (t = 8) | episode length, next 1k |
|---|---|---|---|---|
| 4k–16k | 0.78–0.85 | +0.17 (16k) | −0.40 (16k) | 41–43 |
| 17.4k | 0.68 | +0.15 | −0.35 | 45 |
| 18.4k | 0.69 | +0.17 | −0.35 | 41 |
| 19.5k | 0.63 | +0.12 | −0.34 | 44 |
| 20.5k | 0.66 | +0.15 | −0.33 | 42 |
| 21.5k | **0.54** | **+0.09** | **−0.27** | 43 |
| 22.5k | 0.48 | +0.06 | −0.27 | **72** |
| 23.6k | 0.58 | +0.08 | −0.27 | 75 |
| 24.6k | 0.54 | +0.08 | −0.30 | 45 |
| 25.6k–28.7k | 0.36–0.44 | +0.02 … +0.05 | −0.25 … −0.29 | 69–76 |
| 60k–116k (snapshots) | 0.49–0.58 | −0.03 (60k) | −0.5 … −0.7 | 57–67 |

The order is: the attractor moves (−0.33 → −0.27 between 20.5k and 21.5k) and the cosine
term's preference for pumping halves (+0.15 → +0.09 → +0.06) *one bin before* the double
pump appears (72-step episodes from 22.5k); after it appears the term goes on to endorse the
actor's own path (bang-bang preferred in 36–44% from 25.6k). The training series over the
same window — `crspp_positive`, the CRSPP loss, `bc_weight`, the logged cosine level — show
nothing (0.63–0.67 throughout).

**Their side at matched stage** (`logs/probe64k_state_mc_seed{0,1}`, their code, their
λ-return, per term): at 64k the objective is indifferent (bang-bang better in 49% / 40% of
starts, every term within 0.01; the cosine term mildly favours *their* actor on seed 1,
−0.006), the critic is flat (V SD 0.012 / 0.028) — the same flat landscape ours has from 22k
on. Their agents single-pump on it; ours, once the leaf lets go, drifts to the double pump.
*Reading:* on both sides the single pump is *installed* by the early leaf and the fixture
imitation, and *held* on a flat objective; the difference is that our leaf's preference
collapses at ~21k — with the CRSPP attractor's move — and theirs does not (theirs was weak
at 16k, +0.0005 / +0.007, and is still mildly pro-actor at 64k). Why our attractor moves at
~21k — the CRSPP's training inputs at that point (the positives' composition as own
successes outnumber the fixtures; register row 6.5's mean-field rollout) — is the next
question, not answered here.

## 14. Which learner carries the actor there: counterfactual replays from 20.5k (2026-09-29)

Seed 0's kept 20.5k stage (`runs/causal32k_key0_s20480`) resumed to 28.7k seven more times
under the recipe as shipped except for one learning rate (or the entropy coefficient) held at
zero from the resume on — `train_step` monkeypatched from a copy of the stage, nothing else
changed. These are attribution instruments, not candidate recipes: the recipe is theirs and
every module learns on their schedule; the question is which module's *continued movement*
after 20.5k the actor is following when it leaves the single pump. Read as the share of
successful episodes longer than 50 steps per 1k bin (the single-pump regime's own spread is
25–35%; a double-pump bin is 60–90%), with the median successful length as a check.

| held from 20.5k | double-pump share per 1k bin, 20.5k → 28.7k | overall | median length |
|---|---|---|---|
| nothing (as shipped) | 29 33 **67 77** 33 54 **89 89** | 53% | 42 43 72 75 45 69 69 76 |
| CRSPP | 35 50 **67 65 69** 56 50 58 | 56% | 42 59 75 69 70 67 51 54 |
| world model | 35 56 29 50 53 50 22 45 | 42% | 42 65 40 57 67 47 40 50 |
| critic | 29 33 **67 79** 50 **82 64** 54 | 55% | 42 43 72 73 53 78 69 67 |
| actor entropy bonus (ACTENT = 0) | 31 44 **62 67** 44 **71 73 65** | 56% | — |
| world + critic + CRSPP (actor alone learns) | 35 44 29 33 25 33 38 27 | **33%** | **42 44 42 42 43 42 42 42** |
| critic + CRSPP (world alone learns) | 35 50 **67** 58 57 **60** 44 47 | 51% | 42 59 71 74 74 67 43 48 |
| world + CRSPP (critic alone learns) | 35 44 29 33 25 33 38 27 | 33% | 42 44 42 42 43 42 42 42 |
| world + critic (CRSPP alone learns) | 35 **60** 47 **60** 31 **78 64** 50 | 51% | 42 68 43 76 42 78 75 60 |

What this settles:

- **The actor does not get there on its own.** With every model held at its 20.5k state
  the median successful episode stays at 42 steps in every bin to 28.7k, and the share
  above 50 stays at the single-pump baseline. The drift is not actor ascent on the 20.5k
  objective, and not the entropy bonus.
- **No single learner is necessary; two are each sufficient.** Holding any one of the
  world model, the critic or the CRSPP leaves the double pump in place (the world-model
  hold mutes it, 42%, within the noise of 13–18 successes per bin). Letting the world
  model learn alone brings it back on schedule (67% at 22.5k, medians 71–74); letting the
  CRSPP learn alone brings it back too (medians 68–78 in four of seven bins). Letting the
  critic learn alone reproduces the all-held table bin for bin — the critic is a near
  constant in this window (`critic_value` 0.099 → 0.089 over 8k steps) and its movement
  changes nothing the actor does.

So §13's reading — that the CRSPP attractor's move *causes* the leaf to let go — is
narrowed: the CRSPP's movement is one sufficient route, the world model's is another, and
the double pump appears with either. What they share is the size of what the actor was
holding on to.

**The margin, and what moves it** (the swap probe: the 20.5k actor's plan
against a bang-bang plan from the same 512 real states, the actor *held at 20.5k*, scored
per term through `score_policies` under the 20.5k world model and under a later one):

| world model | bang-bang preferred | cosine, actor − bang | predicted reward, actor − bang |
|---|---|---|---|
| 20.5k stage | 0.68 | +0.150 | 0.000 |
| as shipped, 22.5k | 0.67 | +0.133 | −0.001 |
| as shipped, 24.5k | 0.68 | +0.136 | −0.004 |
| world learned alone to 28.7k | 0.59 | +0.106 | **−0.105** |
| all held to 28.7k (control) | 0.68 | +0.150 | 0.000 |

The single pump's whole standing in the objective is a cosine margin of ~0.15 nats over an
8-step open-loop plan, and nothing else (`g_tail`, `model_uncertainty` and the reward leaf
are 0.000 apart at 20.5k). Under the world model that learned alone, the reward head comes
to favour the actor's own plan by 0.105 — the same size as the cosine margin, cancelling
it — while the cosine leaf itself weakens by a third. Under the as-shipped world model at
22.5k and 24.5k the reward leaf is still flat and the objective still prefers bang-bang on
two thirds of the states with the 20.5k actor held; there the movement is in the CRSPP
(§13's attractor move, the cosine margin +0.15 → +0.09 with each stage's own plan). Two
routes, one shallow basin: the margin is small enough that the ordinary movement of either
learned model, on its own schedule, erases it.

What this does not settle: *why the reward head, learning on the actor's own trajectories,
comes to score the actor's plan above bang-bang* (the reward is −1 per raw step until the
goal, with no action cost, so a true difference can only come from reaching the goal within
the window — or from the head's extrapolation off the actor's action distribution; the
ground-truth check, executing both plans from the same real states in the env, was not
run), and why their agents hold the single pump on a landscape §13 measured as equally flat
at 64k. The next discriminating measurement is on their side: the same per-term margin at
20–28k under their code, and whether their reward head or CRSPP moves by the same amount
over the same window.

## 15. Their side at 1k resolution: the leaf lets go there too (2026-09-29)

Their code and recipe, seed 0, trained to 28,672 steps with the checkpoint copied every
~1k steps (`raif-paper/logs/probe28k_state_mc_seed0_s*`), and each copy read by a probe run
under their code: their λ-return on their actor's sampled rollout
against a bang-bang stand-in from the same 512 real posterior states, split per term
through their own `score`. Their agent single-pumps throughout (median successful length
37–40 in every 1k bin from 14k, the share above 50 steps 0–25%, every episode a success).

| step | bang-bang preferred (λ-return) | 8-step CRSPP margin, bang − actor | reward / IG / critic | actor \|a\| | sign agreement with bang-bang |
|---|---|---|---|---|---|
| 15.4k | 0.66 | +0.037 | 0.000 | 0.87 | 0.63 |
| 16.4k | 0.65 | +0.032 | 0.000 | 0.88 | 0.65 |
| 17.4k | 0.60 | +0.030 | 0.000 | 0.88 | 0.65 |
| 18.4k | 0.59 | +0.020 | 0.000 | 0.88 | 0.68 |
| 19.4k | 0.56 | +0.017 | 0.000 | 0.89 | 0.67 |
| 20.4k | 0.50 | +0.005 | 0.000 | 0.88 | 0.69 |
| 21.4k | 0.52 | −0.002 | 0.000 | 0.88 | 0.69 |
| 22.4k | 0.44 | −0.010 | 0.000 | 0.89 | 0.71 |
| 24.4k | 0.45 | −0.024 | 0.000 | 0.91 | 0.71 |
| 26.4k | 0.42 | −0.019 | 0.000 | 0.91 | 0.73 |
| 28.7k | 0.44 | −0.024 | −0.009 (reward) | 0.92 | 0.74 |

Their leaf lets go on the same schedule as ours: the CRSPP margin for bang-bang decays to
zero at 20.4k and reverses to favour their own actor from 22.4k, the bang-bang share
falling 0.66 → 0.44 — ours went +0.15 → +0.09 (21.5k) → +0.06 (22.5k), 0.66 → 0.48, over
the same bins (§13). §13's reading — that our leaf's collapse is the difference — is
therefore withdrawn: the collapse is the shared phase. The single pump is installed early
by the leaf and the fixture imitation on both sides, and from ~21k on both sides the
objective is flat between the single pump and bang-bang. Their actor holds through it;
ours leaves within one bin.

What differs is the actor's response to a flat landscape. Theirs keeps sharpening: mean
|a| 0.87 → 0.92 and sign agreement with bang-bang 0.63 → 0.74 across the window, an actor
consolidating toward saturated single pumping while nothing in the leaves asks for it.
Ours does not: |a| 0.84–0.88 and sign agreement 0.65–0.67 flat through 24.5k (§14's swap
probes, each stage's own actor), and then the abrupt 72-step bin. §14 says which of our
learners' movement the slide follows (world model or CRSPP, either one); this section says
the objective's preference is not where the two sides part. The divergence is downstream
of the leaves — in how the actor update turns an indifferent objective into movement:
the advantage normalisation (`perc_ema` return scaling), the self-imitation term and the
share of `CAN_SELF_IMITATE` weights, the policy-entropy bonus, the sampled rollouts'
variance. None of these passes through `score_policies`, which is why §11–§14 could not
see it.

Next: decompose the actor update at the 20.5k stage on both sides — the advantage term,
the entropy term and the self-imitation term of one imagined batch, their magnitudes and
the sign of the advantage on "more swing" versus "less swing" starts, and the
self-imitation weight share — ours from `imagine`'s loss pieces, theirs from
`ImagActorCritic.train`.

## 16. The actor update's own statistics, both sides: they match (2026-09-29)

§15 put the divergence downstream of the leaves, in the actor update. Both sides log that
update's pieces every training chunk, and both use the same form — backprop through the
reparameterised imagined rollout (`actor_grad_cont: backprop` in their config, the
reparameterised draw in our `roll_batch`), not REINFORCE — so the logged series are the
same quantities. Ours from the as-shipped seed-0 replay (`runs/causal1k_key0/metrics.jsonl`),
theirs from the §15 run (`train/*` rows, one per ~2.4k steps):

| step | λ-return, ours | theirs | advantage mean, ours | theirs | policy entropy, ours | theirs | return-norm range, ours | theirs (normed sd) |
|---|---|---|---|---|---|---|---|---|
| 16.4k / 16.6k | +0.90 | +0.88 | +0.37 | +0.40 | −0.09 | −0.01 | 2.2 | 0.23 |
| 19.5k / 19.0k | −0.37 | −0.04 | −0.36 | −0.13 | −0.11 | +0.01 | 1.3 | 0.24 |
| 21.5k / 21.4k | −1.17 | −1.03 | −0.85 | −0.79 | −0.03 | −0.03 | 1.5 | 0.42 |
| 23.6k / 23.8k | −1.80 | −1.98 | −0.85 | −0.81 | −0.26 | −0.08 | 2.2 | 0.43 |
| 26.6k / 26.2k | −2.59 | −2.85 | −0.78 | −0.70 | −0.28 | −0.10 | 3.4 | 0.40 |
| 28.7k / 28.6k | −2.95 | −3.35 | −0.73 | −0.63 | −0.22 | −0.10 | 4.1 | 0.37 |

The imagined λ-return falls from +0.9 to −3 on both sides over the same window (the
reward head learning the −1 per step, which the flat critic does not follow), the mean
advantage crosses to −0.8 on both sides at ~21k and stays there, and the self-imitation
term on ours is small throughout (`bc_term` −0.002 … −0.020, weight share 0.02–0.26).
The one visible difference is the policy entropy — ours −0.22 … −0.28 against theirs
−0.08 … −0.10 from 23k, a *less* random actor on our side, which is the wrong direction
for "ours diffuses on a flat landscape".

So the actor update's aggregate signal matches. What is left is what the two sides'
*models* say about the actor's own path: §14 showed that either learned model's movement
is sufficient to carry our actor into the double pump, and the only-world route's reward
head came to favour the actor's plan by as much as the cosine leaf disfavoured it. The
natural reading is model exploitation — our world model, learning on the actor's own
trajectories, comes to predict more progress for the gentle pump than the env delivers,
and the actor follows the model. That is testable on both sides in one measurement: the
actor's own plan rolled in the model against the same plan rolled in the env from the same
real states, the position and velocity error over 8 steps and the predicted-versus-actual
reward, at the 20.5k and 24.5k stages. Not run.

## 17. Model exploitation, precision, and what a population says (2026-09-29, evening)

§16 left the models as the last place the two sides could differ. The check: the actor's
own 8-step plan and a bang-bang plan, from the same real states, rolled in the model and
executed in the env (ours directly; theirs by re-executing their probe's recorded arrays with
the env's physics).

**Position and speed do not discriminate.** Both models run the actor's plan further ahead
of the env than bang-bang by the same margin (step 8: ours +0.022/+0.024 at 20.5k/22.5k,
theirs +0.021/+0.014 at 20.4k/22.4k, both shrinking to ~+0.005 by 28k). Their model
over-predicts speed for both plans, ours under-predicts for both; neither favours a plan by
more than 0.001 of a 0.07 maximum.

**The continuation head did, on seed 0.** Share of starts where the model has the goal
reached within 8 steps, as a fraction of the env's truth:

| stage | ours: actor plan / bang-bang | theirs: actor plan / bang-bang |
|---|---|---|
| 20.5k | 0.86 / 0.68 | 0.55 / 0.52 |
| 22.5k | 0.98 / 0.82 | 0.76 / 0.73 |
| 24.5k | 0.88 / 0.84 | 0.82 / 0.82 |
| 28.7k | 0.97 / 0.92 | 0.77 / 0.74 |

Their head is even-handed at every stage; ours, at the moment the leaf lets go, credits the
actor's own plan with the goal about as often as the env delivers it and under-credits
bang-bang by 16–18 points, narrowing to 5 by 28.7k. Behind it, on seed 0, our head learns
termination earlier: accuracy on terminal rows 0.07 / 0.47 / 0.59 / 0.72 at 16.4k / 20.5k /
24.6k / 28.7k against their seed 0's 0.00 / 0.02 / 0.13 / 0.29 / 0.63 at 16.6k / 19k / 21.4k
/ 23.8k / 26.2k. Everything that feeds that head was then checked and found matched:
terminal-row share of a batch (~1.5% both), head configuration (2 × 32, silu, layer norm,
binary, outscale 1), Bernoulli loss, update cadence (one step per 16 pooled env steps, four
envs = four lanes), batch 16 × 64, replay capacity 500k, the replay's action statistics
(mean |a| 0.870 / 0.872, saturated share 0.44 / 0.41), the world-model losses (dyn/rep KL
1.00–1.03 both, decoder ~0.01 both), and the M/M⁺ alternation (dead on their side too:
`positive_replay/samples` is 0 in their own metrics). The one input left was their float16
compute, which the 2026-09-15 control had cleared only of overflow.

**Precision: negative.** Their seed 0 at `--jax.precision float32` to 32k
(`logs/probe32k_fp32_seed0`, snapshots per 1k): terminal-row accuracy 0.02 / 0.19 / 0.45 /
0.52 / 0.61 at 19k / 21.4k / 23.8k / 26.2k / 28.6k — the float16 timeline within noise; the
per-term margin series is the float16 one column for column (leaf lets go by 19.6k and
reverses, |a| 0.88 → 0.91, sign agreement 0.62 → 0.69); the single pump holds (median 38
steps in every 2k bin, double-pump share ≤ 18%). The register's float16 row stays argued,
now with a behavioural control behind it.

**The population withdraws the seed-0 finding.** Terminal-row accuracy on twelve of our
seeds at 20.5k (`runs/sweep16k_off{4,8,12}_to20k`, the amended lane's healthy seeds resumed
from their 16k checkpoints): 0.02, 0.11, 0.00, 0.36, 0.10, 0.00, 0.03, 0.00, 0.00, 0.00,
0.30, 0.00 — six of twelve at zero, median 0.01; at 16.4k, eleven of sixteen at zero. Their
three values near 21k are 0.10, 0.19, 0.19. Our seed 0's 0.47 is an outlier of ours, not a
property of our head, and "our head learns termination earlier" is withdrawn. Whether the
*plan-relative* asymmetry (the table above) is seed 0's alone, and whether it predicts
which of our seeds double-pump — three of four do at 500k — is the population question the
twelve 20.5k checkpoints can now answer, and is being asked (the goal-reaching probe on all
twelve, then each resumed to 28.7k for its fate). Not answered here.

What stands after this section: the trap mechanism and its fix (§8–§10); the double pump
needs a learned model to move and either the world model or the CRSPP suffices (§14); the
objective's collapse at ~21k, the actor update's statistics, the world-model losses and the
continuation head's timeline are all shared with their side (§15, §16, §17); and the
divergence that remains — ours leaves the single pump, theirs does not — is not yet traced
to a mechanism. The instruments and the checkpoints are in place; the next reads are the
population ones.

## 18. Twelve seeds: the informed, actor-favouring head marks the seeds that double-pump (2026-09-29, late)

The twelve 20.5k checkpoints of §17 (`runs/sweep16k_off{4,8,12}_to20k`, four seeds each)
read with the goal-reaching probe — the share of starts where the model has the goal within
8 steps, as a fraction of the env's truth, for the actor's own plan and for bang-bang — and
then each resumed as shipped to 28,672 (`runs/*_to28k`) and read as the double-pump share
of successful episodes per 1k bin:

| seed | head at 20.5k: actor plan / bang-bang | double-pump share, 20.5k → 28.7k | overall |
|---|---|---|---|
| off4-1 | 0.33 / 0.21 | 58 67 47 60 71 53 85 93 | **67%** |
| off4-3 | 0.54 / 0.47 | 50 73 67 69 71 76 81 50 | **67%** |
| off12-2 | 0.76 / 0.65 | 42 75 47 71 25 62 57 58 | **54%** |
| off8-0 | 0.16 / 0.13 | 59 10 24 35 15 47 24 26 | 29% |
| off4-0 | 0.05 / 0.00 | 9 4 12 12 17 9 4 12 | 10% |
| off8-2 | 0.02 / 0.00 | 33 15 26 11 5 10 19 25 | 17% |
| off12-0 | 0.02 / 0.00 | 9 0 9 8 4 13 14 9 | 8% |
| off4-2, off8-1, off8-3, off12-1, off12-3 | 0.00 / 0.00 | (single pump) | 12–18% |

Twelve of twelve agree: every seed whose continuation head knew the goal at 20.5k and
credited the actor's own plan above bang-bang (gaps +0.07 … +0.12) double-pumps by 28.7k;
every seed whose head was still uninformed holds the single pump through 28.7k; the one
mildly informed head (off8-0, +0.02) sits between. Their heads reach the same knowledge on
the same timeline (§17: 0.10–0.19 at 21k, ~0.6 by 26k) and stay even-handed (gap ≤ 0.03 at
every stage, §17's table), and their agents do not leave the single pump.

So the population narrows the divergence to one object: **once our continuation head knows
the goal, it credits the actor's own plan more than it credits an off-policy plan; theirs,
knowing the same, does not.** Two things this does not yet settle. *Order:* on seed 0 the
informed, asymmetric head precedes the double pump by ~2k (§14, §17); on off4-1 and off4-3
both are present by the 20.5k read, so an informed head could there be learning from
double-pump episodes rather than producing them — a 1k-cadence read of those seeds from
16.4k resolves it. *Source:* the head's architecture, loss, cadence, batch, replay
statistics and precision are matched (§17); what has not been compared is the head trained
on the *same data* on both sides — our replay through their world model and theirs through
ours — which is the experiment that separates the model code from the data it saw.

## 19. The order, at 1k resolution: the head is a marker, not the cause (2026-09-29, night)

The three seeds that double-pump by 28.7k (§18), resumed from their 16.4k checkpoints in
1k stages to 22.5k with every stage kept (`runs/sweep16k_off{4,12}_order_s*`), the
continuation head read at each stage (goal within 8 steps, model / env, actor's plan then
bang-bang) beside the double-pump share of the bin that follows:

| seed | 16.4k | 17.4k | 18.4k | 19.5k | 20.5k | 21.5k | 22.5k |
|---|---|---|---|---|---|---|---|
| off12-2, head | 0.62/0.60 | 0.77/0.73 | 0.67/0.70 | **0.83/0.74** | 0.76/0.65 | 0.95/0.96 | 0.95/0.90 |
| off12-2, double pump % | 17 | 25 | 21 | 25 | **41** | **75** | — |
| off4-1, head | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | **0.30/0.18** | 0.33/0.21 | 0.70/0.49 | 0.69/0.48 |
| off4-1, double pump % | 8 | 11 | **50** | 44 | 58 | 66 | — |
| off4-3, head | 0.00/0.00 | **0.16/0.09** | 0.23/0.14 | 0.64/0.51 | 0.54/0.47 | 0.74/0.57 | 0.86/0.76 |
| off4-3, double pump % | 21 | **46** | 58 | 71 | 50 | 73 | — |

The order is not one order. On off12-2 the head knew the goal from 16.4k and was
even-handed for three stages (gap ≤ 0.04); the gap opened at 19.5k (+0.09) and the double
pump followed at 20.5k — the head first, by one bin, as on seed 0 (§17, by two). On off4-1
the double pump was there at 18.4k (50%) with a head that knew *nothing* of the goal
(0.00 / 0.00 through 18.4k); the head learned the goal, asymmetric from its first informed
read, one bin later. On off4-3 the two appear in the same bin. So the informed,
actor-favouring head is a *marker* of the seeds that leave the single pump — it never
appears on a seed that holds, and it never appears even-handed on ours once the drift is
under way — but it is not the cause: a seed can leave with a head that has learned nothing,
and once it has left, an on-policy head trained on its own double-pump episodes will credit
the actor's plan. That reading also explains why their heads stay even-handed: their actors
never leave, so there is nothing on-policy for the head to favour.

What this leaves. §14 stands: the drift needs a learned model to move, and either the world
model or the CRSPP suffices; off4-1 says the world model's contribution is not its
continuation head. §15 and §16 stand: the objective's collapse at ~21k and the actor
update's statistics are shared. The mechanism that lets our actor slide on a flat landscape
where theirs holds is still not traced; the population instruments (twelve seeds at 16.4k,
20.5k and 28.7k; three at 1k cadence over the onset) are in place for the next read, which
is the cross-data experiment (§18) or a per-seed decomposition of *which* model movement —
transition, decoder, reward head, CRSPP — precedes each seed's onset.

## 20. The ablation at 500k, and the double pump as a phase (2026-09-30)

The 500k ablation (`runs/noactsample_500k`: mean-field acting as before §10, the
divergence-21 closure kept) finished at 500,736 steps in 13.6 h. Beside the act-sampling
run (§12) and the reference tile:

| run | 400–500k mean return per seed | mean | median successful length | double-pump share |
|---|---|---|---|---|
| ablation (mean-field acting) | −72.8 / −68.8 / −71.0 / −67.7 | −70.1 | 36 / 36 / 36 / 35 | 2 / 0 / 5 / 0 % |
| act-time sampling (§12) | −72.4 / −72.0 / −68.9 / −79.9 | −73.3 | 36 / 36 / 36 / 37 | 4 / 4 / 0 / 16 % |
| `crspp_init_100k` (reference) | −67.9 / −67.8 / −67.5 / −69.4 | −68.2 | — | — |

Two things this settles. **Act-time sampling is the trap's fix and nothing else.** Under
the ablation one seed of four sat at −200 at 16k (§12's confound is resolved: the trap
belongs to mean-field acting, not to the divergence-21 closure), recovered by 30k, and by
the converged window the two recipes are within one seed's spread of each other — the
ablation slightly ahead, the sampled run's seed 3 the only one still double-pumping (16%).
The converged tail is not what the act-time sample buys. **The double pump is a phase, not
a destination.** Median successful length per 50k bin (double-pump share in brackets):

| seed | 0–50k | 50–100k | 100–150k | 150–200k | 200–250k | 250–300k | 300–500k |
|---|---|---|---|---|---|---|---|
| ablation 0 | 46 (28) | 61 (94) | 57 (73) | 49 (44) | 46 (17) | 44 (10) | 36 (0–4) |
| ablation 1 | 45 (5) | 63 (64) | 66 (94) | 45 (27) | 41 (2) | 36 (0) | 36 (0) |
| ablation 2 | 51 (52) | 64 (100) | 59 (84) | 48 (27) | 45 (3) | 42 (11) | 36 (1–7) |
| ablation 3 | 44 (8) | 41 (0) | 41 (0) | 41 (7) | 44 (43) | 42 (39) | 36 (0–6) |
| sampled 0 | 49 (50) | 61 (81) | 56 (76) | 47 (28) | 42 (0) | 40 (2) | 36 (3–5) |
| sampled 1 | 64 (60) | 53 (61) | 48 (33) | 43 (8) | 41 (1) | 39 (0) | 36 (1–26) |
| sampled 2 | 48 (46) | 65 (94) | 58 (78) | 48 (30) | 43 (7) | 40 (2) | 36 (0–3) |
| sampled 3 | 41 (16) | 40 (5) | 41 (10) | 41 (21) | 42 (26) | 40 (20) | 37 (12–19) |

On six of eight seeds the double pump is the dominant behaviour from ~50k to ~150k (64–100%
of successes), fades through 200k, and is gone by 250–300k; every seed ends at a 35–37-step
single pump, two steps *faster* than their 38 (§15). Their agents never enter the phase.
So §11–§19's divergence is a divergence in the *transient*: the road to the same converged
policy passes through a slow basin on our side and not on theirs, and the converged
policies match. The acceptance window (§12) never saw the phase because it is over by
then; the shape reads at 60k–116k (§13's last row) were taken inside it.

What the phase costs, in the metric the acceptance is written in: nothing measurable in
the 400–500k tile (the ablation's mean −70.1 sits within one seed's spread of the
reference's −68.2), and about 100k steps of slower episodes on the way. Whether that is
worth tracing further — §18/§19's population instruments and the two proposed probes stand
— is a question about how much of the learning curve, rather than the endpoint, parity is
meant to cover.

## 21. Which model part moves the objective before an onset: none alone, and the holders move too (2026-09-30)

The 1k stages of §19, read as hybrids: every parameter from stage `s`, with exactly one
part taken from stage `s+1` — the transition, the recognition (encoder and posterior), the
decoder, the reward head, the continuation head, the CRSPP, the ensemble — and the
combinations (the whole world model; everything but the actor; the actor alone; all of
`s+1`). Each hybrid read on the actor's own training-side objective (`imagine`'s
`roll_batch` and `lambda_return`, the actor's sampled rollout against a bang-bang
stand-in whose sign rule is held at stage `s`'s decoder, 512 starts from stage `s`'s buffer,
16 rollout keys; the critic held at `s`, inert per §14), for the three drifters at the two
stage pairs bracketing their onsets and for three holders at their run-mates' pairs
(`evidence/decomp.md`). Shift = Δλ(hybrid) − Δλ(base) in units
of 10⁻³; positive is toward the actor's own plan.

| pair | base Δλ (actor − bang) | bang better | transition | recognition | reward | cont | CRSPP | whole world model | all but actor |
|---|---|---|---|---|---|---|---|---|---|
| off4-1* 17→18k | −35.8 | 0.96 | +1.5 | +0.8 | +0.4 | +0.5 | −0.8 | +3.4 | +3.2 |
| off4-1* 18→19k | −31.9 | 0.95 | +0.6 | +0.4 | +0.7 | +0.9 | +0.5 | +3.4 | +3.6 |
| off4-3* 16→17k | −31.3 | 0.91 | −1.8 | +5.0 | +0.4 | 0 | +0.5 | −0.3 | +4.9 |
| off4-3* 17→18k | −25.4 | 0.88 | −0.1 | −0.3 | +1.8 | +0.3 | +0.2 | +3.4 | +2.5 |
| off12-2* 19→20k | −13.8 | 0.74 | +0.5 | +0.2 | +2.9 | 0 | 0 | +2.8 | +2.9 |
| off12-2* 20→21k | −13.9 | 0.87 | −1.8 | −0.6 | +3.0 | +0.1 | −0.5 | +1.0 | −1.1 |
| off4-0 17→18k | −29.7 | 0.90 | −1.4 | +0.5 | 0 | 0 | +1.2 | −1.3 | +0.1 |
| off4-0 18→19k | −34.5 | 0.91 | −1.9 | +0.3 | +0.1 | +0.3 | −0.7 | +0.6 | +0.5 |
| off4-2 17→18k | −35.7 | 0.93 | +2.9 | −0.5 | 0 | 0 | +0.8 | +2.9 | +3.5 |
| off4-2 18→19k | −27.2 | 0.91 | −4.6 | +0.4 | 0 | 0 | +1.0 | −4.6 | −2.9 |
| off12-0 19→20k | −29.0 | 0.82 | −3.1 | +2.0 | +0.1 | +0.5 | +0.8 | +0.6 | +0.5 |
| off12-0 20→21k | −26.5 | 0.80 | +2.7 | +2.1 | +0.8 | +2.6 | +0.5 | +9.7 | +10.7 |

(* drifters. The decoder and the ensemble read exactly 0 in every row; the stand-in's rule
is held, and the ensemble row does not enter the return.)

Three readings, in order of strength. **No single part carries the movement.** On the
drifters the whole world model shifts the objective toward the actor's plan by 3 to 4
(×10⁻³) per 1k while no single part exceeds 3, and the reward head alone gives at most a
third of what it gives together with the transition — co-adapted parts moved separately
interact. The CRSPP contributes nothing at this resolution on any seed (≤ 1.2), and its
parameters moved *more* on two holders than on two of the drifters; §14's CRSPP-alone
route is a route over 8k, not a per-1k signature. **The holders move the same way.** The
largest shift in the study is a holder's (off12-0, 20→21k, +10.7, through the predicted
reward), and off4-2 swings ±3–5 through its transition. Per-stage movement of the model
toward the actor's own plan is ordinary learning on every seed, not an onset signature.
**What differs at base is a level, not a movement:** the predicted-reward term's advantage
for the actor's plan is +2 to +26 on the drifters and 0 to +7 on the holders — the same
shape as §18's continuation-head marker, and by §19's argument as likely a consequence of
on-policy data as a cause.

And one fact this table states more plainly than any before it: **the objective prefers
bang-bang on every seed and every hybrid**, drifter or holder, at every stage through the
onset — the actor's own rollout loses to the stand-in on 74–96% of starts by 14 to 36
(×10⁻³). The drifting actor is moving *away* from what its objective, compared between
these two plans, prefers. The comparison between two distant plans is not the signal the
actor follows; the signal is the local gradient of the return at its own plan, through the
model's dynamics. That is the next read, on both sides: the return's slope along the
direction from the actor's plan toward bang-bang, at the actor — whether the drifting actor
sits below a barrier that the holding one, and theirs, does not.

## 22. The return's slope toward bang-bang: positive on every seed of ours, negative on theirs from 21k (2026-09-30)

§21's closing question, measured with what the later sections call the slope probe, run on
both sides (`evidence/slope.md`): a stand-in policy that moves the actor's mean at every imagined state
toward the bang-bang rule, `m_ε = (1−ε)·m_actor + ε·sign(v̂)`, with the actor's own scale
and noise, and the sampled λ-return `R(ε)` from the same 512 real starts under the same
keys (8 keys; a second key set as the noise floor, ±1–3 ×10⁻³ unpaired, while every ε shares
its keys so the paired differences reproduce in sign on every row). `ε = 0` is the actor
bitwise; `ε = 1` is bang-bang with the actor's noise, §21's stand-in.

| ours, seed stage | R(.05)−R(0) | R(.1)−R(0) | R(.4)−R(0) | R(1)−R(0) | slope at .05 |
|---|---|---|---|---|---|
| off4-1* 17.4k / 18.4k / 19.5k | +1.6 / +1.2 / +1.3 | +3.4 / +3.1 / +2.8 | +16 / +15 / +12 | +37 / +32 / +25 | +31 / +25 / +26 |
| off4-3* 16.4k / 17.4k / 18.4k | +1.4 / +1.1 / +0.8 | +3.0 / +2.5 / +1.9 | +14 / +11 / +9 | +31 / +26 / +19 | +27 / +21 / +17 |
| off12-2* 19.5k / 20.5k / 21.5k | +0.5 / +0.9 / +0.5 | +1.5 / +1.8 / +1.2 | +6 / +6 / +4 | +13 / +13 / +4 | +10 / +19 / +10 |
| off4-0 (holder) 17.4k–19.5k | +1.2 … +1.5 | | | +30 … +35 | +24 … +30 |
| off4-2 (holder) 17.4k–19.5k | +1.2 … +1.6 | | | +26 … +36 | +24 … +31 |
| off12-0 (holder) 19.5k–21.5k | +0.6 … +1.4 | | | +18 … +30 | +12 … +27 |

| theirs, snapshot | R(.05)−R(0) | R(.1)−R(0) | R(.4)−R(0) | R(1)−R(0) | slope at .05 |
|---|---|---|---|---|---|
| 28k run 17.4k | +0.9 | +2.4 | +11 | +26 | +19 |
| 28k run 19.4k | +0.5 | +0.6 | +2.5 | +6 | +10 |
| 28k run 21.4k | −0.0 | −0.6 | −5 | −18 | −0.4 |
| 28k run 23.4k | −0.7 | −1.2 | −11 | −35 | −13 |
| fp32 run 17.6k / 19.6k / 21.6k | +1.0 / −1.0 / −1.5 | +2.3 / −2.4 / −2.9 | +11 / −7 / −15 | +24 / −15 / −44 | +20 / −19 / −31 |

(×10⁻³; * = drifters; R(ε) > R(0) on 78–95% of our starts at ε = .05, every row.)

**There is no barrier.** On every one of our eighteen seed-stages the return rises from the
first step toward bang-bang and keeps rising, near-linearly, to bang-bang itself; drifters
and holders have the same slopes (+10 … +31), and the lowest slopes in the study are a
drifter's. On their side the slope is ours at 17.4k and *reverses* by 19.6k (fp32) / 21.4k
(fp16), staying negative after: moving their actor toward bang-bang lowers their return
once §15's collapse has happened. Their actor holds the single pump because it sits where
its return says to sit. Ours does not: its return says "toward bang-bang" on every seed at
every stage through the onset, the holders do not go, and the drifters go the other way.

So the anomaly on our side is not the landscape and not the actor's position on it; it is
that **the actor's update does not climb the return along this direction** — on any seed.
What the finite difference does not see is the actor loss as built: the return
normalisation, the critic baseline, the rollout weights, the entropy and self-imitation
terms, and whatever the gradient path through the reparameterised draw and the dynamics
does with the slope. The next read is the actual update — one real actor step from each
stage, projected onto the bang-bang direction in action space — and, term by term, which
part of the loss disagrees with the return. The sampled step's straight-through code draw
is matched to theirs (`imagine._sampled_step`; their `jaxutils.py:108`), checked before this
section was written.

## 23. One real actor update, projected: the loss agrees with the return, and self-imitation dominates the gradient (2026-09-30)

§22 asked what the actor's actual update does along the direction its return rises in.
One real `train_step` from each stage (`evidence/update.md`, `evidence/update-groups.md`; the rebuilt actor loss
reproduces the shipped update bitwise), the actor's mean action read before and after on the
512 held states, projected on the bang-bang rule: ⟨Δm·sign(v̂)⟩, in 10⁻⁶, over 8 keys.

| seed | stage 1 / 2 / 3 |
|---|---|
| off4-1* | +121±36 / +68±48 / −6±31 |
| off4-3* | −2803±349 / +108±12 / +82±57 |
| off12-2* | −660±223 / +417±285 / +127±98 |
| off4-0 | +311±38 / +30±18 / −124±28 |
| off4-2 | −29±186 / +475±63 / −67±30 |
| off12-0 | −130±111 / +73±115 / −11±27 |
| theirs, 17.4k / 21.4k | −206±148 / −854±164 |

The actual step is nearly orthogonal to the bang-bang direction (|cos| ≤ 0.21), positive on
10 of 18 rows with no drifter/holder pattern, and about half of it is Adam's history rather
than the batch's gradient; along it the return *falls* on 8 of 18 rows. Their step at 17.4k,
where their slope is still +19, also moves away from bang-bang. So the actor does not climb
the return along this direction on either side, and that is not the divergence.

What the decomposition does say. **The loss agrees with the return**: the full actor loss's
derivative along the bang-bang direction is negative (wants bang-bang) on all 18 rows,
carried entirely by the CRSPP cosine term (−13 … −30 ×10⁻³); reward, information gain,
critic bootstrap and baseline are each within ±0.3; the entropy term is 0 along a direction
that keeps the scale; no ablation of the normalisation, weights, baseline or bootstrap flips
its sign. **The return's own gradient is not "toward bang-bang"**: split by whether the
actor's mean already agrees in sign with v̂, the return-only step moves the disagreeing
third of states toward the rule and *softens* the agreeing two thirds (14 and 15 of 18
rows) — bang-bang is one way up the return, not the steepest. **The self-imitation term
dominates the parameter gradient**: its norm is 0.23–0.57 against the return's 0.012–0.17,
3–30× larger; the plain-gradient step takes its sign on 16 of 18 rows; switching it off
flips the projected step on 9 of 18 rows; and it is the only term that ever opposes the
return along this direction (noisy per batch, positive on 7 of 18). The gradient path
itself sees the slope (pathwise dR/dε = +42, +35 against the finite difference's +31, +25).

The reading that remains strictly justified: the actor's update on our side is, in
parameter space, mostly self-imitation of its own recorded successes plus Adam momentum,
with the return a minority term; the return's minority pull softens saturated actions more
often than it saturates them. Whether self-imitation is what carries the drift is one
counterfactual away — the recipe from the 20.5k stage and from the 16.4k four-seed run with
`bc_scale = 0` (running as this section is written) — and whether its weight against the
return is the same on their side is the measurement after that. §14's all-frozen replay,
where self-imitation alone with frozen models held the single pump, bounds what it can be:
not sufficient on its own.

## 24. Self-imitation off, both sides: the return alone destroys the pump on both (2026-09-30)

§23 left one counterfactual: the recipe with the self-imitation term's scale at zero.

**Ours.** From seed 0's 20.5k stage (`runs/cf_nobc_key0`): 14 of 16 successes in the first bin,
then none — 0 of 8–12 in every bin from 21.5k to 28.7k — while the actor's act-time mean
saturates one-sided (+0.72 at 21.5k, +0.96 by 26k): §5's fixed point, the trap, reached
from a healthy policy in under 1k steps once the imitation term is gone. From the 16.4k
four-seed run (`runs/cf_nobc_off4`, to 22.5k): every seed's double-pump share *rises* first
(seed 0: 5 → 18 → 35 → 67%; seed 3: 33 → 62%; seed 2 last, 39 → 100% at 21.5k) and then the
seed stops succeeding altogether (seeds 0, 1 and 3 by 19.5–20.5k, seed 2 after 21.5k).

**Theirs.** Their seed 0 resumed from its 20.4k snapshot (`raif-paper/logs/probe_nobc_seed0`;
the log directory a copy of the snapshot so their trainer's own reload resumes step, agent
and replay — `--run.from_checkpoint` restarts the counter and is not a resume) with
`loss_scales.actor_refresh = 0`, beside a control resume with it at its shipped 0.5:

| their resume, 20.4k → 24.4k | 20–21k | 21–22k | 22–23k | 23–24k |
|---|---|---|---|---|
| control, imitation on | 16/16 succ, median 37 | 16/18, 37 | 20/21, 39 | 17/19, 39 |
| imitation off | 16/17 succ, median 37 | **1/10, that one an 85-step double pump** | 0/10 | 0/10 |

The same collapse, on the same timescale, through the same door: one double pump on the
way down. So on both sides the actor's return gradient, alone, drives the policy off the
single pump and into failure, and the self-imitation term — the same weight (0.5), the same
mask rule (training ledger rows 6.38, 6.41, 8.1, transcription-tested), a loss value of the
same small size on both sides (ours −0.003 … −0.02, theirs ±0.01) — is what holds the pump
on both. The dependence is matched. What the double pump *is* is now clearer: the partial
expression, under imitation's restraint, of the same pull that unrestrained sends the actor
through a double pump into the fixed point — on their side too, the one time it was let go.

What still differs is that under restraint ours gives way to that pull for 50k–250k steps
of every seed's training and theirs does not. The balance of the two terms in the update
is the last place left: not their weights or values, which match, but their *gradients* on
each side's own data — §23 measured ours (imitation 3–30× the return in norm) and not
theirs. A one-step actor update on their side with the same decomposition (`measure_update`
has the step; the split by term is what is missing) is the direct read; the cross-data
experiment (§18) is the principled one, since the gradients are of the same losses on
different data. Neither is run here.

## 25. Their actor gradient by term, and the their-side reads re-run on the right data (2026-09-30)

§24's last read: the actor gradient on their side split by term, on real batches drawn as
their loop draws them, at their snapshots (`evidence/theirs-grad.md`; the
gradients taken as their optimiser takes them, unscaled, actor parameters only; the four
term gradients sum to the gradient of their own loss to 1e-4 in float16 and 1e-7 in float32).

| their snapshot | ‖g_return‖ | ‖g_imitation‖ | ratio | flagged share | return's dL/dε toward bang-bang | imitation's |
|---|---|---|---|---|---|---|
| fp16 17.4k | 0.068 | 0.37 | 5.2 ± 3.7 | 0.08 | −11 (wants bang-bang) | + |
| fp16 19.4k | 0.081 | 0.22 | 2.9 ± 1.9 | 0.08 | ≈ 0 | + |
| fp16 21.4k | 0.087 | 0.28 | 3.3 ± 3.6 | 0.07 | +7 … +15 (against) | + |
| fp16 23.4k | 0.066 | 0.15 | 2.4 ± 1.3 | 0.06 | against | + |
| fp32 17.6k / 19.6k / 21.6k | 0.056 / 0.108 / 0.054 | 0.27 / 0.22 / 0.26 | 4.7 / 2.3 / 5.8 | 0.08 / 0.08 / 0.07 | −14 / ≈ 0 / against | + |

(8 batches each; the entropy term's gradient is ~0.002 in norm and the world-model entropy
term's ~5e-5, on both sides.)

**Imitation's weight is matched.** Their imitation gradient is 2–6× the return's in norm
against our 3–30× (median ~6), with the same flagged share (0.06–0.08 against our
0.06–0.14); their plain full step takes imitation's sign on 5–8 of 8 batches as ours did on
16 of 18 rows; their return-only step softens |m| and repairs disagreeing states exactly as
ours does (§23). So the balance of the two terms in the update is the same on both sides,
and §24's last candidate is closed.

**What differs is one sign.** Along the direction from the actor's plan toward bang-bang,
their return term's own derivative is negative (wants bang-bang) at 17.4k and *positive*
(against it) from about 21k — §22's slope reversal, seen now from inside the loss — while
ours is negative on all eighteen rows at every stage (§23); and their imitation term opposes
bang-bang on average at every snapshot, where ours does so on 7 of 18 rows. Both terms on
their side agree, after 21k, that the actor should stay where it is; on ours the return
keeps asking for the more aggressive pump and the actor, held by imitation, drifts the other
way. Why the return's sign reverses on their side and never on ours is a statement about
what the two models predict for a more saturated action after 21k, not about the update.

**A caveat found and discharged.** Their snapshot directories link to the parent run's
replay, which kept filling after each snapshot; the earlier their-side reads (§15, §17, §22,
§23) drew their held start states and batches without a time filter, so the states came
from the end of the run at every stage. Re-run with chunks filtered to those saved before
each checkpoint (`evidence/theirs-time-filtered.md`):
every conclusion stands. The CRSPP margin still decays from ~+0.04 to zero or below by
20.4–21.6k and stays negative after, less monotone (a bump at 19.4k, −0.020 at 20.4k, +0.003
at 21.4k); their actor's sharpening holds (|a| 0.88 → 0.92, sign agreement 0.61 → 0.74);
the slope's sign change survives, with the fp32 19.6k read now near zero at small ε, so
"negative from ~19.6k" reads "from about 20–21k". And a claim made in passing — that their
loop alternates general and positive-replay batches — was checked against their metrics
and withdrawn: `positive_replay/samples` is 0 at every logged step of both runs; the
positive dataset is never sampled in their training, as the register's divergence 18 says.

**Where the arc stands.** Every component of the actor's update is now measured on both
sides and matched: the loss form, the gradient path, the imitation term's weight and mask,
the entropy terms, the momentum. The one measured difference between the two actors'
signals is the sign of the return's derivative along the bang-bang direction after ~21k,
which is the models' prediction, not the actor's arithmetic. The next read, if the arc
continues, is which leaf of their return carries that reversal — the CRSPP cosine or the
reward — against ours, where the cosine carries a preference that never reverses (§23);
and after that the cross-data experiment, which says whether the reversal follows the
model code or the data the model saw.

## 26. Their reversal, leaf by leaf: the reward through the continuation head first, then the cosine (2026-09-30)

§25's question — which leaf of their return carries the sign change at ~21k — read at their
snapshots with the return split exactly into its parts (`evidence/slope-leaf.md`:
the λ-return recursion re-run on the recorded per-step terms, parts summing to the total to
1e-9; 512 time-filtered starts, 8 keys, a second key set as the floor). Slope of each part at
ε = 0.05 toward bang-bang, ×10⁻³ per unit ε, return sign (+ wants bang-bang):

| their snapshot | total | CRSPP cosine | reward | info gain, critic |
|---|---|---|---|---|
| 15.4k | +29 | +29 | −0.1 | ±0.4 |
| 17.4k | +21 | +21 | +0.4 | ±0.4 |
| 19.4k | +18 | +25 | −9 | |
| 20.4k | −18 | −0 | −20 | |
| 21.4k | −6 | +20 | −29 | |
| 22.4k | −28 | −1 | −29 | |
| 23.4k | −21 | −18 | −5 | |
| 28.7k | −25 | −3 | −23 | |
| fp32 17.6k / 19.6k / 21.6k / 23.6k | +29 / −5 / −45 / −18 | +26 / +25 / −35 / +10 | +3 / −34 / −11 / −29 | ±2 |

(The ε = 0.05 reads after 19k are noisy — paired standard errors ±10–35, the two key sets
disagreeing in size on two rows but not in sign; the ε = 0.2 slopes agree with half the
noise. No leaf's *level* crosses zero: the reward leaf's level falls from −3.9 to −9.2 as
the head's constant per-step prediction drifts from −0.58 to −1.17, which has nothing to
do with the action.)

**Two stages, neither of which ours has.** At 15.4–17.6k their slope is the cosine leaf
alone, as ours is at every stage (§23). Then, from 19.4–21.4k, the *reward* leaf turns
against bang-bang while the cosine still wants it — and every unit of that runs through
the continuation head: hold the reward predictions at their ε = 0 values and let only
`cont` move, and the reward slope reproduces exactly (−23.2 of −23.3 at 21.4k). It sits on
the 5–15% of starts whose imagined rollout is predicted to terminate — a share that grows
from 2.5% at 17.4k to 13–20% after 21k — where moving toward bang-bang slightly delays the
imagined termination and more steps of the constant negative reward accrue. Then, from
about 21.6k (fp32) / 23.4k (fp16), the cosine leaf's own preference fades and turns
negative: separating its direct cosine change from its own continuation channel, the
direct part goes +27 (17.4k), +9 (21.4k), −26 (23.4k); +9 (fp32 19.6k), −50 (fp32 21.6k). A
more saturated imagined trajectory scores *lower* similarity to their preference rollout
after that point — the leaf stops being a target that bang-bang approaches, which is
§15's collapse of the CRSPP margin seen from the actor's side.

Against ours: our cosine leaf keeps preferring bang-bang at every stage (the preference
rollout is a fixed point bang-bang keeps approaching, §5/§13), and our reward leaf never
discriminates along this direction (±0.3, §23) although our continuation head predicts
termination on a similar share of starts by 20.5k (§17: 15% of seed 0's starts) — on ours
the imagined termination is not moved by the action the way theirs is. So the two models
differ in what they predict for a more saturated action: theirs, after 21k, that it delays
the goal and that it strays from the preference rollout; ours, neither. Whether those
predictions follow the model code or the data the models saw is the cross-data experiment,
launched from here.

## 27. Cross-data: the reversal follows the code, and the object is the cosine leaf's direct response (2026-09-30, night)

The 2×2 (`evidence/crossdata.md`): each side's model code, from its ~20.5k stage's parameters
and optimiser state, trained on its own replay windows and on the other side's — no new
environment interaction, a field-rename bridge whose conventions were checked on both
sources (the action drawn at `obs_t`, the reward of `a_{t−1}`, the terminal row followed by
an `is_first` row; regressing the velocity change gives 0.0029 on `a_t` against −0.0001 on
`a_{t−1}` for both) — for 0, 256 (two replicates) and 1024 updates, read each time with the
stage actor held: the return's slope toward bang-bang, split into the cosine leaf (and its
direct part, net of its own continuation channel) and the reward-through-continuation
channel, plus the share of starts whose imagined rollout terminates. ×10⁻³ per unit ε at
ε = 0.05 / 0.2; + wants bang-bang.

| cell | updates | total | cosine, direct | reward via cont | terminating share |
|---|---|---|---|---|---|
| our code, our data | 0 / 256 / 1024 | +11 / +14 / +10 | +31 / +25 / +22 | −39 / −16 / −6 | .15 / .17 / .17 |
| our code, their data | 0 / 256 / 1024 | +19 / +11 / +14 | +30 / +23 / +22 | −21 / −18 / −2 | .10 / .10 / .12 |
| their code, their data | 0 / 256 / 1024 | −16 / −32 / −42 | −9 / −7 / −16 | −16 / −42 / −20 | .07 / .06 / .09 |
| their code, our data | 0 / 256 / 1024 | −21 / −31 / −48 | +2 / −14 / −22 | −49 / −31 / −12 | .09 / .14 / .15 |

(Replicates agree to ±0–15; the ε = 0.2 totals' standard error is ±4–12. The reward loss
falls identically in all four cells, 0.75 → 0.54 → 0.29, so both models learned the foreign
data equally well; past 256 updates this is overfitting ~22k frozen frames.)

**The sign follows the code.** Our code keeps the return's slope positive on both data
sources and their code keeps it negative on both, at zero updates already and through 1024;
neither side's data ever flips the other's sign, and each code keeps its own drift while
training on the other's data: theirs pushes the cosine leaf's direct part negative on *our*
data (+2 → −22), ours keeps it positive on *theirs* (+30 → +22). **The termination level
follows the data** — both codes predict more terminating rollouts on our states than on
theirs — and its sensitivity to a more saturated action is the same small amount in every
cell, so the continuation channel is not what separates the sides at this stage; §26's
attribution of their first stage to it was a reading of their timeline, and at 20.5k our
return carries the same channel at the same size (−39 / −50 at zero updates). What
separates the sides is one object: **the CRSPP cosine leaf's direct response to a more
saturated imagined trajectory** — ours scores it as *more* similar to the preference
rollout, at every stage and on either data (+22 … +39); theirs as no more similar or less
(−27 … +2), and less with training.

One caveat bounds the claim: every cell starts from its side's stage parameters and
optimiser state, so "code" here means the model code *with* its history — 256–1024 updates
on the other side's data neither flip a sign nor stop a drift, but a from-scratch swap was
not run. Within that bound the answer to §25's question is: the reversal is not learned
from the data; it is what their preference model does and ours does not.

**Where that leaves the arc.** The transient is now traced to the CRSPP's cosine leaf — the
same object §13 watched let go, §14 found sufficient on its own, and §5 found to be a fixed
point on our side. What remains is *why* their preference model's imagined rollout stops
rewarding saturation after 21k and ours never does: the preference trajectory itself
(decoded, on both sides, at 20.5k and 24.5k — ours §5's wall-side fixed point; theirs not
yet read), the feature space the cosine is taken over, and the CRSPP's own training (the
contrast loss and the positive mask; register rows 5 and 6.5). That is a read of two
preference models, not another counterfactual.

## 28. The two preference models read: ours a start-blind fixed point, theirs a start-conditioned rollout; the sign is the CRSPP code, not its history (2026-09-30, night)

The read §27 asked for (`evidence/prefmodel.md`): both sides at their ~20.5k and ~24.5k
stages (ours `causal32k_key0_s20480` / `causal1k_key0_s24576`; theirs
`probe28k_state_mc_seed0_s20400` / `_s24400`), on the 512 held starts of each data source
from §27 read by both codes through the bridge. Three parts: **(A)** the preference rollout
decoded through each side's agent decoder, and the feature space the cosine is taken over;
**(B)** what each CRSPP trains on — the label composition, the rule on expert episodes, the
positive mask; **(C)** the from-scratch swap §27 left open — each side's CRSPP trained from
its own init on identical labelled data with the world model, actor, critic and ensemble
held at the stage, and the cosine leaf's direct slope read at 0 / 64 / 256 / 512 / 1024
updates with the stage actor. Slopes ×10⁻³ per unit ε at ε = 0.05 / 0.2; + wants bang-bang.

**(A) The preference rollouts.** Decoded position at the rollout's eighth step, over the
512 held starts (own / other side's states where they differ):

| | ours 20.5k | ours 24.5k | theirs 20.4k | theirs 24.4k |
|---|---|---|---|---|
| mean | −0.33 | −0.30 | −0.23 / −0.25 | +0.08 / +0.02 |
| sd across starts | **0.002** | **0.001** | 0.21 | 0.55–0.60 |
| corr(t=8, start x) | ≈0 | ≈0 | 0.37–0.46 | 0.43–0.52 |
| share on the goal side (x > 0) | 0 | 0 | 0.17–0.22 | 0.45–0.46 |
| share at the wall (x < −0.9) | 0 | 0 | 0 | 0 |

**Ours is a fixed point.** It does not depend on the start or on which side's data the
starts come from; even the CRSPP posterior at the start decodes with an sd of 0.044 against
a start sd of 0.39, and the rollout holds a small positive velocity (+0.012) at the same
place every time. A *sampled* rollout of our CRSPP, their recipe, is just as start-blind
(−0.316, sd 0.015), so the fixed point is in our CRSPP's weights and not in the mean-field
rollout (register row 6.5). **Theirs is a start-conditioned rollout that drifts toward the
goal**, more so at 24.4k than at 20.4k (endpoint sd 0.2 → 0.6; goal side 20% → 46%).

**The feature space.** Both sides compare `[deter (128), flat stoch (1024)]`. On ours the
preference rollout's stoch part is essentially uniform (norm 1.01), so it adds a constant
of about 0.015; on theirs two sampled one-hots agree in about 1 group of 32, about 0.013. So
on both sides at least 97% of the cosine, **and all of its response to ε**, lives in the
128-d deter: the stoch change is 0.0 ± 0.2 at every step. The cosine level at ε = 0 is
0.62–0.66 ours and 0.45–0.53 theirs. The per-step direct change grows with depth on both
sides — ours +0.7 → +19.4 at 20.5k (to +14 at 24.5k), theirs −0.7 … −2.4 at 20.4k falling
to −15.3 at 24.4k — the deter of the saturated trajectory moving toward (ours) or away from
(theirs) the preferred deter.

**(B) What the CRSPP trains on.** The two sides are the same to the figure. Both replays
hold 48 expert episodes at the start (1,911 rows ours, 1,897 theirs); of the positive rows
47.7% come from expert episodes on both sides, and so does 81.4% of the positive label mass;
rows with ρ > 0 are .176 ours and .173 theirs; a 16×64 batch holds about 180 positive rows,
about 90 of them expert; the positive KL mask covers .17–.18 of rows. The label rule checks
out on the data and is identical: expert rows carry ρ = +1 throughout except 0 on the first
row; own successes follow the 0.4 lookback (1, .4, .16, …, zero within eight steps of the
end); failures carry −0.3^k, so only about 2% of rows fall below −10⁻³. The batches as drawn
match the exported training data exactly. **No asymmetry in how the gold rows are labelled
or weighted** — the hypothesis that we mishandle the expert trajectories is negative.

One fact is new on both sides: **the gated KL sits at its free-nats floor throughout** —
kl / (0.6 · positive share) = 1.000–1.002 at init, at stage and after 1024 updates — so the
masked KL carries essentially no gradient and the contrast term is the CRSPP's only live
training signal.

**(C) The from-scratch swap.** Only the CRSPP trained, 1024 updates on the frozen labelled
data; ours from `R.build` (seed 0), theirs from the `_2` layers of their step-0 checkpoint.

| code / data | N = 0 | 64 | 256 | 512 | 1024 | control: from the stage CRSPP, 0 → 1024 |
|---|---|---|---|---|---|---|
| our code, their data | −1 / −1 | +8 / +9 | +27 / +30 | +28 / +31 | **+27 / +29** | +33/+36 → +29/+32 |
| our code, our data | −3 / −3 | +13 / +16 | +28 / +35 | +30 / +37 | **+29 / +36** | +31/+38 → +30/+37 |
| their code, their data | −3 / −5 | +1 / −1 | +6 / +4 | −9 / −14 | **−16 / −23** | −8/−13 → −20/−29 |
| their code, our data | −2 / −4 | +6 / +6 | +18 / +22 | −1 / −3 | **−27 / −38** | +2/+1 → −26/−36 |

(A second key set agrees in sign at every N ≥ 256; their gradients stayed finite under a
fixed 1e4 fp16 loss scale.) Both CRSPPs start near zero. **Ours converges positive on
either data, theirs negative on either, after a positive phase at 64–256 updates**, and the
from-stage controls end in the same place. From init ours re-forms the same fixed point on
either data (endpoint −0.25 to −0.27, sd 0.003 by 256 updates); theirs re-forms a
start-conditioned, goal-ward rollout (endpoint sd 0.5–0.7). So **§27's caveat closes: the
sign follows the CRSPP code without its history.**

**Reading.** The discriminating object is now narrowed to what the two CRSPP codes learn
from the same contrast signal on the same labelled data: ours a start-blind fixed point
whose deter the saturated trajectory approaches, theirs a start-conditioned rollout whose
deter it leaves. The code differences between the two CRSPPs that could carry this, with
their standing:

- **Where the recognition filter starts** — `execution-ledger-training.md` row 4.1,
  recorded and deliberate: theirs seeds `observe_no_action` from the world model's carried
  latent (`agent.py:197`, `prev_latent`) and resets at `is_first` to the **world model's**
  learned `initial` through the world model's prior head (`nets.py:156-157`, `:57-59`), so
  its CRSPP has no start of its own and the contrast gradient reaches the world model's
  prior head through it; ours starts every window from the CRSPP's own learned `initial`
  and `prior.mode` (`crspp.py` `observe_no_action`) and takes no `is_first`. This is the
  one structural difference that bears directly on the property (A) found separating the
  two rollouts, their start-dependence. Not tested here.
- **The encoder** — their C loss reaches the shared encoder; it was frozen in (C) and theirs
  still went negative, so this is not required for the sign.
- **The mean-field rollout** (register row 6.5) — ruled out for the fixed point by the
  sampled-rollout check in (A).

Caveats: one seed and one stage per side; the held starts come from the same frozen source
as the training data; the direct slope is §27's λ-return split while (A)'s per-step Δcos is
raw and undiscounted; decoding CRSPP latents through the agent decoder assumes a shared
latent geometry that holds only partly (deter cosine between the CRSPP and agent posteriors
0.88 ours, 0.77–0.87 theirs).

**Where that leaves the arc.** The transient's mechanism is traced to one learned object
and one candidate seam for it: row 4.1's start, adjudicated as a deliberate divergence
before any of this was known. The test that would settle it is a counterfactual on the
CRSPP's start, not on a learner: in (C)'s harness, train our CRSPP from scratch with its
filter seeded from the agent's posterior at the window start (their convention) and read
(A)'s start-dependence and (C)'s slope. The probe itself is a scratch script (the seeding
is a monkeypatch of `observe_no_action`, as (C) already wraps it); what it would bear on is
an adjudicated row, so whether to run it is Alex's call.

**Addendum (2026-09-30, late): the start seed is not it.** The probe above was run
(`evidence/prefmodel-seed.md`): our CRSPP trained from scratch on our data at the
20.5k stage with its recognition filter seeded, as data, from (V1) the agent's posterior at
the window's first row and (V2) the world model's learned initial through the world model's
prior head — their `is_first` semantics, and since every window in our data starts at
`is_first`, the faithful analogue — against (V0) its own initial through the same plumbing,
which reproduced the unpatched control (N = 256: +28.0 / +35.0 against +28.1 / +35.2). All
three converge to the same positive slope (+28 … +37 at N = 256 and 1024) and the same
start-blind fixed point at N = 256 (endpoint sd 0.003); by N = 1024 all three, V0 included,
pick up a small start correlation (sd 0.05–0.08, still at the wall side, goal share 0),
which is more training, not the seed. So where the filter starts does not decide what our
CRSPP learns from the contrast signal. What the from-scratch swap's "code" still contains,
with the seed removed: the CRSPP module itself (its GRU and heads; the 480 biases of
entry 065), its loss and optimizer step, and the rollout-and-readout recipe (register row
6.5). The bisection between those is a weight transplant: their trained `_2` layers at 20.4k
into our module, read with our rollout, and ours into theirs — forward equivalence checked
first, as the actor transplant of entry 065 was.

**Addendum (2026-09-30, late): register row 6.5 is excluded at run level.** The sampled
preference rollout already has a from-scratch run: `runs/crspp_sampled_500k` (entry 079's
switch, `sample_scoring=True`, launched 2026-09-22 at 540026e, four seeds, before act-time
sampling, so `act_sampling=False` like its control `noactsample_500k`). Its double-pump
phase (share of successful episodes longer than 50 steps per 25k bin, the rule used
throughout) matches the mean-field control seed for seed:

| seed | sampled preference rollout | expected (control) |
|---|---|---|
| 0 | 50k–225k, peak 99% at 50k | 50k–225k, peak 97% at 50k |
| 1 | 50k–175k, peak 97% at 100k; a second bout 350k–450k | 50k–175k, peak 95% at 100k |
| 2 | 25k–175k, peak 100% | 25k–175k, peak 100% |
| 3 | 225k–350k, peak 55% | 200k–300k, peak 50% |

So whether the preference trajectory is sampled or expected does not touch the transient.
The "recipe" arm of the transplant bisection is thereby narrowed: if their weights read
positive in our module, the difference lies in the readout around the rollout (the start
belief handed to the row, the feature vector, the horizon), not in row 6.5.

## 29. The sign is set by one thing: whether the CRSPP's training scan resets at an episode boundary inside a window (2026-10-01)

The transplant bisection §28 called for (`evidence/prefmodel-transplant.md`) was
built and could not be read against its own key, and the reason led to the answer.

**The transplant.** Their trained CRSPP at 20.4k mapped layer by layer into our module at
20.5k and ours into theirs; 172,992 numbers each way, the count audit §1.2 reconciled (our
20.5k CRSPP carries none of entry 065's extra biases: the `winit` fix removed them). Forward
equivalence held on the same embedding sequences: deters to 6.4e-7 and posterior logits to
1.2e-6 in fp32, codes agreeing 100% (fp16 on their side: ≤ 4e-3, 98.8–99.6%). The
transplanted slopes then read near zero in both directions (+2.0 / +2.1 and +3.7 / +4.1
against native +30.6 / +38.2 and −7.7 / −12.5) and mean nothing: a CRSPP's latent is only
meaningful in the latent space it was trained against, and in a foreign host its posterior
deter has a cosine of 0.09 with the host agent's posterior against 0.88 natively, the cosine
level falling eightfold. So weights cannot be moved between the sides; the bisection had to
be run the other way round — **each trainer run inside the other side's latent space**, on
frozen data, and read there.

**The recipe, once more.** Swapping the preference rollout between sampled and expected on
each side's native stage moves neither sign (ours +24.9 / +31.3 sampled, theirs −7.3 /
−12.5 expected), agreeing with §28's run-level exclusion of row 6.5.

**What decides the sign.** Cosine-direct slope at N = 1024, ×10⁻³ per unit ε at
ε = .05 / .2, read on the host's 512 held starts with the host's stage actor:

| trainer → host latent space | resets the CRSPP scan at mid-window `is_first` | N = 1024 |
|---|---|---|
| ours → ours | no (our code as written) | +30.0 / +36.8 |
| ours → ours | **yes** | **−22.5 / −29.2** |
| theirs → ours | yes | −20.6 / −26.7 |
| ours → theirs | yes | −36.7 / −50.2 |
| ours → theirs | no | +20.0 / +21.6 |
| theirs → theirs (§28 C) | yes (their code as written) | −16 / −23 |
| theirs → theirs, **one line changed**: `is_first` zeroed in the CRSPP scan | no | **+19.7 / +21.0** |

The convention used in *training* sets the sign; the convention at read time does not
(trained with resets and read without: −13 / −18; trained without and read with: +29 / +36).
The trainer's loss, optimiser and precision, the init draw, the latent space, the actor and
the rollout recipe all leave it where it is. With resets the preference rollout is
start-conditioned and drifts toward the goal (t8 sd ≈ 0.47, goal share ≈ 0.46 — §28's
description of theirs); without them it is §28's start-blind fixed point. Sanity: the ours →
theirs run with resets equals §28's start read at N = 0; losses track the native trainers
within about 3%; the two with-resets runs in our space agree within 2 at every read.

**The divergence.** Their `obs_step_no_action` (`nets.py:151-157`) masks the carried state
at `is_first` and adds the initial back, like their main `obs_step`. Our
`crspp.observe_no_action(parts, key, embeds)` and `fit_crspp` take no `is_first` at all, so
our CRSPP's recognition scan carries its state across every episode boundary that falls
inside a 64-step window, while our main world model's `observe` restarts there
(`agent.py:443`). The training ledger's §4 has no row for the mask: 4.2 and 4.4 cover the
step's layers, 4.1 the window-start initial, and the boundary inside the window was never
listed. It is a replication defect, unrecorded until now. Rows at a mid-window boundary
are a small fraction of the batch (the §28 seed probe estimated 1.7%); why so few rows
decide what the whole model learns from the contrast loss is not probed here.

Caveats: one seed and one stage per side; the no-reset variant of their trainer was run on
their data only; the our-code runs with resets returned to the world model's initial (their
convention); the reset to the CRSPP's own initial, the form a fix on our side takes, was
then read and gives the same result — N = 1024 −22.1 / −28.2 against −22.5 / −29.2 for the
world-model initial and +30.0 / +36.8 for no reset, with the same start-conditioned,
goal-ward rollout (t8 sd 0.45, goal share 0.49). The reset itself flips the sign, not where
it resets to. Measured on the data: 17.1 mid-window `is_first` rows per 1024-row batch on
ours (sd 2.6), 1.67%; 1.68% on theirs.

**Where that leaves the arc.** The transient's mechanism now has a single, testable cause:
our CRSPP learns a start-blind fixed point because its filter is never restarted at an
episode boundary inside a window, and the fixed point is what rewards saturation. The fix
is one argument threaded through `observe_no_action`, `crspp_loss`/`fit_crspp` and
`crspp_recognize` and their two call sites, with the scan masking the carry at `is_first`
as the world model's `observe` does. Confirmation is the 500k phase read of §20 on a run
with the fix.

**Addendum (2026-10-01): the fix in the real training loop.** Seed 0's 16k stage resumed
to 28k with the restart in place (`runs/fix1k_key0`, 1k stages kept), against the unfixed
series `causal1k_key0` from the same stage. Three reads.

*The cosine leaf's direct slope* (`evidence/prefmodel-transplant.md` addendum 3; each
checkpoint's own actor, 512 held starts, ×10⁻³ per unit ε at ε .05 / .2):

| step | fixed | its t8 sd / corr | unfixed | its t8 sd / corr |
|---|---|---|---|---|
| 20,480 | +28.2 / +35.1 | .022 / +.23 | +30.3 / +38.0 | .002 / −.03 |
| 24,576 | +12.7 / +16.6 | .046 / +.27 | +20.5 / +24.9 | .002 / +.21 |
| 28,672 | **+0.1 / +1.0** | .142 / +.39 | +15.7 / +19.9 | .002 / +.66 |

*The total return's slope* (the §22 slope probe, 8 keys, two key sets; `evidence/slope-fix1k.md`): fixed +10 / +13 →
+2 / +4 → **−44 / −52** at the same three steps; unfixed +6 / +15 → +7 / −10 → −16 / −8.

*How fast reset training undoes a learned fixed point* (addendum 2; the trained 20.5k and
16k CRSPPs trained on in the harness with restarts, frozen data): +30 → +26 at 256 updates
→ −2 at 1,024 → −29 / −38 at 4,096 (from 16k: +42 → +37 → +8 → −27 / −38), the rollout
start-conditioned by 1,024.

So in the real loop the restart pulls the leaf's response from +28 to zero over the 12k
steps after 16k while the unfixed leaf stays at +16 … +30, and the preference rollout
acquires start-dependence (sd .002 → .142); the harness says the crossing to negative takes
about a thousand CRSPP updates, which is where the resumed run is at 28k. Behaviour over
20–28k on this one seed is unchanged (double-pump share 63% of 82 successes against 53% of
106; success rate 0.58–0.62 against 0.69–0.87 in the last three bins), as expected while
the leaf is still non-negative. The confirmation that remains is from scratch:
`runs/fix32k_key0` (seed 0 to 28k in 4k stages with the restart from step 0), then the
500k phase read.

**Addendum (2026-10-01): from scratch with the restart — the reversal at 21k, and no onset
through 28k.** Seed 0 trained from step 0 with the restart in place (`runs/fix32k_key0`, 4k
stages kept), against the unfixed from-scratch series `causal32k_key0` (same seed, keys and
fixtures). The total return's slope toward bang-bang (the §22 slope probe, two key sets):

| step | fixed from scratch | unfixed |
|---|---|---|
| 20,480 | **−16 / −13** (share up .46) | +19 / +18 (§22; share up .90) |
| 24,576 | **−22 / −29** | +7 / −10 (.76) |
| 28,672 | **−71 / −78** | −16 / −8 (.47) |

The fixed return already scores a more saturated action as worse at 20.5k — their reversal
at ~21k (§22), which no unfixed seed of ours showed — and goes on falling. The cosine leaf's
direct response, read as in addendum 3 (`evidence/prefmodel-transplant.md` addendum 4), is what moves
it: +3 / +8 at 20.5k, −7 / −10 at 24.5k, **−25 / −34** at 28.7k, against the unfixed +30,
+21, +16, and the fixed run's preference rollout is start-conditioned and goal-ward from
20.5k on (t8 sd .15 → .43, goal share 0 → .51) where the unfixed run's stays the fixed point
(sd .002, goal share 0) — §28's description of theirs, now on our side. Behaviour over
the onset window, per 2k bin of successful episodes (median length; share longer than 50):

| bin | fixed | unfixed |
|---|---|---|
| 18k | 41; 22% | 41; 31% |
| 20k | 40; 19% | 42; 31% |
| 22k | 45; 45% | **75; 73%** |
| 24k | 42; 28% | 46; 42% |
| 26k | 40; 27% | **73; 89%** |

Through 20k the two series are alike; from 22k the unfixed run's onset shows (the median
successful episode at 73–75 steps in two of three bins) and the fixed run's does not
(40–45 throughout). Success rate over the last three bins: .91 / .83 / .86 fixed, .79 / .91
/ .69 unfixed. One seed and the onset window only; the phase read of §20 (50k–250k on 6/8
seeds) is the remaining confirmation.

**Addendum (2026-10-01, later): four seeds to 60k — the restart does not remove the phase.**
`runs/fix_500k` (four seeds, dd47e4a, otherwise `actsample_500k`'s configuration). Share of
successful episodes longer than 50 steps / success rate, per 12.5k bin to 62.5k:

| seed | fixed | unfixed (`actsample_500k`) |
|---|---|---|
| 0 | 27 31 29 50 **61**% / SR .63 .93 .74 .64 .65 | 38 27 72 67 **75**% / .57 .97 .86 .90 .99 |
| 1 | 29 23 52 67 **82**% / .46 .62 .65 .64 .74 | 22 48 71 78 **70**% / .54 .78 .83 .96 1.00 |
| 2 | 29 28 44 74 **86**% / .53 .94 .75 .63 .67 | 27 30 49 81 **92**% / .51 .95 .88 .78 .86 |
| 3 | 21 15 11 14 13% / .55 .98 .98 .98 .99 | 22 17 18 10 9% / .65 .98 .99 .99 1.00 |

The return's slope toward bang-bang at ~60k (the §22 slope probe, ε .05, two key sets): fixed
seeds 0–2 **−66 / −66, −45 / −46, −43 / −47**, seed 3 +26 / +24; unfixed seeds 0–2 −14 / −21,
−8 / −5, −10 / −7, seed 3 −34 / −16.

So with the restart the return's preference against saturation is in place on the pumping
seeds and stronger than without it, and seeds 0–2 enter the double pump to the same extent
anyway, with a lower success rate from 25k on. The sign of this slope does not predict the
50k–250k phase: the §27–§29 chain (the restart sets the cosine leaf's sign, which sets the
return's slope at ~21k) holds and is fixed, but it is not what carries the long transient.
The seed-0 result to 28k (no onset through 28k) was the onset delayed, not removed. The
restart stays as a replication fix (training ledger 4.15); the phase's driver is open again.

Against the oracle over the same bins (`raif-paper/logs/oracle_state_mc_seed{0..3}`, training
episodes only; an episode succeeds when it ends before the 100-step limit — their
`episode/success_rate` is a running average over recent episodes, not a per-episode flag),
double-pump share of successes / success rate per 12.5k bin to 62.5k:

| seed | oracle |
|---|---|
| 0 | 25 8 8 6 3% / SR .63 .97 .96 .99 1.00 |
| 1 | 15 12 10 14 8% / .62 .96 .97 .98 .98 |
| 2 | 21 13 8 13 6% / .66 .96 .99 .98 1.00 |
| 3 | 14 20 7 9 6% / .60 .93 .94 .99 .99 |

From 12.5k on every oracle seed succeeds on .93–1.00 of episodes, with a double-pump share
of 6–20%; only our seed 3 looks like that, in both of our runs. *(Corrected 2026-10-02: this
table first read their running-average `success_rate` above 0.5 as a success, which counted
every episode as one and their timeouts as long successes; the conclusions of this paragraph
survive the correction.)* The restart's lower
success rate on seeds 0–2 (.62–.75 against .78–1.00 unfixed) is therefore a move *away*
from the oracle, not toward it: every failure in both of our runs is a 101-step timeout, so
the fixed run fails to reach the goal at all more often. With the restart matched, a
second difference remains that the restart interacts with.

## 30. The phase's driver: the act-time episode start read the prior's probabilities, not its mode (2026-10-01)

The 65k trace (`evidence/trace65k.md`): ours `runs/fix_500k/snapshots_step68608`
(seeds 0–3; also `_step59392`) and the unfixed `runs/actsample_500k/snapshots_step70656`,
against theirs `raif-paper/logs/probe64k_state_mc_seed{0,1}` (loaded at step 65,600; success
.98–.99, double-pump share .04–.08 over the same tail).

**What our failures are.** On seeds 0–2 at 68.6k, 22 of about 62 episodes per seed in the
last 5k steps time out, all alike: the actor pushes right from the reset and keeps pushing
right while the car rolls left for 30–33 steps, reaches no further than x ≈ −0.2 … +0.1, and
makes its first left push at step 78–86, with peak energy a tenth to a quarter of a
single-pump success's. That is a committed push against the velocity, not over-pumping or
dithering. The double pumps begin the same way and recover; single pumps, on both sides,
begin by pushing left.

**The actor.** Mean |a|, σ and entropy match theirs. What differs is its coupling to the
velocity, and only at the reset:

| | corr(a, sign v) | P(push left) at reset: on the training restart | on the acting start | acted | acting from the mode |
|---|---|---|---|---|---|
| ours seeds 0–2 (59k, 69k; unfixed 70k) | .49–.51 | 1.00 | **0** | .05–.27 | 1.00 |
| ours seed 3 | .93 | 1.00 | 1.00 | .84–.92 | 1.00 |
| theirs seeds 0 / 1 | .94 / .90 | .93 / .99 | (one code path) | .92 / .82 | — |

**The mechanism.** The world model's training restart (`RSSMCore.initial_state`, used by
`observe` at `is_first`) is the learned carry and the prior's **mode**, a one-hot code; their
`obs_step`, which serves both acting and training, is the same. Our collection lane's
`episode_start_beliefs` stepped the GRU from `initial_priors`, whose code is the prior's
**probability tensor** at the learned carry. So the first state of every episode on the
acting path was a state training never filters from; with the mode swapped in, it matches
the training restart exactly (deter cosine 1.000). On seeds 0–2 the actor reads the
off-training state as "push right" and the training state as "push left"; on seed 3 it
reads both as "push left", which is why seed 3 looks like the oracle in both runs.

**The test.** Parameters frozen at each snapshot, the lanes run fresh, only the episode
start changed (success rate / double-pump share / median successful length):

| | as is | episode start from the mode |
|---|---|---|
| fixed 68.6k seeds 0 / 1 / 2 | .55/.88/71 · .73/.66/71 · .97/.85/70 | **.97/.05/40 · .98/.02/41 · 1.00/.02/43** |
| unfixed 70.7k seeds 0–2 | 1.00 / .66–.98 / 63–65 | **1.00 / .03–.07 / 40–41** |
| seed 3, both runs | ≈ .04 / 41 | unchanged |

On every pumping seed of both runs the double pump goes, the median successful episode falls
to their 40, and success reaches .97–1.00, with no retraining.

**Where §29's restart fits.** The other reads place the CRSPP restart downstream of the
start. The cosine leaf's direct part still carries the slope (−64 / −65 / −12 on our seeds
0–2, +16 on seed 3, −14 / −7 theirs) but follows the data in the cross-data cell (−42 … −64
on our seed-0 states with either code, −14 … +2 on theirs), so it reflects the states each
agent visits rather than causing them. The fixed run's actor brakes after a right-first start
(pushes right while rolling left) where the unfixed one reverses, which is plausibly why a
bad start ends in a timeout with the restart and a double pump without it; its preference
rollout heads goal-ward on 81–92% of starts but decodes off the track on seeds 0–1 (t8 up to
x ≈ +3) — a candidate for that braking, untested.

Caveats: the test freezes parameters, so it does not show what training from the mode start
converges to; on seed 2 the frozen as-is success (.97) differs from its replay tail (.63)
while the double-pump shares agree; two of their seeds at one snapshot.

**The fix and its test.** `episode_start_beliefs` reads `RSSMCore.initial_state()`, the same
state training restarts from (acting ledger D6, register 31); one test pins the acting start
to `observe`'s first filtered step bitwise and fails on the old code. The confirmation is a
four-seed run from scratch with both fixes over the phase window (50k–250k): the prediction
is no phase on any seed.

**Addendum (2026-10-01): both fixes, four seeds from scratch, to 62.5k.** `runs/fix2_250k`
(f8d9fa5: the CRSPP restart and the act-time start from the mode; otherwise
`actsample_500k`'s configuration). Double-pump share / success rate / median successful
length per 12.5k bin:

| seed | both fixes | restart only (`fix_500k`) | neither (`actsample_500k`) |
|---|---|---|---|
| 0 | 15 15 12 12 **7**% · .61 .97 .95 .97 **.99** · 42 41 41 41 **40** | 27 31 29 50 63% · … .66 · … 69 | 38 27 72 67 75% · … .99 · … 66 |
| 1 | 23 8 9 7 **9**% · .58 .95 .95 .99 **1.00** · 44 41 41 41 **41** | 29 23 52 67 82% · … .74 · … 71 | 22 48 71 78 70% · … 1.00 · … 57 |
| 2 | 34 24 12 9 **8**% · .41 .94 .97 .98 **1.00** · 46 42 41 41 **40** | 29 28 44 74 87% · … .66 · … 69 | 27 30 49 81 92% · … .86 · … 67 |
| 3 | 27 23 14 11 **11**% · .48 .96 .97 1.00 **.98** · 44 42 41 40 **41** | 21 15 11 14 14% · … .99 · … 41 | 22 17 18 10 9% · … 1.00 · … 40 |

Every seed is inside the oracle's band from 12.5k on (oracle, corrected as above: double
pump 6–20%, success .93–1.00, median 39–41), where the two earlier runs had seeds 0–2 at 50–92% double pumps by 37.5k.
The prediction of §30 holds through 62.5k; the run continues to 250k for the rest of the
phase window.

**To 125k** (the unfixed run's peak bins), per 12.5k bin from 62.5k:

| seed | both fixes: double pump · success · median | neither (`actsample_500k`) |
|---|---|---|
| 0 | 7 9 8 7 7% · .99 1.00 1.00 1.00 1.00 · 40 throughout | 80 87 84 82 79% · 1.00 · 63 60 58 58 57 |
| 1 | 4 5 4 3 5% · .99 .99 1.00 1.00 .99 · 40 | 54 60 60 49 36% · 1.00 · 51 52 52 50 48 |
| 2 | 10 6 8 6 5% · .99 1.00 .99 1.00 1.00 · 40 | 96 94 95 92 91% · .96 .99 1.00 1.00 1.00 · 65 64 63 62 61 |
| 3 | 8 9 5 4 3% · 1.00 · 40 | 5 4 2 3 5% · 1.00 · 41 40 40 40 40 |

Through the window where the unfixed seeds 0–2 sat at 36–96% double pumps, every seed with
both fixes stays at 3–10%, success .99–1.00, median 40: the oracle's numbers (over 62.5k–125k
the oracle succeeds on every episode with 1–8% double pumps, median 39–40).

**Addendum (2026-10-02): to 300k, against the oracle read correctly — a shared late failure
phase.** From ~200k our seeds 1 and 3 began timing out (a timeout runs out the step limit:
101 rows ours, 100 theirs), and the run was stopped at 328,704 on a reading that the oracle
never fails there. That reading used the same running-average misread corrected above.
Read correctly, the oracle has its own failure phase: per 25k, seeds 1–3 fall to success
.74–.86 between 225k and 300k (seed 0 to .96), with double-pump shares up to 39%, and all four
recover to .97–1.00 by 350–450k. Over the same steps our seeds 1 and 3 were at .73–.91 and
seeds 0 and 2 at .95–1.00 — inside the oracle's band.

The anatomy is shared (`evidence/rollback.md`): on both sides a timeout is a right-first
start whose actor keeps pushing right while the car rolls back; their long successes are the
same start braking 5–8 steps and reversing at step 29–34. On both sides the cosine leaf of the
return rewards braking at roll-back states, and the preference rollout heads goal-ward from
them. What differs, on two of their seeds against our four: once timeouts appear, their
critic's slope toward reversing is positive at all five snapshots (+150 … +490) and their
seed 0's actor reverses by 300k, where ours is positive at two of six on the failing seeds
(−67 … +181). That is a lead, not a divergence: few seeds, single replay tails, phases offset
by ~25k, and outcomes inside the oracle's band. The test is recovery: the run resumed from
328,704 on 2026-10-02 (from a clean export of 662bfa6, so its provenance reads `unknown`) and
is read at 450k against the oracle's recovery.


**Addendum (2026-10-02): the recovery, to 450k.** Resumed from 328,704; success rate / double-pump
share of successes per 25k from 200k, against the oracle read correctly:

| seed | ours | oracle |
|---|---|---|
| 0 | .99 .98 .98 .95 .97 .94 .97 **1.00 1.00 1.00** · 10 18 24 24 17 6 6 5 4 3% | 1.00 .98 .96 .98 1.00 1.00 1.00 1.00 1.00 1.00 · 13 23 31 30 16 12 15 7 5 1% |
| 1 | .89 .86 .81 .74 .91 .97 **1.00 1.00 1.00 .99** · 7 3 5 13 13 18 5 11 6 6% | .94 .86 .85 .82 .90 .92 .94 .95 .97 1.00 · 15 20 14 10 10 13 9 11 12 3% |
| 2 | .98 1.00 1.00 1.00 .97 .95 .99 **1.00 1.00 1.00** · 18 15 27 26 22 16 8 2 2 4% | .98 .83 .77 .84 .87 .97 .95 .96 .98 .99 · 17 18 8 7 6 6 5 3 1 2% |
| 3 | .93 .83 .87 .79 .86 .93 .99 **1.00 1.00 1.00** · 5 2 3 5 7 7 6 3 4 6% | .99 .98 .93 .74 .86 .96 1.00 1.00 1.00 1.00 · 27 39 29 18 10 10 6 3 2 1% |

Every seed recovers, to .99–1.00 by 375–450k, as every oracle seed does; ours recovers as fast
as the oracle or faster (seed 1 by 350k against the oracle's 425k). The late failure phase is
shared in timing, depth and recovery, so the critic lead above does not show at the level of
outcomes. Returns at 450k are −69 … −78 per seed against the acceptance target's converged
−68 … −69.5 (`amortization-p0-addendum.md` §7, judged on windows at 2M). The run continues to 2M.
