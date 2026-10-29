# trace65k: where the two sides differ at ~65k, with the restart matched

Checkpoints: ours `runs/fix_500k/snapshots_step68608` (step 68,608, seeds 0–3), second point `_step59392`,
unfixed reference `runs/actsample_500k/snapshots_step70656`. Theirs `raif-paper/logs/probe64k_state_mc_seed{0,1}`,
`checkpoint.ckpt` loaded read-only through `prefmodel/common_theirs.load_stage`, which reported **step 65,600** for both seeds
(so this is the 65.6k state, not step 0). Their replay was read "as of the checkpoint" (`tf_chunks`), chains
rebuilt by uuid → successor. One bug of mine, caught and fixed: partially written chunks hold 1024 rows of
which only the first n are valid; `dump_theirs.py` truncates to n.

## 1. What our failures are (replay tail, last 5k env steps per seed; theirs 4 × 1,250 rows)

`read1_episodes.py`, `read1b_phases.py` → `read1_tail5k.json`, `read1b_tail5k.json`. An episode is the rows from
one `is_first` to the next; success means a terminal row; a double pump is a success longer than 50 rows (`cf_double`'s rule).

| run | eps | SR | single / double / timeout | double share of successes | median successful length |
|---|---|---|---|---|---|
| ours 68.6k s0 / s1 / s2 | 63 / 63 / 60 | .65 / .65 / .63 | 10/31/22 · 11/30/22 · 2/36/22 | .76 / .73 / .95 | 70 / 70 / 70.5 |
| ours 68.6k s3 | 115 | 1.00 | 107/8/0 | .07 | 40 |
| ours 59.4k s0–2 | 61–66 | .61–.73 | — | .70–.89 | 69–71 |
| unfixed 70.7k s0–2 | 74–96 | .96–1.00 | timeouts 0/0/3 | .57–.99 | 52–66 |
| theirs 65.6k s0 / s1 | 113 / 110 | .99 / .98 | 107/5/1 · 99/9/2 | .04 / .08 | 40 / 40 |

What a timeout is (ours 68.6k, seeds 0–2, 22 apiece, all alike). Max position −0.18…+0.10 (the goal is at 0.45); max
energy 0.15–0.56 ×10⁻³ against 1.9 ×10⁻³ for a single-pump success; four velocity reversals; |a| .88, 55% saturated,
**sign(a) = sign(v) on only 61% of steps** (single pump: 96%). The first left push comes at step 78–86; on 30–33 steps the
actor pushes right while the car rolls left. Traces (in `read1b`): push right from the reset, climb to x ≈ −0.2, keep
pushing right while rolling back, so energy never builds; swing left too late; reach the left wall at step ~95.
So this is **neither over-pumping nor dithering**: it is a committed, saturated push to the right, against the
velocity, that wastes the energy it builds. Double pumps start the same way (first left push at step ~34, about 10 braking
steps) and recover. Single pumps, on every seed and on both sides, start by pushing left (first left push at step 0–2).
Unfixed doubles also start right, but brake only 0.6–6 steps: they go back and forth with the velocity, so they still succeed.

## 2. The actor

`ours_trace.py` / `theirs_trace.py`. The actor's mean `loc` (after tanh) and its scale are read under the training recognition posterior
over the stream tail: 80 contiguous 64-row windows, with the first 8 rows of each window dropped.

| run | mean |loc| | σ | entropy | corr(loc, sgn v) | sgn agree |
|---|---|---|---|---|---|
| ours 68.6k s0 / s1 / s2 | .93 / .92 / .90 | .21 / .21 / .20 | −.27 / −.28 / −.28 | .49 / .51 / .49 | .71 / .73 / .73 |
| ours 68.6k s3 | .93 | .16 | −.51 | .93 | .96 |
| unfixed 70.7k s0–2 / s3 | .90–.92 / .94 | .18–.19 / .17 | −.36…−.43 / −.49 | .67–.84 / .96 | .82–.90 / .97 |
| theirs s0 / s1 | .92 / .93 | .18 / .21 | −.39 / −.26 | .94 / .90 | .96 / .94 |

Magnitude, scale and entropy match the two sides; only the coupling to velocity differs. These are statistics over the
states each agent *visits*, so a low correlation partly reflects the timeout states themselves. By state bin, moving
left at x ∈ [−0.6, −0.3): our seeds 0–2 have mean loc −0.13…+0.17, while seed 3, the unfixed seeds and theirs have −0.74…−0.94.

**The first step of the episode differs (`reset_belief.py`, `read2b_acting.py`).** At the real `is_first` rows:

| run | P(left), training posterior | P(left), acting path (sampled / mean-field) | P(left), acting path with the initial code taken as its mode | recorded P(left) | deter cos, train vs act |
|---|---|---|---|---|---|
| ours 68.6k s0 / s1 / s2 | 1.00 / 1.00 / 1.00 | 0 / 0 / 0 (both) | 1.00 / 1.00 / 1.00 | .17 / .18 / .11 | .86 / .85 / .69 |
| ours 68.6k s3 | 1.00 | 1.00 | 1.00 | .92 | .89 |
| ours 59.4k s0–2 / s3 | 1.00 / 1.00 | 0 / 1.00 | 1.00 / 1.00 | .15–.23 / .84 | .69–.85 / .87 |
| unfixed 70.7k s0–2 / s3 | 1.00 / 1.00 | 0 / 1.00 | 1.00 / 1.00 | .05–.27 / .89 | .65–.85 / .88 |
| theirs s0 / s1 | .93 / .99 | (same function as training) | — | .92 / .82 | — |

The acting path is `run.make_collect_fn`, using `episode_start_beliefs` = `update_empirical_prior(initial_priors, 0)`; `initial_priors`
hands the GRU the code's **probability tensor** at the learned initial carry (max prob .15–.27 per class). The training restart
(`RSSMCore.observe` with `is_first`) and their `obs_step`, for acting and training alike, use the prior's **mode** one-hot
(`nets.py:57-59`, `get_stoch` → `dist.mode()`). Swap in the mode at act time and the deter matches the training one to cos 1.000,
with P(left) = 1 on every seed. On seeds 0–2 the actor therefore acts on a first-step belief it is never trained on, and that belief
says "push right". About 85% of their episodes start with a right push, and the right push leads into §1's timeout or double pump.
On seed 3 the same off-distribution belief happens to say "left". The split holds one-to-one, seed for seed, at 59k, 69k and on the unfixed run.

## 3. The objective's leaves (slope toward bang-bang, ×10⁻³ per unit ε at ε = .05, key set 1; + wants bang-bang)

The own held starts (slope_probe's windows on our side; rng-5 windows of their 65.6k replay on theirs); 8 keys; the split is exact.

| run | total (k2; ε=.2) | cosine | cos direct | reward | rew via cont | ig | critic | term share |
|---|---|---|---|---|---|---|---|---|
| ours 68.6k s0 | −86 (−79; −96) | −61 | −64 | −20 | −2 | 0 | −5.5 | .11 |
| ours 68.6k s1 | −69 (−71; −95) | −64 | −65 | −5 | −1 | 0 | −0.6 | .06 |
| ours 68.6k s2 | −14 (−17; −19) | −12 | −12 | −1 | 0 | 0 | 0 | .10 |
| ours 68.6k s3 | +14 (+14; +18) | +16 | +16 | −2 | 0 | 0 | 0 | .19 |
| ours 59.4k s0 / s1 / s2 / s3 | −66 / −44 / −43 / +26 | −62 / −44 / −42 / +29 | −63 / −44 / −42 / +25 | −3 / −1 / −1 / −3 | ≈0 | 0 | ≈0 | .06–.19 |
| unfixed 70.7k s0–3 | −5 / −23 / −8 / −5 | −1 / −10 / −8 / −5 | same | 0 / −9 / 0 / −1 | 0 | 0 | −3.6…0 | .09–.16 |
| theirs s0 / s1 | −20 / −12 | −14 / −5 | −14 / −7 | −6 / −7 | 0 / −1 | 0 | 0 | .19 / .16 |

The cosine leaf's direct part carries the slope on both sides. It separates our seed 3 (+16/+25) from our seeds 0–2 (−12…−65)
**and from theirs (−7/−14)**, so it does not separate pumping from oracle-like behaviour: seed 2 at 68.6k (−12, 86% double pump)
sits beside theirs, and seed 3, the oracle-like seed, is the odd one. The 59.4k totals reproduce the addendum's −66/−45/−43/+26.

## 4. The preference rollout (decoded position at t = 8 on the 512 own held starts)

| run | t8 mean | sd | corr(t8, start x) | goal share |
|---|---|---|---|---|
| ours 68.6k s0 / s1 | +0.90 / +2.20 | 1.22 / 2.64 | −.07 / +.03 | .86 / .83 |
| ours 68.6k s2 / s3 | +0.39 / +0.40 | .53 / .32 | +.12 / +.22 | .81 / .92 |
| ours 59.4k s0–3 | +0.36…+3.05 | .38–2.91 | −.15…+.26 | .84–.92 |
| unfixed 70.7k s0–3 | −0.24…−0.40 | .51–.56 | +.27…+.65 | .30–.43 |
| theirs s0 / s1 | −0.14 / −0.27 | .59 / .58 | +.60 / +.61 | .50 / .41 |

No, it does not look like theirs. The fixed run's rollout runs to the goal side on 81–92% of starts, depends only weakly on the start,
and on seeds 0–1 decodes to positions outside the track (x up to +3, where the physical range is ≤ 0.6), so it is off the data manifold.
Seed 3 is the same kind of rollout, so this does not separate the seeds either. Unexpectedly, the *unfixed* run at 70.7k now matches theirs
(start-conditioned, corr .27–.65, goal share .3–.4). §28's start-blind fixed point was a 20k description.

## 5. Cross-data cell at ~65k

Each side's code, from its own ~65k parameters (ours fix_500k s0 @68.6k; theirs seed 0 @65.6k), trained 256 real updates
on its own and the other side's frozen windows (`data_ours68608_s0.npz`, `data_theirs65_s0.npz`; copies of
`cell_ours.py`/`cell_theirs.py`, outputs `cell_*_65k.json`), read with the stage actor on the data source's held starts.
Leaves are ×10⁻³ per unit ε at ε = .05, for N = 0 / 128 / 256.

| cell | total | cos direct | reward |
|---|---|---|---|
| our code, our data | −86 / −65 / −63 | −64 / −62 / −52 | −20 / −4 / −11 |
| our code, their data | −11 / −10 / −14 | −11 / −10 / −14 | 0 / 0 / +1 |
| their code, our data | −48 / −36 / −46 | −43 / −33 / −42 | −4 / −4 / −4 |
| their code, their data | −20 / −36 / −27 | −14 / +1 / +2 | −6 / −37 / −34 |

At 65k the *sign* no longer separates the codes: both are negative on both data sources. The *size* of the cosine-direct
slope follows the **data**: −42…−64 on our seed-0 states with either code, and −14…+2 on theirs with either code. §3's
large negative slopes on our pumping seeds are a property of the states those seeds visit, not of our model code.
(Their code on its own data drifts toward a reward-via-continuation slope, as in §26; key set 2 is noisier there.)

## 6. Zero-training counterfactual on the reset belief (`frozen_eval.py` → `frozen_eval_*.json`)

The snapshot parameters were frozen and collected in the env with `run.make_collect_fn`, as training does (act_sampling on,
4 lanes × 2,000 steps, the same keys for both variants). The only change is that `episode_start_beliefs` takes the initial code as the prior's mode
(an in-process monkeypatch; nothing on disk changed). Cells: SR / double share / median successful length / P(first action left).

| snapshot, seed | as is | mode restart |
|---|---|---|
| fix 68.6k s0 | .55 / .88 / 71 / .19 | **.97 / .05 / 40 / .90** |
| fix 68.6k s1 | .73 / .66 / 71 / .25 | **.98 / .02 / 41 / .87** |
| fix 68.6k s2 | .97 / .85 / 70 / .14 | **1.00 / .02 / 43 / .84** |
| fix 68.6k s3 | 1.00 / .04 / 41 / .91 | 1.00 / .07 / 40 / .93 |
| unfixed 70.7k s0 / s1 / s2 | 1.00 / .88 · .66 · .98 / 63–65 / .03–.23 | **1.00 / .07 · .03 · .04 / 40–41 / .77–.87** |
| unfixed 70.7k s3 | 1.00 / .00 / 41 / .93 | 1.00 / .01 / 41 / .88 |

With no retraining, the mode restart turns every pumping seed of both runs into the oracle's behaviour (SR ≥ .97, double share 2–7%, median 40–43),
and it leaves seed 3 unchanged. (Seed 2's as-is SR here is .97 against .63 in its replay tail: this frozen 8k-step sample differs from the
training-time policy sequence. Double shares agree.)

## Caveats

- Read 1 is one 5k tail per seed. Read 2's correlations are over visited states.
- `reset_belief` uses 16 keys per reset observation, with the reset observations taken from the tail.
- Their acting path was not run separately. It is the same `obs_step` as training (code read: `agent.py:64-65`, `nets.py:131-141`).
  Its recorded first actions agree with the training posterior on 81–85% of resets.
- The "mode-init" variant is a counterfactual on the belief only. No run was made with it.
- One snapshot of theirs per seed, and only two of their seeds.

## Reading

The most likely locus is **acting, not the objective**: the belief our collection loop restarts each episode from. `episode_start_beliefs`
feeds the GRU the initial code's probabilities, while training (ours) and both acting and training on their side feed the code's mode.
On seeds 0–2 that untrained first-step belief makes the actor push right first, and §1 shows that the right-first start is what becomes
the double pump or the timeout. Seed 3 escapes because its off-distribution belief happens to map to "left". The same mapping holds on the
unfixed run. §6 shows that this mismatch, not the CRSPP, produces the double pump at 59–71k on both runs. The restart fix interacts through what follows
the start: the fixed actor brakes (pushes right while rolling left) rather than reversing, so right-first starts turn into timeouts instead of
doubles. The fixed run's off-manifold, goal-ward preference rollout (§4) is the candidate for that, but this is not tested.
The zero-training check (§6) has been run and confirms the acting-side half: the mode restart alone, with frozen parameters, removes the double pump and the
timeouts on all six pumping seeds. **Single next experiment:** a training run of `fix_500k`'s configuration (four seeds, at least 150k)
with `episode_start_beliefs` taking the initial code as the prior's mode, as `RSSMCore.initial_state` and their `initial` do. Read the double-pump share
and SR per 12.5k bin. The prediction is no 50k–250k phase on any seed. This is an acting-ledger change (D6), so it is Alex's call.
