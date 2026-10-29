# R-AIF state-based Mountain Car: acceptance against the oracle, 8 seeds per side (2026-10-10)

**Verdict.** Inconclusive by the pre-registered rule. At the final check (common reach
1.95M agent steps) the 8-seed means of the rule's statistic differ by −1.01 return points
(ours −77.24, theirs −76.23), 95% bootstrap CI [−3.19, +1.04], half-width 2.11 against the
rule's 1.0. On the §7 windows the two sides agree to within half a point: best-200k −69.10
against −68.73, last-500k −72.98 against −73.17, success rate 0.99 on both. The per-seed
spread (standard deviation 2.6 ours and 2.0 theirs on the rule's statistic, 3.5 and 3.1 on
the last-500k mean) is larger than the gap on every statistic. The replication arc closes
here.

**Acceptance criterion.** `docs/design/amortization-p0-addendum.md` §7: match the oracle,
judged on windows (best-200k window, last-500k mean, success rate), with the trajectory
shape part of the target.

## 1. Data

| side | seeds | source | code |
|---|---|---|---|
| ours | 0–3 | `runs/fix2_250k` (`scores.jsonl`), resumed to 2,000,8xx | f8d9fa5 (both boundary fixes, dd47e4a and f8d9fa5), resumed from 662bfa6 at 328,704 |
| ours | 4–7 | `runs/fix2_seeds4to7` (`scores.jsonl`), to 2,000,8xx | 85a7d71, `act_sampling: true`, `sample_scoring: false` (`run.json`) |
| theirs | 0–7 | `raif-paper/logs/oracle_state_mc_seed{N}/metrics.jsonl`, to 1,999,924–1,999,992 | `raif-paper` @ 9b1864e8fab9, the §7 command |

Both of our runs use the parity recipe of `src/hybrid_aif/reference/raif/run.py` with
fixtures `src/hybrid_aif/reference/raif/fixtures/oracle_mc_seed{0..7}.npz`, each seeded from
the oracle's own collect stream for that seed. The oracle's seeds 4–7 were trained for this
comparison with the identical command; their collect streams are the fixtures for our
seeds 4–7.

Row handling, applied identically to both sides:

- **Episodes.** Ours: every `scores.jsonl` row. Theirs: `metrics.jsonl` rows that carry
  `episode/length` and not `prior_episode/length` (the latter are the expert collect
  stream); the return is `episode/score`.
- **Success.** An episode succeeds if it ends before the step limit: theirs
  `episode/length < 100`; ours `length < 101`, because our length counts the reset row, so a
  timeout is 101 rows on our side and 100 on theirs. Their `episode/success_rate` is a
  running average and is not used. This rule misses the few episodes that reach the goal on
  the limit's last step (84 of 380,795 theirs, 76 of 396,226 ours).
- **Resumes.** A resume re-logs the steps between its checkpoint and the stop. Where a
  stream's step goes back, the earlier rows at or past the new step are dropped
  (ours: 92 duplicate rows near 328.7k in `runs/fix2_250k`; theirs: two overlaps on seed 3,
  near 1.04M and 1.69M). The effect on any window below is under 0.01 return points.

## 2. Window definitions

- **Best 200k window.** The highest mean return over the ten complete tiled windows
  [k·200k, (k+1)·200k), k = 0…9, i.e. up to 2,000k. Episodes are placed by their logged
  end step.
- **Last-500k mean.** The mean return over episodes ending in [1,500k, 2,000k), the same
  interval on both sides.
- **SR.** The fraction of episodes in [1,500k, 2,000k) that succeed (the rule above).
- **SR(> −100).** The §7 table's statistic: the fraction of the same episodes returning
  above −100. Stricter than SR; kept for continuity with §7.
- **Bootstrap.** Differences are ours minus theirs; 95% intervals resample seeds with
  replacement (10,000 resamples, `numpy.random.default_rng(0)`), as the stopping rule does.

`run.py --report` (`seed_report` in `src/hybrid_aif/reference/raif/run.py`) uses the same
best-200k rule: a window counts once the log reaches within 1% of its end, so a partial
trailing window is not counted and a log that ends a few dozen steps short of 2,000k (the
oracle's) still counts 1,800–2,000k. Its printed best-200k values for our seeds equal the
table below; run on the oracle's rows it reproduces §7's four-seed table exactly. Its
last-500k is taken relative to each run's last step rather than over [1,500k, 2,000k); the
two agree to 0.1 on every seed.

## 3. The §7 table, 8 seeds per side

| seed | ours best 200k | at | theirs best 200k | at | ours last 500k | theirs last 500k | ours SR | theirs SR | ours SR(>−100) | theirs SR(>−100) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | −68.6 | 1600–1800k | −68.8 | 400–600k | −68.9 | −70.7 | 100.0% | 99.9% | 99.5% | 96.4% |
| 1 | −69.9 | 1600–1800k | −68.7 | 1800–2000k | −72.1 | −71.8 | 99.1% | 99.7% | 96.6% | 95.5% |
| 2 | −68.8 | 400–600k | −69.5 | 1600–1800k | −75.1 | −72.0 | 98.8% | 100.0% | 96.1% | 96.9% |
| 3 | −70.8 | 400–600k | −68.1 | 400–600k | −72.0 | −73.2 | 99.3% | 99.2% | 96.2% | 94.0% |
| 4 | −69.5 | 400–600k | −68.9 | 1400–1600k | −74.8 | −74.7 | 97.9% | 97.6% | 93.1% | 94.5% |
| 5 | −68.1 | 1800–2000k | −69.3 | 1200–1400k | −71.0 | −79.5 | 99.7% | 97.5% | 96.7% | 85.7% |
| 6 | −69.0 | 400–600k | −68.6 | 1600–1800k | −79.8 | −69.5 | 96.5% | 99.8% | 90.5% | 98.1% |
| 7 | −68.2 | 400–600k | −68.0 | 1800–2000k | −70.1 | −74.0 | 100.0% | 96.6% | 98.2% | 94.9% |
| **mean** | **−69.10** | | **−68.73** | | **−72.98** | **−73.17** | **99.0%** | **98.8%** | **95.9%** | **94.5%** |
| sd | 0.91 | | 0.51 | | 3.51 | 3.06 | | | | |

| statistic | ours − theirs | 95% CI |
|---|---|---|
| best 200k | −0.37 | [−1.06, +0.28] |
| last-500k mean | +0.19 | [−2.84, +3.14] |
| SR | +0.1 pt | [−1.0, +1.3] pt |

All eight oracle seeds have a best-200k window in −68.0 to −69.5, the §7 converged band.
Six of ours do; seed 1 reads −69.9 and seed 3 −70.8. Every seed on both sides converges:
the lowest SR in the last 500k is 96.5% (ours) and 96.6% (theirs). The oracle's seeds 0–3
reproduce the §7 table's values exactly.

## 4. The pre-registered stopping rule

`runs/eight_seed/stopping_rule.py`, fixed on 2026-10-04 before the first check. Per seed:
the mean of 200k-window mean returns, windows starting every 50k from 400k up to T, where T is
the step all 16 seeds have reached, rounded down to 50k. Difference of the 8-seed means
(ours minus theirs), 95% bootstrap CI with seeds as the unit (10,000 resamples). Checks every
250k from 750k. STOP-EQUIVALENT if the CI lies inside (−1, +1) and the half-width is at most
1; STOP-DIFFERENT if the CI excludes 0, |diff| > 1 and the half-width is at most 1;
otherwise CONTINUE. Outputs: `runs/eight_seed/checks/*.txt`.

| check (T) | ours mean | theirs mean | diff | 95% CI | half-width | decision |
|---|---|---|---|---|---|---|
| 750k | −72.41 | −80.34 | +7.93 | [+2.27, +13.80] | 5.76 | CONTINUE |
| 1,000k | −80.64 | −81.80 | +1.15 | [−5.39, +7.14] | 6.27 | CONTINUE |
| 1,250k | −79.93 | −79.01 | −0.93 | [−5.23, +2.79] | 4.01 | CONTINUE |
| 1,500k | −78.85 | −77.44 | −1.40 | [−4.20, +1.04] | 2.62 | CONTINUE |
| 1,750k | −78.21 | −76.53 | −1.68 | [−3.98, +0.43] | 2.21 | CONTINUE |
| 1,950k (final) | −77.24 | −76.23 | −1.01 | [−3.19, +1.04] | 2.11 | CONTINUE |

Every run finished (ours at 2,000,896, theirs at 1,999,924–1,999,972), so the final check
is at the common reach rounded down, 1,950k. The rule never reached a STOP decision.

## 5. Trajectory shape

8-seed mean return per bin (range over seeds in brackets) and mean SR:

| bin | ours | SR | theirs | SR |
|---|---|---|---|---|
| 0–2k | −197.7 [−200.0, −193.3] | .04 | −198.5 [−200.0, −191.0] | .06 |
| 2–4k | −179.7 [−194.6, −159.6] | .23 | −188.5 [−199.5, −173.5] | .20 |
| 4–6k | −159.9 [−177.8, −139.7] | .40 | −157.3 [−184.5, −135.0] | .38 |
| 6–10k | −130.0 [−159.5, −115.5] | .67 | −138.0 [−160.0, −115.3] | .57 |
| 10–16k | −103.7 [−111.8, −96.1] | .91 | −107.4 [−122.2, −98.1] | .88 |
| 16–30k | −93.9 [−99.6, −88.4] | .96 | −94.0 [−105.4, −87.7] | .95 |
| 30–50k | −87.6 [−90.1, −84.6] | .98 | −86.6 [−89.3, −84.0] | .98 |
| 50–100k | −81.8 [−83.4, −80.4] | 1.00 | −80.0 [−82.3, −77.6] | 1.00 |
| 100–200k | −80.0 [−82.8, −78.6] | .99 | −79.9 [−82.1, −78.3] | 1.00 |
| 200–300k | −93.3 [−104.2, −84.6] | .91 | −100.6 [−116.3, −89.2] | .87 |
| 300–400k | −78.1 [−81.8, −73.8] | .97 | −86.1 [−108.6, −72.8] | .93 |
| 400–500k | −69.3 [−70.6, −68.3] | 1.00 | −71.5 [−77.7, −68.7] | .99 |
| 500–750k | −74.8 [−81.4, −68.2] | .98 | −83.4 [−96.7, −71.1] | .97 |
| 750–1000k | −87.6 [−122.0, −74.8] | .92 | −78.7 [−88.8, −73.1] | .98 |
| 1000–1250k | −75.9 [−81.7, −70.2] | .98 | −73.5 [−82.0, −69.8] | .99 |
| 1250–1500k | −76.5 [−83.8, −70.5] | .97 | −72.5 [−80.0, −69.5] | .99 |
| 1500–1750k | −74.6 [−86.8, −68.7] | .98 | −73.9 [−76.5, −70.1] | .99 |
| 1750–2000k | −71.6 [−74.2, −68.7] | .99 | −72.5 [−83.1, −69.0] | .99 |

The shapes are the same through 200k: first successes at 1.1–3.0k ours and 1.6–3.6k theirs,
a plateau near −80 over 50–200k, and a failure phase from ~225k whose per-seed SR minimum
(per 25k bin) falls at 250–325k on both sides (ours .68–.97, theirs .39–.96), recovered to
.97–1.00 by 425–500k on every seed of both sides.

After that, in the stopping rule's terms:

- **Early lead for ours at 750k, CI excluding 0** (+7.93, [+2.27, +13.80]). Our seeds
  recover from the shared failure phase faster (300–400k: −78.1 against −86.1) and hold
  better over 500–750k (−74.8 against −83.4).
- **Parity by 1M** (+1.15, [−5.39, +7.14]). Over 750k–1M our seed-mean SR falls to .92: one
  seed of ours (5) fails almost every episode over 800–900k (SR per 50k .23, .00, .50) and
  recovers by 950k. The oracle's deepest 50k bin anywhere after 500k is .66.
- **Theirs ahead by about a point from 1.25M on**, −0.93 to −1.68, with the interval
  straddling 0 at every check.

## 6. Residual: hypotheses, not findings

The only difference the rule detected is our lead at 750k, and it does not persist. What
remains is a gap of about one return point that the data do not resolve. Two hypotheses
are open; neither has been tested.

- **(a) Act-time sampling as an early/late trade.** The comparison run carries the two
  boundary fixes (dd47e4a, the CRSPP filter restart; f8d9fa5, the act-time episode start)
  together with act-time sampling (57235fb, `act_sampling`). The fixes are what brought the
  early phase onto the oracle's curve, and our early lead arrived with them. If acting on a
  sampled code helps early and costs late, the sign change between 750k and 1.25M would
  follow. No 2M run with both fixes and `act_sampling` off exists to test this.
- **(b) float16.** Their compute dtype is float16 and ours float32. Measured and demoted on
  2026-09-15 (`series-sweep.md`: no loss-scaler overflows on the actor, critic or info-gain
  optimizers over 2M steps on any seed); the quantization half of the argument survives and
  would need a run to close. It remains the suspect of last resort.

Register row 6.5 (the sampled preference rollout) is **not** a residual. It was built
(c68c6d3, `sample_scoring`), run from scratch (`runs/crspp_sampled_500k`) and excluded at
run level on 2026-09-30: its double-pump phase matched the mean-field control seed for seed
(`actor-entropy-diagnostic.md`, §28 addendum "register row 6.5 is excluded at run level").

## 7. Why close here

The final half-width is 2.11 at 8 seeds per side. A bootstrap half-width shrinks as
1/√n, so halving it needs about four times the seeds, and reaching the rule's 1.0 needs
about 8 × (2.11)² ≈ 36 seeds per side, 28 more on each side than were run. Our seeds 4–7 ran
as one vmapped 4-seed run that took 82 hours (`runs/fix2_seeds4to7.log` ends "stopped at step
2,000,896 after 82.44 h"). Seven more such batches per side is about 575 hours of compute per
side, to resolve a gap smaller than the spread between seeds of the same code. The
replication's question, whether this implementation of R-AIF behaves as theirs does on this
task, is answered as far as this benchmark can answer it at a reasonable cost: the same
early curve, the same failure phase and recovery, the same converged windows, and a late
difference of about one point that 8 seeds cannot place.

## 8. What a different benchmark would test instead

Mountain Car's converged agents differ by about one point on a −200…−68 scale, and the task
saturates (SR ≈ 0.99 on both sides), so the remaining hypotheses act on a quantity this
benchmark barely resolves. A continuous-control 3D task, as `docs/project-brief.md` points to
(its benchmark suite's continuous-control environments; `amortization-p0-addendum.md` §7
names robosuite `Door` as P4, with Meta-World `door-open-v2` as the fallback), would test
the same machinery where the preference model, the info-gain term and act-time sampling
carry more of the outcome: higher-dimensional state and action, longer horizons and a
success rate that is not already at ceiling. If (a) or (b) matters, it is more likely to show
as a difference in SR or ACR there than as a one-point return gap here.
