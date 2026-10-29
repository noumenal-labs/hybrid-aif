# Execution ledger — the ACTING path, operation by operation

**Status: AUDITED 2026-09-06, ADJUDICATED AND RESOLVED 2026-09-07.** The audit was written
against our `044ce7b` and the oracle at fork commit `9b1864e8`; every row was a finding and
the DIVERGES rows were reported for the main session to rule on. They were ruled on, and the
consolidated parity round closed them on the `flag7-flat-replay` branch. Each divergence
below now carries its disposition — **FIXED**, **ARGUED** (a measured, deliberate
divergence), or **RECORDED** (cosmetic or inert, not fixed) — as its first line.

**Resolution summary (2026-09-07).**

| # | what it was | disposition |
|---|---|---|
| D1 | no reset row, no terminal row — the goal observation never entered replay | **FIXED** — `run_mc.make_collect_fn`, `seed.collect_episode` |
| D2 | the reward column indexed one row late | **FIXED** — same, and pinned by a new env-parity test |
| D3 | `is_success`/`cont` indexed one row late, on a false docstring justification | **FIXED** — same; `Episode.cont`'s docstring corrected |
| D4 | two row conventions inside one buffer | **DISSOLVED** — no conversion needed; verified, not fixed |
| D5 | the recorded action was clipped where theirs is the raw sample | **FIXED** — `habit_action`, `System._habit_fn`, `run_mc` |
| D6 | no GRU step at `is_first`, and mean-field beliefs | **FIXED (first half)** — `run_mc.episode_start_beliefs`; **second half CLOSED IN THE REFERENCE RECIPE 2026-09-28** — the lane acts on the keyed recognition face (`infer_states(key=)`, `Config.act_sampling`); keyless remains expected on every face  **First half REOPENED AND FIXED 2026-10-01:** the first-half fix stepped the GRU from `initial_priors`, whose code is the prior's *probability tensor* at the learned carry; their `obs_step` and our own training restart (`RSSMCore.initial_state`) step from the prior's *mode*, a one-hot. So every episode's first act-time state differed from the state training filters from (deter cosine with the mode start < 1; 1.000 once matched). On seeds where the actor read that state as "push right" it began against gravity, and that start became the double pump or a 101-step timeout: with parameters frozen at 68.6k / 70.7k, starting from the mode removes the double pump on all six pumping seeds (actor-entropy diagnostic §30). `episode_start_beliefs` now reads `RSSMCore.initial_state()` |
| D7 | the habit rolls a whole 8-step plan per acting step | **RECORDED** — behaviourally equivalent, ~10× the work in a loop that is 1.4% of the cycle |
| D8 | fixed-length windows truncate ρ's failure tail and inflate the terminal row | **FIXED** — the flat `StreamBuffer` this branch exists for |
| D9 | R-S's `C` summed every row | **FIXED** — `eval.stability_c`, logged per episode; the FLAG about the denominator's sign is retired |

**Three things the round found that no audit row had recorded**, each written where it
belongs as well as here:

1. **Divergence 26 (the normed layers' biases) is measurable, and coupled to divergence
   17 — and the *pair* is worse too.** The training ledger inferred 26 "inconsequential, and
   cheaper to close than to argue about". Closing it alone moved the tiny world model's
   keyless open-loop prediction from `0.0120` MSE to `0.0386` against a `0.0159` persistence
   baseline — but that was a hybrid (no bias + eqx's initializer) neither codebase has,
   because their `Linear` sets the bias and the initializer together and 17 was open for
   exactly the blocks 26 covers. So the pair was adjudicated and closed properly (Alex,
   2026-09-07) for all six blocks, measured across four init seeds, and **also reverted**:
   the sampled open-loop MSE went 0.0088 → 0.0206 against the 0.0159 persistence baseline,
   losing to persistence on 4/4 seeds (ours beats it on 4/4). The keyless probe is milder,
   1.37× worse and worse on 3/4. Both stay OPEN with numbers attached; the harm localizes to
   the GRU and the encoder, whose fans are the most lopsided in the model, and the
   `outscale: 0.0` heads are exonerated. Full account: training ledger 26 and
   `amortization.md` divergence 17.
2. **Their `imagine` keeps the raw action in `traj['action']`.** `img_step` clips a *local*
   copy (`agent.py:244-251`), so the `ig_head` and the BC term read an unclipped action on
   the training side too. D5 named these as consequences and left them to the sibling audit;
   both are now theirs, and the ensemble is deliberately handed the raw column while the
   recurrent scan gets the rescaled one (training ledger 2.4, 5.2, 6.39).
3. **D5 opened a gap while closing one.** Making the recorded column raw meant our
   world-model scan began training on raw actions where theirs trains on rescaled ones —
   the premise under which training-ledger row 2.4 had been dispositioned "MATCHED (inert)"
   was our column being pre-clipped, and D5 removed it. Closed in the same round
   (`to_world_batch`), and recorded rather than quietly fixed, because "which reader gets
   the raw action and which the rescaled one" is the part a later change could get wrong.

**The info-gain leaf's depth trend is the other way round from the audit's inference.**
Training-ledger 20 reasoned that "open-loop disagreement grows with rollout depth, so theirs
pays a strictly larger `model_uncertainty` cost". Measured at initialization it does not
grow: the leaf steps *down* 6.3% from index 0 to index 1 and is flat thereafter. The
divergence is real and its size is now known; the direction of its level difference was
guessed and guessed wrong.

## Why this exists, and what class of thing it looks for

Three audits already close three surfaces: the resolved-config audit (`amortization.md`
§(e)) closes wrong **values**, the transcription tests close wrong **formulas**, and
cross-run log comparison catches aggregate **numeric drift**. None of them looks at
**control flow** — which operations run, in what order, how many times, on what inputs.
The M/M⁺ alternation being dead code and our missing GRU step at `is_first` were both of
that class. This ledger walks the acting path — one full agent step, end to end — and
gives every operation a disposition, so the surface is exhausted rather than sampled. A
MATCHED row is worth as much as a DIVERGES row: it is what makes the walk a proof of
coverage.

Dispositions: **MATCHED** (same operation, same order, same inputs, same count),
**DIVERGES** (a numbered divergence below, with mechanism and magnitude),
**NEEDS-TRANSCRIPTION-TEST** (the trace shows the call happens; whether the arithmetic
inside it agrees is not a question a trace can answer), **N-A-BY-DESIGN** (no analogue,
with the reason — never as a shrug).

## Method: their code was **executed**, not only read

Their acting path was run in this machine's existing `raif-jepa` virtualenv (python 3.10,
the one the oracle runs themselves used, per `chain_oracle_state_mc_seed1_train.log:1`),
with `--configs gym_mtc tiny debug`, `envs.amount=1`, `envs.parallel=none`, state
observations, and `jax.jit=False` for the call trace. No file in either repository was
modified: the scripts live in the session scratch directory, drive
`embodied.Driver._step` directly, and use a `sys.setprofile` hook for the ordered call
sequence. One monkeypatch was needed — their `JAXAgent._setup` sets
`jax_transfer_guard='disallow'`, which trips under `disable_jit` — and it was applied to
`jax.config.update` in the runner, not to their source.

Three runs were taken: a six-transition call trace with `jit` off; a full 100-step timeout
episode; and a full successful episode driven by a bang-bang policy through their own
wrapper stack (their untrained agent does not reach the flag inside the time limit). Our
side was traced the same way with `jax.disable_jit()` and instrumented counters. Two
further measurements were taken directly off the checked-in oracle export
(`reference/fixtures/oracle_mc_seed0.npz`), which **is** the recorded output of their
acting path and is therefore the strongest available evidence about row conventions.

## Summary counts

| disposition | rows | after 2026-09-07 |
|---|---|---|
| MATCHED | 43 (one of them "matched in effect", C5) | 58 — C5 is now matched by the same *expression*, not merely in effect |
| DIVERGES | 19 rows, **nine** distinct divergences (D1–D9) | 4 rows, **two** divergences (D7 recorded, and B17/F4's pooling) |
| NEEDS-TRANSCRIPTION-TEST | 1 row (B19) | 1 row (B19), one of its four choices now settled |
| N-A-BY-DESIGN | 18 | 18 |
| **total** | **81** | **81** |

The two further transcription-test entries below (T2–T3) are not single-operation rows:
they are functions of theirs we do not implement at all, or a test the repository already
promises and does not have. Section totals: A 17, B 21, C 21, D 11, E 7, F 4.

Of the nine divergences, **two were already on the record** (D2 and D6; `amortization.md`
§(f)'s "four adjacent findings" and the acting-loop `is_first` finding respectively) and
are carried here with measurements attached rather than re-proposed. **One is bounded by
work already in flight** (D8, the `flag7-flat-replay` worktree) and is reported as that
work's quantified value. **Six are new**: D1, D3, D4, D5, D7, D9.

---

## A. The loop and the driver

| # | operation (theirs) | | ours | note |
|---|---|---|---|---|
| A1 | `Driver.reset`: zero action, `reset=True`, `state=None` — `driver.py:25-31` | MATCHED | `build` sets `t_in_episode=0`, a fresh per-lane `env.reset`, `beliefs=episode_start_beliefs` — `run_mc.build` | both start every stream on a reset with a zero previous action, and both take one transition from it (D6) |
| A2 | `env.step(prev_acts)` runs **before** `policy` — `driver.py:53-56` | MATCHED | `env.observe(env_state)` then `infer_states` then `act` — `run_mc.py:415-418` | the policy sees the observation its action is taken at, both sides |
| A3 | `log_`-prefixed action keys withheld from the env — `driver.py:52` | N-A-BY-DESIGN | — | we have no logged action channel; `log_entropy` is theirs alone |
| A4 | `convert()` casts obs and acts to float32/int32/uint8/bool — `driver.py:54,57`, `basics.py` | MATCHED | float32 throughout | the compute dtype (their `float16`) is a separate matter, ARGUED 9 in §(e) |
| A5 | `state is None` ⇒ `_init_policy` ⇒ `policy_initial` — `jaxagent.py:56-57` | MATCHED | `beliefs` ride `SeedState` and are set at build/resume only — `run_mc.py:786` | theirs fires exactly once per process, ours once per run; `_state` survives every `driver(...)` call |
| A6 | **the action is zeroed on `is_last`** — `driver.py:58-60` | **D1, FIXED** | `jnp.where(is_last, 0, action)` — `run_mc.make_collect_fn`; `seed.collect_episode` clears `can_self_imitate` there too | measured: `action[-1] = [-0.]`, `can_self_imitate[-1] = 0` in both the live trace and the fixture |
| A7 | `acts['reset'] = is_last` ⇒ the next `env.step` resets — `driver.py:61` | MATCHED | in-scan `jnp.where(is_last, fresh, stepped)` — `run_mc.make_collect_fn` | now exactly theirs: on an `is_last` row the env is not stepped at all, and the reset lands on the next iteration |
| A8 | `trns = {**obs, **acts}`: row `t` is `(obs_t, a_t)` with `reward_t` caused by `a_{t-1}` — `driver.py:63` | **D2, FIXED** | the `incoming_reward` carry: row `t` records the reward that produced `obs_t` — `run_mc.make_collect_fn` | pinned by `test_our_env_reproduces_their_recorded_observations_and_rewards`, whose negative control is the error of 2 our old convention gave |
| A9 | `_eps[i].clear()` on `is_first`, before the append — `driver.py:64-67` | MATCHED | `split_stream` cuts on `is_last` and carries the tail — `run_mc.py:558-575` | both recover episodes from the flags alone |
| A10 | the **reset transition is a recorded row**: reward 0, `is_first=True`, a real sampled action — `gym.py:56-59` | **D1, FIXED** | row 0 is the reset observation at zero reward with its real action | measured: `reward[:3] = [0, -2, -2]`, and row 0 of fixture episode 0 carries action −1.0 with reward 0 |
| A11 | the **terminal observation is a recorded row**: `obs` = the goal state, `is_terminal=1`, `is_success=1` | **D1/D3, FIXED** | recorded: `is_last` fires on the row whose own observation is at the flag, and the goal state now enters replay | measured live: `state=[0.4507, 0.0633] rew=0.00 last=1 term=1 succ=1` |
| A12 | `on_step` callbacks per env index, `step.increment` before `train_step` — `driver.py:68-72` | MATCHED | one gradient block per `train_every` pooled steps — `run_mc.py:501-546` | same cadence; see E2 |
| A13 | `on_episode` callbacks are **chained** (`ep = fn(ep.copy(), i)`) at `is_last` — `driver.py:73-80` | MATCHED | `label` then `route`, in that order — `run_mc.py:1061-1065` | their chain is `print → per_episode → lookback → add_trajectory`; ours is the last two |
| A14 | `BatchEnv.step` steps all four envs in lockstep — `batch.py:30-39` | MATCHED | the vmapped lane axis, `LANES = 4` | CLOSED 8 in §(e); the trace confirms lockstep, so lane ≡ env worker exactly |
| A15 | `driver(policy, steps=100)` outer batching; `_state` persists across calls — `train.py:206-209` | MATCHED | the chunk loop, `CHUNK_STEPS = 1024` | the outer batch size is bookkeeping on both sides |
| A16 | random prefill until `len(replay) ≥ batch_steps` — `train.py:139-150` | MATCHED | fixture seeding of both buffers — `run_mc.py:767-774` | **unreachable in the oracle run**: `collect_prior` writes into the same `logdir/replay`, so 1911 steps are already there and the prefill branch is skipped. Neither side ever collects with a random policy |
| A17 | `RandomAgent`'s `can_self_imitate=False` for the prefill policy — `train.py:144` | N-A-BY-DESIGN | — | consequence of A16 |

## B. The environment and its wrapper stack

Wrapping order, innermost first (`run/base.py:89-123`): `GymEnv` → `NormalizeAction` →
`ActionRepeat(2)` → `ExpandScalars` → `StateConcatenation('^$')` → `TransitionDelta('^$')`
→ `TimeLimit(100, reset=True)` → `ClipAction` → `MeasureSuccessRate` →
`MeasureRStability` → `MeasureCoverage`.

| # | operation (theirs) | | ours | note |
|---|---|---|---|---|
| B1 | `GymEnv.step` reset branch: full obs, `reward=0.0`, `is_first=True`, `is_last=False`, `is_terminal=False` — `gym.py:56-59` | **D1, FIXED** | the reset row, with all four values | the mechanism behind A10 |
| B2 | non-reset branch: `is_last = done or truncated`, `is_terminal = info.get('is_terminal', done)` — `gym.py:64-70` | **D3, FIXED** | `is_success = reached_goal(env_state)` — the predicate on **this row's** observation | `mtc.py` puts no `is_terminal` in `info`, so `is_terminal == done == (pos ≥ 0.45)` and one column serves both |
| B3 | Mountain Car physics — `mtc.py:156-172` | MATCHED | `MountainCar.physics_step` | **measured**: stepping our env with the fixture's actions reproduces their `state` column to `8.9e-07` (float32 round-off) over twelve episodes |
| B4 | sparse reward `-1.0`, `0.0` on the terminating raw step — `mtc.py:192-196` | MATCHED | `MountainCar.agent_step` | **measured**: reward reproduces with **zero** error under their index convention |
| B5 | `easy_mode` termination `position ≥ 0.45` — `mtc.py:176-177` | MATCHED | `MountainCar.reached_goal` | covered by `test_mountain_car_env.py` |
| B6 | `self.terminated` is sticky: physics is skipped and the state freezes — `mtc.py:159` | MATCHED | `done` carried through the repeat, `jnp.where(done, x, x_next)` | reachable only inside one agent step on either side, since `is_last` resets |
| B7 | reset draws `position ~ U(-0.6, -0.4)`, `velocity = 0` — `mtc.py:205-210` | MATCHED | `MountainCar.reset` | |
| B8 | the observation dict carries `image` (64×64×3, pygame) beside `state`, on every step — `mtc.py:364-372` | N-A-BY-DESIGN | state only | `encoder.cnn_keys='^$'` so nothing reads it, but it **is** written to their replay and re-scaled by `preprocess` every train step |
| B9 | `NormalizeAction` maps `[-1,1]` onto the action box — `wrappers.py:82-101` | MATCHED | — | identity on `Box(-1,1)`; §(e) already dispositions `wrapper.discretize` |
| B10 | `ActionRepeat(2)`: sum reward over two raw steps, break on `is_last or is_terminal` — `wrappers.py:59-69` | MATCHED | the statically unrolled masked pair in `agent_step` | verified equal case by case: `0` if the first raw step reaches the flag, `-1` if the second, `-2` otherwise. Confirmed live (`rew=0.00` on the goal row) |
| B11 | `ExpandScalars` — `wrappers.py:161-169` | N-A-BY-DESIGN | — | shape bookkeeping for their dict spaces |
| B12 | `StateConcatenation('^$')` — `wrappers.py:333-361` | N-A-BY-DESIGN | — | **verified inert**: `concat_keys` is empty, `state_concat_needed` is False, `obs_space` is returned unchanged. §(e)'s N/A confirmed |
| B13 | `TransitionDelta('^$')` — `wrappers.py:240-270` | N-A-BY-DESIGN | — | **verified inert**: `extended_keys` is the empty set (the trace prints `extended_keys: set()`), so no `_delta` key is added |
| B14 | `TimeLimit(100, reset=True)`: `_step` counts non-reset steps, sets `is_last` at 100 — `wrappers.py:35-52` | MATCHED | `is_last = t_in_episode >= time_limit`, with `t_in_episode` the row index | now equal in rows too: 100 applied actions, 101 rows, one driver transition per scan iteration |
| B15 | `ClipAction` clips to `[-1,1]` **on the way into the env only** — `wrappers.py:78-80` | **D5, FIXED** | `habit_action` returns the raw draw; `jnp.clip` into the env and `controls.soft_clip` into the transition, both at the call site | measured: 25.7% of their recorded actions have `|a| > 1`, max 2.271; 41% of ours at init |
| B16 | `MeasureSuccessRate`: `is_success = state[0] ≥ 0.45` on **this** observation, `has_succeeded` its running OR, reset at `is_first` — `wrappers.py:693-722` | **D3, FIXED** | `success = reached_goal(env_state)`, `ran_on = succeeded \| success` | the index now matches the logic. Verified: the predicate on their `obs` column reproduces their `is_success` column with **zero** mismatches over all 192 fixture episodes |
| B17 | `MeasureSuccessRate.success_rate`: a 100-episode window, **per env worker** — `wrappers.py:706-717` | N-A-BY-DESIGN | SR is computed from the pooled `scores.jsonl` — `eval.sr` | their per-worker window is a logging channel; ours is the paper's statistic over the pooled stream. Recorded because the populations differ |
| B18 | `MeasureRStability`: `C += reward + 1.1` on non-terminal rows, `C` closed at `is_last or is_terminal` — `wrappers.py:754-763` | **D9, FIXED** | `eval.stability_c`: `Σ_{t<T} (reward[t] + offset)`, logged per episode as `scores.jsonl`'s `stability` | theirs skips the closing row's reward and includes the reset row's 0; ours now does both |
| B19 | `MeasureRStability.relative_stability` formula — `wrappers.py:742-751` | NEEDS-TRANSCRIPTION-TEST | `eval.relative_stability` | see T1. The `abs`-around-the-ratio half is now settled and the FLAG retired; the other three choices are still untested |
| B20 | `MeasureCoverage`: `coverage_state`, `coverage_mean` — `wrappers.py:766+` | N-A-BY-DESIGN | not computed | a logging channel only; not one of §6's three metrics, and no loss reads it |
| B21 | the measurement wrappers see `{**obs, **action}` — `wrappers.py:634-637` | MATCHED | — | so `is_success` is a function of the observation, which is what B16 turns on |

## C. `Agent.policy` — recognition, the prior step, and the action

The executed call sequence for one non-first transition, verbatim from the trace with
`jit` off (file:line from `raif/aif/`):

```
driver.py:50::_step
  agent.py:59::policy
    agent.py:106::preprocess
    nets.py:305::MultiEncoder.__call__ → nets.py:511::MLP.__call__ → 2× Linear
    nets.py:131::obs_step
      nets.py:248::_mask ×4            # prev_state's 3 leaves + prev_action, by (1 - is_first)
      nets.py:42::initial              # rebuilt EVERY step: Linear → _stats → get_dist
      nets.py:248::_mask ×3            # the initial's 3 leaves, by is_first
      nets.py:167::img_step
        Linear(img_in, dropout=True) → nets.py:209::_gru → Linear(img_out, dropout=True)
        nets.py:231::_stats(img_stats) → nets.py:122::get_dist   # the PRIOR, sampled
      Linear(obs_out) → _stats(obs_stats) → get_dist              # the POSTERIOR, sampled
    agent.py:368::ActiveInferencePlanner.policy
      agent.py:401::ImagActorCritic.policy
        nets.py:511::MLP.__call__ (actor) → nets.py:548::Dist.__call__ → 2× Linear
```

| # | operation (theirs) | | ours | note |
|---|---|---|---|---|
| C1 | `preprocess`: float cast, `/255` for `ndim>3` uint8, `cont = 1 - is_terminal` — `agent.py:106-117` | **D3, FIXED** | `Episode.cont = 1 - is_success`, now on their index | the formula never changed; the column it reads did. The docstring's false freeze justification is replaced by the measurement that refutes it |
| C2 | `MultiEncoder` on the `state` key, `symlog_inputs=true` — `nets.py:305` | MATCHED | `RSSMCore.tiny`'s encoder | ✓ in §(e) |
| C3 | `rssm.initial(len(is_first))` is rebuilt on **every** step and masked away when `is_first` is false — `nets.py:80-82` | MATCHED | `initial_priors` hoisted out of the scan — `run_mc.py:410` | ours hoists what theirs recomputes; identical because the recomputation is a pure read of the same weights |
| C4 | `initial`'s stoch is `get_stoch(deter_0) = dist.mode()`, sharing `img_out`/`img_stats` — `nets.py:203-207` | **DIVERGES D6** | `initial_state` reads `B[f].probs(...)` off the same shared head — `amortized.py:581-601` | the head and its weights are shared on both sides; theirs takes the argmax, ours the probability tensor |
| C5 | `obs_step` soft-clips `prev_action` by `a·min(1, 1/\|a\|)` — `nets.py:134-136` | MATCHED, and now by the same expression | `controls.soft_clip`, applied where a transition consumes the action and nowhere else | the FLAG is discharged: the two were equal only by accident of the action already being clipped (D5). The gradient half of the same expression is training-ledger 22 |
| C6 | `obs_step` masks `prev_state` and `prev_action` by `(1 - is_first)` — `nets.py:137-138` | **D6, FIXED** | `run_mc.episode_start_beliefs` = `update_empirical_prior(initial_priors, 0)` — the mask *and* what follows it | this also closed a train/act asymmetry: `RSSMCore.observe` always took that step at `t = 0` |
| C7 | `obs_step` adds the masked `initial` — `nets.py:139-141` | MATCHED | `initial_priors` handed in as the prior at `t=0` | |
| C8 | `obs_step` then calls **`img_step`** — one GRU step from the initial carry with `prev_action = 0` — `nets.py:142` | **D6, FIXED** | `episode_start_beliefs`, hoisted out of the collection scan and used as the reset target and at `build` | **measured**: our step-0 `deter` equalled `deter_0` exactly (Δ = 0.0), and one GRU step with `a = 0` moves it by `max \|Δ\| = 0.610`, `mean \|Δ\| = 0.115` per unit on a tanh-bounded carry — a recursion, so it displaced the whole episode |
| C9 | `img_step`: `Linear(img_in, dropout=True)` — `nets.py:121` | N-A-BY-DESIGN | no dropout | `Linear._dropout` defaults to 0.0 and the rssm `**kw` sets no `dropout`, so the mask is all-True and the rescale is `/1.0` — numerically inert, but it **consumes an `nj.rng()` draw** (twice per `img_step`). Only matters to anyone diffing RNG streams |
| C10 | `_gru`: `concat(deter, x)` → one Linear at `3·deter` with `act='none'` → the reset/cand/update algebra — `nets.py:209-229` | MATCHED | `RSSMCore`'s GRU | transcription-tested |
| C11 | `_stats` with `unimix=0.01`: `logit = log((1-u)·softmax(logit) + u/K)` — `nets.py:231-241` | MATCHED | `unimix_logits` on both categorical heads | ✓ in §(e) |
| C12 | the prior's stoch is **sampled** (`OneHotDist(logit).sample`) — `nets.py:126` | **DIVERGES D6** | `predict` returns the probability tensor | already on the record; the keyless-face convention |
| C13 | `concat(prior['deter'], embed)` → `obs_out` → `obs_stats` → the **posterior**, sampled — `nets.py:143-148` | **DIVERGES D6** | `recognize` returns `module.probs(...)` | same divergence; the concatenation and the head order match |
| C14 | `post = {'stoch': sampled, 'deter': prior['deter'], **stats}` — the posterior inherits the **prior's** deter — `nets.py:148` | MATCHED | our carry factor is untouched by `recognize`, so it is the predicted `deter` | the two express the same thing: only the code is re-read from the observation |
| C15 | `planner.policy` = `{'action': self.actor(state)}`, **one** actor evaluation — `agent.py:368,401-402` | **D7, RECORDED (not fixed)** | the habit planner rolls `policy_len = 8` steps and keeps the first — `habit.py:106-124` | **measured**: 8 actor evaluations and 16 `amortized.predict` applications per `act()` call, against their 1 and 1. Behaviourally equivalent (the first draw's key does not depend on `policy_len`); ~10× the neural work in a loop that is 1.4% of the cycle |
| C16 | the actor reads `[deter, flat stoch]` via `Input` — `nets.py:712-737` | MATCHED | `objectives.amortized.features` | ✓ in §(e) |
| C17 | `mode='train'`: `log_entropy` is taken **before** the sample, then `action = dist.sample` — `agent.py:71-74` | MATCHED | `habit_action` samples | the entropy read consumes no key; `log_entropy` is a `log_`-prefixed column and never reaches the replay |
| C18 | `mode='eval'` zeroes `log_entropy` — `agent.py:67-70` | N-A-BY-DESIGN | — | `train_augmented` never uses eval mode; §(e) already dispositions the whole `run.eval_*` subtree |
| C19 | `state = ((latent, outs['action']), task_state)` — the carried previous action is the **unmasked** sample — `agent.py:75` | MATCHED | our carry holds the same action | harmless on both sides: the only row where the two differ is `is_last`, and the next step's `is_first` mask zeroes the carried action anyway |
| C20 | `task_state` is `{}` — the planner carries nothing — `agent.py:398-399` | MATCHED | the habit planner is stateless | |
| C21 | one `np.random.default_rng(seed)` feeds **both** `policy` and `train` — `jaxagent.py:33,211-219` | N-A-BY-DESIGN | separate `key` (per lane) and `train_key` (per seed) — `run_mc.py:340-342` | their acting stream is interleaved with their training stream; ours are split by construction. §(e) already dispositions `seed` |

## D. The per-episode callbacks

| # | operation (theirs) | | ours | note |
|---|---|---|---|---|
| D-a | `print_episode` returns `ep` (the chain would break otherwise) — `callbacks.py:62-66` | N-A-BY-DESIGN | the console line in `train` | `length = len(reward) - 1`, which is 100 for a timeout — equal to our row count (D1 cancels here) |
| D-b | `per_episode` logs `length`, `score`, `successfulness`, and the three wrapper channels — `callbacks.py:70-102` | MATCHED | `scores.jsonl` per episode — `run_mc.py:1069-1082` | `score` is the raw summed reward with no offset; our sum agrees despite D1, because the row theirs adds carries reward 0 and the row we drop is the one theirs zeroes |
| D-c | `per_episode` runs **before** `lookback`, so the logged `can_self_imitate` is pre-overwrite — `train.py:131-135` | MATCHED | we log neither | ordering confirmed from the `on_episode` registration order |
| D-d | `lookback_on_episode`: the success gate, the backward carry, the forward shift, the failure decay — `callbacks.py:31-59` | MATCHED | `revise.pref_labels` — `revise.py:67-156` | walked index by index, including their off-by-one list construction (`imitation_mask[i] = c_{i+1}`), which our `lax.scan` with `reverse=True` reproduces exactly |
| D-e | `lookback` mutates `ep` in place through the shallow `ep.copy()` chain — `driver.py:79` | MATCHED | `label` returns a new `Episode` | same effect: the callback after it sees the labelled columns |
| D-f | `add_trajectory`: `has_succeeded[-1]` ⇒ **both** buffers, else `replay` only — `callbacks.py:17-29` | MATCHED | `route` — `run_mc.py:596-606` | one code path, both sides, the fixture's episodes included |
| D-g | `add_trajectory` writes `is_positive_memory` — `callbacks.py:20,27` | N-A-BY-DESIGN | no such column | **verified**: grepped the whole `raif/` tree — the key is written to the buffers and never read. Dead data |
| D-h | `add_trajectory` filters `log_`-prefixed keys at write time — `callbacks.py:21,28` | MATCHED | our `Episode` has no logged columns | this is what keeps `log_entropy` out of their replay; `reset`, `image`, `coverage_*`, `relative_stability`, `success_rate` and `is_terminal` **are** written |
| D-i | `replay.add(..., worker=worker)` but `positive_replay.add(...)` with no worker — `callbacks.py:21,28` | N-A-BY-DESIGN | one buffer per seed, no chain identity | theirs puts all four workers' successes on M⁺'s worker-0 chain; episodes are added whole, so no window splices two workers mid-episode |
| D-j | labelling happens **once per finished episode**, before the write — `train.py:134-135` | MATCHED | `as_collected` → `label` → `route`, at the chunk boundary — `run_mc.py:1060-1065` | ours lags by up to one chunk (the documented replay-lag FLAG) but the schedule is the same |
| D-k | the episode written is the **whole** episode, whatever its length — `callbacks.py:21,28` | **D8, FIXED** | `StreamBuffer.extend` splices whole episodes onto a flat step ring; nothing is padded or truncated | the `flag7-flat-replay` work this round sits on; quantified below |

## E. Counters and the training gate

| # | operation (theirs) | | ours | note |
|---|---|---|---|---|
| E1 | `step.increment()` once per transition **per env** — `train.py:136` | MATCHED | `counters.step += chunk_steps`, pooled over lanes | both count pooled agent steps, and the reset transition counts on their side |
| E2 | `should_train = Ratio(64/1024)`: `repeats = int((step - prev) · ratio)` — `when.py:33-42` | MATCHED | `train_every = 16` pooled steps per gradient block — `run_mc.py:498-505` | **verified from the arithmetic**: with `step` advancing by 1 per call and ratio `1/16`, the return is 0 fifteen times then 1 — never ≥ 2, which is the alternation ruling's mechanism |
| E3 | `Ratio`'s **first** call returns 1 unconditionally — `when.py:37-39` | N-A-BY-DESIGN | our first gradient step lands after the first 16 pooled steps | a one-off 16-step offset at the head of a 1M-step run |
| E4 | `should_sync(updates)`, `should_log`, `should_save` — `train.py:173-186` | N-A-BY-DESIGN | our own metrics/checkpoint cadence | §(e) dispositions all three |
| E5 | training is registered **after** the prefill, so no gradient step runs during it — `train.py:187` | MATCHED | ours seeds before the loop | moot given A16 |
| E6 | the gradient step reads the buffers as they stand at that transition — `train.py:156-172` | **DIVERGES D8** | our buffers are frozen for the chunk | the documented replay lag; ≤1024 pooled steps, ≈0.1% of the episode stream at 1M |
| E7 | `report(batch[tricky_number])` with a random buffer pick — `train.py:177-179` | N-A-BY-DESIGN | `probe` on a fixed M⁺ batch | a logging path; `batch[1]` is always `None` under E2, so `tricky_number` collapses to 0 |

## F. Evaluation and the three metrics

| # | operation (theirs) | | ours | note |
|---|---|---|---|---|
| F1 | `train_augmented` builds **no** eval driver — `train.py:102-212` | MATCHED | `eval.py` folds recorded training episodes; nothing steps an env | ✓; §(e) dispositions `run.eval_*` |
| F2 | ACR = `episode/score` = the raw summed reward — `callbacks.py:64` | MATCHED | `eval.acr` | our per-episode sum agrees (see D-b) |
| F3 | SR from `MeasureSuccessRate`'s per-worker 100-episode window — `wrappers.py:706-717` | N-A-BY-DESIGN | `eval.sr` over the pooled last 100 episodes | see B17 |
| F4 | R-S from `MeasureRStability`, per worker, `C` excluding the terminal row — `wrappers.py:742-763` | **D9, FIXED (accumulation); pooling still ours** | `eval.relative_stability` over `eval.stability_c`, pooled | the accumulation is theirs; the per-worker-vs-pooled population difference stays, recorded at B17 |

---

## The divergences

### D1 — episode row structure: theirs has a reset row and a terminal row, ours has neither

***FIXED 2026-09-07.*** `run_mc.make_collect_fn` now records their driver's rows: `T` applied
actions become `T + 1` rows, opened by a reset row (a complete observation at `reward = 0`
with `is_first` set and a real sampled action) and closed by a terminal row whose
observation *is* the state the episode ended in, with the action zeroed. `seed.collect_episode`
— the bang-bang fallback seeder — adopted the same layout, so the fallback and the fixture put
one convention into one buffer (D4). One scan iteration is now exactly one driver transition,
which also makes our pooled step counter theirs (E1).

**Mechanism.** Three of their operations compose. `GymEnv.step`'s reset branch
(`gym.py:56-59`) emits a *complete observation* with `reward=0.0` and `is_first=True`, and
that observation is a full driver transition: the policy acts on it, the action is
recorded, `step` increments, and `train_step` may fire. At the other end, `is_last` fires
on the transition whose observation is the goal state, so that observation is recorded —
and the driver zeroes the action on it (`driver.py:58-60`). Our loop records neither: our
first row is the reset observation *with its real action and its real reward*, and our
last row is the observation *before* the goal, after which the in-scan reset swaps the
state and the goal observation is discarded.

**Magnitude — measured.** A 100-step timeout episode is 101 rows for them and 100 for us;
a 38-action success is 39 rows for them and 38 for us. So the affected fraction is
`2/(T+1)` — about 2% of a timeout episode and 5% of a converged 34-step one. Concretely,
in their data:

- every episode carries one row with `reward = 0` at a non-goal state (live trace:
  `reward[:3] = [0, -2, -2]`; fixture episode 0 row 0: `obs = [-0.4785, 0]`, `action =
  -1.0`, `reward = 0`);
- every successful episode carries one row whose observation *is* the goal state
  (`obs = [0.4556, 0.054]` in the fixture, `[0.4507, 0.0633]` live), with `is_terminal =
  is_success = has_succeeded = 1`, `reward = -1` or `0`, `action = [-0.]` and
  `can_self_imitate = 0`.

**Consequential?** Yes, in two places, and the two point in opposite directions. Their
reward head is trained to predict `reward = 0` at *every episode-start state*, and 0 is
also the goal reward — a spurious signal we do not reproduce, where we are arguably the
cleaner of the two. Against that, the goal observation itself **never enters our replay**,
so our decoder, our reward head and our CRSPP have never seen the state the task is about;
in a task whose only reward structure is what happens at the flag, that is not obviously
cosmetic. I have not measured a downstream effect and do not claim one.

### D2 — the reward column is indexed one row apart (ALREADY ON THE RECORD, measured, ***FIXED 2026-09-07***)

***FIXED.*** The collection carry gained an `incoming_reward` field: row `t` records the
reward that produced `obs_t`, `0` on a reset row. And the measurement below is now a **test** —
`tests/test_oracle_fixtures.py::test_our_env_reproduces_their_recorded_observations_and_rewards`,
over all four fixtures — with the second row of its table (the error of exactly 2 under our
old convention) asserted as the negative control, so a half-change fails loudly. Re-measured
over all 192 fixture episodes rather than seed 0's twelve: obs `1.25e-06`, reward exactly 0.

Recorded at `amortization.md` §(f), "Four adjacent findings", first bullet. This audit adds
the measurement. Stepping our `MountainCar` with the fixture's action column, from the
fixture's own first observation, over twelve episodes:

| alignment | max abs error |
|---|---|
| theirs (`a_t` at `obs[t]` produces `obs[t+1]` and `reward[t+1]`) | obs `8.94e-07`, reward **0** |
| ours (`reward[t]` is the reward of `a_t`) | reward **2** |

So the two conventions are separated by exactly one row in the data, and our env is
otherwise bitwise faithful to theirs. Note the second row of that table is also the
**missing env-parity test** the fixture module promises ("the fixture doubles as a bitwise
env-parity check", `fixtures/__init__.py:10-11`) and which `test_oracle_fixtures.py` does
not implement — see T4.

### D3 — `is_success`, `is_terminal` and therefore `cont` are indexed one row apart (NEW, ***FIXED 2026-09-07***)

***FIXED.*** `is_success` is now `reached_goal` on the observation the row carries, which is
their `MeasureSuccessRate`'s predicate on the transition it was handed. Verified: the
predicate applied to their `obs` column reproduces their `is_success` column with **zero**
mismatches over all 192 fixture episodes, so exactly one row of a successful episode carries
`cont = 0` on both sides. `Episode.cont`'s docstring no longer offers the false freeze
justification; it states the measurement that refutes it (`is_success`, `is_terminal` and
`has_succeeded` each sum to 1 on their live successful episode, because their driver resets
at `is_last` and the freeze is never recorded).

**Mechanism.** `MeasureSuccessRate` evaluates `state[0] ≥ 0.45` on the observation the
wrapper just produced (`raif/embodied/core/wrappers.py:659-660`), so `is_success[t]` is a predicate about
`obs[t]`. Our loop computes `success = reached_goal(stepped)` — the state *after* the step
— and records it beside `obs[t]`, which is the state *before*. `Episode.cont = 1 -
is_success` then inherits the shift: for the fixture's rows it equals their `1 -
is_terminal` exactly, and for our collected rows it marks the row whose observation is the
last **non**-goal state.

**The docstring's justification is factually wrong.** `Episode.cont` says the mapping is
safe because "the state freezes at the flag, so the predicate stays true down the padded
tail". Measured on their live run: `is_success sum = 1`, `is_terminal sum = 1`,
`has_succeeded sum = 1` for a successful episode. Their driver resets at `is_last`, so the
freeze is never observed and there is exactly one such row. The formula is right for the
fixture; the reason given for it is not, and the reason is what a reader would rely on when
porting a second task.

**Magnitude.** One row per successful episode, plus D8's amplification: after padding, 31
of 64 stored rows in a 34-step success carry `cont = 0`. Their windows carry one.

**Consequential?** Likely. `cont` gates the imagined rollout weight and the value
bootstrap, so ours truncates one state early relative to theirs. Taken together with D2 it
is *internally* consistent — our `(obs, reward, cont)` triple is uniformly the
"outgoing-transition" convention where theirs is the "incoming" one — which is why the
learned value function should be theirs shifted by one state rather than something wrong.
That argument does **not** survive D4.

### D4 — the two conventions are mixed inside one buffer (NEW, ***DISSOLVED 2026-09-07***)

***DISSOLVED, and verified rather than fixed.*** Nothing in `export_oracle_mc.py` or
`fixtures/__init__.py` changed: the collection loop moved to meet the export. With D1/D2/D3
adopted, a fixture episode and a collected episode are the *same* shape of thing — reset row,
terminal row, `reward[t]` caused by `a_{t−1}`, every flag a predicate on `obs[t]` — so no
conversion is needed anywhere and none was added. Checked three ways: the fixture's own
structure (row 0 at reward 0 with a real action, row `T` at `is_success`/`is_last` with a
zeroed action and `can_self_imitate = 0`) against the collected stream's, asserted in
`tests/test_run_mc.py`; the env-parity test above; and
`test_the_oracle_fixture_is_seeded_through_that_same_rule`, which reads the seeded streams'
lengths in steps. The probe batch, drawn from M⁺, is therefore no longer read on rows in a
foreign convention.

`export_oracle_mc.py` copies their replay columns verbatim (`export_oracle_mc.py:87-100`),
so every seed episode in M and M⁺ is in *their* convention — reset row, terminal row,
`is_success` aligned with `obs` — while every episode the agent collects is in ours. The
same gradient step therefore samples sequences under two incompatible indexings of
`reward`, `is_success` and `cont`. The "internally consistent, just shifted" defence of D2
and D3 fails exactly here.

**Magnitude.** 1,911 steps of their convention per seed (48 episodes) against 1M pooled
steps of ours, so ≈0.2% of M at the end of a run. But it is **100% of M⁺ at step 0**, and
M⁺ is what the CRSPP's positive gate and contrastive term are trained on; M⁺ then
accumulates only our-convention episodes on top. The probe batch
(`run_mc._probe_batch`) is drawn from M⁺, so `probe_ret`/`probe_value` are read on
their-convention rows throughout.

**Consequential?** Plausibly, for the CRSPP and for the early trajectory specifically —
which is the "off the floor by ~50k steps" landmark. Not measured.

### D5 — the recorded action is clipped on our side and raw on theirs (NEW, ***FIXED 2026-09-07***)

***FIXED.*** The clip left `habit_action` entirely — the draw comes back raw, as their
`ImagActorCritic.policy`'s does — and moved to the action's *consumers*, which is where
theirs lives: `jnp.clip` on the way into the env (their `ClipAction`) and
`controls.soft_clip` on the way into a transition (their `action_clip`, a stop-gradiented
rescale). Three sites: `run_mc.make_collect_fn`, `habit._from_module`'s internal rollout, and
`imagine.roll_batch` (training ledger 22). `System._habit_fn` no longer passes the box, and
`habit_action` no longer takes it. **Re-measured on our own actor**: 41% of draws leave the
box at initialization (16,000 draws at real posterior states, mean μ 0.63, mean σ 0.87), and
40.6% of *rollout* steps.

**Mechanism.** `ClipAction` (`wrappers.py:78-80`) builds a *new* dict for the env and
leaves the driver's `acts` untouched, so their replay stores the raw actor sample. Their
`obs_step` soft-clips it again before the GRU (`nets.py:134-136`), so the *world model*
sees a clipped action either way — which is what §(e)'s `rssm.action_clip` row asserts, and
that row is right about the GRU and silent about the recorded column. Our
`habit_action` clips the draw before returning it (`habit.py:169-170`), so the clip lands
in the `action` column itself.

**Magnitude — measured.** On their untrained actor over one 101-row episode: `max |a| =
2.271`, and **25.7% of recorded actions lie outside `[-1,1]`**. On our actor at
initialization the fraction is 49% (mean −0.918, sample sd 0.92). It does not vanish as the
policy sharpens, because their `tanh` mean approaches ±1 while σ has a 0.1 floor: at mean
0.9 with σ = 0.2 the fraction is 31%, and at mean 0.99 with σ = 0.1 it is 46%.

**Consequential?** Probably the most consequential of the new ones. The recorded action
column is read in two places where clipping is not idempotent: `start['action']` in the
actor-refresh / BC log-prob (`agent.py:477-479`), where a Gaussian log-density evaluated at
a clipped action is systematically different from one evaluated at the raw draw and where
a third to a half of the batch's targets pile up exactly on ±1; and the ensemble's
`ig_head`, which conditions on `action` (`ig_head.inputs: [deter, stoch, action]`). Both
are the sibling audit's lane to price; the *recording* is this one's.

### D6 — no transition step at `is_first`, and mean-field beliefs throughout

*ALREADY ON THE RECORD; measured here.* ***FIRST HALF FIXED 2026-09-07*** —
`run_mc.episode_start_beliefs` is `update_empirical_prior(initial_priors, 0)`, their
`obs_step`'s masked reset followed by `img_step`, hoisted out of the collection scan and used
both as the in-scan reset target and at `build`. The **second** half (the mean-field face:
their `get_stoch(deter_0) = dist.mode()` against our `.probs`) stays, as adjudicated — the
keyless-scoring-seam decision, unchanged by this round. Worth restating why the first half
was not merely cosmetic: it was also an **internal** train/act asymmetry, since
`RSSMCore.observe` runs `obs_step` at `t = 0` and so our training path always took the step
our acting path skipped.

Both halves were already recorded: the `is_first` GRU step in this audit's own brief, and
the mean-field-versus-sampled belief at `amortization.md` §(f), second bullet. The
executed trace confirms the order (`obs_step` → `_mask` → `initial` → `img_step` → `_gru`
→ prior sample → `obs_out` → posterior sample) and adds two numbers:

- our episode-start posterior `deter` equals the learned `deter_0` **exactly** (max
  |Δ| = 0.0), where one GRU step with `a = 0` moves it by `max |Δ| = 0.610` and `mean |Δ| =
  0.115` per unit on a carry bounded to `[-1,1]` by `tanh`. Because the GRU is a
  recursion, that offset does not decay within the episode — it displaces the whole
  episode's `deter` trajectory, which is the input the actor and every head read.
- their initial stoch is `get_stoch(deter_0) = dist.mode()` (`nets.py:203-207`) — an argmax
  one-hot — off the *same shared* `img_out`/`img_stats` weights our `initial_state` reads
  `.probs` from. So the two differ only in the reduction, as `initial_priors`' docstring
  claims.

***SECOND HALF CLOSED IN THE REFERENCE RECIPE 2026-09-28*** (Alex: "our failure mode is training on
samples but acting on mean-field? … I'd definitely propose fixing that and rerunning"). The
measurement is `actor-entropy-diagnostic.md` §8: in training both halves of the feature the actor
reads are samples (a straight-through one-hot code, and a GRU carry built from one-hots), at act
time both were mean-field (the probability vector, norm ≈ 1 against the one-hot's √32, and a carry
built from probability vectors); on ~1 key in 5 under their initializer the actor's extrapolation
onto that path is a saturated one-sided push from the first 1k steps, while the same actor on its
training features never saturates. Sampling the act-time code once and using the sample for the
actor, the logged entropy and the carry — their `obs_step` — lifts both trapped keys and leaves a
healthy key unchanged; three re-rolls of the exploration stream do not. Their code traps 0 of 20.
The engine gained the one face it lacked for this, `infer_states(key=)` — the recognition
face joining the prediction faces' rule that a key means one particle (`amortization.md`, as
built 2026-09-28) — and the lane acts on it: `Config.act_sampling` (default on;
`--no-act-sampling` keeps the old lane for the ablation; an old `run.json` without the knob
resumes as the old lane). The convention that absence of a key means expected is unchanged. Still
open on this row: the episode-start belief's code is the prior's probability vector where theirs
takes the prior's `mode()` — one step's GRU input, unmeasured.

### D7 — the habit rolls a whole 8-step plan at every acting step and discards seven of it (NEW, ***RECORDED, not fixed***)

***RECORDED 2026-09-07 as cosmetic-and-not-fixed.*** Behaviourally equivalent — the first
draw uses `split(plan_key)[1]`, which does not depend on `policy_len`, and the extra rollout
consumes only the dedicated `plan_key` — so the action taken is the same quantity theirs is.
It is ~10× the neural work per acting step in a loop that is 1.4% of the cycle, i.e. ≈13% of
the cycle at most, and a `policy_len=1` acting path is a change to what the habit planner
*is* rather than a one-line setting. Left as measured cost, not parity debt.

**Mechanism.** `habit._from_module` (`habit.py:106-124`) exists to turn a policy *module*
into a *plan*, so it loops `ctx.policy_len` times: draw an action, push the beliefs through
the transition, draw again. `System.act` then takes `policies[0][0]`. Their
`ImagActorCritic.policy` (`agent.py:401-402`) is one actor evaluation and no rollout.

**Magnitude — measured.** Per `act()` call, with instrumented counters: **8** actor
evaluations and **16** `amortized.predict` applications, against their 1 and 1. (16 rather
than 8 because the CRSPP row's `needs_prev` wrapping advances a paired carry.)

**Consequential?** Not for the trajectory: the first draw uses `split(plan_key)[1]`, which
does not depend on `policy_len`, and the extra rollout consumes only the dedicated
`plan_key`, so the action taken is the same quantity theirs is. It is a pure cost — and it
means the "collection is 1.4% of the cycle" figure in `run_mc`'s docstring is measuring a
collection loop doing roughly an order of magnitude more neural work per agent step than
parity requires. Recorded because "how many times does this operation run" is exactly the
class this audit is for, and because a one-line `policy_len=1` path would make collection
nearly free.

### D8 — fixed-length windows truncate the failure signal and inflate the terminal one

*BOUNDED BY WORK ALREADY IN FLIGHT (`flag7-flat-replay`).* ***FIXED 2026-09-07*** — that
work is this branch, and this round sits on top of it. `StreamBuffer` splices whole episodes
onto a flat step-addressed ring and `sample_windows` draws `seq_len` contiguous steps from
anywhere in it, with an `is_first` column that `RSSMCore.observe` consumes as their
`obs_step`'s mask. Nothing is padded, nothing is truncated: ρ's failure tail is sampled and
the terminal row appears once per episode rather than 31 times per stored window. Two
numbers become exact rather than approximate — capacity is their `replay_size` 5e5 **steps**
verbatim, and one update sees `batch_size · seq_len` = 1024 **real** transitions, their
`batch_steps`, against ≈663 before.

Their `Uniform` replay is a flat step-addressed stream: a 64-step training window may
straddle an episode boundary, and `is_first` inside it resets the recurrent state. Our
`StaticBuffer` stores episode-aligned windows, so `fit_length` (`replay.py:235-270`) either
truncates to the first 64 rows or pads by freezing the last row. This is the
`flag7-flat-replay` item, already settled and in a worktree; the audit's contribution is
the magnitude, which is larger than the FLAG in `run_mc`'s docstring suggests.

**Measured, on a 100-step failure episode.** Its full `pref_label` runs to −1 at the last
step (`ρ[-4:] = [-0.027, -0.090, -0.300, -1.000]`), because the failure decay is
`β^(T-1-t)` with β = 0.3. Stored as its first 64 rows, the window's labels are
`min = -1.5e-19`, `max = 0`. **The entire failure branch of the self-revision mechanism
lives in the discarded tail.** Since a timeout episode is 100 steps and every early
episode is a timeout, the CRSPP's negative signal is effectively absent from our M for as
long as episodes exceed 64 steps — which is most of the run's first phase. Theirs samples
windows that cover the tail and sees ρ down to −1.

**Measured, on a 34-step success padded to 64.** 31 of 64 rows carry `cont = 0` and
`is_success = True` (the frozen final row, repeated), and 30 of 64 carry `reward` exactly
0 — and in this env `reward = 0` is the goal's own reward code. Theirs carries one
terminal row and one zero-reward row (the reset) per 101.

**Consequential?** Yes, and this is the most quantitatively striking result in the audit.
Reported here so that the flat-replay work has a number attached to it, not as a new
action item.

### D9 — R-S's per-episode cumulative reward is accumulated differently (NEW, ***FIXED 2026-09-07***)

***FIXED.*** `eval.stability_c` is their accumulator: `Σ_{t<T} (reward[t] + offset)` over a
`T+1`-row episode — every reward but the closing row's, and one offset for the reset row.
`run_mc.train` logs it per episode as `scores.jsonl`'s `stability` field, because their `C`
is a separate wrapper channel and is **not** recoverable from a return and a length;
`seed_report` reads that field and refuses to fold a log that mixes it with pre-2026-09-07
rows. `eval.evaluate` folds it per row of its `rewards` argument. The FLAGGED note about a
negative denominator is **retired**: theirs is `mean(abs((rolling − max) / max))` with the
`abs` around the whole ratio, algebraically identical to ours for either sign of `best`, so
there was nothing there to close — the real divergence was the accumulation, which the FLAG
did not mention.

**Mechanism.** `MeasureRStability.__call__` (`wrappers.py:754-763`) is an if/else: on a row
with `is_last or is_terminal` it *closes* the accumulator and adds nothing, and otherwise
it adds `reward + 1.1`. So their `C` for an episode is the sum of `reward + offset` over
every row **except the last**, and it includes the reset row's `reward = 0`. Our
`eval.episode_returns` sums `reward + offset` over every row we recorded. Separately,
theirs is one accumulator **per env worker** and the value logged is whichever worker
finished the episode; ours pools the lanes.

**Magnitude.** The offset count is the same on both sides (their `T+1` rows minus the
closing row equals our `T`), so the difference is exactly the reward of the row theirs
excludes and ours includes: **1.0 per successful episode and 2.0 per timeout**, against a
`C` of order −32 (success) to −90 (timeout) — roughly 2%. R-S is a ratio of differences,
so part of that cancels; I would expect an effect in the second decimal of a number their
Table 2 reports as `0.1 ± 0.0`, which is enough to matter for a comparison at that
precision, but I have not computed it end to end.

**One thing this audit can close rather than open.** `eval.relative_stability`'s FLAGGED
note worries that their formula divides by a negative `max(C)` and would return a negative
R-S. It would not: theirs is `np.mean(np.abs((rolling - max) / max))`, with the `abs`
around the *whole ratio*, which is algebraically identical to our `np.mean(np.abs(tail -
best) / np.abs(best))` for either sign of `best`. The FLAG can be retired; the divergence
that actually exists is the `C` accumulation above, which the FLAG does not mention. *(Both
done, 2026-09-07: the FLAG is gone from `eval.relative_stability`'s docstring, replaced by
the algebra that settles it, and the accumulation is `eval.stability_c`.)*

---

## NEEDS-TRANSCRIPTION-TEST

A trace shows that a function was called with arrays of the right shape; it cannot show
that the arithmetic inside it is theirs. These four are the places on the acting path where
that gap is load-bearing.

**T1 — `MeasureRStability.relative_stability` (`wrappers.py:742-751`).** Four choices are
entangled: `np.convolve(C, ones(100)/100, mode='valid')` (which returns an **empty** array
below 100 episodes, so their `relative_stability` is `nan` there while ours shrinks the
kernel and returns a real number), `max` taken over the **whole** history rather than the
window, the `[-min(last_k, len)]` tail slice, and the `abs`-around-the-ratio settled above.
`test_raif_metrics.py` tests our re-expression against itself; nothing tests it against
theirs. A test that reproduces their number from a recorded episode list would pin all
four at once and would have caught D9.

**T2 — `MeasureSuccessRate`'s windowed rate (`wrappers.py:693-722`).** The append-once
semantics (1.0 at the *first* success step, 0.0 at `is_last` only if never successful, and
the deliberate absence of a second append when both hold on the same row) is a small state
machine we do not implement at all. If SR is ever read off their channel rather than
recomputed from the score log, it needs a test.

**T3 — the promised env-parity test does not exist.** ***WRITTEN 2026-09-07:***
`tests/test_oracle_fixtures.py::test_our_env_reproduces_their_recorded_observations_and_rewards`,
parametrized over all four fixtures — obs to `1.3e-06`, reward exactly 0, `is_success`
reproduced from the `obs` column with zero mismatches, and the mis-aligned reward error of
exactly 2 asserted as a negative control. `fixtures/__init__.py`'s promise now names it.
The original finding follows.

**T3 (as found).** `fixtures/__init__.py:10-11` says the
fixture "doubles as a bitwise env-parity check (their recorded rewards must reproduce under
our env stepped with their actions)", and `test_oracle_fixtures.py` only checks loading and
provenance. I ran the check by hand for this audit (D2's table) and it passes at `8.9e-07`
on observations and exactly 0 on rewards — *under their index convention*. It belongs in
the suite, and it is the one test that would fail loudly if D2/D3 were ever "fixed" in only
one of the two places. It also subsumes the one composition this audit checked by hand
rather than by test: `ActionRepeat`'s `break` on `is_last or is_terminal`
(`wrappers.py:59-69`) against our masked unrolled pair, whose three reward cases the
fixture replay exercises directly.

## What this audit did not cover, and why

- **Everything inside the losses.** The world-model loss, `imagine`, actor/critic,
  ensemble/info-gain, CRSPP and self-revision *training*, and the optimizer application
  are the sibling audit's lane and were deliberately left alone. Where an acting-path
  divergence lands in one of them (D5 in the BC log-prob and the `ig_head`; D3 in the
  rollout weight) the row says so and stops.
- **The image branch.** Their env renders a 64×64 frame every step and stores it in the
  replay; `cnn_keys: '^$'` means nothing reads it. Dispositioned N-A, not investigated.
- **Their `Parallel`/`process` env workers.** Traced with `envs.parallel=none` in one
  process; the lockstep claim is read off `BatchEnv.step`, which is the same code either
  way, and `Parallel` only moves where `env.step` executes.
- **Checkpoint/resume paths** on either side.
- **RNG stream equivalence.** Not attempted and not attemptable: their single numpy
  generator interleaves the acting and training streams (C21), and two of their `nj.rng()`
  draws per `img_step` are consumed by an inert dropout mask (C9). Recorded so that nobody
  later mistakes a stream mismatch for a logic error.

## Addendum (2026-09-30): D8's replay, term for term

*From the engine's former `train/replay.py` docstrings (the 2026-09-21 salvage,
now an index). D8 records the capacity and the per-update step count. This is
the rest of the correspondence, which was recorded nowhere else.*

- **Their `Uniform` replay** (`raif/embodied/replay/generic.py`,
  `replays.py::Uniform`): one `deque` per env worker. An item is created at
  *every* step once the stream is `length` long (`stride = 1`, since `overlap`
  is unset), with FIFO removal over items, uniform sampling over items, and
  `seq['is_first'][0] = True` forced on every draw (`generic.py::_sample`).
  Uniform over stride-1 items is uniform over window start offsets, so
  `sample_windows` draws a start offset uniformly, with replacement, and forces
  step 0's flag the same way. `StreamBuffer.windows` is their `len(replay)`,
  `size − seq_len + 1` floored at 0, and `StreamBuffer.ready` is their prefill
  and fallback test, `len(replay)` against `batch_steps` (`run/train.py:136,160`).
- **One stream, not one per worker.** Theirs keys the `deque` by worker. Ours is
  a single ring per buffer, with episodes appended in completion order. The two
  are equivalent because a splice point is a hard reset: at `is_first` the carry
  is replaced and the action zeroed, so which episode neighbours which cannot be
  observed. Eviction is where they differ, and it is the reason for one ring:
  theirs is a single global FIFO over items (`selectors.Fifo`), which one ring
  reproduces and per-lane rings would not. Their positive buffer already pools
  every worker into stream 0 (`callbacks.py::add_trajectory` calls
  `positive_replay.add` with no `worker=`).

*Extended 2026-09-30 from the docstring paring, which cut these from
`reference/raif/run.py`'s module docstring. D8 gives the 34-step and 100-step
examples and ≈663 real steps; these are the run-scale numbers.*

- **What the padding was, over real runs.** Under episode-aligned windows,
  `fit_length` padded a short episode by freezing its final step. At
  `batch_length 64` a converged Mountain Car episode is ≈38–45 agent steps, so
  30–41% of every training window (≈35% on average, 664 real steps of 1024)
  was a frozen copy of one state. On the oracle's own `collect_prior` fixture,
  whose 48 episodes run 39–41 steps, it measured 37.8%. The fraction *rose* as
  the agent improved, because success is what makes an episode short. Over the
  first 2M-step run the padded fraction was ≈40% while the agent performed
  well and ≈0% while it failed. A window's padding fraction correlated −0.24
  with the next window's change in return (−0.28 / −0.29 / −0.24 / −0.29 per
  seed). That is not waste but bias, pointed the wrong way. The engine change
  that removed it is one optional argument, `RSSMCore.observe`'s `is_first`,
  which resets the recurrent carry and zeroes the incoming action where a
  window crosses into a new episode, exactly their `obs_step`'s mask.
- **Rejected: masking the loss on padded steps.** Recorded so it is not
  re-proposed. It removes the bias but not the arithmetic: the update would
  still see only the ≈660 real steps the window happened to contain, against
  their 1024. Every learning rate and loss scale in the P0 addendum's §5 would
  then be off by a factor that moves with the agent's competence.
- **Capacity, as an identity.** `CAPACITY` is their `replay_size` verbatim,
  500,000 steps per buffer. Their capacity counts stride-1 items and one item
  is created per step, so `N` items retain ≈`N` distinct steps. The previous
  regime had to settle for `5e5 / seq_len` = 7812 episode-aligned sequences,
  which held ≈7,800 episodes where their 5e5 flat steps hold ≈12,500.
