# Execution-trace ledger — the TRAINING path

**Status: AUDITED 2026-09-06, ADJUDICATED AND RESOLVED 2026-09-07.** Every row was a
reading of two sources; the DIVERGES rows were *reported for adjudication* in the sense
`amortization.md` §(f) uses, and the numbering of new divergences continues that section's
(which reached 19). They have now been ruled on, and the consolidated parity round acted on
them in the `flag7-flat-replay` branch. §10's headings carry the disposition.

**Resolution summary (2026-09-07).** Of the nine new divergences:

| # | what it was | disposition |
|---|---|---|
| 20 | the info-gain leaf read one rollout step earlier than their call site reads it | **ARGUED — KEEP OURS** (Alex), and **measured**: §10's entry carries the numbers |
| 21 | the shared forward pass recomputed post-step instead of reused | **RECORDED 2026-09-07; CLOSED 2026-09-25** — `fit_world`/`fit_crspp` hand back their pass's states and `train_step` reads them (§21) |
| 22 | `jnp.clip` on the imagined action zeroed a gradient their rescale keeps | **FIXED** — `controls.soft_clip` in `imagine.roll_batch`, and **measured** |
| 22b | the BC target was the clipped recorded action | **FIXED** upstream — the recorded action is now the raw draw (acting ledger D5) |
| 23 | their critic target is a second `score` sample | **RECORDED, cosmetic** — inert against our noise-free leaf |
| 24 | their actor runs `H+1` times, ours `H` | **RECORDED, cosmetic** — the ninth action's only consumer is 20's index, resolved our way |
| 25 | `loss_scales.actor` has no counterpart | **RECORDED, cosmetic** — inert at its resolved 1.0 |
| 26 | normed hidden layers keep a bias outside `MLPConditional` | **OPEN, MEASURED TWICE** — alone *and* paired with 17; the pair loses to the persistence baseline on 4/4 seeds and was reverted |
| 26b | their `Dist` output linears are drawn uniform, not normal | **FIXED** — `neural.uniform_avg`, and the test that asserted the false fact is corrected |
| 27 | `SymlogDist`'s `tol` is not transcribed | **RECORDED, cosmetic** |
| 28 | the cosine row's rollout length is the System's, not `imagine`'s | **RECORDED, inert at parity** |

The **method finding** both audits converged on — a transcription test pins a formula and
silently inherits its author's assumptions about where it is called from — is written up in
`amortization.md` §(a″) as a dated subsection, with the two ports that had the defect (20's
harness and `their_dist_normal`'s) and the two replacements built from the call site.

Ours at `044ce7b`; theirs at `raif-paper` fork commit `9b1864e8`, which is the oracle. The
resolved config is `src/hybrid_aif/reference/fixtures/oracle_state_mc.config.yaml`, and
every branch below is resolved against it rather than against `configs.yaml`'s inheritance
chain.

**Method: their code was EXECUTED, not only read.** A copy of `raif/` in a scratch directory
was driven through **two consecutive** `JAXAgent.train` calls on a synthetic `16×64` batch at
the oracle's own resolved sizes with `jax.jit` off, under a `sys.settrace` hook that recorded
every call and return in their tree with argument shapes and every `nj.rng()` draw. Nothing in
either repository was modified. The config is validated: the four `Optimizer X has N variables`
prints reproduce the oracle log **exactly** (513,314 / 422,528 / 428,034 / 46,431), and nothing
was shrunk — `batch_size 16`, `batch_length 64`, `deter 128`, `actor.units 256`,
`imag_horizon 8`, `ig_models 4` are all theirs.

Three departures, stated because they bound what the trace can settle. `precision` was
`float32` rather than `float16`, so their loss scaler and `apply_if_finite` were not exercised
(row 3.7, which is why that row is N/A). The batch was synthetic rather than replayed, so
`populate_data`'s three zero-fill branches ran where a real batch supplies those columns
(row 1.9). And `jax_transfer_guard='disallow'` had to be neutralized, because under eager mode
every host-to-device transfer of a Python constant raises — a debug guard, no numerics.
Ordering, arity and shape claims below are therefore *measured*; formula-level claims are
still readings, which is what §12 is for.

**What this audit looks at, and why it is a fourth one.** The three standing audits catch
wrong *values* (the resolved-config table, `amortization.md` §(e)), wrong *formulas inside a
chosen function* (`tests/test_oracle_transcription.py`, `tests/test_amortized_objective.py`)
and aggregate numeric *drift* (the cross-run log comparison). None of them looks at control
flow: which operations run, in what order, how many times, and on which inputs. Every
surprise of the last week — the dead `i % 2` alternation, `means.std(0)` standing in for
`preds.std(0)`, the missing act-time GRU step — was of that class. So the surface walked
here is the *sequence*, and the deliverable is an exhausted list rather than a sampled one:
one row per operation of one full training step, each with a disposition.

**Scope.** One `agent.train(data, state)` call, end to end: batch formation and carry
threading, the world-model loss and its components in computation order, the CRSPP loss, the
info-gain ensemble trainer, the imagination rollout, the per-step objective, λ-returns and
return normalization, the actor loss, the critic loss and slow-critic sync, and each
optimizer application. Out of scope (a sibling audit's): the environment and its wrappers,
act-time encoding, the planner call and action sampling, episode/replay writing, per-episode
callbacks, buffer routing, counters. Self-revision is in scope because the CRSPP consumes it.

**Dispositions.** **MATCHED** (the operation, its order, its inputs and its arity agree),
**DIVERGES** (they do not — mechanism, magnitude and consequence in §10), **NTT**
(NEEDS-TRANSCRIPTION-TEST — a difference here would be absorbed into the mathematics and no
existing test pins it, so it is recorded as an open test rather than guessed at), **N/A** (no
analog, with the reason).

**Counts.** 148 rows: **97 MATCHED**, **25 DIVERGES**, **9 NTT**, **17 N/A**. Of the 25
DIVERGES rows, 11 restate divergences already on the record (flag 7, 9, 17, 18, the CRSPP
gradient-scope and mean-field findings, the adjudicated `mixsigma`) and 14 carry **nine new
divergence numbers**, 20 and 21–28. One of the nine (20, the info-gain leaf's index) is
ranked plausibly consequential; three (21, 22, 24) are structural and unmeasured; five are
cosmetic or inert at the resolved config.

*After the 2026-09-07 round:* 20 is ARGUED and measured, 22/22b/26b are FIXED, 26 is open
with a measurement and a stated coupling, and 21/23/24/25/27/28 are recorded as cosmetic or
deliberate. Flag 7 is closed (rows 1.3, 1.11, 2.4). Rows 1.5, 2.22, 3.1, 3.2, 4.1, 4.13,
4.14, 5.1, 6.2, 6.3, 6.5 and 6.23 stand as before — the CRSPP gradient-scope seam, the
cross-call carry, the pre-step-aux reuse and the mean-field/keyless faces, all previously
adjudicated and untouched by this round.

---

## 1. The outer loop and batch formation

| # | operation (theirs) | ours | disp. | note |
|---|---|---|---|---|
| 1.1 | `train.py:158` `for i in range(should_train(step))` | `run_mc.make_chunk_fn`'s inner scan of `train_every // lanes` | MATCHED | one gradient step per 16 pooled agent steps; their `Ratio(64/1024)` returns only 0 or 1, so the loop body runs at most once |
| 1.2 | `train.py:160-167` the `i % 2` M/M⁺ alternation | `Config.alternate = False` | MATCHED | adjudicated 2026-09-02 (divergence 18): the branch is unreachable, the oracle trained every world-model batch on `M`, and ours does too |
| 1.3 | `train.py:162` `next(dataset)` — a random `batch_length` window of the flat stream | `replay.sample_windows` over `StreamBuffer`, a flat step ring | **flag 7, CLOSED** | the `flag7-flat-replay` branch this round sits on. Consequence for this audit is row 1.5, which stands |
| 1.4 | `jaxagent.py:69` `rng = self._next_rngs(...)` — one host-drawn `int64` seed per train call | `k_train` from the seed's own `train_key` split | MATCHED | one fresh key per train step, threaded, not re-derived |
| 1.5 | `jaxagent.py:71` `state = None` ⇒ `_init_train`; otherwise the carry from the previous call | no carry: `observe` starts from `initial_state()` every step | DIVERGES (recorded) | **measured over two consecutive calls**: call 1 enters `None`, runs `train_initial`, and returns `({deter f32[16,128], logit, stoch}, f32[16,1])` = `(post[:, -1], data['action'][:, -1])`; call 2 enters exactly that object and skips the `train_initial`/`rssm.initial` prologue, the rest of the graph being identical (456 rng draws both times). In their run that carry comes from an unrelated random window, since replay draws are uncorrelated. Downstream of flag 7; reported, not acted on |
| 1.6 | `agent.py:83` two carries `state[0]`/`state[1]`, one per buffer | one | N/A | dead with the alternation (row 1.2) |
| 1.7 | `agent.py:81` `preprocess`: cast to f32, `cont = 1 − is_terminal` | `Episode.cont` recorded as `1 − done` | MATCHED | the binary head's target is a *continuation*, not a terminal flag |
| 1.8 | `agent.py:111-112` uint8 image branch `/255` | — | N/A | `cnn_keys: ^$`; this is the state-vector run |
| 1.9 | `agent.py:82,119-128` `populate_data` zero-fills `can_self_imitate`, `pref_label`, `unrelated_action` | `Episode` always carries the first two | MATCHED | `unrelated_action` is written and never read in their tree |
| 1.10 | `agent.py:193-194` `prev_actions = concat([prev_action[:, None], action[:, :-1]], 1)` | `Episode.prev_actions()` = `concat([0, action[:, :-1]])` | MATCHED | same shift; the leading slot differs only through row 1.5, and their `obs_step` zeroes it anyway wherever `is_first` |
| 1.11 | `nets.py:137-141` per-step `is_first` mask and reset to `initial(...)` | `RSSMCore.observe`'s optional per-step `is_first`: the carry is replaced by the learned initial and the incoming action zeroed | MATCHED | their mask exactly, and now exercised rather than merely unnecessary: flat-stream windows cross episode boundaries (flag 7, closed) |
| 1.12 | `train.py:169` `metrics.add(...)`, `updates.increment()` | `StepLosses` returned | N/A | logging |
| 1.13 | `train.py:173` `agent.sync()` every `sync_every: 10` | — | N/A | `jaxagent.sync` returns immediately on a single device |
| 1.14 | `train.py:179` `agent.report(batch)` every `log_every: 300` s | `run_mc` diagnostics | N/A | not part of a train step; it does consume RNG and run a second world-loss forward pass on their side |
| 1.15 | `jaxagent.py:78` `_should_metrics = Every(10)` gating metric transfer | — | N/A | logging cadence |

## 2. `WorldModel.loss` — the order of operations

| # | operation (theirs) | ours | disp. | note |
|---|---|---|---|---|
| 2.1 | `agent.py:191` `embed = self.encoder(data)` — one call, `(B, T)` flattened | `core.observe` calls `core.encoder` per step inside the scan | MATCHED | same function, same inputs; theirs batches the call, ours vmaps and scans. `symlog_inputs: true` is on the net in both |
| 2.2 | `agent.py:195-196` `post, prior = rssm.observe(embed, prev_actions, is_first, prev_latent)` | `RSSMCore.observe` (`lax.scan` of `obs_step`) | MATCHED | one scan, posterior carried forward, `T = 64` steps |
| 2.3 | `nets.py:142` inside `obs_step`: `prior = img_step(prev_state, prev_action)` **first** | `obs_step` runs the GRU then the prior then the encoder then the posterior | MATCHED | the prior's `deter` *is* the posterior's `deter`; both build it before reading either head |
| 2.4 | `nets.py:134-136` `action_clip: 1.0` on the replay action | `soft_clip` in `to_world_batch` | **MATCHED, by the same expression (2026-09-07)** | the old "MATCHED (inert)" reading rested on our recorded action already being clipped, and D5 removed that premise: the column is now the **raw** draw, so their rescale has work to do on our data and we do it. It is applied where the *recurrent scan* consumes the action and nowhere else — the ensemble is handed `batch.prev_actions()` raw, because their `inputs(data)` reads the replay column itself (row 5.2), and the BC term reads the raw column too (row 6.39). A gap this round *opened* by making the column raw and then closed; recorded that way because the reasoning — which reader gets the raw action and which the rescaled one — is the part a later change could get wrong |
| 2.4b | `nets.py:139-141` `obs_step` calls `self.initial(len(is_first))` at **every** timestep for the reset term | no per-step initial call | MATCHED (inert) | **measured**: 64 `initial`/`get_stoch` calls per `observe` and 64 more per `observe_no_action`, a real forward pass through `img_out`/`img_stats` each time. The result is masked by `is_first`, so it contributes nothing (and zero gradient) except at `t=0` |
| 2.5 | `nets.py:180` `img_in` = `Linear(units=32, silu, layer)` over `concat(flat stoch, action)` | `GRUCell.input_proj` + norm + act | MATCHED | 1025 → 32 |
| 2.6 | `nets.py:180,182` `dropout=True` on `img_in`/`img_out` at `dropout: 0.0` | no dropout | MATCHED (inert) | their mask is all-ones at rate 0; it does consume one `nj.rng()` draw per call (**measured**: two draws per `img_step`, before `_gru` and before `_stats`), an RNG-stream difference only |
| 2.7 | `nets.py:209-218` `_gru`: `Linear([deter, x] → 3·deter, act none, norm layer)`, then `tanh(σ(r)·c)` and `σ(u − 1)` | `GRUCell.__call__` | NTT | verified by reading, line for line, including the *joint* LayerNorm over the 3·deter vector before the split and the `−1` update-gate offset. No port test exists |
| 2.8 | `nets.py:182` `img_out` = `Linear(32, silu, layer)`; `nets.py:243` `img_stats` = `Linear(1024)` | `MLPCategorical.from_shapes(depth=1, width=32)` | MATCHED | one normed+activated hidden layer, then the logit linear |
| 2.9 | `nets.py:235-239` unimix: `probs = (1−u)·softmax + u/K`, `logit = log(probs)` | `neural.unimix_logits` | NTT | identical, including that it returns *logits of the mixed probabilities* rather than mixing logits |
| 2.10 | `nets.py:144-147` `obs_out` over `concat(prior['deter'], embed)`, `obs_stats`, sample | `posterior.logits((deter, embed))` then `straight_through_onehot` | MATCHED | parent order (carry, embedding) enforced by `from_declarations` |
| 2.11 | `jaxutils.py:105-108` `OneHotDist.sample` = `sg(hard) + (probs − sg(probs))` | `neural.straight_through_onehot` | MATCHED | same estimator |
| 2.12 | `agent.py:197-198` `C_post, C_prior = rssm.observe_no_action(embed, is_first, prev_latent)` | `crspp.observe_no_action`, called separately in `fit_crspp` and again in `crspp_recognize` (until 2026-09-25: `fit_crspp` now hands back its pass's posterior, over the pre-step encoder's embeddings) | DIVERGES (recorded) → CLOSED 2026-09-25 with 21 | recorded as the CRSPP gradient-scope finding: theirs is one scan inside the world loss over the *shared* `embed`; see rows 4.x and divergence 21 |
| 2.13 | `agent.py:202` `feats = {**post, 'embed': embed}` | `core._features` = `(deter, flat stoch)` | MATCHED | no head declares `embed` as an input in the resolved config |
| 2.14 | `agent.py:203-207` head loop over `{decoder, reward, cont}` with `grad_heads` all three | three NLLs in `world_loss` | MATCHED | no `sg(feats)`: all three differentiate through the states |
| 2.15 | `agent.py:212` `losses['dyn'] = dyn_loss(post, prior, kl, free=1.0)` | `kl_balance(...).dyn` | MATCHED | transcription-tested (§8) |
| 2.16 | `agent.py:213` `losses['rep'] = rep_loss(post, prior, kl, free=1.0)` | `kl_balance(...).rep` | MATCHED | transcription-tested; free bits applied *per direction, before* the 0.5/0.1 mix |
| 2.17 | `agent.py:215` `-dist.log_prob(data['state'])` via `SymlogDist` | `symlog_mse_nll` | NTT | the formula matches; theirs additionally zeroes squared errors below `tol = 1e-8` (`jaxutils.py:164`), which ours does not. **27, RECORDED (cosmetic)**, still untested |
| 2.18 | `agent.py:215` `-DiscDist.log_prob(data['reward'])` | `symlog_disc_nll` | MATCHED | transcription-tested against their two-hot block including the out-of-range clip |
| 2.19 | `agent.py:215` `-Independent(Bernoulli(logit), 0).log_prob(data['cont'])` | `binary_nll` | NTT | `softplus(z) − y·z` is the same number; no port test exists |
| 2.20 | `agent.py:218-219` `scaled = {k: v·scales[k]}`, `model_loss = sum(...)` | `world_loss`'s weighted sum | MATCHED | `loss_scales`: state/reward/cont 1.0, dyn 0.5, rep 0.1 |
| 2.21 | `agent.py:241` `model_loss.mean()` — over batch **and** time | `jnp.mean` over the vmapped per-step losses | MATCHED | |
| 2.22 | `agent.py:236-238` `state = (post[:, -1], action[:, -1])` returned as the next call's carry | not produced | DIVERGES (recorded) | row 1.5 |
| 2.23 | `agent.py:239` `_metrics(...)` | `WorldLosses` | N/A | logging; theirs draws RNG inside `subsample` |

## 3. Optimizer application — the world model

| # | operation (theirs) | ours | disp. | note |
|---|---|---|---|---|
| 3.1 | `agent.py:182` `modules = [encoder, rssm, *heads]` | `fit_world`'s `PART_FIELDS` trainable tuple | DIVERGES (recorded) | **measured**: the trace shows `Optimizer.__call__(modules=[MultiEncoder, CRSPP_RSSM, MultiDecoder, MLP, MLP])`, and `self.rssm` *contains* the CRSPP layers (`img_in_2`, `gru_2`, `img_out_2`, `img_stats_2`, `obs_out_2`, `obs_stats_2`), so theirs is one gradient, one clip and one adam over the world model **and** the preference model together. Recorded as the CRSPP gradient-scope finding |
| 3.2 | `jaxutils.py:409` `nj.grad(wrapped, modules, has_aux=True)` — one backward pass over `model_loss.mean() + C_model_loss_mean` | two `eqx.filter_grad` calls (`fit_world`, `fit_crspp`) | DIVERGES (recorded) | consequence: their encoder receives gradient from the CRSPP's KL and contrast terms; ours receives none (ledger entry 5's deliberate seam) |
| 3.3 | `jaxutils.py:373` `optax.clip_by_global_norm(1000.0)` **first** in the chain | `optax.clip_by_global_norm(DEFAULT_CLIP=1000.0)` before the per-slot steps | MATCHED | **measured**: the chain is `chain.update_fn → clip_by_global_norm.update_fn → scale_by_adam.update_fn → scale.update_fn(−lr)`, and no `init_fn` runs during a train call (every slot was created at `_init_varibs`). One global norm over the whole differentiated set, before adam — with the set differing per row 3.1 |
| 3.4 | `jaxutils.py:376` `optax.scale_by_adam(eps=1e-8)` | `neural.adam(1e-8)` ⇒ `optax.adam(1.0, eps)` | MATCHED | `b1/b2/eps_root` at optax defaults on both sides |
| 3.5 | `jaxutils.py:388` `optax.scale(-lr)`, `lr = 1e-4` | `apply_gradient`'s `lr * updates` after `optax.adam(1.0)` | MATCHED | same composition: clip → adam → scale by −lr |
| 3.6 | `jaxutils.py:379,382,385` `lateclip`/`wd`/`warmup` all falsy ⇒ no transform added | none | MATCHED | the three zeros mean the chain has three fewer links, not three no-ops |
| 3.7 | `jaxutils.py:391-397` `scaling = (COMPUTE_DTYPE == float16)` ⇒ loss ×1e4, grads ÷1e4, `optax.apply_if_finite(max_consecutive_errors=1000)` | none | N/A | downstream of ARGUED divergence 9 (`jax.precision: float16` vs our float32). At float32 the scaler and the finite-guard have nothing to do; recorded so that "their optimizer has three moving parts ours lacks" is on the record rather than rediscovered |
| 3.8 | `jaxutils.py:429` `step += isfinite(norm)` | no step counter in the optimizer | N/A | their counter feeds `warmup` (0 here) and metrics only |
| 3.9 | `jaxutils.py:422-425` one optax state over the whole parameter dict | one `LearnState` entry per slot | MATCHED | adam is elementwise, so per-slot states are arithmetically the one state. The clip is the only non-elementwise part, and row 3.3 covers it |
| 3.10 | — | `fit_world`'s static-zero `lr` slot skip | N/A | our rate grammar; at parity every rate is `1e-4` |

## 4. The CRSPP's half of the world loss

| # | operation (theirs) | ours | disp. | note |
|---|---|---|---|---|
| 4.1 | `nets.py:93` `state=None ⇒ initial_all_zeros`, but `agent.py:198` passes `prev_latent` | `parts.initial()` + `prior.mode` — the CRSPP's *own* learned start | DIVERGES (recorded) | **measured**: the trace shows `observe_no_action(embed, is_first, state={deter:f32[16,128], logit:…, stoch:…})` — the world model's latent, not zeros. Theirs has no CRSPP-specific initial: at `is_first` it resets to the **world model's** learned `initial` and `get_stoch`, so the CRSPP's episode start is the world model's and its gradient flows into `img_out`/`img_stats` |
| 4.2 | `nets.py:194-196` `img_in_2 → gru_2 → img_out_2` with no action parent | `img_step_no_action` / `observe_no_action`'s `parts.gru(deter, code)` | MATCHED | the action's absence is the model's whole content |
| 4.3 | `nets.py:194,196` no `dropout=True` on `img_in_2`/`img_out_2` | none | MATCHED | **measured**: `img_step_no_action` draws **one** `nj.rng()` (the stoch sample) against `img_step`'s three (two dropout masks and the sample). The action-free branch really does differ from the action branch in their code |
| 4.4 | `nets.py:160-163` `obs_out_2`, `obs_stats_2`, sampled posterior code | `parts.posterior.logits((deter, embed))` + straight-through sample | MATCHED | separate parameters from the world model's posterior, same architecture |
| 4.5 | `agent.py:223-224` `dyn_loss(C_post, C_prior)`, `rep_loss(C_post, C_prior)`, `free=1.0` | `crspp_loss`'s `kl_balance` | MATCHED | same balanced KL, same free bits |
| 4.6 | `agent.py:225` `positive_kl_mask = sign(pref_label).clip(0, 1)` | `revise.positive_gate` = `max(0, sign(ρ))` | MATCHED | a hard mask, not a weight |
| 4.7 | `agent.py:226-228` mask folded into the dyn/rep **scales**, then summed | `gate * kl.total` | MATCHED | `mask·(0.5·dyn + 0.1·rep)` either way |
| 4.8 | `agent.py:229` `similarity = cosine_sim(C_post, sg(post))` | `sim` against `sg(agent_features)` | MATCHED | the preference model moves toward the agent, never the reverse |
| 4.9 | `agent.py:165-179` `cosine_sim`'s guards: `sum(x²).clip(1e-8)` per side, product `.clip(1e-8)` | `max(‖a‖·‖b‖, 1e-12)` | NTT | different guard, inert at these norms (O(√1152)); no port test |
| 4.10 | `agent.py:230` `C_losses['contrast'] = -pref_label · similarity` | `-ρ · sim` | MATCHED | the **signed** ρ, so condemned steps push similarity down |
| 4.11 | `agent.py:231` `C_model_loss.mean() + 1.0 · contrast.mean()` | `kl + contrast·contrast_term` | NTT | equal because both terms are `(B, T)` means; the assembly has no port test |
| 4.12 | — | the CRSPP has **no** decoder, reward or cont loss | MATCHED | addendum §2 / IDD §J; theirs has none either |
| 4.13 | `agent.py:183` the CRSPP steps inside `model_opt` | `fit_crspp` with its own clip 1000, its own adam 1e-4 | DIVERGES (recorded) | rows 3.1–3.2 |
| 4.14 | — | `fit_crspp` steps a fourth part, `initial_state_prior:deter` | DIVERGES (recorded) | row 4.1's other half: a parameter set theirs does not have |
| 4.15 | `nets.py:152-157` `obs_step_no_action` masks the carried state at `is_first` and adds `initial` back, as `obs_step` does | `observe_no_action` took no `is_first`: the CRSPP's scan carried across every episode boundary inside a window (1.67% of rows), while the world model's `observe` restarted there | **DEFECT — FIXED 2026-10-01** | found by the actor-entropy diagnostic §29: trained with the carry, our CRSPP learns a start-blind preference rollout that scores a more saturated trajectory as closer; trained with the restart it learns their start-conditioned one. Neither 4.2 nor 4.4 listed the mask. Fixed: `is_first` threaded through `observe_no_action`, `crspp_loss`/`fit_crspp` and `crspp_recognize`, restarting at the CRSPP's own initial (the reset target does not matter: §29) |

## 5. The info-gain ensemble trainer

| # | operation (theirs) | ours | disp. | note |
|---|---|---|---|---|
| 5.1 | `agent.py:186` `epistemic_module.train({**data, **outs['post']})` — **after** the world step, on the **pre-step** posteriors | `fit_ensemble` on `posterior_states(system, params_after_world_step, ...)` | DIVERGES (new, 21) → CLOSED 2026-09-25 | `outs` is `nj.grad`'s aux, so their posteriors are the ones the world loss saw: pre-step parameters, same PRNG draws. Ours recomputes at post-step parameters with a fresh key. See §10 |
| 5.2 | `epistemics.py:51` `inp = sg(inputs(data)[:, :-1])` over `[deter, stoch, action]` | `ensemble.transitions`: `(deter[:, :-1], stoch[:, :-1], action[:, 1:])` | MATCHED | **measured**: their ensemble input is `f32[16, 63, 1153]`, i.e. `T−1 = 63` examples per sequence over `[deter, flat stoch, action]`. The action-index off-by-one is right on both sides: `data['action'][t]` is the action taken *at* `t` in theirs, and our shifted `action[t+1]` is the same quantity |
| 5.3 | `epistemics.py:52` `tar = sg(target(data)[:, 1:])` with `ig_target: [stoch]` | `target = flat stoch[:, 1:]` | MATCHED | the next posterior code, flattened, stop-gradiented |
| 5.4 | `epistemics.py:55` `net._shape = tar.shape[2:]` — a Python mutation that *creates* the `Dist` | `MLPConditional` declares `child_dim` | N/A | their heads' output shape is set at first loss call; a construction-order dependency with no analog |
| 5.5 | `epistemics.py:56` `-net(inp).log_prob(tar).mean()` per member | `mean(vmap(member.diag_nll))` | NTT | `Independent(Normal, 1).log_prob` sums over the 1024 dimensions and then means over `(B, T−1)`; `diag_nll` sums and the caller means |
| 5.6 | `nets.py:575` σ from `minstd = maxstd = 1.0` ⇒ σ ≡ 1.0 | `head="bounded"`, `ENSEMBLE_STD = 1.0` | MATCHED | CLOSED divergence 15; a fixed-σ scaled MSE against a `tanh` mean |
| 5.7 | `epistemics.py:58-61` `probs = uniform(...)`, `nolearn`, `scales = nolearn.clip(1e-8)`, `.sum()` | `learning_gate` + `jnp.sum(scales · per_member)` | MATCHED | `ig_head_random_no_learning: 0.0` makes the gate all-ones; the `.sum()` reduction was CLOSED 2026-09-01 |
| 5.8 | `epistemics.py:48` `self.opt(self.nets, self.loss, data)` — one optimizer over all four | one `filter_grad` over the member tuple, one clip, one `apply_gradient` | MATCHED | members are independent only through their initializations |
| 5.9 | `ig_opt: {clip: 100, eps: 1e-5, lr: 1e-4}` | `ensemble.DEFAULT_CLIP/EPS/LR` | MATCHED | CLOSED 2026-09-02 |
| 5.10 | the ensemble step runs **before** `planner.train` | `fit_ensemble` before `imagine` | MATCHED | so the imagined `IG` leaf reads the freshly stepped ensemble on both sides |

## 6. `ImagActorCritic.train` — the imagination rollout and the two heads

| # | operation (theirs) | ours | disp. | note |
|---|---|---|---|---|
| 6.1 | `agent.py:87-90` `start = flatten(context)` over `(B, T)` ⇒ `B·T = 1024` starts | `flat` reshape of `posterior_states` | MATCHED | every step of the batch is a rollout start |
| 6.2 | `agent.py:87` `context = {**data, **wm_outs['post']}` — the **pre-step** posteriors | recomputed post-step with a fresh key | DIVERGES (new, 21) → CLOSED 2026-09-25 | §10 |
| 6.3 | `agent.py:88` `C_context = {**data, **wm_outs['C_post']}` — the **pre-step** CRSPP posteriors | `crspp_recognize` at post-step CRSPP params, fresh key | DIVERGES (new, 21) → CLOSED 2026-09-25 | theirs is doubly mixed: the *start* is pre-step while `img_step_no_action` inside `imagine_preference` runs at post-step parameters |
| 6.4 | `agent.py:433` `pref_traj = imagine_preference(C_start, 8)` — computed **before** `imagine` | the cosine row's `prepare` rolls it per start under `vmap` | MATCHED | **measured**: `imagine_preference` returns `{deter:f32[9,1024,128], …}` and runs before `WorldModel.imagine`. Order is immaterial (no shared state), and both roll `horizon` steps from the CRSPP posterior |
| 6.5 | `nets.py:199` `img_step_no_action` **samples** the preferred code | `update_empirical_prior`, mean-field | DIVERGES (recorded) | the keyless-face convention; the cosine row's *level* is not comparable to theirs. Already an adjacent finding in §(f) |
| 6.6 | `agent.py:276-278` `pref_traj['cont']`, `pref_traj['weight']` | not computed | N/A | **measured**: `pref_traj` carries `{deter, logit, stoch, cont, weight}` and, unlike `traj`, **no `action` key**; only `deter`/`stoch` reach `cosine_sim` |
| 6.7 | `agent.py:244` `first_cont = 1 − start['is_terminal']` from replay | `imagine(start_cont=batch.cont)` | MATCHED | replay data, not a prediction |
| 6.8 | `agent.py:246` `start` filtered to the RSSM's keys | belief tuple | MATCHED | |
| 6.9 | `agent.py:247` `start['action'] = policy(start)` — the actor sampled at `s_0` | `roll_batch`'s first scan step | MATCHED | the rollout's first action is the actor's, not replay's |
| 6.10 | `agent.py:432` `policy = lambda s: self.actor(sg(s)).sample(...)` | `actor_m.condition_on((sg(feat),)).sample(k)` | MATCHED | CLOSED divergence 12: the actor's *conditioning* state is stop-gradiented every step; the state chain stays live |
| 6.11 | `tfd.Normal.sample` — reparameterized | `Gaussian.sample` = `loc + chol·ε` | MATCHED | `actor_grad_cont: backprop` needs the pathwise gradient |
| 6.12 | `nets.py:170-172` `action_clip: 1.0` inside `img_step`, as a `sg`-ed rescale | `controls.soft_clip`, applied only into `_sampled_step` | **22, FIXED** | and the raw action goes to the term context, as their `traj['action']` is raw. §10 |
| 6.13 | `agent.py:248-253` `scan(step, arange(8), start, imag_unroll=False)` | `lax.scan` over `horizon` | MATCHED | 8 imagined steps, `imag_horizon: 8` |
| 6.14 | `agent.py:250-251` each step: `img_step` then `policy(state)` ⇒ the actor runs **H+1 = 9** times | 8 times | **24, RECORDED (cosmetic)** | **measured**: one `actor(sg(s)).sample` before the scan (rng draw 400) plus one per scan step, and `traj['action']` comes out `f32[9, 1024, 1]`. The ninth action, at `s_H`, is read only by row 6.22's index. §10 |
| 6.15 | `agent.py:255` `traj = concat([start[k][None], v])` ⇒ `H+1` states | `critic` is `H+1`; per-step leaves are `H` | MATCHED | the same indexing, expressed as separate arrays instead of one padded dict |
| 6.16 | `agent.py:256` `cont = heads['cont'](traj).mode()` — a **hard** 0/1 | `_cont_mode` | MATCHED | CLOSED divergence 10; transcription-tested |
| 6.17 | `agent.py:257` `traj['cont'] = concat([first_cont, cont[1:]])` | `Rollout.cont` = modes at `s_1 … s_H`, `first_cont` supplied separately | MATCHED | the head's mode at `s_0` is discarded on both sides |
| 6.18 | `agent.py:259` `weight = cumprod(discount·cont, 0)/discount` | `rollout_weight` | MATCHED | CLOSED divergence 11; transcription-tested, truncation included |
| 6.19 | `agent.py:571` `value = self.net(traj).mean()` at `s_0 … s_H` inside the actor's loss | `critic_sg.predict` per step plus `v0` | MATCHED | the critic's *parameters* get no actor gradient on either side (theirs because `modules = self.actor`), while the *states* stay live so reward gradients flow |
| 6.20 | `agent.py:356` `rewfn = heads['reward'](s).mean()[1:]` | `predicted_reward` row at `ctx.qs = s_{k+1}` | MATCHED | `symlog_disc_mean` = `symexp(Σ softmax·bins)` |
| 6.21 | `agent.py:357` `crsppfn = crspp_traj_sim(s, p)[1:]` — `cos(traj_{k+1}, pref_{k+1})` | cosine row at `ctx.t = k` against `preferred[k]` = pref `s_{k+1}` | MATCHED | same index pairing; row 6.5 is the remaining difference |
| 6.22 | `agent.py:358` `igfn = epistemic_module(s)[1:]` — the ensemble at `(s_{k+1}, a_{k+1})` | `model_uncertainty` row at `(ctx.qs_prev, ctx.action)` = `(s_k, a_k)` | **20, ARGUED (keep ours), measured** | **measured**: each member is called as `MLP(inputs=f32[9, 1024, 1153])` — `1153 = 128 + 1024 + 1`, i.e. `[deter, flat stoch, action]` over all nine trajectory steps — returning `f32[9, 1024]`, and `igfn` then returns `f32[8, 1024]`. A one-step index shift in the info-gain leaf; ranked first of the new findings. §10 |
| 6.23 | `epistemics.py:39` `preds = [dist.sample(...)]`, `mixsigma = preds.std(0)` | the moment-matched `√(mean σ² + var μ)` | DIVERGES (adjudicated) | RESOLVED 2026-09-05; a deliberate variance reduction of the same functional |
| 6.24 | `epistemics.py:43-44` `(ent_avg − avg_ent) / √stoch`, per-dimension means | the row's `per_dim` and `1/√num_vars` | MATCHED | CLOSED divergence 1; transcription-tested in `test_amortized_objective.py` |
| 6.25 | `agent.py:562-564` `G = 1.0·rew + 1.0·crspp + 1.0·(−IG)` | `term_value`'s registered signs, negated once | MATCHED | `G_scales` all 1.0; the `−IG` is the `model_uncertainty` row (adjudicated 2026-08-28) |
| 6.26 | `agent.py:565-568` the two `assert len(...) == len(traj['action']) − 1` | shapes are static | N/A | their runtime check |
| 6.27 | — | `g_tail` mounted in `terms=` but `terminal=True` | MATCHED | so it is *not* folded into the per-step `G`; the critic closes the horizon once, through the λ-return, as theirs does |
| 6.28 | `agent.py:569-576` the λ-return loop | `lambda_return` | MATCHED | transcription-tested, including the bootstrap leaf and which end is dropped |
| 6.29 | `agent.py:570` `disc = traj['cont'][1:]·discount` | `discount * rolls.cont` | MATCHED | `first_cont` never enters `disc` on either side |
| 6.30 | `agent.py:452` `retnorms[key](ret)` — update then read, on `sg(ret)` | `perc_ema(state, sg(ret), ...)` | MATCHED | transcription-tested over a sequence of updates; `perc_ema`, uncorrected, `max: 1.0` |
| 6.30b | `jaxutils.py:280-282` the retnorm `Variable.write` happens **inside** the actor's gradient context and is committed by `nj.grad`'s `context().update(state)`, i.e. before `actor_opt` applies | `perc_ema` runs inside `loss_fn`; `new_retnorm` comes back as aux and is written to `LearnState.moments` after the gradient | MATCHED | **measured**: the writes to `agent/planner/ac/retnorm_extr/{low,high}/value` appear under `ImagActorCritic.loss`. Both sides *use* the pre-update statistic and *commit* the post-update one exactly once per step |
| 6.31 | `agent.py:453-455` `(normed_ret − normed_base) · scales/total`, then `stack(advs).sum(0)` | `advantage = (ret − off)/inv − (base − off)/inv` | NTT | one critic, `scales = {'extr': 1.0}`, `total = 1.0`, so the fold is the identity; no port test covers the fold itself |
| 6.32 | `agent.py:465-466` `logpi = actor(sg(traj)).log_prob(sg(traj['action']))[:-1]` | not computed | N/A | the `reinforce` branch only; `actor_grad_cont: backprop` |
| 6.33 | `agent.py:465,468` a **second** actor forward pass over all `H+1` states for `ent` | entropy taken inside the rollout | MATCHED | deterministic given `sg`-ed states, so the distributions are identical; the arity differs, the numbers do not |
| 6.34 | `agent.py:468` `ent = policy.entropy()[:-1]` at `s_0 … s_{H−1}` | `Rollout.policy_entropy` | NTT | `Independent(Normal, 1).entropy()` = `Gaussian.entropy` here |
| 6.35 | `agent.py:473` `wment = get_dist(traj).entropy()[1:]` at `s_1 … s_H` | `factored_categorical_entropy(prior logits)` per step | MATCHED | the *prior*'s entropy at the state stepped into, summed over the 32 variables |
| 6.36 | `agent.py:467,469-470` `loss = −adv − actent·ent`, **then** `·= sg(weight)[:-1]` | `per_step * weight` | MATCHED | transcription-tested; the parenthesization is the content |
| 6.37 | `agent.py:474` `loss −= wment·wment` — **outside** the weight | `− wment · mean(prior_entropy)` | MATCHED | transcription-tested |
| 6.38 | `agent.py:477-480` `actor_refresh = −log π(start['action'])·can_self_imitate·0.5` | `bc_scale · mean(bc)` | MATCHED | transcription-tested; `can_self_imitate` is the *propagated* mask, a weight in [0, 1] |
| 6.39 | `agent.py:478` `start['action']` is the **raw recorded** action, unclipped | our recorded action, now the raw draw too | **22b, FIXED** | fixed at the collection site, which is where the clip was (acting ledger D5): `habit_action` returns the raw sample, and the box moved to the action's consumers — the env, and the recurrent scan (row 2.4). The BC target is now theirs |
| 6.40 | `agent.py:482` `loss *= loss_scales.actor` | no such factor | **25, RECORDED (cosmetic)** | inert at `1.0`; recorded because a non-unit value is inexpressible in ours |
| 6.41 | `agent.py:484` `loss.mean() + actor_refresh.mean()` | the same two means | MATCHED | transcription-tested |
| 6.42 | `agent.py:437` `self.opt(self.actor, loss, ...)` — the actor stepped **alone** | one `filter_grad` over `(actor, critic)`, applied separately | MATCHED | the objectives are mutually stop-gradiented, so the joint grad equals the two separate ones |
| 6.43 | `actor_opt: {clip: 100, eps: 1e-5, lr: 3e-5}`, clipped over the actor alone | `_clip(actor_grad, 100)`, `adam(1e-5)`, `lr 3e-5` | MATCHED | CLOSED divergence 5 |

## 7. The critic

| # | operation (theirs) | ours | disp. | note |
|---|---|---|---|---|
| 7.1 | `agent.py:440` `critic.train(traj, pref_traj, actor)` — **after** the actor's step, on the actor's returned `traj` | inside the same `loss_fn` | MATCHED | the actor's step cannot move the critic's inputs (the trajectory is already materialized) |
| 7.2 | `agent.py:522` `target = sg(self.score(traj, pref_traj, actor)[1])` — a **second** `score` call | `ret` computed once and reused | **23, RECORDED (cosmetic)** | **measured**: `GFunction.train` calls `GFunction.score` a second time, with four fresh `nj.rng()` draws inside `InformationGain.__call__`. Their second call redraws the ensemble's samples, so their critic target's `IG` term differs from the actor's by sampling noise. Ours has no such noise (row 6.23), so the two collapse; §10 |
| 7.3 | `agent.py:530` `traj = {k: v[:-1]}` before both `self.net` and `self.slow` | `start_features` = `s_0 … s_{H−1}` | MATCHED | the slow copy is read at the *same* truncated states |
| 7.4 | `agent.py:532` `-dist.log_prob(sg(target))` | `symlog_disc_nll(logits, sg(ret))` | MATCHED | 255 bins, two-hot; A12 item 5 chose the code over paper Eq. 9 |
| 7.5 | `agent.py:534` `reg = -dist.log_prob(sg(slow(traj).mean()))` | NLL against `sg(slow.predict(feats))` | MATCHED | transcription-tested; the asymmetric "logprob of the slow copy's point estimate" |
| 7.6 | `agent.py:536-539` the `xent` branch | — | N/A | `critic_slowreg: logprob` |
| 7.7 | `agent.py:542-544` `loss += 1.0·reg`, `(loss·sg(weight)).mean()`, `·= 1.0` | `mean((nll + slowreg·nll_slow)·weight)` | MATCHED | transcription-tested; **not** `mean(nll) + mean(nll_slow)` |
| 7.8 | `agent.py:495` the critic's states are the actor's live rollout | `sg(rolls.start_features)`, `sg(ret)` | MATCHED | equivalent: theirs differentiates only `self.net` |
| 7.9 | `critic_opt` identical to `actor_opt`, clipped over the critic alone | `_clip(critic_grad, 100)` | MATCHED | one `adam_eps` for the pair, since their two blocks agree |
| 7.10 | `agent.py:525` `self.updater()` **after** the critic's step | `_slow_update(slow, new_critic, ...)` after `apply_gradient` | MATCHED | so the EMA tracks the *freshly stepped* critic |
| 7.11 | `jaxutils.py:486-488` `mix = clip(1·need_init + 0.02·need_update, 0, 1)` | `_slow_update` | MATCHED | CLOSED divergence 13 (the first-update hard copy); transcription-tested over a run of steps |
| 7.12 | `jaxutils.py:495` `updates += 1` | `moments[SLOW_KEY]` | MATCHED | the counter rides in `LearnState.moments`, not in `opt_states` |
| 7.13 | `nets.py:514` `slow` gets its own random init | `_value_head` with its own key | MATCHED | and the *first* loss is computed against that unrelated net on both sides |

## 8. Self-revision (ρ), which the CRSPP consumes

| # | operation (theirs) | ours | disp. | note |
|---|---|---|---|---|
| 8.1 | `callbacks.py:38` `can_self_imitate *= has_succeeded[-1]` | `gated` | MATCHED | the destructive success gate, first |
| 8.2 | `callbacks.py:39-45` the backward carry `max(carry·0.4, is_success[t])` with the `0.001` floor | reverse `lax.scan` over `success[1:]` | MATCHED | index for index, seed included; `test_self_revision.reference` is a verbatim port |
| 8.3 | `callbacks.py:46-50` `clip(gated + mask, 0, 1)`, then the **forward** shift | `final`, `positive` | MATCHED | |
| 8.4 | `callbacks.py:51-54` `β^(T−1−t)`, gated off on success, times `1 − gated` | `decays`, `neg` | MATCHED | the failure decay condemns *where you ended up* |
| 8.5 | `callbacks.py:56-58` `pref_label[0] = 0`, `clip(−1, 1)` | `.at[0].set(0.0)`, `clip` | MATCHED | |
| 8.6 | `callbacks.py:31` runs once per **finished** episode, before the buffers are written | `revise.label` before `replay.add` | MATCHED | a stored window's last step is not the episode's last step |

## 9. Cross-cutting

| # | operation (theirs) | ours | disp. | note |
|---|---|---|---|---|
| 9.1 | `nets.py:672-690` `Linear._bias = bias and norm == 'none'` ⇒ normed hidden layers carry **no** bias | `MLPConditional` drops it; `MLPPredictor`, `MLPCategorical` and `GRUCell` keep it | **26, OPEN and MEASURED TWICE** | the divergence-14 closure covered `MLPConditional` only. Tried alone (0.0120 → 0.0386 keyless MSE) and then *paired with 17* as adjudicated, which is their actual configuration: the pair's sampled open-loop MSE is 0.0206 against ours 0.0088 and a 0.0159 persistence baseline, losing to persistence on 4/4 seeds. Reverted; §10 carries the table and the fan-avg mechanism |
| 9.2 | `nets.py:757-763` `winit: normal, fan: avg` on all eight net blocks | `winit="normal"` on the actor and the ensemble; `eqx` defaults elsewhere | DIVERGES (recorded) | divergence 17, explicitly left OPEN for the encoder, decoder, critic, reward/cont heads and the GRU |
| 9.3 | `nets.py:561,564` the `Dist` output linears get `winit='uniform'` (Linear's default), not the block's `normal` | `uniform_avg` on the fused head layer, each half at its own fan | **26b, FIXED** | second moments agree exactly (`U(−ℓ, ℓ)` with `ℓ = √(3·scale)` has std `√scale`, as does their corrected trunc-normal), so this is cosmetic — but the transcription test at `test_oracle_transcription.py:740` pins the *normal* shape for a layer their code draws uniformly |
| 9.4 | `nets.py:562,564` two separate `out`/`std` linears | one fused `2·child_dim` layer | MATCHED | the test already draws each half at its own fan for this reason |
| 9.5 | `Dist.outnorm`, `actor.unimix`, `Linear.outnorm` | — | N/A | assigned and never read in their `nets.py`, or read only by a branch this action space does not take |
| 9.6 | `jax.precision: float16` | float32 | DIVERGES (argued) | ARGUED divergence 9; row 3.7 records its optimizer-machinery consequences |
| 9.6b | `ninjax.py:166` `nj.grad`'s wrapper draws **one** `rng()` from the *outer* context per optimizer and uses it to seed the *inner* pure context the loss body runs in | one explicit key per trainer, split from `train_step`'s | MATCHED | **measured**: 4 such draws per train call, one per `Optimizer.__call__`. Structurally the same discipline as ours — a loss body's randomness descends from one key handed to it — which is why the two streams are comparable in *shape* even though not in value |
| 9.7 | `nj.rng()` draws: per `Linear` dropout call, per `_stats` sample, per `subsample` in metrics, per ensemble sample | one key split per scan step, none in metrics | N/A | **measured**: 456 draws per train call, identical across both calls — 216 in `img_step` (3 × 72), 72 in `img_step_no_action`, 64 + 64 in the two `obs_step`s, 9 for the actor in `imagine`, 8 for the ensemble (4 × the two `score` calls), 4 for `nj.grad`, 4 for `nj.scan`, 1 for the no-learning mask, and **14 in `tensorstats`/`subsample`** — the last of which means their *metrics logging* perturbs the RNG stream, and shape-dependently (5 of 19 `tensorstats` calls are on ≤1024 elements and draw nothing). The stream cannot be matched across two libraries; recorded so "same seed, same numbers" is never expected |
| 9.8 | the cosine row's rollout length | `st.policy_len`, not `imagine`'s `horizon` | **28, RECORDED (inert)** | inert at parity (both 8); a caller passing `horizon != policy_len` would silently compare against a differently-long preferred trajectory |
| 9.9 | five parameter sets stepped per `train` call | five | MATCHED | θ world, φ CRSPP, ω ensemble, ψ actor, χ critic — see §6 of `raif_agent` for the group map |
| 9.10 | the order of the five steps | world → ensemble → CRSPP → actor+critic | MATCHED | theirs is (world+CRSPP) → ensemble → actor → critic; the CRSPP's position differs but nothing reads it in between except row 6.3 |
| 9.11 | `agent.py:94` `outs = {}`, so `replay.prioritize` never fires | no priority | MATCHED | uniform replay |

---

## 10. The new DIVERGES rows: mechanism, magnitude, consequence

Numbering continues `amortization.md` §(f). "Measured" means a number from this session;
everything else is inference from reading and is labelled as such.

### 20. The info-gain leaf is evaluated one rollout step earlier than theirs

***ARGUED DIVERGENCE — ADJUDICATED 2026-09-07 (Alex): KEEP OURS.*** The reason is
structural, and it is now written at the row itself
(`objectives/amortized.py::_ensemble_disagreement`) as well as here. Their indexing needs
`a_{k+1}` at index `k` — an action sampled in the *following* scan iteration — so expressing
it would take one of two things, and both break a contract rather than bend one: extend the
per-step term context with a look-ahead action, which makes a step's score depend on a
decision not yet taken and changes the contract every row is written against; or shift the
leaf array inside the rollout, as a special case the term-tree grammar cannot express. Our
contract is that at index `k` every row is a functional of the same event — taking `a_k` from
`s_k` and arriving at `s_{k+1}` — and that co-location is what makes the fold to `G`
coherent.

***MEASURED, because a blessed divergence still gets measured*** (the lesson of
`means.std(0)`). At initialization, `run_mc.SMALL` sizes with `horizon=8`, 64 independent
rollouts, both indexings computed on identical trajectories; the codebase's own
`step_terms_fn` was checked to return `−L[t]` exactly at every real step, so the harness is
the row and not a mirror of it.

*The leaf, per index.* `L` runs ≈0.031 and its per-index mean is
`[0.0329, 0.0308, 0.0309, 0.0304, 0.0307, 0.0303, 0.0301, 0.0302, 0.0301]`, sd 0.0006 at
index 0 rising to 0.003–0.004 after. So the depth trend is a **6.3% drop** from index 0 to
index 1 and then flat — which corrects the audit's inference above that "open-loop
disagreement grows with rollout depth, so theirs pays a strictly larger cost". At
initialization it does not grow; it steps down once and levels off, and the *sign* of the
level difference is the other way round.

*The gap.* `|L[k+1] − L[k]|` over 512 samples: mean **0.0039**, max 0.0164 — **12.7% of the
leaf's own magnitude** (median 11.4%).

*In `G` and in the advantage.* Reward and cosine-sim are read at `(s_t, a_t)` in both
indexings and cancel exactly, so the whole difference is this leaf. Per-step `G`: mean signed
+0.00022, mean |·| **0.0037**, max 0.0139. λ-return: mean |·| 0.0036, max 0.0145, against the
return's own sd of 0.295 — ≈1.2%. Actor advantage, after the percentile-EMA normalization:
mean |·| 0.0036 against a p5–p95 spread of 0.910, i.e. **0.39% of the advantage's own
spread**, and roughly flat across the horizon (0.0032 at step 0, 0.0044 at step 7) rather
than concentrated early. So the λ-return's early-step weighting does **not** amplify it.

*Not a pure relabelling.* `value_theirs[k] = value_ours[k] + (L[k] − L[k+1])` reads `L[1..8]`
where ours reads `L[0..7]`: it drops `L[0]` entirely and introduces `L[8]`, a `(state,
action)` pair an 8-step actor rollout never otherwise scores. A shifted window over a 9-long
sequence, not a permutation — and even a permutation would change the fold, because the
λ-return's step weights are unequal.

*Scaling with ensemble disagreement.* Sweeping the members' injected mean-spread `s` against
their fixed `σ ≡ 1.0` (20 trajectories per point, 160 step-samples):

| spread `s` | leaf gap (mean) | leaf magnitude | gap / \|leaf\| | advantage gap | gap / advantage spread |
|---|---|---|---|---|---|
| 0.0 | 0.0037 | 0.031 | 12.1% | 0.0034 | 0.39% |
| 0.03 | 0.0038 | 0.031 | 12.5% | 0.0035 | 0.40% |
| 0.1 | 0.0040 | 0.033 | 12.5% | 0.0037 | 0.42% |
| 0.3 | 0.0059 | 0.045 | 13.1% | 0.0055 | 0.67% |
| **1.0 (= σ)** | **0.0082** | 0.153 | 5.4% | **0.0072** | **1.64%** |
| 3.0 | 0.0051 | 0.471 | 1.1% | 0.0048 | 0.39% |
| 10.0 | 0.0018 | 0.942 | 0.2% | 0.0017 | 0.05% |

The absolute gap **peaks at `s ≈ σ`** and falls away on both sides: near `s = 0` the leaf
itself is small, and at large `s` both `L[k]` and `L[k+1]` converge on a function of `s`
alone. Across the whole sweep the advantage gap never exceeds **1.7%** of the advantage's own
spread.

*Caveats, stated because they bound the claim.* Untrained agent, `SMALL` sizes, not the parity
sizes; the sweep perturbs predicted means at read time rather than retraining an ensemble to
a larger organic spread; and a 9th action had to be drawn by hand because a real 8-step
rollout never samples one (the private helpers `imagine` uses internally were called
directly, and fidelity to the real per-step tree was verified).

*The test.* `tests/test_amortized_objective.py::test_the_info_gain_leaf_is_read_one_step_earlier_than_their_call_site_reads_it`,
built from **their call site** (`_their_igfn_from_the_call_site`, a port of
`igfn = lambda s: self.epistemic_module(s)[1:]`) rather than from their expression — which is
this round's method finding applied to the port that had the defect. It pins five things: the
two indexings as two slices of one array, our row's folded leaf as `sum(L[:-1])`, their slice
as a *different* number, their last entry's dependence on `a_H` (perturb the ninth action and
theirs moves while ours cannot), and the formula itself at every index against their own
expression in the many-sample limit.

*The finding as originally written follows.*

**Mechanism.** Their `igfn = lambda s: wm.epistemic_module(s)[1:]` (`agent.py:358`) evaluates
the ensemble over the *whole* `H+1`-step trajectory and then drops index 0. Their
`InformationGain.inputs` is `Input(config.ig_head.inputs, dims='deter')` with
`ig_head.inputs: [deter, stoch, action]` — and note that the `config.update({'ig_head.inputs':
['tensor']})` on the line above does **not** affect it, because `embodied.Config.update`
returns a new object (`embodied/core/config.py:130`) and `self.inputs` is built from the
original. So `G[k]` is scored with the ensemble read at `(s_{k+1}, a_{k+1})`, which is the
disagreement about `s_{k+2}`. Ours reads `(ctx.qs_prev, ctx.action) = (s_k, a_k)` — the
disagreement about `s_{k+1}`, i.e. the transition actually taken at step `k`.

**Measured, in the trace.** Each of the four members is called as
`nets.py:511 MLP.__call__(inputs=f32[9, 1024, 1153])` — `1153 = 128` deter `+ 1024` flat
stoch `+ 1` action, over **all nine** trajectory steps — `InformationGain.__call__` returns
`f32[9, 1024]`, and `igfn` returns `f32[8, 1024]`. So the leaf really is evaluated at nine
states and truncated from the front, and `a_H` really is one of its inputs.

**Magnitude.** Inferred, not measured. Two components. The *shift* moves each step's `IG`
one place earlier in an 8-step window; since the λ-return weights early steps most heavily
this is not a pure relabelling. The *level* difference is systematic: theirs averages the
ensemble's disagreement over `s_1 … s_H` and ours over `s_0 … s_{H−1}`, and open-loop
disagreement grows with rollout depth, so theirs pays a strictly larger `model_uncertainty`
cost — by roughly one step's worth of uncertainty growth out of eight. The leaf currently
sits near `−0.33`/dim at the run's operating point (§(b)'s measurement), against a reward of
about `−2`/step, so a few percent of the leaf is small next to reward but not next to the
*differences* between actions, which is what the advantage sees. Additionally, theirs reads
`a_H` — the ninth sampled action (divergence 24) — for `G[H−1]`, an input ours does not have
at all.

**Consequential?** Plausibly, at the margin. It is the only new finding that changes a term
of `G` on every step of every rollout. Ours is arguably the semantically correct pairing —
the ensemble is *trained* on `(s_t, a_t) → s_{t+1}`, so scoring it at the step's own inputs
is what "information gain from taking this action" means, and their `[1:]` is the reward
head's convention applied to a quantity that does not share it. Under "the oracle is the
code" parity wants theirs. This needs a human, and it is not covered by any existing test:
`test_ensemble_disagreement_matches_its_formula_and_reads_the_steps_inputs` pins the row
against `_ensemble_moments`, which is written at *the row's* indexing, so the port certifies
the leaf's formula and not its position.

### 21. The shared forward pass is recomputed post-step instead of reused

***RECORDED 2026-09-07, not fixed; the docstring that made the opposite claim is
corrected.*** `raif_agent.train_step`'s docstring said the states the other three trainers
read are the freshly stepped world model's "as theirs are", and theirs are the *stale*
pre-step aux. It now says so, with the mechanism, the measurement and the two-part magnitude.
Not fixed because reading stale states is not obviously the better algorithm and the reuse
would mean threading `fit_world`'s loss aux out through the trainer.

**Mechanism.** Their `WorldModel.train` is
`mets, (state, outs, metrics) = self.opt(modules, self.loss, ...)`, and `outs` is `nj.grad`'s
auxiliary output — the forward pass at the **pre-step** parameters, with the PRNG draws that
pass made. Both later consumers read it: `epistemic_module.train({**data, **outs['post']})`
(`agent.py:186`) and `context = {**data, **wm_outs['post']}` / `C_context = … wm_outs['C_post']`
(`agent.py:87-88`). Ours calls `posterior_states` again after `fit_world` with `k_states`,
and `crspp_recognize` again after `fit_crspp` with `k_recognize` — post-step parameters,
independent draws.

**Measured.** `nj.grad`'s `_prerun` is a no-op in both calls (`context().create` is False
under `nj.jit`'s `apply`), so there is no shadow execution of the loss body and `outs` really is
the single forward pass the gradient was taken through — the aux, not a recomputation.

**Magnitude.** Two independent parts. The *parameter* part is one adam step at `lr 1e-4`,
which is small per step and identical in kind to the difference between any two adjacent
steps. The *sampling* part is a fresh straight-through draw of every posterior code in a
`16×64` batch; statistically it is an equally valid draw of the same estimator, so no
quantity's expectation moves. There is also a throughput cost: two extra full `observe`
scans per training step, one for the world model and one for the CRSPP.

**Consequential?** Probably not for the learning trajectory. It is listed because
`raif_agent.train_step`'s docstring currently makes the *opposite* claim — "Step 2 is
deliberately after step 1: the states the other three read are the freshly stepped world
model's, **as theirs are**" — and that clause is wrong. Their downstream trainers read the
stale states, and the docstring should say so whichever way the code goes.

***CLOSED 2026-09-25.*** Taken in the performance round, where it surfaced again as
"the RSSM is run twice per step": `fit_world` and `fit_crspp` now return the states of the
forward pass they differentiated (`return_states=True`; pre-step parameters, that pass's
draws, gradient stopped), and `train_step` reads those for the ensemble, the CRSPP's
context and the imagination starts, with the CRSPP posterior likewise its own pass's. The
two recomputations are gone, and with them the two independent parts above. The key split
is unchanged, so the world, CRSPP and imagination streams draw what they drew; the
`k_states` and `k_recognize` slots are simply unused. Tests pin the returned states to
`posterior_states`/`crspp_recognize` at the *pre-step* parameters under the same key
(`tests/test_amortized_trainers.py`, last two). The CRSPP's embeddings are the pre-step
encoder's too, as their shared `embed` is. Every run recorded before this date
(`crspp_init_100k`, `crspp_sampled_500k` among them) trained with the recomputation; a
comparison across the date straddles this change.

### 22. `jnp.clip` on the imagined action zeroes a gradient their rescale keeps

***FIXED 2026-09-07*** — `controls.soft_clip`, their expression written for a general box
(a rescale of the offset from the box's centre, which reduces to
`a *= sg(c / max(c, |a|))` exactly on their `Box(-1, 1)`), applied in `imagine.roll_batch`,
`habit._from_module` and `run_mc.make_collect_fn`. The gradient claim is verified directly:
at `a = 2.0`, `d(jnp.clip)/da = 0.0` and `d(soft_clip)/da = 0.5 = 1/|a|`, and at four
out-of-box values the rescale's derivative equals `1/|a|` to the digit.

***MEASURED, and the audit's inferred 30–45% was in the right range.*** At initialization,
`run_mc.SMALL` sizes, actor conditioned on 16,000 **real** posterior states drawn from the
fixture-seeded buffer: `P(|a| > 1)` = **41.1%**, mean `|a|` 0.939, max 4.37; the actor's own
statistics are mean μ 0.627 (sd 0.42) and mean σ 0.865. Reproducing `roll_batch`'s step
exactly over 12,000 rollout-step actions gives **40.6%**. Among the clipping steps the mean
`1/|a|` is **0.68**, so what `jnp.clip` discarded was not a vanishing tail of the gradient
but roughly two thirds of its strength on two fifths of the steps.

*After training,* with a caveat that matters: a 20k-step `SMALL` run (≈1,264 actor gradient
steps at `lr 3e-5`) moved the fraction **up**, to 45.0% / 46.7%, with σ drifting toward
`maxstd = 1.0` rather than toward the floor — that budget is far too short to stand in for a
converged actor, so the analytic bracket is the better guide to the trained regime:
`P(|a|>1)` is 0.309 at `(μ 0.9, σ 0.2)`, 0.460 at `(μ 0.99, σ 0.1)` and 0.261 at
`(μ 0, σ 0.89)`. The fraction does not decay away as the policy sharpens, which is the point:
their `tanh` mean approaches ±1 while σ has a 0.1 floor.

***One thing the fix changes beyond the gradient, and it is worth naming.*** Their
`imagine` is `state = img_step(prev, prev.pop('action'))` then
`{**state, 'action': policy(state)}` (`agent.py:244-251`), so `img_step` clips a *local* copy
and `traj['action']` keeps the **raw** sample. The `ig_head` conditions on that column and the
BC term evaluates a density at it, so both read a raw action on their side. Ours now passes
the raw action to the per-step term context and the rescaled one only into the transition,
which is theirs term for term — and it is consistent with 22b, since the *recorded* action is
now raw too.

*The finding as originally written follows.*

**Mechanism.** Theirs (`nets.py:170-172`, inside `img_step`) is
`prev_action *= sg(action_clip / maximum(action_clip, |prev_action|))`: the *scale* is
stop-gradiented, so for `|a| > 1` the forward value is `sign(a)` and the derivative w.r.t.
`a` is `1/|a|`. Ours is `jnp.clip(action, *bounds)` in `roll_batch`, whose derivative outside
the box is exactly zero. For a one-dimensional action the two forward values are identical.

**Magnitude.** Inferred. The actor's head is `Normal(tanh(out), 0.1 + 0.9·σ(raw + 2))`, so
early in training σ ≈ 0.89 with a mean anywhere in `(−1, 1)`; a rough calculation puts
`P(|a| > 1)` in the 30–45% range at initialization. On every such step ours contributes
**no** pathwise gradient from the advantage to the actor (the entropy and BC terms still do),
while theirs contributes a down-scaled one. As σ shrinks the fraction falls, so this is
front-loaded onto exactly the phase `amortization-p0-addendum.md` §7's "off the floor by
~50k" landmark reads.

**Consequential?** Structural and unmeasured; worth measuring before it is worth fixing. The
cheap check is to log the fraction of rollout steps whose sampled action leaves the box.

**22b, the same clip on the other side.** Their BC term evaluates `log π` at
`start['action']`, the raw recorded sample, which their `Dist('normal')` density can score
outside `(−1, 1)`; our recorded action was already clipped at the act site
(`planning/habit.py:170`), so our BC target is the clipped value. The affected steps are only
those with `can_self_imitate > 0`, and the expert seed data is bounded, so this is small — but
it is a collection-side asymmetry that surfaces in the actor loss and belongs on the record.

### 23. Their critic target is a second `score` call, ours is the actor's `ret`

***RECORDED 2026-09-07 as cosmetic-and-not-fixed.*** Their second `score` call redraws the
ensemble's samples, so their critic target's `IG` term differs from the actor's by sampling
noise; our leaf is the noise-free moment-matched estimator (adjudicated 2026-09-05), so a
second evaluation would return the identical number and there is nothing to reproduce. It
would matter immediately if the sampled estimator were ever restored.

**Mechanism.** `GFunction.train` (`agent.py:522`) recomputes
`target = sg(self.score(traj, pref_traj, actor)[1])`. `rewfn` and `crsppfn` are
deterministic, but `igfn` draws one sample per ensemble member per state
(`epistemics.py:39`), so their critic regresses on a `G` whose info-gain term is a
*different* draw from the one the actor's advantage used. Ours computes `ret` once inside
`loss_fn` and uses it for both.

**Magnitude and consequence.** Cosmetic *for us*: our info-gain leaf is the noise-free
moment-matched estimator (adjudicated 2026-09-05), so a second evaluation would return the
identical number and there is nothing to reproduce. Recorded because it is a real difference
in their control flow and because it would matter immediately if the sampled estimator were
ever restored.

### 24. Their actor runs `H+1` times per rollout, ours `H`

***RECORDED 2026-09-07 as cosmetic-and-not-fixed.*** The ninth action's only consumer is
divergence 20's index, and 20 is resolved our way, so there is nothing for a ninth actor
evaluation to feed. It is exhibited rather than merely argued: the new call-site port draws
it (`_their_traj_plan`) precisely to show that their `G[H−1]` reads an `a_H` ours never has.

**Mechanism.** `imagine`'s `step` returns `{**state, 'action': policy(state)}`
(`agent.py:251`), and `start['action'] = policy(start)` seeds it, so `traj['action']` has
`H+1` entries. Their `logpi` and `ent` both take `[:-1]`, discarding the ninth. It survives
in exactly one place: the info-gain leaf's `[1:]` index (divergence 20) reads `a_H`.

**Magnitude and consequence.** With 20 resolved in ours' favour, this is inert — there is no
consumer for a ninth action. With 20 resolved in theirs' favour it becomes a prerequisite:
you cannot score `IG` at `(s_H, a_H)` without sampling `a_H`. It also shifts the RNG stream
by one draw per rollout, which is unmatchable anyway (row 9.7).

### 25. `loss_scales.actor` has no counterpart

***RECORDED 2026-09-07 as cosmetic-and-not-fixed.*** Inert at its resolved `1.0`; ours
cannot express a non-unit value and the transcription port omits the line for the same
reason, so a change on their side would be invisible to both. That last clause is the reason
it is on the record at all.

Inert at its resolved `1.0`. Recorded because ours cannot express a non-unit value and
because the transcription test's port omits the line for the same reason, so a change on
their side would be invisible to both.

### 26. Normed hidden layers keep a bias outside `MLPConditional`

***OPEN, and MEASURED TWICE 2026-09-07 — the audit's "inferred to be inconsequential, and
cheaper to close than to argue about" was wrong on both halves.***

**First attempt: 26 alone.** Dropping the bias from every normed layer in
`MLPPredictor._linear_stack` and `GRUCell.from_shapes` — the encoder, decoder, reward and cont
heads, critic and slow copy, both categorical trunks, and the GRU's two linears — broke a
standing quality assertion: the tiny world model's **keyless open-loop prediction** went from
`0.0120` MSE to `0.0386` against an unchanged `0.0159` persistence baseline, i.e. from 1.3×
better than persistence to 2.4× worse, on `tests/test_rssm_world.py`'s 300-step fit. The
*sampled* open-loop path stayed inside its own 0.6× threshold.

**Why that reading was not yet a reading of theirs: 26 is coupled to 17.** Their `Linear`
decides the bias and the initializer in one constructor: `winit: normal, fan: avg` on every
block **and** no bias where normed. Ours declared `winit="normal"` on the actor and the
ensemble only (`reference/raif_agent.py:275,289`) — divergence 17 was OPEN for exactly the
blocks 26 covers. So 26-alone gave *no bias + eqx's initializer*, a **fourth** combination
neither codebase has, and the 0.0386 was a reading of that hybrid.

**Second attempt: 17 and 26 together, as adjudicated (Alex, 2026-09-07).** Both closed for the
six blocks — their truncated normal on the dense layers, `Initializer`'s uniform default on
the `Dist`/`_stats` output layers, `outscale` per block, and no bias wherever their norm rule
drops it. **Also reverted, on a measurement across four init seeds:**

| open-loop MSE | ours (`eqx` default) | theirs (17+26) | persistence |
|---|---|---|---|
| keyless / mean-field, mean of 4 | **0.01134** | 0.01253 | 0.01592 |
| sampled, mean of 4 | **0.00880** | **0.02063** | 0.01592 |

The pair is much less bad than the hybrid on the keyless probe (0.0125 vs 0.0386), which
confirms the coupling diagnosis — but it is still worse than ours there (1.37× on average,
worse on 3/4 seeds), and on the **sampled** probe, the one `imagine()` actually rolls, it is
decisive: ours beats persistence on 4/4 seeds (0.35–0.78×) and theirs **loses to persistence
on 4/4** (1.04–1.53×), a mean 2.6× regression.

**Where, and the best read on why.** Per-block splices (one seed, directional only) put the
harm in the two blocks with the most lopsided fans: the GRU alone takes the keyless probe to
0.0408 (2.56× persistence) and the encoder to 0.0274, while the prior (0.0011), cont (0.0042),
reward (0.0061) and posterior (0.0068) each *improve* it. Fan-avg vs fan-in rescales a layer
by `sqrt(fan_in / mean(fan_in, fan_out))` — ≈1 when squarish, far from 1 for ours: the GRU's
input projection is `1025 → 32` (theirs ≈2.4× larger), its gate `160 → 384` (≈1.33× larger,
the figure divergence 17's own text predicted), the encoder's first layer `2 → 64` (theirs
≈2.3× smaller). Their config was tuned around a 64×64×3 image encoder where fan_in ≈ fan_out;
a 2-dimensional state observation and a 1025-wide flat code are the opposite regime. The
`outscale: 0.0` heads are **exonerated** — their output layer is zeros either way.

**What the probe settles.** Init moves the training *trajectory*, not the objective's fixed
point, and trajectory is what this replication keeps failing on. A 300-step world-model fit is
a guardrail against a gross regression, not a proof of equivalence in either direction, and
this measured the world model alone. So: the gap is now measured rather than assumed, both
stay OPEN, and a revisit belongs on a trajectory rather than a probe. Full write-up at
`amortization.md` divergence 17; the structural reading of their initializer that the attempt
required is kept at `distributions/neural.py::_linear_stack`.

**Mechanism.** `Linear._bias = bias and norm == 'none'` (`nets.py:664`), so every normed
hidden layer in their stack has no bias term. The divergence-14 closure implemented this for
`MLPConditional` (`use_bias=not (norm and i < depth)`, `neural.py:359`) and nowhere else:
`MLPPredictor._linear_stack` (`neural.py:665`) and `GRUCell.from_shapes` (`neural.py:1473-1474`)
both use `eqx.nn.Linear`'s default. That covers the encoder, the decoder, the reward and cont
heads, the critic and its slow copy, both categorical trunks, and the GRU's two linears.

**Magnitude.** `standardize(xW + b)` is not `standardize(xW)` for a per-feature `b`, so this
is a genuinely larger function family and not a redundant parameterization. The extra
parameters are a few hundred against ~5·10⁵. Inferred to be inconsequential, and cheaper to
close than to argue about.

**26b — FIXED 2026-09-07.** Their `Dist` output linears are drawn `uniform` at fan-avg, not
`normal` (row 9.3): `Dist.inner` forwards only `{outscale, outnorm}` to `Linear`
(`nets.py:556-565`), so the block's `winit` never reaches the head and `Initializer`'s
`uniform` default applies. `neural.uniform_avg` is that branch, ported, and
`MLPConditional.from_shapes` draws the fused head's two halves from it (each still at its own
`(in, child_dim)` fan, because theirs are two separate linears). Second moments agree exactly
— `U(−ℓ, ℓ)` with `ℓ = √(3·scale)` has std `√scale`, as their corrected trunc-normal does —
so nothing numeric moves; what is fixed is the **test that asserted a false fact about their
source**. `test_the_dist_head_linears_are_drawn_uniform_as_theirs_are` ports their `uniform`
branch key-for-key and discriminates it from the normal branch by **support** rather than by
variance (a uniform is bounded strictly inside `√3·std`; their truncated normal runs to
`2/0.8796 ≈ 2.27·std` and, over a few hundred draws, gets there), and it asserts the trunk
takes the *other* branch. This is one of the two ports the method finding is about.

### 27. `SymlogDist`'s `tol` is not transcribed

***RECORDED 2026-09-07 as cosmetic-and-not-fixed.***

`jaxutils.py:164` zeroes squared errors below `1e-8` before the sum. Cosmetic at any
realistic scale; listed for completeness because `symlog_mse_nll` has no port test (row 2.17).

### 28. The cosine row's rollout length is the System's, not `imagine`'s

***RECORDED 2026-09-07 as inert-at-parity and not fixed*** (both are 8). The failure mode
it names — a caller passing `horizon != policy_len` indexing past the preferred trajectory
without a diagnostic — is a missing guard rather than a divergence, and is left as such.

`_cosine_sim_to_pref_rollout` rolls `st.policy_len` steps while the rollout it is scored
against runs `horizon`. Equal at parity (both 8). A caller passing a different `horizon`
would index past — or short of — the preferred trajectory without a diagnostic.

---

## 11. Verdicts on the existing transcription tests

Each port was re-read against the current source at `9b1864e8`, line by line, with the
config branch resolved.

| port | their source | verdict |
|---|---|---|
| `their_score_return` | `agent.py:569-576` | **FAITHFUL.** The `disc`, the `interm` form, the `value[-1]` bootstrap and the `[:-1]` on the reversed list all match |
| `their_moments_perc_ema` | `jaxutils.py:307-310, 338-341` | **FAITHFUL.** `perc_ema` and not `perc_ema_corr`, whole-array `jnp.percentile`, no bias correction, `1/max` floor |
| `their_weight` | `agent.py:257-259` | **FAITHFUL** |
| `their_actor_loss` | `agent.py:463-484` | **FAITHFUL**, with one *stated* omission: `loss *= loss_scales.actor` (1.0) is dropped, which the docstring names. See divergence 25 |
| `their_slow_updater` | `jaxutils.py:483-495` | **FAITHFUL.** The name-remapping of `source` is elided, correctly (it is a key rename, not arithmetic) |
| `their_critic_loss` | `agent.py:528-546` | **FAITHFUL.** `traj[:-1]` before both nets, `slow(...).mean()` as a point estimate, the weight applied to the *sum* |
| `theirs` (two-hot) in `test_their_two_hot_log_prob_is_our_symlog_disc_nll` | `jaxutils.py:201-219` | **FAITHFUL.** `self.dims` is `()` at `dims=0`, so the port's single `.sum(-1)` is right; `log1p` for `log(1 + ·)` is the same function |
| `their_kl_mix` | `nets.py:251-276` + `agent.py:212-219` | **FAITHFUL.** Free bits per direction, before the mix; `Independent(·, 1)` summed over the 32 variables |
| `their_linear` | `nets.py:672-690` | **FAITHFUL** for the `norm: layer` branch, including the asserted-absent bias and `epsilon=1e-3`. Note it is a port of the branch `MLPConditional` takes; row 9.1 is about the *other* classes, which do not take it |
| `their_dist_normal` | `nets.py:554-580` | **FAITHFUL** for the arithmetic (`tanh` mean, `+2.0` shift, the std band, biases kept, norm-free). Was **not faithful on initialization** — their `Dist` linears receive no `winit`, so they are `uniform`, and the surrounding test asserted a normal draw (row 9.3). **FIXED 2026-09-07**: `neural.uniform_avg` ports their `uniform` branch, `MLPConditional` draws the head from it, and `test_the_dist_head_linears_are_drawn_uniform_as_theirs_are` pins it by *support* (the only thing that separates the two, their variances being equal by construction) |
| `their_independent_normal_log_prob` | `tfd.Independent(Normal, 1)` | **FAITHFUL** |
| `their_initializer_normal_avg` | `nets.py:757-763` | **FAITHFUL** for the `dist='normal', fan='avg', scale=1.0` branch, `_fans` on a length-2 shape included |
| `_their_information_gain` | `epistemics.py:33-45` | **FAITHFUL as of 2026-09-05**, and the fix is the right one: `preds = [dist.sample(...)]` then `preds.std(0)` is now the port, where it had silently been `means.std(0)`. Its **harness** was not — `_ensemble_moments` hard-coded `range(HORIZON)` at the row's own indexing, so the test certified the leaf's formula and was blind to divergence 20's position. **FIXED 2026-09-07**: `_ensemble_moments` takes the plan whose length decides how many states are scored, and `_their_igfn_from_the_call_site` ports `igfn = lambda s: epistemic_module(s)[1:]` itself, so both indexings come out as two slices of one array and the position is inside the test's field of view |
| `reference` in `test_self_revision.py` | `callbacks.py:31-59` | **FAITHFUL**, verbatim including the redundant `neg_rates` line and the mutation order |

**Summary.** Thirteen of fourteen ports are faithful to their current source. Two carried
qualifications worth acting on, and **both were acted on 2026-09-07**: the
`their_dist_normal` harness pinned `winit: normal` on layers their code draws uniformly
(cosmetic, but it is a test asserting a false fact), and `_their_information_gain`'s harness
was written at our indexing rather than theirs, which is the same *shape* of mistake the
`means.std(0)` episode was — the port is right and the thing it is fed is not.

**Two of fourteen, and that is the finding.** Both defects are in the *surroundings* of a
faithful port, not inside it, so the general rule is: **a transcription test must be built
from their call site**, with the caller's indexing and argument order transcribed as
literally as the expression, not just the expression. Written up as a dated subsection of
`amortization.md` §(a″), which is where the transcription method lives.

## 12. NEEDS-TRANSCRIPTION-TEST — the open list

Nine rows above carry this disposition. Each was checked by reading in this session and each
**matched**; the point of the row is that nothing enforces it, and the two episodes that
motivated `test_oracle_transcription.py` were both cases where reading had said "matched".
Ranked by what a silent divergence there would cost.

1. **`GRUCell` against `nets.py:209-218`** (row 2.7). The carry update is the world model's
   spine, and the *joint* LayerNorm over the 3·deter vector before the reset/candidate/update
   split is exactly the placement a reimplementation gets wrong.
2. **The CRSPP loss assembly against `agent.py:223-231`** (row 4.11, with 4.5–4.10 as its
   pieces): the ρ gate folded into the dyn/rep scales, the signed contrast term, and the two
   means added rather than one mean of a sum.
3. **`ensemble_loss`'s reduction against `epistemics.py:50-61`** (row 5.5): the per-member
   `Independent(Normal, 1).log_prob(...).mean()` — a sum over 1024 dimensions and a mean over
   `(B, T−1)` — the `.sum()` over members, and the `nolearn.clip(1e-8)` gate. The `.sum()`
   was a real divergence as recently as 2026-09-01 and nothing pins it.
4. **`symlog_mse_nll` against `SymlogDist.log_prob`** (row 2.17), including the `tol = 1e-8`
   zeroing ours omits — divergence 27.
5. **`binary_nll` against `tfd.Bernoulli(logits).log_prob`** (row 2.19).
6. **`unimix_logits` against `_stats`' unimix block** (`nets.py:235-239`, row 2.9): mixing in
   *probability* space and returning `log(probs)`, not mixing logits.
7. **The advantage fold against `agent.py:448-462`** (row 6.31): `(normed_ret − normed_base)`
   times `scales[key]/total`, then `stack(advs).sum(0)`.
8. **`Gaussian.entropy` against `Independent(Normal, 1).entropy()`** (row 6.34) — the
   actor's `actent` term, summed over action dimensions.
9. **`cosine_sim`'s guards against `agent.py:165-179`** (row 4.9): theirs floors each squared
   norm at `1e-8` and then the product at `1e-8`; ours floors only the product, at `1e-12`.
   Inert at these norms, which is exactly the claim a test would make permanent.

Three further formulas were checked by reading and are dispositioned MATCHED rather than NTT
because what was checked is *order*, which the trace settles directly: `obs_step`'s prior-
then-posterior sequence (rows 2.3, 2.10), the optimizer chain's clip → adam → `−lr`
composition (rows 3.3–3.5), and the position of `SlowUpdater` after the critic's step
(row 7.10).

## 13. What this audit could not cover

- **Their float16 path.** The trace ran at `float32`, so `Optimizer`'s loss scaler,
  `apply_if_finite` and conditional step counter (row 3.7) were never entered. They are
  dispositioned from reading `jaxutils.py:391-449` alone.
- **The cross-call carry's *magnitude*.** Its mechanism is fully measured (row 1.5): call 2
  enters call 1's `(post[:, -1], action[:, -1])` and skips the initial-state prologue. How much
  flag 7's consequence actually costs — a stale latent from an unrelated window seeding a
  64-step scan — needs two calls on *correlated replayed* batches, and the trace's batches were
  synthetic. That measurement belongs to the `flag7-flat-replay` worktree, not here.
- **Formula-level equality.** By construction: §12 is that boundary, made explicit. A trace
  shows `lambda_return` was called with three arrays of the right shapes; it cannot show the
  recursion is right.
- **The act path, the environment and its wrappers, replay writing and routing, counters.**
  Out of scope by assignment.
- **`report()`.** Their `WorldModel.report` runs a second world-loss forward pass, an
  `observe` over `[:6, :5]`, and an `imagine_no_action` — every 300 seconds. It is not part of
  a train step, but it consumes RNG and, in `report`, calls `rssm.observe` with the
  **unshifted** `data['action']` (`agent.py:291-293`), which is inconsistent with the training
  path's shift. Noted, not dispositioned: it is a diagnostic on their side.
