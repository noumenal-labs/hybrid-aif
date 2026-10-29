# Actor trace audit — the 608, and the actor's gradient path

**Status: build-time audit, not an adjudication.** Nothing here is a decision. Every
row is either a measurement or a reading of the two sources, and the two findings
that would change code are written as findings, not as proposals.

**Date:** 2026-09-16. **Ours:** `hybrid-aif` @ `8e321d8`, working tree clean.
**Theirs:** `raif-paper` @ `9b1864e8`, the oracle. Doctrine: the oracle is the code.

**Method.** Job 1 is analytic: their `model_opt`'s membership enumerated from
`agent.py:182`, each member's parameter count computed from the resolved config
(`logs/oracle_state_mc_seed0/config.yaml`), laid beside ours counted leaf by leaf out
of `raif_mountain_car(PRNGKey(0))`. Job 2 was done by **executing their code**, not by
reading it: a scratch venv (Python 3.11, tfp 0.25, jax 0.4.30, two stub modules for
`stable_baselines3`/`sb3_contrib` so their `embodied` imports) drives their real
`ImagActorCritic.loss`, `GFunction.score`, `Moments`, `SlowUpdater` and `Dist('normal')`
under `jax: {jit: False}` on a synthetic trajectory, dumping every intermediate; our
**installed** functions are then run on those same arrays and diffed. Where a check is
a reading rather than an execution the row says so. Neither repository was modified;
all instrumentation lives in the session scratch directory. The live 2M run in
`runs/mc_2d` was not touched.

Divergence 20 (info-gain indexing, ours kept), the cosmetics (23, 24, 25, 27, 28, D7),
and the CRSPP→encoder seam (ledger 5, measured inert in Arm A) are argued and are not
re-litigated here. Divergence 29's fix is re-verified as a side effect of the
`perc_ema` and slow-critic rows, because those are the rows it lived in.

---

## Job 1 — the 608, part by part

### 1.1 What their `model_opt` contains

`WorldModel.train` (`agent.py:182`) passes
`modules = [self.encoder, self.rssm, *self.heads.values()]`. The CRSPP is **not** a
separate module on their side: `img_in_2`, `gru_2`, `img_out_2`, `img_stats_2`,
`obs_out_2` and `obs_stats_2` are `self.get(...)` calls inside the one `CRSPP_RSSM`
object (`nets.py:151-201`), so they land in `rssm` and therefore in `model_opt`. That
makes their 513,314 and our 513,922 comparable as like for like: both are
world model + CRSPP, and neither includes the info-gain ensemble, the actor or the
critic.

Resolved sizes: `rssm {deter: 128, stoch: 32, classes: 32, units: 32, norm: layer,
winit: normal, fan: avg, initial: learned}`; encoder/decoder `{mlp_layers: 2,
mlp_units: 64}`; `reward_head`/`cont_head` `{layers: 2, units: 32}`; obs width 2,
action width 1, embed width 64. Two rules do all the arithmetic. Their `Linear` drops
its bias wherever `norm != 'none'` (`nets.py:664`), so every hidden linear in a normed
block costs `in·out + 2·out` (kernel plus the LayerNorm's scale and bias) and no bias.
Their `_stats` linears are built as `self.get(name, Linear, units)` with **no** `**kw`
(`nets.py:233,243`), so they get `norm='none'` and therefore **do** keep a bias.

### 1.2 The reconciliation, residual zero

| their module (`nets.py` / `agent.py`) | shape | theirs | ours (slot) | ours | Δ |
|---|---|---|---|---|---|
| `enc/mlp/h0` | 2→64 +norm | 256 | `recognition["embed"]` layers[0]+norms[0] | 256 | 0 |
| `enc/mlp/h1` | 64→64 +norm | 4,224 | same, layers[1]+norms[1] | 4,224 | 0 |
| `rssm/initial` | (128,) | 128 | `initial_state_prior[0]` `LearnedVector` | 128 | 0 |
| `rssm/img_in` | 1025→32 +norm | 32,864 | `transition[0].map.input_proj` + `norms[0]` | 32,864 | 0 |
| `rssm/gru` | 160→384 +norm | 62,208 | `transition[0].map.gate` + `norms[1]` | 62,208 | 0 |
| `rssm/img_out` | 128→32 +norm | 4,160 | `transition[1].trunk.layers[0]` + `norms[0]` | 4,160 | 0 |
| `rssm/img_stats` | 32→1024 +bias | 33,792 | `transition[1].trunk.layers[1]` | 33,792 | 0 |
| `rssm/obs_out` | 192→32 +norm | 6,208 | `recognition["stoch"].trunk.layers[0]`+`norms[0]` | 6,208 | 0 |
| `rssm/obs_stats` | 32→1024 +bias | 33,792 | `recognition["stoch"].trunk.layers[1]` | 33,792 | 0 |
| `rssm/img_in_2` | 1024→32 +norm | 32,832 | CRSPP `transition[0].map.input_proj`+`norms[0]` | 33,248 | **+32** |
| `rssm/gru_2` | 160→384 +norm | 62,208 | CRSPP `transition[0].map.gate` + `norms[1]` | 62,592 | **+384** |
| `rssm/img_out_2` | 128→32 +norm | 4,160 | CRSPP `transition[1].trunk.layers[0]`+`norms[0]` | 4,192 | **+32** |
| `rssm/img_stats_2` | 32→1024 +bias | 33,792 | CRSPP `transition[1].trunk.layers[1]` | 33,792 | 0 |
| `rssm/obs_out_2` | 192→32 +norm | 6,208 | CRSPP `recognition["stoch"]` layers[0]+norms[0] | 6,240 | **+32** |
| `rssm/obs_stats_2` | 32→1024 +bias | 33,792 | CRSPP `recognition["stoch"].trunk.layers[1]` | 33,792 | 0 |
| — (theirs shares `rssm/initial`) | — | 0 | CRSPP `initial_state_prior[0]` `LearnedVector` | 128 | **+128** |
| `dec/mlp/h0` | 1152→64 +norm | 73,856 | `likelihood[0]` layers[0]+norms[0] | 73,856 | 0 |
| `dec/mlp/h1` | 64→64 +norm | 4,224 | same, layers[1]+norms[1] | 4,224 | 0 |
| `dec/mlp/dist_state/out` | 64→2 +bias | 130 | same, head layer | 130 | 0 |
| `rew/h0`, `rew/h1` | 1152→32, 32→32 +norm | 38,016 | `likelihood[1]` trunk | 38,016 | 0 |
| `rew/dist_out/out` | 32→255 +bias | 8,415 | same, head layer | 8,415 | 0 |
| `cont/h0`, `cont/h1` | 1152→32, 32→32 +norm | 38,016 | `likelihood[2]` trunk | 38,016 | 0 |
| `cont/dist_out/out` | 32→1 +bias | 33 | same, head layer | 33 | 0 |
| **total** | | **513,314** | | **513,922** | **+608** |

Their column sums to exactly the 513,314 their `metrics.jsonl` reports, and ours to
exactly the 513,922 measured from the built agent, so the decomposition is checked at
both ends. **The residual is zero and every one of the 608 is in the CRSPP.**

### 1.3 The two discrepancies, characterised

**(a) 480 — four biases the CRSPP keeps and theirs drops. Ours has the extra
parameters.** Mechanism: their `Linear` sets `self._bias = bias and norm == 'none'`
(`nets.py:664`), so a normed hidden linear has no bias at all. Ours reproduces that
rule, but only under `winit="normal"`:
`distributions/neural.py:1624` is `use_bias = winit == "default" or not norm`, and
`_mlp_layers` at `:778-781` applies the same rule. The world model gets it because
`raif_mountain_car` defaults `winit="normal"` (`reference/raif_agent.py:216`) and
threads it into `tiny_rssm` (`:252`). **`crspp_rssm` is called two lines later
(`:256-264`) and is never given `winit`**, and `crspp_rssm` has no `winit` parameter to
give (`reference/worlds.py:147-159`), so every CRSPP part is built at
`winit="default"`. The four biases are `img_in_2` (32), `gru_2` (384), `img_out_2` (32)
and `obs_out_2` (32). Measured directly off the built pytree: the world model's
`transition[0]` has no `input_proj.bias` or `gate.bias`; the CRSPP's has both.

This is an **architectural** divergence, not bookkeeping — the parameters exist, are
initialized, receive gradient, and are stepped by `fit_crspp`. Whether it matters on
its own: probably not much. A bias in front of a LayerNorm is in the normalizer's null
space for the *value* (the norm subtracts the mean and the bias shifts every unit of a
post-linear vector by a different constant, so it is not exactly null — only a uniform
bias is), and their own reason for dropping it is that the norm's own `bias` already
provides the shift. The `gru_2` case is the one where it is not null at all: their
`_gru_2` normalizes the 384-wide pre-activation and *then* splits it into
reset/candidate/update, so a per-unit bias there is a real, trainable offset on three
gate pre-activations that theirs does not have.

**(b) 128 — the CRSPP's own learned initial carry. Ours has the extra parameters, and
it is known and deliberate.** Already on the record as `execution-ledger-training.md`
rows 4.1 and 4.14. Restating only the part that bears on the count: theirs has no
CRSPP-specific initial at all. `obs_step_no_action` resets to `self.initial(...)`
(`nets.py:156-157`), which is the **world model's** learned `initial` vector pushed
through the **world model's** `img_out`/`img_stats` (`nets.py:57-59`), so the CRSPP's
episode start is the world model's and its gradient flows into the world model's prior
head. Ours has `parts.initial()` plus `parts.prior.mode((deter0,))` — the CRSPP's own
vector through the CRSPP's own prior. So the 128 is one half of a structural difference
whose other half (whose prior head the start runs through, and where that gradient
goes) carries no parameter-count signature at all.

### 1.4 The finding hiding behind the 480

The bias count is a *visible* symptom of `crspp_rssm` never receiving `winit`. The
invisible symptom is the weight draw, which costs no parameters and so does not show up
in the 608 at all: **the CRSPP's six weight matrices are drawn with `eqx.nn.Linear`'s
default `U(±1/√fan_in)` rather than their `winit: normal / fan: avg`.** Measured
standard deviations of the built weights, against the oracle's realized draw (both
their `normal` and their `uniform` branches realize std `√(1/mean(fan_in, fan_out))` —
that is the whole content of divergence 26b, that they have the same second moment and
different shapes):

| CRSPP layer | fan_in | fan_out | ours, measured | theirs, realized | theirs / ours |
|---|---|---|---|---|---|
| `img_in_2` (`input_proj`) | 1024 | 32 | 0.0180 | 0.0435 | 2.42× |
| `gru_2` (`gate`) | 160 | 384 | 0.0455 | 0.0606 | 1.33× |
| `img_out_2` (`trunk[0]`) | 128 | 32 | 0.0508 | 0.1118 | 2.20× |
| `img_stats_2` (`trunk[1]`) | 32 | 1024 | 0.1024 | 0.0435 | 0.43× |
| `obs_out_2` (`trunk[0]`) | 192 | 32 | 0.0414 | 0.0945 | 2.28× |
| `obs_stats_2` (`trunk[1]`) | 32 | 1024 | 0.1019 | 0.0435 | 0.43× |

The world model's corresponding layers were measured in the same run and land on their
values to three figures (`input_proj` 0.0438 vs 0.0435, `gate` 0.0605 vs 0.0606,
`trunk[0]` 0.1116 vs 0.1118, `trunk[1]` 0.0436 vs 0.0435), so the adoption of 17+26 is
complete *on the six blocks it was scoped to* and stops exactly at the CRSPP boundary.

Whether this plausibly matters: the direction is not uniform — the CRSPP's fan-in-heavy
layers start 2.2–2.4× **narrower** than theirs and its two fan-out-heavy `_stats`
layers 2.3× **wider**, which is a systematically different signal-to-noise profile
through the preference model's scan at step zero. It has a direct route into the
actor's objective, because the CRSPP's rolled-out states are one side of the
`cosine_sim_to_pref_rollout` leaf of G, and `series-sweep.md` §5.2 already has that
leaf outside the oracle's envelope in 16 of 16 bins, "2.4× at 0–4k before any
behavioural difference exists". This audit does not claim the init explains that; it
records that the leaf's own model is initialized differently from the oracle's and that
nobody had looked.

---

## Job 2 — the actor's gradient path, operation by operation

The rollout is 8 steps (`imag_horizon: 8`), so their `traj` has 9 entries and the
actor's loss arrays have 8; ours are `(N, 8)` with `N = batch_size · batch_length =
1024` on both sides (their `train_ratio 64` with `batch_steps 1024`; our `run_mc`
`batch_size · seq_len = 1024` every `train_every = 16` pooled steps). Rows marked
**measured** were checked by running their code and diffing against ours; rows marked
**read** were established by reading both sources.

### 2.1 Disposition ledger

| # | their operation | ours | disposition | note |
|---|---|---|---|---|
| **A. the imagination rollout as the actor sees it** ||||
| A1 | `agent.py:431-436` the whole rollout + score lives inside the function `nj.grad` differentiates wrt the actor only | `imagine.py:456-522` `loss_fn` under `eqx.filter_grad(...)((actor, critic))` | MATCHED | read; the world model's params are closure constants on both sides, and the chain rule still runs *through* them |
| A2 | `agent.py:432` `policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())` | `imagine.py:599-600` `actor_m.condition_on((sg(feat),)).sample(act_key)` | MATCHED | read; divergence 12's stop-gradient on the actor's *conditioning* state, at every step |
| A3 | `agent.py:247` `start['action'] = policy(start)` before the scan | `imagine.py:592-600` first scan iteration acts at `s_0` | MATCHED | read; `traj['action'][k] = π(s_k)` on both sides |
| A4 | `agent.py:250` `img_step(prev, prev.pop('action'))` | `imagine.py:614-616` `_sampled_step(..., soft_clip(action, bounds), ...)` | MATCHED | read; the transition sees the rescaled action, every other reader the raw draw |
| A5 | `nets.py:170-172` `prev_action *= sg(clip / max(clip, abs(a)))` | `controls.py:224-257` `soft_clip` | MATCHED | **measured**: value `[0.3, 1.0, 1.0, −1.0]` and gradient `[1, 1, 0.5, 0.25]` on `[0.3, 1, 2, −4]`, i.e. `1/|a|` outside the box, not zero |
| A6 | `nets.py:184-186` `dist.sample` of the prior, straight-through (`jaxutils.py:105-108` `sg(sample) + (probs − sg(probs))`) | `neural.py:982-995` `straight_through_onehot` | MATCHED | read; identical estimator, identical `softmax` of the unimixed logits |
| A7 | `nets.py:235-239` unimix 0.01 mixed into the prior's probabilities before the log | `neural.py:963-979` `unimix_logits`, applied in `MLPCategorical.logits` | MATCHED | read |
| A8 | `agent.py:256` `cont = self.heads['cont'](traj).mode()` — a hard 0/1 | `imagine.py:623` `_cont_mode` | MATCHED | divergence 10, closed; transcription test at `test_oracle_transcription.py:120` |
| A9 | `agent.py:257-259` `traj['weight'] = cumprod(discount·cont)/discount`, `first_cont` leading | `imagine.py:228-249` `rollout_weight` | MATCHED | **measured**: max abs diff 2.1e-08 against their real array |
| A10 | `agent.py:244` `first_cont = 1 − start['is_terminal']` (replay data) | `raif_agent.py:549` `start_cont=batch.cont` | MATCHED | read; `Episode.cont` is `1 − is_terminal` |
| A11 | `agent.py:89` starts flattened over all `B·T = 1024` posterior states, codes straight-through sampled | `raif_agent.py:530-534` + `train/world.py:328-356` | MATCHED | read; both start from a *sampled* posterior, not a mean-field one |
| A12 | `agent.py:433` `pref_traj = imagine_preference(C_start, H)`, with `img_step_no_action` **sampling** the preferred code (`nets.py:199`) | `objectives/amortized.py:443-445` rolls the CRSPP with keyless `update_empirical_prior`, mean-field | DIVERGES (already recorded, ledger row 6.5) | read; see §2.3 rank 2 |
| **B. the advantage** ||||
| B1 | `agent.py:557-564` `G = 1.0·rew + 1.0·crspp + 1.0·(−ig)` | `imagine.py:252-288` `term_value`, signs from `objectives/amortized.py:77-81` | MATCHED | read; registered signs give `+rew +crspp −ig`; `g_tail` is in `terms` but is a terminal row and is absent from the per-step fold (verified: `per_term` keys are exactly the three) |
| B2 | `agent.py:356` `rewfn = ...heads['reward'](s).mean()[1:]` — leaf `k` is the reward at `s_{k+1}` | `imagine.py:618` row evaluated at `new` = `s_{k+1}` | MATCHED | read; divergence 20's shift does *not* touch this leaf, only the one whose parents include the action |
| B3 | `agent.py:357` `crsppfn = lambda s, p: wm.crspp_traj_sim(s, p)[1:]` — leaf `k` is `cos(traj_{k+1}, pref_{k+1})` | `objectives/amortized.py:455-461` `theirs = preferred[ctx.t]` where `preferred = rolled[1:]`, `ours = features(new)` | MATCHED (indexing) | read; both compare step `k+1` against preferred step `k+1`. §5.2 of `series-sweep.md` asked for exactly this check; the *indexing* is right and the divergence is A12's, not an off-by-one |
| B4 | `agent.py:229,282` `cosine_sim(traj, sg(pref_traj))` — the preferred side carries no gradient | `objectives/amortized.py:441-449` `preferred` is loop-invariant, built from `aux_beliefs` constants | MATCHED | read |
| B5 | `agent.py:571` `value = self.net(traj).mean()` — the **fast** critic, live through `traj` | `imagine.py:579,622,632` `critic_sg = sg(critic_m)`, applied to live features | MATCHED | **measured**: `|∂(actor loss)/∂critic| = 0.0` exactly, while `|∂(mean ret)/∂actor| = 0.086 > 0`, so the critic's *parameters* are cut and the pathwise route through the world model is live — theirs has the same two properties |
| B6 | `jaxutils.py:195-199` `DiscDist.mean = symexp(Σ probs·bins)`, 255 bins on [−20, 20] | `neural.py:1308-1310` `symlog_disc_mean` | MATCHED | read; `outscale: 0.0` makes a fresh critic predict exactly 0 on both sides (**measured**: their `base` summed to 0.000000; our `critic_value` is −2.4e-07) |
| B7 | `agent.py:569-576` the λ-return loop, bootstrapped at `value[-1]`, `disc = traj['cont'][1:]·discount` | `imagine.py:291-312` `lambda_return` | MATCHED | **measured**: 3.6e-07 against their real recursion on their real arrays |
| B8 | `agent.py:452` `offset, invscale = self.retnorms[key](ret)` — `Moments.update` then `stats`, percentiles over the whole array, no bias correction, `invscale = max(1/max, hi−lo)` with `max = 1.0` | `imagine.py:315-330` `perc_ema`, state from `LearnState.moments[RETNORM_KEY]` | MATCHED | **measured**: their `Moments` driven for six consecutive calls and diffed against our carry-threaded `perc_ema`, max abs diff 3.0e-08 on `(offset, invscale)`; a single call also matches (`offset` 6.7e-10, `invscale` exact) |
| B9 | `agent.py:453-455,462` `adv = ((ret−off)/inv − (base−off)/inv) · scale/total` | `imagine.py:480` the same two terms; one critic, `scale/total = 1` | MATCHED | **measured**: 0.000e+00 |
| **C. the actor loss, term by term** ||||
| C1 | `agent.py:467` `loss = -adv` (`actor_grad_cont: backprop`) | `imagine.py:487` `per_step = -advantage - ...` | MATCHED | read; the reparameterized branch, not REINFORCE |
| C2 | `agent.py:469` `loss -= actent · ent`, `actent = 3e-4` | `imagine.py:487` | MATCHED | measured within C5 |
| C3 | `agent.py:470` `loss *= sg(traj['weight'])[:-1]` — multiplies the advantage **and** the policy entropy | `imagine.py:486-489` `jnp.mean(per_step * weight)` | MATCHED | measured within C5; the parenthesization is the content |
| C4 | `agent.py:474` `loss -= wment · wment_ent`, **after** the weight multiply, so unweighted | `imagine.py:490` a separate `- wment * jnp.mean(prior_entropy)` | MATCHED | algebraically identical because the term is unweighted and both reduce by an unweighted mean over the same-shaped array |
| C5 | `agent.py:482-484` `loss *= loss_scales.actor` (1.0); `return loss.mean() + actor_refresh.mean()` | `imagine.py:488-492` | MATCHED | **measured**: our `imagine`'s assembly, fed their `adv`/`ent`/`wment`/`weight`/`logpi2`/`can_self_imitate`, reproduces their scalar to 3.0e-08 |
| C6 | `agent.py:466` `logpi = policy.log_prob(sg(traj['action']))[:-1]` | not computed | N-A-BY-DESIGN | under `backprop` it is a metric only (`train/policy_logprob`); ours logs no such series — a coverage gap, not a divergence |
| **D. the entropies** ||||
| D1 | `nets.py:576-577` `tfd.Independent(tfd.Normal(tanh(out), std), 1)` — entropy **summed** over the action dims, and `tanh` does not enter it (a Normal's entropy depends on σ alone) | `neural.py:478-490` `Gaussian(loc=tanh(mean), chol=diag(scale))`, `entropy().midpoint` | MATCHED | **measured**: 8.7e-08 over a `(9, 7, 1)` array of their real means and stds. Confirmed summed, not meaned: their `event_shape` printed as `(1,)` and `batch_shape` as `(9, 5)` |
| D2 | `nets.py:575` `std = (maxstd−minstd)·σ(raw + 2.0) + minstd`, `(0.1, 1.0)` | `neural.py:473-476` the `bounded` head | MATCHED | **measured**: their whole actor — `h0/h1/h2` kernels, three LayerNorm scale/bias pairs, and the two head linears — transplanted into our `MLPConditional`; mean agrees to 7.5e-07 and scale to 1.4e-07 on their real `(9, 7, 1152)` features. That pins the layer order (linear → norm → silu), the LayerNorm `eps = 1e-3`, the absent hidden biases, the fused head's two halves, `tanh`, and `σ(raw+2)` in one number |
| D3 | `agent.py:468` `ent = policy.entropy()[:-1]` — at `s_0 … s_{H−1}`, aligned with `weight[k]` | `imagine.py:625` `policy.entropy().midpoint` inside the scan, emitted at `s_0 … s_{H−1}` | MATCHED | read — and see §2.4: no test pins this alignment, and it is the one index in the loss that is *not* invisible to the arithmetic |
| D4 | `agent.py:473` `wment = self.get_dist(traj).entropy()[1:]` — `Independent(OneHotDist(logit), 1)`, i.e. summed over the 32 variables, at `s_1 … s_H` | `imagine.py:624` `factored_categorical_entropy(logits)` | MATCHED | **measured**: 1.4e-14 against their `Independent(OneHotDist, 1).entropy()` on a shared `(4, 3, 32, 32)` logit array. This is the mean-not-sum divergence that was 1024× at `tiny`; it is closed |
| **E. the self-imitation term** ||||
| E1 | `agent.py:477-478` `policy2 = self.actor(sg(start)); logpi2 = policy2.log_prob(sg(start['action']))` | `imagine.py:640` `actor_m.log_prob(expert, (features(start),))` | MATCHED | read; `start` is a constant on both sides so the `sg` is a no-op |
| E2 | `agent.py:478` `start['action']` is `data['action']` at the *same* index as the posterior start, and `agent.py:193-194` shows `data['action'][t]` is the action taken **at** `t` | `raif_agent.py:542` `start_actions=jnp.reshape(batch.action, (n, -1))` — the `a_t` column, unshifted | MATCHED | read; the alignment is right, but the docstring that explains it is wrong — see §2.5 defect 2 |
| E3 | `agent.py:479-480` `actor_refresh = −logpi2 · sg(can_self_imitate) · 0.5` | `imagine.py:481,491` `bc = −start_logprob · flags`, `+ bc_scale · mean(bc)` | MATCHED | measured within C5 |
| E4 | `agent.py:484` `actor_refresh.mean()` over `(B·T,)`, added outside `loss_scales.actor` | `imagine.py:491` | MATCHED | measured within C5 |
| **F. the optimizer and the slow critic** ||||
| F1 | `jaxutils.py:371-389` `chain(clip_by_global_norm(100), scale_by_adam(eps=1e-5), scale(−lr))`, `lr = 3e-5` | `imagine.py:526-534` `_clip(actor_grad, 100)` then `apply_gradient(..., lr, optimizer=adam(1e-5))` | MATCHED | read; `optax.adam(1.0, eps)` is `scale_by_adam ∘ scale(−1)` and the `lr` multiply afterwards makes the composite `−lr · adam_update`, identical. `b1/b2/eps_root` default identically on both sides |
| F2 | `jaxutils.py:391-393` `apply_if_finite` + loss scaling, **only** when `COMPUTE_DTYPE == float16` | no analogue | N-A-BY-DESIGN | the float16 row of the register; measured and demoted 2026-09-15 |
| F3 | `jaxutils.py:409` one `nj.grad` per optimizer, clipping the actor's own pytree | `imagine.py:524-528` one `filter_grad` over the pair, then two independent clips | MATCHED | read; the two objectives are separated by stop-gradients (`imagine.py:496-498`) so a joint `filter_grad` gives the same two gradients as two separate ones — confirmed by B5's measured zero |
| F4 | `agent.py:522` critic target is a **second** `score` call after the actor's step | one `ret`, reused | MATCHED (equivalent) | read; the second call is on the same `traj` at the same, not-yet-stepped, critic, and `score` is deterministic — divergence 23, confirmed cosmetic rather than merely asserted |
| F5 | `jaxutils.py:483-495` + `agent.py:525` `SlowUpdater()` called **after** the critic's step, `mix = clip(1·need_init + 0.02·need_update, 0, 1)` | `imagine.py:541-543,867-892` `_slow_update(slow, new_critic, ...)` | MATCHED | **measured**: our mixer's first call is a hard copy (`mix = 1.0`) and its counter advances, which is divergence 13's `need_init`; the transcription test at `:331` drives eight steps against their expression |
| F6 | the slow critic never enters the actor's loss | `imagine.py:498` used only in `critic_loss` | MATCHED | read |

Thirty-eight rows. **Thirty-six MATCHED, one DIVERGES (A12, already on the record),
one N-A-BY-DESIGN each for float16 and the unused `logpi` metric.** Fourteen of the
matched rows are matched by *execution* against the oracle's own output rather than by
reading.

### 2.2 The measured diff, in full

Every number below is our installed function run on an array their code produced, with
`jit: False`, at float64 on our side and float32 on theirs (so ~1e-07 is the floor).

| quantity | max abs diff |
|---|---|
| λ-return over 8 steps × 7 trajectories | 3.609e-07 |
| `traj['weight'][:-1]` | 2.073e-08 |
| `(offset, invscale)` over six consecutive `Moments` calls | 2.967e-08 |
| `offset` / `invscale` / `adv` on the real return array | 6.7e-10 / 0.0 / 0.0 |
| the whole actor loss scalar | 3.028e-08 |
| policy entropy, `(9, 7, 1)` | 8.686e-08 |
| policy log-prob, `(9, 7)` | 6.680e-07 |
| `wment` entropy vs `Independent(OneHotDist, 1)` | 1.421e-14 |
| actor head mean / scale, their weights transplanted | 7.5e-07 / 1.4e-07 |

### 2.3 DIVERGES, ranked against the two standing questions

The two standing questions are **(Q1)** "their actor commits from a wide start
(entropy 1.140 → 0.081 within 12k) and ours does not (0.320)" and **(Q2)** "our
seed-to-seed variance is 10× theirs under every configuration tried".

**Rank 1 — the CRSPP's initializer and biases (Job 1, §1.3a/§1.4).** Bears on Q1 and
Q2: **possibly, indirectly, through the advantage.** The CRSPP is one side of the
`cosine_sim_to_pref_rollout` leaf of G, G is what the λ-return folds, and the advantage
is the actor's only signal with any strength (measured: the advantage's pathwise
gradient to the actor is ~190× the `actent` term's after weighting). A preference model
whose scan starts 2.2–2.4× narrower on its fan-in-heavy layers and 2.3× wider on its
`_stats` layers produces a different preferred trajectory at step zero, and the
similarity leaf is already the one series outside the oracle's envelope in every bin.
It cannot be asserted to cause either question — the sign and magnitude of the effect
on the leaf are unmeasured — but it is the only *unknown* divergence this audit found
on the advantage's construction, and it is cheap to measure (a read-only probe on any
existing checkpoint, exactly as `series-sweep.md` §5.2 costed it).

**Rank 2 — the preferred rollout is mean-field where theirs is sampled (A12, ledger row
6.5).** Bears on Q1 and Q2: **plausibly on the leaf's level, not obviously on either
question.** Already recorded, so this is a re-statement with the actor's perspective
added. Their `img_step_no_action` samples, so both sides of their cosine are one-hot
codes with flat-stoch norm `√32`; ours compares a one-hot trajectory against a
*probability* trajectory whose flat-stoch norm is far smaller, which systematically
re-weights `deter` against `stoch` inside the cosine. The row is recorded as "the
cosine row's *level* is not comparable to theirs". What is new here is that the leaf's
level enters the actor's advantage, not merely a diagnostic — so a level shift is a
constant added to every `G[k]`, which the λ-return turns into a roughly constant offset
on `ret` **and** on `base` (the critic learns it), hence largely cancelled in
`ret − base`. That reasoning is why this ranks second and not first: a pure level shift
is absorbed by the baseline. A *shape* shift is not, and nobody has measured which this
is.

**Nothing else.** No divergence in the loss assembly, the advantage's normalization,
either entropy, the weight, the BC term, the optimizer, or the slow critic could
produce either phenomenon, because there is no divergence in any of them: they agree
with the oracle's own output to float32 precision. Stated plainly, as the brief asks:
**the actor's forward pass, its loss, and its gradient topology are faithful, and this
audit does not find a cause for Q1 or Q2 inside them.** That is a result. It moves the
remaining probability mass onto (i) the inputs to the advantage — the G leaves and the
world model that produces the rollout — and (ii) the act-time side, which §5.3 of
`series-sweep.md` had already nominated by exhaustion.

One caveat that bears on how Q1 is *measured*, offered as a caveat and not a finding.
Our `act_entropy` (`run_mc.py:497-530`) reads the actor at **mean-field** act-time
features, theirs at the sampled posterior from `obs_step` (divergence D6, adjudicated
"stands"). The actor's σ is a function of those features, and `plateau-analysis.md`
§4.1 measured `σ_act / σ_train` at 0.91–1.01 across seeds and checkpoints. So up to
about 0.09 nats of the 0.24-nat gap between our 0.320 and their 0.081 could be the
feature path rather than the actor — at 775k, which is not where the gap is being read.
Worth pricing at 12k before the 0.320 is used as an actor property.

### 2.4 Transcription-test verdicts

Every port in `tests/test_oracle_transcription.py` that touches this path was checked
against their **current** source at `9b1864e8`, line by line, and then independently
against real oracle output.

| test | their lines it claims | verdict |
|---|---|---|
| `their_score_return` / `:98` | `agent.py:569-576` | **FAITHFUL.** Expression correct; caller assumption correct too — the port's `cont` argument is `[first_cont, c_1 … c_H]` and `disc = cont[1:]·discount` reproduces `traj['cont'][1:]`, where `traj['cont'] = concat([first_cont, cont_head[1:]])`. Independently confirmed at 3.6e-07 against the oracle's own `score` |
| `:120` cont mode | `agent.py:256` | **FAITHFUL.** Also asserts the zero gradient, which is the half that matters |
| `their_moments_perc_ema` / `:162` | `jaxutils.py:307-310, 338-341` | **FAITHFUL**, and the divergence-29 lesson is now covered at *two* levels: the port threads `(low, high)` by hand in a Python loop (which is what could not see a caller dropping the carry), and `test_raif_acceptance.py:506-544` plus `test_amortized_trainers.py:352-401` now assert the carry at the trainer seam. Independently confirmed at 3.0e-08 against their real `Moments` over six calls |
| `their_weight` / `:242` | `agent.py:257-259` | **FAITHFUL.** The port's `cont` is already the `[1:]` slice and the docstring says so |
| `their_actor_loss` / `:263` | `agent.py:463-484` | **FAITHFUL** to the expression and to the parenthesization, which is what it exists to pin. **One caller assumption it cannot see** (see below) |
| `:296` weight-not-absorbable | — | **FAITHFUL**; keeps divergence 11 from returning quietly |
| `their_slow_updater` / `:331` | `jaxutils.py:483-495` | **FAITHFUL.** Caller assumption (the source is the *freshly stepped* critic, `agent.py:523-525`) holds in `imagine.py:541` |
| `:643` bounded head | `nets.py:511-530, 553-580` | **FAITHFUL**, and now independently confirmed by transplanting the oracle's own trained-shape actor weights: 7.5e-07 on the mean, 1.4e-07 on the scale |
| `:738`, `:795` initializers | `nets.py:740-786` | **FAITHFUL** for the layers they cover. They do **not** cover the CRSPP, which is §1.4's finding — no test asserts anything about how `crspp_rssm` draws its weights |

**NEEDS-TRANSCRIPTION-TEST (one).** `their_actor_loss` takes `adv`, `ent`, `wment_ent`
and `weight` as given arrays, so it certifies the assembly but inherits an assumption
about *which step each row belongs to*. Three of the four are invisible to the
arithmetic — `adv` and `wment_ent` reduce by unweighted means, so a shift in either
would not move the scalar — but **`ent` is not**: it is multiplied element-wise by
`weight[k]` before the mean, so an off-by-one between the policy entropy and the
rollout weight changes the loss and no test would see it. Their alignment is
`ent = policy.entropy()[:-1]` (entropy at `s_0 … s_{H−1}`) against
`weight[:-1]` (the weight of steps `0 … H−1`); ours emits both from the same scan
iteration, which is correct — established here by reading and by the shape trace, not by
a test. The test wanted is one that gives `ent` a step-varying profile (say `ent[k] = k`)
and a `weight` with an early zero, and checks the scalar against their loop. Cheap, and
it closes the last place on this path where a faithful formula could sit on a wrong
caller assumption.

### 2.5 Defects in our code, regardless of parity

1. **`crspp_rssm` takes no `winit` and `raif_mountain_car` gives it none**
   (`reference/worlds.py:147-159`, `reference/raif_agent.py:256-264`). This is Job 1's
   whole 480 plus the invisible init divergence of §1.4. It reads as an oversight
   rather than a decision: `raif_mountain_car`'s own docstring at `:221-225` enumerates
   what `winit` covers — "the encoder, decoder, reward/cont heads, prior, posterior,
   GRU, critic and slow critic" — and the CRSPP, which has a prior, a posterior and a
   GRU of its own, is simply not in the list. The doc at `:231-234` says the CRSPP "is
   built at the *same* sizes", which is now true up to 608 parameters.
2. **`imagine`'s `start_actions` docstring describes a shift that does not happen**
   (`train/imagine.py:386-390`): "with `WorldBatch`'s `action[t] = a_{t−1}`, the action
   taken *at* the start indexed `t` sits at `t+1` (`raif_agent.train_step` does that
   once)". `train_step` passes `batch.action` — an `Episode`, `a_t` convention —
   unshifted, and its own comment at `raif_agent.py:539-541` says so explicitly ("no
   re-indexing is needed here"). The behaviour is right and matches theirs (ledger row
   E2); the docstring is wrong, and it is wrong about exactly the alignment a future
   auditor of the self-imitation term would check first. One-line fix.
3. Not a defect, recorded so the next comparison does not re-find it: their
   `stats/mean_log_entropy` is a per-episode mean (`callbacks.py:98`,
   `ep[key].mean()`) that their logger then averages **over the episodes completed in a
   log window**; ours is written per episode to `scores.jsonl`
   (`run_mc.py:1382`). The per-episode reduction is correct on both sides; whoever bins
   ours for comparison must average over episodes and not over steps, or long episodes
   will be over-weighted relative to theirs.

Nothing else was found. In particular the `bc_logprob` instrumentation defect that
`series-sweep.md` §6.1 and `plateau-analysis.md` §7.2 both record is **fixed at HEAD**:
`ImagineMetrics.bc_term` now carries `bc_scale · mean(−log π · can_self_imitate)`, the
flag-weighted quantity that is comparable to their `train/actor_refresh_mean`, and the
unweighted `bc_logprob` survives beside it with a docstring saying what it is not.

### 2.6 What this audit could not cover

- **The gradient itself, compared end to end.** Every value on the path was diffed
  against the oracle and the gradient's *topology* was probed on our side (which
  parameters receive gradient, and through which route), but no number here compares
  `∂L/∂θ_actor` between the two implementations. Doing so needs their world model and
  ours to hold the same weights, which is a port of the whole RSSM, not of the actor.
  The nearest available proxy is the head transplant (D2), which does compare a real
  gradient-carrying forward pass exactly.
- **The G leaves' values.** `rewfn` and `igfn` were replaced by deterministic
  stand-ins so the two sides could be fed identical arrays. Their content is the
  earlier ledgers' subject and divergence 20's; this audit covers only how the leaves
  are folded, indexed, discounted and normalized.
- **Anything at scale.** Their side ran at `BT = 5` and `BT = 7`, not 1024. Every
  quantity checked is either shape-invariant or reduced by a mean, so this is a
  precision limit rather than a coverage limit — but the percentile in `perc_ema` is
  the one operation whose behaviour depends on the sample size, and it was checked at
  `(8, 7)` and `(8, 64)`, not at `(8, 1024)`.
- **`float16`.** Their `COMPUTE_DTYPE` is set from `jax.precision` at runtime; every
  trace here ran at their `jaxutils.COMPUTE_DTYPE` default of float32, so the loss
  scaler, `apply_if_finite`, and the quantization of the actor's own arithmetic were
  not exercised. That is the register's own standing row and would need a run.
- **The act-time path.** Out of scope by the brief (the two prior ledgers cover it),
  but §2.3's caveat about where our 0.320 is read is the one place it touches this
  audit's conclusion, and pricing it at 12k rather than at 775k was not done here.
