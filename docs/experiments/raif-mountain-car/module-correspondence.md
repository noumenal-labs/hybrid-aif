# Module correspondence — R-AIF's state-based Mountain Car agent, leaf by leaf

**Status: build-time audit, not an adjudication.** Nothing here is a decision. Every row is
either a measurement or a mechanical enumeration of the two module trees, and the one
structural divergence it records is written as a finding that quotes the decision blocking
its "fix", not as a proposal.

**Date:** 2026-09-16. **Ours:** `hybrid-aif` @ `734665b`. **Theirs:** `raif-paper` @
`9b1864e8`, the oracle. Doctrine: the oracle is the code.

This is the artifact the replication never had. Four defects have now been found by chasing
their behavioural symptoms over days rather than by reading the target — the info-gain
leaf's missing `E[σ²]` (which cost the first 2M run), the dropped `moments` carry
(divergence 29, which cost the third), the initializer pair 17+26, and the CRSPP's own
initializer — and the common root cause is that our agent was built from prose
specifications derived by hand from their module tree rather than mechanically from it. So
this table is *complete rather than interesting*. A MATCHED row is worth exactly as much as
a DIVERGES row, because together they are the proof of coverage: 47 leaf blocks, every leaf
array of the built agent claimed by exactly one of them, and the sums checked at both ends.

## Method — where each column comes from

The failure mode being defended against is a column derived from our own documentation, so
neither side's column is.

**Their side** is enumerated from **their own checkpoint**,
`raif-paper/logs/oracle_state_mc_seed0/checkpoint.ckpt` (step 2,000,000): a plain pickle
whose `agent` entry is a flat `{parameter path: ndarray}` dict with 151 entries. That is the
module tree *as actually constructed under the resolved config*, not as a reader resolves
`configs.yaml` by hand — which is how five earlier divergences entered. Their per-block
semantics (initializer, bias rule, norm placement, head formula) are then read from
`raif/aif/nets.py`, `agent.py` and `epistemics.py` at `9b1864e8`, and the values from the
resolved config vendored at `reference/fixtures/oracle_state_mc.config.yaml`.

**Our side** is introspected from `raif_mountain_car(jax.random.PRNGKey(0))` — the built
pytree, through `jax.tree_util.tree_flatten_with_path` — not from docstrings.

**The two are then checked against each other by execution.** Every block of the oracle's
own trained weights was loaded into the corresponding module of ours, and the two forward
passes diffed on identical inputs. Their side ran under their real `nets.py` driven by
`ninjax` in a scratch venv (Python 3.11, jax 0.4.30, tfp 0.25); ours ran in the worktree's
own venv. That check does not depend on any reading of their source being right, and it is
what makes the "MATCHED" in each row a measurement rather than a claim. Its results are §3.

Neither repository was modified. All instrumentation lives in the session scratch directory.
The live 2M run in `runs/mc_2d` was not touched.

**Not re-litigated here** (argued elsewhere, on the record): divergence 20's info-gain
indexing, the cosmetics 23/24/25/27/28/D7, the CRSPP→encoder gradient seam (ledger 5,
measured inert by Arm A), the sampled-std leaf, and `float16` (measured and demoted).

## Conventions the whole table rests on

Four rules of theirs do all the bookkeeping, and every one of them is a place a
hand-derived specification has gone wrong before.

1. **Their `Linear` drops its bias wherever it is normed.** `self._bias = bias and norm ==
   'none'` (`nets.py:664`). So a normed hidden block costs `in·out + 2·out` — kernel plus
   the LayerNorm's `scale` and `bias` — and carries no bias of its own. Ours reproduces the
   rule under `winit="normal"` (`_linear_stack`, `GRUCell.from_shapes`).
2. **Their `Dist` output linears and their `…_stats` linears do *not* get the block's
   `winit`.** `Dist.inner` forwards only `{outscale, outnorm}` (`nets.py:556-565`) and
   `_stats` is `self.get(name, Linear, units)` with no `**kw` at all (`nets.py:233,243`), so
   both take `Initializer`'s **uniform** default at `fan: avg`, and both keep a bias (they
   are unnormed) that starts at `jnp.zeros`. This is divergence 26b, and a transcription
   test once certified the opposite.
3. **Both initializer branches realize the same second moment**, `std = sqrt(outscale /
   mean(fan_in, fan_out))` — their normal branch divides by `0.87962566` precisely to undo
   the truncation. The *support* is what separates them: uniform reaches `sqrt(3)·std`,
   truncated normal reaches `2·std/0.87962566 = 2.274·std`. Every initializer claim below is
   checked on both.
4. **Their block order is linear → (bias) → dropout → reshape → norm → activation**
   (`nets.py:672-690`), with `Norm` a LayerNorm at **eps 1e-3** carrying `scale` (ones) and
   `bias` (zeros). Ours is the same order with `eqx.nn.LayerNorm(w, eps=1e-3)`.

Two sizes recur: the **feature vector** `[deter 128, flat stoch 1024] = 1152`, which is what
`inputs: [deter, stoch]` resolves to for the decoder, reward, cont, actor and critic; and
the **info-gain input** `[deter, stoch, action] = 1153`.

## 1. The correspondence table

`in → out` is the linear's own shape. `params` is the whole block: kernel, plus a bias where
their `Linear` keeps one, plus the LayerNorm's `scale` and `bias` where the block is normed.
`init` is their `Initializer`'s `(dist, fan, outscale)`. Norm is LayerNorm at eps 1e-3,
placed after the linear and before the activation, in every row that has one.

### 1.1 Encoder — their `enc/mlp`, `MultiEncoder`'s MLP branch (`cnn_keys: ^$`, so no CNN)

| theirs | ours | in → out | params | init | bias | norm | act | specifics | disposition |
|---|---|---|---|---|---|---|---|---|---|
| `enc/mlp/h0` | `recognition['embed'].layers[0]` + `norms[0]` | 2 → 64 | 256 | normal, avg, 1.0 | none | layer | silu | `symlog_inputs: true` — the 2-vector observation is symlog'd before this layer | MATCHED |
| `enc/mlp/h1` | `recognition['embed'].layers[1]` + `norms[1]` | 64 → 64 | 4,224 | normal, avg, 1.0 | none | layer | silu | the embedding **is** this activation: `MLP(shape=None)` returns the last hidden, which is our `MLPPredictor(out_activation=True)` | MATCHED |

### 1.2 RSSM — their `rssm/*`, the world-model half of `CRSPP_RSSM`

| theirs | ours | in → out | params | init | bias | norm | act | specifics | disposition |
|---|---|---|---|---|---|---|---|---|---|
| `rssm/initial` | `initial_state_prior[0]` `LearnedVector` | — → 128 | 128 | `jnp.zeros` | — | — | tanh | `initial: learned`; a fresh agent starts at the zero carry and can never leave [-1, 1] | MATCHED |
| — (derived) | `initial_state_prior[1]` `DerivedInitial("mode")` | — | 0 | — | — | — | — | their `get_stoch(deter_0)`: the initial code is the dynamics prior's **argmax** at the initial carry, and has no parameters of its own | MATCHED |
| `rssm/img_in` | `transition[0].map.input_proj` + `norms[0]` | 1025 → 32 | 32,864 | normal, avg, 1.0 | none | layer | silu | input is `[flat stoch 1024, action 1]`, stoch first; called with `dropout=True` at `dropout: 0.0`, which is inert (it does consume one PRNG key on their side) | MATCHED |
| `rssm/gru` | `transition[0].map.gate` + `norms[1]` | 160 → 384 | 62,208 | normal, avg, 1.0 | none | layer | **none** | input `[deter 128, hidden 32]`; the 384 splits `(reset, candidate, update)` **after** the norm, then `tanh(sigmoid(reset)·candidate)` and `sigmoid(update − 1)` | MATCHED |
| `rssm/img_out` | `transition[1].trunk.layers[0]` + `norms[0]` | 128 → 32 | 4,160 | normal, avg, 1.0 | none | layer | silu | | MATCHED |
| `rssm/img_stats` | `transition[1].trunk.layers[1]` | 32 → 1024 | 33,792 | **uniform**, avg, 1.0 | yes, zeros | none | none | reshaped to (32, 32); `unimix: 0.01` mixed into the probabilities before the log, so the stored logit is already unimixed | MATCHED |
| `rssm/obs_out` | `recognition['stoch'].trunk.layers[0]` + `norms[0]` | 192 → 32 | 6,208 | normal, avg, 1.0 | none | layer | silu | input `[deter 128, embed 64]` | MATCHED |
| `rssm/obs_stats` | `recognition['stoch'].trunk.layers[1]` | 32 → 1024 | 33,792 | **uniform**, avg, 1.0 | yes, zeros | none | none | (32, 32), `unimix: 0.01` | MATCHED |

### 1.3 CRSPP — their `rssm/*_2`, the second half of the same `CRSPP_RSSM` object

The CRSPP is not a separate module on their side: `img_in_2`, `gru_2`, `img_out_2`,
`img_stats_2`, `obs_out_2` and `obs_stats_2` are `self.get(...)` calls inside the one
`CRSPP_RSSM`, so they land in `rssm` and therefore in `model_opt`. Ours is a second declared
`System` (`crspp_rssm`) whose parameters live in `scoring["cosine_sim_to_pref_rollout"]`.

| theirs | ours | in → out | params | init | bias | norm | act | specifics | disposition |
|---|---|---|---|---|---|---|---|---|---|
| `rssm/img_in_2` | CRSPP `transition[0].map.input_proj` + `norms[0]` | 1024 → 32 | 32,832 | normal, avg, 1.0 | none | layer | silu | **no action column** — this is the whole of "an RSSM without action encoded into the transition prior"; and no `dropout=True` here | MATCHED |
| `rssm/gru_2` | CRSPP `transition[0].map.gate` + `norms[1]` | 160 → 384 | 62,208 | normal, avg, 1.0 | none | layer | none | same gate arithmetic as `rssm/gru`, separate parameters | MATCHED |
| `rssm/img_out_2` | CRSPP `transition[1].trunk.layers[0]` + `norms[0]` | 128 → 32 | 4,160 | normal, avg, 1.0 | none | layer | silu | | MATCHED |
| `rssm/img_stats_2` | CRSPP `transition[1].trunk.layers[1]` | 32 → 1024 | 33,792 | **uniform**, avg, 1.0 | yes, zeros | none | none | (32, 32), `unimix: 0.01` | MATCHED |
| `rssm/obs_out_2` | CRSPP `recognition['stoch'].trunk.layers[0]` + `norms[0]` | 192 → 32 | 6,208 | normal, avg, 1.0 | none | layer | silu | input `[deter 128, **the world model's** embed 64]` — the CRSPP has no encoder on either side; ours receives it as an external input (ledger entry 5) | MATCHED |
| `rssm/obs_stats_2` | CRSPP `recognition['stoch'].trunk.layers[1]` | 32 → 1024 | 33,792 | **uniform**, avg, 1.0 | yes, zeros | none | none | (32, 32), `unimix: 0.01` | MATCHED |
| *(absent — theirs shares `rssm/initial`)* | CRSPP `initial_state_prior[0]` `LearnedVector` | — → 128 | **0 vs 128** | `jnp.zeros` | — | — | tanh | **the only structural divergence in the agent.** See §4 | **DIVERGES** |

### 1.4 Heads — their `dec`, `rew`, `cont`, all `MLP` over the 1152-wide feature

| theirs | ours | in → out | params | init | bias | norm | act | specifics | disposition |
|---|---|---|---|---|---|---|---|---|---|
| `dec/mlp/h0` | `likelihood[0].trunk.layers[0]` + `norms[0]` | 1152 → 64 | 73,856 | normal, avg, 1.0 | none | layer | silu | | MATCHED |
| `dec/mlp/h1` | `likelihood[0].trunk.layers[1]` + `norms[1]` | 64 → 64 | 4,224 | normal, avg, 1.0 | none | layer | silu | | MATCHED |
| `dec/mlp/dist_state/out` | `likelihood[0].trunk.layers[2]` | 64 → 2 | 130 | **uniform**, avg, 1.0 | yes, zeros | none | none | `vector_dist: symlog_mse`; `SymlogDist(·, dims=1, 'mse', agg='sum')`, mode `= symexp(out)` | MATCHED |
| `rew/h0` | `likelihood[1].trunk.layers[0]` + `norms[0]` | 1152 → 32 | 36,928 | normal, avg, 1.0 | none | layer | silu | | MATCHED |
| `rew/h1` | `likelihood[1].trunk.layers[1]` + `norms[1]` | 32 → 32 | 1,088 | normal, avg, 1.0 | none | layer | silu | | MATCHED |
| `rew/dist_out/out` | `likelihood[1].trunk.layers[2]` | 32 → 255 | 8,415 | **zeros** (`outscale: 0.0`) | yes, zeros | none | none | `symlog_disc`, 255 bins `linspace(-20, 20)`, `mean = symexp(Σ softmax(logits)·bins)`; a fresh head predicts exactly 0 | MATCHED |
| `cont/h0` | `likelihood[2].trunk.layers[0]` + `norms[0]` | 1152 → 32 | 36,928 | normal, avg, 1.0 | none | layer | silu | | MATCHED |
| `cont/h1` | `likelihood[2].trunk.layers[1]` + `norms[1]` | 32 → 32 | 1,088 | normal, avg, 1.0 | none | layer | silu | | MATCHED |
| `cont/dist_out/out` | `likelihood[2].trunk.layers[2]` | 32 → 1 | 33 | **uniform**, avg, 1.0 | yes, zeros | none | none | `dist: binary` — one logit through `tfd.Bernoulli`; **continuation**, not termination | MATCHED |

### 1.5 Actor — their `planner/ac/actor`

| theirs | ours | in → out | params | init | bias | norm | act | specifics | disposition |
|---|---|---|---|---|---|---|---|---|---|
| `ac/actor/h0` | `policy_prior.layers[0]` + `norms[0]` | 1152 → 256 | 295,424 | normal, avg, 1.0 | none | layer | silu | `actor: {layers: 3, units: 256}` — **not** the critic's or the ensemble's size | MATCHED |
| `ac/actor/h1` | `policy_prior.layers[1]` + `norms[1]` | 256 → 256 | 66,048 | normal, avg, 1.0 | none | layer | silu | | MATCHED |
| `ac/actor/h2` | `policy_prior.layers[2]` + `norms[2]` | 256 → 256 | 66,048 | normal, avg, 1.0 | none | layer | silu | | MATCHED |
| `ac/actor/dist_out/out` + `ac/actor/dist_out/std` | `policy_prior.layers[3]` (one fused 256 → 2) | 256 → 1, twice | 514 | **uniform**, avg, 1.0, **each half at its own fan (256, 1)** | yes, zeros | none | none | `dist: normal`: `Normal(tanh(out), minstd + (maxstd − minstd)·sigmoid(raw + 2.0))` with `(0.1, 1.0)`; entropy **summed** over action dims. `unimix: 0.01` is declared on the block but reaches only `dist: onehot`, so it is inert here | MATCHED |

The fused layer is the one place our shape differs from theirs, and it is a representation
choice, not a divergence: their two linears are our one layer of `2·action_dim`, split by
`MLPConditional._head` into `(mean, raw)` in that order, with **each half drawn at its own
`fan_out = 1`** rather than at the fused layer's 2. Same parameter count, same draw law, and
§3 measures the same outputs.

### 1.6 Critic and its slow copy — their `planner/critic/{net,slow}`

| theirs | ours | in → out | params | init | bias | norm | act | specifics | disposition |
|---|---|---|---|---|---|---|---|---|---|
| `critic/net/h0` | `scoring["g_tail"][0].trunk.layers[0]` + `norms[0]` | 1152 → 32 | 36,928 | normal, avg, 1.0 | none | layer | silu | `critic: {layers: 2, units: 32}` | MATCHED |
| `critic/net/h1` | `…layers[1]` + `norms[1]` | 32 → 32 | 1,088 | normal, avg, 1.0 | none | layer | silu | | MATCHED |
| `critic/net/dist_out/out` | `…layers[2]` | 32 → 255 | 8,415 | **zeros** (`outscale: 0.0`) | yes, zeros | none | none | `symlog_disc`, 255 bins over [-20, 20] | MATCHED |
| `critic/slow/h0` | `scoring["g_tail"][1].trunk.layers[0]` + `norms[0]` | 1152 → 32 | 36,928 | normal, avg, 1.0 | none | layer | silu | separate draw on both sides, and irrelevant on both: `SlowUpdater`'s first call is a hard copy (`mix = 1·need_init`), thereafter an EMA at 0.02 every step | MATCHED |
| `critic/slow/h1` | `…layers[1]` + `norms[1]` | 32 → 32 | 1,088 | normal, avg, 1.0 | none | layer | silu | | MATCHED |
| `critic/slow/dist_out/out` | `…layers[2]` | 32 → 255 | 8,415 | **zeros** | yes, zeros | none | none | | MATCHED |

The slow copy is **in no optimizer on either side**: their `GFunction.train` passes
`self.net` to `self.opt`, which is why their `train/params_agent/planner/critic/critic_opt`
reports 46,431 and not 92,862.

### 1.7 Info-gain ensemble — their `wm/epis/ig{0..3}`, `ig_models: 4`

All four members are identical in structure; the table gives one, and the count column is
per member.

| theirs | ours | in → out | params | init | bias | norm | act | specifics | disposition |
|---|---|---|---|---|---|---|---|---|---|
| `epis/ig{i}/h0` | `scoring["model_uncertainty"][i].layers[0]` + `norms[0]` | 1153 → 32 | 36,960 | normal, avg, 1.0 | none | layer | silu | input `[deter 128, flat stoch 1024, action 1]`, in that order. `ig_head: {layers: 2, units: 32}` | MATCHED |
| `epis/ig{i}/h1` | `…layers[1]` + `norms[1]` | 32 → 32 | 1,088 | normal, avg, 1.0 | none | layer | silu | | MATCHED |
| `epis/ig{i}/dist_out/out` + `…/std` | `…layers[2]` (one fused 32 → 2048) | 32 → 1024, twice | 67,584 | **uniform**, avg, 1.0, each half at its own fan (32, 1024) | yes, zeros | none | none | `dist: normal` predicting the **flattened 1024-wide code** (`ig_target: [stoch]`, whose shape their `loss` writes onto `net._shape` at call time). Mean is `tanh(out)`. The block declares no `minstd`/`maxstd`, so `Dist`'s (1.0, 1.0) defaults give `std = (1−1)·sigmoid(raw+2) + 1 ≡ 1.0` — divergence 15 | MATCHED |

**The `std` half is dead on both sides**: 33,792 of each member's 105,632 parameters, 135,168
across the ensemble, exist, are initialized, are stepped by adam's zero-gradient path, and
can never move, because `maxstd − minstd` is exactly 0.0. That is theirs, so it is ours; the
tests now assert both the constant and the zero gradient, because a future change to
`ENSEMBLE_STD` that separated the two bounds would silently wake all 135,168.

## 2. The reconciliation, at both ends

### 2.1 Per optimizer, against their own metrics

Their numbers are read from `train/params_agent/*` in
`raif-paper/logs/oracle_state_mc_seed0/metrics.jsonl` — their own logger counting their own
pytrees — and independently re-derived by summing the checkpoint's arrays under each
optimizer's membership. The two agree exactly, which is why the table below has one "theirs"
column.

| optimizer | membership on their side | theirs | ours | Δ |
|---|---|---|---|---|
| `wm/model_opt` | `[encoder, rssm, *heads]` — and `rssm` **includes the CRSPP**, since the `_2` blocks live in the same object | 513,314 | 513,442 | **+128** |
| `planner/ac/actor_opt` | the actor only | 428,034 | 428,034 | 0 |
| `planner/critic/critic_opt` | `GFunction.net` only — not `slow` | 46,431 | 46,431 | 0 |
| `wm/epis/ig_opt` | the four `ig{i}` nets | 422,528 | 422,528 | 0 |
| *(no optimizer)* | `GFunction.slow` | 46,431 | 46,431 | 0 |
| **total** | | **1,456,738** | **1,456,866** | **+128** |

### 2.2 Per module, leaf by leaf

Every one of the 47 leaf blocks in §1 was summed on both sides and differenced. **Forty-six
are exactly zero. One is +128.** Both coverage directions are asserted: every leaf array of
the built agent is claimed by exactly one row (so a new parameter anywhere fails), and every
array entry in their checkpoint is claimed by a row or is optimizer state (their fifteen
unmapped entries are the four optimizers' `step`/`good_steps`/`grad_scale`, the two
`retnorm_extr` percentiles, and the slow updater's counter).

### 2.3 The residual, stated

**+128, and every one of them is the CRSPP's own learned initial carry.** There is no other
residual anywhere in the agent — not in the world model (340,322 vs 340,322 exactly), not in
the actor, not in the critic or its slow copy, not in any ensemble member.

## 3. The check that does not depend on reading their source

Counts and a careful reading of `nets.py` cannot see an input order, a norm placement, an
`eps`, a squash, or a fan convention. So every block was checked by **execution**: the
oracle's own trained weights from its own checkpoint, loaded into our module, on inputs
identical to the ones their code saw.

| block | ours vs theirs, max abs diff | shape |
|---|---|---|
| `img_in` + `gru` → next carry | 1.639e-07 | (3, 128) |
| `img_out` + `img_stats` → prior logit | 1.431e-06 | (3, 32, 32) |
| `obs_out` + `obs_stats` → posterior logit | 1.907e-06 | (3, 32, 32) |
| `initial` → episode-start carry | **0.0** | (3, 128) |
| `get_stoch(initial)` → episode-start code | **0.0** | (3, 32, 32) |
| `img_in_2` + `gru_2` → next carry | 2.384e-07 | (3, 128) |
| `img_out_2` + `img_stats_2` → CRSPP prior logit | 9.537e-07 | (3, 32, 32) |
| `obs_out_2` + `obs_stats_2` → CRSPP posterior logit | 1.431e-06 | (3, 32, 32) |
| `enc/mlp` → embedding | 4.768e-07 | (3, 64) |
| `dec/mlp` → decoded state | 2.235e-07 | (3, 2) |
| `rew` → 255 logits | 1.907e-06 | (3, 255) |
| `rew` → symexp mean | 9.537e-07 | (3,) |
| `cont` → continuation probability | 5.215e-08 | (3,) |
| `critic/net` → symexp mean | 1.144e-05 | (3,) |
| `actor` → mean | 1.192e-07 | (3, 1) |
| `actor` → std | 7.451e-08 | (3, 1) |
| `actor` → entropy | 4.768e-07 | (3,) |
| `ig0` → mean | 2.384e-07 | (3, 1024) |
| `ig0` → std | **0.0** (both exactly 1.0) | (3, 1024) |

Both sides ran at float32, so ~1e-06 is the floor; the critic's 1.144e-05 is on values of
magnitude 5–9 after `symexp`, i.e. a relative 2e-06. **Nineteen block comparisons, worst
1.1e-05.** Between them these pin, without any of it having been asserted from a reading:
the input concatenation order of all seven multi-parent blocks, the LayerNorm `eps` of
1e-3 and its placement between linear and activation, the absent hidden biases, `silu`, the
GRU's split order and its two offsets, the unimix constant and the point at which it enters,
`symlog` on the encoder's inputs, `symexp` on the decoder's and the two `symlog_disc` heads'
outputs, the 255-bin grid, `tanh` on the actor's and the ensemble's means, the actor's
`(maxstd − minstd)·sigmoid(raw + 2) + minstd`, and the ensemble's constant σ.

### 3.1 And the gradients, leaf by leaf

A matching forward pass can still sit on a different backward one — a norm in the wrong place
relative to the activation, a carry differentiated where theirs is not. So the same transplant
was run backwards: a fixed random cotangent over each block's output, contracted to a scalar,
differentiated with respect to **their parameters** under their code and with respect to **our
leaves** under ours, and diffed leaf by leaf.

**Ninety-three gradient leaves across thirteen blocks. Worst relative difference 4.3e-06**, on
gradient magnitudes spanning 0.0076 to 196. Three results in it are worth naming:

- The **GRU's** gradient reaches `img_in`'s kernel and both LayerNorm pairs identically, which
  is the one place where a wrong norm placement would be invisible to the forward pass on
  trained weights but not to the backward one.
- Differentiating the **CRSPP's** `img_step_no_action` gives **exactly zero** gradient into
  every world-model leaf — `img_in`, `gru`, `img_out`, `img_stats`, `obs_out`, `obs_stats`
  *and* `rssm/initial` — on their side. Our CRSPP is a separate `System` and so trivially has
  the same property. The two are the same function of the same parameters, which is what makes
  §4's divergence a divergence about the *episode start* and nothing else.
- The **ensemble head's `std` half** has a gradient of **exactly 0.0 on both sides**, which
  confirms the dead-parameter reading of §1.7 by measurement on the oracle rather than by
  reading `Dist.inner`.

The initializers are the one property the transplant cannot check, since it overwrites them.
They were checked separately against their `Initializer`'s analytic law on all 40 weight
leaves of a freshly built agent, on **both** the second moment (every ratio in 0.94–1.04)
and the support (which is what separates their normal branch from their uniform one). Every
leaf passes. The same check applied to the pre-`734665b` build would have failed the CRSPP's
six matrices, four of them on the std and all six on the support.

## 4. The one DIVERGES row, and why it is not fixed here

**The CRSPP's own learned initial carry — ours has 128 parameters theirs does not.**

Their `obs_step_no_action` resets an episode start to `self.initial(...)` (`nets.py:156-157`)
— which is the **world model's** learned `initial` vector pushed through the **world
model's** `img_out`/`img_stats` (`nets.py:57-59`). So on their side the CRSPP's episode
start is the world model's, and the gradient of that start flows into the world model's
prior head. Ours is `parts.initial()` plus `parts.prior.mode((deter0,))` — the CRSPP's own
vector through the CRSPP's own prior. The 128 is one half of the difference; the other half
(whose prior head the start runs through, and where that gradient goes) carries no
parameter-count signature at all.

**This is not fixed, because fixing it would break an adjudicated decision.** A10 disposition
5 (`amortization.md:296-299`, adjudicated 2026-08-06) reads, in full: "**CRSPP shares
architecture, NOT parameters, with the world model** (second parameter set, own GRU, one
class). Express as a second `MLPConditional`-family declaration whose parents exclude the
action." A11's refinement, at `amortization.md:874-879`, gives the mechanism behind it:
holding one of the agent's modules in the CRSPP's slots "would be A11 refinement (ii)'s
pytree-forking trap one level up: two Systems' params flatten separately, so the 'shared'
torso forks at the first optimizer step and drifts apart silently." Making the CRSPP's
episode start read the world model's `initial` through the world model's prior head is
exactly the cross-System parameter sharing those two decisions refuse — the second of them
because it does not *work*, not merely because it is untidy — and the
project's standing `adjudicated-decision-fidelity` rule is to quote the decision and flag
the delta rather than re-propose a variant of it. The divergence is already on the record as
`execution-ledger-training.md` rows 4.1 and 4.14 and as the accepted residual in
`test_every_optimizers_parameter_count_matches_the_oracles`; this audit adds the count
reconciliation that shows it is the **only** one.

Two second-order notes on its size, for whoever prices it later. It is 128 of 1,456,866
parameters (0.009%), it is read once per episode rather than once per step, and a fresh
agent's value is the zero vector on both sides — so at initialization the two are *identical*
and the divergence only opens as training moves the two vectors apart. What is unmeasured is
the gradient half: on their side the CRSPP's episode starts pull on the world model's prior
head, which is a channel from the preference model into the world model that ours does not
have. Arm A measured the *other* CRSPP→world-model gradient channel (through the encoder)
and found it behaviourally inert; this one has not been measured separately.

## 5. Structural differences that are not leaf-module differences

Recorded so a future reader of this table does not take "47 rows, 46 matched" for more than
it is. None of these is a parameter; all are already on the record.

1. **Optimizer grouping around the CRSPP.** Theirs is one `model_opt` step per train step
   over one joint loss (`model_loss.mean() + C_model_loss_mean`) with **one**
   `clip_by_global_norm(1000)` over all 513,314 gradients. Ours is `fit_world` then
   `fit_crspp`, two gradient steps with two clips over 340,322 and 173,120. Adam is
   elementwise, so the only arithmetic difference is the clip's scope and the fact that
   `fit_crspp` reads the *stepped* world model. Both halves are what Arm A varied, and Arm A
   measured them behaviourally inert (bands 4–9 unchanged). Divergence 21 and ledger 5.
2. **The float16 apparatus.** Their four optimizers each carry `good_steps` and `grad_scale`
   state for `apply_if_finite` plus a dynamic loss scaler; ours has no analogue.
   N-A-BY-DESIGN, the register's standing `float16` row, measured and demoted.
3. **The preferred rollout is mean-field where theirs samples** (A12 / ledger row 6.5). A
   behavioural divergence in how the CRSPP's *tree* is driven, not in what the tree is.
4. **`retnorm` and the slow-updater counter** are non-parameter state on both sides
   (`retnorm_extr/{low,high}`, `critic/updates/value`); ours carries them in
   `LearnState.moments`. Divergences 29 and 13, both closed.

## 6. What this audit could not cover

- **A gradient of a *loss*, as opposed to of a block.** §3.1 differentiates each block's own
  output under a fixed cotangent, which is the module-tree question. What the world-model
  loss, the actor loss and the four trainers do with those gradients — the scaling, the
  clipping, the order — is the two execution ledgers' subject and `actor-trace-audit.md`'s,
  and nothing here re-opens it.
- **The four ensemble members other than `ig0`, and the slow critic**, were checked by count
  and by construction (same class, same constructor call) rather than by transplant. They
  are the same class as a block that was transplanted, so this is a coverage statement, not
  a doubt.
- **Anything at scale or in float16.** The transplant ran at batch 3 in float32. Every
  quantity checked is shape-invariant, so this is a precision limit rather than a coverage
  limit, but their `COMPUTE_DTYPE` was not exercised.
- **Training-time control flow.** This is a tree audit. What each module is *called with*,
  in what order, and whose gradient reaches it is the two execution ledgers' subject and
  `actor-trace-audit.md`'s; §5 lists only the places where those audits' findings bear on
  how this table should be read.

## 7. The result, plainly

**The module tree is theirs.** Forty-seven leaf blocks, forty-six matched to the parameter and
measured to float32 precision against the oracle's own forward pass *and* its own backward
pass, one divergence of 128 parameters that was already on the record and that an adjudicated
decision blocks. The initializer law holds on all forty weight leaves. Nothing new was found.

That is the result this replication has never had, and it is worth stating as a positive:
the class of defect that has cost four investigations — a block built at the wrong width, an
initializer on the wrong fan, a bias where theirs has none, a head with the wrong squash —
is now excluded by construction rather than by argument, and the tests in
`tests/test_oracle_parity.py` keep it excluded. What remains open about this agent is not
its structure.

## Addendum (2026-09-30): function-level correspondences

*The tables above match modules. The engine's docstrings used to carry the same
matching at function level, as "their X" asides and file:line citations. The
2026-09-21 prose pass removed these asides, and they are collected here from the
salvage (now an index). Each one is an attribution, not a finding. Where a row
above or a ledger row already covers one, that row is the authority.*

| ours | theirs |
|---|---|
| `RSSMCore.img_step`, `System.update_empirical_prior` (amortized), `objectives.amortized.make_transition` | `img_step` |
| `RSSMCore.obs_step`; `update_empirical_prior` then `infer_states` / `inference.amortized.predict` then `recognize` | `obs_step` (`nets.py:131-141`) |
| `System.predict_beliefs` on the amortized path | `open_loop` |
| `RSSMCore.observe(is_first=)` | `obs_step`'s `1 − is_first` mask and reset to `initial` (`nets.py:131-141`) |
| the encoder symlogging its own input | `symlog_inputs: True` |
| `MLPCategorical.mode` | `get_stoch` |
| `LearnedVector` (trainable) / frozen | `initial: 'learned'` / `initial: 'zeros'` |
| `GRUCell` (`input_proj`, `gate`) | `_gru` (`img_in`, `gru`) |
| `unimix_logits` | `_stats`, `unimix 0.01` |
| `factored_categorical_kl` | `Independent(OneHotDist, 1)` |
| `twohot` | `DiscDist` |
| `symlog_mse_nll`, `symlog_disc_nll`, `binary_nll` | the `symlog_mse`, `symlog_disc`, `binary` heads' losses |
| `kl_balance` scales | `configs.yaml:74-75` (`dyn 0.5`, `rep 0.1`, `free 1.0`) |
| `MLPConditional` bounded head | `Dist` `normal` (`nets.py:573-580`) |
| LayerNorm placement, biasless normed layers | `Linear` (`nets.py:688-689`, eps 1e-3; `bias and norm == 'none'`) |
| `_TRUNC_NORMAL_STD` | `Initializer`'s divisor (`nets.py:760`) |
| `trunc_normal_avg` / `uniform_avg` | `Initializer` `normal` / `uniform` at fan-avg (`nets.py:756-763` / `748-753`) |
| `controls.soft_clip` | `action_clip` (`nets.py:131-141,167-172`) |
| `objectives.amortized._diag_entropy` | `ent` in `epistemics.py` |
| `_occupancy` / `_ensemble_disagreement` / `_predicted_reward` | paper 7c (MOP's β) / 7b / `r_τ` |
| `train.imagine.lambda_return` | `agent.py:569-576` |
| `perc_ema`, `init_retnorm` | `Moments(impl='perc_ema')`, zeros at init |
| `_cont_mode` | `heads['cont'](traj).mode()` (`agent.py:256`) |
| `rollout_weight` | `traj['weight']` (`agent.py:257-259`) |
| `_slow_update`; the `SLOW_KEY` counter | `SlowUpdater.__call__` (`jaxutils.py:481-495`); `SlowUpdater.updates` (`:483`) |
| `train.buffers.StreamBuffer`, `sample_windows` | `Uniform` replay, `generic.py::_sample` (see `execution-ledger-acting.md`'s D8 addendum) |

## Addendum (2026-09-30): rationale from the docstring paring

*Recorded nowhere but the text the 2026-09-30 paring pass cut from
`distributions/neural.py::GRUCell`.*

- **§1.2's `rssm/gru` row: why `sigmoid(update − 1)`.** The `−1` offset biases
  the update gate toward keeping the carry. That is what lets a modest
  deterministic path hold information over an episode.
