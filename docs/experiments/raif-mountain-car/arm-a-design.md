# Arm A — can R-AIF's CRSPP-to-encoder gradient be matched inside the library's assumptions?

**Status: design investigation, not adjudication.** Written 2026-09-15 against `hybrid-aif`
`859faf0` (working tree dirty with Arm B's `winit` edits, untouched) and `raif-paper`
`9b1864e8`. Question from Alex: "I'd like to first investigate a way of matching their config
without breaking library assumptions. If it's *only* possible to match their results with
different structural assumptions, that's interesting and we should revisit our decision." No
source was changed, no training was run, and the live `runs/mc_2d` was not read. Two new
numbers come from a read-only gradient probe on a *copy* of the finished Arm B checkpoint
(`runs/armB_init_pair_100k`, step 100,352) and on freshly built step-0 agents; the script is
`gradprobe.py` in the session scratch directory and writes nothing into the repo.

Every claim below is tagged **read** (from source, with file:line), **measured** (a number
from logs or the probe), or **inferred**.

## 0. The answer in one paragraph

Their configuration *can* be matched without changing the library's structural assumptions —
for the experiment, today, in the reference tier, with zero engine change: a composed trainer
in `reference/` takes one `filter_grad` over the world model's eight parts **and** the CRSPP's
four, under one clip, exactly as their `model_opt` does. That composition breaks neither the
one-way import rule nor ledger entry 5 as adjudicated: the embedding still crosses to the CRSPP
as a value through the `external:embed` parent, and the encoder module is still held once, in
the agent's params. What changes is the *gradient scope* — and `amortization.md` §(f) already
records that scope as "a consequence of that decision that was never written down as a
divergence", not as the decision itself. Whether the permanent form should stay a reference-tier
composition or become an engine-generic "a declared row may co-train with the sequence loss"
hook *does* need adjudication, because it bends one A12 sentence (`_gradient_slots`: the scoring
slot's rows "own their own losses and optimizer state"). No hard assumption has to change for
either form; the one that bends is "one parameter set, one trainer". Separately, the probe
measured the gradient we are withholding: at 100k it is 1.2% of the world loss's gradient on
the encoder by norm (contrast alone 0.7%) and dominates 1% of the encoder's coordinates; at
step 0 the contrast is 4%. That lowers the prior that this seam is *the* cause of the drift,
which is why §D asks for the 100k run before any adjudication. So: **yes for the experiment;
adjudication needed only for the permanent form; and the measured size of the seam argues for
running the experiment before deciding anything.**

## A. What their encoder receives from the CRSPP

**The optimizer scope (read).** `WorldModel.train` builds `modules = [self.encoder, self.rssm,
*self.heads.values()]` and takes one `self.opt(modules, self.loss, ...)` step
(`raif/aif/agent.py:182-183`). `self.rssm` is a `CRSPP_RSSM` whose action-free branch —
`img_in_2`, `gru_2`, `img_out_2`, `img_stats_2`, `obs_out_2`, `obs_stats_2`
(`raif/aif/nets.py:151-165, 189-201, 220-229`) — lives inside the same module, so the CRSPP's
parameters are in the same `nj.grad`, the same `clip_by_global_norm(1000)` (first in the chain,
`jaxutils.py:373-374`) and the same adam state (`jaxutils.py:409, 422-425`) as the world
model's. `model_opt` is `adam lr 1e-4, eps 1e-8, clip 1000` (`configs.yaml:76`).

**The loss (read).** `WorldModel.loss` computes `embed = self.encoder(data)` once
(`agent.py:191`), runs both scans over it — `post, prior = rssm.observe(embed, ...)` and
`C_post, C_prior = rssm.observe_no_action(embed, is_first, prev_latent)` (`agent.py:195-198`)
— and returns `model_loss.mean() + C_model_loss_mean` (`agent.py:241`), where

    dyn_C    = max(KL(sg(C_post) ‖ C_prior), 1)                        nets.py:251-260
    rep_C    = max(KL(C_post ‖ sg(C_prior)), 1)                        nets.py:262-276
    mask     = sign(pref_label).clip(0, 1)                             agent.py:225
    C_model_loss = mask · (0.5 · dyn_C + 0.1 · rep_C)                  agent.py:226-228
    contrast     = −pref_label · cos_sim(C_post, sg(post))             agent.py:229-230
    C_model_loss_mean = C_model_loss.mean() + 1.0 · contrast.mean()    agent.py:231

with `loss_scales.contrast: 1.0` (`configs.yaml:79`) and `free: 1.0` on both KL directions
(`configs.yaml:74-75`; `nets.py:251-276`). `cos_sim` compares `[deter, flat stoch]`
(`agent.py:165-179`).

**Which terms reach the encoder, and through what (read).** `C_post` at step *t* depends on
`embed_t` directly through `obs_out_2`/`obs_stats_2` (`nets.py:159-161`), and on `embed_{<t}`
through the carried straight-through sample (`OneHotDist.sample = sg(hard) + probs − sg(probs)`,
`jaxutils.py:105-108`) fed into `img_step_no_action`. Therefore **all three CRSPP terms** reach
the encoder:

- the contrast, through `C_post`'s deter and code — `post` is under `sg`, so the agent's own
  posterior contributes no gradient here;
- the `rep` half of the KL, through `C_post`'s logits directly;
- the `dyn` half of the KL, through `C_prior`, whose `deter` is `gru_2` over the *previous*
  `C_post` code — so via `embed_{t−1}` and earlier.

Both KL halves are per-step floored at one nat (`jnp.maximum(loss, free)`), so wherever the
CRSPP's balanced KL sits below a nat the term is a constant with zero gradient, and the gate
`mask` zeroes them on every non-positive step besides.

**Does it also reach the RSSM (read)?** Only one main-RSSM parameter, and only weakly. At an
`is_first` step `obs_step_no_action` resets its carry to `self.initial(...)` — the **world
model's** learned `initial` vector (`nets.py:155-157`, `:56-60`), so that vector receives the
C-loss gradient through `deter` at episode starts. The `get_stoch` half of that reset goes
through `dist.mode()` (`nets.py:203-207`), and `OneHotDist` overrides only `sample`, not `mode`,
so `img_out`/`img_stats` receive nothing from it. The main `gru`, `img_in`, `obs_out`,
`obs_stats` receive nothing: `post` enters the C-loss only under `sg`, and the C-branch uses its
own `_2` layers throughout. So the CRSPP gradient **stops at the encoder** (plus the shared
initial carry vector), which is the precise form of the plateau analysis's hypothesis H6b.

**Magnitude from their logs (measured; 4-seed means from
`logs/oracle_state_mc_seed*/metrics.jsonl`).**

| bin | `C_contrast_loss_mean` | `C_dyn_loss_mean` (raw, ungated) | `C_model_loss_mean` (gated, scaled) | `similarity_mean` | `model_loss_mean` |
|---|---|---|---|---|---|
| 0–8k | −0.110 | 1.0000 | 0.378 | 0.284 | 6.62 |
| 8–16k | −0.086 | 1.0000 | 0.123 | 0.513 | 3.48 |
| 16–32k | −0.049 | 1.013 | 0.103 | 0.504 | 1.49 |
| 32–64k | −0.033 | 1.244 | 0.101 | 0.444 | 0.897 |
| 64–128k | −0.029 | 1.490 | 0.109 | 0.453 | 0.829 |

Two readings. First, `C_dyn` (= `C_rep`) is exactly 1.0000 through 16k: the CRSPP's KL sat at
the free-bits floor everywhere in their opening window, so **in 0–16k the only CRSPP gradient
their encoder received was the contrast term**; the KL joins from ~16k on. Second, by *loss
value* the CRSPP half is 4–9% of the total (`C_model + contrast` ≈ 0.27/0.04/0.05/0.07/0.08
against 6.6/3.5/1.5/0.90/0.83), but a loss share is not an encoder-gradient share — the
world model's `dyn` half never reaches the encoder (its `post` is `sg`), `rep` is floored, and
the reward head is the encoder's main teacher. The probe below measures the gradient directly.

**Magnitude measured on our side (the probe).** For each Arm B seed and four replay batches
of 16 × 64 from that seed's `M`, at the 100k checkpoint, the gradient of each loss with respect
to the **encoder's 4,480 parameters only** (six leaves), global norms:

| quantity | mean | min | max |
|---|---|---|---|
| ‖∂ world loss / ∂ enc‖ (what `fit_world` applies) | 1.65 | 0.23 | 5.45 |
|   of which reward head / cont head / decoder / rep-KL | 0.94 / 0.65 / 0.06 / 0.38 | | |
| ‖∂ CRSPP loss / ∂ enc‖ (what theirs also applies; ours never) | 0.0159 | 0.0014 | 0.105 |
|   of which contrast / gated KL | 0.0087 / 0.0090 | | KL exactly 0 in 8 of 16 batches |
| ratio CRSPP / world, per batch | **0.012** | 0.000 | 0.050 |
| ratio contrast / world | 0.007 | 0.000 | 0.017 |
| cos(∂world, ∂CRSPP) | +0.09 | −0.42 | +0.50 |
| fraction of encoder coordinates where \|∂CRSPP\| > \|∂world\| | 0.9% | 0.05% | 4.7% |
| per-leaf norm ratio, all six leaves | 0.009–0.013 | | |

Per-seed ratios 0.012 / 0.007 / 0.011 / 0.017 (seed 3, the healthy one, highest). The
world-model's whole-gradient norm was 0.45–5.8, so `clip 1000` is inert on both sides. The
same probe at **step 0** (freshly built agents; buffers hold only the seeded oracle fixtures, so
ρ > 0 on 97% of steps — a regime their run never saw) gives contrast / world = 0.037 (max
0.077), and the *gated KL* / world = 2.1, because at initialization the reward head's
`outscale 0.0` sends the encoder no gradient at all and the world KL is at its floor, while the
CRSPP's KL starts at 2.3 nats. That second number is a property of the fixtures-only buffer and
of our CRSPP's own initial state (divergence 4.1/4.14), not of their run, where `C_dyn` was at
the floor from the first log row; it is recorded so nobody reads the 100k ratio as the whole
story, and because it shows the KL term *can* be the larger channel when it is on.

**What the probe does and does not say (inferred).** The gradient we withhold is small by norm
and weakly aligned with the one we take, at the one late snapshot we have; adam normalizes per
coordinate, and it dominates ~1% of coordinates at 100k, so it would move the encoder little.
Over 6,000 gradient steps a 1–4% bias can still accumulate, and there is no checkpoint inside
the 16–64k window where the drift begins. So the probe lowers the prior on H6b without killing
it; the pre-registered run in §D is what decides.

## B. Exactly where ours stops it, and why

**Where (read).** `train_step` fits the world model first (`reference/raif_agent.py:489-491`,
`train/world.py:256-325`: `eqx.filter_grad` over the eight `PART_FIELDS` parts, one clip 1000,
per-slot `absorb`). It then computes `embeds = embeddings(system, params, world_batch.obs)` — a
plain forward pass of the encoder with no `stop_gradient` anywhere in it
(`train/world.py:359-370`) — and hands the array to `fit_crspp(agent.crspp,
params.scoring[CRSPP_ROW], ..., embeds, batch.pref_label, agent_features, ...)`
(`raif_agent.py:513-525`). `fit_crspp` differentiates `crspp_loss` with respect to the CRSPP's
four parts only (`reference/crspp.py:346-355`); `embeds` is closed over as a constant. So the
"stop" is not a `stop_gradient` on the embedding — it is that the embedding is not an argument
of the function being differentiated. The one explicit `stop_gradient` in `crspp_loss` is on
`agent_features` (`crspp.py:273`), which is their `sg(post)` (`agent.py:229`) and correct on
both sides. The CRSPP also has its own adam state (nested `LearnState` at
`scoring:cosine_sim_to_pref_rollout`), its own clip, and its own learned initial vector
(`crspp.py:188-189`, `worlds.py:232`; execution ledger rows 4.1, 4.13, 4.14).

**Why — the adjudicated content, in its own words (read).** Debt-ledger entry 5
(`amortization-debt-ledger.md:222-251`):

> **Principle:** a scoring row that owns a System may need that System's *beliefs* — per-tick
> carry, neither structure nor params. That is a third kind of row input, and it needs a seam
> (auxiliary beliefs passed per call), plus CRSPP recognition conditioned on the agent's
> embedding *as data* (never as a shared module — A11's pytree-forking commitment).
>
> **Paid (chunk 2c):** three mechanisms, each the smallest that does its job — (i) an
> **external-input parent kind** on `Recognition` (`parents=("deter", "external:embed")`,
> widths in a new static `System.external_inputs`, values supplied per call ...); (ii)
> **auxiliary beliefs** ...; (iii) the producer in the trainer ... with the agent's encoder run
> on the agent's side (`train.world.embeddings`).

And the chunk-2c note that built it (`amortization.md:864-879`):

> Holding the agent's *encoder module* in the CRSPP's `recognition` slot would be A11
> refinement (ii)'s pytree-forking trap one level up: two Systems' params flatten separately,
> so the "shared" torso forks at the first optimizer step and drifts apart silently. The kind
> is general and CRSPP-agnostic — nothing in `inference/` knows what an external input is for.

The A11 commitment it rests on (`amortization.md:371-376`): "(ii) module-object sharing across
entries RAISES (the pytree-forking trap: apparent sharing silently duplicates parameters at the
first optimizer step; shared torsos are E's feature by name)".

**What the record says about the gradient scope specifically (read).** `amortization.md`
§(f), "CRSPP gradient scope" (`:2375-2382`):

> Theirs trains the CRSPP inside `model_opt`'s single loss and single global-norm clip, so its
> loss back-propagates into the **shared encoder** and it uses the world model's learned
> `initial` carry. Ours is a separate `fit_crspp` with its own clip, adam state and
> `LearnedVector`, fed pre-computed embeddings with no gradient to the encoder. Deliberate per
> ledger entry 5 — but the *gradient-scope* difference is a consequence of that decision that
> was never written down as a divergence, and it should be.

The execution ledger then recorded it as rows 3.1–3.2 and 4.13 ("DIVERGES (recorded) ...
theirs is one gradient, one clip and one adam over the world model **and** the preference model
together"), and the plateau analysis §5 item 2 called re-opening it "a re-proposal of a
signed-off seam". This document's reading is narrower and should be checked by Alex: **what
was adjudicated is the data-not-module crossing and the non-sharing of the encoder object;
the gradient scope is, by the record's own words, an unadjudicated consequence.** The test
that pins the seam (`tests/test_amortized_trainers.py:676-686`) asserts exactly the
adjudicated content — the CRSPP's recognition entries are not the agent's encoder object — and
says nothing about gradients; no test asserts the encoder is unchanged by the CRSPP step.

## C. The design space, against the library's assumptions

The assumptions each option is judged against, named once:

- **S1 tier rule** — the engine never imports `reference/` (`tests/test_layering.py`;
  `system.md` "package tiers").
- **S2 ledger 5 / A11(ii)** — the CRSPP conditions on the embedding *as data* through
  `external:embed`; no module object is shared across Systems.
- **S3 A6 + A12** — trainers are pure functions over (System, params, data, LearnState), each
  stepping its own subset; `_gradient_slots` excludes `scoring` because "its rows are trained
  by the objective-side trainers, which own their own losses and optimizer state"
  (`system/core.py:1318-1320`); `Estimator.trainer` is a plain string the engine never resolves
  (`objectives/registry.py:267-278`).
- **S4 tier criterion** — a trainer is engine when its loss is over declarations, reference
  when pinned to an external target; `crspp.py` is reference because A5's fence keeps
  "additional trainers/estimators" out of the generic seam (`amortization.md:1216-1220`;
  `reference/__init__.py`).
- **S5 the three groups** — `recognition` is a closed-loop factor (group 2), `scoring` is
  material interior to `P(a|s)` (group 3); "the groups partition fields by role, never values
  by provenance", and "no validation may ever reject a target for being a function of the
  declared model" (`system.md:213-292`). Nothing in the record says group-2 parameters may be
  trained only by the group-1 evidence; `lr_by_slot` exists precisely so that
  freeze-model-train-encoder and its converse are expressible (A11's rejection of (B)).
- **S6 the term tree** — it scores the *model* for the policy; training losses live in
  trainers; the functional vocabulary is closed (`objective-terms.md:24-48, 202-209`).

Precedent worth stating before the options: **one trainer already steps two owners.**
`imagine` takes one `filter_grad` over `(actor, critic)` — group 2 and group 3 — and steps
`policy_prior` and `scoring:g_tail` from one `LearnState` (`train/imagine.py:509-535`). It
keeps the two *objectives* independent by stop-gradients ("one filter_grad over the pair stays
two independent objectives", `:479-481`). So the *shape* "one function, two parameter owners"
is established; what has no precedent is one objective's gradient reaching another owner's
parameters. The world model's own recognition (`recognition:embed`, `recognition:stoch`) is
trained by `fit_world` through the assembly (`inference/rssm.py:452-453, 464-465`).

### (i) A per-row "recognition gradient" flag

**As literally stated — rejected.** A static marker on the CRSPP row saying "this row's
trainer may also step `recognition:embed`" licenses `fit_crspp` to take a *second* gradient
step on the encoder with the CRSPP's loss. That is not their arithmetic: theirs is one summed
loss, one gradient, one adam moment update. Two sequential adam steps on the encoder per
iteration (world, then CRSPP) is a different optimizer trajectory — adam of a sum is not the
sum of adams — and it would double-count the encoder in `init_learn_state`'s carry or share one
state between two steppers. It also does not touch S3's real content: the engine still cannot
build the joint gradient, because it does not know the CRSPP loss (S1). A flag without a
mechanism.

**As the mechanism it implies — (i′), a co-training hook on `fit_world`.** The engine-generic
form that *does* give their arithmetic is: `fit_world(..., co_losses={row_name: fn})`, a
row-keyed mapping of loss callables forwarded untouched — the exact shape `aux_beliefs` already
has (`imagine(..., aux_beliefs={row: beliefs})`, `amortization.md:898-902`: "keyed by row
name, which is the *caller's* vocabulary ... no R-AIF name appears in an engine signature").
`fit_world` would expose its forward pass (states, embeds, logits) to each callable, sum the
returned scalars into `world_loss`, differentiate the world parts **and** `params.scoring[row]`
jointly, clip once, and step the row's parameters through its existing nested `LearnState` —
generically, because the CRSPP's entry is itself a `SystemParams` whose `_gradient_slots` and
`_write_slots` the engine already knows how to walk. The CRSPP row's `trainer` string would
become `"train.world.fit_world"` (ledger 6's vocabulary: the trainer that steps the row's
parameters), with the loss body supplied by the caller.

What it changes: `fit_world`'s signature (opt-in, default path bit-identical), a small
`WorldForward` aux, ~100–150 engine lines, and **one sentence of A12** — `_gradient_slots`'s
"its rows are trained by the objective-side trainers, which own their own losses and optimizer
state" would become "... unless the row declares that it co-trains with the sequence loss".
Which assumption breaks: none of S1, S2, S4, S5, S6; S3 bends (one parameter set stepped by
the world trainer with a loss the engine did not write). Stays inside the declared vocabulary:
yes — a row, a slot, a per-call mapping keyed by row. `reference/crspp.py` stays reference: it
keeps `crspp_loss`, `resolve`, `observe_no_action`; only `fit_crspp` becomes one of two ways to
step the row. Cost: medium; needs adjudication (a new engine capability, however small).

### (ii) The parity recipe composes the two losses in the reference tier

`reference/raif_agent.py` (or `reference/crspp.py`) gains `fit_world_and_crspp(agent, batch,
key, ...)`: `core, slots = assemble(system, params)`; `parts = resolve(crspp, crspp_params)`;
one `eqx.filter_grad` over `(core parts, CRSPP parts)` of

    world_loss(core′, batch, k_world)[0]
    + crspp_loss(parts′, k_crspp, embeds(core′.encoder, batch.obs), batch.pref_label,
                 sg(features of core′'s posterior states))[0]

then `optax.clip_by_global_norm(1000)` over the concatenated grads and per-slot `absorb` into
the world slots and the nested `scoring:cosine_sim_to_pref_rollout` carry, exactly as
`fit_world` and `fit_crspp` each do today (`world.py:312-325`, `crspp.py:356-368`). Every
helper it needs is already importable from the engine; nothing new is exported.

**Which assumption does it break?** S1: no — reference imports the engine freely, and the
engine is unchanged. S2: no — the embedding still crosses as a value through
`external:embed`; the encoder object is held once, in `params.recognition["embed"]`; there is
no second pytree to fork. What changes is that the value is a *differentiated* input rather
than a closed-over constant — the distinction JAX draws is where the gradient is taken, not
what kind of thing the array is. S3: bends in the same way as (i′), but in the tier that is
allowed to be pinned to R-AIF's exact optimizer scope — which is what S4 says reference is
for. S5: no. S6: not involved. **The spirit of ledger 5:** the record's own §(f) separates the
scope from the decision, so this document reads (ii) as changing an unadjudicated consequence,
not the decision; the plateau analysis read it as a re-proposal. Both readings are on the page
for Alex; under `adjudicated-decision-fidelity` the delta is flagged rather than assumed away.
Stays inside the declared vocabulary: it uses only declared things (slots, the row's nested
carry, the external parent). `reference/crspp.py` stays reference and unchanged; `fit_crspp`
remains as the à-la-carte trainer for anyone who wants the separate step.

Cost: ~80–120 reference lines, a `train_step(joint_crspp=True)` switch, a `run_mc --arm-a`
flag. It also happens to remove one of the two extra `observe` scans divergence 21 records —
the joint forward yields the posterior states — but the experiment should **keep** the
post-step `posterior_states` recomputation so that it varies one thing. *(2026-09-25: the
recomputation is gone — divergence 21 closed — so an Arm A built now varies one thing by
reading the joint forward's states, which is already the step's shape.)*

### (iii) Declare the contrast as a term-tree row over the encoder's output

**Impossible as stated, and a category error besides.** The term tree is evaluated on
*imagined* beliefs under a policy (`objectives/amortized.py:412-462`: the cosine row's `step`
reads `ctx.qs` along the rollout); imagination runs the prior with no observations, so **the
encoder is not in the graph** — a term-tree leaf has no path to encoder parameters whatever
its gradient does. The contrast is also not a lattice coordinate: it is a fitting loss on
replay data labelled by ρ (a per-finished-episode label, `reference/revise.py`), comparing two
Systems' *recognition* posteriors at the same real step; the tree has no ρ, no replay, and a
closed functional vocabulary (entropy, conditional entropy, MI, cross-entropy vs a target) that
does not contain cosine similarity. `objective-terms.md:202-209` already places the one
lattice coordinate with no tree home — policy entropy — in the trainer for the same reason:
"the tree scores the *model*, and the policy is not part of the model". The existing
`cosine_sim_to_pref_rollout` row **is** the tree's coordinate for the cosine as a *value*; its
*training* loss belongs in a trainer, and (i′)/(ii) are the two places a trainer can put it.

### (iv) Anything better

Nothing cleaner than (ii) for the experiment or (i′) for a permanent engine form was found.
Two adjacent things are worth naming so they are not conflated with Arm A:

- **The shared initial carry** (rows 4.1, 4.14): theirs resets the CRSPP scan to the world
  model's learned `initial` and lets the C-loss train it; ours gives the CRSPP its own
  `LearnedVector`. A joint trainer could close this too (share `initial_state_prior:deter`
  between the two scans), but it is a second variable and should be its own arm.
- **The train carry across batches** (rows 1.5, 2.22): their `observe_no_action` starts from
  `prev_latent`, the world model's last posterior of the *previous* training batch; ours
  starts every window from the learned initial. Unrelated to gradient scope; listed because
  it is the other place `observe_no_action`'s inputs differ.

### Summary table

| option | engine change | S1 | S2 | S3 | S4 | S6 | `crspp.py` stays reference | gives their arithmetic |
|---|---|---|---|---|---|---|---|---|
| (i) flag, second step | small | ok | ok | bends | ok | ok | yes | **no** (two adam steps ≠ one) |
| (i′) `fit_world` co-training hook | medium, opt-in | ok | ok | bends (one A12 sentence) | ok | ok | yes | yes |
| (ii) reference-tier composition | **none** | ok | ok | bends, in the tier meant for it | ok | ok | yes, unchanged | yes |
| (iii) term-tree row | — | — | — | — | — | **breaks** | — | **impossible** (no encoder in imagination) |

## D. Recommendation, and the cheapest experiment

**Recommendation.** Do not adjudicate the seam yet. Build (ii) in a worktree as a labelled arm,
run the 100k experiment below against Arm B as control, and decide the permanent form on the
result: if the arm passes, adjudicate between "(ii) stays in reference as R-AIF's optimizer
scope" and "(i′) as an engine capability", with the gradient-scope divergence written down
either way (which §(f) asked for a fortnight ago); if it fails, record the measured magnitude
and the null behavioural result as the closure of rows 3.1–3.2/4.13, and move to the next
arm. The probe's 1.2% is the reason to spend 3 hours of compute before any design time.

**Result transfer.** An (ii) result transfers to (i′) exactly: both compute the same summed
loss, differentiate the same twelve parts, clip once over the same set (inert at these norms),
and step each slot through its existing adam state — arithmetic identical up to float summation
order, with the same two PRNG keys (`k_world`, `k_crspp`) if the joint loss splits them as
`train_step` does today. Only the trainer's *address* differs.

**The arm.** `winit="normal"` (Arm B's configuration, now the parity recipe's) plus the joint
world+CRSPP gradient; everything else as `run_mc` defaults. 4 seeds × 100k pooled steps
(≈ 2.8 h at 10 steps/s). Control: `runs/armB_init_pair_100k` as it stands (same tree, sha
`859faf0` dirty with the same edits) — no re-run. Logging to add before launch (no semantics
change): the training-time `mean(sim)` — their `train/similarity_mean`, which we do not log
(`CRSPPLosses.contrast` is `mean(−ρ·sim)`, a different quantity); the CRSPP loss split into
`kl` and `contrast` (already in `CRSPPLosses`, not in `metrics.jsonl`); and plateau §7.3's four
series (imagined action mean/std, act-time entropy, flag-weighted BC term, continuation-mode-0
fraction).

**Pass bands (pre-stated).** Oracle values are 4-seed means from their logs, binned at 8k on
`episode/length` with success := length < 100 (their time limit), which reproduces
`plateau-analysis.md` §1.2's oracle columns. Arm B values are from its `scores.jsonl` and
`metrics.jsonl`.

| # | series | Arm B (control) | oracle | pass if | role |
|---|---|---|---|---|---|
| 1 | decoder-loss ratio ours/oracle, 8–64k | 0.71–1.13 | 1.0 | stays 0.80–1.25 in every 4k bin from 8k | guard (the arm must not re-sharpen the model) |
| 2 | imagined prior entropy at 60–64k / 64–96k | within a nat | 105.2 / 101.7 | ≥ 103.5 / ≥ 99 | guard |
| 3 | training-time `mean(sim)` at 16–32k / 32–64k / 64–96k | not logged | 0.50 / 0.44 / 0.45 | within ±0.10 of each oracle bin | does the joint gradient reproduce *their* similarity trajectory |
| 4 | success rate 0–8k / 8–16k, pooled | 0.14 / 0.48 | 0.42 / 0.88 | ≥ 0.30 / ≥ 0.75 | "slower off the floor" |
| 5 | first successful episode, per seed | 1.8k, 18.0k, 2.6k, 2.0k | 2.2k, 2.0k, 2.6k, 1.6k | every seed < 8k | same, per seed |
| 6 | episode length at 24–32k, per seed / pooled | 46, 44, 49, 46 / 46 | 44, 46, 44, 48 / 46 | no seed > 50; pooled ≤ 48 | already passes; must not regress |
| 7 | episode length at 72–100k, per seed / pooled | 74, 72, 93, 41 / 70 | 40, 41, 41, 40 / 41 | no seed > 50; pooled ≤ 46 | the delayed drift |
| 8 | seeds with SR < 0.9 for ≥ 16k contiguous in 16–100k | 3 of 4 (seeds 0, 1, 2) | 0 of 4 | ≤ 1 of 4 | the shallow dips |
| 9 | seeds with any 8k bin at SR < 0.5 in 16–100k | 1 of 4 (seed 2, from 64k) | 0 of 4 | 0 of 4 | the attractor |

Rows 4–5 are new: measured here, Arm B is behind the oracle on **all four** seeds in 0–16k
(pooled SR 0.14 vs 0.42 at 0–8k, 0.48 vs 0.88 at 8–16k; run 4 was 0.34 / 0.70), not only on
seed 1 — the initializer change slowed our start while the oracle, with the same initializer,
starts fast, so whatever makes theirs fast early is not the initializer. The "first success at
18k vs their 9–10k" framing in the brief is not what the logs show: their first success is at
1.6–2.6k on every seed and ours at 1.8–2.6k on three; seed 1 alone is the 18k outlier. Row 4
is the honest form of that observation.

**Decision rule.** Rows 1–2 must hold or the arm has changed the world model's speed and the
behavioural rows are confounded. Row 3 says whether the joint gradient does on our side what
it does on theirs; if it fails while 1–2 hold, the implementation is wrong before the
hypothesis is. Rows 4–9: if 7, 8 and 9 pass, H6b is confirmed and the permanent form goes to
adjudication; if 4–5 pass but 7–9 fail, the seam is part of the *start* and not of the
attractor, and the register records it as such; if all of 4–9 are unchanged, the seam is
behaviourally inert at this configuration — the probe's 1.2% would then be the recorded
magnitude and rows 3.1–3.2/4.13 close as measured-and-argued, in the way divergence 20 did.
Falsifier of the framing: an arm that passes 7–9 while row 3 fails.

**Cheaper still, before the run.** Nothing cheaper than the probe already done exists for the
magnitude; the one addition worth its minute is running `gradprobe.py` on the 2M run's
checkpoint once that run is finished (not while it is live), for a second late snapshot.

## E. What is measured, read, and inferred

**Measured.** Every number in §A's two tables and §D's band table: the oracle's series from
`raif-paper/logs/oracle_state_mc_seed{0..3}/metrics.jsonl`; Arm B's from
`runs/armB_init_pair_100k/{scores,metrics}.jsonl`; the gradient norms, ratios, cosines and
coordinate-dominance fractions from `gradprobe.py` on a scratch copy of Arm B's checkpoint and
on step-0 agents built by `run_mc.build`.

**Read.** The gradient paths on both sides (§A, §B), the adjudicated text (§B), the assumptions
(§C), and the impossibility of (iii) (no encoder evaluation in imagination).

**Inferred.** That the withheld gradient is "small enough to doubt, not small enough to
dismiss" — an argument from norms at one late snapshot and one artificial early one, with
adam's per-coordinate normalization as the caveat; and the reading that ledger 5's adjudicated
content excludes the gradient scope, which rests on §(f)'s wording and is Alex's to confirm.
