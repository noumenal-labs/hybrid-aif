# Design: `replay` — a declared model driven through a recorded trace

*Status: ADJUDICATED 2026-09-18 (Alex): R1 amended to a function on the fault line below; R2–R8 as written; acceptance unnumbered at the end. **LANDED the same day** (chunk one: discrete and hybrid paths): `hybrid_aif.replay` (`replay`, `replay_document`, `stack_traces`, `Replay`), `infer_states(mask=)` and `fpi(masks=)`, `tests/test_replay.py`, a section in `../composing-systems.md`. *As built:* `action_log_prob` is the exact probability of the recorded first step under the planner's policy set (not a product of per-factor marginals); a masked-out switching observation leaves its factor at the prior padded to `mixture_k`, so the static and traced mask paths agree shape for shape; `replay_document`'s weight-drawing key is `build_key=` to keep it apart from the planner key; `externals=` is not on `replay` (only the amortized path reads it, and that path is chunk two); `exercised` is an `(M,)` bool array in observation order rather than a dict by address (a dict of Python bools cannot cross a jit boundary). An adversarial review before commit found that the scan's carry was seeded from unpadded priors while the faces carry mixture factors at `mixture_k` — now the seed is padded and the selection helper (`objectives.hybrid.where_tree`) refuses a shape mismatch instead of broadcasting one. `replay(..., legal=)` landed with `action-legality.md` L6 and `Replay.unsupported` with `zero-support-evidence.md` Z2 the same day (per-tick spellings with a leading time axis). **Chunk two landed 2026-09-19:** the amortized path replays through the per-tick faces (`infer_states` = the recognition modules, keyless; `update_empirical_prior` = their `img_step`), which is how the reference agent itself acts (`run_mc.collect`) — the training side's *sampled* filter `RSSMCore.observe` is a different convention and stays the trainers'; R5's draft named `observe` as the belief half and this is the amendment. Under a policy module the counterfactual `chosen` is the habit planner's draw under `key` (per-tick split), and `action_log_prob` is the actor's log-density of the recorded action at the tick's features — the self-imitation quantity, so workstream K's objective exists for continuous actions too. `externals=` takes `(T, width)` arrays sliced per tick. `learn=True` carries a `LearnState` for gradient slots and returns it (`Replay.learn_state`); on the amortized path `learn` itself says learning is `fit_world`'s. R6 on the amortized path: the modalities a recognition module reads are present together or absent together — absent is a predict-only tick, the prior returned as the belief — and a mask that splits them is refused; unread modalities may be `None` in the observation tuple (and `stack_traces` carries them as `None`). `unsupported` is `()` there. *Review round:* a **traced** mask that splits a module's parent tuple cannot be refused at trace time and reduces with `all` (absent) — so `replay` refuses the split concretely, by tick, before the scan, and `infer_states`' docstring says which it is; `None` observations are refused off the amortized path; `externals` are coerced to arrays; the actor is read from the live params inside the scan; and `inference.amortized.read_modalities` is the one definition of "the modalities a module reads". Companion to
`system.md` (the faces) and `model-description.md` (the document a model
travels as). First customer: zeroth-control's model-patch validator, whose
traces set several concrete requirements (see "First customer").*

## What a replay is

Every face in the canonical loop is pure and jit-safe, and the loop itself is
written eagerly on `System.learn`'s docstring. A **replay** is that loop
scanned over a *recorded* trace: at each tick the model infers from the
recorded observations, its own `act` runs as a counterfactual, and the
**recorded** action — not the model's — drives the empirical prior forward.
Teacher forcing, with the model's choice kept beside the record.

Four things want exactly this, none of them application-specific:

- **Behavioural model fitting** (workstream K): the likelihood of a recorded
  action sequence under a model — `Σ_t log P(a_t | o_{≤t})` — is the objective
  parameter recovery and model comparison fit, and it is a replay's by-product.
- **Offline validation of a changed model:** a patched or composed declaration
  is cheap to reject before any live trial if it can be driven through
  recorded episodes and its beliefs and choices compared with the record. This
  is the scoring side of the `model-description.md` loop — the document is the
  genotype, the replay is how a phenotype is read off it without a simulator.
- **Forensics:** the demo plan's "which term drove each action" panel is a
  replay's term trees read tick by tick.
- **Partial evidence** is a fact about real sensors, not a corner case: frames
  drop, modalities arrive at different rates, a trace records only what was
  logged. A replay that cannot mask a modality per tick cannot replay a real
  trace at all, so the mask is part of the feature (R6).

**What a replay is not.** It does not score. Whether a model's beliefs or
choices were *good* against a recorded outcome — calibration, regret, task
utility, truth columns — is the consumer's judgement, in the consumer's terms,
and stays outside the library behind one adapter from the consumer's trace to our
arrays. The function returns vectors; the verdict is someone else's.

## First customer

zeroth-control's model-patch loop (`zeroth-control/docs/model-patch.md`)
validates a proposed change to an active-inference model by rebuilding the
model, driving it through recorded episodes by the observe and advance faces,
asking a decision rule when the model would have acted, and scoring against
the recorded outcome. Its requirements on the replay (2026-09-18) are concrete
enough to be acceptance criteria and general enough not to shape the design:

1. **Teacher forcing with a counterfactual**, deterministic selection; per tick
   the beliefs, the policy posterior, the term tree and the model's chosen
   action. Their decision rule reads the posterior; the replay need not apply it.
2. **Partial evidence now:** their traces drop about one tick in four under a
   poor-sensor condition, and the live belief does a predict-only step there.
3. **Variable-length traces under jit** (up to 200 ticks, ending at a decision
   tick): padding plus validity is fine, but the library should own the convention.
4. **The adapter is theirs**, and so is every score.
5. **Accept a document, not only a built `System`**, so a patched model is
   replayed from exactly the artefact the proposer produced.

Item 2's predict-only step is precisely R6's semantics for a fully masked
tick — the prior pushed through the transition — a useful confirmation that
the exact reading is the one a real consumer already relies on.

## What exists

- The canonical loop is documented on `System.learn` and every face in it is
  pure and jit-safe; nothing scans it over recorded data. `composition.run` is
  a Python loop for *simulating* many minds against `as_env`, not for replaying
  a record.
- The amortized path already has the sequence face: `RSSMCore.observe` filters
  a `(T, …)` observation sequence under `actions[t]` producing `observations[t]`
  (`inference/rssm.py`), and `train.world.posterior_states` batches it.
- The training side's `Episode` (`train/buffers.py`) fixes the row convention:
  **row `t` pairs `obs[t]` with the action the policy drew *at* it**, so
  `action[t]` produced `obs[t+1]`; a reset row opens the episode.
- Discrete evidence enters the FPI as an observed log-likelihood message per
  modality (`inference/discrete.py:40-44`); LG evidence as one `condition` per
  observing modality. So "no evidence from modality m" has an exact meaning on
  both: the zero message, and no conditioning step.

## The decisions

### R1. One function, `replay(system, params, …)`, on the canonical loop — ADJUDICATED (amended)

```python
from hybrid_aif import replay
out = replay(system, params, observations, actions, *, mask=None, valid=None,
             key=None, priors=None, metrics=True, externals=None)
```

*Renamed (2026-10-07, `control-states.md` C10, "Presence, and the name of a flag"; C8,
"Legality by node"):* `mask=` is `present=` on `replay` and `infer_states`, a mapping keyed
by modality name whose entries say whether the modality was observed (a `(T,)` bool on
`replay`); a modality left out is present, `None` means all are, and an unknown key is
refused. `stack_traces` returns `(observations, actions, present, valid)`. `legal=` is keyed
by control-node name (each entry `(T, levels)` or `(T, H, levels)` on `replay`, a node left
out unrestricted), or the bare `(T, P)` mask over the policy set.

`observations` is a tuple, one `(T, …)` array per observation variable in
declaration order (the same typing as `infer_states`, with a leading time
axis); `actions` is `(T, …)` in the System's action layout. Row convention is
`Episode`'s: `actions[t]` is the action taken after `observations[t]`. The loop,
scanned with `lax.scan`:

```
beliefs_t        = infer_states(params, observations[t], priors_t)
q_pi_t, tree_t   = infer_policies(params, beliefs_t)          # the counterfactual
priors_{t+1}     = update_empirical_prior(params, beliefs_t, actions[t])   # the record
```

The belief path follows what the recorded controller *did*; the model's own
policy posterior is computed at every tick and never applied. `priors=None`
starts from `initial_state_prior`, as the first tick of an episode does.

**Why a function and not a face (the fault line, Alex 2026-09-18).** The first
draft made this `System.replay`, a fifth verb beside perceive, learn, act and
carry. Alex: replay "is more of an external *use* of a System/model than
something it intrinsically does." The repo already practises the line that
settles it, and it is worth writing down:

> A **method** is something the agent computes about itself within a tick,
> from its own state, across its own blanket — `infer_states`, `learn`,
> `act`, `update_empirical_prior`, `predict_beliefs`, `score_policies`,
> `as_env`. A **function** is something an outside party does *with* a
> System: represent it (`describe`, `build`), compose it (`merge`,
> `subsystem`), drive it (`composition.run`), train it (`imagine`,
> `fit_world`).

A replay drives a model over a record; it is the sibling of `composition.run`
(a driver over a simulation), not of `act`. So: `hybrid_aif.replay`, a module
exposing `replay`, `replay_document` (R8), `Replay` and `stack_traces` (R7),
exported from the top level. *Naming note:* `train/buffers.py` holds
**experience** replay (`ReplayBuffer`); the two senses are distinct and the
training module keeps its name — renaming it `train/buffers.py` is a
cosmetic option if the collision ever bites.

### R2. What comes back — ADJUDICATED

```python
class Replay(NamedTuple):
    beliefs: tuple[Belief, ...]      # each leaf (T, …): the posterior at every tick
    priors:  tuple[Belief, ...]      # (T, …): the prior each tick was inferred under
    q_pi:    Categorical             # (T, |Π|): the counterfactual policy posterior
    trees:   TermTree | None         # (T,) per leaf, or None under metrics=False
    chosen: Array                    # (T, …): the model's own action, deterministic selection
    action_log_prob: Array | None    # (T,): log P(actions[t]) under the model — R3
    log_evidence: Array | None       # (T,): per-tick log-evidence where the path computes it
    valid: Array                     # (T,): the tick is real, not padding — R7
```

*Amended 2026-10-15 (description-length.md D6):* `free_energy: Array | None`, each tick's
variational free energy (`InferenceDetail.free_energy`), joined the record, and
`log_evidence` became its negation on every path but the amortized one: the exact log evidence
where the tick's posterior is exact (every modality LG, or one categorical factor), a lower
bound otherwise; summed over the valid ticks it is the prequential description length of the
record under the model as it stood at each tick. Before this the field held only the LG
modalities' evidence and was `None` on the discrete path. With `learn=True`, a modality absent
at a tick (`present`, or a padded tick) teaches its likelihood slot nothing there, so the sum
over the valid ticks is prequential whatever the padding.

`chosen` is `act(selection="deterministic", mode="marginal")`'s action at every
tick — the counterfactual read beside the record.
It is computed from the same `q_pi` the function returns, so it costs nothing extra.

Stacked over time, nothing summarised: a scorer wants the vectors
(`ReturnEstimates` is the precedent for returning vectors rather than means).

### R3. The likelihood of the recorded action — ADJUDICATED

`action_log_prob[t]` is `log P(a_t = actions[t])` under the model's **marginal
action posterior** at tick `t` — `act(mode="marginal")`'s distribution over
first actions, which sums `q(π)` over the policies sharing a first step. This
is the quantity workstream K fits and model comparison ranks by, and it is
well-defined on the discrete and hybrid paths. For continuous controls (CEM) and
the habit planner there is no categorical posterior over actions — CEM has a
proposal, the habit planner an actor distribution — so the first chunk returns
`None` there and records the gap (it is ledger entry 11's shape: the planner
protocol has no slot for a continuous action distribution).

### R4. Learning along the trace — an option, off by default — ADJUDICATED

`replay(..., learn=False)`. With `learn=True` the loop places `learn` between
perception and action exactly as the canonical loop does, feeding it the
recorded previous action, and the function returns the final `params` as well:
"what this model would have *learned* from this episode". Conjugate slots only
in the first chunk (gradient slots need a `LearnState` carry; the scan can
thread one, but that is chunk two's business). Off by default because a
scorer compares fixed models.

### R5. Which paths, in which order — ADJUDICATED

Chunk one: the discrete and hybrid paths, which share `infer_states` /
`infer_policies` / `update_empirical_prior` as written. Chunk two: the
amortized path, whose belief half *is* `RSSMCore.observe` (already a scan) and
whose counterfactual half is the habit planner or `score_policies`; `replay`
dispatches to it the way every face dispatches on `_is_amortized`.

### R6. Masked evidence — ADJUDICATED (single chunk)

`infer_states(params, observations, priors, mask=None)` with `mask` a tuple of
booleans (or a `(M,)` array) per observation variable: a masked modality
contributes the **zero log-likelihood message** on the discrete path and is
**skipped** by the LG `condition` sweep; the switching and mixture paths skip
their grow-condition for it. `replay` takes `mask` as `(T, M)` and threads it.
Semantics are exact, not approximate: a missing observation is a variable
that was not clamped, which is the `infer(query, clamped)` layer's reading
(brief, open question 13) arriving one more time.

With a mask, `replay` can also say **what the trace exercised**: an address
whose modality was masked on every tick (`likelihood:m`, and the infogain
leaves keyed by `m`) was never informed, and the result carries
`exercised: dict[address, bool]` so a scorer never silently scores a subset —
the PR body's ask, in the library's own vocabulary.

*Build order:* the mask lands **in the same chunk** as R1–R5 for the discrete
and hybrid paths — a replay that cannot mask a modality cannot replay a real
trace (Alex, 2026-09-18: single chunk). It touches the
inference faces, so it carries its own parity suite: masked ≡ unmasked when
the mask is all true; a fully masked tick ≡ the prior pushed through the
transition (a dropped frame); a partially masked tick ≡ inference on
the sub-model of the unmasked modalities.

*Spelling (2026-10-07):* the keyword is `present=` (see the note under R1).

### R7. Variable-length traces: the function owns padding and validity — ADJUDICATED

Traces differ in length (theirs: up to 200 ticks, ending at the decision
tick), and a scan needs one static `T`. `replay` takes `valid: (T,) bool`
(default: all true) and treats an invalid tick as a **no-op on the carry**:
priors pass through unchanged, and the per-tick outputs at that row are
unspecified and must be read through `Replay.valid`, which the function returns.
A helper, `stack_traces(traces) -> (observations, actions, mask, valid)`, pads
a list of variable-length traces to a common `T` with the convention built in,
so no caller invents one; `jax.vmap(replay, ...)` over the
stacked leading axis is the batch, and the population contract already
guarantees the faces vmap.

### R8. Two entry points: the function, and one that takes a document — ADJUDICATED (amended)

`replay(system, params, …)` takes a built declaration.
`replay_document(document, observations, actions, *, arrays=None, key=None, …)`
takes a **description document** (a `dict`, with its sidecar and, for
recipe-only neural declarations, a key), calls `build` first — with its total
validation — and then `replay`. A changed model is thereby replayed from
exactly the artefact that describes it, and a document that does not describe
a `System` fails there with `build`'s sentence rather than deep in a scan. Two
explicit functions rather than one polymorphic first argument, in the repo's
refuse-by-name style; both exported from the top level beside `describe` and
`build`.

## Not decided here

- Any trace *format* (columns, reset rows, truth): a consumer's, behind its
  adapter. Our arrays are the seam.
- Scoring of any kind.
- Whether `replay` should also be the training side's `Episode` consumer
  (the world-model trainer has its own scan and its own conventions). Placed
  on the roadmap in `../project-brief.md` (flagged 2026-09-18) so it is
  decided rather than left hanging.

## Acceptance

- **Fidelity:** on the T-maze, `replay` over a trace produced by
  `composition.run` (or the eager canonical loop) reproduces the eager loop's
  beliefs and `q_pi` bitwise at every tick.
- **The record wins:** replaying a trace whose recorded actions differ from
  the model's argmax follows the record — the beliefs are those of the recorded
  path, and `action_log_prob` is low where the record disagreed.
- **K's quantity:** `action_log_prob.sum()` over a trace generated by the model
  itself is higher than under a model with a perturbed likelihood; the fitting
  stack's objective exists before the fitting stack does.
- **Hybrid:** the same function on `hill_lg`'s trace, with LG evidence.
- **Masks:** the three parity identities of R6, on both paths.
- **Padding:** a padded trace replays identically to the unpadded one on its
  valid ticks; a batch of two traces of different lengths vmaps.
- **Document:** `replay_document(describe(...).document, ...)` equals
  `replay(system, params, ...)`; an ill-formed document fails with `build`'s sentence.
- The composer's reference gains a section; the notebook gate runs (public
  API).
