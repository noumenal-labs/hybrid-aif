# Design: steps of different durations, and the preferences of a slow tier

*Status: SETTLED 2026-10-20 (Alex), T1–T8, walked 2026-10-19/20; built the same day, see "Build
notes". Customer: the drive agent of a separate, Noumenal-internal repository, whose request of
2026-10-19 is restated under "Why" so this document stands alone; a second reader is a hierarchical
body model whose two tiers are two Systems. Companions: `nesting-systems.md` (the tick semantics,
N1's conditional terminal target), `model-description-0.3.md` 18.5a (a target is a factor record
with parents; the conditional-C gap row), `control-states.md` (the control-node `prior` row, C12
actuators), `per-variable-infogain.md` V11 decisions 6 and 8 (cross-entropy targets inside a
branching search, the conditional terminal target's contraction), `closed-loop-planning.md`
(the per-step weights of the Riccati pass), `body-model.md` on branch `feat/body-model`
(B4, the per-level action cost; unmerged). Sources: Sutton, Precup & Singh 1999, *Between MDPs and
semi-MDPs: a framework for temporal abstraction in reinforcement learning* (Artificial
Intelligence 112); Friston, Parr et al. 2023, *Active inference and intentional behaviour*
(arXiv:2312.07547, the intended-state prior N1 builds on).*

## Why

The drive agent plans at two rates. A fast planner steers along a pile row at a fixed tick; a
slower model, which they build, decides at row ends and U-turns whether to drive the next row or
first ask an operator where the row is, then wait for the answer. Its steps are a row (about
150 s), a lap (about 600 s) or an ask and its wait (10–30 s), and its horizon is the laps left in
the mission. Their request, in order of need: a duration per step that may differ within one
plan; transitions over that duration; preferences that accumulate over it ("holding still for
20 s should cost what 40 fast steps of the speed preference cost", inside G, with no price set
outside); an observation at the step's end (the answer, or a summary of what a row's readings
will have told); closed-loop scoring at this level; and, nice to have, a random wait, its mean a
fine first version.

The library has no clock. Every factor steps once per round, under every planner and every
composition (`nesting-systems.md`, tick-semantics amendment of 2026-10-14), and a slow tier in a
merged composite is a sticky factor, held exactly by a clock factor as parent, scored at every
fast tick. That form is exact and needs nothing here; it is also unusable for the drive agent's
slow model, whose horizon at the fast rate is thousands of ticks with a branch on the operator's
answer at each. Their slow model has to step once per option. This document says what the
library needs so it can, and keeps the objective where hierarchical active inference puts it.

## What was settled before, quoted

**One EFE per planner.** Each level of a hierarchical model has its own generative model and its
own expected free energy; the upper level's choices become priors for the lower level (our
`D_deps` and `H_deps`, N1), and the lower level's result returns as an observation the upper level
has preferences over. The upper level never reads the lower level's costs.

**A target is a factor record with parents** (18.5a, Alex, 2026-10-12): "A target's `parents` are
its conditioning variables: the terminal target's today (`H_deps`), and every target's once the
engine scores a conditional C (a gap row in the build plan; Alex: the library's C is conditional
in general)." And, in the same ruling: "`prior` under a control node is distinct from `target`
under it, as the engine is: a target on actions is a cost inside G under the precision, a prior is
E outside it."

**A target on a discrete control node is scored today** (`controls.py`, `node_risk`): a
`Categorical` over the node's levels in `state_targets`, charged as `−log C(level chosen)` per
step under `risk`. A colleague's B4 (`body-model.md`, proposed 2026-10-10, unmerged), "a cost per
control level in nats, a test's seconds, a slower plan's extra time", is this record under
another spelling: `Categorical(logits=−c)` charges `c_k` up to a constant shared by every level.

**The options framework** (Sutton, Precup & Singh): an option has a multi-step transition model,
an accumulated reward model declared per option, and a duration used for discounting. Two
timescales are two decision processes, and an option's model is written for the whole option.

## Decisions

**T1 — A duration is a property of the round, declared on a control node's levels.** A
`SystemParams.durations` entry, `{"u_option": [300.0, 40.0]}`: one positive real per level of
one discrete control node, in whatever unit the model's preferences are per (ticks of the fast
tier, seconds; the engine checks positivity only). A round is one step of every factor, so a
round has one duration, and the node that carries it sets the clock for the round: a model
declares durations on at most one control node, and `merge` refuses two components that both
do, and a binding that removes the node carrying them. The general form, a table over the joint levels of several control nodes (DRIVE at fast,
DRIVE at slow, ASK at either), is the same dict under a tuple key and a `clock` record in the
document; it is named here and not built. Absent durations mean 1 for every level, today's
behaviour unchanged.

*Why a parameter, not a static declaration:* the box of a continuous control node is the
precedent, numeric, per node, in `params.control`, rendered on the variable record. A duration
may one day be learned or carried as a distribution; `dims`, the static precedent, has no such
future. *Why not asynchronous:* separate durations for actions taken in the same round are
concurrent processes whose round has no single end, a different engine (deferred at T1 of the
walk). *Why not a fixed `clock` per System:* the drive agent's rounds genuinely differ (a row,
a wait).

**T2 — Two tiers are two Systems; one clock per System.** A tier that steps once per option is
its own System, coupled to the fast tier by what the drive agent already does (the slow choice
sets the fast planner's targets, through `H_deps` on the slow control node under `merge`, or
through the preference parameters they set each call) and by the fast tier's filtered belief
carried in as the slow System's factor. The colleague's body model composes the same way: a filtering
System per tick and a choosing System per decision, "two Systems over one belief". A model whose
slow horizon is short in fast ticks keeps the sticky-factor form instead; the two forms are not
rivals, they fit different horizons.

**T3 — A slow step is one application of the option's whole-step kernel.** Each level of the
duration-carrying control node selects a kernel the user writes for the whole step, in today's
declarations: a per-level `F` and innovation under `LinearGaussianTransition` (the switching
pattern, `innovations` parented on the control node), a per-level `f` under
`NonlinearGaussianTransition` (the drive agent's decay `e^(−Δ/τ)` is written into the level's `f`
with that level's Δ), a per-level column under a tabular transition. The engine composes nothing
and integrates nothing. Two helpers make the two mechanical cases one call: `compose(kernel, n)`
for a step that is `n` repeats of a per-tick linear-Gaussian kernel, and a discretiser taking a
continuous-time `(A, B, Q)` and Δ, one Δ per level giving a per-mode kernel whose innovation's
parent axis is the level; both return an ordinary kernel. Both take the Cholesky factor of the
composed noise, so a per-tick noise singular in a direction the step reaches is the user's to
avoid.

*Why not compose a per-tick kernel inside the engine:* a step that is the same per-tick kernel
repeated is a homogeneous drift; neither customer's options are (a row is a new heading with a
learned offset at its end, a reach is a goal-directed trajectory). The whole-step kernel is the
weaker assumption, and the SMDP's option model is written the same way.

*The consistency burden, stated once:* every factor's kernel describes its state at the round's
end. A factor driven by some other control node, in a model whose clock is set by `u_option`,
has a kernel that must still be right for a 40-unit round and a 300-unit round; if the two
differ it reads `u_option` too. The engine cannot check this.

**T4 — The objective never reads durations.** The walk first derived a "temper", Δ times each
cross-entropy against a target, as what EFE gives when a slow step's preference is the fast
tier's per-tick preference accumulated over the step. The derivation is right and the premise is
a modelling choice, the drive agent's, not a consequence of the theory: the upper level's
preferences are its own, on what it observes. Three things were bundled under "duration", and
they come apart:

1. the round's clock (T1, T8): no objective involvement;
2. a fixed cost per option: a `Categorical` target on the control node, scored today (T6);
3. a belief-dependent cost that scales with an option, the lane error over a 300-unit row
   against a 40-unit wait: a target whose parameters depend on the level chosen, a target
   conditional on the control node (T5).

Nothing is expressible with the temper alone. The temper with duration Δ_k is the conditional
target with precision Δ_k·Λ (a Gaussian) or log-table Δ_k·log C (a categorical), up to a
per-level constant, and a constant is the record of item 2 where it is visible. The temper's
constant is also wrong on its own terms: a Gaussian target's cross-entropy carries the density's
normaliser, which cancels today because every plan has the same length, and under the temper
becomes Δ_k times a constant that is negative for a tight preference, a reward for an option's
length. A conditional target scores each level's normalised density once. A helper that writes
the conditional target from a per-tick target and the durations (the scaled precisions, the
quadratic part only) is built with T5 as `accumulated_target` (see there).

**T5 — Per-step targets conditional on a control node, the adopted gap row, scored.**
`obs_targets` and `state_targets` accept a conditional declaration, written as an option's
kernel is written: a `ConditionalCategorical` (log-table, child axis first, then one axis per
parent) or a `ConditionalGaussian` (`loc` `(*parent_shape, d)`, `chol` `(*parent_shape, d, d)`),
with the parents named in a keyed `System(C_deps={"lane": ("u_option",)})` (the SPM letter for
C, keyed by the targeted node as the keyed faces are). `C_deps` is the first structural
declaration written by name: `A_deps`, `B_deps`, `D_deps`, `H_deps` and `B_control_deps` are
node-form tuples by position, and `control-states.md` records declarations by name as "a
candidate to weigh with Phase 5, not a decision" (Alex, 2026-10-06), because the hazards that
compelled keyed faces, per-call data misassigned silently, do not apply to a one-time
declaration that table shapes mostly check. That is a reason it was not urgent, not a reason
against; a new declaration should not take a form expected to retire, and the siblings follow
when that item is decided (Alex, 2026-10-20).
A parent is a discrete control node or a state factor. Resolution reuses N1's mean-field rule
(`resolve_conditional_targets`): at each step the target is `Σ_parents C(· | parents)
q(parents)`, where a clamped control node's `q` is its one-hot level, so with a control parent
the target at the step is simply the level's own distribution, and with a state parent it is the
mean-field average, a mixture for a Gaussian family. The resolved target is then scored exactly
as an unconditional one: KL in a flat planner, cross-entropy inside a branching search (V11
decision 6), cross-entropy on the closed-loop distribution under `feedback="linear_quadratic"`,
and under closed-loop scoring the level's precision `Λ_k` is the step's weight in the Riccati
pass (`W_τ = TᵀΛ_τT`), which already runs per step; a control node's own target is its fixed
cost per level and is never conditional, so `R` is unchanged. A sampling backend sees a conditional target as what it is, a factor
with the control node as parent (`thrml` C1), with no per-option scaling of energies.

*A state parent inside a branching search:* the per-step rule above resolves the target
mean-field and takes the cross-entropy against the resolved distribution, while the conditional
terminal target under branching (V11 decision 8) takes the negative expected log-probability
`−Σ_s q(s) Σ_pa q(pa) log C(s|pa)`; the two differ by a Jensen gap. With a control parent, this
build's purpose, a clamped level makes the resolution a selection and the forms coincide. If a
state-parented per-step target is ever relied on inside a branching search, the expected-log form
is the one to adopt, for the same reason decision 8 adopted it.

*Scope of this build:* control parents on both families, state parents where the same resolver
gives them for free (the categorical case does; a Gaussian target with a state parent is a
mixture and raises, as N1's resolver already says for continuous conditional targets). Beside
it, the third helper of T3's module: `accumulated_target(target, durations)` turns a per-tick
`Gaussian` or `Categorical` target into the conditional target over the levels, precision or
log-table scaled by each level's duration, the same shape as `compose` (a per-tick thing and the
durations in, a per-level thing out). Its docstring says what T4 says: each level is a normalised
density with its own normaliser, and a fixed cost per option is the control node's own target
(T6), never the helper's. The format line "a `target` takes no parents (the engine scores unconditional targets; a conditional
target is refused)" is amended to "a `target`'s `parents` are its conditioning variables, a
discrete control node or a state factor", and the gap row closes.

**T6 — A fixed cost per option is the control node's own target.** Already scored: a
`Categorical` in `state_targets` keyed by the discrete control node charges `−log C(level)` per
step. A cost `c_k` in nats is `Categorical(logits=−c)`; the normaliser is shared across levels and
cancels in ranking. This is the colleague's B4 under the adopted spelling and needs no row. It is inside
G, under γ, which is what distinguishes it from a `prior` on the node (E, outside γ, the habit),
as 18.5a already says.

**T7 — Information is scored once per observation, at each level.** A slow System's channels
read once per round, at the step's end (the answer, a row's summary with its expected
precision); their information gain and ambiguity are properties of that one reading and take no
factor. This is not a rule about durations, it is the tick semantics: there is no per-tick
stream in the slow model for a factor to stand in for. For a static quantity read `n` times, the
pooled reading's information is ½·log(1 + n·σ²_prior/σ²_read), sublinear in `n`, which is why
"multiply by Δ" would be wrong even for a model that wanted it.

**T8 — Execution: the realised action's duration is the elapsed time.** The environment keeps no
clock. `System.elapsed(action, params)` returns the duration of a realised keyed action (the
realised level under a stochastic actuator, since an action node shares its control node's
levels), for an outer loop to advance its own clock by. A random wait is deferred: a duration is
one number per level now, its mean, and a distribution per level is the named extension.

## The format

- A control node's variable record gains `duration`: a positive number per level, or one number
  for every level, written only when declared (`"u_option": {"domain": "categorical", "levels": 2,
  "duration": [300.0, 40.0]}`), beside the box precedent `low`/`high` on a continuous node.
  Schema 0.3 adds it under `leaf`, dependent on `levels`.
- A `target` record takes `parents` (18.5a's spelling), each a discrete control node or a state
  factor, read at the same step. Families: `ConditionalCategorical`, `ConditionalGaussian`
  (the latter added to the families table under `target`).
- `describe` writes both; a document without them builds bit-for-bit as before.
- Pricing: a control node is priced as a node, continuous or not, and its box's numbers are not
  priced; durations are not priced in the same way. A conditional target is an objective record
  and the objective is not priced (`description-format.md`, the generative partition decides).
- `merge`: durations of each component carry over under its prefix; two components that both
  declare durations are refused with a sentence; `C_deps` namespaces like `H_deps`.

## The customer's slow model, written

One control node `u_option` with levels DRIVE (300) and ASK (40). Factors: `pose` (the lane
error, Gaussian), `b` (the row's offset, Gaussian, static). Kernels per level: under DRIVE,
`pose` moves one row with the row's process noise and `b` holds; under ASK, both hold. Channels:
`gps` read every step (its precision under DRIVE the pooled precision of a row's readings),
`answer` read only under ASK (present by the level; the mode-parent form of the next walk item).
Preferences:

- `state_targets["pose"]`: a `ConditionalGaussian` on `u_option`, mean 0 at both levels,
  precision 300·Λ under DRIVE and 40·Λ under ASK, Λ the fast tier's per-tick lane precision;
- `state_targets["u_option"]`: `Categorical(logits=[0, −c_ask])`, the fixed cost of an ask;
- `terminal_targets["b"]` or the row's own target, as they choose.

`feedback="linear_quadratic"` under `planning_belief="posterior"` then credits an answer with
what it lets the corrections remove over every later row (`closed-loop-planning.md`), and the
branching planner with `branch_on="answer"` credits what the slow model decides differently
after it. Their item 3 is met by two declarations they write once; their item 2 by the per-level
kernels; their items 4 and 5 by what exists; their item 1 by `durations`, which here only tells
an outer loop how much time passed.

## Acceptance

1. *Dwell equivalence.* A linear model, one static state (a process noise of 1e-4, since a
   zero is refused), one reading, `planning_belief="posterior"`: the fast tier's `n`
   information-gain leaves then telescope to the pooled reading's gain, where under `"prior"`
   each would read the prior belief and the sum would be `n` times one reading's; the risk leaf
   reads the predictive belief, constant for a static state, under either. A slow System with one
   level of duration `n`, a conditional target of precision `n·Λ`, and a reading of the pooled
   precision, scores the same information gain as `n` fast steps under precision Λ with `n`
   readings (exact to tolerance: `n` readings of a static state pool), and a utility leaf equal to the fast tier's sum up to the density
   normalisers (`½·log det(2π(nΛ)⁻¹)` once against `n·½·log det(2πΛ⁻¹)`), so the test compares
   the quadratic parts, or the difference between two plans, where the constants cancel. Under
   `risk_ambiguity` the entropy terms are bookkept differently (one predicted reading against
   `n`) and the test checks that the gap between the decompositions is unchanged by the switch,
   as the closed-loop tests do.
2. *Payoff across laps.* Two rows visited `L` times; asking costs a fixed wait (the control
   node's target) and tightens `b` for every later visit (the conditional lane target under
   closed-loop scoring). Ask wins for large `L` and loses for `L = 1`; the crossing moves with
   `c_ask` and with the lane precision as the derivation says.
3. *Conditional targets.* A categorical target with a control parent selects the level's column
   (tested against an unconditional target per level); a Gaussian target with a control parent
   selects the level's `(loc, chol)`; a state parent on a categorical target averages mean-field
   and matches the terminal resolver; a Gaussian target with a state parent raises by name; the
   closed-loop Riccati pass takes the level's `Λ`.
4. *Durations are inert in the objective.* A model with durations declared scores every plan
   bit-for-bit as the same model without them.
5. *Round trip.* `describe` → `build` keeps `duration` and `parents`; the schema accepts both;
   a document without them is unchanged. `merge` carries both and refuses two clocks.
6. *Helpers.* `compose(kernel, n)` equals `n` applications in belief space; the discretiser
   matches the matrix exponential's exact discretisation on a scalar system;
   `accumulated_target` on a Gaussian gives precision `Δ_k·Λ` per level and, on a categorical,
   log-table `Δ_k·log C` renormalised per level, and scoring the conditional target equals
   scoring the per-tick target `Δ_k` times up to each level's normaliser.
7. *Elapsed.* `System.elapsed` returns the realised level's duration, under an actuator the
   realised one.

## Deferred, named

- The joint-action clock table and its `clock` document record (T1).
- A random duration per level, the wait's distribution (T8).
- Gaussian targets with state parents (a mixture target per step).
- The asynchronous model, where actions in one round have their own ends.
- Discounting: the library has none, so a duration enters no discount; the SMDP's `γ^Δ` is not
  built.
- A presence table per level of a control node, for a channel present under some levels only:
  parked on 2026-10-20, since the answer is a modality over the action node and a categorical
  channel's null outcome is a deterministic reading (`action-node-modalities.md`); an exact null
  outcome for a designated Gaussian channel would extend `present=` (built 2026-10-21 as a
  per-level entry of `present=`; planning-with-observations.md D4).

## Build notes (2026-10-20)

Built in two parts by subagents on this document, reviewed adversarially, the review's findings
fixed; 2,469 tests, the notebook gate green.

- **Closed loop takes the level's `Λ`, not `R`.** T5 and acceptance 3 first said the Riccati
  pass takes the level's `R`; the build takes the chosen level's state or observation target
  precision as the step's weight, `W_τ = TᵀΛ_τT`, and leaves `R` alone, since a control node's
  own target is its fixed cost per level and is never conditional. The sentences above now say
  so.
- **A relayed parent is refused.** Under `merge`, an `ActionControl` whose driver is a state
  factor relays the control node to the previous step's state. A per-step target reads the
  step's own belief, so a `C_deps` parent that a relay would re-point at a state is refused with
  a sentence (the review found the same-step reading off by one step on a two-maze composite);
  identification with a driver's control node carries, and the composite scores the driver's
  level column at each step.
- **Acceptance 1 holds under `planning_belief="posterior"`** (the information-gain half
  telescopes only there) with a process noise of 1e-4, the engine refusing a zero; the sentence
  above was corrected before the test was written.
- **The payoff toy** (`tests/test_conditional_targets.py`,
  `test_asking_pays_across_laps_and_the_crossing_rises_with_the_asks_cost`): a Gaussian lane
  factor `(e, b, a)`, `u_option ∈ {DRIVE, ASK, IDLE}`, a GPS reading every step and an answer
  about `b` whose noise is a `GridNoiseMap` over the indicator `a` that ASK's kernel sets (a
  channel present under one level only is written, since 2026-10-20, as a modality over the
  action node; see the dated note below). Both plans drive `L` rows and wait once, `IDLE · DRIVE^L` against `ASK · DRIVE^L`, so the crossing
  is the ask's cost against its per-row saving. Under `feedback="linear_quadratic"` and
  `planning_belief="posterior"`, the lane target `300Λ` under DRIVE and `40Λ` under the waits:

  | `c_ask` | L=1 | L=2 | L=3 | L=4 | L=6 | L=8 | crossing |
  |---|---|---|---|---|---|---|---|
  | 20, never − ask-first (nats) | −10.6 | −3.8 | +2.9 | +9.6 | +23.0 | +36.5 | 3 |
  | 45, never − ask-first (nats) | −35.6 | −28.8 | −22.1 | −15.4 | −2.0 | +11.5 | 7 |

  The per-row saving is about 6.7 nats at every `L`. The review's first version of the toy drove
  one more row under never-ask than under ask-first, and its crossing mixed that step-cost
  asymmetry with the mechanism; the idle step removes it. Over every sequence of the three
  levels the global best is to idle throughout, since the toy rewards no row: the test compares
  the two named plans.

  Correction (2026-10-20): the table was first computed with DRIVE holding the indicator, so
  the answer was re-read on every row after an ASK. DRIVE now clears the indicator (ASK's
  innovation still sets it) and the answer is read at the step the ask is taken; the rows above
  are the corrected ones, the crossings unchanged at 3 and 7, the saving about 6.7 nats per
  later row.
- **The channel present under one level, written as a modality over the action node
  (2026-10-20).** The answer channel is now declarable as a linear-Gaussian observation over
  `(lane, a_option)` with a wide noise under DRIVE and IDLE and the answer's sd under ASK, the
  lane factor `(e, b)` carrying no indicator (`docs/design/action-node-modalities.md`,
  `tests/test_action_node_modalities.py`). The indicator form above remains valid and keeps its
  test.
- **Validation.** The new validator also checks a conditional family against its key's carrier
  (a `ConditionalGaussian` on a tabular modality, a `ConditionalCategorical` on a Gaussian
  factor); the plain families still crash past validation there, a pre-existing gap. A
  categorical state parent at a hard zero contracts under a mask so the gradient stays finite,
  as decision 8's terminal contraction does. Durations keep the dtype given.
- **Reviewer could not verify:** the corrected closed-loop leaf of a level-changing plan against
  an independent Riccati reference (the per-step weights were checked at every step, the
  recursion's indexing read); the amortized-path refusal (no fixture).
