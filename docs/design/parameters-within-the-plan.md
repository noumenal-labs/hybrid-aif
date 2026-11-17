# A parameter read within a plan

*Status: design, 2026-11-16, from dialogue with Alex over the survey rover's terrain; built
2026-11-16 (As built, below). Customer: the rover's switching dynamics
(`nonlinear-transitions.md`, the N4 amendment),
whose epistemic interval over the terrain stays closed because a plan redraws the terrain afresh
at every tick. Companions: `terrain-field.md` (the cell read, the one-tick lag, no spatial
prior), `mixture-filtering.md` D2 (mode-owned slots), `parameter-learning.md` D6 (the novelty
term), `planning-with-observations.md` D1 (the precedent for a richer planning carry).*

## What is wanted

The model says a learnable table is one unknown quantity, shared by every tick that reads it. A
plan reads it at its posterior mean and draws each tick's child afresh, so two reads of one row
within a plan are independent, and the novelty term credits the twelfth read of a row as much as
the first. Both are one approximation: the parameter frozen within the plan, pymdp's convention
and the one `compile_tables` hoists on the fused path. The exact treatment of a conjugate table
within a plan is its predictive: an imagined child adds a count to its row for the rest of the
plan (the Pólya urn), the next read of the row is the Dirichlet-multinomial predictive, and the
novelty of a later read is that of the updated counts. Prediction and novelty then come from one
model, which is the test this library sets itself.

The case that motivated this is the repository's survey example (`examples/survey_*.py`, the
project page's demo): a rover crosses a walled field of three terrain classes, reads the class of
the cell under it through a camera, and learns a per-cell table of the field, one Dirichlet row
per cell, which its plans read at the cell each imagined step reaches. There the difference is
the whole epistemic interval. The field's prior is sparse (`FIELD_STRENGTH`, 0.1 pseudo-counts
per cell), which says each cell is nearly one class, unknown which. The predictive after one imagined read of an unmapped cell is `(α_j + 1) / (α_0 + 1)`, about
0.94 for the class just imagined, and the measured per-tick stay rate of the true terrain under the
rover is 0.945. The persistence the switching slots need is in the model as declared; the frozen
reading throws it away. Persistence across cells is not in the model: `terrain-field.md` keeps the
cells independent and says smoothness, if ever wanted, enters there as a basis change. A
two-resolution field (a coarse field centring each cell's prior) was weighed and is not clearly a
library piece; it stays a modelling question for the example, not a capability.

No persistence is declared: the alternative weighed first, a sticky terrain transition with a stay
probability of its own, was withdrawn as a second model under one name (prediction-only mixture
plus the plain map's update). The cell read `cell_mass` is the likelihood of the cell given the
position, the motion model moves the position, and the persistence of the class under a rover that
stays in its cell falls out of the two once the table is read as the one shared quantity it is.

## D1 — Working counts in the planning carry

A learnable tabular declaration (a `ConditionalDirichlet` in `transition` or `likelihood`,
directly or as the home of a `Parameter`) that the System declares *learned within the plan* is
carried through the planning scan as working counts:

- seeded from the posterior the plan starts from;
- absorbing, at each imagined step, the step's expected count statistic: the model's joint
  over child and parents at that step, the working mean times the parents as the tabular reader
  reads them (a transition's `B(s' | p) Π q(p)`, a cell parent as its mass of the pre-step
  belief, a control axis as the taken level's one-hot; a likelihood's `A(o | s) Π q(s)`), at
  learning rate one, no decay, no anchor. *Revised 2026-11-17 (Alex):* this was "the statistic
  `learn` would absorb from the same beliefs", the product `q(child) ⊗ Π q(p)` of the
  post-step marginals. The joint is the exact expectation of `onehot(child) ⊗ onehot(parents)`
  under the model when the child is not observed; the product of marginals is the closed-loop
  face's convention (it has an observed outcome and filtered marginals), and differs from the
  joint wherever child and parents are dependent under the step (their KL is the mutual
  information). A consequence: absorbing the joint scales each parent configuration's counts
  by `1 + q(p)/α_0(p)`, so along an unbranched plan the working mean of a table never moves and
  only its concentration grows (the urn is a martingale); prediction departs from the frozen
  reading only where the plan conditions on a drawn child, in a branch or a closed pair's
  per-mode components, and the novelty declines either way;
- read, at each step, as the posterior the counts define: prediction through its mean, novelty
  through its `novelty()` weights, and every other reader of the table inside the plan (an
  occupancy row, `branch_on`'s expected table) through the same working posterior.

The novelty of a step is that of the counts before the step's absorption, as in the closed loop.
The counts are a belief, keyed by the reader (`transition:<factor>`, `likelihood:<modality>`);
readers of one `Parameter` home that are all learned within the plan absorb into, and read, one
working table, keyed by the home. A reader left frozen reads the declaration, as today.

Declaration: `System(plan_learning=...)`, a tuple of keys in the grammar of `learn`'s rates
(`transition:<factor>`, `likelihood:<modality>`, `parameter:<home>` for every reader of a home),
or `True` for every learnable table; default `()`, today's frozen reading. A key naming no
learnable table is refused with a sentence. The setting is an inference setting, as
`planning_belief` and `mixture_schedule` are: `describe` writes it under
`configuration.inference`, `build` reads it back, it adds nothing to the description length, and
under `merge` and `subsystem` it composes and narrows as `epistemic`'s keys do.

What stays approximate: beliefs within a step are mean-field, so the statistic absorbed is an
expected count, not a drawn one, and a sampled rollout absorbs the sample's. Within a mixture, D2
recovers the draw of the mode.

## D2 — The joint read over a mixture's components

When a categorical factor reads a mixture-carried factor through a cell and is itself a mode
parent of that mixture (the rover's closed pair: `terrain` reads `cell(x)`, `x` switches on
`terrain`), the categorical step is evaluated per component. Component `i` of the mixture (its
pre-step belief, the one-tick lag of `terrain-field.md`) has cell mass `c_i`, so its child belief
is `q_i = table · c_i`, and the grow step of the mixture takes the `(K, M)` log weights
`log w_i + log q_i[j]`, the per-component form `propagate` already accepts, in place of the
marginal's `(M,)`. The factor's carried belief is the marginal `Σ_i w_i q_i`, which every other
reader sees. Today's path is the mean-field collapse of this joint. The joint is exact in the
sense the Gaussian-sum step is, costs `K` categorical steps instead of one, and leaves every
configuration without the closed pair unchanged. It holds whether or not the table is learned
within the plan, under either reduction schedule.

With D1, the working counts of a table read in the closed pair carry a leading component axis.
Component `i` reads `mean(n_i)`, absorbs `onehot(j) ⊗ c_i` into the `(i, j)` child of the grow,
and the reduction merges counts by the assignment that merges the moments: an output slot's counts
are the weight-average of its members' (the Runnalls merge gains an assignment it returns, so any
per-component statistic rides through a reduce). Under `mode_aligned` with one slot per mode the
merge is the whole group. A slot therefore remembers what it imagined: the slot conditioned on
`j` at a cell predicts `j` there again with the urn's probability, the slots' positions separate
by the modes' dynamics, and the Kolchinsky–Tracey bracket of the switching factor's information
gain opens where the map is open and stays closed where the counts are large. The novelty leaf of
such a table is the component average of the per-component leaf, each on its own counts.

A home read in the closed pair by one reader and, learned within the plan, by another reader is
not built in this round and is refused with a sentence.

## Where it plugs in

- `system/core.py`: the static `plan_learning`; validation of its keys against the learnable
  slots (`selectors._resolve_rates` holds the grammar); composition under `merge` and `subsystem`
  as `epistemic` composes.
- `system/faces/plan.py`: the planning carry gains the working tables when any key is set, a
  layer beside `_carry_prev` (`core.py:491`), with `beliefs_of` for the closed-loop and branch
  wrappers; `init_beliefs` seeds them from `params`; the hybrid `make_transition` and
  `make_step_terms` receive the spec.
- `objectives/hybrid.py`: `make_transition` resolves a learned table from the carry instead of
  `.point()` at build time; the categorical branch does the per-component read for the closed
  pair; the mixture branches take `(K, M)` weights and carry the counts through the reduce;
  `make_step_terms` hands the working tables and the per-component beliefs to the estimators
  through `StepContext` (`registry.py:148`).
- `objectives/registry.py`, `objectives/discrete.py`: `_param_transition` and
  `transition_novelty_leaf` read the step's working table when present (per component where D2
  applies), else the declaration; occupancy likewise.
- `distributions/mixture.py`: `_runnalls_merge` returns its assignment; `reduce` and
  `reduce_mode_aligned` accept per-component side statistics and return them merged.
- `objectives/compiled.py`, the fused tabular path: refused with a sentence when a key applies
  ("the fused path hoists its tables"); see A3 (built, below).
- `description/__init__.py`: the static under `configuration.inference`, its schema entry, the
  round trip.
- `docs/capabilities.md`: an "Unreleased (after v0.20.0)" entry.

## What to pin

1. The urn. A one-cell field with a sparse prior and a plan that stays in the cell: the
   per-component predictive of the next terrain after an imagined `j` is `(α_j + 1)/(α_0 + 1)`;
   with `plan_learning=()` it is the frozen mean.
2. The joint read. Components in different cells give `(K, M)` weights `w_i q_i[j]` and the
   marginal `Σ_i w_i q_i`; components in one cell reproduce today's step to float tolerance.
3. The merge of side statistics. Identical components keep their statistic; with one slot per
   mode the merged statistic is the group's weight average; a reduce with a side statistic leaves
   the moments bitwise as without.
4. Novelty along a plan that re-reads one row: the first step's leaf equals the frozen leaf, and
   the leaf declines step by step.
5. The mean-field form on a tabular parent set (a categorical factor with a learnable transition
   over a categorical parent, in a hybrid System): the working counts after `T` steps equal `T`
   `learn` steps on the same beliefs. *Revised 2026-11-17:* `T` absorptions of the joint, and
   the product of marginals `learn` absorbs differs where the table depends on its parent.
6. No key set: the full suite's pinned values stand; a closed pair under the joint read is new
   behaviour only where the pair exists (no shipped fixture has one).
7. The description round trip and the schema; the key under a `merge` composite and a
   `subsystem` view that lacks the reader.
8. Refusals: an unknown or non-learnable key; the fused path; D2's two-reader case.
9. The rover (on the switching example, `tests/test_survey_run.py`): with
   `plan_learning=("transition:terrain",)`, a plan into unmapped ground gives `transition:terrain`
   and `obs:fix` intervals of positive width and slot means that differ; on mapped ground (counts
   well above the prior) the widths are near zero.

## Decisions for Alex

- **A1, the name and the default.** `plan_learning`, default `()`. The frozen reading stays the
  default because pymdp parity and every pinned `G` depend on it; the rover declares its field.
  The alternative, default `True` (every learnable table carried within every plan), was weighed
  against the refusals that would then bite Systems that had not opted in: when this was written
  the fused all-tabular path refused `plan_learning` outright (A3 was deferred), which would have
  caught every tabular System with a learnable table; with A3 built the refusals left are the
  amortized path (it hoists its tables) and a learned likelihood under `present=`, and the cost
  of the carried counts (a tenth to a quarter more per gradient step) would be paid by default.
  Which is why not now.
  *Amended 2026-11-16, for the 0.21.0 decision (Alex: is there an API to unify with sophisticated
  inference?).* `plan_learning` is the parameter half of sophistication: the plan's beliefs are
  updated by what the plan would see. The state half is `branch_on`, by modality name, which
  enumerates outcomes and conditions the state beliefs on each; the parameter half needs no
  enumeration, since a conjugate update is linear in its sufficient statistic, so the expected
  statistic gives the exact predictive along an unbranched plan (the martingale: the working
  mean does not move, the concentration grows) and under branching each branch absorbs its own
  outcome. `epistemic` is the third, orthogonal declaration (what information is valued) and
  already shares the selector grammar with `plan_learning`. Two options:
  (i) **keep the two fields**, `branch_on` (modalities) and `plan_learning` (selector keys), and
  document them as the two halves; the names say which mechanism runs.
  (ii) **one field in the selector grammar**, e.g. `sophisticated=("obs:fix", "transition:terrain")`:
  an observation key means branch on it, a transition/likelihood/parameter key means carry its
  counts; `branch_on` stays as an alias for the observation keys. Cost: a rename and the
  selector resolution, the description key `configuration.inference.plan_learning`, docs,
  tests; about an hour of builder time and a gate run, before the tag (a rename after 0.21.0 is
  a migration note). Lean recorded by the session: (i), because one field suggests one
  mechanism and there are two; the grammar unification is real but the names are doing work.
  *Closed 2026-11-16 (Alex): `sophisticated=` accepted as an alternative syntax over the two
  canonical fields, guarded against mixing; built the same day.*
- **A2, the warrant.** Whether a learnable table read frozen within a plan should mark the plan's
  `G` as `approximate` in its Interval, as the engine marks its other approximations. Not in this
  build; recorded here. *Closed 2026-11-16 (Alex): no flag. Corrected the same day on Alex's
  question: the generative model is ONE, the table a Dirichlet latent shared by every read, and
  `plan_learning` does not change it; it changes how the parameter posterior is carried while
  planning, exactly as sophisticated inference changes how the state beliefs are carried (updated
  along the tree, or not) under one model. The frozen reading holds the parameter posterior fixed
  across the plan's imagined steps: that is expected free energy as conventionally defined
  (Friston et al. 2015; pymdp; sophisticated inference updates state beliefs along the tree, never
  the parameters), exact as that definition, and relative to the plan-conditioned predictive it is
  the mean-field approximation across the steps (independent reads of one row). `plan_learning`
  carries the posterior through the imagined reads, so prediction and novelty come from the one
  model's predictive. The `approximate` warrant marks inexact computation of the declared
  objective (bounds, linearisations); the declared objective under the default is the conventional
  one, and the library does not flag a plan that does not branch either. No flag.*
- **A3, the fused path.** Carrying counts there is the same design with a per-step mean and
  digamma in place of the hoisted tables. Deferred until a tabular customer wants it (the survey
  scheduler reads the field frozen, and its pooled readings are calibrated to that).

## Out of scope

A spatial prior over the field (`terrain-field.md`); the two-resolution field; working
posteriors of non-tabular learnables (an MNIW emission's is a different computation);
cross-factor continuous parents.

## As built (2026-11-16)

`tests/test_plan_learning.py` holds items 1 to 8 of "What to pin"; item 9 (the survey rover)
waits on the example.

- **Declaration** (`system/core.py`, `system/selectors.py`): `System.plan_learning`, a static
  beside `mixture_schedule`, `bool | tuple[str, ...]`, default `()`. `selectors._plan_learning`
  resolves it to one `(key, readers)` per working table: a reader of a parameter variable is keyed
  `parameter:<name>` whether it was named by its own key or the variable's, so the learned readers
  of one variable share one table. Literal keys only (a pattern is refused, naming the learnable
  tables); `validation._validate` resolves it, so every face refuses a bad key, as `epistemic`'s
  keys are refused. A home whose readers are all frozen, or a reader that is not a
  `ConditionalDirichlet`, is "not a learnable table".
- **The carry layering** (the first judgement call): a separate layer, `core._carry_learning`,
  inside `_carry_prev`, whose carry is `WorkingCarry(beliefs, counts, before)`: `counts` after the
  step's absorption, `before` the counts the step read. The hybrid closures take the counts
  explicitly (`transition(qs, action, working) -> (qs, working)`, `step_terms(qs, action, t,
  qs_prev, working=)`), so `BeliefPair`, `prepare_beliefs` and every consumer of the belief tuple
  are untouched; `_carry_prev`'s `qs_prev` is then the previous `WorkingCarry`, the closed loop's
  `beliefs_of` and the branch closure unwrap it, and a branch keeps each child's counts. A layer
  rather than a wrapper in the tuple because the counts are a belief about a table, not about a
  factor, and a home's table is read by several factors at once. The gradient, CEM, enumeration
  and sophisticated planners run through it unchanged (the tests drive enumeration, gradient and
  sophisticated).
- **Reads and absorption** (`objectives/hybrid.py`): `make_plan_learning` builds the static
  `PlanLearning` (tables, closed pairs); `make_transition` reads a learned table through the
  working mean (`_with_counts(decl, n).point()`), and each reader absorbs `decl.stats(child,
  parents)` (since 2026-11-17 the joint, `hybrid.expected_joint`) with the parents in table order (`_table_parents`: a cell parent as `cell_mass` of the
  pre-step belief, a control axis as the step's one-hot, a belief-valued step's probability
  vector, the size-1 axis of a table with no control); readers of one table sum their statistics
  and every reader of a step reads the counts before it. A learned likelihood absorbs
  `stats(q(o), parents)` from the predicted beliefs, `q(o)` through the working mean; it is
  built for a modality over categorical state factors and refused under `present=` (the
  transition does not know the step's availability) and under `branch_on` on that modality (a
  branch would absorb the outcome drawn). Item 5 pins the transition form against `learn`.
- **The step terms** (the second judgement call): `make_step_terms(..., learning=)` returns a
  closure that, at each step, rebuilds the ordinary closure over declarations holding the
  `before` counts (`_learned_step_terms`), the precedent `sensed` set; so novelty, ambiguity,
  occupancy and the likelihood leaves read the working posterior with no estimator changed. A
  closed pair's table is read per component instead: the per-component child beliefs and cell
  masses are recomputed in the step terms from `qs_prev` (the pre-step mixture, which the `(qs,
  qs_prev)` carry already holds whenever a leaf reads the transition's parents) and the `before`
  counts (`pair_read`), and reach the estimators as `StepContext.components`
  (`registry.ComponentTable`); `_param_transition` and the occupancy row take the weight-average
  of the per-component leaves there. Nothing new rides the carry for it.
  *Revised the same day, for cost:* the closure is now built once, over the declarations, and
  the step reads the working posterior at run time through the context: `_learned_step_terms`
  computes, per step, the closed pairs' `components` and the other learned readers' working
  declarations (`StepContext.working`, `(slot, index)` to the declaration holding the `before`
  counts) and hands both to the prebuilt closure as keywords. The readers of a learnable table at
  step time take the working declaration when present, the hoisted one otherwise
  (`registry._held`: `_param_transition`, `_param_likelihood`, the tabular state-information
  leaf, the occupancy row; and the tabular modalities' `q(o)`, ambiguity and utility reads in
  `_make_step_terms`). The values are bitwise unchanged. The rebuild itself turned out to cost
  little (the forward pass was already within a few percent of the frozen one): the extra cost
  of a gradient step of `G` on the survey rover was the backward pass through the novelty
  weights of the working counts, where JAX differentiates `digamma` by `polygamma(1, ·)`, a
  Hurwitz zeta series, at every element of the `(K, L, 3, cells)` counts at every step. `_wnorm`
  now reads `digamma` through a `custom_jvp` whose derivative is the recurrence-and-asymptotic
  trigamma (`dirichlet._trigamma`, relative error below 1e-11 in float64, a few 1e-7 in
  float32); values are unchanged and gradients agree to float tolerance. Measured on the rover
  with its camera (a loaded machine, so read the ratios): `value_and_grad` of `G` over a
  12-step plan 6.9–7.1 ms learned against 4.8 frozen before, 5.7–6.2 against 5.0 after (about
  4.5 ms against 3.9 on a quieter run). What remains is the forward and backward work on the
  counts themselves: the grown counts, the merge of the side statistic through the reduce, and
  the per-component reads.
- **The joint read** (D2, `closed_pairs`, `pair_read`): a categorical factor reading exactly one
  mixture-carried factor through a cell whose linear or nonlinear Gaussian transition switches on
  it. It is in `make_transition` for every caller, so `update_empirical_prior` predicts through it
  too (the filter's prior of a closed pair now carries the joint weights; the tabular factor's
  marginal is unchanged, `Σ_i w_i table·c_i = table·cell_mass`). The `(K, M)` log weights come
  from `_mode_log_weights` over the mode parents with the paired factor's belief batched per
  component; `_mixture_step` and `_nonlinear_mixture_step` take `log_weights=` and `aux=`.
  `tests/test_clock_parity.py` (the frozen survey rover, a closed pair) and
  `tests/test_terrain_field.py` stand unchanged.
- **The merge assignment** (the third judgement call): `MixtureGaussian.reduce(k, aux=None)` and
  `reduce_mode_aligned(k, M, aux=None)` return `(mixture, aux)` when `aux` (a pytree of arrays with
  the component axis) is given, the mixture alone otherwise. `_runnalls_merge` carries `aux` in its
  loop state and merges a pair's entries by the weight share `_merge_moments` uses (one half for
  a dead pair), then gathers the top-k; `aux` never enters a cost, and item 3 pins the moments
  bitwise equal with and without it. A closed pair's counts on the grown mixture are
  `n_i + onehot(l_j) ⊗ c_i ⊗ ...` at slot `i M + j` (`_pair_grown_counts`).
- **Composition and the description**: `merge` resolves each component's setting to its reader
  keys (`resolved_plan_learning`, so a binding that identifies two variables cannot strand a
  `parameter:` key), namespaces and unions them, `True` where every selecting component says
  `True`; `subsystem` keeps the keys whose reader (or variable) the view holds. The setting is not
  a leaf's `Subsystem.config` field. `describe` writes `configuration.inference.plan_learning`
  (`true` or a list) when set; `build` checks its form and the slot word of each key, the schema
  takes a boolean or a list of strings (`description-0.3.json` regenerated), and it costs
  nothing in the description length.
- **Refusals**: the fused all-tabular path and the amortized path refuse a key that applies
  ("hoists its tables", A3; the fused path's refusal is lifted by "A3 built" below);
  `make_plan_learning` refuses D2's two-reader case. A grid axis of one
  cell had no cell mass (`_axis_masses` sized its padding from an empty array); fixed, since the
  urn's one-cell field needs it.

## A3 built (2026-11-16)

The fused all-tabular path carries D1 (D2 does not arise: there is no mixture on it). Pinned in
`tests/test_plan_learning.py` (the "A3" section, eight tests). The fused path's refusal of a key
that applies is removed; the old refusal test is replaced by the A3 tests.

- **The carry** (`objectives/compiled.py`: `FusedLearning`, `working_tables`, `make_learning`;
  `system/faces/plan.py::_fused_learning`): the same `WorkingCarry`, inside `_carry_prev` as on
  the hybrid path, seeded from the posterior's concentrations. `make_plan_learning` resolves the
  tables as on the hybrid path, with every factor categorical, so its refusal of a learned
  likelihood over an action node holds here too. At each step `working_tables` rebuilds the
  `CompiledTables` with each learned table's entries (its mean, a likelihood's conditional
  entropy, its `novelty()` weights) from the counts, every other entry the hoisted one: the
  transition reads the counts before the step, and absorbs at rate one the expected joint
  (`hybrid.expected_joint`: a transition's working mean times its parents in table order
  through the hybrid path's `_table_parents`, the pre-step parent vectors, a control axis as the
  taken level's one-hot or a belief-valued step's probability vector; a likelihood's working
  mean times the predicted parents). The step terms take the per-step tables as a keyword (`tables_t`
  of `make_step_terms`'s closure, and `make_transition`'s), and the three fused estimators that
  read a table (`_param_likelihood_compiled`, `_param_transition_compiled`,
  `_occupancy_transition_compiled`) read `StepContext.tables` when present, so novelty, ambiguity,
  `q(o)` and the occupancy row read the working posterior; the novelty leaf is that of the counts
  before the absorption. Without a key nothing of this is built: the carry is the bare tuple, and
  `q(π)` and every leaf of the enumeration and sophisticated plans on the test model were checked
  bitwise against the code before the change.
- **The pooled absorption**: where `System.readings` pools `n` readings of a learned table under
  the step's option level, the step's statistic is absorbed `n` times (`n` read as
  `pooling.count_at` reads it, the expected count under a belief-valued step), as its novelty
  leaf is scaled by `n`: the step stands for `n` readings, so the novelty of the next read
  declines as `n` absorptions make it decline. Built on the fused path only; the hybrid path's
  absorption is not scaled (no customer pools a table there yet, and a closed pair's per-component
  counts would need the scale threaded through the grow). Recorded as an open edge.
- **The sophisticated planner, per branch** (`make_learning_branch`, reached from
  `plan.py::_branch_fn(fused=)`): the carry records per learned reader what a branch needs
  (`WorkingCarry.reads`: a transition's step joint, times the pooled count; a likelihood's
  count), an empty field on the hybrid path. A branch predicts each outcome through
  the working posterior the step read (the counts before its absorption), and its counts are the
  counts before the step plus, per reader: for a learned likelihood of a branched modality the
  exact conjugate update `onehot(o) ⊗ q(s_P | o)`, the joint posterior over the parents
  (`branching.make_branch_fn(..., evidence=True)` returns it per modality, aligned with the
  product of outcomes); for a learned transition the step's joint conditioned on the branch,
  reweighted along the child by `q(s'_f | o) / q(s'_f)` (exact when the branched outcome reads
  `s'_f` and the parents only through it); for a learned likelihood not branched on, its joint
  under the branch's beliefs. The step's own absorption is replaced, not added to. A tiny model
  pins `G` against a hand-rolled two-step Bayes-adaptive recursion (rtol 1e-5). The gap this
  section first recorded (the outcome-average of the branches' exact updates is the joint
  `A(o | s) q(s)`, while the unbranched step absorbed the product `q(o) ⊗ q(s)`) is *resolved by
  Alex on 2026-11-17*: every step absorbs the joint (D1 revised), so the step's statistic is now
  the outcome-average of its branches', which the tests pin for a likelihood and a transition.
- **The closed pair under the joint (2026-11-17, open)**: `_pair_grown_counts` absorbs
  `onehot(l_j) ⊗ c_i` into the grow's `(i, j)` child: conditioned on the drawn mode `j`, but with
  the cell parent at the component's prior cell mass `c_i`. The joint conditioned on `j` is
  `onehot(l_j) ⊗ (B(l_j | cell) c_i(cell) / q_i(l_j))`, which differs where a component spans
  cells whose rows disagree on `l_j` (it agrees for a one-cell component, the urn's case).
  Not changed in this revision; flagged for Alex.
- **Still refused**: the amortized path (its transitions are networks, so `True` resolves to no
  table and a key names no learnable table; the sentence in `_context_without_control_risk`
  stays as a guard); a likelihood learned within the plan under `present=`, on both paths; a
  learned likelihood under `branch_on` on the hybrid path (its branch closure keeps each child's
  counts and does not absorb the outcome; the fused construction would carry over, not built);
  D2's two-reader case.
- **Cost**: only the learned tables are recomputed per step, the mean in the transition and the
  mean, entropy and `novelty()` in the step terms (jit removes what no leaf reads); everything
  else stays hoisted. The `reads` record adds one parent-shaped array per learned transition
  reader to the carry, which the sophisticated planner's memory estimate counts with the beliefs.

