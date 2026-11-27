# hybrid-aif capabilities — v0.20.0 (2026-11-15)

One page, shipped-only. Details: `project-brief.md` (architecture),
`design/*` (decisions).

## Declare

- One `System` type declares agents *and* environments (`as_env` flips
  the boundary). Typed node menu: Categorical / Gaussian /
  MixtureGaussian / conditional tabular families / linear-Gaussian
  observations & transitions (incl. per-mode switching banks) /
  **neural conditionals** (`MLPConditional`, role-free, frozen or
  learnable) / **neural state-space parts** (`MLPCategorical` factored
  codes, `Deterministic` carries, `PredictorHead` heads, `LearnedVector`
  initials) — roles come from slot placement, not class names. R-AIF's
  `tiny` RSSM is a declared System (`reference.raif.world.tiny_rssm`).
- Four parameter slots, each with structure on `System` and values on
  `SystemParams`: the three generative slots, **`recognition`**
  (amortized inference per variable, A11) and **`scoring`** (the
  parameters an objective row owns — critic, ensemble, a preference
  model — keyed by row name). Term rows are mounted by `terms=(...)`;
  `epistemic=` remains the spelling for the infogain rows.
- Preferences beyond pymdp's C: `obs_targets` — over a tabular
  modality's levels *or* over a **linear-Gaussian modality's continuous
  observation** (Gaussian/mixture density, switching modalities
  included: "prefer observing my pose at the committed site" is one
  designation) — per-step `state_targets`, terminal targets (any
  Distribution as target, covariance-as-tolerance), habit priors
  `policy_prior` (E), hierarchical conditional initial priors (`D_deps`) and
  **conditional terminal priors (`H_deps`)** — intended-state priors per
  arXiv:2312.07547, the hierarchy's target-emission mechanism (v0.5).
- Continuous control nodes (a `Box` in `SystemParams.control`, read by a
  neural transition, a linear-Gaussian `G u` term or a nonlinear map) beside
  discrete ones, mixed spaces as structure; static shapes throughout;
  jit/vmap-safe by construction.
- **Nonlinear transitions** (`NonlinearGaussianTransition(f, innovations,
  propagation)`): `x' = f(x, controls) + ε` with `f` an equinox module the user
  writes (a discrete control arrives as its integer level, a continuous one as
  its value), each belief propagated at its own moments, to first order (the
  default) or by the unscented transform, so a plan that turns far from the
  current heading predicts the turn rather than a frozen linearisation's
  sideways move; Gaussian- or prior-mixture-carried, a control belief entering
  through its moments; `as_env` samples through `f`; described and rebuilt by the
  registered name of `f`'s class, `register_transition("<package>.<name>", cls)`, and its
  fields (`design/nonlinear-transitions.md`).
  **Process noise that depends on the state and the controls**
  (`NonlinearGaussianTransition(..., noise_scale="per_coordinate")`): `f` returns `(mean,
  scale)` and the innovations' Cholesky factor becomes `diag(scale) chol`, the scale read at
  the belief's mean to first order and averaged over the sigma points as `E[s sᵀ]` unscented
  (the expected conditional covariance), at the sampled state in `as_env`; the gradient
  planner differentiates through it, provided the scale is strictly positive, finite and
  differentiable wherever it is evaluated (a `sqrt(1 - rho(d)**2)` with `rho(0) = 1` has a NaN
  derivative at rest: write `sqrt(1 - rho**2 + eps)`); recorded in the document as
  `options.noise_scale`. The neural
  transition steps through the same propagation. What this covers: a transition's noise as a
  function of its own state factor's previous value and its control nodes, per coordinate
  (each coordinate's standard deviation scaled; the correlations are the declared
  innovations'). What it does not: a noise whose correlations change with the state,
  noise read from another state factor (cross-factor continuous parents are not built), or a
  scale on a linear-Gaussian transition (declare an affine `f` instead). An observation's noise
  varies over the observed state by its own mechanism, a `NoiseMap` (below).
- **A Gaussian emission over categorical parents alone** (`gaussian_emission(loc, R_chol)`):
  the linear-Gaussian family with an empty regressor block, a Gaussian mixture whose
  component label is the parent; it sends the parents the exact categorical message of its
  Student-t predictive and learns by the responsibility-weighted conjugate update.
  **A growable factor**: a memoryless categorical factor whose weights (its transition
  column) carry a growth propensity `growth`; its cap is its size, a level with zero weight
  is dead, and `learn` wakes a level for a datum no live level explains better than the prior
  does. **A recurrent datum** (an own-belief read): a modality takes as its datum a state
  factor's previous belief or the previous action, so a factor can emit what the model just
  believed (`design/structure-learning.md` S8). It is declared as an emission of the System that
  one of its own modalities observes: `System(emissions={"x_mean": Emission(("x",),
  realise="mode")}, observes={"prev_x": "x_mean"})`. The datum is the emission realised from the
  posterior of the previous round, by what the emission is: for a categorical factor under the
  identity kernel its probability vector under `stats` (a soft observation), for a Gaussian
  factor its location under `mode`, and for an action node the level taken (a modality that
  observes the action node). **A belief read** (`Emission(parents, realise, kernel)` with
  `observes`, and `Link(source, modality, realise=..., kernel=...)` in a merge as the sugar for the
  pair, `design/nesting-and-coupling.md` N3) generalises it three ways: the datum is the source's
  belief pushed through a static kernel, the identity or a cell indicator over a `Grid` (so a
  categorical modality over the grid's cells reads a continuous position factor); the realisation
  is declared, `sample` (a draw; `replay` takes its key from the tick's, a hand call supplies the
  draw itself), `mode` (the argmax, or the mean of a continuous belief) or `stats` (a categorical
  belief's probability vector, a soft observation; not built for a Gaussian source); and the
  source may be a state factor of another component of a merge, whose emission the source's step
  realises under `run` and the modality reads as it stands at the reader's turn, absent before
  the source first steps. `merge` carries a component's own entries under its prefix,
  `Subsystem.reads` records the sibling ones, and a component bound only by such reads and links
  from an action node into a control node is nested (`nested`). The faces take the datum as any
  observation, keyed by the emission's or action node's name (`System.observes`, section
  "Observing a variable" of `composing-systems.md`); `replay` realises and records the emissions
  (`Replay.emitted`) and feeds each tick's back as the next tick's datum. The description
  document records the pair: an `observes` entry on the modality's likelihood record and a
  record of the static kernel on the emission.

- **A categorical field over a continuous state** (`design/terrain-field.md`): a tabular
  factor may name a Gaussian or mixture-carried factor as a parent when the System declares a
  grid for it, `System(cells={"x": Grid(shape, low, high, axes)})`. The parent's belief reaches
  the table as its mass per cell (`cell_mass`: per component, the product of the marginal
  normal masses between the cell edges, the edge cells absorbing the tails, weighted by the
  mixture weights), flattened row-major as one parent axis of `prod(shape)` entries. The
  transition prior, the conjugate count update and the transition-novelty term all read that
  vector, so a `ConditionalDirichlet` over cells is a terrain map the agent learns by standing
  on it and plans to learn by the novelty term; `merge` and `subsystem` carry the grid; the
  description document records it beside the variable's `dim`; `description_length` prices the
  field as cells times one categorical. The read is one tick behind the dynamics (a categorical
  factor steps before the continuous factor it parents), and a sharp belief deep inside one cell
  gives the novelty term a flat landscape, so its drive needs `policy_len >= 2` and warm starts.
  Not built: the upward message from the field to the position, a spatial prior, and a smooth
  basis (MNLR). The cell mass is also the kernel at the source when another component of a
  merge reads the position as a modality (the cell indicator, `design/nesting-and-coupling.md` N3).

- **Parameters as named variables** (`design/shared-parameters.md`): a tabular parameter (a
  conditional table, or a Dirichlet over one) may be declared once under a name in
  `SystemParams.parameters` and read by several factors or modalities, each holding
  `Parameter("<name>")` in its `transition` or `likelihood` entry. `learn` sums the readers'
  count statistics and updates the table once, under the rate key `parameter:<name>`
  (a rate keyed by a reader raises and names it); novelty on each reader is unchanged; the
  description length prices the variable once; the description document records it under a
  top-level `parameters` entry with each reader pointing at it; `merge` identifies two
  components' variables with `Identify`. A neural (gradient) declaration is refused with NotImplementedError; see the design note for the second build. An `Identify` identifies a variable outright: the two components of one particle share it, with no blanket between them on it, and `write_back` is bookkeeping of that one home between views (`design/nesting-and-coupling.md` N2, N6). A separate mind joined by `couple` never takes another's table this way; it receives a report of counts through an action node.

- **A slow factor's reading pooled over an option level** (`design/step-durations.md` T10):
  `System(readings=(("<factor>", "<control node>"),))` with `SystemParams.readings[<factor>]`, a
  positive count per level of that node, multiplies the novelty term of the factor's learnable
  table (a `ConditionalDirichlet`, directly or through a parameter variable) by the count of the
  level selected at each plan step, so a slow step scores the information of the fast readings it
  pools; no target is scaled and nothing else reads the count. `merge` namespaces it, a
  description document records it on the factor's record, and the description length prices
  nothing for it.

## Infer

- Discrete: mean-field FPI at pymdp parity (function, term, policy
  level). Continuous: exact linear-Gaussian conditioning. Hybrid:
  switching-observation mixture filtering — exact grow, Runnalls
  reduction at a static budget — with per-cell/mode evidence surfaced
  (`infer_states(detail=True)`).
- Filtering under external control: `infer_states` runs every tick
  whether or not the agent acts (operator-coexistence pattern,
  fielded).
- Amortized: a variable with a `recognition` entry is inferred by its
  module. On the neural path the module's output is the posterior (an
  RSSM posterior over `[deter, embed]`); on the analytic path it is a
  measurement combined with the analytic evidence; a factor without an
  entry is inferred by the structured rules. A recognition parent may be an **externally
  supplied vector** (`"external:<name>"`), which is how one model
  conditions on another's embedding as data rather than by sharing a
  module. An externally supplied vector is a sensation of this System; whatever another mind
  sends arrives as a value realised from an action node of the sender (`design/nesting-and-coupling.md` N1, N8).
  `predict_beliefs` is the open-loop rollout.

## Plan

- **Planning that expects to observe** (`design/planning-with-observations.md`):
  `System(planning_belief="posterior")` carries a second belief chain for every
  Gaussian-carried factor, predicted from the previous step's belief after
  conditioning on the observations expected there, read only by the
  information-gain leaves, so a linear-Gaussian channel's leaf is the gain of its
  observation given the plan's earlier ones (Kalman parity pinned; targets read
  the predictive chain as before, bitwise); an observation noise that varies over
  the state (`GridNoiseMap`, `R_chol` at the nodes of a grid read by
  interpolation, differentiable for the gradient planner, evaluated at the
  belief's mean everywhere; describable) makes the gain plan-dependent, so a plan
  detours to where the sensor is precise; `present=` on `infer_policies` and
  `planner_context` says which modalities the plan expects at each step. Under the default
  `"prior"` every step reads the prior belief, so a plan that dwells at a cue is credited
  with its information at each step, the per-step convention pymdp shares; the posterior
  chain removes that for Gaussian-carried factors, and for a categorical channel, whose
  posterior depends on which outcome arrives, the exact remedy is branching on it
  (`branch_on`).
- **Closed-loop scoring of Gaussian targets** (`design/closed-loop-planning.md`):
  `System(feedback="linear_quadratic")`, under `planning_belief="posterior"`, scores a plan as
  the policy that follows it and corrects deviations by the linear-quadratic regulator's
  feedback on the estimate: each Gaussian state, terminal and linear-Gaussian observation
  target on a Gaussian-carried factor is scored by its cross-entropy on the closed-loop state
  distribution (the epistemic chain's posterior plus the estimate spread the corrections
  leave), keys unchanged, and a Gaussian target on a continuous control node charges the
  corrections. An observation the plan expects is then worth what it lets the corrections
  remove beside its information gain; something no control reaches is worth its information
  gain alone. Every planner reads it (a two-part pass: the planning carry records each step's
  linearisation and covariances, the terminal terms run the Riccati pass). Pinned against Monte
  Carlo of the corrected plan executed with a Kalman filter (exact for linear-Gaussian
  factors), against the offline harness's LQG numbers on the lane model, and against Monte
  Carlo of the nonlinear unicycle (0.2 % here, its linearisation error). Nonlinear transitions
  use their propagation's linearisation; refused on mixture-carried factors, neural
  transitions and the amortized path.
- Enumeration planner at pymdp speed parity (`metrics=False` ~45µs/tick on
  the H=2 T-maze against pymdp's 35, dispatch-bound; the prefix-sharing tree
  wins from H=3–4 on; >15k full ticks/s laptop CPU), **CEM over
  continuous action sequences**, the **gradient planner** (Adam over the continuous plan, its
  normalisation floored at the plan dtype's precision relative to the gradient's scale, the best
  plan scored returned), the **`mixed` planner** (a control space of discrete and continuous
  nodes: it enumerates the discrete policies and descends the continuous controls of each with
  the gradient planner's descent, the posterior over the discrete policies taken over their
  optimised `G`), the **`factorised` planner** (several discrete control nodes planned by
  mean-field sweeps: one posterior per node over its own sequences, each scored against the
  others' per-step marginals, at the sum rather than the product of the nodes' sequence counts;
  `q_pi` over the joint first step, the per-node posteriors in `FactorisedPolicies`; exact for
  nodes that share no factor, a measured small gap where a parameter learned within the plan
  couples them; `make_factorised_planner(sweeps=3)`, design `docs/design/factorised-planning.md`),
  the **`sophisticated` planner** —
  Friston et al. 2021's recursion over what the agent will observe: each
  first action scored by its step terms plus the softmax-weighted value of
  the next decision after every outcome of the System's `branch_on`
  modalities (exact over categorical modalities with categorical parents;
  contributed 2026-09-29; below a split depth it evaluates the deepest level's subtrees a batch
  at a time, set by hand or from an opt-in memory budget, with the same result to rounding, and it reads a per-step
  `legal=` mask at every depth: `make_sophisticated_planner(memory_budget=, split_depth=,
  batch_size=)`) — and the **`habit` planner** — act from
  the E designation directly (a policy module or the classical vector),
  the search↔habit dial's reflex end; `score_policies` scores any plan
  as search would (the distillation diagnostic). Planners injected by
  registry name; stochastic planners declare their key. A policy
  module's draw comes back **raw**: a Gaussian head over a bounded box
  puts real mass outside it, and the box belongs to the action's
  consumers — `controls.soft_clip` (a stop-gradiented rescale, R-AIF's
  `action_clip`) wherever a *transition* reads the control node, a clip
  wherever a world consumes the action, and neither where it is recorded.
- EFE returned as a structured **term tree**: one `infogain` term per
  variable (states, parameters, observations — the v0.4 per-variable
  epistemic API), utility/risk/ambiguity/terminal leaves, exact-or-
  bounded Interval values. One `epistemic` selector with wildcards
  (`epistemic=("*",)` = states only; `("transition:*",)` = B-novelty
  class), so ablations are one-line and never lie. A channel's state leaf
  is its information about the *selected* variables among its parents,
  with ambiguity split to match (`design/per-variable-infogain.md` V9).
- **Each decomposition ablates its own epistemic leaf** (`design/per-variable-infogain.md`
  V11): the selector mounts `infogain` under `utility_infogain` and `ambiguity` under
  `risk_ambiguity`, and never touches risk. A channel without a preference carries an
  implicit flat one, so under `risk_ambiguity` its risk is `−H[q(o)]` (the normaliser dropped,
  as pymdp's log-preferences drop it). Where a channel's epistemic leaf is mounted,
  `risk + ambiguity = −(utility + infogain)` per channel at both interval ends, so with
  everything selected the two forms give one G, designated channels or not; where it is
  ablated they differ by exactly that channel's `H[q(o)]` (on a Gaussian factor shared by
  several channels, by the ablated readings' entropy given the selected ones'). (Under
  `planning_belief="posterior"` information gain reads the epistemic chain and risk and
  ambiguity the predictive belief, so the forms differ there by design.) A Gaussian emission
  over categorical parents follows the selector like every channel: its `infogain` is `I(y; z)`
  about its parents (a Kolchinsky–Tracey pair on the predictive mixture), its ambiguity is
  mounted where that is, and unselected it carries its flat-preference risk alone.
- **Information gain on a Gaussian-carried factor is per channel**: one `infogain` leaf per
  linear-Gaussian channel, `I(y_m; x)`, keyed by modality as on every other path (read on the
  epistemic chain under `planning_belief="posterior"`), and two factor-keyed terms for what a
  per-channel sum miscounts when several channels read the factor. Under `utility_infogain`,
  **`observation_interaction`** is the interaction information of the selected channels'
  readings about the selected state, `TC(Y) − TC(Y | x_S)`, so `I(Y; x_S) = Σ_m I(y_m; x_S) −
  observation_interaction`: on the whole factor the readings' total correlation (redundancy),
  and on a V10 sub-block possibly negative (synergy: readings that tell about `x_S` only
  together). Under `risk_ambiguity`, **`observation_correlation`** is the total correlation
  `TC(Y)` of every present channel's readings, the part of the joint risk `KL(q(Y) ‖ ∏ C) = Σ_m
  KL(q(y_m) ‖ C_m) + TC(Y)` a per-channel sum leaves out, mounted whatever the selection, and
  on a sub-block `observation_interaction` is the joint ambiguity's `−TC(Y | x_S)`. Both are
  costs in G, and the two forms give one G under full selection.
- **State targets inside the sophisticated planner** (with `branch_on`) are scored by their
  cross-entropy `−E_q log C(s)`, at each step and at the horizon (the keys stay
  `risk[<factor>]` and `terminal_risk[<factor>]`), so knowledge gained at a branch is not
  charged later; open-loop planners keep the KL (`design/per-variable-infogain.md` V11
  decisions 6 and 8).
- **`uncertainty`**, an opt-in row: `terms=("uncertainty:<factor>",)` (or
  `"uncertainty:<factor>:<dim>"` on a sub-block) charges a continuous factor's predicted
  entropy `H[q(x)]` as a cost, exact on a Gaussian and a Kolchinsky–Tracey pair on a mixture —
  the drive toward a sharper belief, the price of injected noise. `infogain` holds information
  gain only: a continuous factor no channel observes carries no `infogain` leaf, and naming it
  literally in `epistemic` raises toward this row.
- Beyond orthodox EFE, as named rows (`design/objective-terms.md`
  places them on the term lattice): **`occupancy`** (expected transition
  entropy — the maximum-occupancy principle's β term, R-AIF's `wment`;
  a closed-form `derived=True` leaf on a declared tabular transition,
  a neural estimator on a recurrent state-space one, one row name and
  one key either way), `predicted_reward`, `cosine_sim_to_pref_rollout`
  (against a declared preference model) and `g_tail` (a learned value
  closes the horizon). A row declares which **variable families** it
  can read, so an objective is refused per row and per family rather
  than per regime.
- **One estimator, two signs**: a model ensemble's disagreement is
  *curiosity* under **`infogain`** (sign −1 — seek it out) and
  *pessimism* under **`model_uncertainty`** (sign +1 — a cost, which is
  R-AIF's own reading: paper Eq. 6's `−IG`, Eq. 7b's "homeostasis").
  Same estimator function, same trainer, two registered terms — so the
  sign stays a property of the *model* rather than of a call, choosing
  between them is an ablation the `terms=` selector already expresses,
  and the lattice shows the pair (its third same-functional-opposite-sign
  one).
- Every Interval carries **`derived`** — whether a derivation guarantees
  its enclosure — and a tree with any net-estimated leaf folds to
  `derived=False`: the honesty flag, never a warning.
- **Compiling an agent loop** (`design/recompilation.md`): call the faces through
  `system.compiled()`, which returns the System's `infer_states`, `infer_policies`, `act`,
  `update_empirical_prior` and `learn` as compiled functions of the same signatures. The System
  and the planner are static, and the inputs are canonicalised first: parameters, beliefs and
  priors keep their dtypes but become strong-typed JAX arrays; observations, `previous` and
  `externals` become arrays (an integer in the default integer dtype, a float at least float32);
  Python numbers in `lr`, `decay` and `alpha` become arrays (a Python `lr` of `0.0` and `decay`
  of `1.0` stay static); `legal` masks become bool arrays and a mask the eager faces refuse is
  refused before the boundary with the same sentence; `priors=None` becomes `initial_priors`
  completed with the control-node and action-node entries the push returns; typed PRNG keys
  become raw key data. A wrong-shaped observation is refused with a sentence. A second trace of
  a face emits a `hybrid_aif.debug.RetraceWarning` naming what changed (`hybrid_aif.debug.watch`
  does the same for any `eqx.filter_jit` function). Called eagerly, a face re-traces and
  dispatches operation by operation on every call (a mixed-planner call that takes 40 ms
  compiled takes about 4 s), and an eager face called more than ten times on one System warns
  once. The hand-wrapped way: the faces are eager and pure, so the tick (`infer_states`, `act`,
  `update_empirical_prior`, and `learn` where a slot learns) is
  wrapped once per model shape in `eqx.filter_jit` or `jax.jit`, with the System passed as an
  argument and only values changing between ticks, and each face then traces once. Every field
  of `System` is static (`planner`, `policy_len`, `gamma`, ...), so a different value is a
  different compile; the traced way to vary the policy precision is
  `SystemParams.precision`, a scalar array that `infer_policies` and `act` read in place of
  `System.gamma` when present, so annealing it recompiles nothing. Schedules (`lr`, `decay`,
  `alpha`) are passed as arrays: under `eqx.filter_jit` a Python number is static and compiles
  once per value (`replay` turns a nonzero Python `lr` into an array; a Python `0.0` stays
  static on purpose). Start the loop from `initial_priors`, which has the structure
  `update_empirical_prior` returns (parameter and prior leaves hold strong dtypes), not from
  `priors=None`, whose first tick has another structure; a mixture prior declared with fewer
  than `mixture_k` components is the one difference, and
  `hybrid_aif.objectives.hybrid.pad_mixture(q, mixture_k)` removes it (padding changes the first
  filtering step under `mode_aligned`, so the library leaves the declared prior as it is). Pass
  `legal=` on every tick, all-True when nothing is restricted, with the System declaring the maximum number of levels, so a changing set of options is a changing
  value and not a changing shape. An observation is an array of the declared shape and dtype
  (a Python list, or a linear-Gaussian reading of another shape, is refused with a sentence).
  To find what retraced, run with `jax.config.update("jax_explain_cache_misses", True)`, which
  names the changed argument under `jax.jit`; under `eqx.filter_jit` its paths are opaque, so
  count traces with a counter inside the wrapped function.

## Learn

- Conjugate: Dirichlet likelihood/transition/initial-prior learning;
  LG noise (Inverse-Gamma/Wishart) and joint head+noise (MNIW)
  posteriors; **anchored forgetting** (decay toward an anchor the
  declaration chooses: its prior, a uniform table, or a given array; a
  declaration without one does not decay). Per-slot and per-block learning rates with the same
  wildcard grammar; rates may be traced (in-scan curricula).
- Evidence: `System.log_evidence(prior, posterior)` is each slot's closed-form
  log marginal likelihood of the statistics it absorbed (Dirichlet,
  inverse-Gamma, inverse-Wishart, MNIW) with the mass absorbed beside it, a
  slot with no closed form or no mass reported unscored rather than zero.
- The data link along a replay: `Replay.free_energy` is each tick's variational free
  energy (`InferenceDetail.free_energy`, mean-field KL less expected log-likelihoods, less the
  LG evidence) and `Replay.log_evidence` its negation, exact on one categorical factor or
  an all-LG model and a lower bound otherwise; summed over a trace it is the prequential
  description length under the model as it stood at each tick, learning off or on.
  `description_length.resolution_from_evidence` turns `System.log_evidence`'s absorbed
  counts into the per-address resolution the measure takes; `describe_parameters(document)`
  is the document's parameter envelope, one record per array with dtype, shape, domain and
  source (`design/description-length.md` D6, D7).
- **Structure**: `learn` grows a growable factor per datum (AXIOM's expansion rule in
  its maximum-a-posteriori form, the woken level seeded from the datum alone) and, every
  `reduction_period`-th call, merges pairs of its levels whose pooled statistics have no
  lower closed-form evidence than the separate ones (Bayesian model reduction, candidates by
  mutual expected log-likelihood); `System.grow` is the batch form over a window,
  `System.reduce` the explicit reduction; `log_evidence` scores a Dirichlet with dead levels
  over its live ones. `examples/gameworld_growth.py` runs a Gameworld-style task with object
  state given, growing from one level, reduction on and off.
- Gradient: neural slots learn by optax through the same `learn`
  face; optimizer state is an explicit, guarded carry
  (`init_learn_state`). Parameter *novelty* (expected info gain about
  parameters) is a first-class planning term for tabular families, and for a
  matrix-normal inverse-Wishart emission over a continuous state: the information a
  reading carries about its head and noise with the state marginalised, so a learned offset
  is sought where the state is pinned; exact at a known state, an estimate elsewhere,
  including at a prior's low degrees of freedom (`design/parameter-learning.md` D10).
- Carry: `System.carry(fresh, learned)` keeps what `learn` put into the params across a
  rebuild of them, every posterior block it updates (or the ones named) taken from the learned
  params and everything else from the fresh ones; a block whose family, shape or dtype changed
  is refused, and so is a selection that splits a growable factor's blocks.
  A plain copy, exact when the rebuild does not re-express what was learned
  (`design/parameter-learning.md` D9).
- Field-proven: the downstream coverage agent learns per-cell sensor
  noise online, seeded from and replacing its heuristic estimator
  (written up in that repository).
- **Trainers, for losses no per-tick statistic can express**
  (`hybrid_aif.train`): pure functions over
  `(System, params, data, LearnState)` — no Trainer object — each stepping
  only its own addresses and threading the rest of the carry through.
  `fit_world` (a recurrent generative model's sequence loss: KL balancing,
  free bits, symlog heads), `imagine` (a policy module + a value head on
  **sampled** model rollouts: λ-return over the term tree the planners
  score, percentile-EMA advantage normalization, a 255-bin critic with a
  slow-target regularizer and an EMA slow copy, policy/state-prior entropy
  bonuses, a mixed self-imitation term) and `fit_ensemble` (Gaussian NLL
  for the ensemble of predictive heads the two disagreement rows score).
  Each is generic over the *declaration*: what a trainer knows is which
  slots and which registry rows it steps.
- **Reference-tier trainers** (`hybrid_aif.reference`), by the tier rule —
  engine when a loss is stated over declarations, reference when it is
  pinned to an external target: `reference.raif.crspp` (a preference model's
  ρ-gated KL + contrastive cosine term, which A5's scope fence places
  outside the generic seam) and `reference.raif.revise` (a self-revision
  schedule, ported line-for-line from its source). One-way imports, with
  a test walking every engine module to enforce it.
- **Replay, one module and two regimes** (`train/buffers.py`): one
  `Episode` format, `ReplayBuffer` (host-side, dynamic schema, ragged
  episodes, `Dataset`+`Writer`, pluggable Python samplers, `dump()` to
  any registered format) and two device-side buffers with static shapes
  and pytree storage, so a training step can sample *inside*
  jit/`lax.scan` — `StaticBuffer` (episode-aligned sequences, each
  padded or truncated to `seq_len`) and `StreamBuffer` (a flat step
  stream, `capacity` in steps, windows that cross episode boundaries and
  carry `is_first` for `RSSMCore.observe`'s carry reset). Bridged by
  `as_episode`/`episode_columns`. The first fence is host/dynamic vs
  device/static, not storage medium; use `ReplayBuffer` unless you need
  in-jit sampling. The second is a *sequence contract*: whole episodes,
  or windows of a stream. Replay is agent-coupled, so it lives beside the
  trainers — `hybrid_aif.data` is formats, datasets and writers, and the
  import direction is one-way.
- One `LearnState` for the whole agent: slot addresses for the model,
  `scoring:<row>` for each objective row a trainer owns, `policy_prior`
  for the actor, plus a `moments` field for the running statistics that
  are not optimizer state.

## Compose (`hybrid_aif.composition`, v0.5; objective-row composition fenced — see the debt ledger)

- `merge({name: System}, params, bindings) -> (System, SystemParams)`: composition
  by namespaced graph-merge, identifying latents across components —
  control-node identification or relay, a modality's parents,
  **target emission** (`H_deps` across the boundary), `Clamp`
  termination. The one-particle composite is **bitwise identical** to the
  hand-flattened declaration (the composite's arrays ARE the
  components'). Every face inherits: a merge adds namespacing, never
  inference machinery. A merge makes one particle, one set of internal states with one blanket
  to the world. A component is *nested* when every binding across its boundary is a
  belief read or a `Link` from an action node into a control node: its beliefs are its own and the
  rest of the particle is its external world. A binding that identifies a latent outright
  (`Identify`, the relay, a structural `Link` into a modality's parents) makes the components share that
  variable, with messages both ways (`design/nesting-and-coupling.md` N2).
- Bindings name nodes, native or the sugar's; an action-node driver is read at the same step (the two control nodes become one), a state driver a step later. A `Link` from an action node into a control node makes one node in the composite, the driver's decision, which the driven component's factors read as a supplied level (a control node supplied by another participant, `design/nesting-and-coupling.md` N7); a decision is not a belief, so no belief of the driven component is identified with the driver's, and the binding is blanket-shaped at the driven component's boundary. The state-driver form, the relay, binds a state's belief into a factor's parent set a step later and is an identification. Actions speak action nodes (`a_<f>`, one per control node); a kernel `P(a | u)` in `params.actuator` is drawn by the environment.
- `subsystem(name)`: a bona-fide per-mind System (private beliefs,
  boundary neighbours as ordinary latent factors — no partner
  ontology; partners are inferred and learned like any latent). On a nested component the view
  is that component's own model: the rest of the particle is its external world, and it reads
  that world through belief reads and links from action nodes only (`design/nesting-and-coupling.md` N2).
- `driver.run()` with three tick orders, each one tick of time per round
  and every factor stepping once: simultaneous (a Jacobi sweep), sequential,
  and **async** (Gauss-Seidel sweeps in declaration or random order, each
  component's factors stepping at its turn — the statistically faithful mode
  for equilibrium/Ising-like composites; `nesting-systems.md`, tick
  semantics and its 2026-10-14 amendment). A slow tier is a node whose view declares
  `params.durations`: `run` holds it for its option, `clocks=` converting a duration into rounds,
  and there is no second clock (`design/nesting-and-coupling.md` N4). `couple` joins components
  into one closed System that `run` steps, and what crosses an identification is action or
  sensation (`design/nesting-and-coupling.md` N1, N2); a merge is a component of a coupling like
  any other, each of its leaves stepped by name.
- Dynamic action sets: the hard constraint is **`legal=`** (v0.8, below);
  `policy_prior = −inf` masking remains valid (leaf-swap, no re-jit — the
  mission executor's task-selection pattern) and E is the soft prior alone.

## Describe, replay, constrain (v0.8 — what a *composing program* needs; `composing-systems.md` is the reference)

- **Description length**: `hybrid_aif.description.mdl.description_length(document, n)`
  prices a document in
  nats as a chain of codes, a `structure` part (a universal-integer graph code over
  factors, modalities, control nodes, emissions, families, sizes and parent sets, plus each
  family's parametric complexity at resolution `n`, closed form for a table) and a
  `parameters` part (the declared prior's KL from the family's base, Jeffreys for a
  table and a reference prior at an overridable `scale` for a Gaussian family, closed
  form for the Dirichlet, inverse-Gamma, inverse-Wishart and matrix-normal
  inverse-Wishart priors; the posterior's KL when the prior is passed beside the
  params), per address with an `exact` flag, a link the model gives no closed form
  for unscored rather than zero; names, order, placement and spelling cost nothing
  (`design/description-length.md`). A dead level (a growable factor's unused room) is
  priced by the cap's count alone, in its tables and its Gaussian emission's heads.
  `description_length(model, replay=...)` is the description length of the model and a
  replayed trace together, the model part plus the data part (the replay's summed free
  energy), each address's parameters paid for once: by its family link, at the data it is
  judged against along the trace, where the replay did not learn it, by the free energy
  where it did (D6).
- **The model as a document**: `describe(system, params) -> (document, arrays)`
  and `build(document, arrays=, key=) -> (System, SystemParams)`, bitwise round
  trip on every declared model in the repo including the reference agent's nested
  preference model and a merged composite. Description format 0.3, specified in
  `description-format.md`: every variable declared once with its domain and size;
  factor records keyed by child and label (role labels such as `initial`,
  `transition`, `likelihood`, `actuator`, `prior`, `target`, `recognition`, or a
  generic label with every parent's step bracketed, `x[t-1]`); parameters as
  variables, each fixed by a clamp or learned under a `prior` record (the
  matrix-normal-inverse-Wishart pair as two priors); clamps inline below 256
  elements, as refs into a linked `.npz` above (`save`/`load`), or as an `init`
  recipe a key draws; `relations` for the controlled, observed and external
  variables; the objective as a list of registered terms; the planning, inference
  and learning settings under `configuration`; a composite written flat under its
  components' name prefixes. A consumer's own data rides under a reserved
  `extensions` block and every other unknown key is refused. `schema()` returns
  the checked-in JSON Schema. `build` validates the document **before**
  constructing anything (kinds derived from use, parent steps, the families'
  parent rules, the tabular shape rules, domains, the policy-prior length, dead
  observation levels), each fault one sentence naming the address and what to
  write instead, then runs the library's own validator eagerly. A document in the
  earlier `structure`/`declarations` layout (description 0.2) is refused with a sentence
  naming 0.18.0, the last release that reads it.
- **Recorded-trace replay**: `replay(system, params, observations, actions)`
  scans the canonical loop over a record — the recorded action drives the
  carry, the model's own `act` runs beside it as a deterministic
  counterfactual — returning per-tick beliefs, priors, `q_pi`, term trees, the
  chosen action, the exact log-probability of the recorded action (workstream
  K's objective), log-evidence, `exercised` and `unsupported`. Per-tick
  **partial evidence** (`present=`, keyed by modality; a dropped frame is a
  predict-only tick; the same `present=` on `infer_states`), padding and
  batches owned by the library
  (`valid`, `stack_traces`, `vmap`), `replay_document` from a description, and
  `learn=True` for what the model would have learned (a gradient-slot model
  carries its `LearnState` through the scan). All three paths: on the
  amortized path an absent tick is predict-only, `externals=` slice per tick,
  and the recorded action's log-probability is the actor's own density under
  a policy module. A function, not a face: replay is something done *with* a
  model.
- **Hard constraints on the choice, `legal=`**: one keyword on
  `infer_policies`, `act` and `replay` whose general object is a mask over the
  policy set; a first-step or per-step level mask keyed by control node (a
  node left out unrestricted) is its factored special case and expands to it
  (never the reverse). Composed once, after whichever
  planner returned, over the policies it offered — injected planners are bound
  identically — after `log E` and never inside it; G is never re-scored.
  Discrete controls only.
- **Evidence no state explains**: the tabular FPI is tempered — `−inf` is a
  legal table entry, nothing NaNs — and `detail.unsupported` flags, per
  factor, a tick on which no level was compatible with every present modality
  and the prior under the beliefs held about its co-parents (the justified
  statement; it does not name the source). The posterior is the arithmetic's
  own answer, flagged, never replaced. `build` refuses a dead observation
  level and a state that emits nothing; `check_joint_support` is the
  exhaustive, budgeted diagnostic for the disjunction case factored
  likelihoods cannot express.
- **Memoryless factors**: `B_deps=()` with a `(levels, num_controls)` table
  ticks directly; no self-parent workaround.

## Two rollouts, and a tier that says what is whose (v0.9)

- **Expected and sampled rollouts on every face**: `update_empirical_prior`,
  `predict_beliefs` and `score_policies` take an optional `key`. Keyless is the
  expected belief pushed through the transition, bitwise as before on every
  path; keyed is a one-particle carry — a one-hot categorical, a point-mass
  Gaussian, one component's point of a mixture, a straight-through one-hot for
  an amortized code (`inference/sample.py`) — a legal belief of the same
  family for the next tick. A scoring row that rolls a model of its own (the
  CRSPP's) samples when given a key; a planner's `plan_key` never implies it.
  `imagine(sample_scoring=)` and the harness's `--sample-scoring` select the arm;
  every default is the expected rollout. Design: `design/amortization.md`,
  FLAG (1)'s amendment; measured on two seeds at 500k in
  `experiments/raif-mountain-car/pref-rollout-probe-500k.md`.
- **The tier split, done properly** (`tier-split-audit.md`): everything pinned
  to the R-AIF replication lives in `reference/raif/` (`world`, `agent`,
  `crspp`, `revise`, `metrics`, `seed`, `run`, `fixtures/`, `columns`), the
  shared benchmark environments stay at `reference/envs/`, and the engine no
  longer carries the replication's protocol: the training side's `Episode` is
  six generic columns plus a task's own `extras` under a `ColumnSpec` schema;
  the world-model assembly finds its heads by name, shape or `roles=` and runs
  each head's own loss; neural constructors take `init=` and require their
  widths; the actor trainer speaks `policy_entropy_scale`,
  `prior_entropy_scale`, `bc_scale`, `bc_weights` with the replication-only
  terms off by default; the ensemble trainer takes `member_weights=`; the
  CRSPP row registers when the tier is imported; engine error strings name no
  reference module. Engine docstrings describe functionality; the
  replication's narrative moved out, verbatim, to
  `experiments/raif-mountain-car/docstring-salvage-2026-09-21/`, and has since
  been distilled into the acting ledger, the design records and the module
  correspondence; the folder
  is now an index of where each fact lives.
  The one-way import rule is unchanged and tested. Saved runs resume unchanged
  under `python -m hybrid_aif.reference.raif.run`.
- **v0.9.1 (2026-09-30)**: the first performance review, closed. The pieces are
  Runnalls' merge done incrementally; the composition driver's agent tick,
  `replay` and `rollout` compiled; CEM candidates scored once; the encoder out
  of `RSSMCore.observe`'s scan; one Kalman pass on the mode-aligned switching path;
  `lax.scan` for the horizon loops; no explicit inverses. Enumeration and the
  sophisticated planner now fold one prefix-sharing policy tree
  (`planning/tree.py`). Also in this release: the `sophisticated` planner
  (PR #6), one compiled rollout shared by `random_policy` (PR #4; both retired in
  0.22.0, batched `run` the simulation path), the
  switching noise statistic laid out as inference lays it out (PR #5), and the
  two-rollout rule reaching recognition (`infer_states(key=)`, a sampled
  posterior for acting).
- **v0.10.0 (2026-10-02)**: a selection buys information about what it
  selects. Under a partial `epistemic=` selection a channel's info-gain leaf is
  `I(o; S ∩ D)` and its ambiguity `E H[o | S ∩ D]`, on discrete and switching
  linear-Gaussian channels alike; `epistemic=True` and full-cover selections are
  unchanged bit for bit. A minor version because partial selections now compute
  something different (zeroth-harness's switching-infogain brief).
- **v0.11.0 (2026-10-05)**: one control, read by several factors. `System.B_control_deps`
  lets a transition be sliced by other factors' controls at the step they are chosen, as its
  own control is, where reading a state that echoes the control lags a step; both can sit in
  one model. Learning, novelty, occupancy, the generative process and the description
  (`control_parents`) follow it; unsupported paths refuse. The default is unchanged bit for bit
  (zeroth-harness's shared-controls brief). It is the sugar-layer piece of the adjudicated
  control-states design (`design/control-states.md`), which lands next with description 0.3
  (`design/model-description-0.3.md`).
- **v0.12.0 (2026-10-13)**: control nodes, action nodes, composition by `merge()` and
  `couple()`, and description 0.3. Migrating from 0.11.0:
  - Faces are keyed by name: `infer_states`, `step`, `rollout` (retired in 0.22.0) and the
    rest take and return mappings keyed by factor and modality name; positional tuples are
    refused.
  - `couple()` on Systems is `merge()`; `couple()` now wires participants' ports into a
    `Wiring` that `run` takes. (`merge` makes one particle; `couple` joins separate particles,
    and what crosses a wire is action or sensation, `design/nesting-and-coupling.md` N1, N2.)
  - A control node's state factor is `u_<f>` and its action node `a_<f>`; the action face is
    keyed by action node. `num_controls` and `continuous_controls` are gone: a control node's
    size is its levels or its box, declared with the node.
  - A state factor and a modality cannot share a name: `System` refuses `["location"]` naming
    both, and the fix is to name the sensor apart (`location_obs`).
  - Description 0.3 replaces 0.2: `describe` writes variables, factors, relations, clamps and
    configuration (`description-format.md`); `build` reads 0.3 and, for this release only, the
    0.2 layout with a `DeprecationWarning`. Build a 0.2 document and describe it again to
    migrate it.
  - Refusals of independent records within a section are listed together; a parameter table
    may be written as probabilities under the `simplex` domain.
- **v0.13.0 (2026-10-15)**: planning that expects to observe, the description length, and
  closed-form evidence. Migrating from 0.12.0, two behaviours changed:
  - A Dirichlet decay names its anchor. `System.learn` with a decay other than a static `1.0`
    on a `Dirichlet` or `ConditionalDirichlet` declaration that carries no anchor raises,
    where it used to relax toward a uniform table. Declare the anchor: `from_table(...,
    anchor="prior")` keeps the declaration's own concentration (a second table of
    parameters), `anchor="uniform"` keeps a scalar that costs nothing, or pass `alpha0=`
    directly. A run without decay is untouched (`design/parameter-learning.md` D7).
  - One round of a coupled run is one tick under every order. The sequential and async
    drivers step every process once per round on the round's actions, where they used to
    step it after each mind; a composite's trajectory under those orders changes, the
    simultaneous order is unchanged (`design/nesting-systems.md`, the 2026-10-14 amendment).
  New: `System.log_evidence(prior, posterior)` and `hybrid_aif.learning.log_evidence`;
  `hybrid_aif.description.mdl.description_length(document, n)` with its `scale=` for
  Gaussian families; `System(planning_belief="posterior")`, `GridNoiseMap` (and the
  `NoiseMap` base), `R_chol_at`, `present=` on `infer_policies` and `planner_context`,
  `StepContext.qs_epistemic` and `qs_conditioned`; a parametric row runs on the
  linear-Gaussian path. Design records: `design/structure-learning.md`,
  `design/description-length.md`, `design/planning-with-observations.md`.
- **v0.14.0 (2026-10-18)**: structure learning, planning to learn a continuous head or a
  constant tracked beside the state, nonlinear transitions. Migrating from 0.13.0, these
  behaviours changed:
  - `description_length`'s `scale=` is `reference_scale=`, the width of the reference prior a
    Gaussian family's prior link is measured from; rename the keyword. Its totals move: a
    replay's learned parameters are paid for once (`description_length(model, replay=...)`
    adds the data part, and a learned address drops its family link), only a switching
    emission's live modes are priced, and the transition menu has six families where it had
    five (`log(6/5)` more per transition record; a description length is comparable across
    releases only after that shift, `design/description-length.md` D2).
  - `Replay.log_evidence` and `InferenceDetail.free_energy` cover every modality: the tick's
    variational free energy, exact where the posterior is exact and a bound otherwise, where
    the field held only the linear-Gaussian modalities' evidence and was absent on the
    discrete path.
  - A factor or control node may not be named after a reserved key word (`obs` or a slot word).
  - Beliefs are not propagated through a transition whose process noise is zero on a
    dimension (refused, under `jit` too); a constant tracked as a state takes a small process
    noise, and a generative process may still be noiseless
    (`design/per-variable-infogain.md` V10).
  New: structure learning (`design/structure-learning.md`): `gaussian_emission` over
  categorical parents, growable factors whose levels are born in `learn` and merged by
  `System.reduce` or `reduction_period`, a modality that observes an emission of its own System
  (`System.observes`).
  `System.carry(fresh, learned)` keeps learned posteriors across a params rebuild. Novelty over
  a matrix-normal inverse-Wishart emission over a continuous state, which `epistemic` mounts
  where selecting it raised (`design/parameter-learning.md` D10).
  `System(dims=...)` and `epistemic` keys `"<factor>:<dim>"` select dimensions within a
  continuous factor (V10). `NonlinearGaussianTransition(f, innovations, propagation=...)`, a
  transition through the user's own equinox module, first order or unscented, with
  `register_transition` so `describe` can write it (`design/nonlinear-transitions.md`,
  `description-format.md` 4.4). `LinearGaussianObservationMNIW.predictive` and
  `predictive_log_prob`. A neural transition under `planning_belief="posterior"`, which raised,
  steps the epistemic chain. The demo notebook gains Sections 8–11.

- **v0.15.0 (2026-10-20)**: closed-loop scoring of Gaussian targets, each decomposition
  ablating its own epistemic leaf, per-step targets conditional on a control node, and rounds
  of different durations. Migrating from 0.14.0, these behaviours changed
  (`design/per-variable-infogain.md` V11), with no switch restoring the old forms:
  - `risk_ambiguity` scores change wherever a channel is undesignated (its risk is now
    `−H[q(o)]`) or unselected (it now carries no ambiguity, where every channel used to).
  - On a Gaussian-carried factor, `infogain` is per channel (`infogain["<modality>"]`, where
    it was `infogain["<factor>"]`) less an `observation_interaction["<factor>"]` leaf when two
    or more selected channels read it. Under `planning_belief="posterior"` the channels' leaves
    less the interaction equal the old block leaf; under the default `"prior"` the scores
    change, since that leaf was the belief's negentropy `−H[q(x)]`, not information gain. A
    plan that relied on that drive (the price of injected noise) declares
    `terms=("uncertainty:<factor>",)`. The interaction can be negative on a sub-block.
    Under `risk_ambiguity` the factor carries `observation_correlation["<factor>"]` whenever two
    or more channels read it, `epistemic=False` included.
  - A Gaussian emission over categorical parents follows the selector: under
    `utility_infogain` a selected one carries `infogain`, `I(y; z)`, so a structure-learning
    agent planning with it selected has an epistemic drive toward readings that tell its
    levels apart; under `risk_ambiguity` its ambiguity is mounted only where it is selected,
    and its risk is `−H[q(y)]` in every case. `epistemic=True` on a model whose emission is a
    matrix-normal inverse-Wishart one still raises on that emission's parameter novelty; select
    `("z",)` or `("obs:<modality>",)`.
  - `obs:<modality>` beside dimension keys selecting part of the factor that channel observes
    is refused.
  - Under the sophisticated planner with `branch_on`, a state target, per-step and terminal,
    scores its cross-entropy, not its KL (above); a conditional terminal target scores its
    negative expected log-probability under the final belief.
  - A `ConditionalGaussian` or `ConditionalCategorical` target with no parent declared, which
    passed validation and read its parent axis off the array, is refused naming `C_deps`.
  - A component of `merge()` cannot carry an `uncertainty` row of its own; declare the row on
    the composite (`terms=("uncertainty:<component>.<factor>",)`). A component's `True` or
    wildcard no longer names a continuous factor no channel observes.
  - A continuous factor no channel observes carries no `infogain` leaf; a literal `epistemic`
    key naming it raises, naming `terms=("uncertainty:<factor>",)`; `True` and wildcards pass
    over it.
  - The identity `risk + ambiguity = −(utility + infogain)` now holds per channel wherever the
    channel's epistemic leaf is mounted, designated or not, and with everything selected the
    two forms give the same G; it no longer needs a designation, and under ablation the forms
    differ by `Σ H[q(o)]` over the ablated channels (by `H[q(Y)] − H[q(Y_S)]` on a Gaussian
    factor several channels share).
  New: the `uncertainty` row, the `observation_interaction` and `observation_correlation`
  terms, `continuous.uncertainty`, `continuous.observation_interaction`,
  `continuous.observation_correlation`, `continuous.conditional_correlation`,
  `continuous.emission_entropies`, `continuous.emission_info_gain`,
  `continuous.predictive_entropy`, `discrete.flat_risk`, `discrete.cross_entropy_risk`, and
  `register_planner(branching=)` with `planning.planner_branches`;
  `continuous.entropy_lowering` and `continuous.entropy_drop` are gone.
  The `terms=` grammar takes `<row>:<factor>:<dim>` for a state-variable row.
  Also new: `System(feedback="linear_quadratic")` and the
  `hybrid_aif.objectives.closed_loop` module (above; `feedback=None`, the default, leaves every
  score unchanged); the description's `configuration.planning` gains `belief` and `feedback`,
  and `describe` now writes `planning_belief`, which it dropped, so a System planning under
  `"posterior"` was rebuilt under `"prior"`.
  `merge` now carries `planning_belief` and `feedback` from its components (agreeing, as the other
  planning options do); a composite of `"posterior"` components used to plan on the prior.
  Per-step conditional targets (`design/step-durations.md` T5): a
  `state_targets` or `obs_targets` entry may be a `ConditionalCategorical` or a
  `ConditionalGaussian` whose parents `System(C_deps={"lane": ("u_option",)})` names by the
  targeted node, a discrete control node or a categorical state factor. At each step the target
  is resolved under its parents (a clamped level selects its own distribution, a state parent
  averages mean-field) and scored as an unconditional one on both carriers, in a flat or a
  branching planner, and in closed loop, where the Riccati pass takes the level's precision. A
  fixed cost per option stays the control node's own `Categorical` target. The `target` record
  takes `parents`; `ConditionalGaussian` joins the families table under `target`; `merge`
  namespaces `C_deps`.
  A Gaussian target over a modality may also condition on categorical STATE factors
  (`design/terrain-field.md`), scored in the utility form (`decomposition="utility_infogain"`)
  as the parent-belief-weighted average over the levels of the expected log-density of the
  predictive under that level's Gaussian, each level taken relative to its own peak so that a
  tighter level is a stronger pull and never a bonus for being in that level; the weights are
  the parents' predictive beliefs at that step, control parents still select by the action. The
  risk form, a state target with a state parent, and a parent the modality itself reads as a mode
  (a categorical parent of the modality or the mode factor of the mixture it observes) are
  refused with sentences. A Gaussian or ConditionalGaussian target may carry a diagonal standard
  deviation (`chol` with the shape of `loc`; `obs_targets` only), and over a neural or
  diagonal-noise modality such a target is scored through the low-rank predictive in O(pixels x
  state dimension) with no pixel-by-pixel array in the forward or backward pass, which makes an
  image-space preference affordable (a 2304-pixel camera: about two seconds per warm plan call
  with four warm starts, against twenty minutes on the dense path).
  Durations (`design/step-durations.md` T1, T4, T8): `SystemParams.durations`,
  `{"u_option": [300.0, 40.0]}`, gives a round a length per level of one discrete control
  node, in the unit the preferences are per; a System has one clock, and the model description
  writes it as `duration` on the node's variable record. `System.elapsed(action, params)` returns
  the realised action's duration for an outer loop's clock (the realised level under an
  actuator; 1 when none is declared). The objective never reads a duration: a plan scores
  bit-for-bit as without it. `merge` carries a component's durations under its prefix and
  refuses two.
  The `durations` module (`design/step-durations.md` T3, T5): `compose(kernel, n)`
  writes the kernel of `n` repeats of a per-tick linear-Gaussian kernel, `discretise(A, B, Q, dt)`
  the exact zero-order-hold kernel of a continuous-time system over a duration (one `dt` per
  level gives a per-mode kernel), and `accumulated_target(target, durations)` a per-tick
  Gaussian or categorical target as the per-level conditional target of a step.

- **v0.16.0 (2026-10-21)**: a modality over an action node read in planning, presence under a
  control node's levels, and a faster evaluation of the linear-Gaussian epistemic leaves.
  Migrating from 0.15.0 changes no score: the leaves are the same within float tolerance, and
  the two features are declared or passed, not defaulted. For a model migrating from 0.14.0 by
  the respelling above, a dimension selection respells as one `uncertainty` row per dimension,
  `terms=("uncertainty:pose:x", "uncertainty:pose:y", "uncertainty:pose:yaw")`: rows naming
  dimensions of one factor union into one sub-block leaf, the joint entropy, which is the 0.14.0
  leaf under the default `"prior"` chain (under `"posterior"` the 0.15.0 default already equals
  it, so nothing is respelled).
  A modality over an action node is read as the model's grammar already
  declares it. The planning faces (`infer_policies`, `score_policies`, `planner_context`, the
  planners) read it on the tabular and the hybrid path, and the hybrid path's `infer_states`,
  `update_empirical_prior` and `as_env` read it as the tabular path's do, so "the answer
  arrives only when the agent asked" is a channel over `(lane, a_option)` with no indicator
  dimension in the state (`design/action-node-modalities.md`). At a plan step the node's
  belief is `P(a | u_t)`: the level's one-hot under the delta kernel, the kernel's column under
  a declared actuator, the mixture of columns under a soft control step. A tabular channel or
  a Gaussian emission over the node contracts over that belief; a linear-Gaussian channel over
  a Gaussian-carried factor whose mode parents are all action nodes selects its head and noise
  at the node's level, the factor staying one Gaussian, and under
  `planning_belief="posterior"` and `feedback="linear_quadratic"` it conditions the epistemic
  chain at the step's level. A selecting channel refuses a node that is not a point (a declared
  kernel, a soft step, a soft prior) with a sentence naming the mixture-carried route; a channel
  with a state mode parent keeps the mixture rule; an observation target on a selecting channel under `feedback="linear_quadratic"` is
  refused. `learn`, `replay`, `merge`, `subsystem`, the amortized path and a `branch_on`
  modality over an action node keep their refusals.
  `present=` on `infer_policies` and `planner_context` may be given per level of a control
  node (`{"answer": {"u_option": (False, True, False)}}`), read at each step's level of the
  node, so a channel with an observation target that is absent under some levels is scored
  exactly (its leaves are zero there); and `present=` now takes effect on the tabular path,
  which had dropped it.
  Plan evaluation on a hybrid model with several linear-Gaussian channels on one Gaussian-carried
  factor is between two and two and a half times faster than 0.15.0 (about 0.55 to 0.23 ms per
  `infer_policies` call on 1296 policies): the per-channel and interaction leaves are evaluated
  from one factorisation of the stacked predictive covariance per factor per step, the leaves
  unchanged within float tolerance.

- **v0.17.0 (2026-10-21)**: a shared innovation noise across a factor's modes, and the rest of
  the 0.15.0 slowdown closed on the posterior chain with noise maps and on a dimension
  selection. Migrating from 0.16.0 changes no score: the leaves are within float tolerance of
  0.16.0's everywhere, and the one new declaration form is opt-in. A model whose per-level
  innovations share one noise can declare it once and plans in about half the time.
  The fused evaluation of the linear-Gaussian epistemic leaves now covers a
  factor with a dimension selection (`factor:dim` keys), the leaves unchanged within float
  tolerance. A downstream pose model with a three-of-four-dimension selection and noise maps on
  two channels measured 1.45 times faster per `act()` (72.8 to 50.3 ms median).
  A `ConditionalGaussian` innovation may declare its `chol` as `(d, d)` beside a `loc` with the
  parent axes, one noise shared by every mode with the mean per mode (the description format
  writes and reads it), so the covariance recursion is the same for every policy and is
  evaluated once: on the benchmark model of 1296 policies `infer_policies` lowers to 90k flops
  against 320k and runs in about half the time, the scores bitwise equal.
  A modality declared absent for the whole plan (`present={"name": False}`, or a concrete
  all-`False` array) costs nothing in planning: it leaves the fused readings and the epistemic
  chain, and its leaves stay in the tree as zeros. Each noise map is read once per step, shared
  by the chain and the information-gain leaves, and the conditional covariance of a dimension
  selection comes from the trailing block of the belief's Cholesky factor. On the downstream pose
  model the median `act()` is 35.9 ms against 77.9 ms before and 42.4 ms for 0.14.0 (one session
  on a loaded machine, so the figures are relative); the term trees agree within 1.5e-6 and
  `G` is bitwise equal.

- **v0.18.0 (2026-10-22)**: the gradient planner's floor and best-iterate return, and a planner
  for mixed control spaces. Migrating from 0.17.0 changes no score; the gradient planner's
  plans move only in directions where the gradient is at the precision of the evaluation, and
  its result gains two fields.
  The gradient planner floors Adam's normalisation at the precision of the plan's
  dtype relative to the gradient as a whole (`sqrt(eps) · RMS(gradient)`, about 3.4e-4 of the RMS
  in float32; the plan is float32), so a coordinate whose gradient is rounding noise of the
  evaluation of `G` takes a proportionally small step, and it returns the best plan it scored
  (the warm start, every iterate and the final plan) rather than the last iterate, so a result is
  never above its warm start up to the rounding between the tree's total and the scalar score the
  descent minimises. A descent that ends on a non-finite score still returns NaN. The score of
  any plan is unchanged; a descent whose coordinates all stay above the floor is bitwise the
  previous one, and plans found in flat directions move. Against the previous rule the returned
  `G` is better in most cases of a battery and within 1e-4 worse in a few unconverged ones. On
  the sensor-detour agent the end points of twelve starts perturbed by 1e-6 spread by about half
  as much in `G` (4.8e-4 against 9.6e-4). On a downstream pose model the turn-rate step from a
  straight plan is under 1e-3 rad/s against 0.03 to 0.045; the floored descent's last iterate
  still wanders (171.069 against a warm start of 170.994), and the returned plan is the best one
  scored, the warm start.
  The result carries `gradient_trace` (the RMS gradient before each step) and `best_step` (the
  index of the iterate returned, `0` the warm start and `steps` the final plan).
  The `"mixed"` planner searches a control space that mixes discrete and
  continuous nodes (`design/mixed-control-planning.md`): for each discrete policy (the full
  enumeration over the discrete nodes' levels, or a set passed as
  `make_mixed_planner(discrete_policies=...)`), it
  descends `G` over the continuous controls with the gradient planner's own descent, vmapped
  over the discrete policies, so its cost is that of exhaustive discrete search times the
  number of descent steps. The posterior is the softmax of `−γ` times each policy's optimised
  `G` with the policy prior, every score is that of a clamped plan, and the planner returns the
  plans as a `ControlStep` policy set that `infer_policies`, `act` (under `mode="policy"`) and
  `legal=` read. `gradient_plan` shares its descent (`planning.gradient.descend`) with it,
  with its results unchanged; the gradient and CEM planners' refusal of a mixed space now names
  `"mixed"`. On the payoff model, `infer_policies` takes 6 ms for 9 policies, 16 ms for 27 and
  40 ms for 81, jitted with the default 200 steps.

- **v0.18.1 (2026-10-31)**: a `legal=` mask held as a concrete JAX array and closed over by `jax.jit`
  masks the same policies as the eager call, and the refusals of a mask that leaves no legal
  policy, or no legal level at a step, fire under `jit` as they do eagerly; a mask passed as an
  argument of the jitted function is traced and applied without them.
  Description 0.2 (the `structure`/`declarations` layout) is no longer read: `build` refuses
  such a document with a sentence naming 0.18.0, the last release that reads it, where building
  it and describing the result writes description 0.3.
  The sophisticated planner evaluates the whole deepest level unless told
  otherwise. `make_sophisticated_planner(split_depth=, batch_size=)` evaluates the subtrees below
  `split_depth` (1 when only a batch is given) a `batch_size` at a time with `jax.lax.map`;
  `make_sophisticated_planner(memory_budget=)` (bytes, opt-in) chooses both from an estimate of
  the deepest level's memory, deepening the split until one subtree fits and taking a batch that
  divides the node count; a given depth or batch overrides its part of the choice.
  `branching_tree_memory(system, params, beliefs, memory_budget=None)` returns the estimate and
  what a budget would choose, evaluating nothing: run it once at setup and pin the result. Results
  agree with whole-level evaluation up to float rounding (a few ulps, the order of eager against
  `jit`), not bit for bit. A per-step `legal=` mask, refused before, masks illegal actions out of
  each node's softmax at every depth (their `G` is `+inf` there); a first-step mask applies at the
  root as before. Design: `design/sophisticated-memory.md`.

- **v0.19.0 (2026-11-04)**: recognition beside analytic factors, a neural likelihood inverted
  by linearisation, a levelled noise map, more of `merge` and `subsystem`, and a slow-tier mind
  that holds its action for an option's duration. Design:
  `design/recognition-beside-analytic.md`, its amendment of 2026-11-04 included.
  Migrating from 0.18.1 changes two things in inference: a mode-aligned mixture's own mode
  parents now read the evidence of a switching channel over the factor (below), and a neural
  likelihood (`MLPConditional`) over an analytic factor now conditions the belief it observes,
  where 0.18.1 left the belief alone and scored the term at a point, so a model declaring one
  gets a different posterior and free energy, may declare a target on it, and plans with it. It changes no declaration that ran in 0.18.1: a recognition entry
  over analytic factors was refused at validation there (the entry routed the System to the
  neural state-space path, whose validator admits only carries and codes), and is now admitted
  on the analytic path, where the entry is a measurement (below). The mixed case, carries or
  codes beside analytic factors, stays refused.
  **Routing.** A System takes the neural state-space path if and only if it declares a
  `Deterministic` carry or an `MLPCategorical` code, the two families the analytic engine cannot
  hold as a belief. A recognition entry no longer routes. On the analytic path a `PredictorHead`
  likelihood, and a `LearnedVector` or `DerivedInitial` initial prior, are refused with a
  sentence naming the family and the families they need.
  **Recognition beside analytic factors.** A recognition entry is admitted on the analytic path,
  for a state factor or an auxiliary variable. Its output is a measurement of the variable, not
  its posterior: a Gaussian output (an `MLPConditional` with `child_dim` the factor's dimension)
  is an observation of the factor through the identity map with the module's scale as the noise,
  applied after every modality, to every component of a mixture (under
  `mixture_schedule="mode_aligned"`; refused under `"runnalls"`, as a modality over such a factor
  is); a categorical output (an `MLPCategorical` or an `MLPPredictor` with a softmax) is a
  likelihood row over the levels. The module must therefore be calibrated as a likelihood: trained
  under a flat prior over the variable's range, or with its training prior divided out, since a
  module trained as a posterior carries its prior and the analytic prior would be counted twice.
  Parents are `obs:<modality>` (continuous), `external:<name>` and `recognition:<name>` of an
  auxiliary variable, evaluated in an order derived from the parents; a bare factor name, an
  empty parent set, a cycle, an entry's output read by another entry's factor, and a module that
  does not fit the factor's family and width are refused. An entry whose observed parent is
  absent contributes nothing. An entry replaces the engine's inversion of exactly the modalities
  it covers, its `obs:` parents and, through `recognition:` parents, theirs: a covered modality,
  analytic or neural, is not conditioned on again, and enters the free energy as the expected
  negative log-likelihood under the tick's final belief (closed form for a linear-Gaussian
  modality, second order at the mean for a neural one). With `detail=True` the free energy is
  the analytic one minus each message's log normaliser plus its expected log message, plus each
  covered modality's expectation; the normalisers are the detail's `recognition_log_evidence`,
  keyed `recognition:<variable>`, and the expectations are its `covered_terms`, keyed by
  modality. The module is frozen: a nonzero rate on `recognition:<variable>` in
  `learn` is refused. A factor without an entry is inferred by the structured rules, and
  `describe` and `build` round-trip an entry on this path as on the neural one.
  **A neural likelihood is a likelihood.** An `MLPConditional` in `likelihood` over a
  Gaussian-carried or mixture-carried factor, with categorical state or action-node parents, is
  read at a belief as the linear-Gaussian model `(H, c, R)` whose moments match the belief's
  push through the network, by the declaration's static `propagation`: `"first_order"` (the
  default: the Jacobian at the mean, the extended Kalman reading, exact for an affine mean) or
  `"unscented"` (the sigma points' statistical linearisation), the two values a
  `NonlinearGaussianTransition` takes; a non-default value off the `likelihood` slot is
  refused. `infer_states` conditions on it at the belief the factor holds when the modality is
  reached (a noise map is read there too), per component of a mixture at the component's mean
  and per level of a categorical parent (the switching form), its log evidence entering the
  free energy as a linear-Gaussian modality's does; `present=` masks it. A neural modality over
  categorical parents alone is read exactly, as the Gaussian emission it is. Planning reads the
  same model linearised at the belief each term uses, re-taken along the plan, in every term
  (information gain, ambiguity, risk, utility, the cross-modality rows, the closed-loop pass,
  the gradient planner), and an `obs_targets` entry on the modality is admitted as on a
  linear-Gaussian one, a Gaussian over its output space (an image is a preference over pixels).
  `learn` trains the network by the gradient of the free energy `infer_states` reports, so it
  takes the tick's `priors=` (and `externals=`) as `infer_states` does, predicting the prior
  from `prev_beliefs` and `action` when those are given. `as_env` draws through the network. `describe` writes `propagation`
  and `build` reads it; `merge` and `subsystem` carry the modality and its target. Refused by
  name: two continuous factors on one neural modality, a spread prior on an action-node parent
  (a point selects the level), and `mixture_schedule="runnalls"` over a mixture-carried factor,
  as for a linear-Gaussian modality. An affine network (depth 0, constant scale) reproduces the
  linear-Gaussian path's numbers under both propagations. Planning channels, the observation
  model declared beside a neural likelihood for planning, are gone: the likelihood is the
  model.
  **The `approximate` status.** `Interval` carries a static `approximate` beside `derived`:
  `derived` says the enclosure holds on the model the planner was handed, `approximate` that
  this model is a linearised or quadrature reading of the declared one. It propagates by OR and
  is stamped on every leaf about a factor whose transition is an `MLPConditional` or a
  `NonlinearGaussianTransition`, or that reads such a factor through `B_deps`, and on every leaf
  about a neural modality or a modality over such a factor; the policy totals carry it. A
  purely linear-Gaussian model has no approximate leaf. `description_length(..., replay=...)`
  reports `data_exact` beside `data`, the same status on the data part: `False` when a neural
  likelihood, an `MLPConditional` transition or a `NonlinearGaussianTransition` enters the
  replay's free energy, `True` otherwise, `None` without a replay; the model part and its
  `exact` flag are unchanged, and a recognition entry is not priced in the model part.
  **Composition.** `merge` carries recognition entries and external inputs
  under the component prefix, with parents renamed, and `subsystem` slices them. A modality
  whose `A_deps` name an action node (the 0.16.0 form) composes under `merge` and `subsystem`: the
  composite reads the realised action of the node, and a view carries every node a modality of
  the view observes as one of its own control nodes; `learn`, `replay` and `predict_beliefs` keep
  their refusals. A parameter-free `terms=` selector of a component is namespaced on its variable
  segment (`uncertainty:x` becomes `uncertainty:rover.x`), the union equal to the composite's
  declared by hand; a component declaring `scoring` is still refused. A
  `LinearGaussianTransition` that reads a discrete control node keeps its tensors under
  identification with another node and is sliced at a pinned level; averaging over the node's
  levels is a mixture and is refused.
  **A sensed node's prior.** `infer_states` reads the prior over an action node a modality
  observes from, in order, the entry in `priors`, the realised action `previous=` carries for
  that node (as a one-hot), and the uniform default, so a mind that passes its previous action
  need not seed the node by hand; with neither an entry nor a previous action the point-prior
  refusal stands, with its hint.
  **An unread control node.** A discrete control node no transition reads is admitted when
  something else sizes it: a `C_deps` target conditional on it, a modality observing its action
  node, or a per-level `params.durations` entry. Every source that names the node must give it
  the same level count, and a node nothing sizes is refused where the params meet the System
  (`initial_priors` and the faces), no longer at construction; the sugar form `controlled=` keeps
  its construction-time refusal. A relay into such a node under `merge` is refused by name,
  identification with a driver works, and `describe` and `build` round-trip it. Design:
  `design/control-states.md`, amendment.
  **A view contracts a partner's node in a target.** `subsystem` carries a per-step target
  (`obs_targets`, `state_targets`) conditional on another component's control node by
  contracting that parent axis with `partner_priors[node]`, as it terminates the node's
  transition axis: an int slices the level, a Categorical mixes a categorical target's
  preferences under its probabilities in probability space, as a soft step does, and a
  Gaussian target under a spread prior is refused, as is a missing entry. Design:
  `design/nesting-systems.md`, amendment.
  **A levelled noise map.** `GridNoiseMap.values` may lead with one level axis per categorical
  parent of the modality (an action node counts), `(*levels, *grid_shape, d_obs, d_obs)`; a
  level is selected first and the grid read at the belief's mean after, so a levelled map at one
  level equals the plain map declared for that level. It is read in inference, in planning with
  observations, in the information-gain leaves, in the gradient planner (differentiable in the
  mean), in the environment's draw, in `describe` and `build`, and in the description length
  (every level's grid is counted). A map without level axes is unchanged. It is built for a
  channel whose mode parents are all action nodes, or that has none, over a Gaussian-carried
  or a mixture-carried factor (each component reads the map at its own mean); on a state mode
  parent it is refused. A channel over action nodes on a mixture-carried factor selects its
  head and noise at the nodes' levels, as on a Gaussian-carried factor, when its noise is a
  map, and the nodes' priors must then be points; a spread prior is refused by name. A channel
  without a map over action nodes on a mixture-carried factor keeps its switching read. Design:
  `design/planning-with-observations.md` D3, amendment.
  **A switching channel on a mode-aligned mixture.** A switching linear-Gaussian channel over a
  mixture-carried factor whose categorical mode parents the channel does not read (a channel
  over action nodes is the case) now sends those parents the evidence message the
  non-switching update sends, per component with the channel's modes marginalised at their
  belief, and reads its log evidence under the mode belief the tabular stage left, so a channel
  over action nodes with or without a map gives the same factor, mode and evidence posteriors
  when the node's belief is a point. A channel that reads one of the factor's mode parents
  keeps its switching message, which does not read the mixture's banks; under
  `mixture_schedule="runnalls"`, and for a factor without mode parents of its own, nothing
  changes.
  **A mind that holds its action for an option's duration.**
  `composition.hold(system, params, respond, ticks_per_unit=1)` wraps a `RespondFn` so a slow
  tier decides once per option (a temporally extended action, its duration per level in
  `params.durations`) and holds that action while the fast tier runs every round under `run`,
  with no hand-written outer loop. On a decision round it calls `respond`, stores the action and
  reads the duration with `System.elapsed`; it then returns the stored action for `duration *
  ticks_per_unit` rounds, counting down on the host and calling nothing, so the slow System's
  factors step once per option and the fast tier reads the level in force. The first round
  always decides; the wrapped carry is a `HoldCarry(inner, remaining, action)`. Durations are in
  the preference's unit, and a product with `ticks_per_unit` that is shorter than a round or not
  a whole number of rounds is refused. A System with no `params.durations`, or with a non-identity
  actuator kernel in `params.actuator`, is refused at construction. In the composite form the
  held System must own no world process (documented, not checked); the wired form has no such
  restriction. `System.elapsed` is legal under `jit` and `lax.cond`: the positivity check of
  `params.durations` reads concrete values and skips traced ones. Nothing changes for a model
  that does not call `hold`. Design: `design/step-durations.md` T9.
  `examples/demo.ipynb` re-executes clean under this release.

- **v0.20.0 (2026-11-15)**: one engine for composed Systems (`merge` and `couple` return a System,
  `run` steps it), emissions and the mirror, parameters as named variables, a System's nodes named
  by kind, a neural likelihood's update in state space, transition noise that depends on the state
  and the control, compiled faces and warm-started planners, and an API reference.
  **Migration from 0.19.0.** This release BREAKS code that composes or runs Systems through
  `hybrid_aif.composition`, that names nodes with `factor_names`, that declares `System(lagged=)`
  or passes `infer_states(previous=)`, or that uses a keyword renamed below. Each old spelling is
  refused, with no alias: a removed name raises `ImportError` or `AttributeError`, a removed keyword
  `TypeError` (`factor_names=` with a sentence saying how its tuple splits). A single System that
  uses none of these declares, infers, plans and learns as in 0.19.0, except that: its description
  length is larger by `log*(1)`, about 1.05 nats (the graph code counts emissions); a neural
  likelihood's update agrees with 0.19.0 to float32 rounding (below); validation refuses a tabular
  table whose parent axes do not match its categorical parents' level counts; and an eager face
  called more than ten times on one System warns once that `System.compiled()` exists. A model
  document written by 0.19.0 builds as before. One holding a recurrent datum (a likelihood record's
  `options.datum`, from `System(lagged=)`) is read as the declaration in the table below: a state
  factor source becomes an identity emission `<source>_read` (suffixed `_2`, `_3`, ... where the
  name is taken), one for each reading modality, and an action node source is observed directly;
  the replay of such a document equals 0.19.0's bit for bit, its absence flags keyed by the
  emission, and `describe` writes the result in the 0.20.0 form. A datum 0.19.0 also refused (not
  `name[t-1]`, or not a state factor or action node) is refused naming the option. Seeded runs
  of composed Systems draw from a different key layout, so their numbers differ from 0.19.0's.
  Composition and running (the driver module `hybrid_aif.composition.driver` is removed):

  | 0.19.0 | 0.20.0 |
  |---|---|
  | `ActionControl(driver="a.u_v", control="w.u_v")`, a control-node driver | `Link("a.a_v", "w.u_v")`, from the driver's action node; `w.u_v` stays in `w`, driven |
  | `ActionControl(driver="a.x", control="w.u_x")`, a state driver (the relay) | `Link("a.x", "w.u_x")` |
  | `ObservationLink(modality, factors)` | `Link(factors, modality)` |
  | `TargetEmission(target, parents)` | `Link(parents, "terminal:<target>")` |
  | `BoundaryPrior(control, prior)` | `Clamp(node, level)` |
  | the faces of a merge whose components declare a habit, precision or durations, or differ on a planner field, or hold a driven node | `joint_planner(composite, params)`, then the faces of the System it returns |
  | `subsystem(composite, params, name, partner_priors={control node: v})` | `subsystem(composite, params, name, inputs={action node: v})` |
  | `subsystem_factors` | `subsystem_states` |
  | `couple(minds={n: (system, params, respond)}, processes={w: (env, env_params)}, wires=[Wire(a, b)], fixed=...)`, a `Wiring` | `couple({n: (system, params), w: (system, params)}, [Identify(a, b), ...], fixed=...)`, a `(System, SystemParams)`; a world modelled by a System is `mirror(system, params, lower=True)`, an outside environment `black_box(ins=..., outs=...)` |
  | `run(wiring, ticks=..., key=...)` | `run(system, params, ticks, key, steps)` |
  | `run(composite, params, minds, ticks, key, ...)` | `run(*with_mirror(composite, params), ticks, key, steps)`, the minds keyed `"model.<name>"`, the world `"world"` (`per_component=True` where components step their own state factors) |
  | `agent_respond(system, params, planner, **kw)` | `plan(planner, **kw)` as the node's step (`learn=`, `trace=`) |
  | `ancestral_respond(system, params)` | `sample()` for a process sampled from its System, or a custom step |
  | `hold(system, params, respond, ticks_per_unit=k)`, `HoldCarry` | `params.durations` in the node's view and `run(..., clocks={node: k})` |
  | a `RespondFn` `respond(key, obs, carry)` | a step `step(system, key, obs, carry, params) -> (out, carry, params)`, `obs` `None` on the first call |
  | `Trajectory` (`states`, `observations`, `actions`, `realised`, `carries`), `MindRecord`, `ProcessRecord` | the `Trajectory` tree (`received`, `present`, `intended`, `emitted`, `decided`, `params`; states in `carries` under `record_carries=True`) |
  | `System(lagged=(("m", "x"),))` | `System(emissions={"x_read": Emission(("x",), realise=r, kernel="identity")}, observes={"m": "x_read"})`, `r` `"mode"` for a Gaussian source (its mean), `"stats"` for a categorical one (its probabilities); the previous action: `observes={"m": "<action node>"}` |
  | `infer_states(..., previous=...)` | the datum keyed by the variable observed; a sensed action node's previous level as a one-hot in `priors` |

  A perceiver of another node's action node sees it absent in round 0 under `run`, where the wired
  form handed it level 0. Names (`state_names`, `control_names` and `modality_names` generate as
  before, so no key in params, targets, selectors or documents changes):

  | 0.19.0 | 0.20.0 |
  |---|---|
  | `System(factor_names=...)` | `System(state_names=..., control_names=...)`: the first `len(B_deps) - len(controls)` names are the state factors'; under the sugar (`controlled=`) `state_names` alone |
  | `system.factor_names` | `system.node_names` (state factors, then control nodes), or `state_names`, `control_names` |
  | `GenerativeEnv.step(factors=)`, `step_actuated(factors=)` | `states=` |
  | `System.grow(factor=)`, `System.reduce(factor=)` | `state=` |
  | `Subsystem.factors` | `Subsystem.states` |
  | `num_factors` (`Env`, `GenerativeEnv`, the processes, `MountainCar`) | `num_states`, the abstract method of `Env` |
  | `inference.condition(R_chol=)`, `condition_on_expected(R_chol=)`, `objectives.hybrid.condition_expected(R_chol=)` | `noise=`, a Cholesky factor or an `inference.DiagonalNoise` |
  | `Conditioner.R_chol_at` | `Conditioner.noise_at` |
  | `Statics.factor_names` (the estimator registry) | `Statics.state_names` |

  **One engine.** A closed System (a merge or a coupling whose every modality is fed by an
  identification) is stepped by `run(system, params, ticks, key, steps, clocks=None,
  tick_order="simultaneous", carries=None, record_carries=False)`, each leaf, or a non-leaf node
  planned as one mind, by `plan()`, `sample()` or a custom step; `with_mirror` closes a System on
  its own mirror, per component with `per_component=True`. The bindings are `Link`, `Identify` and
  `Clamp`; `couple` returns a System whose leaves keep their own planner configuration, and
  `joint_planner` plans a node as one mind on request. Under `"simultaneous"` every view of a
  round is taken from the home as the round began and write-backs land at its boundary;
  `"sequential"` orders the sources by the listing and steps each driven node after its last
  driver; `"async"` draws a fresh order of the sources each round. A node whose view declares
  durations decides once per option, `clocks=` converting a duration into rounds. `black_box(ins=,
  outs=)` declares an external process by its typed ports, stepped by host Python
  (`env_step(env)` for a positional environment); its inside is listed unscored in the
  description length and its ports priced. `plan(warm_start=True)` plans receding-horizon: the
  carry keeps the plan each round acted on (`MindCarry.plan`) and the next round's planner starts
  from it shifted one step (a planner taking `init`, the gradient and mixed planners; others
  refused), and `plan(trace=True)` also keeps the planner's policy set, the inference detail and
  the parameter variables of the round (`Trace.plans`, `detail`, `parameters`). Design:
  `design/engine-consolidation.md`, `design/couple-and-run.md`, `design/nesting-and-coupling.md`,
  `design/description-length.md` (note of 2026-11-10); `composing-systems.md`; the entries under
  "In the field" below.
  **Emissions and the mirror.** An emission is an out-port of a state factor (`System.emissions`,
  `Emission(parents, realise, kernel)`, `System.emit`, `System.ports`); `System.observes` maps a
  modality to the action node or emission it reads, which is how a belief read, in a component or
  across a merge (`Link(source, modality, realise=..., kernel=...)`), is declared.
  `mirror(system, params)` is the world a System models, a System that acts; `as_env` is
  unchanged. Design: `design/emissions-and-mirror.md`.
  **Parameters as named variables, and a learned field.** A tabular table may be declared once in
  `SystemParams.parameters` and read through `Parameter("<name>")` by several state factors or
  modalities, learned once under `parameter:<name>`, identified across components by `Identify`
  and returned from a view by `write_back`. A tabular state factor may read the cell a continuous
  state factor falls in (`System(cells={"x": Grid(...)})`, `cell_mass`), so a Dirichlet table over
  a grid is a map the agent learns and plans to learn; a slow state factor's novelty may pool a
  declared count of readings per option level (`System.readings`); a Gaussian target over a
  modality may condition on a categorical state factor. Design: `design/shared-parameters.md`,
  `design/terrain-field.md`, `design/step-durations.md` T10.
  **A neural likelihood's update in state space.** An `MLPConditional` likelihood's noise is
  independent across its `p` readings, so `infer_states` (and through it `learn`, `replay` and the
  description length's evidence) and the conditioning step of the planner's epistemic chain update
  a Gaussian or mixture belief by a square-root information form at `O(p d²)`, with no `p` by `p`
  array, by default and with no switch; `inference.condition` takes this form when its noise is an
  `inference.DiagonalNoise`. Measured, one update at 1024 readings over a 6-dimensional state
  factor: 0.07 ms against 5.8 ms in observation space; at 16384 readings, 0.47 ms against 13.2 s;
  and closer in float32 to an exact reference than the observation-space form. Still formed as `p`
  by `p` matrices: the dense noise factor every planning leaf's model carries, the information gain
  and correlation leaves of a Gaussian-carried state factor read with other modalities, the
  sub-block ambiguity, the risk or utility predictive against a dense target, and under
  `"unscented"` every planning leaf and a covered modality's expected log-likelihood. Design:
  `design/recognition-beside-analytic.md` L10.
  **Transition noise that depends on the state and the control.** With
  `NonlinearGaussianTransition(..., noise_scale="per_coordinate")` the map returns `(mean,
  scale)` and the innovations' Cholesky factor is multiplied by `diag(scale)` on the left: read
  at the belief's mean to first order, as `E[s sᵀ]` over the sigma points unscented, at the sampled
  state in `as_env`. The scale must be positive, finite and differentiable wherever it is
  evaluated (a square root that vanishes at rest needs a guard, or the gradient planner returns a
  NaN plan). `describe` records `options.noise_scale` only when set; a map without a scale is
  unchanged. Design: `design/nonlinear-transitions.md`, amendment of 2026-11-13.
  **Compiled faces and planners.** `System.compiled()` returns the five faces compiled once
  behind input canonicalisation, with a retrace watcher (`hybrid_aif.debug`); `mixed_plan` and
  `gradient_plan` take `starts=`, warm starts bound through `make_mixed_planner` and
  `make_gradient_planner` (`starts=1` is bit-identical to before; the gradient planner's face then
  returns every start's plan); `infer_policies(return_policies=True)` returns the policy
  set; the information gain of a diagonal-noise neural modality over a mixture-carried state
  factor and a diagonal image preference are scored in low-rank form. Design:
  `design/recompilation.md` R6, R7; `design/mixed-control-planning.md`;
  `design/recognition-beside-analytic.md` L9.
  **The API reference.** `docs/api` is a Sphinx source (the `docs` extra), one page per group of
  public names; every field of the public field-bearing classes carries its own description.
  **The survey example** (`examples/survey_*.py`, not library API): a rover with a vehicle state
  (heading, speed and energy in the state, a modelled stall), a forward camera read as a neural
  likelihood, a terrain field it learns, and a scheduler merged over it that shares the field;
  the runner's `--settings` picks the scheduler over the rover or the rover alone. It records the
  project page's replay.

- **v0.21.0**: **switching nonlinear dynamics.** A
  `NonlinearGaussianTransition` may name discrete state factors in `B_deps`, as a
  `LinearGaussianTransition` with per-mode `F_j` does: the mode reaches `f` as an integer level
  in its declared place among `controls` (one entry per parent in `B_deps` order; a factor with
  no state parent is unchanged) and the innovations take one axis per discrete parent in that
  order. The factor is carried as a mixture: each component is pushed through `f` once per joint
  mode, to first order or unscented, weighted by the mode's belief, and reduced mode-aligned or
  by Runnalls as the linear bank is; a fix on the factor sends mode evidence to the parents,
  `as_env` steps `f` at the sampled mode, and `describe`/`build` round-trip it. An affine `f` per
  mode equals the linear bank to float tolerance. Was refused with "switching dynamics";
  `objectives.hybrid.refuse_switching_nonlinear` is removed. Design:
  `design/nonlinear-transitions.md` N4, amendment of 2026-11-15.

- **v0.21.0**: **parameters learned within a plan.**
  `System(plan_learning=...)` names the learnable tables a plan reads as learned within it, in
  the grammar of `learn`'s rates (`"transition:<factor>"`, `"likelihood:<modality>"`,
  `"parameter:<name>"` for every reader of a parameter variable) or `True` for every one; the
  default `()` is today's frozen reading. Each named table rides the planning carry as working
  counts (`system.WorkingCarry`), seeded from the posterior the plan starts from and absorbing
  every imagined step's expected count statistic, the model's joint over child and parents
  (the working mean times the parents' beliefs; the product of marginals `learn` absorbs from
  an observed step differs where child and parents are dependent), so a later read of a row
  predicts through the counts the plan has imagined (the Pólya urn: the mean moves where the
  plan conditions on a drawn child, in a branch or a closed pair's modes) and its novelty is
  that of the counts before the step; readers of one parameter variable
  share one working table. A closed pair (a tabular factor reading through a cell the
  mixture-carried factor that switches on it) is read jointly over the mixture's components,
  learned or not: each component reads its own cell, the switching factor grows with the
  per-component weights `w_i q_i[j]`, and the tabular factor's belief is their marginal; a
  learned table read there keeps working counts per component, merged through the reduce by the
  assignment that merges the moments (`MixtureGaussian.reduce(k, aux=)` and
  `reduce_mode_aligned(k, M, aux=)` return `(mixture, aux)`, the moments bitwise as without),
  and its novelty is the components' weight-average. An inference setting: `describe` writes it
  under `configuration.inference`, `build` reads it back, and `merge` unions and `subsystem`
  narrows its keys. The fused all-tabular path carries the same working counts (A3): each
  learned table's mean, entropy and novelty are re-read from the counts at every step and every
  other table stays hoisted; a reader whose readings are pooled (`System.readings`) absorbs its
  statistic `n` times, as its novelty leaf counts `n`; and under the sophisticated planner each
  `branch_on` branch absorbs its own outcome, the exact conjugate update `onehot(o) ⊗ q(s_P | o)`
  for a learned likelihood of the branched modality and the expected statistic under the
  branch's beliefs for every other learned table (`branching.make_branch_fn(evidence=True)`
  returns the per-outcome statistic). Without a key the fused path is bitwise as before.
  Refused with a sentence: a key naming no learnable table, the amortized path (it hoists its
  tables; its transitions are networks), a parameter variable read in a closed pair and by
  another reader learned within the plan, a learned likelihood under `present=`, and on the
  hybrid path a learned likelihood under `branch_on`. A grid axis of one cell now has a cell
  mass (it raised). `System(sophisticated=...)` declares both halves of sophisticated planning
  in one tuple of `epistemic`'s selector keys: an `"obs:<modality>"` key branches on that
  modality (the state half, `branch_on`) and a `"transition:"`, `"likelihood:"` or
  `"parameter:"` key carries that table's counts within the plan (the parameter half,
  `plan_learning`); construction lowers it into those two canonical fields, which every face
  reads and every document carries (`sophisticated` never appears in one), and declaring it
  beside `branch_on` or `plan_learning` is refused. Design:
  `design/parameters-within-the-plan.md` D1, D2, A3, A1.

- **v0.21.0**: **learnable process noise.** `LinearGaussianTransition` and
  `NonlinearGaussianTransition` take `noise_prior=`, an `InverseGamma` or `InverseWishart` over
  the innovations' covariance `Q`, shared across the modes or one per mode (leading axes
  `innovations.parent_shape`). Every consumer reads `chol(Ψ/ν)` (`diag(sqrt(β/α))`), the
  expected-precision point, through `factors.transition_point` (prediction, planning, the
  closed loop, `as_env`, which lowers it or draws it under `param_key`); the declared
  `innovations.chol` keeps only its shape. `learn` absorbs the exact pairwise residual scatter
  `E_q[(x' − F x − G u − b)(·)ᵀ]` from the one-step smoother over `(prev_beliefs, beliefs)`,
  cross-covariance included (`conjugate_stats.pairwise_scatter`), under the rate key
  `transition:<factor>:noise`; a discrete control node selects the mode (the taken level's
  one-hot), a continuous one enters through `G u`, and a nonlinear map is linearised at the
  previous mean (a per-coordinate scale standardises the residual). The evidence has the noise
  priors' closed form (`System.log_evidence`, which now also reports `parameter:<name>`),
  `describe` writes the innovations' Cholesky factor as a learned variable under its prior and
  `build` reads it back, and the description length prices it as an emission's noise prior;
  `carry` and `merge` carry it. A noise is also a **parameter variable**: `SystemParams.parameters`
  may hold an `InverseGamma` or `InverseWishart` that transitions read as
  `noise_prior=Parameter(name)` and linear-Gaussian observations as `noise=Parameter(name)`
  (`R` and `Q` one mechanism); `learn` sums the readers' statistics into one update under
  `parameter:<name>`, the document records the variable once under `parameters` and each
  reader's field as `{"parameter": <name>}`, and it is priced once. `merge` namespaces and
  `Identify` joins noise variables, `subsystem` and `write_back` carry them. `replay` takes
  `lr_by_slot=`. Refused with a sentence: a mixture-carried factor, the unscented rule, the
  noise under `epistemic` (no novelty term yet), `plan_learning` on it (not a table), one prior
  object under two readers, `durations.compose` of a learned noise. Design:
  `design/process-noise-learning.md` D1–D4 and its As built.

- **v0.22.0**: **`run` takes `batch=`.** `run(..., batch=B)` runs `B` replicas of a closed
  System by one program: replica `b` is the unbatched run under `fold_in(key, b)` (or `key[b]`
  for a stack of keys): bitwise on the tabular fixtures under simultaneous and sequential, and
  to float32 rounding where the vmapped program fuses a step differently (the survey rover's
  gradient planner differs from round 0 at the last place), which a long closed loop grows into
  different trajectories, so replicas compare with replicas; `per_replica=["params", ...]`
  names the inputs with a leading replica axis (params, a node's seed in `carries`); `record=`
  names the fields stacked over rounds, `chunk=` runs the scan in compiled chunks and
  `replicas_per_pass=` in host passes, one program for all. A tuple key of `steps`,
  `steps={("agent.rover1", "agent.rover2"): step}`, steps copies of one leaf by one vmapped call
  of the step, simultaneous among themselves, their Dirichlet write-backs summed (equal to the
  ungrouped simultaneous run); copies driven by different option nodes of one driver, or
  feeding different in-ports, are grouped too, the swap pairing their partners' ports as it
  pairs their names. Every round, round 0 included, is one compiled `lax.scan` from the
  reset: a step is a traced function, a check of a value inside it is recorded in
  `Trajectory.faults` (a non-positive Dirichlet sum, two writers of a non-counts variable, a
  level out of range) where it used to refuse on round 0, and `Trajectory.timing` holds the
  compile, the scan and the device count. A slot whose variable the reset left without a value,
  read before it is realised in round 0, is handed a `Maybe(value, present)` every round (a
  custom step reads `obs[slot].value` where `obs[slot].present`); `plan()`'s carry has its
  every round's form from the reset (`MindCarry.started`, a mixture factor padded to
  `mixture_k`). Async under a batch draws each replica's order (refused above four sources).
  **On a CPU host the speedup needs host devices**: replicas run in parallel only across
  devices (each device runs the whole scan over its share, a `shard_map`), and a CPU host has
  one unless `XLA_FLAGS=--xla_force_host_platform_device_count=N` is set before jax is
  imported; on one device a batched survey run is no faster than its replicas one after
  another. Measured on a 16-core host, the one-level survey: 101 replica-ticks per second at
  `batch=16` over 16 devices against 8.7 unbatched; the two-tier survey's ten 800-round
  replicas in 154 s against 1099 s one after another. `examples/survey_run.py --replicas N` and
  `examples/compose_minds.py` set the flag when it is unset (`examples/host_devices.py`).
  **`rollout` is retired** (`hybrid_aif.rollout`, `random_policy`, `History`): a model against
  its world is `run` over `with_mirror`, a bare `Env` a black box stepped by `env_step`, both
  batched; `Env`, `GenerativeEnv`, `as_env` and `broadcast_params` stay. Design:
  `design/batched-run.md` R1–R14.

- **v0.23.0**: **float precision follows the model.** The package computes in the float dtype
  of the arrays it is given: a caller's float array keeps its dtype (at least float32), arrays
  of different dtypes compute in the wider where they meet, a Python number or an integer array
  takes the dtype of the float arrays beside it (alone, the default float dtype), an array the
  package makes takes the dtype of what it is combined with, and nothing is narrowed implicitly
  (the rule is the module docstring of `hybrid_aif/dtypes.py`). Under JAX's 64-bit mode
  (`jax_enable_x64`) a 64-bit model therefore runs in 64-bit, continuous controls included,
  where parts of it were cast to 32-bit before. `cast(tree, dtype, where=None)` moves the float
  arrays of any tree (a `SystemParams`, beliefs, one distribution, a network, an episode) to
  float32 or float64, integer and boolean arrays as they are, `where=` restricting it to a part
  of the tree as `equinox.tree_at` takes it; `SystemParams.cast` is the same call on the
  params. There are two declared exceptions. A neural network computes in the dtype of its
  weights (its `dtype` property) and casts its inputs to it, `dtype=` on its constructor
  choosing the dtype the weights are drawn in; the R-AIF reference draws in float32 (`DTYPE` in
  `reference/raif/columns.py`) and computes in 32-bit in either mode. A replay buffer stores in
  the dtype it was declared with. A model with parts in different float dtypes (a mixed model)
  runs in the wider one and costs set-up time: `run` settles the dtype of each value it carries
  before its loop by tracing the first round, and refuses a value that first widens in a later
  round, naming it (the remedy is one float dtype for the model, its rates and its inputs); a
  hand-written loop over such a model compiles `env.step` twice. Upgrading: with 64-bit mode
  off results are those of 0.22.0, except that a custom step returning a half-precision carry
  from a 32-bit reset keeps it at 32-bit. Under 64-bit mode results change wherever a value was
  rounded to 32-bit; `env.null_action()` is in the default float dtype; and an inline value of
  a model description builds in the default float dtype, so a 32-bit model described with
  inline arrays comes back mixed (pass `arrays="file"`, or `cast` the built params). API:
  "Float precision" in `api/system.rst`; notes dated 2026-11-27 in `design/batched-run.md`,
  `design/recompilation.md`, `design/amortization.md` and `design/gradient-planner-floor.md`.

## In the field / in flight

- **Observing a variable (v0.20.0)**: `System.observes` names, per modality, the variable it observes, an action node or an emission of the System, and
  `Emission(parents, realise="sample", kernel=None)` replaces `Emit`, with `kernel="identity"` or
  a `Grid` indicator as its static kernel beside the kernels of `params.emission` (`kernel=None`
  takes its kernel from there and refuses when it is missing). A read of a belief is such a pair: `System.lagged`, `reads.Read`,
  `read_datum`, the `previous=` and `read_key=` arguments of the faces and
  `MindCarry.posterior` are gone, and the datum reaches inference like any observation, keyed by
  the variable: from the caller for a hand `infer_states` and `learn`, from the emissions `replay`
  realises after each tick and records (`Replay.emitted`), and under `run` from the emission the
  source's step realises. A slot the map names is no in-port (`ports`), one datum feeds every
  slot observing a variable, and the description records an `observes` entry on the likelihood and the static kernel on the
  emission. `mirror` drops an emission only its own System observes, so `as_env` is unchanged.
  Upgrading: a sampled read draws from `fold_in(step key, EMIT_TAG)` rather than the retired
  `read_key`, so a seeded run that sampled a read draws differently; a read of a categorical
  factor's probabilities or of a mean or a cell by `mode`, and everything that does not read a
  belief, is bitwise unchanged (`design/engine-consolidation.md` chunk B;
  `composing-systems.md`).
- **The splice, its bindings and the joint planner (v0.20.0)**: `couple(components, bindings,
  fixed)` returns a `(System, SystemParams)`: the components spliced and joined by
  `Identify(out_port, in_port)` (an action node or an emission with a modality or a control node)
  and `Clamp`, every other binding refused naming `merge`. An identified modality or control node
  is written into `System.observes`; a **driven** control node, the in-port that carries an
  out-port's realised value, has no action node of its own (`action_names` entry `None`), is never
  chosen by a planner, and is described under `relations.driven`. `merge` is the same splice and
  the same bindings, and `Link(action node, control node)` under `merge` is that identification:
  the driven node stays in its component. A leaf keeps its own planner configuration
  (`Subsystem.config`), habit, precision, durations and scoring (`SystemParams.parts`) and
  recognition in a merge and a coupling alike, so a leaf with recognition or a scoring estimator
  is merged or coupled; the composite states a planner field where its leaves agree and holds
  `UNSET` where they differ, and states no habit, precision, durations or scoring of its own.
  `joint_planner(system, params, node=None, *, inputs=None, **config)` plans a node (the whole
  System, or a component that is not a leaf) as one mind: each driven node whose driver is an
  action node inside the node is collapsed into the driver's control node (`driven_nodes(composite,
  name=None)` lists the driven nodes with their variables), the configuration
  is an argument or the value the leaves agree on, and the habit, precision, durations and scoring
  are composed from the leaves' records, what cannot be joined refused with a sentence. A
  composite whose leaves differ, or that holds `parts`, is run leaf by leaf: `policies`,
  `infer_policies`, `act`, `elapsed` and `replay` refuse it, naming `joint_planner`.
  `subsystem(composite, params, name, inputs=None)` slices or averages a driven node and a
  partner's node by the variable that drives it; `partner_priors` and `partner_controls` are
  gone (a view's `inputs` are keyed by the action node, where `partner_priors` was keyed by the
  control node). Upgrading: a merge composes none of its leaves' habit, precision and
  durations, accepts leaves that differ and a leaf that declares scoring, and leaves the driven
  node of a same-step `Link` in its component: pass the merge to `joint_planner(composite,
  params)` wherever it is planned as one mind (the faces; `run` calls it for a step given to a
  non-leaf node).
  `design/engine-consolidation.md` chunk B; `composing-systems.md`.
- **`run` over a closed System (v0.20.0)**: `run(system, params, ticks, key, steps, clocks=None,
  tick_order="simultaneous", carries=None, record_carries=False)` steps a **closed System** (a merge
  or a coupling whose every modality is fed by an identification) and returns a `Trajectory`, a
  tree with one node per component: the params at the root, and per node `received`, `present`,
  `emitted`, `intended`, `start`, `carries`, `carry` and `decided`, with dotted lookup
  (`trajectory["agent.rover"]`). `steps` names a step for every leaf, or for a non-leaf node, which
  is then planned as one mind (`joint_planner`): `plan(planner=None, learn=None, trace=False,
  replace=None, warm_start=False, **act_kwargs)` for a planning mind (a `MindCarry` carries
  `previous`, `action`, `trace` and `plan`; `Trace` holds `q_pi`, the term tree, the enumerated
  policies, the planner's own policy set `plans`, the inference `detail` and the round's
  `parameters`; under `warm_start=True` the planner, one that takes `init`, starts each round from
  the plan the last round acted on shifted one step, zeros first, a held node shifting once per
  decision), `sample()` for a
  process sampled forward from its System, or any `step(system, key, obs, carry, params)`. One
  value per variable per round: an action node is realised from its node's `params.actuator`
  kernel when the node decides, an emission by the step that owns it, and a reader reads a variable
  as it stands at its turn. A node whose control node is driven by another node's action steps
  after that action is realised in the round, so `"simultaneous"` runs the undriven nodes on the
  round's boundary and the driven ones after them, every view of a round taken from the home as
  the round began and every write-back applied at its boundary (two of one Dirichlet variable
  summed as deltas, two write-backs of one variable of another form refused, up front for two
  `plan(learn=...)` that reach it, at the boundary otherwise; live under
  `"sequential"` and `"async"`). The first call of every step has `obs=None` and
  is its reset. `with_mirror(system, params)` couples a System to its own mirror and so closes it.
  A modality may observe another
  component's action node, which the view leaves an ordinary observation. `System.sample_step`
  takes the value of a driven control node under its control node's name. **Clocks**: a stepped
  node whose view declares `params.durations` decides once per option; `clocks={node: rounds per
  unit}` converts the duration of the chosen level (`view.elapsed`) into rounds (1.0 by default; a
  product under one round or not a whole number is refused), the node's step,
  view, write-back and realisation are skipped on held rounds with its out-ports and the variables
  it drives standing, the first round always decides, and `Trajectory.decided` records which rounds
  decided. **The mirror per
  component**: `mirror(system, params, per_component=True)` and `with_mirror(...,
  per_component=True)` mirror a merged or coupled System as the merge of one leaf per component
  (no durations). A foreign state parent of a transition becomes a driven control node of the
  reader fed by an identity emission of the owner; a modality whose parents all belong to one other
  component is an emission of that component, and so is observed as the System observes it; a
  modality with parents in two or more components keeps a one-round lag under `"simultaneous"`
  (known limitation: per-use driven nodes are on the roadmap); a control node driven inside the
  System follows its driver when the mirror is sampled alone. Upgrading: a perceiver of another
  node's action node sees it as absent (`None`, handled through `present=`) in round 0 under `run`,
  where the wired form handed it level 0. **One engine**: `run` over a closed System is the only
  form of `run`, and `couple` takes components and bindings only. Removed, with no alias: the
  composite form (`run(composite, params, minds, ...)`) and the wired form (`run(wiring, ...)`) of
  `run`, `couple`'s keywords `minds=`, `processes=` and `wires=` (`links=` earlier in this cycle),
  `Wire`, and its `fixed=` on a process
  input, `Wiring`, `MindRecord`, `ProcessRecord`, the composite form's record (`Trajectory` in
  0.19.0), `hold` and `HoldCarry`, `agent_respond` and `ancestral_respond` (and their successors
  of this cycle, `agent_step`, `ancestral_step` and the step shape `MindStep`), `particle` and
  `ParticleCarry`, and `MindCarry.emitted` (a `plan()` step returns its emissions).
  **Migration guide** (a 0.19.0 form, or its spelling earlier in this cycle, -> the form under
  `run`; the examples, the survey runner among them, the notebooks and the tests are migrated):
  - `couple(minds={...}, processes={"w": system.as_env(params)}, wires=[Wire(a, b), ...],
    fixed={...})` (`links=[Link(a, b), ...]` earlier in this cycle) (`fixed=` on a process input becoming a `Clamp` on the control node) ->
    `couple({...: (system, params), "w": mirror(system, params, lower=True)}, [Identify(a, b),
    ...])`, the process input named by an action node becoming that action node's control node
    (`world.a_move` -> `world.u_a` in `examples/compose_minds.py`);
  - `run(wiring, ticks=..., key=...)` -> `run(system, params, ticks, key, steps)`;
  - `run(composite, params, minds, ...)` -> `run(*with_mirror(composite, params), ticks, key,
    steps)`, the minds keyed `"model.<name>"` and the world `"world"` (`per_component=True` where
    the composite's components step their own factors, the arena);
  - `agent_respond(system, params, planner, **kw)` or `agent_step(view, planner, **kw)` ->
    `plan(planner, **kw)` (`learn=` and `trace=` for what a hand-written step did; `legal`
    checked on the reset call);
  - `ancestral_respond(system, params)` or `ancestral_step(view)` -> `sample()` for a world
    (another key use), or a custom step drawing from `params.policy_prior`;
  - `hold(system, params, respond, ticks_per_unit=k)` or `hold(step, ticks_per_unit=k)` ->
    durations in the node's params and `clocks={node: k}` (a count is not resumed through
    `carries=`);
  - `particle(cs, cp, steps)` inside a wiring -> the merge as a component of the coupling, each of
    its leaves stepped by name (`steps={"agent.rover": ..., "agent.mission": ...}`);
  - a `RespondFn` `respond(key, obs, carry)` or a `MindStep` `step(key, obs, carry, params)` ->
    `step(system, key, obs, carry, params)`, `obs` `None` on the reset call;
  - `MindRecord`/`ProcessRecord` and the composite form's record (`Trajectory` in 0.19.0,
    `CompositeTrajectory` earlier in this cycle; `actions`, `realised`, `observations`, `states`)
    -> `Trajectory` (`intended`, `emitted`, `received` and `present`, the world's state in
    `carries` under `record_carries=True`);
  - an `Identify` or a `Link` (under `merge`) that names the action node of a driven control node
    is refused in either order of the bindings, naming the driver's action node: a driven node
    and its driver's action node are one variable, so name the driver's action node itself (a
    join on it); a chain of same-step `Link`s through a driven node, which `merge` collapsed onto
    the first driver, is refused the same way.
  `design/engine-consolidation.md` chunk B; `composing-systems.md`.
- **Black boxes (v0.20.0)**: `black_box(ins={...}, outs={...})` (`hybrid_aif.composition`)
  declares an external process, an environment or a hand-written world, by its ports and their
  types alone (`(LEVEL, levels)` or `(FLOAT, width)`): a `System` with one control node per in-port
  (no action node, no reader), one emission per out-port (no parents, the static kernel
  `"black_box"`) and the types in `System.black_box`, with no factor and no table. `couple` binds
  its ports with `Identify` as any component's (types checked, an undriven in-port leaves the
  coupling open, a `Clamp` on one refused); `run` steps it with a custom step, which may be
  arbitrary host Python (hardware, I/O, any Python object as its carry, NumPy or Python numbers
  returned, checked against the port types and stored in one dtype: an integer level on a level
  port, a vector of the width, or one number for width 1, on a float port; never traced or
  compiled), and
  `env_step(env, params=None, *, actions=None)` is that step for a positional environment (bitwise
  the environment `GenerativeEnv.wrap` steps on the same keys). Every face that needs the inside
  refuses a System that is or holds one, with a sentence: `plan()` and `sample()` as its step, a
  sliced step or view, a joint node over it, `mirror`, `with_mirror`, `as_env`, `sample_initial`,
  `sample_step`, `emit` and the inference, planning and learning faces. `describe` writes it under
  the new `relations.black_box` (`{"ins": [...], "outs": [...]}`), the step not being part of the
  document, and `build` rebuilds it exactly; a reader of 0.19.0 refuses a document holding one,
  and every other document is written and read as before. `description_length` prices its ports
  as declared structure (each out-port an emission of the graph code with its size code, an
  undriven in-port a control node) and lists its inside under `black_box:<component>`, every link
  unscored, outside `structure`, `parameters` and `total`, and `exact` `False`; a System of black
  boxes alone totals its ports' codes (`design/description-length.md` D3, notes of 2026-11-10). The survey world
  (`examples/survey_world.world_system`) is one: its placeholder tables are gone, `world_step` is as
  it was, and the runs are bitwise. Upgrading: a world declared as a System only to type its ports
  is `black_box(ins=..., outs=...)`; a positional environment coupled through
  `GenerativeEnv.wrap(env, system)` as a process of the wired form is `black_box(...)` stepped by
  `env_step(env, params)`. `composing-systems.md`, "External processes: a black box".
- **Shipped to robots**: the downstream coverage agent (EFE-driven coverage
  at tuned-heuristic parity with the hand-tuned coefficients removed;
  stage-2 learns localization noise online).
- **Integration underway**: the downstream mission executor — AIF task
  selection + recovery (retry/skip/escalate emergent from the term
  tree) against its policy sockets, using v0.5 `merge`.
- **Amortization program, R-AIF replication**: R-AIF's `tiny`
  architecture is declared, scores G over its own beliefs, learns, and its
  module tree is proven theirs leaf for leaf by execution
  (`experiments/raif-mountain-car/module-correspondence.md`). On state-based
  Mountain Car, 8 seeds per side to 2M steps match the oracle on the acceptance
  windows (best-200k −69.1 against −68.7, last-500k −73.0 against −73.2, SR
  0.99 on both) and in trajectory shape; the pre-registered stopping rule ends
  inconclusive at a −1.0 point difference, 95% CI [−3.2, +1.0], smaller than
  the per-seed spread (`experiments/raif-mountain-car/acceptance-report.md`).
  Debt the arc introduced is tracked in `design/amortization-debt-ledger.md`.
- **Package tiers** (v0.7, re-cut v0.9): engine vs. `hybrid_aif.reference/`
  (declared reference models, concrete environments, fixtures), with
  the one-way import rule enforced by a test.

3,027 tests (3,019 in the default selection; `-m notebook` and `-m slow`
are opt-in); ruff-clean; the executed notebooks (`examples/demo.ipynb`,
`examples/objectives_a_la_carte.ipynb` — one declared T-maze under five
objectives, `examples/hide_and_seek.ipynb` — a prey and a predator coupled
through blankets, each objective swapped one line at a time) are maintained
exhibits and an adoption gate. MIT.

**Composing a new System** — constructor table, shape rules, the factor ×
modality family matrix, the canonical loop with its real return arities, and
how to read the term tree: `docs/composing-systems.md` (added 2026-09-18).
