# hybrid-aif capabilities — v0.18.0 (2026-10-22)

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
- **A Gaussian emission over categorical parents alone** (`gaussian_emission(loc, R_chol)`):
  the linear-Gaussian family with an empty regressor block, a Gaussian mixture whose
  component label is the parent; it sends the parents the exact categorical message of its
  Student-t predictive and learns by the responsibility-weighted conjugate update.
  **A growable factor**: a memoryless categorical factor whose weights (its transition
  column) carry a growth propensity `growth`; its cap is its size, a level with zero weight
  is dead, and `learn` wakes a level for a datum no live level explains better than the prior
  does. **A lagged datum** (`System(lagged=(("modality", "source"), ...))`): a modality reads
  its datum from the carry, a factor's previous belief mean or the previous action, so a
  factor can emit what the model just believed (`design/structure-learning.md`).

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
  module (an RSSM posterior over `[deter, embed]`); the rest fall back
  to the structured row. A recognition parent may be an **externally
  supplied vector** (`"external:<name>"`), which is how one model
  conditions on another's embedding as data rather than by sharing a
  module. `predict_beliefs` is the open-loop rollout.

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
  optimised `G`), the **`sophisticated` planner** —
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
- **Compiling an agent loop** (`design/recompilation.md`): the faces are eager and pure, so the
  tick (`infer_states`, `act`, `update_empirical_prior`, and `learn` where a slot learns) is
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
  **target emission** (`H_deps` across the boundary), boundary-prior
  termination. The one-mind composite is **bitwise identical** to the
  hand-flattened declaration (the composite's arrays ARE the
  components'). Every face inherits: nesting adds namespacing, never
  inference machinery.
- Bindings name control nodes, native or the sugar's; a node driver is read at the same step (the two nodes become one), a state driver a step later. Actions speak action nodes (`a_<f>`, one per control node); a kernel `P(a | u)` in `params.actuator` is drawn by the environment.
- `subsystem(name)`: a bona-fide per-mind System (private beliefs,
  boundary neighbours as ordinary latent factors — no partner
  ontology; partners are inferred and learned like any latent).
- `driver.run()` with three tick orders, each one tick of time per round
  and every factor stepping once: simultaneous (a Jacobi sweep), sequential,
  and **async** (Gauss-Seidel sweeps in declaration or random order, each
  component's factors stepping at its turn — the statistically faithful mode
  for equilibrium/Ising-like composites; `nesting-systems.md`, tick
  semantics and its 2026-10-14 amendment).
- Dynamic action sets: the hard constraint is **`legal=`** (v0.8, below);
  `policy_prior = −inf` masking remains valid (leaf-swap, no re-jit — the
  mission executor's task-selection pattern) and E is the soft prior alone.

## Describe, replay, constrain (v0.8 — what a *composing program* needs; `composing-systems.md` is the reference)

- **Description length**: `hybrid_aif.description.mdl.description_length(document, n)`
  prices a document in
  nats as a chain of codes, a `structure` part (a universal-integer graph code over
  factors, modalities, control nodes, families, sizes and parent sets, plus each
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
  (PR #6), one compiled rollout shared by `random_policy` (PR #4), the
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
  - Faces are keyed by name: `infer_states`, `step`, `rollout` and the rest take and return
    mappings keyed by factor and modality name; positional tuples are refused.
  - `couple()` on Systems is `merge()`; `couple()` now wires participants' ports into a
    `Wiring` that `run` takes.
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
  `System.reduce` or `reduction_period`, a lagged modality (`System.lagged`).
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
  with a state mode parent keeps the mixture rule, and a noise map on a selecting channel is not
  built; an observation target on a selecting channel under `feedback="linear_quadratic"` is
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

- **Unreleased**: a `legal=` mask held as a concrete JAX array and closed over by `jax.jit`
  masks the same policies as the eager call, and the refusals of a mask that leaves no legal
  policy, or no legal level at a step, fire under `jit` as they do eagerly; a mask passed as an
  argument of the jitted function is traced and applied without them.
  Description 0.2 (the `structure`/`declarations` layout) is no longer read: `build` refuses
  such a document with a sentence naming 0.18.0, the last release that reads it, where building
  it and describing the result writes description 0.3.
- **Unreleased**: the sophisticated planner evaluates the whole deepest level unless told
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

## In the field / in flight

- **Shipped to robots**: the downstream coverage agent (EFE-driven coverage
  at tuned-heuristic parity with the hand-tuned coefficients removed;
  stage-2 learns localization noise online).
- **Integration underway**: the downstream mission executor — AIF task
  selection + recovery (retry/skip/escalate emergent from the term
  tree) against its policy sockets, using v0.5 nesting.
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

2,788 tests (2,780 in the default selection; `-m notebook` and `-m slow`
are opt-in); ruff-clean; the executed notebooks (`examples/demo.ipynb`,
`examples/objectives_a_la_carte.ipynb` — one declared T-maze under five
objectives, `examples/hide_and_seek.ipynb` — a prey and a predator coupled
through blankets, each objective swapped one line at a time) are maintained
exhibits and an adoption gate. MIT.

**Composing a new System** — constructor table, shape rules, the factor ×
modality family matrix, the canonical loop with its real return arities, and
how to read the term tree: `docs/composing-systems.md` (added 2026-09-18).
