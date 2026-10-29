# Control states as nodes

*Status: ADJUDICATED (Alex + Opus, 2026-10-03/04); not built. Direction agreed: control
states become ordinary latent nodes that a System declares it chooses (`System.controls`), with
today's pymdp-style declarations kept as sugar and continuous controls first-class. All nine decisions adopted (2026-10-04): C1 revised to a
System-level relation, C2 with path dynamics deferred to the next build, C3 with both objectives
exposed, C6 with its validation, C8 clarified. Scheduled after the R-AIF arc closes, before shared parameters
(`shared-parameters.md`), since both touch the core declaration.*

## Why

Today an action is an index, not a variable. Every transition table carries a trailing control
axis; `num_controls` gives its size per factor (1 when uncontrolled), `controlled` flags the
factors whose axis is planned over, and `continuous_controls` makes the action a float vector
read by `MLPConditional` transitions only.

**The principal reason is that a node is simply correct.** Once a control state (what the agent
intends at a step) is distinguished from an emitted action (what crosses the blanket), the
control state is an ordinary latent variable: it has a prior, it is a parent of the factors it
moves, and a belief over it is meaningful. Only the objective treats it specially, and that is
a matter of how planners use it (below), not of what it is. Correctness then pays twice:

- **Unification.** Discrete and continuous controls become one concept with two families, so a
  continuous control node is as natural as a Gaussian state; the `continuous_controls`
  exception and its restrictions dissolve.
- **New capabilities.** A belief over a control node can be pushed through the transitions, so
  expectations over actions become available wherever the quantity is linear in that belief:
  the predictive state under the policy prior, risk-like terms, the linear-quadratic regime.

Three further things argue against keeping the index as the core representation:

- **The roadmap needs control nodes.** `nesting-systems.md` charters planning as inference as a
  planner backend, "π as a latent whose priors absorb G", reifying the policy variable; the
  brief names the epistemic-priors formulation (arXiv:2504.14898), which turns the epistemic
  term into factors so that planning is pure inference and samplable end to end. An index into
  a table cannot receive a message.
- **The library already does it at its boundaries.** `merge()`'s `ActionControl` rewrites a
  bound control node into an ordinary `B_deps` parent, and `BoundaryPrior` with a distribution
  averages over the control axis — a belief pushed through a control node. The core is the odd
  one out.
- **The declaration is redundant.** `num_controls` restates a table's shape, and `controlled`
  restates "has a control parent", except where it means something else (a continuous control
  of size 1; a blanket port).

## Three objects, kept apart

1. **The control state** — an internal latent: what the agent intends to do at a step.
2. **The policy** — the control states over the horizon: what a planner chooses among.
3. **The emitted action** — the active state at the blanket: a draw (or mode) from the control
   posterior, sent out like the dual of an observation. Ports belong here and only here.

The index representation ran (1) and (3) together. Making (1) a node does not touch (3): an
emitted action stays a port value.

## What is genuinely special about a control node

Not its representation — its role in the objective. Expected free energy is not linear in the
belief over controls: `G` is evaluated with the control *clamped* ("do u"), per policy, and
combined through a softmax. Pushing a mixture over controls through the transitions gives the
predictive under the policy prior, and an entropy of that mixture is not the mixture of
entropies: info gain would credit uncertainty about which policy is followed as information
about the world (the same "whose information" error as zeroth-harness's switching brief). So:

- **Planners clamp.** Enumeration, the sophisticated tree and CEM set control nodes to deltas
  (one-hots, or points for continuous controls) and score each clamp. A clamped node is exactly
  today's index; with a static "clamped" flag the fast path indexes rather than contracts, so
  the maths and the cost are unchanged.
- **Expectations that are linear in the control belief may use distributions**: the predictive
  state under the policy prior, risk-like and occupancy terms, a `BoundaryPrior`'s averaging.
- **Inference backends pass messages** into control nodes. Plain variational free energy on
  control nodes loses information seeking; that is what epistemic priors restore, and why that
  formulation is the chartered backend rather than naive message passing.

## Continuous controls, first-class

A continuous control is a control node with a continuous family, not an exception:

- **Families.** A Gaussian node, optionally bounded; a point mass when clamped.
- **Transitions with a control parent.** Tabular conditionals take a discrete control parent as
  today. **Linear-Gaussian transitions gain a control input**, `x' = F x + G u + b + ε`, which is
  conjugate: a clamped `u` is an offset, and a Gaussian belief over `u` pushes through exactly
  (the linear-quadratic regime, today unreachable because continuous controls require an
  `MLPConditional`). `MLPConditional` transitions keep reading `u` as an input.
- **Planners.** CEM and MPPI propose clamps from a distribution over the control node; a
  gradient planner differentiates through a clamp; enumeration applies only to discrete
  controls. Mixed discrete and continuous action spaces become a planner question (enumerate
  the discrete nodes, optimise the continuous ones per branch), not a declaration error.
- **Bounds.** A continuous control's legal values are a box: a lower and upper bound per
  dimension, which is the node's support. Today's `ContinuousControl` already carries per-dimension
  bounds that CEM clips to; they become sugar for this. More general legal regions (a subspace, a
  polytope) are a later extension. `controls.soft_clip` stays where a transition eats the action.

## Timing: when a control is read

*Added 2026-10-05 from zeroth-harness's brief "One control, read by more than one factor"
(`zeroth-harness/docs/briefs/hybrid-aif-shared-controls.md`).* A transition's **control-node
parents are read at the step the control is chosen**: a control node is an input to the
transition from t to t+1, so a factor that lists `help` among its parents moves at t+1 on a call
made at t. **State parents are read from the previous step, as today**, so a factor that reads a
*state* echoing the control (a `help` state whose table copies the control) still sees it a step
later. Both semantics coexist in one model, per transition — the brief's requirement — and they
fall out of the distinction between a control node and a state that copies it. Today a control
belongs to one factor's trailing axis, so a second factor can only read it through such a state,
a step late; pymdp's JAX agent reaches the same-step read with `B_action_dependencies`,
flattening several controls into one combined axis per factor at construction.

Consequences the brief asked about:

- **Where the code assumes one control per factor** (the build's checklist): the compiled
  transition's slice at `action[f]`; Dirichlet transition learning per slice; policy enumeration
  and the `(H, F)` policy array; `act`'s control marginal; the sophisticated planner's
  branching; `legal=` level masks; `continuous_controls` and CEM; `describe` and the schema;
  replay's action columns; `ActionControl`.
- **`ActionControl` offers both semantics**: binding a control node (same-step) or a state that
  echoes it (the one-tick relay); the node named decides, there is no default (C8), and existing
  composites, which name states, are unchanged.
- **Prices over controls need no echo state**: a target on a control node contributes, under a
  clamped policy, `KL(δ_u ‖ C) = −log C(u)` — a per-control price. An echo state is kept only
  where a lagged read is wanted.

## Decisions to walk

**C1 — The representation.** *ADOPTED (Alex, 2026-10-04), revised the same day.* Control
states are **ordinary nodes** in the factor table, with no kind or flag of their own; `B_deps`,
names, the key grammar, `merge()` namespacing and description apply unchanged. Being a control
state is a **relation between a System and a node** — the System sets its value by decision
rather than by inference — so it is declared once on the System: `System.controls`, the set of
nodes it chooses. Several parts of the machinery read that relation, which is why it lives on
the System and not in a planner's configuration: the planner (chooses them), the filter (treats
an emitted step's control as known, an efference copy), the blanket (emits actions drawn from them),
learning (transition learning needs the action taken), and legality and `E` (defined over them).
A node the System chooses without the planner (a habit net, a fixed random policy) is still a
control state. Composition rewrites the set: a node `merge()` binds to another System's
latent leaves the composite's `controls` and becomes an ordinary driven parent (`ActionControl`
already does this). Cf. Friston et al. (2024, *Supervised structure learning*, Biological
Psychology 193:108891; [doi:10.1016/j.biopsycho.2024.108891](https://doi.org/10.1016/j.biopsycho.2024.108891)): a control variable is "a path
that may or may not be controllable".

**C2 — The policy object.** *ADOPTED (Alex, 2026-10-04); path dynamics deferred to the next
build.* a policy is an assignment to the
control nodes over the horizon (integer, float, or a structured pair when mixed); a control
node may parent several factors (a shared actuator); enumeration runs over control-node levels,
pymdp's ordering kept for sugar-declared models. Read against Friston et al. (2024), where each
factor has latent *paths* and "a policy is not a sequence of actions but simply a combination of
paths", three layers fall out: (1) per step, a policy is a joint assignment of control nodes —
no separate policy variable is enforced; (2) across steps, control nodes may carry their own
transition prior, as paths do (persistence and habit, distinct from `E`); (3) an abstract policy
that unrolls is a higher-level factor whose state parents the control nodes, available, never
required. A control node *is* a controllable path, so "dynamics over control states" and
"dynamics over paths" are one thing: a transition prior from the previous control (the paper's
C, with `E` its empirical prior). What stays planner-side is only the candidate being scored —
an assignment to the control nodes, per step or over a horizon — which is not a separate node.
The control nodes' prior and dynamics *do* enter policy selection: a candidate is scored by −G
plus its log prior under them. Planners score clamped *sequences* because enumeration and the
tree need the joint over a sequence while message passing yields per-step marginals; the two
meet in C7's backend. (The paper re-selects a combination of paths every step and gets
persistence from the path dynamics; scoring whole sequences with the same prior rewards the
same thing.)
*Notation, against the paper.* No field is renamed or added. The paper's tensors map onto
existing slots: A → `likelihood`; B → `transition`; **C (transitions among paths) → the control
node's own `transition` entry**, its previous control as parent; c (preferences over outcomes) →
`obs_targets` / `state_targets` / `terminal_targets`; D → `initial_state_prior`; E (prior over
paths or policies) → `policy_prior`, which keeps sequence-level habits and adds, as a log prior,
to the per-step prior a control node's transition supplies. Beware the vocabulary: some of our
docs say "C designations" for preferences (pymdp's sense), where the paper's C is path dynamics.

*Resolved for now (Alex, 2026-10-04):* control nodes may carry a transition prior from the
previous control, as paths do; independence at every step (today's behaviour) is the special
case whose rows all equal `E`, so declaring dynamics costs nothing by default and adds one
log-prior term per step to a policy's score. The momentum reading: a path selects which
transition applies, so a persistent path keeps the state moving the same way, and action is "a
force that changes the path". Path dynamics are the next build, not this one.

**C3 — Clamp and belief semantics in the faces.** *ADOPTED (Alex, 2026-10-04).* Planners clamp, with a static clamped flag selecting the indexing fast path;
`predict_beliefs` and the transition faces accept a clamp or a belief over a control node. A
term is *linear* in the control belief when its value under a mixture of controls equals the
mixture of its values: utility, ambiguity, occupancy and the predicted state are; risk (it
contains `−H[q(o)]`), info gain, entropy lowering and empowerment are not. The line runs between
expectation terms and entropy terms, not between EFE and its variants. A nonlinear term
evaluated at a mixture is not an error but a different quantity: the objective of committing to
a *randomised* policy, against the expected objective under the policy posterior. Both are to be
exposed, named by what they mean. Pushing a belief through a transition is never forbidden at the
face; the restriction lives at the planner level: classical explicit-search planning scores
clamped policies and never presents a pushed-through mixture as one.

*Phase 2 spelling (Alex, 2026-10-06):* no syntax for the randomised-policy objective. `score_policies`
accepts a belief-valued policy (a `Categorical` per control node per step in place of an integer),
dispatched statically by type as the transition faces are, and computes every leaf under it; the
docstring names the quantity. The expected objective under the policy posterior is the
`q_pi`-weighted `total` of the clamped tree, stated in the docstring, not a function. `infer_policies`
and `act` refuse a belief-valued policy with a sentence. The frame: not disallowed, not easy to do by
accident, no feature unless useful. Control-node priors stay out of Phase 2 (C2: the path-dynamics
build); a node prior in params keeps refusing with a sentence naming that build.

*Prices and selectors (Alex, 2026-10-06):* a target on a control node is a state target like any
other (the target is `C`, what the agent wants; a "price", in conversation, is the risk the agent pays
for diverging from it, never the target itself; code says target and risk): `state_targets` and `terminal_targets` accept a node, and the leaf is plain `risk` keyed by
the node's name (KL(δ(u) ‖ C) = −log C(u) under a clamp, KL(q(u) ‖ C) under a belief-valued
policy); no new term, no new sign. Risk is toggled per variable by the presence of its target, so
no selector is needed; `names.controls` is the set for pricing every node. A control node named in
`epistemic` keeps refusing: a chosen node carries no uncertainty to resolve until path dynamics
give it a prior. The feature is advertised at 0.12.0 with the harness's `help` prices as the
example.

**C4 — Continuous control nodes.** Families (Gaussian, bounded Gaussian), the LG control input
`G u`, `MLPConditional` unchanged, planners as above. *ADOPTED (Alex, 2026-10-04)*, in the
same round, with the linear-quadratic case as the acceptance demo.

*Where the box lives (Alex, 2026-10-08):* no `control_boxes` on the System. A control node is
continuous when its readers read it as a float input (an `MLPConditional`, the LG `G u` term);
tabular readers of the same node are refused, as disagreeing level counts are. The box is a
params entry keyed by the node, the node's support. The action face carries a float vector under
the action node's name. Costs found in the build
are reported, not absorbed. *Amended (Alex, 2026-10-08):* `continuous_controls=` is dropped, as
`num_controls` was: the System cannot write params and holds nothing for the keyword to desugar
into; a continuous control is declared by a float-reading transition and its box in params.
*Amended (Alex, 2026-10-08):* the entry is a `Box(low, high)`, bounds only, and a node's kind is
decided by the presence of its box (an LG reader with `G` and a neural reader can take either kind,
so their shape alone cannot settle it); readers that cannot read that kind, or disagree on its
width, are refused. `TruncatedGaussian` is a family, not a support: it is what beliefs, prices and
the gradient planner's posterior on a continuous node are, and the natural type of a node's prior
when the path-dynamics build arrives, in a slot of its own.

*Bounded Gaussian (Alex, 2026-10-08):* the truncated Gaussian, the Gaussian density restricted to
the node's box and renormalised, implemented in log space. Diagonal covariance in this phase: the
box then factorises into intervals and the normaliser, moments, entropy and KL on a shared box are
closed per dimension; a full covariance on a box needs a numerical normaliser and is a later ruling
if a model needs correlated bounded controls. A clipped Gaussian is not a density and a squashed one
has no closed moments or entropy; CEM's clipping stays as planner mechanics.

*Discrete nodes and the LG factor (Alex, 2026-10-08):* one path. A discrete control node enters an
LG factor as a mode parent, the existing SLDS form `x' = F_j x + b_j + ε_j` with the level in `j`;
a continuous control node enters as `G u`, with `G` per mode or shared on the same rule as `F`. A
Gaussian belief on a continuous node pushes through `G` exactly; a truncated one by moment matching
through its closed first two moments, named as an approximation. A belief on a discrete mode parent
of an LG factor is a Gaussian mixture and stays refused, with a sentence naming the mixture-bounds
work. Not adopted: a one-hot into `G` as a second path for discrete nodes. Its content, an
additive control effect shared across modes (`b_{j,k} = b_j + g_k`, learned jointly with `F`), is a
tied parameterisation of the mode table and belongs in the learnable slot if a model needs it; its
closed form for a belief on a discrete node is a moment-matched collapse of the mixture, which is a
ruling about collapse, not about wiring.

*Readers beyond tables (Alex, 2026-10-08):* each reader reads a control node in its own way, and
the readers of one node must agree on its kind. A tabular transition indexes its control axis by
the level; an LG transition reads a level as a mode index and a float vector as the input to
`G u`; an `MLPConditional` reads a float vector as part of its input. So a discrete node shared
with an LG factor, a native node read by an LG factor, a continuous node read by several factors,
and a mixed discrete-and-continuous action space are all implemented for tabular and LG readers
(the mixed space as structure; its planner is a separate ruling). An `MLPConditional` reads a
discrete control node as a one-hot wherever the node sits in the wiring, as it already read its own
control; the one-hot followed by the first weight matrix is a learned embedding, so nothing is
missing on the neural reader. *(Amended, Alex, 2026-10-08: an earlier version refused a discrete
node on this reader, on a premise that was wrong.)*

*The planner for the LQ demo (Alex, 2026-10-08):* a gradient planner joins the registry: it descends
`score_policies` in the float plan with projection onto the box, the mode of `q(π) ∝ exp(−G)` over
continuous plans, as enumeration is its exact form over a finite set and CEM its sampling form. The
demo plans with it, runs CEM beside it as a cross-check, and is accepted against the Riccati
solution at tight tolerance (on the LG path the epistemic term is constant, so the gradient is
pragmatic by theorem). Whether the planner emits the mode alone or the Laplace Gaussian from the
curvature at the mode (C5: the action is a draw from the node's posterior) is settled in the build
brief. A mixed discrete-and-continuous space is accepted as structure (Q15) and a planner asked to
search one refuses with a sentence until a use needs the product search.

**C5 — The blanket.** The emitted action is a draw from the control node's posterior at the
current step (the act face's sampling, `act_sampling`'s rule), or its mode; ports attach to
the action nodes; `ActionControl` and `BoundaryPrior` retarget from a factor's control
axis to a control node. *ADOPTED (Alex, 2026-10-04)*; the one-tick relay through a bound state stays.

**C6 — Sugar.** `num_controls`, `controlled` and `continuous_controls` remain as declarations
that create one control node per controlled factor, placed as that factor's last transition
parent, so tables keep their shape and pymdp's layout. *ADOPTED (Alex, 2026-10-04)*; every
existing model must run unchanged and bitwise equal through the sugar. Inconsistent explicit
declarations (a factor given both sugar and an explicit control parent, or `num_controls`
disagreeing with the node's levels) are validation errors.
*Amended (Alex, 2026-10-05): one canonical form.* The text above read "every existing model
must run unchanged and bitwise equal through the sugar", which the build plan could only meet
with *virtual* nodes, absent from the factor tuples, giving one model two encodings and breaking
`build(describe(s)) == s` for sugar models. Instead the sugar is translated once, at build time,
into real control nodes appended after the declared factors (one per controlled factor, in
factor order). The built System is exactly the one a native declaration of those nodes gives:
existing factor indices, tables and numerics are preserved bitwise, while belief tuples and
documents gain the nodes. Native wiring is `B_deps`; a control node may sit at any parent
position. `control_parents` therefore leaves description 0.3, and pymdp compatibility lives only
at the input edge (constructor declarations and the 0.2 reader). *Names (Alex, 2026-10-05):* the
node for factor `f` is `u_<f>` (`u_factor_2` when names are generated); a declared name that
clashes is a validation error, and there is no override: to choose the name, declare the node
natively.
*Amended again (Alex, 2026-10-07): `num_controls` dropped; the rule for sugar.* Under C11 the
System holds no sizes, so `num_controls` asserted a level count that nothing could check (at HEAD
the LG, MLP and shared-control paths raised on a disagreement; after C11 the table silently won).
A count with no referee is dropped rather than carried: the sugar is `controlled`,
`continuous_controls` and `B_control_deps`; a controlled factor's levels are its table's trailing
axis, as pymdp infers them when `num_controls` is omitted; the 0.2 reader checks a document's
`num_controls` against its arrays and discards it, and `describe` writes it from the tables. The
rule that decides this and any later case: a compatibility spelling is kept iff it desugars
losslessly into the canonical form at the input edge and everything it asserts is either carried
as canonical structure or checked there; a spelling that asserts what the canonical form does not
hold is dropped. `controlled` and `B_control_deps` (wiring), `continuous_controls` (the box,
carried as `control_boxes` until Phase 3), the trailing/pad axis layout (checked by shape) and the
A/B/C/D/E aliases pass; nothing else pymdp-shaped is held.

**C7 — The inference backend.** Epistemic-priors planning as inference is a planner-registry
entry operating on control nodes; out of scope for this build, but nothing in it may assume a
control is an index (no index-only shortcuts outside the clamped fast path).
*ADOPTED (Alex, 2026-10-04)*: fenced as the next round, with a test that a control node accepts
and emits messages.

*The test (Alex, 2026-10-07):* discharged by the Phase 2 faces, no new API. A non-degenerate
distribution on a control node (C3's belief-valued policy) changes what `score_policies` scores and
comes back out under the node's name in the emitted beliefs; the clamp is the one-hot case.
`InferenceDetail` carries no node entry: nothing observes a node, so an entry would be constant. The test is named for C7 so the generic message-passing layer (project-brief,
deferred) finds the contract it must keep passing.

**C8 — Legality.** `legal=` masks become masks over control-node levels or policies, and the
"no continuous analogue for now" of `action-legality.md` is met by C4's bounds.
*ADOPTED (Alex, 2026-10-04)*; L1–L6 otherwise unchanged. For continuous control nodes legality
is the node's *support* (bounds, a box): illegal values carry zero density, the continuous form
that `action-legality.md` deferred.

*Legality by node (Alex, 2026-10-07):* `legal=` on `act`, `infer_policies`, `replay` and the driver's `agent_respond` is a mapping keyed by control-node name, each entry a boolean mask over that node's levels; a node absent from the mapping is unrestricted. The bare `(P,)` mask over policies is unchanged. The per-state-factor tuple is refused with a sentence naming the keyed form. Unknown keys are refused. For continuous nodes legality is the box (Phase 3). This follows the C10 rule: a keyed mapping that supplies values is complete, one that restricts or selects is partial, and both refuse unknown keys.

*Where a mask applies (Alex, 2026-10-08):* legality masks the choice distribution at every point
where a choice is formed, and never re-prices. For enumeration the only choice point is the policy
softmax, where `log 𝟙[legal(π)]` added before the softmax equals restricting the set first, since
each policy's score is its own; the posterior over the legal set is the conditional given legality
(SPM's `E`, pymdp's restricted policy set, and invalid-action masking in reinforcement learning are
the same operation). Sophisticated search has a choice point at every depth: a first-step mask is
exact at the root; a per-step mask is refused there with a sentence, and would have to be threaded
into each depth's softmax if a schedule under sophisticated planning is ever needed.

*A bound node's clock (Alex, 2026-10-08):* no relay mode and no default; the node a binding names
decides the timing. `merge()` binding System B's control node to A's control node `A.u_<f>` is
same-step: A's decision at t clamps B's node at t, B's factors reading it move from t to t+1 on
the same composite call, and B's node is identified with A's: the composite has one node, A's,
whose entry is A's one-hot, and B's name for it is gone (a mapping keyed by it is refused naming
A's node; no alias). A separate mind for B, under the driver, does not receive A's decision: its
view marginalises A's node under a partner prior, since A's decision reaches B only through the
world (nesting-systems, principle 6). Binding B's node to a state of A is the relay: the state is
read from the previous step, so B sees A's decision a tick later; an echo state is an ordinary
state whose table copies a control node, declared by whoever wants the lag. A binding to a node
that is neither a control node nor a state of the driver is refused naming both spellings.
Legality across a merge: a mask on the node is A's constraint, the only choice point it has, keyed
by A's name; a mask keyed by B's former name is refused naming A's node (the mask-at-every-choice-
point rule across the merge).

**C12 — The active state is a node; `couple()` was a merge.** *ADOPTED (Alex, 2026-10-09).*
The design keeps three objects apart (control state, policy, emitted action), but the code attached
ports to control nodes and gave the emitted action no node: observations are modality nodes with a
likelihood, clamped by data; actions were only values from `act`, keyed by control-node names,
consumed by `as_env`, the driver and the traces. So `ActionControl` binding B's control node to
A's was an identification of two latents, a model merge within one mind, not a blanket coupling;
the relay and `ObservationLink` are merges too (a latent System is a complex latent factor). The
ruling: (1) an **action node** per control node, the dual of a modality: a child of the control node
with a kernel `P(a | u)` that is a delta unless declared otherwise (a stochastic actuator is the
dual of soft evidence, and a belief pushes through it), named as the actuator channel. The
efference copy is the agent's prediction: the control node is clamped to the decision and the
action node carries `P(a | u)` at it; the realised action is process-side, drawn by the actuator,
consumed by the world and recorded beside the intended one, and the agent learns of it only
through a modality whose parent is the action node. Transitions read the control node: `u`'s
content is its role in `B`, and a rollout conditions on the action as taken; an agent that models
its effector as something it can feel fail declares it as a state factor (Alex, 2026-10-09). The
action face, traces, the world and the wires speak action-node names, never control-node names. (2) The pairs are tied: one action node per control node, as a
modality is the sensor channel's type; what is tied is the structural identity of channels by
name, never the kernels, where an agent may be wrong about its senses and about itself alike.
The decoupled cases (a modality that is no sensor, a control with no actuator) are standing
absences on the existing faces (`present=False`, an action nobody consumes); *open question (Alex):*
whether a standing mask is the canonical way to say "this agent believes it has a sense it
lacks", or whether that deserves a round of its own; no pressing need now. (3) `couple()` is
renamed `merge()` everywhere: it identifies latents across components into one model. `couple()`
is the name for coupling through blankets: a wire from one System's action node to another's
observation node, typed, with no latent shared (the build plan's 4b, now inside the arc). (4) A
thorough audit of design docs, user docs and docstrings purges the conflation of control states
with actions, and code resting on it goes unless a feature stands on its own.

**C13 — Coupling through blankets.** *ADOPTED (Alex, 2026-10-09).* A run is a set of
participants wired together: **minds** (a System with a respond function, planning or not) that
take observations on their modalities and emit actions on their action nodes, and **processes**
(an `Env`: a System run forward by `as_env`, or a hand-written sampler) that take actions on
their inputs and emit observations. `Env` is the process role, not a rival to `System`. A
**wire** is an identity between an out-port and an in-port of the same type (level count to
cardinality, width to dimension, same kind), refused otherwise: action node to process input,
process observation to modality, or action node straight to another mind's modality. A wire
carries no kernel; anything that transforms, including an "external likelihood"
`P(o_A | a_B, a_C)`, is a System with no planner run as a process, with the same one-tick lag as a
direct wire. An action node with an actuator kernel is realised **once per tick, at the node**,
and every wire from it carries that value; a process reached by a wire draws nothing. An in-port
fed by nothing is refused with its fix named (a constant for a process input; a mind modality
must be wired or dropped from the mind, the standing-absence form being the open question above);
an out-port read by nothing is ignored. The record is one entry per participant, keyed by the
name given at `couple()`: a mind's observations as received, intended and realised actions, and
carry; a process's states and emitted observations. `run` on a merged composite stays as it is,
the one-process case with wires by name, and its record is unchanged.

*Reading (2026-10-11):* how C11 and C13 correspond to the predictive-coding account of motor
control, where the actuator kernel is the reflex arc and a wired action node is proprioception,
is recorded in `predictive-coding-reading.md`; it rules nothing and lists the roadmap items it
points to.

**C9 — Migration and version.** Internal API changes (policy arrays, replay's action columns,
description records, the R-AIF reference's continuous controls via sugar) are breaking for code
that reads them; declarations are not. *ADOPTED (Alex, 2026-10-04)*: a minor
version, with the sugar making every existing declaration valid and a migration note for readers
of policy arrays. The model-description format changes in the same revision (its state
records lose `controlled`/`num_controls` to `System.controls`), and two observation-record fixes
ride it (Alex, 2026-10-05, from reading zeroth-harness's serialised models): `observes` is renamed
`parents` — it is `A_deps`, the states the modality depends on, not what is observed, and the
state records already say `transition_parents` — with `observes` accepted as an alias on read for
one version; and `source` is written only for an observation `merge()` bound, `"external"` being
the default it already is on read (the schema does not require it and `build` never reads it).

**C10 — Names at the faces, order inside.** *ADOPTED (Alex, 2026-10-05).* Every public face
takes and returns beliefs and actions as mappings keyed by node name (`act` returns
`{"a_help": 2}`; `infer_states` returns `{"who": …, …}`); unnamed models use the generated names.
Positions exist only where tensors are needed (tables, the planner's `(P, H, K)` policy tensor,
scan carries), in the one order the System defines: node index, with an action's entries
following `controls`. Positional data enters or leaves only through an explicit converter on the
System (`to_positional` / `from_positional`, names provisional), which also serves pymdp-shaped
input and the migration of recorded runs; the faces do not accept tuples, since a misordered
tuple is silently misassigned where a misnamed key raises. A mixed discrete/continuous action is
just a mapping whose values differ in dtype and shape, so no pair type is needed. The conversion
happens at trace time: a jitted check (2026-10-05) gave identical compiled op counts for the dict
and tuple forms, differing only in debug metadata and parameter order, and per-call overhead
within 2%. Ships in 0.12.0 with C9's migration note; built as Phase 1's first step, before control
nodes.
*Extended (Alex, 2026-10-05):* observations too — every face that takes or returns them
(`infer_states`, `learn`, `as_env`, rollouts, replay) keys them by modality name, so beliefs,
actions and observations share one scheme at the boundary. Until control nodes exist, an
action is already keyed by its node's eventual name (`u_<f>`), so consumers migrate once.
*(C12, 2026-10-09: the action face is keyed by the action node, `a_<f>`, so consumers migrate
once more.)*
*Names, one source (Alex, 2026-10-06):* the System's name tuples (`factor_names`,
`modality_names`, filled with `factor_{i}` / `modality_{m}` at construction when not declared) are
the single source of names and of node order; keys in `SystemParams`, targets and rate maps are
references to them, validated at every face, and `System.names` is the read-only view.
*Extended again (Alex, 2026-10-06):* the per-variable containers of `SystemParams`
(`likelihood`, `transition`, `initial_state_prior`, …) and of an environment's parameters are keyed by
name too (`params.likelihood["position"]`); the tables inside stay arrays. Conditional on no runtime
cost, which the Phase 1 gate (bitwise baseline, equal compiled graphs) is the proof of. Checkpoints
serialised from positional containers do not load (dict leaves flatten in sorted-name order); accepted,
as no significant project predates the change, and noted in C9's migration note. The public
environment classes keep the name `GenerativeEnv` (an `Env` is a practical concession to the
environment pattern of AI research; everything is a `System`): it is keyed, and also wraps an outside
positional (pymdp-style) environment given a System. The positional processes are private. The
public split between `GenerativeEnv` and `HybridGenerativeEnv` is consolidated into one class if the
build finds no reason beyond history to keep it (the hybrid sibling was added beside the pymdp port
to leave the discrete path bitwise untouched).

*Presence, and the name of a flag (Alex, 2026-10-07):* `infer_states(mask=)` and `replay(mask=)` become `present=`: a mapping keyed by modality, each entry a bool (or a traced flag; a `(T,)` array on replay) saying whether that modality is observed; an absent key means present, `None` means all present, unknown keys are refused. The rule behind the rename: a boolean flag is named for what `True` means, so `present`, `legal` and `valid` all read the same way, and `mask` is dropped from the public faces because it does not say which way it points.

*Complete and partial mappings (Alex, 2026-10-07):* Keyed mappings that supply values (beliefs, observations, actions, params) are complete: a missing key raises. Keyed mappings that restrict or select (targets, `epistemic`, `legal`) are partial: an absent key means unrestricted or unselected. Both refuse unknown keys.

**C11 — No sizes on the System; the policy set is a method.** *ADOPTED (Alex, 2026-10-06).*
A System is the graph: wiring, names, which nodes it chooses. Variable shapes are a property of
the data that flows through the graph and live with the arrays (D8 made the same cut for the
document). So the control-states build adds no static size field: a control node's levels are
read from its readers' tables (readers must agree; a node nothing reads is refused until Phase 2
gives it a prior table), and the continuous box stays in the `continuous_controls` declaration
until Phase 3 moves it into the node's family. `num_controls` was the one static size, kept for a
params-free `System.policies()`; that becomes `System.policies(params)`, by the rule that a
*method* is what the System is or does given its params (`initial_priors`, `predict_beliefs`,
`act`) and a *free function* is a procedure over the pair (`imagine`, `replay`, `describe`,
`merge`). `planning.enumerate_policies` stays the column primitive underneath.

*Recorded, not decided (Alex, 2026-10-06):* declarations by name (`B_deps` keyed by factor name)
would be a better idiom once 0.3 writes parents by name, but the considerations that compelled
C10 — per-call data, misassigned silently, unchecked — do not apply to a one-time declaration
that table shapes mostly check. A candidate to weigh with Phase 5, not a decision. Composite
control nodes are namespaced like any node (`world.u_location`, C1), not `u_world.location`;
`names` holds `states`, `controls` (the chosen nodes) and
`modalities`.

## Acceptance (draft)

- Every existing test passes unchanged through the sugar, and policy matrices, term trees and
  planner outputs are bitwise equal to today's on the T-maze, the hybrid demos and the compiled
  path.
- A shared control node parenting two factors plans correctly against an enumeration oracle.
- A linear-quadratic model with a continuous control node: exact prediction under a Gaussian
  control belief, and a planner matching the closed-form LQR action.
- A mixed discrete/continuous action space plans (enumerate discrete, optimise continuous).
- Merges bind control nodes; `ActionControl` and `BoundaryPrior` reproduce today's composites.
- A control node accepts and emits messages (C7's fence).
- Timing (the harness's acceptance): a factor with a control-node parent moves at t+1 on a
  control taken at t, on a two-factor toy; a factor reading an echo state keeps the relay, in the
  same model; a Dirichlet transition sliced by a shared control learns per slice; a document
  that declares no shared control behaves bit for bit as today.

## Out of scope

The epistemic-priors backend itself; multi-agent message passing beyond today's bindings;
learning the control prior beyond what `E` already does.

## Build notes (interim `B_control_deps`)

*2026-10-05.* `System.B_control_deps` is the shared-control piece of C6's sugar, built ahead of
control nodes for zeroth-harness's same-step read. Entry f lists the controlled factors whose
control slices factor f's transition from t to t+1; the table carries one trailing control axis
per entry, in order (an empty entry keeps the single size-1 axis, so every existing table shape
stays valid), and `None` is today's declaration, bitwise unchanged. Policies, `act`, `legal=` and
replay are untouched: a policy is still one level per controlled factor. Supported on tabular
transitions (compiled and hybrid categorical steps, Dirichlet learning per slice, the generative
process, transition novelty and occupancy) and in the description (`control_parents`, written only
when declared); a continuous control outside its own default entry, non-default entries on
non-tabular transitions or the amortized path, binding or terminating a shared control node in `merge()`,
and `subsystem()` views of a composite that declares it raise. Axes follow the declared order;
with equal sizes a misordered table cannot be detected. After the full build it desugars directly: the control node created for each
controlled factor c is a transition parent of every factor listing c (c itself, under the
default), in the listed position, so the table layouts carry over unchanged.
