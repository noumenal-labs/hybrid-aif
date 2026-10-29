# Nesting Systems — design requirements

*Status: direction confirmed (Alex + review, 2026-07-23); revised after
the factor–System exchange (2026-07-24); committed 7f36b6f. BUILD
ACTIVE as of 2026-08-17 — pulled forward by an application need
(amortization PAUSED at its P1-infra/P1′ seam, see amortization.md).
Unifies Alex's a-priori requirements draft
with the in-session review; builds on `system.md` §"Coupling (v0.1
scope)" (blanket-typed port pairing — the vocabulary this makes real).
The sequencing note at the bottom is historical: parameter learning
(v0.3) and the per-variable epistemic API (v0.4) both landed first,
as it prescribed.*

## The requirements (a priori)

1. **A `System` is implicitly open** — it produces actions and consumes
   observations for other systems; openness = unbound blanket ports
   (action nodes, observation modalities; `as_env` flips them).

2. **A merge of Systems is itself a System**, whose ports are the
   components' unbound ports minus the bound pairs. This dissolves the
   open/closed type question: **closure is a property (zero unbound
   ports), not a type**. Composition is closed under `System`; there is
   exactly one composite concept and no special cases anywhere in the
   signatures.

3. **The Ising stress test (the load-bearing constraint).** A System may
   couple to many partners with overlapping neighborhoods (each node of
   an Ising-like system coupled to its neighbors, for example). There is no tractable
   representation collapsing each node's "partner" into a single
   aggregate System — the would-be partner is the full joint
   marginalized down: intractable, and different for every node.
   Consequence: **coupling composes by flattening into a larger factored
   System, never by black-boxing into opaque sub-agents.** Since a
   System already is a graph of factors with cross-factor
   `B_deps`/`A_deps`, nesting is a namespaced graph-merge, and the Ising
   system is simply an N-factor System that was never aggregated.

4. **Closure via boundary priors.** Any control node may be bound to another
   System's node, or terminated by a boundary prior — a delta (the
   scoping's `data-clamp` factor tag) or a flat distribution ("no one is
   steering") being the two extremes; neither is canonical, and the
   choice is functionally real: a delta on a control node pins the
   dynamics, a uniform marginalizes over them — different composites,
   different rollouts. A composite with zero unbound ports is
   **bounded**: runnable with no external feed. (A derivable predicate,
   never a type; "isolated" — exchange-free boundaries — is at most an
   optimization remark about dead ports.)

5. **Time alignment.** Coupling requires shared tick semantics: whose
   transition fires when. The within-step declaration-order rule
   generalizes to a cross-System order; v1 synchronous. The
   asynchronous case is already chartered by the edge/near-edge/cloud
   lattice (loops exchanging serializable sufficient statistics).

6. **No telepathy, no telekinesis.** Couplings connect blanket ports
   only — wires from action nodes to observation nodes. Reads: one
   System's posterior enters another only through observations.
   Writes: nothing may rewrite another System's internal state
   directly — all influence enters as evidence or dynamics through the
   blanket, and internal-state updates are always posterior inference
   or the system's own transition.
   (Implied by `system.md`'s coupling-blind-to-planning rationale;
   promoted to a stated invariant — it is what keeps multi-agent
   inference honest. The experimenter's `reset(state=...)` override is
   out-of-model scaffolding, not an in-model operation.)
   *2026-10-09 (C12): scope is coupling; a `merge()` identifies latents.*

7. **Faces survive flattening.** After composition it must remain
   possible to run `infer_states`/`act` *for a sub-System* — its factor
   subset plus its blanket — against the rest-as-env. Sub-System
   identity persists as metadata (name prefixes + port bookkeeping),
   not runtime structure; without this, nesting kills the à-la-carte
   principle.

8. **Static composition.** Coupling happens at declaration time in
   Python, yielding one static composite (shapes, deps, names). No
   dynamic re-wiring inside jit.

## The factor–System equivalence, and its exchange rate

A factor graph is closed under black-boxing: any subgraph with its
boundary variables is, semantically, one factor, and factor interfaces
are Markov blankets. So `System` and `Factor` are interchangeable *at
the interface level* — one physics-flavored, one modeling-flavored —
and under the 1:1 reading "planner-per-factor" and "each mind is a
bona-fide System" are the same statement. The design hinges on where
the equivalence stops being free:

- The code's `Factor` declarations (tables, LG heads) are *memoryless
  conditionals*. A black-boxed System is a *process*: marginalize its
  internal states and its input→output kernel becomes
  history-dependent. To use a System as a factor you therefore either
  **flatten** — expose its internal state as ordinary variables; keep
  Markov structure, pay dimensionality; the `merge` builder — or
  accept a **memory kernel** — keep size, pay non-Markovianity and
  approximation; the amortized/distilled backend already on the
  roadmap. Both composition modes are legitimate; v1 builds the first,
  and the Ising constraint (req. 3) is the statement that the first is
  the default.
- Opening or closing the box is a *view*, not a type promotion: a
  Factor is a System viewed opaquely; a System is a Factor viewed
  transparently. The type system needs exactly one of them.
- The equivalence is ultimately a **timescale statement**: a pure
  function is memory too — its static structure — and "state" is just
  structure that changes faster. A System-as-process is a trajectory of
  time-indexed Factors; flattening is the parsimonious refactoring that
  separates what is static from what flows. The library already carries
  this ladder explicitly: data-clamp (frozen) → parameters (slow,
  learnable, priors-on-parameters) → states (fast). A `Factor` is the
  frozen limit, not a different kind of thing.

**What still distinguishes intra- from inter-System machinery after
flattening?** Two answers at two levels, and they differ:

- *World-side (the generative variables): nothing intrinsic.* Once
  merged, the flat graph does not remember
  which edges crossed an old boundary; a boundary survives only as (i)
  the `subsystems` **metadata** — a remembered ontology — and (ii) a
  *statistical* signature (sparse separators / blanket structure) that
  could in principle be re-detected but not uniquely and not cheaply.
  This is exactly why requirement 7's metadata is load-bearing: the
  decomposition cannot be recovered from the wiring alone.
- *Belief-side (inference): a hard distinction, by invariant.*
  Flattening exposes internal **states**, never **beliefs** — the
  composite's variables are world-side; each mind's posterior lives in
  its own carry. Communication between hierarchical levels of one
  mind's model is *one inference process over one model* (structured VI
  message passing within a single carry); the same two levels split
  into two minds may exchange only blanket traffic, each carrying
  beliefs *about* the other. Intra vs inter is soft bookkeeping for the
  world model and a hard boundary for inference — which is the
  no-telepathy invariant (req. 6) seen from the other side.

Env-like components need no special ontology: the homogenization lives
at the **driver seam** — every participant implements one protocol,
`respond(key, obs, carry) -> (action, carry')` — and the familiar cast
is a spectrum through that single seam, not a type hierarchy:

- an `Env`-interface thing (the real world, external software,
  transducer-wrapped hardware): an opaque particle whose insides are
  not simulated — coupled via blanket traffic only, no claims about
  its interior;
- a carry-free ancestral responder (a feedforward generative process);
- a **Bayesian perceiver**: carry + posterior updates, trivial policy —
  filters when acted upon, acts feedforward;
- a full agent: carry + planner. (a further reduction is possible: since
  factor graphs can represent arbitrary functions, and a planner is a
  function, there is no bedrock distinction between agents and nested
  Systems whose planners are subsystems. But both usability and efficiency
  concerns definitely favor shipping `Planner` as a built-in abstraction
  rather than literally a sub-`System` that implements a planning machine.
  Note the reduction is not merely theoretical — it has a chartered
  concrete form: the epistemic-priors planning-as-inference flattening
  (arXiv:2504.14898; the sampling-hardware thread in the brief), where
  planning IS inference over a policy-extended model — π as a latent
  whose priors absorb G. Two details make it clean: the reduction
  reifies the *policy variable*, never the whole carry, which is what
  avoids the self-modeling regress; and it lives behind the existing
  planner registry — a registry entry may be implemented as bespoke
  code (enumeration, CEM) or as inference on an extended System, and
  the seam does not change. So `Planner` ships as the default
  abstraction, with the reduction as a backend, not a rival ontology.)

**The Bayesian-mechanics commitment (recorded as a design principle).**
There is no fundamental dichotomy in this architecture between a
generative process and an external agent's active-inferential process —
"an env is just a System" is that claim, and the running composite is a
Bayesian-mechanical system at the exact/designed limit: each mind's
carry is its internal states, its declaration its model, `infer_states`
the sensory-perturbed internal dynamics, `act` the active states, the
driver the coupling. (The synchronization map between internal states
and encoded beliefs is the identity by construction — a digital
substrate doing exact conjugate updates — and the thermodynamic reading
is suppressed, not contradicted. The `eqx.Module` static-field /
array-leaf split is the timescale ladder rendered as a type system:
static structure is frozen plumbing, leaves are what flows.) Anything
the library commits to that cuts against this reading, other than as a
necessary practical limitation, is a design bug to be rethought.
Corollary, resolving the intra/inter question above in one line:
**a boundary is real exactly insofar as a carry is instantiated behind
it** — world-side, boundaries are metadata; belief-side, they are the
carries; and the one-mind/many-minds choice is precisely the choice of
particle decomposition (one mind = one carry = the internal boundaries
dissolved, legitimately).

## Architecture sketch

`merge({name: System}, bindings) -> System` — a **builder, not a
runtime concept**: namespace factor/modality names (`"boat.x"`),
concatenate factors and modalities, rewrite bound nodes as ordinary
cross-factor edges (a control-node binding becomes a parent in the
target's `B_deps`; an observation binding is an `A` declaration whose
`A_deps` point across the old boundary), record `subsystems` metadata.
Every existing face — planning, inference, `as_env`, the compiled
discrete path, hybrid dispatch — inherits for free, because the output
is just a System. Tractability is whatever the flat System's
tractability is: nesting adds namespacing, not inference machinery,
which is the honest answer the Ising constraint demands.

The residue — what is genuinely new (small, nameable):

- **The builder**: `merge` as a construction abstraction, plus binding
  validation (control-node arity, modality support/shape compatibility) and the
  `subsystems` metadata the other pieces consume.
- **The subjective-view constructor + routing driver — one mechanism,
  the novel core.** No-telepathy (req. 6) rules out per-mind planning
  on a *shared* posterior: each mind needs its own beliefs and its own
  objective (designations, γ, horizon). So `subsystem(name)` derives a
  bona-fide System per mind from the composite + metadata — its factor
  subset plus boundary stubs: others' bound controls as exogenous
  dynamics with declared priors (precedent: the switching demo's
  `ACTION_MODE_LOGW` table); v1 carries boundary-adjacent neighbor
  factors verbatim (shared structure, private beliefs; compressed
  partner models later). The driver interleaves per-mind
  `respond(key, obs, carry) -> (action, carry')` and routes
  actions/observations through the world, with the flat composite as
  ground truth (`as_env`).
  Per-mind planner assignment is a dict entry (`planner: str | None`;
  `None` = env-like ancestral response). One mind is the degenerate
  case: the composite plans all controls jointly — works today.
- **Tick-order semantics across minds**: sequential (declaration order;
  mind B sees A's action this tick) vs simultaneous (last-tick
  actions) — a semantic choice that changes the game structure, and the
  one nesting-specific piece of the synchronization story. The rest is
  inherited from the message-passing / distribution machinery.

Transducer slots land where the integration requirements wanted them:
outermost unbound ports are where frozen-encoder boundary components
attach — ports are the encoder registry's natural home.

## Design criteria

Minimal: one type (`System`), two builders (`merge`, `couple`), one view
(`subsystem`), one driver protocol (`respond`). Efficient: flattening
adds namespacing, never inference machinery. Usable — loud validation 
errors that name their fix, declaration-time statics that make wrong 
programs unrepresentable, canonical usage living in tests/examples 
(agents verify against source, not prose), and a brief-able surface — 
builder/view/driver is three small seams an agent can hold in one prompt.

## Build addendum (2026-08-17): the mission customer, adjudicated

The build's first customer is the downstream mission executor
(its handoff note in that repository): a mission-level System
selecting among a mission's unsettled tasks, choosing
retry/skip/escalate on failure, and setting the task-level System's
targets — consulted at task boundaries (seconds–minutes), operator
coexistence required.

### N1 — **RESOLVED**: target emission is conditional-`h` (`H_deps`), not a leaf-swap helper

"The high System's action sets the low System's terminal targets" is
implemented **properly, per arXiv:2312.07547** (Friston et al.,
*Active Inference and Intentional Behaviour* (https://arxiv.org/abs/2312.07547); Alex is
a co-author): `h` is a prior over final/intended states, explicitly
**reciprocal to `d`** — so the declaration mirrors the initial-prior
slot, and the `D_deps` machinery (v0.2) is the structural precedent
for **`H_deps`** (capital per the sibling `A/B/D_deps` SPM-letter
convention; the paper's lowercase `h` is the declared prior, its
capital `H` the *derived* inductive policy prior — a docstring
distinction, since the derived quantity never gets an API name): a
terminal designation *conditional on a parent
variable*. Under `merge()`, the boundary parent IS the high System's
state or control node — target emission becomes ordinary declared structure
(the same graph-merge as every other binding), with two consequences
that kill the alternative design: **no leaf swaps and no re-jit**
(all tasks' targets are declared once; the high level's choice is an
index into static structure), and the paper's p.13 extensions arrive
free later — *learnable* `h` (Dirichlet counts over intended states =
our slot polymorphism applied to a designation; "planning becomes
habitual") and the **inductive policy prior** capital-`H` (backward
reachability through precision-thresholded `B`, `P(u) = σ(−G − H)`)
as a *second scoring backend* for the same declaration beside today's
soft terminal-risk KL — declaration holds the prior, the
estimator/planner row decides how it enters, our standing registry
pattern. Rejected: the executor-side leaf-swap helper (re-jit
exposure on key-set changes, and the hierarchy lives outside the
declared model — exactly what the no-telepathy architecture exists
to avoid).

### N2 — **RESOLVED** (Alex, 2026-08-17): dynamic action sets via `log_E` masking

Settled tasks are masked with `log_E = −inf` — the habit prior as the
mask carrier: a *leaf swap*, so no re-jit as tasks settle within a
mission. Across missions, cardinality changes are per-shape jit
caching (the coverage port's per-grid-shape precedent, ~0.7 s once).
Needs a guard test (−inf through the policy softmax), not a feature.

*[Note 2026-09-18.]* The guard exists (`tests/test_action_masking.py`) and this
route stays valid. The recommended carrier for a **hard** constraint is now the
`legal=` keyword on `infer_policies`/`act`/`replay` (`action-legality.md`): the
same −inf, composed after `log E` at the softmax, so E is the soft prior alone
— the "mask is the squatter" finding of 2026-08-19, paid. A settled-tasks mask
is the `(policy_len, levels)` spelling of `legal=`.

### N3 — **RESOLVED** (Alex, 2026-08-17): division of labor, and the separator

The downstream **executor is the driver for this application** — the
mission seam ticks at task boundaries (the asynchronous case v1
explicitly defers), and operator coexistence (req. 6's
actions-as-observations) is already mechanically real at their NAV
seam. hybrid-aif ships the builder (`merge`) and subjective views
(`subsystem`) per this doc; the generic synchronous driver stays
minimal, serving the acceptance demo. Recovery
(retry/skip/escalate) and task preferences are modeling on existing
machinery, not library work — escalate-when-retry's-info-gain-is-low
is emergent EFE behavior, not code. **Separator: `.`** (`"boat.x"`,
`"arm.likelihood:pose"`) — namespace prefix on the left via `.`,
slot-qualification via `:` unchanged; the per-variable-infogain doc's
`/` example is superseded.

### Tick semantics (resolved at chunk-2 review, 2026-08-18)

Three driver modes, three games: **simultaneous** (one world step per
tick, merged actions — synchronous field update; simultaneous-move
games; statistically WRONG for equilibrium composites, where
synchronous updates do not preserve the stationary distribution —
the Glauber lesson), **sequential** (one world step per mind in
declaration order — turn-taking; clockwork games; privileges early
minds), **async** (one world step per mind in a random permutation
per sweep, key-reproducible — the statistically faithful mode for
Ising-like composites, Alex's spin-glass point; N steps per sweep is
what asynchrony *means*, not overhead).

*Amendment 2026-10-14 (Alex): the unit of time is the round, under every
order.* One round of updating is one tick of elapsed sim time, whatever
the schedule, and every participant steps once per round. The orders are
algorithms for one update, not amounts of time: **simultaneous** is a
Jacobi sweep, every mind reads the round boundary and the world's factors
then step on the round's actions; **sequential** and **async** are
Gauss-Seidel sweeps in declaration order or in a random permutation per
round, each participant reading current values, fresh where the source has
already stepped this round and the boundary's where it has not. The causal
dependencies within a round are sub-tick structure of one update, as a
sweep is read in numerical methods and in Glauber dynamics, where one sweep
updates each site once on its neighbours' current values. The unit that
steps once per round is a factor: in the merged form each component's
factors step at that component's turn, and a component with no respond
steps at the end of the round; in the wired form a process's factors step
together, at the end of the round. A realised action is consumed once, by
the one step of what it drives; a held control is a mind choosing it
again, a slow tier is a sticky mind, and turn-taking on a board is one tick
per move with "pass" a legal action. This replaces "one world step per
mind" above, which made a mind's model of its own action depend on how
many other minds were wired in, and keeps the spin-glass point: asynchrony
is the random order of one sweep, not N steps of the world. Engine:
`GenerativeEnv.step(factors=...)` steps the named factors and keeps the
rest; the driver uses it for the merged form's per-component turns.

Two clarifications that dissolved apparent problems. (1) The
discrete-path one-tick relay vs the hybrid path's within-step
parent order is NOT an inconsistency: **within-tick ordering is
factorization order of a one-step joint kernel, not a claim about
time**. The discrete path declares p(f_t|f_{t−1}, g_{t−1})
(old-parent; pymdp's B_dependencies kernel); the switching path
declares p(s_t|s_{t−1})·p(x_t|x_{t−1}, s_t) (new-parent; the
standard SLDS kernel) — different declared kernels, each its
family's convention, both legitimate ancestral factorizations.
Unification, if ever wanted: the original scoping's per-edge `lag`
tags. (2) "Instantaneous" factors (likelihoods) are **adiabatically
eliminated dynamics** — settling faster than the tick, idealized as
equilibrated within the atom of time; the license is timescale
separation, and the idealization is falsified exactly when settling
time becomes commensurate with the modeled dynamics (at which point
the factor promotes up the ladder into explicit dynamics).

### Partner modeling and acting-on-the-world (adjudicated 2026-08-18)

**Action, without essences (adjudicated through two rounds).** The
role decides what is an active state — the same move as
closure-is-a-property. If the output of policy selection `u`
directly conditions another component's dynamics (the `ActionControl`
binding), the binding is a merge; across a blanket, `u`'s action node
is the active state, wired to observation nodes, and this is the boundary
interface working as it should — a policy-selection machine must
ultimately condition other systems' sensory dynamics somehow.
Determinism of *transmission* is not determinism of *effect*: `u`
enters the receiving factor's kernel identically as itself and
conspires with that factor's other `B_deps`, so noise, partiality,
and interference live in the receiving dynamics (the classical
"shaky hand" is one kernel choice among many, made self-consistent).
Interposing an **effector/body factor** — a state of one's own whose
dynamics take `u` as one parent among others — is an available
modeling choice, wanted when the body has its own dynamics (inertia,
temporally correlated noise) or should be directly observable by
others; it is an option, not the paradigm, and neither pattern is a
degenerate case of the other. *(Amended 2026-10-09, C12.)*

**Hierarchy and multi-agent influence are ONE mechanism at two
tiers (Alex's unification, recorded as principle).** A parameter is
a slow state (the timescale ladder), so `H_deps` target emission and
`ActionControl` are the same thing — conditioning another system's
variables through a shared parent — at the slow and fast tiers
respectively. Telekinesis-freedom for target emission follows: on
the deep-model reading, a level's parameters are its blanket toward
the level above (level n's states are level n−1's empirical priors).
The honest difference between "writing priors" and "world action" is
timescale, not kind. **Principle: hierarchical models and
multi-agent models are formally indistinguishable, apart from
special machinery for modeling other agents' policies.** Subjective
distinctions are not to be reified into channels.

**There is no partner ontology (deflated at adjudication — Alex's
challenge, upheld against the first draft's "ladder").** What
`subsystem()` materializes for a partner's bound node is an **ordinary
uncontrolled latent factor**: `partner_priors=` sets its declared
dynamics (uniform is a default argument, not a semantic rung),
inference about the partner is likelihood inversion through whatever
depends on it (standard filtering), and learning it is ordinary
Dirichlet learning — responsibility-weighted soft counts on a latent
variable, machinery that already ships. "Stub" is CONSTRUCTOR
vocabulary (the view must be a well-formed System, so bound nodes get
materialized), never a type. Who allocates it splits by setting: in
the derived-view setting the *designer* declared the coupling, so
allocation is bookkeeping; in the ecological setting — where an
agent does not know which parts of its environment exist, let alone
which ones plan — partner structure is *learned* like any other
latent structure, and **agenthood itself is a structure-learning
hypothesis** (BMR / birth-merge, roadmap), not an ontology the
library pre-allocates.

**The one bright line: latents modeled as planners.** Dedicated
machinery is justified only where the modeled dynamics of a latent
is itself policy selection, and only for efficiency:
**transform-of-own-model** as an offered primitive (a partner =
own-model as reference + a minimal stored transform; thousands of
partners stored cheaply and simulated as a population batch through
the transforms — the homogeneous-batch vmap design's intended
customer, with "paradigm" = an equivalence class of static structure,
so paradigm formation is jit-class formation); and **E-module
surrogates** for the partner's planner (the amortization program's
contribution — avoiding nested planning). How ordinary latent-factor
learning could *bootstrap into* the transform representation is
recorded as an open, paper-grade research question, not a roadmap
item. No-telepathy is untouched throughout: modeling a partner's
beliefs as one's own latents is beliefs-about-beliefs; telepathy
would be reading their carry.

**Recorded convergence, not built:** the observation/latent
distinction may not be context-free — an observed variable is a
clamped variable, and clamping could equally be sampling from
dynamics driven by external states. Third independent arrival at the
generic `infer(query, clamped)` layer (after D3's polymorphic slots
and amortization A1's role-free families); noted on that roadmap
item.

### Build chunks

1. `merge()` — namespacing across the full v0.4 key surface
   (factor/modality names, epistemic/rate/`obs:` keys, wildcards,
   `LearnState` keys, `continuous_controls`), binding validation
   (incl. continuous controls), `H_deps` (N1) with the soft
   terminal-risk backend, `subsystems` metadata; acceptance: one-mind
   composite ≡ hand-flattened, bitwise.
2. `subsystem()` views + minimal sync driver + the doc's acceptance
   demo; then the mission demo against the real executor sockets
   (high System selecting subgoals via N2 masking, emitting the low
   System's terminal prior via N1, one failure → recovery branch).

## Acceptance sketch + sequencing

Acceptance demo: two Systems from the existing zoo coupled; run one-mind
and many-minds; §2(iv)-style check that the one-mind composite is
identical to the hand-flattened declaration.

Sequencing: parameter learning first (waiting customer: p_fix moving
inside the coverage agent), then the `merge` builder, then MNLR.

## Addendum (2026-09-30): rationale from the docstring paring

*Recorded nowhere but the text the 2026-09-30 paring pass cut from the
composition and System docstrings. The behaviour each describes stays in its
docstring. What is recorded here is why.*

- **N1: where a terminal target reads its parents
  (`system/core.py::System.resolve_terminal_targets`).** N1 records the
  declaration, not where `q(parents)` is read. For terminal risk the parents
  are read from the final predicted beliefs of the rollout being scored, the
  same belief tuple the KL's argument comes from. This is the symmetric
  choice: an initial prior averages over the parents' marginals at the episode
  start, a terminal prior over theirs at the horizon end. A parent whose own
  value the plan can steer therefore makes the target plan-dependent. That is
  deliberate, since it is precisely the supraordinate conditioning of the
  paper's p.13.
- **Build chunk 1: wildcards resolve, then union
  (`composition/coupling.py::resolved_epistemic`).** This is the honest
  namespacing strategy for wildcards, and it is deliberate. The V7 pattern
  grammar matches whole `:`-segments with no partial globs, so `"a.*"` is not
  expressible as a pattern. Making a subsystem's `("*",)` mean "my state
  variables" would require teaching the matcher about prefixes, which changes
  the grammar. Instead the pattern is expanded against the component's own
  variable table at coupling time, and the composite carries the resulting
  literal keys. The consequence: a wildcard is fixed at coupling time, so a
  component's `("*",)` does not later widen to variables the composite gained.
  That is the honest reading, since the wildcard was written against the
  smaller model. It is also required because a named variable must never be
  silently dropped (the "toggle lies" failure), and "everything eligible, for
  this component only" has no other spelling.
- **The habit prior composes as a sum over the composite policy set
  (`merge`, `_compose_policy_prior`).** A component's habit is a prior over
  its own controls, and the composite's joint policy space is the product, so
  independent habits add in log space. This is the composition the product
  policy space licenses, and the only one that leaves a single component's
  habit meaning what it meant. It is the same independent-habits reading that
  makes an unbound control node stay free. A component whose control nodes were
  merged away can no longer index the composite's policy set, and it raises rather
  than being reindexed on a guess.
- **Partner modeling: a joint habit does not restrict to a view
  (`_view_policy_prior`).** A mind that owns only part of the composite's
  controls has a smaller policy set, and a joint habit does not restrict to
  it, because marginalizing a habit is not a defined operation. The fix is to
  declare the mind's own `policy_prior` on the returned params.
- **Partner modeling: projecting the epistemic selector onto a view
  (`_view_epistemic`).** `True` and `False` pass through unchanged: they select
  over "whatever this System has", and a view is a System. So `True` mounts
  the boundary stubs too, which is the right reading: a mind may be curious
  about its neighbours' states, and the stubs are where it holds beliefs about
  them. An explicit key tuple is different in kind. It is a union of
  declarations, one per component, and each component could only ever have
  named its own variables. Projecting it back onto one mind therefore takes
  that mind's own keys. A key naming a stub came from the partner's
  declaration about itself, not from a curiosity of ours.
