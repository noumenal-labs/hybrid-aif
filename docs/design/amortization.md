# Amortization plan

*Status: adjudicated (Alex + Fable) — A1–A10 all resolved
(2026-08-04/06); goals ranked and build plan revised to goal (2)
primacy 2026-08-06 (see "Goals, ranked"); drafted 2026-08-03. Prompted by roadmap
input: amortizing AIF components (R-AIF-class agents) is an important
near-term goal. Binding reference: 
R-AIF (arXiv:2409.14216, Nguyen, Yang, Buckley & Ororbia, ICRA 2025; 
JAX implementation at NACLab/self-revising-active-inference — same ecosystem,
usable as a numerical oracle).*

> **Debt ledger:** complexity this arc introduced, with the principle that removes each item and its gate — `amortization-debt-ledger.md` (live list, opened 2026-08-27).

## The design claim

Amortization is not a new architecture for this library; it is three
existing seams exercised harder. Three orthogonal dials:

1. **Family dial** (per declaration slot): tabular/LG ↔ mixture ↔
   neural (frozen or learnable). A network in a slot is another typed
   declaration family — D3's polymorphic slot extended one notch:
   point value | posterior over parameters | **parametrized
   function**. Applies to likelihood, transition, and (workstream I)
   preferences.
2. **Search↔habit dial**: `q(π) ∝ exp(−γ·G + log E)` with E's
   representation per-planner (design/system.md — "CEM proposal dist
   and distilled habit nets are both E"). Full search (uniform E) ↔
   E-as-CEM-proposal ↔ pure habit (act from E, G never computed).
3. **Horizon dial**: explicit rollout for H steps, amortized value at
   the leaves (the Dreamer/R-AIF recipe). H=∞ classical, H=0
   actor-critic. The learned G-tail is an estimator-registry row
   (per-variable-infogain.md V4) and the tree's first deliberately
   non-exact leaf.

Strategic rhyme, recorded: the epistemic-priors flattening
(arXiv:2504.14898) that matters for thermodynamic samplers is the
same planning-as-inference seam — amortized nets and sampling
backends both live behind registries. One architecture serves both
roadmap threads. Narratively, a habit net is the policy posterior
pushed down the timescale ladder to the parameter tier (cf.
nesting-systems.md's clamp→param→state ladder).

## Decisions

### A1 — **RESOLVED**: The neural conditional family (what the slot holds)

**Decision (revised at adjudication — the role-free reading won; both
registry rows accepted).**
ONE new family, `MLPConditional` (equinox module; parents → Gaussian
output head; frozen ⇔ no trainable marking): the family contract is
(i) conditional log-density evaluation, (ii) ancestral sampling,
(iii) an inference strategy named in the registry. Gaussian output
heads first (mean+diag-scale MLPs) — every downstream consumer
(mixture filter, term estimators) already speaks Gaussian.

**Layering rule, made explicit (Alex's adjudication point).** Roles
("observation" vs "transition" vs "prior") are SLOT vocabulary, not
class ontology — the tabular families already work this way
(`TabularObservation`/`Transition`/`Prior` are aliases over the same
`ConditionalCategorical`/`Dirichlet` classes; role comes from slot
placement). The lower layer (`distributions/`, `inference/` —
AIF-free, open question 13) holds role-free conditional families
only; the AIF layer contributes slot placement, the t−1 unrolling
convention, and the action axis. Under the generic reading even the
action axis dissolves: an action is a parent variable that happens to
be decided rather than inferred, so `MLPConditional` takes a tuple of
parents and never knows one is "the action." An initial prior is not
a different class either — it is a different graph position
(parentless factor, child unclamped, vs. observation factors whose
children are data-clamped each tick); the slot encodes exactly that.
`MLPObservation`/`MLPTransition` exist at most as slot-alias names.
**Recorded naming debt:** `LinearGaussianObservation` bakes a role
into a mathematically role-free linear-Gaussian conditional (partly
the v0.2 restriction's fossil); renaming a just-stabilized v0.4
surface is churn without a customer, so the rename is assigned to the
generic `infer(query, clamped)` refactor, which restructures that
boundary anyway.

**The real fork inside A1: how state inference works when dynamics
are neural.** Two candidate rows for the registry:

- (a) **Structured approximate filtering** — moment-matched/unscented
  propagation through the net, keeping our filter-owned inference.
  Exactness story degrades gracefully; no recognition model to train;
  slower per step.
- (b) **Amortized recognition (RSSM-style)** — a trained encoder
  q_φ(s|o, s_prev) is *the family's inference strategy*; inference
  itself becomes a learned component (this is what R-AIF/Dreamer do,
  and what phase 3 pixels will need anyway).

**Decision: both are registry rows; (b) is the R-AIF track's row, (a)
is the diagnostic/fallback row.** The registry (V4) was built so that
"which inference" is a per-family static choice — this is its first
big test outside the epistemic terms.

### A2 — **RESOLVED** (direction; details reviewed at build): The gradient learning rule

**Decision.** The Learnable protocol's `stats()/absorb()` split maps
onto `grad`/`optimizer step`: `stats` = per-tick gradient
contributions (or the minibatch itself, for replay-driven learning),
`absorb` = optax update. Optimizer state is a new explicit carry —
`LearnState` pytree returned and re-accepted by `learn`, sibling to
SystemParams, never hidden in a module attribute (pure-functional
discipline; the noise-posterior carry in the coverage port is the
precedent). Initialization is **explicit** (adjudicated at build:
Alex asked whether the dropped-carry silent-restart hazard could be
guarded; it can): the first call passes
`learn_state=system.init_learn_state(params)`, and `None` with
gradient slots present raises — a fresh start is a visible act at
the call site, never an omission. Zero steady-state ceremony (the
carry threading already existed); the A6 trainers absorb even the
one-time init. `lr_by_slot`/`decay_by_slot` grammar applies unchanged
(a slot-qualified key scales that module's gradient step; wildcards
per V7).

### A3 — **RESOLVED**: Habits — the E representation, in every planner

**Decision (amended at adjudication — Alex caught the omission).** An
`E` designation may hold a policy module (beliefs → action
distribution; equinox) or the classic static E-vector. E participates
in ALL planners, starting with the standard one: the **enumeration
planner** adds `log E(π)` inside its softmax — `q(π) = σ(ln E − γ·G)`,
E's most classical home (SPM/pymdp's E-vector position). For a policy
module, `E(π)` per enumerated policy is the product of its per-step
action probabilities; for a static vector, a lookup. Two new
planner-registry entries complete the dial: `"habit"` (act from E
directly — the H=0 end; G optionally scored for diagnostics but not
used) and `"cem"` with an `e_proposal=True` mode (E
initializes/tempers the CEM proposal). The search↔habit dial is then
one formula: γ and search budget high with E flat = classical; E
trained with γ or budget low = amortized. `score_policies` remains
the teacher face; `distill(traces) -> E-module` is the supervised
path from teacher traces (workstream G's chartered columns: beliefs +
per-policy term trees).

### A4 — **RESOLVED**: The learned value tail (critic) and honest trees

**Decision.** A `g_tail` estimator-registry row: after H explicit
rollout steps, a value module v_ψ(beliefs) estimates the remaining
EFE. Trained TD-style on imagined rollouts (A6).

**Tree honesty (adjudicated through three rounds — the final form).**
The Interval contract disentangles into two orthogonal axes:
**width** (point vs. spread — the constructor choice) and **warrant**
(does a derivation guarantee the enclosure?). The existing machinery
implicitly promised warrant everywhere; amortized leaves are the
first that cannot. Resolution: a static flag **`derived: bool =
True`** on Interval — no new constructor, no new vocabulary.
`Interval.exact(x)` keeps its width-only semantics ("the estimate is
a point"), so the critic's output is `Interval.exact(v, derived=
False)` and net-predicted *bounds* (a genuine grade: the epistemic
axis is orthogonal to spread) are the bound-pair constructor with
`derived=False`. Propagation is the algebra of guarantees: a value is
derived iff every input to it is derived (AND), which is why
downstream poisoning is principled rather than bookkeeping.
Rejected en route: `.believed` / `.belief` (marks a psychological
status — any claim is trivially believed; and `.belief` collides with
the `Beliefs` seam type); `estimate` as the uncertified term (the V4
registry already uses "estimator" for every leaf computation, exact
ones included). `derived` asserts *conditional* soundness —
"guaranteed given a correct implementation" — the same status as any
theorem-in-code; bugs are the test suite's jurisdiction. Default
`True`: every existing site is a genuine derivation (zero churn), and
the unusual thing gets marked at its call site. Future grade noted,
not built: *statistically* derived (conformal-style coverage), which
would widen the flag to a three-valued enum — the boolean is designed
to be widened, not un-designed.

**Enforcement is structural, not disciplinary (Alex's follow-up):**
`derived` is a field of the estimator-registry row — declared once at
registration, inherited by every Interval the row constructs, never
passed per-call. Rows whose estimator carries trainable parameters
get NO default (the row constructor demands an explicit `derived=`),
so the only place a trust label can be wrong is the one reviewed line
that registers the amortized estimator; the propagation AND does the
rest.

### A5 — Learned preferences + self-revision (workstream I opens)

**Proposal.** A preference designation whose target Distribution is
produced by a trainable module (the family dial applied to C).
First concrete entry: R-AIF's prior-preference model + self-revision
schedule — preferences updated from the agent's own successful
trajectories (hindsight-flavored; exact mechanism pinned from the
paper in phase 0). Scope fence, priced at adjudication: the *seam* (a
designation holding a trainable module) is fully generic by
construction — it does not know what its module was trained on, so
excluding generic preference learning costs zero code and no design
distortion. What stays out is only *additional trainers/estimators*
(IRL-style losses, preferences-over-parameters machinery) — later
workstream-I passes on the same seam.

### A6 — **RESOLVED**: Where training loops live

**Decision.** Not on System (faces are per-call, pure). A new
`hybrid_aif/train/` module: `distill()` (teacher traces → E),
`imagine()` (Dreamer-style actor+value optimization on world-model
rollouts — reuses `predict_beliefs`/rollout machinery), `fit_world()`
(replay-driven world-model learning through `learn`). Each is a
function over (System, params, data/buffer, LearnState) — the library
stays à-la-carte; no Trainer god-object — i.e., no
Keras/Lightning-style class owning model + optimizer + buffer + loop
as mutable attributes and becoming the de facto API. Pure functions
over explicit state keep every trainer jit-able, resumable from its
returned state, and composable with the faces rather than wrapping
them.

### A7 — **RESOLVED** (Alex): phase 1 is state-based Mountain Car

Continuous sparse-reward control without the pixel front-end:
forces A1 (nonlinear dynamics), A2, A8's CEM, and the
preference/epistemic balance, while the observation model stays
trivial. Pixels (and the transducer contract, workstream H) enter at
phase 3.

### A8 — **RESOLVED** (Alex): R-AIF replication is the binding acceptance

Their tasks, their metrics (cumulative reward, stability, success
rate), their JAX code as oracle. Revisable if the replication
surfaces a principled divergence — but divergence must be argued in
this doc, not discovered in a demo notebook.

### A9 — **RESOLVED**: CEM planner (workstream E's middle, promoted)

**Decision.** Enumeration cannot serve continuous actions, so the
classical end of the slider needs CEM/MPPI. CEM first (simplest,
and A3's `e_proposal` hooks into it); gradient/MPPI variants stay
roadmap. Lands behind the existing planner-injection seam.

**Continuous-action families are a positive list (adjudicated
2026-08-17, at chunk-2 review).** The build's fence — a continuous-
controlled factor's transition must be `MLPConditional` — is a
one-member list, not a principle: the requirement is "accepts a
continuous action parent." Our LG transitions condition on a level
only because the shipped class is a per-action bank of `F/b` maps
(the pymdp inheritance); the canonical continuous-action world model
is the LQG form `x' = Fx + Gu + w`, and a linear-in-control LG family
is conjugate-learnable with shipped machinery (augmented-head
`[F|G|b]` MatrixNormal/MNIW regression over `(x, u)`). Chartered as
the list's second member on first customer; its switching-mixture
extension (piecewise-linear dynamics) is a goal-(3)/P1b-adjacent
note.

### A10 — **RESOLVED**: P0 findings, adjudicated dispositions

P0's addendum (`amortization-p0-addendum.md`) pinned the target and
raised eleven gap notes; adjudicated 2026-08-06.

**Acceptance philosophy (supersedes tolerance drafting):** match
paper numbers; shortfalls are discussed, never pre-excused. P1's
target is *generated* (their code run state-based — the env already
returns a `state` key their config filters out), not invented. P3
target: Table 2 (−68.3 / SR 1.0 / R-S 0.1), `tiny` @ 2M steps —
achievability established by Alex's replication in the sibling
`raif-jepa` repo. P4: `robosuite_Door` (425.8 / SR 1.0), Meta-World
`door-open-v2` as the cheaper-not-weaker fallback.

**Dispositions:**
1. Categorical output head joins `MLPConditional` on the P3 track
   (their latents are 32×32 one-hot); Gaussian head stays A1 default.
   *Adjudicated through a second round (Alex: why not conjugate
   Categorical/Dirichlet latents at P1, changing one axis instead of
   two?):* "categorical at P1" and "categorical at P3" would share a
   label, not a representation — a conjugate categorical latent for
   MC is a grid *discretization* (fixed cell semantics, an
   approximation floor, and a second forced discretization of the
   continuous actions), while R-AIF's is a *learned distributed code*
   (32 factored variables, no state-space geometry). The
   discontinuity at P3 is representation-learning itself, which no P1
   family choice removes; attribution at P3 comes from the oracle,
   not from single-axis variation. Conjugate family choice follows
   the variable's intrinsic type (MC state: two continuous numbers ⇒
   Gaussian/mixture); R-AIF's categorical is a gradient-regime
   stability trick whose rationale doesn't transfer to the conjugate
   regime. Constructive residue: **P1b (optional)** — the fully
   discrete-conjugate variant as the pure-conjugacy left end of the
   slider, where sparse-reward exploration is driven by our own
   Dirichlet parameter novelty (`infogain` over `transition:*`) —
   count-based systematic exploration from machinery we already ship,
   no neural anything. In scope after P1 passes, first thing dropped
   under schedule pressure.
2. The GRU `deter` carry is the **first customer of the scoped
   `Deterministic(map, rule)` Factor member** (designed at scoping,
   unbuilt for lack of a customer). `Beliefs` is already an opaque
   pytree — a deterministic component rides through the seam free.
3. Their epistemic term = 4-MLP ensemble disagreement — this is
   **workstream H's chartered "ensemble info-gain" landing** as an
   `ensemble_disagreement` registry row, `derived=False`, sibling to
   (not a variant of) the infogain rows.
4. A5 amended: a preference designation may name its own
   *instrumental estimator row* (R-AIF's: `cosine_sim_to_pref_rollout`
   — a deterministic-function factor, the union's second customer).
   Distribution-target designations remain the default form.
5. CRSPP shares architecture, NOT parameters, with the world model
   (second parameter set, own GRU, one class). Express as a second
   `MLPConditional`-family declaration whose parents exclude the
   action.
6. Critic head: follow the code (255-bin discrete log-prob), note the
   paper/code divergence; Alex reviews at implementation.
7. Sign convention: ours (G minimized) doc-wide; convert at the
   registry boundary; verify the math, don't copy the design.
8. Their BC term lives inside the actor loss: `imagine()` gains an
   optional supervised mixing term; mixed-vs-alternating default
   decided at P2 (replication fidelity favors mixed).
9. Buffer alternation (strict 50/50 positive/negative) stays in the
   caller's loop — trainers remain batch-level and pure; a three-line
   `alternate(a, b)` iterator utility ships so the recipe stays one
   line. (Rejected alternative: trainers owning buffer policy — bakes
   an R-AIF-specific data policy into the generic signature.)
10. A1(a) (structured filtering through a net) has no R-AIF oracle;
    validated against our own tabular/LG exactness tests, and its row
    docs say so.
11. **Ratified (Alex):** a fully-`derived=False` tree in amortized
    mode is the flag working as designed — decomposition is the
    diagnostic value, the flag is the honesty; no mitigation needed.

## Goals, ranked (adjudicated 2026-08-06 — Alex)

1. **Enable amortized modules in hybrid-aif** — the "slider" means
   amortized functions swap in *piecemeal*, with no wholly different
   model type. (Infrastructure: A1–A6.)
2. **THE MAIN TEST: replicate R-AIF's results *using the same model
   architecture*, inside hybrid-aif.** Rationale: the company wants
   to build on R-AIF; if hybrid-aif cannot express and reproduce that
   work, the repo does not earn its keep for the internal jobs it
   exists for. Architecture parity is the claim, not just task
   parity.
3. **Marginal:** how progressively more classical variants of the
   same architecture fare (the slider prods — the P2 figure, P1b,
   the search-swapped ablation). A bespoke hybrid-aif agent with no
   architectural constraint from R-AIF addresses neither (1) nor (2)
   — cool to see, never a milestone.

*(Process note, recorded: the original plan below through its first
build chunks was drafted before the corrected framing above;
chunk 2's env/continuous-action work is on goal (2)'s critical path,
CEM is goal-(3)/workstream-E infrastructure, and its Mountain Car
experiment is DEMOTED from P1-acceptance to an integration
smoke-test of the new machinery.)*

### A11 — **RESOLVED** (Alex + Fable, 2026-08-19): the `recognition` slot

Where amortized recognition lives, adjudicated against five
alternatives. **Decision: a fourth slot on `SystemParams`** —
`recognition={variable_name: module}` — with `System` carrying only
the structural declaration (which variables are amortized; the
module's parents = the conditioning set, always explicit), and the
inference registry's `"amortized"` row consuming the slot: **the
registry names the strategy; the slot holds the strategy's
parameters.** Variable-keyed (Alex: recognition models are about
Q(x); `recognition:stoch` — R-AIF's latent's name — `recognition:pose`
for the mission model). Everything inherits free: rate keys,
freezing, LearnState, wildcards, merge() namespacing. Degenerate
case = the slider one layer down: a variable without an entry falls
back to the structured/exact row.

Rejected: (B) welding the encoder into the generative family — an
inference choice is not a world claim; freeze-model-train-encoder
becomes unexpressible. (C) injection-like-planners — planners are
stateless registry code; recognition has trained parameters needing
a persistent, addressable home. (A-as-proposed) the designation on
`System` alone — D with its bottom half missing. Recorded horizon
(E): recognition as declared q-side graph structure under the
generic `infer(query, clamped)` layer — the FOURTH independent
arrival at that layer; D is precisely E restricted to disjoint
per-variable stars with opaque modules, so migration retracts
nothing. **Refined (Alex, same day, leaning E): D is adopted AS E's restricted
syntax, never a rival.** Three commitments keep the migration
frictionless: (i) each recognition entry's declared semantics IS a
q-side factor (child = variable, parents = conditioning set) — E is a
syntax superset; (ii) module-object sharing across entries RAISES
(the pytree-forking trap: apparent sharing silently duplicates
parameters at the first optimizer step; shared torsos are E's
feature by name); (iii) parent sets explicit and validated, so
structured Q arrives as a validated relaxation. When the generic
layer lands, the dict remains the default sugar — complicated Q is
an opt-in, not a second mechanism. Convergence noted: workstream H's transducers are the
frozen case of the same declaration shape — pixels at P3 land in
this home.

### A12 — chunk 2b adjudications (Alex + Fable/Opus, 2026-08-20 → 25)

The six decisions 2b was dispatched on, previously recorded only in the
session transcript. Companion framing adopted alongside them:
`objective-terms.md` (the term lattice — conditional × functional × sign;
`occupancy` is a NEW leaf, never a sign-flipped `entropy_lowering`,
because `H[q(s')] = E_q H[p(s'|s,a)] + I(s';s)` and the flip would count
`I(s';s)` twice against 7b) and `system.md`'s 2026-08-21 addendum (the
kernel vocabulary and the three groups of `SystemParams`).

1. **RESOLVED — the `scoring` slot + the `terms=` selector.** A
   row-name-keyed field on `SystemParams` — `scoring={"g_tail": (critic,
   slow), "ensemble_disagreement": (m0..m3), "cosine_sim_to_pref_rollout":
   crspp_params}` — holding the parameters of estimator-registry rows that
   carry their own (group 3 of the system.md addendum; the registry names
   the strategy, the slot holds the strategy's parameters — A11's rule
   applied to the objective side). Mounting is declared on `System` via a
   general `terms=(...)` selector, never inferred from the slot's
   contents. **Intended architecture:** one selector mechanism, with
   `epistemic=` as its *restricted syntax* (the A11 pattern) — a
   variable name dispatching by family to an `infogain` estimator row.
   **Built in 2b (amended 2026-08-27):** two *sibling* selectors over
   one suffix grammar — `epistemic=` still runs the V1–V8 resolver
   against the `(kind, family)` dispatch table, `terms=` consults the
   by-name row table, and the four `infogain` estimators live only in
   the former, so no `terms=` spelling reaches them (`"infogain"` is a
   term, not a row; `"ensemble_disagreement:stoch"` is not registered by
   name). The merge that makes `epistemic=` literal sugar is a refactor
   of the v0.4 resolver (eligibility, channel fan-out of `predictive`
   leaves, the one-estimator-per-variable rule, the compiled path) under
   its parity suite — recorded as DEBT, its own chunk, scheduled with
   `register_estimator`'s first customer, when a user row first needs
   the infogain machinery. Alex: "I don't like the debt, but it's not
   worth interrupting the amortization arc." V1–V8 carry over
   untouched. Rates/freezing/LearnState/
   wildcards/merge() namespacing inherit free. Names rejected en route:
   `modules` (doesn't distinguish from `recognition`), `strategies`
   (opaque), `estimators` (reads as the Interval/bounds machinery),
   `aux`/`auxiliary` (negative condition, unenforceable membership).
   `scoring` is the positive condition: contents compute numbers that
   rank options; mechanically, a scoring entry returns an `Interval`,
   never a `Distribution`. **Open at build: the `terms=` key grammar**
   (row-prefixed keys like `"occupancy:transition:stoch"` are a new key
   shape) must be checked against the V7 wildcard ladder *before* code.
2. **RESOLVED — CRSPP is a declared System, no opaque type.** Their
   "RSSM without action encoded into the transition prior" is
   `tiny_rssm` minus the action binding and minus heads
   (`crspp_rssm()`, ~5 lines); its rollout is the shipped
   `predict_beliefs`. It is a group-1 declared model *owned by* a
   group-3 row — role assigned by the consumer, not the type (the
   nesting no-partner-ontology doctrine again). Standing rule (Alex):
   first-class AIF constructs only as sugar over primitives the library
   would want anyway.
3. **RESOLVED by citation — the actor lives in the E designation.**
   Already the design of record twice over (`system.md` Planning,
   confirmed 2026-07-21: "a learned habit network for the future
   amortized backend"; A3 above). The actor is a closed-loop kernel
   (returns `P(a|s)`), so it was never `scoring` material. Fidelity
   consequence in the replication config: `planner="habit"` — G is a
   *training* signal consumed by `imagine()`, not an act-time score;
   CEM/`e_proposal` act-time planning is the goal-(3) slider prod.
   Sub-decision RESOLVED (Alex, 2026-08-26): rename `log_E` →
   `policy_prior` (`Array | Categorical | policy module`), with a
   read-only letter alias `E` completing the existing `A`/`B`/`D`
   property convention — D is the prior over initial states, E the prior
   over policies; a module entry is a *conditional* policy prior (house
   precedent: `D_deps`/`H_deps`). No `log_E` compatibility shim: no
   external users exist, the downstream repository's one call site migrates in the same
   pass, and a `log_E` accessor would be ill-defined for modules anyway.
   The friction-9 γ-independence note moves verbatim onto the new field.
   (`habit` — the earlier candidate — rejected as naming one *use* of
   the field, the H=0 end, rather than what it is.)
4. **RESOLVED — R-AIF's odd terms ship as named rows, plainly.**
   `occupancy` (their 7c/`wment`; MOP's β — see `objective-terms.md`
   for why this is principled, not replication-scoped),
   `predicted_reward` (their `r_τ`), `cosine_sim_to_pref_rollout`
   (their `sim`; disposition 4's name). All sign −1, all
   `derived=False`. The latter two sit OFF the lattice's closed
   functional vocabulary — the registry admits off-lattice rows,
   `derived=False` is the marker, and the lattice maps the principled
   ones; `predicted_reward` is a bare linear utility with no target
   distribution, which is exactly the kind of row the honesty flag
   exists for.
5. **RESOLVED (Alex, 2026-08-26) — critic head follows the code** (A10 disposition 6's
   deferred review): 255-bin `symlog_disc` `−log_prob` plus the
   slow-target logprob regularizer, not paper Eq. 9's MSE. The oracle
   is the code; the divergence is noted in the row's docs.
6. **RESOLVED (Alex, 2026-08-26) — expert seeding by oracle-fixture
   export.** The seed data is a confound on the acceptance claim in
   BOTH directions (a bang-bang controller is likely *better* than
   their PPO expert on quality — ≈45 steps/episode, inferable from
   their 1,000 steps ≈ 22 episodes — and definitely *worse* on
   diversity: deterministic control gives M⁺ ≈22 near-copies of one
   trajectory, a qualitative change to the distribution CRSPP learns
   preferences from). Resolution dissolves the tradeoff: the oracle
   run IS their pipeline, so its `collect_prior` episodes already
   exist — **export them from `raif-paper` as a checked-in `.npz`
   fixture** ((obs, action, reward, is_success, can_self_imitate) per
   step, per oracle seed if their protocol collects per-seed), and
   seed M⁺ from the fixture. Exact parity, zero new dependencies, and
   the fixture doubles as a bitwise env-parity test (their recorded
   rewards must reproduce under our env stepped with their actions).
   Provenance ({raif-paper SHA, command, seed}) recorded with the
   fixture or it rots into magic data. A scripted bang-bang
   `sign(velocity)` controller ships as the documented fallback
   seeder (~15 lines, 2c) for fixture-less runs and later demos.
   Seed stats are reported next to the acceptance numbers either way
   — under the fixture that is a parity check, not a confession.
   Rejected: torch/SB3 as dev deps (unneeded given the export);
   filtered random episodes (strictly weaker).

**Scope split (adopted):** 2b = the G side — `scoring` slot + `terms=`,
`Interval.derived` + the registry-row `derived` field (A4's enforcement
half, still unbuilt), the five rows (`g_tail`, `ensemble_disagreement`,
`occupancy`, `predicted_reward`, `cosine_sim_to_pref_rollout`),
`crspp_rssm()`, the habit planner consuming the E-designated actor, and
unfencing `infer_policies`/`score_policies`/`act` on amortized Systems.
Acceptance: G scores over the declared RSSM with an honest
all-`derived=False` tree, and the agent acts. **2c** = the training
side — `imagine()` (which MUST sample: the measured 2.8×-vs-1.3× gap of
chunk 2a; and which MUST condition the CRSPP posterior on the batch's
observations before imagining — ledger entry 5, 2b's episode-start
placeholder is not parity-faithful), λ-return/retnorm/actent/wment/BC, critic + slow EMA, ensemble
NLL, CRSPP ρ-gated KL + contrast, the per-episode `pref_label` backward
pass, `alternate()`, dual buffers, seeding. Acceptance: the full loop
runs on MC and learns. **2d** = the 4-seed 1M-step run against the
oracle — compute, not code, and the actual goal-(2) acceptance.

## Build plan (phased; each phase independently green; revised 2026-08-06 to goal ranking)

**PAUSED 2026-08-17** at the P1-infra/P1′ seam: the nesting-Systems
build is pulled forward by an application need. Everything through
P1-infra is committed and green; P1′ is fully specified above and
resumes from this doc. The state-based oracle run (Alex's side,
raif-jepa env) can proceed during the pause — P1′ then resumes with
its target already generated.

- **P0 — pin the target.** DONE (the addendum; adjudicated).
- **P1-infra — family + gradient learning + env + continuous
  actions + CEM.** DONE as build chunks 1–2 (see process note: the
  chunk-2 experiment is an integration test, not an acceptance).
- **P1′ — R-AIF's architecture, state-based, in hybrid-aif. THE
  goal-(2) milestone.** Their `tiny` model expressed as a System:
  GRU `deter` (the `Deterministic(map)` node), categorical stoch
  (the categorical `MLPConditional` head), A1(b) amortized
  recognition as the family's inference row, `ensemble_disagreement`
  epistemic row, CRSPP preference row (A10 disp. 4/5), critic
  `g_tail` (A4), actor as the E module trained by `imagine()` with
  the BC term (A3/A6, A10 disp. 8), self-revision + dual buffers
  (A5, A10 disp. 9), scripted-expert seeding. Acceptance: match the
  state-based oracle (their code, state keys, `tiny`). This is the
  P3 stack minus pixels — nothing in it is built early that goal (2)
  didn't need.
- **P3 — pixels.** P1′ + the conv-encoder path (workstream H's first
  shipped instance). Acceptance: Table 2 (−68.3 / SR 1.0 / R-S 0.1),
  `tiny` @ 2M steps, 4 seeds.
- **P4 — robot arm.** `robosuite_Door` per their suite (Meta-World
  `door-open-v2` fallback), paper numbers.
- **Goal-(3) shelf (optional, after (2) is served):** P2 the slider
  figure; P1b the discrete-conjugate left end (A10 disp. 1); P3b the
  search-swapped ablation of R-AIF itself (their architecture, CEM
  replacing the actor, MC-G replacing the critic — zero architectural
  DOF; natural home is their codebase/the raif-jepa fork, or ours
  once A1(b) exists).

## Sequencing

Bumps workstreams E (A9), G-remainder (teacher traces), H (phase 3),
I (A5). The generic `infer(query, clamped)` layer and the
nesting-Systems build slide back — both fully documented, and
nesting's ordering constraint (namespace the refactored selector
surface) is already satisfied. The v0.4 refactor is load-bearing
here: the estimator registry is the amortization seam, and the
epistemic/rate selector grammar is family-agnostic, so neural
families arrive without touching the API we just stabilized.

## A12 implementation note: the `terms=` key grammar

*Status: **build-time notes, not adjudication** (Opus, 2026-08-26,
chunk 2b Task 0). A12 item 1 left "the `terms=` key grammar must be
checked against the V7 wildcard ladder *before* code" open at build.
This section is that check: what the build implements, and — flagged
inline as **FLAG** — the places where the adjudicated specs
underdetermine the answer and the build took the conservative option
for Alex's review. Nothing here overrides A12, A4, A11 or V7; where a
reading was forced, the flag says so.*

### (a) Row keys and the variable key space

A `terms=` entry is **`<row>` or `<row>:<variable key>`**, where
`<row>` is a *literal* estimator-registry row name and `<variable key>`
is the existing V1/V8/A11 grammar unchanged (bare state name,
`obs:<modality>`, `<slot>:<name>[:<block>]`).

Non-collision is structural rather than lexical:

- the row segment is resolved against a **closed finite table** — the
  registered rows — exactly as the first segment of a parameter key is
  resolved against `_KEY_KINDS`. A first segment that is not a
  registered row name raises and lists the rows;
- `terms=` deliberately does **not** accept a bare variable name.
  There is therefore no "is `g_tail` a row or a state variable?"
  question to answer: in `terms=` the first segment is always a row,
  and in `epistemic=` it is never one. The two selectors keep separate
  parsers over one shared *suffix* grammar;
- registration enforces the only remaining hazard: a row name may not
  contain `:` and may not be one of the slot words or `obs` (a row
  called `obs` would make `obs:cue` ambiguous between the two
  selectors' readings). This is a one-line guard at
  `registry.register_row`, checked at import.

Consequence worth stating: a *state variable* may legally be named
`g_tail`. Nothing breaks, because that name is only ever read as a
variable by `epistemic=` and only ever as a row by `terms=`.

**Eligibility is per row and per family** (amended 2026-08-31, the
tabular `occupancy` build; `amortization-debt-ledger.md` entry 2 paid).
2b resolved `terms=` only on an amortized `System` and raised wholesale
otherwise. That was a placeholder for family dispatch, and it is now
one: a registry row declares the **variable families** it carries an
estimator for (`registry.Estimator.families` — `code`/`carry` on the
amortized path, `categorical`/`gaussian`/`mixture` on the declared
one), a row name may hold **several estimators keyed by family**
(`occupancy` is the first: a closed-form tabular leaf beside the neural
one), and a mount resolves to `(row, family)` exactly as `ESTIMATORS`
resolves `(kind, family)` for `infogain`. A key naming a target whose
family the row cannot read raises a `NotImplementedError` — the type
`registry.prepare` already uses for a missing estimator — naming the
row, the family and the fact that registering a `(row, family)`
estimator is what would make the key legal. A row with no factor
target (`g_tail`) is checked against the System's *regime*, since what
it reads is the rollout as a whole. The build note is in
`objective-terms.md`.

**FLAG (1).** `epistemic=` remains the **only** spelling that mounts
`infogain` rows in 2b. A12 calls `epistemic=` "permanent sugar for
`terms=("infogain:<var>",)`", and the build honours that by desugaring
internally — but the *reverse* spelling, `terms=("infogain:stoch",)`,
raises with a message pointing at `epistemic=`. Reason: making it work
means re-expressing V1–V8's whole resolution (eligibility per family,
the one-estimator-per-variable rule, the channel-vs-variable keying,
the deferral errors) under the row grammar, which is a resolver merge,
not a spelling. Doing it half-way would create a second path to the
same leaves that could disagree with the first — and V1–V8 must stay
bitwise. The sugar is permanent either way, so the merge is optional
work with no user waiting for it.

### (b) Wildcards and the V7 specificity ladder

V7's ladder is a statement about *variable* segments: literal segments
are validated against a finite domain, `*` matches one whole segment,
and "a wildcard never validates less than its literal part."
Row keys inherit that verbatim **for the variable-key part**, and
nothing for the row part:

- **`terms=("*",)` raises.** A wildcard in the row segment would mean
  "mount every registered row", which is not a class of variables but
  a set of independent modelling commitments, most of which need
  parameters this System does not carry. There is no domain against
  which to validate it and no reading of it that is not a guess. It
  raises naming the rows.
- **`"occupancy:transition:*"` is legal** — the row's target class,
  validated by V7's literal projection (so `"occupancy:transitoin:*"`
  raises as the typo it provably is, and a model with no transition
  declarations resolves to "all of none" silently, per V7's
  empty-through-the-wildcard rule). The class it ranges over is the
  transitions whose **family** the row has an estimator for, so the
  wildcard skips an amortized carry or an LG factor silently while
  naming one literally raises — one rule, both regimes.
- **`"occupancy:*"` raises.** A single-segment `*` means "every state
  variable" in V7, and `occupancy` scores a *transition kernel*, whose
  key is `transition:<factor>`. The pattern's shape cannot address the
  row's target kind, and the literal part (the row name) pins that
  kind — so it is a provable mistake, not an empty class. Same for
  `"occupancy:obs:*"`.
- **`"occupancy:*:*"` raises.** The wildcarded slot segment would
  range over `likelihood`/`initial_state_prior`/`recognition` as well,
  none of which `occupancy` can read. The error names the one legal
  slot. (Silently filtering to transitions was the alternative; it
  makes `*:*` mean something different here than everywhere else.)
- **A row whose target is a single named thing takes no wildcard.**
  `predicted_reward:obs:*` and `g_tail:*` raise. `predicted_reward`
  scores *the* reward channel (a class of reward channels is not a
  modelling concept), and V7's motivating argument — enumeration is
  not robust, a later-added variable silently isn't mounted — has no
  purchase on a row with exactly one target by construction.
- **Rows with no target at all** (`g_tail`,
  `cosine_sim_to_pref_rollout`) accept the bare row name only; any
  suffix raises, saying what the row scores instead.

The ladder itself (specificity = literal segments, most specific wins,
ties raise) is **not** needed by `terms=`: selection is an idempotent
union, exactly as `epistemic=` is, so two patterns selecting the same
leaf is not a conflict. The ladder stays where it is required — the
rate mappings.

**FLAG (2).** `predicted_reward` requires its modality **literally**
(`terms=("predicted_reward:obs:reward",)`); there is no
inferred-by-head-loss form. Identifying "the reward head" by its
declared loss would read `params.likelihood`, and the term-tree *shape*
must be a function of the `System` alone (A12 item 1: two
`SystemParams` against one `System` must produce the same shape).
Naming the channel is one string and keeps that invariant exact.

### (b′) What `metrics=` returns under a habit — and where the diagnostic lives

*(Two rounds with Alex, 2026-08-27. The first build passed the
protocol's metrics-on default through, so `infer_policies` under
`planner="habit"` rolled the plan out and returned its G. Round one
made that opt-in via a per-planner default mark; round two withdrew
the mark and fixed the question instead. Recorded in full because the
withdrawn version is the instructive part.)*

**The rule, stated for the docstrings.** `metrics=` toggles whether
the *by-products of the selection computation* are returned. Their
**content** is not a matter of taste: it is whatever **term tree** that
computation produced — registered terms, principled or ablated or
off-lattice, not "G" specifically. A search planner built a tree in
order to select, so it returns it. The habit planner's selection
computation is the actor's distribution, which is already returned in
full as `q_pi` plus the plan, and it produces **no term tree at all**.
So under `metrics=True` it returns `{}` — an *empty* tree, "requested;
the computation had none" — kept distinguishable from `None`, "not
requested". `metrics` therefore keeps its uniform `True` default
everywhere, and honouring it under a habit is free.

**Why empty rather than fillable.** This is the term tree's subject
matter, not a gap in its vocabulary: **the tree scores the model, and
the policy is not part of the model.** `objective-terms.md` already
decided exactly this for MOP's `α` term ("Where MOP's α term lives:
the trainer, not the tree") — policy-side quantities like `log π(plan)`
and `H[π]` have no term-tree home, and the asymmetry is principled. An
actor's choice is consequently not decomposable into leaves, and the
honest account of "why this action" under a habit is `q_pi` and the
plan, both already returned.

**The diagnostic exists and already had a face.** The tree G *would*
assign the habit's plan is real and useful — it is the distillation gap
the search-habit slider is read off — and it is
`System.score_policies(params, beliefs, plan)`, which A3 names as the
teacher face and which has taken float plans over continuous actions
since A9. Four lines:

```python
_, _, plan = get_planner("habit")(system.planner_context(params, beliefs, key=k))
habit_G = total(system.score_policies(params, beliefs, plan))
_, tree = system.infer_policies(params, beliefs, planner=cem, plan_key=k)
gap = habit_G.midpoint - total(tree).midpoint.min()
```

Scoring is thereby never silently attached to acting: it is one call,
on the plan you name, through the same closures every planner uses.

**Tried and withdrawn: a per-planner `metrics_default` mark.** Round
one added `register_planner(..., metrics_default=False)` plus
`metrics: bool | None` on the faces, resolving `None` through the mark
(the `needs_key` precedent). It worked, and it was wrong on two counts
worth keeping on the record: it patched the *default* of a toggle whose
**meaning** was the actual question, and it grew a second per-planner
protocol knob to deliver a quantity the library already had a face for.
Deleted: the kwarg, `planner_metrics_default`, the `None` resolution,
and the habit planner's private scoring helper. The general lesson,
since this project has met it before: when a face's default feels wrong
for one implementation, check whether the *semantics* are underspecified
and whether the thing being defaulted into existence is something an
existing face returns better.

### (c) `scoring:` rate keys, and `merge()`

**The key space gains a fifth slot word.** `scoring:<row name>`
addresses a scoring entry, exactly as `recognition:<variable>`
addresses a recognition module (A11's grammar, inherited not
extended): it appears in `_slot_declarations`, so `_write_slots`,
`merge()` namespacing (`"mind.scoring:g_tail"` — the prefix attaches
to the *name* segment, i.e. `scoring:mind.g_tail`, like every other
qualified key) and the validation errors come for free.

- **`epistemic=("scoring:*",)` raises**, with the same argument A11
  makes for `recognition`: scoring parameters are not world claims, so
  they carry no planning-time novelty term. A wildcard that merely
  *ranges over* the slot (`"*:*"`) drops scoring entries silently —
  the `initial_state_prior`/`recognition` precedent.
- **Namespacing, and where it runs out.** The *slot* key namespaces
  like every other (`scoring:g_tail` → `scoring:mind.g_tail`: the
  prefix attaches to the name segment), and so does the variable part
  of a `terms=` key (`occupancy:transition:stoch` →
  `occupancy:transition:mind.stoch`). The **row segment does not** —
  a row name is library vocabulary, not a model name — and that is
  where composition stops: two components' critics would both answer
  to the bare key `g_tail` in the composite, with nothing to
  distinguish them. So `merge()` **raises** on a component that
  declares `terms=` or anything in either half of `scoring`, naming
  the key grammar as the obstacle and pointing at the
  composite as the place to declare the G side (whose namespaced
  factor names then address exactly one component's variables). A
  component's *world model* merges exactly as it did. The same fence
  covers a **policy module** in `policy_prior`, for a different and
  more interesting reason: a module is a conditional prior over its
  own beliefs, so composing two is a question about the joint action
  distribution of two minds, not a log-space addition — `subsystem()`
  refuses to restrict one for the same reason.

**FLAG (3).** A scoring entry is **one** addressable unit: no third
key segment. `scoring["g_tail"]` holds `(critic, slow)` and
`scoring["ensemble_disagreement"]` a 4-tuple, so a `…:critic` /
`…:2` sub-grammar is imaginable, but V1's multiplicity rule is about
*independent posterior blocks*, which these are not, and the only
consumer that needs the critic apart from its slow copy is 2c's
trainer, which holds the tuple in Python anyway.

**FLAG (4).** Rates and `LearnState` inherit the **key grammar**, not
the stepping. `learn` raises a named `NotImplementedError` on a
literal `scoring:` rate key, and `init_learn_state` carries no scoring
entries. Reason: `learn` is the per-tick face and a scoring row has no
per-tick local statistic — a critic's loss is a TD target over
imagined rollouts, an ensemble's a next-step NLL, CRSPP's a ρ-gated
KL, all of them 2c's trainers. Resolving a rate that nothing would
ever apply is exactly the "toggle lies" failure D6 exists to kill, so
the fence is loud and names the trainers that lift it. A12's "rates /
freezing / LearnState / wildcards inherit free" is satisfied in the
sense that costs nothing and claims nothing: the addresses exist and
validate today, and 2c fills them in when it has the loss to spend
them on.

**FLAG (5).** The `merge()` hazard the addendum asks to guard is
**not** "a scoring entry used as a subsystem" — `merge()` takes
`Mapping[str, System]`, so a `SystemParams` in that position is a type
error before any library code runs (it gets a message anyway, naming
the kernel criterion, since it costs one `isinstance`). The real
hazard is the CRSPP: it *is* a declared `System` (A12 item 2), so it
is exactly the thing someone would pass to `merge()` as a component,
which would flatten a group-3 row's private model into the agent's own
declared graph — its factors becoming the agent's factors, planned
over and learned as if they were world claims. The blanket fence above
catches it wherever it is detectable: a component with a non-empty
`System.scoring` raises, so an agent and its own row-owned model
cannot be merged together. What is *not* detectable is merging the
CRSPP alone with some third System while an uninvolved agent points at
it; that is harmless in this library (params are immutable and the
composite gets a copy of the structure), and recorded here rather than
guarded.

**FLAG (6) — RESOLVED against the build (Alex, 2026-08-27; rationale
in `amortization-debt-ledger.md` entry 3).** The build first put the
CRSPP structure in a single static `System.preference_model` field,
reasoning that a row-keyed mapping would be guessing a second
customer's shape. Overruled, and rightly: `scoring` *already has* the
shape, so using it is not a guess. Adopted and implemented:

    System.scoring       = ((row name, declared structure), ...)   static
    SystemParams.scoring = {row name: parameters}

— the same name on both types, which is `System.recognition` /
`SystemParams.recognition` exactly one slot over (A11's rule: the
registry names the strategy, the slot holds the strategy's material).
A tuple of pairs on the static side because static fields must hash
(the `subsystems` precedent; `dict(system.scoring)` is the lookup
form), duplicate row names raising rather than letting the last win.
A row declares that it owns a structure (`needs_structure=True` on its
registry row) and the resolver then insists it finds one; conversely a
structure entry that `terms=` does not mount **raises**, because
mounting is the `terms=` selector's job alone and a structure entry
must never become a second, quieter way to switch a row on.

## A12 implementation note: chunk 2c

*Status: **build-time notes, not adjudication** (Opus, 2026-08-28, chunk
2c). A12's scope split assigned 2c the training side; this section
records the seam ledger entry 5 asked for, the fixture format (so the
oracle-side export can be written to it), and — flagged inline as
**FLAG** — the places where the adjudicated specs underdetermine the
answer and the build took the conservative option for Alex's review.
Nothing here overrides A2–A6, A10, A11, A12 or the addendum; where a
reading was forced, the flag says so.*

### (a) The ledger-5 seam: how the CRSPP posterior reaches the cosine row

Ledger entry 5 named three things that had to arrive together — CRSPP
recognition conditioned on the agent's embedding *as data*, a per-call
belief input to a scoring row, and both defined by the consumer that
holds the observation sequence. They are three separate mechanisms, and
each is the smallest one that does its job:

**1. An external-input parent kind, on `Recognition`.** A11 says a
recognition entry's parents are explicit and that "a parent may be an
externally supplied vector"; the build makes that spelling literal.
`Recognition(variable="stoch", parents=("deter", "external:embed"))`,
with the *width* declared statically in a new
`System.external_inputs: tuple[tuple[str, int], ...]` and the *value*
supplied per call (`infer_states(..., externals={"embed": e})`, and
`inference.amortized.recognize`'s new `externals` argument). The width
is declared so the module's `parent_dims` are still validated at
declaration time; the value is data, which is the whole point. Holding
the agent's *encoder module* in the CRSPP's `recognition` slot would be
A11 refinement (ii)'s pytree-forking trap one level up: two Systems'
params flatten separately, so the "shared" torso forks at the first
optimizer step and drifts apart silently. The kind is general and
CRSPP-agnostic — nothing in `inference/` knows what an external input is
for.

**2. Auxiliary beliefs, keyed by row, passed per call.** `Statics` gains
an `aux_beliefs` field alongside `scoring` (parameters) and `declared`
(structure), narrowed to one row by `_mount_statics` exactly as those
two are; `System.planner_context` and `System.score_policies` gain an
`aux_beliefs=` keyword. `cosine_sim_to_pref_rollout` starts its
preferred trajectory from `aux_beliefs[<its own row name>]` when one is
supplied and from the CRSPP's declared episode start when none is —
which keeps 2b's behaviour as the honest default for a first-tick
diagnostic while making the trainer's path parity-faithful. On a
non-amortized System the argument raises: no row there owns a model.

**3. The producer, in the trainer.** `reference/crspp.py` holds
`observe_no_action` (their scan: carry forward with no action parent,
read the action-free prior, read the posterior at `(deter, embed)`,
carry a straight-through **sample**) and `crspp_recognize`, which vmaps
it over a batch of the agent's embeddings and returns belief tuples in
the CRSPP's declaration order, and
`reference.raif_agent.train_step` threads them in as
`imagine(..., aux_beliefs={"cosine_sim_to_pref_rollout": beliefs})` — keyed by
row name, which is the *caller's* vocabulary: the engine's `imagine` forwards
the mapping to `_amortized_scoring` without reading a key out of it, so no
R-AIF name appears in an engine signature. The agent's encoder is run by
`train.world.embeddings`, on the agent's side of the seam.

Two consequences worth stating. First, the scoring closures are now
built **per start state** inside `imagine`'s `vmap`, because a row may
need *this* start's auxiliary beliefs; under `vmap` the Python runs once
and the tracers carry the batch axis, so the cost is one trace.
Second, `System._amortized_scoring` was extracted so that
`_amortized_context` (planners) and `imagine` (the trainer) consume the
*same* closure builder — A12's habit-metrics rule, structurally: there
is one scoring path and `imagine` is its second consumer, not a second
path.

**FLAG (1).** The CRSPP's *preferred trajectory* is still rolled by the
keyless `predict_beliefs`, i.e. mean-field over the code, while the
agent's imagined rollout samples. Their `img_step_no_action` samples on
both sides. Sampling the preferred rollout means giving the row a key,
which the planner seam does not carry ("the keyless faces stay as they
are"), so the build left it mean-field and recorded the divergence.
`reference.crspp.img_step_no_action` is the sampled step, ready for whoever
lifts it.

*Amendment 2026-09-20 (Alex): lift it.* Two probes measured the divergence
before the ruling: at 100k a −0.12 level shift on the leaf
(`../experiments/raif-mountain-car/pref-rollout-probe.md`); at 500k, on two
seeds, −0.09 and shrinking, with every hypothesis for why sampling would
*help* failing — a keyless mode rollout does not reproduce the sampled leaf
and drifts from its carry, mean-field orders imagined trajectories like the
expected sampled leaf where one draw agrees with another at 0.3, and
mean-field predicts best open loop (`pref-rollout-probe-500k.md`). The
ruling is therefore **not** made on the replication's behalf. It is made
because sampled and mean-field belief rollouts are two general shapes the
library should carry: sampled latent rollouts are the norm of the neural
world-model family (PlaNet, the Dreamers, R-AIF), a sampled carry is
one-particle filtering, and expectation propagation is the norm of tabular
AIF; the engine sits across both, the brief already routes expectations
through pluggable samplers, and with both present their difference becomes
a first-class diagnostic on any model. Costs weighed: the choice is static,
so the mean-field path compiles unchanged and pays nothing; the cost is API
surface, an optional key in the scoring seam (`registry.Statics` carries
it; `imagine` already draws one per start; the keyless faces stay
keyless). Chasing the one remaining R-AIF divergence — the slow early
phase, seeds 0–2 at −100…−125 through 150k against the oracle's −88 by
15k, a training-dynamics effect no checkpoint probe can see — is what the
shape is *used for* first, not why it exists. Build order (Alex): this
change as its own reviewed round, a from-scratch sampled run to 500k
against `crspp_init_100k`'s early phase, the tier-split refactor
(`../tier-split-audit.md`) in parallel with the run.

*As built, 2026-09-21.* The sampled rollout is one function with two carries:
`inference.amortized.predict(..., key=None)` returns the probability tensor
keyless and a straight-through one-hot draw keyed (with one code the key is
used as given, so the trainer's rollouts are bitwise what they were; with
several it is split per code). `update_empirical_prior` and `predict_beliefs`
take `key=` on **every** path — a new `inference/sample.py` collapses a
pushed Categorical, Gaussian or MixtureGaussian onto one draw of its own
family, so a sampled carry is a legal belief for the next tick. The scoring
seam carries the key as `registry.Statics.key`, folded per row so two
model-rolling rows never share a stream; `score_policies(key=)` is the one
planning-side entry, and the cosine row rolls the CRSPP sampled when a key is
present. **A planner's `plan_key` does not reach the rows** — K3 as proposed
would have forwarded it, and the review showed that would have made
`infer_policies`/`act` under a search planner score a sampled rollout by
default, against the ruling that defaults stay expected; the scoring key is a
separate argument (`planner_context(scoring_key=)`). `imagine(sample_scoring=
False)` is the static switch; its key is *derived* from the start's key, not
split off it, so switching it on leaves the actor's draws and the imagined
trajectories bitwise unchanged and isolates the preferred rollout. Two cuts
failed review on the way: the first split the key and moved the trajectories;
the second derived it with `fold_in(k, 1)`, which under partitionable threefry
*is* `split(k, n)[1]` — the actor's step-1 key — so the preferred code and the
imagined code would have shared Gumbel noise; the tag is now one no split
reaches (`imagine.SCORING_KEY_TAG`), pinned by a test. A collapsed categorical
floors its zeros at `LOG_ZERO` so it stays readable; a collapsed Gaussian or
mixture is a point mass and is documented as a carry, not a prior to condition
(a floor scale, or a `condition` robust to zero covariance, is a design choice
left open). `run_mc`'s probe reads the rollout the run trained on. The reference training
step and `run_mc` carry the switch (`--sample-scoring`; absent in an old
`run.json` means expected). Defaults are the expected rollout everywhere
(Alex: no reason to make sampling the trainer's default). Not built: a sampled
planner on the tabular paths — Monte Carlo expected free energy with several
particles per policy — flagged on the roadmap.

*As built, 2026-09-28.* The recognition face joins the rule: `infer_states(key=)`
returns the posterior collapsed onto one draw (`System._collapse_posterior`:
`inference.amortized.collapse_codes` on the amortized path — codes collapse, carries
pass through, one split per factor — and `inference.sample.collapse_beliefs` on the
others); keyless, bitwise what it was. The reference recipe's collection lane acts on
that sample (`Config.act_sampling`, default on), which closed acting-ledger D6's second
half in the recipe; the measurement is `../experiments/raif-mountain-car/`
`actor-entropy-diagnostic.md` §8. The convention is unchanged: absence of a key means
expected, on every face.

### (b) `scoring:` rates and `LearnState` — ledger entry 6, paid

`registry.Estimator` gains one static field, `trainer: str | None`,
naming the trainer that steps the row's parameters. That is what lifts
A12 FLAG (4)'s fence *per row* rather than wholesale: a `scoring:<row>`
rate key resolves and a `LearnState` entry exists exactly when
`trainer` is set (today `g_tail`, `ensemble_disagreement`,
`cosine_sim_to_pref_rollout`), and a key naming a row without one still
raises, saying so and listing the rows that have one. `System.learn`
never sees any of this — it refuses an amortized System outright — so
the addresses reach only the trainers that own the losses.

`init_learn_state` now builds **one carry for the whole agent**: the
world model's slots, `scoring:<row>` per trained row, and the actor's,
under a new bare `ACTOR_KEY = "policy_prior"`. Each trainer steps its
own subset and threads the rest through untouched, which is what keeps
A6's trainers à-la-carte while the carry stays one object to checkpoint.
`_resolve_learn_state` was relaxed from equality to "no *missing*
keys" so a whole-agent carry passes `fit_world`'s staleness check.

Three shapes worth recording, since a scoring entry's shape is the
row's own business (`objectives.amortized.scoring_opt_state`): `g_tail`'s
carry covers the **critic only** (the slow copy is an EMA, and giving it
adam moments nothing writes would be a lie); the ensemble's covers all
four members under one optimizer, as their single `ig_opt` does; and the
CRSPP's is a **nested `LearnState`** over its own slot addresses,
because a row that owns a System owns its addresses too — so
`fit_crspp(..., lr_by_slot={"transition:*": 0.0})` freezes the
preference model's dynamics exactly as it would the world model's.

**FLAG (2).** `policy_prior` takes a *bare* key rather than joining the
slot-qualified grammar. It is not one of the five parameter slots — it
is group 2 of `system.md`'s addendum, a closed-loop factor with exactly
one entry — so there is nothing for a `<slot>:<name>` address to name,
and it is deliberately **not** rate-addressable: the actor has its own
optimizer in R-AIF (`adam 3e-5 clip 100`), which `imagine` takes as an
argument. Putting it in the rate namespace would also put it in the
epistemic namespace, which A11's argument forbids.

**FLAG (3).** The return-normalizer's percentile EMAs ride in a **new
second field**, `LearnState.moments`, rather than in `opt_states`. They
are not optax state, nothing steps them, and `_resolve_learn_state`'s
staleness check is about the optimizer's keys. The alternative — a
separate carry threaded beside `LearnState` — was rejected because A6's
"resumable from its own return value" then needs two objects.

### (c) The fixture format, as implemented

`reference/fixtures/` (a package, so the `.npz` files drop in beside the
loader). One `.npz` per oracle seed with a `provenance.json` **beside
it**, same stem. Eight arrays, each concatenated over episodes along
axis 0, `N` = total agent steps:

    obs               (N, 2)  float32   (position, velocity) — their `state` key
    action            (N, 1)  float32   Box(-1, 1); the action taken AT obs[i]
    reward            (N,)    float32   their per-agent-step reward (-2 typical)
    is_success        (N,)    bool      the goal predicate held at this step
    has_succeeded     (N,)    bool      it has held at some step of this episode
    can_self_imitate  (N,)    bool      expert-authored (True throughout, for
                                        `collect_prior` output)
    is_first          (N,)    bool      first step of an episode
    is_last           (N,)    bool      last step of an episode

`provenance.json`, all seven keys required and all checked:
`{repo, commit_sha, command, seed, collected_at, n_episodes, n_steps}`.
A missing provenance file raises; so does a missing key, and so does a
record whose counts disagree with the arrays (a stale provenance beside
a re-export). Episode boundaries are recoverable from the flags alone,
and `load_fixture` checks that they agree — equal counts, interleaved,
first step first, last step last — so a corrupted export fails at load
rather than becoming a silently mis-split batch. `save_fixture` writes
the same format, so our side has one implementation and the export
script has something to check against.

`rho` is **not** in the format: it is derived
(`reference.revise.pref_labels`) and is computed once per finished episode at
collection time, which is where their `on_episode` callback computes it.
It is carried as a ninth `Episode` column filled by
`reference.revise.label`, because computing it later from a stored
fixed-length window would compute a different quantity — the window's
last step is not the episode's last step.

### (d) Flagged choices in the trainers

**FLAG (4) — RESOLVED (Alex, 2026-08-28): option (b), a second row.**
The build first read the divergence as a *scale*: their config's
`G_scales` looked like `IG −1.0` against our `infogain` term's library
sign of −1 into G, so `imagine(..., g_scales=)` took their per-term
mapping and the reference agent passed `{"infogain": -1.0}`. Both the
diagnosis and the mechanism were wrong, and the paper settles it —
2409.14216 p. 6, not the config:

- **Eq. 6** writes the value as `Σ γ^(t−τ)(r + sim − IG)`, and the prose
  names the last term "the negative information gain `−IG_τ`";
- **Eq. 7b**'s prose says the policy "is trained to reduce the model
  parameter's uncertainty (Equation 7b) or information gain (IG) …
  takes more certain actions to maintain 'homeostasis'".

So R-AIF **penalizes** the ensemble's Jensen gap *by design*: exploration
in their agent is carried by 7c (`occupancy`) and 7a instead, and the
minus sign is a modelling claim about what the agent wants, not a weight
someone happened to set. (The config reading was also just wrong on its
own terms — see the addendum §3 correction: `configs.yaml` says
`ig: 1.0`, and the sign lives in `agent.py:561`'s
`neg_param_info_gain = -igfn(traj)`.)

**Built:** a `model_uncertainty` term of sign **+1** and a
`model_uncertainty` estimator row over the *same estimator function*,
`derived=False`, `parametric=True`, mounted `terms=("model_uncertainty:
<code>",)` with its parameters at `SystemParams.scoring
["model_uncertainty"]`, its rate at `scoring:model_uncertainty`, and the
same ensemble NLL as its trainer. `reference.raif_agent` mounts it with
`epistemic=False` — the curiosity row is **not** in the parity config —
and `g_scales` is gone from `imagine` entirely.

**Why a row and not a knob**, stated because it is the general rule and
not a local preference. A term's sign is registered once, in
`objectives/base.py`, and every consumer of a term tree reads it from
there: `total`, the `derived` fold, a diagnostic naming the leaves, the
lattice in `objective-terms.md`. A trainer kwarg that flipped it would
make G mean two things under one name and force each of those consumers
to be told which — the sign would stop being a property of the model and
become a property of the call. Two rows over one estimator costs one
registration and keeps the property where it belongs; it also makes the
*pair* legible, which a knob never could: one functional, two signs,
exactly like `ambiguity`/`occupancy` (`objective-terms.md` records it as
the lattice's third such pair). And it turns the choice into an
**ablation** — mount either row, or both against separate ensembles —
which is what the goal-(3) slider prods want.

Bookkeeping only, as predicted: the acceptance numbers below are
bit-identical across the change, because `−sign(model_uncertainty)·leaf`
and the old `−sign(infogain)·(−1)·leaf` are the same arithmetic. What
changed is what the term tree is *called*, which is the point.

One guard came with it: mounting both rows against **one** tuple of nets
raises (`objectives.amortized.check_ensemble_ownership`), because the
scoring slot is a pytree and an apparently shared ensemble forks at the
first optimizer step — A11 refinement (ii)'s rule, one slot over.
Separate nets are legal, and `fit_ensemble(..., row=)` says which to
step.

**FLAG (5) — `occupancy` is not mounted by the reference agent.** Their
critic's value (Eq. 6) has no such term: the state prior's entropy is the
*actor's* `wment` bonus (paper Eq. 10's zeta), not part of the value the
critic learns. So `imagine` computes it itself from the rollout's prior
logits and `TERMS` omits the row. Mounting `occupancy` puts it in G *as
well*, and a caller who does that should pass `wment=0.0`;
`imagine`'s docstring says so. The row itself is untouched and remains
available for a MOP-style G, which is what `objective-terms.md` placed
it for.

**FLAG (6) — `g_tail` is not folded twice.** The row is the
horizon-closing leaf of an *explicit-rollout* G; inside imagination the
same critic closes the horizon through the lambda-return. So `imagine`
drives `step_terms_fn` and deliberately ignores `terminal_terms_fn`,
reading the critic from `scoring["g_tail"]` directly.

*[LANDED 2026-09-06 — FLAG (7) is closed; the record below stands as the
adjudication that deferred it. `StreamBuffer` splices episodes onto a flat
step-addressed ring and samples `seq_len` contiguous steps with an `is_first`
column, and `RSSMCore.observe` takes an optional per-step `is_first` that resets
the carry to the learned initial and zeroes the previous action — their
`obs_step` masking exactly. Measured: frozen-tail share of each training window
35.3% → 0.0%, real steps per update 663 → 1024 (their figure). Capacity is now
their `replay_size` 5e5 steps verbatim rather than 5e5/64 sequences. See history
entry 059.]*

**FLAG (7) — episode-aligned windows, so long episodes truncate.**
`StaticBuffer` stores fixed-length windows from each episode's *first*
step, because that is `fit_world`'s documented sequence contract (no
per-step `is_first` reset inside the recurrent scan). Their replay
samples a flat stream and resets per clip through `is_first`. Matching
them means threading an `is_first` argument into `RSSMCore.observe`,
which is a change to the sequence contract rather than to the buffer, so
the build left the contract alone and recorded the cost: with
`batch_length 64` and `TimeLimit(100)`, steps 64-99 of a *failing*
episode are never trained on.

**FLAG (8) — the slow-critic regularizer's weight.** The addendum names
the term ("plus a slow-target logprob regularizer") without a scale;
DreamerV3's `slowreg` is `1.0` and the build uses that, as an argument.

**FLAG (9) — no per-step trajectory weight.** DreamerV3 weights the
actor and critic losses by the cumulative continuation product
(`traj['weight']`). Section 3 mentions the continuation only as the
lambda-return's discount, which is implemented; the weight is not, and
is a one-line addition if the repo has it.

**FLAG (10) — R-S divides by `|max(C)|`.** Section 6 writes `max(C)`,
which is negative on Mountain Car (per-step reward -2, offset +1.1), so
the literal formula returns a *negative* R-S where Table 2 reports
`0.1 +/- 0.0`. `reference/eval.py` divides by the magnitude, which
reproduces a positive "lower is better" number.

**FLAG (11) — RESOLVED BY THE ORACLE (Alex, 2026-08-28).** The build's
first pass read §4's shift as *backward*, reasoning from the fact that
§4 states `pref_label[0] = 0` separately. Wrong, and settled by fact
rather than by argument: A10 disposition 7 makes the code the oracle,
and the code is
`raif-paper/raif/embodied/run/callbacks.py::lookback_on_episode`.
`reference/revise.py::pref_labels` is now a line-for-line port of it, with
`tests/test_self_revision.py::test_the_port_matches_the_numpy_reference`
running the reference implementation verbatim against ours over 64
random episodes (both outputs, lengths 1–12, flags independent — the
shapes hand-written cases do not reach). Four things the file settled,
each of which the earlier reading got wrong or missed:

- **the shift is FORWARD**: `positive_mask =
  concat([has_succeeded[:1], final_imitation_mask[:-1]])`, so
  `positive_mask_t = final_mask_{t−1}`. Slot 0 is filled from
  `has_succeeded[0]` — which is 1 whenever the episode succeeds at its
  very first step — and *then* overridden by the separate
  `pref_label[0] = 0`. That is why §4 lists the override separately: it
  is doing real work on that case, not restating the shift. Read
  forward, ρ rises monotonically into the goal step and the goal step
  itself carries ρ = 1 (the backward reading put the 1 one step early
  and left the goal at 0);
- **the carry's indexing and its seed**: the loop runs `t = T−1 … 1` and
  writes `mask[t−1] = q(max(disc·mask[t], is_success[t]))`, seeded with
  `max(can_self_imitate_gated[T−1], 0)` — so `mask[T−1]` is that seed,
  untouched by the recurrence and unthresholded;
- **the failure decay's exponent is `T−1−t`**, not `T−t`: their
  `[fail]*T`-cumprod-reversed-divided-by-`fail` construction gives
  exponents `T−1 … 0`, so the **last** step of a failed episode is
  condemned at **−1** rather than at `−β`. §4's prose is corrected with
  the citation (the addendum's one contradicted clause; nothing else in
  §4 is touched, it being the pinned target);
- **`can_self_imitate` is OVERWRITTEN by the propagated mask**
  (`data["can_self_imitate"] = final_imitation_mask`), so the callback
  writes *two* keys back and the actor's BC term is gated by a float
  weight in [0, 1] carrying backward credit from a success — not by the
  recorded expert flag. This is the mechanism behind §4's note that
  "the code's `can_self_imitate` picks up *the agent's own* near-success
  steps, not only expert steps"; it was a remark before and is now
  arithmetic. `pref_labels` returns both, `label` writes both,
  `Episode.can_self_imitate` is a float column in memory (the recorded
  flag stays a bool on disk and is cast on load, exactly as `pref_label`
  is simply absent there), and `imagine`'s BC gate reads the mask.

One nuance recorded rather than smoothed: the failure branch's expert
exemption (`neg_rates *= (1 − can_self_imitate)`) reads the
*success-gated* flag, so it can never fire — in a failed episode the
gate has already zeroed every flag, and in a successful one
`failure_decays` is already zero. §4 describes it ("zeroed … on expert
steps") because the line is there. Ported verbatim and pinned by a test,
because it is what the file does and a later change to the gate would
give it teeth.

### (d′) Replay: one module, two regimes (Alex, 2026-08-28)

The chunk shipped a second buffer type — static shapes, a pytree, sampled
inside jit — beside the `ReplayBuffer` workstream G already had, in a
different package, each docstring pointing at the other's. Adjudicated:
**replay is agent-coupled, so it does not live in `data/`; one concept,
one module, two regimes.** Both now sit in `train/buffers.py` —
`ReplayBuffer` (host-side, dynamic schema, ragged episodes, `Dataset` +
`Writer`, pluggable Python samplers, `dump()`) and `StaticBuffer`
(renamed from `SequenceBuffer`, the name now stating the constraint that
earns it) — with one docstring carrying the regime table, the
correspondences (`num_steps` ≈ `seq_len`; `max_steps` ≈
`capacity·seq_len`) and the guidance: *use `ReplayBuffer` unless you need
in-jit sampling*. `hybrid_aif.data` is formats, datasets and writers, and
no longer re-exports either buffer: re-exporting from `train` would
invert the layering, and `data` must never import `train`.

The fence between them, stated once so it is not mis-stated again, is
**host-side/dynamic vs device-side/static — not storage medium.** Both
are in-memory; `ReplayBuffer` samples *for training* too, and its
`dump()` is an optional exit rather than its identity. What
`StaticBuffer` buys is the one thing a dynamic schema cannot give:
`sample(buffer, key, n)` as a pure function of a PRNG key, hence a
training step that samples inside `lax.scan`.

The bridge is one function each way — `as_episode(columns)` (filling in
the self-revision columns a plain collection loop has no opinion about,
and ignoring extras a `ReplayBuffer` legitimately carries) and
`episode_columns(ep)` — so collect-with-one, train-from-the-other is two
lines, which is the concrete payoff of the merge and is what
`tests/test_replay_and_seeding.py` demonstrates end to end on a ragged
episode.

### (d″) The tier audit: which trainers are engine (Alex, 2026-08-28)

Chunk 2c added five trainer-side modules to `train/`, and the tier rule
(`system.md`'s 2026-08-28 addendum: pinned to an external target / free
to churn / never imported by the engine) had not been applied to them
one at a time. Audited, with the criterion stated first: **a trainer is
engine when its loss is stated over *declarations* — a slot, a registry
row, a belief carry — and reference when it is pinned to an external
target.** Verdicts:

- `crspp.py` → **`reference/`**. A5's scope fence draws exactly this
  line: the *seam* (a designation holding a trainable module) is generic
  by construction, and "what stays out is only *additional
  trainers/estimators*". A preference model's ρ-gated KL plus contrastive
  cosine is one of those.
- `revise.py` → **`reference/`**. A verbatim port of someone else's file
  (`callbacks.py::lookback_on_episode`) is pinned-to-external-target by
  definition — its specification *is* fidelity to that file, which is
  clause 1 of the tier.
- `world.py` → engine. Its loss is over the declared slots of any
  recurrent generative model; `RSSMCore` is the assembly it reads them
  through, not a paper.
- `imagine.py` → engine. Actor + critic over `policy_prior` and a
  `g_tail` row, driven by the same scoring closures every planner uses.
  Its one R-AIF-shaped argument was renamed in this pass:
  `crspp_beliefs` → `aux_beliefs`, a mapping **keyed by row name** and
  forwarded untouched, so the engine holds zero R-AIF names in
  signatures (prose examples, correctly pathed at `reference.*`, stay).
- `ensemble.py` → engine. Gaussian NLL for an ensemble of predictive
  heads; which of the two disagreement rows owns the nets is a
  parameter.
- `replay.py` → engine (and see (d′)): the `Episode` format plus two
  buffers, generic over what a trainer reads.

**`Episode`'s R-AIF-shaped flag columns stay in the engine** —
`is_success`, `has_succeeded`, `can_self_imitate`, `pref_label` — with
their provenance documented on the field (Alex's call). They are a
*format*, shared by both buffers and by the fixture loader, and moving
the format to the reference tier would put the engine's buffers below
their own record type. What was tightened instead is the docstring: each
column says what it means generically and cites where the convention
came from.

Consequence worth stating: `tests/test_layering.py` now actively guards
both moved modules — an engine module importing `hybrid_aif.reference.crspp`
or `...revise` fails it. The row's `trainer=` field is a **string
marker**, tested for presence and printed in errors and never imported,
which is what lets the engine-side `cosine_sim_to_pref_rollout` row name
`reference.crspp.fit_crspp` without violating the rule.

### (e) Two bugs the trainers surfaced, both in chunk-2b/1 code

Recorded because neither is visible without training (a third — the
self-revision misreading — is FLAG (11) above):

1. **`ensemble_disagreement` NaN'd under jit.** The member NLL went
   through `MLPConditional.nll` -> `Gaussian.log_prob`, whose triangular
   solve on a 36x36 diagonal Cholesky produced **NaN gradients under
   jit** (finite eagerly). Fixed by adding
   `MLPConditional.diag_nll` — the same number computed on the diagonal,
   which is what `diag_moments` already exists for — and using it in
   `fit_ensemble`. Equality with `nll` is asserted in the tests.
2. **`_diag_entropy` could take `log 0`.** One of its two callers passes
   the ensemble's standard deviation *across members*, which is exactly
   0 wherever four heads agree bitwise. Floored at the dtype's smallest
   normal, which changes nothing wherever the deviation is
   representable.

### (f) Acceptance, measured

Their sizes are 2d's; this ran at a reduced model with raised rates (the
structural numbers are theirs — the objective, the lambda-return, the
signs, the gates, the batch shape's meaning). Seed: 4 bang-bang
episodes, mean return -95.5, SR 1.0, mean length 48.5.

| criterion | measurement |
|---|---|
| (a) losses finite and moving | world 7.19 -> 1.04, ensemble 36.2 -> 14.1, CRSPP 0.58 -> 0.19, actor -54.5 -> -86.6, critic 11.08 -> 8.47; no non-finite step; CRSPP gate 96.9% positive |
| (b) open-loop improves | model/persistence MSE ratio 15.10 -> 0.141 (so from *worse* than persistence to 7x better) |
| (c) critic tracks the lambda-returns | residual 1820.7 -> 3.45 against a constant-predictor baseline of 2526.2; level 49.71 vs 50.22 |
| (d) BC drives log pi(a_expert) up | -1.54 with the term at their 0.5 against -7355.5 with it ablated, same init and same batches |

Criterion (c) needed one substantive choice, flagged in the test: it is
measured with the actor frozen and `discount=0`, i.e. with the bootstrap
off. Two reasons, both measured and both about budget rather than code —
in the joint loop the target *runs away* (the info-gain leaf is a
differential entropy, unbounded below, and grows as the ensemble fits:
ret mean 53 -> 199 over 150 steps), and at their `discount = 1 - 1/333`
the TD fixed point is about 1/(1-gamma) times the per-step value, so
with G about 35/step the target is about 1e4 and the bootstrap lifts it
as fast as the critic rises. Both are faithful to their objective and
are 2d's to live with; what the phase isolates is whether the critic's
head, loss and optimizer fit the returns they are handed.

The whole acceptance file runs in about 55 s. A `slow`-marked twin runs
ten steps of the same loop at their **full `tiny` sizes, their 16x64
batch, `imag_horizon 8` and their rates** — shape parity, not learning —
and is deselected by default (`pytest -m slow`).

## A12 implementation note: chunk 2d

*Status: **build-time notes plus two adjudications** (Opus + Alex,
2026-09-01, chunk 2d), **extended 2026-09-02** by a parity audit against
the oracle's own resolved config (§(e)) which closed divergences 4–8 and
argued a ninth. A12's scope split assigned 2d "the 4-seed 1M-step run
against the oracle — compute, not code". The compute turned up code:
**nine** divergences so far, none of which any test constrained. Seven are
closed, two are kept and **argued here** per A8's rule ("divergence must
be argued in this doc, not discovered in a demo notebook"). Nothing here
overrides A2–A6, A10, A11 or A12; §§5–7 of the addendum are untouched.*

**The audit method, and it is the durable part of this note.** Divergences
1–3 were found by *running* (a smoke curve that unlearned the task).
Divergences 4–8 were found by *reading one file*: the oracle run's own
**resolved** config,
`raif-paper/logs/oracle_state_mc_seed0/config.yaml` — the config the
reference run actually executed, with every default already substituted
and dumped by their own driver. Everything before 2026-09-02 read
`raif/configs.yaml` instead, which is an **inheritance chain** whose
leaves a human resolves by hand; three of the five divergences were
hand-resolutions that had gone wrong, and one (`envs.amount`) was a field
nobody had thought to look up at all. The rule, for the next replication
and for any other target:

> **Resolve the target's config by running it, not by reading it.** A
> config with defaults and overrides is a program; its output is the only
> statement of what it means. Diff the *resolved* dump field by field,
> and treat any field you cannot map as a divergence until you have
> argued otherwise.

§(e) is that diff, carried out to exhaustion — every field of the 197-line
dump, dispositioned — so the config surface of this replication is now
*closed* rather than sampled. `tests/test_oracle_parity.py` is its
executable half: literals transcribed from the dump, asserted against
ours, with the config line in a comment beside each.

### (a) Divergences 1–5: what the smoke surfaced (four closed, one argued)

All three live in the ensemble info-gain estimator — the one leaf whose
magnitude nothing in the suite constrained, because
`test_ensemble_disagreement_matches_its_formula` reimplemented the row's
own expression and so could only ever confirm that the row agreed with
itself. That test now transcribes `raif/aif/epistemics.py` instead, and
a width-independence guard sits beside it.

**1. Summed vs per-dimension-mean entropy — CLOSED (bug).** Their
`_entropy(...).mean(-1)` (`epistemics.py:41-43`) averages the
per-dimension entropies before the `1/sqrt(stoch)` scale; ours summed
them. At their `tiny` sizes the flat code is `32 × 32`, so our leaf was
**1024× too large**. Measured at 51k agent steps:
`model_uncertainty` **+783/step** against `predicted_reward`'s **−1.76**.
Since their `G_scales` are all `1.0`, those two terms are *meant* to be
commensurate; instead reward was 0.2% of G. The consequence was not
subtle — the actor maximised certainty, the cheapest certainty on
Mountain Car is to stand still, and all four seeds learned the task by
≈20k steps and then **unlearned it** back to the −200 floor by 50k.
Post-fix the terms read `model_uncertainty +0.72`, `predicted_reward
−0.90`, `cosine_sim 0.60`, and the λ-return fell from 3340 (critic 318,
never catching it) to 2.4 (critic 2.1).

**2. `ig_head_random_no_learning` — CLOSED (transcribed, inert at
parity).** `epistemics.py:57-61` draws one uniform per member per
gradient step and scales a losing member's loss to `1e-8` instead of
`1.0`, for ensemble diversity. Transcribed into
`train/ensemble.py::learning_gate` with their default rate — **`0.0`**,
per `configs.yaml:106` *and* the oracle's own resolved-config dumps
(`chain_oracle_state_mc_seed{1,2,3}_train*.log:252`). So closing it
changes no number at parity, which is the point: a divergence you cannot
express is a divergence you cannot rule out. A static-zero rate
short-circuits without consuming a key, so the inertness is provable
rather than statistical, and `reference/raif_agent.py` states the value
explicitly instead of inheriting the default.

**3. `mixsigma` — KEPT, and argued (Alex, 2026-09-01).** See below.

**4. The loss reduction over members — CLOSED (Alex, 2026-09-01: "no engine
cost, so the standard applies").** Their last line is
`(scales * jnp.array(losses)).sum()` (`epistemics.py:61`); ours reduced with
`jnp.mean`, i.e. theirs scaled by `1/M`. Now `jnp.sum`, one line. It reads
as bookkeeping — adam's update is `m/(√v+eps)` and so *nearly* invariant to a
global gradient scale — and it was fixed rather than argued precisely because
"nearly" is not "exactly": **`eps` and a global-norm clip are the two places
adam sees the raw scale**, so a 4× loss is a genuine 4× there. Which makes the
next item the reason the change is currently small in practice.

**5. The optimizer groups' `eps`, and `ig_opt`'s `clip` — CLOSED
2026-09-02.** The resolved config's four optimizer lines are

    model_opt:  {clip: 1000.0, eps: 1e-08, lr: 0.0001, opt: adam, wd: 0.0}
    actor_opt:  {clip:  100.0, eps: 1e-05, lr: 3e-05,  opt: adam, wd: 0.0}
    critic_opt: {clip:  100.0, eps: 1e-05, lr: 3e-05,  opt: adam, wd: 0.0}
    ig_opt:     {clip:  100.0, eps: 1e-05, lr: 0.0001, opt: adam, wd: 0.0}

so **`eps` is a group setting and the groups disagree**: `1e-8` on the
world model — which is also optax's default, hence already ours — and
`1e-5`, a thousand times larger, on all three amortized heads. We ran
optax's `1e-8` everywhere and clipped the ensemble not at all. Both halves
are now theirs. The `clip` half also corrects a **false claim** that stood
in `train/ensemble.py`'s docstring ("their `ig_opt` declares none"); the
claim is deleted, not softened.

The engine cost this note previously cited as the reason to wait turned out
to be one function: `distributions.neural.adam(eps)`, a per-slot
`optax.adam(1.0, eps=…)` beside the shared `OPTIMIZER`, which is
**unchanged** — it is the default for slots with no opinion, and one
recipe's group setting has no business in it. `imagine` and `fit_ensemble`
each take an `adam_eps` (their group's value as the default), the reference
recipe passes its own constants (`raif_agent.ACTOR_EPS`, `ENSEMBLE_EPS`,
`ENSEMBLE_CLIP`) so the parity config states what it runs, and `fit_world`
needs nothing because `model_opt` already agrees. `eps` enters adam's
*update* and not its `init`, so no checkpoint migrates.

**Why this mattered enough to close, in one line: with `eps` 1000× smaller
than theirs and no clip at all, our adam was *more* scale-free than theirs,
so item 4 was nearly inert for us while being a real 4× for them.** These
two are what make the `.sum()` reduction bite as it does in their runs.
`test_oracle_parity.py` pins all four groups — including `model_opt`,
**because it already matched**: the one group that needs no per-slot
optimizer is exactly the agreement a later "let's tidy the shared
optimizer" would break silently. The `eps` there is *measured* off the
optimizer (adam's first step on a scalar is `-g/(|g| + eps)`, so one step
identifies it), not read off a constructor argument, because "which number
is in the config" and "which number the optimizer uses" are the two things
that came apart.

### (a′) Divergences 6–9: the resolved-config audit (three closed, one argued)

Found 2026-09-02 by the method above. None was visible in `configs.yaml`
without resolving an inheritance chain by hand, and none was constrained by
a test.

**6. The critic was 3×256; theirs is 2×32 — CLOSED.** The resolved
`critic: {layers: 2, units: 32, dist: symlog_disc, bins: 255, outscale: 0.0}`.

**7. Each ensemble member was 3×256; theirs is 2×32 — CLOSED.** The
resolved `ig_head: {layers: 2, units: 32, dist: normal, outscale: 1.0}`.

6 and 7 are one mistake with one cause, and the cause is sharper than "a
doc was vague": **the addendum contradicted itself, and the build followed
the wrong half.** Its §2 IG-ensemble bullet said `layers 3, units 256`. Its
*very next* bullet — "Sizes" — says that under `tiny` the regexes
`.*layers 2` and `.*units 32` apply, with `actor {layers 3, units 256}`
named as the exception, which is exactly right and is the whole reason the
actor is named separately at all. `raif_agent.py` was built from the first
bullet. The resolved config settles it: `critic: {layers: 2, units: 32}`,
`ig_head: {layers: 2, units: 32}`, `actor: {layers: 3, units: 256}`. Both
bullets now carry a dated correction pointing here.

The cost was large: the critic fell from 361k parameters to 46k and the
four-member ensemble from 3.3M to 422k — ≈8× each, ≈50× on the hidden width,
≈4.4M → ≈1.5M on the whole agent. The fix is per-group sizes in
`reference/raif_agent.py` (`ACTOR_LAYERS/UNITS`, `CRITIC_*`,
`ENSEMBLE_*`), each cited to its own config block, with
`head_layers=`/`head_units=` kept as the *one* knob that moves all three —
that is what tests and smokes want, and it now defaults to `None` ("use each
group's own number") rather than to a size.

Three things worth recording about the shape of this error. First, it is the
`test_ensemble_disagreement_matches_its_formula` failure mode again in a
different costume: a size read from prose that no test compared against the
source. `tests/test_oracle_parity.py` now asserts the widths on the **built
modules**, because the bug was in the *wiring* — one pair of arguments
reaching three constructors — and a test on `ACTOR_UNITS` alone would have
passed while the critic was 256 wide. Second, **the direction is not
obviously bad and that is the danger**: an over-large critic and ensemble
are more expressive, so nothing in a loss curve says "too big". What they
cost is wall clock (§(b′): **1.65×** on the whole cycle) and, for the
ensemble specifically, *disagreement* — the size of the ensemble changes the
very spread the epistemic term is made of, so §(b)'s measured
`σ / spread ≈ 55` was a number about the wrong agent. It has now been
**re-measured** at 2×32 (see §(b)'s dated block): the ratio rises to ≈73 and
the ruling is strengthened — but *not* by the mechanism the first draft of
this paragraph guessed. It predicted smaller heads would disagree more; they
disagree **less** (spread-of-means 0.0032 → 0.0024, member σ unchanged), and
the ratio rose because the denominator shrank. The prediction is corrected
there rather than quietly dropped. Third: when a doc and a config disagree,
**the config that ran wins** — which is this note's audit rule, arrived at
from the other direction.

**8. Their `envs: {amount: 4}` — CLOSED.** They run **four parallel
Mountain Cars per seed** behind one agent, and their `Ratio(64/1024)`
counts the *pool*: four lanes each stepping once is four agent steps, and
16 of those buy one gradient step. We stepped one environment per seed, so
each gradient step saw a quarter of the recent experience theirs sees, from
a single trajectory rather than four. `run_mc` now carries a **lane axis**
inside the seed `vmap` (16 environments in flight at 4 seeds × 4 lanes),
with `--lanes` defaulting to 4 and `--lanes 1` reproducing the old harness.
The gradient arithmetic is unchanged by construction — one block is
`train_every // lanes` = 4 steps per lane, the seed counter still advances
16 per block — and what changes is only how much environment a gradient
step has seen. Each lane is genuinely its own stream (own env state, own
key, own auto-reset, own pending-episode carry, own `label`/routing); the
fixtures still seed the buffers **once per seed**, because a lane is a
collection worker and not a member of the population. Two failure modes a
lane axis invites are asserted directly in `tests/test_run_mc.py`: lanes
that are *copies* (four identical streams at 4× the cost), and a stream
layout that interleaves the lanes, which would splice unrelated steps into
one "episode" while every flag still looked plausible.

**9. `jax.precision: float16` — KEPT, and argued.** Their whole JAX stack
runs in **float16** (`jax: {precision: float16, platform: cpu, jit: true}`);
we run float32 throughout. Kept, for two reasons of different kinds.

The first is that *float32 is a strictly tighter compute representation*.
Reduced precision in DreamerV3-lineage code is a throughput measure with a
correctness cost that its authors then spend machinery containing —
loss-scaling, `float32` accumulators on the reductions that need them,
`eps` floors chosen so that a `√v + eps` does not underflow. The direction
of the divergence is therefore favourable: any behaviour we reproduce, we
reproduce without float16's rounding, and a result that survives at float32
is not an artefact of it. (The converse is the real risk, and it is worth
naming: a target *tuned* at float16 may depend on it. Their `actor_opt` /
`critic_opt` / `ig_opt` `eps: 1e-5` — a thousand times optax's default —
looks exactly like such a dependence, since 1e-8 is below float16's normal
range entirely. We now match that `eps`, so the *tuned* consequence of
their precision is reproduced even though the precision is not, which is
the substantive half of this divergence closed by divergence 5.)

The second reason is scope. Matching float16 is not a knob; it is a
**mixed-precision pass across the stack** — a dtype policy per module, a
loss scaler with its own state in `LearnState`, `float32` master weights,
and an audit of every reduction, `logsumexp`, KL and Cholesky in
`distributions/` and `inference/` for underflow. That is a project with its
own adjudication, and it would land as a pervasive change to engine code
whose current dtype discipline is uniform and simple.

**So: named as a suspect of LAST resort.** If the run misses §7's
landmarks, the order to check is (i) §(b)'s sampled-std estimator, (ii) the
padding fraction of §(c), (iii) anything §(e) leaves marked DIVERGES, and
only then this.

*[LANDED 2026-09-06 — suspects (i) and (ii) are both discharged, and the
ordering was right: (i) WAS the fault, though not as an accuracy difference.
The sampled-std leaf dropped `E[sigma^2]` from the law of total variance, so it
went large NEGATIVE and paid the agent to keep model uncertainty high (fixed
aba6978; history entry 058). (ii) is closed by the flat-stream replay above,
after being measured and largely exonerated as a cause — a converged seed held
expert performance for 500k steps at 41% padding. This float16 clause is
untouched and still last in line.]* A divergence that makes our arithmetic *more* accurate is
the least likely explanation of a worse curve, and it is the most expensive
to close.

### (a″) Divergences 10–13: the transcription sweep (four closed, two formulas confirmed)

*Alex's second 2026-09-02 addendum: "the semantic half". The config audit
(§(a′), §(e)) exhausts what a config can say; it says nothing about whether
a formula was transcribed correctly. The two worst bugs of chunk 2d were
both caught by comparing our expression against a **line-for-line port** of
theirs, and both had been invisible to a test that reimplemented our own
expression. So five not-yet-transcribed hot formulas got that same
treatment. `tests/test_oracle_transcription.py` is the result: each port is
a paste of their source with `tfd`/`nj` swapped for JAX and nothing else
moved, its `file:line` above it, and the resolved config branch named where
their code has an `if`.*

**Two formulas came back exact and are now pinned.**

- **The λ-return recursion — MATCHED.** `agent.py:568-576`. The
  `interm = G + disc·v[1:]·(1−λ)` form, the bootstrap leaf `value[-1]`, the
  reverse recursion and *which* end of the reversed list is dropped are all
  ours already. Pinned against their Python loop on random horizons rather
  than against a re-derivation.
- **`kl_balance` / free bits — MATCHED, clamp order included.** This was the
  reviewer's named risk and it deserved the check: theirs applies
  `jnp.maximum(loss, free)` **inside** `dyn_loss`/`rep_loss`
  (`nets.py:251-276`), i.e. **per term and before** `agent.py:212-219`'s
  `0.5`/`0.1` mix; and because `get_dist` is
  `Independent(OneHotDist(logit), 1)` with `classes: 32`, the KL is
  **summed** over the 32 stochastic variables (and over classes) to one
  scalar per element *before* the clamp. Ours is
  `dyn = max(dyn_raw, 1.0); rep = max(rep_raw, 1.0); 0.5·dyn + 0.1·rep`
  over a `sum(axis=(-2,-1))` KL — the same order, the same reduction. The
  test's small-`scale` case is the one that separates clamp-before from
  clamp-after (both terms under the floor ⇒ total exactly `0.6`), and a
  companion guards the sum-not-mean, which is divergence 1's mirror image.
- **`retnorm`'s percentile EMA — MATCHED**, over a *sequence* of updates
  (one would not distinguish the two live questions). Whole-array
  percentiles; `offset = low`; `invscale = max(1/max, high − low)`;
  update-then-read. And **deliberately un-bias-corrected**: `impl` is
  `perc_ema`, not `perc_ema_corr`, so from zeros the first update gives
  `0.01·p5 / 0.01·p95` and `invscale` sits at its `1.0` floor for many early
  steps. That is a real property of their early training, not a bug to fix,
  and ours reproduces it.

**Four divergences, all closed** (all local edits; the only API surface
added is one optional keyword, named below):

**10. The continuation that discounts the return was a probability; theirs
is a hard mode — CLOSED.** `cont = self.heads['cont'](traj).mode()`
(`agent.py:256`), and `tfd.Bernoulli.mode()` is `probs > 0.5`. Ours passed
`sigmoid` straight through. Two consequences, neither cosmetic: a trajectory
the model believes has ended is **truncated** under theirs (`disc = 0`
exactly) and merely *discounted* under ours — at a cont probability of 0.6,
theirs discounts by 0.997 and ours by 0.598 — and the mode's zero gradient
removes an actor gradient path (`∂ret/∂disc · ∂disc/∂s`) that ours had.
`train/imagine.py::_cont_mode`.

**11. Their per-step `traj['weight']` was not applied — CLOSED.**
`agent.py:257-259` builds
`weight = cumprod(discount · cont, 0) / discount` with
`cont[0] = 1 − is_terminal(s_0)`, and both losses multiply by it:
`loss *= sg(traj['weight'])[:-1]` in the actor (`agent.py:470`) and
`loss = (loss · sg(traj['weight'])).mean()` in the critic
(`agent.py:543`). We took plain means. Now
`train/imagine.py::rollout_weight`. Three things are worth stating because
each is a place this could have gone wrong:

  - It is a **cumulative product**, so one step whose continuation mode is 0
    zeroes that step *and every later one*. Combined with divergence 10 this
    is the mechanism by which a rollout stops contributing past its own
    predicted end.
  - Their **parenthesization** decides which terms it scales. The multiply
    happens after `loss -= actent·ent` and *before* `loss -= wment·wment_ent`,
    so the advantage and the **policy** entropy are weighted and the **prior**
    entropy is not. Ours now matches term for term, which is the whole
    content of `test_the_actor_loss_assembly_is_theirs_on_random_inputs`.
  - `first_cont` is **replay data**, not a prediction, so it is a new
    optional keyword — `imagine(..., start_cont=)` — which
    `raif_agent.train_step` fills from `batch.cont`. **FLAG, and it is the
    highest-risk line of this session:** under our replay a *padded* start is
    exactly a terminal one (`fit_length` freezes the final step and
    `Episode.cont = 1 − is_success`, which stays true down the tail), so
    passing it removes the frozen-at-the-flag starts from both losses. That
    reads as an interaction with §(c)'s padding FLAG rather than an
    accident — theirs never has such a start at all, since it samples a flat
    stream — and it should *reduce* the padding divergence for the
    imagination half rather than amplify it. The exposure is bounded: the
    imagined states after a frozen start are also at the flag, so the head
    zeroes `weight` from `k=1` regardless, and what `start_cont` decides is
    only the `k=0` term of those starts — ≈1/8 of ≈47% of the loss mass at
    convergence. `start_cont=None` (all-ones) is the dial if it needs to be
    A/B'd.

    **Measured, and it settles the flag.** `ImagineMetrics` gained
    `rollout_weight_mean` for exactly this question, and `run_mc` logs it per
    chunk. Over the 2026-09-02 verdict smoke's first chunks it reads 0.545 →
    0.580 → 0.646 → 0.687 and keeps climbing. So the ≈45% of loss mass the
    weight removes at initialization is **an untrained continuation head, not
    a padding tax**: a fresh `cont` head (outscale 1.0, random init) predicts
    termination on roughly half its inputs, and as the world model learns
    what actually ends an episode the weight rises toward 1 on the states
    that do continue. The number to watch on the long run is whether it
    *stops* rising — a weight that plateaus well under 1 at convergence would
    be the padding interaction actually biting, and it is now a logged column
    rather than an argument.

**12. The actor's conditioning state was not stop-gradiented — CLOSED.**
Theirs is explicit and in two places: `policy = lambda s:
self.actor(sg(s)).sample(...)` for the rollout actions (`agent.py:432`) and
`policy = self.actor(sg(traj))` for the entropy (`agent.py:465`). So the
gradient of a later reward reaches the actor's parameters only through the
*direct* action at each step, never through
`a_k → s_{k+1} → (actor input at k+1) → a_{k+1}`. Ours conditioned on the
live scan carry and kept that second path. **Same forward value, materially
different gradient** — which is exactly why no loss-scalar test would have
caught it, and why the fix is one `jax.lax.stop_gradient` in
`roll_batch.step` with their two line numbers beside it. The state chain
itself stays live: reward gradients through the dynamics are untouched.

**13. The slow critic's first update is a hard copy — CLOSED.**
`SlowUpdater.__call__` (`jaxutils.py:481-495`) is
`mix = clip(1.0·(updates == 0) + fraction·(updates % period == 0), 0, 1)`,
so with `slow_critic_fraction 0.02` and `slow_critic_update 1` the **first**
call has `mix = 1.0` — a full copy of the freshly stepped critic — and only
thereafter the 0.02 EMA. We EMA-ed from step one. Since `slow` is built from
its own PRNG key (theirs is too — `nets.MLP(name='slow')`), the `slowreg`
target was an unrelated net pulling the critic at **weight 1.0** for the
≈150 steps a 0.02 EMA needs to catch up. Note what is *not* changed: the
independent initialization stays, because their first *loss* is also
computed against that independent net — the copy happens after the step,
which is why the counter and not the constructor is the place to fix this.

That counter is a second `LearnState.moments` entry, which resolved a
recorded FINDING as a side effect: `run_mc.scan_stable`'s note asked for
either `init_learn_state` to seed the entry or "`train.imagine` owns a
public seeder of exactly this shape". Two keys make the second option
obviously right — `train.imagine.init_moments()` — and what stays open, and
Alex's, is only whether `init_learn_state` should *call* it, since that puts
an `imagine`-owned key in `system/core.py`.

**One setup note the sweep turned up, recorded rather than acted on.** Their
actor head is `Normal(tanh(out), std)` with `std = (maxstd − minstd)·σ(raw +
2) + minstd`, resolved to `minstd 0.1, maxstd 1.0`; ours is
`MLPConditional`'s unbounded mean with `1e-3 + softplus(raw)`. The
*aggregation* matches (both sum over action dims), so the assembly test
above is valid, but the entropy and BC-logprob **levels** are not
comparable. That is divergence 16 in §(f), where it belongs — it is a
declaration's parameterization, not a formula's assembly.

#### The method's own failure mode: a port must come from their CALL SITE

*Added 2026-09-07, after the two execution-trace audits
(`execution-ledger-acting.md`, `execution-ledger-training.md`) audited these
fourteen ports and converged on the same finding from opposite directions.*

**The rule.** A transcription test pins a **formula**. It does not pin
**where that formula is called from**, and it silently inherits whatever the
port's author assumed about that. So a port must be built from their **call
site** — the caller's indexing, its slicing and its argument order
transcribed as literally as the expression is — and not from the expression
alone. Write down the call site's `file:line` beside the definition's.

**Two of the fourteen had exactly that defect**, and both were invisible to a
green test:

- **`_their_information_gain`'s harness** (`test_amortized_objective.py`). The
  port of `epistemics.py:33-45` was faithful. What fed it,
  `_ensemble_moments`, evaluated the four members at *our* rollout indexing —
  `(s_t, a_t)` — because that is the indexing the row is written at. Their
  caller is `igfn = lambda s: self.epistemic_module(s)[1:]`
  (`agent.py:358`) over a 9-step trajectory, so at `G` index `k` theirs reads
  `(s_{k+1}, a_{k+1})`. The test therefore certified the leaf's arithmetic and
  was **structurally incapable of seeing its position** — divergence 20, found
  by a trace and not by the test that existed to catch it. This is the *same
  shape* of mistake as the `means.std(0)` episode, where the port and the row
  shared one wrong expression and the test confirmed only that they agreed
  with each other.
- **`their_dist_normal`'s harness** (`test_oracle_transcription.py`). The
  arithmetic was faithful — `tanh` mean, `+2.0` shift, the std band, biases
  kept, norm-free. The surrounding initializer test asserted `winit: normal`
  for the two head linears, and their `Dist.inner` passes only
  `{outscale, outnorm}` to `Linear` (`nets.py:556-565`), so the block's
  `winit` never reaches them and `Initializer`'s **`uniform`** default applies.
  A test asserting a false fact about their source — divergence 26b.

**Both are now fixed by building the port from the call site**, which is worth
stating as the constructive half of the rule:
`test_the_info_gain_leaf_is_read_one_step_earlier_than_their_call_site_reads_it`
ports `igfn` itself, so both indexings fall out as two slices of one array and
the position is *inside* the test's field of view;
`test_the_dist_head_linears_are_drawn_uniform_as_theirs_are` ports the
`uniform` branch of `Initializer` and discriminates it from the normal branch
by **support** rather than by second moment, the two being identical in
variance by construction.

**Why the rule generalizes.** A formula-only port is a test of *our*
understanding of a function in isolation, and every one of the four defects
this estimator and that head have accumulated lived in the surrounding
context: a reduction (`.mean(-1)` vs a sum), an argument (`preds` vs `means`),
an index (`[1:]`), an initializer default. None of them is inside the
expression. The corollary for a reviewer: when a port is presented, ask *what
was passed to it and where does theirs get that from* — the answer is the part
no green test has checked.

### (b) Argued divergence: the ensemble's `mixsigma` is the std of the member MEANS

Their `__call__` (`epistemics.py:38-40`) draws **one sample per member**
and takes the spread of those samples:

    preds = jnp.asarray([dist.sample(seed=nj.rng()) for dist in dists])
    mixsigma = preds.std(0)

Ours takes the spread of the members' **means**. Ours is theirs in the
zero-member-noise limit; the difference is how much of each head's own
*aleatoric* σ is mixed into a quantity that is supposed to measure
*epistemic* across-member disagreement (their 7b).

**RE-MEASURED at their sizes, 2026-09-02 — the ruling stands and the
argument got stronger.** The two tables below were measured at ensemble
sizes that were **wrong** (3×256 per member; theirs is 2×32 — divergence 7),
so `examples/ensemble_leaf_variants.py` was re-run against the post-fix
verdict smoke's 30,720-step checkpoint at the correct 2×32. The refreshed
numbers, replacing the first table's:

| | seed 0 | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|
| ours (std of means) | −0.787 | −0.778 | −0.781 | −0.777 |
| theirs (std of samples) | −0.0580 | −0.0580 | −0.0580 | −0.0580 |
| ratio theirs/ours | 0.074 | 0.075 | 0.074 | 0.075 |
| absolute gap | 0.729 | 0.720 | 0.723 | 0.719 |
| mean member σ | 0.1732 | 0.1751 | 0.1736 | 0.1744 |
| mean spread-of-means | 0.00226 | 0.00244 | 0.00236 | 0.00247 |
| **σ / spread** | **76** | **72** | **74** | **71** |

Every qualitative claim survives, and the decisive last row moves **in the
ruling's favour**: `σ / spread` rises from ≈55 to ≈73. Theirs is still
exactly flat in σ (−0.0580 at every scale from 0.25× to 4×, across-key sd
≈5e-5), ours still moves with both terms, and the two still coincide exactly
at σ = 0, confirming ours is theirs' zero-noise limit.

**One prediction in this note was wrong, and correcting it is the point of
re-measuring.** An earlier draft of §(a′) reasoned that smaller heads would
*disagree more*. They disagree **less**: the spread-of-means fell from
≈0.0032 to ≈0.0024 while the member σ barely moved (≈0.174 either way). So
the ratio rose because the numerator held and the denominator shrank, not
for the reason given. The direction of the conclusion is unaffected — a
single draw per member is even more overwhelmingly aleatoric at their sizes
than at ours — but the mechanism was mis-stated, and a guess that survives
only because its conclusion happened to hold is exactly the kind of thing
this note exists to catch.

**Still open, and it bears on these numbers:** their members' σ is the
**constant 1.0**, not a learned ≈0.17 (divergence 15, §(f) — `ig_head`
passes no `minstd`/`maxstd`, so their `Dist` defaults collapse the σ map).
At σ ≡ 1.0 the `σ / spread` ratio would be ~400 rather than ~73, which
strengthens the ruling again; but it also means their `avg_ent` term is a
constant and their IG is a different functional from ours in a second way.
The table above is therefore **ours at their sizes**, not theirs.

**Ruling (Alex, 2026-09-01): keep ours.** His rationale, for the record:
*"What's elegant is also often what's best — a better estimator that
simplifies the library."* Three reasons, the third measured:

1. **It estimates the intended quantity with lower variance.** Four
   members × one draw each is a 4-sample estimate of a standard
   deviation. The means are available in closed form; using them removes
   the Monte-Carlo error entirely rather than averaging it down.
2. **It needs no PRNG key at scoring time**, which preserves the
   keyless-face convention (`inference/amortized.py`: "prediction must be
   a pure function of the belief"). Closing this divergence would have
   required a new `StepContext.key` field, a `needs_key` declaration on
   the row, and a key threaded through `score_policies`,
   `evaluate_policies` and CEM's scorer — three planner modules that have
   no key today — or else a silent fallback to the means, which would
   make the leaf's *identity* depend on which face called it. That was
   the engine arbitrariness the ruling declined to buy.
3. **The measured difference is large, and in our favour on
   informativeness.** `examples/ensemble_leaf_variants.py` computes both
   leaves on the same imagined rollouts from 2d's 30,720-step smoke
   checkpoint (4 seeds × 256 start states × 8 rollout steps; theirs
   averaged over 32 key draws):

   | | seed 0 | seed 1 | seed 2 | seed 3 |
   |---|---|---|---|---|
   | ours (std of means) | −0.723 | −0.732 | −0.723 | −0.730 |
   | theirs (std of samples) | −0.0580 | −0.0580 | −0.0580 | −0.0580 |
   | ratio theirs/ours | 0.080 | 0.079 | 0.080 | 0.079 |
   | absolute gap | 0.665 | 0.674 | 0.665 | 0.672 |
   | mean member σ | 0.1745 | 0.1749 | 0.1734 | 0.1739 |
   | mean spread-of-means | 0.0032 | 0.0031 | 0.0032 | 0.0031 |
   | **σ / spread** | **54** | **57** | **54** | **56** |

   The last row is the whole argument. Each head's own σ is ≈55× the
   spread of the heads' means, so a single draw per member is
   overwhelmingly aleatoric noise. The σ-sweep shows what that does —
   scaling every member's σ by 0.25 → 4× (seed 0):

   | σ × | 0.00 | 0.25 | 0.50 | 1.00 | 2.00 | 4.00 |
   |---|---|---|---|---|---|---|
   | ours | 14.407 | −0.478 | −0.600 | −0.723 | −0.845 | −0.968 |
   | theirs | 14.407 | −0.057 | −0.058 | −0.058 | −0.058 | −0.058 |

   At σ = 0 the two coincide exactly, confirming ours is theirs' limit.
   Everywhere else **theirs is flat**: once σ dominates the spread,
   `std_i(μ_i + σ ε_i) → σ·std_i(ε_i)`, so both of their entropy terms
   scale together and the difference collapses to a constant — the
   expected log spread of four standard normals — carrying *no*
   information about disagreement at all. Ours is
   `mean log(spread) − mean log(σ)`, which moves with both. The
   across-key sd of theirs is ≈5e-5 on the batch mean, so this is not a
   sampling artefact of the diagnostic: the two estimators genuinely
   measure different things in the regime the agent is actually in.

**Attributability — and 2d's FIRST SUSPECT.** This is a real behavioural
difference, not a cosmetic one: in G's own units the leaf differs by
≈0.67, against a `predicted_reward` of ≈−0.90. Under theirs the epistemic
term would be ~15× smaller and nearly negligible beside reward; under ours
it is comparable. **If the 1M-step run lags §7's landmarks (off the −200
floor by ~50k, dip near 200k, converged 400–500k), this divergence is the
first thing to check**, by rerunning the diagnostic and, if wanted,
building the sampled variant behind the key seam sketched above. Naming
it here is the point of A8's rule: a gap that is argued in advance is
attributable afterwards, and a replication that beat the oracle on an
unrecorded estimator change would have been an accidental ablation
wearing a result's clothes.

### (b′) What the size fixes cost, measured

The throughput consequence of divergences 6 and 7, because a 50× size
change is not free and the run's wall clock is a planning input. `benchmark`
at parity settings, CPU-only, four seeds vmapped:

| | s/gradient step | agent steps/s per seed | aggregate | vmap(4) ratio | 1M × 4 seeds |
|---|---|---|---|---|---|
| before (3×256 critic + ensemble) | 2.366 | 6.76 | 27.05 | 1.62× | ≈41 h |
| after (their 2×32) | **1.434** | **11.16** | **44.64** | **2.20×** | **≈25 h** |

**1.65× on the whole cycle**, and it lands where it should: the critic is
evaluated at `N·H = 8192` points per gradient step and each ensemble member
at `B·(T−1) = 1008`, so shrinking those two modules from ≈4.0M of the
agent's ≈4.4M parameters to ≈0.5M of its ≈1.5M is most of the work removed.
The `vmap` ratio improving (1.62 → 2.20×) is the expected direction —
smaller per-seed kernels leave more of the machine idle at width 1.

The lane axis (divergence 8) cost nothing measurable: collection is 1.4% of
the cycle at four seeds, and the lanes are a *batch* axis over an env that
was already cheap, so 1024 pooled steps are now 256 scan iterations of width
4 instead of 1024 of width 1. `run_mc`'s module docstring carries the table.

### (b″) The verdict smoke: post-fix, against the pre-fix run and the oracle

30,720 pooled agent steps, 4 seeds × 4 lanes, parity settings, 0.79 h wall
clock. Read at the **same** horizon and with the same statistics as the
pre-fix run and — the comparison that was missing before — as the oracle's
own early curve, recomputed from `logs/oracle_state_mc_seedN/scores.jsonl`.
`best-5k` is the best tiled 5,000-step window, the smoke-scale analogue of
§7's tiled 200k.

| | seed 0 | seed 1 | seed 2 | seed 3 | mean final-100 | mean best-5k |
|---|---|---|---|---|---|---|
| **post-fix** final-100 | −128.5 | −80.0 | −82.5 | −78.9 | **−92.5** ± 20.9 | **−86.8** ± 11.3 |
| pre-fix final-100 | −102.9 | −150.2 | −89.4 | −82.3 | −106.2 ± 26.5 | −83.5 ± 3.9 |
| **oracle** final-100 | −90.4 | −88.5 | −87.3 | −87.8 | **−88.5** ± 1.1 | **−87.6** ± 1.0 |

**The headline: the post-fix run beats the pre-fix run by 13.7 points of
mean final-100 and lands 4.0 points off the oracle's own early curve, on a
statistic where the pre-fix run was 17.7 points off it.** Three of the four
seeds (−78.9, −80.0, −82.5) are *ahead* of every oracle seed, and they get
there earlier: at 12k pooled steps the smoke already reads −82 to −97, which
is the level the oracle reaches at 28–30k.

**What is honestly worse: seed 0, and the variance.** Seed 0 sits at −128.5
in a dip it entered around 15k (−97 → −116 → −131) and was recovering from
by the end (−131 → −128 → −125 → −128). It is the only seed below the
oracle, and it is what makes the post-fix spread (±20.9) worse than the
oracle's (±1.1). Two reasons not to read it as a regression, and one to keep
watching it: the pre-fix run had the same failure on a *different* seed and
worse (seed 1 relapsed to −196 and finished at −150.2, never recovering
inside 30k); the oracle itself dips universally near 200k with full
recoveries, which §7 says the replication should **share**; and 30k is far
too early to distinguish "a dip it recovers from" from "a seed that
unlearned the task", which is precisely what divergence 1 looked like and
why the long run is the test. Seed 0's trajectory is the first thing to read
off `scores.jsonl` at 200k.

Diagnostics at the final chunk, all four seeds, all 120 metric rows finite:
`world` 1.25–1.40 (from 6.26), `predicted_reward` −0.81 to −1.01 against
`model_uncertainty` +0.78 — the two terms **commensurate**, which is what
divergence 1's fix was for and what the `G_scales` of 1.0 assume;
`cosine_sim` ≈+0.59; `critic_value` ≈+0.38 against `ret` ≈1.0–1.55, so the
critic is tracking rather than diverging; `rollout_weight` 0.59–0.73, up
from 0.55 at initialization (see §(a″)'s FLAG).

### (c) The run harness

`reference/run_mc.py` — the resumable 4-seed harness (in-scan auto-reset,
host-side episode boundary / `label` / routing / `alternate`,
`eqx.tree_serialise_leaves` checkpointing with the partial episodes kept,
`scores.jsonl`/`metrics.jsonl`, a `--report` mode printing the
§7-comparable table, and `--benchmark`). Its module docstring carries the
chunk-size and buffer-capacity flags, the measured throughput table, and
one flag worth repeating here: **the padding fraction grows as the agent
improves.** `StaticBuffer` stores episode-aligned windows, so at
`batch_length 64` a converged ≈34-step Mountain Car episode is ≈47%
frozen-at-the-flag padding, and the fraction rises with skill. *[LANDED
2026-09-06: closed by the flat-stream `StreamBuffer`; measured at realistic
38–45-step episodes the share was 35.3% and is now 0.0%. The ≈47% figure here
assumed ≈34-step episodes.]* If
convergence lags §7 and the estimator above checks out, this is the
second suspect.

### (d) The population contract

`tests/test_population.py` — one static `System`, every numeric carrier on
a leading batch axis, so four training seeds are one `jax.vmap`. Every
face passed unmodified (`infer_states`, conjugate and gradient `learn`
with per-member traced rates, `act` under the habit, enumeration and CEM
planners, `update_empirical_prior`, the whole canonical loop,
`StaticBuffer` `add`/`sample`), compared leaf for leaf against sequential
members — `allclose` on floats, **exact** on discrete leaves. One finding:
`init_learn_state` leaves `LearnState.moments` empty while `imagine` adds
its retnorm key on first call, so the whole-agent carry is not
`lax.scan`-stable out of the box; worked around in
`run_mc.scan_stable`, with the two candidate engine fixes recorded there.

*[LANDED 2026-09-14 — `System.init_learn_state` now seeds `moments` declaratively through `objectives.amortized.scoring_moments`, `imagine` raises on a missing entry, and `run_mc.scan_stable` is deleted (ledger entry 14 paid; divergence 29).]*
`system.md` gains a dated paragraph distinguishing **population**
(`vmap`: independent replicas, one declaration) from **composition**
(`merge()`: one merged model, a new System).

### (e) The complete resolved-config diff — the config surface, exhausted

*Alex's first 2026-09-02 addendum, and its motive in his words: "I do have
to wonder whether another prod would reveal other discrepancies." The
oracle's resolved config is finite — 197 lines — so the way to answer that
permanently is to walk all of it. Below, **every** key is dispositioned. The
`seed{1,2,3}` copies were diffed against `seed0` as a cross-check: they
differ in exactly three lines each — `expname`, `logdir`, `seed`. Nothing
else varies between seeds, so one table covers the whole oracle record.*

Dispositions: **✓** matches (with where ours lives), **CLOSED** (fixed —
divergence number in §(a)/§(a′)/§(a″)), **ARGUED** (kept, with where),
**OPEN** (a live divergence, listed and ranked in §(f)), **N/A** (no analog
— with the reason, never as a shrug). Where a whole subtree shares one
reason it gets one row, and says so.

| field | theirs | | ours / why |
|---|---|---|---|
| `G_scales.{rew,crspp,ig}` | 1.0, 1.0, 1.0 | ✓ | no per-term weight; `imagine.term_value` folds at ±1, and the `−IG` sign is the `model_uncertainty` row (adjudicated 2026-08-28) |
| `actent` | 3e-4 | ✓ | `imagine.DEFAULT_ACTENT` |
| `wment` | 3e-4 | ✓ | `imagine.DEFAULT_WMENT` |
| `actor.{layers,units}` | 3, 256 | ✓ | `raif_agent.ACTOR_LAYERS/UNITS` |
| `actor.act` | silu | ✓ | `activation="silu"` |
| `actor.inputs` | `[deter, stoch]` | ✓ | `objectives.amortized.features` = concat(deter, flat stoch) |
| `actor.outscale` | 1.0 | ✓ | final layer at its random init — 1.0 *is* the absence of the knob |
| `actor.symlog_inputs` | false | ✓ | `MLPConditional` never symlogs |
| `actor.norm` | layer | CLOSED 14 | `MLPConditional.from_shapes(norm=True)` — their `linear → LayerNorm(eps 1e-3) → silu`, hidden biases dropped as theirs are |
| `actor.{minstd,maxstd}` | 0.1, 1.0 | CLOSED 16 | `head="bounded"` with `raif_agent.ACTOR_MINSTD/MAXSTD`: σ ∈ (0.1, 1.0) via their `lo + (hi−lo)·σ(raw+2)` |
| `actor.{fan,winit}` | avg, normal | CLOSED 17 | `winit="normal"` — `neural.trunc_normal_avg`, biases zeroed, head halves drawn at their own fans |
| `actor.outnorm` | false | N/A | `Linear._outnorm` is assigned and never read in their `nets.py` — dead config |
| `actor.unimix` | 0.01 | N/A | read only by `Dist`'s `onehot` branch; this action space is continuous |
| `actor_dist_cont` | normal | CLOSED 16 | `head="bounded"`: `Normal(tanh(out), std)`, transcription-tested against their `Dist.inner` |
| `actor_dist_disc`, `actor_grad_disc` | onehot, reinforce | N/A | continuous action space |
| `actor_grad_cont` | backprop | ✓ | reparameterized sample through the rollout |
| `actor_opt.{opt,lr,clip,wd,warmup,lateclip}` | adam, 3e-5, 100, 0, 0, 0 | ✓ | `imagine.DEFAULT_LR/CLIP`; the three zeros are falsy ⇒ their chain adds no wd/warmup/lateclip transform, and ours has none |
| `actor_opt.eps` | 1e-5 | CLOSED 5 | `imagine.DEFAULT_EPS`, via `neural.adam` |
| `batch_size`, `batch_length` | 16, 64 | ✓ | `raif_agent.BATCH_SIZE/BATCH_LENGTH` |
| `cont_head.{act,norm,dist,inputs,layers,units,outscale}` | silu, layer, binary, `[deter,stoch]`, 2, 32, 1.0 | ✓ | `RSSMCore.tiny`'s `cont` head; LayerNorm eps 1e-3 |
| `cont_head.outnorm` | false | N/A | dead config |
| `cont_head.{fan,winit}` | avg, normal | **OPEN, MEASURED** | divergences 17+26 as a pair: attempted 2026-09-07 and reverted — sampled open-loop MSE 0.0088 → 0.0206 over four seeds, losing to the 0.0159 persistence baseline on 4/4. See divergence 17 (it *improved* the probe in isolation: 0.0042 keyless) |
| `critic.{layers,units}` | 2, 32 | CLOSED 6 | `raif_agent.CRITIC_LAYERS/UNITS` — was 3×256 |
| `critic.{dist,bins}` | symlog_disc, 255 | ✓ | `_value_head`; A12 item 5 chose the code over paper Eq. 9 |
| `critic.outscale` | 0.0 | ✓ | `_value_head(out_scale=0.0)` |
| `critic.{act,norm,inputs,symlog_inputs}` | silu, layer, `[deter,stoch]`, false | ✓ | `_value_head` |
| `critic.outnorm` | false | N/A | dead config |
| `critic.{fan,winit}` | avg, normal | **OPEN, MEASURED** | divergences 17+26 as a pair: attempted 2026-09-07 and reverted — sampled open-loop MSE 0.0088 → 0.0206 over four seeds, losing to the 0.0159 persistence baseline on 4/4. See divergence 17. At `outscale: 0.0` their own `Initializer` short-circuits to zeros, so this block's output layer is already theirs either way |
| `critic_opt.{opt,lr,clip,eps,wd,warmup,lateclip}` | identical to `actor_opt`, leaf for leaf | ✓ / CLOSED 5 | the same group; `imagine` takes **one** `adam_eps` and **one** `clip` for the pair, which is why the two cannot drift apart in ours the way two config blocks could in theirs |
| `critic_slowreg` | logprob | ✓ | NLL against `slow.predict()` — the asymmetric form, transcription-tested |
| `data_loaders` | 8 | N/A | their `Batcher` worker count — data-loading concurrency, no numerical effect |
| `decoder.{mlp_keys,mlp_layers,mlp_units,vector_dist,inputs,outscale,act,norm}` | state, 2, 64, symlog_mse, `[deter,stoch]`, 1.0, silu, layer | ✓ | `RSSMCore.tiny`'s `decoder` |
| `decoder.{fan,winit}` | avg, normal | **OPEN, MEASURED** | divergences 17+26 as a pair: attempted 2026-09-07 and reverted — sampled open-loop MSE 0.0088 → 0.0206 over four seeds, losing to the 0.0159 persistence baseline on 4/4. See divergence 17 (0.0181 sampled in isolation — second worst on that probe) |
| `decoder.{cnn,cnn_blocks,cnn_depth,cnn_keys,cnn_sigmoid,image_dist,minres,resize}` | resnet, 0, 16, `^$`, false, mse, 4, stride | N/A (one subtree, one reason) | the image branch. `cnn_keys: ^$` matches no obs key, so their `MultiDecoder._cnn` is built, never called, and allocates no parameters. This is the state-vector run |
| `dyn_loss.{impl,free}` | kl, 1.0 | ✓ | `kl_balance`, clamp before the mix — transcription-tested |
| `rep_loss.{impl,free}` | kl, 1.0 | ✓ | same |
| `encoder.{mlp_keys,mlp_layers,mlp_units,symlog_inputs,act,norm}` | state, 2, 64, true, silu, layer | ✓ | `RSSMCore.tiny`'s `encoder`: `depth = mlp_layers − 1` with `out_activation=True` is their two activated 64-layers, the last activation *being* the embedding |
| `encoder.{fan,winit}` | avg, normal | **OPEN, MEASURED** | divergences 17+26 as a pair, reverted 2026-09-07 — and the per-block splice puts most of the harm HERE and at the GRU: its first layer is `2 → 64`, where fan-avg makes theirs ≈2.3× *smaller* than eqx's, and 0.0274 keyless in isolation |
| `encoder.{cnn,cnn_blocks,cnn_depth,cnn_keys,minres,resize}` | — | N/A (subtree) | image branch, unreached |
| `env.{atari,dm,dmlab,gymrobotics,loconav,metaworld,minecraft,robosuite}.*` (24 leaves) | — | N/A (one subtree) | per-suite kwargs; their `make_env` reads only `config.env[suite]` and the suite is `gym`, so none of these blocks is consulted |
| `envs.amount` | 4 | CLOSED 8 | `run_mc.LANES = 4` |
| `envs.parallel` | process | N/A | ours is a vmapped lane axis in one process — the same four independent streams, and a pure-JAX env gains nothing from a process boundary |
| `envs.restart` | true | N/A | their `RestartOnException` crash-recovery wrapper; a pure-JAX env cannot throw |
| `envs.{reset,checks,discretize}` | true, false, 0 | N/A | **unread in their code** — grepped: only `amount`/`parallel`/`restart` are consumed. The live reset/discretize flags are `wrapper.reset` / `wrapper.discretize` |
| `eval_dir` | `''` | N/A | read only by `train_holdout`; `run.script` is `train` |
| `expname`, `logdir`, `logroot` | — | N/A | paths and labels |
| `filter` | `.*` | N/A | their `TerminalOutput` metric-name regex |
| `rc` | false | N/A | sets EGL env vars for gymrobotics/metaworld only |
| `grad_heads` | `[decoder, reward, cont]` | ✓ | all three differentiated, features not stop-gradiented (`train/world.py`) |
| `horizon` | 333 | ✓ | `imagine.DEFAULT_DISCOUNT = 1 − 1/333` |
| `ig_head.{layers,units}` | 2, 32 | CLOSED 7 | `raif_agent.ENSEMBLE_LAYERS/UNITS` — was 3×256 |
| `ig_head.{inputs,act,outscale}` | `[deter,stoch,action]`, silu, 1.0 | ✓ | `ensemble.transitions` supplies exactly those three parents |
| `ig_head.dist` | normal | CLOSED 15 | `head="bounded"` with `minstd = maxstd = raif_agent.ENSEMBLE_STD`: σ ≡ 1.0 exactly (zero gradient), `tanh` mean — their fixed-σ scaled MSE, and their constant `avg_ent`, both follow |
| `ig_head.norm` | layer | CLOSED 14 | `MLPConditional.from_shapes(norm=True)` |
| `ig_head.{fan,winit}` | avg, normal | CLOSED 17 | `winit="normal"` |
| `ig_head_random_no_learning` | 0.0 | CLOSED 2 | `ensemble.NO_LEARNING` + `learning_gate`, provably inert at this value |
| `ig_models` | 4 | ✓ | `raif_agent.ENSEMBLE_SIZE` |
| `ig_target` | `[stoch]` | ✓ | the target is the next flattened code |
| `ig_opt.{opt,lr,wd,warmup}` | adam, 1e-4, 0, 0 | ✓ | `ensemble.DEFAULT_LR` |
| `ig_opt.{eps,clip}` | 1e-5, 100.0 | CLOSED 5 | `ensemble.DEFAULT_EPS/CLIP` |
| `imag_horizon` | 8 | ✓ | `raif_agent.IMAG_HORIZON` |
| `imag_unroll` | false | ✓ | a real `lax.scan`, not an unrolled one |
| `jax.precision` | float16 | ARGUED 9 | §(a′) divergence 9 — float32, argued as a strictly tighter representation and named a suspect of last resort |
| `jax.jit` | true | ✓ | `eqx.filter_jit` on the chunk |
| `jax.platform` | cpu | ✓ | the oracle ran CPU-only; so does the replication |
| `jax.{policy_devices,train_devices}` | `[0]`, `[0]` | N/A | single device ⇒ their `sync()` and `pmean` paths are inert |
| `jax.{debug,debug_nans,logical_cpus,prealloc,metrics_every}` | false, false, 0, false, 10 | N/A (subtree) | XLA/debug/metric-flush plumbing; no numerical effect at these values |
| `loss_scales.{vector,reward,cont}` | 1.0, 1.0, 1.0 | ✓ | `world_loss`'s three scales |
| `loss_scales.{dyn,rep}` | 0.5, 0.1 | ✓ | `kl_balance` |
| `loss_scales.{actor,critic}` | 1.0, 1.0 | ✓ | no scale applied, which is 1.0 |
| `loss_scales.slowreg` | 1.0 | ✓ | `imagine.DEFAULT_SLOWREG` |
| `loss_scales.contrast` | 1.0 | ✓ | `crspp.DEFAULT_CONTRAST` |
| `loss_scales.actor_refresh` | 0.5 | ✓ | `raif_agent.BC_SCALE` |
| `loss_scales.image` | 1.0 | N/A | image branch absent |
| `model_opt.{opt,lr,clip,eps,wd,warmup,lateclip}` | adam, 1e-4, 1000, 1e-8, 0, 0, 0 | ✓ | `world.DEFAULT_LR/CLIP/EPS`. `eps` **already** agreed (it is optax's default) — asserted in `test_oracle_parity.py` for exactly that reason |
| `replay` | uniform | ✓ | `StaticBuffer.sample` draws uniformly |
| `replay_online` | false | ✓ | no recency bias |
| `replay_size` | 5e5 | ARGUED | `run_mc.CAPACITY = 5e5/64` sequences — the unit change is `run_mc`'s capacity FLAG |
| `retnorm.{impl,perclo,perchi,decay,max}` | perc_ema, 5, 95, 0.99, 1.0 | ✓ | `imagine.perc_ema` / `DEFAULT_RETNORM` — transcription-tested, un-bias-corrected included |
| `return_lambda` | 0.95 | ✓ | `imagine.DEFAULT_LAMBDA` |
| `reward_head.{act,norm,bins,dist,inputs,layers,units}` | silu, layer, 255, symlog_disc, `[deter,stoch]`, 2, 32 | ✓ | `RSSMCore.tiny`'s `reward` head |
| `reward_head.outscale` | 0.0 | ✓ | `out_scale=0.0` — their zero-init, and ours already had it |
| `reward_head.outnorm` | false | N/A | dead config |
| `reward_head.{fan,winit}` | avg, normal | **OPEN, MEASURED** | divergences 17+26 as a pair: attempted 2026-09-07 and reverted — sampled open-loop MSE 0.0088 → 0.0206 over four seeds, losing to the 0.0159 persistence baseline on 4/4. See divergence 17 (it *improved* the probe in isolation: 0.0061 keyless; `outscale: 0.0` as the critic) |
| `rssm.{deter,stoch,classes,units}` | 128, 32, 32, 32 | ✓ | `RSSMCore.tiny` |
| `rssm.unimix` | 0.01 | ✓ | `unimix_logits` on both categorical heads, *before* the KL — which is load-bearing |
| `rssm.initial` | learned | ✓ | `LearnedVector` behind a tanh, unfrozen |
| `rssm.{norm,act,unroll}` | layer, silu, false | ✓ | LayerNorm eps 1e-3 on the GRU input+gate and the prior/posterior trunks; `lax.scan` in `observe` |
| `rssm.action_clip` | 1.0 | ✓ **by the same expression** (2026-09-07) | `controls.soft_clip` *is* their `a *= sg(c / max(c, \|a\|))`, written for a general box and reducing to theirs exactly on `Box(-1, 1)`. The FLAG is discharged, and it had been pointing at the smaller half of the difference: the old `jnp.clip` agreed in *value* on a 1-D box but had a **zero derivative** outside it, so the advantage's pathwise gradient to the actor was discarded on 40.6% of rollout steps at initialization, where theirs keeps it at mean strength `1/\|a\| = 0.68` (training ledger 22). It is now applied where a *transition* consumes the action and nowhere else — `roll_batch`, the habit's internal rollout, the collection loop's belief push, and `to_world_batch` for the replay action — while the recorded column, the BC target and the `ig_head`'s input stay **raw**, as theirs are (acting ledger D5) |
| `rssm.{fan,winit}` | avg, normal | **OPEN, MEASURED** | divergences 17+26 as a pair, reverted 2026-09-07 — and the per-block splice puts most of the harm HERE — the GRU's `1025 → 32` input projection is where fan-avg vs fan-in differs most (≈2.4×), and this block alone takes the keyless probe to 0.0408, 2.56× persistence |
| `run.script` | train | ✓ | their `train_augmented` is the two-buffer loop we replicate |
| `run.train_ratio` | 64.0 | ✓ (and see divergence 18, now adjudicated) | `TRAIN_EVERY = 16` = 64/1024 **pooled** agent steps per gradient step |
| `run.steps` | 2000000 | **OPEN** | `run_mc.DEFAULT_STEPS = 1_000_000` — divergence 19 |
| `run.train_fill` | 0 | N/A | read only by their other scripts; `train_augmented` prefills to `batch_steps` |
| `run.{eval_every,eval_initial,eval_eps,eval_samples,eval_fill}` | 1e6, true, 1, 1, 0 | N/A (subtree) | `train_augmented` has no eval driver at all |
| `run.{log_every,save_every,log_zeros,log_keys_sum,log_keys_mean,log_keys_max,log_keys_video}` | 300, 900, false, `^$`, `(log_entropy)`, `^$`, `[image]` | N/A (subtree) | logging/checkpoint cadence and metric-name regexes |
| `run.sync_every` | 10 | N/A | `agent.sync()` is a no-op when policy and train devices coincide |
| `run.from_checkpoint` | `''` | N/A | resume-from-external-checkpoint path |
| `run.{actor_addr,actor_batch}` | tcp://…, 32 | N/A | their `parallel` script only, which raises `NotImplementedError` |
| `seed` | 0 | N/A | ours keys `jax.random.PRNGKey(seed_offset + s)`, theirs `np.random.default_rng`; a seed is a label, and the streams were never comparable. Four seeds either way |
| `self_imitation.{disc,thr,fail}` | 0.4, 0.001, 0.3 | ✓ | `revise.ALPHA/EPSILON/BETA` |
| `slow_critic_fraction` | 0.02 | ✓ | `imagine.DEFAULT_SLOW_FRACTION` |
| `slow_critic_update` | 1 | ✓ | `imagine.DEFAULT_SLOW_PERIOD` — and the `need_init` branch beside it is CLOSED 13 |
| `task` | gym_mtc | ✓ | `reference/envs/mountain_car.py`, checked line-for-line against their `mtc.py` |
| `wrapper.{length,repeat,reset}` | 100, 2, true | ✓ | `MountainCar.time_limit/action_repeat`; in-scan auto-reset on `is_last` |
| `wrapper.r_reward_offset` | 1.1 | ✓ | `reference/eval.MC_OFFSET` (a metric, not the env's reward) |
| `wrapper.discretize` | 0 | ✓ | falsy ⇒ their `NormalizeAction` (identity on `Box(-1,1)`), not `DiscretizeAction`; ours is continuous |
| `wrapper.checks` | false | N/A | their `CheckSpaces` debug wrapper |
| `wrapper.{delta,stateconcat,stateconcat_target}` | `^$`, `^$`, state | N/A | their `TransitionDelta` / `StateConcatenation` are constructed but `re.match('^$', k)` matches no obs key, so neither adds a key and the third is unread |

**What this table is worth, stated plainly.** The config surface is now
*closed*: every leaf has a disposition and a reason, and the seed
cross-check says the four oracle runs share it. So "would another prod at
the config reveal more?" is answered — no — and the remaining risk has moved
entirely into three places a config cannot describe: the **formulas**
(§(a″), now swept), the **declaration parameterizations** (divergences
14–17, which are *class* facts and not config facts), and the **loop**
(18–19). That is a better place for it to be, because each of those three is
enumerable too.

### (f) OPEN divergences the 2026-09-02 audit surfaced and did **not** close

*Reported for adjudication, not fixed. Every one is either outside the
adjudicated set or contradicts something inside it, and the session that
found them was closing a named list immediately before a 36-hour unattended
run — which is the situation this note already describes as "how a result
stops being attributable". Ranked by suspected behavioural significance;
numbers continue §(a″)'s.*

**18. Their `i % 2` buffer alternation is INERT — the oracle never sampled
M⁺. Highest significance, and it contradicts an adjudicated decision.** This
is the audit's largest finding, and it was verified against their source
directly rather than taken on report:

- `train_augmented`'s inner loop is `for i in range(should_train(step))`,
  and `i % 2 == 0` selects the **general** replay
  (`embodied/run/train.py:158-167`).
- `should_train = Ratio(train_ratio / batch_steps) = Ratio(64/1024) =
  Ratio(0.0625)`, whose `__call__` is
  `repeats = int((step − prev) · ratio); prev += repeats / ratio`
  (`embodied/core/when.py:33-42`).
- `train_step` is a per-transition `on_step` callback, and their driver runs
  the callbacks and *then* increments: `[fn(trn, i) for fn in
  self._on_steps]; step += 1` (`embodied/core/driver.py:70-72`). So `step`
  grows by exactly 1 between consecutive `train_step` calls.
- Therefore `should_train(step)` returns 0 fifteen times and then 1, and
  **never 2 or more**. `i` is always 0. `positive_dataset` is created
  (`train.py:153`) and never `next()`-ed.

So the oracle trained *every* gradient step on `M` — uniform replay over
everything, successes included because they are routed to both buffers — and
never once on `M⁺`. Our harness alternates strictly, which gives successful
episodes and the expert seed data roughly **half** of all gradient steps: a
large, systematic up-weighting of success.

Why it is flagged rather than fixed. Strict alternation is not an accident
of our build, it is **adjudicated** — A10 disposition 9 puts the schedule in
the caller's loop, IDD §I and `raif_agent.train_step`'s docstring both state
it, and `run_mc.alternation` implements it with a documented fallback. A
signed-off decision does not get quietly replaced by a variant; the delta
gets quoted and flagged. Two further reasons for caution: the mechanism
*would* activate at `train_ratio ≥ 128`, so this is a property of **this
config** rather than of their algorithm; and their paper presents the
alternation as a feature, so "their code does not do what their paper says"
is a claim that deserves a human. The A/B is cheap when wanted — a
`use_positive` schedule of all-`False` reproduces the oracle exactly, and
`alternation()` is the single function to change.

*Adjudicated 2026-09-02 (Alex): parity = match the code, not the paper
text.* `run_mc`/`raif_agent` now default to world-model batches from `M`
only (`Config.alternate = False`); `alternate()` stays as the documented
paper-described option, and alternation-ON is the labeled post-parity A/B
(`run_mc --alternate`). Ledger entry 16 (paid); the ruling's record is
`amortization-p0-addendum.md` §7's FLAG-18 note.

**19. `run.steps: 2000000`, against our `DEFAULT_STEPS = 1_000_000`, and it
bears on how any verdict is read.** The oracle ran **2M** agent steps per
seed. §7's landmark is convergence by 400–500k, so 1M should reach the
target ACR — but two of the four oracle seeds place their *best 200k window*
in the second million (1800–2000k, 1600–1800k), and §7's "last-500k" column
therefore means 1.5M–2M for them and 0.5M–1M for us. Either raise `--steps`
to 2000000, or state in the comparison which oracle window a 1M run is being
read against. A ~2× wall-clock decision, so Alex's.

**14. `MLPConditional` has no LayerNorm; their `actor.norm` and
`ig_head.norm` are both `layer`.** Every hidden layer of their MLPs is
`Linear → Norm(layer, eps 1e-3) → silu` (`nets.py:688-689`); ours is
`activation(linear(h))`. This hits **only** the two `MLPConditional` slots —
`MLPPredictor`/`MLPCategorical` (the whole world model, the critic, the
CRSPP) do carry LayerNorm at eps 1e-3 — but those two are the actor (3×256
over a 1152-dim input) and the ensemble members, i.e. exactly the regime
DreamerV3 uses layernorm for. Closing it means adding a `norm=` to a
declaration family, which is an engine change with an adjudication of its
own — and note it also changes bias handling, since their normed layers
carry **no** bias (`Linear._bias = bias and norm == 'none'`).

*Closed 2026-09-02 (adjudicated with 15–17):* `MLPConditional.from_shapes`
grew a `norm=` (default **off**, so no existing declaration moves): hidden
layers run `linear → LayerNorm(eps 1e-3) → act` with no bias, the head layer
stays norm-free with its bias, matching their placement exactly
(transcription test against a verbatim port of their `MLP`+`Dist`). The
parity recipe opts in on the actor and the ensemble.

**15. Their ensemble members' σ is the constant 1.0, and this re-frames
§(b).** `ig_head` passes no `minstd`/`maxstd`, so their `Dist.__init__`
defaults give `minstd = maxstd = 1.0` and hence
`std = (1 − 1)·σ(raw + 2) + 1 ≡ 1.0` exactly, with the mean `tanh(out)` in
(−1, 1) against a one-hot target. Ours learns a per-dimension σ (measured
≈0.17). Three consequences, the third mattering most:

1. their Gaussian NLL is a **fixed-σ scaled MSE** — a different loss from
   ours, so the members fit differently;
2. their `avg_ent` term (`epistemics.py:42`) is a **constant** (1.4189), so
   their IG reduces to `mean_d log(mixsigma_d)/√32` alone, while ours moves
   with both terms;
3. §(b)'s ruling rests on a measured `mean member σ / spread-of-means ≈ 55`,
   and that was measured on **our** σ≈0.17 heads. At σ ≡ 1.0 the ratio is
   ~300. The ruling's *direction* is unaffected — it argues theirs is nearly
   uninformative and ours is not, and a larger ratio strengthens that — but
   the number quoted in it is not theirs, and §(a′) already records that it
   also needs re-measuring at 2×32.

*Closed 2026-09-02:* the parity ensemble members are built with
`head="bounded"` and `minstd = maxstd = raif_agent.ENSEMBLE_STD (1.0)`, which
is σ ≡ 1.0 with an exactly-zero gradient on the σ half — parameters present
and never learned, as theirs are. Consequences 1 and 2 follow without further
code; consequence 3's re-measurement remains §(a′)'s open item.

**16. The actor head's parameterization.** Theirs: mean `tanh(out)` ∈
(−1,1), σ ∈ (0.1, 1.0) via `(maxstd − minstd)·σ(raw + 2) + minstd`. Ours:
unbounded mean with a post-hoc `jnp.clip` to the declared box, σ = `1e-3 +
softplus(raw)` with no ceiling and a near-zero floor. Two behavioural
consequences worth naming on a *sparse-reward* task: our σ can **collapse
toward 1e-3**, removing exploration, where their 0.1 floor forbids it and
`actent` at 3e-4 is far too weak to substitute; and our unbounded mean can
drift outside `[-1, 1]`, where the clip makes the policy effectively
deterministic while the entropy term still reports a large Gaussian entropy
and the BC `log π` is evaluated on the *unclipped* density. Closing it is a
new head parameterization on `MLPConditional` — an engine change (though as
built it alters no existing user: the option defaults off).

*Closed 2026-09-02:* `head="bounded"` — `Normal(tanh(out),
minstd + (maxstd − minstd)·σ(raw + 2.0))`, the `+2.0` shift included,
transcription-tested against a verbatim port of their `Dist.inner`'s
`'normal'` branch (`nets.py:573-580`). The parity actor runs it at their
`(0.1, 1.0)`, and the density, the entropy and the BC `log π` are the bounded
head's own. *[Amended 2026-09-07: the "post-hoc `jnp.clip` at the act site" is
**gone**. A sample `tanh(out) + σ·ε` can still leave the box — 41% of the time
at initialization, and their trained agent's recorded actions run to
`|a| = 2.27` — and what their code does with that mass is three different
things at three sites, none of them a clip at the act site: `ClipAction` into
the env, `action_clip`'s stop-gradiented rescale into the RSSM, and *nothing*
to the recorded column. Ours now matches all three (acting ledger D5,
training ledger 22). The BC `log π` in particular is evaluated at the **raw**
recorded draw, which is what their `start['action']` is.]*

**17. Initialization: `winit: normal` + `fan: avg` on all eight net
blocks.** Theirs is a truncated normal on [−2,2] with
`std = sqrt(scale / mean(fan_in, fan_out)) / 0.87962566`
(`nets.py:757-763`); ours is `eqx.nn.Linear`'s default, uniform on
±1/√fan_in for weight *and* bias. Two differences at once — the
distribution, and fan-avg versus fan-in — worth ~1.3× on the GRU gate and
more on the widest layers. Lowest significance of the group on a 1M-step
run, but it changes the early trajectory and hence the "off the floor by
~50k" landmark, and it interacts with 14: with no norm on the actor, init
scale matters much more there.

*Closed 2026-09-02 — for the two `MLPConditional` slots only:*
`winit="normal"` implements their initializer exactly
(`neural.trunc_normal_avg`: trunc-normal on [−2, 2], `std =
sqrt(1/mean(fans))/0.87962566…`, biases zeroed; the fused head layer's two
halves each drawn at fan_out = child_dim, since theirs are two separate
linears), and the parity actor and ensemble use it. The other blocks —
encoder, decoder, critic, reward/cont heads, the GRU — keep `eqx` defaults.

*ATTEMPTED FOR THE OTHER SIX, MEASURED, AND REVERTED — 2026-09-07.* That
scope note ("out of this adjudication's scope") is superseded: 17 and 26 were
adjudicated as a pair (Alex) and closed together for the `encoder`,
`decoder`, `critic`, `reward_head`, `cont_head` and `rssm` blocks, because
their `Linear` decides the initializer and the bias in one constructor and
closing 26 alone had already been measured to land on a fourth combination
neither codebase has. **The pair made our tiny world model worse**, so both
stay OPEN. Four init seeds, `test_rssm_world.py`'s 300-step fit, nothing but
the initialization varying:

| open-loop MSE | ours (`eqx` default) | theirs (17+26) | persistence |
|---|---|---|---|
| keyless / mean-field, mean of 4 | **0.01134** | 0.01253 | 0.01592 |
| keyless, per seed | 0.0120 / 0.0062 / 0.0193 / 0.0079 | 0.0147 / 0.0149 / 0.0102 / 0.0103 | |
| sampled, mean of 4 | **0.00880** | **0.02063** | 0.01592 |
| sampled, per seed | 0.0089 / 0.0123 / 0.0056 / 0.0083 | 0.0194 / 0.0165 / 0.0223 / 0.0243 | |

The **sampled** path is the decisive one — it is what `imagine()` rolls, and
the sharper instrument. Ours beats persistence on 4/4 seeds (ratios
0.35–0.78); theirs **loses to persistence on 4/4** (1.04–1.53), a mean 2.6×
regression. The keyless path is milder and mixed: 1.37× worse on average,
worse on 3/4 seeds, but with a *tighter* spread than ours.

*Where the harm is, and the best read on why.* Splicing one block at a time
(single seed, so directional only) puts it in the two blocks with the most
lopsided fans — the GRU takes the keyless probe to 0.0408 (2.56× persistence)
by itself and the encoder to 0.0274, while the prior (0.0011), cont (0.0042),
reward (0.0061) and posterior (0.0068) each *improve* it in isolation. So the
`outscale: 0.0` heads are exonerated and this is not a bad initializer, it is
a badly matched one. Fan-avg versus fan-in rescales a layer by
`sqrt(fan_in / mean(fan_in, fan_out))`, ≈1 for a squarish layer and far from
1 for ours: the GRU's input projection is `1025 → 32` (theirs ≈2.4× *larger*)
and its gate `160 → 384` (≈1.33× larger — the very figure this divergence's
text predicted), while the encoder's first layer is `2 → 64` (theirs ≈2.3×
*smaller*). Their config was tuned around a 64×64×3 image encoder, where
fan_in ≈ fan_out and the two conventions nearly agree; a 2-dimensional state
observation and a 1025-wide flat code are the opposite regime. Dropping the
bias compounds it exactly where fan_in is smallest, since `standardize(xW+b)`
and `standardize(xW)` differ most when `xW` has few terms.

*What the probe does and does not settle.* Init affects the training
*trajectory*, not the objective's fixed point, and trajectory is what this
replication keeps failing on — two runs have dipped hard mid-training. A
300-step world-model fit is a guardrail against a gross regression, not a
proof of behavioural equivalence in either direction, and this measurement is
of the **world model alone** (no actor, no critic, no CRSPP, no 1M-step
trajectory). What it establishes is that adopting their initialization is not
free and not obviously right for our sizes; what it cannot establish is that
keeping ours is right either. The honest statement is that the parity gap
here is now *measured* rather than assumed, and that whoever revisits it
should do so on a trajectory, not a probe.

The apparatus (a `winit` argument threaded from `RSSMCore.tiny` through the
six blocks) is not kept — nothing would use it — but the part that took
reading their source is recorded at `neural._linear_stack` so a retry need
not re-derive it: under their `winit: normal` only an `MLP`'s dense `h{i}`
layers get the truncated normal, the **output** layer being a `Dist` whose
`inner` forwards only `{outscale, outnorm}` to `Linear`
(`nets.py:556-565`) and which therefore takes `Initializer`'s *uniform*
default at fan-avg (divergence 26b); the RSSM's `img_stats`/`obs_stats`
likewise take the plain defaults, built with no `**kw` at all
(`nets.py:231-233`); and every bias theirs keeps starts at `jnp.zeros`.

**Four adjacent findings, recorded rather than acted on.** An unrecorded
finding is one that has to be found again.

- **Reward/`cont` indexing — CLOSED 2026-09-07.** In `embodied`,
  `reward[t]`/`is_terminal[t]` accompany `obs[t]` and were *caused by*
  `action[t−1]`; `run_mc` recorded `reward[t]` as the reward from the action
  taken **at** `obs[t]`. So their reward head learned "reward of arriving at
  s" and ours "reward of leaving s". On Mountain Car the two are nearly the
  same function of state, so this looked low on its own — **except** that
  `export_oracle_mc.py` copies the oracle's columns verbatim, so the seed
  fixtures were in *their* convention and misaligned by one step against
  everything the agent collected, inside the buffer one gradient step samples
  from. *[The "look on its own terms" this asked for became the acting-path
  execution ledger, which measured it (stepping our env with their recorded
  actions reproduces their rewards with **exactly zero** error under their
  convention and an error of **2** under ours) and found two more rows of the
  same family: `is_success`/`cont` shifted the same way, and neither of their
  boundary rows recorded at all — so the goal observation never entered our
  replay. All three are now theirs (`run_mc.make_collect_fn`,
  `seed.collect_episode`), the two conventions are one convention, and the
  env-parity test the fixture module had promised for a fortnight exists and
  asserts the misalignment as a negative control.]*
- **Act-time belief is mean-field where theirs is a sample.**
  `inference.amortized`'s keyless faces return the probability tensor by
  design ("prediction must be a pure function of the belief"), so during
  *collection* our actor and GRU see smeared codes while inside `imagine`
  they see straight-through one-hot samples. Theirs samples in both — a
  train/act mismatch ours has and theirs does not. Note it is the same
  keyless-face convention §(b) invoked in *favour* of our `mixsigma`, so
  that convention has a cost as well as a benefit and both belong on the
  record. *[Stands 2026-09-07 — and it had a **second**, separable half that
  did not: our acting path read the episode-start posterior at `deter_0`
  itself, where their `obs_step` masks the carry, zeroes the previous action
  and *then* calls `img_step`. Measured: our step-0 `deter` equalled `deter_0`
  exactly, and the step theirs takes moves it by `max |Δ| = 0.610` on a
  `tanh`-bounded carry — and a GRU is a recursion, so it displaced the whole
  episode. That half was also an internal train/act asymmetry, since
  `RSSMCore.observe` runs `obs_step` at `t = 0`. Closed:
  `run_mc.episode_start_beliefs`. The mean-field-vs-sampled half above is
  untouched.]*
- **The CRSPP's preferred rollout is mean-field** for the same reason, so
  the `cosine_sim` row's *level* is not directly comparable to theirs
  (cosine is scale-invariant but not smearing-invariant).
- **CRSPP gradient scope.** Theirs trains the CRSPP inside `model_opt`'s
  single loss and single global-norm clip, so its loss back-propagates into
  the **shared encoder** and it uses the world model's learned `initial`
  carry. Ours is a separate `fit_crspp` with its own clip, adam state and
  `LearnedVector`, fed pre-computed embeddings with no gradient to the
  encoder. Deliberate per ledger entry 5 — but the *gradient-scope*
  difference is a consequence of that decision that was never written down
  as a divergence, and it should be.

*Tier-split refactor, phase two, as built (2026-09-21; `../tier-split-audit.md`
T1–T6).* The engine's `Episode` lost R-AIF's four columns: it carries `obs`,
`action`, `reward`, a **recorded** `cont`, `is_first`, `is_last` and an
`extras` dict allocated from a per-task `ColumnSpec` schema on the buffers
(T1); `reference/raif/columns.py` holds their four columns, the pad rules,
the on-disk fixture order and `cont = 1 − is_success`. The CRSPP row is
registered from `reference/raif/crspp.py` when the reference tier is imported
(`import hybrid_aif.reference`; the engine cannot import it for you, and a
document naming the row says so when it is missing), and the optimizer-carry shape is
`Estimator.opt_state`, so no engine function switches on a row's name (T2).
`RSSMCore.from_declarations` no longer names their three losses: the decoder
is the encoder's modality, reward and continuation are found by the names
`reward`/`cont`, else by shape (one single-logit binary head, one other
scalar head), else `roles=` names them; `__check_init__` checks the two
scalar heads' widths, and `world_loss` runs each head's own `nll` (T3; the
R-AIF assembly is inferred unchanged, bitwise). The engine's initializer
argument is `init="default" | "trunc_normal_avg"` and the sized constructors
require `width`/`hidden`; the reference tier keeps their `winit` and maps it
once (`world.init_of`) (T4). `imagine` speaks `policy_entropy_scale`,
`prior_entropy_scale`, `bc_scale`, `bc_weights`, with the last two off by
default and R-AIF's values set in `reference/raif/agent.py` (T5). The
ensemble trainer takes `member_weights=` and the random member gate is gone
(their rate was 0.0, the register keeps the record); `transitions` takes the
carry and code arrays; `embeddings` lives with the CRSPP (T6). `alternate`
lives with the harness that offers it. `seed_stats` names its success column. A description recipe now always records `width`; a depth-0 recipe
written before this date lacks it and needs `"width": 0` added to build.

*Phase three, as built (2026-09-21; T9 and the rest of T7).* Fourteen engine
modules had their docstrings and comments rewritten to describe functionality;
every fact removed was copied first, verbatim, into
`../experiments/raif-mountain-car/docstring-salvage-2026-09-21/{neural,trainers,carriers,objectives}.md`
under a heading per symbol, with a note of where else it is recorded. The pass
was proven code-free by a docstring-insensitive AST comparison against the
previous commit on every touched module; the only code tokens changed are error
strings, which no longer quote the replication's config, files or ledgers
(`validation.py`'s "the CRSPP", the amortized path's "their initial: 'learned'",
"their q(stoch | deter, e(o))", "their ig_models: 4", "IDD Alg. 2", "addendum §2"
and the like). Two stale docstrings named by the audit were corrected
(`neural._linear_stack`'s "not kept"; `objectives/base.py`'s `G_scales`), and
several that had gone stale during phase two were caught on the way. Version
0.9.0 marks the three phases together.

## Addendum (2026-09-30): four facts from the docstring salvage

*The 2026-09-21 prose pass copied what it removed from the engine's docstrings
into `../experiments/raif-mountain-car/docstring-salvage-2026-09-21/`. That
folder is now a one-page index saying where each passage's fact is recorded.
Nearly all of them were already recorded here, in the ledgers or in the
register; these four were recorded nowhere but the removed text.*

1. **(A1) Why `inference/amortized.py` is named for its strategy.** A1 makes
   inference a per-family static choice. For the epistemic terms that choice
   has a real dispatch table (`objectives/registry.py`); inference strategies
   have none yet, and the choice is made by declaration type inside
   `System.infer_states`. The module is the amortized row of *that* dispatch,
   one module per row and named after the strategy, so the table can be written
   when a second neural row (A1(a)'s structured filtering) has to sit beside it.
2. **(A2) `System.learn`'s two return forms.** A conjugate-only model gets bare
   params back, which is the v0.4 contract, kept bitwise when gradient slots
   arrived. A model with a gradient slot, or a call that passes `learn_state`,
   gets `(params, LearnState)`.
3. **(A9) The positive list, as built.** A9's "one-member list" still describes
   the LG/mixture path's validator (`objectives/hybrid.py`), where only
   `MLPConditional` accepts a continuous action parent. The amortized path's
   carry (`Deterministic`) and code nodes take the action as an ordinary input
   vector, so they accept one too (`inference/amortized.py::validate_structure`).
   The linear-in-control LG family is still chartered and unbuilt.
4. **(chunk 2c §(e) item 2) The `log 0` in `_diag_entropy` appeared only under
   jit.** XLA's fused reduction returns a hard zero where the eager path returns
   a denormal, so the across-member deviation hit exactly 0 only once the
   trainer was jitted. The leaf then became `-inf`, the folded G `±inf`, and the
   λ-return's percentile normalizer NaN, which poisoned actor and critic in one
   step. The fix is a floor rather than an added epsilon because a floor changes
   nothing wherever the deviation is representable.

*§(c)'s pointer, corrected (2026-09-30):* `run.py`'s module docstring no longer
carries the throughput table or the FLAGs §(c) says it does. The four-seed
throughput row is §(b′)'s table and the one-seed row is in the addendum below.
The chunk-size FLAG's replay lag is the acting ledger's row E6, the lanes FLAG
is §(a′)'s lane-axis paragraph, and capacity is D8 and its addendum in
`../experiments/raif-mountain-car/execution-ledger-acting.md`.

## Addendum (2026-09-30): rationale from the docstring paring

*The 2026-09-30 paring pass cut rationale out of source docstrings. Each fact
below was recorded nowhere but the removed text. The source symbol is named,
and the docstring keeps the contract where one is stated.*

**A1 — the neural conditional family.**

- **`distributions/neural.py`: frozen is a static field, not `stop_gradient`.**
  `frozen` (default `False`) is what makes "no trainable marking" structural.
  `System._slot_learnable` reads it at trace time, so a frozen slot allocates
  no optimizer state, computes no gradient, and comes back from `learn`
  bitwise. A `stop_gradient` still traces the backward pass, still needs an
  optimizer-state entry, and lives at the call site rather than in the
  declaration, where every other capability in the library lives. Freeze with
  `dataclasses.replace(decl, frozen=True)`.
- **`neural.py`, the Gaussian head's scale.** The floor exists because an
  unfloored scale can collapse toward zero on a memorized point, which sends
  the conditional log-density to `+inf` and its gradient to garbage.
  `softplus` rather than `exp` keeps the map smooth and its gradient bounded
  for large negative pre-activations. In the bounded head the `+2.0` shift
  starts the scale near its ceiling at a zero pre-activation (σ(2) ≈ 0.88 of
  the way up), so a fresh head explores.
- **`objectives/hybrid.py::_mlp_step`: why first-order propagation.** The
  neural prediction row on the LG/mixture path propagates `q = N(μ, Σ)` by the
  extended-Kalman step, `μ' = m(μ, a)`, `Σ' = J Σ Jᵀ + diag σ(μ, a)²`,
  `J = ∂m/∂x |_μ`, with the head's own heteroscedastic noise evaluated at the
  same point. It was chosen over sampling and unscented alternatives for
  three reasons. It is deterministic, since the `transition(qs, action)` seam
  carries no PRNG key and prediction must be a pure function of the belief.
  It is exact whenever the net's mean head is affine, which is what makes it
  testable against the LG path. And it costs one extra `jacfwd` rather than
  `2d+1` forward passes.

**The neural pieces of the world model (`distributions/neural.py`).** These
were proposed for `amortization-p0-addendum.md` §§2–3, but that document pins
*their* target. These facts are about why our classes are shaped as they are,
so they are recorded here.

- **Where the loss pieces live (A6).** `kl_balance` with its free-bits floor,
  `symlog_mse_nll`, `symlog_disc_nll` and `binary_nll` live beside the family
  whose parameterization they read, not in the trainer that sums them.
- **`MLPPredictor` is deliberately not a Distribution or a Conditional.** What
  those heads are is a raw output plus a named loss. Treating a 255-way bin
  classifier over a scalar as a density family would buy nothing that
  `symlog_disc_nll` and `symlog_disc_mean` do not already give.
  `symlog_inputs` is on the declaration rather than at the call site for two
  reasons. It is part of what the parametrized function is, since a net fit on
  symlog'd inputs is a different function of its parents. And it is the
  input-side twin of the `symlog_mse`/`symlog_disc` losses, which squash their
  targets. It is off by default and earns its keep only where a parent is an
  observation in unbounded units.
- **`PredictorHead`: the loss name is part of the slot.** Without it the slot
  would be ambiguous, because 255 numbers are either a 255-dimensional
  observation or a binned scalar, and nothing in the trunk says which. The
  three loss names are a positive list, not an exhaustive taxonomy. This is
  the same one-member-at-a-time discipline as the continuous-action
  transition families (A9), so an unknown name raises rather than guessing.
  `symlog` is DreamerV3's `sign(x)·ln(|x|+1)` squash (`symexp` inverts it).
  With it, one fixed loss and one fixed bin range serve targets whose scales
  differ by orders of magnitude, which is why inputs and regression targets
  are both symlog'd.
- **`unimix_logits`: two load-bearing consequences.** No class probability can
  reach zero, so `log_prob` of any one-hot is finite and the posterior/prior
  KL can never diverge. And the gradient of a saturated logit stays bounded,
  which is the stability property the categorical latent was chosen for.
- **`kl_balance`: the rep/dyn asymmetry.** `rep = KL(q ‖ sg(p))` at
  `rep_scale` trains the posterior toward the prior, so the representation
  stays predictable. With the defaults it pulls one fifth as hard as `dyn`,
  and that asymmetry is what stops the posterior from collapsing onto an
  unpredictable code.
- **`twohot`: why a classifier is a regression head.** The gradient is a
  classification gradient, but the decoded expectation is continuous.

**A2 — the gradient learning rule.**

- **`system/core.py::LearnState` is a sibling of `SystemParams`, not a field of
  it.** Parameters and optimizer state have different lifetimes: the first is
  checkpointed, the second restarted. They also have different consumers:
  `as_env` reads params, and nothing but `learn` reads this.
- **`System.learn`: `decay != 1.0` raises on a neural slot.** The leaky-count
  relaxation has no worked-out analogue for network weights. Weight decay
  toward an anchor is a different regularizer with a different justification,
  so the face defers rather than inventing one.

**A9 — `controls.py`: why action spaces cannot mix.** If one controlled
factor is continuous, all of them must be. A flat float vector cannot carry a
level index without either an integer sub-vector (a second representation) or
a float that means an index (a lie that would silently round). Hybrid action
spaces are not supported.

**A11 — the amortized row.**

- **`System.infer_states` with `update_empirical_prior` is one filtering
  step.** `update_empirical_prior` runs the carry and the code prior, the
  action-conditioned half. `infer_states` runs recognition. Together they are
  predict then correct, decomposed along the library's own seam.
- **`factors.py::DerivedInitial` is a named marker rather than an omission.**
  A missing entry would be a hole in the one-declaration-per-factor rule, and
  "derived" is a positive claim about where the number comes from.
  `reduction` names how the transition's distribution is reduced to the
  carried value. `"mode"` is the only one built, because the initial-prior
  seam takes no PRNG key, so `"sample"` would have nothing to draw with.

**A12 item 6 — `reference/raif/seed.py`: why bang-bang works on Mountain Car.**
The car is an undamped oscillator, so pushing *with* the velocity pumps energy
in every step. It is the textbook energy-shaping controller, and deterministic,
which is exactly its weakness as seed data (item 6's ≈22 near-copies).

**Chunk 2d §(b) — `train/ensemble.py::EnsembleLosses.per_member`.** It is
returned rather than folded away because four members converging on the
*same* NLL is the failure mode that silently zeroes the epistemic term, and it
is invisible in the total. It is always the unscaled NLL. What a member's fit
costs is a diagnostic, and multiplying it by its weight would make a
down-weighted member look well fit.

**Chunk 2d §(b′) — the one-seed row of the throughput table.** From `run.py`'s
former module docstring: `benchmark` at parity settings (their `tiny`,
`16 × 64`, `imag_horizon 8`, `LANES = 4`, one gradient step per 16 pooled
agent steps), CPU-only on an Apple M3 Max, the oracle's machine class. One
seed alone ran at 0.787 s per gradient step and 20.33 agent steps/s, against
the four-seed row's 1.434 s and 11.16 per seed. Four seeds one after another
would take ≈55 h against the vmapped ≈25 h. Collection is 1.4% of the cycle at
four seeds and 0.4% at one.

**Chunk 2d §(c) — the run harness (`reference/raif/run.py`).**

- **What resume guarantees, and what it does not.** Every number the loop
  reads is restored: the counters, the episode stream, the replay contents and
  the PRNG keys the process died holding. So the run *continues* rather than
  restarting. Bitwise continuation across a process boundary is not
  guaranteed. It should hold with the same chunk size, JAX version and device,
  since the keys are saved and CPU XLA is deterministic, but nothing asserts
  it. A differently shaped chunk re-splits the key stream, and asserting
  bitwise identity across processes would make an XLA version bump look like
  a data bug. `tests/test_run_mc.py` tests what is claimed: counters,
  finiteness and a continuing trajectory. The oracle lost a seed to host
  reboots twice (P0 addendum §7). Here a reboot costs at most one checkpoint
  interval.
- **`SeedState`: no `crspp_params`, and `train_key` per seed.** The CRSPP's
  parameters *are* `params.scoring["cosine_sim_to_pref_rollout"]`. `Agent`
  returns them separately only for the trainer's convenience, and storing them
  twice is a divergence waiting to happen. `params` and `learn_state` have only
  the seed axis; the collection carriers have a further lane axis. `key` is per
  lane, because each lane's env noise, action draw and reset must be its own.
  `train_key` is per seed, because the gradient step is one event per seed
  however many lanes fed it. Drawing it from a lane's key would make the
  training trajectory depend on the lane count in a way no `--lanes` A/B could
  disentangle. The split itself is the acting ledger's row C21.
- **`probe`: what "the distillation gap" is (a FLAG).** 2d's brief asks for
  one and the adjudicated docs do not define it, so this is the reading the
  build took. It is the gap between the λ-return of the explicit rollout and
  the `g_tail` critic's value at the same states, that is, how far the learned
  value tail sits from the quantity it is fit to. This is A4's honest-tail
  question and chunk 2c's acceptance criterion (c) as a running number. It is
  read through `train.imagine.rollout_returns`, the same rollout arithmetic the
  trainer differentiates, so the diagnostic cannot drift from the objective.
  The reading *not* built is the actor-side gap, habit against an act-time
  search planner (A3/A12 item 3's planner-to-habit distillation). That needs a
  CEM plan per probe state per chunk, which is goal (3)'s slider measurement
  rather than 2d's, and it belongs beside the ablation that mounts a search
  planner.
- **`_probe_batch`.** It is fixed for the whole run so the gap is a trajectory
  rather than a fresh draw's noise. It is drawn from M⁺, the seed
  distribution, so the gap is measured on states the agent is meant to have
  learned to prefer rather than wherever it happens to be.
- **`_console` flushes.** Python block-buffers stdout when it is not a tty,
  so a 40-hour run's status lines would sit in a 4 KB buffer for tens of
  minutes. During 2d's own smoke this was measured as "the run looks hung".
  The status line is the only live signal a long run has.
- **`train`: `run.json` records the commit and `dirty`.** The config alone
  does not identify a run. A two-day trajectory is attributable to a commit
  only if the commit is written down beside it. This was added 2026-09-05,
  when the first 2M run's post-mortem had to reconstruct from the calendar
  which scoring code produced it. `dirty` matters as much as the SHA, since a
  run launched from an edited tree is reproducible only by luck.
- **`parse`: a knob absent from a stored `run.json` is not defaulted.** The
  current default describes what a run launched today would build, while the
  checkpoint was built by whatever the code did then. `winit` cost a failed
  resume this way. It was absent because the flag postdated the run, so it
  defaulted to `"normal"` and deserialised post-17+26 shapes onto a
  pre-17+26 checkpoint. That failed loudly only because the shapes moved. A
  knob that changes values and not shapes (an eps, an outscale, a unimix)
  would have resumed silently into a different agent, which is divergence
  29's hazard class exactly. So absence is tolerated only where it has a
  known historical meaning, and the value recorded is what the code did
  before the knob existed, never the field's current default. Anything else
  refuses.
- **`train`: the actor channel (added 2026-09-06).** Every one of these series
  was already computed in `ImagineMetrics` and never written out. That is why
  the first 2M run's dip could be characterised only through `bc_logprob`, a
  derived symptom (a confident actor far from its imitation target scores
  about −50) rather than the quantities that would say why. The second run's
  dip went to the −200 floor for ~150k steps per seed, where the oracle's
  mid-run dip reaches only −88 (their `episode/score`, 750k–1M), and these
  series exist to explain that gap. `policy_entropy` and `bc_weight` say
  whether the policy collapsed to its σ floor and how much of the batch still
  carried imitation weight. `advantage` against `retnorm_low/high` says
  whether the actor's learning signal blew up because the return normaliser's
  percentile band collapsed underneath it.
- **`train`: G's per-term decomposition is logged.** It is the term tree's own
  diagnostic content (chunk 2c: "folded but not lost"). 2d's first question
  about a curve that will not converge is which term dominated G, and an
  unlogged run cannot answer it.
