# hybrid-aif: project brief

*For team review. Companion docs: `../README.md` (project spec),
`scoping-review.md` (full analysis of reference repos & papers),
`de-risk-report.md` (deep-dives on the three hardest design items, with
citations). This brief summarizes where we've landed and what's still open.*

## Why build this

`pymdp` handles discrete active inference well; this project explores the
complementary **hybrid continuous-discrete** (mixture-model) setting, which, to
our knowledge, existing tools do not yet cover. Two findings make it matter:

1. **Hybrid structure is where continuous epistemic AIF begins.** Koudahl et
   al. (2021) prove the epistemic (information-gain) term of expected free
   energy is *constant* in linear-Gaussian systems with additive control — a
   purely Gaussian continuous agent has nothing to be curious about.
   Switching/mixture structure is the minimal model class that restores a
   genuine epistemic drive. GMM/MNLR nodes aren't a checkbox; they're the
   point.

2. **Nobody has done the core composition.** Existing hybrid-AIF work (Parr &
   Friston 2018; Collis et al. 2024; Priorelli & Stoianov 2025) sidesteps the
   hard part by planning only at the discrete level. No published system
   propagates fixed-K mixture beliefs through policy rollouts with a
   principled reduction schedule and mixture-entropy info-gain estimators.
   The ingredients are all validated, decades-old machinery — Gaussian-sum
   filtering and mixture reduction from target tracking, pairwise entropy
   bounds from information theory, Pólya-Gamma augmentation from Bayesian
   statistics — but their composition under `jit`/`vmap` with differentiable
   EFE is open territory. Low implementation risk, genuine novelty.

There is also no JAX library to lean on for the kernels (dynamax recently
added RBPF-SLDS *filtering*, but no variational SLDS learning, smoothing, or
planning) — we write them, and they're small.

### Landscape check (July 2026)

We surveyed the full open-source AIF landscape — ActiveInference.jl,
RxInfer/ReactiveBayes, SPM/DEM, cpp-AIF, the deep-AIF repo family, robotics
controllers (AIC/u-AIC, AIMAPP), and VERSES' AXIOM (source-level) — to confirm
the gap and harvest ideas. Verdict: **the core bet stands everywhere.**
ActiveInference.jl is discrete-only (its in-progress v0.2 adds recursive tree
search, still discrete). RxInfer has mixture nodes, PG-augmented multinomial
regression, and an experimental rSLDS example, but no EFE function and no
reduction schedule. SPM's mixed models plan only at the discrete level.
AXIOM's planner is sampled CEM with Dirichlet-parameter novelty — no mixture
propagation, no state-epistemic term, no per-block control. Strategic caveat:
VERSES is pointing the same conjugate-mixture lineage at robotics behind an
academic-only license (AXIOM, CAVI-CMN) and is patenting in this space — our
levers are execution speed and a permissive license. Adjacent greenfield the
survey exposed: no MuJoCo/Isaac AIF bridge exists, and no GPU-native AIF
outside Python.

### The obvious objection: "why not just glue existing libraries together?"

There is a fair argument for why no hybrid AIF library exists: AIF planning
is most natural over discrete supports, and the standard template bolts
continuous perception onto a discrete planner — VAE/encoder periphery in
front, discrete AIF behind. R-AIF and SPM's mixed models are both instances.
So why not NumPyro for continuous inference, pymdp for discrete planning,
and duct tape between them?

Three responses, and an empirical posture:

1. **The duct tape is the product.** The discrete↔continuous boundary — BMA
   messages down, integrated evidence up, mixture collapse, precision
   coupling — is precisely what every bolt-together project re-derives by
   hand (SPM's `spm_MDP_DEM`, Priorelli's interface messages, and Collis'
   rSLDS sandwich are three hand-built instances of the same joint). We will
   instantiate this pattern many times across projects; the reusable,
   tested version of the boundary is the thing worth building.
2. **Some things the sandwich cannot do.** EFE's epistemic terms live in the
   *joint* hybrid belief: a discrete planner over encoder codes discards
   exactly the continuous uncertainty the info-gain term needs (and per
   Koudahl et al., the purely-Gaussian slices contribute no epistemic drive
   at all). Mixture-belief rollouts, term trees spanning the boundary, and
   end-to-end gradients through planning are structural capabilities, not
   integration details.
3. **We reuse wheels where they exist.** NumPyro stays as the sampling
   backend, the pymdp-jax skeleton is forked rather than rewritten, and the
   mixture machinery is imported from fifty years of tracking literature.
   New code concentrates exactly where the survey found nothing to reuse.

Posture: v0.1 will show empirically how much is genuinely duct tape — where
gluing is honest, we glue; we don't prejudge.

## What the library is

A JAX package for specifying and running active-inference (and ablated:
model-based RL) agents built from a **typed node menu** — Categorical,
Linear-Gaussian, GMM, MNLR, custom — composed in hierarchical templates,
with policy search vmapped across both policies and agents.

Key design commitments (rationale in the companion docs):

- **`System` as the single abstraction**: anything on either side of a Markov
  blanket. An env *is* a `System`; systems nest and compose; the outermost
  boundary interfaces with external environments (robot hardware, simulators,
  other software). **Composition is a first-class theme, not a feature**:
  hierarchy and multi-agent influence are one mechanism at two timescale
  tiers (`design/nesting-systems.md`), and its instruments — `couple()` /
  `subsystem()` and the tick-semantics `driver` — live together under
  `hybrid_aif.composition` (v0.7).
- **Per-node objective control and metrics.** Every node carries a mean-field
  block id; epistemic/KL terms toggle per block, and EFE/VFE come back as
  structured term trees (per-variable × per-policy × per-term) on demand.
  Where estimators are approximate, the API says so — mixture entropies
  return closed-form differentiable *lower/upper bound pairs*
  (Kolchinsky–Tracey), a bias-honesty feature no other AIF library has.
  The Eindhoven constrained-FFG formalism (van de Laar et al.,
  arXiv:2306.08014) is the theoretical grounding for exactly this per-block
  objective structure, and their "epistemic priors" recasting of EFE as VFE
  (arXiv:2504.14898) is worth supporting as an alternative formulation and
  correctness cross-check for our explicit rollouts — with a strategic bonus:
  it flattens planning into *pure inference*, the form that sampling-based
  hardware acceleration (where expectations are cheap but the classical
  info-gain term is not) can exploit.
- **Explicit policy search first** (enumeration for discrete; CEM/MPPI/
  gradient for continuous; sophisticated-inference tree search and MCTS via
  mctx), amortized actor-critic as an optional backend later. Explicit search
  is what keeps per-term toggling and metrics trivially correct.
- **Conjugate closed forms as fast paths, never as architecture.**
  Categorical–Dirichlet, Linear-Gaussian–NIW, GMM via indicator CAVI, and
  MNLR via Pólya-Gamma-augmented CAVI — where, remarkably, variational
  inference needs *no PG sampling at all*: the entire update is a `tanh`
  expression plus Gaussian linear algebra, with a monotone-ELBO guarantee
  (Durante & Rigon 2019), and a working pure-JAX precedent exists (CAVI-CMN,
  from the group behind pymdp). Everything else: optax gradients or NumPyro
  NUTS on exact likelihoods.
- **Mixture rollouts with static shapes.** Propagate all K·M components
  exactly, reduce back to K by Runnalls' KL-bound merge (fixed trip count —
  jit-safe) or, when the mixture comes from a declared discrete mode factor,
  the IMM schedule (no argmin, gradient-friendly). Moment-matched collapse as
  the cheap ablation. Correctness is testable against exact enumeration on
  small problems, with moment-preservation as an exact invariant.
- **Model representation: typed Python spec + versioned JSON view.** Pydantic
  `NodeSpec` classes are the source of truth (spec/compile layer only —
  validated once at load, compiled to equinox pytrees; nothing pydantic on
  the runtime hot path, and the JSON Schema, not the library, is the
  contract, so the validator is swappable if load time ever matters); a
  generated, versioned JSON
  schema lets external tools author models directly without touching Python;
  custom functions enter by registry reference, failing loudly if unregistered
  (*as built, 0.14:* a registered, namespaced name such as `"drive.unicycle"`,
  `register_transition`, rather than an import path such as `"pkg.mod:Thing"`:
  a transition record can only name code the reading process installed, and
  loading it imports nothing; a `Deterministic` map's `"impl"` is still an import
  path, imported at build). This is the pattern every production
  factor-graph/PPL shop converged on (RxInferServer, PyMC ModelBuilder,
  Infer.NET). ONNX is out (wrong semantics); NumPyro provides no model
  representation (confirmed) but compiles easily *from* our spec.
- **Behavioral model fitting as a first-class capability.** Fitting agent
  parameters to empirical action sequences — priors over parameters,
  MCMC/VI via NumPyro, parameter-recovery validation, model comparison — is
  the entire raison d'être of ActiveInference.jl (the Mathys group's Julia
  package) and a workflow beyond pymdp's current scope. Our agents are JAX-native
  with differentiable action likelihoods, so a `fit(spec, behavior, priors)`
  API is nearly free and opens the computational-psychiatry user base
  alongside robotics. Keep the fitting stack an optional extra so the core
  stays lean.
- **Modern-agent expressiveness.** The framework should be able to *build*
  R-AIF-style agents (learned preference priors — any node can carry a
  "phenotypic prior"/homeostatic target, with online preference learning as
  an option; this includes terminal/goal-state priors — SPM's `H` vectors,
  goal attractors distinct from per-step preferences — as first-class members
  of the preference-node family, not an afterthought) and JEPA-style agents (latent-space prediction objectives —
  AIF has always in principle allowed moving the prediction-error term to
  latent space; the extra ingredient is an anti-collapse regularizer, and
  SIGReg is a one-hyperparameter, provably-sufficient candidate that ports
  to JAX trivially). A July 2026 theory paper (Arnez & Gomez-Villa,
  arXiv:2607.13612, in `/papers`; proofs machine-verified in Lean) makes
  this precise: the JEPA objective is an *exact* AIF variational free energy
  iff the regularizer is SIGReg (VICReg/LogDet carry an irreducible
  distortion — and their "safe" fallback tier is the same Kolchinsky–Tracey
  estimator we adopt for mixture entropies), and the one EFE term no JEPA
  world model computes is the **state-epistemic value**, a policy-conditioned
  future-state coverage signal. Computing policy-conditioned latent entropies
  is exactly what our mixture-belief machinery does — and the paper's
  empirical predictions are explicitly left untested, so the first
  implementation is up for grabs. Combining this with learned preferences
  *in* the latent space is likewise unexplored; both are research
  opportunities that fall out of the architecture rather than blockers.

## What we're reusing

- **pymdp (JAX)** — build on its *architecture*: equinox Agent-as-pytree, vmap
  layering (policies × agents), `lax.scan` rollouts with masked fixed-window
  history (a worked solution to JAX static shapes), the `policy_search` hook,
  SI tree + mctx planners, and the test methodology. Their discrete test
  suite is our parity oracle. Cross-factor dynamics are supported there
  (`B_dependencies`) and we inherit that.
- **stable-worldmodel** — the data pipeline design: ~130-line format registry,
  Lance episode schema (fast, S3/HF-streamable, interoperable with their CLI
  tooling), clip-index windowing, ReplayBuffer = Dataset + Writer, HER-style
  goal relabeling. Also its protocol layering (`Dynamics`/`Objective`/
  `Solver`) as the shape of our planner interfaces.
- **PGMax** — pattern reference only (tempered log-space numerics,
  scan + checkpoint discipline).
- Numerics to import day one: pymdp's stable entropy/Dirichlet-KL, PGMax's
  logsumexp utilities, R-AIF's KL-balancing/free-bits/symlog as optional node
  stabilizers, `jax.checkpoint` for differentiable planning.

## Integration requirements

Three integration requirements beyond the core plan. Each lands as a
**contract on an interface the plan already builds**, not as a new
subsystem, and none changes v0.1.

### Frozen pretrained world models (V-JEPA-class)

The encoder lives **on the System boundary, owned by the System** — a
Markov blanket has sensory states, and a frozen encoder is one. Killing the
perception↔planning stitching boilerplate is part of this library's reason
to exist, so frozen models are wrapped as first-class **transducers**:
boundary components declared in the model spec like any node (`impl` by
registry reference, declared output shape, declared uncertainty model). The
System runtime owns their execution, batching, and logging — latents land
in dataset columns automatically. What keeps this from becoming "run
arbitrary ML models": a transducer has **no inference semantics**. The
engine never inverts it, differentiates through it, or trains it (only its
precision is learnable). "Is the latent an observation or a belief?" then
has a principled answer: it arrives as an observation on a latent modality —
the pseudo-observation reading we already committed to, and a *frozen*
encoder is literally the constant-noise encoder model of the SIGReg
correspondence — and becomes a belief by uncertainty injection through
machinery we already have: fixed/learned precision on the pseudo-observation
node (the default), or input-dependent precision from encoder
ensemble/dropout passes (workstream H).

Execution cost is dominated by a useful asymmetry: perception runs **once
per real tick**, while planning rollouts run thousands of times per tick in
latent space and never touch the encoder. So torch/ONNX frozen encoders run
host-side at the tick boundary, sequenced with the jitted
inference/planning step by the System runtime — no JAX port required. Only
a frozen *predictor* (latent dynamics) used inside imagination must be
JAX-native or pay a per-step callback; it enters as a custom transition map
with the Petersen quadrature/linearization rules (see open question 12).
Adapters follow the format-registry pattern that already bounds the data
pipeline: the library ships a small blessed set (V-JEPA-class first), the
contract is library-owned, and a user adapter is ~20 lines
(stable-worldmodel's `MegaWrapper` pipeline and `*_emb` conventions are the
working precedent). Carry-forward caution: frozen encoders are the
configuration stable-worldmodel showed most brittle under distribution
shift (94%→12% by eval provenance) — a reason *for* workstream J's
robustness protocol, not against the adapter.

**Teacher–student distillation** (bootstrapping a light AIF planner from a
heavy world model) is the amortized backend already on the roadmap
(habit-net distillation from visit counts; the SIGReg paper's Prop. 7
amortized-EFE policy). What it concretely adds to build requirements is
small: (i) a **teacher-trace export contract** — rollouts log beliefs,
actions, and per-policy EFE term trees as ordinary dataset columns, which
the metrics trees + format registry make nearly free; (ii) a `distill()`
trainer (optax) fitting a student policy to the teacher's Boltzmann policy
posterior, optionally supervised on the *decomposed* terms so the student
stays interpretable. What remains genuinely unspecified — whether the
student is a policy only or also a compressed world model — belongs to the
research tier, and the export contract serves both answers.

Scope clarification this forces: "no arbitrary pretrained models" means
**no inference semantics for foreign models** — the engine never inverts,
trains, or message-passes through them. Wrapping them as declared
transducers with library-owned execution (and, for predictors, as custom
maps by registry reference) is in scope.

### Three-tier compute (edge / near-edge / cloud)

A deployment lens the architecture already satisfies; the requirement is to
keep it true. AIF natively factors computation by timescale, and the
timescales map onto the tiers:

| Tier | Loop | Why it fits |
|------|------|-------------|
| Edge | State inference/filtering; distilled student policies; precision-based fault handling | Conjugate updates and fixed-K mixtures are deterministic, static-shape, no runtime jit — predictable latency |
| Near-edge | Explicit policy search (vmapped mixture rollouts) at replanning frequency | Needs a GPU; tolerates tens-of-ms latency |
| Cloud | Learning: parameter updates, BMR/structure learning, behavioral fitting, distillation training, benchmark sweeps | Slow-timescale, batch-shaped |

The tier mapping is a **capability lattice, not a partition**: a more
capable tier can absorb a lower tier's loops (a near-edge GPU can also run
filtering; the cloud can run all three, as it does in simulation), and the
bound on upward absorption is **deadlines, not FLOPs** — state inference
and action emission must meet the actuation deadline wherever they run, so
moving them off-device is gated by round-trip latency.

The binding design rule is correspondingly about interfaces, not placement:
**loops exchange only value-semantic, serializable pytrees — sufficient
statistics of beliefs, parameters, and policy posteriors — never callables
or shared mutable state.** Colocated deployments pass the same pytrees in
memory and may fuse all three loops into one jit'd program (the
pymdp-style fused rollout remains the default for simulation); a deployment
boundary serializes the identical objects (beliefs are small pytrees;
parameters use the sidecar format; structure uses the versioned spec). This
contract is AIF-native rather than bolted on: timescale-separated levels
already communicate by passing distribution parameters as messages —
pseudo-observations upward, empirical priors downward (parametric depth) —
so the deployment interface and the hierarchical-model interface are the
same kind of object. One honest asymmetry: **end-to-end gradient flows
cannot cross a serialization boundary** — training that differentiates
through planning (differentiable EFE, distillation losses through rollouts)
requires colocation; cloud learning otherwise consumes logged traces.

Distillation is the mechanism that moves capability down-tier. Explicitly
out of scope: fleet orchestration and serving infrastructure — hybrid-aif
provides the separable pieces and the contracts, not the wiring between
tiers. The static-shape discipline chosen for jit correctness doubles as
the edge story.

### Thermodynamic samplers (p-bits)

One interface, no hardware code. The rule: every expectation in the
objective and inference layers routes through the estimator registry, and
estimators that sample route through a `SamplerBackend` protocol — default
PRNG-MC with static sample counts; an external/hardware backend implements
the same protocol via host callback (the identical pattern already accepted
for the optional Pólya-Gamma Gibbs backend). Concretely: no `jax.random`
hardwired inside estimator internals. First natural hardware targets:
discrete-block Gibbs sweeps (categorical conditionals are Ising/Potts-shaped
— the native p-bit workload) and mixture-indicator sampling; more generally
any expectation-shaped term (risk cross-entropies, ambiguity). The known gap
— info-gain terms are entropy *functionals*, not plain expectations — has
two routes the plan already carries: closed-form Kolchinsky–Tracey bounds
(no sampling needed), or the epistemic-priors formulation, which converts
the epistemic term into extra factors so planning becomes pure inference and
hence samplable end-to-end. This is why epistemic priors earned their place
beyond cross-check value. Out of scope: drivers, netlist compilation,
hardware-specific codegen — the deliverable is the seam.

## v0.1

Categorical node at parity with pymdp-jax on T-maze (their tests as oracle) +
Linear-Gaussian node on a mountain-car-class task + one GMM demo; single
explicit planner; metrics tree exposed. **Unchanged by the integration
requirements above** — they bind interfaces v0.1 builds anyway (boundary
observations, estimator registry, serialization, dataset columns).

## Build status (as of 2026-07-21, commit ef38398; audited — STALE below this line: for current shipped capabilities see docs/capabilities.md, maintained per release)

Adopted designs are in `design/`.
Statuses below come from a design-conformance audit, not self-report.
284+ tests, all green; pymdp parity holds at function, term, and policy
level; the closed-loop T-maze agent shows epistemic cue-seeking that
ablates correctly (globally and per modality block).

- **Done**: A (distribution contract + families), B (discrete engine),
  v0.1's discrete half (T-maze parity + demo, enumeration planner, metrics
  trees).
- **Partial**: C (LG + Runnalls/GPB1 built; IMM, GMM-CAVI, MNLR, smoother
  pending), D (term trees done; estimator registry + SamplerBackend
  pending), E (enumeration only), F (System + rollout done; act↔rollout
  adapter and nesting pending), G (registry/HDF5/folder/ReplayBuffer done;
  Lance, goal relabeling, teacher-trace columns pending), J (T-maze only).
- **Not started**: H, I, K.
- **Workstream L (performance) — phase 1 landed**: the diagnosis (NOT the
  metrics trees, ~15% and skippable via `metrics=False`; the structural
  cost was log-canonical tables paying exp/log_softmax round-trips per
  rollout step plus duplicated contractions) was confirmed by the fix. A
  compiled prob-space view (`objectives/compiled.py`) now hoists every
  loop-invariant softmax/entropy/novelty table out of the planning scan
  and carries raw probability vectors through it; `objectives/discrete.py`
  remains the reference implementation, with leaf-level parity tests at
  1e-5. Policy inference went from ~115µs to ~70-80µs on the T-maze
  benchmark (~1.5-2x vs pymdp, from 3.3-4.9x; run-to-run noise is large
  on µs workloads). Side effect: the planner seam's `Beliefs` type is now
  an opaque pytree — the hybrid-typed-seam v0.2 item at the type level.
  Phase 1 does NOT exhaust L. The remaining gap is decomposed (measured,
  same machine, back-to-back): stacking the per-step TermTree through the
  scan is ~half of what's left — 103µs default vs 50µs with `total()`
  folded into the scan carry vs 37µs pymdp (pymdp accumulates a scalar G
  in its carry and materializes nothing else; it computes q(o) per
  step, as its design intends).
  Phase 2 item 1 landed (2026-07-21): the planner seam learned `metrics`
  — under `metrics=False` the enumeration planner folds G into its scan
  carry and stacks no leaves. Measured: ~34-56µs vs pymdp's ~34-101µs
  run-to-run — **pymdp parity at the floor** (34 vs 34); the residual on
  typical runs is the Interval bound-pair, G's lower and upper carried
  end-to-end — a quantity pymdp does not carry, with its price now printed as the
  benchmark's `metrics=False` row (the toggle-costs-known principle).
  Remaining phase-2 candidates, only if ever needed: bound-pair collapse
  to one scalar for all-exact models; the FPI/state-inference path
  (dominates large-batch ticks). Absolute scale: >12k full planning
  ticks/sec on laptop CPU — not a control-rate concern.
- **v0.1: COMPLETE** (all items closed; see examples/ for the two demos
  and the ranking-fidelity test; planner injection restored).
- **v0.2 hybrid System: LANDED** (2026-07-23; design in
  `design/hybrid-system.md`, confirmed): typed factor families dispatched
  by declaration type behind the unchanged faces; mode-coupled mixture
  transitions at a static Runnalls budget (per-mode F for SLDS);
  state_entropy as the hybrid epistemic term (block-grain toggle); scope
  guards as loud errors (mixture filtering → C; hierarchical conditional
  D → F); planner-free `score_policies` face; env `reset(state=...)`
  point-start override (agent prior and process start are separate
  claims). Both v0.1 demos now *are* Systems, at bitwise/1e-5 parity with
  their former hand-rolled enclosures (two acceptance rounds). 422 tests; discrete path
  untouched by identity. **Finding** (the acceptance test's
  lasting lesson): a System declaration costs ~25-30 lines over a minimal
  hand-rolled enclosure and pays for itself when the model is used
  through more than one face — the closed-loop demo got 14% shorter; the
  open-loop scoring-only demo stayed ~35 lines longer, of which 9 are its
  deliberately non-System exact-validation reference. The layers remain
  fully usable à la carte. The two follow-ups queued by the audit landed
  next day: `predict_beliefs` (belief trajectories as a face — the last
  seam-dip) and `D_deps` (hierarchical conditional priors, discrete-path
  native); the notebook gained the spelled-out EFE math the System
  rewrite had moved off the demo page.
- **Open questions resolved since circulation**: 1 (belief contract — see
  `design/conditional-families.md`), 10 (numpy iterators de facto), and
  from the scoping review: System semantics, toggle granularity,
  cross-factor deps, equinox. Newly bindable: 6 partially (grad-away-from-
  boundary verified; IMM rule unconfirmed). Still open: 2, 3, 5, 7, 8, 9,
  11 (partially), 12.

## Workstreams

Streams marked **[E]** engineering-tier (references exist) / **[R]**
research-tier (novel, needs experiments).

| # | Stream | Contents | Depends on |
|---|--------|----------|------------|
| A | **Node/distribution core** [E] | pytree distribution contract; Categorical + Gaussian; block ids; message/update interface | — |
| B | **Discrete engine parity** [E] | pymdp-jax semantics behind the node abstraction; parity tests | A |
| C | **Continuous/mixture kernels** [E→R] | LGSSM filter/smoother; GMM indicator-CAVI; PG-CAVI MNLR; Runnalls/IMM reduction; NumPyro fallback | A |
| D | **Objective/metrics layer** [E] | EFE term registry; per-node estimators (incl. entropy bound pairs); ablation flags; metrics trees; expectations via pluggable `SamplerBackend` | A |
| E | **Policy search** [E] | enumeration; CEM/MPPI/gradient; SI tree (landed 2026-09-29 as the exact, unpruned sophisticated recursion over declared `branch_on` modalities — PR #6, external contribution) + mctx | A, D |
| F | **System/env + rollout** [E] | functional Env; scan rollouts; `System` composition; tick runtime with transducer slots (host-side encoders sequenced with the jitted step); hierarchical conditional priors (`D_deps`, landed: ancestral joint sampling in `as_env`, mean-field marginal resolution via `initial_priors`; fully tabular hierarchical models classify discrete) | — |
| G | **Data pipeline** [E] | format registry; HDF5 + Lance; ReplayBuffer; goal relabeling; teacher-trace columns (beliefs + per-policy term trees) | — |
| H | **Learned-representation nodes** [R] | JEPA pseudo-observation node + SIGReg (exact-VFE per arXiv:2607.13612); state-epistemic coverage term; ensemble info-gain; frozen-encoder adapter (V-JEPA-class latents as pseudo-observations with injected precision) | A, C, D |
| I | **Preference learning** [R] | phenotypic-prior node family; learned/online preference priors | A, D |
| J | **Benchmarks & eval** [E] | T-maze suite; continuous envs; diagnostics; robustness protocol; epistemic-behavior envs (gameworld, ambigym — both permissive) | F, G |
| K | **Behavioral model fitting** [E] | `fit(spec, behavior, priors)` via NumPyro; parameter recovery; model comparison; optional extra | A, B |
| L | **Performance** [E] | compiled prob-space params view; fused rollout/term kernels; benchmark regression harness vs pymdp (see Build status for the diagnosis) | B, E, F |

Critical path A → (B ∥ C) → D → E; F, G, J, K parallel once A/B interfaces
exist; H and I are the research tier.

## Post-v0.1 roadmap candidates (from the landscape survey)

Harvested from the packages surveyed; none block v0.1, all are cheap-to-
moderate extensions of machinery we're already building. Items marked
**[deferred]** are different in kind: decided deferrals of work we have
already designed, carrying a stated gate on when they must be paid.

- **[ACTIVE 2026-08 → LANDED 2026-10-10: the R-AIF acceptance met, 1ef76b8, `experiments/raif-mountain-car/acceptance-report.md`] Amortization program** (`design/amortization.md`,
  A1–A9 adjudicated): the search↔habit slider — neural conditional
  families, gradient learning, habits/E in every planner, learned
  G-tail (`derived` flag on Interval), learned preferences, trainers.
  **R-AIF replication is the binding acceptance** (arXiv:2409.14216).
  Bumps workstreams E (CEM), G (teacher traces), H (encoders, phase
  3), I (preferences); the two [deferred] items below slid behind it and
  are unblocked since, their gates restated in place.
  **Debt introduced by the arc is tracked in
  `design/amortization-debt-ledger.md`** (live; opened 2026-08-27).

- **[framing, 2026-08-20] The objective-term lattice**
  (`design/objective-terms.md`): every term-tree leaf is a point in
  *(declared conditional) × (functional) × (sign)*, over a closed
  four-functional vocabulary — which places `ambiguity` and the new
  `occupancy` row (maximum occupancy principle,
  doi:10.1038/s41467-024-49711-1) as the same functional on different
  conditionals with opposite signs, and caught a double-counting error
  in the chunk-2b plan before it was built. Carries the staged decision
  on a public `register_estimator` for user-defined objectives (open it
  on the second customer, not the first) — the surface for "Bayesian ML
  generally, AIF as the default framing."

- **[deferred 2026-08-27] `epistemic=` as literal sugar over `terms=`**
  (`design/amortization.md` A12 item 1, amended). Chunk 2b shipped two
  sibling selectors over one suffix grammar — the four `infogain`
  estimators are reachable only through `epistemic=`'s family dispatch,
  never by a `terms=` row key. The intended architecture is one
  mechanism; the merge is a refactor of the v0.4 resolver (eligibility,
  channel fan-out, one-estimator-per-variable, the compiled path) under
  its parity suite. **Gate: `register_estimator`'s first customer** — the
  moment a user-defined row needs the infogain machinery. Known debt,
  deliberately not paid mid-arc.

- **[deferred → PAID 2026-08-03, v0.4.0] Per-variable info-gain terms
  in the objectives layer** (`design/per-variable-infogain.md`, V1–V8;
  owed 2026-07-25, from `design/parameter-learning.md` D6). The term tree
  is organized by EFE decomposition with per-term static toggles, so
  epistemic value about *states* and about *parameters* are separate code
  paths behind separate switches. Treating parameters as variables gives
  one `info_gain` per variable — the correct factoring, and the one a
  public API should present. Deferred because it touches the TermTree and
  carries its own parity surface; the parameter-learning pass is
  deliberately built one level short of it (per-declaration selectors over
  one namespace, one walk over capable declarations) so the refactor
  rewrites a single walk, not several parallel paths. **Gate: land before
  any public release** — an API-shape debt gets harder to pay once the
  current shape has users. The landing surface already exists (2026-07-27):
  `System._term_targets` resolves every term-participation selector to
  index tuples in one place, and both planning paths consume them — the
  refactor re-bases the TermTree on top of that single resolver.
- **[deferred] Generic message-passing layer: `infer(query, clamped)`**
  (owed 2026-07-26; the fork is open question 13). A lower API layer
  speaking variables and factors, with parameters as variables one chooses
  to query, and the AIF vocabulary layered above it. **Motivation:** It makes
  the code/math structure cleaner, enables finer-grained querying (and 
  updating), and also makes visible how the active-inference framework 
  connects to the more fundamental message-passing machinery it rests on, 
  at no performance cost (if done properly).

  Cheap to reach, because `distributions/` and `inference/` are already
  AIF-free (see question 13), and the parameter-learning pass moves toward
  it: polymorphic declaration slots are the generic representation, and its
  per-slot learning rates are already a soft query set, with `0.0` meaning
  "not in the query" and intermediate values meaning damping. Two rules
  preserve performance:
  node-type dispatch must resolve at trace time, and schedules must stay
  static. Design the node-update registry to return a **(prepare, step)
  pair** so the compiled path keeps hoisting loop invariants out of the
  planning scan; cheap now, expensive to retrofit.

  **Gate: blocked on the per-variable info-gain refactor above** — PAID
  2026-08-03 (v0.4.0), so this item is unblocked; the amortization program
  it queued behind closed 2026-10-10 (entry 113), and its gate is now open
  question 13 below (Alex's word). *Note (2026-10-17):* the forward rule for a
  deterministic nonlinear node now exists for transitions
  (`NonlinearGaussianTransition`, f69bab2, after Petersen et al. 2018,
  [arXiv:1903.09136](https://arxiv.org/abs/1903.09136)); the same paper's backward (smoothing) rule is the
  reference for the continuous half of this layer.
  Third independent arrival (2026-08-18, nesting partner-modeling
  adjudication): the observation/latent distinction itself may not be
  context-free — an observed variable is a clamped variable, and
  clamping generalizes to sampling from externally-driven dynamics.
  Three arrivals at this layer from three programs is evidence the
  layer is real.
  Carries a recorded naming debt to pay on arrival:
  `LinearGaussianObservation` bakes a role into a mathematically
  role-free linear-Gaussian conditional (the v0.2 restriction's
  fossil); roles are slot vocabulary, not class ontology.
- **[deferred] Batched updates across same-family factors** (owed
  2026-07-28). We batch across agents, modes, and policies — the idiom is
  "hoist the batch into the model" — but a Python loop over modalities
  still unrolls at trace time (`inference/discrete.py`). Collapsing
  shape-identical factors into one stacked update is semantics-preserving
  *because the FPI sweep is synchronous (Jacobi)* — treat that as a design
  invariant; a sequential sweep would turn batching into a scheduling
  problem. The right strategy is **lower, then batch**: after the Learnable
  lowering (`expected_log_params()`, expected-precision), point and
  posterior slots of one shape are indistinguishable arrays, so related
  families batch together for free. Exact-shape grouping is day-scale;
  padded shape-buckets (32/64/128) are the expensive half — learning
  self-masks (a padded factor sends the zero message; absorbing zero is a
  no-op) but `decay` must be forced to identity on dead slots and every
  objective-side reduction over the factor axis (novelty, entropy) needs
  explicit masks. **Gate: a measured inefficiency** — compile-time creep
  from trace unrolling or profiled runtime cost on a real model with many
  same-shape factors. Deferred indefinitely if that never materializes;
  if paid, sequence after the two refactors above, which reshape the same
  internals.
- **[flagged 2026-08-19 → LANDED 2026-09-18 as `legal=`, `design/action-legality.md`]
  Explicit `action_mask` input, separate from `policy_prior`.** Before
  `legal=`, masking rode the habit prior (−inf leaves, the N2 pattern), a
  collision once E is *learned*; it now composes additively at the policy
  softmax (8c004c8, entry 073). Origin: the mission build's peer review — E's legitimate
  tenants are habits and declared action priors; the mask is the
  squatter.
- **[flagged 2026-08-19; obs_targets half LANDED 2026-08-19; per-step targets conditional on
  a discrete control node LANDED 2026-10-20, dba02aa, `C_deps`, ships in 0.15.0,
  `design/step-durations.md` T5] Conditional continuous targets.** `obs_targets` on LG modalities is built —
  Gaussian/mixture C designations over a (possibly switching) LG
  modality's observation space, utility and risk/ambiguity leaves
  keyed by modality, decompositions coherent at both interval
  endpoints (`hybrid-system.md` "Amendment: LG modality C
  designations"; `tests/test_lg_obs_targets.py` carries the friction-6
  behavioral case, both workaround failure modes included). What
  remains flagged: **H_deps with a continuous child** (a terminal
  target conditioned on a discrete parent, resolved into a continuous
  density), and `obs_targets` on **neural** modalities, which needs
  A1's amortized predictive. Origin: friction log finding 6 — the
  first external consumer paid a real workaround tax (a standing
  state-target mixture needed a calibration dance with two measured
  failure modes).
- **[flagged 2026-08-19] Explicit entropy handle for channel-observed
  continuous variables** (friction log 12): the one-estimator-per-
  variable rule is right as the default, but a deliberate opt-in
  (e.g. an estimator-qualified selector key) for the pure
  uncertainty-aversion leaf has a customer sketch on file; off the
  critical path by the customer's own correction (open-loop predicted
  covariance is dynamics-driven, so the leaf doesn't value detours —
  info gain does).
- **[flagged 2026-08-19] merge() validation errors: aggregate
  mismatches** (`merge()` since b428329, `couple()` being the wiring API;
  one field per round-trip today, `composition/merge.py` `_agree` — policy_len, then
  gamma, then mixture_schedule; friction log positive-finding nit).
- **LANDED (b2c58ae, ships in 0.14.0): Bayesian model reduction & online structure learning** (SPM
  `spm_MDP_VB_sleep`/`spm_MDP_structure_learning`; AXIOM's component
  birth/merge rules): closed-form reduced-Dirichlet log evidence is pure
  tensor math, pairs naturally with mixture-node component birth/death, and
  is philosophically continuous with our Runnalls reduction. Strongest
  single candidate. (Permissive JAX precedent: dimarkov/bmr4pml.)
  *Design 2026-10-14: `design/structure-learning.md`, S1–S8 settled.* A
  learned mixture is a categorical factor with a conditional emission; a
  growable factor declares its cap as its size, a level is live while its
  concentration is positive, growth is AXIOM's Algorithm 1 inside the learn
  face (per-datum predictive score against the empty level plus `log α`),
  reduction merges two levels when the closed-form evidence of the merged
  statistics is no lower, on a schedule (every `n`-th learn call; the same
  primitive a hierarchical model's slow tier steps on). Needs on the way:
  the Gaussian emission over categorical parents only (the 2026-09-18 row
  below), `log_evidence` per conjugate family (shared with the description
  length item and the zeroth-harness brief), a modality whose datum is a
  previous-tick belief of a named factor (AXIOM's recurrent mixture, S8).
  Source: AXIOM (Heins et al. 2025, arXiv:2505.24784).
- **AXIOM's pixel layer** (no priority set, 2026-10-13): the slot mixture
  over pixel tokens whose component parameters are latent states, the
  presence and moving gates, the unused counter, the nearest-object
  relation. Engine requirement if taken up: a likelihood whose per-component
  parameters are variables rather than clamps. Not needed by the structure
  learning above, which works on any categorical factor with conjugate
  emissions.
- **Non-conjugate parameters in the learn face** (2026-10-14): a gradient
  step on the closed-form evidence for a hyperparameter with no conjugate
  update (the growth propensity `α` first; the Dirichlet anchors and the
  MNIW priors are the same case), then the Laplace posterior around that
  maximum as the general facility; `design/model-description-0.3.md` 18.5e
  names the gap (a conjugate hyperparameter today has only a call-time rate
  of zero).
- **Clone-structured cognitive graphs** (Raju, Guntupalli, Zhou,
  Lázaro-Gredilla & George, [arXiv:2212.01508](https://arxiv.org/abs/2212.01508)): sequence learning
  over cloned discrete states, with Viterbi and Baum–Welch over the clone
  graph as the tools; later than the structure learning above, which it
  would sit on (a clone is a level).
- **Controllability beliefs** (SPM `spm_MDP_VB_XXX`'s `k`): beliefs about
  which factors are controllable, alongside the terminal-state priors (`H`
  — probabilistically grounded in arXiv:2312.07547: `h` as the
  final-state prior reciprocal to `d`, with the derived inductive policy
  prior as a second scoring backend; conditional/learnable `h` chartered
  there and taken up by the nesting build)
  already promoted into the preference-node family above — neither has been
  ported by any library.
- **Threshold-free fault tolerance via online precision learning**
  (Baioumy/Pezzato line): failing sensors auto-down-weighted through the
  per-block precision machinery we already plan — a production-robotics
  differentiator no library exposes.
- **Generalized-coordinates variant of the Linear-Gaussian node** (SPM
  DEM/LAP — "generalised filtering"): a *constructor, not an engine* —
  `generalized(jacobian, embedding_order, smoothness) → (F, Q_chol)`
  embedding the derivative/shift operator and the smoothness-determined
  correlated-noise Q, run on the existing predict/condition machinery;
  plus a sensor-stream → generalized-coordinates embedding utility.
  Colored-noise robustness and delay compensation (Meera & Wisse 2021);
  no maintained Python equivalent exists. Nonlinear/hierarchical cases via
  the per-order deterministic-map rules rather than porting DEM's
  scaffolding.
- **Surprise-weighted (prioritized) replay** (an internal poster;
  reference implementation in an internal repository, assessed):
  proportional PER with a surprise priority *re-scored under the current
  model* on a schedule — the re-scoring is the non-PER idea worth keeping.
  Reimplement from spec, do not port (the reference's EFE path is dead
  code and its IS correction is computed but never applied). Decomposition:
  `prioritized_sampler` plugin + stable-episode-id/eviction hook in
  `ReplayBuffer` + a trainer-owned re-scoring step; use per-step VFE (which
  we compute natively) as the score, per-clip granularity, applied IS
  weights. ~2–4 days once a training loop exists to own the refresh.
- **Planner-layer extras**: free-energy Occam-window pruning (SPM `zeta`/
  `chi`) and EFE-guided incremental tree expansion (Champion's BTAI) as
  alternatives to visit-count expansion; habit-network distillation from
  MCTS visit counts (Fountas et al.) as the blueprint for the amortized
  backend; contrastive decoder-free EFE (Mazzaglia, MIT) on the estimator
  menu for JEPA latent nodes.
- **Production interface patterns**: behavior-trees-as-priors (Pezzato's
  T-RO 2023 design) for gating continuous controllers from discrete
  decisions; an edge latency/memory benchmark answering the
  hardware-oriented-AIF line (arXiv:2508.13177); a MuJoCo/Isaac bridge
  (greenfield — none exists).

- **[flagged 2026-09-18] A discrete latent observed through continuous
  features, declared directly.** The first model a composing agent writes
  for perceptual grounding is a categorical "which object / which mode"
  factor read off continuous features, and `validate_structure` refuses a
  `LinearGaussianObservation` over a categorical factor (family-purity, D3
  of `design/hybrid-system.md`). Both workable routes are documented in
  `composing-systems.md` — bin the features into a tabular modality, or
  carry a `MixtureGaussian` factor switched by the discrete one — and the
  second shows the machinery already exists: a per-level Gaussian emission
  over a categorical parent *is* a mixture likelihood, and the switching
  observation already turns continuous evidence into mode responsibilities.
  What is missing is the declaration: a per-level Gaussian emission family
  (a `ConditionalGaussian` in the likelihood slot over categorical parents)
  that compiles to that path without the user declaring a carrier. Closed
  form, no new inference; the estimator work is the per-channel infogain
  leaf for such a modality. Surfaced by the composition probe
  (`agent-composition-probe.md`, item c2) and by zeroth-control's model-patch
  loop, whose proposer will try this first. **Gate: the first composer that
  needs it** — zeroth-control's structural patches are the likely customer.
  *Amendment 2026-10-14 (Alex): settled as S2 of `design/structure-learning.md`.*
  No new family: the per-level Gaussian with a Normal-Inverse-Wishart prior
  is `LinearGaussianObservationMNIW` with an empty regressor block, so the
  build is the validation rule admitting zero continuous parents, the
  responsibility message for "`y` observed, `z` unknown, no continuous
  state", the learn dispatch at `dx = 0`, and a constructor alias so the
  declaration reads as a Gaussian emission over a categorical parent. The
  per-level score is the posterior predictive (a Student-t), which is also
  what the growth rule thresholds. The customer arrived from the other
  side: structure learning needs it before any composer does.

- **[flagged 2026-09-18] One scan over a record, or two.** `replay`
  (`design/replay.md`) scans the canonical loop over a recorded trace; the
  world-model trainer (`train/world.py`, `RSSMCore.observe`) scans its own
  loop over the training side's `Episode`, with its own reset and window
  conventions. Whether the trainer should consume traces through `replay` —
  one driver, one row convention — or the two scans stay separate because
  they serve different tiers, is undecided. **Gate: the first time the two
  conventions disagree on a real trace**, or the amortized chunk of `replay`
  (R5), whichever comes first.
  *Amendment 2026-09-20 (Alex): the gate fired with replay's chunk two
  (cd69715).* What it found: the **row** conventions agree (`Episode` row `t`
  pairs `obs[t]` with the action drawn at it; `replay` reads `actions[t]` as
  taken after `observations[t]`; the reset row is an adapter matter). The
  **belief** conventions do not: the trainer's scan (`RSSMCore.observe`)
  samples the straight-through latent under a key, and that sampling is the
  loss's own forward pass; `replay` uses the keyless per-tick faces, which is
  how the reference agent acts. Same trace, two belief sequences — the
  mean-field/sampled split of divergence-register **6.5**, now at the level of
  the whole filter rather than one scoring leaf. Either "one scan" puts a key
  into the belief seam, which is the engine-key decision reserved for the 6.5
  round. **The round ran**: the probe over the same recorded episodes came
  first (entry 078, `pref-rollout-probe-500k.md`), then both shapes landed
  (`key=` on every face, c68c6d3, entry 079; `infer_states(key=)`, 57235fb,
  entry 087). The one-scan question itself is still open: `train/world.py`
  does not consume traces through `replay`. Open beside it (Alex): whether
  sampled and mean-field filtering are two *general* shapes the library should
  offer, which turns on whether `RSSMCore` is a generic model type or R-AIF's
  — see the tier-split review below. *Ruled 2026-09-20 (Alex): both shapes,
  as general library features (`design/amortization.md`, FLAG (1)'s
  amendment); the trainer-through-`replay` question stays open on its own.*

- **[flagged 2026-09-21] Monte Carlo expected free energy on the tabular
  planners.** The faces now carry a sampled rollout on every path
  (`update_empirical_prior(key=)`, one particle), and the amortized scoring
  seam uses it because the Dreamer-style trainer averages over many imagined
  trajectories. The enumerating and CEM planners on the discrete and hybrid
  paths still score the expected rollout only: scoring one sampled trajectory
  per policy would make G a noisy estimate with nothing to average it. A
  sampled planner needs several particles per policy and a rule for folding
  them (the mean of G, or G of the mean belief, are different estimators),
  which is Monte Carlo AIF / sophisticated-inference territory with its own
  choices. **Gate: the first model whose expected rollout is a poor summary of
  its predictive distribution** (multimodal transitions where the mean belief
  lands between modes), or a consumer asking for it.

- **[flagged 2026-09-20 → LANDED 2026-09-21 in three phases: 45d198e,
  b165a0d, 540026e; entries 080–082] Tier-split review: what in the engine is still
  R-AIF's.** The 2026-08-28 addendum (`design/system.md`) placed
  `inference/rssm.py` in the engine as generic machinery and moved only its
  *placement* (`worlds.tiny_rssm`) to `reference/`; the module still opens
  "R-AIF's RSSM world model" and carries `RSSMCore.tiny`, their config row,
  as a constructor. A survey on this date finds no engine module importing
  reference (the layering test holds) but fourteen engine modules whose
  docstrings narrate the replication (`distributions/neural.py`,
  `train/imagine.py`, `objectives/amortized.py`, `system/core.py` …), and a
  reference tier whose R-AIF content (`crspp`, `eval`, `revise`, `seed`,
  `run_mc`, `raif_agent`, `fixtures/`) sits flat beside the shared targets
  (`envs/`, `worlds`) under names that do not say so. Questions (Alex):
  whether `RSSMCore` is a generic model type to be disentangled from the
  replication or belongs in `reference/` wholesale; whether sampled and
  mean-field filtering are two general shapes; a `reference/raif/` subtree
  even before a second occupant. **The review ran the same day:
  `tier-split-audit.md` (a forensic audit, kept beside `de-risk-report.md` and
  `scoping-review.md` rather than under `design/`), decisions T1–T10 ruled
  2026-09-20.** Order: the
  refactor round (one, reviewed, a minor version), then the filtering
  question with 6.5. The move waited for `crspp_init_100k` to
  reach its 500k gate, since a resume imported `reference.run_mc` by path; it did.
  *[LANDED 2026-09-21, phase one of three: `reference/raif/` exists,
  `RSSMCore.tiny` is `reference.raif.world.tiny_core`, `seed_stats` is the
  engine's, and a resume imports nothing by path — only the launch command
  changed. Phase two (T1–T6) landed as b165a0d; phase three (T9, the prose
  pass) with the 0.9.0 bump.]*

- **[flagged 2026-09-21; done 2026-09-30] Distil the docstring salvage.** The prose pass copied
  what it removed, verbatim, into
  `experiments/raif-mountain-car/docstring-salvage-2026-09-21/` by module and
  symbol — the shape of the procedure, not of a history. Owed: for each section,
  confirm the fact is in its home (register, ledgers, plateau analysis, design
  notes) or move it there — the few recorded nowhere else first (the `G_scales`
  adjudication under `imagine.term_value`, the initializer probe's numbers) — then
  delete the folder or reduce it to a one-page index. **Gate: before the next
  replication doc round**, or whenever the engine's `History:` pointers are next
  rewritten.

- **[landed 2026-09-29, external] Sophisticated inference** (Friston et
  al. 2021, arXiv:2006.04120): the `"sophisticated"` planner scores each
  first action by its step terms plus the expected value of the next
  decision over the outcomes of the System's `branch_on` modalities — the
  exact recursion, no pruning, cost `(A·n)^(H-1)·A`. What it opens: the
  paper's Occam pruning and pymdp's budgeted tree (`planning/si.py`:
  node/branch budgets, probability thresholds) when horizons grow; branching
  on continuous or amortized modalities (sampled outcomes — the two-rollout
  rule's keyed form) which the exact form refuses; and a policy prior E over
  the first step, which the planner currently declines. Contributed as the
  repo's first external code PR. Since a0f4916 (entry 092) it folds
  over the shared expansion in `planning/tree.py`.

Licensing note for anything above: AXIOM/CAVI-CMN/VBGS are academic-license
(reimplement ideas only); Fountas and k-cybulski/dem are GPL (ideas only);
the Priorelli and SR-AIF repos have no license (unusable as-is); gameworld,
ambigym, contrastive-aif, bmr4pml, and the Pezzato ROS packages are
permissive.

## Open questions for the team

1. **Belief-object contract.** *[resolved: `design/conditional-families.md`; `Gaussian.from_natural` exists; no `ConstraintsSpec` was built]* Shapes declared upfront; but natural vs moment
   parameterization per node (conjugate updates want natural; collapse/EFE
   arithmetic wants moments — likely both views with one canonical)? What
   exactly is the pytree layout that makes vmap/scan uniform across node
   types? Related design fork surfaced by RxInfer's `@constraints` API:
   per-node `block` ids (current plan) vs a separate composable
   `ConstraintsSpec` declaring factorization/form constraints apart from the
   model — the latter is cleaner if users will want to try several
   variational families over one model. That fork is about where
   factorization constraints live; the adjacent but distinct question of
   whether the *API vocabulary* is variables/factors or
   states/parameters/observations is question 13.
2. **Custom-update escape hatch.** The registry-reference mechanism is
   settled; the *API* for a custom update rule (signature, what messages it
   receives/returns, how it declares its block) is not.
3. **NumPyro integration depth.** NUTS for whole-model/offline fitting is the
   obvious fit; VI paths don't need it at all (PG-CAVI is pure JAX). Where
   exactly do we offload — per-node sampling messages, whole-model, or
   offline only?
4. **EFE estimator registry.** *[partial: a registry keyed by (kind, family) in `objectives/registry.py`; no `SamplerBackend`]* Who provides which estimator per node type
   (exact enumeration / closed-form / quadrature / MC / ensemble
   disagreement), and how are errors surfaced beyond entropy bound pairs
   (e.g. E[softmax] for MNLR rollouts: probit moderation vs quadrature vs
   MC)? Plus the `SamplerBackend` abstraction (see integration
   requirements): what exactly does the protocol promise — batch shapes,
   latency class, bias/variance declaration — so hardware samplers can
   implement it honestly?
5. **MNLR category ordering.** Stick-breaking is order-dependent (documented
   pathology with documented mitigations). Default = order by prior/empirical
   frequency; do we also want the symmetric one-vs-each variant in v1?
6. **Gradient-based planners vs merge discontinuities.** *[partial: the gradient planner landed (b66e09c) and refuses mixtures; the IMM rule unconfirmed]* Runnalls' argmin
   makes EFE discontinuous at merge boundaries (harmless for enumerated
   policies; IMM avoids it structurally). Confirm IMM-when-differentiating is
   an acceptable rule, or invest in soft merges.
7. **Preference supervision contract.** Preferences-as-priors is settled;
   for *learned* preferences, what supervision is realistic on our robots
   (success predicate? demonstrations? both?), and which preference nodes
   must stay unconditional (or conditioned only on agent-external structure)?
8. **Real-time constraints and tier placement.** Planning frequency, latency
   budget, and which loop runs on which tier of the edge / near-edge / cloud
   split (see integration requirements for the default mapping) — these
   decide how much amortization we eventually need, whether jit compile
   times matter operationally, and what the belief-snapshot serialization
   must guarantee (size, versioning, frequency).
9. **Production interface.** *[partial: `couple()` wirings with per-participant records (7a356c2, entry 112); the streaming contract open]* What does the boundary expose — observations/
   actions only, or (some) internal latents too, breaking Markov-blanket
   discipline for diagnostics? Streaming/online and offline/episodic both
   need support; how far does the info-dict-as-namespace pattern (from
   stable-worldmodel) take us? For the streaming/asynchronous contract,
   RxInfer's reactive message passing (infinite-stream inference with
   forgetting factors, RxEnvironments) is the reference for what "done"
   looks like.
10. **Data loading.** *[resolved: plain numpy end to end, `data/`]* grain vs plain numpy iterators feeding
    `jnp.asarray` at the device boundary.
11. **Benchmark suite.** *[partial: T-maze and Mountain Car under `reference/envs/`, `examples/benchmark_pymdp.py`; no ladder]* Proposed start: T-maze family, 1–2 continuous-control
    envs, violation-of-expectation and probing diagnostics, and an
    eval-provenance/robustness protocol (evaluation states from expert vs
    random trajectories can swing reported success 94%→12% — worth
    institutionalizing from day one). Candidate additions from the survey:
    gameworld (VERSES' benchmark envs, MIT) and ambigym (ambiguity/curiosity
    envs purpose-built for validating epistemic terms), plus Champion's
    "contrarian information-gatherer" ablation ladder (arXiv:2303.01618) —
    the failure mode where naive deep-EFE agents maximize info gain against
    model misspecification is exactly what our per-block toggles and metrics
    trees should make visible.
12. **Frozen-model interop.** For using a frozen predictor *inside* jitted
    rollouts: port weights to JAX per supported model, or accept a host
    callback per step? And which V-JEPA-class models do we commit to
    maintaining adapters for (each is a weight-port liability)?
13. **Is the primary API generic message passing, with AIF as a layer above
    it?** Today the public vocabulary is AIF's — `infer_states` / `learn` /
    `act`, with a visible state-vs-parameter split. The alternative is a
    lower layer speaking variables and factors, one `infer(query, clamped)`
    in which parameters are simply variables one chooses to query, with the
    AIF vocabulary layered on top. **The lower layer is already AIF-free:**
    `distributions/` and `inference/` import nothing from `objectives/` or
    `planning/` and contain no AIF concepts in code, so this is a façade
    plus unbundling `System` (which mixes graph structure, backend config,
    and AIF policy in one class) — not an extraction from entangled code.
    Two rules keep it cheap: node-type dispatch must resolve at trace time
    (as `_is_hybrid` and `_as_log_likelihood` already do, so it compiles
    away), and schedules must stay static — generic in node types, never in
    scheduling order, or jit is lost. Sequencing is the **[deferred]**
    roadmap item; what stays open here is whether the AIF layer eventually
    becomes a *thin* façade or remains the primary surface with the generic
    layer as a documented substrate.
    Predictive coding belongs to this layer (2026-10-11): it is the
    Gaussian mean-field or Laplace member of the constrained-BFE family with
    gradient steps in place of closed-form updates, a backend beside the
    epistemic-priors one, with prospective-configuration learning as an
    optional local learner; see `docs/design/predictive-coding-reading.md`.

## Where to read more

`scoping-review.md` for the full analysis of the reference repos and papers
(architecture recommendation §3, risk register §6); `de-risk-report.md` for
the MNLR, mixture-propagation, and model-representation deep-dives, each with
a minimal correctness-test design and full citations.
