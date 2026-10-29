# hybrid-aif: scoping review

A vetting of the claims and presuppositions in `README.md` against the reference
repos (`/reference`) and papers (`/papers`), plus open questions and a
parallelization map for development. The README itself is preserved as-is; all
commentary lives here.

Method: the three reference codebases and all eight papers were read in full
(repos file-by-file; papers end-to-end including appendices). File paths,
function names, and equation numbers below refer to the reference material.

---

## 1. Claim-by-claim vetting of the README

### 1.1 Verified

**"pymdp-jax: efficient multi-agent policy search in JAX, for discrete models" — TRUE.**
As of v1.0+, the JAX implementation *is* the main `pymdp` package (the earlier NumPy
implementation moved to a `legacy/` subpackage). Two nested vmap levels: across policies
(`control.update_posterior_policies` vmaps `compute_neg_efe_fixed_states` over
the policy matrix) and across agents (`Agent` is an equinox PyTree with a
leading batch dim; `infer_states`/`infer_policies`/`sample_action` all vmap over
it). The whole perception–planning–action–learning loop is `lax.scan`/jit
compatible (`envs/rollout.py`). Discrete-only is also confirmed: pymdp is discrete
by design; soft categorical observations enter through `preprocess_fn`, i.e.
discretize-at-the-boundary.

Caveats: "multi-agent" means *homogeneous-shape batched independent agents* —
no interaction machinery. Heterogeneous shapes exist only via padding/block-diag
kernels, only for state inference, and cost a lot of specialized code.

**"stable-worldmodel as a reference for the data pipeline" — TRUE, with a clear boundary.**
The data layer is the best-engineered, least framework-entangled part of that
repo. Worth borrowing: the ~130-line format registry
(`data/format.py`: `Format.detect/open_reader/open_writer`, streaming `Writer`
protocol), the Lance on-disk schema (episode-contiguous rows,
`episode_idx`/`step_idx`, JPEG-blob image columns; ~3× faster and ~3× smaller
than HDF5 in their benchmarks, S3/HF streaming for free), the clip-index
windowing `Dataset` design, the `ReplayBuffer` = Dataset + Writer duality, the
`GoalDataset` HER-style goal relabeling, and the info-dict-as-schema column
conventions. Not worth borrowing: torch reader internals, the sync-Python-loop
`EnvPool` (wrong reference for vmap-across-agents), the Lightning/Hydra training
stack.

**"Core active inference functions (ablated → model-based RL)" — SUPPORTED.**
Friston et al. 2015 ([doi:10.1080/17588928.2015.1020053](https://doi.org/10.1080/17588928.2015.1020053), Eq. 12) gives the exact nesting:
EFE ⊃ KL control (drop ambiguity) ⊃ expected utility (drop outcome entropy
too), plus fixed-γ ablations. pymdp's boolean flags
(`use_utility/use_states_info_gain/use_param_info_gain`) are working precedent.

**"Metrics on demand (EFE + risk/epistemic breakdown, per-variable/per-policy)" — SUPPORTED, one caveat.**
pymdp's `maths.calc_vfe` already returns a full per-term decomposition dict, and
the EFE terms are small pure functions (`compute_expected_utility`,
`compute_info_gain`, `calc_pA/pB_info_gain`) evaluated per policy — so
per-policy × per-term metrics fall out naturally if the objective layer returns
structured term trees rather than scalars. The caveat is §1.3 below:
*per-variable* breakdowns exist only at mean-field-block granularity. JAX
detail: expose metrics behind static flags (recompile per configuration) —
pymdp's approach — rather than dynamic branching.

### 1.2 Overstated or needs rewording

**"Model-specification language" (re: pymdp) — applies in part.**
pymdp's `distribution.py` and `compile_model` provide a shape allocator with
named indexing and automatic dependency derivation. Parametric CPDs, priors and
hierarchy are outside its scope; our extension targets them.

**"Parametric depth (the γ term in Deeply Felt Affect)" — conflates two things.**
In [doi:10.1162/neco_a_01341](https://doi.org/10.1162/neco_a_01341), γ is expected precision over policies (Gamma prior,
rate β; the affective-charge update is `AC = (π − π̄)·G_π`). "Parametric depth"
is the *architectural pattern*: a higher level's hidden states emit empirical
priors onto a lower level's **parameters** (in their model, β). The paper says
explicitly (p. 20) the pattern applies to *any* slow lower-level parameter
(initial states, policy priors, precisions). Suggested reading for the design:
parametric depth = general capability (hierarchical priors on parameters);
γ/AC = one instance. As written the README under-scopes its own feature.

**"Enable/disable epistemic and prior/KL terms on any random variable" — well-defined only at mean-field-block granularity.**
The per-variable additive split of the EFE exists *because* the posterior
factorizes over those variables (per-modality utility, per-factor info gain,
per-parameter-block novelty — all separate summands under mean field). With a
structured/correlated posterior (SVMP, full-covariance Gaussian), the KL terms
have cross-terms and do not cleanly separate. Design implication: **toggle
granularity = mean-field block**, exposed as per-node/per-block flags; document
that structured posteriors forfeit exact separability. This same constraint
governs the Metrics bullet.

**"Preference learning from R-AIF" — the pattern transplants; the pipeline does not.**
R-AIF ([arXiv:2409.14216](https://arxiv.org/abs/2409.14216)) is a DreamerV3-derived agent. CRSPP is a second,
action-free RSSM contrastively anchored (stop-gradient + cosine similarity,
Eq. 5) to the world model's latent space, and it requires: seed expert
demonstrations (1k–20k steps), a per-step binary success signal, hindsight
episode relabeling (the signed decay schedule of Algorithm 1), and a dual
replay-buffer scheme. There is no preference *distribution* anywhere —
preferences enter behavior as a scalar cosine-similarity reward inside an
actor-critic λ-return (Eq. 8). Also nonstandard: its EFE deliberately
sign-flips the future-state-entropy term (maximized, App. I), so it should be
one selectable EFE variant, not the canonical decomposition.

What transplants cleanly: (a) preference model = action-free prior dynamics
over the agent's own latents, trained with a signed hindsight-decayed rate;
(b) the ensemble-disagreement info-gain estimator (Eq. 13) as a pluggable
epistemic term; (c) KL balancing / free bits / symlog as stability toggles.
Note the README's bibliography line is also imprecise: R-AIF has continuous
*actions* but *discrete (categorical) latent states* — itself a useful data
point that categorical latents work for continuous control.

**"JEPA-style latent surprise" — coherent only after work the papers don't do.**
LeWorldModel ([arXiv:2603.19312](https://arxiv.org/abs/2603.19312)) is not an AIF paper: the objective is
deterministic latent MSE (no likelihood semantics, no calibrated scale) plus
SIGReg (Epps–Pulley test toward N(0, I) marginals — provable anti-collapse via
Cramér–Wold, one effective hyperparameter). The predictor is deterministic (no
epistemic value without adding ensembles/dropout), "surprise" is an
evaluation-time diagnostic (Sec. 5.2), and planning is CEM against an *encoded
goal image*. The coherent AIF framing — which the library must construct — is:
**encoder outputs are pseudo-observations at a higher hierarchical level, with
learned precision**; then latent MSE ∝ Gaussian NLL and SIGReg is (loosely) a
marginal prior. That framing aligns exactly with the README's hierarchical and
per-variable-precision goals. New design constraint to promote: any
learned-latent node needs a first-class **anti-collapse mechanism** (SIGReg is
attractive and trivially portable to JAX).

**R-AIF + JEPA are folded together in the README but conflict at a specific point.**
LeWM's planning *requires a goal observation* — precisely the preference scheme
R-AIF identifies as impractical (Table 1 note d) and replaces with CRSPP.
Combining them (CRSPP-style learned preference priors *inside* a JEPA latent
space, two contrastively-tied learned spaces, no decoder anywhere) is genuinely
novel research the library would be doing itself. Present them as alternative
preference-node types; flag the synthesis as research-tier.

**"Possible reference for compute kernels" (PGMax) — pattern reference only, and situational.**
The reusable part is ~300 lines of discrete-BP idiom (`infer/bp.py`,
`factor/enum.py`, `factor/update_utils.py`): flat arrays + precompiled integer
wirings, segment scatter ops, sum-then-subtract updates, tempered logsumexp
numerics, scan + checkpoint + damping discipline. The other ~60% of the code is
the vgroup/fgroup/fgraph compilation machinery — exactly the general engine we
exclude — and the layers separate cleanly. Important qualifier (agreed in
discussion): the ragged flat-array + scatter layout is motivated by *arbitrary
topology with heterogeneous arities*; for structured templates vmapped across
agents/policies (our case), dense batched einsum — the pymdp-jax approach — is
usually simpler *and* faster on GPU. Also: PGMax is dormant (last commit
2024-10) and crashes on modern JAX (`jax.lib.xla_bridge` removal). Reference,
never a dependency.

**"GMM and MNLR nodes" — no closed-form message coverage anywhere in the reference set.**
Not in pymdp (categorical/Dirichlet only), not in PGMax, not in the worked
recipes of the constrained-BFE paper (all exp-family/conjugate or Gaussian; the
paper explicitly defers sampling-based messages to future work, §8). Nuance:
GMMs are *conditionally* conjugate — mean-field CAVI with explicit component
indicators (Bishop-style variational GMM, NIW priors) is closed-form and is the
natural "hybrid node" design; MNLR is non-conjugate and needs augmentation
(Pólya-Gamma) or bounds, or gradient/sampling updates. The genuinely hard part
is **mixture propagation through EFE rollouts**: components multiply
exponentially over the policy horizon, so a collapse strategy (moment matching
/ pruning / resampling) is a core design decision, not an optimization. Net:
the NumPyro/sampling backend is a primary inference path for these nodes, not
polish.

**"possibly use ONNX" — unsupported by anything in the reference set.**
ONNX captures deterministic dataflow, not probabilistic factor semantics
(distribution families, conjugacy, message types). It could serve as an
*export* format for deterministic maps (learned encoders, custom likelihood
functions) but not as the model representation. Suggest dropping or rescoping.

**`System` class — no precedent; must be designed fresh.**
None of the references has a composable agent/env abstraction with
Markov-blanket boundaries. Closest patterns: stable-worldmodel's uniform
info-dict namespace (one dict of named arrays flowing env → policy → solver →
model → objective → dataset) and its small `typing.Protocol` surfaces
(`Dynamics`, `Objective`, `Costable`, `Solver`) with `isinstance` feature
detection; and pymdp's functional `Env` (pure `reset`/`step`, vmappable). Both
keep agent and env as separate roles.

---

## 2. The factor-graph scope question

Clarified position (from discussion): "out of scope" means *focus*, not dogma —
if the right engine needs to be factor-graph-shaped, so be it, but deriving
closed-form inference for arbitrary factors is the wrong core focus.

The literature sharpens this. The constrained-BFE paper ([doi:10.3390/e23070807](https://doi.org/10.3390/e23070807),
Table 1) shows the entire VI zoo — BP, structured/naive VMP, EP, Laplace,
mean-field variational Laplace, EM — falls out of one recipe: write the Bethe
free energy on a Forney graph, pick *local* constraints per edge/node, take
stationary points. That is an existence proof that the flexibility the README
wants (per-variable inference choices, per-variable term toggling, hybrid
schemes on one graph) is *most naturally expressed* in factor-graph + local
constraint vocabulary. But the same paper is explicit that outside
conjugate-exponential-family pairings there are no closed-form messages
(§4 p. 10) — which is exactly the maintenance trap (ForneyLab/RxInfer carry
that burden permanently).

pymdp is the proof that the trap is avoidable: it delivers hierarchy,
factorized wiring, per-term toggles and multiple inference algorithms on a
fixed POMDP template (A/B/C/D/E roles), a deliberate trade of topological
flexibility for clarity and speed.

**Recommended resolution.** Reframe the scope line from "no factor graph
engine" to:

> A **typed node menu** — Categorical, Linear-Gaussian, GMM, MNLR, custom —
> with hand-derived updates only for the node/edge pairings we actually
> support, composed in **fixed hierarchical templates** (System-of-POMDP-like
> levels), with two escape hatches: custom update functions, and a
> sampling/NumPyro backend. Adopt the constrained-BFE framing (node + local
> constraints → messages; free energy as an evaluable score) as internal design
> vocabulary without exposing a general graph builder.

Each node then carries: (i) distribution type, (ii) mean-field block id
(= toggle and metrics granularity), (iii) inference method
{conjugate closed-form, SVMP, Laplace/EP, MCMC/NumPyro}. This is ~90% of the
flexibility for ~20% of a general engine's complexity. The one README phrase in
genuine jeopardy is "complex wiring if desired" — arbitrary topology *is* the
general engine; suggest qualifying it to "hierarchical/nested templates."

---

## 3. Architecture recommendation (synthesis)

Build on the **skeleton** of pymdp-jax, extending its **math** behind a node
abstraction; borrow the **data layer** from stable-worldmodel; use PGMax and
LeWM as pattern references.

- From pymdp-jax: equinox Agent-as-PyTree with static config; batch-dim-as-agent
  vmap layering; `lax.scan` rollouts with fixed-window masked history buffers
  (a worked solution to JAX static shapes you'd otherwise rediscover painfully);
  the `policy_search(agent, qs, rng_key)` hook; the SI tree
  (preallocated, jit-compatible) and mctx integration; dependency-indexed
  factor wiring (`A_dependencies`/`B_dependencies` + `factor_dot`); the test
  methodology (cross-method parity, SPM validation). Its discrete tests become
  our parity oracle for the Categorical node.
- Everything indexed by discrete state must be rebuilt behind the node
  abstraction: info gain by outcome enumeration, exhaustive policy enumeration,
  FPI/VMP categorical updates, Dirichlet-only learning, one-hot observations.
- Policy search: explicit search as the default (enumeration for discrete;
  CEM/MPPI/gradient for continuous — stable-worldmodel's solver layer is the
  model), amortized actor-critic as an optional backend. This is also what
  keeps per-term toggling and the Metrics bullet trivial: with explicit search,
  toggling a term is arithmetic; with a critic, terms are entangled in the
  value target (retrain per configuration or use decomposed critics).
- Learning API: a general "minimize F w.r.t. Q(params)" hook with conjugate
  closed-form as a fast path and optax gradients otherwise. Do not architect
  around conjugacy.
- Numerics to import on day one: pymdp's stable xlogx/entropy and Dirichlet KL;
  PGMax's tempered logsumexp utilities and MSG clipping constants; R-AIF's KL
  balancing / free bits / symlog as optional node stabilizers; `jax.checkpoint`
  on scan bodies for differentiable planning.

---

## 4. Open questions

Grouped; roughly ordered by how much they block other work.

**Architecture / representation**
1. What is the belief/distribution object contract (pytree layout, natural vs
   moment parameterization, static shape metadata) such that vmap/scan work
   uniformly across node types? This is the load-bearing interface.
2. `System` semantics: is an env just another `System`? How do nested systems
   compose objectives and share the info-dict namespace? What exactly lives on
   the boundary (Markov blanket = typed observation/action edges?)?
3. Do we adopt pymdp's cross-factor transition mechanism
   (`B_dependencies`), or support more general coupling?
4. Equinox (pymdp precedent) vs flax/nnx for module system? mctx for tree
   search? grain vs plain iterators for data loading?

**Inference**

5. Which node/edge pairings get closed-form updates in v1?
   (Categorical–Dirichlet; Linear-Gaussian–NIW; GMM via indicator CAVI; MNLR
   via Pólya-Gamma or gradient?) What is the exact escape-hatch API for custom
   update functions?
6. NumPyro integration depth: per-node sampling messages, whole-model MCMC, or
   both? (pymdp uses NumPyro only for offline behavioral fitting — a soft
   precedent.)
7. Mixture propagation through rollouts: which collapse strategy (moment
   matching, pruning, resampling), and is it per-node configurable?
8. How do deterministic/custom maps (README: "custom functions for any map")
   avoid the delta-factor singularity problem (entropy paper, Thm 8–9)?

**Objectives / metrics**

9. Per-node EFE estimator registry: exact enumeration (discrete), closed-form
   Gaussian, quadrature/sigma-point, sampled, ensemble-disagreement — who
   provides which estimator, and how are estimator errors surfaced?
10. Confirm toggle-and-metrics granularity = mean-field block; what is the
    reporting behavior when a user requests a per-variable breakdown under a
    structured posterior (error, approximation, or block-level rollup)?
11. Precision hyperpriors (γ-style) as first-class nodes — does parametric
    depth fall out of "hierarchical priors on any slow parameter" for free?
12. Preference node types: static C, encoded-goal cost, CRSPP-style learned
    prior. What is the minimal supervision contract for the learned variant
    (success predicate? demos? both?), and is that acceptable for our robots?

**Production / robotics**

13. Real-time constraints: planning frequency, latency budget, deployment
    target (edge GPU? CPU?) — these determine how much amortization we
    ultimately need and whether jit compile times matter operationally.
14. What does the production interface consume/produce (the "Modular" bullet)?
    Streaming observations vs episodic datasets; who owns the encoder for pixel
    inputs; interop with the rest of the Noumenal stack.
15. Multi-agent: is homogeneous-batch (pymdp-style independent agents) enough,
    or do agents share an env / interact? Heterogeneous shapes needed? (Both
    are expensive; pymdp's padding kernels show the engineering cost of heterogeneity.)

**Project**

16. Data pipeline: adopt the Lance schema byte-compatibly (free interop with
    `swm` CLI tooling: inspect/preview/convert) or start with hdf5/folder and
    add Lance later behind the format registry?
17. What is v0.1? Proposal: Categorical node at parity with pymdp-jax on
    T-maze (their tests as oracle) + Linear-Gaussian node on a
    mountain-car-class task + one GMM demo, single explicit planner, metrics
    tree exposed.
18. Which benchmark suite from day one? (T-maze family; 1–2 continuous-control
    envs; LeWM's violation-of-expectation and probing diagnostics;
    stable-worldmodel's provenance/FoV robustness protocol — their 94%→12%
    eval-provenance result is a caution worth institutionalizing.)

---

## 5. Parallel workstreams

Streams marked **[E]** engineering-tier (well-understood, references exist) or
**[R]** research-tier (novel synthesis, needs experiments).

| # | Stream | Contents | Depends on |
|---|--------|----------|------------|
| A | **Node/distribution core** [E] | Distribution-object pytree contract; exp-family base; Categorical + Gaussian; message/update interface; mean-field block ids | — (blocks most; interface stub early) |
| B | **Discrete engine parity** [E] | Port pymdp-jax semantics behind node abstraction; FPI/VMP; Dirichlet learning; pymdp tests as parity oracle | A (interface only) |
| C | **Continuous/mixture kernels** [E→R] | Linear-Gaussian filter/smoother; GMM indicator-CAVI; MNLR augmentation; Laplace; NumPyro fallback; mixture-collapse strategies | A (interface only) |
| D | **Objective/metrics layer** [E] | EFE term registry keyed by block; per-node estimators; ablation flags (AIF→KL-control→EU); structured metrics tree (per-variable × per-policy × per-term) | A; testable against B first |
| E | **Policy search** [E] | Enumeration; CEM/MPPI/gradient solvers; SI tree + mctx port; `policy_search` hook; hybrid action spaces | A, D interfaces |
| F | **System/env + rollout** [E] | Functional Env; scan rollout with masked windows; `System` composition design; info-dict conventions | independent from day one |
| G | **Data pipeline** [E] | Format registry; hdf5 + Lance backends; ReplayBuffer; goal relabeling | independent from day one |
| H | **Learned-representation nodes** [R] | JEPA pseudo-observation node with learned precision; SIGReg in JAX; ensemble info-gain | A, C, D |
| I | **Preference learning** [R] | Preference-node family (static C / encoded goal / CRSPP-style prior); hindsight rate machinery | A, D; G for buffers |
| J | **Benchmarks & eval harness** [E] | T-maze suite; continuous envs; VoE + probing diagnostics; provenance/FoV robustness protocol | F, G; consumes everything else |

Critical path: **A → (B ∥ C) → D → E**. F, G, J are fully parallel from the
start. H and I are the research tier — schedule them as experiments with
success criteria, not as features with deadlines, and note that their
*combination* (learned preferences in a JEPA latent space) is novel work no
paper has demonstrated.

---

## 6. Risk register

1. **JAX jit-correctness at the API surface is hard.** pymdp's 2026 changelog
   documents the work of getting jit-correctness right across its API surface;
   we adopt its conventions (traceable constructors, static config fields, fixed-size
   buffers) from day one; parity tests across inference methods.
2. **Mixture propagation blowup** through policy rollouts (see Q7) — the main
   technical unknown in the "must include" scope.
3. **Representation collapse** in any learned-latent node; anti-collapse must
   be a first-class objective component, not a training trick.
4. **Scope creep toward the general engine.** The node-menu framing (§2) is
   the guardrail; every "just one more factor pairing" request should be priced
   as such.
5. **Critic entanglement vs togglability.** If amortized policies become
   load-bearing for production latency, per-term toggling and on-demand metrics
   degrade (terms fused into value targets). Keep explicit search first-class.
6. **Reference decay.** PGMax already broken on modern JAX; pin known-good
   versions of anything we pattern-match against, vendor nothing.

---

## 7. Suggested README edits (not applied)

- "parametric depth (the γ term …)" → "parametric depth (higher levels priming
  lower-level parameters; e.g. the γ-precision mechanism of Deeply Felt
  Affect)".
- Qualify per-variable term toggling and per-variable metrics with "at
  mean-field-block granularity".
- "complex wiring if desired" → "hierarchical/nested templates over a typed
  node menu" (or keep the phrase and accept §2's consequence).
- Note next to GMM/MNLR that sampling/NumPyro is a primary path, and that
  mixture collapse during rollouts is in scope.
- Note next to R-AIF that its preference learning presupposes demos + a success
  signal, and next to the JEPA bullet that latent-surprise nodes require an
  anti-collapse mechanism; the two preference schemes are alternatives, their
  synthesis is research.
- Rescope ONNX to "possible export format for deterministic maps" or drop.
