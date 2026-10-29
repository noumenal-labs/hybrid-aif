# hybrid-aif: de-risk report

Follow-up to `scoping-review.md`, resolving its three highest-uncertainty items
via targeted literature/ecosystem research (web sources checked July 2026).
One-line verdicts:

1. **MNLR node** — Pólya-Gamma-augmented CAVI on the stick-breaking
   construction is the closed-form fast path; for VI it requires *zero PG
   sampling* (the whole update is a `tanh` expression + Gaussian linear
   algebra). Risk: low–moderate.
2. **Mixture propagation through EFE rollouts** — solved composition of
   standard tracking machinery: exact grow, Runnalls/IMM reduce-to-static-K,
   Kolchinsky–Tracey entropy bounds for info gain. Risk: moderate-low. **No
   one in the AIF literature has done this composition** — it is a genuine
   differentiator.
3. **Model representation** — NumPyro has no serialization (confirmed,
   v0.21.0); no existing format covers our needs; recommended: typed
   pydantic spec + versioned generated JSON schema, custom functions by
   registry reference. ONNX dropped from the representation role.

---

## 1. MNLR node: Pólya-Gamma augmentation

### The math

Polson, Scott & Windle 2013 (JASA 108(504); arXiv:1205.0310): conditioned on a
Pólya-Gamma auxiliary ω, a logistic likelihood in ψ = x'β is exactly Gaussian
(quadratic exponent −ωψ²/2 + κψ), and p(ω|ψ) = PG(b, ψ) stays in-family with
closed-form mean **E[ω] = (b/2c)·tanh(c/2)** (limit b/4 as c→0). The
stick-breaking multinomial extension (Linderman, Johnson & Adams, NeurIPS
2015; arXiv:1506.05843) turns a K-class categorical with linear-Gaussian
parents into K−1 separately-augmentable binomials, so given ω the categorical
child becomes a diagonal-Gaussian pseudo-observation of its parent.

The decisive fact for us (Durante & Rigon, *Statistical Science* 2019;
arXiv:1711.06999): under mean-field CAVI, the optimal q(ω) update needs only
E[ω] — the tanh expression — **never a PG sample**. And PG mean-field VB is
provably *identical* to Jaakkola–Jordan tangent-bound CAVI, inheriting
monotone-ELBO convergence. So the entire MNLR node is `jnp.tanh` + `einsum` +
PSD solves: jit/vmap-trivial, no control flow, no samplers.

Message directions both exist: downward, a categorical message from expected
stick log-probabilities; upward, a Gaussian pseudo-likelihood on the
continuous parent (precision B'·E[Ω]·B) — which is exactly what lets an MNLR
node sit inside a Gaussian message-passing level.

### Precedent and pathologies

- **rSLDS** (Linderman et al., AISTATS 2017; arXiv:1610.08466) used precisely
  this construction for continuous-state-dependent discrete transitions — the
  same coupling our GMM/switching nodes need, so **one PG utility serves both
  the MNLR node and recurrent transitions**.
- **CAVI-CMN** (Heins et al., arXiv:2408.16429 — the VERSES/pymdp-adjacent
  group) is a working **pure-JAX PG-CAVI MNLR** implementation (accuracy on
  par with NUTS/BBVI at a fraction of wall-clock). Existence proof of exactly
  our node. **License caveat: VERSES academic license — design reference only,
  do not vendor.**
- **RxInfer/ReactiveMP** also ships PG-augmented multinomial regression
  (`MultinomialPolya`/`BinomialPolya` nodes: stick-breaking + PG *samplers* +
  cubature integration) — shipped prior art to cite and benchmark against.
  Our Durante–Rigon closed-form CAVI (no sampling, no cubature, monotone
  ELBO) is a legitimate improvement to advertise over their update rule.
- **Known pathology — stick-breaking order dependence**: the construction is
  asymmetric in categories (confirmed in print: rSLDS mitigates via greedy
  decision-list reordering at init; Nassar et al., ICLR 2019 propose
  tree-structured stick-breaking as the fix). Mitigations, in cost order:
  (a) expose category ordering in the node API, default = order by
  empirical/prior frequency; (b) document that permutation changes the model
  family; (c) offer Titsias one-vs-each (NeurIPS 2016) as a symmetric
  PG-augmentable alternative; (d) tree-structured sticks for large K.
- Comparison literature (Depraetere & Vandebroek, *Comp. Stats* 2017): the
  Bouchard/Böhning quadratic softmax bounds are loose — do not build on them.
  Knowles–Minka NCVMP is more accurate but has no ELBO-ascent guarantee —
  awkward for a library that wants hard convergence invariants.
- Mean-field variance underestimation applies (as to all bound-based VB);
  MCMC is the calibration referee in tests.

### JAX ecosystem (checked July 2026)

- **NumPyro** `TruncatedPolyaGamma`: PG(1,0) truncated to [0, 2.5], meant for
  ω-as-latent inside NUTS. *Not* an exact PG(b,c) sampler; no PG Gibbs kernel.
- **`polyagamma` (zoj613, v2.0.2)**: the good exact C sampler; not JAX-aware —
  usable only via `pure_callback` (host round-trip, CPU-only, outside JAX
  PRNG). Fine for an offline Gibbs backend someday; wrong for the hot path.
- **No maintained JAX-native exact PG(b,c) sampler exists.** Doesn't matter:
  for VI you never need one, and for MCMC you don't augment at all (NUTS on
  the exact softmax likelihood). **PG is for the VI fast path, not the
  sampling path** — this inverts a common assumption.
- dynamax: no SLDS/PG in main; `lindermanlab/dynamax-slds` fork is embryonic.

### EFE-layer note

PG gives closed-form *messages/ELBO*, not closed-form predictive class
probabilities E[softmax(ψ)] for rollouts. Use MacKay probit-style moderation
σ(μ/√(1+πs²/8)) per stick, quadrature, or MC there — an estimator-registry
decision (scoping review Q9), not an inference-layer one.

### Correctness tests

1. E[ω] tanh formula vs Monte Carlo from `polyagamma` across a (b,c) grid
   incl. c≈0 (guard the limit with `jnp.where` + Taylor).
2. K=2 reduction to Bayesian logistic regression: exact agreement with a
   Durante–Rigon JJ-CAVI reference (same algorithm), NUTS agreement within MC
   error.
3. K>2 synthetic recovery + permutation-sensitivity quantification.
4. Invariant: ELBO monotone non-decreasing per sweep — violations are bugs.

One subtle derivation step to be careful with: under a soft q(z), the PG shape
parameters b_k = E[𝟙[z ≥ k]] are expectations (residual-count bookkeeping).

---

## 2. Mixture propagation and mixture-entropy EFE terms

### Propagation: grow-then-reduce, all shape-static

Every practical method from 50 years of tracking is "propagate all K·M
children exactly, then reduce to budget":

- **Default: Runnalls 2007** (*IEEE TAES* 43(3)) — greedy pairwise merge
  minimizing a closed-form KL upper bound; merge = exact moment matching.
  Reduction from K·M to K is exactly K·M−K merges → a **fixed-trip-count**
  `lax.fori_loop` over a masked dense cost matrix. jit/vmap-safe; microseconds
  at (K=8, M=4) scales. Crouse et al., FUSION 2011 (survey): Runnalls is the
  best cheap greedy; ISE/COWA buy little at much higher cost.
- **Structural variant: IMM/GPB2** (Blom & Bar-Shalom 1988; Murphy 1998 tech
  report) — when the mixture comes from a *declared discrete mode factor*,
  collapse within mode groups: **no argmin at all**, component count fixed at
  M by model structure. Prefer this when gradients matter (below).
- **Cheap flag: GPB1** — moment-matched collapse to a single Gaussian
  ("GMM node, Gaussian rollout" ablation).
- **Sampling path**: Rao-Blackwellized particle filtering (Chen & Liu 2000;
  Doucet et al., UAI 2000) with static particle count — the fallback for
  strongly non-Gaussian cases, not the planner default (EFE comparisons
  across policies need common random numbers to control variance).
- Avoid in jit: threshold pruning to data-dependent counts, Salmond
  clustering with variable cluster counts, reduce-until-ε schedules.

**Differentiability**: gradients flow through the moment-matched merge
arithmetic and are well-defined a.e., but the argmin pair selection makes EFE
*discontinuous at merge-decision boundaries*. Irrelevant for enumerated-policy
softmax selection; for gradient-based continuous planners, use the IMM
schedule (no argmin) — soft-merge/straight-through documented as future work.

**Belief type implied**: `MixtureGaussian(K)` with **static K** (weights (K,),
means (K,D), chols (K,D,D)); dead components = weight 0 (masking), never shape
change; `Gaussian` is the K=1 degenerate case.

### Entropy / info-gain estimators (no closed form exists for GMM entropy)

- **Default: Kolchinsky & Tracey 2017** (*Entropy* 19(7); arXiv:1706.02419,
  use the corrected arXiv formulas): pairwise-distance estimators giving a
  closed-form differentiable **upper bound** (KL distance) and **lower bound**
  (Chernoff-α/Bhattacharyya), exact in both limits (identical components;
  well-separated), max error log K, O(K²), one vmap.
- **Risk/KL between two GMMs**: Hershey & Olsen, ICASSP 2007 variational
  approximation — closed form, differentiable, same pairwise machinery.
- **Ambiguity term** E_q(s)[H(p(o|s))] is exact per component — no estimator
  needed there.
- **MC entropy** (static N): the validation oracle and opt-in high-accuracy
  path (weight-gradient caveat: discrete component index needs stratified
  sampling or Gumbel-softmax if differentiated).
- API suggestion: **bound-typed returns** — `EntropyBound(lower, upper)` —
  and matched bound directions across EFE terms so biases partially cancel.
  No other AIF library surfaces estimator bias; cheap differentiator.

### Prior art: the gap

- **Koudahl, Kouw & de Vries 2021** (*Entropy* 23(12):1565): the epistemic
  term of EFE is **constant in linear-Gaussian systems** with additive
  control; switching/multiplicative structure restores a genuine epistemic
  drive. This makes mixture/switching nodes the *minimal model class where
  continuous-state epistemic AIF does anything* — the theoretical core of
  hybrid-aif's reason to exist, not a feature checkbox.
- Existing hybrid-AIF work all sidesteps mixture-through-horizon EFE by
  planning only at the discrete level: Collis et al. 2024 (arXiv:2409.01066,
  rSLDS + discrete EFE planner), Parr & Friston 2018 (*Neural Comp.* 30(9),
  BMA collapse each cycle), Priorelli & Stoianov 2025 (arXiv:2402.11658;
  *Entropy* 27(6):570).
- **Nobody propagates fixed-K mixture beliefs through policy rollouts with a
  principled reduction schedule and mixture-entropy info-gain estimators.**
  All ingredients are 30–50-year-old validated machinery; the composition
  under jit/vmap with differentiable EFE is open. Consequence: no reference
  implementation exists to check against — the correctness tests below are
  load-bearing.
- No off-the-shelf JAX SLDS smoother: dynamax main now ships RBPF-SLDS
  *filtering* (added mid-2026) but no variational SLDS learning, smoothing,
  or planning — useful as an MC baseline, not the analytic kernels we need;
  `lindermanlab/ssm` has mature (r)SLDS but is numpy/autograd. We write the
  GSF/IMM kernels ourselves; they are small (per-component Kalman steps
  under two vmaps).
  Pattern references: dynamax's `(params, belief) → scan` kernel style;
  Zhao & Linderman, ICML 2023 (JAX SVAE) for structured messages.

### Correctness tests

Model: 1D state, M=2 modes, linear-Gaussian per mode, known switching matrix,
H=2–3 → exact predictive mixture by enumeration (dozen lines of numpy).
Assert: (i) grow-without-reduce matches enumeration to machine precision;
(ii) Runnalls/IMM collapse preserves mixture mean+covariance exactly at every
step (moment matching is exact by construction — strong invariant);
(iii) KT-lower ≤ MC entropy (N=10⁶, seeded) ≤ KT-upper, error ≤ log K;
(iv) collapsed-pipeline EFE vs exact enumeration within documented tolerance
**and policy rankings agree** (ranking fidelity is what planning needs);
(v) limiting cases (identical / well-separated components); plus a `jax.grad`
finite-difference smoke test away from a merge boundary.

---

## 3. Model representation

### NumPyro: definitive no

No model serialization or interchange format as of v0.21.0 (May 2026). Models
are Python callables under effect handlers. `render_model` /
`get_model_relations` give lossy best-effort introspection and **do not see
inside `scan`** — the construct we care most about. State persistence (pickled
SVI params, ArviZ netCDF for samples) is checkpointing, not model interchange.
A `spec → NumPyro callable` compiler is easy; the reverse is unsupported.

### Survey conclusion

No existing format covers continuous + mixtures + hierarchy + custom CPDs +
time. The space splits cleanly:

- **Discrete-tabular graph formats** (UAI, BIF/XMLBIF, NET, XDSL): right
  *shape* (explicit factors/scopes), wrong *content* — no continuous
  distributions, no parametric CPDs, no time. (pgmpy round-trips them all,
  discrete-only; pyAgrum adds O3PRM's object-oriented template hierarchy and
  does DBNs purely by *naming convention* + `unroll2TBN` — the 2-slice
  precedent.)
- **PPL languages** (BUGS/JAGS, Stan, GraphPPL, NumPyro): right content, but
  they are code with no stable serialized structure (Stan's MIR is explicitly
  non-public; GraphPPL's graph holds Julia functions).
- Two decisive datapoints: **pomegranate 0.x** proved closed-menu JSON models
  work (first-class GMMs!) and then dropped the format in the 1.x torch
  rewrite — the format must be owned by the spec layer, not the runtime.
  And **every team shipping factor-graph/PPL models in production**
  (RxInferServer, PyMC ModelBuilder, Infer.NET) converged on the same
  pattern: *typed config + code by registry reference*, never graph
  serialization.

### Verdict: Python-typed spec + versioned JSON view

- **Layer 1 (source of truth)**: frozen pydantic-v2 `NodeSpec` classes
  matching the node menu exactly — `CategoricalNode`, `LinearGaussianNode`,
  `GMMNode`, `MNLRNode`, `CustomNode` — each carrying distribution type,
  `block` (mean-field block id = toggle/metrics granularity), `inference`
  method, named shape labels (keep pymdp's `elements`), and wiring by parent
  names (`depends_on` / `controlled_by`). Hierarchy = named levels whose nodes
  reference lower levels' outputs as pseudo-observations (O3PRM precedent:
  fragment templates with interfaces, not arbitrary nesting). `compile()`
  turns spec → equinox runtime pytree (dynamax-style params + props).
- **Layer 2 (interchange)**: `to_json()`/`from_json()` with `format_version`
  and a checked-in JSON Schema **generated from the pydantic classes** (never
  hand-maintained). Large tensors by sidecar reference (`.npz`/safetensors).
  **Time**: no temporal syntax — `depends_on` entries carry `lag: 1` tags
  (pyAgrum's 2TBN convention); the compiler owns `lax.scan` unrolling.
- **Authoring symmetry**: because the JSON schema is versioned and validated,
  external tools can *author* models as JSON directly — write a model down,
  treat it as a Markov blanket spec — without touching Python; `from_json`
  is a first-class entry point, not just a re-import path. The pydantic
  classes are where the schema is *defined*, not the only place models are
  *written*.
- **Code-only, deliberately**: custom maps, custom update rules, custom EFE
  estimators. In JSON these are
  `{"type": "custom", "impl": "pkg.mod:Thing", "config": {...}}` — name +
  registry + config, hard load-time error if unregistered. Four precedents
  for this contract: ONNX custom ops, Stan external C++
  (`--allow-undefined` + `USER_HEADER`), JAGS modules, RxInferServer/PyMC
  ModelBuilder registries. The format declares the interface; code supplies
  the semantics; loading without the code fails loudly.
- **Rejected as overkill**: ONNX/tensor IRs (wrong semantics); a general
  factor/edge/message schema (rebuilds the excluded engine in JSON);
  posterior/trace serialization in the model format (that's checkpointing);
  expression ASTs inside JSON (that's rebuilding Stan); format expressiveness
  ahead of compiler support (the schema must never promise what the runtime
  can't compile).
- Estimated size: a 200–400-line pydantic module — smaller than a general
  JSON factor-graph language because the node menu is closed. Bonus interop:
  we could *emit* UAI for all-Categorical subgraphs as a debugging aid.
- **Performance note**: pydantic is confined to the spec/compile layer —
  validation runs once per model structure at load time, `compile()` hands
  off to equinox pytrees, and the runtime hot path never touches a pydantic
  object. Load cost is milliseconds (pydantic-v2 validation is Rust-backed)
  against the seconds of XLA compilation that follow, and parameters bypass
  JSON via sidecars, so parsing scales with node count only. If spec loading
  ever profiles hot (e.g. sweeping thousands of distinct structures), the
  versioned JSON Schema — not the library — is the contract, so `from_json`
  can be re-backed by a leaner validator (e.g. msgspec) without changing the
  format or callers.

---

## 4. Cross-cutting notes

- **One PG utility serves two masters**: the MNLR observation/likelihood node
  and rSLDS-style recurrent transitions (continuous state → discrete mode)
  use the same augmentation and the same tanh/Gaussian updates.
- **Deterministic custom maps** (scoping-review Q8): Petersen, Hoffmann &
  Rostalski 2019 (arXiv:1903.09136, added to `/papers`) tabulates forward
  (quadrature/sigma-point) and backward (RTS-type) approximate Gaussian
  message rules for deterministic nonlinear nodes — the moment-matching
  answer to the delta-factor singularity, complementing the entropy paper's
  Thm 8 free-energy bookkeeping. Together these give the "custom functions
  for any map" feature a principled inference story in the Gaussian regime.
- **Risk summary**: MNLR low–moderate (published updates, JAX existence
  proof, one careful derivation step); mixture propagation moderate-low
  (validated machinery, novel composition, no reference implementation —
  tests are load-bearing); model representation low (convergent
  production-proven pattern).

## Key additional references (beyond `/papers`)

Polson, Scott & Windle 2013 (arXiv:1205.0310) · Linderman, Johnson & Adams
2015 (arXiv:1506.05843) · Durante & Rigon 2019 (arXiv:1711.06999) · Linderman
et al. 2017 rSLDS (arXiv:1610.08466) · Nassar et al. 2019 (arXiv:1811.12386) ·
Titsias 2016 (arXiv:1609.07410) · Heins et al. 2024 CAVI-CMN
(arXiv:2408.16429; github.com/VersesTech/cavi-cmn — restrictive license) ·
Runnalls 2007 (IEEE TAES 43(3)) · Crouse et al. 2011 (FUSION) · Huber et al.
2008 (IEEE MFI) · Kolchinsky & Tracey 2017 (arXiv:1706.02419) · Hershey &
Olsen 2007 (ICASSP) · Koudahl et al. 2021 (Entropy 23(12):1565) · Collis et
al. 2024 (arXiv:2409.01066) · Parr & Friston 2018 (Neural Comp. 30(9)) ·
Priorelli & Stoianov 2025 (arXiv:2402.11658; Entropy 27(6):570) · Ghahramani
& Hinton 2000 (Neural Comp. 12(4)) · Zhao & Linderman 2023 (ICML) · Murphy
1998 switching-KF report · Petersen et al. 2019 (arXiv:1903.09136) · dynamax
(github.com/probml/dynamax) · lindermanlab/ssm · polyagamma
(github.com/zoj613/polyagamma).
