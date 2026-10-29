# Conditional families & contract revision 1

*Status: reviewed and adopted (Alex + Claude, 2026-07-20/21); implemented in
contract revision 1. The Appendix "To design" items remain open — they are
the System design session's agenda, not part of this revision. Companion to
the contract in `src/hybrid_aif/distributions/base.py`; addresses the five
friction findings from workstreams B/C plus Alex's prior
categorical/conditional-categorical distinction.*

## The problem

A conditional distribution (CPD / Markov kernel) — `P(child | parents)` as a
table of simplexes, or a parent-indexed family of Gaussians — is not a
batched distribution, even though both are "arrays of parameters":

1. **Normalization semantics**: one simplex *per parent configuration*, over
   the child axis only.
2. **Parent axes are model structure**, wired by `depends_on` — not
   vectorization. Batch axes mean "many independent copies"; parent axes
   mean "one object with conditional structure."
3. **Its operations are different**: contraction against parent *beliefs*
   (expected log-likelihood for FPI; Bayesian model averaging), conditioning
   on parent values, and conjugate count-updates via outer products with
   parent posteriors.

Workstream B hit this concretely: pymdp's likelihood tensors are
child-axis-*first* `(num_obs, *parent_sizes)` while our contract puts event
structure *last* — so `Dirichlet`-as-likelihood-prior had no honest
representation and the learning API stayed raw arrays.

## Options

**A. Generic `Conditional` wrapper.** `Conditional(family_params,
parent_shape)`: any unconditional family, with designated leading axes
reinterpreted as parent axes. One class; `Conditional[Gaussian]` gives
mode-indexed dynamics (what `mixture_rollout.propagate` already consumes
implicitly) for free. Risk: reinterpretation is the move the contract banned
(`to_event()`-style) — though here it would be explicit, typed, and once.

**B. Per-family classes.** `ConditionalCategorical`, `ConditionalDirichlet`
(later `ConditionalGaussian`): tailored ops, no reinterpretation semantics,
some duplication.

**C. Status quo.** Raw arrays at the model layer; nodes carry the
semantics. Zero new machinery; B's friction persists into every consumer.

## Recommendation: A′ — generic core, blessed specializations

A thin generic `Conditional` (distributions layer — it *is* math: a Markov
kernel) storing the table **child/levels-axis-first, parent axes trailing
the child axes** internally — the pymdp/SPM layout, so their model tensors
adopt zero-copy — but with the layout *never leaking*: all use goes through
methods, and `batch_shape`/`parent_shape` metadata make the axis roles
explicit. Family-specific ops live on small subclasses where tailoring
pays:

- `ConditionalCategorical`: `expected_log(parent_beliefs) -> Array`
  (the FPI contraction), `average(parent_beliefs) -> Categorical` (BMA),
  `condition_on(parent_values) -> Categorical`.
- `ConditionalDirichlet`: `expected_log()` (digamma table, what FPI wants
  when averaging over parameters), `update(child_obs, parent_beliefs, lr)
  -> ConditionalDirichlet` (the conjugate count update — replaces the
  array-valued API in `learning/dirichlet_updates.py`),
  `to_categorical_table()` (posterior-mean CPD).

Wiring names (which parent axis is which factor) stay node-layer; the
distributions layer knows only shapes and roles. (Alex's earlier
factor-graph language put Variable references on the factor object itself —
the JAX-native version of that design is exactly this split: object
references don't survive jit, so identity lives in the spec and compiles
away.)

**Where the generic wrapper applies**: `Conditional[Gaussian]` — mode-indexed
linear-Gaussian dynamics/emissions, formalizing the per-mode transition
stacks `mixture_rollout.propagate` already consumes, and the
mode-conditioned-dynamics half of the future switching node (the
rSLDS-style node where a discrete Markov factor gates continuous dynamics).
Later: `Conditional[MixtureGaussian]` (mode-indexed mixture emissions),
mode-indexed parameter priors (per-regime NIW/Wishart).

**Taxonomy note**: the tabular `Conditional` covers **discrete parents**
only (parent axes = table indices). Conditionals with *continuous* parents
— MNLR's P(z|x), linear-Gaussian maps P(x'|x) — are **functional
conditionals**: parametric maps with their own classes, not tables. Both
are Markov kernels; only one is an indexable array.

## The other four friction resolutions (proposed)

1. **Contract §3 wording**: "elementwise operations broadcast over batch
   axes; algorithms involving discrete selection (e.g. `reduce`) or
   heterogeneous-list inputs (e.g. `fpi`) batch via `vmap`." (B and C both
   flagged the current absolutism.)
2. **`Interval.clip_lower(x)`** method; `kl_to` uses it instead of raw
   `jnp.maximum` before construction.
3. **Weight gauge**: one spec sentence — `log_weights` are defined up to an
   additive constant; all methods must be gauge-invariant; methods may
   return any gauge (`reduce` returns normalized — allowed, not required).
4. **Leaf broadcasting**: keep strict leaf consistency (no implicit
   broadcast gauge). XLA fuses `broadcast_to` under jit, so the materialized
   copy C worried about is virtual in practice; simplicity of the contract
   wins. Revisit only if profiling ever says otherwise.

## Joint families (added after review — Alex's normalization-scope point)

P(X), P(X,Y), and P(X|Y) can share one table geometry; what differs is the
**normalization scope**:

| Family | Simplex scope | Structure metadata |
|---|---|---|
| `Categorical` | one trailing axis | shape-derived |
| `JointCategorical` | all `event_ndims` trailing axes jointly | static `event_ndims` (not shape-derivable — the case the contract's static-field clause exists for) |
| `ConditionalCategorical` | child axes, one simplex **per parent config** | child/parent split |

Conversion algebra (the reason the distinction pays):
`JointCategorical.to_conditional(child_axes)` (normalize along children),
`.marginalize(axes)`, and chain-rule product `Conditional × Categorical →
Joint`. Near-term consumer: learning cross-factor transitions
(`B_dependencies`) needs joint posteriors over parent configurations —
pymdp builds these cross-products explicitly; `JointCategorical` is the typed
home.

Same trio at the Dirichlet level, with the subtlety made explicit: a
**joint-Dirichlet** (one Dirichlet on the product simplex; what BMR over
joint counts consumes) is *not* the product of **conditional-Dirichlets**
(per-parent-config priors; what CPD learning consumes) — different priors,
both wanted.

## Resolutions from review (Alex, 2026-07-21)

- **Option A′ adopted** (pending anything reopening it in implementation).
- **Child-axis-first internal layout**: agreed — computationally immaterial
  (einsum is layout-agnostic, XLA fuses transposes); pymdp-convention
  consumers dominate.
- **`Conditional[Gaussian]` ships now** (it formalizes what
  `mixture_rollout.propagate` takes).
- **Naming: `Conditional`.**
- Prior-implementation comparison absorbed: parent identity by reference →
  identity in the spec layer, compiled away (see above).
- **Joint families added to scope** of this revision (section above).

## Appendix: agenda for the System design session (accumulated)

Decisions already taken in review (to be reflected in the contract revision):
- **Tuples over lists** for multi-modality observations/state collections
  (hashable, fixed pytree structure); `fpi` and envs converge on tuples.
- **No-op actions for uncontrolled factors stay** as the internal
  representation (static shapes, uniform vmap treatment — the pymdp move),
  but are **hidden at the spec layer**: a NodeSpec without `controlled_by`
  gets the trivial action axis inserted by the compiler; planners exclude
  no-op factors from enumeration. Users never see them.
- **Named dimensions**: nothing so far forecloses them. Policy: names live
  in the spec layer (`elements` labels, already planned) and at
  human-facing boundaries (metrics trees, dataset columns); runtime kernels
  stay unlabeled arrays. Revisit only if agent-written glue turns out to
  want runtime labels.

To design (RESOLVED — see `system.md` for the decisions):
- **Agent vs Env vs System** (Alex's framing): an AIF *Agent* is a System
  with preference structure — a C/H over outcomes, "something with a reward
  function" — whose outputs are driven homeostatically; an *Env* may be a
  plain forward process (ancestral sampling in a generative model), no
  preferences required. F's two-faces recommendation (same tensors run
  forward = env, run backward = agent) is the mechanism; Alex's point is
  that the C-structure is what *distinguishes* the faces. Candidate shape:
  `System` = structure + params + optional preference block; `as_env()` /
  `act()` faces; nesting via blanket-typed edges. F also recommends a
  declared action-space structure on `Env`.
- **Trajectory logging conventions** (G's warnings): T+1 obs vs T actions —
  decide once before any dataset accumulates; float32 casting at the
  logger; an attrs note for action alignment.
- **`Categorical.sample` dtype** pinned by the contract (int32).

## Migration cost if adopted

Additive: the `Conditional` core + `ConditionalCategorical` /
`ConditionalDirichlet` / `Conditional[Gaussian]` + `JointCategorical` (and
joint-Dirichlet where BMR needs it) + the conversion algebra; subsuming
`learning/dirichlet_updates.py`'s array API (kept as the internal kernel);
one adapter simplification in `tests/test_discrete_parity.py`; `fpi`'s
model inputs upgrade from `list[Array]` to conditional objects with the
array path retained; plus the four small contract amendments from §"The
other four friction resolutions" and the reviewed decisions in the
appendix (tuples, sample-dtype pinning). No behavior changes; all 182
tests stay green throughout.
