# Per-variable info gain — the objectives layer learns the generic reading

*Status: adjudicated (Alex + Fable, 2026-07-30) — V1–V8 all resolved;
drafted 2026-07-28. Pays the first [deferred]
roadmap item (owed from `parameter-learning.md` D6, gate: before any
public release). Gates the generic `infer(query, clamped)` layer, and —
per the 2026-07-28 sequencing decision — precedes the nesting-Systems
build, so that nesting namespaces the refactored selector surface
rather than the one being retired.*

## The debt being paid

D6 recorded it exactly: in a generic factor-graph treatment parameters
are just variables, so there is no `include_param_info_gain` at all —
there is one `infogain` term per variable, and a parameter's novelty
is the same *term* as a state's (never the same estimator; see V4).
The parameter-learning pass deliberately stopped one level short:
per-declaration selectors over one key space, resolved in one place
(`System._term_targets`), consumed by both planning paths. This
refactor re-bases the term tree on that resolver.

What exists today, concretely:

- Three static toggles on `System`: `include_state_info_gain`,
  `include_param_info_gain`, `include_state_entropy`, each
  `bool | tuple[str, ...]` over their own name space (modality names,
  declaration keys, factor names respectively).
- `_term_targets(params)` resolves all three to index tuples;
  `objectives/hybrid.make_step_terms` consumes indices only.
- The term tree is `term_name -> block_name -> Interval`, term names
  organized by EFE decomposition (`utility`/`state_info_gain`/
  `novelty`/... or `risk`/`ambiguity`/...), block names by
  modality/factor.
- The compiled path (`objectives/compiled.py`) hoists loop-invariant
  tables (A-novelty weights, entropies) out of the planning scan;
  transition novelty is per-action and stays inside.

## Decisions

### V1 — **RESOLVED**: What is a variable? (the namespace)

**Terminology (strict, FFG).** In this doc a **variable** is a node
one can hold beliefs about; a **factor** is a function node. The
declarations — likelihood, transition, initial prior — *are* the
factors. `pose_cell` is a state **variable** (this document uses
"variable" where pymdp says "state factor"), and
`transition:pose_cell` names the FFG **factor** giving its dynamics.
A learnable factor exposes its parameters as additional variables.

**Proposal.** The epistemic namespace is the union of:

- **state variables**, by bare name (`"pose_cell"`), and
- **parameter variables of Learnable declarations**, by the
  slot-qualified key already used by `lr_by_slot` and
  `include_param_info_gain` (`"likelihood:pose"`,
  `"transition:pose_cell"`).

Observations are not epistemic targets (they are the data), and
non-learnable declarations expose no parameter variable. The two
sub-namespaces cannot collide: state names are bare, parameter keys
are always slot-qualified. This is the same key space the spec unified
at D6 (parameter-learning.md:230–255), extended by the bare state
names — one grammar, no third vocabulary.

**Multiplicity: the posterior's factorization decides the variable
set.** One declaration key can cover several parameter variables — one
per *independent block of the declaration's parameter posterior*:

- `LinearGaussianObservation` with `noise: InverseGamma` and
  `head: MatrixNormal` — two independent posteriors, **two
  variables**: `likelihood:pose:noise`, `likelihood:pose:head`. (This
  ships in v0.3 and is currently unaddressable below the declaration —
  noise and head cannot even take different learning rates.)
- `LinearGaussianObservationMNIW` — head and noise coupled in one
  joint posterior, **one variable**: `likelihood:pose`.
- `ConditionalDirichlet` — formally a product of independent
  per-parent-configuration Dirichlets, but a *homogeneous* product
  whose novelty leaf already aggregates over it weighted by
  `q(parents)`: kept as **one variable** (per-column addressing is
  finer than any current use; revisit only on demand).
- A future mixture family (weights + component means + covariances)
  gets one variable per independent block, however it couples them.

Grammar: an optional third segment addresses a block within a
heterogeneous family; the two-segment key is a **group selector** for
all parameter variables of that factor. Segment order is
slot-first (`likelihood:pose:noise`, not `pose:likelihood:noise`) —
adjudication note: the variable-first order was considered and
rejected because not every variable has (say) a likelihood, so
leading with the factor type makes the key an explicit claim that
`pose` is the target of one, and it keeps the existing `lr_by_slot`
key grammar unchanged as the prefix. The extension lands once, in
the resolver (`_term_targets` / `_check_slot_key`), so
`lr_by_slot`/`decay_by_slot` inherit sub-block addressing in the same
change — closing the v0.3 rate-granularity gap as a side effect.

**Nesting constraint (forward-looking, costs nothing now).** Keys must
survive path-qualification on the left under the nesting graph-merge
(`"arm/likelihood:pose:noise"`). Bare-vs-qualified stays decidable
after a prefix; nothing else in this design inspects key internals.

### V2 — **RESOLVED**: The API shape (what replaces three toggles)

**Decision: one selector, `epistemic`, plus the decomposition toggle
it already respects.** Name adjudicated over `use_epistemic`: the
value is a selector (`bool | tuple[str, ...]`), not a switch —
`epistemic=(...)` reads as "the epistemic set is these variables",
and dropping the verb prefix advertises the switch-to-selector shift
that is the point of the refactor.

```python
epistemic: bool | tuple[str, ...] = True   # variables named per V1
```

- `True` — every eligible variable (current default behavior).
- `False` / `()` — pragmatic agent, no epistemic drive (the current
  all-toggles-off ablation).
- A tuple — exactly these variables carry epistemic terms.

`include_state_entropy` folds in rather than surviving separately: it
is the *continuous lowering* of state info gain (the hybrid epistemic
term for LG blocks, block-grain by design), not an independent drive —
selecting a continuous state variable in `epistemic` mounts its
entropy term **where no eligible channel observes it** — a
channel-observed variable gets the channel term only (the
one-estimator-per-variable rule, adjudicated at the v0.4 step-2
review: mounting both provably kills switching exhibits; doc
qualified 2026-08-19 after the mission build measured the unqualified
promise, friction log 12) — selecting a discrete one mounts its info
gain. Related identity, for à-la-carte users (cross-ref
hybrid-system.md D4): risk + entropy-leaf = −E_q[ln C], the
tax-free expected-preference form; a risk-family term taken alone
carries −H(q) (the "certainty tax", measured in the field —
friction log 11/13). The
three old toggles are deleted, not deprecated: the library is
pre-public (that is the point of the gate), and the only external
caller is the coverage port, whose migration is a two-line diff we
own.

**Alternative considered and declined:** keeping `infogain` and
`entropy` as two selectors, in case conflating them hides a real
modeling choice. It does not: a continuous state variable whose
entropy term you want off while its likelihood's parameters stay
epistemic is expressible under the single selector as
`epistemic=(..., "likelihood:pose")` — the parameter key without the
bare state name.

### V3 — **RESOLVED**: The term-tree shape (what the leaves are called)

**Decision.** Under `utility_infogain` the epistemic leaves unify
under one term name — spelled `infogain`, matching the
`param_infogain_*` selector spelling O4 already standardized, and the
`utility_infogain` decomposition name itself:

```
infogain -> variable_name -> Interval
```

- state leaves — under the default `predictive` estimator (V6), keyed
  by the observing channel: `infogain["vision"]` is today's
  `state_info_gain` leaf for that modality (or the entropy lowering
  when the observed block is continuous). Under V6's `posterior`
  estimator they are keyed by state variable: `infogain["pose_cell"]`.
  Bare keys cannot collide with the slot-qualified parameter keys
  either way.
- `infogain["likelihood:pose"]` — parameter novelty (today's
  `novelty` leaf), one block per parameter *variable* under V1's
  multiplicity rule (so a two-slot LG factor yields
  `infogain["likelihood:pose:noise"]` and `...:head` blocks, while
  MNIW's joint posterior yields the single `infogain["likelihood:pose"]`)
  — instead of the current per-kind lumping.

Under `risk_ambiguity` nothing changes structurally (`ambiguity` is a
per-modality property of the likelihood, not a per-variable drive);
the `epistemic` selector still gates which modalities' ambiguity
mounts, preserving today's Form-D behavior and the D2 coherence rule:
every appearance of `I(o;θ|s)` remains exactly one leaf, now named by
its variable.

*Landed 2026-08-19 (LG modality C designations,
`hybrid-system.md` amendment).* An LG modality's continuous
`obs_targets` designation lands on the **existing** term names, keyed by
modality exactly like the tabular ones — `utility["look"]` /
`risk["look"]`, beside the channel's `infogain["look"]` — so V3's
grammar absorbed the new leaves without a new term. The
`ambiguity["look"]` leaf that completes the decomposition follows this
section's reading to the letter: it is mounted for every LG modality
under `risk_ambiguity`, designated or not, because it is a property of
the likelihood rather than a per-variable drive.

Consumers of the old names (`state_info_gain`, `novelty`,
`state_entropy` as term names) migrate; `register_term` signs move
with them. Migration means *changing the consumer's code to the new
API* — the old term names and toggles are removed outright, preserved
nowhere (confirmed at adjudication; pre-public is the point of the
gate). Parity acceptance maps old-tree leaves onto new-tree leaves
bijectively and requires 1e-5 (bitwise where no reassociation).

### V4 — **RESOLVED**: One term, many estimators (the registry seam)

The refactor unifies *naming, selection, and tree placement* — never
the math. `infogain` dispatches per variable to an estimator chosen
by (variable kind, family):

| variable | family | estimator | status |
|---|---|---|---|
| state, discrete | Categorical | `predictive` — pymdp-parity bound, channel-keyed (V6) | exists |
| state, discrete | Categorical | `posterior` — canonical Bayesian surprise, variable-keyed (V6) | new, this refactor |
| state, continuous | LG mixture | entropy lowering (today's `state_entropy`) | exists |
| params of a tabular factor | Dirichlet | `_wnorm` novelty (A- and B-forms) | exists |
| params, LG noise block | IG/IW | **none** | loud error today |
| params, LG head block | MatrixNormal | **none** | loud error today |
| params, LG joint | MNIW | **none** | loud error today |

*2026-10-17:* the MNIW row has an estimator, `blended_entropy`, its leaf an estimate rather
than a bracket; an MNIW emission with discrete parents still raises (parameter-learning.md
D10).

**Proposal.** This dispatch table *is* workstream D's pending
"estimator registry," landed at its natural size: a static mapping
resolved at trace time (the (prepare, step) discipline from the
generic-infer roadmap entry — prepare hoists loop invariants, step
runs in the scan). Missing entries stay loud errors at planning time,
exactly the current `_param_novelty_targets(planning=True)` behavior —
deriving the IG/IW novelty form (closed-form scalar/diagonal, or the
Kolchinsky–Tracey bound pair as the tree's first non-exact leaf) is
**out of scope for this refactor** and becomes a registry entry
someone adds later without touching the walk.

### V5 — **RESOLVED**: Acceptance

1. Old-tree/new-tree leaf bijection at 1e-5 on both demo Systems and
   the T-maze suite; bitwise where the computation is untouched.
2. `epistemic=False` ≡ old all-toggles-off, bitwise.
3. Selector errors: unknown name, parameter key on a non-learnable
   declaration, bare name of a declaration slot — loud, at
   construction or first face call, matching `_check_slot_key` today.
4. Compiled-path invariance: A-novelty weight hoisting survives (same
   op count in the jaxpr, or measured µs parity on the T-maze
   benchmark); `metrics=False` G-fold path unaffected.
5. Coverage-port migration diff reviewed as part of acceptance (the
   customer call sites are the API's first users).
6. Notebook re-executed (adoption gate); its EFE-math section renamed
   leaves verified in prose.
7. Estimator pair (V6): on a single-modality discrete model the
   `posterior` leaves sum to the `predictive` leaf at 1e-5 (the bound
   is tight there); on a two-channel shared-parent model the
   `predictive` total strictly exceeds the `posterior` total (the
   overcount, demonstrated); `posterior` on a hybrid model raises
   loudly.

### V6 — **RESOLVED**: state epistemic leaves (channel/variable mismatch; the canonical estimator)

**The finding (build step 1 pre-read).** Today's state info gain
(`objectives/discrete.py:197`) produces one leaf per *observation
channel*: `I(o_m; s_D(m))`, the mutual information between modality
`m` and its dependency set of state variables — jointly. There is no
per-variable leaf in that math, so V3's original example
(`infogain["pose_cell"]` as a state leaf) assumed a keying the
implementation cannot supply without new math.

**The lineage, precisely.** The canonical EFE state-epistemic term is
`E_{q(o)}[KL(q(s|o) ‖ q(s))] = I(o; s)` — joint over all states and
observations, computable state-side (`H(q(s)) − E_o[H(q(s|o))]`,
posteriors per outcome) or observation-side (`H(q(o)) − E_{q(s)}
[H(p(o|s))]`, no posteriors). SPM's `spm_MDP_G` computes the
observation-side form over the **joint outcome space** — exact,
exponential in modalities. pymdp uses a per-modality sum, a computationally efficient form,
`Σ_m [H(q(o_m)) − E H(A_m)]`: the ambiguity half is exact (channels
conditionally independent given states) but `Σ_m H(q(o_m)) ≥
H(q(o))`, so the sum **overcounts redundancy between channels
observing overlapping variables** — an upper bound, exact for
channels with disjoint parents. We inherited the pymdp form knowingly,
at parity.

**The fact that decides the shape.** Under a mean-field family —
`q(s)` and every per-outcome posterior factorized — joint entropies
split per variable, so the state-side per-variable sum
`Σ_f [H(q(s_f)) − E_o H(q(s_f|o))]` equals `I(o; s)` **exactly within
the family**. The per-variable decomposition is not an approximation
for factorized beliefs; it is the canonical term, keyed the way the
generic reading wants. (pymdp's family is also mean-field across
factors — the same exactness is available there too.) The
price is computational: posteriors per joint outcome (SPM's cost),
and `q(s)`-dependence puts it inside the planning scan — unhoistable,
unlike today's loop-invariant ambiguity tables.

**Scope limit (not mean-field everywhere).** The hybrid path is
deliberately *not* mean-field: the switching observation's
pre-reduction mixture is a structured joint between mode and
continuous block (the correlation that makes fix credit-assignment
work). Across such a pair, per-variable marginals *undercount* — the
correlation information belongs to no marginal. The exactness claim
is discrete-path-only.

**Decision.**
1. **Term-level unification (option b).** One `infogain` term; the
   *selector* speaks variables (V1/V2 unchanged): a bare state name
   mounts every channel whose dependency set involves that variable
   (selection counts epistemic value wherever the variable is
   involved — deliberately not a partition). Under the default
   estimator, state leaf *blocks* stay channel-keyed: bitwise math,
   rename-only bijection (V5.1), per-channel ablation retained.
2. **The state estimator becomes the registry's first two-entry row
   (V4).** `predictive` — today's pymdp-parity bound: channel-keyed,
   hoistable, **the default**. `posterior` — the canonical state-side
   form: variable-keyed leaves, exact-in-family, joint-outcome cost,
   chosen statically per System; **eligibility gated to the discrete
   path** (loud error on hybrid models, per the scope limit above).
3. **Default revisited at the public-release gate**, informed by
   measured cost on realistic models — not before.

### V7 — **RESOLVED**: class selection via segment wildcards

**The ask (Alex, 2026-07-30):** turn info gain on for state variables
and off for parameter variables — pymdp's original granularity —
without enumerating names (enumeration is not robust: a later-added
variable silently isn't mounted, the "toggle lies" failure D6 exists
to kill).

**Decision.** The key grammar is positional (`slot:name:block`), so
`*` matches a whole segment, resolved at trace time against the
variable table:

- `epistemic=("*",)` — all state variables, no parameters (the pymdp
  granularity, robust to model growth);
- `epistemic=("*:*",)` — all parameter variables, no states;
- `epistemic=("likelihood:*",)` / `("transition:*",)` — A- / B-novelty
  classes (pymdp's per-slot flavor, free);
- composable with literal keys: `("*", "likelihood:pose")`.

**One syntax everywhere the key system appears** (adjudicated over an
epistemic-only fence): wildcards apply to `epistemic` AND the rate
mappings (`lr_by_slot`/`decay_by_slot`). The asymmetry that argued for
the fence — selection is an idempotent union, rate assignment must be
single-valued — is paid with a precedence rule instead: **specificity
= number of literal segments; most-specific pattern wins; ties raise**
(`likelihood:pose:noise` ⊳ `likelihood:pose` ⊳ `likelihood:*` ⊳
`*:*`; `likelihood:*` vs `*:pose` matching the same variable is a
tie → ValueError). This makes base-plus-override the idiom:
`{"*:*": 0.999, "likelihood:pose": 0.97}`. Consequence, accepted: step
1's group+sub-key conflict error is **revised** into precedence — the
group key is the base rate, the sub-key the override — since with a
specificity ladder in the language, raising there would be the
inconsistency.

Validation: **a wildcard never validates less than its literal part.**
Each literal segment is checked against its finite domain (slots,
declared names, family blocks) — `likelihod:*` raises as the typo it
provably is; literals that pin a specific declaration get the same
participation checks a literal key gets (`likelihood:pose:*` on a
point-valued `pose` raises "holds a point value"). Silence is reserved
for classes empty *through the wildcarded segment* (`likelihood:*`
with no learnable likelihoods — legitimately "all of none"). Full-
segment `*` only (no partial globs — name-coupling brittleness for no
expressive gain). Reinterpreting the deleted legacy labels was
considered and declined (a second vocabulary, strictly less
expressive).

**Docstrings are part of the deliverable:** `System.epistemic`,
`learn`'s rate mappings, and the resolver each carry the grammar, the
specificity ladder and tie-raise, the literal-validation rule, and
worked examples — agents building on the system get the whole
contract at the call site.

**Sequencing:** lands as a small follow-up after the step-2 rebase
(resolver + `_check_epistemic_key` + rate resolution + tests), since
step 2 owns those files while in flight.

### V8 — **RESOLVED**: observation variables join the namespace (`obs:` keys)

**How the loss slipped in (recorded as the lesson).** V2 collapsed a
*channel*-granular selector (`include_state_info_gain` over modality
names) into a variable-granular one, but the draft's V3 example
reasoned about variable-keyed state leaves that never existed, so the
expressiveness cost was invisible at adjudication; V6's pre-read fixed
the leaf keying but not the selector's reach, and the build's test
migration surfaced it (a T-maze ablation config became unreachable).
Lesson: when collapsing selectors, diff the reachable configuration
sets before/after — do not audit against the doc's model of the old
API.

**Decision (re-grounded 2026-07-30, Alex).** Not a term-address hack:
**observations are variables**, and V1's namespace was missing the
kind. Variables = states ∪ parameters ∪ **observations** — the FFG
reading (and e.g. Parr's generalized free energy, where observation
variables collapse to deltas once realized; the generic
`infer(query, clamped)` layer already speaks this vocabulary, and GFE
modeling stays a future *estimator* question, not an API question).
The MI leaf `I(o_m; s_deps)` is indexed equally by the likelihood
factor or the observation variable — selecting `obs:cue` *is*
selecting the observation variable for epistemic treatment, and it
mounts the leaf `infogain["cue"]`. Mechanics: a fourth key kind,
`obs:<modality>` (`kind="obs"` in the variable table — no new
machinery). Explicit prefix chosen over a names-unique-across-kinds
collision checker: duplicated state/obs names are idiomatic in simple
AIF models (the T-maze has both `location`s), and the prefix answers
"state or observation?" at the call site (V1's slot-first rationale).
Bare names remain state variables only. Composition: unions with all
other key kinds; `obs:*` under V7 restores pymdp's per-modality
granularity as a class; validation per V7's literal-projection rule
(unknown name raises, known-but-ineligible resolves empty, matching
bare state names). One-estimator-per-variable exclusivity (step 2's
judgment call) unchanged. Lands with step 3 (same resolver as V7).

**The duality this surfaces (V6 addendum).** pymdp's per-modality sum
is the per-variable decomposition on the *observation* side — exact
under a family factorized over observation variables (its overcount
is precisely the neglected o-side correlation); V6's `posterior`
estimator is the same decomposition on the *state* side. `predictive`
and `posterior` are duals, each exact-in-family on its own side of
the same mutual information — neither is a hack.

## Field note (2026-08-19, from the first external consumer)

If your agent prefers idling, print the tree (`score_policies`):
Dirichlet **parameter novelty** on learnable transitions ("keep the
learning opportunities open") and **belief-contraction state info
gain** (contractive dynamics destroy predictive entropy, so holding
still scores well from a broad prior) are the usual suspects — and
the per-variable selector is the knob (`epistemic=("done_0", ...)`
fixed both in one line for the mission model).

## Build plan

Three steps, each independently green; the walk is rewritten once, in
step 2.

1. **Resolver + grammar** — `_term_targets` learns the unified
   `epistemic` selector and the optional third key segment (variable
   set per V1's multiplicity rule); `lr_by_slot`/`decay_by_slot`
   accept sub-block keys in the same change; `_check_slot_key` errors
   extended (V5.3). Old toggles still work in this step (the resolver
   serves both), so it lands green without touching the tree.
2. **Tree rebase** — `objectives/` term names move to V3's shape;
   the estimator dispatch becomes V4's registry (a static mapping,
   (prepare, step) discipline); compiled path keeps its hoisting
   (V5.4). Old toggles deleted here; demos, tests, and the parity
   harness migrate in the same commit.
   2b. **The `posterior` estimator (V6)** — the registry's second
   state entry: canonical state-side info gain, variable-keyed
   leaves, discrete path only (loud error on hybrid), acceptance
   V5.7. Separable from 2 if the rebase runs long.
3. **Wildcards + observation variables (V7, V8)** — segment-`*`
   patterns in the epistemic resolver AND the rate mappings, with the
   specificity ladder (step 1's group+sub-key error revised into
   base-plus-override), literal-projection validation, the
   `obs:<modality>` key kind (observation variables join the
   namespace), and the docstring contract; lands between the rebase
   and the exhibit.
4. **Exhibit + customer** — notebook re-execution (V5.6), coverage
   port migration reviewed as acceptance (V5.5), benchmark run
   (V5.4's measured half).

## Addendum (2026-09-30): rationale from the docstring paring

*Recorded nowhere but the text the 2026-09-30 paring pass cut from
`system/selectors.py::_param_blocks`.*

- **V1's multiplicity rule, applied to `MLPConditional`.** A neural slot
  exposes one variable. Its weights are a single posterior-free trainable
  block with no independently addressable factorization to name. Per-layer or
  per-head rates would be a different grammar, not a finer reading of this
  one.

## Addendum (2026-10-02): a selection buys information about what it selects (V9)

*Adjudicated (Alex + Opus, 2026-10-01/02) on a consumer finding:
zeroth-harness's brief "Switching LG channels: whose information is the info
gain?" (its `docs/briefs/hybrid-aif-switching-infogain.md`). Not yet built.*

**The finding.** V6 decision 1 keeps each channel's leaf joint over its
dependency set, `I(o_m; s_D(m))`, and a bare variable name mounts every channel
that involves it. So a selection that names some of a channel's parents buys
information about the others too. On a switching LG channel the extra is
`I(y; mode | x)`, up to `H(q(mode))` per step; it rises when an unselected mode
is made *less* informative; and it reversed a consumer's fix. The same holds on
the discrete path. The joint form survived V6 because that refactor changed
naming and never math (V4), not because the partial-selection case was weighed.

**Decisions.**

1. **Selection-relative value, every family.** A channel's leaf is its
   information about the *selected* variables among its parents,
   `I(o; S ∩ D)`, with the unselected parents marginalised under the current
   belief. The chain rule with the selected parents first,
   `I(o; D) = I(o; S ∩ D) + I(o; D \ S | S ∩ D)`, attributes the remainder to
   the unselected parents. `epistemic=True` selects everything, so every
   default-configured model, pymdp parity included, is unchanged bit for bit.
2. **Keying unchanged.** Leaves stay channel-keyed (V6); a leaf's value is
   defined relative to the System's selection, which is static. The
   unselected remainder is not computed.
3. **Ambiguity splits the same way.** Under a partial selection a channel's
   ambiguity is `E H[o | S ∩ D]`, so `risk + ambiguity = −(utility + infogain)`
   keeps holding at both interval endpoints.
4. **Estimators on a switching LG channel.**
   - Mode sets only the noise (shared `H`, `c`): `H[y | x]` is the entropy of
     the `q(mode)`-weighted zero-mean noise mixture, one KT pair per step.
   - Only discrete parents selected: `I(y; S) = H[q(y)] − Σ_s q(s) H[q(y | s)]`,
     each conditional predictive a mixture over the factor's components; KT
     pairs, any per-mode `H_j`, `c_j`.
   - The factor selected with an unselected mode that changes `H` or `c`:
     `H[y | x]` depends on `x`. Estimated by sigma points over each component
     of `q(x)` with a KT pair at each point, then clipped into a closed-form
     guaranteed bracket: lower `H[y | x, mode]` (conditioning reduces
     entropy); upper the smaller of `H[y | x, mode] + H(q(mode))` and the
     Gaussian maximum-entropy bound at the expected conditional covariance
     (`Σ_j q_j R_j` plus the expected spread of `H_j x + c_j`, by concavity of
     `log det`). A `sigma_points` utility lands in
     `distributions/gaussian.py`, reusable for unscented prediction
     (`system.md`'s deterministic-node rule).
5. **Bounds: shipped as they are.** Shares are guaranteed intervals; the
   midpoint of a difference is the difference of midpoints, so subtraction
   widens the reported width but not the value compared against a price.
   Tighter mixture bounds (a direct bound on the information, or a better
   member of the pairwise-distance family) go to the roadmap as their own
   round; the build pins today's midpoint error against Monte Carlo.
   *Amended 2026-10-02 (Alex), after review:* in decision 4's third case (the
   factor selected, an unselected mode that changes `H` or `c`) the leaf's
   conditional entropy is the guaranteed bracket itself; the sigma-point
   estimate is dropped from the estimator (it was not an enclosure). A tighter
   guaranteed refinement for that case joins the tighter mixture bounds on the
   roadmap. `sigma_points` stays in `distributions/gaussian.py` for unscented
   prediction.
   *Amended 2026-10-02 (Alex), after a scoping round:* the roadmap item is
   narrowed to that third case. Cheap tightenings of the general mixture pair
   (the best Chernoff coefficient on a grid for the lower bound; the smallest
   of the KL bound, the moment-matched Gaussian maximum entropy and
   `Σ w H_i + H(w)` for the upper) moved widths by under 0.005 nats on the
   brief's mixtures and random ones; a literature survey found no family that
   fits mode-dependent covariances at `d ≤ 6` better; and quadrature point
   estimates (unscented, Gauss–Hermite) were *worse* than the KT midpoint on
   the brief's wide-around-narrow mixtures (errors to 0.6 nats against 0.04).
   The general bounds and the `Interval` type stay as they are. The candidate
   for the third case is a closed-form lower bound: the pointwise KT lower
   bound averaged over `q(x)` through Jensen's inequality, since the
   Bhattacharyya distance between two per-mode noise Gaussians is quadratic in
   `x` and its exponential integrates in closed form against a Gaussian.
   *Amended again 2026-10-02 (Alex), on a consumer measurement:* the general
   bounds are not fine at the consumer's dimensionality. On a 3-component 4-D
   belief through an `[I I]` head the KT intervals are 0.43–0.6 nats wide and
   even the non-switching leaf's midpoint is about 0.07 nats off, where the 2-D
   tests above showed 0.09–0.35 and 0.04; zeroth-harness measured a noise-only
   block share 0.03 nats above a provable ceiling. The roadmap item is reopened
   as research on mixture information at `d ≈ 4`. Two leads go with it. (1) The
   *data-processing ceiling*: with a shared head and an unselected noise that is
   Loewner-smallest, `I(y; x)` is at most the non-switching leaf at that noise.
   A clip on the info-gain leaf alone was built and withdrawn: where it binds it
   breaks `risk + ambiguity = −(utility + infogain)` at an endpoint, because
   intervals cannot carry the constraint that ties `H[q(y)]` to `H[y | x]`; a
   method that bounds the information directly can use it soundly. (2) An
   *amortized estimator* of the mixing term `I(Y; K)` (the component label's
   information), trained once over mixture parameters: a registry row chosen per
   System, reporting a zero-width `derived=False` interval as every learned row
   does (`amortization.md` A4), optionally clipped into the KT interval. No
   change to `Interval`; it is worth building only if it is accurate where
   quadrature is not. Consumers meanwhile keep noise ratios extreme, where the
   pair is tight.
   *Reopened as an option, 2026-10-02 (Alex):* decision 4's rejected option of
   reporting the third case's sigma-point estimate as an estimate was rejected
   partly on a false premise — that an interval which is not a bracket would be
   a first; `derived=False` already marks every learned row. Measured, the
   sigma-point estimate is no worse than the bracket's midpoint on the review's
   adversarial model (+0.15/+0.28/+0.32 both) and much better on random per-mode
   models (+0.03–0.13 against about +0.5). Candidate: an opt-in estimator row for
   that case, `derived=False` (static, so conservative where the exact path is
   taken), beside the guaranteed bracket as the default. Not built; no known
   consumer has the case. It does not reach the consumer's case, which is the
   first one (shared head): there `H[y | x]` is constant in `x`, so the error is
   the KT pair's own, and only the leads above address it.
6. **The discrete path in the same round.** `p(o | S ∩ D)` is the likelihood
   contracted with `q` over the unselected parents, exact; it is recomputed
   per planning step, so only partial selections pay it.

**Acceptance.** On one model per family: the selected share plus the
remainder equals the joint leaf (exact on the discrete path, within the
intervals on the switching path); the identity above under a partial
selection; the brief's Monte Carlo oracle inside the interval, for shared and
per-mode `H`/`c`; identical modes give a mode share of zero; an uninformative
extra mode with small weight does not raise the selected share; and
`epistemic=True` leaves every existing leaf bitwise unchanged.

### Build notes (V9)

- **Where the selection lives.** `selectors._channel_selections` resolves the
  `epistemic` tuple to one parent mask per channel (in `A_deps` order), or
  `None` when the channel is scored over its whole dependency set: under
  `True`/`False`, when every parent is selected, when none is (no state leaf
  mounts), and when the channel's observation variable is selected
  (`obs:<modality>` buys everything the observation reveals, V8's reading). The
  masks reach the estimators through `Statics.selected`; a `None` mask takes the
  pre-V9 code, so the joint leaves are bitwise unchanged (verified against the
  pre-V9 tree on the T-maze, the synthetic two-factor model and the mission
  model, both decompositions). Under `risk_ambiguity` a partial mask applies to
  the ambiguity of mounted channels only; an unmounted channel keeps its full
  ambiguity, the documented wedge of `epistemic=False`.
- **Discrete.** `p(o | S)` contracts the (mean) table over the unselected
  parents' beliefs; `E H[o | S]` replaces the hoisted conditional-entropy table
  for that channel, in its info gain and its ambiguity alike, on both carriers.
- **Switching LG, factor selected.** The head `[H | c]` is tested statically,
  per selected mode configuration. A shared head (no mode axes) takes the
  zero-mean noise mixture's KT pair, which is a guaranteed enclosure and is
  intersected with the guaranteed bracket below. A per-mode head is compared
  at run time, per selected configuration: where its values are exactly equal
  across the unselected modes (the brief's own `broadcast_to(eye, …)`
  declaration) `H[y | x]` is again constant in `x` and the same KT pair is
  used; otherwise, a difference of any size included, the leaf reports the
  guaranteed bracket itself, `[H[y|x,mode], min(H[y|x,mode] + H(q(u)), ½
  log|2πe E_x Cov(y|x)|)]` (decision 5's amendment). Both results are computed
  and selected with `jnp.where`, so shapes stay static.
- **The expected spread in the max-entropy bound.** With `y | x, u ~ N(G_u x̃,
  R_u)`, `G_u = [H_u | c_u]`, `x̃ = (x, 1)`, the law of total covariance over the
  unselected modes gives `Cov(y | x) = Σ_u q_u R_u + Σ_u q_u (G_u − Ḡ) x̃ x̃ᵀ
  (G_u − Ḡ)ᵀ`, `Ḡ = Σ_u q_u G_u`. The spread is linear in `x̃ x̃ᵀ`, so its
  expectation under `q(x)` needs only the second moment `E[x̃ x̃ᵀ] = [[Σ_k w_k
  (Σ_k + μ_k μ_kᵀ), μ̄], [μ̄ᵀ, 1]]`, the component moments averaged over the
  mixture weights. Concavity of `log det` then bounds `E_x H[y | x] ≤ ½
  log|2πe E_x Cov(y | x)|`.
- **Measured.** On the brief's toy the block share's midpoint is 1.010 / 1.011
  / 1.018 nats at held noise 0.15 / 0.3 / 10 m (Monte Carlo 0.993 / 1.007 /
  1.018), below the no-extra-mode 1.051. On a two-mode-parent model with
  per-mode gain, offset and noise, the modes-only shares' midpoint errors are
  +0.03 and +0.11 nats, most of it the predictive entropy's KT pair, which the
  joint leaf carries too (its own error there is +0.13); the factor-selected
  shares, which take the bracket, err by +0.55 and +0.50. On the reviewer's
  sign-flip case (`H = ±1`, `R = 0.01`, `q(x) = N(0, σ²)`) the bracketed share
  is [0.458, 1.151] / [1.614, 2.308] / [2.709, 3.402] at σ = 0.3 / 1 / 3
  (Monte Carlo 0.653 / 1.677 / 2.731). The tests pin these.

## Addendum (2026-10-17): where `risk_ambiguity` does not follow the selection

*Recorded (Alex, 2026-10-17) as debt, not fixed.* V9's principle is that a selection buys
information about what it selects. Under `utility_infogain` the selection decides every
epistemic leaf. Under `risk_ambiguity` it holds only in part, measured on the T-maze and on a
linear-Gaussian model with a Gaussian-carried factor:

1. **An unselected channel keeps its full ambiguity.** `epistemic=True`, `False` and a partial
   tuple mount ambiguity on every channel, tabular and linear-Gaussian alike. A selection
   narrows only the ambiguity of channels whose information-gain leaf it mounts (V9's partial
   masks). So under `risk_ambiguity`, `epistemic=False` does not remove epistemic value.
2. **This document disagrees with itself on it.** V3's "the `epistemic` selector still gates
   which modalities' ambiguity mounts" is stale: the 2026-08-19 landing note under it ("mounted
   for every LG modality under `risk_ambiguity`, designated or not") and V9's build notes ("an
   unmounted channel keeps its full ambiguity, the documented wedge of `epistemic=False`")
   describe the code.
3. **On a Gaussian-carried factor the two decompositions differ by a channel information
   gain.** That path mounts no channel leaf, only the factor block's entropy leaf. Under
   `risk_ambiguity` it adds each channel's ambiguity as well; under `utility_infogain` it does
   not, so the two differ by exactly the unmounted channels' information gain (history entry on
   the linear-Gaussian ambiguity landing).

4. **On a channel without a preference, `risk_ambiguity` drops part of the information
   gain.** Risk is mounted only on designated channels, ambiguity on every channel, so an
   undesignated channel's `−H[q(o)]` appears nowhere. On the T-maze, the policy that visits
   the cue gains 0.595 nats of information about the context under `utility_infogain` and is
   rewarded for it; under `risk_ambiguity` the same policy is penalised by its ambiguity,
   0.098. With a flat preference on the cue channel its risk would carry `−H[q(cue)] =
   −0.693`, and `0.693 − 0.098 = 0.595`. So the two decompositions are equal only on
   designated channels, and can choose different policies. `composing-systems.md`'s "equal by
   construction" was qualified the same day. *Checked (2026-10-17): not a pymdp parity gap.*
   pymdp 1.0.3 has no risk-ambiguity form; its expected free energy
   (`control.compute_neg_efe_policy`) is utility plus information gain over every modality,
   which `utility_infogain` matches. The gap is this library's own `risk_ambiguity`.

Nothing here makes ambiguity "always paid". Per channel `o` and selected parents `S`, both
decompositions regroup three primitives: the cross-entropy against a preference `CE = −E log
C(o)`, the predictive entropy `H[q(o)]` and the conditional entropy `E H[o | S]`. `utility =
−CE` and `infogain = H[q(o)] − E H[o | S]`; `risk = CE − H[q(o)]` and `ambiguity = E H[o |
S]`. A channel's epistemic content is `I(o; S)` under either; `utility_infogain` holds it in
one leaf, `risk_ambiguity` spreads it over two, `−H[q(o)]` inside risk and the ambiguity. What
the selector picks is the same in both, the `S` side (`obs:<m>` taking all of `o`'s parents).
The fix that makes the two agree under any selection: a channel whose `I(o; S)` is selected
gets risk `CE − H[q(o)]` (the `CE` absent without a designation) and ambiguity `E H[o | S]`;
any other channel gets risk `CE` alone and no ambiguity. *Prioritised (Alex, 2026-10-17):
next after V10.*

## V10 — **RESOLVED** (2026-10-17): selecting dimensions within a continuous factor

*Adjudicated (Alex + Opus, 2026-10-17) on a consumer case.* An agent learns a map's row offset
`b` by carrying it as a static dimension of its pose factor: a pile reading `y = n·p + b − n·a
+ e` couples `b` to the pose, so the two must share one Gaussian block (across factors,
mean-field inference would drop their covariance, the thing that lets a reading teach `b`).
Its planning value is the information about `b` alone, the pose marginalised, ½ log(1 + σ_b² /
(σ_r² + nᵀΣ_pose n)) for one reading. The selector names whole factors; this decision lets it
name dimensions.

1. **Keys `factor:dim`, with optional names on the System (Alex).** `System(..., dims={"pose":
   ("x", "y", "yaw", "b1", "b2")})`; `epistemic=("pose:b1", "pose:b2")`. Keys on one factor
   form one sub-block (selection is a union); `pose:*` is the whole factor. Undeclared
   dimensions are named `dim0`, `dim1`, … by position, so a key works without a declaration;
   declaring names is recommended, since positional names change meaning when the state is
   reordered. Index keys (`pose:3`) and declared groups (`blocks={"pose": {"offsets": (3,
   4)}}`) were set aside: the first does not survive a reordering, the second fixes the
   groupings in advance. A key's first segment is a reserved kind word or a factor name, so a
   factor may not be named after a kind word.
2. **The leaf (Alex).** Keys stay as they are (V9 decision 2): the value is relative to the
   selection. On a Gaussian-carried factor the block leaf becomes `H[q_epi(x_S)] −
   H[q_cond(x_S)]` under the epistemic chain, the information this step's expected readings
   carry about `x_S` with the other dimensions marginalised, and `−H[q(x_S)]` without it; both
   exact, read off the marginal covariance `Σ_SS`. Under `risk_ambiguity` the factor's
   channels' ambiguity becomes `E H[y | x_S]`, closed form since `y | x_S` is Gaussian. On a
   mixture-carried factor a sub-block selection is refused: its channel leaf would need a
   Kolchinsky–Tracey bracket over `y | x_S` per component and a within-factor mask beside V9's
   parent mask; that is the next step when a consumer needs it.
3. **Where names live (Alex): on the System,** beside `factor_names`, as static structure. A
   model description gains an optional `dims` list beside a real variable's `dim`, written only
   when the names were declared, so documents without names round-trip unchanged. Composition
   prefixes the factor segment only (`arm.pose:b1`). Names are unique within a factor, match its
   dimension count, and are single segments without `:`, `.`, `*` or brackets; `dimN` may be
   declared only at position `N`.
4. **Constants tracked as state take a small process noise** (*revised the same day, Alex, on
   an audit; first resolved as "a dimension with zero process noise is a legitimate declaration
   of a static quantity"*). An unknown constant is uncertain but time-invariant, so as a state
   dimension its natural declaration has no process noise. The library cannot hold it so: a
   belief's covariance must stay positive definite, and a sampled rollout
   (`predict_beliefs(key=...)`, `update_empirical_prior(key=...)`, a prediction after
   `infer_states(key=...)`) collapses the belief to a point, after which a noiseless dimension
   stays a point and the next prediction's Cholesky is all NaN. An audit of every consumer of a
   linear-Gaussian transition's process noise found no other use of it alone. So a factor of
   the process noise with a zero on its diagonal is refused wherever beliefs are propagated
   (every face that predicts, `replay` and the agent driver included, under `jit` too), and a
   constant takes a small noise (the consumer's 1e-4 m per √s drifts about 6 mm over an
   hour against a 0.3 m prior). This is a numerical concession, not a modelling claim: a
   noiseless state dimension and a parameter are the same object, an uncertain constant; the
   state route is taken because only within a state block does the library keep the
   covariance between the constant and the pose. Parameters are unaffected (a clamped one is
   known, a learned one changes only by `learn`). A generative process (`as_env`) only samples
   and may be noiseless; a model description's `cholesky` domain keeps its strictly positive
   diagonal for every factor, as before. A continuous control carried as a belief now
   adds its spread to the process noise as a non-square factor, exact whatever the sum's rank.

**Acceptance.** On the Gaussian path the sub-block leaf equals the hand-computed marginal
entropy drop under both `planning_belief` settings, and on the consumer's shape the closed form
above; selecting every dimension equals selecting the factor, bitwise; expected position
readings earlier in the plan raise the offset's leaf; ambiguity under `risk_ambiguity` equals
`E H[y | x_S]` by hand; a mixture-carried factor, an unknown dimension, malformed `dims=` and a
factor named after a kind word are refused; a description with `dims` round-trips and one
without writes none; a merge gives `arm.pose:b1`; a belief-valued continuous control beside a
small-noise dimension gives finite results, and a zero process noise is refused.

### Build notes (V10)

- **Where the selection lives.** `System.dims` holds the names; `selectors._epistemic_variables`
  adds one `<factor>:<dim>` key (kind `dim`) per dimension of each continuous factor on the
  linear-Gaussian path, and `selectors._dim_selections` resolves the selector to one sorted
  tuple of positions per factor, or `None` for a factor selected whole or not at all. A
  dimension key mounts what its factor's name mounts. The tuples reach the block estimator
  through `Statics.dims` and the ambiguity through `lg_ambiguity(dims=)`, both from
  `make_step_terms(factor_dims=)`; with no factor selected in part both stay empty and the
  pre-V10 code runs.
- **Estimators.** The block leaf reads `continuous.marginal(q, S)`, `N(μ_S, Σ_SS)`, on each
  belief it already read; the ambiguity is `continuous.sub_block_obs_entropy`, on the
  predictive belief as every ambiguity leaf is.
- **`obs:` beside a sub-block.** *(Superseded by V11: the path now has channel leaves, and
  `obs:<m>` beside a sub-block of the factor its channel observes is refused; V11 build notes,
  "Sub-blocks".)* `obs:<m>` on a channel of a Gaussian-carried factor mounts
  nothing (that path has no channel leaf), so it does not widen the factor's sub-block back to
  the whole: under `risk_ambiguity` the channel's ambiguity stays `E H[y | x_S]`. Under
  `risk_ambiguity` that ambiguity now depends on the belief (it grows with the uncertainty of
  the unselected dimensions), where the whole-factor ambiguity on this path is constant; the
  block leaf beside it is the addendum's item 3, carried over.
- **Static dimensions.** `hybrid._push_input` returns the innovation after a belief-valued
  control as the non-square factor `[Q_chol | G S^½]`, whose product with its transpose is
  `Q + G S Gᵀ`, the only thing `predict` reads: no factorisation of the sum is taken, so a
  singular sum is exact, with finite gradients (reachable only through a control's own
  singular spread, since a zero process noise is refused). (A first version factored the sum,
  with a pivot trick for a zero diagonal and an eigendecomposition otherwise; a review found its
  gradients NaN off the axes and its always-run eigendecomposition 38–85% slower.)
  `hybrid._with_process_noise` refuses, inside `make_transition`, a process-noise factor with
  a zero on its diagonal: a `ValueError` for a concrete factor, `equinox.error_if` for a traced
  one (decision 4). A first placement in `validate_structure` checked concrete values only, so
  `replay` and the agent driver, which trace their params, skipped it, and it refused a
  noiseless `as_env`, which needs no positive-definite belief.

## V11 — **RESOLVED** (2026-10-18): each decomposition ablates its own epistemic leaf

*Adjudicated (Alex + Opus, 2026-10-18), resolving the 2026-10-17 addendum above.*

1. **The selector governs each decomposition's own epistemic leaf (Alex).** Under
   `utility_infogain` it mounts `infogain`, `I(o; S)`; under `risk_ambiguity` it mounts
   `ambiguity`, `E H[o | S]`. Risk is never touched by the selector: under the risk-ambiguity
   reading `−H[q(o)]` belongs to risk, `KL(q(o) ‖ C)` rewarding a diversity of expected
   outcomes alongside the match to the preference, and is not epistemic. Where a channel's
   epistemic leaf is mounted, under any selection `S`, `risk + ambiguity = −(utility +
   infogain)` holds per channel; where it is ablated the two forms differ by exactly that
   channel's `H[q(o)]`, because ablating "information gain" and ablating "ambiguity" are
   different ablations of different readings (on a Gaussian factor several channels read, the
   ablated readings' entropy given the selected ones', decision 3). The addendum's item 1 is
   this, fixed; item 2's stale V3 sentence is superseded by it.
2. **A channel without a designation carries an implicit flat preference (Alex).** A flat
   preference is not inert, it is a pull toward flatness, so such a channel has a risk: under
   `risk_ambiguity` its leaf is `−H[q(o)]`. The preference's normaliser is dropped in both
   forms, as pymdp's unnormalised log-preferences drop it (`C = 0` gives utility 0): it is the
   same for every policy, and a flat preference over a continuous channel has none (it is
   improper), so dropping it is the one rule that treats every channel alike. Under
   `utility_infogain` the channel's utility is then 0 and no leaf is mounted, as today. A
   declared uniform preference is a normalised distribution, and its leaf keeps the
   normaliser. The addendum's item 4 is this, fixed: with everything selected the two forms are
   equal on every channel, designated or not, and the T-maze's cue policy scores the same under
   both.
3. **The Gaussian-carried path is per channel, with the readings' overlap its own terms
   (Alex).** That path's information gain was one leaf per factor: under the epistemic chain the
   entropy drop `H[q_epi(x)] − H[q_cond(x)]`, the joint information `I(Y; x)` of the step's
   readings; without the chain, `−H[q(x)]`, which is not information gain. It becomes one leaf
   per channel, `I(y_m; x_S)`, keyed by modality as on every other path, plus factor-keyed
   terms for what a per-channel sum miscounts. *(Revised the same day, Alex, on an adversarial
   review: first resolved as one term, `observation_redundancy`, the total correlation of the
   selected channels' readings, mounted in both forms. That made the risk side follow the
   selection, which decision 1 forbids, and called a quantity that can be negative a
   redundancy.)* The sensors' noises are independent given the state, so with `S` the selected
   dimensions (the whole factor unless V10 selects a sub-block), `Y` the readings of the
   channels concerned and `TC` the total correlation `Σ_m H[y_m] − H[Y]`:

   - `I(Y; x_S) = Σ_m I(y_m; x_S) − (TC(Y) − TC(Y | x_S))`;
   - `KL(q(Y) ‖ ∏_m C_m) = Σ_m KL(q(y_m) ‖ C_m) + TC(Y)` (a flat preference's normaliser
     dropped, decision 2);
   - `E H[Y | x_S] = Σ_m E H[y_m | x_S] − TC(Y | x_S)`, and `TC(Y | x) = 0`.

   Two terms carry the corrections, each registered with sign `+1`. Under `utility_infogain`,
   **`observation_interaction`** is `TC(Y) − TC(Y | x_S)` over the selected channels, the
   interaction information of their readings about `x_S` (for two readings exactly `I(y_1; y_2)
   − I(y_1; y_2 | x_S)`; for more, `Σ_m I(y_m; x_S) − I(Y; x_S)`): it follows the selection and
   is ablated with information gain. On the whole factor it is the total correlation `TC(Y) ≥
   0`, the information the sensors share and a per-channel sum counts twice (redundancy); on a
   sub-block it can be negative, where the readings carry information about `x_S` only
   together (synergy, which a per-channel sum misses). Under `risk_ambiguity`,
   **`observation_correlation`** is `TC(Y)` over every linear-Gaussian channel on the factor
   present at that step, the part of the joint risk a per-channel sum leaves out: mounted
   wherever two or more such channels read the factor, whatever the selection (risk is never
   the selector's), on the predictive belief. Beside it, `observation_interaction` is the joint
   ambiguity's correction `−TC(Y_S | x_S)` over the selected channels, mounted only when a
   strict sub-block is selected (it is 0 on the whole factor) and ablated with ambiguity. So G
   is the joint expected free energy in both forms, and with every channel selected (on the
   whole factor or a sub-block) `G_ig = Σ CE − Σ_m I(y_m; x_S) + TC(Y) − TC(Y | x_S) = G_ra`.
   With the channels `M_S` selected, `G_ig − G_ra = H[q(Y)] − H[q(Y_{M_S})]` per step: the
   entropy of the ablated channels' readings given the selected ones', which is their joint
   entropy `H[q(Y)]` when none is selected; on channels that share no factor it is decision
   1's `Σ H[q(o)]`. Under `planning_belief="posterior"` the information gain and the
   `utility_infogain` interaction read the epistemic chain and risk, ambiguity and the
   `risk_ambiguity` terms the predictive one (planning-with-observations.md D1), so the two
   forms are not equal there, by that design. A per-block leaf (keys colliding with a factor's
   state-target risk) and a per-channel sum without the corrections (pymdp's convention,
   a per-channel sum that counts overlapping sensors once per channel) were set aside.
4. **The drive toward low predicted uncertainty becomes an opt-in row, `uncertainty` (Alex).**
   The chain-less leaf `−H[q(x)]` was filed under `infogain` without being information gain,
   and two demo sections rely on the drive it gave (the thrust model's price for injected noise,
   the sensor detour's contrast between the two settings). It moves to a named row selected
   with `terms=("uncertainty:<factor>",)`, its leaf `H[q(x)]` (or `H[q(x_S)]` on a sub-block)
   charged as a cost: a sibling of `occupancy` in the entropy family of objective-terms.md's
   lattice, the opposite sign on a different quantity (the belief's entropy, not the
   transition's conditional entropy). A factor no channel observes moves with it (Alex: not to
   be left hanging): it has no information to gain, so it carries no `infogain` leaf, and its
   entropy-lowering leaf becomes its `uncertainty` row, a Gaussian- or mixture-carried
   factor's entropy (a Kolchinsky–Tracey pair on a mixture) alike. `infogain` then holds
   information gain only. A literal `epistemic` key naming such a factor raises, naming
   `terms=("uncertainty:<factor>",)`, so a selection never silently loses the leaf it asked
   for; `epistemic=True` and wildcards resolve to nothing there.
5. **Release (Alex).** `risk_ambiguity` scores change wherever a channel is undesignated or
   unselected, and Gaussian-path `utility_infogain` scores change under the default
   `planning_belief="prior"`, which no longer carries the entropy drive unless `uncertainty` is
   selected. A behaviour change for the migration note; no switch restores the old forms.

6. **Per-step state targets inside branching planners take the cross-entropy form (Alex).** The
   sophisticated planner scores each step after a branch on that branch's updated belief, and a
   state target's KL there charges the branch for being narrow: averaged over outcomes,
   `E_o KL(q(s | o) ‖ C) = KL(q(s) ‖ C) + I(s; o)`, so knowledge gained at a branch is charged
   at every later step and credited once. A robot 50/50 between two identical rooms, a sign
   that tells which, a flat state preference and nothing it does depending on the answer: reading
   the sign cost `(k − 1) log 2` for `k` later steps. Observation targets do not have this: per
   branch and step `risk + ambiguity = −(utility + infogain)` with utility linear in the belief,
   so knowledge only uses up later information gain; a state target's entropy has no such
   partner, since a state is not observed. The literature does not meet the case: sophisticated
   inference (Friston et al. 2021) scores outcome risk at every step and states only through
   priors over final states. So inside a branching planner a per-step state target is scored by
   its cross-entropy, `−E_q log C(s)`, linear in the belief and so charging nothing for
   knowing, while the terminal target keeps its KL (one charge against one credit, as with final
   states in the literature) and open-loop planners are unchanged (they read the predictive chain,
   planning-with-observations.md D5). What it gives up is the de-risking entropy on intermediate
   steps in closed loop only. Scoring the risk on the outcome distribution across branches (the
   branches' beliefs mixed by their outcome probabilities and the planner's choices) keeps that
   term without the charge, but a risk on a mixture is not an expectation over branches, so it
   breaks the tree's bottom-up fold; it is the recorded upgrade if de-risking inside closed-loop
   search is needed. *(2026-10-19: the terminal clause is superseded by decision 8.)*

7. **A Gaussian emission over categorical parents gets an information-gain estimator (Alex,
   on the review).** Decision 1 needs every channel to carry an epistemic leaf the selector can
   mount; the emission had only an ambiguity leaf, mounted whatever the selection, and no
   information gain, so `utility_infogain` gave a structure-learning agent no drive toward
   readings that tell its levels apart. Its leaf is `I(y; z_S) = H[q(y)] − Σ_s q(s) H[q(y |
   s)]`, `z_S` the selected parents: `H[q(y)]` is the predictive mixture over the parents'
   configurations, a Kolchinsky–Tracey pair, and the conditional term is exact given every
   parent and a KT pair per configuration given some (V9's "factor unselected" estimator, the
   emission having no state). A matrix-normal inverse-Wishart emission is read through its point
   lowering, as its ambiguity is. It then follows decisions 1 and 2 like every channel: the
   selector mounts its `infogain` or its ambiguity, and it carries the flat-preference risk
   `−H[q(y)]` (a preference over it is refused, structure-learning.md S7).

8. **Inside branching planners the terminal target takes the cross-entropy form too (Alex,
   2026-10-19; the deliberation is recorded below, "V11 — decision 8").** Decision 6 kept the
   terminal target's KL. The tree reads it on the deepest children before their branch, so a plan
   that reads at its last step escapes the terminal charge that reading earlier pays, and the
   planner postpones reading by the last step's information gain about the targeted factor.
   Inside a branching planner the terminal target is now scored by its cross-entropy `−E_q log
   H(s)`, keyed `terminal_risk[<factor>]` as before. A conditional terminal target is scored by
   its negative expected log-probability under the mean-field final belief, `−Σ q(s) Σ_pa Π
   q(pa) log h(s | pa)`, linear in each factor's belief; the resolved target `Σ_pa h(s | pa) q(pa)` that
   open loop scores would charge, by Jensen, for learning the parents. Every quantity the tree
   evaluates per branch is then a preference's energy or a genuine information term, and the
   value of reading comes from the observation terms, credited once.

   What is given up, in closed loop only, is the de-risking entropy `−H[Q(s)]`, with `Q` the
   spread of the agent's end states: the branches' beliefs mixed by their outcome probabilities
   and the planner's choices. The cross-entropy is linear, so the tree scores `−E_Q log H(s)` on
   that mixture exactly, and only the entropy is dropped. A broad preference therefore no longer
   favours a plan that spreads its end states to match it, whether the spread comes from the
   dynamics (with the room known and a flat target, shuffling it to 50/50 over staying) or from
   acting on an answer ("go where the sign says" over "always go left"). What it does not
   change is the value of reading: where later actions do not depend on the answer, the mixture
   is the same whether or not the agent reads, so the dropped entropy cannot favour either plan.
   Decision 6 had already given up the same term on per-step targets. Open-loop planners keep
   the KL on the predictive belief, which is that mixture (planning-with-observations.md D5),
   and so keep the term.

   The route back, unbuilt. For a fixed mixture `m` of the end-state beliefs, `−H[m] = Σ_b w_b
   E_{q_b} ln m(s)`, so with `m` held fixed the spread term decomposes over branches and the
   tree folds again. `m` depends on the choices the fold makes, so it would be iterated: fold with
   `m` fixed, recompute `m` from the chosen policy, one tree search per iteration and root action.
   The fixed-`m` surrogate is a lower bound on the true score (Gibbs: `E_m ln m ≥ E_m ln m₀`), so,
   unlike EM, an iteration is not guaranteed to improve it; it is a heuristic fixed point, to be
   built if distribution-matching state targets are wanted in closed loop. Scope as decision 6:
   a planner registered `branching=True` on a System that declares `branch_on`. *(2026-10-19: for linear-Gaussian factors the end-state spread is computable
   in closed form without a tree; `closed-loop-planning.md` L6.)*

**Acceptance.** Under full selection the two forms give the same G at both interval ends on the
T-maze, a switching linear-Gaussian model and a Gaussian-carried model with two overlapping
sensors (under `"prior"`), also with a sub-block selected, and on a growing-mixture model's
emission; per channel under any partial selection; under ablation they differ by exactly `Σ
H[q(o)]` over the ablated channels where these share no factor, and by `H[q(Y)] − H[q(Y_S)]` on a
Gaussian factor several channels read; on the Gaussian path `Σ_m I(y_m; x_S) −
observation_interaction` equals the old block leaf under the epistemic chain, and the interaction
equals `TC(Y) − TC(Y | x_S)` computed by hand, negative on a synergistic pair;
`observation_correlation` equals `TC(Y)` by hand whatever the selection; `epistemic=False`
leaves risk, `observation_correlation` included, untouched; `uncertainty` equals the old chain-less leaf with the opposite sign, on observed
and unobserved factors, Gaussian- and mixture-carried; a literal key naming an unobserved factor
raises; the demo's
Sections 2 and 10 re-executed with their prose rewritten to the new leaves, brought for review;
in the sophisticated planner, the two-room case no longer charges for reading the sign, its
terminal KL is kept (superseded by decision 8), and open-loop scores are unchanged. *(Amended 2026-10-19, decision 8:)*
in the sophisticated planner the two-room terminal target charges reading first and reading
last alike; with a sign readable at the first step only, reading beats waiting by `log 2` at
every horizon, with and without a terminal target, where the per-branch KL scores reading `(k −
2) log 2` above waiting, `(k − 1) log 2` with a terminal target (a charge at each of the `k − 1`
later steps and at the horizon, less the one credit); a conditional
terminal target's score averaged over branches that learn its parent equals its score at the
averaged belief; both carriers give the same terminal leaf.

### Build notes (V11)

- **Where the selection lives.** `selectors._eligible_state_blocks` makes every linear-Gaussian
  channel observing a continuous factor eligible, a Gaussian-carried one included (and a
  Gaussian emission over categorical parents alone, decision 7), and returns the continuous factors
  no eligible channel observes in place of the entropy-lowering blocks. `_term_targets` returns
  three tuples, the entropy-lowering one gone; a literal key naming an unobserved factor (or one
  of its dimensions) raises a `ValueError` naming the row, after every key has been checked, so
  an unknown key is still reported first. `make_step_terms`, on both carriers, mounts
  `ambiguity` on `state_infogain_modalities` only, and under `risk_ambiguity` a `risk` leaf
  `−H[q(o)]` on every undesignated channel (`discrete.flat_risk`; on a linear-Gaussian channel
  `−continuous.predictive_entropy`, exact on a Gaussian belief and a Kolchinsky–Tracey pair on a
  mixture predictive). The registry's `(state, continuous)` estimator is gone. A literal key on
  a factor observed only by channels without an estimator (a neural likelihood) names them in
  the refusal. `composition.resolved_epistemic` skips such a factor (and its dimensions) when it
  expands a component's `True` or a pattern, so a mixed merge does not hand the composite a
  literal key it would refuse.
- **The corrections.** `observation_interaction` and `observation_correlation`, both
  registered with sign `+1`, are computed in `make_step_terms` (`continuous.
  observation_interaction`, `observation_correlation`, `conditional_correlation`), not by an
  `infogain` estimator, over the linear-Gaussian channels observing a Gaussian-carried factor
  alone: the interaction over the mounted ones (two or more; under `risk_ambiguity` only with a
  sub-block), the correlation over all of them under `risk_ambiguity`. The `utility_infogain`
  interaction reads the epistemic belief, the `risk_ambiguity` terms the predictive one. A
  channel absent at a step is replaced by an independent unit-noise reading with a zero head,
  which adds nothing to any total correlation, so the shapes stay static.
- **Sub-blocks.** A V10 sub-block now scores each channel's `I(y_m; x_S)` (with
  `sub_block_obs_entropy`) and the interaction `TC(Y) − TC(Y | x_S)`. *(Adjudicated, Alex, on
  the review:)* `obs:<m>` on a channel whose observed factor the same selection selects in part
  is refused (`selectors._dim_selections`, a `ValueError`): `obs:<m>` asks for everything the
  readings reveal, the sub-block for the information about `x_S` alone, and a channel carries one
  leaf. V10's build note, under which `obs:<m>` there mounted nothing, is superseded.
- **The row.** `uncertainty` is a `STATE_TARGET` row on the `gaussian` and `mixture` families,
  `derived=True`, keyed by the factor, reading the predictive belief under either planning
  belief. The `terms=` grammar of a state-variable row takes `<row>:<factor>:<dim>`; keys on one
  factor merge into one sub-block (`Mount.dims`), every dimension is the whole factor, and a
  mixture's marginal is its components' marginals. A description writes the key's tail as the
  term's `on`.
- **Consumers.** `examples/hill_lg.py` declares the row (`include_entropy`) beside the
  position's information gain (`epistemic`); `examples/hybrid_switching.py`, whose `x` no channel
  observes, declares it in place of the old block. `examples/sensor_detour.py` plans its
  map-blind arm as `"prior"` with the row and `epistemic=False`, which reproduces the old prior
  plan and its scores (−12.796) exactly; the default information gain under `"prior"` reads the
  map and also detours. Under `"posterior"` the single fix's leaf equals the old block leaf to
  float precision. The gradient planner had not converged at its 300 steps: its score ranged
  over 0.18 nats under 1e-6 perturbations of the starting plan (the review found the same
  before V11); at 10,000
  steps its printed score, −3.556, is stable under small perturbations of the starting plan
  (it is still descending in the fourth decimal), and the plan reaches (0, 0.5) in two steps,
  holds there for four and turns for the dock.
- **Emissions (decision 7).** `selectors._eligible_state_blocks` makes every Gaussian emission
  over categorical parents eligible; its leaf is the `(state, emission)` registry estimator
  (`continuous.emission_info_gain` over `emission_entropies`), its mask the channel's parent
  mask. Under `risk_ambiguity` `make_step_terms` mounts `−H[q(y)]` as its risk always and `E H[y
  | S]` as its ambiguity where selected. `epistemic=True` on a model with a matrix-normal
  inverse-Wishart emission still raises on that emission's parameter novelty (no estimator, as
  before); `("z",)` or `obs:<m>` selects the state leaf alone. `examples/growing_mixture.py`,
  `examples/gameworld_growth.py` and demo Section 11 declare `epistemic=False` and do not plan
  through `infer_policies`, so their outputs are unchanged. A neural likelihood carries neither
  risk nor ambiguity, as before.
- **Branching planners (decision 6).** `register_planner(..., branching=True)` marks the
  sophisticated planner (`planning.planner_branches`); `System._plan` builds its context with
  `branching=True`, as `System.planner_context` does for a System whose `planner` is
  branching, and `score_policies` (open loop) never does. *(On the review:)* the swap is scoped
  to a plan that actually branches: `System._planner_context` keeps the flag only when the
  System declares `branch_on` modalities, so a branch closure exists. Without them the
  sophisticated planner is closed-loop enumeration over beliefs no outcome updates, a state
  target's KL charges no branch, and it scores as enumeration does (equal to it at
  `policy_len` 1, as its docstring states). `branch_on` itself is accepted with any planner
  marked branching, a registered one included, in a System and in a description. The flag
  reaches both
  `make_step_terms` (tabular and distribution-object), which then score each state target by
  `discrete.cross_entropy_risk`, keyed `risk[<factor>]` as before: the sophisticated planner
  returns no term tree, so the key is read only by a caller of `step_terms_fn`. Terminal targets
  keep the KL (superseded by decision 8, as noted at the end of this item). A planner passed as an undecorated callable is open-loop to this flag. The
  terminal KL is read on the deepest children before their branch, so in the two-room case a
  plan that reads the sign at its last step escapes the terminal charge that reading first
  pays: one charge against one credit, but not for every reading time (as built under decision
  6). *(2026-10-19, decision
  8:)* the flag also reaches both carriers' `make_terminal_terms`, which then score each
  terminal target by `discrete.terminal_cross_entropy_risk`, keyed `terminal_risk[<factor>]`
  as before; the timing asymmetry is gone.
- **Limitations kept from HEAD (found on the review).** A mixture-carried factor read by several
  linear-Gaussian channels sums their leaves per channel with no `observation_interaction` or
  `observation_correlation` (the corrections are built for a Gaussian-carried factor alone), so
  information the channels share is counted twice there, as it was before V11. A Gaussian
  emission's information gain (decision 7) reads the predictive categorical belief over its
  parents under `planning_belief="posterior"`, as a tabular channel's does
  (planning-with-observations.md D2's convention), not a conditioned one.

### V11 — decision 8 (opened 2026-10-18, decided 2026-10-19): a state target inside a branching search

*Decided 2026-10-19 (Alex): reading 3, recorded as decision 8 above. The question as it was
put follows, then what settled it.* Decision 6
scores per-step state targets by cross-entropy inside branching planners and keeps the terminal
target's KL. The terminal KL is read on the deepest children before their branch, so a plan that
reads at its last step escapes the terminal charge that a plan reading earlier pays (the planner
prefers to postpone reading, by exactly the last step's information gain about the targeted
factor). That exposes the general question.

**The example.** A room `s ∈ {A, B}`, `q(s) = (½, ½)`, constant; a flat preference `P̃(s) = (½,
½)` at every step; at step 1, READ observes a sign `o = s` exactly, WAIT observes nothing; nothing
after depends on the answer. Open loop (risk on the predictive belief, D5): READ scores `−log 2`
(its information gain, credited once), WAIT `0`. In the tree with a per-branch KL, every later
step's risk is raised by `I(s; o)`: `E_o KL[q(s | o) ‖ P̃] = KL[q(s) ‖ P̃] + I(s; o)`, so READ
scores `−log 2 + (k − 1) log 2`. With an outcome preference instead (a light showing the room
every step) the outcome form's per-branch decomposition `risk + ambiguity = −(utility +
infogain)` ties the entropy credit to that step's observation, and both plans end up knowing, so
they tie: the outcome form never meets this.

**Where it comes from.** `KL[Q(s) ‖ P̃(s)] = −E_Q ln P̃(s) − H[Q(s)]`. Alex's framing: a state
target is KL control on latent states; `P̃(s)`, like `P̃(o)`, is ad hoc in practice (the reading
of preferences as the model's own marginals is a separate roadmap item), so no relation such as
`P̃(o) = Σ P(o | s) P̃(s)` need hold, and the preference itself enters as its cross-entropy. The
contested piece is `−H[Q(s)]`, which in expected free energy's state form (Da Costa et al. 2020)
comes from the objective's `ln Q(s | π)` term: in a branching search, whose entropy is it? Three
readings, each an option:

1. *The branch's belief* (the per-branch KL, today's behaviour before decision 6): knowledge is
   charged as narrowness; a plan that never learns keeps collecting the entropy as if it read out
   its state every step. Unearned in the tree.
2. *The breadth of where the agent ends up* (Alex's de-risking reading): with actions fixed the
   branch beliefs average back to the predictive, so breadth would be scored on the mixture across
   branches. Principled, but a risk on a mixture is not an expectation over branches, so it breaks
   the tree's bottom-up fold.
3. *None inside the tree* (cross-entropy, decision 6 extended to the terminal target): every
   quantity the tree evaluates per branch is a preference's energy or a genuine information term;
   the de-risking entropy is given up in closed loop.

The literature does not meet the case: sophisticated inference (Friston et al. 2021) scores
outcome risk at every step and reaches states only through priors over final states. A
cheaper variant of the terminal choice keeps its KL but charges it after the last branch, which
removes the timing asymmetry and makes information about a terminally targeted factor worth
nothing in closed loop.

**What settled it (2026-10-19).** Readings 2 and 3 agree on the example: with the actions fixed,
the branches average back to the predictive belief, `(½, ½)` for both plans, so the spread term
does not tell them apart, and under both the sign's observation terms credit reading with `log 2`
once. They differ wherever plans differ in how broadly their end states spread, whether through
the dynamics or through acting on an answer: reading 2 favours the plan that spreads its end
states to match a broad preference, and reading 3 is indifferent. *(Corrected on the review: the
first account of this, given in discussion, named only acting on an answer.)* Reading 2 can be
approximated by a fixed-point iteration (decision 8's route back), but that iteration is not
guaranteed to improve, and it multiplies planning cost. The question touches only branching
planners. So reading 3 was adopted, with the iteration recorded.

### Build note (V11, 2026-10-21): one factorisation per Gaussian-carried factor per step

The per-channel `I(y_m; x)` leaves and the `observation_interaction` (or, under
`risk_ambiguity`, `observation_correlation`) leaf of a Gaussian-carried factor are evaluated from
one factorisation of the stacked predictive covariance `S = H Σ Hᵀ + blockdiag(R_m)` per factor
per step (`continuous.linear_gaussian_readings`, called from `hybrid._make_step_terms`; the
registry's linear-Gaussian state row reads the per-channel gains from `StepContext.readings` and
falls back to `state_info_gain` for a channel the fused route does not cover). Each gain is
`½ log|S_mm| − ½ log|R_m|` from the Cholesky factor of the diagonal block; the interaction is
`Σ_m I_m − I(Y; x)` with `I(Y; x) = ½ log|S| − ½ log|blockdiag(R)|` from one Cholesky factor of `S`,
which for jointly Gaussian readings equals `TC(Y) − TC(Y | x)`. The leaf names, term keys, mounted
leaves and the exactness of every Interval are unchanged. The general route (`predictive_entropy`,
`observation_interaction`, `observation_correlation`) still serves a mixture-carried factor, a
switching channel and a partial channel selection (a sub-block selection takes the fused
route below since the 2026-10-21 addition); a channel that selects
its head over an action node is built per step and takes the fused route. A Gaussian belief's
predictive entropy no longer builds the one-component mixture, the total correlation takes
Cholesky log-determinants, and a matrix of side at most 4 is factored by unrolled elementwise
operations (a batched `jnp.linalg.cholesky` is several times slower at these sizes).

A report of plan evaluation 2.4 times slower in 0.15.0 than in 0.14.0 on an unchanged hybrid model
was bisected to the commit that added these leaves (45f446a, decision 3 above): the objective is
the intended one, its evaluation re-factorised what the channel leaves had factored and built a
mixture and an Interval for each Gaussian predictive, inside the policy `vmap` and the step `scan`.
Measured on the bisect's model (2 Gaussian-carried factors, 4 channels with two on one factor,
one categorical factor, `epistemic=True`, `policy_len=4`; median of 40 calls, CPU):

| plan evaluation, `infer_policies` | 0.15.0 | fused |
|---|---|---|
| 1296 policies, prior chain | 0.54 ms, 451k flops, 3440 HLO lines | 0.22 ms, 322k flops, 2724 lines |
| 1296 policies, posterior chain | 0.68 ms, 536k flops, 4781 lines | 0.33 ms, 409k flops, 4142 lines |
| 81 policies, prior chain | 0.13 ms, 36.5k flops, 3473 lines | 0.10 ms, 26.3k flops, 2763 lines |
| 81 policies, posterior chain | 0.16 ms, 48.1k flops, 4796 lines | 0.12 ms, 38.2k flops, 4163 lines |

The leaves agree with the general route to a relative 1e-6 (channel gains) and an absolute 2.4e-5
(the interaction) in float32; `tests/test_plan_evaluation_cost.py` pins both and an upper bound on
the lowered flops of the 1296-policy call.

#### Addition (2026-10-21): the fused route over a sub-block selection

A factor selected in part (V10) takes the fused route too. `continuous.sub_block_readings(q,
models, dims, present)` factors two stacked covariances per factor per step: the predictive `Σ_Y
= H Σ Hᵀ + blockdiag(R_m)` and the conditional `Σ_{Y|S} = H Σ_given Hᵀ + blockdiag(R_m)`, with
`Σ_given = Σ − Σ_{:,S} Σ_SS⁻¹ Σ_{S,:}` computed once. From their diagonal blocks, `h_m = ½
log|Σ_Y,mm|` and `c_m = ½ log|Σ_{Y|S},mm|`: the channel leaf `I(y_m; x_S) = h_m − c_m`, the
ambiguity `E H[y_m | x_S] = ½ d_m (1 + log 2π) + c_m`, and the correlations `TC(Y)` and `TC(Y |
x_S)` from the two factorisations, so that the interaction is `TC(Y) − TC(Y | x_S) = Σ_m
I(y_m; x_S) − I(Y; x_S)`, which is negative where readings are informative about `x_S` only
together. `hybrid._make_step_terms` admits such a factor into its reading groups whenever no
channel observing it has a partial selection. Under `utility_infogain` the gains and the
interaction are the group's; under `risk_ambiguity` the group supplies `TC(Y)`, the channels'
ambiguity leaves and, where the mounted channels are all the channels observing the factor, the
`−TC(Y | x_S)` leaf (a mounted subset takes `conditional_correlation`). A channel selecting a
mixture-carried factor, a switching channel and a partial channel selection keep the general
route. Term keys, mounted leaves and the exactness of every Interval are unchanged;
`tests/test_fused_sub_block.py` compares the leaves with the general route on a four-dimensional
factor with a three-dimensional selection read by a two-dimension block, by a channel over all
four dimensions and by a channel on another factor, noise maps on two channels, per-step
presence, both decompositions and both chains: the worst deviations are 2.4e-7 absolute (gains,
relative 2.6e-7) and 2.4e-7 absolute on the interaction; under `risk_ambiguity` the `−TC(Y | x_S)`
leaf is bitwise, `TC(Y)` is within 6e-7 and the ambiguity within one ulp. The whole-factor
interaction leaf is now `Σ_m h_m − ½ log|S|` (the log-determinants of the blocks and of `S`, with
the noise terms cancelled before the sum), within 1e-7 of 0.16.0's value. Under per-step presence
the blocks are factored on their own and only the masked stack is factored for the joint, so a
step costs one extra factorisation per covariance, not two.

Measured on the benchmark model above with `pos` of four dimensions, `(x, y, vx)` selected, noise
maps on its two channels (median of 40 calls after warm-up in a fresh process, CPU; the scores
agree to the printed digits):

| `infer_policies` | general route | fused sub-block route |
|---|---|---|
| 1296 policies, prior chain | 1.04 ms, 2.47M flops, 6929 lines | 0.92 ms, 1.85M flops, 5086 lines |
| 1296 policies, posterior chain | 1.37 ms, 2.91M flops, 9774 lines | 1.19 ms, 2.31M flops, 7993 lines |
| 81 policies, prior chain | 0.218 ms, 202k flops, 7710 lines | 0.198 ms, 152k flops, 5839 lines |
| 81 policies, posterior chain | 0.307 ms, 264k flops, 10611 lines | 0.289 ms, 218k flops, 8802 lines |

The saving there is 10 to 13 percent of time and a fifth to a quarter of the flops: the noise
maps' per-policy interpolation and the control-dependent mean dominate. On a downstream
hybrid model (a pose factor of four dimensions with a three-dimension selection, two channels with
`GridNoiseMap` noise, posterior chain, gradient planner of 100 Adam steps) the median `act()`
over 150 steps fell from 72.4–73.2 ms (0.16.0) to 50.1–50.6 ms.

#### Addition (2026-10-21): the posterior chain with noise maps, three exact savings

On the same downstream model with a fourth channel (`pose_fix`) declared `present=False` for the
plan, `act()` was 52.8 ms against 25.9 ms for 0.14.0. With the noise fixed the 100 Adam steps
cost under a millisecond, so the cost was the backward pass through the noise maps' dependence on
the mean, and three changes remove it without touching the objective:

1. *A statically absent channel is dropped.* `present` entries that are Python `False` or a
   concrete all-`False` array, with no per-level entry, are recorded on the `Presence`
   (`absent`, read by `presence.static_absent`); a traced availability stays dynamic. The channel
   leaves the fused reading groups (`hybrid._make_step_terms`; its leaves are zeros, as
   `mask_absent` made them) and the epistemic chain's conditioners (the `System` planner context),
   so the stacked readings are the live channels' (3×3 where they were 7×7, under the unrolled
   Cholesky limit instead of LAPACK, and no masked copies of the stack).
2. *One read of each map per step.* `_step_pair` reads each conditioner's noise factor once at the
   epistemic mean (conditioning leaves the mean unchanged) and the pair carries it
   (`BeliefPair.noises`, seeded by `pair_beliefs`); `condition_expected` takes it
   (`inference.linear_gaussian.condition_on_expected` returns the covariance update with the mean
   `g.loc` itself, no `K (y − Hμ − c)` recompute), and the information-gain leaves under
   `utility_infogain` pass it to `linear_gaussian_readings` / `sub_block_readings` (`noises=`).
   Under `risk_ambiguity` the leaves read the predictive member's mean, which a nonlinear
   (unscented) chain can move off the epistemic one, so they read the maps themselves; so do the
   leaves of the prior chain.
3. *`Σ_given` from the trailing block.* For the selected dimensions leading, `Σ_given =
   blockdiag(0, L_TT L_TTᵀ)` with `L_TT` the trailing block of the belief's factor (the Schur
   complement), so `Σ_{Y|S} = (H_T L_TT)(H_T L_TT)ᵀ + R`, no factorisation. A non-leading
   selection is permuted to the front and the factor is the Cholesky factor of the permuted
   covariance: one unrolled factorisation of the state's own side (a state of side four or less),
   in place of the Cholesky of `Σ_SS` and a solve; the route is exact, and its cost was not
   measured separately from the leading case. `observation_interaction` and
   `conditional_correlation` keep `_given`.

Measured on the standalone rebuild of the model (`gradient_plan`, 100 Adam steps, `policy_len` 6,
median of 30 in a fresh process per row, CPU, machine under load so only the ratios carry):

| | `gradient_plan` | HLO lines, grad of `G` | fusions |
|---|---|---|---|
| main before | 77.8 ms | 20192 | 739 |
| (1) alone | 51.5 ms | 17893 | 606 |
| (2) alone | 64.3 ms | 15500 | 604 |
| (3) alone | 69.3 ms | 19968 | 702 |
| all three | 32.7 ms | 13164 | 445 |

On the downstream script (`act()` median over 150 steps, same session): 77.9 ms before, 35.9 ms
after, 42.4 ms for 0.14.0. Equality against the route without each change
(`tests/test_posterior_chain_cost.py`): every leaf is within 2e-6 of the route without the
change, and the gradient of `G` with respect to the controls within 5e-7 ((2) alone leaves the
trees bitwise; (1) replaces a masked seven-by-seven factorisation by the live three-by-three;
(3) is the largest, 1.4e-6, under either decomposition); `G` of the model bitwise,
and the 100-step plan within 4e-5 on controls of size 0.6 (the float differences of two compiled
programs, which also separate the previous route from 0.16.0's own build). The test bound on the
lowered gradient is 1.3 times the HLO line count after.
