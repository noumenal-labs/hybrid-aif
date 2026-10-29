# System design — decisions

*Status: confirmed (Alex, 2026-07-21) and implemented in `system.py`
(commit ef38398), with two known code-vs-design gaps tracked for the next
revision: planner injection (currently hardwired to the enumeration
planner) and the trajectory-logging convention (no logger built yet).
Resolves the "To design" items queued in `conditional-families.md`'s
appendix.*

## The type landscape

**`System` is a real type — the unit of coupling and of model
specification.** It carries: static structure (node/factor declarations,
modality & factor metadata, action-space structure, wiring), a params
pytree (generative tensors, preference parameters, learnable-prior
counts), optional *preference designations*, and an inference config.
Methods are pure; belief state lives in the rollout carry, never on the
System (a stateful OO wrapper can be added later as sugar).

**`Env` is a minimal interface, not a System subclass**: anything steppable
(`reset`/`step`, pure, key-threaded). Systems *produce* one via
`as_env()` (delegating to the existing `GenerativeEnv` machinery — forward
ancestral sampling over the same tensors). Non-System things implement
`Env` directly: external simulators, and the transducer-wrapped real world.

**`Agent` is not a type.** An agent is a System being *used* through its
inference faces with a planner supplied — agency is an activity, not a
class. Rationale (Alex's coupling argument): part of any agent's
environment is other agents, and coupling must be blind to whether the
other side plans; planner-presence therefore cannot be a type distinction.

## Preferences

No `Preference` wrapper type. A **preference designation is a mapping from
a mean-field block id to a target `Distribution`, plus a temporal scope** —
and since every variable (observation modality, state factor, parameter,
explicit precision variable) lives in some block, *any* System variable can
carry a homeostatic set-point. The objective layer's risk term for a
designated block is KL(q(block) ‖ target): for observation blocks this
recovers the classic C term; for state blocks, the KL-control /
risk-over-states form; for parameter/precision blocks (e.g. a γ set-point),
it is the README's "prior/KL terms on any variable including parameters"
principle in homeostatic use. Targets may be `Conditional` —
per-timestep/contextual C is a conditional with a time/context parent; same
for D or anything else.

Temporal scopes: `standing` (every rollout step — classic C),
`terminal` (scored against the final state/observation — H, the
plan-backward-to-a-goal constraint; hence `final_obs` in the logging
convention), and later `scheduled` (via conditional targets). Same
mechanism, three scopes; H is not a separate feature.

Precision is a property of any distribution; explicit higher-level
representation or estimation of a precision (γ-style affective inference)
is wiring, not a special mechanism.

## Factors (the node-layer local-function concept)

The distributions layer stays strictly PDFs. The node layer's local
function is the less-committal concept — a **`Factor`, a tagged union**:

- `Stochastic(Distribution | Conditional)` — probabilistic factors;
- `Deterministic(map, approximation_rule)` — deterministic nodes with
  per-node message treatment (quadrature/sigma-point forward, RTS-type
  backward — Petersen et al. 2019; linear maps exact). No delta
  `Distribution` exists: a delta has no honest `log_prob`/`entropy`, and
  free-energy bookkeeping for deterministic nodes needs the dedicated rule
  (Şenöz et al., Thm 8), not a degenerate density;
- data clamps — observations as constraints (CFFG data-constraint
  reading), applied by the engine rather than represented as densities.

This gives custom maps (registry-reference functions) their typed home.

## Planning

Planners are **injected, never owned**: `infer_policies(..., planner=...)`.
A System's inference config may carry a *default planner spec* — name +
config, resolved through the registry (the same
custom-function-by-reference pattern) — so ownership in the data structure
is only ever a name and Systems remain serializable pytrees. Policy
inference is thereby as swappable as state inference (Alex's engine-parity
argument).

**Policy priors (E / habits)** are a designation alongside C, with a
per-planner representation: a `Categorical` over the policy set for
enumerated planners; the initial proposal distribution for CEM/MPPI (which
*is* those planners' policy prior, usually unnamed); a learned habit
network for the future amortized backend — making planner-to-habit
distillation literally "learning E." Composition:
`q_π ∝ softmax(−γ·G + log E)`.

## Faces / API

pymdp-familiar named faces, deliberately not a generic query API:

- `as_env(system) -> GenerativeEnv` — the forward face.
- `infer_states(system, params, evidence) -> beliefs` — state inference
  via the configured engine (`fpi` today).
- `infer_policies(system, params, beliefs, planner) -> (q_pi, metrics)` —
  policy inference; `metrics` is the EFE term tree (workstream D).
  Epistemic-term toggles are block-grain, matching the metrics granularity:
  `include_state_info_gain` / `include_param_info_gain` accept `True` /
  `False` / a tuple of block names (these terms factorize per *modality*
  block; exact per-state-factor attribution doesn't exist when a modality
  has multiple factor parents — a factor-scoped MI term can be added as a
  distinct registered term if wanted).
- `act(key, system, params, beliefs, planner) -> (action, control_state)`
  — wraps `infer_policies` and samples/selects the control state.

*[Amendment 2026-09-18 — signatures as shipped.]* The bullets above record
the faces as *proposed*. As built they are methods on `System`, and three of
the four return more than proposed: `infer_states -> (beliefs, detail)`,
`infer_policies -> (q_pi, term_tree)`, `act -> (action, q_pi, metrics)`,
`update_empirical_prior -> priors`; `as_env(params)` returns an environment
*and* its params. The `include_state_info_gain` / `include_param_info_gain`
toggles were superseded by the `epistemic=` selector (per-variable-infogain.md),
and a state variable named there mounts a leaf keyed by each modality that
observes it. The reference for the shipped shapes is `docs/composing-systems.md`;
this section stays as the record of the proposal.

**Every face is pure and jit-safe** (friction log #14, measured by the
downstream integration: 1–3.5 s/event eager at K=3–6 vs milliseconds under
`eqx.filter_jit`): in a control loop, wrap the faces you call per tick in
`eqx.filter_jit` once per model shape, keep per-call variation
leaf-valued, and pay the compile at a stationary moment (mission-start
prewarm). The one footgun: anything that changes a leaf's *shape* (e.g. a
mixture target whose component count shrinks) forces a retrace — pad with
dead (−inf-log-weight) components instead, which the padding-invariance
test battery blesses.

A fully generic `infer(query, evidence)` over arbitrary variable subsets
was considered and rejected as scope creep; the signatures above are shaped
so that, if it is ever wanted, it can be added as a dispatcher over the
same machinery with the named faces as sugar. Nothing forecloses it.

**Population vs composition (2026-09-01, P1′ chunk 2d).** Two ways to have
more than one mind, orthogonal and composable, and neither is the other.
A **population** is `jax.vmap` over the faces: independent replicas of
**one static System** (it is all-`static` fields, so it holds no numeric
leaf and is *closed over*, never mapped), with every numeric carrier on a
leading batch axis — `SystemParams`, the belief carry, `LearnState`
(optimizer moments and the return normalizer's EMAs), the replay buffer,
the env state, the key. No interaction, no declaration change, no
synchronisation, because there is nothing shared to synchronise. A
**composition** is `merge()`: several minds inside one joint generative
model, interacting through shared latents, and it returns *a new
System* — a different declaration, one belief carry, one G. So "four
training seeds" is a population (`reference/raif/run.py` runs exactly that:
a seed is an agent whose entire numeric state is batched, which is why a
reboot loses one checkpoint interval rather than one seed), while "two
agents in a room" is a composition, and a population **of** coupled
Systems is a `vmap` over `couple()`'s output needing no further machinery.
The brief's "vmapped across agents" commitment is now a tested thing rather
than a claim: `tests/test_population.py` runs `infer_states`, conjugate and
gradient `learn` (per-member traced rates included), `act` under the habit,
enumeration and CEM planners, `update_empirical_prior`, the whole canonical
loop, and `StaticBuffer` `add`/`sample` batched against sequential
members — leaf for leaf, `allclose` on floats and **exact** on discrete
leaves (an action's level index, a ring cursor, an `is_last` flag), which is
the honest split: batching preserves arithmetic to float32 and control flow
bitwise. Every face passed unmodified; the one thing 2d had to work around
was that `init_learn_state` leaves `LearnState.moments` empty while
`imagine` adds its retnorm key on first call, so the whole-agent carry is
not `lax.scan`-stable out of the box (`reference/run_mc.py::scan_stable`,
which carries the finding and the two candidate engine fixes).

*[LANDED 2026-09-14 — `System.init_learn_state` now seeds `moments` declaratively through `objectives.amortized.scoring_moments`, `imagine` raises on a missing entry, and `run_mc.scan_stable` is deleted (ledger entry 14 paid; divergence 29).]*

## Trajectory logging convention (binding for all loggers)

- One row per tick `t`: `obs_t`, `action_t`, and optional `belief_*`
  columns; inference and action are instantaneous within a tick
  (`action_t` follows `obs_t`).
- The terminal observation is kept as an episode-level `final_obs` column
  (not padded into the step columns) — special because terminal-state
  preferences (H) score it.
- Loggers cast floats to float32 explicitly (numpy's float64 default would
  silently lock dataset schemas).
- Datasets record the alignment convention in file-level attrs.

## Coupling (v0.1 scope)

One System's action nodes pair with another's observation modalities; its
observation modalities pair with the other's outputs (blanket-typed
edges). v0.1 builds exactly one coupling shape: agent-System × env
(`Env` interface) through the existing `rollout` driver, whose
`policy_fn(key, history, t)` seam is where `act` plugs in. Hierarchical
nesting (System-of-Systems, transducer slots at the outermost boundary)
is designed here in vocabulary but deliberately unbuilt until after v0.1.

## Addendum 2026-08-21: kernels, and the three groups of `SystemParams`

*Status: vocabulary and criterion ADOPTED (Alex + Opus, 2026-08-21).
Arrived from the amortization program: chunk 2b needs a home for four
trained parameter sets (actor, critic, IG ensemble, CRSPP) and the
question "why is `recognition` its own field?" had only a negative
answer ("it isn't generative"). The positive criterion below comes from
an embedded-POMDP formalization of Alex's (draft notes; not in the repo 
— the two facts this section uses are restated in full below, so nothing 
here depends on the source).*

### Vocabulary

A **kernel** is one factor of a joint: a conditional distribution
regarded as a map — hand it a point, it returns a *distribution* over the
next thing. `conditional-families.md` already fixes this sense ("a
conditional distribution (CPD / Markov kernel)"). A `Deterministic` node
is the degenerate case, a Dirac kernel, which is why it needs no
special-casing in the `transition` slot.

**Two graphs, and they must never be left implicit.** The **declared
model graph** is what `factors.py` populates and the user writes down.
The **closed loop** is world + agent, what `driver.run()` realizes. The
load-bearing point: the declared model is *not a subgraph* of the closed
loop — it is interior to one of its nodes. In the embedded joint the
agent contributes exactly two factors:

    Agent:  Q(s_{t+1} | o_{t+1}, s_t, a_t, θ_L, θ_T)  ·  P(a_t | s_t, θ_π)

belief update and action selection. The internal generative model
`P(o|s,θ_L)P(s|s₋,θ_T)P(s₀|θ_S)` appears nowhere in that product; it is
*used only for inference*, i.e. it lives inside `Q`.

### The criterion, and the three groups

*(Criterion restated 2026-08-25 — the original deletion test ("a field is
a closed-loop factor iff deleting it removes a factor") fails on the
default configurations: an empty `recognition` falls back to the
structured/exact row and `log_E=None` still yields `P(a|s)` through the
γ-softmax, so deletion removes nothing — the loop's two agent-side
factors always exist, and those fields hold one implementation's
parameters, not the factors' existence.)*

A field is a **closed-loop factor's own field** iff its contents,
evaluated at the loop boundary, *return the distribution the loop then
acts on* — the recognition module returns the belief update itself; the
E/habit module returns `P(a|s)` itself. Everything else is material
consumed *interior* to computing one of those two distributions, and
leaves the factorization's shape untouched however its numbers change.
That test partitions every field of `SystemParams`:

1. **The declared model** — `transition`, `likelihood`,
   `initial_state_prior`. Positional (indexed by graph position).
   Interior to `Q`.
2. **The closed-loop factors** — `recognition` (belief update,
   variable-keyed) and the E/habit designation (action selection). These
   two and no others.
3. **Scoring material** — `obs_targets`, `state_targets`,
   `terminal_targets` (variable-keyed) and the row-keyed field the
   amortization program adds. No kernel of its own; interior to
   `P(a_t|s_t,θ_π)`. The formalism's `P_θ(a_t)` shorthand *is* this whole
   group, compressed into one symbol.

So groups 1 and 3 are each interior to one of the two group-2 factors,
which is why they are separate fields from the factors that consume them.
Note that group membership is a **role, not a type**: the IG ensemble and
CRSPP are *built out of* kernels but consumed to produce a real number, so
they are group 3. What a row returns — a `Distribution` or an `Interval` —
is the mechanical form of the test.

### What this licenses

- **`recognition` keeps its own field, positively justified.** It is a
  closed-loop factor; the group-3 field is not. A11's peer-slot decision
  was right, but its stated reason ("an inference choice is not a world
  claim") only said what recognition *isn't*.
- **The actor lives in the E designation** — already the design of record
  (this doc's "Planning" section, 2026-07-21; `amortization.md` A3), not a
  new proposal. It is a kernel, so it was never group-3 material.
- **The group-3 field is therefore homogeneous** (critic, IG ensemble,
  CRSPP) and can be named for what its contents do: scoring.
- **Two enforcement points:** a test asserting every `SystemParams` field
  is in exactly one group, and a `merge()` error for an attempt to
  compose a group-3 entry (it has no kernel, so it cannot be a subsystem).

### What this does NOT claim

This is a statement about which factor consumes what, and about which
graph a declaration belongs to. It is **not** an ontological split between
"model" and "machinery," and it does not assert that group-2 or group-3
parameters lie outside any possible joint density.

Nor does it assert the groups' *values* are independent. The groups
partition **fields by role**, never values by provenance — and one
rigorous reading of AIF (self-evidencing: preferences ARE the generative
model's marginal likelihood over outcomes, C = p̃(o)) requires exactly
the coupling a naive independence reading would forbid. That reading
stays expressible, deliberately: a target designation accepts *any*
Distribution, including one computed from the group-1 declarations
(`p(o) = Σ_s p(o|s)p(s)` is a contraction over the declared A and D),
and because params are immutable and the caller owns the loop, the tie
"C tracks the model's own marginal as it learns" is a two-line caller
idiom — recompute the marginal after `learn`, `dataclasses.replace` the
target. **No validation may ever reject a target for being a function
of the declared model** — that would foreclose the rigorous reading to
enforce an independence nobody claimed. Declarative *tied* targets
(the coupling as a declaration the library maintains, with
self-consistency questions that come with it) are a distant roadmap
item gated on research, recorded here so the seam stays visible. The inclusive reading —
a joint that contains the dynamics of Q — is left open, and is what this
project's "P may be identically the physics" commitment favors
(Alex, 2026-08-21, rejecting an earlier draft of this section that had
asserted the split as doctrine).

The neutrality is the point, and it is why the criterion is worth
adopting: the *factorization* above is identical whether one reads `Q` as
literally the physical dynamics (the psychophysical-identity route) or as
an idealized approximand with only `Q` realized in finite time (FEP as
usually stated). The two readings differ in interpretation, not in which
factors exist and what consumes what — so a library organized on the
factorization serves both with the same fields, while one organized on
"aspects of P" would have to take a side.

### The open items above, since resolved (2026-08-26)

All three were settled in the chunk-2b adjudications (`amortization.md`
A12): the E designation's field is renamed `log_E` → **`policy_prior`**
with a read-only letter alias `E` completing the `A`/`B`/`D` property
convention and NO `log_E` compatibility shim (no external users;
the downstream repository's one call site migrates in the same pass); the group-3 field
is named **`scoring`**; and the groups stay **flat** — the grouping is
this document plus the membership test, not nesting (a nested layout
costs a path level on every access and every wildcard key and enforces
nothing extra).

## Addendum 2026-08-28: package tiers

The package has **two tiers**, and one import rule between them.

**The engine tier** is everything under `src/hybrid_aif/` outside
`reference/`: `System`, the factors, the distributions, inference,
planning, objectives, `composition/` (`coupling` + `driver`, the
compositional-AIF instruments, gathered 2026-08-28) — and the environment
*abstraction*, `envs/base.py` (`Env`, `Obs`, `State`) and
`envs/generative.py` (`GenerativeEnv`, which is `as_env`'s return type).
Its shape is ours to choose and its API is what consumers depend on.

**The reference tier** is `src/hybrid_aif/reference/`: `worlds.py` (the
declared reference world models — R-AIF's `tiny` RSSM, `crspp_rssm`) and
`reference/envs/` (`mountain_car.py`, `tmaze.py`). Three clauses define
it: content pinned to an external target (a paper's architecture, a
benchmark env, an oracle's numbers), free to churn without an engine API
break, and never imported by the engine.

**The cut is abstraction vs. instances.** `rollout` is typed by `Env` and
`as_env` returns a `GenerativeEnv`, so both are load-bearing engine
types; a concrete T-maze or Mountain Car is a *target* we reproduce, and
targets are reference. The same cut is why `inference/rssm.py`
(`RSSMCore`, engine machinery) stays put while `worlds.tiny_rssm` (that
machinery *placed* to match a paper) moves.

**The one-way rule** — reference imports the engine freely; the engine
imports reference never — is enforced by `tests/test_layering.py`, which
walks every engine module, parses its imports with `ast` (not by string
search: `system.py`'s fence messages name
`hybrid_aif.reference.envs.mountain_car` on purpose) and names the
offending module and import when it fires.

Chunk 2c's content — the oracle fixtures and seeders, the eval harness,
the assembled reference agents — lands in `reference/` from the start.

*Amendment 2026-09-21 (tier-split refactor, `../tier-split-audit.md` T8):*
the tier gained a subtree per external target. Shared targets stay at the
top (`envs/`: the pymdp T-maze, the Mountain Car benchmark); everything
pinned to R-AIF alone lives in `reference/raif/` — `world.py` (was
`worlds.py`, plus `tiny_core`, which was `RSSMCore.tiny` in the engine),
`agent.py` (was `raif_agent.py`), `crspp.py`, `revise.py`, `metrics.py`
(was `eval.py`), `seed.py`, `run.py` (was `run_mc.py`; the CLI is
`python -m hybrid_aif.reference.raif.run`), `fixtures/`. Seed statistics
moved up to the engine (`train.buffers.seed_stats`), being generic over
`Episode`. The one-way rule and its test are unchanged; the prefix test
covers any subtree. Engine error strings no longer name reference modules
(T7) — so the sentence above about `system.py`'s fences naming
`hybrid_aif.reference.envs.mountain_car` on purpose is historical; the
layering test parses by `ast` regardless. Checkpoints carry no module paths,
so saved runs resume unchanged under the new launch command.

## Addendum 2026-08-28: the `system` package

`system.py` had reached 3,280 lines, 2,846 of them the `System` class. It is
now a package with the same import path. **`from hybrid_aif.system import
System, SystemParams` resolves exactly as before** — that invariant is the
point of the change, and `system/__init__.py` exists to hold it: no call site
in `src/`, `tests/` or `examples/` was edited, including the ones that reach
for private names (`coupling.py`'s `_KEY_KINDS` / `_is_pattern`, the tests'
`_carry_prev`, `_param_blocks`, `_slot_learnable`).

**Three files, plus one that went a layer out.**

- `system/core.py` — `System` itself, `SystemParams`, `LearnState`,
  `Subsystem`, `Belief`, `_carry_prev`, and the faces. ~1,960 lines.
- `system/selectors.py` — the key grammar and its resolution: the declaration
  predicates (`_slot_learnable`, `_param_blocks`, `_gradient_learnable`), the
  pattern helpers, the namespaces a key ranges over (`_effective_names`,
  `_slot_declarations`, `_epistemic_variables`), and the resolvers
  (`_resolve_rates`, `_term_targets`, `_resolve_param_novelty`, `_row_mounts`).
  ~800 lines.
- `system/validation.py` — `_validate`, `_validate_H_deps` and the `_check_*`
  family. ~520 lines.
- `hybrid_aif/learning/conjugate_stats.py` — `learn`'s per-slot statistics
  (`_lg_noise_stats`, `_lg_joint_stats`, `_mlp_evidence`) and `_lower_learnable`.
  ~335 lines. It went to the existing `learning/` package rather than into
  `system/` because it is workstream-B arithmetic that `learn` *uses*, not
  structure the System *owns*.

**Why this cut and not another.** The class's four largest concerns were the
faces (~980 lines), selector/key resolution (~570), validation (~450) and the
conjugate statistics (~260). Only the first is `System`'s own subject matter —
the other three are things done *to* a declared System, each with a different
reason to change: the grammar changes when a new selector is designed,
validation when a new declaration family lands, the statistics when a new
conjugate family does. Splitting the faces would have cut across the pytree.

The selector file earns its own boundary twice over: it is where the V1–V8
variable grammar and A12's `terms=` row grammar sit side by side, and so it is
where `amortization-debt-ledger.md` entry 1's resolver merge — `epistemic=` as
literal sugar over `terms=` — becomes a change between two functions in one
file instead of one across a 3,000-line class. Isolating it is the first step
of paying that debt, and the ledger entry now names it as the landing site.

**The mechanism is delegation, not inheritance or mixins.** Each moved helper
is a module-level function taking the `System` explicitly — `_resolve_rates(
system, mapping, what, params)` — and `System` keeps a thin method forwarding
to it. So `self._resolve_rates(...)` inside a face is unchanged, and every test
that calls a private method still works. `System` stays a plain `eqx.Module`
pytree with static fields and no derived state. Where a helper was a
`@staticmethod` and reads no structure (`_check_slot_key`, `_factor_levels`,
`_belief_value`) it stays static, taking only what it checks; `_lower_learnable`
turned out to read nothing off the System at all and now takes only params.

**Type references, not import cycles.** The sibling modules need `System` and
`SystemParams` for annotations only, so they use `from __future__ import
annotations` with a `TYPE_CHECKING` import — nothing imports `core` back.
`validation` and `conjugate_stats` do import `selectors` at runtime (for the
declaration predicates), one-way and acyclic.

Behaviour is unchanged by construction and by check: every moved definition was
compared to its original at the AST level after normalizing away the permitted
rewrites (`self` renamed, the `system: System` annotation added, redundant
annotation quotes dropped), and all 62 matched. The suite passes 942/942
unchanged, as does the notebook gate.

One consequence for `tests/test_layering.py`: its `rglob("*.py")` walk picks up
the new modules automatically, and the reference-tier fence messages it mentions
now live in `system/core.py` (Mountain Car) and `system/selectors.py` (CRSPP).
Its docstring still says `system.py`; the test itself is unaffected.

## Addendum 2026-09-29: prefix-sharing evaluation of the policy tree

The enumeration planner scored every policy of the `b^H` grid by its own scan
from the root, `H·b^H` transitions and step-term calls, though the grid's
policies share prefixes by construction (performance review, item 11). The
sophisticated planner (PR #6) already walks a tree of action prefixes node by
node. Both are folds over one object — a tree with a belief and a step term
tree at every node — and the fold is what differs.

- **E1. One expansion, two folds.** `planning/tree.py` owns the level-wise
  expansion: from `n_t` beliefs at depth `t`, every first-step action is
  applied to every node (`n_t·b` children, one `vmap` over nodes and one over
  actions) and scored by `step_terms(qs_next, action, t)`. The enumeration
  planner folds by *path sums* — a policy's leaves are the sum of its
  ancestors' node leaves, the horizon-end term added at the leaf; the
  sophisticated planner folds by *softmax expectation at each level* after
  branching each child over its outcomes. `evaluate_policies` stays as the
  evaluator of an *arbitrary* policy set (CEM's samples, `score_policies`'
  plans), which has no prefixes to share.
- **E2. The policy order is the grid's.** Children of node `i` at depth `t`
  are `i·b + a` in the grid's own action order, so a leaf's index is the
  policy's index in `enumerate_policies` (time-major, last step fastest) and a
  policy's ancestor at depth `t` is `p // b^(H−1−t)`. Nothing downstream —
  `policy_posterior`, `legal_logits`, `sample_action`, `action_marginals`,
  the metrics tree's per-policy axis — sees a different set or order.
- **E3. The tree contract is unchanged.** Under `metrics=True` a policy's
  leaf is the sum along its path, so every leaf still carries a leading
  per-policy axis and `total(tree)` is the same `G` up to float
  reassociation (the Planner protocol's allowance). Under `metrics=False` the
  expansion carries one `Interval` per node, seeded with the step tree's static
  warrant exactly as `evaluate_policies_total` seeds its scan.
- **E4. The grid is built once.** `enumerate_policies` is memoised on its
  (static, hashable) arguments, so a trace no longer rebuilds the product.
- **E5. Cost.** Transitions and step-term calls fall from `H·b^H` to
  `b·(b^H − 1)/(b − 1)`; the leaf level still holds `b^H` beliefs, so peak
  memory is unchanged. The levels are unrolled in Python (`H` is static and
  small), each one a vectorised call.
- **E6. Parity before sharing.** The enumeration planner moves first, under a
  test that holds the scan evaluator's tree and `G` to 1e-6 on every existing
  fixture; the sophisticated planner moves onto the same expansion only after
  that, under its own test against its recursive form.

## Addendum 2026-09-30: rationale from the docstring paring

*Recorded nowhere but the text the 2026-09-30 paring pass cut from
`System.infer_policies`'s docstring.*

- **E3's `metrics=False`, measured, and the rule behind it.** `metrics=False`
  asks the planner to skip the tree entirely. The enumeration planner then
  carries one G per node of its policy tree instead of stacking per-step
  leaves, which is about half the wall time on the T-maze benchmark. The
  posterior is the same either way, up to float reassociation (~1e-7).
  Metrics are a reporting toggle with a known cost, never a semantics switch.

## Addendum 2026-10-23: `system/faces/`, and the engine/face rule

*Supersedes the 2026-08-28 addendum's remark that splitting the faces would cut across the
pytree: a face is a function of the System, so it can live apart from the class that declares it.*

`core.py` had grown back to 4,541 lines, about 1,800 of them the engines behind three families of
faces. The engines now live in `system/faces/`, one module per family of faces, and `core.py` keeps the
declarations, construction and validation glue, the names and declared-structure helpers, and every
public method (2,743 lines).

- `faces/infer.py` — `infer_states`, `predict_beliefs`, `update_empirical_prior` and what they
  run (`_infer_states`, `_infer_states_expected`, the lagged-modality fill, `_present_select`).
- `faces/learn.py` — `learn`, `carry`, `reduce`, `grow`, `init_learn_state` and their engines
  (`_learn`, `_gradient_slots`, `_resolve_learn_state`, the evidence and statistics helpers).
- `faces/plan.py` — `planner_context`, `infer_policies`, `score_policies`, `act` and the
  planning closures (`_planner_context`, `_context_without_control_risk`, `_amortized_scoring`,
  `_amortized_context`, `_habit_fn`, `_branch_fn`, `_plan`, `_belief_policies`).

**The rule.** The engine packages (`inference`, `learning`, `planning`, `objectives`) hold
algorithms over arrays and declarations and never import a System; a face module reads a System's
declarations and parameters and assembles calls into them. Dependencies run `system.faces` →
engine packages, and `tests/test_layering.py` asserts the engine side by `ast`. Two engine
modules predate the rule and still import `hybrid_aif.system` (`learning/structure.py` and
`learning/conjugate_stats.py`, the first for annotations only, the second also for the
`_slot_learnable` predicate); the test names them as exceptions, left for the owner to decide.

**Mechanics.** A moved method became a module-level function whose first parameter is the System
(`system`); its body is the method's with `self` renamed and calls to other moved methods
rewritten, nothing else. The public method on `System` keeps its signature and docstring and
returns the face function's result. A private engine that another module or a test reaches by name
on a System (`_infer_states`, `_learn`, `_plan`, `_planner_context`, `_update_empirical_prior`, and
the like) keeps a one-line private delegate on `System`. A helper shared by several faces
(`_control_targets`, `_check_plan`, `_habit_module`, `_with_lagged`, `_slot_declarations`) stays
on `System`. `core.py` imports the face modules at its foot, since they import it.

The change is a move: `learn`, `infer_states`, `infer_policies`, `act`, `score_policies`,
`planner_context`, `update_empirical_prior`, `log_evidence` and `action_marginals` return bitwise
the same arrays on the T-maze, the two payoff models, a switching mixture, a learnable-Dirichlet
model and a tiny RSSM, before and after.
