# Declare a System — the composer's reference

*A reference, not a design record: what you write down to declare a `System`
and run it, in the final shape the library ships. The design docs under
`docs/design/` record why those shapes were chosen; nothing here should need
them. Added 2026-09-18 from `docs/agent-composition-probe.md`: a naive agent
given only `README.md`, `CLAUDE.md`, `capabilities.md` and two design docs got
a composed, learning agent running, but had to drop into `src/` four times for
conventions no document stated. Those four things are below. Everything here
is recovered from `System`'s field declarations in `src/hybrid_aif/system/core.py` and checked against
a running model (`examples/compose_target_factor.py`).*

`from hybrid_aif.system import System, SystemParams` — note that these are
**not** re-exported from `hybrid_aif/__init__`, which carries the
distribution and factor types only.

## The `System` constructor

Every field is static (`eqx.field(static=True)`), so a `System` holds no
numeric leaf: it is hashable, closed over by `jit`, and patched with
`dataclasses.replace`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `B_deps` | `tuple[tuple[int, ...], ...]` | *required* | Per factor, the factor indices its transition conditions on. Length fixes the factor count. Parents must be declared earlier (continuous factors exclude self). |
| `controlled` | `tuple[bool, ...]` | *required* | Per factor, whether a control node (`u_<f>`, with action node `a_<f>`) indexes its transition. |
| `policy_len` | `int` | *required* | Planning horizon (policy length for the enumeration planner). |
| `A_deps` | `tuple[tuple[int, ...], ...]` | `()` | Per modality, the factor indices it observes. Length fixes the modality count. |
| `planner` | `str` | `"enumeration"` | Registry name: `"enumeration"`, `"cem"`, `"gradient"`, `"habit"`, `"sophisticated"`. Resolved at call time; a `planner=` argument to a face overrides it. |
| `branch_on` | `tuple[str, ...]` | `()` | Modalities (by name) whose outcomes the `"sophisticated"` planner branches on: each categorical over categorical factors. Refused on any other planner. The planner's policy set is then the first step's actions, `(A, 1, F)`. Inside it a state target, per-step and terminal, is scored by its cross-entropy, not its KL, so a branch is not charged for what it learns (`design/per-variable-infogain.md` V11 decisions 6 and 8). |
| `planning_belief` | `"prior" \| "posterior"` | `"prior"` | Which belief a plan's information-gain leaves read on a Gaussian-carried factor: the predictive belief, or a second chain conditioned on the observations the plan expects (`design/planning-with-observations.md`). Refused beside `branch_on`. |
| `feedback` | `None \| "linear_quadratic"` | `None` | How a plan's Gaussian targets are scored. `None`: open loop, on the predictive belief. `"linear_quadratic"`: closed loop, as the policy that follows the plan and corrects deviations by the linear-quadratic regulator's feedback on the estimate; each Gaussian state, terminal and linear-Gaussian observation target on a Gaussian-carried factor is scored by its cross-entropy on the closed-loop state distribution, keys unchanged, and a Gaussian target on a continuous control node also charges the corrections. Requires `planning_belief="posterior"`; see "Closed-loop scoring" below. |
| `gamma` | `float` | `16.0` | Policy precision in `q_pi ∝ softmax(−gamma·G + log E)`. |
| `num_iters` | `int` | `16` | Mean-field FPI sweeps in `infer_states`. |
| `decomposition` | `"utility_infogain" \| "risk_ambiguity"` | `"utility_infogain"` | Which EFE decomposition the term tree carries. The `epistemic` selector mounts each one's own epistemic leaf, `infogain` or `ambiguity`, and never touches risk; a channel without a preference has an implicit flat one, its risk `−H[q(o)]` under `risk_ambiguity`. With everything selected the two give the same G at both interval ends; a channel whose leaf is ablated makes them differ by its `H[q(o)]` (per-variable-infogain.md V11). Under `planning_belief="posterior"` they differ by design. |
| `epistemic` | `bool \| tuple[str, ...]` | `True` | The infogain-row selector (see "Reading the term tree"). `True` = every eligible variable, `False`/`()` = none, a tuple names variables; `*` wildcards per segment. |
| `mixture_k` | `int \| None` | `None` | Shared static Runnalls budget. **Required** as soon as any factor carries a `MixtureGaussian`. |
| `mixture_schedule` | `"runnalls" \| "mode_aligned"` | `"runnalls"` | Mixture compression. `mode_aligned` keeps slots mode-owned and requires `mixture_k % M == 0`; it is needed to filter a factor whose *transition* switches. A switching *observation* (mode parents in `A_deps`) works under either. |
| `D_deps` | `tuple[tuple[int, ...], ...] \| None` | `None` | Parents of each factor's *initial* prior. Ancestral: parents earlier, tabular families. `None` = all unconditional. |
| `H_deps` | `tuple[tuple[int, ...], ...] \| None` | `None` | Parents of each factor's *terminal* target. Resolved against the beliefs at the same step, so parents need not be earlier; a factor may not parent its own. |
| `C_deps` | `tuple[tuple[str, tuple[str, ...]], ...]` | `()` | Parents of each conditional *per-step* target, **by name**: `{"lane": ("u_option",), "cue": ("context",)}` at construction (a key is a state factor's `state_targets` entry or a modality's `obs_targets` entry; a parent a discrete control node or a categorical state factor), stored as pairs sorted by key; `dict(system.C_deps)` to read. The target under a key is a `ConditionalCategorical` (child axis first, then one axis per parent in this order) or a `ConditionalGaussian` (`loc` `(*levels, d)`, `chol` `(*levels, d, d)`; control parents only). At each step the target is resolved under its parents, a clamped control node's level selecting its own distribution and a state parent averaging mean-field, then scored as an unconditional one (the KL in a flat planner, the cross-entropy in a branching one, the closed-loop cross-entropy under `feedback`, whose Riccati pass takes the level's precision). The first structural declaration written by name; `D_deps`, `H_deps` and `B_control_deps` are positional and follow (`design/step-durations.md` T5). |
| `B_control_deps` | `tuple[tuple[int, ...], ...] \| None` | `None` | Per factor, the *controlled* factors whose control slices its transition. State parents are read from the previous step; control parents at the step the control is chosen. The table carries one trailing control axis per entry, in the declared order, each sized by that control node's level count (with equal sizes a misordered table cannot be detected); an empty entry keeps the one size-1 axis. `None` = each controlled factor sliced by its own control. A linear-Gaussian reader takes a discrete node as a mode axis and a continuous one through `G u`; a neural reader takes a continuous node only; a nonlinear reader (`NonlinearGaussianTransition`) hands its map each node's entry, a discrete node as its integer level (also a mode axis of its innovations), a continuous one as its value; policies are over the control nodes. |
| `modality_names` | `tuple[str, ...]` | generated | One name per modality, in node order; the term-tree and `obs_targets` keys. `None` at construction fills in `modality_{m}`, so the stored field is never `None`. |
| `factor_names` | `tuple[str, ...]` | generated | One name per factor, in node order; the `epistemic`/`state_targets`/rate keys. `None` at construction fills in `factor_{i}`. No node may be named after a key's reserved first segment (`obs` or a slot word). |
| `dims` | `tuple[tuple[str, tuple[str, ...]], ...]` | `()` | Names for a continuous state factor's dimensions, given as `{factor: (name, ...)}` and stored as pairs in factor order; `dict(system.dims)` to read. `epistemic` selects a dimension as `<factor>:<name>`; an undeclared factor's dimensions are `dim0`, `dim1`, … by position. One unique name per dimension, each a single key segment (no `:`, `.`, `*` or brackets); `dimN` may name position `N` only. |
| `action_names` | `tuple[str, ...]` | from the sugar | One action node per control node, in `controls` order: the actuator channel, a child of its control node. A native declaration (`controls=`) names them, paired with the control nodes by position, both names free; the sugar names the pair `u_<f>` / `a_<f>`. A modality may observe an action node: `A_deps` names action node `k` as `len(B_deps) + k`. |
| `recognition` | `tuple[Recognition, ...] \| None` | `None` | Which variables are inferred by an amortized module and what each conditions on. Parameters live in `SystemParams.recognition`. |
| `terms` | `tuple[str, ...]` | `()` | Which estimator-registry rows G carries beyond the infogain rows. Grammar: `<row>`, `<row>:<slot>:<name>`, `<row>:obs:<modality>`. The row segment is always literal. |
| `scoring` | `tuple[tuple[str, object], ...]` | `()` | Structural half of the scoring slot, keyed by row name. `dict(system.scoring)` to read. |
| `external_inputs` | `tuple[tuple[str, int], ...]` | `()` | `(name, width)` vectors a recognition factor may condition on but this System does not declare; supplied per call as `infer_states(externals={...})`. |
| `subsystems` | `tuple[tuple[str, Subsystem], ...] \| None` | `None` | Provenance recorded by `merge()`. Metadata: no face computes from it; a refusal reads it to name what a node a binding removed became. |

`SystemParams` fields: `transition` and `initial_state_prior` (dicts keyed by
factor name), `likelihood` (keyed by modality name, default `{}`), `recognition`,
`obs_targets`, `state_targets`, `terminal_targets`, `scoring` (name-keyed
dicts), `policy_prior` (E, with a read-only `E` alias), `control` (each continuous control
node's box, a `Box(low, high)` of finite per-dimension bounds with `low < high`, keyed by the
node's name: a node with an entry is continuous, its readers read a float vector of the box's
width, and the CEM and gradient planners search inside it), `actuator` (the kernel
`P(a | u)` of an action node that is not a delta, a `(levels, levels)` probability table keyed by
the action node's name, column `u` the realised action's distribution: the environment draws the
realised action from it, the control node stays clamped to the decision, and the planning
faces, whose rollouts read the control node, return the same values with or without it). The entries are the
declaration objects (tables, LG heads, nets): `params.likelihood["cue"]`. A tuple container raises at every
face; pymdp-shaped tuples convert with `system.from_positional("params", params)`
(and back with `to_positional`). Slot addresses (`likelihood:cue`, `lr_by_slot`,
`LearnState`, `describe`) name the same entries. Checkpoints serialised from
tuple containers do not load (dict leaves flatten in sorted-name order).

## Shape rules

> These four cost the probe most of its time. None of them was written down.
>
> 1. **Every transition table carries a trailing control axis — including an
>    uncontrolled factor's, where it has size 1.** So for every factor,
>    `parent_ndims == len(B_deps[f]) + 1`. `make_transition` slices that axis
>    unconditionally (`objectives/compiled.py:127`,
>    `jnp.take(..., action[f], axis=-1)`); a table without it is silently
>    mis-sliced and surfaces as an `opt_einsum` arity error several frames
>    deep, not as a validation failure. The rule is currently stated only in
>    an error string inside `learn` (`System.learn`'s transition-arity
>    `ValueError` in `system/core.py`).
> 2. **`parent_ndims` is a *count* of parent axes, not their sizes.**
>    `ConditionalCategorical(log_table, parent_ndims=2)`, never
>    `parent_ndims=(3, 2)` (which fails as
>    `TypeError: '<' not supported between instances of 'tuple' and 'int'`,
>    `distributions/conditional.py:122`).
> 3. **Tabular tables are child-first:**
>    `(..., num_levels, *parent_shape)` — `distributions/conditional.py:97-140`.
>    A transition is therefore `(next, *parents, control)` and a likelihood
>    `(level, *parents)`. If your table is parent-leading, use
>    `ConditionalCategorical.from_parent_leading(log_table, parent_ndims)`;
>    `from_probs(table, parent_ndims)` takes probabilities rather than logs.
> 4. **`ConditionalGaussian` is the exception**: parent axes *lead* there
>    (`loc: (..., *parent_shape, dim)`), because the family is a Gaussian
>    whose innermost batch axes are the parent axes. The child-first
>    convention governs tabular families only
>    (`distributions/conditional.py:21`).
>
> Two more worth knowing: `num_controls[f]` is `1`, not `0`, for an
> uncontrolled factor; and observations are typed per modality — an `int32`
> level index for a tabular modality, a float array for an LG one.
>
> A **memoryless** (reactive) factor is `B_deps[f] = ()` with a table of shape
> `(levels, num_controls)` and `parent_ndims=1`: the control axis is its only
> parent axis, and the slice at the control node's level is next tick's distribution.
> No need to declare the factor as its own parent with a flat table.

## Which modality families may observe which factor families

Enforced by `objectives.hybrid.validate_structure`
(`objectives/hybrid.py:250`), which runs from `System._validate` at
**face-call time**, not at construction.

| Modality declaration | Categorical factor | Gaussian factor | MixtureGaussian factor |
|---|---|---|---|
| `ConditionalCategorical` / `ConditionalDirichlet` (tabular) | ✅ | ❌ `hybrid.py:398` | ❌ `hybrid.py:398` |
| `LinearGaussianObservation` | ❌ unless its head has width one (`gaussian_emission`: a Gaussian emission over categorical parents alone, `design/structure-learning.md` S2) | ✅ (no switching parents) | ✅ |
| `LinearGaussianObservation` + categorical switching parents | — | ❌ `hybrid.py:429` | ✅ |
| `MLPConditional` (neural) | width-checked only | width-checked only | width-checked only |

An `MLPConditional` modality is not family-checked: `validate_structure`
only asserts one parent per factor named in `A_deps`, at the parent's width
(`hybrid.py:432-446`), so a checkmark would overstate what is tested.
`PredictorHead` is not in this table at all — it is a declaration of the
*amortized* path (neural state-space Systems), which `System._validate`
routes to `_amortized_structure` instead (`system/validation.py::_validate`).

A modality's `A_deps` is family-pure in its *observed* factor: the first
entry is the factor being observed, and for an LG modality any entries
**after** it must be categorical — those are switching (mode) parents, not
co-observed factors (`hybrid.py:405`, `hybrid.py:415`).

**"A discrete latent with continuous observations" is the first thing a
composer tries, and it is refused as declared.** A 3-state "which object is
meant" factor with a `LinearGaussianObservation` over it raises:

```
ValueError: modality 'distance' is linear-Gaussian but observes categorical
factor 'target'; a modality is family-pure in the factor it observes
(cross-family explaining-away is not supported). Two routes work: …
```

Two routes that do work:

- **Bin the features** into a tabular modality over the discrete factor.
  This is what `examples/compose_target_factor.py` does, and it is the route
  that keeps conjugate (Dirichlet) likelihood learning.
- **Declare a continuous carrier factor and switch it**: the discrete factor
  becomes a switching *parent* named after the observed factor in `A_deps`,
  and the observed factor must be carried as a `MixtureGaussian` (a
  1-component mixture is fine) with `mixture_k` set — the error at
  `hybrid.py:424-431` says exactly this.

*(This is a current constraint, not superseded history: `design/hybrid-system.md`
§D3 introduced it under a "v0.2 scope" heading and a dated amendment there
confirms it still holds as of v0.7. Its later amendments relax D4 for LG
`obs_targets`, not the family-purity rule. Making the direct declaration
first-class is a roadmap candidate in `project-brief.md` — flagged 2026-09-18.)*

## Names at the faces

`system.names` is a view of the System's name tuples: `names.factors` and
`names.modalities` are `factor_names` / `modality_names` themselves (a System
fills in `factor_{i}` / `modality_{m}` for any it was not given), `names.controls`
the control nodes `u_<f>` and `names.actions` their action nodes, one per control node.
Every tuple is in node-index order, the order of every positional structure
(tables, policy columns, what the converters produce); iterating a mapping gives
no such guarantee, so take order from `names`. The name tuples are the single
source of names and order: keys in `SystemParams`, targets and `lr_by_slot` refer
to them and are validated at every face.

Every face takes and returns beliefs, observations and actions as mappings keyed
by name: beliefs by factor (`factor_names`, else the generated `factor_{i}`),
observations by modality (`modality_names`, else `modality_{m}`), and actions by
action node, one per control node (`a_<factor>` for each *controlled* factor;
uncontrolled factors carry no entry); the value is the level emitted from the
control node's posterior. A discrete control's value is its integer level (a float
raises); a continuous one's is that factor's block of the action vector, of the
control's declared width. An action keyed by its control node is refused, naming
the action node. Entries are addressed by name, never by position: a face builds
its mapping in node order, but a mapping that has passed through `jax.jit`, `vmap`,
`lax.scan` or `tree_map` comes back with its keys sorted. A missing or unknown key
raises, naming both; a tuple or bare array raises too, since a misordered tuple is
silently misassigned where a misnamed key fails.

```python
obs = {"location_obs": 2, "reward": 0, "cue": 0}  # a modality and a factor never share a name
beliefs, _ = system.infer_states(params, obs)        # {"location": Categorical, "context": ...}
action, q_pi, tree = system.act(key, params, beliefs)  # {"a_location": Array(3, dtype=int32)}
```

Positional data (pymdp-shaped input, recorded runs) enters and leaves through one
converter: `system.to_positional(kind, mapping)` / `system.from_positional(kind,
positional)`, with `kind` one of `"beliefs"`, `"observations"`, `"actions"`,
`"detail"`, `"params"`; a complete `present` mapping has the observations'
structure and converts as `"observations"`. The positional
action is an int32 `(..., num_factors)` array with the no-op 0 on uncontrolled
factors (`from_positional` checks that they are 0), or the flat float
`(..., action_dim)` vector under continuous controls. Inside,
everything stays positional (tables, `A_deps`/`B_deps`, scan carries, the
planner's `(P, H, F)` policy tensor, which `score_policies` and `predict_beliefs`
take positionally and a bare `legal=` mask indexes, and `PlannerContext`).

The `InferenceDetail` that `infer_states(..., detail=True)` returns, and
`learn(detail=)` takes, follows the same rule: its per-modality fields
(`pre_reduce`, `mode_log_evidence`, `log_evidence`) are keyed by modality name and
`unsupported` by factor name (empty on the amortized path); its positional form
converts with kind `"detail"`. `aux_beliefs` (on `planner_context`,
`score_policies`, `imagine`, `rollout_returns`) is keyed by scoring-row name; each
value is the belief carry of that row's own nested System (the one in
`System.scoring`), keyed by *that* System's factor names, as its own faces return
it. A positional carry raises, naming
`dict(system.scoring)[row].from_positional("beliefs", ...)`; a row with no nested
System takes its entry as given. Design: `design/control-states.md` C10.

**Environments.** `system.as_env(params)` returns `(env, env_params)`: a
`hybrid_aif.envs.GenerativeEnv`, one public class for discrete and hybrid models,
whose `reset`/`step` take and return observations, states (by factor name) and
actions keyed as above, over the System's own positional process (discrete tables,
or the hybrid declarations); `env_params` keys its containers by name. The env is
the actuator: an action node with a kernel in `params.actuator` has the action it is
given realised as a draw from that kernel, and the world moves on the realised
action; `env.step_actuated(...)` returns it beside the observations and states.
`GenerativeEnv.wrap(env, system)` adopts an outside positional (pymdp-style)
environment the same way, its own params passing through untouched.
For an amortized model's filter states, `system.beliefs_from_states(params,
states)` gives the keyed beliefs `imagine` starts from.

## The canonical loop, with its real return arities

Three of the four faces return more than the design doc's signatures
suggest. This loop was once in `System.learn`'s docstring; this reference is
now its home:

```python
beliefs, detail = system.infer_states(params, obs, priors)   # (beliefs, InferenceDetail|None)
params          = system.learn(params, obs, beliefs, prev_beliefs, prev_action)
action, q_pi, tree = system.act(key, params, beliefs)        # (action, Categorical, TermTree|None)
priors          = system.update_empirical_prior(params, beliefs, action)
prev_beliefs, prev_action = beliefs, action
```

- `infer_states -> (beliefs, detail)`; `detail` is `None` unless
  `detail=True`, and the beliefs are bitwise identical either way.
- `infer_policies -> (q_pi, term_tree)` — the same pair, on the policy side.
- `act -> (action, q_pi, metrics)`; `metrics` is `None` when `metrics=False`
  and `{}` under the habit planner (that selection produced no tree).
- `learn -> params` for a conjugate-only model, and `(params, learn_state)`
  when the model has a gradient slot or `learn_state` is passed. Start that
  carry with `learn_state=system.init_learn_state(params)`; `None` with
  gradient slots present raises. On the amortized path `learn` raises: train
  with `train.world.fit_world`.
- `learn`'s `action` argument is **a_{t-1}**, the action that produced *this*
  tick's observations, from the carry — never the action `act` is about to
  return.
- `update_empirical_prior` returns the next tick's `priors`; pass `None` on
  the first tick, where `initial_priors(params)` is used. Its `action` is the
  mapping `act` returned (`{}` when nothing is controlled).
- `system.action_marginals(q_pi)` gives the first-step posterior of each control
  node, keyed `u_<factor>`.

**Hard constraints on the choice.** `act(..., legal=...)` and
`infer_policies(..., legal=...)` take a mask separate from `policy_prior` (E is
the soft prior; `legal` is the hard constraint; they never share a slot). A
mapping keyed by control-node name — each entry a bool `(levels,)` over that
node's levels for this tick's preconditions, or a bool `(policy_len, levels)`
per step of the plan; a node left out is unrestricted, an unknown key raises —
or a bare bool array over the enumerated policy set
(`legal={"u_location": jnp.asarray([True, True, False, True, True])}`). A
tuple by state factor is refused. An illegal level gets exactly
zero mass in `q_pi` and the action marginals; G is not re-scored; the mask is a
traced array, so changing it never retraces. Discrete controls only. Design:
`design/action-legality.md`.

**Two rollouts: expected and sampled.** Every prediction face, and the recognition
face, takes an optional `key`. `update_empirical_prior(params, beliefs, action)` returns the **expected**
belief, the distribution pushed through the transition, on every path; with
`key=` it returns the **sampled** belief, one draw carried as a belief of the
same family (a one-hot categorical, a point-mass Gaussian, one component's point
of a mixture, a straight-through one-hot for an amortized code).
`predict_beliefs(..., key=)` splits the key once per step. `infer_states(..., key=)`
returns the posterior collapsed the same way — one-particle filtering, so an agent
that acts on samples hands the actor, the scoring rows and the next transition one
draw (on the amortized path the codes collapse and the carries, deterministic given
their inputs, pass through); keyless it is the mean-field posterior, bitwise as before. `score_policies(...,
key=)` hands the key to scoring rows that roll a model of their own, which then
take the sampled rollout; rows that roll nothing ignore it, and a planner's
`plan_key` never reaches them (`infer_policies`/`act` score the expected rollout
as before). Which rollout a call gets is decided by the key alone, never by the
model, and the keyless call is bitwise what it was. A sampled belief is a legal
carry on every path; it is a point mass, so a collapsed Gaussian or mixture
cannot be conditioned or given a finite entropy, while a collapsed categorical
can (its zeros are floored at `LOG_ZERO`). A sampled *planner* on the tabular
paths (several particles per policy, averaged) is a roadmap item, not a feature.
Design: `design/amortization.md`, FLAG (1)'s amendment of 2026-09-20.

**Compiled cores.** `replay` and `rollout` check their arguments eagerly and
then run a compiled core, so a second call with the same model, trace length
and switches reuses the program; pass the same `policy_fn` object to `rollout`
to share it. The composition driver's `agent_respond` compiles its tick once per
System and belief layout (a mixture System's first tick pads its priors, so
twice), with the parameters as an argument rather than a baked-in constant; an
all-illegal `legal=` mask is still refused eagerly, when the responder is built.

**The training side's episode format.** `train.buffers.Episode` carries the
engine's columns — `obs`, `action`, `reward`, `cont` (one minus terminal, as
recorded), `is_first`, `is_last` — and a task's own per-step columns in
`extras`, a dict by name. A buffer is allocated with the task's schema,
`StaticBuffer.empty(..., extras={name: ColumnSpec(shape, dtype, pad)})`, and
holds exactly those columns; `pad` says whether `fit_length` freezes the last
value ("hold", a flag about the final state) or writes zeros ("zero", nothing
happened there). `episode(obs, action, reward, is_first=, is_last=, cont=,
**extras)` is the one constructor; `as_episode`/`episode_columns` bridge the
host-side `ReplayBuffer`, whose dynamic schema carries the extras flat.
`seed_stats(episodes, success=name)` reports a success rate only when a success
column is named. What ends an episode, and which flags it records, is the
task's business: the R-AIF replication keeps its four columns and its
`cont = 1 − is_success` in `reference/raif/columns.py`.

**Trainer knobs, in the engine's words.** `imagine(policy_entropy_scale=,
prior_entropy_scale=, bc_scale=, bc_weights=, start_actions=)`: the policy
entropy bonus defaults to DreamerV3's, the state-prior bonus and the weighted
behaviour cloning default to off. `fit_ensemble(member_weights=)` weights each
member's loss (a zero removes that member's gradient for the step — its optimizer
moments still apply; the general form of a member gate). `fit_world(roles=)` names which declared modality is the reward
and which the continuation when the names `reward`/`cont` are not used; the
decoder is always the modality the encoder reads, and each head contributes
its own loss, whatever its family. `fit_world(return_states=True)` adds a fourth
return, the posterior states of the pass the gradient was taken through
(gradient stopped), for a training step whose other trainers start from the
world model's states without recognising the batch a second time. Neural constructors take `init="default"`
or `init="trunc_normal_avg"`, and the sized ones (`MLPPredictor`, `MLPCategorical`,
`PredictorHead`, `GRUCell`, `Deterministic.gru`) require their width. A row a
reference subtree defines (the CRSPP's) is registered when that package is
imported: `import hybrid_aif.reference` before building a System or a document
that names it.

Written eager for readability; every face is pure and jit-safe. Wrap the
per-tick faces in `eqx.filter_jit` once per model *shape*, and pay the
compile at a stationary moment.

### `learn`'s per-slot rates: the key grammar

`lr` applies to every learnable slot. `lr_by_slot` overrides it by
**slot-qualified key**: `"likelihood:<modality>"`, `"transition:<factor>"`,
`"initial_state_prior:<factor>"`. `decay` and `decay_by_slot` use the same keys.
A third segment addresses one independent posterior block, as in
`"likelihood:pose:noise"` or `"likelihood:pose:head"` for a
`LinearGaussianObservation`. The joint `LinearGaussianObservationMNIW` has no
blocks. Any segment may be `*`. **The most specific match wins**, where
specificity is the number of literal segments, and two matches of equal
specificity on one variable raise:

```python
decay_by_slot={"*:*": 0.999, "likelihood:pose": 0.97}
lr_by_slot={"likelihood:pose": 0.0, "likelihood:pose:noise": 1.0}  # head frozen
```

The second line freezes the head and learns the noise: the three-segment key
is more specific than the two-segment one, so it overrides it for the noise
block. State keys and `obs:` keys are not rate-addressable and raise.

**A static `0.0` and a traced zero are different.** A static Python `0.0`
skips the slot at trace time, decay included, so its einsum is never emitted.
A traced rate multiplies the statistic, which is what makes an in-scan
curriculum expressible: `lr_by_slot={"transition:x": jnp.where(t < 1000, 1.0,
0.0)}`. `jnp.float32(0.0)` is traced, so it pays the FLOPs a static `0.0`
would skip. A frozen slot that should still drift under decay needs a traced
zero, because a static one skips the decay too.

**Decay** is `α ← decay·α + (1−decay)·anchor + lr·stats`, anchored at the
declaration's `alpha0`: its prior (`from_table(..., anchor="prior")`), a
uniform table (`anchor="uniform"`, a scalar), or an array it was built with.
`decay == 1.0` is exact conjugacy and reads no anchor. `decay != 1.0` raises
on a declaration without an anchor and on a neural slot. Design: `design/parameter-learning.md`
D1 and D7, `design/per-variable-infogain.md` V1 and V7, `design/amortization.md` A2.

## Reading the term tree, and checking a selector took effect

`tree` is `{row_name: {leaf_key: Interval}}`, each `Interval` carrying one
value per policy. On the probe's model (factors `target`, `mode`;
modalities `distance`, `size` over `target`, `answer` over both):

| `epistemic=` | `tree["infogain"]` keys |
|---|---|
| `True` | `answer, distance, size, likelihood:distance, likelihood:size` |
| `False` | *(row absent)* |
| `("target",)` | `answer, distance, size` |
| `("mode",)` | `answer` |
| `("obs:distance",)` | `distance` |
| `("*",)` | `answer, distance, size` |
| `("likelihood:*",)` | `likelihood:distance, likelihood:size` |

**State-infogain leaves are keyed by *modality*, not by the state variable
you named.** Naming a state factor mounts the channel infogain of every
modality that observes it, which is why `("target",)` yields three leaves
and `("mode",)` one. Parameter variables are unambiguous: they key as
`<slot>:<name>`.

**A leaf's *value* is the channel's information about the variables you
selected among its parents** (`design/per-variable-infogain.md` V9):
`I(o; S ∩ D)`, with the unselected parents marginalised under the current
belief. A selection that covers every parent of a channel, including the
default `epistemic=True`, gives the channel's joint info gain `I(o; D)`, and
so does `obs:<modality>`: selecting the observation variable buys everything
it reveals. Under `risk_ambiguity` the selector mounts `ambiguity` exactly where it would
mount `infogain`, with the same selection, and the risk of every channel is untouched by it:
a mounted channel's `risk + ambiguity` equals `−(utility + infogain)`, and an unmounted one
carries its risk alone, which on a channel without a preference is `−H[q(o)]` (design:
`design/per-variable-infogain.md` V11). A Gaussian emission over categorical parents alone
(`gaussian_emission`) is a channel like any other: its leaf is `I(y; z)` about the parents you
selected, and it has no preference, so under `risk_ambiguity` its risk is `−H[q(y)]`.

**On a Gaussian-carried factor** each linear-Gaussian channel has its own leaf, `I(y_m; x)`,
keyed by modality, and where two or more mounted channels read the factor a term
`observation_interaction`, keyed by the factor, holds what the per-channel leaves miscount:
on the whole factor the total correlation of the readings, `Σ_m H[q(y_m)] − H[q(Y)]`, the
information the sensors share and the leaves count twice. It is a cost in G, so the
information the step's readings carry together is `sum(tree["infogain"][m] for m in channels)
− tree["observation_interaction"][factor]`. Under `risk_ambiguity` the joint risk of the
readings, `KL(q(Y) ‖ ∏_m C_m)`, is the channels' risks plus their total correlation, which
the term `observation_correlation` holds whenever two or more channels read the factor,
whatever the selection (risk is not the selector's). A continuous factor no channel with an
information-gain estimator observes has no information to gain and no `infogain` leaf; its
predicted entropy is the opt-in `uncertainty` row (`terms=("uncertainty:<factor>",)`), a
cost. In a `merge()` the row is the composite's to declare.

On the probe's model, after one tick observing `(0, 0, 0)`, the `answer`
leaf over the nine two-step plans of the `mode` control (pick0, pick1, ask)
is:

| `epistemic=` | `answer` leaf | why |
|---|---|---|
| `True`, `("target", "mode")`, `("obs:answer",)` | plans with one `ask`: 0.151 nats; `ask, ask`: 0.302; no `ask`: 0 | the joint `I(answer; target, mode)` |
| `("target",)` | the same values | `mode` is set by the plan, so imagination holds it at a point and `I(answer; mode \| target) = 0` |
| `("mode",)` | 0 for every plan | the answer tells you nothing about a `mode` the plan already fixed |

So **a selector's effect shows in the key *set* and in the values.** To check
which channels a selector mounted, compare `sorted(tree["infogain"])`
against the modalities your `A_deps` says observe the variables you named.
To check what a leaf pays for, compare it against the same leaf under
`epistemic=True`.

**Planning that expects to observe.** `System(planning_belief="posterior")` makes the
information-gain leaves of a Gaussian-carried factor's channels (and their interaction) the gain
of each step's observations given the plan's earlier ones: the planning carry holds, beside the predictive belief
every target reads, a belief conditioned on the observations the plan expects
(`objectives.hybrid.BeliefPair`; a mounted row sees it as `StepContext.qs_epistemic`
and, after the step's own observations, `qs_conditioned`). With a constant noise the
gain is still the same for every plan; a noise that varies over the state
(`GridNoiseMap`) or a per-step availability (`infer_policies(present=)`, a bool or a
`(policy_len,)` array per modality) is what makes plans differ. Design:
`design/planning-with-observations.md`.

**Closed-loop scoring.** `System(planning_belief="posterior", feedback="linear_quadratic")`
scores a plan as the policy that follows it and corrects deviations from it by linear feedback on
the estimate, `u_τ = ū_τ + K_τ (x̂_τ − m̄_τ)`, the gains the finite-horizon linear-quadratic
regulator's for the plan's Gaussian targets. The state at each step is then `N(m̄_τ, Σ_τ + C_τ)`:
the predictive mean, the epistemic chain's posterior covariance, and the spread the corrections
leave on the estimate. Each Gaussian state and terminal target on a Gaussian-carried factor is
scored by its cross-entropy on it (`risk[<factor>]`, `terminal_risk[<factor>]`), a Gaussian
observation target on a linear-Gaussian channel by its utility on the reading's closed-loop
distribution (`utility[<modality>]`; under `risk_ambiguity`, `risk[<modality>]` = `−utility −
H[q(y)]`, the predictive entropy, so the switch moves the two decompositions alike), and a
Gaussian target on a continuous control node, whose precision is the regulator's control cost,
adds the corrections' expected cost to its leaf. A control with no target costs nothing to
correct with (the regulator is unconstrained and does not see the box). So what the plan's
readings will reveal is worth what it lets the corrections remove, beside its information gain,
and an answer about something no control reaches is worth its information gain alone. A
nonlinear transition is linearised as it propagates (Jacobians at the mean under
`first_order`, sigma-point averages of the Jacobians under `unscented`). Categorical targets
keep their scoring, so a mixed model's G mixes a closed-loop continuous part and an open-loop
categorical part. Refused: `planning_belief="prior"`, a target on a mixture-carried factor (or on
a channel observing one), a `MixtureGaussian` target, a target on a factor with a neural
transition, and the amortized path. Design: `design/closed-loop-planning.md`.

**Selecting dimensions.** `epistemic=("pose:b1",)` selects one dimension of a continuous
factor (named with `System(dims={"pose": ("x", "y", "yaw", "b1", "b2")})`, or `pose:dim3`
by position). Keys on one factor select one sub-block, and the leaves keep their keys: on a
Gaussian-carried factor each channel's leaf is its information about those dimensions with the
others marginalised, `I(y_m; x_S)`, and the interaction is the interaction information `TC(Y)
− TC(Y | x_S)`, so the leaves less the interaction are `H[q(x_S)] − H[q(x_S | y)]` (on the
epistemic chain under `planning_belief="posterior"`, on the predictive belief under
`"prior"`). On a sub-block it can be negative: two readings that each say little about `x_S`
but together pin it (synergy). Under `risk_ambiguity` each channel observing the factor takes
`E H[y | x_S]` as its ambiguity, and `observation_interaction` is `−TC(Y | x_S)`, so the
ambiguity is the readings' joint `E H[Y | x_S]`. `pose:*`, or every dimension, is the factor
itself, bit for bit. A sub-block of a mixture-carried factor is refused, and so is
`obs:<modality>` beside a sub-block of the factor that channel observes: the channel's leaf
would be asked to be about two different sets of variables. The `uncertainty` row takes the same keys,
`terms=("uncertainty:pose:b1",)`, and charges `H[q(x_S)]`. Design:
`design/per-variable-infogain.md` V10, V11.

## Merging Systems: the bindings

`merge({name: System}, {name: params}, bindings)` makes components one
`(System, SystemParams)`, namespacing every node `"<component>.<name>"`, and
identifies latents across them, each binding rewritten as an ordinary edge.
Bindings name nodes by their namespaced names.

**`ActionControl(driver, control)`.** `control` names a control node of one
component (`"world.u_location"`), identified with the driven control node of
another (same step) or with a state of it (relay). The driver's spelling decides
when the driven factors read the decision:

| `driver` | read | what the composite holds |
|---|---|---|
| a control node of the driving component (`"agent.u_location"`) | at the same step | one node: every factor that read the driven node reads the driver's, from t to t+1 on the call that chooses it; the driven node leaves `controls` and `names.controls`, and its action node goes with it |
| a state factor of it (`"agent.location"`) | a step later | the one-tick relay: the state becomes a `B_deps` parent of the one factor reading the node, in its place, its control axis re-read as that parent and a no-op axis of size 1 appended |

Level counts must agree (a 5-level node binds a 5-level node, or the state of a
5-level factor), and a continuous pair must agree on width; the driver's box in
`params.control` governs the identified node and the driven node's box is
dropped. A relay from a continuous state is refused. A driver that is neither a
control node nor a state factor is refused, naming both spellings. The factor
spelling of the control node, `ActionControl(..., control="world.location")`, is
refused naming its node: `control="world.u_location"` replaces it. Bindings compose in
any order: a chain of same-step bindings is one node, a node identified with one
a `BoundaryPrior` terminates is terminated with it, and a relay into a node that
more than one factor reads is refused whichever binding comes first. A lag on a same-step
read is declared, not configured: an echo state whose table copies a control
node is an ordinary state, and binding from it gives the relay.

```python
from hybrid_aif.composition import ActionControl, BoundaryPrior, ObservationLink, merge

composite, params = merge(
    {"agent": agent, "world": world},
    {"agent": agent_params, "world": world_params},
    [
        ActionControl(driver="agent.u_location", control="world.u_location"),  # same step
        ObservationLink(modality="agent.location", factors=("world.location",)),
    ],
)
composite.names.controls  # ("agent.u_location",)
```

**`BoundaryPrior(control, prior)`** terminates a discrete control node, replacing
its decision with a fixed prior: an `int` level slices the control axis of every
factor that reads the node (a delta), and a `Categorical` or probability vector
averages each reader's table over it (uniform: no one is steering). A reader of
several nodes keeps its other control axes. Native control nodes merge like the
sugar's, and a composite with native nodes is described like any other.

**Names a binding removed.** A node a binding removed is gone from the
composite, and a mapping keyed by it or by its action node (an action, a node
belief, `legal=`, a target) is refused with a sentence naming what it became:
under a same-step binding, the driver's node; under the relay, the state it reads;
under a boundary prior, nothing. A target on a removed node is refused at
`merge()`. The `Subsystem` record names nodes: `controls` (still chosen here),
`actions` (each named control node's action node), `identified` (`(node, driver)`
for each same-step identification), `relayed` (`(node, state)`) and `terminated`.

**One clock.** `params.durations` (a round's length per level of one discrete control node)
carries under its component's prefix, and a `subsystem` view copies it. A round steps every
factor once, so a composite has one clock: two components that both declare durations are
refused at `merge()`, as is a binding that removes the node carrying them
(`design/step-durations.md` T1).

**Legality across a merge.** A same-step binding leaves one choice point, the
driver's, so a mask on the identified node is the driver's constraint: key it
by the driver's node, where it applies as any node's mask does, on `act`,
`infer_policies`, `replay` and `agent_respond`. A mask keyed by the driven node
is refused naming the driver's node, and one keyed by a relayed node is refused
naming the relay.

**The views.** `subsystem(composite, params, name)` gives one mind's System. The
driven mind's view has no node for the identified control: its factor's axis is
terminated by a prior over the partner's node, uniform by default, or given in
`partner_priors` keyed by the node's name (a relay stub's own control node too).
The driver's view keeps its node, and a boundary stub of
the driven mind reads it at the same step. Boxes and targets on a node travel
with the view that chooses it.

## The same model as a document

Everything above is the Python form. The same declaration travels as a JSON
document, the form a program (a proposer, a validator, an external tool) reads,
diffs and patches. The specification is `description-format.md`; the design is
`design/model-description-0.3.md`.

```python
from hybrid_aif import describe, build, schema

document, arrays = describe(system, params)       # arrays: {} when every clamp is inline
system2, params2 = build(document, arrays=arrays)  # bitwise the declaration back
```

The document names every variable once under `variables` (a latent, an
observation, a control node, an action node, a parameter), each with its domain
and size; writes each conditional distribution as a factor record keyed by its
child and a label (`factors["location"]["transition"]`), with its `parents` by
name and its static arguments, the parameter variables, under `parameters`;
keeps `controls`, `observed` and `external` under `relations`; gives the fixed
parameters' values under `clamps` (inline, a ref into the arrays file, or an
`init` recipe drawn with `build(..., key=...)`), a learned parameter being a
`prior` record under its variable instead; writes the objective as a list of
terms; and the planner, horizon and the other settings under `configuration`.
The four shape rules above are what `build` checks a table against: child-first,
then one axis per parent in the order the record lists them, a control node
among them where the transition reads it.

A merged composite is written flat, its variables carrying the components'
prefixes (`world.location`): the bindings are structure, and `build` rebuilds
each component's record from the prefixes, so `run` and `subsystem()` work on the
built model as on the merged one.

`build` checks the whole document before it constructs anything, each fault one
sentence naming the address and what to write instead, then runs the library's
own validator. A consumer's own data rides under `extensions`, keyed by its name,
and any other unknown top-level key is refused. `schema()` returns the JSON
Schema (2020-12), checked in as `src/hybrid_aif/description/description-0.3.json`.
`save(description, path)` writes `path.json` and, when there are refs, `path.npz`
beside it; `load(path)` reads both. Not in the document: optimizer state
(`LearnState`), beliefs, policies, replay, and the planner's and learner's
call-time arguments.

## Replaying a recorded trace

A replay drives a declared model through what a controller actually did: at
each tick the model infers from the recorded observations, its own policy
inference runs beside the record as a counterfactual, and the **recorded**
action drives the empirical prior forward. Design: `design/replay.md`. It is a
function, not a method, because it is something done *with* a model rather
than something the model does within a tick (the same line that puts
`describe` and `merge` outside the class).

```python
from hybrid_aif import replay, replay_document, stack_traces

out = replay(system, params, observations, actions)  # {modality: (T, …)}, {a_<factor>: (T, …)}
out = replay_document(document, observations, actions, arrays=arrays)  # build first, then replay
```

`actions[t]` is the action taken after `observations[t]`. Every field of the
result is stacked over time: `beliefs` and `priors` per tick (keyed by factor),
the counterfactual `q_pi` and its `trees`, `chosen` (the model's own deterministic
choice, keyed by action node `a_<factor>`), `action_log_prob` (the exact probability of the recorded action under
the model's policy posterior — the fitting objective of behavioural model
comparison; `-inf` where the model gives the recorded action no probability), `log_evidence` where the path computes one, `valid`, and `exercised` — which
modalities were present on at least one valid tick, so a scorer never silently
scores an address the trace never informed. The per-tick `trees` are the
expensive part of a replay: every tick's planner call keeps its per-policy
leaves and the scan stacks them over `T`. Pass `metrics=False` to keep
`q_pi` and drop the trees, which roughly halves the planner's share of the
tick; the default stays on because the trees are what a replay is usually
read for.

**Partial evidence.** `present` maps a modality to a `(T,)` bool, whether it was
observed at each tick; a modality left out was observed at every tick (`None`: all
were), and an unknown key raises. An absent modality contributes nothing that tick:
the zero log-likelihood message on the discrete path, a skipped conditioning step
on the LG path. A tick where every modality is absent is a predict-only tick, the
prior pushed through the transition. The same `present=` keyword is on
`infer_states` for a single tick, one flag per modality (`{"cue": False}`; a
Python `False` skips the update statically). Whatever value sits in the record
where a modality is absent is ignored, so a padded or dropped frame may leave
anything there.

**Evidence no state explains.** Several modalities can jointly leave an
observation tuple that no state produces (a disjunction the factored likelihoods
cannot express, a dark camera). The tabular inference is *tempered*: `-inf` is a
legal table entry, nothing ever produces `NaN`, and at such a tick the posterior
is the arithmetic's own answer — the prior on the contradicted factor,
reweighted by whatever evidence still discriminates — **flagged**, never
replaced. `infer_states(..., detail=True).unsupported` is one bool per factor, keyed by factor name — "under
the beliefs held about its co-parents, no level of this factor was compatible
with the evidence"; it does not name which factor the contradiction is about,
since a contradiction reaches co-parents through the expected message — and
`replay` surfaces it as one `(T,)` array per factor, keyed by factor name. `build` refuses `NaN`
and a dead observation level (one no state produces); `check_joint_support(system,
params, max_combinations=...)` enumerates the observation tuples no state
configuration explains and refuses above its budget rather than sampling.
Design: `design/zero-support-evidence.md`. An exact zero in a table with several parents is a hard exclusion:
the mean-field message is the *expected* log-likelihood over the co-parents, so
while a co-parent is uncertain a consistent observation can still meet a `-inf`
and be flagged and ignored (the T-maze cue with a one-hot cue table does exactly
this). Floor tables you do not mean as exclusions, e.g. by `1e-16`.

**Variable-length traces.** `valid` is `(T,)` bool; an invalid row leaves the
carry untouched and its outputs are unspecified. `stack_traces([(obs, actions,
present_or_None), …])` (each keyed as `replay` takes it) pads a list of traces to one
length and returns `(observations, actions, present, valid)` with a leading batch axis;
`jax.vmap(lambda o, a, m, v: replay(system, params, o, a, present=m, valid=v))`
replays the batch.

`learn=True` places `learn` between perception and action from the second tick
on and returns the final `params` — what the model would have learned from the
episode — with the optimizer carry beside them for a model with gradient slots.
It is not available on the amortized path, where `System.learn` raises (train
with `fit_world`). Replay itself runs on all three paths: on the amortized path the recorded action's likelihood is the
actor's own log-density, the counterfactual is the habit planner's draw under
`key`, `externals=` supplies recognition inputs per tick, and a tick whose
recognised modalities are absent is predict-only.
