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
| `state_names` | `tuple[str, ...]` | generated | One name per state factor, in node order; the `epistemic`/`state_targets`/rate keys. `None` at construction fills in `factor_{i}`. No node may be named after a key's reserved first segment (`obs` or a slot word). |
| `control_names` | `tuple[str, ...]` | from the sugar | One name per control node, in `controls` order. `None` fills in the sugar's `u_<f>` per controlled state factor; a native declaration (`controls=`) names its control nodes. `node_names` (read only) is `state_names` followed by `control_names`, the list the index-based fields (`B_deps`, `A_deps`, `D_deps`, `H_deps`, `controls`) index into. |
| `dims` | `tuple[tuple[str, tuple[str, ...]], ...]` | `()` | Names for a continuous state factor's dimensions, given as `{factor: (name, ...)}` and stored as pairs in factor order; `dict(system.dims)` to read. `epistemic` selects a dimension as `<factor>:<name>`; an undeclared factor's dimensions are `dim0`, `dim1`, … by position. One unique name per dimension, each a single key segment (no `:`, `.`, `*` or brackets); `dimN` may name position `N` only. |
| `action_names` | `tuple[str, ...]` | from the sugar | One action node per control node, in `controls` order: the actuator channel, a child of its control node. A native declaration (`controls=`) names them, paired with the control nodes by position, both names free; the sugar names the pair `u_<f>` / `a_<f>`. A modality may observe an action node: `A_deps` names action node `k` as `len(B_deps) + k`. |
| `emissions` | `tuple[tuple[str, Emission], ...]` | `()` | The outputs the System realises from its state factors, **by name**: `{"fix": ("x",), "img": Emission(("x", "terrain"), realise="mode"), "cell": Emission(("x",), realise="mode", kernel=GRID)}` at construction (a tuple of parents is a draw), stored as `(name, Emission)` pairs in the order given. An emission is the action node of a state factor: its parents are state factors and control nodes (a control parent reads the node's realised level), its kernel is `SystemParams.emission[name]` (a likelihood of any family, in the parent order given) or the static kernel the declaration names, and its port type is read off the kernel as a modality's is. `kernel=None` means the kernel is the entry `params.emission[name]`, which must be there; a static kernel is declared on the emission: `kernel="identity"` (the **identity** of its one state factor: a categorical factor's level, or under `stats` its probability vector; a continuous factor's mean under `mode` and a draw under `sample`, `stats` being deferred) or a `Grid` (an indicator over the cells of one continuous factor: from a point the cell it falls in, from a belief `cell_mass`, of which `mode` is the largest cell). It has no node index: nothing in a model scores it, so no `C_deps` key or parent, `readings` entry, `branch_on` modality, target key or `epistemic` key may name it; the one way a model reads it is `observes`. `Emission.realise` is `"sample"` (a draw from the kernel), `"mode"` (its most probable level or mean) or `"stats"` (its probabilities, or its mean and covariance). `names.emissions` lists them; `emit` realises them. |
| `observes` | `tuple[tuple[str, str], ...]` | `()` | Which variable each slot observes, given as `{"cell_obs": "cell"}` and stored as `(slot, variable)` pairs. A slot is a modality or a control node and a variable an action node or emission of this System; the slot's port type must equal the variable's. A control node in `observes` is **driven**: the variable's realised value is its level, and its `action_names` entry is `None` (nothing chooses it, so nothing realises it): it is an input, no planner chooses it, and `policies`, `infer_policies`, `act` and `elapsed` refuse naming it. A slot the map does not name observes the variable of its own name and is an in-port (`ports`). The datum of an observed variable is an observation like any other: `infer_states`, `learn` and `replay` take observations **keyed by variable** (one datum feeds every slot observing it, a key naming a renamed slot is refused naming the variable), and the datum of a variable the System generates may be left out, the slot then being absent. An own-belief read is an emission of the System that one of its own modalities observes; `replay` and `run` realise the emissions after a round and feed them back as the next round's observation. |
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
width, and the CEM and gradient planners search inside it), `emission` (the kernel of each emission, a likelihood of any family keyed by the emission's name; no inference, planning or learning face reads it: `emit`, `ports`, `sample_initial`/`sample_step`, `describe` and the description length do), `actuator` (the kernel
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
> Two more worth knowing: the trailing control axis has size `1`, not `0`, for an
> uncontrolled discrete factor (`System` no longer carries a public `num_controls`
> field — sizes are inferred from the transition tables); and observations are typed per modality — an `int32`
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

`system.names` is a view of the System's name tuples: `names.states`,
`names.controls` and `names.modalities` are `state_names` / `control_names` /
`modality_names` themselves (a System fills in `factor_{i}` / `u_<f>` /
`modality_{m}` for any it was not given), and `names.actions` the action nodes,
one per control node.
Every tuple is in node-index order, the order of every positional structure
(tables, policy columns, what the converters produce); iterating a mapping gives
no such guarantee, so take order from `names`. The name tuples are the single
source of names and order: keys in `SystemParams`, targets and `lr_by_slot` refer
to them and are validated at every face.

Every face takes and returns beliefs, observations and actions as mappings keyed
by name: beliefs by state factor (`state_names`, else the generated `factor_{i}`),
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
action is an int32 `(..., num_states)` array with the no-op 0 on uncontrolled
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

**Ports and emissions.** `system.ports(params)` returns `Ports(ins, outs)`, each a mapping
from a port's name to its type, `(LEVEL, levels)` for a table and `(FLOAT, dimension)` for a
linear-Gaussian or neural kernel. The in-ports are the modalities and the control nodes (each
under its own name: a link into a control node drives it); the out-ports are the action nodes
(typed as their control nodes) and the emissions (typed by their kernels). A port is its name:
there is no node index. `system.emit(key, params, values)` realises every emission, each by its
own `realise`, with `split(key, E)[e]` for emission `e`. `values` holds each parent by name: a
state factor as a point (a level or a float vector, a world's state) or as a belief (an agent's
posterior), a control node as a level. From points a `sample` conditions the kernel and draws,
exactly the draw the forward process makes of an observation; from beliefs it draws the parents
and then the kernel, an exact predictive draw, and `mode` and `stats` are those of the predictive
distribution. A neural kernel is realised by a draw only.

**The mirror.** `hybrid_aif.system.mirror.mirror(system, params)` returns the World a System
models, as a System that acts: every modality becomes an emission of the same name over the same
parents (an observed action node becoming the control node it realises), the state factors,
transitions and priors are kept, and everything that belongs to the planner, the learner or the
posterior is dropped. Parameter variables are kept as they are; `lower=True` replaces every
learnable declaration with its posterior mean, or with one draw per slot under `param_key=`. A
System that declares emissions has no mirror. `sample_initial` and `sample_step` run a System's
emissions forward from a point state: a control node an input drives takes the input, and a free
one is drawn from its habit with a key folded apart from the process's own (uniform without
one; the mirror turns `policy_prior` into a distribution over the one-step level tuples it
enumerates, each holding the mass of the first steps equal to it, and its `policy_len` is 1). Every
emission is realised by its declared rule.

**Environments.** `system.as_env(params)` returns `(env, env_params)`: a
`hybrid_aif.envs.GenerativeEnv`, one public class for discrete and hybrid models,
whose `reset`/`step` take and return observations, states (by factor name) and
actions keyed as above, over the System's mirror run forward by a positional process (discrete
tables, or the hybrid declarations); `env_params` keys its containers by name. The env is
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
to share it. `run`'s `plan()` step compiles its tick once per view and belief
layout (a mixture System's first tick pads its priors, so twice), with the parameters as an
argument rather than a baked-in constant; an all-illegal `legal=` mask is refused on the step's
reset call, where the params are concrete.

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
block. State keys and `obs:` keys are not rate-addressable and raise. A parameter variable
read by several factors has one rate, `"parameter:<name>"`; the key of a reader
(`"transition:<factor>"`) raises and names it, except that a static `0.0` on a reader keeps its
statistic out of the variable. A literal key outranks a pattern: `{"*:*": 0.0, "parameter:T": 1.0}`
learns `T` from all its readers, `{"*:*": 0.0}` freezes everything, and
`{"parameter:T": 1.0, "transition:a": 0.0}` learns `T` from every reader but `a`.

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

`merge({name: System}, {name: params}, bindings)` makes components one particle, one
`(System, SystemParams)` with one blanket to the world, namespacing every node
`"<component>.<name>"`, each binding rewritten as an ordinary edge. Bindings name nodes by
their namespaced names. There are three kinds: `Link`, an arc of the graphical model of the whole,
`Identify`, two parameter variables made one, and `Clamp`, a control node held at a constant. A
binding is one of two shapes. A binding that identifies a latent outright (`Identify`, a relay,
a structural `Link` into a modality's parents, which makes one modality read several components'
factors, a terminal link) makes the components share that variable, with messages both ways and
no blanket between them on it. A binding that is blanket-shaped (a `Link` from an action node
into a control node, or a belief read, a `Link` with a `realise`, in which a modality of one
component takes another component's belief as its datum, pushed through a kernel at the source
and realised by `sample`, `mode` or `stats`) leaves its component *nested*: its beliefs are its
own, the rest of the particle is its external world, and it reads that world only as sensing.
Separate minds that exchange only action and sensation are joined by `couple`, not `merge`
(`design/nesting-and-coupling.md` N2, N3).

`Link(source, target, realise=None, kernel=None)` reads from its two ends and `realise`:

| `source` | `target` | `realise` | meaning |
|---|---|---|---|
| an action node (`"agent.a_location"`) | a control node (`"world.u_location"`) | none | `Identify` of the two ports: the node is driven by the realised decision (below) |
| a state factor (`"agent.location"`) | a control node | none | the relay: read a step later (below) |
| state factors, one name or a tuple | a modality | none | structural: the modality's complete parent list, in its likelihood's axis order, one link per modality |
| one state factor of another component | a modality | `"sample"`, `"mode"` or `"stats"` | a belief read (below); `kernel` is the identity when `None` or `"identity"`, or a `Grid` |
| state factors | `"terminal:<factor>"` | none | the factor's terminal target is conditioned on these factors (its `H_deps`) |

A link from a realised port (an action node, an emission) takes no `realise`: the port is
realised once, by its own kernel, and every link from it carries that value. A `kernel` without a
`realise` is refused. A link from an action node or an emission into a modality is a coupling's
(`couple` carries it); a `merge` refuses it, naming `couple`. A control node as a source is refused,
naming its action node.

**Port identifications: `Identify(out_port, in_port)`.** An out-port is an action node or an
emission, an in-port a modality or a control node; `Identify` joins one of each (written in either
order) into one variable, which the in-port observes: it writes `(in_port, out_port)` into the
composite's `observes`. Every in-port is identified at most once, an out-port may feed many, and
the two port types must be equal. A modality in-port reads the out-port's datum as an observation
(`world.gauge` observed by two leaves is one variable, two observing slots). A control node
in-port is a **driven** node: it stays in its own leaf (a component that is not itself composed
of others), carries the *realised* value of the
out-port (control-states.md C13), has no action node of its own (`action_names` entry `None`)
and is never chosen by a planner (`driven_nodes(composite, name=None)` lists a component's driven
nodes, each with the variable that drives it; a `Subsystem` record's `controls` and the planner's
action nodes leave them out). `Identify("mission.option", "rover.u_opt")` under `merge` and
`Identify("agent.a_v", "world.v")` under `couple` are one form, and `Link(action node, control
node)` under `merge` is the same binding. A composite that carries driven slots is planned as one
mind through `joint_planner`, which collapses each driven node into its driver's (below), or
through its leaves' views (`subsystem`).

**`couple(components, bindings, fixed)`** returns a `(System, SystemParams)`: the splice of the
components (the stage of `merge` that renames and concatenates the components' variables and
tables before any binding is applied) followed by port identifications and `Clamp`s (`fixed` maps
a discrete control node to a level or a distribution, as a `Clamp`). It refuses every other
binding, naming `merge`: a structural, relay, terminal or belief-read `Link`, an `Identify` of
parameter variables. A coupling holds `|A| + |B| - k` decision variables for parts of `|A|` and
`|B|` (a control node and its action node are one decision variable, so the zoo agent has 14,
the world 13 and the coupling with the world's node driven 14 + 13 - 4 = 23), a variable an
out-port feeds to several in-ports counted once; its document writes each identified variable
once and each observing slot's likelihood record names it (`observes`), a driven node is a slot
of its driver's variable (no variable of its own, written under `relations.driven` with the
action node it had under `relations.driven_actions`), and each leaf's own record, its scoring
terms (each tagged with its `leaf`) and the planner fields it holds unset under `components`. A
merge may
take a coupling as a component and a coupling a merge. A coupling is run leaf by leaf: every
planning face (`policies`, `infer_policies`, `act`, `elapsed`, `replay`) refuses it, and a leaf is
planned through its view, or the whole as one mind through `joint_planner`. `ports` lists a
control node no `observes` entry drives as an in-port (it can be driven), so a coupling's `ports`
are not empty when a leaf chooses its own node;
closedness is a property of the modalities alone (no modality left unfed).

**Per-leaf configuration.** A leaf (a component that is not itself composed) keeps its own planner
configuration on its `Subsystem` record (`config`: `policy_len`, `planner`, `gamma`, `num_iters`,
`decomposition`, `planning_belief`, `feedback`, `mixture_schedule`, `mixture_k`, `epistemic`,
`terms`, `scoring`), and its recognition variables on the record (`recognition`). Its habit,
precision, durations and scoring parameters are `SystemParams.parts[<leaf>]` (a `PartParams`,
present when the leaf holds any). `merge` and `couple` alike state a planner field where their
leaves agree and hold `UNSET` where they differ, and state no habit, precision, durations or
scoring of their own: a composite is not one planning problem until `joint_planner` makes it
one. A composite's own planner field is authoritative wherever it is set (edit it with
`dataclasses.replace`), and the leaf's record speaks where it holds `UNSET`. A merged component
brings the composite's fields as its leaves' configuration. `Subsystem.boundary` is `"coupled"`
for a top-level component of a coupling (`nested` is false). A leaf's view is its own System
namespaced, with its recognition, scoring and configuration.

**The joint planner.** `joint_planner(system, params, node=None, *, inputs=None, **config)`
returns the planning problem of `node` (a component that is not a leaf, or `None` for the whole
System) as one mind: a `(System, SystemParams)` whose faces plan every control node of the node
together.

- *Control identifications collapse.* A driven control node whose driver is an action node
  inside the node leaves the System: every factor that read it reads the driver's control node,
  so one mind plans one decision. A driven node whose driver lies outside the node stays driven
  (give `inputs`, as for `subsystem`). An emission identified with a modality stays an emission
  and an observing slot: the joint model reads its own belief. A kernel in `params.actuator` on
  the driver's action node is refused unless it is the identity (otherwise the intention and the
  value planned on would differ).
- *Configuration.* Each of `policy_len`, `planner`, `gamma`, `num_iters`, `decomposition`,
  `planning_belief`, `feedback`, `mixture_schedule`, `mixture_k` is the argument of
  `joint_planner`, else the composite's value where it states one (for the whole System; a
  field stated on an enclosing composite does not reach into a nested node), else the value the
  node's leaves agree on; leaves that differ are refused, naming the field and each leaf's value
  (`epistemic` and `terms` are also arguments).
- *Habit, precision, durations, scoring.* The composite's own, where it states them. Otherwise
  the habit is the sum in log space of the leaves' habits over the joint policy set (a leaf whose
  own policy set the joint view does not have, a policy module, and a continuous node are
  refused); the precision is the one value the leaves declare (all of them, or none); the
  durations are one clock on one node the planner chooses; the scoring is the one leaf's that
  declares it (two are refused).
- *Recognition* is carried per leaf under the leaf's names.

A node that is itself a leaf is refused (it is its own planner). A leaf that a caller steps as a
process is planned here as a self-model: `joint_planner` does not see the caller's steps.
`joint_planner(merge(...), params)` of a merge plans it exactly as one planning problem, and is
what `run` plans when a step is given to a non-leaf node (`steps={"agent": plan()}`).

**A link into a control node.** `Link(driver, node)` names a control node of one
component (`"world.u_location"`), identified with the action node of another (same step) or
with a state of it (relay). The driver's spelling decides when the driven factors read the
decision:

| `driver` | read | what the composite holds |
|---|---|---|
| an action node of the driving component (`"agent.a_location"`) | at the same step | the control node stays in its component, driven (it carries the driver's realised value, and `action_names` holds `None`); in a joint planner's System it is one node with the driver's: every factor that read it reads the driver's, from t to t+1 on the call that chooses it, and the driven node leaves `controls` and `names.controls` with its action node |
| a state factor of it (`"agent.location"`) | a step later | the one-tick relay: the state becomes a `B_deps` parent of the one factor reading the node, in its place, its control axis re-read as that parent and a no-op axis of size 1 appended |

Level counts must agree (a 5-level node binds a 5-level node, or the state of a
5-level factor), and a continuous pair must agree on width; the driver's box in
`params.control` governs the identified node and the driven node's box is
dropped. A relay from a continuous state is refused. A driver that is neither an
action node nor a state factor is refused, naming both spellings, and a control node as the
driver is refused naming its action node. The factor
spelling of the control node, `Link(..., "world.location")`, is
refused naming its node: `"world.u_location"` replaces it. Bindings compose in
any order: a driven node and its driver's action node are one variable, named by the driver's
action node, so a binding that names the driven node's own action node (a chain written through
it) is refused in either order, naming the driver's action node (one action node driving two
nodes, the join, is the spelling); a node identified with one
a `Clamp` terminates is terminated with it, and a relay into a node that
more than one factor reads is refused whichever binding comes first. A lag on a same-step
read is declared, not configured: an echo state whose table copies a control
node is an ordinary state, and binding from it gives the relay. With an action node as the
driver, the one node in the composite is the driving component's decision, which the driven
component's factors read as a supplied level (a control node supplied by another participant,
`design/nesting-and-coupling.md` N7); the driven component plans no decision for it. A decision is not a belief, so no
belief of the driven component is identified with the driver's, and the binding is
blanket-shaped at the driven component's boundary: the component is nested. With a state
factor as the driver, the relay binds a state's belief into the factor's parent set a step
later and is an identification.

```python
from hybrid_aif.composition import Clamp, Link, joint_planner, merge

composite, params = merge(
    {"agent": agent, "world": world},
    {"agent": agent_params, "world": world_params},
    [
        Link("agent.a_location", "world.u_location"),  # same step
        Link(("world.location",), "agent.location"),  # the modality's parents
    ],
)
composite.names.controls  # ("agent.u_location", "world.u_location"): the world's is driven
mind, mind_params = joint_planner(composite, params)
mind.names.controls  # ("agent.u_location",)
```

**`Clamp(node, level)`** terminates a discrete control node, replacing
its decision with a fixed prior: an `int` level slices the control axis of every
factor that reads the node (a delta), and a `Categorical` or probability vector
averages each reader's table over it (uniform: no one is steering). A reader of
several nodes keeps its other control axes. Native control nodes merge like the
sugar's, and a composite with native nodes is described like any other.

**`Identify(a, b)`** identifies two components' parameter variables (named
tables declared in `SystemParams.parameters`, read through `Parameter("<name>")`) as one:
the composite keeps `a`'s declaration under its namespaced name, every factor, modality or emission
that read `b` reads it, and the two must be the same family with the same table shape
(anything else raises naming both). The identification is one home shared outright by the
components of one particle; `write_back` is bookkeeping of that home between views, not a
crossing. Nothing in one participant clamps another participant's parameters: separate minds
exchange a parameter only as a report of counts through an action node
(`design/nesting-and-coupling.md` N2, N6). Under `run` (below) the write-back happens each
round a view learns and the caller never calls it; `write_back` is for a hand-written loop over
views. `subsystem` views carry the variables their readers read
under the composite's names (so a rate key is `"parameter:<component>.<name>"`, which `merge`
namespaces as it does any key), and `write_back(composite, params, name, before, after)` sets the composite's
variables to what a view learned: `before` is the view's params as `subsystem` returned them and
`after` the same view's params after learning. Each variable `before` holds must equal the
composite's current entry, and is then replaced by `after`'s, decay included. A stale `before`
raises (take the view again from the composite after any write-back), because evidence from several
views at once is accumulated by learning on the composite, not by writing views back. The counts the second
name's component had learned are discarded: only `a`'s declaration survives the binding. A shared
variable's reader can be kept out of its updates with a static `0.0` on the reader's own rate key
(`lr_by_slot={"transition:mission.map": 0.0}`), which is how a scheduler that never observes the
table avoids adding its predictions to the counts.
A binding that rewrites a reader's control axis (a `Clamp`, a relay) refuses a
reader of a shared variable, since the table is not the reader's alone.

**Observing a variable (`System.observes`).** A modality may observe
a variable the System generates, an action node or an emission: `System(...,
emissions={"cell": Emission(("x",), realise="mode", kernel=GRID)}, observes={"cell_obs":
"cell"})` is a System that reads its own position belief as the cell it falls in. The variable
set (state factors, control nodes, action nodes, emissions, and the modalities and control nodes
`observes` does not name) is derived from the declarations. Four things read the map and nothing
else does: `ports` (the slots the map names are no in-ports, and each emission is an out-port
typed by its kernel: the identity's is its parent's, a `Grid`'s is a level per cell), the keyed
faces (`infer_states`, `learn` and `replay` take observations and `present` flags keyed by
variable, fan one datum out to every slot observing it, refuse a key that names a renamed slot,
naming the variable, and take the datum of a variable the System generates as optional, the slot
being absent when it is left out), `describe` and `build` (a likelihood record carries an
`observes` entry where its variable differs from the slot's name, and an emission with a static
kernel is a record of its kernel and realisation under the label `actuator`, with no family) and
the description length (a slot the map names is no variable of its own, so a coupling of `A` and
`B` with `k` identified variables counts `|A| + |B| - k` decision variables, and the level count
of an identified variable is priced once). Nothing computes the datum inside the
faces: a hand loop realises the emissions after a round, `system.emit(fold_in(key, EMIT_TAG),
params, {**beliefs, <control levels>})`, and passes them to the next round's `infer_states` and
`learn`; `replay` does that after each tick and records them (`Replay.emitted`); `run` takes
each from what the step that owns it returned (a `plan()` step realises every emission of its
view from its posterior), as it stands at the reader's turn. `mirror` drops every emission of
the System (an out-port: the mirror models what the System
senses) with the `observes` entries.

**A belief read, `Link(source, modality, realise=..., kernel=...)`,** makes a modality of
one component take its datum from a state factor of another: sensing inside one particle, with
no action node. `realise` is required: a read has no default. It is sugar for a pair: an
`Emission` of the source's component, named by the modality's local name under the source's
component (`rover.cell_obs` for `mission.cell_obs`), and an `observes` entry of the composite
mapping the modality to it (a name the source's component already uses is refused, naming the
explicit form: an `Emission` declared on the source's System and `observes` on the reader's). The
emission is the source's posterior pushed through `kernel` (the identity,
or a `Grid`, the indicator of its cells over a continuous source) and realised by `realise`
(`"sample"`, a draw; `"mode"`, the argmax of a categorical belief or the mean of a continuous
one; `"stats"`, a categorical belief's probability vector, the filter's soft observation). The
modality's own parents explain the datum, as for any observation. `merge` records the read on
the reader's `Subsystem.reads` and carries a component's own `observes` entries too, under its
prefix. `subsystem` leaves the reading component's modality an ordinary observation, and `run`
supplies every read of a stepped node (its own emissions and reads inside it as well) each round:
a read of an emission takes the value the source's step realised from its posterior (a `plan()`
step with `System.emit`, a custom step the same way), a draw keyed by the step's key folded with
`EMIT_TAG`, as it stands at the reader's turn (the round before under `"simultaneous"`, and under
the other orders when the reader steps before its source), and `None` (absent) before the source
has realised one. `plan()` treats a `None` observation as absent.

```python
from hybrid_aif import Grid
from hybrid_aif.composition import Link, merge, nested

composite, params = merge(
    {"rover": rover, "mission": scheduler},
    {"rover": rover_params, "mission": scheduler_params},
    [Link("rover.x", "mission.cell_obs", realise="mode", kernel=Grid(shape=(3, 3), low=0.0, high=3.0))],
)
nested(composite, "mission")  # True: its only binding across the boundary is a sibling read
```

Refused: a modality and a source in one component (a read inside a component is an emission and
`System.observes`); a modality whose level count is not the source's under the identity kernel,
nor the grid's cell count under an indicator; a categorical modality reading a continuous
source with no grid; `"stats"` of a Gaussian source, which is deferred. `nested(composite, name)`
(also `Subsystem.nested`) is true when at least one binding crosses the component's boundary
and every one is a belief read or a link from an action node into a control node; a relay, a
structural link into a modality or a terminal target, or an `Identify` across it makes the
component not nested (`design/nesting-and-coupling.md` N2, N3).

**Names a binding removed.** A node a binding removed is gone from the
composite, and a mapping keyed by it or by its action node (an action, a node
belief, `legal=`, a target) is refused with a sentence naming what it became:
in a joint planner's System, the driver's node; under the relay, the state it reads;
under a `Clamp`, nothing. A target on a node a relay or `Clamp` removed is refused at
`merge()`, one on a node a joint planner collapses at `joint_planner`. The `Subsystem` record
names nodes: `controls` (chosen here), `actions` (each named control node's action node; a driven
node's is the one it had before it was driven), `identified` (`(driven node, driver's node)` for
each action node identified with a control node), `relayed` (`(node, state)`) and `terminated`.

**One clock.** `params.durations` (a round's length per level of one discrete control node)
belongs to a leaf, is `parts[<leaf>].durations` in a composite, and a `subsystem` view copies it.
A round steps every
factor once, so a joint planner has one clock: leaves that declare durations on two nodes are
refused at `joint_planner`, as is a collapse that removes the node carrying them; a merge or a
coupling keeps each leaf's durations in its `parts`
(`design/step-durations.md` T1). A slow component is given its own pace by declaring its durations
under `run` of a closed System (the clocks section below), not by a second clock inside the
composite (`design/nesting-and-coupling.md` N4).

**Legality across a merge.** A joint planner's System has one choice point for an action node
identified with a control node, the driver's, so a mask on the identified node is the driver's
constraint: key it
by the driver's node, where it applies as any node's mask does, on `act`,
`infer_policies`, `replay` and `plan()`. A mask keyed by the driven node
is refused naming the driver's node, and one keyed by a relayed node is refused
naming the relay.

**The views.** `subsystem(composite, params, name, inputs=None)` gives one mind's System. The
driven mind's view has no node for the identified control: its factor's axis is
terminated by a prior over the partner's node, uniform by default, or given in
`inputs` keyed by the action node of the node it is the level of (a relay stub's own control
node too). A driven node of the component itself (`Identify(out_port, node)`) is kept as an
input with no action node unless `inputs` names its driving variable: an `int` slices every table
that reads the node and every carried target conditional on it, a `Categorical` or probability
vector averages them, `None` averages uniformly (a driven continuous node is never sliced, and
a Gaussian target takes a level). A leaf takes its configuration, habit (when the view holds
all of the leaf's control nodes), precision, durations and scoring from its own record; a leaf
that shares no latent with another gets its own System, recognition included, and an amortized
composite's view is refused only where it would carry stubs.
The driver's view keeps its node, and a boundary stub of
the driven mind reads it at the same step. Boxes and targets on a node travel
with the view that chooses it. A view of a nested component is that component's own model; the
prior over the partner's node is the component's belief about its external world inside the
particle (`design/nesting-and-coupling.md` N2).

## Running a closed System: `run`, `plan` and `sample`

A **closed System** is a merge or a coupling in which every modality is fed by an identification
(`System.observes`): nothing is perceived from outside. `with_mirror(system, params)` closes any
System by coupling it to its mirror: each modality `observes` does not name is identified with the
world's emission of that name, each action node with the world's control node it realises,
and the world's copy of a control node the model drives internally by the same action node
(`names=("model", "world")` names the two components; `lower=True` gives the world each learnable
declaration at its posterior mean, or at one draw under `param_key`; `per_component=True` is
the mirror per component, below). `run(system, params, ticks, key, steps, clocks=None,
tick_order="simultaneous", carries=None, record_carries=False)` steps a closed System and
returns a `Trajectory`.

**Steps.** `steps` maps each **stepped node** to its step: a leaf (a component that is not itself
composed), or a non-leaf node, which is then planned as one mind (the step is handed
`joint_planner(system, params, node)`). The keys are an antichain over the System's components
that covers every leaf: naming a node and a leaf beneath it is refused, and naming a node with a
`sample()` leaf beneath it says to step that node's other leaves instead (the joint view would
take the process's tables as part of its model). A step is
`step(system, key, obs, carry, params) -> (out, carry', params')`:

- `system` and `params` are its view, taken from the parameter home each round (the leaf's
  `subsystem` view, or the joint view), and equal views share one compiled program;
- `obs` maps each modality slot, and each control node a variable drives, by the slot's own name
  (`agent.rover.gauge`), to the value of the variable it observes, `None` where that variable has
  no value yet; a `plan()` step is handed its driven discrete control nodes as a view sliced at
  the realised level (`subsystem(..., inputs=)`, averaged uniformly before the driver's first
  realisation), every other step the unsliced view and the input in `obs`;
- `out` maps its out-ports to values: the action nodes of the control nodes it chooses (an
  intended level, or a float vector for a continuous node) and its emissions. A step that omits an
  emission is refused on its first round, naming it; one that omits an action node, or returns a
  key that is no out-port of its node, a level out of range or a value of the wrong shape, is
  refused naming the node;
- the params returned are the very object handed in when the step did not learn. A learner's
  parameter variables are written back to the home (`write_back`, the view as `before`) and the
  next view is taken from the new home; params that differ outside declared parameter variables
  are refused.

`plan(planner=None, learn=None, trace=False, replace=None, warm_start=False, **act_kwargs)` is the
step of a
planning mind: infer from the carried prior (a `None` datum is an absent modality; a modality that
observes a variable of its own view takes that variable's datum), learn when `learn` is set and a
previous round exists (`True`: every learnable slot; a mapping of `System.learn` keywords; a
sequence of slot keys, which learn at rate 1 and the rest at 0), act, push the posterior through
the transition as the next prior and realise every emission of the view from the posterior. Its
carry is a `MindCarry` (`beliefs`, `learn_state`, `previous`, `action`, `trace`, `plan`). Under
`trace=True` the round's `Trace` is in `carry.trace`: `q_pi`, the term `tree`, `policies` (the
enumerated set, or `None`), `plans` (the policy set as the planner returned it: every start's final
plan for the gradient planner, a `ControlStep` for the mixed one), `detail` (the round's
`InferenceDetail`: the log-evidence per modality, the free energy) and `parameters` (the view's
parameter variables the round planned with, after its learning). Under `warm_start=True` the plan
is receding-horizon: `carry.plan` keeps the continuous controls of the plan the round acted on
(the policy of highest `q_pi`, `(policy_len, action_dim)`), and the next round hands the planner
that plan shifted one step on (its steps after the first, the last step repeated) as its `init`;
the first round starts from zeros. The shift is by one of the node's own steps, so a node held
by its durations shifts once per decision. The planner must take an `init` (the mark
`takes_init`: the gradient and mixed planners, made or registered); any other is refused, on
construction for a planner given and on the reset for a view's registered one. A per-part timing
of a round (inference against planning) is not recorded: a round is one compiled program, timed
whole or not at all. `replace={"policy_len": 2}` sets view fields, which on a non-leaf node is also
where the joint view's configuration is given. A view with a driven continuous control node is
refused for `plan()`, naming a custom step.
`sample()` is the step of a process: its first call is `System.sample_initial` (or the partial
point state `carries=` seeds), each round `System.sample_step` on the unsliced view with the
driven control nodes taken from `obs`, returning its emissions and the action node of each control
node nothing drives, drawn from its habit or uniformly. A custom step is any function of that
shape; the view's `emit` realises its emissions (`view.emit(fold_in(key, EMIT_TAG), params,
{**beliefs, <control levels>})`).

**Rounds and variables.** Each variable (an action node or an emission) has one value per round.
An action node is realised from its node's `params.actuator` kernel, a draw at the intended level,
when the node decides, and stands until realised again; an emission is realised by the step that
owns it. A modality, or a driven control node, reads the variable it observes as it stands at its
turn, never a value that does not yet exist: a read of an emission or an action node is the value
realised by the owner's last step, so a reader that steps before its source in the round's order
sees the round before's and one that steps after it sees this round's (under `"sequential"` and
`"async"` the order is the derived one, below).

**Timing.** A round is one tick under every order and every stepped node steps once. A stepped
node whose control node is identified with another stepped node's action node steps after that
action is realised in the round, carrying the realised value. `"simultaneous"` (default) steps the
nodes none of whose control nodes is driven by an action node on the round's boundary (phase one,
their action nodes realised together), then the others in an order that respects the rule (phase
two). A phase-two node reads a control node driven by another node's **action** as realised this
round, and every other slot, a control node driven by an **emission** included, at the boundary,
so what a simultaneous reader sees does not depend on the order the components are listed in.
The same holds for params under `"simultaneous"`: every node's view in a round is taken from the
parameter home as the round began, and the params each learner returns are applied at the round's
boundary, after its last realisation and before the next round's reads. Two nodes of one round
that write back the same parameter variable are combined only where the variable is Dirichlet
counts (a `Dirichlet` or `ConditionalDirichlet`): the round-start concentration plus the sum of
their deltas, the posterior given both nodes' data (a sum that is not positive is refused as
`write_back` refuses one). For any other form, two `plan()` steps whose `learn` both reach the
variable (`learn=True` every learnable slot, a sequence or mapping the slots it names) are
refused before the first round, and two other steps that both wrote it back in one round are
refused at that round's boundary, each naming the variable and the nodes. Under
`"sequential"` and `"async"` a write-back is live: the next node's view is taken from the home
as it then stands.
`"sequential"` steps the listing's sources, the nodes none of whose control nodes an action
drives, in listing order, each reading what stands; each driven node steps as soon as the last of
its drivers has realised, before the sweep returns to nodes made ready earlier (a chain follows its
source, a join waits for its last driver and steps there, nodes made ready by the same
realisation keep listing order), so a driven node's own place in the listing does not matter.
`"async"` is sequential with a fresh random listing of the sources each
round (reproducible from the run key): one total order per round, every node reading what stands
at its turn, each driven node placed by the same derived rule, and no boundary read. Sequential is
a systematic scan, async a random scan, simultaneous a synchronous update. A cycle of
action-to-control identifications is refused. The actuator is applied
only by the node that decides.

**Clocks.** A stepped node whose view declares `params.durations` (one control node of the node
carrying a duration per level, in the unit its preferences are per) decides once per option.
Declaring durations is what holds a node; `clocks` is only the unit conversion. On a decision
round the node steps, and `run` reads the duration of the level it chose
(`view.elapsed(action, view_params)`) times the node's entry of `clocks` (the rounds one unit of
duration lasts, a positive finite number, 1.0 for a node it omits). The product must be a whole
number of rounds of at least one, or `run` refuses it, saying which control node and what the
product was; the node then does not step for that many rounds, the decision round included. On a
held round there is no view, no step call, no write-back and no realisation: the node's action
nodes and emissions keep the values they were realised with, the variables it drives keep their
level, and the node's own carry is untouched, so `carries[t].trace` of a held `plan(trace=True)`
node reads the decision's plan. The first round always decides, and `carries=` seeds a step,
never a count. `clocks` names stepped nodes that declare durations. An action node of a held node
may not carry a non-identity actuator kernel (the node decides once, so one draw would stand for
the whole option); an identity kernel is accepted. Durations on a control node that another
node drives are refused (the node chooses no option); a mirror carries none. A joint planner
over leaves whose durations differ is refused at `joint_planner`, while leaves in different nodes
keep one clock each.
`Trajectory.decided` is a bool per round and node: `False` on a round the node spent inside its
option, with `received` the values its slots held, `intended` the held action and `emitted` the
standing values.

```python
# the survey's two-tier agent: the scheduler decides once per option, three rounds long
rover = plan(make_gradient_planner(steps=40, starts=4), learn={"lr": 1.0}, trace=True,
             warm_start=True)
steps = {"agent.mission": plan(trace=True), "agent.rover": rover, "world": world_step}
traj = run(system, params, 400, key, steps, clocks={"agent.mission": 3 / SURVEY_TICKS})
traj["agent.mission"].decided      # True on rounds 0, 3, 6, ...
```

**The first call is the reset.** Before round 0 every step is called once with `obs=None`, in node
order, with its seeded carry (`carries=`) or `None`, and the keys `k_reset` for the first node and
`fold_in(k_reset, j)` after. What it returns stands entering round 0 and nothing is written back.
`plan()` returns nothing and leaves the declared initial priors as the first prior; `sample()`
returns the emissions of the initial state and the state as its carry; a custom step returns
`{}, carry, params`.

**Keys.** `key, k_reset = split(key)`; each round `key, k_tick = split(key)` and `k_tick` splits in
`2N + 1` for `N` stepped nodes: node `j` steps with key `j`, its action nodes are realised with
key `N + j` (the last key for the nodes of phase one under `"simultaneous"`), and node `i` of the
`A` action nodes of the System (those of driven control nodes excluded) draws its realisation from
`split(fold_in(k, ACTUATOR_TAG), A)[i]`; an emission draws from the step key folded with
`EMIT_TAG`.

**The record.** A `Trajectory` is a tree with one node per component. The root holds `params`
(the home after the last round); every node holds `parts` (its components, by name relative to
the node) and, as mappings keyed relative to the node (a non-leaf node's are the prefixed union of
its parts'): `received` and `present` (what each modality slot and driven node held for its step,
`ticks + 1` entries, the last what the slots hold after the last round; an absent value is zeros),
`emitted` (the realised action nodes and emissions, `ticks` entries), `intended` (the action
nodes as chosen), `start` (the out-ports holding a value entering round 0), `carries` (the carry
after each round, `ticks + 1` entries, when `record_carries=True`), `carry` (after the last
round) and
`decided` (a bool per round: whether the node stepped, which a node with durations does not on a
held round). `trajectory["agent.rover"]` walks the tree by dotted path, and a
jointly planned node is a leaf of the tree.

```python
from hybrid_aif.composition import plan, run, sample, with_mirror

system, params = with_mirror(model, model_params)       # the model and the world it models
traj = run(system, params, 100, key, {"model": plan(), "world": sample()})
traj["world"].emitted["gauge"]      # what the world emitted, one entry per round
traj["model"].carry.beliefs         # the mind's carried prior
```

**The mirror per component.** `mirror(system, params, per_component=True)` takes a merged or
coupled System and returns the *merge* of one mirror per component, a leaf each, named as the
component is (`a.hand` stays `a.hand`), so that a parameter variable several components read stays
one variable and each component is stepped on its own by `run`. A component's own state factors,
transitions, priors, control nodes and modalities are mirrored as the whole System's are (every
modality an emission of the component, its likelihood the kernel), with these differences, and the
mirror carries no durations (a world is stepped every round).

- A state factor of **another component** that one of the component's transitions reads (the
  arena's `lock_b` reads `b.hand`) becomes a **driven control node** of the reading component,
  `u_<owner>_<factor>`: in a tabular table the parent's axis moves behind the others, and the node
  is fed by an **identity emission** of the factor's owner, `<factor>_read`,
  `Emission((factor,), "mode", "identity")` (the mode of a point is the point, so the value is
  exact and draws nothing). An emission-driven control node is read at the round's boundary under
  `"simultaneous"` and as it stands under `"sequential"`, which is how the components of the System
  read each other's states: the previous round's state under simultaneous, and under sequential
  the state as the owner left it when the owner steps before the reader in the round's order. In
  the arena that order is `model.a, world.a, model.b, world.b, model.arena, world.arena`: the
  sources in listing order, each world leaf right after the model leaf that drives it.
- A **modality whose parents are all states of one other component** becomes an emission of that
  component, named `<reader>_<modality>`: an emission is the state showing itself, and the state is
  the owner's, so it is drawn from the owner's state as the owner's step leaves it, as the System
  observes it. `with_mirror` identifies the model's modality with it.
- A modality with parents in **two or more components** (the `status` of one robot reads both
  robots' cells) stays an emission of its own component, with each foreign parent fed through the
  driven node above. **Known limitation**: a foreign factor gets one driven node, and a transition
  needs the factor at the boundary while a modality needs it after the owner has stepped, so such a
  modality reads the boundary value under `"simultaneous"` (and under `"sequential"` and
  `"async"` when the owner steps after the reader in the round's order): one round older than the
  System observes it. Per-use driven
  nodes are on the roadmap.
- A control node that one component's action node drives inside the System is driven in the mirror
  by the same action node, so a leaf sampled alone follows its driver.

`with_mirror(system, params, per_component=True)` couples the System with this merge in place of
the mirror of the whole, each component of the world a leaf that `run` steps with `sample()`.
Under `"sequential"` each mind steps and then the world leaf it drives, so the next mind reads the
world as that move left it this round, which is how a merged System steps its components: the
arena's minds are bitwise to the composite form of `run` in release 0.19.0 under
`"simultaneous"` and `"sequential"`. Under `"async"` the arena is not comparable to that form,
which permuted the minds only and stepped the arena's factors at a fixed turn, while `run`
permutes every source, the arena's leaf included, and places each world leaf after its driver.

```python
def idle(system, key, obs, carry, params):
    return {}, carry, params          # a leaf that chooses and emits nothing

system, params = with_mirror(arena, arena_params, per_component=True)
steps = {"model.a": plan(), "model.b": plan(), "model.arena": idle,
         "world.a": sample(), "world.b": sample(), "world.arena": sample()}
traj = run(system, params, 100, key, steps)
```

Refused, with the reason and what to do: a foreign state parent of a neural, nonlinear or
deterministic transition (its parents go through a map that is code); a foreign discrete parent of
a linear-Gaussian transition (a continuous factor reads its discrete parents within the step, which
the one driven node cannot serve); a foreign continuous state parent (a continuous control node
needs a bounded box, and cross-component continuous parents are not built); a foreign parent of a
transition that reads a parameter variable (the axis would move in a table every reader shares); a
foreign `D_deps` parent; a habit on a component; a node shared by components (the collapse of
`joint_planner`; mirror the merge); a modality that reads another component's action node;
components that are themselves composed (the whole mirror takes those); and a System with no
components (mirror it whole).

## External processes: a black box

An environment, or a world whose dynamics are hand-written code, enters a coupling as one node of
the factor graph whose inside is not stated: an opaque map from its in-ports and its own carried
state to its out-ports and its next state. `black_box(ins=..., outs=...)`
(`hybrid_aif.composition`) declares it by its ports alone, each a name and a type, `(LEVEL,
levels)` or `(FLOAT, width)` (`hybrid_aif.controls.LEVEL` and `FLOAT`), and returns a real
`(System, SystemParams)`: one control node per in-port, with no action node and no reader, one
emission per out-port, with no parents and the static kernel `"black_box"`, and
`System.black_box` holding every port's type, the mark every face reads. It holds no state
factor, no table and no control or modality kernel; the params are empty.

```python
from hybrid_aif.composition import Identify, black_box, couple, plan, run
from hybrid_aif.controls import FLOAT, LEVEL

world = black_box(
    ins={"v": (FLOAT, 2)},
    outs={"fix": (FLOAT, 2), "img": (FLOAT, 2304), "gauge": (LEVEL, 4)},
)
system, params = couple(
    {"agent": agent, "world": world},
    [
        Identify("agent.rover.a_v", "world.v"),       # an action node drives the in-port
        Identify("world.fix", "agent.rover.fix"),     # an out-port feeds a modality
        Identify("world.img", "agent.rover.img"),
        Identify("world.gauge", "agent.rover.gauge"),
    ],
)
traj = run(system, params, 400, key, {"agent": plan(), "world": world_step},
           record_carries=True)
traj["world"].carries                                 # the state world_step carried, per round
```

`couple` binds its ports with `Identify` exactly as it binds any component's: the types of the two
ends must agree, an in-port is driven at most once, and an in-port nothing drives leaves the
coupling open, which `run` refuses naming the in-port. A `Clamp` on an in-port is refused (it would
remove the node; drive it from a component that emits the constant). Under `run` the black box is
a leaf stepped by a custom step, the step contract of every leaf: `obs` holds the in-ports by the
view's names (`None` on the reset call), `out` every out-port, and the carry is whatever state the
step keeps; the reset call, the timing rules, the keys and the `Trajectory` (`received`,
`emitted`, `start`, `carries`) are those of any leaf. `env_step(env, params=None, *, actions=None)`
is that step for a positional (pymdp-style) environment: one out-port per observation in the
environment's order, level in-ports filling a level action at their factors (`actions` maps an
in-port to its factor, in-port `k` at factor `k` by default), float in-ports concatenated into the
continuous action; on the same keys it steps the environment as `GenerativeEnv.wrap` does, without
a modelling System to name it.

Refused, each with a sentence that says what to do, because each needs what lies between the
ports: `plan()` and `sample()` as the black box's step, a step marked `sliced` on it, a node
planned jointly with a black box beneath it (`joint_planner` over it too), `mirror` and
`with_mirror` of a System that is or holds one, `as_env`, `sample_initial`, `sample_step`,
`emit`, the inference, planning and learning faces of such a System, and a `subsystem` view of
the black box sliced at a driven level (its unsliced view, its ports under the coupling's names,
is what `run` hands its step). `describe` writes it under `relations.black_box`, its ports with
their types and no records (the step is code, so it is not part of the document);
`build` rebuilds the same System. In its description length the ports are priced as the declared
structure they are (each out-port an emission of the graph code with its size code, an in-port
nothing drives a control node), and the inside is unscored: it is listed under
`black_box:<component>` with every link `None`, its four `(address, link)` pairs are in
`DescriptionLength.unscored`, nothing of it enters `structure`, `parameters` or `total`, and
`exact` is `False`.

**A process outside the computation graph.** A black box's step may be arbitrary host Python:
real hardware (an agent hooked to a camera), a socket, a simulator in another language. `run`
calls a custom step once per round from its own Python loop and never traces, compiles, batches
or differentiates it; the step may do I/O, keep any Python object as its carry (no JAX pytree is
asked of it), and return NumPy arrays or Python numbers, which `run` checks against the port's
declared type and stores in one dtype: a level port takes one integer level in range (a Python or
NumPy integer of any width, stored as `int32`) and refuses a float or a vector; a float port takes
a vector of its width (an integer vector converted to the library's float dtype, and one number
accepted for width 1); a refusal names the port and its type. `run` stores the carry object a
step returns as it is (`record_carries=True` keeps a reference per round, not a copy), so a step
whose carry is recorded returns a new object each round rather than mutating the old one. Where the user's code
owns the timing instead (a control loop paced by the hardware), the agent's own faces
(`infer_states`, `act`, `learn`) are called from that loop and `run` is not used. What does not
work: compiling, batching (`vmap`) or differentiating through the external process, and
wall-clock scheduling under `run`, whose round is a logical tick (the step blocks the loop for as
long as it takes).

**When to declare a black box, and when to open it.** Structure is taken out of the box only where
it pays.

- *A world run forward as a model.* When the world is a model you mean to sample, run its mirror
  of, or compare with the agent's own model, state it as a System of factors and kernels and step
  it with `sample()` (`with_mirror` builds one from the agent's model). A black box is for a
  process whose dynamics are code you will not state: an environment from elsewhere, a
  hand-written simulator.
- *A component another mind models.* A planner that plans jointly over a node, or a mind whose
  model includes the process, needs its inside. A black box is coupled to minds, never planned
  over; open it where some mind must model it.
- *Shared variables.* A black box shares nothing but its ports: no parameter variable
  (`Identify` of two parameters), no structural `Link`, no belief read into it. Two components that
  must share a latent or a parameter are both stated.
- *Pricing a world.* A black box has no description length the library computes: its floor is a
  code length for its program, which is not built, so it is reported unscored and left out of the
  totals. To compare worlds by description length, state them.
- *Partial opening.* Open the part that pays and keep the rest in the box: declare the part to be
  sampled, priced or modelled (a gauge as a table, say) as a System of its own, keep the
  hand-written remainder a black box, and couple the two through ports, the box reading the
  opened part's emissions on in-ports of its own.

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
