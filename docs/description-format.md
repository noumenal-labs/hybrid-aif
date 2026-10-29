# The model description format, version 0.3

A *model description* is a JSON document that states a hybrid-aif model completely: its
variables, the conditional distributions (factors) over them, the parameters those factors take,
the objective a planner scores, and the settings of the System that runs it. The library reads a
document with `build` and writes one with `describe`:

```python
from hybrid_aif import build, describe
from hybrid_aif.description import load, save

system, params = build(document, arrays=arrays, key=key)   # arrays and key only when needed
document, arrays = describe(system, params)                 # arrays: {} when all inline
save(describe(system, params), "model")                     # model.json (+ model.npz)
document, arrays = load("model.json")
```

`build` checks the whole document before it constructs anything. Every fault is reported as one
sentence that names where the fault is (`factors['cue']['likelihood']`, `clamps['B_location']`,
`variables['x']`) and what to write instead. `schema()` returns the JSON Schema (2020-12) of the
format, checked into the library as `hybrid_aif/description/description-0.3.json`.

This page is the whole specification. Section 12 holds four complete documents that build as
written; reading one of them first is a good way in.

## 1. The top level

| key | required | holds |
|---|---|---|
| `hybrid_aif_description` | yes | the format version, `"0.3"` |
| `library_version` | no | the library version that wrote the document (informational) |
| `variables` | yes | every variable, with its domain and size (section 2) |
| `factors` | no | the factor records, keyed by child variable and label (section 3) |
| `relations` | no | `controls`, `observed`, `external` (section 5) |
| `clamps` | no | the values of the fixed parameter variables (section 6) |
| `objective` | no | the terms the planner scores; absent means `{"decomposition": "utility_infogain"}` (section 8) |
| `configuration` | yes | planning, inference and learning settings (section 9) |
| `arrays_file` | no | the name of the arrays file the clamps' refs point into (section 6) |
| `extensions` | no | a consumer's own data, keyed by the consumer's name; never read |

Any other top-level key is refused, and the sentence points at `extensions`: an object keyed by
consumer name whose values are objects (`{"my_tool": {...}}`). The library carries it untouched.

A document whose `hybrid_aif_description` is an earlier version (description 0.2, the layout
with `structure` and `declarations` sections) is refused with a sentence: hybrid-aif 0.18.0 is the
last release that reads it, and building it there and `describe`-ing the result writes it in this
format. `describe` writes this format only.

A document carries data and never code. A transition written as code, a
`NonlinearGaussianTransition` (`x' = f(x, controls) + ε`, `f` an equinox module), is written by
the name its `f`'s class is registered under and the values of `f`'s fields (section 4.4); `build`
instantiates only transition classes the reading process registered, and imports nothing for
them. A `Deterministic` map is the exception: its `"impl": "package.module:name"` is imported at
build time (section 4.3), and importing a module runs its top-level code.

## 2. Variables

`variables` is an object keyed by variable name. Every variable the document mentions anywhere is
declared here, once. A variable record is a *domain* and a *size*, nothing else.

**Names.** One namespace holds every variable: a latent state, an observation, a control node, an
action node and a parameter may not share a name. A name may not contain `[` or `]` (they write
a parent's step, section 3.3). The character `.` is reserved for composites (section 7).

**Order.** The order of `variables` is the order of the model's nodes (the latents' order, the
control nodes' order, the observations' order). Reordering the keys changes the node layout of
the same model, never the model. Same-step reads run in this order (section 3.5).

### 2.1 Per-step variables

A per-step variable takes a value at every step: a latent state, an observation, a control node,
an action node, an auxiliary variable of recognition, an external input.

| domain | size fields | meaning |
|---|---|---|
| `categorical` | `levels` | one of `levels` values |
| `categorical` | `levels`, `shape: [k]` | `k` independent categoricals of `levels` values each (a factored code, read as its flattened one-hot of width `k × levels`) |
| `real` | `dim` | a vector of `dim` numbers |
| `real` | `dim`, `dims` | a latent's vector of `dim` numbers with a name for each dimension, in order: the names `epistemic` selects a dimension by (`"on": "pose:b1"`). Written only when the System declares them (`System(dims=...)`); without it the dimensions are `dim0`, `dim1`, … by position. Unique, one per dimension, without `:`, `.`, `*` or brackets; `dimN` names position `N` only |
| `real` | `dim`, `low`, `high` | a continuous control node with its box: `low` and `high` are a number per dimension, or one number for every dimension |
| `categorical` | `levels`, `duration` | a discrete control node that sets the round's clock: `duration` is a positive number per level, or one positive number for every level, in the unit the preferences are per; written only when declared, on at most one node (`design/step-durations.md` T1) |

Only a control node carries `low`/`high`, and a continuous control node must (its box is its
support). A record carries the size fields of its domain only: `levels` with `categorical`, `dim`
(and the bounds) with a per-step `real`, `shape` with a parameter or a categorical code. An action node has its control node's domain and size; a mismatch is refused naming
both.

### 2.2 Parameter variables

A parameter variable is static: one value for the whole run, named in some factor's
`parameters`. Its record is a domain and a `shape`:

```json
"B_location": {"domain": "log_simplex", "shape": [3, 3, 3]}
```

| domain | values | checked at build |
|---|---|---|
| `log_simplex` | normalised log-probabilities along the distribution's axis (`-inf` allowed for an impossible entry) | no NaN; entries ≤ 0; each distribution's exponentials sum to one within 1e-3 |
| `simplex` | probabilities along the distribution's axis (`0` for an impossible entry), lowered to their logs on read | no NaN; entries ≥ 0; each distribution sums to one within 1e-3 (a distribution of zeros, like one of `-inf`, is impossible entries) |
| `logits` | unnormalised log-probabilities | no NaN |
| `positive` | strictly positive numbers | > 0 everywhere |
| `nonnegative` | finite numbers ≥ 0; a Dirichlet concentration with a zero, a level on the simplex's face (a structural zero, or a dead level of a growable factor) | finite, ≥ 0 everywhere |
| `cholesky` | lower-triangular matrices with a positive diagonal | triangular, diagonal > 0 |
| `spd` | symmetric positive-definite matrices | symmetric, eigenvalues > 0 |
| `unit_interval` | numbers in [0, 1] | no field takes it |
| `integer_levels` | non-negative integers | no field takes it |
| `real` | any numbers | nothing |

The domain of a parameter is fixed by the field that names it (the tables in section 4 give it);
`build` refuses a declared domain that disagrees. A hand-written document may omit `domain` on a
parameter variable; the field's domain is then used. Two allowances: a field whose domain is
`logits` accepts a variable declared `log_simplex` (normalised logits are logits), and a
`ConditionalCategorical` table accepts `logits` (the family normalises over the child axis;
`describe` writes `logits` for a table that is not normalised). A field whose domain is
`log_simplex` or `logits` also accepts a variable declared `simplex`: its clamp is written as
probabilities and `build` takes their logs, so a hand-written table need not be normalised in
log space. `describe` always writes `log_simplex`, the engine's own representation. No family
field in section 4 takes `unit_interval` or `integer_levels`, so a variable declared with either
is refused as a mismatch.

### 2.3 Tree variables

A network's weights are one *tree variable*: a record that is a tree of parameter records, one
leaf per array.

```json
"embed_recognition_layers": {
  "layers": [
    {"weight": {"domain": "real", "shape": [16, 2]}, "bias": {"domain": "real", "shape": [16]}},
    {"weight": {"domain": "real", "shape": [6, 16]}, "bias": {"domain": "real", "shape": [6]}}
  ],
  "norms": [
    {"weight": {"domain": "real", "shape": [16]}, "bias": {"domain": "real", "shape": [16]}},
    {"weight": {"domain": "real", "shape": [6]}, "bias": {"domain": "real", "shape": [6]}}
  ]
}
```

A mapping that contains any of the record keys (`domain`, `levels`, `dim`, `low`, `high`,
`shape`) is a leaf; any other mapping or list is a tree node. The trees each network family
holds are listed in section 4.3. A tree variable whose clamp is an `init` recipe (section 6) may
be declared as `{"domain": "real"}` alone: the recipe fixes its leaves.

### 2.4 Per-step or static: derived from use

No record says whether a variable is per-step or static. A variable named in any factor's or
term's `parameters`, or the child of a `prior` record, is static. A variable that is the child or
a parent of any other record, or appears in `relations`, is per-step. A variable used both ways,
or used nowhere, is refused.

## 3. Factors

`factors` is keyed by child variable; under each child, by a *label* naming one conditional on
that child:

```json
"factors": {
  "location": {
    "initial":    {"family": "Categorical", "parameters": {"logits": "D_location"}},
    "transition": {"family": "ConditionalCategorical",
                   "parents": ["location", "u_location"],
                   "parameters": {"log_table": "B_location"}}
  }
}
```

### 3.1 The record

| field | holds |
|---|---|
| `family` | the family's class name (section 4) |
| `parents` | the per-step variables the factor conditions on, in order. Order matters: a table's axes and a network's inputs follow it |
| `parameters` | the factor's static arguments, keyed by the family's field names, each value a parameter variable's name; a field that is itself a module is a nested record of names (`"innovations": {"loc": ..., "chol": ...}`) |
| `options` | the family's configuration (activation, head, loss, ...); absent options take the family's defaults |
| `steps` | the steps the record applies at: `"0"`, `"1+"` or `"all"` (section 3.4) |

Layout is never written: the number of parent axes, the parents' widths and the child's size are
derived from `parents` and the variables' domains, and every array's shape is checked against
them.

### 3.2 Labels

A label is either a *role label* or a *generic label*. A role label implies the record's time
structure and the engine slot it fills; a generic label is the factor's own name, and the record
states its time structure itself.

| label | partition | child | parents read at | applies at | fills |
|---|---|---|---|---|---|
| `initial` | generative | a latent | the same step | the first step only (`"0"`) | the initial-state prior; its parents are the latents it is conditional on (earlier in variable order) |
| `transition` | generative | a latent | the step before | every step after the first (`"1+"`) | the transition |
| `likelihood` | generative | an observed variable | the same step | every step (`"all"`) | the observation model; parents are latents or action nodes |
| `actuator` | generative | an action node | the same step | every step | the kernel `P(a \| u)` of a stochastic actuator; its one parent is the action node's control node. Omitted when the action realises the decision exactly |
| `prior` | generative | a parameter variable | — | — | the prior the parameter is learned under (section 4.2) |
| `policy_prior` | generative | the first control node | — | plan time | E: one distribution over the joint policy set of every control node, across the horizon |
| `target` | objective | a latent, an observation or a control node | the same step | every step of a plan | the preference the planner scores: C under an observation, the risk target under a latent, a cost under a control node; its parents, if any, are discrete control nodes or latents it is conditional on (a conditional family, one distribution per configuration) |
| `terminal` | objective | a latent or a control node | the same step (the plan's end) | the plan's end | the target scored at the plan's end; its parents are latents it is conditional on |
| `recognition` | q-side | a latent, an auxiliary variable, or the first control node | the same step | every step | an amortized posterior (section 3.6) |

The partition matters: a `target` or `terminal` record is a factor of the *objective*, read by
the planner's scoring and never by inference; a `recognition` record is a factor of the
approximate posterior *q*. The reader never lowers one partition into another.

A record under a role label may write `steps` only as the label's own value; `steps` does not
apply to `target`, `terminal`, `policy_prior` and `prior` records. A `target`'s `parents` are
its conditioning variables, a discrete control node or a state factor, read at the same step (the
engine resolves the target at each step of a plan under them; `design/step-durations.md` T5). A
`prior` and a `policy_prior` take no parents.

Under a generic label, the record's child and its lowered time structure decide the slot: a latent
child with `steps: "0"` is its initial record, with `steps: "1+"` its transition; an observed
child with `steps: "all"` and same-step parents its likelihood; an action node's its actuator.
Each latent has exactly one initial and one transition record, under whichever labels.

### 3.3 Brackets: the step a parent is read at

A parent's offset is written on its name, `t` being the child's step: `x[t]` reads `x` at the
child's step, `x[t-1]` at the step before. A bare name means the offset the label implies. Under
a role label, parents are bare except where one overrides the label: the switching transition
reads its own state a step back and the mode at its own step,

```json
"transition": {"family": "LinearGaussianTransition", "parents": ["x", "mode[t]"], ...}
```

Under a generic label every parent is bracketed; a bare name is refused. Any bracket other than
`[t]` and `[t-1]` is refused.

### 3.4 Steps

`steps` is `"0"` (the first step only), `"1+"` (every step from the second on) or `"all"`; it is
`"all"` when absent. A record that applies at the first step (`"0"` or `"all"`) and reads a parent
at `t-1` is refused, naming `"1+"` as the fix: the first step has no step before it.

### 3.5 What a transition reads

Each transition family reads its parents in its own way, and `build` refuses a parent at an offset
the family cannot read:

| family | its own previous value | other latents | control nodes |
|---|---|---|---|
| `ConditionalCategorical` | optional, `x` (t-1), an ordinary table axis | at t-1 (bare) | at t-1 (bare) |
| `LinearGaussianTransition` | required, listed first (the `F x` term) | discrete mode parents at t, written `mode[t]` | at t-1: a discrete node is a mode axis, a continuous node enters `G u` |
| `NonlinearGaussianTransition` | required, listed first (`f`'s `x`) | none | at t-1, `f`'s `controls` in order: a discrete node as its level (and a leading axis of the innovations), a continuous node as its value |
| `MLPConditional` | required, listed first | none | at t-1, read as inputs |
| `Deterministic` (a carry) | required, listed first (the carry) | at t-1 | at t-1 |
| `MLPCategorical` (a code) | not allowed | at t, written `deter[t]` | at t-1 |

A control node's decision moves its readers at the next step: a transition reads a control node
at the step it is chosen, the step before the child. A transition reads the control node, never
its action node: an action node is read by likelihoods and by its own actuator record, and a
transition naming it is refused naming the control node to write. Same-step reads among latents
(an `initial`'s parents, a transition's `[t]` parents) run in variable order, so a parent is
declared before its child; a cycle is refused, and an order that puts a parent after its child is
refused naming a valid order.

A `LinearGaussianTransition` with a discrete mode parent, like a `MixtureGaussian` initial,
makes its latent mixture-carried, and `configuration.inference.mixture_k` (section 9) is then
required; `build` refuses the document without it.

### 3.6 Recognition records

A `recognition` record is an amortized posterior: `q` over its child given its parents, a network.
In a recognition record a latent parent denotes its prediction at that step (the prior its
generative records produce), an observed parent its data, an external input its data. Parents are
read at the child's step.

- A latent may have a recognition record; a parent of one may not be a latent that has its own
  (no amortized posterior reads another).
- An *auxiliary variable*, such as an observation embedding, is a per-step variable whose only
  record is a recognition one; other recognition records may read it. The evaluation order is
  derived from the parents (an auxiliary variable is inferred before what reads it); a cycle is
  refused.
- An external input (section 5) is read by recognition records only.
- A recognition record under the first control node is the *policy module*: a network from the
  beliefs to an action distribution, which the `habit` planner acts from. Its parents are every
  latent, in variable order (it reads them concatenated).

## 4. Families

The keys of `parameters` are the family's field names. Every family below lists its parameter
keys with the domain of each, its options with their defaults, and the labels it appears under.

### 4.1 Distributions and conditionals

| family | parameters (domain) | labels | notes |
|---|---|---|---|
| `Categorical` | `logits` (logits), shape `[levels]` | initial, target, terminal, policy_prior | unnormalised log-probabilities; under `policy_prior` the shape is `[P]`, P the number of policies |
| `ConditionalCategorical` | `log_table` (log_simplex) | initial, transition, likelihood, actuator, target, terminal, generic | a table, child axis first, then one axis per parent in order: shape `[child levels, *parent levels]`. A transition that reads no control node may carry one more trailing axis of size 1. Under `target`, a preference conditional on the record's parents (discrete control nodes or latents) |
| `JointCategorical` | `logits` (logits) | target, terminal | a categorical over several axes |
| `Gaussian` | `loc` (real) `[d]`, `chol` (cholesky) `[d, d]` | initial, target, terminal | |
| `ConditionalGaussian` | `loc` (real) `[*levels, d]`, `chol` (cholesky) `[*levels, d, d]`, or `[d, d]` shared by every configuration | target | a Gaussian per configuration of the record's parents, one leading axis per parent in order; the parents are discrete control nodes (a latent parent, a mixture per step, is refused) |
| `MixtureGaussian` | `log_weights` (log_simplex) `[K]`, `means` (real) `[K, d]`, `chols` (cholesky) `[K, d, d]` | initial, target, terminal | a mixture-carried latent needs `inference.mixture_k` |
| `LinearGaussianTransition` | `F` (real), `innovations`: `{loc (real), chol (cholesky)}`, `G` (real, optional) | transition | `x' = F x + G u + b + ε`. `innovations` has one leading axis per discrete parent (states and discrete control nodes, in order), or one axis of size 1 for none, on `loc` and on `chol`, or on `loc` alone with a `chol` of `[d, d]` shared by every mode; `F` and `G` are shared `[d, d]` / `[d, du]` or per mode with the innovations' leading axes. `G` is omitted when the factor reads no continuous control node |
| `NonlinearGaussianTransition` | `f` (a tree variable, optional), `innovations`: `{loc (real), chol (cholesky)}`; options `f` (required) and `propagation` (`"first_order"`) | transition | `x' = f(x, controls) + ε`, `f` a registered transition function (section 4.4). `innovations` has one leading axis per discrete control node, in order, or one axis of size 1 for none |
| `LinearGaussianObservation` | `head` (real) `[d_obs, d+1]`, `noise` (cholesky) `[d_obs, d_obs]`; or `noise` as a map, `{"values": <variable>}` (cholesky) `[*grid, d_obs, d_obs]` with option `noise_map: {low, high, axes}` | likelihood | `y = H x + c + ε`, the head being `[H \| c]`; parents: the observed continuous latent, then categorical mode parents (leading axes of head and noise). Either parameter may be learned (section 4.2). A noise map is `R_chol` at the nodes of a grid over the latent's coordinates `axes`, spanning `low` to `high` per axis, read by interpolation at the belief's mean; a clamp, on a non-switching modality. With categorical parents only the head is `[*levels, d_obs, 1]`, the offset column: a Gaussian emission over them, `y = c_j + ε_j` (`design/structure-learning.md` S2) |
| `DerivedInitial` | none; option `reduction` (`"mode"`) | initial | a code's initial value read off its own transition at the first step |
| `LearnedVector` | `value` (real) `[dim]`; option `activation` (`"tanh"`) | initial | a carry's episode-start value, trainable |

### 4.2 Learned parameters and their priors

A parameter variable with a clamp is fixed; one with a `prior` record under it is learned, its
hyperparameters clamped parameter variables of their own. A parameter with neither, or both, is
refused.

| prior family | parameters (domain) | learns | lowers to |
|---|---|---|---|
| `Dirichlet` | `concentration` (positive, or nonnegative where a level has a zero), `alpha0` (positive, optional), `growth` (positive scalar, optional) | a `ConditionalCategorical`'s `log_table` or a `Categorical`'s `logits` | a Dirichlet over the table's columns (or the categorical's probabilities); `concentration` has the table's shape; `alpha0` is the anchor a decay relaxes toward, the table's shape or a scalar (a uniform table); `growth` is the growth propensity of a growable factor, written on its weights (the transition column of a memoryless latent) and on its initial prior, whose dead levels are zeros (`design/structure-learning.md` S3, S6); a zero without `growth` is a structural zero, a level that stays absent until data lands on it |
| `InverseGamma` | `alpha`, `beta`, `alpha0`, `beta0` (positive; the last two optional) | a `LinearGaussianObservation`'s `noise` | a conjugate prior on the noise variance |
| `InverseWishart` | `Psi` (spd), `nu` (positive), `Psi0`, `nu0` (optional) | a `LinearGaussianObservation`'s `noise` | a conjugate prior on the noise covariance |
| `MatrixNormal` | `M` (real), `V_inv` (spd), `M0`, `V_inv0` (optional), `row_cov` (a variable name, optional) | a `LinearGaussianObservation`'s `head` | a matrix-normal prior on the head |

**The matrix-normal-inverse-Wishart pair.** The joint prior on a linear-Gaussian observation's
head and noise, `R ~ IW(Psi, nu)` and `W | R ~ MN(M, R, V)`, is written as two priors: an
`InverseWishart` under the noise variable, and a `MatrixNormal` under the head whose `row_cov`
names the noise variable. `build` recognises the pair and lowers it to the joint family. With the
noise clamped (or under an inverse-gamma prior), the head's record alone is the standalone
matrix-normal head. `row_cov`, when given, must name the factor's own noise variable.

The hyperparameters are clamped: a document is the model at one moment, so a clamp means "given
with this document", never "constant". Learning moves the hyperparameters' values; the posterior
counts are the next document's clamps. A hyperparameter with a prior of its own is refused.

### 4.3 Networks

A network's weights are one tree variable, named under the family's network field, with its
configuration in `options`. Its layout (input widths, output width) comes from the wiring: the
parents' widths in order (a categorical parent as its one-hot, a code as its flattened one-hot) and
the child's size.

| family | network field | tree | options (defaults) | labels |
|---|---|---|---|---|
| `MLPConditional` | `layers` | `{"layers": [linear, ...], "norms": [norm, ...]}` | `activation` (`"tanh"`), `min_scale` (0.001), `head` (`"softplus"` or `"bounded"`), `minstd` (0.1), `maxstd` (1.0), `norm` (false) | transition, likelihood, recognition |
| `MLPPredictor` | `layers` | as above | `activation` (`"silu"`), `norm` (true), `out_activation` (false), `symlog_inputs` (false) | recognition (an auxiliary variable's encoder) |
| `MLPCategorical` | `trunk` | as above | `unimix` (0.01), `activation` (`"silu"`), `norm` (true), `out_activation` (false), `symlog_inputs` (false) | transition (a code), recognition |
| `PredictorHead` | `trunk` | as above | `loss` (required: `"symlog_mse"`, `"symlog_disc"` or `"binary"`), `num_bins` (255), `bin_low` (-20.0), `bin_high` (20.0), `activation` (`"silu"`), `norm` (true), `out_activation` (false), `symlog_inputs` (false) | likelihood |
| `Deterministic` | `map` | `{"input_proj": linear, "gate": linear, "norms": [norm, norm]}` | `rule` (`"carry"`), `map` (`"GRUCell"`), `activation` (`"silu"`), `norm` (true) | transition (a carry) |

A `linear` is `{"weight": [out, in], "bias": [out]}` (no `bias` when the layer has none); a `norm`
is `{"weight": [w], "bias": [w]}`. Each layer reads the one before (its `in` is the previous
`out`). `norm` says whether the tree carries norms, one per activated layer: an
`MLPConditional`'s hidden layers (every layer but the last); a trunk's (`MLPPredictor`,
`MLPCategorical`, `PredictorHead`) hidden layers, and its last layer too when `out_activation` is
true; a GRU's two (the input projection and the gate). The outputs:
`MLPConditional` emits a Gaussian over its child (`2 × child size` numbers); `MLPPredictor` the
child's vector; `MLPCategorical` the code's logits (`k × levels`); `PredictorHead` the child's
vector under `symlog_mse`, `num_bins` logits under `symlog_disc`, one logit under `binary`.

A `Deterministic` over a named function (no weights) writes no parameters and
`"options": {"rule": "carry", "map": {"impl": "package.module:name"}}`; the function is imported at
build and refused if it does not import. A lambda cannot be described.

### 4.4 Registered transition functions

A `NonlinearGaussianTransition`'s mean map `f` is code. The process that writes or reads a
document registers the class of `f` under a namespaced name,
`register_transition("<package>.<name>", cls)` (at least one dot, each segment an identifier),
and the record names it:

```json
"transition": {
  "family": "NonlinearGaussianTransition",
  "parents": ["pose", "u_pose"],
  "parameters": {
    "f": "pose_transition_f",
    "innovations": {"loc": "pose_transition_innovations_loc",
                    "chol": "pose_transition_innovations_chol"}
  },
  "options": {"f": {"name": "drive.unicycle", "static": {"mode": "rear"}},
              "propagation": "unscented"}
}
```

with `"pose_transition_f": {"h": {"domain": "real", "shape": []}, "rates": {"domain": "real",
"shape": [3]}}` under `variables` and its values under `clamps`.

- `options.f` is `{"name": ..., "static": {...}}`: the registered name, and `f`'s static fields
  (`eqx.field(static=True)`) as JSON values (`null`, a boolean, a number, a string, or a list of
  these, read back as a tuple); `static` is absent when `f` has none.
- `parameters.f` is a tree variable keyed by `f`'s leaf fields, one real leaf each, clamped in
  the usual forms (inline, or a ref into the arrays file); it takes no prior and no `init`
  recipe, and is absent when `f` has no leaves. A leaf is a floating-point value.
- `propagation` is `"first_order"` or `"unscented"`.

`build` looks the name up and refuses an unknown one, naming `register_transition`. It compares
the document's field names, leaf and static, with the fields the registered class declares and
refuses a document whose fields are missing, extra, or swapped between leaf and static, naming
them: a document written for another version of `f` fails rather than rebuilding with code that
means something else. A class does not fix its leaves' shapes; they are checked once `f` is
rebuilt, by validating the System (`f` must return the latent's dimension for the factor's
control layout). `describe` refuses an `f` whose class is not registered (a subclass of a
registered class included), an integer or boolean leaf, and a static field of any other kind,
each naming the field.

## 5. Relations

`relations` holds the relations no single factor owns.

| key | value | meaning |
|---|---|---|
| `controls` | `{control node: action node}` | the System chooses each key and emits it as the paired action node. A control node is known only by being a key here and by having no initial or transition; an action node only by being a value and by its optional actuator record |
| `observed` | `[observation, ...]` | the variables the System receives data for at every step; each has a likelihood record |
| `external` | `[input, ...]` | vectors the System receives as data with no likelihood (domain `real` with `dim`), read only by recognition records |

A variable plays at most one of these roles. Every per-step variable that is in none of them and
has an initial and a transition record is a latent. A control node and its action node are named
freely; `describe` writes `u_f` and `a_f` for the control node a latent `f` declares with the
pymdp-style keywords. An action node's only record is its actuator: a variable with any other
record cannot be the value of `controls`.

One limit: a model on the amortized path (one with recognition records or the network families
`Deterministic`, `MLPCategorical`, `PredictorHead`, `LearnedVector`, `DerivedInitial`) reads its
control nodes by name. There, the control node is `u_<f>` after the latent `f` whose transition
reads it, and its action node `a_<f>`; any other pair is refused, naming the pair to write.

## 6. Clamps

`clamps` is keyed by parameter variable. A clamp is one of:

- **an inline value**: a number or a nested list of numbers in the variable's shape. A single
  number broadcasts to the declared shape (`"A_cue_concentration": 1.0`). Inline values take the
  build's default dtype (float32, or float64 when the process enables 64-bit). `-Infinity` is
  accepted for an impossible entry of a log table, and a table declared `simplex` (section 2.2)
  is written as probabilities, `0` for an impossible entry;
- **a ref**, `{"ref": "<file>#<key>"}`: the array stored under `<key>` in the arrays file, with its
  own dtype. The document names the file in `arrays_file`; `build(document, arrays=...)` takes
  the file's contents as a mapping from key to array, and a ref whose key is missing is refused
  naming the file;
- **an init recipe**, `{"init": {...}}`: a network's values drawn at build with
  `build(..., key=...)`; without a key it is refused, naming the variable. A learned vector's
  `{"init": {}}` is its zero vector and needs no key. The recipe gives what the draw needs beyond
  the wiring and the options, and a missing required argument is refused:

  | family | init arguments |
  |---|---|
  | `MLPConditional` | `width`, `depth`, `init_scale`, `init` |
  | `MLPPredictor`, `PredictorHead` | `width` (required), `depth`, `out_scale`, `init` |
  | `MLPCategorical` | `width` (required), `depth`, `init` |
  | `Deterministic` (GRU) | `hidden` (required), `init` |
  | `LearnedVector` | none (`{"init": {}}` is the zero vector) |

  `init` is `"default"` or `"trunc_normal_avg"`. If the variable's record lists its leaves, the
  drawn shapes are checked against them.

A tree variable's clamp is a tree of clamps in the variable's tree: each leaf inline, a ref (one
arrays-file key per leaf), or the whole tree one `init` recipe.

`describe(system, params, arrays="auto", inline_max=256)` writes an array inline when it has at
most `inline_max` elements and as a ref otherwise; `arrays="inline"` and `arrays="file"` force one
placement. It writes a constant array as the number it broadcasts from (except under `"file"`).
`save` writes `<name>.json` and, when there are refs, `<name>.npz` beside it, pointing every ref
and `arrays_file` at it; `load` reads both.

## 7. Composites

A model `merge()` composed from components is written flat. Its variables carry the components'
names as prefixes, `<component>.<name>` (`world.location`, `agent.u_location`), and the reserved
`.` is what makes a document a composite: every per-step variable of a document with dotted names
belongs to a component, and a document that mixes dotted and plain per-step names is refused. A
nested composite has nested prefixes (`outer.inner.x`). The default parameter names keep the
child's prefix (`world.B_location`), and a dotted parameter belongs to its factor's component: a
factor naming a parameter under another component's prefix is refused.

The bindings of the merge are already structure, and nothing records them:

- a **bound modality** is a likelihood whose parent lies under another prefix
  (`agent.seen` reading `world.location`);
- an **identified control node** (one component's decision driving another's factor at the same
  step) is the driver's control node in the parent list of every factor that read the identified
  one; the identified node and its action node are absent;
- a **relay** (a control node identified with another component's state, read a step later) is
  that state as a transition parent at the step before;
- a **target emission** is a terminal target's parent under another prefix;
- a **terminated** control node (fixed by a boundary prior) is simply absent, its readers' tables
  averaged or sliced over its axis.

`build` reconstructs each component's record from the prefixes, in the document's variable
order: its latents and observations, the control nodes under its prefix still present with their
action nodes, its bound modalities (those with a parent under another prefix) and its target
emissions. That record is what `run` routes observations and actions by, and what `subsystem()`
derives a component's view from.

## 8. The objective

```json
"objective": {
  "decomposition": "utility_infogain",
  "terms": [{"term": "predicted_reward", "on": "reward"}]
}
```

The objective is a list of `terms`. A term record is
`{"term": <name>, "on": <variable>, "parameters": {...}, "options": {...}, "model": {...}}`, every
field but `term` optional:

- `term` is a registered term name;
- `on` names the variable the term is scored on; the registered term says what of it (the
  transition of a latent, the channel of an observation). A term that targets variables and is
  written without `on` applies to every eligible variable; a term with no target takes no `on`.
  The partial selections are written as patterns: for `infogain`, `"*"` (every latent),
  `"obs:*"` (every observation) and `"*:*"` (every learned parameter); for a term that reads a
  latent's transition, `"transition:*"`. `infogain` may also name dimensions of a real latent,
  `"<latent>:<dim>"` (`"pose:b1"`, or `"pose:dim3"` by position; `"pose:*"` is the whole
  latent), and is then scored on those dimensions with the others marginalised;
- `parameters` names the term's own parameter variables (a network is one tree variable, an
  ensemble a list of them), and `options` gives, for each, its `family`, its options and its
  layout (`parent_dims`, and `child_dim` or `out_dim`), since a term parameter has no wiring to
  derive them from;
- `model` nests a whole document (its `variables` to `configuration` sections) for a term that
  scores against a model of its own. `{"ref": "<path>"}` in its place names a document file,
  resolved relative to this one by `load`.

A term's sign is part of the term: +1 enters the expected free energy G as a cost, −1 as a value.
The document never writes a sign or a weight; the policy precision (`planning.gamma`, or the
parameter `planning.precision` names) is the only scale.

| term | measures | `on` | sign |
|---|---|---|---|
| `utility` | the expected log-preference of the predicted observations under each observation's target | none: every observation with a `target` | −1 |
| `infogain` | information gain: about a latent through the channels that observe it, about everything an observation reveals, or about a learned parameter (novelty) | a latent, an observation, a learned parameter; absent: every eligible variable | −1 |
| `risk` | the divergence of the predicted observations from each observation's target; an observation without a target carries an implicit flat preference, its risk `−H[q(o)]` (the preference's normaliser dropped) | none: every observation | +1 |
| `ambiguity` | the expected conditional entropy `E H[o \| S]` of each channel the `infogain` terms select, given the selected latents `S`; no channel carries it without an `infogain` term | none: the channels the `infogain` terms select | +1 |
| `observation_interaction` | where several linear-Gaussian channels observe one Gaussian latent, what a per-channel sum miscounts: under `utility` with `infogain` the interaction information `TC(Y) − TC(Y \| x_S)` of the selected channels' readings about the selected state; under `risk` with `ambiguity` the joint ambiguity's `−TC(Y \| x_S)` when a strict sub-block of the latent is selected | not written: the pair mounts it | +1 |
| `observation_correlation` | under `risk` with `ambiguity`, where several linear-Gaussian channels observe one Gaussian latent, the total correlation `TC(Y)` of their readings, the part of the joint risk a per-channel sum leaves out, whatever the selection | not written: the pair mounts it | +1 |
| `occupancy` | the expected entropy of a latent's transition, `E_q H[B(·\|s, u)]` | a latent (its transition) | −1 |
| `uncertainty` | the entropy of a continuous latent's predicted belief, `H[q(x)]` (`H[q(x_S)]` on selected dimensions; a Kolchinsky–Tracey pair on a mixture), as a cost | a continuous latent, `<latent>`, or one of its dimensions, `<latent>:<dim>` (records on one latent merge into one sub-block); absent: every continuous latent | +1 |
| `predicted_reward` | a reward head's decoded prediction | an observation with a one-number `PredictorHead` | −1 |
| `g_tail` | a learned critic's value of the final step, closing the horizon | none; parameters `critic` and, optionally, `slow` (its slowly updated copy) | −1 |
| `model_uncertainty` | an ensemble's disagreement about the next code, as a cost | a code latent; parameter `ensemble` | +1 |
| `cosine_sim_to_pref_rollout` | the cosine similarity of the predicted trajectory to a preference model's imagined one (registered by `import hybrid_aif.reference`) | none; `model`: the preference model | −1 |

Two scores are implied by records rather than listed: a `target` under a latent or a control node
is scored as the divergence of its predicted belief from the target (`risk`, +1) under either
pair, and a `terminal` record as `terminal_risk` (+1) at the plan's end. A `terminal_risk` term in
`terms` is refused, pointing at the `terminal` record that scores it.

**The two pairs.** The expected free energy is written as one of two decompositions, and the pair
is a choice of terms: `utility` with `infogain`, or `risk` with `ambiguity`. The utility form
mounts a utility leaf per targeted observation and an information-gain leaf per selected
variable; the risk form a risk leaf per observation (a flat preference's `−H[q(o)]` where none is
targeted) and an ambiguity leaf on each selected channel. A mix the term tree cannot build
(`utility` with `ambiguity`, or `risk` without `ambiguity`) is refused, naming the two pairs. `infogain` beside `risk` and `ambiguity` selects the
learned parameters scored for novelty and the channels that carry ambiguity, each given the
latents (or dimensions) selected.

**The sugar.** `"decomposition": "utility_infogain"` (or `"risk_ambiguity"`) is the standard pair
with every eligible variable; further terms are listed beside it in `terms`. Writing the pair's
own terms beside `decomposition` is refused: to choose the variables, write the pair as terms. An
objective listing only, say, `occupancy` scores only occupancy. An unknown term name is refused,
naming the registered terms; a term's `parameters` are checked against the names the term
declares.

`infogain` on a code latent takes the parameters of the ensemble estimator,
`{"ensemble": [...]}`, as `model_uncertainty` does.

## 9. Configuration

`configuration` holds what a System stores beyond the model, in three sub-blocks. Every field maps
to one System setting; an absent sub-block or field takes the default.

| sub-block | field | default | meaning |
|---|---|---|---|
| `planning` | `horizon` | required | the plan length |
| | `planner` | `"enumeration"` | the planner by registry name: `"enumeration"`, `"sophisticated"`, `"cem"`, `"gradient"`, `"habit"` |
| | `gamma` | 16.0 | the policy precision, a static setting of the System |
| | `precision` | absent | the name of a parameter variable holding the policy precision as a parameter (`SystemParams.precision`): a clamped variable of domain `positive` and shape `[]`. Present, it is the precision the planners read in place of `gamma`, and a parameter flows traced through the faces, so changing its value recompiles nothing; absent, `gamma` is the precision |
| | `branch_on` | `[]` | the observations the `"sophisticated"` planner branches on |
| | `belief` | `"prior"` | the belief a plan's information gain reads on a Gaussian-carried latent: `"prior"` (predict only) or `"posterior"` (conditioned on the observations the plan expects) |
| | `feedback` | absent | `"linear_quadratic"` scores Gaussian targets in closed loop (`System.feedback`); absent, open loop |
| `inference` | `iterations` | 16 | fixed-point iterations of state inference |
| | `mixture_k` | null | the shared mixture budget, required when a latent is mixture-carried |
| | `mixture_schedule` | `"runnalls"` | `"runnalls"` or `"mode_aligned"` |
| `learning` | `frozen` | `[]` | the network and learned-vector variables a trainer does not move |
| | `reduction_period` | null | every how many `learn` calls Bayesian model reduction runs on the growable latents (null: on explicit `reduce` calls only) |
| | `reduction_pairs` | 8 | the candidate pairs a reduction tests |

`build` checks each value: `horizon`, `iterations` and `reduction_pairs` are positive integers,
`mixture_k` and `reduction_period` null or a positive integer, `gamma` a number, `precision` the
name of a declared variable of domain `positive` and shape `[]`, clamped (a prior record on it is
refused: the engine holds the precision fixed), `planner` a
registered planner, `mixture_schedule` one of its two values, and `branch_on` observed variables,
given only with the `"sophisticated"` planner. `belief` is `"prior"` or `"posterior"`, the latter
refused beside `branch_on`; `feedback` is `"linear_quadratic"`, given only with `belief`
`"posterior"`. `describe` writes `belief` and `feedback` only when they differ from the default.
A document records everything needed to rebuild the agent's behaviour, planning options
included, so they live here beside the planner, the horizon and the precision; what the
description length prices is decided by the generative partition (section 3.2), and this section
is not priced. A control node is priced as a node, and a `duration` is not priced, as the bounds of a
continuous node's box are not, nor is the `precision` parameter: the numbers are not the
model's structure.

`learning.frozen` names the weights of networks and learned vectors. Naming a conjugate parameter
(a parameter learned under a prior, or a hyperparameter) is refused: the engine holds a
conjugate parameter fixed only by a learning rate of zero passed to `learn`.

**Not in the document**, by design: a planner's own settings (candidate counts, step counts) and
learning rates and decay, which are arguments to `act` and `learn`; and numeric precision, which
is the process's 64-bit flag.

## 10. Default names

`describe` names the parameter variables it writes; the names are the user's to change, and
nothing parses them.

- a tabular record under a role label uses the active-inference letters: `A_<child>` (likelihood),
  `B_<child>` (transition), `C_<child>` (target), `D_<child>` (initial), `H_<child>` (terminal),
  `K_<child>` (actuator), and `E` for the policy prior;
- the policy precision is `precision`;
- a prior's hyperparameter is `<parameter>_<field>` (`A_cue_concentration`);
- everything else is `<child>_<label>_<field>`, nested fields joined the same way
  (`x_transition_innovations_loc`); a network is `<child>_<label>_<network field>`
  (`embed_recognition_layers`), and a nonlinear transition's `f` is `<child>_transition_f`; a term's parameter `<term>_<parameter>` (`g_tail_critic`), with
  `_<i>` per ensemble member;
- in a composite, the name keeps the child's prefix (`world.B_location`).

## 11. What build refuses

Every fault is one sentence naming the address, what was found and what to write instead, and
a refusal carries one of them, or several when records that are checked independently of one
another fail together: the variable records, the factor records, the clamps' pairing with their
variables, and, once the document is read, the values each factor's parameters take. Each of
these checks runs to its end and the refusal lists every failing record on its own line ("the
document has 3 faults:" ...), so a document is corrected in one pass; a variable several factors
read appears under each of them. A fault in an earlier check hides the later ones, which depend
on it. Among the faults: an unknown top-level key, family, option, parameter field, relation, configuration
field or term; a variable used both per-step and static, or nowhere; a parameter with neither a
clamp nor a prior, or both; a parent bracket other than `[t]`/`[t-1]`, a bare parent under a
generic label, a first-step record reading `t-1`; a parent at an offset its family cannot read;
two initial or two transition records on one latent; a same-step cycle or misordering; an array
whose shape disagrees with its variable or with the wiring (tables are child-first, one axis per
parent in order); a domain check that fails; an action node whose size differs from its control
node's; an observed variable with no likelihood; an external input read by a non-recognition
record; a recognition record reading another amortized posterior, or recognition records in a
cycle; a policy prior whose length disagrees with the policy set (the sentence names the horizon
and the control nodes' levels), or a second policy prior; a dotted name outside a composite; a
transition function name no class is registered under, or whose class declares other fields than
the document writes (section 4.4).

Two are records the format admits and the engine does not yet run, refused with a pointer to
the gap: a `prior` under a control node (a per-step prior on a decision), and a conjugate
parameter in `learning.frozen`.

## 12. Worked examples

Each of these documents builds as written. `describe` writes 12.1, 12.2 and 12.4 back as written
(numbers at float32 precision). 12.3 draws its networks from `init` recipes, so `describe` writes
the drawn weights as values or refs, lists each network's leaves in `variables`, and writes every
option with its default filled in; describing that document again gives it unchanged.

### 12.1 A tabular model with a stochastic actuator and a learned likelihood

Two latents (where the agent is, which context holds), one control node with a noisy actuator,
two observations. The cue's likelihood is learned under a Dirichlet prior; everything else is
clamped.

```json
{
  "hybrid_aif_description": "0.3",
  "library_version": "0.11.0",
  "variables": {
    "location": {"domain": "categorical", "levels": 3},
    "context": {"domain": "categorical", "levels": 2},
    "u_location": {"domain": "categorical", "levels": 3},
    "a_location": {"domain": "categorical", "levels": 3},
    "position": {"domain": "categorical", "levels": 3},
    "cue": {"domain": "categorical", "levels": 2},
    "D_location": {"domain": "logits", "shape": [3]},
    "B_location": {"domain": "log_simplex", "shape": [3, 3, 3]},
    "D_context": {"domain": "logits", "shape": [2]},
    "B_context": {"domain": "log_simplex", "shape": [2, 2]},
    "A_position": {"domain": "log_simplex", "shape": [3, 3]},
    "A_cue": {"domain": "log_simplex", "shape": [2, 2]},
    "A_cue_concentration": {"domain": "positive", "shape": [2, 2]},
    "K_a_location": {"domain": "log_simplex", "shape": [3, 3]}
  },
  "factors": {
    "location": {
      "initial": {"family": "Categorical", "parameters": {"logits": "D_location"}},
      "transition": {
        "family": "ConditionalCategorical",
        "parents": ["location", "u_location"],
        "parameters": {"log_table": "B_location"}
      }
    },
    "context": {
      "initial": {"family": "Categorical", "parameters": {"logits": "D_context"}},
      "transition": {
        "family": "ConditionalCategorical",
        "parents": ["context"],
        "parameters": {"log_table": "B_context"}
      }
    },
    "a_location": {
      "actuator": {
        "family": "ConditionalCategorical",
        "parents": ["u_location"],
        "parameters": {"log_table": "K_a_location"}
      }
    },
    "position": {
      "likelihood": {
        "family": "ConditionalCategorical",
        "parents": ["location"],
        "parameters": {"log_table": "A_position"}
      }
    },
    "cue": {
      "likelihood": {
        "family": "ConditionalCategorical",
        "parents": ["context"],
        "parameters": {"log_table": "A_cue"}
      }
    },
    "A_cue": {
      "prior": {"family": "Dirichlet", "parameters": {"concentration": "A_cue_concentration"}}
    }
  },
  "relations": {
    "controls": {"u_location": "a_location"},
    "observed": ["position", "cue"]
  },
  "clamps": {
    "D_location": 0.0,
    "B_location": [
      [[-0.02, -4.605, -4.605], [-0.02, -4.605, -4.605], [-0.02, -4.605, -4.605]],
      [[-4.605, -0.02, -4.605], [-4.605, -0.02, -4.605], [-4.605, -0.02, -4.605]],
      [[-4.605, -4.605, -0.02], [-4.605, -4.605, -0.02], [-4.605, -4.605, -0.02]]
    ],
    "D_context": 0.0,
    "B_context": [[-0.01, -4.605], [-4.605, -0.01]],
    "A_position": [[-0.02, -4.605, -4.605], [-4.605, -0.02, -4.605], [-4.605, -4.605, -0.02]],
    "A_cue_concentration": 1.0,
    "K_a_location": [
      [-0.10536051565782628, -2.995732273553991, -2.995732273553991],
      [-2.995732273553991, -0.10536051565782628, -2.995732273553991],
      [-2.995732273553991, -2.995732273553991, -0.10536051565782628]
    ]
  },
  "objective": {"decomposition": "utility_infogain"},
  "configuration": {"planning": {"planner": "enumeration", "horizon": 2, "gamma": 16.0}}
}
```

### 12.2 A hybrid switching model

A discrete mode chosen through a control node, and a continuous latent `x` whose linear-Gaussian
dynamics switch with the mode, read at the same step (`mode[t]`). `x` is observed through a
linear-Gaussian sensor and has a Gaussian target.

```json
{
  "hybrid_aif_description": "0.3",
  "library_version": "0.11.0",
  "variables": {
    "mode": {"domain": "categorical", "levels": 2},
    "x": {"domain": "real", "dim": 1},
    "u_mode": {"domain": "categorical", "levels": 3},
    "a_mode": {"domain": "categorical", "levels": 3},
    "y": {"domain": "real", "dim": 1},
    "D_mode": {"domain": "logits", "shape": [2]},
    "B_mode": {"domain": "log_simplex", "shape": [2, 3]},
    "x_initial_log_weights": {"domain": "log_simplex", "shape": [2]},
    "x_initial_means": {"domain": "real", "shape": [2, 1]},
    "x_initial_chols": {"domain": "cholesky", "shape": [2, 1, 1]},
    "x_transition_F": {"domain": "real", "shape": [2, 1, 1]},
    "x_transition_innovations_loc": {"domain": "real", "shape": [2, 1]},
    "x_transition_innovations_chol": {"domain": "cholesky", "shape": [2, 1, 1]},
    "y_likelihood_head": {"domain": "real", "shape": [1, 2]},
    "y_likelihood_noise": {"domain": "cholesky", "shape": [1, 1]},
    "x_target_loc": {"domain": "real", "shape": [1]},
    "x_target_chol": {"domain": "cholesky", "shape": [1, 1]}
  },
  "factors": {
    "mode": {
      "initial": {"family": "Categorical", "parameters": {"logits": "D_mode"}},
      "transition": {
        "family": "ConditionalCategorical",
        "parents": ["u_mode"],
        "parameters": {"log_table": "B_mode"}
      }
    },
    "x": {
      "initial": {
        "family": "MixtureGaussian",
        "parameters": {
          "log_weights": "x_initial_log_weights",
          "means": "x_initial_means",
          "chols": "x_initial_chols"
        }
      },
      "transition": {
        "family": "LinearGaussianTransition",
        "parents": ["x", "mode[t]"],
        "parameters": {
          "F": "x_transition_F",
          "innovations": {
            "loc": "x_transition_innovations_loc",
            "chol": "x_transition_innovations_chol"
          }
        }
      },
      "target": {
        "family": "Gaussian",
        "parameters": {"loc": "x_target_loc", "chol": "x_target_chol"}
      }
    },
    "y": {
      "likelihood": {
        "family": "LinearGaussianObservation",
        "parents": ["x"],
        "parameters": {"head": "y_likelihood_head", "noise": "y_likelihood_noise"}
      }
    }
  },
  "relations": {
    "controls": {"u_mode": "a_mode"},
    "observed": ["y"]
  },
  "clamps": {
    "D_mode": 0.0,
    "B_mode": [[-0.693, -0.083, -2.526], [-0.693, -2.526, -0.083]],
    "x_initial_log_weights": -0.693,
    "x_initial_means": [[-1.0], [1.0]],
    "x_initial_chols": 0.6,
    "x_transition_F": 0.9,
    "x_transition_innovations_loc": [[-1.2], [1.2]],
    "x_transition_innovations_chol": [[[0.3]], [[0.45]]],
    "y_likelihood_head": [[1.0, 0.0]],
    "y_likelihood_noise": [[0.5]],
    "x_target_loc": [3.0],
    "x_target_chol": [[0.8]]
  },
  "objective": {"decomposition": "utility_infogain"},
  "configuration": {
    "planning": {"planner": "enumeration", "horizon": 3, "gamma": 16.0},
    "inference": {"iterations": 16, "mixture_k": 4, "mixture_schedule": "runnalls"}
  }
}
```

The same transition under a generic label states its whole time structure: every parent
bracketed, the steps explicit. This block in place of the `x` factors above builds the same model.

```json
{
  "x": {
    "initial": {
      "family": "MixtureGaussian",
      "parameters": {
        "log_weights": "x_initial_log_weights",
        "means": "x_initial_means",
        "chols": "x_initial_chols"
      }
    },
    "switching": {
      "family": "LinearGaussianTransition",
      "parents": ["x[t-1]", "mode[t]"],
      "steps": "1+",
      "parameters": {
        "F": "x_transition_F",
        "innovations": {
          "loc": "x_transition_innovations_loc",
          "chol": "x_transition_innovations_chol"
        }
      }
    },
    "target": {
      "family": "Gaussian",
      "parameters": {"loc": "x_target_loc", "chol": "x_target_chol"}
    }
  }
}
```

### 12.3 An amortized model

A recurrent world model: a deterministic carry `deter` (a GRU), a factored code `stoch` (two
categoricals of four classes) read off the carry at the same step, a continuous control node with
its box, a vector observation and a reward head. `embed` is an auxiliary variable: an encoder of
the observation that the code's recognition record reads. Every network is drawn by an `init`
recipe (`build(document, key=...)`). The objective scores the predicted reward and closes the
horizon with a critic. The control node is named `u_deter` and its action `a_deter`, after the
carry whose transition reads it: on the amortized path the names are fixed (section 5).

```json
{
  "hybrid_aif_description": "0.3",
  "library_version": "0.11.0",
  "variables": {
    "deter": {"domain": "real", "dim": 8},
    "stoch": {"domain": "categorical", "levels": 4, "shape": [2]},
    "u_deter": {"domain": "real", "dim": 1, "low": -1.0, "high": 1.0},
    "a_deter": {"domain": "real", "dim": 1},
    "state": {"domain": "real", "dim": 2},
    "reward": {"domain": "real", "dim": 1},
    "embed": {"domain": "real", "dim": 6},
    "deter_initial_value": {"domain": "real", "shape": [8]},
    "deter_transition_map": {"domain": "real"},
    "stoch_transition_trunk": {"domain": "real"},
    "state_likelihood_trunk": {"domain": "real"},
    "reward_likelihood_trunk": {"domain": "real"},
    "stoch_recognition_trunk": {"domain": "real"},
    "embed_recognition_layers": {"domain": "real"},
    "g_tail_critic": {"domain": "real"}
  },
  "factors": {
    "deter": {
      "initial": {
        "family": "LearnedVector",
        "parameters": {"value": "deter_initial_value"},
        "options": {"activation": "tanh"}
      },
      "transition": {
        "family": "Deterministic",
        "parents": ["deter", "stoch", "u_deter"],
        "parameters": {"map": "deter_transition_map"},
        "options": {"rule": "carry", "map": "GRUCell", "activation": "silu", "norm": true}
      }
    },
    "stoch": {
      "initial": {"family": "DerivedInitial", "options": {"reduction": "mode"}},
      "transition": {
        "family": "MLPCategorical",
        "parents": ["deter[t]"],
        "parameters": {"trunk": "stoch_transition_trunk"},
        "options": {"unimix": 0.01, "activation": "silu", "norm": true}
      },
      "recognition": {
        "family": "MLPCategorical",
        "parents": ["deter", "embed"],
        "parameters": {"trunk": "stoch_recognition_trunk"},
        "options": {"unimix": 0.01, "activation": "silu", "norm": true}
      }
    },
    "state": {
      "likelihood": {
        "family": "PredictorHead",
        "parents": ["deter", "stoch"],
        "parameters": {"trunk": "state_likelihood_trunk"},
        "options": {"loss": "symlog_mse", "activation": "silu", "norm": true}
      }
    },
    "reward": {
      "likelihood": {
        "family": "PredictorHead",
        "parents": ["deter", "stoch"],
        "parameters": {"trunk": "reward_likelihood_trunk"},
        "options": {"loss": "symlog_disc", "num_bins": 21, "activation": "silu", "norm": true}
      }
    },
    "embed": {
      "recognition": {
        "family": "MLPPredictor",
        "parents": ["state"],
        "parameters": {"layers": "embed_recognition_layers"},
        "options": {"activation": "silu", "norm": true, "out_activation": true, "symlog_inputs": true}
      }
    }
  },
  "relations": {
    "controls": {"u_deter": "a_deter"},
    "observed": ["state", "reward"]
  },
  "clamps": {
    "deter_initial_value": 0.0,
    "deter_transition_map": {"init": {"hidden": 8}},
    "stoch_transition_trunk": {"init": {"width": 16, "depth": 1}},
    "state_likelihood_trunk": {"init": {"width": 16, "depth": 1}},
    "reward_likelihood_trunk": {"init": {"width": 16, "depth": 1, "out_scale": 0.0}},
    "stoch_recognition_trunk": {"init": {"width": 16, "depth": 1}},
    "embed_recognition_layers": {"init": {"width": 16, "depth": 1}},
    "g_tail_critic": {"init": {"width": 16, "depth": 1, "out_scale": 0.0}}
  },
  "objective": {
    "terms": [
      {"term": "predicted_reward", "on": "reward"},
      {
        "term": "g_tail",
        "parameters": {"critic": "g_tail_critic"},
        "options": {
          "critic": {"family": "PredictorHead", "loss": "symlog_disc", "num_bins": 21, "parent_dims": [16]}
        }
      }
    ]
  },
  "configuration": {"planning": {"planner": "cem", "horizon": 4, "gamma": 16.0}}
}
```

### 12.4 A merged composite

An agent and a world, merged so that the agent's control node drives the world's location at
the same step and the agent's observation reads the world's location. The world's own control
node is absent; the agent's observation `agent.seen` is a bound modality.

```json
{
  "hybrid_aif_description": "0.3",
  "library_version": "0.11.0",
  "variables": {
    "agent.location": {"domain": "categorical", "levels": 3},
    "world.location": {"domain": "categorical", "levels": 3},
    "agent.u_location": {"domain": "categorical", "levels": 3},
    "agent.a_location": {"domain": "categorical", "levels": 3},
    "agent.seen": {"domain": "categorical", "levels": 3},
    "agent.D_location": {"domain": "logits", "shape": [3]},
    "agent.B_location": {"domain": "log_simplex", "shape": [3, 3, 3]},
    "world.D_location": {"domain": "logits", "shape": [3]},
    "world.B_location": {"domain": "log_simplex", "shape": [3, 3, 3]},
    "agent.A_seen": {"domain": "log_simplex", "shape": [3, 3]},
    "agent.C_seen": {"domain": "logits", "shape": [3]}
  },
  "factors": {
    "agent.location": {
      "initial": {"family": "Categorical", "parameters": {"logits": "agent.D_location"}},
      "transition": {
        "family": "ConditionalCategorical",
        "parents": ["agent.location", "agent.u_location"],
        "parameters": {"log_table": "agent.B_location"}
      }
    },
    "world.location": {
      "initial": {"family": "Categorical", "parameters": {"logits": "world.D_location"}},
      "transition": {
        "family": "ConditionalCategorical",
        "parents": ["world.location", "agent.u_location"],
        "parameters": {"log_table": "world.B_location"}
      }
    },
    "agent.seen": {
      "likelihood": {
        "family": "ConditionalCategorical",
        "parents": ["world.location"],
        "parameters": {"log_table": "agent.A_seen"}
      },
      "target": {"family": "Categorical", "parameters": {"logits": "agent.C_seen"}}
    }
  },
  "relations": {
    "controls": {"agent.u_location": "agent.a_location"},
    "observed": ["agent.seen"]
  },
  "clamps": {
    "agent.D_location": 0.0,
    "agent.B_location": [
      [[-0.02, -4.605, -4.605], [-0.02, -4.605, -4.605], [-0.02, -4.605, -4.605]],
      [[-4.605, -0.02, -4.605], [-4.605, -0.02, -4.605], [-4.605, -0.02, -4.605]],
      [[-4.605, -4.605, -0.02], [-4.605, -4.605, -0.02], [-4.605, -4.605, -0.02]]
    ],
    "world.D_location": 0.0,
    "world.B_location": [
      [[-0.02, -4.605, -4.605], [-0.02, -4.605, -4.605], [-0.02, -4.605, -4.605]],
      [[-4.605, -0.02, -4.605], [-4.605, -0.02, -4.605], [-4.605, -0.02, -4.605]],
      [[-4.605, -4.605, -0.02], [-4.605, -4.605, -0.02], [-4.605, -4.605, -0.02]]
    ],
    "agent.A_seen": [[-0.02, -4.605, -4.605], [-4.605, -0.02, -4.605], [-4.605, -4.605, -0.02]],
    "agent.C_seen": [0.0, 0.0, 2.0]
  },
  "objective": {"decomposition": "utility_infogain"},
  "configuration": {"planning": {"planner": "enumeration", "horizon": 2, "gamma": 16.0}}
}
```
