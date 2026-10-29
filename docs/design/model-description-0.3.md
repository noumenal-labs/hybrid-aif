# Model description 0.3: variables and factors

*Status: ADJUDICATED 2026-10-05 (Alex + Opus); not built. D1–D8 settled (D3 as restated; D4's
evidence clamps deferred, its role proposal adopted; D5 tentative; D8's name a first pass). Amends `model-description.md` (M2,
M3, M4) for the document format only; the runtime `System`/`SystemParams` are unchanged except
where `control-states.md` changes them. To ship in the same round as the control-states build,
so consumers meet one format revision, not two.*

## Why

Reading zeroth-harness's serialised models (2026-10-05), Alex found the top-level split hard to
parse. M3 calls it mechanical: `structure` is every static `System` field verbatim and
`declarations` every `SystemParams` slot by address, divided by Equinox's static/array
partition. So the layers mirror how the runtime stores things, not modelling concepts:
`structure` declares *variables* (states, observations), `declarations` declares *factors*
(families, parents) and carries parameter *values*, which are in fact clamps on parameter
variables.

The library already speaks the vocabulary a cleaner document would use. The epistemic namespace
(`per-variable-infogain.md` V1, V8) treats a bare name as a state variable, `obs:` as an
observation variable, and a slot address (`likelihood:pose`) as a parameter variable; a
learnable parameter is already a variable with a prior family, and a fixed array is a delta.
The document should say so.

## The shape, as proposed

- **`variables`** — every variable with its kind: `state` (control nodes included after the
  control-states build), `observation`, `parameter`.
- **`factors`** — transitions, likelihoods, initial priors and the other slots: each with its
  family, its parent variables (`parents`, and `control_parents` until the native control form
  replaces it), and its parameter variables in named slots.
- **Relations** — which variables the System chooses (`controls`); where each observation comes
  from (`source`, written only when a composite bound it).
- **Clamps** — a delta on a variable given with the model. Today's array `values` is exactly a
  clamp on a parameter variable, so one syntax covers a fixed parameter and a learned one (a
  prior family instead of a clamp).

**The guard against scope creep:** the document reorganises how a model is written; it adds no
semantics the engine cannot run. Every factor maps to a slot the runtime has; there are no
free-standing factor-graph factors. `describe`/`build` translate, as M2 already does between
states/observations and factors/modalities.

## Decisions to walk

**D1 — The top-level layout.** *ADOPTED (Alex).* `variables`, `factors`, relations, clamps,
replacing `structure`/`declarations`. *Names ADOPTED (Alex, 2026-10-10):* `relations` and
`clamps`, the plain words; `blanket` and `ports` were set aside because bindings and
`external_inputs` are not blanket ports.

**D2 — Parameters as variables, clamp or prior.** *ADOPTED (Alex); AMENDED (Alex, 2026-10-12).*
Each parameter is a variable; a factor names it among its arguments; it is either clamped
(values given in `clamps`) or learned, and a learned parameter's prior is a **factor** declared
under it in `factors` with the `prior` label, its hyperparameters clamped parameter variables of
their own, exactly as a per-step variable's distribution is a factor. The first cut put the
prior family on the variable record as a shorthand; that is withdrawn: a variable with a factor
is inferred, a variable with a clamp is fixed, a parameter with neither or both is refused. The
word for a factor's static arguments is `parameters`, in the broad sense of the arguments of a
function (Alex); per-step arguments stay `parents`, see Q17 and Q18.

**D3 — Relations.** *ADOPTED as restated (Alex).* A factor carries its own wiring: a likelihood's
or transition's parents (today's `A_deps`, `B_deps`, and `control_parents`), and a
linear-Gaussian factor's variables by role (input; `H`, `c`, noise slots), live in that factor's
record. The relations section holds only relations no single factor owns: `controls` (the
variables the System chooses, `control-states.md` C1), `source` (where a composite's
observation comes from), composition bindings, and `external_inputs`.

**D4 — Evidence on states.** *DEFERRED (Alex).* Clamps on state or observation variables given
with the model are not needed: initial-state priors cover what is known at the start, and data
clamps observation variables at runtime, not at the spec level. *Role proposal ADOPTED (Alex):* by
C1's logic, being observed is a relation (the System receives data for the variable at
runtime), as being a control is (the System chooses it); in a hierarchy a lower level's states
are an upper level's observations. So the variables list marks the real kind
difference — per-step variables against static parameters — and *observed* and *controlled*
are both relations on the System. No new runtime machinery: modalities stay, and
`describe`/`build` translate. *AMENDED (Alex, 2026-10-12):* the per-step/static difference is
**derived**, not marked: a variable named in a factor's `parameters` is static, a variable that
is the child or a parent of a per-step factor is per-step, a variable in both uses or in neither
is refused. A variable record is a **domain** and a size, nothing else: M6's tags
(`log_simplex`, `positive`, `cholesky`, `unit_interval`, `integer_levels`, `real`) move from the
array field to the variable, joined by `categorical` with `levels` and `real` with `dim` and
optional bounds for the per-step variables. `describe` writes a parameter's domain from the
family, `build` checks it against the family's field and refuses a mismatch; a hand-written
document may omit it on a parameter.

**D5 — Naming the arrays file.** *ADOPTED, tentatively (Alex: "not confident on this one").* "Sidecar" is a bare engineering term. Keep
`"arrays_file"` and per-array `{"ref": …}`; drop the top-level `"sidecar"` flag, which
`arrays_file` makes redundant; the write mode becomes `describe(arrays="file")` beside
`"inline"` and `"auto"`. ("Values" was considered; "arrays" seemed usefully concrete. Revisit before the build if it still grates.)

**D6 — Migration.** `describe` writes 0.3; `build` still loads 0.2 documents, converting on read,
so existing stored documents (zeroth-harness's) keep working. *ADOPTED (Alex):* for one release,
with a deprecation warning, then dropped; a consumer converts a stored document by building it
and describing it again.

*(Note 2026-10-23: the 0.2 reader is removed after 0.18.0, ahead of a public release with no
external 0.2 documents; a 0.2 document is refused with a sentence naming 0.18.0 as the last
release that reads it.)*

**D7 — Bundling.** *ADOPTED (Alex).* Ships with the control-states build, which already revises the format
(state records lose `controlled`/`num_controls`; `observes` → `parents`; `source` defaulted).

**D8 — The `static` block.** *ADOPTED (Alex, 2026-10-05); the name `options` is a first pass.*
`static` names a JAX mechanism (the fields a class marks `static=True`), not a modelling role,
and what lands there is two different things. (1) Layout bookkeeping the factor's wiring already
implies — `parent_ndims` (the number of trailing parent axes: the length of the factor's
`parents` plus its control axes), `parent_dims`, `child_dim`, `event_ndims` — is no longer
written; `build` computes it from the wiring and the variables' sizes and still checks every
array's shape against it. (2) Genuine family configuration — a network's `activation`, `head`,
`min_scale`, `symlog_inputs`; a code's `unimix`; a critic's `loss`, `num_bins` — moves to an
`options` record on the factor. `frozen` is neither: "not trained" is a parameter variable's
clamp-or-learned status, so it moves under D2.
*Amended by Q18.3 (2026-10-12):* `frozen` is learning configuration, not a clamp-or-learned
status; see Q18.3.


## Q17 — The factor section

*ADOPTED (Alex, 2026-10-11), walked 2026-10-10/11.* `factors` is keyed by child variable, and
under each child by a **label** naming one conditional on it. Each record holds `family`,
`parents`, `parameters` naming its static arguments by the family's field names, and
`options`; values live in `clamps` keyed by parameter variable, and a learned parameter's prior
is a `prior` record under it in `factors` (D1, D2 as amended). `transition_parents`, `observes` and `control_parents` retire into `parents`, with
control nodes among the parents.

**Time is stated per parent.** Each parent entry carries the offset at which the factor reads
it, zero for the same step, one for the step before; a record also carries the steps it is
active at, every step by default, or the first step only. This is the generic spelling. It was
chosen over a lag on the whole factor because the engine already runs a factor the whole-factor
spelling cannot write: the hybrid System's continuous transition reads its own state one step
back and the discrete mode at the same step (`hybrid-system.md`, the within-step ordering rule),
the standard form of a switching linear dynamical system. A whole-factor lag could state that
only as a hidden engine rule or through a delayed copy variable the engine does not have. The
per-parent form is a strict superset, and the engine's existing acyclicity check on same-step
parents (a valid order named in the refusal) is the one rule the generality carries.

**The role words are sugar.** A label that is one of `initial`, `transition`, `likelihood` or
`actuator` implies the record's time structure, so a reader interested in AIF models writes
`factors["location"]["transition"]` and never sees an offset:

| label | parents read at | active at |
|---|---|---|
| `initial` | the same step | the first step only |
| `transition` | the step before | every step after the first |
| `likelihood` | the same step | every step |
| `actuator` | the same step | every step |

A record under a role label may still give an explicit offset on a parent (the hybrid
transition is `transition` with the mode's offset written as zero); the one guard is that an
explicit entry may not contradict what the label implies where the engine cannot run the result
(`initial` with a parent read a step back). Any other label is the factor's own name and states
its offsets and scope explicitly; it is also the name a flat factor-graph form would key by, so
that later lift is a reader change. A lag on the whole factor as a third spelling was considered
and dropped: the role words cover the common cases and the per-parent form the rest. Likelihood
and actuator are told apart by the child, an observed variable (D4's relation) or an action
node, not by anything on the record; an `actuator` record is written only when a kernel is
declared, the delta being the default. The reader lowers every spelling to per-parent offsets,
which is all the engine sees, and refuses what the engine cannot run: two every-step
conditionals on one latent child, a same-step cycle, an offset the families do not support.
`describe` writes the role label whenever one applies.

The sugar/description relation this settles: the same as sugar's relation to the API, a
spelling lowered once on read, the engine never seeing it. Whether the same-step factors on a
sensor and an effector should become one role is deferred to a predictive-coding backend
(`predictive-coding-reading.md`). Per-family slot names (Q18) and the written syntax of an
offset and a scope are Q18's.

**Acceptance bake-off (Alex, 2026-10-11).** Once the 0.3 reader exists, a small fixed set of
models given in prose, each built by several naive agents (Sonnet or Haiku) with the format
document as their only reference, in 0.2 and in 0.3 with the 0.3 arm split between the role
spelling and the generic one; scored on building first try, correction rounds, and a short
reading test on a given document. The 0.2 arm carries pymdp's letters as a prior every agent
has, named as a confound. It doubles as the reader's acceptance test and as evidence on which
spelling agents reach for.

**Run (2026-10-13).** The record is `../experiments/description-bakeoff/`: four models, 24
writing cells, 12 reading runs. 0.3 role built 7 cells of 8 against 0.2's 4, with no 0.3 cell
refused on layout; the reading means were 5.50 (0.2), 5.75 (role) and 6.00 (generic) out of
six, the generic arm's edge a single Haiku answer on a record the test document had named
`slips`. The run changed the reader three times (a composite's uncontrolled transition table
padded to the engine's layout, a transition naming an action node refused naming the control
node, the legacy reader refusing a module field under `arrays` by name) and left the default
spelling role (Alex, 2026-10-13): the generic arm wrote no better, and `describe` has no names to
invent, so a generic default would be the role document with brackets and `steps` added. Open
from the run, both rulings rather than fixes: a `simplex` domain the reader takes probabilities
in, and a refusal naming every clamp failing its domain at once.

## Q18 — The per-family mapping

**Q18.1 — Control and action nodes.** *ADOPTED (Alex, 2026-10-12).* Both are ordinary per-step
variables, each with its own domain and size (a continuous control node's box is its domain's
bounds); the action node's size is checked equal to its control node's, a mismatch refused
naming both, as a wire is typed at both ends. The pairing is the `controls` relation, a mapping
from each control node to its action node ("the System chooses this node and emits it as that
one"); nothing else marks either variable. An `actuator` record appears under the action node
only when a kernel is declared, with the control node as its same-step parent; a modality whose
parent is an action node is an ordinary likelihood record; the 0.2 `controlled` sugar is not
accepted. `clamps` stays top-level (a clamp is arguably a relation, but `relations` is the small
section a reader scans for structure and `clamps` is where the bulk lives). A factor's per-step
arguments are `parents`, ordered, mapping to table axes or to the linear-Gaussian inputs by
position; its static arguments are `parameters`, by the family's field names; the variables'
derived kinds let the reader check that each names only its own kind.

**A worked example** (two state factors, one control with a stochastic actuator, one learned
likelihood; a `Categorical`'s parameter is its `logits`, unnormalised, and a normalised table's
domain is `log_simplex`):

```jsonc
{
  "hybrid_aif_description": "0.3",
  "library_version": "0.12.0",

  "variables": {
    "location":    {"domain": "categorical", "levels": 4},
    "context":     {"domain": "categorical", "levels": 2},
    "u_location":  {"domain": "categorical", "levels": 4},
    "a_location":  {"domain": "categorical", "levels": 4},
    "position":    {"domain": "categorical", "levels": 4},
    "cue":         {"domain": "categorical", "levels": 2},
    "D_location":  {"domain": "logits",      "shape": [4]},
    "B_location":  {"domain": "log_simplex", "shape": [4, 4, 4]},
    "D_context":   {"domain": "logits",      "shape": [2]},
    "B_context":   {"domain": "log_simplex", "shape": [2, 2]},
    "A_position":  {"domain": "log_simplex", "shape": [4, 4]},
    "A_cue":       {"domain": "log_simplex", "shape": [2, 2]},
    "A_cue_concentration": {"domain": "positive", "shape": [2, 2]},
    "K_a_location": {"domain": "log_simplex", "shape": [4, 4]}
  },

  "factors": {
    "location": {
      "initial":    {"family": "Categorical", "parameters": {"logits": "D_location"}},
      "transition": {"family": "ConditionalCategorical",
                     "parents": ["location", "u_location"],
                     "parameters": {"log_table": "B_location"}}
    },
    "context": {
      "initial":    {"family": "Categorical", "parameters": {"logits": "D_context"}},
      "transition": {"family": "ConditionalCategorical", "parents": ["context"],
                     "parameters": {"log_table": "B_context"}}
    },
    "position": {
      "likelihood": {"family": "ConditionalCategorical", "parents": ["location"],
                     "parameters": {"log_table": "A_position"}}
    },
    "cue": {
      "likelihood": {"family": "ConditionalCategorical", "parents": ["context"],
                     "parameters": {"log_table": "A_cue"}}
    },
    "A_cue": {
      "prior":      {"family": "Dirichlet", "parameters": {"concentration": "A_cue_concentration"}}
    },
    "a_location": {
      "actuator":   {"family": "ConditionalCategorical", "parents": ["u_location"],
                     "parameters": {"log_table": "K_a_location"}}
    }
  },

  "relations": {
    "controls": {"u_location": "a_location"},
    "observed": ["position", "cue"]
  },

  "clamps": {
    "D_location":  [-1.386, -1.386, -1.386, -1.386],
    "D_context":   [-0.693, -0.693],
    "B_location":  {"ref": "arrays.npz#B_location"},
    "B_context":   [[-0.105, -2.303], [-2.303, -0.105]],
    "A_position":  {"ref": "arrays.npz#A_position"},
    "A_cue_concentration": 1.0,
    "K_a_location": {"ref": "arrays.npz#K_a_location"}
  }
}
```

What the example makes visible: a control node is known only by `controls` and by having no
transition; an action node only by being a value in `controls` and by its actuator record; an
observed variable only by `observed`; a parameter only by being named in some factor's
`parameters`; a learned one by its `prior` record, a fixed one by its clamp. `A_cue` has a factor
and no clamp, `A_cue_concentration` a clamp and no factor. (Names and the `logits` domain
follow Q18.3 and the leftovers; the example was aligned to them when the build found the
difference, 2026-10-13.)

**Q18.2 — A merged composite.** *ADOPTED (Alex, 2026-10-12).* A composite is written flat, as
M8 writes it, and `describe` on a merged System can write nothing else: `merge()` returns a plain
System with its bindings already applied and keeps no copy of its components. The bindings
dissolve into structure. An `ActionControl` identification is the driver's control node in the
parent list of every factor that read the identified one; a relay is that node's state as a
parent at offset 1; an `ObservationLink` is a modality whose parent lies under another prefix; a
`TargetEmission` is a terminal target's parent under another prefix; a `BoundaryPrior` is a
transition table averaged (or sliced, for an integer prior) over the terminated node's axis, the
node absent and the table's variable one axis smaller. None of these needs a record, so there is
no provenance section: M8's `subsystems` block, whose records restate the parent lists and name
nodes the document does not contain, is dropped. Membership is the name: the separator `.` is
reserved to composite namespaces, a dotted variable belongs to the component its prefix names,
and a document with dotted names is a composite whose per-component record `build` reconstructs
from the prefixes (a mind's modalities are those under its prefix, the control nodes it chooses
are those under its prefix still present, a bound modality one with a parent under another
prefix). Component order is the document's variable order. What this loses is one refusal: after
a round trip a System no longer knows that a removed node ever existed, so naming it gets
"unknown variable" rather than what it became; the runtime record's fields that hold this
(`identified`, `relayed`, `terminated`) do not survive the round trip and the build brief
decides whether to drop them from the record and keep that refusal at merge time. A nested
spelling, each component's own document under `components` with a `bindings` list, lowered by
running the merge, is a later second spelling for hand-authored composites and is not in 0.3.
A `Wiring` (C13) is not described in 0.3; a wire is an identity between typed ports and lives
in the harness.

**Q18.3 — The family field names.** *ADOPTED (Alex, 2026-10-12).* The keys of a factor's
`parameters` are the family's constructor field names, and a field that is itself a module
nests as a record under its name (`"innovations": {"loc": "mu_x", "chol": "L_x"}` for a
linear-Gaussian transition); the nested record carries no family of its own, since the outer
family fixes it, and layout bookkeeping is not written (D8). `G` is omitted when the factor
reads no continuous control node. A conjugate posterior over one parameter (InverseGamma or
InverseWishart over a noise, MatrixNormal over a head) is a `prior` record under that
parameter with the hyperparameters as clamped variables, as the Dirichlet is. The matrix-normal
inverse-Wishart family is not an exception: its model is `R ~ IW(Psi, nu)` and `W | R ~ MN(M,
R, V)`, so it is a `prior` under the noise variable and a `prior` under the head whose
`parameters` name the noise variable as its row covariance; the reader recognises the pair and
lowers it to the engine's joint family, and with the noise clamped the head's record alone is
the engine's standalone matrix-normal head (the engine docstring's "cannot be expressed per
field" means per independent field, and is reworded in the build). Learned hyperparameters: a
document is the model at one moment, so a clamp means "given with this document", never
"constant"; inference moves beliefs over variables that have factors and learning moves the
values of clamps, the posterior counts being the next document's clamp. Neither the
hyperparameters nor network weights are a third status: a network's weights are clamped values
a trainer moves, and the constructor recipe is a third way of giving a clamp's value beside
inline and `ref`, `{"init": {…}}`, drawn at build with a key. *Delta from D8:* `frozen` is not
a model fact but learning configuration, which clamped variables the trainer moves, and goes
with the other configuration under 18.5; the same holds for whether conjugate learning updates
a prior's hyperparameters. A variable record may be a tree of variable records (a product
variable) with a clamp that is a tree of values and one arrays-file key per leaf; the leaf is
the primitive, `describe` writes one tree variable per network, and the likelihood's
`parameters` name it once (`{"layers": "theta_z"}`). The product is apt where nothing else
names a part separately, which is why it serves the network and not the MNIW pair.

**Q18.4 — The written syntax of offset and scope.** *ADOPTED (Alex, 2026-10-12).* A parent's
offset is written on the parent's name in dynamic-Bayesian-network form, `t` being the child's
step: `x[t]` for the same step, `x[t-1]` for the step before; a bare name means the offset the
record's label implies. Under a role label parents are bare except where one overrides the label
(the switching transition is `"parents": ["x", "mode[t]", "u_x"]`); under a generic label every
parent is bracketed. Any other bracket is refused naming the two forms, and a variable name may
not contain one. The steps at which a record applies are the field `steps`, with values `"0"`
(the first step only), `"1+"` (every step from the second on) and `"all"`; `"all"` is the
default when the field is absent (Alex), and a record applying at step 0 that reads a parent at
`t-1` is refused naming `"1+"` as the fix, which is Q17's guard. A side map of offsets keyed by
parent name and an `@` suffix were considered: the map keeps names free of syntax but needs the
format document to read, and `@0` reads as an absolute step index, the thing `steps` says, not
as an offset. The bracket form is one spelling, readable without the document, and the generic
record shows its whole time structure in the parent list. `describe` writes the role label when
one applies and brackets only the overrides.

**Q18.5 — The declarations D1–D8 do not place.** Walked in five parts.

*18.5a, targets — ADOPTED (Alex, 2026-10-12).* A preference target is a distribution over a
variable the model declares, with a family and parameters, so it is a factor record under that
variable, with two role labels: `target`, scored at every step of a plan (C under an observed
child, the risk target under a latent one), and `terminal`, scored at the plan's end (h). A
target's `parents` are its conditioning variables: the terminal target's today (`H_deps`), and
every target's once the engine scores a conditional C (a gap row in the build plan; Alex: the
library's C is conditional in general). The conditional initial prior (`D_deps`) is the
`initial` record's parents. `steps` does not apply to these labels, their time being plan time;
a step set on a `target`, under which `terminal` would be the target at the last step, waits on
a general step set and is not pressed. The raw-logit convention of conventional AIF is the
`Categorical` family's own parameterisation (`logits`, unnormalised), so a C written as logits
is a `Categorical` whose parameter variable has that domain, and the normaliser the engine
adds is a constant per modality that no policy comparison sees. **The caution (Alex):** a
`target` record is a factor of the *objective*, read by the planner's scoring and never by
inference. Conventional AIF's C is a prior only in the planning-as-inference model, and folding
it into the generative model proper would be ill-posed beside the state prior and likelihood
over the same observation. The role labels are the partition: `initial`, `transition`,
`likelihood`, `actuator` and `prior` are generative roles, `target` and `terminal` objective
roles, and the reader never lowers an objective record into a generative kernel. The further
goal this leaves open is C as the model's own marginal likelihood, which the same record admits
as a family the engine does not yet have; an arbitrary `target` declaration stays available
beside it.

*18.5b, the policy prior — ADOPTED (Alex, 2026-10-12).* E is a static parameter like any other,
and the object it is a prior over is not a new variable: it is the control nodes' trajectory
over the plan horizon, the planner's enumerated policy set. E is written as a factor record
under the control node with the role label `policy_prior` (`{"family": "Categorical",
"parameters": {"logits": "E"}}`), meaning what the engine's field means: one distribution over
the joint policy set of every control node in `controls` order across the horizon, written once
under the first control node, a second refused as a second E; `steps` does not appear on it;
`build` checks E's length against the policy set the planning configuration implies and names
the horizon and the node levels when they disagree. This is the first role label with no
generic spelling beneath it: it lowers to the engine's slot directly, and a generic form waits
on a general step set or on multi-step offsets. A per-step prior on a control node, conditional
on a state or not (the habit, psychologically the realistic form), is the ordinary `prior`
record under the node with its parents, a factor like any other that the engine does not score
yet (a gap row in the build plan); the two may coexist. `prior` under a control node is
distinct from `target` under it, as the engine is: a target on actions is a cost inside G under
the precision, a prior is E outside it. Considered and set aside: E in the planning block
(Alex: E is a parameter of the generative model) and a `support`/`steps: "plan"` field (one
non-default value, and it folded plan time and the joint over control nodes into one word).

*18.5c, recognition — ADOPTED (Alex, 2026-10-12).* A recognition entry is a q-side factor
(A11), and it is written where 18.5a put the objective records: under the variable's block, with
the role label `recognition`, the third partition beside the generative and objective labels.
The record has the factor shape: `family`, `parents`, `parameters` (the network as a tree
variable, 18.3) and `options`. A11's prefixed parent grammar (`obs:`, `recognition:`,
`external:`) goes: variables are one namespace, so an observed parent is its name with 18.4's
brackets; an auxiliary q-side variable (an embedding) is a variable in `variables` with a domain
whose only record is a `recognition` one, which is what marks it q-side; an external input is a
variable listed in the relation `external` beside `observed`, data the System receives with no
likelihood, its width its domain, and `external_inputs` retires into it. One reading rule,
stated once: in a `recognition` record a latent parent denotes its prediction at that step, the
prior the generative record produces, and an observed or external parent its data. Evaluation
order is derived from the explicit parents, not declared; a cycle, or a parent that is another
amortized posterior, is refused (A11's constraints as checks). The habit module, a policy from
beliefs to an action distribution, is on this reading the control node's `recognition` record
(an amortized q over the control node given beliefs, which is what an actor is), with the
planning configuration set to act from it; the document form is recorded and the engine's
storage under `policy_prior` is the build's concern. The per-variable q annotations
`predictive-coding-reading.md` defers (factorisation and form constraints) land in the same
slot when they come. Set aside: a `recognition` section of its own, which would have put the q
side in a second list a reader reconciles against the first.

*18.5d, scoring — ADOPTED (Alex, 2026-10-12/13).* One `objective` block holding a list of
`terms`, each a record `{"term": <name>, "on": <variable>, "parameters": {…}, "model": {…}}`.
`term` is the registry's term name (not "row", the implementation's word). `on` names a
variable and the registered estimator says what of it (`occupancy` on a latent is its
transition, `predicted_reward` on an observed variable its channel); a term that targets
variables and is written without `on` applies to every eligible one, which is what
`epistemic=True` means today; wildcards stay for the partial cases; a term with no target has
no `on`. `epistemic`'s parameter-block grammar (`likelihood:pose:noise`) dissolves, the block
being the parameter variable (`R_pose`) named like any other. A term's parameters are
variables (a critic a tree variable, an ensemble a list of them) carried on the selection, so
the engine's pairing rule between `terms` and `scoring` cannot be violated; a term that owns a
model nests a document under `model` or references one with `{"ref": "<path>"}`, M4's word
with a path resolved relative to the document. Term names are a registry, an unknown one
refused naming the menu (M7). **Decomposition is a choice of terms, not a logging choice**
(checked in the compiled objective): the utility form mounts a utility leaf per targeted
observation and an information-gain leaf per selected variable; the risk form a risk leaf per
targeted observation and an ambiguity leaf on every channel and no information-gain leaf, the
two agreeing only when every channel is both targeted and selected. So the pair is written as
terms (`utility` and `infogain`, or `risk` and `ambiguity`), `decomposition` stays as sugar on
the block for the compact standard case, expanded to the pair with its defaults with further
terms listed beside it, and a mix the term tree cannot build (utility with ambiguity) is
refused naming the two pairs until the tree is general. A pure occupancy objective lists only
occupancy. `ambiguity` takes no `on`: it is a property of each channel and every channel's must
be present for the risk form to be EFE (`per-variable-infogain.md`), the selection reaching it
through its conditioning. **Valence** is the registered term's, +1 a cost and −1 a value
(utility, infogain, occupancy, g_tail, predicted_reward −1; risk, ambiguity,
model_uncertainty, terminal_risk +1); the document never writes a sign, a term's sign being
part of what the term is, and no per-term weight exists in the engine, the precision being the
only scale; the format document states beside each term what it measures and how it enters G.
Build items: a parametric row declares its parameter names as a family declares its fields;
`describe` writes the list from `terms`, `epistemic`, `scoring` and `decomposition`, and
`build` reconstructs the four, `infogain` staying a selector in the engine.

*18.5e, configuration — ADOPTED (Alex, 2026-10-13).* One `configuration` block with three
sub-blocks holding exactly what a System stores beyond the model, every field mapping to one
System field and absent sub-blocks taking the engine's defaults: `planning` (`planner`,
`horizon`, `gamma` the policy precision, `branch_on` for the sophisticated planner),
`inference` (`iterations`, `mixture_k`, `mixture_schedule`), `learning` (`frozen`, the clamped
variables a trainer does not move). Planner-specific settings (a candidate count, a step
count) and learning rates and decay are call-time arguments to `act` and `learn`, not stored,
and so not the document's; numeric precision is the process's x64 flag, not stored on a System
and not written, an inline clamp taking the build's default dtype and an arrays-file clamp its
own (a `dtype=` cast on `build` is a one-line convenience if wanted). `frozen` is the module's
flag for a gradient family today; for a conjugate hyperparameter the engine has only a
call-time rate of zero, so naming one there is a gap row (a stored default the learn step
reads).

*The leftovers — ADOPTED (Alex, 2026-10-13).* Default parameter names: `describe` writes the
AIF letters where a role label applies to a tabular record (`A_<child>`, `B_<child>`,
`C_<child>`, `D_<child>`, `H_<child>`, `K_<child>`, `E`), a hyperparameter as
`<parameter>_<field>` (`A_cue_concentration`), and otherwise `<child>_<label>_<field>` with nested
fields joined the same way (`x_transition_innovations_loc`); names are the user's to change
and nothing parses them; the dot stays reserved for composite prefixes. A scalar clamp
broadcasts to the declared shape, as the box bounds do. Key order is declaration order: JSON
objects carry order in every reader that matters, so the `name` lists do not return; the
variable order fixes node order and the recognition order is derived from parents, so a tool
that reorders keys changes the node layout of the same model, never the model. Q20 stays with
D6: the 0.2 reader lives one release with a warning.

**Q18.6 — After the bake-off (Alex, 2026-10-13).** *ADOPTED.* Two agent-facing conveniences the
bake-off asked for (Q17, "Run"). (a) A `simplex` parameter domain: a table written as
probabilities (entries ≥ 0, each distribution summing to one within tolerance, `0` for an
impossible entry), accepted wherever a field takes `log_simplex` or `logits` and lowered to its
logs on read. `describe` keeps writing `log_simplex`, the engine's own representation, because
the bitwise round trip of a native System is an invariant that exp then log would break; so
`simplex` is a writing convenience lowered once on read, the relation the role labels have to
the API. (b) Refusals listed per section: the reader's checks stop at the first fault by
design, and that stays across sections, whose checks depend on one another; within a section
whose records are independent (the variable records, the factor records, the clamps' pairing
with their variables, the values each factor's parameters take) every failing record is
reported in one refusal, one line each; a variable several factors read appears under each,
the one cause named each time. Library-wide collection was declined: across sections
a list would be one fault and its echoes, and the engine's validators are the same kind.

**Q18 is closed (2026-10-13).** Next: the build brief for the 0.3 writer and reader, with the
Q17 bake-off as its acceptance test, drafted for Alex's read before an agent takes it. Both landed: the writer and reader with
`../description-format.md` as the specification, and the bake-off under
`../experiments/description-bakeoff/`.

## Description length (noted 2026-10-13)

A canonical document is also a candidate complexity term for model search, where agents revise
documents and a description-length penalty keeps the search honest. Nothing is built; the note
records what the canonical form must keep for that use. The length is computed on the
structure the document carries, never on its text, and three things in 0.3 make the dumb
costs avoidable. **Names cost nothing:** nothing in the engine parses a name and key order is
declaration order, so the form measured alpha-renames every variable and every user label to
its position (a composite's dotted prefixes are names too). **Placement and spelling cost
nothing:** the form measured is `describe(build(doc))`, which writes role labels, no
`decomposition` sugar, logs for probabilities, and arrays inline or in the file by a threshold,
so inline against ref, role against generic, and `simplex` against `log_simplex` measure
alike. **Parameters cost by precision, not by digits:** a clamp's length is a stated code
length per entry at a stated precision, a constant array broadcast from one number costing
one entry, and a learned parameter costing its prior's hyperparameters rather than a value.
The length is then a sum over variables of a size code, over factors of a family code plus one
code per parent, and over parameters of the value code, with sizes and families drawn from a
fixed menu: the two-part form with the structure as the first part. The measure itself is
written when the model-search work is scheduled.
