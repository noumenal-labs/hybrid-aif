# Design: the model description — a `System` as a document

*The format is description 0.3: see `model-description-0.3.md` (the design) and `../description-format.md` (the specification).*

*Status: ADJUDICATED 2026-09-18 — M2 (naming amended), M4 and M6 by Alex, the rest accepted without amendment; **LANDED the same day**: `hybrid_aif.description` (`describe` / `build` / `schema` / `save` / `load`), the checked-in schema, `tests/test_description.py`, the top-level exports, and a section in `../composing-systems.md`; **0.2 the same day (M11)**. Companion to `system.md` (what a `System` is) and
`../composing-systems.md` (how one is written down in Python). Motivating
customer: zeroth-control's model-patch loop (`zeroth-control/docs/model-patch.md`),
whose phase-3 "structure" patches need the declaration itself to travel as data.*

## What was adjudicated before, and what this amends

`README.md` (design principles, "Model representation"): *"Source of truth: typed
Python spec (pydantic) carrying distribution type, mean-field block id, inference
method, and wiring-by-name; exported/imported as a versioned JSON schema so
external tools can author models without touching Python. Custom functions by
registry reference."*

`project-brief.md` ("Model representation: typed Python spec + versioned JSON
view"): *"Pydantic `NodeSpec` classes are the source of truth (spec/compile layer
only — validated once at load, compiled to equinox pytrees; nothing pydantic on
the runtime hot path, and the JSON Schema, not the library, is the contract, so
the validator is swappable if load time ever matters); a generated, versioned
JSON schema lets external tools author models directly … custom functions enter
by registry reference (`"impl": "pkg.mod:Thing"`), failing loudly if
unregistered."* And, on deployment (brief, "Three-tier compute"): *"parameters use
the sidecar format; structure uses the versioned spec."*

`system.md` ("Planners are injected, never owned"): *"ownership in the data
structure is only ever a name and Systems remain serializable pytrees."*

**The amendment this doc proposes (Alex, 2026-09-18: "if we're not already using
Pydantic, I'd skip it. It's heavy at runtime, and if we can produce a valid JSON
schema or equivalent from our Equinox code, let's").** Everything in the quoted
text survives except the *mechanism*: there is no pydantic spec layer. The
source of truth stays the Equinox declaration — `System` (all-static) plus
`SystemParams` — and the document is **derived from it**, both ways: a
`describe` that writes the document and a `build` that reads one, with the JSON
Schema generated from the same class inventory. What the brief called "the
spec/compile layer" collapses to one module with no runtime footprint, because
the compile target already *is* the declaration. The contract is still the
schema and not the library, as the brief wanted.

## Why now, and what it is for

The composition probe (`../agent-composition-probe.md`) found that a program
composing or patching a model today does `dataclasses.replace` on positional
tuples, that a half-applied structural patch escapes validation as a bare
`IndexError`, and that the addressable parameter namespace exists only as a
private helper with names and kinds but no shapes, dtypes or ranges. The patch
loop needs three things from us, and one document serves all three:

1. **a genotype** — the thing a proposer edits and a validator diffs;
2. **a grammar check** — "does this document describe a `System` at all?",
   answerable before any compute, with a sentence naming the fault;
3. **an envelope** — every patchable address with its family, shape, dtype and
   admissible domain, so the kernel's allowed ranges can be *derived* per model
   rather than hand-written per embodiment.

## The decisions

### M1. One module, three functions, no new dependency

`hybrid_aif.description` (`spec` is taken by the vocabulary of the brief,
`schema` by JSON Schema itself):

```python
describe(system, params, *, arrays="inline" | "sidecar") -> dict     # the document
build(document, *, arrays=None, key=None) -> (System, SystemParams)  # its inverse
schema() -> dict                                                     # JSON Schema 2020-12
```

Plain `dict`/`list`/`str`/`int`/`float`/`bool` throughout, `json.dumps`-able
as is. **No pydantic**, no new runtime import on any face. The document carries
`"hybrid_aif_description": "0.1"` and the library version that wrote it;
`build` refuses a major it does not know.

*Alternative rejected:* pydantic `NodeSpec` classes as the brief proposed —
a second declaration of every family, kept in sync by hand, for a validator the
Equinox field metadata can already generate.

### M2. The document is name-keyed, in the *variable* vocabulary — ADJUDICATED 2026-09-18 (naming amended)

Runtime `SystemParams` slots are positional tuples aligned with `B_deps` /
`A_deps`. The document keys every declaration by the address the library
already uses everywhere else (`selectors._slot_declarations`):
`transition:<state>`, `likelihood:<observation>`, `initial_state_prior:<state>`,
`recognition:<variable>`, `scoring:<row>`; targets by `obs_targets:<observation>`,
`state_targets:<state>`, `terminal_targets:<state>`; and `policy_prior`.

**Vocabulary (Alex's ruling):** the document says `states` and `observations`,
never `factors` or `modalities`. "Factor" is ambiguous between an AIF state
factor and a factor-graph factor (which a likelihood also is), and in a
hierarchical System a lower level's states are effectively an upper level's
observations. The library's epistemic namespace already speaks this way — a
bare name is a state *variable*, an `obs:` prefix an observation *variable*
(`per-variable-infogain.md` V1/V8) — so the document adopts it. The runtime
field names (`factor_names`, `modality_names`, `A_deps`, `B_deps`) are
unchanged; `describe`/`build` translate.

The two lists are ordered records, so the wiring is written by name and
resolved to indices on `build`:

```jsonc
"states":       [ {"name": "target", "transition_parents": ["target"],
                   "controlled": false, "num_controls": 1}, … ],
"observations": [ {"name": "distance", "observes": ["target"], "source": "external"}, … ]
```

`source` is `"external"` for a total-System input, or
`"<subsystem>.<state>"` for an observation `merge()` bound to another
component's state — read off the `subsystems` provenance when present. An
internally bound observation is still listed here, because it *is* an
observation of the flat runtime System; `source` is what keeps it from being
mistaken for an input.

Consequence: `describe` **requires** names (`factor_names` and
`modality_names` today optional). A `System` without them is described with
the library's own generated names and a `generated_names` record saying which
tuple was missing, so `describe` never fails on an existing model and `build`
restores the declaration exactly (names unset where they were unset). *[As built:
the record is `{"states": bool, "observations": bool}`.]*

*Why name-keyed:* the half-applied patch in the probe (four `A_deps`, three
likelihoods) is impossible to *write* in a name-keyed document: an observation
is one record with both its wiring and its declaration. Diffs are readable
("`likelihood:colour` added") instead of positional.

### M3. Two layers, one document: structure and declarations

```jsonc
{
  "hybrid_aif_description": "0.1", "library_version": "0.7.0",
  "structure":     { /* every static System field, verbatim; "states" and "observations" lists carry the wiring by name (M2) */ },
  "declarations":  { "transition:target": { "family": "ConditionalCategorical",
                                            "static": {"parent_ndims": 2},
                                            "arrays": {"log_table": {"shape": [3,3,1], "dtype": "float32",
                                                                     "domain": "real", "values": [[…]] /* or {"ref": …}, M4 */}} },
                     … },
                     "obs_targets:answer": { "family": "Categorical", … },
                     "policy_prior": { … } },                      /* targets and E are addresses too */
  "sidecar": false                                                 /* true when any array is a ref */
}
/* external_inputs and subsystems ride inside "structure", verbatim */
```

A declaration is `family` (the class name from the public node menu,
`hybrid_aif.distributions.__all__` plus the `factors` module's LG classes and
`Deterministic`/`DerivedInitial`), its **static** fields (whatever the class
marks `static=True`, verbatim), and its **arrays** (every inexact leaf, with
shape, dtype and a per-field *domain* tag — see M6 — and either inline `values`
or a sidecar `ref`). The three-way split is mechanical: `eqx.partition` on
`is_inexact_array` gives arrays; the static fields are read off the dataclass;
nothing is hand-listed per family.

### M4. Arrays: a linked `.npz` above a size threshold, inline below; neural weights by recipe — ADJUDICATED 2026-09-18

Alex: "inline array values in JSON … gets really heavy to load, really fast.
I'd use some binary format for any arrays, linked .npz for example. Maybe under
a certain size limit we could use inline values … it's about (a) agent
usability and (b) loading speed." And: there is no *default* here — one mode or
the other is chosen at every `describe`.

So the rule is **per array**, with the mode chosen per call:

- `describe(system, params, arrays="auto", inline_max=256)` — the default
  mode. An array with at most `inline_max` elements is written inline as
  nested lists (a T-maze table, a 3×3×1 transition: readable by an agent in
  the document itself); anything larger is written as `{"ref": "<key>"}` and
  returned beside the document as `dict[key, ndarray]` for `np.savez`. The
  document records `"arrays_file": "<stem>.npz"` when any ref exists.
- `arrays="sidecar"` — every array a ref (a deployment artefact; the 1.4M-
  parameter reference agent).
- `arrays="inline"` — every array inline (tests, tiny models, a document meant
  to be read whole).

Each array entry carries `shape`, `dtype`, `domain` (M6) and exactly one of
`values` or `ref`; the schema says so, and `build(document, arrays=…)` names
the `.npz` in its error when a ref is met with no arrays supplied. The `.npz`
keys are the addresses (`transition:target/log_table`), so the sidecar is
readable on its own.

- **Recipe** for neural declarations: an `MLPConditional`, `MLPPredictor`,
  `GRUCell` or `LearnedVector` is *also* described by its constructor recipe
  (`sizes`, `activation`, `head`, `norm`, `winit`, `frozen`, …) so that a
  document may omit its arrays and `build(…, key=…)` draws them fresh. **This is
  how a proposer adds "an amortized trainable function": by recipe, never by
  weights.** A document with neither arrays nor a key for such a declaration
  refuses, naming the declaration.

*Not in the document:* `LearnState` (optimizer moments — a checkpoint concern,
`eqx.tree_serialise_leaves` stays), beliefs, policies, replay. Structure and
parameters only, the two things the brief's deployment section names.

### M5. `build` is where total validation lives

`build` validates the **document** before it constructs anything, with
sentence errors that name the address and the repair:

- every factor named in a `transition_parents`/`observes` list exists;
- every factor has exactly one `transition:` and one `initial_state_prior:`
  declaration, every modality exactly one `likelihood:` — the arity check the
  runtime lacks (`../agent-composition-probe.md` §(c)4);
- every array's shape agrees with the shape the wiring implies (child levels,
  parent shape, the trailing control axis — the four rules in
  `../composing-systems.md` become checks that say which rule was broken);
- every static field is one the family declares; every family is in the menu.

Then it constructs the `System` and `SystemParams` and runs the existing
`System._validate` eagerly, so the family-purity and ordering checks fire at
build time rather than at the first face call. **The constructor itself is not
changed** — `System(...)` stays cheap and lazy for the library's own callers;
the document path is where program-generated models arrive, so that is where
the eager, total check belongs. (The probe's item 3 asked for validation "at
construction"; this is the same guarantee delivered at the boundary a
program-generated model actually crosses, without touching the hot path.)

### M6. Domains: the envelope, per field, from the class — ADJUDICATED 2026-09-18

Each array field carries a `domain` tag written by the family, not the
document author: `"real"` (logits, `loc`, `F`, `H`), `"positive"`
(`concentration`, `alpha`, `beta`), `"log_simplex"` (`log_weights`, a
`Categorical`'s logits *as read*), `"cholesky"` (`chol`, `chols`, `R_chol`:
lower-triangular, positive diagonal), `"unit_interval"`, `"integer_levels"`.
`build` checks the cheap ones (positivity, triangularity) and reports the tag
for the rest. This is item 2 of the probe's list — public parameter enumeration
with ranges — delivered as a property of the description rather than a second
API: `describe(...)["declarations"]` **is** the enumeration.

**ADJUDICATED 2026-09-18 (Alex):** a per-model **patchable** marking stays
entirely on the customer's kernel side and out of hybrid-aif. The kernel is
theirs; we describe what exists and what its domain is.

### M7. Registry references for everything that is a name

Planner (`"planner": "cem"`), scoring rows (`terms`), estimator rows,
`Deterministic.map` and any `map` that is not one of ours: `"impl":
"pkg.mod:name"`, resolved by import on `build`, **refused if not importable or
not registered** — the brief's rule, verbatim. `Deterministic` with a lambda
`map` cannot be described; `describe` says so and names the node.

### M8. Nested Systems are described recursively

A scoring row that owns a `System` (the CRSPP:
`scoring:cosine_sim_to_pref_rollout`) is described as
`{"family": "System", "system": {…}, "params": {…}}` — one document nested in
another, the same grammar. `merge()`'s `subsystems` provenance is carried
verbatim as metadata (no face reads it, `system.md`), and the coupled System is
described flat, as it is at runtime.

### M9. The round-trip contract, and the fixtures that enforce it

`build(*describe(system, params)) == (system, params)` **bitwise** — every
static field equal, every array leaf `array_equal`, every dtype preserved — on:
the T-maze (`reference.envs.tmaze`), the probe's model
(`examples/compose_target_factor.py`), `hill_lg`, `hybrid_switching`,
`noise_learning`, `tiny_rssm`, and `raif_mountain_car` (which covers the neural
menu, the recognition slot, the scoring slot and the nested CRSPP). Plus the
schema test: every document `describe` writes validates against `schema()`, and
the schema is checked into the repo as a versioned file so external tools have
the contract without importing us.

### M10. Exports

`System`, `SystemParams`, `describe`, `build` from `hybrid_aif` (the probe's
item 4, one line each). The `hybrid_aif.system` path stays.

### M11. Unknown top-level keys are refused; consumer data has one reserved place — ADJUDICATED 2026-09-18 (Alex), LANDED as description 0.2

The first consumer found that an unknown top-level key was silently accepted,
and used the accident to carry its own binding block inside our document. Alex:
"it should be a deliberate reserved namespace rather than an accident of the
schema." So, as of description **0.2**: the top-level keys are exactly
`hybrid_aif_description`, `library_version`, `sidecar`, `arrays_file`,
`structure`, `declarations` and **`extensions`**; anything else is refused by
name, with the sentence pointing at `extensions`. `extensions` is an object
keyed by consumer name (`{"zeroth_control": {...}}`), each value an object,
**opaque to the library**: `build` never reads inside it, `save`/`load` carry
it untouched, `describe(..., extensions=)` writes it on request, and the schema
closes the top level with `additionalProperties: false`. 0.1 documents still
build (same major; the change only tightens).

## Not decided here

- **A replay interface** (the probe's item 6; zeroth-control's `replay(model,
  episodes)`). Different concern — driving a *built* model through a recorded
  trace and stating which addresses the trace can exercise. Its own round, after
  this one lands, because it needs the trace format the customer is still
  settling.
- **Discrete latent over continuous observation** (probe item 5) and the "v0.2"
  wording (item 7): documentation and roadmap, not this doc.
- **Whether `System` itself should become name-keyed at runtime.** The document
  is; the runtime tuples stay. Revisit only if a second customer wants names at
  the face.

## Build plan, once adjudicated

One chunk. `description.py` (describe/build/schema, ~400 lines), the checked-in
schema file, the round-trip suite over the seven fixtures, a document-level
validation suite (one test per sentence `build` can say), the two exports, a
short section in `../composing-systems.md` ("The same model as a document") and
a `[LANDED]` note here. The notebook gate runs because the top-level exports are
public API. No engine face changes.
