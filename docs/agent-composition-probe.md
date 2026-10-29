# A naive agent composes a model: friction report

**Commit read:** `734665b147c31d939243a8542fe46efc2f14114c` — "Thread `winit` to
the CRSPP; assert per-optimizer parameter counts against theirs"
(Wed Sep 16 10:55:14 2026).
**Date of probe:** 2026-09-18.

## Method

An LLM agent with no prior exposure to this repository was asked to compose an
active-inference generative model out of the library's parts and to report
honestly where it succeeded and where it got stuck. It was given, and read
first, only `README.md`, `CLAUDE.md`, `docs/capabilities.md`,
`docs/design/system.md` and `docs/design/hybrid-system.md`; it wrote down its
beliefs about the library's abstractions *before* reading any code, marking
each KNOWN-FROM-DOCS / GUESSED / NOT FOUND. It then tried to build the
smallest composed model those docs licensed, dropping into `examples/`,
`docs/design/*` and finally `src/` only where the top-level docs failed —
recording each drop and what it was looking for. Effort was capped at roughly
40 minutes; the repository was read-only throughout (no edits, no commits) and
scratch code ran against an editable install in a separate virtualenv. The
point was the friction, not a finished agent.

The report follows verbatim. Two corrections found while preparing this PR are
appended at the end rather than edited in.

---

## (a) Abstraction map, from the four docs only

| Concept | Status | What the docs gave me |
|---|---|---|
| model | KNOWN | `System` (static structure) + `SystemParams` (values); `Agent` is not a type |
| factor / node menu | KNOWN (names) / GUESSED (ctors) | Categorical, Gaussian, MixtureGaussian, ConditionalCategorical, LG transition/observation, MLP* |
| slots | KNOWN | `transition` / `likelihood` / `initial_state_prior` / `recognition` / `scoring` |
| preference | KNOWN (names) / GUESSED (ctors) | `obs_targets`, `state_targets`, `terminal_targets`, `policy_prior` (E), `D_deps`, `H_deps` |
| policy | KNOWN | planner injected by registry name: enumeration / cem / habit |
| objective terms | KNOWN (concept) / GUESSED (row names) | `terms=(...)`, `epistemic=` wildcards, term tree of `Interval`s with `derived` |
| family swap | KNOWN | family **is** the declaration's type in `SystemParams` |
| amortized | KNOWN | `recognition` per variable; `MLPConditional` frozen or learnable |
| learning | KNOWN | both: conjugate (Dirichlet/NIW/MNIW) and optax, through one `learn` face |
| runtime add-a-part | NOT FOUND | only `couple()`; nothing about patching a declaration |
| serialisable spec | GUESSED | README promises pydantic + versioned JSON schema; capabilities.md silent |
| parameter enumeration w/ ranges | NOT FOUND | — |
| validator/grammar | GUESSED | `_validate` named in prose only |

## (b) What I built — it ran

Plan A (the natural model: discrete "which target", two **continuous**
features) was **rejected**. Plan B is tabular: factors `target` (3) + `mode`
(idle/asked), modalities `distance`/`size` (binned features, learnable
`ConditionalDirichlet`) and `answer` (gated by mode), preference
`obs_targets["answer"]` favouring a resolved answer, actions
{pick0, pick1, ask}.

```python
system = System(
    B_deps=((0,), (1,)), num_controls=(1, 3), controlled=(False, True), policy_len=2,
    A_deps=((0,), (0,), (0, 1)), planner="enumeration", gamma=4.0,
    factor_names=("target", "mode"), modality_names=("distance", "size", "answer"),
    epistemic=("target",))
beliefs, _fe = system.infer_states(params, obs, priors)
action, _, tree = system.act(key, params, beliefs, selection="deterministic")
priors = system.update_empirical_prior(params, beliefs, action)
label = (H.Categorical(logt([0., 1., 0.])), H.Categorical(logt([1., 0.])))  # supervisor: candidate 1
p2 = system.learn(p2, amb_obs, label, lr=1.0)   # x8
```

```
t=0 q(target)=[0.826 0.048 0.126] action=[0 2]      # 2 = "ask"
t=4 q(target)=[0.978 0.000 0.022] action=[0 2]
TERMS: {'infogain': {'answer': Interval(f32[9]), 'distance': …, 'size': …},
        'utility':  {'answer': Interval(f32[9])}}
before label q(target) = [0.826 0.048 0.126]
after  label q(target) = [0.361 0.583 0.055]
```

Everything asked for ran: 5 observations, posterior, chosen action, conjugate
likelihood update from a label, posterior flipped.

## (c) Friction, ordered

1. **Trailing control axis on *every* transition table, even uncontrolled
   ones.** ~12 min lost. Undocumented; `make_transition` does
   `jnp.take(..., axis=-1)` unconditionally
   (`src/hybrid_aif/objectives/compiled.py:127`), so a 2-D table fails as
   `ValueError: Number of einsum subscripts, 1, must be equal to the number of
   operands, 2` four frames deep. The rule is stated **only** in an error
   string inside `learn` (`src/hybrid_aif/system/core.py:1231-1236`).
   → *"`transition[f]` always carries a trailing control axis (size 1 if
   uncontrolled): `parent_ndims == len(B_deps[f]) + 1`", and `_validate`
   should say so.*
2. **Discrete latent + continuous observation is not supported.**
   `src/hybrid_aif/objectives/hybrid.py:410`: "modality 'distance' is
   linear-Gaussian but observes categorical factor 'target'; v0.2 supports
   family-pure modalities only". Good message, but it cites v0.2 from a v0.7
   library, and hybrid-system.md hides the rule in a "v0.2 scope" section a
   reader assumes is superseded. → *capabilities.md should carry the supported
   (factor family × modality family) matrix.*
3. **No constructor reference anywhere in the four docs.** I got every field
   name from `__dataclass_fields__` and their semantics from comments at
   `src/hybrid_aif/system/core.py:394-464`. → *one "declare a System" table in
   capabilities.md.*
4. **`System` is not exported from `hybrid_aif/__init__`** (top level exports
   distributions only). Path found in system.md prose. → *export
   `System`/`SystemParams`.*
5. **Face signatures in docs are wrong.** system.md says
   `infer_states(...) -> beliefs`; it returns `(beliefs, fe)`, and `act`
   returns 3 values. The correct canonical loop exists — in the `System.learn`
   **docstring**. → *lift that loop into README/capabilities.*
6. **`parent_ndims` is a count, not sizes; tables are child-first**
   `(num_levels, *parent_shape)` —
   `src/hybrid_aif/distributions/conditional.py:95-140`.
7. **`epistemic=("target",)` is accepted but the term tree keys `infogain` by
   modality**, so I cannot see whether my selector took effect.

## (d) Step 3 answers

- **Serialisable declaration: does not exist.** No pydantic, no
  `to_json`/`from_json`/schema in `src/hybrid_aif/`; pydantic isn't a
  dependency. README's "typed Python spec (pydantic)… versioned JSON schema"
  is unshipped. What *does* work: `System` is an all-static `eqx.Module`, so
  **patching is `dataclasses.replace`** — I added a fourth modality to a live
  declaration and its params and re-inferred successfully. Diffing is
  structural, in Python, not over a document.
- **Parameter enumeration: partial and private.**
  `system/selectors.py::_epistemic_variables` returns exactly the addressable
  variable namespace (`target`, `mode`, `obs:distance`, `likelihood:distance`,
  …) — names and kinds only, **no ranges, bounds or dtypes**. Nothing a
  kernel/envelope could bound automatically.
- **Validator: real but not total.** `system/validation.py::_validate` (+
  `objectives/hybrid.py::validate_structure`) fires at face-call time, not
  construction, with excellent named messages for family/ordering errors. But
  my deliberately ill-formed patch (4 `A_deps`, 3 likelihoods) produced a bare
  `IndexError: list index out of range` — exactly the failure mode a
  patch-and-diff agent will hit first.

## (e) Assessment

An LLM *can* get a composed, learning agent running from this repo — I did, in
one sitting, and the conjugate-learning story is genuinely first-class. But not
from the docs alone: I dropped into `src/` four times, and three of those were
for shape conventions that no document states. The docs are written as
*decision records* — they explain why the design is right, at length — where a
composing agent needs a reference: a constructor table, a family-compatibility
matrix, the canonical loop, and the shape rules. The bigger structural gap for
zeroth-control is that the model has no document form: composition, diffing and
patching all live in Python object surgery against private selector helpers, and
validation is incomplete exactly where a program-generated patch would land.
Closing item 1 and item 3 above plus a `System.describe()` that dumps the
declaration and its parameter addresses would move this from "an expert can" to
"an agent can".

---

## Corrections found while preparing this PR

Both are places where the probe was right about the friction and wrong about a
detail; they are recorded here rather than edited above, so the report stays
the artefact it was.

1. **`infer_states` returns `(beliefs, detail)`, not `(beliefs, fe)`** — the
   second slot is an `InferenceDetail` (`None` unless `detail=True`), the
   inference-side twin of `infer_policies`' `(q_pi, term_tree)`. The friction
   (item c5, the doc's signature is wrong) stands; the probe's guess at what
   the second element *was* did not.
2. **Item c7 is sharper than the probe could see, and item c2 has a
   workaround.** The `epistemic=` selector *does* take effect — naming a state
   factor mounts a leaf on every modality that observes it, so `("target",)`
   gives three leaves, `("mode",)` gives one, and `("likelihood:*",)` gives the
   two parameter leaves. What is genuinely invisible is *attribution*: the leaf
   is keyed by modality and its value is that modality's joint info gain, so
   `("target",)`, `("mode",)` and `("obs:answer",)` produce a numerically
   identical `answer` leaf. A selector's effect is legible in the key **set**,
   never in a key's name or value. Separately, discrete-latent/continuous-
   observation *is* expressible as a switching LG observation — the discrete
   factor becomes a mode parent named after the observed factor in `A_deps`,
   and that factor must be carried as a `MixtureGaussian` with `mixture_k` set
   (`objectives/hybrid.py:424-431` says so in its error). The probe never saw
   this because no document mentions it.

Both corrections are folded into `docs/composing-systems.md`.
