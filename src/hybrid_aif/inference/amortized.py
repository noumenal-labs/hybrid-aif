"""The amortized inference path: recognition modules as a variable's inference strategy.

The path ``System.infer_states`` takes, by declaration type, for a neural state-space System.
AIF-free and System-free: functions take unpacked declarations and return arrays. Each
recognition entry is a q-side factor (child = the variable, parents = its declared conditioning
set), evaluated in declaration order. Two factor families live here:

``carry`` (:class:`~hybrid_aif.factors.Deterministic`)
    Computed forward, never inferred. Its own previous value is the map's carry argument, so
    ``B_deps`` names its other previous-step parents; its control node's value, if it reads
    one, is the last input (an RSSM's ``deter``).

``code`` (:class:`~hybrid_aif.distributions.MLPCategorical`)
    A factored-categorical one-hot code whose transition parents are read within the step
    (``deter_t``), so they must be declared earlier.

The belief carry is a tuple of per-factor values: a carry's vector, and a code's
``(num_vars, num_classes)`` probability tensor. :func:`head_parents` of it is the
``[deter, flat stoch]`` feature vector, with the expected one-hot where a sampled code would
sit. :func:`predict` without a key carries the probability tensor (the expected rollout); with
a key it carries a straight-through one-hot draw (the sampled rollout).

Design: docs/design/amortization.md A1(b), A10 disposition 2, A11
"""

from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.controls import ControlLayout
from hybrid_aif.distributions import (
    Gaussian,
    MixtureGaussian,
    MLPCategorical,
    MLPConditional,
    MLPPredictor,
)
from hybrid_aif.distributions.neural import LearnedVector, PredictorHead, straight_through_onehot
from hybrid_aif.factors import DerivedInitial, Deterministic, Recognition
from hybrid_aif.inference.sample import collapse

#: The two factor families of this path (see the module docstring).
CARRY = "carry"
CODE = "code"

#: Parent-address kinds inside a resolved :class:`RecognitionPlan`.
STATE = "state"
OBS = "obs"
RECOGNITION = "recognition"
EXTERNAL = "external"
#   ``EXTERNAL``: a vector this System does not declare, supplied per call as data; its
#   width is declared in ``System.external_inputs`` so ``parent_dims`` are still validated.


class RecognitionPlan(NamedTuple):
    """One :class:`~hybrid_aif.factors.Recognition` entry, resolved to indices.

    ``parents`` are ``(kind, index-or-name)`` addresses in the module's parent order; ``factor`` is
    the p-side factor index this module infers, or ``None`` for an auxiliary q-side variable.
    """

    variable: str
    factor: int | None
    parents: tuple[tuple[str, int | str], ...]


class AmortizedStructure(NamedTuple):
    """What :func:`validate_structure` resolved: families, value widths, q-side plan."""

    families: tuple[str, ...]
    widths: tuple[int, ...]
    plan: tuple[RecognitionPlan, ...]


def read_modalities(plan: tuple[RecognitionPlan, ...]) -> tuple[int, ...]:
    """The observation modalities the recognition modules read, in index order.

    Shared by :func:`recognize`, ``System.infer_states``'s mask handling and ``hybrid_aif.replay``:
    on this path they are present together or absent together.
    """
    return tuple(sorted({ref for e in plan for kind, ref in e.parents if kind == OBS}))


def collapse_codes(structure: AmortizedStructure, beliefs: tuple, key: Array) -> tuple:
    """The belief tuple with each code drawn to a one-hot; carries pass through.

    ``key`` is split per factor, carries included, so a factor's key does not move when another is
    added.
    """
    keys = jax.random.split(key, len(beliefs))
    return tuple(
        collapse(b, k) if fam == CODE else b
        for b, fam, k in zip(beliefs, structure.families, keys, strict=True)
    )


def _flat(value: Array) -> Array:
    """A factor's value as a parent vector: a code's flattened one-hot, a carry as is."""
    return jnp.reshape(jnp.asarray(value), (-1,))


def head_parents(beliefs: tuple[Array, ...], deps: tuple[int, ...]) -> tuple[Array, ...]:
    """The parent tuple a modality's head reads: its ``A_deps`` factors, flattened.

    Over ``A_deps = (deter, stoch)`` this is the ``[deter, flat stoch]`` feature vector.
    """
    return tuple(_flat(beliefs[g]) for g in deps)


def action_parent(
    f: int,
    action: Array | None,
    controlled: tuple[bool, ...],
    num_controls: tuple[int, ...],
    controls: ControlLayout | None,
) -> tuple[Array, ...]:
    """Factor ``f``'s control input, as a 0- or 1-tuple (empty when uncontrolled).

    The one-hot of a discrete control level, or the factor's block of the flat continuous value.
    """
    if not controlled[f]:
        return ()
    if action is None:
        raise ValueError(
            f"factor {f} is controlled, so predicting its next value needs an action "
            "(a_{t-1} for the step being taken)"
        )
    action = jnp.asarray(action)
    if controls is not None:
        return (controls.block(f, action).astype(jnp.float32),)
    #  one level per control node: a controlled factor's node is its rank among them
    return (jax.nn.one_hot(action[sum(controlled[:f])], num_controls[f]),)


# ------------------------------------------------------------------- validation


def _decl_width(f: int, B_f, D_f, facs: tuple[str, ...]) -> int:
    """A factor's value width, read off the declaration pair that fixes it."""
    if isinstance(B_f, Deterministic):
        if not isinstance(D_f, LearnedVector):
            raise ValueError(
                f"factor {facs[f]!r} has a Deterministic (carry) transition, so its "
                f"initial_state_prior is the episode-start *value* of the carry — a "
                f"LearnedVector (a learned initial), not a {type(D_f).__name__}. The "
                "carry has no density to hold a prior over (design: amortization.md A10 "
                "disposition 2, A11)."
            )
        return D_f.dim
    if not isinstance(D_f, DerivedInitial):
        raise ValueError(
            f"factor {facs[f]!r} has a categorical-code (MLPCategorical) transition, so its "
            f"initial code is read off that transition at the episode start — declare "
            f"DerivedInitial(), not a {type(D_f).__name__} (the code's prior at the initial "
            "carry; design: amortization.md A11)."
        )
    return B_f.flat_dim


def validate_structure(
    recognition: tuple[Recognition, ...] | None,
    recognition_params: Mapping[str, object],
    A: tuple,
    B: tuple,
    D: tuple,
    A_deps: tuple[tuple[int, ...], ...],
    B_deps: tuple[tuple[int, ...], ...],
    controlled: tuple[bool, ...],
    num_controls: tuple[int, ...],
    obs_targets: Mapping[str, object],
    state_targets: Mapping[str, object],
    terminal_targets: Mapping[str, object],
    modality_names: tuple[str, ...],
    factor_names: tuple[str, ...],
    controls: ControlLayout | None = None,
    external_inputs: tuple[tuple[str, int], ...] = (),
) -> AmortizedStructure:
    """Validate an amortized (neural state-space) declaration; resolve its structure.

    Enforces, every error naming the offending declaration:

    - **families**: every transition is a ``carry`` or ``code`` node, paired with its initial-prior
      form (:class:`LearnedVector` / :class:`~hybrid_aif.factors.DerivedInitial`);
    - **ancestrality**: a code's transition parents must be declared earlier (the error prints a
      valid order); a carry reads its parents at ``t−1`` and may not name itself in ``B_deps``;
    - **widths**: every map/head/module's parent widths against what its slot supplies;
    - **recognition**: modules and declarations correspond, parents resolve and are declared
      earlier, an amortized posterior has the shape of the transition it is KL-balanced against, no
      carry has a recognition entry, and every code has one unless the System declares no modality
      (prediction-only); no shared module objects and no amortized variable as a parent;
    - **designations**: an amortized System carries no target designations (a preference on this
      path is a scoring row).

    Design: docs/design/amortization.md A11
    """
    mods, facs = modality_names, factor_names
    n_factors = len(B_deps)
    if len(B) != n_factors or len(D) != n_factors:
        raise ValueError(
            f"declarations disagree with the System: {len(B)} transition(s) and {len(D)} "
            f"initial prior(s) for {n_factors} factor(s)"
        )

    families: list[str] = []
    for f, B_f in enumerate(B):
        if isinstance(B_f, Deterministic):
            families.append(CARRY)
        elif isinstance(B_f, MLPCategorical):
            families.append(CODE)
        else:
            raise ValueError(
                f"factor {facs[f]!r}: {type(B_f).__name__} is not a transition declaration of "
                "the amortized path, whose factors are a Deterministic carry node or an "
                "MLPCategorical code (design: amortization.md A11). A tabular or "
                "linear-Gaussian factor belongs to a System without recognition declarations; "
                "amortized recognition over a *tabular* variable has no registry row (its "
                "exact row is what infer_states already does). A declaration holding both a "
                "carry or code and such a factor is refused (recognition-beside-analytic.md R1)."
            )
    widths = tuple(_decl_width(f, B[f], D[f], facs) for f in range(n_factors))

    for f, fam in enumerate(families):
        deps = B_deps[f]
        if f in deps:
            raise ValueError(
                f"factor {facs[f]!r} names itself in B_deps. A carry node's own previous value "
                "is the map's carry argument (passed separately, not a parent) and a code's "
                "own previous value reaches it through the carry; B_deps names the *other* "
                "parents."
                if fam == CARRY
                else f"factor {facs[f]!r} names itself in B_deps, which is a within-step cycle: "
                "a code's transition parents are read at this step, not the previous one."
            )
        if fam == CODE:
            late = [facs[g] for g in deps if g >= f]
            if late:
                order = ", ".join(
                    repr(facs[i])
                    for i in sorted(range(n_factors), key=lambda i: families[i] == CODE)
                )
                raise ValueError(
                    f"within-step ordering: factor {facs[f]!r} (declaration index {f}) is a "
                    f"code whose transition parent(s) {late} are not declared earlier. A code "
                    "is sampled from its parents' values *at this step* (the code reads the carry "
                    "at t, not t-1), so parents come before children — D2's "
                    f"within-step rule, extended to a Deterministic parent. A valid order is: "
                    f"({order})."
                )
    # No transition-family check is needed here: both families of this path take their
    # action as an ordinary parent/input *vector*, so both accept a continuous one; the
    # widths are checked below.
    for f, fam in enumerate(families):
        supplied = tuple(widths[g] for g in B_deps[f])
        if controlled[f]:
            supplied += (controls.dim_of(f),) if controls is not None else (num_controls[f],)
        if fam == CARRY:
            declared = getattr(B[f].map, "input_dims", None)
            carry_dim = getattr(B[f].map, "deter_dim", None)
            if declared is not None and tuple(declared) != supplied:
                raise ValueError(
                    f"factor {facs[f]!r}: its carry map takes inputs of width {tuple(declared)} "
                    f"but the slot supplies {supplied} — the B_deps parents' values at t−1, then "
                    "the control node's value when the factor reads one (the transition slot's "
                    "parent order, amortization.md A1/A9)"
                )
            if carry_dim is not None and int(carry_dim) != widths[f]:
                raise ValueError(
                    f"factor {facs[f]!r}: its carry map carries {carry_dim} numbers but its "
                    f"initial_state_prior declares a {widths[f]}-dimensional initial value"
                )
        elif B[f].parent_dims != supplied:
            raise ValueError(
                f"factor {facs[f]!r}: its code transition declares parent_dims="
                f"{B[f].parent_dims} but the slot supplies parents of width {supplied} — the "
                "B_deps parents' values at *this* step, then the action when the factor is "
                "controlled (a code parent as its flattened one-hot)"
            )

    if len(A) != len(A_deps):
        raise ValueError(f"{len(A)} likelihood(s) for {len(A_deps)} modality/ies")
    for m, A_m in enumerate(A):
        if not isinstance(A_m, PredictorHead):
            raise ValueError(
                f"modality {mods[m]!r}: {type(A_m).__name__} is not a modality declaration of "
                "the amortized path, whose modalities are PredictorHead output heads (a vector "
                "decoder, a scalar reward head, a binary continuation head)"
            )
        if not A_deps[m]:
            raise ValueError(
                f"modality {mods[m]!r}: a predictor head reads at least one factor, got empty "
                "A_deps"
            )
        supplied = tuple(widths[g] for g in A_deps[m])
        if A_m.parent_dims != supplied:
            raise ValueError(
                f"modality {mods[m]!r}: its head declares parent_dims={A_m.parent_dims} but "
                f"the slot supplies parents of width {supplied} — one per factor named in "
                "A_deps, in that order (a code as its flattened one-hot)"
            )

    designated = [
        f"{what}[{name!r}]"
        for what, table, allowed in (
            ("obs_targets", obs_targets, set(mods)),
            ("state_targets", state_targets, set(facs)),
            ("terminal_targets", terminal_targets, set(facs)),
        )
        for name in table
        if name in allowed
    ]
    if designated:
        raise NotImplementedError(
            f"designation(s) {sorted(designated)} on an amortized System: scoring a preference "
            "over a predicted-through-a-net observation, a deterministic carry or a one-hot "
            "code needs preference machinery this path does not have: a net has no closed-form "
            "pushforward of the belief, so there is nothing to take a KL against (amortization.md "
            "A1). The scoring rows are how an amortized System expresses preferences — "
            "System(terms=('predicted_reward:obs:<modality>', '<a row scoring the rollout "
            "against a model of its own>', ...)) with their parameters in SystemParams.scoring "
            "(A12 item 4). Raising rather than "
            "ignoring: a silently dropped designation is the 'toggle lies' failure."
        )

    plan = _resolve_recognition(
        recognition,
        recognition_params,
        tuple(families),
        widths,
        A,
        mods,
        facs,
        _external_widths(external_inputs),
    )
    return AmortizedStructure(families=tuple(families), widths=widths, plan=plan)


def _external_widths(external_inputs: tuple[tuple[str, int], ...]) -> dict[str, int]:
    """``System.external_inputs`` as a mapping, with the declaration validated.

    A tuple of ``(name, width)`` pairs on the static field; duplicate names raise.
    """
    names = [name for name, _ in external_inputs]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise ValueError(
            f"external_inputs declares {duplicated} twice; one width per external input"
        )
    bad = sorted(name for name, width in external_inputs if int(width) <= 0)
    if bad:
        raise ValueError(f"external_inputs {bad} declare a non-positive width")
    return {name: int(width) for name, width in external_inputs}


def check_recognition_entries(
    recognition: tuple[Recognition, ...] | None,
    recognition_params: Mapping[str, object],
) -> tuple[Recognition, ...]:
    """The checks on a recognition declaration that do not depend on the path it is read on.

    One entry per variable, the structural declaration and the parameter slot naming the same
    variables, and no module object held by two entries. Returns the declared entries.
    """
    declared = tuple(recognition or ())
    names = [r.variable for r in declared]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        raise ValueError(
            f"recognition: variable(s) {duplicated} are declared twice; one recognition "
            "module per variable (a variable's inference strategy is single-valued)"
        )
    missing = sorted(set(names) - set(recognition_params))
    extra = sorted(set(recognition_params) - set(names))
    if missing or extra:
        raise ValueError(
            "recognition: the structural declaration and the parameter slot disagree — "
            f"declared without a module: {missing}; module without a declaration: {extra}. "
            "System.recognition says which variables are amortized and what each module "
            "conditions on; SystemParams.recognition holds the modules (design: "
            "amortization.md A11 — the registry names the strategy, the slot holds its "
            "parameters)."
        )

    shared: dict[int, str] = {}
    for entry in declared:
        module = recognition_params[entry.variable]
        owner = shared.setdefault(id(module), entry.variable)
        if owner != entry.variable:
            raise ValueError(
                f"recognition: {entry.variable!r} and {owner!r} hold the *same module object*. "
                "That looks like a shared torso and is not one: the slot is a pytree, so "
                "flattening duplicates the leaves and the first optimizer step forks them into "
                "two independent parameter sets that then drift apart silently. Give each entry "
                "its own module (structurally identical copies are fine — they are genuinely "
                "separate parameters), and note that *declared* shared q-side structure is a "
                "feature of the generic q-graph layer this slot is the restricted syntax of, "
                "not something to smuggle in by aliasing (design: amortization.md A11, "
                "refinement (ii))."
            )

    return declared


def check_analytic_parents(
    recognition: tuple[Recognition, ...] | None,
    modality_names: tuple[str, ...],
    external_inputs: tuple[tuple[str, int], ...] = (),
    factor_names: tuple[str, ...] = (),
) -> None:
    """Refuse a recognition entry whose conditioning set the analytic path cannot read.

    Parents are ``obs:<modality>``, ``external:<name>`` or ``recognition:<name>``; an empty set or a
    bare factor name is refused (recognition-beside-analytic.md R2).
    """
    externals = _external_widths(external_inputs)
    declared = {r.variable for r in recognition or ()}
    for entry in recognition or ():
        if not entry.parents:
            raise ValueError(
                f"recognition:{entry.variable}: an empty parent set is refused on the analytic "
                "path; a measurement conditions on observations, external inputs or earlier "
                "recognition variables (recognition-beside-analytic.md R2)"
            )
        for parent in entry.parents:
            parts = parent.split(":")
            kind = parts[0] if len(parts) == 2 else None
            name = parts[1] if kind is not None else None
            ok = (
                (kind == OBS and name in modality_names)
                or (kind == EXTERNAL and name in externals)
                or (kind == RECOGNITION and name in declared and name != entry.variable)
            )
            if ok and kind == RECOGNITION and name in factor_names:
                raise ValueError(
                    f"recognition:{entry.variable}: parent {parent!r} is the output of the entry "
                    f"for the factor {name!r}; an amortized variable is not a parent of another "
                    "entry's module (the restricted syntax is disjoint stars over observations, "
                    "externals and auxiliary variables; recognition-beside-analytic.md R2)"
                )
            if not ok:
                raise ValueError(
                    f"recognition:{entry.variable}: parent {parent!r} is not readable on the "
                    "analytic path, whose parents are 'obs:<modality>' (a declared modality), "
                    "'external:<name>' (a declared external input) and 'recognition:<name>' "
                    "(another entry); a bare factor name reads the prior's prediction, which "
                    "would count the prior twice (recognition-beside-analytic.md R2)"
                )
    analytic_order(recognition)


#: The two kinds of measurement a recognition entry sends on the analytic path.
GAUSSIAN_KIND = "gaussian"
CATEGORICAL_KIND = "categorical"


class Measurement(NamedTuple):
    """One recognition entry's message to its factor, evaluated on the tick's data.

    ``kind`` is :data:`GAUSSIAN_KIND` (``loc``, ``scale``: a diagonal Gaussian read as an
    observation of the factor through the identity map) or :data:`CATEGORICAL_KIND` (``row``: a
    log-probability row over the factor's levels). ``present`` is ``True``, or a traced 0-d bool;
    a statically absent entry produces no :class:`Measurement`.
    """

    variable: str
    factor: int
    kind: str
    loc: Array | None
    scale: Array | None
    row: Array | None
    present: bool | Array


def analytic_order(recognition: tuple[Recognition, ...] | None) -> tuple[Recognition, ...]:
    """The entries in an evaluation order derived from their ``recognition:<name>`` parents.

    Each entry follows every entry it reads; entries with no ordering between them keep their
    declaration order. A cycle is refused (model-description-0.3.md 18.5c: the order is derived
    from the explicit parents, so a cycle has none).
    """
    entries = tuple(recognition or ())
    declared = {e.variable for e in entries}
    reads = {
        e.variable: {p.split(":", 1)[1] for p in e.parents if p.startswith(f"{RECOGNITION}:")}
        & declared
        for e in entries
    }
    ordered: list[Recognition] = []
    done: set[str] = set()
    remaining = list(entries)
    while remaining:
        ready = next((e for e in remaining if reads[e.variable] <= done), None)
        if ready is None:
            stuck = sorted(e.variable for e in remaining)
            raise ValueError(
                f"recognition: the entries {stuck} read one another through "
                "'recognition:<name>' parents in a cycle; the evaluation order is derived from "
                "the explicit parents, so a cycle has none (model-description-0.3.md 18.5c)"
            )
        ordered.append(ready)
        done.add(ready.variable)
        remaining.remove(ready)
    return tuple(ordered)


def _all_present(flags: list[bool | Array]) -> bool | Array:
    """The conjunction of presence flags: ``False`` if any is, else ``True`` or a traced bool."""
    if any(f is False for f in flags):
        return False
    traced = [jnp.asarray(f, bool) for f in flags if f is not True]
    if not traced:
        return True
    return traced[0] if len(traced) == 1 else jnp.all(jnp.stack(traced))


def _module_fits(module: object, kind: str, width: int | None) -> bool:
    """Whether ``module`` is the family a factor of ``kind`` takes (width checked when known)."""
    if kind == GAUSSIAN_KIND:
        return isinstance(module, MLPConditional) and width in (None, module.child_dim)
    if isinstance(module, MLPCategorical):
        return module.num_vars == 1 and width in (None, module.num_classes)
    return isinstance(module, MLPPredictor) and width in (None, module.out_dim)


def _output_width(module: object) -> int | None:
    """The width of the vector an auxiliary module outputs, where its type declares one."""
    if isinstance(module, MLPConditional):
        return module.child_dim
    if isinstance(module, MLPCategorical):
        return module.flat_dim
    return getattr(module, "out_dim", None)


def check_analytic_recognition(
    recognition: tuple[Recognition, ...] | None,
    recognition_params: Mapping[str, object],
    modality_names: tuple[str, ...],
    factor_names: tuple[str, ...],
    control_names: tuple[str, ...],
    kinds: tuple[str, ...],
    widths: tuple[int | None, ...],
    observation_widths: tuple[int | None, ...],
    external_inputs: tuple[tuple[str, int], ...] = (),
    categorical_observations: tuple[bool, ...] = (),
) -> None:
    """The structural checks on a recognition declaration read on the analytic path.

    Parents resolve and the entries have an evaluation order (:func:`check_analytic_parents`); a
    variable is a state factor or an auxiliary name, never a control node or a modality; an
    auxiliary some entry reads; each factor's module fits its family (``kinds``, ``widths``); and
    each module's ``parent_dims`` match the widths of its parents (an observation's width where
    known, an external's declared width, an earlier entry's output width). Refusals cite
    recognition-beside-analytic.md R2. A categorical observation (a level index, no vector) is
    refused as a parent; ``categorical_observations`` flags the modalities that are one.
    """
    check_analytic_parents(recognition, modality_names, external_inputs, factor_names)
    entries = tuple(recognition or ())
    for entry in entries:
        for parent in entry.parents:
            kind, name = parent.split(":", 1)
            if (
                kind == OBS
                and name in modality_names
                and categorical_observations
                and categorical_observations[modality_names.index(name)]
            ):
                raise ValueError(
                    f"recognition:{entry.variable}: the parent {parent!r} is a categorical "
                    "observation, a level index that is no vector for a module to read; a "
                    "module's parents are continuous observations, external inputs and other "
                    "entries (recognition-beside-analytic.md R2)"
                )
    for entry in entries:
        if entry.variable in control_names or entry.variable in modality_names:
            raise ValueError(
                f"recognition:{entry.variable}: the name is a "
                f"{'control node' if entry.variable in control_names else 'modality'}, not a "
                "state factor or an auxiliary variable; an entry names a factor it measures or "
                "a new auxiliary name (recognition-beside-analytic.md R2)"
            )
    read = {p.split(":", 1)[1] for e in entries for p in e.parents if p.startswith("recognition:")}
    externals = _external_widths(external_inputs)
    out_width: dict[str, int | None] = {}
    for entry in analytic_order(entries):
        module = recognition_params[entry.variable]
        if entry.variable in factor_names:
            f = factor_names.index(entry.variable)
            if not _module_fits(module, kinds[f], widths[f]):
                raise _family_refusal(entry.variable, module, kinds[f], widths[f])
            out_width[entry.variable] = widths[f]
        else:
            if entry.variable not in read:
                raise ValueError(
                    f"recognition:{entry.variable}: the name is neither a state factor "
                    f"{list(factor_names)} nor read by any entry as 'recognition:"
                    f"{entry.variable}'; an auxiliary no entry reads is inert, which is most "
                    "likely a mistyped factor name (recognition-beside-analytic.md R2)"
                )
            out_width[entry.variable] = _output_width(module)
        supplied: list[int | None] = []
        for parent in entry.parents:
            kind, name = parent.split(":", 1)
            if kind == OBS:
                supplied.append(observation_widths[modality_names.index(name)])
            elif kind == EXTERNAL:
                supplied.append(externals[name])
            else:
                supplied.append(out_width[name])
        declared = tuple(getattr(module, "parent_dims", ()))
        if len(declared) != len(supplied) or any(
            w is not None and w != d for w, d in zip(supplied, declared, strict=True)
        ):
            raise ValueError(
                f"recognition:{entry.variable}: its module declares parent_dims={declared} but "
                f"the parents {list(entry.parents)} supply widths {tuple(supplied)} (None where "
                "the parent's width is not a vector); the module reads its parents in the "
                "declared order (recognition-beside-analytic.md R2)"
            )


def _family_refusal(variable: str, module: object, kind: str, width: int) -> ValueError:
    want = (
        f"an MLPConditional with child_dim={width} (a diagonal Gaussian over the factor's "
        "dimensions)"
        if kind == GAUSSIAN_KIND
        else f"an MLPCategorical (one variable of {width} classes) or an MLPPredictor with "
        f"out_dim={width} (a probability vector over the levels)"
    )
    return ValueError(
        f"recognition:{variable}: a {type(module).__name__} does not fit the {kind} factor "
        f"{variable!r} on the analytic path, whose module for it is {want} "
        "(recognition-beside-analytic.md R2)"
    )


def _auxiliary_output(module: object, parents: tuple[Array, ...]) -> Array:
    """An auxiliary variable's value: the module's output vector (a conditional's mean, a
    categorical's flattened probabilities)."""
    if isinstance(module, MLPConditional):
        return module.diag_moments(parents)[0]
    if isinstance(module, MLPCategorical):
        return _flat(module.probs(parents))
    return jnp.asarray(module(parents))


def _measurement_of(
    variable: str, module: object, parents: tuple[Array, ...], kind: str, width: int
) -> tuple[Array | None, Array | None, Array | None]:
    """``(loc, scale, row)`` of a factor entry's output, with the module checked against the
    factor's family and width (R2)."""
    if not _module_fits(module, kind, width):
        raise _family_refusal(variable, module, kind, width)
    if kind == GAUSSIAN_KIND:
        loc, scale = module.diag_moments(parents)
        return loc, scale, None
    logits = module.logits(parents)[0] if isinstance(module, MLPCategorical) else module(parents)
    return None, None, jax.nn.log_softmax(logits, axis=-1)


def analytic_recognize(
    order: tuple[Recognition, ...],
    recognition_params: Mapping[str, object],
    observations: tuple[Array | None, ...],
    externals: Mapping[str, Array] | None,
    present: tuple[bool | Array, ...] | None,
    modality_names: tuple[str, ...],
    factor_names: tuple[str, ...],
    kinds: tuple[str, ...],
    widths: tuple[int, ...],
) -> tuple[Measurement, ...]:
    """Evaluate the recognition entries once, in ``order``, for the analytic path.

    Each module is applied to its parents: ``obs:<m>`` the tick's observation, ``external:<n>`` the
    supplied vector, ``recognition:<n>`` an earlier entry's output. A factor's entry yields a
    :class:`Measurement` (``kinds[f]`` and ``widths[f]`` give the factor's family and width, and
    the module must fit them); an auxiliary variable's entry yields its output vector for later
    entries. An entry whose observed parents are not all present contributes nothing: a static
    absence produces no measurement, a traced one is carried in ``Measurement.present``.
    """
    factor_of = {name: f for f, name in enumerate(factor_names)}
    supplied = {} if externals is None else externals
    values: dict[str, Array] = {}
    presence: dict[str, bool | Array] = {}
    out: list[Measurement] = []
    for entry in order:
        flags: list[bool | Array] = []
        for parent in entry.parents:
            kind, name = parent.split(":", 1)
            if kind == OBS:
                flags.append(True if present is None else present[modality_names.index(name)])
            elif kind == RECOGNITION:
                flags.append(presence[name])
        here = _all_present(flags)
        presence[entry.variable] = here
        if here is False:
            continue
        parents: list[Array] = []
        for parent in entry.parents:
            kind, name = parent.split(":", 1)
            if kind == OBS:
                obs = observations[modality_names.index(name)]
                if obs is None:
                    raise ValueError(
                        f"recognition:{entry.variable} conditions on observation {name!r}, "
                        "whose value is None while it is marked present; pass its datum, or "
                        "mark the modality absent with present={name: False}"
                    )
                parents.append(jnp.asarray(obs))
            elif kind == EXTERNAL:
                if name not in supplied:
                    raise ValueError(
                        f"recognition:{entry.variable} conditions on external input {name!r} "
                        f"and the externals mapping carries {sorted(supplied)}; an external "
                        "parent is supplied per call and has no default"
                    )
                parents.append(jnp.asarray(supplied[name]))
            else:
                parents.append(_flat(values[name]))
        module = recognition_params[entry.variable]
        f = factor_of.get(entry.variable)
        if f is None:
            values[entry.variable] = _auxiliary_output(module, tuple(parents))
            continue
        loc, scale, row = _measurement_of(
            entry.variable, module, tuple(parents), kinds[f], widths[f]
        )
        values[entry.variable] = loc if row is None else jnp.exp(row)
        out.append(Measurement(entry.variable, f, kinds[f], loc, scale, row, here))
    return tuple(out)


def gaussian_message_expectation(q: Gaussian, y: Array, scale: Array) -> Array:
    """``E_q[log m(s)]`` for the measurement ``m(s) = N(y; s, R)``, ``R = diag(scale**2)``, under a
    Gaussian belief ``q`` with mean ``mu`` and covariance ``S``:

        log N(y; mu, R) - 1/2 tr(R^-1 S)
    """
    d = y.shape[-1]
    log_density = (
        -0.5 * jnp.sum(((y - q.loc) / scale) ** 2, axis=-1)
        - jnp.sum(jnp.log(scale), axis=-1)
        - 0.5 * d * jnp.log(2.0 * jnp.pi)
    )
    trace = jnp.sum(jnp.sum(q.chol**2, axis=-1) / scale**2, axis=-1)
    return log_density - 0.5 * trace


def mixture_message_expectation(q: MixtureGaussian, y: Array, scale: Array) -> Array:
    """``E_q[log m(s)]`` under a mixture belief: each component's
    :func:`gaussian_message_expectation` weighted by the belief's component weights (a dead
    component weighs zero)."""
    weights = jax.nn.softmax(q.log_weights, axis=-1)  # gauge-free; dead components weigh zero
    per_component = gaussian_message_expectation(q.components, y, scale)
    return jnp.sum(jnp.where(weights > 0, weights * per_component, 0.0), axis=-1)


def covered_modalities(
    recognition: tuple[Recognition, ...] | None, modality_names: tuple[str, ...]
) -> frozenset[int]:
    """The modalities the recognition entries cover, by index: every ``obs:<m>`` parent of an
    entry, which includes those reached through ``recognition:<name>`` parents (an entry that
    reads another entry covers what that one reads, and the other entry's own ``obs:`` parents
    are in the union). The engine does not condition on a covered modality; the entry's message
    does (design: recognition-beside-analytic.md L3).
    """
    return frozenset(
        modality_names.index(p.split(":", 1)[1])
        for e in (recognition or ())
        for p in e.parents
        if p.startswith(f"{OBS}:") and p.split(":", 1)[1] in modality_names
    )


def expected_gaussian_nll(
    locs: Array, chols: Array, H: Array, c: Array, R_chol: Array, y: Array, weights: Array
) -> Array:
    """``- E_q[log N(y; H x + c, R R^T)]`` under a Gaussian belief per component, weighted.

    ``locs`` ``(K, d)`` and ``chols`` ``(K, d, d)`` are the components of the belief; ``H``
    ``(K, M, d_obs, d)``, ``c`` ``(K, M, d_obs)`` and ``R_chol`` ``(K, M, d_obs, d_obs)`` the
    observation model of each component under each of ``M`` mode configurations; ``weights``
    ``(K, M)`` the joint weight of each pair (zero weighs nothing). Each pair contributes the
    closed-form expectation of the quadratic,
    ``- log N(y; H mu + c, R) + 1/2 tr(R^-1 H Sigma H^T)``.
    """
    d_obs = y.shape[-1]

    def pair(mu, chol, H, c, R):
        z = jax.scipy.linalg.solve_triangular(R, y - (H @ mu + c), lower=True)
        A = jax.scipy.linalg.solve_triangular(R, H @ chol, lower=True)
        return 0.5 * (d_obs * jnp.log(2.0 * jnp.pi) + jnp.sum(z**2) + jnp.sum(A**2)) + jnp.sum(
            jnp.log(jnp.diagonal(R))
        )

    per_pair = jax.vmap(jax.vmap(pair, (None, None, 0, 0, 0)), (0, 0, 0, 0, 0))(
        locs, chols, H, c, R_chol
    )
    return jnp.sum(jnp.where(weights > 0, weights * per_pair, 0.0))


def categorical_message_log_evidence(prior_log_probs: Array, log_row: Array) -> Array:
    """``log Z = log sum_k q_prior_k r_k`` for the likelihood row ``r`` (``log_row`` is ``log r``)
    against the belief ``q_prior`` the message is applied to."""
    return jax.scipy.special.logsumexp(prior_log_probs + log_row, axis=-1)


def categorical_message_expectation(probs: Array, log_row: Array) -> Array:
    """``E_q[log m] = sum_k q_k log r_k`` for the belief ``q`` (``probs``) and the row ``r``."""
    return jnp.sum(jnp.where(probs > 0, probs * log_row, 0.0), axis=-1)


def _resolve_recognition(
    recognition: tuple[Recognition, ...] | None,
    recognition_params: Mapping[str, object],
    families: tuple[str, ...],
    widths: tuple[int, ...],
    A: tuple,
    mods: tuple[str, ...],
    facs: tuple[str, ...],
    externals: Mapping[str, int] = {},  # noqa: B006 (read-only, never mutated)
) -> tuple[RecognitionPlan, ...]:
    """The recognition declaration and its parameter slot, resolved to a q-side plan."""
    declared = check_recognition_entries(recognition, recognition_params)

    factor_of = {name: f for f, name in enumerate(facs)}
    for entry in declared:
        f = factor_of.get(entry.variable)
        if f is not None and families[f] == CARRY:
            raise ValueError(
                f"recognition: {entry.variable!r} is a Deterministic carry factor, whose value "
                "is computed forward and carried, never inferred (rule='carry'). An amortized "
                "posterior over it has no referent, so drop the recognition entry (design: "
                "amortization.md A10 disposition 2)."
            )

    # The amortized variables that are p-side factors: the syntax is disjoint
    # per-variable stars, so none of them may appear in a conditioning set.
    amortized_factors = {r.variable for r in declared if r.variable in factor_of}
    plan: list[RecognitionPlan] = []
    seen: dict[str, int] = {}  # recognition variable -> value width
    for entry in declared:
        module = recognition_params[entry.variable]
        f = factor_of.get(entry.variable)
        if f is not None:
            if not isinstance(module, MLPCategorical):
                raise ValueError(
                    f"recognition:{entry.variable}: the amortized posterior over a code factor "
                    f"is an MLPCategorical (the same family as its transition prior — the two "
                    f"are what the balanced KL trains), got {type(module).__name__}"
                )
            width = widths[f]
        else:
            width = getattr(module, "out_dim", None)
            if width is None:
                raise ValueError(
                    f"recognition:{entry.variable}: an auxiliary recognition variable's module "
                    f"must declare its output width (out_dim), and {type(module).__name__} does "
                    "not; declare the variable as a factor, or use an MLPPredictor/PredictorHead"
                )
        supplied: list[int] = []
        addresses: list[tuple[str, int | str]] = []
        for parent in entry.parents:
            parts = parent.split(":")
            if (len(parts) == 1 or parts[0] == RECOGNITION) and parts[-1] in amortized_factors:
                raise NotImplementedError(
                    f"recognition:{entry.variable} conditions on {parent!r}, which is itself an "
                    "amortized variable. This slot is the *restricted* syntax of the q-side "
                    "graph — disjoint per-variable stars over p-side values, observations and "
                    "auxiliary encodings — so a q-factor over another q-factor's child "
                    "(structured Q, q(a|b)q(b)) is not supported here. Condition each "
                    "q-factor on p-side values, observations or auxiliary encodings instead "
                    "(design: amortization.md A11, refinement (iii))."
                )
            if len(parts) == 1:
                g = factor_of.get(parent)
                if g is None:
                    raise ValueError(
                        f"recognition:{entry.variable}: unknown parent {parent!r}; a bare parent "
                        f"name is a factor, and the factors are {list(facs)} (an observation is "
                        "'obs:<modality>', an earlier recognition variable "
                        "'recognition:<name>')"
                    )
                supplied.append(widths[g])
                addresses.append((STATE, g))
            elif parts[0] == OBS and len(parts) == 2:
                if parts[1] not in mods:
                    raise ValueError(
                        f"recognition:{entry.variable}: unknown observation parent {parent!r}; "
                        f"the modalities are {list(mods)}"
                    )
                m = mods.index(parts[1])
                supplied.append(A[m].target_dim)
                addresses.append((OBS, m))
            elif parts[0] == EXTERNAL and len(parts) == 2:
                if parts[1] not in externals:
                    raise ValueError(
                        f"recognition:{entry.variable}: unknown external input {parent!r}; "
                        f"this System declares external_inputs={sorted(externals)}. An "
                        "external parent is a vector supplied per call (A11: 'a parent may "
                        "be an externally supplied vector'), and its width is declared "
                        "statically so the module's parent_dims can still be validated."
                    )
                supplied.append(externals[parts[1]])
                addresses.append((EXTERNAL, parts[1]))
            elif parts[0] == RECOGNITION and len(parts) == 2:
                if parts[1] not in seen:
                    raise ValueError(
                        f"recognition:{entry.variable}: parent {parent!r} is not an *earlier* "
                        f"recognition variable; the ones declared before it are "
                        f"{sorted(seen)} (recognition modules are evaluated in declaration "
                        "order, the q-side twin of parents-before-children)"
                    )
                supplied.append(seen[parts[1]])
                addresses.append((RECOGNITION, parts[1]))
            else:
                raise ValueError(
                    f"recognition:{entry.variable}: unknown parent address {parent!r}; a "
                    "conditioning set names factors ('<factor>'), observations "
                    "('obs:<modality>'), earlier recognition variables "
                    "('recognition:<name>') and externally supplied vectors "
                    "('external:<name>', declared in System.external_inputs)"
                )
        if tuple(getattr(module, "parent_dims", ())) != tuple(supplied):
            raise ValueError(
                f"recognition:{entry.variable}: its module declares parent_dims="
                f"{getattr(module, 'parent_dims', None)} but the declared conditioning set "
                f"{list(entry.parents)} supplies parents of width {tuple(supplied)}"
            )
        seen[entry.variable] = width
        plan.append(RecognitionPlan(variable=entry.variable, factor=f, parents=tuple(addresses)))

    unamortized = [facs[f] for f, fam in enumerate(families) if fam == CODE and facs[f] not in seen]
    if unamortized and A:
        raise NotImplementedError(
            f"factor(s) {unamortized} are categorical codes with no recognition entry, so they "
            "fall back to the structured/exact inference row — which for a code has no "
            "implementation: moment-matched (unscented) filtering through a categorical head is "
            "A1(a)'s row, designed and unbuilt (amortization.md A1(a), A10 disposition 10). "
            "Declare "
            "System(recognition=(Recognition(variable, parents), ...)) with the module in "
            "SystemParams.recognition — A1(b)."
        )
    return tuple(plan)


# ------------------------------------------------------------------- the faces


def initial_state(
    B: tuple,
    D: tuple,
    B_deps: tuple[tuple[int, ...], ...],
    families: tuple[str, ...],
) -> tuple[Array, ...]:
    """The episode-start belief tuple: carries from their declarations, codes derived.

    A carry's initial value is its :class:`LearnedVector`, activated; a code's is its transition at
    the already-initialized within-step parents, as a probability tensor.
    """
    out: list[Array] = [jnp.zeros(())] * len(families)
    for f, fam in enumerate(families):
        if fam == CARRY:
            out[f] = D[f]()
        else:
            out[f] = B[f].probs(tuple(_flat(out[g]) for g in B_deps[f]))
    return tuple(out)


def _predict(
    B: tuple,
    B_deps: tuple[tuple[int, ...], ...],
    families: tuple[str, ...],
    controlled: tuple[bool, ...],
    num_controls: tuple[int, ...],
    controls: ControlLayout | None,
    beliefs: tuple[Array, ...],
    action: Array | None,
    key: Array | None,
) -> tuple[tuple[Array, ...], dict[int, Array]]:
    """:func:`predict`, also returning each code's logits (``{factor: logits}``)."""
    codes = [f for f, fam in enumerate(families) if fam == CODE]
    if key is None:
        keys: dict[int, Array] = {}
    elif len(codes) == 1:  # one code: the key is its key (the trainer's own convention)
        keys = {codes[0]: key}
    else:
        keys = dict(zip(codes, jax.random.split(key, len(codes)), strict=True))
    new = list(beliefs)
    logits: dict[int, Array] = {}
    for f, fam in enumerate(families):
        if fam == CARRY:
            inputs = tuple(_flat(beliefs[g]) for g in B_deps[f])
            new[f] = B[f](
                beliefs[f],
                *inputs,
                *action_parent(f, action, controlled, num_controls, controls),
            )
        else:
            logits[f] = B[f].logits(tuple(_flat(new[g]) for g in B_deps[f]))
            if key is None:
                new[f] = jax.nn.softmax(logits[f], axis=-1)
            else:
                new[f] = straight_through_onehot(keys[f], logits[f])
    return tuple(new), logits


def predict(
    B: tuple,
    B_deps: tuple[tuple[int, ...], ...],
    families: tuple[str, ...],
    controlled: tuple[bool, ...],
    num_controls: tuple[int, ...],
    controls: ControlLayout | None,
    beliefs: tuple[Array, ...],
    action: Array | None,
    key: Array | None = None,
) -> tuple[Array, ...]:
    """One open-loop step of the belief tuple, as the transition seam sees it.

    Carries advance from the previous step's values (plus the action); codes are then read off
    their transitions at this step's parents, in declaration order. No observation enters. Keyless,
    a code's belief is its probability tensor (the expected rollout); keyed, a straight-through
    one-hot draw from the same logits (the sampled rollout). With one code the key is used as
    given; with several it is split per code in declaration order.
    """
    return _predict(B, B_deps, families, controlled, num_controls, controls, beliefs, action, key)[
        0
    ]


def recognize(
    plan: tuple[RecognitionPlan, ...],
    recognition_params: Mapping[str, object],
    priors: tuple[Array, ...],
    observations: tuple[Array | None, ...],
    externals: Mapping[str, Array] | None,
    modality_names: tuple[str, ...],
) -> tuple[Array, ...]:
    """Run the recognition modules, replacing the amortized variables' beliefs.

    ``priors`` is the predicted belief tuple (:func:`predict`'s output), so its carries are already
    this tick's; the modules read them, the observation and each other, in declaration order.
    ``externals`` supplies the :data:`EXTERNAL` parents as data; a declared external with no value
    raises. ``modality_names`` names the modalities in error sentences.
    """
    out = list(priors)
    values: dict[str, Array] = {}
    supplied = {} if externals is None else externals
    for entry in plan:
        parents = []
        for kind, ref in entry.parents:
            if kind == STATE:
                parents.append(_flat(out[ref]))
            elif kind == EXTERNAL:
                if ref not in supplied:
                    raise ValueError(
                        f"recognition:{entry.variable} conditions on external input {ref!r} "
                        f"and the externals mapping carries {sorted(supplied)}; an external "
                        "parent is supplied per call and has no default (design: "
                        "amortization.md A11)"
                    )
                parents.append(jnp.asarray(supplied[ref]))
            elif kind == OBS:
                obs = observations[ref]
                if obs is None:
                    raise ValueError(
                        f"recognition:{entry.variable} conditions on observation "
                        f"{modality_names[ref]!r}, whose "
                        "value is None; only a modality no recognition module reads may be None, "
                        "a conditioned-on one may not"
                    )
                parents.append(jnp.asarray(obs))
            else:
                parents.append(_flat(values[ref]))
        module = recognition_params[entry.variable]
        if entry.factor is None:
            values[entry.variable] = module(tuple(parents))
        else:
            out[entry.factor] = module.probs(tuple(parents))
            values[entry.variable] = out[entry.factor]
    return tuple(out)
