"""The key grammar and its resolution against a declared System.

One namespace serves every selector a System carries: ``learn``'s per-slot rates, the ``epistemic``
union, the novelty leaves' block keys and the ``terms=`` rows. Two grammars live here: the
*variable* grammar (``"pose"``, ``"obs:cue"``, ``"likelihood:pose:noise"``, with wildcards) and the
``terms=`` *row* grammar (``"row"``, ``"row:<slot>:<name>"``, ``"row:obs:<modality>"``), whose tail
is the variable grammar. Every function takes the ``System`` (and its ``SystemParams`` where
declarations matter) explicitly, and ``System`` keeps a thin delegating method for each. Resolution
is static Python at trace time; rate values may be traced and are never inspected.

Design: docs/design/per-variable-infogain.md V1–V11; docs/design/amortization.md A11 A12.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
from jax import Array

from hybrid_aif.factors import (
    LGObservation,
    LinearGaussianObservation,
    LinearGaussianObservationMNIW,
    LinearGaussianTransition,
    TabularObservation,
)
from hybrid_aif.inference import amortized as _amortized
from hybrid_aif.inference.emission import is_emission
from hybrid_aif.learning.slots import slot_learnable as _slot_learnable
from hybrid_aif.objectives import hybrid as _hybrid
from hybrid_aif.objectives import registry

if TYPE_CHECKING:
    from hybrid_aif.system import System, SystemParams


# ------------------------------------------------------ what a declaration can learn


def _param_blocks(decl) -> tuple[str, ...]:
    """The independently-addressable posterior blocks of a learnable declaration.

    These are the legal third key segments, in canonical order. A
    :class:`~hybrid_aif.factors.LinearGaussianObservation` with both a learnable noise and a
    ``MatrixNormal`` head has two, ``noise`` and ``head``; ``LinearGaussianObservationMNIW`` (one
    joint posterior), ``ConditionalDirichlet`` and ``MLPConditional`` have none.
    Design: docs/design/per-variable-infogain.md V1.
    """
    if isinstance(decl, LinearGaussianObservationMNIW):
        return ()
    if isinstance(decl, LinearGaussianObservation):
        return tuple(
            block
            for block, learnable in (("noise", decl.learnable_noise), ("head", decl.learnable_head))
            if learnable
        )
    return ()


# ----------------------------------------------------------------------- the grammar


class _EpistemicVar(NamedTuple):
    """One node of the epistemic namespace.

    ``kind`` is ``"state"`` (``index`` a factor), ``"dim"`` (``index`` a continuous factor,
    ``block`` the name of one of its dimensions), ``"obs"`` (``index`` a modality) or ``"param"``
    (``index`` a declaration index in ``slot``; ``block`` one posterior block, or ``None`` for the
    declaration's group).
    """

    kind: str
    index: int
    slot: str | None = None
    block: str | None = None


_SLOT_WORDS = ("likelihood", "transition", "initial_state_prior", "recognition", "scoring")
_KEY_KINDS = (
    "initial_state_prior",
    "likelihood",
    "obs",
    "recognition",
    "scoring",
    "transition",
)
#   the reserved first segments of a qualified key: the five parameter slots plus the ``obs``
#   prefix. Any other first segment is a continuous factor's name (``pose:b1``, one of its
#   dimensions), so no factor may be named after one of these. A bare (single-segment) key is a
#   state variable.
_DIM_NAME = re.compile(r"dim(0|[1-9][0-9]*)")
#   the generated name of an undeclared dimension, ``dim<position>``
_DIM_RESERVED = (":", ".", "*", "[", "]")
#   characters a declared dimension name may not contain


def _is_pattern(key: str) -> bool:
    """Whether a key is a wildcard pattern (some segment is exactly ``*``)."""
    return "*" in key.split(":")


def _specificity(pattern: str) -> int:
    """A pattern's rank on the specificity ladder: how many of its segments are literal."""
    return sum(1 for seg in pattern.split(":") if seg != "*")


def _pattern_matches(pattern: str, key: str) -> bool:
    """Segment-wise match: ``*`` matches one whole segment; segment counts must be equal."""
    segs, parts = pattern.split(":"), key.split(":")
    if len(segs) != len(parts):
        return False
    return all(seg in ("*", part) for seg, part in zip(segs, parts, strict=True))


# -------------------------------------------------- the namespaces a key ranges over


def _families(system: System, params: SystemParams) -> tuple[str, ...]:
    return _hybrid.factor_families(
        params.transition, params.initial_state_prior, system._B, system._facs
    )


def _factor_dim(params: SystemParams, f: int) -> int:
    """A continuous factor's dimension count: its linear-Gaussian transition's ``F``, else its
    initial prior's."""
    decl = params.transition[f]
    if isinstance(decl, LinearGaussianTransition):
        return decl.dim
    return params.initial_state_prior[f].dim


def _dim_names(system: System, params: SystemParams) -> dict[int, tuple[str, ...]]:
    """Each continuous state factor's dimension names, by factor index: those ``System.dims``
    declares, else ``dim0``, ``dim1``, … by position. Empty off the hybrid path (a tabular or
    amortized System has no dimension keys)."""
    if system._is_amortized(params) or not system._is_hybrid(params):
        return {}
    declared = dict(system.dims)
    out: dict[int, tuple[str, ...]] = {}
    for f, family in enumerate(system._families(params)):
        if family == _hybrid.CATEGORICAL:
            continue
        name = system._facs[f]
        out[f] = declared.get(name) or tuple(f"dim{i}" for i in range(_factor_dim(params, f)))
    return out


def _slot_declarations(
    system: System, params: SystemParams
) -> dict[str, tuple[str, int | str, object]]:
    """The slot-qualified key space: ``"<slot>:<name>"`` → ``(slot, index, declaration)``.

    Keys are ``likelihood:<modality>``, ``transition:<factor>``, ``initial_state_prior:<factor>``,
    ``recognition:<variable>`` and ``scoring:<row>``; the last two use the variable or row name as
    the index, since those slots are dicts. Shared by ``learn``'s rates, the ``epistemic``
    selector's parameter half and the novelty leaves' block keys.
    """
    mods, facs = (system.modality_names, system._facs)
    table: dict[str, tuple[str, int | str, object]] = {}
    for m, a in enumerate(params.likelihood):
        table[f"likelihood:{mods[m]}"] = ("likelihood", m, a)
    for f, b in enumerate(params.transition):
        table[f"transition:{facs[f]}"] = ("transition", f, b)
    for f, d in enumerate(params.initial_state_prior):
        table[f"initial_state_prior:{facs[f]}"] = ("initial_state_prior", f, d)
    for name, module in params.recognition.items():
        table[f"recognition:{name}"] = ("recognition", name, module)
    for name, entry in params.scoring.items():
        table[f"scoring:{name}"] = ("scoring", name, entry)
    return table


def _write_slots(
    system: System, params: SystemParams, updated: Mapping[str, object]
) -> SystemParams:
    """New params with the slot-keyed declarations in ``updated`` written back.

    The inverse of :func:`_slot_declarations`. Unknown keys raise. The containers keep the
    form they came in (keyed by name, or the positional view).
    """
    table = system._slot_declarations(params)
    unknown = sorted(set(updated) - set(table))
    if unknown:
        raise ValueError(
            f"cannot write slot key(s) {unknown}: this model's slots are {sorted(table)}"
        )
    mods, facs = (system.modality_names, system._facs)
    containers: dict[str, dict | list] = {}
    for slot in ("likelihood", "transition", "initial_state_prior"):
        container = getattr(params, slot)
        keyed = isinstance(container, Mapping)
        containers[slot] = dict(container) if keyed else list(container)
    recognition = dict(params.recognition)
    scoring = dict(params.scoring)
    for key, decl in updated.items():
        slot, index, _ = table[key]
        if slot == "recognition":
            recognition[index] = decl
        elif slot == "scoring":
            scoring[index] = decl
        elif isinstance(containers[slot], dict):
            containers[slot][(mods if slot == "likelihood" else facs)[index]] = decl
        else:
            containers[slot][index] = decl
    return dataclasses.replace(
        params,
        **{
            slot: values if isinstance(values, dict) else tuple(values)
            for slot, values in containers.items()
        },
        recognition=recognition,
        scoring=scoring,
    )


def _epistemic_variables(system: System, params: SystemParams) -> dict[str, _EpistemicVar]:
    """The epistemic namespace: every variable one can hold beliefs about, by its key.

    State variables by bare factor name (``"pose_cell"``), and each dimension of a continuous one
    by ``<factor>:<dim>`` (``"pose:b1"``, :func:`_dim_names`); observation variables by
    ``obs:<modality>`` (``"obs:cue"``), whether or not they are eligible epistemic targets; and the
    parameter variables of learnable declarations by slot-qualified key, plus one block-qualified
    key per independent posterior block (:func:`_param_blocks`), the two-segment key being the
    group. The sub-namespaces cannot collide, since only state names are bare and no factor is
    named after a reserved first segment; keys are matched by exact lookup, never parsed.
    """
    mods, facs = (system.modality_names, system._facs)
    out: dict[str, _EpistemicVar] = {name: _EpistemicVar("state", f) for f, name in enumerate(facs)}
    for f, names in _dim_names(system, params).items():
        for name in names:
            out[f"{facs[f]}:{name}"] = _EpistemicVar("dim", f, block=name)
    for m, name in enumerate(mods):
        out[f"obs:{name}"] = _EpistemicVar("obs", m)
    for key, (slot, i, decl) in system._slot_declarations(params).items():
        if slot == "scoring" or not _slot_learnable(decl):
            continue  # an objective-side parameter set is not a variable one has
            #            beliefs about — no novelty leaf, so no epistemic namespace entry
        out[key] = _EpistemicVar("param", i, slot, None)
        for block in _param_blocks(decl):
            out[f"{key}:{block}"] = _EpistemicVar("param", i, slot, block)
    return out


def _expand_epistemic(
    system: System, pattern: str, params: SystemParams, variables: Mapping[str, _EpistemicVar]
) -> tuple[str, ...]:
    """One ``epistemic`` wildcard pattern → the variables it selects.

    Validated by :func:`_check_pattern`, then matched segment-wise against the variable table. A
    wildcarded first segment ranges over the parameter slots only (observation variables answer to
    ``obs:``, a factor's dimensions to ``<factor>:*``). Initial-prior and recognition parameter
    variables reached through a wildcard drop silently, since neither has a planning-time novelty
    term; naming or pinning either slot raises.
    """
    system._check_pattern(pattern, "epistemic", params, epistemic=True)
    slot_wildcarded = pattern.split(":")[0] == "*"
    return tuple(
        key
        for key, var in variables.items()
        if _pattern_matches(pattern, key)
        and not (var.kind in ("obs", "dim") and slot_wildcarded)
        and not (var.kind == "param" and var.slot in ("initial_state_prior", "recognition"))
    )


def _eligible_state_blocks(
    system: System, params: SystemParams
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Which channels can carry a state information-gain leaf, and which continuous factors none
    of them observes: ``(modalities, unobserved factors)``.

    Eligible modalities are the tabular ones, the LG modalities observing a continuous factor and
    the Gaussian emissions over categorical parents alone (whose information is about those
    parents). A neural likelihood has no information-gain estimator and is not eligible. An
    unobserved continuous factor (one no eligible channel observes) has no information to gain,
    so it carries no ``infogain`` leaf; its entropy is the ``uncertainty`` row (design:
    per-variable-infogain.md V11 decisions 4 and 7). Empty on the amortized path, whose state
    epistemic term is resolved by :func:`_amortized_epistemic`.
    """
    if system._is_amortized(params):
        return (), ()
    if not system._is_hybrid(params):
        return tuple(range(len(params.likelihood))), ()
    families = system._families(params)
    eligible_sig = tuple(
        m
        for m, a in enumerate(params.likelihood)
        if isinstance(a, TabularObservation)
        or is_emission(a)
        or (
            isinstance(a, LGObservation)
            and families[system.A_deps[m][0]] in (_hybrid.GAUSSIAN, _hybrid.MIXTURE)
        )
    )
    channelled = {
        system.A_deps[m][0]
        for m in eligible_sig
        if isinstance(params.likelihood[m], LGObservation) and not is_emission(params.likelihood[m])
    }
    unobserved = tuple(
        f for f, fam in enumerate(families) if fam != _hybrid.CATEGORICAL and f not in channelled
    )
    return eligible_sig, unobserved


# ------------------------------------------------------------------------ resolution


def _resolve_rates(
    system: System, mapping: Mapping[str, float | Array] | None, what: str, params: SystemParams
) -> dict[str, float | Array] | None:
    """A rate mapping (``lr_by_slot`` / ``decay_by_slot``) → one rate per parameter variable.

    Keys are literal or wildcarded parameter keys (``"likelihood:pose"``,
    ``"likelihood:pose:noise"``, ``"likelihood:*"``, ``"*:*:*"``). The most specific match (most
    literal segments) wins and equal-specificity matches on one variable raise, so every variable
    gets one rate. A declaration's group key is the base for its blocks, so ``{"likelihood:pose":
    0.0, "likelihood:pose:noise": 1.0}`` freezes the head and learns the noise. Literal keys get
    :func:`_check_slot_key`'s errors, patterns :func:`_check_pattern`'s. A ``scoring:<row>`` key
    resolves only for a row with a trainer.
    """
    if mapping is None:
        return None
    table = system._slot_declarations(params)
    patterns: list[tuple[str, int]] = []
    for key in mapping:
        if _is_pattern(key):
            system._check_pattern(key, what, params, epistemic=False)
        else:
            system._check_slot_key(key, what, table)
        patterns.append((key, _specificity(key)))
    resolved: dict[str, float | Array] = {}
    for key, (slot, name, decl) in table.items():
        if slot == "scoring":
            # A scoring address resolves only for a row something can actually step: the
            # trainer that owns the loss reads the resolved rate exactly as fit_world reads a
            # slot's. A scoring entry is ONE addressable unit, so it has no blocks.
            if registry.scoring_trainer(name) is None:
                continue
            best, winners = -1, []
            for pattern, spec in patterns:
                if not _pattern_matches(pattern, key):
                    continue
                if spec > best:
                    best, winners = spec, [pattern]
                elif spec == best:
                    winners.append(pattern)
            if best >= 0:
                if len(winners) > 1:
                    raise ValueError(
                        f"{what}: {sorted(winners)} both match {key!r} with the same "
                        f"specificity ({best} literal segment(s)), so which rate governs it "
                        "is ambiguous; make one of them more specific"
                    )
                resolved[key] = mapping[winners[0]]
            continue
        if not _slot_learnable(decl):
            continue
        # every variable answers to its own address and — for a block — to its
        # declaration's group key, the base a block key overrides
        addresses = [(key, (key,))] + [
            (f"{key}:{block}", (f"{key}:{block}", key)) for block in _param_blocks(decl)
        ]
        for address, aliases in addresses:
            best, winners = -1, []
            for pattern, spec in patterns:
                if not any(_pattern_matches(pattern, alias) for alias in aliases):
                    continue
                if spec > best:
                    best, winners = spec, [pattern]
                elif spec == best:
                    winners.append(pattern)
            if best < 0:
                continue
            if len(winners) > 1:
                raise ValueError(
                    f"{what}: {sorted(winners)} both match {address!r} with the same "
                    f"specificity ({best} literal segment(s)), so which rate governs it "
                    "is ambiguous; the most specific pattern wins, so make one of them "
                    "more specific (or name the variable literally)"
                )
            resolved[address] = mapping[winners[0]]
    return resolved


def _switching_mniw_refusal(key: str) -> NotImplementedError:
    return NotImplementedError(
        f"epistemic: {key} is a matrix-normal inverse-Wishart emission with discrete parents, "
        "whose novelty has no estimator: at a fixed head the reading is a mixture over the "
        "modes, so each draw would need a Kolchinsky-Tracey bracket (design "
        "parameter-learning.md D10). Leave it out of the selection, or pass a tuple naming the "
        "other variables."
    )


def _switching_mniw(decl) -> bool:
    """An MNIW emission with discrete parents: a leading parent (mode) axis on any of its
    parameters, or no state columns at all (an emission over categorical parents alone)."""
    if not isinstance(decl, LinearGaussianObservationMNIW):
        return False
    return (
        decl.M.ndim > 2
        or decl.V_inv.ndim > 2
        or decl.Psi.ndim > 2
        or jnp.ndim(decl.nu) > 0
        or decl.M.shape[-1] < 2
    )


def _resolve_param_novelty(
    system: System,
    sel: bool | tuple[str, ...],
    params: SystemParams,
    planning: bool,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The parameter half of :func:`_term_targets`: ``(modality, factor)`` index tuples.

    ``True`` selects every learnable likelihood and transition slot whose family has a novelty
    estimator; a tuple uses the rate key space, with the same errors (a point-valued slot raises). A
    learnable family with no registry entry raises either way, initial-prior keys raise, and block
    keys resolve to their declaration.
    """
    if sel is False:
        return (), ()
    table = system._slot_declarations(params)
    unregistered = tuple(
        key
        for key, (slot, _, decl) in table.items()
        if slot not in ("initial_state_prior", "recognition", "scoring")
        and _slot_learnable(decl)
        and registry.novelty_estimator(slot, decl) is None
    )
    if sel is True:
        if unregistered and planning:
            raise NotImplementedError(
                f"epistemic=True with learnable slot(s) {sorted(unregistered)} whose "
                "parameter family has no novelty estimator in the objectives registry "
                "(continuous parameter novelty is deferred: design "
                "parameter-learning.md D6/step 3, per-variable-infogain.md V4), and the "
                "selector must not silently ignore a learnable slot. Pass an explicit "
                "tuple naming the tabular variables to include, or False."
            )
        switching = tuple(
            f"likelihood:{system.modality_names[m]}"
            for m, a in enumerate(params.likelihood)
            if _switching_mniw(a)
        )
        if switching and planning:
            raise _switching_mniw_refusal(switching[0])
        learn_m = tuple(
            m
            for m, a in enumerate(params.likelihood)
            if _slot_learnable(a)
            and registry.novelty_estimator("likelihood", a) is not None
            and not _switching_mniw(a)
        )
        learn_f = tuple(
            f
            for f, b in enumerate(params.transition)
            if _slot_learnable(b) and registry.novelty_estimator("transition", b) is not None
        )
        return learn_m, learn_f
    mods_sel: list[int] = []
    facs_sel: list[int] = []
    for key in sel:
        system._check_slot_key(key, "epistemic", table)
        group = ":".join(key.split(":")[:2])
        slot, i, decl = table[group]
        if slot == "initial_state_prior":
            raise ValueError(
                f"epistemic: {key} names an initial-state prior, which has "
                "no planning-time novelty term (planning never revisits the episode "
                "start); select likelihood/transition slots only"
            )
        if slot == "scoring":
            raise ValueError(
                f"epistemic: {key} names a scoring row — objective-side parameters, which "
                "carry no planning-time novelty term: planning scores what the world model "
                "does not know, and a critic is not a world claim (design: amortization.md "
                "A12; rows are mounted by System.terms). Select likelihood/transition "
                "slots only."
            )
        if slot == "recognition":
            raise ValueError(
                f"epistemic: {key} names an amortized recognition module — inference-side "
                "parameters, which carry no planning-time novelty term: planning scores "
                "what the world model does not know, and a recognition module is not a "
                "world claim (design: amortization.md A11; its chartered epistemic landing "
                "is the ensemble_disagreement row, A10 disposition 3). Select "
                "likelihood/transition slots only."
            )
        if registry.novelty_estimator(slot, decl) is None:
            raise NotImplementedError(
                f"epistemic: {key} names a {registry.param_family(decl)} parameter "
                "variable, whose novelty estimator is deferred — the objectives registry "
                "has no entry for it (design parameter-learning.md D6/step 3, "
                "per-variable-infogain.md V4)"
            )
        if _switching_mniw(decl):
            raise _switching_mniw_refusal(key)
        target = mods_sel if slot == "likelihood" else facs_sel
        if i not in target:  # sub-block keys of one declaration share its index
            target.append(i)
    return tuple(sorted(mods_sel)), tuple(sorted(facs_sel))


def _term_targets(
    system: System, params: SystemParams, planning: bool = True
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Resolve the ``epistemic`` selector to declaration-index tuples.

    Returns ``(state_infogain_modalities, param_infogain_modalities, param_infogain_factors)``,
    the indices the step-term builders take. ``True`` mounts every eligible variable; ``False`` /
    ``()`` none. In a tuple:

    - ``True`` mounts no information-gain leaf on a channel whose parents are all action nodes
      (nothing is scored about an action node), under ``utility_infogain``;
    - a bare state name mounts every eligible channel whose parents include that factor (the leaf's
      value is relative to the selected parents, :func:`_channel_selections`). Named literally, a
      continuous factor no eligible channel observes raises: it has no information to gain, and
      its entropy is the ``uncertainty`` row; a wildcard passes over it;
    - ``<factor>:<dim>`` mounts what its factor's name does, the leaf's value then relative to the
      selected dimensions (:func:`_dim_selections`);
    - ``obs:<modality>`` mounts exactly that channel's leaf (an ineligible channel resolves empty;
      an unknown modality raises);
    - a parameter key mounts that variable's novelty leaf.

    Any of these may be a wildcard pattern (:func:`_expand_epistemic`); selection is a union. A
    match with no registered novelty estimator raises, literal or wildcard. Block keys resolve to
    their declaration's index. :func:`_validate` calls this with ``planning=False``.
    """
    sel = system.epistemic
    if sel is False or sel == ():
        return (), (), ()
    eligible_sig, unobserved = system._eligible_state_blocks(params)
    if sel is True:
        param_m, param_f = system._resolve_param_novelty(True, params, planning)
        if system.decomposition == "utility_infogain":
            n = len(system.B_deps)  # a channel over action nodes alone has no state to inform
            eligible_sig = tuple(
                m
                for m in eligible_sig
                if not (system.A_deps[m] and all(g >= n for g in system.A_deps[m]))
            )
        return eligible_sig, param_m, param_f
    variables = system._epistemic_variables(params)
    sig: set[int] = set()
    param_keys: list[str] = []
    named_unobserved: list[str] = []
    for key in sel:
        if _is_pattern(key):
            matched = system._expand_epistemic(key, params, variables)
        else:
            system._check_epistemic_key(key, params, variables)
            matched = (key,)
            if variables[key].kind in ("state", "dim") and variables[key].index in unobserved:
                named_unobserved.append(key)
        for name in matched:
            var = variables[name]
            if var.kind in ("state", "dim"):
                sig.update(m for m in eligible_sig if var.index in system.A_deps[m])
            elif var.kind == "obs":
                if var.index in eligible_sig:  # ineligible channel: resolves empty
                    sig.add(var.index)
            else:
                param_keys.append(name)
    if named_unobserved:
        key = named_unobserved[0]
        f = variables[key].index
        factor = system._facs[f]
        blind = [system.modality_names[m] for m, deps in enumerate(system.A_deps) if f in deps]
        seen_by = (
            f" ({', '.join(repr(m) for m in blind)} observe{'s' if len(blind) == 1 else ''} it "
            "but ha" + ("s" if len(blind) == 1 else "ve") + " no information-gain estimator)"
            if blind
            else ""
        )
        raise ValueError(
            f"epistemic: {key!r} names the continuous factor {factor!r}, which no channel with an "
            f"information-gain estimator observes{seen_by}, so there is no information about it to "
            "gain and no infogain leaf to mount. The drive toward a sharper belief about it is "
            f'the uncertainty row, terms=("uncertainty:{factor}",) (design: '
            "per-variable-infogain.md V11 decision 4)"
        )
    param_m, param_f = system._resolve_param_novelty(tuple(param_keys), params, planning)
    return tuple(sorted(sig)), param_m, param_f


def _channel_selections(
    system: System, params: SystemParams
) -> tuple[tuple[bool, ...] | None, ...]:
    """Per modality, which of its parents (``A_deps`` order) the ``epistemic`` selector selects.

    A channel's state info gain is its information about the selected parents only, the others
    marginalised under the current belief; this returns that selection as a mask, or ``None``
    when the channel is scored over its whole dependency set. ``None`` is returned for every
    channel under ``True``/``False``/``()``, for a channel whose parents are all selected, for one
    with none selected (it mounts no state leaf), and for one whose observation variable
    (``obs:<modality>``) is selected, since selecting the observation buys everything it reveals.
    The exception is ``True`` with a channel over a state factor and an action node whose
    actuator kernel is declared: it is selected on its state parents alone (:func:`_state_only`).

    Design: docs/design/per-variable-infogain.md V9
    """
    sel = system.epistemic
    if isinstance(sel, bool) or sel == ():
        return tuple(
            _state_only(system, params, deps) if sel is True else None for deps in system.A_deps
        )
    variables = system._epistemic_variables(params)
    states: set[int] = set()
    observed: set[int] = set()
    for key in sel:
        if _is_pattern(key):
            matched = system._expand_epistemic(key, params, variables)
        else:
            system._check_epistemic_key(key, params, variables)
            matched = (key,)
        for name in matched:
            var = variables[name]
            if var.kind in ("state", "dim"):  # a dimension selects its factor among the parents
                states.add(var.index)
            elif var.kind == "obs":
                observed.add(var.index)
    out: list[tuple[bool, ...] | None] = []
    for m, deps in enumerate(system.A_deps):
        mask = tuple(f in states for f in deps)
        out.append(None if m in observed or all(mask) or not any(mask) else mask)
    return tuple(out)


def _state_only(system: System, params: SystemParams, deps: tuple[int, ...]):
    """The parent mask of a channel that reads a state factor and an action node whose actuator
    kernel is declared: the state parents alone, so the channel's information gain is about
    the state, not about the action the kernel leaves uncertain. ``None`` otherwise (under the
    delta kernel the action node is a point, which carries no information)."""
    n = len(system.B_deps)
    declared = [g >= n and system.action_names[g - n] in params.actuator for g in deps]
    if not any(declared) or all(g >= n for g in deps):
        return None
    return tuple(not flag for flag in declared)


def _dim_selections(system: System, params: SystemParams) -> tuple[tuple[int, ...] | None, ...]:
    """Per state factor, the dimensions the ``epistemic`` selector selects within it.

    A sorted tuple of positions for a factor selected through some of its ``<factor>:<dim>`` keys
    (one sub-block: selection is a union); ``None`` for a factor selected whole (its bare name, or
    every dimension) or not at all, and for every factor under ``True``/``False``/``()``. A
    sub-block of a Gaussian-carried factor scores its leaves on the marginal of those dimensions;
    one of a mixture-carried factor raises, as does ``obs:<modality>`` on a channel of a factor
    selected in part.

    Design: docs/design/per-variable-infogain.md V10
    """
    sel = system.epistemic
    if isinstance(sel, bool) or sel == ():
        return (None,) * len(system._B)
    variables = system._epistemic_variables(params)
    names = _dim_names(system, params)
    whole: set[int] = set()
    picked: dict[int, set[int]] = {}
    observed: dict[int, str] = {}
    for key in sel:
        if _is_pattern(key):
            matched = system._expand_epistemic(key, params, variables)
        else:
            system._check_epistemic_key(key, params, variables)
            matched = (key,)
        for name in matched:
            var = variables[name]
            if var.kind == "state":
                whole.add(var.index)
            elif var.kind == "dim":
                picked.setdefault(var.index, set()).add(names[var.index].index(var.block))
            elif var.kind == "obs":
                observed.setdefault(var.index, name)
    if not picked:
        return (None,) * len(system._B)
    families = system._families(params)
    out: list[tuple[int, ...] | None] = []
    for f in range(len(system._B)):
        dims = picked.get(f)
        if f in whole or not dims or len(dims) == len(names[f]):
            out.append(None)
            continue
        if families[f] == _hybrid.MIXTURE:
            chosen = [f"{system._facs[f]}:{names[f][i]}" for i in sorted(dims)]
            raise NotImplementedError(
                f"epistemic: {chosen} select part of {system._facs[f]!r}, which is carried as a "
                "Gaussian mixture (it has discrete parents or a mixture prior). Information about "
                "a sub-block of a mixture-carried factor would need a Kolchinsky-Tracey bracket "
                "over the reading given those dimensions, per component, and a within-factor mask "
                "beside the channel's parent mask, which are not built (design: "
                "per-variable-infogain.md V10). Select the whole factor, or carry it as one "
                "Gaussian."
            )
        out.append(tuple(sorted(dims)))
    for m, key in observed.items():
        deps = system.A_deps[m]
        if deps and out[deps[0]] is not None:
            f = deps[0]
            chosen = [f"{system._facs[f]}:{names[f][i]}" for i in out[f]]
            raise ValueError(
                f"epistemic: {key!r} selects everything the channel's readings reveal, and "
                f"{chosen} select part of the factor {system._facs[f]!r} it observes, in the same "
                "selection; a channel's leaf is about one set of variables, so the two ask for "
                f"different leaves on one channel. Drop {key!r} to score the channel's "
                f"information about the sub-block, or name {system._facs[f]!r} whole (design: "
                "per-variable-infogain.md V10 build notes)"
            )
    return tuple(out)


def _amortized_epistemic(
    system: System, params: SystemParams, structure: _amortized.AmortizedStructure
) -> tuple[int, ...]:
    """Which code variables carry an ``infogain`` leaf on the amortized path.

    The ``epistemic`` selector resolved against the usual namespace, filtered to factored
    categorical codes (estimator: ``ensemble_disagreement``); a deterministic carry resolves empty.
    """
    sel = system.epistemic
    if sel is False or sel == ():
        return ()
    codes = tuple(f for f in range(len(system._B)) if structure.families[f] == _amortized.CODE)
    if sel is True:
        system._term_targets(params)  # raises on the learnable-but-unregistered slots
        return codes
    variables = system._epistemic_variables(params)
    out: list[int] = []
    for key in sel:
        if _is_pattern(key):
            matched = system._expand_epistemic(key, params, variables)
        else:
            system._check_epistemic_key(key, params, variables)
            matched = (key,)
        for name in matched:
            var = variables[name]
            if var.kind == "state" and var.index in codes and var.index not in out:
                out.append(var.index)
            elif var.kind == "param":
                system._resolve_param_novelty((name,), params, True)  # raises: no estimator
    return tuple(sorted(out))


def _declared_structures(system: System) -> dict[str, object]:
    """``System.scoring`` as a dict, with row names validated and duplicate keys refused."""
    names = [name for name, _ in system.scoring]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise ValueError(
            f"System.scoring declares row(s) {duplicated} twice; one structure per row "
            "(the entry is that row's declared model, and it is single-valued)"
        )
    unknown = sorted(set(names) - set(registry.ROWS))
    if unknown:
        raise ValueError(
            f"System.scoring names unknown objective row(s) {unknown}; the registered rows "
            f"are {sorted(registry.ROWS)} (hybrid_aif.objectives.registry.ROWS). A row defined "
            "by a reference subtree is registered when that package is imported "
            "(import hybrid_aif.reference)."
        )
    return dict(system.scoring)


def _row_family_error(
    key: str, name: str, what: str, family: str, eligible: tuple[str, ...]
) -> NotImplementedError:
    """The refusal for a ``terms=`` row that has no estimator for its target's family.

    The message names the missing ``(row, family)`` registry entry.
    """
    return NotImplementedError(
        f"terms: {key!r} — the {name!r} row carries an estimator for family/families "
        f"{list(eligible)}, and {what} is {family!r}. A term is a bucket and its estimator is "
        f"dispatched by (term, variable family), so what would make this key legal is a "
        f"{family!r} estimator registered under the {name!r} row — not a reinterpretation of "
        "the ones it has (design: objective-terms.md's lattice; "
        "amortization-debt-ledger.md entry 2)."
    )


def _row_mounts(system: System, params: SystemParams) -> tuple[registry.Mount, ...]:
    """Resolve the ``terms=`` selector to registry rows and declaration indices.

    The row segment is literal, resolved against :data:`~hybrid_aif.objectives.registry.ROWS`; the
    rest is the ``epistemic`` variable grammar, restricted to the row's target kind:

    - a row with no target takes the bare row name only, and is checked against the System's
      families as a whole;
    - a slot-target row's slot segment is literal, its name literal or ``*``;
    - an obs-target row's modality is literal.

    A row mounts on a variable only if the registry holds an estimator for that variable's family
    (``code``/``carry`` on the amortized path, ``categorical``/``gaussian``/``mixture`` otherwise):
    a wildcard drops ineligible members silently, a literal name raises (:func:`_row_family_error`).
    Selection is a union. Every ``System.scoring`` entry must name a row ``terms`` mounts, and every
    mounted row that owns a structure must find one.

    Design: docs/design/amortization.md A12.
    """
    declared = system._declared_structures()
    if not system.terms:
        if declared:
            raise ValueError(
                f"System.scoring declares structure for row(s) {sorted(declared)} but "
                "System.terms mounts none of them, so nothing would ever read it. Mounting "
                "is the terms= selector's job (A12 item 1: never inferred from a slot's "
                "contents) — add the row to terms=, or drop the structure entry."
            )
        return ()
    amortized = system._is_amortized(params)
    families = (
        system._amortized_structure(params).families if amortized else system._families(params)
    )
    table = system._slot_declarations(params)
    mods, facs = (system.modality_names, system._facs)
    mounts: list[registry.Mount] = []

    def by_family(key: str, name: str, eligible: tuple[str, ...]) -> list[registry.Mount]:
        """A wildcard's expansion, one mount per family so each reaches its estimator."""
        groups: dict[str, list[int]] = {}
        for f, family in enumerate(families):
            if family in eligible:
                groups.setdefault(family, []).append(f)
        return [registry.Mount(key, name, tuple(idx), fam) for fam, idx in groups.items()]

    def check_regime(key: str, name: str, eligible: tuple[str, ...]) -> None:
        """A targetless row reads the rollout, so the *System* is what must be eligible."""
        if eligible and not set(families) & set(eligible):
            raise _row_family_error(
                key, name, "this System's variables are", "/".join(sorted(set(families))), eligible
            )

    for key in system.terms:
        name, _, rest = key.partition(":")
        row = registry.get_row(name)
        eligible = registry.row_families(name)
        if row.target is None:
            if rest:
                raise ValueError(
                    f"terms: {key!r} — the {name!r} row scores the whole rollout and takes "
                    f"no target, so its key is the bare row name {name!r}"
                )
            check_regime(key, name, eligible)
            mounts.append(registry.Mount(key, name))
            continue
        parts = rest.split(":") if rest else []
        if row.target == registry.SLOT_TARGET:
            slot = registry.row_slot(name)
            if len(parts) != 2 or parts[0] != slot:
                raise ValueError(
                    f"terms: {key!r} — the {name!r} row scores a declared {slot}, so its key "
                    f"is '{name}:{slot}:<factor>' (or '{name}:{slot}:*' for the class). A "
                    "wildcard in the slot segment would range over slots the row cannot "
                    f"read; there is exactly one it can, {slot!r}."
                )
            if parts[1] == "*":
                mounts.extend(by_family(key, name, eligible))
            else:
                system._check_slot_key(rest, "terms", table, learnable=False)
                f = facs.index(parts[1])
                if families[f] not in eligible:
                    raise _row_family_error(
                        key,
                        name,
                        f"factor {parts[1]!r}'s ({type(params.transition[f]).__name__}) family",
                        families[f],
                        eligible,
                    )
                mounts.append(registry.Mount(key, name, (f,), families[f]))
            continue
        if row.target == registry.STATE_TARGET:
            #  a state variable — a bare factor name, or one of a continuous factor's dimensions,
            #  as ``epistemic=`` takes them
            if len(parts) not in (1, 2) or (
                len(parts) == 2 and (parts[0] == "*" or parts[0] in _KEY_KINDS)
            ):
                raise ValueError(
                    f"terms: {key!r} — the {name!r} row scores a *state variable*, so its key "
                    f"is '{name}:<factor>' (or '{name}:*' for the class): one bare name, the "
                    f"same segment the epistemic selector takes, or '{name}:<factor>:<dim>' for "
                    "dimensions of a continuous factor. It is not slot-qualified — the row "
                    "reads a belief, not a declaration."
                )
            if parts[0] == "*":
                mounts.extend(by_family(key, name, eligible))
            else:
                if parts[0] not in facs:
                    raise ValueError(
                        f"terms: {key!r} names unknown state variable {parts[0]!r}; the factors "
                        f"are {sorted(facs)}"
                    )
                f = facs.index(parts[0])
                if families[f] not in eligible:
                    raise _row_family_error(
                        key, name, f"state variable {parts[0]!r}'s family", families[f], eligible
                    )
                dims = None
                if len(parts) == 2 and parts[1] != "*":
                    names = _dim_names(system, params).get(f, ())
                    if parts[1] not in names:
                        raise ValueError(
                            f"terms: {key!r} names unknown dimension {parts[1]!r} of "
                            f"{parts[0]!r}; its dimensions are {list(names)}"
                        )
                    dims = (names.index(parts[1]),)
                mounts.append(registry.Mount(key, name, (f,), families[f], dims))
            continue
        if len(parts) != 2 or parts[0] != "obs" or parts[1] == "*":
            raise ValueError(
                f"terms: {key!r} — the {name!r} row scores one observation channel, so its "
                f"key is '{name}:obs:<modality>' with the modality named literally. There is "
                "no wildcard form: the row has exactly one target by construction, and "
                "which channel it is must be readable off the System alone (the term tree's "
                "shape may not depend on the parameters)."
            )
        if parts[1] not in mods:
            raise ValueError(
                f"terms: {key!r} names unknown modality {parts[1]!r}; the modalities are "
                f"{sorted(mods)}"
            )
        check_regime(key, name, eligible)
        mounts.append(registry.Mount(key, name, (mods.index(parts[1]),)))
    seen: dict[tuple[str, tuple[int, ...]], registry.Mount] = {}
    for mount in mounts:  # union semantics: the same row on the same target mounts once
        first = seen.setdefault((mount.name, mount.index), mount)
        if first.dims is not None:  # dimension keys on one factor select one sub-block
            dims = None if mount.dims is None else tuple(sorted({*first.dims, *mount.dims}))
            if dims is not None and len(dims) == len(_dim_names(system, params)[mount.index[0]]):
                dims = None  # every dimension is the whole factor
            seen[(mount.name, mount.index)] = first._replace(dims=dims)
    resolved = tuple(seen.values())
    mounted = {mount.name for mount in resolved}
    inert = sorted(set(declared) - mounted)
    if inert:
        raise ValueError(
            f"System.scoring declares structure for row(s) {inert}, which System.terms does "
            f"not mount (it mounts {sorted(mounted)}); nothing would ever read it. Mounting "
            "is the terms= selector's job — add the row to terms=, or drop the entry."
        )
    starving = sorted(
        mount.name
        for mount in resolved
        if registry.get_row(mount.name).needs_structure and mount.name not in declared
    )
    if starving:
        raise ValueError(
            f"row(s) {starving} own a declared model and found none: put its structure in "
            "System.scoring and its parameters in SystemParams.scoring under the same row "
            "name — the two halves of the slot, exactly as System.recognition and "
            "SystemParams.recognition pair up (design: amortization.md A11's rule, A12 "
            "item 2)."
        )
    return resolved
