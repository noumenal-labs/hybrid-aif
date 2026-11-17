"""Declaration-time validation: loud, specific errors before anything is traced.

:func:`_validate` (with :func:`_validate_H_deps`) checks a System's declared structure and runs from
every face. The ``_check_*`` functions check one key of the selector grammar
(:mod:`hybrid_aif.system.selectors`), giving the rate mappings, ``epistemic`` and ``terms=`` the
same errors for the same mistake; :func:`_check_plan` checks a policy array against the control
nodes. ``System``
keeps a thin method for each.

Design: docs/design/system.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.controls import (
    FLOAT,
    LEVEL,
    _observation_axes,
    declared_parents,
    effective_control_deps,
)
from hybrid_aif.distributions import (
    ConditionalCategorical,
    ConditionalDirichlet,
    ConditionalGaussian,
    Gaussian,
    MLPConditional,
)
from hybrid_aif.distributions.neural import LearnedVector, PredictorHead
from hybrid_aif.factors import (
    DerivedInitial,
    GradientDeclaration,
    LGObservation,
    LinearGaussianObservationMNIW,
    TabularObservation,
)
from hybrid_aif.inference import amortized as _amortized
from hybrid_aif.objectives import hybrid as _hybrid
from hybrid_aif.objectives import registry
from hybrid_aif.reads import BLACK_BOX
from hybrid_aif.system.selectors import (
    _DIM_NAME,
    _DIM_RESERVED,
    _KEY_KINDS,
    _SLOT_WORDS,
    _dim_names,
    _factor_dim,
    _param_blocks,
    _slot_learnable,
)

if TYPE_CHECKING:
    from hybrid_aif.system import System, SystemParams
    from hybrid_aif.system.selectors import _EpistemicVar


# ------------------------------------------------------------ the declared structure


def _check_dim_names(factor: str, names: tuple[str, ...], where: str | None = None) -> None:
    """One factor's declared dimension names, checked without its params: at least one, unique,
    single key segments, and ``dimN`` only at position ``N``. The count is checked against the
    factor's dimension where params are validated (:func:`_validate_dims`). ``where`` names the
    declaration in the messages (default ``dims[<factor>]``)."""
    where = where or f"dims[{factor!r}]"
    if not names:
        raise ValueError(f"{where} names no dimensions; name every dimension, in order")
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(
            f"{where} repeats {repeated}: each dimension of a factor needs its own name"
        )
    for i, name in enumerate(names):
        bad = [c for c in _DIM_RESERVED if c in name]
        if not name or bad:
            raise ValueError(
                f"{where} has the name {name!r}: a dimension name is one non-empty key "
                f"segment, without any of {list(_DIM_RESERVED)} (an epistemic key spells it "
                f"'{factor}:<dim>')"
            )
        if _DIM_NAME.fullmatch(name) and name != f"dim{i}":
            raise ValueError(
                f"{where} declares {name!r} at position {i}: 'dimN' is the generated "
                f"name of position N, so it may name position N only; call position {i} "
                f"'dim{i}' or something else"
            )


def _validate_dims(system: System, params: SystemParams) -> None:
    """``System.dims`` against the declarations: each entry names a continuous state factor on
    the linear-Gaussian path, one name per dimension."""
    if not system.dims:
        return
    facs = system._facs
    controls = [name for name, _ in system.dims if name not in facs]
    if controls:
        raise ValueError(
            f"dims names control node(s) {controls}; dimension names address a continuous "
            "state factor's dimensions in the epistemic selector, and a control node is chosen, "
            "not inferred"
        )
    if system._is_amortized(params):
        raise ValueError(
            f"dims names dimensions of {[name for name, _ in system.dims]} on an amortized "
            "System, whose latents are a network's codes and carries, not Gaussian beliefs a "
            "dimension of can be selected; dimension keys are built on the linear-Gaussian path"
        )
    continuous = _dim_names(system, params)
    for name, names in system.dims:
        f = facs.index(name)
        if f not in continuous:
            raise ValueError(
                f"dims names dimensions of {name!r}, a categorical factor, which has levels "
                "rather than dimensions; dims= is for continuous factors"
            )
        d = _factor_dim(params, f)
        if len(names) != d:
            raise ValueError(
                f"dims[{name!r}] declares {len(names)} name(s) {list(names)} for a factor of "
                f"dimension {d}: name every dimension, in order"
            )


def _validate_analytic_declaration(system: System, params: SystemParams) -> None:
    """Refuse what belongs to the neural path, and check the recognition entries' structure."""
    facs = system._facs
    for m, head in enumerate(params.likelihood):
        if isinstance(head, PredictorHead):
            raise ValueError(
                f"modality {system.modality_names[m]!r} is a PredictorHead, a family of the "
                "neural path, where every factor is a Deterministic carry or an MLPCategorical "
                "code; use an analytic likelihood here (design: recognition-beside-analytic.md R1)"
            )
    for f, d in enumerate(params.initial_state_prior):
        if isinstance(d, LearnedVector | DerivedInitial):
            raise ValueError(
                f"factor {facs[f]!r} declares a {type(d).__name__} initial prior, a family of "
                "the neural path, where every factor is a Deterministic carry or an "
                "MLPCategorical code; use an analytic initial prior here (design: "
                "recognition-beside-analytic.md R1)"
            )
    _amortized.check_recognition_entries(system.recognition, params.recognition)
    if not system.recognition:
        return
    s = len(system.B_deps)
    families = system._families(params)[:s]
    if system.mixture_schedule != "mode_aligned":
        for entry in system.recognition:
            if entry.variable in facs and families[facs.index(entry.variable)] == _hybrid.MIXTURE:
                raise ValueError(
                    f"recognition:{entry.variable}: a measurement of the mixture-carried factor "
                    f"{entry.variable!r} under mixture_schedule={system.mixture_schedule!r}, "
                    "where a cross-mode Runnalls merge destroys the per-mode evidence the "
                    "update needs, as for a modality over it; declare "
                    'System(mixture_schedule="mode_aligned") '
                    "(recognition-beside-analytic.md R3)"
                )
    kinds = tuple(
        _amortized.CATEGORICAL_KIND if fam == _hybrid.CATEGORICAL else _amortized.GAUSSIAN_KIND
        for fam in families
    )
    widths = tuple(
        _factor_levels(params, f) if fam == _hybrid.CATEGORICAL else _factor_dim(params, f)
        for f, fam in enumerate(families)
    )
    obs_widths = tuple(
        a.child_dim
        if isinstance(a, MLPConditional)
        else a.H.shape[-2]
        if isinstance(a, LGObservation)
        else None
        for a in params.likelihood
    )
    _amortized.check_analytic_recognition(
        system.recognition,
        params.recognition,
        system.modality_names,
        facs[:s],
        tuple(system._table.control_names),
        kinds,
        widths,
        obs_widths,
        system.external_inputs,
        tuple(isinstance(a, TabularObservation) for a in params.likelihood),
    )


def _validate_propagation_slots(system: System, params: SystemParams) -> None:
    """Refuse a non-default ``propagation`` on an ``MLPConditional`` outside the ``likelihood``
    slot, where it is not read (design: recognition-beside-analytic.md L4)."""
    for slot in ("transition", "recognition"):
        decls = getattr(params, slot, None)
        if not decls:
            continue
        names = (
            tuple(decls)
            if isinstance(decls, Mapping)
            else (system._facs if slot == "transition" else range(len(decls)))
        )
        items = decls.values() if isinstance(decls, Mapping) else decls
        for name, decl in zip(names, items, strict=False):
            if isinstance(decl, MLPConditional) and decl.propagation != "first_order":
                raise ValueError(
                    f"{slot}[{name!r}]: an MLPConditional declares propagation="
                    f"{decl.propagation!r}, which is read only on a likelihood (how a Gaussian "
                    "belief on the observed factor is pushed through the network); a "
                    f'{slot} entry ignores it. Leave it at the default "first_order" here'
                )


def _validate(
    system: System, params: SystemParams, face: str = "this face", declared: bool = False
) -> None:
    """Fail loudly on keys that name nothing and on ill-formed structure.

    Keys are designation keys and selector names; structure covers ``D_deps`` / ``H_deps``, hybrid
    family pairing, within-step ordering, family-pure modalities, a missing ``mixture_k``, and the
    amortized path's own declaration. ``declared`` says the caller has already run
    :func:`_validate_analytic_declaration` on these parameters.
    """
    mods, facs = (system.modality_names, system._facs)
    _validate_cells(system, params, facs)
    for term in system.terms:  # a mounted estimator that owns parameters needs its entry
        row = term.partition(":")[0]
        try:
            parametric = registry.get_row(row).parametric
        except ValueError:
            continue  # an unknown estimator is refused where the terms are read
        mountable = system._is_amortized(params) or set(registry.row_families(row)) != {"code"}
        if parametric and mountable and row not in params.scoring:
            raise ValueError(
                f"{face}: terms mounts the estimator {row!r}, which owns parameters, and "
                f"SystemParams.scoring carries no {row!r} entry (it holds "
                f"{sorted(params.scoring)}); a missing entry is a mismatch, not a silent skip"
            )
    targeted_controls = set(facs) | set(
        system._table.control_names
    )  # a control node takes a target
    checks = [
        ("obs_targets", set(params.obs_targets), set(mods)),
        ("state_targets", set(params.state_targets), targeted_controls),
        ("terminal_targets", set(params.terminal_targets), targeted_controls),
    ]
    for what, got, allowed in checks:
        unknown = got - allowed
        for name in sorted(unknown):
            system._refuse_emission(name, what)
        if unknown:
            raise ValueError(
                f"{what}: unknown block name(s) {sorted(unknown)}; "
                f"known blocks are {sorted(allowed)}"
            )
    if not declared and not system._is_amortized(params):
        _validate_analytic_declaration(system, params)
    _validate_dims(system, params)
    _validate_propagation_slots(system, params)
    system._term_targets(params, planning=False)  # one namespace, V1/V5.3
    system._plan_learning(params)  # its keys name learnable tables
    system._dim_selections(params)  # a sub-block of a mixture-carried factor raises
    _validate_observes(system, params)
    deps_all = system._effective_D_deps()
    if len(deps_all) != len(system._B):
        raise ValueError(f"D_deps has {len(deps_all)} entries for {len(system._B)} factors")
    for f, (d, ddeps) in enumerate(zip(params.initial_state_prior, deps_all, strict=True)):
        if ddeps:
            late = [g for g in ddeps if g >= f]
            if late:
                raise ValueError(
                    f"D_deps for factor {facs[f]!r} names factor(s) "
                    f"{[facs[g] for g in late]} at or after it; conditional priors "
                    "sample ancestrally — parents must be declared earlier"
                )
            if not isinstance(d, ConditionalCategorical | ConditionalDirichlet):
                raise ValueError(
                    f"factor {facs[f]!r} declares D_deps but its D is "
                    f"{type(d).__name__}; hierarchical priors are tabular "
                    "(ConditionalCategorical, or ConditionalDirichlet for a "
                    "learnable one; child-first)"
                )
            if d.parent_ndims != len(ddeps):
                raise ValueError(
                    f"factor {facs[f]!r}: conditional D has parent_ndims="
                    f"{d.parent_ndims} but D_deps names {len(ddeps)} parent(s)"
                )
        elif isinstance(d, ConditionalCategorical | ConditionalDirichlet):
            raise ValueError(
                f"factor {facs[f]!r} has a conditional (hierarchical) D but no "
                "D_deps entry naming its parent factor(s); declare "
                "System(D_deps=...) so the conditioning parent is identified"
            )
    system._validate_node_targets(params, face)
    system._durations(params, face)
    _validate_diagonal_targets(params)
    _validate_C_deps(system, params, facs)
    system._validate_H_deps(params, facs)
    _validate_control_deps(system, params, facs, face)
    _validate_cells(system, params, facs)
    _validate_table_arity(system, params, facs)
    _validate_likelihood_parents(system, params, facs)
    _validate_emissions(system, params, facs)
    _validate_readings(system, params)
    system._row_mounts(params)  # raises on a non-amortized System that declares rows
    space = system._space(params)  # node kinds and sizes; readers that disagree raise
    if system._is_amortized(params):
        system._amortized_structure(params, face)
        system._row_mounts(params)  # the terms= keys, at declaration time
        return
    if system._is_hybrid(params):
        families = system._families(params)
        for f, hdeps in enumerate(system._effective_H_deps()):
            bad = [facs[g] for g in hdeps if families[g] != _hybrid.CATEGORICAL]
            if bad:
                raise ValueError(
                    f"H_deps for factor {facs[f]!r} names non-categorical parent(s) "
                    f"{bad}; a conditional terminal target is a table indexed by its "
                    "parents' discrete levels (a continuous conditional target is not "
                    "supported; see System.resolve_terminal_targets)"
                )
        for f, ddeps in enumerate(deps_all):
            bad = [facs[g] for g in ddeps if families[g] != _hybrid.CATEGORICAL]
            if bad:
                raise ValueError(
                    f"D_deps for factor {facs[f]!r} names non-categorical "
                    f"parent(s) {bad}; hierarchical priors condition on "
                    "discrete levels only"
                )
        _hybrid.validate_structure(
            params.likelihood,
            params.transition,
            params.initial_state_prior,
            params.obs_targets,
            system._engine_A_deps,
            system._B,
            system._table,
            space,
            system.mixture_k,
            mods,
            facs,
            system.mixture_schedule,
            system._sensed_nodes(params, face),
            system._cell_grids,
        )


def _validate_control_deps(
    system: System, params: SystemParams, facs: tuple[str, ...], face: str = "this face"
) -> None:
    """``B_control_deps`` against the transition declarations.

    A tabular transition carries ``B_deps`` parent axes, then one control axis per entry sized by
    that control's level count (a single size-1 axis for an empty entry). A linear-Gaussian
    transition reads a discrete node as a mode axis and a continuous one through ``G``, a neural
    one a discrete node's one-hot and a continuous node's value (both checked with their
    families). Design: docs/design/control-states.md (Timing; C4).
    """
    table = system._table
    if table.default_wiring:
        return
    if system._is_amortized(params):
        _refuse_shared_controls(system, facs, "on the amortized path", face)
        return
    if table.sugar is None:
        _validate_table_arity(system, params, facs)
        return
    declared = system._cdeps(face)
    for f, (b, deps) in enumerate(zip(params.transition, declared, strict=True)):
        if not isinstance(b, ConditionalCategorical | ConditionalDirichlet):
            continue
        n_parents = len(system._B[f])
        sizes = tuple(system._num_controls(params, face)[c] for c in deps) or (1,)
        if b.parent_ndims != n_parents + len(sizes):
            raise ValueError(
                f"factor {facs[f]!r}: transition has parent_ndims={b.parent_ndims}, but B_deps "
                f"names {n_parents} parent(s) and B_control_deps {len(deps)} control(s), so it "
                f"needs {n_parents + len(sizes)} (the state parents, then one control axis per "
                "B_control_deps entry, or one size-1 axis when the entry is empty)"
            )
        got = tuple(b.parent_shape[n_parents:])
        if got != sizes:
            raise ValueError(
                f"factor {facs[f]!r}: trailing control axes have sizes {got}, but "
                f"B_control_deps names {[facs[c] for c in deps]} with level counts {sizes}; "
                "each control axis must have its control's level count"
            )


def reference_note(system: System, params: SystemParams, message: str) -> str:
    """A sentence naming the parameter variable behind a refusal that names a reader of one.

    Empty when no reader the ``message`` quotes reads a named parameter variable. A reader's
    table is its variable's, so the refusal the variable's shape draws is the variable's too."""
    names_of = {"transition": system._facs, "likelihood": system.modality_names}
    hits = [
        f"{'factor' if slot == 'transition' else 'modality'} {names_of[slot][i]!r} reads "
        f"parameter variable {home!r}"
        for slot, i, home in getattr(params, "references", ())
        if repr(names_of[slot][i]) in message
    ]
    return f" ({'; '.join(hits)})" if hits else ""


def _validate_cells(system: System, params: SystemParams, facs: tuple[str, ...]) -> None:
    """A grid is declared only for a continuous state factor, over coordinates its state has."""
    if not system.cells:
        return
    if system._is_amortized(params):
        raise NotImplementedError(
            "cells on an amortized (neural state-space) System: the amortized path carries codes "
            "and does not read a tabular factor's continuous parent through a grid; drop cells= "
            "or declare the factors analytically"
        )
    hybrid = system._is_hybrid(params)
    families = system._families(params) if hybrid else None
    for g, grid in system._cell_grids.items():
        if families is None or families[g] == _hybrid.CATEGORICAL:
            raise ValueError(
                f"cells declares a grid for {facs[g]!r}, which is a categorical factor; a grid "
                "discretises a continuous factor (a Gaussian or mixture position), so drop the "
                "entry or declare the factor continuous"
            )
        dim = params.initial_state_prior[g].dim
        beyond = [a for a in grid.axes if a >= dim]
        if beyond:
            raise ValueError(
                f"cells[{facs[g]!r}] spans coordinate(s) {beyond}, but the factor has {dim} "
                f"dimension(s); name axes below {dim}"
            )


def _validate_table_arity(system: System, params: SystemParams, facs: tuple[str, ...]) -> None:
    """A tabular transition has one parent axis per declared parent (``B_deps``, control nodes
    included), plus one size-1 axis when it reads no control node; any other axis would be read
    at level 0 silently."""
    _validate_cells(system, params, facs)
    table = system._table
    grids = system._cell_grids
    for f, b in enumerate(params.transition):
        if not isinstance(b, ConditionalCategorical | ConditionalDirichlet):
            continue
        arity, reads = table.arity[f], bool(table.control_parents[f])
        n = b.parent_ndims
        padded = not reads and n == arity + 1 and b.parent_shape[-1] == 1
        if n != arity and not padded:
            raise ValueError(
                f"factor {facs[f]!r}: its transition table has parent_ndims={n} (parent axes "
                f"{tuple(b.parent_shape)}), and B_deps declares {arity} parent(s); give the table "
                "one axis per declared parent"
                + ("" if reads else ", plus at most one trailing axis of size 1")
            )
        for pos, (what, g) in enumerate(declared_parents(table, f)):
            grid = grids.get(g) if what == "state" else None
            levels = _factor_levels(params, g) if what == "state" and grid is None else None
            if levels is not None and b.parent_shape[pos] != levels:
                raise ValueError(
                    f"factor {facs[f]!r}: its transition table's parent axis {pos} has "
                    f"{b.parent_shape[pos]} entries, but its parent {facs[g]!r} has {levels} "
                    "levels; give the table one entry per level of each parent"
                )
            if grid is not None and b.parent_shape[pos] != grid.n_cells:
                raise ValueError(
                    f"factor {facs[f]!r}: its transition table's parent axis {pos} has "
                    f"{b.parent_shape[pos]} entries, but the grid declared for {facs[g]!r} has "
                    f"{grid.n_cells} cells (shape {grid.shape}); give the table one entry per "
                    "cell, flattened row-major"
                )


def _validate_likelihood_parents(
    system: System, params: SystemParams, facs: tuple[str, ...]
) -> None:
    """A tabular likelihood's parent axis has as many entries as its state parent has levels (a
    grid parent: one per cell), and one axis per declared parent."""
    n_states = len(facs)
    grids = system._cell_grids
    for m, a in enumerate(params.likelihood):
        if not isinstance(a, ConditionalCategorical | ConditionalDirichlet):
            continue
        deps = system.A_deps[m]
        name = system.modality_names[m]
        if a.parent_ndims != len(deps):
            continue  # the table's own arity is the declaration family's check
        for pos, g in enumerate(deps):
            if g >= n_states:
                continue  # a sensed action node
            size = grids[g].n_cells if g in grids else _factor_levels(params, g)
            if size is not None and a.parent_shape[pos] != size:
                raise ValueError(
                    f"modality {name!r}: its likelihood table's parent axis {pos} has "
                    f"{a.parent_shape[pos]} entries, but its parent {facs[g]!r} has {size} "
                    "levels; give the table one entry per level of each parent"
                )


def _validate_emissions(system: System, params: SystemParams, facs: tuple[str, ...]) -> None:
    """The emission kernels against the declaration: one per emission that has no static kernel,
    each a likelihood of a family (a table, a linear-Gaussian or neural conditional, in point or
    learnable form), typed as a port, with one parent axis per parent and one entry per level of
    a discrete parent (a grid parent: one per cell); and each static kernel over a parent it can
    read: the identity of one state factor, a Grid over a continuous one, within its
    coordinates."""
    if not system.emissions and not params.emission:
        return
    decls = system._emission_decls(params)
    table = system._table
    grids = system._cell_grids
    space = system._space(params) if table.controls else None
    families = None
    for name, emit in system.emissions:
        if emit.kernel is None:
            continue
        system._static_type(params, emit, name)
        if emit.kernel == BLACK_BOX:
            continue
        families = families or system._families(params)
        (parent,) = emit.parents
        f = table.state_names.index(parent)
        if emit.kernel == "identity":
            if emit.realise == "stats" and families[f] != _hybrid.CATEGORICAL:
                raise ValueError(
                    f"emission {name!r} is the identity of the continuous factor {parent!r} "
                    "with realise='stats', the sufficient statistics of a Gaussian belief, "
                    "which are deferred; declare 'sample' or 'mode', or read the factor through "
                    "an indicator over a Grid"
                )
            continue
        if families[f] == _hybrid.CATEGORICAL:
            raise ValueError(
                f"emission {name!r} reads the categorical factor {parent!r} through an indicator "
                "kernel, which is the cell a continuous factor falls in; read a categorical "
                "factor through the identity (kernel='identity')"
            )
        dim = _factor_dim(params, f)
        if max(emit.kernel.axes) >= dim:
            raise ValueError(
                f"emission {name!r} reads the continuous factor {parent!r} ({dim}-dimensional) "
                f"through a grid over coordinate(s) {list(emit.kernel.axes)}; every axis must be a "
                "coordinate of the factor"
            )
    for (name, emit), decl in zip(system._kernelled(params), decls, strict=True):
        where = f"emission {name!r}"
        if not isinstance(
            decl, ConditionalCategorical | ConditionalDirichlet | LGObservation | MLPConditional
        ):
            raise ValueError(
                f"{where}: its kernel params.emission[{name!r}] is a {type(decl).__name__}, which "
                "is no likelihood family; declare a table (ConditionalCategorical, or "
                "ConditionalDirichlet), a linear-Gaussian observation or a neural conditional"
            )
        system._emission_types(params)  # a kernel that gives no port type raises, naming it
        if isinstance(
            decl, ConditionalCategorical | ConditionalDirichlet
        ) and decl.parent_ndims != len(emit.parents):
            raise ValueError(
                f"{where}: its table has {decl.parent_ndims} parent axis/es and the emission "
                f"names {len(emit.parents)} parent(s) {list(emit.parents)}; give the table one "
                "axis per parent, in that order"
            )
        axes = _observation_axes(decl, len(emit.parents))
        for pos, (parent, axis) in enumerate(zip(emit.parents, axes or (), strict=False)):
            if axis is None:
                continue
            if parent in table.control_names:
                k = table.control_names.index(parent)
                size = space.sizes[k] if space.kinds[k] == LEVEL else None
            else:
                f = table.state_names.index(parent)
                size = grids[f].n_cells if f in grids else _factor_levels(params, f)
            if size is not None and axis != size:
                raise ValueError(
                    f"{where}: its kernel's parent axis {pos} has {axis} entries, but its parent "
                    f"{parent!r} has {size} levels; give the kernel one entry per level of each "
                    "discrete parent"
                )


def _validate_readings(system: System, params: SystemParams) -> None:
    """``System.readings`` against the params: an entry per declared table, each a positive count
    per level of a discrete control node of this System, over a learnable (Dirichlet) table."""
    declared = {name for name, _ in system.readings}
    given = set(params.readings)
    if not declared and not given:
        return
    if system._is_amortized(params):
        raise NotImplementedError(
            "readings on an amortized (neural state-space) System: the pooled count scales a "
            "tabular table's novelty term, which the amortized path does not compute"
        )
    if declared - given or given - declared:
        raise ValueError(
            f"readings declares {sorted(declared)} and params.readings holds {sorted(given)}; "
            "each declared table needs its counts, and a count names a declared table"
        )
    table = system._table
    space = system._space(params)
    facs, mods = table.state_names, system.modality_names
    _, selected_mods, selected_facs = system._term_targets(params, planning=False)
    for name, node in system.readings:
        slot = "likelihood" if name in mods else "transition"
        decl = (
            params.likelihood[mods.index(name)]
            if name in mods
            else params.transition[facs.index(name)]
        )
        if not isinstance(decl, ConditionalDirichlet):
            raise ValueError(
                f"readings: {slot}:{name} holds a {type(decl).__name__}, and a pooled reading "
                "scales the novelty of a learnable table (a ConditionalDirichlet, directly or "
                "through a parameter variable)"
            )
        k = table.control_names.index(node)
        if space.kinds[k] != LEVEL:
            raise ValueError(
                f"readings: {name!r} pools over {node!r}, a continuous control node; a count is "
                "one per level of a discrete node"
            )
        chosen = (
            mods.index(name) in selected_mods if name in mods else facs.index(name) in selected_facs
        )
        if not chosen:
            raise ValueError(
                f"readings: {name!r} pools the novelty term {slot}:{name}, which epistemic does "
                f"not select, so the count would scale nothing; add {slot}:{name!r} to epistemic"
            )
        counts = jnp.asarray(params.readings[name])
        if counts.shape != (space.sizes[k],):
            raise ValueError(
                f"readings[{name!r}] has shape {tuple(counts.shape)}; node {node!r} has "
                f"{space.sizes[k]} level(s), one positive count each"
            )
        if not isinstance(counts, jax.core.Tracer) and not bool(jnp.all(counts > 0.0)):
            raise ValueError(f"readings[{name!r}] must be positive counts, got {counts.tolist()}")


def _refuse_shared_controls(
    system: System, facs: tuple[str, ...], where: str, face: str = "this face"
) -> None:
    """Raise unless ``B_control_deps`` is absent or every entry is its factor's default."""
    if system._table.default_wiring:
        return
    if system._table.sugar is None:
        raise NotImplementedError(
            f"control nodes {list(system._table.control_names)} wired other than one per "
            f"factor, as its last parent, are not supported {where} (tabular transitions only)"
        )
    default = effective_control_deps(None, system._ctl(face))
    for f, deps in enumerate(system._cdeps(face)):
        if deps != default[f]:
            raise NotImplementedError(
                f"factor {facs[f]!r}: B_control_deps entry {[facs[c] for c in deps]} differs "
                f"from the default {[facs[c] for c in default[f]]}; shared controls are not "
                f"supported {where} (tabular transitions only)"
            )


def _validate_observes(system: System, params: SystemParams) -> None:
    """A slot's port type is the type of the variable it observes (``System.observes``): a
    modality's likelihood gives it ``(LEVEL, levels)`` or ``(FLOAT, dimension)``, a control node
    its level count or width, and the variable's is the emission's or the action node's. The
    amortized path has its own recurrent carry and observes none."""
    if not system.observes:
        return
    if system._is_amortized(params):
        raise NotImplementedError(
            "observes: a modality observing an emission or an action node reads a datum the "
            "caller supplies, and the amortized path has its own recurrent carry and reads none"
        )
    space = system._space(params) if system._table.controls else None
    nodes = system._table.control_names
    types = {
        **system._modality_types(params),
        **{u: (space.kinds[k], int(space.sizes[k])) for k, u in enumerate(nodes)},
    }
    generated = {
        **system._emission_types(params),
        **{
            a: (space.kinds[k], int(space.sizes[k]))
            for k, a in enumerate(system.action_names or ())
        },
    }
    for slot, variable in system.observes:
        got, want = types[slot], generated[variable]
        if got != want:
            kind = {"level": "levels", "float": "dimensions"}
            raise ValueError(
                f"observes: {slot!r} observes {variable!r}, whose port is {want[1]} "
                f"{kind[want[0]]} ({want[0]}), but the slot takes {got[1]} {kind[got[0]]} "
                f"({got[0]}); the port types must be equal"
            )


def _validate_H_deps(system: System, params: SystemParams, facs: tuple[str, ...]) -> None:
    """``H_deps`` against the terminal designations.

    Every conditional target must have parents and every parented target be conditional: the
    ``D_deps`` checks without the ancestral-ordering rule, plus a self-parent rejection (a target
    conditioned on its own final belief would move with its argument).
    Design: docs/design/nesting-systems.md N1.
    """
    num_factors = len(system._B)
    if system._table.H_deps is not None and len(system._table.H_deps) != num_factors:
        raise ValueError(
            f"H_deps has {len(system._table.H_deps)} entries for {num_factors} factors; declare "
            "one entry per *factor* (empty for an unconditional target). Terminal "
            "designations are state-block designations — a modality carries a standing "
            "C designation (obs_targets), never an H one, so it has no H_deps entry"
        )
    for f, hdeps in enumerate(system._effective_H_deps()):
        target = params.terminal_targets.get(facs[f])
        if hdeps:
            bad = [g for g in hdeps if not 0 <= g < num_factors]
            if bad:
                raise ValueError(
                    f"H_deps for factor {facs[f]!r} names {bad}, which are not factor "
                    f"indices; the factors are {list(facs)} (parents of a terminal "
                    "target are declared factors of this System — a modality cannot "
                    "parent one)"
                )
            if f in hdeps:
                raise ValueError(
                    f"H_deps for factor {facs[f]!r} names the factor itself; a terminal "
                    "target conditioned on its own final belief is a moving target, not "
                    "a preference (condition it on another factor — under merge() "
                    "typically a higher System's factor)"
                )
            if target is None:
                raise ValueError(
                    f"H_deps names parent(s) {[facs[g] for g in hdeps]} for factor "
                    f"{facs[f]!r}, which carries no terminal_targets entry; declare the "
                    "conditional target (a ConditionalCategorical over its levels, one "
                    "column per parent configuration) or drop the H_deps entry"
                )
            if isinstance(target, ConditionalDirichlet):
                raise NotImplementedError(
                    f"factor {facs[f]!r}: a *learnable* conditional terminal target "
                    "(Dirichlet counts over intended states — the p.13 extension of "
                    "arXiv:2312.07547, 'planning becomes habitual') is not built; "
                    "System.learn has no h statistic. Declare a point "
                    "ConditionalCategorical target."
                )
            if not isinstance(target, ConditionalCategorical):
                raise ValueError(
                    f"factor {facs[f]!r} declares H_deps but its terminal target is "
                    f"{type(target).__name__}; a conditional terminal designation is a "
                    "ConditionalCategorical (child-first table over the factor's levels, "
                    "one column per parent configuration)"
                )
            if target.parent_ndims != len(hdeps):
                raise ValueError(
                    f"factor {facs[f]!r}: conditional terminal target has parent_ndims="
                    f"{target.parent_ndims} but H_deps names {len(hdeps)} parent(s)"
                )
            sizes = tuple(_factor_levels(params, g) for g in hdeps)
            if all(s is not None for s in sizes) and target.parent_shape != sizes:
                raise ValueError(
                    f"factor {facs[f]!r}: conditional terminal target has parent axes "
                    f"{target.parent_shape} but its H_deps parent(s) "
                    f"{[facs[g] for g in hdeps]} have {sizes} level(s); the table is "
                    "child-first — (num_levels, *parent_shape) — so a parent-leading "
                    "table needs ConditionalCategorical.from_parent_leading"
                )
        elif isinstance(target, ConditionalCategorical | ConditionalDirichlet):
            raise ValueError(
                f"factor {facs[f]!r} has a conditional terminal target but no H_deps "
                "entry naming its parent factor(s); declare System(H_deps=...) so the "
                "conditioning parent is identified (under merge(), a target-emission "
                "binding declares it for you)"
            )


_CONDITIONAL_TARGETS = ConditionalCategorical | ConditionalDirichlet | ConditionalGaussian


def _carrier(system: System, params: SystemParams, facs: tuple[str, ...], key: str) -> str | None:
    """``"categorical"`` or ``"gaussian"`` for the node ``key`` names, by its likelihood or
    transition (``None`` when neither is told apart)."""
    mods = system.modality_names
    if key in mods:
        a = params.likelihood[mods.index(key)]
        if isinstance(a, TabularObservation):
            return "categorical"
        return "gaussian" if isinstance(a, LGObservation) else None
    return "categorical" if _factor_levels(params, facs.index(key)) is not None else "gaussian"


def _refuse_shared_modes(
    system: System,
    params: SystemParams,
    facs: tuple[str, ...],
    key: str,
    state_parents: list[str],
    where: str,
) -> None:
    """A target's state parent may not also be a categorical parent of its modality, or the mode
    factor of the mixture the modality observes: the levels would then pair with the predictive's
    components, and the weighted expectation reads the marginal predictive instead."""
    m = system.modality_names.index(key)
    deps = system._engine_A_deps[m]
    states = system._table.num_states
    shared = {facs[g] for g in deps[1:] if g < states}
    observed = deps[0]
    if observed < states and system._is_hybrid(params):
        if system._families(params)[observed] == _hybrid.MIXTURE:
            shared |= {facs[g] for g in system._B[observed] if g < states}
    for p in state_parents:
        if p in shared:
            raise NotImplementedError(
                f"{where}: the target on {key!r} is conditional on {p!r}, which {key!r} itself "
                "reads as a mode (a categorical parent of the modality, or the mode factor of "
                "the mixture it observes), so the levels of the target would pair with the "
                "predictive's components, which is not built; condition on a state factor the "
                "modality does not read"
            )


def _refuse_diagonal(where: str, what: str) -> ValueError:
    return ValueError(
        f"{where} is a Gaussian with a diagonal standard deviation (a chol without its second "
        f"axis), a form for a modality's obs_targets entry; {what} reads a full chol"
    )


def _validate_diagonal_targets(params: SystemParams) -> None:
    """A diagonal Gaussian preference (a standard deviation per dimension, :attr:`Gaussian.
    diagonal` and :attr:`ConditionalGaussian.diagonal`) is a preference over a modality: the
    objective terms of a state or terminal target read a full factor."""
    for scope, given in (
        ("state_targets", params.state_targets),
        ("terminal_targets", params.terminal_targets),
    ):
        for key, target in given.items():
            if isinstance(target, Gaussian | ConditionalGaussian) and target.diagonal:
                raise _refuse_diagonal(f"{scope}[{key!r}]", "a state or terminal target")
    for f, prior in enumerate(params.initial_state_prior):
        if isinstance(prior, Gaussian) and prior.diagonal:
            raise _refuse_diagonal(f"initial_state_prior[{f}]", "an initial prior")
    transition = params.transition
    for key, decl in transition.items() if isinstance(transition, dict) else enumerate(transition):
        inn = getattr(decl, "innovations", None)
        if isinstance(inn, Gaussian | ConditionalGaussian) and inn.diagonal:
            raise _refuse_diagonal(f"transition[{key!r}].innovations", "an innovation")


def _validate_C_deps(system: System, params: SystemParams, facs: tuple[str, ...]) -> None:
    """``C_deps`` against the per-step designations.

    Every key names a target that is conditional, with one table axis (or one leading axis of
    its ``loc``/``chol``) per parent, each sized by the parent's levels: a discrete control
    node's, or a categorical state factor's. A conditional family under a key without an entry,
    a continuous control node or a non-categorical factor as a parent, a learnable conditional
    target, and a Gaussian modality target with a state parent under the risk decomposition (a
    mixture target per step; the utility form takes one) are refused.
    The names themselves are checked at construction (:meth:`System._check_C_deps`).
    Design: docs/design/step-durations.md T5.
    """
    mods = system.modality_names
    table = system._table
    controls = table.control_names
    c_deps = dict(system.C_deps)
    if c_deps and system._is_amortized(params):
        raise NotImplementedError(
            f"C_deps={dict(c_deps)!r} on the amortized (neural state-space) path: a conditional "
            "per-step target is resolved by the tabular and linear-Gaussian scoring closures, "
            "and the amortized rows score their own models; declare unconditional targets there"
        )
    hybrid = bool(c_deps) and system._is_hybrid(params)
    families = system._families(params) if hybrid else None
    space = system._space(params) if c_deps else None
    for key, parents in c_deps.items():
        if key in mods:
            scope, target = "obs_targets", params.obs_targets.get(key)
        else:
            scope, target = "state_targets", params.state_targets.get(key)
        where = f"{scope}[{key!r}]"
        if target is None:
            raise ValueError(
                f"C_deps names parent(s) {list(parents)} for {key!r}, which carries no {scope} "
                "entry; declare the conditional target (one distribution per parent "
                "configuration) or drop the C_deps entry"
            )
        sizes = []
        for p in parents:
            if p in controls:
                k = controls.index(p)
                if space.kinds[k] == FLOAT:
                    raise ValueError(
                        f"C_deps[{key!r}] names {p!r}, a continuous control node; a conditional "
                        "target holds one distribution per level of its parents, so a parent is "
                        "a discrete control node or a categorical state factor"
                    )
                sizes.append(space.levels[k])
            else:
                g = facs.index(p)
                n = _factor_levels(params, g)
                if n is None:
                    raise ValueError(
                        f"C_deps[{key!r}] names the factor {p!r}, which is not categorical; a "
                        "conditional target holds one distribution per level of its parents, so "
                        "a state parent is a categorical factor"
                    )
                sizes.append(n)
        state_parents = [p for p in parents if p not in controls]
        if isinstance(target, ConditionalDirichlet):
            raise NotImplementedError(
                f"{where}: a learnable conditional target (Dirichlet counts over the preferred "
                "levels) is not built; declare a point ConditionalCategorical target"
            )
        carrier = _carrier(system, params, facs, key)
        if isinstance(target, ConditionalGaussian) and carrier == "categorical":
            raise ValueError(
                f"{where}: a ConditionalGaussian target on {key!r}, a categorical "
                f"{'modality' if key in mods else 'factor'}; a Gaussian target is declared on a "
                "linear-Gaussian channel or a Gaussian factor, and a categorical one takes a "
                "ConditionalCategorical"
            )
        if isinstance(target, ConditionalCategorical) and carrier == "gaussian":
            raise ValueError(
                f"{where}: a ConditionalCategorical target on {key!r}, a Gaussian "
                f"{'channel' if key in mods else 'factor'}; a categorical target is declared on "
                "a tabular modality or a categorical factor, and a Gaussian one takes a "
                "ConditionalGaussian"
            )
        if isinstance(target, ConditionalGaussian):
            if state_parents and key in mods and system.decomposition == "risk_ambiguity":
                raise NotImplementedError(
                    f"{where}: a Gaussian target conditional on the state factor(s) "
                    f"{state_parents} is a mixture target at each step when scored by risk, "
                    "which is not built; the utility form (decomposition='utility_infogain') "
                    "takes a state parent and scores the parent-belief-weighted expected "
                    "log-density, each level relative to its peak. Use that decomposition, "
                    "condition on control nodes only, or write a ConditionalCategorical "
                    "target over a categorical factor"
                )
            if state_parents and key not in mods:
                raise NotImplementedError(
                    f"{where}: a Gaussian state target conditional on the state factor(s) "
                    f"{state_parents} is not built; condition a modality target on it under "
                    "decomposition='utility_infogain', or condition on control nodes only"
                )
            if state_parents and key in mods:
                _refuse_shared_modes(system, params, facs, key, state_parents, where)
            what = "ConditionalGaussian target"
            if hybrid and key in facs and families[facs.index(key)] == _hybrid.GAUSSIAN:
                d = params.initial_state_prior[facs.index(key)].dim
                if target.dim != d:
                    raise ValueError(
                        f"{where}: the conditional target is over R^{target.dim} but the factor "
                        f"is {d}-dimensional; loc is (*levels, {d}) and chol (*levels, {d}, {d}) "
                        f"or ({d}, {d})"
                    )
        elif isinstance(target, ConditionalCategorical):
            what = "ConditionalCategorical target"
            levels = None
            if key in mods:
                a = params.likelihood[mods.index(key)]
                if isinstance(a, ConditionalCategorical | ConditionalDirichlet):
                    levels = a.num_levels
            else:
                levels = _factor_levels(params, facs.index(key))
            if levels is not None and target.num_levels != levels:
                raise ValueError(
                    f"{where}: the conditional target has {target.num_levels} levels on its "
                    f"child axis but {key!r} has {levels}; the table is child-first, "
                    f"({levels}, *parent levels)"
                )
        else:
            raise ValueError(
                f"{where} declares C_deps parent(s) {list(parents)} but the target is a "
                f"{type(target).__name__}; a conditional target is a ConditionalCategorical "
                "(a child-first table, one column per parent configuration) or a "
                "ConditionalGaussian (loc (*levels, d), chol (*levels, d, d), or (d, d) shared by "
                "every level)"
            )
        if target.parent_ndims != len(parents):
            raise ValueError(
                f"{where}: the {what} has parent_ndims={target.parent_ndims} but C_deps names "
                f"{len(parents)} parent(s) {list(parents)}"
            )
        if target.parent_shape != tuple(sizes):
            raise ValueError(
                f"{where}: the {what} has parent axes {target.parent_shape} but its C_deps "
                f"parent(s) {list(parents)} have {sizes} level(s), in that order"
            )
    for scope, targets in (
        ("obs_targets", params.obs_targets),
        ("state_targets", params.state_targets),
    ):
        for key, target in targets.items():
            if key in c_deps or key in controls or not isinstance(target, _CONDITIONAL_TARGETS):
                continue
            raise ValueError(
                f"{scope}[{key!r}] is a {type(target).__name__}, a target conditional on parents, "
                f"but no C_deps entry names them; declare System(C_deps={{{key!r}: (<parent>, "
                "...)}}) with its parents (discrete control nodes or state factors) in the "
                "table's axis order"
            )
    for key, target in params.terminal_targets.items():
        if isinstance(target, ConditionalGaussian):
            raise ValueError(
                f"terminal_targets[{key!r}] is a ConditionalGaussian; a terminal target is "
                "conditional through H_deps on state factors, as a ConditionalCategorical, and "
                "a per-step target conditional on a control node is a state_targets entry "
                "under C_deps"
            )


def _factor_levels(params: SystemParams, f: int) -> int | None:
    """A factor's discrete level count (its transition's child axis), or ``None`` if not tabular."""
    b = params.transition[f]
    return b.num_levels if isinstance(b, ConditionalCategorical | ConditionalDirichlet) else None


# --------------------------------------------------------------------------- one key


def _check_slot_key(
    key: str,
    what: str,
    table: Mapping[str, tuple[str, int, object]],
    learnable: bool = True,
) -> None:
    """Check one slot-qualified key.

    It must be known, name a learnable declaration, and give any third segment as one of that
    declaration's posterior blocks. ``learnable=False`` checks only that the key names a declaration
    (a ``terms=`` row such as ``occupancy`` may read a point-valued kernel).
    """
    parts = key.split(":")
    if len(parts) == 3:
        group = ":".join(parts[:2])
        _check_slot_key(group, what, table, learnable)  # unknown / point-valued first
        _, _, decl = table[group]
        blocks = _param_blocks(decl)
        if parts[2] in blocks:
            return
        if isinstance(decl, LinearGaussianObservationMNIW):
            raise ValueError(
                f"{what}: {key!r} addresses a sub-block of an MNIW declaration, whose "
                "head and noise share ONE joint posterior (the head's row covariance "
                f"IS the noise) — they are a single variable, named {group!r}; there "
                "are no ':noise'/':head' sub-keys to set independently"
            )
        if blocks:
            raise ValueError(
                f"{what}: unknown block {parts[2]!r} in {key!r}; {group} has learnable "
                f"posterior block(s) {list(blocks)}"
            )
        raise ValueError(
            f"{what}: {group} holds a single parameter posterior, so {key!r} addresses "
            f"nothing; name the declaration itself ({group!r})"
        )
    if key not in table:
        known = sorted(k for k, (_, _, d) in table.items() if not learnable or _slot_learnable(d))
        hint = (
            " (keys are slot-qualified: 'likelihood:<modality>', 'transition:<factor>', "
            "'initial_state_prior:<factor>', 'recognition:<variable>', 'scoring:<row>')"
            if ":" not in key
            else ""
        )
        raise ValueError(
            f"{what}: unknown slot key {key!r}{hint}; "
            f"{'learnable slots' if learnable else 'the slots'} are {known}"
        )
    slot, name, decl = table[key]
    if slot == "scoring":
        trainer = registry.scoring_trainer(name)
        if trainer is None:
            raise NotImplementedError(
                f"{what}: {key} names the scoring row {name!r}, which declares no trainer "
                "(registry.Estimator.trainer), so a rate for it would apply to nothing; the rows "
                "with one are "
                f"{sorted(n for n in registry._scoring_names() if registry.scoring_trainer(n))}"
            )
        # a scoring entry is one addressable unit: no sub-keys; its rate reaches the row's
        # trainer, never the per-tick learn face
        return
    if not learnable:
        return
    if not _slot_learnable(decl):
        if isinstance(decl, GradientDeclaration):
            raise ValueError(
                f"{what}: {key} holds a frozen {type(decl).__name__}, which has no "
                "trainable parameters; declare it with frozen=False to make the slot "
                "learnable (design: amortization.md A1 — frozen ⇔ no trainable marking)."
            )
        if isinstance(decl, DerivedInitial):
            raise ValueError(
                f"{what}: {key} holds a DerivedInitial — this factor's initial value is "
                "read off its own transition at t=0, so it has no parameters of its own "
                "to rate, freeze or learn (address transition:<factor> instead)."
            )
        if isinstance(decl, LGObservation):
            suggestion = "an InverseGamma or InverseWishart noise"
        elif slot == "initial_state_prior":
            suggestion = "a Dirichlet (or ConditionalDirichlet with a D_deps entry)"
        else:
            suggestion = "a ConditionalDirichlet"
        raise ValueError(
            f"{what}: {key} holds a {type(decl).__name__} (a point value); declare "
            f"{suggestion} to make it learnable."
        )


def _check_epistemic_key(
    system: System, key: str, params: SystemParams, variables: Mapping[str, _EpistemicVar]
) -> None:
    """One literal ``epistemic`` key: a known variable, or a loud error listing the namespace.

    Parameter-shaped keys defer to :func:`_check_slot_key`, so the selector and ``learn``'s rates
    share their errors; wildcard patterns go to :func:`_check_pattern`.
    """
    if key in variables:
        return
    for segment in key.split(":"):
        system._refuse_emission(segment, "epistemic")
    if key in system._table.control_names:
        raise ValueError(
            f"epistemic: {key!r} is a control node, which the System chooses rather than infers, "
            "so it carries no uncertainty to resolve; select state, observation or parameter "
            "variables"
        )
    parts = key.split(":")
    if parts[0] == "obs":
        mods, _ = (system.modality_names, system._facs)
        if len(parts) != 2:
            raise ValueError(
                f"epistemic: {key!r} is not a variable name; an observation variable is "
                "spelled 'obs:<modality>' (observations hold no parameter blocks, so "
                "there is no third segment)"
            )
        raise ValueError(
            f"epistemic: unknown observation variable {key!r}; the modalities are {sorted(mods)}"
        )
    if parts[0] == "scoring":
        raise ValueError(
            f"epistemic: {key!r} names a scoring row — an *objective*-side parameter set, "
            "which carries no planning-time novelty term of its own: planning scores what "
            "the world model does not know, and a critic is not a world claim (the same "
            "argument A11 makes for recognition modules). A row's own epistemic value, "
            "where it has one, is the row (ensemble_disagreement is selected by the "
            "variable it scores). Select likelihood/transition slots only."
        )
    if parts[0] in system._facs:
        _check_dim_key(system, key, params)
    if ":" in key:
        _check_slot_key(key, "epistemic", system._slot_declarations(params))
    states = sorted(k for k, v in variables.items() if v.kind == "state")
    obs_keys = sorted(k for k, v in variables.items() if v.kind == "obs")
    param_keys = sorted(k for k, v in variables.items() if v.kind == "param")
    raise ValueError(
        f"epistemic: unknown variable {key!r}; state variables are {states}, observation "
        f"variables are {obs_keys} and "
        f"learnable parameter variables are {param_keys}"
    )


def _check_dim_key(system: System, key: str, params: SystemParams) -> None:
    """A key whose first segment is a state factor that is not one of its dimensions: why not."""
    factor, _, rest = key.partition(":")
    names = _dim_names(system, params).get(system._facs.index(factor))
    if names is None:
        where = (
            "an amortized System's latent"
            if system._is_amortized(params)
            else "a categorical factor, which has levels rather than dimensions"
        )
        raise ValueError(
            f"epistemic: {key!r} selects a dimension of {factor!r}, {where}; select the factor "
            f"by its bare name {factor!r}"
        )
    if ":" in rest:
        raise ValueError(
            f"epistemic: {key!r} has {key.count(':') + 1} segments; a dimension of a factor is "
            f"spelled '{factor}:<dim>'"
        )
    how = "declared with System(dims=...)" if factor in dict(system.dims) else "named by position"
    raise ValueError(
        f"epistemic: {factor!r} has no dimension {rest!r}; its dimensions ({how}) are {list(names)}"
    )


def _check_pattern(
    system: System, pattern: str, what: str, params: SystemParams, *, epistemic: bool
) -> None:
    """Check one wildcard pattern's literal projection.

    A wildcard never validates less than its literal part. Every literal segment is checked against
    its finite domain (first segments, declared names, the family's posterior blocks), so
    ``"likelihod:*"`` raises. A pattern whose slot and name are literal gets a literal key's
    participation checks, and in ``epistemic`` a pinned ``initial_state_prior`` raises. A class
    empty through its wildcarded segment (``"likelihood:*"`` with no learnable likelihood) resolves
    silently to nothing.
    Design: docs/design/per-variable-infogain.md V7.
    """
    segs = pattern.split(":")
    mods, facs = (system.modality_names, system._facs)
    table = system._slot_declarations(params)
    hint = (
        "keys are '<state>', '<state>:<dim>', 'obs:<modality>', '<slot>:<name>' or "
        "'<slot>:<name>:<block>'"
        if epistemic
        else "rate keys are '<slot>:<name>' or '<slot>:<name>:<block>'"
    )
    if len(segs) == 1:  # "*" — every state variable
        if not epistemic:
            raise ValueError(
                f"{what}: {pattern!r} selects state variables, which hold no parameters "
                f"and take no rate; {hint}"
            )
        return
    if len(segs) > 3:
        raise ValueError(f"{what}: {pattern!r} has {len(segs)} segments; {hint}")
    kind = segs[0]
    if epistemic and kind in facs:  # "<factor>:*", every dimension of a factor
        if len(segs) != 2:
            raise ValueError(
                f"{what}: {pattern!r} — a dimension of a factor is spelled '{kind}:<dim>', so "
                f"'{kind}:*' (the whole factor) has two segments"
            )
        if facs.index(kind) not in _dim_names(system, params):
            _check_dim_key(system, pattern, params)
        return
    allowed = [k for k in _KEY_KINDS if k != "terminal"] if epistemic else sorted(_SLOT_WORDS)
    if kind != "*" and kind not in allowed:
        raise ValueError(
            f"{what}: unknown first segment {kind!r} in {pattern!r}; the legal first "
            f"segments are {allowed}{' or a state factor' if epistemic else ''} ({hint})"
        )
    if kind == "obs":
        if len(segs) == 3:
            raise ValueError(
                f"{what}: {pattern!r} — an observation variable is spelled "
                "'obs:<modality>'; observations hold no parameter blocks, so there is "
                "no third segment"
            )
        return  # segs[1] is the wildcard (a literal one is not a pattern)
    if epistemic and kind == "initial_state_prior":
        raise ValueError(
            f"epistemic: {pattern!r} pins the initial-state-prior slot, which has no "
            "planning-time novelty term (planning never revisits the episode start); "
            "select likelihood/transition slots only"
        )
    if epistemic and kind == "scoring":
        raise ValueError(
            f"epistemic: {pattern!r} pins the scoring slot, which holds *objective*-side "
            "parameters (the critic, the info-gain ensemble, a row's own model) and carries no "
            "planning-time novelty term: a scoring row is not a world claim (design: "
            "amortization.md A12, the same argument A11 makes for recognition). Rows are "
            "mounted by System.terms; select likelihood/transition slots here."
        )
    if epistemic and kind == "recognition":
        raise ValueError(
            f"epistemic: {pattern!r} pins the recognition slot, which holds *inference*-side "
            "parameters and carries no planning-time novelty term: planning scores what the "
            "world model does not know, and a recognition module is not a world claim "
            "(design: amortization.md A11). Its chartered epistemic landing is the "
            "ensemble_disagreement row (A10 disposition 3); select likelihood/transition "
            "slots only."
        )
    slots = _SLOT_WORDS if kind == "*" else (kind,)
    if segs[1] != "*":
        groups = [f"{s}:{segs[1]}" for s in slots if f"{s}:{segs[1]}" in table]
        if not groups:
            raise ValueError(
                f"{what}: unknown declaration name {segs[1]!r} in {pattern!r}; the "
                f"modalities are {sorted(mods)} and the factors are {sorted(facs)}"
            )
        if len(slots) == 1:  # slot AND name literal: the pattern pins one declaration
            _check_slot_key(groups[0], what, table)
            if len(segs) == 3 and segs[2] != "*":
                _check_slot_key(pattern, what, table)
            return
    if len(segs) == 3 and segs[2] != "*":
        candidates = [
            decl
            for key, (slot, _, decl) in table.items()
            if slot in slots
            and _slot_learnable(decl)
            and (segs[1] == "*" or key == f"{slot}:{segs[1]}")
        ]
        blocks = sorted({b for decl in candidates for b in _param_blocks(decl)})
        if candidates and segs[2] not in blocks:
            raise ValueError(
                f"{what}: unknown block {segs[2]!r} in {pattern!r}; the learnable "
                f"declaration(s) the pattern ranges over hold posterior block(s) {blocks}"
            )


def _check_plan(system: System, plan: Array, what: str, params: SystemParams) -> Array:
    """A policy array checked against this System's control nodes, dtype-normalized.

    Int32 level indices with one column per control node, or float vectors of the flat action
    width when every node is continuous; leading axes are the caller's. A space mixing the two
    has no positional plan: it is keyed by node.
    Design: docs/design/amortization.md A9.
    """
    plan = jnp.asarray(plan)
    space = system._space(params)
    if space.mixed:
        raise ValueError(
            f"{what}: this System's control space mixes discrete and continuous nodes, which no "
            "single array carries; pass the plan keyed by control node, {node: (P, H) levels or "
            "(P, H, d) values}"
        )
    layout = space.layout
    if plan.ndim not in (2, 3):
        tail = "action_dim" if layout is not None else "num_control_nodes"
        raise ValueError(
            f"{what} must be (num_policies, horizon, {tail}) or a single "
            f"(horizon, {tail}) plan, got shape {plan.shape}"
        )
    if layout is None:
        if not jnp.issubdtype(plan.dtype, jnp.integer):
            raise ValueError(f"{what} must be integer control indices, got {plan.dtype}")
        width = len(system._table.controls)
        if plan.shape[-1] != width:
            raise ValueError(
                f"{what} last axis is {plan.shape[-1]} but the System has {width} control "
                f"node(s) {list(system._table.control_names)}: a plan has one column per "
                "control node, in node order"
            )
        return plan.astype(jnp.int32)
    if jnp.issubdtype(plan.dtype, jnp.integer):
        raise ValueError(
            f"{what} has integer dtype {plan.dtype}, but this System's control nodes are "
            "continuous: a continuous control value is a float vector of their widths, not a "
            "level index "
            "(design: controls.py, amortization.md A9)"
        )
    if plan.shape[-1] != layout.action_dim:
        raise ValueError(
            f"{what} last axis is {plan.shape[-1]} but this System's flat continuous "
            f"control value is {layout.action_dim}-dimensional (its control nodes' widths, "
            "concatenated in node order)"
        )
    return plan


# ------------------------------------------------------------------- the observations


def _check_observations(modalities: tuple[str, ...], params, observations: tuple, face: str):
    """Refuse an observation whose form no face reads, with a sentence naming the modality.

    ``observations`` is the positional tuple (``None`` entries are absent). A linear-Gaussian
    modality takes an array of shape ``(obs_dim,)``: a Python list, a Python number, a scalar
    array or an array with a leading batch axis is refused (vmap the face over a batch). Any
    other modality refuses a Python list or tuple.
    """
    for name, decl, obs in zip(modalities, params.likelihood, observations, strict=False):
        if obs is None:
            continue
        if not isinstance(decl, LGObservation):
            if isinstance(obs, list | tuple):
                raise ValueError(
                    f"{face}: observation {name!r} is a Python {type(obs).__name__}; pass an "
                    "array (jnp.asarray(...)), a scalar integer level for a categorical modality"
                )
            continue
        want = (int(decl.obs_dim),)
        shape = tuple(getattr(obs, "shape", ()))
        if isinstance(obs, list | tuple) or shape != want:
            seen = (
                f"a Python {type(obs).__name__}"
                if not hasattr(obs, "shape")
                else f"an array of shape {shape}"
            )
            hint = (
                "; a batch of observations is a vmap over the face, not a leading axis"
                if len(shape) > 1 and shape[-1:] == want
                else ""
            )
            raise ValueError(
                f"{face}: observation {name!r} is a linear-Gaussian modality of dimension "
                f"{want[0]}, so its observation is an array of shape {want}; got {seen}{hint}"
            )
