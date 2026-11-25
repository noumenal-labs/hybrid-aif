"""Family-dispatched hybrid planning closures over distribution-object beliefs.

Beliefs (``Categorical | Gaussian | MixtureGaussian`` per factor) cross the
planner's scan as distribution objects, and every per-factor computation
dispatches on its declaration's type. Per step, a categorical factor contracts
through ``expected_states``; a Gaussian factor with no discrete parents predicts
exactly; a Gaussian or mixture factor with discrete parents grows through every
mode (weighted by the parents' post-transition belief) and is reduced back to
``mixture_k`` components. A mixture-family belief always carries exactly
``mixture_k`` components, padded with dead components on entry to planning. A
nonlinear transition propagates each Gaussian (or mixture component) to first order
or by the unscented transform, at the step's own belief. A table learned within the plan
(``System.plan_learning``) is read and absorbed through working counts the carry holds, and a
closed pair (a tabular factor reading through a cell the mixture that switches on it) is read
jointly over the mixture's components.

Design: docs/design/hybrid-system.md D1–D4; docs/design/nonlinear-transitions.md N2–N4;
docs/design/parameters-within-the-plan.md D1, D2
"""

import math
from collections.abc import Callable, Mapping
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hybrid_aif.cells import Grid, cell_mass, component_cell_mass, with_cell_parents
from hybrid_aif.controls import (
    LEVEL,
    ControlLayout,
    ControlRead,
    ControlSpace,
    ControlStep,
    ControlTable,
    SensedNodes,
    action_node_beliefs,
    control_read,
    declared_parents,
    has_control_axes,
    is_belief,
    is_distribution,
    lg_read,
    node_entry,
    take_controls,
)
from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalGaussian,
    Distribution,
    Gaussian,
    Learnable,
    MixtureGaussian,
    MLPConditional,
)
from hybrid_aif.distributions.gaussian import sigma_points
from hybrid_aif.distributions.truncated import TruncatedGaussian
from hybrid_aif.factors import (
    LGObservation,
    LinearGaussianObservation,
    LinearGaussianObservationMNIW,
    LinearGaussianTransition,
    NonlinearGaussianTransition,
    TabularObservation,
    TabularPrior,
    TabularTransition,
    lg_point,
    transition_point,
)
from hybrid_aif.inference.emission import is_emission
from hybrid_aif.inference.linear_gaussian import DiagonalNoise, condition_on_expected, predict
from hybrid_aif.inference.linearise import linear_model
from hybrid_aif.inference.mixture_rollout import propagate
from hybrid_aif.interval import Interval
from hybrid_aif.objectives import registry
from hybrid_aif.objectives.base import INFOGAIN, TermTree, merge
from hybrid_aif.objectives.continuous import (
    ComponentHead,
    DiagonalNoiseObservation,
    conditional_correlation,
    dense_target,
    diagonal_target_parts,
    emission_entropies,
    lg_ambiguity,
    lg_utility,
    linear_gaussian_readings,
    lowrank_capable,
    lowrank_utility,
    observation_correlation,
    observation_interaction,
    predictive_entropy,
    predictive_obs,
    risk,
    state_info_gain,
    sub_block_readings,
)
from hybrid_aif.objectives.discrete import (
    TargetParents,
    ambiguity,
    cross_entropy_risk,
    expected_obs,
    flat_risk,
    resolve_conditional_targets,
    resolve_step_targets,
    terminal_cross_entropy_risk,
    terminal_risk,
    utility,
)
from hybrid_aif.objectives.pooling import Pooled, scale_pooled
from hybrid_aif.objectives.presence import Presence, mask_absent, presence_at, static_absent

#: The term holding what a per-channel sum miscounts when several channels read one Gaussian
#: factor: the interaction information of their readings about the selected state under
#: ``utility_infogain``, its ambiguity correction ``−TC(Y | x_S)`` under ``risk_ambiguity``.
INTERACTION = "observation_interaction"
#: The term holding the total correlation of a step's predicted readings on one Gaussian factor,
#: part of the joint risk under ``risk_ambiguity``.
CORRELATION = "observation_correlation"

#: Log-weight of dead (padding) components (design: hybrid-system.md D2).
PAD_LOG_WEIGHT = -jnp.inf

CATEGORICAL = "categorical"
GAUSSIAN = "gaussian"
MIXTURE = "mixture"

Belief = Categorical | Gaussian | MixtureGaussian


def factor_families(
    B: tuple,
    D: tuple,
    B_deps: tuple[tuple[int, ...], ...],
    state_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Belief family per factor, derived from the declaration types.

    ``mixture`` means carried as a MixtureGaussian in planning: declared mixture
    priors, and Gaussian priors on factors with discrete parents. A neural
    (:class:`~hybrid_aif.distributions.MLPConditional`) transition is
    ``gaussian``-family and may not have discrete parents.
    """
    fams = []
    for f, (B_f, D_f) in enumerate(zip(B, D, strict=True)):
        if isinstance(B_f, MLPConditional):
            if B_deps[f]:
                raise NotImplementedError(
                    f"factor {state_names[f]!r}: a neural (MLPConditional) transition with "
                    f"discrete parent(s) {[state_names[g] for g in B_deps[f]]} would grow the "
                    "belief through one mode-conditioned net per mode; P1's prediction row is "
                    "the parentless linearized one (design: amortization.md A1(a), build plan "
                    "P1). Drop the parents, or declare a LinearGaussianTransition."
                )
            if not isinstance(D_f, Gaussian):
                raise ValueError(
                    f"factor {state_names[f]!r}: MLPConditional transition requires a "
                    f"Gaussian prior (D), got {type(D_f).__name__} (the linearized prediction "
                    "row carries a single Gaussian; mixture-carried neural factors are not "
                    "built — design: amortization.md A1(a))"
                )
            fams.append(GAUSSIAN)
        elif isinstance(B_f, TabularTransition):
            if not isinstance(D_f, TabularPrior):
                raise ValueError(
                    f"factor {state_names[f]!r}: tabular transition (B) requires a "
                    f"tabular prior (D) — Categorical/Dirichlet, or a conditional form "
                    f"with a System D_deps entry for hierarchical priors — got "
                    f"{type(D_f).__name__}"
                )
            fams.append(CATEGORICAL)
        elif isinstance(B_f, NonlinearGaussianTransition):
            # discrete state parents switch f (the mode reaches it as a level), so the
            # factor is carried as a mode-aligned mixture, as a switching linear one is
            if isinstance(D_f, MixtureGaussian):
                fams.append(MIXTURE)
            elif isinstance(D_f, Gaussian):
                fams.append(MIXTURE if B_deps[f] else GAUSSIAN)
            else:
                raise ValueError(
                    f"factor {state_names[f]!r}: NonlinearGaussianTransition requires a "
                    f"Gaussian or MixtureGaussian prior (D), got {type(D_f).__name__}"
                )
        elif isinstance(B_f, LinearGaussianTransition):
            if isinstance(D_f, MixtureGaussian):
                fams.append(MIXTURE)
            elif isinstance(D_f, Gaussian):
                fams.append(MIXTURE if B_deps[f] else GAUSSIAN)
            else:
                raise ValueError(
                    f"factor {state_names[f]!r}: LinearGaussianTransition requires a "
                    f"Gaussian or MixtureGaussian prior (D), got {type(D_f).__name__}"
                )
        else:
            raise ValueError(
                f"factor {state_names[f]!r}: unknown transition declaration type "
                f"{type(B_f).__name__}"
            )
    return tuple(fams)


def _valid_order(families: tuple[str, ...], B_deps: tuple[tuple[int, ...], ...]) -> list[int]:
    """A declaration order satisfying the parents-before-children rule."""
    remaining = list(range(len(families)))
    placed: set[int] = set()
    order: list[int] = []
    while remaining:
        progressed = False
        for f in list(remaining):
            if families[f] == CATEGORICAL or all(d in placed for d in B_deps[f]):
                order.append(f)
                placed.add(f)
                remaining.remove(f)
                progressed = True
        if not progressed:  # pragma: no cover - self-deps are rejected before ordering
            raise ValueError("within-step dependencies contain a cycle")
    return order


def _validate_lg_obs_shapes(
    A_m: LinearGaussianObservation,
    d: int,
    parent_shape: tuple[int, ...],
    modality_name: str,
    factor_name: str,
    selects: bool = False,
) -> None:
    """Validate a (possibly switching) LG observation's H/c/R_chol shapes.

    Each array is shared (event shape only) or per-mode (leading ``parent_shape``
    axes). ``d_obs`` comes from ``H``; ``d`` must match the observed factor. Named
    categorical parents require at least one per-mode array. A noise map on a switching
    channel is built where the channel ``selects`` (its parents are all action nodes, so the
    observed factor is read at one level); its leading level axes are ``parent_shape``.
    """
    d_obs = A_m.H.shape[-2]
    if A_m.noise_map:
        if parent_shape and not selects:
            raise NotImplementedError(
                f"modality {modality_name!r}: a noise map on a switching LG observation is built "
                "only for a channel whose mode parents are all action nodes; a map is read at "
                "the observed factor's mean, and a channel on a state mode parent has one mean "
                "per mode rather than one (design: planning-with-observations.md D3)"
            )
        noise = A_m.noise
        bad = [a for a in getattr(noise, "axes", ()) if not 0 <= a < d]
        if bad:
            raise ValueError(
                f"modality {modality_name!r}: the noise map's axes {noise.axes} name "
                f"coordinate(s) {bad} outside the observed factor {factor_name!r}'s {d} dimensions"
            )
        R_chol = A_m.R_chol_at(jnp.zeros(d, A_m.H.dtype))
    else:
        R_chol = A_m.R_chol
    specs = [
        ("H", A_m.H, 2, (d_obs, d)),
        ("c", A_m.c, 1, (d_obs,)),
        ("R_chol", R_chol, 2, (d_obs, d_obs)),
    ]
    any_per_mode = False
    for name, arr, event_ndims, event_shape in specs:
        got_event = arr.shape[arr.ndim - event_ndims :]
        if got_event != event_shape:
            raise ValueError(
                f"modality {modality_name!r}: {name} has trailing shape {got_event}, expected "
                f"{event_shape} (d_obs={d_obs} from H, state dim d={d} from factor "
                f"{factor_name!r}); H/c/R_chol must agree on d_obs and match the factor's dim"
            )
        leading = arr.shape[: arr.ndim - event_ndims]
        if leading == ():
            continue
        if leading != parent_shape:
            raise ValueError(
                f"modality {modality_name!r}: {name} has leading (per-mode) axes {leading}, "
                f"but the switching parents' shape is {parent_shape} "
                f"(sizes of the categorical A_deps parents); a per-mode array must lead with "
                "exactly the parent axes (the F_j convention), or be shared with none"
            )
        any_per_mode = True
    if parent_shape and not any_per_mode:
        raise ValueError(
            f"modality {modality_name!r} names categorical switching parent(s) in A_deps "
            f"(shape {parent_shape}) but none of H/c/R_chol carries per-mode axes, so the "
            "observation does not depend on the mode; drop the parents or make an array per-mode"
        )


def _check_mlp_parents(
    decl: MLPConditional,
    expected_dims: tuple[int, ...],
    child_dim: int | None,
    what: str,
    convention: str,
) -> None:
    """Check a neural declaration's parent widths (and child dim) against its slot.

    The slot supplies the parent order and widths; a discrete parent's width is
    its level count (one-hot). ``child_dim=None`` skips the child check.
    """
    if decl.parent_dims != expected_dims:
        raise ValueError(
            f"{what}: MLPConditional declares parent_dims={decl.parent_dims} but the slot "
            f"supplies parents of width {expected_dims} — {convention}"
        )
    if child_dim is not None and decl.child_dim != child_dim:
        raise ValueError(
            f"{what}: MLPConditional declares child_dim={decl.child_dim} but the slot's "
            f"child is {child_dim}-dimensional"
        )


def _validate_nonlinear(
    decl: NonlinearGaussianTransition,
    D: tuple,
    f: int,
    name: str,
    table: ControlTable,
    kinds: tuple[str, ...],
    sizes: tuple[int, ...],
) -> None:
    """Check a nonlinear transition against its factor: the innovations' parent axes are the
    factor's discrete parents in declared (``B_deps``) order, its discrete state parents (the
    modes that switch it) and its discrete control nodes, its dimension the prior's, and ``f``
    (by ``jax.eval_shape``) maps the ``(d,)`` state and the factor's parent entries in declared
    order (an int32 level per discrete state parent or discrete node, a float vector per
    continuous one) to a ``(d,)`` float mean."""
    D_f = D[f]
    declared = declared_parents(table, f)

    def discrete(what: str, i: int) -> bool:
        return what == "state" or not kinds or kinds[i] == LEVEL

    def levels(what: str, i: int) -> int:
        return D[i].num_levels if what == "state" else sizes[i]

    expected = tuple(levels(what, i) for what, i in declared if discrete(what, i)) or (1,)
    got = decl.innovations.parent_shape
    if got != expected:
        raise ValueError(
            f"factor {name!r}: innovations parent_shape {got} != {expected} (expected one axis "
            "per discrete parent, the discrete state parents that switch it and its discrete "
            "control nodes, in declared (B_deps) order, or (1,) for a factor with none)"
        )
    d = decl.dim
    if D_f.dim != d:
        raise ValueError(
            f"factor {name!r}: the NonlinearGaussianTransition's innovations are {d}-dimensional "
            f"and the prior (D) is {D_f.dim}-dimensional; they must agree"
        )
    layout = tuple(
        jax.ShapeDtypeStruct((), jnp.int32)
        if discrete(what, i)
        else jax.ShapeDtypeStruct((sizes[i],), jnp.float32)
        for what, i in declared
    )

    def entry(what: str, i: int) -> str:
        label = table.state_names[i] if what == "state" else table.control_names[i]
        if what == "state":
            return f"int32 mode level of {label!r}"
        return f"int32 level of {label!r}" if discrete(what, i) else f"float[{sizes[i]}] {label!r}"

    described = "(" + ", ".join(entry(what, i) for what, i in declared) + ")"
    x = jax.ShapeDtypeStruct((d,), decl.innovations.loc.dtype)
    try:
        out = jax.eval_shape(lambda x, controls: decl.f(x, controls), x, layout)
    except Exception as exc:
        raise ValueError(
            f"factor {name!r}: the NonlinearGaussianTransition's f fails on the factor's "
            f"layout, f(x: float[{d}], controls: {described}): {exc}"
        ) from exc

    def text(value) -> str:
        shape, dtype = getattr(value, "shape", None), getattr(value, "dtype", None)
        return f"{dtype}{list(shape)}" if shape is not None else type(value).__name__

    def is_vector(value) -> bool:
        dtype = getattr(value, "dtype", None)
        return (
            getattr(value, "shape", None) == (d,)
            and dtype is not None
            and jnp.issubdtype(dtype, jnp.floating)
        )

    call = f"f(x: float[{d}], controls: {described})"
    if decl._scaled:
        if not (isinstance(out, tuple) and len(out) == 2 and all(map(is_vector, out))):
            what = (
                "(" + ", ".join(text(v) for v in out) + ")"
                if isinstance(out, tuple | list)
                else text(out)
            )
            raise ValueError(
                f"factor {name!r}: the NonlinearGaussianTransition's f returns {what} for "
                f"{call}; under noise_scale='per_coordinate' it must return the pair (mean, "
                f"scale), (float[{d}], float[{d}])"
            )
    elif isinstance(out, tuple | list):
        raise ValueError(
            f"factor {name!r}: the NonlinearGaussianTransition's f returns a "
            f"{type(out).__name__} of {len(out)} entries for {call}, and noise_scale=None "
            f"declares the next state's mean alone, float[{d}]; a map that returns (mean, "
            "scale) is declared with noise_scale='per_coordinate'"
        )
    elif not is_vector(out):
        raise ValueError(
            f"factor {name!r}: the NonlinearGaussianTransition's f returns {text(out)} for "
            f"{call}; it must return the next state's mean, float[{d}]"
        )


def validate_structure(
    A: tuple,
    B: tuple,
    D: tuple,
    obs_targets: dict[str, Distribution],
    A_deps: tuple[tuple[int, ...], ...],
    B_deps: tuple[tuple[int, ...], ...],
    table: ControlTable,
    space: ControlSpace,
    mixture_k: int | None,
    modality_names: tuple[str, ...],
    state_names: tuple[str, ...],
    mixture_schedule: str = "runnalls",
    sensed: "SensedNodes | None" = None,
    cells: dict[int, Grid] | None = None,
) -> tuple[str, ...]:
    """Validate hybrid declarations; returns the per-factor belief families.

    Enforces family-consistent B/D pairs; that a continuous factor's ``B_deps``
    name only earlier categorical factors (the error prints a valid order);
    innovation, ``F`` and ``G`` shapes; ``mixture_k`` when any factor is mixture-carried;
    family-pure modalities; switching LG observations (``A_deps`` = one continuous
    factor, then categorical mode parents; the observed factor mixture-carried);
    and LG ``obs_targets`` as Gaussian/mixture densities over the observation
    space (a neural modality's C designation raises). Neural declarations are
    checked on parent widths. ``table`` and ``space`` are the System's control nodes and
    their kinds: a linear-Gaussian factor's mode axes are its discrete parents (state factors
    and discrete control nodes) in declared order, and ``G`` reads its continuous control
    nodes, concatenated in declared order.
    """
    facs, mods = state_names, modality_names
    families = factor_families(B, D, B_deps, facs)
    kinds, sizes = space.kinds, space.sizes
    for f, fam in enumerate(families):
        if fam != CATEGORICAL:
            continue
        for g in B_deps[f]:
            if families[g] != CATEGORICAL and (cells is None or g not in cells):
                raise ValueError(
                    f"factor {facs[f]!r}: its transition names the continuous factor {facs[g]!r} "
                    "as a parent, and a table reads a discrete parent; declare "
                    f"cells={{{facs[g]!r}: Grid(...)}} on the System to read the cell "
                    f"{facs[g]!r} falls in"
                )

    for f, fam in enumerate(families):
        if fam == CATEGORICAL:
            continue
        if isinstance(B[f], MLPConditional):
            # Transition-slot parent order (the AIF layer's convention, A1): the
            # factor's own previous value, then its control nodes in declared order —
            # a discrete node's one-hot, a continuous node's value (A9).
            # factor_families has already rejected discrete parents here.
            action_dims = tuple(sizes[k] for k, _ in table.control_parents[f])
            _check_mlp_parents(
                B[f],
                (D[f].dim, *action_dims),
                D[f].dim,
                f"factor {facs[f]!r}",
                "the factor's own previous value, then its control nodes in declared order — "
                "a continuous node's value (the continuous action vector), or the one-hot of a "
                "discrete level (the "
                "transition slot's parent order, amortization.md A1/A9)",
            )
            continue
        deps = B_deps[f]
        wrong_family = [facs[d] for d in deps if families[d] != CATEGORICAL]
        if wrong_family:
            raise ValueError(
                f"factor {facs[f]!r}: a continuous factor's B_deps may only name "
                f"categorical (discrete) factors, got {wrong_family} (self-dependence "
                "is excluded for continuous factors)"
            )
        late = [facs[d] for d in deps if d >= f]
        if late:
            order = ", ".join(repr(facs[i]) for i in _valid_order(families, B_deps))
            raise ValueError(
                f"within-step ordering: factor {facs[f]!r} (declaration index {f}) names "
                f"parent(s) {late} not declared earlier; factors are computed in "
                "declaration order and a continuous factor's discrete parents must "
                f"precede it. A valid declaration order is: ({order})."
            )
        if isinstance(B[f], NonlinearGaussianTransition):
            _validate_nonlinear(B[f], D, f, facs[f], table, kinds, sizes)
            continue
        read = lg_read(table, kinds, f)
        expected = tuple(D[i].num_levels if what == "state" else sizes[i] for what, i in read.axes)
        expected = expected or (1,)
        got = B[f].innovations.parent_shape
        if got != expected:
            raise ValueError(
                f"factor {facs[f]!r}: innovations parent_shape {got} != {expected} "
                "(expected one axis per discrete parent — state factors and discrete control "
                "nodes, in declared order — or (1,) for a factor with none)"
            )
        d = B[f].innovations.dim
        if B[f].F.shape not in ((d, d), expected + (d, d)):
            raise ValueError(
                f"factor {facs[f]!r}: F shape {B[f].F.shape} must be ({d}, {d}) — one map "
                f"shared across modes — or {expected + (d, d)} — per-mode F_j conditioned "
                "like the innovations (innovations parent_shape + (d, d))"
            )
        G = B[f].G
        du = sum(sizes[k] for k in read.floats)
        if G is None and read.floats:
            raise ValueError(
                f"factor {facs[f]!r} reads continuous control node(s) "
                f"{[table.control_names[k] for k in read.floats]}: declare its G, ({d}, {du}) "
                f"shared or {expected + (d, du)} per mode, for the x' = F x + G u + b term"
            )
        if G is not None and not read.floats:
            raise ValueError(
                f"factor {facs[f]!r} declares G, which reads continuous control nodes, and it "
                "reads none: give it a continuous control parent in B_deps (a node with a box "
                "in params.control), or drop G"
            )
        if G is not None and G.shape not in ((d, du), expected + (d, du)):
            raise ValueError(
                f"factor {facs[f]!r}: G shape {G.shape} must be ({d}, {du}) — shared across "
                f"modes — or {expected + (d, du)} — per mode, like F; du = {du} is the width "
                "of its continuous control nodes, concatenated in declared order"
            )

    if MIXTURE in families and mixture_k is None:
        which = [facs[f] for f, fam in enumerate(families) if fam == MIXTURE]
        raise ValueError(
            f"factor(s) {which} carry mixture beliefs, so System.mixture_k (the shared "
            "Runnalls budget) must be set"
        )

    if mixture_schedule == "mode_aligned":
        for f, fam in enumerate(families):
            if fam != MIXTURE:
                continue
            modes = 1
            for d in B_deps[f]:
                modes *= D[d].num_levels
            if mixture_k % modes != 0:
                raise ValueError(
                    f"mixture_schedule='mode_aligned': factor {facs[f]!r} grows through "
                    f"M={modes} modes, so System.mixture_k={mixture_k} must be a multiple "
                    f"of {modes} (each mode owns mixture_k//M slots); set mixture_k to a "
                    f"multiple of {modes}"
                )

    fam_count = len(families)

    def family_of(p: int) -> str:
        return CATEGORICAL if p >= fam_count else families[p]

    def name_of(p: int) -> str:
        return facs[p] if p < fam_count else sensed.names[p - fam_count]

    def levels_of(p: int) -> int:
        return D[p].num_levels if p < fam_count else sensed.levels[p - fam_count]

    for m, A_m in enumerate(A):
        deps = A_deps[m]
        if isinstance(A_m, TabularObservation):
            bad = [name_of(d) for d in deps if family_of(d) != CATEGORICAL]
            if bad:
                raise ValueError(
                    f"modality {mods[m]!r} is categorical but observes continuous "
                    f"factor(s) {bad}; a modality is family-pure in the factor it observes "
                    "(cross-family explaining-away is not supported; bin the continuous factor's "
                    "features into a tabular modality, or observe it through a "
                    "LinearGaussianObservation — docs/composing-systems.md, the family matrix)"
                )
        elif isinstance(A_m, LGObservation):
            if len(deps) < 1:
                raise ValueError(
                    f"modality {mods[m]!r}: an LG modality observes one continuous factor "
                    f"(optionally followed by categorical switching parents), or is a Gaussian "
                    f"emission over categorical parents alone, got A_deps {deps}"
                )
            obs_f, mode_parents = deps[0], deps[1:]
            #  a channel whose mode parents are all action nodes selects its head and noise at
            #  the step's level, so the Gaussian-carried factor it observes stays one Gaussian
            selects = (
                bool(mode_parents)
                and selects_by_nodes(A_m, family_of(obs_f))
                and all(g >= len(families) for g in mode_parents)
            )
            if is_emission(A_m):
                # the Gaussian emission over categorical parents alone (structure-learning.md
                # S2): every parent selects the offset and the noise, nothing is filtered
                non_cat = [name_of(g) for g in deps if family_of(g) != CATEGORICAL]
                if non_cat:
                    raise ValueError(
                        f"modality {mods[m]!r} is a Gaussian emission (a head of width one, the "
                        f"offset column only), whose parents are all categorical, but A_deps "
                        f"{deps} names non-categorical factor(s) {non_cat}; give the head a "
                        "regressor block for the continuous factor, or name categorical "
                        "parents only"
                    )
                if A_m.noise_map:
                    raise ValueError(
                        f"modality {mods[m]!r}: a noise map varies over an observed continuous "
                        "state, and a Gaussian emission over categorical parents observes none; "
                        "declare a per-configuration noise instead"
                    )
                if mods[m] in obs_targets:
                    raise NotImplementedError(
                        f"obs_targets[{mods[m]!r}]: a preference over a Gaussian emission's "
                        "observation is not built; its predictive is a mixture over the parents' "
                        "configurations, whose divergence from a Gaussian target has no closed "
                        "form (design: structure-learning.md S7)"
                    )
                parent_shape = tuple(levels_of(p) for p in deps)
                _validate_lg_obs_shapes(A_m, 0, parent_shape, mods[m], name_of(obs_f))
                continue
            if family_of(obs_f) == CATEGORICAL:
                raise ValueError(
                    f"modality {mods[m]!r} is linear-Gaussian but observes categorical "
                    f"factor {name_of(obs_f)!r}; a modality is family-pure in the factor it "
                    "observes "
                    "(cross-family explaining-away is not supported). Three routes work: bin the "
                    "features into a tabular modality over the discrete factor; carry a "
                    "MixtureGaussian factor observed by this modality and make the discrete factor "
                    "its switching parent (docs/composing-systems.md, the family matrix); or, "
                    "for a Gaussian whose mean and noise the discrete factor selects with no "
                    "continuous state behind it, declare a Gaussian emission (a head of width "
                    "one, gaussian_emission(loc, R_chol); design: structure-learning.md S2)"
                )
            non_cat = [name_of(p) for p in mode_parents if family_of(p) != CATEGORICAL]
            if non_cat:
                raise ValueError(
                    f"modality {mods[m]!r}: a switching LG observation's parents after the "
                    f"observed factor {name_of(obs_f)!r} must be categorical (the discrete mode), "
                    f"got non-categorical {non_cat} in A_deps {deps}"
                )
            parent_shape = tuple(levels_of(p) for p in mode_parents)
            _validate_lg_obs_shapes(
                A_m, D[obs_f].dim, parent_shape, mods[m], name_of(obs_f), selects
            )
            if mode_parents and family_of(obs_f) != MIXTURE and not selects:
                raise ValueError(
                    f"modality {mods[m]!r} is a switching LG observation (categorical parent(s) "
                    f"{[name_of(p) for p in mode_parents]}), but the observed factor "
                    f"{facs[obs_f]!r} "
                    f"is {family_of(obs_f)}-family; conditioning it on an uncertain mode grows it "
                    "to a mixture, so it must be carried as a mixture. Declare its D as a "
                    "MixtureGaussian (a 1-component mixture is fine) and set System.mixture_k."
                )
        elif isinstance(A_m, MLPConditional):
            if not deps:
                raise ValueError(
                    f"modality {mods[m]!r}: a neural (MLPConditional) modality observes at "
                    "least one factor, got empty A_deps"
                )
            _check_mlp_parents(
                A_m,
                tuple(levels_of(g) if family_of(g) == CATEGORICAL else D[g].dim for g in deps),
                None,
                f"modality {mods[m]!r}",
                "one parent per factor named in A_deps, in that order (a categorical parent "
                "as its belief's probability vector, the one-hot convention, "
                "amortization.md A1)",
            )
            continuous = [name_of(g) for g in deps if family_of(g) != CATEGORICAL]
            if len(continuous) > 1:
                raise NotImplementedError(
                    f"modality {mods[m]!r}: a neural modality names the continuous factors "
                    f"{continuous}; it is read as a linear-Gaussian model of one continuous "
                    "factor, so a neural modality over two continuous factors is not built "
                    "(design: recognition-beside-analytic.md L1). Merge the factors into one "
                    "or split the modality"
                )
            if continuous:
                if family_of(deps[0]) == CATEGORICAL:
                    raise ValueError(
                        f"modality {mods[m]!r}: a neural modality names its continuous factor "
                        f"{continuous[0]!r} first in A_deps, then its categorical parents, got "
                        f"A_deps {deps}"
                    )
                mode_parents = deps[1:]
                selects = (
                    bool(mode_parents)
                    and family_of(deps[0]) == GAUSSIAN
                    and all(g >= len(families) for g in mode_parents)
                )
                if mode_parents and family_of(deps[0]) != MIXTURE and not selects:
                    raise ValueError(
                        f"modality {mods[m]!r} is a switching neural observation (categorical "
                        f"parent(s) {[name_of(p) for p in mode_parents]}), but the observed "
                        f"factor {facs[deps[0]]!r} is {family_of(deps[0])}-family; conditioning "
                        "it on an uncertain mode grows it to a mixture, so it must be carried "
                        "as a mixture. Declare its D as a MixtureGaussian (a 1-component "
                        "mixture is fine) and set System.mixture_k."
                    )
        else:
            raise ValueError(
                f"modality {mods[m]!r}: unknown observation declaration type {type(A_m).__name__}"
            )

    for m, A_m in enumerate(A):
        if not (isinstance(A_m, LGObservation | MLPConditional) and mods[m] in obs_targets):
            continue
        neural = isinstance(A_m, MLPConditional)
        target = obs_targets[mods[m]]
        d_obs = A_m.child_dim if neural else A_m.H.shape[-2]
        kind = "a neural" if neural else "an LG"
        if not isinstance(target, Gaussian | MixtureGaussian | ConditionalGaussian):
            raise ValueError(
                f"obs_targets[{mods[m]!r}]: {kind} modality's C designation is a preference over "
                f"its *continuous* observation, so the target is a Gaussian or MixtureGaussian "
                f"over R^{d_obs} (covariance-as-tolerance, the same reading state_targets has), "
                f"or a ConditionalGaussian over it per level of its C_deps parents, got "
                f"{type(target).__name__}"
            )
        if target.dim != d_obs:
            obs_f = A_deps[m][0]
            read_off = "child_dim" if neural else "H"
            raise ValueError(
                f"obs_targets[{mods[m]!r}]: target dim {target.dim} != the modality's "
                f"observation dim {d_obs} (read off {read_off}); a C designation lives on the "
                f"observation space, not on the observed factor's state space (factor "
                f"{facs[obs_f]!r} is {D[obs_f].dim}-dimensional; designate it via state_targets "
                "if that is what you meant)"
            )
    return families


# ------------------------------------------------------------------ static-K padding


def pad_mixture(m: MixtureGaussian, k: int) -> MixtureGaussian:
    """Bring a mixture to exactly ``k`` components.

    Over budget reduces (Runnalls); under budget pads with dead components
    (``PAD_LOG_WEIGHT`` log-weight, component 0's moments).
    """
    n = m.num_components
    if n > k:
        return m.reduce(k)
    if n == k:
        return m
    pad = k - n
    dead = jnp.full(m.log_weights.shape[:-1] + (pad,), PAD_LOG_WEIGHT, m.log_weights.dtype)
    return MixtureGaussian(
        log_weights=jnp.concatenate([m.log_weights, dead], axis=-1),
        means=jnp.concatenate([m.means, jnp.repeat(m.means[..., :1, :], pad, axis=-2)], axis=-2),
        chols=jnp.concatenate([m.chols, jnp.repeat(m.chols[..., :1, :, :], pad, axis=-3)], axis=-3),
    )


def where_tree(flag, new, old):
    """``new`` where ``flag`` else ``old``, leaf by leaf, over two pytrees of one structure.

    Raises :class:`ValueError` on a leaf-shape mismatch rather than broadcasting.
    """
    import jax.tree_util as jtu

    def pick(a, b):
        if jnp.shape(a) != jnp.shape(b):
            raise ValueError(
                f"where_tree: the two branches differ in shape, {jnp.shape(a)} against "
                f"{jnp.shape(b)}; bring both to one shape (pad_mixture / prepare_beliefs) first"
            )
        return jnp.where(flag, a, b)

    return jtu.tree_map(pick, new, old)


def as_mixture(q: Gaussian | MixtureGaussian, k: int) -> MixtureGaussian:
    """Promote a Gaussian to a one-component mixture, then pad/reduce to ``k``."""
    if isinstance(q, Gaussian):
        q = MixtureGaussian(
            log_weights=jnp.zeros(q.batch_shape + (1,), q.loc.dtype),
            means=q.loc[..., None, :],
            chols=q.chol[..., None, :, :],
        )
    return pad_mixture(q, k)


def prepare_beliefs(
    beliefs: tuple[Belief, ...], families: tuple[str, ...], mixture_k: int | None
) -> tuple[Belief, ...]:
    """Beliefs as the planning carry: mixture-family factors at exactly ``mixture_k``."""
    return tuple(
        as_mixture(q, mixture_k) if fam == MIXTURE else q
        for q, fam in zip(beliefs, families, strict=True)
    )


class BeliefPair(NamedTuple):
    """A Gaussian-carried factor's planning beliefs at one step (design:
    planning-with-observations.md D1, D2): ``predictive``, predict only, which every leaf but
    the information gain reads (and every target, except a Gaussian one under
    ``System.feedback``, which also reads the conditioned covariances); ``epistemic``,
    predicted from the previous step's ``conditioned`` belief, the belief before this step's
    observations; ``conditioned``,
    that belief after conditioning on the observations expected at this step; and ``step``,
    the transitions taken, which indexes a per-step availability; ``noises``, by modality
    column, each conditioning modality's noise at the epistemic mean (a Cholesky factor, or a
    neural modality's :class:`~hybrid_aif.inference.linear_gaussian.DiagonalNoise`), which the
    chain read to condition and the information-gain leaves read again (empty: the leaves read
    the noise at the mean themselves)."""

    predictive: Gaussian
    epistemic: Gaussian
    conditioned: Gaussian
    step: Array
    noises: dict[int, "Array | DiagonalNoise"] = {}  # noqa: RUF012


class Conditioner(NamedTuple):
    """One linear-Gaussian modality the epistemic chain conditions on: the factor it observes,
    its column in the availability mask, its head, and the noise at a mean (a channel whose mode
    parents are action nodes reads per-level noise there: a noise map's levelled factors at the
    mean, or arrays that ignore it). A channel whose mode
    parents are action nodes (``nodes``, their engine positions) holds per-level arrays, and
    ``modality`` and ``factor_name`` name it in the sentence a non-point step is refused with;
    the step selects its level (:func:`select_conditioner`). A neural modality has no constant
    head: ``read(q, levels)`` gives ``(H, c, noise)`` linearised at the belief ``q``
    (:func:`neural_read`), ``levels`` the step's levels of its action-node parents, the noise a
    :class:`~hybrid_aif.inference.linear_gaussian.DiagonalNoise`, and ``H`` and ``c`` are zero
    templates of the head's shapes; :meth:`at` returns the conditioner as a constant one at
    ``q``, whose ``noise_at`` gives that noise, so the chain conditions on it in state space."""

    factor: int
    column: int
    H: Array
    c: Array
    #: The observation noise at a mean ``loc``: a lower-triangular Cholesky factor, or a
    #: :class:`~hybrid_aif.inference.linear_gaussian.DiagonalNoise` (state-space update).
    noise_at: Callable[[Array], Array | DiagonalNoise]
    nodes: tuple[int, ...] = ()
    modality: str = ""
    factor_name: str = ""
    read: Callable[..., tuple[Array, Array, DiagonalNoise]] | None = None

    def at(self, q: Gaussian) -> "Conditioner":
        """This conditioner read at the belief ``q``: itself when its head is constant."""
        if self.read is None:
            return self
        H, c, noise = self.read(q)
        return self._replace(H=H, c=c, noise_at=lambda loc, noise=noise: noise, read=None)


class NeuralHead(NamedTuple):
    """A neural modality as the step terms read it: the declaration, and the levels (one per
    categorical parent, in order) of the parents it is read at when they are action nodes at a
    point level; ``None`` when the categorical parents are beliefs (:func:`neural_model`)."""

    decl: MLPConditional
    levels: tuple[Array, ...] | None = None


def _mode_grid(sizes: tuple[int, ...]) -> tuple[Array, ...]:
    """One-hots of every joint configuration of categorical parents of the given ``sizes``,
    row-major: one ``(M, size)`` array per parent."""
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in sizes], indexing="ij"), axis=-1)
    grid = grid.reshape(-1, len(sizes))
    return tuple(jax.nn.one_hot(grid[:, i], n) for i, n in enumerate(sizes))


def neural_read(
    decl: MLPConditional,
    q: Gaussian | MixtureGaussian,
    sizes: tuple[int, ...] = (),
    levels: tuple[Array, ...] | None = None,
) -> tuple[Array, Array, DiagonalNoise]:
    """``(H, c, noise)`` of the neural conditional ``decl`` linearised at the belief ``q`` over
    its first parent, the declaration's ``propagation`` (:func:`~hybrid_aif.inference.linearise.
    linear_model`), the noise a :class:`~hybrid_aif.inference.linear_gaussian.DiagonalNoise`.

    The categorical parents (``sizes``, their widths) are held at the one-hots of ``levels`` when
    given. On a Gaussian the head is ``(d_obs, d)``. On a mixture it is one per component, and
    without ``levels`` one per joint mode configuration of the parents too: ``(K, M, ...)``
    (``M = 1`` with none), the switching form. Design: recognition-beside-analytic.md L1, L4."""
    prop = getattr(decl, "propagation", "first_order")

    def head_at(loc: Array, chol: Array, onehots: tuple[Array, ...]):
        return linear_model(decl, loc, chol, (loc, *onehots), 0, prop)

    fixed = (
        ()
        if levels is None
        else tuple(jax.nn.one_hot(lv, n) for lv, n in zip(levels, sizes, strict=True))
    )
    if isinstance(q, Gaussian):
        if sizes and levels is None:
            raise ValueError(
                "a neural modality with categorical parents is read at a Gaussian belief only "
                "where those parents are fixed at levels; a Gaussian-carried factor observed "
                "through an uncertain mode is carried as a mixture"
            )
        return head_at(q.loc, q.chol, fixed)
    comps = q.components
    if levels is not None or not sizes:
        out = jax.vmap(lambda loc, chol: head_at(loc, chol, fixed))(comps.loc, comps.chol)
        return jax.tree.map(lambda a: a[:, None], out)
    grid = _mode_grid(sizes)

    def per_component(loc: Array, chol: Array):
        return jax.vmap(lambda *onehots: head_at(loc, chol, onehots))(*grid)

    return jax.vmap(per_component)(comps.loc, comps.chol)


def _categorical_only(deps: tuple[int, ...], families: tuple[str, ...]) -> bool:
    """Whether every parent in ``deps`` is a categorical factor or an action node."""
    return all(g >= len(families) or families[g] == CATEGORICAL for g in deps)


def neural_emission(decl: MLPConditional) -> LinearGaussianObservation:
    """The Gaussian emission a neural conditional over categorical parents alone tabulates: for
    every joint parent configuration ``e``, ``c = m(e)`` and ``R = diag sigma(e)^2``, with no
    state columns (design: recognition-beside-analytic.md L4)."""
    sizes = tuple(decl.parent_dims)
    grid = _mode_grid(sizes)
    mean, scale = jax.vmap(lambda *onehots: decl.diag_moments(onehots))(*grid)
    d_obs = decl.child_dim
    mean = mean.reshape(*sizes, d_obs)
    noise = jax.vmap(jnp.diag)(scale).reshape(*sizes, d_obs, d_obs)
    return LinearGaussianObservation(H=jnp.zeros((*sizes, d_obs, 0)), c=mean, noise=noise)


def neural_model(head: NeuralHead, q: Gaussian | MixtureGaussian):
    """The neural modality ``head`` read at the belief ``q``: a point
    :class:`~hybrid_aif.factors.LinearGaussianObservation` over a Gaussian belief, a
    :class:`~hybrid_aif.objectives.continuous.ComponentHead` (a head per component and mode
    configuration) over a mixture."""
    sizes = tuple(head.decl.parent_dims[1:])
    H, c, noise = neural_read(head.decl, q, sizes, head.levels)
    R_chol = noise.chol()
    # first order reads the decoder's diagonal noise at the mean, straight from the head (not as
    # the diagonal of the dense factor, whose gradient would fill a d_obs by d_obs array); the
    # unscented noise also carries the residuals' correlated part, so it is read dense
    diag = noise.scale if noise.low_rank is None else None
    if isinstance(q, MixtureGaussian):
        return ComponentHead(H, c, R_chol, diag)
    obs = DiagonalNoiseObservation(H=H, c=c, noise=R_chol, scale=diag)
    return obs


def neural_conditioner(
    decl: MLPConditional,
    *,
    factor: int,
    column: int,
    nodes: tuple[int, ...] = (),
    modality: str = "",
    factor_name: str = "",
) -> Conditioner:
    """The :class:`Conditioner` of a neural modality over one Gaussian-carried factor, its
    head linearised at the belief the chain conditions (design: recognition-beside-analytic.md
    L4). ``nodes`` are its action-node parents, whose levels the step selects."""
    sizes = tuple(decl.parent_dims[1:])

    def read(q: Gaussian, levels: tuple[Array, ...] | None = None):
        return neural_read(decl, q, sizes, levels)

    def no_constant_noise(loc: Array) -> Array:
        raise ValueError(
            f"modality {modality!r}: a neural modality's noise is read at the belief "
            "(Conditioner.at), not at a mean"
        )

    d, d_obs = decl.parent_dims[0], decl.child_dim
    return Conditioner(
        factor=factor,
        column=column,
        H=jnp.zeros((d_obs, d)),
        c=jnp.zeros((d_obs,)),
        noise_at=no_constant_noise,
        nodes=nodes,
        modality=modality,
        factor_name=factor_name,
        read=read,
    )


def soft_selection_error(
    modality: str, factor: str, node: str, why: str, mixture: bool = False
) -> ValueError:
    """The refusal for a selecting channel whose action node is not a point."""
    if mixture:
        return ValueError(
            f"modality {modality!r} selects its noise map by action node {node!r}, and the "
            f"map over the mixture-carried factor {factor!r} it observes is read at one level "
            f"only while that node's belief is a point; here it is {why}. Clamp the node to "
            "one level"
        )
    return ValueError(
        f"modality {modality!r} selects its head and noise by action node {node!r}, and the "
        f"Gaussian-carried factor {factor!r} it observes stays one Gaussian only while that "
        f"node's belief is a point; here it is {why}. Carry {factor!r} as a mixture (a "
        "MixtureGaussian initial prior and System.mixture_k) so the channel mixes over the "
        "node's levels, or clamp the node to one level"
    )


def point_level(
    entry: object,
    column: int,
    sensed: SensedNodes,
    modality: str,
    factor: str,
    mixture: bool = False,
) -> Array:
    """The level a selecting channel reads action node ``column`` at, from the node's entry at
    one plan step: the clamped level under the delta kernel. A probability vector or a node with
    a declared actuator kernel is not a point and raises."""
    name = sensed.names[sensed.columns.index(column)]
    if is_belief(entry):
        raise soft_selection_error(
            modality, factor, name, "a probability vector over its levels at this step", mixture
        )
    if column in sensed.kernels:
        raise soft_selection_error(
            modality,
            factor,
            name,
            "spread over its levels by the kernel declared in params.actuator",
            mixture,
        )
    return jnp.asarray(entry, jnp.int32)


def prior_point_level(
    q: Categorical, node: str, modality: str, factor: str, mixture: bool = False
) -> Array:
    """The level of an action node's prior, which must be a point (one level holding all the
    mass): the efference copy under the delta kernel. A prior that is not raises, at once for a
    concrete one and when the computation runs for a traced one."""
    probs = q.probs
    peak = jnp.max(probs, axis=-1)
    level = jnp.argmax(probs, axis=-1).astype(jnp.int32)
    if isinstance(peak, jax.core.Tracer):
        why = "spread over its levels in the prior the call was given"
        error = soft_selection_error(modality, factor, node, why, mixture)
        return eqx.error_if(level, peak < 1.0 - 1e-6, str(error))
    if float(peak) < 1.0 - 1e-6:
        why = f"the prior {np.asarray(probs).tolist()}"
        raise soft_selection_error(modality, factor, node, why, mixture)
    return level


def _take_modes(arr: Array, event_ndims: int, levels: tuple[Array, ...]) -> Array:
    """``arr`` with its leading mode axes indexed at ``levels`` (a shared array has none)."""
    if arr.ndim == event_ndims:
        return arr
    for level in levels:
        arr = jnp.take(arr, level, axis=0)
    return arr


def select_observation(model: LinearGaussianObservation, levels: tuple[Array, ...]):
    """A linear-Gaussian observation's head and noise at the action nodes' ``levels``: a
    point declaration over its factor alone, which every unconditional channel's leaf scores."""
    if model.noise_map:
        noise = model.noise.select(levels) if model.noise_levels else model.noise
    else:
        noise = _take_modes(model.R_chol, 2, levels)
    return LinearGaussianObservation(
        H=_take_modes(model.H, 2, levels),
        c=_take_modes(model.c, 1, levels),
        noise=noise,
    )


def select_conditioner(
    cond: Conditioner, action: object, sensed: SensedNodes | None
) -> Conditioner:
    """``cond`` read at the step's levels of its action nodes: a plain conditioner is itself, a
    selecting one an unconditional conditioner with its level's head and noise."""
    if not cond.nodes:
        return cond
    levels = tuple(
        point_level(
            node_entry(action, sensed.columns[d - sensed.offset], sensed.layout),
            sensed.columns[d - sensed.offset],
            sensed,
            cond.modality,
            cond.factor_name,
        )
        for d in cond.nodes
    )
    if cond.read is not None:  # a neural head is read at the belief, at these levels
        return cond._replace(
            read=lambda q, read=cond.read: read(q, levels),
            nodes=(),
        )
    return cond._replace(
        H=_take_modes(cond.H, 2, levels),
        c=_take_modes(cond.c, 1, levels),
        noise_at=lambda loc, read=cond.noise_at: _take_modes(read(loc), 2, levels),
        nodes=(),
    )


def selects_by_nodes(model: object, family: str) -> bool:
    """Whether a linear-Gaussian channel over action nodes alone selects its head and noise at
    the nodes' levels before it is read: always on a Gaussian-carried factor, and on a
    mixture-carried one when the noise is a map, which each component reads at its own mean."""
    return family == GAUSSIAN or (family == MIXTURE and bool(getattr(model, "noise_map", False)))


def selecting_channels(A: tuple, A_deps: tuple[tuple[int, ...], ...], families, offset: int):
    """The linear-Gaussian channels that select by action node, each at its modality index with
    its mode parents' engine positions: those observing a Gaussian-carried factor whose mode
    parents are all action nodes (positions from ``offset``)."""
    return {
        m: deps[1:]
        for m, (a, deps) in enumerate(zip(A, A_deps, strict=True))
        if isinstance(a, LGObservation | MLPConditional)
        and not is_emission(a)
        and len(deps) > 1
        and deps[0] < offset
        and selects_by_nodes(a, families[deps[0]])
        and all(g >= offset for g in deps[1:])
    }


def _seed_noises(q: Gaussian, factor: int, conditioners: tuple["Conditioner", ...]) -> dict:
    """The ``noises`` a pair over ``factor`` carries from its seed: a conditioner's noise at the
    seed's mean (a Cholesky factor, or a neural modality's
    :class:`~hybrid_aif.inference.linear_gaussian.DiagonalNoise`), and zeros of its shape for one
    that selects by action node (its level is read at the step)."""
    out = {}
    for cond in conditioners:
        if cond.factor != factor:
            continue
        if cond.nodes and cond.read is not None:
            levels = tuple(jnp.zeros((), jnp.int32) for _ in cond.nodes)
            shape = jax.eval_shape(lambda cond=cond, levels=levels: cond.read(q, levels)[2])
            out[cond.column] = jax.tree.map(lambda a: jnp.zeros(a.shape, a.dtype), shape)
        elif cond.nodes:
            d_obs = cond.H.shape[-2]
            out[cond.column] = jnp.zeros((d_obs, d_obs), cond.H.dtype)
        else:
            out[cond.column] = cond.at(q).noise_at(q.loc)
    return out


def pair_beliefs(
    qs: tuple[Belief, ...],
    families: tuple[str, ...],
    conditioners: tuple["Conditioner", ...] = (),
) -> tuple[Belief, ...]:
    """The planning carry with each Gaussian-carried factor as a :class:`BeliefPair` seeded
    from its current belief, a posterior already, so nothing conditions the seed. With the
    ``conditioners`` the chain will condition on, each pair also carries their noise factors
    (:attr:`BeliefPair.noises`), so one read of a noise map per step serves the chain and the
    leaves."""
    return tuple(
        BeliefPair(q, q, q, jnp.asarray(0, jnp.int32), _seed_noises(q, f, tuple(conditioners)))
        if fam == GAUSSIAN
        else q
        for f, (q, fam) in enumerate(zip(qs, families, strict=True))
    )


def predictive_beliefs(qs: tuple[Belief, ...]) -> tuple[Belief, ...]:
    """The carry with every pair read as its predictive member."""
    return tuple(q.predictive if isinstance(q, BeliefPair) else q for q in qs)


def cell_view(qs: tuple[Belief, ...], cells: dict[int, Grid] | None) -> tuple[Belief, ...]:
    """``qs`` with each factor in ``cells`` (factor index to grid) replaced by its predictive
    belief's mass over the grid's cells, the form a tabular factor reads a cell parent in."""
    if not cells:
        return qs
    cat = with_cell_parents(predictive_beliefs(qs), cells)
    return tuple(cat[i] if i in cells else q for i, q in enumerate(qs))


def epistemic_beliefs(qs: tuple[Belief, ...]) -> tuple[Belief, ...]:
    """The carry with every pair read as its epistemic member, the belief before the step's
    observations."""
    return tuple(q.epistemic if isinstance(q, BeliefPair) else q for q in qs)


def conditioned_beliefs(qs: tuple[Belief, ...]) -> tuple[Belief, ...]:
    """The carry with every pair read as its conditioned member, the belief after the step's
    expected observations."""
    return tuple(q.conditioned if isinstance(q, BeliefPair) else q for q in qs)


def condition_expected(
    q: Gaussian,
    cond: Conditioner,
    flag: Array | None,
    noise: Array | DiagonalNoise | None = None,
) -> Gaussian:
    """``q`` conditioned on the observation ``cond`` is expected to give at its mean: zero
    innovation, so the mean is ``q.loc`` itself and the covariance takes the Kalman update;
    ``flag`` (``None``: present) leaves ``q`` where the modality is absent. ``noise`` is the
    noise at ``q``'s mean when the caller has read it (``None``: read here): a lower-triangular
    Cholesky factor, or a :class:`~hybrid_aif.inference.linear_gaussian.DiagonalNoise`, which
    takes the update in state space
    (:func:`~hybrid_aif.inference.linear_gaussian.condition_on_expected`)."""
    if noise is None:
        noise = cond.noise_at(q.loc)
    posterior = condition_on_expected(q, cond.H, noise)
    return posterior if flag is None else where_tree(flag, posterior, q)


# ---------------------------------------------------------------- transition closure


def _categorical_step(
    B_f: ConditionalCategorical,
    parent_beliefs: tuple[Categorical, ...],
    action_f: "Array | tuple[Array, ...] | ControlRead",
) -> Categorical:
    """v0.1 ``expected_states`` semantics for one factor; control-only parents allowed.

    ``action_f`` holds one level per trailing control axis (a scalar for the one axis; empty
    for a table without one), or a :class:`~hybrid_aif.controls.ControlRead`.
    """
    index = action_f if isinstance(action_f, tuple | ControlRead) else (action_f,)
    sliced = take_controls(B_f.log_table, index, log=True) if index else B_f.log_table
    if not parent_beliefs:
        return Categorical(logits=jax.nn.log_softmax(sliced, axis=-1))  # as expected_states
    return ConditionalCategorical(log_table=sliced, parent_ndims=len(parent_beliefs)).average(
        parent_beliefs
    )


class _LGInputs:
    """A linear-Gaussian factor's declaration read at one step's control entries: the
    innovations ``(loc, chol)``, ``F`` and ``G`` with its discrete control axes indexed away,
    and its continuous control input as ``(mean, root)``, ``root`` a factor of the input's
    covariance (``None`` for a clamp, the whole input ``None`` without one). ``chol`` has no mode
    axes when the innovations share one covariance factor across their parents (``shared``)."""

    def __init__(self, loc, chol, F, G, u_mean, u_root, trivial, shared=False):
        self.loc, self.chol, self.F, self.G = loc, chol, F, G
        self.u_mean, self.u_root, self.trivial = u_mean, u_root, trivial
        self.shared = shared


def _control_root(v) -> Array:
    """Square-root factor ``L`` with ``L Lᵀ = Cov(v)``, finite on zero-variance boxes."""
    if isinstance(v, Gaussian):
        return v.chol
    if isinstance(v, TruncatedGaussian):
        var = v.variance()
        pos = var > 0.0
        sd = jnp.where(pos, jnp.sqrt(jnp.where(pos, var, 1.0)), 0.0)
        return sd[..., :, None] * jnp.eye(v.dim, dtype=sd.dtype)
    return jnp.linalg.cholesky(v.cov)


def _lg_inputs(B_f: LinearGaussianTransition, read, entries: dict[int, object]) -> _LGInputs:
    """Index ``B_f``'s discrete control axes at their levels and gather its ``G u`` input.

    ``read`` is the factor's :class:`~hybrid_aif.controls.LGRead`; ``entries`` maps each
    control node it reads to that node's entry at this step. A continuous node's entry is its
    value (a clamp) or a Gaussian / truncated Gaussian over it (a belief, read through its
    first two moments). A noise prior is read at its point (:func:`transition_point`).
    """
    B_f = transition_point(B_f)
    loc, chol, F, G = B_f.innovations.loc, B_f.innovations.chol, B_f.F, B_f.G
    shared = B_f.innovations.shared_chol
    n = len(read.axes)
    controls = [(axis, i) for axis, (what, i) in enumerate(read.axes) if what == "control"]
    for axis, k in sorted(controls, reverse=True):
        level = entries[k]
        loc = jnp.take(loc, level, axis=axis - n - 1)
        if not shared:
            chol = jnp.take(chol, level, axis=axis - n - 2)
        if F.ndim > 2:
            F = jnp.take(F, level, axis=axis - n - 2)
        if G is not None and G.ndim > 2:
            G = jnp.take(G, level, axis=axis - n - 2)
        n -= 1
    u_mean = u_root = None
    if read.floats:
        values = [entries[k] for k in read.floats]
        if any(is_distribution(v) for v in values):
            means, roots = [], []
            for v in values:
                if is_distribution(v):
                    means.append(jnp.asarray(v.mean(), jnp.float32))
                    root = _control_root(v)
                    roots.append(jnp.asarray(root, jnp.float32))
                else:
                    v = jnp.asarray(v, jnp.float32)
                    means.append(v)
                    roots.append(jnp.zeros((v.shape[-1], v.shape[-1]), jnp.float32))
            u_mean = jnp.concatenate(means, axis=-1)
            u_root = jax.scipy.linalg.block_diag(*roots) if len(roots) > 1 else roots[0]
        else:
            u_mean = jnp.concatenate([jnp.asarray(v, jnp.float32) for v in values], axis=-1)
    return _LGInputs(loc, chol, F, G, u_mean, u_root, not read.axes, shared)


def _push_input(loc: Array, chol: Array, G: Array | None, u_mean, u_root) -> tuple[Array, Array]:
    """The innovation after the control input: ``b + G u`` for a clamp; for a belief
    ``N(m, S)`` over ``u``, mean ``b + G m`` and covariance ``Q + G S Gᵀ`` (exact for a Gaussian
    belief, the moment-matched push for a truncated one).

    The covariance is returned as the non-square factor ``[Q_chol | G S^½]``, ``u_root`` being
    ``S^½``: its product with its transpose is ``Q + G S Gᵀ``, which is all
    :func:`~hybrid_aif.inference.linear_gaussian.predict` reads, so no factorisation of the sum
    is taken and the result is exact whatever its rank."""
    if u_mean is None:
        return loc, chol
    loc = loc + (G @ u_mean[..., None])[..., 0]
    if u_root is None:
        return loc, chol
    pushed = G @ u_root
    batch = jnp.broadcast_shapes(chol.shape[:-2], pushed.shape[:-2])
    chol = jnp.broadcast_to(chol, batch + chol.shape[-2:])
    pushed = jnp.broadcast_to(pushed, batch + pushed.shape[-2:])
    return loc, jnp.concatenate([chol, pushed], axis=-1)


def _gaussian_step(q: Gaussian, inputs: _LGInputs) -> Gaussian:
    """Exact predict through the control-conditioned innovation (no discrete state parents).

    ``x' = F x + G u + b + ε``: a clamped ``u`` is an offset; a Gaussian belief over it pushes
    through exactly (mean ``G μ_u``, covariance ``G Σ_u Gᵀ`` added to the innovation); a truncated
    Gaussian belief pushes through its closed mean and variance, a moment-matched approximation.
    """
    loc, chol, F, G = inputs.loc, inputs.chol, inputs.F, inputs.G
    if inputs.trivial:  # the (1,) parent axis of a factor with no discrete parent
        loc = loc[..., 0, :]
        chol = chol if inputs.shared else chol[..., 0, :, :]
        if F.ndim > 2:
            F = F[..., 0, :, :]
        if G is not None and G.ndim > 2:
            G = G[..., 0, :, :]
    loc, chol = _push_input(loc, chol, G, inputs.u_mean, inputs.u_root)
    return predict(q, F, loc, chol)


#: The unscented transform's scaling (Julier & Uhlmann 2004; the scaled form of Julier 2002):
#: ``α = 1`` and ``κ = 0`` put the sigma points at ``μ ± √n`` columns of the covariance root
#: with weight ``1/(2n)`` each and none on the centre's mean; ``β = 2`` (optimal for a Gaussian
#: belief) adds ``1 − α² + β = 2`` to the centre's covariance weight, so every covariance weight
#: is positive and the propagated covariance is positive semidefinite.
UNSCENTED_KAPPA = 0.0
UNSCENTED_BETA = 2.0


def _push_conditional(
    g: Callable[[Array], tuple[Array, Array | None]],
    loc: Array,
    root: Array,
    propagation: str,
) -> tuple[Array, Array, Array | None]:
    """The moments of ``y`` under ``z ~ N(loc, root rootᵀ)`` and ``y | z ~ N(m(z), diag s(z) Q
    diag s(z))``, the predictive of a Gaussian transition whose noise is scaled by its input:
    ``g(z)`` returns ``(m(z), s(z))``, ``s`` ``None`` for a noise that does not depend on ``z``.

    Returns ``(mean, spread, ss)``: ``mean`` and ``spread`` the mean and covariance of ``m(z)``,
    and ``ss`` the second moment of the scale, ``E[s sᵀ]`` (``None`` without a scale). The
    predictive covariance is ``spread + Q ∘ ss`` (``∘`` elementwise), since
    ``E[diag s Q diag s] = Q ∘ E[s sᵀ]``: the law of total covariance, ``Cov y = Cov E[y|z] +
    E[Cov(y|z)]``. The caller adds the noise, ``Q`` itself without a scale.

    ``"first_order"``: ``m(loc)``, ``(J root)(J root)ᵀ`` with ``J = ∂m/∂z`` at ``loc``, and
    ``ss = s(loc) s(loc)ᵀ``, the scale read at the mean. ``"unscented"``: the sigma points of
    the belief (:data:`UNSCENTED_KAPPA`, :data:`UNSCENTED_BETA`), ``m``'s weighted mean and
    covariance over them, and ``ss = Σ_i w_i s_i s_iᵀ`` with the mean weights ``w_i``, the
    points' estimate of the expectation (not ``E[s] E[s]ᵀ``, which drops the scale's own
    spread). The neural and the nonlinear transitions both step through it."""
    if propagation == "first_order":
        jac = jax.jacfwd(lambda z: g(z)[0])(loc)
        pushed = jac @ root
        mean, scale = g(loc)
        ss = None if scale is None else scale[:, None] * scale[None, :]
        return mean, pushed @ pushed.T, ss
    points, weights = sigma_points(loc, root, kappa=UNSCENTED_KAPPA)
    ys, scales = jax.vmap(g)(points)
    mean = weights @ ys
    dev = ys - mean
    cov_weights = weights.at[0].add(UNSCENTED_BETA)  # α = 1: the centre's 1 − α² + β
    # The second moment is positive semidefinite (and Q ∘ ss with it, a Schur product) because
    # every mean weight is non-negative, which UNSCENTED_KAPPA = 0 gives; a negative kappa makes
    # the centre's weight negative and ss can be indefinite, so a change of kappa revisits this.
    ss = None if scales is None else (scales * weights[:, None]).T @ scales
    return mean, (dev * cov_weights[:, None]).T @ dev, ss


def _mlp_step(
    q: Gaussian,
    B_f: MLPConditional,
    action_parents: tuple[Array, ...],
) -> Gaussian:
    """First-order (extended-Kalman) moment propagation through a neural transition.

    With ``q = N(μ, Σ)`` and the net giving ``(m(x, a), diag σ(x, a)²)``:
    ``μ' = m(μ, a)``, ``Σ' = J Σ Jᵀ + diag σ(μ, a)²`` with ``J = ∂m/∂x |_μ``. Exact
    for an affine mean head, approximate otherwise. ``action_parents`` are already in
    parent form, one per control node the factor reads (a level's one-hot, a continuous value).
    The step is :func:`_push_conditional` with the head's scale ``σ`` and ``Q = I``, the path a
    nonlinear transition with a scaled noise takes.
    Design: docs/design/amortization.md A1
    """
    mu = q.loc
    tail = tuple(a.astype(mu.dtype) for a in action_parents)
    mean, spread, ss = _push_conditional(
        lambda x: B_f.diag_moments((x, *tail)), mu, q.chol, "first_order"
    )
    return Gaussian.from_cov(mean, spread + jnp.eye(mean.shape[-1], dtype=ss.dtype) * ss)


class _NLInputs(NamedTuple):
    """A nonlinear transition's inputs at one step: the innovation ``(loc, chol)`` at the
    discrete parent levels (mode and control); ``controls``, one entry per parent in declared
    order (an int32 level for a discrete state parent or control node, a float vector, or
    ``None`` where a belief sits); and ``beliefs``, each belief-valued continuous entry as
    ``(position, mean, root)``, ``root`` a factor of its covariance."""

    loc: Array
    chol: Array
    controls: tuple
    beliefs: tuple[tuple[int, Array, Array], ...]


def _nl_inputs(
    B_f: NonlinearGaussianTransition,
    parents: tuple[tuple[int, int], ...],
    kinds: tuple[str, ...],
    entries: dict[int, object],
    modes: tuple[tuple[int, Array], ...] = (),
) -> _NLInputs:
    """Read ``B_f``'s parents at this step: its control parents (``(node, position)`` pairs)
    at ``entries`` and its discrete state parents at the mode levels ``modes`` (``(position,
    level)`` pairs), merged in declared order. A discrete parent, mode or control node, indexes
    the innovations and reaches ``f`` as its level; a continuous node's clamp reaches ``f`` as
    its value, and a Gaussian or truncated Gaussian belief over it through its first two
    moments."""
    slots = sorted(
        [(pos, "control", k) for k, pos in parents] + [(pos, "mode", lvl) for pos, lvl in modes],
        key=lambda slot: slot[0],
    )
    levels, controls, beliefs = [], [], []
    for at, (_, what, value) in enumerate(slots):
        if what == "mode":
            level = jnp.asarray(value, jnp.int32)
            levels.append(level)
            controls.append(level)
            continue
        k = value
        entry = entries[k]
        if kinds[k] == LEVEL:
            level = jnp.asarray(entry, jnp.int32)
            levels.append(level)
            controls.append(level)
        elif is_distribution(entry):
            root = _control_root(entry)
            mean = jnp.asarray(entry.mean(), jnp.float32)
            beliefs.append((at, mean, jnp.asarray(root, jnp.float32)))
            controls.append(None)
        else:
            controls.append(jnp.asarray(entry, jnp.float32))
    noise = transition_point(B_f).innovations.condition_on(tuple(levels) or (0,))
    return _NLInputs(noise.loc, noise.chol, tuple(controls), tuple(beliefs))


def _nonlinear_moments(
    B_f: NonlinearGaussianTransition, loc: Array, chol: Array, inputs: _NLInputs
) -> tuple[Array, Array, Array | None]:
    """The mean and covariance of ``f(x, controls)`` under ``x ~ N(loc, chol cholᵀ)`` and the
    step's control beliefs, before the innovations, and the second moment of ``f``'s scale
    (``None`` under ``noise_scale=None``): the joint ``z = (x, u_beliefs)`` with a
    block-diagonal root is pushed through ``f`` by :func:`_push_conditional`, to first order
    (``f(μ_z)``, ``J Σ_z Jᵀ`` with ``J = ∂f/∂z`` at ``μ_z``, which is ``J_x Σ J_xᵀ + J_u S
    J_uᵀ``) or by the unscented transform, the scale read over the same ``z``."""
    d = loc.shape[-1]
    spans, at = [], d
    for _, mean, _ in inputs.beliefs:
        spans.append((at, at + mean.shape[-1]))
        at += mean.shape[-1]
    loc_z = jnp.concatenate([loc, *(mean for _, mean, _ in inputs.beliefs)], axis=-1)
    roots = [chol, *(root for _, _, root in inputs.beliefs)]
    root_z = jax.scipy.linalg.block_diag(*roots) if len(roots) > 1 else chol

    def g(z: Array) -> tuple[Array, Array | None]:
        controls = list(inputs.controls)
        for (pos, _, _), (lo, hi) in zip(inputs.beliefs, spans, strict=True):
            controls[pos] = z[lo:hi]
        return B_f._mean_and_scale(z[:d], tuple(controls))

    return _push_conditional(g, loc_z, root_z, B_f.propagation)


def _nonlinear_step(q: Gaussian, B_f: NonlinearGaussianTransition, inputs: _NLInputs) -> Gaussian:
    """``x' = f(x, controls) + ε`` for a Gaussian belief: the moments of ``f``'s output
    (:func:`_nonlinear_moments`) plus the innovation's offset and covariance, ``Q = chol
    cholᵀ``, or under a scale ``Q ∘ E[s sᵀ]``, the expected ``diag(s) Q diag(s)``."""
    mean, cov, ss = _nonlinear_moments(B_f, q.loc, q.chol, inputs)
    noise = inputs.chol @ inputs.chol.T
    if ss is not None:
        noise = noise * ss
    return Gaussian.from_cov(mean + inputs.loc, cov + noise)


def _nonlinear_mixture_step(
    q: MixtureGaussian,
    B_f: NonlinearGaussianTransition,
    read: Callable[[tuple[Array, ...]], _NLInputs],
    mode_shape: tuple[int, ...],
    parent_beliefs: tuple[Categorical, ...],
    k: int,
    schedule: str,
    *,
    log_weights: Array | None = None,
    aux=None,
):
    """A mixture-carried factor through a nonlinear transition, switched by its discrete state
    parents: each of the ``K`` components through :func:`_nonlinear_step` once per joint mode
    ``j`` of the ``M = prod(mode_shape)`` (``read`` gives the step's inputs at the mode's
    parent levels: ``f`` takes them, the innovations are conditioned on them), weighted by the
    component's weight times the mode's belief (:func:`_mode_log_weights`), then reduced to
    ``k`` as :func:`_mixture_step` reduces. The grown mixture is component-major (slot
    ``k M + j``), :func:`~hybrid_aif.inference.mixture_rollout.propagate`'s layout, so the
    mode-aligned reduction keeps each mode's slots. With no state parent ``M = 1`` and the step
    is the per-component one. ``log_weights`` and ``aux`` are :func:`_mixture_step`'s. Design:
    docs/design/nonlinear-transitions.md N4 (amended 2026-11-15)."""
    M = math.prod(mode_shape)
    if log_weights is not None:
        logw = log_weights
    else:
        logw = _mode_log_weights(parent_beliefs) if parent_beliefs else jnp.zeros((M,))

    def mode(j: Array) -> Gaussian:
        levels = jnp.unravel_index(j, mode_shape) if mode_shape else ()
        inputs = read(tuple(levels))

        def one(mean: Array, chol: Array) -> Gaussian:
            return _nonlinear_step(Gaussian(loc=mean, chol=chol), B_f, inputs)

        return jax.vmap(one)(q.means, q.chols)

    stepped = jax.vmap(mode)(jnp.arange(M))  # (M, K, ...)
    K, d = q.num_components, q.dim
    means = jnp.swapaxes(stepped.loc, 0, 1).reshape((K * M, d))
    chols = jnp.swapaxes(stepped.chol, 0, 1).reshape((K * M, d, d))
    log_w = (q.log_weights[:, None] + logw).reshape((K * M,))
    grown = MixtureGaussian(log_weights=log_w, means=means, chols=chols)
    return _reduce_grown(grown, k, schedule, M, aux)


def _reduce_grown(grown: MixtureGaussian, k: int, schedule: str, M: int, aux=None):
    """The reduce half of grow and reduce, under ``schedule``; with ``aux`` (a per-component
    side statistic of the grown mixture) ``(mixture, aux)``, the statistic merged by the
    assignment that merges the moments."""
    if schedule == "mode_aligned":
        return grown.reduce_mode_aligned(k, M, aux=aux)
    return grown.reduce(k, aux=aux)


def _step_pair(
    q: "Gaussian | BeliefPair",
    step: Callable[[Gaussian], Gaussian],
    conditioners: tuple[Conditioner, ...] | list[Conditioner],
    present: Array | None,
    at_step: Callable[[Conditioner], Conditioner] | None = None,
) -> "Gaussian | BeliefPair":
    """One Gaussian-carried factor through ``step``; a :class:`BeliefPair` steps both members,
    the epistemic one from the previous conditioned belief and then conditioned on the
    ``conditioners`` that ``present`` (the step's ``(M,)`` bool vector) admits, each read at the
    step's action nodes by ``at_step`` (:func:`select_conditioner`). Each conditioner's noise is
    read once, at the epistemic mean (conditioning leaves the mean unchanged), and a pair that
    carries ``noises`` returns them for the step's leaves."""
    if not isinstance(q, BeliefPair):
        return step(q)
    epistemic = step(q.conditioned)
    conditioned = epistemic
    noises = {}
    for cond in conditioners:
        flag = None if present is None else present[cond.column]
        cond_t = cond if at_step is None else at_step(cond)
        cond_t = cond_t.at(epistemic)  # a neural head is read at the belief it conditions
        # the mean is the epistemic one throughout: conditioning leaves it unchanged
        noises[cond.column] = cond_t.noise_at(epistemic.loc)
        conditioned = condition_expected(conditioned, cond_t, flag, noises[cond.column])
    return BeliefPair(
        step(q.predictive), epistemic, conditioned, q.step + 1, noises if q.noises else {}
    )


def _mode_log_weights(parent_beliefs: tuple[Categorical, ...]) -> Array:
    """Joint log mode weights over the discrete parents, flattened row-major."""
    lw = parent_beliefs[0].log_probs
    for b in parent_beliefs[1:]:
        lw = lw[..., :, None] + b.log_probs[..., None, :]
        lw = lw.reshape(lw.shape[:-2] + (-1,))
    return lw


def _mixture_step(
    q: MixtureGaussian,
    B_f: LinearGaussianTransition,
    parent_beliefs: tuple[Categorical, ...],
    inputs: _LGInputs,
    k: int,
    schedule: str,
    *,
    log_weights: Array | None = None,
    aux=None,
):
    """The switching composition: exact grow through all modes, then reduce(k).

    ``schedule`` is ``"runnalls"`` (merge anywhere) or ``"mode_aligned"`` (Runnalls
    within each producing mode's group, so slots stay mode-owned). The modes are the discrete
    state parents; the control input enters each mode's offset (and innovation, for a belief).
    ``log_weights`` replaces the modes' joint log belief (``(M,)``) by given weights, ``(M,)``
    or per component ``(K, M)`` (the joint read of a closed pair,
    parameters-within-the-plan.md D2); ``aux`` is a per-component side statistic of the grown
    mixture (``(K M, ...)``, component-major) merged through the reduce, and the step then
    returns ``(mixture, aux)``.
    """
    d = B_f.dim
    bs = inputs.loc.reshape((-1, d))
    M = bs.shape[0]
    Q_chols = (
        jnp.broadcast_to(inputs.chol, (M, d, d))
        if inputs.shared
        else inputs.chol.reshape((-1, d, d))
    )
    if log_weights is not None:
        logw = log_weights
    else:
        logw = _mode_log_weights(parent_beliefs) if parent_beliefs else jnp.zeros((M,))
    F = inputs.F
    Fs = jnp.broadcast_to(F, (M, d, d)) if F.ndim == 2 else F.reshape((-1, d, d))
    if inputs.u_mean is not None:
        G = inputs.G
        du = G.shape[-1]
        Gs = jnp.broadcast_to(G, (M, d, du)) if G.ndim == 2 else G.reshape((-1, d, du))
        bs, Q_chols = _push_input(bs, Q_chols, Gs, inputs.u_mean, inputs.u_root)
    grown = propagate(q, Fs, bs, Q_chols, logw)
    return _reduce_grown(grown, k, schedule, M, aux)


_NO_PROCESS_NOISE = (
    "a Gaussian (linear or nonlinear) transition's process noise has a zero on its Cholesky "
    "factor's diagonal: a belief's covariance must stay positive definite, and a dimension with no "
    "process noise collapses to a point once a sampled rollout draws it, after which every "
    "prediction is NaN. A constant tracked as a state takes a small process noise (design: "
    "per-variable-infogain.md V10). A generative process may be noiseless; a belief may not."
)


def _with_process_noise(
    B_f: LinearGaussianTransition | NonlinearGaussianTransition,
) -> LinearGaussianTransition | NonlinearGaussianTransition:
    """``B_f`` for propagating beliefs: refused when its process noise is singular on a
    dimension (a zero diagonal entry of its factor), at once for a concrete factor and when
    the computation runs for a traced one. A noise prior is lowered to its point first
    (:func:`~hybrid_aif.factors.transition_point`)."""
    B_f = transition_point(B_f)
    chol = B_f.innovations.chol
    if not isinstance(chol, jax.core.Tracer):
        if np.any(np.diagonal(np.asarray(chol), axis1=-2, axis2=-1) == 0.0):
            raise ValueError(_NO_PROCESS_NOISE)
        return B_f
    singular = jnp.any(jnp.diagonal(chol, axis1=-2, axis2=-1) == 0.0)
    guarded = eqx.error_if(chol, singular, _NO_PROCESS_NOISE)
    return eqx.tree_at(lambda b: b.innovations.chol, B_f, guarded)


# ------------------------------------------------- parameters learned within the plan


class WorkingTable(NamedTuple):
    """One table a plan learns within the plan (design: parameters-within-the-plan.md D1): its
    key in the carry (``transition:<factor>``, ``likelihood:<modality>`` or
    ``parameter:<name>``), the ``readers`` that read it and absorb into it (``("transition",
    f)`` or ``("likelihood", m)``), and whether its counts carry a leading component axis, one
    working table per component of the mixture a closed pair's reader reads (``components``,
    D2)."""

    key: str
    readers: tuple[tuple[str, int], ...]
    components: bool = False


class WorkingCarry(NamedTuple):
    """The planning carry of a plan that learns tables within the plan (built by
    ``system.core._carry_learning`` and :func:`~hybrid_aif.objectives.compiled.make_learning`):
    the belief tuple, the working counts after the step's absorption, and the counts the step
    read (before it), which its novelty and every other step leaf read. Keyed as
    ``System.plan_learning`` resolves its tables. On the fused all-tabular path ``reads``
    keeps, per learned reader (``"<slot>:<index>"``), what a branch needs to absorb its outcome
    in place of the step's expected statistic: a transition's parents as the step read them, a
    likelihood's reading count (:func:`~hybrid_aif.objectives.compiled.make_learning`); empty on
    the hybrid path. Design: parameters-within-the-plan.md D1, A3.
    """

    beliefs: tuple
    counts: dict
    before: dict
    reads: dict = {}  # noqa: RUF012 (a NamedTuple default, never mutated)


class PlanLearning(NamedTuple):
    """The tables a plan carries as working counts (:attr:`System.plan_learning` resolved), and
    the closed pairs (categorical factor to the mixture-carried factor it reads through a cell
    and switches), whose reads are joint over the mixture's components (D2)."""

    tables: tuple[WorkingTable, ...]
    pairs: tuple[tuple[int, int], ...] = ()

    def of(self, slot: str, i: int) -> WorkingTable | None:
        """The working table reader ``(slot, i)`` reads, or ``None`` (it reads frozen)."""
        return next((t for t in self.tables if (slot, i) in t.readers), None)

    def seed(self, A: tuple, B: tuple, k: int | None) -> dict[str, Array]:
        """The working counts a plan starts from: each table's concentration, the posterior
        the plan starts from, broadcast to ``k`` components for a closed pair's table."""
        out = {}
        for t in self.tables:
            slot, i = t.readers[0]
            n = (A if slot == "likelihood" else B)[i].concentration
            out[t.key] = jnp.broadcast_to(n, (k, *n.shape)) if t.components else n
        return out


def closed_pairs(
    B: tuple,
    B_deps: tuple[tuple[int, ...], ...],
    families: tuple[str, ...],
    cells: dict[int, Grid] | None,
) -> dict[int, int]:
    """The closed pairs: a tabular factor ``f`` reading exactly one mixture-carried factor ``g``
    through a cell, where ``g``'s (linear or nonlinear Gaussian) transition switches on ``f``.
    Maps ``f`` to ``g``. Design: parameters-within-the-plan.md D2."""
    cells = cells or {}
    out = {}
    for f, fam in enumerate(families):
        if fam != CATEGORICAL or f >= len(B_deps):
            continue
        gs = [
            g
            for g in B_deps[f]
            if g in cells
            and g < len(families)
            and families[g] == MIXTURE
            and f in B_deps[g]
            and isinstance(B[g], LinearGaussianTransition | NonlinearGaussianTransition)
        ]
        if len(gs) == 1:
            out[f] = gs[0]
    return out


def make_plan_learning(
    resolved: tuple[tuple[str, tuple[tuple[str, int], ...]], ...],
    A: tuple,
    B: tuple,
    A_deps: tuple[tuple[int, ...], ...],
    B_deps: tuple[tuple[int, ...], ...],
    families: tuple[str, ...],
    cells: dict[int, Grid] | None,
    *,
    modality_names: tuple[str, ...],
    state_names: tuple[str, ...],
) -> PlanLearning:
    """The :class:`PlanLearning` of ``System._plan_learning``'s resolved tables, its closed
    pairs found (:func:`closed_pairs`). Refused with a sentence: a likelihood learned within the
    plan whose modality reads anything but categorical state factors, and a parameter variable
    read in a closed pair by one reader and learned within the plan by another (D2)."""
    pairs = closed_pairs(B, B_deps, families, cells)
    tables = []
    for key, readers in resolved:
        components = any(slot == "transition" and i in pairs for slot, i in readers)
        if components and len(readers) > 1:
            names = [
                f"{slot}:{(state_names if slot == 'transition' else modality_names)[i]}"
                for slot, i in readers
            ]
            raise NotImplementedError(
                f"plan_learning: {key!r} is read by {names}, and one of them reads it in a closed "
                "pair (a categorical factor reading a mixture-carried factor through a cell that "
                "switches on it), where the working counts are kept per component of the "
                "mixture; a table shared between such a reader and another reader learned within "
                "the plan is not built (design: parameters-within-the-plan.md D2); learn one of "
                "them within the plan, or declare the readers' tables apart"
            )
        for slot, m in readers:
            if slot != "likelihood":
                continue
            bad = [g for g in A_deps[m] if g >= len(families) or families[g] != CATEGORICAL]
            if bad or A[m].parent_ndims != len(A_deps[m]):
                raise NotImplementedError(
                    f"plan_learning: likelihood:{modality_names[m]} is learned within the plan "
                    "from the expected observation over its parents' predicted beliefs, which "
                    "is built for a modality reading categorical state factors alone; this one "
                    "reads an action node or a continuous factor (design: "
                    "parameters-within-the-plan.md D1)"
                )
        tables.append(WorkingTable(key, tuple(readers), components))
    return PlanLearning(tuple(tables), tuple(sorted(pairs.items())))


def _with_counts(decl, counts: Array):
    """``decl`` (a ``ConditionalDirichlet``) holding ``counts`` as its concentration."""
    return eqx.tree_at(lambda d: d.concentration, decl, counts)


def _log_mean(counts: Array, parent_ndims: int) -> Array:
    """The log posterior-mean table of a child-first count table (leading axes batch)."""
    return jnp.log(counts / counts.sum(axis=-(parent_ndims + 1), keepdims=True))


def _outer(vectors: tuple[Array, ...]) -> Array:
    """The outer product of probability vectors, in order."""
    out = vectors[0]
    for v in vectors[1:]:
        out = out[..., None] * v
    return out


def expected_joint(counts: Array, parent_ndims: int, parents: tuple[Array, ...]) -> Array:
    """The expected count statistic of a step whose child is not observed: the model's joint
    over child and parents, ``mean(counts)[c | p] · Π q(p_i)`` (child-first, the parents in
    table order). The exact expectation of ``onehot(c) ⊗ onehot(p)`` under the table's mean and
    the parent beliefs; the product ``q(c) ⊗ Π q(p_i)`` ``learn`` absorbs from beliefs differs
    from it wherever child and parents are dependent (their KL is the mutual information) (design:
    parameters-within-the-plan.md D1, resolved 2026-11-17)."""
    mean = counts / counts.sum(axis=-(parent_ndims + 1), keepdims=True)
    return mean * _outer(parents)


def _table_parents(
    decl,
    f: int,
    state_probs: tuple[Array, ...],
    action: object,
    table: ControlTable,
    layout: ControlLayout | None,
) -> tuple[Array, ...]:
    """Factor ``f``'s parent vectors in its table's axis order, the count statistic's
    parents as ``learn`` forms them: the state parents' probabilities (``B_deps`` order), a
    control axis as the step's level one-hot (a belief-valued step's probability vector), and
    the size-1 axis of a table with no control parent as a one-hot."""
    states = iter(state_probs)
    out: list[Array] = []
    dtype = decl.concentration.dtype
    for what, k in declared_parents(table, f):
        if what == "state":
            out.append(next(states))
            continue
        entry = node_entry(action, k, layout)
        size = decl.parent_shape[len(out)]
        out.append(
            jnp.asarray(entry, dtype)
            if is_belief(entry)
            else jax.nn.one_hot(jnp.asarray(entry, jnp.int32), size, dtype=dtype)
        )
    if decl.parent_ndims == len(out) + 1:
        out.append(jax.nn.one_hot(0, decl.parent_shape[-1], dtype=dtype))
    return tuple(out)


class PairRead(NamedTuple):
    """A closed pair's tabular factor read per component of the mixture it reads (D2):
    ``child`` the per-component child beliefs ``(K, L)``, ``parents`` the per-component state
    parent probabilities in ``B_deps`` order (``(K, ·)`` each, the cell parent the component's
    cell mass), ``weights`` the components' normalized weights ``(K,)``."""

    child: Array
    parents: tuple[Array, ...]
    weights: Array


def pair_read(
    log_table: Array,
    parent_ndims: int,
    f: int,
    g: int,
    B_deps: tuple[tuple[int, ...], ...],
    read: tuple[Belief, ...],
    mixture: MixtureGaussian,
    grid: Grid,
    index,
) -> PairRead:
    """Factor ``f`` read per component of ``g``'s pre-step ``mixture``: component ``i`` has cell
    mass ``c_i`` and child belief ``q_i = table_i · c_i`` (the other parents read as ``read``
    holds them). ``log_table`` is ``(L, *parents)``, or ``(K, L, *parents)`` per component;
    ``index`` is the step's control read of ``f`` (``()`` without one)."""
    masses = component_cell_mass(mixture, grid)  # (K, cells)
    K = masses.shape[0]
    shared = tuple(
        None if h == g else jnp.broadcast_to(read[h].probs, (K, read[h].probs.shape[-1]))
        for h in B_deps[f]
    )
    parents = tuple(masses if p is None else p for p in shared)

    def one(lt: Array, ps: tuple[Array, ...]) -> Array:
        beliefs = tuple(Categorical.from_probs(jnp.maximum(q, 1e-30)) for q in ps)
        return _categorical_step(
            ConditionalCategorical(log_table=lt, parent_ndims=parent_ndims), beliefs, index
        ).probs

    per = log_table.ndim > parent_ndims + 1
    child = jax.vmap(one, in_axes=(0 if per else None, 0))(log_table, parents)
    return PairRead(child, parents, jax.nn.softmax(mixture.log_weights, axis=-1))


def make_transition(
    B: tuple,
    B_deps: tuple[tuple[int, ...], ...],
    families: tuple[str, ...],
    mixture_k: int | None,
    mixture_schedule: str = "runnalls",
    *,
    table: ControlTable,
    space: ControlSpace,
    conditioners: tuple[Conditioner, ...] = (),
    present: Presence | None = None,
    sensed: SensedNodes | None = None,
    cells: dict[int, Grid] | None = None,
    learning: PlanLearning | None = None,
    A: tuple = (),
    A_deps: tuple[tuple[int, ...], ...] = (),
):
    """Family-dispatched one-step transition over distribution-object beliefs.

    Factors are computed in declaration order: categorical factors read
    pre-transition parent beliefs, mode-coupled continuous factors their discrete
    parents' post-transition beliefs. A learnable transition predicts through its
    mean table; a neural one through :func:`_mlp_step`. ``table`` is the System's
    :class:`~hybrid_aif.controls.ControlTable` and ``space`` its node kinds and sizes. The
    ``action`` (a control step) is one level per control node (an int array), the flat float
    vector of an
    all-continuous space, or a :class:`~hybrid_aif.controls.ControlStep` (a mixed space, or
    a belief-valued step). Each reader reads a control node its own way: a table indexes (or,
    under a belief, contracts) its control axis; a linear-Gaussian transition takes a discrete
    node as a mode index and a continuous one through ``G u`` (``x' = F_j x + G_j u + b_j +
    ε_j``); a neural one takes a discrete node's one-hot and a continuous node's value as
    inputs; a nonlinear one (:class:`~hybrid_aif.factors.NonlinearGaussianTransition`) gives
    ``f`` a discrete node's level and a continuous node's value, a belief over a continuous
    node entering through its moments. A Gaussian-carried factor arriving as a
    :class:`BeliefPair` steps both members, the epistemic one after conditioning on its
    ``conditioners`` (the linear-Gaussian modalities observing it) where ``present``
    (:class:`~hybrid_aif.objectives.presence.Presence`) admits them at the step; a conditioner
    that selects by action node reads its level from the step's ``action`` (``sensed`` names
    the observed action nodes). A tabular factor reads a parent in ``cells`` (factor index to
    :class:`~hybrid_aif.cells.Grid`) as the mass of its belief in each cell
    (:func:`~hybrid_aif.cells.cell_mass`).

    A closed pair (:func:`closed_pairs`: a tabular factor reading through a cell the
    mixture-carried factor that switches on it) is read jointly over the mixture's components:
    component ``i`` (its pre-step belief) has cell mass ``c_i`` and child belief ``q_i``, the
    switching factor grows with the per-component weights ``w_i q_i[j]``, and the tabular
    factor's belief is the marginal ``Σ_i w_i q_i`` (design: parameters-within-the-plan.md D2).

    With ``learning`` (:class:`PlanLearning`) the closure is ``transition(qs, action, working)
    -> (qs, working)``: a table learned within the plan is read through the working counts'
    mean, and each of its readers absorbs the step's expected joint (:func:`expected_joint`;
    ``A`` and ``A_deps`` are the likelihoods and their parents, for a learned likelihood); a
    closed pair's counts are per component, absorb ``onehot(j) ⊗ c_i`` into the grow's ``(i,
    j)`` child and merge through the reduce by the moments' assignment (D1, D2).
    """
    cells = cells or {}
    declared = B
    B = tuple(b.point() if isinstance(b, Learnable) else b for b in B)
    B = tuple(
        _with_process_noise(b)
        if isinstance(b, LinearGaussianTransition | NonlinearGaussianTransition)
        else b
        for b in B
    )
    kinds, sizes, layout = space.kinds, space.sizes, space.layout
    conditioners_of: dict[int, list[Conditioner]] = {}
    for cond in conditioners:
        conditioners_of.setdefault(cond.factor, []).append(cond)
    lg_reads = {
        f: lg_read(table, kinds, f)
        for f, b in enumerate(B)
        if isinstance(b, LinearGaussianTransition)
    }
    pairs = closed_pairs(declared, B_deps, families, cells)
    tables = learning.tables if learning is not None else ()

    def table_of(slot: str, i: int) -> WorkingTable | None:
        return None if learning is None else learning.of(slot, i)

    def step(qs: tuple[Belief, ...], action: Array, working: dict) -> tuple:
        if not isinstance(action, ControlStep):
            action = jnp.asarray(action) if layout is not None else jnp.asarray(action, jnp.int32)

        def entries(f: int) -> dict[int, object]:
            return {k: node_entry(action, k, layout) for k, _ in table.control_parents[f]}

        def at_step(cond: Conditioner) -> Conditioner:
            return select_conditioner(cond, action, sensed)

        pair = next((q for q in qs if isinstance(q, BeliefPair)), None)
        present_t = (
            None
            if present is None or pair is None or not conditioners
            else presence_at(present, action, pair.step, layout)
        )

        new = list(qs)
        read = cell_view(qs, cells)
        absorbed: dict[str, Array] = {}  # a plain working table's summed statistics
        merged: dict[str, Array] = {}  # a closed pair's counts after the grow and reduce
        joint: dict[int, tuple] = {}  # g -> (f, its PairRead, working table or None, index)

        def absorb(key: str, stats: Array) -> None:
            absorbed[key] = stats if key not in absorbed else absorbed[key] + stats

        for f, fam in enumerate(families):
            if fam == CATEGORICAL:
                if has_control_axes(table, f, B[f].parent_ndims):
                    index = control_read(action, f, table)
                else:
                    index = ()
                wt = table_of("transition", f)
                if f in pairs:
                    g = pairs[f]
                    pn = B[f].parent_ndims
                    log_table = _log_mean(working[wt.key], pn) if wt is not None else B[f].log_probs
                    pr = pair_read(log_table, pn, f, g, B_deps, read, qs[g], cells[g], index)
                    new[f] = Categorical.from_probs(jnp.sum(pr.weights[:, None] * pr.child, 0))
                    joint[g] = (f, pr, wt)
                    continue
                parents = tuple(read[i] for i in B_deps[f])
                B_f = B[f] if wt is None else _with_counts(declared[f], working[wt.key]).point()
                new[f] = _categorical_step(B_f, parents, index)
                if wt is not None:
                    held = predictive_beliefs(qs)
                    state_probs = tuple(
                        cell_mass(held[i], cells[i]) if i in cells else qs[i].probs
                        for i in B_deps[f]
                    )
                    decl = declared[f]
                    absorb(
                        wt.key,
                        expected_joint(
                            working[wt.key],
                            decl.parent_ndims,
                            _table_parents(decl, f, state_probs, action, table, layout),
                        ),
                    )
            elif isinstance(B[f], MLPConditional):
                given = entries(f)
                inputs = tuple(
                    jax.nn.one_hot(given[k], sizes[k])
                    if kinds[k] == LEVEL
                    else jnp.asarray(given[k])
                    for k, _ in table.control_parents[f]
                )
                new[f] = _step_pair(
                    qs[f],
                    lambda q, B_f=B[f], inputs=inputs: _mlp_step(q, B_f, inputs),
                    conditioners_of.get(f, ()),
                    present_t,
                    at_step,
                )
            elif isinstance(B[f], NonlinearGaussianTransition) and fam == GAUSSIAN:
                inputs = _nl_inputs(B[f], table.control_parents[f], kinds, entries(f))
                new[f] = _step_pair(
                    qs[f],
                    lambda q, B_f=B[f], inputs=inputs: _nonlinear_step(q, B_f, inputs),
                    conditioners_of.get(f, ()),
                    present_t,
                    at_step,
                )
            elif fam == GAUSSIAN:
                inputs = _lg_inputs(B[f], lg_reads[f], entries(f))
                new[f] = _step_pair(
                    qs[f],
                    lambda q, inputs=inputs: _gaussian_step(q, inputs),
                    conditioners_of.get(f, ()),
                    present_t,
                    at_step,
                )
            else:
                mode_shape = tuple(new[i].probs.shape[-1] for i in B_deps[f])
                extra: dict = {}
                if f in joint:
                    h, pr, wt = joint[f]
                    mode_beliefs = tuple(
                        Categorical.from_probs(jnp.maximum(pr.child, 1e-30)) if i == h else new[i]
                        for i in B_deps[f]
                    )
                    extra["log_weights"] = _mode_log_weights(mode_beliefs)  # (K, M)
                    if wt is not None:
                        extra["aux"] = _pair_grown_counts(
                            working[wt.key],
                            declared[h],
                            h,
                            pr,
                            B_deps[f].index(h),
                            mode_shape,
                            action,
                            table,
                            layout,
                        )
                if isinstance(B[f], NonlinearGaussianTransition):
                    states = tuple(
                        pos
                        for pos, (what, _) in enumerate(declared_parents(table, f))
                        if what == "state"
                    )
                    given = entries(f)

                    def nl_read(levels, f=f, states=states, given=given) -> _NLInputs:
                        return _nl_inputs(
                            B[f],
                            table.control_parents[f],
                            kinds,
                            given,
                            tuple(zip(states, levels, strict=True)),
                        )

                    out = _nonlinear_mixture_step(
                        qs[f],
                        B[f],
                        nl_read,
                        mode_shape,
                        tuple(new[i] for i in B_deps[f]),
                        mixture_k,
                        mixture_schedule,
                        **extra,
                    )
                else:
                    out = _mixture_step(
                        qs[f],
                        B[f],
                        tuple(new[i] for i in B_deps[f]),
                        _lg_inputs(B[f], lg_reads[f], entries(f)),
                        mixture_k,
                        mixture_schedule,
                        **extra,
                    )
                if "aux" in extra:
                    out, merged[joint[f][2].key] = out
                new[f] = out
        for t in tables:
            for slot, m in t.readers:
                if slot != "likelihood":
                    continue
                parents = tuple(new[g].probs for g in A_deps[m])
                absorb(t.key, expected_joint(working[t.key], A[m].parent_ndims, parents))
        after = dict(working)
        for key, stats in absorbed.items():
            after[key] = working[key] + stats
        after.update(merged)
        return tuple(new), after

    if learning is None or not tables:

        def transition(qs: tuple[Belief, ...], action: Array) -> tuple[Belief, ...]:
            return step(qs, action, {})[0]

        return transition

    def learned_transition(qs: tuple[Belief, ...], action: Array, working: dict) -> tuple:
        return step(qs, action, working)

    return learned_transition


def _pair_grown_counts(
    counts: Array,
    decl,
    f: int,
    pr: PairRead,
    pos: int,
    mode_shape: tuple[int, ...],
    action: object,
    table: ControlTable,
    layout: ControlLayout | None,
) -> Array:
    """A closed pair's working counts on the grown mixture, component-major ``(K M, ...)``:
    the grow's ``(i, j)`` child holds component ``i``'s counts plus ``onehot(l_j) ⊗ c_i`` (and
    the other parents, the control axes' one-hots), ``l_j`` the tabular factor's level in joint
    mode ``j`` (its position ``pos`` among the switching factor's mode parents). Design:
    parameters-within-the-plan.md D2."""
    M = math.prod(mode_shape)
    levels = jnp.unravel_index(jnp.arange(M), mode_shape)[pos]
    onehots = jax.nn.one_hot(levels, decl.num_levels, dtype=counts.dtype)  # (M, L)
    outer = jax.vmap(lambda ps: _outer(_table_parents(decl, f, ps, action, table, layout)))(
        pr.parents
    )  # (K, *parents)
    P = outer.ndim - 1
    stats = onehots[None, :, :, *([None] * P)] * outer[:, None, None]  # (K, M, L, *parents)
    grown = counts[:, None] + stats
    return grown.reshape((counts.shape[0] * M, *counts.shape[1:]))


# --------------------------------------------------------------------- term closures


def _shares_factorisation(chans: tuple[int, ...], masks: tuple) -> bool:
    """Whether the channels ``chans`` of one factor take the fused evaluation of their leaves:
    none has a partial selection of its dependency set."""
    return all(masks[m] is None for m in chans)


def _fuses_correlation(chans: tuple[int, ...]) -> bool:
    """Whether the channels ``chans`` of one factor take the fused evaluation of their total
    correlation under ``risk_ambiguity`` (a partial selection does not affect it)."""
    return True


def make_step_terms(
    A: tuple,
    B: tuple,
    A_deps: tuple[tuple[int, ...], ...],
    B_deps: tuple[tuple[int, ...], ...],
    families: tuple[str, ...],
    obs_targets: dict[str, Distribution],
    state_targets: dict[str, Distribution],
    *,
    decomposition: str,
    state_infogain_modalities: tuple[int, ...],
    param_infogain_modalities: tuple[int, ...],
    param_infogain_factors: tuple[int, ...],
    modality_names: tuple[str, ...],
    state_names: tuple[str, ...],
    mounts: tuple[registry.Mount, ...] = (),
    channel_selection: tuple[tuple[bool, ...] | None, ...] = (),
    factor_dims: tuple[tuple[int, ...] | None, ...] = (),
    control_deps: tuple[tuple[int, ...], ...] | None = None,
    present: Presence | None = None,
    scoring: Mapping[str, object] | None = None,
    branching: bool = False,
    cross_entropy_targets: frozenset[str] = frozenset(),
    c_deps: Mapping[str, TargetParents] | None = None,
    layout: ControlLayout | None = None,
    sensed: SensedNodes | None = None,
    cells: dict[int, Grid] | None = None,
    pooled: Pooled | None = None,
    learning: PlanLearning | None = None,
):
    """Per-step term evaluator over distribution-object beliefs.

    Tabular modalities use :mod:`hybrid_aif.objectives.discrete`. A modality's C designation
    is scored against its predicted observation, as ``utility`` under ``utility_infogain`` or
    ``risk`` under ``risk_ambiguity``. ``state_infogain_modalities`` are the channels the
    ``epistemic`` selector mounts: each carries an ``infogain`` leaf under ``utility_infogain``
    and an ``ambiguity`` leaf under ``risk_ambiguity``, and no other channel carries either.
    Under ``risk_ambiguity`` a channel without a designation carries an implicit flat
    preference, its ``risk`` leaf ``−H[q(o)]`` (the preference's normaliser dropped, as under
    ``utility_infogain``, where its utility is 0 and mounts no leaf); a Gaussian emission over
    categorical parents alone is such a channel. Where several linear-Gaussian channels observe
    one Gaussian-carried factor alone, two factor-keyed terms hold what a per-channel sum
    miscounts: under ``utility_infogain``, ``observation_interaction`` is the interaction
    information ``TC(Y) − TC(Y | x_S)`` of the mounted channels' readings about the selected
    state; under ``risk_ambiguity``, ``observation_correlation`` is the total correlation
    ``TC(Y)`` of every present channel's readings (part of the joint risk, mounted whatever the
    selection), and ``observation_interaction`` is ``−TC(Y | x_S)`` over the mounted channels
    when a sub-block is selected (part of the joint ambiguity). So ``G`` is the joint
    expected free energy in both forms, and the two are equal under full selection (design:
    per-variable-infogain.md V11). The ``*_modalities``/``*_factors`` arguments are resolved
    declaration indices (from ``System._term_targets``). Epistemic leaves come from
    :mod:`hybrid_aif.objectives.registry`; ``mounts`` are ``System.terms``' resolved rows. With
    ``param_infogain_factors`` nonempty, or a mounted row that reads the previous step, the
    closure needs ``qs_prev`` as its fourth argument. ``channel_selection`` holds one parent
    mask per modality (``A_deps`` order, ``None`` = all; empty = all channels in full); a
    mounted channel with a partial mask scores its info gain and ambiguity relative to the
    selected parents (design: per-variable-infogain.md V9). ``factor_dims`` holds, per factor,
    the selected dimensions of a Gaussian-carried factor (``None`` = the whole factor; empty =
    every factor whole): each mounted channel observing it scores ``I(y; x_S)``, or ``E H[y |
    x_S]`` under ``risk_ambiguity`` (design: per-variable-infogain.md V10). ``control_deps`` is
    ``System.B_control_deps``. A :class:`BeliefPair` in the carry is read as its predictive
    member by every consumer but the linear-Gaussian information-gain leaves and, under
    ``utility_infogain``, the interaction, which read its epistemic member
    (``StepContext.qs_epistemic``). ``pooled`` multiplies a declared table's novelty leaf by the
    count of the step's level of its control node. ``present`` is a
    :class:`~hybrid_aif.objectives.presence.Presence`: a modality absent at a step has its
    leaves zeroed there and drops out of its factor's interaction and correlation.
    ``branching`` marks a closure built for a branching (closed-loop) planner,
    where each state target is scored by its cross-entropy ``−E_q log C(s)``, linear in the
    belief, in place of its KL (design: per-variable-infogain.md V11 decision 6); the key stays
    ``risk[<factor>]``. ``cross_entropy_targets`` names the factors whose state targets are
    scored by that cross-entropy in an open-loop closure, the targets a closed-loop pass corrects
    (:mod:`hybrid_aif.objectives.closed_loop`). ``c_deps`` names the parents of each conditional
    target (``System.C_deps`` resolved to columns and engine positions): the target is resolved
    at each step under the step's control entries and beliefs
    (:func:`~hybrid_aif.objectives.discrete.resolve_step_targets`, ``layout`` the flat layout of
    a mixed control space) and then scored as an unconditional one.
    ``sensed`` names the action nodes some modality observes: ``A_deps`` then holds their
    engine positions after the state factors, and each step's belief tuple is extended by
    ``P(a | u_t)`` of each (:func:`~hybrid_aif.controls.action_node_beliefs`) for the channels'
    parent lookups: an action node's belief at a step is a parent only, no leaf is keyed by it
    and nothing about it is carried, and a channel over action nodes alone has no
    information-gain leaf (its ambiguity, risk and utility leaves are scored as any
    channel's). A tabular channel or a Gaussian
    emission contracts over the node's belief; a linear-Gaussian channel over a Gaussian-carried
    factor whose mode parents are all action nodes takes its head and noise at the node's level,
    which must be a point (:func:`point_level`), and is scored as an unconditional channel over
    its factor. ``cells`` (factor index to :class:`~hybrid_aif.cells.Grid`) names the continuous
    factors a tabular factor reads as a parent: ``StepContext.qs_prev_cells`` holds the previous
    step's beliefs with each such factor's mass over its cells, which the transition leaves read
    as a categorical parent; ``qs_prev`` keeps the Gaussian or mixture.

    With ``learning`` (:class:`PlanLearning`) the closure is ``step_terms(qs, action, t,
    qs_prev=None, working=None)``, ``working`` the working counts before the step's absorption:
    every reader of a table learned within the plan reads their posterior (prediction through
    its mean, novelty through its weights, an occupancy row likewise), the one closure built
    here handed those declarations per step (``StepContext.working``, and the tabular
    modalities' reads); a closed pair's table is read per component of the
    mixture's pre-step belief, its leaves the weight-average of the components' own
    (``components``, :class:`~hybrid_aif.objectives.registry.ComponentTable` by factor; design:
    parameters-within-the-plan.md D1, D2).
    """
    if learning is not None and learning.tables:
        return _learned_step_terms(
            A,
            B,
            A_deps,
            B_deps,
            families,
            obs_targets,
            state_targets,
            learning,
            dict(
                decomposition=decomposition,
                state_infogain_modalities=state_infogain_modalities,
                param_infogain_modalities=param_infogain_modalities,
                param_infogain_factors=param_infogain_factors,
                modality_names=modality_names,
                state_names=state_names,
                mounts=mounts,
                channel_selection=channel_selection,
                factor_dims=factor_dims,
                control_deps=control_deps,
                present=present,
                scoring=scoring,
                branching=branching,
                cross_entropy_targets=cross_entropy_targets,
                c_deps=c_deps,
                layout=layout,
                sensed=sensed,
                cells=cells,
                pooled=pooled,
            ),
        )
    options = dict(
        decomposition=decomposition,
        state_infogain_modalities=state_infogain_modalities,
        param_infogain_modalities=param_infogain_modalities,
        param_infogain_factors=param_infogain_factors,
        modality_names=modality_names,
        state_names=state_names,
        mounts=mounts,
        channel_selection=channel_selection,
        factor_dims=factor_dims,
        control_deps=control_deps,
        present=present,
        scoring=scoring,
        branching=branching,
        cross_entropy_targets=cross_entropy_targets,
        c_deps=c_deps,
        layout=layout,
        cells=cells,
        pooled=pooled,
    )
    if sensed is None:
        return _make_step_terms(
            A, B, A_deps, B_deps, families, obs_targets, state_targets, **options
        )
    selecting = selecting_channels(A, A_deps, families, sensed.offset)
    if not selecting:
        inner = _make_step_terms(
            A, B, A_deps, B_deps, families, obs_targets, state_targets, **options
        )

        def contracting_step_terms(qs, action, t, qs_prev=None, **step):
            extended = (
                *qs,
                *(Categorical.from_probs(p) for p in action_node_beliefs(action, sensed)),
            )
            return inner(extended, action, t, qs_prev, **step)

        return contracting_step_terms
    trimmed = tuple(deps[:1] if m in selecting else deps for m, deps in enumerate(A_deps))
    options["channel_selection"] = tuple(
        None if m in selecting else mask for m, mask in enumerate(channel_selection)
    )

    def step_terms(qs, action, t, qs_prev=None, **step):
        extended = (
            *qs,
            *(Categorical.from_probs(p) for p in action_node_beliefs(action, sensed)),
        )

        def levels_of(m: int) -> tuple[Array, ...]:
            return tuple(
                point_level(
                    node_entry(action, sensed.columns[g - sensed.offset], sensed.layout),
                    sensed.columns[g - sensed.offset],
                    sensed,
                    modality_names[m],
                    state_names[A_deps[m][0]],
                    families[A_deps[m][0]] == MIXTURE,
                )
                for g in selecting[m]
            )

        A_t = tuple(
            (
                NeuralHead(a, levels_of(m))
                if isinstance(a, MLPConditional)
                else select_observation(lg_point(a), levels_of(m))
            )
            if m in selecting
            else a
            for m, a in enumerate(A)
        )
        inner = _make_step_terms(
            A_t, B, trimmed, B_deps, families, obs_targets, state_targets, **options
        )
        return inner(extended, action, t, qs_prev, **step)

    return step_terms


def _learned_step_terms(
    A: tuple,
    B: tuple,
    A_deps: tuple[tuple[int, ...], ...],
    B_deps: tuple[tuple[int, ...], ...],
    families: tuple[str, ...],
    obs_targets: dict[str, Distribution],
    state_targets: dict[str, Distribution],
    learning: PlanLearning,
    options: dict,
):
    """:func:`make_step_terms` over the working counts of the tables learned within the plan:
    at each step the readers read the counts the step reads (before its absorption), handed to
    the one closure built here as per-step declarations (``tables``), and a closed pair's table
    is read per component of the pre-step mixture (:func:`pair_read`, ``components``). Design:
    parameters-within-the-plan.md D1, D2."""
    pairs = dict(learning.pairs)
    cells = options["cells"] or {}
    table = options["control_deps"]
    inner = make_step_terms(A, B, A_deps, B_deps, families, obs_targets, state_targets, **options)

    def step_terms(qs, action, t, qs_prev=None, working=None):
        if working is None:
            raise ValueError(
                "this step-term closure reads tables learned within the plan, so it needs the "
                "step's working counts (working=); drive it through the planning carry "
                "System.planner_context builds"
            )
        tables = {}
        components = {}
        prev = None if qs_prev is None else predictive_beliefs(qs_prev)
        for wt in learning.tables:
            for slot, i in wt.readers:
                if not wt.components:
                    held = A if slot == "likelihood" else B
                    tables[slot, i] = _with_counts(held[i], working[wt.key])
                    continue
                if prev is None:  # no leaf reads the parents: nothing to read per component
                    continue
                g, pn = pairs[i], B[i].parent_ndims
                index = control_read(action, i, table) if has_control_axes(table, i, pn) else ()
                pr = pair_read(
                    _log_mean(working[wt.key], pn),
                    pn,
                    i,
                    g,
                    B_deps,
                    cell_view(prev, cells),
                    prev[g],
                    cells[g],
                    index,
                )
                components[i] = registry.ComponentTable(
                    working[wt.key], pr.weights, pr.child, pr.parents
                )
        return inner(qs, action, t, qs_prev, components=components or None, tables=tables or None)

    return step_terms


def _make_step_terms(
    A: tuple,
    B: tuple,
    A_deps: tuple[tuple[int, ...], ...],
    B_deps: tuple[tuple[int, ...], ...],
    families: tuple[str, ...],
    obs_targets: dict[str, Distribution],
    state_targets: dict[str, Distribution],
    *,
    decomposition: str,
    state_infogain_modalities: tuple[int, ...],
    param_infogain_modalities: tuple[int, ...],
    param_infogain_factors: tuple[int, ...],
    modality_names: tuple[str, ...],
    state_names: tuple[str, ...],
    mounts: tuple[registry.Mount, ...] = (),
    channel_selection: tuple[tuple[bool, ...] | None, ...] = (),
    factor_dims: tuple[tuple[int, ...] | None, ...] = (),
    control_deps: tuple[tuple[int, ...], ...] | None = None,
    present: Presence | None = None,
    scoring: Mapping[str, object] | None = None,
    branching: bool = False,
    cross_entropy_targets: frozenset[str] = frozenset(),
    c_deps: Mapping[str, TargetParents] | None = None,
    layout: ControlLayout | None = None,
    cells: dict[int, Grid] | None = None,
    pooled: Pooled | None = None,
):
    """:func:`make_step_terms` over declarations every channel reads whole. Its closure takes
    two per-step keywords from :func:`_learned_step_terms`: ``components`` (factor index to the
    :class:`~hybrid_aif.objectives.registry.ComponentTable` of a closed pair's working table) and
    ``tables`` (``(slot, index)`` to the declaration holding a reader's working counts), so the
    closure is built once and the step reads the working posterior at run time."""
    mods, facs = modality_names, state_names
    # the channels no plan step observes: their leaves are zeros, with nothing computed for them
    absent = static_absent(present)
    c_deps = dict(c_deps or {})
    obs_c_deps = {k: v for k, v in c_deps.items() if k in obs_targets}
    state_c_deps = {k: v for k, v in c_deps.items() if k in state_targets}
    # LG slots are lowered to point declarations once here, so the per-step
    # leaves below read H/c/R_chol as fields rather than re-factorising a
    # distributional noise on every access.
    declared = A
    # the approximation status reads the declarations, before a neural emission is lowered
    approx = registry.approximate_status(B, declared, B_deps, A_deps, facs, mods)
    A = tuple(
        lg_point(neural_emission(a))
        if isinstance(a, MLPConditional) and _categorical_only(A_deps[m], families)
        else lg_point(a)
        if isinstance(a, LGObservation)
        else NeuralHead(a)
        if isinstance(a, MLPConditional)
        else a
        for m, a in enumerate(A)
    )
    # a neural modality is read at each step's belief as the linear-Gaussian model linearised
    # there (design: recognition-beside-analytic.md L4); one over categorical parents alone was
    # lowered above to the Gaussian emission it tabulates
    neural_mods = tuple(m for m, a in enumerate(A) if isinstance(a, NeuralHead))
    for m in neural_mods:
        if A_deps[m][0] >= len(families) or families[A_deps[m][0]] == CATEGORICAL:
            raise NotImplementedError(
                f"modality {mods[m]!r}: a neural modality whose first parent is categorical "
                "alongside a continuous one has no belief to linearise at; list the continuous "
                "factor first (design: recognition-beside-analytic.md L1)"
            )
    cat_mods = tuple(m for m, a in enumerate(A) if isinstance(a, TabularObservation))
    # a Gaussian emission over categorical parents alone has no state behind it: its leaves
    # read its parents' configurations (per-variable-infogain.md V11 decision 7)
    emission_mods = tuple(m for m, a in enumerate(A) if is_emission(a))
    lg_mods = tuple(
        m
        for m, a in enumerate(A)
        if (isinstance(a, LGObservation) and m not in emission_mods) or m in neural_mods
    )
    emission_sel = tuple(m for m in emission_mods if m in state_infogain_modalities)

    obs_idx = tuple(m for m in cat_mods if mods[m] in obs_targets)
    lg_obs_idx = tuple(m for m in lg_mods if mods[m] in obs_targets)
    ig_decomposition = decomposition == "utility_infogain"
    sig_all = state_infogain_modalities if ig_decomposition else ()
    # the selector mounts each decomposition's own epistemic leaf: ambiguity, under risk_ambiguity
    amb_idx = (
        () if ig_decomposition else tuple(m for m in cat_mods if m in state_infogain_modalities)
    )
    lg_amb_idx = (
        () if ig_decomposition else tuple(m for m in lg_mods if m in state_infogain_modalities)
    )
    # an undesignated channel's implicit flat preference: risk −H[q(o)], under risk_ambiguity
    flat_idx = () if ig_decomposition else tuple(m for m in cat_mods if mods[m] not in obs_targets)
    lg_flat_idx = (
        () if ig_decomposition else tuple(m for m in lg_mods if mods[m] not in obs_targets)
    )
    # a partial mask applies only where the selection mounted the channel's state leaf
    masks = tuple(
        channel_selection[m] if channel_selection and m in state_infogain_modalities else None
        for m in range(len(A))
    )
    partial = any(mask is not None for mask in masks)
    sub_blocks = tuple(factor_dims) if factor_dims else (None,) * len(B)
    any_sub_block = any(d is not None for d in sub_blocks)
    # the LG channels observing each Gaussian-carried factor alone, and the mounted ones among them
    observing = {
        f: tuple(m for m in lg_mods if A_deps[m] == (f,))
        for f, fam in enumerate(families)
        if fam == GAUSSIAN
    }
    # the joint risk's total correlation, under risk_ambiguity, whatever the selection
    correlated = (
        ()
        if ig_decomposition
        else tuple((f, chans) for f, chans in observing.items() if len(chans) > 1)
    )
    # the interaction information of the mounted channels' readings about the selected state
    # (utility_infogain), or the joint ambiguity's correction on a sub-block (risk_ambiguity)
    interacting = tuple(
        (f, mounted)
        for f, chans in observing.items()
        if len(mounted := tuple(m for m in chans if m in state_infogain_modalities)) > 1
        and (ig_decomposition or sub_blocks[f] is not None)
    )

    # the factors whose channels' leaves share the factorisations of the stacked predictive
    # covariance per step (continuous.linear_gaussian_readings, and sub_block_readings on a factor
    # selected in part): the mounted channels' gains and their interaction under
    # utility_infogain, whenever no mounted channel has a partial selection; the total
    # correlation of the channels under risk_ambiguity
    reading_groups = (
        tuple(
            (f, mounted)
            for f, chans in observing.items()
            if _shares_factorisation(chans, masks)
            and (mounted := tuple(m for m in chans if m in sig_all))
        )
        if ig_decomposition
        else tuple(g for g in correlated if _fuses_correlation(g[1]))
    )
    # the groups on a factor selected in part with no partial channel selection also read the
    # conditional covariance: the channels' ambiguity and the correlation given the selection
    sub_groups = {
        f: chans
        for f, chans in reading_groups
        if sub_blocks[f] is not None and _shares_factorisation(chans, masks)
    }
    # a channel observing a Gaussian factor alone takes the factor's sub-block in its ambiguity,
    # read off the group's conditional covariance where there is one
    fused_amb_idx = tuple(
        m for m in lg_amb_idx if len(A_deps[m]) == 1 and A_deps[m][0] in sub_groups
    )
    lg_rest_idx = tuple(m for m in lg_amb_idx if m not in fused_amb_idx)
    lg_amb_dims = tuple(
        sub_blocks[A_deps[m][0]]
        if len(A_deps[m]) == 1 and families[A_deps[m][0]] == GAUSSIAN
        else None
        for m in lg_rest_idx
    )

    mniw_mods = tuple(
        m
        for m in param_infogain_modalities
        if isinstance(declared[m], LinearGaussianObservationMNIW)
    )
    statics = registry.Statics(
        index=(),
        modality_names=mods,
        state_names=facs,
        A=A,
        B=B,
        A_deps=A_deps,
        B_deps=B_deps,
        B_control_deps=control_deps,
        selected=masks if partial else (),
        dims=sub_blocks if any_sub_block else (),
        scoring=scoring,
    )
    mounted = (
        # channel-keyed state leaves, one estimator per observation family
        (registry.STATE, registry.CATEGORICAL, tuple(m for m in sig_all if m in cat_mods)),
        (registry.STATE, registry.LINEAR_GAUSSIAN, tuple(m for m in sig_all if m in lg_mods)),
        (registry.STATE, registry.EMISSION, tuple(m for m in sig_all if m in emission_mods)),
        # the slot-qualified parameter variables
        (
            registry.param_kind("likelihood"),
            registry.DIRICHLET,
            tuple(m for m in param_infogain_modalities if m not in mniw_mods),
        ),
        (registry.param_kind("transition"), registry.DIRICHLET, param_infogain_factors),
    )
    infogain_steps = tuple(
        registry.prepare(kind, family, statics._replace(index=idx))
        for kind, family, idx in mounted
        if idx
    )
    if mniw_mods:  # it reads the declaration, not its point lowering
        infogain_steps += (
            registry.prepare(
                registry.param_kind("likelihood"),
                registry.MNIW,
                statics._replace(index=mniw_mods, A=declared),
            ),
        )
    row_steps = registry.prepare_step_rows(mounts, statics)

    def sub(collection, idx):
        return tuple(collection[i] for i in idx)

    def lg_mode_beliefs(qs: tuple[Belief, ...], idx: tuple[int, ...]):
        """Each LG modality's discrete (mode) parent beliefs; empty when it does not switch."""
        return tuple(tuple(qs[g] for g in A_deps[m][1:]) for m in idx)

    def step_terms(
        qs: tuple[Belief, ...],
        action: Array,
        t: Array,
        qs_prev: tuple[Belief, ...] | None = None,
        *,
        components: dict | None = None,
        tables: dict | None = None,
    ) -> TermTree:
        paired = any(isinstance(q, BeliefPair) for q in qs)
        qs_in = qs
        qs_epistemic = epistemic_beliefs(qs) if paired else None
        qs_conditioned = conditioned_beliefs(qs) if paired else None
        qs_all = predictive_beliefs(qs)  # the state factors, then the action nodes read
        qs = qs_all[: len(facs)]
        qs_prev_cells = None
        if qs_prev is not None:
            qs_prev = predictive_beliefs(qs_prev)
            qs_prev_cells = cell_view(qs_prev, cells) if cells else None
        obs_t = (
            resolve_step_targets(obs_targets, obs_c_deps, action, qs, layout)
            if obs_c_deps
            else obs_targets
        )
        state_t = (
            resolve_step_targets(state_targets, state_c_deps, action, qs, layout)
            if state_c_deps
            else state_targets
        )
        present_t = None if present is None else presence_at(present, action, t, layout)

        # a neural modality as the linear-Gaussian model at this step's belief over its factor:
        # the epistemic chain's (the belief before the step's observations), else the predictive
        def read_at(m: int, epistemic: bool = True):
            f = A_deps[m][0]
            use = epistemic and qs_epistemic is not None and families[f] == GAUSSIAN
            return neural_model(A[m], qs_epistemic[f] if use else qs[f])

        # a tabular modality learned within the plan reads its working posterior (D1)
        A_t = tuple(
            read_at(m) if m in neural_mods else tables.get(("likelihood", m), a) if tables else a
            for m, a in enumerate(A)
        )
        # the risk and utility leaves read the predictive belief; the two reads differ only
        # under the quadrature rule, whose model depends on the belief's spread
        A_p = tuple(
            read_at(m, False)
            if m in neural_mods
            and qs_epistemic is not None
            and getattr(A[m].decl, "propagation", "first_order") == "unscented"
            else a
            for m, a in enumerate(A_t)
        )

        def present_at(chans):
            return None if present_t is None else present_t[np.asarray(chans)]

        readings: dict[int, Interval] = {}
        total_correlations: dict[int, Array] = {}
        conditional_correlations: dict[int, Array] = {}
        ambiguities: dict[int, Array] = {}
        for f, chans in reading_groups:
            from_pair = ig_decomposition and qs_epistemic is not None
            held = qs_epistemic if from_pair else qs
            if not isinstance(held[f], Gaussian):
                continue
            live = tuple(m for m in chans if m not in absent)

            def zero_leaf(q=held[f]):
                return jnp.zeros_like(q.loc[..., 0])

            # the chain's reads of the noise at the epistemic mean, which these leaves read at
            chols = (
                tuple(qs_in[f].noises.get(m) for m in live)
                if from_pair and live and qs_in[f].noises
                else None
            )
            gains: tuple[Interval, ...] = ()
            if f in sub_groups:
                if live:
                    fused = sub_block_readings(
                        held[f], sub(A_t, live), sub_blocks[f], present_at(live), chols
                    )
                    interaction = fused.interaction
                    total, conditional = fused.total_correlation, fused.conditional_correlation
                    gains, ambiguity_of = fused.gains, fused.ambiguity
                else:
                    interaction = total = conditional = zero_leaf()
                    ambiguity_of = ()
                if ig_decomposition:
                    total_correlations[f] = interaction
                else:
                    total_correlations[f] = total
                    conditional_correlations[f] = conditional
                    ambiguities.update(zip(live, ambiguity_of, strict=True))
                    ambiguities.update({m: zero_leaf() for m in chans if m in absent})
            elif live:
                gains, total_correlations[f] = linear_gaussian_readings(
                    held[f], sub(A_t, live), present_at(live), chols
                )
            else:
                total_correlations[f] = zero_leaf()
            if ig_decomposition:
                readings.update(zip(live, gains, strict=True))
                readings.update({m: Interval.exact(zero_leaf()) for m in chans if m in absent})
        if ig_decomposition:
            # a neural channel's information gain is evaluated here, at its step's model, and
            # handed to the mounted estimator as a reading (the estimator holds the declaration)
            for m in neural_mods:
                if m not in sig_all or m in readings:
                    continue
                f = A_deps[m][0]
                if m in absent:
                    readings[m] = Interval.exact(jnp.zeros((), qs[f].loc.dtype))
                    continue
                switching = len(A_deps[m]) > 1
                belief = qs[f] if qs_epistemic is None or switching else qs_epistemic[f]
                leaf = state_info_gain(
                    (belief,),
                    (A_t[m],),
                    (mods[m],),
                    mode_beliefs=(lg_mode_beliefs(qs_all, (m,))[0],),
                    selected=(masks[m],),
                    dims=(sub_blocks[f] if not switching else None,),
                )[INFOGAIN]
                readings[m] = leaf[mods[m]]
        trees: list[TermTree] = []
        if obs_idx:
            q_obs = expected_obs(qs_all, sub(A_t, obs_idx), sub(A_deps, obs_idx))
            names = sub(mods, obs_idx)
            trees.append(
                utility(q_obs, obs_t, names) if ig_decomposition else risk(q_obs, obs_t, names)
            )
        if lg_obs_idx:
            # a diagonal preference over a diagonal-noise predictive is scored from the low-rank
            # predictive, without a d_obs by d_obs array; every other pair takes the dense path
            fast: dict[str, Interval] = {}
            slow: list[int] = []
            for m, parents in zip(lg_obs_idx, lg_mode_beliefs(qs_all, lg_obs_idx), strict=True):
                if (
                    ig_decomposition
                    and diagonal_target_parts(obs_t[mods[m]]) is not None
                    and lowrank_capable(A_p[m], parents)
                ):
                    fast[mods[m]] = lowrank_utility(
                        qs[A_deps[m][0]], A_p[m], parents, obs_t[mods[m]]
                    )
                else:
                    slow.append(m)
            tree = {"utility" if ig_decomposition else "risk": fast}
            if slow:
                dense_t = {mods[m]: dense_target(obs_t[mods[m]]) for m in slow}
                q_lg = tuple(
                    predictive_obs(qs[A_deps[m][0]], A_p[m], parents)
                    for m, parents in zip(slow, lg_mode_beliefs(qs_all, tuple(slow)), strict=True)
                )
                names = sub(mods, tuple(slow))
                score = lg_utility if ig_decomposition else risk
                tree = merge(tree, score(q_lg, dense_t, names))
            trees.append(tree)
        if flat_idx:
            q_flat = expected_obs(qs_all, sub(A_t, flat_idx), sub(A_deps, flat_idx))
            trees.append(flat_risk(q_flat, sub(mods, flat_idx)))
        if lg_flat_idx:
            trees.append(
                {
                    "risk": {
                        mods[m]: -predictive_entropy(qs[A_deps[m][0]], A_p[m], parents)
                        for m, parents in zip(
                            lg_flat_idx, lg_mode_beliefs(qs_all, lg_flat_idx), strict=True
                        )
                    }
                }
            )
        if amb_idx:
            selected = sub(masks, amb_idx) if partial else None
            trees.append(
                ambiguity(
                    qs_all, sub(A_t, amb_idx), sub(A_deps, amb_idx), sub(mods, amb_idx), selected
                )
            )
        if lg_rest_idx:
            trees.append(
                lg_ambiguity(
                    sub(A_t, lg_rest_idx),
                    lg_mode_beliefs(qs_all, lg_rest_idx),
                    sub(mods, lg_rest_idx),
                    q_blocks=tuple(qs[A_deps[m][0]] for m in lg_rest_idx),
                    selected=sub(masks, lg_rest_idx) if partial else None,
                    dims=lg_amb_dims if any_sub_block else None,
                )
            )
        if fused_amb_idx:
            trees.append(
                {"ambiguity": {mods[m]: Interval.exact(ambiguities[m]) for m in fused_amb_idx}}
            )
        if emission_mods and not ig_decomposition:
            # undesignated (a preference over one is refused): its flat-preference risk, and
            # its ambiguity where the selector mounts it
            risk_leaves, amb_leaves = {}, {}
            for m in emission_mods:
                h, h_cond = emission_entropies(
                    A_t[m], tuple(qs_all[g] for g in A_deps[m]), masks[m]
                )
                risk_leaves[mods[m]] = -h
                if m in emission_sel:
                    amb_leaves[mods[m]] = h_cond
            trees.append({"risk": risk_leaves})
            if amb_leaves:
                trees.append({"ambiguity": amb_leaves})
        if infogain_steps or row_steps:
            ctx = registry.StepContext(
                qs=qs_all,
                action=action,
                qs_prev=qs_prev,
                qs_prev_cells=qs_prev_cells,
                components=components,
                working=tables,
                t=t,
                qs_epistemic=qs_epistemic,
                qs_conditioned=qs_conditioned,
                readings=readings or None,
            )
            if infogain_steps:
                leaves: dict[str, Interval] = {}
                for step in infogain_steps:
                    leaves.update(step(ctx))
                trees.append({INFOGAIN: leaves})
            trees.extend({term: step(ctx)} for term, step in row_steps)

        def group_approx(chans) -> bool:
            # a leaf over a group of channels is approximate iff any channel it reads is
            # (design: recognition-beside-analytic.md L7)
            return any(approx.modalities.get(mods[m], False) for m in chans)

        if correlated:
            trees.append(
                {
                    CORRELATION: {
                        facs[f]: Interval.exact(
                            total_correlations[f]
                            if f in total_correlations
                            else observation_correlation(qs[f], sub(A_t, chans), present_at(chans)),
                            approximate=group_approx(chans),
                        )
                        for f, chans in correlated
                    }
                }
            )
        if interacting:
            if ig_decomposition:
                # it reads the belief the information-gain leaves read
                # (planning-with-observations.md D1)
                held = qs if qs_epistemic is None else qs_epistemic
                leaves = {
                    facs[f]: total_correlations[f]
                    if f in total_correlations
                    else observation_interaction(
                        held[f], sub(A_t, chans), sub_blocks[f], present_at(chans)
                    )
                    for f, chans in interacting
                }
            else:
                leaves = {
                    facs[f]: -conditional_correlations[f]
                    if f in conditional_correlations and sub_groups[f] == chans
                    else -conditional_correlation(
                        qs[f], sub(A_t, chans), sub_blocks[f], present_at(chans)
                    )
                    for f, chans in interacting
                }
            flags = {facs[f]: group_approx(chans) for f, chans in interacting}
            trees.append(
                {
                    INTERACTION: {
                        k: Interval.exact(v, approximate=flags[k]) for k, v in leaves.items()
                    }
                }
            )
        if state_t and (branching or not cross_entropy_targets):
            score = cross_entropy_risk if branching else risk
            trees.append(score(qs, state_t, facs))
        elif state_t:
            kl_targets = {k: v for k, v in state_t.items() if k not in cross_entropy_targets}
            ce_targets = {k: v for k, v in state_t.items() if k in cross_entropy_targets}
            if kl_targets:
                trees.append(risk(qs, kl_targets, facs))
            trees.append(cross_entropy_risk(qs, ce_targets, facs))
        tree = registry.stamp_tree(merge(*trees), approx)
        tree = scale_pooled(tree, pooled, action, layout)
        if present_t is not None:
            tree = mask_absent(tree, present_t, mods)
        return tree

    return step_terms


def make_terminal_terms(
    terminal_targets: dict[str, Distribution],
    state_names: tuple[str, ...],
    h_deps: tuple[tuple[int, ...], ...] | None = None,
    branching: bool = False,
    cross_entropy_targets: frozenset[str] = frozenset(),
    approximation: registry.Approximation | None = None,
):
    """Family-generic terminal-risk evaluator, or None without H designations.

    Conditional terminal targets (``h_deps``) resolve against the final beliefs. ``branching``
    marks a closure built for a branching planner, where each terminal target is scored by its
    cross-entropy (:func:`~hybrid_aif.objectives.discrete.terminal_cross_entropy_risk`; design:
    per-variable-infogain.md V11 decision 8). ``cross_entropy_targets`` names the factors whose
    terminal targets an open-loop closure scores by their cross-entropy, the targets a
    closed-loop pass corrects (:mod:`hybrid_aif.objectives.closed_loop`). ``approximation`` is
    the System's static :class:`~hybrid_aif.objectives.registry.Approximation`, stamped on the
    leaves.
    """
    if not terminal_targets:
        return None

    def stamped(tree: TermTree) -> TermTree:
        return tree if approximation is None else registry.stamp_tree(tree, approximation)

    ce_targets = {k: v for k, v in terminal_targets.items() if k in cross_entropy_targets}
    terminal_targets = {k: v for k, v in terminal_targets.items() if k not in cross_entropy_targets}
    deps = h_deps if h_deps is not None else ((),) * len(state_names)
    conditional = any(deps[f] for f, name in enumerate(state_names) if name in terminal_targets)

    def terminal_terms(qs_final: tuple[Belief, ...]) -> TermTree:
        qs_final = predictive_beliefs(qs_final)
        if branching:
            every = {**terminal_targets, **ce_targets}
            return stamped(terminal_cross_entropy_risk(qs_final, every, deps, state_names))
        targets = (
            resolve_conditional_targets(terminal_targets, deps, qs_final, state_names)
            if conditional
            else terminal_targets
        )
        tree = terminal_risk(qs_final, targets, state_names)
        if ce_targets:
            no_deps = ((),) * len(state_names)
            covered = terminal_cross_entropy_risk(qs_final, ce_targets, no_deps, state_names)
            tree = {"terminal_risk": {**tree["terminal_risk"], **covered["terminal_risk"]}}
        return stamped(tree)

    return terminal_terms
