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
or by the unscented transform, at the step's own belief.

Design: docs/design/hybrid-system.md D1–D4; docs/design/nonlinear-transitions.md N2–N4
"""

from collections.abc import Callable, Mapping
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

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
)
from hybrid_aif.inference.emission import is_emission
from hybrid_aif.inference.linear_gaussian import condition_on_expected, predict
from hybrid_aif.inference.mixture_rollout import propagate
from hybrid_aif.interval import Interval
from hybrid_aif.objectives import registry
from hybrid_aif.objectives.base import INFOGAIN, TermTree, merge
from hybrid_aif.objectives.continuous import (
    conditional_correlation,
    emission_entropies,
    lg_ambiguity,
    lg_utility,
    linear_gaussian_readings,
    observation_correlation,
    observation_interaction,
    predictive_entropy,
    predictive_obs,
    risk,
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


def refuse_switching_nonlinear(B_f, parents: tuple[int, ...], factor_names, f: int) -> None:
    """Raise for a :class:`NonlinearGaussianTransition` with discrete state parents."""
    if isinstance(B_f, NonlinearGaussianTransition) and parents:
        raise NotImplementedError(
            f"factor {factor_names[f]!r}: a NonlinearGaussianTransition with discrete "
            f"state parent(s) {[factor_names[g] for g in parents]} is switching "
            "dynamics, which would need f to take the mode, or one f per mode; that is "
            "not built (design: nonlinear-transitions.md N4). Drop the state parents, "
            "read the mode through a discrete control node, or declare a "
            "LinearGaussianTransition with per-mode F"
        )


def factor_families(
    B: tuple,
    D: tuple,
    B_deps: tuple[tuple[int, ...], ...],
    factor_names: tuple[str, ...],
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
                    f"factor {factor_names[f]!r}: a neural (MLPConditional) transition with "
                    f"discrete parent(s) {[factor_names[g] for g in B_deps[f]]} would grow the "
                    "belief through one mode-conditioned net per mode; P1's prediction row is "
                    "the parentless linearized one (design: amortization.md A1(a), build plan "
                    "P1). Drop the parents, or declare a LinearGaussianTransition."
                )
            if not isinstance(D_f, Gaussian):
                raise ValueError(
                    f"factor {factor_names[f]!r}: MLPConditional transition requires a "
                    f"Gaussian prior (D), got {type(D_f).__name__} (the linearized prediction "
                    "row carries a single Gaussian; mixture-carried neural factors are not "
                    "built — design: amortization.md A1(a))"
                )
            fams.append(GAUSSIAN)
        elif isinstance(B_f, TabularTransition):
            if not isinstance(D_f, TabularPrior):
                raise ValueError(
                    f"factor {factor_names[f]!r}: tabular transition (B) requires a "
                    f"tabular prior (D) — Categorical/Dirichlet, or a conditional form "
                    f"with a System D_deps entry for hierarchical priors — got "
                    f"{type(D_f).__name__}"
                )
            fams.append(CATEGORICAL)
        elif isinstance(B_f, NonlinearGaussianTransition):
            refuse_switching_nonlinear(B_f, B_deps[f], factor_names, f)
            if isinstance(D_f, MixtureGaussian):
                fams.append(MIXTURE)
            elif isinstance(D_f, Gaussian):
                fams.append(GAUSSIAN)
            else:
                raise ValueError(
                    f"factor {factor_names[f]!r}: NonlinearGaussianTransition requires a "
                    f"Gaussian or MixtureGaussian prior (D), got {type(D_f).__name__}"
                )
        elif isinstance(B_f, LinearGaussianTransition):
            if isinstance(D_f, MixtureGaussian):
                fams.append(MIXTURE)
            elif isinstance(D_f, Gaussian):
                fams.append(MIXTURE if B_deps[f] else GAUSSIAN)
            else:
                raise ValueError(
                    f"factor {factor_names[f]!r}: LinearGaussianTransition requires a "
                    f"Gaussian or MixtureGaussian prior (D), got {type(D_f).__name__}"
                )
        else:
            raise ValueError(
                f"factor {factor_names[f]!r}: unknown transition declaration type "
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
) -> None:
    """Validate a (possibly switching) LG observation's H/c/R_chol shapes.

    Each array is shared (event shape only) or per-mode (leading ``parent_shape``
    axes). ``d_obs`` comes from ``H``; ``d`` must match the observed factor. Named
    categorical parents require at least one per-mode array.
    """
    d_obs = A_m.H.shape[-2]
    if A_m.noise_map:
        if parent_shape:
            raise NotImplementedError(
                f"modality {modality_name!r}: a noise map on a switching LG observation is not "
                "built; a map is read at the observed factor's mean, and a mode-conditioned "
                "channel has one mean per mode rather than one (design: "
                "planning-with-observations.md D3)"
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
    D_f: Gaussian | MixtureGaussian,
    f: int,
    name: str,
    table: ControlTable,
    kinds: tuple[str, ...],
    sizes: tuple[int, ...],
) -> None:
    """Check a nonlinear transition against its factor: the innovations' parent axes are the
    factor's discrete control nodes, its dimension the prior's, and ``f`` (by
    ``jax.eval_shape``) maps the ``(d,)`` state and the factor's control entries (an int32
    level per discrete node, a float vector per continuous one) to a ``(d,)`` float mean."""
    parents = table.control_parents[f]
    discrete = [k for k, _ in parents if not kinds or kinds[k] == LEVEL]
    expected = tuple(sizes[k] for k in discrete) or (1,)
    got = decl.innovations.parent_shape
    if got != expected:
        raise ValueError(
            f"factor {name!r}: innovations parent_shape {got} != {expected} (expected one axis "
            "per discrete control node, in declared order, or (1,) for a factor with none)"
        )
    d = decl.dim
    if D_f.dim != d:
        raise ValueError(
            f"factor {name!r}: the NonlinearGaussianTransition's innovations are {d}-dimensional "
            f"and the prior (D) is {D_f.dim}-dimensional; they must agree"
        )
    layout = tuple(
        jax.ShapeDtypeStruct((), jnp.int32)
        if not kinds or kinds[k] == LEVEL
        else jax.ShapeDtypeStruct((sizes[k],), jnp.float32)
        for k, _ in parents
    )
    entries = [
        f"int32 level of {table.control_names[k]!r}"
        if not kinds or kinds[k] == LEVEL
        else f"float[{sizes[k]}] {table.control_names[k]!r}"
        for k, _ in parents
    ]
    described = "(" + ", ".join(entries) + ")"
    x = jax.ShapeDtypeStruct((d,), decl.innovations.loc.dtype)
    try:
        out = jax.eval_shape(lambda x, controls: decl.f(x, controls), x, layout)
    except Exception as exc:
        raise ValueError(
            f"factor {name!r}: the NonlinearGaussianTransition's f fails on the factor's "
            f"layout, f(x: float[{d}], controls: {described}): {exc}"
        ) from exc
    shape = getattr(out, "shape", None)
    dtype = getattr(out, "dtype", None)
    if shape != (d,) or dtype is None or not jnp.issubdtype(dtype, jnp.floating):
        what = f"{dtype}{list(shape)}" if shape is not None else type(out).__name__
        raise ValueError(
            f"factor {name!r}: the NonlinearGaussianTransition's f returns {what} for "
            f"f(x: float[{d}], controls: {described}); it must return the next state's mean, "
            f"float[{d}]"
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
    factor_names: tuple[str, ...],
    mixture_schedule: str = "runnalls",
    sensed: "SensedNodes | None" = None,
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
    facs, mods = factor_names, modality_names
    families = factor_families(B, D, B_deps, facs)
    kinds, sizes = space.kinds, space.sizes

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
        if isinstance(B[f], NonlinearGaussianTransition):
            _validate_nonlinear(B[f], D[f], f, facs[f], table, kinds, sizes)
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
                and family_of(obs_f) == GAUSSIAN
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
            _validate_lg_obs_shapes(A_m, D[obs_f].dim, parent_shape, mods[m], name_of(obs_f))
            if A_m.noise_map and family_of(obs_f) == MIXTURE:
                raise NotImplementedError(
                    f"modality {mods[m]!r}: a noise map over the mixture-carried factor "
                    f"{facs[obs_f]!r} is not built; a map is read at a Gaussian-carried factor's "
                    "mean (design: planning-with-observations.md D3)"
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
                "as its belief's probability vector — the one-hot convention, "
                "amortization.md A1)",
            )
        else:
            raise ValueError(
                f"modality {mods[m]!r}: unknown observation declaration type {type(A_m).__name__}"
            )

    for m, A_m in enumerate(A):
        if not (isinstance(A_m, LGObservation) and mods[m] in obs_targets):
            continue
        target, d_obs = obs_targets[mods[m]], A_m.H.shape[-2]
        if not isinstance(target, Gaussian | MixtureGaussian | ConditionalGaussian):
            raise ValueError(
                f"obs_targets[{mods[m]!r}]: an LG modality's C designation is a preference over "
                f"its *continuous* observation, so the target is a Gaussian or MixtureGaussian "
                f"over R^{d_obs} (covariance-as-tolerance, the same reading state_targets has), "
                f"or a ConditionalGaussian over it per level of its C_deps parents, got "
                f"{type(target).__name__}"
            )
        if target.dim != d_obs:
            obs_f = A_deps[m][0]
            raise ValueError(
                f"obs_targets[{mods[m]!r}]: target dim {target.dim} != the modality's "
                f"observation dim {d_obs} (read off H); a C designation lives on the observation "
                f"space, not on the observed factor's state space (factor {facs[obs_f]!r} is "
                f"{D[obs_f].dim}-dimensional — designate it via state_targets if that is what "
                "you meant)"
            )
    mlp_targeted = sorted(
        mods[m]
        for m, A_m in enumerate(A)
        if isinstance(A_m, MLPConditional) and mods[m] in obs_targets
    )
    if mlp_targeted:
        raise ValueError(
            f"obs_targets on neural modality(ies) {mlp_targeted}: a C designation over a "
            "continuous predicted observation is scored against the modality's predicted "
            "observation density, and a net has no closed-form pushforward of the belief — the "
            "LG modalities have one (H, c, R) and are supported; this needs the amortized "
            "predictive that is not built (design: amortization.md A1). Designate the underlying "
            "factor via state_targets, or declare a LinearGaussianObservation. Raising rather "
            'than ignoring: a silently dropped designation is exactly the "toggle lies" failure.'
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
    column, the Cholesky factor of each conditioning channel's noise at the epistemic mean, which
    the chain read to condition and the information-gain leaves read again (empty: the leaves
    read the noise at the mean themselves)."""

    predictive: Gaussian
    epistemic: Gaussian
    conditioned: Gaussian
    step: Array
    noises: dict[int, Array] = {}  # noqa: RUF012


class Conditioner(NamedTuple):
    """One linear-Gaussian modality the epistemic chain conditions on: the factor it observes,
    its column in the availability mask, its head, and the noise at a mean. A channel whose mode
    parents are action nodes (``nodes``, their engine positions) holds per-level arrays, and
    ``modality`` and ``factor_name`` name it in the sentence a non-point step is refused with;
    the step selects its level (:func:`select_conditioner`)."""

    factor: int
    column: int
    H: Array
    c: Array
    R_chol_at: Callable[[Array], Array]
    nodes: tuple[int, ...] = ()
    modality: str = ""
    factor_name: str = ""


def soft_selection_error(modality: str, factor: str, node: str, why: str) -> ValueError:
    """The refusal for a selecting channel whose action node is not a point."""
    return ValueError(
        f"modality {modality!r} selects its head and noise by action node {node!r}, and the "
        f"Gaussian-carried factor {factor!r} it observes stays one Gaussian only while that "
        f"node's belief is a point; here it is {why}. Carry {factor!r} as a mixture (a "
        "MixtureGaussian initial prior and System.mixture_k) so the channel mixes over the "
        "node's levels, or clamp the node to one level"
    )


def point_level(
    entry: object, column: int, sensed: SensedNodes, modality: str, factor: str
) -> Array:
    """The level a selecting channel reads action node ``column`` at, from the node's entry at
    one plan step: the clamped level under the delta kernel. A probability vector or a node with
    a declared actuator kernel is not a point and raises."""
    name = sensed.names[sensed.columns.index(column)]
    if is_belief(entry):
        raise soft_selection_error(
            modality, factor, name, "a probability vector over its levels at this step"
        )
    if column in sensed.kernels:
        raise soft_selection_error(
            modality,
            factor,
            name,
            "spread over its levels by the kernel declared in params.actuator",
        )
    return jnp.asarray(entry, jnp.int32)


def prior_point_level(q: Categorical, node: str, modality: str, factor: str) -> Array:
    """The level of an action node's prior, which must be a point (one level holding all the
    mass): the efference copy under the delta kernel. A prior that is not raises, at once for a
    concrete one and when the computation runs for a traced one."""
    probs = q.probs
    peak = jnp.max(probs, axis=-1)
    level = jnp.argmax(probs, axis=-1).astype(jnp.int32)
    if isinstance(peak, jax.core.Tracer):
        why = "spread over its levels in the prior the call was given"
        error = soft_selection_error(modality, factor, node, why)
        return eqx.error_if(level, peak < 1.0 - 1e-6, str(error))
    if float(peak) < 1.0 - 1e-6:
        why = f"the prior {np.asarray(probs).tolist()}"
        raise soft_selection_error(modality, factor, node, why)
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
    return LinearGaussianObservation(
        H=_take_modes(model.H, 2, levels),
        c=_take_modes(model.c, 1, levels),
        noise=_take_modes(model.R_chol, 2, levels),
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
    R_chol = _take_modes(cond.R_chol_at(None), 2, levels)
    return cond._replace(
        H=_take_modes(cond.H, 2, levels),
        c=_take_modes(cond.c, 1, levels),
        R_chol_at=lambda loc, R_chol=R_chol: R_chol,
        nodes=(),
    )


def selecting_channels(A: tuple, A_deps: tuple[tuple[int, ...], ...], families, offset: int):
    """The linear-Gaussian channels that select by action node, each at its modality index with
    its mode parents' engine positions: those observing a Gaussian-carried factor whose mode
    parents are all action nodes (positions from ``offset``)."""
    return {
        m: deps[1:]
        for m, (a, deps) in enumerate(zip(A, A_deps, strict=True))
        if isinstance(a, LGObservation)
        and not is_emission(a)
        and len(deps) > 1
        and deps[0] < offset
        and families[deps[0]] == GAUSSIAN
        and all(g >= offset for g in deps[1:])
    }


def _seed_noises(q: Gaussian, factor: int, conditioners: tuple["Conditioner", ...]) -> dict:
    """The ``noises`` a pair over ``factor`` carries from its seed: a conditioner's noise factor
    at the seed's mean, and zeros of its shape for one that selects by action node (its level is
    read at the step)."""
    out = {}
    for cond in conditioners:
        if cond.factor != factor:
            continue
        if cond.nodes:
            R = cond.R_chol_at(None)
            out[cond.column] = jnp.zeros(R.shape[-2:], R.dtype)
        else:
            out[cond.column] = cond.R_chol_at(q.loc)
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


def epistemic_beliefs(qs: tuple[Belief, ...]) -> tuple[Belief, ...]:
    """The carry with every pair read as its epistemic member, the belief before the step's
    observations."""
    return tuple(q.epistemic if isinstance(q, BeliefPair) else q for q in qs)


def conditioned_beliefs(qs: tuple[Belief, ...]) -> tuple[Belief, ...]:
    """The carry with every pair read as its conditioned member, the belief after the step's
    expected observations."""
    return tuple(q.conditioned if isinstance(q, BeliefPair) else q for q in qs)


def condition_expected(
    q: Gaussian, cond: Conditioner, flag: Array | None, R_chol: Array | None = None
) -> Gaussian:
    """``q`` conditioned on the observation ``cond`` is expected to give at its mean: zero
    innovation, so the mean is ``q.loc`` itself and the covariance takes the Kalman update;
    ``flag`` (``None``: present) leaves ``q`` where the modality is absent. ``R_chol`` is the
    noise factor at ``q``'s mean when the caller has read it (``None``: read here)."""
    if R_chol is None:
        R_chol = cond.R_chol_at(q.loc)
    posterior = condition_on_expected(q, cond.H, R_chol)
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


def _lg_inputs(B_f: LinearGaussianTransition, read, entries: dict[int, object]) -> _LGInputs:
    """Index ``B_f``'s discrete control axes at their levels and gather its ``G u`` input.

    ``read`` is the factor's :class:`~hybrid_aif.controls.LGRead`; ``entries`` maps each
    control node it reads to that node's entry at this step. A continuous node's entry is its
    value (a clamp) or a Gaussian / truncated Gaussian over it (a belief, read through its
    first two moments).
    """
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
                    root = v.chol if isinstance(v, Gaussian) else jnp.linalg.cholesky(v.cov)
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
    Design: docs/design/amortization.md A1
    """
    mu = q.loc
    tail = tuple(a.astype(mu.dtype) for a in action_parents)
    head = B_f.condition_on((mu, *tail))
    jac = jax.jacfwd(lambda x: B_f.condition_on((x, *tail)).loc)(mu)
    return Gaussian.from_cov(head.loc, jac @ q.cov @ jnp.swapaxes(jac, -1, -2) + head.cov)


#: The unscented transform's scaling (Julier & Uhlmann 2004; the scaled form of Julier 2002):
#: ``α = 1`` and ``κ = 0`` put the sigma points at ``μ ± √n`` columns of the covariance root
#: with weight ``1/(2n)`` each and none on the centre's mean; ``β = 2`` (optimal for a Gaussian
#: belief) adds ``1 − α² + β = 2`` to the centre's covariance weight, so every covariance weight
#: is positive and the propagated covariance is positive semidefinite.
UNSCENTED_KAPPA = 0.0
UNSCENTED_BETA = 2.0


class _NLInputs(NamedTuple):
    """A nonlinear transition's inputs at one step: the innovation ``(loc, chol)`` at the
    discrete control levels; ``controls``, one entry per control parent in declared order (an
    int32 level, a float vector, or ``None`` where a belief sits); and ``beliefs``, each
    belief-valued continuous entry as ``(position, mean, root)``, ``root`` a factor of its
    covariance."""

    loc: Array
    chol: Array
    controls: tuple
    beliefs: tuple[tuple[int, Array, Array], ...]


def _nl_inputs(
    B_f: NonlinearGaussianTransition,
    parents: tuple[tuple[int, int], ...],
    kinds: tuple[str, ...],
    entries: dict[int, object],
) -> _NLInputs:
    """Read ``B_f``'s control parents (``(node, position)`` pairs in declared order) at this
    step's ``entries``: a discrete node indexes the innovations and reaches ``f`` as its
    level; a continuous node's clamp reaches ``f`` as its value, and a Gaussian or truncated
    Gaussian belief over it through its first two moments."""
    levels, controls, beliefs = [], [], []
    for pos, (k, _) in enumerate(parents):
        entry = entries[k]
        if kinds[k] == LEVEL:
            level = jnp.asarray(entry, jnp.int32)
            levels.append(level)
            controls.append(level)
        elif is_distribution(entry):
            root = entry.chol if isinstance(entry, Gaussian) else jnp.linalg.cholesky(entry.cov)
            mean = jnp.asarray(entry.mean(), jnp.float32)
            beliefs.append((pos, mean, jnp.asarray(root, jnp.float32)))
            controls.append(None)
        else:
            controls.append(jnp.asarray(entry, jnp.float32))
    noise = B_f.innovations.condition_on(tuple(levels) or (0,))
    return _NLInputs(noise.loc, noise.chol, tuple(controls), tuple(beliefs))


def _nonlinear_moments(
    B_f: NonlinearGaussianTransition, loc: Array, chol: Array, inputs: _NLInputs
) -> tuple[Array, Array]:
    """The mean and covariance of ``f(x, controls)`` under ``x ~ N(loc, chol cholᵀ)`` and the
    step's control beliefs, before the innovations: the joint ``z = (x, u_beliefs)`` with a
    block-diagonal root is pushed through ``f``, to first order (``f(μ_z)``, ``J Σ_z Jᵀ`` with
    ``J = ∂f/∂z`` at ``μ_z``, which is ``J_x Σ J_xᵀ + J_u S J_uᵀ``) or by the unscented
    transform (:data:`UNSCENTED_KAPPA`, :data:`UNSCENTED_BETA`)."""
    d = loc.shape[-1]
    spans, at = [], d
    for _, mean, _ in inputs.beliefs:
        spans.append((at, at + mean.shape[-1]))
        at += mean.shape[-1]
    loc_z = jnp.concatenate([loc, *(mean for _, mean, _ in inputs.beliefs)], axis=-1)
    roots = [chol, *(root for _, _, root in inputs.beliefs)]
    root_z = jax.scipy.linalg.block_diag(*roots) if len(roots) > 1 else chol

    def g(z: Array) -> Array:
        controls = list(inputs.controls)
        for (pos, _, _), (lo, hi) in zip(inputs.beliefs, spans, strict=True):
            controls[pos] = z[lo:hi]
        return B_f.f(z[:d], tuple(controls))

    if B_f.propagation == "first_order":
        jac = jax.jacfwd(g)(loc_z)
        pushed = jac @ root_z
        return g(loc_z), pushed @ pushed.T
    points, weights = sigma_points(loc_z, root_z, kappa=UNSCENTED_KAPPA)
    ys = jax.vmap(g)(points)
    mean = weights @ ys
    dev = ys - mean
    cov_weights = weights.at[0].add(UNSCENTED_BETA)  # α = 1: the centre's 1 − α² + β
    return mean, (dev * cov_weights[:, None]).T @ dev


def _nonlinear_step(q: Gaussian, B_f: NonlinearGaussianTransition, inputs: _NLInputs) -> Gaussian:
    """``x' = f(x, controls) + ε`` for a Gaussian belief: the moments of ``f``'s output
    (:func:`_nonlinear_moments`) plus the innovation's offset and covariance."""
    mean, cov = _nonlinear_moments(B_f, q.loc, q.chol, inputs)
    return Gaussian.from_cov(mean + inputs.loc, cov + inputs.chol @ inputs.chol.T)


def _nonlinear_mixture_step(
    q: MixtureGaussian,
    B_f: NonlinearGaussianTransition,
    inputs: _NLInputs,
    k: int,
    schedule: str,
) -> MixtureGaussian:
    """A mixture-carried factor through a nonlinear transition: each component through
    :func:`_nonlinear_step` at its own moments, its weight kept, then reduced to ``k`` as
    :func:`_mixture_step` reduces (one mode)."""

    def one(mean: Array, chol: Array) -> Gaussian:
        return _nonlinear_step(Gaussian(loc=mean, chol=chol), B_f, inputs)

    stepped = jax.vmap(one)(q.means, q.chols)
    grown = MixtureGaussian(log_weights=q.log_weights, means=stepped.loc, chols=stepped.chol)
    if schedule == "mode_aligned":
        return grown.reduce_mode_aligned(k, 1)
    return grown.reduce(k)


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
        # the mean is the epistemic one throughout: conditioning leaves it unchanged
        noises[cond.column] = cond_t.R_chol_at(epistemic.loc)
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
) -> MixtureGaussian:
    """The switching composition: exact grow through all modes, then reduce(k).

    ``schedule`` is ``"runnalls"`` (merge anywhere) or ``"mode_aligned"`` (Runnalls
    within each producing mode's group, so slots stay mode-owned). The modes are the discrete
    state parents; the control input enters each mode's offset (and innovation, for a belief).
    """
    d = B_f.dim
    bs = inputs.loc.reshape((-1, d))
    M = bs.shape[0]
    Q_chols = (
        jnp.broadcast_to(inputs.chol, (M, d, d))
        if inputs.shared
        else inputs.chol.reshape((-1, d, d))
    )
    logw = _mode_log_weights(parent_beliefs) if parent_beliefs else jnp.zeros((M,))
    F = inputs.F
    Fs = jnp.broadcast_to(F, (M, d, d)) if F.ndim == 2 else F.reshape((-1, d, d))
    if inputs.u_mean is not None:
        G = inputs.G
        du = G.shape[-1]
        Gs = jnp.broadcast_to(G, (M, d, du)) if G.ndim == 2 else G.reshape((-1, d, du))
        bs, Q_chols = _push_input(bs, Q_chols, Gs, inputs.u_mean, inputs.u_root)
    grown = propagate(q, Fs, bs, Q_chols, logw)
    if schedule == "mode_aligned":
        return grown.reduce_mode_aligned(k, M)
    return grown.reduce(k)


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
    the computation runs for a traced one."""
    chol = B_f.innovations.chol
    if not isinstance(chol, jax.core.Tracer):
        if np.any(np.diagonal(np.asarray(chol), axis1=-2, axis2=-1) == 0.0):
            raise ValueError(_NO_PROCESS_NOISE)
        return B_f
    singular = jnp.any(jnp.diagonal(chol, axis1=-2, axis2=-1) == 0.0)
    guarded = eqx.error_if(chol, singular, _NO_PROCESS_NOISE)
    return eqx.tree_at(lambda b: b.innovations.chol, B_f, guarded)


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
    the observed action nodes).
    """
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

    def transition(qs: tuple[Belief, ...], action: Array) -> tuple[Belief, ...]:
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
        for f, fam in enumerate(families):
            if fam == CATEGORICAL:
                parents = tuple(qs[i] for i in B_deps[f])
                if has_control_axes(table, f, B[f].parent_ndims):
                    index = control_read(action, f, table)
                else:
                    index = ()
                new[f] = _categorical_step(B[f], parents, index)
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
            elif isinstance(B[f], NonlinearGaussianTransition):
                inputs = _nl_inputs(B[f], table.control_parents[f], kinds, entries(f))
                if fam == GAUSSIAN:
                    new[f] = _step_pair(
                        qs[f],
                        lambda q, B_f=B[f], inputs=inputs: _nonlinear_step(q, B_f, inputs),
                        conditioners_of.get(f, ()),
                        present_t,
                        at_step,
                    )
                else:
                    new[f] = _nonlinear_mixture_step(
                        qs[f], B[f], inputs, mixture_k, mixture_schedule
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
                parents = tuple(new[i] for i in B_deps[f])
                new[f] = _mixture_step(
                    qs[f],
                    B[f],
                    parents,
                    _lg_inputs(B[f], lg_reads[f], entries(f)),
                    mixture_k,
                    mixture_schedule,
                )
        return tuple(new)

    return transition


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
    factor_names: tuple[str, ...],
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
    (``StepContext.qs_epistemic``). ``present`` is a
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
    its factor.
    """
    options = dict(
        decomposition=decomposition,
        state_infogain_modalities=state_infogain_modalities,
        param_infogain_modalities=param_infogain_modalities,
        param_infogain_factors=param_infogain_factors,
        modality_names=modality_names,
        factor_names=factor_names,
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

        def contracting_step_terms(qs, action, t, qs_prev=None):
            extended = (
                *qs,
                *(Categorical.from_probs(p) for p in action_node_beliefs(action, sensed)),
            )
            return inner(extended, action, t, qs_prev)

        return contracting_step_terms
    trimmed = tuple(deps[:1] if m in selecting else deps for m, deps in enumerate(A_deps))
    options["channel_selection"] = tuple(
        None if m in selecting else mask for m, mask in enumerate(channel_selection)
    )

    def step_terms(qs, action, t, qs_prev=None):
        extended = (
            *qs,
            *(Categorical.from_probs(p) for p in action_node_beliefs(action, sensed)),
        )
        A_t = tuple(
            select_observation(
                lg_point(a),
                tuple(
                    point_level(
                        node_entry(action, sensed.columns[g - sensed.offset], sensed.layout),
                        sensed.columns[g - sensed.offset],
                        sensed,
                        modality_names[m],
                        factor_names[A_deps[m][0]],
                    )
                    for g in selecting[m]
                ),
            )
            if m in selecting
            else a
            for m, a in enumerate(A)
        )
        inner = _make_step_terms(
            A_t, B, trimmed, B_deps, families, obs_targets, state_targets, **options
        )
        return inner(extended, action, t, qs_prev)

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
    factor_names: tuple[str, ...],
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
):
    """:func:`make_step_terms` over declarations every channel reads whole."""
    mods, facs = modality_names, factor_names
    # the channels no plan step observes: their leaves are zeros, with nothing computed for them
    absent = static_absent(present)
    c_deps = dict(c_deps or {})
    obs_c_deps = {k: v for k, v in c_deps.items() if k in obs_targets}
    state_c_deps = {k: v for k, v in c_deps.items() if k in state_targets}
    # LG slots are lowered to point declarations once here, so the per-step
    # leaves below read H/c/R_chol as fields rather than re-factorising a
    # distributional noise on every access.
    declared = A
    A = tuple(lg_point(a) if isinstance(a, LGObservation) else a for a in A)
    cat_mods = tuple(m for m, a in enumerate(A) if isinstance(a, TabularObservation))
    # a Gaussian emission over categorical parents alone has no state behind it: its leaves
    # read its parents' configurations (per-variable-infogain.md V11 decision 7)
    emission_mods = tuple(m for m, a in enumerate(A) if is_emission(a))
    lg_mods = tuple(
        m for m, a in enumerate(A) if isinstance(a, LGObservation) and m not in emission_mods
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
        factor_names=facs,
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
    ) -> TermTree:
        paired = any(isinstance(q, BeliefPair) for q in qs)
        qs_in = qs
        qs_epistemic = epistemic_beliefs(qs) if paired else None
        qs_conditioned = conditioned_beliefs(qs) if paired else None
        qs_all = predictive_beliefs(qs)  # the state factors, then the action nodes read
        qs = qs_all[: len(facs)]
        if qs_prev is not None:
            qs_prev = predictive_beliefs(qs_prev)
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
                        held[f], sub(A, live), sub_blocks[f], present_at(live), chols
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
                    held[f], sub(A, live), present_at(live), chols
                )
            else:
                total_correlations[f] = zero_leaf()
            if ig_decomposition:
                readings.update(zip(live, gains, strict=True))
                readings.update({m: Interval.exact(zero_leaf()) for m in chans if m in absent})
        trees: list[TermTree] = []
        if obs_idx:
            q_obs = expected_obs(qs_all, sub(A, obs_idx), sub(A_deps, obs_idx))
            names = sub(mods, obs_idx)
            trees.append(
                utility(q_obs, obs_t, names) if ig_decomposition else risk(q_obs, obs_t, names)
            )
        if lg_obs_idx:
            q_lg = tuple(
                predictive_obs(qs[A_deps[m][0]], A[m], parents)
                for m, parents in zip(lg_obs_idx, lg_mode_beliefs(qs_all, lg_obs_idx), strict=True)
            )
            names = sub(mods, lg_obs_idx)
            trees.append(
                lg_utility(q_lg, obs_t, names) if ig_decomposition else risk(q_lg, obs_t, names)
            )
        if flat_idx:
            q_flat = expected_obs(qs_all, sub(A, flat_idx), sub(A_deps, flat_idx))
            trees.append(flat_risk(q_flat, sub(mods, flat_idx)))
        if lg_flat_idx:
            trees.append(
                {
                    "risk": {
                        mods[m]: -predictive_entropy(qs[A_deps[m][0]], A[m], parents)
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
                    qs_all, sub(A, amb_idx), sub(A_deps, amb_idx), sub(mods, amb_idx), selected
                )
            )
        if lg_rest_idx:
            trees.append(
                lg_ambiguity(
                    sub(A, lg_rest_idx),
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
                h, h_cond = emission_entropies(A[m], tuple(qs_all[g] for g in A_deps[m]), masks[m])
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

        if correlated:
            trees.append(
                {
                    CORRELATION: {
                        facs[f]: Interval.exact(
                            total_correlations[f]
                            if f in total_correlations
                            else observation_correlation(qs[f], sub(A, chans), present_at(chans))
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
                        held[f], sub(A, chans), sub_blocks[f], present_at(chans)
                    )
                    for f, chans in interacting
                }
            else:
                leaves = {
                    facs[f]: -conditional_correlations[f]
                    if f in conditional_correlations and sub_groups[f] == chans
                    else -conditional_correlation(
                        qs[f], sub(A, chans), sub_blocks[f], present_at(chans)
                    )
                    for f, chans in interacting
                }
            trees.append({INTERACTION: {k: Interval.exact(v) for k, v in leaves.items()}})
        if state_t and (branching or not cross_entropy_targets):
            score = cross_entropy_risk if branching else risk
            trees.append(score(qs, state_t, facs))
        elif state_t:
            kl_targets = {k: v for k, v in state_t.items() if k not in cross_entropy_targets}
            ce_targets = {k: v for k, v in state_t.items() if k in cross_entropy_targets}
            if kl_targets:
                trees.append(risk(qs, kl_targets, facs))
            trees.append(cross_entropy_risk(qs, ce_targets, facs))
        tree = merge(*trees)
        if present_t is not None:
            tree = mask_absent(tree, present_t, mods)
        return tree

    return step_terms


def make_terminal_terms(
    terminal_targets: dict[str, Distribution],
    factor_names: tuple[str, ...],
    h_deps: tuple[tuple[int, ...], ...] | None = None,
    branching: bool = False,
    cross_entropy_targets: frozenset[str] = frozenset(),
):
    """Family-generic terminal-risk evaluator, or None without H designations.

    Conditional terminal targets (``h_deps``) resolve against the final beliefs. ``branching``
    marks a closure built for a branching planner, where each terminal target is scored by its
    cross-entropy (:func:`~hybrid_aif.objectives.discrete.terminal_cross_entropy_risk`; design:
    per-variable-infogain.md V11 decision 8). ``cross_entropy_targets`` names the factors whose
    terminal targets an open-loop closure scores by their cross-entropy, the targets a
    closed-loop pass corrects (:mod:`hybrid_aif.objectives.closed_loop`).
    """
    if not terminal_targets:
        return None
    ce_targets = {k: v for k, v in terminal_targets.items() if k in cross_entropy_targets}
    terminal_targets = {k: v for k, v in terminal_targets.items() if k not in cross_entropy_targets}
    deps = h_deps if h_deps is not None else ((),) * len(factor_names)
    conditional = any(deps[f] for f, name in enumerate(factor_names) if name in terminal_targets)

    def terminal_terms(qs_final: tuple[Belief, ...]) -> TermTree:
        qs_final = predictive_beliefs(qs_final)
        if branching:
            every = {**terminal_targets, **ce_targets}
            return terminal_cross_entropy_risk(qs_final, every, deps, factor_names)
        targets = (
            resolve_conditional_targets(terminal_targets, deps, qs_final, factor_names)
            if conditional
            else terminal_targets
        )
        tree = terminal_risk(qs_final, targets, factor_names)
        if ce_targets:
            no_deps = ((),) * len(factor_names)
            covered = terminal_cross_entropy_risk(qs_final, ce_targets, no_deps, factor_names)
            tree = {"terminal_risk": {**tree["terminal_risk"], **covered["terminal_risk"]}}
        return tree

    return terminal_terms
