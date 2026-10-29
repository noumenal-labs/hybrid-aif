"""The planning faces of :class:`~hybrid_aif.system.System`: ``planner_context``,
``infer_policies``, ``score_policies`` and ``act``."""

from collections.abc import Callable, Mapping
from typing import Literal

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax import Array

from hybrid_aif.controls import (
    FLOAT,
    LEVEL,
    ControlLayout,
    ControlStep,
    ControlTarget,
    SensedNodes,
    control_risk,
    target_on_box,
)
from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalDirichlet,
    Dirichlet,
    Gaussian,
    Learnable,
    TruncatedGaussian,
)
from hybrid_aif.factors import (
    LinearGaussianObservation,
    LinearGaussianObservationMNIW,
    LinearGaussianTransition,
    NonlinearGaussianTransition,
)
from hybrid_aif.objectives import TermTree, registry
from hybrid_aif.objectives import amortized as _amortized_terms
from hybrid_aif.objectives import closed_loop as _closed_loop
from hybrid_aif.objectives import continuous as _continuous
from hybrid_aif.objectives import hybrid as _hybrid
from hybrid_aif.objectives.compiled import (
    compile_tables,
    make_step_terms,
    make_terminal_terms,
    make_transition,
)
from hybrid_aif.objectives.presence import LevelEntry, Presence
from hybrid_aif.planning.base import (
    Planner,
    PlannerContext,
    UseBound,
    get_planner,
    planner_branches,
    planner_legal_depth,
    planner_needs_key,
)
from hybrid_aif.planning.enumerate import (
    enumerate_policies,
    evaluate_policies,
    legal_logits,
    legal_steps,
    sample_action,
)
from hybrid_aif.planning.gradient import PlanPosterior
from hybrid_aif.planning.habit import habit_action
from hybrid_aif.system import keyed
from hybrid_aif.system.core import AmortizedScoring, Belief, System, SystemParams, _carry_prev


def _gamma(system, params: SystemParams):
    """The policy precision: ``params.precision`` (a traced scalar array) when present, else
    the static ``System.gamma``."""
    return system.gamma if params.precision is None else params.precision


def _refuse_legal_if_unusable(system, legal, params: SystemParams) -> None:
    """``legal=`` is a level mask over discrete controls; two declarations have none."""
    params = system._p(params)
    if legal is None:
        return
    if system._space(params).continuous:
        raise ValueError(
            "legal= is a mask over discrete control levels, and this System's control nodes "
            "are continuous, where no level set exists; their constraint is the box in "
            "params.control (design: action-legality.md L4)"
        )
    if isinstance(legal, Mapping) and system._space(params).mixed:
        names = system._names(params)
        floats = [n for n, k in zip(names.controls, names.kinds, strict=True) if k == FLOAT]
        named = [k for k in legal if k in floats]
        if named:
            raise ValueError(
                f"legal= names the continuous control node(s) {named}, where no level set "
                "exists; their constraint is the box in params.control (design: "
                "action-legality.md L4). Mask the discrete nodes"
            )
    if params.policy_prior is not None and callable(params.policy_prior):
        raise ValueError(
            "legal= with a policy module in policy_prior: the module is an actor the habit "
            "planner samples per step, not a distribution over an enumerated policy set, so "
            "a level mask has nothing to compose with (design: action-legality.md L4)"
        )


def planner_context(
    system,
    params: SystemParams,
    beliefs: Mapping[str, Belief],
    key: Array | None = None,
    aux_beliefs: Mapping[str, object] | None = None,
    scoring_key: Array | None = None,
    horizon: int | None = None,
    present: Mapping[str, bool | Array] | None = None,
) -> PlannerContext:
    params = system._params_in(params, "planner_context")
    return system._planner_context(
        params,
        system._beliefs_in(params, beliefs, "planner_context"),
        key=key,
        aux_beliefs=system._aux_in(aux_beliefs, "planner_context"),
        scoring_key=scoring_key,
        horizon=horizon,
        face="planner_context",
        present=present,
        branching=planner_branches(system.planner),
    )


def _aux_in(
    system, aux_beliefs: Mapping[str, object] | None, face: str
) -> dict[str, object] | None:
    """``{row: {factor: belief}}`` as the row-keyed positional carries the rows read.

    A row whose structure in ``System.scoring`` is a nested System takes that System's beliefs
    keyed by its factor names (a positional carry raises, pointing at the nested System's
    ``from_positional("beliefs", ...)``). An entry for any other row passes through as given.
    """
    if aux_beliefs is None:
        return None
    declared = dict(system.scoring)
    out: dict[str, object] = {}
    for row, carry in aux_beliefs.items():
        nested = declared.get(row)
        if isinstance(nested, System):
            where = f"{face} aux_beliefs[{row!r}]"
            if not isinstance(carry, Mapping):
                raise TypeError(
                    f"{where} takes the nested System's beliefs as a mapping keyed by its "
                    f"factor names ({list(nested._names().factors)}); convert positional data "
                    f"with dict(system.scoring)[{row!r}].from_positional('beliefs', ...)"
                )
            out[row] = keyed.beliefs_in(nested._names(), carry, where)
        else:
            out[row] = carry
    return out


def _planner_context(
    system,
    params: SystemParams,
    beliefs: tuple[Belief, ...],
    key: Array | None = None,
    aux_beliefs: Mapping[str, object] | None = None,
    scoring_key: Array | None = None,
    horizon: int | None = None,
    face: str = "this face",
    present: Mapping[str, bool | Array] | None = None,
    branching: bool = False,
) -> PlannerContext:
    """:meth:`planner_context` over the positional belief tuple and row-keyed positional
    carries. ``horizon`` is the length of the plan the context will score (``None``:
    ``policy_len``); the amortized path hands it to the scoring rows. Targets on control
    nodes are split off here and scored around the step terms (:func:`control_risk`).
    ``branching`` marks a branching planner (:func:`planner_branches`); when this System also
    declares ``branch_on`` modalities, so that the plan branches, the step and terminal terms
    score a state target by its cross-entropy (design: per-variable-infogain.md V11 decisions
    6 and 8).
    Without ``branch_on`` the planner scores as an open-loop one does."""
    params = system._p(params)
    branching = branching and bool(system.branch_on)
    if system._is_amortized(params):
        system._refuse_sensed(face)
    system._validate_node_targets(params, face)
    params, control_targets = system._control_targets(params)
    ctx = _context_without_control_risk(
        system,
        params,
        beliefs,
        key,
        aux_beliefs,
        scoring_key,
        horizon=horizon,
        face=face,
        present=present,
        branching=branching,
        control_targets=control_targets,
    )
    if not control_targets:
        return ctx
    return ctx._replace(
        step_terms_fn=control_risk(ctx.step_terms_fn, control_targets, ctx.policy_len)
    )


def _validate_node_targets(system, params: SystemParams, face: str = "this face") -> None:
    """A target on a discrete control node is a Categorical over its levels; on a continuous
    one a Gaussian over its values, or a TruncatedGaussian on its box."""
    params = system._p(params)
    controls = system._table.control_names
    named = [
        (what, k, v)
        for what, targets in (
            ("state_targets", params.state_targets),
            ("terminal_targets", params.terminal_targets),
        )
        for k, v in targets.items()
        if k in controls
    ]
    if not named:
        return
    space = system._space(params)
    levels = dict(zip(controls, space.levels, strict=True))
    kinds = dict(zip(controls, space.kinds, strict=True))
    sizes = dict(zip(controls, space.sizes, strict=True))
    boxes = dict(zip(controls, space.boxes, strict=True))
    for what, k, v in named:
        if kinds[k] == FLOAT:
            ok = isinstance(v, Gaussian | TruncatedGaussian) and (
                v.batch_shape == () and v.dim == sizes[k]
            )
            if not ok:
                raise ValueError(
                    f"{face}: {what}[{k!r}] is a target on continuous control node {k!r}: "
                    f"pass a Gaussian over its ({sizes[k]},) values, or a TruncatedGaussian "
                    f"on its box; got a {type(v).__name__}"
                )
            target_on_box(v, boxes[k], f"{face}: {what}[{k!r}]")
            continue
        if not isinstance(v, Categorical) or v.logits.shape != (levels[k],):
            got = (
                f"a Categorical of shape {tuple(v.logits.shape)}"
                if isinstance(v, Categorical)
                else f"a {type(v).__name__}"
            )
            raise ValueError(
                f"{face}: {what}[{k!r}] is a target on control node {k!r}, a Categorical over "
                f"its {levels[k]} level(s); got {got}"
            )


def _present_in(
    system,
    present: Mapping[str, bool | Array | Mapping[str, Array]] | None,
    horizon: int,
    face: str,
    params: SystemParams,
) -> Presence | None:
    """``present`` as a :class:`~hybrid_aif.objectives.presence.Presence`, or ``None`` when
    every modality is present throughout. An entry is a bool (the whole plan), a ``(horizon,)``
    bool array (per step), or a one-key mapping from a discrete control node's name to a bool
    per level of that node (read at each step's level of the node). A modality left out is
    present; an unknown modality or node, a continuous node, a mapping with other than one
    key, a wrong length and a non-bool dtype are refused."""
    if present is None:
        return None
    mods = system.modality_names
    unknown = sorted(set(present) - set(mods))
    if unknown:
        raise ValueError(
            f"{face}: present= names {unknown}, which are not modalities of this System "
            f"({list(mods)})"
        )
    space = system._space(params)
    nodes = system._table.control_names
    discrete = [n for k, n in enumerate(nodes) if space.kinds[k] == LEVEL]
    columns, by_level, never = [], [], set()
    for m, name in enumerate(mods):
        flag = present.get(name, True)
        if isinstance(flag, Mapping):
            if len(flag) != 1:
                raise ValueError(
                    f"{face}: present[{name!r}] under control-node levels takes one control "
                    f"node per entry, got {sorted(flag)}; discrete control nodes: {discrete}"
                )
            ((node, per_level),) = flag.items()
            if node not in nodes:
                raise ValueError(
                    f"{face}: present[{name!r}] names {node!r}, which is not a control node; "
                    f"discrete control nodes: {discrete}"
                )
            k = nodes.index(node)
            if space.kinds[k] != LEVEL:
                raise ValueError(
                    f"{face}: present[{name!r}] names the continuous control node {node!r}; "
                    f"a per-level availability needs a discrete node: {discrete}"
                )
            try:
                arr = jnp.asarray(per_level)
            except TypeError:
                arr = jnp.zeros((), jnp.int32)
            if not jnp.issubdtype(arr.dtype, jnp.bool_):
                raise ValueError(
                    f"{face}: present[{name!r}][{node!r}] must be a bool per level, got "
                    f"{type(per_level).__name__} of dtype {arr.dtype}"
                )
            if arr.shape != (space.sizes[k],):
                raise ValueError(
                    f"{face}: present[{name!r}][{node!r}] has shape {arr.shape}; node "
                    f"{node!r} has {space.sizes[k]} level(s), one bool each"
                )
            by_level.append(LevelEntry(m, k, arr, name, node))
            columns.append(jnp.ones((horizon,), bool))
            continue
        if not (isinstance(flag, bool) or jnp.issubdtype(jnp.asarray(flag).dtype, jnp.bool_)):
            raise ValueError(
                f"{face}: present[{name!r}] must be a bool or a bool array, got dtype "
                f"{jnp.asarray(flag).dtype}"
            )
        if not isinstance(flag, jax.core.Tracer) and not np.asarray(flag).any():
            never.add(m)  # known absent at every step, though the mask may be traced
        flag = jnp.asarray(flag, bool)
        if flag.ndim == 0:
            flag = jnp.broadcast_to(flag, (horizon,))
        elif flag.shape != (horizon,):
            raise ValueError(
                f"{face}: present[{name!r}] has shape {flag.shape}; a per-step availability "
                f"is a ({horizon},) bool array, one entry per plan step"
            )
        columns.append(flag)
    return Presence(jnp.stack(columns, axis=-1), tuple(by_level), frozenset(never))


def _conditioners(
    system,
    params: SystemParams,
    families: tuple[str, ...],
    sensed: SensedNodes | None = None,
) -> tuple["_hybrid.Conditioner", ...]:
    """The linear-Gaussian modalities the epistemic chain conditions on: those observing one
    Gaussian-carried factor and no mode parent, and those whose mode parents are all action
    nodes, which select their head and noise at the step's level (a switching channel on a
    state mode parent keeps the predictive belief; design: planning-with-observations.md D2,
    action-node-modalities.md S4)."""
    out = []
    engine = system._engine_A_deps
    for m, decl in enumerate(params.likelihood):
        if not isinstance(decl, LinearGaussianObservation | LinearGaussianObservationMNIW):
            continue
        deps = engine[m]
        if deps[0] >= len(families) or families[deps[0]] != _hybrid.GAUSSIAN:
            continue  # an emission over action nodes alone has no Gaussian factor to condition
        nodes = deps[1:]
        if nodes and (sensed is None or any(d < sensed.offset for d in nodes)):
            continue
        model = _continuous.lg_point(decl)
        R_chol = model.R_chol_at if not nodes else (lambda loc, R=model.R_chol: R)
        out.append(
            _hybrid.Conditioner(
                factor=deps[0],
                column=m,
                H=model.H,
                c=model.c,
                R_chol_at=R_chol,
                nodes=nodes,
                modality=system.modality_names[m],
                factor_name=system._facs[deps[0]],
            )
        )
    return tuple(out)


def _refuse_feedback_amortized(system, face: str) -> None:
    """Raise when ``feedback`` is set on the amortized path, naming the setting."""
    if system.feedback is not None:
        raise ValueError(
            f"{face}: feedback={system.feedback!r} on the amortized (neural state-space) path; "
            "the closed-loop pass reads each step's linearisation and the epistemic chain's "
            "covariances, which decoded codes do not carry (design: closed-loop-planning.md "
            "L3, L4). Plan with feedback=None"
        )


def _context_without_control_risk(
    system,
    params: SystemParams,
    beliefs: tuple[Belief, ...],
    key: Array | None = None,
    aux_beliefs: Mapping[str, object] | None = None,
    scoring_key: Array | None = None,
    horizon: int | None = None,
    face: str = "this face",
    present: Mapping[str, bool | Array] | None = None,
    branching: bool = False,
    control_targets: tuple[ControlTarget, ...] = (),
) -> PlannerContext:
    """:meth:`_planner_context` before control-node targets are scored. Under ``feedback``
    the closures carry the closed-loop record (:func:`~hybrid_aif.objectives.closed_loop.
    closed_loop`), whose pass reads ``control_targets`` for the corrections' cost."""
    if system._is_amortized(params):
        _refuse_feedback_amortized(system, face)
        if system.planning_belief == "posterior" or present is not None:
            raise NotImplementedError(
                f'{face}: planning_belief="posterior" and present= are built on the '
                "tabular and linear-Gaussian paths; the amortized path's observations are "
                "decoded by networks, with no closed-form conditioning to plan with "
                "(design: planning-with-observations.md D2)"
            )
        return _amortized_context(
            system, params, beliefs, key, aux_beliefs, scoring_key, horizon=horizon, face=face
        )
    if aux_beliefs:
        raise ValueError(
            f"aux_beliefs={sorted(aux_beliefs)} on a non-amortized System: auxiliary "
            "beliefs are the per-call belief carry of a scoring row's own model, and "
            "no row on the tabular/LG paths owns one (design: amortization.md A12 "
            "item 2; amortization-debt-ledger.md entry 5)"
        )
    if any(
        isinstance(b, ConditionalCategorical | Dirichlet | ConditionalDirichlet) for b in beliefs
    ):
        raise ValueError(
            "beliefs must be marginal distributions; resolve conditional "
            "(hierarchical) or learnable priors with System.initial_priors(params) first"
        )
    system._validate(params, face)
    sensed = system._sensed_nodes(params, face)
    A_deps = system._engine_A_deps
    (
        state_infogain_modalities,
        param_infogain_modalities,
        param_infogain_factors,
    ) = system._term_targets(params)
    mounts = system._row_mounts(params)
    channel_selection = system._channel_selections(params)
    # A row over the transition's *parents* needs the same (qs, qs_prev) carry
    # transition novelty does — one predicate, whichever mounted it.
    carry_prev = bool(param_infogain_factors) or any(
        registry.row_variant(m.name, m.family).reads_prev for m in mounts
    )
    present_mask = system._present_in(
        present, system.policy_len if horizon is None else int(horizon), face, params
    )
    if system._is_hybrid(params):
        families = system._families(params)
        mods, facs = (system.modality_names, system._facs)
        space = system._space(params)
        # a modality no step observes conditions nothing
        never = _hybrid.static_absent(present_mask)
        conditioners = (
            tuple(
                c for c in _conditioners(system, params, families, sensed) if c.column not in never
            )
            if system.planning_belief == "posterior"
            else ()
        )
        cov = None
        lowered = tuple(b.point() if isinstance(b, Learnable) else b for b in params.transition)
        if system.feedback is not None:
            cov = _closed_loop.coverage(
                families=families,
                B=lowered,
                A=tuple(params.likelihood),
                A_deps=A_deps,
                state_targets=params.state_targets,
                terminal_targets=params.terminal_targets,
                obs_targets=params.obs_targets,
                control_targets=control_targets,
                decomposition=system.decomposition,
                factor_names=facs,
                modality_names=mods,
                space=space,
                c_deps=system._effective_C_deps(),
            )
            if not cov.targets:
                cov = None
        covered = frozenset(() if cov is None else (t.key for t in cov.targets if t.column is None))
        transition_fn = _hybrid.make_transition(
            params.transition,
            system._B,
            families,
            system.mixture_k,
            system.mixture_schedule,
            table=system._table,
            space=space,
            conditioners=conditioners,
            present=present_mask,
            sensed=sensed,
        )
        step_terms_fn = _hybrid.make_step_terms(
            params.likelihood,
            params.transition,
            A_deps,
            system._B,
            families,
            params.obs_targets,
            params.state_targets,
            decomposition=system.decomposition,
            state_infogain_modalities=state_infogain_modalities,
            param_infogain_modalities=param_infogain_modalities,
            param_infogain_factors=param_infogain_factors,
            modality_names=mods,
            factor_names=facs,
            mounts=mounts,
            channel_selection=channel_selection,
            factor_dims=system._dim_selections(params),
            control_deps=system._table,
            present=present_mask,
            scoring=params.scoring,
            branching=branching,
            cross_entropy_targets=covered,
            c_deps=system._effective_C_deps(),
            layout=space.layout,
            sensed=sensed,
        )
        terminal_terms_fn = _hybrid.make_terminal_terms(
            params.terminal_targets,
            facs,
            system._effective_H_deps(),
            branching=branching,
            cross_entropy_targets=covered,
        )
        init_beliefs = _hybrid.prepare_beliefs(beliefs, families, system.mixture_k)
        if system.planning_belief == "posterior":
            init_beliefs = _hybrid.pair_beliefs(init_beliefs, families, conditioners)
        if carry_prev:
            transition_fn, step_terms_fn, terminal_terms_fn, init_beliefs = _carry_prev(
                transition_fn, step_terms_fn, terminal_terms_fn, init_beliefs
            )
        if cov is not None:
            transition_fn, step_terms_fn, terminal_terms_fn, init_beliefs = (
                _closed_loop.closed_loop(
                    transition_fn,
                    step_terms_fn,
                    terminal_terms_fn,
                    init_beliefs,
                    cov=cov,
                    B=lowered,
                    table=system._table,
                    space=space,
                    horizon=system.policy_len if horizon is None else int(horizon),
                    present=present_mask,
                    beliefs_of=(lambda carry: carry[0]) if carry_prev else (lambda c: c),
                )
            )
        return PlannerContext(
            branch_fn=_branch_fn(system, params, families, carry_prev),
            transition_fn=transition_fn,
            step_terms_fn=step_terms_fn,
            terminal_terms_fn=terminal_terms_fn,
            controls=system._table,
            levels=system._control_levels(params),
            policy_len=system.policy_len if horizon is None else int(horizon),
            gamma=_gamma(system, params),
            policy_prior=params.policy_prior,
            init_beliefs=init_beliefs,
            continuous_controls=space.layout,
            key=key,
            kinds=space.kinds,
            sizes=space.sizes,
            boxes=space.boxes,
        )
    tables = compile_tables(
        params.likelihood,
        params.transition,
        params.obs_targets,
        params.state_targets,
        params.terminal_targets,
    )
    num_factors = len(system._B)
    transition_fn = make_transition(tables, system._B, system._table)
    step_terms_fn = make_step_terms(
        tables,
        A_deps,
        system._B,
        num_factors,
        decomposition=system.decomposition,
        state_infogain_modalities=state_infogain_modalities,
        param_infogain_modalities=param_infogain_modalities,
        param_infogain_factors=param_infogain_factors,
        modality_names=system.modality_names,
        factor_names=system._facs,
        mounts=mounts,
        channel_selection=channel_selection,
        control_deps=system._table,
        branching=branching,
        c_deps=system._effective_C_deps(),
        sensed=sensed,
        present=present_mask,
    )
    terminal_terms_fn = make_terminal_terms(
        tables, num_factors, system._facs, system._effective_H_deps(), branching=branching
    )
    init_beliefs = tuple(b.probs for b in beliefs)
    if carry_prev:
        transition_fn, step_terms_fn, terminal_terms_fn, init_beliefs = _carry_prev(
            transition_fn, step_terms_fn, terminal_terms_fn, init_beliefs
        )
    return PlannerContext(
        branch_fn=_branch_fn(system, params, None, carry_prev),
        transition_fn=transition_fn,
        step_terms_fn=step_terms_fn,
        terminal_terms_fn=terminal_terms_fn,
        controls=system._table,
        levels=system._control_levels(params),
        policy_len=system.policy_len if horizon is None else int(horizon),
        gamma=_gamma(system, params),
        policy_prior=params.policy_prior,
        init_beliefs=init_beliefs,
        continuous_controls=None,  # the tabular fast path is discrete by construction
        key=key,
        kinds=system._space(params).kinds,
        sizes=system._space(params).sizes,
        boxes=system._space(params).boxes,
    )


def _amortized_scoring(
    system,
    params: SystemParams,
    aux_beliefs: Mapping[str, object] | None = None,
    key: Array | None = None,
    horizon: int | None = None,
    face: str = "this face",
) -> "AmortizedScoring":
    """The amortized path's scoring closures, unwrapped: the one G path.

    Shared by :meth:`_amortized_context` and :func:`hybrid_aif.train.imagine.imagine`, so an
    imagined plan and a planned one are scored by the same closures. ``needs_prev`` in the
    result says whether the caller must supply the previous step's beliefs. ``aux_beliefs`` is
    the per-call belief carry of a scoring row's own model, keyed by row name: a row that rolls
    out its own model starts from it instead of from the model's declared episode start.
    ``horizon`` is the length of the plan these closures will score (``None``:
    ``policy_len``), a static int a row indexing by step sizes its tables to. Raises when no
    objective row is mounted.
    """
    params = system._p(params)
    system._validate(params, face)
    params, control_targets = system._control_targets(
        params
    )  # scored around the step terms, as planning
    horizon = system.policy_len if horizon is None else int(horizon)
    if horizon < 1:
        raise ValueError(f"the scored horizon must be at least 1, got {horizon}")
    _refuse_feedback_amortized(system, face)
    structure = system._amortized_structure(params, face)
    mounts = system._row_mounts(params)
    codes = system._amortized_epistemic(params, structure)
    mods, facs = (system.modality_names, system._facs)
    layout = system._control_layout(params)
    statics = registry.Statics(
        index=(),
        modality_names=mods,
        factor_names=facs,
        controlled=system._ctl(face),
        num_controls=system._num_controls(params, face),
        A=params.likelihood,
        B=params.transition,
        A_deps=system.A_deps,
        B_deps=system._B,
        structure=structure,
        scoring=params.scoring,
        declared=dict(system.scoring),
        policy_len=system.policy_len,
        horizon=horizon,
        controls=layout,
        aux_beliefs=dict(aux_beliefs or {}),
        key=key,
    )
    step_terms_fn = _amortized_terms.make_step_terms(mounts, codes, statics)
    terminal_terms_fn = _amortized_terms.make_terminal_terms(mounts, statics)
    if step_terms_fn is None and terminal_terms_fn is None:
        raise NotImplementedError(
            "scoring an amortized (neural state-space) System with no objective row "
            "mounted: G would be an empty fold. Declare the rows this model's G carries — "
            f"System(terms=(...)) for the named rows ({', '.join(sorted(registry.ROWS))}) "
            "and epistemic=(<code variable>,) for the "
            "ensemble info gain — with their parameters in SystemParams.scoring (design: "
            "amortization.md A12). The inference faces (infer_states, "
            "update_empirical_prior, predict_beliefs, train.fit_world) need none of this."
        )
    if step_terms_fn is None:  # a g_tail and nothing else: the tail is the whole of G
        step_terms_fn = _amortized_terms.no_step_terms
    if control_targets:
        step_terms_fn = control_risk(step_terms_fn, control_targets, horizon)
    return AmortizedScoring(
        structure=structure,
        step_terms_fn=step_terms_fn,
        terminal_terms_fn=terminal_terms_fn,
        needs_prev=_amortized_terms.needs_prev(mounts, codes),
        controls=layout,
    )


def _amortized_context(
    system,
    params: SystemParams,
    beliefs: tuple[Belief, ...],
    key: Array | None,
    aux_beliefs: Mapping[str, object] | None = None,
    scoring_key: Array | None = None,
    horizon: int | None = None,
    face: str = "this face",
) -> PlannerContext:
    """:meth:`planner_context` on the amortized path.

    ``transition_fn`` is the open-loop step ``update_empirical_prior`` runs; the step and
    terminal terms come from :meth:`_amortized_scoring` (the mounted ``terms=`` rows, the
    ``epistemic`` selector's ``ensemble_disagreement`` leaf, the critic's ``g_tail``), built
    for a plan of ``horizon`` steps (``None``: ``policy_len``). Beliefs are per-factor values
    (a carry vector, a code's probability tensor). ``branch_on`` raises here.
    """
    params = system._p(params)
    scoring = system._amortized_scoring(
        params, aux_beliefs, key=scoring_key, horizon=horizon, face=face
    )
    structure, layout = scoring.structure, scoring.controls
    transition_fn = _amortized_terms.make_transition(
        params.transition,
        system._B,
        structure.families,
        system._ctl(face),
        system._num_controls(params, face),
        layout,
    )
    step_terms_fn, terminal_terms_fn = scoring.step_terms_fn, scoring.terminal_terms_fn
    init_beliefs = tuple(beliefs)
    wrapped = scoring.needs_prev
    if system.branch_on:
        raise ValueError(
            f"branch_on={system.branch_on!r} on the amortized path: branching is exact over a "
            "categorical modality's categorical parents, and this path predicts its outcomes "
            "with heads over codes and carries. Declare no branch_on here (the sophisticated "
            "planner then folds the next decision over the prediction itself)."
        )
    if wrapped:
        transition_fn, step_terms_fn, terminal_terms_fn, init_beliefs = _carry_prev(
            transition_fn, step_terms_fn, terminal_terms_fn, init_beliefs
        )
    return PlannerContext(
        transition_fn=transition_fn,
        step_terms_fn=step_terms_fn,
        terminal_terms_fn=terminal_terms_fn,
        controls=system._table,
        levels=system._control_levels(params),
        policy_len=system.policy_len if horizon is None else int(horizon),
        gamma=_gamma(system, params),
        policy_prior=params.policy_prior,
        init_beliefs=init_beliefs,
        continuous_controls=layout,
        key=key,
        habit_fn=_habit_fn(system, params, layout, carried_prev=wrapped),
        kinds=system._space(params).kinds,
        sizes=system._space(params).sizes,
        boxes=system._space(params).boxes,
    )


def _habit_fn(
    system, params: SystemParams, layout: ControlLayout | None, carried_prev: bool
) -> "Callable[[object, Array], Array] | None":
    """The ``(beliefs, key) -> action`` closure for a policy-module E, else ``None``.

    The System supplies the module's reading of the beliefs (the feature vector every head on
    the amortized path takes) and unwraps a ``(qs, qs_prev)`` carry when ``carried_prev``. The
    draw is returned unclipped: the declared box is applied by the action's consumer
    (:func:`~hybrid_aif.planning.habit.habit_action`).
    """
    params = system._p(params)
    del layout  # the box belongs to the action's consumers, not to the draw
    module = system._habit_module(params)
    if module is None:
        return None

    def habit(carry, key: Array) -> Array:
        beliefs = carry[0] if carried_prev else carry
        return habit_action(key, module, beliefs, _amortized_terms.features)

    return habit


def _branch_fn(
    system, params: SystemParams, families: tuple[str, ...] | None, carry_prev: bool
) -> "Callable[[object], tuple[Array, object]] | None":
    """The ``branch_on`` modalities' outcome closure over the planning carry
    (:func:`hybrid_aif.planning.branching.make_branch_fn`), or ``None``
    without any. ``families`` is the hybrid path's (``None``: all tabular).
    Refuses a modality it cannot branch on, with a sentence."""
    params = system._p(params)
    if not system.branch_on:
        return None
    from hybrid_aif.planning.branching import make_branch_fn

    mods, facs = (system.modality_names, system._facs)
    likelihoods, indices = list(params.likelihood), []
    for name in system.branch_on:
        if name not in mods:
            raise ValueError(
                f"branch_on names {name!r}, which is not a modality; the modalities are "
                f"{list(mods)}"
            )
        m = mods.index(name)
        A = likelihoods[m]
        if isinstance(A, ConditionalDirichlet):
            A = A.point()  # a learnable table branches on its expected table
        parents = system.A_deps[m]
        nodes = [
            system.action_names[g - len(system.B_deps)] for g in parents if g >= len(system.B_deps)
        ]
        if nodes:
            raise ValueError(
                f"branch_on names {name!r}, which observes action node(s) {nodes}; a branch "
                "reads the state beliefs alone, and an action node's belief comes from the "
                "step's action, which a branch does not have"
            )
        continuous = [
            facs[f] for f in parents if families is not None and families[f] != "categorical"
        ]
        if not isinstance(A, ConditionalCategorical) or continuous:
            raise ValueError(
                f"branch_on names {name!r}, and only a categorical modality over "
                "categorical factors can be branched on (its outcomes are finite and the "
                "belief after each is exact over its parents)"
                + (f"; it observes {continuous}" if continuous else "")
            )
        likelihoods[m] = A
        indices.append(m)
    branch = make_branch_fn(likelihoods, system.A_deps, indices)
    if not carry_prev:
        return branch

    def wrapped(carry):
        qs, prev = carry
        p, after = branch(qs)
        return p, (
            after,
            jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (p.shape[0], *x.shape)), prev),
        )

    return wrapped


def _legal_logits(
    system, params: SystemParams, legal, policies: Array, face: str = "this face"
) -> Array | None:
    """``legal=`` (keyed by control node, or one bool per policy) as the log-indicator over
    ``policies``: each node's entry masks that node's column, ``(levels,)`` at the first
    step or ``(H, levels)`` per step; a node left out is unrestricted."""
    if isinstance(legal, list):
        legal = keyed.legal_list(legal, face)
    if legal is None or not (isinstance(legal, Mapping | tuple)):
        return legal_logits(legal, (), system.policy_len, (), policies=policies, face=face)
    names = system._names(params)
    entries = keyed.legal_in(names, legal, face)
    levels = system._control_levels(params)
    return legal_logits(
        entries,
        levels,
        system.policy_len,
        (True,) * len(levels),
        policies=policies,
        names=names.controls,
        face=face,
    )


def _legal_steps(system, params: SystemParams, legal, ctx: PlannerContext, face: str):
    """A keyed ``legal=`` as the bool ``(policy_len, A)`` mask over the first-step actions that
    a planner reading :attr:`PlannerContext.legal_steps` takes, or ``None`` for a mask that is
    not keyed by control node (a bare array is one bool per returned policy)."""
    if isinstance(legal, list):
        legal = keyed.legal_list(legal, face)
    if not isinstance(legal, Mapping | tuple):
        return None
    names = system._names(params)
    entries = keyed.legal_in(names, legal, face)
    levels = system._control_levels(params)
    actions = enumerate_policies(levels, 1, (True,) * len(levels))[:, 0, :]
    return legal_steps(entries, levels, ctx.policy_len, actions, names=names.controls, face=face)


def _plan(
    system,
    params: SystemParams,
    beliefs: tuple[Belief, ...],
    planner: Planner | None,
    use_bound: UseBound,
    metrics: bool,
    plan_key: Array | None = None,
    legal: "Mapping[str, Array] | Array | None" = None,
    face: str = "this face",
    present: Mapping[str, bool | Array] | None = None,
) -> tuple[Categorical, TermTree | None, Array]:
    params = system._p(params)
    _refuse_legal_if_unusable(system, legal, params)  # before any planner question
    planner_fn = get_planner(system.planner) if planner is None else planner
    if plan_key is None and planner_needs_key(planner_fn):
        raise ValueError(
            f"planner {system.planner!r} is stochastic (it declared needs_key=True) but no "
            "plan_key was given: pass plan_key= to infer_policies / act / score_policies. "
            "It is not split off act's action-selection key — a plan's randomness is the "
            "caller's to control, and silently deriving it would make two different "
            "quantities share one stream."
        )
    ctx = system._planner_context(
        params,
        beliefs,
        key=plan_key,
        face=face,
        present=present,
        branching=planner_branches(planner_fn),
    )
    if legal is not None and planner_legal_depth(planner_fn):
        ctx = ctx._replace(legal_steps=_legal_steps(system, params, legal, ctx, face))
    q_pi, tree, policies = planner_fn(ctx, use_bound=use_bound, metrics=metrics)
    if isinstance(policies, PlanPosterior):
        return q_pi, tree, policies
    clamped_steps = isinstance(policies, ControlStep) and _is_clamped(policies, ctx.kinds)
    if not clamped_steps and (
        isinstance(policies, Mapping | ControlStep)
        or (ctx.continuous_controls is None and not jnp.issubdtype(policies.dtype, jnp.integer))
    ):
        raise ValueError(
            f"{face} selects among clamped policies, and the planner returned a belief-valued "
            "policy set: return integer levels, or score a belief-valued policy with "
            "score_policies"
        )
    if legal is not None:
        #  the hard constraint is composed HERE, once, over the policies the planner
        #  actually returned — so an injected planner that never heard of legality is
        #  constrained exactly as the built-in ones are
        positional = policies
        if clamped_steps:  # a mixed set: the levels of each discrete node, 0 for a float one
            positional = jnp.stack(
                [
                    e if kind == LEVEL else jnp.zeros(e.shape[:2], jnp.int32)
                    for e, kind in zip(policies.entries, ctx.kinds, strict=True)
                ],
                axis=-1,
            )
        if ctx.legal_steps is not None:  # the planner masked every depth; the root is row 0
            indicator = jnp.where(ctx.legal_steps[0], 0.0, -jnp.inf)
        else:
            indicator = system._legal_logits(params, legal, positional, face)
        q_pi = Categorical(logits=q_pi.log_probs + indicator)
    return q_pi, tree, policies


def _is_clamped(policies: ControlStep, kinds: tuple[str, ...]) -> bool:
    """Whether a planner's :class:`~hybrid_aif.controls.ControlStep` set clamps every node:
    integer levels ``(P, H)`` for a discrete node, a float array ``(P, H, width)`` for a
    continuous one."""
    if len(policies.entries) != len(kinds):
        return False
    for entry, kind in zip(policies.entries, kinds, strict=True):
        if not hasattr(entry, "dtype"):
            return False
        integer = jnp.issubdtype(entry.dtype, jnp.integer)
        if (kind == LEVEL) != integer or entry.ndim != (2 if kind == LEVEL else 3):
            return False
    return True


def infer_policies(
    system,
    params: SystemParams,
    beliefs: Mapping[str, Belief],
    planner: Planner | None = None,
    use_bound: UseBound = "midpoint",
    metrics: bool = True,
    plan_key: Array | None = None,
    legal: "Mapping[str, Array] | Array | None" = None,
    present: Mapping[str, bool | Array] | None = None,
) -> tuple[Categorical, TermTree | None]:
    params = system._params_in(params, "infer_policies")
    qs = system._beliefs_in(params, beliefs, "infer_policies")
    q_pi, tree, _ = system._plan(
        params, qs, planner, use_bound, metrics, plan_key, legal, "infer_policies", present
    )
    return q_pi, tree


def score_policies(
    system,
    params: SystemParams,
    beliefs: Mapping[str, Belief],
    policies: "Array | Mapping[str, Array | Categorical]",
    aux_beliefs: Mapping[str, object] | None = None,
    key: Array | None = None,
) -> TermTree:
    params = system._params_in(params, "score_policies")
    qs = system._beliefs_in(params, beliefs, "score_policies")
    if isinstance(policies, Mapping):
        policies, single = _belief_policies(system, params, policies, "score_policies")
    else:
        policies = system._check_plan(policies, "policies", params)
        single = policies.ndim == 2
    if single:
        policies = jtu.tree_map(lambda x: x[None], policies)
    ctx = system._planner_context(
        params,
        qs,
        aux_beliefs=system._aux_in(aux_beliefs, "score_policies"),
        scoring_key=key,
        horizon=int(jtu.tree_leaves(policies)[0].shape[1]),
        face="score_policies",
    )
    tree = evaluate_policies(
        policies,
        ctx.init_beliefs,
        ctx.transition_fn,
        ctx.step_terms_fn,
        ctx.terminal_terms_fn,
    )
    if single:
        tree = jtu.tree_map(lambda x: jnp.squeeze(x, axis=0), tree)
    return tree


def _belief_policies(
    system, params: SystemParams, policies: Mapping[str, object], face: str
) -> "tuple[Array | ControlStep, bool]":
    """A keyed policy as its positional form, and whether it is a single ``(H,)`` policy.

    All clamps over one kind of node give the positional plan (the level array, or the flat
    float vector); anything else gives a :class:`~hybrid_aif.controls.ControlStep` of
    per-node entries with leading ``(P, H)`` axes."""
    names = system._names(params)
    keyed.check_mapping(names, "controls", policies, face, label="policies")
    if not names.controls:
        raise ValueError(
            f"{face}: this System has no control node to key a policy by; pass the "
            "positional (P, H, 0) plan"
        )
    space = system._space(params)
    entries, leads, beliefs = [], [], []
    for k, name in enumerate(names.controls):
        value = policies[name]
        if space.kinds[k] == FLOAT:
            size = space.sizes[k]
            if isinstance(value, Gaussian | TruncatedGaussian):
                if value.dim != size:
                    raise ValueError(
                        f"{face}: policies[{name!r}] is a belief over continuous control "
                        f"node {name!r}, a {type(value).__name__} over its ({size},) values; "
                        f"got dimension {value.dim}"
                    )
                entries.append(value)
                leads.append(tuple(value.batch_shape))
                beliefs.append(k)
                continue
            value = jnp.asarray(value)
            if not jnp.issubdtype(value.dtype, jnp.floating) or value.shape[-1:] != (size,):
                raise ValueError(
                    f"{face}: policies[{name!r}] is the value of continuous control node "
                    f"{name!r}: pass a float (P, H, {size}) or (H, {size}) array, or a "
                    f"Gaussian over it; got {value.dtype} {tuple(value.shape)}"
                )
            entries.append(value.astype(jnp.float32))
            leads.append(tuple(value.shape[:-1]))
            continue
        if isinstance(value, Categorical):
            probs = value.probs
            if probs.shape[-1] != space.sizes[k]:
                raise ValueError(
                    f"{face}: policies[{name!r}] is a belief over control node {name!r}, a "
                    f"Categorical over its {space.sizes[k]} level(s); got {probs.shape[-1]}"
                )
            entries.append(probs)
            leads.append(tuple(probs.shape[:-1]))
            beliefs.append(k)
            continue
        value = jnp.asarray(value)
        if not jnp.issubdtype(value.dtype, jnp.integer):
            raise ValueError(
                f"{face}: policies[{name!r}] has dtype {value.dtype}; a node's entry is its "
                "levels (an integer array) or a Categorical over them"
            )
        entries.append(value.astype(jnp.int32))
        leads.append(tuple(value.shape))
    if len(set(leads)) > 1 or (leads and len(leads[0]) not in (1, 2)):
        raise ValueError(
            f"{face}: the entries of a keyed policy share one leading shape, (P, H) or (H,); "
            f"got {dict(zip(names.controls, leads, strict=True))}"
        )
    single = bool(leads) and len(leads[0]) == 1
    if not beliefs and not space.mixed:
        if space.layout is not None:
            plan = jnp.concatenate(entries, axis=-1)
        else:
            plan = jnp.stack(entries, axis=-1)
        return system._check_plan(plan, "policies", params), single
    if system._is_amortized(params):
        raise NotImplementedError(
            f"{face}: a belief-valued or mixed policy reads control nodes one entry at a "
            "time, and this System is on the amortized path, which reads one flat action; "
            "score clamped policies"
        )
    if system._is_hybrid(params):
        families = system._families(params)
        p = system._p(params)
        for k in beliefs:
            readers = system._table.readers[k]
            if space.kinds[k] == LEVEL:
                continuous = [
                    names.factors[f] for f, _ in readers if families[f] != _hybrid.CATEGORICAL
                ]
                if continuous:
                    raise NotImplementedError(
                        f"{face}: control node {names.controls[k]!r} is given a belief and "
                        f"is a mode parent of continuous factor(s) {continuous}, whose "
                        "prediction under it is a Gaussian mixture; that waits on the "
                        "mixture-bounds work (docs/design/per-variable-infogain.md, V9 "
                        "decision 5): clamp it (an integer entry)"
                    )
                continue
            neural = [
                names.factors[f]
                for f, _ in readers
                if not isinstance(
                    p.transition[f], LinearGaussianTransition | NonlinearGaussianTransition
                )
            ]
            if neural:
                raise NotImplementedError(
                    f"{face}: continuous control node {names.controls[k]!r} is given a "
                    f"belief and is read by {neural}, whose transitions take a value; clamp "
                    "it (a float entry)"
                )
    return ControlStep(entries=tuple(entries)), single


def act(
    system,
    key: Array,
    params: SystemParams,
    beliefs: Mapping[str, Belief],
    selection: Literal["stochastic", "deterministic"] = "stochastic",
    alpha: float | None = None,
    mode: Literal["marginal", "policy"] | None = None,
    planner: Planner | None = None,
    use_bound: UseBound = "midpoint",
    metrics: bool = True,
    plan_key: Array | None = None,
    legal: "Mapping[str, Array] | Array | None" = None,
) -> tuple[dict[str, Array], Categorical, TermTree | None]:
    params = system._params_in(params, "act")
    names = system._names(params)
    q_pi, tree, policies = system._plan(
        params,
        system._beliefs_in(params, beliefs, "act"),
        planner,
        use_bound,
        metrics,
        plan_key,
        legal,
        "act",
    )
    if isinstance(policies, PlanPosterior):
        action = policies.mode[0, 0] if selection == "deterministic" else policies.first_step(key)
        return keyed.actions_out(names, action), q_pi, tree
    mixed_set = isinstance(policies, ControlStep)
    if mode is None:
        mode = "marginal" if names.layout is None and not mixed_set else "policy"
    action = sample_action(
        key,
        q_pi,
        policies=policies,
        num_controls=system._control_levels(params),
        mode=mode,
        selection=selection,
        alpha=alpha,
    )
    return keyed.actions_out(names, action), q_pi, tree
