"""The estimator registry: how each objective leaf is computed.

:data:`ESTIMATORS` maps ``(variable kind, family)`` to the ``infogain`` estimator
the ``epistemic`` selector mounts:

====================  ===================  ====================  ========
variable kind         family               estimator             leaf key
====================  ===================  ====================  ========
``state``             ``categorical``      ``predictive``        channel
``state``             ``linear_gaussian``  ``predictive``        channel
``state``             ``emission``         ``predictive``        channel
``param:likelihood``  ``dirichlet``        ``wnorm``             variable
``param:transition``  ``dirichlet``        ``wnorm``             variable
``param:likelihood``  ``mniw``             ``blended_entropy``   variable
====================  ===================  ====================  ========

A variable with no entry (the LG noise and head blocks, ``mlp`` weights) makes
:func:`prepare` raise rather than silently lose its leaf, as does an MNIW emission with
discrete parents, refused when the selector resolves. A continuous factor no channel
observes has no information to gain and carries no ``infogain`` leaf; its entropy is the
``uncertainty`` row, mounted by ``System.terms``. :data:`ROWS` holds
the name-keyed rows ``System.terms`` mounts; one row name may carry several
estimators, dispatched by the mounted variable's family. Every entry is a
``prepare(statics) -> step(ctx)`` pair: ``prepare`` hoists loop invariants at trace
time, ``step`` returns ``{leaf key: Interval}`` inside the planning scan, on either
the distribution-object carrier or the fused prob-space one. Each row's ``derived``
warrant is stamped onto every leaf it returns.

Design: docs/design/per-variable-infogain.md V4, V11; docs/design/objective-terms.md
"""

import dataclasses
import zlib
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import xlogy

from hybrid_aif.controls import control_read, num_control_axes, take_controls
from hybrid_aif.distributions import (
    ConditionalDirichlet,
    Dirichlet,
    InverseGamma,
    InverseWishart,
    MLPCategorical,
    MLPConditional,
)
from hybrid_aif.distributions.neural import LearnedVector, PredictorHead
from hybrid_aif.factors import (
    Deterministic,
    LinearGaussianObservation,
    LinearGaussianObservationMNIW,
)
from hybrid_aif.interval import Interval
from hybrid_aif.objectives import continuous, discrete
from hybrid_aif.objectives.base import INFOGAIN, register_term

# ------------------------------------------------------------------- key vocabulary

STATE = "state"

#: families
CATEGORICAL = "categorical"
LINEAR_GAUSSIAN = "linear_gaussian"
EMISSION = "emission"  # a Gaussian emission over categorical parents alone
GAUSSIAN = "gaussian"  # a state carried as one Gaussian (the hybrid path's belief family)
MIXTURE = "mixture"  # a state carried as a Gaussian mixture
DIRICHLET = "dirichlet"
INVERSE_GAMMA = "inverse_gamma"
INVERSE_WISHART = "inverse_wishart"
MATRIX_NORMAL = "matrix_normal"
MNIW = "mniw"
MLP = "mlp"
MLP_CATEGORICAL = "mlp_categorical"
PREDICTOR = "predictor"
DETERMINISTIC = "deterministic"
LEARNED_VECTOR = "learned_vector"

#: leaf keyings: the observing channel, the factor block, the variable key
CHANNEL = "channel"
BLOCK = "block"
VARIABLE = "variable"
SINGLE = "single"  # one leaf, keyed by the conditional the row read (a ROWS-only keying)

#: what a :data:`ROWS` entry is mounted on: a declared kernel in a slot
#: (``"occupancy:transition:stoch"``), an observation channel (``"predicted_reward:obs:reward"``),
#: a state variable by bare name (``"model_uncertainty:stoch"``), or nothing (the whole rollout).
SLOT_TARGET = "slot"
OBS_TARGET = "obs"
STATE_TARGET = "state_variable"


def param_kind(slot: str) -> str:
    """The variable kind of a declaration slot's parameter variables."""
    return f"param:{slot}"


class Statics(NamedTuple):
    """Build-time facts an estimator's ``prepare`` may hoist from (trace time).

    ``index`` is the declaration indices this estimator scores; the rest is the
    model structure needed to name and address them. ``tables`` carries the fused
    path's :class:`~hybrid_aif.objectives.compiled.CompiledTables`. ``scoring``,
    ``declared`` and ``aux_beliefs`` are narrowed to one row by :func:`mount_statics`.
    ``key`` is a PRNG key for sampled rollouts (``None`` means expected), folded
    per row so no two rows share a stream. ``selected`` holds one parent mask per
    modality (``A_deps`` order; ``None`` = the whole dependency set), empty when
    every channel is scored over its whole set. ``dims`` holds, per factor, the
    selected dimensions of a Gaussian-carried factor (positions; ``None`` = the
    whole factor), empty when no factor is selected in part. ``policy_len`` is the System's
    declared plan length; ``horizon`` is the length of the plan actually being
    scored (``policies.shape[1]``, or ``imagine``'s ``horizon``), which a row
    indexing by step sizes its per-step tables to.
    """

    index: tuple[int, ...]
    modality_names: tuple[str, ...] = ()
    controlled: tuple[bool, ...] = ()
    num_controls: tuple[int, ...] = ()
    factor_names: tuple[str, ...] = ()
    A: tuple = ()
    B: tuple = ()
    A_deps: tuple[tuple[int, ...], ...] = ()
    B_deps: tuple[tuple[int, ...], ...] = ()
    B_control_deps: tuple[tuple[int, ...], ...] | None = None  # System.B_control_deps
    tables: object | None = None
    # --- what a ROWS entry hoists from, beyond the declared model above
    target: str | None = None  # the resolved variable key the row was mounted on, if any
    scoring: object | None = None  # this row's own parameters, out of SystemParams.scoring
    declared: object | None = None  # this row's own structure, out of System.scoring
    structure: object | None = None  # the amortized declaration (families, widths, q-plan)
    policy_len: int = 0
    horizon: int = 0  # the scored plan's length: what a step-indexed row sizes its tables to
    controls: object | None = None  # the flat continuous-action layout, when there is one
    aux_beliefs: object | None = None  # this row's own model's per-call beliefs
    key: object | None = None  # a PRNG key for rows that roll a model: None = expected rollout
    selected: tuple = ()  # per-modality parent masks for state info gain (V9); () = all full
    dims: tuple = ()  # per-factor selected dimensions (V10); () = every factor whole


class StepContext(NamedTuple):
    """Per-step dynamics an estimator's ``step`` reads inside the planning scan.

    ``qs`` are the post-transition beliefs (distribution objects, or probability
    vectors on the fused carrier); ``qs_prev`` the previous step's, present only
    when a mounted leaf reads the step's inputs. ``parents``, ``q_obs`` and
    ``h_cond`` are the fused path's per-modality quantities, shared with the other
    terms. ``qs_dist`` is ``None`` except along an imagined rollout, where ``qs``
    holds a straight-through sample and ``qs_dist`` the distribution it was drawn
    from.
    """

    qs: tuple
    action: Array
    qs_prev: tuple | None = None
    parents: dict | None = None
    q_obs: dict | None = None
    h_cond: dict | None = None
    t: Array | None = None  # the rollout step, for a row reading a per-step precomputed table
    qs_dist: tuple | None = None  # the *distributions* the beliefs came from, when they differ
    qs_epistemic: tuple | None = None  # the beliefs conditioned on the observations expected
    #   at the plan's earlier steps, before this step's (planning-with-observations.md D1, D2);
    #   None under planning_belief="prior"
    qs_conditioned: tuple | None = None  # the same after this step's expected observations
    readings: dict | None = None  # channel index -> its ``infogain`` Interval, evaluated once
    #   per Gaussian-carried factor from one factorisation of the stacked predictive covariance
    #   (:func:`~hybrid_aif.objectives.continuous.linear_gaussian_readings`); a channel absent
    #   from it is scored by :func:`~hybrid_aif.objectives.continuous.state_info_gain`


Step = Callable[[StepContext], dict[str, Interval]]


class Mount(NamedTuple):
    """One resolved ``terms=`` entry: which row, on what, in which family.

    ``key`` is the ``terms=`` string as written, ``index`` the declaration indices
    the row scores, and ``family`` the target's variable family, which picks the
    estimator (:func:`row_variant`); ``None`` for a row with no target. ``dims`` is the
    sub-block a state-variable row was mounted on through ``<factor>:<dim>`` keys (sorted
    positions; ``None``: the whole factor).
    """

    key: str
    name: str
    index: tuple[int, ...] = ()
    family: str | None = None
    dims: tuple[int, ...] | None = None


@dataclasses.dataclass(frozen=True)
class Estimator:
    """One registry entry: an estimator, its leaf keying, and its warrant.

    In :data:`ESTIMATORS` (mounted by ``epistemic``) ``term`` is
    :data:`~hybrid_aif.objectives.base.INFOGAIN` and ``target`` is unused; in
    :data:`ROWS` (mounted by ``System.terms``) ``term`` is the row's own term and
    ``target`` what it is mounted on. ``families`` lists the variable families the
    row has this estimator for (``()`` = any). ``parametric`` rows read
    ``SystemParams.scoring`` and must state ``derived`` explicitly;
    ``needs_structure`` rows also own a declared structure in ``System.scoring``.
    ``trainer`` names the trainer that steps the row's parameters, as a plain
    string that is only tested for ``None`` or printed, never imported; a
    ``scoring:<row>`` rate key is legal only when it is set. ``parameters`` names
    a parametric row's parameters as a model description's term record writes
    them: several names name the items of its ``SystemParams.scoring`` entry in
    order (an entry holding only the first item may hold it bare), and one name
    holds the whole tuple as a list. It is empty for a row whose parameters are
    those of its own declared model (design: model-description-0.3.md, Q18.5d).
    """

    name: str  # which estimator: "predictive" | "wnorm" | "blended_entropy" | a row name
    keying: str  # how its leaves are keyed: CHANNEL | BLOCK | VARIABLE | SINGLE
    prepare: Callable[[Statics], Step]  # distribution-object carrier
    prepare_compiled: Callable[[Statics], Step] | None = None  # fused prob-space carrier
    term: str = INFOGAIN  # which term its leaves land under
    target: str | None = None
    #   ROWS only: SLOT_TARGET | OBS_TARGET | STATE_TARGET | None (the row has no target)
    families: tuple[str, ...] = ()  # ROWS only: the variable families it can read (() = any)
    parametric: bool = False  # does the estimator read SystemParams.scoring?
    needs_structure: bool = False  # does it own a declared structure in System.scoring?
    derived: bool | None = None  # warrant stamped on each leaf; None = True, illegal if parametric
    terminal: bool = False  # ROWS only: scored once at the horizon end, not per step
    reads_prev: bool = False  # does ``step`` read the previous step's beliefs?
    #: ``(entry, declared) -> optimizer carry`` for the row's SystemParams.scoring entry, when
    #: a trainer steps it: an optax state for a parameter tree, a LearnState for an owned model.
    opt_state: Callable[[object, object], object] | None = None
    trainer: str | None = None  # which trainer steps this row's parameters (None = none)
    parameters: tuple[str, ...] = ()  # the names of the scoring entry's items, in order

    def __post_init__(self) -> None:
        if self.parametric and self.derived is None:
            raise ValueError(
                f"estimator row {self.name!r} carries trainable parameters, so it gets no "
                "'derived' default: state derived=False (a net's number that nothing bounds) "
                "or derived=True (a genuine derivation *through* trained parameters) at "
                "registration. The only place a trust label can be wrong should be the one "
                "reviewed line that registers the row (design: amortization.md A4)."
            )
        if self.derived is None:
            object.__setattr__(self, "derived", True)


def scoring_trainer(name: str) -> str | None:
    """Which trainer steps ``SystemParams.scoring[name]``, or ``None`` if none exists.

    Looks ``name`` up as an estimator name in both :data:`ROWS` and
    :data:`ESTIMATORS`. Raises :class:`KeyError` for an unknown name.
    """
    if name in ROWS:
        return ROWS[name].trainer
    for estimator in ESTIMATORS.values():
        if estimator.name == name:
            return estimator.trainer
    raise KeyError(
        f"unknown scoring entry {name!r}; SystemParams.scoring is keyed by estimator name, "
        f"and the registered names are {sorted(_scoring_names())}"
    )


def _scoring_names() -> set[str]:
    return set(ROWS) | {estimator.name for estimator in ESTIMATORS.values()}


def _sub(collection, index: tuple[int, ...]) -> tuple:
    return tuple(collection[i] for i in index)


def _needs_prev(qs_prev) -> None:
    if qs_prev is None:
        raise ValueError(
            "a leaf over the transition's parents (parameter info gain, occupancy) needs the "
            "previous step's beliefs; this closure must be driven through the (qs, qs_prev) "
            "planning carry System.planner_context builds"
        )


# --------------------------------------------------- distribution-object estimators


def _channel_masks(st: Statics) -> tuple | None:
    """The mounted channels' parent masks, or ``None`` when every one is scored in full."""
    if not st.selected:
        return None
    masks = _sub(st.selected, st.index)
    return masks if any(mask is not None for mask in masks) else None


def _state_categorical(st: Statics) -> Step:
    A, deps = _sub(st.A, st.index), _sub(st.A_deps, st.index)
    names = _sub(st.modality_names, st.index)
    masks = _channel_masks(st)

    def step(ctx: StepContext) -> dict[str, Interval]:
        return discrete.state_info_gain(ctx.qs, A, deps, names, selected=masks)[INFOGAIN]

    return step


def _state_linear_gaussian(st: Statics) -> Step:
    index, A_deps = st.index, st.A_deps
    A, names = _sub(st.A, index), _sub(st.modality_names, index)
    masks = _channel_masks(st)
    # a channel on a Gaussian factor selected in part reads its information about the sub-block
    # (per-variable-infogain.md V10); a switching channel's factor is mixture-carried, never one
    dims = (
        tuple(st.dims[A_deps[m][0]] if len(A_deps[m]) == 1 else None for m in index)
        if st.dims
        else None
    )

    def step(ctx: StepContext) -> dict[str, Interval]:
        # a non-switching channel reads the epistemic chain when there is one, so its leaf is
        # the gain of this step's observation given the plan's earlier ones; a switching
        # channel keeps the predictive belief (planning-with-observations.md D2)
        epistemic = ctx.qs if ctx.qs_epistemic is None else ctx.qs_epistemic
        fused = {} if ctx.readings is None else ctx.readings
        rest = tuple(i for i, m in enumerate(index) if m not in fused)
        leaves: dict[str, Interval] = {}
        if rest:

            def pick(seq):
                return None if seq is None else tuple(seq[i] for i in rest)

            leaves = continuous.state_info_gain(
                tuple(
                    (epistemic if len(A_deps[index[i]]) == 1 else ctx.qs)[A_deps[index[i]][0]]
                    for i in rest
                ),
                pick(A),
                pick(names),
                mode_beliefs=tuple(tuple(ctx.qs[g] for g in A_deps[index[i]][1:]) for i in rest),
                selected=pick(masks),
                dims=pick(dims),
            )[INFOGAIN]
        return {
            name: fused[m] if m in fused else leaves[name]
            for m, name in zip(index, names, strict=True)
        }

    return step


def _state_emission(st: Statics) -> Step:
    """``I(y; S)`` per Gaussian emission over categorical parents alone, ``S`` its selected
    parents (:func:`continuous.emission_info_gain`)."""
    index, A_deps = st.index, st.A_deps
    A, names = _sub(st.A, index), _sub(st.modality_names, index)
    masks = _channel_masks(st)

    def step(ctx: StepContext) -> dict[str, Interval]:
        parents = tuple(tuple(ctx.qs[g] for g in A_deps[m]) for m in index)
        return continuous.emission_info_gain(A, parents, names, selected=masks)[INFOGAIN]

    return step


def _uncertainty(st: Statics) -> Step:
    """``H[q(x)]`` of each mounted continuous factor's predictive belief, or of its marginal
    over the sub-block the row was mounted on (the ``uncertainty`` row)."""
    index, names = st.index, _sub(st.factor_names, st.index)
    dims = _sub(st.dims, index) if st.dims else (None,) * len(index)

    def step(ctx: StepContext) -> dict[str, Interval]:
        blocks = tuple(
            ctx.qs[f] if d is None else continuous.marginal(ctx.qs[f], d)
            for f, d in zip(index, dims, strict=True)
        )
        return continuous.uncertainty(blocks, names)["uncertainty"]

    return step


def _param_likelihood(st: Statics) -> Step:
    A, deps = _sub(st.A, st.index), _sub(st.A_deps, st.index)
    names = _sub(st.modality_names, st.index)

    def step(ctx: StepContext) -> dict[str, Interval]:
        q_obs = discrete.expected_obs(ctx.qs, A, deps)
        return discrete.param_info_gain(A, ctx.qs, q_obs, deps, names)[INFOGAIN]

    return step


def _param_transition(st: Statics) -> Step:
    index = st.index
    B, deps = _sub(st.B, index), _sub(st.B_deps, index)
    names = _sub(st.factor_names, index)

    def step(ctx: StepContext) -> dict[str, Interval]:
        _needs_prev(ctx.qs_prev)
        return discrete.transition_param_info_gain(
            B, ctx.qs, ctx.qs_prev, ctx.action, index, deps, names, st.B_control_deps
        )[INFOGAIN]

    return step


def _param_likelihood_mniw(st: Statics) -> Step:
    """``I(θ; y)`` per matrix-normal inverse-Wishart emission (:func:`continuous.param_info_gain`),
    the state read as the linear-Gaussian state leaf reads it. ``st.A`` holds the declarations,
    not their point lowerings; the draws are hoisted out of the planning scan."""
    index, A_deps = st.index, st.A_deps
    names = _sub(st.modality_names, index)
    models = _sub(st.A, index)
    draws = tuple(continuous.mniw_draws(model) for model in models)

    def step(ctx: StepContext) -> dict[str, Interval]:
        beliefs = ctx.qs if ctx.qs_epistemic is None else ctx.qs_epistemic
        return {
            discrete.param_key("likelihood", name): Interval.exact(
                continuous.param_info_gain(beliefs[A_deps[m][0]], model, draw)
            )
            for m, name, model, draw in zip(index, names, models, draws, strict=True)
        }

    return step


# ------------------------------------------------------- fused prob-space estimators


def _state_categorical_compiled(st: Statics) -> Step:
    """Reads ``ctx.h_cond``, which already holds ``E[H[o | S]]`` for a partially selected
    channel (:func:`~hybrid_aif.objectives.compiled.make_step_terms`)."""
    index, mods = st.index, st.modality_names

    def step(ctx: StepContext) -> dict[str, Interval]:
        return {
            mods[m]: Interval.exact(-xlogy(ctx.q_obs[m], ctx.q_obs[m]).sum(axis=-1) - ctx.h_cond[m])
            for m in index
        }

    return step


def _param_likelihood_compiled(st: Statics) -> Step:
    index, mods, wA = st.index, st.modality_names, st.tables.wA

    def step(ctx: StepContext) -> dict[str, Interval]:
        return {
            discrete.param_key("likelihood", mods[m]): Interval.exact(
                jnp.sum(
                    ctx.q_obs[m] * discrete.contract_probs(wA[m], ctx.parents[m], keep_child=True),
                    axis=-1,
                )
            )
            for m in index
        }

    return step


def _param_transition_compiled(st: Statics) -> Step:
    index, facs, B_deps, wB = st.index, st.factor_names, st.B_deps, st.tables.wB
    cdeps = st.B_control_deps
    # Static per-factor flag: does the transition table carry its control axes?
    has_control = tuple(
        wB[f].ndim == len(B_deps[f]) + 1 + num_control_axes(cdeps, f) for f in index
    )

    def step(ctx: StepContext) -> dict[str, Interval]:
        _needs_prev(ctx.qs_prev)
        leaves = {}
        for f, ctl in zip(index, has_control, strict=True):
            w = take_controls(wB[f], control_read(ctx.action, f, cdeps)) if ctl else wB[f]
            fd = discrete.contract_probs(
                w, tuple(ctx.qs_prev[g] for g in B_deps[f]), keep_child=True
            )
            leaves[discrete.param_key("transition", facs[f])] = Interval.exact(
                jnp.sum(ctx.qs[f] * fd, axis=-1)
            )
        return leaves

    return step


def _occupancy_transition(st: Statics) -> Step:
    """``E_{q(s)} H[B(·|s,a)]`` of one declared tabular transition (the ``occupancy`` row).

    The expectation contracts the table's conditional entropy over the parent
    beliefs, so the leaf differs from the post-transition belief's entropy by
    ``I(s'; s)``. Design: docs/design/objective-terms.md "Build note: the tabular
    ``occupancy`` row".
    """
    index = st.index
    B, deps = _sub(st.B, index), _sub(st.B_deps, index)
    names = _sub(st.factor_names, index)
    cdeps = st.B_control_deps
    # Static per-factor flag: does the declaration carry its control axes?
    has_control = tuple(
        B_f.parent_ndims == len(d) + num_control_axes(cdeps, f)
        for f, B_f, d in zip(index, B, deps, strict=True)
    )

    def step(ctx: StepContext) -> dict[str, Interval]:
        _needs_prev(ctx.qs_prev)
        return {
            discrete.param_key("transition", name): Interval.exact(
                discrete.transition_occupancy_leaf(
                    B_f,
                    tuple(ctx.qs_prev[g] for g in d),
                    control_read(ctx.action, f, cdeps) if ctl else None,
                )
            )
            for f, name, B_f, d, ctl in zip(index, names, B, deps, has_control, strict=True)
        }

    return step


def _occupancy_transition_compiled(st: Statics) -> Step:
    index, facs, B_deps, B_probs = st.index, st.factor_names, st.B_deps, st.tables.B_probs
    cdeps = st.B_control_deps
    has_control = tuple(
        B_probs[f].ndim == len(B_deps[f]) + 1 + num_control_axes(cdeps, f) for f in index
    )

    def step(ctx: StepContext) -> dict[str, Interval]:
        _needs_prev(ctx.qs_prev)
        leaves = {}
        for f, ctl in zip(index, has_control, strict=True):
            table = (
                take_controls(B_probs[f], control_read(ctx.action, f, cdeps)) if ctl else B_probs[f]
            )
            h = -xlogy(table, table).sum(axis=-(len(B_deps[f]) + 1))
            leaves[discrete.param_key("transition", facs[f])] = Interval.exact(
                discrete.contract_probs(
                    h, tuple(ctx.qs_prev[g] for g in B_deps[f]), keep_child=False
                )
            )
        return leaves

    return step


# ---------------------------------------------------------------------- the registry

ESTIMATORS: dict[tuple[str, str], Estimator] = {
    (STATE, CATEGORICAL): Estimator(
        "predictive", CHANNEL, _state_categorical, _state_categorical_compiled
    ),
    (STATE, LINEAR_GAUSSIAN): Estimator("predictive", CHANNEL, _state_linear_gaussian),
    (STATE, EMISSION): Estimator("predictive", CHANNEL, _state_emission),
    (param_kind("likelihood"), DIRICHLET): Estimator(
        "wnorm", VARIABLE, _param_likelihood, _param_likelihood_compiled
    ),
    (param_kind("transition"), DIRICHLET): Estimator(
        "wnorm", VARIABLE, _param_transition, _param_transition_compiled
    ),
    # an estimate, not a bracket: an interpolated H[y] less a sampled E H[y | θ]
    (param_kind("likelihood"), MNIW): Estimator(
        "blended_entropy", VARIABLE, _param_likelihood_mniw, derived=False
    ),
}


#: The name-keyed rows ``System.terms`` mounts; the amortized ones are added at import
#: by :mod:`hybrid_aif.objectives.amortized`. The entry here is a name's first
#: registration, which every family-independent fact is read off.
ROWS: dict[str, Estimator] = {}

#: ``row name -> family -> estimator``: which estimator computes a row for a family.
_ROW_VARIANTS: dict[str, dict[str, Estimator]] = {}

#: Fields a row's family variants must agree on (only ``prepare``/``derived`` may differ).
_SHARED_ROW_FIELDS = ("term", "keying", "target", "parametric", "needs_structure", "terminal")


def register_row(row: Estimator, *, slot: str | None = None) -> Estimator:
    """Register a ``terms=``-selectable row under its own name.

    ``slot`` restricts a :data:`SLOT_TARGET` row to one declaration slot. A row
    name is one segment and may not be a slot word or ``obs``. A name registered
    again is a family variant, admitted only when both registrations declare
    disjoint ``families`` and agree on :data:`_SHARED_ROW_FIELDS` and ``slot``;
    anything else raises :class:`ValueError`.
    """
    reserved = ("likelihood", "transition", "initial_state_prior", "recognition", "scoring", "obs")
    if ":" in row.name or row.name in reserved:
        raise ValueError(
            f"row name {row.name!r} collides with the variable key grammar: a row name is one "
            f"segment and may not be one of {sorted(reserved)} (a terms= key is "
            "'<row>' or '<row>:<variable key>', so the first segment must decide which it is)"
        )
    if slot is not None and row.target != SLOT_TARGET:
        raise ValueError(f"row {row.name!r} takes no slot restriction (target={row.target!r})")
    variants = _ROW_VARIANTS.setdefault(row.name, {})
    if row.name in ROWS:
        first = ROWS[row.name]
        clash = sorted(set(row.families) & set(variants))
        if clash:
            raise ValueError(f"row {row.name!r} is already registered for family/families {clash}")
        if not row.families or not first.families:
            raise ValueError(
                f"row {row.name!r} is already registered; a second estimator under one row name "
                "is admitted only as a *family variant*, so both registrations must declare "
                "disjoint families= (one term, one bucket, one estimator per family — "
                "objective-terms.md's lattice)"
            )
        differing = [f for f in _SHARED_ROW_FIELDS if getattr(row, f) != getattr(first, f)]
        if differing:
            raise ValueError(
                f"row {row.name!r}'s family variants disagree on {differing}; a variant may "
                "differ only in its estimator and its warrant (prepare/derived) — everything "
                "else is read off the row without knowing which family is in play"
            )
        if _ROW_SLOT[row.name] != slot:
            raise ValueError(f"row {row.name!r}'s family variants disagree on slot=")
    else:
        ROWS[row.name] = row
        _ROW_SLOT[row.name] = slot
    variants.update({family: row for family in row.families})
    return row


#: A :data:`SLOT_TARGET` row's one legal slot word, or ``None`` for "any".
_ROW_SLOT: dict[str, str | None] = {}


def row_slot(name: str) -> str | None:
    """The declaration slot a :data:`SLOT_TARGET` row is restricted to, if any."""
    return _ROW_SLOT.get(name)


def by_name(name: str) -> Estimator | None:
    """The estimator called ``name``: a ``terms=`` row, or an ``ESTIMATORS`` entry by
    its own name; ``None`` when neither."""
    if name in ROWS:
        return ROWS[name]
    for est in ESTIMATORS.values():
        if est.name == name:
            return est
    return None


def get_row(name: str) -> Estimator:
    """One ``terms=`` row by name; unknown names fail loudly, listing the rows."""
    if name not in ROWS:
        raise ValueError(
            f"unknown objective row {name!r}; the registered rows are {sorted(ROWS)} "
            "(hybrid_aif.objectives.registry.ROWS). The infogain rows are not selected here "
            "— they are what the 'epistemic' selector mounts."
        )
    return ROWS[name]


def row_families(name: str) -> tuple[str, ...]:
    """Every variable family the row has an estimator for (``()`` = no restriction)."""
    return tuple(sorted(_ROW_VARIANTS.get(name, {})))


def row_variant(name: str, family: str | None = None) -> Estimator:
    """The estimator a row uses for one variable family.

    Falls back to the row's own entry when ``family`` is ``None`` or has no variant;
    eligibility is checked earlier, by the ``terms=`` resolver.
    """
    row = get_row(name)
    return _ROW_VARIANTS.get(name, {}).get(family, row) if family is not None else row


def get(kind: str, family: str) -> Estimator | None:
    """The registered estimator for a variable, or ``None`` when there is none."""
    return ESTIMATORS.get((kind, family))


def prepare_row(
    name: str, statics: Statics, family: str | None = None, compiled: bool = False
) -> Step:
    """Resolve one :data:`ROWS` entry and run its ``prepare`` stage (trace time).

    ``family`` picks the estimator and ``compiled`` its carrier; the variant's
    ``derived`` flag is stamped onto every leaf.
    """
    row = row_variant(name, family)
    stage = row.prepare_compiled if compiled else row.prepare
    if stage is None:
        raise NotImplementedError(
            f"the {name!r} row's estimator for family {family!r} has no fused prob-space "
            "carrier; plan this model through the distribution-object path (a non-tabular "
            "declaration routes there by itself)"
        )
    return stamp(stage(statics), row.derived)


def prepare(kind: str, family: str, statics: Statics, compiled: bool = False) -> Step:
    """Resolve one estimator and run its ``prepare`` stage (trace time).

    Raises :class:`NotImplementedError` for a variable with no registered
    estimator, so a selected variable never silently loses its leaf.
    """
    est = get(kind, family)
    if est is None:
        raise NotImplementedError(
            f"no infogain estimator registered for variable kind {kind!r}, family "
            f"{family!r}; the objectives registry (hybrid_aif.objectives.registry) "
            f"holds {sorted(ESTIMATORS)}"
        )
    stage = est.prepare_compiled if compiled else est.prepare
    if stage is None:  # pragma: no cover - every fused-path family has an entry
        raise NotImplementedError(
            f"estimator {est.name!r} for ({kind!r}, {family!r}) has no fused prob-space "
            "implementation; plan this model through the distribution-object path"
        )
    return stamp(stage(statics), est.derived)


def stamp(step: Step, derived: bool) -> Step:
    """Wrap a ``step`` so every Interval it returns carries its row's warrant.

    The flag is ANDed with each Interval's own, so an underived leaf stays underived.
    """
    if derived:
        return step

    def stamped(ctx: StepContext) -> dict[str, Interval]:
        return {key: Interval(iv.lower, iv.upper, derived=False) for key, iv in step(ctx).items()}

    return stamped


def param_family(decl) -> str:
    """The parameter-posterior family of a learnable declaration.

    An LG observation with independent blocks reports its noise or head block's
    family, which has no estimator, so the error names a real block.
    """
    if isinstance(decl, MLPConditional):
        return MLP
    if isinstance(decl, MLPCategorical):
        return MLP_CATEGORICAL
    if isinstance(decl, PredictorHead):
        return PREDICTOR
    if isinstance(decl, Deterministic):
        return DETERMINISTIC
    if isinstance(decl, LearnedVector):
        return LEARNED_VECTOR
    if isinstance(decl, LinearGaussianObservationMNIW):
        return MNIW
    if isinstance(decl, LinearGaussianObservation):
        if isinstance(decl.noise, InverseGamma):
            return INVERSE_GAMMA
        if isinstance(decl.noise, InverseWishart):
            return INVERSE_WISHART
        return MATRIX_NORMAL
    if isinstance(decl, ConditionalDirichlet | Dirichlet):
        return DIRICHLET
    return type(decl).__name__


def novelty_estimator(slot: str, decl) -> Estimator | None:
    """The parameter-novelty estimator for a declaration in ``slot``, if registered."""
    return get(param_kind(slot), param_family(decl))


def mount_statics(mount: Mount, statics: Statics) -> Statics:
    """Narrow the shared statics to one mount.

    ``scoring``, ``declared`` and ``aux_beliefs`` arrive as row-keyed mappings and
    leave as this row's entry; ``key`` is folded with the row name; ``dims`` is the
    mount's own sub-block, never the ``epistemic`` selection's.
    """
    row = get_row(mount.name)
    dims = ()
    if mount.dims is not None:
        dims = tuple(
            mount.dims if f in mount.index else None for f in range(len(statics.factor_names))
        )
    return statics._replace(
        index=mount.index,
        target=mount.key,
        dims=dims,
        scoring=_require(statics.scoring, mount.name) if row.parametric else None,
        declared=dict(statics.declared or {}).get(mount.name) if row.needs_structure else None,
        aux_beliefs=dict(statics.aux_beliefs or {}).get(mount.name),
        key=None
        if statics.key is None
        else jax.random.fold_in(statics.key, zlib.crc32(mount.name.encode()) & 0x7FFFFFFF),
    )


def prepare_step_rows(
    mounts: tuple[Mount, ...], statics: Statics, compiled: bool = False
) -> tuple[tuple[str, Step], ...]:
    """``(term, step)`` per mounted row, for the tabular and LG planning paths.

    Raises :class:`NotImplementedError` for a terminal row, which these per-step
    paths cannot score.
    """
    for mount in mounts:
        if get_row(mount.name).terminal:
            raise NotImplementedError(
                f"the {mount.name!r} row is scored once at the horizon end, and the declared "
                "(tabular/LG) planning path has no terminal-row seam — only terminal risk. "
                "Mounting it here would silently drop its leaf."
            )
    return tuple(
        (get_row(m.name).term, prepare_row(m.name, mount_statics(m, statics), m.family, compiled))
        for m in mounts
    )


def _require(scoring, name: str) -> object:
    if scoring is None or name not in scoring:
        raise ValueError(
            f"the {name!r} row is mounted on this System but SystemParams.scoring carries no "
            f"{name!r} entry (it holds {sorted(scoring or ())}); the System declares which rows "
            "are scored and the slot holds their parameters, so a missing entry is a mismatch, "
            "not a silent skip (design: amortization.md A12 item 1 — mounting is never inferred "
            "from the slot's contents)."
        )
    return scoring[name]


# --------------------------------------------------------------------- the tabular rows

register_term("occupancy", -1.0)
#: what a per-channel sum miscounts when several channels read one Gaussian factor, a cost in G:
#: under ``utility_infogain`` the interaction information ``TC(Y) − TC(Y | x_S)`` of the
#: readings about the selected state, subtracted from the channels' information gains (negative
#: under synergy); under ``risk_ambiguity`` the joint ambiguity's ``−TC(Y | x_S)`` on a
#: sub-block (per-variable-infogain.md V11 decision 3)
register_term("observation_interaction", 1.0)
#: the total correlation ``TC(Y)`` of a step's predicted readings on one Gaussian factor, the part
#: of the joint risk a per-channel sum leaves out (``risk_ambiguity``; V11 decision 3)
register_term("observation_correlation", 1.0)
#: a continuous belief's entropy, charged as a cost (per-variable-infogain.md V11 decision 4)
register_term("uncertainty", 1.0)

#: The ``occupancy`` row over a declared tabular transition (its amortized variant is
#: registered in :mod:`hybrid_aif.objectives.amortized`). Design: objective-terms.md.
OCCUPANCY_TABULAR = register_row(
    Estimator(
        "occupancy",
        VARIABLE,
        _occupancy_transition,
        _occupancy_transition_compiled,
        term="occupancy",
        target=SLOT_TARGET,
        families=(CATEGORICAL,),
        derived=True,
        reads_prev=True,
    ),
    slot="transition",
)

#: The ``uncertainty`` row: the entropy of a continuous factor's predictive belief, a cost, on
#: ``"uncertainty:<factor>"`` or a sub-block's ``"uncertainty:<factor>:<dim>"``. Exact on a
#: Gaussian, a Kolchinsky–Tracey pair on a mixture. Design: per-variable-infogain.md V11.
UNCERTAINTY = register_row(
    Estimator(
        "uncertainty",
        BLOCK,
        _uncertainty,
        term="uncertainty",
        target=STATE_TARGET,
        families=(GAUSSIAN, MIXTURE),
        derived=True,
    )
)
