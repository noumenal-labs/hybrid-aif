"""The amortized path's objective layer: the estimator rows a neural state-space G carries.

Registers five rows over declared neural state-space material and builds the
family-dispatched closures ``System.planner_context`` hands a planner (the
amortized counterpart of :mod:`hybrid_aif.objectives.hybrid`). Every leaf here is
``derived=False``. Each term's sign is fixed at registration:

===============================  ================  =======================  ======
row                              term              leaf                     sign
===============================  ================  =======================  ======
``occupancy``                    own               ``E_q H[p(s'|s,a)]``     −1
``ensemble_disagreement``        infogain          MC mutual information    −1
``model_uncertainty``            own               the same leaf            **+1**
``predicted_reward``             own               a scalar head's mean     −1
``g_tail``                       own               the critic at the tail   −1
===============================  ================  =======================  ======

``predicted_reward`` is off-lattice (no target distribution); with it mounted, G
bounds nothing.

Design: docs/design/amortization.md A4, A12; docs/design/objective-terms.md
"""

from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import xlogy

from hybrid_aif.distributions import MLPConditional
from hybrid_aif.distributions.neural import PredictorHead, init_opt_state
from hybrid_aif.inference import amortized as _amortized
from hybrid_aif.interval import Interval
from hybrid_aif.objectives import registry
from hybrid_aif.objectives.base import INFOGAIN, TermTree, merge, register_term
from hybrid_aif.objectives.discrete import param_key

register_term("predicted_reward", -1.0)
register_term("g_tail", -1.0)
register_term("model_uncertainty", 1.0)

_LOG_2PI = float(jnp.log(2.0 * jnp.pi))

#: The block key of a single-leaf row: it names the conditional the row read.
CRITIC_BLOCK = "critic"


def features(beliefs: Sequence[Array]) -> Array:
    """``concat([deter, flat stoch, …])`` — the feature vector every head reads.

    Each belief in declaration order: a carry contributes its vector, a code its
    flattened probability tensor.
    """
    return jnp.concatenate([_amortized._flat(q) for q in beliefs], axis=-1)


def _diag_entropy(scale: Array) -> Array:
    """Entropy of a diagonal Gaussian with the given per-dimension scale.

    The scale is floored at the dtype's smallest normal before the log, so an
    exactly-zero scale gives a finite leaf. Design: docs/design/amortization.md
    (entropy floor)
    """
    floor = jnp.finfo(jnp.asarray(scale).dtype).tiny
    return jnp.sum(jnp.log(jnp.maximum(scale, floor)), axis=-1) + 0.5 * scale.shape[-1] * (
        1.0 + _LOG_2PI
    )


def _action_vector(
    action: Array | None,
    controlled: tuple[bool, ...],
    num_controls: tuple[int, ...],
    controls: object | None,
) -> Array:
    """The whole control step as one float vector: the ensemble's third parent.

    The flat continuous vector where one is declared, else the control nodes' one-hots
    concatenated in node order (the controlled factors' order).
    """
    if action is None:
        raise ValueError("the ensemble_disagreement row scores an action and was given none")
    if controls is not None:
        return jnp.asarray(action, jnp.float32)
    return jnp.concatenate(
        [
            jax.nn.one_hot(jnp.asarray(action)[k], n)
            for k, n in enumerate(n for n, c in zip(num_controls, controlled, strict=True) if c)
        ]
    )


def _action_width(
    controlled: tuple[bool, ...], num_controls: tuple[int, ...], controls: object | None
) -> int:
    return (
        controls.action_dim
        if controls is not None
        else sum(n for n, c in zip(num_controls, controlled, strict=True) if c)
    )


# ------------------------------------------------------------------ the estimators


def _occupancy(st: registry.Statics) -> registry.Step:
    """``E_{q(s)} H[p(s'|s,a)]`` of one declared transition (the ``occupancy`` row).

    The parents are a deterministic carry, so the expectation is the entropy of
    the post-transition belief itself, closed form over the factored categorical.
    Stored as the entropy under a −1 term. Design: docs/design/objective-terms.md
    """
    index, names = st.index, tuple(st.factor_names[f] for f in st.index)

    def step(ctx: registry.StepContext) -> dict[str, Interval]:
        # ``qs_dist`` is the distribution the belief came from, and differs from it on
        # exactly one path: an imagined rollout carries a straight-through *sample*
        # (whose entropy is identically zero, with an infinite gradient at the corner).
        qs = ctx.qs if ctx.qs_dist is None else ctx.qs_dist
        return {
            param_key("transition", name): Interval.exact(-xlogy(qs[f], qs[f]).sum(axis=(-2, -1)))
            for f, name in zip(index, names, strict=True)
        }

    return step


def _ensemble_disagreement(st: registry.Statics) -> registry.Step:
    """Ensemble information gain over the next code, by moment-matched Monte Carlo.

    An ensemble of :class:`MLPConditional` heads predicts the next flattened code
    from ``[deter, flat stoch, action]`` of the step's inputs (``qs_prev`` and the
    action); the leaf is the per-dimension mean of ``ent(mixture) − E_i[ent(σ_i)]``,
    scaled ``1/sqrt(num_vars)``. Raises ``ValueError`` at prepare time unless
    exactly one code variable is selected, the scoring entry is a non-empty tuple
    of ``MLPConditional`` heads, the System declares exactly one carry node, and
    every member's ``parent_dims``/``child_dim`` match ``([deter, flat code,
    action], flat code)``. Design: docs/design/amortization.md chunk 2d §(b)
    """
    if len(st.index) != 1:
        raise ValueError(
            "the ensemble_disagreement row scores ONE code variable (the scoring slot holds "
            f"one ensemble), but epistemic selects {[st.factor_names[f] for f in st.index]}. "
            "Name a single variable."
        )
    (code,) = st.index
    name = st.factor_names[code]
    members = st.scoring
    if not isinstance(members, tuple) or not members:
        raise ValueError(
            f"scoring['ensemble_disagreement'] must be a tuple of MLPConditional heads, got "
            f"{type(members).__name__}"
        )
    bad = [i for i, m in enumerate(members) if not isinstance(m, MLPConditional)]
    if bad:
        raise ValueError(
            f"ensemble_disagreement member(s) {bad} are not MLPConditional heads; the ensemble "
            "is Gaussian heads over the next code"
        )
    structure = st.structure
    carries = [f for f, fam in enumerate(structure.families) if fam == _amortized.CARRY]
    if len(carries) != 1:
        raise ValueError(
            f"the ensemble reads [deter, flat stoch, action] and so needs exactly one carry "
            f"node; this System declares {[st.factor_names[f] for f in carries]}"
        )
    (carry,) = carries
    flat = structure.widths[code]
    wanted = (
        structure.widths[carry],
        flat,
        _action_width(st.controlled, st.num_controls, st.controls),
    )
    wrong = {i: m.parent_dims for i, m in enumerate(members) if m.parent_dims != wanted}
    if wrong:
        raise ValueError(
            f"ensemble_disagreement member(s) {wrong} declare those parent widths but the row "
            f"supplies {wanted} — [carry, flat {name}, action]"
        )
    wrong_child = {i: m.child_dim for i, m in enumerate(members) if m.child_dim != flat}
    if wrong_child:
        raise ValueError(
            f"ensemble_disagreement member(s) {wrong_child} predict that many numbers but the "
            f"target is the next {name} code, flattened: {flat}"
        )
    num_vars = st.B[code].num_vars
    scale = 1.0 / jnp.sqrt(float(num_vars))
    controlled, num_controls, controls = st.controlled, st.num_controls, st.controls

    def step(ctx: registry.StepContext) -> dict[str, Interval]:
        _needs_prev(ctx.qs_prev)
        #  The leaf is read at (s_k, a_k), the transition step k actually took: at index
        #  k every row is a functional of the same event — taking a_k from s_k and
        #  arriving at s_{k+1} — which is what makes the fold to G coherent.
        parents = (
            _amortized._flat(ctx.qs_prev[carry]),
            _amortized._flat(ctx.qs_prev[code]),
            _action_vector(ctx.action, controlled, num_controls, controls),
        )
        heads = [m.diag_moments(parents) for m in members]
        means = jnp.stack([mu for mu, _ in heads])
        scales = jnp.stack([sd for _, sd in heads])
        #  Moment-matched mutual information between the next code and the member
        #  index. The mixture over members has Var[x] = E_i[sigma_i^2] + Var_i[mu_i]
        #  (law of total variance; ``var`` is population, exact because the members
        #  ARE the mixture's components). The leaf is the per-dimension mean of
        #  ent(mixture) - mean_i ent(member) — ``_diag_entropy`` is the joint entropy,
        #  hence the division by ``flat`` — scaled by 1/sqrt(num_vars). Gaussian
        #  moment matching makes it an upper bound on the true mutual information,
        #  and it is >= 0 by AM-GM (tests/test_infogain_contract.py). Stored as the
        #  gain itself; the term it lands under fixes the direction (infogain -1,
        #  model_uncertainty +1).
        per_dim = 1.0 / float(flat)
        mixture_scale = jnp.sqrt(jnp.mean(scales**2, axis=0) + means.var(axis=0))
        disagreement = per_dim * (_diag_entropy(mixture_scale) - _diag_entropy(scales).mean(axis=0))
        return {name: Interval.exact(scale * disagreement)}

    return step


def _predicted_reward(st: registry.Statics) -> registry.Step:
    """The declared scalar head's mean under the rolled-out beliefs.

    Reads a declared :class:`PredictorHead` with a one-number target, decoded by
    its own ``predict``. Raises ``ValueError`` at prepare time when the named
    modality's declaration is anything else.
    """
    (m,) = st.index
    head, deps, name = st.A[m], st.A_deps[m], st.modality_names[m]
    if not isinstance(head, PredictorHead) or head.target_dim != 1:
        raise ValueError(
            f"terms={st.target!r} names modality {name!r}, whose declaration is a "
            f"{type(head).__name__}"
            + (f" predicting {head.target_dim} numbers" if isinstance(head, PredictorHead) else "")
            + ". The row decodes a scalar reward prediction: a PredictorHead whose target is one "
            "number (symlog_disc, or any scalar loss)."
        )

    def step(ctx: registry.StepContext) -> dict[str, Interval]:
        return {name: Interval.exact(head.predict(_amortized.head_parents(ctx.qs, deps)))}

    return step


def _g_tail(st: registry.Statics) -> registry.Step:
    """The learned value tail: the critic closes the horizon after the explicit steps.

    ``scoring['g_tail']`` holds ``(critic, slow)`` or the critic alone; only the
    critic is read. It is a one-number :class:`PredictorHead` over the final
    step's :func:`features`, whose mean is a value (negative EFE), stored under a
    −1 term. Raises ``ValueError`` at prepare time when the critic is not such a
    head or its ``parent_dims`` is not the feature width.
    """
    entry = st.scoring
    critic = entry[0] if isinstance(entry, tuple) else entry
    if not isinstance(critic, PredictorHead) or critic.target_dim != 1:
        raise ValueError(
            "scoring['g_tail'] holds (critic, slow) — or the critic alone — where the critic is "
            "a PredictorHead predicting one number over [carry, flat code], got "
            f"{type(critic).__name__}"
        )
    width = int(sum(st.structure.widths))
    if critic.parent_dims != (width,):
        raise ValueError(
            f"the critic declares parent_dims={critic.parent_dims} but reads the whole feature "
            f"vector, {width} numbers wide ([deter, flat stoch], concatenated)"
        )

    def step(ctx: registry.StepContext) -> dict[str, Interval]:
        return {CRITIC_BLOCK: Interval.exact(critic.predict((features(ctx.qs),)))}

    return step


def _needs_prev(qs_prev) -> None:
    if qs_prev is None:
        raise ValueError(
            "the ensemble estimator (infogain / model_uncertainty) scores the step's inputs, so "
            "it needs the previous step's beliefs; this closure must be driven through the "
            "(qs, qs_prev) planning carry System.planner_context builds"
        )


# ---------------------------------------------------------------------- the rows

OCCUPANCY = registry.register_row(
    registry.Estimator(
        "occupancy",
        registry.VARIABLE,
        _occupancy,
        term="occupancy",
        target=registry.SLOT_TARGET,
        families=(_amortized.CODE,),
        derived=False,
    ),
    slot="transition",
)
PREDICTED_REWARD = registry.register_row(
    registry.Estimator(
        "predicted_reward",
        registry.CHANNEL,
        _predicted_reward,
        term="predicted_reward",
        target=registry.OBS_TARGET,
        families=(_amortized.CODE,),
        derived=False,
    )
)
G_TAIL = registry.register_row(
    registry.Estimator(
        "g_tail",
        registry.SINGLE,
        _g_tail,
        term="g_tail",
        families=(_amortized.CODE,),
        parametric=True,
        derived=False,
        terminal=True,
        trainer="train.imagine.imagine",
        parameters=("critic", "slow"),
        opt_state=lambda entry, declared: init_opt_state(
            entry[0] if isinstance(entry, tuple) else entry
        ),
    )
)

#: The ``infogain`` estimator for a factored-categorical code, selected by
#: ``epistemic=("stoch",)``: curiosity, entering G with sign −1.
registry.ESTIMATORS[(registry.STATE, registry.MLP_CATEGORICAL)] = registry.Estimator(
    "ensemble_disagreement",
    registry.VARIABLE,
    _ensemble_disagreement,
    parametric=True,
    derived=False,
    reads_prev=True,
    trainer="train.ensemble.fit_ensemble",
    opt_state=lambda entry, declared: init_opt_state(entry),
    parameters=("ensemble",),
)

MODEL_UNCERTAINTY = registry.register_row(
    registry.Estimator(
        "model_uncertainty",
        registry.VARIABLE,
        _ensemble_disagreement,
        term="model_uncertainty",
        target=registry.STATE_TARGET,
        families=(_amortized.CODE,),
        parametric=True,
        derived=False,
        reads_prev=True,
        trainer="train.ensemble.fit_ensemble",
        opt_state=lambda entry, declared: init_opt_state(entry),
        parameters=("ensemble",),
    )
)
#: Pessimism: the same estimator under a +1 term, so ensemble disagreement is a cost.
#: Each mounted ensemble row owns its own ``scoring`` entry (:func:`check_ensemble_ownership`).
#: Design: objective-terms.md "Every shipped row, placed".


# -------------------------------------------------------------------- the builders


#: One resolved ``terms=`` entry (defined in the registry).
Mount = registry.Mount


def make_transition(
    B: tuple,
    B_deps: tuple[tuple[int, ...], ...],
    families: tuple[str, ...],
    controlled: tuple[bool, ...],
    num_controls: tuple[int, ...],
    controls: object | None,
) -> Callable[[tuple, Array], tuple]:
    """One rollout step for the planner seam, as a closure.

    Runs the same :func:`hybrid_aif.inference.amortized.predict` as
    ``update_empirical_prior``.
    """

    def transition(beliefs: tuple, action: Array) -> tuple:
        return _amortized.predict(
            B, B_deps, families, controlled, num_controls, controls, tuple(beliefs), action
        )

    return transition


def make_step_terms(
    mounts: tuple[Mount, ...], epistemic_codes: tuple[int, ...], statics: registry.Statics
) -> Callable[..., TermTree] | None:
    """The per-step term closure: every mounted non-terminal row, merged.

    ``mounts`` are the resolved ``terms=`` rows; ``epistemic_codes`` the code
    variables ``epistemic`` put an ``infogain`` (``ensemble_disagreement``) leaf on.
    Returns ``None`` when nothing is mounted. The closure takes ``qs_prev`` (see
    :func:`needs_prev`) and ``qs_dist``, the distributions a sampled belief carry
    came from.
    """
    check_ensemble_ownership(mounts, epistemic_codes, statics.scoring)
    steps = [
        (
            registry.get_row(m.name).term,
            registry.prepare_row(m.name, registry.mount_statics(m, statics), m.family),
        )
        for m in mounts
        if not registry.get_row(m.name).terminal
    ]
    if epistemic_codes:
        steps.append(
            (
                INFOGAIN,
                registry.prepare(
                    registry.STATE,
                    registry.MLP_CATEGORICAL,
                    statics._replace(
                        index=epistemic_codes,
                        scoring=registry._require(statics.scoring, "ensemble_disagreement"),
                    ),
                ),
            )
        )
    if not steps:
        return None

    def step_terms(
        qs: tuple,
        action: Array,
        t: Array,
        qs_prev: tuple | None = None,
        qs_dist: tuple | None = None,
    ) -> TermTree:
        ctx = registry.StepContext(qs=qs, action=action, qs_prev=qs_prev, t=t, qs_dist=qs_dist)
        return merge(*({term: fn(ctx)} for term, fn in steps))

    return step_terms


def make_terminal_terms(
    mounts: tuple[Mount, ...], statics: registry.Statics
) -> Callable[[tuple], TermTree] | None:
    """The horizon-end closure: the ``g_tail`` critic, when it is mounted."""
    steps = [
        (
            registry.get_row(m.name).term,
            registry.prepare_row(m.name, registry.mount_statics(m, statics), m.family),
        )
        for m in mounts
        if registry.get_row(m.name).terminal
    ]
    if not steps:
        return None

    def terminal_terms(qs: tuple) -> TermTree:
        ctx = registry.StepContext(qs=qs, action=None)
        return merge(*({term: fn(ctx)} for term, fn in steps))

    return terminal_terms


def no_step_terms(
    qs: tuple,
    action: Array,
    t: Array,
    qs_prev: tuple | None = None,
    qs_dist: tuple | None = None,
) -> TermTree:
    """A per-step closure with no leaves: the horizon-end term is the whole of G."""
    return {}


#: The two rows that read an ensemble of :class:`MLPConditional` heads over the next
#: code — the same estimator under opposite signs. ``ensemble_disagreement`` is reached
#: by ``epistemic=`` (it is an ``infogain`` estimator), ``model_uncertainty`` by
#: ``terms=``; both keep their parameters in ``SystemParams.scoring`` under their own
#: name, and :mod:`hybrid_aif.train.ensemble` trains either.
ENSEMBLE_ROWS: tuple[str, ...] = ("ensemble_disagreement", "model_uncertainty")


def check_ensemble_ownership(
    mounts: tuple[Mount, ...], epistemic_codes: tuple[int, ...], scoring: Mapping[str, object]
) -> None:
    """Each mounted ensemble row owns its own nets.

    Raises when two mounted ensemble rows share a tuple of nets or any member:
    the scoring slot is a pytree, so apparent sharing would fork into independent
    copies at the first optimizer step. Separate nets per row are legal.
    """
    mounted = [m.name for m in mounts if m.name in ENSEMBLE_ROWS]
    if epistemic_codes:
        mounted.append("ensemble_disagreement")
    present = [name for name in dict.fromkeys(mounted) if name in scoring]
    if len(present) < 2:
        return
    owners: dict[int, str] = {}
    for name in present:
        entry = scoring[name]
        members = entry if isinstance(entry, tuple) else (entry,)
        for obj in (entry, *members):
            owner = owners.setdefault(id(obj), name)
            if owner != name:
                raise ValueError(
                    f"the {owner!r} and {name!r} rows are both mounted and hold the *same "
                    "module object(s)*. They share an estimator, never parameters: the scoring "
                    "slot is a pytree, so flattening duplicates the leaves and the first "
                    "optimizer step forks them into two independent nets that then drift apart "
                    "silently (design: amortization.md A11 refinement (ii), the same rule the "
                    "recognition slot enforces). Give each mounted row its own ensemble — "
                    "structurally identical copies are fine, they are genuinely separate "
                    "parameters — or mount only one of the two. The pair exists so that one "
                    "estimator can enter G as curiosity (infogain, sign -1) or as pessimism "
                    "(model_uncertainty, sign +1), not so that one ensemble can do both."
                )
    return


def needs_prev(mounts: tuple[Mount, ...], epistemic_codes: tuple[int, ...]) -> bool:
    """Whether the mounted rows need the ``(qs, qs_prev)`` planning carry.

    True when the ensemble estimator is mounted, under either ``infogain`` or
    ``model_uncertainty``.
    """
    return bool(epistemic_codes) or any(
        registry.row_variant(m.name, m.family).reads_prev for m in mounts
    )


def scoring_opt_state(name: str, entry: object, declared: object = None):
    """The optimizer carry for one ``SystemParams.scoring`` entry, by the row's ``opt_state``.

    An optax state for a parameter tree, a ``LearnState`` for an owned model; raises
    for a row with no trainer.
    """
    row = registry.by_name(name)
    if row is None or row.opt_state is None:
        raise KeyError(
            f"no optimizer-carry shape is declared for scoring row {name!r}; set opt_state on "
            "its Estimator beside the trainer that steps it (design: amortization.md A6)"
        )
    return row.opt_state(entry, declared)


#: ``LearnState.moments`` keys for the ``g_tail`` trainer's running statistics
#: (:func:`hybrid_aif.train.imagine.imagine`): the return normalizer's ``(low, high)``
#: percentile EMAs, and the slow critic's update counter.
RETNORM_KEY = "imagine.retnorm"
SLOW_KEY = "imagine.slow_updates"


def scoring_moments(name: str) -> dict[str, Array]:
    """The fresh ``LearnState.moments`` entries one scoring row's trainer carries.

    Created up front so a trainer fills its entries and never adds one, keeping
    the carry's pytree structure fixed across calls. Only ``g_tail`` has any
    (:data:`RETNORM_KEY`, :data:`SLOW_KEY`); every other row returns ``{}``.
    """
    if name == "g_tail":
        return {RETNORM_KEY: jnp.zeros((2,), jnp.float32), SLOW_KEY: jnp.zeros((), jnp.int32)}
    return {}
