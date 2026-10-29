"""The learning faces of :class:`~hybrid_aif.system.System`: ``learn``, ``carry``, ``reduce``,
``grow`` and ``init_learn_state``."""

import dataclasses
from collections.abc import Iterable, Mapping

import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.controls import ControlSpace
from hybrid_aif.distributions import ConditionalDirichlet, Learnable, MLPConditional
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianObservationMNIW
from hybrid_aif.inference.detail import InferenceDetail
from hybrid_aif.learning import carry as _carry
from hybrid_aif.learning import conjugate_stats
from hybrid_aif.learning import structure as _structure
from hybrid_aif.learning.slots import gradient_learnable as _gradient_learnable
from hybrid_aif.objectives import amortized as _amortized_terms
from hybrid_aif.objectives import registry
from hybrid_aif.system import keyed, validation
from hybrid_aif.system.core import (
    ACTOR_KEY,
    REDUCTION_KEY,
    Belief,
    LearnState,
    SystemParams,
)


def _absorbing(key: str, decl, stats, lr, decay):
    """``decl.absorb`` with the slot's key prefixed to any refusal it raises."""
    try:
        return decl.absorb(stats, lr=lr, decay=decay)
    except ValueError as exc:
        raise ValueError(f"{key}: {exc}") from None


def learn(
    system,
    params: SystemParams,
    observations: Mapping[str, Array],
    beliefs: Mapping[str, Belief],
    prev_beliefs: Mapping[str, Belief] | None = None,
    action: Mapping[str, Array] | None = None,
    lr: float | Array = 1.0,
    decay: float | Array = 1.0,
    lr_by_slot: Mapping[str, float | Array] | None = None,
    decay_by_slot: Mapping[str, float | Array] | None = None,
    detail: InferenceDetail | None = None,
    learn_state: "LearnState | None" = None,
    prior: SystemParams | None = None,
) -> "SystemParams | tuple[SystemParams, LearnState]":
    params = system._params_in(params, "learn")
    names = system._names(params)
    observations_in = keyed.observations_in(names, system._with_lagged(observations), "learn")
    validation._check_observations(names.modalities, params, observations_in, "learn")
    out = system._learn(
        params,
        observations_in,
        system._beliefs_in(params, beliefs, "learn"),
        None if prev_beliefs is None else system._beliefs_in(params, prev_beliefs, "learn"),
        None if action is None else keyed.actions_in(names, action, "learn"),
        lr=lr,
        decay=decay,
        lr_by_slot=lr_by_slot,
        decay_by_slot=decay_by_slot,
        detail=None if detail is None else keyed.detail_in(names, detail, "learn"),
        learn_state=learn_state,
        face="learn",
        prior=None if prior is None else system._params_in(prior, "learn"),
    )
    if isinstance(out, tuple):
        return system._params_out(out[0]), out[1]
    return system._params_out(out)


def _learn(
    system,
    params: SystemParams,
    observations: tuple[Array, ...],
    beliefs: tuple[Belief, ...],
    prev_beliefs: tuple[Belief, ...] | None = None,
    action: Array | None = None,
    lr: float | Array = 1.0,
    decay: float | Array = 1.0,
    lr_by_slot: Mapping[str, float | Array] | None = None,
    decay_by_slot: Mapping[str, float | Array] | None = None,
    detail: InferenceDetail | None = None,
    learn_state: "LearnState | None" = None,
    face: str = "this face",
    present: tuple[bool | Array, ...] | None = None,
    prior: SystemParams | None = None,
    grow: bool = True,
) -> "SystemParams | tuple[SystemParams, LearnState]":
    """:meth:`learn` over positional tuples and the positional action. ``present`` is one
    flag per modality: a modality absent at the tick teaches its likelihood slot nothing
    (a static ``False`` skips it, a traced flag scales its rate). ``grow`` runs the growth
    step and the reduction schedule; the batch form of growth calls back with it off."""
    system._refuse_sensed(face)
    params = system._p(params)
    prior = None if prior is None else system._p(prior)
    if system._is_amortized(params):
        system._amortized_structure(params, face)
        raise NotImplementedError(
            "System.learn on an amortized (neural state-space) System: its objective is one "
            "sequence loss over every slot at once — reconstruction + reward + continuation "
            "NLLs and the balanced posterior/prior KL, backpropagated through the recurrent "
            "carry (amortization-p0-addendum.md §3) — not a per-tick sufficient statistic "
            "this face could absorb slot by slot. Train it with "
            "hybrid_aif.train.world.fit_world(system, params, learn_state, batch), whose "
            "LearnState is keyed by these same slot addresses (design: amortization.md "
            "A2/A6/A11)."
        )
    system._validate(params, face)
    mods, facs = (system.modality_names, system._facs)
    lr_rates = system._resolve_rates(lr_by_slot, "lr_by_slot", params)
    decay_rates = system._resolve_rates(decay_by_slot, "decay_by_slot", params)
    if len(observations) != len(params.likelihood):
        raise ValueError(
            f"got {len(observations)} observations for {len(params.likelihood)} modalities"
        )

    def rate(key: str, base, resolved, block: str | None = None) -> float | Array:
        # _resolve_rates has applied the specificity ladder; the group key is the base
        # for any block without its own address
        if resolved is None:
            return base
        if block is not None and f"{key}:{block}" in resolved:
            return resolved[f"{key}:{block}"]
        return resolved.get(key, base)

    def skipped(r) -> bool:
        # a static Python 0.0 skips the slot at trace time, decay included
        return isinstance(r, int | float) and float(r) == 0.0

    def observed(m: int, r):
        """The rate under the tick's availability: a traced flag scales it to zero."""
        flag = True if present is None else present[m]
        return r if flag is True else r * jnp.asarray(flag, jnp.float32)

    if system.lagged:
        observations, present = system._fill_lagged(observations, present, prev_beliefs, action)
    # structure moves with the weights: a growable factor whose weights' rate is a static
    # zero neither grows nor reduces, and a traced rate gates both at run time
    gates: dict[int, bool | Array] = {}
    for f in _structure.growable_factors(system, params) if grow else ():
        r = rate(f"transition:{facs[f]}", lr, lr_rates)
        if skipped(r):
            continue
        gates[f] = True if isinstance(r, int | float) else jnp.asarray(r) > 0.0
    growable = tuple(gates)
    growth = None
    if growable:
        beliefs, growth = _structure.grow_beliefs(
            system, params, observations, beliefs, present, gates
        )

    new_A = list(params.likelihood)
    for m, a in enumerate(params.likelihood):
        key = f"likelihood:{mods[m]}"
        if present is not None and present[m] is False:
            continue  # statically absent: the slot keeps its statistics
        if isinstance(a, LinearGaussianObservationMNIW):
            r = rate(key, lr, lr_rates)
            if skipped(r):
                continue
            syy, syx, sxx, count = _lg_joint_stats(system, m, a, observations[m], beliefs, detail)
            new_A[m] = a.absorb(
                a.stats(syy, syx, sxx, count),
                lr=observed(m, r),
                decay=rate(key, decay, decay_rates),
            )
            continue
        if isinstance(a, LinearGaussianObservation):
            if not (a.learnable_noise or a.learnable_head):
                continue
            # two independent posteriors: ``…:noise``/``…:head`` rate them separately
            r_head = rate(key, lr, lr_rates, "head")
            if a.learnable_head and not skipped(r_head):
                raise NotImplementedError(
                    f"modality {mods[m]!r}: learning an independent MatrixNormal head "
                    "is not built (its shared-V CAVI update under non-spherical noise "
                    "is a genuine approximation) — use LinearGaussianObservationMNIW, "
                    "the coupled family, or freeze the slot with a static rate of 0.0 "
                    f"({key}:head alone freezes the head and leaves the noise learning; "
                    "design parameter-learning.md D8)"
                )
            r_noise = rate(key, lr, lr_rates, "noise")
            if not a.learnable_noise or skipped(r_noise):
                continue
            scatter, count = _lg_noise_stats(system, m, a, observations[m], beliefs, detail)
            new_A[m] = dataclasses.replace(
                a,
                noise=a.noise.absorb(
                    a.noise.stats(scatter, count),
                    lr=observed(m, r_noise),
                    decay=rate(key, decay, decay_rates, "noise"),
                ),
            )
            continue
        if not isinstance(a, Learnable):
            continue
        r = rate(key, lr, lr_rates)
        if skipped(r):
            continue
        parents = tuple(beliefs[f] for f in system.A_deps[m])
        stats = a.stats(observations[m], parents)
        new_A[m] = _absorbing(key, a, stats, lr=observed(m, r), decay=rate(key, decay, decay_rates))

    new_B = list(params.transition)
    active = [
        (f, r)
        for f, b in enumerate(params.transition)
        if isinstance(b, Learnable)
        and not skipped(r := rate(f"transition:{facs[f]}", lr, lr_rates))
    ]
    if active:
        if prev_beliefs is None:
            raise ValueError(
                f"learnable transition(s) {[facs[f] for f, _ in active]} have a "
                "nonzero-or-traced rate, so learn needs prev_beliefs (the previous "
                "tick's posteriors from the carry); pass them, or freeze the slot(s) "
                "with a static rate of 0.0"
            )
        controlled_active = [f for f, _ in active if system._table.control_parents[f]]
        if action is None and controlled_active:
            raise ValueError(
                f"learnable controlled transition(s) "
                f"{[facs[f] for f in controlled_active]} have a nonzero-or-traced "
                "rate, so learn needs action (a_{t-1}, the action that produced "
                "this tick, from the carry)"
            )
    for f, r in active:
        b = params.transition[f]
        deps = system._B[f]
        parents = [prev_beliefs[g] for g in deps]
        control_parents = system._table.control_parents[f]
        if control_parents:
            #  one one-hot per control axis, at its parent position: counts land on the
            #  taken slice
            for k, pos in control_parents:
                parents.insert(
                    pos,
                    jax.nn.one_hot(action[k], b.parent_shape[pos], dtype=b.concentration.dtype),
                )
        elif b.parent_ndims == len(deps) + 1:  # the size-1 axis of a factor with no control
            parents.append(jax.nn.one_hot(0, b.parent_shape[-1], dtype=b.concentration.dtype))
        elif b.parent_ndims != len(deps):
            raise ValueError(
                f"factor {facs[f]!r}: transition declaration has parent_ndims="
                f"{b.parent_ndims} but B_deps names {len(deps)} parent(s) "
                "(+1 for a trailing control axis)"
            )
        key = f"transition:{facs[f]}"
        stats = b.stats(beliefs[f].probs, tuple(parents))
        new_B[f] = _absorbing(key, b, stats, lr=r, decay=rate(key, decay, decay_rates))

    new_D = list(params.initial_state_prior)
    for f, d in enumerate(params.initial_state_prior):
        if not isinstance(d, Learnable):
            continue
        key = f"initial_state_prior:{facs[f]}"
        r = rate(key, lr, lr_rates)
        if skipped(r):
            continue
        if isinstance(d, ConditionalDirichlet):
            raise NotImplementedError(
                f"factor {facs[f]!r}: updating a hierarchical (ConditionalDirichlet) "
                "initial prior needs the parent-weighted t=0 statistic, which is not "
                "built (design doc D5, build-plan step 4); freeze it with "
                f"lr_by_slot={{{key!r}: 0.0}}"
            )
        d_rate = rate(key, decay, decay_rates)
        new_D[f] = _absorbing(key, d, d.stats(beliefs[f].probs), lr=r, decay=d_rate)
    for f in growable:
        # a woken level's weight is the prior's pseudo-count for a new level plus its
        # datum, and the initial prior holds it live as the weights do
        new_B[f] = _structure.wake_weights(new_B[f], growth, f)
        new_D[f] = _structure.mirror_initial_prior(new_D[f], new_B[f], growth, f)

    # ------------------------------------------------------- the gradient slots
    grad_slots = system._gradient_slots(params)
    grad_active = [
        (key, slot, i, decl, r)
        for key, slot, i, decl in grad_slots
        if not skipped(r := rate(key, lr, lr_rates))
    ]
    grad_trans = [i for _, slot, i, _, _ in grad_active if slot == "transition"]
    if grad_trans:
        if prev_beliefs is None:
            raise ValueError(
                f"gradient-learnable transition(s) {[facs[i] for i in grad_trans]} have a "
                "nonzero-or-traced rate, so learn needs prev_beliefs (the previous tick's "
                "posteriors from the carry — the net is fit on (s_{t-1}, a_{t-1}) -> s_t); "
                "pass them, or freeze the slot(s) with a static rate of 0.0"
            )
        need_action = [facs[i] for i in grad_trans if system._table.control_parents[i]]
        if need_action and action is None:
            raise ValueError(
                f"gradient-learnable controlled transition(s) {need_action} have a "
                "nonzero-or-traced rate, so learn needs action (a_{t-1}, the action that "
                "produced this tick, from the carry — it is the net's one-hot parent)"
            )
    opt = system._resolve_learn_state(learn_state, grad_slots)
    for key, slot, i, decl, r in grad_active:
        child, parents = _mlp_evidence(
            system,
            slot,
            i,
            decl,
            observations,
            beliefs,
            prev_beliefs,
            action,
            system._space(params),
        )
        updated, opt[key] = decl.absorb(
            decl.stats(child, parents),
            opt[key],
            lr=r,
            decay=rate(key, decay, decay_rates),
        )
        (new_A if slot == "likelihood" else new_B)[i] = updated

    new_params = dataclasses.replace(
        params,
        likelihood=tuple(new_A),
        transition=tuple(new_B),
        initial_state_prior=tuple(new_D),
    )
    moments = {} if learn_state is None else dict(learn_state.moments)
    if growable and system.reduction_period is not None:
        # the reduction schedule: every reduction_period-th call reduces each growable
        # factor against the prior the posterior was learned from (structure-learning.md S4)
        if learn_state is None or REDUCTION_KEY not in moments:
            raise ValueError(
                f"reduction_period={system.reduction_period} schedules Bayesian model "
                "reduction inside learn, whose counter rides learn_state: pass "
                "learn_state=system.init_learn_state(params) on the first call and thread "
                "the returned LearnState through"
            )
        if prior is None:
            raise ValueError(
                "a scheduled reduction scores the merged levels' evidence against the prior "
                "the posterior was learned from; pass prior= (the params learning started "
                "from) to learn"
            )
        calls = moments[REDUCTION_KEY] + 1
        fire = jnp.mod(calls, int(system.reduction_period)) == 0
        for f in growable:
            new_params = jax.lax.cond(
                fire & jnp.asarray(gates[f]),
                lambda p, f=f: _structure.reduce(system, p, prior, f, system.reduction_pairs),
                lambda p: p,
                new_params,
            )
        moments[REDUCTION_KEY] = calls
    if not grad_slots and learn_state is None:
        return new_params  # conjugate-only model: the bare-params return
    #  `moments` is a cross-trainer carry; pass it through (see `train/world.py`)
    return new_params, LearnState(opt_states=opt, moments=moments)


def carry(
    system,
    fresh: SystemParams,
    learned: SystemParams,
    slots: Iterable[str] | None = None,
) -> SystemParams:
    fresh_p = system._params_in(fresh, "carry")
    learned_p = system._params_in(learned, "carry")
    growable = _structure.growable_factors(system, fresh_p)
    return system._params_out(
        _carry.carry_learned(system, fresh_p, learned_p, slots, growable=growable)
    )


def reduce(
    system,
    params: SystemParams,
    prior: SystemParams,
    factor: str | None = None,
    n_pair: int | None = None,
) -> SystemParams:
    params = system._params_in(params, "reduce")
    prior = system._params_in(prior, "reduce")
    system._validate(params, "reduce")
    growable = _structure.growable_factors(system, params)
    if factor is not None:
        if factor not in system._facs or system._facs.index(factor) not in growable:
            raise ValueError(
                f"reduce: {factor!r} is not a growable factor; the growable factors are "
                f"{[system._facs[f] for f in growable]}"
            )
        growable = (system._facs.index(factor),)
    pairs = system.reduction_pairs if n_pair is None else int(n_pair)
    for f in growable:
        params = _structure.reduce(system, params, prior, f, pairs)
    return system._params_out(params)


def grow(
    system,
    params: SystemParams,
    observations: Mapping[str, Array],
    factor: str,
    rounds: int = 1,
    lr: float | Array = 1.0,
) -> SystemParams:
    params = system._params_in(params, "grow")
    names = system._names(params)
    system._validate(params, "grow")
    growable = _structure.growable_factors(system, params)
    if factor not in system._facs or system._facs.index(factor) not in growable:
        raise ValueError(
            f"grow: {factor!r} is not a growable factor; the growable factors are "
            f"{[system._facs[f] for f in growable]}"
        )
    f = system._facs.index(factor)
    mods = tuple(m for m, deps in enumerate(system.A_deps) if deps == (f,))
    given = dict(observations)
    unknown = sorted(set(given) - set(system.modality_names))
    if unknown:
        raise ValueError(f"grow: unknown modality name(s) {unknown}")
    obs = tuple(
        given.get(names.modalities[m]) if m in mods else None for m in range(len(params.likelihood))
    )
    missing = [names.modalities[m] for m in mods if obs[m] is None]
    if missing:
        raise ValueError(
            f"grow: factor {factor!r} is read by modalities {missing}, whose windows are "
            "missing from observations"
        )
    return system._params_out(_structure.grow_batch(system, params, obs, f, int(rounds), lr))


def _gradient_slots(
    system, params: SystemParams
) -> tuple[tuple[str, str, int, MLPConditional], ...]:
    """The slots that learn by gradient, as ``(key, slot, index, decl)``, in declaration order.

    Their presence makes :meth:`learn` return a :class:`LearnState`. The ``scoring`` slot is
    excluded: its rows are stepped by the objective-side trainers.
    """
    params = system._p(params)
    return tuple(
        (key, slot, i, decl)
        for key, (slot, i, decl) in system._slot_declarations(params).items()
        if slot != "scoring" and _gradient_learnable(decl)
    )


def init_learn_state(system, params: SystemParams) -> LearnState:
    params = system._params_in(params, "init_learn_state")
    opt_states = {key: decl.init_opt_state() for key, _, _, decl in system._gradient_slots(params)}
    moments: dict[str, object] = {}
    for name, entry in params.scoring.items():
        if registry.scoring_trainer(name) is None:
            continue
        opt_states[f"scoring:{name}"] = _amortized_terms.scoring_opt_state(
            name, entry, dict(system.scoring).get(name)
        )
        moments.update(_amortized_terms.scoring_moments(name))
    if system._habit_module(params) is not None:
        opt_states[ACTOR_KEY] = params.policy_prior.init_opt_state()
    if system.reduction_period is not None and _structure.growable_factors(system, params):
        moments[REDUCTION_KEY] = jnp.zeros((), jnp.int32)
    return LearnState(opt_states=opt_states, moments=moments)


def _resolve_learn_state(
    learn_state: "LearnState | None", grad_slots: tuple[tuple[str, str, int, object], ...]
) -> dict[str, object]:
    """The optimizer-state map for this call, checked against the model's gradient slots.

    Every gradient slot has an entry regardless of rate, so the carry's structure depends on the
    declarations alone. ``None`` with gradient slots present, or a carry missing a slot's key,
    raises rather than re-initializing.
    """
    expected = {key for key, _, _, _ in grad_slots}
    if not expected:
        return {}
    if learn_state is None:
        raise ValueError(
            f"this model has gradient-learnable slot(s) {sorted(expected)}, so learn "
            "needs their optimizer state: pass learn_state=system.init_learn_state("
            "params) on the first call and thread the returned LearnState through the "
            "carry thereafter. (None does not silently re-initialize — a dropped carry "
            "must fail at the call that would have restarted the optimizer; design: "
            "amortization.md A2.)"
        )
    missing = sorted(expected - set(learn_state.opt_states))
    if missing:
        raise ValueError(
            f"learn_state carries optimizer state for "
            f"{sorted(learn_state.opt_states)} but this model's gradient-learnable "
            f"slots include {missing}; the carry is stale (or belongs to a different "
            "model). Call system.init_learn_state(params) for a fresh one (design: "
            "amortization.md A2)."
        )
    return dict(learn_state.opt_states)


def _belief_value(q: Belief) -> Array:
    return conjugate_stats._belief_value(q)


def _mlp_evidence(
    system,
    slot: str,
    i: int,
    decl: MLPConditional,
    observations: tuple[Array, ...],
    beliefs: tuple[Belief, ...],
    prev_beliefs: tuple[Belief, ...] | None,
    action: Array | None,
    space: ControlSpace | None = None,
) -> tuple[Array, tuple[Array, ...]]:
    return conjugate_stats._mlp_evidence(
        system, slot, i, decl, observations, beliefs, prev_beliefs, action, space
    )


def _lg_noise_stats(
    system,
    m: int,
    a: LinearGaussianObservation,
    y: Array,
    beliefs: tuple[Belief, ...],
    detail: InferenceDetail | None,
) -> tuple[Array, Array]:
    return conjugate_stats._lg_noise_stats(system, m, a, y, beliefs, detail)


def _lg_joint_stats(
    system,
    m: int,
    a: "LinearGaussianObservationMNIW",
    y: Array,
    beliefs: tuple[Belief, ...],
    detail: InferenceDetail | None,
) -> tuple[Array, Array, Array, Array]:
    return conjugate_stats._lg_joint_stats(system, m, a, y, beliefs, detail)
