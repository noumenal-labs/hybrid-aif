"""The learning faces of :class:`~hybrid_aif.system.System`: ``learn``, ``carry``, ``reduce``,
``grow`` and ``init_learn_state``."""

import dataclasses
from collections.abc import Iterable, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hybrid_aif.cells import cell_mass
from hybrid_aif.controls import ControlSpace
from hybrid_aif.distributions import ConditionalDirichlet, Learnable, MLPConditional
from hybrid_aif.factors import (
    GaussianTransition,
    LinearGaussianObservation,
    LinearGaussianObservationMNIW,
)
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
from hybrid_aif.system.faces import infer as _infer


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
    priors: Mapping[str, Belief] | None = None,
    externals: Mapping[str, Array] | None = None,
) -> "SystemParams | tuple[SystemParams, LearnState]":
    params = system._params_in(params, "learn")
    names = system._names(params)
    observations_in = keyed.observations_in(names, observations, "learn")
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
        priors=None if priors is None else system._beliefs_in(params, priors, "learn"),
        externals=externals,
        present=_infer._absent_unsupplied(system, observations_in, None),
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
    priors: tuple[Belief, ...] | None = None,
    externals: Mapping[str, Array] | None = None,
) -> "SystemParams | tuple[SystemParams, LearnState]":
    """:meth:`learn` over positional tuples and the positional action. ``present`` is one
    flag per modality: a modality absent at the tick teaches its likelihood slot nothing
    (a static ``False`` skips it, a traced flag scales its rate). ``grow`` runs the growth
    step and the reduction schedule; the batch form of growth calls back with it off. ``priors``
    is the tuple of prior beliefs the tick's inference read (see :func:`learn`)."""
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
    reads = {(slot, i): home for slot, i, home in params.references}
    # the readers of a noise variable, by the field that holds it (process-noise-learning.md D4)
    noise_reads = {(slot, i): home for slot, i, home in params.noise_references}
    homes = dict(params.parameters)  # the named tables, updated once from their readers' evidence
    home_stats: dict[str, Array] = {}  # a parameter variable's summed readers' statistic
    lr_rates = system._resolve_rates(lr_by_slot, "lr_by_slot", params)
    _refuse_recognition_rates(params, lr, lr_rates)
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

    def contribute(home: str, stats: Array, weight: Array | None = None) -> None:
        """Add one reader's statistic (scaled by its availability, when traced) to its variable's
        sum."""
        scaled = stats if weight is None else jax.tree_util.tree_map(lambda s: s * weight, stats)
        home_stats[home] = (
            scaled
            if home not in home_stats
            else jax.tree_util.tree_map(jnp.add, home_stats[home], scaled)
        )

    def home_rate_of(home: str) -> float | Array:
        return rate(f"parameter:{home}", lr, lr_rates)

    def excluded(slot: str, name: str) -> bool:
        """A referencing reader whose own rate key is a static 0.0: it adds no statistic."""
        return lr_rates is not None and skipped(lr_rates.get(f"{slot}:{name}", 1.0))

    def noise_excluded(slot: str, name: str) -> bool:
        """A reader of a noise variable whose own noise key (or group key) is a static 0.0."""
        return lr_rates is not None and (
            skipped(lr_rates.get(f"{slot}:{name}:noise", 1.0))
            or skipped(lr_rates.get(f"{slot}:{name}", 1.0))
        )

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
            home = noise_reads.get(("likelihood", m))
            if home is not None:  # one reader of a noise variable: its statistic joins the sum
                if skipped(home_rate_of(home)) or noise_excluded("likelihood", mods[m]):
                    continue
                stats = a.noise.stats(
                    *_lg_noise_stats(system, m, a, observations[m], beliefs, detail)
                )
                flag = True if present is None else present[m]
                contribute(home, stats, None if flag is True else jnp.asarray(flag, jnp.float32))
                continue
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
        home = reads.get(("likelihood", m))
        r = rate(key, lr, lr_rates) if home is None else home_rate_of(home)
        if skipped(r) or (home is not None and excluded("likelihood", mods[m])):
            continue
        parents = tuple(beliefs[f] for f in system.A_deps[m])
        stats = a.stats(observations[m], parents)
        if home is not None:
            flag = True if present is None else present[m]
            contribute(home, stats, None if flag is True else jnp.asarray(flag, jnp.float32))
            continue
        new_A[m] = _absorbing(key, a, stats, lr=observed(m, r), decay=rate(key, decay, decay_rates))

    new_B = list(params.transition)
    active = [
        (f, r)
        for f, b in enumerate(params.transition)
        if isinstance(b, Learnable)
        and not skipped(
            r := (
                home_rate_of(reads[("transition", f)])
                if ("transition", f) in reads
                else rate(f"transition:{facs[f]}", lr, lr_rates)
            )
        )
        and not (("transition", f) in reads and excluded("transition", facs[f]))
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
        cells = system._cell_parents()
        parents = [
            cell_mass(prev_beliefs[g], cells[g]) if g in cells else prev_beliefs[g] for g in deps
        ]
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
        if ("transition", f) in reads:
            contribute(reads[("transition", f)], stats)
            continue
        new_B[f] = _absorbing(key, b, stats, lr=r, decay=rate(key, decay, decay_rates))

    # ------------------------------------------- the process noise (process-noise-learning.md)
    noisy = []
    for f, b in enumerate(params.transition):
        if not (isinstance(b, GaussianTransition) and b.learnable_noise):
            continue
        key = f"transition:{facs[f]}"
        home = noise_reads.get(("transition", f))
        if home is None:
            r = rate(key, lr, lr_rates, "noise")
        else:
            r = home_rate_of(home)
            if noise_excluded("transition", facs[f]):
                continue
        if not skipped(r):
            noisy.append((f, b, key, home, r))
    if noisy:
        if prev_beliefs is None:
            raise ValueError(
                f"the process noise of {[facs[f] for f, *_ in noisy]} has a nonzero-or-traced "
                "rate, so learn needs prev_beliefs (the previous tick's posteriors from the "
                "carry): the statistic is the residual between two ticks' states; pass them, or "
                "freeze the noise with a static rate of 0.0 ('transition:<factor>:noise')"
            )
        need = [facs[f] for f, *_ in noisy if system._table.control_parents[f]]
        if need and action is None:
            raise ValueError(
                f"the process noise of controlled factor(s) {need} has a nonzero-or-traced rate, "
                "so learn needs action (a_{t-1}, the action that produced this tick, from the "
                "carry): it selects the mode and the control input of the residual"
            )
        space = system._space(params)
    for f, b, key, home, r in noisy:
        stats = b.noise_prior.stats(
            *conjugate_stats._transition_noise_stats(
                system, params, f, b, prev_beliefs[f], beliefs[f], action, space
            )
        )
        if home is not None:
            contribute(home, stats)
            continue
        new_B[f] = dataclasses.replace(
            new_B[f],
            noise_prior=_absorbing(
                f"{key}:noise",
                b.noise_prior,
                stats,
                lr=r,
                decay=rate(key, decay, decay_rates, "noise"),
            ),
        )

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
    tick_priors = None
    if any(slot == "likelihood" for _, slot, _, _, _ in grad_active):
        tick_priors = _tick_priors(system, params, priors, prev_beliefs, action, face)
    for key, slot, i, decl, r in grad_active:
        if slot == "likelihood":
            stats = _likelihood_stats(
                system, params, i, decl, observations, tick_priors, present, externals, face
            )
        else:
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
            stats = decl.stats(child, parents)
        updated, opt[key] = decl.absorb(
            stats,
            opt[key],
            lr=r,
            decay=rate(key, decay, decay_rates),
        )
        (new_A if slot == "likelihood" else new_B)[i] = updated

    for home, stats in home_stats.items():
        key = f"parameter:{home}"
        homes[home] = _absorbing(
            key, homes[home], stats, lr=home_rate_of(home), decay=rate(key, decay, decay_rates)
        )
    for (slot, i), home in reads.items():  # every reader's position holds the updated variable
        (new_A if slot == "likelihood" else new_B)[i] = homes[home]
    for (slot, i), home in noise_reads.items():  # and every noise reader's field
        target = new_A if slot == "likelihood" else new_B
        field = keyed.noise_field(slot, target[i])
        target[i] = dataclasses.replace(target[i], **{field: homes[home]})
    new_params = dataclasses.replace(
        params,
        likelihood=tuple(new_A),
        transition=tuple(new_B),
        initial_state_prior=tuple(new_D),
        parameters=homes,
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
    state: str | None = None,
    n_pair: int | None = None,
) -> SystemParams:
    params = system._params_in(params, "reduce")
    prior = system._params_in(prior, "reduce")
    system._validate(params, "reduce")
    growable = _structure.growable_factors(system, params)
    if state is not None:
        if state not in system._facs or system._facs.index(state) not in growable:
            raise ValueError(
                f"reduce: {state!r} is not a growable state factor; the growable state factors "
                f"are {[system._facs[f] for f in growable]}"
            )
        growable = (system._facs.index(state),)
    pairs = system.reduction_pairs if n_pair is None else int(n_pair)
    for f in growable:
        params = _structure.reduce(system, params, prior, f, pairs)
    return system._params_out(params)


def grow(
    system,
    params: SystemParams,
    observations: Mapping[str, Array],
    state: str,
    rounds: int = 1,
    lr: float | Array = 1.0,
) -> SystemParams:
    params = system._params_in(params, "grow")
    names = system._names(params)
    system._validate(params, "grow")
    growable = _structure.growable_factors(system, params)
    if state not in system._facs or system._facs.index(state) not in growable:
        raise ValueError(
            f"grow: {state!r} is not a growable state factor; the growable state factors are "
            f"{[system._facs[f] for f in growable]}"
        )
    f = system._facs.index(state)
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
            f"grow: state factor {state!r} is read by modalities {missing}, whose windows are "
            "missing from observations"
        )
    return system._params_out(_structure.grow_batch(system, params, obs, f, int(rounds), lr))


def _refuse_recognition_rates(params: SystemParams, lr, resolved) -> None:
    """Refuse a nonzero resolved rate on a recognition module: on the analytic path it is frozen.

    ``resolved`` is the ``lr_by_slot`` mapping after the specificity ladder
    (``System._resolve_rates``), so a rate reaches a module however its key is spelled; a module
    no key reaches takes the base ``lr``. An explicit zero (a Python number or a concrete
    array) is accepted, and so is a frozen module; a traced rate is refused.
    """

    def nonzero(r) -> bool:
        if isinstance(r, jax.core.Tracer):
            raise ValueError(
                "a recognition module's rate is traced, so it cannot be checked as zero; "
                "give the recognition slot a static zero (lr_by_slot={'recognition:<name>': 0.0})"
            )
        return bool(np.any(np.asarray(r) != 0.0))

    mapping = {} if resolved is None else resolved
    for variable, module in params.recognition.items():
        key = f"recognition:{variable}"
        addressed = [k for k in mapping if k == key or k.startswith(key + ":")]
        rates = [mapping[k] for k in addressed]
        if not addressed and getattr(module, "learnable", False):
            rates = [lr]
        if any(nonzero(r) for r in rates):
            raise ValueError(
                f"{key}: a recognition module is frozen on the analytic path, so its rate must "
                f"be zero (lr_by_slot={{{key!r}: 0.0}}, or freeze the module); training it "
                "means minimising the free energy through the sweep "
                "(recognition-beside-analytic.md R5)"
            )


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
        if slot != "scoring"
        and not (slot == "recognition" and not system._is_amortized(params))
        and _gradient_learnable(decl)
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


def _tick_priors(
    system,
    params: SystemParams,
    priors: tuple[Belief, ...] | None,
    prev_beliefs: tuple[Belief, ...] | None,
    action: Array | None,
    face: str,
) -> tuple[Belief, ...]:
    """The prior beliefs the tick's inference read: ``priors`` as given, else the prediction from
    the previous tick's beliefs and action when both are present, else the initial priors."""
    if priors is not None:
        return priors
    if prev_beliefs is not None:
        return _infer._update_empirical_prior(system, params, prev_beliefs, action, face=face)
    return tuple(system._initial_priors(params, face))


def _likelihood_stats(
    system,
    params: SystemParams,
    m: int,
    decl: MLPConditional,
    observations: tuple[Array, ...],
    priors: tuple[Belief, ...],
    present: tuple[bool | Array, ...] | None,
    externals: Mapping[str, Array] | None,
    face: str,
) -> MLPConditional:
    """The gradient, in the declaration's parameters, of the free energy ``infer_states`` reports
    for the tick: the inference pass itself, run from the tick's ``priors`` with every other
    input held fixed, differentiated with respect to modality ``m``'s decoder. One objective
    serves inference and learning (design: recognition-beside-analytic.md L5).
    """

    def free_energy(d: MLPConditional) -> Array:
        likelihood = tuple(d if i == m else a for i, a in enumerate(params.likelihood))
        _, info = _infer._infer_states_expected(
            system,
            dataclasses.replace(params, likelihood=likelihood),
            observations,
            priors,
            True,
            externals,
            present,
            face=face,
        )
        return info.free_energy

    return eqx.filter_grad(free_energy)(decl)


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
