"""The inference faces of :class:`~hybrid_aif.system.System`: ``infer_states``, ``predict_beliefs``
and ``update_empirical_prior``."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.distributions import Gaussian, MixtureGaussian, MLPConditional
from hybrid_aif.factors import LGObservation, TabularObservation
from hybrid_aif.inference import amortized as _amortized
from hybrid_aif.inference import emission as _emission
from hybrid_aif.inference import sample as _sample
from hybrid_aif.inference.detail import InferenceDetail
from hybrid_aif.inference.discrete import SUPPORT_FLOOR, fpi_support, mean_field_free_energy
from hybrid_aif.inference.linear_gaussian import condition
from hybrid_aif.inference.mixture_filter import (
    condition_mixture_components,
    condition_mixture_switching,
    mode_bank_evidence,
    mode_evidence_message,
    switching_mode_message,
)
from hybrid_aif.objectives import continuous as _continuous
from hybrid_aif.objectives import hybrid as _hybrid
from hybrid_aif.objectives.discrete import expected_states
from hybrid_aif.system import keyed, validation
from hybrid_aif.system.core import Belief, SystemParams


def _present_select(present, updated: list, previous: list | None, log_ev):
    """Keep ``updated`` where a modality is present, ``previous`` where it is masked out.

    ``present`` is ``True`` (static: nothing to select, ``previous`` is ``None``) or a
    traced 0-d bool; a static ``False`` never reaches here (the caller skips the
    modality). Only the entries the update touched are selected; the masked-out
    modality's ``log_evidence`` reads 0.0.
    """
    if present is True:
        return updated, log_ev
    kept = [
        u if u is p else _hybrid.where_tree(present, u, p)
        for u, p in zip(updated, previous, strict=True)
    ]
    return kept, jnp.where(present, log_ev, jnp.zeros_like(log_ev))


def infer_states(
    system,
    params: SystemParams,
    observations: Mapping[str, Array | None],
    priors: Mapping[str, Belief] | None = None,
    detail: bool = False,
    externals: Mapping[str, Array] | None = None,
    present: Mapping[str, bool | Array] | None = None,
    support_floor: float = SUPPORT_FLOOR,
    key: Array | None = None,
    previous: Mapping[str, Belief | Array] | None = None,
) -> tuple[dict[str, Belief], InferenceDetail | None]:
    params = system._params_in(params, "infer_states")
    names = system._names(params)
    positional_priors = (
        None if priors is None else system._beliefs_in(params, priors, "infer_states")
    )
    prev_beliefs, prev_action = _previous_in(system, previous, "infer_states")
    sensed = None
    if system._sensed_columns:
        sensed = system._sensed_priors(params, priors, "infer_states")
        if positional_priors is None:
            positional_priors = system._initial_priors(params, "infer_states")
        positional_priors = (*positional_priors, *sensed)
    observations_in = keyed.observations_in(
        names, system._with_lagged(observations), "infer_states"
    )
    validation._check_observations(names.modalities, params, observations_in, "infer_states")
    beliefs, info = system._infer_states(
        params,
        observations_in,
        positional_priors,
        detail,
        externals,
        None if present is None else keyed.present_in(names, present, "infer_states"),
        support_floor,
        key,
        "infer_states",
        previous=(prev_beliefs, prev_action),
    )
    posteriors = None
    if sensed is not None:
        s = system._table.num_states
        posteriors = dict(zip(system._sensed, beliefs[s:], strict=True))
        beliefs = beliefs[:s]
    return system._beliefs_out(params, beliefs, priors, posteriors), (
        None if info is None else keyed.detail_out(names, info)
    )


def _infer_states(
    system,
    params: SystemParams,
    observations: tuple[Array | None, ...],
    priors: tuple[Belief, ...] | None = None,
    detail: bool = False,
    externals: Mapping[str, Array] | None = None,
    present: tuple[bool | Array, ...] | Array | None = None,
    support_floor: float = SUPPORT_FLOOR,
    key: Array | None = None,
    face: str = "this face",
    previous: tuple | None = None,
) -> tuple[tuple[Belief, ...], InferenceDetail | None]:
    """:meth:`infer_states` over positional tuples; ``present`` holds one flag per modality,
    or is a ``(M,)`` bool array; ``previous`` is ``(prev_beliefs, prev_action, first)``,
    the carry the lagged modalities read (``first`` a traced flag for a tick with no
    carry)."""
    params = system._p(params)
    if system.lagged:
        prev_beliefs, prev_action, first = (
            (None, None, False) if previous is None else (*previous, False)[:3]
        )
        observations, present = system._fill_lagged(
            observations,
            system._present(present, len(params.likelihood)),
            prev_beliefs,
            prev_action,
            first,
        )
    beliefs, info = _infer_states_expected(
        system, params, observations, priors, detail, externals, present, support_floor, face
    )
    if key is None:
        return beliefs, info
    return _collapse_posterior(system, params, beliefs, key, face), info


def _collapse_posterior(
    system, params: SystemParams, beliefs: tuple[Belief, ...], key: Array, face: str = "this face"
) -> tuple[Belief, ...]:
    """One draw per factor, as a belief of its family; family-aware on the amortized path."""
    params = system._p(params)
    if system._is_amortized(params):
        return _amortized.collapse_codes(system._amortized_structure(params, face), beliefs, key)
    return _sample.collapse_beliefs(beliefs, key)


def _infer_states_expected(
    system,
    params: SystemParams,
    observations: tuple[Array, ...],
    priors: tuple[Belief, ...] | None = None,
    detail: bool = False,
    externals: Mapping[str, Array] | None = None,
    present: tuple[bool | Array, ...] | Array | None = None,
    support_floor: float = SUPPORT_FLOOR,
    face: str = "this face",
) -> tuple[tuple[Belief, ...], InferenceDetail | None]:
    """The mean-field posterior: :meth:`infer_states` without the collapse."""
    params = system._p(params)
    masks = system._present(present, len(params.likelihood))
    if system._is_amortized(params):
        system._refuse_sensed(face)
        system._validate(params, face)
        structure = system._amortized_structure(params, face)
        if len(observations) != len(params.likelihood):
            raise ValueError(
                f"got {len(observations)} observations for {len(params.likelihood)} "
                "modalities (pass None for a modality no recognition module reads)"
            )
        priors = tuple(system._initial_priors(params, face) if priors is None else priors)
        if masks is not None:
            #  the modalities the modules read are present together (update) or absent
            #  together (predict-only); others may be masked freely
            read = _amortized.read_modalities(structure.plan)
            flags = [masks[m] for m in read]
            if flags and all(f is False for f in flags):
                return priors, None
            if any(f is False for f in flags) and not all(f is False for f in flags):
                raise ValueError(
                    "infer_states(present=...) on the amortized path: the recognition "
                    f"modules read modalities {read} as one parent tuple, so they are present "
                    "together or absent together; these flags keep some and drop others"
                )
            #  traced flags cannot be refused here: they reduce with ``all``, so a split
            #  reads as absent (``replay`` refuses it on the concrete flags)
            traced = [f for f in flags if f is not True]
            together = None if not traced else jnp.all(jnp.stack(traced))
        else:
            together = None
        beliefs = _amortized.recognize(
            structure.plan,
            params.recognition,
            priors,
            tuple(observations),
            externals,
            system._names().modalities,
        )
        if together is not None:  # traced flags: all present, or a predict-only tick
            beliefs = tuple(_hybrid.where_tree(together, beliefs, priors))
        return beliefs, None
    n_mods = len(params.likelihood)
    pre_reduce: list[MixtureGaussian | None] = [None] * n_mods
    mode_log_ev: list[Array | None] = [None] * n_mods
    log_ev: list[Array | None] = [None] * n_mods
    flags: list[Array | None] = [None] * len(params.transition)
    free_energy: list[Array | None] = [None]
    # each LG modality's log evidence, the term the tick's free energy subtracts
    evidence_terms: list[Array | None] = [None] * n_mods

    def _detail() -> InferenceDetail | None:
        if not detail:
            return None
        total = jnp.zeros(()) if free_energy[0] is None else free_energy[0]
        for ev in evidence_terms:
            if ev is not None:
                total = total - ev
        return InferenceDetail(
            pre_reduce=tuple(pre_reduce),
            mode_log_evidence=tuple(mode_log_ev),
            log_evidence=tuple(log_ev),
            unsupported=tuple(flags),
            free_energy=total,
        )

    if not system._is_hybrid(params):
        A_deps = system.A_deps
        if system._sensed_columns:
            #  an observed action node joins the sweep as one more categorical variable,
            #  after the state factors; its prior is the caller's (the agent's prediction)
            n, s = len(system.B_deps), system._table.num_states
            at = {n + k: s + i for i, k in enumerate(system._sensed_columns)}
            A_deps = tuple(tuple(at.get(g, g) for g in deps) for deps in A_deps)
        priors_used = system._initial_priors(params, face) if priors is None else priors
        beliefs, sub_flags = fpi_support(
            params.likelihood,
            A_deps,
            priors_used,
            observations,
            num_iters=system.num_iters,
            masks=masks,
            support_floor=support_floor,
        )
        flags[:] = list(sub_flags)[: len(flags)]
        if detail:
            free_energy[0] = mean_field_free_energy(
                params.likelihood, A_deps, priors_used, beliefs, observations, masks
            )
        return beliefs, _detail()
    s, n_sensed = system._table.num_states, len(system._sensed_columns)
    if n_sensed and (priors is None or len(priors) != s + n_sensed):
        system._refuse_sensed(face)  # only infer_states carries the observed nodes' priors
    A_deps = system._engine_A_deps
    families = system._families(params)
    mods, facs = (system.modality_names, system._facs)
    for m, a in enumerate(params.likelihood):
        if isinstance(a, LGObservation):
            continue
        if isinstance(a, MLPConditional):
            raise NotImplementedError(
                f"modality {mods[m]!r} declares a neural (MLPConditional) likelihood: "
                "conditioning state beliefs *through* a net needs an inference strategy "
                "the registry does not have: structured/moment-matched filtering "
                "through the net (design: amortization.md A1(a)) or an amortized "
                "recognition model q(s | o, s_prev) (A1(b), the P3 track). The "
                "declaration itself is legal and gradient-learnable through learn(); "
                "only infer_states through it is unbuilt."
            )
        continuous = [facs[f] for f in A_deps[m] if f < s and families[f] != _hybrid.CATEGORICAL]
        if continuous:
            raise NotImplementedError(
                f"modality {mods[m]!r} observes continuous factor(s) {continuous} with a "
                f"non-linear-Gaussian likelihood ({type(a).__name__}): discrete-given-"
                "continuous / MNLR observation models are out of scope"
            )
    system._validate(params, face)
    priors = system._initial_priors(params, face) if priors is None else priors
    out: list[Belief] = list(priors)
    cat_f = (
        *(f for f, fam in enumerate(families) if fam == _hybrid.CATEGORICAL),
        *range(s, s + n_sensed),
    )
    cat_m = tuple(m for m, a in enumerate(params.likelihood) if isinstance(a, TabularObservation))
    # a Gaussian emission over categorical parents alone joins the tabular sweep as the
    # one-row tensor of its predictive scores (structure-learning.md S2); its scores are
    # densities, so the support check skips it
    emission_m = tuple(m for m, a in enumerate(params.likelihood) if _emission.is_emission(a))
    if cat_f:
        pos = {f: i for i, f in enumerate(cat_f)}
        sub_m = cat_m + emission_m
        sub_lls: list = [params.likelihood[m] for m in cat_m]
        sub_obs: list = [observations[m] for m in cat_m]
        for m in emission_m:
            shape = tuple(priors[f].probs.shape[-1] for f in A_deps[m])
            if masks is not None and masks[m] is False:
                sub_lls.append(jnp.zeros((1, *shape)))
            else:
                sub_lls.append(
                    _emission.emission_log_likelihood(params.likelihood[m], observations[m], shape)
                )
            sub_obs.append(jnp.zeros((), jnp.int32))
        sub_deps = tuple(tuple(pos[f] for f in A_deps[m]) for m in sub_m)
        sub_masks = None if masks is None else tuple(masks[m] for m in sub_m)
        sub_posteriors, sub_flags = fpi_support(
            tuple(sub_lls),
            sub_deps,
            tuple(priors[f] for f in cat_f),
            tuple(sub_obs),
            num_iters=system.num_iters,
            masks=sub_masks,
            support_floor=support_floor,
            support_exempt=tuple(range(len(cat_m), len(sub_m))),
        )
        for f, q, flag in zip(cat_f, sub_posteriors, sub_flags, strict=True):
            out[f] = q
            if f < s:
                flags[f] = flag
        if detail:  # the tabular stage's free energy, before any LG mode message
            free_energy[0] = mean_field_free_energy(
                tuple(sub_lls),
                sub_deps,
                tuple(priors[f] for f in cat_f),
                sub_posteriors,
                tuple(sub_obs),
                sub_masks,
            )
    # a noise map is read at the tick's prior mean, whichever modality over the factor
    # conditions first (the extended filter's convention)
    prior_locs = {f: q.loc for f, q in enumerate(out) if isinstance(q, Gaussian)}
    for m, a in enumerate(params.likelihood):
        if not isinstance(a, LGObservation) or m in emission_m:
            continue  # categorical modalities and emissions handled by the FPI sub-model
        present = True if masks is None else masks[m]
        deps = A_deps[m]
        f, obs_parents = deps[0], deps[1:]
        if obs_parents and families[f] == _hybrid.GAUSSIAN and all(g >= s for g in obs_parents):
            #  a channel read through action nodes alone selects its head and noise at the
            #  node's level, the node's prior being a point; the factor stays one Gaussian
            a = _hybrid.select_observation(
                _continuous.lg_point(a),
                tuple(
                    _hybrid.prior_point_level(priors[g], system._sensed[g - s], mods[m], facs[f])
                    for g in obs_parents
                ),
            )
            obs_parents = ()
        before = None
        if present is not True:
            before = list(out)
            if obs_parents:
                #  the switching update pads the factor to the static budget, so its
                #  absence must leave the factor at the SAME shape: the prior, padded
                before[f] = _hybrid.pad_mixture(before[f], system.mixture_k)
            if present is False:
                out = before  # statically absent: the factors keep their priors
                continue
        if obs_parents:
            # Switching observation: grow the factor to K*M, read the mode message from
            # the grown weights before reducing (so no mode-owned slots are needed;
            # mixture-filtering.md Guidance)
            q = out[f]
            mode_lp = _hybrid._mode_log_weights(tuple(out[g] for g in obs_parents))
            grown, per_mode_ev = condition_mixture_switching(
                q, a.H, a.c, a.R_chol, observations[m], mode_lp
            )
            updated = switching_mode_message(tuple(out[g] for g in obs_parents), per_mode_ev)
            for g, q_mode in zip(obs_parents, updated, strict=True):
                out[g] = q_mode
            pre_reduce[m], mode_log_ev[m] = grown, per_mode_ev
            log_ev[m] = jax.scipy.special.logsumexp(mode_lp + per_mode_ev)
            out[f] = _hybrid.pad_mixture(grown, system.mixture_k)
            out, log_ev[m] = _present_select(present, out, before, log_ev[m])
            evidence_terms[m] = log_ev[m]
            continue
        if families[f] != _hybrid.MIXTURE:
            R_chol = a.R_chol_at(prior_locs[f]) if a.noise_map else a.R_chol
            out[f], log_ev[m] = condition(out[f], a.H, a.c, R_chol, observations[m])
            out, log_ev[m] = _present_select(present, out, before, log_ev[m])
            evidence_terms[m] = log_ev[m]
            continue
        if system.mixture_schedule != "mode_aligned":
            raise NotImplementedError(
                f"modality {mods[m]!r} observes mixture-carried factor {facs[f]!r} under "
                "mixture_schedule='runnalls': filtering a mixture belief against evidence "
                "needs mode-owned slots (the per-mode evidence a cross-mode Runnalls merge "
                'destroys). Declare System(mixture_schedule="mode_aligned") to enable it.'
            )
        q = out[f]
        posterior, log_ev[m], ll = condition_mixture_components(
            q, a.H, a.c, a.R_chol, observations[m]
        )
        mode_parents = tuple(g for g in system._B[f] if families[g] == _hybrid.CATEGORICAL)
        if mode_parents:
            mode_beliefs = tuple(out[g] for g in mode_parents)
            # the evidence of y reads the mode belief the tabular stage left, as the
            # switching path does, not the mixture's own bank masses, which still carry
            # the prior mode mass
            n_modes = 1
            for b in mode_beliefs:
                n_modes *= b.probs.shape[-1]
            log_ev[m] = jax.scipy.special.logsumexp(
                _hybrid._mode_log_weights(mode_beliefs)
                + mode_bank_evidence(q.log_weights, ll, n_modes)
            )
            updated = mode_evidence_message(mode_beliefs, q.log_weights, ll)
            for g, q_mode in zip(mode_parents, updated, strict=True):
                out[g] = q_mode
            mode_log_ev[m] = ll
        out[f] = posterior
        out, log_ev[m] = _present_select(present, out, before, log_ev[m])
        evidence_terms[m] = log_ev[m]
    return tuple(out), _detail()


def _previous_in(
    system, previous: Mapping[str, Belief | Array] | None, face: str
) -> tuple[tuple[Belief | None, ...] | None, Array | None]:
    """A keyed ``previous`` as ``(prev_beliefs, prev_action)``: one belief per state factor
    (``None`` where absent) and the positional action (``None`` without any entry)."""
    if previous is None:
        return None, None
    facs = system._facs[: len(system.B_deps)]
    actions = tuple(system.action_names or ())
    unknown = sorted(set(previous) - set(facs) - set(actions))
    if unknown:
        raise ValueError(
            f"{face}: previous names {unknown}, which are neither state factors nor action nodes"
        )
    beliefs = tuple(previous.get(name) for name in facs)
    action = None
    if any(name in previous for name in actions):
        action = jnp.asarray(
            [jnp.asarray(previous.get(name, 0), jnp.int32) for name in actions], jnp.int32
        )
    return beliefs, action


def _fill_lagged(
    system,
    observations: tuple[Array | None, ...],
    present: tuple[bool | Array, ...] | None,
    prev_beliefs: tuple[Belief | None, ...] | None,
    prev_action: Array | None,
    first: Array | bool = False,
) -> tuple[tuple[Array | None, ...], tuple[bool | Array, ...] | None]:
    """The observations with every lagged modality's datum read from the carry where none
    was given: the source factor's previous belief mean, or the previous action's level.
    A modality with neither is absent, as is every lagged modality when ``first`` (a traced
    flag for the tick with no carry)."""
    sources = system._lagged_sources
    if not sources:
        return observations, present
    obs = list(observations)
    flags = list(present) if present is not None else [True] * len(obs)
    for m, (kind, index) in sources.items():
        if obs[m] is not None:
            continue
        if kind == "factor":
            q = None if prev_beliefs is None else prev_beliefs[index]
            datum = None if q is None else q.mean()
        else:
            datum = None if prev_action is None else jnp.asarray(prev_action[index])
        if datum is None:
            obs[m], flags[m] = None, False
            continue
        obs[m] = datum
        if first is not False:
            flag = flags[m]
            flags[m] = (
                jnp.logical_not(first)
                if flag is True
                else jnp.logical_and(jnp.asarray(flag, bool), jnp.logical_not(first))
            )
    return tuple(obs), tuple(flags)


def _present(
    present: tuple[bool | Array, ...] | Array | None, n_mods: int
) -> tuple[bool | Array, ...] | None:
    """``present`` as one flag per modality: Python bools static, anything else a 0-d
    bool."""
    if present is None:
        return None
    if isinstance(present, tuple | list):
        flags = tuple(present)
    else:
        present = jnp.asarray(present)
        if present.shape != (n_mods,):
            raise ValueError(
                f"present has shape {present.shape} but this System has {n_mods} modalities; "
                f"pass one flag per modality, shape ({n_mods},)"
            )
        flags = tuple(present[m] for m in range(n_mods))
    if len(flags) != n_mods:
        raise ValueError(f"present has {len(flags)} entries for {n_mods} modalities")
    return tuple(f if isinstance(f, bool) else jnp.asarray(f, bool) for f in flags)


def predict_beliefs(
    system,
    params: SystemParams,
    beliefs: Mapping[str, Belief],
    policy: Array,
    key: Array | None = None,
) -> list[dict[str, Belief]]:
    params = system._params_in(params, "predict_beliefs")
    steps = _predict_beliefs(
        system,
        params,
        system._beliefs_in(params, beliefs, "predict_beliefs"),
        policy,
        key,
        "predict_beliefs",
    )
    plan = jnp.asarray(policy)
    return [
        system._beliefs_out(
            params, b, sensed=system._sensed_predictions(params, plan[t], "predict_beliefs")
        )
        for t, b in enumerate(steps)
    ]


def _predict_beliefs(
    system,
    params: SystemParams,
    beliefs: tuple[Belief, ...],
    policy: Array,
    key: Array | None = None,
    face: str = "this face",
) -> list[tuple[Belief, ...]]:
    """:meth:`predict_beliefs` over the positional belief tuple."""
    params = system._p(params)
    policy = system._check_plan(policy, "policy", params)
    if policy.ndim != 2:
        raise ValueError(
            f"policy must be a single (horizon, ·) plan, got {policy.shape}: one column per "
            "control node"
        )
    horizon = policy.shape[0]
    if horizon == 0:
        return []
    keys = None if key is None else jax.random.split(key, horizon)
    # The first step runs unrolled: it brings the carry to its canonical
    # form (mixture factors padded to ``mixture_k``), which the scan needs fixed.
    first = system._update_empirical_prior(
        params, beliefs, policy[0], key=None if keys is None else keys[0], face=face
    )
    if horizon == 1:
        return [first]

    def step(carry, xs):
        action, k_t = xs
        pushed = system._update_empirical_prior(params, carry, action, key=k_t, face=face)
        return pushed, pushed

    _, rest = jax.lax.scan(step, first, (policy[1:], None if keys is None else keys[1:]))

    def at(t: int) -> tuple[Belief, ...]:
        return jax.tree_util.tree_map(lambda a: a[t], rest)

    return [first, *(at(t) for t in range(horizon - 1))]


def update_empirical_prior(
    system,
    params: SystemParams,
    beliefs: Mapping[str, Belief],
    action: Mapping[str, Array],
    key: Array | None = None,
) -> dict[str, Belief]:
    params = system._params_in(params, "update_empirical_prior")
    names = system._names(params)
    positional = keyed.actions_in(names, action, "update_empirical_prior")
    pushed = system._update_empirical_prior(
        params,
        system._beliefs_in(params, beliefs, "update_empirical_prior"),
        positional,
        key=key,
        face="update_empirical_prior",
    )
    return system._beliefs_out(
        params, pushed, sensed=system._sensed_predictions(params, positional)
    )


def _update_empirical_prior(
    system,
    params: SystemParams,
    beliefs: tuple[Belief, ...],
    action: Array | None,
    key: Array | None = None,
    face: str = "this face",
) -> tuple[Belief, ...]:
    """:meth:`update_empirical_prior` over the positional tuple and the positional action
    (one level per control node, the flat float vector of an all-continuous space, or a
    mixed space's :class:`~hybrid_aif.controls.ControlStep`; ``None`` for a rollout with no
    action parent)."""
    params = system._p(params)
    if system._is_amortized(params):
        system._refuse_sensed(face)
        system._validate(params, face)
        structure = system._amortized_structure(params, face)
        return _amortized.predict(
            params.transition,
            system._B,
            structure.families,
            system._ctl(face),
            system._num_controls(params, face),
            system._control_layout(params),
            tuple(beliefs),
            action,
            key,
        )
    if not system._is_hybrid(params):
        validation._validate_table_arity(system, params, system._facs)
        if params.control or not system._table.default_wiring:
            system._validate(
                params, face
            )  # a box on a table's node raises; shared-control tables checked
        pushed = expected_states(beliefs, params.transition, action, system._B, system._table)
        return pushed if key is None else _sample.collapse_beliefs(pushed, key)
    system._validate(params, face)
    families = system._families(params)
    transition = _hybrid.make_transition(
        params.transition,
        system._B,
        families,
        system.mixture_k,
        system.mixture_schedule,
        table=system._table,
        space=system._space(params),
    )
    pushed = transition(_hybrid.prepare_beliefs(beliefs, families, system.mixture_k), action)
    return pushed if key is None else _sample.collapse_beliefs(pushed, key)
