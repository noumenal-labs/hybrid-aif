"""The inference faces of :class:`~hybrid_aif.system.System`: ``infer_states``, ``predict_beliefs``
and ``update_empirical_prior``."""

import math
from collections.abc import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.controls import LEVEL
from hybrid_aif.distributions import Categorical, MixtureGaussian, MLPConditional
from hybrid_aif.factors import LGObservation, TabularObservation
from hybrid_aif.inference import amortized as _amortized
from hybrid_aif.inference import emission as _emission
from hybrid_aif.inference import sample as _sample
from hybrid_aif.inference.detail import InferenceDetail
from hybrid_aif.inference.discrete import SUPPORT_FLOOR, fpi_support, mean_field_free_energy
from hybrid_aif.inference.linear_gaussian import condition
from hybrid_aif.inference.linearise import linearise
from hybrid_aif.inference.mixture_filter import (
    _per_mode,
    condition_mixture_components,
    condition_mixture_switching_components,
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


def _measurements(
    system,
    params: SystemParams,
    priors: tuple[Belief, ...],
    observations: tuple[Array | None, ...],
    externals: Mapping[str, Array] | None,
    masks: tuple[bool | Array, ...] | None,
    categorical: tuple[bool, ...],
) -> tuple[_amortized.Measurement, ...]:
    """The tick's recognition messages, from one :func:`~hybrid_aif.inference.amortized.
    analytic_recognize` pass over the entries in derived order (before any update).

    ``categorical[f]`` says whether state factor ``f`` is categorical; ``priors`` give each
    factor's width.
    """
    if not system.recognition:
        return ()
    s = len(categorical)
    kinds = tuple(
        _amortized.CATEGORICAL_KIND if cat else _amortized.GAUSSIAN_KIND for cat in categorical
    )
    widths = tuple(
        q.probs.shape[-1] if isinstance(q, Categorical) else q.mean().shape[-1] for q in priors[:s]
    )
    return _amortized.analytic_recognize(
        _amortized.analytic_order(system.recognition),
        params.recognition,
        observations,
        externals,
        masks,
        system.modality_names,
        system._facs[:s],
        kinds,
        widths,
    )


def _append_rows(sub_lls, sub_deps, sub_obs, sub_masks, exempt, rows, pos):
    """The tabular sweep's inputs extended by each categorical measurement as a one-row tensor
    over its factor (``pos`` maps a factor to its place in the sweep), support-exempt like an
    emission's scores: the row is a likelihood over the levels, not a probability of an event."""
    if not rows:
        return sub_lls, sub_deps, sub_obs, sub_masks, exempt
    n = len(sub_lls)
    masks = (True,) * n if sub_masks is None else tuple(sub_masks)
    return (
        (*sub_lls, *(r.row[None] for r in rows)),
        (*sub_deps, *((pos[r.factor],) for r in rows)),
        (*sub_obs, *(jnp.zeros((), jnp.int32) for _ in rows)),
        (*masks, *(r.present for r in rows)),
        (*exempt, *range(n, n + len(rows))),
    )


def _categorical_terms(rows, pre, final, pos, terms: list, log_z_by_name: dict) -> None:
    """Record each categorical measurement's ``- log Z + E_q[log m]`` (``terms``) and ``log Z``.

    ``log Z = log sum_k q_pre_k r_k`` reads the belief the sweep gives without the rows (``pre``),
    ``E_q[log m] = sum_k q_k log r_k`` the final belief; a measurement absent at the tick adds
    nothing.
    """
    for r in rows:
        log_z = _amortized.categorical_message_log_evidence(pre[pos[r.factor]].log_probs, r.row)
        term = -log_z + _amortized.categorical_message_expectation(final[r.factor].probs, r.row)
        if r.present is not True:
            log_z, term = (jnp.where(r.present, x, jnp.zeros_like(x)) for x in (log_z, term))
        terms.append(term)
        log_z_by_name[f"recognition:{r.variable}"] = log_z


def _neural_continuous(system, m: int, families, s: int) -> list[int]:
    """The continuous factors a neural modality reads; more than one is refused by name."""
    continuous = [
        g for g in system._engine_A_deps[m] if g < s and families[g] != _hybrid.CATEGORICAL
    ]
    if len(continuous) > 1:
        raise NotImplementedError(
            f"modality {system.modality_names[m]!r} is a neural likelihood over the continuous "
            f"factors {[system._facs[g] for g in continuous]}; a neural modality reads exactly "
            "one continuous factor, the rest of its parents being categorical (design: "
            "recognition-beside-analytic.md L1)"
        )
    return continuous


def _config_grid(sizes: tuple[int, ...]) -> Array:
    """The ``(M, len(sizes))`` int grid of mode configurations, row-major, ``M = prod(sizes)``."""
    total = math.prod(sizes)
    if not sizes:
        return jnp.zeros((1, 0), jnp.int32)
    return jnp.stack(jnp.unravel_index(jnp.arange(total), sizes), axis=-1).astype(jnp.int32)


def _belief_components(q) -> tuple[Array, Array]:
    """``(locs, chols)`` of a Gaussian or mixture belief, with a leading component axis."""
    if isinstance(q, MixtureGaussian):
        return q.means, q.chols
    return q.loc[None], q.chol[None]


def _dense_model(system, a, m: int, q, out, priors, families, s: int):
    """Modality ``m``'s observation model at the belief ``q`` over its factor, as
    ``(H, c, R_chol, axes)`` with leading ``(K, M)`` axes: ``K`` the belief's components (one for a
    Gaussian), ``M`` the mode configurations over the parents ``axes`` (one where there are none),
    row-major in ``axes`` order.

    A linear-Gaussian modality reads its declared arrays (a noise map at each component's mean;
    action nodes alone select their level). A neural one is linearised at each component's mean
    and, per configuration of its categorical state parents, at their one-hots; its action-node
    parents are held at the one-hot of their point prior's level (``priors[g]``).
    """
    mods, facs = system.modality_names, system._facs
    deps = system._engine_A_deps[m]
    locs, chols = _belief_components(q)
    k = locs.shape[0]
    mixture = isinstance(q, MixtureGaussian)

    def level_of(g: int, f: int):
        return _hybrid.prior_point_level(
            priors[g], system._sensed[g - s], mods[m], facs[f], mixture
        )

    if isinstance(a, MLPConditional):
        f = next(g for g in deps if g < s and families[g] != _hybrid.CATEGORICAL)
        axes = tuple(g for g in deps if g < s and g != f)
        node_vec = {
            g: jax.nn.one_hot(level_of(g, f), a.parent_dims[deps.index(g)]) for g in deps if g >= s
        }
        sizes = tuple(out[g].probs.shape[-1] for g in axes)
        grid = _config_grid(sizes)

        def at(loc, chol, cfg):
            parents = tuple(
                loc
                if g == f
                else jax.nn.one_hot(cfg[axes.index(g)], sizes[axes.index(g)])
                if g in axes
                else node_vec[g]
                for g in deps
            )
            return linearise(a, loc, chol, parents, deps.index(f), a.propagation)

        H, c, R = jax.vmap(lambda loc, chol: jax.vmap(lambda cfg: at(loc, chol, cfg))(grid))(
            locs, chols
        )
        return H, c, R, axes
    f, obs_parents = deps[0], deps[1:]
    if (
        obs_parents
        and _hybrid.selects_by_nodes(a, families[f])
        and all(g >= s for g in obs_parents)
    ):
        a = _hybrid.select_observation(
            _continuous.lg_point(a), tuple(level_of(g, f) for g in obs_parents)
        )
        obs_parents = ()
    if obs_parents:
        n_modes = math.prod(out[g].probs.shape[-1] for g in obs_parents)
        Hm, cm, Rm = (
            _per_mode(a.H, n_modes, 2),
            _per_mode(a.c, n_modes, 1),
            _per_mode(a.R_chol, n_modes, 2),
        )
        tile = lambda x: jnp.broadcast_to(x, (k, *x.shape))  # noqa: E731
        return tile(Hm), tile(cm), tile(Rm), tuple(obs_parents)
    R = a.R_chol_at(locs) if a.noise_map else jnp.broadcast_to(a.R_chol, (k, *a.R_chol.shape))
    H, c = jnp.broadcast_to(a.H, (k, *a.H.shape)), jnp.broadcast_to(a.c, (k, *a.c.shape))
    return H[:, None], c[:, None], R[:, None], ()


def _joint_weights(system, q, out, axes: tuple[int, ...], f: int) -> Array:
    """The ``(K, M)`` weight of each (component, mode configuration) pair under the belief.

    A Gaussian weighs one. A mixture's components weigh by their weights; where the modes are
    the factor's own mode parents (mode-aligned banks) a component holds its bank's
    configuration, otherwise the configurations weigh by the mode beliefs.
    """
    if not isinstance(q, MixtureGaussian):
        return jnp.ones((1, 1))
    w = jax.nn.softmax(q.log_weights, axis=-1)
    if not axes:
        return w[:, None]
    n_modes = math.prod(out[g].probs.shape[-1] for g in axes)
    own = tuple(g for g in system._B[f] if g < len(out) and isinstance(out[g], Categorical))
    k = w.shape[-1]
    if own == axes and system.mixture_schedule == "mode_aligned" and k % n_modes == 0:
        bank = jnp.arange(k) // (k // n_modes)
        return w[:, None] * (bank[:, None] == jnp.arange(n_modes)[None, :])
    q_modes = jax.nn.softmax(_hybrid._mode_log_weights(tuple(out[g] for g in axes)), axis=-1)
    return w[:, None] * q_modes[None, :]


def _neural_table(a: MLPConditional, y: Array, deps: tuple[int, ...], sizes: tuple[int, ...]):
    """``log p(y | parents)`` at every configuration of a neural modality's categorical
    parents (one-hots), shaped ``sizes``."""
    grid = _config_grid(sizes)

    def one(cfg):
        return a.log_prob(y, tuple(jax.nn.one_hot(cfg[i], n) for i, n in enumerate(sizes)))

    return jax.vmap(one)(grid).reshape(sizes)


def expected_nll(system, a, m: int, y: Array, out, priors, families, s: int) -> Array:
    """``- E_q[log p(y | s)]`` of modality ``m`` (declaration ``a``) under the beliefs ``out``.

    A linear-Gaussian or neural modality over a continuous factor takes the closed-form Gaussian
    expectation of the quadratic per component, weighted (:func:`~hybrid_aif.inference.amortized.
    expected_gaussian_nll`; for a neural one, with ``(H, c, R)`` linearised at each component's
    mean, ``-log p(y | mu) + 1/2 tr(J^T R^-1 J Sigma)`` to first order). An emission or neural
    modality over categorical parents alone takes the expectation of its log table
    under the mean-field parent beliefs.
    """
    deps = system._engine_A_deps[m]
    y = jnp.asarray(y)
    cont = [g for g in deps if g < s and families[g] != _hybrid.CATEGORICAL]
    if not cont:
        sizes = tuple(out[g].probs.shape[-1] for g in deps)
        if isinstance(a, MLPConditional):
            table = _neural_table(a, y, deps, sizes)
        elif _emission.is_emission(a):
            table = jnp.broadcast_to(_emission.emission_scores(a, y), sizes)
        else:
            raise TypeError(f"modality {system.modality_names[m]!r}: no expected log-likelihood")
        joint = out[deps[0]].probs
        for g in deps[1:]:
            joint = joint[..., None] * out[g].probs
        return -jnp.sum(joint * table)
    f = cont[0] if isinstance(a, MLPConditional) else deps[0]
    q = out[f]
    H, c, R, axes = _dense_model(system, a, m, q, out, priors, families, s)
    locs, chols = _belief_components(q)
    return _amortized.expected_gaussian_nll(
        locs, chols, H, c, R, y, _joint_weights(system, q, out, axes, f)
    )


def _condition_mixture(system, out: list, f: int, H, c, R_chol, y, who: str, families):
    """Condition the mode-aligned mixture factor ``f`` of ``out`` on ``y = H x + c + eps``.

    Every component takes the Kalman update and the weights are re-scored; ``H``, ``c`` and
    ``R_chol`` are each one array for every component or one per component (``(K, d_obs, d)``,
    ``(K, d_obs)``, ``(K, d_obs, d_obs)``: a linearised neural modality reads each component at
    its own mean). Where the factor has
    categorical mode parents, their beliefs in ``out`` are replaced by the mode evidence message
    and the returned log evidence reads the mode belief the tabular stage left. Returns
    ``(posterior, log_evidence, per_component_loglik or None)``; the caller stores the posterior.
    """
    if system.mixture_schedule != "mode_aligned":
        raise NotImplementedError(
            f"{who} observes mixture-carried factor {system._facs[f]!r} under "
            "mixture_schedule='runnalls': filtering a mixture belief against evidence "
            "needs mode-owned slots (the per-mode evidence a cross-mode Runnalls merge "
            'destroys). Declare System(mixture_schedule="mode_aligned") to enable it.'
        )
    q = out[f]
    posterior, log_ev, ll = condition_mixture_components(q, H, c, R_chol, y)
    mode_parents = tuple(g for g in system._B[f] if families[g] == _hybrid.CATEGORICAL)
    if not mode_parents:
        return posterior, log_ev, None
    mode_beliefs = tuple(out[g] for g in mode_parents)
    # the evidence of y reads the mode belief the tabular stage left, as the
    # switching path does, not the mixture's own bank masses, which still carry
    # the prior mode mass
    n_modes = 1
    for b in mode_beliefs:
        n_modes *= b.probs.shape[-1]
    log_ev = jax.scipy.special.logsumexp(
        _hybrid._mode_log_weights(mode_beliefs) + mode_bank_evidence(q.log_weights, ll, n_modes)
    )
    updated = mode_evidence_message(mode_beliefs, q.log_weights, ll)
    for g, q_mode in zip(mode_parents, updated, strict=True):
        out[g] = q_mode
    return posterior, log_ev, ll


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
    prev_beliefs, prev_action = _previous_in(system, params, previous, "infer_states")
    sensed = None
    if system._sensed_columns:
        sensed = system._sensed_priors(params, priors, "infer_states", prev_action)
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
    validation._validate_analytic_declaration(system, params)  # before any family derivation
    n_mods = len(params.likelihood)
    pre_reduce: list[MixtureGaussian | None] = [None] * n_mods
    mode_log_ev: list[Array | None] = [None] * n_mods
    log_ev: list[Array | None] = [None] * n_mods
    flags: list[Array | None] = [None] * len(params.transition)
    free_energy: list[Array | None] = [None]
    # each LG modality's log evidence, the term the tick's free energy subtracts
    evidence_terms: list[Array | None] = [None] * n_mods
    # each recognition message's ``- log Z + E_q[log m]`` and its log normaliser, and each
    # covered modality's ``- E_q[log p(o | s)]`` by modality name
    message_terms: list[Array] = []
    message_log_z: dict[str, Array] = {}
    covered_terms: dict[str, Array] = {}

    def _detail() -> InferenceDetail | None:
        if not detail:
            return None
        total = jnp.zeros(()) if free_energy[0] is None else free_energy[0]
        for ev in evidence_terms:
            if ev is not None:
                total = total - ev
        for term in message_terms:
            total = total + term
        for entry in system.recognition or ():  # an absent entry's normaliser is zero
            if entry.variable in system._facs:
                message_log_z.setdefault(f"recognition:{entry.variable}", jnp.zeros(()))
        return InferenceDetail(
            pre_reduce=tuple(pre_reduce),
            mode_log_evidence=tuple(mode_log_ev),
            log_evidence=tuple(log_ev),
            unsupported=tuple(flags),
            free_energy=total,
            recognition_log_evidence=dict(message_log_z),
            covered_terms=dict(covered_terms),
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
        rows = _measurements(
            system,
            params,
            tuple(priors_used),
            observations,
            externals,
            masks,
            (True,) * len(system.B_deps),
        )
        # a modality a recognition entry covers is not inverted: the entry's message stands for it
        tab_covered = _amortized.covered_modalities(system.recognition, system.modality_names)
        kept = tuple(m for m in range(len(params.likelihood)) if m not in tab_covered)
        lls, deps, obs_all, masks_all, exempt = _append_rows(
            tuple(params.likelihood[m] for m in kept),
            tuple(A_deps[m] for m in kept),
            tuple(observations[m] for m in kept),
            None if masks is None else tuple(masks[m] for m in kept),
            (),
            rows,
            {f: f for f in range(len(priors_used))},
        )
        beliefs, sub_flags = fpi_support(
            lls,
            deps,
            priors_used,
            obs_all,
            num_iters=system.num_iters,
            masks=masks_all,
            support_floor=support_floor,
            support_exempt=exempt,
        )
        flags[:] = list(sub_flags)[: len(flags)]
        pre = beliefs
        if detail and rows:
            pre, _ = fpi_support(
                tuple(params.likelihood[m] for m in kept),
                tuple(A_deps[m] for m in kept),
                priors_used,
                tuple(observations[m] for m in kept),
                num_iters=system.num_iters,
                masks=None if masks is None else tuple(masks[m] for m in kept),
                support_floor=support_floor,
            )
        if detail:
            free_energy[0] = mean_field_free_energy(
                tuple(params.likelihood[m] for m in kept),
                tuple(A_deps[m] for m in kept),
                priors_used,
                pre,
                tuple(observations[m] for m in kept),
                None if masks is None else tuple(masks[m] for m in kept),
            )
            _categorical_terms(
                rows,
                pre,
                beliefs,
                {f: f for f in range(len(priors_used))},
                message_terms,
                message_log_z,
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
            _neural_continuous(system, m, families, s)  # refuses two continuous parents
            continue
        continuous = [facs[f] for f in A_deps[m] if f < s and families[f] != _hybrid.CATEGORICAL]
        if continuous:
            raise NotImplementedError(
                f"modality {mods[m]!r} observes continuous factor(s) {continuous} with a "
                f"non-linear-Gaussian likelihood ({type(a).__name__}): discrete-given-"
                "continuous / MNLR observation models are out of scope"
            )
    system._validate(params, face, declared=True)
    priors = system._initial_priors(params, face) if priors is None else priors
    out: list[Belief] = list(priors)
    measurements = _measurements(
        system,
        params,
        tuple(priors),
        observations,
        externals,
        masks,
        tuple(fam == _hybrid.CATEGORICAL for fam in families[:s]),
    )
    rows = tuple(r for r in measurements if r.kind == _amortized.CATEGORICAL_KIND)
    cat_f = (
        *(f for f, fam in enumerate(families) if fam == _hybrid.CATEGORICAL),
        *range(s, s + n_sensed),
    )
    covered = _amortized.covered_modalities(system.recognition, mods)
    cat_m = tuple(
        m
        for m, a in enumerate(params.likelihood)
        if isinstance(a, TabularObservation) and m not in covered
    )
    # a Gaussian emission over categorical parents alone joins the tabular sweep as the
    # one-row tensor of its predictive scores (structure-learning.md S2); its scores are
    # densities, so the support check skips it
    emission_m = tuple(
        m for m, a in enumerate(params.likelihood) if _emission.is_emission(a) and m not in covered
    )
    # a neural likelihood over categorical parents alone joins the sweep as its log-density table
    neural_tab_m = tuple(
        m
        for m, a in enumerate(params.likelihood)
        if isinstance(a, MLPConditional)
        and m not in covered
        and not any(g < s and families[g] != _hybrid.CATEGORICAL for g in A_deps[m])
    )
    pos: dict[int, int] = {}
    row_pre: tuple = ()
    if cat_f:
        pos = {f: i for i, f in enumerate(cat_f)}
        sub_m = cat_m + emission_m + neural_tab_m
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
        for m in neural_tab_m:
            shape = tuple(priors[f].probs.shape[-1] for f in A_deps[m])
            if masks is not None and masks[m] is False:
                sub_lls.append(jnp.zeros((1, *shape)))
            else:
                sub_lls.append(
                    _neural_table(
                        params.likelihood[m], jnp.asarray(observations[m]), A_deps[m], shape
                    )[None]
                )
            sub_obs.append(jnp.zeros((), jnp.int32))
        sub_deps = tuple(tuple(pos[f] for f in A_deps[m]) for m in sub_m)
        sub_masks = None if masks is None else tuple(masks[m] for m in sub_m)
        exempt = tuple(range(len(cat_m), len(sub_m)))
        lls, deps, obs_all, masks_all, exempt_all = _append_rows(
            tuple(sub_lls), sub_deps, tuple(sub_obs), sub_masks, exempt, rows, pos
        )
        sub_posteriors, sub_flags = fpi_support(
            lls,
            deps,
            tuple(priors[f] for f in cat_f),
            obs_all,
            num_iters=system.num_iters,
            masks=masks_all,
            support_floor=support_floor,
            support_exempt=exempt_all,
        )
        for f, q, flag in zip(cat_f, sub_posteriors, sub_flags, strict=True):
            out[f] = q
            if f < s:
                flags[f] = flag
        if detail:  # the tabular stage's free energy, before any LG mode message
            pre = sub_posteriors
            if rows:  # the belief the categorical messages are applied to: the sweep without them
                pre, _ = fpi_support(
                    tuple(sub_lls),
                    sub_deps,
                    tuple(priors[f] for f in cat_f),
                    tuple(sub_obs),
                    num_iters=system.num_iters,
                    masks=sub_masks,
                    support_floor=support_floor,
                    support_exempt=exempt,
                )
            free_energy[0] = mean_field_free_energy(
                tuple(sub_lls),
                sub_deps,
                tuple(priors[f] for f in cat_f),
                pre,
                tuple(sub_obs),
                sub_masks,
            )
            row_pre = pre
    # a neural modality's linearisation and a noise map are read at the belief the factor holds
    # when the modality is reached (the sequential extended filter): the tick's prior for the
    # first modality over it, the running posterior after an earlier one; a mixture-carried
    # factor is read at each of its current components' means
    for m, a in enumerate(params.likelihood):
        neural = isinstance(a, MLPConditional)
        if not (isinstance(a, LGObservation) or neural) or m in emission_m or m in covered:
            continue  # tabular, emissions and neural tables: the FPI sub-model; covered: the entry
        if neural and m in neural_tab_m:
            continue
        present = True if masks is None else masks[m]
        deps = A_deps[m]
        if neural:
            f = _neural_continuous(system, m, families, s)[0]
            #  the state mode parents switch the model; action nodes hold their point level
            obs_parents = tuple(g for g in deps if g < s and g != f)
        else:
            f, obs_parents = deps[0], deps[1:]
        if (
            not neural
            and obs_parents
            and _hybrid.selects_by_nodes(a, families[f])
            and all(g >= s for g in obs_parents)
        ):
            #  a channel read through action nodes alone selects its head and noise at the
            #  node's level, the node's prior being a point; the factor keeps its family
            a = _hybrid.select_observation(
                _continuous.lg_point(a),
                tuple(
                    _hybrid.prior_point_level(
                        priors[g],
                        system._sensed[g - s],
                        mods[m],
                        facs[f],
                        families[f] == _hybrid.MIXTURE,
                    )
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
        if neural:
            #  linearised at the belief the factor holds now, per component and, with
            #  state mode parents, per mode configuration (``(K, M, ...)``)
            lin_H, lin_c, lin_R, _ = _dense_model(system, a, m, out[f], out, priors, families, s)
        if obs_parents:
            # Switching observation: grow the factor to K*M, read the mode message from
            # the grown weights before reducing (so no mode-owned slots are needed;
            # mixture-filtering.md Guidance)
            q = out[f]
            mode_lp = _hybrid._mode_log_weights(tuple(out[g] for g in obs_parents))
            if neural:
                grown, per_mode_ev, ll_km = condition_mixture_switching_components(
                    q, lin_H, lin_c, lin_R, observations[m], mode_lp, per_component=True
                )
            else:
                grown, per_mode_ev, ll_km = condition_mixture_switching_components(
                    q, a.H, a.c, a.R_chol, observations[m], mode_lp
                )
            updated = switching_mode_message(tuple(out[g] for g in obs_parents), per_mode_ev)
            for g, q_mode in zip(obs_parents, updated, strict=True):
                out[g] = q_mode
            pre_reduce[m], mode_log_ev[m] = grown, per_mode_ev
            log_ev[m] = jax.scipy.special.logsumexp(mode_lp + per_mode_ev)
            own_modes = tuple(g for g in system._B[f] if families[g] == _hybrid.CATEGORICAL)
            disjoint = not any(g in obs_parents for g in own_modes)
            if own_modes and disjoint and system.mixture_schedule == "mode_aligned":
                #  the factor's own mode parents, none of which the channel reads, take the
                #  evidence as the non-switching update sends it: per component, the channel's
                #  modes marginalised at their belief; a channel that reads one of them sends
                #  its own mode message and the banks are not read
                ll = jax.scipy.special.logsumexp(mode_lp + ll_km, axis=-1)
                mode_beliefs = tuple(out[g] for g in own_modes)
                n_modes = 1
                for b in mode_beliefs:
                    n_modes *= b.probs.shape[-1]
                log_ev[m] = jax.scipy.special.logsumexp(
                    _hybrid._mode_log_weights(mode_beliefs)
                    + mode_bank_evidence(q.log_weights, ll, n_modes)
                )
                for g, q_mode in zip(
                    own_modes, mode_evidence_message(mode_beliefs, q.log_weights, ll), strict=True
                ):
                    out[g] = q_mode
            out[f] = _hybrid.pad_mixture(grown, system.mixture_k)
            out, log_ev[m] = _present_select(present, out, before, log_ev[m])
            evidence_terms[m] = log_ev[m]
            continue
        if neural:
            H, c, R_chol = (x[:, 0] for x in (lin_H, lin_c, lin_R))  # one mode configuration
            if families[f] != _hybrid.MIXTURE:
                H, c, R_chol = H[0], c[0], R_chol[0]
        elif families[f] != _hybrid.MIXTURE:
            H, c = a.H, a.c
            R_chol = a.R_chol_at(out[f].loc) if a.noise_map else a.R_chol
        else:
            H, c = a.H, a.c
            R_chol = a.R_chol_at(out[f].means) if a.noise_map else a.R_chol
        if families[f] != _hybrid.MIXTURE:
            out[f], log_ev[m] = condition(out[f], H, c, R_chol, observations[m])
            out, log_ev[m] = _present_select(present, out, before, log_ev[m])
            evidence_terms[m] = log_ev[m]
            continue
        posterior, log_ev[m], ll = _condition_mixture(
            system, out, f, H, c, R_chol, observations[m], f"modality {mods[m]!r}", families
        )
        if ll is not None:
            mode_log_ev[m] = ll
        out[f] = posterior
        out, log_ev[m] = _present_select(present, out, before, log_ev[m])
        evidence_terms[m] = log_ev[m]
    for ms in measurements:
        if ms.kind != _amortized.GAUSSIAN_KIND:
            continue
        # a Gaussian measurement of the factor through the identity map, applied after every
        # modality's update so it multiplies the posterior the tick's evidence has left
        f, present = ms.factor, ms.present
        d = ms.loc.shape[-1]
        H, c, R_chol = jnp.eye(d), jnp.zeros(d), jnp.diag(ms.scale)
        before = None if present is True else list(out)
        if families[f] == _hybrid.MIXTURE:
            posterior, log_z, _ = _condition_mixture(
                system, out, f, H, c, R_chol, ms.loc, f"recognition entry {ms.variable!r}", families
            )
            out[f] = posterior
        else:
            out[f], log_z = condition(out[f], H, c, R_chol, ms.loc)
        out, log_z = _present_select(present, out, before, log_z)
        if detail:
            expect = (
                _amortized.mixture_message_expectation
                if families[f] == _hybrid.MIXTURE
                else _amortized.gaussian_message_expectation
            )(out[f], ms.loc, ms.scale)
            term = -log_z + expect
            message_terms.append(term if present is True else jnp.where(present, term, 0.0))
            message_log_z[f"recognition:{ms.variable}"] = log_z
    if detail:
        _categorical_terms(rows, row_pre, out, pos, message_terms, message_log_z)
        for m in sorted(covered):
            a = params.likelihood[m]
            present = True if masks is None else masks[m]
            term = jnp.zeros(())
            if present is not False:
                term = expected_nll(system, a, m, observations[m], out, priors, families, s)
                term = term if present is True else jnp.where(present, term, 0.0)
                message_terms.append(term)
            covered_terms[mods[m]] = term
    return tuple(out), _detail()


def _previous_in(
    system, params: SystemParams, previous: Mapping[str, Belief | Array] | None, face: str
) -> tuple[tuple[Belief | None, ...] | None, tuple[Array | None, ...] | None]:
    """A keyed ``previous`` as ``(prev_beliefs, prev_action)``: one belief per state factor
    (``None`` where absent) and one entry per action node (``None`` where ``previous`` does not
    name it), or ``None`` without any entry. A discrete node's entry is an int32 level, checked
    against the node's level count (eagerly when concrete, by ``eqx.error_if`` when traced); a
    continuous node's entry is its float value."""
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
    if not any(name in previous for name in actions):
        return beliefs, None
    space = system._space(params)
    entries: list[Array | None] = []
    for k, name in enumerate(actions):
        if name not in previous:
            entries.append(None)
            continue
        if space.kinds[k] != LEVEL:
            entries.append(jnp.asarray(previous[name], float))
            continue
        given, count = jnp.asarray(previous[name]), space.levels[k]
        if not jnp.issubdtype(given.dtype, jnp.integer):
            raise ValueError(
                f"{face}: previous[{name!r}] is {previous[name]!r}, and action node {name!r} "
                f"takes a level: pass an integer in [0, {count})"
            )
        level = given.astype(jnp.int32)
        bad = (level < 0) | (level >= count)
        message = (
            f"{face}: previous[{name!r}] is a level outside action node {name!r}'s "
            f"{count} level(s); pass an integer in [0, {count})"
        )
        if isinstance(level, jax.core.Tracer):
            level = eqx.error_if(level, jnp.any(bad), message)
        elif bool(jnp.any(bad)):
            raise ValueError(message)
        entries.append(level)
    return beliefs, tuple(entries)


def _fill_lagged(
    system,
    observations: tuple[Array | None, ...],
    present: tuple[bool | Array, ...] | None,
    prev_beliefs: tuple[Belief | None, ...] | None,
    prev_action: Array | tuple[Array | None, ...] | None,
    first: Array | bool = False,
) -> tuple[tuple[Array | None, ...], tuple[bool | Array, ...] | None]:
    """The observations with every lagged modality's datum read from the carry where none
    was given: the source factor's previous belief mean, or the previous action's level (the
    entry of a node ``prev_action`` does not name is absent, like a missing carry). A modality
    with neither is absent, as is every lagged modality when ``first`` (a traced flag for the
    tick with no carry)."""
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
            entry = None if prev_action is None else prev_action[index]
            datum = None if entry is None else jnp.asarray(entry)
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
