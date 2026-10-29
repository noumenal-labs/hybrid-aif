"""Recorded-trace replay: a declared model driven through what a controller actually did.

At each tick the model infers from the recorded observations, runs its own policy
inference as a counterfactual, and advances its empirical prior with the recorded
action (teacher forcing). ``actions[t]`` is the action taken after
``observations[t]``; every output is stacked over time. Observations, actions, the
presence flags, beliefs and per-factor flags are keyed by name, as at every face
(:mod:`hybrid_aif.system.keyed`). :func:`stack_traces` pads variable-length traces into one
batch for ``jax.vmap``. Scoring the result is the consumer's job.

Design: docs/design/replay.md
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hybrid_aif.description import build
from hybrid_aif.distributions import Categorical
from hybrid_aif.inference import amortized as _amortized
from hybrid_aif.inference.discrete import SUPPORT_FLOOR
from hybrid_aif.learning import structure as _structure
from hybrid_aif.objectives import TermTree
from hybrid_aif.objectives import hybrid as _hybrid
from hybrid_aif.objectives.amortized import features as _amortized_features
from hybrid_aif.planning.enumerate import sample_action
from hybrid_aif.system import Belief, LearnState, System, SystemParams, keyed

__all__ = ["Replay", "replay", "replay_document", "stack_traces"]


class Replay(NamedTuple):
    """What :func:`replay` returns. Every leaf carries a leading ``(T, …)`` axis.

    ``beliefs`` are the posteriors at every tick, ``priors`` the priors each was
    inferred under, both keyed by factor name; ``q_pi`` is the counterfactual policy
    posterior and ``trees`` its term tree (``None`` under ``metrics=False``); ``chosen``
    is the model's own deterministic choice, keyed by the action nodes' names
    (``System.names.actions``; under a policy
    module, the habit planner's draw under ``key``). ``action_log_prob`` is
    ``log P(actions[t])`` under the policy posterior, summed over policies sharing
    that first step: ``-inf`` where no policy begins with it, the module's
    log-density under a policy module, and ``None`` over a continuous action space
    with no module. ``free_energy`` is the per-tick variational free energy
    (``InferenceDetail.free_energy``), ``None`` on the amortized path; ``log_evidence`` is
    ``−free_energy``, the tick's log evidence where its posterior is exact (every modality
    LG, or one categorical factor) and a lower bound otherwise, so its sum over the valid
    ticks is the prequential description length of the record under the model as it stood
    at each tick. ``valid`` marks real ticks. ``exercised`` maps each modality to a bool,
    present on at least one valid tick (``None`` without ``present``). ``unsupported`` maps
    each factor to a ``(T,)`` bool (``None`` for a continuous factor; empty on the
    amortized path): under the beliefs held about its co-parents, no level was
    compatible with the tabular evidence; co-parents of the contradicted factor are
    flagged too. ``params``/``learn_state`` are the final parameters (containers keyed by name) and
    optimizer carry under ``learn=True``, else ``None``.
    Design: docs/design/zero-support-evidence.md Z2
    """

    beliefs: dict[str, Belief]
    priors: dict[str, Belief]
    q_pi: Categorical
    trees: TermTree | None
    chosen: dict[str, Array]
    action_log_prob: Array | None
    log_evidence: Array | None
    free_energy: Array | None
    valid: Array
    exercised: dict[str, Array] | None
    unsupported: dict[str, Array | None]
    params: SystemParams | None
    learn_state: LearnState | None


_select = _hybrid.where_tree  # shape-checked: a carry must select between equal shapes


def _check_shapes(
    system: System, observations: tuple[Array, ...], actions: Array, amortized: bool
) -> int:
    present = [o for o in observations if o is not None]
    lagged = set(system._lagged_sources)
    if len(present) < len(observations) and not amortized:
        mods = system._names().modalities
        missing = [mods[m] for m, o in enumerate(observations) if o is None and m not in lagged]
    else:
        missing = []
    if missing:
        raise ValueError(
            f"replay: observations {missing} are None; on the tabular and hybrid paths "
            "every modality is read every tick — use present= for a modality that was not "
            "observed"
        )
    lengths = {int(jnp.shape(o)[0]) for o in present} | {int(jnp.shape(actions)[0])}
    if len(lengths) != 1:
        raise ValueError(
            f"replay: observations and actions disagree on the trace length; leading axes are "
            f"{sorted(lengths)}. Pad to one T (stack_traces does) and mark padding with valid="
        )
    return lengths.pop()


def _check_action_columns(names: keyed.FaceNames, actions: Mapping[str, Array]) -> None:
    """Each action node's column is ``(T,)`` integer levels, or ``(T, d)`` floats when its
    control node is continuous."""
    for i, k in enumerate(names.actions):
        v = actions[k]
        if names.kind(i) == "level":
            want, kind = "(T,)", "a level per tick (int)"
            ok = v.ndim == 1 and jnp.issubdtype(v.dtype, jnp.integer)
        else:
            d = names.sizes[i]
            want, kind = f"(T, {d})", f"the node's {d}-wide value per tick (float)"
            ok = v.ndim == 2 and v.shape[1] == d and jnp.issubdtype(v.dtype, jnp.floating)
        if not ok:
            raise ValueError(
                f"replay: actions[{k!r}] must be {want} — {kind} — got shape "
                f"{tuple(v.shape)} of dtype {v.dtype}"
            )


def _positional_trace(
    system: System,
    params: SystemParams,
    observations: Mapping[str, Array | None],
    actions: Mapping[str, Array],
    present: Mapping[str, Array] | None,
    valid: Array | None,
) -> tuple[tuple[Array | None, ...], Array, Array | None]:
    """The keyed trace as replay's positional ``(observations, actions, present)``, the flags a
    ``(T, M)`` bool array (a modality ``present`` leaves out is observed at every tick)."""
    names = system._names(params)
    if names.mixed:
        raise NotImplementedError(
            "replay plans every tick, and this System's control space mixes discrete and "
            "continuous nodes, which replay does not plan over: replay a System whose "
            "control nodes are of one kind, or score the recorded plan with "
            "System.score_policies"
        )
    lagged = {modality for modality, _ in system.lagged}
    if lagged:  # a lagged modality's trace may be left out: its datum comes from the carry
        observations = {**{name: None for name in lagged}, **dict(observations)}
    obs = keyed.observations_in(names, observations, "replay")
    keyed.check_mapping(names, "actions", actions, "replay")
    acts = {k: jnp.asarray(v) for k, v in actions.items()}
    if names.layout is None:
        floats = [k for k, v in acts.items() if not jnp.issubdtype(v.dtype, jnp.integer)]
        if floats:
            raise ValueError(
                f"replay: actions {floats} are not integer; a discrete control's action is a "
                "(T,) array of level indices"
            )
    _check_action_columns(names, acts)
    lengths = [int(jnp.shape(o)[0]) for o in obs if o is not None]
    lengths += [] if valid is None else [int(jnp.shape(valid)[0])]
    if not acts and not lengths:
        raise ValueError(
            "replay: nothing sets the trace length (no observations and no controlled factor); "
            "pass valid= with one entry per tick"
        )
    lead = (lengths[0],) if lengths else None
    positional_actions = keyed.actions_in(names, acts, "replay", lead_shape=lead)
    if present is None:
        return obs, positional_actions, None
    T = int(jnp.shape(positional_actions)[0])
    flags = keyed.present_in(names, present, "replay", absent=jnp.ones((T,), bool))
    bad = {n: tuple(jnp.shape(present[n])) for n in present if tuple(jnp.shape(present[n])) != (T,)}
    if bad:
        raise ValueError(f"replay: present entries are ({T},) bool, one flag per tick; got {bad}")
    return obs, positional_actions, jnp.stack([jnp.asarray(f, bool) for f in flags], axis=-1)


def replay(
    system: System,
    params: SystemParams,
    observations: Mapping[str, Array | None],
    actions: Mapping[str, Array],
    *,
    present: Mapping[str, Array] | None = None,
    valid: Array | None = None,
    legal: Mapping[str, Array] | Array | None = None,
    priors: Mapping[str, Belief] | None = None,
    key: Array | None = None,
    externals: Mapping[str, Array] | None = None,
    metrics: bool = True,
    learn: bool = False,
    lr: float | Array = 1.0,
    prior: SystemParams | None = None,
    support_floor: float = SUPPORT_FLOOR,
) -> Replay:
    """Drive ``(system, params)`` through a recorded trace (design: ``replay.md`` R1–R7).

    ``observations`` maps each modality name to a ``(T, …)`` array, as
    :meth:`System.infer_states` takes them plus a time axis; ``actions`` maps each
    action node's name (``System.names.actions``) to its ``(T, …)`` column, as
    :meth:`System.act` returns them plus a time axis; each clamps its control node at its tick
    (the efference copy). ``present`` maps a modality to a ``(T,)`` bool, whether it was observed at
    each tick (a modality left out was observed at every tick; ``None``: all were; an
    unknown key raises); ``valid`` is ``(T,)`` bool, the non-padding rows (an invalid tick
    leaves the carry untouched, its outputs unspecified). ``legal`` is the per-tick
    constraint in ``infer_policies``' spellings with a leading time axis, applied to
    ``chosen`` and ``action_log_prob``: keyed by control node, each entry ``(T, levels)``
    or ``(T, H, levels)`` (a node left out is unrestricted), or a ``(T, P)`` bool array
    over the policy set. ``priors`` seeds the first tick, keyed by
    factor (``None``: the initial priors). ``key`` drives a
    stochastic planner and is split per tick; selection is deterministic.
    ``externals`` supplies ``external:<name>`` recognition inputs as
    ``(T, width)`` arrays. ``learn=True`` runs :meth:`System.learn` between
    perception and action from the second tick and returns the final params (and
    ``LearnState`` for gradient slots); a modality absent at a tick (``present``, or a padded
    tick) teaches its likelihood slot nothing there; on the amortized path ``System.learn``
    raises (train with ``fit_world``). ``prior`` is the model's prior params, which a
    scheduled reduction (``System.reduction_period``) scores its merges against and returns a
    killed level to; ``None`` takes ``params``, right when the replay starts from the prior and
    wrong when ``params`` were already learned. ``lr`` is the learning rate: a nonzero Python
    number is passed to the compiled scan as a float32 array, so a schedule of rates compiles
    once, and a Python ``0.0`` is static (it skips learning at trace time).
    ``metrics=False`` drops the per-tick term trees. On the amortized path the
    belief half is the per-tick faces (``infer_states``,
    ``update_empirical_prior``).
    """
    system._refuse_sensed("replay")
    view = system._params_in(params, "replay")  # the positional view the scan carries
    amortized = system._is_amortized(view)
    names = system._names(view)
    observations, actions, mask = _positional_trace(
        system, view, observations, actions, present, valid
    )
    T = _check_shapes(system, observations, actions, amortized)
    n_mods = len(system.A_deps)
    discrete = names.layout is None
    valid = jnp.ones((T,), bool) if valid is None else jnp.asarray(valid, bool)
    if valid.shape != (T,):
        raise ValueError(f"replay: valid must be ({T},), got {valid.shape}")
    if mask is not None:
        mask = jnp.asarray(mask, bool)
        if mask.shape != (T, n_mods):
            raise ValueError(f"replay: present entries are ({T},) bool, one flag per tick")
    legal = _check_legal(system, view, legal, T)
    externals = None if externals is None else {k: jnp.asarray(v) for k, v in externals.items()}
    for name, x in (externals or {}).items():
        if int(x.shape[0]) != T:
            raise ValueError(
                f"replay: externals[{name!r}] has leading axis {x.shape[0]} but the trace "
                f"has {T} ticks"
            )
    if amortized and mask is not None and not isinstance(mask, jax.core.Tracer):
        #  the modalities a recognition module reads are present together or absent
        #  together; inside the scan the mask is a tracer and infer_states cannot refuse a
        #  split, so it is refused here, concretely, by tick — a mask that is itself traced
        #  (replay under vmap/jit) cannot be checked and reduces with ``all`` (absent)
        read = _amortized.read_modalities(system._amortized_structure(params, "replay").plan)
        if read:
            rows = mask[:, list(read)]
            split = np.flatnonzero(np.asarray(jnp.any(rows, axis=-1) & ~jnp.all(rows, axis=-1)))
            if split.size:
                raise ValueError(
                    f"replay: at tick(s) {split.tolist()}, present= keeps some of the modalities "
                    f"{list(read)} the recognition modules read and drops others; on the "
                    "amortized path they are present together or absent together (a dropped "
                    "frame is False across them)"
                )
    scheduled = system.reduction_period is not None and bool(
        not amortized and _structure.growable_factors(system, view)
    )
    learn_state0 = (
        system.init_learn_state(params)
        if learn and not amortized and (system._gradient_slots(params) or scheduled)
        else None
    )
    prior_view = view if prior is None else system._params_in(prior, "replay")
    start = (
        system._initial_priors(params, "replay")
        if priors is None
        else system._beliefs_in(view, priors, "replay")
    )
    if not amortized and system._is_hybrid(params):
        #  the faces carry mixture factors at exactly mixture_k; the seed must too, or
        #  the scan's carry changes shape after the first tick
        start = _hybrid.prepare_beliefs(start, system._families(params), system.mixture_k)
    plan_keys = None if key is None else jax.random.split(key, T)
    out = _replay_core(
        system,
        view,
        observations,
        actions,
        mask,
        valid,
        legal,
        plan_keys,
        externals,
        start,
        learn_state0,
        prior_view,
        discrete=discrete,
        metrics=metrics,
        learn=learn,
        lr=_traced_rate(lr) if learn else None,
        support_floor=support_floor,
    )
    #  the control nodes' entries, one per tick: never inferred, each tick's is the prior
    nodes = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (T, *x.shape)), system._node_priors(view)
    )
    return out._replace(
        beliefs=keyed.beliefs_out(names, out.beliefs, nodes),
        priors=keyed.beliefs_out(names, out.priors, nodes),
        chosen=keyed.actions_out(names, out.chosen),
        exercised=None if out.exercised is None else keyed.observations_out(names, out.exercised),
        params=None if out.params is None else system._params_out(out.params),
        #  the amortized path flags nothing: no factor has a tabular-evidence flag there
        unsupported={}
        if not out.unsupported
        else dict(zip(names.factors, out.unsupported, strict=True)),
    )


def _traced_rate(lr):
    """``lr`` as a float32 array when it is a nonzero Python number, so a schedule of rates
    compiles once; an array passes through, and a Python ``0.0`` stays static (it skips the
    learning at trace time)."""
    if isinstance(lr, int | float) and not isinstance(lr, bool) and float(lr) != 0.0:
        return jnp.asarray(lr, jnp.float32)
    return lr


@eqx.filter_jit
def _replay_core(
    system: System,
    params: SystemParams,
    observations,
    actions,
    mask,
    valid,
    legal,
    plan_keys,
    externals,
    start,
    learn_state0,
    prior_params,
    *,
    discrete: bool,
    metrics: bool,
    learn: bool,
    lr,
    support_floor: float,
) -> Replay:
    """The scan behind :func:`replay`, compiled once per static configuration.

    Concrete checks run in :func:`replay`; shapes and the Python switches are
    static here. Only under ``learn`` do the parameters, learn state and previous
    tick's beliefs and action ride the carry; otherwise the carry is the priors.
    """
    T = int(actions.shape[0])
    ts = jnp.arange(T, dtype=jnp.int32)
    select_key = jax.random.PRNGKey(0)  # deterministic selection never reads it
    carry_prev = learn or bool(system.lagged)  # the previous beliefs and action ride the carry

    def step(carry, xs):
        if carry_prev:
            priors_t, prev_beliefs, prev_action, params_t, ls_t = carry
        else:
            priors_t, params_t, ls_t = carry, params, None
        obs_t, act_t, mask_t, valid_t, plan_key_t, legal_t, ext_t, t = xs
        if system.lagged:
            #  a lagged modality reads the carry; the first tick has none, so it is absent
            obs_t, mask_t = system._fill_lagged(
                obs_t, system._present(mask_t, len(obs_t)), prev_beliefs, prev_action, t == 0
            )
        beliefs, detail = system._infer_states(
            params_t,
            obs_t,
            priors_t,
            detail=True,
            present=mask_t,
            support_floor=support_floor,
            externals=ext_t,
            face="replay",
        )
        if learn:
            learned = system._learn(
                params_t,
                obs_t,
                beliefs,
                prev_beliefs,
                prev_action,
                lr=lr,
                learn_state=ls_t,
                face="replay",
                present=mask_t,
                prior=prior_params,  # what a scheduled reduction scores merges against
            )
            new_params, new_ls = learned if isinstance(learned, tuple) else (learned, None)
            #  the first tick has no carry to learn a transition from
            params_t = _select(t > 0, new_params, params_t)
            if ls_t is not None:
                ls_t = _select(t > 0, new_ls, ls_t)
        q_pi, tree, policies = system._plan(
            params_t, beliefs, None, "midpoint", metrics, plan_key_t, legal_t, "replay"
        )
        chosen = sample_action(
            select_key,
            q_pi,
            policies=policies,
            num_controls=system._control_levels(params_t),
            mode="marginal" if discrete else "policy",
            selection="deterministic",
        )
        if discrete:
            #  compared per state factor (each owner's column, the no-op 0 elsewhere)
            first = _per_factor(system, policies[:, 0, :])
            taken = _per_factor(system, jnp.asarray(act_t, first.dtype))
            same = jnp.all(first == taken[None, :], axis=-1)
            action_log_prob = jnp.log(jnp.sum(q_pi.probs * same, axis=-1))
        elif system._habit_module(params_t) is not None:  # a policy module: its own density
            actor = system._habit_module(params_t)
            action_log_prob = actor.log_prob(act_t, (_amortized_features(beliefs),))
        else:
            action_log_prob = None
        free_energy = None if detail is None else detail.free_energy
        log_evidence = None if free_energy is None else -free_energy
        priors_next = system._update_empirical_prior(params_t, beliefs, act_t, face="replay")
        new_carry = (priors_next, beliefs, act_t, params_t, ls_t) if carry_prev else priors_next
        carry = _select(valid_t, new_carry, carry)
        flags = tuple(
            None if f is None else f & valid_t for f in (detail.unsupported if detail else ())
        )  # padding rows never count as unsupported
        return carry, (
            beliefs,
            priors_t,
            q_pi,
            tree,
            chosen,
            action_log_prob,
            log_evidence,
            free_energy,
            flags,
        )

    carry0 = (
        (start, start, jnp.zeros_like(actions[0]), params, learn_state0) if carry_prev else start
    )
    xs = (observations, actions, mask, valid, plan_keys, legal, externals, ts)
    carry_final, out = jax.lax.scan(step, carry0, xs)
    params_final, ls_final = carry_final[3:] if learn else (None, None)
    (
        beliefs,
        priors_out,
        q_pi,
        trees,
        chosen,
        action_log_prob,
        log_evidence,
        free_energy,
        flags,
    ) = out
    return Replay(
        beliefs=beliefs,
        priors=priors_out,
        q_pi=q_pi,
        trees=trees,
        chosen=chosen,
        action_log_prob=action_log_prob,
        log_evidence=log_evidence,
        free_energy=free_energy,
        valid=valid,
        exercised=None if mask is None else jnp.any(mask & valid[:, None], axis=0),
        unsupported=tuple(flags),
        params=params_final,
        learn_state=ls_final,
    )


def _per_factor(system: System, levels: Array) -> Array:
    """Levels with one column per control node as one column per state factor: each factor's
    own node's column, the no-op 0 where it owns none (as given when the nodes are not owned
    by state factors)."""
    sugar = system._table.sugar
    if sugar is None:
        return levels
    column = {f: k for k, f in enumerate(sugar.owners)}
    zero = jnp.zeros(levels.shape[:-1], levels.dtype)
    return jnp.stack(
        [levels[..., column[f]] if f in column else zero for f in range(len(system._B))],
        axis=-1,
    )


def _check_legal(system: System, params: SystemParams, legal, T: int):
    """``legal`` with a leading time axis on every array entry, validated tick by tick.

    A mapping keyed by control node (``(T, levels)`` or ``(T, H, levels)`` per node) or a
    ``(T, P)`` bool array over the policy set. Each concrete tick slice goes through
    :meth:`System._legal_logits`, whose sentences name the node.
    """
    if legal is None:
        return None
    names = system._names(params)
    if names.layout is not None:
        raise ValueError(
            "replay: legal= is a level mask, and this System's control nodes are continuous; "
            "their constraint is the box in params.control"
        )
    if isinstance(legal, list):
        legal = keyed.legal_list(legal, "replay", form="(T, P)")
    if isinstance(legal, Mapping | tuple):
        entries = keyed.legal_in(names, legal, "replay")
        checked = {}
        for name, entry in zip(names.controls, entries, strict=True):
            if entry is None:
                continue
            m = jnp.asarray(entry, bool)
            if m.ndim < 2 or m.shape[0] != T:
                raise ValueError(
                    f"replay: legal[{name!r}] must carry a leading time axis of {T} — "
                    f"(T, levels) or (T, H, levels) — got shape {tuple(m.shape)}"
                )
            checked[name] = m
        legal = checked
        slice_t = lambda t: {k: m[t] for k, m in legal.items()}  # noqa: E731
    else:
        legal = jnp.asarray(legal, bool)
        if legal.ndim != 2 or legal.shape[0] != T:
            raise ValueError(
                f"replay: legal= as a bare array is (T, P) over the policy set, got "
                f"{tuple(legal.shape)}"
            )
        slice_t = lambda t: legal[t]  # noqa: E731
    policies = system.policies(system._params_out(params))
    #  tick 0: the spelling and shape sentences (shape checks work on tracers)
    system._legal_logits(params, slice_t(0), policies, "replay")
    leaves = tuple(legal.values()) if isinstance(legal, dict) else (legal,)
    if any(isinstance(m, jax.core.Tracer) for m in leaves):
        return legal  # traced (replay under vmap/jit): the per-tick value checks cannot run
    #  every tick, vectorised: no level left legal, or no policy left legal
    if isinstance(legal, dict):
        ok = jnp.ones((T, policies.shape[0]), bool)
        for name, m in legal.items():
            column = names.controls.index(name)
            dead = np.flatnonzero(
                ~np.asarray(jnp.all(jnp.any(m, axis=-1), axis=tuple(range(1, m.ndim - 1))))
            )
            if dead.size:
                raise ValueError(
                    f"replay: at tick(s) {dead.tolist()}, legal[{name!r}] masks every level of "
                    f"control node {name!r} at some step; at least one level must stay legal at "
                    "every masked step"
                )
            if m.ndim == 2:  # (T, levels): the first step
                ok = ok & m[:, policies[:, 0, column]]
            else:  # (T, H, levels): per step — ok[t, p] = all_h m[t, h, policies[p, h, k]]
                idx = policies[None, :, :, column][..., None]  # (1, P, H, 1)
                picked = jnp.take_along_axis(m[:, None, :, :], idx, axis=-1)[..., 0]  # (T, P, H)
                ok = ok & jnp.all(picked, axis=-1)
    else:
        ok = legal
    none_left = np.flatnonzero(~np.asarray(jnp.any(ok, axis=-1)))
    if none_left.size:
        raise ValueError(
            f"replay: at tick(s) {none_left.tolist()}, legal= leaves no legal policy; at least one "
            "policy must be legal at every tick"
        )
    return legal


def replay_document(
    document: Mapping[str, Any],
    observations: Mapping[str, Array | None],
    actions: Mapping[str, Array],
    *,
    arrays: Mapping[str, np.ndarray] | None = None,
    build_key: Array | None = None,
    **kwargs: Any,
) -> Replay:
    """:func:`replay` from a description document (design: ``replay.md`` R8).

    Builds with :func:`hybrid_aif.build` (full validation, either document layout) and
    replays the result. ``arrays`` is the document's arrays file, ``build_key`` draws the
    values of its ``init`` clamps; other keywords are :func:`replay`'s.
    """
    floor = kwargs.get("support_floor", SUPPORT_FLOOR)
    system, params = build(document, arrays=arrays, key=build_key, support_floor=floor)
    return replay(system, params, observations, actions, **kwargs)


def stack_traces(
    traces: Sequence[
        tuple[Mapping[str, Array | None], Mapping[str, Array], Mapping[str, Array] | None]
    ],
) -> tuple[dict[str, Array | None], dict[str, Array], dict[str, Array], Array]:
    """Pad variable-length traces to one ``T`` and stack them: the batch convention (R7).

    Each trace is ``(observations, actions, present_or_None)``, keyed as :func:`replay` takes
    them, with leading axes of one length. Returns ``(observations, actions, present, valid)``
    with a leading batch axis: padding rows are zeros, absent (``present`` ``False``) and
    invalid (``valid`` ``False``). Replay a batch with
    ``jax.vmap(lambda o, a, m, v: replay(system, params, o, a, present=m, valid=v))``.
    """
    if not traces:
        raise ValueError("stack_traces: no traces")
    for i, (obs, acts, mask) in enumerate(traces):
        for part, kind, noun in (
            (obs, "observations", "modality"),
            (acts, "actions", "action node"),
            (mask, "observations", "modality"),
        ):
            if part is not None and not isinstance(part, Mapping):
                what = "present flags" if part is mask else kind
                raise TypeError(
                    f"stack_traces takes trace {i}'s {what} as a mapping keyed by {noun} name, "
                    f"as replay does; convert positional data with "
                    f"system.from_positional({kind!r}, ...)"
                )
    mods = tuple(traces[0][0])
    actions_keys = tuple(traces[0][1])
    unread = {n: traces[0][0][n] is None for n in mods}  # None: a modality nothing reads
    lengths: list[int] = []
    for i, (obs, acts, mask) in enumerate(traces):
        if set(obs) != set(mods) or set(acts) != set(actions_keys):
            raise ValueError(
                f"stack_traces: trace {i} is keyed {sorted(obs)} / {sorted(acts)}, trace 0 "
                f"{sorted(mods)} / {sorted(actions_keys)}; every trace describes the same "
                "modalities and action nodes"
            )
        if {n: obs[n] is None for n in mods} != unread:
            raise ValueError(
                f"stack_traces: trace {i} has None for different modalities than trace 0; "
                "every trace leaves the same (unread) modalities as None"
            )
        own = {int(jnp.shape(v)[0]) for v in acts.values()}
        own |= {int(jnp.shape(o)[0]) for o in obs.values() if o is not None}
        if not own:
            raise ValueError(
                f"stack_traces: trace {i} has no observation and no action array, so it has no "
                "length to pad"
            )
        if len(own) != 1:
            raise ValueError(
                f"stack_traces: trace {i}'s observations and actions have leading axes "
                f"{sorted(own)}; a trace's leading axes agree"
            )
        n = own.pop()
        if mask is not None:
            unknown = sorted(set(mask) - set(mods))
            if unknown:
                raise ValueError(
                    f"stack_traces: trace {i}'s present flags name unknown modalities {unknown}; "
                    f"its observations are {sorted(mods)}"
                )
            bad = {k: tuple(jnp.shape(v)) for k, v in mask.items() if tuple(jnp.shape(v)) != (n,)}
            if bad:
                raise ValueError(f"stack_traces: trace {i}'s present entries {bad} are not ({n},)")
        lengths.append(n)
    T = max(lengths)

    def pad(x: Array, length: int) -> Array:
        x = jnp.asarray(x)
        return jnp.concatenate([x, jnp.zeros((T - length, *x.shape[1:]), x.dtype)], axis=0)

    def stacked(part: int, name: str) -> Array:
        return jnp.stack([pad(t[part][name], n) for t, n in zip(traces, lengths, strict=True)])

    observations = {n: None if unread[n] else stacked(0, n) for n in mods}
    actions = {k: stacked(1, k) for k in actions_keys}
    masks = {
        name: jnp.stack(
            [
                pad(
                    jnp.ones((n,), bool)
                    if m is None or name not in m
                    else jnp.asarray(m[name], bool),
                    n,
                )
                for (_, _, m), n in zip(traces, lengths, strict=True)
            ]
        )
        for name in mods
    }
    valid = jnp.stack([jnp.arange(T) < n for n in lengths])
    return observations, actions, masks, valid
