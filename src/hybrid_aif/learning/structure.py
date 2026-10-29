"""Online structure learning: levels that are born and merged.

A growable factor is a memoryless categorical factor (``B_deps = ()``, no control parent)
whose weights, the transition column ``(K_max, 1)``, are a :class:`ConditionalDirichlet` with a
growth propensity ``growth`` (``α``). Its cap ``K_max`` is its size; a level is live when its
weight concentration is positive and dead otherwise, so liveness is a property of params,
traced and carried by the learn face. A dead level's emission parameters sit at their prior.

**Growth** (S3) is one operation of the learn face, per datum: every level's score is the
posterior predictive log-density of the tick's data under the level, summed over the modalities
that read the factor (the other parents contracted with their beliefs); the first dead level
scores the data under the prior, and a level is woken when the best live score falls below
``ℓ_empty + log α``, the maximum-a-posteriori form of the Chinese-restaurant assignment under a
truncated stick-breaking prior. The woken level takes the datum hard, and the ordinary
responsibility-weighted update then seeds it from the datum alone. :func:`grow_batch` is the same
rule over a window of data (AXIOM Algorithm 1): rounds of scoring every datum under every live
level, waking a level for the worst-explained datum while it falls below the threshold, then one
absorb of the window under the responsibilities the woken levels left.

**Reduction** (S4) merges two levels whose pooled statistics have no lower evidence than their
separate ones: the merged level keeps its own prior once and the killed level's data, the killed
level returns to dead (its weight to zero, its emissions to their prior), in every slot the
factor is an axis of; :func:`reduce` scores candidate pairs by mutual expected log-likelihood and
tests the best ones greedily.

Design: docs/design/structure-learning.md S3, S4
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalDirichlet,
    Dirichlet,
    InverseGamma,
    InverseWishart,
)
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianObservationMNIW
from hybrid_aif.inference.discrete import LOG_ZERO, _observed_log_likelihood
from hybrid_aif.inference.emission import emission_scores, is_emission
from hybrid_aif.learning import evidence as _evidence

if TYPE_CHECKING:
    from hybrid_aif.system import Belief, System, SystemParams


# ------------------------------------------------------------------------- the factors


def growable_factors(system: System, params: SystemParams) -> tuple[int, ...]:
    """The factors whose weights carry a growth propensity, each checked for the form growth
    and reduction read: memoryless, uncontrolled, its weights a ``(K_max, 1)``
    :class:`ConditionalDirichlet` with ``growth``, its initial prior a :class:`Dirichlet` with
    the same ``growth`` and the same dead levels."""
    facs = system._facs
    out = []
    for f, b in enumerate(params.transition):
        if not (isinstance(b, ConditionalDirichlet) and b.growth is not None):
            continue
        name = facs[f]
        if system.B_deps[f] or system._table.control_parents[f] or b.concentration.ndim != 2:
            raise ValueError(
                f"factor {name!r}: its transition carries a growth propensity, and a growable "
                "factor is memoryless and uncontrolled (B_deps = (), no control node), its "
                "weights the (K_max, 1) transition column; drop growth or the parents"
            )
        d = params.initial_state_prior[f]
        if not (isinstance(d, Dirichlet) and d.growth is not None):
            raise ValueError(
                f"factor {name!r}: a growable factor's initial prior mirrors its weights, a "
                "Dirichlet over K_max levels with the same growth propensity and zeros at the "
                f"dead levels; got {type(d).__name__}"
                + ("" if not isinstance(d, Dirichlet) else " without growth")
            )
        _refuse_unbuilt(system, params, f)
        out.append(f)
    return tuple(out)


def _refuse_unbuilt(system: System, params: SystemParams, f: int) -> None:
    """Growth scores and reduction merges what reads a growable factor; refuse what they do not
    handle rather than leave it unscored or unmerged."""
    name = system._facs[f]
    readers = [system._facs[g] for g, deps in enumerate(system.B_deps) if f in deps]
    readers += [system._facs[g] for g, deps in enumerate(system._effective_D_deps()) if f in deps]
    if readers:
        raise NotImplementedError(
            f"factor {name!r} is growable and parents the transition or initial prior of "
            f"{sorted(set(readers))}; growth and reduction act on the modalities over a growable "
            "factor, and a factor it parents is not built (design: structure-learning.md S3, S4)"
        )
    for m, deps in enumerate(system.A_deps):
        if f not in deps:
            continue
        decl, mod = params.likelihood[m], system.modality_names[m]
        if isinstance(decl, ConditionalDirichlet | ConditionalCategorical):
            continue
        if is_emission(decl):
            if isinstance(decl, LinearGaussianObservation):
                raise NotImplementedError(
                    f"modality {mod!r} over the growable factor {name!r} is a point emission: its "
                    "levels' offsets are fixed, so a woken level cannot be seeded from its datum, "
                    "and a merge has no emission evidence to weigh against the weights' (a "
                    "learnable noise would pool scatter taken about two offsets); declare it with "
                    "gaussian_emission (the MNIW family)"
                )
            continue
        raise NotImplementedError(
            f"modality {mod!r} reads the growable factor {name!r} and is a "
            f"{type(decl).__name__}; growth scores a growable factor's tabular modalities and "
            "its Gaussian emissions over categorical parents alone, and a switching or neural "
            "modality over it is not built (design: structure-learning.md S3)"
        )


def live_levels(weights: ConditionalDirichlet) -> Array:
    """``(K_max,)`` bool: the levels whose weight concentration is positive."""
    return weights.concentration[:, 0] > 0.0


# -------------------------------------------------------------------------- the scores


def _modality_scores(decl, y: Array, parent_shape: tuple[int, ...]) -> Array | None:
    """One modality's predictive log-density of its datum per parent configuration,
    ``(*parent_shape,)``; ``None`` for a family without a closed form."""
    if is_emission(decl):
        return jnp.broadcast_to(emission_scores(decl, y), parent_shape)
    if isinstance(decl, ConditionalDirichlet):
        table = jnp.maximum(jnp.log(decl.mean()), LOG_ZERO)
        return _observed_log_likelihood(table, jnp.asarray(y))
    if isinstance(decl, ConditionalCategorical):
        return _observed_log_likelihood(jnp.maximum(decl.log_probs, LOG_ZERO), jnp.asarray(y))
    return None


def level_scores(
    system: System,
    params: SystemParams,
    f: int,
    observations: tuple[Array, ...],
    beliefs: tuple[Belief, ...],
    present: tuple[bool | Array, ...] | None = None,
) -> tuple[Array, Array]:
    """``(ℓ, observed)``: ``ℓ_j`` for every level ``j`` of factor ``f``, the tick's data scored
    under the level by every modality that reads the factor, each modality's other parents
    contracted with their beliefs (mean-field), summed; and whether any of those modalities is
    present at the tick. An absent modality contributes nothing."""
    k = params.transition[f].concentration.shape[0]
    total = jnp.zeros((k,))
    observed = jnp.asarray(False)
    for m, decl in enumerate(params.likelihood):
        deps = system.A_deps[m]
        if f not in deps:
            continue
        flag = True if present is None else present[m]
        if flag is False:
            continue
        shape = tuple(beliefs[g].probs.shape[-1] for g in deps)
        full = _modality_scores(decl, observations[m], shape)
        if full is None:
            continue
        axis = deps.index(f)
        for pos in reversed(range(len(deps))):
            if pos == axis:
                continue
            full = jnp.tensordot(full, beliefs[deps[pos]].probs, axes=([pos], [0]))
            if pos < axis:
                axis -= 1
        if flag is not True:
            full = jnp.where(jnp.asarray(flag), full, jnp.zeros_like(full))
        total = total + full
        observed = observed | jnp.asarray(flag)
    return total, observed


def growth_decision(scores: Array, live: Array, log_growth: Array) -> tuple[Array, Array]:
    """``(grow, woken)``: whether the datum wakes a level, and which (the first dead one).

    ``grow`` holds when a dead level exists and the best live score is below the dead level's
    (the score under the prior) plus ``log α``.
    """
    dead = ~live
    woken = jnp.argmax(dead)
    best_live = jnp.max(jnp.where(live, scores, -jnp.inf))
    return jnp.any(dead) & (best_live < scores[woken] + log_growth), woken


def hard_assignment(k: int, level: Array) -> Categorical:
    """The one-hot belief on ``level`` over ``k`` levels."""
    return Categorical(logits=jnp.where(jnp.arange(k) == level, 0.0, -jnp.inf))


class Growth:
    """What the per-tick growth step decided, per growable factor: ``grew`` and ``woken``."""

    def __init__(self, grew: dict[int, Array], woken: dict[int, Array]):
        self.grew, self.woken = grew, woken


def grow_beliefs(
    system: System,
    params: SystemParams,
    observations: tuple[Array, ...],
    beliefs: tuple[Belief, ...],
    present: tuple[bool | Array, ...] | None,
    gates: dict[int, bool | Array],
) -> tuple[tuple[Belief, ...], Growth]:
    """The per-tick growth step for the growable factors ``gates`` names, each gated by its
    flag: the beliefs with the woken level hard-assigned where a level is woken, and what was
    decided. A tick with no modality over the factor present wakes nothing."""
    out = list(beliefs)
    grew: dict[int, Array] = {}
    woken: dict[int, Array] = {}
    for f, gate in gates.items():
        weights = params.transition[f]
        scores, observed = level_scores(system, params, f, observations, beliefs, present)
        grow, level = growth_decision(scores, live_levels(weights), jnp.log(weights.growth))
        grow = grow & observed & jnp.asarray(gate)
        k = weights.concentration.shape[0]
        hard = hard_assignment(k, level)
        out[f] = Categorical(logits=jnp.where(grow, hard.logits, beliefs[f].logits))
        grew[f], woken[f] = grow, level
    return tuple(out), Growth(grew, woken)


def wake_weights(weights: ConditionalDirichlet, growth: Growth, f: int) -> ConditionalDirichlet:
    """The weights with the growth propensity added at the woken level: a live level's count
    is the prior's pseudo-count for a new level plus the data it took."""
    if f not in growth.grew:
        return weights
    k = weights.concentration.shape[0]
    at = (jnp.arange(k) == growth.woken[f])[:, None]
    conc = jnp.where(
        growth.grew[f] & at, weights.concentration + weights.growth, weights.concentration
    )
    return dataclasses.replace(weights, concentration=conc)


def mirror_initial_prior(d: Dirichlet, weights: ConditionalDirichlet, growth: Growth, f: int):
    """The initial prior with the woken level's count copied from the weights, so a new
    episode starts with the level live."""
    if f not in growth.grew:
        return d
    k = d.concentration.shape[0]
    at = jnp.arange(k) == growth.woken[f]
    conc = jnp.where(growth.grew[f] & at, weights.concentration[:, 0], d.concentration)
    return dataclasses.replace(d, concentration=conc)


# ----------------------------------------------------------------------- the batch form


def _own_modalities(system: System, f: int) -> tuple[int, ...]:
    return tuple(m for m, deps in enumerate(system.A_deps) if deps == (f,))


def grow_batch(
    system: System,
    params: SystemParams,
    observations: tuple[Array, ...],
    f: int,
    rounds: int,
    lr: float | Array,
) -> SystemParams:
    """Growth over a window of ``N`` data for factor ``f`` (AXIOM Algorithm 1).

    ``observations[m]`` is ``(N, …)`` for every modality over ``f`` alone, the others unread. Each
    of ``rounds`` rounds scores every datum under every level of the working copy, takes the
    worst-explained datum and, when its best live score falls below the growth threshold, wakes
    the first dead level for it, seeded from that datum alone, which then takes the datum hard;
    the responsibilities of the window under the (seeded) working copy are then absorbed into
    ``params`` at rate ``lr``, which is the next round's working copy. The rounds are sequential
    because a second level is needed only given the first. A woken level is its prior plus the
    data it took, its weight the growth propensity plus that data, and the initial prior holds
    it live as the weights do. One datum and one round is the learn face's growth step.
    """
    mods = _own_modalities(system, f)
    if not mods:
        raise ValueError(
            f"factor {system._facs[f]!r}: the batch form of growth reads the modalities over "
            "the factor alone (A_deps == (f,)), and none is declared"
        )
    for m, deps in enumerate(system.A_deps):
        if f in deps and m not in mods:
            raise NotImplementedError(
                f"modality {system.modality_names[m]!r} reads factor {system._facs[f]!r} beside "
                "other parents; the batch form of growth scores a factor's own modalities, and a "
                "modality with other parents is the per-tick form's (System.learn)"
            )
    n = int(jnp.shape(observations[mods[0]])[0])
    weights = params.transition[f]
    k = weights.concentration.shape[0]
    log_growth = jnp.log(weights.growth)
    n_facs, n_mods = len(params.transition), len(params.likelihood)
    present = tuple(m in mods for m in range(n_mods))
    own_rates = {"*:*": 0.0, **_own_slots(system, f, mods, lr)}

    def scores_of(working: SystemParams) -> Array:
        """``(N, K)``: every datum under every level of the working copy."""
        total = jnp.zeros((n, k))
        for m in mods:
            decl = working.likelihood[m]
            total = total + jax.vmap(lambda y, d=decl: _modality_scores(d, y, (k,)))(
                observations[m]
            )
        return total

    def responsibilities(working: SystemParams, hard: Array) -> Array:
        """``(N, K)``: the posterior over the level per datum under the working copy, a datum
        a woken level took held at its one-hot."""
        live = live_levels(working.transition[f])
        log_w = jnp.where(live, jnp.log(working.transition[f].mean()[:, 0]), -jnp.inf)
        logits = log_w[None, :] + scores_of(working)
        soft = jax.nn.softmax(jnp.where(live[None, :], logits, -jnp.inf), axis=-1)
        taken = jnp.any(hard, axis=-1)
        return jnp.where(taken[:, None], hard, soft)

    def wake(working: SystemParams, level: Array, y_index: Array) -> SystemParams:
        """The working copy with ``level`` seeded from datum ``y_index`` alone."""
        onehot = hard_assignment(k, level)
        beliefs = tuple(onehot if g == f else None for g in range(n_facs))
        obs = tuple(
            jnp.take(observations[m], y_index, axis=0) if m in mods else None for m in range(n_mods)
        )
        stepped = system._learn(
            working,
            obs,
            beliefs,
            prev_beliefs=beliefs,
            lr_by_slot={
                "*:*": 0.0,
                **{f"likelihood:{system.modality_names[m]}": 1.0 for m in mods},
            },
            face="grow",
            present=present,
            grow=False,
        )
        if isinstance(stepped, tuple):
            stepped = stepped[0]
        b = working.transition[f]
        conc = b.concentration.at[level, 0].set(b.growth + 1.0)
        return dataclasses.replace(
            stepped,
            transition=tuple(
                dataclasses.replace(x, concentration=conc) if g == f else x
                for g, x in enumerate(stepped.transition)
            ),
        )

    def absorb_window(resp: Array) -> SystemParams:
        """``params`` absorbing the window under ``resp``: the woken levels (those the window
        gave data) carry the growth propensity too, and the initial prior mirrors them."""

        def absorb_one(acc: SystemParams, i: Array) -> SystemParams:
            belief = Categorical(logits=jnp.log(resp[i]))
            beliefs_i = tuple(belief if g == f else None for g in range(n_facs))
            obs_i = tuple(observations[m][i] if m in mods else None for m in range(n_mods))
            stepped = system._learn(
                acc,
                obs_i,
                beliefs_i,
                prev_beliefs=beliefs_i,
                lr=lr,
                lr_by_slot=own_rates,
                face="grow",
                present=present,
                grow=False,
            )
            return stepped[0] if isinstance(stepped, tuple) else stepped

        out = jax.lax.fori_loop(0, n, lambda i, acc: absorb_one(acc, i), params)
        woken = live_levels(out.transition[f]) & ~live_levels(params.transition[f])
        b = out.transition[f]
        b_conc = jnp.where(woken[:, None], b.concentration + b.growth, b.concentration)
        d = out.initial_state_prior[f]
        d_conc = jnp.where(woken, b_conc[:, 0], d.concentration)
        return dataclasses.replace(
            out,
            transition=tuple(
                dataclasses.replace(x, concentration=b_conc) if g == f else x
                for g, x in enumerate(out.transition)
            ),
            initial_state_prior=tuple(
                dataclasses.replace(x, concentration=d_conc) if g == f else x
                for g, x in enumerate(out.initial_state_prior)
            ),
        )

    def round_step(carry, _):
        working, hard = carry
        live = live_levels(working.transition[f])
        scores = scores_of(working)
        best = jnp.max(jnp.where(live[None, :], scores, -jnp.inf), axis=-1)  # (N,)
        worst = jnp.argmin(jnp.where(jnp.any(hard, axis=-1), jnp.inf, best))
        grow, level = growth_decision(scores[worst], live, log_growth)
        seeded = wake(working, level, worst)
        working = jax.tree_util.tree_map(lambda a, b: jnp.where(grow, a, b), seeded, working)
        hard = jnp.where(grow, hard.at[worst].set(jnp.arange(k) == level), hard)
        return (absorb_window(responsibilities(working, hard)), hard), None

    hard0 = jnp.zeros((n, k), bool)
    (working, _), _ = jax.lax.scan(round_step, (params, hard0), None, length=int(rounds))
    return working


def _own_slots(system: System, f: int, mods: tuple[int, ...], lr) -> dict[str, float | Array]:
    """The slots a window absorbs into: the factor's modalities, its weights and its initial
    prior, which the per-tick form absorbs alike."""
    out = {f"likelihood:{system.modality_names[m]}": lr for m in mods}
    out[f"transition:{system._facs[f]}"] = lr
    out[f"initial_state_prior:{system._facs[f]}"] = lr
    return out


# --------------------------------------------------------------------------- reduction


def _part(decl):
    """The object holding a slot's conjugate statistics: the declaration, or the noise posterior
    of a point linear-Gaussian head."""
    if isinstance(decl, LinearGaussianObservation) and decl.learnable_noise:
        return decl.noise
    return decl


def _with_part(decl, part):
    if isinstance(decl, LinearGaussianObservation):
        return dataclasses.replace(decl, noise=part)
    return part


def _natural(decl):
    """A conjugate slot's natural coordinates, the data a posterior holds additive in them."""
    decl = _part(decl)
    if isinstance(decl, Dirichlet | ConditionalDirichlet):
        return (decl.concentration,)
    if isinstance(decl, LinearGaussianObservationMNIW):
        nat_mean = decl.M @ decl.V_inv
        return (decl.V_inv, nat_mean, decl.Psi + nat_mean @ jnp.swapaxes(decl.M, -1, -2), decl.nu)
    if isinstance(decl, InverseWishart):
        return (decl.Psi, decl.nu)
    if isinstance(decl, InverseGamma):
        return (decl.alpha, decl.beta)
    return None


def _from_natural(decl, nat):
    part = _part(decl)
    if isinstance(part, Dirichlet | ConditionalDirichlet):
        return dataclasses.replace(part, concentration=nat[0])
    if isinstance(part, LinearGaussianObservationMNIW):
        v_inv, nat_mean, phi, nu = nat
        m = jnp.swapaxes(
            jnp.linalg.solve(jnp.swapaxes(v_inv, -1, -2), jnp.swapaxes(nat_mean, -1, -2)), -1, -2
        )
        return dataclasses.replace(
            part, M=m, V_inv=v_inv, Psi=phi - nat_mean @ jnp.swapaxes(m, -1, -2), nu=nu
        )
    if isinstance(part, InverseWishart):
        return _with_part(decl, dataclasses.replace(part, Psi=nat[0], nu=nat[1]))
    return _with_part(decl, dataclasses.replace(part, alpha=nat[0], beta=nat[1]))


def _axes_of(system: System, slot: str, index: int, decl, f: int) -> tuple[tuple[int, bool], ...]:
    """The axes of a slot's arrays that run over factor ``f``'s levels, as ``(axis, child)``:
    ``child`` when the axis is the factor's own distribution (its weights, its initial prior)."""
    if slot == "likelihood":
        deps = system.A_deps[index]
        if f not in deps:
            return ()
        pos = deps.index(f)
        if isinstance(decl, LinearGaussianObservation | LinearGaussianObservationMNIW):
            return ((pos, False),)  # parent axes lead
        return ((1 + pos, False),)  # child-first tables
    if slot == "transition":
        deps = system.B_deps[index]
        out = [(0, True)] if index == f else []
        out += [(1 + i, False) for i, g in enumerate(deps) if g == f]
        return tuple(out)
    deps = system._effective_D_deps()[index]
    out = [(0, True)] if index == f else []
    out += [(1 + i, False) for i, g in enumerate(deps) if g == f]
    return tuple(out)


def _pool(arr: Array, prior: Array, axis: int, j: Array, k: Array, child: bool, kill) -> Array:
    """``arr`` with level ``k``'s data (beyond ``prior``) pooled into level ``j`` along ``axis``,
    level ``k`` reset to ``kill`` (``0`` on a child axis, its prior on a parent axis)."""
    data = arr - prior
    take = jnp.take(data, k, axis=axis)
    idx = jnp.arange(arr.shape[axis])
    shape = [1] * arr.ndim
    shape[axis] = arr.shape[axis]
    is_j = (idx == j).reshape(shape)
    is_k = (idx == k).reshape(shape)
    pooled = jnp.where(is_j, data + jnp.expand_dims(take, axis), data)
    merged = prior + pooled
    return jnp.where(is_k, kill, merged)


def _effective_prior(decl, prior) -> tuple[Array, ...] | None:
    """The prior's natural coordinates with a woken level's pseudo-count where the prior is
    dead and the posterior live."""
    nat_prior = _natural(prior)
    if nat_prior is None:
        return None
    if isinstance(decl, Dirichlet | ConditionalDirichlet) and decl.growth is not None:
        conc = nat_prior[0]
        post = decl.concentration
        return (jnp.where((conc == 0.0) & (post > 0.0), decl.growth, conc),)
    return nat_prior


def merge_levels(
    system: System, params: SystemParams, prior: SystemParams, f: int, j: Array, k: Array
) -> SystemParams:
    """``params`` with level ``k`` of factor ``f`` merged into level ``j`` in every slot the
    factor is an axis of: the merged level keeps its own prior and the data of both, the killed
    level returns to dead."""
    new = {}
    for slot in ("likelihood", "transition", "initial_state_prior"):
        decls = list(getattr(params, slot))
        for i, decl in enumerate(decls):
            axes = _axes_of(system, slot, i, decl, f)
            if not axes:
                continue
            nat = _natural(decl)
            if nat is None:
                continue
            nat_prior = _effective_prior(decl, getattr(prior, slot)[i])
            merged = []
            for arr, p in zip(nat, nat_prior, strict=True):
                for axis, child in axes:
                    arr = _pool(arr, p, axis, j, k, child, 0.0 if child else p)
                merged.append(arr)
            decls[i] = _from_natural(decl, tuple(merged))
        new[slot] = tuple(decls)
    return dataclasses.replace(params, **new)


def factor_evidence(system: System, params: SystemParams, prior: SystemParams, f: int) -> Array:
    """The summed evidence of every slot factor ``f`` is an axis of, its own initial prior
    excepted: that prior mirrors the weights (the woken levels copied, the counts absorbed
    alike by default), so scoring it would count the weights' evidence twice."""
    total = jnp.zeros(())
    for slot in ("likelihood", "transition", "initial_state_prior"):
        for i, decl in enumerate(getattr(params, slot)):
            if slot == "initial_state_prior" and i == f:
                continue
            if not _axes_of(system, slot, i, decl, f):
                continue
            nats = _evidence.log_evidence(getattr(prior, slot)[i], decl)
            if nats is None:
                continue
            total = total + jnp.sum(nats)
    return total


def pair_scores(system: System, params: SystemParams, f: int) -> Array:
    """``(K, K)``: the mutual expected log-likelihood of every pair of live levels (each level's
    mean datum scored under the other's predictive, summed over the modalities over the factor
    alone), ``-inf`` on the diagonal, on dead levels and below the diagonal."""
    weights = params.transition[f]
    k = weights.concentration.shape[0]
    live = live_levels(weights)
    total = jnp.zeros((k, k))
    for m in _own_modalities(system, f):
        decl = params.likelihood[m]
        if is_emission(decl):
            means = jnp.broadcast_to(decl.c, (k, decl.obs_dim))  # (K, d), a shared offset too
            scored = jax.vmap(lambda y, d=decl: emission_scores(d, y))(means)  # [j, k] = ℓ_k(μ_j)
        elif isinstance(decl, ConditionalDirichlet | ConditionalCategorical):
            table = (
                decl.mean() if isinstance(decl, ConditionalDirichlet) else jnp.exp(decl.log_probs)
            )
            log_table = jnp.maximum(jnp.log(table), LOG_ZERO)  # (num_obs, K)
            scored = table.T @ log_table  # [j, k] = Σ_y mean_j(y) log p_k(y)
        else:
            continue
        total = total + scored + scored.T
    idx = jnp.arange(k)
    upper = idx[:, None] < idx[None, :]
    return jnp.where(upper & live[:, None] & live[None, :], total, -jnp.inf)


def reduce(
    system: System,
    params: SystemParams,
    prior: SystemParams,
    f: int,
    n_pair: int,
) -> SystemParams:
    """Bayesian model reduction (Friston et al., arXiv:1805.07092), as SPM's
    ``spm_MDP_VB_sleep``/``spm_MDP_log_evidence`` apply it to Dirichlet counts. The test here
    compares the pooled level's evidence with the two levels' own (AXIOM's merge, Heins et al.
    2025, A.10) rather than scoring a reduced prior with ``spm_MDP_log_evidence``.

    On factor ``f`` the ``n_pair`` best-scoring live pairs are tested greedily, a merge kept when
    the merged slots' evidence is no lower than the separate levels'.
    """
    k = params.transition[f].concentration.shape[0]
    scores = pair_scores(system, params, f)
    flat = scores.reshape(-1)
    _, order = jax.lax.top_k(flat, min(int(n_pair), int(flat.shape[0])))
    pairs = jnp.stack([order // k, order % k], axis=-1)

    def try_pair(acc: SystemParams, pair: Array) -> SystemParams:
        j, kk = pair[0], pair[1]
        live = live_levels(acc.transition[f])
        candidate = merge_levels(system, acc, prior, f, j, kk)
        gain = factor_evidence(system, candidate, prior, f) - factor_evidence(system, acc, prior, f)
        accept = live[j] & live[kk] & (j != kk) & (gain >= 0.0) & jnp.isfinite(scores[j, kk])
        return jax.tree_util.tree_map(lambda a, b: jnp.where(accept, a, b), candidate, acc)

    def body(i, acc):
        return try_pair(acc, pairs[i])

    return jax.lax.fori_loop(0, pairs.shape[0], body, params)
