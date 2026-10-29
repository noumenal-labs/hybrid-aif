"""Per-family sufficient statistics and gradient evidence for ``System.learn``.

Design: ``docs/design/parameter-learning.md`` (D8, O3) and
``docs/design/amortization.md`` (A1/A2). ``System.learn`` owns the *policy* —
which slots step, at what rate, under what decay — and this module owns the
per-slot arithmetic it steps with, one function per statistic:

- :func:`_lg_noise_stats` — the responsibility-weighted expected residual
  scatter and mass for an ``InverseGamma`` / ``InverseWishart`` LG noise;
- :func:`_lg_joint_stats` — the MNIW regression moments over the augmented
  state, for a coupled head-and-noise posterior;
- :func:`_mlp_evidence` — the ``(child, parents)`` pair a gradient-learnable
  neural slot is scored against (A2's optax path has evidence rather than a
  conjugate statistic, but extracting it is the same per-slot job);
- :func:`_lower_learnable` — the other direction: every learnable declaration
  replaced by its point form, or by one posterior draw per slot, which is how
  ``as_env`` runs a learned model forward.

Switching modalities read the pre-reduction mixture off ``InferenceDetail``
(O3): the reduced belief no longer carries the mode-conditional posterior these
statistics need. A function that reads declared structure (parent order, control
widths, names for its errors) takes the ``System`` explicitly; the two that read
only params or a belief do not. ``System`` keeps a thin method for each.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.controls import LEVEL, node_entry
from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalDirichlet,
    Gaussian,
    InverseGamma,
    Learnable,
    MixtureGaussian,
    MLPConditional,
)
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianObservationMNIW
from hybrid_aif.inference.emission import is_emission
from hybrid_aif.inference.mixture_filter import _per_mode
from hybrid_aif.learning.slots import slot_learnable as _slot_learnable

if TYPE_CHECKING:
    from hybrid_aif.inference.detail import InferenceDetail
    from hybrid_aif.system import Belief, System, SystemParams


# -------------------------------------------------------- posteriors to point values


def _lower_learnable(params: SystemParams, param_key: Array | None) -> SystemParams:
    """Replace every learnable declaration with its point form, or — under
    ``param_key`` — a declaration built from one posterior draw per slot
    (keys split in declaration order: likelihood, transition, prior).

    A neural declaration passes through untouched in both modes: it is
    already a point-valued *function*, with no posterior over its weights to
    average or to draw from (A1). ``param_key`` therefore randomizes nothing
    on a neural slot — the env is the net itself, run forward."""
    learnable = [
        (slot, i)
        for slot in ("likelihood", "transition", "initial_state_prior")
        for i, decl in enumerate(getattr(params, slot))
        if _slot_learnable(decl) and not isinstance(decl, MLPConditional)
    ]
    if not learnable:
        return params
    keys = dict(
        zip(learnable, jax.random.split(param_key, len(learnable)), strict=True)
        if param_key is not None
        else ()
    )

    def lower(slot: str, i: int, decl):
        if isinstance(decl, MLPConditional):
            return decl
        if isinstance(decl, LinearGaussianObservationMNIW):
            return decl.point() if param_key is None else decl.sample(keys[(slot, i)])
        if isinstance(decl, LinearGaussianObservation):
            if not (decl.learnable_noise or decl.learnable_head):
                return decl
            key_i = keys.get((slot, i))
            noise = decl.noise
            if decl.learnable_noise:
                noise = noise.point() if param_key is None else noise.sample(key_i)
            head = decl.head
            if decl.learnable_head:
                head = head.point() if param_key is None else head.sample(key_i, row_chol=noise)
            return LinearGaussianObservation(head=head, noise=noise)
        if not isinstance(decl, Learnable):
            return decl
        if param_key is None:
            return decl.point()
        drawn = decl.sample(keys[(slot, i)])
        if isinstance(decl, ConditionalDirichlet):
            return ConditionalCategorical(log_table=jnp.log(drawn), parent_ndims=decl.parent_ndims)
        return Categorical(logits=jnp.log(drawn))

    return dataclasses.replace(
        params,
        **{
            slot: tuple(lower(slot, i, d) for i, d in enumerate(getattr(params, slot)))
            for slot in ("likelihood", "transition", "initial_state_prior")
        },
    )


# -------------------------------------------------------------- one tick's statistic


def _belief_value(q: Belief) -> Array:
    """A belief's point evidence for a neural slot: the posterior mean.

    Uniform across families by the ``Distribution.mean()`` contract — a
    continuous belief gives its location, a ``Categorical`` its probability
    vector (the one-hot expectation, hence the soft-one-hot convention A1
    documents for discrete parents).
    """
    return q.mean()


def _mlp_evidence(
    system: System,
    slot: str,
    i: int,
    decl: MLPConditional,
    observations: tuple[Array, ...],
    beliefs: tuple[Belief, ...],
    prev_beliefs: tuple[Belief, ...] | None,
    action: Array | None,
    space=None,
) -> tuple[Array, tuple[Array, ...]]:
    """One tick's ``(child, parents)`` evidence for a gradient-learnable slot.

    The slot supplies the parent order — the class is role-free (A1) — and
    ``objectives.hybrid.validate_structure`` has already checked the widths
    against exactly these conventions:

    - ``likelihood``: parents are the ``A_deps`` factors' beliefs in order,
      child is this modality's observation;
    - ``transition``: parents are the factor's own *previous* belief and then its
      control nodes in declared order — a discrete node's one-hot, a continuous
      node's value (A9; ``space`` gives the nodes' kinds and sizes); child is its
      current belief. Discrete
      ``B_deps`` parents are rejected at declaration time (they would grow the
      belief to a mixture — see ``objectives.hybrid.factor_families``).
    """
    if slot == "likelihood":
        parents = tuple(_belief_value(beliefs[g]) for g in system.A_deps[i])
        return jnp.asarray(observations[i]), parents
    child = _belief_value(beliefs[i])
    parents = (_belief_value(prev_beliefs[i]),)
    for k, _ in system._table.control_parents[i]:
        value = node_entry(action, k, space.layout)
        parents += (
            jax.nn.one_hot(value, space.sizes[k], dtype=child.dtype)
            if space.kinds[k] == LEVEL
            else jnp.asarray(value).astype(child.dtype),
        )
    return child, parents


def _joint_responsibility(parents: tuple[Belief, ...]) -> Array:
    """The parents' joint belief ``q(z_1) ⊗ q(z_2) ⊗ …`` (mean-field), ``(*parent_shape,)``: the
    responsibility each parent configuration takes for the tick's datum."""
    out = parents[0].probs
    for q in parents[1:]:
        out = out[..., None] * q.probs
    return out


def _lg_noise_stats(
    system: System,
    m: int,
    a: LinearGaussianObservation,
    y: Array,
    beliefs: tuple[Belief, ...],
    detail: InferenceDetail | None,
) -> tuple[Array, Array]:
    """One tick's noise sufficient statistic for modality ``m`` (D8).

    Returns ``(scatter, count)``: the responsibility-weighted expected
    residual scatter ``E_q[(y − Hx − c)(y − Hx − c)ᵀ]`` — the ``HΣHᵀ``
    term carried in full; drop it and the recovered noise is biased low —
    and the responsibility mass. Diagonal-only for an ``InverseGamma``
    noise, full for ``InverseWishart``.

    Switching modalities (categorical parents in ``A_deps``) need the
    mode-conditional posterior, which the reduced belief no longer
    carries (the ``pad_mixture`` loss, O3): they require ``detail`` from
    ``infer_states(..., detail=True)`` and read the pre-reduction K·M
    mixture, whose slot layout is ``k·M + j`` (component-major).
    """
    deps = system.A_deps[m]
    f, obs_parents = deps[0], deps[1:]
    diag_only = isinstance(a.noise, InverseGamma)
    if is_emission(a):
        # the emission over categorical parents alone: the responsibility is the parents'
        # joint belief, the residual is the datum less the configuration's offset
        w = _joint_responsibility(tuple(beliefs[g] for g in deps))
        c = jnp.broadcast_to(a.c, (*w.shape, a.obs_dim))
        e = y - c
        scatter = w[..., None, None] * jnp.einsum("...o,...p->...op", e, e)
        if diag_only:
            return jnp.diagonal(scatter, axis1=-2, axis2=-1), w
        return scatter, w
    if obs_parents:
        if detail is None or detail.pre_reduce[m] is None:
            mods, _ = (system.modality_names, system._facs)
            raise ValueError(
                f"modality {mods[m]!r}: learning a switching noise needs the "
                "mode-conditional posterior, which the reduced belief no longer "
                "carries; run infer_states(..., detail=True) and pass detail= "
                "(design parameter-learning.md O3/D8)"
            )
        grown = detail.pre_reduce[m]
        parent_sizes = tuple(beliefs[g].probs.shape[-1] for g in obs_parents)
        n_modes = 1
        for size in parent_sizes:
            n_modes *= size
        k = grown.num_components // n_modes
        w = jax.nn.softmax(grown.log_weights).reshape(k, n_modes)
        count = jnp.sum(w, axis=0)  # q(mode config), (M,)
        mu = grown.means.reshape(k, n_modes, -1)
        chol = grown.chols.reshape(k, n_modes, grown.dim, grown.dim)
        d_obs = a.obs_dim
        # H and c are shared or carry one leading axis per switching parent; the
        # per-mode layout is the conditioning kernel's own, so the two cannot drift.
        h = _per_mode(a.H, n_modes, 2)  # (M, d_obs, dx)
        c = _per_mode(a.c, n_modes, 1)  # (M, d_obs)
        e = y - jnp.einsum("mox,kmx->kmo", h, mu) - c  # (K, M, d_obs)
        hl = jnp.einsum("mox,kmxz->kmoz", h, chol)  # H L, (K, M, d_obs, dim)
        scatter_km = jnp.einsum("kmo,kmp->kmop", e, e) + hl @ jnp.swapaxes(hl, -1, -2)
        # JOINT weights (they sum to q(j) over k): the statistic is
        # q(j)·E[·|j], not the conditional — weight the scatter and the
        # count by the same responsibility, or the update is unweighted
        # evidence at weighted count and inflates on off-mode ticks.
        scatter = jnp.einsum("km,kmop->mop", w, scatter_km)
        if diag_only:
            scatter = jnp.diagonal(scatter, axis1=-2, axis2=-1)
            return scatter.reshape(*parent_sizes, d_obs), count.reshape(parent_sizes)
        return (
            scatter.reshape(*parent_sizes, d_obs, d_obs),
            count.reshape(parent_sizes),
        )
    q = beliefs[f]
    if isinstance(q, Gaussian):
        weights = jnp.ones(())
        mu = q.loc[None, :]
        chol = q.chol[None, :, :]
    elif isinstance(q, MixtureGaussian):
        weights = jax.nn.softmax(q.log_weights)
        mu, chol = q.means, q.chols
    else:
        mods, _ = (system.modality_names, system._facs)
        raise ValueError(
            f"modality {mods[m]!r}: LG noise learning needs a Gaussian or mixture "
            f"posterior on the observed factor, got {type(q).__name__}"
        )
    e = y - jnp.einsum("ox,kx->ko", a.H, mu) - a.c  # (K, d_obs)
    hl = jnp.einsum("ox,kxz->koz", a.H, chol)
    scatter_k = jnp.einsum("ko,kp->kop", e, e) + hl @ jnp.swapaxes(hl, -1, -2)
    scatter = jnp.einsum("k,kop->op", jnp.atleast_1d(weights), scatter_k)
    count = jnp.asarray(1.0, dtype=scatter.dtype)
    if diag_only:
        return jnp.diagonal(scatter, axis1=-2, axis2=-1), count
    return scatter, count


def _lg_joint_stats(
    system: System,
    m: int,
    a: LinearGaussianObservationMNIW,
    y: Array,
    beliefs: tuple[Belief, ...],
    detail: InferenceDetail | None,
) -> tuple[Array, Array, Array, Array]:
    """One tick's MNIW statistic for modality ``m`` (D8): the joint,
    responsibility-weighted regression moments over the augmented state
    ``x̃ = [x; 1]`` — ``syy = q·yyᵀ``, ``syx = y ⊗ E[x̃·1_j]``,
    ``sxx = E[x̃x̃ᵀ·1_j]``, and the responsibility mass. No residual is
    formed: the noise scatter emerges inside ``absorb`` from the natural
    parameters, so the ``HΣHᵀ`` bookkeeping is implicit and exact.
    Switching modalities read the pre-reduction mixture (O3), exactly as
    the noise-only statistic does."""
    deps = system.A_deps[m]
    f, obs_parents = deps[0], deps[1:]
    if is_emission(a):
        # zero regressor width: the augmented state is the constant [1], so the moments are
        # the responsibilities themselves (structure-learning.md S2)
        w = _joint_responsibility(tuple(beliefs[g] for g in deps))
        d_obs = y.shape[-1]
        syy = w[..., None, None] * jnp.einsum("o,p->op", y, y)
        syx = w[..., None, None] * y[:, None]
        sxx = w[..., None, None] * jnp.ones((1, 1), y.dtype)
        return syy, syx.reshape(*w.shape, d_obs, 1), sxx, w

    def moments(w, mu, chol):
        # w: (K,) or (K, M); mu: (K[, M], dx); chol: (K[, M], dx, dx)
        ones = jnp.ones((*mu.shape[:-1], 1), dtype=mu.dtype)
        mu_aug = jnp.concatenate([mu, ones], axis=-1)
        sig = chol @ jnp.swapaxes(chol, -1, -2)
        p_aug = mu.shape[-1] + 1
        sig_aug = jnp.zeros((*sig.shape[:-2], p_aug, p_aug), dtype=sig.dtype)
        sig_aug = sig_aug.at[..., :-1, :-1].set(sig)
        outer = mu_aug[..., :, None] * mu_aug[..., None, :]
        return mu_aug, sig_aug + outer

    if obs_parents:
        if detail is None or detail.pre_reduce[m] is None:
            mods, _ = (system.modality_names, system._facs)
            raise ValueError(
                f"modality {mods[m]!r}: learning a switching head/noise needs the "
                "mode-conditional posterior; run infer_states(..., detail=True) and "
                "pass detail= (design parameter-learning.md O3/D8)"
            )
        grown = detail.pre_reduce[m]
        parent_sizes = tuple(beliefs[g].probs.shape[-1] for g in obs_parents)
        n_modes = 1
        for size in parent_sizes:
            n_modes *= size
        k = grown.num_components // n_modes
        w = jax.nn.softmax(grown.log_weights).reshape(k, n_modes)
        mu = grown.means.reshape(k, n_modes, -1)
        chol = grown.chols.reshape(k, n_modes, grown.dim, grown.dim)
        mu_aug, exx_k = moments(w, mu, chol)
        count = jnp.sum(w, axis=0)
        e1 = jnp.einsum("km,kmp->mp", w, mu_aug)
        sxx = jnp.einsum("km,kmpq->mpq", w, exx_k)
        syx = count[:, None, None] * 0.0 + jnp.einsum("o,mp->mop", y, e1)
        syy = count[:, None, None] * jnp.einsum("o,p->op", y, y)
        return (
            syy.reshape(*parent_sizes, y.shape[-1], y.shape[-1]),
            syx.reshape(*parent_sizes, y.shape[-1], mu_aug.shape[-1]),
            sxx.reshape(*parent_sizes, mu_aug.shape[-1], mu_aug.shape[-1]),
            count.reshape(parent_sizes),
        )
    q = beliefs[f]
    if isinstance(q, Gaussian):
        w = jnp.ones((1,))
        mu = q.loc[None, :]
        chol = q.chol[None, :, :]
    elif isinstance(q, MixtureGaussian):
        w = jax.nn.softmax(q.log_weights)
        mu, chol = q.means, q.chols
    else:
        mods, _ = (system.modality_names, system._facs)
        raise ValueError(
            f"modality {mods[m]!r}: MNIW learning needs a Gaussian or mixture "
            f"posterior on the observed factor, got {type(q).__name__}"
        )
    mu_aug, exx_k = moments(w, mu, chol)
    e1 = jnp.einsum("k,kp->p", w, mu_aug)
    sxx = jnp.einsum("k,kpq->pq", w, exx_k)
    syx = jnp.einsum("o,p->op", y, e1)
    syy = jnp.einsum("o,p->op", y, y)
    count = jnp.asarray(1.0, dtype=sxx.dtype)
    return syy, syx, sxx, count
