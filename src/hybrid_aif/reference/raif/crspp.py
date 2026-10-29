"""The CRSPP (prior-preference model): its recognition scan, its loss and its scoring row.

The loss, with ``rho`` from :func:`hybrid_aif.reference.raif.revise.pref_labels`::

    L = mean_t[ gate_t · KL_balanced(q2_t ‖ p2_t) ]
        + contrast · mean_t[ −rho_t · cos_sim(C_post_t, sg(post_t)) ]
    gate_t = max(0, sgn rho_t)          contrast = 1.0

The CRSPP is a declared System with no decoder and no action parent. Its posterior
conditions on ``("deter", "external:embed")``, the agent's encoder output handed over as a
vector (:func:`crspp_recognize`), and the resulting beliefs reach the scoring row as
``aux_beliefs["cosine_sim_to_pref_rollout"]``, keyed by the CRSPP's factor names.

Design: ``docs/design/amortization-p0-addendum.md`` §4.
"""

from collections.abc import Mapping
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from hybrid_aif.distributions.neural import (
    MLPCategorical,
    kl_balance,
    straight_through_onehot,
)
from hybrid_aif.factors import Deterministic
from hybrid_aif.inference import amortized as _amortized
from hybrid_aif.interval import Interval
from hybrid_aif.objectives import registry
from hybrid_aif.objectives.amortized import features
from hybrid_aif.objectives.base import register_term
from hybrid_aif.reference.raif.revise import positive_gate
from hybrid_aif.system import LearnState, System, SystemParams
from hybrid_aif.train.world import assemble

#: Their preference-model optimizer is the **model** group's: ``adam 1e-4``,
#: ``clip 1000`` (the CRSPP nets are trained inside the world-model loss's optimizer
#: scope, ``configs.yaml:76``).
DEFAULT_LR = 1e-4
DEFAULT_CLIP = 1000.0

#: ``loss_scales.contrast`` (``configs.yaml``; addendum §4).
DEFAULT_CONTRAST = 1.0

#: The row whose parameters this trainer steps, and the address that follows.
ROW = "cosine_sim_to_pref_rollout"
STATE_KEY = f"scoring:{ROW}"

#: The CRSPP's parts, in **canonical order** — the order its one global-norm clip
#: sees, stated once for the reason :data:`hybrid_aif.inference.rssm.PART_FIELDS` is.
PART_ORDER: tuple[str, ...] = ("gru", "prior", "posterior", "initial")


class CRSPPParts(NamedTuple):
    """The CRSPP's declarations, resolved: the parts its scan runs and their slot addresses."""

    gru: Deterministic
    prior: MLPCategorical
    posterior: MLPCategorical
    initial: object
    embed_name: str
    slots: tuple[str, ...]

    @property
    def parts(self) -> tuple[object, ...]:
        """The four trainable parts, in the order :attr:`slots` addresses them."""
        return tuple(getattr(self, name) for name in PART_ORDER)

    def with_parts(self, replaced: dict[int, object]) -> "CRSPPParts":
        """A copy with the parts at those positions replaced (the gradient's swap)."""
        return self._replace(**{PART_ORDER[i]: part for i, part in replaced.items()})


def resolve(system: System, params: SystemParams) -> CRSPPParts:
    """The CRSPP's parts and slot addresses, validated against A12 item 2's shape.

    The declaration must be one uncontrolled carry, one code whose parent is that carry, no
    modality, and one recognition entry over ``(carry, external embedding)``; anything else
    raises.
    """
    params = system._p(params)
    structure = system._amortized_structure(params)
    facs = system.names.states
    carries = [f for f, fam in enumerate(structure.families) if fam == _amortized.CARRY]
    codes = [f for f, fam in enumerate(structure.families) if fam == _amortized.CODE]
    if len(carries) != 1 or len(codes) != 1 or params.likelihood:
        raise ValueError(
            "the CRSPP is one carry, one code and no modality at all (it has no decoder of "
            f"its own — addendum §2/IDD §J); this System declares carries "
            f"{[facs[f] for f in carries]}, codes {[facs[f] for f in codes]} and "
            f"{len(params.likelihood)} modality/ies"
        )
    carry, code = carries[0], codes[0]
    if system.names.controls:
        readers = [
            f
            for f, deps in zip(facs, system.B_deps, strict=False)
            if set(deps) & set(system.controls)
        ]
        raise ValueError(
            "the CRSPP's whole content is that the action does NOT enter its transition "
            f"prior (paper §3.2), but factor(s) {readers} read the control node(s) "
            f"{list(system.names.controls)}"
        )
    if len(structure.plan) != 1 or structure.plan[0].factor != code:
        raise ValueError(
            "the CRSPP's recognition side is exactly one q-side factor — the posterior over "
            f"its code — got {[p.variable for p in structure.plan]}. Its posterior is their "
            "obs_out_2/obs_stats_2 (nets.py:189,220)."
        )
    parents = structure.plan[0].parents
    external = [ref for kind, ref in parents if kind == _amortized.EXTERNAL]
    if [kind for kind, _ in parents] != [_amortized.STATE, _amortized.EXTERNAL]:
        raise ValueError(
            "the CRSPP's posterior conditions on (its own carry, the AGENT's observation "
            f"embedding as an external input) in that order — got {list(parents)}. The "
            "embedding crosses as data, never as a shared module (design: amortization.md "
            "A11 refinement (ii); amortization-debt-ledger.md entry 5)."
        )
    return CRSPPParts(
        gru=params.transition[carry],
        prior=params.transition[code],
        posterior=params.recognition[structure.plan[0].variable],
        initial=params.initial_state_prior[carry],
        embed_name=external[0],
        slots=(
            f"transition:{facs[carry]}",
            f"transition:{facs[code]}",
            f"recognition:{structure.plan[0].variable}",
            f"initial_state_prior:{facs[carry]}",
        ),
    )


def _features(deter: Array, code: Array) -> Array:
    """``[deter, flat stoch]`` — the vector the cosine similarity compares on."""
    return jnp.concatenate([deter, code.reshape(-1)], axis=-1)


def observe_no_action(
    parts: CRSPPParts, key: Array, embeds: Array, is_first: Array | None = None
) -> tuple[Array, Array, Array, Array]:
    """Their ``observe_no_action``: filter one embedding sequence, no actions.

    ``embeds`` is ``(T, embed_dim)``. ``is_first``, optional ``(T,)`` bool, marks episode
    starts inside the sequence: there the carry is replaced by the CRSPP's own initial state
    before the step, as their ``obs_step_no_action`` does. The scan always starts from that
    state. Returns ``(deters, codes, post_logits, prior_logits)`` stacked over time; the
    carried posterior code is a straight-through sample. Unbatched.
    """
    embeds = jnp.asarray(embeds)
    if embeds.ndim != 2:
        raise ValueError(f"observe_no_action takes ONE sequence (T, embed), got {embeds.shape}")
    deter0 = parts.initial()
    code0 = parts.prior.mode((deter0,))
    if is_first is None:
        first = jnp.zeros((embeds.shape[0],), jnp.bool_)
    else:
        first = jnp.asarray(is_first).astype(jnp.bool_)
        if first.shape != (embeds.shape[0],):
            raise ValueError(f"is_first has shape {first.shape}, expected ({embeds.shape[0]},)")

    def step(carry, xs):
        deter, code = carry
        k, embed, restart = xs
        deter = jnp.where(restart, deter0, deter)
        code = jnp.where(restart, code0, code)
        deter = parts.gru(deter, code.reshape(-1))  # no action parent: the whole point
        prior_logits = parts.prior.logits((deter,))
        post_logits = parts.posterior.logits((deter, embed))
        code = straight_through_onehot(k, post_logits)
        return (deter, code), (deter, code, post_logits, prior_logits)

    keys = jax.random.split(key, embeds.shape[0])
    _, out = jax.lax.scan(step, (deter0, code0), (keys, embeds, first))
    return out


def img_step_no_action(parts: CRSPPParts, key: Array, deter: Array, code: Array):
    """One *imagined* CRSPP step: their ``img_step_no_action``. ``(deter', code')``."""
    deter = parts.gru(deter, code.reshape(-1))
    logits = parts.prior.logits((deter,))
    return deter, straight_through_onehot(key, logits)


def _batch_is_first(is_first: Array | None, shape: tuple[int, int]) -> Array:
    """``is_first`` as a ``(B, T)`` bool array, all-False when absent."""
    if is_first is None:
        return jnp.zeros(shape, jnp.bool_)
    first = jnp.asarray(is_first).astype(jnp.bool_)
    if first.shape != shape:
        raise ValueError(f"is_first has shape {first.shape}, expected {shape}")
    return first


def crspp_recognize(
    system: System,
    params: SystemParams,
    embeds: Array,
    *,
    key: Array,
    is_first: Array | None,
) -> tuple[Array, ...]:
    """The CRSPP posterior over a batch's embeddings, as belief tuples.

    ``embeds`` is ``(B, T, embed_dim)`` (:func:`embeddings`). ``is_first`` ``(B, T)`` restarts
    the filter at episode starts (:func:`observe_no_action`); it is required, and ``None``
    declares that every window is one episode. Returns
    the positional belief tuple in the CRSPP's declaration order with leaves ``(B, T, ...)``;
    ``aux_beliefs`` takes it keyed: ``system.from_positional("beliefs", ...)``.
    """
    parts = resolve(system, params)
    structure = system._amortized_structure(params)
    embeds = jnp.asarray(embeds)
    if embeds.ndim != 3:
        raise ValueError(f"crspp_recognize takes (B, T, embed), got {embeds.shape}")
    keys = jax.random.split(key, embeds.shape[0])
    first = _batch_is_first(is_first, embeds.shape[:2])
    deters, codes, _, _ = jax.vmap(lambda k, e, f: observe_no_action(parts, k, e, f))(
        keys, embeds, first
    )
    values = {_amortized.CARRY: deters, _amortized.CODE: codes}
    return tuple(values[family] for family in structure.families)


class CRSPPLosses(NamedTuple):
    """The preference model's loss, decomposed.

    ``total`` is what the gradient is taken of. ``kl`` is the gated, averaged balanced KL and
    ``kl_raw`` the same before the gate. ``contrast`` is the cosine term before its scale.
    ``positive`` is the fraction of steps the gate admitted.
    """

    total: Array
    kl: Array
    kl_raw: Array
    contrast: Array
    positive: Array


def crspp_loss(
    parts: CRSPPParts,
    key: Array,
    embeds: Array,
    rho: Array,
    agent_features: Array,
    *,
    is_first: Array | None,
    contrast: float = DEFAULT_CONTRAST,
    dyn_scale: float = 0.5,
    rep_scale: float = 0.1,
    free_nats: float = 1.0,
    return_states: bool = False,
) -> tuple[Array, CRSPPLosses] | tuple[Array, tuple[CRSPPLosses, tuple[Array, Array]]]:
    """The CRSPP loss on one batch: ``(total, CRSPPLosses)``. Differentiable in ``parts``.

    ``embeds`` ``(B, T, embed)``, ``rho`` ``(B, T)``, ``agent_features`` ``(B, T,
    feature_dim)``: the agent's ``[deter, flat stoch]`` at the same steps, compared under a
    stop-gradient. ``is_first`` ``(B, T)`` restarts the filter at episode starts
    (:func:`observe_no_action`); it is required, and ``None`` declares that every window is one
    episode. With ``return_states=True`` the auxiliary is
    ``(CRSPPLosses, (deters, codes))``, the pass's own ``(B, T, ...)`` carries and codes, not
    gradient-stopped.
    """
    embeds, rho = jnp.asarray(embeds), jnp.asarray(rho)
    agent_features = jax.lax.stop_gradient(jnp.asarray(agent_features))
    if embeds.ndim != 3 or rho.shape != embeds.shape[:2]:
        raise ValueError(
            f"crspp_loss wants embeds (B, T, embed) and rho (B, T), got {embeds.shape} and "
            f"{rho.shape}"
        )
    b = embeds.shape[0]
    first = _batch_is_first(is_first, embeds.shape[:2])

    def one(k: Array, embed: Array, f: Array) -> tuple[Array, Array, Array, Array, Array]:
        deters, codes, post_logits, prior_logits = observe_no_action(parts, k, embed, f)
        kl = kl_balance(
            post_logits,
            prior_logits,
            dyn_scale=dyn_scale,
            rep_scale=rep_scale,
            free_nats=free_nats,
        )
        feats = jax.vmap(_features)(deters, codes)
        return kl.total, kl.dyn_raw + kl.rep_raw, feats, deters, codes

    kl_total, kl_raw, feats, deters, codes = jax.vmap(one)(jax.random.split(key, b), embeds, first)
    gate = positive_gate(rho)
    if feats.shape[-1] != agent_features.shape[-1]:
        raise ValueError(
            f"the CRSPP's feature vector is {feats.shape[-1]} numbers wide but the agent's is "
            f"{agent_features.shape[-1]}; the cosine similarity compares [deter, flat stoch] "
            "against the same layout (A10 disposition 5 — same architecture, different "
            "parameters)"
        )
    sim = jnp.sum(feats * agent_features, axis=-1) / jnp.maximum(
        jnp.linalg.norm(feats, axis=-1) * jnp.linalg.norm(agent_features, axis=-1), 1e-12
    )
    kl = jnp.mean(gate * kl_total)
    contrast_term = jnp.mean(-rho * sim)
    total = kl + contrast * contrast_term
    losses = CRSPPLosses(
        total=total,
        kl=kl,
        kl_raw=jnp.mean(kl_raw),
        contrast=contrast_term,
        positive=jnp.mean(gate),
    )
    if return_states:
        return total, (losses, (deters, codes))
    return total, losses


def fit_crspp(
    system: System,
    params: SystemParams,
    learn_state: LearnState | None,
    embeds: Array,
    rho: Array,
    agent_features: Array,
    *,
    key: Array,
    is_first: Array | None,
    lr: float | Array = DEFAULT_LR,
    lr_by_slot: Mapping[str, float | Array] | None = None,
    clip: float | None = DEFAULT_CLIP,
    return_states: bool = False,
    **loss_kwargs,
) -> (
    tuple[SystemParams, LearnState, CRSPPLosses]
    | tuple[SystemParams, LearnState, CRSPPLosses, tuple[Array, ...]]
):
    """One gradient step of preference learning: ``(crspp_params, learn_state, losses)``.

    ``system``/``params`` are the CRSPP's own pair (``System.scoring[ROW]``,
    ``SystemParams.scoring[ROW]``) and ``learn_state`` is the nested carry at
    ``scoring:cosine_sim_to_pref_rollout``, so ``lr_by_slot`` addresses the CRSPP's slots.
    One global clip over the whole model precedes the per-slot steps. With
    ``return_states=True`` a fourth element follows: the pass's posterior as the belief tuple
    :func:`crspp_recognize` returns, gradient stopped. ``is_first`` and ``**loss_kwargs`` go
    to :func:`crspp_loss`.
    """
    parts = resolve(system, params)
    opt = System._resolve_learn_state(learn_state, system._gradient_slots(params))
    rates = system._resolve_rates(lr_by_slot, "lr_by_slot", params)
    fields = tuple(i for i, part in enumerate(parts.parts) if getattr(part, "learnable", True))

    def loss_fn(trainable: tuple):
        swapped = dict(zip(fields, trainable, strict=True))
        return crspp_loss(
            parts.with_parts(swapped),
            key,
            embeds,
            rho,
            agent_features,
            is_first=is_first,
            return_states=True,
            **loss_kwargs,
        )

    trainable = tuple(parts.parts[i] for i in fields)
    grads, (out, (deters, codes)) = eqx.filter_grad(loss_fn, has_aux=True)(trainable)
    if clip is not None:
        clipper = optax.clip_by_global_norm(clip)
        grads, _ = clipper.update(grads, clipper.init(grads))

    updated: dict[str, object] = {}
    for i, part, grad in zip(fields, trainable, grads, strict=True):
        address = parts.slots[i]
        rate = lr if rates is None else rates.get(address, lr)
        if isinstance(rate, int | float) and float(rate) == 0.0:
            continue
        updated[address], opt[address] = part.absorb(grad, opt[address], lr=rate)
    carried = {} if learn_state is None else dict(learn_state.moments)
    new_params = system._write_slots(params, updated)
    new_state = LearnState(opt_states=opt, moments=carried)
    if not return_states:
        return new_params, new_state, out
    structure = system._amortized_structure(params)
    values = {_amortized.CARRY: deters, _amortized.CODE: codes}
    beliefs = tuple(jax.lax.stop_gradient(values[family]) for family in structure.families)
    return new_params, new_state, out, beliefs


#: The similarity leaf's block key in the term tree.
CRSPP_BLOCK = "crspp"


def _crspp_opt_state(entry, declared):
    if declared is None:
        raise ValueError(
            "the cosine_sim_to_pref_rollout row owns a declared System (its CRSPP), so its "
            "optimizer carry is that System's own LearnState and needs the structure from "
            "System.scoring; none was found"
        )
    return declared.init_learn_state(entry)


def cosine_sim_to_pref_rollout(st: registry.Statics) -> registry.Step:
    """Their ``sim``: cosine similarity to the CRSPP's imagined preferred state.

    The CRSPP is the System in this row's scoring slot. Its rollout is
    ``update_empirical_prior`` with no action parent, so the preferred trajectory is computed
    once in ``prepare``, for the ``st.horizon`` steps of the plan being scored: sampled with
    ``st.key``, expected without. Comparison is on ``[deter, flat stoch]``. The rollout starts
    from ``aux_beliefs[<this row's name>]`` (:func:`crspp_recognize`, passed through
    ``score_policies`` or ``imagine``); with none it starts from the CRSPP's declared episode
    start, a generic trajectory.

    Design: ``docs/design/amortization-debt-ledger.md`` entry 5.
    """
    system, params = st.declared, st.scoring
    horizon = int(st.horizon)
    if horizon < 1:
        raise ValueError(
            f"the {ROW} row rolls its preferred trajectory to the scored plan's horizon and "
            f"Statics.horizon is {st.horizon}; the amortized scoring path sets it from the plan "
            "being scored (System._amortized_scoring(horizon=))"
        )
    start = st.aux_beliefs if st.aux_beliefs is not None else system._initial_priors(params)
    keys = None if st.key is None else jax.random.split(st.key, horizon)

    def roll(qs, k_t):
        pushed = system._update_empirical_prior(params, qs, None, key=k_t)
        return pushed, features(pushed)

    _, preferred = jax.lax.scan(roll, tuple(start), keys, length=horizon)
    width = int(sum(st.structure.widths))
    if preferred.shape[-1] != width:
        raise ValueError(
            f"the preference model's feature vector is {preferred.shape[-1]} numbers wide but "
            f"this System's is {width}; cosine similarity compares [deter, flat stoch] against "
            "the same layout (their CRSPP shares the world model's architecture, not its "
            "parameters — A10 disposition 5)"
        )

    def step(ctx: registry.StepContext) -> dict[str, Interval]:
        ours = features(ctx.qs)
        theirs = jnp.take(preferred, ctx.t, axis=0)
        norm = jnp.linalg.norm(ours) * jnp.linalg.norm(theirs)
        return {CRSPP_BLOCK: Interval.exact(jnp.sum(ours * theirs) / jnp.maximum(norm, 1e-12))}

    return step


register_term(ROW, -1.0)

COSINE_SIM = registry.register_row(
    registry.Estimator(
        ROW,
        registry.SINGLE,
        cosine_sim_to_pref_rollout,
        term=ROW,
        families=(_amortized.CODE,),
        parametric=True,
        needs_structure=True,
        derived=False,
        trainer="reference.raif.crspp.fit_crspp",
        opt_state=lambda entry, declared: _crspp_opt_state(entry, declared),
    )
)


def embeddings(system: System, params: SystemParams, obs: Array) -> Array:
    """The agent's encoder output on a batch of observations: ``(..., embed_dim)``.

    ``obs`` may carry any number of leading axes.
    """
    core, _ = assemble(system, params)
    flat = jnp.reshape(jnp.asarray(obs), (-1, core.obs_dim))
    out = jax.vmap(lambda o: core.encoder((o,)))(flat)
    return jnp.reshape(out, (*jnp.shape(obs)[:-1], core.embed_dim))
