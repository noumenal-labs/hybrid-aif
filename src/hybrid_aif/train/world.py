"""``fit_world``: replay-driven world-model learning.

A pure function over explicit state: ``(System, params, data, LearnState)`` in,
the stepped triple out. The objective is DreamerV3's world-model loss, averaged
over batch and time::

    L = 1.0·decoder_nll + 1.0·reward_nll + 1.0·cont_nll
        + 0.5·max(KL(sg q ‖ p), 1) + 0.1·max(KL(q ‖ sg p), 1)

built from the heads' own ``nll`` and
:func:`~hybrid_aif.distributions.neural.kl_balance`. Sequences go through
:meth:`~hybrid_aif.inference.rssm.RSSMCore.observe` (one ``lax.scan`` each) under
one ``vmap``. The loss is stated over the assembly read out of the System's slots
(:func:`assemble`), and the stepped parts are written back by slot key.

Design: docs/design/amortization.md A2, A6
"""

import dataclasses
from collections.abc import Mapping
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from hybrid_aif.distributions.neural import (
    kl_balance,
)
from hybrid_aif.inference import amortized as _amortized
from hybrid_aif.inference.rssm import PART_FIELDS, RSSMCore, RSSMState
from hybrid_aif.system import LearnState, System, SystemParams

#: DreamerV3's world-model optimizer defaults: adam at ``lr 1e-4`` with a global-norm
#: ``clip 1000`` over the whole model. The learning rate is applied by :func:`fit_world`
#: after the optax step — the transformation stays static and the rate stays
#: *traceable*, as ``distributions.neural.OPTIMIZER`` documents for a neural slot.
DEFAULT_LR = 1e-4
DEFAULT_CLIP = 1000.0

#: The world-model adam's ``eps``: optax's default, hence ``distributions.neural.OPTIMIZER``'s,
#: so each part steps through its own ``absorb`` and no per-slot optimizer is built.
DEFAULT_EPS = 1e-8


class WorldBatch(NamedTuple):
    """A batch of replay sequences, all shapes static.

    ``obs`` ``(B, T, obs_dim)``, ``action`` ``(B, T, action_dim)``, ``reward``
    ``(B, T)``, ``cont`` ``(B, T)``, and optionally ``is_first`` ``(B, T)``.
    ``action[b, t]`` is ``a_{t−1}``, the action that produced ``obs[b, t]``.
    ``cont`` is ``1 − done``. ``is_first`` marks where an episode begins inside a
    sequence and restarts the recurrent carry there; episode-aligned batches may
    leave it ``None``.
    """

    obs: Array
    action: Array
    reward: Array
    cont: Array
    is_first: Array | None = None

    def check(self, core: RSSMCore) -> tuple[int, int]:
        """Validate against a core and return ``(B, T)``."""
        if self.obs.ndim != 3 or self.obs.shape[-1] != core.obs_dim:
            raise ValueError(f"obs has shape {self.obs.shape}, expected (B, T, {core.obs_dim})")
        b, t = self.obs.shape[0], self.obs.shape[1]
        expected = {
            "action": (self.action.shape, (b, t, core.action_dim)),
            "reward": (self.reward.shape, (b, t)),
            "cont": (self.cont.shape, (b, t)),
        }
        if self.is_first is not None:
            expected["is_first"] = (self.is_first.shape, (b, t))
        bad = {k: v for k, v in expected.items() if tuple(v[0]) != v[1]}
        if bad:
            raise ValueError(
                "WorldBatch shapes disagree with obs "
                f"{self.obs.shape}: "
                + "; ".join(f"{k} is {got} but should be {want}" for k, (got, want) in bad.items())
            )
        return b, t


class WorldLosses(NamedTuple):
    """The objective, decomposed — every field a scalar mean over batch and time.

    ``total`` is the differentiated loss (scales applied). ``decoder``, ``reward``
    and ``cont`` are the unscaled NLLs; ``dyn``/``rep`` the two KL directions after
    the free-bits floor and before their scales, ``dyn_raw``/``rep_raw`` before the
    floor (``dyn > dyn_raw`` means the floor binds).
    """

    total: Array
    decoder: Array
    reward: Array
    cont: Array
    dyn: Array
    rep: Array
    dyn_raw: Array
    rep_raw: Array


def world_loss(
    core: RSSMCore,
    batch: WorldBatch,
    key: Array,
    *,
    dyn_scale: float = 0.5,
    rep_scale: float = 0.1,
    free_nats: float = 1.0,
    decoder_scale: float = 1.0,
    reward_scale: float = 1.0,
    cont_scale: float = 1.0,
    return_states: bool = False,
) -> tuple[Array, WorldLosses] | tuple[Array, tuple[WorldLosses, RSSMState]]:
    """The world-model loss on one batch: ``(total, WorldLosses)``.

    With ``return_states=True`` the auxiliary is ``(WorldLosses, RSSMState)``, the
    posterior states of this forward pass, leaves ``(B, T, ...)``. Pure and
    differentiable in ``core``; ``key`` draws the straight-through posterior
    samples. Defaults are DreamerV3's loss scales and one-nat free-bits floor.
    """
    b, _ = batch.check(core)

    def one_sequence(
        k: Array,
        obs: Array,
        action: Array,
        reward: Array,
        cont: Array,
        is_first: Array | None = None,
    ):
        states, post_logits, prior_logits = core.observe(k, action, obs, is_first=is_first)
        #  each head's own loss, per step, on the state vector it reads
        parents = jax.vmap(core._features)(states)
        decoder_nll = jax.vmap(core.decoder.nll)(obs, parents)
        reward_nll = jax.vmap(core.reward.nll)(reward, parents)
        cont_nll = jax.vmap(core.cont.nll)(cont, parents)
        kl = kl_balance(
            post_logits,
            prior_logits,
            dyn_scale=dyn_scale,
            rep_scale=rep_scale,
            free_nats=free_nats,
        )
        total = (
            decoder_scale * decoder_nll
            + reward_scale * reward_nll
            + cont_scale * cont_nll
            + kl.total
        )
        losses = WorldLosses(
            total=total,
            decoder=decoder_nll,
            reward=reward_nll,
            cont=cont_nll,
            dyn=kl.dyn,
            rep=kl.rep,
            dyn_raw=kl.dyn_raw,
            rep_raw=kl.rep_raw,
        )
        return losses, states

    columns = (jax.random.split(key, b), batch.obs, batch.action, batch.reward, batch.cont)
    if batch.is_first is None:
        per_step, states = jax.vmap(one_sequence)(*columns)
    else:
        per_step, states = jax.vmap(one_sequence)(*columns, batch.is_first)
    losses = WorldLosses(*(jnp.mean(term) for term in per_step))
    if return_states:
        return losses.total, (losses, states)
    return losses.total, losses


def assemble(system: System, params: SystemParams, roles: dict[str, str] | None = None):
    """The world-model assembly behind a declared System: ``(core, slot addresses)``.

    :meth:`~hybrid_aif.inference.rssm.RSSMCore.from_declarations` over the
    validated amortized structure; ``roles`` names the reward and continuation
    modalities when names and shapes do not decide.
    """
    params = system._p(params)
    structure = system._amortized_structure(params)
    mods, facs = (system.modality_names, system._facs)
    return RSSMCore.from_declarations(
        structure,
        params.likelihood,
        params.transition,
        params.initial_state_prior,
        params.recognition,
        system.A_deps,
        mods,
        facs,
        roles=roles,
    )


def fit_world(
    system: System,
    params: SystemParams,
    learn_state: LearnState | None,
    batch: WorldBatch,
    *,
    key: Array,
    lr: float | Array = DEFAULT_LR,
    lr_by_slot: Mapping[str, float | Array] | None = None,
    clip: float = DEFAULT_CLIP,
    roles: dict[str, str] | None = None,
    return_states: bool = False,
    **loss_kwargs,
) -> (
    tuple[SystemParams, LearnState, WorldLosses]
    | tuple[SystemParams, LearnState, WorldLosses, RSSMState]
):
    """One gradient step of world-model learning: ``(params, learn_state, losses)``.

    With ``return_states=True`` a fourth element follows: the posterior
    :class:`~hybrid_aif.inference.rssm.RSSMState` of the forward pass (pre-step
    parameters, gradient stopped). ``roles`` goes to :func:`assemble`,
    ``**loss_kwargs`` to :func:`world_loss`. Pure, jit-able, resumable from its
    own return value; ``learn_state`` starts as ``system.init_learn_state(params)``
    and ``None`` raises.

    ``lr`` may be traced. ``clip`` is one global-norm clip over the whole model
    (``None`` disables it), summed in :data:`~hybrid_aif.inference.rssm.PART_FIELDS`
    order; each slot then steps through its own ``LearnState`` entry.
    ``lr_by_slot`` is ``System.learn``'s rate grammar: a static ``0.0`` skips that
    slot's step at trace time, a traced zero multiplies it. A frozen slot is not
    differentiated, holds no optimizer state, and comes back bitwise.
    """
    core, slots = assemble(system, params, roles=roles)
    batch.check(core)
    opt = System._resolve_learn_state(learn_state, system._gradient_slots(params))
    rates = system._resolve_rates(lr_by_slot, "lr_by_slot", params)
    fields = tuple(name for name in PART_FIELDS if getattr(core, name).learnable)
    addresses = tuple(getattr(slots, name) for name in fields)

    def loss_fn(parts: tuple):
        model = dataclasses.replace(core, **dict(zip(fields, parts, strict=True)))
        return world_loss(model, batch, key, return_states=True, **loss_kwargs)

    trainable = tuple(getattr(core, name) for name in fields)
    grads, (losses, states) = eqx.filter_grad(loss_fn, has_aux=True)(trainable)
    if clip is not None:
        clipper = optax.clip_by_global_norm(clip)
        grads, _ = clipper.update(grads, clipper.init(grads))

    updated: dict[str, object] = {}
    for address, part, grad in zip(addresses, trainable, grads, strict=True):
        rate = lr if rates is None else rates.get(address, lr)
        if isinstance(rate, int | float) and float(rate) == 0.0:
            continue  # static zero: this slot takes no step, and its carry is untouched
        updated[address], opt[address] = part.absorb(grad, opt[address], lr=rate)
    #  `moments` is a cross-trainer carry (the return normalizer's EMA, the slow
    #  critic's counter): this trainer owns none of it and passes it through unchanged.
    carried = {} if learn_state is None else dict(learn_state.moments)
    new_params = system._write_slots(params, updated)
    new_state = LearnState(opt_states=opt, moments=carried)
    if return_states:
        return new_params, new_state, losses, jax.lax.stop_gradient(states)
    return new_params, new_state, losses


def posterior_states(
    system: System,
    params: SystemParams,
    batch: WorldBatch,
    *,
    key: Array,
    roles: dict[str, str] | None = None,
) -> RSSMState:
    """The recognition model's posterior states over a replay batch: ``(B, T, ...)``.

    :meth:`~hybrid_aif.inference.rssm.RSSMCore.observe` under one ``vmap``; the
    codes are straight-through samples. The start states for
    :func:`~hybrid_aif.train.imagine.imagine` and the targets for
    :func:`~hybrid_aif.train.ensemble.fit_ensemble`. ``roles`` goes to :func:`assemble`.
    """
    core, _ = assemble(system, params, roles=roles)
    b, _ = batch.check(core)

    def one(k: Array, obs: Array, action: Array, is_first: Array | None = None) -> RSSMState:
        states, _, _ = core.observe(k, action, obs, is_first=is_first)
        return states

    columns = (jax.random.split(key, b), batch.obs, batch.action)
    if batch.is_first is None:
        return jax.vmap(one)(*columns)
    return jax.vmap(one)(*columns, batch.is_first)


def belief_tuple(structure, state: RSSMState) -> tuple[Array, ...]:
    """An :class:`~hybrid_aif.inference.rssm.RSSMState` as the positional belief tuple (internal).

    Per-factor values in declaration order: the carry's vector for ``carry``, the code's tensor
    for ``code``. Leading batch axes pass through. The faces take beliefs keyed by factor name:
    :meth:`System.beliefs_from_states <hybrid_aif.system.System.beliefs_from_states>`.
    """
    values = {_amortized.CARRY: state.deter, _amortized.CODE: state.stoch}
    missing = sorted(set(structure.families) - set(values))
    if missing:
        raise ValueError(
            f"belief_tuple: this System declares factor families {missing}, which the RSSM "
            "assembly does not produce (it is one carry and one code)"
        )
    return tuple(values[family] for family in structure.families)
