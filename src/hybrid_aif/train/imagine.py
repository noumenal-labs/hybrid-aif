"""``imagine``: Dreamer-style actor + critic optimization on world-model rollouts.

The actor is the module in the ``policy_prior`` slot; the critic and its slow copy
are the ``(critic, slow)`` pair in ``scoring['g_tail']``. One call takes one
gradient step on both from a batch of imagined trajectories. Rollouts carry a
straight-through one-hot sample of the code; the per-step term tree comes from
``System._amortized_scoring``, folded into a value by :func:`term_value`. The
critic fits the λ-return (two-hot ``symlog_disc`` NLL plus ``slowreg`` toward the
slow critic); the actor maximizes the percentile-normalized advantage,
reparameterized, plus entropy bonuses and an optional weighted behaviour-cloning
term. The ``g_tail`` row is not folded into the per-step value (the λ-return
closes the horizon); a caller who also mounts ``occupancy`` should pass
``prior_entropy_scale=0.0``.

Design: docs/design/amortization.md A3, A4, A6, A12 chunk 2c §(d), chunk 2d §(a″)
"""

import dataclasses
from collections.abc import Mapping
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from hybrid_aif.controls import soft_clip
from hybrid_aif.distributions.neural import (
    PredictorHead,
    adam,
    apply_gradient,
    factored_categorical_entropy,
    symlog_disc_nll,
)
from hybrid_aif.inference import amortized as _amortized
from hybrid_aif.interval import Interval
from hybrid_aif.objectives.amortized import RETNORM_KEY, SLOW_KEY, features, scoring_moments
from hybrid_aif.objectives.base import term_sign
from hybrid_aif.system import ACTOR_KEY, LearnState, System, SystemParams, keyed

#: DreamerV3's actor/critic optimizer defaults: adam at ``lr 3e-5``, global-norm
#: ``clip 100``, ``eps 1e-5``; built per slot by :func:`hybrid_aif.distributions.neural.adam`.
DEFAULT_LR = 3e-5
DEFAULT_CLIP = 100.0
DEFAULT_EPS = 1e-5

#: DreamerV3's discount, ``1 − 1/333`` (a 333-step horizon), and the λ-return's λ.
DEFAULT_DISCOUNT = 1.0 - 1.0 / 333.0
DEFAULT_LAMBDA = 0.95

#: The tag the scoring rows' key is folded in with (``sample_scoring=True``). Under
#: JAX's partitionable threefry ``fold_in(k, d) == split(k, n)[d]`` for every ``n > d``,
#: so a SMALL tag would hand the rows a key the rollout scan also draws from. A tag
#: far above any realistic split count is reached by no ``split``.
SCORING_KEY_TAG = 0x7A3C0DE5

#: The actor loss's entropy bonuses and the BC term's scale (see the module docstring).
DEFAULT_POLICY_ENTROPY_SCALE = 3e-4  # DreamerV3's `actent`
DEFAULT_PRIOR_ENTROPY_SCALE = 0.0  # a bonus on the state prior's entropy: off unless asked
DEFAULT_BC_SCALE = 0.0  # weighted behaviour cloning on the start actions: off unless asked

#: The slow critic: its EMA fraction, its update period (every step), and DreamerV3's
#: ``slowreg`` weight on the slow-target logprob regularizer.
DEFAULT_SLOW_FRACTION = 0.02
DEFAULT_SLOW_PERIOD = 1
DEFAULT_SLOWREG = 1.0

#: Percentile return normalization ``(perclo, perchi, decay, max)``: DreamerV3's
#: ``retnorm: perc_ema`` defaults.
DEFAULT_RETNORM = (5.0, 95.0, 0.99, 1.0)

#: This trainer's two ``LearnState.moments`` entries, seeded by ``System.init_learn_state``
#: (:func:`~hybrid_aif.objectives.amortized.scoring_moments`).
MOMENT_KEYS = (RETNORM_KEY, SLOW_KEY)

#: The critic's ``scoring`` address, hence its ``LearnState`` / ``lr_by_slot`` key.
ROW = "g_tail"
CRITIC_KEY = f"scoring:{ROW}"


class ImagineMetrics(NamedTuple):
    """Plain arrays, one scalar each — what a training loop logs and a test asserts.

    ``value`` is the mean per-step G as a value (negated EFE), ``ret`` the mean
    λ-return and ``critic_value`` the critic's mean at the same states.
    ``per_term`` is the value's decomposition keyed by term name.
    """

    actor_loss: Array
    critic_loss: Array
    value: Array
    ret: Array
    critic_value: Array
    advantage: Array
    policy_entropy: Array
    prior_entropy: Array
    #: ``mean(log pi(a_start | s_0))`` over **every** start, weighted or not: the
    #: actor's raw density at the replay draws, a distribution-shift diagnostic. It is
    #: *not* the BC term's contribution — a product of means is not the mean of a
    #: product — and :attr:`bc_term` is the quantity to compare against a BC loss.
    bc_logprob: Array
    #: ``mean(bc_weights)`` (zero when no BC inputs were passed).
    bc_weight: Array
    #: The BC term's actual contribution to the actor loss:
    #: ``bc_scale · mean(−log pi(a_start | s_0) · bc_weights)``, weight and scale folded in.
    bc_term: Array
    retnorm_low: Array
    retnorm_high: Array
    #: The mean per-step rollout weight (:func:`rollout_weight`): how much of the
    #: horizon the continuation head is currently truncating — 1.0 means "nothing", and
    #: a value near ``1/horizon`` means the rollouts are being cut at their first step.
    #: It is also the direct read on ``start_cont``'s effect.
    rollout_weight_mean: Array
    per_term: dict[str, Array]


class Rollout(NamedTuple):
    """One imagined trajectory's per-step quantities. Leaves ``(H,)`` or ``(H+1,)``."""

    value: Array  # (H,) G as a value (negated EFE), per step taken
    per_term: dict[str, Array]  # (H,) each, the value's decomposition
    critic: Array  # (H+1,) the critic at s_0 … s_H
    cont: Array  # (H,) the continuation head's MODE at s_1 … s_H — 0.0/1.0, see `_cont_mode`
    prior_entropy: Array  # (H,) H[p(stoch | deter)] at each step
    policy_entropy: Array  # (H,) H[pi(a | s)] at s_0 … s_{H-1}
    start_logprob: Array  # log pi(a_start | s_0) — the BC term's content
    start_features: Array  # (H, feature) features at s_0 … s_{H-1}, for the critic loss


def _cont_mode(probability: Array) -> Array:
    """The continuation head's mode: a hard 0/1, ``probability > 0.5``.

    A step the head believes has ended is truncated (``disc = 0``), and the
    threshold carries no gradient to the actor.
    """
    return (probability > 0.5).astype(probability.dtype)


def rollout_weight(cont: Array, discount: float, start_cont: Array) -> Array:
    """The per-step weight of a rollout step: ``(N, H)``.

    Entry ``k`` is ``start_cont · discount**k · Π_{j=1..k} cont(s_j)``, so a zero
    continuation zeroes that step and every later one. ``cont`` is
    :attr:`Rollout.cont`; ``start_cont`` ``(N,)`` is ``1 − is_terminal(s_0)`` from
    replay data.
    """
    disc = discount * jnp.asarray(cont)
    running = jnp.cumprod(disc[:, :-1], axis=1)  # cont at s_1 … s_{H-1}
    ones = jnp.ones((running.shape[0], 1), running.dtype)
    return jnp.asarray(start_cont)[:, None] * jnp.concatenate([ones, running], axis=1)


def term_value(tree) -> tuple[Array, dict[str, Array]]:
    """A term tree folded into a value (negative EFE): ``(value, per term)``.

    Folds with the registered term signs and negates once; Interval leaves
    contribute their midpoint. There is no per-term weight: a term's sign is its
    row's registration. Raises ``ValueError`` on a tree with no leaves.
    """
    per_term: dict[str, Array] = {}
    total = None
    for term, blocks in tree.items():
        scale = -term_sign(term)
        part = None
        for leaf in blocks.values():
            point = leaf.midpoint if isinstance(leaf, Interval) else leaf
            part = point if part is None else part + point
        if part is None:
            continue
        per_term[term] = scale * part
        total = per_term[term] if total is None else total + per_term[term]
    if total is None:
        raise ValueError("an imagined term tree with no leaves: G would be an empty fold")
    return total, per_term


def lambda_return(value: Array, reward: Array, disc: Array, return_lambda: float) -> Array:
    """DreamerV3's λ-return over an imagined trajectory.

    ``value`` is ``(H+1,)``, the critic at ``s_0 … s_H``; ``reward`` and ``disc`` are
    ``(H,)`` indexed by the step taken (``reward[k]`` is collected moving into
    ``s_{k+1}``, ``disc[k] = discount · cont(s_{k+1})``). Returns ``(H,)``, entry
    ``k`` the return estimate at ``s_k``, bootstrapped by ``value[H]``.
    """
    interm = reward + disc * value[1:] * (1.0 - return_lambda)

    def back(carry: Array, xs: tuple[Array, Array]) -> tuple[Array, Array]:
        i, d = xs
        out = i + d * return_lambda * carry
        return out, out

    _, out = jax.lax.scan(back, value[-1], (interm, disc), reverse=True)
    return out


def perc_ema(
    state: Array, x: Array, perclo: float, perchi: float, decay: float, cap: float
) -> tuple[Array, Array, Array]:
    """Percentile-EMA return normalization: ``(new state, offset, invscale)``.

    ``state`` is the ``(low, high)`` pair of percentile EMAs, updated from this
    batch's returns before use. ``invscale = max(1/cap, high − low)``, so a small
    spread is never amplified.
    """
    lo, hi = jnp.percentile(x, perclo), jnp.percentile(x, perchi)
    low = state[0] * decay + lo * (1.0 - decay)
    high = state[1] * decay + hi * (1.0 - decay)
    return jnp.stack([low, high]), low, jnp.maximum(1.0 / cap, high - low)


def init_retnorm() -> Array:
    """A fresh ``(low, high)`` percentile-EMA carry: zeros, as seeded under :data:`RETNORM_KEY`."""
    return scoring_moments(ROW)[RETNORM_KEY]


def imagine(
    system: System,
    params: SystemParams,
    start_beliefs: Mapping[str, Array],
    learn_state: LearnState | None,
    *,
    key: Array,
    horizon: int | None = None,
    start_actions: Mapping[str, Array] | None = None,
    bc_weights: Array | None = None,
    start_cont: Array | None = None,
    aux_beliefs: Mapping[str, Mapping[str, Array]] | None = None,
    cont_modality: str | None = None,
    discount: float = DEFAULT_DISCOUNT,
    return_lambda: float = DEFAULT_LAMBDA,
    policy_entropy_scale: float = DEFAULT_POLICY_ENTROPY_SCALE,
    prior_entropy_scale: float = DEFAULT_PRIOR_ENTROPY_SCALE,
    bc_scale: float = DEFAULT_BC_SCALE,
    slow_fraction: float = DEFAULT_SLOW_FRACTION,
    slow_period: int = DEFAULT_SLOW_PERIOD,
    slowreg: float = DEFAULT_SLOWREG,
    retnorm: tuple[float, float, float, float] = DEFAULT_RETNORM,
    actor_lr: float | Array = DEFAULT_LR,
    critic_lr: float | Array = DEFAULT_LR,
    lr_by_slot: Mapping[str, float | Array] | None = None,
    clip: float | None = DEFAULT_CLIP,
    adam_eps: float = DEFAULT_EPS,
    sample_scoring: bool = False,
) -> tuple[SystemParams, LearnState, ImagineMetrics]:
    """One actor+critic gradient step on imagined rollouts: ``(params, learn_state, metrics)``.

    Gradients: the actor's flows through its reparameterised draws and the dynamics, with the
    critic and the actor's conditioning state stop-gradiented; the critic regresses on the
    stop-gradiented λ-return at stop-gradiented start features.

    Pure, jit-able and resumable from its own return value; steps only
    :data:`ACTOR_KEY` and ``scoring:g_tail``, threading every other ``LearnState``
    entry through.

    ``start_beliefs`` is batched beliefs keyed by factor name (leaves ``(N, ...)``), e.g.
    :func:`~hybrid_aif.train.world.posterior_states` flattened over batch and time.
    ``horizon`` defaults to ``policy_len`` and must be at least 1.
    ``start_actions`` (keyed by the action nodes' names, ``System.names.actions``; leaves
    ``(N, ...)``) and ``bc_weights`` ``(N,)``
    (in [0, 1]) are the replay action taken at each start state and its behaviour-cloning
    weight; with ``WorldBatch``'s ``action[t] = a_{t−1}`` that action sits at
    ``t+1``. Pass both or neither. ``aux_beliefs`` maps a scoring row name to the
    belief carry of that row's own nested System, keyed by its factor names (leaves
    ``(N, ...)``), as :meth:`~hybrid_aif.system.System.planner_context` takes it.
    ``cont_modality`` names the continuation head (``None``: the unique ``binary``
    head). ``start_cont`` ``(N,)`` is ``1 − is_terminal(s_0)``; omitted, every
    start counts as non-terminal. ``sample_scoring`` (static) gives the scoring
    rows their own key without changing the actor's draws.

    ``actor_lr``/``critic_lr`` may be traced; ``lr_by_slot`` overrides them by
    address, a static ``0.0`` skipping that slot. ``clip`` is a per-slot global-norm
    clip (``None`` disables it); ``adam_eps`` builds the two optimizers.

    Raises ``ValueError`` when the System declares no policy module, when
    ``scoring['g_tail']`` is not a ``(critic, slow)`` pair, when ``learn_state`` is
    ``None`` or lacks this trainer's optimizer or ``moments`` entries (never
    re-seeded), and on a shape mismatch in ``start_actions``/``bc_weights``/
    ``start_cont``.
    """
    horizon = int(system.policy_len if horizon is None else horizon)
    if horizon < 1:
        raise ValueError(f"imagine: horizon must be at least 1, got {horizon}")
    actor = system._habit_module(params)
    if actor is None:
        raise ValueError(
            "imagine trains the actor in the policy_prior designation (A3/A12 item 3) and this "
            "System declares no policy *module* there. Put a module (beliefs -> an action "
            "distribution) in SystemParams.policy_prior; a log-space vector over an enumerated "
            "policy set is the classical E and has nothing to backpropagate."
        )
    entry = params.scoring.get(ROW)
    if not isinstance(entry, tuple) or len(entry) != 2:
        raise ValueError(
            "imagine needs the critic and its slow copy: SystemParams.scoring['g_tail'] holds "
            f"the pair (critic, slow), got {type(entry).__name__}. The slow copy is a training "
            "target regularizer moved by EMA (``slow_fraction``, default 0.02) — it is not read at "
            "scoring time, which is why the row itself accepts the critic alone (A12 item 5)."
        )
    critic, slow = entry
    names = system._names(system._p(params))
    start_beliefs = keyed.beliefs_in(names, start_beliefs, "imagine")
    if start_actions is not None:
        start_actions = keyed.actions_in(names, start_actions, "imagine")
    cont_head, cont_deps = _cont_head(system, params, cont_modality)
    structure = system._amortized_structure(params, "imagine")
    code, carry = _code_and_carry(structure, system)
    layout = system._control_layout(params)
    bounds = None if layout is None else layout.bounds
    action_dim = actor.child_dim  # the actor declares the action's width

    n = int(jnp.shape(start_beliefs[carry])[0])
    flags, expert_actions = _bc_inputs(start_actions, bc_weights, n, action_dim, bc_scale)
    aux = None if not aux_beliefs else system._aux_in(aux_beliefs, "imagine")
    first_cont = jnp.ones((n,), jnp.float32) if start_cont is None else jnp.asarray(start_cont)
    if jnp.shape(first_cont) != (n,):
        raise ValueError(f"imagine: start_cont must be ({n},), got {jnp.shape(first_cont)}")

    optimizer = adam(adam_eps)  # per-slot adam with this trainer's eps, not optax's default
    opt = _resolve(learn_state, actor, critic, optimizer)
    moments = _moments(learn_state)
    retnorm_state = moments[RETNORM_KEY]
    rates = system._resolve_rates(lr_by_slot, "lr_by_slot", params)
    a_rate = actor_lr if rates is None else rates.get(ACTOR_KEY, actor_lr)
    c_rate = critic_lr if rates is None else rates.get(CRITIC_KEY, critic_lr)
    keys = jax.random.split(key, n)

    def loss_fn(pair: tuple):
        actor_m, critic_m = pair
        rolls = roll_batch(
            system,
            params,
            actor_m,
            critic_m,
            tuple(start_beliefs),
            keys,
            horizon=horizon,
            bounds=bounds,
            cont=(cont_head, cont_deps),
            code_and_carry=(code, carry),
            aux=aux,
            expert_actions=expert_actions,
            sample_scoring=sample_scoring,
        )
        disc = discount * rolls.cont
        ret = jax.vmap(lambda v, r, d: lambda_return(v, r, d, return_lambda))(
            rolls.critic, rolls.value, disc
        )
        base = rolls.critic[:, :-1]
        new_retnorm, offset, invscale = perc_ema(
            retnorm_state, jax.lax.stop_gradient(ret), *retnorm
        )
        advantage = (ret - offset) / invscale - (base - offset) / invscale
        bc = -rolls.start_logprob * flags
        #  The rollout weight multiplies the advantage and the POLICY entropy; the prior
        #  entropy bonus and the BC term are left unweighted.
        weight = jax.lax.stop_gradient(rollout_weight(rolls.cont, discount, first_cont))
        per_step = -advantage - policy_entropy_scale * rolls.policy_entropy
        actor_loss = (
            jnp.mean(per_step * weight)
            - prior_entropy_scale * jnp.mean(rolls.prior_entropy)
            + bc_scale * jnp.mean(bc)
        )
        # The critic regresses on the return at STOP-GRADIENTED states, so the
        # actor's path contributes nothing to its gradient and one filter_grad over
        # the pair stays two independent objectives.
        feats = jax.lax.stop_gradient(rolls.start_features)
        target = jax.lax.stop_gradient(ret)
        slow_target = jax.lax.stop_gradient(jax.vmap(jax.vmap(lambda f: slow.predict((f,))))(feats))
        logits = jax.vmap(jax.vmap(lambda f: critic_m((f,))))(feats)
        bins = critic_m.bins
        nll = jax.vmap(jax.vmap(lambda lg, y: symlog_disc_nll(lg, y, bins)))
        #  The SAME weight, applied to the sum of the two NLLs rather than to each mean.
        critic_loss = jnp.mean((nll(logits, target) + slowreg * nll(logits, slow_target)) * weight)
        metrics = ImagineMetrics(
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            value=jnp.mean(rolls.value),
            ret=jnp.mean(ret),
            critic_value=jnp.mean(base),
            advantage=jnp.mean(advantage),
            policy_entropy=jnp.mean(rolls.policy_entropy),
            prior_entropy=jnp.mean(rolls.prior_entropy),
            bc_logprob=jnp.mean(rolls.start_logprob),
            bc_weight=jnp.mean(flags),
            bc_term=bc_scale * jnp.mean(bc),
            retnorm_low=new_retnorm[0],
            retnorm_high=new_retnorm[1],
            rollout_weight_mean=jnp.mean(weight),
            per_term={k: jnp.mean(v) for k, v in rolls.per_term.items()},
        )
        return actor_loss + critic_loss, (new_retnorm, metrics)

    grads, (new_retnorm, metrics) = eqx.filter_grad(loss_fn, has_aux=True)((actor, critic))
    actor_grad, critic_grad = grads
    if clip is not None:
        actor_grad = _clip(actor_grad, clip)
        critic_grad = _clip(critic_grad, clip)

    new_actor, new_critic = actor, critic
    if not _static_zero(a_rate):
        new_actor, opt[ACTOR_KEY] = apply_gradient(
            actor, actor_grad, opt[ACTOR_KEY], lr=a_rate, optimizer=optimizer
        )
    if not _static_zero(c_rate):
        new_critic, opt[CRITIC_KEY] = apply_gradient(
            critic, critic_grad, opt[CRITIC_KEY], lr=c_rate, optimizer=optimizer
        )
    # The slow critic: an EMA (slow_fraction, every slow_period updates) toward the
    # freshly stepped critic — and a HARD COPY on the very first update.
    new_slow, new_updates = _slow_update(
        slow, new_critic, moments[SLOW_KEY], slow_fraction, slow_period
    )
    stepped = dataclasses.replace(
        system._write_slots(params, {CRITIC_KEY: (new_critic, new_slow)}),
        policy_prior=new_actor,
    )
    moments[RETNORM_KEY] = new_retnorm
    moments[SLOW_KEY] = new_updates
    return stepped, LearnState(opt_states=opt, moments=moments), metrics


def roll_batch(
    system: System,
    params: SystemParams,
    actor_m,
    critic_m,
    start_beliefs: tuple,
    keys: Array,
    *,
    horizon: int,
    bounds: tuple[Array, Array] | None,
    cont: tuple[PredictorHead, tuple[int, ...]],
    code_and_carry: tuple[int, int],
    aux: Mapping[str, tuple[Array, ...]] | None,
    expert_actions: Array,
    sample_scoring: bool = False,
) -> Rollout:
    """A batch of sampled imagined trajectories: one :class:`Rollout` with a leading axis.

    Shared by :func:`imagine` and :func:`rollout_returns`. ``actor_m``/``critic_m``
    are passed explicitly so the gradient can substitute traced copies; the critic
    is stop-gradiented here, and so is the actor's conditioning state at every step
    (the actor's gradient flows through its reparameterised draw and the dynamics).
    """
    code, carry = code_and_carry
    cont_head, cont_deps = cont
    critic_sg = jax.lax.stop_gradient(critic_m)

    def one(k: Array, start: tuple, aux_start, expert: Array) -> Rollout:
        # The scoring closures are built HERE, per start state, because a row may
        # need *this* start's auxiliary beliefs — under vmap the Python runs once and
        # the tracers carry the batch axis. The mapping goes through by row name; no
        # row name is written down in this module.
        #  The scoring rows' key is DERIVED from the start's key (folded in with a tag no
        #  `split` reaches — see SCORING_KEY_TAG), not split off it, so switching the
        #  sampled rollout on leaves the actor's own draws — and hence the imagined
        #  trajectories — bitwise as they were: the switch isolates the rows.
        scoring = system._amortized_scoring(
            params,
            aux_start,
            key=jax.random.fold_in(k, SCORING_KEY_TAG) if sample_scoring else None,
            horizon=horizon,
            face="imagine",
        )
        step_terms = scoring.step_terms_fn

        def step(beliefs: tuple, xs: tuple[Array, Array]):
            step_key, t = xs
            act_key, code_key = jax.random.split(step_key)
            feat = features(beliefs)
            #  The actor's CONDITIONING state is stop-gradiented at every step, which cuts
            #  the path `a_k -> s_{k+1} -> (actor input at k+1) -> a_{k+1}`. The state
            #  chain itself stays live, so reward gradients through the dynamics are
            #  untouched; what goes is the actor-input path only.
            policy = actor_m.condition_on((jax.lax.stop_gradient(feat),))
            action = policy.sample(act_key)  # reparameterized: the gradient flows through the draw
            #  The TRANSITION sees the action brought into the box by a stop-gradiented
            #  RESCALE (`soft_clip`); every other reader — the per-step term context, hence
            #  any row conditioning on the action — sees the RAW draw. A hard `clip` would be
            #  flat outside the box and discard the advantage's pathwise gradient to the
            #  actor on every clipping step; the rescale keeps it, at strength `1/|a|`.
            new, logits, probs = _sampled_step(
                params, system, carry, code, beliefs, soft_clip(action, bounds), code_key
            )
            dist = tuple(probs if f == code else new[f] for f in range(len(new)))
            value, per_term = term_value(step_terms(new, action, t, beliefs, dist))
            return new, (
                value,
                per_term,
                critic_sg.predict((features(new),)),
                _cont_mode(cont_head.predict(_amortized.head_parents(new, cont_deps))),
                factored_categorical_entropy(logits),
                policy.entropy().midpoint,
                feat,
            )

        ts = jnp.arange(horizon, dtype=jnp.int32)
        _, out = jax.lax.scan(step, start, (jax.random.split(k, horizon), ts))
        value, per_term, v_next, continuation, h_prior, h_pi, feats = out
        v0 = critic_sg.predict((features(start),))
        return Rollout(
            value=value,
            per_term=per_term,
            critic=jnp.concatenate([v0[None], v_next]),
            cont=continuation,
            prior_entropy=h_prior,
            policy_entropy=h_pi,
            start_logprob=actor_m.log_prob(expert, (features(start),)),
            start_features=feats,
        )

    return jax.vmap(one, in_axes=(0, 0, 0 if aux is not None else None, 0))(
        keys, start_beliefs, aux, expert_actions
    )


class ReturnEstimates(NamedTuple):
    """Per-state λ-returns and critic values over a batch of imagined rollouts.

    Both ``(N, H)``: entry ``[i, k]`` is the return estimate at, and the critic's
    value of, the ``k``-th state of trajectory ``i``. ``value`` is the per-step
    G-as-a-value the return was folded from.
    """

    ret: Array
    critic: Array
    value: Array


def rollout_returns(
    system: System,
    params: SystemParams,
    start_beliefs: Mapping[str, Array],
    *,
    key: Array,
    horizon: int | None = None,
    aux_beliefs: Mapping[str, Mapping[str, Array]] | None = None,
    cont_modality: str | None = None,
    discount: float = DEFAULT_DISCOUNT,
    return_lambda: float = DEFAULT_LAMBDA,
    sample_scoring: bool = False,
) -> ReturnEstimates:
    """The λ-returns and critic values an :func:`imagine` step would fit, without stepping.

    Arguments mean what they mean in :func:`imagine`, ``sample_scoring`` included.
    """
    start_beliefs = keyed.beliefs_in(system._names(), start_beliefs, "rollout_returns")
    horizon = int(system.policy_len if horizon is None else horizon)
    actor = system._habit_module(params)
    critic = params.scoring[ROW][0]
    cont_head, cont_deps = _cont_head(system, params, cont_modality)
    structure = system._amortized_structure(params, "imagine")
    code, carry = _code_and_carry(structure, system)
    layout = system._control_layout(params)
    n = int(jnp.shape(start_beliefs[carry])[0])
    rolls = roll_batch(
        system,
        params,
        actor,
        critic,
        tuple(start_beliefs),
        jax.random.split(key, n),
        horizon=horizon,
        bounds=None if layout is None else layout.bounds,
        cont=(cont_head, cont_deps),
        code_and_carry=(code, carry),
        aux=None if not aux_beliefs else system._aux_in(aux_beliefs, "rollout_returns"),
        expert_actions=jnp.zeros((n, actor.child_dim), jnp.float32),
        sample_scoring=sample_scoring,
    )
    ret = jax.vmap(lambda v, r, d: lambda_return(v, r, d, return_lambda))(
        rolls.critic, rolls.value, discount * rolls.cont
    )
    return ReturnEstimates(ret=ret, critic=rolls.critic[:, :-1], value=rolls.value)


# ------------------------------------------------------------------------- helpers


def _sampled_step(params, system, carry: int, code: int, beliefs, action, key):
    """One sampled imagination step: ``(beliefs', prior logits, prior probs)``.

    The arithmetic of ``inference.amortized.predict``, except that the code is
    carried forward as a straight-through draw; the probabilities are returned for
    leaves that read the distribution.
    """
    del carry  # the declared families say which factor is the carry
    params = system._p(params)
    structure = system._amortized_structure(params, "imagine")
    new, logits = _amortized._predict(
        params.transition,
        system._B,
        structure.families,
        system._ctl("imagine"),
        system._num_controls(params, "imagine"),
        system._control_layout(params),
        tuple(beliefs),
        jnp.asarray(action, jnp.float32),
        key,
    )
    return new, logits[code], jax.nn.softmax(logits[code], axis=-1)


def _code_and_carry(structure, system: System) -> tuple[int, int]:
    """The one code factor and the one carry node this rollout advances."""
    carries = [f for f, fam in enumerate(structure.families) if fam == _amortized.CARRY]
    codes = [f for f, fam in enumerate(structure.families) if fam == _amortized.CODE]
    facs = system._facs
    if len(carries) != 1 or len(codes) != 1:
        raise ValueError(
            "imagine rolls one carry node and one code factor; this System "
            f"declares carries {[facs[f] for f in carries]} and codes {[facs[f] for f in codes]}"
        )
    return codes[0], carries[0]


def _cont_head(
    system: System, params: SystemParams, name: str | None
) -> tuple[PredictorHead, tuple[int, ...]]:
    """The continuation head and its ``A_deps``: what discounts the λ-return.

    Unnamed, the unique ``binary`` head is used; none, several, or a named
    non-``binary`` head raises.
    """
    params = system._p(params)
    mods = system.modality_names
    if name is not None:
        if name not in mods:
            raise ValueError(f"imagine: unknown cont_modality {name!r}; the modalities are {mods}")
        m = mods.index(name)
    else:
        binary = [
            i
            for i, head in enumerate(params.likelihood)
            if isinstance(head, PredictorHead) and head.loss == "binary"
        ]
        if len(binary) != 1:
            raise ValueError(
                "the λ-return is discounted by the *predicted continuation*, which is a binary "
                f"head; this System declares {len(binary)} of them "
                f"({[mods[i] for i in binary]}). Name it with cont_modality=."
            )
        m = binary[0]
    head = params.likelihood[m]
    if not isinstance(head, PredictorHead) or head.loss != "binary":
        raise ValueError(
            f"modality {mods[m]!r} is a {type(head).__name__}, not the binary continuation head "
            "the λ-return needs"
        )
    return head, system.A_deps[m]


def _bc_inputs(
    start_actions: Array | None, bc_weights: Array | None, n: int, action_dim: int, scale
) -> tuple[Array, Array]:
    """``(weights, actions)`` for the self-imitation term, with the no-BC case zeroed.

    Raises when only one of the two is given. With neither, the weights are zeros
    (``log π`` is still computed and reported).
    """
    if (start_actions is None) != (bc_weights is None):
        raise ValueError(
            "imagine: pass start_actions and bc_weights together or neither — the BC "
            "term is −log pi(a_start) * bc_weights, so one without the other is half a "
            "term (design: amortization.md A10 disposition 8)"
        )
    if start_actions is None:
        return jnp.zeros((n,), jnp.float32), jnp.zeros((n, action_dim), jnp.float32)
    actions = jnp.asarray(start_actions, jnp.float32)
    flags = jnp.asarray(bc_weights, jnp.float32)
    if actions.shape != (n, action_dim) or flags.shape != (n,):
        raise ValueError(
            f"imagine: start_actions must be ({n}, {action_dim}) and bc_weights ({n},), "
            f"got {actions.shape} and {flags.shape}"
        )
    return flags, actions


def _resolve(learn_state: LearnState | None, actor, critic, optimizer) -> dict[str, object]:
    if learn_state is None:
        raise ValueError(
            "imagine needs its optimizer carry: pass "
            "learn_state=system.init_learn_state(params) on the first call and thread the "
            "returned LearnState through thereafter. None does not silently re-initialize adam "
            "(design: amortization.md A2's explicit-init rule)."
        )
    opt = dict(learn_state.opt_states)
    missing = [k for k in (ACTOR_KEY, CRITIC_KEY) if k not in opt]
    if missing:
        raise ValueError(
            f"imagine: learn_state.opt_states is missing {missing} (present: {sorted(opt)}). "
            "system.init_learn_state(params) seeds the actor's and the critic's adam state, so "
            "this LearnState was built by hand or passed through a trainer that returned "
            "LearnState without carrying opt_states. A missing state is not "
            "re-seeded here: restarting adam mid-run silently resets its moments every step."
        )
    return opt


def _moments(learn_state: LearnState) -> dict[str, object]:
    """The mutable ``moments`` map for this call; raises if either entry is missing.

    A missing entry is never re-seeded, since restarting the normalizer mid-run
    would silently rescale the advantage.
    """
    moments = dict(learn_state.moments)
    missing = [k for k in MOMENT_KEYS if k not in moments]
    if missing:
        raise ValueError(
            f"imagine: learn_state.moments is missing {missing} (present: "
            f"{sorted(moments)}). system.init_learn_state(params) seeds both entries when "
            "the g_tail row is mounted, so this LearnState was either built by hand or passed "
            "through a trainer between init and here that returned LearnState without "
            "carrying `moments`. Fix: start from "
            "system.init_learn_state(params) and thread every trainer's returned LearnState "
            "through unchanged. A missing entry is not re-seeded here, because restarting the "
            "return normalizer or the slow-critic counter mid-run silently rescales the actor's "
            "advantage."
        )
    return moments


def _clip(grads, clip: float):
    clipper = optax.clip_by_global_norm(clip)
    out, _ = clipper.update(grads, clipper.init(grads))
    return out


def _static_zero(rate) -> bool:
    return isinstance(rate, int | float) and float(rate) == 0.0


def _slow_update(slow, fast, updates: Array, fraction: float, period: int):
    """DreamerV3's slow-target update: ``(slow, updates + 1)``.

    ::

        need_init   = (updates == 0)
        need_update = (updates % period == 0)
        mix = clip(1.0 * need_init + fraction * need_update, 0, 1)
        slow = mix * fast + (1 - mix) * slow

    so the first call hard-copies the freshly stepped critic. Design:
    docs/design/amortization.md chunk 2d §(a″) item 13
    """
    need_init = (updates == 0).astype(jnp.float32)
    need_update = (jnp.mod(updates, int(period)) == 0).astype(jnp.float32)
    mix = jnp.clip(1.0 * need_init + fraction * need_update, 0.0, 1.0)
    mixed = jax.tree_util.tree_map(lambda s, f: mix * f + (1.0 - mix) * s, slow, fast)
    return mixed, updates + 1
