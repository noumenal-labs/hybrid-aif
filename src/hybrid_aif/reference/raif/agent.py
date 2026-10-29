"""The assembled R-AIF agent, and its training step.

The five parameter sets of their training step and where they live:

======================  ==============================================  ==========
θ  world model          the declared slots (``hybrid_aif.reference.raif.world``)  group 1
ψ  actor                ``SystemParams.policy_prior``                   group 2
χ  critic + slow copy   ``scoring["g_tail"]``                           group 3
ω  info-gain ensemble   ``scoring["model_uncertainty"]``               group 3
φ  CRSPP                ``scoring["cosine_sim_to_pref_rollout"]``       group 3
======================  ==============================================  ==========

Sizes are the oracle's resolved config: the actor is 3 × 256, the critic and each ensemble
member 2 × 32, and the world model is ``tiny``.

Design: ``docs/experiments/raif-mountain-car/module-correspondence.md``.
"""

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.controls import soft_clip
from hybrid_aif.distributions import MLPConditional
from hybrid_aif.distributions.neural import PredictorHead
from hybrid_aif.reference.raif.columns import CAN_SELF_IMITATE, PREF_LABEL
from hybrid_aif.reference.raif.crspp import ROW as CRSPP_ROW
from hybrid_aif.reference.raif.crspp import STATE_KEY as CRSPP_STATE_KEY
from hybrid_aif.reference.raif.crspp import CRSPPLosses, embeddings, fit_crspp
from hybrid_aif.reference.raif.world import CODE_NAME, crspp_rssm, init_of, tiny_rssm
from hybrid_aif.system import LearnState, System, SystemParams
from hybrid_aif.train.buffers import Episode
from hybrid_aif.train.ensemble import EnsembleLosses, fit_ensemble, transitions
from hybrid_aif.train.imagine import ImagineMetrics, imagine
from hybrid_aif.train.world import (
    WorldBatch,
    WorldLosses,
    assemble,
    fit_world,
)

#: The three head sizes, each from its own block of the oracle's resolved config
#: (``raif-paper/logs/oracle_state_mc_seed0/config.yaml``)::
#:
#:     actor:   {layers: 3, units: 256, outscale: 1.0, act: silu, dist: normal}
#:     critic:  {layers: 2, units:  32, outscale: 0.0, dist: symlog_disc, bins: 255}
#:     ig_head: {layers: 2, units:  32, outscale: 1.0, dist: normal}
ACTOR_LAYERS, ACTOR_UNITS = 3, 256
CRITIC_LAYERS, CRITIC_UNITS = 2, 32
ENSEMBLE_LAYERS, ENSEMBLE_UNITS = 2, 32

#: The actor's size under its old name. ``raif_mountain_car(head_units=…,
#: head_layers=…)`` still overrides **all three** at once, which is what the test and
#: smoke sizes want; the parity build reads the three pairs above.
HEAD_LAYERS = ACTOR_LAYERS
HEAD_UNITS = ACTOR_UNITS

#: ``ig_models: 4`` (resolved config; ``configs.yaml:102-106``).
ENSEMBLE_SIZE = 4

#: The actor head's std band, ``actor: {minstd: 0.1, maxstd: 1.0}``: their head is
#: ``Normal(tanh(out), minstd + (maxstd − minstd)·σ(raw + 2))``, which
#: :class:`~hybrid_aif.distributions.MLPConditional`'s ``head="bounded"`` implements.
ACTOR_MINSTD, ACTOR_MAXSTD = 0.1, 1.0

#: The ensemble heads' σ, a constant: ``ig_head`` declares no ``minstd``/``maxstd``,
#: so their defaults (1.0, 1.0) give ``std ≡ 1.0`` and the member NLL is a scaled MSE.
ENSEMBLE_STD = 1.0

#: Their action box, ``Box(-1, 1, (1,))``, and so the magnitude of their
#: ``rssm.action_clip: 1.0``. Read by :func:`to_world_batch`.
ACTION_LOW, ACTION_HIGH = -1.0, 1.0

#: Their Mountain Car training protocol (addendum §5), for the recipe's defaults.
BATCH_SIZE = 16
BATCH_LENGTH = 64
IMAG_HORIZON = 8
TRAIN_EVERY = 16  # train_ratio 64 over batch_steps 1024 = 1 gradient step / 16 agent steps

#: Their optimizer groups (addendum §5): the world model and the CRSPP at ``adam 1e-4``
#: (``clip 1000``), the actor and critic at ``adam 3e-5`` (``clip 100``), the info-gain
#: ensemble at ``adam 1e-4``. Defaults of :func:`train_step`'s arguments.
WORLD_LR = 1e-4
ACTOR_LR = 3e-5
ENSEMBLE_LR = 1e-4

#: The rest of those optimizer lines, from the oracle's resolved config
#: (``logs/oracle_state_mc_seed0/config.yaml``)::
#:
#:     model_opt:  {clip: 1000.0, eps: 1e-08, lr: 0.0001, opt: adam, wd: 0.0}
#:     actor_opt:  {clip:  100.0, eps: 1e-05, lr: 3e-05,  opt: adam, wd: 0.0}
#:     critic_opt: {clip:  100.0, eps: 1e-05, lr: 3e-05,  opt: adam, wd: 0.0}
#:     ig_opt:     {clip:  100.0, eps: 1e-05, lr: 0.0001, opt: adam, wd: 0.0}
#:
#: ``eps`` differs by group: ``1e-8`` on the world model (optax's default) and ``1e-5``
#: on the amortized heads, passed explicitly. ``ACTOR_EPS`` covers the critic too:
#: :func:`~hybrid_aif.train.imagine.imagine` takes one ``adam_eps`` for the pair.
ACTOR_EPS = 1e-5
ENSEMBLE_EPS = 1e-5
ENSEMBLE_CLIP = 100.0

#: The self-imitation term's scale (``configs.yaml:79``; A10 disposition 8).
BC_SCALE = 0.5

#: Paper Eq. 10's η and ζ: the policy-entropy bonus (their ``actent``) and the
#: state-prior-entropy bonus (their ``wment``), both 3e-4 in the resolved config.
ACTENT = 3e-4
WMENT = 3e-4

#: The rows this agent mounts: their critic's value ``reward + sim − IG``, closed at the
#: horizon by the critic (paper Eq. 6, all ``G_scales`` 1.0). ``model_uncertainty`` is
#: the ``−IG``: the ensemble-disagreement estimator under a term of sign +1, so it enters
#: G as a cost and the curiosity row is not mounted. ``occupancy`` is absent because the
#: state-prior entropy is the actor's ``wment`` bonus, not part of the critic's value.
#: Design: ``docs/design/amortization-p0-addendum.md`` §3.
TERMS = (
    "predicted_reward:obs:reward",
    "cosine_sim_to_pref_rollout",
    f"model_uncertainty:{CODE_NAME}",
    "g_tail",
)


class Agent(NamedTuple):
    """Everything a training loop threads: the declared agent and its optimizer carry.

    ``system``/``params`` are the agent with the G side mounted. ``crspp``/``crspp_params``
    are the preference model's pair, also reachable as ``system.scoring``/``params.scoring``
    under the CRSPP row and named here for :func:`~hybrid_aif.reference.raif.crspp.fit_crspp`.
    ``learn_state`` is the one optimizer carry for all five parameter sets.
    """

    system: System
    params: SystemParams
    crspp: System
    crspp_params: SystemParams
    learn_state: LearnState


def raif_mountain_car(
    key: Array,
    *,
    obs_dim: int = 2,
    action_dim: int = 1,
    horizon: int = IMAG_HORIZON,
    head_layers: int | None = None,
    head_units: int | None = None,
    actor_layers: int = ACTOR_LAYERS,
    actor_units: int = ACTOR_UNITS,
    critic_layers: int = CRITIC_LAYERS,
    critic_units: int = CRITIC_UNITS,
    ensemble_layers: int = ENSEMBLE_LAYERS,
    ensemble_units: int = ENSEMBLE_UNITS,
    ensemble_size: int = ENSEMBLE_SIZE,
    units: int = 32,
    num_bins: int = 255,
    bin_low: float = -20.0,
    bin_high: float = 20.0,
    winit: str = "normal",
    **tiny: object,
) -> Agent:
    """R-AIF's agent for state-based Mountain Car: ``Agent(system, params, crspp, …, learn_state)``.

    ``winit`` is the initializer of the world model, the CRSPP and the critic (``"normal"``
    theirs, ``"default"`` eqx's); the actor and ensemble use ``"normal"`` either way. ``tiny``
    keyword arguments go to :func:`~hybrid_aif.reference.raif.world.tiny_core`, and the CRSPP is
    built at the same sizes with the agent's ``embed``. ``head_layers``/``head_units``
    override all three head sizes at once; ``None`` keeps each head's own. The planner is
    ``"habit"``: G trains the actor through ``imagine`` and is not an act-time score. The
    ensemble enters G as the ``model_uncertainty`` cost, so ``epistemic=False`` (:data:`TERMS`).

    Design: ``docs/experiments/raif-mountain-car/divergence-register.md`` (divergences 17, 26).
    """
    if head_layers is not None:
        actor_layers = critic_layers = ensemble_layers = int(head_layers)
    if head_units is not None:
        actor_units = critic_units = ensemble_units = int(head_units)
    keys = jax.random.split(key, 5 + int(ensemble_size))
    system, params = tiny_rssm(
        obs_dim, action_dim, key=keys[0], policy_len=horizon, units=units, winit=winit, **tiny
    )
    core, _ = assemble(system, params)
    flat, feature_dim = core.prior.flat_dim, core.deter_dim + core.prior.flat_dim
    crspp, crspp_params = crspp_rssm(
        key=keys[1],
        policy_len=horizon,
        deter=core.deter_dim,
        num_vars=core.num_vars,
        num_classes=core.num_classes,
        units=units,
        embed=core.embed_dim,  # its posterior reads the agent's embedding
        winit=winit,  # their CRSPP_RSSM carries `winit: normal` as the world model does
    )
    critic = _value_head(
        keys[2], feature_dim, critic_units, critic_layers, num_bins, bin_low, bin_high, winit
    )
    slow = _value_head(
        keys[3], feature_dim, critic_units, critic_layers, num_bins, bin_low, bin_high, winit
    )
    #  the actor and the ensemble members at their own blocks' resolved settings: `norm:
    #  layer` hidden blocks, the bounded `dist: normal` head and the `winit: normal / fan:
    #  avg` initializer at `outscale: 1.0`. The actor's std lies in (ACTOR_MINSTD,
    #  ACTOR_MAXSTD); the ensemble's is the constant ENSEMBLE_STD.
    actor = MLPConditional.from_shapes(
        (feature_dim,),
        action_dim,
        key=keys[4],
        width=actor_units,
        depth=actor_layers,
        activation="silu",
        norm=True,
        head="bounded",
        minstd=ACTOR_MINSTD,
        maxstd=ACTOR_MAXSTD,
        init=init_of("normal"),
    )
    ensemble = tuple(
        MLPConditional.from_shapes(
            (core.deter_dim, flat, action_dim),
            flat,
            key=k,
            width=ensemble_units,
            depth=ensemble_layers,
            activation="silu",
            norm=True,
            head="bounded",
            minstd=ENSEMBLE_STD,
            maxstd=ENSEMBLE_STD,
            init=init_of("normal"),
        )
        for k in keys[5 : 5 + int(ensemble_size)]
    )
    system = dataclasses.replace(
        system,
        planner="habit",  # act from the actor: G trains, it does not select (A12 item 3)
        epistemic=False,  # the curiosity row is the ensemble's — see TERMS
        terms=TERMS,
        scoring=(("cosine_sim_to_pref_rollout", crspp),),
    )
    params = dataclasses.replace(
        params,
        policy_prior=actor,
        scoring={
            "g_tail": (critic, slow),
            "model_uncertainty": ensemble,
            "cosine_sim_to_pref_rollout": crspp_params,
        },
    )
    system._validate(params)
    return Agent(
        system=system,
        params=params,
        crspp=crspp,
        crspp_params=crspp_params,
        learn_state=system.init_learn_state(params),
    )


def _value_head(
    key: Array,
    feature_dim: int,
    units: int,
    layers: int,
    num_bins: int,
    low: float,
    high: float,
    winit: str = "default",
) -> PredictorHead:
    """A 255-bin ``symlog_disc`` value head over ``[deter, flat stoch]`` (A12 item 5)."""
    return PredictorHead.from_shapes(
        (feature_dim,),
        loss="symlog_disc",
        num_bins=num_bins,
        bin_low=low,
        bin_high=high,
        key=key,
        width=units,
        depth=layers,
        activation="silu",
        out_scale=0.0,  # their critic outscale 0.0: a fresh critic predicts ~0
        init=init_of(winit),
    )


class StepLosses(NamedTuple):
    """One training step's four loss reports, one per parameter set."""

    world: WorldLosses
    imagine: ImagineMetrics
    ensemble: EnsembleLosses
    crspp: CRSPPLosses


def train_step(
    agent: Agent,
    batch: Episode,
    *,
    key: Array,
    horizon: int = IMAG_HORIZON,
    world_lr: float | Array = WORLD_LR,
    actor_lr: float | Array = ACTOR_LR,
    critic_lr: float | Array = ACTOR_LR,
    ensemble_lr: float | Array = ENSEMBLE_LR,
    crspp_lr: float | Array = WORLD_LR,
    bc_scale: float = BC_SCALE,
    sample_scoring: bool = False,
    actor_eps: float = ACTOR_EPS,
    ensemble_eps: float = ENSEMBLE_EPS,
    ensemble_clip: float | None = ENSEMBLE_CLIP,
) -> tuple[Agent, StepLosses]:
    """One gradient step of all five parameter sets, in their order: ``(agent, StepLosses)``.

    ``batch`` is one replay draw as a batched :class:`~hybrid_aif.train.buffers.Episode`
    (leaves ``(B, T, ...)``). :func:`to_world_batch` shifts its actions to the ``a_{t−1}``
    convention for the recurrent trainers; the BC term reads ``action`` in the ``a_t``
    convention with ``can_self_imitate`` (by then the imitation mask), and the CRSPP reads
    ``pref_label``. Both columns come from :func:`hybrid_aif.reference.raif.revise.label`
    at collection time. The order::

        1. fit_world       the sequence loss over the declared slots; its forward
                           pass's posterior states are the ONE recognition 2-4 read
        2. fit_ensemble    Gaussian NLL on (s_t, a_t) -> s_{t+1}
        3. fit_crspp       the rho-gated KL + the contrastive cosine term; its
                           forward pass's posterior is the cosine row's start
        4. imagine         actor + critic on sampled rollouts from those states

    Steps 2-4 read step 1's pre-step states, gradient stopped, and do not differentiate the
    world model; step 4 starts the cosine row from step 3's pass. Collection, routing and
    batch sourcing are the caller's (:mod:`hybrid_aif.reference.raif.run`); every
    world-model batch is drawn from ``M`` in the parity recipe.

    Design: ``docs/experiments/raif-mountain-car/execution-ledger-training.md`` §21.
    """
    #  five-way, two unused: the world, CRSPP and imagination streams keep their draws
    k_world, _, k_crspp, _, k_imagine = jax.random.split(key, 5)
    system = agent.system
    world_batch = to_world_batch(batch)
    params, learn_state, world, states = fit_world(
        system,
        agent.params,
        agent.learn_state,
        world_batch,
        key=k_world,
        lr=world_lr,
        return_states=True,
    )
    beliefs = system.beliefs_from_states(params, states)  # {factor: (B, T, ...)}

    params, learn_state, ens = fit_ensemble(
        system,
        params,
        learn_state,
        #  the raw shifted action, not `world_batch.action`: the ensemble reads the
        #  replay column unclipped; only the recurrent scan sees the rescaled one
        transitions(states.deter, states.stoch, batch.prev_actions()),
        lr=ensemble_lr,
        clip=ensemble_clip,  # their ig_opt clip: 100.0
        adam_eps=ensemble_eps,  # their ig_opt eps: 1e-5
    )

    #  the pre-step encoder, like the states: their CRSPP reads the world loss's own `embed`
    embeds = embeddings(system, agent.params, world_batch.obs)
    crspp_params, crspp_state, crspp_losses, crspp_posterior = fit_crspp(
        agent.crspp,
        params.scoring[CRSPP_ROW],
        learn_state.opt_states[CRSPP_STATE_KEY],
        embeds,
        batch.extras[PREF_LABEL],
        jax.vmap(jax.vmap(lambda d, s: jnp.concatenate([d, s.reshape(-1)])))(
            states.deter, states.stoch
        ),
        key=k_crspp,
        is_first=batch.is_first,
        lr=crspp_lr,
        return_states=True,
    )
    params = system._write_slots(params, {CRSPP_STATE_KEY: crspp_params})
    opt = dict(learn_state.opt_states)
    opt[CRSPP_STATE_KEY] = crspp_state
    learn_state = LearnState(opt_states=opt, moments=learn_state.moments)

    n = states.deter.shape[0] * states.deter.shape[1]
    flat = {k: jnp.reshape(x, (n, *x.shape[2:])) for k, x in beliefs.items()}
    flat_crspp = tuple(jnp.reshape(x, (n, *x.shape[2:])) for x in crspp_posterior)
    params, learn_state, metrics = imagine(
        system,
        params,
        flat,
        learn_state,
        key=k_imagine,
        horizon=horizon,
        # the action taken AT step t is episodes.action[t]; the start states are the
        # posterior states at the same indices, so no re-indexing is needed here —
        # WorldBatch's a_{t-1} convention is the one that shifts (see prev_actions()).
        start_actions=system.from_positional("actions", jnp.reshape(batch.action, (n, -1)), params),
        bc_weights=jnp.reshape(batch.extras[CAN_SELF_IMITATE].astype(jnp.float32), (n,)),
        #  their `first_cont = 1 - start['is_terminal']` (agent.py:244), the leading
        #  factor of the per-step rollout weight. Passed in the `a_t` convention (the
        #  start states' own indices), NOT the shifted world-model one.
        start_cont=jnp.reshape(batch.cont.astype(jnp.float32), (n,)),
        #  the seam is keyed by row name; the carry is keyed by the CRSPP's factor names.
        aux_beliefs={CRSPP_ROW: agent.crspp.from_positional("beliefs", flat_crspp)},
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        bc_scale=bc_scale,
        policy_entropy_scale=ACTENT,
        prior_entropy_scale=WMENT,
        sample_scoring=sample_scoring,
        adam_eps=actor_eps,  # their actor_opt/critic_opt eps: 1e-5
    )
    return (
        agent._replace(
            params=params, crspp_params=params.scoring[CRSPP_ROW], learn_state=learn_state
        ),
        StepLosses(world=world, imagine=metrics, ensemble=ens, crspp=crspp_losses),
    )


def to_world_batch(batch: Episode) -> WorldBatch:
    """A batched :class:`~hybrid_aif.train.buffers.Episode` as a
    :class:`~hybrid_aif.train.world.WorldBatch`.

    Shifts replay's ``a_t`` (the action taken at ``obs[t]``) to ``a_{t−1}`` (the action that
    produced it) with :meth:`~hybrid_aif.train.buffers.Episode.prev_actions`, and carries
    ``is_first`` so :meth:`~hybrid_aif.inference.rssm.RSSMCore.observe` restarts the carry where
    a window crosses an episode boundary. The shifted action is rescaled into the action box
    with :func:`~hybrid_aif.controls.soft_clip`, as their ``obs_step`` does; the recorded column
    stays raw.

    Design: ``docs/experiments/raif-mountain-car/execution-ledger-acting.md`` D5.
    """
    prev = batch.prev_actions()
    box = (jnp.full(prev.shape[-1:], ACTION_LOW), jnp.full(prev.shape[-1:], ACTION_HIGH))
    return WorldBatch(
        obs=batch.obs,
        action=soft_clip(prev, box),
        reward=batch.reward,
        cont=batch.cont,
        is_first=batch.is_first,
    )


def act(agent: Agent, beliefs, key: Array) -> dict[str, Array]:
    """The habit's action at ``beliefs``: ``System.act`` under ``planner="habit"``.

    ``beliefs`` and the action are keyed as the faces key them. ``plan_key`` and the action key
    are split from ``key``, not shared.
    """
    act_key, plan_key = jax.random.split(key)
    action, _, _ = agent.system.act(
        act_key, agent.params, beliefs, plan_key=plan_key, metrics=False
    )
    return action
