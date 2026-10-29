"""The resumable multi-seed Mountain Car run, and its report and benchmark modes.

One chunk is :data:`CHUNK_STEPS` pooled agent steps per seed, jitted and vmapped over
seeds: ``n_blocks`` × [``train_every`` pooled collect steps → one :func:`train_step`].
Each seed collects from :data:`LANES` independent environment streams that share its
parameters and pool its step count, so a block is ``train_every // lanes`` steps per lane.
Collection auto-resets in the scan and records the fixture format.

Between chunks, on the host: episodes are split out per lane at ``is_last`` (a chunk
boundary's partial episode is carried), labelled by
:func:`~hybrid_aif.reference.raif.revise.label` and routed (success → both buffers, failure
→ ``M``). Every world-model batch comes from ``M`` unless ``--alternate``. Both buffers are
:class:`~hybrid_aif.train.buffers.StreamBuffer` rings of :data:`CAPACITY` steps, and each
seed's are seeded once from its oracle fixture through the same routing.

A checkpoint (every ``--checkpoint-every`` minutes and at shutdown, SIGINT/SIGTERM
included) holds every numeric carrier, the buffers, the keys, the counters and the partial
episodes, so ``--resume`` continues the run; bitwise continuation is not asserted.

``<out>/scores.jsonl``   per finished episode: ``{step, seed, lane, return, length, success,
                         stability, log_entropy, coverage}``. ``step`` is the pooled agent
                         step it ended on; ``return`` the raw summed reward; ``stability``
                         R-S's ``C``; ``log_entropy`` the episode's mean act-time policy
                         entropy (:func:`act_entropy`); ``coverage`` the lane's cumulative
                         grid fraction (:class:`Coverage`).
``<out>/metrics.jsonl``  per chunk, per seed: the trainers' losses and :func:`probe`'s
                         diagnostics.
``<out>/checkpoint/``    the resumable state.
``<out>/run.json``       the resolved configuration and the launching commit.

Modes: no flag = train; ``--report`` = the §7 table from ``scores.jsonl``;
``--benchmark`` = the throughput table (:func:`benchmark`)::

    python -m hybrid_aif.reference.raif.run --seeds 0,1,2,3 --steps 1000000 --out runs/mc_2d
    python -m hybrid_aif.reference.raif.run --out runs/mc_2d --resume
    python -m hybrid_aif.reference.raif.run --out runs/mc_2d --report

Design: ``docs/design/amortization.md`` P1′ chunk 2d;
``docs/design/amortization-p0-addendum.md`` §§5–7.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import pathlib
import signal
import sys
import time
from collections.abc import Mapping
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hybrid_aif.controls import soft_clip
from hybrid_aif.objectives.amortized import features as _amortized_features
from hybrid_aif.reference.envs.mountain_car import MountainCar
from hybrid_aif.reference.raif import metrics as ev
from hybrid_aif.reference.raif.agent import (
    ACTOR_LR,
    BATCH_LENGTH,
    BATCH_SIZE,
    ENSEMBLE_LR,
    IMAG_HORIZON,
    TRAIN_EVERY,
    WORLD_LR,
    Agent,
    raif_mountain_car,
    to_world_batch,
    train_step,
)
from hybrid_aif.reference.raif.columns import EXTRAS, IS_SUCCESS, RECORDED, raif_episode, succeeded
from hybrid_aif.reference.raif.crspp import ROW as CRSPP_ROW
from hybrid_aif.reference.raif.crspp import crspp_recognize, embeddings
from hybrid_aif.reference.raif.fixtures import load_fixture
from hybrid_aif.reference.raif.revise import label
from hybrid_aif.system import LearnState, System, SystemParams
from hybrid_aif.train.buffers import Episode, StreamBuffer, extend, sample_windows, seed_stats
from hybrid_aif.train.imagine import rollout_returns
from hybrid_aif.train.world import assemble, posterior_states

#: Pooled agent steps per device chunk: their ``batch_steps``, and 64 gradient steps.
#: Episodes finished inside a chunk reach the buffers at its end (``--chunk-steps``).
CHUNK_STEPS = 1024

#: Environment lanes per seed: their ``envs: {amount: 4}``, four streams behind one
#: agent with the step count pooled. ``--lanes 1`` collects from a single stream.
LANES = 4

#: Their ``replay_size 5e5`` steps per buffer (§5);
#: :class:`~hybrid_aif.train.buffers.StreamBuffer` counts steps.
CAPACITY = 500_000

#: Their Mountain Car budget: 1M agent steps per seed, 4 seeds (§5). One counted step
#: is one driver transition, reset and terminal rows included, so a 100-action
#: episode counts 101.
DEFAULT_STEPS = 1_000_000
DEFAULT_SEEDS = (0, 1, 2, 3)

#: The probe batch the distillation gap is measured on — smaller than a training
#: batch because it is a diagnostic paid every chunk. See :func:`probe`.
PROBE_BATCH = 4

#: Test/smoke sizes. **Not parity**: `--small` exists so the resume round trip and
#: the throughput ratio are testable in seconds. Their ``tiny`` is the default.
SMALL = dict(
    horizon=4,
    deter=16,
    num_vars=6,
    num_classes=6,
    units=8,
    mlp_units=16,
    embed=16,
    head_units=32,
    head_layers=2,
)


# ============================================================== the per-seed carry


class SeedState(NamedTuple):
    """One seed's entire numeric state, the thing a checkpoint holds; leaves are stacked over seeds.

    ``params``/``learn_state`` carry only the seed axis. The collection carriers carry a further
    leading lane axis (:data:`LANES`). ``key`` is per lane (env noise, action draw, reset);
    ``train_key`` is per seed, one gradient step per seed however many lanes fed it. The
    CRSPP's parameters live in ``params.scoring``, not in a field of their own.
    """

    params: SystemParams
    learn_state: LearnState
    beliefs: dict[str, Array]  # (lanes, ...) the habit's filter carry, per lane, by factor
    env_state: tuple[Array, ...]  # (lanes, ...) MountainCar's ``(x,)``, per lane
    incoming_reward: Array  # (lanes,) float32 — the reward that PRODUCED env_state
    t_in_episode: Array  # (lanes,) int32, against the TimeLimit
    succeeded: Array  # (lanes,) bool, this episode's running has_succeeded
    key: Array  # (lanes, 2) the collection keys
    train_key: Array  # (2,) the gradient step's own stream, per SEED

    def collect_carry(self) -> tuple:
        """This seed's collection carry: the tuple :func:`make_collect_fn` threads."""
        return (
            self.beliefs,
            self.env_state,
            self.incoming_reward,
            self.t_in_episode,
            self.succeeded,
            self.key,
        )


class Stream(NamedTuple):
    """A chunk's recorded collection stream in the fixture format, one row per driver transition.

    The columns are :data:`hybrid_aif.train.buffers.EPISODE_FIELDS` minus ``can_self_imitate``,
    plus the telemetry column ``log_entropy`` (dropped by :func:`as_collected`). From
    :func:`make_chunk_fn` the leaves are ``(seeds, lanes, chunk_steps // lanes, ...)``. ``obs``
    is the row's observation, ``reward`` the reward that produced it, ``action`` the raw sample
    drawn at it and ``is_success`` the goal predicate on it (:func:`make_collect_fn`).
    """

    obs: Array
    action: Array
    reward: Array
    is_success: Array
    has_succeeded: Array
    is_first: Array
    is_last: Array
    #: The act-time differential entropy of the distribution this row's action was
    #: drawn from. Telemetry, not replay data, and computed without a key; its
    #: per-episode mean is their ``stats/mean_log_entropy``.
    log_entropy: Array


# ============================================================ the device-side chunk


#: The collection carry an agent-step scan threads: :class:`SeedState` minus the trained
#: things and the training key, leaves on a leading lane axis.
CollectCarry = tuple


def _action_bounds(system: System, params: SystemParams) -> tuple[Array, Array] | None:
    """The control box in ``params.control``, or ``None``: what the env and the transition clip
    to."""
    layout = system._control_layout(params)
    return None if layout is None else layout.bounds


def episode_start_beliefs(system: System, params: SystemParams) -> dict[str, Array]:
    """The belief carry an episode's first filtering step reads.

    The episode-start state the world model's training restart uses
    (:meth:`~hybrid_aif.inference.rssm.RSSMCore.initial_state`: the learned carry and the
    prior's mode at it, a one-hot code), advanced one GRU step with a zero action, as their
    ``obs_step`` does at ``is_first``. Pure; the collection scan hoists it and broadcasts it
    across lanes.

    Design: ``docs/experiments/raif-mountain-car/execution-ledger-acting.md`` D6.
    """
    core, _ = assemble(system, params)
    initial = system.beliefs_from_states(params, core.initial_state())
    layout = system._control_layout(params)
    dim = 1 if layout is None else int(layout.action_dim)
    return system.update_empirical_prior(
        params,
        initial,
        system.from_positional("actions", jnp.zeros((dim,), jnp.float32), params),
    )


def act_entropy(system: System, params: SystemParams, beliefs: Mapping[str, Array]) -> Array:
    """The act-time policy entropy ``H[pi(a | s_t)]`` at ``beliefs``: a scalar.

    The entropy of the actor's raw ``Normal`` head at the features
    :func:`~hybrid_aif.planning.habit.habit_action` uses, summed over action dimensions. Takes
    no key and draws nothing. Its per-episode mean is their ``stats/mean_log_entropy``; the
    imagined :attr:`hybrid_aif.train.imagine.ImagineMetrics.policy_entropy` is a different
    quantity.

    Design: ``docs/experiments/raif-mountain-car/series-sweep.md`` §5.3.
    """
    module = system._habit_module(params)
    if module is None:
        raise ValueError(
            "act_entropy needs a policy MODULE in SystemParams.policy_prior — a vector E "
            "over an enumerated policy set has no act-time density to take the entropy of"
        )
    features = _amortized_features(system.to_positional("beliefs", beliefs))
    return jnp.sum(module.condition_on((features,)).entropy().midpoint)


#: The tag the act-time code sample's key is folded in with: large, so it cannot coincide
#: with any ``split(k_act, n)[i]`` a planner might take from the same key (``fold_in(k, i)``
#: equals ``split(k, n)[i]`` for small ``i``; see ``train.imagine.SCORING_KEY_TAG``).
ACT_SAMPLE_KEY_TAG = 0x5A11CE5


def make_collect_fn(system: System, env: MountainCar, steps: int, *, act_sampling: bool = True):
    """``(params, carry) → (carry, Stream)`` over ``steps`` steps per lane. Pure.

    ``carry``'s leaves have a leading lane axis of any width; the lanes are vmapped with
    ``params`` and the episode-start belief broadcast. The returned ``Stream`` leaves are
    ``(steps, lanes, ...)``.

    Each lane-step is one transition of their driver. A row is ``(obs_t, reward_t, flags_t,
    a_t)`` with ``reward_t`` caused by ``a_{t−1}``, so an episode of ``T`` applied actions is
    ``T + 1`` rows: a reset row with a real action at zero reward, and a terminal row whose
    observation is the final state, whose action is zeroed and after which the env is not
    stepped.

    ``reward``
        the incoming reward of the previous step's action; 0 on a reset row.
    ``is_success``
        the goal predicate on this row's observation; ``has_succeeded`` is its running OR.
    ``action``
        the raw actor sample, zeroed on ``is_last``. The env receives it clipped and the
        transition :func:`~hybrid_aif.controls.soft_clip`-ed; the recorded column is raw.
    ``is_last``
        the goal predicate on this row, or ``t_in_episode`` reaching the time limit.

    On ``is_last`` a ``jnp.where`` swaps in a fresh env state, :func:`episode_start_beliefs`,
    zero incoming reward and ``t_in_episode = 0``, so episodes straddle chunk boundaries and
    the host recovers them from the flags. With ``act_sampling`` the recognized code is
    sampled (``infer_states(key=)``) and the actor, ``log_entropy`` and the transition read
    that sample; without, the lane acts on the mean-field posterior.

    Design: ``docs/experiments/raif-mountain-car/execution-ledger-acting.md`` D1–D6.
    """
    time_limit = int(env.time_limit)

    def collect(params: SystemParams, carry: CollectCarry):
        #  the reset targets, hoisted out of the scan: a read of params, broadcast over
        #  the lanes by `in_axes=None`
        bounds = _action_bounds(system, params)
        initial = episode_start_beliefs(system, params)

        def lane(inner, initial):
            beliefs, env_state, incoming, t_ep, succeeded, key = inner
            key, k_act, k_plan, k_step, k_reset = jax.random.split(key, 5)
            obs = env.observe(env_state)[0]  # the observation this ROW is
            is_first = t_ep == 0
            #  their MeasureSuccessRate, on this observation
            success = env.reached_goal(env_state)
            ran_on = succeeded | success
            #  their easy_mode termination, or TimeLimit._step reaching `duration`
            is_last = success | (t_ep >= time_limit)
            #  keyed, the posterior is one sample, read by the actor, the entropy and the
            #  carry alike; its key is derived from the act key so no other stream moves
            k_code = jax.random.fold_in(k_act, ACT_SAMPLE_KEY_TAG) if act_sampling else None
            observed = system.from_positional("observations", (obs, None, None))
            beliefs, _ = system.infer_states(params, observed, beliefs, key=k_code)
            #  their `log_entropy`, off the beliefs the actor reads; draws no key
            h_act = act_entropy(system, params, beliefs)
            chosen, _, _ = system.act(k_act, params, beliefs, plan_key=k_plan, metrics=False)
            action = system.to_positional("actions", chosen, params)
            #  their `ClipAction` into the env and their `action_clip` rescale into the
            #  transition; the recorded column stays raw
            _, stepped, reward, _ = env.agent_step(
                k_step, env_state, action if bounds is None else jnp.clip(action, *bounds)
            )
            carried = system.update_empirical_prior(
                params,
                beliefs,
                system.from_positional("actions", soft_clip(action, bounds), params),
            )
            _, fresh = env.reset(k_reset)

            def pick(a, b):
                return jnp.where(is_last, a, b)

            row = Stream(
                obs=obs,
                action=jnp.where(is_last, jnp.zeros_like(action), action),
                reward=incoming,
                is_success=success,
                has_succeeded=ran_on,
                is_first=is_first,
                is_last=is_last,
                log_entropy=h_act,
            )
            return (
                jax.tree_util.tree_map(pick, initial, carried),
                jax.tree_util.tree_map(pick, fresh, stepped),
                jnp.where(is_last, 0.0, reward).astype(jnp.float32),
                jnp.where(is_last, 0, t_ep + 1).astype(jnp.int32),
                ran_on & ~is_last,
                key,
            ), row

        def step(inner, _):
            return jax.vmap(lane, in_axes=(0, None))(inner, initial)

        return jax.lax.scan(step, carry, None, length=int(steps))

    return collect


def make_chunk_fn(
    system: System,
    crspp: System,
    env: MountainCar,
    *,
    chunk_steps: int = CHUNK_STEPS,
    train_every: int = TRAIN_EVERY,
    lanes: int = LANES,
    batch_size: int = BATCH_SIZE,
    horizon: int = IMAG_HORIZON,
    world_lr: float = WORLD_LR,
    actor_lr: float = ACTOR_LR,
    ensemble_lr: float = ENSEMBLE_LR,
    sample_scoring: bool = False,
    act_sampling: bool = True,
):
    """The jitted, seed-vmapped chunk: ``(state, positive, general, use_positive) → …``.

    Returns ``(state, stream, losses)``, the stream's leaves ``(seeds, lanes, chunk_steps //
    lanes, ...)`` with each lane's steps contiguous, and the losses' ``(seeds, n_blocks)``.
    The outer scan is the blocks, with ``use_positive`` (the host's schedule) as its ``xs``;
    the inner is the ``train_every // lanes`` lane-steps, so a gradient step follows every
    ``train_every`` pooled steps. ``positive``/``general`` are constant through the chunk.
    """
    if chunk_steps % train_every:
        raise ValueError(f"chunk_steps {chunk_steps} must be a multiple of {train_every=}")
    if lanes < 1 or train_every % lanes:
        raise ValueError(
            f"lanes must be a positive divisor of train_every={train_every}, got {lanes} — "
            "their Ratio(64/1024) counts POOLED agent steps, so one gradient block is "
            "train_every // lanes steps in each lane and the division must be exact"
        )
    per_lane = train_every // int(lanes)
    collect_fn = make_collect_fn(system, env, per_lane, act_sampling=act_sampling)

    def one_seed(state: SeedState, positive: StreamBuffer, general: StreamBuffer, ups: Array):
        def block(st: SeedState, use_positive: Array):
            collected = collect_fn(st.params, st.collect_carry())
            (beliefs, env_state, incoming, t_ep, succeeded, key), rows = collected

            #  one gradient step: both buffers are sampled and the host's `use_positive`
            #  selects one. The keys come from the seed's `train_key`, not a lane's.
            train_key, k_pos, k_gen, k_train = jax.random.split(st.train_key, 4)
            batch = jax.tree_util.tree_map(
                lambda a, b: jnp.where(use_positive, a, b),
                sample_windows(positive, k_pos, batch_size),
                sample_windows(general, k_gen, batch_size),
            )
            agent = Agent(
                system=system,
                params=st.params,
                crspp=crspp,
                crspp_params=st.params.scoring[CRSPP_ROW],
                learn_state=st.learn_state,
            )
            agent, losses = train_step(
                agent,
                batch,
                key=k_train,
                horizon=horizon,
                world_lr=world_lr,
                actor_lr=actor_lr,
                critic_lr=actor_lr,
                ensemble_lr=ensemble_lr,
                crspp_lr=world_lr,
                sample_scoring=sample_scoring,
            )
            return SeedState(
                params=agent.params,
                learn_state=agent.learn_state,
                beliefs=beliefs,
                env_state=env_state,
                incoming_reward=incoming,
                t_in_episode=t_ep,
                succeeded=succeeded,
                key=key,
                train_key=train_key,
            ), (rows, losses)

        state, (rows, losses) = jax.lax.scan(block, state, ups)
        #  (n_blocks, per_lane, lanes, ...) -> (lanes, n_blocks * per_lane, ...)
        stream = jax.tree_util.tree_map(
            lambda x: jnp.reshape(jnp.moveaxis(x, 2, 0), (int(lanes), -1, *x.shape[3:])), rows
        )
        return state, stream, losses

    return eqx.filter_jit(jax.vmap(one_seed))


# ================================================================= host-side splitting


#: Stream columns a resumed ``pending`` may lack: telemetry, backfilled with NaN. A
#: pending tail missing any other column is a corrupt checkpoint.
OPTIONAL_STREAM_COLUMNS = frozenset({"log_entropy"})


def _carried(pending: dict, column: str, chunk: dict):
    """``pending[column]``, or NaN of the right length where an older run lacked it."""
    if column in pending:
        return pending[column]
    if column not in OPTIONAL_STREAM_COLUMNS:
        raise KeyError(
            f"the resumed pending episode has no {column!r} column, and it is not telemetry "
            f"({sorted(OPTIONAL_STREAM_COLUMNS)} are). This checkpoint's partial episodes cannot "
            "be continued by this code."
        )
    held = len(next(iter(pending.values())))
    return np.full((held, *np.shape(chunk[column])[1:]), np.nan, np.asarray(chunk[column]).dtype)


def split_stream(
    pending: dict[str, np.ndarray] | None, chunk: dict[str, np.ndarray]
) -> tuple[list[dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """``(finished episodes, the new pending tail)``: one lane's stream split on ``is_last``.

    An episode a chunk boundary cut is the tail here and the head of the next chunk's
    ``pending``.
    """
    cols = (
        chunk
        if pending is None
        else {k: np.concatenate([_carried(pending, k, chunk), chunk[k]]) for k in chunk}
    )
    out, start = [], 0
    for end in np.flatnonzero(cols["is_last"]).tolist():
        out.append({k: v[start : end + 1] for k, v in cols.items()})
        start = end + 1
    return out, {k: v[start:] for k, v in cols.items()}


#: Their ``bin_size`` for the ``gym`` suite (``wrappers.py:795``): a 10 × 10 grid over
#: ``(position, velocity)``, so 100 cells.
COVERAGE_BINS = 10


class Coverage:
    """Their ``MeasureCoverage`` for Mountain Car, one instance per lane.

    The observation box (position ``[-1.2, 0.6]``, velocity ``[-0.07, 0.07]``) is cut into
    :data:`COVERAGE_BINS` per axis at the interior edges, the cell id is
    ``COVERAGE_BINS · bin(position) + bin(velocity)``, and coverage is the fraction of cells
    visited. The counter is cumulative over the lane's life, not per episode, and is carried
    across chunks and checkpoints. It reads recorded ``obs`` on the host only.

    Design: ``docs/experiments/raif-mountain-car/series-sweep.md`` §5.3.
    """

    def __init__(self, env: MountainCar, bins: int = COVERAGE_BINS, seen: set[int] | None = None):
        low = np.asarray([env.min_position, -env.max_speed], np.float64)
        high = np.asarray([env.max_position, env.max_speed], np.float64)
        #  their `np.linspace(lows, highs, bin_size + 1)[1:-1]`: interior edges, (bins-1, 2)
        self.edges = np.linspace(low, high, bins + 1)[1:-1]
        self.bins = int(bins)
        self.total = float(bins**2)
        self.seen: set[int] = set(seen or ())

    def update(self, obs: np.ndarray) -> float:
        """Count one episode's rows in order, then return the coverage.

        ``obs`` is ``(steps, 2)``, reset and terminal rows included.
        """
        rows = np.asarray(obs, np.float64).reshape(-1, 2)
        #  their per-factor `np.digitize(state[i], binss[:, i])`, vectorised over rows
        pos = np.digitize(rows[:, 0], self.edges[:, 0])
        vel = np.digitize(rows[:, 1], self.edges[:, 1])
        self.seen.update((pos * self.bins + vel).tolist())
        return len(self.seen) / self.total


def as_collected(cols: dict[str, np.ndarray]) -> Episode:
    """One host-split episode as a labelled :class:`~hybrid_aif.train.buffers.Episode`.

    ``can_self_imitate`` is zero as recorded; :func:`hybrid_aif.reference.raif.revise.label`
    then replaces it with the imitation mask.
    """
    t = int(cols["obs"].shape[0])
    return label(
        raif_episode(
            can_self_imitate=np.zeros((t,), np.float32),
            **{k: cols[k] for k in RECORDED if k != "can_self_imitate"},
        )
    )


def route(positive: StreamBuffer, general: StreamBuffer, ep: Episode, add_fn=extend):
    """Buffer routing: success → both buffers, failure → ``M`` only. Fixture episodes included."""
    general = add_fn(general, ep)
    if succeeded(ep):
        positive = add_fn(positive, ep)
    return positive, general


def alternate(a: object, b: object):
    """``a, b, a, b, …`` — the paper's strict M/M⁺ alternation as an iterator (A10
    disposition 9). Not the parity default; see :func:`alternation`."""
    while True:
        yield a
        yield b


def alternation(grad_index: int, n_blocks: int, ready: np.ndarray) -> Array:
    """The ``--alternate`` schedule: ``(seeds, n_blocks)`` bool. Not the default.

    The paper's strict ``i % 2`` alternation, drawing from ``M`` while a seed's ``M⁺`` is not
    ``ready``. The parity default is :func:`m_only`, because the oracle's alternation is inert.

    Design: ``docs/design/amortization-debt-ledger.md`` entry 16.
    """
    turn = ((np.arange(n_blocks) + int(grad_index)) % 2 == 0)[None, :]
    return jnp.asarray(turn & np.asarray(ready, bool)[:, None])


def m_only(n_seeds: int, n_blocks: int) -> Array:
    """The parity default's ``use_positive`` schedule: all ``False``, every batch from ``M``."""
    return jnp.zeros((int(n_seeds), int(n_blocks)), bool)


# ======================================================================== the probe


def probe(
    system: System,
    crspp: System,
    params: SystemParams,
    batch: Episode,
    key: Array,
    *,
    sample_scoring: bool = False,
    act_sampling: bool = True,
):
    """Diagnostics on a fixed probe batch, per seed:
    ``(ret, critic, gap, value, mu_act, mu_train, sigma_act, sigma_train, mu_gap)``, batch means.

    ``gap`` is the distillation gap ``ret − critic``: the λ-return of the explicit rollout
    against the ``g_tail`` critic's value at the same states, both from
    :func:`~hybrid_aif.train.imagine.rollout_returns`. The actor is also read on two feature
    paths over the same windows: the training features, and the collection lane's filter run
    from a cold start at each window's first step (``episode_start_beliefs`` at ``is_first``,
    ``infer_states``, keyed when ``act_sampling``, then the prior pushed by the recorded
    action). ``mu_*`` are its mean actions, ``sigma_*`` its per-dimension scales, and
    ``mu_gap`` the mean absolute difference of the two means.

    Design: ``docs/experiments/raif-mountain-car/actor-entropy-diagnostic.md`` §8.
    """
    world = to_world_batch(batch)
    k_states, k_recognize, k_roll = jax.random.split(key, 3)
    states = posterior_states(system, params, world, key=k_states)
    beliefs = system.to_positional("beliefs", system.beliefs_from_states(params, states))
    n = beliefs[0].shape[0] * beliefs[0].shape[1]
    flat = tuple(jnp.reshape(x, (n, *x.shape[2:])) for x in beliefs)
    aux = crspp_recognize(
        crspp,
        params.scoring[CRSPP_ROW],
        embeddings(system, params, world.obs),
        key=k_recognize,
        is_first=batch.is_first,
    )
    est = rollout_returns(
        system,
        params,
        system.from_positional("beliefs", flat),
        key=k_roll,
        aux_beliefs={
            CRSPP_ROW: crspp.from_positional(
                "beliefs", tuple(jnp.reshape(x, (n, *x.shape[2:])) for x in aux)
            )
        },
        sample_scoring=sample_scoring,  # the rollout the critic was fit to, not another
    )
    ret, critic = jnp.mean(est.ret), jnp.mean(est.critic)

    #  the act-time path: the lane's filter over the same windows (keys derived, not
    #  split, so the columns above keep their draws)
    bounds = _action_bounds(system, params)
    initial = episode_start_beliefs(system, params)
    k_act = jax.random.fold_in(key, ACT_SAMPLE_KEY_TAG)

    def window(obs, act, first, k):
        def step(carry, xs):
            qs, kk = carry
            o, a, f = xs
            kk, k_code = jax.random.split(kk)
            qs = jax.tree_util.tree_map(lambda i_, b: jnp.where(f, i_, b), initial, qs)
            qs, _ = system.infer_states(
                params,
                system.from_positional("observations", (o, None, None)),
                qs,
                key=k_code if act_sampling else None,
            )
            feat = _amortized_features(system.to_positional("beliefs", qs))
            pushed = system.update_empirical_prior(
                params, qs, system.from_positional("actions", soft_clip(a, bounds), params)
            )
            return (pushed, kk), feat

        return jax.lax.scan(step, (initial, k), (obs, act, first))[1]

    keys = jax.random.split(k_act, batch.obs.shape[0])
    f_act = jnp.reshape(jax.vmap(window)(batch.obs, batch.action, batch.is_first, keys), (n, -1))
    f_train = jax.vmap(lambda *b: _amortized_features(b))(*flat)
    module = system._habit_module(params)

    def read(feats):
        return jax.vmap(lambda f: module.diag_moments((f,)))(feats)  # (mean, scale), (n, dim)

    mu_act, sigma_act = read(f_act)
    mu_train, sigma_train = read(f_train)
    return (
        ret,
        critic,
        ret - critic,
        jnp.mean(est.value),
        jnp.mean(mu_act),
        jnp.mean(mu_train),
        jnp.mean(sigma_act),
        jnp.mean(sigma_train),
        jnp.mean(jnp.abs(mu_act - mu_train)),
    )


# ======================================================================= the harness


def _code_provenance() -> dict[str, str | bool]:
    """``{sha, dirty}`` for the checkout this run is launched from, best-effort.

    A missing ``git`` or a tarball checkout records ``unknown`` rather than raising.
    """
    import subprocess

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ("git", "-C", str(pathlib.Path(__file__).resolve().parent), *args),
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    sha = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"sha": sha or "unknown", "dirty": bool(status) if status is not None else "unknown"}


#: What a knob's absence from a stored ``run.json`` means: what the code did before the
#: knob existed, not today's default. A knob missing from both this table and the
#: stored file makes :func:`parse` refuse the resume.
_ABSENT_MEANS: dict[str, object] = {
    "lanes": 1,  # single-stream collection
    "winit": "default",  # eqx's initializers
    "sample_scoring": False,  # an expected (mean-field) preferred rollout
    "act_sampling": False,
}


@dataclasses.dataclass
class Config:
    """The resolved run configuration — written to ``run.json`` and re-read on resume."""

    out: str
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    steps: int = DEFAULT_STEPS
    chunk_steps: int = CHUNK_STEPS
    train_every: int = TRAIN_EVERY
    lanes: int = LANES
    batch_size: int = BATCH_SIZE
    seq_len: int = BATCH_LENGTH
    capacity: int = CAPACITY
    horizon: int = IMAG_HORIZON
    time_limit: int = 100
    probe_batch: int = PROBE_BATCH
    checkpoint_every: float = 15.0  # minutes
    metrics_every: int = 1  # chunks
    small: bool = False
    seed_offset: int = 0  # the agent-init key per seed is `seed_offset + seed`
    fixtures: bool = True
    #  False = every world-model batch from M (the oracle's behaviour); True = the
    #  paper's i % 2 alternation. See `alternation`.
    alternate: bool = False
    #  the world model's and the critic's initializer: "normal" is theirs, "default"
    #  the engine's. A field so that a run's arm is recorded in `run.json`.
    winit: str = "normal"
    #  the CRSPP's preferred rollout inside `imagine`: False = expected, True = sampled
    sample_scoring: bool = False
    act_sampling: bool = True  # the act-time code is a sample (their obs_step), not the mean field

    @property
    def tiny(self) -> dict:
        return dict(SMALL) if self.small else {}

    @property
    def path(self) -> pathlib.Path:
        return pathlib.Path(self.out)


def build(cfg: Config):
    """Step 0: ``(system, crspp, env, state, buffers, probe_batch, seed_reports, horizon)``.

    Deterministic in the config alone, so ``--resume`` rebuilds the shapes by calling it and
    then overwrites the leaves. ``horizon`` is resolved once, for both the agent and
    :func:`make_chunk_fn`.
    """
    env = MountainCar(time_limit=cfg.time_limit)
    tiny = dict(cfg.tiny)
    horizon = int(tiny.pop("horizon", cfg.horizon))
    agents = [
        raif_mountain_car(
            jax.random.PRNGKey(cfg.seed_offset + s), horizon=horizon, winit=cfg.winit, **tiny
        )
        for s in cfg.seeds
    ]
    system, crspp = agents[0].system, agents[0].crspp

    buffers, reports = [], []
    for s in cfg.seeds:
        positive = StreamBuffer.empty(cfg.capacity, cfg.seq_len, 2, 1, extras=EXTRAS)
        general = StreamBuffer.empty(cfg.capacity, cfg.seq_len, 2, 1, extras=EXTRAS)
        if cfg.fixtures:
            episodes, provenance = load_fixture(fixture_path(s))
            reports.append((s, seed_stats(episodes, success=IS_SUCCESS), provenance))
            for ep in episodes:
                #  the same routing rule as collected episodes; these all succeeded
                positive, general = route(positive, general, label(ep))
        buffers.append([positive, general])

    lanes = int(cfg.lanes)
    keys = [jax.random.PRNGKey(1_000_000 + cfg.seed_offset + s) for s in cfg.seeds]
    #  one env reset and one key per lane, so the lanes do not start in lockstep
    resets = [jax.vmap(env.reset)(jax.random.split(jax.random.fold_in(k, 0), lanes)) for k in keys]
    state = _stack(
        [
            SeedState(
                params=a.params,
                learn_state=a.learn_state,
                beliefs=_lanes(episode_start_beliefs(system, a.params), lanes),
                env_state=r[1],
                incoming_reward=jnp.zeros((lanes,), jnp.float32),
                t_in_episode=jnp.zeros((lanes,), jnp.int32),
                succeeded=jnp.zeros((lanes,), jnp.bool_),
                key=jax.random.split(jax.random.fold_in(k, 1), lanes),
                train_key=jax.random.fold_in(k, 2),
            )
            for a, r, k in zip(agents, resets, keys, strict=True)
        ]
    )
    batch = _probe_batch(cfg, buffers)
    return system, crspp, env, state, buffers, batch, reports, horizon


def _lanes(tree, lanes: int):
    """``tree`` with every leaf broadcast onto a leading lane axis.

    Lanes share the learned initial belief and diverge through their own reset and key.
    """
    return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (int(lanes), *jnp.shape(x))), tree)


def _probe_batch(cfg: Config, buffers: list) -> Episode:
    """A fixed ``(seeds, probe_batch, seq_len, ...)`` batch, drawn once, from M⁺.

    Under ``--no-fixtures`` M⁺ starts empty and the probe is a batch of zeros until it fills.
    """
    return _stack(
        [
            sample_windows(buffers[i][0], jax.random.PRNGKey(7_000_000 + i), cfg.probe_batch)
            for i in range(len(cfg.seeds))
        ]
    )


def fixture_path(seed: int) -> pathlib.Path:
    """``reference/raif/fixtures/oracle_mc_seed<N>.npz`` — the oracle's own collect stream."""
    import hybrid_aif.reference.raif.fixtures as fx

    return pathlib.Path(fx.__file__).parent / f"oracle_mc_seed{int(seed)}.npz"


def _stack(xs: list):
    return jax.tree_util.tree_map(lambda *a: jnp.stack(a), *xs)


# =================================================================== checkpointing


class Counters(NamedTuple):
    """What the JSON sidecar carries: the loop's position, not its arrays."""

    step: int
    chunk: int
    grad_index: int
    episodes: tuple[int, ...]
    wall_seconds: float


def save_checkpoint(
    cfg: Config, state, buffers, pending, counters: Counters, coverage=None
) -> pathlib.Path:
    """Serialise everything numeric, then the sidecar, whose presence means the checkpoint is valid.

    Files land as ``.tmp`` and are renamed, and ``checkpoint.json`` is written last, so a
    process killed mid-checkpoint leaves the previous one intact.
    """
    directory = cfg.path / "checkpoint"
    directory.mkdir(parents=True, exist_ok=True)
    _serialise(directory / "state.eqx", state)
    _serialise(directory / "buffers.eqx", buffers)
    flat = {}
    for i, per_seed in enumerate(pending):
        for j, held in enumerate(per_seed):
            for name, value in (held or {}).items():
                flat[f"{i}:{j}:{name}"] = np.asarray(value)
    tmp = directory / "pending.tmp"
    with tmp.open("wb") as handle:  # a path would gain a second `.npz` suffix
        np.savez(handle, **flat)
    tmp.replace(directory / "pending.npz")
    record = {"counters": counters._asdict(), "config": dataclasses.asdict(cfg)}
    if coverage is not None:
        #  the visited-cell sets per seed per lane: coverage is cumulative over the
        #  lane's life, so a resume continues it
        record["coverage"] = [[sorted(c.seen) for c in per_seed] for per_seed in coverage]
    tmp = directory / "checkpoint.json.tmp"
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    tmp.replace(directory / "checkpoint.json")
    return directory


def _serialise(path: pathlib.Path, tree) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    eqx.tree_serialise_leaves(tmp, tree)
    tmp.replace(path)


#: One pending-episode slot per seed per lane: the type the host's boundary carry is.
PendingGrid = list[list[dict[str, np.ndarray] | None]]


def empty_pending(cfg: Config) -> PendingGrid:
    """``[seeds][lanes]`` of ``None`` — nothing in flight. One place builds the grid."""
    return [[None] * int(cfg.lanes) for _ in cfg.seeds]


def load_coverage(cfg: Config, env: MountainCar) -> list[list[Coverage]]:
    """``[seeds][lanes]`` of :class:`Coverage`, restored from the sidecar if it is there.

    Without a ``coverage`` key the counters start empty.
    """
    path = cfg.path / "checkpoint" / "checkpoint.json"
    saved = None
    if path.exists():
        saved = json.loads(path.read_text()).get("coverage")
    return [
        [Coverage(env, seen=None if saved is None else saved[i][j]) for j in range(int(cfg.lanes))]
        for i in range(len(cfg.seeds))
    ]


def load_checkpoint(cfg: Config, state, buffers):
    """``(state, buffers, pending, counters)`` restored onto freshly built shapes."""
    directory = cfg.path / "checkpoint"
    record = json.loads((directory / "checkpoint.json").read_text())
    state = eqx.tree_deserialise_leaves(directory / "state.eqx", state)
    buffers = eqx.tree_deserialise_leaves(directory / "buffers.eqx", buffers)
    with np.load(directory / "pending.npz") as data:
        held = empty_pending(cfg)
        for name in data.files:
            index, lane, column = name.split(":", 2)
            slot = held[int(index)][int(lane)] or {}
            slot[column] = np.asarray(data[name])
            held[int(index)][int(lane)] = slot
    counters = dict(record["counters"])
    counters["episodes"] = tuple(counters["episodes"])
    return state, buffers, held, Counters(**counters)


# ========================================================================= the loop


def _console(*args) -> None:
    """``print`` with a flush, so status lines reach a redirected log."""
    print(*args, flush=True)


def train(cfg: Config, *, resume: bool = False, log=_console) -> Counters:
    """The run. Returns the final counters; writes everything else to ``cfg.out``."""
    cfg.path.mkdir(parents=True, exist_ok=True)
    system, crspp, env, state, buffers, probe_batch, reports, horizon = build(cfg)
    pending: PendingGrid = empty_pending(cfg)
    counters = Counters(
        step=0, chunk=0, grad_index=0, episodes=(0,) * len(cfg.seeds), wall_seconds=0.0
    )
    #  the last WINDOW returns per seed, for the console line only (the record is
    #  scores.jsonl)
    recent: list[collections.deque] = [collections.deque(maxlen=ev.WINDOW) for _ in cfg.seeds]

    #  their MeasureCoverage, one per lane, cumulative over the lane's life (`Coverage`)
    coverage = (
        load_coverage(cfg, env)
        if resume
        else [[Coverage(env) for _ in range(int(cfg.lanes))] for _ in cfg.seeds]
    )

    if resume:
        state, buffers, pending, counters = load_checkpoint(cfg, state, buffers)
        for i, s in enumerate(cfg.seeds):
            recent[i].extend(
                r["return"] for r in read_scores(cfg.path / "scores.jsonl") if r["seed"] == s
            )
        log(
            f"resumed at step {counters.step:,} (chunk {counters.chunk}, "
            f"{sum(counters.episodes):,} episodes)"
        )
    else:
        #  the commit the run was launched from, and whether the tree was dirty
        record = dataclasses.asdict(cfg) | {"code": _code_provenance()}
        (cfg.path / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        for s, stats, provenance in reports:
            log(
                f"seed {s} seeded from {provenance['commit_sha'][:12]}: "
                f"{stats.n_episodes} episodes, mean return {stats.mean_return:.1f}, "
                f"SR {stats.success_rate:.2f}, mean length {stats.mean_length:.1f}"
            )

    chunk_fn = make_chunk_fn(
        system,
        crspp,
        env,
        chunk_steps=cfg.chunk_steps,
        train_every=cfg.train_every,
        lanes=cfg.lanes,
        batch_size=cfg.batch_size,
        horizon=horizon,
        sample_scoring=cfg.sample_scoring,
        act_sampling=cfg.act_sampling,
    )
    probe_fn = eqx.filter_jit(
        jax.vmap(
            lambda p, b, k: probe(
                system,
                crspp,
                p,
                b,
                k,
                sample_scoring=cfg.sample_scoring,
                act_sampling=cfg.act_sampling,
            )
        )
    )
    add_fn = eqx.filter_jit(extend)
    n_blocks = cfg.chunk_steps // cfg.train_every
    scores = (cfg.path / "scores.jsonl").open("a")
    metrics = (cfg.path / "metrics.jsonl").open("a")

    stopping = {"now": False}

    def _stop(signum, _frame):
        stopping["now"] = True
        log(f"signal {signum}: finishing the chunk, then checkpointing")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _stop)
        except ValueError:  # not the main thread (a test, or a driver process)
            pass

    started, last_checkpoint = time.time(), time.time()
    try:
        while counters.step < cfg.steps and not stopping["now"]:
            if cfg.alternate:  # the labeled A/B, not the parity default — see `alternation`
                #  their fallback test is `len(positive_replay) < args.batch_steps`
                #  (`run/train.py:160`) — windows against batch_size · batch_length
                ready = np.asarray([b[0].ready(cfg.batch_size * cfg.seq_len) for b in buffers])
                ups = alternation(counters.grad_index, n_blocks, ready)
            else:
                ups = m_only(len(cfg.seeds), n_blocks)
            t0 = time.time()
            state, stream, losses = chunk_fn(
                state, _stack([b[0] for b in buffers]), _stack([b[1] for b in buffers]), ups
            )
            jax.block_until_ready(losses.world.total)
            device_seconds = time.time() - t0

            step = counters.step + cfg.chunk_steps
            counts = list(counters.episodes)
            columns = {name: np.asarray(getattr(stream, name)) for name in Stream._fields}
            for i, s in enumerate(cfg.seeds):
                for j in range(int(cfg.lanes)):
                    #  the pooled agent step each episode ended on: `consumed` counts
                    #  steps in this lane, starting negative by the pending head's
                    #  length, and one lane-step is `lanes` pooled steps
                    held = pending[i][j]
                    consumed = -len(held["reward"]) if held is not None else 0
                    episodes, pending[i][j] = split_stream(
                        held, {k: v[i, j] for k, v in columns.items()}
                    )
                    for ep_cols in episodes:
                        ep = as_collected(ep_cols)
                        buffers[i][0], buffers[i][1] = route(
                            buffers[i][0], buffers[i][1], ep, add_fn=add_fn
                        )
                        counts[i] += 1
                        consumed += int(ep_cols["reward"].shape[0])
                        total = float(np.sum(ep_cols["reward"]))
                        recent[i].append(total)
                        scores.write(
                            json.dumps(
                                {
                                    "step": counters.step + consumed * int(cfg.lanes),
                                    "seed": int(s),
                                    "lane": j,
                                    "return": total,
                                    "length": int(ep_cols["reward"].shape[0]),
                                    "success": bool(ep_cols["has_succeeded"][-1]),
                                    #  R-S's accumulator, which drops the is_last
                                    #  row's reward (`ev.stability_c`)
                                    "stability": ev.stability_c(ep_cols["reward"], ev.MC_OFFSET),
                                    #  their `stats/mean_log_entropy`: the act-time
                                    #  policy entropy averaged over the episode
                                    "log_entropy": float(np.mean(ep_cols["log_entropy"])),
                                    #  their `episode/coverage_mean`, read off this
                                    #  lane's cumulative counter at the episode's end
                                    "coverage": coverage[i][j].update(ep_cols["obs"]),
                                }
                            )
                            + "\n"
                        )
            scores.flush()
            counters = Counters(
                step=step,
                chunk=counters.chunk + 1,
                grad_index=counters.grad_index + n_blocks,
                episodes=tuple(counts),
                wall_seconds=counters.wall_seconds + (time.time() - t0),
            )

            if counters.chunk % max(1, cfg.metrics_every) == 0:
                gaps = probe_fn(
                    state.params,
                    probe_batch,
                    jax.random.split(jax.random.PRNGKey(counters.chunk), len(cfg.seeds)),
                )
                for i, s in enumerate(cfg.seeds):
                    #  G's decomposition, keyed by term name
                    per_term = {
                        f"term.{name}": float(value[i, -1])
                        for name, value in losses.imagine.per_term.items()
                    }
                    metrics.write(
                        json.dumps(
                            {
                                "step": step,
                                "seed": int(s),
                                **per_term,
                                "world": float(losses.world.total[i, -1]),
                                "decoder": float(losses.world.decoder[i, -1]),
                                "dyn": float(losses.world.dyn[i, -1]),
                                "rep": float(losses.world.rep[i, -1]),
                                "actor": float(losses.imagine.actor_loss[i, -1]),
                                "critic": float(losses.imagine.critic_loss[i, -1]),
                                "ensemble": float(losses.ensemble.total[i, -1]),
                                "crspp": float(losses.crspp.total[i, -1]),
                                "crspp_positive": float(losses.crspp.positive[i, -1]),
                                "ret": float(losses.imagine.ret[i, -1]),
                                "critic_value": float(losses.imagine.critic_value[i, -1]),
                                "bc_logprob": float(losses.imagine.bc_logprob[i, -1]),
                                #  their `train/actor_refresh_mean`, a term of the
                                #  actor loss; `bc_logprob` is the unmasked mean
                                #  and not comparable (`series-sweep.md` §6.1)
                                "bc_term": float(losses.imagine.bc_term[i, -1]),
                                #  the actor channel: `policy_entropy` and
                                #  `bc_weight` show a collapse to the σ floor and
                                #  the imitation weight left in the batch;
                                #  `advantage` against `retnorm_low/high` shows
                                #  the return normaliser's band
                                "advantage": float(losses.imagine.advantage[i, -1]),
                                "policy_entropy": float(losses.imagine.policy_entropy[i, -1]),
                                "prior_entropy": float(losses.imagine.prior_entropy[i, -1]),
                                "bc_weight": float(losses.imagine.bc_weight[i, -1]),
                                "retnorm_low": float(losses.imagine.retnorm_low[i, -1]),
                                "retnorm_high": float(losses.imagine.retnorm_high[i, -1]),
                                "value": float(losses.imagine.value[i, -1]),
                                #  how much of the imagined horizon the continuation
                                #  head is truncating — the direct read on `start_cont`
                                "rollout_weight": float(losses.imagine.rollout_weight_mean[i, -1]),
                                "probe_ret": float(gaps[0][i]),
                                "probe_critic": float(gaps[1][i]),
                                "distill_gap": float(gaps[2][i]),
                                "probe_value": float(gaps[3][i]),
                                #  the actor on the act-time and the training feature
                                #  paths, same states: a widening gap is the actor
                                #  acting off the distribution it trains on
                                "probe_mu_act": float(gaps[4][i]),
                                "probe_mu_train": float(gaps[5][i]),
                                "probe_sigma_act": float(gaps[6][i]),
                                "probe_sigma_train": float(gaps[7][i]),
                                "probe_mu_gap": float(gaps[8][i]),
                            }
                        )
                        + "\n"
                    )
                metrics.flush()

            log(
                f"[{counters.step:>9,}/{cfg.steps:,}] {device_seconds:5.1f}s/chunk "
                f"({cfg.chunk_steps / device_seconds:6.1f} steps/s) "
                f"eps={sum(counters.episodes):>6,} "
                f"return={'/'.join(f'{np.mean(r):.0f}' if r else '—' for r in recent)} "
                f"world={float(jnp.mean(losses.world.total[:, -1])):7.3f} "
                f"finite={bool(jnp.all(jnp.isfinite(losses.world.total)))}"
            )

            if (time.time() - last_checkpoint) / 60.0 >= cfg.checkpoint_every:
                save_checkpoint(cfg, state, buffers, pending, counters, coverage)
                last_checkpoint = time.time()
                log(f"checkpoint at step {counters.step:,}")
    finally:
        scores.close()
        metrics.close()
        #  never mask the exception that got us here: a failure to write the
        #  shutdown checkpoint is logged, not raised over the real cause
        try:
            save_checkpoint(cfg, state, buffers, pending, counters, coverage)
            written = "checkpoint written"
        except Exception as exc:  # noqa: BLE001
            written = f"CHECKPOINT FAILED: {exc!r}"
        log(
            f"stopped at step {counters.step:,} after {(time.time() - started) / 3600:.2f} h; "
            f"{written}"
        )
    return counters


# ============================================================================ eval


def read_scores(path: pathlib.Path) -> list[dict]:
    if not pathlib.Path(path).exists():
        return []
    with pathlib.Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


class SeedReport(NamedTuple):
    """One seed's §7 row. Every field is a *window*, not an endpoint — §7's ruling."""

    seed: int
    episodes: int
    steps: int
    best_window: float
    best_window_at: str
    final_acr: float
    last_500k: float
    sr_window: float
    sr_oracle: float
    rs: float


def seed_report(
    rows: list[dict],
    seed: int,
    *,
    window: int = ev.WINDOW,
    span: int = 200_000,
    tail: int = 500_000,
    offset: float = ev.MC_OFFSET,
    threshold: float = -100.0,
) -> SeedReport:
    """The §7-comparable numbers for one seed, from ``scores.jsonl`` rows.

    ``best_window``   the best mean over complete tiled ``span``-step windows (the log reaching
                      within 1% of a window's end); a partial trailing window is not counted
    ``final_acr``     §6's ACR: the mean over the last ``window`` episodes (an endpoint read)
    ``last_500k``     the mean over episodes in the final ``tail`` steps
    ``sr_window``     §6's SR: the fraction of the last ``window`` episodes reaching the goal
    ``sr_oracle``     the oracle table's SR: the fraction of the last ``tail`` steps' episodes
                      returning above ``threshold``, a different statistic from §6's
    ``rs``            §6's R-S at the Mountain Car offset, over each row's ``C``
                      (:func:`_stability`)

    Design: ``docs/design/amortization-p0-addendum.md`` §7.
    """
    mine = sorted((r for r in rows if r["seed"] == seed), key=lambda r: r["step"])
    if not mine:
        raise ValueError(f"no episodes logged for seed {seed}")
    returns = np.asarray([r["return"] for r in mine], np.float64)
    steps = np.asarray([r["step"] for r in mine], np.int64)
    last = int(steps[-1])

    best, best_at = -np.inf, "—"
    #  a window counts once the log reaches it to within 1% of its end: the run's last
    #  episode can end a few dozen steps short of a window that the run's budget covers
    for low in range(0, last + span // 100 - span + 1, span):
        inside = (steps >= low) & (steps < low + span)
        if inside.sum() >= 2 and returns[inside].mean() > best:
            best, best_at = float(returns[inside].mean()), f"{low // 1000}–{(low + span) // 1000}k"
    if not np.isfinite(best):  # a run shorter than one window: the whole run IS the window
        best, best_at = float(returns.mean()), f"0–{last // 1000}k (partial)"
    in_tail = steps > last - tail
    return SeedReport(
        seed=seed,
        episodes=len(mine),
        steps=last,
        best_window=best,
        best_window_at=best_at,
        final_acr=ev.acr(returns[-window:]),
        last_500k=float(returns[in_tail].mean()),
        sr_window=ev.sr(np.asarray([r["success"] for r in mine[-window:]])),
        sr_oracle=float(np.mean(returns[in_tail] > threshold)),
        rs=ev.relative_stability(_stability(mine, offset), offset=offset, window=window),
    )


def _stability(rows: list[dict], offset: float) -> np.ndarray:
    """R-S's ``C`` per episode: the logged ``stability`` field, or the fallback for older logs.

    A log without the field falls back to ``return + offset·length``, which over-counts by the
    final row's reward. A log mixing both kinds raises ``ValueError``.
    """
    old = [r for r in rows if "stability" not in r]
    new = [r for r in rows if "stability" in r]
    if old and new:
        raise ValueError(
            f"scores.jsonl mixes {len(old)} episode row(s) with no 'stability' "
            f"field and {len(new)} with one: R-S's C is accumulated differently in the two "
            "(their MeasureRStability drops the is_last row's reward), so folding them into "
            "one history would report a number that is neither. Re-run, or report the halves "
            "separately."
        )
    if new:
        return np.asarray([r["stability"] for r in new], np.float64)
    return np.asarray([r["return"] + offset * r["length"] for r in old], np.float64)


def report(cfg: Config, log=_console) -> list[SeedReport]:
    """Print the §7-comparable table from ``scores.jsonl``."""
    rows = read_scores(cfg.path / "scores.jsonl")
    if not rows:
        log(f"no episodes in {cfg.path / 'scores.jsonl'}")
        return []
    present = sorted({r["seed"] for r in rows})
    out = [seed_report(rows, s) for s in present]
    log("")
    log("R-AIF state-based Mountain Car — addendum §7 comparison (window-honest)")
    log(
        f"{'seed':>4} {'steps':>10} {'eps':>7} {'best 200k':>11} {'at':>12} "
        f"{'final-100':>10} {'last-500k':>10} {'SR(§6)':>7} {'SR(>-100)':>10} {'R-S':>6}"
    )
    for r in out:
        log(
            f"{r.seed:>4} {r.steps:>10,} {r.episodes:>7,} {r.best_window:>11.1f} "
            f"{r.best_window_at:>12} {r.final_acr:>10.1f} {r.last_500k:>10.1f} "
            f"{r.sr_window:>7.2f} {r.sr_oracle:>10.2f} {r.rs:>6.3f}"
        )
    log("")
    log(
        "oracle (8 seeds): best-200k −68.0…−69.5 / final-100 −67.6…−110.0 / "
        "last-500k −69.5…−79.4 / SR(>-100) 0.86…0.98"
    )
    log(
        "target (§7):   converged ACR −68 to −69.5 on the WINDOWS, plus the oracle's shape: "
        "first successes by ~2–4k, a plateau near −80 over 50–200k, a failure phase at "
        "~225–450k recovered by 375–450k, converged after"
    )
    if out:
        log(
            f"mean best-200k over {len(out)} seed(s): "
            f"{np.mean([r.best_window for r in out]):.1f} ± "
            f"{np.std([r.best_window for r in out]):.1f}"
        )
    return out


# ======================================================================= benchmark


def benchmark(cfg: Config, *, reps: int = 3, log=_console) -> dict:
    """Jitted steps/s for the full train cycle, one seed against ``len(seeds)`` vmapped.

    Measured at the run's shapes with the run's chunk function. Reports both rates, their
    ratio, the projected wall clock for ``cfg.steps`` × seeds, and the collection half's own
    time (``collect_seconds``) from the same :func:`make_collect_fn`.
    """
    out = {}
    for n in (1, len(cfg.seeds)):
        sub = dataclasses.replace(cfg, seeds=tuple(cfg.seeds[:n]))
        system, crspp, env, state, buffers, _, _, horizon = build(sub)
        chunk_fn = make_chunk_fn(
            system,
            crspp,
            env,
            chunk_steps=sub.chunk_steps,
            train_every=sub.train_every,
            lanes=sub.lanes,
            batch_size=sub.batch_size,
            horizon=horizon,
            sample_scoring=sub.sample_scoring,
            act_sampling=sub.act_sampling,
        )
        n_blocks = sub.chunk_steps // sub.train_every
        ups = m_only(n, n_blocks)  # the parity default's schedule; timing is identical either way
        args = (_stack([b[0] for b in buffers]), _stack([b[1] for b in buffers]), ups)
        t0 = time.time()
        state, _, losses = chunk_fn(state, *args)
        jax.block_until_ready(losses.world.total)
        compile_seconds = time.time() - t0
        t0 = time.time()
        for _ in range(reps):
            state, _, losses = chunk_fn(state, *args)
        jax.block_until_ready(losses.world.total)
        seconds = (time.time() - t0) / reps

        #  the collection half alone, at the same shapes and the same POOLED step count
        #  (`chunk_steps // lanes` steps in each of `lanes` lanes)
        only_collect = eqx.filter_jit(
            jax.vmap(
                make_collect_fn(
                    system, env, sub.chunk_steps // int(sub.lanes), act_sampling=sub.act_sampling
                )
            )
        )
        carry = state.collect_carry()
        jax.block_until_ready(only_collect(state.params, carry)[1].reward)
        t0 = time.time()
        for _ in range(reps):
            jax.block_until_ready(only_collect(state.params, carry)[1].reward)
        collect_seconds = (time.time() - t0) / reps

        out[n] = {
            "seeds": n,
            "compile_seconds": compile_seconds,
            "chunk_seconds": seconds,
            "collect_seconds": collect_seconds,
            "collect_fraction": collect_seconds / seconds,
            "agent_steps_per_second": sub.chunk_steps / seconds,
            "grad_steps_per_second": n_blocks / seconds,
            "total_agent_steps_per_second": n * sub.chunk_steps / seconds,
        }
        log(
            f"{n} seed(s): {seconds:7.2f} s/chunk of {sub.chunk_steps} steps "
            f"= {sub.chunk_steps / seconds:7.2f} agent steps/s per seed, "
            f"{n * sub.chunk_steps / seconds:7.2f} aggregate; "
            f"{seconds / n_blocks:6.3f} s/gradient step; "
            f"collect {collect_seconds:.2f}s = {100 * collect_seconds / seconds:.1f}% "
            f"(compile {compile_seconds:.1f}s)"
        )
    one, many = out[1], out[len(cfg.seeds)]
    ratio = many["total_agent_steps_per_second"] / one["total_agent_steps_per_second"]
    hours = cfg.steps / many["agent_steps_per_second"] / 3600.0
    log(f"vmap({len(cfg.seeds)}) aggregate throughput ratio vs 1 seed: {ratio:.2f}×")
    sequential = cfg.steps * len(cfg.seeds) / one["agent_steps_per_second"] / 3600.0
    log(f"projected wall clock for {cfg.steps:,} steps × {len(cfg.seeds)} seeds: {hours:.1f} h")
    log(f"the same run sequentially: {sequential:.1f} h")
    out["sequential_hours"] = sequential
    out["ratio"] = ratio
    out["projected_hours"] = hours
    return out


# ============================================================================ CLI


def parse(argv: list[str] | None = None) -> tuple[Config, argparse.Namespace]:
    p = argparse.ArgumentParser(prog="python -m hybrid_aif.reference.raif.run", description=__doc__)
    p.add_argument("--out", required=True, help="run directory")
    p.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help=f"agent steps per seed (default {DEFAULT_STEPS:,}); on --resume it EXTENDS the run",
    )
    p.add_argument("--chunk-steps", type=int, default=CHUNK_STEPS)
    p.add_argument(
        "--lanes",
        type=int,
        default=LANES,
        help=(
            f"parallel env lanes per seed (default {LANES} = their envs.amount); must divide "
            "--train-every, since their train_ratio counts POOLED agent steps. 1 is a "
            "single-stream harness."
        ),
    )
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--seq-len", type=int, default=BATCH_LENGTH)
    p.add_argument(
        "--capacity", type=int, default=CAPACITY, help="replay steps per buffer, not sequences"
    )
    p.add_argument("--horizon", type=int, default=IMAG_HORIZON)
    p.add_argument("--checkpoint-every", type=float, default=15.0, help="minutes")
    p.add_argument("--metrics-every", type=int, default=1, help="chunks")
    p.add_argument("--small", action="store_true", help="test-scale sizes; NOT parity")
    p.add_argument("--no-fixtures", action="store_true", help="start with empty buffers")
    p.add_argument(
        "--alternate",
        action="store_true",
        help=(
            "alternate world-model batches M/M+ per the PAPER — a labeled A/B, not parity: "
            "the oracle's own alternation is inert and it trains every batch on M"
        ),
    )
    p.add_argument(
        "--sample-scoring",
        action="store_true",
        help=(
            "roll the CRSPP's preferred trajectory SAMPLED inside imagine (theirs, ledger 6.5) "
            "instead of expected/mean-field (the default)"
        ),
    )
    p.add_argument(
        "--no-act-sampling",
        action="store_true",
        help=("act on the mean-field posterior instead of a sampled code; kept for the ablation"),
    )
    p.add_argument(
        "--winit",
        choices=("normal", "default"),
        default="normal",
        help=(
            "the world model's and the critic's initializer. 'normal' is theirs "
            "and the parity default; 'default' is the engine's, the isolation control's arm"
        ),
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    args = p.parse_args(argv)
    cfg = Config(
        out=args.out,
        seeds=tuple(int(s) for s in args.seeds.split(",") if s.strip() != ""),
        steps=DEFAULT_STEPS if args.steps is None else args.steps,
        chunk_steps=args.chunk_steps,
        lanes=args.lanes,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        capacity=args.capacity,
        horizon=args.horizon,
        checkpoint_every=args.checkpoint_every,
        metrics_every=args.metrics_every,
        small=args.small,
        fixtures=not args.no_fixtures,
        alternate=args.alternate,
        winit=args.winit,
        sample_scoring=args.sample_scoring,
        act_sampling=not args.no_act_sampling,
    )
    if args.resume and (cfg.path / "run.json").exists():
        #  the stored configuration wins: every knob fixes a shape, except `--steps` and
        #  the two reporting cadences, which the checkpoint does not hold
        stored = json.loads((cfg.path / "run.json").read_text())
        stored["seeds"] = tuple(stored["seeds"])
        #  `code` is provenance, not configuration; a resume from another commit is
        #  reported, not refused
        launched_from = stored.pop("code", None)
        #  a knob absent from a stored run.json takes the value it had before the knob
        #  existed (`_ABSENT_MEANS`), never today's default; any other absence refuses
        for knob, before_it_existed in _ABSENT_MEANS.items():
            stored.setdefault(knob, before_it_existed)
        missing = sorted({f.name for f in dataclasses.fields(Config)} - set(stored) - {"out"})
        if missing:
            raise ValueError(
                f"{cfg.path / 'run.json'} records no {missing}, so this resume cannot know what "
                "the checkpoint was built with. Defaulting them would rebuild whatever today's "
                "code builds, against a checkpoint from whatever it built then. Add the value the "
                "run was actually launched with to its run.json (and to reference.raif.run's "
                "_ABSENT_MEANS if the knob postdates runs that are still resumable)."
            )
        #  `out` is where this invocation writes, not the stored run's old path
        stored["out"] = args.out
        if args.steps is not None:
            stored["steps"] = args.steps
        stored["checkpoint_every"] = args.checkpoint_every
        stored["metrics_every"] = args.metrics_every
        cfg = Config(**stored)
        now = _code_provenance()
        if launched_from and launched_from.get("sha") != now["sha"]:
            print(
                f"note: resuming a run launched from {str(launched_from.get('sha'))[:12]} "
                f"on a checkout at {str(now['sha'])[:12]}",
                file=sys.stderr,
            )
    return cfg, args


def main(argv: list[str] | None = None) -> int:
    cfg, args = parse(argv)
    if args.report:
        report(cfg)
    elif args.benchmark:
        benchmark(cfg)
    else:
        train(cfg, resume=args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
