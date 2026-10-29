"""Training loops: pure functions over (System, params, data, optimizer carry).

Each trainer is a function over explicit state (jit-able, resumable from its own
return value), stepping its own ``LearnState`` addresses and threading the rest:

=================================================  ===============================
:func:`~hybrid_aif.train.world.fit_world`          a recurrent generative model,
                                                   from replay sequences
:func:`~hybrid_aif.train.imagine.imagine`          a policy module and a value
                                                   head, on imagined rollouts
:func:`~hybrid_aif.train.ensemble.fit_ensemble`    an ensemble of predictive heads
                                                   (what a disagreement row scores)
:mod:`~hybrid_aif.train.buffers`                    the ``Episode`` format and the
                                                   three buffers
=================================================  ===============================

Trainers pinned to an external target live in :mod:`hybrid_aif.reference`, which
imports this package, never the reverse. ``from hybrid_aif.train import imagine``
binds the function, not the module of the same name.

Design: docs/design/amortization.md A6
"""

from hybrid_aif.train.buffers import (
    EPISODE_FIELDS,
    ColumnSpec,
    Episode,
    ReplayBuffer,
    Sampler,
    StaticBuffer,
    StreamBuffer,
    add,
    as_episode,
    episode,
    episode_columns,
    extend,
    fit_length,
    sample,
    sample_windows,
    stack_episodes,
    uniform_sampler,
)
from hybrid_aif.train.ensemble import (
    EnsembleBatch,
    EnsembleLosses,
    ensemble_loss,
    fit_ensemble,
    transitions,
)
from hybrid_aif.train.imagine import (
    ImagineMetrics,
    ReturnEstimates,
    Rollout,
    imagine,
    init_retnorm,
    lambda_return,
    perc_ema,
    roll_batch,
    rollout_returns,
    term_value,
)
from hybrid_aif.train.world import (
    WorldBatch,
    WorldLosses,
    assemble,
    fit_world,
    posterior_states,
    world_loss,
)

__all__ = [
    "EPISODE_FIELDS",
    "ColumnSpec",
    "EnsembleBatch",
    "EnsembleLosses",
    "Episode",
    "ImagineMetrics",
    "ReturnEstimates",
    "Rollout",
    "ReplayBuffer",
    "Sampler",
    "StaticBuffer",
    "StreamBuffer",
    "WorldBatch",
    "WorldLosses",
    "add",
    "as_episode",
    "assemble",
    "ensemble_loss",
    "episode",
    "episode_columns",
    "extend",
    "fit_ensemble",
    "fit_length",
    "fit_world",
    "imagine",
    "init_retnorm",
    "lambda_return",
    "perc_ema",
    "posterior_states",
    "roll_batch",
    "rollout_returns",
    "sample",
    "sample_windows",
    "stack_episodes",
    "term_value",
    "transitions",
    "uniform_sampler",
    "world_loss",
]
