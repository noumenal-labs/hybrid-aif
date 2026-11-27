"""The eight world models whose forward samples are pinned bit for bit.

Each case builds a System and its parameters, with the options its forward run is made under;
:func:`record` runs the environment ``as_env`` returns for 50 ticks (reset, then 50 steps) and
returns every observation, state and realised action as arrays. ``tests/data/as_env_<case>.npz``
holds that record as the environment produced it before the mirror existed, and
``tests/test_mirror.py`` checks that the mirror-based ``as_env`` reproduces it exactly.
"""

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_aif import (
    Categorical,
    ConditionalCategorical,
    ConditionalGaussian,
    Gaussian,
    LinearGaussianObservation,
    LinearGaussianTransition,
    MixtureGaussian,
    System,
    SystemParams,
)
from hybrid_aif.controls import LEVEL
from hybrid_aif.distributions import MLPConditional
from tests.test_action_node_modalities import maze_with_cue_by_node
from tests.test_action_nodes import pusher
from tests.test_d_deps import hierarchical_tmaze
from tests.test_learn import _sense_system
from tests.test_system import tmaze_system

TICKS = 50
KEY = jax.random.PRNGKey(11)


class Case(NamedTuple):
    system: System
    params: SystemParams
    param_key: object = None
    states: tuple[str, ...] | None = None


def _switching_with_a_categorical_reading() -> tuple[System, SystemParams]:
    """A mode (categorical) gating a linear-Gaussian reading of ``x`` with per-mode noise, and a
    categorical reading of the mode."""
    system = System(
        A_deps=((1, 0), (0,)),
        B_deps=((), ()),
        controlled=(False, False),
        policy_len=1,
        mixture_k=6,
        mixture_schedule="runnalls",
        modality_names=("y", "z"),
        state_names=("mode", "x"),
    )
    x_innov = ConditionalGaussian(
        loc=jnp.zeros((1, 1)), chol=jnp.asarray([[[0.05]]]), parent_ndims=1
    )
    noise = jnp.asarray([[[0.3]], [[0.9]]])
    params = SystemParams(
        likelihood={
            "y": LinearGaussianObservation(H=jnp.asarray([[1.0]]), c=jnp.zeros(1), noise=noise),
            "z": ConditionalCategorical(log_table=jnp.log(0.9 * jnp.eye(2) + 0.05), parent_ndims=1),
        },
        transition={
            "mode": ConditionalCategorical(
                log_table=jnp.log(jnp.asarray([[0.5], [0.5]])), parent_ndims=1
            ),
            "x": LinearGaussianTransition(F=jnp.asarray([[1.0]]), innovations=x_innov),
        },
        initial_state_prior={
            "mode": Categorical(logits=jnp.zeros(2)),
            "x": MixtureGaussian(
                log_weights=jnp.log(jnp.asarray([0.5, 0.3, 0.2])),
                means=jnp.asarray([[-1.0], [0.5], [1.5]]),
                chols=jnp.asarray([[[0.4]], [[0.3]], [[0.5]]]),
            ),
        },
    )
    return system, params


def _mlp_likelihood() -> tuple[System, SystemParams]:
    """A continuous factor with a neural transition and a neural reading."""
    dim, levels = 2, 3
    system = System(
        A_deps=((0,),),
        B_deps=((),),
        controlled=(True,),
        policy_len=1,
        modality_names=("y",),
        state_names=("x",),
        epistemic=False,
    )
    kw = dict(width=8, depth=1, dtype=jnp.float32)  # the recording's dtype, in either mode
    params = SystemParams(
        likelihood={"y": MLPConditional.from_shapes((dim,), 1, key=jax.random.PRNGKey(7), **kw)},
        transition={
            "x": MLPConditional.from_shapes((dim, levels), dim, key=jax.random.PRNGKey(8), **kw)
        },
        initial_state_prior={"x": Gaussian(loc=jnp.zeros(dim), chol=jnp.eye(dim))},
    )
    return system, params


def _non_identity_actuator() -> tuple[System, SystemParams]:
    system, params = pusher(feel=jnp.eye(3))
    kernel = jnp.asarray([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
    return system, dataclasses.replace(params, actuator={"push": kernel})


def cases() -> dict[str, Case]:
    """The eight cases, by name."""
    tmaze = tmaze_system()
    look = maze_with_cue_by_node()
    hier = hierarchical_tmaze()
    sense = _sense_system()
    push = pusher()
    return {
        "tmaze": Case(*tmaze),
        "look": Case(*look),
        "actuator": Case(*_non_identity_actuator()),
        "d_deps": Case(*hier),
        "param_key": Case(sense[0], sense[1], param_key=jax.random.PRNGKey(3)),
        "switching": Case(*_switching_with_a_categorical_reading()),
        "mlp": Case(*_mlp_likelihood()),
        "masked": Case(*push, states=()),
    }


def action_for(env, t: int) -> dict:
    """A fixed intended action at tick ``t``: the levels cycle, a vector holds a small value."""
    out = {}
    for k, name in enumerate(env.names.actions):
        size = env.names.sizes[k] if env.names.sizes else 2
        if env.names.kind(k) == LEVEL:
            out[name] = jnp.asarray(t % size, jnp.int32)
        else:
            out[name] = jnp.full((size,), 0.1 * (t % 5))
    return out


def record(env, env_params, states=None, ticks: int = TICKS) -> dict[str, np.ndarray]:
    """Reset and step ``env`` for ``ticks`` rounds under :func:`action_for`; every observation,
    state and realised action as one array per name per tick (``obs/<name>`` and so on, stacked
    over time)."""
    keys = jax.random.split(KEY, ticks + 1)
    obs, state = env.reset(keys[0], env_params)
    seq = {"obs": [obs], "state": [state], "action": []}
    for t in range(ticks):
        act = action_for(env, t)
        kw = {} if states is None else {"states": states}
        obs, state, realised = env.step_actuated(keys[t + 1], state, act, env_params, **kw)
        seq["obs"].append(obs)
        seq["state"].append(state)
        seq["action"].append(realised)
    out = {}
    for kind, steps in seq.items():
        for name in steps[0]:
            out[f"{kind}/{name}"] = np.stack([np.asarray(s[name]) for s in steps])
    return out
