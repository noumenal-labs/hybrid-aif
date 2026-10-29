"""Pure-functional environment interface (pymdp's ``Env``, with value semantics).

An :class:`Env` is an ``equinox.Module`` carrying only static structure; every
array-valued parameter travels in a separate ``params`` pytree passed to each
call, so one env instance vmaps across a batch of per-agent parameter sets.

Conventions (chosen to compose with :func:`hybrid_aif.inference.discrete.fpi`):

- observations: tuple of scalar ``int32`` indices, one per modality — the
  list-of-index-arrays form ``fpi`` consumes directly;
- state: tuple of scalar ``int32`` indices, one per factor;
- action: ``int32`` array of shape ``(num_factors,)``, one action level per
  factor (uncontrolled factors expose a single no-op action).

``reset`` and ``step`` are pure and unjitted; callers jit/vmap/scan them.
:meth:`System.as_env <hybrid_aif.system.System.as_env>` wraps its process in a
:class:`~hybrid_aif.envs.GenerativeEnv`, which speaks name-keyed mappings instead.
"""

import abc
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jax import Array

Obs = tuple[Array, ...]
State = tuple[Array, ...]


class Env(eqx.Module):
    """Abstract functional environment. See module docstring for conventions."""

    @property
    @abc.abstractmethod
    def num_factors(self) -> int:
        """Number of state factors."""

    def null_action(self) -> Any:
        """The no-op action, with the structure and shape every action this env takes.

        Default: the level-index action, ``(num_factors,)`` int32 zeros.
        """
        return jnp.zeros((self.num_factors,), jnp.int32)

    @abc.abstractmethod
    def reset(self, key: Array, params: Any) -> tuple[Obs, State]:
        """Sample an initial state and an observation of it."""

    @abc.abstractmethod
    def step(self, key: Array, state: State, action: Array, params: Any) -> tuple[Obs, State]:
        """Sample the next state given ``action``, and an observation of it."""
