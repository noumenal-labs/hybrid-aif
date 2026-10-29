"""The public environment: a generative process behind name-keyed faces.

:class:`GenerativeEnv` takes and returns observations ``{modality: value}``, states
``{factor: value}`` and actions ``{action: value}``, keyed by action node
(:mod:`hybrid_aif.system.keyed`), and its :class:`GenerativeEnvParams` key their containers
by name. The environment is the actuator: an action node with a kernel ``P(a | u)``
(``actuator``) has the action it is given realised as a draw from that kernel, and the world
moves on the realised action. Underneath, a positional process samples in node order:
:meth:`System.as_env <hybrid_aif.system.System.as_env>` builds one from the System's own
declarations (the discrete or the hybrid process, chosen by the declarations), and
:meth:`GenerativeEnv.wrap` adopts an outside positional (pymdp-style) environment given the
System whose names it speaks.
"""

from collections.abc import Iterable, Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.controls import check_kernel
from hybrid_aif.envs._process import DiscreteProcess
from hybrid_aif.envs.base import Env
from hybrid_aif.system import keyed

__all__ = ["GenerativeEnv", "GenerativeEnvParams", "broadcast_params", "make"]

#: The tag the actuator folds into a step's key ("actu"): no index the processes split a key
#: into reaches it, so the realised action is drawn independently of the world's step.
ACTUATOR_TAG = 0x61637475

KeyedObs = dict[str, Array]
KeyedState = dict[str, Array]


class GenerativeEnvParams(eqx.Module):
    """An environment's parameters: one entry per modality (``likelihood``) and per factor
    (``transition``, ``initial_state_prior``), and the actuator's kernels (``actuator``).

    Public faces key the containers by name (``params.likelihood["cue"]``); the processes read
    them as tuples in node order. A discrete process holds probability tables, a hybrid one the
    System's declaration objects. ``actuator`` maps an action node's name to its kernel
    ``P(a | u)``, a ``(levels, levels)`` probability table with one column per intended level;
    an action node without an entry realises the intended action exactly.
    """

    likelihood: Any  # AIF A
    transition: Any  # AIF B
    initial_state_prior: Any  # AIF D
    actuator: Any = eqx.field(default_factory=dict)


def make(
    A: tuple[Array, ...],
    B: tuple[Array, ...],
    D: tuple[Array, ...],
    A_deps: tuple[tuple[int, ...], ...] | None = None,
    B_deps: tuple[tuple[int, ...], ...] | None = None,
) -> "tuple[DiscreteProcess, GenerativeEnvParams]":
    """Build a positional (pymdp-style) process + params (tuple containers) from tensors;
    defaults: A on all factors, B on own factor. Wrap it with :meth:`GenerativeEnv.wrap` given a
    System, converting the params with ``system.from_positional("params", ...)``.

    Multi-modality/multi-factor collections are canonically tuples; any
    sequence is accepted at runtime.
    """
    if A_deps is None:
        A_deps = tuple(tuple(range(len(B))) for _ in A)
    if B_deps is None:
        B_deps = tuple((f,) for f in range(len(B)))
    env = DiscreteProcess(
        A_deps=tuple(tuple(d) for d in A_deps),
        B_deps=tuple(tuple(d) for d in B_deps),
    )

    def as_f32(ts):
        return tuple(jnp.asarray(t, jnp.float32) for t in ts)

    return env, GenerativeEnvParams(
        likelihood=as_f32(A), transition=as_f32(B), initial_state_prior=as_f32(D)
    )


def broadcast_params(params: Any, batch_size: int) -> Any:
    """Broadcast every leaf to a leading ``(batch_size,)`` axis (shared parameters)."""
    return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (batch_size, *x.shape)), params)


class GenerativeEnv(Env):
    """A positional process ``inner`` with name-keyed faces; ``names`` is static.

    ``params`` given as :class:`GenerativeEnvParams` must key their containers by name (as
    :meth:`System.as_env` returns them; a tuple container raises, naming
    ``system.from_positional("params", ...)``); any other params (an outside environment's own)
    pass to ``inner`` untouched.
    """

    inner: Env
    names: keyed.FaceNames = eqx.field(static=True)
    factor_actions: bool = eqx.field(static=True, default=False)
    #   whether ``inner`` takes the action one level per state factor (an outside, pymdp-style
    #   environment) rather than one per control node (the System's own process)

    @classmethod
    def wrap(cls, env: Env, system, params: Any = None) -> "GenerativeEnv":
        """An outside positional environment, keyed by ``system``'s names.

        ``env`` speaks the positional convention (:mod:`hybrid_aif.envs.base`): observations and
        states as tuples, the action as one level per factor (or the flat continuous vector). Its
        factor and modality counts are checked against the System's; their *order* must match the
        System's node order too, which cannot be checked. ``params`` (the System's, keyed) are
        needed when a control node is continuous: its box lives in ``params.control``. Wrapping an
        already keyed env raises.
        """
        if isinstance(env, GenerativeEnv):
            raise TypeError(
                "GenerativeEnv.wrap takes a positional environment, and this one is already a "
                "GenerativeEnv (keyed); use it as it is, or wrap its .inner"
            )
        names = system._face_names(params)
        if env.num_factors != len(names.factors):
            raise ValueError(
                f"GenerativeEnv.wrap: the environment has {env.num_factors} factor(s), the System "
                f"{len(names.factors)} ({list(names.factors)})"
            )
        num_modalities = getattr(env, "num_modalities", None)
        if num_modalities is not None and num_modalities != len(names.modalities):
            raise ValueError(
                f"GenerativeEnv.wrap: the environment has {num_modalities} modality/ies, the "
                f"System {len(names.modalities)} ({list(names.modalities)})"
            )
        return cls(env, names, factor_actions=True)

    @property
    def num_factors(self) -> int:
        return self.inner.num_factors

    @property
    def num_modalities(self) -> int:
        return self.inner.num_modalities

    def null_action(self) -> dict[str, Array]:
        """The process's no-op action, keyed by action node (levels or blocks)."""
        if self.factor_actions:
            return keyed.factor_actions_out(self.names, self.inner.null_action())
        return keyed.actions_out(self.names, self.inner.null_action())

    def _params(self, params: Any) -> Any:
        """Keyed :class:`GenerativeEnvParams` as the positional view the process reads; tuple
        containers raise, naming the converter. An outside env's own params pass through."""
        if isinstance(params, GenerativeEnvParams):
            return keyed.params_in(self.names, params, "GenerativeEnv")
        return params

    def _state_in(self, state: Mapping[str, Array], face: str) -> tuple:
        keyed.check_mapping(self.names, "beliefs", state, face, label="state")
        return tuple(state[n] for n in self.names.factors)

    def _out(self, obs, state) -> tuple[KeyedObs, KeyedState]:
        return keyed.observations_out(self.names, obs), keyed.beliefs_out(self.names, state)

    def reset(
        self, key: Array, params: Any, state: Mapping[str, Array] | None = None
    ) -> tuple[KeyedObs, KeyedState]:
        """Sample (or, with ``state``, set) an initial state and an observation of it."""
        if state is None:
            obs, drawn = self.inner.reset(key, self._params(params))
        else:
            obs, drawn = self.inner.reset(key, self._params(params), self._state_in(state, "reset"))
        return self._out(obs, drawn)

    def step(
        self,
        key: Array,
        state: Mapping[str, Array],
        action: Mapping[str, Array],
        params: Any,
        factors: Iterable[str] | None = None,
    ) -> tuple[KeyedObs, KeyedState]:
        """Sample the next state under ``action`` and an observation of it (the intended action,
        realised through ``params.actuator``; see :meth:`step_actuated`). ``factors`` names the
        state factors to step; the others keep their values (see :meth:`step_actuated`)."""
        obs, nxt, _ = self.step_actuated(key, state, action, params, factors)
        return obs, nxt

    def step_actuated(
        self,
        key: Array,
        state: Mapping[str, Array],
        action: Mapping[str, Array],
        params: Any,
        factors: Iterable[str] | None = None,
    ) -> tuple[KeyedObs, KeyedState, dict[str, Array]]:
        """:meth:`step`, also returning the realised action, keyed by action node.

        ``action`` is the intended action. An action node with a kernel in ``params.actuator``
        is realised as a draw from its column at the intended level, with a key folded from
        ``key``; any other is realised as given (and ``key`` is used as :meth:`step` always
        uses it). The world moves on the realised action.

        ``factors`` names the state factors to step, by factor name; the others keep their
        values and the observation is of the state so made. It is the System's own process that
        steps in parts (the driver's sequential and async orders step each component's factors
        at that component's turn); an outside environment wrapped by :meth:`wrap` has no parts
        to name, and is refused.
        """
        mask = None if factors is None else self._factor_mask(factors)
        convert = keyed.factor_actions_in if self.factor_actions else keyed.actions_in
        positional = convert(self.names, action, "step")  # checks the intended action
        realised = self._actuate(key, action, params)
        if realised is not action:
            positional = convert(self.names, realised, "step")
        state_in, params_in = self._state_in(state, "step"), self._params(params)
        if mask is None:
            obs, nxt = self.inner.step(key, state_in, positional, params_in)
        else:
            obs, nxt = self.inner.step(key, state_in, positional, params_in, mask)
        return (*self._out(obs, nxt), realised)

    def _factor_mask(self, factors: Iterable[str]) -> tuple[bool, ...]:
        """``factors`` as one bool per state factor in node order; unknown names, and an
        outside environment, are refused."""
        if self.factor_actions:
            raise ValueError(
                "step(factors=...) steps part of the System's own process; an environment "
                "wrapped by GenerativeEnv.wrap has no parts to name"
            )
        if isinstance(factors, str):
            raise TypeError(
                f"step(factors=...) takes the factor names as a sequence, got the string "
                f"{factors!r}; pass ({factors!r},)"
            )
        wanted = set(factors)
        unknown = sorted(wanted - set(self.names.factors))
        if unknown:
            raise ValueError(
                f"step(factors=...) names {unknown}, which are not state factors of this "
                f"process; its factors are {list(self.names.factors)}"
            )
        return tuple(n in wanted for n in self.names.factors)

    def _actuate(self, key: Array, action: Mapping[str, Array], params: Any) -> dict[str, Array]:
        """The realised action: each action node with a kernel drawn from its column at the
        intended level, under a key tagged apart from every key the process splits off ``key``;
        ``action`` itself, untouched, when there is no kernel."""
        kernels = getattr(params, "actuator", None) or {}
        if not kernels:
            return action
        keyed.check_mapping(
            self.names, "actions", kernels, "GenerativeEnv", label="params.actuator", partial=True
        )
        keys = jax.random.split(jax.random.fold_in(key, ACTUATOR_TAG), len(self.names.actions))
        out = {}
        for k, a in enumerate(self.names.actions):
            if a not in kernels:
                out[a] = action[a]
                continue
            table = check_kernel(
                kernels[a],
                f"GenerativeEnv: params.actuator[{a!r}]",
                self.names.kind(k),
                self.names.sizes[k] if self.names.sizes else None,
            )
            log_p = jnp.log(table)
            column = jnp.moveaxis(
                jnp.take(log_p, jnp.asarray(action[a], jnp.int32), axis=-1), 0, -1
            )
            out[a] = jax.random.categorical(keys[k], column).astype(jnp.int32)
        return out
