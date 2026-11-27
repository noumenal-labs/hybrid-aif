"""A black box: an external process declared to ``couple`` and ``run`` by its ports and its step.

An environment, or a world whose dynamics are hand-written, enters a coupling as one node of the
factor graph whose inside is not stated: an opaque map from its in-ports and its own carried
state to its out-ports and its next state. :func:`black_box` declares it by its ports alone,
each a name and a type (``(LEVEL, levels)`` or ``(FLOAT, width)``), and returns a real
``(System, SystemParams)`` that :func:`~hybrid_aif.composition.couple.couple` takes as a component
like any other: its in-ports are control nodes with no action node and no reader (an
:class:`~hybrid_aif.composition.merge.Identify` from an out-port drives one), its out-ports
emissions with no parents and the static kernel ``"black_box"`` (an ``Identify`` with a modality
or a control node feeds the other side), and ``System.black_box`` holds the types, the explicit
mark. It holds no factor, no table and no control or modality kernel.

Under :func:`~hybrid_aif.composition.runner.run` a black box is stepped by a custom step, the
step contract of every leaf: ``step(system, key, obs, carry, params) -> (out, carry, params)``,
``obs`` holding the in-ports (``None`` on the reset call) and ``out`` every out-port, by the names
of ``system`` (the view under the coupling's names). The step may be arbitrary host Python (a
camera, a socket): ``run`` calls it once per round from its Python loop and never traces,
compiles or batches it, its carry may be any Python object, and the NumPy arrays or Python
numbers it returns are converted and checked against the port types. :func:`env_step` makes that
step of a positional (pymdp-style) environment. Every face that needs the inside refuses a
System that is or holds a black box, each naming what to do: ``plan()`` and ``sample()`` as its
step, ``mirror``, ``with_mirror``, ``joint_planner`` over it, ``as_env``, ``sample_initial`` and
``sample_step``, the inference, planning and learning faces, and a sliced ``subsystem`` view.
``describe`` writes it as a black box with its ports and types (``relations.black_box``), and
its description length is unscored, listed by name and excluded from the totals.

Design: docs/composing-systems.md, "External processes"; docs/design/description-length.md D3.
"""

from collections.abc import Callable, Mapping
from numbers import Integral
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hybrid_aif.controls import FLOAT, LEVEL
from hybrid_aif.dtypes import as_floats
from hybrid_aif.envs.base import Env
from hybrid_aif.envs.generative import GenerativeEnv
from hybrid_aif.reads import BLACK_BOX, Emission
from hybrid_aif.system import System, SystemParams, keyed

__all__ = ["black_box", "env_step", "is_black_box"]

#: A port's type: ``(LEVEL, levels)`` or ``(FLOAT, width)``.
PortType = tuple[str, int]


def _types(ports: object, what: str) -> dict[str, PortType]:
    """``ports`` as a name-to-type mapping, each type checked."""
    if ports is None:
        return {}
    if not isinstance(ports, Mapping):
        raise TypeError(
            f"black_box({what}=...) maps each port's name to its type, (LEVEL, levels) or "
            f"(FLOAT, width); got a {type(ports).__name__}"
        )
    out: dict[str, PortType] = {}
    for name, kind_size in ports.items():
        if not isinstance(name, str) or not name or keyed.NAMESPACE_SEPARATOR in name:
            raise ValueError(
                f"black_box({what}=...) names the port {name!r}; a port's name is a non-empty "
                "string without a dot (the coupling prefixes it with the component's name)"
            )
        if not (isinstance(kind_size, tuple | list) and len(kind_size) == 2):
            raise ValueError(
                f"black_box({what}[{name!r}]) is {kind_size!r}; a port's type is (LEVEL, levels) "
                "or (FLOAT, width)"
            )
        kind, size = kind_size
        if kind not in (LEVEL, FLOAT) or isinstance(size, bool) or not isinstance(size, Integral):
            raise ValueError(
                f"black_box({what}[{name!r}]) is ({kind!r}, {size!r}); a port's type is "
                f"({LEVEL!r}, levels) or ({FLOAT!r}, width), the size an integer"
            )
        if size < 1:
            raise ValueError(
                f"black_box({what}[{name!r}]) has size {size}; a port carries at least one level "
                "or one dimension"
            )
        out[name] = (kind, int(size))
    return out


def black_box(
    ins: Mapping[str, PortType] | None = None, outs: Mapping[str, PortType] | None = None
) -> tuple[System, SystemParams]:
    """An external process declared by its ports alone, as ``(System, SystemParams)``.

    ``ins`` maps each in-port's name to its type, ``outs`` each out-port's, a type being
    ``(LEVEL, levels)`` or ``(FLOAT, width)`` (``hybrid_aif.controls.LEVEL`` and ``FLOAT``). The
    System has one control node per in-port, with no action node and no reader, and one emission
    per out-port, with no parents and the static kernel ``"black_box"``, in the order given;
    ``System.black_box`` holds every port's type. The params hold nothing.

    Couple it as any component: ``Identify("agent.a_v", "world.v")`` drives the in-port ``v``
    with an action node, ``Identify("world.fix", "agent.fix")`` feeds a modality from the
    out-port ``fix``; the types of the two ends must agree, and an in-port nothing drives leaves
    the coupling open, which ``run`` refuses. Step it under ``run`` with a custom step (a function
    of the step contract, or :func:`env_step`); ``System.ports`` reads its ports back.

    Refused: no port at all, a name given as both an in-port and an out-port, a dotted or empty
    name, a type that is not one of the two forms.
    """
    ins_t, outs_t = _types(ins, "ins"), _types(outs, "outs")
    if not ins_t and not outs_t:
        raise ValueError(
            "black_box() declares no port; a black box is its ports, so give ins=, outs= or both"
        )
    both = sorted(set(ins_t) & set(outs_t))
    if both:
        raise ValueError(
            f"black_box() names {both} as both an in-port and an out-port; one namespace holds "
            "a System's variables, so name the two apart"
        )
    system = System(
        B_deps=((),) * len(ins_t),
        state_names=(),
        control_names=tuple(ins_t),
        controls=tuple(range(len(ins_t))),
        action_names=(None,) * len(ins_t),
        emissions={o: Emission((), "sample", BLACK_BOX) for o in outs_t},
        black_box={**ins_t, **outs_t},
        policy_len=1,
    )
    params = SystemParams(transition={}, initial_state_prior={})
    system._validate(system._params_in(params, "black_box"))
    return system, params


def is_black_box(system: System) -> bool:
    """Whether ``system`` states nothing but black-box ports: a black box (:func:`black_box`), a
    view of one taken from a coupling, or a coupling or merge of black boxes alone (no state
    factor, no modality, every control node and emission a port). A composite that holds a black
    box beside a stated component is not one."""
    return isinstance(system, System) and system._only_ports


def _short(name: str) -> str:
    """A port's own name, the coupling's prefix dropped."""
    return name.rpartition(keyed.NAMESPACE_SEPARATOR)[2]


def _action_axes(params: Any, factors: int) -> list[int] | None:
    """The action levels each factor's transition takes, read off a tabular environment's own
    params (``params.transition``, one table per factor, the action axis last), or ``None`` where
    the params do not show them."""
    tables = getattr(params, "transition", None)
    if not isinstance(tables, tuple | list) or len(tables) != factors:
        return None
    shapes = [np.shape(t) for t in tables]
    if not all(len(shape) >= 2 for shape in shapes):
        return None
    return [int(shape[-1]) for shape in shapes]


def env_step(env: Env, params: Any = None, *, actions: Mapping[str, int] | None = None) -> Callable:
    """The step of a black box that runs a positional environment.

    ``env`` speaks the positional convention (:mod:`hybrid_aif.envs.base`): ``reset(key,
    params) -> (obs, state)`` and ``step(key, state, action, params) -> (obs, state)``, the
    observations a tuple in modality order and the action ``env.null_action()``'s shape.
    The black box the step runs declares one out-port per observation, in that order, each of the
    observation's type; its in-ports fill the action. Level in-ports make a level action (an
    integer vector, one level per state factor): each in-port is one level at its position,
    ``actions`` mapping an in-port's name to its state factor (by default in-port ``k`` at state
    factor ``k``), the other state factors at ``null_action``'s level. Float in-ports make a
    continuous action: their vectors concatenated in declaration order (in the widest of their
    float dtypes) are the whole vector, of ``null_action``'s size. In-ports of both kinds are
    refused. Where the environment's own params show its transitions (a tabular process:
    ``params.transition``, one table per state factor, the action axis last), a level outside a
    state factor's action axis and an in-port mapped onto a state factor that takes no action are
    refused; elsewhere the environment receives the levels as given. ``params`` are the
    environment's own, passed to every call.

    The reset call (``obs is None``) is ``env.reset(key, params)``, or ``env.reset(key, params,
    state)`` when ``run(carries=...)`` seeds the carry with a positional state; every other call
    is ``env.step(key, carry, action, params)``. The carry is the positional state, the params
    come back as given. The keys are ``run``'s, untouched, so the step's draws are the
    environment's own under the same key. A ``GenerativeEnv`` (keyed) is refused, naming its
    ``inner``.
    """
    if isinstance(env, GenerativeEnv):
        raise TypeError(
            "env_step takes a positional environment, and this one is a GenerativeEnv (keyed by "
            "a modelling System's names); pass its .inner, whose observations and action are "
            "positional"
        )
    if not isinstance(env, Env):
        raise TypeError(
            f"env_step takes a positional environment (hybrid_aif.envs.base.Env), got a "
            f"{type(env).__name__}"
        )
    positions = dict(actions or {})

    def ports(system: System) -> tuple[list[str], list[str]]:
        if not is_black_box(system):
            raise ValueError(
                "env_step runs a positional environment as the step of a black box, and the "
                "node it was given is not one; declare the environment with black_box(ins=..., "
                "outs=...) and couple that"
            )
        boxed = dict(system.black_box)
        ins = [u for u in system._table.control_names if u in boxed]
        outs = [e for e, _ in system.emissions if e in boxed]
        return ins, outs

    def action_of(system: System, ins: list[str], obs: Mapping[str, Array | None]) -> Array:
        null = jnp.asarray(env.null_action())
        kinds = {dict(system.black_box)[u][0] for u in ins}
        if len(kinds) > 1:
            raise ValueError(
                f"env_step: the in-ports {[_short(u) for u in ins]} are of both kinds, and a "
                "positional action is a level vector or a float vector; declare one kind"
            )
        #  a variable the reset left without a value is handed as run's Maybe (value, present)
        missing = [u for u in ins if obs.get(u) is None or hasattr(obs.get(u), "present")]
        if missing:
            raise ValueError(
                f"env_step: in-port(s) {missing} hold no value this round; the environment "
                "takes a whole action every step, so drive every in-port from a variable that "
                "has a value from round 0"
            )
        if kinds == {FLOAT}:
            vector = jnp.concatenate([jnp.ravel(v) for v in as_floats(*(obs[u] for u in ins))])
            if vector.size != null.size:
                raise ValueError(
                    f"env_step: the in-ports {[_short(u) for u in ins]} make a vector of width "
                    f"{vector.shape[0]}, and the environment's action has shape "
                    f"{tuple(null.shape)}; declare in-ports whose widths sum to the action's"
                )
            return vector
        action = null
        axes = _action_axes(params, null.shape[0])
        for k, u in enumerate(ins):
            at = positions.get(_short(u), k)
            if not 0 <= at < null.shape[0]:
                raise ValueError(
                    f"env_step: in-port {_short(u)!r} sits at factor {at}, and the environment's "
                    f"action has {null.shape[0]} position(s); pass actions={{port: factor}}"
                )
            level = jnp.asarray(obs[u], jnp.int32)  # traced inside run's compiled rounds
            if axes is not None and axes[at] < 2:
                raise ValueError(
                    f"env_step: in-port {_short(u)!r} sits at factor {at}, which the "
                    "environment's transitions take no action on (one level on its action axis); "
                    "map it onto a controlled factor with actions={port: factor}"
                )
            concrete = not isinstance(level, jax.core.Tracer)
            if concrete and axes is not None and not 0 <= int(level) < axes[at]:
                raise ValueError(
                    f"env_step: in-port {_short(u)!r} holds level {int(level)}, and factor {at} "
                    f"of the environment takes {axes[at]} action levels; drive it within "
                    f"[0, {axes[at]})"
                )
            if not concrete and axes is not None:
                # inside run's compiled rounds the level is traced: checked as the round runs
                level = eqx.error_if(
                    level,
                    (level < 0) | (level >= axes[at]),
                    f"env_step: in-port {_short(u)!r} holds a level outside [0, {axes[at]}), "
                    f"and factor {at} of the environment takes {axes[at]} action levels; drive "
                    f"it within [0, {axes[at]})",
                )
            action = action.at[at].set(level.astype(null.dtype))
        return action

    def step(system, key, obs, carry, params_view):
        ins, outs = ports(system)
        unknown = sorted(set(positions) - {_short(u) for u in ins})
        if unknown:
            raise ValueError(
                f"env_step(actions=...) names {unknown}, which are no in-port of the black box; "
                f"its in-ports are {[_short(u) for u in ins]}"
            )
        if obs is None:
            if carry is None:
                observed, state = env.reset(key, params)
            else:
                observed, state = env.reset(key, params, carry)
        else:
            observed, state = env.step(key, carry, action_of(system, ins, obs), params)
        observed = tuple(observed)
        if len(observed) != len(outs):
            raise ValueError(
                f"env_step: the environment returned {len(observed)} observation(s), and the "
                f"black box declares {len(outs)} out-port(s) {[_short(e) for e in outs]}; "
                "declare one out-port per observation, in the environment's order"
            )
        return dict(zip(outs, observed, strict=True)), state, params_view

    return step
