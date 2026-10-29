"""Coupling through blankets: participants, typed ports, and the wires between them.

A run's participants are **minds** and **processes**, each named at :func:`couple`. A mind is
``(System, SystemParams, RespondFn)``: its in-ports are its modalities, its out-ports its action
nodes. A process is ``(GenerativeEnv, env_params)``: its in-ports are its action inputs (the
env's action nodes), its out-ports its observations. A port is written ``"participant.port"``
(the participant's name up to the first dot, the port's name after it).

Every port has a type, a kind and a size: a discrete port carries a level, its size the level
count; a continuous port carries a float vector, its size the width. A mind's action nodes are
typed by its control space (``params.control``, its readers' tables), its modalities by their
likelihoods; a process's inputs by its env's names, its observations by ``env_params.likelihood``.

A :class:`Wire` is an identity between an out-port and an in-port of one type: an action node to
a process input, a process observation to a mind's modality, or an action node to a mind's
modality. It carries no kernel; anything that transforms is a System run as a process. An action
node is realised once, when its mind chooses, from that mind's ``params.actuator`` (a delta where
there is no entry); the realised value stands until the mind chooses again, and every wire from
the node carries it. :func:`couple` checks the participants, the
ports and the wires once and returns a :class:`Wiring`, which
:func:`~hybrid_aif.composition.driver.run` executes.

Design: docs/design/control-states.md C12, C13; docs/design/nesting-systems.md principle 6.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hybrid_aif.controls import FLOAT, LEVEL
from hybrid_aif.envs.base import Env
from hybrid_aif.envs.generative import GenerativeEnv, GenerativeEnvParams
from hybrid_aif.system import System, SystemParams
from hybrid_aif.system.core import _observation_type

if TYPE_CHECKING:
    from hybrid_aif.composition.driver import RespondFn

__all__ = ["Wire", "Wiring", "couple"]

#: A port's type: its kind (``"level"`` or ``"float"``) and its size (a level count or a width).
PortType = tuple[str, int]


class Wire(NamedTuple):
    """An identity from the out-port ``source`` to the in-port ``target``, both written
    ``"participant.port"`` and of one type."""

    source: str
    target: str


class Mind(NamedTuple):
    """A mind as :func:`couple` holds it: its System, its params, its respond function, and its
    action nodes' kernels as ``params.actuator`` declares them, checked."""

    system: System
    params: SystemParams
    respond: "RespondFn"
    kernels: dict[str, Array]


class Process(NamedTuple):
    """A process as :func:`couple` holds it: a keyed env and its params."""

    env: GenerativeEnv
    params: Any


class Wiring(NamedTuple):
    """A checked coupling: the participants, the wires, and what feeds every in-port.

    ``minds`` and ``processes`` are keyed by participant name in the order given to
    :func:`couple` (minds respond in that order under ``"sequential"``); ``wires`` as given;
    ``fixed`` holds the constant of each process input held at one; ``types`` gives every port's
    :data:`PortType`, keyed ``"participant.port"``; ``feeds`` maps every in-port to the out-port
    its wire comes from, or to ``None`` for a process input in ``fixed``.
    """

    minds: dict[str, Mind]
    processes: dict[str, Process]
    wires: tuple[Wire, ...]
    fixed: dict[str, Array]
    types: dict[str, PortType]
    feeds: dict[str, str | None]

    def null(self, port: str) -> Array:
        """The value an out-port holds before its participant has emitted: level 0, or zeros
        of the port's width."""
        kind, size = self.types[port]
        if kind == LEVEL:
            return jnp.zeros((), jnp.int32)
        return jnp.zeros((size,), jnp.float32)


def describe_type(kind: str, size: int) -> str:
    """A port type in words, for a sentence."""
    if kind == LEVEL:
        return f"discrete, {size} levels"
    return f"continuous, width {size}"


def _process_observation_types(name: str, env: GenerativeEnv, params: Any) -> dict[str, PortType]:
    """A process's observations, typed from ``env_params.likelihood``; an outside env's own
    params give a continuous observation's width from its shape, and a discrete one no level
    count, which is refused."""
    mods = env.names.modalities
    likelihood = getattr(params, "likelihood", None)
    if isinstance(params, GenerativeEnvParams) and isinstance(likelihood, Mapping):
        return {m: _observation_type(likelihood[m], f"env_params.likelihood[{m!r}]") for m in mods}
    obs, _ = jax.eval_shape(lambda k: env.reset(k, params), jax.random.PRNGKey(0))
    out: dict[str, PortType] = {}
    for m in mods:
        shaped = obs[m]
        if jnp.issubdtype(shaped.dtype, jnp.integer):
            raise ValueError(
                f"couple(): process {name!r} emits {m!r} as a level, and its params give no level "
                "count; pass its params as GenerativeEnvParams keyed by name "
                "(system.from_positional('params', ...)), whose likelihood types the port"
            )
        if shaped.ndim != 1:
            raise ValueError(
                f"couple(): process {name!r} emits {m!r} with shape {tuple(shaped.shape)}; a "
                "continuous port carries a vector, shape (width,), so emit one"
            )
        out[m] = (FLOAT, int(shaped.shape[0]))
    return out


def _check_name(name: object, taken: set[str]) -> str:
    if not isinstance(name, str) or not name or "." in name:
        raise ValueError(
            f"couple(): participant name {name!r} is not a name; give each participant a "
            "non-empty string without a dot, since a port is written 'participant.port'"
        )
    if name in taken:
        raise ValueError(
            f"couple(): two participants are named {name!r}; give every mind and process its "
            "own name"
        )
    taken.add(name)
    return name


def _mind(name: str, entry: object) -> tuple[Mind, dict[str, PortType], dict[str, PortType]]:
    """A mind's entry checked, with its in-port and out-port types keyed by port name."""
    if not (isinstance(entry, tuple | list) and len(entry) == 3):
        raise TypeError(
            f"couple(): minds[{name!r}] is a {type(entry).__name__}; a mind is "
            "(System, SystemParams, RespondFn)"
        )
    system, params, respond = entry
    if not isinstance(system, System) or not isinstance(params, SystemParams):
        raise TypeError(
            f"couple(): minds[{name!r}] holds ({type(system).__name__}, "
            f"{type(params).__name__}, ...); a mind is (System, SystemParams, RespondFn)"
        )
    if not callable(respond):
        raise TypeError(
            f"couple(): minds[{name!r}]'s third entry is a {type(respond).__name__}; pass its "
            "RespondFn (agent_respond(...) or ancestral_respond(...))"
        )
    positional = system._params_in(params, "couple")
    names = system._names(positional)
    outs = {a: (names.kind(k), names.sizes[k]) for k, a in enumerate(names.actions)}
    ins = dict(system._modality_types(positional))
    kernels = system._actuator(params, f"couple(): minds[{name!r}]")
    return Mind(system, params, respond, kernels), ins, outs


def _process(name: str, entry: object) -> tuple[Process, dict[str, PortType], dict[str, PortType]]:
    """A process's entry checked, with its in-port and out-port types keyed by port name."""
    if isinstance(entry, Env) and not isinstance(entry, GenerativeEnv):
        raise TypeError(
            f"couple(): processes[{name!r}] is a bare positional {type(entry).__name__}; a "
            "process speaks names, so pass (GenerativeEnv.wrap(env, system), env_params)"
        )
    if isinstance(entry, Env) or not (isinstance(entry, tuple | list) and len(entry) == 2):
        raise TypeError(
            f"couple(): processes[{name!r}] is a {type(entry).__name__}; a process is "
            "(GenerativeEnv, env_params), as System.as_env returns it"
        )
    env, params = entry
    if isinstance(env, System):
        raise TypeError(
            f"couple(): processes[{name!r}] is a System; run it as a process with "
            "system.as_env(params), which gives (GenerativeEnv, env_params)"
        )
    if not isinstance(env, GenerativeEnv):
        raise TypeError(
            f"couple(): processes[{name!r}] holds a bare positional {type(env).__name__}; a "
            "process speaks names, so pass (GenerativeEnv.wrap(env, system), env_params)"
        )
    kernels = getattr(params, "actuator", None) or {}
    if kernels:
        raise ValueError(
            f"couple(): process {name!r} declares params.actuator {sorted(kernels)}; an actuator "
            "belongs to the mind that emits the action, in its params.actuator, so give the "
            "process its params with actuator={}"
        )
    names = env.names
    if names.actions and not names.sizes:
        raise ValueError(
            f"couple(): process {name!r} gives its inputs {list(names.actions)} no level count or "
            "width; wrap the env with GenerativeEnv.wrap(env, system, params), whose params "
            "type them"
        )
    ins = {a: (names.kind(k), names.sizes[k]) for k, a in enumerate(names.actions)}
    outs = _process_observation_types(name, env, params)
    return Process(env, params), ins, outs


def _fixed_value(port: str, value: object, kind: str, size: int) -> Array:
    """A constant for a process input, checked against the input's type like a wire's source."""
    value = jnp.asarray(value)
    want = describe_type(kind, size)
    if kind == LEVEL:
        ok = value.shape == () and jnp.issubdtype(value.dtype, jnp.integer)
        if ok:
            ok = 0 <= int(value) < size
        if not ok:
            raise ValueError(
                f"couple(): fixed[{port!r}] is {np.asarray(value)!r} (dtype {value.dtype}, shape "
                f"{tuple(value.shape)}); process input {port!r} is {want}, so pass one integer "
                "level in range"
            )
        return value.astype(jnp.int32)
    ok = jnp.issubdtype(value.dtype, jnp.floating) and value.ndim == 1
    ok = ok and value.shape == (size,)
    if not ok:
        raise ValueError(
            f"couple(): fixed[{port!r}] has dtype {value.dtype} and shape {tuple(value.shape)}; "
            f"process input {port!r} is {want}, so pass a float vector of that width"
        )
    return value


def couple(
    *,
    minds: Mapping[str, tuple] | None = None,
    processes: Mapping[str, tuple] | None = None,
    wires: Sequence[Wire] = (),
    fixed: Mapping[str, object] | None = None,
) -> Wiring:
    """Participants and the wires between them, checked into a :class:`Wiring`.

    ``minds`` maps a name to ``(System, SystemParams, RespondFn)``, ``processes`` a name to
    ``(GenerativeEnv, env_params)`` (``System.as_env`` gives one; ``GenerativeEnv.wrap`` keys an
    outside positional env). ``wires`` are :class:`Wire` s between ``"participant.port"``
    spellings; ``fixed`` holds a constant per tick for a process input, keyed the same way.

    Refused, each with its fix named: an unknown participant or port; a wire against the
    allowed directions or between ports of two types; an in-port fed twice; a process input
    neither wired nor in ``fixed``; a mind's modality fed by no wire; ``fixed`` naming anything
    but a process input, or a value of another type; a process whose ``env_params.actuator`` is
    non-empty (an actuator is the emitting mind's); a port whose participant gives it no level
    count or width. An out-port read by nothing is ignored.

    Scope: a modality is typed by its likelihood, so a mind or a process couples when each of its
    modalities has a tabular, neural (``MLPConditional``) or linear-Gaussian likelihood; an
    amortized mind (a ``PredictorHead`` likelihood) cannot be coupled.
    """
    minds = dict(minds or {})
    processes = dict(processes or {})
    fixed = dict(fixed or {})
    taken: set[str] = set()
    mind_table: dict[str, Mind] = {}
    proc_table: dict[str, Process] = {}
    #  per participant: (role, in-port types, out-port types)
    ports: dict[str, tuple[str, dict[str, PortType], dict[str, PortType]]] = {}
    for name, entry in minds.items():
        _check_name(name, taken)
        mind_table[name], ins, outs = _mind(name, entry)
        ports[name] = ("mind", ins, outs)
    for name, entry in processes.items():
        _check_name(name, taken)
        proc_table[name], ins, outs = _process(name, entry)
        ports[name] = ("process", ins, outs)

    types: dict[str, PortType] = {}
    for name, (_, ins, outs) in ports.items():
        types.update({f"{name}.{p}": t for p, t in ins.items()})
        types.update({f"{name}.{p}": t for p, t in outs.items()})

    def resolve(port: object, end: str) -> tuple[str, str, str]:
        """``(participant, role, port)`` for one end of a wire, or the sentence that refuses it."""
        what = "an out-port" if end == "source" else "an in-port"
        if not isinstance(port, str) or "." not in port:
            raise ValueError(
                f"couple(): a wire's {end} is {port!r}; write a port as 'participant.port', "
                f"participants {sorted(ports)}"
            )
        who, name = port.split(".", 1)
        if who not in ports:
            raise ValueError(
                f"couple(): wire {end} {port!r} names participant {who!r}, which is not one; the "
                f"participants are {sorted(ports)}"
            )
        role, ins, outs = ports[who]
        side, other = (outs, ins) if end == "source" else (ins, outs)
        if name not in side:
            candidates = sorted(f"{who}.{p}" for p in side)
            hint = (
                f" ({port!r} is {'an in-port' if end == 'source' else 'an out-port'}, so it "
                f"is a wire's {'target' if end == 'source' else 'source'})"
                if name in other
                else ""
            )
            raise ValueError(
                f"couple(): wire {end} {port!r} is not {what} of {role} {who!r}{hint}; its "
                f"{'out' if end == 'source' else 'in'}-ports are {candidates}"
            )
        return who, role, name

    feeds: dict[str, str | None] = {}
    checked: list[Wire] = []
    for wire in wires:
        if not isinstance(wire, Wire):
            raise TypeError(
                f"couple(): wires holds a {type(wire).__name__}; pass Wire(source, target)"
            )
        _, s_role, _ = resolve(wire.source, "source")
        _, t_role, _ = resolve(wire.target, "target")
        if s_role == "process" and t_role == "process":
            raise ValueError(
                f"couple(): Wire({wire.source!r}, {wire.target!r}) runs from a process's "
                "observation to a process's input; a wire runs from an action node to a process "
                "input or a mind's modality, or from a process observation to a mind's modality"
            )
        s_type, t_type = types[wire.source], types[wire.target]
        if s_type != t_type:
            raise ValueError(
                f"couple(): Wire({wire.source!r}, {wire.target!r}) joins "
                f"{describe_type(*s_type)} to {describe_type(*t_type)}; a wire is an identity, "
                "so both ends must have one type"
            )
        if wire.target in feeds:
            raise ValueError(
                f"couple(): in-port {wire.target!r} is the target of two wires, from "
                f"{feeds[wire.target]!r} and {wire.source!r}; an in-port takes one wire"
            )
        feeds[wire.target] = wire.source
        checked.append(wire)

    inputs = {f"{n}.{p}" for n, (role, ins, _) in ports.items() if role == "process" for p in ins}
    constants: dict[str, Array] = {}
    for port, value in fixed.items():
        if port not in inputs:
            raise ValueError(
                f"couple(): fixed= names {port!r}, which is not a process input; fixed= holds a "
                f"constant for a process input, one of {sorted(inputs)}"
            )
        if port in feeds:
            raise ValueError(
                f"couple(): process input {port!r} is both wired (from {feeds[port]!r}) and in "
                "fixed=; an in-port takes one feed, so drop one"
            )
        constants[port] = _fixed_value(port, value, *types[port])
        feeds[port] = None

    for name, (role, ins, _) in ports.items():
        for p in ins:
            port = f"{name}.{p}"
            if port in feeds:
                continue
            if role == "process":
                raise ValueError(
                    f"couple(): process input {port!r} is fed by no wire; wire an action node "
                    f"to it, or hold it at a constant with fixed={{{port!r}: value}}"
                )
            raise ValueError(
                f"couple(): mind modality {port!r} is fed by no wire; wire a process "
                "observation or an action node to it, or drop the modality from the mind"
            )

    return Wiring(mind_table, proc_table, tuple(checked), constants, types, feeds)
