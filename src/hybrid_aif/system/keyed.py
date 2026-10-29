"""Names at the faces: beliefs, observations and actions as mappings keyed by name.

Every public face of a :class:`~hybrid_aif.system.System` takes and returns these three as
mappings, while tables, dependency indices and scan carries stay positional. The conversion
happens here, at trace time, from static names; no traced value is ever used as a key.

- **beliefs**: ``{factor_name: belief}``, one entry per factor;
- **observations**: ``{modality_name: observation}``, one entry per modality;
- **actions**: ``{action: value}``, one entry per action node (``a_<factor>`` for the sugar's
  nodes), each the value emitted for its control node: a level index for a discrete node, a
  float vector for a continuous one.

Which control nodes are continuous, and how wide, is read off the params (a continuous node's
box is in ``SystemParams.control``), so converting a continuous node's value takes them: the
System's converters accept ``params=``, and without them every node reads as discrete.

Names are ``System.factor_names`` / ``modality_names``, or the generated ``factor_{i}`` /
``modality_{m}``. Entries are addressed by name, never by position: a face builds its mapping in
node order, but one that has passed through ``jax.jit``, ``vmap``, ``lax.scan`` or ``tree_map``
comes back with its keys sorted. A mapping with a missing or unknown key raises, naming both; a
tuple, list or bare array raises too, pointing at :meth:`System.from_positional`, which (with
:meth:`System.to_positional`) is the one way positional data enters or leaves.

Mappings that supply values (beliefs, observations, actions, params) are complete: a missing key
raises. Mappings that restrict or select are partial, an absent key meaning unrestricted:
``present`` (one flag per modality, absent = observed) and ``legal`` (one level mask per
control node, absent = every level legal). Both kinds refuse unknown keys; an action mapping
keyed by a control node, and a control-node mapping keyed by an action node, are refused naming
the node to key by. An
:class:`~hybrid_aif.inference.detail.InferenceDetail` is keyed too: its per-modality fields by
modality name, ``unsupported`` by factor name (kind ``"detail"``). So are the per-variable
containers of :class:`~hybrid_aif.system.SystemParams` and of an environment's parameters
(kind ``"params"``): ``likelihood`` by modality name, ``transition`` and
``initial_state_prior`` by factor name.

Design: docs/design/control-states.md C10.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp

from hybrid_aif import controls
from hybrid_aif.controls import FLOAT, LEVEL, ControlLayout, ControlStep
from hybrid_aif.inference.detail import InferenceDetail

Kind = Literal["beliefs", "observations", "actions", "detail", "params"]
KINDS: tuple[str, ...] = ("beliefs", "observations", "actions", "detail", "params")

#: The prefix of the name the control sugar gives a factor's control node: ``u_<f>``.
CONTROL_PREFIX = "u_"
#: The namespace separator of composite names (``"<component>.<name>"``).
NAMESPACE_SEPARATOR = controls.NAMESPACE_SEPARATOR


def control_name(factor: str) -> str:
    """The name the control sugar gives the control node of ``factor``: ``"u_<factor>"``,
    namespaced like any node (``"world.u_location"`` for ``"world.location"``)."""
    return controls.control_name(factor)


def action_name(factor: str) -> str:
    """The name the control sugar gives the action node of ``factor``: ``"a_<factor>"``,
    namespaced like any node."""
    return controls.action_name(factor)


def generated_names(prefix: str, count: int) -> tuple[str, ...]:
    """The names a System fills in when none are declared: ``<prefix>_0``, ``<prefix>_1``, …"""
    return tuple(f"{prefix}_{i}" for i in range(count))


class Names(NamedTuple):
    """A System's names (``System.names``): a view of its name tuples, in node-index order.

    ``states`` are the state factors' names and ``controls`` the control nodes', both from
    ``System.factor_names`` (declared, or the generated ``factor_{i}`` / ``u_factor_{i}``);
    ``modalities`` is ``System.modality_names``; ``actions`` the action nodes' (the action
    keys), one per control node in the same order, from ``System.action_names``. Node-index
    order is the order of every positional structure (tables, policy columns, the converters'
    tuples); iterating a mapping gives no such guarantee, so take order from here.
    """

    states: tuple[str, ...]
    controls: tuple[str, ...]
    modalities: tuple[str, ...]
    actions: tuple[str, ...] = ()


class FaceNames(NamedTuple):
    """The internal form of :class:`Names`, with what the conversions need. All static.

    ``factors`` are the state factors' names (the belief keys), ``controls`` the control nodes'
    (the keys of ``legal`` and of a belief-valued policy), ``actions`` their action nodes' (the
    action keys); ``controlled_factors`` gives each control node's state factor (its owner
    on a sugar-shaped System, else its first reader), the column a per-factor positional action
    holds it in (``None`` when two nodes would share a column); ``kinds`` gives each node's kind
    (:data:`~hybrid_aif.controls.LEVEL` or :data:`~hybrid_aif.controls.FLOAT`; empty when the
    params were not read, which reads every node as discrete) and ``sizes`` its width; ``layout``
    is the flat action layout of an all-continuous space (``None`` otherwise). A belief mapping
    may also carry a discrete control node's entry (``node_beliefs``) and the entry of an action
    node a modality observes (``sensed``). ``gone`` pairs each control
    or action node a ``merge()`` binding removed with the sentence that refuses a key naming it.
    """

    factors: tuple[str, ...]
    modalities: tuple[str, ...]
    controlled_factors: tuple[int, ...] | None
    layout: ControlLayout | None = None
    controls: tuple[str, ...] = ()
    kinds: tuple[str, ...] = ()
    sizes: tuple[int, ...] = ()
    gone: tuple[tuple[str, str], ...] = ()
    actions: tuple[str, ...] = ()
    sensed: tuple[str, ...] = ()

    def kind(self, k: int) -> str:
        """Node ``k``'s kind (discrete when the params were not read)."""
        return self.kinds[k] if self.kinds else LEVEL

    @property
    def mixed(self) -> bool:
        """The space holds both discrete and continuous nodes."""
        return LEVEL in self.kinds and FLOAT in self.kinds

    @property
    def node_beliefs(self) -> tuple[str, ...]:
        """The control nodes that carry a belief entry: the discrete ones."""
        return tuple(n for k, n in enumerate(self.controls) if self.kind(k) == LEVEL)

    @property
    def public(self) -> Names:
        return Names(self.factors, self.controls, self.modalities, self.actions)

    def expected(self, kind: str) -> tuple[str, ...]:
        """The keys a mapping of ``kind`` carries (``"beliefs"``, ``"observations"``,
        ``"actions"``, ``"controls"``; a ``present`` flag, a ``legal`` mask or a ``"detail"``
        field converts as one of these)."""
        if kind == "beliefs":
            return self.factors
        if kind == "observations":
            return self.modalities
        if kind == "actions":
            return self.actions
        if kind == "controls":
            return self.controls
        raise ValueError(
            f"kind must be one of ('beliefs', 'observations', 'actions', 'controls'), got {kind!r}"
        )


def _noun(kind: str) -> str:
    return {
        "beliefs": "factor",
        "observations": "modality",
        "actions": "action node",
        "controls": "control node",
    }[kind]


def _crossed(names: FaceNames, kind: str, key: str) -> str | None:
    """The sentence's tail for an action keyed by its control node, or a control-node mapping
    keyed by an action node; ``None`` for any other key."""
    if kind == "actions" and key in names.controls:
        action = names.actions[names.controls.index(key)]
        return f"the control node of action {action!r}; key the action by {action!r}"
    if kind == "controls" and key in names.actions:
        node = names.controls[names.actions.index(key)]
        return f"the action node of control node {node!r}; key it by {node!r}"
    return None


def check_mapping(
    names: FaceNames,
    kind: str,
    mapping: Any,
    face: str,
    label: str | None = None,
    optional: tuple[str, ...] = (),
    partial: bool = False,
) -> None:
    """Raise unless ``mapping`` is a Mapping with exactly the expected keys for ``kind``.

    ``label`` names the argument in the sentence when it is not ``kind`` itself (``"present"``,
    ``"state"``); it has ``kind``'s structure and converts as ``kind``. Keys in ``optional`` may
    be present or absent; with ``partial`` every key may be absent (a mapping that restricts or
    selects), and only unknown keys raise.
    """
    expected = names.expected(kind)
    what = kind if label is None else label
    if not isinstance(mapping, Mapping):
        positional = isinstance(mapping, tuple | list) or hasattr(mapping, "shape")
        how = (
            f"convert positional data with system.from_positional({kind!r}, ...)"
            if positional
            else f"got {type(mapping).__name__}"
        )
        raise TypeError(
            f"{face} takes {what} as a mapping keyed by {_noun(kind)} name "
            f"({list(expected)}); {how}"
        )
    missing = [] if partial else [k for k in expected if k not in mapping]
    unknown = [k for k in mapping if k not in expected and k not in optional]
    gone = dict(names.gone)
    removed = [k for k in unknown if k in gone]
    if removed:
        what = kind if label is None else label
        raise ValueError(f"{face}: {what} names {removed[0]!r}, {gone[removed[0]]}")
    crossed = [(k, tail) for k in unknown if (tail := _crossed(names, kind, k)) is not None]
    if crossed:
        what = kind if label is None else label
        raise ValueError(f"{face}: {what} names {crossed[0][0]!r}, {crossed[0][1]}")
    if missing or unknown:
        plural = label is None
        parts = []
        if missing:
            parts.append(f"{'are' if plural else 'is'} missing {missing}")
        if unknown:
            noun = "key" if len(unknown) == 1 else "keys"
            parts.append(f"{'have' if plural else 'has'} unknown {noun} {unknown}")
        keys = "known keys" if partial else "expected keys"
        raise ValueError(f"{face}: {what} {' and '.join(parts)}; {keys} {list(expected)}")


def _check_length(names: FaceNames, kind: str, positional: Any, want: int) -> tuple:
    if isinstance(positional, Mapping):
        raise TypeError(
            f"from_positional({kind!r}, ...) takes positional data; this is already a mapping"
        )
    out = tuple(positional)
    if len(out) != want:
        raise ValueError(
            f"from_positional({kind!r}, ...) got {len(out)} entries for "
            f"{want} {_noun(kind)}(s) {list(names.expected(kind))}"
        )
    return out


# ------------------------------------------------------------------ beliefs and observations


def beliefs_in(names: FaceNames, mapping: Any, face: str) -> tuple:
    """``{factor: belief}`` as the positional belief tuple of the state factors. A discrete
    control node's entry is optional and is not part of the tuple."""
    check_mapping(names, "beliefs", mapping, face, optional=(*names.node_beliefs, *names.sensed))
    return tuple(mapping[n] for n in names.factors)


def beliefs_out(names: FaceNames, beliefs, nodes: Mapping[str, Any] | None = None) -> dict:
    """The positional belief tuple as ``{factor: belief}``, in factor order, followed by the
    control nodes' entries ``nodes`` when given."""
    out = dict(zip(names.factors, tuple(beliefs), strict=True))
    if nodes:
        out.update(nodes)
    return out


def sensed_beliefs_in(names: FaceNames, mapping: Mapping[str, Any]) -> dict:
    """The observed action nodes' entries a belief mapping carries (possibly none)."""
    return {n: mapping[n] for n in names.sensed if n in mapping}


def node_beliefs_in(names: FaceNames, mapping: Mapping[str, Any]) -> dict:
    """The control nodes' entries a belief mapping carries (possibly none)."""
    return {n: mapping[n] for n in names.node_beliefs if n in mapping}


def observations_in(names: FaceNames, mapping: Any, face: str) -> tuple:
    """``{modality: observation}`` as the positional observation tuple."""
    check_mapping(names, "observations", mapping, face)
    return tuple(mapping[n] for n in names.modalities)


def observations_out(names: FaceNames, observations) -> dict:
    """The positional observation tuple as ``{modality: observation}``, in modality order."""
    return dict(zip(names.modalities, tuple(observations), strict=True))


def present_in(names: FaceNames, present: Any, face: str, absent: Any = True) -> tuple:
    """``{modality: flag}`` as the positional flag tuple, one per modality. The mapping is
    partial: a modality it leaves out is observed, its flag ``absent`` (default the static
    ``True``). An entry that is itself a mapping (a per-level availability under a control node)
    raises."""
    check_mapping(names, "observations", present, face, label="present", partial=True)
    for name, flag in present.items():
        if isinstance(flag, Mapping):
            raise ValueError(
                f"{face}: present[{name!r}] under the levels of a control node is a planning "
                "form (infer_policies, planner_context); at this face pass the flag for this tick"
            )
    return tuple(present[n] if n in present else absent for n in names.modalities)


def legal_list(legal: list, face: str, form: str = "(P,)") -> Any:
    """A Python list given as ``legal=``: the bare mask over the policy set, as an array.

    A list of bools is that mask; anything else (a list of per-node masks) raises, naming
    the array form and the keyed one.
    """
    try:
        out = jnp.asarray(legal, bool)
    except (TypeError, ValueError):
        out = None
    if out is None or not all(
        isinstance(v, bool) or str(getattr(v, "dtype", "")) == "bool" for v in _flat(legal)
    ):
        raise TypeError(
            f"{face}: a list as legal= is the {form} bool mask over the policy set; a mask over a "
            "control node's levels is keyed by the node, {node: a boolean mask over its levels}"
        )
    return out


def _flat(value: Any) -> list:
    if isinstance(value, list | tuple):
        return [v for item in value for v in _flat(item)]
    return [value]


def legal_in(names: FaceNames, legal: Any, face: str) -> tuple:
    """``{node: mask over its levels}`` as one entry per control node, in node order, ``None``
    for a node the mapping leaves out (every level legal).

    A tuple or list (the per-state-factor spelling) raises, naming the keyed form; unknown keys
    raise, naming the control nodes.
    """
    if isinstance(legal, tuple | list):
        raise TypeError(
            f"{face}: legal= is keyed by control node, {{node: a boolean mask over its levels}} "
            f"for nodes {list(names.controls)} (or one bool per policy), not a tuple by state "
            "factor"
        )
    check_mapping(names, "controls", legal, face, label="legal", partial=True)
    return tuple(legal.get(n) for n in names.controls)


# ------------------------------------------------------------------------------------ actions


def factor_actions_in(names: FaceNames, mapping: Any, face: str, lead_shape: tuple | None = None):
    """``{node: value}`` as a per-state-factor positional action, for an outside environment
    that speaks pymdp's convention.

    Discrete controls give an int32 ``(..., num_factors)`` array (each control in its owning
    factor's column), the other columns the no-op level 0; continuous ones the flat float
    ``(..., action_dim)`` vector. Leading axes are read off the entries; ``lead_shape`` sizes
    them when no factor is controlled (default ``()``). A level given as a non-integer dtype,
    and a continuous block of the wrong width, raise (both are static, so the checks run at
    trace time).
    """
    check_mapping(names, "actions", mapping, face)
    keys = names.actions
    _check_factor_columns(names, face)
    _refuse_mixed(names, face)
    values = [jnp.asarray(mapping[k]) for k in keys]
    if names.layout is not None:
        _check_continuous(names, keys, values, face)
        return jnp.concatenate(values, axis=-1)
    _check_levels(names, keys, values, face)
    cols = [v.astype(jnp.int32) for v in values]
    if cols:
        shape = jnp.broadcast_shapes(*(c.shape for c in cols))
    else:
        shape = tuple(lead_shape or ())
    by_factor = dict(zip(names.controlled_factors, cols, strict=True))
    zero = jnp.zeros(shape, jnp.int32)
    return jnp.stack(
        [
            jnp.broadcast_to(by_factor[f], shape) if f in by_factor else zero
            for f in range(len(names.factors))
        ],
        axis=-1,
    )


def factor_actions_out(names: FaceNames, action, check: bool = False) -> dict:
    """A per-state-factor positional action (an outside environment's) as ``{node: value}``.

    With ``check``, a concrete nonzero uncontrolled column of a level-index action raises: it
    would be dropped, and no factor can take it.
    """
    action = jnp.asarray(action)
    keys = names.actions
    _check_factor_columns(names, "a per-factor action")
    _refuse_mixed(names, "a per-factor action")
    if names.layout is not None:
        return _split_flat(names, action)
    n = len(names.factors)
    if action.shape[-1:] != (n,):
        raise ValueError(
            f"a positional action has one level per factor, (..., {n}) for "
            f"{list(names.factors)}; got shape {tuple(action.shape)}"
        )
    if not jnp.issubdtype(action.dtype, jnp.integer):
        raise ValueError(
            f"actions {list(keys)} are not integer (positional dtype {action.dtype}); a discrete "
            f"control's action is a level index (an integer array){_params_hint(names)}"
        )
    if check and not isinstance(action, jax.core.Tracer):
        stray = [
            names.factors[f]
            for f in range(n)
            if f not in names.controlled_factors and bool(jnp.any(action[..., f] != 0))
        ]
        if stray:
            raise ValueError(
                f"from_positional('actions', ...): uncontrolled factor(s) {stray} carry a "
                "nonzero level; an uncontrolled factor has only the no-op level 0"
            )
    return {k: action[..., f] for k, f in zip(keys, names.controlled_factors, strict=True)}


def _check_factor_columns(names: FaceNames, face: str) -> None:
    """Raise unless each control node has a state factor's column of its own."""
    if names.controlled_factors is None:
        raise ValueError(
            f"{face}: a per-factor action holds each control node in a state factor's column, "
            f"and two of the control nodes {list(names.actions)} are first read by the same "
            "factor; use the System's own process (System.as_env), which takes one level per "
            "control node"
        )


def _refuse_mixed(names: FaceNames, face: str) -> None:
    if names.mixed:
        raise ValueError(
            f"{face}: a per-factor positional action is a level array or one flat float vector, "
            "and this System's control space mixes discrete and continuous nodes; use the "
            "System's own process (System.as_env), which takes one entry per control node"
        )


def _check_continuous(names: FaceNames, keys, values, face: str) -> None:
    owners = names.controlled_factors
    for k, (key, v) in enumerate(zip(keys, values, strict=True)):
        if names.kind(k) != FLOAT:
            continue
        want = names.sizes[k]
        if v.shape[-1:] != (want,):
            whose = f"the control on {names.factors[owners[k]]!r}" if owners else "the control"
            raise ValueError(
                f"{face}: actions[{key!r}] is {whose}, a (..., {want}) block; got shape "
                f"{tuple(v.shape)}"
            )
        if not jnp.issubdtype(v.dtype, jnp.floating):
            raise ValueError(
                f"{face}: actions[{key!r}] has dtype {v.dtype}; this System's controls "
                f"{list(keys)} take float blocks"
            )


def _check_levels(names: FaceNames, keys, values, face: str) -> None:
    floats = [
        key
        for k, (key, v) in enumerate(zip(keys, values, strict=True))
        if names.kind(k) == LEVEL and not jnp.issubdtype(v.dtype, jnp.integer)
    ]
    if floats:
        hint = _params_hint(names)
        raise ValueError(
            f"{face}: actions {floats} are not integer; a discrete control's action is a level "
            f"index (an integer array){hint}"
        )


def _params_hint(names: FaceNames) -> str:
    """The pointer at ``params=`` when the names were read without the params."""
    if names.kinds or not names.controls:
        return ""
    return (
        " (a continuous node's box lives in params.control: pass params= — to the converter, "
        "or to GenerativeEnv.wrap — to read its float value)"
    )


def _split_flat(names: FaceNames, action) -> dict:
    want = names.layout.action_dim
    keys = names.actions
    if action.shape[-1:] != (want,):
        raise ValueError(
            f"a continuous action is the flat (..., {want}) float vector, got shape "
            f"{tuple(action.shape)}"
        )
    if not jnp.issubdtype(action.dtype, jnp.floating):
        raise ValueError(
            f"a continuous action is the flat (..., {want}) float vector, got dtype "
            f"{action.dtype}; this System's controls {list(keys)} take float blocks"
        )
    return {key: names.layout.node_block(k, action) for k, key in enumerate(keys)}


def actions_in(names: FaceNames, mapping: Any, face: str, lead_shape: tuple | None = None):
    """``{node: value}`` as the positional action: one column per control node.

    Discrete controls give an int32 ``(..., K)`` array in node order; continuous ones the flat
    float ``(..., action_dim)`` vector (the nodes' values, in node order); a mixed space a
    :class:`~hybrid_aif.controls.ControlStep`, one entry per node. Leading axes are read off
    the entries; ``lead_shape`` sizes them when there is no control node (default ``()``). A
    level given as a non-integer dtype, and a continuous value of the wrong width, raise (both
    are static, so the checks run at trace time).
    """
    check_mapping(names, "actions", mapping, face)
    keys = names.actions
    values = [jnp.asarray(mapping[k]) for k in keys]
    _check_levels(names, keys, values, face)
    _check_continuous(names, keys, values, face)
    if names.layout is not None:
        return jnp.concatenate(values, axis=-1)
    if names.mixed:
        return ControlStep(
            entries=tuple(
                v.astype(jnp.int32) if names.kind(k) == LEVEL else v for k, v in enumerate(values)
            )
        )
    cols = [v.astype(jnp.int32) for v in values]
    if not cols:
        return jnp.zeros((*(lead_shape or ()), 0), jnp.int32)
    shape = jnp.broadcast_shapes(*(c.shape for c in cols))
    return jnp.stack([jnp.broadcast_to(c, shape) for c in cols], axis=-1)


def actions_out(names: FaceNames, action) -> dict:
    """The positional action (one column per control node, the flat continuous vector, or a
    mixed space's :class:`~hybrid_aif.controls.ControlStep`) as ``{node: value}``."""
    keys = names.actions
    if isinstance(action, ControlStep):
        return dict(zip(keys, action.entries, strict=True))
    action = jnp.asarray(action)
    if names.layout is not None:
        return _split_flat(names, action)
    if action.shape[-1:] != (len(keys),):
        raise ValueError(
            f"a positional action has one level per control node, (..., {len(keys)}) for "
            f"{list(keys)}; got shape {tuple(action.shape)}"
        )
    if not jnp.issubdtype(action.dtype, jnp.integer):
        hint = _params_hint(names)
        raise ValueError(
            f"actions {list(keys)} are not integer (positional dtype {action.dtype}); a discrete "
            f"control's action is a level index (an integer array){hint}"
        )
    return {k: action[..., i] for i, k in enumerate(keys)}


# ------------------------------------------------------------------------------------- detail

_PER_MODALITY = ("pre_reduce", "mode_log_evidence", "log_evidence")


def detail_in(names: FaceNames, detail: Any, face: str) -> InferenceDetail:
    """A keyed :class:`InferenceDetail` as the positional one the internals read."""
    if not isinstance(detail, InferenceDetail):
        raise TypeError(f"{face} takes detail as an InferenceDetail, got {type(detail).__name__}")
    fields = {f: getattr(detail, f) for f in (*_PER_MODALITY, "unsupported")}
    if not all(isinstance(v, Mapping) for v in fields.values()):
        raise TypeError(
            f"{face} takes detail keyed by name (its per-modality fields by modality, "
            "unsupported by factor), as infer_states(detail=True) returns it; convert "
            "positional data with system.from_positional('detail', ...)"
        )
    out = {}
    for f in _PER_MODALITY:
        check_mapping(names, "observations", fields[f], face, label=f"detail.{f}")
        out[f] = tuple(fields[f][n] for n in names.modalities)
    unsupported = fields["unsupported"]
    if unsupported:
        check_mapping(names, "beliefs", unsupported, face, label="detail.unsupported")
        out["unsupported"] = tuple(unsupported[n] for n in names.factors)
    else:
        out["unsupported"] = ()
    out["free_energy"] = detail.free_energy
    return InferenceDetail(**out)


def detail_out(names: FaceNames, detail: InferenceDetail) -> InferenceDetail:
    """The positional :class:`InferenceDetail` keyed: per-modality fields by modality name,
    ``unsupported`` by factor name (empty where the path flags nothing)."""
    out = {f: observations_out(names, getattr(detail, f)) for f in _PER_MODALITY}
    out["unsupported"] = beliefs_out(names, detail.unsupported) if detail.unsupported else {}
    out["free_energy"] = detail.free_energy
    return InferenceDetail(**out)


# ------------------------------------------------------------------------------------- params

#: The per-variable containers of ``SystemParams`` (and of an environment's parameters).
SLOTS = ("likelihood", "transition", "initial_state_prior")


def _slot_names(names: FaceNames, slot: str) -> tuple[str, ...]:
    return names.modalities if slot == "likelihood" else names.factors


def _keyed_slots(params: Any) -> list[str]:
    """The slots held as non-empty mappings (an empty container is of either form)."""
    return [slot for slot in SLOTS if len(c := getattr(params, slot)) and isinstance(c, Mapping)]


def params_positional(names: FaceNames, params: Any, face: str = "params") -> Any:
    """``params`` with its per-variable containers as tuples in node order (the internal form).

    Idempotent: positional containers pass through. Keyed ones are read by name, so their
    insertion order never matters; a missing or unknown key raises.
    """
    out = {}
    for slot in SLOTS:
        container = getattr(params, slot)
        if not isinstance(container, Mapping):
            continue
        kind = "observations" if slot == "likelihood" else "beliefs"
        nodes = [n for n in names.controls if n in container]
        if nodes and slot != "likelihood":
            raise NotImplementedError(
                f"{face}: params.{slot} has an entry for control node(s) {nodes}, and a prior "
                "over a control node arrives with the path-dynamics build (docs/design/"
                "control-states.md C2); leave it out, so each node is uniform over its levels "
                "at every step"
            )
        check_mapping(names, kind, container, face, label=f"params.{slot}")
        out[slot] = tuple(container[n] for n in _slot_names(names, slot))
    return dataclasses.replace(params, **out) if out else params


def params_keyed(names: FaceNames, params: Any) -> Any:
    """``params`` with its per-variable containers keyed by name, in node order. Idempotent."""
    out = {}
    for slot in SLOTS:
        container = getattr(params, slot)
        if not isinstance(container, Mapping):
            out[slot] = dict(zip(_slot_names(names, slot), tuple(container), strict=True))
    return dataclasses.replace(params, **out) if out else params


def params_in(names: FaceNames, params: Any, face: str) -> Any:
    """Face-side check and conversion: keyed containers in, the positional view out.

    A positional container (a non-empty tuple or list) raises, naming the converter; an empty
    one is accepted where no name is expected.
    """
    for slot in SLOTS:
        container = getattr(params, slot)
        if isinstance(container, Mapping):
            continue
        if len(tuple(container)) == 0 and not _slot_names(names, slot):
            continue
        raise TypeError(
            f"{face} takes params.{slot} as a mapping keyed by "
            f"{'modality' if slot == 'likelihood' else 'factor'} name "
            f"({list(_slot_names(names, slot))}); convert positional containers with "
            "system.from_positional('params', params)"
        )
    return params_positional(names, params, face)


def to_positional(names: FaceNames, kind: str, mapping: Any):
    """The positional form of a name-keyed ``mapping`` of ``kind`` (see module docstring)."""
    face = f"to_positional({kind!r}, ...)"
    if kind == "params":
        return params_in(names, mapping, face)
    if kind == "detail":
        return detail_in(names, mapping, face)
    if kind == "beliefs":
        return beliefs_in(names, mapping, face)
    if kind == "observations":
        return observations_in(names, mapping, face)
    if kind == "actions":
        return actions_in(names, mapping, face)
    raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


def from_positional(names: FaceNames, kind: str, positional: Any) -> dict:
    """The name-keyed form of positional data of ``kind`` (see module docstring)."""
    if kind == "params":
        keyed_slots = _keyed_slots(positional)
        if keyed_slots:
            raise TypeError(
                f"from_positional('params', ...) takes positional containers; "
                f"{['params.' + s for s in keyed_slots]} are already mappings"
            )
        for slot in SLOTS:
            want = len(_slot_names(names, slot))
            got = len(tuple(getattr(positional, slot)))
            if got != want:
                raise ValueError(
                    f"from_positional('params', ...): params.{slot} has {got} entries for "
                    f"{want} name(s) {list(_slot_names(names, slot))}"
                )
        return params_keyed(names, positional)
    if kind == "detail":
        if not isinstance(positional, InferenceDetail) or isinstance(
            positional.pre_reduce, Mapping
        ):
            raise TypeError(
                "from_positional('detail', ...) takes a positional InferenceDetail (tuples in "
                "modality and factor order)"
            )
        return detail_out(names, positional)
    if kind == "beliefs":
        return beliefs_out(names, _check_length(names, kind, positional, len(names.factors)))
    if kind == "observations":
        return observations_out(
            names, _check_length(names, kind, positional, len(names.modalities))
        )
    if kind == "actions":
        if isinstance(positional, Mapping):
            raise TypeError(
                "from_positional('actions', ...) takes positional data; this is already a mapping"
            )
        return actions_out(names, positional)
    raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
