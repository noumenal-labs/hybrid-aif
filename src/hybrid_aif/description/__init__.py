"""A ``System`` and its parameters as a document: :func:`describe`, :func:`build`, :func:`schema`.

The document is plain JSON, and its specification is ``docs/description-format.md``. It names
every variable once under ``variables``, with a domain and a size; writes each conditional
distribution as a factor record keyed by its child variable and a label under ``factors``; holds
the relations no single factor owns (``controls``, ``observed``, ``external``) under
``relations``; gives the value of every fixed parameter variable under ``clamps``; writes the
planning objective as a list of terms under ``objective``; and the System's settings beyond the
model under ``configuration``.
Per-step and static variables are told apart by use, never marked: a variable named in a factor's
``parameters`` is static, a child or parent of a per-step factor is per-step.

A clamp is an inline value, a ``{"ref": "<file>#<key>"}`` into the arrays file, or an
``{"init": {...}}`` constructor recipe drawn at build with a key. :func:`describe` writes an array
inline up to ``inline_max`` elements and into the arrays file above it; the arrays file is a
``dict[str, ndarray]`` that :func:`save` writes as an ``.npz`` beside the ``.json``. :func:`build`
validates the document in full, each error naming the address and what to write instead, then runs
``System._validate``. Not in the document: ``LearnState``, beliefs, policies, replay, the planner's
and the learner's call-time arguments, numeric precision.

Design: docs/design/model-description-0.3.md; docs/design/model-description.md M4, M5, M7, M11.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import json
import math
import pathlib
import re
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hybrid_aif import distributions as _dist
from hybrid_aif import factors as _factors
from hybrid_aif.controls import FLOAT, Box, action_name, control_name, declared_parents
from hybrid_aif.inference.discrete import SUPPORT_FLOOR, _as_log_likelihood
from hybrid_aif.objectives import registry as _registry
from hybrid_aif.objectives.base import INFOGAIN
from hybrid_aif.objectives.closed_loop import FEEDBACK
from hybrid_aif.planning.base import planner_branches
from hybrid_aif.system import _KEY_KINDS, System, SystemParams
from hybrid_aif.system.core import Subsystem
from hybrid_aif.system.validation import _check_dim_names

VERSION = "0.3"
KEY = "hybrid_aif_description"
EXTENSIONS = "extensions"
SCHEMA_ID = f"https://noumenal.ai/schemas/hybrid-aif/description-{VERSION}.json"

#: The top-level keys a document may carry. Anything else is refused by name: a consumer's own
#: data rides in ``extensions``, keyed by the consumer, and nowhere else.
TOP_LEVEL_KEYS = frozenset(
    {
        KEY,
        "library_version",
        "variables",
        "factors",
        "relations",
        "clamps",
        "objective",
        "configuration",
        "arrays_file",
        EXTENSIONS,
    }
)

__all__ = [
    "ParameterRecord",
    "describe_parameters",
    "EXTENSIONS",
    "FAMILIES",
    "KEY",
    "LABELS",
    "TOP_LEVEL_KEYS",
    "VERSION",
    "Description",
    "DescriptionError",
    "build",
    "check_joint_support",
    "check_observation_support",
    "describe",
    "load",
    "register_transition",
    "registered_transitions",
    "save",
    "schema",
]


class DescriptionError(ValueError):
    """A document that does not describe a ``System``, with the address and the repair."""


class Description(NamedTuple):
    """What :func:`describe` returns: the JSON document and its arrays file.

    ``arrays`` is keyed by the key part of the document's ``ref`` strings, ready for
    ``np.savez``; empty when every clamp went inline.
    """

    document: dict[str, Any]
    arrays: dict[str, np.ndarray]


# ================================================================== the vocabulary

#: The role labels, by partition. A label that is none of these is a generic label: the factor's
#: own name, under which every parent is bracketed. Design: model-description-0.3.md Q17, Q18.5.
GENERATIVE_LABELS = ("initial", "transition", "likelihood", "actuator", "prior", "policy_prior")
OBJECTIVE_LABELS = ("target", "terminal")
Q_LABELS = ("recognition",)
LABELS = GENERATIVE_LABELS + OBJECTIVE_LABELS + Q_LABELS

#: The steps a record applies at, and the label-implied offset (0: the same step, 1: the step
#: before) and steps of the four labels with a time structure.
STEPS = ("0", "1+", "all")
_IMPLIED = {
    "initial": (0, "0"),
    "transition": (1, "1+"),
    "likelihood": (0, "all"),
    "actuator": (0, "all"),
    "recognition": (0, "all"),
}
#: Labels whose time is plan time (``steps`` does not apply) or that carry no time at all.
_UNTIMED = ("target", "terminal", "policy_prior", "prior")

PER_STEP_DOMAINS = ("categorical", "real")
PARAMETER_DOMAINS = (
    "log_simplex",
    "simplex",
    "logits",
    "positive",
    "nonnegative",
    "cholesky",
    "spd",
    "unit_interval",
    "integer_levels",
    "real",
)

#: The written form of an offset on a parent name: ``x[t]`` (the child's step) or ``x[t-1]``.
_BRACKET = re.compile(r"^(?P<name>[^\[\]]*)(?:\[(?P<offset>[^\[\]]*)\])?$")
_OFFSETS = {"t": 0, "t-1": 1}
_SEP = "."

#: The default per-step and parameter-variable letters of a tabular record under a role label.
_LETTERS = {
    "initial": "D",
    "transition": "B",
    "likelihood": "A",
    "target": "C",
    "terminal": "H",
    "actuator": "K",
}


def _library_version() -> str:
    import hybrid_aif  # at call time: the package is initialised by then

    return hybrid_aif.__version__


# ================================================================== the family menu


class _Family(NamedTuple):
    """How one family is written: its parameter keys (the constructor's field names, a nested
    module as a nested record), which of them may be absent, the key holding a network (a tree
    variable), its options with their defaults, and the labels it may appear under."""

    cls: type
    parameters: tuple[str, ...]
    optional: tuple[str, ...] = ()
    nested: Mapping[str, tuple[str, ...]] = {}
    network: str | None = None
    options: Mapping[str, object] = {}
    labels: tuple[str, ...] = ()
    mapped: Mapping[
        str, tuple[str, ...]
    ] = {}  # a key that may be a nested record instead of a name
    reads_self: bool = False  # a transition reading its own previous value, listed first
    reads_same_step: bool = False  # a transition reading its other latents at the child's step


_TABULAR_LABELS = ("initial", "transition", "likelihood", "actuator", "terminal")
_MLP_OPTIONS = {
    "activation": "tanh",
    "min_scale": 1e-3,
    "head": "softplus",
    "minstd": 0.1,
    "maxstd": 1.0,
    "norm": False,
}
_TRUNK_OPTIONS = {
    "activation": "silu",
    "norm": True,
    "out_activation": False,
    "symlog_inputs": False,
}

#: Every family a document may name, by class name, with how it is written. Design:
#: model-description-0.3.md Q18.3.
FAMILIES: dict[str, _Family] = {
    "Categorical": _Family(
        _dist.Categorical, ("logits",), labels=("initial", "target", "terminal", "policy_prior")
    ),
    "ConditionalCategorical": _Family(
        _dist.ConditionalCategorical, ("log_table",), labels=(*_TABULAR_LABELS, "target")
    ),
    "JointCategorical": _Family(_dist.JointCategorical, ("logits",), labels=("target", "terminal")),
    "Gaussian": _Family(_dist.Gaussian, ("loc", "chol"), labels=("initial", "target", "terminal")),
    "ConditionalGaussian": _Family(_dist.ConditionalGaussian, ("loc", "chol"), labels=("target",)),
    "MixtureGaussian": _Family(
        _dist.MixtureGaussian,
        ("log_weights", "means", "chols"),
        labels=("initial", "target", "terminal"),
    ),
    "LinearGaussianTransition": _Family(
        _factors.LinearGaussianTransition,
        ("F", "innovations", "G"),
        optional=("G",),
        nested={"innovations": ("loc", "chol")},
        labels=("transition",),
        reads_self=True,
        reads_same_step=True,
    ),
    "NonlinearGaussianTransition": _Family(
        _factors.NonlinearGaussianTransition,
        ("f", "innovations"),
        optional=("f",),
        nested={"innovations": ("loc", "chol")},
        options={"f": None, "propagation": "first_order"},
        labels=("transition",),
        reads_self=True,
    ),
    "LinearGaussianObservation": _Family(
        _factors.LinearGaussianObservation,
        ("head", "noise"),
        labels=("likelihood",),
        options={"noise_map": None},
        mapped={"noise": ("values",)},
    ),
    "MLPConditional": _Family(
        _dist.MLPConditional,
        ("layers",),
        network="layers",
        options=_MLP_OPTIONS,
        labels=("transition", "likelihood", "recognition"),
        reads_self=True,
    ),
    "MLPPredictor": _Family(
        _dist.MLPPredictor,
        ("layers",),
        network="layers",
        options=_TRUNK_OPTIONS,
        labels=("recognition",),
    ),
    "MLPCategorical": _Family(
        _dist.MLPCategorical,
        ("trunk",),
        network="trunk",
        options={"unimix": 0.01, **_TRUNK_OPTIONS},
        labels=("transition", "recognition"),
        reads_same_step=True,
    ),
    "PredictorHead": _Family(
        _dist.PredictorHead,
        ("trunk",),
        network="trunk",
        options={
            "loss": None,
            "num_bins": 255,
            "bin_low": -20.0,
            "bin_high": 20.0,
            **_TRUNK_OPTIONS,
        },
        labels=("likelihood",),
    ),
    "Deterministic": _Family(
        _factors.Deterministic,
        ("map",),
        optional=("map",),
        network="map",
        options={"rule": "carry", "map": "GRUCell", "activation": "silu", "norm": True},
        labels=("transition",),
        reads_self=True,
    ),
    "LearnedVector": _Family(
        _dist.LearnedVector, ("value",), options={"activation": "tanh"}, labels=("initial",)
    ),
    "DerivedInitial": _Family(
        _factors.DerivedInitial, (), options={"reduction": "mode"}, labels=("initial",)
    ),
    "Dirichlet": _Family(
        _dist.Dirichlet,
        ("concentration", "alpha0", "growth"),
        optional=("alpha0", "growth"),
        labels=("prior",),
    ),
    "InverseGamma": _Family(
        _dist.InverseGamma,
        ("alpha", "beta", "alpha0", "beta0"),
        optional=("alpha0", "beta0"),
        labels=("prior",),
    ),
    "InverseWishart": _Family(
        _dist.InverseWishart,
        ("Psi", "nu", "Psi0", "nu0"),
        optional=("Psi0", "nu0"),
        labels=("prior",),
    ),
    "MatrixNormal": _Family(
        _dist.MatrixNormal,
        ("M", "V_inv", "M0", "V_inv0", "row_cov"),
        optional=("M0", "V_inv0", "row_cov"),
        labels=("prior",),
    ),
}


# ======================================================== the transition registry


class _Registered(NamedTuple):
    """A registered transition function: its class, and its field schema, each field's name
    with whether it is static (``True``) or a pytree leaf (``False``), in declaration order."""

    cls: type
    fields: tuple[tuple[str, bool], ...]


#: The registered transition functions, by name. Design: nonlinear-transitions.md N5.
_TRANSITIONS: dict[str, _Registered] = {}
_NAMESPACED = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+$")


def register_transition(name: str, cls: type) -> type:
    """Register ``cls``, the class of a :class:`~hybrid_aif.NonlinearGaussianTransition`'s
    ``f``, under ``name``, so that :func:`describe` writes such an ``f`` as its name and its
    fields and :func:`build` rebuilds it.

    ``name`` is namespaced, ``"<package>.<name>"`` (at least one dot, each segment an
    identifier), so two packages cannot collide silently. ``cls`` is a callable
    :class:`equinox.Module` subclass. The registration records ``cls``'s field schema: each
    field's name and whether it is static (``eqx.field(static=True)``) or a leaf; a leaf's shape
    is an instance's, not the class's, and is checked when the built System is validated. A name
    is registered once and a class under one name: registering a taken name, or a class already
    registered, raises :class:`ValueError`. Returns ``cls``.
    Design: docs/design/nonlinear-transitions.md N5.
    """
    if not isinstance(name, str) or not _NAMESPACED.match(name):
        raise ValueError(
            f"register_transition: the name {name!r} is not namespaced; write "
            "'<package>.<name>' (at least one dot, each segment an identifier), e.g. "
            "'drive.unicycle', so two packages cannot register one name"
        )
    if not isinstance(cls, type) or not issubclass(cls, eqx.Module):
        raise TypeError(
            f"register_transition({name!r}, ...): a transition function is an equinox.Module "
            f"subclass, got {cls!r}; register the class of the f you pass "
            "NonlinearGaussianTransition"
        )
    if not any("__call__" in vars(k) for k in cls.__mro__ if k is not object):
        raise TypeError(
            f"register_transition({name!r}, {cls.__qualname__}): the class defines no __call__; "
            "f is called as f(x, controls)"
        )
    if name in _TRANSITIONS:
        raise ValueError(
            f"register_transition: {name!r} is already registered to "
            f"{_TRANSITIONS[name].cls.__module__}.{_TRANSITIONS[name].cls.__qualname__}; a "
            "name is registered once (choose another name under your package's namespace)"
        )
    taken = [n for n, entry in _TRANSITIONS.items() if entry.cls is cls]
    if taken:
        raise ValueError(
            f"register_transition: {cls.__qualname__} is already registered as {taken[0]!r}; a "
            "class is registered under one name, the one describe writes"
        )
    fields = tuple((f.name, bool(f.metadata.get("static", False))) for f in dataclasses.fields(cls))
    _TRANSITIONS[name] = _Registered(cls, fields)
    return cls


def registered_transitions() -> dict[str, type]:
    """The registered transition functions, ``{name: class}``, in registration order."""
    return {name: entry.cls for name, entry in _TRANSITIONS.items()}


def _schema_text(fields: tuple[tuple[str, bool], ...]) -> str:
    leaves = [n for n, static in fields if not static]
    static = [n for n, s in fields if s]
    return f"leaves {leaves}, static {static}"


#: The families a parametric term's parameters may hold, with the options a term record writes
#: for them (configuration and, since a term parameter has no wiring, its layout).
_TERM_FAMILIES = ("MLPConditional", "MLPPredictor", "PredictorHead")
_TERM_LAYOUT = {
    "MLPConditional": ("parent_dims", "child_dim"),
    "MLPPredictor": ("parent_dims", "out_dim"),
    "PredictorHead": ("parent_dims", "out_dim"),
}

#: The constructor-recipe arguments an ``{"init": {...}}`` clamp may give, per family: what a
#: draw needs beyond the wiring and the options.
_INIT_ARGS = {
    "MLPConditional": ("width", "depth", "init_scale", "init"),
    "MLPPredictor": ("width", "depth", "out_scale", "init"),
    "MLPCategorical": ("width", "depth", "init"),
    "PredictorHead": ("width", "depth", "out_scale", "init"),
    "Deterministic": ("hidden", "init"),
    "LearnedVector": (),
}
#: The recipe arguments a draw cannot do without.
_INIT_REQUIRED = {
    "MLPPredictor": ("width",),
    "MLPCategorical": ("width",),
    "PredictorHead": ("width",),
    "Deterministic": ("hidden",),
}


def _head_width(loss: str, num_bins: int, vector: int) -> int:
    """How many numbers a ``PredictorHead``'s trunk emits: the vector's width under
    ``symlog_mse``, one logit per bin under ``symlog_disc``, one logit under ``binary``."""
    return vector if loss == "symlog_mse" else (int(num_bins) if loss == "symlog_disc" else 1)


#: The domain of each parameter field, written by the family (M6, moved to the variable by D4).
#: Anything not listed is ``"real"``.
_DOMAINS: dict[tuple[str, str], str] = {
    ("Categorical", "logits"): "logits",
    ("JointCategorical", "logits"): "logits",
    ("ConditionalCategorical", "log_table"): "log_simplex",
    ("MixtureGaussian", "log_weights"): "log_simplex",
    ("Gaussian", "chol"): "cholesky",
    ("ConditionalGaussian", "chol"): "cholesky",
    ("MixtureGaussian", "chols"): "cholesky",
    ("LinearGaussianTransition", "innovations.chol"): "cholesky",
    ("LinearGaussianObservation", "noise"): "cholesky",
    ("LinearGaussianObservation", "noise.values"): "cholesky",
    ("Dirichlet", "concentration"): "positive",
    ("Dirichlet", "alpha0"): "positive",
    ("Dirichlet", "growth"): "positive",
    ("InverseGamma", "alpha"): "positive",
    ("InverseGamma", "beta"): "positive",
    ("InverseGamma", "alpha0"): "positive",
    ("InverseGamma", "beta0"): "positive",
    ("InverseWishart", "Psi"): "spd",
    ("InverseWishart", "Psi0"): "spd",
    ("InverseWishart", "nu"): "positive",
    ("InverseWishart", "nu0"): "positive",
    ("MatrixNormal", "V_inv"): "spd",
    ("MatrixNormal", "V_inv0"): "spd",
}

#: A domain a field also admits: normalised log-probabilities are logits, and probabilities
#: (``simplex``) are lowered to their logs on read.
_TIGHTER = {"logits": ("log_simplex", "simplex"), "log_simplex": ("simplex",)}
#: A looser domain a field admits because its family normalises it: a table's unnormalised log
#: values (the family normalises over the child axis), which :func:`describe` writes as logits.
_LOOSER = {
    ("ConditionalCategorical", "log_table"): ("logits",),
    # a zero is a level on the simplex's face: a structural zero, or a growable factor's dead
    # level (structure-learning.md S6)
    ("Dirichlet", "concentration"): ("nonnegative",),
}


def _domain(family: str, field: str) -> str:
    return _DOMAINS.get((family, field), "real")


#: The families that put a model on the amortized path.
_AMORTIZED_FAMILIES = (
    "Deterministic",
    "MLPCategorical",
    "PredictorHead",
    "LearnedVector",
    "DerivedInitial",
)

# ======================================================================= describe

#: The arrays file :func:`describe` names; :func:`save` renames it after the document.
ARRAYS_FILE = "arrays.npz"


def _qualified(child: str, leaf: Callable[[str], str]) -> str:
    """A default name built from ``child``'s own name, kept under ``child``'s composite prefix
    (``world.position`` and ``A_<child>`` give ``world.A_position``)."""
    space, sep, name = child.rpartition(_SEP)
    return f"{space}{sep}{leaf(name)}"


def _constant(arr: np.ndarray) -> bool:
    """Whether every element equals the first (a clamp that a scalar broadcasts to)."""
    first = arr.flat[0]
    return bool(np.all(arr == first)) and not (isinstance(first, float) and math.isnan(first))


class _Writer:
    """The document under construction, its placement policy and the arrays it accumulates.

    ``slots`` maps each engine declaration, ``(slot, name)``, to the variables it was written with
    (its main parameter variable and, for a linear-Gaussian observation, its ``head`` and
    ``noise``), which the objective reads to write a parameter variable as a term's ``on``.
    """

    def __init__(self, mode: str, inline_max: int, key_prefix: str = "") -> None:
        if mode not in ("auto", "inline", "file"):
            raise ValueError(f"arrays= must be 'auto', 'inline' or 'file', got {mode!r}")
        self.mode, self.inline_max, self.key_prefix = mode, int(inline_max), key_prefix
        self.arrays: dict[str, np.ndarray] = {}
        self.variables: dict[str, Any] = {}
        self.factors: dict[str, dict[str, Any]] = {}
        self.clamps: dict[str, Any] = {}
        self.frozen: list[str] = []
        self.slots: dict[tuple[str, str], dict[str, str]] = {}

    def nested(self, prefix: str) -> _Writer:
        """A writer for a term's own model, sharing this one's arrays file under ``prefix/``."""
        inner = _Writer(self.mode, self.inline_max, f"{self.key_prefix}{prefix}/")
        inner.arrays = self.arrays
        return inner

    def _claim(self, name: str, address: str) -> str:
        if name in self.variables:
            raise DescriptionError(
                f"{address}: the default name {name!r} is already a variable of this document; "
                "rename the variable that holds it in the System, or describe the System with "
                "names that leave the default free"
            )
        return name

    def value(self, arr: object, key: str) -> Any:
        """One array as a clamp: a scalar when it is constant, inline up to the threshold, else a
        ref into the arrays file under ``key``."""
        arr = np.asarray(arr)
        if self.mode != "file" and arr.size > 1 and _constant(arr):
            return arr.flat[0].item()
        if self.mode == "inline" or (self.mode == "auto" and arr.size <= self.inline_max):
            return arr.tolist()
        full = f"{self.key_prefix}{key}"
        self.arrays[full] = arr
        return {"ref": f"{ARRAYS_FILE}#{full}"}

    def param(self, name: str, value: object, domain: str, address: str) -> str:
        """Declare a clamped parameter variable."""
        self._claim(name, address)
        arr = np.asarray(value)
        self.variables[name] = {"domain": domain, "shape": list(arr.shape)}
        self.clamps[name] = self.value(arr, name)
        return name

    def learned(self, name: str, shape: tuple[int, ...], domain: str, address: str) -> str:
        """Declare a parameter variable with no clamp (its prior record follows)."""
        self._claim(name, address)
        self.variables[name] = {"domain": domain, "shape": [int(s) for s in shape]}
        return name

    def tree(self, name: str, tree: Any, address: str) -> str:
        """Declare a clamped tree variable: one leaf record and one clamp per array."""
        self._claim(name, address)

        def records(node: Any) -> Any:
            if isinstance(node, Mapping):
                return {k: records(v) for k, v in node.items()}
            if isinstance(node, list):
                return [records(v) for v in node]
            return {"domain": "real", "shape": list(np.shape(node))}

        def clamps(node: Any, path: str) -> Any:
            if isinstance(node, Mapping):
                return {k: clamps(v, f"{path}/{k}") for k, v in node.items()}
            if isinstance(node, list):
                return [clamps(v, f"{path}/{i}") for i, v in enumerate(node)]
            return self.value(node, path)

        self.variables[name] = records(tree)
        self.clamps[name] = clamps(tree, name)
        return name

    def record(self, child: str, label: str, record: dict[str, Any]) -> None:
        self.factors.setdefault(child, {})[label] = record

    def prior(self, var: str, family: str, fields: Mapping[str, object], address: str) -> None:
        """A ``prior`` record under ``var``: each array hyperparameter a clamped variable named
        ``<var>_<field>``; a string (``row_cov``) names a variable as it stands."""
        parameters: dict[str, str] = {}
        for field, value in fields.items():
            if value is None:
                continue
            if isinstance(value, str):
                parameters[field] = value
                continue
            domain = _domain(family, field)
            if (family, field) == ("Dirichlet", "concentration") and np.any(
                np.asarray(value) == 0.0
            ):
                domain = "nonnegative"  # a level on the simplex's face (S6)
            parameters[field] = self.param(f"{var}_{field}", value, domain, f"{address}/{field}")
        self.record(var, "prior", {"family": family, "parameters": parameters})


def _record(family: str, parents=(), parameters=None, options=None) -> dict[str, Any]:
    out: dict[str, Any] = {"family": family}
    if parents:
        out["parents"] = list(parents)
    if parameters:
        out["parameters"] = parameters
    if options:
        out["options"] = options
    return out


def _linear_tree(layer: eqx.nn.Linear) -> dict[str, Any]:
    return {"weight": layer.weight, **({"bias": layer.bias} if layer.bias is not None else {})}


def _norm_tree(norm: eqx.nn.LayerNorm, address: str) -> dict[str, Any]:
    if norm.weight is None or norm.eps != 1e-3:
        raise DescriptionError(
            f"{address}: a LayerNorm with eps={norm.eps} and weight={norm.weight is not None} is "
            "not one the network families build (eps 1e-3, with a weight), so its document "
            "would build a different network; construct the network with its family's "
            "from_shapes"
        )
    return {"weight": norm.weight, **({"bias": norm.bias} if norm.bias is not None else {})}


def _mlp_tree(module, address: str) -> dict[str, Any]:
    tree: dict[str, Any] = {"layers": [_linear_tree(layer) for layer in module.layers]}
    if module.norms:
        tree["norms"] = [_norm_tree(n, f"{address}/norms/{i}") for i, n in enumerate(module.norms)]
    return tree


def _trunk_options(trunk) -> dict[str, Any]:
    return {
        "activation": trunk.activation,
        "norm": bool(trunk.norms),
        "out_activation": bool(trunk.out_activation),
        "symlog_inputs": bool(trunk.symlog_inputs),
    }


def _network(w: _Writer, module, var: str, address: str) -> tuple[str, dict[str, Any]]:
    """A network family's tree variable and its options; a frozen one joins
    ``learning.frozen``. Returns ``(parameter key, options)``."""
    if isinstance(module, _dist.MLPConditional):
        w.tree(var, _mlp_tree(module, address), address)
        options = {
            "activation": module.activation,
            "min_scale": module.min_scale,
            "head": module.head,
            "minstd": module.minstd,
            "maxstd": module.maxstd,
            "norm": bool(module.norms),
        }
        frozen, key = module.frozen, "layers"
    elif isinstance(module, _dist.MLPPredictor):
        w.tree(var, _mlp_tree(module, address), address)
        options, frozen, key = _trunk_options(module), module.frozen, "layers"
    elif isinstance(module, _dist.MLPCategorical):
        w.tree(var, _mlp_tree(module.trunk, address), address)
        options = {"unimix": module.unimix, **_trunk_options(module.trunk)}
        frozen, key = module.trunk.frozen, "trunk"
    else:  # a PredictorHead: the callers pass the four network families only
        w.tree(var, _mlp_tree(module.trunk, address), address)
        options = {
            "loss": module.loss,
            "num_bins": module.num_bins,
            "bin_low": module.bin_low,
            "bin_high": module.bin_high,
            **_trunk_options(module.trunk),
        }
        frozen, key = module.trunk.frozen, "trunk"
    if frozen:
        w.frozen.append(var)
    return key, options


_NETWORKS = (_dist.MLPConditional, _dist.MLPPredictor, _dist.MLPCategorical, _dist.PredictorHead)


def _encode(
    w: _Writer, decl: object, base: str, main: str | None, address: str, slot: tuple | None = None
) -> dict[str, Any]:
    """The factor record (``family``, ``parameters``, ``options``) for one engine declaration.

    ``base`` prefixes the default names (``<base>_<field>``); ``main``, when given, names the
    tabular table (an AIF letter). ``slot`` records the variables against the engine address.
    Parents are the caller's.
    """
    kind = type(decl)

    def field(name: str, value: object, family: str, key: str | None = None) -> str:
        return w.param(
            f"{base}_{key or name}", value, _domain(family, name), f"{address}/{key or name}"
        )

    def remember(**variables: str) -> None:
        if slot is not None:
            w.slots[slot] = variables

    if kind is _dist.ConditionalDirichlet:
        _check_layout(decl.parent_ndims == decl.concentration.ndim - 1, address, "parent_ndims")
        var = w.learned(
            main or f"{base}_log_table", decl.concentration.shape, "log_simplex", address
        )
        w.prior(
            var,
            "Dirichlet",
            {"concentration": decl.concentration, "alpha0": decl.alpha0, "growth": decl.growth},
            address,
        )
        remember(main=var)
        return _record("ConditionalCategorical", parameters={"log_table": var})
    if kind is _dist.ConditionalCategorical:
        _check_layout(decl.parent_ndims == decl.log_table.ndim - 1, address, "parent_ndims")
        domain = "log_simplex" if _normalised(np.asarray(decl.log_table), 0) else "logits"
        var = w.param(main or f"{base}_log_table", decl.log_table, domain, address)
        remember(main=var)
        return _record("ConditionalCategorical", parameters={"log_table": var})
    if kind is _dist.Dirichlet:
        var = w.learned(main or f"{base}_logits", decl.concentration.shape, "logits", address)
        w.prior(
            var,
            "Dirichlet",
            {"concentration": decl.concentration, "alpha0": decl.alpha0, "growth": decl.growth},
            address,
        )
        remember(main=var)
        return _record("Categorical", parameters={"logits": var})
    if kind is _dist.Categorical:
        var = w.param(main or f"{base}_logits", decl.logits, "logits", address)
        remember(main=var)
        return _record("Categorical", parameters={"logits": var})
    if kind is _dist.JointCategorical:
        _check_layout(decl.event_ndims == decl.logits.ndim, address, "event_ndims")
        var = w.param(main or f"{base}_logits", decl.logits, "logits", address)
        return _record("JointCategorical", parameters={"logits": var})
    if kind in (_dist.Gaussian, _dist.MixtureGaussian):
        family = kind.__name__
        names = FAMILIES[family].parameters
        return _record(family, parameters={n: field(n, getattr(decl, n), family) for n in names})
    if kind is _dist.ConditionalGaussian:
        _check_layout(decl.parent_ndims == decl.loc.ndim - 1, address, "parent_ndims")
        family = kind.__name__
        names = FAMILIES[family].parameters
        return _record(family, parameters={n: field(n, getattr(decl, n), family) for n in names})
    if kind is _factors.LinearGaussianTransition:
        inn = decl.innovations
        _check_layout(
            inn.parent_ndims == inn.loc.ndim - 1, f"{address}/innovations", "parent_ndims"
        )
        parameters: dict[str, Any] = {
            "F": field("F", decl.F, "LinearGaussianTransition"),
            "innovations": {
                "loc": w.param(
                    f"{base}_innovations_loc", inn.loc, "real", f"{address}/innovations"
                ),
                "chol": w.param(
                    f"{base}_innovations_chol", inn.chol, "cholesky", f"{address}/innovations"
                ),
            },
        }
        if decl.G is not None:
            parameters["G"] = field("G", decl.G, "LinearGaussianTransition")
        return _record("LinearGaussianTransition", parameters=parameters)
    if kind is _factors.NonlinearGaussianTransition:
        return _encode_nonlinear(w, decl, base, address)
    if kind in (_factors.LinearGaussianObservation, _factors.LinearGaussianObservationMNIW):
        return _encode_lg_observation(w, decl, base, address, remember)
    if kind is _factors.Deterministic:
        return _encode_deterministic(w, decl, base, address, remember)
    if kind is _dist.LearnedVector:
        var = field("value", decl.value, "LearnedVector")
        if decl.frozen:
            w.frozen.append(var)
        remember(main=var)
        return _record(
            "LearnedVector", parameters={"value": var}, options={"activation": decl.activation}
        )
    if kind is _factors.DerivedInitial:
        return _record("DerivedInitial", options={"reduction": decl.reduction})
    if isinstance(decl, _NETWORKS):
        var = f"{base}_{FAMILIES[kind.__name__].network}"
        key, options = _network(w, decl, var, address)
        remember(main=var)
        return _record(kind.__name__, parameters={key: var}, options=options)
    raise DescriptionError(
        f"{address}: {type(decl).__name__} is not a describable family; the menu is "
        f"{sorted(FAMILIES)}"
    )


def _normalised(arr: np.ndarray, axis: int) -> bool:
    """Whether ``arr`` holds normalised log-probabilities along ``axis`` (within the tolerance
    the ``log_simplex`` domain check allows)."""
    try:
        _check_domain("log_simplex", arr, "", axis)
    except DescriptionError:
        return False
    return True


def _check_layout(holds: bool, address: str, what: str) -> None:
    if not holds:
        raise DescriptionError(
            f"{address}: its {what} does not follow from its arrays' shapes (it carries leading "
            "batch axes), and a document derives the layout from the shapes and the wiring; "
            "declare the factor without batch axes"
        )


def _encode_lg_observation(w: _Writer, decl, base: str, address: str, remember) -> dict[str, Any]:
    """A linear-Gaussian observation: ``head`` and ``noise`` parameter variables, each clamped or
    learned through a prior record. The matrix-normal-inverse-Wishart family is the pair of
    priors, inverse-Wishart under the noise and matrix-normal under the head naming the noise as
    its row covariance (``row_cov``); an independent matrix-normal head names none. Design:
    model-description-0.3.md Q18.3."""
    head_var, noise_var = f"{base}_head", f"{base}_noise"
    joint = isinstance(decl, _factors.LinearGaussianObservationMNIW)
    if joint:
        head = _dist.MatrixNormal(M=decl.M, V_inv=decl.V_inv, M0=decl.M0, V_inv0=decl.V_inv0)
        noise = _dist.InverseWishart(Psi=decl.Psi, nu=decl.nu, Psi0=decl.Psi0, nu0=decl.nu0)
    else:
        head, noise = decl.head, decl.noise
    if isinstance(head, _dist.MatrixNormal):
        w.learned(head_var, head.M.shape, "real", f"{address}/head")
    else:
        w.param(head_var, head, "real", f"{address}/head")
    options: dict[str, Any] = {}
    noise_record: Any = noise_var
    if isinstance(noise, _dist.InverseGamma | _dist.InverseWishart):
        shape = noise.Psi.shape if joint else np.shape(noise.point())
        w.learned(noise_var, shape, "cholesky", f"{address}/noise")
    elif _factors._is_noise_map(noise):
        if isinstance(head, _dist.MatrixNormal):
            raise DescriptionError(
                f"{address}: a noise map beside a learned head is not built; clamp the head"
            )
        if not isinstance(noise, _factors.GridNoiseMap):
            raise DescriptionError(
                f"{address}/noise: a {type(noise).__name__} noise map cannot be described; the "
                "shipped map is GridNoiseMap (values at the nodes of a grid, read by "
                "interpolation), which a document writes as the noise's values with the grid's "
                "extent in options"
            )
        noise_record = {
            "values": w.param(f"{noise_var}_values", noise.values, "cholesky", f"{address}/noise")
        }
        options["noise_map"] = {
            "low": [float(x) for x in np.asarray(noise.low)],
            "high": [float(x) for x in np.asarray(noise.high)],
            "axes": [int(a) for a in noise.axes],
        }
    else:
        w.param(noise_var, noise, "cholesky", f"{address}/noise")
    if isinstance(head, _dist.MatrixNormal):
        fields = {"M": head.M, "V_inv": head.V_inv, "M0": head.M0, "V_inv0": head.V_inv0}
        if joint:
            fields["row_cov"] = noise_var
        w.prior(head_var, "MatrixNormal", fields, f"{address}/head")
    if isinstance(noise, _dist.InverseGamma):
        fields = {"alpha": noise.alpha, "beta": noise.beta, "alpha0": noise.alpha0}
        fields["beta0"] = noise.beta0
        w.prior(noise_var, "InverseGamma", fields, f"{address}/noise")
    elif isinstance(noise, _dist.InverseWishart):
        fields = {"Psi": noise.Psi, "nu": noise.nu, "Psi0": noise.Psi0, "nu0": noise.nu0}
        w.prior(noise_var, "InverseWishart", fields, f"{address}/noise")
    remember(main=head_var)
    return _record(
        "LinearGaussianObservation",
        parameters={"head": head_var, "noise": noise_record},
        options=options or None,
    )


def _encode_deterministic(w: _Writer, decl, base: str, address: str, remember) -> dict[str, Any]:
    """A deterministic node: a GRU carry map as a tree variable, or a named function by
    registry reference in its options (M7)."""
    mapping = decl.map
    if isinstance(mapping, _dist.GRUCell):
        var = f"{base}_map"
        tree: dict[str, Any] = {
            "input_proj": _linear_tree(mapping.input_proj),
            "gate": _linear_tree(mapping.gate),
        }
        if mapping.norms:
            tree["norms"] = [
                _norm_tree(n, f"{address}/map/norms/{i}") for i, n in enumerate(mapping.norms)
            ]
        w.tree(var, tree, f"{address}/map")
        if decl.frozen:
            w.frozen.append(var)
        remember(main=var)
        options = {
            "rule": decl.rule,
            "map": "GRUCell",
            "activation": mapping.activation,
            "norm": bool(mapping.norms),
        }
        return _record("Deterministic", parameters={"map": var}, options=options)
    if isinstance(mapping, eqx.Module):
        raise DescriptionError(
            f"{address}/map: a Deterministic map of type {type(mapping).__name__} has no document "
            "form; the describable maps are a GRUCell (Deterministic.gru) and a function named "
            "at module level"
        )
    if decl.frozen:
        raise DescriptionError(
            f"{address}: a frozen Deterministic over a named function has no parameter variable "
            "to list in configuration.learning.frozen; declare it unfrozen (a function has no "
            "parameters to train)"
        )
    return _record("Deterministic", options={"rule": decl.rule, "map": _impl(mapping, address)})


def _encode_nonlinear(w: _Writer, decl, base: str, address: str) -> dict[str, Any]:
    """A nonlinear transition: ``f`` by its registered name, its leaves one tree variable keyed
    by field name, its static fields JSON values in ``options.f.static``; the innovations as a
    linear-Gaussian transition's (nonlinear-transitions.md N5)."""
    f = decl.f
    names = [n for n, entry in _TRANSITIONS.items() if entry.cls is type(f)]
    if not names:
        raise DescriptionError(
            f"{address}/f: a NonlinearGaussianTransition's f is code, and a document carries "
            f"data only; register its class, register_transition('<package>.<name>', "
            f"{type(f).__qualname__}), and the document names it and writes its fields"
        )
    name = names[0]
    leaves: dict[str, np.ndarray] = {}
    static: dict[str, Any] = {}
    for field_name, is_static in _TRANSITIONS[name].fields:
        value, where = getattr(f, field_name), f"{address}/f/{field_name}"
        if is_static:
            static[field_name] = _static_value(value, where)
        else:
            leaves[field_name] = _float_leaf(value, where)
    inn = decl.innovations
    _check_layout(inn.parent_ndims == inn.loc.ndim - 1, f"{address}/innovations", "parent_ndims")
    parameters: dict[str, Any] = {}
    if leaves:
        parameters["f"] = w.tree(f"{base}_f", leaves, f"{address}/f")
    parameters["innovations"] = {
        "loc": w.param(f"{base}_innovations_loc", inn.loc, "real", f"{address}/innovations"),
        "chol": w.param(f"{base}_innovations_chol", inn.chol, "cholesky", f"{address}/innovations"),
    }
    spec: dict[str, Any] = {"name": name}
    if static:
        spec["static"] = static
    return _record(
        "NonlinearGaussianTransition",
        parameters=parameters,
        options={"f": spec, "propagation": decl.propagation},
    )


def _float_leaf(value: object, address: str) -> np.ndarray:
    """A leaf field of ``f`` as the floating-point array a clamp writes."""
    floating = isinstance(value, float) or (
        hasattr(value, "dtype") and np.issubdtype(np.dtype(value.dtype), np.floating)
    )
    if not floating:
        raise DescriptionError(
            f"{address}: f's leaf field holds a {type(value).__name__}"
            + (f" of dtype {value.dtype}" if hasattr(value, "dtype") else "")
            + ", and a document writes a leaf as a floating-point array; hold it as floats, or "
            "declare it static (eqx.field(static=True)) with a number, string or tuple of them"
        )
    # at the precision the model computes with, so a rebuilt f describes the same
    return np.asarray(jnp.asarray(value))


def _static_value(value: object, address: str) -> Any:
    """A static field of ``f`` as JSON: ``None``, a bool, an int, a finite float or a string,
    or a tuple of these (written as a list, read back as a tuple)."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, tuple):
        return [_static_value(v, address) for v in value]
    raise DescriptionError(
        f"{address}: f's static field holds a {type(value).__name__}, which a document cannot "
        "write; a static field is None, a bool, an int, a finite float, a string, or a tuple of "
        "these"
    )


def _static_json(value: Any) -> bool:
    """Whether a document's static value is one :func:`_static_value` writes."""
    if value is None or isinstance(value, bool | int | str):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return isinstance(value, list) and all(_static_json(v) for v in value)


def _as_tuples(value: Any) -> Any:
    """A static field's JSON value as ``f`` holds it: every list a tuple."""
    if isinstance(value, list):
        return tuple(_as_tuples(v) for v in value)
    return value


def _impl(value: object, address: str) -> dict[str, str]:
    name = getattr(value, "__qualname__", "")
    if not callable(value) or not name or "<" in name:
        raise DescriptionError(
            f"{address}: a lambda or local function cannot be described; declare it at module "
            "level so it can be named by registry reference (M7)"
        )
    return {"impl": f"{value.__module__}:{name}"}


def describe(
    system: System,
    params: SystemParams,
    *,
    arrays: str = "auto",
    inline_max: int = 256,
    extensions: Mapping[str, Mapping[str, Any]] | None = None,
) -> Description:
    """The document for ``(system, params)``, plus its arrays file.

    ``arrays="auto"`` writes an array inline when it has at most ``inline_max`` elements and into
    the arrays file otherwise; ``"inline"`` and ``"file"`` force one placement. A constant array
    is written as the scalar it broadcasts from, except under ``"file"``. The document names the
    arrays file in ``arrays_file`` when any clamp refers to it. ``extensions`` is the one place
    for a consumer's own data, keyed by consumer name and carried untouched.

    Every variable, factor and term is written; what has no document form is refused by
    address: a lambda, a module outside the family menu, a scoring entry no term mounts, a
    recognition variable or external input named like a node (the document has one namespace;
    a node and a modality of one name the System itself refuses), and a nonlinear transition
    whose ``f``'s class is not registered (:func:`register_transition`): a registered ``f`` is
    written as its name and its fields, since a document carries data and never code.
    Design: docs/design/model-description-0.3.md; the format: docs/description-format.md.
    """
    params = system._params_in(params, "describe")
    try:
        system._validate(params, "describe")
    except Exception as exc:  # the library's own validators speak in sentences already
        raise DescriptionError(f"the System is invalid, so it has no document: {exc}") from exc
    writer = _Writer(arrays, inline_max)
    body = _write_system(writer, system, params, "")
    document: dict[str, Any] = {KEY: VERSION, "library_version": _library_version(), **body}
    if writer.arrays:
        document["arrays_file"] = ARRAYS_FILE
    if extensions is not None:
        document[EXTENSIONS] = _check_extensions(extensions)
    return Description(document, writer.arrays)


def _check_extensions(extensions: object) -> dict[str, Any]:
    if not isinstance(extensions, Mapping) or not all(
        isinstance(k, str) and isinstance(v, Mapping) for k, v in extensions.items()
    ):
        raise DescriptionError(
            f"{EXTENSIONS!r} is an object keyed by consumer name, each value an object "
            "(e.g. {'zeroth_control': {...}}); the library never reads inside it"
        )
    return {k: dict(v) for k, v in extensions.items()}


def _is_tabular(decl: object) -> bool:
    return isinstance(
        decl,
        _dist.Categorical
        | _dist.Dirichlet
        | _dist.ConditionalCategorical
        | _dist.ConditionalDirichlet,
    )


def _letter(letter: str, child: str) -> str:
    return _qualified(child, lambda name: f"{letter}_{name}")


def _base(child: str, label: str) -> str:
    return _qualified(child, lambda name: f"{name}_{label}")


def _with_parents(record: dict[str, Any], parents: list[str]) -> dict[str, Any]:
    if not parents:
        return record
    out = {"family": record["family"], "parents": parents}
    out.update({k: v for k, v in record.items() if k != "family"})
    return out


def _categorical(levels: int, shape: tuple[int, ...] = ()) -> dict[str, Any]:
    out: dict[str, Any] = {"domain": "categorical", "levels": int(levels)}
    if shape:
        out["shape"] = [int(s) for s in shape]
    return out


def _real(dim: int) -> dict[str, Any]:
    return {"domain": "real", "dim": int(dim)}


def _bound(values: Array) -> Any:
    arr = np.asarray(values)
    return arr.flat[0].item() if arr.size > 1 and _constant(arr) else arr.tolist()


def _state_domain(params: SystemParams, f: int, name: str, where: str) -> dict[str, Any]:
    """A state factor's domain, read off its initial prior (or, for a code, its transition)."""
    d, b = params.initial_state_prior[f], params.transition[f]
    if isinstance(d, _dist.Categorical):
        return _categorical(d.logits.shape[-1])
    if isinstance(d, _dist.Dirichlet):
        return _categorical(d.concentration.shape[-1])
    if isinstance(d, _dist.ConditionalCategorical):
        return _categorical(d.log_table.shape[-(d.parent_ndims + 1)])
    if isinstance(d, _dist.ConditionalDirichlet):
        return _categorical(d.concentration.shape[-(d.parent_ndims + 1)])
    if isinstance(d, _dist.Gaussian):
        return _real(d.loc.shape[-1])
    if isinstance(d, _dist.MixtureGaussian):
        return _real(d.means.shape[-1])
    if isinstance(d, _dist.LearnedVector):
        return _real(d.dim)
    return _categorical(b.num_classes, (b.num_vars,))  # a code, its initial read off itself


def _output_domain(module: object) -> dict[str, Any]:
    """The domain of an auxiliary recognition variable: its module's declared output width."""
    return _real(module.out_dim)


def _observation_domain(decl: object, address: str) -> dict[str, Any]:
    if isinstance(decl, _dist.ConditionalCategorical | _dist.ConditionalDirichlet):
        return _categorical(decl.num_levels)
    if isinstance(decl, _dist.MLPConditional):
        return _real(decl.child_dim)
    if isinstance(decl, _dist.PredictorHead):
        return _real(decl.target_dim)
    return _real(decl.obs_dim)  # a linear-Gaussian observation


def _write_system(w: _Writer, system: System, params: SystemParams, where: str) -> dict[str, Any]:
    """Every section of the document but its version keys, for a positional ``params``."""
    table = system._table
    states, nodes = table.state_names, table.control_names
    actions, mods = tuple(system.action_names), tuple(system.modality_names)
    recognition = tuple(system.recognition or ())
    aux = [r.variable for r in recognition if r.variable not in states]
    externals = [name for name, _ in system.external_inputs]
    kinds: dict[str, str] = {}
    for kind, names in (
        ("a state", states),
        ("a control node", nodes),
        ("an action node", actions),
        ("an observation", mods),
        ("a recognition variable", aux),
        ("an external input", externals),
    ):
        for name in names:
            if "[" in name or "]" in name:
                raise DescriptionError(
                    f"{where}{kind} is named {name!r}, and a variable name may not contain a "
                    "bracket (the document writes a parent's step as name[t] or name[t-1]); "
                    "rename it"
                )
            if name in kinds:
                raise DescriptionError(
                    f"{where}the System names {name!r} both as {kinds[name]} and as {kind}, and a "
                    "description has one namespace of variables; rename one of them "
                    "(model-description-0.3.md Q18.5c)"
                )
            kinds[name] = kind
    space = system._space(params) if nodes else None
    durations = system._durations(params, "describe") if nodes else {}

    # ------------------------------------------------------------ per-step variables
    declared_dims = dict(system.dims)
    for f, s in enumerate(states):
        w.variables[s] = _state_domain(params, f, s, where)
        if s in declared_dims:  # only declared names are written; dim0, dim1, … are implied
            w.variables[s]["dims"] = list(declared_dims[s])
    for k, u in enumerate(nodes):
        if space.kinds[k] == FLOAT:
            box = params.control[u]
            w.variables[u] = {
                **_real(box.dim),
                "low": _bound(box.low),
                "high": _bound(box.high),
            }
        else:
            w.variables[u] = _categorical(space.sizes[k])
            if u in durations:
                w.variables[u]["duration"] = _bound(durations[u])
    for k, a in enumerate(actions):
        w.variables[a] = (
            _real(space.sizes[k]) if space.kinds[k] == FLOAT else _categorical(space.sizes[k])
        )
    for m, o in enumerate(mods):
        w.variables[o] = _observation_domain(params.likelihood[m], f"{where}likelihood:{o}")
    for v in aux:
        w.variables[v] = _output_domain(params.recognition[v])
    for name, width in system.external_inputs:
        w.variables[name] = _real(width)

    # ------------------------------------------------------------- the factor blocks
    order = [*states, *nodes, *actions, *mods, *aux]
    children = [r.variable for r in recognition]
    needs = {
        r.variable: [p.partition(":")[2] for p in r.parents if p.startswith("recognition:")]
        for r in recognition
    }
    if _evaluation_order(order, needs)[0] != children:
        #  build derives the evaluation order from the parents, taking records in document
        #  order where the parents leave it free: put them in the System's order
        for pos, child in zip(sorted(order.index(c) for c in children), children, strict=True):
            order[pos] = child
    for v in order:
        w.factors[v] = {}

    n_nodes = len(system.B_deps)

    def parent_name(g: int) -> str:
        return states[g] if g < len(states) else actions[g - n_nodes]

    for f, s in enumerate(states):
        d = params.initial_state_prior[f]
        record = _encode(
            w,
            d,
            _base(s, "initial"),
            _letter("D", s) if _is_tabular(d) else None,
            f"{where}initial_state_prior:{s}",
            ("initial_state_prior", s),
        )
        d_parents = [states[g] for g in table.D_deps[f]] if table.D_deps else []
        w.record(s, "initial", _with_parents(record, d_parents))
        b = params.transition[f]
        record = _encode(
            w,
            b,
            _base(s, "transition"),
            _letter("B", s) if _is_tabular(b) else None,
            f"{where}transition:{s}",
            ("transition", s),
        )
        spec = FAMILIES[record["family"]]
        same_step = spec.reads_same_step
        b_parents = [s] if spec.reads_self else []
        for what, i in declared_parents(table, f):
            if what == "state":
                b_parents.append(f"{states[i]}[t]" if same_step else states[i])
            else:
                b_parents.append(nodes[i])
        w.record(s, "transition", _with_parents(record, b_parents))

    for m, o in enumerate(mods):
        a = params.likelihood[m]
        record = _encode(
            w,
            a,
            _base(o, "likelihood"),
            _letter("A", o) if _is_tabular(a) else None,
            f"{where}likelihood:{o}",
            ("likelihood", o),
        )
        lagged = dict(system.lagged).get(o)
        if lagged is not None:  # the datum read from the carry (structure-learning.md S8)
            record = {**record, "options": {**record.get("options", {}), "datum": f"{lagged}[t-1]"}}
        w.record(o, "likelihood", _with_parents(record, [parent_name(g) for g in system.A_deps[m]]))

    for k, a_name in enumerate(actions):
        if a_name not in params.actuator:
            continue
        with np.errstate(divide="ignore"):
            log_kernel = np.log(np.asarray(params.actuator[a_name], np.float64))
        var = w.param(_letter("K", a_name), log_kernel, "log_simplex", f"{where}actuator:{a_name}")
        record = _record("ConditionalCategorical", [nodes[k]], {"log_table": var})
        w.record(a_name, "actuator", record)

    for r in recognition:
        module = params.recognition[r.variable]
        record = _encode(
            w,
            module,
            _base(r.variable, "recognition"),
            None,
            f"{where}recognition:{r.variable}",
            ("recognition", r.variable),
        )
        parents = [p.partition(":")[2] if ":" in p else p for p in r.parents]
        w.record(r.variable, "recognition", _with_parents(record, parents))

    if params.policy_prior is not None:
        if not nodes:
            raise DescriptionError(
                f"{where}policy_prior is declared on a System with no control node, and a "
                "document writes it under the first control node; drop it or declare the "
                "control nodes"
            )
        prior = params.policy_prior
        if isinstance(prior, _dist.Categorical) or hasattr(prior, "shape"):
            logits = prior.logits if isinstance(prior, _dist.Categorical) else prior
            var = w.param("E", logits, "logits", f"{where}policy_prior")
            w.record(nodes[0], "policy_prior", _record("Categorical", parameters={"logits": var}))
        else:  # a policy module: the control node's amortized q given the beliefs
            record = _encode(w, prior, _base(nodes[0], "recognition"), None, f"{where}policy_prior")
            w.record(nodes[0], "recognition", _with_parents(record, list(states)))

    for slot, label, letter in (
        ("state_targets", "target", "C"),
        ("obs_targets", "target", "C"),
        ("terminal_targets", "terminal", "H"),
    ):
        for name, target in getattr(params, slot).items():  # keys checked by the validator
            record = _encode(
                w,
                target,
                _base(name, label),
                _letter(letter, name) if _is_tabular(target) else None,
                f"{where}{slot}:{name}",
            )
            parents: list[str] = []
            if label == "terminal" and name in states and table.H_deps:
                parents = [states[g] for g in table.H_deps[states.index(name)]]
            elif label == "target":
                parents = list(dict(system.C_deps).get(name, ()))
            w.record(name, label, _with_parents(record, parents))

    precision = None
    if params.precision is not None:
        precision = w.param("precision", params.precision, "positive", f"{where}precision")

    objective = _write_objective(w, system, params, where)
    relations: dict[str, Any] = {}
    if nodes:
        relations["controls"] = dict(zip(nodes, actions, strict=True))
    if mods:
        relations["observed"] = list(mods)
    if externals:
        relations["external"] = externals
    return {
        "variables": w.variables,
        "factors": {child: block for child, block in w.factors.items() if block},
        "relations": relations,
        "clamps": w.clamps,
        "objective": objective,
        "configuration": _write_configuration(system, w.frozen, precision),
    }


def _evaluation_order(order: list[str], needs: Mapping[str, list[str]]) -> tuple[list, list]:
    """The recognition evaluation order derived from the parents: each record once the
    auxiliary variables it reads (``needs``) are placed, earliest in ``order`` first. Returns
    ``(placed, stuck)``; ``stuck`` is non-empty when the records read one another in a cycle."""
    pending = [v for v in order if v in needs]
    placed: list[str] = []
    while pending:
        ready = next((v for v in pending if all(p in placed for p in needs[v])), None)
        if ready is None:
            return placed, pending
        placed.append(ready)
        pending.remove(ready)
    return placed, []


def _cycle(stuck: list[str], needs: Mapping[str, list[str]]) -> list[str]:
    """One cycle among ``stuck`` records, followed from the first through unplaced needs."""
    path, node = [], stuck[0]
    while node not in path:
        path.append(node)
        node = next(p for p in needs[node] if p in stuck)
    return path[path.index(node) :]


# ------------------------------------------------------------------- the objective

_PAIR_TERMS = ("utility", "risk", "ambiguity")


def _write_objective(
    w: _Writer, system: System, params: SystemParams, where: str
) -> dict[str, Any]:
    """The term list from ``decomposition``, ``epistemic``, ``terms`` and ``scoring``: the sugar
    ``decomposition`` for the default pair (every targeted observation, every eligible variable),
    the pair as terms otherwise, and one record per mounted row. Design: Q18.5d."""
    scoring = dict(params.scoring)
    owned = dict(system.scoring)
    estimators = {name: entry for name, entry in scoring.items() if name not in _registry.ROWS}
    claimed: set[str] = set()
    out: dict[str, Any] = {}
    pair: list[dict[str, Any]] = []
    if system.epistemic is True and not estimators:
        out["decomposition"] = system.decomposition
    else:
        if system.decomposition == "utility_infogain":
            if params.obs_targets:
                pair.append({"term": "utility"})
        else:
            pair += [{"term": "risk"}, {"term": "ambiguity"}]
        if system.epistemic is True:
            infogain = [{"term": INFOGAIN}]
        elif not system.epistemic:
            infogain = []
        else:
            infogain = [
                {"term": INFOGAIN, "on": on}
                for key in system.epistemic
                for on in _epistemic_on(w, key, where)
            ]
        for name, entry in estimators.items():
            estimator = _registry.by_name(name)
            home = next(
                (
                    rec
                    for rec in infogain
                    if "parameters" not in rec and rec.get("on", "*") not in system.modality_names
                ),
                None,
            )
            if estimator is None or home is None:
                raise DescriptionError(
                    f"{where}scoring[{name!r}] holds parameters no term reads: no infogain term "
                    "on a variable its estimator scores is selected; select one (epistemic) or "
                    "drop the entry"
                )
            home["parameters"], home["options"] = _term_parameters(
                w, estimator, entry, INFOGAIN, f"{where}scoring:{name}"
            )
            claimed.add(name)
        pair += infogain
    rows: list[dict[str, Any]] = []
    for key in system.terms:
        row, _, rest = key.partition(":")
        estimator = _registry.ROWS[row]  # the validator refuses an unregistered row
        record: dict[str, Any] = {"term": row}
        on = _row_on(w, estimator, row, rest, where)
        if on is not None:
            record["on"] = on
        if row in scoring and row not in claimed:
            if estimator.needs_structure:
                inner_system = owned[row]  # the validator pairs a model row's two halves
                inner = w.nested(row)
                record["model"] = _write_system(
                    inner, inner_system, inner_system._p(scoring[row]), f"{where}{row}/"
                )
            else:
                record["parameters"], record["options"] = _term_parameters(
                    w, estimator, scoring[row], row, f"{where}scoring:{row}"
                )
            claimed.add(row)
        rows.append(record)
    unclaimed = sorted(set(scoring) - claimed)
    if unclaimed:
        raise DescriptionError(
            f"{where}SystemParams.scoring holds {unclaimed}, which no term in System.terms mounts, "
            "and a document carries a term's parameters on its term record; mount the row or drop "
            "the entry"
        )
    if pair or rows or "decomposition" not in out:
        out["terms"] = pair + rows
    return out


def _epistemic_on(w: _Writer, key: str, where: str) -> list[str]:
    """One ``epistemic`` key as the variable its infogain term names in ``on`` (a wildcard as
    written): a latent by name, a dimension of one as ``<latent>:<dim>``, an observation by name,
    a declaration's parameter variable."""
    if "*" in key.split(":") or ":" not in key:
        return [key]
    head, _, rest = key.partition(":")
    if head == "obs":
        return [rest]
    if head not in _KEY_KINDS:  # a dimension of a latent
        return [key]
    return [w.slots[(head, rest.partition(":")[0])]["main"]]


def _row_on(w: _Writer, estimator, row: str, rest: str, where: str) -> str | None:
    """A ``terms`` entry's target as the variable its term record names in ``on``; ``None`` for
    a row with no target, and for the whole class (a wildcard over every eligible variable)."""
    if not rest:
        return None
    target = estimator.target
    if target == _registry.SLOT_TARGET:
        slot, _, name = rest.partition(":")
        fixed = _registry.row_slot(row)
        if name == "*" and slot == fixed:
            return None
        if "*" in rest.split(":"):
            return rest
        if slot == fixed:
            return name
        return w.slots[(slot, name)]["main"]
    if target == _registry.OBS_TARGET:
        return None if rest == "obs:*" else (rest[4:] if "*" not in rest else rest)
    if target == _registry.STATE_TARGET:
        return None if rest == "*" else rest
    return rest


def _term_network(w: _Writer, module: object, var: str, address: str) -> dict[str, Any]:
    """A term parameter's tree variable, and its options: the family, its configuration and,
    since no wiring implies it, its layout."""
    family = type(module).__name__
    if family not in _TERM_FAMILIES:
        raise DescriptionError(
            f"{address}: a term parameter of type {family} has no document form; the families a "
            f"term's parameters may hold are {list(_TERM_FAMILIES)}"
        )
    _, options = _network(w, module, var, address)
    trunk = module.trunk if family == "PredictorHead" else module
    layout: dict[str, Any] = {"parent_dims": list(trunk.parent_dims)}
    if family == "MLPConditional":
        layout["child_dim"] = module.child_dim
    elif family == "MLPPredictor" or module.loss == "symlog_mse":
        layout["out_dim"] = trunk.out_dim
    return {"family": family, **options, **layout}


def _term_parameters(
    w: _Writer, estimator, entry: object, term: str, address: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """A parametric term's ``parameters`` and ``options``, named by the estimator's declared
    parameter names: one name holds the entry's tuple as a list; several name its items."""
    names = estimator.parameters
    if not names:
        raise DescriptionError(
            f"{address}: the {estimator.name!r} estimator declares no parameter names, so its "
            "parameters cannot be written; declare them at registration (Estimator.parameters)"
        )
    if len(names) == 1:
        if not isinstance(entry, tuple):
            raise DescriptionError(
                f"{address}: {estimator.name!r} holds its {names[0]!r} as a tuple, got "
                f"{type(entry).__name__}"
            )
        variables, options = [], []
        for i, item in enumerate(entry):
            var = f"{term}_{names[0]}_{i}"
            options.append(_term_network(w, item, var, f"{address}/{i}"))
            variables.append(var)
        shared = all(o == options[0] for o in options)
        return {names[0]: variables}, {names[0]: options[0] if shared else options}
    items = entry if isinstance(entry, tuple) else (entry,)
    if len(items) > len(names):
        raise DescriptionError(
            f"{address}: {estimator.name!r} holds {len(items)} items but names {list(names)}"
        )
    parameters, options = {}, {}
    for i, (name, item) in enumerate(zip(names, items, strict=False)):
        parameters[name] = f"{term}_{name}"
        options[name] = _term_network(w, item, parameters[name], f"{address}/{i}")
    return parameters, options


# --------------------------------------------------------------- the configuration

#: The System's own defaults for the settings ``configuration`` writes.
_DEFAULTS = {
    f.name: f.default for f in dataclasses.fields(System) if f.default is not dataclasses.MISSING
}
_SCHEDULES = ("runnalls", "mode_aligned")
_BELIEFS = ("prior", "posterior")
_FEEDBACK = FEEDBACK


def _planners() -> list[str]:
    from hybrid_aif.planning.base import _PLANNERS

    return sorted(_PLANNERS)


def _write_configuration(
    system: System, frozen: list[str], precision: str | None = None
) -> dict[str, Any]:
    """The System's settings beyond the model, a sub-block omitted at its defaults (``planning``
    always carries the horizon, which has none). ``precision`` names the parameter variable
    holding ``SystemParams.precision``, written when there is one. Design: Q18.5e."""
    planning: dict[str, Any] = {
        "planner": system.planner,
        "horizon": int(system.policy_len),
        "gamma": system.gamma,
    }
    if precision is not None:
        planning["precision"] = precision
    if system.branch_on:
        planning["branch_on"] = list(system.branch_on)
    if system.planning_belief != "prior":
        planning["belief"] = system.planning_belief
    if system.feedback is not None:
        planning["feedback"] = system.feedback
    out: dict[str, Any] = {"planning": planning}
    inference = {
        "iterations": int(system.num_iters),
        "mixture_k": system.mixture_k,
        "mixture_schedule": system.mixture_schedule,
    }
    defaults = {
        "iterations": _DEFAULTS["num_iters"],
        "mixture_k": _DEFAULTS["mixture_k"],
        "mixture_schedule": _DEFAULTS["mixture_schedule"],
    }
    if inference != defaults:
        out["inference"] = inference
    learning: dict[str, Any] = {}
    if frozen:
        learning["frozen"] = list(frozen)
    if system.reduction_period is not None:
        learning["reduction_period"] = int(system.reduction_period)
    if system.reduction_pairs != _DEFAULTS["reduction_pairs"]:
        learning["reduction_pairs"] = int(system.reduction_pairs)
    if learning:
        out["learning"] = learning
    return out


# ========================================================================== build


def build(
    document: Mapping[str, Any],
    *,
    arrays: Mapping[str, np.ndarray] | None = None,
    key: Array | None = None,
    support_floor: float = SUPPORT_FLOOR,
) -> tuple[System, SystemParams]:
    """The ``(System, SystemParams)`` a document describes.

    Validates the document first, every fault a :class:`DescriptionError` naming the address and
    what to write instead, then constructs the declaration and runs ``System._validate``.
    ``arrays`` is the arrays file for ``ref`` clamps, keyed by the key part of the ref; ``key``
    draws the values of ``{"init": ...}`` clamps. A document in an earlier version is refused
    with a sentence naming the release that reads it.
    """
    version = document.get(KEY) if isinstance(document, Mapping) else None
    if version is None:
        raise DescriptionError(f"not a hybrid-aif description: the {KEY!r} key is missing")
    parts = str(version).split(".")
    try:
        major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        major, minor = -1, -1
    current = tuple(int(p) for p in VERSION.split("."))
    if major != current[0] or minor > current[1] or minor < 0:
        raise DescriptionError(
            f"this document is description version {version}; this library reads {VERSION}"
        )
    if EXTENSIONS in document:
        _check_extensions(document[EXTENSIONS])
    if minor < current[1]:
        raise DescriptionError(
            f"this document is description {version}, in the structure/declarations layout, "
            f"which this release does not read; build it with hybrid-aif 0.18.0 and describe it "
            f"again to write description {VERSION}"
        )
    unknown = sorted(set(document) - TOP_LEVEL_KEYS)
    if unknown:
        raise DescriptionError(
            f"the document has top-level key(s) {unknown} this library does not know. The "
            f"keys are {sorted(TOP_LEVEL_KEYS)}; a consumer's own data belongs under "
            f"{EXTENSIONS!r}, keyed by the consumer's name, where it is carried and never read"
        )
    reader = _Reader(arrays, key)
    system, params = reader.system(document, "")
    try:
        system._validate(params, "build")
    except Exception as exc:  # the library's own validators speak in sentences already
        raise DescriptionError(
            f"the document builds, but the System it describes is invalid: {exc}"
        ) from exc
    check_observation_support(system, params, support_floor=support_floor)
    return system, params


class ParameterRecord(NamedTuple):
    """One parameter array a document declares: its ``address`` (the variable's name, with a
    leaf's path for a tree variable), ``dtype``, ``shape``, ``domain`` and ``source``
    (``"clamp"``, ``"ref"``, ``"init"`` or ``"prior"``)."""

    address: str
    dtype: str
    shape: tuple[int, ...]
    domain: str
    source: str


def describe_parameters(
    document: Mapping[str, Any], arrays: Mapping[str, np.ndarray] | None = None
) -> tuple[ParameterRecord, ...]:
    """The parameter envelope of a document: one record per static variable's array, read off
    the document's own records without constructing the model. An inline clamp has the build's
    default float dtype, a ref the dtype of its array when ``arrays`` is given, a learned
    variable (one with a ``prior`` record) or an ``init`` recipe the default; a tree variable
    gives one record per leaf. The document is validated as far as its variables, factors and
    kinds, so a malformed one raises :class:`DescriptionError`."""
    if not isinstance(document, Mapping) or document.get(KEY) is None:
        raise DescriptionError(f"not a hybrid-aif description: the {KEY!r} key is missing")
    model = _Model(_Reader(arrays, None), document, "")
    model._variables()
    model._factors()
    model._relations()
    model._configuration()
    model._objective_records()
    model._kinds()
    default = str(jnp.zeros(()).dtype)
    clamps = document.get("clamps", {}) or {}
    priors = {fac.child for fac in model.factors if fac.label == "prior"}
    out: list[ParameterRecord] = []

    def leaf(address: str, record: Mapping[str, Any], clamp: Any, source: str) -> None:
        domain = str(record.get("domain", "real"))
        shape = tuple(int(x) for x in record.get("shape", ()))
        dtype = default
        if isinstance(clamp, Mapping) and "ref" in clamp:
            source = "ref"
            if arrays is not None:
                key = str(clamp["ref"]).partition("#")[2]
                if key not in arrays:
                    raise DescriptionError(
                        f"clamps[{address!r}]: the ref {clamp['ref']!r} names a key the arrays "
                        "file does not hold"
                    )
                dtype = str(np.asarray(arrays[key]).dtype)
                shape = tuple(int(x) for x in np.shape(arrays[key])) or shape
        elif isinstance(clamp, Mapping) and "init" in clamp:
            source = "init"
        elif clamp is not None and not shape:
            try:
                shape = tuple(int(x) for x in np.shape(np.asarray(clamp, np.float64)))
            except (TypeError, ValueError) as exc:
                raise DescriptionError(
                    f"clamps[{address!r}]: the clamp is not a number or a nested list of numbers"
                ) from exc
        out.append(ParameterRecord(address, dtype, shape, domain, source))

    def walk(address: str, record: Any, clamp: Any, source: str) -> None:
        if isinstance(record, Mapping) and any(k in record for k in _LEAF_KEYS):
            leaf(address, record, clamp, source)
        elif isinstance(record, Mapping):
            for key, sub in record.items():
                sub_clamp = clamp.get(key) if isinstance(clamp, Mapping) else None
                walk(f"{address}/{key}", sub, sub_clamp, source)
        elif isinstance(record, list):
            for i, sub in enumerate(record):
                sub_clamp = clamp[i] if isinstance(clamp, list) and i < len(clamp) else None
                walk(f"{address}/{i}", sub, sub_clamp, source)

    for name in model.static:
        record = model.variables[name]
        clamp = clamps.get(name)
        source = "prior" if name in priors else "clamp"
        if (
            isinstance(clamp, Mapping)
            and "init" in clamp
            and not any(k in record for k in _LEAF_KEYS)
        ):
            # an init recipe over a tree declared as {"domain": "real"} alone: one record
            leaf(name, {"domain": record.get("domain", "real")}, clamp, "init")
            continue
        walk(name, record, clamp, source)
    return tuple(out)


# ============================================================ evidence support (Z5/Z6)


def check_observation_support(
    system: System, params: SystemParams, *, support_floor: float = SUPPORT_FLOOR
) -> None:
    """Check that every level of every tabular modality is produced by some state.

    Linear in the tables; :func:`build` always runs it. It reports a dead observation level (no
    state configuration above ``support_floor``) by modality and level; jointly excluding
    combinations are :func:`check_joint_support`'s.
    """
    mods = system.modality_names
    params = system._params_in(params, "check_observation_support")
    for m, a in enumerate(params.likelihood):
        if not isinstance(a, _factors.TabularObservation):
            continue  # LG has no zero support
        table = np.asarray(_as_log_likelihood(a))  # the FPI's own tempered lowering
        by_level = table.reshape(table.shape[0], -1)
        dead = np.flatnonzero(by_level.max(axis=1) < support_floor)
        if dead.size:
            raise DescriptionError(
                f"likelihood:{mods[m]}: observation level(s) {dead.tolist()} are produced by no "
                f"state (their best log-likelihood is below the support floor "
                f"{support_floor:.1f}); every level of a tabular modality needs some state that "
                "can produce it. (This check does not see combinations of modalities that "
                "jointly exclude every state — run check_joint_support for that.)"
            )
        empty = np.flatnonzero(by_level.max(axis=0) < support_floor)
        if empty.size:
            raise DescriptionError(
                f"likelihood:{mods[m]}: parent configuration(s) {empty.tolist()} (flat index over "
                "the parent axes) produce no observation level at all — the column is all "
                "-inf, so it is not a distribution. Give every state some observation"
            )


def check_joint_support(
    system: System,
    params: SystemParams,
    *,
    max_combinations: int,
    support_floor: float = SUPPORT_FLOOR,
) -> list[dict[str, int]]:
    """The observation tuples no state configuration explains: exhaustive, within a budget.

    Over the tabular modalities with all-categorical parents, returns each observation tuple whose
    best joint log-likelihood is below ``support_floor``, as ``{modality: level}``. When the product
    of observation levels and state levels exceeds ``max_combinations`` it raises with the count and
    the budget; it never samples.
    """
    mods = system.modality_names
    params = system._params_in(params, "check_joint_support")
    families = system._families(params) if system._is_hybrid(params) else None
    tabular = [
        m
        for m, a in enumerate(params.likelihood)
        if isinstance(a, _factors.TabularObservation)
        and all(families is None or families[f] == "categorical" for f in system.A_deps[m])
    ]
    if not tabular:
        return []
    factors = sorted({f for m in tabular for f in system.A_deps[m]})
    levels = {f: int(params.initial_state_prior[f].num_levels) for f in factors}
    obs_levels = [int(params.likelihood[m].num_levels) for m in tabular]
    count = math.prod(obs_levels) * math.prod(levels[f] for f in factors)  # exact, no wrap
    if count > max_combinations:
        raise ValueError(
            f"check_joint_support: {count} (observation tuple, state configuration) pairs over "
            f"{len(tabular)} tabular modalit{'y' if len(tabular) == 1 else 'ies'} and "
            f"{len(factors)} factor(s) exceed the budget max_combinations={max_combinations}; "
            "raise the budget or check a sub-model. This diagnostic never samples "
            "the space"
        )
    pos = {f: i for i, f in enumerate(factors)}
    n_obs, n_fac = len(tabular), len(factors)
    joint = np.zeros((*obs_levels, *[levels[f] for f in factors]))
    for i, m in enumerate(tabular):
        table = np.asarray(_as_log_likelihood(params.likelihood[m]))  # tempered, as the FPI
        laid = _lay_into_joint(table, i, system.A_deps[m], pos, n_obs, n_fac)
        joint = joint + np.broadcast_to(laid, joint.shape)
    best = joint.reshape(*obs_levels, -1).max(axis=-1)
    bad = np.argwhere(best < support_floor)
    return [{mods[m]: int(level) for m, level in zip(tabular, combo, strict=True)} for combo in bad]


def _lay_into_joint(table, i, deps, pos, n_obs, n_fac):
    """A modality's ``(level, *parents)`` table laid into the joint's ``(*obs, *states)`` axes.

    The level axis goes to observation slot ``i``, each parent to its factor's state slot, every
    other axis is 1, so a broadcast add composes the modalities.
    """
    order = sorted(range(len(deps)), key=lambda k: pos[deps[k]])
    arr = np.transpose(table, [0] + [1 + k for k in order])
    shape = [1] * (n_obs + n_fac)
    shape[i] = table.shape[0]
    for k in order:
        shape[n_obs + pos[deps[k]]] = table.shape[1 + k]
    return arr.reshape(shape)


# ------------------------------------------------------------------- the records


class _Factor(NamedTuple):
    """One factor record, lowered: every parent with its offset (0: the child's step, 1: the
    step before) and the steps the record applies at (``None`` for plan-time and static
    records)."""

    child: str
    label: str
    family: str
    parents: tuple[tuple[str, int], ...]
    parameters: Mapping[str, Any]
    options: Mapping[str, Any]
    steps: str | None
    address: str


class _Learned(NamedTuple):
    """A parameter variable learned through its ``prior`` record."""

    var: str
    family: str
    parameters: Mapping[str, Any]
    address: str


_RECORD_KEYS = ("family", "parents", "parameters", "options", "steps")
_LEAF_KEYS = ("domain", "levels", "dim", "dims", "low", "high", "shape", "duration")
_TERM_KEYS = ("term", "on", "parameters", "options", "model")
_RELATIONS = ("controls", "observed", "external")
_CONFIGURATION = {
    "planning": ("planner", "horizon", "gamma", "precision", "branch_on", "belief", "feedback"),
    "inference": ("iterations", "mixture_k", "mixture_schedule"),
    "learning": ("frozen", "reduction_period", "reduction_pairs"),
}
_BODY_KEYS = ("variables", "factors", "relations", "clamps", "objective", "configuration")


def _parse_parent(entry: object, address: str) -> tuple[str, int | None]:
    """``x``, ``x[t]`` or ``x[t-1]`` as ``(name, offset)``; ``None`` for a bare name."""
    if not isinstance(entry, str):
        raise DescriptionError(f"{address}: a parent is a variable name, got {entry!r}")
    match = _BRACKET.match(entry)
    if (
        match is None
        or not match.group("name")
        or (match.group("offset") is not None and match.group("offset") not in _OFFSETS)
    ):
        raise DescriptionError(
            f"{address}: the parent {entry!r} is not a variable name with an optional step; "
            "write name[t] for the child's step or name[t-1] for the step before"
        )
    offset = match.group("offset")
    return match.group("name"), None if offset is None else _OFFSETS[offset]


def _default_dtype() -> np.dtype:
    return np.dtype(jnp.zeros((), float).dtype)


def _check_domain(domain: str, arr: np.ndarray, address: str, axis: int | None = -1) -> None:
    """The cheap checks of a parameter domain (M6): positivity, triangularity, symmetry and a
    positive spectrum, and a simplex in log space or in probabilities. ``axis`` is the axis a
    distribution runs along; ``None`` makes the whole array one distribution."""
    if not arr.size:
        return
    if domain in ("log_simplex", "simplex", "logits") and np.isnan(arr).any():
        impossible = "0" if domain == "simplex" else "-inf"
        raise DescriptionError(
            f"{address} contains NaN (domain {domain!r}); {impossible} is legal for an impossible "
            "entry, NaN is not a value"
        )
    if domain == "log_simplex":
        with np.errstate(divide="ignore", invalid="ignore"):
            peak = np.max(arr, axis=axis, keepdims=True)
            total = np.log(np.sum(np.exp(arr - peak), axis=axis)) + np.squeeze(peak, axis)
        total = np.atleast_1d(total)
        live = np.isfinite(total)
        if np.any(np.abs(total[live]) > 1e-3) or np.any(arr > 1e-6):
            raise DescriptionError(
                f"{address} must be normalised log-probabilities (domain 'log_simplex': each "
                "distribution's entries are <= 0 and their exponentials sum to one); normalise "
                "it, write it as probabilities under the domain 'simplex', or give the variable "
                "the domain 'logits' where its family takes logits"
            )
    if domain == "simplex":
        total = np.atleast_1d(np.sum(arr, axis=axis))
        live = total > 0  # a distribution of zeros is one of impossible entries, as all -inf is
        if np.any(arr < 0) or np.any(np.abs(total[live] - 1.0) > 1e-3):
            raise DescriptionError(
                f"{address} must be probabilities (domain 'simplex': entries >= 0 and each "
                "distribution sums to one); normalise it"
            )
    if domain == "positive" and not np.all(arr > 0):
        raise DescriptionError(
            f"{address} must be strictly positive everywhere (domain 'positive')"
        )
    if domain == "nonnegative" and not np.all(np.isfinite(arr) & (arr >= 0)):
        raise DescriptionError(
            f"{address} must be finite and nonnegative everywhere (domain 'nonnegative'; a zero "
            "is a dead level of a growable factor)"
        )
    if domain in ("cholesky", "spd") and arr.ndim < 2:
        raise DescriptionError(
            f"{address} must be a matrix (domain {domain!r}), got shape {list(arr.shape)}"
        )
    if domain == "cholesky":
        if not np.allclose(np.triu(arr, k=1), 0.0):
            raise DescriptionError(
                f"{address} must be lower-triangular (domain 'cholesky'); it has entries above "
                "the diagonal"
            )
        if not np.all(np.diagonal(arr, axis1=-2, axis2=-1) > 0):
            raise DescriptionError(
                f"{address} must have a strictly positive diagonal (domain 'cholesky')"
            )
    if domain == "spd":
        sym = np.allclose(arr, np.swapaxes(arr, -1, -2), rtol=1e-5, atol=1e-6)
        if not sym or not np.all(np.linalg.eigvalsh(np.asarray(arr, np.float64)) > 0):
            raise DescriptionError(f"{address} must be symmetric positive definite (domain 'spd')")


def _lowered(domain: str, arr: np.ndarray, dtype) -> np.ndarray:
    """A checked clamp in the domain its field takes: probabilities become their logs, taken in
    float64 so a small probability is a finite log, in ``dtype``."""
    if domain == "simplex":
        with np.errstate(divide="ignore"):
            return np.log(np.asarray(arr, np.float64)).astype(dtype)
    return arr


def _assemble(cls: type, address: str, **fields: Any) -> Any:
    """A module from its field values, without its constructor, then its own checks."""
    module = object.__new__(cls)
    for f in dataclasses.fields(cls):
        if f.name in fields:
            value = fields[f.name]
        elif f.default_factory is not dataclasses.MISSING:
            value = f.default_factory()
        else:  # every decoder passes the fields that have no default
            value = f.default
        object.__setattr__(module, f.name, value)
    check = getattr(cls, "__check_init__", None)
    if check is not None:
        try:
            check(module)
        except Exception as exc:
            raise DescriptionError(f"{address} ({cls.__name__}): {exc}") from exc
    return module


def _linear(tree: Mapping[str, Any]) -> eqx.nn.Linear:
    weight = tree["weight"]
    bias = tree.get("bias")
    return _assemble(
        eqx.nn.Linear,
        "a linear layer",
        weight=weight,
        bias=bias,
        in_features=int(weight.shape[1]),
        out_features=int(weight.shape[0]),
        use_bias=bias is not None,
    )


def _layer_norm(tree: Mapping[str, Any]) -> eqx.nn.LayerNorm:
    weight = tree["weight"]
    bias = tree.get("bias")
    return _assemble(
        eqx.nn.LayerNorm,
        "a layer norm",
        shape=tuple(int(s) for s in weight.shape),
        eps=1e-3,
        use_weight=True,
        use_bias=bias is not None,
        weight=weight,
        bias=bias,
    )


def _tree_shapes(tree: Any) -> Any:
    if isinstance(tree, Mapping):
        return {k: _tree_shapes(v) for k, v in tree.items()}
    if isinstance(tree, list | tuple):
        return [_tree_shapes(v) for v in tree]
    return list(np.shape(tree))


def _record_shapes(record: Any) -> Any:
    if _is_leaf(record):
        return list(record.get("shape", []))
    if isinstance(record, Mapping):
        return {k: _record_shapes(v) for k, v in record.items()}
    return [_record_shapes(v) for v in record]


def _module_tree(family: str, module: Any) -> Any:
    """The array tree a network family's variable holds, from a drawn module."""
    if family == "Deterministic":
        cell = module.map
        tree = {"input_proj": _linear_tree(cell.input_proj), "gate": _linear_tree(cell.gate)}
        if cell.norms:
            tree["norms"] = [_norm_tree(n, "") for n in cell.norms]
        return tree
    if family == "LearnedVector":
        return module.value
    trunk = module.trunk if family in ("MLPCategorical", "PredictorHead") else module
    return _mlp_tree(trunk, "")


def _is_leaf(record: object) -> bool:
    """Whether a variable record is a leaf (a domain and a size) rather than a tree node."""
    return isinstance(record, Mapping) and bool(set(record) & set(_LEAF_KEYS))


def _parameter_names(parameters: object) -> list[str]:
    """Every variable a record's ``parameters`` names (nested records and lists included)."""
    if isinstance(parameters, str):
        return [parameters]
    if isinstance(parameters, Mapping):
        return [n for v in parameters.values() for n in _parameter_names(v)]
    if isinstance(parameters, list):
        return [n for v in parameters for n in _parameter_names(v)]
    return []


class _Init(NamedTuple):
    """An ``{"init": {...}}`` clamp: the constructor-recipe arguments of a draw."""

    args: Mapping[str, Any]


class _Reader:
    """The arrays file and the PRNG key one build draws from (shared by nested models)."""

    def __init__(self, arrays: Mapping[str, np.ndarray] | None, key: Array | None) -> None:
        self.arrays = dict(arrays or {})
        self.key = key

    def next_key(self, address: str) -> Array:
        if self.key is None:
            raise DescriptionError(
                f"{address} is clamped by an init recipe and build() was given no key= to draw it "
                "with; pass key=jax.random.PRNGKey(...) or give its values"
            )
        self.key, sub = jax.random.split(self.key)
        return sub

    def system(self, document: Mapping[str, Any], where: str) -> tuple[System, SystemParams]:
        return _Model(self, document, where).build()


class _Model:
    """One document (or a term's own model) being read: parse, derive, lower, construct."""

    def __init__(self, reader: _Reader, document: Mapping[str, Any], where: str) -> None:
        self.reader, self.doc, self.where = reader, document, where
        self.faults: list[str] = []
        self.lagged: dict[str, str] = {}  # modality → its datum's source at the previous tick

    def fail(self, sentence: str) -> DescriptionError:
        return DescriptionError(f"{self.where}{sentence}")

    @contextlib.contextmanager
    def noting(self):
        """One independent record's checks: a refusal is kept for :meth:`flush`, and the next
        record is checked."""
        try:
            yield
        except DescriptionError as exc:
            self.faults.append(str(exc))

    def flush(self) -> None:
        """Raise the faults a phase kept: one as its own sentence, several listed one per line,
        so a document whose records fail independently is corrected in one pass."""
        faults, self.faults = self.faults, []
        if len(faults) == 1:
            raise DescriptionError(faults[0])
        if faults:
            raise DescriptionError(
                f"{self.where}the document has {len(faults)} faults:\n"
                + "\n".join(f"- {f}" for f in faults)
            )

    # ------------------------------------------------------------------- parsing

    def build(self) -> tuple[System, SystemParams]:
        if self.where:
            unknown = sorted(set(self.doc) - set(_BODY_KEYS))
            if unknown:
                raise self.fail(
                    f"a term's model has key(s) {unknown}; its keys are {list(_BODY_KEYS)}"
                )
        self._variables()
        self._factors()
        self._relations()
        self._configuration()
        self._objective_records()
        self._kinds()
        self._dots()
        self._domains()
        self._slots()
        self._order()
        self._clamps()
        return self._construct()

    def _section(self, name: str, default: Any = None) -> Any:
        value = self.doc.get(name, default)
        if value is None:
            raise self.fail(f"the document has no {name!r} section")
        if not isinstance(value, Mapping):
            raise self.fail(f"{name} must be an object, got {type(value).__name__}")
        return value

    def _variables(self) -> None:
        self.variables: Mapping[str, Any] = self._section("variables")
        for name, record in self.variables.items():
            address = f"variables[{name!r}]"
            with self.noting():
                if not name or "[" in name or "]" in name:
                    raise self.fail(
                        f"{address}: a variable name is non-empty and contains no bracket (a "
                        "parent's step is written name[t] or name[t-1]); rename it"
                    )
                self._check_record(record, address)
        self.flush()

    def _check_record(self, record: object, address: str) -> None:
        if _is_leaf(record):
            unknown = sorted(set(record) - set(_LEAF_KEYS))
            if unknown:
                raise self.fail(
                    f"{address} has key(s) {unknown}; a variable record is a domain and a size: "
                    f"{list(_LEAF_KEYS)}"
                )
            domain = record.get("domain")
            if domain is not None and domain not in (*PER_STEP_DOMAINS, *PARAMETER_DOMAINS):
                raise self.fail(
                    f"{address}: unknown domain {domain!r}; the domains are "
                    f"{sorted({*PER_STEP_DOMAINS, *PARAMETER_DOMAINS})}"
                )
            for size in ("levels", "dim"):
                value = record.get(size)
                if value is not None and (not isinstance(value, int) or value < 1):
                    raise self.fail(f"{address}.{size} must be a positive integer, got {value!r}")
            if "levels" in record and domain != "categorical":
                raise self.fail(f"{address}: levels sizes a categorical domain; write that domain")
            if "dim" in record and (domain != "real" or "shape" in record):
                raise self.fail(
                    f"{address}: dim sizes a per-step real domain, and a parameter's size is its "
                    "shape; write one of them with domain 'real'"
                )
            if ("low" in record or "high" in record) and "dim" not in record:
                raise self.fail(f"{address}: low and high bound a real domain with its dim")
            if "dims" in record:
                self._check_dims(record, address)
            if "duration" in record:
                self._check_duration(record, address)
            shape = record.get("shape")
            if shape is not None and (
                not isinstance(shape, list) or not all(isinstance(s, int) and s >= 0 for s in shape)
            ):
                raise self.fail(f"{address}.shape must be a list of sizes, got {shape!r}")
            return
        if isinstance(record, Mapping) and record:
            for k, v in record.items():
                self._check_record(v, f"{address}.{k}")
            return
        if isinstance(record, list) and record:
            for i, v in enumerate(record):
                self._check_record(v, f"{address}[{i}]")
            return
        raise self.fail(
            f"{address} must be a variable record ({{'domain': ..., 'levels' | 'dim' | 'shape': "
            "...}) or a tree of them, got " + repr(record)
        )

    def _check_duration(self, record: Mapping[str, Any], address: str) -> None:
        """A categorical variable's ``duration``: a positive number, or one per level."""
        value = record["duration"]
        if "levels" not in record or "shape" in record:
            raise self.fail(
                f"{address}: duration gives a round's length per level of a discrete control "
                "node; write it beside that node's levels"
            )
        numbers = value if isinstance(value, list) else [value]
        if not numbers or not all(
            isinstance(x, int | float) and not isinstance(x, bool) and x > 0 for x in numbers
        ):
            raise self.fail(
                f"{address}.duration must be a positive number, or a list of positive numbers, "
                f"got {value!r}"
            )
        if isinstance(value, list) and len(value) != record["levels"]:
            raise self.fail(
                f"{address}.duration has {len(value)} number(s) for {record['levels']} level(s): "
                "give one per level, or one number for every level"
            )

    def _check_dims(self, record: Mapping[str, Any], address: str) -> None:
        """A real variable's ``dims``: one name per dimension, each a key segment."""
        names = record["dims"]
        if "dim" not in record or "low" in record or "high" in record:
            raise self.fail(
                f"{address}: dims names the dimensions of a latent's real domain; write it beside "
                "that dim"
            )
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise self.fail(f"{address}.dims must be a list of names, got {names!r}")
        if len(names) != record["dim"]:
            raise self.fail(
                f"{address}.dims has {len(names)} name(s) for dim {record['dim']}: name every "
                "dimension, in order"
            )
        try:
            _check_dim_names("<latent>", tuple(names), f"{address}.dims")
        except ValueError as exc:
            raise self.fail(str(exc)) from None

    def _factors(self) -> None:
        factors = self._section("factors", {})
        self.factors: list[_Factor] = []
        for child, block in factors.items():
            with self.noting():
                if child not in self.variables:
                    raise self.fail(
                        f"factors[{child!r}]: no variable of that name; declare it under variables"
                    )
                if not isinstance(block, Mapping):
                    raise self.fail(f"factors[{child!r}] must be an object keyed by label")
            if not isinstance(block, Mapping):
                continue
            for label, record in block.items():
                with self.noting():
                    self.factors.append(self._lower(child, label, record))
        self.flush()

    def _lower(self, child: str, label: str, record: object) -> _Factor:
        """One record as per-parent offsets and steps: the role label's implied time, explicit
        brackets as overrides, a generic label's every parent bracketed."""
        address = f"factors[{child!r}][{label!r}]"
        if not isinstance(record, Mapping):
            raise self.fail(f"{address} must be a factor record (an object)")
        unknown = sorted(set(record) - set(_RECORD_KEYS))
        if unknown:
            raise self.fail(
                f"{address} has key(s) {unknown}; a factor record has {list(_RECORD_KEYS)}"
            )
        family = record.get("family")
        if family not in FAMILIES:
            raise self.fail(f"{address}: unknown family {family!r}; the menu is {sorted(FAMILIES)}")
        if label in LABELS and label != "prior" and label not in FAMILIES[family].labels:
            raise self.fail(
                f"{address}: a {family} record is not written under the label {label!r}; it "
                f"appears under {list(FAMILIES[family].labels)}"
            )
        steps = record.get("steps")
        implied: int | None
        if label in _UNTIMED:
            if steps is not None:
                raise self.fail(
                    f"{address}: steps does not apply to a {label!r} record (its time is plan "
                    "time, or it has none); drop it"
                )
            implied = 0
        elif label in LABELS:
            implied, label_steps = _IMPLIED[label]
            if steps is not None and steps != label_steps:
                raise self.fail(
                    f"{address}: a {label!r} record applies at steps {label_steps!r}; drop steps, "
                    "or write the record under a generic label with every parent bracketed"
                )
            steps = label_steps
        else:
            implied = None
            if steps is None:
                steps = "all"
            elif steps not in STEPS:
                raise self.fail(f"{address}: steps must be one of {list(STEPS)}, got {steps!r}")
        raw = record.get("parents", [])
        if not isinstance(raw, list):
            raise self.fail(f"{address}.parents must be a list of variable names")
        parents: list[tuple[str, int]] = []
        for entry in raw:
            name, offset = _parse_parent(entry, f"{address}.parents")
            if offset is None:
                if implied is None:
                    raise self.fail(
                        f"{address}: under the generic label {label!r} every parent is bracketed; "
                        f"write {name}[t] (the child's step) or {name}[t-1] (the step before)"
                    )
                offset = implied
            if name not in self.variables:
                raise self.fail(
                    f"{address}.parents names {name!r}, which is not a variable; the variables "
                    f"are {list(self.variables)}"
                )
            if any(name == p for p, _ in parents):
                raise self.fail(f"{address}.parents names {name!r} twice")
            parents.append((name, offset))
        late = [p for p, o in parents if o == 1]
        if late and label in LABELS and label != "transition":
            when = {
                "initial": "at the first step",
                "terminal": "at the plan's end",
                "target": "at the plan's steps",
            }.get(label, "at the child's step")
            raise self.fail(
                f"{address}: a record under {label!r} reads its parents {when}; write {late[0]}"
            )
        if late and steps in ("0", "all"):
            raise self.fail(
                f"{address} applies at step 0 (steps {steps!r}) and reads {late[0]}[t-1], "
                'which the first step does not have; write steps "1+"'
            )
        for part in ("parameters", "options"):
            if not isinstance(record.get(part, {}), Mapping):
                raise self.fail(f"{address}.{part} must be an object")
        options = dict(record.get("options", {}))
        if label == "likelihood" and "datum" in options:
            #  the datum read from the carry, a source at the previous tick (S8)
            datum = options.pop("datum")
            source, offset = _parse_parent(datum, f"{address}.options.datum")
            if offset != 1 or source not in self.variables:
                raise self.fail(
                    f"{address}.options.datum is {datum!r}; a lagged datum names a state or an "
                    "action node at the previous tick, as name[t-1]"
                )
            self.lagged[child] = source
        return _Factor(
            child,
            label,
            family,
            tuple(parents),
            record.get("parameters", {}),
            options,
            None if label in _UNTIMED else steps,
            address,
        )

    def _check_target_parents(self, fac: _Factor, states: set[str], nodes: set[str]) -> None:
        """A ``target``'s parents are its conditioning variables, discrete control nodes or
        latents read at the same step, and a target with parents is written as a conditional
        family (``System.C_deps``; design: step-durations.md T5)."""
        conditional = ("ConditionalCategorical", "ConditionalGaussian")
        if not fac.parents:
            if fac.family in conditional:
                raise self.fail(
                    f"{fac.address}: a {fac.family} target is conditional on its parents, and "
                    "the record names none; write parents (a discrete control node or a latent "
                    "per leading axis, in order), or write an unconditional family"
                )
            return
        if fac.child in nodes:
            raise self.fail(
                f"{fac.address}: a target under the control node {fac.child!r} is its fixed "
                "cost per level and takes no parents; drop them"
            )
        bad = [p for p, _ in fac.parents if p not in states and p not in nodes]
        if bad:
            raise self.fail(
                f"{fac.address}: a target's parents are discrete control nodes or latents, and "
                f"{bad[0]!r} is neither"
            )
        if fac.family not in conditional:
            raise self.fail(
                f"{fac.address}: a target with parents is a conditional preference, a "
                f"ConditionalCategorical (a table over them) or a ConditionalGaussian (a "
                f"Gaussian per configuration), and the engine reads no conditional "
                f"{fac.family}; drop the parents or write one of those"
            )

    def _relations(self) -> None:
        relations = self._section("relations", {})
        unknown = sorted(set(relations) - set(_RELATIONS))
        if unknown:
            raise self.fail(f"relations has key(s) {unknown}; the relations are {list(_RELATIONS)}")
        controls = relations.get("controls", {})
        if not isinstance(controls, Mapping):
            raise self.fail("relations.controls maps each control node to its action node")
        self.controls: dict[str, str] = dict(controls)
        self.observed: list[str] = list(relations.get("observed", []))
        self.external: list[str] = list(relations.get("external", []))
        named: dict[str, str] = {}
        for what, names in (
            ("a control node in relations.controls", list(self.controls)),
            ("an action node in relations.controls", list(self.controls.values())),
            ("observed", self.observed),
            ("external", self.external),
        ):
            for name in names:
                if not isinstance(name, str) or name not in self.variables:
                    raise self.fail(f"relations names {name!r}, which is not a variable")
                if name in named:
                    raise self.fail(
                        f"relations names {name!r} as {named[name]} and as {what}; a variable "
                        "plays one of these roles"
                    )
                named[name] = what

    def _configuration(self) -> None:
        conf = self._section("configuration")
        unknown = sorted(set(conf) - set(_CONFIGURATION))
        if unknown:
            raise self.fail(
                f"configuration has sub-block(s) {unknown}; its sub-blocks are "
                f"{list(_CONFIGURATION)}"
            )
        for block, fields in _CONFIGURATION.items():
            given = conf.get(block, {})
            if not isinstance(given, Mapping):
                raise self.fail(f"configuration.{block} must be an object")
            extra = sorted(set(given) - set(fields))
            if extra:
                raise self.fail(
                    f"configuration.{block} has field(s) {extra}; its fields are {list(fields)} "
                    "(planner-specific settings and learning rates are call-time arguments)"
                )
        planning = conf.get("planning", {})
        inference, learning = conf.get("inference", {}), conf.get("learning", {})

        def count(value: object, where: str, what: str, allow_none: bool = False) -> object:
            if allow_none and value is None:
                return value
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise self.fail(
                    f"configuration.{where} must be a positive integer ({what}), got {value!r}"
                )
            return value

        horizon = count(planning.get("horizon"), "planning.horizon", "the plan length")
        planner = planning.get("planner", _DEFAULTS["planner"])
        if planner not in _planners():
            raise self.fail(
                f"configuration.planning.planner names {planner!r}; the registered planners are "
                f"{_planners()}"
            )
        gamma = planning.get("gamma", _DEFAULTS["gamma"])
        if isinstance(gamma, bool) or not isinstance(gamma, int | float):
            raise self.fail(
                f"configuration.planning.gamma is the policy precision, a number; got {gamma!r}"
            )
        precision = planning.get("precision")
        if precision is not None and (
            not isinstance(precision, str) or precision not in self.variables
        ):
            raise self.fail(
                "configuration.planning.precision names the parameter variable holding the "
                f"policy precision, declared under variables; got {precision!r}"
            )
        branch_on = planning.get("branch_on", [])
        if not isinstance(branch_on, list) or any(m not in self.observed for m in branch_on):
            raise self.fail(
                "configuration.planning.branch_on lists observed variables (relations.observed: "
                f"{self.observed}); got {branch_on!r}"
            )
        if branch_on and not planner_branches(planner):
            raise self.fail(
                f"configuration.planning.branch_on names {branch_on}, which only a branching "
                f"planner ('sophisticated', or one registered with branching=True) branches on, "
                f"and the planner is {planner!r}; drop branch_on or plan with 'sophisticated'"
            )
        belief = planning.get("belief", "prior")
        if belief not in _BELIEFS:
            raise self.fail(
                f"configuration.planning.belief is one of {list(_BELIEFS)} (the belief a plan's "
                f"information gain reads), got {belief!r}"
            )
        if belief == "posterior" and branch_on:
            raise self.fail(
                'configuration.planning.belief "posterior" conditions the plan on expected '
                "observations, and branch_on branches it on their outcomes; declare one of the two"
            )
        feedback = planning.get("feedback")
        if feedback is not None and feedback not in _FEEDBACK:
            raise self.fail(
                f"configuration.planning.feedback is one of {list(_FEEDBACK)}, or absent (open "
                f"loop), got {feedback!r}"
            )
        if feedback is not None and belief != "posterior":
            raise self.fail(
                f"configuration.planning.feedback {feedback!r} scores targets on the closed-loop "
                'state distribution, which needs belief "posterior"; write "belief": "posterior"'
            )
        schedule = inference.get("mixture_schedule", _DEFAULTS["mixture_schedule"])
        if schedule not in _SCHEDULES:
            raise self.fail(
                f"configuration.inference.mixture_schedule is one of {list(_SCHEDULES)}, got "
                f"{schedule!r}"
            )
        self.config = {
            "planner": planner,
            "policy_len": horizon,
            "gamma": float(gamma),
            "precision": precision,
            "branch_on": tuple(branch_on),
            "planning_belief": belief,
            "feedback": feedback,
            "num_iters": count(
                inference.get("iterations", _DEFAULTS["num_iters"]),
                "inference.iterations",
                "fixed-point iterations",
            ),
            "mixture_k": count(
                inference.get("mixture_k"), "inference.mixture_k", "the mixture budget", True
            ),
            "mixture_schedule": schedule,
            "reduction_period": count(
                learning.get("reduction_period"),
                "learning.reduction_period",
                "learn calls between reductions",
                True,
            ),
            "reduction_pairs": count(
                learning.get("reduction_pairs", _DEFAULTS["reduction_pairs"]),
                "learning.reduction_pairs",
                "candidate pairs per reduction",
            ),
        }
        frozen = learning.get("frozen", [])
        if not isinstance(frozen, list) or not all(isinstance(v, str) for v in frozen):
            raise self.fail("configuration.learning.frozen must be a list of variable names")
        for name in frozen:
            if name not in self.variables:
                raise self.fail(
                    f"configuration.learning.frozen names {name!r}, which is not a variable"
                )
        self.frozen = set(frozen)

    def _objective_records(self) -> None:
        objective = self.doc.get("objective")
        if objective is None:
            objective = {"decomposition": "utility_infogain"}
        if not isinstance(objective, Mapping):
            raise self.fail("objective must be an object")
        unknown = sorted(set(objective) - {"decomposition", "terms"})
        if unknown:
            raise self.fail(f"objective has key(s) {unknown}; its keys are decomposition and terms")
        terms = objective.get("terms", [])
        if not isinstance(terms, list):
            raise self.fail("objective.terms must be a list of term records")
        for i, record in enumerate(terms):
            address = f"objective.terms[{i}]"
            if not isinstance(record, Mapping) or not isinstance(record.get("term"), str):
                raise self.fail(f"{address} must be a term record with a 'term' name")
            unknown = sorted(set(record) - set(_TERM_KEYS))
            if unknown:
                raise self.fail(
                    f"{address} has key(s) {unknown}; a term record has {list(_TERM_KEYS)}"
                )
            on = record.get("on")
            if on is not None and (
                not isinstance(on, str)
                or ("*" not in on and on.partition(":")[0] not in self.variables)
            ):  # a variable, or a dimension of one (``pose:b1``)
                raise self.fail(f"{address}.on names {on!r}, which is not a variable")
            for name in _parameter_names(record.get("parameters", {})):
                if name not in self.variables:
                    raise self.fail(f"{address}.parameters names {name!r}, which is not a variable")
        self.objective, self.terms = objective, terms

    # ------------------------------------------------------------ kinds and roles

    def _kinds(self) -> None:
        """Per-step against static, derived from use (D4 as amended): a variable named in some
        ``parameters`` is static, a child or parent of a per-step record is per-step."""
        static: dict[str, str] = {}
        per_step: dict[str, str] = {}
        for fac in self.factors:
            if fac.label == "prior":
                if fac.child in self.controls:
                    raise self.fail(
                        f"{fac.address}: a per-step prior on the control node {fac.child!r} is a "
                        "factor the engine does not score; drop the "
                        "record (a prior over the whole plan is the policy_prior record)"
                    )
                if "prior" not in FAMILIES[fac.family].labels:
                    raise self.fail(
                        f"{fac.address}: a {fac.family} record is not written under the label "
                        "'prior'; a parameter's prior is one of "
                        f"{[f for f, spec in FAMILIES.items() if 'prior' in spec.labels]}"
                    )
                static.setdefault(fac.child, fac.address)
            else:
                per_step.setdefault(fac.child, fac.address)
            for parent, _ in fac.parents:
                per_step.setdefault(parent, fac.address)
            for name in _parameter_names(fac.parameters):
                static.setdefault(name, f"{fac.address}.parameters")
        for i, record in enumerate(self.terms):
            for name in _parameter_names(record.get("parameters", {})):
                static.setdefault(name, f"objective.terms[{i}].parameters")
        if self.config["precision"] is not None:
            static.setdefault(self.config["precision"], "configuration.planning.precision")
        for name in [*self.controls, *self.controls.values(), *self.observed, *self.external]:
            per_step.setdefault(name, "relations")
        for u, a in self.controls.items():
            held = [f for f in self.factors if f.child == a and f.label in LABELS]
            held = [f for f in held if f.label != "actuator"]
            if held:
                raise self.fail(
                    f"relations.controls pairs {u!r} with {a!r}, and the action node of {u!r} is "
                    f"a variable whose one record is an actuator, while {a!r} carries a record "
                    f"under {held[0].label!r}; pair {u!r} with an action variable of its own"
                )
        for name in self.variables:
            if name in static and name in per_step:
                raise self.fail(
                    f"variable {name!r} is static where {static[name]} names it and per-step "
                    f"where {per_step[name]} uses it; a variable is one or the other"
                )
            if name not in static and name not in per_step:
                raise self.fail(
                    f"variable {name!r} is used by no factor, relation or term; drop it, or name "
                    "it where it belongs"
                )
        self.static = [v for v in self.variables if v in static]
        self.per_step = [v for v in self.variables if v in per_step]
        for name in self.per_step:
            record = self.variables[name]
            if (
                not _is_leaf(record)
                or record.get("domain") not in PER_STEP_DOMAINS
                or (record.get("domain") == "categorical" and "levels" not in record)
                or (record.get("domain") == "real" and "dim" not in record)
            ):
                raise self.fail(
                    f"variables[{name!r}]: {name!r} is per-step (used at {per_step[name]}), so its "
                    "record is {'domain': 'categorical', 'levels': n} or {'domain': 'real', "
                    "'dim': d}"
                )
        labels: dict[str, set[str]] = {}
        for fac in self.factors:
            labels.setdefault(fac.child, set()).add(fac.label)
        roles = {*self.controls, *self.controls.values(), *self.observed, *self.external}
        self.nodes = [v for v in self.per_step if v in self.controls]
        self.actions = [self.controls[u] for u in self.nodes]
        self.mods = [v for v in self.per_step if v in self.observed]
        self.aux = [
            v
            for v in self.per_step
            if v not in roles and labels.get(v) and labels[v] <= {"recognition"}
        ]
        self.states = [v for v in self.per_step if v not in roles and v not in self.aux]

    def _dots(self) -> None:
        """The dot is reserved to composite prefixes (Q18.2): dotted per-step names make the
        document a composite whose every per-step variable belongs to a component."""
        dotted = [v for v in self.per_step if _SEP in v]
        plain = [v for v in self.per_step if _SEP not in v]
        prefixes = {v.rsplit(_SEP, k)[0] for v in dotted for k in range(1, v.count(_SEP) + 1)}
        if dotted and plain:
            raise self.fail(
                f"variable(s) {dotted[:3]} are dotted, which makes the document a composite whose "
                f"variables each belong to a component ('<component>.<name>'), and {plain[:3]} "
                "carry no component; the dot is reserved to composite prefixes, so rename "
                "(model-description-0.3.md Q18.2)"
            )
        for name in self.static:
            if _SEP in name and name.rpartition(_SEP)[0] not in prefixes:
                raise self.fail(
                    f"variable {name!r} is dotted, and the dot is reserved to composite prefixes: "
                    f"{name.rpartition(_SEP)[0]!r} is not a component of this document; rename it"
                )
        self.components = self._component_order(dotted)
        for fac in self.factors:
            if fac.label == "prior" or _SEP not in fac.child:
                continue
            home = fac.child.rpartition(_SEP)[0]
            for name in _parameter_names(fac.parameters):
                if _SEP in name and name.rpartition(_SEP)[0] != home:
                    raise self.fail(
                        f"{fac.address} names {name!r}, a parameter of the component "
                        f"{name.rpartition(_SEP)[0]!r}, and its child belongs to {home!r}; a "
                        "parameter belongs to its factor's component"
                    )

    def _component_order(self, dotted: list[str]) -> list[str]:
        """The component names a composite's dotted variables imply, each parent before its
        nested components, in first-appearance order (the document's variable order)."""
        tree: dict[str, Any] = {}
        for name in dotted:
            node = tree
            for part in name.split(_SEP)[:-1]:
                node = node.setdefault(part, {})
        out: list[str] = []

        def walk(node: Mapping[str, Any], prefix: str) -> None:
            for part, children in node.items():
                out.append(f"{prefix}{part}")
                walk(children, f"{prefix}{part}{_SEP}")

        walk(tree, "")
        return out

    def _domains(self) -> None:
        """Sizes of the per-step variables; the control nodes' boxes; the action nodes' match."""
        self.width: dict[str, int] = {}
        self.levels: dict[str, int | None] = {}
        for name in self.per_step:
            record = self.variables[name]
            if record["domain"] == "categorical":
                shape = record.get("shape", [])
                self.levels[name] = None if shape else record["levels"]
                self.width[name] = record["levels"] * math.prod(shape)
            else:
                self.levels[name] = None
                self.width[name] = record["dim"]
                if ("low" in record or "high" in record) and name not in self.controls:
                    raise self.fail(
                        f"variables[{name!r}]: bounds (low, high) give a continuous control "
                        f"node's box, and {name!r} is not a control node; drop them"
                    )
        self.durations: dict[str, Array] = {}
        for name in self.per_step:
            record = self.variables[name]
            if "duration" not in record:
                continue
            if name not in self.controls:
                raise self.fail(
                    f"variables[{name!r}]: duration is a discrete control node's round length "
                    f"per level, and {name!r} is not a control node; drop it"
                )
            self.durations[name] = jnp.broadcast_to(
                jnp.asarray(record["duration"], jnp.float32), (record["levels"],)
            )
        if len(self.durations) > 1:
            raise self.fail(
                f"variables {sorted(self.durations)} all declare a duration: a round steps every "
                "factor once, so it has one clock; declare it on one control node"
            )
        self.boxes: dict[str, Box] = {}
        for u, a in zip(self.nodes, self.actions, strict=True):
            record, target = self.variables[u], self.variables[a]
            if record["domain"] == "real":
                if "low" not in record or "high" not in record:
                    raise self.fail(
                        f"variables[{u!r}]: a continuous control node's domain carries its box: "
                        "give low and high (a number per dimension, or one number for all)"
                    )
                dim = (record["dim"],)
                try:
                    low = np.broadcast_to(np.asarray(record["low"], np.float32), dim)
                    high = np.broadcast_to(np.asarray(record["high"], np.float32), dim)
                    self.boxes[u] = Box(jnp.asarray(low), jnp.asarray(high))
                except Exception as exc:
                    raise self.fail(f"variables[{u!r}]: its bounds are not a box: {exc}") from exc
            size = ("levels", "dim")[record["domain"] == "real"]
            if target["domain"] != record["domain"] or target.get(size) != record.get(size):
                raise self.fail(
                    f"variables[{a!r}] is the action node of {u!r}, and its domain "
                    f"({target['domain']}, {size} {target.get(size)}) differs from the control "
                    f"node's ({record['domain']}, {size} {record.get(size)}); an action node "
                    "has its control node's domain and size"
                )
        for name in self.external:
            if self.variables[name]["domain"] != "real":
                raise self.fail(
                    f"variables[{name!r}]: an external input is a vector, domain 'real' with dim"
                )

    # ----------------------------------------------------------------- the slots

    def _slots(self) -> None:
        """Every record mapped to the engine slot it lowers to, with the parent checks of
        each slot and family."""
        self.initial: dict[str, _Factor] = {}
        self.transition: dict[str, _Factor] = {}
        self.likelihood: dict[str, _Factor] = {}
        self.actuator: dict[str, _Factor] = {}
        self.targets: dict[str, _Factor] = {}
        self.terminals: dict[str, _Factor] = {}
        self.recognition: dict[str, _Factor] = {}
        self.priors: dict[str, _Factor] = {}
        self.policy_prior: _Factor | None = None
        self.habit: _Factor | None = None
        states, nodes = set(self.states), set(self.nodes)
        first = self.nodes[0] if self.nodes else None
        for fac in self.factors:
            misplaced = [p for p, _ in fac.parents if p in self.external]
            if misplaced and fac.label != "recognition":
                raise self.fail(
                    f"{fac.address} reads the external input(s) {misplaced}; an external input "
                    "is read by recognition records only"
                )

        def put(table: dict, fac: _Factor, what: str) -> None:
            if fac.child in table:
                raise self.fail(
                    f"{fac.address}: {fac.child!r} already has {what} "
                    f"({table[fac.child].address}); a variable takes one"
                )
            table[fac.child] = fac

        for fac in self.factors:
            child, label = fac.child, fac.label
            if child in self.static:  # its one record is its prior (_kinds)
                if fac.parents:
                    raise self.fail(f"{fac.address}: a prior on a parameter takes no parents")
                put(self.priors, fac, "a prior record")
            elif label in ("target", "terminal"):
                if (
                    child not in states
                    and child not in nodes
                    and not (label == "target" and child in self.observed)
                ):
                    where = (
                        "a latent, an observation or a control node"
                        if label == "target"
                        else "a latent or a control node"
                    )
                    raise self.fail(
                        f"{fac.address}: a {label} sits under {where}, and {child!r} is none of "
                        f"these; write it under the variable it scores"
                    )
                if label == "target":
                    self._check_target_parents(fac, states, nodes)
                    put(self.targets, fac, "a target")
                    continue
                if fac.parents and child not in states:
                    raise self.fail(
                        f"{fac.address}: a terminal under the control node {child!r} is "
                        "unconditional; drop its parents"
                    )
                bad = [p for p, _ in fac.parents if p not in states]
                if bad:
                    raise self.fail(
                        f"{fac.address}: a terminal's parents are latents, and {bad[0]!r} is not "
                        "one"
                    )
                if fac.parents and fac.family != "ConditionalCategorical":
                    raise self.fail(
                        f"{fac.address}: a terminal with parents is a table over them "
                        f"(ConditionalCategorical), and the engine reads no conditional "
                        f"{fac.family}; drop the parents or write the table"
                    )
                put(self.terminals, fac, "a terminal")
            elif child in nodes:
                if label == "policy_prior" or label == "recognition":
                    if child != first:
                        raise self.fail(
                            f"{fac.address}: the {label} record over the plan sits under the "
                            f"first control node, {first!r}"
                        )
                    if label == "policy_prior":
                        if fac.parents:
                            raise self.fail(f"{fac.address}: the policy prior takes no parents")
                        self.policy_prior = fac
                    else:
                        if [p for p, _ in fac.parents] != self.states:
                            raise self.fail(
                                f"{fac.address}: a control node's recognition record (the policy "
                                "module) reads the beliefs over every latent, in variable order: "
                                f"write parents {self.states}"
                            )
                        self.habit = fac
                    continue
                raise self.fail(
                    f"{fac.address}: {child!r} is a control node, chosen rather than generated; it "
                    "takes a policy_prior or recognition record (under the first control node), "
                    "a target or a terminal, and no other"
                )
            elif label == "policy_prior":
                raise self.fail(
                    f"{fac.address}: the policy prior sits under the first control node"
                )
            elif label == "recognition":
                if child not in states and child not in self.aux:
                    raise self.fail(
                        f"{fac.address}: a recognition record sits under a latent or an auxiliary "
                        f"variable, and {child!r} is observed, an action node or an input"
                    )
                put(self.recognition, fac, "a recognition record")
            elif child in self.actions:
                if fac.steps != "all" or not (label == "actuator" or label not in LABELS):
                    raise self.fail(
                        f"{fac.address}: {child!r} is an action node, whose one record is its "
                        "actuator"
                    )
                u = self.nodes[self.actions.index(child)]
                if fac.parents != ((u, 0),):
                    raise self.fail(
                        f"{fac.address}: an actuator reads its control node at the same step: "
                        f"write parents [{u!r}]"
                    )
                put(self.actuator, fac, "an actuator")
            elif child in self.observed:
                if fac.steps != "all" or not (label == "likelihood" or label not in LABELS):
                    raise self.fail(
                        f"{fac.address}: {child!r} is observed, and an observation takes a "
                        "likelihood (every step) and a target"
                    )
                bad = [
                    f"{p}[t-1]" if o else p
                    for p, o in fac.parents
                    if o or (p not in states and p not in self.actions)
                ]
                if bad or not fac.parents:
                    raise self.fail(
                        f"{fac.address}: a likelihood reads latents or action nodes at the "
                        f"observation's step; {bad or 'it reads none'} cannot be read"
                    )
                put(self.likelihood, fac, "a likelihood")
            elif child in self.external:
                raise self.fail(
                    f"{fac.address}: {child!r} is an external input, data the System receives; it "
                    "takes no record"
                )
            else:  # a latent: every other per-step variable is handled above
                if label in ("likelihood", "actuator"):
                    raise self.fail(
                        f"{fac.address}: a {label} sits under an observed variable or an action "
                        f"node, and {child!r} is a latent; list it in relations.observed if it is "
                        "observed"
                    )
                if fac.steps == "0":
                    put(self.initial, fac, "an initial record (step 0)")
                elif fac.steps == "1+":
                    put(self.transition, fac, "a transition (every step after the first)")
                else:
                    raise self.fail(
                        f"{fac.address}: a latent's conditional at the first step is its initial "
                        "record and at later steps its transition; write two records, steps "
                        '"0" and "1+" (a code\'s initial read off its own transition is the '
                        "DerivedInitial family)"
                    )
        for s in self.states:
            for table, what in ((self.initial, "initial"), (self.transition, "transition")):
                if s not in table:
                    raise self.fail(
                        f"the latent {s!r} has no {what} record; every latent has an initial and "
                        "a transition (an observed variable is listed in relations.observed)"
                    )
        for o in self.observed:
            if o not in self.likelihood:
                raise self.fail(
                    f"{o!r} is observed (relations.observed) and has no likelihood record; add one "
                    "or drop it from observed"
                )
        for fac in self.initial.values():
            bad = [p for p, _ in fac.parents if p not in states]
            if bad:
                raise self.fail(
                    f"{fac.address}: an initial record's parents are latents, and {bad[0]!r} is "
                    "not one"
                )
            if fac.parents and fac.family != "ConditionalCategorical":
                raise self.fail(
                    f"{fac.address}: an initial record with parents is a table over them "
                    f"(ConditionalCategorical), and the engine reads no conditional "
                    f"{fac.family} prior; drop the parents or write the table"
                )
        for fac in self.initial.values():
            if fac.family == "MixtureGaussian" and self.config["mixture_k"] is None:
                raise self.fail(
                    f"{fac.address}: a MixtureGaussian initial makes {fac.child!r} "
                    "mixture-carried, so configuration.inference.mixture_k (the shared mixture "
                    "budget) is required"
                )
        self._amortized_names()
        for u, a in zip(self.nodes, self.actions, strict=True):
            for fac in self.transition.values():
                if any(p == a for p, _ in fac.parents):
                    raise self._reads_action(fac, u, a)
            if not any(u == p for fac in self.transition.values() for p, _ in fac.parents):
                raise self.fail(
                    f"the control node {u!r} is read by no transition; a control node moves "
                    "at least one latent, so list it among a transition's parents"
                )
        for fac in self.transition.values():
            self._check_transition(fac)
        for fac in self.recognition.values():
            for p, o in fac.parents:
                if o or p in nodes or p in self.actions:
                    raise self.fail(
                        f"{fac.address}: a recognition record reads latents (their prediction), "
                        "observations, auxiliary variables and inputs at the child's step; "
                        f"{p!r} cannot be read"
                    )
                if p in states and p in self.recognition:
                    raise self.fail(
                        f"{fac.address}: its parent {p!r} is itself an amortized posterior (it "
                        "has a recognition record); a recognition record reads a latent's "
                        "prediction, so a parent may not be another amortized posterior"
                    )
        self.rec_order = self._recognition_order()

    def _amortized_names(self) -> None:
        """On the amortized path the engine reads a control node by its pymdp-style name: the
        node of a latent ``f`` whose transition reads it is ``u_<f>``, its action ``a_<f>``."""
        amortized = bool(self.recognition) or any(
            fac.family in _AMORTIZED_FAMILIES for fac in self.factors
        )
        if not amortized:
            return
        for u, a in zip(self.nodes, self.actions, strict=True):
            readers = [
                s for s, fac in self.transition.items() if any(p == u for p, _ in fac.parents)
            ]
            pairs = {control_name(f): action_name(f) for f in readers}
            if pairs.get(u) != a:
                want = ", ".join(f"{{{k!r}: {v!r}}}" for k, v in pairs.items()) or "none"
                raise self.fail(
                    f"relations.controls pairs {u!r} with {a!r}, and on the amortized path (a "
                    "model with recognition records or network families) a control node is named "
                    "u_<f> after the latent f whose transition reads it and its action node "
                    f"a_<f>; write {want}"
                )

    def _check_transition(self, fac: _Factor) -> None:
        """A transition's parents against what its family reads: its own previous value, its
        other latents at the step before or the child's step, its control nodes at the step
        they are chosen."""
        family, child = fac.family, fac.child
        self_reader = FAMILIES[family].reads_self
        same_step = FAMILIES[family].reads_same_step
        names = [p for p, _ in fac.parents]
        if self_reader and (child not in names or names[0] != child):
            raise self.fail(
                f"{fac.address}: a {family} transition reads {child!r}'s own previous value; list "
                f"{child!r} first among its parents"
            )
        for p, o in fac.parents:
            if p == child and o == 0:
                raise self.fail(
                    f"{fac.address}: a transition reads its own child at the step before, never at "
                    f"the child's step; write {child}[t-1] (or {child})"
                )
            if p == child:
                if family == "MLPCategorical":
                    raise self.fail(
                        f"{fac.address}: a code reads its own past through its other parents; it "
                        "cannot name itself"
                    )
                continue
            if p in self.nodes:
                if o != 1:
                    raise self.fail(
                        f"{fac.address}: a transition reads the control node {p!r} at the step it "
                        f"is chosen, the step before the child: write {p} (or {p}[t-1])"
                    )
            elif p in self.states:
                if family in ("MLPConditional", "NonlinearGaussianTransition"):
                    article = "an" if family == "MLPConditional" else "a"
                    raise self.fail(
                        f"{fac.address}: {article} {family} transition reads its own previous "
                        f"value and control nodes only, not {p!r}"
                    )
                want = 0 if same_step else 1
                if o != want:
                    form = f"{p}[t]" if want == 0 else p
                    raise self.fail(
                        f"{fac.address}: a {family} transition reads the latent {p!r} at "
                        f"{'the child' if want == 0 else 'the previous'} step: write {form}"
                    )
            elif p in self.actions:
                raise self._reads_action(fac, self.nodes[self.actions.index(p)], p)
            else:
                raise self.fail(
                    f"{fac.address}: a transition reads latents and control nodes; {p!r} is neither"
                )

    def _reads_action(self, fac: _Factor, u: str, a: str) -> DescriptionError:
        return self.fail(
            f"{fac.address}: {a!r} is the action node of {u!r}; a transition reads the control "
            f"node, so write {u!r} in its place (the action node is read by likelihoods and by "
            "its own actuator record)"
        )

    def _recognition_order(self) -> list[str]:
        """The evaluation order of the recognition records, derived from their parents: each
        auxiliary parent first, in document order otherwise; a cycle is refused."""
        aux = set(self.aux)
        needs = {v: [p for p, _ in fac.parents if p in aux] for v, fac in self.recognition.items()}
        placed, stuck = _evaluation_order(list(self.recognition), needs)
        if stuck:
            cycle = _cycle(stuck, needs)
            raise self.fail(
                f"the recognition records of {cycle} read one another in a cycle "
                f"({' -> '.join([*cycle, cycle[0]])}); an auxiliary variable is inferred before "
                "what reads it"
            )
        return placed

    def _order(self) -> None:
        """Same-step reads among latents (an initial's parents, a transition's ``[t]`` parents)
        run in variable order, which is the node order: a cycle is refused, and so is an order
        that puts a parent after its child, naming a valid order."""
        edges: list[tuple[str, str, str]] = []
        for fac in [*self.initial.values(), *self.transition.values()]:
            for p, o in fac.parents:
                if o == 0 and p in self.states and p != fac.child:
                    edges.append((p, fac.child, fac.address))
        index = {s: i for i, s in enumerate(self.states)}
        late = [(p, c, a) for p, c, a in edges if index[p] > index[c]]
        if not late:
            return
        incoming = {s: {p for p, c, _ in edges if c == s} for s in self.states}
        order: list[str] = []
        while len(order) < len(self.states):
            ready = [s for s in self.states if s not in order and incoming[s] <= set(order)]
            if not ready:
                stuck = [s for s in self.states if s not in order]
                raise self.fail(
                    f"the latents {stuck} read one another at the same step in a cycle; a "
                    "same-step read runs parent before child"
                )
            order.append(ready[0])
        p, c, address = late[0]
        raise self.fail(
            f"{address} reads {p!r} at the child's step, so {p!r} must be declared before "
            f"{c!r} (the variable order is the node order); a valid order of the latents is "
            f"{order}"
        )

    # ------------------------------------------------------------------ the clamps

    def _clamps(self) -> None:
        self.clamps: Mapping[str, Any] = self._section("clamps", {})
        for name in self.clamps:
            with self.noting():
                if name not in self.variables:
                    raise self.fail(f"clamps[{name!r}]: no variable of that name")
                if name not in self.static:
                    raise self.fail(
                        f"clamps[{name!r}]: {name!r} is per-step, and a clamp gives a parameter's "
                        "value (a per-step variable's data arrives at run time)"
                    )
        for name in self.static:
            has_prior, has_clamp = name in self.priors, name in self.clamps
            with self.noting():
                if has_prior and has_clamp:
                    raise self.fail(
                        f"variable {name!r} has a clamp and a prior record; a parameter is either "
                        "fixed (a clamp) or learned (a prior)"
                    )
                if not has_prior and not has_clamp:
                    raise self.fail(
                        f"parameter variable {name!r} has neither a clamp nor a prior record; "
                        f"give its value under clamps, or a prior record under factors[{name!r}]"
                    )
        self.flush()
        self.owner: dict[str, str] = {}
        sources = [(fac.address, fac.parameters) for fac in self.factors] + [
            (f"objective.terms[{i}]", rec.get("parameters", {})) for i, rec in enumerate(self.terms)
        ]
        for address, parameters in sources:
            refs = {v for k, v in parameters.items() if k == "row_cov" and isinstance(v, str)}
            for name in _parameter_names(parameters):
                if name in refs:
                    continue
                if self.owner.get(name) == address:
                    raise self.fail(f"{address} names {name!r} twice; a variable fills one field")
                if name in self.owner:
                    raise self.fail(
                        f"{address} names {name!r}, which {self.owner[name]} names already; a "
                        "parameter variable belongs to one factor"
                    )
                self.owner[name] = address

    # -------------------------------------------------------------------- values

    def _array(self, clamp: Any, shape, domain: str, address: str, dtype=None) -> np.ndarray:
        """One clamp leaf as an array: inline (the build's default dtype, a scalar broadcast to
        the declared shape) or a ref into the arrays file (its own dtype)."""
        if isinstance(clamp, Mapping):
            if set(clamp) == {"ref"} and isinstance(clamp["ref"], str):
                ref = clamp["ref"]
                _, sep, key = ref.rpartition("#")
                if not sep or not key:
                    raise self.fail(
                        f"{address}: a ref is written {{'ref': '<file>#<key>'}}, got {ref!r}"
                    )
                if key not in self.reader.arrays:
                    raise self.fail(
                        f"{address} refers to {ref!r}, and build() was given no arrays= holding "
                        f"the key {key!r}; load the file the document names in arrays_file and "
                        "pass it as arrays=, or describe with arrays='inline'"
                    )
                arr = np.asarray(self.reader.arrays[key])
            elif set(clamp) == {"init"}:
                raise self.fail(
                    f"{address}: an init recipe draws a network's weights or a learned vector, "
                    "not this parameter; give its value"
                )
            else:
                raise self.fail(
                    f"{address}: a clamp is a value, {{'ref': '<file>#<key>'}} or "
                    "{'init': {...}}; got an object with keys " + str(sorted(clamp))
                )
        else:
            try:
                arr = np.asarray(clamp, dtype=dtype or _default_dtype())
            except (TypeError, ValueError) as exc:
                raise self.fail(
                    f"{address}: the clamp is not a number or a nested list of numbers"
                ) from exc
            if arr.ndim == 0 and shape:
                arr = np.broadcast_to(arr, tuple(shape)).copy()
        if shape is not None and tuple(arr.shape) != tuple(shape):
            raise self.fail(
                f"{address}: the clamp has shape {list(arr.shape)} and the variable declares "
                f"{list(shape)}"
            )
        return arr

    def _domain_of(self, var: str, family: str, field: str, address: str) -> str:
        field_domain = _domain(family, field)
        declared = self.variables[var].get("domain") if _is_leaf(self.variables[var]) else None
        if declared is None or declared == field_domain:
            return field_domain
        if declared in _TIGHTER.get(field_domain, ()) or declared in _LOOSER.get(
            (family, field), ()
        ):
            return declared
        raise self.fail(
            f"{address}: the variable {var!r} declares the domain {declared!r} and its field "
            f"takes {field_domain!r}; write {field_domain!r} or drop the domain"
        )

    def value(self, var: str, family: str, field: str, address: str, axis: int = -1):
        """A parameter variable's value: its clamp as an array, or :class:`_Learned` when a
        prior record learns it."""
        domain = self._domain_of(var, family, field, address)
        if var in self.priors:
            prior = self.priors[var]
            return _Learned(var, prior.family, prior.parameters, prior.address)
        record = self.variables[var]
        if not _is_leaf(record) or "shape" not in record:
            raise self.fail(
                f"variables[{var!r}]: a parameter variable's record gives its shape "
                "({'domain': ..., 'shape': [...]})"
            )
        clamp = self.clamps[var]
        inline = not isinstance(clamp, Mapping)
        arr = self._array(
            clamp,
            record["shape"],
            domain,
            f"clamps[{var!r}]",
            dtype=np.float64 if domain == "simplex" and inline else None,
        )
        _check_domain(domain, arr, f"clamps[{var!r}]", axis)
        kept = arr.dtype if not inline and np.issubdtype(arr.dtype, np.floating) else None
        return jnp.asarray(_lowered(domain, arr, kept or _default_dtype()))

    def hyper(self, learned: _Learned, field: str, required: bool = True):
        """A prior's hyperparameter, by its field (``None`` when an optional one is absent)."""
        var = learned.parameters.get(field)
        if var is None:
            if required:
                raise self.fail(f"{learned.address}: a {learned.family} prior needs {field!r}")
            return None
        if not isinstance(var, str) or var not in self.variables:
            raise self.fail(f"{learned.address}.parameters.{field} must name a variable")
        if var in self.priors:
            raise self.fail(
                f"{learned.address}: the hyperparameter {var!r} has a prior of its own, and the "
                "engine learns one level of parameters; clamp it"
            )
        return self.value(var, learned.family, field, f"{learned.address}.parameters.{field}")

    def _check_prior(self, learned: _Learned, allowed: tuple[str, ...], where: str) -> None:
        if learned.family not in allowed:
            raise self.fail(
                f"{learned.address}: the {learned.family} prior cannot learn {where}; it takes "
                f"{list(allowed)}"
            )
        family = FAMILIES[learned.family]
        unknown = sorted(set(learned.parameters) - set(family.parameters))
        if unknown:
            raise self.fail(
                f"{learned.address}: {learned.family} has no parameter(s) {unknown}; its "
                f"parameters are {list(family.parameters)}"
            )

    def _shape(self, learned: _Learned, var_field: str, value) -> None:
        declared = self.variables[learned.var].get("shape")
        if declared is not None and list(np.shape(value)) != list(declared):
            raise self.fail(
                f"{learned.address}: the prior's {var_field} has shape {list(np.shape(value))} "
                f"and the variable {learned.var!r} declares {declared}"
            )

    def network(self, var: str, address: str):
        """A network's tree variable: its tree of arrays, or the :class:`_Init` recipe."""
        if var in self.priors:
            raise self.fail(
                f"{self.priors[var].address}: a network's weights are clamped values a trainer "
                "moves, and take no prior record"
            )
        clamp = self.clamps[var]
        if isinstance(clamp, Mapping) and "init" in clamp:
            if set(clamp) != {"init"} or not isinstance(clamp["init"], Mapping):
                raise self.fail(f"clamps[{var!r}]: an init clamp is {{'init': {{...}}}}")
            return _Init(clamp["init"])
        record = self.variables[var]
        return self._tree(clamp, None if _is_leaf(record) else record, f"clamps[{var!r}]")

    def _tree(self, clamp: Any, record: Any, address: str) -> Any:
        if record is not None and _is_leaf(record):
            domain = record.get("domain")
            if domain not in (None, "real"):
                raise self.fail(
                    f"{address}: a network's leaf is real, and its record declares the domain "
                    f"{domain!r}; write 'real' or drop it"
                )
            return jnp.asarray(self._array(clamp, record.get("shape"), "real", address))
        if isinstance(clamp, Mapping) and not ({"ref", "init"} & set(clamp)):
            if record is not None and (
                not isinstance(record, Mapping) or set(record) != set(clamp)
            ):
                raise self.fail(
                    f"{address}: the clamp's tree {sorted(clamp)} differs from the variable's "
                    f"{sorted(record) if isinstance(record, Mapping) else record!r}"
                )
            return {
                k: self._tree(v, None if record is None else record[k], f"{address}.{k}")
                for k, v in clamp.items()
            }
        if isinstance(clamp, list) and clamp and all(isinstance(v, Mapping) for v in clamp):
            if record is not None and (not isinstance(record, list) or len(record) != len(clamp)):
                raise self.fail(f"{address}: the clamp's tree differs from the variable's")
            return [
                self._tree(v, None if record is None else record[i], f"{address}[{i}]")
                for i, v in enumerate(clamp)
            ]
        return jnp.asarray(self._array(clamp, None, "real", address))

    # ------------------------------------------------------------------ decoding

    def decode(self, fac: _Factor, role: str, slot: tuple[str, str] | None = None) -> Any:
        """The engine module one lowered record declares, its layout derived from the wiring
        and the variables' sizes (D8)."""
        spec = FAMILIES[fac.family]
        unknown = sorted(set(fac.parameters) - set(spec.parameters))
        if unknown:
            raise self.fail(
                f"{fac.address}: {fac.family} has no parameter(s) {unknown}; its parameters are "
                f"{list(spec.parameters)}"
            )
        missing = [p for p in spec.parameters if p not in fac.parameters and p not in spec.optional]
        if missing and not (fac.family == "Deterministic"):
            raise self.fail(f"{fac.address}: {fac.family} needs the parameter(s) {missing}")
        for key, value in fac.parameters.items():
            fields = spec.nested.get(key)
            mapped = spec.mapped.get(key)
            if fields is not None:
                if not isinstance(value, Mapping) or set(value) != set(fields):
                    raise self.fail(
                        f"{fac.address}.parameters.{key} is a nested record naming {list(fields)}"
                    )
            elif mapped is not None and isinstance(value, Mapping):
                if set(value) != set(mapped):
                    raise self.fail(
                        f"{fac.address}.parameters.{key} as a map is a record naming {list(mapped)}"
                    )
            elif not isinstance(value, str):
                raise self.fail(f"{fac.address}.parameters.{key} must name a variable")
        unknown = sorted(set(fac.options) - set(spec.options))
        if unknown:
            raise self.fail(
                f"{fac.address}: {fac.family} has no option(s) {unknown}; its options are "
                f"{list(spec.options)}"
            )
        options = {**spec.options, **fac.options}
        module = getattr(self, f"_decode_{fac.family}")(fac, options, role)
        if slot is not None:
            for var in _parameter_names(fac.parameters):
                self.param_slot[var] = slot
        return module

    def _dim(self, fac: _Factor, name: str | None = None) -> int:
        """The size of a real variable a Gaussian or linear-Gaussian family is over."""
        name = name or fac.child
        if self.variables[name].get("domain") != "real":
            raise self.fail(
                f"{fac.address}: a {fac.family} is over a real variable, and {name!r} is "
                "categorical"
            )
        return self.width[name]

    def _trailing(
        self, fac: _Factor, var: str, shape, want: tuple, exact: bool = False, note: str = ""
    ) -> None:
        """``var``'s shape ends in ``want`` (is exactly ``want`` when ``exact``)."""
        shape = tuple(int(n) for n in shape)
        ok = shape == want if exact else shape[len(shape) - len(want) :] == want
        if not ok or len(shape) < len(want):
            raise self.fail(
                f"{fac.address}: {var!r} has shape {list(shape)}, and over {fac.child!r} "
                f"(dim {self.width.get(fac.child)}) its "
                + ("shape is " if exact else "last axes are ")
                + f"{list(want)}{note}"
            )

    def _levels_of(self, fac: _Factor, name: str) -> int:
        levels = self.levels.get(name)
        if levels is None:
            raise self.fail(
                f"{fac.address}: a table reads and emits categorical variables, and {name!r} is "
                "not one (a real variable or a code is read by a linear-Gaussian or network "
                "family)"
            )
        return levels

    def _check_table(self, fac: _Factor, var: str, shape, pad: bool) -> None:
        names = [p for p, _ in fac.parents]
        expected = [self._levels_of(fac, fac.child)] + [self._levels_of(fac, p) for p in names]
        if list(shape) == expected or (pad and list(shape) == [*expected, 1]):
            return
        raise self.fail(
            f"{fac.address}: the table {var!r} has shape {list(shape)}; tables are child-first, "
            f"so {fac.child!r} ({expected[0]} levels) then each parent in order "
            f"{dict(zip(names, expected[1:], strict=True))} gives {expected}"
            + (
                " (or one more trailing axis of size 1, as a transition with no control node may "
                "carry)"
                if pad
                else ""
            )
        )

    def _decode_ConditionalCategorical(self, fac: _Factor, options, role: str):
        var = fac.parameters["log_table"]
        address = f"{fac.address}.parameters.log_table"
        if role == "actuator":
            if var in self.priors:
                raise self.fail(
                    f"{self.priors[var].address}: an actuator kernel is clamped; the engine does "
                    "not learn it"
                )
            domain = self._domain_of(var, "ConditionalCategorical", "log_table", address)
            if domain not in ("log_simplex", "simplex"):
                raise self.fail(
                    f"{address}: an actuator kernel is a probability table, so {var!r} holds "
                    "normalised log-probabilities, domain 'log_simplex' (or probabilities, "
                    "domain 'simplex')"
                )
            arr = self._array(
                self.clamps[var],
                self.variables[var].get("shape"),
                domain,
                f"clamps[{var!r}]",
                dtype=np.float64,
            )
            _check_domain(domain, arr, f"clamps[{var!r}]", 0)
            arr = _lowered(domain, arr, np.float64)
            self._check_table(fac, var, arr.shape, False)
            return jnp.asarray(np.exp(np.asarray(arr, np.float64)).astype(np.float32))
        value = self.value(var, "ConditionalCategorical", "log_table", address, axis=0)
        pad = role == "transition" and not any(p in self.nodes for p, _ in fac.parents)
        if isinstance(value, _Learned):
            self._check_prior(value, ("Dirichlet",), "a table")
            conc = self.hyper(value, "concentration")
            self._shape(value, "concentration", conc)
            self._check_table(fac, var, conc.shape, pad)
            conc = self._padded(fac, conc, pad)
            growth = self.hyper(value, "growth", required=False)
            return _assemble(
                _dist.ConditionalDirichlet,
                fac.address,
                concentration=conc,
                parent_ndims=conc.ndim - 1,
                alpha0=self.hyper(value, "alpha0", required=False),
                growth=growth,
            )
        self._check_table(fac, var, value.shape, pad)
        value = self._padded(fac, value, pad)
        return _assemble(
            _dist.ConditionalCategorical, fac.address, log_table=value, parent_ndims=value.ndim - 1
        )

    def _padded(self, fac: _Factor, table, pad: bool):
        """A composite's uncontrolled transition table with the trailing axis of size 1 the
        engine reads a composite's tables by; a plain System reads either layout."""
        if pad and self.components and table.ndim == 1 + len(fac.parents):
            return table[..., None]
        return table

    def _decode_Categorical(self, fac: _Factor, options, role: str):
        var = fac.parameters["logits"]
        value = self.value(var, "Categorical", "logits", f"{fac.address}.parameters.logits")
        if isinstance(value, _Learned):
            self._check_prior(value, ("Dirichlet",), "logits")
            conc = self.hyper(value, "concentration")
            self._shape(value, "concentration", conc)
            shape = conc.shape
            growth = self.hyper(value, "growth", required=False)
            module = _assemble(
                _dist.Dirichlet,
                fac.address,
                concentration=conc,
                alpha0=self.hyper(value, "alpha0", required=False),
                growth=growth,
            )
        else:
            shape = value.shape
            module = _assemble(_dist.Categorical, fac.address, logits=value)
        if role == "policy_prior":
            self._check_policy_prior(fac, var, shape)
        elif list(shape) != [self._levels_of(fac, fac.child)]:
            raise self.fail(
                f"{fac.address}: {var!r} has shape {list(shape)}, and a Categorical over "
                f"{fac.child!r} has one entry per level: [{self.levels[fac.child]}]"
            )
        return module

    def _check_policy_prior(self, fac: _Factor, var: str, shape) -> None:
        """E's length against the policy set the planning configuration implies (Q18.5b)."""
        if any(self.variables[u]["domain"] == "real" for u in self.nodes):
            raise self.fail(
                f"{fac.address}: a policy prior is a distribution over the enumerated policy set, "
                "and a continuous control node has none"
            )
        levels = [self.levels[u] for u in self.nodes]
        horizon = 1 if self.config["planner"] == "sophisticated" else self.config["policy_len"]
        count = math.prod(levels) ** horizon
        if list(shape) != [count]:
            raise self.fail(
                f"{fac.address}: {var!r} has shape {list(shape)}, and the policy set over the "
                f"control nodes {dict(zip(self.nodes, levels, strict=True))} at horizon {horizon} "
                f"has {count} policies, so its shape is [{count}]"
            )

    def _plain(self, fac: _Factor, cls: type, **extra: Any):
        family = cls.__name__
        fields = {
            name: self.value(var, family, name, f"{fac.address}.parameters.{name}")
            for name, var in fac.parameters.items()
        }
        for name, value in fields.items():
            if isinstance(value, _Learned):
                raise self.fail(
                    f"{value.address}: the {family} field {name!r} is clamped; it takes no prior"
                )
        return _assemble(cls, fac.address, **fields, **{k: f(fields) for k, f in extra.items()})

    def _decode_JointCategorical(self, fac: _Factor, options, role: str):
        var = fac.parameters["logits"]
        logits = self.value(
            var, "JointCategorical", "logits", f"{fac.address}.parameters.logits", axis=None
        )
        if isinstance(logits, _Learned):
            raise self.fail(
                f"{logits.address}: the JointCategorical field 'logits' is clamped; it takes no "
                "prior"
            )
        return _assemble(
            _dist.JointCategorical, fac.address, logits=logits, event_ndims=logits.ndim
        )

    def _gaussian_shapes(self, fac: _Factor, module) -> None:
        d = self._dim(fac)
        names = fac.parameters
        self._trailing(fac, names["loc"], module.loc.shape, (d,), exact=True)
        self._trailing(fac, names["chol"], module.chol.shape, (d, d), exact=True)

    def _decode_Gaussian(self, fac: _Factor, options, role: str):
        module = self._plain(fac, _dist.Gaussian)
        self._gaussian_shapes(fac, module)
        return module

    def _decode_ConditionalGaussian(self, fac: _Factor, options, role: str):
        """A Gaussian per configuration of the record's parents (a conditional target): ``loc``
        ``[*levels, d]``, ``chol`` ``[*levels, d, d]`` or, shared by every configuration,
        ``[d, d]``, one leading axis per parent in order."""
        family, names = "ConditionalGaussian", fac.parameters
        loc = self.value(names["loc"], family, "loc", f"{fac.address}.parameters.loc")
        chol = self.value(names["chol"], family, "chol", f"{fac.address}.parameters.chol")
        for value in (loc, chol):
            if isinstance(value, _Learned):
                raise self.fail(f"{value.address}: a conditional target is clamped")
        d = self._dim(fac)
        parents = [p for p, _ in fac.parents]
        modes = tuple(self._levels_of(fac, p) for p in parents)
        note = f" (one leading axis per parent, in order: {dict(zip(parents, modes, strict=True))})"
        self._trailing(fac, names["loc"], loc.shape, (*modes, d), True, note)
        if tuple(chol.shape) != (d, d):
            self._trailing(fac, names["chol"], chol.shape, (*modes, d, d), True, note)
        return _assemble(
            _dist.ConditionalGaussian, fac.address, loc=loc, chol=chol, parent_ndims=len(parents)
        )

    def _decode_MixtureGaussian(self, fac: _Factor, options, role: str):
        module = self._plain(fac, _dist.MixtureGaussian)
        d, names = self._dim(fac), fac.parameters
        k = int(np.shape(module.log_weights)[-1]) if np.ndim(module.log_weights) else 0
        self._trailing(fac, names["log_weights"], module.log_weights.shape, (k,), exact=True)
        self._trailing(fac, names["means"], module.means.shape, (k, d), exact=True)
        self._trailing(fac, names["chols"], module.chols.shape, (k, d, d), exact=True)
        return module

    def _innovations(self, fac: _Factor, what: str):
        """A Gaussian transition's innovations, one leading axis per discrete parent (or one of
        size 1 for none). Returns ``(innovations, modes, note)``, the leading axes' sizes and the
        sentence that names them."""
        inner = fac.parameters["innovations"]
        address = f"{fac.address}.parameters.innovations"
        loc = self.value(inner["loc"], "ConditionalGaussian", "loc", f"{address}.loc")
        chol = self.value(inner["chol"], "ConditionalGaussian", "chol", f"{address}.chol")
        for value in (loc, chol):
            if isinstance(value, _Learned):
                raise self.fail(f"{value.address}: {what} is clamped")
        d = self._dim(fac)
        parents = [p for p, _ in fac.parents if p != fac.child and self.levels.get(p)]
        modes = tuple(self.levels[p] for p in parents) or (1,)
        note = (
            f" (one leading axis per discrete parent, in order: "
            f"{dict(zip(parents, modes, strict=True))})"
            if parents
            else " (one leading axis of size 1: the factor reads no discrete parent)"
        )
        self._trailing(fac, inner["loc"], loc.shape, (*modes, d), True, note)
        if tuple(chol.shape) != (d, d):  # a noise shared across the modes is [d, d]
            self._trailing(fac, inner["chol"], chol.shape, (*modes, d, d), True, note)
        innovations = _assemble(
            _dist.ConditionalGaussian, address, loc=loc, chol=chol, parent_ndims=loc.ndim - 1
        )
        return innovations, modes, note

    def _decode_LinearGaussianTransition(self, fac: _Factor, options, role: str):
        family = "LinearGaussianTransition"
        fields = {"F": self.value(fac.parameters["F"], family, "F", f"{fac.address}.parameters.F")}
        if "G" in fac.parameters:
            fields["G"] = self.value(
                fac.parameters["G"], family, "G", f"{fac.address}.parameters.G"
            )
        what = "a linear-Gaussian transition"
        for value in fields.values():
            if isinstance(value, _Learned):
                raise self.fail(f"{value.address}: {what} is clamped")
        innovations, modes, note = self._innovations(fac, what)
        d = self._dim(fac)
        F = fields["F"].shape
        if tuple(F) != (d, d):
            self._trailing(fac, fac.parameters["F"], F, (*modes, d, d), True, note)
        return _assemble(
            _factors.LinearGaussianTransition, fac.address, innovations=innovations, **fields
        )

    def _decode_NonlinearGaussianTransition(self, fac: _Factor, options, role: str):
        """``f`` rebuilt from its registered class: the document's fields checked against the
        class's schema (names, static or leaf), then assembled from their values; only a
        registered class is instantiated. Shapes are checked when the System is validated."""
        spec, address = options["f"], f"{fac.address}.options.f"
        if (
            not isinstance(spec, Mapping)
            or not isinstance(spec.get("name"), str)
            or set(spec) - {"name", "static"}
            or not isinstance(spec.get("static", {}), Mapping)
        ):
            raise self.fail(
                f"{address} is {{'name': '<package>.<name>', 'static': {{...}}}}: the name f's "
                "class is registered under (register_transition) and its static fields; got "
                f"{spec!r}"
            )
        name, static = spec["name"], dict(spec.get("static", {}))
        unwritable = sorted(k for k, v in static.items() if not _static_json(v))
        if unwritable:
            raise self.fail(
                f"{address}.static: {unwritable} hold values a static field cannot take; a static "
                "field is null, a bool, an int, a finite number, a string, or a list of these"
            )
        entry = _TRANSITIONS.get(name)
        if entry is None:
            raise self.fail(
                f"{address}: no transition function is registered as {name!r}; a document "
                "refers only to code the reading process installed, so call "
                f"register_transition({name!r}, <its class>) before build (registered: "
                f"{sorted(_TRANSITIONS)})"
            )
        leaves: dict[str, Any] = {}
        var = fac.parameters.get("f")
        if var is not None:
            if var in self.priors:
                raise self.fail(
                    f"{self.priors[var].address}: f's fields are point values and take no prior "
                    "record; clamp them"
                )
            record, clamp = self.variables[var], self.clamps[var]
            if isinstance(clamp, Mapping) and "init" in clamp:
                raise self.fail(f"clamps[{var!r}]: f's fields are given, not drawn; give values")
            if not isinstance(clamp, Mapping) or {"ref", "init"} & set(clamp):
                raise self.fail(
                    f"clamps[{var!r}]: f's fields are clamped field by field, "
                    "{'<field>': <value or ref>, ...}; got a single value"
                )
            if _is_leaf(record) or not isinstance(record, Mapping):
                raise self.fail(
                    f"variables[{var!r}]: f's leaves are a tree variable keyed by field name "
                    "({'<field>': {'domain': 'real', 'shape': [...]}, ...})"
                )
            nested = [k for k, v in record.items() if not _is_leaf(v)]
            if nested:
                raise self.fail(
                    f"variables[{var!r}]: f's leaf fields are arrays, and {nested} are trees"
                )
            leaves = self._tree(clamp, record, f"clamps[{var!r}]")
        schema_leaves = {n for n, s in entry.fields if not s}
        schema_static = {n for n, s in entry.fields if s}
        written = set(leaves) | set(static)
        differences = []
        for label, names in (
            ("missing", sorted((schema_leaves | schema_static) - written)),
            ("not fields of the class", sorted(written - schema_leaves - schema_static)),
            ("written as leaves but static", sorted(set(leaves) & schema_static)),
            ("written as static but leaves", sorted(set(static) & schema_leaves)),
        ):
            if names:
                differences.append(f"{label}: {names}")
        if differences:
            cls = entry.cls
            raise self.fail(
                f"{address}: the document's fields of {name!r} differ from those of the class "
                f"registered under it, {cls.__module__}.{cls.__qualname__} "
                f"({_schema_text(entry.fields)}): {'; '.join(differences)}. A document written "
                "for another version of f is refused rather than rebuilt with code that means "
                "something else; describe the model again with the current f"
            )
        fields = {**leaves, **{k: _as_tuples(v) for k, v in static.items()}}
        f = _assemble(entry.cls, f"{fac.address}.parameters.f", **fields)
        innovations, _, _ = self._innovations(fac, "a nonlinear transition's process noise")
        return _assemble(
            _factors.NonlinearGaussianTransition,
            fac.address,
            f=f,
            innovations=innovations,
            propagation=options["propagation"],
        )

    def _decode_LinearGaussianObservation(self, fac: _Factor, options, role: str):
        """``head`` and ``noise``, each clamped or learned: inverse-gamma or inverse-Wishart
        under the noise, matrix-normal under the head. A matrix-normal head whose ``row_cov``
        names an inverse-Wishart noise is the joint MNIW family (Q18.3); without ``row_cov`` the
        two priors are independent."""
        family = "LinearGaussianObservation"
        head_var, noise_var = fac.parameters["head"], fac.parameters["noise"]
        head = self.value(head_var, family, "head", f"{fac.address}.parameters.head")
        d_obs = self._dim(fac)
        parents = [p for p, _ in fac.parents]
        if parents and all(self.variables[p].get("domain") == "categorical" for p in parents):
            d_x = 0  # a Gaussian emission over categorical parents alone: the offset column
        else:
            d_x = self._dim(fac, fac.parents[0][0])
        grid = options.get("noise_map")
        if isinstance(noise_var, Mapping):
            if not isinstance(grid, Mapping) or set(grid) != {"low", "high", "axes"}:
                raise self.fail(
                    f"{fac.address}: a noise map's values need options.noise_map with the grid's "
                    "low, high and axes"
                )
            values_var = noise_var["values"]
            address = f"{fac.address}.parameters.noise.values"
            values = self.value(values_var, family, "noise.values", address)
            if isinstance(values, _Learned):
                raise self.fail(f"{address}: a noise map is a clamp; it takes no prior")
            self._trailing(fac, values_var, values.shape, (d_obs, d_obs))
            noise = _assemble(
                _factors.GridNoiseMap,
                f"{fac.address}.parameters.noise",
                values=values,
                low=jnp.asarray(grid["low"], values.dtype),
                high=jnp.asarray(grid["high"], values.dtype),
                axes=tuple(int(a) for a in grid["axes"]),
            )
            if isinstance(head, _Learned):
                raise self.fail(
                    f"{fac.address}: a noise map beside a learned head is not built; clamp the head"
                )
            self._trailing(fac, head_var, head.shape, (d_obs, d_x + 1))
            return _assemble(
                _factors.LinearGaussianObservation, fac.address, head=head, noise=noise
            )
        if grid is not None:
            raise self.fail(
                f"{fac.address}: options.noise_map belongs with a noise written as a map, "
                '{"values": <variable>}'
            )
        noise = self.value(noise_var, family, "noise", f"{fac.address}.parameters.noise")
        if isinstance(noise, _Learned):
            self._check_prior(noise, ("InverseGamma", "InverseWishart"), "a noise")
            fields = (
                ("alpha", "beta", "alpha0", "beta0")
                if noise.family == "InverseGamma"
                else ("Psi", "nu", "Psi0", "nu0")
            )
            noise_fields = {f: self.hyper(noise, f, required=f in fields[:2]) for f in fields}
        joint = False
        if isinstance(head, _Learned):
            self._check_prior(head, ("MatrixNormal",), "a head")
            row_cov = head.parameters.get("row_cov")
            if row_cov is not None and row_cov != noise_var:
                raise self.fail(
                    f"{head.address}: row_cov names {row_cov!r}, and a head's row covariance is "
                    f"its likelihood's noise, {noise_var!r}"
                )
            if row_cov is not None and not (
                not isinstance(noise, _Learned) or noise.family == "InverseWishart"
            ):
                raise self.fail(
                    f"{head.address}: row_cov names {noise_var!r}, whose prior is "
                    f"{noise.family}; the joint prior pairs a matrix-normal head with an "
                    "inverse-Wishart noise, so give the noise that prior or drop row_cov"
                )
            joint = row_cov is not None and isinstance(noise, _Learned)
            mn = {
                field: self.hyper(head, field, required=field in ("M", "V_inv"))
                for field in ("M", "V_inv", "M0", "V_inv0")
            }
            self._shape(head, "M", mn["M"])
            self._trailing(fac, head_var, mn["M"].shape, (d_obs, d_x + 1))
            if not joint:
                head = _assemble(_dist.MatrixNormal, f"{fac.address}.parameters.head", **mn)
        else:
            self._trailing(fac, head_var, head.shape, (d_obs, d_x + 1))
        if joint:
            self._shape(noise, "Psi", noise_fields["Psi"])
            self._trailing(fac, noise_var, noise_fields["Psi"].shape, (d_obs, d_obs))
            return _assemble(
                _factors.LinearGaussianObservationMNIW, fac.address, **mn, **noise_fields
            )
        if isinstance(noise, _Learned):
            cls = _dist.InverseGamma if noise.family == "InverseGamma" else _dist.InverseWishart
            noise = _assemble(cls, f"{fac.address}.parameters.noise", **noise_fields)
            point = np.shape(noise.point())
            declared = self.variables[noise_var].get("shape")
            if declared is not None and list(point) != list(declared):
                raise self.fail(
                    f"{fac.address}: the noise prior's point has shape {list(point)} and "
                    f"{noise_var!r} declares {declared}"
                )
            self._trailing(fac, noise_var, point, (d_obs, d_obs))
        else:
            self._trailing(fac, noise_var, noise.shape, (d_obs, d_obs))
        return _assemble(_factors.LinearGaussianObservation, fac.address, head=head, noise=noise)

    # ------------------------------------------------------------------ networks

    def _layout(self, fac: _Factor, role: str) -> tuple[tuple[int, ...], int]:
        """``(parent widths, child width)`` a network record's wiring implies: the parents in
        order (a discrete one as its one-hot), or, for a control node's policy module, the
        beliefs over every latent concatenated."""
        if role == "habit":
            return (sum(self.width[s] for s in self.states),), sum(
                self.width[u] for u in self.nodes
            )
        return tuple(self.width[p] for p, _ in fac.parents), self.width[fac.child]

    def _init_args(self, init: _Init, family: str, address: str) -> dict[str, Any]:
        allowed = _INIT_ARGS[family]
        unknown = sorted(set(init.args) - set(allowed))
        if unknown:
            raise self.fail(
                f"{address}: an init recipe for {family} takes {list(allowed)}, not {unknown}"
            )
        missing = [a for a in _INIT_REQUIRED.get(family, ()) if a not in init.args]
        if missing:
            raise self.fail(
                f"{address}: an init recipe for {family} needs {missing[0]!r}; write "
                f"{{'init': {{{missing[0]!r}: ...}}}}"
            )
        return dict(init.args)

    def _draw(self, address: str, ctor: Callable, *args: Any, **kwargs: Any) -> Any:
        """A network drawn by its family's constructor from an init recipe."""
        try:
            return ctor(*args, **kwargs)
        except (TypeError, ValueError) as exc:
            raise self.fail(f"{address}: the init recipe does not draw a network: {exc}") from exc

    def _check_drawn(self, var: str, family: str, module: Any, address: str) -> None:
        record = self.variables[var]
        if _is_leaf(record) and "shape" not in record:
            return
        want = _record_shapes(record)
        got = _tree_shapes(_module_tree(family, module))
        if want != got:
            raise self.fail(
                f"{address}: the init recipe draws a network of shapes {got}, and the variable "
                f"{var!r} declares {want}"
            )

    def _mlp(self, tree: Any, var: str, norm: bool, address: str) -> tuple[tuple, tuple]:
        if (
            not isinstance(tree, Mapping)
            or "layers" not in tree
            or not set(tree)
            <= {
                "layers",
                "norms",
            }
        ):
            raise self.fail(
                f"clamps[{var!r}]: a network's tree is {{'layers': [...], 'norms': [...]}}"
            )
        layers = tuple(_linear(t) for t in tree["layers"])
        for i in range(1, len(layers)):
            if layers[i].in_features != layers[i - 1].out_features:
                raise self.fail(
                    f"clamps[{var!r}]: layer {i} takes {layers[i].in_features} inputs and layer "
                    f"{i - 1} emits {layers[i - 1].out_features}; each layer reads the one before"
                )
        norms = tuple(_layer_norm(t) for t in tree.get("norms", []))
        if bool(norms) != bool(norm):
            raise self.fail(
                f"{address}: options.norm is {norm} and the network's tree "
                f"{'carries' if norms else 'carries no'} norms"
            )
        return layers, norms

    def _trunk(
        self,
        var: str,
        address: str,
        parent_dims: tuple[int, ...],
        out_dim: int,
        options: Mapping[str, Any],
        frozen: bool,
    ):
        tree = self.network(var, address)
        layers, norms = self._mlp(tree, var, options["norm"], address)
        return _assemble(
            _dist.MLPPredictor,
            address,
            layers=layers,
            norms=norms,
            parent_dims=parent_dims,
            out_dim=out_dim,
            activation=options["activation"],
            out_activation=options["out_activation"],
            frozen=frozen,
            symlog_inputs=options["symlog_inputs"],
        )

    def _network_var(self, fac: _Factor, key: str) -> str:
        var = fac.parameters[key]
        self.gradient_vars.add(var)
        return var

    def _decode_MLPConditional(self, fac: _Factor, options, role: str, layout=None):
        var = self._network_var(fac, "layers")
        parent_dims, child_dim = layout or self._layout(fac, role)
        frozen = var in self.frozen
        config = {k: options[k] for k in ("activation", "min_scale", "head", "minstd", "maxstd")}
        tree = self.network(var, fac.address)
        if isinstance(tree, _Init):
            module = self._draw(
                f"clamps[{var!r}]",
                _dist.MLPConditional.from_shapes,
                parent_dims,
                child_dim,
                key=self.reader.next_key(f"clamps[{var!r}]"),
                frozen=frozen,
                norm=options["norm"],
                **config,
                **self._init_args(tree, "MLPConditional", f"clamps[{var!r}]"),
            )
            self._check_drawn(var, "MLPConditional", module, f"clamps[{var!r}]")
            return module
        layers, norms = self._mlp(tree, var, options["norm"], fac.address)
        return _assemble(
            _dist.MLPConditional,
            fac.address,
            layers=layers,
            parent_dims=parent_dims,
            child_dim=child_dim,
            norms=norms,
            frozen=frozen,
            **config,
        )

    def _decode_MLPPredictor(self, fac: _Factor, options, role: str, layout=None):
        var = self._network_var(fac, "layers")
        parent_dims, out_dim = layout or self._layout(fac, role)
        frozen = var in self.frozen
        tree = self.network(var, fac.address)
        if isinstance(tree, _Init):
            module = self._draw(
                f"clamps[{var!r}]",
                _dist.MLPPredictor.from_shapes,
                parent_dims,
                out_dim,
                key=self.reader.next_key(f"clamps[{var!r}]"),
                activation=options["activation"],
                norm=options["norm"],
                out_activation=options["out_activation"],
                frozen=frozen,
                symlog_inputs=options["symlog_inputs"],
                **self._init_args(tree, "MLPPredictor", f"clamps[{var!r}]"),
            )
            self._check_drawn(var, "MLPPredictor", module, f"clamps[{var!r}]")
            return module
        return self._trunk(var, fac.address, parent_dims, out_dim, options, frozen)

    def _decode_MLPCategorical(self, fac: _Factor, options, role: str):
        var = self._network_var(fac, "trunk")
        parent_dims, _ = self._layout(fac, role)
        record = self.variables[fac.child]
        if record.get("domain") != "categorical":
            raise self.fail(
                f"{fac.address}: an MLPCategorical emits a code, and {fac.child!r} is not "
                "categorical"
            )
        num_vars, num_classes = math.prod(record.get("shape", [])), record["levels"]
        frozen = var in self.frozen
        tree = self.network(var, fac.address)
        if isinstance(tree, _Init):
            if options["out_activation"] or options["symlog_inputs"]:
                raise self.fail(
                    f"clamps[{var!r}]: the init recipe of an MLPCategorical draws a trunk with "
                    "out_activation and symlog_inputs off; give the weights instead"
                )
            module = self._draw(
                f"clamps[{var!r}]",
                _dist.MLPCategorical.from_shapes,
                parent_dims,
                num_vars,
                num_classes,
                key=self.reader.next_key(f"clamps[{var!r}]"),
                activation=options["activation"],
                norm=options["norm"],
                unimix=options["unimix"],
                frozen=frozen,
                **self._init_args(tree, "MLPCategorical", f"clamps[{var!r}]"),
            )
            self._check_drawn(var, "MLPCategorical", module, f"clamps[{var!r}]")
            return module
        trunk = self._trunk(var, fac.address, parent_dims, num_vars * num_classes, options, frozen)
        return _assemble(
            _dist.MLPCategorical,
            fac.address,
            trunk=trunk,
            num_vars=num_vars,
            num_classes=num_classes,
            unimix=options["unimix"],
        )

    def _decode_PredictorHead(self, fac: _Factor, options, role: str, layout=None):
        var = self._network_var(fac, "trunk")
        loss = options["loss"]
        if loss is None:
            raise self.fail(f"{fac.address}: a PredictorHead needs options.loss")
        if layout is not None:
            parent_dims, out_dim = layout
        else:
            parent_dims, child = self._layout(fac, role)
            if loss != "symlog_mse" and child != 1:
                raise self.fail(
                    f"{fac.address}: a {loss} head predicts one number, and {fac.child!r} has dim "
                    f"{child}; give it dim 1, or use the symlog_mse loss for a vector"
                )
            out_dim = _head_width(loss, options["num_bins"], child)
        frozen = var in self.frozen
        bins = {k: options[k] for k in ("num_bins", "bin_low", "bin_high")}
        tree = self.network(var, fac.address)
        if isinstance(tree, _Init):
            if options["out_activation"]:
                raise self.fail(
                    f"clamps[{var!r}]: the init recipe of a PredictorHead draws a trunk with "
                    "out_activation off; give the weights instead"
                )
            module = self._draw(
                f"clamps[{var!r}]",
                _dist.PredictorHead.from_shapes,
                parent_dims,
                loss=loss,
                key=self.reader.next_key(f"clamps[{var!r}]"),
                out_dim=out_dim if loss == "symlog_mse" else None,
                activation=options["activation"],
                norm=options["norm"],
                frozen=frozen,
                symlog_inputs=options["symlog_inputs"],
                **bins,
                **self._init_args(tree, "PredictorHead", f"clamps[{var!r}]"),
            )
            self._check_drawn(var, "PredictorHead", module, f"clamps[{var!r}]")
            return module
        trunk = self._trunk(var, fac.address, parent_dims, out_dim, options, frozen)
        return _assemble(_dist.PredictorHead, fac.address, trunk=trunk, loss=loss, **bins)

    def _decode_Deterministic(self, fac: _Factor, options, role: str):
        mapping = options["map"]
        if isinstance(mapping, Mapping) and set(mapping) == {"impl"}:
            if fac.parameters:
                raise self.fail(
                    f"{fac.address}: a Deterministic over a named function takes no parameters"
                )
            function = _import(mapping["impl"], f"{fac.address}.options.map", self.fail)
            return _assemble(
                _factors.Deterministic, fac.address, map=function, rule=options["rule"]
            )
        if mapping != "GRUCell":
            raise self.fail(
                f"{fac.address}: options.map is 'GRUCell' (its weights the parameter 'map') or "
                f"{{'impl': 'package.module:name'}}, got {mapping!r}"
            )
        if "map" not in fac.parameters:
            raise self.fail(
                f"{fac.address}: a GRUCell map names its tree variable under parameters.map"
            )
        var = self._network_var(fac, "map")
        widths, deter_dim = self._layout(fac, role)
        input_dims = widths[1:]
        frozen = var in self.frozen
        tree = self.network(var, fac.address)
        if isinstance(tree, _Init):
            module = self._draw(
                f"clamps[{var!r}]",
                _factors.Deterministic.gru,
                deter_dim,
                input_dims,
                key=self.reader.next_key(f"clamps[{var!r}]"),
                activation=options["activation"],
                norm=options["norm"],
                frozen=frozen,
                **self._init_args(tree, "Deterministic", f"clamps[{var!r}]"),
            )
            if module.rule != options["rule"]:
                module = dataclasses.replace(module, rule=options["rule"])
            self._check_drawn(var, "Deterministic", module, f"clamps[{var!r}]")
            return module
        if (
            not isinstance(tree, Mapping)
            or not {"input_proj", "gate"} <= set(tree)
            or not set(tree) <= {"input_proj", "gate", "norms"}
        ):
            raise self.fail(
                f"clamps[{var!r}]: a GRUCell's tree is {{'input_proj': ..., 'gate': ..., "
                "'norms': [...]}}"
            )
        norms = tuple(_layer_norm(t) for t in tree.get("norms", []))
        if bool(norms) != bool(options["norm"]):
            raise self.fail(f"{fac.address}: options.norm disagrees with the map's norms")
        cell = _assemble(
            _dist.GRUCell,
            f"{fac.address}.parameters.map",
            input_proj=_linear(tree["input_proj"]),
            gate=_linear(tree["gate"]),
            norms=norms,
            deter_dim=deter_dim,
            input_dims=input_dims,
            activation=options["activation"],
        )
        return _assemble(
            _factors.Deterministic, fac.address, map=cell, rule=options["rule"], frozen=frozen
        )

    def _decode_LearnedVector(self, fac: _Factor, options, role: str):
        var = self._network_var(fac, "value")
        frozen = var in self.frozen
        clamp = self.clamps.get(var)
        if isinstance(clamp, Mapping) and "init" in clamp:
            self._init_args(_Init(clamp["init"]), "LearnedVector", f"clamps[{var!r}]")
            value = jnp.zeros((self.width[fac.child],), jnp.float32)
        else:
            value = self.value(var, "LearnedVector", "value", f"{fac.address}.parameters.value")
            if isinstance(value, _Learned):
                raise self.fail(f"{value.address}: a learned vector is clamped and trained")
            self._trailing(fac, var, value.shape, (self.width[fac.child],), exact=True)
        return _assemble(
            _dist.LearnedVector,
            fac.address,
            value=value,
            activation=options["activation"],
            frozen=frozen,
        )

    def _decode_DerivedInitial(self, fac: _Factor, options, role: str):
        return _assemble(_factors.DerivedInitial, fac.address, reduction=options["reduction"])

    def _decode_prior_family(self, fac: _Factor, options, role: str):  # pragma: no cover
        raise self.fail(f"{fac.address}: a {fac.family} record is a prior under a parameter")

    _decode_Dirichlet = _decode_InverseGamma = _decode_InverseWishart = _decode_prior_family
    _decode_MatrixNormal = _decode_prior_family

    def term_network(self, var: str, spec: object, address: str):
        """A term parameter's network: its family, configuration and layout from the term
        record's options."""
        if not isinstance(spec, Mapping) or spec.get("family") not in _TERM_FAMILIES:
            raise self.fail(
                f"{address}: a term parameter's options give its family, one of "
                f"{list(_TERM_FAMILIES)}, with its configuration and layout"
            )
        family = spec["family"]
        layout_keys = _TERM_LAYOUT[family]
        known = {*FAMILIES[family].options, *layout_keys}
        unknown = sorted(set(spec) - known - {"family"})
        if unknown:
            raise self.fail(f"{address}: {family} has no option(s) {unknown}")
        options = {**FAMILIES[family].options, **{k: v for k, v in spec.items() if k != "family"}}
        needed = [
            k
            for k in layout_keys
            if k not in options
            and not (k == "out_dim" and options.get("loss") not in (None, "symlog_mse"))
        ]
        if needed:
            raise self.fail(
                f"{address}: a term parameter has no wiring, so its options give its layout; "
                f"write {needed}"
            )
        parent_dims = tuple(int(d) for d in options.get("parent_dims", ()))
        fake = _Factor("", "", family, (), {FAMILIES[family].network: var}, {}, None, address)
        if family == "MLPConditional":
            return self._decode_MLPConditional(
                fake, options, "term", (parent_dims, int(options["child_dim"]))
            )
        if family == "MLPPredictor":
            return self._decode_MLPPredictor(
                fake, options, "term", (parent_dims, int(options["out_dim"]))
            )
        loss = options["loss"]
        out_dim = _head_width(loss, options["num_bins"], int(options.get("out_dim", 0)))
        return self._decode_PredictorHead(fake, options, "term", (parent_dims, out_dim))

    # ----------------------------------------------------------------- objective

    def _epistemic_key(self, on: str, address: str) -> str:
        if "*" in on.split(":"):
            return on
        if on in self.states or on.partition(":")[0] in self.states:  # a latent, or a dimension
            return on
        if on in self.observed:
            return f"obs:{on}"
        if on in self.param_slot:
            slot, name = self.param_slot[on]
            return f"{slot}:{name}"
        raise self.fail(
            f"{address}: infogain on {on!r}, which is not a latent, an observation or a "
            "parameter of a factor; name one of those"
        )

    def _row_key(self, estimator, row: str, on: str | None, address: str) -> str:
        target = estimator.target
        if target is None:
            if on is not None:
                raise self.fail(f"{address}: the term {row!r} has no target, so it takes no on")
            return row
        slot = _registry.row_slot(row)
        if on is None:
            if target == _registry.SLOT_TARGET:
                return f"{row}:{slot or '*'}:*"
            return f"{row}:obs:*" if target == _registry.OBS_TARGET else f"{row}:*"
        if "*" in on.split(":"):
            return f"{row}:{on}"
        if target == _registry.SLOT_TARGET:
            if slot == "transition" and on in self.states:
                return f"{row}:transition:{on}"
            if slot == "likelihood" and on in self.observed:
                return f"{row}:likelihood:{on}"
            if slot is None and on in self.param_slot:
                return f"{row}:{self.param_slot[on][0]}:{self.param_slot[on][1]}"
            raise self.fail(
                f"{address}: the term {row!r} reads a "
                + ("latent's transition" if slot == "transition" else f"{slot or 'factor'}")
                + f", and {on!r} has none"
            )
        if target == _registry.OBS_TARGET:
            if on not in self.observed:
                raise self.fail(
                    f"{address}: the term {row!r} reads an observation; {on!r} is not one"
                )
            return f"{row}:obs:{on}"
        if on not in self.states and on.partition(":")[0] not in self.states:
            raise self.fail(f"{address}: the term {row!r} reads a latent; {on!r} is not one")
        return f"{row}:{on}"

    def _term_entry(self, estimator, record: Mapping[str, Any], address: str) -> Any:
        """A parametric term's ``SystemParams.scoring`` entry from its parameters and options."""
        names = estimator.parameters
        given = record.get("parameters", {})
        options = record.get("options", {})
        unknown = sorted(set(given) - set(names))
        if unknown or not given:
            raise self.fail(
                f"{address}: the term {record['term']!r} takes the parameter(s) {list(names)}, got "
                f"{sorted(given)}"
            )
        if not isinstance(options, Mapping) or set(options) != set(given):
            raise self.fail(
                f"{address}.options gives each parameter's family, configuration and layout, keyed "
                f"as its parameters: {sorted(given)}"
            )
        if len(names) == 1:
            variables = given[names[0]]
            if not isinstance(variables, list) or not variables:
                raise self.fail(f"{address}.parameters.{names[0]} is a list of variables")
            spec = options[names[0]]
            specs = spec if isinstance(spec, list) else [spec] * len(variables)
            if len(specs) != len(variables):
                raise self.fail(f"{address}.options.{names[0]} gives one record per variable")
            return tuple(
                self.term_network(v, s, f"{address}.parameters.{names[0]}[{i}]")
                for i, (v, s) in enumerate(zip(variables, specs, strict=True))
            )
        present = [n for n in names if n in given]
        if present != list(names[: len(present)]):
            raise self.fail(
                f"{address}: the term {record['term']!r} names its parameters in order "
                f"{list(names)}; a later one is given only with the earlier ones"
            )
        items = tuple(
            self.term_network(given[n], options[n], f"{address}.parameters.{n}") for n in present
        )
        return items[0] if len(items) == 1 else items

    def _objective(self) -> None:
        """``decomposition``, ``epistemic``, ``terms`` and ``scoring`` from the term list:
        ``decomposition`` expanded to its pair, the pair read back off its terms, a mix the
        term tree cannot build refused (Q18.5d)."""
        decomposition = self.objective.get("decomposition")
        if decomposition is not None and decomposition not in (
            "utility_infogain",
            "risk_ambiguity",
        ):
            raise self.fail(
                f"objective.decomposition is 'utility_infogain' or 'risk_ambiguity', got "
                f"{decomposition!r}"
            )
        pair: list[str] = []
        infogain: list[tuple[str, Mapping[str, Any]]] = []
        rows: list[tuple[str, Mapping[str, Any]]] = []
        menu = sorted({*_PAIR_TERMS, INFOGAIN, *_registry.ROWS})
        for i, record in enumerate(self.terms):
            address = f"objective.terms[{i}]"
            name = record["term"]
            if name in _PAIR_TERMS:
                extra = sorted(set(record) - {"term"})
                if extra:
                    raise self.fail(
                        f"{address}: {name} takes no {extra}: it applies to every "
                        + (
                            "channel the infogain terms select"
                            if name == "ambiguity"
                            else "targeted observation"
                        )
                    )
                pair.append(name)
            elif name == INFOGAIN:
                infogain.append((address, record))
            elif name in _registry.ROWS:
                rows.append((address, record))
            elif name == "terminal_risk":
                raise self.fail(
                    f"{address}: terminal_risk is scored by a terminal record, under the latent "
                    "whose end of plan it scores; write that record instead of the term"
                )
            else:
                raise self.fail(
                    f"{address}: unknown term {name!r}; the registered terms are {menu} "
                    "(hybrid_aif.objectives.registry), and a term a reference subtree defines is "
                    "registered when that package is imported (import hybrid_aif.reference)"
                )
        if decomposition is not None:
            listed = sorted({*pair, *([INFOGAIN] if infogain else [])})
            if pair or infogain:
                raise self.fail(
                    f"objective.decomposition {decomposition!r} expands to its pair over every "
                    f"eligible variable, and terms lists {listed} "
                    "beside it; write the pair as terms instead (utility and infogain, or risk "
                    "and ambiguity) to choose the variables"
                )
            epistemic: bool | tuple[str, ...] = True
        else:
            utility, risk = "utility" in pair, {"risk", "ambiguity"} & set(pair)
            if (utility and risk) or (risk and risk != {"risk", "ambiguity"}):
                raise self.fail(
                    f"objective.terms lists {sorted(set(pair))}, which no term tree builds; the "
                    "pairs are utility with infogain, or risk with ambiguity"
                )
            decomposition = "risk_ambiguity" if risk else "utility_infogain"
            unaimed = [a for a, r in infogain if "on" not in r]
            if unaimed and len(infogain) > 1:
                raise self.fail(
                    f"{unaimed[0]}: an infogain term without on selects every eligible variable; "
                    "drop the other infogain terms, or give this one an on"
                )
            if not infogain:
                epistemic = False
            elif unaimed:
                epistemic = True
            else:
                epistemic = tuple(self._epistemic_key(r["on"], a) for a, r in infogain)
        scoring: dict[str, Any] = {}
        for address, record in infogain:
            if "model" in record:
                raise self.fail(f"{address}: an infogain term owns no model")
            if "parameters" not in record:
                if "options" in record:
                    raise self.fail(f"{address}: options describe a term's parameters")
                continue
            keys = set(record["parameters"])
            matches = [
                est
                for est in _registry.ESTIMATORS.values()
                if est.parametric and est.parameters and keys <= set(est.parameters)
            ]
            if len(matches) != 1:
                raise self.fail(
                    f"{address}: no infogain estimator takes the parameters {sorted(keys)}"
                )
            if matches[0].name in scoring:
                raise self.fail(
                    f"{address}: a second infogain term gives {matches[0].name}'s parameters"
                )
            scoring[matches[0].name] = self._term_entry(matches[0], record, address)
        terms: list[str] = []
        owned: list[tuple[str, System]] = []
        for address, record in rows:
            row = record["term"]
            estimator = _registry.ROWS[row]
            terms.append(self._row_key(estimator, row, record.get("on"), address))
            if "model" in record:
                if not estimator.needs_structure:
                    raise self.fail(f"{address}: the term {row!r} owns no model")
                if row in scoring:
                    raise self.fail(f"{address}: a second {row!r} term gives its model")
                model = record["model"]
                if isinstance(model, Mapping) and set(model) == {"ref"}:
                    raise self.fail(
                        f"{address}.model refers to {model['ref']!r}; a model given by path is "
                        "read by load(), which nests it, and build() takes the nested document"
                    )
                if not isinstance(model, Mapping):
                    raise self.fail(f"{address}.model is a nested document")
                inner_system, inner_params = _Model(
                    self.reader, model, f"{self.where}{row}/"
                ).build()
                owned.append((row, inner_system))
                scoring[row] = inner_params
            elif "parameters" in record or "options" in record:
                if not estimator.parametric:
                    raise self.fail(f"{address}: the term {row!r} takes no parameters")
                if row in scoring:
                    raise self.fail(f"{address}: a second {row!r} term gives its parameters")
                scoring[row] = self._term_entry(estimator, record, address)
        self.decomposition, self.epistemic = decomposition, epistemic
        self.term_keys, self.scoring, self.owned = tuple(terms), scoring, tuple(owned)

    # ---------------------------------------------------------------- construction

    def _construct(self) -> tuple[System, SystemParams]:
        self.param_slot: dict[str, tuple[str, str]] = {}
        self.gradient_vars: set[str] = set()
        states, nodes, actions, mods = self.states, self.nodes, self.actions, self.mods
        n_states = len(states)
        index = {s: i for i, s in enumerate(states)}
        index.update({u: n_states + k for k, u in enumerate(nodes)})
        n_nodes = n_states + len(nodes)
        action_index = {a: n_nodes + k for k, a in enumerate(actions)}

        def decode(fac: _Factor, role: str, slot: tuple[str, str] | None = None) -> Any:
            with self.noting():
                return self.decode(fac, role, slot)
            return None

        initial = {
            s: decode(self.initial[s], "initial", ("initial_state_prior", s)) for s in states
        }
        transition = {
            s: decode(self.transition[s], "transition", ("transition", s)) for s in states
        }
        likelihood = {o: decode(self.likelihood[o], "likelihood", ("likelihood", o)) for o in mods}
        recognition = {
            v: decode(self.recognition[v], "recognition", ("recognition", v))
            for v in self.rec_order
        }
        actuator = {a: decode(fac, "actuator") for a, fac in self.actuator.items()}
        policy_prior = None
        if self.policy_prior is not None:
            policy_prior = decode(self.policy_prior, "policy_prior")
        if self.habit is not None:
            policy_prior = decode(self.habit, "habit")
        obs_targets, state_targets, terminal_targets = {}, {}, {}
        for v, fac in self.targets.items():
            (obs_targets if v in self.observed else state_targets)[v] = decode(fac, "target")
        for v, fac in self.terminals.items():
            terminal_targets[v] = decode(fac, "terminal")
        self.flush()
        self._objective()
        self._check_frozen()

        def deps(fac: _Factor, drop_self: bool) -> tuple[int, ...]:
            return tuple(index[p] for p, _ in fac.parents if not (drop_self and p == fac.child))

        B_deps = tuple(
            deps(self.transition[s], FAMILIES[self.transition[s].family].reads_self) for s in states
        ) + ((),) * len(nodes)
        A_deps = tuple(
            tuple(
                index[p] if p in index else action_index[p] for p, _ in self.likelihood[o].parents
            )
            for o in mods
        )
        D = [deps(self.initial[s], False) for s in states]
        H = [deps(self.terminals[s], False) if s in self.terminals else () for s in states]
        pad = ((),) * len(nodes)
        D_deps = None if not any(D) else (*D, *pad)
        H_deps = None if not any(H) else (*H, *pad)
        C_deps = tuple(
            (v, tuple(p for p, _ in fac.parents)) for v, fac in self.targets.items() if fac.parents
        )
        recognition_decl = tuple(
            _factors.Recognition(
                v,
                tuple(self._recognition_parent(p) for p, _ in self.recognition[v].parents),
            )
            for v in self.rec_order
        )
        external_inputs = tuple((name, self.width[name]) for name in self.external)
        reserved = sorted((set(states) | set(nodes)) & set(_KEY_KINDS))
        if reserved:
            raise self.fail(
                f"variables {reserved} are named after reserved first segments of the epistemic "
                f"key grammar {list(_KEY_KINDS)}, which a System refuses as node names; rename "
                "them"
            )
        named = [v for v, r in self.variables.items() if _is_leaf(r) and "dims" in r]
        stray = [v for v in named if v not in states]
        if stray:
            raise self.fail(
                f"variables {stray} carry dims, which names the dimensions of a latent for the "
                "epistemic selector; only a latent's record carries it"
            )
        kwargs: dict[str, Any] = dict(
            B_deps=B_deps,
            A_deps=A_deps,
            planner=self.config["planner"],
            gamma=self.config["gamma"],
            num_iters=self.config["num_iters"],
            decomposition=self.decomposition,
            epistemic=self.epistemic,
            mixture_k=self.config["mixture_k"],
            mixture_schedule=self.config["mixture_schedule"],
            reduction_period=self.config["reduction_period"],
            reduction_pairs=self.config["reduction_pairs"],
            lagged=tuple((o, self.lagged[o]) for o in mods if o in self.lagged),
            D_deps=D_deps,
            H_deps=H_deps,
            C_deps=C_deps,
            modality_names=tuple(mods),
            factor_names=(*states, *nodes),
            dims={v: tuple(self.variables[v]["dims"]) for v in named},
            recognition=recognition_decl or None,
            terms=self.term_keys,
            scoring=self.owned,
            external_inputs=external_inputs,
            subsystems=self._subsystems(A_deps, H),
            branch_on=self.config["branch_on"],
            planning_belief=self.config["planning_belief"],
            feedback=self.config["feedback"],
            controls=tuple(range(n_states, n_nodes)),
            action_names=tuple(actions),
            policy_len=self.config["policy_len"],
        )
        system = System(**kwargs)  # every refusal of the constructor is checked above
        params = SystemParams(
            precision=self._precision(),
            transition=transition,
            initial_state_prior=initial,
            likelihood=likelihood,
            recognition=recognition,
            obs_targets=obs_targets,
            state_targets=state_targets,
            terminal_targets=terminal_targets,
            scoring=self.scoring,
            policy_prior=policy_prior,
            control=dict(self.boxes),
            actuator=actuator,
            durations=dict(self.durations),
        )
        return system, params

    def _precision(self) -> Array | None:
        """The policy precision's clamp, from the variable ``configuration.planning.precision``
        names: a positive scalar (``None`` when the document names none)."""
        name = self.config["precision"]
        if name is None:
            return None
        address = "configuration.planning.precision"
        record = self.variables[name]
        if name in self.priors:
            raise self.fail(
                f"{address} names {name!r}, which has a prior record; the engine holds the policy "
                "precision fixed, so give its value under clamps"
            )
        if (
            not _is_leaf(record)
            or record.get("domain") not in (None, "positive")
            or record.get("shape") != []
        ):
            raise self.fail(
                f"variables[{name!r}] holds the policy precision ({address}): its record is "
                "{'domain': 'positive', 'shape': []}"
            )
        arr = self._array(self.clamps[name], [], "positive", f"clamps[{name!r}]")
        _check_domain("positive", arr, f"clamps[{name!r}]")
        return jnp.asarray(arr)

    def _recognition_parent(self, name: str) -> str:
        if name in self.observed:
            return f"obs:{name}"
        if name in self.external:
            return f"external:{name}"
        if name in self.aux:
            return f"recognition:{name}"
        return name

    def _check_frozen(self) -> None:
        hyper = {n for fac in self.priors.values() for n in _parameter_names(fac.parameters)}
        for name in sorted(self.frozen - self.gradient_vars):
            if name in self.priors or name in hyper:
                raise self.fail(
                    f"configuration.learning.frozen names {name!r}, a conjugate parameter, and the "
                    "engine holds one fixed only by a call-time learning rate of zero; drop it and "
                    "pass lr_by_slot at learn"
                )
            raise self.fail(
                f"configuration.learning.frozen names {name!r}, which no trainer moves; the list "
                "names the weights of networks and learned vectors"
            )

    def _subsystems(self, A_deps, H) -> tuple[tuple[str, Subsystem], ...] | None:
        """The per-component record of a composite, reconstructed from the name prefixes: what
        the document shows (Q18.2)."""
        if not self.components:
            return None
        out = []
        for name in self.components:
            prefix = f"{name}{_SEP}"
            facs = tuple(i for i, s in enumerate(self.states) if s.startswith(prefix))
            mods = tuple(m for m, o in enumerate(self.mods) if o.startswith(prefix))
            chosen = tuple(u for u in self.nodes if u.startswith(prefix))
            action_of = dict(zip(self.nodes, self.actions, strict=True))
            out.append(
                (
                    name,
                    Subsystem(
                        factors=facs,
                        modalities=mods,
                        controls=chosen,
                        actions=tuple((u, action_of[u]) for u in chosen),
                        free_modalities=tuple(m for m in mods if all(g in facs for g in A_deps[m])),
                        bound_modalities=tuple(
                            (m, tuple(g for g in A_deps[m] if g not in facs))
                            for m in mods
                            if any(g not in facs for g in A_deps[m])
                        ),
                        target_emissions=tuple(
                            (f, H[f]) for f in facs if H[f] and any(g not in facs for g in H[f])
                        ),
                    ),
                )
            )
        return tuple(out)


def _import(ref: str, address: str, fail: Callable[[str], DescriptionError]) -> Callable:
    module_name, sep, qualname = str(ref).partition(":")
    if not sep:
        raise fail(f"{address}: impl {ref!r} is not 'package.module:name'")
    try:
        obj: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            obj = getattr(obj, part)
    except (ImportError, AttributeError) as exc:
        raise fail(
            f"{address}: impl {ref!r} could not be imported ({exc}); a custom function enters by "
            "registry reference and is refused when unresolvable"
        ) from exc
    return obj


# ========================================================================= schema


def schema() -> dict[str, Any]:
    """JSON Schema (2020-12) for the document: keys, types, the clamp forms (a value, exactly one
    of ``ref`` or ``init``, or a tree of clamps), the role labels' time rules, the bracket form
    of a parent and the term record. What needs the model to check (shapes against the wiring,
    kinds from use) is :func:`build`'s.

    Also writes the checked-in copy (``python -m hybrid_aif.description --write-schema``).
    """
    names = {"type": "array", "items": {"type": "string", "minLength": 1}}
    number_or_list = {"anyOf": [{"type": "number"}, {"type": "array", "items": {"type": "number"}}]}
    record_properties = {
        "family": {"enum": sorted(FAMILIES)},
        "parents": {"type": "array", "items": {"$ref": "#/$defs/parent"}},
        "parameters": {"type": "object", "additionalProperties": {"$ref": "#/$defs/names"}},
        "options": {"type": "object"},
    }

    def record(steps: object) -> dict[str, Any]:
        properties = dict(record_properties)
        if steps is not None:
            properties["steps"] = steps
        return {
            "type": "object",
            "required": ["family"],
            "additionalProperties": False,
            "properties": properties,
        }

    labelled = {label: record({"const": _IMPLIED[label][1]}) for label in _IMPLIED}
    labelled.update({label: record(None) for label in _UNTIMED})
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        "title": "hybrid-aif model description",
        "description": (
            "A hybrid-aif System and its parameters as a document: variables with their domains, "
            "factors keyed by child and label, the relations no factor owns, clamps on parameter "
            "variables, the objective's terms and the configuration. A consumer's own data rides "
            "under 'extensions', keyed by consumer; no other unknown key is allowed. The format: "
            "docs/description-format.md."
        ),
        "type": "object",
        "required": [KEY, "variables", "configuration"],
        "additionalProperties": False,
        "properties": {
            KEY: {"const": VERSION},
            "library_version": {"type": "string"},
            "arrays_file": {"type": "string"},
            **{key: {"$ref": f"#/$defs/body/properties/{key}"} for key in _BODY_KEYS},
            EXTENSIONS: {
                "type": "object",
                "description": "Consumer data, keyed by consumer name; opaque to the library.",
                "additionalProperties": {"type": "object"},
            },
        },
        "$defs": {
            "parent": {"type": "string", "pattern": r"^[^\[\]]+(\[t(-1)?\])?$"},
            "names": {
                "anyOf": [
                    {"type": "string", "minLength": 1},
                    {"type": "array", "items": {"type": "string", "minLength": 1}},
                    {"type": "object", "additionalProperties": {"type": "string"}},
                ]
            },
            "leaf": {
                "type": "object",
                "additionalProperties": False,
                "minProperties": 1,
                "dependentSchemas": {
                    "levels": {
                        "required": ["domain"],
                        "properties": {"domain": {"const": "categorical"}},
                    },
                    "dim": {
                        "required": ["domain"],
                        "properties": {"domain": {"const": "real"}},
                        "not": {"required": ["shape"]},
                    },
                    "dims": {"required": ["dim"]},
                    "low": {"required": ["dim"]},
                    "high": {"required": ["dim"]},
                    "duration": {"required": ["levels"]},
                },
                "properties": {
                    "domain": {"enum": sorted({*PER_STEP_DOMAINS, *PARAMETER_DOMAINS})},
                    "levels": {"type": "integer", "minimum": 1},
                    "dim": {"type": "integer", "minimum": 1},
                    "dims": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string", "pattern": "^[^:.*\\[\\]]+$"},
                    },
                    "low": number_or_list,
                    "high": number_or_list,
                    "duration": {
                        "anyOf": [
                            {"type": "number", "exclusiveMinimum": 0},
                            {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "number", "exclusiveMinimum": 0},
                            },
                        ]
                    },
                    "shape": {"type": "array", "items": {"type": "integer", "minimum": 0}},
                },
            },
            "variable": {
                "anyOf": [
                    {"$ref": "#/$defs/leaf"},
                    {
                        "type": "object",
                        "minProperties": 1,
                        "additionalProperties": {"$ref": "#/$defs/variable"},
                    },
                    {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/variable"}},
                ]
            },
            "clamp": {
                "anyOf": [
                    {"type": "number"},
                    {"type": "array", "items": {"$ref": "#/$defs/clamp"}},
                    {
                        "type": "object",
                        "required": ["ref"],
                        "additionalProperties": False,
                        "properties": {"ref": {"type": "string", "pattern": "^[^#]+#.+$"}},
                    },
                    {
                        "type": "object",
                        "required": ["init"],
                        "additionalProperties": False,
                        "properties": {"init": {"type": "object"}},
                    },
                    {
                        "type": "object",
                        "not": {"anyOf": [{"required": ["ref"]}, {"required": ["init"]}]},
                        "additionalProperties": {"$ref": "#/$defs/clamp"},
                    },
                ]
            },
            "block": {
                "type": "object",
                "properties": labelled,
                "additionalProperties": record({"enum": list(STEPS)}),
            },
            "term": {
                "type": "object",
                "required": ["term"],
                "additionalProperties": False,
                "properties": {
                    "term": {"type": "string", "minLength": 1},
                    "on": {"type": "string", "minLength": 1},
                    "parameters": {
                        "type": "object",
                        "additionalProperties": {"$ref": "#/$defs/names"},
                    },
                    "options": {"type": "object"},
                    "model": {
                        "anyOf": [
                            {"$ref": "#/$defs/body"},
                            {
                                "type": "object",
                                "required": ["ref"],
                                "additionalProperties": False,
                                "properties": {"ref": {"type": "string"}},
                            },
                        ]
                    },
                },
            },
            "body": {
                "type": "object",
                "required": ["variables", "configuration"],
                "additionalProperties": False,
                "properties": {
                    "variables": {
                        "type": "object",
                        "propertyNames": {"pattern": r"^[^\[\]]+$"},
                        "additionalProperties": {"$ref": "#/$defs/variable"},
                    },
                    "factors": {
                        "type": "object",
                        "additionalProperties": {"$ref": "#/$defs/block"},
                    },
                    "relations": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "controls": {
                                "type": "object",
                                "additionalProperties": {"type": "string"},
                            },
                            "observed": names,
                            "external": names,
                        },
                    },
                    "clamps": {"type": "object", "additionalProperties": {"$ref": "#/$defs/clamp"}},
                    "objective": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "decomposition": {"enum": ["utility_infogain", "risk_ambiguity"]},
                            "terms": {"type": "array", "items": {"$ref": "#/$defs/term"}},
                        },
                    },
                    "configuration": {
                        "type": "object",
                        "required": ["planning"],
                        "additionalProperties": False,
                        "properties": {
                            "planning": {
                                "type": "object",
                                "required": ["horizon"],
                                "additionalProperties": False,
                                "properties": {
                                    "planner": {"type": "string"},
                                    "horizon": {"type": "integer", "minimum": 1},
                                    "gamma": {"type": "number"},
                                    "precision": {"type": "string", "minLength": 1},
                                    "branch_on": names,
                                    "belief": {"enum": list(_BELIEFS)},
                                    "feedback": {"enum": list(_FEEDBACK)},
                                },
                            },
                            "inference": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "iterations": {"type": "integer", "minimum": 1},
                                    "mixture_k": {
                                        "anyOf": [
                                            {"type": "integer", "minimum": 1},
                                            {"type": "null"},
                                        ]
                                    },
                                    "mixture_schedule": {"enum": list(_SCHEDULES)},
                                },
                            },
                            "learning": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {"frozen": names},
                            },
                        },
                    },
                },
            },
        },
    }


# ====================================================================== save/load


def _with_arrays_file(document: Mapping[str, Any], name: str | None) -> dict[str, Any]:
    """``document`` with every clamp's ref (nested models' included) pointing at ``name``."""

    def refs(node: Any) -> Any:
        if isinstance(node, Mapping):
            if set(node) == {"ref"} and isinstance(node["ref"], str) and "#" in node["ref"]:
                return {"ref": f"{name}#{node['ref'].rpartition('#')[2]}"}
            return {k: refs(v) for k, v in node.items()}
        if isinstance(node, list):
            return [refs(v) for v in node]
        return node

    def body(doc: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(doc)
        if "clamps" in out:
            out["clamps"] = refs(out["clamps"])
        objective = out.get("objective")
        if isinstance(objective, Mapping) and isinstance(objective.get("terms"), list):
            terms = []
            for term in objective["terms"]:
                if (
                    isinstance(term, Mapping)
                    and isinstance(term.get("model"), Mapping)
                    and ("variables" in term["model"])
                ):
                    term = {**term, "model": body(term["model"])}
                terms.append(term)
            out["objective"] = {**objective, "terms": terms}
        return out

    out = body(document)
    if name is None:
        out.pop("arrays_file", None)
    else:
        out["arrays_file"] = name
    return out


def save(
    description: Description, path: str | pathlib.Path
) -> tuple[pathlib.Path, pathlib.Path | None]:
    """Write ``<path>.json`` and, when there are refs, ``<path>.npz`` beside it.

    The document names the arrays file in ``arrays_file`` and every ref points into it, for
    :func:`load`.
    """
    path = pathlib.Path(path)
    stem = path.with_suffix("") if path.suffix in (".json", ".npz") else path
    json_path, npz_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
    if description.arrays:
        np.savez(npz_path, **description.arrays)
        document = _with_arrays_file(description.document, npz_path.name)
    else:
        document = _with_arrays_file(description.document, None)
        npz_path = None
    json_path.write_text(json.dumps(document, indent=2) + "\n")
    return json_path, npz_path


def load(path: str | pathlib.Path) -> Description:
    """Read a document written by :func:`save`, with its arrays file if it names one.

    A term's model given as ``{"ref": "<path>"}``, a document of its own resolved relative to
    this one, is nested in place, its arrays joining this document's under ``<term>/``.
    """
    json_path = pathlib.Path(path)
    if json_path.suffix != ".json":
        json_path = json_path.with_suffix(".json")
    document = json.loads(json_path.read_text())
    arrays: dict[str, np.ndarray] = {}
    if document.get("arrays_file"):
        with np.load(json_path.parent / document["arrays_file"]) as data:
            arrays = {k: np.asarray(data[k]) for k in data.files}
    objective = document.get("objective")
    if isinstance(objective, Mapping) and isinstance(objective.get("terms"), list):
        terms = []
        for term in objective["terms"]:
            model = term.get("model") if isinstance(term, Mapping) else None
            if isinstance(model, Mapping) and set(model) == {"ref"}:
                inner = load(json_path.parent / model["ref"])
                name = document.get("arrays_file") or ARRAYS_FILE
                nested = {k: v for k, v in inner.document.items() if k in _BODY_KEYS}

                if "clamps" in nested:
                    nested["clamps"] = _prefixed(nested["clamps"], name, term["term"])
                arrays.update({f"{term['term']}/{k}": v for k, v in inner.arrays.items()})
                if inner.arrays:
                    document["arrays_file"] = name
                term = {**term, "model": nested}
            terms.append(term)
        document["objective"] = {**objective, "terms": terms}
    return Description(document, arrays)


def _prefixed(node: Any, name: str, row: str) -> Any:
    """Clamps with every ref pointing at ``name`` under the key prefix ``<row>/``."""
    if isinstance(node, Mapping):
        if set(node) == {"ref"} and "#" in str(node["ref"]):
            return {"ref": f"{name}#{row}/{node['ref'].rpartition('#')[2]}"}
        return {k: _prefixed(v, name, row) for k, v in node.items()}
    if isinstance(node, list):
        return [_prefixed(v, name, row) for v in node]
    return node


def _schema_path() -> pathlib.Path:
    return pathlib.Path(__file__).parent / f"description-{VERSION}.json"


if __name__ == "__main__":  # pragma: no cover
    import sys

    if "--write-schema" in sys.argv:
        target = _schema_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(schema(), indent=2) + "\n")
        print(f"wrote {target}")
