"""A System's faces compiled once: :meth:`System.compiled` returns a :class:`CompiledSystem`.

Each face of the eager System (``infer_states``, ``infer_policies``, ``act``,
``update_empirical_prior``, ``learn``) is an ``eqx.filter_jit`` boundary of this object, with the
System and any planner callable static and only values traced. Ahead of the boundary the inputs
are canonicalised, so the input-type differences that make a loop retrace cannot occur:

- parameters, beliefs, priors and the learn carries: NumPy leaves become JAX arrays and weak
  types become strong, each leaf keeping its dtype;
- observations, ``externals`` and ``action``: a Python or NumPy number or array
  becomes a strong array; an integer becomes the default integer dtype (int32 unless x64 is
  enabled), a float keeps its dtype unless it is narrower than float32, and a Python float
  becomes the default float dtype;
- schedules (``lr``, ``decay``, ``alpha``, the per-slot rates): a Python number becomes an array
  of the default float dtype, except a Python ``lr`` of exactly 0.0 and a Python ``decay`` of
  exactly 1.0, which ``learn`` reads at trace time and which stay static;
- an observation of the wrong form is refused with a sentence before the boundary, by the eager
  faces' own check;
- ``legal``: a list is read as the eager faces read it, each mask becomes a bool array (an
  integer mask is read as bool; a float mask is refused), and a mask the eager faces refuse
  (every level of a node illegal at some step, a bare mask of the wrong length) is refused
  before the boundary, with the eager sentence, when the System's controls are all discrete
  and, for a bare mask, the planner is the System's enumeration;
- ``present``: each flag becomes a bool array;
- PRNG keys: typed keys become raw ``uint32`` key data;
- ``priors``: ``None`` becomes :meth:`System.initial_priors`, and the result is completed to
  the key set :meth:`System.update_empirical_prior` returns: a discrete control node's entry
  (uniform), and an observed action node's entry (uniform), so the first tick shares a compile
  with the later ones.
  A mixture prior declared with fewer components than ``mixture_k`` still differs from a pushed
  one, as it does eagerly; pad it with :func:`hybrid_aif.objectives.hybrid.pad_mixture` to
  remove the extra trace.

Equal Systems share compiled programs: two compiled objects of one System, or a view equal to
its composite, reuse one compile, and a ``trace_counts`` entry counts only the traces its own
calls caused. The face signatures and return structures are the eager faces'. Static values stay
static: a face's ``metrics``, ``detail``, ``selection``, ``mode``, ``use_bound`` and
``support_floor`` arguments, the planner callable, and every field of the System. Each face is
watched by :func:`hybrid_aif.debug.watch`: a second trace of a face emits a
:class:`~hybrid_aif.debug.RetraceWarning` naming what changed, comparing against the last trace
of the same face and System by value, whichever compiled object made it.

Design: docs/design/recompilation.md R6, R7.
"""

from __future__ import annotations

import collections
import contextlib
import functools
import inspect
import weakref
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hybrid_aif import debug
from hybrid_aif.distributions import Categorical
from hybrid_aif.inference.discrete import SUPPORT_FLOOR
from hybrid_aif.system import keyed, validation

if TYPE_CHECKING:
    from hybrid_aif.system.core import System, SystemParams

FACES = ("infer_states", "infer_policies", "act", "update_empirical_prior", "learn")


# ------------------------------------------------------------------------- canonicalisation


def _is_seq(x: Any) -> bool:
    return isinstance(x, list | tuple)


def _strong(x: Any) -> Any:
    """An array leaf as a strong-typed ``jax.Array`` of its own dtype (a NumPy array becomes a
    JAX array); other leaves unchanged."""
    if isinstance(x, jax.Array | np.ndarray | np.generic):
        x = jnp.asarray(x)
        if jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
            return x
        return jax.lax.convert_element_type(x, x.dtype)
    return x


def _float_dtype(dtype: Any) -> Any:
    """``dtype`` unless it is narrower than float32."""
    return jnp.promote_types(dtype, jnp.float32)


def _value(x: Any) -> Any:
    """A value leaf as a strong array: integers in the default integer dtype, floats in their
    own dtype but at least float32 (a Python float in the default float dtype), booleans as
    bool. Lists, tuples, strings and ``None`` are left for the face's own refusal."""
    if x is None or _is_seq(x) or isinstance(x, str):
        return x
    if isinstance(x, bool | np.bool_):
        return jnp.asarray(x, jnp.bool_)
    if isinstance(x, int | np.integer):
        return jnp.asarray(x, jnp.result_type(int))
    if isinstance(x, float | np.floating):
        return jnp.asarray(x, jnp.result_type(float))
    if isinstance(x, jax.Array | np.ndarray):
        x = jnp.asarray(x)
        if x.dtype == jnp.bool_:
            return jax.lax.convert_element_type(x, jnp.bool_)
        if jnp.issubdtype(x.dtype, jnp.integer):
            return jax.lax.convert_element_type(x, jnp.result_type(int))
        if jnp.issubdtype(x.dtype, jnp.floating):
            return jax.lax.convert_element_type(x, _float_dtype(x.dtype))
        return _strong(x)
    return x


def _tree(fn: Callable, tree: Any, *, seq_leaf: bool = False) -> Any:
    if tree is None:
        return None
    is_leaf = _is_seq if seq_leaf else None
    return jax.tree_util.tree_map(fn, tree, is_leaf=is_leaf)


def _values(tree: Any) -> Any:
    return _tree(_value, tree, seq_leaf=True)


def _flag(x: Any) -> Any:
    """A ``present`` flag as a bool array (a list or tuple of flags too)."""
    if x is None:
        return x
    return jnp.asarray(x, jnp.bool_)


def _key(x: Any) -> Any:
    """A PRNG key as raw ``uint32`` key data, whichever form it came in."""
    if x is None:
        return None
    x = jnp.asarray(x) if not isinstance(x, jax.Array) else x
    if jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
        return jax.random.key_data(x)
    return _strong(x)


def _rate(x: Any, static: float | None = None) -> Any:
    """A rate in the default float dtype (an array keeps its own dtype if it is at least
    float32), except a Python number equal to ``static``, which ``learn`` reads at trace time
    (a learning rate of 0.0 skips the slot, a decay of 1.0 is no decay and needs no anchor) and
    so stays a Python number."""

    def leaf(v: Any) -> Any:
        if isinstance(v, float | int | np.floating | np.integer) and not isinstance(v, bool):
            if static is not None and isinstance(v, float | int) and v == static:
                return v
            return jnp.asarray(v, jnp.result_type(float))
        v = jnp.asarray(v)
        return jax.lax.convert_element_type(v, _float_dtype(v.dtype))

    return _tree(leaf, x)


def _mask(x: Any, face: str, what: str) -> Any:
    """One ``legal`` mask as a bool array: a bool or integer mask is read as bool, a float mask
    is refused."""
    if x is None:
        return None
    arr = jnp.asarray(x)
    if arr.dtype == jnp.bool_:
        return arr
    if jnp.issubdtype(arr.dtype, jnp.integer):
        return arr != 0
    raise ValueError(
        f"{face}: {what} is a {arr.dtype} array, and a legal mask is boolean: pass True for a "
        "legal level and False for an illegal one"
    )


def _legal(system, params, legal, planner, face: str):
    """``legal=`` as the bool masks the faces read, refused with the eager sentence when the
    eager faces would refuse it (a concrete mask that leaves no legal level, a bare mask of the
    wrong length)."""
    if legal is None:
        return None
    bare = not isinstance(legal, Mapping | list | tuple)
    if isinstance(legal, list):
        legal = keyed.legal_list(legal, face)
        bare = True
    if isinstance(legal, Mapping):
        legal = {k: _mask(v, face, f"legal[{k!r}]") for k, v in legal.items()}
    elif not _is_seq(legal):
        legal = _mask(legal, face, "legal")
    else:
        return legal  # the per-state-factor spelling: the face raises, naming the keyed form
    inner = system._params_in(params, face)
    if system._control_layout(inner) is None and (
        not bare or (planner is None and system.planner == "enumeration")
    ):
        system._legal_logits(inner, legal, system.policies(params), face)
    return legal


def _complete_priors(system, params, priors, face: str):
    """``priors`` (``None`` for the initial priors) with every key ``update_empirical_prior``
    returns: each discrete control node's entry, uniform when absent, and each observed action
    node's entry, uniform when absent."""
    out = dict(system.initial_priors(params) if priors is None else priors)
    params = system._params_in(params, face)
    for name, q in system._node_priors(params).items():
        out.setdefault(name, q)
    if system._sensed_columns:
        system._check_sensed_kinds(params, face)
        levels = system._control_levels(params)
        for k, name in zip(system._sensed_columns, system._sensed, strict=True):
            if name not in out:
                out[name] = Categorical(logits=jnp.zeros((levels[k],)))
    return out


# ----------------------------------------------------------------------- the jit boundaries
#
# Module-level pure functions, one per face, with the System first, jitted once here: every
# compiled object shares them, so equal Systems share programs. ``eqx.filter_jit`` treats the
# System (all static fields) and every non-array argument as static and every array as traced.
# The body of each runs once per trace, which is what ``_TRACES`` counts.

_TRACES: collections.Counter = collections.Counter()
_SNAPSHOT_LIMIT = 256


class _SnapshotStore(collections.OrderedDict):
    """Bounded LRU store of the last traced signature per ``(face, System)``."""

    def __init__(self, limit: int = _SNAPSHOT_LIMIT) -> None:
        super().__init__()
        self.limit = limit

    def get(self, key: Any, default: Any = None) -> Any:
        if key in self:
            self.move_to_end(key)
            return self[key]
        return default

    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self.limit:
            self.popitem(last=False)


#: The last traced call signature per ``(face, System)``, shared by every watcher.
_SNAPSHOTS: _SnapshotStore = _SnapshotStore()


def _counted(face: str) -> Callable:
    def decorate(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args):
            _TRACES[face] += 1
            return fn(*args)

        return wrapper

    return decorate


@_counted("infer_states")
def _infer_states(
    system,
    params,
    observations,
    priors,
    externals,
    present,
    key,
    detail,
    support_floor,
):
    return system.infer_states(
        params,
        observations,
        priors,
        detail=detail,
        externals=externals,
        present=present,
        support_floor=support_floor,
        key=key,
    )


@_counted("infer_policies")
def _infer_policies(
    system, params, beliefs, planner, use_bound, metrics, plan_key, legal, present, return_policies
):
    extra = {"return_policies": True} if return_policies else {}
    return system.infer_policies(
        params,
        beliefs,
        planner,
        use_bound=use_bound,
        metrics=metrics,
        plan_key=plan_key,
        legal=legal,
        present=present,
        **extra,
    )


@_counted("act")
def _act(
    system,
    key,
    params,
    beliefs,
    selection,
    alpha,
    mode,
    planner,
    use_bound,
    metrics,
    plan_key,
    legal,
):
    return system.act(
        key,
        params,
        beliefs,
        selection=selection,
        alpha=alpha,
        mode=mode,
        planner=planner,
        use_bound=use_bound,
        metrics=metrics,
        plan_key=plan_key,
        legal=legal,
    )


@_counted("update_empirical_prior")
def _update_empirical_prior(system, params, beliefs, action, key):
    return system.update_empirical_prior(params, beliefs, action, key)


@_counted("learn")
def _learn(
    system,
    params,
    observations,
    beliefs,
    prev_beliefs,
    action,
    lr,
    decay,
    lr_by_slot,
    decay_by_slot,
    detail,
    learn_state,
    prior,
    priors,
    externals,
):
    return system.learn(
        params,
        observations,
        beliefs,
        prev_beliefs=prev_beliefs,
        action=action,
        lr=lr,
        decay=decay,
        lr_by_slot=lr_by_slot,
        decay_by_slot=decay_by_slot,
        detail=detail,
        learn_state=learn_state,
        prior=prior,
        priors=priors,
        externals=externals,
    )


_IMPLS = {
    "infer_states": _infer_states,
    "infer_policies": _infer_policies,
    "act": _act,
    "update_empirical_prior": _update_empirical_prior,
    "learn": _learn,
}
_JITS = {face: eqx.filter_jit(impl) for face, impl in _IMPLS.items()}
_ARGNAMES = {face: tuple(inspect.signature(impl).parameters) for face, impl in _IMPLS.items()}


class CompiledSystem:
    """The faces of one :class:`~hybrid_aif.system.System`, each compiled once.

    Built by :meth:`System.compiled`. The methods take the arguments of the System's eager faces
    and return what they return; the System itself is ``system``. ``trace_counts`` maps each face
    that has traced through this object to the number of its traces; a steady loop holds every
    count at one. ``watch=False`` leaves the counts and drops the
    :class:`~hybrid_aif.debug.RetraceWarning`.
    """

    def __init__(self, system: System, *, watch: bool = True):
        self.system = system
        try:
            hash(system)
            ident: Any = system
        except TypeError:  # pragma: no cover  a System holding an unhashable field
            ident = id(system)
            with contextlib.suppress(TypeError):
                weakref.finalize(
                    system,
                    lambda i=ident: [_SNAPSHOTS.pop((face, i), None) for face in FACES],
                )
        self._watched = {
            face: debug.watch(
                _JITS[face],
                name=f"CompiledSystem.{face}",
                argnames=_ARGNAMES[face],
                warn=watch,
                clock=lambda face=face: _TRACES[face],
                store=_SNAPSHOTS,
                key=(face, ident),
            )
            for face in FACES
        }

    @property
    def trace_counts(self) -> dict[str, int]:
        """Traces so far per face that has traced through this object."""
        return {f: w.traces for f, w in self._watched.items() if w.traces}

    def _call(self, face: str, *args):
        return self._watched[face](self.system, *args)

    def _observations(self, params, observations, face: str):
        system = self.system
        params = system._params_in(params, face)
        names = system._names(params)
        raw = keyed.observations_in(names, observations, face)
        validation._check_observations(names.modalities, params, raw, face)
        return _values(observations)

    # ------------------------------------------------------------------------------ faces

    def infer_states(
        self,
        params: SystemParams,
        observations: Mapping[str, Array | None],
        priors: Mapping[str, Any] | None = None,
        detail: bool = False,
        externals: Mapping[str, Array] | None = None,
        present: Mapping[str, bool | Array] | None = None,
        support_floor: float = SUPPORT_FLOOR,
        key: Array | None = None,
    ):
        """:meth:`System.infer_states`, compiled: ``(beliefs, detail)``."""
        self.system._refuse_black_box("infer_states")
        observations = self._observations(params, observations, "infer_states")
        priors = _complete_priors(self.system, params, priors, "infer_states")
        return self._call(
            "infer_states",
            _tree(_strong, params),
            observations,
            _tree(_strong, priors),
            _values(externals),
            _tree(_flag, present, seq_leaf=True),
            _key(key),
            detail,
            support_floor,
        )

    def infer_policies(
        self,
        params: SystemParams,
        beliefs: Mapping[str, Any],
        planner: Callable | None = None,
        use_bound: str = "midpoint",
        metrics: bool = True,
        plan_key: Array | None = None,
        legal: Mapping[str, Array] | Array | None = None,
        present: Mapping[str, bool | Array] | None = None,
        return_policies: bool = False,
    ):
        """:meth:`System.infer_policies`, compiled: ``(q_pi, tree)``."""
        return self._call(
            "infer_policies",
            _tree(_strong, params),
            _tree(_strong, beliefs),
            planner,
            use_bound,
            metrics,
            _key(plan_key),
            _legal(self.system, params, legal, planner, "infer_policies"),
            _tree(_flag, present, seq_leaf=True),
            return_policies,
        )

    def act(
        self,
        key: Array,
        params: SystemParams,
        beliefs: Mapping[str, Any],
        selection: Literal["stochastic", "deterministic"] = "stochastic",
        alpha: float | Array | None = None,
        mode: Literal["marginal", "policy"] | None = None,
        planner: Callable | None = None,
        use_bound: str = "midpoint",
        metrics: bool = True,
        plan_key: Array | None = None,
        legal: Mapping[str, Array] | Array | None = None,
    ):
        """:meth:`System.act`, compiled: ``(action, q_pi, metrics)``; ``key`` is traced."""
        return self._call(
            "act",
            _key(key),
            _tree(_strong, params),
            _tree(_strong, beliefs),
            selection,
            _rate(alpha),
            mode,
            planner,
            use_bound,
            metrics,
            _key(plan_key),
            _legal(self.system, params, legal, planner, "act"),
        )

    def update_empirical_prior(
        self,
        params: SystemParams,
        beliefs: Mapping[str, Any],
        action: Mapping[str, Array],
        key: Array | None = None,
    ):
        """:meth:`System.update_empirical_prior`, compiled."""
        return self._call(
            "update_empirical_prior",
            _tree(_strong, params),
            _tree(_strong, beliefs),
            _values(action),
            _key(key),
        )

    def learn(
        self,
        params: SystemParams,
        observations: Mapping[str, Array],
        beliefs: Mapping[str, Any],
        prev_beliefs: Mapping[str, Any] | None = None,
        action: Mapping[str, Array] | None = None,
        lr: float | Array = 1.0,
        decay: float | Array = 1.0,
        lr_by_slot: Mapping[str, float | Array] | None = None,
        decay_by_slot: Mapping[str, float | Array] | None = None,
        detail: Any = None,
        learn_state: Any = None,
        prior: SystemParams | None = None,
        priors: Mapping[str, Any] | None = None,
        externals: Mapping[str, Array] | None = None,
    ):
        """:meth:`System.learn`, compiled: the parameters, or ``(parameters, LearnState)``."""
        observations = self._observations(params, observations, "learn")
        return self._call(
            "learn",
            _tree(_strong, params),
            observations,
            _tree(_strong, beliefs),
            _tree(_strong, prev_beliefs),
            _values(action),
            _rate(lr, 0.0),
            _rate(decay, 1.0),
            _rate(lr_by_slot, 0.0),
            _rate(decay_by_slot, 1.0),
            _tree(_strong, detail),
            _tree(_strong, learn_state),
            _tree(_strong, prior),
            _tree(_strong, priors),
            _values(externals),
        )

    def __getattr__(self, name: str):
        # every other System attribute (names, policies, as_env, initial_priors, ...) is the
        # System's own, so a loop needs one object
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.system, name)
