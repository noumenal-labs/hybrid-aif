"""Control nodes: the table every consumer reads, and the space their values live in.

A System's control nodes are ordinary nodes of its factor table that it sets by decision.
:func:`control_table` derives the :class:`ControlTable` (which factors read which node, at
which parent position; the pymdp-style :class:`SugarView` when there is one), and
:func:`control_index` / :func:`control_read` / :func:`take_controls` / :func:`parent_levels` slice
a transition by a control step holding one level per control node.

A node is *continuous* (a float vector) or *discrete* (a level index), and the readers of a
node agree on which: a table indexes its control axis by a level, a linear-Gaussian transition
reads a level as a mode index and a float vector through its ``G u`` term, a neural transition
reads a level as its one-hot and a float vector as part of its input. A continuous node's
support is a :class:`Box`, held in ``SystemParams.control`` under the node's name;
:func:`control_space` reads each node's kind and size off the params (:class:`ControlSpace`).
Over an all-continuous space a control step is a flat float vector, the nodes' blocks
concatenated in node order (:class:`ControlLayout`); over a mixed space a step is a
:class:`ControlStep`, one entry per node.

Design: docs/design/control-states.md C1, C4, C6, C11; docs/design/amortization.md A9
"""

import dataclasses
import functools
from collections.abc import Mapping
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import xlogy

from hybrid_aif.interval import Interval

#: A node read as a level index.
LEVEL = "level"
#: A node read as a float vector.
FLOAT = "float"


class Box(eqx.Module):
    """A continuous control node's support: ``dim`` numbers in ``[low, high]`` elementwise.

    ``low``/``high`` are float ``(dim,)`` arrays (scalars broadcast to the other's shape); every
    bound is finite and ``low < high`` in every dimension. Concrete bounds are checked here and
    again where :func:`control_space` reads the box; traced ones there, by ``equinox.error_if``.
    """

    low: Array
    high: Array

    def __init__(self, low: Array, high: Array):
        low, high = jnp.asarray(low, jnp.float32), jnp.asarray(high, jnp.float32)
        shape = jnp.broadcast_shapes(low.shape, high.shape)
        if len(shape) != 1 or shape[0] == 0:
            raise ValueError(
                f"Box takes (dim,) bounds with dim ≥ 1, got shapes {low.shape} and {high.shape}: "
                "pass one low and one high per dimension"
            )
        self.low, self.high = jnp.broadcast_to(low, shape), jnp.broadcast_to(high, shape)
        _check_bounds("Box", self.low, self.high)

    @property
    def dim(self) -> int:
        return self.low.shape[-1]


def _bound_sentences(what: str) -> tuple[str, str]:
    return (
        f"{what} has a non-finite bound: a control box is finite in every dimension, so pass "
        "finite low and high",
        f"{what} has low >= high in some dimension: pass low < high in every dimension of the box",
    )


def _check_bounds(what: str, low: Array, high: Array) -> bool:
    """Raise on concrete non-finite or unordered bounds; ``False`` when they are traced."""
    infinite, inverted = _bound_sentences(what)
    finite = jnp.all(jnp.isfinite(low) & jnp.isfinite(high))
    ordered = jnp.all(low < high)
    if isinstance(finite, jax.core.Tracer) or isinstance(ordered, jax.core.Tracer):
        return False
    if not bool(finite):
        raise ValueError(infinite)
    if not bool(ordered):
        raise ValueError(inverted)
    return True


class ControlLayout(NamedTuple):
    """Flat float layout of a control step over an all-continuous control space.

    ``node_spans[k]`` is control node ``k``'s ``(start, stop)`` slice of the flat vector, in node
    order; ``factor_nodes[f]`` lists the nodes state factor ``f`` reads, in its parent order.
    ``low``/``high`` are the concatenated box bounds, ``(action_dim,)`` float arrays read from
    ``SystemParams.control``. Built per call from the params (:func:`control_space`).
    """

    node_spans: tuple[tuple[int, int], ...]
    factor_nodes: tuple[tuple[int, ...], ...]
    action_dim: int
    low: Array
    high: Array

    @property
    def bounds(self) -> tuple[Array, Array]:
        """``(low, high)`` as float arrays of shape ``(action_dim,)``."""
        return jnp.asarray(self.low, jnp.float32), jnp.asarray(self.high, jnp.float32)

    def structure(self) -> "ControlLayout":
        """The layout without its bounds: hashable, for static fields."""
        return self._replace(low=None, high=None)

    def dim_of(self, f: int) -> int | None:
        """State factor ``f``'s control width (its nodes' widths summed), or ``None`` if it reads
        no control node."""
        nodes = self.factor_nodes[f]
        if not nodes:
            return None
        return sum(self.node_spans[k][1] - self.node_spans[k][0] for k in nodes)

    def node_block(self, k: int, action: Array) -> Array:
        """Control node ``k``'s block of the flat vector."""
        start, stop = self.node_spans[k]
        return jnp.asarray(action)[..., start:stop]

    def block(self, f: int, action: Array) -> Array:
        """State factor ``f``'s control input: its nodes' blocks, in its parent order."""
        nodes = self.factor_nodes[f]
        if not nodes:
            raise ValueError(f"factor {f} reads no continuous control")
        if len(nodes) == 1:
            return self.node_block(nodes[0], action)
        return jnp.concatenate([self.node_block(k, action) for k in nodes], axis=-1)


class ControlSpace(NamedTuple):
    """A System's control nodes as its params declare them: per node its kind (:data:`LEVEL` or
    :data:`FLOAT`) and size (a level count, or a float width), the box of each continuous node
    (``None`` for a discrete one), and the flat :class:`ControlLayout` when every node is
    continuous. See :func:`control_space`."""

    kinds: tuple[str, ...]
    sizes: tuple[int, ...]
    boxes: tuple[object, ...]
    layout: ControlLayout | None

    @property
    def continuous(self) -> bool:
        """Every node is continuous (and there is at least one)."""
        return bool(self.kinds) and all(k == FLOAT for k in self.kinds)

    @property
    def mixed(self) -> bool:
        """The space holds both a discrete and a continuous node."""
        return LEVEL in self.kinds and FLOAT in self.kinds

    @property
    def levels(self) -> tuple[int, ...]:
        """Each node's level count, ``1`` for a continuous node."""
        return tuple(s if k == LEVEL else 1 for k, s in zip(self.kinds, self.sizes, strict=True))


def soft_clip(action: Array, bounds: tuple[Array, Array] | None) -> Array:
    """Bring an action into its box by a stop-gradiented rescale, not a clip.

    With ``centre = (high + low)/2`` and ``half = (high − low)/2``::

        offset = action − centre
        action' = centre + offset · sg(half / max(half, |offset|))

    The forward value equals ``clip(action, low, high)``; the derivative outside the box is
    ``half/|offset|`` rather than zero. On a box symmetric about zero this is DreamerV3's
    ``action_clip``. ``bounds=None`` is the identity. Apply it wherever a transition reads a
    continuous control node's value and nowhere else: the replay column, an imagined
    trajectory's action and every log-probability that reads it see the raw draw.

    Design: docs/design/amortization.md (the ``rssm.action_clip`` row)
    """
    if bounds is None:
        return action
    low, high = (jnp.asarray(b, jnp.float32) for b in bounds)
    centre, half = (high + low) / 2.0, (high - low) / 2.0
    offset = jnp.asarray(action, jnp.float32) - centre
    scale = half / jnp.maximum(half, jnp.abs(offset))
    return centre + offset * jax.lax.stop_gradient(scale)


# ------------------------------------------------------------------ shared discrete controls

ControlDeps = tuple[tuple[int, ...], ...]


def effective_control_deps(
    control_deps: ControlDeps | None, controlled: tuple[bool, ...]
) -> ControlDeps:
    """Per factor, the controlled factors whose control slices its transition.

    ``None`` (the default) is each controlled factor's own control and nothing for the rest.
    """
    if control_deps is not None:
        return control_deps
    return tuple((f,) if c else () for f, c in enumerate(controlled))


def num_control_axes(control_deps: "ControlDeps | ControlTable | None", f: int) -> int:
    """How many trailing control axes factor ``f``'s transition table carries.

    One per control parent (per entry of ``control_deps[f]``, or of a
    :class:`ControlTable`'s ``control_parents[f]``); a factor with none keeps the single
    trailing axis, of size 1, as does every factor under ``None``.
    """
    if control_deps is None:
        return 1
    if isinstance(control_deps, ControlTable):
        return max(1, len(control_deps.control_parents[f]))
    return max(1, len(control_deps[f]))


class ControlStep(NamedTuple):
    """One step of a policy, positionally: one entry per control node, in node order.
    A discrete node's entry is an int32 level (a clamp) or a
    probability vector over its levels (a belief); a continuous node's is its float value (a
    clamp) or a :class:`~hybrid_aif.distributions.Gaussian` /
    :class:`~hybrid_aif.distributions.TruncatedGaussian` over it (a belief). Every reader
    dispatches statically: on the node's kind, then on the entry's type (a level indexes a
    table's control axis, a probability vector contracts it, :func:`take_controls`). Leading
    axes (policies, steps) are the caller's, as for the positional level array.
    """

    entries: tuple[object, ...]


def is_belief(index: Array) -> bool:
    """Whether a discrete node's entry is a probability vector (a belief) rather than a level.
    Static."""
    return jnp.issubdtype(jnp.result_type(index), jnp.floating)


def is_distribution(entry: object) -> bool:
    """Whether a continuous node's entry is a belief (a distribution) rather than a value."""
    return hasattr(entry, "log_prob") and hasattr(entry, "mean")


class ControlTarget(NamedTuple):
    """A target on a control node, as the planning closures read it: the node's ``column``,
    its ``name`` (the leaf's key) and whether it scores the last step only (a terminal target).
    A discrete node's target is ``log_target``, its normalized log-probabilities over the levels;
    a continuous node's is ``target``, a density over its values, with ``span`` its block of a
    flat control step (``None`` when steps arrive as :class:`ControlStep`)."""

    column: int
    name: str
    log_target: Array | None
    terminal: bool
    target: object = None
    span: tuple[int, int] | None = None


def node_risk(entry: Array, log_target: Array) -> Interval:
    """``KL(q(u) ‖ C)`` for one discrete control node at one step: ``−log C(u)`` when ``entry``
    is a level (the clamp, ``KL(δ(u) ‖ C)``), the full KL when it is a probability vector."""
    if not is_belief(entry):
        return Interval.exact(-jnp.take(log_target, entry, axis=-1))
    #  a level with no belief mass contributes nothing, even where the target forbids it
    cross = jnp.where(entry > 0, entry * log_target, 0.0)
    return Interval.exact(jnp.sum(xlogy(entry, entry) - cross, axis=-1))


def continuous_node_risk(entry: object, target: object) -> Interval:
    """``KL(q(u) ‖ C)`` for one continuous control node at one step.

    A value ``u`` (the clamp) gives ``−log C(u)``. A belief gives the KL: a Gaussian against a
    Gaussian target in closed form; a truncated Gaussian against a truncated target on the same
    box in closed form; a truncated Gaussian against a Gaussian target as ``−H[q] − E_q[log C]``,
    the cross term read off ``q``'s closed mean and variance; a Gaussian against a truncated
    target is infinite (the belief has mass outside the target's box).
    """
    from hybrid_aif.distributions import Gaussian, TruncatedGaussian

    if not is_distribution(entry):
        return Interval.exact(-target.log_prob(jnp.asarray(entry, jnp.float32)))
    if isinstance(entry, TruncatedGaussian) and isinstance(target, Gaussian):
        return Interval.exact(-entry.entropy().lower - _expected_log_gaussian(entry, target))
    if isinstance(entry, Gaussian) and isinstance(target, TruncatedGaussian):
        #  a Gaussian belief puts mass outside the target's box, where the target has none
        return Interval.exact(jnp.full(entry.batch_shape, jnp.inf, jnp.float32))
    return entry.kl_to(target)


def _expected_log_gaussian(q, target) -> Array:
    """``E_q[log N(u; m, S)]`` from ``q``'s mean and diagonal variance."""
    from jax.scipy.linalg import cho_solve, solve_triangular

    dev = q.mean() - target.loc
    z = solve_triangular(target.chol, dev[..., None], lower=True)[..., 0]
    prec = cho_solve((target.chol, True), jnp.broadcast_to(jnp.eye(target.dim), target.chol.shape))
    trace = jnp.sum(jnp.diagonal(prec, axis1=-2, axis2=-1) * q.variance(), axis=-1)
    half_log_det = jnp.sum(jnp.log(jnp.diagonal(target.chol, axis1=-2, axis2=-1)), axis=-1)
    log_2pi = jnp.log(2.0 * jnp.pi)
    return -0.5 * (target.dim * log_2pi + trace + jnp.sum(z**2, axis=-1)) - half_log_det


def control_risk(step_terms_fn, control_targets: tuple[ControlTarget, ...], horizon: int):
    """``step_terms_fn`` with a ``risk`` leaf per control node with a target, keyed by its name
    (a
    terminal target: a ``terminal_risk`` leaf, nonzero at the last of ``horizon`` steps). The
    node's value at a step is the step's entry: a level or a :class:`ControlStep`'s
    probability vector for a discrete node, a float block or a :class:`ControlStep`'s belief
    for a continuous one. Further arguments (the amortized closures' previous beliefs) pass
    through."""

    def step_terms(q, action, t, *rest):
        tree = dict(step_terms_fn(q, action, t, *rest))
        for ct in control_targets:
            if isinstance(action, ControlStep):
                entry = action.entries[ct.column]
            elif ct.span is not None:
                entry = jnp.asarray(action)[..., ct.span[0] : ct.span[1]]
            else:
                entry = action[ct.column]
            leaf = (
                node_risk(entry, ct.log_target)
                if ct.target is None
                else continuous_node_risk(entry, ct.target)
            )
            term = "terminal_risk" if ct.terminal else "risk"
            if ct.terminal:
                last = t == horizon - 1
                leaf = Interval(
                    jnp.where(last, leaf.lower, 0.0), jnp.where(last, leaf.upper, 0.0), leaf.derived
                )
            tree[term] = {**tree.get(term, {}), ct.name: leaf}
        return tree

    return step_terms


def _batched_zero(action: Array) -> Array:
    """Level 0, carrying ``action``'s batching: the no-op read of a size-1 control axis."""
    if action.shape[-1:] == (0,) or not jnp.issubdtype(action.dtype, jnp.integer):
        return jnp.zeros(action.shape[:-1], jnp.int32)
    return action[..., 0] * 0


def control_index(
    action: Array, f: int, control_deps: "ControlDeps | ControlTable | None"
) -> tuple[Array, ...]:
    """The level indices into factor ``f``'s trailing control axes, in axis order.

    Given a :class:`ControlTable`, ``action`` (a control step) holds one level per control node
    (or is a :class:`ControlStep`) and factor ``f`` reads its control parents' columns; a factor
    with none reads level 0 of its size-1 axis (carrying the step's batching under the default
    wiring, a constant otherwise).
    Given a per-factor declaration, ``action`` holds one level per factor: ``None`` reads
    ``action[f]``, an empty entry level 0, a non-empty one each listed factor's level.
    """
    if isinstance(control_deps, ControlTable):
        parents = control_deps.control_parents[f]
        if isinstance(action, ControlStep):
            return tuple(action.entries[k] for k, _ in parents) if parents else (jnp.int32(0),)
        if parents:
            return tuple(action[k] for k, _ in parents)
        if control_deps.default_wiring:
            return (_batched_zero(action),)
        return (jnp.int32(0),)
    if control_deps is None:
        return (action[f],)
    deps = control_deps[f]
    return tuple(action[c] for c in deps) if deps else (jnp.int32(0),)


class ControlRead(NamedTuple):
    """Levels for control axes that do not all trail: ``index[i]`` reads the parent axis at
    ``positions[i]`` of a transition with ``arity`` parents (see :func:`control_read`)."""

    index: tuple[Array, ...]
    positions: tuple[int, ...]
    arity: int


def control_read(
    action: Array, f: int, control_deps: "ControlDeps | ControlTable | None"
) -> "tuple[Array, ...] | ControlRead":
    """What :func:`take_controls` needs to slice factor ``f``'s control axes: the levels of
    :func:`control_index`, as a :class:`ControlRead` when a :class:`ControlTable` places the
    control parents anywhere but after the state parents."""
    index = control_index(action, f, control_deps)
    if isinstance(control_deps, ControlTable):
        parents, arity = control_deps.control_parents[f], control_deps.arity[f]
        positions = tuple(pos for _, pos in parents)
        if parents and positions != tuple(range(arity - len(parents), arity)):
            return ControlRead(index, positions, arity)
    return index


def declared_parents(table: "ControlTable", f: int) -> tuple[tuple[str, int], ...]:
    """Factor ``f``'s parents in declared (``B_deps``) order: ``("state", g)`` for a state
    factor, ``("control", k)`` for control node column ``k``."""
    at = {pos: k for k, pos in table.control_parents[f]}
    states = iter(table.B_deps[f])
    return tuple(
        ("control", at[pos]) if pos in at else ("state", next(states))
        for pos in range(table.arity[f])
    )


class LGRead(NamedTuple):
    """How a linear-Gaussian factor reads its parents: ``axes`` lists its mode axes in order,
    each ``("state", g)`` or ``("control", k)`` (a discrete parent); ``floats`` the continuous
    control nodes its ``G u`` term reads, in declared order. With no mode axis the declaration
    carries the trivial ``(1,)`` axis."""

    axes: tuple[tuple[str, int], ...]
    floats: tuple[int, ...]


def lg_read(table: "ControlTable", kinds: tuple[str, ...], f: int) -> LGRead:
    """Factor ``f``'s :class:`LGRead` under the node kinds ``kinds`` (empty: all discrete)."""
    axes, floats = [], []
    for what, i in declared_parents(table, f):
        if what == "control" and kinds and kinds[i] == FLOAT:
            floats.append(i)
        else:
            axes.append((what, i))
    return LGRead(axes=tuple(axes), floats=tuple(floats))


def node_entry(action: object, k: int, layout: "ControlLayout | None") -> object:
    """Control node ``k``'s entry in a positional step: a :class:`ControlStep`'s entry, its
    block of a flat continuous vector, or its column of a level array."""
    if isinstance(action, ControlStep):
        return action.entries[k]
    if layout is not None:
        return layout.node_block(k, action)
    return jnp.asarray(action)[..., k]


def has_control_axes(control_deps: "ControlDeps | ControlTable | None", f: int, n: int) -> bool:
    """Whether factor ``f``'s table, with ``n`` parent axes, carries control axes to slice: a
    :class:`ControlTable` factor with no control parent may omit the size-1 axis."""
    if isinstance(control_deps, ControlTable) and not control_deps.control_parents[f]:
        return n == control_deps.arity[f] + 1
    return True


def _read_axis(table: Array, index: Array, axis: int, log: bool) -> Array:
    """``table`` with one control axis read: indexed at a level, or contracted against a
    probability vector (the mixture ``Σ_u q(u) table[..., u, ...]``, in log space with
    ``log``, where a zero weight or a zero-probability entry contributes nothing and a mixture
    of zeros is ``-inf`` with a finite gradient)."""
    if not is_belief(index):
        return jnp.take(table, index, axis=axis)
    moved = jnp.moveaxis(table, axis, -1)
    if log:
        live = index > 0
        weights = jnp.where(live, jnp.log(jnp.where(live, index, 1.0)), -jnp.inf)
        z = moved + weights
        dead = jnp.all(jnp.isneginf(z), axis=-1)
        out = jax.scipy.special.logsumexp(jnp.where(dead[..., None], 0.0, z), axis=-1)
        return jnp.where(dead, -jnp.inf, out)
    return jnp.sum(moved * index, axis=-1)


def take_controls(
    table: Array, index: "Array | tuple[Array, ...] | ControlRead", log: bool = False
) -> Array:
    """Slice ``table``'s control axes at ``index``: one level per trailing axis, in order, or a
    :class:`ControlRead` naming each level's parent position.

    A level may be a probability vector over the axis (a belief-valued policy's entry): the
    axis is then contracted against it, as a mixture of the slices. ``log`` says ``table``
    holds log-probabilities, so the mixture is taken in probability space.
    """
    if isinstance(index, ControlRead):
        order = sorted(zip(index.positions, range(len(index.index)), strict=True), reverse=True)
        for removed, (pos, i) in enumerate(order):
            table = _read_axis(table, index.index[i], pos - index.arity + removed, log)
        return table
    if not isinstance(index, tuple):
        index = (index,)
    for i in reversed(index):
        table = _read_axis(table, i, -1, log)
    return table


def parent_levels(
    state_levels: tuple,
    action: Array,
    f: int,
    control_deps: "ControlDeps | ControlTable | None",
    n: int,
) -> tuple:
    """Factor ``f``'s full parent index in table axis order: its state parents' values
    ``state_levels`` (``B_deps`` order) with the control levels at their positions, for a
    table with ``n`` parent axes."""
    if not isinstance(control_deps, ControlTable):
        return (*state_levels, *control_index(action, f, control_deps))
    read = control_index(action, f, control_deps)
    parents = control_deps.control_parents[f]
    if not parents:
        return (
            tuple(state_levels)
            if not has_control_axes(control_deps, f, n)
            else (
                *state_levels,
                *read,
            )
        )
    out = list(state_levels)
    for (_, pos), level in sorted(zip(parents, read, strict=True), key=lambda pr: pr[0][1]):
        out.insert(pos, level)
    return tuple(out)


def validate_control_deps(
    control_deps: ControlDeps | None,
    controlled: tuple[bool, ...],
    factor_names: tuple[str, ...],
) -> None:
    """Check ``System.B_control_deps`` against the control declaration (no tables needed).

    One entry per factor; each names distinct, controlled factors (``factor_names``, one per
    factor, for the sentences).
    """
    if control_deps is None:
        return
    facs = factor_names
    if not isinstance(control_deps, tuple) or len(control_deps) != len(controlled):
        raise ValueError(
            f"B_control_deps must be a tuple with one entry per factor ({len(controlled)}), "
            f"got {control_deps!r}"
        )
    for f, deps in enumerate(control_deps):
        if not isinstance(deps, tuple) or not all(
            isinstance(c, int) and not isinstance(c, bool) for c in deps
        ):
            raise ValueError(
                f"B_control_deps[{f}] ({facs[f]!r}) must be a tuple of factor indices, got {deps!r}"
            )
        bad = [c for c in deps if not 0 <= c < len(controlled)]
        if bad:
            raise ValueError(
                f"B_control_deps for factor {facs[f]!r} names {bad}, which are not factor "
                f"indices; the factors are {list(facs)}"
            )
        if len(set(deps)) != len(deps):
            raise ValueError(
                f"B_control_deps for factor {facs[f]!r} names a control twice: "
                f"{[facs[c] for c in deps]}"
            )
        uncontrolled = [facs[c] for c in deps if not controlled[c]]
        if uncontrolled:
            raise ValueError(
                f"B_control_deps for factor {facs[f]!r} names {uncontrolled}, which are not "
                "controlled factors; an entry lists the controlled factors whose control slices "
                "this factor's transition (declare controlled=True for them, or read the factor "
                "as a state parent through B_deps)"
            )


# ------------------------------------------------------------------------- control nodes


class SugarView(NamedTuple):
    """The pymdp-style declaration a System's control nodes amount to, when they amount to one.

    A System is *sugar-shaped* when every control node is named ``u_<f>`` after a distinct state
    factor ``f`` (its *owner*), the nodes follow their owners' order, and every transition lists
    its control parents after its state parents. Then ``controlled[f]`` says whether ``f`` owns a
    node; ``owners[k]`` is node ``k``'s owner; ``B_control_deps`` lists, per state factor, the
    owners of its control parents in axis order, or is ``None`` when each controlled factor reads
    exactly its own node and nothing else reads a node. ``B_control_deps`` is pymdp's
    ``B_action_dependencies``.
    """

    controlled: tuple[bool, ...]
    owners: tuple[int, ...]
    B_control_deps: ControlDeps | None


class ControlTable(NamedTuple):
    """A System's control nodes as every consumer reads them. All static; see
    :func:`control_table`.

    State nodes precede control nodes, so a state node's index is also its *engine position*
    (the index into belief tuples); control node ``controls[k]`` is *column* ``k`` of a policy or
    a control step. ``B_deps``, ``D_deps`` and ``H_deps`` are the state factors' entries with the
    control parents left out (``D_deps`` / ``H_deps`` ``None`` when undeclared);
    ``control_parents[f]`` lists factor ``f``'s control parents as ``(column, position)``, the
    position being the parent's place in the declared ``B_deps`` entry (``arity[f]`` long), and
    ``readers[k]`` the ``(factor, position)`` pairs reading column ``k``. ``default_wiring``
    holds when each factor has at most one control parent, as its last parent, and each node has
    exactly one reader (on a sugar-shaped System: when its ``B_control_deps`` is ``None``).
    ``sugar`` is the :class:`SugarView`, or ``None`` when the System is not sugar-shaped. Which
    nodes are continuous is a property of the params (:func:`control_space`).
    """

    states: tuple[int, ...]
    controls: tuple[int, ...]
    state_names: tuple[str, ...]
    control_names: tuple[str, ...]
    B_deps: tuple[tuple[int, ...], ...]
    control_parents: tuple[tuple[tuple[int, int], ...], ...]
    arity: tuple[int, ...]
    readers: tuple[tuple[tuple[int, int], ...], ...]
    D_deps: tuple[tuple[int, ...], ...] | None
    H_deps: tuple[tuple[int, ...], ...] | None
    default_wiring: bool
    sugar: SugarView | None

    @property
    def num_states(self) -> int:
        return len(self.states)

    @property
    def num_controls(self) -> int:
        """The number of control nodes (policy columns)."""
        return len(self.controls)

    def column(self, node: int) -> int:
        """The policy/step column of control node ``node``."""
        return node - len(self.states)


def _check_index_tuple(value: object, what: str) -> None:
    if not isinstance(value, tuple) or not all(
        isinstance(i, int) and not isinstance(i, bool) for i in value
    ):
        raise ValueError(f"{what} must be a tuple of node indices, got {value!r}")


@functools.cache
def control_table(
    B_deps: tuple[tuple[int, ...], ...],
    A_deps: tuple[tuple[int, ...], ...],
    factor_names: tuple[str, ...],
    controls: tuple[int, ...],
    D_deps: tuple[tuple[int, ...], ...] | None,
    H_deps: tuple[tuple[int, ...], ...] | None,
) -> ControlTable:
    """Validate a System's control declaration and derive its :class:`ControlTable`.

    Takes the System's stored node-form fields (``factor_names`` one per node). Raises, with a
    sentence, on a control node that is not after every state node, that has parents of its own,
    that is an ``A_deps`` / ``D_deps`` / ``H_deps`` parent, or that no factor reads.
    """
    names = factor_names
    n = len(B_deps)
    _check_index_tuple(controls, "controls")
    if len(set(controls)) != len(controls) or tuple(sorted(controls)) != controls:
        raise ValueError(
            f"controls must list distinct node indices in ascending (node) order, got {controls!r}"
        )
    bad = [c for c in controls if not 0 <= c < n]
    if bad:
        raise ValueError(f"controls names {bad}, which are not node indices (0..{n - 1})")
    s = n - len(controls)
    if controls != tuple(range(s, n)):
        raise ValueError(
            f"control node(s) {[names[c] for c in controls]} must follow every state node: "
            f"declare the state factors first, then the control nodes (controls="
            f"{tuple(range(s, n))} for these {n} nodes)"
        )
    is_control = set(controls)
    for c in controls:
        if B_deps[c]:
            raise NotImplementedError(
                f"control node {names[c]!r} lists transition parents "
                f"{[names[g] for g in B_deps[c]]}: a control node's own dynamics (a prior over "
                "the next control) arrive with the path-dynamics build (docs/design/"
                "control-states.md C2); its B_deps entry must be ()"
            )
    for what, deps_all in (("D_deps", D_deps), ("H_deps", H_deps)):
        if deps_all is None:
            continue
        if len(deps_all) != n:
            if controls:
                raise ValueError(
                    f"{what} has {len(deps_all)} entries for {n} nodes ({s} state factor(s) "
                    f"and {len(controls)} control node(s)): declare one entry per node, () for "
                    "each control node"
                )
            continue  # without control nodes the face validators name the count mismatch
        for c in controls:
            if deps_all[c]:
                why = (
                    "a prior over a control node arrives with the path-dynamics build "
                    "(docs/design/control-states.md C2)"
                    if what == "D_deps"
                    else "a control node's terminal target is unconditional (use a "
                    "plain terminal_targets entry)"
                )
                raise NotImplementedError(
                    f"{what} gives control node {names[c]!r} parents "
                    f"{[names[g] for g in deps_all[c]]}: {why}; its {what} entry must be ()"
                )
        for f in range(s):
            read = [names[g] for g in deps_all[f] if g in is_control]
            if read:
                raise NotImplementedError(
                    f"{what} for factor {names[f]!r} names control node(s) {read}: a control "
                    f"node as a {'prior' if what == 'D_deps' else 'terminal-target'} parent is "
                    "not supported"
                )
    for m, deps in enumerate(A_deps):
        read = [names[g] for g in deps if g in is_control]
        if read:
            raise NotImplementedError(
                f"A_deps for modality {m} names control node(s) {read}, which a modality does "
                "not read; name the action node instead (action node k is A_deps index "
                "len(B_deps) + k)"
            )
    state_deps, control_parents, readers = [], [], [[] for _ in controls]
    for f in range(s):
        deps = B_deps[f]
        state_deps.append(tuple(g for g in deps if g not in is_control))
        parents = tuple((g - s, pos) for pos, g in enumerate(deps) if g in is_control)
        if len({k for k, _ in parents}) != len(parents):
            raise ValueError(f"B_deps for factor {names[f]!r} names a control node twice")
        control_parents.append(parents)
        for k, pos in parents:
            readers[k].append((f, pos))
    unread = [names[c] for k, c in enumerate(controls) if not readers[k]]
    if unread:
        raise ValueError(
            f"control node(s) {unread} are read by no factor: a control node must be read by at "
            "least one factor until it has a prior of its own"
        )
    arity = tuple(len(B_deps[f]) for f in range(s))
    default = all(
        len(readers[k]) == 1 and readers[k][0][1] == arity[readers[k][0][0]] - 1
        for k in range(len(controls))
    ) and all(len(p) <= 1 for p in control_parents)
    states = tuple(range(s))
    state_names = names[:s]
    sugar = _sugar_view(state_names, names[s:], control_parents, arity)
    if sugar is not None:
        default = sugar.B_control_deps is None
    return ControlTable(
        states=states,
        controls=controls,
        state_names=state_names,
        control_names=names[s:],
        B_deps=tuple(state_deps),
        control_parents=tuple(control_parents),
        arity=arity,
        readers=tuple(tuple(r) for r in readers),
        D_deps=None if D_deps is None else _state_entries(D_deps, n, s),
        H_deps=None if H_deps is None else _state_entries(H_deps, n, s),
        default_wiring=default,
        sugar=sugar,
    )


def _state_entries(deps_all: tuple, n: int, s: int) -> tuple:
    """The state factors' entries of a per-node declaration (passed whole when its count is
    off, for the validators to name)."""
    return tuple(deps_all[:s]) if len(deps_all) == n else tuple(deps_all)


def _sugar_view(state_names, control_names, control_parents, arity):
    """The :class:`SugarView` of a control declaration, or ``None`` if it is not sugar-shaped."""
    by_name = {control_name(f): i for i, f in enumerate(state_names)}
    owners = tuple(by_name.get(name, -1) for name in control_names)
    if any(o < 0 for o in owners) or list(owners) != sorted(set(owners)):
        return None
    for f, parents in enumerate(control_parents):
        if sorted(pos for _, pos in parents) != list(range(arity[f] - len(parents), arity[f])):
            return None
        if [pos for _, pos in parents] != sorted(pos for _, pos in parents):
            return None
    controlled = tuple(f in owners for f in range(len(state_names)))
    deps = tuple(tuple(owners[k] for k, _ in parents) for parents in control_parents)
    default = all(deps[f] == ((f,) if c else ()) for f, c in enumerate(controlled))
    return SugarView(
        controlled=controlled,
        owners=owners,
        B_control_deps=None if default else deps,
    )


#: The namespace separator of composite names (``merge()``'s ``"<component>.<name>"``).
NAMESPACE_SEPARATOR = "."


def control_name(factor: str) -> str:
    """The name of the control node the sugar creates for ``factor``: ``"u_<factor>"``,
    namespaced like any node (``"world.u_location"`` for ``"world.location"``)."""
    space, sep, leaf = factor.rpartition(NAMESPACE_SEPARATOR)
    return f"{space}{sep}u_{leaf}"


def action_name(factor: str) -> str:
    """The name of the action node the sugar creates for ``factor``: ``"a_<factor>"``,
    namespaced like any node (``"world.a_location"`` for ``"world.location"``)."""
    space, sep, leaf = factor.rpartition(NAMESPACE_SEPARATOR)
    return f"{space}{sep}a_{leaf}"


def check_kernel(table: object, what: str, kind: str, size: int | None) -> Array:
    """An action node's kernel ``P(a | u)`` checked: a ``(size, size)`` table over a discrete
    node's levels whose columns are finite, nonnegative and sum to one. ``what`` names the
    entry in the sentence; ``size`` ``None`` (the node's levels unknown) checks squareness only.
    Concrete values raise here, traced ones through ``equinox.error_if``."""
    if kind != LEVEL:
        raise NotImplementedError(
            f"{what} is a kernel on the action of a continuous control node, whose values have no "
            "levels to tabulate P(a | u) over; drop the entry"
        )
    table = jnp.asarray(table, jnp.float32)
    n = size if size is not None else (table.shape[0] if table.ndim == 2 else -1)
    if table.shape != (n, n):
        raise ValueError(
            f"{what} has shape {tuple(table.shape)}; it is P(a | u) over the node's {n} level(s), "
            f"a ({n}, {n}) table with one column per decision"
        )
    sentence = (
        f"{what} is not a probability table: pass finite, nonnegative columns that each sum to one"
    )
    bad = (
        ~jnp.all(jnp.isfinite(table))
        | jnp.any(jnp.abs(jnp.sum(table, axis=0) - 1.0) > 1e-5)
        | jnp.any(table < 0)
    )
    if not isinstance(bad, jax.core.Tracer):
        if bool(bad):
            raise ValueError(sentence)
        return table
    return eqx.error_if(table, bad, sentence)


class SensedNodes(NamedTuple):
    """The action nodes some modality observes, as a planning closure reads them at a step.

    ``columns`` are the nodes' control columns, in the order the engine appends their beliefs
    after the state factors (engine position ``offset + i`` is ``columns[i]``); ``names`` and
    ``levels`` their action-node names and level counts; ``kernels`` the declared actuator
    kernels ``P(a | u)`` by control column (a node absent from it has the delta kernel);
    ``layout`` the flat layout of an all-continuous space (``None`` otherwise)."""

    offset: int
    columns: tuple[int, ...]
    names: tuple[str, ...]
    levels: tuple[int, ...]
    kernels: Mapping[int, Array]
    layout: "ControlLayout | None" = None


def action_node_beliefs(action: object, sensed: SensedNodes) -> tuple[Array, ...]:
    """``P(a | u_t)`` of each observed action node at one plan step, as probability vectors, in
    ``sensed.columns`` order. A clamped level ``u`` gives the one-hot of ``u`` under the delta
    kernel and the kernel's column ``u`` under a declared one; a probability vector over the
    node's levels (a soft step) gives the mixture of the kernel's columns it weights, the
    vector itself under the delta kernel. ``action`` is the step, one level per control node, a
    flat vector or a :class:`ControlStep`."""
    out = []
    for k, n in zip(sensed.columns, sensed.levels, strict=True):
        entry = node_entry(action, k, sensed.layout)
        kernel = sensed.kernels.get(k)
        if is_belief(entry):
            out.append(entry if kernel is None else jnp.einsum("au,...u->...a", kernel, entry))
            continue
        u = jnp.asarray(entry, jnp.int32)
        if kernel is None:
            out.append(jax.nn.one_hot(u, n, dtype=jnp.float32))
        else:
            out.append(jnp.moveaxis(jnp.take(kernel, u, axis=-1), 0, -1))
    return tuple(out)


def _admits(decl: object) -> frozenset[str]:
    """The kinds a transition declaration can read a control parent as: a table a level only, a
    linear-Gaussian transition a level (a mode index) and, with ``G``, a float vector, a neural
    one either (a level as its one-hot)."""
    from hybrid_aif.distributions import ConditionalCategorical, ConditionalDirichlet
    from hybrid_aif.factors import LinearGaussianTransition

    if isinstance(decl, ConditionalCategorical | ConditionalDirichlet):
        return frozenset({LEVEL})
    if isinstance(decl, LinearGaussianTransition):
        return frozenset({LEVEL, FLOAT}) if decl.G is not None else frozenset({LEVEL})
    return frozenset({LEVEL, FLOAT})


def _reader_name(decl: object) -> str:
    return type(decl).__name__


def _float_width(table: ControlTable, decl: object, f: int, k: int, kinds) -> int | None:
    """The width factor ``f``'s declaration gives continuous node ``k``, or ``None`` when it
    gives none of its own (a linear-Gaussian ``G`` reads its continuous parents together)."""
    parents = table.control_parents[f]
    j = [c for c, _ in parents].index(k)
    if hasattr(decl, "parent_dims") and getattr(decl, "parent_dims", None):
        dims = tuple(decl.parent_dims)  # the factor's previous value, then its control parents
        at = len(dims) - len(parents) + j
        return int(dims[at]) if 0 <= at < len(dims) else None
    inputs = getattr(getattr(decl, "map", None), "input_dims", None)
    if inputs:
        at = len(inputs) - len(parents) + j
        return int(inputs[at]) if 0 <= at < len(inputs) else None
    return None


def _level_axis(table: ControlTable, decl: object, f: int, pos: int, kinds) -> int | None:
    """The size of factor ``f``'s parent axis for the discrete control parent at ``pos``, or
    ``None`` when its family carries no such axis."""
    arity = table.arity[f]
    shape = getattr(decl, "parent_shape", None)
    if shape is None and hasattr(decl, "innovations"):
        shape = decl.innovations.parent_shape
        #  a linear-Gaussian transition's mode axes skip its continuous control parents
        pos -= sum(1 for c, p in table.control_parents[f] if p < pos and kinds[c] == FLOAT)
    if shape is not None:
        return int(shape[pos]) if pos < len(shape) else None
    dims = getattr(decl, "parent_dims", None)
    if dims:
        #  a neural reader: the factor's previous value, then its control parents in order
        parents = table.control_parents[f]
        j = [p for _, p in parents].index(pos)
        at = len(dims) - len(parents) + j
        return int(dims[at]) if 0 <= at < len(dims) else None
    del arity
    return None


def target_on_box(target, box: Box, what: str):
    """``target`` checked to live on ``box`` when it is a truncated Gaussian: its bounds must
    equal the box's. Concrete bounds raise here; traced ones when the computation runs
    (``equinox.error_if``). Any other target passes through."""
    from hybrid_aif.distributions import TruncatedGaussian

    if not isinstance(target, TruncatedGaussian):
        return target
    sentence = (
        f"{what} is a TruncatedGaussian on another box than the node's: pass one whose low and "
        "high equal the node's Box, or a Gaussian"
    )
    same = jnp.all(target.low == box.low) & jnp.all(target.high == box.high)
    if not isinstance(same, jax.core.Tracer):
        if not bool(same):
            raise ValueError(sentence)
        return target
    return dataclasses.replace(target, low=eqx.error_if(target.low, ~same, sentence))


def _fenced_box(name: str, box: Box) -> Box:
    """``box`` with its bounds checked: finite, and ``low < high`` in every dimension. Concrete
    bounds raise here; traced ones raise when the computation runs (``equinox.error_if``)."""
    what = f"params.control[{name!r}]"
    if _check_bounds(what, box.low, box.high):
        return box
    infinite, inverted = _bound_sentences(what)
    low = eqx.error_if(box.low, ~(jnp.isfinite(box.low) & jnp.isfinite(box.high)), infinite)
    low = eqx.error_if(low, ~(low < box.high), inverted)
    return dataclasses.replace(box, low=low)


def control_space(
    table: ControlTable,
    transition: tuple,
    control: Mapping[str, object] | None,
    factor_names: tuple[str, ...],
) -> ControlSpace:
    """Each control node's kind and size, read off the params; the flat layout when every node
    is continuous.

    ``transition`` is the positional per-state-factor declaration tuple and ``control`` the
    ``SystemParams.control`` mapping. A node with an entry in ``control`` is continuous: the
    entry is its :class:`Box`, and every reader must read it as a float vector of the box's
    width. A node
    without one is discrete, its level count read off its readers' parent axes, which must agree.
    Each sentence names what to pass instead.
    """
    names = table.control_names
    control = dict(control or {})
    unknown = [k for k in control if k not in names]
    if unknown:
        raise ValueError(
            f"params.control has entries {unknown}, which are not control nodes; key a box by "
            f"one of the control nodes {list(names)}"
        )
    kinds: list[str] = []
    boxes: list[object] = []
    for k, name in enumerate(names):
        box = control.get(name)
        readers = table.readers[k]
        admits = {factor_names[f]: _admits(transition[f]) for f, _ in readers}
        if box is None:
            floats_only = [r for r, a in admits.items() if LEVEL not in a]
            if floats_only:
                raise ValueError(
                    f"control node {name!r} is read as a float vector by {floats_only}: pass "
                    f"its box as params.control[{name!r}] = Box(low, high)"
                )
            kinds.append(LEVEL)
            boxes.append(None)
            continue
        if not isinstance(box, Box):
            raise ValueError(
                f"params.control[{name!r}] is the box of control node {name!r}: pass "
                f"Box(low, high), got a {type(box).__name__}"
            )
        box = _fenced_box(name, box)
        levels_only = [r for r, a in admits.items() if FLOAT not in a]
        if levels_only:
            readers_by = {factor_names[f]: _reader_name(transition[f]) for f, _ in readers}
            raise ValueError(
                f"control node {name!r} has a box in params.control, so it is continuous, and "
                f"{ {r: readers_by[r] for r in levels_only} } read it as a level: the readers of "
                "one node must agree on its kind; give a linear-Gaussian reader a G term, read "
                "a level-indexed node through a table, or drop the box to make it discrete"
            )
        kinds.append(FLOAT)
        boxes.append(box)
    sizes: list[int] = []
    for k, name in enumerate(names):
        readers = table.readers[k]
        if kinds[k] == FLOAT:
            dim = boxes[k].dim
            widths = {
                factor_names[f]: w
                for f, _ in readers
                if (w := _float_width(table, transition[f], f, k, kinds)) is not None
            }
            wrong = {r: w for r, w in widths.items() if w != dim}
            if wrong:
                raise ValueError(
                    f"control node {name!r} is {dim}-dimensional (its box in params.control), "
                    f"and reader(s) {wrong} read it at another width: the readers of one node "
                    "must agree on its dimension; declare each reader's input at the box's width"
                )
            sizes.append(dim)
            continue
        found = {
            factor_names[f]: size
            for f, pos in readers
            if (size := _level_axis(table, transition[f], f, pos, kinds)) is not None
        }
        if not found:
            raise ValueError(
                f"control node {name!r}: none of its readers "
                f"{[factor_names[f] for f, _ in readers]} carries an axis for it, so its level "
                "count is unknown"
            )
        if len(set(found.values())) > 1:
            raise ValueError(
                f"control node {name!r} is read with different level counts {found}: every "
                "factor that reads a control node must give it the same number of levels"
            )
        sizes.append(next(iter(found.values())))
    layout = None
    if kinds and all(k == FLOAT for k in kinds):
        spans, at = [], 0
        for size in sizes:
            spans.append((at, at + size))
            at += size
        layout = ControlLayout(
            node_spans=tuple(spans),
            factor_nodes=tuple(tuple(c for c, _ in parents) for parents in table.control_parents),
            action_dim=at,
            low=jnp.concatenate([jnp.asarray(b.low, jnp.float32) for b in boxes]),
            high=jnp.concatenate([jnp.asarray(b.high, jnp.float32) for b in boxes]),
        )
    return ControlSpace(kinds=tuple(kinds), sizes=tuple(sizes), boxes=tuple(boxes), layout=layout)
