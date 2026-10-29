"""Level-wise expansion of the policy tree, and the two folds the planners take over it.

:func:`expand_level` applies every first-step action to every node of one level.
:func:`evaluate_policy_tree` and :func:`evaluate_policy_tree_total` fold the tree
by path sums into the per-policy term tree and G of the enumerated grid;
:func:`evaluate_branching_tree` branches each child over its outcomes and folds by
a softmax expectation per level (sophisticated planning). The children of node
``i`` sit at ``i·b + a``, so leaf order is the order of
:func:`~hybrid_aif.planning.enumerate.enumerate_policies` and policy ``p``'s
depth-``t`` ancestor is ``p // b^(H − 1 − t)``.

Design: docs/design/system.md (policy tree)
"""

import math

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import Array

from hybrid_aif.interval import Interval
from hybrid_aif.objectives.base import TermTree, total
from hybrid_aif.planning.base import (
    Beliefs,
    BranchFn,
    StepTermsFn,
    TerminalTermsFn,
    TransitionFn,
    UseBound,
)
from hybrid_aif.planning.enumerate import _merge_trees, _step_warrant, enumerate_policies


def expand_level(
    qs_nodes: Beliefs,
    actions: Array,
    transition_fn: TransitionFn,
    step_terms: StepTermsFn,
    t: int,
) -> tuple[Beliefs, TermTree]:
    """Push every node through every action: one level of the policy tree.

    ``qs_nodes`` is a belief pytree with a leading node axis ``n``; ``actions`` is
    int32 ``(b, K)``. Returns the children's beliefs, leading axis
    ``n·b`` (child of ``i`` under ``a`` at ``i·b + a``), and each child's step term
    tree with the same leading axis. ``t`` is the static step index, passed to
    ``step_terms`` as an int32 scalar.
    """
    n = jtu.tree_leaves(qs_nodes)[0].shape[0]
    b = actions.shape[0]
    step = jnp.int32(t)

    def one(qs: Beliefs, action: Array) -> tuple[Beliefs, TermTree]:
        qs_next = transition_fn(qs, action)
        return qs_next, step_terms(qs_next, action, step)

    over_actions = jax.vmap(one, in_axes=(None, 0))
    children, trees = jax.vmap(over_actions, in_axes=(0, None))(qs_nodes, actions)

    def flatten(x: Array) -> Array:
        return x.reshape((n * b,) + x.shape[2:])

    return jtu.tree_map(flatten, children), jtu.tree_map(flatten, trees)


def _first_step_actions(
    num_controls: tuple[int, ...], controlled: tuple[bool, ...]
) -> tuple[Array, Array]:
    """The grid's first-step control set ``(b, len(num_controls))``."""
    return enumerate_policies(num_controls, 1, controlled)[:, 0, :]


def _repeat_nodes(tree, branching: int):
    """Carry every node's leaves down to its ``branching`` children (child ``i·b + a`` of ``i``)."""
    return jtu.tree_map(lambda x: jnp.repeat(x, branching, axis=0), tree)


def evaluate_policy_tree(
    num_controls: tuple[int, ...],
    controlled: tuple[bool, ...],
    policy_len: int,
    init_qs: Beliefs,
    transition_fn: TransitionFn,
    step_terms: StepTermsFn,
    terminal_terms: TerminalTermsFn | None = None,
) -> tuple[TermTree, Array]:
    """Score every policy of the grid by summing the step term trees along its path.

    ``terminal_terms`` at the horizon-end beliefs is merged in once. Returns the
    term tree (leading per-policy axis on every leaf) and the aligned policy set,
    ``enumerate_policies(num_controls, policy_len, controlled)``.
    """
    policies = enumerate_policies(num_controls, policy_len, controlled)
    actions = _first_step_actions(num_controls, controlled)
    b = actions.shape[0]
    qs = jtu.tree_map(lambda x: x[None], init_qs)
    out: TermTree = {}
    for t in range(policy_len):
        qs, tree = expand_level(qs, actions, transition_fn, step_terms, t)
        out = _merge_trees(_repeat_nodes(out, b), tree)
    if terminal_terms is not None:
        out = _merge_trees(out, jax.vmap(terminal_terms)(qs))
    return out, policies


def evaluate_policy_tree_total(
    num_controls: tuple[int, ...],
    controlled: tuple[bool, ...],
    policy_len: int,
    init_qs: Beliefs,
    transition_fn: TransitionFn,
    step_terms: StepTermsFn,
    terminal_terms: TerminalTermsFn | None = None,
) -> tuple[Interval, Array]:
    """Score every policy of the grid straight to its G, one Interval per node.

    The metrics-free sibling of :func:`evaluate_policy_tree`: each level's step
    tree is folded with :func:`~hybrid_aif.objectives.base.total` and added to the
    parent's G (an empty level passes it through), then the terminal terms. Seeded
    with the step tree's static ``derived`` warrant. Returns the per-policy G and
    the aligned policy set.
    """
    policies = enumerate_policies(num_controls, policy_len, controlled)
    actions = _first_step_actions(num_controls, controlled)
    b = actions.shape[0]
    warrant = _step_warrant(init_qs, transition_fn, step_terms, policies)
    qs = jtu.tree_map(lambda x: x[None], init_qs)
    g = Interval(jnp.zeros((1,)), jnp.zeros((1,)), warrant)
    for t in range(policy_len):
        qs, tree = expand_level(qs, actions, transition_fn, step_terms, t)
        g = _repeat_nodes(g, b)
        if tree:
            g = g + total(tree)
    if terminal_terms is not None:
        g = g + total(jax.vmap(terminal_terms)(qs))
    return g, policies


def _point(tree: TermTree, use_bound: UseBound) -> Array:
    """The chosen point of ``total(tree)``, one value per leading index."""
    g = total(tree)
    return {"midpoint": g.midpoint, "lower": g.lower, "upper": g.upper}[use_bound]


# Bytes the deepest level holds per child: BRANCHING_BELIEF_FACTOR times its belief plus
# BRANCHING_BYTES_PER_NODE. The fixed part is the top of the range measured on 2026-10-23 on
# categorical beliefs (about 100 to 290 bytes; mostly unfused step-term intermediates); the part
# in proportion to the belief covers intermediates that grow with it (a Gaussian belief's
# covariance products; measured 3.9 to 4.9 beliefs per child on Gaussian beliefs of 16 to 64
# dimensions, so 6 leaves a margin of about a fifth). Both err high.
# Design: docs/design/sophisticated-memory.md (SM2).
BRANCHING_BELIEF_FACTOR = 6
BRANCHING_BYTES_PER_NODE = 290


def belief_bytes(init_qs: Beliefs) -> int:
    """Bytes of one belief: its leaves' sizes, from their static shapes and dtypes."""
    return sum(
        int(math.prod(x.shape)) * jnp.dtype(x.dtype).itemsize for x in jtu.tree_leaves(init_qs)
    )


def _per_child(belief_bytes: int) -> int:
    return BRANCHING_BELIEF_FACTOR * belief_bytes + BRANCHING_BYTES_PER_NODE


def branching_tree_bytes(
    n_actions: int,
    n_outcomes: int,
    horizon: int,
    belief_bytes: int,
    split_depth: int | None = None,
    batch_size: int | None = None,
) -> int:
    """Estimated bytes the deepest level of :func:`evaluate_branching_tree` holds.

    Whole-level (``batch_size`` ``None``): every one of the ``(A·m)^(H−1)·A`` deepest children
    at once, each :data:`BRANCHING_BELIEF_FACTOR` beliefs plus :data:`BRANCHING_BYTES_PER_NODE`.
    Chunked at ``split_depth``: ``batch_size`` subtrees at once, each ``(A·m)^(H−1−d)·A``
    children, plus the ``(A·m)^d`` nodes at the split. ``batch_size`` is taken to divide the
    node count, as :func:`chunk_choice` chooses it; a remainder would add its own subtrees.
    """
    per_child = _per_child(belief_bytes)
    width = n_actions * n_outcomes
    if batch_size is None or split_depth is None or not 1 <= split_depth <= horizon - 1:
        return width ** (horizon - 1) * n_actions * per_child
    nodes = width**split_depth
    subtree = width ** (horizon - 1 - split_depth) * n_actions * per_child
    return min(batch_size, nodes) * subtree + nodes * per_child


def _largest_divisor(n: int, at_most: int) -> int:
    """The largest divisor of ``n`` that is at most ``at_most`` (at least 1)."""
    return next(k for k in range(max(1, min(at_most, n)), 0, -1) if n % k == 0)


def chunk_choice(
    n_actions: int,
    n_outcomes: int,
    horizon: int,
    belief_bytes: int,
    budget: int,
    split_depth: int | None = None,
) -> tuple[int, int | None]:
    """``(split_depth, batch_size)`` that keep the estimate (:func:`branching_tree_bytes`)
    within ``budget``; ``batch_size`` is ``None`` when the whole level fits, or when no node lies
    at ``split_depth`` to chunk over (outside ``1 … horizon − 1``).

    A given ``split_depth`` is kept, with the largest batch that fits. Without one, the split
    deepens from 1 until a single subtree fits. The batch is the largest divisor of the node
    count that fits, so ``lax.map`` has no remainder to evaluate beside its batches. When not
    even one subtree fits at any depth, the batch is 1 at the depth whose estimate is least (the
    nodes held at the split grow with depth as the subtrees shrink), and the budget is a best
    effort.
    """
    if branching_tree_bytes(n_actions, n_outcomes, horizon, belief_bytes) <= budget:
        return (1 if split_depth is None else split_depth), None
    if horizon < 2 or (split_depth is not None and not 1 <= split_depth <= horizon - 1):
        return (1 if split_depth is None else split_depth), None
    per_child = _per_child(belief_bytes)
    width = n_actions * n_outcomes
    depths = list(range(1, horizon)) if split_depth is None else [split_depth]
    held = {
        d: (width**d * per_child, width ** (horizon - 1 - d) * n_actions * per_child)
        for d in depths
    }
    for d in depths:
        nodes_bytes, subtree = held[d]
        if budget - nodes_bytes >= subtree:
            return d, _largest_divisor(width**d, (budget - nodes_bytes) // subtree)
    least = min(depths, key=lambda d: sum(held[d]))
    return least, 1


def _soft_value(G: Array, gamma: float, ok: Array | None) -> Array:
    """``Σ_a softmax(−γ G)_a G_a`` along the last axis; an action with ``ok`` false has
    ``G = +∞`` there, so it takes no weight and adds nothing."""
    if ok is None:
        return jnp.sum(jax.nn.softmax(-gamma * G, axis=1) * G, axis=1)
    weights = jax.nn.softmax(jnp.where(ok, -gamma * G, -jnp.inf), axis=1)
    return jnp.sum(weights * jnp.where(ok, G, 0.0), axis=1)


def evaluate_branching_tree(
    actions: Array,
    horizon: int,
    gamma: float,
    init_qs: Beliefs,
    transition_fn: TransitionFn,
    step_terms: StepTermsFn,
    terminal_terms: TerminalTermsFn | None = None,
    branch_fn: BranchFn | None = None,
    use_bound: UseBound = "midpoint",
    legal: Array | None = None,
    split_depth: int = 1,
    batch_size: int | None = None,
) -> Array:
    """The root's per-action G under sophisticated (closed-loop) planning.

    Each child is scored by the ``use_bound`` point of its step tree (zero when
    empty) and, short of the horizon, branched by ``branch_fn`` over its ``m``
    outcomes (``m = 1`` without a branch closure); the last depth adds the
    terminal terms. The fold runs bottom-up::

        G(node, a) = g_step(child) + Σ_o p(child, o) · V(child, o)
        V(node)    = Σ_a softmax(−γ · G(node, ·))_a · G(node, a)

    with ``V = 0`` below the deepest children. Returns ``G`` at the root, ``(b,)``.
    The deepest level holds ``(b·m)^(H−1)·b`` beliefs.

    ``legal`` is a bool ``(horizon, b)``: ``legal[t, a]`` false gives action ``a`` at depth
    ``t ≥ 1`` the value ``+∞`` in that node's softmax, so it carries no weight in ``V``; the
    tree still expands it. Row 0 is not read: the root's ``G`` is returned unmasked. A mask
    with no true entry in a row has no action to weigh at that depth, and gives ``NaN``. With
    ``batch_size`` set, the ``(b·m)^split_depth`` nodes at depth ``split_depth`` (1 ≤
    ``split_depth`` ≤ ``horizon − 1``) have their subtrees evaluated that many at a time with
    :func:`jax.lax.map` instead of all at once; the result is the same up to float rounding, and the
    deepest level held at once shrinks to ``batch_size`` subtrees.
    """
    if horizon < 1:
        raise ValueError(f"evaluate_branching_tree: horizon must be at least 1, got {horizon}")
    if batch_size is not None and batch_size < 1:
        raise ValueError(
            f"evaluate_branching_tree: batch_size must be at least 1, got {batch_size}"
        )
    if batch_size is not None and split_depth < 1:
        raise ValueError(
            "evaluate_branching_tree: split_depth must be at least 1 (the root is one node, "
            f"with nothing to chunk over), got {split_depth}"
        )
    b = actions.shape[0]

    def values(qs: Beliefs, t_start: int) -> tuple[Array, Array]:
        """``(G, V)`` of the nodes ``qs`` at depth ``t_start``: ``G`` ``(n, b)``, ``V`` ``(n,)``."""
        levels: list[tuple[int, Array, Array | None]] = []
        v_tail = None
        for t in range(t_start, horizon):
            n = jtu.tree_leaves(qs)[0].shape[0]
            if batch_size is not None and t == split_depth and t > t_start and batch_size < n:
                v_tail = jax.lax.map(
                    lambda q, t=t: values(jtu.tree_map(lambda x: x[None], q), t)[1][0],
                    qs,
                    batch_size=batch_size,
                )
                break
            children, tree = expand_level(qs, actions, transition_fn, step_terms, t)
            g = _point(tree, use_bound) if tree else jnp.zeros((n * b,))
            p = None
            if t + 1 < horizon:
                if branch_fn is None:
                    p, qs = jnp.ones((n * b, 1)), children
                else:
                    p, after = jax.vmap(branch_fn)(children)
                    qs = jtu.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), after)
            elif terminal_terms is not None:
                g = g + _point(jax.vmap(terminal_terms)(children), use_bound)
            levels.append((t, g, p))

        v = v_tail
        for t, g, p in reversed(levels):
            if p is not None:
                g = g + jnp.sum(p * v.reshape(p.shape), axis=1)
            G = g.reshape(-1, b)
            ok = legal[t][None, :] if legal is not None and t >= 1 else None
            v = _soft_value(G, gamma, ok)
        return G, v

    G, _ = values(jtu.tree_map(lambda x: x[None], init_qs), 0)
    return G[0]
