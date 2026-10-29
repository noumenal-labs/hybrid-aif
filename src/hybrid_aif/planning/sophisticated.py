"""Sophisticated planning: score an action by what the agent would do after
each thing it might observe (Friston et al. 2021, "Sophisticated Inference").

The value of an action is its step terms plus the expected value of the next
decision over the outcomes of the System's ``branch_on`` modalities, each made
under the root's softmax::

    G(a | q) = step(a | q') + E_{o ~ p(o | q')} [ V(q'_o) ]
    V(q)     = Σ_a σ(-γ G(· | q))_a G(a | q)

with ``q' = transition(q, a)``, to ``policy_len``. Without a branch closure this
is closed-loop enumeration; at ``policy_len`` 1 it equals enumeration. Registered as
branching, so when the System declares ``branch_on`` its context scores a state target, per-step
and terminal, by its cross-entropy, which charges nothing for what a branch learns (design:
per-variable-infogain.md V11 decisions 6 and 8). The
returned policy set is the first step's actions, ``(A, 1, K)``. Cost grows as
``(A · n)^(H-1) · A`` for ``n`` branched outcomes, and so does the memory of evaluating the
deepest level whole. Below ``split_depth`` the planner can instead evaluate the subtrees a
``batch_size`` at a time, with the same result up to float rounding and memory in proportion
to the batch, set by hand or from a memory budget (see :func:`make_sophisticated_planner` and
:func:`branching_tree_memory`). pymdp's ``planning/si.py`` implements a
budgeted, pruned form of this recursion.
The recursion is the one SPM's ``spm_forwards`` implements (``toolbox/DEM/spm_MDP_VB_XX.m``).
SPM prunes actions below a sixteenth of the most probable and branches over a few probable
states; this branches every outcome of ``branch_on``, unpruned.

A per-step ``legal=`` mask is read at every depth: an action illegal at depth ``t`` has
``G = +∞`` in that node's softmax, so it takes no weight in ``V``. The tree still expands it.

Design: docs/design/sophisticated-memory.md
"""

from typing import NamedTuple

import jax

from hybrid_aif.distributions import Categorical
from hybrid_aif.objectives.base import TermTree
from hybrid_aif.planning.base import (
    Planner,
    PlannerContext,
    UseBound,
    refuse_mixed,
    register_planner,
)
from hybrid_aif.planning.enumerate import enumerate_policies
from hybrid_aif.planning.tree import (
    belief_bytes,
    branching_tree_bytes,
    chunk_choice,
    evaluate_branching_tree,
)


class BranchingTreeMemory(NamedTuple):
    """The memory of the sophisticated planner's tree, estimated: ``whole_level_bytes`` the
    deepest level evaluated whole; given a budget, ``split_depth`` and ``batch_size`` the
    :func:`~hybrid_aif.planning.tree.chunk_choice` they would choose (both ``None`` when the
    whole level fits) and ``estimated_bytes`` the estimate under that choice. All but the first
    are ``None`` without a budget."""

    whole_level_bytes: int
    split_depth: int | None
    batch_size: int | None
    estimated_bytes: int | None


def _tree_sizes(ctx: PlannerContext) -> tuple[int, int, int, int]:
    """``(A, m, H, belief bytes)``: first-step actions, branched outcomes (1 without a branch
    closure), horizon and one belief's bytes, from static shapes."""
    n_actions = enumerate_policies(ctx.num_controls, 1, ctx.controlled).shape[0]
    if ctx.branch_fn is None:
        outcomes = 1
    else:
        outcomes = jax.eval_shape(ctx.branch_fn, ctx.init_beliefs)[0].shape[0]
    return n_actions, outcomes, ctx.policy_len, belief_bytes(ctx.init_beliefs)


def chunk_plan(
    ctx: PlannerContext,
    memory_budget: int | None,
    split_depth: int | None,
    batch_size: int | None,
) -> tuple[int, int | None]:
    """The ``(split_depth, batch_size)`` for
    :func:`~hybrid_aif.planning.tree.evaluate_branching_tree`.

    A given ``batch_size`` is used as it is, at ``split_depth`` (1 when none is given). Otherwise,
    with a ``memory_budget`` the deepest level's estimated bytes decide
    (:func:`~hybrid_aif.planning.tree.chunk_choice`): the whole level when it fits, else the
    largest batch that fits at the given ``split_depth`` or, with none given, at the shallowest
    depth where one subtree fits; where none does, batch 1 at the depth with the least estimate,
    and the budget is a best effort. Without a budget or a batch, the whole level. Read from
    static shapes, at trace time.
    """
    if batch_size is not None:
        return (1 if split_depth is None else split_depth), batch_size
    if memory_budget is None or ctx.policy_len < 2:
        return (1 if split_depth is None else split_depth), None
    return chunk_choice(*_tree_sizes(ctx), memory_budget, split_depth)


def branching_tree_memory(
    system, params, beliefs, memory_budget: int | None = None
) -> BranchingTreeMemory:
    """The memory of ``system``'s sophisticated tree at ``beliefs``, estimated, and what
    ``memory_budget`` (bytes) would choose; nothing is evaluated.

    The planner context is built as :meth:`~hybrid_aif.system.System.planner_context` builds it,
    and the numbers are those :func:`make_sophisticated_planner` uses with the same budget, so a
    caller runs this once at setup to see the tree's memory, then pins what it chose with
    ``make_sophisticated_planner(split_depth=..., batch_size=...)``. The estimate errs high.
    """
    ctx = system.planner_context(params, beliefs)
    sizes = _tree_sizes(ctx)
    whole = branching_tree_bytes(*sizes)
    if memory_budget is None:
        return BranchingTreeMemory(whole, None, None, None)
    depth, batch = chunk_choice(*sizes, memory_budget)
    if batch is None:
        return BranchingTreeMemory(whole, None, None, whole)
    return BranchingTreeMemory(whole, depth, batch, branching_tree_bytes(*sizes, depth, batch))


def sophisticated_plan(
    ctx: PlannerContext,
    use_bound: UseBound = "midpoint",
    metrics: bool = True,
    memory_budget: int | None = None,
    split_depth: int | None = None,
    batch_size: int | None = None,
) -> tuple[Categorical, TermTree | None, jax.Array]:
    """The sophisticated planner with its memory options.

    ``batch_size`` sets the number of subtrees evaluated at once below ``split_depth`` (1 when
    only ``batch_size`` is given). With neither, ``memory_budget`` (bytes) chooses them from the
    estimate (:func:`chunk_plan`): the largest batch that fits, deepening the split until one
    subtree fits unless ``split_depth`` is given. With none of the three, the whole deepest
    level is evaluated. The ``q_pi``, ``G`` and plan are the same at every setting up to float
    rounding.

    A per-step ``legal=`` mask that is traced is not checked: one leaving no legal action at a
    depth gives a ``NaN`` ``q_pi``.
    """
    refuse_mixed(ctx, "sophisticated")
    if ctx.continuous_controls is not None:
        raise ValueError(
            "the sophisticated planner enumerates each step's actions, and a continuous "
            'action space has none to enumerate: plan with "cem"'
        )
    if ctx.policy_prior is not None:
        raise NotImplementedError(
            "a policy prior E is over action sequences, and the sophisticated planner has none "
            "past the first step: declare no policy_prior, or plan with enumeration"
        )
    actions = enumerate_policies(ctx.num_controls, 1, ctx.controlled)[:, 0, :]  # (A, K)
    depth, batch = chunk_plan(ctx, memory_budget, split_depth, batch_size)
    legal = ctx.legal_steps
    if legal is not None and legal.shape[0] == 1:
        legal = None  # a first-step mask is the root's alone, applied by the System
    G = evaluate_branching_tree(
        actions,
        ctx.policy_len,
        ctx.gamma,
        ctx.init_beliefs,
        ctx.transition_fn,
        ctx.step_terms_fn,
        ctx.terminal_terms_fn,
        ctx.branch_fn,
        use_bound,
        legal=legal,
        split_depth=depth,
        batch_size=batch,
    )
    q_pi = Categorical(logits=-ctx.gamma * G)
    return q_pi, ({} if metrics else None), actions[:, None, :]


def make_sophisticated_planner(
    memory_budget: int | None = None, split_depth: int | None = None, batch_size: int | None = None
) -> Planner:
    """A sophisticated :class:`~hybrid_aif.planning.base.Planner` with the memory options of
    :func:`sophisticated_plan` bound, e.g. ``make_sophisticated_planner(batch_size=64)`` or
    ``make_sophisticated_planner(memory_budget=2**30)``. With no option it evaluates the whole
    level, as the registered planner does; :func:`branching_tree_memory` shows what a budget
    would choose."""
    if split_depth is not None and split_depth < 1:
        raise ValueError(
            "make_sophisticated_planner: split_depth must be at least 1 (the root is one node, "
            f"with nothing to chunk over), got {split_depth}"
        )
    if batch_size is not None and batch_size < 1:
        raise ValueError(
            f"make_sophisticated_planner: batch_size must be at least 1, got {batch_size}"
        )

    def planner(ctx: PlannerContext, use_bound: UseBound = "midpoint", metrics: bool = True):
        return sophisticated_plan(ctx, use_bound, metrics, memory_budget, split_depth, batch_size)

    planner.needs_key = False  # type: ignore[attr-defined]
    planner.branching = True  # type: ignore[attr-defined]
    planner.legal_depth = True  # type: ignore[attr-defined]
    return planner


@register_planner("sophisticated", branching=True, legal_depth=True)
def sophisticated_planner(
    ctx: PlannerContext, use_bound: UseBound = "midpoint", metrics: bool = True
) -> tuple[Categorical, TermTree | None, jax.Array]:
    """The :class:`~hybrid_aif.planning.base.Planner` face; ``metrics=True`` returns ``{}``.

    Registered as ``"sophisticated"`` with :func:`sophisticated_plan`'s defaults: the whole
    deepest level is evaluated."""
    return sophisticated_plan(ctx, use_bound, metrics)
