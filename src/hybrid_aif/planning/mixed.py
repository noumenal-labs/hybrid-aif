"""Mixed planner: enumerate the discrete control nodes, descend the continuous ones.

A control space may hold discrete nodes (levels) beside continuous ones (boxes). The planner
takes a set of discrete policies, ``(P, policy_len, K_discrete)`` over the discrete nodes in node
order (by default the full enumeration over their levels, ordered as pymdp orders its policies:
time-major, the last column at the last step fastest), and for each one descends ``G`` over the
continuous controls with :func:`~hybrid_aif.planning.gradient.descend`, the descent of the
gradient planner (Adam's floor and the best iterate it scored included), projected onto the
continuous nodes' boxes. The descents run together, vmapped over the discrete policies, and by
default all start from the same continuous plan: ``init``, or the box centre. The cost is that of
exhaustive search over the discrete set times the number of descent steps.

With ``starts > 1`` each discrete policy is descended from ``starts`` continuous warm starts:
the first is ``init`` (or the box centre) and the rest form a Latin hypercube over
the whole continuous plan, mapped to the box
(:func:`~hybrid_aif.planning.gradient.spread_starts`; deterministic, no key). The starts are an
optimisation device: a discrete policy keeps the plan of the start whose descent scored lowest,
as :func:`~hybrid_aif.planning.gradient.gradient_plan` does, so its ``G`` is the best its descents
found, ``P`` stays the number of discrete policies and the posterior keeps its meaning. Every
start's outcome is kept for inspection in ``start_scores`` and ``start_plans``. ``init`` may
carry a leading ``(starts, policy_len, action_dim)`` axis to give the starts explicitly.

Every ``G`` the planner reports is the expected free energy of a clamped plan: the discrete
policy and its optimised continuous controls. The policy posterior is the softmax of ``−γ`` times
each discrete policy's optimised ``G`` plus the log policy prior, as for the enumerating planner
(:func:`~hybrid_aif.planning.enumerate.policy_posterior`, after pymdp's
``control.update_posterior_policies``). The prior is the System's ``policy_prior`` over its
discrete enumeration, restricted to the supplied set (each policy's entry at its place in the
enumeration); without a ``policy_prior`` the posterior has no prior term. The selected plan is a
discrete policy together with its continuous controls: the registered planner returns the ``P``
plans as a :class:`~hybrid_aif.controls.ControlStep` policy set, one entry per control node with
leading axes ``(P, policy_len)``.

A space with no continuous node is refused with a sentence naming the enumerating planner, and a
space with no discrete node with one naming the gradient planner.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import Array

from hybrid_aif.controls import FLOAT, LEVEL, ControlStep
from hybrid_aif.distributions import Categorical
from hybrid_aif.interval import Interval
from hybrid_aif.objectives.base import TermTree, total
from hybrid_aif.planning.base import Planner, PlannerContext, UseBound, register_planner
from hybrid_aif.planning.enumerate import (
    _prior_logits,
    enumerate_policies,
    evaluate_policies,
    evaluate_policies_total,
    policy_posterior,
)
from hybrid_aif.planning.gradient import (
    DEFAULT_LR_FRACTION,
    DEFAULT_STEPS,
    Optimizer,
    _point,
    check_descent,
    descend_scored,
    warm_starts,
)


class MixedResult(NamedTuple):
    """One mixed plan, over ``P`` discrete policies, each with optimised continuous controls.
    With several starts, ``plan`` is the best start's for each discrete policy and ``trace``,
    ``gradient_trace`` and ``best_step`` are that start's (``starts`` is 1 for a single start)."""

    #: The posterior over the ``P`` discrete policies.
    q_pi: Categorical
    #: Their term trees at the optimised plans (``None`` under ``metrics=False``).
    tree: TermTree | None
    #: The discrete policies, ``(P, policy_len, K_discrete)``.
    discrete: Array
    #: Their optimised continuous controls, ``(P, policy_len, action_dim)`` (the continuous
    #: nodes' blocks concatenated in node order).
    plan: Array
    #: The expected free energy of each clamped plan, ``(P,)``.
    G: Interval
    #: The scored ``G`` before each step, ``(P, steps)``.
    trace: Array
    #: The root mean square of the gradient before each step, ``(P, steps)``.
    gradient_trace: Array
    #: The iterate returned, ``(P,)``, as in
    #: :class:`~hybrid_aif.planning.gradient.GradientResult`.
    best_step: Array
    #: Every start's final score, ``(P, starts)``.
    start_scores: Array
    #: Every start's plan, ``(P, starts, policy_len, action_dim)``.
    start_plans: Array


def _nodes(ctx: PlannerContext, kind: str) -> list[int]:
    return [k for k, kk in enumerate(ctx.kinds) if kk == kind]


def _refuse_unmixed(ctx: PlannerContext) -> None:
    if len(ctx.sizes) != len(ctx.kinds) or len(ctx.boxes) != len(ctx.kinds):
        raise ValueError(
            "the mixed planner reads each control node's size and box from the planner context, "
            f"and this context holds {len(ctx.sizes)} sizes and {len(ctx.boxes)} boxes for "
            f"{len(ctx.kinds)} control node(s): build the context with System.planner_context"
        )
    if not _nodes(ctx, FLOAT):
        raise ValueError(
            "the mixed planner descends G over continuous control nodes, and this System's "
            'control nodes are all discrete: plan with the "enumeration" planner'
        )
    if not _nodes(ctx, LEVEL):
        raise ValueError(
            "the mixed planner enumerates discrete control nodes, and this System's control "
            'nodes are all continuous: plan with the "gradient" planner'
        )


def control_steps(ctx: PlannerContext, discrete: Array, plan: Array) -> ControlStep:
    """Clamped steps over every control node from the discrete levels ``(..., policy_len,
    K_discrete)`` and the continuous plan ``(..., policy_len, action_dim)``: a
    :class:`~hybrid_aif.controls.ControlStep` with each node's entry, a level array
    ``(..., policy_len)`` or a float block ``(..., policy_len, width)``."""
    entries: list[object] = []
    d_at = c_at = 0
    for k, kind in enumerate(ctx.kinds):
        if kind == LEVEL:
            entries.append(discrete[..., d_at])
            d_at += 1
        else:
            entries.append(plan[..., c_at : c_at + ctx.sizes[k]])
            c_at += ctx.sizes[k]
    return ControlStep(entries=tuple(entries))


def _policy_index(discrete: Array, levels: tuple[int, ...]) -> Array:
    """Each policy's place in the full enumeration over ``levels``: its levels read as a mixed
    radix number, step by step, the last node fastest."""
    horizon = discrete.shape[1]
    radices = list(levels) * horizon
    strides = [1] * len(radices)
    for i in range(len(radices) - 2, -1, -1):
        strides[i] = strides[i + 1] * radices[i + 1]
    flat = discrete.reshape(discrete.shape[0], -1)
    return jnp.sum(flat * jnp.asarray(strides, jnp.int32), axis=-1)


def _check_discrete(ctx: PlannerContext, discrete: Array, levels: tuple[int, ...]) -> Array:
    discrete = jnp.asarray(discrete)
    if not jnp.issubdtype(discrete.dtype, jnp.integer):
        raise ValueError(f"discrete_policies are integer levels, got dtype {discrete.dtype}")
    want = (ctx.policy_len, len(levels))
    if discrete.ndim != 3 or discrete.shape[1:] != want or discrete.shape[0] < 1:
        raise ValueError(
            f"discrete_policies has shape {tuple(discrete.shape)}; the mixed planner takes "
            f"(P, {ctx.policy_len}, {len(levels)}): policies over the horizon and one column per "
            "discrete control node, in node order"
        )
    if not isinstance(discrete, jax.core.Tracer):
        high = jnp.asarray(levels, discrete.dtype)
        if bool(jnp.any((discrete < 0) | (discrete >= high))):
            raise ValueError(f"discrete_policies hold a level outside the nodes' levels {levels}")
    return discrete.astype(jnp.int32)


def mixed_plan(
    ctx: PlannerContext,
    *,
    discrete_policies: Array | None = None,
    steps: int = DEFAULT_STEPS,
    init: Array | None = None,
    starts: int = 1,
    lr: float | Array | None = None,
    optimizer: Optimizer = "adam",
    use_bound: UseBound = "midpoint",
    metrics: bool = True,
) -> MixedResult:
    """Search a mixed control space against ``ctx``; see the module docstring.

    ``discrete_policies`` is the set of discrete policies, int ``(P, policy_len,
    K_discrete)``, one column per discrete control node in node order (default: the full
    enumeration over their levels). ``steps``, ``lr`` and ``optimizer`` set each descent as in
    :func:`~hybrid_aif.planning.gradient.gradient_plan`. ``init`` is the warm start of the
    continuous controls shared by every discrete policy, ``(policy_len, action_dim)`` or
    ``(action_dim,)`` broadcast over the steps (default: the boxes' centre), or with a leading
    ``(starts, policy_len, action_dim)`` axis to give every start. ``starts`` is the number of
    continuous warm starts of each discrete policy, of which the best is kept (see the module
    docstring). ``use_bound`` picks the point of ``G``'s interval that is descended and that
    the posterior is taken over.
    Convergence of each descent is read from the last entries of its slice of ``trace`` and
    ``gradient_trace``.
    """
    _refuse_unmixed(ctx)
    check_descent(steps, lr, optimizer)
    discrete_nodes, float_nodes = _nodes(ctx, LEVEL), _nodes(ctx, FLOAT)
    levels = tuple(ctx.levels[k] for k in discrete_nodes)
    horizon = ctx.policy_len
    default = discrete_policies is None
    if default:
        discrete = enumerate_policies(levels, horizon, (True,) * len(levels))
    else:
        discrete = _check_discrete(ctx, discrete_policies, levels)
    low = jnp.concatenate([ctx.boxes[k].low for k in float_nodes])
    high = jnp.concatenate([ctx.boxes[k].high for k in float_nodes])
    rate = DEFAULT_LR_FRACTION * (high - low) if lr is None else jnp.asarray(lr, low.dtype)
    plan0 = warm_starts(init, starts, horizon, low, high)
    count_starts, num_policies = plan0.shape[0], discrete.shape[0]
    args = (
        ctx.init_beliefs,
        ctx.transition_fn,
        ctx.step_terms_fn,
        ctx.terminal_terms_fn,
    )

    def descend_one(levels_one: Array, start: Array):
        def score(plan: Array) -> Array:
            clamped = jtu.tree_map(lambda x: x[None], control_steps(ctx, levels_one, plan))
            return _point(evaluate_policies_total(clamped, *args), use_bound)[0]

        return descend_scored(score, start, low, high, rate, steps=steps, optimizer=optimizer)

    pairs = jnp.repeat(discrete, count_starts, axis=0)
    starts_of = jnp.tile(plan0, (num_policies, 1, 1))
    plans, best_steps, traces, gradient_traces, scores = jax.vmap(descend_one)(pairs, starts_of)

    def by_policy(x: Array) -> Array:
        return x.reshape(num_policies, count_starts, *x.shape[1:])

    start_plans, start_scores = by_policy(plans), by_policy(scores)
    pick = jnp.argmin(jnp.where(jnp.isnan(start_scores), jnp.inf, start_scores), axis=1)

    def chosen(x: Array) -> Array:
        return jnp.take_along_axis(
            by_policy(x), pick.reshape((-1, 1) + (1,) * (x.ndim - 1)), axis=1
        )[:, 0]

    plan, best_step = chosen(plans), chosen(best_steps)
    trace, gradient_trace = chosen(traces), chosen(gradient_traces)
    clamped = control_steps(ctx, discrete, plan)
    if metrics:
        tree = evaluate_policies(clamped, *args)
        G = total(tree)
    else:
        tree = None
        G = evaluate_policies_total(clamped, *args)

    prior = None
    if ctx.policy_prior is not None:
        prior = _prior_logits(ctx.policy_prior, "mixed")
        count = math.prod(levels) ** horizon
        if prior.shape != (count,):
            raise ValueError(
                f"policy_prior has shape {tuple(prior.shape)}; the mixed planner reads it over "
                f"the {count} policies enumerated over the discrete nodes' levels"
            )
        if not default:
            prior = prior[_policy_index(discrete, levels)]
    q_pi = policy_posterior(G, ctx.gamma, prior, use_bound=use_bound)
    return MixedResult(
        q_pi=q_pi,
        tree=tree,
        discrete=discrete,
        plan=plan,
        G=G,
        trace=trace,
        gradient_trace=gradient_trace,
        best_step=best_step,
        start_scores=start_scores,
        start_plans=start_plans,
    )


def make_mixed_planner(**kwargs) -> Planner:
    """A mixed :class:`~hybrid_aif.planning.base.Planner` with ``kwargs`` bound (those of
    :func:`mixed_plan`), e.g. ``make_mixed_planner(steps=100, discrete_policies=held)``. It takes
    a warm start of the continuous controls (``takes_init``): an ``init`` passed to the call
    replaces a bound one."""

    def planner(
        ctx: PlannerContext,
        use_bound: UseBound = "midpoint",
        metrics: bool = True,
        init: Array | None = None,
    ):
        bound = kwargs if init is None else {**kwargs, "init": init}
        return _face(mixed_plan(ctx, use_bound=use_bound, metrics=metrics, **bound), ctx)

    planner.needs_key = False  # type: ignore[attr-defined]
    planner.takes_init = True  # type: ignore[attr-defined]
    return planner


def _face(result: MixedResult, ctx: PlannerContext):
    return result.q_pi, result.tree, control_steps(ctx, result.discrete, result.plan)


@register_planner("mixed")
def mixed_planner(
    ctx: PlannerContext,
    use_bound: UseBound = "midpoint",
    metrics: bool = True,
    init: Array | None = None,
) -> tuple[Categorical, TermTree | None, ControlStep]:
    """The :class:`~hybrid_aif.planning.base.Planner` face of this module.

    Registered as ``"mixed"`` with :func:`mixed_plan`'s defaults; ``init`` is :func:`mixed_plan`'s
    warm start of the continuous controls (``takes_init``). Returns ``(q_pi, tree, policies)``:
    the posterior over the discrete policies, their term trees, and the ``P`` clamped plans as a
    :class:`~hybrid_aif.controls.ControlStep` with leading axes ``(P, policy_len)``.
    """
    return _face(mixed_plan(ctx, use_bound=use_bound, metrics=metrics, init=init), ctx)


mixed_planner.takes_init = True  # type: ignore[attr-defined]
