"""CEM planner: sampled policy search over continuous action sequences.

A diagonal-Gaussian proposal over ``(policy_len, action_dim)`` sequences, refit to
its own elites ``cem_iters`` times: draw ``num_candidates`` sequences clipped to
the declared box, score them with the context's closures
(:func:`~hybrid_aif.planning.enumerate.evaluate_policies_total`), and refit
``mean``/``std`` to the ``elite_frac`` with the lowest G. The result is the final
candidate set with the proposal mean prepended as row 0 and
``q_pi = softmax(−γ·G)`` over it: :func:`mean_action` is the receding-horizon
action, and sampling ``q_pi`` acts from the searched posterior. The loop is one
``lax.scan``; a fixed key gives a bitwise-identical plan. ``min_std`` floors the
refit standard deviation (a proposal floor; it never changes G). E as the
proposal is not built and raises.

Design: docs/design/amortization.md A9
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.distributions import Categorical
from hybrid_aif.interval import Interval
from hybrid_aif.objectives.base import TermTree, total
from hybrid_aif.planning.base import (
    Planner,
    PlannerContext,
    UseBound,
    refuse_mixed,
    register_planner,
)
from hybrid_aif.planning.enumerate import (
    evaluate_policies,
    evaluate_policies_total,
    policy_posterior,
)

#: Defaults, in units of the declared box width per action dimension: the initial
#: proposal covers half the box, and the refit std never falls below a tenth of it.
INIT_STD_FRACTION = 0.5
MIN_STD_FRACTION = 0.1


class CEMResult(NamedTuple):
    """One CEM plan: the posterior, the metrics, and the proposal it ended on.

    ``candidates`` is ``(num_candidates + 1, policy_len, action_dim)`` with row 0
    the final proposal ``mean``; ``G``, ``q_pi`` and every ``tree`` leaf share that
    leading axis.
    """

    q_pi: Categorical
    tree: TermTree | None
    candidates: Array
    G: Interval
    mean: Array
    std: Array


def mean_action(candidates: Array) -> Array:
    """The receding-horizon action: the final proposal mean's first step, ``candidates[0, 0]``."""
    return jnp.asarray(candidates)[0, 0]


def _point(G: Interval, use_bound: UseBound) -> Array:
    return {"midpoint": G.midpoint, "lower": G.lower, "upper": G.upper}[use_bound]


def _elite_count(num_candidates: int, elite_frac: float) -> int:
    if num_candidates < 2:
        raise ValueError(f"num_candidates must be at least 2, got {num_candidates}")
    if not 0.0 < elite_frac <= 1.0:
        raise ValueError(f"elite_frac must lie in (0, 1], got {elite_frac}")
    return max(1, min(num_candidates, int(round(elite_frac * num_candidates))))


def _layout(ctx: PlannerContext):
    refuse_mixed(ctx, "cem")
    if ctx.continuous_controls is None:
        raise ValueError(
            "the CEM planner plans over continuous control sequences, and this System's control "
            "nodes are discrete (level indices): give a node a box in params.control and a "
            'float-reading transition, or plan with the "enumeration" planner (design: '
            "amortization.md A9)."
        )
    return ctx.continuous_controls


def cem_plan(
    ctx: PlannerContext,
    key: Array,
    *,
    num_candidates: int = 64,
    cem_iters: int = 5,
    elite_frac: float = 0.125,
    init_mean: Array | None = None,
    init_std: float | Array | None = None,
    min_std: float | Array | None = None,
    use_bound: UseBound = "midpoint",
    metrics: bool = True,
) -> CEMResult:
    """Run CEM against ``ctx``; see the module docstring for the algorithm.

    ``init_mean`` (default: the box midpoint) is the warm start, e.g. the previous
    plan shifted one step. ``init_std``/``min_std`` default to
    :data:`INIT_STD_FRACTION`/:data:`MIN_STD_FRACTION` of the box width; explicit
    values are in action units, scalars or ``(action_dim,)`` arrays.
    ``metrics=False`` skips the term tree without changing the posterior.
    """
    layout = _layout(ctx)
    if ctx.policy_prior is not None:
        raise NotImplementedError(
            "policy_prior with the CEM planner: E's representation for a sampled planner is the "
            "proposal distribution itself, not a vector over an enumerated policy set (design: "
            "amortization.md A3 — the 'cem' planner has no e_proposal mode). Drop "
            "the policy_prior designation, or plan with the enumeration "
            "planner."
        )
    if cem_iters < 1:
        raise ValueError(f"cem_iters must be at least 1, got {cem_iters}")
    n_elite = _elite_count(num_candidates, elite_frac)
    horizon, action_dim = ctx.policy_len, layout.action_dim
    low, high = layout.bounds
    width = high - low

    mean = 0.5 * (low + high) + jnp.zeros((horizon, action_dim), low.dtype)
    if init_mean is not None:
        mean = jnp.broadcast_to(jnp.asarray(init_mean, low.dtype), (horizon, action_dim))
    std = jnp.broadcast_to(
        jnp.asarray(INIT_STD_FRACTION * width if init_std is None else init_std, low.dtype),
        (horizon, action_dim),
    )
    floor = jnp.broadcast_to(
        jnp.asarray(MIN_STD_FRACTION * width if min_std is None else min_std, low.dtype),
        (horizon, action_dim),
    )

    def score(candidates: Array) -> Interval:
        return evaluate_policies_total(
            candidates,
            ctx.init_beliefs,
            ctx.transition_fn,
            ctx.step_terms_fn,
            ctx.terminal_terms_fn,
        )

    def iteration(carry, _):
        k, mean, std, _, _ = carry
        k, k_draw = jax.random.split(k)
        eps = jax.random.normal(k_draw, (num_candidates, horizon, action_dim), low.dtype)
        candidates = jnp.clip(mean + std * eps, low, high)
        scored = score(candidates)
        g = _point(scored, use_bound).astype(low.dtype)
        elite = candidates[jax.lax.top_k(-g, n_elite)[1]]
        return (
            k,
            elite.mean(axis=0),
            jnp.maximum(elite.std(axis=0), floor),
            candidates,
            scored,
        ), None

    blank = jnp.zeros((num_candidates, horizon, action_dim), low.dtype)
    #  the carry holds the last iteration's scores (an Interval, whose warrant flag is
    #  static structure): its initial value is zeros in that same structure
    scored0 = jax.tree_util.tree_map(
        lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), jax.eval_shape(score, blank)
    )
    init = (key, mean, std, blank, scored0)
    (_, mean, std, candidates, scored), _ = jax.lax.scan(iteration, init, None, length=cem_iters)

    candidates = jnp.concatenate([mean[None], candidates], axis=0)
    if metrics:
        tree = evaluate_policies(
            candidates,
            ctx.init_beliefs,
            ctx.transition_fn,
            ctx.step_terms_fn,
            ctx.terminal_terms_fn,
        )
        G = total(tree)
    else:
        tree = None
        #  the sampled candidates were scored by the last iteration; only the mean is new
        G = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b]), score(mean[None]), scored)
    return CEMResult(
        q_pi=policy_posterior(G, ctx.gamma, None, use_bound=use_bound),
        tree=tree,
        candidates=candidates,
        G=G,
        mean=mean,
        std=std,
    )


def make_cem_planner(**kwargs) -> Planner:
    """A CEM :class:`~hybrid_aif.planning.base.Planner` with ``kwargs`` bound.

    Pass it per call, e.g. ``infer_policies(..., planner=make_cem_planner(
    num_candidates=256))``. It carries the ``needs_key`` mark.
    """

    def planner(
        ctx: PlannerContext, use_bound: UseBound = "midpoint", metrics: bool = True
    ) -> tuple[Categorical, TermTree | None, Array]:
        result = _run(ctx, use_bound=use_bound, metrics=metrics, **kwargs)
        return result.q_pi, result.tree, result.candidates

    planner.needs_key = True  # type: ignore[attr-defined]
    return planner


def _run(ctx: PlannerContext, **kwargs) -> CEMResult:
    if ctx.key is None:
        raise ValueError(
            "the CEM planner is stochastic and found no PRNG key in its PlannerContext: pass "
            "plan_key= to System.infer_policies / act / planner_context (it is not split off "
            "the action-selection key — a plan's randomness is the caller's to control)."
        )
    return cem_plan(ctx, ctx.key, **kwargs)


@register_planner("cem", needs_key=True)
def cem_planner(
    ctx: PlannerContext, use_bound: UseBound = "midpoint", metrics: bool = True
) -> tuple[Categorical, TermTree | None, Array]:
    """The :class:`~hybrid_aif.planning.base.Planner` face of this module.

    Registered as ``"cem"`` with :func:`cem_plan`'s defaults. Returns
    ``(q_pi, tree, candidates)``, the candidates as in :class:`CEMResult`.
    """
    result = _run(ctx, use_bound=use_bound, metrics=metrics)
    return result.q_pi, result.tree, result.candidates
