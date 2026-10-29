"""The habit planner: act from E, the H=0 end of the search-habit dial.

Registered as ``"habit"``, it acts from E and scores nothing. E may be a policy
module (beliefs → an action distribution, rolled out autoregressively through the
transition) or a vector over the enumerated policy set (``q(π) = E(π)``). Under
``metrics=True`` it returns ``{}`` (requested; the selection produced no term tree),
distinct from ``None`` (not requested). To see what G says of the habit's plan,
score it explicitly::

    _, _, plan = get_planner("habit")(system.planner_context(params, beliefs, key=k))
    habit_G = total(system.score_policies(params, beliefs, plan))

Design: docs/design/amortization.md A3, A12 implementation note §(b′)
"""

from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.controls import soft_clip
from hybrid_aif.distributions import Categorical
from hybrid_aif.objectives.base import TermTree
from hybrid_aif.planning.base import PlannerContext, UseBound, refuse_mixed, register_planner
from hybrid_aif.planning.enumerate import enumerate_policies


@register_planner("habit", needs_key=True)
def habit_planner(
    ctx: PlannerContext, use_bound: UseBound = "midpoint", metrics: bool = True
) -> tuple[Categorical, TermTree | None, Array]:
    """The :class:`~hybrid_aif.planning.base.Planner` face of this module.

    Returns ``(q_pi, tree, policies)``. For a policy module the policy set is the
    one sampled plan and ``q_pi`` a point mass on it; for a vector E the policy set
    is the enumeration and ``q_pi`` is E. The module case draws from
    :attr:`~hybrid_aif.planning.base.PlannerContext.key`; the vector case ignores
    it. ``metrics=True`` returns ``{}``, ``False`` returns ``None``; ``use_bound``
    is unused.
    """
    refuse_mixed(ctx, "habit")
    if ctx.habit_fn is not None:
        return _from_module(ctx, metrics)
    if ctx.policy_prior is None:
        raise ValueError(
            'the "habit" planner acts from E and this System declares none: put a policy '
            "module (beliefs -> an action distribution) or a log-space vector over the "
            "enumerated policy set in SystemParams.policy_prior. With no E there is no habit "
            "— plan with the enumeration or CEM planner, which score G instead (design: "
            "amortization.md A3, the search-habit dial)."
        )
    return _from_vector(ctx, metrics)


def _from_module(ctx: PlannerContext, metrics: bool) -> tuple[Categorical, TermTree | None, Array]:
    """Roll the policy module forward: act, push, act again — the plan it induces."""
    if ctx.key is None:
        raise ValueError(
            'the "habit" planner draws one action per step from the policy module and found no '
            "PRNG key in its PlannerContext: pass plan_key= to System.infer_policies / act / "
            "score_policies (it is not split off the action-selection key — a habit's "
            "randomness is the caller's to control)."
        )
    bounds = None if ctx.continuous_controls is None else ctx.continuous_controls.bounds

    def step(state, _):
        carry, key = state
        key, draw = jax.random.split(key)
        action = ctx.habit_fn(carry, draw)  # raw: the box belongs to the action's consumers
        #  only the transition sees the action brought into the box, and by a
        #  stop-gradiented rescale (`soft_clip`)
        return (ctx.transition_fn(carry, soft_clip(action, bounds)), key), action

    _, actions = jax.lax.scan(step, (ctx.init_beliefs, ctx.key), None, length=ctx.policy_len)
    policies = actions[None]
    #  a point mass: the habit named one plan, so there is nothing to weigh
    return Categorical(logits=jnp.zeros((1,))), ({} if metrics else None), policies


def _from_vector(ctx: PlannerContext, metrics: bool) -> tuple[Categorical, TermTree | None, Array]:
    """``q(π) = E(π)`` over the enumerated policy set: no γ, no G — the H=0 limit."""
    if ctx.continuous_controls is not None:
        raise ValueError(
            'the "habit" planner was given a vector E over an enumerated policy set, but this '
            "System's action space is continuous, where no such set exists (design: "
            "amortization.md A9). Declare a policy module in policy_prior — the amortized "
            "actor, which this planner samples per step — or plan with the CEM planner."
        )
    policies = enumerate_policies(ctx.num_controls, ctx.policy_len, ctx.controlled)
    prior = ctx.policy_prior
    logits = prior.log_probs if isinstance(prior, Categorical) else jnp.asarray(prior)
    if logits.shape != (policies.shape[0],):
        raise ValueError(
            f"policy_prior has shape {logits.shape} but this System's policy set has "
            f"{policies.shape[0]} member(s); a habit prior is one weight per policy"
        )
    return Categorical(logits=logits), ({} if metrics else None), policies


def habit_action(
    key: Array,
    module,
    beliefs,
    features,
    selection: Literal["stochastic", "deterministic"] = "stochastic",
) -> Array:
    """One action from a policy module: the closure a System hands this planner.

    ``features`` maps the belief carry to the module's parent vector. The draw is
    returned raw, never clipped: the environment clips it, a transition rescales
    it with :func:`~hybrid_aif.controls.soft_clip` (as :func:`_from_module` does),
    and a replay column records it raw.
    """
    parents = (features(beliefs),)
    if selection == "deterministic":
        return module.condition_on(parents).mean()
    return module.sample(key, parents)
