"""Factorised planner: one policy posterior per control node, updated by mean-field sweeps.

A System with several discrete control nodes plans over their product: the enumeration planner
scores every sequence of joint actions, ``(Π_k n_k)^H`` of them. This planner keeps, for each
control node ``k``, a posterior ``q(π_k)`` over that node's own sequences, the ``n_k^H`` rows of
``enumerate_policies((n_k,), H)``, and updates the nodes in node order (Gauss–Seidel): node
``k``'s sequences are scored with every other node ``j`` entered as its current per-step
marginal over levels, a ``(P_k, H, n_j)`` probability vector in a
:class:`~hybrid_aif.controls.ControlStep` (each reader of a control node consumes a probability
vector, design: nesting-and-coupling.md N8), and ``q(π_k) ∝ softmax(−γ G + log E_k)``. Sweep 0
scores each node against the others' uniform marginals. The sweep count is static, so the program
is one unrolled loop; its cost is ``sweeps · Σ_k n_k^H`` sequence evaluations (the flat scan of
:func:`~hybrid_aif.planning.enumerate.evaluate_policies`), against the product's ``(Π_k n_k)^H``.

The message is the expected sufficient statistic: the belief push is exact under the mixture of
the other nodes' policies, and the nonlinear leaves (novelty, risk) are evaluated at the expected
statistic rather than averaged over ``q(π_j)``, the variational message of a
conjugate-exponential factor, moment-matched, not an exact expectation. With nodes that share no
factor the planner equals enumeration's action marginals after one sweep; with a shared factor
(a parameter learned within the plan) the gap is small and not zero.

The returned ``q_pi`` is over the joint *first-step* actions, the grid
``enumerate_policies(num_controls, 1)`` (``(A, 1, K)``), the product of the nodes' first-step
marginals; the per-node detail is a :class:`FactorisedPolicies`. The policy prior is ``None`` or
a tuple of per-node log-weights over each node's own sequences; a vector over the joint set is
refused. Discrete nodes only. ``movement`` reports how far the last sweep moved; convergence is
not enforced.

Design: docs/design/factorised-planning.md (F1–F9)
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.controls import ControlStep
from hybrid_aif.distributions import Categorical
from hybrid_aif.objectives.base import TermTree, total
from hybrid_aif.planning.base import (
    Planner,
    PlannerContext,
    UseBound,
    refuse_mixed,
    register_planner,
)
from hybrid_aif.planning.enumerate import (
    _prior_logits,
    enumerate_policies,
    evaluate_policies,
    evaluate_policies_total,
    policy_posterior,
)

#: The default number of sweeps.
DEFAULT_SWEEPS = 3


class FactorisedPolicies(NamedTuple):
    """The factorised planner's policy set: the joint first-step grid ``q_pi`` is over, and the
    per-node posteriors of the last sweep, in node order."""

    #: The joint first-step actions, int32 ``(A, 1, K)``: ``enumerate_policies(num_controls, 1)``.
    first: Array
    #: Each node's own sequences, int32 ``(n_k^H, H)``.
    grids: tuple[Array, ...]
    #: Each node's posterior over its sequences.
    posteriors: tuple[Categorical, ...]
    #: Each node's term tree of its last scoring (leading ``n_k^H`` axis), or ``None`` under
    #: ``metrics=False``.
    trees: tuple[TermTree, ...] | None
    #: Each node's per-step marginal over its levels, ``(H, n_k)``.
    marginals: tuple[Array, ...]
    #: The largest change of any node's marginal in the last sweep, a scalar.
    movement: Array


def _refuse(ctx: PlannerContext) -> None:
    refuse_mixed(ctx, "factorised")
    if ctx.continuous_controls is not None:
        raise ValueError(
            "the factorised planner cannot serve a continuous action space: there is no "
            "finite policy grid to enumerate per control node (design: amortization.md A9). "
            'Plan with the "gradient" or "cem" planner — System(planner="cem"), or '
            "infer_policies(planner=make_cem_planner(...)) with a plan_key."
        )


def _node_priors(ctx: PlannerContext, counts: list[int]) -> list[Array | None]:
    """The policy prior as one log-weight vector per node (``None`` for no prior)."""
    prior = ctx.policy_prior
    if prior is None:
        return [None] * len(counts)
    if not isinstance(prior, tuple | list):
        if not hasattr(prior, "shape") and not isinstance(prior, Categorical) and callable(prior):
            _prior_logits(prior, "factorised")  # the policy-module sentence
        raise ValueError(
            "policy_prior is one vector over the joint policy set, and the factorised planner "
            "keeps one posterior per control node, which no joint prior factorises into: "
            "declare it as a tuple with one entry per control node in node order, each a "
            "log-weight vector (or a Categorical) over that node's own "
            f"{' / '.join(str(c) for c in counts)} sequences, or plan with the "
            '"enumeration" planner'
        )
    if len(prior) != len(counts):
        raise ValueError(
            f"policy_prior has {len(prior)} entries for {len(counts)} control node(s); the "
            "factorised planner takes one per control node, in node order (None for no prior)"
        )
    out: list[Array | None] = []
    for k, (entry, count) in enumerate(zip(prior, counts, strict=True)):
        if entry is None:
            out.append(None)
            continue
        logits = _prior_logits(entry, "factorised")
        if logits.shape != (count,):
            raise ValueError(
                f"policy_prior[{k}] has shape {tuple(logits.shape)}; the factorised planner "
                f"reads it over control node {k}'s own {count} sequences"
            )
        out.append(logits)
    return out


def _step_marginals(q: Categorical, grid: Array, n: int) -> Array:
    """``(H, n)``: the probability of each level at each step under ``q`` over ``grid``."""
    one_hot = jax.nn.one_hot(grid, n, dtype=q.probs.dtype)  # (P, H, n)
    return jnp.einsum("p,pha->ha", q.probs, one_hot)


def _first_logits(q: Categorical, grid: Array, n: int) -> Array:
    """``(n,)``: the log-probability of each level at the first step, by logsumexp."""
    hit = grid[:, 0][:, None] == jnp.arange(n, dtype=grid.dtype)[None, :]  # (P, n)
    return jax.nn.logsumexp(jnp.where(hit, q.log_probs[:, None], -jnp.inf), axis=0)


def factorised_plan(
    ctx: PlannerContext,
    sweeps: int = DEFAULT_SWEEPS,
    use_bound: UseBound = "midpoint",
    metrics: bool = True,
) -> tuple[Categorical, FactorisedPolicies]:
    """Plan ``ctx`` by ``sweeps`` mean-field sweeps over its control nodes; see the module
    docstring. Returns ``(q_pi, policies)``: the posterior over the joint first-step actions and
    the :class:`FactorisedPolicies`."""
    _refuse(ctx)
    sweeps = int(sweeps)
    if sweeps < 1:
        raise ValueError(f"the factorised planner takes sweeps >= 1, got {sweeps}")
    levels = tuple(int(n) for n in ctx.num_controls)
    horizon = int(ctx.policy_len)
    grids = [enumerate_policies((n,), horizon, (True,))[:, :, 0] for n in levels]
    counts = [int(g.shape[0]) for g in grids]
    priors = _node_priors(ctx, counts)
    args = (ctx.init_beliefs, ctx.transition_fn, ctx.step_terms_fn, ctx.terminal_terms_fn)

    marginals = [jnp.full((horizon, n), 1.0 / n, jnp.float32) for n in levels]
    posteriors = [Categorical(logits=jnp.zeros((c,), jnp.float32)) for c in counts]
    trees: list[TermTree | None] = [None] * len(levels)
    movement = jnp.zeros((), jnp.float32)

    def steps_for(k: int) -> ControlStep:
        count = counts[k]
        entries = tuple(
            grids[k]
            if j == k
            else jnp.broadcast_to(marginals[j][None], (count, horizon, levels[j]))
            for j in range(len(levels))
        )
        return ControlStep(entries=entries)

    for sweep in range(sweeps):
        last = sweep == sweeps - 1
        movement = jnp.zeros((), jnp.float32)
        for k, n in enumerate(levels):
            if n == 1 and not (last and metrics):
                continue  # one level: a constant marginal, scored once for the tree
            steps = steps_for(k)
            if metrics:  # one evaluator for every sweep, so movement is not reassociation
                tree = evaluate_policies(steps, *args)
                trees[k] = tree
                G = total(tree)
            else:
                G = evaluate_policies_total(steps, *args)
            if n == 1:
                continue
            q = policy_posterior(G, ctx.gamma, priors[k], use_bound=use_bound)
            new = _step_marginals(q, grids[k], n)
            movement = jnp.maximum(movement, jnp.max(jnp.abs(new - marginals[k])))
            posteriors[k], marginals[k] = q, new

    first = enumerate_policies(levels, 1, (True,) * len(levels))
    logits = jnp.zeros((first.shape[0],), jnp.float32)
    for k, n in enumerate(levels):
        logits = logits + _first_logits(posteriors[k], grids[k], n)[first[:, 0, k]]
    policies = FactorisedPolicies(
        first=first,
        grids=tuple(grids),
        posteriors=tuple(posteriors),
        trees=tuple(trees) if metrics else None,
        marginals=tuple(marginals),
        movement=movement,
    )
    return Categorical(logits=logits), policies


def make_factorised_planner(sweeps: int = DEFAULT_SWEEPS) -> Planner:
    """A factorised :class:`~hybrid_aif.planning.base.Planner` with ``sweeps`` bound (static: the
    sweeps are an unrolled loop). The registered ``"factorised"`` planner takes the default."""
    sweeps = int(sweeps)
    if sweeps < 1:
        raise ValueError(f"the factorised planner takes sweeps >= 1, got {sweeps}")

    def planner(ctx: PlannerContext, use_bound: UseBound = "midpoint", metrics: bool = True):
        q_pi, policies = factorised_plan(ctx, sweeps, use_bound=use_bound, metrics=metrics)
        return q_pi, ({} if metrics else None), policies

    planner.needs_key = False  # type: ignore[attr-defined]
    planner.branching = False  # type: ignore[attr-defined]
    planner.legal_depth = False  # type: ignore[attr-defined]
    planner.__name__ = f"factorised(sweeps={sweeps})"
    return planner


@register_planner("factorised")
def factorised_planner(
    ctx: PlannerContext, use_bound: UseBound = "midpoint", metrics: bool = True
) -> tuple[Categorical, TermTree | None, FactorisedPolicies]:
    """The :class:`~hybrid_aif.planning.base.Planner` face of this module.

    Registered as ``"factorised"`` with :data:`DEFAULT_SWEEPS` sweeps. Returns ``(q_pi, tree,
    policies)``: the posterior over the joint first-step actions, ``{}`` (``None`` under
    ``metrics=False``; the per-node trees are in ``policies.trees``), and the
    :class:`FactorisedPolicies`.
    """
    q_pi, policies = factorised_plan(ctx, DEFAULT_SWEEPS, use_bound=use_bound, metrics=metrics)
    return q_pi, ({} if metrics else None), policies
