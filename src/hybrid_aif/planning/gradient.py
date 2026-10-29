"""Gradient planner: the mode of ``q(π) ∝ exp(−γG)`` over continuous plans, by projected descent.

Starting from an initial plan (the box midpoint, or ``init``), the planner descends the expected
free energy ``G`` of the float plan ``(policy_len, action_dim)`` for a fixed number of steps,
projecting the plan onto the control nodes' box after each one. ``G`` is the planner context's
own score (the closures :meth:`~hybrid_aif.system.System.score_policies` folds), so the plan is
the mode of the policy posterior over continuous plans, as enumeration is its exact form over a
finite set and CEM its sampling form. The step is Adam (the default) or a plain gradient step.
Adam's step size is scaled to the box: by default ``0.05·(high − low)`` per dimension at the
first step, decaying linearly to zero over the steps, so the default 200 steps travel up to five
box widths and then settle. The default is tuned to the box: a very wide box (hundreds of the
curvature's standard deviations) may need more steps or an explicit ``lr=``. The loop is one
``lax.scan``, deterministic given its inputs.

Adam's per-coordinate normalisation is floored at the precision of the plan's dtype: the root of
the second moment is not allowed below ``sqrt(machine epsilon) · RMS(gradient)``, the root mean
square taken over every plan coordinate at that step. A coordinate whose gradient is rounding
noise of the evaluation of ``G`` then takes a step in proportion to that noise rather than a full
one, and a coordinate whose second moment is above the floor is normalised exactly as without it.
The plan returned is the best one scored: the iterate with the lowest ``G`` among the warm start,
every plan the descent visits and the final plan, so a result is never worse than its warm start
(up to the rounding between the term tree's total, which ``G`` reports, and the scalar score the
descent minimises). A descent whose final plan scores a non-finite ``G`` returns that plan and its
score, so a non-finite result is the signal of a non-finite descent. A gradient whose mean square
overflows leaves the floor at zero for that step.

``posterior="mode"`` (the default) emits the plan alone. ``posterior="laplace"`` also emits the
Laplace approximation around it, restricted to the box: the curvature's Gaussian has covariance
``(γ ∇²G)⁻¹`` over the flattened plan (``jax.hessian`` at the plan), and the emitted posterior is
the :class:`~hybrid_aif.distributions.TruncatedGaussian` on the box with the plan as ``loc`` and
that covariance's diagonal as ``scale²`` (the cross-step correlations are not carried). With a
box narrower than the curvature's standard deviation the posterior is close to uniform over the
box, which is the correct restriction. A
curvature that is not positive definite is refused, eagerly with a sentence and under a trace by
``equinox.error_if``. The registered planner returns the plan as a one-member policy set
``(1, policy_len, action_dim)``, wrapped in a :class:`PlanPosterior` under ``"laplace"`` so that
``act`` can draw the emitted action from it.

Design: docs/design/control-states.md C4 (The planner for the LQ demo), C5
"""

from collections.abc import Callable
from typing import Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.distributions import Categorical, Gaussian, TruncatedGaussian
from hybrid_aif.interval import Interval
from hybrid_aif.objectives.base import TermTree, total
from hybrid_aif.planning.base import (
    Planner,
    PlannerContext,
    UseBound,
    refuse_mixed,
    register_planner,
)
from hybrid_aif.planning.enumerate import evaluate_policies, evaluate_policies_total

#: Defaults: this many steps, Adam's first step this fraction of the box width.
DEFAULT_STEPS = 200
DEFAULT_LR_FRACTION = 0.05
#: Adam's moments: a short second-moment memory (β₂ = 0.95), so the step size recovers once
#: the large gradients of a far start are behind it. ``_ADAM_EPS`` is an absolute term added to
#: the denominator, so that a step is exactly zero where the whole gradient is exactly zero.
_ADAM_B1, _ADAM_B2, _ADAM_EPS = 0.9, 0.95, 1e-8

Optimizer = Literal["adam", "gradient"]
Posterior = Literal["mode", "laplace"]


class PlanPosterior(NamedTuple):
    """A continuous plan and the Laplace posterior around it.

    ``mode`` is the plan as a one-member policy set ``(1, policy_len, action_dim)``;
    ``posterior`` is the :class:`~hybrid_aif.distributions.TruncatedGaussian` on the box over
    the flattened plan ``(policy_len · action_dim,)``, time-major.
    """

    mode: Array
    posterior: TruncatedGaussian

    def first_step(self, key: Array) -> Array:
        """A draw of the first step's action from the posterior (in the box by construction)."""
        dim = self.mode.shape[-1]
        return self.posterior.sample(key)[..., :dim]


class GradientResult(NamedTuple):
    """One gradient plan: ``plan`` ``(policy_len, action_dim)``, its ``G`` and term ``tree``
    (``None`` under ``metrics=False``), ``q_pi`` the point mass on it, ``curvature`` the Laplace
    Gaussian ``N(plan, (γ∇²G)⁻¹)`` over the flattened plan and ``posterior`` its restriction to
    the box (both ``None`` under ``posterior="mode"``), and ``trace`` the scored ``G`` before
    each step. ``gradient_trace`` ``(steps,)`` is the root mean square of the gradient over all
    plan coordinates before each step, and ``best_step`` is the int32 index of the plan returned:
    ``t`` for the plan held before step ``t`` (``0`` is the warm start) and ``steps`` for the
    final plan. ``plan`` is the best plan scored over the descent (the warm start and the final
    plan included), ``G`` and ``tree`` are its own, and ``curvature`` is taken at it."""

    q_pi: Categorical
    tree: TermTree | None
    plan: Array
    G: Interval
    posterior: TruncatedGaussian | None
    curvature: Gaussian | None
    trace: Array
    gradient_trace: Array
    best_step: Array


def _point(G: Interval, use_bound: UseBound) -> Array:
    return {"midpoint": G.midpoint, "lower": G.lower, "upper": G.upper}[use_bound]


def check_descent(
    steps: int, lr: float | Array | None, optimizer: str, posterior: str | None = None
) -> None:
    """Raise, with a sentence, on an option of the descent that is not defined: an unknown
    ``optimizer`` (or ``posterior``, when given), negative ``steps``, or a plain gradient step
    without an ``lr``."""
    if optimizer not in ("adam", "gradient"):
        raise ValueError(f'optimizer must be "adam" or "gradient", got {optimizer!r}')
    if posterior is not None and posterior not in ("mode", "laplace"):
        raise ValueError(f'posterior must be "mode" or "laplace", got {posterior!r}')
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")
    if optimizer == "gradient" and lr is None:
        raise ValueError(
            "a plain gradient step's size is set by G's scale, not the box: pass lr= with "
            'optimizer="gradient", or use the default "adam"'
        )


def descend(
    score: Callable[[Array], Array],
    plan0: Array,
    low: Array,
    high: Array,
    rate: Array,
    *,
    steps: int,
    optimizer: Optimizer,
) -> tuple[Array, Array, Array, Array]:
    """Projected descent of the scalar ``score(plan)`` from ``plan0`` inside the box ``[low,
    high]`` (broadcast against the plan's trailing axis), by Adam or a plain gradient step.

    Returns ``(plan, best_step, trace, gradient_trace)``: the best plan scored (the warm start,
    every iterate, the final plan; a non-finite final score is returned as it is), the int32
    index of that plan, the scored value before each step and the root mean square of the
    gradient over all plan coordinates before each step. Adam's step is ``rate`` decayed
    linearly to zero over ``steps`` and its normalisation is floored at ``sqrt(machine epsilon)
    · RMS(gradient)``. This is the descent :func:`gradient_plan` runs; see the module docstring.
    """
    value_and_grad = jax.value_and_grad(score)

    def step(carry, t):
        plan, m, v, best_plan, best_g, best_t = carry
        g_value, grad = value_and_grad(plan)
        better = g_value.astype(best_g.dtype) <= best_g
        best_plan = jnp.where(better, plan, best_plan)
        best_g = jnp.where(better, g_value.astype(best_g.dtype), best_g)
        best_t = jnp.where(better, t.astype(jnp.int32), best_t)
        rms = jnp.sqrt(jnp.mean(grad**2))
        if optimizer == "adam":
            m = _ADAM_B1 * m + (1.0 - _ADAM_B1) * grad
            v = _ADAM_B2 * v + (1.0 - _ADAM_B2) * grad**2
            m_hat = m / (1.0 - _ADAM_B1 ** (t + 1.0))
            v_hat = v / (1.0 - _ADAM_B2 ** (t + 1.0))
            floor = jnp.where(jnp.isfinite(rms), jnp.sqrt(jnp.finfo(plan.dtype).eps) * rms, 0.0)
            scale = jnp.maximum(jnp.sqrt(v_hat), floor) + _ADAM_EPS
            update = (1.0 - t / max(steps, 1)) * m_hat / scale
        else:
            update = grad
        plan = jnp.clip(plan - rate * update, low, high)
        return (plan, m, v, best_plan, best_g, best_t), (g_value, rms)

    zeros = jnp.zeros_like(plan0)
    start = (plan0, zeros, zeros, plan0, jnp.asarray(jnp.inf, plan0.dtype), jnp.int32(0))
    (last, _, _, best_plan, best_g, best_t), (trace, gradient_trace) = jax.lax.scan(
        step, start, jnp.arange(steps, dtype=low.dtype)
    )
    last_g = score(last).astype(best_g.dtype)
    final_best = ~jnp.isfinite(last_g) | (last_g <= best_g)
    plan = jnp.where(final_best, last, best_plan)
    best_step = jnp.where(final_best, jnp.int32(steps), best_t)
    return plan, best_step, trace, gradient_trace


def gradient_plan(
    ctx: PlannerContext,
    *,
    steps: int = DEFAULT_STEPS,
    lr: float | Array | None = None,
    optimizer: Optimizer = "adam",
    init: Array | None = None,
    posterior: Posterior = "mode",
    use_bound: UseBound = "midpoint",
    metrics: bool = True,
) -> GradientResult:
    """Descend ``G`` over the float plan against ``ctx``; see the module docstring.

    ``init`` is the starting plan, ``(policy_len, action_dim)`` or ``(action_dim,)`` broadcast
    over the steps (default: the box midpoint). ``steps`` and ``lr`` set the descent;
    ``optimizer`` is ``"adam"`` or ``"gradient"``. Adam's ``lr`` is its first step, decayed
    linearly to zero (default :data:`DEFAULT_LR_FRACTION` of the box width per dimension); a
    plain gradient step's size depends on ``G``'s scale, so ``"gradient"`` takes an explicit
    ``lr``, held fixed. ``use_bound`` picks the point of ``G``'s interval that is descended.
    Adam's normalisation is floored at ``sqrt(machine epsilon of the plan's dtype) ·
    RMS(gradient)``, and the plan returned is the one with the lowest scored ``G`` over the warm
    start, the iterates and the final plan, unless the final plan's score is not finite (see
    the module docstring). Convergence is read from the last entries of ``trace`` and
    ``gradient_trace``, since a descent from a warm start can raise ``G`` over its first steps
    before settling, and ``best_step`` names the iterate returned.
    """
    refuse_mixed(ctx, "gradient")
    layout = ctx.continuous_controls
    if layout is None:
        raise ValueError(
            "the gradient planner descends G over continuous plans, and this System's control "
            "nodes are discrete (level indices): give a node a box in params.control and a "
            'float-reading transition, or plan with the "enumeration" planner'
        )
    check_descent(steps, lr, optimizer, posterior)
    horizon, dim = ctx.policy_len, layout.action_dim
    low, high = layout.bounds
    rate = DEFAULT_LR_FRACTION * (high - low) if lr is None else jnp.asarray(lr, low.dtype)
    plan0 = (
        jnp.broadcast_to(0.5 * (low + high), (horizon, dim))
        if init is None
        else jnp.broadcast_to(jnp.asarray(init, low.dtype), (horizon, dim))
    )
    plan0 = jnp.clip(plan0, low, high)

    def score(plan: Array) -> Array:
        G = evaluate_policies_total(
            plan[None],
            ctx.init_beliefs,
            ctx.transition_fn,
            ctx.step_terms_fn,
            ctx.terminal_terms_fn,
        )
        return _point(G, use_bound)[0]

    plan, best_step, trace, gradient_trace = descend(
        score, plan0, low, high, rate, steps=steps, optimizer=optimizer
    )

    if metrics:
        tree = evaluate_policies(
            plan[None],
            ctx.init_beliefs,
            ctx.transition_fn,
            ctx.step_terms_fn,
            ctx.terminal_terms_fn,
        )
        G = total(tree)
    else:
        tree = None
        G = evaluate_policies_total(
            plan[None],
            ctx.init_beliefs,
            ctx.transition_fn,
            ctx.step_terms_fn,
            ctx.terminal_terms_fn,
        )
    curvature = restricted = None
    if posterior == "laplace":
        curvature = _laplace(score, plan, ctx.gamma)
        restricted = TruncatedGaussian(
            loc=curvature.loc,
            scale=jnp.sqrt(jnp.diagonal(curvature.cov)),
            low=jnp.tile(low, horizon),
            high=jnp.tile(high, horizon),
        )
    return GradientResult(
        q_pi=Categorical(logits=jnp.zeros((1,), low.dtype)),
        tree=tree,
        plan=plan,
        G=G,
        posterior=restricted,
        curvature=curvature,
        trace=trace,
        gradient_trace=gradient_trace,
        best_step=best_step,
    )


_NOT_PD = (
    "the curvature of G at the plan is not positive definite, so it defines no Laplace "
    'posterior: plan with posterior="mode", or give the plan a strict minimum (a target on the '
    "control node)"
)


def _laplace(score, plan: Array, gamma: float) -> Gaussian:
    """The Gaussian around ``plan`` with covariance ``(γ ∇²G)⁻¹`` over the flattened plan."""
    shape = plan.shape
    flat = plan.reshape(-1)
    hessian = jax.hessian(lambda x: score(x.reshape(shape)))(flat)
    precision = gamma * 0.5 * (hessian + hessian.T)
    smallest = jnp.min(jnp.linalg.eigvalsh(precision))
    if not isinstance(smallest, jax.core.Tracer) and not bool(smallest > 0.0):
        raise ValueError(f"{_NOT_PD} (smallest eigenvalue {float(smallest):.3g})")
    precision = eqx.error_if(precision, ~(smallest > 0.0), _NOT_PD)
    cov = jnp.linalg.inv(precision)
    return Gaussian.from_cov(flat, 0.5 * (cov + cov.T))


def make_gradient_planner(**kwargs) -> Planner:
    """A gradient :class:`~hybrid_aif.planning.base.Planner` with ``kwargs`` bound (those of
    :func:`gradient_plan`), e.g. ``make_gradient_planner(posterior="laplace", steps=400)``."""

    def planner(ctx: PlannerContext, use_bound: UseBound = "midpoint", metrics: bool = True):
        return _face(gradient_plan(ctx, use_bound=use_bound, metrics=metrics, **kwargs), ctx)

    planner.needs_key = False  # type: ignore[attr-defined]
    return planner


def _face(result: GradientResult, ctx: PlannerContext):
    policies = result.plan[None]
    if result.posterior is not None:
        policies = PlanPosterior(mode=policies, posterior=result.posterior)
    return result.q_pi, result.tree, policies


@register_planner("gradient")
def gradient_planner(
    ctx: PlannerContext, use_bound: UseBound = "midpoint", metrics: bool = True
) -> tuple[Categorical, TermTree | None, Array]:
    """The :class:`~hybrid_aif.planning.base.Planner` face of this module.

    Registered as ``"gradient"`` with :func:`gradient_plan`'s defaults (Adam, the mode alone).
    Returns ``(q_pi, tree, policies)``: the point mass, the tree at the plan, and the plan as
    a one-member policy set ``(1, policy_len, action_dim)``.
    """
    return _face(gradient_plan(ctx, use_bound=use_bound, metrics=metrics), ctx)
