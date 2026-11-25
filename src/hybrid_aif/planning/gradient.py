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

Warm starts. ``starts > 1`` descends from that many continuous plans together (one vmapped
descent): the first is ``init`` (or the box midpoint), the rest form a Latin hypercube
over the whole plan ``(policy_len, action_dim)``, mapped to the box (:func:`spread_starts`).
The construction is deterministic and needs no
key. The plan returned is the descent whose final score is lowest, with its trace, and
``start_scores`` holds every descent's final score, ``(starts,)``, so the spread across local
optima can be read. ``init`` may carry a leading ``(starts, policy_len, action_dim)`` axis to
supply the starts explicitly.

Design: docs/design/control-states.md C4 (The planner for the LQ demo), C5
"""

from collections.abc import Callable
from typing import Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hybrid_aif.distributions import Categorical, Gaussian, TruncatedGaussian
from hybrid_aif.distributions.gaussian import cholesky_of_inverse
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
    """A continuous plan and the Laplace posterior around it."""

    #: The plans as a policy set, ``(starts, policy_len, action_dim)`` (one member for one start),
    #: the best start's at the index ``q_pi`` is a point mass on.
    mode: Array
    #: The :class:`~hybrid_aif.distributions.TruncatedGaussian` on the box over the flattened
    #: plan, ``(policy_len · action_dim,)``, time-major.
    posterior: TruncatedGaussian

    def first_step(self, key: Array) -> Array:
        """A draw of the first step's action from the posterior (in the box by construction)."""
        dim = self.mode.shape[-1]
        return self.posterior.sample(key)[..., :dim]


class GradientResult(NamedTuple):
    """One gradient plan. With several starts, ``plan`` is the best start's and ``trace``,
    ``gradient_trace`` and ``best_step`` are its own."""

    #: The point mass on ``pick``, ``(starts,)``.
    q_pi: Categorical
    #: The term tree of ``plan`` (the picked start's slice of ``start_tree``).
    tree: TermTree | None
    #: The best plan scored over the descent (the warm start and the final plan included),
    #: ``(policy_len, action_dim)``: the best start's.
    plan: Array
    #: The expected free energy of ``plan``, ``(1,)`` (the picked start's slice of ``start_G``).
    G: Interval
    #: Every start's plan, ``(starts, policy_len, action_dim)``.
    start_plans: Array
    #: The int32 index of ``plan`` among ``start_plans``.
    pick: Array
    #: The expected free energy over ``start_plans`` (a policy axis of ``starts`` entries),
    #: ``(starts,)``.
    start_G: Interval
    #: The term tree over ``start_plans`` (``None`` under ``metrics=False``).
    start_tree: TermTree | None
    #: The restriction of ``curvature`` to the box (``None`` under ``posterior="mode"``).
    posterior: TruncatedGaussian | None
    #: The Laplace Gaussian ``N(plan, (γ∇²G)⁻¹)`` over the flattened plan, taken at ``plan``
    #: (``None`` under ``posterior="mode"``).
    curvature: Gaussian | None
    #: The scored ``G`` before each step.
    trace: Array
    #: The root mean square of the gradient over all plan coordinates before each step,
    #: ``(steps,)``.
    gradient_trace: Array
    #: The int32 index of the plan returned: ``t`` for the plan held before step ``t`` (``0`` is
    #: the warm start) and ``steps`` for the final plan.
    best_step: Array
    #: Each start's final score (the descended point of ``G``), ``(starts,)``; ``(1,)`` for one
    #: start.
    start_scores: Array


def _plastic_alphas(dim: int) -> np.ndarray:
    """The increments ``phi_d^-(j+1)``, ``j = 0 .. dim - 1``, of the generalised golden ratio
    ``phi_d``, the positive root of ``x^(dim+1) = x + 1`` (Newton's method)."""
    phi = 2.0
    for _ in range(60):
        phi -= (phi ** (dim + 1) - phi - 1.0) / ((dim + 1) * phi**dim - 1.0)
    return phi ** -(np.arange(dim) + 1.0)


def spread_starts(count: int, horizon: int, low: Array, high: Array) -> Array:
    """``count`` deterministic, well-spread plans ``(count, horizon, dim)`` strictly inside the box
    ``[low, high]``: a Latin hypercube over the flattened plan (time-major). The order of the
    points on each coordinate is that of the Kronecker sequence of the generalised golden ratio,
    ``frac(0.5 + i * alpha_j)`` for ``i = 1 .. count`` (which decorrelates the coordinates), and
    each coordinate's values are then replaced by ``(rank + 0.5) / count`` of the box, so every
    coordinate takes exactly those ``count`` values, in a different order per coordinate. For
    ``count = 1`` the point is the three-quarter point (``0.75``) of the box on every coordinate,
    not the centre, which is the default ``init``. No key is needed."""
    dim = horizon * low.shape[0]
    alphas = _plastic_alphas(dim)
    index = np.arange(1, count + 1, dtype=np.float64)[:, None]
    sequence = np.mod(0.5 + index * alphas[None, :], 1.0)
    ranks = np.argsort(np.argsort(sequence, axis=0, kind="stable"), axis=0, kind="stable")
    unit = (ranks + (0.75 if count == 1 else 0.5)) / count
    unit = jnp.asarray(unit, low.dtype).reshape(count, horizon, low.shape[0])
    return low + unit * (high - low)


def warm_starts(init: Array | None, starts: int, horizon: int, low: Array, high: Array) -> Array:
    """The warm-start plans ``(S, horizon, dim)`` of a descent: ``init`` (``(horizon, dim)``,
    ``(dim,)`` broadcast over the steps, or the box midpoint) first and ``starts - 1`` spread
    points after it (:func:`spread_starts`); an ``init`` with a leading axis ``(S, horizon,
    dim)`` is taken as the starts themselves, and ``starts`` must then be ``S`` or left at 1.
    Every start is clipped to the box."""
    if starts < 1:
        raise ValueError(f"starts must be at least 1, got {starts}")
    dim = low.shape[0]
    if init is not None and jnp.ndim(init) == 3:
        given = jnp.asarray(init, low.dtype)
        if given.shape[1:] != (horizon, dim) or starts not in (1, given.shape[0]):
            raise ValueError(
                f"init has shape {tuple(given.shape)} and starts={starts}; an init with a "
                f"leading axis is (starts, {horizon}, {dim}) and starts, when given, matches it"
            )
        return jnp.clip(given, low, high)
    first = jnp.broadcast_to(
        0.5 * (low + high) if init is None else jnp.asarray(init, low.dtype), (horizon, dim)
    )
    plans = first[None]
    if starts > 1:
        rest = spread_starts(starts - 1, horizon, low, high)
        plans = jnp.concatenate([plans, rest])
    return jnp.clip(plans, low, high)


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
    return descend_scored(score, plan0, low, high, rate, steps=steps, optimizer=optimizer)[:4]


def descend_scored(
    score: Callable[[Array], Array],
    plan0: Array,
    low: Array,
    high: Array,
    rate: Array,
    *,
    steps: int,
    optimizer: Optimizer,
) -> tuple[Array, Array, Array, Array, Array]:
    """:func:`descend` with the score of the returned plan appended: ``(plan, best_step, trace,
    gradient_trace, final_score)``."""
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
    final_score = jnp.where(final_best, last_g, best_g)
    return plan, best_step, trace, gradient_trace, final_score


def gradient_plan(
    ctx: PlannerContext,
    *,
    steps: int = DEFAULT_STEPS,
    lr: float | Array | None = None,
    optimizer: Optimizer = "adam",
    init: Array | None = None,
    starts: int = 1,
    posterior: Posterior = "mode",
    use_bound: UseBound = "midpoint",
    metrics: bool = True,
) -> GradientResult:
    """Descend ``G`` over the float plan against ``ctx``; see the module docstring.

    ``init`` is the starting plan, ``(policy_len, action_dim)`` or ``(action_dim,)`` broadcast
    over the steps (default: the box midpoint). ``starts`` descends from that many warm starts
    together (``init`` first, then spread points of the box; or ``init`` with a leading ``(starts,
    policy_len, action_dim)`` axis) and returns the best final plan, with every start's plan in
    ``start_plans``, the best one's index in ``pick`` and every start's final score in
    ``start_scores``. ``steps`` and ``lr`` set the descent;
    ``optimizer`` is ``"adam"`` or ``"gradient"``. Adam's ``lr`` is its first step, decayed
    linearly to zero (default :data:`DEFAULT_LR_FRACTION` of the box width per dimension); a
    plain gradient step's size depends on ``G``'s scale, so ``"gradient"`` takes an explicit
    ``lr``, held fixed. ``use_bound`` picks the point of ``G``'s interval that is descended.
    Adam's normalisation is floored at ``sqrt(machine epsilon of the plan's dtype) ·
    RMS(gradient)``, and the plan returned is the one with the lowest scored ``G`` over the warm
    start, the iterates and the final plan, unless the final plan's score is not finite (see
    the module docstring). With several starts the best final score wins; a NaN score never
    wins, and when every start's is NaN the first start's is returned. Convergence is read from
    the last entries of ``trace`` and
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
    horizon = ctx.policy_len
    low, high = layout.bounds
    rate = DEFAULT_LR_FRACTION * (high - low) if lr is None else jnp.asarray(lr, low.dtype)
    plan0 = warm_starts(init, starts, horizon, low, high)

    def score(plan: Array) -> Array:
        G = evaluate_policies_total(
            plan[None],
            ctx.init_beliefs,
            ctx.transition_fn,
            ctx.step_terms_fn,
            ctx.terminal_terms_fn,
        )
        return _point(G, use_bound)[0]

    if plan0.shape[0] == 1:
        plan, best_step, trace, gradient_trace, final = descend_scored(
            score, plan0[0], low, high, rate, steps=steps, optimizer=optimizer
        )
        start_scores = final[None]
        start_plans, pick = plan[None], jnp.zeros((), jnp.int32)
    else:
        plans, best_steps, traces, gradient_traces, start_scores = jax.vmap(
            lambda p0: descend_scored(score, p0, low, high, rate, steps=steps, optimizer=optimizer)
        )(plan0)
        pick = jnp.argmin(jnp.where(jnp.isnan(start_scores), jnp.inf, start_scores)).astype(
            jnp.int32
        )
        start_plans = plans
        plan, best_step = plans[pick], best_steps[pick]
        trace, gradient_trace = traces[pick], gradient_traces[pick]

    if metrics:
        tree = evaluate_policies(
            start_plans,
            ctx.init_beliefs,
            ctx.transition_fn,
            ctx.step_terms_fn,
            ctx.terminal_terms_fn,
        )
        start_G = total(tree)
    else:
        tree = None
        start_G = evaluate_policies_total(
            start_plans,
            ctx.init_beliefs,
            ctx.transition_fn,
            ctx.step_terms_fn,
            ctx.terminal_terms_fn,
        )
    start_tree = tree
    G = jax.tree_util.tree_map(lambda x: x[pick][None], start_G)
    if tree is not None:
        tree = jax.tree_util.tree_map(lambda x: x[pick][None], tree)
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
        q_pi=Categorical(
            logits=jnp.where(
                jnp.arange(start_plans.shape[0]) == pick, 0.0, jnp.finfo(low.dtype).min
            ).astype(low.dtype)
        ),
        tree=tree,
        plan=plan,
        G=G,
        start_plans=start_plans,
        pick=pick,
        start_G=start_G,
        start_tree=start_tree,
        posterior=restricted,
        curvature=curvature,
        trace=trace,
        gradient_trace=gradient_trace,
        best_step=best_step,
        start_scores=start_scores,
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
    return Gaussian(loc=flat, chol=cholesky_of_inverse(precision))


def make_gradient_planner(**kwargs) -> Planner:
    """A gradient :class:`~hybrid_aif.planning.base.Planner` with ``kwargs`` bound (those of
    :func:`gradient_plan`), e.g. ``make_gradient_planner(posterior="laplace", steps=400)``. It
    takes a warm start (``takes_init``): an ``init`` passed to the call replaces a bound one."""

    def planner(
        ctx: PlannerContext,
        use_bound: UseBound = "midpoint",
        metrics: bool = True,
        init: Array | None = None,
    ):
        bound = kwargs if init is None else {**kwargs, "init": init}
        return _face(gradient_plan(ctx, use_bound=use_bound, metrics=metrics, **bound), ctx)

    planner.needs_key = False  # type: ignore[attr-defined]
    planner.takes_init = True  # type: ignore[attr-defined]
    return planner


def _face(result: GradientResult, ctx: PlannerContext):
    policies = result.start_plans
    if result.posterior is not None:
        policies = PlanPosterior(mode=policies, posterior=result.posterior)
    return result.q_pi, result.start_tree, policies


@register_planner("gradient")
def gradient_planner(
    ctx: PlannerContext,
    use_bound: UseBound = "midpoint",
    metrics: bool = True,
    init: Array | None = None,
) -> tuple[Categorical, TermTree | None, Array]:
    """The :class:`~hybrid_aif.planning.base.Planner` face of this module.

    Registered as ``"gradient"`` with :func:`gradient_plan`'s defaults (Adam, the mode alone);
    ``init`` is :func:`gradient_plan`'s warm start (``takes_init``). Returns ``(q_pi, tree,
    policies)``: the point mass on the best start, the tree over every start's plan, and those
    plans as the policy set ``(starts, policy_len, action_dim)`` (one member for one start).
    """
    return _face(gradient_plan(ctx, use_bound=use_bound, metrics=metrics, init=init), ctx)


gradient_planner.takes_init = True  # type: ignore[attr-defined]
