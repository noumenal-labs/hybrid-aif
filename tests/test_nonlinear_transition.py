"""Nonlinear transitions, propagated along the plan.

A :class:`~hybrid_aif.factors.NonlinearGaussianTransition` declares ``x' = f(x, controls) + ε``
with ``f`` an equinox module. A belief is pushed through ``f`` at its own moments, to first order
or by the unscented transform. These tests pin: the linear path reproduced by an affine ``f``;
the unicycle's 180° turn against Monte Carlo, with the frozen linearisation's error for contrast;
the customer's failure (a lane preference and a target behind the robot) as a planning test that
the frozen linearisation fails; a control carried as a belief; integer levels reaching ``f``;
every planner; the epistemic chain; the refusals; and the surface (``as_env``, description
length, learning).

Design: docs/design/nonlinear-transitions.md N1–N6.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import (
    Box,
    Categorical,
    ConditionalCategorical,
    ConditionalGaussian,
    DescriptionError,
    Gaussian,
    LinearGaussianObservation,
    LinearGaussianTransition,
    MixtureGaussian,
    NonlinearGaussianTransition,
    System,
    SystemParams,
    describe,
)
from hybrid_aif.composition import ActionControl, BoundaryPrior, merge
from hybrid_aif.controls import ControlStep
from hybrid_aif.description.mdl import description_length, resolution_term
from hybrid_aif.learning.carry import learnable_blocks
from hybrid_aif.objectives import hybrid as _hybrid
from hybrid_aif.objectives import total
from hybrid_aif.planning import gradient_plan

KEY = jax.random.PRNGKey(0)

# --------------------------------------------------------------------------- the maps


class Affine(eqx.Module):
    """``F_m x + G u``: the linear path written as a map (``m`` a discrete level, ``u`` a
    continuous value)."""

    F: jax.Array  # (levels, d, d)
    G: jax.Array  # (d, du)

    def __call__(self, x, controls):
        level, u = controls
        return self.F[level] @ x + self.G @ u


class Unicycle(eqx.Module):
    """The unicycle at speed ``v`` and turn rate ``w`` over a step ``h``, uncontrolled."""

    h: float
    v: float
    w: float

    def __call__(self, x, controls):
        del controls
        return jnp.stack(
            [
                x[0] + self.h * self.v * jnp.cos(x[2]),
                x[1] + self.h * self.v * jnp.sin(x[2]),
                x[2] + self.h * self.w,
            ]
        )


class SteeredUnicycle(eqx.Module):
    """The unicycle steered by a discrete turn-rate level, ``rates[level]``; the level must
    arrive as an integer."""

    h: float
    v: float
    rates: jax.Array

    def __call__(self, x, controls):
        (level,) = controls
        assert jnp.issubdtype(level.dtype, jnp.integer), level.dtype
        w = self.rates[level]
        return jnp.stack(
            [
                x[0] + self.h * self.v * jnp.cos(x[2]),
                x[1] + self.h * self.v * jnp.sin(x[2]),
                x[2] + self.h * w,
            ]
        )


class RateUnicycle(eqx.Module):
    """The unicycle steered by a continuous turn rate ``u[0]``."""

    h: float
    v: float

    def __call__(self, x, controls):
        (u,) = controls
        return jnp.stack(
            [
                x[0] + self.h * self.v * jnp.cos(x[2]),
                x[1] + self.h * self.v * jnp.sin(x[2]),
                x[2] + self.h * u[0],
            ]
        )


class DrivenUnicycle(eqx.Module):
    """The unicycle driven by a continuous ``(speed, turn rate)``: linear in the control."""

    h: float

    def __call__(self, x, controls):
        (u,) = controls
        return jnp.stack(
            [
                x[0] + self.h * u[0] * jnp.cos(x[2]),
                x[1] + self.h * u[0] * jnp.sin(x[2]),
                x[2] + self.h * u[1],
            ]
        )


def _noise(loc, chol) -> ConditionalGaussian:
    loc = jnp.asarray(loc, jnp.float32)
    shape = loc.shape[:-1]
    chol = jnp.broadcast_to(jnp.asarray(chol, jnp.float32), (*shape, loc.shape[-1], loc.shape[-1]))
    return ConditionalGaussian(loc=loc, chol=chol, parent_ndims=len(shape))


def _close(a, b, rtol=1e-5, atol=1e-6) -> None:
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb, strict=True):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=rtol, atol=atol)


# ------------------------------------------------- N6 (1): an affine f is the linear path

F_MODES = jnp.asarray([[[1.0, 0.1], [0.0, 0.9]], [[0.95, -0.2], [0.1, 1.0]]])
G_U = jnp.asarray([[0.2], [0.5]])
B_MODES = jnp.asarray([[0.05, 0.0], [-0.1, 0.02]])
Q_MODES = jnp.stack([0.1 * jnp.eye(2), jnp.asarray([[0.12, 0.0], [0.03, 0.08]])])


def _affine_pair(carry: str, propagation: str):
    """The same model twice, its transition linear-Gaussian and an affine map: a 2-d state, a
    discrete mode node ``m`` (per-mode ``F``, offset and noise) and a continuous node ``u``
    (``G u``), observed by a linear-Gaussian channel, with a state, an observation and a
    terminal target."""
    system = System(
        A_deps=((0,),),
        B_deps=((1, 2), (), ()),
        controls=(1, 2),
        action_names=("m", "u"),
        factor_names=("x", "u_m", "u_u"),
        modality_names=("y",),
        policy_len=3,
        mixture_k=2 if carry == "mixture" else None,
        planning_belief="posterior" if carry == "posterior" else "prior",
    )
    innovations = ConditionalGaussian(loc=B_MODES, chol=Q_MODES, parent_ndims=1)
    lg = LinearGaussianTransition(F=F_MODES, innovations=innovations, G=G_U)
    nl = NonlinearGaussianTransition(Affine(F_MODES, G_U), innovations, propagation)
    if carry == "mixture":
        prior = MixtureGaussian(
            log_weights=jnp.log(jnp.asarray([0.3, 0.7])),
            means=jnp.asarray([[0.4, -0.2], [-0.3, 0.5]]),
            chols=jnp.stack([0.2 * jnp.eye(2), jnp.asarray([[0.3, 0.0], [0.1, 0.15]])]),
        )
    else:
        prior = Gaussian(loc=jnp.asarray([0.4, -0.2]), chol=jnp.asarray([[0.3, 0.0], [0.1, 0.2]]))

    def params(B):
        return SystemParams(
            likelihood={
                "y": LinearGaussianObservation(
                    H=jnp.asarray([[1.0, 0.0]]), c=jnp.zeros(1), noise=jnp.asarray([[0.2]])
                )
            },
            transition={"x": B},
            initial_state_prior={"x": prior},
            obs_targets={"y": Gaussian(loc=jnp.asarray([1.0]), chol=jnp.asarray([[0.5]]))},
            state_targets={"x": Gaussian(loc=jnp.asarray([1.0, 0.0]), chol=jnp.eye(2))},
            terminal_targets={"x": Gaussian(loc=jnp.asarray([1.0, 0.5]), chol=0.5 * jnp.eye(2))},
            control={"u_u": Box(-jnp.ones(1), jnp.ones(1))},
        )

    return system, params(lg), params(nl)


@pytest.mark.parametrize("propagation", ["first_order", "unscented"])
@pytest.mark.parametrize("carry", ["gaussian", "mixture", "posterior"])
def test_an_affine_map_reproduces_the_linear_path(carry, propagation):
    """Both propagations are exact for an affine ``f``: the one-step beliefs and every leaf of
    the term tree match the linear path's, under clamped and belief-valued controls,
    Gaussian-carried (its epistemic chain too) and mixture-carried."""
    system, lg, nl = _affine_pair(carry, propagation)
    beliefs = system.initial_priors(lg)
    action = {"m": jnp.int32(1), "u": jnp.asarray([0.3])}
    _close(
        system.update_empirical_prior(nl, beliefs, action),
        system.update_empirical_prior(lg, beliefs, action),
    )
    levels = jnp.asarray([[0, 1, 1], [1, 0, 1]], jnp.int32)
    values = jnp.asarray([[[0.3], [-0.5], [0.9]], [[0.0], [0.4], [-0.2]]])
    clamped = {"u_m": levels, "u_u": values}
    _close(system.score_policies(nl, beliefs, clamped), system.score_policies(lg, beliefs, clamped))
    belief = Gaussian(loc=values, chol=jnp.full((2, 3, 1, 1), 0.3))
    believed = {"u_m": levels, "u_u": belief}
    _close(
        system.score_policies(nl, beliefs, believed), system.score_policies(lg, beliefs, believed)
    )


# --------------------------------------------- N6 (2): the unicycle's 180° turn vs Monte Carlo

TURN_H, TURN_V, TURN_W, TURN_T = 0.1, 0.5, np.pi / 2, 20
TURN_Q = np.diag([0.01, 0.01, 0.005])  # the process noise's Cholesky factor (Q its square)


def _monte_carlo(heading_sd: float, draws: int = 200_000):
    """The final pose's mean and covariance after the turn, by sampling (float64, seeded)."""
    rng = np.random.default_rng(0)
    xs = rng.multivariate_normal(np.zeros(3), np.diag([0.1, 0.1, heading_sd]) ** 2, size=draws)
    for _ in range(TURN_T):
        xs = np.stack(
            [
                xs[:, 0] + TURN_H * TURN_V * np.cos(xs[:, 2]),
                xs[:, 1] + TURN_H * TURN_V * np.sin(xs[:, 2]),
                xs[:, 2] + TURN_H * TURN_W,
            ],
            1,
        )
        xs += rng.standard_normal(xs.shape) @ TURN_Q.T
    return xs.mean(0), np.cov(xs.T)


@pytest.fixture(scope="module")
def monte_carlo():
    return {sd: _monte_carlo(sd) for sd in (0.2, 0.5)}


def _turn(decl, heading_sd: float) -> Gaussian:
    system = System(B_deps=((),), factor_names=("pose",), policy_len=1, epistemic=False)
    params = SystemParams(
        transition={"pose": decl},
        initial_state_prior={
            "pose": Gaussian(loc=jnp.zeros(3), chol=jnp.diag(jnp.asarray([0.1, 0.1, heading_sd])))
        },
    )
    step = jax.jit(lambda b: system.update_empirical_prior(params, b, {}))
    beliefs = system.initial_priors(params)
    for _ in range(TURN_T):
        beliefs = step(beliefs)
    return beliefs["pose"]


def _reference_ukf_turn(heading_sd: float):
    """The unscented transform written out in float64 (α = 1, β = 2, κ = 0)."""
    n, beta = 3, 2.0
    m, P, Q = np.zeros(3), np.diag([0.1, 0.1, heading_sd]) ** 2, TURN_Q @ TURN_Q.T
    wm = np.full(2 * n + 1, 1 / (2 * n))
    wm[0] = 0.0
    wc = wm.copy()
    wc[0] += beta
    for _ in range(TURN_T):
        L = np.linalg.cholesky(n * P)
        pts = [m] + [m + L[:, i] for i in range(n)] + [m - L[:, i] for i in range(n)]
        ys = np.array(
            [
                [
                    p[0] + TURN_H * TURN_V * np.cos(p[2]),
                    p[1] + TURN_H * TURN_V * np.sin(p[2]),
                    p[2] + TURN_H * TURN_W,
                ]
                for p in pts
            ]
        )
        m = wm @ ys
        dev = ys - m
        P = (wc[:, None] * dev).T @ dev + Q
    return m, P


@pytest.mark.parametrize("heading_sd", [0.2, 0.5])
def test_the_turn_against_monte_carlo(monte_carlo, heading_sd):
    """First order is within the measured error of the sampled final position (0.013 m at a
    heading sd of 0.2, 0.075 m at 0.5), unscented within 0.005 m at both; the unscented
    covariance is the float64 transform's; the frozen linearisation about the initial heading
    is 1.3 m off whatever the spread."""
    mc_mean, mc_cov = monte_carlo[heading_sd]
    noise = _noise(jnp.zeros((1, 3)), TURN_Q)
    unicycle = Unicycle(TURN_H, TURN_V, TURN_W)
    first = _turn(NonlinearGaussianTransition(unicycle, noise), heading_sd)
    sigma = _turn(NonlinearGaussianTransition(unicycle, noise, "unscented"), heading_sd)
    frozen = _turn(
        LinearGaussianTransition(
            F=jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, TURN_H * TURN_V], [0.0, 0.0, 1.0]]),
            innovations=_noise(jnp.asarray([[TURN_H * TURN_V, 0.0, TURN_H * TURN_W]]), TURN_Q),
        ),
        heading_sd,
    )

    def error(q: Gaussian) -> float:
        return float(np.linalg.norm(np.asarray(q.loc[:2]) - mc_mean[:2]))

    assert error(first) < {0.2: 0.02, 0.5: 0.08}[heading_sd]
    assert error(sigma) < 0.005
    assert error(frozen) > 1.2
    ref_mean, ref_cov = _reference_ukf_turn(heading_sd)
    np.testing.assert_allclose(np.asarray(sigma.loc), ref_mean, atol=1e-4)
    np.testing.assert_allclose(np.asarray(sigma.cov), ref_cov, atol=1e-4)
    sd = {name: np.sqrt(np.diag(np.asarray(q.cov))[:2]) for name, q in (("1", first), ("u", sigma))}
    mc_sd = np.sqrt(np.diag(mc_cov)[:2])
    if heading_sd == 0.2:  # both spreads match the samples at a moderate heading spread
        np.testing.assert_allclose(sd["1"], mc_sd, atol=0.005)
        np.testing.assert_allclose(sd["u"], mc_sd, atol=0.005)
    else:  # at a wide one first order overstates the along-track spread more than unscented
        assert abs(sd["u"][0] - mc_sd[0]) < abs(sd["1"][0] - mc_sd[0])


# ------------------------------------- N6 (3): the customer's failure, reproduced as planning

PLAN_H, PLAN_V, PLAN_W = 0.5, 0.5, np.pi / 2
PLAN_Q = jnp.diag(jnp.asarray([0.02, 0.02, 0.02]))
#: The lane: stay near y = 0, nothing said of x or the heading.
LANE = Gaussian(loc=jnp.zeros(3), chol=jnp.diag(jnp.asarray([100.0, 0.4, 100.0])))
#: Behind the robot and a little to its left, facing back.
BEHIND = Gaussian(loc=jnp.asarray([-1.0, 0.3, np.pi]), chol=0.5 * jnp.eye(3))
#: The customer's frozen linearisation at the initial heading: ``y' = y + h v θ``.
FROZEN_F = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, PLAN_H * PLAN_V], [0.0, 0.0, 1.0]])
RATES = jnp.asarray([-PLAN_W, 0.0, PLAN_W])


def _steered(decl, **overrides):
    wiring = {"B_deps": ((),), "controlled": (True,), "factor_names": ("pose",)}
    if "controls" in overrides:
        wiring = {}
    system = System(**{**wiring, "policy_len": 4, "epistemic": False, **overrides})
    params = SystemParams(
        transition={"pose": decl},
        initial_state_prior={"pose": Gaussian(loc=jnp.zeros(3), chol=0.05 * jnp.eye(3))},
        state_targets={"pose": LANE},
        terminal_targets={"pose": BEHIND},
    )
    return system, params


def _steered_nonlinear(propagation="first_order"):
    return NonlinearGaussianTransition(
        SteeredUnicycle(PLAN_H, PLAN_V, RATES), _noise(jnp.zeros((3, 3)), PLAN_Q), propagation
    )


def _steered_frozen():
    offsets = jnp.stack([jnp.asarray([PLAN_H * PLAN_V, 0.0, PLAN_H * w]) for w in RATES])
    return LinearGaussianTransition(F=FROZEN_F, innovations=_noise(offsets, PLAN_Q))


def _first_action(system, params) -> int:
    """The first action of the best enumerated plan."""
    beliefs = system.initial_priors(params)
    policies = system.policies(params)
    G = np.asarray(total(system.score_policies(params, beliefs, policies)).midpoint)
    return int(policies[int(np.argmin(G)), 0, 0])


@pytest.mark.parametrize("propagation", ["first_order", "unscented"])
def test_the_plan_turns_toward_a_target_behind(propagation):
    """With a lane preference and a target behind on the left, the plan propagated through
    the unicycle turns left at once; the frozen linearisation predicts the turn as a sideways
    move the lane charges for, and drives straight on."""
    system, params = _steered(_steered_nonlinear(propagation))
    assert _first_action(system, params) == 2  # turn left, ω > 0
    q_pi, _ = system.infer_policies(params, system.initial_priors(params))[:2]
    best = system.policies(params)[int(jnp.argmax(q_pi.probs))]
    assert int(best[0, 0]) == 2
    frozen, frozen_params = _steered(_steered_frozen())
    assert _first_action(frozen, frozen_params) == 1  # straight on, away from the target


def test_the_gradient_plan_turns_toward_a_target_behind():
    """The same with a continuous turn rate: the gradient planner, whose rollout re-linearises
    at every iteration, plans the full left turn first; the frozen linearisation does not
    turn toward the target."""
    box = {"u_w": Box(-PLAN_W * jnp.ones(1), PLAN_W * jnp.ones(1))}
    over = dict(
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("w",),
        factor_names=("pose", "u_w"),
        planner="gradient",
    )
    nl = NonlinearGaussianTransition(
        RateUnicycle(PLAN_H, PLAN_V), _noise(jnp.zeros((1, 3)), PLAN_Q)
    )
    frozen = LinearGaussianTransition(
        F=FROZEN_F,
        innovations=_noise(jnp.asarray([[PLAN_H * PLAN_V, 0.0, 0.0]]), PLAN_Q),
        G=jnp.asarray([[0.0], [0.0], [PLAN_H]]),
    )
    first = {}
    for name, decl in (("nonlinear", nl), ("frozen", frozen)):
        system, params = _steered(decl, **over)
        params = dataclasses.replace(params, control=box)
        beliefs = system.initial_priors(params)
        plan = gradient_plan(system.planner_context(params, beliefs), steps=100).plan
        assert np.all(np.isfinite(np.asarray(plan)))
        first[name] = float(plan[0, 0])

        def score(plan, system=system, params=params, beliefs=beliefs):
            return total(system.score_policies(params, beliefs, plan)).midpoint

        grad = jax.grad(score)(jnp.full((4, 1), 0.3))
        assert np.all(np.isfinite(np.asarray(grad)))
    assert first["nonlinear"] > 0.9 * PLAN_W
    assert first["frozen"] < 0.5 * PLAN_W


# ---------------------------------------------------- N6 (4): a control carried as a belief


def _driven(propagation: str):
    system = System(
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("u",),
        factor_names=("pose", "u_u"),
        policy_len=1,
        planner="gradient",
        epistemic=False,
    )
    params = SystemParams(
        transition={
            "pose": NonlinearGaussianTransition(
                DrivenUnicycle(0.5), _noise(jnp.zeros((1, 3)), 0.01 * jnp.eye(3)), propagation
            )
        },
        initial_state_prior={"pose": Gaussian(loc=jnp.zeros(3), chol=jnp.eye(3))},
        control={"u_u": Box(-2.0 * jnp.ones(2), 2.0 * jnp.ones(2))},
    )
    pos = system._p(params)
    step = _hybrid.make_transition(
        pos.transition,
        system._B,
        system._families(pos),
        None,
        table=system._table,
        space=system._space(pos),
    )
    return lambda q, u: step((q,), ControlStep(entries=(u,)))[0]


X_BELIEF = Gaussian(
    loc=jnp.asarray([0.2, -0.1, 0.4]),
    chol=jnp.asarray([[0.1, 0.0, 0.0], [0.02, 0.1, 0.0], [0.0, 0.0, 0.3]]),
)
U_BELIEF = Gaussian(loc=jnp.asarray([0.8, 0.5]), chol=jnp.asarray([[0.2, 0.0], [0.05, 0.3]]))


def test_first_order_adds_the_control_spread_through_its_jacobian():
    """``f`` linear in ``u``: ``Σ' = J_x Σ J_xᵀ + J_u S J_uᵀ + Q`` at the means, by hand."""
    h, Q = 0.5, 0.01**2 * np.eye(3)
    m, S = np.asarray(X_BELIEF.loc, np.float64), np.asarray(X_BELIEF.cov, np.float64)
    mu, Su = np.asarray(U_BELIEF.loc, np.float64), np.asarray(U_BELIEF.cov, np.float64)
    c, s = np.cos(m[2]), np.sin(m[2])
    J_x = np.asarray([[1, 0, -h * mu[0] * s], [0, 1, h * mu[0] * c], [0, 0, 1]])
    J_u = np.asarray([[h * c, 0], [h * s, 0], [0, h]])
    want_mean = m + np.asarray([h * mu[0] * c, h * mu[0] * s, h * mu[1]])
    want_cov = J_x @ S @ J_x.T + J_u @ Su @ J_u.T + Q
    got = _driven("first_order")(X_BELIEF, U_BELIEF)
    np.testing.assert_allclose(np.asarray(got.loc), want_mean, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(got.cov), want_cov, rtol=1e-4, atol=1e-6)
    #  the clamp at the mean drops J_u S J_uᵀ alone
    clamp = _driven("first_order")(X_BELIEF, U_BELIEF.loc)
    np.testing.assert_allclose(
        np.asarray(got.cov - clamp.cov), J_u @ Su @ J_u.T, rtol=1e-4, atol=1e-6
    )


def test_unscented_matches_monte_carlo_over_state_and_control():
    """The sigma points are drawn over the joint ``(x, u)``: the moments agree with samples
    of both."""
    rng = np.random.default_rng(1)
    n = 400_000
    x = rng.multivariate_normal(np.asarray(X_BELIEF.loc), np.asarray(X_BELIEF.cov), size=n)
    u = rng.multivariate_normal(np.asarray(U_BELIEF.loc), np.asarray(U_BELIEF.cov), size=n)
    h = 0.5
    y = np.stack(
        [
            x[:, 0] + h * u[:, 0] * np.cos(x[:, 2]),
            x[:, 1] + h * u[:, 0] * np.sin(x[:, 2]),
            x[:, 2] + h * u[:, 1],
        ],
        1,
    ) + 0.01 * rng.standard_normal((n, 3))
    got = _driven("unscented")(X_BELIEF, U_BELIEF)
    np.testing.assert_allclose(np.asarray(got.loc), y.mean(0), atol=1e-3)
    np.testing.assert_allclose(np.asarray(got.cov), np.cov(y.T), atol=3e-3)
    #  first order misses the mean the heading's spread pulls in (1.6 cm here)
    first = _driven("first_order")(X_BELIEF, U_BELIEF)
    assert float(np.max(np.abs(np.asarray(first.loc) - y.mean(0)))) > 1e-2


def test_a_truncated_control_belief_enters_through_its_moments():
    from hybrid_aif import TruncatedGaussian

    truncated = TruncatedGaussian(
        loc=jnp.asarray([0.8, 0.5]), scale=jnp.asarray([0.4, 0.4]), low=-0.2, high=1.0
    )
    moments = Gaussian.from_cov(truncated.mean(), truncated.cov)
    for propagation in ("first_order", "unscented"):
        step = _driven(propagation)
        _close(step(X_BELIEF, truncated), step(X_BELIEF, moments), rtol=1e-4, atol=1e-6)


# --------------------------------------- N6 (5): integer levels, every planner, the posterior


@pytest.mark.parametrize("planner", ["enumeration", "sophisticated"])
def test_discrete_levels_reach_f_as_integers_under_each_planner(planner):
    """``SteeredUnicycle`` asserts an integer level at trace time; enumeration and the
    sophisticated recursion both plan the left turn."""
    system, params = _steered(_steered_nonlinear(), policy_len=2, planner=planner)
    beliefs = system.initial_priors(params)
    q_pi, _ = system.infer_policies(params, beliefs)[:2]
    assert np.all(np.isfinite(np.asarray(q_pi.probs)))
    action = system.act(KEY, params, beliefs, selection="deterministic")[0]
    assert int(action["a_pose"]) == 2


def test_planning_belief_posterior_steps_the_epistemic_chain_through_f():
    """``planning_belief="posterior"``: the epistemic member is predicted through ``f`` from
    the conditioned one, so a position channel shrinks the heading-driven spread the next step
    reads; the information-gain leaf changes and stays finite."""
    obs = {
        "gps": LinearGaussianObservation(H=jnp.eye(3)[:2], c=jnp.zeros(2), noise=0.1 * jnp.eye(2))
    }
    over = dict(A_deps=((0,),), modality_names=("gps",), epistemic=True, policy_len=3)
    trees = {}
    for belief in ("prior", "posterior"):
        system, params = _steered(_steered_nonlinear(), planning_belief=belief, **over)
        params = dataclasses.replace(params, likelihood=obs)
        beliefs = system.initial_priors(params)
        tree = system.score_policies(params, beliefs, system.policies(params))
        assert all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree_util.tree_leaves(tree))
        trees[belief] = np.asarray(tree["infogain"]["gps"].midpoint)  # channel-keyed (V11)
        ctx = system.planner_context(params, beliefs)
        if belief == "posterior":
            (pair,) = ctx.init_beliefs
            assert isinstance(pair, _hybrid.BeliefPair)
            (one,) = ctx.transition_fn(ctx.init_beliefs, jnp.asarray([2], jnp.int32))
            _close(one.epistemic, one.predictive)  # nothing was observed before the first step
            assert float(jnp.trace(one.conditioned.cov)) < float(jnp.trace(one.predictive.cov))
            (two,) = ctx.transition_fn((one,), jnp.asarray([2], jnp.int32))
            assert float(jnp.trace(two.epistemic.cov)) < float(jnp.trace(two.predictive.cov))
    assert not np.allclose(trees["prior"], trees["posterior"])


# ------------------------------------------------------------------- N6 (6): the refusals


def test_switching_dynamics_are_refused():
    system = System(B_deps=((1,), ()), factor_names=("pose", "mode"), policy_len=1)
    params = SystemParams(
        transition={
            "pose": NonlinearGaussianTransition(
                Unicycle(0.1, 0.5, 0.0), _noise(jnp.zeros((2, 3)), 0.01 * jnp.eye(3))
            ),
            "mode": ConditionalCategorical(log_table=jnp.zeros((2, 2)), parent_ndims=1),
        },
        initial_state_prior={
            "pose": Gaussian(loc=jnp.zeros(3), chol=jnp.eye(3)),
            "mode": Categorical(logits=jnp.zeros(2)),
        },
        state_targets={},
    )
    with pytest.raises(NotImplementedError, match="switching dynamics"):
        system.update_empirical_prior(params, system.initial_priors(params), {})
    # with a control node of its own too, the switching refusal is the one raised
    controlled = System(
        B_deps=((1,), ()), controlled=(True, False), factor_names=("pose", "mode"), policy_len=1
    )
    with pytest.raises(NotImplementedError, match="switching dynamics"):
        controlled.update_empirical_prior(params, controlled.initial_priors(params), {})


def test_a_neural_transition_steps_the_epistemic_chain():
    """Under ``planning_belief="posterior"`` a neural transition steps both beliefs of the
    pair, as the linear and nonlinear ones do: targets read the predictive belief, so risk is
    unchanged, and the channel's information gain reads the conditioned chain, so it is smaller
    once an expected reading has tightened the belief."""
    from tests.test_cem import lq_system

    trees = {}
    for belief in ("prior", "posterior"):
        system, params = lq_system(
            planning_belief=belief, epistemic=("x",), A_deps=((0,),), modality_names=("y",)
        )
        params = dataclasses.replace(
            params,
            likelihood={
                "y": LinearGaussianObservation(H=jnp.eye(1), c=jnp.zeros(1), noise=0.1 * jnp.eye(1))
            },
        )
        plan = jnp.asarray([[[0.3], [0.2], [0.1]]])
        trees[belief] = system.score_policies(params, system.initial_priors(params), plan)
        assert np.all(np.isfinite(np.asarray(total(trees[belief]).midpoint)))
    assert np.array_equal(
        np.asarray(trees["prior"]["risk"]["x"].lower),
        np.asarray(trees["posterior"]["risk"]["x"].lower),
    )
    assert np.all(
        np.asarray(trees["posterior"]["infogain"]["y"].lower)
        < np.asarray(trees["prior"]["infogain"]["y"].lower)
    )


class Flat(eqx.Module):
    def __call__(self, x, controls):
        return x[:2]


class NeedsTwo(eqx.Module):
    def __call__(self, x, controls):
        a, b = controls
        return x


def test_a_map_of_the_wrong_shape_or_layout_is_refused():
    system = System(B_deps=((),), factor_names=("pose",), policy_len=1)

    def params(f):
        return SystemParams(
            transition={
                "pose": NonlinearGaussianTransition(f, _noise(jnp.zeros((1, 3)), jnp.eye(3)))
            },
            initial_state_prior={"pose": Gaussian(loc=jnp.zeros(3), chol=jnp.eye(3))},
        )

    with pytest.raises(ValueError, match=r"f returns float32\[2\].*must return .*float\[3\]"):
        system._validate(params(Flat()))
    with pytest.raises(ValueError, match=r"f fails on the factor's layout, f\(x: float\[3\]"):
        system._validate(params(NeedsTwo()))
    with pytest.raises(TypeError, match="callable equinox.Module"):
        NonlinearGaussianTransition(lambda x, c: x, _noise(jnp.zeros((1, 3)), jnp.eye(3)))
    with pytest.raises(ValueError, match="propagation='sigma'"):
        NonlinearGaussianTransition(Flat(), _noise(jnp.zeros((1, 3)), jnp.eye(3)), "sigma")
    #  the innovations carry one axis per discrete control node
    controlled = System(B_deps=((),), controlled=(True,), factor_names=("pose",), policy_len=1)
    bad = dataclasses.replace(
        params(Unicycle(0.1, 0.5, 0.0)),
        transition={
            "pose": NonlinearGaussianTransition(
                SteeredUnicycle(0.1, 0.5, RATES), _noise(jnp.zeros((3, 3)), jnp.eye(3))
            ),
        },
        initial_state_prior={"pose": Gaussian(loc=jnp.zeros(2), chol=jnp.eye(2))},
    )
    with pytest.raises(ValueError, match="the prior .D. is 2-dimensional"):
        controlled._validate(bad)


def test_zero_process_noise_is_refused():
    noiseless = NonlinearGaussianTransition(
        Unicycle(0.1, 0.5, 0.0), _noise(jnp.zeros((1, 3)), jnp.diag(jnp.asarray([0.1, 0.1, 0.0])))
    )
    system = System(B_deps=((),), factor_names=("pose",), policy_len=1)
    params = SystemParams(
        transition={"pose": noiseless},
        initial_state_prior={"pose": Gaussian(loc=jnp.zeros(3), chol=jnp.eye(3))},
    )
    beliefs = system.initial_priors(params)
    with pytest.raises(ValueError, match="zero on its Cholesky factor's diagonal"):
        system.update_empirical_prior(params, beliefs, {})
    with pytest.raises(Exception, match="zero on its Cholesky factor's diagonal"):
        jax.jit(lambda p: system.update_empirical_prior(p, beliefs, {}))(params)
    #  a generative process may be noiseless
    env, env_params = system.as_env(params)
    assert env is not None and env_params is not None


def test_describe_refuses_naming_the_registry(monkeypatch):
    from hybrid_aif import description as _description

    monkeypatch.setattr(_description, "_TRANSITIONS", {})
    system, params = _steered(_steered_nonlinear())
    with pytest.raises(DescriptionError, match="register_transition"):
        describe(system, params)


def test_merge_carries_the_declaration_and_refuses_slicing_its_inputs():
    robot, rp = _steered(_steered_nonlinear(), policy_len=1)
    twin, tp = _steered(_steered_nonlinear(), policy_len=1)
    composite, cp = merge({"robot": robot, "twin": twin}, {"robot": rp, "twin": tp})
    assert isinstance(cp.transition["robot.pose"], NonlinearGaussianTransition)
    beliefs = composite.initial_priors(cp)
    one = composite.update_empirical_prior(
        cp, beliefs, {"robot.a_pose": jnp.int32(2), "twin.a_pose": jnp.int32(0)}
    )
    alone = robot.update_empirical_prior(rp, robot.initial_priors(rp), {"a_pose": jnp.int32(2)})
    _close(one["robot.pose"], alone["pose"])
    #  identifying two control nodes keeps f's inputs whole
    same, sp = merge(
        {"robot": robot, "twin": twin},
        {"robot": rp, "twin": tp},
        [ActionControl("robot.u_pose", "twin.u_pose")],
    )
    assert same.names.controls == ("robot.u_pose",)
    #  terminating the node f reads would slice its inputs
    with pytest.raises(NotImplementedError, match="f's signature"):
        merge(
            {"robot": robot, "twin": twin},
            {"robot": rp, "twin": tp},
            [BoundaryPrior("twin.u_pose", 1)],
        )


# --------------------------------------------------------------- N6 (7): the surface


def test_as_env_samples_through_f():
    system, params = _steered(
        NonlinearGaussianTransition(
            SteeredUnicycle(PLAN_H, PLAN_V, RATES), _noise(jnp.zeros((3, 3)), 1e-6 * jnp.eye(3))
        )
    )
    env, env_params = system.as_env(params)
    start = {"pose": jnp.asarray([0.3, -0.2, 0.5])}
    _, state = env.reset(KEY, env_params, state=start)
    _, nxt = env.step(KEY, state, {"a_pose": jnp.int32(2)}, env_params)
    want = SteeredUnicycle(PLAN_H, PLAN_V, RATES)(start["pose"], (jnp.int32(2),))
    np.testing.assert_allclose(np.asarray(nxt["pose"]), np.asarray(want), atol=1e-4)


class Tabled(eqx.Module):
    """A map with float fields (a scalar and a table), an integer array and a static field."""

    h: float
    gains: jax.Array
    index: jax.Array
    name: str = eqx.field(static=True, default="tabled")

    def __call__(self, x, controls):
        return x + self.h * self.gains[self.index]


def test_description_length_counts_the_maps_fields():
    f = Tabled(0.1, jnp.ones((4, 3)), jnp.asarray(2, jnp.int32))
    system = System(B_deps=((),), factor_names=("pose",), policy_len=1)
    params = SystemParams(
        transition={"pose": NonlinearGaussianTransition(f, _noise(jnp.zeros((1, 3)), jnp.eye(3)))},
        initial_state_prior={"pose": Gaussian(loc=jnp.zeros(3), chol=jnp.eye(3))},
    )
    n = 50.0
    link = description_length(system, n, params=params).per_address["transition:pose"]
    k = (1 + 12) + 3 + 6  # f's floats, the offset, the noise's triangle
    assert link.family == pytest.approx(resolution_term(k, n))
    assert not link.exact and "13 parameters" in link.note


def test_learn_and_carry_leave_f_alone():
    decl = _steered_nonlinear()
    assert learnable_blocks(decl) == ()
    obs = {"gps": LinearGaussianObservation(H=jnp.eye(3)[:2], c=jnp.zeros(2), noise=jnp.eye(2))}
    system, params = _steered(decl, A_deps=((0,),), modality_names=("gps",))
    params = dataclasses.replace(params, likelihood=obs)
    beliefs = system.initial_priors(params)
    posterior, _ = system.infer_states(params, {"gps": jnp.asarray([0.1, 0.0])}, priors=beliefs)
    learned = system.learn(
        params,
        {"gps": jnp.asarray([0.1, 0.0])},
        posterior,
        prev_beliefs=beliefs,
        action={"a_pose": jnp.int32(1)},
    )
    _close(learned.transition["pose"], decl)
    moved = dataclasses.replace(
        params,
        transition={"pose": eqx.tree_at(lambda d: d.f.v, decl, 9.0)},
    )
    carried = system.carry(params, moved)
    assert float(carried.transition["pose"].f.v) == PLAN_V
    with pytest.raises(ValueError, match="NonlinearGaussianTransition .a point value."):
        system.learn(
            params,
            {"gps": jnp.asarray([0.1, 0.0])},
            posterior,
            lr_by_slot={"transition:pose": 0.5},
        )


def test_replay_runs_through_f_and_prices_with_its_data():
    """A recorded trace replays through the nonlinear transition (the prior each tick is the
    previous posterior pushed through ``f``), and its free energy is the data part of the
    description length."""
    from hybrid_aif import replay

    obs = {
        "gps": LinearGaussianObservation(H=jnp.eye(3)[:2], c=jnp.zeros(2), noise=0.1 * jnp.eye(2))
    }
    system, params = _steered(_steered_nonlinear(), A_deps=((0,),), modality_names=("gps",))
    params = dataclasses.replace(params, likelihood=obs)
    env, env_params = system.as_env(params)
    _, state = env.reset(KEY, env_params, state={"pose": jnp.zeros(3)})
    levels, seen = [2, 2, 1, 0, 2], []
    for t, level in enumerate(levels):
        o, state = env.step(
            jax.random.fold_in(KEY, t), state, {"a_pose": jnp.int32(level)}, env_params
        )
        seen.append(o["gps"])
    trace = replay(
        system,
        params,
        {"gps": jnp.stack(seen)},
        {"a_pose": jnp.asarray(levels, jnp.int32)},
        metrics=False,
    )
    assert np.all(np.isfinite(np.asarray(trace.free_energy)))
    length = description_length(system, params=params, replay=trace)
    assert np.isfinite(length.total)
