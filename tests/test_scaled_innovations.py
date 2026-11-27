"""A nonlinear transition whose process noise depends on the state and the controls.

Under ``noise_scale="per_coordinate"`` a :class:`~hybrid_aif.factors.NonlinearGaussianTransition`'s
map returns ``(mean, scale)`` and the innovations' Cholesky factor is ``diag(scale) chol``. These
tests pin: a scale of ones is the plain map (inference, planning, sampling, the document); the
predicted covariance against a closed form (a scale affine in the state) and against Monte Carlo,
first order and unscented; sampling at the sampled state; a filter calibrated where a constant
noise is not; gradients through the scale; jit and vmap; the document; the refusals; and the
neural transition's step, which goes through the same propagation.

Design: docs/design/nonlinear-transitions.md, amendment of 2026-11-13.
"""

import dataclasses
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import (
    Box,
    Categorical,
    ConditionalCategorical,
    DescriptionError,
    Gaussian,
    LinearGaussianObservation,
    MixtureGaussian,
    NonlinearGaussianTransition,
    System,
    SystemParams,
    build,
    describe,
    register_transition,
)
from hybrid_aif import description as _description
from hybrid_aif.composition import Clamp, merge
from hybrid_aif.controls import ControlStep
from hybrid_aif.description.mdl import description_length
from hybrid_aif.distributions import MLPConditional
from hybrid_aif.objectives import hybrid as _hybrid
from hybrid_aif.objectives import total
from tests.test_nonlinear_transition import (
    F_MODES,
    G_U,
    _affine_pair,
    _close,
    _noise,
)

KEY = jax.random.PRNGKey(0)

# --------------------------------------------------------------------------- the maps


class AffineOnes(eqx.Module):
    """``F_m x + G u``, as the plain map or with a scale of ones."""

    F: jax.Array
    G: jax.Array
    scaled: bool = eqx.field(static=True, default=True)

    def __call__(self, x, controls):
        level, u = controls
        mean = self.F[level] @ x + self.G @ u
        return (mean, jnp.ones_like(mean)) if self.scaled else mean


class AffineScale(eqx.Module):
    """``F x + b`` with the scale ``c + B x``, affine in the state: the closed-form case."""

    F: jax.Array
    b: jax.Array
    c: jax.Array
    B: jax.Array

    def __call__(self, x, controls):
        del controls
        return self.F @ x + self.b, self.c + self.B @ x


class SpeedScale(eqx.Module):
    """``(position, speed)`` driven by a commanded speed ``u``: ``p' = p + h v``, ``v' = a v +
    (1 − a) u``, the noise on both growing with the speed, ``s = sqrt(1 + (k v)²)``, and the
    speed's also with the command, ``sqrt(1 + (k v)² + (k u)²)``."""

    h: float
    a: float
    k: float

    def __call__(self, x, controls):
        (u,) = controls
        p, v = x[0], x[1]
        mean = jnp.stack([p + self.h * v, self.a * v + (1.0 - self.a) * u[0]])
        base = 1.0 + (self.k * v) ** 2
        return mean, jnp.sqrt(jnp.stack([base, base + (self.k * u[0]) ** 2]))


class SpeedPlain(eqx.Module):
    """:class:`SpeedScale`'s mean alone: the constant-noise model."""

    h: float
    a: float

    def __call__(self, x, controls):
        (u,) = controls
        return jnp.stack([x[0] + self.h * x[1], self.a * x[1] + (1.0 - self.a) * u[0]])


class Unicycle(eqx.Module):
    """A unicycle driven by ``(speed, turn rate)``, its position noise growing with the
    commanded speed and its heading noise with the heading's sine."""

    h: float
    k: float

    def __call__(self, x, controls):
        (u,) = controls
        mean = jnp.stack(
            [
                x[0] + self.h * u[0] * jnp.cos(x[2]),
                x[1] + self.h * u[0] * jnp.sin(x[2]),
                x[2] + self.h * u[1],
            ]
        )
        sxy = jnp.sqrt(1.0 + (self.k * u[0]) ** 2)
        scale = jnp.stack([sxy, sxy, 1.0 + 0.5 * jnp.sin(x[2]) ** 2])
        return mean, scale


class Pair(eqx.Module):
    """Returns a pair whatever it is declared as."""

    def __call__(self, x, controls):
        return x, jnp.ones_like(x)


class Mean(eqx.Module):
    def __call__(self, x, controls):
        return x


class Short(eqx.Module):
    def __call__(self, x, controls):
        return x, jnp.ones(x.shape[0] - 1)


def _scaled(f, chol, propagation="first_order", loc=None):
    d = chol.shape[-1]
    loc = jnp.zeros((1, d)) if loc is None else loc
    return NonlinearGaussianTransition(f, _noise(loc, chol), propagation, "per_coordinate")


# ------------------------------------------------------------- a scale of ones is the plain map


def _ones_pair(carry: str, propagation: str):
    """``_affine_pair``'s System with the plain affine map and the same map with a scale of
    ones."""
    system, _, nl = _affine_pair(carry, propagation)
    plain = dataclasses.replace(
        nl,
        transition={
            "x": dataclasses.replace(nl.transition["x"], f=AffineOnes(F_MODES, G_U, False))
        },
    )
    ones = dataclasses.replace(
        nl,
        transition={
            "x": dataclasses.replace(
                nl.transition["x"], f=AffineOnes(F_MODES, G_U), noise_scale="per_coordinate"
            )
        },
    )
    return system, plain, ones


def _exact(a, b) -> None:
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb, strict=True):
        assert np.array_equal(np.asarray(x), np.asarray(y))


@pytest.mark.parametrize("propagation", ["first_order", "unscented"])
@pytest.mark.parametrize("carry", ["gaussian", "mixture", "posterior"])
def test_a_scale_of_ones_is_the_plain_map(carry, propagation):
    """Prediction, inference and every leaf of the term tree under clamped and belief-valued
    controls: bit for bit to first order (the noise is ``Q ∘ 1``), to rounding unscented (the
    weights' sum of ones)."""
    system, plain, ones = _ones_pair(carry, propagation)
    same = _exact if propagation == "first_order" else lambda a, b: _close(a, b, 1e-6, 1e-7)
    beliefs = system.initial_priors(plain)
    action = {"m": jnp.int32(1), "u": jnp.asarray([0.3])}
    prior_plain = system.update_empirical_prior(plain, beliefs, action)
    prior_ones = system.update_empirical_prior(ones, beliefs, action)
    same(prior_ones, prior_plain)
    if carry != "mixture":  # filtering a mixture needs mode_aligned slots, refused here
        obs = {"y": jnp.asarray([0.7])}
        same(
            system.infer_states(ones, obs, priors=prior_ones)[0],
            system.infer_states(plain, obs, priors=prior_plain)[0],
        )
    levels = jnp.asarray([[0, 1, 1], [1, 0, 1]], jnp.int32)
    values = jnp.asarray([[[0.3], [-0.5], [0.9]], [[0.0], [0.4], [-0.2]]])
    for given in (values, Gaussian(loc=values, chol=jnp.full((2, 3, 1, 1), 0.3))):
        policies = {"u_m": levels, "u_u": given}
        same(
            system.score_policies(ones, beliefs, policies),
            system.score_policies(plain, beliefs, policies),
        )


def test_a_scale_of_ones_samples_the_plain_draws():
    """``as_env`` draws the same noise from the same key."""
    system, plain, ones = _ones_pair("gaussian", "first_order")
    start = {"x": jnp.asarray([0.4, -0.2])}
    action = {"m": jnp.int32(1), "u": jnp.asarray([0.3])}
    draws = []
    for params in (plain, ones):
        env, env_params = system.as_env(params)
        _, state = env.reset(KEY, env_params, state=start)
        draws.append(
            [env.step(jax.random.PRNGKey(t), state, action, env_params)[1]["x"] for t in range(5)]
        )
    _exact(draws[0], draws[1])


# ----------------------------------------------- the predicted covariance: a closed form


F2 = jnp.asarray([[1.0, 0.2], [0.0, 0.9]])
B_OFF = jnp.asarray([0.1, -0.05])
C2 = jnp.asarray([1.0, 1.5])
BS = jnp.asarray([[0.6, 0.0], [0.3, 0.8]])
Q2 = jnp.asarray([[0.2, 0.0], [0.15, 0.1]])
MU2 = jnp.asarray([0.5, 0.4])
P2 = jnp.asarray([[0.3, 0.0], [0.1, 0.25]])


def _one_step(decl, mu, chol, u=None):
    """One prediction of a single uncontrolled (or ``u``-controlled) factor."""
    controlled = u is not None
    width = (u.loc if isinstance(u, Gaussian) else jnp.asarray(u)).shape[-1] if controlled else 0
    system = System(
        B_deps=((1,), ()) if controlled else ((),),
        controls=(1,) if controlled else (),
        action_names=("u",) if controlled else (),
        state_names=("x",),
        control_names=("u_u",) if controlled else (),
        policy_len=1,
        epistemic=False,
    )
    params = SystemParams(
        transition={"x": decl},
        initial_state_prior={"x": Gaussian(loc=mu, chol=chol)},
        control={"u_u": Box(-10.0 * jnp.ones(width), 10.0 * jnp.ones(width))} if controlled else {},
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
    q = Gaussian(loc=mu, chol=chol)
    if not controlled:
        return step((q,), jnp.zeros((0,), jnp.int32))[0]
    return step((q,), ControlStep(entries=(u,)))[0]


def test_a_scale_affine_in_the_state_against_its_closed_form():
    """``x' = F x + b + diag(c + B x) L ε``: the predictive covariance is ``F P Fᵀ + Q ∘ (s̄ s̄ᵀ
    + B P Bᵀ)``, ``s̄ = c + B μ``. The unscented rule is exact here (the scale's second moment is
    quadratic); first order reads the scale at the mean and drops ``Q ∘ B P Bᵀ``, exactly."""
    m, P = np.asarray(MU2, np.float64), np.asarray(P2 @ P2.T, np.float64)
    F, Q = np.asarray(F2, np.float64), np.asarray(Q2 @ Q2.T, np.float64)
    c, B = np.asarray(C2, np.float64), np.asarray(BS, np.float64)
    s_bar = c + B @ m
    want_mean = F @ m + np.asarray(B_OFF, np.float64)
    want_cov = F @ P @ F.T + Q * (np.outer(s_bar, s_bar) + B @ P @ B.T)
    f = AffineScale(F2, B_OFF, C2, BS)
    sigma = _one_step(_scaled(f, Q2, "unscented"), MU2, P2)
    np.testing.assert_allclose(np.asarray(sigma.loc), want_mean, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(sigma.cov), want_cov, rtol=1e-5, atol=1e-6)
    first = _one_step(_scaled(f, Q2, "first_order"), MU2, P2)
    np.testing.assert_allclose(np.asarray(first.loc), want_mean, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(first.cov), want_cov - Q * (B @ P @ B.T), rtol=1e-5, atol=1e-6
    )
    #  the expected covariance, not the covariance at the expected scale: they differ here
    assert np.max(np.abs(Q * (B @ P @ B.T))) > 1e-3


# ----------------------------------------------- the predicted covariance: Monte Carlo

X3 = Gaussian(
    loc=jnp.asarray([0.2, -0.1, 0.6]),
    chol=jnp.asarray([[0.1, 0.0, 0.0], [0.02, 0.1, 0.0], [0.0, 0.0, 0.4]]),
)
U3 = Gaussian(loc=jnp.asarray([1.2, 0.5]), chol=jnp.asarray([[0.4, 0.0], [0.05, 0.3]]))
Q3 = jnp.asarray([[0.05, 0.0, 0.0], [0.03, 0.04, 0.0], [0.0, 0.0, 0.03]])
UNI_H, UNI_K = 0.5, 1.5
SPEED_X = Gaussian(loc=jnp.asarray([0.3, 1.0]), chol=jnp.asarray([[0.2, 0.0], [0.1, 0.5]]))
SPEED_U = Gaussian(loc=jnp.asarray([1.0]), chol=jnp.asarray([[0.4]]))
SPEED_Q = jnp.asarray([[0.05, 0.0], [0.04, 0.06]])


def _joint(x_belief, u_belief, n, seed):
    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal(np.asarray(x_belief.loc), np.asarray(x_belief.cov), size=n)
    u = rng.multivariate_normal(np.asarray(u_belief.loc), np.asarray(u_belief.cov), size=n)
    return rng, x, u


def _moments(y):
    """The sample mean and covariance of ``y`` and the covariance's standard error."""
    mean = y.mean(0)
    dev = y - mean
    prods = dev[:, :, None] * dev[:, None, :]
    return mean, np.cov(y.T), np.sqrt(np.var(prods, axis=0) / len(y))


def test_the_predictive_against_monte_carlo_over_state_and_control():
    """:class:`SpeedScale` (a mean affine in its inputs, a scale that is not) under a belief
    over the state and over the control (the gradient planner's case): the unscented moments
    agree with 400,000 draws whose noise is drawn at each draw's own state and control, within
    four standard errors; first order, which reads the scale at the mean, is short of the
    speed's variance by ``Q ∘ Cov(s)``, more than ten standard errors."""
    k, h, a = 2.0, 0.1, 0.8
    rng, x, u = _joint(SPEED_X, SPEED_U, 400_000, 2)
    mean = np.stack([x[:, 0] + h * x[:, 1], a * x[:, 1] + (1 - a) * u[:, 0]], 1)
    base = 1.0 + (k * x[:, 1]) ** 2
    scale = np.sqrt(np.stack([base, base + (k * u[:, 0]) ** 2], 1))
    y = mean + scale * (rng.standard_normal((len(x), 2)) @ np.asarray(SPEED_Q, np.float64).T)
    mc_mean, mc_cov, se = _moments(y)
    f = SpeedScale(h, a, k)
    sigma = _one_step(_scaled(f, SPEED_Q, "unscented"), SPEED_X.loc, SPEED_X.chol, SPEED_U)
    first = _one_step(_scaled(f, SPEED_Q, "first_order"), SPEED_X.loc, SPEED_X.chol, SPEED_U)
    np.testing.assert_allclose(
        np.asarray(sigma.loc), mc_mean, atol=4 * np.sqrt(np.diag(mc_cov) / len(y)).max()
    )
    assert np.all(np.abs(np.asarray(sigma.cov) - mc_cov) < 4 * se), (
        np.asarray(sigma.cov),
        mc_cov,
        se,
    )
    short = mc_cov[1, 1] - float(first.cov[1, 1])
    assert short > 10 * se[1, 1], (short, se[1, 1])


def test_the_noise_term_against_monte_carlo_on_the_unicycle():
    """The noise term alone, ``E[diag s Q diag s] = Q ∘ E[s sᵀ]``, of :class:`Unicycle` (the
    position's scale grows with an uncertain commanded speed, the heading's with the heading's
    sine) against its sample average over 400,000 draws of the state and the control: the
    unscented estimate within 1.5 % on every entry (0.2 % on the position, 1 % on the heading,
    whose scale is a squared sine), first order (the scale at the mean) short by
    ``Q ∘ Cov(s)``, several percent on the position."""
    rng, x, u = _joint(X3, U3, 400_000, 3)
    sxy = np.sqrt(1.0 + (UNI_K * u[:, 0]) ** 2)
    scale = np.stack([sxy, sxy, 1.0 + 0.5 * np.sin(x[:, 2]) ** 2], 1)
    Q = np.asarray(Q3 @ Q3.T, np.float64)
    want = Q * (scale.T @ scale / len(scale))
    f = Unicycle(UNI_H, UNI_K)

    def noise_term(propagation):
        scaled = _one_step(_scaled(f, Q3, propagation), X3.loc, X3.chol, U3)
        plain = NonlinearGaussianTransition(_MeanOf(f), _noise(jnp.zeros((1, 3)), Q3), propagation)
        spread = _one_step(plain, X3.loc, X3.chol, U3).cov - Q3 @ Q3.T
        return np.asarray(scaled.cov - spread, np.float64)

    np.testing.assert_allclose(noise_term("unscented"), want, rtol=1.5e-2, atol=1e-7)
    first = noise_term("first_order")
    assert np.all(np.diag(want - first)[:2] > 0.03 * np.diag(want)[:2])


class _MeanOf(eqx.Module):
    """A scaled map's mean alone."""

    inner: eqx.Module

    def __call__(self, x, controls):
        return self.inner(x, controls)[0]


# ------------------------------------------------------------- sampling at the sampled state


def test_as_env_draws_the_scale_at_the_sampled_state():
    """From starts drawn from the prior, the step's noise standardised by each start's own scale
    is ``N(0, I)``; standardised by the scale at the prior's mean it is not."""
    f = SpeedScale(0.1, 0.8, 2.0)
    chol = jnp.asarray([[0.05, 0.0], [0.02, 0.08]])
    system = System(
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("u",),
        state_names=("x",),
        control_names=("u_u",),
        policy_len=1,
    )
    prior = Gaussian(loc=jnp.asarray([0.0, 1.0]), chol=jnp.diag(jnp.asarray([0.5, 1.0])))
    params = SystemParams(
        transition={"x": _scaled(f, chol, loc=jnp.asarray([[0.01, -0.02]]))},
        initial_state_prior={"x": prior},
        control={"u_u": Box(-3.0 * jnp.ones(1), 3.0 * jnp.ones(1))},
    )
    env, env_params = system.as_env(params)
    action = {"u": jnp.asarray([0.5])}
    n = 20_000

    def draw(key):
        k0, k1 = jax.random.split(key)
        _, state = env.reset(k0, env_params)
        _, nxt = env.step(k1, state, action, env_params)
        return state["x"], nxt["x"]

    x0, x1 = jax.jit(jax.vmap(draw))(jax.random.split(KEY, n))
    mean, scale = jax.vmap(lambda x: f(x, (action["u"],)))(x0)
    _, at_mean = f(prior.loc, (action["u"],))
    L = np.asarray(chol, np.float64)
    offset = np.asarray([0.01, -0.02])

    def standardised(s):
        resid = (np.asarray(x1 - mean, np.float64) - offset) / np.asarray(s, np.float64)
        return np.linalg.solve(L, resid.T).T

    z = standardised(scale)
    np.testing.assert_allclose(z.mean(0), 0.0, atol=0.03)
    np.testing.assert_allclose(np.cov(z.T), np.eye(2), atol=0.04)
    wrong = np.cov(standardised(jnp.broadcast_to(at_mean, scale.shape)).T)
    assert np.max(np.abs(wrong - np.eye(2))) > 0.3


# ------------------------------------------------- a calibrated filter, noise growing with speed

SPEEDS = (0.0, 1.0, 2.0, 3.0)  # the commanded speed of each band
FILTER_TICKS, FILTER_EPISODES = 60, 64
SPEED_CHOL = jnp.asarray([[0.02, 0.0], [0.0, 0.1]])


def _speed_model(f, chol, **over):
    wiring = dict(
        A_deps=((0,),),
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("u",),
        state_names=("x",),
        control_names=("u_u",),
        modality_names=("pos",),
        policy_len=1,
        epistemic=False,
    )
    system = System(**{**wiring, **over})
    params = SystemParams(
        likelihood={
            "pos": LinearGaussianObservation(
                H=jnp.asarray([[1.0, 0.0]]), c=jnp.zeros(1), noise=jnp.asarray([[0.05]])
            )
        },
        transition={
            "x": f
            if chol is None
            else NonlinearGaussianTransition(f, _noise(jnp.zeros((1, 2)), chol))
        },
        initial_state_prior={
            "x": Gaussian(loc=jnp.zeros(2), chol=jnp.diag(jnp.asarray([0.1, 0.5])))
        },
        control={"u_u": Box(-5.0 * jnp.ones(1), 5.0 * jnp.ones(1))},
    )
    return system, params


def _coverage():
    """Per commanded speed, the share of ticks whose true speed lies in the filter's 90 %
    interval, for the scaled model (the world's own) and the constant-noise model whose noise is
    the scaled one's at the middle speed, 1.5."""
    f = SpeedScale(0.1, 0.8, 2.0)
    true_system, true_params = _speed_model(_scaled(f, SPEED_CHOL), None)
    s_mid = np.asarray(f(jnp.asarray([0.0, 1.5]), (jnp.asarray([1.5]),))[1])
    constant_chol = jnp.diag(jnp.asarray(s_mid, jnp.float32)) @ SPEED_CHOL
    models = {
        "scaled": (true_system, true_params),
        "constant": _speed_model(SpeedPlain(0.1, 0.8), constant_chol),
    }
    env, env_params = true_system.as_env(true_params)
    start_chol = jnp.diag(jnp.asarray([0.1, 0.5]))

    def episode(key, speed):
        """A world run at one commanded speed: true states and position readings."""
        action = {"u": speed[None]}
        _, state = env.reset(key, env_params, state={"x": jnp.stack([0.0, speed])})

        def tick(state, t):
            obs, state = env.step(jax.random.fold_in(key, t), state, action, env_params)
            return state, (state["x"], obs["pos"])

        return jax.lax.scan(tick, state, jnp.arange(FILTER_TICKS))[1]

    def filt(system, params, seen, speed):
        """The filter's posterior means and standard deviations over a run."""
        action = {"u": speed[None]}

        def tick(beliefs, reading):
            prior = system.update_empirical_prior(params, beliefs, action)
            beliefs = system.infer_states(params, {"pos": reading}, priors=prior)[0]
            return beliefs, (beliefs["x"].loc, jnp.sqrt(jnp.diag(beliefs["x"].cov)))

        start = {"x": Gaussian(loc=jnp.stack([0.0, speed]), chol=start_chol)}
        return jax.lax.scan(tick, start, seen)[1]

    runs = jax.jit(jax.vmap(episode))
    filters = {
        name: jax.jit(jax.vmap(lambda seen, speed, s=system, p=params: filt(s, p, seen, speed)))
        for name, (system, params) in models.items()
    }
    out = {}
    for band, speed in enumerate(SPEEDS):
        keys = jax.random.split(jax.random.fold_in(KEY, band), FILTER_EPISODES)
        speeds = jnp.full((FILTER_EPISODES,), speed)
        truth, seen = runs(keys, speeds)
        for name, run in filters.items():
            locs, sds = run(seen, speeds)
            err = np.abs(np.asarray(truth - locs))[:, 10:, 1]  # speed, past the transient
            out[(name, speed)] = float(np.mean(err < 1.645 * np.asarray(sds)[:, 10:, 1]))
    return out


@pytest.fixture(scope="module")
def coverage():
    return _coverage()


def test_a_filter_with_speed_noise_is_calibrated(coverage):
    """The world's speed noise grows with the speed and the commanded speed. The filter with
    the scaled model covers the true speed about 90 % of the time at every speed; the
    constant-noise model, right only at the middle speed, over-covers at rest and under-covers
    at speed."""
    for speed in SPEEDS:
        assert 0.86 <= coverage[("scaled", speed)] <= 0.94, (speed, coverage)
    assert coverage[("constant", 0.0)] > 0.97, coverage
    assert coverage[("constant", 3.0)] < 0.80, coverage


# ---------------------------------------------------------------- gradients, jit and vmap


def test_gradients_flow_through_the_scale():
    """The predicted covariance's gradient in the belief's mean, the control and the map's
    leaf ``k`` agree with central differences (float64 arithmetic on float32 inputs is not
    available, so the step is wide); with a scale that does not vary they vanish where only the
    scale depends on the input."""
    x_chol = jnp.diag(jnp.asarray([0.1, 0.3]))

    def spread(mu, u, k, propagation):
        q = _one_step(_scaled(SpeedScale(0.1, 0.8, k), SPEED_CHOL, propagation), mu, x_chol, u)
        return jnp.trace(q.cov)

    mu, u, k = jnp.asarray([0.0, 1.0]), jnp.asarray([0.5]), jnp.asarray(2.0)
    for propagation in ("first_order", "unscented"):
        g_mu, g_u, g_k = jax.grad(spread, argnums=(0, 1, 2))(mu, u, k, propagation)
        eps = 1e-2
        for arg, grad in ((0, g_mu), (1, g_u)):
            base = (mu, u)[arg]
            for i in range(base.shape[0]):
                e = jnp.zeros_like(base).at[i].set(eps)
                args_p = [mu, u, k]
                args_m = [mu, u, k]
                args_p[arg], args_m[arg] = base + e, base - e
                fd = (spread(*args_p, propagation) - spread(*args_m, propagation)) / (2 * eps)
                np.testing.assert_allclose(float(grad[i]), float(fd), rtol=2e-2, atol=1e-5)
        fd_k = (spread(mu, u, k + eps, propagation) - spread(mu, u, k - eps, propagation)) / (
            2 * eps
        )
        np.testing.assert_allclose(float(g_k), float(fd_k), rtol=2e-2)
        assert float(g_u[0]) > 0.0  # the commanded speed widens the speed's noise
        #  with k = 0 the scale is one everywhere: the control's gradient is zero
        g_u0 = jax.grad(spread, argnums=1)(mu, u, jnp.asarray(0.0), propagation)
        assert float(jnp.abs(g_u0[0])) < 1e-6


def test_the_gradient_planner_differentiates_through_the_scale():
    """The term tree's gradient in a continuous plan agrees with central differences and
    moves with the scale's leaf (with ``k = 0`` the scale is one everywhere): the planner sees
    that the command widens the noise."""
    system, params = _speed_model(_scaled(SpeedScale(0.1, 0.8, 2.0), SPEED_CHOL), None)
    params = dataclasses.replace(
        params,
        state_targets={"x": Gaussian(loc=jnp.asarray([0.0, 1.0]), chol=jnp.eye(2))},
    )
    beliefs = system.initial_priors(params)

    def score(plan, k):
        p = dataclasses.replace(
            params, transition={"x": _scaled(SpeedScale(0.1, 0.8, k), SPEED_CHOL)}
        )
        return total(system.score_policies(p, beliefs, plan)).midpoint

    plan = jnp.full((1, 1), 1.5)
    g_scaled = jax.grad(score)(plan, 2.0)
    g_flat = jax.grad(score)(plan, 0.0)
    assert np.all(np.isfinite(np.asarray(g_scaled)))
    eps = 1e-2
    fd = (score(plan + eps, 2.0) - score(plan - eps, 2.0)) / (2 * eps)
    np.testing.assert_allclose(float(g_scaled[0, 0]), float(fd), rtol=2e-2)
    assert abs(float(g_scaled[0, 0]) - float(g_flat[0, 0])) > 0.1


def test_jit_and_vmap_agree_with_the_eager_step():
    f = Unicycle(UNI_H, UNI_K)
    for propagation in ("first_order", "unscented"):
        decl = _scaled(f, Q3, propagation)
        eager = _one_step(decl, X3.loc, X3.chol, U3.loc)
        jitted = jax.jit(lambda loc, decl=decl: _one_step(decl, loc, X3.chol, U3.loc))(X3.loc)
        _close(jitted, eager, 1e-6, 1e-6)
        locs = X3.loc + jnp.asarray([[0.0, 0.0, 0.0], [0.1, -0.2, 0.5], [0.0, 0.3, -1.0]])
        batched = jax.vmap(lambda loc, decl=decl: _one_step(decl, loc, X3.chol, U3.loc))(locs)
        for i in range(3):
            one = _one_step(decl, locs[i], X3.chol, U3.loc)
            _close(jax.tree_util.tree_map(lambda a, i=i: a[i], batched), one, 1e-6, 1e-6)


def test_a_mixture_carried_factor_steps_each_component_with_its_own_scale():
    f = AffineScale(F2, B_OFF, C2, BS)
    prior = MixtureGaussian(
        log_weights=jnp.log(jnp.asarray([0.3, 0.7])),
        means=jnp.stack([MU2, -MU2]),
        chols=jnp.stack([P2, 0.5 * P2]),
    )
    system = System(B_deps=((),), state_names=("x",), policy_len=1, mixture_k=2, epistemic=False)
    for propagation in ("first_order", "unscented"):
        decl = _scaled(f, Q2, propagation)
        params = SystemParams(transition={"x": decl}, initial_state_prior={"x": prior})
        stepped = system.update_empirical_prior(params, system.initial_priors(params), {})["x"]
        for i in range(2):
            one = _one_step(decl, prior.means[i], prior.chols[i])
            np.testing.assert_allclose(stepped.means[i], one.loc, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(
                stepped.chols[i] @ stepped.chols[i].T, one.cov, rtol=1e-5, atol=1e-6
            )


# --------------------------------------------------------------------------- the record


@pytest.fixture
def registry(monkeypatch):
    """An empty registry for the test, the module's own restored afterwards."""
    monkeypatch.setattr(_description, "_TRANSITIONS", {})


def _speed_params(noise_scale="per_coordinate", k=2.0, **over):
    f = SpeedScale(0.1, 0.8, k) if noise_scale is not None else SpeedPlain(0.1, 0.8)
    decl = NonlinearGaussianTransition(
        f, _noise(jnp.zeros((1, 2)), SPEED_CHOL), "unscented", noise_scale
    )
    return _speed_model(decl, None, **over)


def test_the_document_records_the_scale_and_builds_it_back(registry):
    register_transition("tests.speed_scale", SpeedScale)
    register_transition("tests.speed_plain", SpeedPlain)
    system, params = _speed_params()
    doc = describe(system, params)
    document = json.loads(json.dumps(doc.document))
    text = json.dumps(document)
    assert '"noise_scale": "per_coordinate"' in text
    built, built_params = build(document, arrays=doc.arrays)
    decl = built_params.transition["x"]
    assert decl.noise_scale == "per_coordinate" and isinstance(decl.f, SpeedScale)
    beliefs = system.initial_priors(params)
    action = {"u": jnp.asarray([1.0])}
    _close(
        built.update_empirical_prior(built_params, beliefs, action),
        system.update_empirical_prior(params, beliefs, action),
    )
    #  a map without a scale writes no option, and its document builds the plain map
    plain_system, plain_params = _speed_params(None)
    plain_doc = json.loads(json.dumps(describe(plain_system, plain_params).document))
    assert "noise_scale" not in json.dumps(plain_doc)
    _, rebuilt = build(plain_doc, arrays=describe(plain_system, plain_params).arrays)
    assert rebuilt.transition["x"].noise_scale is None


def _strip_option(document):
    """``document`` with every ``noise_scale`` option removed: the record as a reader that
    ignores it would see it."""
    if isinstance(document, dict):
        return {k: _strip_option(v) for k, v in document.items() if k != "noise_scale"}
    if isinstance(document, list):
        return [_strip_option(v) for v in document]
    return document


def test_a_document_that_drops_the_option_is_refused_not_misread(registry):
    """The map returns a pair, so a record without the option (a map declared to return the
    mean alone) fails validation rather than building a model whose noise is unscaled; an
    unknown value is refused naming the values."""
    register_transition("tests.speed_scale", SpeedScale)
    system, params = _speed_params()
    doc = describe(system, params)
    stripped = _strip_option(json.loads(json.dumps(doc.document)))
    with pytest.raises(ValueError, match="noise_scale=None declares"):
        built, built_params = build(stripped, arrays=doc.arrays)
        built.update_empirical_prior(
            built_params, built.initial_priors(built_params), {"u": jnp.ones(1)}
        )
    bad = json.loads(json.dumps(doc.document).replace('"per_coordinate"', '"matrix"'))
    with pytest.raises(DescriptionError, match="noise_scale='matrix'"):
        build(bad, arrays=doc.arrays)


def test_description_length_prices_the_map_as_before():
    """The scale is the map's code and leaves: the family term is the plain map's with the
    same leaves, and the note says the noise is scaled."""
    scaled_system, scaled = _speed_params()
    plain_system, plain = _speed_params(None)
    n = 50.0
    a = description_length(scaled_system, n, params=scaled).per_address["transition:x"]
    b = description_length(plain_system, n, params=plain).per_address["transition:x"]
    #  SpeedScale has one leaf more (k) than SpeedPlain
    from hybrid_aif.description.mdl import resolution_term

    assert a.family == pytest.approx(resolution_term(3 + 2 + 3, n))
    assert b.family == pytest.approx(resolution_term(2 + 2 + 3, n))
    assert "scaled per coordinate" in a.note and "scaled" not in b.note


# ---------------------------------------------------------------------------- the refusals


def _validated(decl, d=2):
    system = System(B_deps=((),), state_names=("x",), policy_len=1)
    params = SystemParams(
        transition={"x": decl},
        initial_state_prior={"x": Gaussian(loc=jnp.zeros(d), chol=jnp.eye(d))},
    )
    system._validate(params)
    return system, params


def test_the_declaration_and_the_return_must_agree():
    chol = 0.1 * jnp.eye(2)
    with pytest.raises(ValueError, match=r"returns a tuple of 2 entries.*noise_scale=None"):
        _validated(NonlinearGaussianTransition(Pair(), _noise(jnp.zeros((1, 2)), chol)))
    with pytest.raises(ValueError, match=r"returns float32\[2\].*must return the pair"):
        _validated(_scaled(Mean(), chol))
    #  the scale ``Short`` makes with ``jnp.ones`` is in the default float dtype
    scale = "float64" if jax.config.jax_enable_x64 else "float32"
    with pytest.raises(ValueError, match=rf"returns \(float32\[2\], {scale}\[1\]\)"):
        _validated(_scaled(Short(), chol))
    with pytest.raises(ValueError, match="noise_scale='diagonal'"):
        NonlinearGaussianTransition(Mean(), _noise(jnp.zeros((1, 2)), chol), noise_scale="diagonal")
    #  read directly, a map's return is checked against the declaration when traced
    with pytest.raises(TypeError, match="noise_scale=None declares"):
        NonlinearGaussianTransition(Pair(), _noise(jnp.zeros((1, 2)), chol))._mean(jnp.zeros(2), ())
    with pytest.raises(TypeError, match="returns the pair"):
        _scaled(Mean(), chol)._mean_and_scale(jnp.zeros(2), ())


def test_the_refusals_of_a_nonlinear_transition_apply_unchanged():
    """A zero process noise and a merge that slices ``f``'s inputs are refused with the plain
    map's sentences; switching dynamics, once refused, are carried as a mode mixture for the
    scaled map too (nonlinear-transitions.md N4, amended 2026-11-15), the scale read per mode."""
    f = AffineScale(F2, B_OFF, C2, BS)
    system = System(
        B_deps=((0,), (0,)),
        state_names=("mode", "x"),
        policy_len=1,
        mixture_k=2,
        mixture_schedule="mode_aligned",
    )
    params = SystemParams(
        transition={
            "mode": ConditionalCategorical(log_table=jnp.zeros((2, 2)), parent_ndims=1),
            "x": _scaled(f, jnp.broadcast_to(Q2, (2, 2, 2)), loc=jnp.zeros((2, 2))),
        },
        initial_state_prior={
            "mode": Categorical(logits=jnp.zeros(2)),
            "x": Gaussian(loc=jnp.zeros(2), chol=jnp.eye(2)),
        },
    )
    prior = system.update_empirical_prior(params, system.initial_priors(params), {})
    assert isinstance(prior["x"], MixtureGaussian) and prior["x"].num_components == 2
    assert np.all(np.isfinite(np.asarray(prior["x"].chols)))
    noiseless = _scaled(f, jnp.diag(jnp.asarray([0.1, 0.0])))
    lone = System(B_deps=((),), state_names=("x",), policy_len=1)
    lone_params = SystemParams(
        transition={"x": noiseless},
        initial_state_prior={"x": Gaussian(loc=jnp.zeros(2), chol=jnp.eye(2))},
    )
    with pytest.raises(ValueError, match="zero on its Cholesky factor's diagonal"):
        lone.update_empirical_prior(lone_params, lone.initial_priors(lone_params), {})
    robot, _, rp = _ones_pair("gaussian", "first_order")
    twin, _, tp = _ones_pair("gaussian", "first_order")
    with pytest.raises(NotImplementedError, match="f's signature"):
        merge({"robot": robot, "twin": twin}, {"robot": rp, "twin": tp}, [Clamp("twin.u_m", 1)])


def test_closed_loop_scoring_reads_the_scaled_covariances():
    """``feedback="linear_quadratic"`` accepts the scaled map as it accepts the plain one: its
    gains read the mean's Jacobians and the recorded covariances carry the scale, so the
    closed-loop score moves with the scale's coefficient ``k`` (``k = 0`` a scale of one)."""
    target = {"x": Gaussian(loc=jnp.asarray([0.0, 1.0]), chol=0.3 * jnp.eye(2))}
    plan = jnp.full((2, 1), 1.5)
    scores = {}
    for feedback in (None, "linear_quadratic"):
        for k in (0.0, 2.0):
            system, params = _speed_params(
                k=k, feedback=feedback, policy_len=2, planning_belief="posterior", epistemic=True
            )
            params = dataclasses.replace(params, state_targets=target)
            tree = system.score_policies(params, system.initial_priors(params), plan)
            scores[(feedback, k)] = float(total(tree).midpoint)
    assert all(np.isfinite(v) for v in scores.values())
    closed = scores[("linear_quadratic", 2.0)] - scores[("linear_quadratic", 0.0)]
    assert abs(closed) > 0.1, scores
    assert scores[("linear_quadratic", 2.0)] != scores[(None, 2.0)]


# ------------------------------------------------- the neural transition, through the same path


def _old_mlp_step(q, net, parents):
    """The neural step as a first-order formula written out: ``m(μ)``, ``J Σ Jᵀ + diag σ(μ)²``."""
    head = net.condition_on((q.loc, *parents))
    jac = jax.jacfwd(lambda x: net.condition_on((x, *parents)).loc)(q.loc)
    return head.loc, jac @ q.cov @ jac.T + head.cov


def test_the_neural_transition_steps_as_before():
    """The neural step goes through the shared propagation with ``Q = I`` and the head's
    scale; its mean is bit for bit, its covariance to rounding (the spread is formed as ``(J
    L)(J L)ᵀ``)."""
    net = MLPConditional.from_shapes((3, 2), 3, key=jax.random.PRNGKey(7), width=16, depth=2)
    for act in (jnp.asarray([0.5, -0.3]), jnp.asarray([0.0, 1.0])):
        got = _hybrid._mlp_step(X3, net, (act,))
        mean, cov = _old_mlp_step(X3, net, (act,))
        assert np.array_equal(np.asarray(got.loc), np.asarray(mean))
        np.testing.assert_allclose(np.asarray(got.cov), np.asarray(cov), rtol=1e-6, atol=1e-8)


# ------------------------------------------------- review additions: kinks, dimension one, merge


class Displaced(eqx.Module):
    """:class:`SpeedPlain`'s mean with the decoder-error design's scale form, ``0.2 + sqrt(1 −
    rho(d)² + guard)``, ``rho = exp(−d²)``, ``d = h u`` the commanded displacement: kinked at
    ``d = 0`` without the guard."""

    h: float
    a: float
    guard: float = eqx.field(static=True, default=1e-6)

    def __call__(self, x, controls):
        (u,) = controls
        mean = jnp.stack([x[0] + self.h * x[1], self.a * x[1] + (1.0 - self.a) * u[0]])
        rho = jnp.exp(-((self.h * u[0]) ** 2))
        s = 0.2 + jnp.sqrt(1.0 - rho**2 + self.guard)
        return mean, jnp.stack([s, s])


@pytest.mark.parametrize("propagation", ["first_order", "unscented"])
def test_a_guarded_scale_has_a_finite_gradient_at_rest(propagation):
    """At zero displacement the unguarded ``sqrt(1 − rho²)`` has a NaN or infinite derivative,
    which reaches the predicted covariance under both rules (the unscented centre point's zero
    weight does not cancel it); with ``+ eps`` under the root the gradient is finite, the
    control clamped or carried as a belief."""
    x_chol = jnp.diag(jnp.asarray([0.1, 0.3]))
    mu = jnp.asarray([0.0, 0.0])

    def spread(u, guard, belief):
        decl = _scaled(Displaced(0.5, 0.8, guard), SPEED_CHOL, propagation)
        given = Gaussian(loc=u, chol=jnp.asarray([[0.2]])) if belief else u
        return jnp.trace(_one_step(decl, mu, x_chol, given).cov)

    zero = jnp.zeros(1)
    for belief in (False, True):
        assert np.all(np.isfinite(np.asarray(jax.grad(spread)(zero, 1e-6, belief))))
    assert not np.all(np.isfinite(np.asarray(jax.grad(spread)(zero, 0.0, False))))


@pytest.mark.parametrize("propagation", ["first_order", "unscented"])
def test_the_gradient_planner_starts_at_rest_with_a_guarded_scale(propagation):
    """The gradient planner's descent from the box's midpoint, a zero command, returns a finite
    plan that moves toward the target with the guarded scale; the unguarded one returns NaN."""
    from hybrid_aif.planning.gradient import descend

    plans = {}
    for guard in (0.0, 1e-6):
        decl = _scaled(Displaced(0.5, 0.8, guard), SPEED_CHOL, propagation)
        system, params = _speed_model(decl, None, policy_len=2)
        params = dataclasses.replace(
            params, state_targets={"x": Gaussian(loc=jnp.asarray([2.0, 1.0]), chol=jnp.eye(2))}
        )
        beliefs = system.initial_priors(params)

        def score(plan, system=system, params=params, beliefs=beliefs):
            return total(system.score_policies(params, beliefs, plan[None])).midpoint[0]

        plans[guard], *_ = descend(
            score,
            jnp.zeros((2, 1)),
            -5.0 * jnp.ones(1),
            5.0 * jnp.ones(1),
            jnp.asarray(0.1),
            steps=30,
            optimizer="adam",
        )
    assert np.all(np.isfinite(np.asarray(plans[1e-6])))
    assert float(plans[1e-6][0, 0]) > 0.5  # it moves toward the target ahead
    assert not np.all(np.isfinite(np.asarray(plans[0.0])))


class Scalar(eqx.Module):
    """A one-dimensional state, ``x' = 0.9 x``, its scale ``1 + x²``."""

    def __call__(self, x, controls):
        del controls
        return 0.9 * x, 1.0 + x**2


def test_a_one_dimensional_state():
    """``d = 1``: the predictive variance is ``0.81 P + q² E[(1 + x²)²]``, exact unscented
    (``E[(1 + x²)²] = 1 + 2(μ² + P) + μ⁴ + 6 μ² P + 3 P²``; sigma points match moments to degree
    three only, so the unscented value is checked to its own rule), first order at the mean."""
    mu, sd, q = 0.5, 0.3, 0.2
    P = sd**2
    first = _one_step(_scaled(Scalar(), jnp.asarray([[q]])), jnp.asarray([mu]), jnp.asarray([[sd]]))
    np.testing.assert_allclose(
        float(first.cov[0, 0]), 0.81 * P + q**2 * (1 + mu**2) ** 2, rtol=1e-5
    )
    sigma = _one_step(
        _scaled(Scalar(), jnp.asarray([[q]]), "unscented"), jnp.asarray([mu]), jnp.asarray([[sd]])
    )
    points = np.asarray([mu + sd, mu - sd])  # n = 1, kappa = 0: two points, weight 1/2 each
    want = 0.81 * P + q**2 * np.mean((1 + points**2) ** 2)
    np.testing.assert_allclose(float(sigma.cov[0, 0]), want, rtol=1e-5)
    np.testing.assert_allclose(float(sigma.loc[0]), 0.9 * mu, rtol=1e-6)


def test_merge_carries_the_scaled_declaration():
    """A merged composite keeps ``noise_scale`` and its members step as they do alone."""
    robot, _, rp = _ones_pair("gaussian", "unscented")
    twin, _, tp = _ones_pair("gaussian", "unscented")
    composite, cp = merge({"robot": robot, "twin": twin}, {"robot": rp, "twin": tp})
    decl = cp.transition["robot.x"]
    assert decl.noise_scale == "per_coordinate" and decl.propagation == "unscented"
    action = {"robot.m": jnp.int32(1), "robot.u": jnp.asarray([0.3])}
    action |= {"twin.m": jnp.int32(0), "twin.u": jnp.asarray([0.0])}
    one = composite.update_empirical_prior(cp, composite.initial_priors(cp), action)
    alone = robot.update_empirical_prior(
        rp, robot.initial_priors(rp), {"m": jnp.int32(1), "u": jnp.asarray([0.3])}
    )
    _close(one["robot.x"], alone["x"])
