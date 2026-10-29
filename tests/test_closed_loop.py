"""Closed-loop scoring of Gaussian targets, ``System(feedback="linear_quadratic")``.

A plan is scored as the policy that follows it and corrects deviations by the linear-quadratic
regulator's feedback on the estimate. These tests pin: the score against Monte Carlo of that
policy executed with a Kalman filter in the true linear-Gaussian system, and against an
independent numpy implementation of the formulas; ``feedback=None`` unchanged; a factor no
control reaches scored as the open-loop cross-entropy; the lane model's answer, worth its
pragmatic value in closed loop and only its information gain in open loop (the offline
harness's numbers); the nonlinear unicycle against Monte Carlo of its corrected closed loop;
the two decompositions moved alike; the refusals; the description; and the gradient planner.

Design: docs/design/closed-loop-planning.md
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import (
    Box,
    ConditionalGaussian,
    Gaussian,
    LinearGaussianObservation,
    LinearGaussianTransition,
    MixtureGaussian,
    NonlinearGaussianTransition,
    System,
    SystemParams,
)
from hybrid_aif.description import build, describe
from hybrid_aif.distributions import MLPConditional
from hybrid_aif.objectives import total
from hybrid_aif.objectives.base import term_sign
from hybrid_aif.objectives.closed_loop import riccati_gains
from hybrid_aif.planning import gradient_plan
from hybrid_aif.planning.enumerate import evaluate_policies

KEY = jax.random.PRNGKey(0)


def _score(system, params, plan, present=None, beliefs=None):
    """The term tree of one float plan ``(H, du)``, every leaf a scalar."""
    beliefs = system.initial_priors(params) if beliefs is None else beliefs
    ctx = system.planner_context(params, beliefs, present=present)
    tree = evaluate_policies(
        jnp.asarray(plan)[None],
        ctx.init_beliefs,
        ctx.transition_fn,
        ctx.step_terms_fn,
        ctx.terminal_terms_fn,
    )
    return jax.tree_util.tree_map(lambda x: x[0], tree)


def _leaf(tree, term, key) -> float:
    iv = tree[term][key]
    assert float(iv.lower) == float(iv.upper)
    return float(iv.lower)


def _gauss_ce(mean, cov, mu, target_cov) -> float:
    """``−E log N(z; mu, target_cov)`` under ``z ~ N(mean, cov)``."""
    lam = np.linalg.inv(target_cov)
    dev = mean - mu
    _, logdet = np.linalg.slogdet(2 * np.pi * target_cov)
    return 0.5 * (dev @ lam @ dev + np.trace(lam @ cov) + logdet)


def _log_gauss(z, mu, cov) -> np.ndarray:
    """``log N(z; mu, cov)`` over the leading axis of ``z``."""
    lam = np.linalg.inv(cov)
    dev = z - mu
    _, logdet = np.linalg.slogdet(2 * np.pi * cov)
    return -0.5 * (np.einsum("ni,ij,nj->n", dev, lam, dev) + logdet)


# ------------------------------------------------------------- a small linear-Gaussian model

HORIZON = 5
F_X = np.array([[1.0, 0.3, 0.0], [0.0, 0.95, 0.0], [0.0, 0.0, 1.0]])
G_X = np.array([[0.05], [0.3], [0.0]])  # the third dimension, an offset, is not reached
B_X = np.array([0.01, 0.0, 0.0])
Q_CHOL = np.diag([0.02, 0.03, 0.005])
M0 = np.array([0.0, 0.1, 0.2])
S0_CHOL = np.diag([0.1, 0.1, 0.3])
H_Y = np.array([[1.0, 0.0, 1.0]])  # position plus offset
H_Z = np.array([[0.0, 0.0, 1.0]])  # the offset
R_Y, R_Z = 0.05, 0.04
PRESENT_Y = np.array([True, True, False, True, True])
PRESENT_Z = np.array([False, False, True, False, False])
STATE_TARGET = (np.array([0.5, 0.0, 0.0]), np.diag([0.2, 0.5, 2.0]) ** 2)
TERMINAL_TARGET = (np.array([1.0, 0.0, 0.0]), np.diag([0.1, 0.3, 3.0]) ** 2)
OBS_TARGET = (np.array([0.8]), np.array([[0.3**2]]))
CONTROL_TARGET = (np.array([0.0]), np.array([[1.0]]))
PLAN = np.array([[0.3], [-0.2], [0.1], [0.4], [0.0]])


def _f32(x):
    return jnp.asarray(x, jnp.float32)


def small_lg(decomposition="utility_infogain", feedback="linear_quadratic", epistemic=True):
    system = System(
        A_deps=((0,), (0,)),
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("u",),
        factor_names=("x", "u_x"),
        modality_names=("y", "z"),
        policy_len=HORIZON,
        planner="gradient",
        planning_belief="posterior",
        feedback=feedback,
        decomposition=decomposition,
        epistemic=epistemic,
    )
    params = SystemParams(
        transition={
            "x": LinearGaussianTransition(
                F=_f32(F_X),
                G=_f32(G_X),
                innovations=ConditionalGaussian(
                    loc=_f32(B_X)[None], chol=_f32(Q_CHOL)[None], parent_ndims=1
                ),
            )
        },
        initial_state_prior={"x": Gaussian(loc=_f32(M0), chol=_f32(S0_CHOL))},
        likelihood={
            "y": LinearGaussianObservation(H=_f32(H_Y), c=jnp.zeros(1), noise=R_Y * jnp.eye(1)),
            "z": LinearGaussianObservation(H=_f32(H_Z), c=jnp.zeros(1), noise=R_Z * jnp.eye(1)),
        },
        obs_targets={"y": Gaussian.from_cov(_f32(OBS_TARGET[0]), _f32(OBS_TARGET[1]))},
        state_targets={
            "x": Gaussian.from_cov(_f32(STATE_TARGET[0]), _f32(STATE_TARGET[1])),
            "u_x": Gaussian.from_cov(_f32(CONTROL_TARGET[0]), _f32(CONTROL_TARGET[1])),
        },
        terminal_targets={
            "x": Gaussian.from_cov(_f32(TERMINAL_TARGET[0]), _f32(TERMINAL_TARGET[1]))
        },
        control={"u_x": Box(-5.0 * jnp.ones(1), 5.0 * jnp.ones(1))},
    )
    return system, params


SMALL_PRESENT = {"y": jnp.asarray(PRESENT_Y), "z": jnp.asarray(PRESENT_Z)}


def _reference(plan):
    """The closed-loop leaves by the formulas, in float64 numpy: the filter's covariances, the
    Riccati gains, the estimate spread forward, and each target's cross-entropy on
    ``N(m̄, Σ + C)``. Returns the leaves, the gains and the nominal means."""
    Q = Q_CHOL @ Q_CHOL.T
    channels = ((H_Y, R_Y**2, PRESENT_Y), (H_Z, R_Z**2, PRESENT_Z))
    sig, mean = S0_CHOL @ S0_CHOL.T, M0.copy()
    means, preds, priors, posts = [M0.copy()], [], [], []
    pred = sig.copy()
    for t in range(HORIZON):
        mean = F_X @ mean + G_X @ plan[t] + B_X
        pred = F_X @ pred @ F_X.T + Q
        prior = F_X @ sig @ F_X.T + Q
        post = prior
        for H, R, present in channels:
            if present[t]:
                S = H @ post @ H.T + R
                post = post - post @ H.T @ np.linalg.solve(S, H @ post)
        sig = post
        means.append(mean)
        preds.append(pred)
        priors.append(prior)
        posts.append(post)
    lam_x = np.linalg.inv(STATE_TARGET[1])
    lam_h = np.linalg.inv(TERMINAL_TARGET[1])
    lam_y = np.linalg.inv(OBS_TARGET[1])
    R = np.linalg.inv(CONTROL_TARGET[1])
    W = [lam_x + (H_Y.T @ lam_y @ H_Y if PRESENT_Y[t] else 0.0) for t in range(HORIZON)]
    W[-1] = W[-1] + lam_h
    P = W[-1]
    K = [None] * HORIZON
    for t in range(HORIZON - 1, -1, -1):
        K[t] = -np.linalg.pinv(R + G_X.T @ P @ G_X) @ G_X.T @ P @ F_X
        P = (W[t - 1] if t > 0 else 0.0) + F_X.T @ P @ (F_X + G_X @ K[t])
    C = np.zeros((3, 3))
    leaves = {"risk_x": 0.0, "terminal_x": 0.0, "utility_y": 0.0, "risk_u": 0.0}
    for t in range(HORIZON):
        leaves["risk_u"] += -_log_gauss(plan[t][None], *CONTROL_TARGET)[0] + 0.5 * np.trace(
            R @ K[t] @ C @ K[t].T
        )
        A = F_X + G_X @ K[t]
        C = A @ C @ A.T + priors[t] - posts[t]
        S = posts[t] + C
        leaves["risk_x"] += _gauss_ce(means[t + 1], S, *STATE_TARGET)
        if PRESENT_Y[t]:
            leaves["utility_y"] -= _gauss_ce(
                H_Y @ means[t + 1], H_Y @ S @ H_Y.T + R_Y**2, *OBS_TARGET
            )
    leaves["terminal_x"] = _gauss_ce(means[-1], S, *TERMINAL_TARGET)
    return leaves, np.asarray(K), np.asarray(means)


def _monte_carlo(plan, gains, means, n=200_000, seed=1):
    """Execute ``u_t = ū_t + K_t (x̂_t − m̄_t)`` with a Kalman filter in the true system: the
    realised cost of every leaf, per draw."""
    rng = np.random.default_rng(seed)
    Q = Q_CHOL @ Q_CHOL.T
    x = M0 + rng.standard_normal((n, 3)) @ S0_CHOL.T
    est = np.broadcast_to(M0, (n, 3)).copy()
    sig = S0_CHOL @ S0_CHOL.T
    cost = {k: np.zeros(n) for k in ("risk_x", "terminal_x", "utility_y", "risk_u")}
    for t in range(HORIZON):
        u = plan[t] + (est - means[t]) @ gains[t].T
        cost["risk_u"] -= _log_gauss(u, *CONTROL_TARGET)
        x = x @ F_X.T + u @ G_X.T + B_X + rng.standard_normal((n, 3)) @ Q_CHOL.T
        est = est @ F_X.T + u @ G_X.T + B_X
        sig = F_X @ sig @ F_X.T + Q
        for name, H, r, present in (("y", H_Y, R_Y, PRESENT_Y), ("z", H_Z, R_Z, PRESENT_Z)):
            if not present[t]:
                continue
            y = x @ H.T + r * rng.standard_normal((n, 1))
            if name == "y":
                cost["utility_y"] += _log_gauss(y, *OBS_TARGET)
            S = H @ sig @ H.T + r**2
            gain = sig @ H.T @ np.linalg.inv(S)
            est = est + (y - est @ H.T) @ gain.T
            sig = sig - gain @ H @ sig
        cost["risk_x"] -= _log_gauss(x, *STATE_TARGET)
    cost["terminal_x"] -= _log_gauss(x, *TERMINAL_TARGET)
    return cost


LIB_KEYS = {
    "risk_x": ("risk", "x"),
    "terminal_x": ("terminal_risk", "x"),
    "utility_y": ("utility", "y"),
    "risk_u": ("risk", "u_x"),
}


def test_the_score_is_the_formulas_and_the_monte_carlo_cost_of_the_corrected_plan():
    """Every covered leaf equals the numpy implementation of the formulas (float32 against
    float64, 1e-4 relative) and the Monte Carlo cost of executing the plan with a Kalman filter
    and the Riccati corrections in the true system, within four standard errors (200,000
    draws, float64)."""
    system, params = small_lg()
    tree = _score(system, params, _f32(PLAN), SMALL_PRESENT)
    want, gains, means = _reference(PLAN)
    cost = _monte_carlo(PLAN, gains, means)
    for name, (term, key) in LIB_KEYS.items():
        got = _leaf(tree, term, key)
        assert got == pytest.approx(want[name], rel=1e-4, abs=1e-4), name
        draws = cost[name]
        se = draws.std() / np.sqrt(draws.size)
        assert abs(draws.mean() - got) < 4 * se + 1e-4 * abs(got), (name, draws.mean(), got, se)
    # the corrections act: the closed-loop state leaf is below the open-loop cross-entropy
    open_loop = _score(
        dataclasses.replace(system, feedback=None), params, _f32(PLAN), SMALL_PRESENT
    )
    pred = system.predict_beliefs(params, system.initial_priors(params), _f32(PLAN))
    ce = sum(_gauss_ce(np.asarray(q["x"].loc), np.asarray(q["x"].cov), *STATE_TARGET) for q in pred)
    assert _leaf(tree, "risk", "x") < ce - 0.1
    # the control target's leaf is its open-loop leaf plus the corrections' cost
    assert _leaf(tree, "risk", "u_x") > _leaf(open_loop, "risk", "u_x") + 1e-3


def test_the_riccati_gains_match_numpy():
    _, gains, _ = _reference(PLAN)
    lam_x = np.linalg.inv(STATE_TARGET[1])
    lam_y = np.linalg.inv(OBS_TARGET[1])
    W = np.stack([lam_x + (H_Y.T @ lam_y @ H_Y if PRESENT_Y[t] else 0.0) for t in range(HORIZON)])
    W[-1] += np.linalg.inv(TERMINAL_TARGET[1])
    R = np.broadcast_to(np.linalg.inv(CONTROL_TARGET[1]), (HORIZON, 1, 1))
    K = riccati_gains(
        _f32(np.broadcast_to(F_X, (HORIZON, 3, 3))),
        _f32(np.broadcast_to(G_X, (HORIZON, 3, 1))),
        _f32(W),
        _f32(R),
    )
    np.testing.assert_allclose(np.asarray(K), gains, rtol=1e-4, atol=1e-5)


# ------------------------------------------------------------ feedback=None is unchanged


@pytest.mark.parametrize("decomposition", ["utility_infogain", "risk_ambiguity"])
def test_without_feedback_the_carry_and_the_tree_are_the_open_loop_ones(decomposition):
    """``feedback=None`` builds the plain closures: the carry is the belief tuple with no
    record, and a state target is scored by its KL on the predictive belief."""
    system, params = small_lg(decomposition, feedback=None)
    beliefs = system.initial_priors(params)
    ctx = system.planner_context(params, beliefs, present=SMALL_PRESENT)
    assert isinstance(ctx.init_beliefs, tuple) and len(ctx.init_beliefs) == 1
    a = _score(system, params, _f32(PLAN), SMALL_PRESENT)
    assert set(a["risk"]) >= {"x", "u_x"}
    q = system.predict_beliefs(params, beliefs, _f32(PLAN))
    kl = sum(
        float(q_t["x"].kl_to(params.state_targets["x"]).lower) for q_t in q
    )  # open loop scores a state target by its KL
    assert _leaf(a, "risk", "x") == pytest.approx(kl, rel=1e-5)


# ----------------------------------------------------- a factor no control reaches


def _two_factors():
    """``x`` steered by ``u``; ``w``, a random walk no control reaches, read on some steps and
    targeted."""
    system = System(
        A_deps=((0,), (1,)),
        B_deps=((2,), (), ()),
        controls=(2,),
        action_names=("u",),
        factor_names=("x", "w", "u_x"),
        modality_names=("y", "v"),
        policy_len=4,
        planning_belief="posterior",
        feedback="linear_quadratic",
        planner="gradient",
    )
    params = SystemParams(
        transition={
            "x": LinearGaussianTransition(
                F=jnp.eye(1),
                G=jnp.eye(1),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 1)), chol=0.1 * jnp.eye(1)[None], parent_ndims=1
                ),
            ),
            "w": LinearGaussianTransition(
                F=jnp.asarray([[1.0, 0.1], [0.0, 1.0]]),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 2)), chol=0.1 * jnp.eye(2)[None], parent_ndims=1
                ),
            ),
        },
        initial_state_prior={
            "x": Gaussian(loc=jnp.zeros(1), chol=0.3 * jnp.eye(1)),
            "w": Gaussian(loc=jnp.asarray([0.2, 0.0]), chol=0.4 * jnp.eye(2)),
        },
        likelihood={
            "y": LinearGaussianObservation(H=jnp.eye(1), c=jnp.zeros(1), noise=0.1 * jnp.eye(1)),
            "v": LinearGaussianObservation(
                H=jnp.asarray([[1.0, 0.0]]), c=jnp.zeros(1), noise=0.05 * jnp.eye(1)
            ),
        },
        state_targets={
            "x": Gaussian(loc=jnp.ones(1), chol=0.2 * jnp.eye(1)),
            "w": Gaussian(loc=jnp.zeros(2), chol=jnp.diag(jnp.asarray([0.3, 0.5]))),
        },
        control={"u_x": Box(-3.0 * jnp.ones(1), 3.0 * jnp.ones(1))},
    )
    return system, params


def test_a_factor_no_control_reaches_scores_its_open_loop_cross_entropy():
    system, params = _two_factors()
    plan = jnp.asarray([[0.4], [0.3], [0.2], [0.1]])
    present = {"v": jnp.asarray([True, False, True, True])}
    tree = _score(system, params, plan, present)
    beliefs = system.initial_priors(params)
    target = params.state_targets["w"]
    want = sum(
        _gauss_ce(
            np.asarray(q["w"].loc),
            np.asarray(q["w"].cov),
            np.asarray(target.loc),
            np.asarray(target.cov),
        )
        for q in system.predict_beliefs(params, beliefs, plan)
    )
    assert _leaf(tree, "risk", "w") == pytest.approx(want, rel=1e-5)
    # while the steered factor's leaf takes the corrections
    ce_x = sum(
        _gauss_ce(np.asarray(q["x"].loc), np.asarray(q["x"].cov), np.ones(1), np.eye(1) * 0.04)
        for q in system.predict_beliefs(params, beliefs, plan)
    )
    assert _leaf(tree, "risk", "x") < ce_x - 0.5


# ------------------------------------------------------------------------- the lane

LANE_H = 6
#: The offline harness's exact LQG numbers (``cl_validate/cases.py``, ``lane``): the
#: cross-entropy closed-loop score without the answer less with it, at preference sd 0.02 and
#: 0.2 on ``p − b``.
HARNESS_ANSWER_VALUE = {0.02: 224.0194491251923, 0.2: 2.2401944912519234}


def lane(pref_sd: float, ask: bool, feedback="linear_quadratic", decomposition="utility_infogain"):
    """The lane, linearised, in the coordinates ``(e, θ, b)``, ``e = p − b`` the offset from the
    row: ``e`` moves by ``h v θ``, the heading by the steering control, the row's offset ``b``
    is constant. The estimator reads ``p = e + b`` every step, the pile ``e`` at steps 3 and 5,
    an answer reads ``b`` at step 1 when ``ask``. The target is on ``e`` (``θ`` and ``b`` held
    loosely, sd 1000)."""
    h, v = 0.5, 0.5
    system = System(
        A_deps=((0,), (0,), (0,)),
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("steer",),
        factor_names=("lane", "u_steer"),
        modality_names=("estimator", "pile", "answer"),
        policy_len=LANE_H,
        planner="gradient",
        planning_belief="posterior",
        feedback=feedback,
        decomposition=decomposition,
    )
    sigma0 = np.diag([0.1**2, 0.05**2, 0.3**2])  # over (p, θ, b)
    to_e = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def channel(row, sd):
        return LinearGaussianObservation(H=_f32([row]), c=jnp.zeros(1), noise=sd * jnp.eye(1))

    params = SystemParams(
        transition={
            "lane": LinearGaussianTransition(
                F=_f32([[1.0, h * v, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                G=_f32([[0.0], [h], [0.0]]),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 3)),
                    chol=jnp.diag(_f32([1e-2, 1e-2, 1e-3]))[None],
                    parent_ndims=1,
                ),
            )
        },
        initial_state_prior={"lane": Gaussian.from_cov(jnp.zeros(3), _f32(to_e @ sigma0 @ to_e.T))},
        likelihood={
            "estimator": channel([1.0, 0.0, 1.0], 0.1),
            "pile": channel([1.0, 0.0, 0.0], 0.02),
            "answer": channel([0.0, 0.0, 1.0], 0.02),
        },
        state_targets={
            "lane": Gaussian(loc=jnp.zeros(3), chol=jnp.diag(_f32([pref_sd, 1e3, 1e3])))
        },
        control={"u_steer": Box(-10.0 * jnp.ones(1), 10.0 * jnp.ones(1))},
    )
    present = {
        "pile": jnp.asarray([k in (3, 5) for k in range(LANE_H)]),
        "answer": jnp.asarray([ask and k == 1 for k in range(LANE_H)]),
    }
    return system, params, present


@pytest.mark.parametrize("pref_sd", [0.02, 0.2])
def test_an_answer_is_worth_its_pragmatic_value_in_closed_loop_and_only_its_gain_open(pref_sd):
    """The answer of ``b`` lowers the closed-loop risk by the harness's exact LQG amount; open
    loop the risk is the same with and without it, and G differs by the information leaves
    alone. Closed loop, G differs by those and the pragmatic value."""
    plan = jnp.zeros((LANE_H, 1))
    trees = {}
    for feedback in ("linear_quadratic", None):
        for ask in (False, True):
            system, params, present = lane(pref_sd, ask, feedback)
            trees[feedback, ask] = _score(system, params, plan, present)
    gain = {
        fb: _leaf(trees[fb, False], "risk", "lane") - _leaf(trees[fb, True], "risk", "lane")
        for fb in ("linear_quadratic", None)
    }
    assert gain["linear_quadratic"] == pytest.approx(HARNESS_ANSWER_VALUE[pref_sd], rel=1e-4)
    assert gain[None] == 0.0

    def info(tree):
        """The information side's value: minus its contribution to G."""
        return -sum(
            term_sign(term) * float(iv.lower)
            for term in ("infogain", "observation_interaction")
            for iv in tree.get(term, {}).values()
        )

    for fb in ("linear_quadratic", None):
        g_diff = float(total(trees[fb, False]).lower - total(trees[fb, True]).lower)
        info_diff = info(trees[fb, True]) - info(trees[fb, False])
        assert info_diff > 0.1  # the answer is informative
        pragmatic = gain[fb]
        assert g_diff == pytest.approx(pragmatic + info_diff, rel=1e-4, abs=1e-3)


#: The harness's closed-loop cross-entropy, ``Σ_τ ½ tr(Λ T S_τ Tᵀ)`` (the mean part and the
#: normaliser dropped), for its scalar and double-integrator cases (``cases.py``, horizon 6).
HARNESS_CE = {
    "scalar, every step": 65.03906232132705,
    "scalar, every third step": 74.1235773074338,
    "double integrator": 16.363977285801713,
}


def _harness_case(name: str):
    """One harness case as a single-factor System at its nominal (zero) plan: ``(system,
    params, present, the target's covariance)``."""
    if name.startswith("scalar"):
        F, G, q_sd = np.eye(1), np.eye(1), np.sqrt([0.01])
        H, r_sd, sigma0 = np.eye(1), np.sqrt(0.1), np.eye(1)
        every = 1 if name == "scalar, every step" else 3
        reads = [k % every == 0 for k in range(6)]
        target_sd = np.array([0.1])
    else:
        h = 0.5
        F, G = np.array([[1.0, h], [0.0, 1.0]]), np.array([[0.5 * h * h], [h]])
        q_sd, H, r_sd = np.sqrt([1e-4, 1e-3]), np.array([[1.0, 0.0]]), 0.05
        sigma0 = np.diag([0.2**2, 0.1**2])
        reads = [k in (0, 2, 4) for k in range(6)]
        target_sd = np.array([0.05, 1e3])  # the harness's T = [1, 0]; velocity held loosely
    d = F.shape[0]
    system = System(
        A_deps=((0,),),
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("u",),
        factor_names=("x", "u_x"),
        modality_names=("y",),
        policy_len=6,
        planner="gradient",
        planning_belief="posterior",
        feedback="linear_quadratic",
        epistemic=False,
    )
    target_cov = np.diag(target_sd**2)
    params = SystemParams(
        transition={
            "x": LinearGaussianTransition(
                F=_f32(F),
                G=_f32(G),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, d)), chol=jnp.diag(_f32(q_sd))[None], parent_ndims=1
                ),
            )
        },
        initial_state_prior={"x": Gaussian.from_cov(jnp.zeros(d), _f32(sigma0))},
        likelihood={
            "y": LinearGaussianObservation(H=_f32(H), c=jnp.zeros(1), noise=r_sd * jnp.eye(1))
        },
        state_targets={"x": Gaussian(loc=jnp.zeros(d), chol=jnp.diag(_f32(target_sd)))},
        control={"u_x": Box(-10.0 * jnp.ones(1), 10.0 * jnp.ones(1))},
    )
    return system, params, {"y": jnp.asarray(reads)}, target_cov


@pytest.mark.parametrize("name", sorted(HARNESS_CE))
def test_the_harness_cross_entropy_cases(name):
    """The closed-loop state leaf less its normaliser is the harness's exact LQG cross-entropy
    (the plan holds the mean on the target, so the mean part is zero)."""
    system, params, present, target_cov = _harness_case(name)
    tree = _score(system, params, jnp.zeros((6, 1)), present)
    normaliser = 6 * 0.5 * np.linalg.slogdet(2 * np.pi * target_cov)[1]
    assert _leaf(tree, "risk", "x") - normaliser == pytest.approx(HARNESS_CE[name], rel=1e-4)


@pytest.mark.parametrize(("ask", "want"), [(False, 610.3206475657521), (True, 386.3011984405598)])
def test_the_harness_lane_cross_entropy(ask, want):
    system, params, present = lane(0.02, ask)
    tree = _score(system, params, jnp.zeros((LANE_H, 1)), present)
    target_cov = np.diag([0.02, 1e3, 1e3]) ** 2
    normaliser = LANE_H * 0.5 * np.linalg.slogdet(2 * np.pi * target_cov)[1]
    assert _leaf(tree, "risk", "lane") - normaliser == pytest.approx(want, rel=1e-4)


# --------------------------------------------------------- the nonlinear unicycle


class RateUnicycle(eqx.Module):
    """The unicycle at speed ``v`` over a step ``h``, steered by a continuous turn rate."""

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


UNI_H, UNI_V, UNI_T = 0.2, 1.0, 8
UNI_Q = np.diag([0.01, 0.01, 0.03])
UNI_M0 = np.array([0.0, 0.3, 0.0])
UNI_S0 = np.diag([0.05, 0.1, 0.1])
UNI_READ = 0.05  # the lateral reading's noise sd
UNI_TARGET = (np.zeros(3), np.diag([1e3, 0.1, 1e3]) ** 2)  # stay on the line y = 0
UNI_PLAN = np.array([[-0.8], [-0.6], [-0.4], [-0.2], [0.0], [0.0], [0.2], [0.0]])


def unicycle(propagation: str):
    system = System(
        A_deps=((0,),),
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("turn",),
        factor_names=("pose", "u_pose"),
        modality_names=("lateral",),
        policy_len=UNI_T,
        planner="gradient",
        planning_belief="posterior",
        feedback="linear_quadratic",
    )
    params = SystemParams(
        transition={
            "pose": NonlinearGaussianTransition(
                RateUnicycle(UNI_H, UNI_V),
                ConditionalGaussian(loc=jnp.zeros((1, 3)), chol=_f32(UNI_Q)[None], parent_ndims=1),
                propagation,
            )
        },
        initial_state_prior={"pose": Gaussian(loc=_f32(UNI_M0), chol=_f32(UNI_S0))},
        likelihood={
            "lateral": LinearGaussianObservation(
                H=_f32([[0.0, 1.0, 0.0]]), c=jnp.zeros(1), noise=UNI_READ * jnp.eye(1)
            )
        },
        state_targets={"pose": Gaussian.from_cov(_f32(UNI_TARGET[0]), _f32(UNI_TARGET[1]))},
        control={"u_pose": Box(-3.0 * jnp.ones(1), 3.0 * jnp.ones(1))},
    )
    return system, params


def _uni_f(x, u):
    return np.stack(
        [
            x[..., 0] + UNI_H * UNI_V * np.cos(x[..., 2]),
            x[..., 1] + UNI_H * UNI_V * np.sin(x[..., 2]),
            x[..., 2] + UNI_H * u[..., 0],
        ],
        -1,
    )


def _uni_jac(x):
    return np.array(
        [
            [1.0, 0.0, -UNI_H * UNI_V * np.sin(x[2])],
            [0.0, 1.0, UNI_H * UNI_V * np.cos(x[2])],
            [0.0, 0.0, 1.0],
        ]
    )


UNI_G = np.array([[0.0], [0.0], [UNI_H]])


def _uni_reference():
    """The first-order linearisation along the nominal: the nominal means, the gains, and the
    closed-loop state leaf by the formulas (float64)."""
    Q = UNI_Q @ UNI_Q.T
    Hr = np.array([[0.0, 1.0, 0.0]])
    mean, sig = UNI_M0.copy(), UNI_S0 @ UNI_S0.T
    means, Fs, priors, posts = [mean.copy()], [], [], []
    for t in range(UNI_T):
        F = _uni_jac(mean)
        Fs.append(F)
        mean = _uni_f(mean, UNI_PLAN[t])
        prior = F @ sig @ F.T + Q
        S = Hr @ prior @ Hr.T + UNI_READ**2
        post = prior - prior @ Hr.T @ np.linalg.solve(S, Hr @ prior)
        sig = post
        means.append(mean.copy())
        priors.append(prior)
        posts.append(post)
    lam = np.linalg.inv(UNI_TARGET[1])
    P = lam
    K = [None] * UNI_T
    for t in range(UNI_T - 1, -1, -1):
        K[t] = -np.linalg.pinv(UNI_G.T @ P @ UNI_G) @ UNI_G.T @ P @ Fs[t]
        P = (lam if t > 0 else 0.0) + Fs[t].T @ P @ (Fs[t] + UNI_G @ K[t])
    C, leaf = np.zeros((3, 3)), 0.0
    for t in range(UNI_T):
        A = Fs[t] + UNI_G @ K[t]
        C = A @ C @ A.T + priors[t] - posts[t]
        leaf += _gauss_ce(means[t + 1], posts[t] + C, *UNI_TARGET)
    return leaf, np.asarray(K), np.asarray(means)


def _uni_monte_carlo(gains, means, n=200_000, seed=2):
    """The corrected plan executed in the true nonlinear system with an extended Kalman
    filter: the realised state cost per draw."""
    rng = np.random.default_rng(seed)
    Q = UNI_Q @ UNI_Q.T
    Hr = np.array([[0.0, 1.0, 0.0]])
    x = UNI_M0 + rng.standard_normal((n, 3)) @ UNI_S0.T
    est = np.broadcast_to(UNI_M0, (n, 3)).copy()
    sig = np.broadcast_to(UNI_S0 @ UNI_S0.T, (n, 3, 3)).copy()
    cost = np.zeros(n)
    for t in range(UNI_T):
        u = UNI_PLAN[t] + (est - means[t]) @ gains[t].T
        J = np.zeros((n, 3, 3))
        J[:, 0, 0] = J[:, 1, 1] = J[:, 2, 2] = 1.0
        J[:, 0, 2] = -UNI_H * UNI_V * np.sin(est[:, 2])
        J[:, 1, 2] = UNI_H * UNI_V * np.cos(est[:, 2])
        x = _uni_f(x, u) + rng.standard_normal((n, 3)) @ UNI_Q.T
        est = _uni_f(est, u)
        sig = J @ sig @ np.swapaxes(J, 1, 2) + Q
        y = x[:, 1:2] + UNI_READ * rng.standard_normal((n, 1))
        S = sig[:, 1, 1] + UNI_READ**2
        gain = sig[:, :, 1] / S[:, None]
        est = est + gain * (y - est[:, 1:2])
        sig = sig - gain[:, :, None] * (Hr @ sig)
        cost -= _log_gauss(x, *UNI_TARGET)
    return cost


@pytest.fixture(scope="module")
def unicycle_mc():
    leaf, gains, means = _uni_reference()
    return leaf, _uni_monte_carlo(gains, means)


@pytest.mark.parametrize(
    ("propagation", "tolerance"), [("first_order", 0.005), ("unscented", 0.005)]
)
def test_the_unicycle_scores_its_corrected_closed_loop(unicycle_mc, propagation, tolerance):
    """The linearised closed-loop score against Monte Carlo of the corrected plan executed in
    the true nonlinear system with an extended Kalman filter. Under ``first_order`` the score is
    the numpy linearisation's to 1e-4; against the simulation it is within 0.5 % under either
    propagation (measured: 0.17 % first order, 0.20 % unscented, whose statistical linearisation
    and sigma-point moments differ from the extended filter's). The tolerance is the
    linearisation's error, not sampling: the heading's spread is
    about 0.1 rad, where ``sin`` departs from its tangent by ``θ³/6`` and the cosine's curvature
    biases the lateral mean by about ``h v θ²/2`` per step, a fraction of a percent of the cost
    here; the Monte Carlo standard error is two orders smaller."""
    leaf, draws = unicycle_mc
    system, params = unicycle(propagation)
    got = _leaf(_score(system, params, _f32(UNI_PLAN)), "risk", "pose")
    if propagation == "first_order":
        assert got == pytest.approx(leaf, rel=1e-4)
    se = draws.std() / np.sqrt(draws.size)
    assert se < 1e-3 * abs(got)
    assert got == pytest.approx(draws.mean(), rel=tolerance)


# -------------------------------------------------- the two decompositions, moved alike


@pytest.mark.parametrize("epistemic", [True, ("obs:z",), ("x:dim2",)])
def test_the_switch_moves_both_decompositions_alike(epistemic):
    """The closed-loop risk of an observation target is ``−utility_cl − H[q(y)]``: per channel,
    ``risk + utility`` is the same with and without the switch (``−H[q(y)]``, the predictive
    entropy), and the gap between the two forms' G is what it is open loop (under
    ``planning_belief="posterior"`` the forms read different chains, so the gap is not zero;
    design: per-variable-infogain.md V11). Full selection, a channel's leaf alone, and a
    sub-block."""
    trees = {}
    for form in ("utility_infogain", "risk_ambiguity"):
        for fb in ("linear_quadratic", None):
            system, params = small_lg(form, fb, epistemic)
            trees[form, fb] = _score(system, params, _f32(PLAN), SMALL_PRESENT)
    for fb in ("linear_quadratic", None):
        assert "utility" not in trees["risk_ambiguity", fb]
    shift = _leaf(trees["utility_infogain", "linear_quadratic"], "utility", "y") - _leaf(
        trees["utility_infogain", None], "utility", "y"
    )
    risk_shift = _leaf(trees["risk_ambiguity", "linear_quadratic"], "risk", "y") - _leaf(
        trees["risk_ambiguity", None], "risk", "y"
    )
    assert shift > 1e-3  # feedback narrows the reading, so its expected log-preference rises
    assert risk_shift == pytest.approx(-shift, rel=1e-5)
    gap = {
        fb: float(
            total(trees["utility_infogain", fb]).lower - total(trees["risk_ambiguity", fb]).lower
        )
        for fb in ("linear_quadratic", None)
    }
    assert gap["linear_quadratic"] == pytest.approx(gap[None], abs=1e-4)
    for form in ("utility_infogain", "risk_ambiguity"):
        for term in ("infogain", "ambiguity", "observation_interaction", "observation_correlation"):
            a = trees[form, "linear_quadratic"].get(term, {})
            b = trees[form, None].get(term, {})
            assert set(a) == set(b)
            for key in a:
                assert float(a[key].lower) == float(b[key].lower)


# ---------------------------------------------------------------------------- refusals


def test_feedback_under_the_prior_belief_is_refused():
    with pytest.raises(ValueError, match='planning_belief="posterior"'):
        dataclasses.replace(small_lg()[0], planning_belief="prior")
    with pytest.raises(ValueError, match="feedback must be None or one of"):
        dataclasses.replace(small_lg()[0], feedback="covariance_steering")


def test_a_target_on_a_mixture_carried_factor_is_refused():
    system = System(
        A_deps=((0,),),
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("u",),
        factor_names=("x", "u_x"),
        modality_names=("y",),
        policy_len=2,
        mixture_k=2,
        planner="gradient",
        planning_belief="posterior",
        feedback="linear_quadratic",
    )
    params = SystemParams(
        transition={
            "x": LinearGaussianTransition(
                F=jnp.eye(1),
                G=jnp.eye(1),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 1)), chol=0.1 * jnp.eye(1)[None], parent_ndims=1
                ),
            )
        },
        initial_state_prior={
            "x": MixtureGaussian(
                log_weights=jnp.log(jnp.asarray([0.5, 0.5])),
                means=jnp.asarray([[-1.0], [1.0]]),
                chols=jnp.stack([0.2 * jnp.eye(1)] * 2),
            )
        },
        likelihood={
            "y": LinearGaussianObservation(H=jnp.eye(1), c=jnp.zeros(1), noise=0.1 * jnp.eye(1))
        },
        state_targets={"x": Gaussian(loc=jnp.zeros(1), chol=jnp.eye(1))},
        control={"u_x": Box(-jnp.ones(1), jnp.ones(1))},
    )
    with pytest.raises(ValueError, match=r"state_targets\['x'\].*'x'.*mixture"):
        system.planner_context(params, system.initial_priors(params))
    obs_only = dataclasses.replace(
        params,
        state_targets={},
        obs_targets={"y": Gaussian(loc=jnp.zeros(1), chol=jnp.eye(1))},
    )
    with pytest.raises(ValueError, match=r"obs_targets\['y'\].*'x'.*mixture"):
        system.planner_context(obs_only, system.initial_priors(obs_only))


def test_a_mixture_target_is_refused():
    system, params = small_lg()
    target = MixtureGaussian(
        log_weights=jnp.log(jnp.asarray([0.5, 0.5])),
        means=jnp.asarray([[0.5], [1.0]]),
        chols=jnp.stack([0.3 * jnp.eye(1)] * 2),
    )
    params = dataclasses.replace(params, obs_targets={"y": target})
    with pytest.raises(ValueError, match=r"obs_targets\['y'\] is a MixtureGaussian"):
        system.planner_context(params, system.initial_priors(params))


def test_a_neural_transition_under_a_target_is_refused():
    system = System(
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("u",),
        factor_names=("x", "u_x"),
        policy_len=2,
        planner="gradient",
        planning_belief="posterior",
        feedback="linear_quadratic",
        epistemic=False,
    )
    params = SystemParams(
        transition={"x": MLPConditional.from_shapes((2, 1), 2, key=KEY, width=4, depth=1)},
        initial_state_prior={"x": Gaussian(loc=jnp.zeros(2), chol=jnp.eye(2))},
        state_targets={"x": Gaussian(loc=jnp.zeros(2), chol=jnp.eye(2))},
        control={"u_x": Box(-jnp.ones(1), jnp.ones(1))},
    )
    with pytest.raises(ValueError, match=r"'x'.*neural"):
        system.planner_context(params, system.initial_priors(params))


def test_the_amortized_path_is_refused():
    from tests.test_amortized_objective import agent

    system, params, _ = agent(planning_belief="posterior", feedback="linear_quadratic")
    with pytest.raises(ValueError, match="feedback='linear_quadratic' on the amortized"):
        system.planner_context(params, system.initial_priors(params), key=KEY)


# ------------------------------------------------------------------------- description


def test_belief_and_feedback_round_trip_through_a_description():
    system, params, present = lane(0.2, True)
    document, arrays = describe(system, params)
    planning = document["configuration"]["planning"]
    assert planning["belief"] == "posterior"
    assert planning["feedback"] == "linear_quadratic"
    rebuilt, rebuilt_params = build(document, arrays=arrays)
    assert rebuilt.planning_belief == "posterior"
    assert rebuilt.feedback == "linear_quadratic"
    plan = jnp.full((LANE_H, 1), 0.1)
    a = _score(system, params, plan, present)
    b = _score(rebuilt, rebuilt_params, plan, present)
    assert float(total(a).lower) == pytest.approx(float(total(b).lower), rel=1e-6)
    # the defaults are not written
    open_loop, _ = small_lg(feedback=None)
    plain = dataclasses.replace(open_loop, planning_belief="prior")
    planning = describe(plain, params_of_small())[0]["configuration"]["planning"]
    assert "belief" not in planning and "feedback" not in planning


def params_of_small():
    return small_lg()[1]


def test_a_posterior_system_rebuilds_as_posterior():
    """``describe`` writes ``planning_belief``: a System planning under ``"posterior"`` is
    rebuilt under it, and plans as it did."""
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "examples" / "sensor_detour.py"
    spec = importlib.util.spec_from_file_location("sensor_detour", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    system, params = module.make_agent(planning_belief="posterior")
    d = describe(system, params)
    rebuilt, rebuilt_params = build(d.document, arrays=d.arrays)
    assert rebuilt.planning_belief == "posterior"
    beliefs = system.initial_priors(params)
    plan = jnp.zeros((system.policy_len, 2))
    a = system.score_policies(params, beliefs, plan)
    b = rebuilt.score_policies(rebuilt_params, rebuilt.initial_priors(rebuilt_params), plan)
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-6, atol=1e-6)


def test_feedback_without_the_posterior_belief_is_refused_by_build():
    from hybrid_aif import DescriptionError

    system, params, _ = lane(0.2, True)
    document, arrays = describe(system, params)
    del document["configuration"]["planning"]["belief"]
    with pytest.raises(DescriptionError, match='"belief": "posterior"'):
        build(document, arrays=arrays)


# ------------------------------------------------------------------- the gradient planner


@pytest.mark.parametrize("model", ["lane", "unicycle"])
def test_the_gradient_planner_descends_the_closed_loop_score(model):
    if model == "lane":
        system, params, present = lane(0.05, True)
        params = dataclasses.replace(
            params,
            initial_state_prior={
                "lane": Gaussian(
                    loc=jnp.asarray([0.3, 0.0, 0.0]), chol=params.initial_state_prior["lane"].chol
                )
            },
        )
    else:
        system, params = unicycle("first_order")
        present = None
    beliefs = system.initial_priors(params)
    ctx = system.planner_context(params, beliefs, present=present)

    def g(plan):
        tree = evaluate_policies(
            plan[None],
            ctx.init_beliefs,
            ctx.transition_fn,
            ctx.step_terms_fn,
            ctx.terminal_terms_fn,
        )
        return total(tree).midpoint[0]

    start = jnp.zeros((system.policy_len, 1))
    grad = jax.grad(g)(start)
    assert np.all(np.isfinite(np.asarray(grad))) and float(jnp.abs(grad).max()) > 0.0
    result = gradient_plan(ctx, steps=150, init=start)
    assert np.all(np.isfinite(np.asarray(result.plan)))
    assert float(result.G.midpoint[0]) < float(g(start)) - 1e-3


# -------------------------------------------------- where a nonlinear factor is linearised


def test_a_nonlinear_factor_is_linearised_at_the_pre_step_belief():
    """The recorded ``F`` of a step is the Jacobian of ``f`` at the belief the step starts from,
    not at the belief it reaches: the first turn moves the heading by ``h ω``, where the two
    Jacobians differ in their heading column."""
    system, params = unicycle("first_order")
    ctx = system.planner_context(params, system.initial_priors(params))
    _, record = ctx.transition_fn(ctx.init_beliefs, _f32(UNI_PLAN[0]))
    F = np.asarray(record.F[0])[:3, :3]

    def jacobian(theta):
        J = np.eye(3)
        J[0, 2] = -UNI_H * UNI_V * np.sin(theta)
        J[1, 2] = UNI_H * UNI_V * np.cos(theta)
        return J

    before, after = UNI_M0[2], UNI_M0[2] + UNI_H * float(UNI_PLAN[0, 0])
    assert abs(jacobian(before)[0, 2] - jacobian(after)[0, 2]) > 1e-3
    np.testing.assert_allclose(F, jacobian(before), atol=1e-6)


# -------------------------------------------------- composition keeps the planning options


def test_merge_keeps_the_planning_belief_and_the_feedback_switch():
    from hybrid_aif.composition import merge

    def mind(name):
        system = System(
            B_deps=((),),
            controlled=(True,),
            factor_names=(name,),
            policy_len=2,
            planner="gradient",
            epistemic=False,
            planning_belief="posterior",
            feedback="linear_quadratic",
        )
        params = SystemParams(
            transition={
                name: LinearGaussianTransition(
                    F=jnp.eye(1),
                    innovations=ConditionalGaussian(
                        loc=jnp.zeros((1, 1)), chol=0.1 * jnp.ones((1, 1, 1)), parent_ndims=1
                    ),
                    G=jnp.ones((1, 1)),
                )
            },
            initial_state_prior={name: Gaussian(loc=jnp.zeros(1), chol=jnp.eye(1))},
            control={f"u_{name}": Box(-jnp.ones(1), jnp.ones(1))},
        )
        return system, params

    a, ap = mind("x")
    b, bp = mind("y")
    composite, _ = merge({"a": a, "b": b}, {"a": ap, "b": bp})
    assert composite.planning_belief == "posterior"
    assert composite.feedback == "linear_quadratic"
    open_loop = dataclasses.replace(b, feedback=None)
    with pytest.raises(ValueError, match="disagree on feedback"):
        merge({"a": a, "b": open_loop}, {"a": ap, "b": bp})
