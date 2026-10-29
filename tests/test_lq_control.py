"""The linear-quadratic demo: the gradient planner's plan is the LQ regulator's.

``examples/lq_control.py`` declares a double integrator with a continuous control node read
through ``G u``, Gaussian observation targets and a Gaussian target on the node. On that model the
expected free energy is, up to terms the plan does not move,

    J(u) = Σ_{t=1..H} ½ (H x_t + c − m)ᵀ S⁻¹ (H x_t + c − m) + Σ_{t=0..H−1} ½ u_tᵀ R u_t

over the mean path ``x_{t+1} = F x_t + G u_t + b``: the utility's mean term (``C(y) = N(m, S)``)
and the control target (``C(u) = N(0, R⁻¹)``). The tests solve it with the finite-horizon Riccati
recursion written out below and compare.

Design: docs/design/control-states.md C4 (The planner for the LQ demo).
"""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.objectives import total
from hybrid_aif.planning import cem_plan, gradient_plan, make_gradient_planner

_SPEC = importlib.util.spec_from_file_location(
    "lq_control", Path(__file__).resolve().parents[1] / "examples" / "lq_control.py"
)
lq = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lq)

KEY = jax.random.PRNGKey(0)
#: The gradient planner against Riccati, and the Laplace covariance against the inverse
#: curvature (float32 throughout).
TIGHT = 1e-4
#: CEM, a sampling search, against Riccati.
LOOSE = 5e-2


def _matrices(params):
    """``F, G, b, H, c, m, S⁻¹, R`` of the declared model (float64)."""
    B = params.transition["x"]
    A = params.likelihood["y"]
    target, u_target = params.obs_targets["y"], params.state_targets["u_u"]
    F, G = np.asarray(B.F, np.float64), np.asarray(B.G, np.float64)
    b = np.asarray(B.innovations.loc[0], np.float64)
    H, c = np.asarray(A.H, np.float64), np.asarray(A.c, np.float64)
    m = np.asarray(target.loc, np.float64)
    S_inv = np.linalg.inv(np.asarray(target.cov, np.float64))
    R = np.linalg.inv(np.asarray(u_target.cov, np.float64))
    return F, G, b, H, c, m, S_inv, R


def riccati_plan(params, x0, horizon: int) -> np.ndarray:
    """The open-loop optimal plan by the finite-horizon Riccati recursion.

    Cost-to-go ``V_t(x) = ½ xᵀ P_t x + p_tᵀ x``. The state cost at ``t ≥ 1`` is
    ``½ xᵀ Q x + qᵀ x`` with ``Q = Hᵀ S⁻¹ H``, ``q = Hᵀ S⁻¹ (c − m)``; ``V_H`` is that cost.
    Backwards, with ``W = R + Gᵀ P G`` and the affine input ``z = F x + b``:
    ``u_t = −W⁻¹ Gᵀ (P (F x + b) + p) = K_t x + k_t``, and ``V_t`` collects the minimised
    quadratic plus the state cost at ``t`` (none at ``t = 0``). Forwards from ``x_0``, the
    gains give the plan.
    """
    F, G, b, H, c, m, S_inv, R = _matrices(params)
    Q = H.T @ S_inv @ H
    q = H.T @ S_inv @ (c - m)
    P, p = Q.copy(), q.copy()
    gains = []
    for t in range(horizon - 1, -1, -1):
        W = R + G.T @ P @ G
        K = -np.linalg.solve(W, G.T @ P @ F)
        k = -np.linalg.solve(W, G.T @ (P @ b + p))
        gains.append((K, k))
        A_cl = F + G @ K  # closed loop: x' = A_cl x + (G k + b)
        e = G @ k + b
        P_next = A_cl.T @ P @ A_cl + K.T @ R @ K
        p_next = A_cl.T @ (P @ e + p) + K.T @ R @ k
        if t >= 1:
            P_next, p_next = P_next + Q, p_next + q
        P, p = P_next, p_next
    gains.reverse()
    x, plan = np.asarray(x0, np.float64), []
    for K, k in gains:
        u = K @ x + k
        plan.append(u)
        x = F @ x + G @ u + b
    return np.stack(plan)


def lq_hessian(params, horizon: int) -> np.ndarray:
    """``∇²J`` over the flattened plan: ``blockdiag(R) + Σ_t Φ_tᵀ Q Φ_t`` with ``Φ_t`` the map
    from the plan to ``x_t``."""
    F, G, _, H, _, _, S_inv, R = _matrices(params)
    du = G.shape[1]
    Q = H.T @ S_inv @ H
    M = np.kron(np.eye(horizon), R)
    for t in range(1, horizon + 1):
        Phi = np.zeros((F.shape[0], horizon * du))
        for s in range(t):
            Phi[:, s * du : (s + 1) * du] = np.linalg.matrix_power(F, t - 1 - s) @ G
        M += Phi.T @ Q @ Phi
    return M


@pytest.fixture(scope="module")
def model():
    system, params = lq.make_lq()
    beliefs = system.initial_priors(params)
    ctx = system.planner_context(params, beliefs)
    return system, params, beliefs, ctx


def test_the_gradient_plan_is_the_riccati_plan(model):
    system, params, beliefs, ctx = model
    want = riccati_plan(params, np.asarray(beliefs["x"].loc), system.policy_len)
    #  plain gradient steps at 1/λ_max of the curvature: a contraction to the minimiser
    lr = 1.0 / np.linalg.eigvalsh(lq_hessian(params, system.policy_len)).max()
    got = gradient_plan(ctx, optimizer="gradient", lr=float(lr), steps=400)
    np.testing.assert_allclose(np.asarray(got.plan), want, atol=TIGHT)
    #  the plan is interior: the box does not bind
    assert np.all(np.abs(want) < lq.BOX / 2)
    #  the default (Adam, step scaled to the box) lands close too, and its trace descends
    adam = gradient_plan(ctx)
    np.testing.assert_allclose(np.asarray(adam.plan), want, atol=1e-3)
    assert float(adam.trace[-1]) < float(adam.trace[0])


@pytest.mark.parametrize("start", ["low face", "high face", "inside", "alternating faces"])
def test_the_default_step_crosses_the_box(model, start):
    """Adam's default step is scaled to the box width, so 200 steps reach the mode from either
    face of a 20-wide box, or from a far start."""
    system, params, beliefs, ctx = model
    want = riccati_plan(params, np.asarray(beliefs["x"].loc), system.policy_len)
    init = {
        "low face": -lq.BOX,
        "high face": lq.BOX,
        "inside": 5.0,
        "alternating faces": jnp.asarray([[-lq.BOX], [lq.BOX], [-lq.BOX], [lq.BOX]]),
    }[start]
    got = gradient_plan(ctx, init=init)
    np.testing.assert_allclose(np.asarray(got.plan), want, atol=1e-3)


def test_cem_lands_near_the_riccati_plan(model):
    system, params, beliefs, ctx = model
    want = riccati_plan(params, np.asarray(beliefs["x"].loc), system.policy_len)
    got = cem_plan(ctx, KEY, num_candidates=256, cem_iters=12, min_std=1e-3)
    np.testing.assert_allclose(np.asarray(got.mean), want, atol=LOOSE)


def test_the_laplace_covariance_is_the_inverse_curvature(model):
    system, params, _, ctx = model
    assert system.gamma == 1.0  # q(π) ∝ exp(−G): the covariance is (∇²G)⁻¹ itself
    got = gradient_plan(ctx, optimizer="gradient", lr=0.05, steps=50, posterior="laplace")
    want = np.linalg.inv(lq_hessian(params, system.policy_len))
    np.testing.assert_allclose(np.asarray(got.curvature.cov), want, atol=TIGHT, rtol=TIGHT)
    np.testing.assert_array_equal(np.asarray(got.curvature.loc), np.asarray(got.plan).ravel())
    #  the emitted posterior is that Gaussian restricted to the box, per step
    np.testing.assert_allclose(
        np.asarray(got.posterior.scale) ** 2, np.diag(want), atol=TIGHT, rtol=TIGHT
    )


def test_the_epistemic_term_is_constant_along_the_plan(model):
    """On the LG path the predicted covariance does not depend on the plan, so the info gain
    is the same for every plan: the gradient is pragmatic."""
    system, params, beliefs, _ = model
    plans = jnp.stack(
        [
            jnp.zeros((system.policy_len, 1)),
            jnp.full((system.policy_len, 1), 3.0),
            jax.random.uniform(KEY, (system.policy_len, 1), minval=-5.0, maxval=5.0),
        ]
    )
    tree = system.score_policies(params, beliefs, plans)
    infogain = tree["infogain"]
    assert infogain, "the epistemic term is mounted"
    for leaf in jax.tree_util.tree_leaves(infogain):
        leaf = np.asarray(leaf)
        np.testing.assert_allclose(leaf, np.broadcast_to(leaf[:1], leaf.shape), rtol=1e-6)
        assert np.any(leaf != 0.0)


def test_the_planner_faces_emit_the_mode_or_a_laplace_draw(model):
    system, params, beliefs, _ = model
    want = riccati_plan(params, np.asarray(beliefs["x"].loc), system.policy_len)
    mode = system.act(KEY, params, beliefs, selection="deterministic")[0]["u"]
    np.testing.assert_allclose(np.asarray(mode), want[0], atol=1e-3)
    #  under "mode" the stochastic selection emits the mode too: one plan, a point mass
    q_pi, tree = system.infer_policies(params, beliefs)
    assert q_pi.probs.shape == (1,) and set(tree) >= {"utility", "risk", "infogain"}
    again = system.act(KEY, params, beliefs)[0]["u"]
    np.testing.assert_array_equal(np.asarray(again), np.asarray(mode))
    #  under "laplace" a stochastic act draws from the posterior's first step, in the box
    laplace = make_gradient_planner(posterior="laplace")
    draws = np.stack(
        [
            np.asarray(system.act(k, params, beliefs, planner=laplace)[0]["u"])
            for k in jax.random.split(KEY, 12)
        ]
    )
    assert draws.shape == (12, 1) and np.all(np.abs(draws) <= lq.BOX)
    assert np.std(draws) > 0.1 and abs(np.mean(draws) - want[0, 0]) < 0.5
    det = system.act(KEY, params, beliefs, planner=laplace, selection="deterministic")[0]["u"]
    np.testing.assert_allclose(np.asarray(det), want[0], atol=1e-3)


def test_a_plan_with_no_strict_minimum_has_no_laplace_posterior():
    """With no targets, G does not depend on the plan at all: its curvature is
    zero, and the Laplace posterior is refused."""
    import dataclasses

    system, params = lq.make_lq()
    no_targets = dataclasses.replace(params, obs_targets={}, state_targets={})
    ctx = system.planner_context(no_targets, system.initial_priors(no_targets))
    with pytest.raises(ValueError, match="not positive definite"):
        gradient_plan(ctx, steps=5, posterior="laplace")


def test_the_gradient_planner_refuses_a_discrete_space():
    from tests.test_system import tmaze_system

    system, params = tmaze_system()
    with pytest.raises(ValueError, match="control nodes are discrete"):
        system.infer_policies(
            params, system.initial_priors(params), planner=make_gradient_planner()
        )


def test_score_policies_reads_the_plan_keyed_by_node(model):
    """The keyed form of a continuous plan scores as the positional one."""
    system, params, beliefs, _ = model
    plan = jnp.linspace(-1.0, 1.0, system.policy_len)[:, None]
    keyed = total(system.score_policies(params, beliefs, {"u_u": plan}))
    positional = total(system.score_policies(params, beliefs, plan))
    np.testing.assert_array_equal(np.asarray(keyed.midpoint), np.asarray(positional.midpoint))
    expected = 0.5 * float(jnp.sum(plan**2)) / lq.CONTROL_TARGET_SD**2
    risk = system.score_policies(params, beliefs, plan)["risk"]["u_u"]
    #  the control target's leaf is −log C(u): the quadratic plus the Gaussian's normaliser
    log_norm = 0.5 * np.log(2.0 * np.pi * lq.CONTROL_TARGET_SD**2)
    np.testing.assert_allclose(
        float(jnp.sum(risk.midpoint)), expected + system.policy_len * log_norm, rtol=1e-5
    )


def test_a_binding_box_gives_laplace_draws_inside_it_not_piled_on_the_face():
    """With a box narrower than the unconstrained first step, the mode sits on the face; the
    posterior is the curvature's Gaussian restricted to the box, so draws stay inside without
    clipping and almost none land exactly on the face."""
    system, params = lq.make_lq(box=0.5)
    beliefs = system.initial_priors(params)
    ctx = system.planner_context(params, beliefs)
    got = gradient_plan(ctx, posterior="laplace")
    assert float(got.plan[0, 0]) == pytest.approx(0.5)  # the mode on the face
    draws = np.asarray(jax.vmap(got.posterior.sample)(jax.random.split(KEY, 400)))[:, 0]
    assert np.all((draws >= -0.5) & (draws <= 0.5))
    assert np.mean(draws == 0.5) < 0.01 and np.std(draws) > 0.05
    laplace = make_gradient_planner(posterior="laplace")
    action = system.act(KEY, params, beliefs, planner=laplace)[0]["u"]
    assert -0.5 <= float(action[0]) <= 0.5


def test_a_jitted_laplace_act_without_curvature_raises():
    """The positive-definite check fires under a trace too (``equinox.error_if``)."""
    import dataclasses

    system, params = lq.make_lq()
    no_targets = dataclasses.replace(params, obs_targets={}, state_targets={})
    beliefs = system.initial_priors(no_targets)
    laplace = make_gradient_planner(posterior="laplace", steps=5)

    @jax.jit
    def act(p, b):
        return system.act(KEY, p, b, planner=laplace)[0]["u"]

    with pytest.raises(Exception, match="not positive definite"):
        jax.block_until_ready(act(no_targets, beliefs))
