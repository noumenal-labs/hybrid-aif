"""Acceptance-fix 4: per-mode F_j in LinearGaussianTransition (the SLDS pattern).

``F`` may be ``(d, d)`` — shared across modes, current behavior bitwise — or
``(*parent_shape, d, d)`` matching ``innovations.parent_shape``, sliced by the
same mode/control index as the innovations everywhere: the mixture grow
composition, the Gaussian ``predict`` path, and the hybrid generative process's
sampling. Shape mismatches fail loudly at ``validate_structure`` time.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalGaussian,
    Gaussian,
)
from hybrid_aif.factors import LinearGaussianTransition
from hybrid_aif.inference.linear_gaussian import predict
from hybrid_aif.inference.mixture_rollout import propagate
from hybrid_aif.objectives.hybrid import pad_mixture
from hybrid_aif.system import System, SystemParams
from tests.test_hybrid_switching_system import (
    ACTION_MODE_LOGW,
    INIT_BELIEF,
    INNOVATIONS,
    K_BUDGET,
    initial_beliefs,
    make_params,
    make_system,
)

F_SHARED = jnp.asarray([[0.9]])
F_MODES = jnp.asarray([[[0.6]], [[1.4]]])  # (M=2, 1, 1): genuinely different per mode


def slds_params(F):
    """The switching fixture's model with the given F (shared or per-mode)."""
    ref = make_params()
    return dataclasses.replace(
        ref,
        transition={
            **ref.transition,
            "x": LinearGaussianTransition(F=F, innovations=INNOVATIONS),
        },
    )


# ------------------------------------------------------------- shared-F regression


def test_shared_f_stack_matches_shared_matrix():
    """An F_j stack whose modes are all the shared matrix reproduces the
    shared-F results. (The shared ``(d, d)`` branch itself is the pre-change
    code verbatim — its bitwise regression guard is the untouched switching
    suite; across the two *representations* only compiled-graph reassociation
    differs, last-ulp.)"""
    system = make_system()
    _, tree_shared = system.infer_policies(make_params(), initial_beliefs())
    stacked = jnp.broadcast_to(F_SHARED, (2, 1, 1))
    _, tree_stacked = system.infer_policies(slds_params(stacked), initial_beliefs())
    for term, blocks in tree_shared.items():
        for block, iv in blocks.items():
            np.testing.assert_allclose(
                np.asarray(tree_stacked[term][block].lower),
                np.asarray(iv.lower),
                atol=1e-5,
                err_msg=f"{term}/{block}",
            )
            np.testing.assert_allclose(
                np.asarray(tree_stacked[term][block].upper),
                np.asarray(iv.upper),
                atol=1e-5,
                err_msg=f"{term}/{block}",
            )


# ------------------------------------------------------------ mixture (grow) path


def test_per_mode_grow_matches_manual_propagate():
    system, params = make_system(), slds_params(F_MODES)
    ctx = system.planner_context(params, initial_beliefs())
    for u in range(3):
        action = jnp.asarray([u, 0], jnp.int32)
        mode_next, x_next = ctx.transition_fn(ctx.init_beliefs, action)
        # mode weights are the parent's post-transition belief this step
        logw = Categorical(logits=ACTION_MODE_LOGW[u]).log_probs
        np.testing.assert_allclose(np.asarray(mode_next.log_probs), np.asarray(logw), atol=1e-6)
        ref = propagate(
            pad_mixture(INIT_BELIEF, K_BUDGET), F_MODES, INNOVATIONS.loc, INNOVATIONS.chol, logw
        ).reduce(K_BUDGET)
        np.testing.assert_allclose(np.asarray(x_next.means), np.asarray(ref.means), atol=1e-5)
        np.testing.assert_allclose(np.asarray(x_next.chols), np.asarray(ref.chols), atol=1e-5)
        np.testing.assert_allclose(
            np.asarray(jnp.exp(x_next.log_weights)), np.asarray(jnp.exp(ref.log_weights)), atol=1e-5
        )


def test_per_mode_f_ranking_differs_from_shared():
    """The point of SLDS: different F_j must actually change the scores."""
    system = make_system()
    q_shared, _ = system.infer_policies(make_params(), initial_beliefs())
    q_modes, _ = system.infer_policies(slds_params(F_MODES), initial_beliefs())
    assert not np.allclose(np.asarray(q_shared.logits), np.asarray(q_modes.logits), atol=1e-3)


# --------------------------------------------------- Gaussian (predict) path


F_CTRL = jnp.asarray([[[0.5]], [[1.0]], [[1.5]]])  # (num_controls=3, 1, 1)
INNOV_CTRL = ConditionalGaussian(
    loc=jnp.asarray([[-0.1], [0.0], [0.1]]),
    chol=jnp.full((3, 1, 1), 0.05),
    parent_ndims=1,
)
GOAL_1D = Gaussian(loc=jnp.asarray([1.0]), chol=jnp.asarray([[0.1]]))


def ctrl_system_and_params(F=F_CTRL):
    system = System(
        B_deps=((),),
        controlled=(True,),
        policy_len=2,
        factor_names=("state",),
        modality_names=(),
    )
    params = SystemParams(
        transition={"state": LinearGaussianTransition(F=F, innovations=INNOV_CTRL)},
        initial_state_prior={"state": Gaussian(loc=jnp.zeros(1), chol=0.2 * jnp.eye(1))},
        state_targets={"state": GOAL_1D},
    )
    return system, params


def test_per_control_f_predict_path_matches_manual():
    system, params = ctrl_system_and_params()
    belief = Gaussian(loc=jnp.asarray([0.4]), chol=0.15 * jnp.eye(1))
    ctx = system.planner_context(params, {"state": belief})
    for u in range(3):
        (pushed,) = ctx.transition_fn((belief,), jnp.asarray([u], jnp.int32))
        innov = INNOV_CTRL.condition_on((u,))
        ref = predict(belief, F_CTRL[u], innov.loc, innov.chol)
        np.testing.assert_allclose(np.asarray(pushed.loc), np.asarray(ref.loc), atol=1e-6)
        np.testing.assert_allclose(np.asarray(pushed.chol), np.asarray(ref.chol), atol=1e-6)


# ------------------------------------------------------------------ env sampling


def test_env_step_slices_f_by_the_sampled_mode():
    # deterministic mode CPD and near-zero innovation noise isolate F_j x + b_j
    det_mode = ConditionalCategorical(
        log_table=jnp.log(jnp.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])).T, parent_ndims=1
    )
    tiny = ConditionalGaussian(loc=INNOVATIONS.loc, chol=jnp.full((2, 1, 1), 1e-6), parent_ndims=1)
    ref = make_params()
    params = dataclasses.replace(
        ref,
        transition={"mode": det_mode, "x": LinearGaussianTransition(F=F_MODES, innovations=tiny)},
    )
    env, env_params = make_system().as_env(params)
    state = {"mode": jnp.asarray(0, jnp.int32), "x": jnp.asarray([0.5])}
    for u, (f_j, b_j) in enumerate(zip((0.6, 1.4), (-1.2, 1.2), strict=True)):
        action = {"a_mode": jnp.asarray(u, jnp.int32)}
        _, nxt = env.step(jax.random.PRNGKey(u), state, action, env_params)
        assert int(nxt["mode"]) == u  # deterministic mode draw
        assert np.isclose(float(nxt["x"][0]), f_j * 0.5 + b_j, atol=1e-3)


def test_env_step_slices_f_by_the_control():
    system, params = ctrl_system_and_params()
    env, env_params = system.as_env(params)
    state = {"state": jnp.asarray([2.0])}
    for u, f_u in enumerate((0.5, 1.0, 1.5)):
        action = {"a_state": jnp.asarray(u, jnp.int32)}
        _, nxt = env.step(jax.random.PRNGKey(u), state, action, env_params)
        b_u = float(INNOV_CTRL.loc[u, 0])
        assert np.isclose(float(nxt["state"][0]), f_u * 2.0 + b_u, atol=0.3)  # 0.05-scale noise


# -------------------------------------------------------------------- validation


def test_wrong_f_shape_fails_loudly():
    system = make_system()
    bad = slds_params(jnp.zeros((3, 1, 1)))  # parent_shape is (2,), not (3,)
    with pytest.raises(ValueError, match="F shape"):
        system.planner_context(bad, initial_beliefs())
    system_ctrl, params_ctrl = ctrl_system_and_params(F=jnp.zeros((2, 1, 1)))
    with pytest.raises(ValueError, match="F shape"):
        system_ctrl.planner_context(
            params_ctrl, {"state": params_ctrl.initial_state_prior["state"]}
        )
