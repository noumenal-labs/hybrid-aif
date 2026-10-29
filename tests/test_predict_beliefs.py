"""predict_beliefs: the open-loop imagination face, both paths."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_amortized_objective import agent as amortized_agent
from tests.test_hybrid_switching_system import make_params, make_system
from tests.test_system import _initial_beliefs, tmaze_system

KEY = jax.random.PRNGKey(0)


def _stepwise(system, params, beliefs, policy, key):
    """The reference rollout: one ``update_empirical_prior`` per step, in Python."""
    keys = None if key is None else jax.random.split(key, policy.shape[0])
    out, qs = [], beliefs
    for t in range(policy.shape[0]):
        qs = system.update_empirical_prior(
            params,
            qs,
            system.from_positional("actions", policy[t], params),
            key=None if key is None else keys[t],
        )
        out.append(qs)
    return out


def _leaves_equal(a, b, tol: float = 1e-6) -> None:
    """Within ``tol`` (one float32 ulp covers scan-body fusion against the eager loop)."""
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb, strict=True):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=tol, atol=tol)


def test_discrete_matches_update_empirical_prior_chain():
    system, params = tmaze_system()
    beliefs = _initial_beliefs(system, params)
    policy = system.policies(params)[3]  # one (H, K) plan from the enumeration
    steps = system.predict_beliefs(params, beliefs, policy)
    qs = beliefs
    for t, got in enumerate(steps):
        qs = system.update_empirical_prior(params, qs, system.from_positional("actions", policy[t]))
        assert list(got) == list(qs)
        for name in got:
            assert jnp.array_equal(got[name].probs, qs[name].probs)


def test_hybrid_trajectory_matches_planner_transition_closure():
    system, params = make_system(), make_params()
    policy = jnp.asarray([[2], [2], [2]], jnp.int32)  # commit-right
    start = dict(params.initial_state_prior)
    steps = system.predict_beliefs(params, start, policy)
    ctx = system.planner_context(params, start)
    qs = ctx.init_beliefs
    for t, got in enumerate(steps):
        qs = ctx.transition_fn(qs, policy[t])
        assert jnp.allclose(got["x"].means, qs[1].means, atol=1e-6)
        assert jnp.allclose(got["x"].log_weights, qs[1].log_weights, atol=1e-6)


@pytest.mark.parametrize("keyed", [False, True])
def test_hybrid_path_matches_the_stepwise_loop(keyed):
    """Against the per-step reference, from an un-padded start (K=2 -> mixture_k): bitwise
    keyless; the sampled rollout's collapsed means differ by one float32 ulp."""
    system, params = make_system(policy_len=4), make_params()
    policy = jnp.asarray([[2], [0], [1], [2]], jnp.int32)
    key = KEY if keyed else None
    start = dict(params.initial_state_prior)
    got = system.predict_beliefs(params, start, policy, key=key)
    want = _stepwise(system, params, start, policy, key)
    assert isinstance(got, list) and len(got) == 4
    _leaves_equal(got, want)


@pytest.mark.parametrize("keyed", [False, True])
def test_discrete_path_matches_the_stepwise_loop(keyed):
    system, params = tmaze_system(policy_len=3)
    beliefs = _initial_beliefs(system, params)
    policy = system.policies(params)[5]
    key = KEY if keyed else None
    _leaves_equal(
        system.predict_beliefs(params, beliefs, policy, key=key),
        _stepwise(system, params, beliefs, policy, key),
    )


@pytest.mark.parametrize("keyed", [False, True])
def test_amortized_path_matches_the_stepwise_loop(keyed):
    system, params, _ = amortized_agent()
    beliefs = system.initial_priors(params)
    plan = jnp.full((3, 1), 0.3, jnp.float32)
    key = KEY if keyed else None
    _leaves_equal(
        system.predict_beliefs(params, beliefs, plan, key=key),
        _stepwise(system, params, beliefs, plan, key),
        tol=1e-6,  # the MLP transition's fused matmuls: one float32 ulp
    )


def test_one_step_horizon_is_the_single_update():
    system, params = make_system(), make_params()
    start = dict(params.initial_state_prior)
    (got,) = system.predict_beliefs(params, start, jnp.asarray([[2]]))
    want = system.update_empirical_prior(params, start, {"a_mode": jnp.asarray(2)})
    _leaves_equal(got, want)


def test_policy_shape_and_dtype_validated():
    system, params = tmaze_system()
    beliefs = _initial_beliefs(system, params)
    with pytest.raises(ValueError, match="integer"):
        system.predict_beliefs(params, beliefs, jnp.zeros((2, 2)))
    with pytest.raises(ValueError, match="one column per control node"):
        system.predict_beliefs(params, beliefs, jnp.zeros((2, 2), jnp.int32))
