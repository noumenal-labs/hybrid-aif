"""Hand-checkable hybrid fixture: the switching demo's model as a System.

Mirrors ``examples/hybrid_switching.py``'s constants (replicated here so the
test is hermetic): a controlled discrete mode factor (B parents = control
only, ``B_deps=()``) plus a 1-D mixture factor whose ``B_deps`` name the mode
factor. ``System.infer_policies`` with an injected planner scoring the demo's
6 policies must reproduce the demo's per-policy G Intervals, where the
reference numbers come from the demo's own composition (``rollout_reduced``
per step + risk/entropy Interval sums).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalGaussian,
    MixtureGaussian,
)
from hybrid_aif.factors import LinearGaussianTransition
from hybrid_aif.inference import rollout_reduced
from hybrid_aif.objectives import total
from hybrid_aif.objectives.continuous import risk, uncertainty
from hybrid_aif.planning.enumerate import evaluate_policies, policy_posterior
from hybrid_aif.system import System, SystemParams

# ----------------------------------------------------- demo constants (replicated)

F = jnp.asarray([[0.9]])
INNOVATIONS = ConditionalGaussian(  # mode-indexed N(b_j, Q_j)
    loc=jnp.asarray([[-1.2], [1.2]]),
    chol=jnp.asarray([[[0.3]], [[0.45]]]),
    parent_ndims=1,
)
ACTION_MODE_LOGW = jnp.log(
    jnp.asarray(
        [
            [0.50, 0.50],  # dither
            [0.92, 0.08],  # commit-left
            [0.08, 0.92],  # commit-right
        ]
    )
)
INIT_BELIEF = MixtureGaussian(
    log_weights=jnp.log(jnp.asarray([0.5, 0.5])),
    means=jnp.asarray([[-1.0], [1.0]]),
    chols=jnp.asarray([[[0.6]], [[0.6]]]),
)
TARGET = MixtureGaussian(
    log_weights=jnp.log(jnp.asarray([0.85, 0.15])),
    means=jnp.asarray([[3.0], [-3.0]]),
    chols=jnp.asarray([[[0.8]], [[1.0]]]),
)
HORIZON = 3
K_BUDGET = 4
POLICIES = jnp.asarray(
    [[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 2, 2], [0, 0, 2], [2, 1, 2]], jnp.int32
)
COMMIT_RIGHT = 2  # index of the demo's best policy


def make_system(**overrides):
    defaults = dict(
        A_deps=(),
        B_deps=((), (0,)),
        controlled=(True, False),
        policy_len=HORIZON,
        mixture_k=K_BUDGET,
        modality_names=(),
        factor_names=("mode", "x"),
        terms=("uncertainty:x",),  # x is unobserved: its entropy is the uncertainty row
    )
    return System(**{**defaults, **overrides})


def make_params():
    return SystemParams(
        likelihood={},
        # child-first CPD layout: (num_modes, num_actions)
        transition={
            "mode": ConditionalCategorical(log_table=ACTION_MODE_LOGW.T, parent_ndims=1),
            "x": LinearGaussianTransition(F=F, innovations=INNOVATIONS),
        },
        initial_state_prior={"mode": Categorical(logits=jnp.zeros(2)), "x": INIT_BELIEF},
        obs_targets={},
        state_targets={"x": TARGET},
        terminal_targets={},
    )


def initial_beliefs():
    return {"mode": Categorical(logits=jnp.zeros(2)), "x": INIT_BELIEF}


def six_policy_planner(ctx, use_bound="midpoint", metrics=True):
    """The demo's hand-declared policy set through the injection seam."""
    pols = POLICIES[..., None]  # the mode's control node, one column
    tree = evaluate_policies(
        pols, ctx.init_beliefs, ctx.transition_fn, ctx.step_terms_fn, ctx.terminal_terms_fn
    )
    return (
        policy_posterior(total(tree), ctx.gamma, ctx.policy_prior, use_bound=use_bound),
        tree,
        pols,
    )


def demo_scores(policy):
    """The demo's reduced pipeline, verbatim composition: per-step grow->reduce(k),
    G = risk + belief entropy summed over the horizon's belief sequence."""
    m, g = INIT_BELIEF, None
    for t in range(policy.shape[0]):
        m, _ = rollout_reduced(
            m,
            jnp.broadcast_to(F, (2, 1, 1)),
            INNOVATIONS.loc,
            INNOVATIONS.chol,
            ACTION_MODE_LOGW[policy[t]],
            horizon=1,
            k=K_BUDGET,
        )
        r = risk((m,), {"x": TARGET}, ("x",))
        h = uncertainty((m,), ("x",))
        g_t = total({**r, **h})
        g = g_t if g is None else g + g_t
    return g


def test_infer_policies_reproduces_demo_g_intervals():
    system, params = make_system(), make_params()
    q_pi, tree = system.infer_policies(params, initial_beliefs(), planner=six_policy_planner)
    g = total(tree)
    g_demo = jax.vmap(demo_scores)(POLICIES)
    np.testing.assert_allclose(np.asarray(g.lower), np.asarray(g_demo.lower), atol=1e-4)
    np.testing.assert_allclose(np.asarray(g.upper), np.asarray(g_demo.upper), atol=1e-4)
    assert int(jnp.argmin(g.midpoint)) == COMMIT_RIGHT
    assert int(jnp.argmax(q_pi.probs)) == COMMIT_RIGHT
    # both terms present, one leaf per designated/continuous block, per-policy axes
    assert set(tree) == {"risk", "uncertainty"}
    assert set(tree["risk"]) == {"x"} and set(tree["uncertainty"]) == {"x"}
    assert tree["risk"]["x"].lower.shape == (len(POLICIES),)


def test_infer_policies_jits():
    system, params = make_system(), make_params()
    f = jax.jit(lambda p, b: system.infer_policies(p, b, planner=six_policy_planner)[0].logits)
    logits = f(params, initial_beliefs())
    q_pi, _ = system.infer_policies(params, initial_beliefs(), planner=six_policy_planner)
    np.testing.assert_allclose(np.asarray(logits), np.asarray(q_pi.logits), atol=1e-5)


def test_act_through_injected_planner():
    system, params = make_system(), make_params()
    a, q_pi, tree = system.act(
        jax.random.PRNGKey(0),
        params,
        initial_beliefs(),
        selection="deterministic",
        mode="policy",
        planner=six_policy_planner,
    )
    positional = system.to_positional("actions", a)
    assert positional.shape == (1,) and positional.dtype == jnp.int32
    assert int(a["a_mode"]) == COMMIT_RIGHT  # first action of the commit-right policy


def test_the_unobserved_factor_s_entropy_is_the_uncertainty_row_not_an_infogain_leaf():
    """``x`` has no channel, so no information to gain: no ``infogain`` leaf under any selector,
    a literal key naming it raises toward the row, and the row toggles with ``terms=``."""
    params, beliefs = make_params(), initial_beliefs()
    _, tree_on = make_system().infer_policies(params, beliefs, planner=six_policy_planner)
    assert "infogain" not in tree_on and set(tree_on["uncertainty"]) == {"x"}
    with pytest.raises(ValueError, match=r"uncertainty:x"):
        make_system(epistemic=("x",)).infer_policies(params, beliefs, planner=six_policy_planner)
    _, tree_off = make_system(terms=()).infer_policies(params, beliefs, planner=six_policy_planner)
    assert set(tree_off) == {"risk"}
    # toggling the entropy off changes the ranking input but risk leaves are identical
    np.testing.assert_allclose(
        np.asarray(tree_off["risk"]["x"].lower),
        np.asarray(tree_on["risk"]["x"].lower),
        atol=1e-6,
    )


def test_enumeration_planner_agrees_on_the_shared_policies():
    """The default enumeration planner over the full 27-policy product ranks the
    demo's 6 policies identically (same closures, superset policy set)."""
    system, params = make_system(), make_params()
    q_pi, tree = system.infer_policies(params, initial_beliefs())
    g = total(tree)
    pols = np.asarray(system.policies(params))[:, :, 0]  # mode-factor controls
    demo = np.asarray(POLICIES)
    idx = [int(np.where((pols == d).all(axis=1))[0][0]) for d in demo]
    g_demo = jax.vmap(demo_scores)(POLICIES)
    np.testing.assert_allclose(np.asarray(g.midpoint)[idx], np.asarray(g_demo.midpoint), atol=1e-4)


def test_update_empirical_prior_is_family_dispatched():
    system, params = make_system(), make_params()
    action = {"a_mode": jnp.asarray(COMMIT_RIGHT, jnp.int32)}
    pushed = system.update_empirical_prior(params, initial_beliefs(), action)
    mode_prior, x_prior = pushed["mode"], pushed["x"]
    assert isinstance(mode_prior, Categorical)
    np.testing.assert_allclose(
        np.asarray(mode_prior.probs), np.asarray(jnp.exp(ACTION_MODE_LOGW[2])), atol=1e-6
    )
    assert isinstance(x_prior, MixtureGaussian)
    assert x_prior.num_components == K_BUDGET  # static-K invariant holds across ticks
    # commit-right pushes mass right: mean above the init mixture's mean (0)
    assert float(x_prior.mean()[0]) > 0.5


def test_as_env_hybrid_forward_sampling():
    system, params = make_system(), make_params()
    env, env_params = system.as_env(params)
    obs, state = env.reset(jax.random.PRNGKey(3), env_params)
    assert obs == {}  # no modalities declared
    assert state["mode"].dtype == jnp.int32 and state["mode"].shape == ()
    assert state["x"].shape == (1,) and jnp.issubdtype(state["x"].dtype, jnp.floating)
    action = {"a_mode": jnp.asarray(COMMIT_RIGHT, jnp.int32)}
    _, state2 = env.step(jax.random.PRNGKey(4), state, action, env_params)
    assert state2["mode"].dtype == jnp.int32 and state2["x"].shape == (1,)
    # same key, same trajectory: pure sampling
    _, state2b = env.step(jax.random.PRNGKey(4), state, action, env_params)
    np.testing.assert_allclose(np.asarray(state2["x"]), np.asarray(state2b["x"]))
    assert int(state2["mode"]) == int(state2b["mode"])
