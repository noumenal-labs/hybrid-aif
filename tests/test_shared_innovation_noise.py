"""A shared innovation noise: a ``ConditionalGaussian`` whose ``chol`` omits the parent axes.

``loc`` ``(*parent_shape, d)`` beside ``chol`` ``(d, d)``: one noise for every mode, the mean per
mode. A model declared so scores bitwise as the same model with the noise stacked per level, on
the enumerating, the branching and the gradient planners, under the posterior chain and with
feedback; the description writes and reads the form; a ``chol`` carrying some of the parent
axes is refused. Design: docs/design/policy-independent-leaves.md H2, H3.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.description import build, describe
from hybrid_aif.distributions import ConditionalGaussian, Gaussian, MixtureGaussian
from hybrid_aif.durations import compose, discretise
from hybrid_aif.factors import LinearGaussianTransition
from hybrid_aif.inference.mixture_rollout import propagate
from hybrid_aif.objectives import total
from hybrid_aif.planning import gradient_plan
from hybrid_aif.planning.enumerate import evaluate_policies
from tests.test_closed_loop import PLAN, SMALL_PRESENT, _f32, small_lg
from tests.test_plan_evaluation_cost import _model

Q = jnp.asarray([[0.2, 0.0], [0.05, 0.1]])


def _identical(a, b, label=""):
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b), label
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        assert np.array_equal(np.asarray(x), np.asarray(y)), label


def _stacked(params, name):
    """``params`` with the named factor's shared noise written as equal slices per level."""
    inn = params.transition[name].innovations
    stacked = ConditionalGaussian(
        loc=inn.loc, chol=inn.stacked_chol(), parent_ndims=inn.parent_ndims
    )
    transition = dict(params.transition)
    transition[name] = dataclasses.replace(params.transition[name], innovations=stacked)
    return dataclasses.replace(params, transition=transition)


# ------------------------------------------------------------------------ the declaration


def test_a_shared_chol_is_a_declaration():
    loc = jnp.zeros((3, 2))
    shared = ConditionalGaussian(loc=loc, chol=Q, parent_ndims=1)
    stacked = ConditionalGaussian(loc=loc, chol=jnp.broadcast_to(Q, (3, 2, 2)), parent_ndims=1)
    assert shared.shared_chol and not stacked.shared_chol
    assert shared.parent_shape == stacked.parent_shape == (3,)
    assert shared.batch_shape == stacked.batch_shape == ()
    np.testing.assert_array_equal(shared.stacked_chol(), stacked.chol)
    for i in range(3):
        _identical(shared.condition_on((i,)), stacked.condition_on((i,)))
    np.testing.assert_array_equal(shared.gaussians().chol, stacked.gaussians().chol)
    two = ConditionalGaussian(loc=jnp.zeros((2, 3, 2)), chol=Q, parent_ndims=2)
    assert two.shared_chol and two.stacked_chol().shape == (2, 3, 2, 2)
    batched = ConditionalGaussian(
        loc=jnp.zeros((5, 3, 2)), chol=jnp.broadcast_to(Q, (5, 2, 2)), parent_ndims=1
    )
    assert batched.shared_chol and batched.batch_shape == (5,)
    assert batched.stacked_chol().shape == (5, 3, 2, 2)


def test_a_chol_with_some_of_the_parent_axes_is_refused():
    with pytest.raises(ValueError, match="only some of the parent axes"):
        ConditionalGaussian(
            loc=jnp.zeros((2, 3, 2)), chol=jnp.broadcast_to(Q, (3, 2, 2)), parent_ndims=2
        )
    with pytest.raises(ValueError, match="one factor shared by all of them"):
        ConditionalGaussian(loc=jnp.zeros((3, 2)), chol=jnp.ones((2, 2, 2)), parent_ndims=1)


# ------------------------------------------------------------------------ the planners


def test_the_enumerating_planner_scores_the_same():
    system, params, beliefs = _model(shared_noise=True)
    stacked = _stacked(params, "pos")
    assert params.transition["pos"].innovations.shared_chol
    assert not stacked.transition["pos"].innovations.shared_chol
    for chain in ("prior", "posterior"):
        system = dataclasses.replace(system, planning_belief=chain)
        a_pi, a_tree = system.infer_policies(params, beliefs)
        b_pi, b_tree = system.infer_policies(stacked, beliefs)
        _identical(a_pi, b_pi, chain)
        _identical(a_tree, b_tree, chain)


def test_the_branching_planner_scores_the_same():
    system, params, beliefs = _model(shared_noise=True)
    system = dataclasses.replace(system, planner="sophisticated", policy_len=2)
    stacked = _stacked(params, "pos")
    a_pi, a_tree = system.infer_policies(params, beliefs)
    b_pi, b_tree = system.infer_policies(stacked, beliefs)
    _identical(a_pi, b_pi)
    _identical(a_tree, b_tree)


def _shared_lg():
    system, params = small_lg()
    inn = params.transition["x"].innovations
    shared = ConditionalGaussian(loc=inn.loc, chol=inn.chol[0], parent_ndims=1)
    shared_params = dataclasses.replace(
        params,
        transition={
            **params.transition,
            "x": dataclasses.replace(params.transition["x"], innovations=shared),
        },
    )
    return system, params, shared_params


def _score_and_grad(system, params):
    ctx = system.planner_context(params, system.initial_priors(params), present=SMALL_PRESENT)

    def g(plan):
        tree = evaluate_policies(
            plan[None],
            ctx.init_beliefs,
            ctx.transition_fn,
            ctx.step_terms_fn,
            ctx.terminal_terms_fn,
        )
        return total(tree).midpoint[0], tree

    plan = _f32(PLAN)
    (value, tree), grad = jax.value_and_grad(g, has_aux=True)(plan)
    return value, grad, tree


def test_feedback_and_the_gradient_planner_score_the_same():
    """``small_lg`` plans on the posterior chain with ``feedback="linear_quadratic"``."""
    system, stacked_params, shared_params = _shared_lg()
    assert system.feedback == "linear_quadratic" and system.planning_belief == "posterior"
    _identical(_score_and_grad(system, shared_params), _score_and_grad(system, stacked_params))
    steps = {}
    for label, params in (("shared", shared_params), ("stacked", stacked_params)):
        ctx = system.planner_context(params, system.initial_priors(params), present=SMALL_PRESENT)
        steps[label] = gradient_plan(ctx, steps=10, init=jnp.zeros((system.policy_len, 1)))
    _identical(steps["shared"].plan, steps["stacked"].plan)
    _identical(steps["shared"].G, steps["stacked"].G)


def test_a_mixture_carried_switching_factor_scores_the_same():
    """A mixture-carried factor whose state mode parent indexes the mean only."""
    from tests import test_hybrid_switching_system as sw

    def scores(chol):
        inn = ConditionalGaussian(loc=sw.INNOVATIONS.loc, chol=chol, parent_ndims=1)
        params = dataclasses.replace(
            sw.make_params(),
            transition={
                **sw.make_params().transition,
                "x": LinearGaussianTransition(F=sw.F, innovations=inn),
            },
        )
        system = sw.make_system()
        return system.infer_policies(params, sw.initial_beliefs(), planner=sw.six_policy_planner)

    base = jnp.asarray([[0.3]])
    _identical(scores(base), scores(jnp.broadcast_to(base, (2, 1, 1))))


def test_a_mixture_carried_factor_reads_the_shared_noise():
    """A switching factor (two modes of ``loc``) with a shared noise grows the same mixture."""
    m = MixtureGaussian(
        log_weights=jnp.log(jnp.asarray([0.5, 0.5])),
        means=jnp.asarray([[0.0, 0.0], [1.0, 1.0]]),
        chols=jnp.stack([0.3 * jnp.eye(2)] * 2),
    )
    loc = jnp.asarray([[0.0, 0.1], [0.2, -0.1], [0.0, 0.0]])
    shared = ConditionalGaussian(loc=loc, chol=Q, parent_ndims=1)
    stacked = ConditionalGaussian(loc=loc, chol=shared.stacked_chol(), parent_ndims=1)
    Fs = jnp.broadcast_to(jnp.eye(2), (3, 2, 2))
    w = jnp.zeros(3)
    _identical(propagate(m, Fs, shared, None, w), propagate(m, Fs, stacked, None, w))


# ------------------------------------------------------------------------ the description


def test_the_description_round_trips_the_shared_form():
    system, params, beliefs = _model(shared_noise=True)
    document, arrays = describe(system, params)
    variables = document["variables"]
    assert variables["pos_transition_innovations_chol"]["shape"] == [2, 2]
    assert variables["pos_transition_innovations_loc"]["shape"] == [3, 2]
    rebuilt_system, rebuilt = build(document, arrays=arrays)
    inn = rebuilt.transition["pos"].innovations
    assert inn.shared_chol and inn.chol.shape == (2, 2) and inn.loc.shape == (3, 2)
    _identical(rebuilt.transition["pos"], params.transition["pos"])
    _identical(
        rebuilt_system.infer_policies(rebuilt, beliefs)[0],
        system.infer_policies(params, beliefs)[0],
    )


def test_the_description_still_round_trips_a_stacked_noise():
    system, params, _ = _model()
    document, arrays = describe(system, params)
    _, rebuilt = build(document, arrays=arrays)
    inn = rebuilt.transition["pos"].innovations
    assert not inn.shared_chol and inn.chol.shape == (3, 2, 2)


# ------------------------------------------------------------------------ durations


def _kernel(shared, shared_F=True):
    loc = jnp.asarray([[0.0, 0.1], [0.2, -0.1], [0.0, 0.0]])
    chol = Q if shared else jnp.broadcast_to(Q, (3, 2, 2))
    F = jnp.asarray([[1.0, 0.1], [0.0, 0.9]])
    if not shared_F:
        F = F * jnp.asarray([1.0, 0.9, 0.8])[:, None, None]
    return LinearGaussianTransition(
        F=F, innovations=ConditionalGaussian(loc=loc, chol=chol, parent_ndims=1)
    )


def test_compose_keeps_a_shared_noise_shared_and_equals_the_stacked_composition():
    shared = compose(_kernel(True), 5)
    stacked = compose(_kernel(False), 5)
    assert shared.innovations.shared_chol and not stacked.innovations.shared_chol
    np.testing.assert_array_equal(shared.innovations.loc, stacked.innovations.loc)
    np.testing.assert_allclose(
        shared.innovations.stacked_chol(), stacked.innovations.chol, atol=1e-6
    )
    np.testing.assert_array_equal(shared.F, stacked.F)


def test_compose_under_a_per_mode_map_gives_a_per_mode_noise():
    shared = compose(_kernel(True, shared_F=False), 4)
    stacked = compose(_kernel(False, shared_F=False), 4)
    assert not shared.innovations.shared_chol
    _identical(shared, stacked)


def test_discretise_then_compose_and_a_shared_target_kernel():
    """``discretise`` writes its kernel per mode; the shared form of that noise composes to the
    same kernel."""
    kernel = discretise(
        jnp.asarray([[0.0, 1.0], [0.0, -0.5]]), None, 0.1 * jnp.eye(2), 0.5, modes=3
    )
    inn = kernel.innovations
    shared = dataclasses.replace(
        kernel,
        innovations=ConditionalGaussian(loc=inn.loc, chol=inn.chol[0], parent_ndims=1),
    )
    a, b = compose(shared, 3), compose(kernel, 3)
    np.testing.assert_allclose(a.innovations.stacked_chol(), b.innovations.chol, atol=1e-6)
    np.testing.assert_array_equal(a.innovations.loc, b.innovations.loc)


def test_a_shared_noise_through_an_as_env_step_matches_the_stacked_one():
    system, params, _ = _model(shared_noise=True)
    stacked = _stacked(params, "pos")
    inn = params.transition["pos"].innovations
    for level in range(3):
        a = inn.condition_on((level,))
        b = stacked.transition["pos"].innovations.condition_on((level,))
        _identical(a, b)
        key = jax.random.PRNGKey(level)
        np.testing.assert_array_equal(a.sample(key), b.sample(key))
    assert isinstance(a, Gaussian)


@pytest.mark.parametrize("on_obs", [False, True])
@pytest.mark.parametrize("feedback", [None, "linear_quadratic"])
def test_a_conditional_target_with_a_shared_chol_scores_the_same(on_obs, feedback):
    """A conditional Gaussian target whose levels share one sd, declared once or per level,
    open loop and under the closed-loop pass."""
    from tests import test_conditional_targets as ct

    system, params = ct._lg(policy_len=3, feedback=feedback, on_obs=on_obs)
    key = "obs_targets" if on_obs else "state_targets"
    name = "y" if on_obs else "x"
    loc = getattr(params, key)[name].loc
    sd = jnp.asarray([[0.5]])
    stacked = ConditionalGaussian(loc=loc, chol=jnp.broadcast_to(sd, (2, 1, 1)), parent_ndims=1)
    shared = ConditionalGaussian(loc=loc, chol=sd, parent_ndims=1)
    beliefs = system.initial_priors(params)
    held = ct._held(1, 3)
    plan = {"u": held[:, 0], "v": jnp.asarray([[0.2], [-0.1], [0.3]])} if feedback else held
    trees = []
    for target in (stacked, shared):
        p = dataclasses.replace(params, **{key: {name: target}})
        trees.append(system.score_policies(p, beliefs, plan))
    _identical(*trees)
