"""A neural likelihood in planning: the linear-Gaussian model linearised at each step's belief.

An ``MLPConditional`` in ``params.likelihood`` presents at every plan step as ``o = H x + c +
eps'`` read off the network at the step's belief, per component for a mixture, per level for
categorical parents, and every objective term reads it; a target on it is admitted. Pinned: an
affine network gives the linear-Gaussian path's leaves, scores and chosen action under every
planner the model admits and both propagations; a curved decoder's information gain against a
Monte Carlo mutual information; a Gaussian target moves the plan toward the preferred image;
the closed-loop pass and the gradient planner run through it; the refusals.

Design: docs/design/recognition-beside-analytic.md L1, L4.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import (
    Box,
    ConditionalCategorical,
    ConditionalGaussian,
    Gaussian,
    LinearGaussianObservation,
    LinearGaussianTransition,
    MixtureGaussian,
    System,
    SystemParams,
)
from hybrid_aif.distributions import Categorical, MLPConditional
from hybrid_aif.objectives import total
from hybrid_aif.planning import gradient_plan
from hybrid_aif.planning.enumerate import evaluate_policies
from tests.test_hybrid_lg_system import make_params, make_system

PROPAGATIONS = ("first_order", "unscented")
D, D_OBS = 2, 3


def affine(parent_dims, key, propagation="first_order"):
    """A depth-0 conditional: an affine mean and a constant scale (its scale rows zeroed)."""
    decl = MLPConditional.from_shapes(
        parent_dims, D_OBS, key=key, depth=0, frozen=True, propagation=propagation
    )
    weight = decl.layers[-1].weight.at[D_OBS:].set(0.0)
    return eqx.tree_at(lambda m: m.layers[-1].weight, decl, weight)


def curved(key=None, propagation="first_order", parent_dims=(D,)):
    key = jax.random.PRNGKey(3) if key is None else key
    return MLPConditional.from_shapes(
        parent_dims, D_OBS, key=key, width=12, depth=2, frozen=True, propagation=propagation
    )


def lg_equivalent(decl):
    """The linear-Gaussian observation an affine conditional over one continuous parent is."""
    mean, scale = decl._head((jnp.zeros(D),))
    H = jax.jacfwd(lambda x: decl._head((x,))[0])(jnp.zeros(D))
    return LinearGaussianObservation(H=H, c=mean, noise=jnp.diag(scale))


IMAGE_TARGET = Gaussian(loc=jnp.array([0.2, 0.0, 0.1]), chol=0.5 * jnp.eye(D_OBS))


def _gaussian_models(decl, target=True, **overrides):
    """The hill-class model with ``position`` replaced by ``decl`` and by its LG equivalent."""
    system = make_system(**overrides)
    base = dataclasses.replace(
        make_params(), obs_targets={"position": IMAGE_TARGET} if target else {}
    )
    neural = dataclasses.replace(base, likelihood={"position": decl})
    lg = dataclasses.replace(base, likelihood={"position": lg_equivalent(decl)})
    return system, neural, lg


def _assert_trees_close(a, b, atol=1e-5):
    assert set(a) == set(b)
    for term in a:
        assert set(a[term]) == set(b[term])
        for key in a[term]:
            np.testing.assert_allclose(
                np.asarray(a[term][key].lower), np.asarray(b[term][key].lower), atol=atol
            )
            np.testing.assert_allclose(
                np.asarray(a[term][key].upper), np.asarray(b[term][key].upper), atol=atol
            )


# --------------------------------------------- (a) an affine network is the linear-Gaussian model


@pytest.mark.parametrize("propagation", PROPAGATIONS)
@pytest.mark.parametrize("decomposition", ["utility_infogain", "risk_ambiguity"])
@pytest.mark.parametrize("belief", ["prior", "posterior"])
def test_an_affine_network_over_a_gaussian_factor_is_the_lg_model(
    propagation, decomposition, belief
):
    decl = affine((D,), jax.random.PRNGKey(0), propagation)
    system, neural, lg = _gaussian_models(decl, decomposition=decomposition, planning_belief=belief)
    beliefs = system.initial_priors(lg)
    q_n, tree_n = system.infer_policies(neural, beliefs)
    q_l, tree_l = system.infer_policies(lg, beliefs)
    np.testing.assert_allclose(np.asarray(q_n.logits), np.asarray(q_l.logits), atol=1e-4)
    _assert_trees_close(tree_n, tree_l)
    key = jax.random.PRNGKey(1)
    act_n, _, _ = system.act(key, neural, beliefs, selection="deterministic", mode="policy")
    act_l, _, _ = system.act(key, lg, beliefs, selection="deterministic", mode="policy")
    assert int(act_n["a_state"]) == int(act_l["a_state"])


def _unit_system(planner="gradient", **overrides):
    """A 2-D state a continuous input pushes, observed through one modality ``y``."""
    system = System(
        A_deps=((0,),),
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("u",),
        factor_names=("x", "u_x"),
        modality_names=("y",),
        policy_len=4,
        planner=planner,
        planning_belief="posterior",
        **overrides,
    )
    return system


def _unit_params(likelihood, target=True):
    return SystemParams(
        transition={
            "x": LinearGaussianTransition(
                F=jnp.array([[1.0, 0.3], [0.0, 0.9]]),
                G=jnp.array([[0.0], [0.4]]),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, D)), chol=0.05 * jnp.eye(D)[None], parent_ndims=1
                ),
            )
        },
        initial_state_prior={"x": Gaussian(loc=jnp.array([0.1, -0.2]), chol=0.2 * jnp.eye(D))},
        likelihood={"y": likelihood},
        obs_targets={"y": IMAGE_TARGET} if target else {},
        state_targets={},
        control={"u_x": Box(-5.0 * jnp.ones(1), 5.0 * jnp.ones(1))},
    )


@pytest.mark.parametrize("propagation", PROPAGATIONS)
def test_an_affine_network_gives_the_lg_gradient_plan(propagation):
    decl = affine((D,), jax.random.PRNGKey(0), propagation)
    system = _unit_system()
    p_n, p_l = _unit_params(decl), _unit_params(lg_equivalent(decl))
    ctxs = [system.planner_context(p, system.initial_priors(p), horizon=4) for p in (p_n, p_l)]
    start = jnp.full((4, 1), 0.1)
    plans = [gradient_plan(ctx, steps=30, init=start) for ctx in ctxs]
    np.testing.assert_allclose(
        np.asarray(plans[0].plan), np.asarray(plans[1].plan), atol=1e-3, rtol=1e-3
    )
    np.testing.assert_allclose(
        float(plans[0].G.midpoint[0]), float(plans[1].G.midpoint[0]), atol=1e-3, rtol=1e-3
    )


@pytest.mark.parametrize("propagation", PROPAGATIONS)
def test_an_affine_network_gives_the_lg_leaves_along_a_continuous_plan(propagation):
    decl = affine((D,), jax.random.PRNGKey(0), propagation)
    system = _unit_system()
    plan = jnp.asarray([[0.3], [-0.2], [0.1], [0.4]])
    trees = []
    for lik in (decl, lg_equivalent(decl)):
        params = _unit_params(lik)
        ctx = system.planner_context(params, system.initial_priors(params))
        tree = evaluate_policies(
            plan[None],
            ctx.init_beliefs,
            ctx.transition_fn,
            ctx.step_terms_fn,
            ctx.terminal_terms_fn,
        )
        trees.append(jax.tree_util.tree_map(lambda x: x[0], tree))
    _assert_trees_close(*trees)


def test_a_sophisticated_plan_over_a_neural_modality_scores_as_the_lg_one():
    """The sophisticated planner needs a tabular modality to branch on; the neural one beside
    it changes nothing about the structure of the plan, only its leaves."""
    from tests.test_planning_with_observations import _belief

    decl = affine((D,), jax.random.PRNGKey(0))
    system, neural, lg = _gaussian_models(decl, planner="sophisticated")
    beliefs = {"state": _belief()}
    q_n, _ = system.infer_policies(neural, beliefs)
    q_l, _ = system.infer_policies(lg, beliefs)
    np.testing.assert_allclose(np.asarray(q_n.logits), np.asarray(q_l.logits), atol=1e-4)


# ------------------------------------------ (b) the mode-aligned mixture with a categorical parent

M, K = 3, 6
STICKY = jnp.log(jnp.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]))
SHIFT = jnp.asarray([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.5]])
TERRAIN = Categorical.from_probs(jnp.asarray([0.5, 0.3, 0.2]))
_WITHIN = np.asarray([[0.6, 0.4], [0.5, 0.5], [0.3, 0.7]])
X_PRIOR = MixtureGaussian(
    log_weights=jnp.log(jnp.asarray((np.asarray(TERRAIN.probs)[:, None] * _WITHIN).reshape(-1))),
    means=jnp.asarray([[-1.0, 0.0], [-0.6, 0.3], [0.0, -0.2], [0.3, 0.1], [0.9, 0.4], [1.3, 0.6]]),
    chols=jnp.stack([0.4 * jnp.eye(2)] * K),
)


def _mixture_system(**overrides):
    return System(
        A_deps=((1, 0),),
        B_deps=((0,), (0,)),
        controlled=(False, True),
        policy_len=2,
        mixture_k=K,
        mixture_schedule="mode_aligned",
        modality_names=("img",),
        factor_names=("terrain", "x"),
        **overrides,
    )


def _mixture_params(likelihood, target=True):
    push = jnp.asarray([-0.3, 0.0, 0.3])
    loc = SHIFT[:, None, :] + push[None, :, None] * jnp.asarray([1.0, 0.5])
    return SystemParams(
        likelihood={"img": likelihood},
        transition={
            "terrain": ConditionalCategorical(log_table=STICKY, parent_ndims=1),
            "x": LinearGaussianTransition(
                F=0.9 * jnp.eye(2),
                innovations=ConditionalGaussian(
                    loc=loc, chol=jnp.broadcast_to(0.3 * jnp.eye(2), (M, 3, 2, 2)), parent_ndims=2
                ),
            ),
        },
        initial_state_prior={"terrain": TERRAIN, "x": X_PRIOR},
        obs_targets={"img": IMAGE_TARGET} if target else {},
        state_targets={},
    )


def _switching_equivalent(decl):
    """The switching LG observation an affine conditional over ``(x, terrain)`` is: a shared
    ``H``, a per-terrain offset."""
    H = jax.jacfwd(lambda x: decl._head((x, jnp.zeros(M)))[0])(jnp.zeros(D))
    cs = jnp.stack([decl._head((jnp.zeros(D), jax.nn.one_hot(j, M)))[0] for j in range(M)])
    _, scale = decl._head((jnp.zeros(D), jnp.zeros(M)))
    return LinearGaussianObservation(H=H, c=cs, noise=jnp.diag(scale))


@pytest.mark.parametrize("propagation", PROPAGATIONS)
@pytest.mark.parametrize("decomposition", ["utility_infogain", "risk_ambiguity"])
def test_an_affine_network_over_a_mixture_with_a_mode_parent_is_the_switching_lg_model(
    propagation, decomposition
):
    decl = affine((D, M), jax.random.PRNGKey(7), propagation)
    system = _mixture_system(decomposition=decomposition)
    p_n = _mixture_params(decl)
    p_l = _mixture_params(_switching_equivalent(decl))
    beliefs = system.initial_priors(p_l)
    q_n, tree_n = system.infer_policies(p_n, beliefs)
    q_l, tree_l = system.infer_policies(p_l, beliefs)
    np.testing.assert_allclose(np.asarray(q_n.logits), np.asarray(q_l.logits), atol=1e-4)
    _assert_trees_close(tree_n, tree_l, atol=2e-4)
    assert (
        float(
            jnp.abs(
                tree_n["utility" if decomposition == "utility_infogain" else "risk"]["img"].lower
            ).max()
        )
        > 0.0
    )


def test_a_curved_network_over_a_mixture_scores_each_component_at_its_own_mean():
    """Two components far apart read a curved decoder through different heads; the information
    gain differs from the one-head reading of the mixture's mean."""
    decl = curved(parent_dims=(D, M))
    system = _mixture_system(decomposition="utility_infogain")
    params = _mixture_params(decl, target=False)
    beliefs = system.initial_priors(params)
    q, tree = system.infer_policies(params, beliefs)
    assert np.all(np.isfinite(np.asarray(q.logits)))
    gain = np.asarray(tree["infogain"]["img"].midpoint)
    assert np.all(np.isfinite(gain)) and np.all(gain > 0.0)


# ------------------------------------------------------ (c) a curved decoder's information gain


def _mc_information(decl, loc, chol, n_x=3000, n_y=1500):
    """``I(y; x) = E[log p(y | x) - log p(y)]`` by nested Monte Carlo through the decoder."""
    keys = jax.random.split(jax.random.PRNGKey(5), 3)
    xs = loc + jax.random.normal(keys[0], (n_x, D)) @ chol.T
    means, scales = jax.vmap(lambda x: decl._head((x,)))(xs)

    def log_cond(y, mu, sd):
        return jnp.sum(-0.5 * ((y - mu) / sd) ** 2 - jnp.log(sd) - 0.5 * jnp.log(2 * jnp.pi), -1)

    x_y = loc + jax.random.normal(keys[1], (n_y, D)) @ chol.T
    mu_y, sd_y = jax.vmap(lambda x: decl._head((x,)))(x_y)
    y = mu_y + sd_y * jax.random.normal(keys[2], mu_y.shape)
    log_py_x = log_cond(y, mu_y, sd_y)
    pair = jax.vmap(lambda yy: log_cond(yy, means, scales))(y)  # (n_y, n_x)
    log_py = jax.scipy.special.logsumexp(pair, axis=-1) - jnp.log(n_x)
    return float(jnp.mean(log_py_x - log_py))


@pytest.mark.parametrize("propagation", PROPAGATIONS)
def test_the_information_gain_of_a_curved_decoder_matches_monte_carlo_under_a_tight_belief(
    propagation,
):
    decl = curved(propagation=propagation)
    system, neural, _ = _gaussian_models(
        decl, target=False, policy_len=1, planning_belief="prior", epistemic=True
    )
    tight = Gaussian(loc=jnp.array([0.3, -0.1]), chol=jnp.diag(jnp.array([0.08, 0.06])))
    ctx = system.planner_context(neural, {"state": tight})
    carry = ctx.transition_fn(ctx.init_beliefs, jnp.asarray([2], jnp.int32))
    tree = ctx.step_terms_fn(carry, jnp.asarray([2], jnp.int32), jnp.asarray(0, jnp.int32))
    (predicted,) = carry
    gain = float(tree["infogain"]["position"].midpoint)
    reference = _mc_information(decl, predicted.loc, predicted.chol)
    assert gain == pytest.approx(reference, abs=0.15)


# ----------------------------------------------------------- (d) a target on a neural modality


def _closed_form_kl(mean_p, cov_p, mean_q, cov_q):
    mean_p, cov_p, mean_q, cov_q = (
        np.asarray(a, np.float64) for a in (mean_p, cov_p, mean_q, cov_q)
    )
    k = mean_p.shape[0]
    dev = mean_q - mean_p
    inv_q = np.linalg.inv(cov_q)
    return 0.5 * (
        np.trace(inv_q @ cov_p)
        + dev @ inv_q @ dev
        - k
        + np.linalg.slogdet(cov_q)[1]
        - np.linalg.slogdet(cov_p)[1]
    )


def test_a_gaussian_target_on_a_neural_modality_prefers_the_closer_image():
    decl = curved()
    system = make_system(decomposition="risk_ambiguity", epistemic=False, policy_len=1)
    base = make_params()
    belief = Gaussian(loc=jnp.array([0.3, -0.1]), chol=0.1 * jnp.eye(2))

    def predicted_state(u):
        ctx = system.planner_context(
            dataclasses.replace(base, likelihood={"position": decl}), {"state": belief}
        )
        return ctx.transition_fn(ctx.init_beliefs, jnp.asarray([u], jnp.int32))[0]

    # the image of the state thrust level 2 reaches is the preferred one
    preferred = predicted_state(2)
    image, _ = decl._head((preferred.loc,))
    target = Gaussian(loc=image, chol=0.2 * jnp.eye(D_OBS))
    params = dataclasses.replace(
        base, likelihood={"position": decl}, obs_targets={"position": target}
    )
    q, tree = system.infer_policies(params, {"state": belief})
    assert int(jnp.argmax(q.logits)) == 2
    # each policy's risk leaf is the KL between the linearised predictive and the target
    for u in range(3):
        pred = predicted_state(u)
        H = np.asarray(jax.jacfwd(lambda x: decl._head((x,))[0])(pred.loc))
        mean, scale = decl._head((pred.loc,))
        c = np.asarray(mean) - H @ np.asarray(pred.loc)
        cov = H @ np.asarray(pred.chol @ pred.chol.T) @ H.T + np.diag(np.asarray(scale) ** 2)
        kl = _closed_form_kl(
            H @ np.asarray(pred.loc) + c, cov, np.asarray(target.loc), np.asarray(target.cov)
        )
        risk = np.asarray(tree["risk"]["position"].midpoint)
        assert risk[u] == pytest.approx(kl, rel=2e-3, abs=2e-3)


# ------------------------------------------------------------------ (e) the closed-loop pass


def _lq_system(**overrides):
    return _unit_system(feedback="linear_quadratic", decomposition="utility_infogain", **overrides)


def test_the_closed_loop_pass_credits_an_affine_neural_modality_as_the_lg_one():
    decl = affine((D,), jax.random.PRNGKey(0))
    system = _lq_system()
    plan = jnp.asarray([[0.3], [-0.2], [0.1], [0.4]])
    trees = []
    for lik in (decl, lg_equivalent(decl)):
        params = _unit_params(lik)
        ctx = system.planner_context(params, system.initial_priors(params))
        tree = evaluate_policies(
            plan[None],
            ctx.init_beliefs,
            ctx.transition_fn,
            ctx.step_terms_fn,
            ctx.terminal_terms_fn,
        )
        trees.append(jax.tree_util.tree_map(lambda x: x[0], tree))
    _assert_trees_close(*trees, atol=1e-4)
    assert "y" in trees[0]["utility"]


def test_a_closed_loop_plan_over_a_curved_decoder_runs_and_credits_the_modality():
    decl = curved()
    system = _lq_system()
    params = _unit_params(decl)
    ctx = system.planner_context(params, system.initial_priors(params))
    plan = jnp.asarray([[0.3], [-0.2], [0.1], [0.4]])

    def total_g(p):
        tree = evaluate_policies(
            p[None], ctx.init_beliefs, ctx.transition_fn, ctx.step_terms_fn, ctx.terminal_terms_fn
        )
        return tree["utility"]["y"].midpoint[0], tree

    value, tree = total_g(plan)
    assert np.isfinite(float(value))
    open_loop = _unit_system(feedback=None, decomposition="utility_infogain")
    ctx_open = open_loop.planner_context(params, open_loop.initial_priors(params))
    tree_open = evaluate_policies(
        plan[None],
        ctx_open.init_beliefs,
        ctx_open.transition_fn,
        ctx_open.step_terms_fn,
        ctx_open.terminal_terms_fn,
    )
    # the correction the feedback adds to the modality's leaf is a nonzero, finite amount
    gap = float(tree["utility"]["y"].midpoint[0] - tree_open["utility"]["y"].midpoint[0])
    assert np.isfinite(gap) and abs(gap) > 0.0
    grad = jax.grad(lambda p: total_g(p)[0])(plan)
    assert np.all(np.isfinite(np.asarray(grad)))


# ------------------------------------------------------------- (f) the gradient planner


@pytest.mark.parametrize("propagation", PROPAGATIONS)
def test_the_gradient_planner_runs_through_a_curved_decoder(propagation):
    decl = curved(propagation=propagation)
    system = _unit_system(decomposition="utility_infogain")
    params = _unit_params(decl)
    ctx = system.planner_context(params, system.initial_priors(params))

    def g(plan):
        tree = evaluate_policies(
            plan[None],
            ctx.init_beliefs,
            ctx.transition_fn,
            ctx.step_terms_fn,
            ctx.terminal_terms_fn,
        )
        return total(tree).midpoint[0]

    start = jnp.full((4, 1), 0.1)
    grad = jax.grad(g)(start)
    assert np.all(np.isfinite(np.asarray(grad))) and float(jnp.abs(grad).max()) > 0.0
    result = gradient_plan(ctx, steps=40, init=start)
    assert np.all(np.isfinite(np.asarray(result.plan)))
    assert float(result.G.midpoint[0]) <= float(g(start)) + 1e-4


# ------------------------------------------------------------------------------ (g) refusals


def test_a_neural_modality_over_two_continuous_factors_is_refused_by_name():
    decl = MLPConditional.from_shapes((D, D), D_OBS, key=jax.random.PRNGKey(0), frozen=True)
    system = System(
        A_deps=((0, 1),),
        B_deps=((), ()),
        controlled=(True, False),
        policy_len=1,
        modality_names=("img",),
        factor_names=("pose", "light"),
    )
    transition = {
        "pose": LinearGaussianTransition(
            F=jnp.eye(D),
            innovations=ConditionalGaussian(
                loc=jnp.zeros((2, D)), chol=jnp.stack([0.1 * jnp.eye(D)] * 2), parent_ndims=1
            ),
        ),
        "light": LinearGaussianTransition(
            F=jnp.eye(D),
            innovations=ConditionalGaussian(
                loc=jnp.zeros((1, D)), chol=0.1 * jnp.eye(D)[None], parent_ndims=1
            ),
        ),
    }
    params = SystemParams(
        likelihood={"img": decl},
        transition=transition,
        initial_state_prior={
            "pose": Gaussian(loc=jnp.zeros(D), chol=jnp.eye(D)),
            "light": Gaussian(loc=jnp.zeros(D), chol=jnp.eye(D)),
        },
    )
    with pytest.raises(NotImplementedError, match=r"'pose'.*'light'|\['pose', 'light'\]"):
        system.infer_policies(params, system.initial_priors(params))


def test_a_neural_target_of_the_wrong_family_or_dimension_is_refused():
    decl = curved()
    system, neural, _ = _gaussian_models(decl)
    wrong_family = dataclasses.replace(
        neural, obs_targets={"position": Categorical(logits=jnp.zeros(D_OBS))}
    )
    with pytest.raises(ValueError, match=r"obs_targets\['position'\].*neural modality"):
        system.infer_policies(wrong_family, system.initial_priors(neural))
    wrong_dim = dataclasses.replace(
        neural, obs_targets={"position": Gaussian(loc=jnp.zeros(2), chol=jnp.eye(2))}
    )
    with pytest.raises(ValueError, match=r"target dim 2 != .*observation dim 3 .*child_dim"):
        system.infer_policies(wrong_dim, system.initial_priors(neural))


# ------------------------------------------ (h) a neural modality over categorical parents alone

LEVELS = 3
_TRANSITION = np.random.RandomState(0).uniform(0.1, 1.0, (LEVELS, LEVELS, LEVELS))
_TRANSITION = _TRANSITION / _TRANSITION.sum(0, keepdims=True)


def _categorical_model(likelihood, decomposition):
    system = System(
        A_deps=((0,),),
        B_deps=((0,),),
        controlled=(True,),
        policy_len=2,
        modality_names=("img",),
        factor_names=("z",),
        decomposition=decomposition,
    )
    params = SystemParams(
        likelihood={"img": likelihood},
        transition={"z": ConditionalCategorical(log_table=jnp.log(_TRANSITION), parent_ndims=2)},
        initial_state_prior={"z": Categorical(logits=jnp.zeros(LEVELS))},
    )
    return system, params


def _emission_equivalent(decl):
    """The Gaussian emission a neural conditional over one categorical parent tabulates."""
    mean, scale = jax.vmap(lambda e: decl.diag_moments((e,)))(jnp.eye(LEVELS))
    return LinearGaussianObservation(
        H=jnp.zeros((LEVELS, D_OBS, 0)), c=mean, noise=jax.vmap(jnp.diag)(scale)
    )


@pytest.mark.parametrize("decomposition", ["utility_infogain", "risk_ambiguity"])
def test_a_neural_modality_over_categorical_parents_is_the_hand_declared_emission(decomposition):
    decl = MLPConditional.from_shapes(
        (LEVELS,), D_OBS, key=jax.random.PRNGKey(3), width=8, depth=1, frozen=True
    )
    trees, policies = [], []
    for likelihood in (decl, _emission_equivalent(decl)):
        system, params = _categorical_model(likelihood, decomposition)
        q, tree = system.infer_policies(params, system.initial_priors(params))
        trees.append(tree)
        policies.append(q)
    _assert_trees_close(*trees)
    np.testing.assert_allclose(
        np.asarray(policies[0].logits), np.asarray(policies[1].logits), atol=1e-5
    )
    # a read that is exact: no leaf and not the total is stamped approximate
    flags = [iv.approximate for blocks in trees[0].values() for iv in blocks.values()]
    assert flags and not any(flags) and not total(trees[0]).approximate


# ------------------------------------------------ (i) unscented: the leaves read the predictive


def test_under_unscented_the_risk_leaf_reads_the_predictive_belief_not_the_epistemic_one():
    from hybrid_aif.distributions.gaussian import sigma_points

    decl = curved(propagation="unscented")
    system, neural, _ = _gaussian_models(
        decl, decomposition="risk_ambiguity", epistemic=False, policy_len=2,
        planning_belief="posterior",
    )  # fmt: skip
    ctx = system.planner_context(neural, system.initial_priors(neural))
    first, second = jnp.asarray([2], jnp.int32), jnp.asarray([1], jnp.int32)
    carry = ctx.transition_fn(ctx.transition_fn(ctx.init_beliefs, first), second)
    pair = carry[0]
    # an epistemic member far from the predictive one: the two reads of the head differ
    far = Gaussian(loc=pair.predictive.loc + 0.5, chol=3.0 * pair.predictive.chol)
    tree = ctx.step_terms_fn((pair._replace(epistemic=far),), second, jnp.asarray(1, jnp.int32))
    target = neural.obs_targets["position"]

    def sigma_point_risk(q):
        points, w = sigma_points(q.loc, q.chol, 0.0)
        mean_i, scale_i = jax.vmap(lambda x: decl._head((x,)))(points)
        mean = w @ mean_i
        dev = mean_i - mean
        cov = (dev.T * w) @ dev + jnp.diag(w @ scale_i**2)
        return _closed_form_kl(mean, cov, target.loc, target.cov)

    risk = float(tree["risk"]["position"].midpoint)
    assert risk == pytest.approx(sigma_point_risk(pair.predictive), abs=1e-4)
    assert abs(risk - sigma_point_risk(far)) > 1e-3


@pytest.mark.parametrize("propagation", PROPAGATIONS)
@pytest.mark.parametrize("belief", ["prior", "posterior"])
def test_both_decompositions_give_one_score_for_a_curved_decoder(propagation, belief):
    """Ambiguity and information gain read the chain's belief, risk and utility the predictive
    one, so the two decompositions agree under either propagation."""
    totals = {}
    for decomposition in ("utility_infogain", "risk_ambiguity"):
        decl = curved(propagation=propagation)
        system, neural, _ = _gaussian_models(
            decl, decomposition=decomposition, policy_len=3, planning_belief=belief
        )
        _, tree = system.infer_policies(neural, system.initial_priors(neural))
        totals[decomposition] = np.asarray(total(tree).midpoint)
    np.testing.assert_allclose(totals["utility_infogain"], totals["risk_ambiguity"], atol=1e-3)


# ------------------------------------------------------------------- (j) refusals, wording


def test_propagation_outside_a_likelihood_slot_is_refused_at_validation():
    system, neural, _ = _gaussian_models(curved())
    module = MLPConditional.from_shapes((D,), D, key=jax.random.PRNGKey(1), frozen=True)
    odd = dataclasses.replace(module, propagation="unscented")
    transition = {"state": odd}
    bad = dataclasses.replace(neural, transition=transition)
    with pytest.raises(ValueError, match=r"propagation.*read only on a likelihood"):
        system._validate(bad, "test")


def test_a_learnable_decoder_under_default_epistemic_says_how_to_plan_it():
    decl = MLPConditional.from_shapes((D,), D_OBS, key=jax.random.PRNGKey(3), width=8, depth=1)
    system, neural, _ = _gaussian_models(decl, target=False, epistemic=True)
    with pytest.raises(NotImplementedError, match=r"learnable neural likelihood.*frozen"):
        system.infer_policies(neural, system.initial_priors(neural))
