"""The CEM planner and the continuous action space (amortization.md A9).

Four things get bound here:

- **CEM finds the optimum**, checked against exhaustive search on a problem whose
  answer is known independently: an exactly affine neural transition (a depth-0
  ``MLPConditional`` with hand-set weights *is* ``x' = F x + G a``) plus a
  Gaussian preference makes G a quadratic in the action sequence, so a fine grid
  over whole sequences is a ground-truth oracle — and CEM, searching
  continuously, must match or beat its best;
- **the mechanics**: determinism under a fixed key, jit-safety, static shapes,
  bound clipping, the ``min_std`` exploration floor, the warm start;
- **the continuous-action declaration**, whose every fence raises (mixed action
  spaces, level counts on a continuous factor, non-neural transitions,
  enumeration, action marginals, integer plans);
- **the parent convention**: a continuous action reaches the net as its raw
  vector, in prediction *and* in ``learn``'s gradient — the A9 half of the
  convention chunk 1 wrote the one-hot half of.
"""

import dataclasses
import importlib.util
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.controls import Box
from hybrid_aif.distributions import Categorical, Gaussian, MLPConditional
from hybrid_aif.factors import LinearGaussianTransition
from hybrid_aif.objectives import total
from hybrid_aif.planning import get_planner, planner_needs_key
from hybrid_aif.planning.cem import cem_plan, make_cem_planner, mean_action
from hybrid_aif.system import System, SystemParams

KEY = jax.random.PRNGKey(0)
TARGET = 0.5  # the goal position of the toy LQ problem: interior, so a* is not clipped


def affine_net(F: float = 1.0, G: float = 1.0, sd: float = 0.05) -> MLPConditional:
    """``x' = F x + G a`` with constant scale ``sd``, as an *exactly* affine net.

    A depth-0 :class:`MLPConditional` is one ``Linear`` over ``[x; a]`` emitting
    ``(mean, pre-softplus scale)``, so setting its weight and bias by hand gives a
    linear system with a known optimum — and A1(a)'s linearized prediction row is
    exact on it (chunk 1's ``test_linearized_prediction_matches_the_lg_predict``).
    """
    net = MLPConditional.from_shapes((1, 1), 1, key=KEY, depth=0, min_scale=1e-3, init_scale=sd)
    weight = jnp.asarray([[F, G], [0.0, 0.0]])
    bias = jnp.asarray([0.0, float(jnp.log(jnp.expm1(jnp.asarray(sd - 1e-3))))])
    net = eqx.tree_at(lambda m: m.layers[0].weight, net, weight)
    return eqx.tree_at(lambda m: m.layers[0].bias, net, bias)


def lq_system(
    horizon: int = 3,
    *,
    low: float = -1.0,
    high: float = 1.0,
    target_sd: float = 0.2,
    F: float = 1.0,
    G: float = 1.0,
    **system_kwargs,
) -> tuple[System, SystemParams]:
    """One continuous factor, one continuous control, a Gaussian goal designation."""
    system = System(
        **{
            "A_deps": (),
            "B_deps": ((),),
            "controlled": (True,),
            "policy_len": horizon,
            "planner": "cem",
            "gamma": 8.0,
            "epistemic": False,
            "factor_names": ("x",),
            **system_kwargs,
        }
    )
    params = SystemParams(
        likelihood={},
        transition={"x": affine_net(F=F, G=G)},
        initial_state_prior={"x": Gaussian(loc=jnp.zeros(1), chol=1e-2 * jnp.eye(1))},
        state_targets={
            "x": Gaussian(loc=jnp.asarray([TARGET]), chol=target_sd * jnp.eye(1)),
        },
        control={"u_x": box(low, high)},
    )
    return system, params


def box(low: float = -1.0, high: float = 1.0, dim: int = 1) -> Box:
    """A ``dim``-wide control box ``[low, high]``, as ``params.control`` holds it."""
    return Box(jnp.full((dim,), low), jnp.full((dim,), high))


def _belief(x: float = 0.0, sd: float = 1e-2) -> dict[str, Gaussian]:
    return {"x": Gaussian(loc=jnp.asarray([x]), chol=sd * jnp.eye(1))}


def _plan(system, params, beliefs, key=KEY, **kwargs):
    ctx = system.planner_context(params, beliefs)
    return cem_plan(ctx, key, **kwargs)


# ------------------------------------------------------------------ CEM vs the oracle


def test_cem_matches_an_exhaustive_grid_over_whole_action_sequences():
    """The optimality check, against a ground truth that does not come from CEM:
    a 41-point grid over each of three steps (68,921 sequences) scored through
    ``score_policies`` — the same term machinery the planner uses. CEM searches
    the continuum, so it must come within float noise of the grid's best G, and
    may beat it.

    The exploration floor is turned down here (``min_std`` small): a floored
    proposal deliberately does *not* converge — that is what it is for — so an
    optimality test has to ask for the converged search."""
    system, params = lq_system(horizon=3)
    beliefs = _belief(0.0)

    grid = jnp.linspace(-1.0, 1.0, 41)
    sequences = jnp.stack(jnp.meshgrid(grid, grid, grid, indexing="ij"), axis=-1)
    sequences = sequences.reshape(-1, 3, 1)
    grid_G = np.asarray(total(system.score_policies(params, beliefs, sequences)).midpoint)
    best = grid_G.min()
    best_first = float(sequences[grid_G.argmin(), 0, 0])

    plan = _plan(system, params, beliefs, num_candidates=256, cem_iters=10, min_std=1e-4)
    cem_G = float(np.asarray(total(system.score_policies(params, beliefs, plan.mean)).midpoint))
    assert cem_G <= best + 1e-3
    # the analytic optimum of x' = x + a toward TARGET from x=0 is a_0 = TARGET,
    # then hold — the grid agrees, and so must CEM
    assert best_first == pytest.approx(TARGET, abs=0.05)
    assert float(mean_action(plan.candidates)[0]) == pytest.approx(TARGET, abs=0.05)
    np.testing.assert_allclose(np.asarray(plan.mean[1:, 0]), 0.0, atol=0.05)


def test_the_returned_posterior_ranks_the_candidates_and_row_zero_is_the_mean():
    """``q_pi = softmax(-gamma G)`` over the final candidate set *plus* the refit
    mean at row 0 — so the recommended action is inside the posterior reported
    beside it, and better G means more mass."""
    system, params = lq_system(horizon=2)
    beliefs = _belief(0.0)
    plan = _plan(system, params, beliefs, num_candidates=32, cem_iters=3)

    assert plan.candidates.shape == (33, 2, 1)
    np.testing.assert_allclose(np.asarray(plan.candidates[0]), np.asarray(plan.mean), atol=0.0)

    g = np.asarray(plan.G.midpoint)
    probs = np.asarray(plan.q_pi.probs)
    assert probs.shape == (33,)
    assert probs.argmax() == g.argmin()
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)
    # the mean is at least as good as the median candidate (the refit did something)
    assert g[0] <= np.median(g)


def test_a_terminal_only_designation_scores_on_both_paths():
    """A System whose *only* designation is terminal has an empty per-step tree, and
    the metrics-free accumulator used to fold it with ``total`` — which rightly
    refuses an empty tree. Emptiness is static, so the per-step fold is skipped and
    the horizon-end term is the whole of G; both paths must agree.

    (Found while comparing designation forms on Mountain Car: it is a
    pre-existing gap on the metrics-free path, not a CEM-specific one — the
    enumeration planner with ``metrics=False`` hits the same tree.)
    """
    system, params = lq_system(horizon=3)
    terminal = dataclasses.replace(
        params,
        state_targets={},
        terminal_targets={"x": Gaussian(loc=jnp.asarray([TARGET]), chol=0.2 * jnp.eye(1))},
    )
    beliefs = _belief(0.0)
    with_tree = _plan(system, terminal, beliefs, num_candidates=16, cem_iters=2, metrics=True)
    without = _plan(system, terminal, beliefs, num_candidates=16, cem_iters=2, metrics=False)
    assert sorted(with_tree.tree) == ["terminal_risk"]
    np.testing.assert_allclose(
        np.asarray(with_tree.G.midpoint), np.asarray(without.G.midpoint), atol=1e-5
    )
    # ...and the optimum still steers to the target, judged only at the horizon end
    assert float(mean_action(with_tree.candidates)[0]) > 0.0


def test_metrics_toggle_keeps_the_posterior_and_drops_the_tree():
    """The Planner protocol's rule: metrics are a reporting toggle with a known
    cost, never a semantics switch."""
    system, params = lq_system(horizon=3)
    beliefs = _belief(-0.2)
    with_tree = _plan(system, params, beliefs, num_candidates=16, cem_iters=2, metrics=True)
    without = _plan(system, params, beliefs, num_candidates=16, cem_iters=2, metrics=False)
    assert sorted(with_tree.tree) == ["risk"] and sorted(with_tree.tree["risk"]) == ["x"]
    assert without.tree is None
    np.testing.assert_allclose(
        np.asarray(with_tree.q_pi.probs), np.asarray(without.q_pi.probs), atol=1e-5
    )
    np.testing.assert_allclose(np.asarray(with_tree.mean), np.asarray(without.mean), atol=0.0)


# ---------------------------------------------------------------------- the mechanics


def test_cem_is_deterministic_under_a_fixed_key_and_jits():
    system, params = lq_system(horizon=3)
    beliefs = _belief(0.1)

    @eqx.filter_jit
    def run(params, beliefs, key):
        return _plan(params=params, system=system, beliefs=beliefs, key=key, num_candidates=32)

    a = run(params, beliefs, KEY)
    b = run(params, beliefs, KEY)
    c = run(params, beliefs, jax.random.PRNGKey(1))
    np.testing.assert_array_equal(np.asarray(a.candidates), np.asarray(b.candidates))
    assert not np.array_equal(np.asarray(a.candidates), np.asarray(c.candidates))
    assert a.candidates.shape == c.candidates.shape  # static shapes across keys


def test_shapes_are_static_in_the_budget_and_the_horizon():
    system, params = lq_system(horizon=5)
    plan = _plan(system, params, _belief(), num_candidates=12, cem_iters=2)
    assert plan.candidates.shape == (13, 5, 1)
    assert plan.mean.shape == plan.std.shape == (5, 1)
    assert np.asarray(plan.G.midpoint).shape == (13,)


def test_candidates_respect_the_declared_box():
    """The bounds are part of the declaration, so the proposal is clipped to them —
    a tight box that excludes the unconstrained optimum saturates instead."""
    system, params = lq_system(horizon=2, low=-0.1, high=0.1)
    plan = _plan(system, params, _belief(), num_candidates=64, cem_iters=5)
    cand = np.asarray(plan.candidates)
    assert cand.min() >= -0.1 - 1e-6 and cand.max() <= 0.1 + 1e-6
    assert float(plan.mean[0, 0]) == pytest.approx(0.1, abs=1e-2)  # pinned at the wall


def test_min_std_is_a_floor_under_the_refit_proposal():
    """The P1 exploration mechanism: elites collapse, the floor keeps a temperature
    under the next replan. It is a proposal-distribution floor — the same objective
    scores the candidates either way."""
    system, params = lq_system(horizon=3)
    tight = _plan(system, params, _belief(), num_candidates=64, cem_iters=6, min_std=1e-4)
    floored = _plan(system, params, _belief(), num_candidates=64, cem_iters=6, min_std=0.3)
    assert float(jnp.max(tight.std)) < 0.3
    np.testing.assert_array_less(0.3 - 1e-6, np.asarray(floored.std))


def test_the_warm_start_is_the_proposal_mean_not_a_hint():
    """``init_mean`` starts the search where the caller says (an MPC loop passes the
    previous plan shifted one step); with a narrow ``init_std`` the search stays
    there, which is what makes the shift meaningful."""
    system, params = lq_system(horizon=3)
    warm = jnp.full((3, 1), -0.8)
    plan = _plan(
        system,
        params,
        _belief(),
        num_candidates=32,
        cem_iters=1,
        init_mean=warm,
        init_std=1e-3,
        min_std=1e-4,
    )
    np.testing.assert_allclose(np.asarray(plan.mean), np.asarray(warm), atol=1e-2)


def test_elite_fraction_and_budget_validation_are_loud():
    system, params = lq_system(horizon=2)
    with pytest.raises(ValueError, match="elite_frac must lie in"):
        _plan(system, params, _belief(), elite_frac=0.0)
    with pytest.raises(ValueError, match="num_candidates must be at least 2"):
        _plan(system, params, _belief(), num_candidates=1)
    with pytest.raises(ValueError, match="cem_iters must be at least 1"):
        _plan(system, params, _belief(), cem_iters=0)


def test_policy_prior_with_cem_is_refused_rather_than_ignored():
    """The CEM planner has no ``e_proposal`` mode; a silently dropped E designation would be a
    toggle that lies, so it raises."""
    system, params = lq_system(horizon=2)
    with_E = dataclasses.replace(params, policy_prior=jnp.zeros(4))
    with pytest.raises(NotImplementedError, match="no e_proposal mode"):
        _plan(system, with_E, _belief())


# ------------------------------------------------------------------ the planner seam


def test_the_registry_entry_is_stochastic_and_the_faces_demand_a_plan_key():
    system, params = lq_system(horizon=3)
    beliefs = _belief(0.0)
    assert planner_needs_key(get_planner("cem"))
    assert not planner_needs_key(get_planner("enumeration"))

    with pytest.raises(ValueError, match="stochastic .* but no plan_key"):
        system.infer_policies(params, beliefs)

    q_pi, tree = system.infer_policies(params, beliefs, plan_key=KEY)
    assert np.asarray(q_pi.probs).shape == (65,)  # the default budget + the mean
    assert sorted(tree) == ["risk"]


def test_act_selects_a_continuous_action_through_the_seam():
    """``act`` over a continuous action space returns a float action vector, drawn
    from the planner's *own* candidate set; marginals raise instead of pretending."""
    system, params = lq_system(horizon=3)
    beliefs = _belief(0.0)
    action, q_pi, _ = system.act(
        KEY,
        params,
        beliefs,
        selection="deterministic",
        planner=make_cem_planner(num_candidates=32, cem_iters=4),
        plan_key=KEY,
        metrics=False,
    )
    action = action["a_x"]
    assert action.shape == (1,)
    assert not jnp.issubdtype(action.dtype, jnp.integer)
    assert float(action[0]) == pytest.approx(TARGET, abs=0.1)
    assert np.asarray(q_pi.probs).shape == (33,)

    with pytest.raises(ValueError, match="action marginals are a distribution over control"):
        system.act(KEY, params, beliefs, mode="marginal", plan_key=KEY)


def test_the_enumeration_planner_refuses_a_continuous_action_space():
    system, params = lq_system(horizon=2, planner="enumeration")
    with pytest.raises(ValueError, match="cannot serve a continuous action space"):
        system.infer_policies(params, _belief())
    with pytest.raises(ValueError, match="no finite policy set to enumerate"):
        system.policies(params)


def test_score_policies_and_predict_beliefs_speak_the_continuous_language():
    system, params = lq_system(horizon=2)
    beliefs = _belief(0.0)
    plan = jnp.asarray([[0.25], [0.25]])

    tree = system.score_policies(params, beliefs, plan)
    assert np.asarray(tree["risk"]["x"].midpoint).shape == ()

    steps = system.predict_beliefs(params, beliefs, plan)
    np.testing.assert_allclose(float(steps[0]["x"].mean()[0]), 0.25, atol=1e-4)
    np.testing.assert_allclose(float(steps[1]["x"].mean()[0]), 0.50, atol=1e-4)

    with pytest.raises(ValueError, match="has integer dtype"):
        system.score_policies(params, beliefs, jnp.zeros((2, 1), jnp.int32))
    with pytest.raises(ValueError, match="flat continuous control value is 1-dimensional"):
        system.score_policies(params, beliefs, jnp.zeros((2, 3)))


# ------------------------------------------------- the continuous-control declaration


def test_the_action_reaches_the_net_as_its_raw_vector_not_a_one_hot():
    """A9's half of A1's parent convention: prediction through the affine net
    reproduces ``x' = F x + G a`` for a *continuous* ``a``, which a one-hot could
    not express."""
    system, params = lq_system(horizon=1, F=0.5, G=2.0)
    got = system.update_empirical_prior(params, _belief(0.3), {"a_x": jnp.asarray([-0.25])})
    assert float(got["x"].mean()[0]) == pytest.approx(0.5 * 0.3 + 2.0 * -0.25, abs=1e-5)


def test_learn_fits_the_continuous_action_parent():
    """A2 through the continuous slot: ticks of ``(x, a) -> x'`` from a known map
    move the net toward it, with the LearnState carry threaded as ever.

    The *sensitivity to the action* is the load-bearing assertion — a one-hot
    parent could not carry it, and a net fed a constant action could not learn it.
    """
    system, params = lq_system(horizon=1)
    truth = affine_net(F=0.5, G=2.0)

    def predict(p, x, a):
        return float(p.transition["x"].condition_on((jnp.asarray([x]), jnp.asarray([a]))).loc[0])

    def slope(p):  # d(prediction)/d(action) — 2.0 in the truth
        return predict(p, 0.0, 0.5) - predict(p, 0.0, -0.5)

    before = abs(predict(params, 0.3, -0.25) - (0.5 * 0.3 + 2.0 * -0.25))
    learn_state = system.init_learn_state(params)

    @eqx.filter_jit
    def tick(params, learn_state, key):
        kx, ka = jax.random.split(key)
        x = jax.random.uniform(kx, (1,), minval=-1.0, maxval=1.0)
        a = jax.random.uniform(ka, (1,), minval=-1.0, maxval=1.0)
        return system.learn(
            params,
            {},
            {"x": Gaussian(loc=truth.condition_on((x, a)).loc, chol=1e-3 * jnp.eye(1))},
            prev_beliefs={"x": Gaussian(loc=x, chol=1e-3 * jnp.eye(1))},
            action={"a_x": a},
            lr=1e-2,
            learn_state=learn_state,
        )

    key = KEY
    for _ in range(600):
        key, k = jax.random.split(key)
        params, learn_state = tick(params, learn_state, k)

    after = abs(predict(params, 0.3, -0.25) - (0.5 * 0.3 + 2.0 * -0.25))
    assert after < 0.4 * before
    assert slope(params) == pytest.approx(2.0, abs=0.5)


def _three_controls():
    """Three factors, two continuous nodes (on ``a`` and ``c``), each read by its own net."""
    system = System(
        A_deps=(),
        B_deps=((), (), ()),
        controlled=(True, False, True),
        policy_len=1,
        epistemic=False,
        factor_names=("a", "b", "c"),
    )
    gauss = Gaussian(loc=jnp.zeros(1), chol=jnp.eye(1))
    params = SystemParams(
        likelihood={},
        transition={
            "a": MLPConditional.from_shapes((1, 2), 1, key=KEY, depth=0),
            "b": MLPConditional.from_shapes((1,), 1, key=KEY, depth=0),
            "c": MLPConditional.from_shapes((1, 1), 1, key=KEY, depth=0),
        },
        initial_state_prior={"a": gauss, "b": gauss, "c": gauss},
        control={"u_a": box(-1.0, 1.0, dim=2), "u_c": box(0.0, 3.0)},
    )
    return system, params


def test_control_layout_lays_out_the_flat_action_vector():
    system, params = _three_controls()
    layout = system._control_layout(params)
    assert layout.action_dim == 3 and system.action_dim(params) == 3
    assert layout.node_spans == ((0, 2), (2, 3))
    np.testing.assert_array_equal(np.asarray(layout.low), [-1.0, -1.0, 0.0])
    np.testing.assert_array_equal(np.asarray(layout.high), [1.0, 1.0, 3.0])
    np.testing.assert_array_equal(np.asarray(layout.block(2, jnp.asarray([7.0, 8.0, 9.0]))), [9.0])
    assert layout.dim_of(1) is None
    assert system._control_layout(dataclasses.replace(params, control={})) is None


def test_the_declaration_fences_are_loud():
    """Every way of declaring a continuous control wrong, and its error."""
    system, params = lq_system(horizon=1)
    beliefs = _belief()
    action = {"a_x": jnp.zeros(1)}
    with pytest.raises(ValueError, match="which are not control nodes"):
        system.update_empirical_prior(
            dataclasses.replace(params, control={"u_y": box()}), beliefs, action
        )
    with pytest.raises(ValueError, match=r"pass Box\(low, high\), got a Gaussian"):
        system.update_empirical_prior(
            dataclasses.replace(params, control={"u_x": Gaussian(jnp.zeros(1), jnp.eye(1))}),
            beliefs,
            action,
        )
    with pytest.raises(ValueError, match=r"Box takes \(dim,\) bounds"):
        Box(jnp.zeros((2, 1)), jnp.ones((2, 1)))
    with pytest.raises(TypeError, match="unexpected keyword argument .continuous_controls."):
        System(
            A_deps=(),
            B_deps=((),),
            controlled=(True,),
            continuous_controls=(None,),
            policy_len=1,
        )


def test_a_continuous_control_needs_a_float_reading_transition():
    """A table reads a level and a linear-Gaussian transition without ``G`` reads a mode, so a
    box on the node they read is refused, naming what to declare."""
    system, params = lq_system(horizon=2)
    lg = dataclasses.replace(
        params,
        transition={
            "x": LinearGaussianTransition(
                F=jnp.eye(1),
                innovations=__import__("hybrid_aif").ConditionalGaussian(
                    loc=jnp.zeros((1, 1)), chol=jnp.ones((1, 1, 1)), parent_ndims=1
                ),
            ),
        },
    )
    with pytest.raises(ValueError, match="give a linear-Gaussian reader a G term"):
        system.infer_policies(lg, _belief(), plan_key=KEY)

    tabular = System(
        A_deps=(),
        B_deps=((),),
        controlled=(True,),
        policy_len=1,
        epistemic=False,
    )
    tab_params = SystemParams(
        likelihood={},
        transition={
            "factor_0": __import__("hybrid_aif").ConditionalCategorical(
                log_table=jnp.zeros((2, 1)), parent_ndims=1
            ),
        },
        initial_state_prior={"factor_0": Categorical(logits=jnp.zeros(2))},
        control={"u_factor_0": box()},
    )
    beliefs = {"factor_0": Categorical(logits=jnp.zeros(2))}
    with pytest.raises(ValueError, match="read it as a level"):
        tabular.update_empirical_prior(tab_params, beliefs, {"a_factor_0": jnp.zeros(1)})


def test_the_slot_checks_the_continuous_action_width():
    """``parent_dims`` is checked against the *box's width*, not a level count — the A9
    branch of chunk 1's slot/class meeting point; the box is the referee of a continuous
    node's dimension."""
    system, params = lq_system(horizon=1)
    wrong = dataclasses.replace(
        params, transition={"x": MLPConditional.from_shapes((1, 4), 1, key=KEY, depth=0)}
    )
    with pytest.raises(ValueError, match="must agree on its dimension"):
        system.update_empirical_prior(wrong, _belief(), {"a_x": jnp.zeros(1)})


# ---------------------------------------------------------------- the P1 experiment


_SPEC = importlib.util.spec_from_file_location(
    "mountain_car_experiment", Path(__file__).resolve().parents[1] / "examples" / "mountain_car.py"
)
mc_experiment = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mc_experiment)


def test_the_p1_experiment_loop_runs_and_learns():
    """A smoke test of the whole P1 loop — observe, learn, plan, act, predict —
    for a handful of episodes at a tiny budget. No assertion that it *solves*
    Mountain Car: that is the experiment's job (and its report), not CI's.
    """
    from hybrid_aif.reference.envs.mountain_car import MountainCar

    system, params = mc_experiment.make_agent(KEY, horizon=4, width=16, depth=1)
    env = dataclasses.replace(MountainCar(), time_limit=12)
    episode = mc_experiment.make_episode(system, env, num_candidates=8, cem_iters=2, lr=1e-2)
    learn_state = system.init_learn_state(params)
    before = params.transition["state"].layers[0].weight

    for i in range(3):
        params, learn_state, log = episode(jax.random.PRNGKey(i), params, learn_state)
        assert np.isfinite(float(log.reward)) and float(log.reward) <= 0.0
        assert float(log.reward) >= -2.0 * env.time_limit
        assert int(log.agent_steps) == env.time_limit  # never solved at this budget
        assert np.isfinite(float(log.model_error))

    assert not np.array_equal(
        np.asarray(params.transition["state"].layers[0].weight), np.asarray(before)
    )
    assert set(learn_state.opt_states) == {"transition:state"}
