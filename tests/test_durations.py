"""Durations: a round's clock on a control node's levels, and the three helpers.

``SystemParams.durations``, ``System.elapsed``, the ``duration`` record of the model description,
``merge``'s one-clock rule, and :func:`~hybrid_aif.compose` / :func:`~hybrid_aif.discretise` /
:func:`~hybrid_aif.accumulated_target`. The objective never reads a duration.

Design: docs/design/step-durations.md T1, T3, T4, T8; acceptance 4-7.
"""

import copy
import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import (
    Categorical,
    ConditionalCategorical,
    ConditionalGaussian,
    Gaussian,
    LinearGaussianTransition,
    accumulated_target,
    build,
    compose,
    describe,
    discretise,
)
from hybrid_aif.composition.merge import ActionControl, BoundaryPrior, merge, subsystem
from hybrid_aif.description import DescriptionError
from hybrid_aif.reference.raif.world import tiny_rssm
from tests.test_action_nodes import SHIFT, pusher
from tests.test_continuous_nodes import _native_continuous
from tests.test_hybrid_switching_system import initial_beliefs, make_params, make_system
from tests.test_shared_controls import two_control_model
from tests.test_system import tmaze_system

KEY = jax.random.PRNGKey(0)
TMAZE_DURATIONS = {"u_location": jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])}


def _with(params, **fields):
    return dataclasses.replace(params, **fields)


def _same_tree(a, b):
    leaves_a, leaves_b = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(leaves_a) == len(leaves_b)
    for x, y in zip(leaves_a, leaves_b, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


# ------------------------------------------------------------------------------ the declaration


def test_a_duration_names_one_control_node_with_a_number_per_level():
    system, params = tmaze_system(policy_len=2)
    beliefs = system.initial_priors(params)
    policies = system.policies(params)
    for durations in (TMAZE_DURATIONS, {"u_location": 3.0}):
        system.score_policies(_with(params, durations=durations), beliefs, policies)
    with pytest.raises(
        ValueError, match=r"not a control node; the control nodes are \['u_location'\]"
    ):
        system._validate(_with(params, durations={"a_location": jnp.ones(5)}), "a test")
    with pytest.raises(ValueError, match=r"has shape \(4,\).*has 5 level"):
        system._validate(_with(params, durations={"u_location": jnp.ones(4)}), "a test")
    for bad in ([1.0, 2.0, 0.0, 4.0, 5.0], [1.0, -2.0, 3.0, 4.0, 5.0]):
        with pytest.raises(ValueError, match="non-positive entry"):
            system._validate(_with(params, durations={"u_location": jnp.asarray(bad)}), "a test")


def test_a_duration_on_a_continuous_control_node_is_refused():
    system, params = _native_continuous()
    with pytest.raises(ValueError, match="continuous control node"):
        system._validate(_with(params, durations={"u_u": jnp.ones(1)}), "a test")


def test_a_round_has_one_clock_so_two_nodes_with_durations_are_refused():
    system, params = two_control_model()
    one = {"u_help": jnp.asarray([1.0, 2.0])}
    both = {"u_help": jnp.asarray([1.0, 2.0]), "u_mode": jnp.asarray([1.0, 2.0, 3.0])}
    system._validate(_with(params, durations=one), "a test")
    with pytest.raises(ValueError, match="one clock"):
        system._validate(_with(params, durations=both), "a test")


def test_a_duration_leaves_every_plans_term_tree_bit_for_bit_as_without_it():
    system, params = tmaze_system(policy_len=2)
    beliefs = system.initial_priors(params)
    policies = system.policies(params)
    plain = system.score_policies(params, beliefs, policies)
    timed = system.score_policies(_with(params, durations=TMAZE_DURATIONS), beliefs, policies)
    _same_tree(plain, timed)
    lg, lg_params = make_system(), make_params()
    beliefs = initial_beliefs()
    policies = lg.policies(lg_params)
    plain = lg.score_policies(lg_params, beliefs, policies)
    clock = {"u_mode": jnp.asarray([300.0, 40.0, 7.0])}
    timed = lg.score_policies(_with(lg_params, durations=clock), beliefs, policies)
    _same_tree(plain, timed)


# ------------------------------------------------------------------------------------- elapsed


def test_elapsed_is_one_when_no_durations_are_declared():
    system, params = tmaze_system()
    assert float(system.elapsed({"a_location": 2}, params)) == 1.0


def test_elapsed_is_the_realised_levels_duration_and_works_under_jit():
    system, params = tmaze_system()
    timed = _with(params, durations=TMAZE_DURATIONS)
    for level in range(5):
        assert float(system.elapsed({"a_location": level}, timed)) == level + 1.0
    jitted = jax.jit(lambda action, p: system.elapsed(action, p))
    assert float(jitted({"a_location": jnp.asarray(3)}, timed)) == 4.0
    scalar = _with(params, durations={"u_location": 2.5})
    assert float(system.elapsed({"a_location": 4}, scalar)) == 2.5
    with pytest.raises(ValueError, match="the action has no entry for 'a_location'"):
        system.elapsed({"u_location": 1}, timed)


def test_under_an_actuator_elapsed_is_the_realised_actions_duration_not_the_decided_one():
    system, params = pusher(kernel=SHIFT, noise=0.0)
    timed = _with(params, durations={"u_push": jnp.asarray([10.0, 20.0, 30.0])})
    env, env_params = system.as_env(timed)
    _, state = env.reset(KEY, env_params)
    _, _, realised = env.step_actuated(KEY, state, {"push": 0}, env_params)
    assert int(realised["push"]) == 1  # SHIFT realises u + 1
    assert float(system.elapsed(realised, timed)) == 20.0
    assert float(system.elapsed({"push": 0}, timed)) == 10.0


# ------------------------------------------------------------------------------ the format


def _timed_tmaze(durations=None):
    system, params = tmaze_system()
    return system, _with(params, durations=TMAZE_DURATIONS if durations is None else durations)


def test_a_duration_round_trips_through_the_document():
    system, params = _timed_tmaze()
    document = json.loads(json.dumps(describe(system, params, arrays="inline").document))
    assert document["variables"]["u_location"]["duration"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert "duration" not in document["variables"]["a_location"]
    built, built_params = build(document)
    np.testing.assert_array_equal(
        np.asarray(built_params.durations["u_location"]), np.asarray(TMAZE_DURATIONS["u_location"])
    )
    again = json.loads(json.dumps(describe(built, built_params, arrays="inline").document))
    assert again == document


def test_one_duration_for_every_level_is_written_as_one_number_and_read_back_per_level():
    system, params = _timed_tmaze({"u_location": 2.5})
    document = json.loads(json.dumps(describe(system, params, arrays="inline").document))
    assert document["variables"]["u_location"]["duration"] == 2.5
    _, built_params = build(document)
    np.testing.assert_array_equal(np.asarray(built_params.durations["u_location"]), [2.5] * 5)


def test_a_document_without_a_duration_builds_with_none_and_describes_without_one():
    system, params = tmaze_system()
    document = json.loads(json.dumps(describe(system, params, arrays="inline").document))
    assert "duration" not in document["variables"]["u_location"]
    built, built_params = build(document)
    assert built_params.durations == {}
    assert built == system


def test_the_schema_accepts_a_duration_and_refuses_a_non_positive_one():
    jsonschema = pytest.importorskip("jsonschema")
    from hybrid_aif import schema

    validator = jsonschema.Draft202012Validator(schema())
    system, params = _timed_tmaze()
    document = json.loads(json.dumps(describe(system, params, arrays="inline").document))
    validator.validate(document)
    for bad in ([1.0, 0.0, 3.0, 4.0, 5.0], -1.0):
        broken = copy.deepcopy(document)
        broken["variables"]["u_location"]["duration"] = bad
        assert list(validator.iter_errors(broken))


def test_a_duration_that_is_not_a_discrete_control_nodes_is_refused_in_a_sentence():
    system, params = _timed_tmaze()
    document = json.loads(json.dumps(describe(system, params, arrays="inline").document))
    on_state = copy.deepcopy(document)
    del on_state["variables"]["u_location"]["duration"]
    on_state["variables"]["context"]["duration"] = 2.0
    with pytest.raises(DescriptionError, match="not a control node"):
        build(on_state)
    wrong_length = copy.deepcopy(document)
    wrong_length["variables"]["u_location"]["duration"] = [1.0, 2.0]
    with pytest.raises(DescriptionError, match="2 number.*5 level"):
        build(wrong_length)
    non_positive = copy.deepcopy(document)
    non_positive["variables"]["u_location"]["duration"] = [1.0, 0.0, 3.0, 4.0, 5.0]
    with pytest.raises(DescriptionError, match="positive number"):
        build(non_positive)
    rssm_system, rssm_params = tiny_rssm(
        2, 1, key=KEY, units=4, deter=8, num_vars=3, num_classes=4, embed=5, mlp_units=6
    )
    continuous = json.loads(
        json.dumps(describe(rssm_system, rssm_params, arrays="inline").document)
    )
    continuous["variables"]["u_deter"]["duration"] = 2.0
    with pytest.raises(DescriptionError, match="beside that node's levels"):
        build(continuous, arrays=describe(rssm_system, rssm_params, arrays="inline").arrays)


# --------------------------------------------------------------------------------------- merge


def _two_tmazes(first=None, second=None):
    a, ap = tmaze_system(policy_len=2)
    b, bp = tmaze_system(policy_len=2)
    ap = _with(ap, durations={} if first is None else first)
    bp = _with(bp, durations={} if second is None else second)
    return {"a": a, "b": b}, {"a": ap, "b": bp}


def test_merge_carries_one_components_durations_under_its_prefix():
    systems, params = _two_tmazes(second=TMAZE_DURATIONS)
    composite, composite_params = merge(systems, params)
    assert list(composite_params.durations) == ["b.u_location"]
    np.testing.assert_array_equal(
        np.asarray(composite_params.durations["b.u_location"]),
        np.asarray(TMAZE_DURATIONS["u_location"]),
    )
    action = {"a.a_location": 0, "b.a_location": 3}
    assert float(composite.elapsed(action, composite_params)) == 4.0
    _, view_params = subsystem(composite, composite_params, "b")
    assert list(view_params.durations) == ["b.u_location"]
    assert subsystem(composite, composite_params, "a")[1].durations == {}


def test_merge_refuses_two_components_that_both_declare_durations():
    systems, params = _two_tmazes(first=TMAZE_DURATIONS, second=TMAZE_DURATIONS)
    with pytest.raises(ValueError, match=r"one clock"):
        merge(systems, params)


def test_merge_refuses_a_binding_that_identifies_the_clock_node_with_a_driver():
    systems, params = _two_tmazes(second=TMAZE_DURATIONS)
    with pytest.raises(ValueError, match=r"removes control node 'b.u_location'.*durations"):
        merge(systems, params, [ActionControl("a.u_location", "b.u_location")])


def test_merge_refuses_a_boundary_prior_that_terminates_the_clock_node():
    systems, params = _two_tmazes(second=TMAZE_DURATIONS)
    with pytest.raises(ValueError, match=r"removes control node 'b.u_location'.*durations"):
        merge(systems, params, [BoundaryPrior("b.u_location", 1)])


def test_a_clock_node_that_is_the_driver_of_an_identification_carries_its_durations():
    systems, params = _two_tmazes(first=TMAZE_DURATIONS)
    composite, composite_params = merge(
        systems, params, [ActionControl("a.u_location", "b.u_location")]
    )
    assert list(composite_params.durations) == ["a.u_location"]
    assert float(composite.elapsed({"a.a_location": 2}, composite_params)) == 3.0


def test_a_float_duration_keeps_its_dtype_under_x64():
    system, params = tmaze_system(policy_len=2)
    with jax.enable_x64(True):
        durations = {"u_location": jnp.asarray([0.1, 0.2, 0.3, 0.4, 0.5], jnp.float64)}
        timed = _with(params, durations=durations)
        assert system.elapsed({"a_location": 0}, timed).dtype == jnp.float64
        assert system._durations(timed, "elapsed")["u_location"].dtype == jnp.float64


# --------------------------------------------------------------------------------- the helpers


def _belief_step(F, b, Q, G, u, loc, cov):
    return F @ loc + G @ u + b, F @ cov @ F.T + Q


def test_compose_is_n_applications_of_the_per_tick_kernel_in_belief_space():
    rng = np.random.default_rng(0)
    d, du, n = 2, 1, 5
    F = jnp.asarray(rng.normal(size=(2, d, d)) * 0.4 + np.eye(d) * 0.5, jnp.float32)
    G = jnp.asarray(rng.normal(size=(2, d, du)), jnp.float32)
    chol = jnp.asarray(np.tril(rng.normal(size=(2, d, d))) + 1.5 * np.eye(d), jnp.float32)
    b = jnp.asarray(rng.normal(size=(2, d)), jnp.float32)
    kernel = LinearGaussianTransition(
        F=F, innovations=ConditionalGaussian(loc=b, chol=chol, parent_ndims=1), G=G
    )
    stepped = compose(kernel, n)
    loc0 = jnp.asarray([0.3, -0.7], jnp.float32)
    cov0 = jnp.asarray([[0.5, 0.1], [0.1, 0.8]], jnp.float32)
    u = jnp.asarray([0.9], jnp.float32)
    for mode in range(2):
        loc, cov = loc0, cov0
        for _ in range(n):
            loc, cov = _belief_step(
                F[mode], b[mode], chol[mode] @ chol[mode].T, G[mode], u, loc, cov
            )
        Fn, Gn = stepped.F[mode], stepped.G[mode]
        Qn = stepped.innovations.chol[mode] @ stepped.innovations.chol[mode].T
        got_loc, got_cov = _belief_step(Fn, stepped.innovations.loc[mode], Qn, Gn, u, loc0, cov0)
        np.testing.assert_allclose(got_loc, loc, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(got_cov, cov, rtol=1e-4, atol=1e-5)


def test_compose_keeps_a_shared_map_shared():
    inn = ConditionalGaussian(loc=jnp.zeros((1, 2)), chol=0.3 * jnp.eye(2)[None], parent_ndims=1)
    kernel = LinearGaussianTransition(F=0.9 * jnp.eye(2), innovations=inn)
    stepped = compose(kernel, 3)
    assert stepped.F.shape == (2, 2) and stepped.G is None
    np.testing.assert_allclose(stepped.F, 0.9**3 * np.eye(2), rtol=1e-6)
    with pytest.raises(ValueError, match="positive int"):
        compose(kernel, 0)


def test_discretise_matches_the_closed_form_on_a_scalar_system():
    a, q, b, dt = -0.7, 0.4, 1.3, 0.9
    kernel = discretise(jnp.asarray([[a]]), jnp.asarray([[b]]), jnp.asarray([[q]]), dt)
    F = np.exp(a * dt)
    Qd = q * (np.exp(2 * a * dt) - 1) / (2 * a)
    G = b * (np.exp(a * dt) - 1) / a
    np.testing.assert_allclose(kernel.F, [[F]], rtol=1e-5)
    np.testing.assert_allclose(kernel.G, [[G]], rtol=1e-5)
    chol = np.asarray(kernel.innovations.chol)[0, 0, 0]
    np.testing.assert_allclose(chol**2, Qd, rtol=1e-5)


def test_discretise_takes_one_duration_per_level_as_a_per_mode_kernel():
    A, Q = jnp.asarray([[-0.5]]), jnp.asarray([[0.2]])
    dts = jnp.asarray([300.0, 40.0]) / 100.0
    kernel = discretise(A, None, Q, dts)
    assert kernel.F.shape == (2, 1, 1) and kernel.innovations.parent_shape == (2,)
    assert kernel.G is None
    np.testing.assert_allclose(
        np.asarray(kernel.F)[:, 0, 0], np.exp(-0.5 * np.asarray(dts)), rtol=1e-5
    )
    with pytest.raises(ValueError, match="modes=3"):
        discretise(A, None, Q, dts, modes=3)


def test_composing_a_discretised_tick_n_times_is_the_discretisation_of_n_ticks():
    with jax.enable_x64(True):
        rng = np.random.default_rng(1)
        d, du, n, dt = 2, 1, 4, 0.15
        A = jnp.asarray(rng.normal(size=(d, d)) * 0.5 - np.eye(d))
        B = jnp.asarray(rng.normal(size=(d, du)))
        L = np.tril(rng.normal(size=(d, d))) + 1.2 * np.eye(d)
        Q = jnp.asarray(L @ L.T)
        one = discretise(A, B, Q, dt)
        long = discretise(A, B, Q, n * dt)
        composed = compose(one, n)
        np.testing.assert_allclose(composed.F, long.F, atol=1e-6)
        np.testing.assert_allclose(composed.G, long.G, atol=1e-6)
        np.testing.assert_allclose(composed.innovations.chol, long.innovations.chol, atol=1e-6)


def test_an_accumulated_gaussian_target_has_each_levels_duration_times_the_precision():
    durations = jnp.asarray([300.0, 40.0, 1.0])
    cov = jnp.asarray([[0.5, 0.1], [0.1, 0.4]])
    target = Gaussian(loc=jnp.asarray([0.2, -0.1]), chol=jnp.linalg.cholesky(cov))
    conditional = accumulated_target(target, durations)
    assert isinstance(conditional, ConditionalGaussian) and conditional.parent_ndims == 1
    precision = np.linalg.inv(np.asarray(cov))
    for k, dur in enumerate(np.asarray(durations)):
        chol = np.asarray(conditional.chol[k])
        np.testing.assert_allclose(np.linalg.inv(chol @ chol.T), dur * precision, rtol=1e-4)
        np.testing.assert_array_equal(np.asarray(conditional.loc[k]), np.asarray(target.loc))


def test_an_accumulated_categorical_target_is_each_levels_log_table_renormalised():
    durations = jnp.asarray([3.0, 0.5])
    logits = jnp.log(jnp.asarray([0.6, 0.3, 0.1]))
    conditional = accumulated_target(Categorical(logits=logits), durations)
    assert isinstance(conditional, ConditionalCategorical) and conditional.parent_ndims == 1
    assert conditional.log_table.shape == (3, 2)  # child axis first, then the level axis
    log_c = np.asarray(jax.nn.log_softmax(logits))
    for k, dur in enumerate(np.asarray(durations)):
        column = np.asarray(conditional.log_probs[:, k])
        raw = dur * log_c
        np.testing.assert_allclose(column, raw - np.log(np.sum(np.exp(raw))), rtol=1e-5)
        np.testing.assert_allclose(np.sum(np.exp(column)), 1.0, rtol=1e-5)


def test_scoring_an_accumulated_target_is_the_per_tick_score_times_the_duration_up_to_a_constant():
    durations = jnp.asarray([3.0, 0.5])
    logits = jnp.log(jnp.asarray([0.6, 0.3, 0.1]))
    conditional = accumulated_target(Categorical(logits=logits), durations)
    log_c = np.asarray(jax.nn.log_softmax(logits))
    rng = np.random.default_rng(2)
    gaps = {0: [], 1: []}
    for _ in range(5):
        q = rng.dirichlet(np.ones(3))
        for k, dur in enumerate(np.asarray(durations)):
            level = -float(q @ np.asarray(conditional.log_probs[:, k]))
            gaps[k].append(level - dur * -float(q @ log_c))
    for k in gaps:  # the difference is the level's normaliser, whatever q is
        np.testing.assert_allclose(gaps[k], gaps[k][0], atol=1e-5)
    #  and the Gaussian case: a Gauss-Hermite rule is exact for the quadratic, so the difference
    #  (the level's normaliser) is the same under every q
    target = Gaussian(loc=jnp.asarray([0.0]), chol=jnp.asarray([[0.7]]))
    conditional_g = accumulated_target(target, durations)
    nodes, weights = np.polynomial.hermite_e.hermegauss(12)
    weights = weights / weights.sum()
    for k, dur in enumerate(np.asarray(durations)):
        diffs = []
        for mean, var in ((0.3, 0.2), (-1.0, 0.9), (2.0, 0.05)):
            x = jnp.asarray(mean + np.sqrt(var) * nodes)[:, None]
            level_ce = -float(weights @ np.asarray(conditional_g.condition_on((k,)).log_prob(x)))
            tick_ce = -float(weights @ np.asarray(target.log_prob(x)))
            diffs.append(level_ce - dur * tick_ce)
        np.testing.assert_allclose(diffs, diffs[0], atol=1e-4)
