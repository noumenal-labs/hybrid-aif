"""The float dtype a value takes on the way in, and casting a tree of arrays."""

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest

from hybrid_aif import Categorical, cast
from hybrid_aif.composition import plan, run, sample
from hybrid_aif.composition.couple import with_mirror
from hybrid_aif.dtypes import as_float, as_floats
from tests.test_closed_loop import small_lg


def _float_dtypes(tree) -> set:
    return {
        x.dtype
        for x in jtu.tree_leaves(tree)
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
    }


def test_a_float_array_keeps_its_dtype_and_a_number_beside_it_does_not_widen_it():
    with jax.enable_x64(True):
        narrow, wide = jnp.ones(2, jnp.float32), jnp.ones(2, jnp.float64)
        assert as_float(narrow).dtype == jnp.float32
        assert as_float(wide).dtype == jnp.float64
        assert {x.dtype for x in as_floats(narrow, 2.0, 3)} == {jnp.dtype(jnp.float32)}
        assert {x.dtype for x in as_floats(narrow, jnp.arange(2))} == {jnp.dtype(jnp.float32)}
        assert {x.dtype for x in as_floats(narrow, wide, 1.0)} == {jnp.dtype(jnp.float64)}
        half = as_float(jnp.ones(2, jnp.float16))
        assert half.dtype == jnp.float32


def test_numbers_and_integers_alone_take_the_default_float_dtype():
    with jax.enable_x64(False):
        assert as_float(2).dtype == as_float([1.0, 2.0]).dtype == jnp.float32
        assert not as_float(2.0).weak_type
    with jax.enable_x64(True):
        assert as_float(2).dtype == as_float(jnp.arange(3)).dtype == jnp.float64
        assert {x.dtype for x in as_floats(1.0, 2, True)} == {jnp.dtype(jnp.float64)}


def test_cast_moves_every_float_array_and_nothing_else():
    with jax.enable_x64(True):
        _, params = small_lg()
        assert len(_float_dtypes(params)) == 2
        wide = cast(params, jnp.float64)
        assert _float_dtypes(wide) == {jnp.dtype(jnp.float64)}
        assert jtu.tree_structure(wide) == jtu.tree_structure(params)
        for before, after in zip(jtu.tree_leaves(params), jtu.tree_leaves(wide), strict=True):
            np.testing.assert_array_equal(np.asarray(before, np.float64), np.asarray(after))
        narrow = cast(wide, "float32")
        assert _float_dtypes(narrow) == {jnp.dtype(jnp.float32)}
        levels = {"u": jnp.arange(3), "on": jnp.asarray(True), "x": jnp.ones(2, jnp.float32)}
        got = cast(levels, jnp.float64)
        assert got["u"].dtype == levels["u"].dtype and got["on"].dtype == jnp.bool_
        assert got["x"].dtype == jnp.float64


def test_cast_with_where_moves_one_part_and_leaves_the_rest():
    with jax.enable_x64(True):
        _, params = small_lg()
        narrow = cast(params, jnp.float32)
        part = cast(narrow, jnp.float64, where=lambda p: p.likelihood["y"])
        assert _float_dtypes(part.likelihood["y"]) == {jnp.dtype(jnp.float64)}
        assert _float_dtypes(part.likelihood["z"]) == {jnp.dtype(jnp.float32)}
        assert _float_dtypes(part.transition) == {jnp.dtype(jnp.float32)}
        two = cast(narrow, jnp.float64, where=lambda p: (p.likelihood["y"], p.control["u_x"]))
        assert two.control["u_x"].low.dtype == jnp.float64
        assert _float_dtypes(two.likelihood["z"]) == {jnp.dtype(jnp.float32)}


def test_the_params_method_is_the_function():
    with jax.enable_x64(True):
        _, params = small_lg()
        by_method = params.cast(jnp.float64, where=lambda p: p.transition["x"])
        by_function = cast(params, jnp.float64, where=lambda p: p.transition["x"])
        assert jtu.tree_structure(by_method) == jtu.tree_structure(by_function)
        for a, b in zip(jtu.tree_leaves(by_method), jtu.tree_leaves(by_function), strict=True):
            assert a.dtype == b.dtype
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_a_model_cast_to_64_bit_is_planned_in_64_bit():
    with jax.enable_x64(True):
        system, params = small_lg()
        wide = params.cast(jnp.float64)
        prior = {"x": wide.initial_state_prior["x"]}
        _, _, plans = system.infer_policies(wide, prior, return_policies=True)
        assert plans.dtype == jnp.float64


@pytest.mark.skipif(jax.config.jax_enable_x64, reason="the refusal is for 64-bit mode off")
def test_cast_to_float64_is_refused_with_64_bit_mode_off():
    _, params = small_lg()
    with pytest.raises(ValueError, match="needs JAX's 64-bit mode"):
        cast(params, jnp.float64)


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.float16, "nonsense"])
def test_cast_refuses_a_dtype_it_does_not_compute_in(dtype):
    with pytest.raises(ValueError, match="pass float32 or float64"):
        cast({"x": jnp.ones(2)}, dtype)


# ------------------------------------------------------------------ whole models, in one dtype


def _example(name: str):
    path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tmaze():
    from tests.test_system import tmaze_system

    return tmaze_system()[:2]


def _toy():
    from tests.test_mixed_planner import toy

    return toy()[:2]


MODELS = {
    "small_lg": lambda: small_lg(),
    "tmaze": _tmaze,
    "toy": _toy,
    "switching_filter": lambda: _example("switching_filter").make_agent(epistemic=True)[:2],
    "lq_control": lambda: _example("lq_control").make_lq()[:2],
    "sensor_detour": lambda: _example("sensor_detour").make_agent()[:2],
}


def _gaussians(beliefs) -> list:
    from hybrid_aif import Gaussian

    return [
        b
        for b in jtu.tree_leaves(beliefs, is_leaf=lambda x: isinstance(x, Gaussian))
        if isinstance(b, Gaussian)
    ]


def _faces(system, params, dtype, plan_key=None):
    """Every face of one tick of ``system`` at ``params`` (cast to ``dtype``), each result's
    float dtypes: the initial priors, ``infer_states`` on observations drawn from the model's
    own world (cast to ``dtype``), ``act``, ``update_empirical_prior``, ``infer_policies`` with
    and without metrics, ``Gaussian.natural()`` on each Gaussian belief, and the planner with
    its policies."""
    key = jax.random.PRNGKey(0)
    priors = system.initial_priors(params)
    env, env_params = system.as_env(params)
    observations, _ = env.reset(key, env_params)
    observations = cast(observations, dtype)
    beliefs = system.infer_states(params, observations, priors)[0]
    action = system.act(key, params, beliefs, plan_key=plan_key)[0]
    pushed = system.update_empirical_prior(params, beliefs, action)
    total = system.infer_policies(params, beliefs, metrics=False, plan_key=plan_key)
    planned = system.infer_policies(params, priors, return_policies=True, plan_key=plan_key)
    natural = [g.natural() for g in _gaussians((priors, beliefs, pushed))]
    return {
        "observations": observations,
        "initial_priors": priors,
        "infer_states": beliefs,
        "act": action,
        "update_empirical_prior": pushed,
        "infer_policies(metrics=False)": total,
        "infer_policies": planned,
        "natural": natural,
    }


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("name", sorted(MODELS))
def test_a_model_in_one_dtype_computes_in_that_dtype(name, dtype):
    """Under 64-bit mode a model whose every float array is float32 returns float32 only, and
    one whose every float array is float64 returns float64 only: from each face of one tick
    (:func:`_faces`) and from three ticks of ``run`` against its mirror world (the world the
    model itself describes), carries included. Some cases hold even where the package makes
    an array in the default float dtype (the float64 cases, the default under 64-bit mode):
    those are guards, each paired with the other dtype's case, which fails there."""
    with jax.enable_x64(True):
        system, params = MODELS[name]()
        params = cast(params, dtype)
        want = {jnp.dtype(dtype)}
        assert _float_dtypes(params) == want
        for face, result in _faces(system, params, dtype).items():
            got = _float_dtypes(result)  # a face may return no float (a discrete action)
            assert got == want or (face in ("act", "observations", "natural") and not got), face
        both, both_params = with_mirror(system, params)
        steps = {"model": plan(), "world": sample()}
        trajectory = run(both, both_params, 3, jax.random.PRNGKey(0), steps, record_carries=True)
        assert _float_dtypes(trajectory) == want


# ------------------------------------------------- one dtype at the faces the six do not reach

DTYPES = pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["float32", "float64"])


@DTYPES
def test_a_recognition_entry_infers_in_the_models_dtype(dtype):
    """The analytic path with a recognition entry (a Gaussian measurement of ``x`` read off a
    module of the image): beliefs and the free energy's terms in the model's dtype, the image
    present and absent (an absent modality the entry covers records a zero term)."""
    import dataclasses

    from hybrid_aif.distributions import MLPConditional
    from hybrid_aif.factors import Recognition
    from tests import test_recognition_analytic as rec

    with jax.enable_x64(True):
        module = MLPConditional.from_shapes(
            (16,), 2, key=jax.random.PRNGKey(5), width=8, depth=1, frozen=True, dtype=dtype
        )
        image = MLPConditional.from_shapes(
            (2, 3), 16, key=rec.KEY, width=8, depth=1, frozen=True, dtype=dtype
        )
        system, params = rec.build(recognition=(Recognition("x", ("obs:img",)),))
        params = dataclasses.replace(
            cast(params, dtype),
            likelihood={**cast(params.likelihood, dtype), "img": image},
            recognition={"x": module},
        )
        want = {jnp.dtype(dtype)}
        assert _float_dtypes(params) == want
        observations = cast({**rec.OBS, "img": rec.IMG}, dtype)
        priors = cast(rec.PRIORS, dtype)
        for present in (None, {"img": False}):
            beliefs, detail = system.infer_states(
                params, observations, priors=priors, detail=True, present=present
            )
            assert _float_dtypes(beliefs) == want
            assert _float_dtypes(detail) == want


@DTYPES
def test_a_recognition_entry_over_a_gaussian_factor_infers_in_its_dtype(dtype):
    """``small_lg`` with a recognition entry for its Gaussian factor read off ``y`` (which the
    entry then covers): the beliefs and the covered modality's expected term are in the model's
    dtype."""
    import dataclasses

    from hybrid_aif.distributions import MLPConditional
    from hybrid_aif.factors import Recognition

    with jax.enable_x64(True):
        system, params = small_lg()
        params = cast(params, dtype)
        module = MLPConditional.from_shapes(
            (1,), 3, key=jax.random.PRNGKey(1), width=4, depth=1, frozen=True, dtype=dtype
        )
        system = dataclasses.replace(system, recognition=(Recognition("x", ("obs:y",)),))
        params = dataclasses.replace(params, recognition={"x": module})
        want = {jnp.dtype(dtype)}
        assert _float_dtypes(params) == want
        observations = {"y": jnp.ones(1, dtype), "z": jnp.ones(1, dtype)}
        beliefs, detail = system.infer_states(params, observations, detail=True)
        assert set(detail.covered_terms) == {"y"}
        assert _float_dtypes((beliefs, detail)) == want


def _conjugate(noise: str, dtype):
    """One latent ``x`` seen by ``y``, whose noise is learned: an inverse-Wishart noise beside a
    fixed head (``"noise"``), or the head and the noise together (``"mniw"``)."""
    from hybrid_aif.distributions import InverseWishart
    from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianObservationMNIW
    from tests.test_lg_head_learning import _mniw_system_and_params

    if noise == "mniw":
        likelihood = LinearGaussianObservationMNIW.from_head_noise(
            jnp.asarray([[1.5, 0.3]]), jnp.asarray([[0.4]]), strength=2.0
        )
    else:
        likelihood = LinearGaussianObservation(
            H=jnp.asarray([[1.5]]),
            c=jnp.asarray([0.3]),
            noise=InverseWishart(Psi=jnp.asarray([[0.5]]), nu=jnp.asarray(4.0)),
        )
    system, params = _mniw_system_and_params(likelihood)
    return system, cast(params, dtype)


@DTYPES
@pytest.mark.parametrize("noise", ["noise", "mniw"])
def test_a_conjugate_learner_learns_in_its_dtype(noise, dtype):
    """A learned observation noise (and head) over a Gaussian belief: one tick of
    ``infer_states`` and ``learn`` keeps the beliefs and every learned parameter in the model's
    dtype, and the learned values move."""
    with jax.enable_x64(True):
        system, params = _conjugate(noise, dtype)
        want = {jnp.dtype(dtype)}
        assert _float_dtypes(params) == want
        observations = {"y": jnp.asarray([1.2], dtype)}
        beliefs, detail = system.infer_states(params, observations, detail=True)
        learned = system.learn(params, observations, beliefs, detail=detail)
        assert _float_dtypes((beliefs, learned)) == want
        moved = [
            not np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(jtu.tree_leaves(params), jtu.tree_leaves(learned), strict=True)
        ]
        assert any(moved)


@DTYPES
def test_a_free_continuous_control_node_is_drawn_in_its_boxs_dtype(dtype):
    """``sample_step`` with no input draws a free continuous control node uniformly over its
    box, in the box's dtype: the next state stays in the model's dtype."""
    with jax.enable_x64(True):
        system, params = small_lg()
        params = cast(params, dtype)
        assert params.control["u_x"].low.dtype == dtype
        state = {"x": params.initial_state_prior["x"].loc}
        assert state["x"].dtype == dtype
        emitted, stepped = system.sample_step(jax.random.PRNGKey(1), params, state)
        assert _float_dtypes((emitted, stepped)) == {jnp.dtype(dtype)}


@DTYPES
def test_the_risk_of_a_truncated_control_belief_is_in_its_dtype(dtype):
    """``KL(q(u) || C)`` of a truncated Gaussian belief over a continuous control node against
    a Gaussian target, both in one dtype, is in that dtype."""
    from hybrid_aif.controls import continuous_node_risk
    from hybrid_aif.distributions import Gaussian, TruncatedGaussian

    with jax.enable_x64(True):
        belief = TruncatedGaussian(
            loc=jnp.asarray([0.2, -0.1], dtype),
            scale=jnp.asarray([0.5, 0.3], dtype),
            low=-jnp.ones(2, dtype),
            high=jnp.ones(2, dtype),
        )
        target = Gaussian(loc=jnp.zeros(2, dtype), chol=0.7 * jnp.eye(2, dtype=dtype))
        assert _float_dtypes((belief, target)) == {jnp.dtype(dtype)}
        risk = continuous_node_risk(belief, target)
        assert _float_dtypes(risk) == {jnp.dtype(dtype)}
        assert np.isfinite(float(risk.lower))


@DTYPES
@pytest.mark.parametrize("planner", ["factorised", "habit"])
def test_the_factorised_and_habit_planners_plan_in_the_models_dtype(planner, dtype):
    """The factorised planner over two T-mazes and the habit planner over an amortized agent
    plan in the model's dtype, with metrics and without."""
    import dataclasses

    with jax.enable_x64(True):
        if planner == "factorised":
            from tests.test_factorised_planner import two_mazes

            system, params, _ = two_mazes()
            system = dataclasses.replace(system, planner="factorised")
        else:
            from tests.test_amortized_objective import agent

            system, params, _ = agent()
        params = cast(params, dtype)
        want = {jnp.dtype(dtype)}
        assert _float_dtypes(params) == want
        priors = system.initial_priors(params)
        assert _float_dtypes(priors) == want
        key = jax.random.PRNGKey(0)
        for metrics in (True, False):
            planned = system.infer_policies(params, priors, metrics=metrics, plan_key=key)
            assert _float_dtypes(planned) == want, metrics


@DTYPES
def test_a_state_conditional_target_scores_in_the_models_dtype(dtype):
    """A Gaussian target on an observation conditional on a categorical state factor (one
    Gaussian per level, weighted by the factor's belief) scores a plan in the model's dtype."""
    from tests import test_conditional_target_state_parent as target

    with jax.enable_x64(True):
        system = target._system()
        params = cast(target._params(target._conditional()), dtype)
        want = {jnp.dtype(dtype)}
        assert _float_dtypes(params) == want
        plan_ = cast(target.PLAN, dtype)
        scored = system.score_policies(params, system.initial_priors(params), plan_)
        assert _float_dtypes(scored) == want


@DTYPES
def test_an_emissions_entropies_are_in_its_dtype(dtype):
    """The predictive entropy of a Gaussian emission over a categorical parent (a mixture of
    one Gaussian per level, no state behind it) and its entropy given the parent are in the
    emission's dtype."""
    from hybrid_aif import LinearGaussianObservation
    from hybrid_aif.objectives.continuous import emission_entropies

    with jax.enable_x64(True):
        emission = LinearGaussianObservation(
            H=jnp.zeros((3, 2, 0), dtype),
            c=jnp.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype),
            noise=0.5 * jnp.eye(2, dtype=dtype),
        )
        parent = Categorical.from_probs(jnp.asarray([0.2, 0.5, 0.3], dtype))
        assert _float_dtypes((emission, parent)) == {jnp.dtype(dtype)}
        predictive, given = emission_entropies(emission, (parent,))
        assert _float_dtypes((predictive, given)) == {jnp.dtype(dtype)}
        assert float(predictive.upper) > float(given.upper)


@DTYPES
def test_a_neural_emission_lowers_in_its_dtype(dtype):
    """A neural emission over a categorical parent lowered to its linear-Gaussian form (an
    offset and a noise per level, no state columns) keeps the network's dtype."""
    from hybrid_aif.distributions import MLPConditional
    from hybrid_aif.objectives.hybrid import neural_emission

    with jax.enable_x64(True):
        decl = MLPConditional.from_shapes(
            (3,), 2, key=jax.random.PRNGKey(2), width=4, depth=1, frozen=True, dtype=dtype
        )
        assert _float_dtypes(decl) == {jnp.dtype(dtype)}
        lowered = neural_emission(decl)
        assert lowered.H.shape == (3, 2, 0)
        assert _float_dtypes(lowered) == {jnp.dtype(dtype)}


@DTYPES
def test_pooled_readings_scale_the_novelty_in_the_models_dtype(dtype):
    """A count of readings per option (pooled readings) scales a table's novelty in the model's
    dtype: the plan's posterior and its term tree."""
    from tests.test_pooled_readings import _model, _plan

    with jax.enable_x64(True):
        system, params = _model((("s", "u_s"),), [3.0, 5.0])
        params = cast(params, dtype)
        want = {jnp.dtype(dtype)}
        assert _float_dtypes(params) == want
        assert _float_dtypes(_plan(system, params)) == want


@DTYPES
def test_a_compiled_learn_takes_a_python_rate_in_the_tables_dtype(dtype):
    """``System.compiled().learn`` with Python rates learns a model in the model's dtype."""
    from tests.test_learn import _sense_system

    with jax.enable_x64(True):
        system, params = _sense_system()
        params = cast(params, dtype)
        want = {jnp.dtype(dtype)}
        assert _float_dtypes(params) == want
        compiled = system.compiled()
        observations = {m: jnp.asarray(1, jnp.int32) for m in system.names.modalities}
        beliefs = compiled.infer_states(params, observations)[0]
        learned = compiled.learn(params, observations, beliefs, lr=0.5)
        assert _float_dtypes((beliefs, learned)) == want


def test_a_python_rate_widens_no_table_of_a_mixed_model_compiled_as_eager():
    """float32 tables beside a float64 initial prior, which ``learn`` does not update: a Python
    rate learns the tables in float32, ``System.compiled().learn`` bit for bit as
    ``System.learn``; a float64 array rate is the caller's and widens them."""
    from tests.test_learn import _sense_system

    f32, f64 = jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)
    with jax.enable_x64(True):
        system, params = _sense_system()
        params = cast(cast(params, f32), f64, where=lambda p: p.initial_state_prior)
        assert _float_dtypes((params.likelihood, params.transition)) == {f32}
        assert _float_dtypes(params.initial_state_prior) == {f64}
        priors = cast(system.initial_priors(params), f32)
        observations = {m: jnp.asarray(1, jnp.int32) for m in system.names.modalities}
        beliefs = system.infer_states(params, observations, priors)[0]
        assert _float_dtypes(beliefs) == {f32}
        before = params.likelihood["sense"].concentration
        eager = system.learn(params, observations, beliefs, lr=0.3)
        compiled = system.compiled().learn(params, observations, beliefs, lr=0.3)
        a, b = eager.likelihood["sense"].concentration, compiled.likelihood["sense"].concentration
        assert a.dtype == f32 and b.dtype == f32
        assert not np.array_equal(np.asarray(a), np.asarray(before)), "nothing was learned"
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        wide = system.compiled().learn(
            params, observations, beliefs, lr=jnp.asarray(0.3, jnp.float64)
        )
        assert wide.likelihood["sense"].concentration.dtype == f64


def test_a_run_learning_at_a_rate_wider_than_its_model_widens_what_it_learns():
    """A float32 model whose ``plan(learn=...)`` step is given a float64 array rate: the step
    learns from the second round on, so the first round returns every float in float32 and the
    wider table first shows on the settled state. The run settles there: the learned table and
    the beliefs that read it are float64, and with a float32 or a Python rate every float is
    float32."""
    from tests.test_run_nested import _shared

    f32, f64 = jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)
    with jax.enable_x64(True):
        composite, composite_params, _, _ = _shared()
        system, params = with_mirror(composite, composite_params)
        params = cast(params, f32)
        assert _float_dtypes(params) == {f32}

        def learning(lr):
            steps = {"model.rover": plan(learn={"lr": lr}), "model.mission": plan()}
            steps["world"] = sample()
            return run(system, params, 3, jax.random.PRNGKey(0), steps, record_carries=True)

        for narrow in (0.5, jnp.asarray(0.5, f32)):
            assert _float_dtypes(learning(narrow)) == {f32}
        wide = learning(jnp.asarray(0.5, f64))
        field = wide.params.parameters["model.rover.field"].concentration
        assert field.dtype == f64
        assert not np.array_equal(
            np.asarray(field), np.asarray(params.parameters["model.rover.field"].concentration)
        ), "nothing was learned"
        beliefs = wide.parts["model"].parts["rover"].carry.beliefs
        assert beliefs["model.rover.map"].logits.dtype == f64
        assert _float_dtypes(wide.emitted) == {f32}


@DTYPES
def test_a_black_box_float_port_keeps_the_dtype_its_step_returns(dtype):
    """A black box's float out-port value keeps the float dtype its step returns (an integer
    vector takes the default float dtype): what it emits and what its partner receives."""
    from hybrid_aif.composition import Identify, black_box, couple
    from hybrid_aif.controls import FLOAT

    with jax.enable_x64(True):
        system, params = couple(
            {
                "brain": black_box(outs={"force": (FLOAT, 2)}),
                "body": black_box(ins={"force": (FLOAT, 2)}),
            },
            [Identify("brain.force", "body.force")],
        )

        def push(system, key, obs, carry, params):
            name = next(n for n in system.names.emissions if n.endswith("force"))
            return {name: jnp.full((2,), 0.1, dtype)}, carry, params

        def idle(system, key, obs, carry, params):
            return {}, carry, params

        record = ("received", "emitted")
        rec = run(
            system, params, 3, jax.random.PRNGKey(0), {"brain": push, "body": idle}, record=record
        )
        emitted, received = rec["brain"].emitted["force"], rec["body"].received["force"]
        assert emitted.dtype == received.dtype == dtype
        assert float(emitted[0, 0]) == float(jnp.asarray(0.1, dtype))


@DTYPES
def test_the_survey_rover_infers_in_its_dtype(dtype):
    """The survey rover (a mixture-carried position read by a neural camera whose parents are
    the position and the terrain's level) infers in the model's dtype, its camera built in it."""
    from tests import test_survey_run as survey

    with jax.enable_x64(True):
        module, world = survey.example("survey_run"), survey.example("survey_world")
        setting, _ = survey.case_setting(module, "one-level")
        camera = survey.small_camera(world.IMG_PIXELS, dtype=dtype)
        system, params, _, _ = module.rover_of(setting, camera, None)
        params = cast(params, dtype)
        want = {jnp.dtype(dtype)}
        assert _float_dtypes(params) == want
        state = cast(world.initial_state(), dtype)
        readings = world.observe(jax.random.PRNGKey(0), state)
        observations = {k: v for k, v in readings.items() if k in system.modality_names}
        assert _float_dtypes(observations) == want
        beliefs = system.infer_states(params, observations, system.initial_priors(params))
        assert _float_dtypes(beliefs) == want


# ------------------------------------------------------------------ a mixed model under run


def _mixed_pair():
    """The pusher pair of ``float_composite`` against its mirror world, the world's transition
    made non-trivial: the model in float32 but for its transitions, which are float64 (a mixed
    model), and the same values all in float64."""
    import equinox as eqx

    from tests.test_merge_gaussian_control import float_composite

    composite, composite_params = float_composite()
    system, params = with_mirror(composite, composite_params)
    params = eqx.tree_at(
        lambda p: p.transition["world.c.x"].F, params, 0.9 * jnp.eye(2, dtype=float) + 0.01
    )
    mixed = cast(cast(params, jnp.float32), jnp.float64, where=lambda p: p.transition)
    return system, mixed, cast(mixed, jnp.float64)


def _push(system, key, obs, carry, params):
    if obs is None:
        return {}, carry, params
    return {"model.c.push": jnp.full((2,), 0.3, jnp.float32)}, carry, params


def _moved(system, key, obs, carry, params):
    """A world that moves its state by its own transition, ``x' = F x + G v``, with no noise,
    and emits it: from the initial prior's mean at the reset."""
    if obs is None:
        x = params.initial_state_prior["world.c.x"].loc
    else:
        kernel = params.transition["world.c.x"]
        x = kernel.F @ carry["x"] + kernel.G @ obs["world.c.v"]
    return {"world.c.fix": x}, {"x": x}, params


def test_a_mixed_model_runs_round_after_round_in_the_dtype_its_parts_meet_in():
    """A float32 model with float64 transitions emits float32 at the reset (its initial prior is
    float32) and float64 from round 0 on (the state has met the transition): the run keeps the
    float64 values, in what the world emits and carries and in what the model receives, and
    matches the same model all in float64 exactly, which a value rounded to float32 between
    rounds (a relative error near 1e-8) would not."""
    with jax.enable_x64(True):
        system, mixed, wide = _mixed_pair()
        assert _float_dtypes(mixed) == {jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)}
        assert mixed.initial_state_prior["world.c.x"].loc.dtype == jnp.float32
        assert mixed.transition["world.c.x"].F.dtype == jnp.float64
        assert _float_dtypes(wide) == {jnp.dtype(jnp.float64)}
        steps = {"model.c": _push, "world": _moved}
        record = ("received", "emitted")
        runs = [
            run(system, p, 6, jax.random.PRNGKey(0), steps, record=record, record_carries=True)
            for p in (mixed, wide)
        ]
        got, want = runs
        assert got["world"].emitted["c.fix"].dtype == jnp.float64
        assert got["model.c"].received["fix"].dtype == jnp.float64
        assert _float_dtypes(got["world"].carries) == {jnp.dtype(jnp.float64)}
        moved = np.asarray(got["world"].emitted["c.fix"])
        assert np.abs(moved).max() > 1.0  # the state has moved away from zero
        assert np.any(moved != moved.astype(np.float32).astype(np.float64))
        np.testing.assert_allclose(moved, want["world"].emitted["c.fix"], rtol=1e-12, atol=0)
        np.testing.assert_allclose(
            got["model.c"].received["fix"], want["model.c"].received["fix"], rtol=1e-12, atol=0
        )


def test_a_mixed_model_sampled_by_its_mirror_world_emits_in_the_wider_dtype():
    """The same mixed model against the world ``sample()`` makes of it: what the world emits and
    what the model receives are float64 every round, not rounded to float32."""
    with jax.enable_x64(True):
        system, mixed, _ = _mixed_pair()
        assert mixed.likelihood["model.c.fix"].noise.dtype == jnp.float32
        steps = {"model.c": _push, "world": sample()}
        record = ("received", "emitted")
        got = run(system, mixed, 4, jax.random.PRNGKey(0), steps, record=record)
        emitted = got["world"].emitted["c.fix"]
        received = got["model.c"].received["fix"]
        assert emitted.dtype == received.dtype == jnp.float64
        later = np.asarray(emitted)[1:]
        assert np.any(later != later.astype(np.float32).astype(np.float64))


def test_a_value_wider_than_the_state_it_replaces_is_refused():
    """Between rounds a float is only widened into the scan's state: a float64 value where the
    state holds float32 is refused, naming the value and the remedy; a Python number takes the
    state's dtype, and a narrower float is widened."""
    from hybrid_aif.composition.runner import _like

    with jax.enable_x64(True):
        narrow, wide = jnp.ones(2, jnp.float32), jnp.full(2, 0.1, jnp.float64)
        assert wide.dtype == jnp.float64 and narrow.dtype == jnp.float32
        what = "'x', returned by the step of 'world'"
        with pytest.raises(ValueError, match="'x', returned by the step of 'world' is float64"):
            _like(wide, narrow, what)
        with pytest.raises(ValueError, match=r"float32 the round before.*cast\(params, dtype\)"):
            _like(wide, narrow, what)
        assert _like(jnp.full(2, 0.1), narrow, what).dtype == jnp.float32
        widened = _like(narrow, wide, what)
        assert widened.dtype == jnp.float64
        kept = _like(wide, narrow, what, settling=True)
        assert kept.dtype == jnp.float64 and float(kept[0]) == 0.1
