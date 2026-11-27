"""The float dtype rule at the arrays the package makes (``hybrid_aif.dtypes``).

An array the package creates takes its dtype from the values it is combined with, the default
float dtype when there is nothing to read one from, and no value is narrowed implicitly. With
JAX's 64-bit mode off every float is float32, so each test runs under ``jax.enable_x64(True)``,
asserts the dtypes of what it builds before it asserts anything about the result, and checks a
value where a float32 rounding would change it.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Categorical, ConditionalCategorical, LinearGaussianObservation, replay
from hybrid_aif.composition.merge import subsystem
from hybrid_aif.dtypes import cast
from hybrid_aif.inference.discrete import fpi, mean_field_free_energy

F32, F64 = np.dtype(np.float32), np.dtype(np.float64)


def float_dtypes(tree) -> set:
    """The float dtypes among the arrays of ``tree``."""
    return {
        np.dtype(x.dtype)
        for x in jax.tree_util.tree_leaves(tree)
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
    }


# ------------------------------------------------------------------ the mean-field sweep


def _sweep_inputs(prior_dtype, likelihood_dtype):
    """One factor of three levels, one modality of two outcomes."""
    table = jnp.asarray([[0.7, 0.2, 0.4], [0.3, 0.8, 0.6]], likelihood_dtype)
    ll = ConditionalCategorical(log_table=jnp.log(table), parent_ndims=1)
    prior = Categorical(logits=jnp.log(jnp.asarray([0.5, 0.3, 0.2], prior_dtype)))
    return (ll,), ((0,),), (prior,), (jnp.asarray(1, jnp.int32),)


def test_the_sweep_runs_in_the_widest_dtype_of_its_priors_and_likelihoods():
    """float32 priors beside a float64 likelihood: the sweep's carry starts in float64 (it
    raised a scan carry mismatch when it started in the priors' float32), and the posterior is
    the all-float64 one."""
    with jax.enable_x64(True):
        lls, deps, priors, obs = _sweep_inputs(jnp.float32, jnp.float64)
        assert priors[0].logits.dtype == F32 and lls[0].log_table.dtype == F64
        (q,) = fpi(lls, deps, priors, obs)
        assert q.logits.dtype == F64
        wide = _sweep_inputs(jnp.float64, jnp.float64)
        (want,) = fpi(*wide[:2], (cast(priors[0], jnp.float64),), obs)
        np.testing.assert_allclose(np.asarray(q.probs), np.asarray(want.probs), rtol=1e-12)
        narrow = _sweep_inputs(jnp.float32, jnp.float32)
        (q32,) = fpi(*narrow)
        assert q32.logits.dtype == F32


def test_the_free_energy_of_a_float32_sweep_is_float32():
    """The tick's free energy starts its sum in the posteriors' dtype: float32 for an all-float32
    sweep (a float64 start widened it), float64 for an all-float64 one."""
    with jax.enable_x64(True):
        for dtype in (jnp.float32, jnp.float64):
            lls, deps, priors, obs = _sweep_inputs(dtype, dtype)
            posteriors = fpi(lls, deps, priors, obs)
            assert float_dtypes((lls, priors, posteriors)) == {np.dtype(dtype)}
            energy = mean_field_free_energy(lls, deps, priors, posteriors, obs)
            assert energy.dtype == dtype


# ------------------------------------------------------------------------------ replay


def test_a_replay_of_a_model_with_one_float64_offset_learns_in_float64():
    """The process-noise model in float32 but for its observation offset, float64: the replay's
    learning carry widens to float64 where the two meet (it raised a scan carry mismatch), and
    the learned noise is the all-float64 model's."""
    from tests import test_process_noise as pn

    with jax.enable_x64(True):
        system = pn._system()
        prior = pn.InverseWishart(Psi=pn._f32(4.0 * np.eye(2)), nu=pn._f32(4.0))
        observation = LinearGaussianObservation(
            H=pn._f32(pn.H), c=jnp.zeros(2, jnp.float64), noise=pn._f32(np.sqrt(pn.R) * np.eye(2))
        )
        params = dataclasses.replace(pn._params(prior), likelihood={"y": observation})
        assert float_dtypes(params) == {F32, F64}
        ys = pn._trace(40)
        kwargs = dict(learn=True, lr=0.0, lr_by_slot={"transition:z:noise": 1.0}, metrics=False)
        out = replay(system, params, {"y": ys}, {}, **kwargs)
        learned = out.params.transition["z"].noise_prior
        assert learned.Psi.dtype == F64 and out.beliefs["z"].loc.dtype == F64
        wide = replay(system, cast(params, jnp.float64), {"y": ys}, {}, **kwargs)
        np.testing.assert_allclose(
            np.asarray(learned.Psi),
            np.asarray(wide.params.transition["z"].noise_prior.Psi),
            rtol=1e-8,
        )


def test_a_float32_growing_model_replays_and_learns_in_float32():
    """A float32 document model under 64-bit mode (a Gaussian source, a categorical source, a
    growable mixture): its beliefs, free energy and learned params stay float32; the sweep,
    the free energy's sum and a Python learning rate are made in the model's dtype."""
    from hybrid_aif.description import build
    from tests.test_v019_lagged_documents import document, sources_trace

    with jax.enable_x64(True):
        system, params = build(document("sources"))
        assert F64 in float_dtypes(params)  # inline values build in the default dtype
        params = cast(params, jnp.float32)
        assert float_dtypes(params) == {F32}
        xs, kinds = sources_trace()
        obs = {"obs_x": xs, "obs_kind": kinds}
        out = replay(system, params, obs, {}, learn=True, lr=0.3, metrics=False)
        assert float_dtypes((out.beliefs, out.free_energy, out.params)) == {F32}


def test_a_python_learning_rate_is_not_rounded_to_float32_for_a_float64_model():
    """``lr=0.1`` on a float64 model learns as the float64 array 0.1 does (it was passed to the
    scan as float32, 0.1 rounded)."""
    from tests.test_learn import _sense_system

    with jax.enable_x64(True):
        system, params = _sense_system()
        params = cast(params, jnp.float64)
        assert float_dtypes(params) == {F64}
        observations = {m: jnp.asarray([0, 1, 0, 1], jnp.int32) for m in system.names.modalities}
        actions = {n: jnp.zeros((4,), jnp.int32) for n in system.names.actions}
        by_number = replay(system, params, observations, actions, learn=True, lr=0.1).params
        by_array = replay(
            system, params, observations, actions, learn=True, lr=jnp.asarray(0.1, jnp.float64)
        ).params
        for a, b in zip(
            jax.tree_util.tree_leaves(by_number), jax.tree_util.tree_leaves(by_array), strict=True
        ):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_a_python_learning_rate_takes_the_dtype_of_the_tables_it_scales():
    """A float32 model with one unrelated float64 leaf (its policy precision) learns at
    ``lr=0.1`` in float32: the rate takes the dtype of the statistics it scales, as the number
    would, and learns exactly as the float32 array 0.1 does. A float64 array rate given by the
    caller keeps its dtype and widens the tables it scales (and only those)."""
    from tests.test_learn import _sense_system

    with jax.enable_x64(True):
        system, params = _sense_system()
        params = cast(params, jnp.float32)
        params = dataclasses.replace(params, precision=jnp.asarray(4.0, jnp.float64))
        assert float_dtypes(params) == {F32, F64}
        assert float_dtypes((params.likelihood, params.transition)) == {F32}
        observations = {m: jnp.asarray([0, 1, 0, 1], jnp.int32) for m in system.names.modalities}
        actions = {n: jnp.zeros((4,), jnp.int32) for n in system.names.actions}

        def learned(lr):
            return replay(system, params, observations, actions, learn=True, lr=lr)

        by_number = learned(0.1)
        tables = (by_number.params.likelihood, by_number.params.transition)
        assert float_dtypes((tables, by_number.beliefs)) == {F32}
        by_array = learned(jnp.asarray(0.1, jnp.float32))
        for a, b in zip(
            jax.tree_util.tree_leaves(by_number.params),
            jax.tree_util.tree_leaves(by_array.params),
            strict=True,
        ):
            assert a.dtype == b.dtype
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        wide = learned(jnp.asarray(0.1, jnp.float64)).params
        for before, after in zip(
            jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(wide), strict=True
        ):
            if after.dtype == F32:  # a table the rate does not scale is left as it was
                np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
        assert F64 in float_dtypes((wide.likelihood, wide.transition))


# ------------------------------------------------------------------------- the views


def test_a_view_averages_a_partner_node_in_the_tables_dtype():
    """A float64 composite's view averages the partner's node out of a table with weights in the
    table's float64: the uniform weight is 1/5 and a declared prior keeps its values (both were
    rounded to float32)."""
    from tests.test_merge_nodes import same_step

    with jax.enable_x64(True):
        composite, cp = same_step()
        assert float_dtypes(cp.transition) == {F64}
        table = np.exp(np.asarray(cp.transition["world.location"].log_probs, np.float64))
        _, uniform = subsystem(composite, cp, "world")
        got = uniform.transition["world.location"]
        assert got.log_table.dtype == F64
        np.testing.assert_allclose(
            np.exp(np.asarray(got.log_probs))[..., 0], table.mean(axis=-1), rtol=1e-13
        )
        prior = jnp.asarray([0.1, 0.2, 0.3, 0.15, 0.25], jnp.float64)
        _, weighted = subsystem(composite, cp, "world", inputs={"agent.a_location": prior})
        got = np.exp(np.asarray(weighted.transition["world.location"].log_probs))[..., 0]
        np.testing.assert_allclose(got, table @ np.asarray(prior), rtol=1e-13)


def test_a_float32_model_with_a_modality_absent_infers_in_float32():
    """``infer_states`` on the float32 document model with the lagged readings left out: the
    absent modalities' log-likelihood tensors in the sweep and the free energy are made in
    float32 (the tensors were float64 zeros, which widened the posteriors)."""
    from hybrid_aif.description import build
    from tests.test_v019_lagged_documents import document, sources_trace

    with jax.enable_x64(True):
        system, params = build(document("sources"))
        params = cast(params, jnp.float32)
        assert float_dtypes(params) == {F32}
        xs, kinds = sources_trace()
        beliefs, detail = system.infer_states(
            params, {"obs_x": xs[0], "obs_kind": kinds[0]}, detail=True
        )
        assert float_dtypes(beliefs) == {F32}
        assert detail.free_energy.dtype == F32


def test_the_free_energy_of_a_float32_linear_gaussian_model_is_float32():
    """A model with no categorical factor sums its free energy from the modalities' evidence
    alone: the sum starts in their float32 (a float64 zero widened it)."""
    from tests import test_process_noise as pn

    with jax.enable_x64(True):
        system, params = pn._system(), pn._params(pn._iw())
        assert float_dtypes(params) == {F32}
        beliefs, detail = system.infer_states(params, {"y": pn.Y0}, detail=True)
        assert float_dtypes(beliefs) == {F32}
        assert detail.free_energy.dtype == F32
        wide = cast(params, jnp.float64)
        _, wide_detail = system.infer_states(wide, {"y": pn.Y0}, detail=True)
        assert wide_detail.free_energy.dtype == F64


# ------------------------------------------------------------------------------- run


def test_the_first_warm_start_is_in_the_control_box_dtype():
    """``plan(warm_start=True)`` hands the planner zeros in its first round, in the dtype of the
    control box: float64 for a float64 box (they were float32 zeros)."""
    from hybrid_aif.composition import plan, run
    from hybrid_aif.planning import make_gradient_planner
    from tests.test_plan_warm_start import reading, steered

    inner, seen = make_gradient_planner(steps=0), []

    def planner(ctx, use_bound="midpoint", metrics=True, init=None):
        seen.append(np.dtype(init.dtype))
        return inner(ctx, use_bound=use_bound, metrics=metrics, init=init)

    planner.needs_key, planner.takes_init = False, True
    with jax.enable_x64(True):
        system, params = steered()
        assert params.control["mind.u_u"].low.dtype == F64
        step = plan(planner, warm_start=True)
        run(system, params, 1, jax.random.PRNGKey(0), {"mind": step, "world": reading})
        assert seen and seen[0] == F64


def test_a_maybe_level_vector_and_a_null_value_keep_or_take_the_float_dtype():
    """The vector a sliced view takes for a variable that may have no value keeps a float64
    vector's dtype (it was cast to float32), and a continuous node's null value, with nothing
    to read a dtype from, takes the default float dtype."""
    from hybrid_aif.composition.runner import Maybe, _absent, _level_of, _null
    from hybrid_aif.controls import FLOAT, LEVEL

    with jax.enable_x64(True):
        probs = jnp.asarray([0.2, 0.3, 0.5], jnp.float64)
        got = _level_of(Maybe(probs, jnp.asarray(True)), 3)
        assert got.dtype == F64
        np.testing.assert_array_equal(np.asarray(got), np.asarray(probs))
        absent = _level_of(Maybe(probs, jnp.asarray(False)), 3)
        assert absent.dtype == F64 and float(absent[0]) == 1.0 / 3.0
        one_hot = _level_of(Maybe(jnp.asarray(1, jnp.int32), jnp.asarray(True)), 3, F32)
        assert one_hot.dtype == F32
        assert _null((FLOAT, 2)).dtype == F64 and _null((LEVEL, 3)).dtype == jnp.int32
    with jax.enable_x64(False):
        assert _null((FLOAT, 2)).dtype == F32
    del _absent


def test_the_actuator_draws_from_a_float64_kernel_unrounded():
    """The actuator's draws at a float64 kernel are the categorical draws of its float64 log
    column (the kernel was cast to float32, so the draws came from another stream)."""
    from hybrid_aif.composition.runner import _actuate
    from hybrid_aif.envs.generative import ACTUATOR_TAG

    with jax.enable_x64(True):
        kernel = jnp.asarray([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3], [0.1, 0.3, 0.6]], jnp.float64)
        assert kernel.dtype == F64
        got, want = [], []
        for s in range(64):
            key = jax.random.PRNGKey(s)
            got.append(int(_actuate(key, 1, [(0, kernel, jnp.asarray(2))])[0]))
            k = jax.random.split(jax.random.fold_in(key, ACTUATOR_TAG), 1)[0]
            want.append(int(jax.random.categorical(k, jnp.log(kernel[:, 2]))))
        assert got == want


# ---------------------------------------------------- the beliefs over control and action nodes


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_control_and_action_node_beliefs_take_the_models_dtype(dtype):
    """A model in one float dtype: the uniform prior over its control node, the uniform prior
    over its observed action node (eager and compiled) and the prediction of the action node at
    a clamped level are in that dtype (they were made in the default float dtype, or float32)."""
    from hybrid_aif.system.compiled import _complete_priors
    from tests.test_action_node_modalities import scalar_model

    with jax.enable_x64(True):
        system, params = scalar_model()
        params = cast(params, dtype)
        assert float_dtypes(params) == {np.dtype(dtype)}
        priors = system.initial_priors(params)
        assert float_dtypes(priors["u"]) == {np.dtype(dtype)}  # the control node's prior
        pushed = system.update_empirical_prior(params, priors, {"a": jnp.asarray(1, jnp.int32)})
        assert set(pushed) == {"x", "u", "a"}
        assert float_dtypes(pushed) == {np.dtype(dtype)}  # "a": the clamped level's one-hot
        np.testing.assert_array_equal(np.asarray(pushed["a"].probs), [0.0, 1.0, 0.0])
        states = {"x": priors["x"]}
        completed = _complete_priors(system, params, states, "infer_states")
        assert set(completed) == {"x", "u", "a"}
        assert float_dtypes(completed) == {np.dtype(dtype)}


# ------------------------------------------------------------------------------ learn


def test_a_traced_presence_flag_does_not_round_the_learning_rate():
    """A float64 model learning at ``lr=0.1`` under a traced presence flag (as a replay's scan
    hands it) learns what it learns with the modality statically present: the flag scales the
    rate in the rate's dtype (it was a float32 1.0, which rounded 0.1 to float32)."""
    from tests.test_learn import _sense_system, _tick_inputs

    with jax.enable_x64(True):
        system, params = _sense_system()
        params = cast(params, jnp.float64)
        assert float_dtypes(params) == {F64}
        obs, beliefs = _tick_inputs(system, params)
        positional_obs = tuple(obs[m] for m in system.names.modalities)
        positional_beliefs = tuple(beliefs[f] for f in system.names.states)
        inner = system._params_in(params, "learn")

        def learned(present):
            return system._learn(inner, positional_obs, positional_beliefs, lr=0.1, present=present)

        static = learned(None)
        traced = jax.jit(learned)((jnp.asarray(True), jnp.asarray(True)))
        a, b = static.likelihood[0].concentration, traced.likelihood[0].concentration
        assert a.dtype == F64 and b.dtype == F64
        np.testing.assert_array_equal(np.asarray(b), np.asarray(a))


# ------------------------------------------------------------------------- black boxes


def test_a_float64_in_port_drives_a_positional_environment_unrounded():
    """``env_step``'s continuous action is its float in-ports' values in their own dtype: a
    float64 force of 0.1 steps the environment as 0.1 does (it was cast to float32)."""
    from hybrid_aif.composition import Identify, black_box, couple, env_step, run
    from hybrid_aif.controls import FLOAT
    from hybrid_aif.reference.envs.mountain_car import MountainCar
    from tests.test_black_box import names_of

    with jax.enable_x64(True):
        car = MountainCar()
        controller = black_box(ins={"x": (FLOAT, 2)}, outs={"force": (FLOAT, 1)})
        plant = black_box(ins={"force": (FLOAT, 1)}, outs={"x": (FLOAT, 2)})
        system, params = couple(
            {"brain": controller, "car": plant},
            [Identify("brain.force", "car.force"), Identify("car.x", "brain.x")],
        )
        force = jnp.full((1,), 0.1, jnp.float64)
        assert force.dtype == F64 and float(force[0]) != float(np.float32(0.1))

        def push(system, key, obs, carry, params):
            return {names_of(system)["force"]: force}, None, params

        key = jax.random.PRNGKey(0)
        rec = run(system, params, 3, key, {"brain": push, "car": env_step(car)})
        _, state = car.reset(jax.random.fold_in(jax.random.split(key)[1], 1))
        for t in range(3):
            _, state = car.step(None, state, force)
            np.testing.assert_allclose(rec["car"].emitted["x"][t], state[0], rtol=1e-14, atol=0)


# ------------------------------------------------------------------------ the description


def test_a_built_kernel_stored_in_float16_is_built_in_float32():
    """An actuator kernel the arrays file stores in float16 builds at least float32, its values
    the float16 ones; one stored in float32 builds in float32 under 64-bit mode."""
    from hybrid_aif.description import build, describe
    from tests.test_action_nodes import pusher

    with jax.enable_x64(True):
        kernel = jnp.asarray([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]], jnp.float32)
        system, params = pusher(kernel=kernel)
        document, arrays = describe(system, params, arrays="file")
        keys = [k for k in arrays if k.startswith("K_push")]
        assert len(keys) == 1
        single = {**arrays, keys[0]: np.asarray(arrays[keys[0]], np.float32)}
        _, built = build(document, arrays=single)
        assert built.actuator["push"].dtype == F32
        half = {**arrays, keys[0]: np.asarray(arrays[keys[0]], np.float16)}
        assert half[keys[0]].dtype == np.float16
        _, built = build(document, arrays=half)
        got = built.actuator["push"]
        assert got.dtype == F32
        want = np.exp(np.asarray(half[keys[0]], np.float64))  # the file holds log values
        np.testing.assert_allclose(np.asarray(got), want, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(got), np.asarray(kernel), rtol=2e-3)


def test_a_built_kernel_takes_its_stored_dtype_and_a_duration_the_default():
    """``build`` gives an actuator kernel read from the arrays file its stored dtype, and an
    inline duration the build's default dtype (docs/description-format.md): both float64 under
    64-bit mode, the kernel's values those stored (both were made float32)."""
    from hybrid_aif.description import build, describe
    from tests.test_action_nodes import pusher

    with jax.enable_x64(True):
        kernel = jnp.asarray([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]], jnp.float64)
        system, params = pusher(kernel=kernel)
        params = dataclasses.replace(params, durations={"u_push": jnp.asarray([2.0, 1.0, 3.0])})
        assert params.actuator["push"].dtype == F64
        document, arrays = describe(system, params, arrays="file")
        stored = [np.asarray(a) for a in arrays.values() if np.shape(a) == (3, 3)]
        assert stored and all(a.dtype == F64 for a in stored)
        _, built = build(document, arrays=arrays)
        got = built.actuator["push"]
        assert got.dtype == F64
        np.testing.assert_allclose(np.asarray(got), np.asarray(kernel), rtol=1e-15)
        assert float(got[0, 0]) != float(np.float32(0.7))
        assert built.durations["u_push"].dtype == F64
        _, inline = build(describe(system, params, arrays="inline").document)
        assert inline.actuator["push"].dtype == F64 and inline.durations["u_push"].dtype == F64
        narrow = {k: np.asarray(v, np.float32) for k, v in arrays.items()}
        _, from_float32 = build(document, arrays=narrow)
        assert from_float32.actuator["push"].dtype == F32


# ------------------------------------------------------------------- processes and draws


def test_a_positional_process_keeps_its_tables_dtype_and_its_null_action_is_default():
    """``envs.generative.make`` keeps float64 tables float64 (they were cast to float32), and
    a continuous process's null action, with nothing to read a dtype from, is in the default
    float dtype."""
    from hybrid_aif.envs.generative import make

    with jax.enable_x64(True):
        A = (jnp.asarray([[0.9, 0.2], [0.1, 0.8]], jnp.float64),)
        B = (jnp.asarray([[[0.7], [0.3]], [[0.3], [0.7]]], jnp.float64),)
        D = (jnp.asarray([0.6, 0.4], jnp.float64),)
        _, params = make(A, B, D)
        assert float_dtypes(params) == {F64}
        assert float(params.likelihood[0][1, 0]) == 0.1
        _, narrow = make(*(tuple(t.astype(jnp.float32) for t in ts) for ts in (A, B, D)))
        assert float_dtypes(narrow) == {F32}
        from tests.test_closed_loop import small_lg

        system, params = small_lg()
        env, _ = system.as_env(params)
        assert env.inner.null_action().dtype == F64
    with jax.enable_x64(False):
        assert env.inner.null_action().dtype == F32


def test_a_gaussian_draws_in_the_wider_of_its_mean_and_factor_dtypes():
    """A Gaussian with a float32 mean and a float64 factor draws its standard normal in float64
    (it drew in the mean's float32, another stream): the draw is ``loc + chol @ eps`` with
    ``eps`` the float64 normal of the key. A mixture draws the same way."""
    from hybrid_aif.distributions import Gaussian, MixtureGaussian

    with jax.enable_x64(True):
        key = jax.random.PRNGKey(4)
        loc, chol = jnp.asarray([0.5, -0.5], jnp.float32), 0.3 * jnp.eye(2, dtype=jnp.float64)
        q = Gaussian(loc=loc, chol=chol)
        assert (q.loc.dtype, q.chol.dtype) == (F32, F64)
        eps = jax.random.normal(key, (2,), jnp.float64)
        np.testing.assert_allclose(np.asarray(q.sample(key)), np.asarray(loc + chol @ eps))
        assert Gaussian(loc=loc, chol=chol.astype(jnp.float32)).sample(key).dtype == F32
        mix = MixtureGaussian(
            log_weights=jnp.zeros((1,), jnp.float32), means=loc[None], chols=chol[None]
        )
        _, key_x = jax.random.split(key)
        eps = jax.random.normal(key_x, (2,), jnp.float64)
        np.testing.assert_allclose(np.asarray(mix.sample(key)), np.asarray(loc + chol @ eps))


class _Pushed(eqx.Module):
    """``x + u`` with a scale of ones: a nonlinear map with a per-coordinate scale."""

    def __call__(self, x, controls):
        (u,) = controls
        return x + u, jnp.ones_like(x + u)


def test_a_nonlinear_process_steps_on_an_unrounded_input_and_a_float64_draw():
    """A nonlinear transition with a per-coordinate scale, a float32 innovation mean and a
    float64 factor, read by a float64 continuous input: the process steps on the input as given
    (it was cast to float32) and draws its noise in float64, the wider of the innovations'
    dtypes (it drew in the mean's float32): the next state is ``x + u + loc + chol ε``."""
    from hybrid_aif import ConditionalGaussian, Gaussian, NonlinearGaussianTransition
    from hybrid_aif.controls import Box
    from hybrid_aif.system import System, SystemParams

    with jax.enable_x64(True):
        loc = jnp.asarray([[0.01, -0.02]], jnp.float32)
        chol = jnp.asarray([[[0.05, 0.0], [0.02, 0.08]]], jnp.float64)
        noise = ConditionalGaussian(loc=loc, chol=chol, parent_ndims=1)
        decl = NonlinearGaussianTransition(_Pushed(), noise, "first_order", "per_coordinate")
        system = System(
            B_deps=((1,), ()),
            controls=(1,),
            action_names=("u",),
            state_names=("x",),
            control_names=("u_u",),
            policy_len=1,
        )
        params = SystemParams(
            transition={"x": decl},
            initial_state_prior={"x": Gaussian(loc=jnp.zeros(2), chol=jnp.eye(2))},
            control={"u_u": Box(-3.0 * jnp.ones(1), 3.0 * jnp.ones(1))},
        )
        env, env_params = system.as_env(params)
        u = jnp.asarray([0.1], jnp.float64)
        assert u.dtype == F64 and loc.dtype == F32
        state = {"x": jnp.asarray([0.3, -0.4], jnp.float64)}
        key = jax.random.PRNGKey(3)
        _, nxt = env.step(key, state, {"u": u}, env_params)
        #  the process's key for factor 0: split into observation and state keys, one per factor
        k_state = jax.random.split(key)[1]
        eps = jax.random.normal(jax.random.split(k_state, 1)[0], (2,), jnp.float64)
        want = state["x"] + u + loc[0] + chol[0] @ eps
        assert nxt["x"].dtype == F64
        np.testing.assert_allclose(np.asarray(nxt["x"]), np.asarray(want), rtol=1e-14, atol=1e-15)


# ------------------------------------------------------------------------------ cells


def test_a_float32_point_falls_in_the_cell_of_float32_edges():
    """``cell_index`` of a float32 point under 64-bit mode compares it with edges in float32:
    a point equal to a float32 edge is not beyond it. Float64 edges would put the points whose
    float32 value lies above the decimal edge (0.1 among them) in the next cell."""
    from hybrid_aif.cells import Grid, cell_index

    with jax.enable_x64(True):
        grid = Grid(shape=(10,), low=0.0, high=1.0)
        edges = np.linspace(0.0, 1.0, 11)[1:-1]
        points = jnp.asarray(edges.astype(np.float32)[:, None])
        assert points.dtype == F32
        above = np.asarray(points[:, 0], np.float64) > edges
        assert above.any() and not above.all()  # some float32 edges sit above the decimal ones
        got = np.asarray(cell_index(points, grid))
        np.testing.assert_array_equal(got, np.arange(9))  # each point on its own lower edge
        wide = np.asarray(cell_index(jnp.asarray(edges[:, None] + 1e-12), grid))
        np.testing.assert_array_equal(wide, np.arange(1, 10))  # float64 points beyond each edge


# -------------------------------------------------------------------------- controls


def test_soft_clip_computes_in_the_wider_of_the_action_and_its_bounds():
    """A float32 action against float64 bounds is brought into the box in float64: an action
    beyond the box lands on the float64 bound itself, not its float32 rounding."""
    from hybrid_aif.controls import soft_clip

    with jax.enable_x64(True):
        low, high = jnp.asarray([-0.3], jnp.float64), jnp.asarray([0.7], jnp.float64)
        assert float(high[0]) != float(np.float32(0.7))
        action = jnp.asarray([2.0], jnp.float32)
        got = soft_clip(action, (low, high))
        assert got.dtype == F64
        assert float(got[0]) == 0.7
        inside = soft_clip(jnp.asarray([0.25], jnp.float32), (low, high))
        assert inside.dtype == F64
        np.testing.assert_allclose(float(inside[0]), float(np.float32(0.25)), rtol=1e-15)


def test_the_risk_of_a_clamped_continuous_node_reads_its_value_unrounded():
    """``-log C(u)`` at a clamp ``u``: a float64 value against a float64 target is scored at its
    float64 digits (``-log N(0.1; 0, 0.01²)`` to 1e-12), and a float64 value against a float32
    target gives a float64 risk."""
    from hybrid_aif.controls import continuous_node_risk
    from hybrid_aif.distributions import Gaussian

    with jax.enable_x64(True):
        u = jnp.asarray([0.1], jnp.float64)
        assert u.dtype == F64 and float(u[0]) != float(np.float32(0.1))
        target = Gaussian(loc=jnp.zeros(1, jnp.float64), chol=0.01 * jnp.eye(1, dtype=jnp.float64))
        assert float_dtypes(target) == {F64}
        risk = continuous_node_risk(u, target)
        assert risk.lower.dtype == F64
        want = 0.5 * (0.1 / 0.01) ** 2 + np.log(0.01) + 0.5 * np.log(2.0 * np.pi)
        np.testing.assert_allclose(float(risk.lower), want, rtol=1e-12)
        narrow = cast(target, jnp.float32)
        assert float_dtypes(narrow) == {F32}
        assert continuous_node_risk(u, narrow).lower.dtype == F64
        assert continuous_node_risk(u.astype(jnp.float32), narrow).lower.dtype == F32


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_the_infinite_risk_of_a_gaussian_belief_against_a_truncated_target_is_in_its_dtype(dtype):
    """A Gaussian belief over a continuous node against a truncated target has infinite risk,
    in the dtype of the belief and the target."""
    from hybrid_aif.controls import continuous_node_risk
    from hybrid_aif.distributions import Gaussian, TruncatedGaussian

    with jax.enable_x64(True):
        belief = Gaussian(loc=jnp.asarray([0.2], dtype), chol=0.3 * jnp.eye(1, dtype=dtype))
        target = TruncatedGaussian(
            loc=jnp.zeros(1, dtype),
            scale=jnp.ones(1, dtype),
            low=-jnp.ones(1, dtype),
            high=jnp.ones(1, dtype),
        )
        assert float_dtypes((belief, target)) == {np.dtype(dtype)}
        risk = continuous_node_risk(belief, target)
        assert float_dtypes(risk) == {np.dtype(dtype)}
        assert float(risk.lower) == np.inf


def test_the_planners_continuous_bounds_keep_the_boxs_float64_values():
    """The flat bounds a planner reads for a float64 box are the box's float64 values, both
    ends, neither rounded to float32."""
    from hybrid_aif.controls import Box
    from tests.test_closed_loop import small_lg

    with jax.enable_x64(True):
        system, params = small_lg()
        params = dataclasses.replace(
            cast(params, jnp.float64),
            control={"u_x": Box(jnp.asarray([-0.3], jnp.float64), jnp.asarray([0.7], jnp.float64))},
        )
        assert float_dtypes(params) == {F64}
        ctx = system.planner_context(params, system.initial_priors(params))
        low, high = ctx.continuous_controls.bounds
        assert low.dtype == F64 and high.dtype == F64
        assert (float(low[0]), float(high[0])) == (-0.3, 0.7)


# --------------------------------------------------------------- the truncated Gaussian


def test_a_float64_truncated_gaussian_keeps_a_variance_below_float32s_smallest_normal():
    """The variance's lower guard is the smallest normal of the variance's own dtype: a float64
    truncated Gaussian of scale 1e-25 has variance 1e-50 (a float32 guard would raise it to
    1.2e-38)."""
    from hybrid_aif.distributions import TruncatedGaussian

    with jax.enable_x64(True):
        q = TruncatedGaussian(
            loc=jnp.zeros(1, jnp.float64),
            scale=jnp.full(1, 1e-25, jnp.float64),
            low=-jnp.ones(1, jnp.float64),
            high=jnp.ones(1, jnp.float64),
        )
        assert float_dtypes(q) == {F64}
        var = q.variance()
        assert var.dtype == F64
        np.testing.assert_allclose(float(var[0]), 1e-50, rtol=1e-12)


def test_a_truncated_gaussian_scores_a_float64_point_unrounded():
    """``log_prob`` reads its point as given: a float64 point against a float64 distribution is
    scored at its float64 digits (scipy's truncated normal to 1e-12), and against a float32
    distribution gives a float64 log density."""
    from scipy import stats

    from hybrid_aif.distributions import TruncatedGaussian

    with jax.enable_x64(True):
        loc, scale, low, high = 0.2, 0.01, -1.0, 1.0
        q = TruncatedGaussian(*(jnp.full(1, v, jnp.float64) for v in (loc, scale, low, high)))
        assert float_dtypes(q) == {F64}
        x = jnp.asarray([0.21], jnp.float64)
        assert float(x[0]) != float(np.float32(0.21))
        got = q.log_prob(x)
        assert got.dtype == F64
        want = stats.truncnorm.logpdf(0.21, (low - loc) / scale, (high - loc) / scale, loc, scale)
        np.testing.assert_allclose(float(got), want, rtol=1e-12)
        narrow = cast(q, jnp.float32)
        assert float_dtypes(narrow) == {F32}
        assert narrow.log_prob(x).dtype == F64
        assert narrow.log_prob(x.astype(jnp.float32)).dtype == F32


def test_a_float64_truncated_gaussian_draws_its_uniforms_in_float64():
    """A float64 truncated Gaussian whose box is far in the tails draws ``loc + scale Φ⁻¹(u)``
    with ``u`` the key's float64 uniform (a float32 uniform is another stream)."""
    from jax.scipy.special import ndtri

    from hybrid_aif.distributions import TruncatedGaussian

    with jax.enable_x64(True):
        loc, scale = 0.3, 0.5
        q = TruncatedGaussian(*(jnp.full(1, v, jnp.float64) for v in (loc, scale, -20.0, 20.0)))
        assert float_dtypes(q) == {F64}
        key = jax.random.PRNGKey(7)
        got = q.sample(key, (64,))
        assert got.dtype == F64
        u = jnp.clip(jax.random.uniform(key, (64, 1), jnp.float64), 1e-7, 1.0 - 1e-7)
        want = loc + scale * ndtri(u)
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_a_nonlinear_process_draws_its_noise_in_its_innovations_dtype(dtype):
    """The process of a nonlinear transition with a per-coordinate scale, all in one dtype,
    steps in that dtype under 64-bit mode: a float32 one draws its noise in float32 (a draw in
    the default float dtype would make its next state float64), and the next state is
    ``x + u + loc + chol ε`` with ``ε`` the key's normal in that dtype."""
    from hybrid_aif import ConditionalGaussian, Gaussian, NonlinearGaussianTransition
    from hybrid_aif.controls import Box
    from hybrid_aif.system import System, SystemParams

    with jax.enable_x64(True):
        loc = jnp.asarray([[0.01, -0.02]], dtype)
        chol = jnp.asarray([[[0.05, 0.0], [0.02, 0.08]]], dtype)
        noise = ConditionalGaussian(loc=loc, chol=chol, parent_ndims=1)
        decl = NonlinearGaussianTransition(_Pushed(), noise, "first_order", "per_coordinate")
        system = System(
            B_deps=((1,), ()),
            controls=(1,),
            action_names=("u",),
            state_names=("x",),
            control_names=("u_u",),
            policy_len=1,
        )
        params = SystemParams(
            transition={"x": decl},
            initial_state_prior={
                "x": Gaussian(loc=jnp.zeros(2, dtype), chol=jnp.eye(2, dtype=dtype))
            },
            control={"u_u": Box(-3.0 * jnp.ones(1, dtype), 3.0 * jnp.ones(1, dtype))},
        )
        assert float_dtypes(params) == {np.dtype(dtype)}
        env, env_params = system.as_env(params)
        u = jnp.asarray([0.1], dtype)
        state = {"x": jnp.asarray([0.3, -0.4], dtype)}
        key = jax.random.PRNGKey(3)
        _, nxt = env.step(key, state, {"u": u}, env_params)
        k_state = jax.random.split(key)[1]
        eps = jax.random.normal(jax.random.split(k_state, 1)[0], (2,), dtype)
        want = state["x"] + u + loc[0] + chol[0] @ eps
        assert nxt["x"].dtype == dtype
        tol = 1e-6 if dtype == jnp.float32 else 1e-14
        np.testing.assert_allclose(np.asarray(nxt["x"]), np.asarray(want), rtol=tol, atol=tol)


def test_the_null_action_of_a_mixed_control_space_is_in_the_default_float_dtype():
    """A process over a discrete and a continuous control node: its null action is level 0 and
    a zero vector in the default float dtype (float64 under 64-bit mode, float32 without)."""
    from tests.test_mixed_planner import toy

    for x64, want in ((True, F64), (False, F32)):
        with jax.enable_x64(x64):
            system, params = toy()
            env, _ = system.as_env(params)
            level, value = env.inner.null_action().entries
            assert level.dtype == jnp.int32 and int(level) == 0
            assert value.dtype == want and value.shape == (1,)


# --------------------------------------------------------------------------- planners


def test_the_mixed_planner_descends_within_a_float64_boxs_own_bounds():
    """The mixed planner over a float64 model with a box of ``[-0.3, 0.7]``: every discrete
    policy's plan is the gradient planner's on that policy's continuous subproblem (to 1e-13;
    the bounds rounded to float32 move the rate and the start), in float64, and a plan held at
    the upper bound is 0.7 itself."""
    from hybrid_aif.controls import Box
    from hybrid_aif.planning import gradient_plan, mixed_plan
    from tests.test_mixed_planner import STEPS, fix_levels, planned, toy

    with jax.enable_x64(True):
        system, params = toy(1)
        box = Box(jnp.asarray([-0.3], jnp.float64), jnp.asarray([0.7], jnp.float64))
        params = dataclasses.replace(cast(params, jnp.float64), control={"u_c": box})
        assert float_dtypes(params) == {F64}
        ctx = planned(system, params)
        result = mixed_plan(ctx, steps=STEPS)
        assert result.plan.dtype == F64
        plans = np.asarray(result.plan)
        assert np.any(plans == 0.7) and np.any((plans > -0.3) & (plans < 0.7))
        for p in range(result.plan.shape[0]):
            by_hand = gradient_plan(fix_levels(ctx, result.discrete[p]), steps=STEPS)
            np.testing.assert_allclose(plans[p], np.asarray(by_hand.plan), rtol=1e-13, atol=0)


def test_a_float32_box_over_a_float64_model_is_planned_by_gradient_descent():
    """The gradient planner over a float64 model whose box is float32: the plan stays in the
    box's float32 and the scores in the model's float64 (a best score kept in the plan's
    dtype would be a scan carry the float64 score cannot fit)."""
    from tests.test_closed_loop import small_lg

    with jax.enable_x64(True):
        system, params = small_lg()
        params = cast(cast(params, jnp.float64), jnp.float32, where=lambda p: p.control)
        assert float_dtypes(params.control) == {F32}
        assert float_dtypes((params.transition, params.likelihood)) == {F64}
        q_pi, G, plans = system.infer_policies(
            params, system.initial_priors(params), return_policies=True
        )
        assert plans.dtype == F32
        assert float_dtypes(G) == {F64}
        assert q_pi.logits.dtype == F32  # a point mass over the descent's starts, as the plan


# ----------------------------------------------------------------------- descriptions


def test_a_leafs_habit_and_durations_build_in_the_default_float_dtype():
    """A merged pair of leaves, one with a habit and durations, described inline and built under
    64-bit mode: the leaf's habit and durations are float64 with their values, not float32."""
    from hybrid_aif.composition import merge
    from hybrid_aif.description import build, describe
    from tests.test_unread_control_node import model

    with jax.enable_x64(True):
        a, ap = model("durations", durations=[0.1, 0.2])
        b, bp = model("durations", durations=[3.0, 4.0])
        habit = jnp.log(jnp.asarray([0.1, 0.2, 0.3, 0.4], jnp.float64))
        ap = dataclasses.replace(ap, policy_prior=habit)
        _, cp = merge({"a": a, "b": b}, {"a": ap, "b": bp})
        assert cp.parts["a"].durations["a.u_opt"].dtype == F64
        document, arrays = describe(*merge({"a": a, "b": b}, {"a": ap, "b": bp}), arrays="inline")
        _, built = build(document, arrays=arrays)
        part = built.parts["a"]
        assert part.policy_prior.dtype == F64 and part.durations["a.u_opt"].dtype == F64
        np.testing.assert_allclose(np.asarray(part.policy_prior), np.asarray(habit), rtol=1e-15)
        assert [float(v) for v in part.durations["a.u_opt"]] == [0.1, 0.2]


def test_a_continuous_nodes_box_builds_in_the_default_float_dtype():
    """A box of ``[-0.3, 0.7]`` described and built under 64-bit mode is float64 at its values
    at both bounds, neither made float32."""
    from hybrid_aif.controls import Box
    from hybrid_aif.description import build, describe
    from tests.test_closed_loop import small_lg

    with jax.enable_x64(True):
        system, params = small_lg()
        box = Box(jnp.asarray([-0.3], jnp.float64), jnp.asarray([0.7], jnp.float64))
        params = dataclasses.replace(cast(params, jnp.float64), control={"u_x": box})
        assert float_dtypes(params) == {F64}
        _, built = build(describe(system, params, arrays="inline").document)
        got = built.control["u_x"]
        assert got.low.dtype == F64 and got.high.dtype == F64
        assert (float(got.low[0]), float(got.high[0])) == (-0.3, 0.7)


# ------------------------------------------------- continuous control nodes in a transition


def _one_input(dtype):
    """``x' = F x + G u + ε``, one continuous node ``u`` (``test_continuous_nodes``), a target
    on ``x``, all in ``dtype``."""
    from tests.test_continuous_nodes import _gauss, _native_continuous

    system, params = _native_continuous()
    params = dataclasses.replace(params, state_targets={"x": _gauss((1.0, 0.0), 1.0)})
    return system, cast(params, dtype)


def _two_inputs(dtype):
    """``x' = F x + G [u_a; u_b] + ε``: one transition reads two continuous nodes, a target on
    ``x``, all in ``dtype``, one step."""
    from hybrid_aif.system import System, SystemParams
    from tests.test_continuous_nodes import _box, _gauss, _lg

    system = System(
        B_deps=((1, 2), (), ()),
        controls=(1, 2),
        action_names=("a", "b"),
        state_names=("x",),
        control_names=("u_a", "u_b"),
        policy_len=1,
        planner="gradient",
        epistemic=False,
    )
    params = SystemParams(
        transition={"x": _lg(jnp.zeros((1, 2)), G=jnp.asarray([[0.1, 0.3], [0.5, -0.2]]))},
        initial_state_prior={"x": _gauss()},
        control={"u_a": _box(), "u_b": _box()},
        state_targets={"x": _gauss((1.0, 0.0), 1.0)},
    )
    return system, cast(params, dtype)


def _pushed_risk(params, prior, inputs, covs, steps):
    """``Σ_t KL(q_t ‖ C)`` of a linear-Gaussian factor pushed by hand in float64 NumPy: each step
    ``m' = F m + G u``, ``S' = F S Fᵀ + Q Qᵀ + G S_u Gᵀ``."""
    from hybrid_aif import Gaussian

    decl = params.transition["x"]
    F_, G_ = np.asarray(decl.F, np.float64), np.asarray(decl.G, np.float64)
    Q_ = np.asarray(decl.innovations.chol[0], np.float64)
    m, S = np.asarray(prior.loc, np.float64), np.asarray(prior.cov, np.float64)
    target = params.state_targets["x"]
    want = 0.0
    for _ in range(steps):
        m = F_ @ m + G_ @ inputs + np.asarray(decl.innovations.loc[0], np.float64)
        S = F_ @ S @ F_.T + Q_ @ Q_.T + G_ @ covs @ G_.T
        q = Gaussian.from_cov(jnp.asarray(m), jnp.asarray(S))
        want += float(q.kl_to(target).lower)
    return want


def test_a_clamped_float64_input_pushes_the_state_unrounded():
    """``update_empirical_prior`` at a float64 clamp ``u = 0.1``: over a float64 model the mean
    is ``F m + G u`` to 1e-14 (the clamp rounded to float32 misses it); over a float32 model the
    mean
    takes the clamp's float64 (a float32 clamp keeps it float32)."""
    with jax.enable_x64(True):
        u = jnp.asarray([0.1], jnp.float64)
        assert float(u[0]) != float(np.float32(0.1))
        system, params = _one_input(jnp.float64)
        assert float_dtypes(params) == {F64}
        q = system.initial_priors(params)["x"]
        pushed = system.update_empirical_prior(params, {"x": q}, {"u": u})["x"]
        decl = params.transition["x"]
        want = np.asarray(decl.F) @ np.asarray(q.loc) + np.asarray(decl.G) @ np.asarray(u)
        np.testing.assert_allclose(np.asarray(pushed.loc), want, rtol=1e-14, atol=0)
        system, narrow = _one_input(jnp.float32)
        assert float_dtypes(narrow) == {F32}
        q = system.initial_priors(narrow)
        assert system.update_empirical_prior(narrow, q, {"u": u})["x"].loc.dtype == F64
        kept = system.update_empirical_prior(narrow, q, {"u": u.astype(jnp.float32)})["x"]
        assert float_dtypes(kept) == {F32}


def test_a_float64_belief_over_an_input_is_scored_unrounded():
    """``score_policies`` with a Gaussian belief ``N(0.1, 0.3²)`` over the input at each of two
    steps: over a float64 model the risk is the by-hand float64 push (to 1e-12; its mean or its
    factor rounded to float32 moves it near 1e-8), and over a float32 model every term the push
    reaches is float64."""
    from hybrid_aif import Gaussian
    from hybrid_aif.objectives import total

    with jax.enable_x64(True):
        belief = Gaussian(
            loc=jnp.full((2, 1), 0.1, jnp.float64), chol=jnp.full((2, 1, 1), 0.3, jnp.float64)
        )
        assert float_dtypes(belief) == {F64}
        system, params = _one_input(jnp.float64)
        assert float_dtypes(params) == {F64}
        q = system.initial_priors(params)["x"]
        tree = system.score_policies(params, {"x": q}, {"u_u": belief})
        want = _pushed_risk(params, q, np.asarray([0.1]), np.asarray([[0.09]]), steps=2)
        np.testing.assert_allclose(float(total(tree).lower), want, rtol=1e-12)
        system, narrow = _one_input(jnp.float32)
        assert float_dtypes(narrow) == {F32}
        tree = system.score_policies(narrow, system.initial_priors(narrow), {"u_u": belief})
        assert float_dtypes(tree) == {F64}


def test_a_belief_and_a_clamp_on_two_inputs_of_one_transition_keep_their_dtypes():
    """One transition reads a Gaussian belief over ``u_a`` and a clamp ``u_b = 0.2``: over a
    float64 model the risk is the by-hand float64 push (to 1e-12; the clamp rounded to float32
    misses it); over a float32 model with a float32 belief and a float64 clamp the scores are
    float64, and the pushed state's factor takes the clamp's float64 (the clamp's zero block in
    the inputs' factor is made in the clamp's dtype)."""
    from hybrid_aif import Gaussian
    from hybrid_aif.controls import ControlStep
    from hybrid_aif.objectives import hybrid as _hybrid
    from hybrid_aif.objectives import total

    with jax.enable_x64(True):
        clamp = jnp.full((1, 1), 0.2, jnp.float64)
        assert float(clamp[0, 0]) != float(np.float32(0.2))
        system, params = _two_inputs(jnp.float64)
        assert float_dtypes(params) == {F64}
        belief = Gaussian(
            loc=jnp.full((1, 1), 0.1, jnp.float64), chol=jnp.full((1, 1, 1), 0.3, jnp.float64)
        )
        assert float_dtypes(belief) == {F64} and not belief.loc.weak_type
        q = system.initial_priors(params)["x"]
        tree = system.score_policies(params, {"x": q}, {"u_a": belief, "u_b": clamp})
        covs = np.diag([0.09, 0.0])
        want = _pushed_risk(params, q, np.asarray([0.1, 0.2]), covs, steps=1)
        np.testing.assert_allclose(float(total(tree).lower), want, rtol=1e-12)
        system, narrow = _two_inputs(jnp.float32)
        assert float_dtypes(narrow) == {F32}
        belief = cast(belief, jnp.float32)
        q = system.initial_priors(narrow)["x"]
        tree = system.score_policies(narrow, {"x": q}, {"u_a": belief, "u_b": clamp})
        assert float_dtypes(tree) == {F64}
        pos = system._p(narrow)
        step = _hybrid.make_transition(
            pos.transition,
            system._B,
            system._families(pos),
            None,
            table=system._table,
            space=system._space(pos),
        )
        one = Gaussian(loc=belief.loc[0], chol=belief.chol[0])
        (pushed,) = step((q,), ControlStep(entries=(one, clamp[0])))
        assert pushed.loc.dtype == F64 and pushed.chol.dtype == F64


def test_a_float64_policy_for_a_continuous_node_is_scored_unrounded():
    """``score_policies`` given a float64 plan of 0.1 for a continuous node scores that plan,
    not its float32 rounding: over a float64 model the score differs from the rounded plan's,
    and over a float32 model the total is float64."""
    from hybrid_aif.objectives import total

    with jax.enable_x64(True):
        plan_ = jnp.full((2, 1), 0.1, jnp.float64)
        rounded = plan_.astype(jnp.float32).astype(jnp.float64)
        assert not np.array_equal(np.asarray(plan_), np.asarray(rounded))
        system, params = _one_input(jnp.float64)
        assert float_dtypes(params) == {F64}
        q = system.initial_priors(params)
        exact = float(total(system.score_policies(params, q, {"u_u": plan_})).lower)
        near = float(total(system.score_policies(params, q, {"u_u": rounded})).lower)
        assert exact != near
        want = _pushed_risk(params, q["x"], np.asarray([0.1]), np.zeros((1, 1)), steps=2)
        np.testing.assert_allclose(exact, want, rtol=1e-12)
        system, narrow = _one_input(jnp.float32)
        assert float_dtypes(narrow) == {F32}
        tree = system.score_policies(narrow, system.initial_priors(narrow), {"u_u": plan_})
        assert total(tree).lower.dtype == F64


class _Shifted(eqx.Module):
    """``x + u``: a nonlinear map with no scale, whose first-order push is exact."""

    def __call__(self, x, controls):
        (u,) = controls
        return x + u


def _nonlinear(dtype):
    """``x' = x + u + loc + ε`` as a nonlinear transition (first order), a target on ``x``,
    all in ``dtype``."""
    from hybrid_aif import ConditionalGaussian, Gaussian, NonlinearGaussianTransition
    from hybrid_aif.controls import Box
    from hybrid_aif.system import System, SystemParams

    noise = ConditionalGaussian(
        loc=jnp.asarray([[0.01, -0.02]]),
        chol=jnp.asarray([[[0.05, 0.0], [0.02, 0.08]]]),
        parent_ndims=1,
    )
    system = System(
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("u",),
        state_names=("x",),
        control_names=("u_u",),
        policy_len=2,
        planner="gradient",
        epistemic=False,
    )
    params = SystemParams(
        transition={"x": NonlinearGaussianTransition(_Shifted(), noise, "first_order")},
        initial_state_prior={"x": Gaussian(loc=jnp.asarray([0.3, -0.2]), chol=0.2 * jnp.eye(2))},
        control={"u_u": Box(-3.0 * jnp.ones(2), 3.0 * jnp.ones(2))},
        state_targets={"x": Gaussian(loc=jnp.asarray([1.0, 0.0]), chol=jnp.eye(2))},
    )
    return system, cast(params, dtype)


def test_a_nonlinear_transition_pushes_a_float64_clamp_unrounded():
    """``update_empirical_prior`` of ``x' = x + u + loc`` at a float64 clamp: over a float64 model
    the mean is ``m + u + loc`` to 1e-14 (the clamp rounded to float32 misses it); over a float32
    model the mean takes the clamp's float64."""
    with jax.enable_x64(True):
        u = jnp.asarray([0.1, -0.3], jnp.float64)
        assert np.all(np.asarray(u) != np.asarray(u, np.float32))
        system, params = _nonlinear(jnp.float64)
        assert float_dtypes(params) == {F64}
        q = system.initial_priors(params)["x"]
        pushed = system.update_empirical_prior(params, {"x": q}, {"u": u})["x"]
        loc = params.transition["x"].innovations.loc[0]
        want = np.asarray(q.loc) + np.asarray(u) + np.asarray(loc)
        np.testing.assert_allclose(np.asarray(pushed.loc), want, rtol=1e-14, atol=0)
        system, narrow = _nonlinear(jnp.float32)
        assert float_dtypes(narrow) == {F32}
        q = system.initial_priors(narrow)
        assert system.update_empirical_prior(narrow, q, {"u": u})["x"].loc.dtype == F64
        kept = system.update_empirical_prior(narrow, q, {"u": u.astype(jnp.float32)})["x"]
        assert float_dtypes(kept) == {F32}


def test_a_nonlinear_transition_scores_a_float64_belief_over_its_input_unrounded():
    """``score_policies`` of ``x' = x + u + loc`` with a Gaussian belief over ``u`` at each of two
    steps: over a float64 model the risk is the exact push ``m + μ_u + loc``, ``S + S_u + Q Qᵀ``
    (to 1e-12; the belief's mean or factor rounded to float32 moves it), and over a float32 model
    every term is float64."""
    from hybrid_aif import Gaussian
    from hybrid_aif.objectives import total

    with jax.enable_x64(True):
        mean = np.asarray([0.1, -0.3])
        root = np.asarray([[0.3, 0.0], [0.1, 0.2]])
        belief = Gaussian(
            loc=jnp.asarray(np.broadcast_to(mean, (2, 2)), jnp.float64),
            chol=jnp.asarray(np.broadcast_to(root, (2, 2, 2)), jnp.float64),
        )
        assert float_dtypes(belief) == {F64}
        system, params = _nonlinear(jnp.float64)
        assert float_dtypes(params) == {F64}
        q = system.initial_priors(params)["x"]
        got = float(total(system.score_policies(params, {"x": q}, {"u_u": belief})).lower)
        noise = params.transition["x"].innovations
        Q_ = np.asarray(noise.chol[0])
        m, S, want = np.asarray(q.loc), np.asarray(q.cov), 0.0
        for _ in range(2):
            m, S = m + mean + np.asarray(noise.loc[0]), S + root @ root.T + Q_ @ Q_.T
            pushed = Gaussian.from_cov(jnp.asarray(m), jnp.asarray(S))
            want += float(pushed.kl_to(params.state_targets["x"]).lower)
        np.testing.assert_allclose(got, want, rtol=1e-12)
        system, narrow = _nonlinear(jnp.float32)
        assert float_dtypes(narrow) == {F32}
        tree = system.score_policies(narrow, system.initial_priors(narrow), {"u_u": belief})
        assert float_dtypes(tree) == {F64}


@pytest.mark.parametrize("kind", ["linear", "nonlinear"])
def test_a_float32_mixture_belief_is_pushed_in_float32(kind):
    """A float32 model whose prior over ``x`` is a two-component mixture, pushed by a float32
    clamp through a transition with no discrete parent: the pushed mixture, its log weights
    included, stays float32 under 64-bit mode (the single mode's log weight made in the default
    float dtype would widen the weights)."""
    from hybrid_aif import MixtureGaussian

    with jax.enable_x64(True):
        system, params = (_one_input if kind == "linear" else _nonlinear)(jnp.float32)
        assert float_dtypes(params) == {F32}
        dim = params.initial_state_prior["x"].loc.shape[-1]
        q = MixtureGaussian(
            log_weights=jnp.log(jnp.asarray([0.3, 0.7], jnp.float32)),
            means=jnp.asarray([[0.3, -0.2], [-0.4, 0.5]], jnp.float32),
            chols=jnp.broadcast_to(0.2 * jnp.eye(dim, dtype=jnp.float32), (2, dim, dim)),
        )
        assert float_dtypes(q) == {F32}
        params = dataclasses.replace(params, initial_state_prior={"x": q})
        system = dataclasses.replace(system, mixture_k=2)
        u = jnp.full((params.control["u_u"].low.shape[0],), 0.1, jnp.float32)
        pushed = system.update_empirical_prior(params, {"x": q}, {"u": u})["x"]
        assert isinstance(pushed, MixtureGaussian)
        assert float_dtypes(pushed) == {F32}


# ------------------------------------------------------------------- the analytic faces


@pytest.mark.parametrize("kernel", ["single", "per_level"])
def test_a_linear_gaussian_emission_is_realised_in_its_kernels_dtype(kernel):
    """``emit`` of a float32 linear-Gaussian emission kernel by its statistics and its mode,
    under 64-bit mode: a kernel over a Gaussian state (one component) and a kernel with one
    head per level of a categorical parent beside it (a mixture over the levels) give float32
    means and covariances (mixture weights started as default-dtype ones would widen them)."""
    from hybrid_aif import Emission, Gaussian, LinearGaussianObservation
    from tests.test_hybrid_lg_system import make_params as hill_params
    from tests.test_hybrid_lg_system import make_system as hill_system

    with jax.enable_x64(True):
        key = jax.random.PRNGKey(0)
        for realise in ("stats", "mode"):
            if kernel == "single":
                system = dataclasses.replace(
                    hill_system(), emissions={"gps": Emission(("state",), realise)}
                )
                decl = LinearGaussianObservation(
                    H=jnp.asarray([[1.0, 0.5]]), c=jnp.asarray([0.2]), noise=jnp.asarray([[0.3]])
                )
                params = cast(dataclasses.replace(hill_params(), emission={"gps": decl}), F32)
                values = {"state": Gaussian(loc=jnp.asarray([0.4, -0.2]), chol=0.3 * jnp.eye(2))}
            else:
                system = hill_system(
                    A_deps=((0,),),
                    B_deps=((), (1,)),
                    controlled=(True, False),
                    state_names=("state", "mode"),
                    emissions={"gps": Emission(("state", "mode"), realise)},
                )
                decl = LinearGaussianObservation(
                    H=jnp.asarray([[[1.0, 0.5]], [[0.2, -1.0]]]),
                    c=jnp.asarray([[0.2], [-0.4]]),
                    noise=jnp.asarray([[[0.3]], [[0.6]]]),
                )
                mode = ConditionalCategorical(log_table=jnp.log(jnp.eye(2)), parent_ndims=1)
                params = hill_params()
                params = dataclasses.replace(
                    params,
                    transition={**params.transition, "mode": mode},
                    initial_state_prior={
                        **params.initial_state_prior,
                        "mode": Categorical(logits=jnp.zeros(2)),
                    },
                    emission={"gps": decl},
                )
                params = cast(params, F32)
                values = {
                    "state": Gaussian(loc=jnp.asarray([0.4, -0.2]), chol=0.3 * jnp.eye(2)),
                    "mode": Categorical.from_probs(jnp.asarray([0.3, 0.7])),
                }
            values = cast(values, F32)
            assert float_dtypes((params, values)) == {F32}
            out = system.emit(key, params, values)
            assert float_dtypes(out) == {F32}, realise


def test_an_observed_action_node_takes_a_uniform_prior_in_the_models_dtype():
    """``infer_states`` on a float32 T-maze whose cue is read through its action node, given no
    belief over the node: the node's uniform prior, and so its posterior and every belief, are
    float32 under 64-bit mode (a prior made in the default float dtype would widen them), with
    priors given for the states alone and with none."""
    from tests.test_action_node_modalities import maze_with_cue_by_node

    with jax.enable_x64(True):
        system, params = maze_with_cue_by_node()
        params = cast(params, F32)
        assert float_dtypes(params) == {F32}
        observations = {m: jnp.asarray(1, jnp.int32) for m in system.names.modalities}
        states = system.initial_priors(params)
        for priors in (None, {k: states[k] for k in system.state_names}):
            beliefs, _ = system.infer_states(params, observations, priors)
            assert "act" in beliefs
            assert float_dtypes(beliefs) == {F32}


@pytest.mark.parametrize("branch_on", [(), ("cue",)])
def test_a_float32_tree_plans_in_float32(branch_on):
    """The sophisticated planner over the float32 T-maze under 64-bit mode, with no branching
    modality and branching on the cue: every float it returns is float32, each child's weight
    included (ones when nothing branches, the branch's outcome probabilities when the cue does;
    either made in the default float dtype would widen the scores)."""
    from tests.test_system import tmaze_system

    with jax.enable_x64(True):
        system, params = tmaze_system()[:2]
        system = dataclasses.replace(system, planner="sophisticated", branch_on=branch_on)
        params = cast(params, F32)
        assert float_dtypes(params) == {F32} and system.policy_len >= 2
        ctx = system.planner_context(params, system.initial_priors(params))
        assert (ctx.branch_fn is None) == (not branch_on)
        out = system.infer_policies(params, system.initial_priors(params), return_policies=True)
        assert float_dtypes(out) == {F32}


def test_a_float32_tree_with_no_step_terms_plans_in_float32():
    """The sophisticated planner over a float32 model with no target and no epistemic term:
    every step's score is zero (the posterior is uniform) and is float32 under 64-bit mode
    (the empty step's zeros made in the default float dtype would widen it)."""
    from tests.test_action_node_modalities import scalar_model

    with jax.enable_x64(True):
        system, params = scalar_model()
        system = dataclasses.replace(system, planner="sophisticated", epistemic=False)
        params = cast(params, F32)
        assert float_dtypes(params) == {F32} and system.policy_len >= 2
        assert not params.state_targets and not params.obs_targets
        q_pi, *rest = system.infer_policies(
            params, system.initial_priors(params), return_policies=True
        )
        probs = np.asarray(q_pi.probs)
        np.testing.assert_array_equal(probs, np.full_like(probs, probs[0]))
        assert float_dtypes((q_pi, rest)) == {F32}


# ---------------------------------------------------------------------- structure learning


def _growable(dtype):
    """``test_structure_learning``'s growable factor ``z`` (six levels, the first live, a
    Gaussian emission ``y``) with three levels made live, in ``dtype``."""
    from tests.test_structure_learning import _growable_system

    system, params = _growable_system()
    weights = jnp.zeros((6, 1)).at[0, 0].set(3.0).at[1, 0].set(2.0).at[2, 0].set(1.0)
    params = dataclasses.replace(
        params,
        transition={"z": dataclasses.replace(params.transition["z"], concentration=weights)},
    )
    return system, cast(params, dtype)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_the_structure_scores_are_in_the_weights_dtype(dtype):
    """The per-level scores of a datum, the pair scores and the factor's evidence after one
    learning step, of a growable model in one dtype, are in that dtype under 64-bit mode (each
    sum started as default-dtype zeros would widen a float32 model's to float64)."""
    from hybrid_aif.learning.structure import factor_evidence, level_scores, pair_scores

    with jax.enable_x64(True):
        system, params = _growable(dtype)
        assert float_dtypes(params) == {np.dtype(dtype)}
        y = jnp.asarray([0.5, -0.3], dtype)
        beliefs, _ = system.infer_states(params, {"y": y}, None)
        assert float_dtypes(beliefs) == {np.dtype(dtype)}
        inner = system._p(params)
        scores, observed = level_scores(system, inner, 0, (y,), (beliefs["z"],))
        assert scores.dtype == dtype and bool(observed)
        assert pair_scores(system, inner, 0).dtype == dtype
        learned = system._p(system.learn(params, {"y": y}, beliefs, prev_beliefs=beliefs))
        assert float_dtypes(learned) == {np.dtype(dtype)}
        evidence = factor_evidence(system, learned, inner, 0)
        assert evidence.dtype == dtype and float(evidence) != 0.0


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_the_batch_form_of_growth_learns_in_the_models_dtype(dtype):
    """``System.grow`` over a window of twenty data on a growable model in one dtype absorbs the
    window (the emission's degrees of freedom rise by twenty) and returns every learned
    parameter in that dtype under 64-bit mode: the window's scores, its responsibilities and
    what they absorb stay in a float32 model's float32."""
    from tests.test_structure_learning import _stream

    with jax.enable_x64(True):
        system, params = _growable(dtype)
        assert float_dtypes(params) == {np.dtype(dtype)}
        ys, labels = _stream(n=40)
        window = jnp.asarray(ys[labels != 1][:20], dtype)
        assert window.shape == (20, 2)
        grown = system.grow(params, {"y": window}, "z", rounds=1)
        assert float_dtypes(grown) == {np.dtype(dtype)}
        before = float(np.sum(np.asarray(params.likelihood["y"].nu, np.float64)))
        after = float(np.sum(np.asarray(grown.likelihood["y"].nu, np.float64)))
        np.testing.assert_allclose(after, before + 20.0, rtol=1e-6)


# ---------------------------------------------------------------- the factorised planner


def _factorised_model(levels):
    """One observed state factor ``s0`` and one controlled factor ``s{f}`` per entry of
    ``levels``, its control node of that many levels (none for ``()``), planned by the
    factorised planner over two steps."""
    from hybrid_aif.system import System, SystemParams

    def cc(table, parent_ndims):
        return ConditionalCategorical(
            log_table=jnp.log(jnp.asarray(table)), parent_ndims=parent_ndims
        )

    k = max(len(levels), 1)
    system = System(
        A_deps=((0,),),
        B_deps=tuple((f,) for f in range(k)),
        controlled=(True,) * len(levels) if levels else (False,),
        policy_len=2,
        planner="factorised",
        modality_names=("o",),
        state_names=tuple(f"s{f}" for f in range(k)),
        gamma=4.0,
    )
    moves = [np.roll(np.eye(3), j, axis=0) * 0.9 + 0.1 / 3 for j in range(3)]
    transition = {
        f"s{f}": cc(np.stack(moves[:n], -1), 2) if levels else cc(moves[0], 1)
        for f, n in enumerate(levels or (1,))
    }
    params = SystemParams(
        likelihood={"o": cc(np.eye(3) * 0.8 + 0.2 / 3, 1)},
        transition=transition,
        initial_state_prior={
            f"s{f}": Categorical.from_probs(jnp.asarray([0.5, 0.3, 0.2])) for f in range(k)
        },
        obs_targets={"o": Categorical(logits=jnp.asarray([0.0, 2.0, -1.0]))},
    )
    return system, params


@pytest.mark.parametrize("levels", [(3, 1), (1, 1), ()], ids=["three_and_one", "ones", "none"])
def test_the_factorised_planner_plans_a_float64_model_in_float64(levels):
    """The factorised planner over a float64 model returns float64 only, its per-node
    marginals, posteriors and movement included: with a node of one level beside one of three
    (the one-level node's marginal and posterior are its starting ones), with every node of one
    level (no sweep moves a marginal) and with no control node (the joint first-step logits are
    the starting zeros). Each of those starts made in float32 would show."""
    with jax.enable_x64(True):
        system, params = _factorised_model(levels)
        params = cast(params, F64)
        assert float_dtypes(params) == {F64}
        assert system._control_levels(params) == levels
        q_pi, tree, policies = system.infer_policies(
            params, system.initial_priors(params), return_policies=True
        )
        assert len(policies.marginals) == len(levels)
        assert float_dtypes((q_pi, tree, policies)) == {F64}
        assert policies.movement.dtype == F64


def test_a_float32_model_with_a_neural_table_absent_infers_in_float32():
    """``infer_states`` on a float32 model whose image is read by a neural likelihood over a
    categorical factor alone (a table of log densities in the sweep), the image left out: the
    absent table's zeros are float32 under 64-bit mode (default-dtype zeros would widen the
    posterior and the free energy)."""
    from hybrid_aif.distributions import MLPConditional
    from hybrid_aif.system import System, SystemParams
    from tests import test_recognition_analytic as rec

    with jax.enable_x64(True):
        system = System(
            A_deps=((0,), (0,)),
            B_deps=((0,),),
            controlled=(False,),
            policy_len=1,
            modality_names=("cue", "img"),
            state_names=("terrain",),
        )
        image = MLPConditional.from_shapes(
            (3,), 16, key=rec.KEY, width=8, depth=1, frozen=True, dtype=F32
        )
        params = SystemParams(
            likelihood={
                "cue": ConditionalCategorical(log_table=rec.CUE, parent_ndims=1),
                "img": image,
            },
            transition={"terrain": ConditionalCategorical(log_table=rec.STICKY, parent_ndims=1)},
            initial_state_prior={"terrain": rec.TERRAIN},
        )
        params = cast(params, F32)
        assert float_dtypes(params) == {F32}
        priors = cast({"terrain": rec.TERRAIN}, F32)
        beliefs, detail = system.infer_states(
            params,
            {"cue": jnp.asarray(1), "img": None},
            priors=priors,
            present={"img": False},
            detail=True,
        )
        assert float_dtypes(beliefs) == {F32}
        assert detail.free_energy.dtype == F32


def test_the_first_ticks_priors_take_the_widest_dtype_of_the_tables_and_the_priors():
    """A float32 model whose transitions are float64: ``initial_priors`` widens the float32
    initial prior to float64, the dtype the first tick's posterior takes, its values unchanged
    (not left in its own float32)."""
    from tests.test_closed_loop import small_lg

    with jax.enable_x64(True):
        system, params = small_lg()
        params = cast(cast(params, F32), F64, where=lambda p: p.transition)
        assert float_dtypes(params.initial_state_prior) == {F32}
        assert float_dtypes(params.transition) == {F64}
        priors = system.initial_priors(params)
        assert float_dtypes(priors) == {F64}
        given = params.initial_state_prior["x"]
        np.testing.assert_array_equal(np.asarray(priors["x"].loc), np.asarray(given.loc))
        np.testing.assert_array_equal(np.asarray(priors["x"].chol), np.asarray(given.chol))


# ------------------------------------------------------------ the state run settles from


def _settles_alike(monkeypatch, call):
    """The state ``call`` settles from the reset read off the structure and the state round 0's
    trace gives (``test_settle``): the tree structure, each leaf's shape and dtype, and the
    float dtypes of the first."""
    from tests.test_settle import forms, settled

    static = settled(monkeypatch, "structure", call)
    traced = settled(monkeypatch, "traced", call)
    assert forms(static) == forms(traced)
    return float_dtypes(static)


@pytest.mark.parametrize("dtype", [F32, F64], ids=["float32", "float64"])
def test_an_unset_probability_vector_is_settled_in_the_models_dtype(monkeypatch, dtype):
    """A rover and a mission in one dtype under 64-bit mode, the mission reading the rover's
    position through its cells by their probabilities: the zeros ``run`` gives that probability
    vector, which the rover's ``plan()`` leaves unset at the reset, read off the structure, are
    in the model's dtype, the dtype round 0 gives them (not float32 for a float64 model, nor
    the default float dtype for a float32 one)."""
    from tests.test_sibling_read import _composite, _run

    with jax.enable_x64(True):
        composite, params = _composite(realise="stats")
        params = cast(params, dtype)
        assert float_dtypes(params) == {dtype}
        assert _settles_alike(monkeypatch, lambda: _run(composite, params, 2)) == {dtype}


# ------------------------------------------------------------------- views and joint habits


def test_a_views_uniform_average_over_a_partner_node_is_stored_unrounded():
    """A float64 composite's view averages the partner's node out of a table with the uniform
    weight 1/5 in float64: the stored table (not only its normalisation) is the mean over the
    node to 1e-13 (a float32 1/5 would scale it by 1 + 1.5e-8)."""
    from tests.test_merge_nodes import same_step

    with jax.enable_x64(True):
        composite, cp = same_step()
        assert float_dtypes(cp.transition) == {F64}
        assert composite._control_levels(cp) == (5,)
        table = np.exp(np.asarray(cp.transition["world.location"].log_probs, np.float64))
        _, uniform = subsystem(composite, cp, "world")
        got = uniform.transition["world.location"]
        assert got.log_table.dtype == F64
        stored = np.exp(np.asarray(got.log_table))[..., 0]
        np.testing.assert_allclose(stored, table.mean(axis=-1), rtol=1e-13)


def test_the_joint_habit_of_float16_leaf_habits_is_at_least_float32():
    """Leaf habits declared as Categoricals over float16 logits sum into a joint habit of at
    least float32 (a caller's float array keeps its dtype, at least float32), the float16 log
    probabilities summed in float32."""
    from hybrid_aif.composition import joint_planner, merge
    from tests.test_joint_planner import OBSERVATION_BINDINGS, _two_nodes

    with jax.enable_x64(True):
        habit_a = Categorical(logits=jnp.linspace(-1.0, 1.0, 5).astype(jnp.float16))
        habit_w = Categorical(logits=jnp.linspace(1.0, -2.0, 5).astype(jnp.float16))
        assert float_dtypes((habit_a, habit_w)) == {np.dtype(np.float16)}
        (agent, ap), (world, wp) = _two_nodes(policy_prior=habit_w)
        ap = dataclasses.replace(ap, policy_prior=habit_a)
        composite, cp = merge(
            {"agent": agent, "world": world},
            {"agent": ap, "world": wp},
            list(OBSERVATION_BINDINGS),
        )
        _, mp = joint_planner(composite, cp)
        assert mp.policy_prior.dtype == F32
        a = np.asarray(habit_a.log_probs, np.float32)
        w = np.asarray(habit_w.log_probs, np.float32)
        np.testing.assert_array_equal(np.asarray(mp.policy_prior), (a[:, None] + w).reshape(-1))
