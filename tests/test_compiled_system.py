"""The compiled System and the retrace watcher: a tick loop through ``System.compiled()`` traces
each face once and agrees with the eager faces, input-type differences cannot retrace it, a real
retrace is named, and an eager loop is nudged toward the compiled object.

Design: docs/design/recompilation.md R6, R7.
"""

import collections
import dataclasses
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Categorical, debug
from hybrid_aif.composition.merge import subsystem
from hybrid_aif.debug import RetraceWarning
from hybrid_aif.system import System
from hybrid_aif.system import compiled as compiled_module
from tests.test_loop_invariance import (
    _learnable_tmaze,
    _payoff_mixed,
    _small_lg_gradient,
    _tmaze,
)

TICKS = 20
#: The default float dtype's name, the dtype ``jnp.ones`` makes: float32, or float64 under JAX's
#: 64-bit mode.
DEFAULT_FLOAT = "float64" if jax.config.jax_enable_x64 else "float32"
FACES = ("infer_states", "infer_policies", "act", "update_empirical_prior", "learn")

MODELS = {
    "tmaze": _tmaze,
    "payoff_mixed": _payoff_mixed,
    "small_lg_gradient": _small_lg_gradient,
}


@pytest.fixture(autouse=True)
def _fresh_compile_caches():
    """Equal Systems share compiled programs, so a trace count starts from empty caches."""
    jax.clear_caches()
    compiled_module._SNAPSHOTS.clear()


@pytest.fixture
def traces(monkeypatch):
    """Count, per face, the calls that happen while tracing (a Tracer among the arguments)."""
    counts = collections.Counter()

    def counting(face):
        original = getattr(System, face)

        def wrapper(self, *args, **kwargs):
            leaves = jax.tree_util.tree_leaves((args, kwargs))
            if any(isinstance(x, jax.core.Tracer) for x in leaves):
                counts[face] += 1
            return original(self, *args, **kwargs)

        return wrapper

    for face in FACES:
        monkeypatch.setattr(System, face, counting(face))
    return counts


def _assert_close(a, b):
    la, ta = jax.tree_util.tree_flatten(a)
    lb, tb = jax.tree_util.tree_flatten(b)
    assert ta == tb
    for x, y in zip(la, lb, strict=True):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=2e-4, rtol=1e-3)


@pytest.mark.parametrize("model", sorted(MODELS))
def test_twenty_compiled_ticks_trace_each_face_once_and_match_the_eager_faces(model, traces):
    (system, params), options = MODELS[model]()
    compiled = system.compiled()
    env, env_params = system.as_env(params)
    key = jax.random.PRNGKey(3)
    key, k0 = jax.random.split(key)
    obs, state = env.reset(k0, env_params)
    priors_c = priors_e = system.initial_priors(params)
    for node, (level, count) in options.get("clamp", {}).items():
        clamped = Categorical.from_probs(jax.nn.one_hot(level, count))
        priors_c = priors_e = {**priors_c, node: clamped}
    for _ in range(TICKS):
        key, ka, ke = jax.random.split(key, 3)
        beliefs_c, _ = compiled.infer_states(params, obs, priors_c)
        beliefs_e, _ = system.infer_states(params, obs, priors_e)
        _assert_close(beliefs_c, beliefs_e)
        action_c, q_c, _ = compiled.act(ka, params, beliefs_c, metrics=False)
        action_e, q_e, _ = system.act(ka, params, beliefs_e, metrics=False)
        _assert_close(action_c, action_e)
        _assert_close(q_c, q_e)
        q_c2, _ = compiled.infer_policies(params, beliefs_c, metrics=False)
        assert q_c2 is not None
        priors_c = compiled.update_empirical_prior(params, beliefs_c, action_c)
        priors_e = system.update_empirical_prior(params, beliefs_e, action_c)
        _assert_close(priors_c, priors_e)
        obs, state = env.step(ke, state, action_c, env_params)
    assert dict(traces) == {
        "infer_states": 1,
        "act": 1,
        "infer_policies": 1,
        "update_empirical_prior": 1,
    }
    assert compiled.trace_counts == {
        "infer_states": 1,
        "act": 1,
        "infer_policies": 1,
        "update_empirical_prior": 1,
    }


def test_the_learning_face_traces_once_and_matches_the_eager_face(traces):
    (system, params), _ = _learnable_tmaze()
    compiled = system.compiled()
    env, env_params = system.as_env(params)
    key = jax.random.PRNGKey(5)
    obs, state = env.reset(key, env_params)
    params_c = params_e = params
    priors, prev_beliefs, prev_action = None, None, None
    for _ in range(TICKS):
        key, ka, ke = jax.random.split(key, 3)
        beliefs, _ = compiled.infer_states(params_c, obs, priors)
        action, _, _ = compiled.act(ka, params_c, beliefs, metrics=False)
        if prev_beliefs is not None:
            params_c = compiled.learn(
                params_c, obs, beliefs, prev_beliefs=prev_beliefs, action=prev_action, lr=0.5
            )
            params_e = system.learn(
                params_e,
                obs,
                beliefs,
                prev_beliefs=prev_beliefs,
                action=prev_action,
                lr=jnp.float32(0.5),
            )
        priors = compiled.update_empirical_prior(params_c, beliefs, action)
        prev_beliefs, prev_action = beliefs, action
        obs, state = env.step(ke, state, action, env_params)
    _assert_close(params_c, params_e)
    assert traces["learn"] == 1
    assert compiled.trace_counts["learn"] == 1


def test_a_python_float_rate_causes_no_second_trace(traces):
    (system, params), _ = _learnable_tmaze()
    compiled = system.compiled()
    env, env_params = system.as_env(params)
    obs, _ = env.reset(jax.random.PRNGKey(0), env_params)
    beliefs, _ = compiled.infer_states(params, obs)
    prev_beliefs = beliefs
    action = {n: jnp.zeros((), jnp.int32) for n in system.names.actions}
    for lr in (1.0, 0.5, 0.25, np.float64(0.125)):
        compiled.learn(params, obs, beliefs, prev_beliefs=prev_beliefs, action=action, lr=lr)
    assert traces["learn"] == 1


def test_a_changed_observation_shape_warns_once_and_names_the_observation():
    (system, params), _ = _tmaze()
    compiled = system.compiled()
    env, env_params = system.as_env(params)
    obs, _ = env.reset(jax.random.PRNGKey(0), env_params)
    size = params.likelihood["cue"].log_table.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RetraceWarning)
        compiled.infer_states(params, obs)
        compiled.infer_states(params, {**obs, "cue": jnp.asarray(1, jnp.uint8)})  # same trace
    one_hot = jax.nn.one_hot(obs["cue"], size)  # an accepted form, but another shape
    with pytest.warns(RetraceWarning, match=r"observations\['cue'\]") as caught:
        compiled.infer_states(params, {**obs, "cue": one_hot})
    assert len([w for w in caught if issubclass(w.category, RetraceWarning)]) == 1
    assert "infer_states" in str(caught[0].message)
    assert compiled.trace_counts["infer_states"] == 2


def test_a_wrong_linear_gaussian_shape_is_refused_eagerly_with_a_sentence():
    (system, params), _ = _small_lg_gradient()
    compiled = system.compiled()
    env, env_params = system.as_env(params)
    obs, _ = env.reset(jax.random.PRNGKey(0), env_params)
    name = next(iter(obs))
    with pytest.raises(ValueError, match=f"observation '{name}'"):
        compiled.infer_states(params, {**obs, name: jnp.zeros((1, *obs[name].shape))})
    assert compiled.trace_counts == {}


def test_the_standalone_watcher_names_a_changed_shape_and_a_changed_static():
    @eqx.filter_jit
    def total(x, scale):
        return x.sum() * scale

    watched = debug.watch(total)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RetraceWarning)
        watched(jnp.ones(3), 2.0)
        watched(jnp.ones(3), 2.0)
    assert watched.traces == 1
    assert jnp.ones(3).dtype == DEFAULT_FLOAT
    changed = rf"x changed from {DEFAULT_FLOAT}\[3\] to {DEFAULT_FLOAT}\[4\]"
    with pytest.warns(RetraceWarning, match=changed):
        watched(jnp.ones(4), 2.0)
    with pytest.warns(RetraceWarning, match=r"scale is a static .* from 2.0 to 3.0"):
        watched(jnp.ones(4), 3.0)
    assert watched.traces == 3


def test_the_watcher_names_a_system_field_and_a_planner():
    (system, params), _ = _tmaze()
    other = dataclasses.replace(system, gamma=system.gamma + 1.0)

    @eqx.filter_jit
    def policies(system, params, beliefs, planner=None):
        return system.infer_policies(params, beliefs, planner, metrics=False)[0]

    watched = debug.watch(policies)
    env, env_params = system.as_env(params)
    beliefs, _ = system.infer_states(params, env.reset(jax.random.PRNGKey(0), env_params)[0])
    watched(system, params, beliefs)
    with pytest.warns(RetraceWarning, match="static field 'gamma' of 'system'"):
        watched(other, params, beliefs)


def test_a_subsystem_view_and_its_composite_compile_and_trace_once(traces):
    from hybrid_aif.composition.merge import merge
    from tests.test_noise_map_mixture import _levelled, _showcase

    system, params = _showcase(_levelled(), planner="mixed", policy_len=1)
    composite, composite_params = merge({"m": system}, {"m": params})
    view, view_params = subsystem(composite, composite_params, "m")
    for model, model_params in ((view, view_params), (composite, composite_params)):
        jax.clear_caches()
        compiled_module._SNAPSHOTS.clear()
        compiled = model.compiled()
        env, env_params = model.as_env(model_params)
        key = jax.random.PRNGKey(1)
        obs, state = env.reset(key, env_params)
        priors = model.initial_priors(model_params)
        sensor = next(n for n in model.names.actions if n.endswith("sensor"))
        priors = {**priors, sensor: Categorical.from_probs(jax.nn.one_hot(0, 2))}
        for _ in range(4):
            key, ka, ke = jax.random.split(key, 3)
            beliefs, _ = compiled.infer_states(model_params, obs, priors)
            action, _, _ = compiled.act(ka, model_params, beliefs, metrics=False)
            priors = compiled.update_empirical_prior(model_params, beliefs, action)
            obs, state = env.step(ke, state, action, env_params)
        assert compiled.trace_counts["act"] == 1
        assert compiled.trace_counts["update_empirical_prior"] == 1
        assert compiled.trace_counts["infer_states"] == 1


def test_the_eager_nudge_fires_once_after_the_eleventh_call():
    (system, params), _ = _tmaze()
    env, env_params = system.as_env(params)
    obs, _ = env.reset(jax.random.PRNGKey(0), env_params)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RetraceWarning)
        for _ in range(debug.EAGER_CALL_LIMIT):
            system.infer_states(params, obs)
    with pytest.warns(RetraceWarning, match=r"System\.compiled\(\)") as caught:
        system.infer_states(params, obs)
    assert len(caught) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", RetraceWarning)
        for _ in range(5):
            system.infer_states(params, obs)


# ------------------------------------------------------------ legality, presence, structure


def _tmaze_obs():
    (system, params), _ = _tmaze()
    env, env_params = system.as_env(params)
    obs, _ = env.reset(jax.random.PRNGKey(0), env_params)
    beliefs, _ = system.infer_states(params, obs)
    return system, params, obs, beliefs


def _refusal(call):
    with pytest.raises(ValueError) as caught:
        call()
    return str(caught.value)


@pytest.mark.parametrize("face", ["act", "infer_policies"])
@pytest.mark.parametrize(
    "mask",
    [
        {"u_location": jnp.zeros((5,), bool)},
        {"u_location": jnp.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], bool)},
        jnp.zeros((25,), bool),
    ],
    ids=["level", "per-step", "bare"],
)
def test_a_mask_the_eager_faces_refuse_is_refused_with_the_same_sentence(face, mask):
    system, params, _, beliefs = _tmaze_obs()
    compiled = system.compiled(watch=False)
    key = jax.random.PRNGKey(2)
    if face == "act":
        eager = _refusal(lambda: system.act(key, params, beliefs, metrics=False, legal=mask))
        got = _refusal(lambda: compiled.act(key, params, beliefs, metrics=False, legal=mask))
    else:
        eager = _refusal(lambda: system.infer_policies(params, beliefs, legal=mask))
        got = _refusal(lambda: compiled.infer_policies(params, beliefs, legal=mask))
    assert got == eager
    assert compiled.trace_counts == {}


def test_legal_given_as_lists_traces_and_matches_eager():
    system, params, _, beliefs = _tmaze_obs()
    compiled = system.compiled()
    keyed_list = {"u_location": [False, False, False, True, False]}
    key = jax.random.PRNGKey(2)
    eager = system.act(key, params, beliefs, metrics=False, legal=keyed_list)[0]
    got = compiled.act(key, params, beliefs, metrics=False, legal=keyed_list)[0]
    _assert_close(got, eager)
    bare = [bool(i % 2) for i in range(25)]
    q_eager = system.infer_policies(params, beliefs, metrics=False, legal=bare)[0]
    q = compiled.infer_policies(params, beliefs, metrics=False, legal=bare)[0]
    _assert_close(q, q_eager)
    with pytest.raises(TypeError, match="a list as legal="):
        compiled.infer_policies(params, beliefs, legal=[1] * 25)


def test_a_float_legal_mask_is_refused_with_a_sentence():
    system, params, _, beliefs = _tmaze_obs()
    compiled = system.compiled()
    with pytest.raises(ValueError, match=rf"legal\['u_location'\] is a {DEFAULT_FLOAT} array"):
        compiled.act(jax.random.PRNGKey(0), params, beliefs, legal={"u_location": jnp.ones((5,))})


def test_present_given_as_a_list_is_traced():
    system, params, obs, beliefs = _tmaze_obs()
    compiled = system.compiled()
    got = compiled.infer_policies(params, beliefs, metrics=False, present={"cue": [True, False]})
    want = system.infer_policies(params, beliefs, metrics=False, present={"cue": [True, False]})
    _assert_close(got[0], want[0])


def test_numpy_committed_and_weak_parameters_cost_no_second_trace():
    system, params, obs, _ = _tmaze_obs()
    compiled = system.compiled()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RetraceWarning)
        compiled.infer_states(params, obs)
        compiled.infer_states(jax.tree_util.tree_map(np.asarray, params), obs)
        compiled.infer_states(jax.device_put(params, jax.devices()[0]), obs)
        compiled.infer_states(params, {k: np.asarray(v) for k, v in obs.items()})
    assert compiled.trace_counts["infer_states"] == 1


def test_typed_and_raw_prng_keys_share_a_trace():
    system, params, _, beliefs = _tmaze_obs()
    compiled = system.compiled()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RetraceWarning)
        compiled.act(jax.random.PRNGKey(0), params, beliefs, metrics=False)
        compiled.act(jax.random.key(1), params, beliefs, metrics=False)
    assert compiled.trace_counts["act"] == 1


def test_incomplete_priors_share_the_pushed_structure():
    """A modality that observes an action node, and a prior without a control node's entry: the
    incomplete prior is completed to the keys the push returns."""
    from tests.test_noise_map_mixture import _levelled, _sensor, _showcase

    system, params = _showcase(_levelled(), planner="mixed", policy_len=1)
    compiled = system.compiled()
    env, env_params = system.as_env(params)
    key = jax.random.PRNGKey(1)
    obs, state = env.reset(key, env_params)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RetraceWarning)
        #  the first tick's prior names the observed node's level, as a pushed prior does
        priors = {**system.initial_priors(params), "sensor": _sensor(0)}
        for tick in range(3):
            key, ka, ke = jax.random.split(key, 3)
            if tick == 1:  # a prior that leaves out the control node's entry
                priors = {n: q for n, q in priors.items() if n in ("terrain", "x", "sensor")}
            beliefs, _ = compiled.infer_states(params, obs, priors)
            action, _, _ = compiled.act(ka, params, beliefs, metrics=False)
            priors = compiled.update_empirical_prior(params, beliefs, action)
            obs, state = env.step(ke, state, action, env_params)
    assert compiled.trace_counts["infer_states"] == 1


def test_a_retrace_through_a_second_compiled_object_is_still_named():
    system, params, obs, _ = _tmaze_obs()
    first, second = system.compiled(), system.compiled()
    first.infer_states(params, obs)
    second.infer_states(params, obs)  # a cache hit
    one_hot = jax.nn.one_hot(obs["cue"], params.likelihood["cue"].log_table.shape[0])
    with pytest.warns(RetraceWarning, match=r"observations\['cue'\]"):
        second.infer_states(params, {**obs, "cue": one_hot})
    assert (first.trace_counts, second.trace_counts) == (
        {"infer_states": 1},
        {"infer_states": 1},
    )


def test_the_watcher_does_not_count_a_dispatch_to_a_cached_program():
    @eqx.filter_jit
    def total(x):
        return x.sum()

    watched = debug.watch(total)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RetraceWarning)
        watched(jnp.ones(3))
        watched(np.ones(3, DEFAULT_FLOAT))  # a new dispatch signature, the same traced program
    assert watched.traces == 1
    jax.clear_caches()
    with pytest.warns(RetraceWarning, match="cache was cleared"):
        watched(jnp.ones(3))
