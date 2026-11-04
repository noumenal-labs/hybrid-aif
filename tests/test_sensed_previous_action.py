"""A sensed action node's prior for ``infer_states`` defaults to the previous realised action.

The node's prior is the entry ``priors`` carries, else the one-hot at the level ``previous``
names, else uniform (which a channel that selects its head and noise at the node refuses).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Categorical
from tests.test_action_node_modalities import scalar_model

Y = {"y": jnp.asarray([0.55])}


def _run(system, params, priors=None, previous=None):
    return system.infer_states(params, Y, priors, detail=True, previous=previous)


def test_the_default_prior_equals_an_explicit_one_hot_prior():
    system, params = scalar_model()
    beliefs = system.initial_priors(params)
    for level in range(3):
        point = Categorical.from_probs(jax.nn.one_hot(level, 3))
        ref, ref_info = _run(system, params, {**beliefs, "a": point})
        got, info = _run(system, params, beliefs, previous={"a": jnp.asarray(level)})
        np.testing.assert_allclose(got["x"].loc, ref["x"].loc, rtol=1e-6)
        np.testing.assert_allclose(got["x"].cov, ref["x"].cov, rtol=1e-6)
        np.testing.assert_allclose(info.log_evidence["y"], ref_info.log_evidence["y"], rtol=1e-6)


def test_an_explicit_prior_overrides_previous():
    system, params = scalar_model()
    beliefs = system.initial_priors(params)
    point = Categorical.from_probs(jax.nn.one_hot(1, 3))
    ref, _ = _run(system, params, {**beliefs, "a": point})
    got, _ = _run(system, params, {**beliefs, "a": point}, previous={"a": jnp.asarray(0)})
    np.testing.assert_allclose(got["x"].loc, ref["x"].loc, rtol=1e-6)
    other, _ = _run(system, params, beliefs, previous={"a": jnp.asarray(0)})
    assert not np.allclose(other["x"].loc, ref["x"].loc)


def test_with_neither_the_point_prior_refusal_still_fires():
    system, params = scalar_model()
    beliefs = system.initial_priors(params)
    pattern = r"selects its head and noise by action node 'a'.*is a point.*MixtureGaussian"
    with pytest.raises(ValueError, match=pattern):
        _run(system, params, beliefs)
    with pytest.raises(ValueError, match=pattern):
        _run(system, params, beliefs, previous={})


def test_a_traced_previous_action_works_under_jit():
    system, params = scalar_model()
    beliefs = system.initial_priors(params)
    run = jax.jit(
        lambda level: system.infer_states(params, Y, beliefs, previous={"a": level})[0]["x"].loc
    )
    for level in range(3):
        point = Categorical.from_probs(jax.nn.one_hot(level, 3))
        ref = system.infer_states(params, Y, {**beliefs, "a": point})[0]["x"].loc
        np.testing.assert_allclose(run(jnp.asarray(level)), ref, rtol=1e-5)


def test_an_unnamed_node_falls_to_uniform_not_to_level_zero():
    from tests.test_noise_map_mixture import _levelled, _showcase

    system, params = _showcase(_levelled())
    beliefs = system.initial_priors(params)
    y = {"fix": jnp.array([0.2, 0.1])}
    pattern = r"noise map by action node 'sensor'.*belief is a point"
    with pytest.raises(ValueError, match=pattern):
        system.infer_states(params, y, beliefs, previous={"a": jnp.array([0.7, 0.5])})


def test_a_continuous_node_entry_rides_beside_a_named_discrete_node():
    from tests.test_noise_map_mixture import _levelled, _sensor, _showcase

    system, params = _showcase(_levelled())
    beliefs = system.initial_priors(params)
    y = {"fix": jnp.array([0.2, 0.1])}
    ref = system.infer_states(params, y, {**beliefs, "sensor": _sensor(1)})[0]["x"].means
    for previous in (
        {"sensor": 1},
        {"a": jnp.array([0.3, -0.2]), "sensor": jnp.asarray(1)},
    ):
        got = system.infer_states(params, y, beliefs, previous=previous)[0]
        np.testing.assert_allclose(got["x"].means, ref, rtol=1e-6)
        np.testing.assert_allclose(got["sensor"].probs, [0.0, 1.0])


@pytest.mark.parametrize("level", [3, -1])
def test_an_out_of_range_level_is_refused_eagerly_and_under_jit(level):
    system, params = scalar_model()
    beliefs = system.initial_priors(params)
    with pytest.raises(ValueError, match=r"previous\['a'\].*3 level\(s\)"):
        _run(system, params, beliefs, previous={"a": level})
    run = jax.jit(
        lambda lv: system.infer_states(params, Y, beliefs, previous={"a": lv})[0]["x"].loc
    )
    with pytest.raises(Exception, match=r"previous\['a'\].*3 level\(s\)"):
        jax.block_until_ready(run(jnp.asarray(level)))


@pytest.mark.parametrize("level", [1.7, -0.5])
def test_a_non_integer_level_is_refused(level):
    system, params = scalar_model()
    beliefs = system.initial_priors(params)
    with pytest.raises(ValueError, match=r"previous\['a'\] is .* takes a level"):
        _run(system, params, beliefs, previous={"a": level})
