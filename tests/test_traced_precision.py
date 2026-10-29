"""``SystemParams.precision``: the policy precision as a traced parameter.

Present, it equals ``System(gamma=value)`` bitwise on every planner that reads the precision;
changing its value recompiles nothing; the description format writes it as a parameter; and
``merge`` carries one precision for the composite.

Design: docs/design/recompilation.md R5.
"""

import copy
import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import SystemParams
from hybrid_aif import description as D
from hybrid_aif.composition.merge import merge
from hybrid_aif.description import mdl
from hybrid_aif.planning import make_gradient_planner
from tests.test_closed_loop import small_lg
from tests.test_merge import OBSERVATION_BINDINGS, agent_and_world
from tests.test_system import tmaze_system

GAMMA = 7.0


def _same(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _tmaze():
    return tmaze_system(policy_len=2)


def _sophisticated():
    system, params = tmaze_system(policy_len=2)
    return dataclasses.replace(system, planner="sophisticated", branch_on=("cue",)), params


def _gradient():
    return small_lg()


@pytest.mark.parametrize("make", [_tmaze, _sophisticated, _gradient])
def test_precision_in_the_parameters_equals_the_static_gamma_bitwise(make):
    system, params = make()
    static = dataclasses.replace(system, gamma=GAMMA)
    traced = dataclasses.replace(params, precision=GAMMA)
    env, env_params = system.as_env(params)
    obs, _ = env.reset(jax.random.PRNGKey(0), env_params)
    beliefs, _ = system.infer_states(params, obs)
    _same(
        static.infer_policies(params, beliefs, metrics=False),
        system.infer_policies(traced, beliefs, metrics=False),
    )
    key = jax.random.PRNGKey(1)
    _same(static.act(key, params, beliefs), system.act(key, traced, beliefs))


def test_the_laplace_covariance_reads_the_traced_precision_bitwise():
    system, params = _gradient()
    laplace = make_gradient_planner(posterior="laplace")
    env, env_params = system.as_env(params)
    obs, _ = env.reset(jax.random.PRNGKey(0), env_params)
    beliefs, _ = system.infer_states(params, obs)
    key = jax.random.PRNGKey(1)
    static = dataclasses.replace(system, gamma=GAMMA)
    traced = dataclasses.replace(params, precision=GAMMA)
    got = system.act(key, traced, beliefs, planner=laplace)
    _same(static.act(key, params, beliefs, planner=laplace), got)
    other = system.act(key, dataclasses.replace(params, precision=1.0), beliefs, planner=laplace)
    assert not np.array_equal(np.asarray(got[0]["u"]), np.asarray(other[0]["u"]))


def test_an_absent_precision_reads_the_systems_gamma():
    system, params = _tmaze()
    assert params.precision is None
    env, env_params = system.as_env(params)
    obs, _ = env.reset(jax.random.PRNGKey(0), env_params)
    beliefs, _ = system.infer_states(params, obs)
    _same(
        system.infer_policies(params, beliefs, metrics=False),
        system.infer_policies(
            dataclasses.replace(params, precision=system.gamma), beliefs, metrics=False
        ),
    )


def test_annealing_the_precision_compiles_once():
    system, params = _tmaze()
    params = dataclasses.replace(params, precision=1.0)
    env, env_params = system.as_env(params)
    obs, _ = env.reset(jax.random.PRNGKey(0), env_params)
    beliefs, _ = system.infer_states(params, obs)
    traces = []

    @eqx.filter_jit
    def plan(system, params, beliefs):
        traces.append(1)
        return system.infer_policies(params, beliefs, metrics=False)[0]

    posteriors = []
    for step in range(10):
        annealed = dataclasses.replace(params, precision=jnp.float32(1.0 + step))
        posteriors.append(np.asarray(plan(system, annealed, beliefs).logits))
    assert len(traces) == 1
    assert not np.array_equal(posteriors[0], posteriors[-1])


def test_a_precision_that_is_not_a_scalar_is_refused():
    _, params = _tmaze()
    with pytest.raises(ValueError, match="policy precision, a scalar"):
        dataclasses.replace(params, precision=jnp.ones(2))


def test_precision_round_trips_through_the_description():
    system, params = _tmaze()
    params = dataclasses.replace(params, precision=GAMMA)
    description = D.describe(system, params)
    document = description.document
    assert document["variables"]["precision"] == {"domain": "positive", "shape": []}
    assert document["clamps"]["precision"] == GAMMA
    assert document["configuration"]["planning"]["precision"] == "precision"
    built_system, built = D.build(document, arrays=description.arrays)
    assert built.precision.dtype == params.precision.dtype
    np.testing.assert_array_equal(np.asarray(built.precision), np.asarray(params.precision))
    again = D.describe(built_system, built)
    assert again.document == document
    assert (
        D.describe(system, dataclasses.replace(params, precision=None))
        .document["configuration"]["planning"]
        .get("precision")
        is None
    )


def test_the_description_length_does_not_price_the_precision():
    system, params = _tmaze()
    plain = D.describe(system, params)
    traced = D.describe(system, dataclasses.replace(params, precision=GAMMA))
    lengths = [
        mdl.description_length(d.document, arrays=d.arrays, n=100.0) for d in (plain, traced)
    ]
    assert lengths[0].total == lengths[1].total


def _traced_document():
    system, params = _tmaze()
    description = D.describe(system, dataclasses.replace(params, precision=GAMMA))
    return copy.deepcopy(description.document), description.arrays


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda d: d["configuration"]["planning"].update(precision="nowhere"), "precision names"),
        (lambda d: d["clamps"].update(precision=-1.0), "positive"),
        (lambda d: d["variables"]["precision"].update(shape=[2]), "policy precision"),
    ],
)
def test_the_description_refuses_a_bad_precision(edit, match):
    document, arrays = _traced_document()
    edit(document)
    with pytest.raises(D.DescriptionError, match=match):
        D.build(document, arrays=arrays)


def _components(precisions):
    (agent, ap), (world, wp) = agent_and_world()
    ap = dataclasses.replace(ap, precision=precisions[0])
    wp = dataclasses.replace(wp, precision=precisions[1])
    return {"agent": agent, "world": world}, {"agent": ap, "world": wp}


def test_merge_carries_the_one_precision_the_components_share():
    systems, params = _components((GAMMA, GAMMA))
    _, composite = merge(systems, params, OBSERVATION_BINDINGS)
    assert float(composite.precision) == GAMMA
    systems, params = _components((None, None))
    _, composite = merge(systems, params, OBSERVATION_BINDINGS)
    assert composite.precision is None


def test_merge_refuses_components_that_disagree_on_the_precision():
    systems, params = _components((GAMMA, 3.0))
    with pytest.raises(ValueError, match="different params.precision"):
        merge(systems, params, OBSERVATION_BINDINGS)
    systems, params = _components((GAMMA, None))
    with pytest.raises(ValueError, match="declare params.precision and the others do not"):
        merge(systems, params, OBSERVATION_BINDINGS)


def test_the_field_is_a_system_params_field():
    assert "precision" in {f.name for f in dataclasses.fields(SystemParams)}
