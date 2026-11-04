"""A neural likelihood (``MLPConditional``) and its target through ``describe``, ``build``,
``merge`` and ``subsystem``: carried as a linear-Gaussian one is (design:
recognition-beside-analytic.md L8)."""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import ConditionalGaussian, Gaussian
from hybrid_aif.composition.merge import merge, subsystem
from hybrid_aif.description import build, describe
from hybrid_aif.distributions import MLPConditional
from hybrid_aif.factors import LinearGaussianTransition
from hybrid_aif.system import System, SystemParams

KEY = jax.random.PRNGKey(0)


def _neural(propagation):
    module = MLPConditional.from_shapes((2,), 3, key=KEY, width=4, depth=1)
    return dataclasses.replace(module, propagation=propagation)


def component(propagation="first_order", target=True):
    """One continuous factor x (2-d) observed through a neural y (3-d), with a Gaussian target."""
    system = System(
        A_deps=((0,),),
        B_deps=((),),
        controlled=(False,),
        policy_len=1,
        factor_names=("x",),
        modality_names=("y",),
    )
    params = SystemParams(
        likelihood={"y": _neural(propagation)},
        transition={
            "x": LinearGaussianTransition(
                F=jnp.eye(2),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 2)), chol=0.1 * jnp.eye(2)[None], parent_ndims=1
                ),
            )
        },
        initial_state_prior={"x": Gaussian(loc=jnp.zeros(2), chol=jnp.eye(2))},
        obs_targets=({"y": Gaussian(loc=jnp.ones(3), chol=0.5 * jnp.eye(3))} if target else {}),
    )
    return system, params


def _same(a, b):
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        assert np.array_equal(np.asarray(x), np.asarray(y))


@pytest.mark.parametrize("propagation", ["first_order", "unscented"])
def test_propagation_and_the_target_round_trip_through_describe_and_build(propagation):
    system, params = component(propagation)
    document, arrays = describe(system, params)
    document = json.loads(json.dumps(document))
    options = document["factors"]["y"]["likelihood"]["options"]
    assert options["propagation"] == propagation
    assert "target" in document["factors"]["y"]
    system2, params2 = build(document, arrays=arrays)
    assert getattr(params2.likelihood["y"], "propagation", "first_order") == propagation
    _same(params, params2)
    _same(system, system2)


def test_a_neural_modality_and_its_target_carry_through_merge_and_subsystem():
    a, ap = component()
    b, bp = component(target=False)
    comp, cp = merge({"a": a, "b": b}, {"a": ap, "b": bp})
    assert cp.likelihood["a.y"] is ap.likelihood["y"]
    assert set(cp.obs_targets) == {"a.y"}
    view, vp = subsystem(comp, cp, "a")
    assert vp.likelihood["a.y"] is ap.likelihood["y"]
    assert set(vp.obs_targets) == {"a.y"}
    assert subsystem(comp, cp, "b")[1].obs_targets == {}
