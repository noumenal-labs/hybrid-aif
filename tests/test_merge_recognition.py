"""``merge`` and ``subsystem`` carry a recognition entry and external inputs by component."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_aif import ConditionalGaussian, Gaussian, LinearGaussianObservation
from hybrid_aif.composition.merge import merge, subsystem
from hybrid_aif.distributions import MLPConditional
from hybrid_aif.factors import LinearGaussianTransition, Recognition
from hybrid_aif.system import System, SystemParams

KEY = jax.random.PRNGKey(0)


def component(recognition=(), module=None, external=()):
    """One continuous factor x (2-d) observed through y (2-d)."""
    system = System(
        A_deps=((0,),),
        B_deps=((),),
        controlled=(False,),
        policy_len=1,
        factor_names=("x",),
        modality_names=("y",),
        recognition=recognition,
        external_inputs=external,
    )
    params = SystemParams(
        likelihood={
            "y": LinearGaussianObservation(H=jnp.eye(2), c=jnp.zeros(2), noise=0.3 * jnp.eye(2))
        },
        transition={
            "x": LinearGaussianTransition(
                F=jnp.eye(2),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 2)), chol=0.1 * jnp.eye(2)[None], parent_ndims=1
                ),
            )
        },
        initial_state_prior={"x": Gaussian(loc=jnp.zeros(2), chol=jnp.eye(2))},
        recognition={} if module is None else {"x": module},
    )
    return system, params


def _module(width=2):
    module = MLPConditional.from_shapes((width,), 2, key=KEY, width=2, depth=1)
    return dataclasses.replace(module, frozen=True)


def test_a_recognition_entry_round_trips_through_merge_and_subsystem():
    module = _module()
    a, ap = component((Recognition("x", ("obs:y",)),), module)
    b, bp = component()
    comp, cp = merge({"a": a, "b": b}, {"a": ap, "b": bp})
    assert comp.recognition == (Recognition("a.x", ("obs:a.y",)),)
    assert cp.recognition["a.x"] is module
    view, vp = subsystem(comp, cp, "a")
    assert view.recognition == comp.recognition
    assert vp.recognition["a.x"] is module
    other, op = subsystem(comp, cp, "b")
    assert other.recognition is None and not op.recognition
    q, _ = view.infer_states(vp, {"a.y": jnp.asarray([0.5, -0.5])})
    assert np.isfinite(np.asarray(q["a.x"].loc)).all()


def test_external_inputs_carry_under_the_prefix_and_return_to_their_component():
    a, ap = component((Recognition("x", ("external:e",)),), _module(3), external=(("e", 3),))
    b, bp = component()
    comp, cp = merge({"a": a, "b": b}, {"a": ap, "b": bp})
    assert comp.external_inputs == (("a.e", 3),)
    assert comp.recognition == (Recognition("a.x", ("external:a.e",)),)
    view, _ = subsystem(comp, cp, "a")
    assert view.external_inputs == (("a.e", 3),)
    assert subsystem(comp, cp, "b")[0].external_inputs == ()


def test_two_components_declaring_one_external_name_stay_distinct():
    a, ap = component(external=(("e", 3),))
    b, bp = component(external=(("e", 3),))
    comp, _ = merge({"a": a, "b": b}, {"a": ap, "b": bp})
    assert comp.external_inputs == (("a.e", 3), ("b.e", 3))
