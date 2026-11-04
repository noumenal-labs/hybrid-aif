"""A modality over an action node under ``merge`` and ``subsystem``.

The component's modality keeps reading the realised action of its own node: ``merge`` shifts the
node's index to the composite's, ``subsystem`` re-expresses it in the view's, and the composite's
``as_env`` emits the reading at the realised level. ``learn`` and ``replay`` keep their refusals.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Categorical, ConditionalCategorical
from hybrid_aif.composition.merge import ActionControl, BoundaryPrior, merge, subsystem
from hybrid_aif.replay import replay
from tests.test_action_node_modalities import LEVEL_SD, PROCESS_SD, scalar_model
from tests.test_merge_gaussian_control import mission

KEY = jax.random.PRNGKey(0)


def _pair():
    fast, fast_p = scalar_model(horizon=2)
    slow, slow_p = mission()
    slow = dataclasses.replace(
        slow,
        planner=fast.planner,
        num_iters=fast.num_iters,
        mixture_k=fast.mixture_k,
        mixture_schedule=fast.mixture_schedule,
        planning_belief=fast.planning_belief,
        policy_len=fast.policy_len,
        epistemic=fast.epistemic,
    )
    system, params = merge({"slow": slow, "fast": fast}, {"slow": slow_p, "fast": fast_p})
    return (slow, slow_p), (fast, fast_p), (system, params)


def test_the_node_index_is_the_composites_and_the_composite_reads_it():
    _, _, (system, _) = _pair()
    #  factors: slow.leg, fast.x; nodes: slow.u_option, fast.u; the composite has 4 columns
    assert system.A_deps == ((0,), (1, 4 + 1))
    assert system._sensed == ("fast.a",)
    assert system._sensed_columns == (1,)


def test_inference_on_the_composite_and_the_view_agrees_with_the_standalone_fast_system():
    (_, _), (fast, fast_p), (system, params) = _pair()
    view, view_p = subsystem(system, params, "fast")
    assert view._sensed == ("fast.a",)
    y = jnp.asarray([0.55])
    own = fast.initial_priors(fast_p)
    for level in range(3):
        copy = Categorical.from_probs(jax.nn.one_hot(level, 3))
        ref, _ = fast.infer_states(fast_p, {"y": y}, {**own, "a": copy})
        beliefs = system.initial_priors(params)
        got, _ = system.infer_states(
            params, {"slow.region": jnp.asarray(0), "fast.y": y}, {**beliefs, "fast.a": copy}
        )
        np.testing.assert_allclose(got["fast.x"].loc, ref["x"].loc, rtol=1e-5)
        np.testing.assert_allclose(got["fast.x"].cov, ref["x"].cov, rtol=1e-5)
        vbeliefs = view.initial_priors(view_p)
        seen, _ = view.infer_states(view_p, {"fast.y": y}, {**vbeliefs, "fast.a": copy})
        np.testing.assert_allclose(seen["fast.x"].loc, ref["x"].loc, rtol=1e-5)
        np.testing.assert_allclose(seen["fast.x"].cov, ref["x"].cov, rtol=1e-5)


def test_the_composites_world_emits_the_fix_with_the_noise_of_the_realised_level():
    _, _, (system, params) = _pair()
    env, eparams = system.as_env(params)
    _, state = env.reset(KEY, eparams)
    for level in (0, 1):
        keys = jax.random.split(jax.random.PRNGKey(level), 400)
        actions = {"slow.option": jnp.asarray(0), "fast.a": jnp.asarray(level)}
        draws = jax.vmap(
            lambda k, actions=actions: env.step_actuated(k, state, actions, eparams)[0]["fast.y"]
        )(keys)
        np.testing.assert_allclose(
            np.asarray(draws)[:, 0].std(), np.hypot(LEVEL_SD[level], PROCESS_SD), rtol=0.15
        )


def test_a_view_that_does_not_read_the_node_does_not_carry_it():
    _, _, (system, params) = _pair()
    slow_view, _ = subsystem(system, params, "slow")
    assert slow_view._sensed == ()
    assert slow_view.A_deps == ((0,),)


def test_an_identified_node_is_read_at_the_drivers_action_and_carried_by_the_view():
    """A node of the fast tier made one with the slow tier's keeps the fix's selection: the
    composite's modality reads the driver's action node, and the fast view carries that node."""
    fast, fast_p = scalar_model(horizon=2)
    slow, slow_p = mission()
    #  a slow tier whose node has the fast node's three levels
    t = np.zeros((2, 2, 3))
    t[:, :, 0], t[:, :, 1], t[:, :, 2] = [[0.9, 0.2], [0.1, 0.8]], [[0.3, 0.1], [0.7, 0.9]], 0.5
    slow_p = dataclasses.replace(
        slow_p, transition={"leg": ConditionalCategorical(jnp.log(jnp.asarray(t)), 2)}
    )
    slow = dataclasses.replace(
        slow,
        planner=fast.planner,
        num_iters=fast.num_iters,
        mixture_k=fast.mixture_k,
        mixture_schedule=fast.mixture_schedule,
        planning_belief=fast.planning_belief,
        policy_len=fast.policy_len,
        epistemic=fast.epistemic,
    )
    system, params = merge(
        {"slow": slow, "fast": fast},
        {"slow": slow_p, "fast": fast_p},
        [ActionControl("slow.u_option", "fast.u")],
    )
    assert system.action_names == ("slow.option",)
    assert system._sensed == ("slow.option",)
    assert system.A_deps == ((0,), (1, 3))
    view, view_p = subsystem(system, params, "fast")
    assert view._sensed == ("slow.option",)
    y = jnp.asarray([0.55])
    copy = Categorical.from_probs(jax.nn.one_hot(1, 3))
    ref, _ = fast.infer_states(fast_p, {"y": y}, {**fast.initial_priors(fast_p), "a": copy})
    got, _ = view.infer_states(
        view_p, {"fast.y": y}, {**view.initial_priors(view_p), "slow.option": copy}
    )
    np.testing.assert_allclose(got["fast.x"].loc, ref["x"].loc, rtol=1e-5)


def test_learn_and_replay_keep_their_refusals():
    _, _, (system, params) = _pair()
    obs = {"slow.region": jnp.asarray(0), "fast.y": jnp.asarray([0.5])}
    beliefs = system.initial_priors(params)
    with pytest.raises(NotImplementedError, match=r"learn: .*fast\.a"):
        system.learn(params, obs, beliefs)
    with pytest.raises(NotImplementedError, match=r"replay: .*fast\.a"):
        replay(system, params, obs, {"slow.option": jnp.asarray(0), "fast.a": jnp.asarray(0)})


def test_the_composites_description_round_trips():
    from hybrid_aif.description import build, describe

    _, _, (system, params) = _pair()
    document, arrays = describe(system, params, arrays="inline")
    rebuilt, _ = build(document, arrays=arrays)
    assert rebuilt.A_deps == system.A_deps
    assert rebuilt.subsystems == system.subsystems


def test_a_boundary_prior_on_a_node_a_modality_observes_is_refused_naming_the_modality():
    fast, fast_p = scalar_model(horizon=2)
    slow, slow_p = mission()
    with pytest.raises(NotImplementedError, match=r"modality 'fast\.y' observes the action node"):
        merge(
            {"slow": slow, "fast": fast},
            {"slow": slow_p, "fast": fast_p},
            [BoundaryPrior("fast.u", 1)],
        )


def _two_node_model():
    """``scalar_model`` with a second node and scalar state, both read by the one modality."""
    system, params = scalar_model(horizon=2)
    system = dataclasses.replace(
        system,
        A_deps=((0, 4, 5),),
        B_deps=((2,), (3,), (), ()),
        controls=(2, 3),
        action_names=("a", "b"),
        factor_names=("x", "z", "u", "v"),
    )
    likelihood = dataclasses.replace(
        params.likelihood["y"],
        noise=jnp.broadcast_to(jnp.asarray(LEVEL_SD).reshape(3, 1, 1, 1), (3, 3, 1, 1)),
    )
    return system, dataclasses.replace(
        params,
        transition={"x": params.transition["x"], "z": params.transition["x"]},
        initial_state_prior={
            "x": params.initial_state_prior["x"],
            "z": params.initial_state_prior["x"],
        },
        likelihood={"y": likelihood},
    )


def test_a_modality_observing_one_action_node_twice_after_identification_is_refused():
    fast, fast_p = _two_node_model()
    slow, slow_p = mission()
    t = np.zeros((2, 2, 3))
    t[:, :, 0], t[:, :, 1], t[:, :, 2] = [[0.9, 0.2], [0.1, 0.8]], [[0.3, 0.1], [0.7, 0.9]], 0.5
    slow_p = dataclasses.replace(
        slow_p, transition={"leg": ConditionalCategorical(jnp.log(jnp.asarray(t)), 2)}
    )
    with pytest.raises(ValueError, match=r"observes one action node twice"):
        merge(
            {"slow": slow, "fast": fast},
            {"slow": slow_p, "fast": fast_p},
            [ActionControl("slow.u_option", "fast.u"), ActionControl("slow.u_option", "fast.v")],
        )
