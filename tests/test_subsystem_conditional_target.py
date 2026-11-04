"""A per-step target conditional on a partner's control node, carried by ``subsystem``.

The view contracts that parent axis of the target with the node's pinned prior, as it does a
factor's transition axis: a level slices it, a distribution mixes a categorical target's
preferences in probability space, as a soft step does, and a Gaussian target takes a level. A
target conditional on the view's own node or on a state factor is carried unchanged.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import (
    Categorical,
    ConditionalCategorical,
    ConditionalGaussian,
    Gaussian,
    LinearGaussianObservation,
)
from hybrid_aif.composition.merge import ActionControl, merge, subsystem
from hybrid_aif.controls import Box, ControlStep
from hybrid_aif.distributions import MixtureGaussian
from hybrid_aif.factors import LinearGaussianTransition
from hybrid_aif.objectives import total
from hybrid_aif.objectives.discrete import resolve_step_targets
from hybrid_aif.system import System, SystemParams
from tests.test_merge_gaussian_control import _cc, mission

OPTIONS = 2
M = 2
TARGET_LOC = np.asarray([[1.0, 1.0], [-1.0, 0.5]])
PLANS = jnp.asarray(np.random.default_rng(3).uniform(-1, 1, (4, 2, 2)), jnp.float32)
TARGET_CHOL = np.stack([0.5 * np.eye(2), 0.8 * np.eye(2)])


def _rover(target=None, c_deps=(), per_step=True):
    """A rover whose terrain reads the option node and whose position ``x`` has a target that
    may be conditional on it (``target`` is the declared target, ``c_deps`` its parents)."""
    system = System(
        A_deps=((1,),),
        B_deps=((0, 3), (0, 2), (), ()),
        controls=(2, 3),
        action_names=("a_v", "a_opt"),
        factor_names=("terrain", "x", "v", "u_opt"),
        modality_names=("fix",),
        policy_len=2,
        planner="gradient",
        num_iters=5,
        mixture_k=4,
        mixture_schedule="mode_aligned",
        planning_belief="prior",
        C_deps=c_deps,
    )
    rng = np.random.default_rng(1)
    t = np.zeros((M, M, OPTIONS))
    t[:, :, 0], t[:, :, 1] = [[0.9, 0.2], [0.1, 0.8]], [[0.3, 0.1], [0.7, 0.9]]
    F = np.asarray([[1.0, 0.1], [0.0, 0.9]]) + 0.05 * rng.standard_normal((M, 2, 2))
    G = 0.3 * rng.standard_normal((M, 2, 2))
    params = SystemParams(
        likelihood={
            "fix": LinearGaussianObservation(H=jnp.eye(2), c=jnp.zeros(2), noise=0.3 * jnp.eye(2))
        },
        transition={
            "terrain": _cc(t, 2),
            "x": LinearGaussianTransition(
                F=jnp.asarray(F, jnp.float32),
                G=jnp.asarray(G, jnp.float32),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((M, 2)),
                    chol=jnp.broadcast_to(0.2 * jnp.eye(2), (M, 2, 2)),
                    parent_ndims=1,
                ),
            ),
        },
        initial_state_prior={
            "terrain": Categorical.from_probs(jnp.asarray([0.5, 0.5])),
            "x": MixtureGaussian(
                log_weights=jnp.log(jnp.ones(M) / M),
                means=jnp.zeros((M, 2)),
                chols=jnp.broadcast_to(0.5 * jnp.eye(2), (M, 2, 2)),
            ),
        },
        state_targets={"x": target if target is not None else Gaussian(jnp.ones(2), jnp.eye(2))},
        control={"v": Box(-jnp.ones(2), jnp.ones(2))},
    )
    return system, params


def _conditional_target(shared_chol=False):
    chol = TARGET_CHOL[0] if shared_chol else TARGET_CHOL
    return ConditionalGaussian(
        loc=jnp.asarray(TARGET_LOC, jnp.float32),
        chol=jnp.asarray(chol, jnp.float32),
        parent_ndims=1,
    )


def _two_tier(rover_system, rover_params):
    ms, mp = mission()
    return merge(
        {"mission": ms, "rover": rover_system},
        {"mission": mp, "rover": rover_params},
        [ActionControl("mission.u_option", "rover.u_opt")],
    )


def _scores(view, vp):
    q = view.initial_priors(vp)
    return np.asarray(total(view.score_policies(vp, q, {"rover.v": PLANS})).lower)


@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("shared_chol", [False, True])
def test_a_point_prior_slices_the_target_as_a_hand_sliced_composite_does(level, shared_chol):
    cond_system, cond_params = _rover(_conditional_target(shared_chol), (("x", ("u_opt",)),))
    cs, cp = _two_tier(cond_system, cond_params)
    view, vp = subsystem(cs, cp, "rover", partner_priors={"mission.u_option": level})
    assert view.C_deps == ()
    chol = TARGET_CHOL[0] if shared_chol else TARGET_CHOL[level]
    by_hand = Gaussian(loc=jnp.asarray(TARGET_LOC[level], jnp.float32), chol=jnp.asarray(chol))
    ref_system, ref_params = _rover(by_hand)
    ref_cs, ref_cp = _two_tier(ref_system, ref_params)
    ref, rp = subsystem(ref_cs, ref_cp, "rover", partner_priors={"mission.u_option": level})
    got = vp.state_targets["rover.x"]
    np.testing.assert_allclose(got.loc, by_hand.loc)
    np.testing.assert_allclose(got.chol, by_hand.chol)
    np.testing.assert_allclose(_scores(view, vp), _scores(ref, rp), rtol=1e-5)


def _tabular_pair(parents=("u_s",), table=None):
    """Two tabular components sharing an option node; the follower's factor ``s`` has a target
    conditional on ``parents`` (its option node ``u_s``, and the state factor ``w`` before it
    when given), the table child-first."""
    lead = System(
        A_deps=((0,),),
        B_deps=((0, 1), ()),
        controls=(1,),
        action_names=("pick",),
        factor_names=("d", "u_d"),
        modality_names=("seen",),
        policy_len=2,
        planning_belief="prior",
    )
    lp = SystemParams(
        likelihood={"seen": _cc(np.eye(2), 1)},
        transition={"d": _cc(np.stack([np.eye(2)] * 2, -1), 2)},
        initial_state_prior={"d": Categorical.from_probs(jnp.asarray([0.5, 0.5]))},
    )
    follow = System(
        A_deps=((0,),),
        B_deps=((0, 2), (1,), ()),
        controls=(2,),
        action_names=("choose",),
        factor_names=("s", "w", "u_s"),
        modality_names=("obs",),
        policy_len=2,
        planning_belief="prior",
        C_deps=(("s", tuple(parents)),),
    )
    t = np.zeros((2, 2, 2))
    t[:, :, 0], t[:, :, 1] = [[0.9, 0.2], [0.1, 0.8]], [[0.3, 0.1], [0.7, 0.9]]
    if table is None:
        table = jnp.asarray([[0.0, 1.0], [2.0, -1.0]])  # (child, option)
    fp = SystemParams(
        likelihood={"obs": _cc(np.eye(2), 1)},
        transition={"s": _cc(t, 2), "w": _cc(np.eye(2)[:, :, None], 2)},
        initial_state_prior={
            "s": Categorical.from_probs(jnp.asarray([0.6, 0.4])),
            "w": Categorical.from_probs(jnp.asarray([0.5, 0.5])),
        },
        state_targets={"s": ConditionalCategorical(table, len(parents))},
    )
    return merge(
        {"lead": lead, "follow": follow},
        {"lead": lp, "follow": fp},
        [ActionControl("lead.u_d", "follow.u_s")],
    ), table


def test_a_soft_prior_mixes_a_categorical_targets_preferences_in_probability_space():
    (cs, cp), tab = _tabular_pair()
    weights = jnp.asarray([0.25, 0.75])
    view, vp = subsystem(
        cs, cp, "follow", partner_priors={"lead.u_d": Categorical(logits=jnp.log(weights))}
    )
    assert view.C_deps == ()
    got = vp.state_targets["follow.s"]
    assert isinstance(got, Categorical)
    want = jnp.log(jax.nn.softmax(tab, axis=0) @ weights)
    np.testing.assert_allclose(got.logits, want, rtol=1e-5)
    geometric = jax.nn.log_softmax(tab, axis=0) @ weights
    assert not np.allclose(jax.nn.log_softmax(geometric), want, atol=1e-3)


@pytest.mark.parametrize("w", [[0.5, 0.5], [0.25, 0.75]])
def test_the_views_soft_target_is_the_composites_soft_step_resolution(w):
    (cs, cp), _ = _tabular_pair(table=jnp.asarray([[0.38, -0.4], [1.92, 0.31]]))
    weights = jnp.asarray(w)
    view, vp = subsystem(
        cs, cp, "follow", partner_priors={"lead.u_d": Categorical(logits=jnp.log(weights))}
    )
    soft_step = resolve_step_targets(
        {"s": cp.state_targets["follow.s"]},
        {"s": (("control", 0),)},
        ControlStep(entries=(weights,)),
        (),
    )["s"]
    np.testing.assert_allclose(
        jax.nn.log_softmax(vp.state_targets["follow.s"].logits),
        jax.nn.log_softmax(soft_step.logits),
        rtol=1e-5,
        atol=1e-6,
    )


def test_a_point_prior_on_a_categorical_target_is_a_slice():
    (cs, cp), tab = _tabular_pair()
    view, vp = subsystem(cs, cp, "follow", partner_priors={"lead.u_d": 1})
    np.testing.assert_allclose(vp.state_targets["follow.s"].logits, tab[:, 1])


def test_a_soft_prior_on_a_gaussian_target_is_refused_with_the_level_to_pass():
    cs, cp = _two_tier(*_rover(_conditional_target(), (("x", ("u_opt",)),)))
    with pytest.raises(NotImplementedError, match=r"mixture of Gaussians.*pass an int level"):
        subsystem(
            cs, cp, "rover", partner_priors={"mission.u_option": Categorical(logits=jnp.zeros(2))}
        )


def test_a_partner_node_without_an_entry_is_refused_naming_the_entry():
    cs, cp = _two_tier(*_rover(_conditional_target(), (("x", ("u_opt",)),)))
    with pytest.raises(ValueError, match=r"partner_priors has no entry.*partner_priors=\{'mission"):
        subsystem(cs, cp, "rover")


def test_a_target_on_the_views_own_node_is_carried_unchanged():
    ms, mp = mission()
    ms = dataclasses.replace(ms, C_deps=(("region", ("u_option",)),))
    mp = dataclasses.replace(
        mp,
        obs_targets={"region": ConditionalCategorical(jnp.asarray([[0.0, 1.0], [2.0, -1.0]]), 1)},
    )
    rs, rp = _rover()
    cs, cp = merge({"mission": ms, "rover": rs}, {"mission": mp, "rover": rp})
    view, vp = subsystem(cs, cp, "mission")
    assert view.C_deps == (("mission.region", ("mission.u_option",)),)
    assert vp.obs_targets["mission.region"] is cp.obs_targets["mission.region"]


def test_a_target_on_a_state_factor_is_carried_unchanged():
    ms, mp = mission()
    ms = dataclasses.replace(ms, C_deps=(("region", ("leg",)),))
    mp = dataclasses.replace(
        mp,
        obs_targets={"region": ConditionalCategorical(jnp.asarray([[0.0, 1.0], [2.0, -1.0]]), 1)},
    )
    rs, rp = _rover()
    cs, cp = merge({"mission": ms, "rover": rs}, {"mission": mp, "rover": rp})
    view, vp = subsystem(cs, cp, "mission")
    assert view.C_deps == (("mission.region", ("mission.leg",)),)
    assert vp.obs_targets["mission.region"] is cp.obs_targets["mission.region"]


def test_a_contracted_axis_leaves_the_other_parents_in_their_order():
    table = jnp.asarray(np.random.default_rng(2).standard_normal((2, 2, 2)), jnp.float32)
    (cs, cp), _ = _tabular_pair(("w", "u_s"), table)
    view, vp = subsystem(cs, cp, "follow", partner_priors={"lead.u_d": 1})
    assert view.C_deps == (("follow.s", ("follow.w",)),)
    got = vp.state_targets["follow.s"]
    assert got.parent_ndims == 1
    np.testing.assert_allclose(got.log_table, table[:, :, 1])
    weights = jnp.asarray([0.4, 0.6])
    view, vp = subsystem(
        cs, cp, "follow", partner_priors={"lead.u_d": Categorical(logits=jnp.log(weights))}
    )
    want = jnp.log(jnp.einsum("cwu,u->cw", jax.nn.softmax(table, axis=0), weights))
    np.testing.assert_allclose(vp.state_targets["follow.s"].log_table, want, rtol=1e-5)
