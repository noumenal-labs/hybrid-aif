"""``merge()`` identifies control nodes: the latent a driver names decides when the node is read.

``ActionControl(driver, control)`` names a control node of the driven subsystem. A driver that is
a control node of the driving subsystem is read at the same step: the two nodes become one, the
driver's. A driver that is a state of it is read a step later, through the one-tick relay.
``BoundaryPrior`` names a control node too. These tests pin the timing of both spellings against a
hand computation, the identification against a hand-built System with one shared control node,
the refusals (unknown drivers, disagreeing control nodes, keys naming a node a binding removed),
the boundary prior's numerics, and a continuous pair read through an LG factor's ``G u``.

The zoo pair is ``tests.test_merge``'s: a T-maze agent and the same model as the world it acts
on, the agent's modalities reading the world's factors.

Design: docs/design/control-states.md C1, C5, C8 (A bound node's clock).
"""

import dataclasses
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Categorical, ConditionalCategorical
from hybrid_aif.composition.driver import agent_respond
from hybrid_aif.composition.merge import ActionControl, BoundaryPrior, merge, subsystem
from hybrid_aif.description import build, describe
from hybrid_aif.reference.envs.tmaze import CENTRE, CUE
from hybrid_aif.system import System, SystemParams
from tests.test_continuous_nodes import G2, F, _box, _gauss, _lg
from tests.test_h_deps import _low
from tests.test_merge import DRIVE, OBSERVATION_BINDINGS, _obs, agent_and_world, merged

KEY = jax.random.PRNGKey(0)
SAME_STEP = ActionControl(driver="agent.u_location", control="world.u_location")


def same_step(policy_len=2):
    """The zoo pair with the agent's node driving the world's at the same step."""
    return merged(policy_len=policy_len, bindings=[SAME_STEP, *OBSERVATION_BINDINGS])


def _mixed_beliefs(system, params):
    """Beliefs that are not point masses, so a hand computation has something to mix."""
    beliefs = dict(system.initial_priors(params))
    beliefs["agent.location"] = Categorical.from_probs(jnp.asarray([0.6, 0.1, 0.1, 0.1, 0.1]))
    beliefs["world.location"] = Categorical.from_probs(jnp.asarray([0.5, 0.2, 0.1, 0.1, 0.1]))
    return beliefs


def _point(level: int) -> Categorical:
    return Categorical.from_probs(jnp.zeros(5).at[level].set(1.0))


# ------------------------------------------------------------------------------- the clock


def test_a_node_driver_is_read_at_the_same_step_and_a_state_driver_a_step_later():
    """B's factor moves on A's decision at t under a node driver, at t+1 under a state driver."""
    (_, ap), (_, wp) = agent_and_world()
    world_table = np.exp(np.asarray(wp.transition["location"].log_probs))  # (next, prev, u)
    agent_table = np.exp(np.asarray(ap.transition["location"].log_probs))
    move = {"agent.a_location": jnp.asarray(CUE, jnp.int32)}
    turn = {"agent.a_location": jnp.asarray(2, jnp.int32)}  # the next tick's decision differs

    same, same_p = same_step()
    beliefs = _mixed_beliefs(same, same_p)
    qa = np.asarray(beliefs["agent.location"].probs)
    qw = np.asarray(beliefs["world.location"].probs)
    pushed = same.update_empirical_prior(same_p, beliefs, move)
    want = world_table[:, :, CUE] @ qw  # the decision taken at t moves the world at t
    np.testing.assert_allclose(
        np.asarray(pushed["world.location"].probs), want, rtol=1e-6, atol=1e-12
    )
    again = same.update_empirical_prior(same_p, pushed, turn)
    want = world_table[:, :, 2] @ np.asarray(pushed["world.location"].probs)  # ... and at t+1
    np.testing.assert_allclose(
        np.asarray(again["world.location"].probs), want, rtol=1e-6, atol=1e-12
    )

    later, later_p = merged()  # the relay: the agent's location drives the world's
    relay = np.exp(np.asarray(later_p.transition["world.location"].log_probs))[..., 0]
    pushed = later.update_empirical_prior(later_p, beliefs, move)
    want = np.einsum("nld,l,d->n", relay, qw, qa)  # the agent's location at t, not its decision
    np.testing.assert_allclose(
        np.asarray(pushed["world.location"].probs), want, rtol=1e-6, atol=1e-12
    )
    qa1 = agent_table[:, :, CUE] @ qa
    np.testing.assert_allclose(
        np.asarray(pushed["agent.location"].probs), qa1, rtol=1e-6, atol=1e-12
    )
    #  the second decision is not read: the world follows where the first one put the agent
    again = later.update_empirical_prior(later_p, pushed, turn)
    want = np.einsum("nld,l,d->n", relay, np.asarray(pushed["world.location"].probs), qa1)
    np.testing.assert_allclose(
        np.asarray(again["world.location"].probs), want, rtol=1e-6, atol=1e-12
    )

    #  from the centre, a move to the cue, then another decision: the same step against one
    #  tick behind
    sharp = {**beliefs, "agent.location": _point(CENTRE), "world.location": _point(CENTRE)}
    at_once = same.update_empirical_prior(same_p, sharp, move)
    after = same.update_empirical_prior(same_p, at_once, turn)["world.location"].probs
    one_late = later.update_empirical_prior(later_p, sharp, move)
    two_late = later.update_empirical_prior(later_p, one_late, turn)["world.location"].probs
    assert float(at_once["world.location"].probs[CUE]) > 0.999
    np.testing.assert_allclose(np.asarray(after), world_table[:, CUE, 2], rtol=1e-6, atol=1e-12)
    assert float(one_late["world.location"].probs[CENTRE]) > 0.999
    assert float(two_late[CUE]) > 0.999


# ------------------------------------------------------------------------- identification


def _one_shared_node(policy_len=2):
    """The same-step composite declared longhand: one control node read by two factors."""
    (agent, ap), (world, wp) = agent_and_world(policy_len)
    system = System(
        A_deps=((2,), (2, 3), (2, 3)),
        B_deps=((0,), (1,), (2,), (3,)),
        controlled=(True, False, False, False),
        B_control_deps=((0,), (), (0,), ()),
        policy_len=policy_len,
        gamma=16.0,
        epistemic=True,
        modality_names=("agent.location_obs", "agent.reward", "agent.cue"),
        factor_names=("agent.location", "agent.context", "world.location", "world.context"),
    )
    params = SystemParams(
        likelihood={f"agent.{m}": a for m, a in ap.likelihood.items()},
        transition={
            "agent.location": ap.transition["location"],
            "agent.context": ap.transition["context"],
            "world.location": wp.transition["location"],
            "world.context": wp.transition["context"],
        },
        initial_state_prior={
            "agent.location": ap.initial_state_prior["location"],
            "agent.context": ap.initial_state_prior["context"],
            "world.location": wp.initial_state_prior["location"],
            "world.context": wp.initial_state_prior["context"],
        },
        obs_targets={"agent.reward": ap.obs_targets["reward"]},
    )
    return system, params


def _assert_trees_equal(got, want):
    got_leaves, got_def = jax.tree_util.tree_flatten(got)
    want_leaves, want_def = jax.tree_util.tree_flatten(want)
    assert got_def == want_def
    for a, b in zip(got_leaves, want_leaves, strict=True):
        assert bool(jnp.all(jnp.isfinite(b)))
        assert jnp.array_equal(a, b)


def test_a_same_step_composite_is_one_shared_control_node():
    composite, cp = same_step()
    hand, hp = _one_shared_node()
    for field in dataclasses.fields(System):
        if field.name != "subsystems":
            assert getattr(composite, field.name) == getattr(hand, field.name), field.name
    assert composite.names.controls == ("agent.u_location",)
    assert "world.u_location" not in composite.factor_names

    beliefs = _mixed_beliefs(composite, cp)
    policies = composite.policies(cp)
    assert jnp.array_equal(policies, hand.policies(hp)) and policies.shape == (25, 2, 1)
    _assert_trees_equal(
        composite.score_policies(cp, beliefs, policies), hand.score_policies(hp, beliefs, policies)
    )
    plan = jnp.asarray([[CUE], [2]], jnp.int32)
    for got, want in zip(
        composite.predict_beliefs(cp, beliefs, plan),
        hand.predict_beliefs(hp, beliefs, plan),
        strict=True,
    ):
        assert set(got) == set(want)
        for name in want:
            assert jnp.array_equal(got[name].probs, want[name].probs), name
    move = {"agent.a_location": jnp.asarray(CUE, jnp.int32)}
    got = composite.update_empirical_prior(cp, beliefs, move)
    want = hand.update_empirical_prior(hp, beliefs, move)
    assert set(got) == set(want)
    for name in want:
        assert jnp.array_equal(got[name].probs, want[name].probs), name
    for selection in ("stochastic", "deterministic"):
        a_c, q_c, tree_c = composite.act(KEY, cp, beliefs, selection=selection)
        a_h, q_h, tree_h = hand.act(KEY, hp, beliefs, selection=selection)
        assert set(a_c) == set(a_h) == {"agent.a_location"}
        assert jnp.array_equal(a_c["agent.a_location"], a_h["agent.a_location"])
        assert jnp.array_equal(q_c.logits, q_h.logits)
        _assert_trees_equal(tree_c, tree_h)


def test_the_same_step_record_and_its_document():
    composite, cp = same_step()
    meta = dict(composite.subsystems)
    assert meta["world"].identified == (("world.u_location", "agent.u_location"),)
    assert meta["world"].relayed == () and meta["agent"].identified == ()
    assert meta["agent"].controls == ("agent.u_location",) and meta["world"].controls == ()
    #  the document writes the identification as structure: every factor that read the world's
    #  node reads the agent's, and the world's node is absent

    document, arrays = describe(composite, cp)
    assert "world.u_location" not in document["variables"]
    assert document["factors"]["world.location"]["transition"]["parents"] == [
        "world.location",
        "agent.u_location",
    ]
    rebuilt = build(document, arrays=arrays)[0]
    assert dict(rebuilt.subsystems)["world"].identified == ()
    assert dict(rebuilt.subsystems)["agent"].controls == ("agent.u_location",)


def test_a_nested_merge_requalifies_the_identification():
    inner, ip = same_step()
    solo, sp = agent_and_world()[0]
    outer, op = merge({"pair": inner, "solo": solo}, {"pair": ip, "solo": sp})
    meta = dict(outer.subsystems)
    pair = (("pair.world.u_location", "pair.agent.u_location"),)
    assert meta["pair"].identified == pair and meta["pair.world"].identified == pair
    assert meta["pair.agent"].identified == ()
    with pytest.raises(ValueError, match="key it by 'pair.agent.u_location'"):
        outer.act(KEY, op, outer.initial_priors(op), legal={"pair.world.u_location": [1] * 5})


# -------------------------------------------------------------------------------- refusals


def test_a_driver_that_is_neither_a_node_nor_a_state_is_refused_naming_both():
    (agent, ap), (world, wp) = agent_and_world()
    subs, ps = {"agent": agent, "world": world}, {"agent": ap, "world": wp}
    with pytest.raises(ValueError, match="neither a control node nor a state factor") as err:
        merge(subs, ps, [ActionControl("agent.u_locaiton", "world.u_location")])
    assert "agent.u_location" in str(err.value) and "'agent.location'" in str(err.value)
    with pytest.raises(ValueError, match="pass 'world.u_location', the control node of"):
        merge(subs, ps, [ActionControl("agent.u_location", "world.location")])
    with pytest.raises(ValueError, match="pairs 'agent.u_location' with itself"):
        merge(subs, ps, [ActionControl("agent.u_location", "agent.u_location")])
    with pytest.raises(ValueError, match=r"already merged \(with 'agent.u_location'\)"):
        merge(subs, ps, [SAME_STEP, SAME_STEP])
    with pytest.raises(ValueError, match="pairs 'agent.u_location' with itself"):
        merge(subs, ps, [SAME_STEP, ActionControl("world.u_location", "agent.u_location")])


def test_disagreeing_control_nodes_are_refused():
    (_, _), (world, wp) = agent_and_world()
    low, lp = _low(policy_len=world.policy_len)  # a 3-level node against the world's 5
    with pytest.raises(ValueError, match="'low.u_pos' has 3 level.s. but 'world.u_location' has 5"):
        merge(
            {"low": low, "world": world},
            {"low": lp, "world": wp},
            [ActionControl("low.u_pos", "world.u_location")],
        )


def test_a_mask_on_a_bound_away_node_names_the_node_it_became():
    composite, cp = same_step()
    beliefs, _ = composite.infer_states(cp, _obs(composite))
    mask = jnp.ones(5, bool)
    with pytest.raises(ValueError, match="act: legal names 'world.u_location'") as err:
        composite.act(KEY, cp, beliefs, legal={"world.u_location": mask})
    assert "key it by 'agent.u_location', the node it became" in str(err.value)
    with pytest.raises(ValueError, match="key it by 'agent.u_location'"):
        composite.infer_policies(cp, beliefs, legal={"world.u_location": mask})
    with pytest.raises(ValueError, match="agent_respond: legal names 'world.u_location'"):
        agent_respond(composite, cp, legal={"world.u_location": mask})
    with pytest.raises(ValueError, match="actions names 'world.a_location'"):
        composite.update_empirical_prior(
            cp, beliefs, {"agent.a_location": 0, "world.a_location": 0}
        )
    #  the driven mind's view has no node to mask: the choice is the driver's
    world_view, wvp = subsystem(composite, cp, "world")
    with pytest.raises(ValueError, match="which another subsystem chooses"):
        agent_respond(world_view, wvp, legal={"world.u_location": mask})

    #  a mask keyed by the driver's node applies as any node's mask does
    only = jnp.zeros(5, bool).at[2].set(True)
    action, _, _ = composite.act(
        KEY, cp, beliefs, legal={"agent.u_location": only}, selection="deterministic"
    )
    free, _, _ = composite.act(KEY, cp, beliefs, selection="deterministic")
    assert int(action["agent.a_location"]) == 2 and int(free["agent.a_location"]) == CUE


def test_a_mask_on_a_relayed_node_names_the_relay():
    composite, cp = merged()
    beliefs, _ = composite.infer_states(cp, _obs(composite))
    with pytest.raises(ValueError, match="identified with the state 'agent.location'") as err:
        composite.act(KEY, cp, beliefs, legal={"world.u_location": jnp.ones(5, bool)})
    assert "constrain the choice that moves 'agent.location'" in str(err.value)
    world_view, wvp = subsystem(composite, cp, "world")
    with pytest.raises(ValueError, match="identified with the state 'agent.location'"):
        agent_respond(world_view, wvp, legal={"world.u_location": jnp.ones(5, bool)})


def test_a_target_on_a_bound_away_node_is_refused_naming_the_drivers_node():
    (agent, ap), (world, wp) = agent_and_world()
    target = Categorical(logits=jnp.zeros(5))
    wp = dataclasses.replace(wp, state_targets={"u_location": target})
    with pytest.raises(ValueError, match="declare the target on 'agent.u_location'"):
        merge(
            {"agent": agent, "world": world},
            {"agent": ap, "world": wp},
            [SAME_STEP, *OBSERVATION_BINDINGS],
        )
    with pytest.raises(ValueError, match="merged with the state 'agent.location'"):
        merge(
            {"agent": agent, "world": world},
            {"agent": ap, "world": wp},
            [DRIVE, *OBSERVATION_BINDINGS],
        )
    #  a target on the driver's node travels with it, a risk leaf under its name
    ap = dataclasses.replace(ap, state_targets={"u_location": target})
    composite, cp = merge(
        {"agent": agent, "world": world},
        {"agent": ap, "world": dataclasses.replace(wp, state_targets={})},
        [SAME_STEP, *OBSERVATION_BINDINGS],
    )
    _, tree = composite.infer_policies(cp, composite.infer_states(cp, _obs(composite))[0])
    assert set(tree["risk"]) == {"agent.u_location"}


# --------------------------------------------------------------------- the boundary prior


@pytest.mark.parametrize(
    "prior", [CUE, Categorical(logits=jnp.asarray([0.0, 1.0, 2.0, 0.5, -1.0]))]
)
def test_a_boundary_prior_names_a_node_and_terminates_its_reader(prior):
    (_, _), (_, wp) = agent_and_world()
    composite, cp = merged(
        bindings=[*OBSERVATION_BINDINGS, BoundaryPrior("world.u_location", prior)]
    )
    assert composite.names.controls == ("agent.u_location",)
    declared = wp.transition["location"]
    if isinstance(prior, int):
        want = jnp.take(declared.log_table, CUE, axis=-1)[..., None]
    else:
        mixed = jnp.einsum("...u,u->...", jnp.exp(declared.log_probs), prior.probs)
        want = jnp.log(mixed)[..., None]
    got = cp.transition["world.location"]
    assert isinstance(got, ConditionalCategorical) and got.parent_ndims == 2
    assert jnp.array_equal(got.log_table, want)
    with pytest.raises(ValueError, match="pass 'world.u_location', the control node of"):
        merged(bindings=[*OBSERVATION_BINDINGS, BoundaryPrior("world.location", prior)])


# ------------------------------------------------------------------------- the views


def test_the_driven_view_has_no_node_and_the_drivers_view_keeps_it():
    composite, cp = same_step()
    (_, _), (_, wp) = agent_and_world()

    world, wvp = subsystem(composite, cp, "world")
    assert world.names.controls == () and world.names.states == ("world.location", "world.context")
    table = np.exp(np.asarray(wp.transition["location"].log_probs))
    got = np.exp(np.asarray(wvp.transition["world.location"].log_probs))[..., 0]
    np.testing.assert_allclose(got, table.mean(axis=-1), rtol=1e-6)  # a uniform partner prior
    pinned, pvp = subsystem(composite, cp, "world", partner_priors={"agent.u_location": CUE})
    assert jnp.array_equal(
        pvp.transition["world.location"].log_table,
        jnp.take(cp.transition["world.location"].log_table, CUE, axis=-1)[..., None],
    )
    with pytest.raises(ValueError, match="which are not control nodes the view of 'world' reads"):
        subsystem(composite, cp, "world", partner_priors={"agent.location": CUE})

    agent, avp = subsystem(composite, cp, "agent")
    assert agent.names.controls == ("agent.u_location",)
    assert agent.B_control_deps == ((0,), (), (0,), ())  # the world's factor reads it at once
    beliefs = _mixed_beliefs(agent, avp)
    pushed = agent.update_empirical_prior(avp, beliefs, {"agent.a_location": CUE})
    want = table[:, :, CUE] @ np.asarray(beliefs["world.location"].probs)
    np.testing.assert_allclose(
        np.asarray(pushed["world.location"].probs), want, rtol=1e-6, atol=1e-12
    )


# --------------------------------------------------------------------------- continuous


def _lg_mind(name: str, G, dim: int = 1):
    """One linear-Gaussian factor read through ``G u`` by its own continuous node."""
    system = System(
        B_deps=((),),
        controlled=(True,),
        factor_names=(name,),
        policy_len=1,
        planner="gradient",
        epistemic=False,
    )
    params = SystemParams(
        transition={name: _lg(jnp.zeros((1, 2)), G=G)},
        initial_state_prior={name: _gauss()},
        control={f"u_{name}": _box(dim)},
    )
    return system, params


def test_a_continuous_node_drives_another_minds_g_u_at_the_same_step():
    a, ap = _lg_mind("x", G2)
    G_y = jnp.asarray([[0.3], [-0.2]])
    b, bp = _lg_mind("y", G_y)
    composite, cp = merge({"a": a, "b": b}, {"a": ap, "b": bp}, [ActionControl("a.u_x", "b.u_y")])
    assert composite.names.controls == ("a.u_x",) and set(cp.control) == {"a.u_x"}
    beliefs = composite.initial_priors(cp)
    u = jnp.asarray([0.7])
    pushed = composite.update_empirical_prior(cp, beliefs, {"a.a_x": u})
    for name, G in (("a.x", G2), ("b.y", G_y)):
        want = F @ beliefs[name].loc + G @ u  # both read a's value at the step it is chosen
        np.testing.assert_allclose(
            np.asarray(pushed[name].loc), np.asarray(want), rtol=1e-6, atol=1e-12
        )

    wide, wp = _lg_mind("y", jnp.asarray([[0.3, 0.1], [-0.2, 0.0]]), dim=2)
    with pytest.raises(ValueError, match="'a.u_x' has 1 dimension.s. but 'b.u_y' has 2"):
        merge({"a": a, "b": wide}, {"a": ap, "b": wp}, [ActionControl("a.u_x", "b.u_y")])
    with pytest.raises(NotImplementedError, match="bind from 'a.u_x' for a same-step read"):
        merge({"a": a, "b": b}, {"a": ap, "b": bp}, [ActionControl("a.x", "b.u_y")])


# ------------------------------------------------------------- order and combinations


def _three(**extra):
    """The agent and two copies of the world, each world with its own control node."""
    (agent, ap), (world, wp) = agent_and_world()
    subs = {"agent": agent, "w1": world, "w2": world, **{k: v[0] for k, v in extra.items()}}
    ps = {"agent": ap, "w1": wp, "w2": wp, **{k: v[1] for k, v in extra.items()}}
    return subs, ps


def _assert_same_composite(one, other):
    (s1, p1), (s2, p2) = one, other
    assert s1 == s2
    got, want = jax.tree_util.tree_flatten(p1), jax.tree_util.tree_flatten(p2)
    assert got[1] == want[1]
    for a, b in zip(got[0], want[0], strict=True):
        assert jnp.array_equal(a, b)


def _refusal(subs, ps, bindings) -> str:
    with pytest.raises(ValueError) as err:
        merge(subs, ps, bindings)
    return str(err.value)


def test_a_chain_of_same_step_bindings_is_one_node_in_either_order():
    subs, ps = _three()
    first = ActionControl("agent.u_location", "w1.u_location")
    second = ActionControl("w1.u_location", "w2.u_location")
    forward = merge(subs, ps, [first, second])
    _assert_same_composite(forward, merge(subs, ps, [second, first]))
    composite, cp = forward
    assert composite.names.controls == ("agent.u_location",)
    meta = dict(composite.subsystems)
    assert meta["w1"].identified == (("w1.u_location", "agent.u_location"),)
    assert meta["w2"].identified == (("w2.u_location", "agent.u_location"),)
    #  A's decision at t moves both worlds at t
    (_, _), (_, wp) = agent_and_world()
    table = np.exp(np.asarray(wp.transition["location"].log_probs))
    beliefs = dict(composite.initial_priors(cp))
    beliefs["w2.location"] = Categorical.from_probs(jnp.asarray([0.5, 0.2, 0.1, 0.1, 0.1]))
    pushed = composite.update_empirical_prior(cp, beliefs, {"agent.a_location": CUE})
    for name in ("w1.location", "w2.location"):
        want = table[:, :, CUE] @ np.asarray(beliefs[name].probs)
        np.testing.assert_allclose(np.asarray(pushed[name].probs), want, rtol=1e-6, atol=1e-12)


def test_a_boundary_prior_and_a_same_step_binding_agree_in_either_order():
    (agent, ap), (world, wp) = agent_and_world()
    subs, ps = {"agent": agent, "world": world}, {"agent": ap, "world": wp}
    prior = BoundaryPrior("agent.u_location", CUE)
    forward = merge(subs, ps, [prior, SAME_STEP, *OBSERVATION_BINDINGS])
    _assert_same_composite(forward, merge(subs, ps, [SAME_STEP, prior, *OBSERVATION_BINDINGS]))
    composite, cp = forward
    assert composite.names.controls == ()
    declared = wp.transition["location"]
    want = jnp.take(declared.log_table, CUE, axis=-1)[..., None]
    assert jnp.array_equal(cp.transition["world.location"].log_table, want)
    #  the identified node ends at a terminated one: the key is dropped, not re-keyed
    beliefs = composite.initial_priors(cp)
    with pytest.raises(ValueError, match="which a BoundaryPrior terminated") as err:
        composite.update_empirical_prior(cp, beliefs, {"world.a_location": 0})
    assert "drop the key" in str(err.value) and "another subsystem" not in str(err.value)


def test_a_relay_into_a_node_a_same_step_binding_shares_is_refused_alike_in_either_order():
    (x, xp), _ = agent_and_world()
    subs, ps = _three(x=(x, xp))
    relay = ActionControl("x.location", "agent.u_location")
    node = ActionControl("agent.u_location", "w1.u_location")
    forward = _refusal(subs, ps, [node, relay])
    assert forward == _refusal(subs, ps, [relay, node])
    assert "['agent.location', 'w1.location'] read" in forward
    assert "made one with ['w1.u_location'] by a same-step ActionControl" in forward
    assert "bind from 'x.u_location' for a same-step read" in forward


def test_a_relay_into_a_node_its_component_shares_names_the_wiring():
    from tests.test_shared_controls import shared_model
    from tests.test_subsystem import _binary

    team, tp = shared_model(policy_len=1)
    z, zp = _binary("z", controlled=True)
    message = _refusal(
        {"z": z, "team": team}, {"z": zp, "team": tp}, [ActionControl("z.z", "team.u_help")]
    )
    assert "['team.help', 'team.who'] read (shared within its component)" in (message)
    assert "bind from 'z.u_z' for a same-step read" in message


def test_a_mix_of_a_node_driver_and_a_relay():
    """One decision drives one world at once and the other a tick later."""
    subs, ps = _three()
    composite, cp = merge(
        subs,
        ps,
        [
            ActionControl("agent.u_location", "w1.u_location"),
            ActionControl("agent.location", "w2.u_location"),
        ],
    )
    assert composite.names.controls == ("agent.u_location",)
    beliefs = {
        **composite.initial_priors(cp),
        **{f"{n}.location": _point(CENTRE) for n in ("agent", "w1", "w2")},
    }
    move = {"agent.a_location": CUE}
    one = composite.update_empirical_prior(cp, beliefs, move)
    assert float(one["w1.location"].probs[CUE]) > 0.999
    assert float(one["w2.location"].probs[CENTRE]) > 0.999
    two = composite.update_empirical_prior(cp, one, move)
    assert float(two["w2.location"].probs[CUE]) > 0.999


def test_an_outer_merge_naming_an_inner_bound_away_node_says_what_it_became():
    inner, ip = same_step()
    solo, sp = agent_and_world()[0]
    subs, ps = {"pair": inner, "solo": solo}, {"pair": ip, "solo": sp}
    became = re.escape("a control node merge() identified with 'pair.agent.u_location'")
    with pytest.raises(ValueError, match=f"BoundaryPrior names 'pair.world.u_location', {became}"):
        merge(subs, ps, [BoundaryPrior("pair.world.u_location", 0)])
    with pytest.raises(ValueError, match=f"ActionControl names 'pair.world.u_location', {became}"):
        merge(subs, ps, [ActionControl("pair.world.u_location", "solo.u_location")])
    with pytest.raises(ValueError, match=f"ActionControl names 'pair.world.u_location', {became}"):
        merge(subs, ps, [ActionControl("solo.u_location", "pair.world.u_location")])


def test_a_discrete_state_relayed_into_a_continuous_node_is_a_kind_mismatch():
    (agent, ap), _ = agent_and_world()
    b, bp = _lg_mind("y", jnp.asarray([[0.3], [-0.2]]))
    subs, ps = {"agent": agent, "b": b}, {"agent": ap, "b": bp}
    kind = "node kind mismatch: driver 'agent.location' is a discrete state"
    with pytest.raises(ValueError, match=kind):
        merge(subs, ps, [ActionControl("agent.location", "b.u_y")])


def test_the_driven_view_of_a_continuous_pair_names_the_node():
    a, ap = _lg_mind("x", G2)
    b, bp = _lg_mind("y", jnp.asarray([[0.3], [-0.2]]))
    composite, cp = merge({"a": a, "b": b}, {"a": ap, "b": bp}, [ActionControl("a.u_x", "b.u_y")])
    with pytest.raises(NotImplementedError, match=r"continuous control node \['a.u_x'\]"):
        subsystem(composite, cp, "b")
