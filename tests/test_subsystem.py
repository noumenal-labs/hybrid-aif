"""``subsystem()``: the subjective view (design ``docs/design/nesting-systems.md``).

Requirement 7 — *faces survive flattening* — is the claim these tests pin: after
composition it must remain possible to run ``infer_states``/``act`` **for a
sub-System**, its factor subset plus its blanket, against the rest-as-env. The
view is a plain ``(System, SystemParams)`` pair, so "it works" means every face
works, and the interesting content is in what crosses the boundary:

- a component nobody bound gives back *itself* (with namespaced keys) — the
  fidelity floor, checked bitwise on ``q_pi``;
- a bound component gains **boundary stubs**: the neighbours it must model, with
  another mind's control node carried as exogenous dynamics under a declared prior;
- what does *not* cross: other minds' designations, epistemic keys, and beliefs
  (there are none to cross — a System holds no belief state at all).

The zoo pair is ``tests.test_merge``'s: a T-maze agent and the same model as
the world it acts on.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Categorical, ConditionalCategorical
from hybrid_aif.composition.merge import (
    ActionControl,
    ObservationLink,
    TargetEmission,
    merge,
    subsystem,
    subsystem_factors,
)
from hybrid_aif.reference.envs.tmaze import CENTRE, CUE
from hybrid_aif.system import System, SystemParams
from tests.test_merge import (
    DRIVE,
    OBSERVATION_BINDINGS,
    _obs,
    agent_and_world,
    merged,
)
from tests.test_system import tmaze_system

KEY = jax.random.PRNGKey(0)


def unmerged_pair():
    """The zoo pair merged with *no* bindings: two components side by side."""
    (agent, ap), (world, wp) = agent_and_world()
    return merge({"agent": agent, "world": world}, {"agent": ap, "world": wp}, [])


# ------------------------------------------------------------------- view fidelity


def test_an_unbound_components_view_is_the_component_itself():
    """The fidelity floor: with nothing bound there is no boundary to model, so the
    view is the original un-merged System under namespaced keys — same structure,
    same arrays, and (the check that matters) the same ``q_pi`` bitwise."""
    composite, cp = unmerged_pair()
    view, vp = subsystem(composite, cp, "agent")
    agent, ap = tmaze_system()

    assert subsystem_factors(composite, "agent") == (0, 1)  # no stubs at all
    assert view.names.states == ("agent.location", "agent.context")
    assert view.modality_names == ("agent.location_obs", "agent.reward", "agent.cue")
    for field in ("A_deps", "B_deps", "controlled", "policy_len", "gamma"):
        assert getattr(view, field) == getattr(agent, field), field
    assert view.names.controls == ("agent.u_location",) and agent.names.controls == ("u_location",)
    assert view._control_levels(vp) == agent._control_levels(ap)
    assert set(vp.obs_targets) == {"agent.reward"}  # the world's designations stayed home

    beliefs_v, _ = view.infer_states(vp, _obs(view))
    beliefs_a, _ = agent.infer_states(ap, _obs(agent))
    assert set(beliefs_v) == {f"agent.{k}" for k in beliefs_a}
    for k, b in beliefs_a.items():
        assert jnp.array_equal(beliefs_v[f"agent.{k}"].probs, b.probs)
    q_v, _ = view.infer_policies(vp, beliefs_v)
    q_a, _ = agent.infer_policies(ap, beliefs_a)
    assert jnp.array_equal(q_v.logits, q_a.logits)


def test_the_bound_agents_view_is_the_composite_itself():
    """The exact relationship where the stubs *do* bite, pinned rather than waved at.

    In the zoo pair every other factor is a direct parent of something of the
    agent's (its three modalities read the world), and the world's own control node
    was merged away — so the agent's view carries all four factors with the
    composite's own declarations, i.e. it **is** the composite minus the metadata.
    The stubs therefore change the policy enumeration not at all (25 policies, the
    agent's node alone). What differs from the *un-merged* T-maze agent is not the
    view but the model it is a view of: the merge put a one-tick relay between
    the agent's location and the world's, so the two are different agents by
    construction — which is why the many-minds path claims no equality with the
    one-mind path.
    """
    composite, cp = merged()
    view, vp = subsystem(composite, cp, "agent")

    fields = [f.name for f in dataclasses.fields(System) if f.name != "subsystems"]
    for name in fields:
        assert getattr(view, name) == getattr(composite, name), name
    got, want = jax.tree_util.tree_flatten(vp), jax.tree_util.tree_flatten(cp)
    assert got[1] == want[1]
    for a, b in zip(got[0], want[0], strict=True):
        assert a is b  # parameters are shared by reference; only beliefs are private
    assert view.policies(vp).shape[0] == 25

    agent, ap = tmaze_system()
    q_v, _ = view.infer_policies(vp, view.infer_states(vp, _obs(view))[0])
    q_a, _ = agent.infer_policies(ap, agent.infer_states(ap, _obs(agent))[0])
    assert q_v.logits.shape == q_a.logits.shape
    assert not jnp.allclose(q_v.logits, q_a.logits)  # the relay, not a bug


# ------------------------------------------------------------------ boundary stubs


def test_a_partners_control_node_becomes_an_exogenous_stub():
    """The world's view of the agent: the partner's action is *mode dynamics with a
    declared prior* (the switching demo's ``ACTION_MODE_LOGW`` pattern), uniform by
    default — never a control node of ours."""
    composite, cp = merged()
    assert subsystem_factors(composite, "world") == (0, 2, 3)
    view, vp = subsystem(composite, cp, "world")
    assert view.names.states == ("agent.location", "world.location", "world.context")
    assert view.controlled == (False, False, False)
    assert view.names.controls == ()  # no control node: every one is somebody else's
    assert view.modality_names == ()  # the world's sensors were stripped; it is blind
    assert view.B_deps == ((0,), (1, 0), (2,))  # world.location still reads the stub

    # the stub's transition is the agent's own, averaged over a uniform partner prior
    marginal = np.asarray(jnp.exp(vp.transition["agent.location"].log_probs))[:, CENTRE, 0]
    assert np.isclose(marginal.sum(), 1.0, atol=1e-5)
    assert np.isclose(marginal[CUE], 0.2, atol=1e-4)  # 1 of 5 moves reaches the cue
    assert np.isclose(marginal[CENTRE], 0.6, atol=1e-4)  # three moves are no-ops


def test_partner_priors_declare_what_this_mind_expects_of_the_other():
    """Requirement 4's two extremes, seen from inside one mind: a delta pins the
    partner's move, a distribution marginalizes over it — different beliefs about
    the same partner, and functionally different views."""
    composite, cp = merged()
    view, vp = subsystem(composite, cp, "world", partner_priors={"agent.u_location": CUE})
    pushed = view.update_empirical_prior(
        vp, view.initial_priors(vp), view.from_positional("actions", jnp.zeros(0, jnp.int32))
    )
    assert np.isclose(float(pushed[view.names.states[0]].probs[CUE]), 1.0, atol=1e-4)

    one_hot = jnp.zeros(5).at[CUE].set(1.0)
    tilted, tp = subsystem(composite, cp, "world", partner_priors={"agent.u_location": one_hot})
    assert tilted.names.controls == ()  # the partner's node is not ours
    # the delta and the one-hot distribution are the same view, as they should be:
    # slicing the control axis and averaging it against a point mass agree
    stub = "agent.location"
    assert jnp.allclose(
        jnp.exp(tp.transition[stub].log_probs), jnp.exp(vp.transition[stub].log_probs)
    )

    uniform, up = subsystem(composite, cp, "world")
    assert not jnp.allclose(
        jnp.exp(up.transition[stub].log_probs), jnp.exp(vp.transition[stub].log_probs)
    )


def test_a_target_emission_parent_is_a_stub_in_the_low_minds_view():
    """N1 from below: the high System's factor emitting our terminal prior is a
    boundary stub, and the ``H_deps`` declaration survives into the view."""
    (agent, ap), (world, wp) = agent_and_world()
    target = ConditionalCategorical(
        log_table=jnp.log(jnp.asarray([[0.9, 0.8, 0.7, 0.6, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])),
        parent_ndims=1,
    )
    wp = dataclasses.replace(wp, terminal_targets={"context": target})
    composite, cp = merge(
        {"agent": agent, "world": world},
        {"agent": ap, "world": wp},
        [*OBSERVATION_BINDINGS, TargetEmission("world.context", ("agent.location",))],
    )
    view, vp = subsystem(composite, cp, "world")
    assert view.names.states == ("agent.location", "world.location", "world.context")
    assert view.H_deps == ((), (), (0,), ())  # the world's free control node's entry last
    assert set(vp.terminal_targets) == {"world.context"}
    _, tree = view.infer_policies(vp, view.initial_priors(vp))
    assert "world.context" in tree["terminal_risk"]


def _binary(name, controlled, modality=False):
    """One 2-level factor, its next value the action (or its own value, uncontrolled)."""
    table = jnp.log(jnp.asarray([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]))
    system = System(
        A_deps=((0,),) if modality else (),
        B_deps=((0,),),
        controlled=(controlled,),
        policy_len=1,
        epistemic=False,
        modality_names=("see",) if modality else (),
        factor_names=(name,),
    )
    params = SystemParams(
        likelihood={"see": ConditionalCategorical(log_table=jnp.log(jnp.eye(2)), parent_ndims=1)}
        if modality
        else {},
        transition={
            name: ConditionalCategorical(
                log_table=table if controlled else jnp.log(jnp.eye(2))[..., None],
                parent_ndims=2,
            ),
        },
        initial_state_prior={name: Categorical(logits=jnp.zeros(2))},
    )
    return system, params


def test_the_stub_set_closes_over_what_the_stubs_own_dynamics_need():
    """A stub is carried verbatim, so its own parents come with it — the closure
    that makes "direct neighbours" the *minimal well-formed* set rather than a
    truncation that would silently change a partner's dynamics.

    Chain ``x <- y <- z``: the mind behind ``x`` observes ``y`` (a direct
    neighbour), and ``y``'s declared transition reads ``z`` — so ``z`` enters the
    view through the closure alone, as an exogenous stub under a uniform prior.
    """
    x, xp = _binary("x", controlled=False, modality=True)
    y, yp = _binary("y", controlled=True)
    z, zp = _binary("z", controlled=True)
    composite, cp = merge(
        {"x": x, "y": y, "z": z},
        {"x": xp, "y": yp, "z": zp},
        [
            ActionControl(driver="z.z", control="y.u_y"),
            ObservationLink(modality="x.see", factors=("y.y",)),
        ],
    )
    assert subsystem_factors(composite, "x") == (0, 1, 2)  # own, direct, closure
    view, vp = subsystem(composite, cp, "x")
    assert view.factor_names == ("x.x", "y.y", "z.z")
    assert view.B_deps == ((0,), (1, 2), (2,))
    assert view.controlled == (False, False, False)  # z's node is somebody else's
    view._validate(vp)  # well-formed: every parent index resolves inside the view
    assert subsystem_factors(composite, "z") == (2,)  # nothing flows back the other way


# ------------------------------------------------------------------- every face runs


@pytest.mark.parametrize("name", ["agent", "world"])
def test_every_face_runs_standalone_on_the_view(name):
    """Requirement 7 literally: the sub-System runs its own faces against its own
    ``as_env``, with no composite in sight.

    The world is designated here so that it has *something* to score: its own
    sensors were stripped and its control node was merged, so without a preference its term
    tree is empty and ``infer_policies`` raises the library's standing "no included
    leaves" error — a property of a System with nothing to plan for, not of the view.
    """
    composite, cp = merged()
    cp = dataclasses.replace(
        cp, state_targets={"world.context": Categorical(logits=jnp.asarray([1.0, -1.0]))}
    )
    view, vp = subsystem(composite, cp, name)
    env, ep = view.as_env(vp)
    obs, state = env.reset(KEY, ep)
    assert len(state) == len(view.names.states) and len(obs) == len(view.A_deps)

    beliefs, _ = view.infer_states(vp, obs)
    q_pi, tree = view.infer_policies(vp, beliefs)
    action, _, _ = view.act(KEY, vp, beliefs)
    pushed = view.update_empirical_prior(vp, beliefs, action)
    assert view.to_positional("actions", action).shape == (len(view.names.controls),)
    assert np.isclose(float(q_pi.probs.sum()), 1.0, atol=1e-5)
    for b in [*beliefs.values(), *pushed.values()]:
        assert np.isclose(float(b.probs.sum()), 1.0, atol=1e-4)
        assert bool(jnp.all(jnp.isfinite(b.probs)))
    assert tree is not None
    _, state = env.step(KEY, state, action, ep)
    assert all(int(s) >= 0 for s in state.values())


def test_the_agents_view_still_seeks_the_cue():
    """Sanity that the view is the *same model*, not just a well-formed one."""
    composite, cp = merged(policy_len=3)
    view, vp = subsystem(composite, cp, "agent")
    beliefs, _ = view.infer_states(vp, _obs(view))
    action, _, _ = view.act(KEY, vp, beliefs, selection="deterministic")
    assert int(action["agent.a_location"]) == CUE


# ------------------------------------------------------------ what does not cross


def test_designations_and_epistemic_keys_are_restricted_to_the_view():
    (agent, ap), (world, wp) = agent_and_world()
    ap = dataclasses.replace(ap, state_targets={"context": Categorical(logits=jnp.zeros(2))})
    wp = dataclasses.replace(wp, terminal_targets={"context": Categorical(logits=jnp.zeros(2))})
    composite, cp = merge(
        {
            "agent": dataclasses.replace(agent, epistemic=("location", "obs:reward")),
            "world": dataclasses.replace(world, epistemic=("location",)),
        },
        {"agent": ap, "world": wp},
        [DRIVE, *OBSERVATION_BINDINGS],
    )
    assert set(composite.epistemic) == {"agent.location", "obs:agent.reward", "world.location"}

    view, vp = subsystem(composite, cp, "world")
    assert set(view.epistemic) == {"world.location"}  # the agent's keys are not ours...
    assert set(vp.state_targets) == set()  # ... nor its designations
    assert set(vp.terminal_targets) == {"world.context"}

    agent_view, avp = subsystem(composite, cp, "agent")
    assert set(agent_view.epistemic) == {"agent.location", "obs:agent.reward"}
    assert set(avp.state_targets) == {"agent.context"}


def test_epistemic_bools_pass_through():
    composite, cp = merged()
    assert subsystem(composite, cp, "agent")[0].epistemic is True
    (agent, ap), (world, wp) = agent_and_world()
    off, op = merge(
        {
            "agent": dataclasses.replace(agent, epistemic=False),
            "world": dataclasses.replace(world, epistemic=False),
        },
        {"agent": ap, "world": wp},
        [DRIVE, *OBSERVATION_BINDINGS],
    )
    assert subsystem(off, op, "agent")[0].epistemic is False


def test_the_view_records_which_factors_are_its_own():
    """The stub boundary survives as metadata: a stub belongs to no entry."""
    composite, cp = merged()
    view, _ = subsystem(composite, cp, "agent")
    meta = dict(view.subsystems)
    assert set(meta) == {"agent"}
    assert meta["agent"].factors == (0, 1)  # world.location/context are stubs
    assert meta["agent"].controls == ("agent.u_location",)
    world_view, _ = subsystem(composite, cp, "world")
    assert dict(world_view.subsystems)["world"].relayed == (("world.u_location", "agent.location"),)


def test_a_view_of_a_nested_composite_keeps_its_children():
    inner, ip = merged()
    outer, op = merge({"pair": inner, "sup": _mode()[0]}, {"pair": ip, "sup": _mode()[1]}, [])
    view, vp = subsystem(outer, op, "pair")
    assert view.factor_names[0] == "pair.agent.location"
    assert set(dict(view.subsystems)) == {"pair", "pair.agent", "pair.world"}
    view._validate(vp)


def _mode():
    from tests.test_h_deps import _high

    return _high(policy_len=2)


# ------------------------------------------------------------------- loud failures


def test_views_of_unknown_or_unrecorded_components_raise():
    composite, cp = merged()
    with pytest.raises(ValueError, match="unknown subsystem 'agnet'"):
        subsystem(composite, cp, "agnet")
    plain, pp = tmaze_system()
    with pytest.raises(ValueError, match="carries no subsystems metadata"):
        subsystem(plain, pp, "agent")


def test_partner_priors_must_name_a_partner_node():
    composite, cp = merged()
    with pytest.raises(ValueError, match="which are not control nodes the view of 'world' reads"):
        subsystem(composite, cp, "world", partner_priors={"world.location": 0})
    with pytest.raises(ValueError, match="not one of its 5 control level"):
        subsystem(composite, cp, "world", partner_priors={"agent.u_location": 9})
    with pytest.raises(ValueError, match="one probability per level"):
        subsystem(composite, cp, "world", partner_priors={"agent.u_location": jnp.zeros(3)})


def test_a_joint_habit_prior_does_not_project_onto_one_mind():
    (agent, ap), (world, wp) = agent_and_world(policy_len=1)
    composite, cp = merge(
        {"agent": agent, "world": world},
        {
            "agent": dataclasses.replace(ap, policy_prior=jnp.zeros(5)),
            "world": dataclasses.replace(wp, policy_prior=jnp.arange(5.0)),
        },
        OBSERVATION_BINDINGS,  # both nodes chosen: a genuinely joint policy set
    )
    assert cp.policy_prior.shape == (25,)
    with pytest.raises(NotImplementedError, match="chooses only part of its control nodes"):
        subsystem(composite, cp, "agent")


def test_a_habit_prior_survives_when_the_view_keeps_every_control_node():
    composite, cp = unmerged_pair()
    cp = dataclasses.replace(cp, policy_prior=jnp.arange(25.0))
    with pytest.raises(NotImplementedError):
        subsystem(composite, cp, "agent")  # the world's node is the agent's no business
    solo, sp = merge({"agent": tmaze_system()[0]}, {"agent": tmaze_system()[1]}, [])
    sp = dataclasses.replace(sp, policy_prior=jnp.arange(25.0))
    view, vp = subsystem(solo, sp, "agent")
    assert jnp.array_equal(vp.policy_prior, sp.policy_prior)
