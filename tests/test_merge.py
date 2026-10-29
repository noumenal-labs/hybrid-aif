"""``merge()``: the nesting builder (design ``docs/design/nesting-systems.md``).

A merge of Systems *is* a System (requirement 2), so the acceptance criterion
is an identity, not a behavior: the one-mind composite must be **the same
declaration** as the flattening written out longhand — every static field, every
params leaf, and every face (``infer_states``, ``infer_policies``, ``act``,
``as_env`` rollout). ``test_one_mind_composite_is_the_hand_flattened_declaration``
is that check; the rest of the file pins the pieces it is made of:

- the namespacing surface (names, designation keys, the ``epistemic`` selector
  including wildcards, ``learn``'s rate keys);
- the three binding kinds plus boundary-prior termination, with their node typing;
- ``policy_prior`` composition over the joint policy set;
- the ``subsystems`` metadata (what ``subsystem()`` will consume in chunk 2), and
  its closure under a second merge;
- loud validation everywhere a binding can be wrong.

The zoo pair is the T-maze twice over: an *agent* (``tests.test_system``'s
``tmaze_system``) and a *world* — the same generative model with its own sensors
stripped, i.e. the process the agent acts on. That is the canonical agent×env
merge: the agent's location factor is identified, through the relay, with the world's
location control node (binding a), and the agent's three modalities read the world's
factors (binding b).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Categorical, ConditionalCategorical
from hybrid_aif.composition.merge import (
    ActionControl,
    BoundaryPrior,
    ObservationLink,
    TargetEmission,
    merge,
    namespaced_key,
    resolved_epistemic,
)
from hybrid_aif.reference.envs.tmaze import CENTRE, CUE
from hybrid_aif.system import System, SystemParams
from tests.test_h_deps import _high, _low
from tests.test_learn import _sense_system
from tests.test_system import tmaze_system

KEY = jax.random.PRNGKey(0)
OBS = (jnp.asarray(CENTRE), jnp.asarray(0), jnp.asarray(0))


def _obs(system, obs=OBS):
    """The positional tick ``obs`` keyed by ``system``'s modality names."""
    return system.from_positional("observations", obs)


OBSERVATION_BINDINGS = [
    ObservationLink(modality="agent.location_obs", factors=("world.location",)),
    ObservationLink(modality="agent.reward", factors=("world.location", "world.context")),
    ObservationLink(modality="agent.cue", factors=("world.location", "world.context")),
]
DRIVE = ActionControl(driver="agent.location", control="world.u_location")


def agent_and_world(policy_len=2, **agent_kwargs):
    """The zoo pair: a T-maze agent and the same model as the world it acts on."""
    agent, ap = tmaze_system(policy_len=policy_len, **agent_kwargs)
    world = dataclasses.replace(agent, A_deps=(), modality_names=())
    wp = dataclasses.replace(ap, likelihood={}, obs_targets={})
    return (agent, ap), (world, wp)


def merged(policy_len=2, bindings=None, **agent_kwargs):
    (agent, ap), (world, wp) = agent_and_world(policy_len, **agent_kwargs)
    return merge(
        {"agent": agent, "world": world},
        {"agent": ap, "world": wp},
        [DRIVE, *OBSERVATION_BINDINGS] if bindings is None else bindings,
    )


def hand_flattened(policy_len=2):
    """The same model declared longhand, with no merge machinery involved.

    The only piece that is not a verbatim copy is the world's location
    transition: its control axis becomes an ordinary parent axis (the driver's
    levels), which in the child-first layout ``(child, *parents, control)`` is the
    identical table with a no-op control axis appended.
    """
    (agent, ap), (world, wp) = agent_and_world(policy_len)
    driven = wp.transition["location"]
    system = System(
        A_deps=((2,), (2, 3), (2, 3)),
        B_deps=((0,), (1,), (2, 0), (3,)),
        controlled=(True, False, False, False),
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
            "world.location": ConditionalCategorical(
                log_table=driven.log_table[..., None], parent_ndims=driven.parent_ndims + 1
            ),
            "world.context": wp.transition["context"],
        },
        initial_state_prior={
            "agent.location": ap.initial_state_prior["location"],
            "agent.context": ap.initial_state_prior["context"],
            "world.location": wp.initial_state_prior["location"],
            "world.context": wp.initial_state_prior["context"],
        },
        obs_targets={"agent.reward": ap.obs_targets["reward"]},
        state_targets={},
        terminal_targets={},
    )
    return system, params


# ------------------------------------------------------------------- acceptance


def test_one_mind_composite_is_the_hand_flattened_declaration():
    """The doc's §2(iv)-style check: identical declaration, identical faces.

    Bitwise throughout — the composite's arrays *are* the components' arrays (the
    only rewrite is an ``expand_dims`` on the bound factor's table), so no
    reassociation is involved and there is nothing to give a tolerance to.
    """
    composite, cp = merged()
    hand, hp = hand_flattened()

    fields = [f.name for f in dataclasses.fields(System) if f.name != "subsystems"]
    for name in fields:
        assert getattr(composite, name) == getattr(hand, name), name
    assert composite.subsystems is not None and hand.subsystems is None  # the only difference

    got, want = jax.tree_util.tree_flatten(cp), jax.tree_util.tree_flatten(hp)
    assert got[1] == want[1]  # same pytree structure, designation keys included
    for a, b in zip(got[0], want[0], strict=True):
        assert jnp.array_equal(a, b)

    beliefs_c, _ = composite.infer_states(cp, _obs(composite))
    beliefs_h, _ = hand.infer_states(hp, _obs(hand))
    assert set(beliefs_c) == set(beliefs_h)
    for name, b in beliefs_h.items():
        assert jnp.array_equal(beliefs_c[name].probs, b.probs)

    q_c, tree_c = composite.infer_policies(cp, beliefs_c)
    q_h, tree_h = hand.infer_policies(hp, beliefs_h)
    assert jnp.array_equal(q_c.logits, q_h.logits)
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda x, y: bool(jnp.array_equal(x, y)), tree_c, tree_h)
    )
    a_c, _, _ = composite.act(KEY, cp, beliefs_c)
    a_h, _, _ = hand.act(KEY, hp, beliefs_h)
    assert jnp.array_equal(
        composite.to_positional("actions", a_c), hand.to_positional("actions", a_h)
    )

    env_c, ep_c = composite.as_env(cp)
    env_h, ep_h = hand.as_env(hp)
    obs_c, state_c = env_c.reset(KEY, ep_c)
    obs_h, state_h = env_h.reset(KEY, ep_h)
    for _ in range(3):
        obs_c, state_c = env_c.step(KEY, state_c, a_c, ep_c)
        obs_h, state_h = env_h.step(KEY, state_h, a_h, ep_h)
    assert set(obs_c) == set(obs_h) and set(state_c) == set(state_h)
    assert all(jnp.array_equal(obs_c[k], obs_h[k]) for k in obs_h)
    assert all(jnp.array_equal(state_c[k], state_h[k]) for k in state_h)


def test_composite_still_solves_the_tmaze_through_the_boundary():
    """Sanity that the flattening is the *same model*, not just the same arrays:
    the agent still seeks the cue, now reading the world's factors."""
    composite, cp = merged(policy_len=3)
    beliefs, _ = composite.infer_states(cp, _obs(composite))
    action, _, _ = composite.act(KEY, cp, beliefs, selection="deterministic")
    assert int(action["agent.a_location"]) == CUE


# --------------------------------------------------------------------- bindings


def test_action_control_rewrites_the_control_node_as_a_parent():
    composite, cp = merged()
    assert composite.B_deps[2] == (2, 0)  # world.location now depends on agent.location
    assert composite.names.controls == ("agent.u_location",)
    assert int(composite.policies(cp)[..., 0].max()) + 1 == 5  # the agent's node has 5 levels
    assert composite.controlled == (True, False, False, False)
    assert cp.transition["world.location"].parent_ndims == 3  # own + driver + no-op control
    assert composite.policies(cp).shape[0] == 25  # only the agent's node is chosen


def test_action_control_relays_with_one_tick_lag():
    """Requirement 5, v1 synchronous: a ``B_deps`` parent is read at the previous
    step, so the driven factor follows one tick behind (documented in merge())."""
    composite, cp = merged()
    env, ep = composite.as_env(cp)
    _, state = env.reset(KEY, ep)
    move = composite.from_positional("actions", jnp.asarray([CUE], jnp.int32))
    _, state1 = env.step(KEY, state, move, ep)
    assert (int(state1["agent.location"]), int(state1["world.location"])) == (
        CUE,
        CENTRE,
    )  # agent moved, world has not
    _, state2 = env.step(KEY, state1, move, ep)
    assert int(state2["world.location"]) == CUE  # ... and now it has


def test_observation_link_points_a_modality_across_the_boundary():
    composite, cp = merged()
    assert composite.A_deps == ((2,), (2, 3), (2, 3))
    assert composite.modality_names == ("agent.location_obs", "agent.reward", "agent.cue")
    # the world contributes no modality at all: its sensors were stripped
    assert len(cp.likelihood) == 3


def test_target_emission_declares_H_deps_across_the_boundary():
    """Binding (c). The full N1 semantics live in ``test_h_deps.py``; here we pin
    that the binding writes the composite's ``H_deps`` and the metadata records it."""
    (agent, ap), (world, wp) = agent_and_world()
    target = ConditionalCategorical(
        log_table=jnp.log(jnp.asarray([[0.9, 0.8, 0.7, 0.6, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])),
        parent_ndims=1,
    )
    wp = dataclasses.replace(wp, terminal_targets={"context": target})
    composite, cp = merge(
        {"agent": agent, "world": world},
        {"agent": ap, "world": wp},
        [
            *OBSERVATION_BINDINGS,
            TargetEmission(target="world.context", parents=("agent.location",)),
        ],
    )
    #  the state factors' entries, then the two (free) control nodes'
    assert composite.H_deps == ((), (), (), (0,), (), ())
    assert "world.context" in cp.terminal_targets
    meta = dict(composite.subsystems)
    assert meta["world"].target_emissions == ((3, (0,)),)
    beliefs, _ = composite.infer_states(cp, _obs(composite))
    _, tree = composite.infer_policies(cp, beliefs)
    assert "world.context" in tree["terminal_risk"]


@pytest.mark.parametrize("prior", [CUE, Categorical(logits=jnp.zeros(5))])
def test_boundary_prior_terminates_a_control_node(prior):
    """Requirement 4: a delta *pins* the dynamics, a uniform *marginalizes* over
    them — different composites, both runnable with no external feed."""
    composite, cp = merged(
        bindings=[*OBSERVATION_BINDINGS, BoundaryPrior("world.u_location", prior)]
    )
    assert composite.controlled == (True, False, False, False)
    assert composite.names.controls == ("agent.u_location",)  # the terminated node is gone
    assert cp.transition["world.location"].parent_ndims == 2  # control axis retired, not re-read
    beliefs, _ = composite.infer_states(cp, _obs(composite))
    pushed = composite.update_empirical_prior(
        cp, beliefs, composite.from_positional("actions", jnp.zeros(1, jnp.int32))
    )
    probs = np.asarray(pushed["world.location"].probs)
    if isinstance(prior, int):
        assert np.argmax(probs) == CUE and probs[CUE] > 0.9  # pinned to the CUE move
    else:
        # marginalized: from CENTRE a uniform steerer reaches CUE and MIDDLE with 1/5
        # each and stays put on the three unconnected moves
        assert np.isclose(probs[CUE], 0.2, atol=1e-3) and int((probs > 0.05).sum()) == 3


def test_delta_and_uniform_boundary_priors_differ():
    delta, dp = merged(bindings=[*OBSERVATION_BINDINGS, BoundaryPrior("world.u_location", CUE)])
    flat, fp = merged(
        bindings=[*OBSERVATION_BINDINGS, BoundaryPrior("world.u_location", jnp.full(5, 0.2))]
    )
    beliefs_d, _ = delta.infer_states(dp, _obs(delta))
    beliefs_f, _ = flat.infer_states(fp, _obs(flat))
    q_d, _ = delta.infer_policies(dp, beliefs_d)
    q_f, _ = flat.infer_policies(fp, beliefs_f)
    assert not jnp.allclose(q_d.logits, q_f.logits)


# ------------------------------------------------------------------ namespacing


def test_namespaced_key_qualifies_the_name_segment_only():
    assert namespaced_key("arm", "pose") == "arm.pose"
    assert namespaced_key("arm", "obs:cue") == "obs:arm.cue"
    assert namespaced_key("arm", "likelihood:pose") == "likelihood:arm.pose"
    assert namespaced_key("arm", "likelihood:pose:noise") == "likelihood:arm.pose:noise"
    with pytest.raises(ValueError, match="wildcard"):
        namespaced_key("arm", "likelihood:*")


def test_names_and_designation_keys_are_namespaced():
    composite, cp = merged()
    assert composite.names.states == (
        "agent.location",
        "agent.context",
        "world.location",
        "world.context",
    )
    assert set(cp.obs_targets) == {"agent.reward"}
    (agent, ap), (world, wp) = agent_and_world()
    ap = dataclasses.replace(
        ap,
        state_targets={"context": Categorical(logits=jnp.asarray([1.0, -1.0]))},
        terminal_targets={"location": Categorical(logits=jnp.zeros(5))},
    )
    composite, cp = merge(
        {"agent": agent, "world": world}, {"agent": ap, "world": wp}, [*OBSERVATION_BINDINGS]
    )
    assert set(cp.state_targets) == {"agent.context"}
    assert set(cp.terminal_targets) == {"agent.location"}


def test_default_positional_names_are_namespaced_too():
    (agent, ap), (world, wp) = agent_and_world()
    anonymous = System(  # the agent declared afresh, with every name generated
        A_deps=agent.A_deps,
        B_deps=agent._table.B_deps,
        controlled=(True, False),
        policy_len=agent.policy_len,
        gamma=agent.gamma,
        epistemic=agent.epistemic,
    )
    #  the same tables, keyed by the generated names the anonymous declaration answers to
    anon_params = anonymous.from_positional("params", agent.to_positional("params", ap))
    anon_params = dataclasses.replace(
        anon_params, obs_targets={"modality_1": ap.obs_targets["reward"]}
    )
    composite, _ = merge(
        {"a": anonymous, "world": world},
        {"a": anon_params, "world": wp},
        [ActionControl(driver="a.factor_0", control="world.u_location")],
    )
    assert composite.factor_names[:2] == ("a.factor_0", "a.factor_1")
    assert composite.modality_names[:1] == ("a.modality_0",)


def test_epistemic_true_survives_and_mixed_selectors_resolve():
    composite, _ = merged()
    assert composite.epistemic is True  # both components mount everything
    (agent, ap), (world, wp) = agent_and_world()
    pragmatic = dataclasses.replace(world, epistemic=False)
    composite, _ = merge(
        {"agent": agent, "world": pragmatic},
        {"agent": ap, "world": wp},
        [DRIVE, *OBSERVATION_BINDINGS],
    )
    assert isinstance(composite.epistemic, tuple)
    assert set(composite.epistemic) == {
        "agent.location",
        "agent.context",
        "obs:agent.location_obs",
        "obs:agent.reward",
        "obs:agent.cue",
    }
    off, _ = merge(
        {"agent": dataclasses.replace(agent, epistemic=()), "world": pragmatic},
        {"agent": ap, "world": wp},
        [DRIVE, *OBSERVATION_BINDINGS],
    )
    assert off.epistemic is False


def test_wildcards_resolve_against_the_components_own_variables():
    """Resolve-then-union: a component's ``("*",)`` becomes *its* state variables,
    literally, at merge time — the strategy documented in ``resolved_epistemic``
    (the pattern grammar has no partial globs, so ``"a.*"`` is not expressible)."""
    (agent, ap), (world, wp) = agent_and_world()
    starred = dataclasses.replace(agent, epistemic=("*",))
    assert resolved_epistemic(starred, ap) == ("location", "context")
    pragmatic_world = dataclasses.replace(world, epistemic=())
    composite, cp = merge(
        {"agent": starred, "world": pragmatic_world},
        {"agent": ap, "world": wp},
        [DRIVE, *OBSERVATION_BINDINGS],
    )
    assert set(composite.epistemic) == {"agent.location", "agent.context"}
    beliefs, _ = composite.infer_states(cp, _obs(composite))
    _, tree = composite.infer_policies(cp, beliefs)
    # ... and the *world's* wildcard means the world's variables — which is visible
    # here because V6's "wherever the variable is involved" rule survives the merge:
    # after the observation bindings no channel observes the agent's own factors, so
    # its wildcard mounts nothing, while the world's mounts all three channels.
    assert "infogain" not in tree
    flipped, fp = merge(
        {"agent": dataclasses.replace(agent, epistemic=()), "world": world},
        {"agent": ap, "world": wp},
        [DRIVE, *OBSERVATION_BINDINGS],
    )
    assert set(flipped.epistemic) == {"world.location", "world.context"}
    _, tree = flipped.infer_policies(fp, flipped.infer_states(fp, _obs(flipped))[0])
    assert set(tree["infogain"]) == {"agent.location_obs", "agent.reward", "agent.cue"}


def test_learn_rate_keys_and_parameter_variables_are_namespaced():
    """The rate/``LearnState`` key space is the same one grammar, so it namespaces
    the same way: ``"likelihood:sense"`` becomes ``"likelihood:sensor.sense"``."""
    sensor, sp = _sense_system()
    high, hp = _high()
    composite, cp = merge({"sensor": sensor, "high": high}, {"sensor": sp, "high": hp}, [])
    assert "likelihood:sensor.sense" in composite._slot_declarations(cp)
    assert "likelihood:sensor.sense" in composite.epistemic
    obs = _obs(composite, (jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32)))
    beliefs = composite.initial_priors(cp)
    updated = composite.learn(cp, obs, beliefs, lr_by_slot={"likelihood:sensor.sense": 0.0})
    assert jnp.array_equal(
        updated.likelihood["sensor.sense"].concentration,
        cp.likelihood["sensor.sense"].concentration,
    )  # frozen by the namespaced key
    moved = composite.learn(cp, obs, beliefs, lr_by_slot={"likelihood:sensor.sense": 1.0})
    assert not jnp.array_equal(
        moved.likelihood["sensor.sense"].concentration, cp.likelihood["sensor.sense"].concentration
    )
    with pytest.raises(ValueError, match="unknown slot key"):
        composite.learn(cp, obs, beliefs, lr_by_slot={"likelihood:sense": 1.0})


def test_a_merged_control_node_stays_learnable_with_the_driver_as_its_parent():
    """The rewrite is an axis *re-reading*, so a learnable transition survives it: the
    Dirichlet counts are still per (child, parents) config — the driver's belief now
    weights the statistic where the action one-hot used to."""
    from hybrid_aif import ConditionalDirichlet

    (agent, ap), (world, wp) = agent_and_world(policy_len=1)
    counts = ConditionalDirichlet(
        concentration=jnp.exp(wp.transition["location"].log_probs) + 1.0, parent_ndims=2
    )
    wp = dataclasses.replace(wp, transition={**wp.transition, "location": counts})
    composite, cp = merge(
        {"agent": agent, "world": world},
        {"agent": ap, "world": wp},
        [DRIVE, *OBSERVATION_BINDINGS],
    )
    driven = cp.transition["world.location"]
    assert isinstance(driven, ConditionalDirichlet) and driven.parent_ndims == 3
    assert "transition:world.location" in composite._slot_declarations(cp)
    beliefs, _ = composite.infer_states(cp, _obs(composite))
    updated = composite.learn(cp, _obs(composite), beliefs, prev_beliefs=beliefs)
    delta = np.asarray(updated.transition["world.location"].concentration - driven.concentration)
    assert np.isclose(delta.sum(), 1.0, atol=1e-5)  # one observation's worth of count
    assert delta[..., 0].sum() > 0.99  # ... in the no-op control slot, per the rewrite


def test_a_uniform_anchor_survives_the_control_axis_rewrite():
    """A scalar ``alpha0`` (the uniform anchor) has no control axis to re-read, so the rewrite
    keeps it whole, and a decaying step on the composite still relaxes toward it."""
    from hybrid_aif import ConditionalDirichlet

    (agent, ap), (world, wp) = agent_and_world(policy_len=1)
    counts = ConditionalDirichlet(
        concentration=jnp.exp(wp.transition["location"].log_probs) + 1.0,
        parent_ndims=2,
        alpha0=jnp.asarray(1.0),
    )
    wp = dataclasses.replace(wp, transition={**wp.transition, "location": counts})
    composite, cp = merge(
        {"agent": agent, "world": world},
        {"agent": ap, "world": wp},
        [DRIVE, *OBSERVATION_BINDINGS],
    )
    driven = cp.transition["world.location"]
    assert driven.alpha0.shape == ()
    beliefs, _ = composite.infer_states(cp, _obs(composite))
    drifted = composite.learn(
        cp, _obs(composite), beliefs, prev_beliefs=beliefs, lr=jnp.asarray(0.0), decay=0.5
    )
    np.testing.assert_allclose(
        np.asarray(drifted.transition["world.location"].concentration),
        np.asarray(0.5 * driven.concentration + 0.5),
        atol=1e-6,
    )


def test_a_uniform_anchor_survives_cutting_several_control_axes():
    """``_cut_axes`` re-reads a table's control axes; a scalar anchor has none and is kept."""
    from hybrid_aif import ConditionalDirichlet
    from hybrid_aif.composition.merge import _cut_axes

    conc = jnp.ones((3, 4, 2, 2))  # child, one state parent, two control axes
    for anchor, want in ((jnp.asarray(1.0), ()), (jnp.full((3, 4, 2, 2), 0.5), None)):
        decl = ConditionalDirichlet(concentration=conc, parent_ndims=3, alpha0=anchor)
        one = _cut_axes(decl, "x", 2, [(1, 1, None)])
        both = _cut_axes(decl, "x", 2, [(1, 1, None), (2, 0, None)])
        assert one.concentration.shape == (3, 4, 2) and one.parent_ndims == 2
        assert both.concentration.shape == (3, 4, 1) and both.parent_ndims == 2
        for out in (one, both):
            assert out.alpha0.shape == (want if want is not None else out.concentration.shape)
        np.testing.assert_allclose(
            np.asarray(one.absorb(jnp.zeros((3, 4, 2)), decay=0.5).concentration),
            np.asarray(0.5 * one.concentration + 0.5 * (anchor if anchor.ndim == 0 else 0.5)),
        )


def test_D_deps_offsets_into_the_composite_frame():
    from tests.test_d_deps import hierarchical_tmaze

    hier, hp = hierarchical_tmaze()
    high, high_p = _high(policy_len=hier.policy_len)
    composite, cp = merge({"high": high, "hier": hier}, {"high": high_p, "hier": hp}, [])
    #  'hier.context' still reads 'hier.regime'; the control node's entry is empty
    assert composite.D_deps == ((), (), (), (1,), ())
    priors = composite.initial_priors(cp)
    reference = hier.initial_priors(hp)
    for name, want in reference.items():
        assert jnp.array_equal(priors[f"hier.{name}"].probs, want.probs)


# ------------------------------------------------------------------------ policy_prior


def test_policy_prior_composes_as_a_sum_over_the_joint_policy_set():
    """Independent habits add in log space over the product policy space; the
    per-component index is recomputed from the composite enumeration's own digits."""
    (agent, ap), (world, wp) = agent_and_world(policy_len=1)
    la = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    lw = jnp.asarray([10.0, 20.0, 30.0, 40.0, 50.0])
    composite, cp = merge(
        {"agent": agent, "world": world},
        {
            "agent": dataclasses.replace(ap, policy_prior=la),
            "world": dataclasses.replace(wp, policy_prior=lw),
        },
        [*OBSERVATION_BINDINGS],  # both control nodes stay chosen: 25 joint policies
    )
    policies = composite.policies(cp)
    assert policies.shape[0] == 25 and cp.policy_prior.shape == (25,)
    for p in range(25):
        want = float(la[policies[p, 0, 0]] + lw[policies[p, 0, 2]])
        assert np.isclose(float(cp.policy_prior[p]), want)
    beliefs, _ = composite.infer_states(cp, _obs(composite))
    q_pi, _ = composite.infer_policies(cp, beliefs)
    assert np.isclose(float(q_pi.probs.sum()), 1.0, atol=1e-5)


def test_policy_prior_absent_everywhere_stays_None():
    _, cp = merged()
    assert cp.policy_prior is None


def test_policy_prior_on_a_merged_control_node_raises():
    (agent, ap), (world, wp) = agent_and_world(policy_len=1)
    wp = dataclasses.replace(wp, policy_prior=jnp.zeros(5))
    with pytest.raises(ValueError, match="were merged in this call"):
        merge(
            {"agent": agent, "world": world},
            {"agent": ap, "world": wp},
            [DRIVE, *OBSERVATION_BINDINGS],
        )


def test_policy_prior_of_the_wrong_size_raises():
    (agent, ap), (world, wp) = agent_and_world(policy_len=1)
    ap = dataclasses.replace(ap, policy_prior=jnp.zeros(4))
    with pytest.raises(ValueError, match="one weight per policy"):
        merge({"agent": agent, "world": world}, {"agent": ap, "world": wp}, OBSERVATION_BINDINGS)


# --------------------------------------------------------------------- metadata


def test_subsystems_metadata_records_the_control_nodes():
    composite, _ = merged()
    meta = dict(composite.subsystems)
    assert set(meta) == {"agent", "world"}
    assert meta["agent"].factors == (0, 1) and meta["agent"].modalities == (0, 1, 2)
    assert meta["agent"].controls == ("agent.u_location",)  # the composite plans it
    assert meta["agent"].actions == (("agent.u_location", "agent.a_location"),)
    assert meta["agent"].relayed == () and meta["agent"].terminated == ()
    assert meta["agent"].free_modalities == ()  # all three read across the boundary
    assert meta["agent"].bound_modalities == ((0, (2,)), (1, (2, 3)), (2, (2, 3)))
    assert meta["world"].factors == (2, 3) and meta["world"].modalities == ()
    assert meta["world"].relayed == (("world.u_location", "agent.location"),)
    assert meta["world"].controls == ()


def test_boundary_prior_is_recorded_as_a_terminated_control_node():
    composite, _ = merged(bindings=[*OBSERVATION_BINDINGS, BoundaryPrior("world.u_location", CUE)])
    assert dict(composite.subsystems)["world"].terminated == ("world.u_location",)


def test_merge_is_closed_and_qualifies_nested_metadata():
    """Requirement 2 taken literally: a composite merges again, names gain another
    prefix, and the inner ontology survives as qualified metadata."""
    inner, ip = merged()
    sup, sp = _high(policy_len=inner.policy_len)
    outer, op = merge({"pair": inner, "sup": sup}, {"pair": ip, "sup": sp}, [])
    assert outer.factor_names[0] == "pair.agent.location"
    assert outer.names.states[-1] == "sup.mode"
    meta = dict(outer.subsystems)
    assert set(meta) == {"pair", "sup", "pair.agent", "pair.world"}
    assert meta["pair"].factors == (0, 1, 2, 3) and meta["pair.world"].factors == (2, 3)
    # still merged, still recorded
    assert meta["pair.world"].relayed == (("pair.world.u_location", "pair.agent.location"),)
    assert all(key.startswith("pair.") or key.startswith("obs:pair.") for key in outer.epistemic)
    beliefs, _ = outer.infer_states(op, _obs(outer))
    q_pi, _ = outer.infer_policies(op, beliefs)
    assert q_pi.probs.shape[0] == outer.policies(op).shape[0]


# ------------------------------------------------------------------- validation


def test_unknown_binding_names_list_the_candidates():
    (agent, ap), (world, wp) = agent_and_world()
    subs, ps = {"agent": agent, "world": world}, {"agent": ap, "world": wp}
    with pytest.raises(ValueError, match="'agent.locaiton' is neither a control node nor a state"):
        merge(subs, ps, [ActionControl(driver="agent.locaiton", control="world.u_location")])
    with pytest.raises(ValueError, match="unknown modality"):
        merge(subs, ps, [ObservationLink(modality="agent.vision", factors=("world.location",))])
    with pytest.raises(ValueError, match="world.u_location"):  # candidates are listed
        merge(subs, ps, [ActionControl(driver="agent.location", control="world.loc")])
    with pytest.raises(TypeError, match="unknown binding"):
        merge(subs, ps, ["agent.location -> world.location"])


def test_node_typing_errors():
    (agent, ap), (world, wp) = agent_and_world()
    subs, ps = {"agent": agent, "world": world}, {"agent": ap, "world": wp}
    low, lp = _low(policy_len=agent.policy_len)  # a 3-level control node
    with pytest.raises(ValueError, match="node arity mismatch"):
        merge(
            {"low": low, "world": world},
            {"low": lp, "world": wp},
            [ActionControl(driver="low.pos", control="world.u_location")],
        )
    with pytest.raises(ValueError, match=r"already merged \(with \['agent.location'\]\)"):
        merge(
            subs,
            ps,
            [
                ActionControl(driver="agent.location", control="world.u_location"),
                ActionControl(driver="world.context", control="world.u_location"),
            ],
        )
    with pytest.raises(ValueError, match="a state factor, where a binding names a control node"):
        merge(subs, ps, [ActionControl(driver="agent.location", control="world.context")])
    with pytest.raises(ValueError, match="already merged"):
        merge(subs, ps, [DRIVE, DRIVE])
    with pytest.raises(ValueError, match="support mismatch"):
        merge(subs, ps, [ObservationLink(modality="agent.reward", factors=("world.context",) * 2)])
    with pytest.raises(ValueError, match="parent axis"):
        merge(subs, ps, [ObservationLink(modality="agent.reward", factors=("world.location",))])
    with pytest.raises(ValueError, match="a state factor, where a binding names a control node"):
        merge(subs, ps, [BoundaryPrior("world.context", 0)])
    with pytest.raises(ValueError, match="one probability per level"):
        merge(subs, ps, [BoundaryPrior("world.u_location", jnp.zeros(3))])
    with pytest.raises(ValueError, match="not one of its 5 control level"):
        merge(subs, ps, [BoundaryPrior("world.u_location", 9)])


def test_inference_config_disagreement_raises():
    (agent, ap), (world, wp) = agent_and_world()
    slow = dataclasses.replace(world, policy_len=3)
    with pytest.raises(ValueError, match="disagree on policy_len"):
        merge({"agent": agent, "world": slow}, {"agent": ap, "world": wp}, OBSERVATION_BINDINGS)
    hot = dataclasses.replace(world, gamma=1.0)
    with pytest.raises(ValueError, match="disagree on gamma"):
        merge({"agent": agent, "world": hot}, {"agent": ap, "world": wp}, OBSERVATION_BINDINGS)


def test_structural_errors_and_bad_keys_raise_at_build_time():
    (agent, ap), (world, wp) = agent_and_world()
    with pytest.raises(ValueError, match="keyed alike"):
        merge({"agent": agent}, {"world": wp}, [])
    with pytest.raises(ValueError, match="at least one subsystem"):
        merge({}, {}, [])
    with pytest.raises(ValueError, match="namespace separator"):
        merge({"a.b": agent}, {"a.b": ap}, [])
    # the composite is validated before it is returned: a target emission onto an
    # *unconditional* designation fails here, with the face's own message
    wp_bad = dataclasses.replace(wp, terminal_targets={"context": Categorical(logits=jnp.zeros(2))})
    with pytest.raises(ValueError, match="ConditionalCategorical"):
        merge(
            {"agent": agent, "world": world},
            {"agent": ap, "world": wp_bad},
            [*OBSERVATION_BINDINGS, TargetEmission("world.context", ("agent.location",))],
        )


def test_target_emission_does_not_override_a_components_own_H_deps():
    (agent, ap), (world, wp) = agent_and_world()
    target = ConditionalCategorical(
        log_table=jnp.log(jnp.asarray([[0.9, 0.8, 0.7, 0.6, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])),
        parent_ndims=1,
    )
    own = dataclasses.replace(world, H_deps=((), (0,), ()))
    wp = dataclasses.replace(wp, terminal_targets={"context": target})
    with pytest.raises(ValueError, match="already declares H_deps"):
        merge(
            {"agent": agent, "world": own},
            {"agent": ap, "world": wp},
            [*OBSERVATION_BINDINGS, TargetEmission("world.context", ("agent.location",))],
        )
