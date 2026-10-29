"""Action nodes: one per control node, the dual of a modality, and the face every action crosses.

A System declares its action nodes beside its control nodes (``action_names``, one per control
node; the sugar's ``u_<f>`` gets ``a_<f>``), and every face that carries an action keys it by
action node: ``act``, the converters, ``update_empirical_prior``, ``learn``, ``as_env`` and
``GenerativeEnv``, rollouts, replay and ``stack_traces``, the driver. A mapping keyed by a control
node where an action is expected is refused naming the action node. The control node is clamped
to the decision (the efference copy); a kernel ``P(a | u)`` in ``params.actuator`` is drawn by the
environment, the actuator, and the agent hears of the realised action only through a modality on
the action node. ``merge()`` identifies any latents, a relay from an uncontrolled state included,
and binds native control nodes; a model description writes the composite flat, and its reader
reconstructs each component's record from the names.

Design: docs/design/control-states.md C12.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import chi2

from hybrid_aif import Categorical, ConditionalCategorical
from hybrid_aif.composition.driver import (
    MindCarry,
    _check_action,
    agent_respond,
    ancestral_respond,
    run,
)
from hybrid_aif.composition.merge import ActionControl, BoundaryPrior, merge, subsystem
from hybrid_aif.description import build, describe
from hybrid_aif.envs import GenerativeEnv
from hybrid_aif.envs._process import DiscreteProcess
from hybrid_aif.replay import replay, stack_traces
from hybrid_aif.rollout import random_policy, rollout
from hybrid_aif.system import System, SystemParams
from tests.test_merge import OBSERVATION_BINDINGS, agent_and_world, merged
from tests.test_system import tmaze_system

KEY = jax.random.PRNGKey(0)

#: P(a | u) over three levels, column u the realised action's distribution given decision u.
KERNEL = jnp.asarray([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])
#: A deterministic actuator that realises u + 1 (mod 3).
SHIFT = jnp.roll(jnp.eye(3), 1, axis=0)
#: A proprioceptive likelihood P(o | a).
FEEL = jnp.asarray([[0.8, 0.1, 0.1], [0.15, 0.7, 0.15], [0.05, 0.2, 0.75]])


def _roll(noise: float = 0.0) -> jnp.ndarray:
    """``x' = x + u (mod 3)``, as a child-first ``(x', x, u)`` table, with optional noise."""
    table = jnp.stack([jnp.roll(jnp.eye(3), u, axis=0) for u in range(3)], axis=-1)
    return (1.0 - noise) * table + noise / 3.0


def pusher(kernel=None, feel=None, noise: float = 0.05):
    """A native System: a state ``x`` moved by the control node ``u_push`` (action ``push``),
    seen through ``see``; with ``feel``, a modality on the action node itself."""
    sensed = feel is not None
    system = System(
        A_deps=((0,), (2,)) if sensed else ((0,),),
        B_deps=((0, 1), ()),
        controls=(1,),
        action_names=("push",),
        factor_names=("x", "u_push"),
        modality_names=("see", "feel") if sensed else ("see",),
        policy_len=1,
        epistemic=False,
    )
    likelihood = {
        "see": ConditionalCategorical(log_table=jnp.log(0.8 * jnp.eye(3) + 0.2 / 3), parent_ndims=1)
    }
    if sensed:
        likelihood["feel"] = ConditionalCategorical(log_table=jnp.log(feel), parent_ndims=1)
    params = SystemParams(
        likelihood=likelihood,
        transition={"x": ConditionalCategorical(log_table=jnp.log(_roll(noise)), parent_ndims=2)},
        initial_state_prior={"x": Categorical(logits=jnp.log(jnp.asarray([0.6, 0.3, 0.1])))},
        actuator={} if kernel is None else {"push": kernel},
    )
    return system, params


# ------------------------------------------------------------------------- names and declaration


def test_action_nodes_sit_in_names_beside_their_control_nodes():
    sugar, _ = tmaze_system()
    assert sugar.action_names == ("a_location",)
    assert sugar.names.controls == ("u_location",) and sugar.names.actions == ("a_location",)
    unnamed = System(B_deps=((), ()), controlled=(True, False), policy_len=1)
    assert unnamed.names.controls == ("u_factor_0",)
    assert unnamed.names.actions == ("a_factor_0",)
    native, _ = pusher()
    assert native.names.states == ("x",) and native.names.controls == ("u_push",)
    assert native.names.actions == ("push",) and native.names.modalities == ("see",)
    assert System(B_deps=((),), policy_len=1).names.actions == ()


def test_a_native_control_node_and_its_action_node_are_named_freely():
    common = dict(B_deps=((0, 1), ()), controls=(1,), policy_len=1)
    with pytest.raises(ValueError, match=r"declared without action nodes: pass action_names="):
        System(**common, factor_names=("x", "u_push"))
    free = System(**common, factor_names=("x", "go"), action_names=("move",))
    assert free.names.controls == ("go",) and free.names.actions == ("move",)
    with pytest.raises(ValueError, match=r"action_names has 2 name\(s\) for 1 control node"):
        System(**common, factor_names=("x", "u_push"), action_names=("push", "pull"))
    with pytest.raises(ValueError, match=r"action_names \['x'\] are also factor or modality"):
        System(**common, factor_names=("x", "u_x"), action_names=("x",))
    with pytest.raises(ValueError, match="for a System with no control nodes"):
        System(B_deps=((),), action_names=("push",), policy_len=1)
    with pytest.raises(ValueError, match="the sugar names each controlled factor's action node"):
        System(B_deps=((),), controlled=(True,), action_names=("push",), policy_len=1)
    #  the sugar's pair is a native declaration too: the built Systems are one
    sugar = System(B_deps=((0,),), controlled=(True,), factor_names=("x",), policy_len=1)
    twin = System(
        B_deps=((0, 1), ()),
        controls=(1,),
        factor_names=("x", "u_x"),
        action_names=("a_x",),
        policy_len=1,
    )
    assert sugar == twin


# ----------------------------------------------------- the action face speaks action-node names


def _tmaze_trace(system, params, T=3):
    env, ep = system.as_env(params)
    obs, state = env.reset(KEY, ep)
    observations, actions = [obs], []
    for t in range(T - 1):
        action = {"a_location": jnp.asarray(t + 1, jnp.int32)}
        obs, state = env.step(jax.random.fold_in(KEY, t), state, action, ep)
        observations.append(obs)
        actions.append(action)
    actions.append({"a_location": jnp.asarray(0, jnp.int32)})
    stack = lambda rows: {k: jnp.stack([r[k] for r in rows]) for k in rows[0]}  # noqa: E731
    return stack(observations), stack(actions)


def test_every_action_face_keys_by_action_node_and_refuses_the_control_node():
    system, params = tmaze_system()
    beliefs = system.initial_priors(params)
    action, _, _ = system.act(KEY, params, beliefs)
    assert list(action) == ["a_location"]
    assert list(system.from_positional("actions", jnp.asarray([2], jnp.int32))) == ["a_location"]
    by_control = {"u_location": jnp.asarray(2, jnp.int32)}
    refusal = r"names 'u_location', the control node of action 'a_location'; key the action by"
    with pytest.raises(ValueError, match=f"update_empirical_prior: actions {refusal}"):
        system.update_empirical_prior(params, beliefs, by_control)
    with pytest.raises(ValueError, match=f"to_positional\\('actions', ...\\): actions {refusal}"):
        system.to_positional("actions", by_control)
    obs = {"location_obs": jnp.asarray(0), "reward": jnp.asarray(0), "cue": jnp.asarray(0)}
    posterior, _ = system.infer_states(params, obs)
    with pytest.raises(ValueError, match=f"learn: actions {refusal}"):
        system.learn(params, obs, posterior, prev_beliefs=beliefs, action=by_control)
    env, ep = system.as_env(params)
    assert set(env.null_action()) == {"a_location"}
    _, state = env.reset(KEY, ep)
    with pytest.raises(ValueError, match=f"step: actions {refusal}"):
        env.step(KEY, state, by_control, ep)
    outside = DiscreteProcess(A_deps=system.A_deps, B_deps=system._B)
    wrapped = GenerativeEnv.wrap(outside, system)
    assert set(wrapped.null_action()) == {"a_location"}
    with pytest.raises(ValueError, match=f"step: actions {refusal}"):
        wrapped.step(KEY, state, by_control, system.to_positional("params", ep))
    batched = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (2, *x.shape)), ep)
    traj = rollout(KEY, env, batched, random_policy(system, params), 3, 2)
    assert set(traj.actions) == {"a_location"}
    observations, actions = _tmaze_trace(system, params)
    out = replay(system, params, observations, actions, metrics=False)
    assert list(out.chosen) == ["a_location"]
    with pytest.raises(ValueError, match=f"replay: actions {refusal}"):
        replay(system, params, observations, {"u_location": actions["a_location"]})
    _, stacked, _, _ = stack_traces([(observations, actions, None)] * 2)
    assert list(stacked) == ["a_location"]
    #  legal= stays keyed by control node: an action node there is refused naming the node
    with pytest.raises(ValueError, match="the action node of control node 'u_location'; key it"):
        system.act(KEY, params, beliefs, legal={"a_location": jnp.ones(5, bool)})


def test_the_driver_routes_and_records_action_nodes():
    composite, cp = merged()
    view, vp = subsystem(composite, cp, "agent")
    traj = run(composite, cp, {"agent": ancestral_respond(view, vp)}, ticks=3, key=KEY)
    assert list(traj.actions) == list(composite.names.actions) == ["agent.a_location"]
    assert list(traj.realised) == ["agent.a_location"]
    assert jnp.array_equal(traj.realised["agent.a_location"], traj.actions["agent.a_location"])
    action, _ = ancestral_respond(view, vp)(KEY, {}, MindCarry())
    assert list(action) == ["agent.a_location"]
    with pytest.raises(ValueError, match="a control node; key the action by its action node"):
        _check_action(
            {"agent.u_location": jnp.asarray(0, jnp.int32)},
            ("agent.a_location",),
            "agent",
            {"agent.u_location": "agent.a_location"},
        )


# --------------------------------------------------------------- the efference copy and actuator


def test_the_efference_copy_under_the_delta_kernel_is_the_clamp_bitwise():
    """The control node is clamped to the decision: the keyed face is the positional push of the
    level, bitwise, and a kernel in params.actuator does not enter it."""
    system, params = tmaze_system()
    beliefs = system.initial_priors(params)
    positional = system.to_positional("beliefs", beliefs)
    noisy = jnp.full((5, 5), 0.05) + 0.75 * jnp.eye(5)
    with_kernel = dataclasses.replace(params, actuator={"a_location": noisy})
    for level in range(5):
        keyed = system.update_empirical_prior(params, beliefs, {"a_location": level})
        clamped = system._update_empirical_prior(
            system._p(params), positional, jnp.asarray([level], jnp.int32)
        )
        kernel = system.update_empirical_prior(with_kernel, beliefs, {"a_location": level})
        for i, name in enumerate(system.names.states):
            assert jnp.array_equal(keyed[name].probs, clamped[i].probs)
            assert jnp.array_equal(kernel[name].probs, keyed[name].probs)
        assert jnp.array_equal(kernel["u_location"].probs, keyed["u_location"].probs)


def test_the_actuator_realises_the_action_from_its_kernel():
    system, params = pusher(kernel=KERNEL)
    env, ep = system.as_env(params)
    _, state = env.reset(KEY, ep)
    #  n = 20000 draws per intended level; Pearson's statistic over the three columns has
    #  3 x (3 - 1) = 6 degrees of freedom, compared with its 0.999 quantile
    n = 20000
    statistic = 0.0
    for u in range(3):
        keys = jax.random.split(jax.random.fold_in(KEY, u), n)
        draw = jax.vmap(lambda k, u=u: env.step_actuated(k, state, {"push": u}, ep)[2]["push"])
        observed = np.bincount(np.asarray(draw(keys)), minlength=3)
        expected = n * np.asarray(KERNEL[:, u], np.float64)
        statistic += float(np.sum((observed - expected) ** 2 / expected))
    assert statistic < chi2.ppf(0.999, 6), statistic
    #  the delta kernel realises the intended action, and step is step_actuated's first two
    plain, pp = pusher()
    env0, ep0 = plain.as_env(pp)
    for k in jax.random.split(KEY, 4):
        obs, nxt, got = env0.step_actuated(k, state, {"push": 2}, ep0)
        assert int(got["push"]) == 2
        again = env0.step(k, state, {"push": 2}, ep0)
        assert jax.tree_util.tree_all(jax.tree_util.tree_map(jnp.array_equal, (obs, nxt), again))
    with pytest.raises(ValueError, match="params.actuator names 'u_push', the control node"):
        system.as_env(dataclasses.replace(params, actuator={"u_push": KERNEL}))
    with pytest.raises(ValueError, match=r"a \(3, 3\) table with one column per decision"):
        system.as_env(dataclasses.replace(params, actuator={"push": KERNEL[:2]}))


def test_the_agents_control_belief_is_unchanged_by_the_draw():
    """The actuator realises another action than the intended one; the world moves on it, and
    the agent, which pushes the intended action, holds the delta-kernel agent's beliefs."""
    system, params = pusher(kernel=SHIFT, noise=0.0)
    env, ep = system.as_env(params)
    _, state = env.reset(KEY, ep)
    _, moved, realised = env.step_actuated(KEY, state, {"push": 1}, ep)
    assert int(realised["push"]) == 2  # the shift actuator realises u + 1
    assert int(moved["x"]) == (int(state["x"]) + 2) % 3  # the world moved on the realised push
    beliefs = system.initial_priors(params)
    pushed = system.update_empirical_prior(params, beliefs, {"push": 1})
    plain, pp = pusher(noise=0.0)
    delta = plain.update_empirical_prior(pp, plain.initial_priors(pp), {"push": 1})
    for name in ("x", "u_push"):
        assert jnp.array_equal(pushed[name].probs, delta[name].probs)
    #  the driver records both: the intended action and the realised one
    composite, cp = merge({"m": system}, {"m": params})
    view, vp = subsystem(composite, cp, "m")
    traj = run(composite, cp, {"m": ancestral_respond(view, vp)}, ticks=6, key=KEY)
    assert jnp.array_equal(traj.realised["m.push"], (traj.actions["m.push"] + 1) % 3)
    x = traj.states["m.x"]
    assert jnp.array_equal(x[1:], (x[:-1] + traj.realised["m.push"]) % 3)


def test_a_modality_on_the_action_node_updates_its_belief_by_the_likelihood():
    system, params = pusher(kernel=KERNEL, feel=FEEL)
    assert system._sensed == ("push",)
    beliefs = system.initial_priors(params)
    pushed = system.update_empirical_prior(params, beliefs, {"push": 1})
    np.testing.assert_allclose(
        np.asarray(pushed["push"].probs), np.asarray(KERNEL[:, 1]), rtol=1e-6
    )
    posterior, _ = system.infer_states(
        params, {"see": jnp.asarray(2), "feel": jnp.asarray(0)}, pushed
    )
    want = FEEL[0] * KERNEL[:, 1]
    np.testing.assert_allclose(
        np.asarray(posterior["push"].probs), np.asarray(want / want.sum()), rtol=1e-5
    )
    see = jnp.exp(params.likelihood["see"].log_table)[2] * pushed["x"].probs
    np.testing.assert_allclose(
        np.asarray(posterior["x"].probs), np.asarray(see / see.sum()), rtol=1e-5
    )
    assert jnp.array_equal(posterior["u_push"].probs, pushed["u_push"].probs)
    #  the delta kernel predicts the decision itself
    plain, pp = pusher(feel=FEEL)
    delta = plain.update_empirical_prior(pp, plain.initial_priors(pp), {"push": 1})
    assert jnp.array_equal(delta["push"].probs, jnp.asarray([0.0, 1.0, 0.0]))
    #  the world emits the realised action to the modality
    exact, ep = pusher(kernel=SHIFT, feel=jnp.eye(3))
    env, eparams = exact.as_env(ep)
    _, state = env.reset(KEY, eparams)
    for k in jax.random.split(KEY, 3):
        obs, state, realised = env.step_actuated(k, state, {"push": 0}, eparams)
        assert int(obs["feel"]) == int(realised["push"]) == 1


# ----------------------------------------------------------------------------------- merge()


def _source():
    """An uncontrolled three-level state ``s`` that cycles."""
    system = System(B_deps=((0,),), controlled=(False,), factor_names=("s",), policy_len=1)
    table = jnp.roll(jnp.eye(3), 1, axis=0)[..., None] * 0.9 + 0.1 / 3
    params = SystemParams(
        transition={"s": ConditionalCategorical(log_table=jnp.log(table), parent_ndims=2)},
        initial_state_prior={"s": Categorical(logits=jnp.log(jnp.asarray([0.2, 0.5, 0.3])))},
    )
    return system, params


def test_a_merge_identifies_a_control_node_with_any_state():
    """The relay from a state of no controlled factor: the reader's control axis is re-read as
    that state, a step later, against the hand computation."""
    (src, sp), (w, wp) = _source(), pusher()
    composite, cp = merge(
        {"src": src, "w": w}, {"src": sp, "w": wp}, [ActionControl("src.s", "w.u_push")]
    )
    assert composite.names.controls == () and composite.names.actions == ()
    record = dict(composite.subsystems)["w"]
    assert record.relayed == (("w.u_push", "src.s"),)
    assert record.actions == (("w.u_push", "w.push"),)
    beliefs = composite.initial_priors(cp)
    pushed = composite.update_empirical_prior(cp, beliefs, {})
    B = _roll(0.05)  # (x', x, u), u read as s
    want = jnp.einsum("yxs,x,s->y", B, beliefs["w.x"].probs, beliefs["src.s"].probs)
    np.testing.assert_allclose(np.asarray(pushed["w.x"].probs), np.asarray(want), rtol=1e-6)
    with pytest.raises(ValueError, match="names 'w.push', the action node of a control node merge"):
        composite.update_empirical_prior(cp, beliefs, {"w.push": 0})


def test_a_merge_binds_native_control_nodes():
    (a, ap), (b, bp) = pusher(), pusher()
    composite, cp = merge(
        {"a": a, "b": b}, {"a": ap, "b": bp}, [ActionControl("a.u_push", "b.u_push")]
    )
    assert composite.names.controls == ("a.u_push",) and composite.names.actions == ("a.push",)
    assert dict(composite.subsystems)["b"].identified == (("b.u_push", "a.u_push"),)
    beliefs = composite.initial_priors(cp)
    pushed = composite.update_empirical_prior(cp, beliefs, {"a.push": 2})
    for name in ("a.x", "b.x"):
        want = _roll(0.05)[:, :, 2] @ beliefs[name].probs
        np.testing.assert_allclose(np.asarray(pushed[name].probs), np.asarray(want), rtol=1e-6)
    with pytest.raises(ValueError, match="key it by 'a.push'"):
        composite.update_empirical_prior(cp, beliefs, {"a.push": 2, "b.push": 2})
    with pytest.raises(ValueError, match="key it by 'a.u_push'"):
        agent_respond(composite, cp, legal={"b.u_push": jnp.ones(3, bool)})
    view, vp = subsystem(composite, cp, "a")
    traj = run(composite, cp, {"a": ancestral_respond(view, vp)}, ticks=3, key=KEY)
    assert list(traj.actions) == ["a.push"]
    #  a terminated native node, and a kernel that moves with its node
    one, op = merge({"a": a, "b": b}, {"a": ap, "b": bp}, [BoundaryPrior("b.u_push", 1)])
    assert dict(one.subsystems)["b"].terminated == ("b.u_push",)
    (k, kp) = pusher(kernel=KERNEL)
    kept, kpp = merge({"k": k}, {"k": kp})
    assert jnp.array_equal(kpp.actuator["k.push"], KERNEL)


# ----------------------------------------------------------------------------- the description


def test_a_composite_s_component_records_are_rebuilt_from_its_names():
    """A relay and a terminated node are structure in the document: the rebuilt records carry
    the components' factors, modalities and chosen nodes, and nothing about a removed node."""
    relay, rp = merged()
    document, arrays = describe(relay, rp)
    rebuilt = build(document, arrays=arrays)[0]
    agent, world = dict(rebuilt.subsystems)["agent"], dict(rebuilt.subsystems)["world"]
    assert agent == dict(relay.subsystems)["agent"]
    assert world.factors == (2, 3) and world.controls == () and world.relayed == ()
    (agent_s, ap), (world_s, wp) = agent_and_world()
    ended, ep = merge(
        {"agent": agent_s, "world": world_s},
        {"agent": ap, "world": wp},
        [*OBSERVATION_BINDINGS, BoundaryPrior("world.u_location", 1)],
    )
    document, arrays = describe(ended, ep)
    assert "world.u_location" not in document["variables"]
    rebuilt = build(document, arrays=arrays)[0]
    assert dict(rebuilt.subsystems)["world"].terminated == ()
    assert dict(rebuilt.subsystems)["agent"] == dict(ended.subsystems)["agent"]


# --------------------------------------------------------------------- the actuator, further


def test_the_realised_action_and_the_next_state_are_drawn_independently():
    """With a noisy world, the joint (a, x') over draws is the kernel column times the
    transition at the realised action: the actuator's key is tagged apart from the world's."""
    system, params = pusher(kernel=KERNEL, noise=0.05)
    env, ep = system.as_env(params)
    _, state = env.reset(KEY, ep, {"x": jnp.asarray(0, jnp.int32)})

    def draw(k):
        _, nxt, realised = env.step_actuated(k, state, {"push": 1}, ep)
        return realised["push"], nxt["x"]

    #  n = 40000 draws; Pearson's statistic over the 9 (a, x') cells has 8 degrees of
    #  freedom, compared with its 0.999 quantile
    n = 40000
    a, x = jax.vmap(draw)(jax.random.split(KEY, n))
    observed = np.zeros((3, 3))
    np.add.at(observed, (np.asarray(a), np.asarray(x)), 1)
    B = np.asarray(_roll(0.05), np.float64)  # (x', x, u)
    expected = n * np.asarray(KERNEL[:, 1], np.float64)[:, None] * B[:, 0, :].T
    statistic = float(np.sum((observed - expected) ** 2 / expected))
    assert statistic < chi2.ppf(0.999, 8), statistic


@pytest.mark.parametrize(
    "bad",
    [KERNEL * 1.1, jnp.asarray([[1.2, 0.2, 0.1], [-0.2, 0.7, 0.1], [0.0, 0.1, 0.8]])]
    + [KERNEL.at[0, 0].set(jnp.nan)],
    ids=["sums", "negative", "nan"],
)
def test_a_malformed_kernel_is_refused_eagerly_and_under_jit(bad):
    system, params = pusher(kernel=KERNEL)
    with pytest.raises(ValueError, match="is not a probability table"):
        system.as_env(dataclasses.replace(params, actuator={"push": bad}))
    env, ep = system.as_env(params)
    _, state = env.reset(KEY, ep)
    wrong = dataclasses.replace(ep, actuator={"push": bad})
    with pytest.raises(ValueError, match="is not a probability table"):
        env.step_actuated(KEY, state, {"push": 1}, wrong)
    with pytest.raises(Exception, match="is not a probability table"):
        jax.jit(env.step_actuated)(KEY, state, {"push": 1}, wrong)
    sensed, sp = pusher(kernel=KERNEL, feel=FEEL)
    beliefs = sensed.initial_priors(sp)

    def push(p):
        return sensed.update_empirical_prior(p, beliefs, {"push": 1})["push"].probs

    with pytest.raises(Exception, match="is not a probability table"):
        jax.jit(push)(dataclasses.replace(sp, actuator={"push": bad}))


def test_a_kernel_on_a_continuous_node_is_refused():
    """A kernel P(a | u) is a table over a node's levels; a continuous node has none, so the
    entry is refused rather than given a meaning here."""
    from tests.test_continuous_nodes import _native_continuous

    system, params = _native_continuous()
    with pytest.raises(NotImplementedError, match="no levels to tabulate"):
        system.as_env(dataclasses.replace(params, actuator={"u": jnp.eye(1)}))


def test_a_declared_kernel_leaves_every_planning_face_unchanged():
    """The planner's rollouts read the control node, so a kernel changes nothing a planning face
    returns; its effects are the realised draw, a modality on the action node, and the trace."""
    system, params = tmaze_system()
    levels = system._control_levels(params)[0]
    kernel = (
        jnp.full((levels, levels), 0.2 / (levels - 1))
        .at[jnp.arange(levels), jnp.arange(levels)]
        .set(0.8)
    )
    with_kernel = dataclasses.replace(params, actuator={system.names.actions[0]: kernel})
    beliefs = system.initial_priors(params)

    def leaves(x):
        return jax.tree_util.tree_leaves(x)

    plain_act = system.act(KEY, params, beliefs)
    kernel_act = system.act(KEY, with_kernel, beliefs)
    assert jax.tree_util.tree_structure(plain_act) == jax.tree_util.tree_structure(kernel_act)
    for a, b in zip(leaves(plain_act), leaves(kernel_act), strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    plain_q, plain_tree = system.infer_policies(params, beliefs)
    kernel_q, kernel_tree = system.infer_policies(with_kernel, beliefs)
    assert np.array_equal(np.asarray(plain_q.logits), np.asarray(kernel_q.logits))
    for a, b in zip(leaves(plain_tree), leaves(kernel_tree), strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    policies = system.policies(params)
    plain_scored = system.score_policies(params, beliefs, policies)
    kernel_scored = system.score_policies(with_kernel, beliefs, policies)
    for a, b in zip(leaves(plain_scored), leaves(kernel_scored), strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------- naming, and observed action nodes


def test_malformed_action_names_are_refused():
    common = dict(B_deps=((0, 1, 2), (), ()), controls=(1, 2), policy_len=1)
    with pytest.raises(ValueError, match=r"action_names repeats \['go'\]"):
        System(**common, factor_names=("x", "u_go", "u_stop"), action_names=("go", "go"))
    one = dict(B_deps=((0, 1), ()), controls=(1,), policy_len=1)
    with pytest.raises(ValueError, match=r"action_names \['see'\] are also factor or modality"):
        System(
            **one,
            A_deps=((0,),),
            modality_names=("see",),
            factor_names=("x", "u_see"),
            action_names=("see",),
        )
    with pytest.raises(ValueError, match=r"action_names \[''\] have an empty name"):
        System(**one, factor_names=("x", "u_"), action_names=("",))
    #  a native pair is named freely; the u_ prefix is the sugar's convention only
    assert System(**one, factor_names=("x", "q_x"), action_names=("a_x",)).names.controls == (
        "q_x",
    )


def _hybrid_sensed():
    """A tabular state ``s`` and a linear-Gaussian ``x``, ``s`` moved by ``u_go`` (action
    ``go``), with a modality on the action node."""
    from tests.test_continuous_nodes import _gauss, _lg

    system = System(
        A_deps=((3,),),
        B_deps=((0, 2), (), ()),
        controls=(2,),
        action_names=("go",),
        factor_names=("s", "x", "u_go"),
        modality_names=("feel",),
        policy_len=1,
        epistemic=False,
    )
    params = SystemParams(
        likelihood={"feel": ConditionalCategorical(log_table=jnp.log(FEEL), parent_ndims=1)},
        transition={
            "s": ConditionalCategorical(log_table=jnp.log(_roll(0.05)), parent_ndims=2),
            "x": _lg(jnp.zeros((1, 2))),
        },
        initial_state_prior={"s": Categorical(logits=jnp.zeros(3)), "x": _gauss()},
    )
    return system, params


def test_every_face_that_does_not_read_an_observed_action_node_refuses_it():
    from hybrid_aif.replay import replay

    system, params = pusher(feel=FEEL)
    beliefs = system.initial_priors(params)
    obs = {"see": jnp.asarray(0), "feel": jnp.asarray(0)}
    remedy = r"observe action node\(s\) \['push'\], which {face} does not read; drop the action"
    for face, call in (
        ("learn", lambda: system.learn(params, obs, beliefs)),
        (
            "replay",
            lambda: replay(
                system,
                params,
                {k: jnp.zeros(2, jnp.int32) for k in obs},
                {"push": jnp.zeros(2, jnp.int32)},
            ),
        ),
        ("merge", lambda: merge({"m": system}, {"m": params})),
    ):
        with pytest.raises(NotImplementedError, match=remedy.format(face=face)):
            call()
    document, arrays = describe(system, params)  # a modality on an action node is a likelihood
    assert document["factors"]["feel"]["likelihood"]["parents"] == ["push"]
    assert build(document, arrays=arrays)[0] == system
    plain, pp = pusher()
    composite, cp = merge({"m": plain}, {"m": pp})
    sensed = dataclasses.replace(composite, A_deps=((0,), (2,)), modality_names=("m.see", "m.feel"))
    with pytest.raises(NotImplementedError, match=r"^subsystem: .*\['m.push'\], which subsystem"):
        subsystem(sensed, cp, "m")
    #  the planning faces and the hybrid path's inference faces read it
    looking = dataclasses.replace(system, epistemic=True)
    tree = looking.score_policies(params, beliefs, looking.policies(params))
    assert set(tree) == {"infogain"} and set(tree["infogain"]) == {"see"}  # none for feel (S2)
    assert tree["infogain"]["see"].lower.shape == (3,)
    assert looking.infer_policies(params, beliefs)[0].probs.shape == (3,)
    hybrid, hp = _hybrid_sensed()
    hb = hybrid.initial_priors(hp)
    pushed = hybrid.update_empirical_prior(hp, hb, {"go": 1})
    assert jnp.array_equal(pushed["go"].probs, jnp.asarray([0.0, 1.0, 0.0]))
    posterior, _ = hybrid.infer_states(hp, {"feel": jnp.asarray(0)}, pushed)
    want = FEEL[0] * jnp.asarray([0.0, 1.0, 0.0])
    np.testing.assert_allclose(
        np.asarray(posterior["go"].probs), np.asarray(want / want.sum()), rtol=1e-5
    )
    env, eparams = hybrid.as_env(hp)
    obs, state = env.reset(KEY, eparams)
    obs, state, realised = env.step_actuated(KEY, state, {"go": 1}, eparams)
    assert int(realised["go"]) == 1 and obs["feel"].shape == ()


def test_a_modality_on_a_continuous_action_node_is_refused():
    from tests.test_continuous_nodes import _native_continuous

    system, params = _native_continuous()
    sensed = dataclasses.replace(system, A_deps=((2,),), modality_names=("feel",))
    feel = ConditionalCategorical(log_table=jnp.zeros((2, 1)), parent_ndims=1)
    sp = dataclasses.replace(params, likelihood={"feel": feel})
    with pytest.raises(NotImplementedError, match=r"action node\(s\) \['u'\] of a continuous"):
        sensed.infer_states(sp, {"feel": jnp.asarray(0)})


def test_predict_beliefs_carries_the_observed_action_nodes_prediction():
    system, params = pusher(kernel=KERNEL, feel=FEEL)
    steps = system.predict_beliefs(
        params, system.initial_priors(params), jnp.asarray([[1], [2]], jnp.int32)
    )
    for step, u in zip(steps, (1, 2), strict=True):
        np.testing.assert_allclose(
            np.asarray(step["push"].probs), np.asarray(KERNEL[:, u]), rtol=1e-6
        )
    plain, pp = pusher(feel=FEEL)
    (step,) = plain.predict_beliefs(pp, plain.initial_priors(pp), jnp.asarray([[2]], jnp.int32))
    assert jnp.array_equal(step["push"].probs, jnp.asarray([0.0, 0.0, 1.0]))


# --------------------------------------------------------------------------- merge(), further


def test_a_binding_that_would_drop_a_kernel_is_refused():
    (a, ap), (b, bp) = pusher(kernel=KERNEL), pusher()
    same = ActionControl("a.u_push", "b.u_push")
    with pytest.raises(ValueError, match="kernels params.actuator holds .* differ"):
        merge({"a": a, "b": b}, {"a": ap, "b": bp}, [same])
    with pytest.raises(ValueError, match="kernels params.actuator holds .* differ"):
        merge({"a": b, "b": a}, {"a": bp, "b": ap}, [same])
    composite, cp = merge({"a": a, "b": a}, {"a": ap, "b": ap}, [same])  # equal kernels
    assert set(cp.actuator) == {"a.push"}
    (src, sp) = _source()
    with pytest.raises(ValueError, match="kernel for its action node 'w.push' that would be"):
        merge({"src": src, "w": a}, {"src": sp, "w": ap}, [ActionControl("src.s", "w.u_push")])
    with pytest.raises(ValueError, match="kernel for its action node 'w.push' that would be"):
        merge({"w": a}, {"w": ap}, [BoundaryPrior("w.u_push", 0)])


def test_partner_priors_keyed_by_an_action_node_name_the_control_node():
    (agent, ap), (world, wp) = agent_and_world()
    composite, cp = merge(
        {"agent": agent, "world": world},
        {"agent": ap, "world": wp},
        [ActionControl("agent.u_location", "world.u_location"), *OBSERVATION_BINDINGS],
    )
    with pytest.raises(ValueError, match="the action node of control node 'agent.u_location'"):
        subsystem(composite, cp, "world", partner_priors={"agent.a_location": 0})
    with pytest.raises(ValueError, match="partner_priors names 'world.u_location', a control node"):
        subsystem(composite, cp, "world", partner_priors={"world.u_location": 0})
