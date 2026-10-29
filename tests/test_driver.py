"""The minimal synchronous driver (design ``docs/design/nesting-systems.md``).

The doc's acceptance demo, completed: two Systems from the zoo merged and run
**both ways** —

- *one mind*: the composite plans every control jointly against one carry, which
  is the flattening's own reading and works with no new machinery at all. Pinned
  bitwise against the chunk-1 hand-flattened baseline, faces and rollout alike.
- *many minds*: each component on its own :func:`~hybrid_aif.composition.merge.subsystem`
  view, interleaved by :func:`~hybrid_aif.composition.driver.run`. **No equality is claimed
  between the two paths** — they are different agents by construction (one models
  the partner's action as exogenous under a declared prior; the other plans it),
  which is the whole point of the particle decomposition. What is asserted is
  structural sanity.

Plus the three invariants the driver exists to keep: requirement 6 (no telepathy —
carries are disjoint pytrees and one mind's carry cannot reach another's beliefs),
requirement 5 (tick order is a real semantic choice), and loud routing errors.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Categorical, ConditionalCategorical
from hybrid_aif.composition.driver import (
    MindCarry,
    agent_respond,
    ancestral_respond,
    run,
)
from hybrid_aif.composition.merge import ActionControl, ObservationLink, merge, subsystem
from hybrid_aif.reference.envs.tmaze import CUE
from hybrid_aif.system import System, SystemParams
from tests.test_merge import OBSERVATION_BINDINGS, agent_and_world, hand_flattened, merged

KEY = jax.random.PRNGKey(0)


def one_mind_rollout(system, params, key, ticks):
    """The one-mind path: one carry, one planner, every control planned jointly."""
    env, ep = system.as_env(params)
    key, k_reset = jax.random.split(key)
    obs, state = env.reset(k_reset, ep)
    beliefs, states, actions = None, [state], []
    for _ in range(ticks):
        key, k_act, k_env = jax.random.split(key, 3)
        beliefs, _ = system.infer_states(params, obs, priors=beliefs)
        action, _, _ = system.act(k_act, params, beliefs, metrics=False)
        beliefs = system.update_empirical_prior(params, beliefs, action)
        obs, state = env.step(k_env, state, action, ep)
        states.append(state)
        actions.append(action)
    return states, actions


def zoo_minds(composite, params, **kwargs):
    """One mind per component of the zoo pair: the agent plans, the world responds."""
    agent_view = subsystem(composite, params, "agent", **kwargs)
    world_view = subsystem(composite, params, "world", **kwargs)
    return {
        "agent": agent_respond(*agent_view),
        "world": ancestral_respond(*world_view),
    }


# ------------------------------------------------------------------- acceptance


def test_the_acceptance_demo_runs_both_ways():
    """The doc's demo. One-mind is bitwise the hand-flattened baseline; many-minds
    runs and produces sane trajectories — shapes, finiteness, actions in range."""
    composite, cp = merged()
    hand, hp = hand_flattened()

    got_states, got_actions = one_mind_rollout(composite, cp, KEY, 5)
    want_states, want_actions = one_mind_rollout(hand, hp, KEY, 5)
    for a, b in zip(got_states, want_states, strict=True):
        assert set(a) == set(b) and all(jnp.array_equal(a[k], b[k]) for k in b)
    for a, b in zip(got_actions, want_actions, strict=True):
        assert jnp.array_equal(
            composite.to_positional("actions", a), hand.to_positional("actions", b)
        )

    traj = run(composite, cp, zoo_minds(composite, cp), ticks=5, key=KEY)
    assert len(traj.states) == 4 and len(traj.observations) == 3
    assert all(s.shape == (6,) for s in traj.states.values())
    assert all(o.shape == (6,) for o in traj.observations.values())
    actions = composite.to_positional("actions", traj.actions)
    states = composite.to_positional("beliefs", traj.states)
    assert actions.shape == (5, 1)  # one column per action node: the agent's
    assert bool(jnp.all(actions[:, 0] < 5))
    assert bool(jnp.all(states[0] < 5)) and bool(jnp.all(states[1] < 2))
    for belief in traj.carries["agent"].beliefs.values():
        assert bool(jnp.all(jnp.isfinite(belief.probs)))
        assert np.isclose(float(belief.probs.sum()), 1.0, atol=1e-4)
    assert traj.carries["world"].beliefs is None  # the ancestral responder carries nothing


def test_the_many_minds_path_still_solves_the_tmaze():
    """Sanity beyond well-formedness: the agent mind, planning on its own view with
    the world's move exogenous, still goes to the cue first."""
    composite, cp = merged(policy_len=3)
    minds = {"agent": agent_respond(*subsystem(composite, cp, "agent"), selection="deterministic")}
    traj = run(composite, cp, minds, ticks=2, key=KEY)
    assert int(traj.actions["agent.a_location"][0]) == CUE


def test_two_acting_minds_route_their_own_actions():
    """Both control nodes chosen: the agent's mind and the world's mind each fill
    their own entry of the composite action vector, and nothing else."""
    (agent, ap), (world, wp) = agent_and_world(policy_len=1)
    composite, cp = merge(
        {"agent": agent, "world": world}, {"agent": ap, "world": wp}, OBSERVATION_BINDINGS
    )
    minds = {
        "agent": agent_respond(*subsystem(composite, cp, "agent")),
        "world": ancestral_respond(*subsystem(composite, cp, "world")),
    }
    traj = run(composite, cp, minds, ticks=4, key=KEY)
    actions = composite.to_positional("actions", traj.actions)
    assert actions.shape == (4, 2)  # the agent's node, the world's node
    assert bool(jnp.all(actions[:, 0] < 5)) and bool(jnp.all(actions[:, 1] < 5))


# --------------------------------------------------------- requirement 6: privacy


def _arena(policy_len=3):
    """Two minds, each scoring a token that is the sum of both hands, mod 3.

    ``arena.token_a`` becomes ``a``'s hand plus ``arena.lock_b``, which copies ``b``'s hand;
    ``arena.token_b`` the mirror. Every influence is a state relay, read a step later, so a
    mind's move lands with the other's *current* hand, which it watches: ``a`` wants its
    token at 2, ``b`` wants its own at 1, and each mind's best move is a function of what
    it sees in the other's hand. Two minds, opposed preferences, and each reading the
    other's move fresh or stale by the order it responds in — which is what makes tick
    order bite.
    """
    move = jnp.zeros((3, 3, 3)).at[jnp.arange(3), :, jnp.arange(3)].set(1.0)
    token = np.zeros((3, 3, 3, 3), np.float32)
    for t in range(3):
        for lock in range(3):
            for u in range(3):
                token[(u + lock) % 3, t, lock, u] = 1.0

    def mind(prefers):
        system = System(
            A_deps=((0,), (0,)),
            B_deps=((0,),),
            controlled=(True,),
            policy_len=policy_len,
            epistemic=False,
            modality_names=("see", "watch"),
            factor_names=("hand",),
        )
        exact = ConditionalCategorical(log_table=jnp.log(jnp.eye(3)), parent_ndims=1)
        params = SystemParams(
            likelihood={"see": exact, "watch": exact},
            transition={"hand": ConditionalCategorical(log_table=jnp.log(move), parent_ndims=2)},
            initial_state_prior={"hand": Categorical.from_probs(jnp.eye(3)[0])},
            obs_targets={"see": Categorical(logits=jnp.zeros(3).at[prefers].set(4.0))},
        )
        return system, params

    sum_table = ConditionalCategorical(log_table=jnp.log(jnp.asarray(token)), parent_ndims=3)
    copy_table = ConditionalCategorical(log_table=jnp.log(move), parent_ndims=2)
    arena = System(
        A_deps=(),
        B_deps=((0, 1), (1,), (2, 3), (3,)),
        controlled=(True, True, True, True),
        policy_len=policy_len,
        epistemic=False,
        modality_names=(),
        factor_names=("token_a", "lock_b", "token_b", "lock_a"),
    )
    arena_params = SystemParams(
        likelihood={},
        transition={
            "token_a": sum_table,
            "lock_b": copy_table,
            "token_b": sum_table,
            "lock_a": copy_table,
        },
        initial_state_prior={f: Categorical.from_probs(jnp.eye(3)[0]) for f in arena.names.states},
    )
    a, apar = mind(2)
    b, bpar = mind(1)
    return merge(
        {"a": a, "b": b, "arena": arena},
        {"a": apar, "b": bpar, "arena": arena_params},
        [
            ActionControl(driver="a.hand", control="arena.u_token_a"),
            ActionControl(driver="b.hand", control="arena.u_lock_b"),
            ActionControl(driver="b.hand", control="arena.u_token_b"),
            ActionControl(driver="a.hand", control="arena.u_lock_a"),
            ObservationLink(modality="a.see", factors=("arena.token_a",)),
            ObservationLink(modality="b.see", factors=("arena.token_b",)),
            ObservationLink(modality="a.watch", factors=("b.hand",)),
            ObservationLink(modality="b.watch", factors=("a.hand",)),
        ],
    )


def arena_minds(composite, params):
    return {
        name: agent_respond(*subsystem(composite, params, name), selection="deterministic")
        for name in ("a", "b")
    }


def test_the_arena_gives_both_minds_the_other_as_an_exogenous_stub():
    composite, cp = _arena()
    assert composite.names.states == (
        "a.hand",
        "b.hand",
        "arena.token_a",
        "arena.lock_b",
        "arena.token_b",
        "arena.lock_a",
    )
    view, vp = subsystem(composite, cp, "a")
    #  its own hand, the hand it watches, and the arena factors it reads or is read by
    assert view.names.states == ("a.hand", "b.hand", "arena.token_a", "arena.lock_b")
    assert view.controlled == (True, False, False, False)  # b's node is not ours to plan
    assert view.names.controls == ("a.u_hand",)
    assert set(vp.obs_targets) == {"a.see"}


def test_carries_are_disjoint_pytrees():
    """Requirement 6, structurally: two minds' carries share no leaf, so there is no
    object through which one mind's posterior could reach another's."""
    composite, cp = _arena()
    traj = run(composite, cp, arena_minds(composite, cp), ticks=3, key=KEY)
    leaves = {
        name: {id(x) for x in jax.tree_util.tree_leaves(carry)}
        for name, carry in traj.carries.items()
    }
    assert leaves["a"] and leaves["b"]
    assert not (leaves["a"] & leaves["b"])


def test_one_minds_carry_cannot_reach_anothers_beliefs():
    """The same invariant behaviourally: seed ``a`` with a different carry and ``b``'s
    beliefs after its first response are bitwise unchanged — the world has not moved
    yet, so the only way ``a`` could have touched them is telepathy."""
    composite, cp = _arena()
    minds = arena_minds(composite, cp)
    view, vp = subsystem(composite, cp, "a")
    skewed = {
        n: Categorical(logits=jnp.arange(float(b.probs.shape[-1])))
        for n, b in view.initial_priors(vp).items()
    }
    plain = run(composite, cp, minds, ticks=1, key=KEY)
    seeded = run(composite, cp, minds, ticks=1, key=KEY, carries={"a": MindCarry(beliefs=skewed)})

    got_b, want_b = seeded.carries["b"].beliefs, plain.carries["b"].beliefs
    for k in want_b:
        assert jnp.array_equal(got_b[k].probs, want_b[k].probs)
    got_a, want_a = seeded.carries["a"].beliefs, plain.carries["a"].beliefs
    assert any(not jnp.array_equal(got_a[k].probs, want_a[k].probs) for k in want_a)


# ------------------------------------------------------- requirement 5: tick order


def test_tick_order_is_a_real_semantic_choice():
    """Requirement 5. The same minds, the same key, the same fixture: simultaneous
    (every mind reads the tick's boundary; nobody sees anybody's move until next tick)
    and sequential (``a``'s hand steps at ``a``'s turn, so ``b`` watches it fresh and the
    arena reads both hands as moved) give different trajectories. Same shapes,
    different game."""
    composite, cp = _arena()
    minds = arena_minds(composite, cp)
    together = run(composite, cp, minds, ticks=4, key=KEY, tick_order="simultaneous")
    in_turn = run(composite, cp, minds, ticks=4, key=KEY, tick_order="sequential")

    assert {k: s.shape for k, s in together.states.items()} == {
        k: s.shape for k, s in in_turn.states.items()
    }
    assert {k: v.shape for k, v in together.actions.items()} == {
        k: v.shape for k, v in in_turn.actions.items()
    }
    assert any(not jnp.array_equal(together.states[k], in_turn.states[k]) for k in in_turn.states)
    #  the arena has no mind, so it steps at the end of the round and reads the hands as moved
    #  this round; under simultaneous it reads the boundary's
    lock_b, b_hand = in_turn.states["arena.lock_b"], in_turn.states["b.hand"]
    assert jnp.array_equal(lock_b[1:], b_hand[1:])
    assert not jnp.array_equal(together.states["arena.lock_b"][1:], together.states["b.hand"][1:])


def test_simultaneous_is_the_default():
    composite, cp = _arena()
    minds = arena_minds(composite, cp)
    default = run(composite, cp, minds, ticks=3, key=KEY)
    named = run(composite, cp, minds, ticks=3, key=KEY, tick_order="simultaneous")
    assert all(jnp.array_equal(default.states[k], named.states[k]) for k in named.states)


# --------------------------------------------------------------- the two responders


def test_ancestral_respond_draws_from_the_declared_habit():
    """The doc's ``planner: None`` case: a feedforward generative process over its
    own control nodes, its prior the declared ``policy_prior`` where there is one."""
    (agent, ap), (world, wp) = agent_and_world(policy_len=1)
    composite, cp = merge(
        {"agent": agent, "world": world}, {"agent": ap, "world": wp}, OBSERVATION_BINDINGS
    )
    view, vp = subsystem(composite, cp, "world")
    pinned = dataclasses.replace(vp, policy_prior=jnp.full(5, -1e9).at[CUE].set(0.0))
    respond = ancestral_respond(view, pinned)
    for k in jax.random.split(KEY, 8):
        action, carry = respond(k, {}, MindCarry())
        assert int(action["world.a_location"]) == CUE
        assert carry == MindCarry()


def test_ancestral_respond_is_uniform_without_a_habit():
    composite, cp = merged()
    view, vp = subsystem(composite, cp, "world")  # its node was merged away: nothing to draw
    respond = ancestral_respond(view, vp)
    action, _ = respond(KEY, {}, MindCarry())
    assert jnp.array_equal(view.to_positional("actions", action), jnp.zeros(0, jnp.int32))


def test_agent_respond_threads_the_empirical_prior():
    composite, cp = merged()
    view, vp = subsystem(composite, cp, "agent")
    respond = agent_respond(view, vp)
    action, carry = respond(KEY, view.as_env(vp)[0].reset(KEY, view.as_env(vp)[1])[0], MindCarry())
    assert carry.beliefs is not None and carry.learn_state is None
    _, carry2 = respond(KEY, {m: jnp.zeros((), jnp.int32) for m in view.modality_names}, carry)
    first = view.names.states[0]
    assert not jnp.array_equal(carry2.beliefs[first].probs, carry.beliefs[first].probs)
    assert view.to_positional("actions", action).shape == (1,)


# ------------------------------------------------------------------- loud failures


def test_an_unchosen_control_node_raises():
    composite, cp = _arena()
    minds = {"a": agent_respond(*subsystem(composite, cp, "a"))}
    with pytest.raises(ValueError, match=r"control node\(s\) \['b.u_hand'\] are chosen by no mind"):
        run(composite, cp, minds, ticks=1, key=KEY)


def test_routing_errors_name_their_fix():
    composite, cp = merged()
    with pytest.raises(ValueError, match="not components of this composite"):
        run(composite, cp, {"agnt": ancestral_respond(*subsystem(composite, cp, "world"))}, 1, KEY)
    with pytest.raises(ValueError, match="tick_order must be"):
        run(composite, cp, zoo_minds(composite, cp), 1, KEY, tick_order="round-robin")
    plain, pp = hand_flattened()
    with pytest.raises(ValueError, match="routes by the subsystems metadata"):
        run(plain, pp, {}, 1, KEY)


def test_a_mind_returning_the_wrong_action_keys_raises():
    composite, cp = merged()

    def wrong(key, obs, carry):
        return {"u_agent.context": jnp.zeros((), jnp.int32)}, carry

    def positional(key, obs, carry):
        return jnp.zeros(1, jnp.int32), carry

    with pytest.raises(ValueError, match=r"its action nodes are \['agent.a_location'\]"):
        run(composite, cp, {"agent": wrong}, ticks=1, key=KEY)
    with pytest.raises(TypeError, match="returned a positional action"):
        run(composite, cp, {"agent": positional}, ticks=1, key=KEY)


def test_async_mode_is_reproducible_and_distinct():
    """tick_order='async': a random permutation per sweep — reproducible under
    the run key, and a genuinely different game from both fixed orders on the
    arena fixture (the equilibrium-faithful mode; see the driver docstring)."""
    composite, params = _arena()
    minds = arena_minds(composite, params)
    kw = dict(ticks=6, key=jax.random.PRNGKey(6))
    a1 = run(composite, params, minds, tick_order="async", **kw)
    a2 = run(composite, params, minds, tick_order="async", **kw)
    for f in a1.states:
        np.testing.assert_array_equal(np.asarray(a1.states[f]), np.asarray(a2.states[f]))
    sim = run(composite, params, minds, tick_order="simultaneous", **kw)
    seq = run(composite, params, minds, tick_order="sequential", **kw)
    diff_sim = any(
        not np.array_equal(np.asarray(a1.states[f]), np.asarray(sim.states[f])) for f in a1.states
    )
    diff_seq = any(
        not np.array_equal(np.asarray(a1.states[f]), np.asarray(seq.states[f])) for f in a1.states
    )
    assert diff_sim and diff_seq, "async must differ from both fixed orders on the arena"


def test_async_rejects_nothing_new_and_bad_order_names_all_three():
    composite, params = _arena()
    minds = arena_minds(composite, params)
    with pytest.raises(ValueError, match="'simultaneous', 'sequential', or 'async'"):
        run(composite, params, minds, ticks=1, key=jax.random.PRNGKey(0), tick_order="rando")
