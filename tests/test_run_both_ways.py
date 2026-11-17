"""The zoo pair and the arena run both ways (design ``docs/design/nesting-systems.md``).

The doc's acceptance demo, completed: two Systems from the zoo merged and run **both ways**:

- *one mind*: the composite plans every control jointly against one carry, which is the
  flattening's own reading and works with no new machinery at all. Pinned bitwise against the
  hand-flattened baseline, faces and rollout alike.
- *many minds*: the composite coupled to its own mirror (``with_mirror``) and stepped by
  :func:`~hybrid_aif.composition.runner.run`, each component on its own view. **No equality is
  claimed between the two paths**: they are different agents by construction (one models the
  partner's action as exogenous under a declared prior; the other plans it), which is the whole
  point of the particle decomposition. What is asserted is structural sanity, and the many-minds
  runs are held bitwise to what the composite form of ``run`` recorded on the same models
  (``tests/data/migration_driver.npz``) under its key layout (``_old_layout``).

Plus the three invariants the runner exists to keep: requirement 6 (no telepathy: carries are
disjoint pytrees and one mind's carry cannot reach another's beliefs), requirement 5 (tick order
is a real semantic choice), and loud routing errors. The arena is coupled to the mirror of each of
its components (``per_component=True``); its runs under ``"simultaneous"`` and ``"sequential"``
are held to the composite form by ``tests/test_clock_parity.py``.
"""

import dataclasses
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Categorical, ConditionalCategorical
from hybrid_aif.composition import MindCarry, plan, run, sample, with_mirror
from hybrid_aif.composition.merge import Link, merge, subsystem
from hybrid_aif.reference.envs.tmaze import CUE
from hybrid_aif.system import System, SystemParams
from tests.test_merge import OBSERVATION_BINDINGS, agent_and_world, hand_flattened, merged
from tests.test_run_parity import _run_old_keys, ancestral, idle

KEY = jax.random.PRNGKey(0)
FIXTURE = np.load(pathlib.Path(__file__).parent / "data" / "migration_driver.npz")


def same(got, name):
    np.testing.assert_array_equal(np.asarray(got), FIXTURE[name], err_msg=name)


def states(trajectory, factor, node="world"):
    return np.asarray([c[factor] for c in trajectory[node].carries])


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


def zoo_steps(**agent):
    """One step per leaf of the zoo pair coupled to its mirror: the agent plans, the world model
    draws its node from the habit, the mirror is sampled."""
    return {"model.agent": plan(**agent), "model.world": ancestral(), "world": sample()}


# ------------------------------------------------------------------- acceptance


def test_the_acceptance_demo_runs_both_ways():
    """The doc's demo. One-mind is bitwise the hand-flattened baseline; many-minds runs, produces
    sane trajectories (shapes, finiteness, actions in range) and is the composite form's run."""
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

    system, params = with_mirror(composite, cp)
    traj = _run_old_keys(2, system, params, 5, KEY, zoo_steps(), record_carries=True)
    world = traj["world"].carries
    assert len(world) == 6 and set(world[0]) == {f"world.{f}" for f in composite.names.states}
    agent = traj["model.agent"]
    assert list(agent.intended) == ["a_location"] and agent.intended["a_location"].shape == (5,)
    assert bool(jnp.all(agent.intended["a_location"] < 5))
    assert all(v.shape == (6,) for v in agent.received.values()) and len(agent.received) == 3
    for belief in agent.carry.beliefs.values():
        assert bool(jnp.all(jnp.isfinite(belief.probs)))
        assert np.isclose(float(belief.probs.sum()), 1.0, atol=1e-4)
    assert traj["model.world"].carry is None  # the habit draw carries nothing
    for f in composite.names.states:
        same(states(traj, f"world.{f}"), f"acceptance/states/{f}")
    same(agent.intended["a_location"], "acceptance/actions/agent.a_location")
    same(agent.emitted["a_location"], "acceptance/realised/agent.a_location")
    for m, values in agent.received.items():
        same(values, f"acceptance/observations/agent.{m}")


def test_the_many_minds_path_solves_the_tmaze():
    """Sanity beyond well-formedness: the agent mind, planning on its own view with the world's
    move exogenous, goes to the cue first."""
    system, params = with_mirror(*merged(policy_len=3))
    steps = {
        "model.agent": plan(selection="deterministic"),
        "model.world": idle,
        "world": sample(),
    }
    traj = _run_old_keys(1, system, params, 2, KEY, steps, record_carries=True)
    agent = traj["model.agent"]
    assert int(agent.intended["a_location"][0]) == CUE
    same(agent.intended["a_location"], "tmaze/actions/agent.a_location")
    same(agent.emitted["a_location"], "tmaze/realised/agent.a_location")
    for f in ("agent.location", "agent.context", "world.location", "world.context"):
        same(states(traj, f"world.{f}"), f"tmaze/states/{f}")
    for m, values in agent.received.items():
        same(values, f"tmaze/observations/agent.{m}")


def test_two_acting_minds_route_their_own_actions():
    """Both control nodes chosen: the agent's leaf and the world model's leaf each realise their
    own action node, and nothing else. The same run, longer, is held to the composite form by
    ``test_run_parity`` (the zoo pair)."""
    (agent, ap), (world, wp) = agent_and_world(policy_len=1)
    composite, cp = merge(
        {"agent": agent, "world": world}, {"agent": ap, "world": wp}, OBSERVATION_BINDINGS
    )
    system, params = with_mirror(composite, cp)
    traj = run(system, params, 4, KEY, zoo_steps())
    for leaf in ("model.agent", "model.world"):
        assert list(traj[leaf].emitted) == ["a_location"]
        assert traj[leaf].emitted["a_location"].shape == (4,)
        assert bool(jnp.all(traj[leaf].emitted["a_location"] < 5))


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
            state_names=("hand",),
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
        state_names=("token_a", "lock_b", "token_b", "lock_a"),
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
            Link("a.hand", "arena.u_token_a"),
            Link("b.hand", "arena.u_lock_b"),
            Link("b.hand", "arena.u_token_b"),
            Link("a.hand", "arena.u_lock_a"),
            Link(("arena.token_a",), "a.see"),
            Link(("arena.token_b",), "b.see"),
            Link(("b.hand",), "a.watch"),
            Link(("a.hand",), "b.watch"),
        ],
    )


def arena(policy_len=3):
    """The arena coupled to the mirror of each of its components: one leaf per mind, the
    mindless arena a model leaf with nothing to choose, one world leaf per component."""
    return with_mirror(*_arena(policy_len), per_component=True)


def arena_steps():
    return {
        "model.a": plan(selection="deterministic"),
        "model.b": plan(selection="deterministic"),
        "model.arena": idle,
        "world.a": sample(),
        "world.b": sample(),
        "world.arena": sample(),
    }


def world_states(trajectory, composite):
    """Each factor of the arena per round, from the world leaf that holds it."""
    return {
        f: states(trajectory, f"world.{f}", f"world.{f.split('.')[0]}")
        for f in composite.names.states
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
    """Requirement 6, structurally: two minds' carries share no leaf, so there is no object
    through which one mind's posterior could reach another's."""
    system, params = arena()
    traj = run(system, params, 3, KEY, arena_steps())
    leaves = {
        name: {id(x) for x in jax.tree_util.tree_leaves(traj[name].carry)}
        for name in ("model.a", "model.b")
    }
    assert leaves["model.a"] and leaves["model.b"]
    assert not (leaves["model.a"] & leaves["model.b"])


def test_one_minds_carry_cannot_reach_anothers_beliefs():
    """The same invariant behaviourally: seed ``a`` with a different carry and ``b``'s beliefs
    after its first step are bitwise unchanged: the world has not moved yet, so the only way
    ``a`` could have touched them is telepathy."""
    system, params = arena()
    view, vp = subsystem(system, params, "model.a")
    skewed = {
        n: Categorical(logits=jnp.arange(float(b.probs.shape[-1])))
        for n, b in view.initial_priors(vp).items()
    }
    plain = run(system, params, 1, KEY, arena_steps())
    seeded = run(system, params, 1, KEY, arena_steps(), carries={"model.a": MindCarry(skewed)})

    got_b, want_b = seeded["model.b"].carry.beliefs, plain["model.b"].carry.beliefs
    for k in want_b:
        assert jnp.array_equal(got_b[k].probs, want_b[k].probs)
    got_a, want_a = seeded["model.a"].carry.beliefs, plain["model.a"].carry.beliefs
    assert any(not jnp.array_equal(got_a[k].probs, want_a[k].probs) for k in want_a)


# ------------------------------------------------------- requirement 5: tick order


def test_tick_order_is_a_real_semantic_choice():
    """Requirement 5. The same minds, the same key, the same System: simultaneous (every node
    reads the round's boundary, except a world leaf reading its mind's realised action) and
    sequential (each world leaf steps right after the mind that drives it and the arena leaf reads
    the hands as moved) give different trajectories. Same shapes, different game."""
    composite, _ = _arena()
    system, params = arena()
    kw = dict(record_carries=True)
    together = world_states(run(system, params, 4, KEY, arena_steps(), **kw), composite)
    in_turn = world_states(
        run(system, params, 4, KEY, arena_steps(), tick_order="sequential", **kw), composite
    )
    assert {k: s.shape for k, s in together.items()} == {k: s.shape for k, s in in_turn.items()}
    assert any(not np.array_equal(together[k], in_turn[k]) for k in in_turn)
    #  the arena's lock copies b's hand: as moved this round under sequential, as the round began
    #  under simultaneous
    assert np.array_equal(in_turn["arena.lock_b"][1:], in_turn["b.hand"][1:])
    assert np.array_equal(together["arena.lock_b"][1:], together["b.hand"][:-1])
    assert not np.array_equal(together["arena.lock_b"][1:], together["b.hand"][1:])


def test_simultaneous_is_the_default():
    composite, _ = _arena()
    system, params = arena()
    default = run(system, params, 3, KEY, arena_steps(), record_carries=True)
    named = run(
        system, params, 3, KEY, arena_steps(), tick_order="simultaneous", record_carries=True
    )
    want = world_states(named, composite)
    assert all(np.array_equal(v, want[k]) for k, v in world_states(default, composite).items())


# --------------------------------------------------------------- the two steps


def _world_view():
    (agent, ap), (world, wp) = agent_and_world(policy_len=1)
    composite, cp = merge(
        {"agent": agent, "world": world}, {"agent": ap, "world": wp}, OBSERVATION_BINDINGS
    )
    view, vp = subsystem(composite, cp, "world")
    return view, vp


def _drawn(view, params, keys):
    """The world view's free node as ``sample()`` draws it, one draw per key."""
    step = sample()
    _, state, _ = step(view, KEY, None, None, params)
    return [int(step(view, k, {}, state, params)[0]["world.a_location"]) for k in keys]


def test_sample_draws_a_free_node_from_the_declared_habit():
    """The doc's ``planner: None`` case: a generative process over its own control nodes, its
    prior the declared ``policy_prior`` where there is one."""
    view, vp = _world_view()
    pinned = dataclasses.replace(vp, policy_prior=jnp.full(5, -1e9).at[CUE].set(0.0))
    assert _drawn(view, pinned, jax.random.split(KEY, 8)) == [CUE] * 8


def test_sample_draws_a_free_node_uniformly_without_a_habit():
    """With no habit each level of the free node is drawn about equally often; a view whose node
    was merged away has no action node to return."""
    view, vp = _world_view()
    draws = np.bincount(_drawn(view, vp, jax.random.split(KEY, 500)), minlength=5)
    assert draws.min() > 60 and draws.max() < 140
    composite, cp = merged()
    view, vp = subsystem(composite, cp, "world")  # its node was merged away: nothing to draw
    step = sample()
    _, state, _ = step(view, KEY, None, None, vp)
    out, _, _ = step(view, KEY, {}, state, vp)
    assert not set(out) & {a for a in view.action_names if a}


def test_sample_follows_the_habit_in_the_params_it_is_handed():
    view, vp = _world_view()
    step = sample()
    for level in (1, 3, 2):
        pinned = dataclasses.replace(vp, policy_prior=jnp.full(5, -1e9).at[level].set(0.0))
        _, state, _ = step(view, KEY, None, None, pinned)
        for k in jax.random.split(KEY, 4):
            out, state, given = step(view, k, {}, state, pinned)
            assert int(out["world.a_location"]) == level and given is pinned


def test_plan_threads_the_empirical_prior():
    composite, cp = merged()
    view, vp = subsystem(composite, cp, "agent")
    step = plan()
    out, carry, given = step(view, KEY, None, None, vp)  # the reset: nothing out, a fresh carry
    # holding the declared initial priors, and no round run yet
    assert out == {} and given is vp and not bool(carry.started)
    assert carry._replace(beliefs=None, started=None) == MindCarry()
    initial = view.initial_priors(vp)
    for name, belief in carry.beliefs.items():
        np.testing.assert_array_equal(belief.probs, initial[name].probs)
    obs = view.as_env(vp)[0].reset(KEY, view.as_env(vp)[1])[0]
    action, carry, given = step(view, KEY, obs, carry, vp)
    assert given is vp
    assert carry.beliefs is not None and carry.learn_state is None
    zeros = {m: jnp.zeros((), jnp.int32) for m in view.modality_names}
    _, carry2, _ = step(view, KEY, zeros, carry, vp)
    first = view.names.states[0]
    assert not jnp.array_equal(carry2.beliefs[first].probs, carry.beliefs[first].probs)
    assert list(action) == ["agent.a_location"]


# ------------------------------------------------------------------- loud failures


def test_a_leaf_with_no_step_is_refused():
    system, params = arena()
    steps = {k: v for k, v in arena_steps().items() if k != "model.b"}
    with pytest.raises(ValueError, match=r"steps has no step for the leaf 'model.b'"):
        run(system, params, 1, KEY, steps)


def test_routing_errors_name_their_fix():
    system, params = with_mirror(*merged())
    steps = {"model.agnt": idle, "model.world": idle, "world": sample()}
    with pytest.raises(ValueError, match=r"steps names \['model.agnt'\], which are no component"):
        run(system, params, 1, KEY, steps)
    with pytest.raises(ValueError, match="tick_order must be"):
        run(system, params, 1, KEY, zoo_steps(), tick_order="round-robin")
    plain, pp = hand_flattened()
    with pytest.raises(ValueError, match="has no components"):
        run(plain, pp, 1, KEY, {})


def test_a_step_returning_the_wrong_action_keys_raises():
    system, params = with_mirror(*merged())

    def wrong(view, key, obs, carry, params):
        return {"model.agent.u_location": jnp.zeros((), jnp.int32)}, carry, params

    def positional(view, key, obs, carry, params):
        return jnp.zeros(1, jnp.int32), carry, params

    steps = {"model.world": idle, "world": sample()}
    with pytest.raises(ValueError, match=r"its out-ports are \['model.agent.a_location'\]"):
        run(system, params, 1, KEY, {"model.agent": wrong, **steps})
    with pytest.raises(TypeError, match="a step returns a mapping from its out-ports"):
        run(system, params, 1, KEY, {"model.agent": positional, **steps})


def test_async_is_reproducible_and_distinct():
    """``tick_order="async"`` draws a fresh random order of the sources each round
    (sequential as a random scan: the model leaves and the arena leaf, each world leaf placed
    right after the mind that drives it); reproducible under the run key, and a different game
    from both fixed orders on the arena. The composite form's async permuted the two minds
    alone and stepped the arena at a fixed turn, so the two forms' async runs are not compared;
    the comparable orders are held by ``test_clock_parity``."""
    composite, _ = _arena()
    system, params = arena()
    kw = dict(record_carries=True)
    key = jax.random.PRNGKey(6)

    def game(order):
        traj = run(system, params, 6, key, arena_steps(), tick_order=order, **kw)
        return world_states(traj, composite)

    a1, a2 = game("async"), game("async")
    for f in a1:
        np.testing.assert_array_equal(a1[f], a2[f])
    sim, seq = game("simultaneous"), game("sequential")
    assert any(not np.array_equal(a1[f], sim[f]) for f in a1)
    assert any(not np.array_equal(a1[f], seq[f]) for f in a1)


def test_a_bad_order_name_names_all_three():
    system, params = arena()
    with pytest.raises(ValueError, match="'simultaneous', 'sequential', or 'async'"):
        run(system, params, 1, KEY, arena_steps(), tick_order="rando")
