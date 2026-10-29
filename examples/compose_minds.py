"""Two robots, two charging docks: one mind by ``merge()``, two minds by ``couple()``.

A corridor of ``n`` cells with a charging dock at each end. Two robots move one cell a tick
(left, stay or right; a move into a wall stays) and the task is to end up one robot per dock.
Each robot's own model is one :class:`System`:

    factors     me       {cell}   moved by the robot's action node ``move``
                other    {cell}   the other robot, modelled as a random walk
    modalities  me_obs, other_obs   the two positions, exact
                status   {apart on the docks, both on one dock, otherwise}, over (me, other)
    preference  apart on the docks +4, both on one dock -2, otherwise 0

The two robots' models are composed two ways:

- **merged** (:func:`merged`): ``merge()`` makes the two models one ``System`` whose
  modalities read the robots' own position factors (each model's ``other`` goes unread), and
  one mind plans over both robots' moves jointly, acting on a whole joint policy
  (``act(..., mode="policy")``; on each node's own marginal, ``act``'s default, the two
  dock assignments tie node by node and the one mind behaves as the coupled pair does);
- **coupled** (:func:`coupled_episode`): each robot is a mind of its own, its model as
  declared, wired to a world process through ``couple()``, as ``hide_and_seek.py`` does.

From a symmetric start (both robots in the middle cell) the joint planner assigns the docks
at once. Each coupled mind wants a dock and values the two equally (the problem is
symmetric), so any deterministic rule breaks the tie the same way for both and they never
part; stochastic selection breaks it by chance. From an asymmetric start the two
compositions behave alike.

Run as a script it prints the identity check and the comparison::

    .venv/bin/python examples/compose_minds.py

Design: docs/design/nesting-systems.md; docs/composing-systems.md
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from hybrid_aif import Categorical, ConditionalCategorical
from hybrid_aif.composition import ObservationLink, Wire, agent_respond, couple, merge, run
from hybrid_aif.system import System, SystemParams

N = 5  # corridor cells; the docks are the two ends
MOVES = (-1, 0, 1)  # left, stay, right
STATUS = ("apart on the docks", "both on one dock", "otherwise")
PREFERENCE = (4.0, -2.0, 0.0)
SYMMETRIC, ASYMMETRIC = (2, 2), (1, 2)  # starting cells (robot a, robot b)


def _cc(table, parent_ndims: int) -> ConditionalCategorical:
    return ConditionalCategorical(
        log_table=jnp.log(jnp.clip(jnp.asarray(table), 1e-16)), parent_ndims=parent_ndims
    )


def _one_hot(cell: int, n: int) -> Categorical:
    return Categorical.from_probs(jnp.eye(n)[cell])


def step(cell: int, move: int, n: int = N) -> int:
    """The cell a move lands on; a move into a wall stays."""
    return min(max(cell + MOVES[move], 0), n - 1)


def own_moves(n: int = N) -> np.ndarray:
    """``(cell', cell, move)``: a robot's own move, exact."""
    table = np.zeros((n, n, len(MOVES)))
    for cell in range(n):
        for a in range(len(MOVES)):
            table[step(cell, a, n), cell, a] = 1.0
    return table


def random_walk(n: int = N) -> np.ndarray:
    """``(other', other, me, 1)``: the other robot as a robot models it, each move equally
    likely whatever either robot does."""
    table = np.zeros((n, n, n, 1))
    for other in range(n):
        for a in range(len(MOVES)):
            table[step(other, a, n), other, :, 0] += 1.0 / len(MOVES)
    return table


def status_table(n: int = N) -> np.ndarray:
    """``(status, me, other)``: the two robots apart on the two docks, both on one dock, or
    neither, exact."""
    docks = (0, n - 1)
    table = np.zeros((len(STATUS), n, n))
    for a in range(n):
        for b in range(n):
            if a in docks and b in docks and a != b:
                table[0, a, b] = 1.0
            elif a == b and a in docks:
                table[1, a, b] = 1.0
            else:
                table[2, a, b] = 1.0
    return table


def robot(me: int, other: int, policy_len: int = 2) -> tuple[System, SystemParams]:
    """One robot's own model, starting at ``me`` and believing the other at ``other``."""
    system = System(
        A_deps=((0,), (1,), (0, 1)),
        B_deps=((0, 2), (1, 0), ()),
        controls=(2,),
        action_names=("move",),
        factor_names=("me", "other", "u_move"),
        modality_names=("me_obs", "other_obs", "status"),
        policy_len=policy_len,
        gamma=16.0,
        epistemic=False,
    )
    params = SystemParams(
        likelihood={
            "me_obs": _cc(np.eye(N), 1),
            "other_obs": _cc(np.eye(N), 1),
            "status": _cc(status_table(), 2),
        },
        transition={"me": _cc(own_moves(), 2), "other": _cc(random_walk(), 3)},
        initial_state_prior={"me": _one_hot(me, N), "other": _one_hot(other, N)},
        obs_targets={"status": Categorical(logits=jnp.asarray(PREFERENCE))},
        state_targets={},
        terminal_targets={},
    )
    return system, params


# ------------------------------------------------------------------ merged: one mind

LINKS = (
    ObservationLink(modality="a.other_obs", factors=("b.me",)),
    ObservationLink(modality="a.status", factors=("a.me", "b.me")),
    ObservationLink(modality="b.other_obs", factors=("a.me",)),
    ObservationLink(modality="b.status", factors=("b.me", "a.me")),
)


def merged(start=SYMMETRIC, policy_len: int = 2) -> tuple[System, SystemParams]:
    """The two robots' models as one System: every modality reads the robots' own position
    factors, and the composite plans over both action nodes."""
    a, b = start
    (sa, pa), (sb, pb) = robot(a, b, policy_len), robot(b, a, policy_len)
    return merge({"a": sa, "b": sb}, {"a": pa, "b": pb}, list(LINKS))


def longhand(start=SYMMETRIC, policy_len: int = 2) -> tuple[System, SystemParams]:
    """The merged System written out with no merge machinery: the six nodes in the
    composite's order (each robot's two state factors, then the two control nodes), each
    robot's modalities over the robots' own positions."""
    a, b = start
    (_, pa), (_, pb) = robot(a, b, policy_len), robot(b, a, policy_len)
    system = System(
        A_deps=((0,), (2,), (0, 2), (2,), (0,), (2, 0)),
        B_deps=((0, 4), (1, 0), (2, 5), (3, 2), (), ()),
        controls=(4, 5),
        action_names=("a.move", "b.move"),
        factor_names=("a.me", "a.other", "b.me", "b.other", "a.u_move", "b.u_move"),
        modality_names=(
            "a.me_obs",
            "a.other_obs",
            "a.status",
            "b.me_obs",
            "b.other_obs",
            "b.status",
        ),
        policy_len=policy_len,
        gamma=16.0,
        epistemic=False,
    )
    params = SystemParams(
        likelihood={
            **{f"a.{m}": t for m, t in pa.likelihood.items()},
            **{f"b.{m}": t for m, t in pb.likelihood.items()},
        },
        transition={
            **{f"a.{f}": t for f, t in pa.transition.items()},
            **{f"b.{f}": t for f, t in pb.transition.items()},
        },
        initial_state_prior={
            **{f"a.{f}": d for f, d in pa.initial_state_prior.items()},
            **{f"b.{f}": d for f, d in pb.initial_state_prior.items()},
        },
        obs_targets={"a.status": pa.obs_targets["status"], "b.status": pb.obs_targets["status"]},
        state_targets={},
        terminal_targets={},
    )
    return system, params


def identity_check(start=SYMMETRIC) -> dict[str, bool]:
    """Whether the merged System is the longhand one: every static field but the
    ``subsystems`` record, every parameter array, and the beliefs and policy scores the two
    return on the first observation, each compared bitwise."""
    comp, cp = merged(start)
    hand, hp = longhand(start)
    fields = [f.name for f in dataclasses.fields(System) if f.name != "subsystems"]
    got, want = jax.tree_util.tree_flatten(cp), jax.tree_util.tree_flatten(hp)
    a, b = start
    obs = {"a.me_obs": a, "a.other_obs": b, "b.me_obs": b, "b.other_obs": a}
    obs = {k: jnp.asarray(v) for k, v in obs.items()}
    obs |= {"a.status": jnp.asarray(2), "b.status": jnp.asarray(2)}
    beliefs_c, _ = comp.infer_states(cp, obs)
    beliefs_h, _ = hand.infer_states(hp, obs)
    q_c, _ = comp.infer_policies(cp, beliefs_c)
    q_h, _ = hand.infer_policies(hp, beliefs_h)
    return {
        "static fields": all(getattr(comp, f) == getattr(hand, f) for f in fields),
        "parameter arrays": got[1] == want[1]
        and all(bool(jnp.array_equal(x, y)) for x, y in zip(got[0], want[0], strict=True)),
        "beliefs": all(
            bool(jnp.array_equal(beliefs_c[n].probs, q.probs)) for n, q in beliefs_h.items()
        ),
        "policy scores": bool(jnp.array_equal(q_c.logits, q_h.logits)),
    }


@eqx.filter_jit
def _merged_tick(system, params, key, obs, beliefs, selection, mode):
    beliefs, _ = system.infer_states(params, obs, priors=beliefs)
    action, _, _ = system.act(key, params, beliefs, selection=selection, metrics=False, mode=mode)
    return action, system.update_empirical_prior(params, beliefs, action)


def merged_episode(
    key, start=SYMMETRIC, selection="deterministic", ticks: int = 8, mode: str = "policy"
):
    """One episode of the one mind: the merged System infers, acts and is stepped as its own
    world. ``mode="policy"`` acts on a whole joint policy; ``mode="marginal"`` (``act``'s
    default for discrete controls) on each action node's own first-step marginal. Returns
    each robot's cell per tick, ``(ticks + 1,)``."""
    system, params = merged(start)
    env, env_params = system.as_env(params)
    key, k_reset = jax.random.split(key)
    obs, state = env.reset(k_reset, env_params)
    beliefs, cells = None, [(int(state["a.me"]), int(state["b.me"]))]
    for _ in range(ticks):
        key, k_act, k_env = jax.random.split(key, 3)
        action, beliefs = _merged_tick(system, params, k_act, obs, beliefs, selection, mode)
        obs, state = env.step(k_env, state, action, env_params)
        cells.append((int(state["a.me"]), int(state["b.me"])))
    path = np.asarray(cells)
    return path[:, 0], path[:, 1]


# --------------------------------------------------------------- coupled: two minds


def world(start=SYMMETRIC):
    """The corridor as a process: both positions, each moved by its own action input,
    emitting the two positions and the status."""
    system = System(
        A_deps=((0,), (1,), (0, 1)),
        B_deps=((0, 2), (1, 3), (), ()),
        controls=(2, 3),
        action_names=("a_move", "b_move"),
        factor_names=("a", "b", "u_a", "u_b"),
        modality_names=("a_pos", "b_pos", "status"),
        policy_len=1,
        epistemic=False,
    )
    params = SystemParams(
        likelihood={
            "a_pos": _cc(np.eye(N), 1),
            "b_pos": _cc(np.eye(N), 1),
            "status": _cc(status_table(), 2),
        },
        transition={"a": _cc(own_moves(), 2), "b": _cc(own_moves(), 2)},
        initial_state_prior={"a": _one_hot(start[0], N), "b": _one_hot(start[1], N)},
    )
    env, env_params = system.as_env(params)
    return env, dataclasses.replace(env_params, actuator={})


WIRES = (
    Wire("a.move", "world.a_move"),
    Wire("b.move", "world.b_move"),
    Wire("world.a_pos", "a.me_obs"),
    Wire("world.b_pos", "a.other_obs"),
    Wire("world.status", "a.status"),
    Wire("world.b_pos", "b.me_obs"),
    Wire("world.a_pos", "b.other_obs"),
    Wire("world.status", "b.status"),
)


def coupled(start=SYMMETRIC, selection="deterministic", policy_len: int = 2):
    """The two robots as two minds, each its own model, wired to the corridor."""
    a, b = start
    ra, rb = robot(a, b, policy_len), robot(b, a, policy_len)
    return couple(
        minds={
            "a": (*ra, agent_respond(*ra, selection=selection)),
            "b": (*rb, agent_respond(*rb, selection=selection)),
        },
        processes={"world": world(start)},
        wires=list(WIRES),
    )


def coupled_episode(key, start=SYMMETRIC, selection="deterministic", ticks: int = 8):
    """One episode of the two minds through ``run``. Returns each robot's cell per tick."""
    record = run(coupled(start, selection), ticks=ticks, key=key)
    states = record["world"].states
    return np.asarray(states["a"]), np.asarray(states["b"])


# ----------------------------------------------------------------------- the comparison


def split_tick(path_a, path_b) -> int | None:
    """The first tick at which the robots are apart on the two docks, ``None`` if never."""
    for t, (a, b) in enumerate(zip(path_a, path_b, strict=True)):
        if {int(a), int(b)} == {0, N - 1}:
            return t
    return None


def split_by_tick(episode, episodes: int, ticks: int = 8, seed: int = 0, **kwargs):
    """The fraction of ``episodes`` (keys ``seed`` onward) split by each tick, ``(ticks + 1,)``,
    and the first split tick of each episode (``-1`` for never)."""
    first = []
    for i in range(episodes):
        t = split_tick(*episode(jax.random.PRNGKey(seed + i), ticks=ticks, **kwargs))
        first.append(-1 if t is None else t)
    first = np.asarray(first)
    by_tick = np.asarray([np.mean((first >= 0) & (first <= t)) for t in range(ticks + 1)])
    return by_tick, first


def policy_counts(policy_len: int = 2) -> dict[str, int]:
    """How many policies each planner scores a tick: the merged mind's joint ones, and one
    coupled mind's own."""
    comp, cp = merged(SYMMETRIC, policy_len)
    one, op = robot(*SYMMETRIC, policy_len)
    return {"merged": int(comp.policies(cp).shape[0]), "coupled": int(one.policies(op).shape[0])}


def main() -> None:
    for claim, ok in identity_check().items():
        print(f"merged vs longhand, {claim}: bitwise {ok}")
    counts = policy_counts()
    print(f"policies per tick: merged {counts['merged']}, each coupled mind {counts['coupled']}")
    for start in (SYMMETRIC, ASYMMETRIC):
        print(f"start {start}:")
        path = merged_episode(jax.random.PRNGKey(0), start)
        print(f"  merged, deterministic: split at tick {split_tick(*path)}")
        path = coupled_episode(jax.random.PRNGKey(0), start)
        print(f"  coupled, deterministic: split at tick {split_tick(*path)}")
        _, first = split_by_tick(coupled_episode, 32, start=start, selection="stochastic")
        print(f"  coupled, stochastic: first split ticks over 32 episodes {first.tolist()}")


if __name__ == "__main__":
    main()
