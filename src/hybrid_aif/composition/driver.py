"""The driver: minds and processes on a clock, in either of two forms.

A mind is a :data:`RespondFn`, ``respond(key, obs, carry) -> (action, carry')``.
:func:`agent_respond` and :func:`ancestral_respond` build the two ends of that spectrum.
:func:`run` closes the loop in one of two forms:

- **Wired** (:func:`~hybrid_aif.composition.couple.couple`): minds and processes joined by
  typed wires, from an action node to a process input or a mind's modality, and from a process
  observation to a mind's modality. Each action node is realised from its mind's
  ``params.actuator`` when the mind chooses, and every wire from it carries the realised value;
  the processes step on their realised inputs. The record is one entry per participant,
  :class:`MindRecord` or :class:`ProcessRecord`.
- **Composite** (:func:`~hybrid_aif.composition.merge.merge`): one process, the composite's
  ``as_env``, with the wires by name: observations route in by the ``subsystems`` metadata
  (each mind sees its own modalities, keyed by name); actions route out by action-node name,
  each mind's action nodes merged into the composite action, realised at its nodes, on which the
  world moves. The record is a :class:`Trajectory`.

In both forms a realised action is drawn once, when its mind chooses, and every wire from the
node carries it for the rest of the round; no other mind's choice redraws it.

Beliefs never route: each mind's posterior lives in its own carry, and minds reach each other
only through the wires (in the composite form, through the world).

A round is one tick of time under every order, and every factor steps once per round; the
orders are algorithms for one update, not amounts of time. ``"simultaneous"`` (default): every
mind responds to the round's boundary, then every process (in the composite form, the world)
steps once on the round's actions. ``"sequential"``: minds respond in declaration order (the
order given to :func:`~hybrid_aif.composition.couple.couple`, or the order of ``minds`` in the
composite form), each reading current values, fresh where the source has already stepped this
round: in the wired form a direct wire carries an earlier mind's act to a later mind, and the
processes step once at the end of the round; in the composite form each component's factors
step at that component's turn, and a component with no mind steps at the end. ``"async"``: as
sequential, in a fresh random order per round (reproducible from the run key). Each records
one row per tick. The driver is a Python loop, so carries need no shared structure; each
``respond`` may be jitted.

Design: docs/design/nesting-systems.md (architecture sketch; tick semantics);
docs/design/control-states.md C13.
"""

import dataclasses
from collections.abc import Callable, Mapping
from typing import Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hybrid_aif.composition.couple import Wiring
from hybrid_aif.controls import FLOAT
from hybrid_aif.distributions import Categorical
from hybrid_aif.envs.generative import ACTUATOR_TAG
from hybrid_aif.planning.base import Planner
from hybrid_aif.system import Belief, LearnState, System, SystemParams, keyed

TickOrder = Literal["simultaneous", "sequential", "async"]


class MindCarry(NamedTuple):
    """One mind's internal state between ticks, the only place its beliefs live.

    ``beliefs`` is ``None`` before the first tick (use the declared initial priors); ``learn_state``
    holds the carry of ``System.learn`` once a mind learns online. Carries are never merged, so two
    minds share no leaf.
    """

    beliefs: Mapping[str, Belief] | None = None
    learn_state: LearnState | None = None


#: The driver protocol: ``obs`` is this mind's own modalities keyed by name, ``action`` an int32
#: level per action node of its *view*, keyed by action-node name as :meth:`System.act`
#: returns it.
RespondFn = Callable[[Array, Mapping[str, Array], MindCarry], tuple[Mapping[str, Array], MindCarry]]


def agent_respond(
    system: System,
    params: SystemParams,
    planner: Planner | None = None,
    **act_kwargs,
) -> RespondFn:
    """A full agent as a :data:`RespondFn`: infer, act, carry the empirical prior.

    Closed over one mind's view (typically :func:`hybrid_aif.composition.merge.subsystem`'s
    output): ``infer_states`` with the carried beliefs as priors, ``act``, then
    ``update_empirical_prior`` under the chosen action. ``act_kwargs`` pass through to
    :meth:`System.act`; a ``legal`` among them (keyed by control node, or one bool per policy) is
    checked once, here. A System with no designations gives a Bayesian perceiver (flat ``q_pi``,
    the same filtering).
    """

    legal = act_kwargs.get("legal")
    if (
        legal is not None
        and system._control_layout(system._params_in(params, "agent_respond")) is None
        and not isinstance(legal, jax.core.Tracer)
    ):
        #  inside the compiled tick the mask is traced and a level set that masks every
        #  level cannot be refused; refuse it here, once, on the concrete value
        system._legal_logits(
            system._params_in(params, "agent_respond"),
            legal,
            system.policies(params),
            "agent_respond",
        )
    kwargs = tuple(sorted(act_kwargs.items()))

    def respond(
        key: Array, obs: Mapping[str, Array], carry: MindCarry
    ) -> tuple[dict[str, Array], MindCarry]:
        return _agent_tick(system, params, planner, kwargs, key, obs, carry)

    return respond


@eqx.filter_jit
def _agent_tick(system, params, planner, kwargs, key, obs, carry):
    """One compiled tick: infer, act, push; ``params`` is an argument, so the program is shared."""
    beliefs, _ = system.infer_states(params, obs, priors=carry.beliefs)
    action, _, _ = system.act(key, params, beliefs, planner=planner, metrics=False, **dict(kwargs))
    pushed = system.update_empirical_prior(params, beliefs, action)
    return action, carry._replace(beliefs=pushed)


def ancestral_respond(system: System, params: SystemParams) -> RespondFn:
    """An env-like component as a carry-free :data:`RespondFn`.

    The action is drawn from the declared prior over the mind's own control nodes and emitted
    keyed by their action nodes: ``policy_prior``'s first step if declared, else uniform per
    node. A component with no control nodes responds with the empty action. The carry passes
    through.
    """
    if any(k != "level" for k in system._space(params).kinds):
        raise NotImplementedError(
            "ancestral_respond draws a discrete control level per control node, and this System "
            "has continuous control nodes (A9); supply a RespondFn that samples from their boxes "
            "in params.control instead"
        )
    if system._table.sugar is not None:  # one draw per state factor, as the sugar lays them out
        levels = tuple(
            n if c else 1
            for n, c in zip(
                system._num_controls(params, "ancestral_respond"),
                system._ctl("ancestral_respond"),
                strict=True,
            )
        )
    else:
        levels = system._control_levels(system._params_in(params, "ancestral_respond"))
    policy_prior = params.policy_prior
    policies = system.policies(params) if policy_prior is not None else None
    logits = None
    if policy_prior is not None:
        logits = (
            policy_prior.log_probs
            if isinstance(policy_prior, Categorical)
            else jnp.asarray(policy_prior)
        )

    names = system._names()

    def respond(
        key: Array, obs: Mapping[str, Array], carry: MindCarry
    ) -> tuple[dict[str, Array], MindCarry]:
        del obs
        if logits is not None:
            chosen = policies[Categorical(logits=logits).sample(key), 0].astype(jnp.int32)
            return keyed.actions_out(names, chosen), carry
        keys = jax.random.split(key, len(levels))
        draws = [jax.random.randint(k, (), 0, n) for k, n in zip(list(keys), levels, strict=True)]
        drawn = jnp.stack(draws).astype(jnp.int32) if draws else jnp.zeros((0,), jnp.int32)
        if system._table.sugar is None:
            return keyed.actions_out(names, drawn), carry
        return keyed.factor_actions_out(names, drawn), carry

    return respond


class Trajectory(NamedTuple):
    """What :func:`run` produced, recorded at tick boundaries.

    ``states``/``observations`` carry ``ticks + 1`` rows per composite factor/modality, keyed by
    name (row 0 is the reset); ``actions`` maps each composite action node (by name) to its
    ``(ticks,)`` intended levels, the composite action the minds emitted as of each tick's last
    sub-step, and ``realised`` to the realised levels standing after the tick's last sub-step,
    on which the world moved (equal to ``actions`` for an action node without a kernel in
    ``params.actuator``); ``carries`` holds
    each mind's final carry.
    """

    states: dict[str, Array]
    observations: dict[str, Array]
    actions: dict[str, Array]
    carries: dict[str, MindCarry]
    realised: dict[str, Array] | None = None


class MindRecord(NamedTuple):
    """What a wired :func:`run` recorded of one mind.

    ``observations`` maps each of its modalities to ``ticks + 1`` rows as the mind received
    them: row ``t < ticks`` is what it responded to at tick ``t``, row ``ticks`` what its
    in-ports hold after the last tick. Under ``"sequential"`` and ``"async"`` a later mind's row
    ``t`` already holds the acts earlier minds made at tick ``t``. Under ``"simultaneous"`` the
    rows are the tick boundaries a :class:`ProcessRecord` keeps. ``actions`` and ``realised``
    map each of its action nodes to ``ticks`` rows: the intended action it emitted, and the
    value the node realised from it, standing after the tick's last step. ``carry`` is its
    final carry.
    """

    observations: dict[str, Array]
    actions: dict[str, Array]
    realised: dict[str, Array]
    carry: MindCarry


class ProcessRecord(NamedTuple):
    """What a wired :func:`run` recorded of one process: ``ticks + 1`` rows per factor
    (``states``) and per emitted modality (``observations``), keyed by name; row 0 is the
    reset, row ``t + 1`` the end of tick ``t``."""

    states: dict[str, Array]
    observations: dict[str, Array]


def run(
    composite: "System | Wiring",
    params: SystemParams | None = None,
    minds: Mapping[str, RespondFn] | None = None,
    ticks: int | None = None,
    key: Array | None = None,
    tick_order: TickOrder = "simultaneous",
    state: Mapping[str, Array] | None = None,
    carries: Mapping[str, MindCarry] | None = None,
    param_key: Array | None = None,
    *,
    states: Mapping[str, Mapping[str, Array]] | None = None,
):
    """Run minds and processes for a number of ticks; see the module docstring.

    **Wired form**, ``run(wiring, ticks=..., key=..., tick_order="simultaneous", carries=None,
    states=None) -> dict[str, MindRecord | ProcessRecord]``: ``wiring`` is
    :func:`~hybrid_aif.composition.couple.couple`'s result, and ``ticks`` and ``key`` are given
    by keyword; ``carries`` seeds the minds' carries and ``states`` the processes' resets (a
    point state keyed by factor name), each keyed by participant name. Every process steps once
    per tick, at the end, on the acts of the tick. Under ``"simultaneous"`` every mind responds
    to the tick's boundary; under ``"sequential"`` and ``"async"`` a mind's action nodes are
    realised as it chooses, so a later mind sees an earlier mind's act this tick through a
    direct wire. A mind's observation row ``t`` is what it responded to at tick ``t``, and row
    ``ticks`` its in-ports after the last tick; a process's rows are the tick boundaries (row 0
    the reset, row ``t + 1`` the end of tick ``t``). Under ``"sequential"`` and ``"async"`` a
    later mind's row ``t`` therefore already holds the acts earlier minds made at tick ``t`` on a
    direct wire. Before a mind has chosen, its action nodes intend level 0 (zeros, for a
    continuous node), undrawn, and every wire and process reached from one carries that null
    value as given. A node is realised when its mind chooses, and the processes step on the
    realised values standing at the end of the tick, once.

    Keys: the run key splits into ``(key, k_reset)``; process ``p`` (in :func:`couple` order)
    resets with ``k_reset`` for ``p = 0`` and ``fold_in(k_reset, p)`` after. Each tick
    ``(key, k_tick) = split(key)`` and ``k_tick`` splits into ``2 M + 2`` keys for ``M`` minds:
    mind ``j`` responds with key ``j``; its action nodes are realised with key ``M + j`` under
    sequential and async, and every mind's with the last key under simultaneous, which is also
    the tick's step key under every order. Realising with key ``k`` draws node ``i`` of the ``N``
    (minds in :func:`couple` order, each mind's action nodes in its node order) from its
    kernel's column at the intended level with key ``i`` of ``split(fold_in(k, ACTUATOR_TAG),
    N)``; the step with key ``k`` steps process ``p`` with ``k`` for ``p = 0`` and
    ``fold_in(k, p)`` after, on its realised inputs. A wiring of one process, a merged
    composite's ``as_env``, with the minds on its ``subsystem`` views and the wires by name,
    reproduces the composite form bitwise under ``"simultaneous"`` when the minds are given in
    the composite's component order, so that the action nodes, and the kernels keyed by them,
    fall in the composite's action order; under the other orders the composite form steps each
    component's factors at its turn, which a wiring of one process cannot, and the two differ.

    **Composite form**, ``run(composite, params, minds, ticks, key, tick_order="simultaneous",
    state=None, carries=None, param_key=None) -> Trajectory``: ``minds`` is keyed by subsystem
    name, in the order ``"sequential"`` follows. Every control node of the composite must be
    chosen by a named mind, or this raises (give the component a mind, or fix the node with a
    :class:`~hybrid_aif.composition.merge.BoundaryPrior`); a component that chooses no control
    node needs none. ``state`` overrides the env's initial draw with a known point state keyed
    by composite factor name, ``carries`` seeds the minds' carries, and ``param_key`` draws
    ground-truth tables from learnable declarations, as in :meth:`System.as_env`. Under
    ``"sequential"`` and ``"async"`` the composite steps a component's factors at that
    component's turn, on the realised actions standing then (every other mind's last realised
    action, level 0 before a mind has chosen), and the factors of the components that have no
    mind at the end of the tick; under ``"simultaneous"`` the action is re-zeroed each tick and
    the whole composite steps once on it. Every partial step emits the composite's observations
    afresh, so a later mind reads a stepped factor's new value and a new draw of any noisy
    modality, including one over factors that did not step. The composite's kernels
    (``params.actuator``) are drawn at the nodes with the wired form's keys over the composite's
    action order, and the world is stepped on the realised action with an ``env_params`` whose
    actuator mapping is empty, so the world draws nothing.
    """
    if isinstance(composite, Wiring):
        if any(v is not None for v in (params, minds, state, param_key)):
            raise TypeError(
                "run(wiring, ticks=..., key=...) takes ticks and key by keyword and nothing "
                "else by position, and its minds, their params and the processes come from "
                "couple(), so pass no params, minds, state or param_key"
            )
        if ticks is None or key is None:
            raise TypeError("run(wiring, ticks=..., key=...) needs both ticks and key")
        return _run_wired(composite, ticks, key, tick_order, carries, states)
    if states is not None:
        raise TypeError(
            "run(composite, ...) takes states= only with a Wiring; seed the composite's reset "
            "with state= keyed by composite factor name"
        )
    missing = [
        n
        for n, v in (("params", params), ("minds", minds), ("ticks", ticks), ("key", key))
        if v is None
    ]
    if missing:
        raise TypeError(f"run(composite, params, minds, ticks, key) is missing {missing}")
    return _run_composite(
        composite, params, minds, ticks, key, tick_order, state, carries, param_key
    )


def _run_composite(
    composite: System,
    params: SystemParams,
    minds: Mapping[str, RespondFn],
    ticks: int,
    key: Array,
    tick_order: TickOrder = "simultaneous",
    state: Mapping[str, Array] | None = None,
    carries: Mapping[str, MindCarry] | None = None,
    param_key: Array | None = None,
) -> Trajectory:
    """Run ``minds`` against the composite for ``ticks``: :func:`run`'s composite form.

    ``minds`` is keyed by subsystem name, in the order ``"sequential"`` follows. Every control
    node of the composite must be chosen by a named mind, or this raises (give the component a
    mind, or fix the node with a :class:`~hybrid_aif.composition.merge.BoundaryPrior`); a
    component that chooses no control node needs none. ``state`` overrides the env's initial draw
    with a known point state keyed by composite factor name, ``carries`` seeds the minds'
    carries, and ``param_key`` draws ground-truth tables from learnable declarations, as in
    :meth:`System.as_env`. Under ``"sequential"`` and ``"async"`` a component's factors step at
    its turn on the realised actions standing then (every other mind's last realised action,
    level 0 before a mind has chosen), and a component with no mind steps at the end of the
    tick, each partial step emitting the observations afresh; under ``"simultaneous"`` the
    action is re-zeroed each tick and the composite steps once on it.
    """
    if tick_order not in ("simultaneous", "sequential", "async"):
        raise ValueError(
            f"tick_order must be 'simultaneous', 'sequential', or 'async', got {tick_order!r}. "
            "A round is one tick under every order and every factor steps once per round. "
            "simultaneous = every mind reads the round's boundary, then the composite steps once "
            "on the merged actions (synchronous field update; simultaneous-move games); "
            "sequential = minds respond in declaration order and each component's factors step "
            "at its turn, reading values fresh where the source has stepped this round (fixed "
            "turn-taking); async = as sequential in a RANDOM permutation per round (no ordering "
            "privilege; the statistically faithful mode for equilibrium/Ising-like composites, "
            "where synchronous updates do not preserve the stationary distribution). The choice "
            "changes the game structure (design: nesting-systems.md, requirement 5 + "
            "tick-semantics note)."
        )
    if composite.subsystems is None:
        raise ValueError(
            "run() routes by the subsystems metadata, and this System carries none; build the "
            "composite with hybrid_aif.composition.merge.merge()"
        )
    table = dict(composite.subsystems)
    unknown = sorted(set(minds) - set(table))
    if unknown:
        raise ValueError(
            f"minds names {unknown}, which are not components of this composite; its "
            f"components are {sorted(table)}"
        )

    c_names = composite._names()
    c_mods, c_facs = c_names.modalities, c_names.factors
    action_of = dict(zip(c_names.controls, c_names.actions, strict=True))
    routes: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    claimed: dict[str, str] = {}
    for name in minds:
        chosen = [node for node in table[name].controls if node in action_of]
        # the mind's modalities and its action nodes, by composite name
        routes[name] = (
            tuple(c_mods[m] for m in table[name].modalities),
            tuple(action_of[node] for node in chosen),
        )
        for node in chosen:
            claimed[node] = name
    unclaimed = [node for node in c_names.controls if node not in claimed]
    if unclaimed:
        raise ValueError(
            f"control node(s) {unclaimed} are chosen by no mind, so nothing would ever emit "
            "their action; give the owning component a RespondFn (ancestral_respond(...) for an "
            "env-like one), or fix the node with a BoundaryPrior naming it at merge time"
        )
    #  which composite factors step at which component's turn (sequential, async); the factors
    #  of the components with no mind step at the end of the tick
    factors_of = {name: tuple(c_facs[i] for i in table[name].factors) for name in minds}
    minded = {f for fs in factors_of.values() for f in fs}
    unminded = tuple(f for f in c_facs if f not in minded)

    env, env_params = composite.as_env(params, param_key)
    key, k_reset = jax.random.split(key)
    obs, world = env.reset(k_reset, env_params, state)
    live = {name: MindCarry() for name in minds}
    live.update(carries or {})

    def no_op() -> dict[str, Array]:
        return {k: jnp.zeros((), jnp.int32) for k in c_names.actions}

    action = no_op()
    realised = no_op()
    #  the actuator draws at the node; the world steps on the realised action under params whose
    #  actuator is empty, so the world draws nothing
    kernels = dict(env_params.actuator or {})
    quiet = dataclasses.replace(env_params, actuator={})
    order = {a: i for i, a in enumerate(c_names.actions)}
    states_log, obs_log, action_log, realised_log = [world], [obs], [], []

    def actuate(k: Array, nodes: tuple[str, ...]) -> dict[str, Array]:
        """``nodes`` realised from the intended ``action`` at their composite indices."""
        drawn = _actuate(
            k,
            len(c_names.actions),
            [(order[a], kernels[a], action[a]) for a in nodes if a in kernels],
        )
        return {a: drawn.get(order[a], action[a]) for a in nodes}

    names = list(minds.keys())
    # tick_order="async": round orders from a host-side generator seeded by the run key
    # (folded, so the per-tick key stream is untouched).
    sweep_seed = jax.random.key_data(jax.random.fold_in(key, len(names)))
    sweep_rng = np.random.default_rng(np.asarray(sweep_seed, np.uint32).ravel())
    for _ in range(ticks):
        key, k_tick = jax.random.split(key)
        keys = list(jax.random.split(k_tick, 2 * len(minds) + 2))
        if tick_order == "simultaneous":
            action = no_op()
        if tick_order == "async":
            # a fresh random permutation per round: no fixed ordering privilege
            perm = [int(i) for i in sweep_rng.permutation(len(names))]
        else:
            perm = list(range(len(names)))
        for j in perm:
            name, respond = names[j], minds[names[j]]
            mods, ports = routes[name]
            chosen, live[name] = respond(keys[j], {m: obs[m] for m in mods}, live[name])
            action = {**action, **_check_action(chosen, ports, name, action_of)}
            if tick_order in ("sequential", "async"):
                k_step = keys[len(minds) + j]
                realised = {**realised, **actuate(k_step, ports)}
                if factors_of[name]:
                    obs, world = env.step(k_step, world, realised, quiet, factors_of[name])
        if tick_order == "simultaneous":
            realised = actuate(keys[-1], c_names.actions)
            obs, world = env.step(keys[-1], world, realised, quiet)
        elif unminded:
            obs, world = env.step(keys[-1], world, realised, quiet, unminded)
        states_log.append(world)
        obs_log.append(obs)
        action_log.append(action)
        realised_log.append(realised)

    return Trajectory(
        states={f: jnp.stack([s[f] for s in states_log]) for f in c_facs},
        observations={m: jnp.stack([o[m] for o in obs_log]) for m in c_mods},
        actions={
            k: jnp.stack([a[k] for a in action_log]) if action_log else jnp.zeros((0,), jnp.int32)
            for k in c_names.actions
        },
        carries=live,
        realised={
            k: jnp.stack([a[k] for a in realised_log])
            if realised_log
            else jnp.zeros((0,), jnp.int32)
            for k in c_names.actions
        },
    )


def _run_wired(
    wiring: Wiring,
    ticks: int,
    key: Array,
    tick_order: TickOrder = "simultaneous",
    carries: Mapping[str, MindCarry] | None = None,
    states: Mapping[str, Mapping[str, Array]] | None = None,
) -> dict[str, "MindRecord | ProcessRecord"]:
    """:func:`run`'s wired form."""
    if tick_order not in ("simultaneous", "sequential", "async"):
        raise ValueError(
            f"tick_order must be 'simultaneous', 'sequential', or 'async', got {tick_order!r}; "
            "every process steps once per tick, at the end; simultaneous = every mind responds "
            "to the tick's boundary; sequential = minds respond in couple() order, a later one "
            "seeing an earlier one's act on a direct wire; async = as sequential in a random "
            "order per tick"
        )
    mind_names, proc_names = list(wiring.minds), list(wiring.processes)
    for given, label, table in (
        (carries, "carries", mind_names),
        (states, "states", proc_names),
    ):
        unknown = sorted(set(given or {}) - set(table))
        if unknown:
            kind = "minds" if label == "carries" else "processes"
            raise ValueError(
                f"run(): {label} names {unknown}, which are not {kind} of this wiring; its "
                f"{kind} are {table}"
            )

    m_count = len(mind_names)
    #  the action nodes, in the fixed order the actuator keys follow
    nodes = [(n, a) for n in mind_names for a in wiring.minds[n].system.names.actions]
    port_types = {
        n: {a: wiring.types[f"{n}.{a}"] for a in wiring.minds[n].system.names.actions}
        for n in mind_names
    }
    action_of = {
        n: dict(
            zip(
                wiring.minds[n].system.names.controls,
                wiring.minds[n].system.names.actions,
                strict=True,
            )
        )
        for n in mind_names
    }
    intended = {n: {a: wiring.null(f"{n}.{a}") for a in port_types[n]} for n in mind_names}
    realised = {n: dict(intended[n]) for n in mind_names}

    def process_key(k: Array, p: int) -> Array:
        return k if p == 0 else jax.random.fold_in(k, p)

    key, k_reset = jax.random.split(key)
    obs: dict[str, dict[str, Array]] = {}
    world: dict[str, dict[str, Array]] = {}
    for p, n in enumerate(proc_names):
        proc = wiring.processes[n]
        obs[n], world[n] = proc.env.reset(
            process_key(k_reset, p), proc.params, (states or {}).get(n)
        )

    def value(port: str) -> Array:
        who, name = port.split(".", 1)
        if who in realised:
            return realised[who][name]
        return obs[who][name]

    def mind_inputs(n: str) -> dict[str, Array]:
        return {m: value(wiring.feeds[f"{n}.{m}"]) for m in wiring.minds[n].system.modality_names}

    def process_inputs(n: str) -> dict[str, Array]:
        out = {}
        for a in wiring.processes[n].env.names.actions:
            port = f"{n}.{a}"
            source = wiring.feeds[port]
            out[a] = wiring.fixed[port] if source is None else value(source)
        return out

    index = {node: i for i, node in enumerate(nodes)}

    def realise(k: Array, chosen: list[str]) -> None:
        """The action nodes of the ``chosen`` minds realised from their kernels."""
        draws = [
            (index[(n, a)], wiring.minds[n].kernels[a], intended[n][a])
            for n in chosen
            for a in port_types[n]
            if a in wiring.minds[n].kernels
        ]
        drawn = _actuate(k, len(nodes), draws)
        for n in chosen:
            realised[n] = {a: drawn.get(index[(n, a)], intended[n][a]) for a in port_types[n]}

    def step_processes(k: Array) -> None:
        """Every process stepped once on its realised inputs."""
        for p, n in enumerate(proc_names):
            proc = wiring.processes[n]
            obs[n], world[n] = proc.env.step(
                process_key(k, p), world[n], process_inputs(n), proc.params
            )

    live = {n: MindCarry() for n in mind_names}
    live.update(carries or {})
    received_log: dict[str, list] = {n: [] for n in mind_names}
    action_log: dict[str, list] = {n: [] for n in mind_names}
    realised_log: dict[str, list] = {n: [] for n in mind_names}
    states_log = {n: [world[n]] for n in proc_names}
    obs_log = {n: [obs[n]] for n in proc_names}

    sweep_seed = jax.random.key_data(jax.random.fold_in(key, m_count))
    sweep_rng = np.random.default_rng(np.asarray(sweep_seed, np.uint32).ravel())
    for _ in range(ticks):
        key, k_tick = jax.random.split(key)
        keys = list(jax.random.split(k_tick, 2 * m_count + 2))
        if tick_order == "async":
            perm = [int(i) for i in sweep_rng.permutation(m_count)]
        else:
            perm = list(range(m_count))
        for j in perm:
            n = mind_names[j]
            received = mind_inputs(n)
            received_log[n].append(received)
            chosen, live[n] = wiring.minds[n].respond(keys[j], received, live[n])
            intended[n] = _check_action(
                chosen, tuple(port_types[n]), n, action_of[n], port_types[n]
            )
            if tick_order in ("sequential", "async"):
                realise(keys[m_count + j], [n])
        if tick_order == "simultaneous":
            realise(keys[-1], mind_names)
        step_processes(keys[-1])
        for n in mind_names:
            action_log[n].append(dict(intended[n]))
            realised_log[n].append(dict(realised[n]))
        for n in proc_names:
            states_log[n].append(world[n])
            obs_log[n].append(obs[n])
    for n in mind_names:
        received_log[n].append(mind_inputs(n))

    def stacked(rows: list, keys_: tuple[str, ...], empty) -> dict[str, Array]:
        return {k: jnp.stack([r[k] for r in rows]) if rows else empty(k) for k in keys_}

    out: dict[str, MindRecord | ProcessRecord] = {}
    for n in mind_names:
        actions = tuple(port_types[n])

        def empty(a: str, n: str = n) -> Array:
            null = wiring.null(f"{n}.{a}")
            return jnp.zeros((0, *null.shape), null.dtype)

        out[n] = MindRecord(
            observations=stacked(received_log[n], wiring.minds[n].system.modality_names, None),
            actions=stacked(action_log[n], actions, empty),
            realised=stacked(realised_log[n], actions, empty),
            carry=live[n],
        )
    for n in proc_names:
        names = wiring.processes[n].env.names
        out[n] = ProcessRecord(
            states=stacked(states_log[n], names.factors, None),
            observations=stacked(obs_log[n], names.modalities, None),
        )
    return out


def _actuate(k: Array, n: int, draws: list[tuple[int, Array, Array]]) -> dict[int, Array]:
    """The actuator's draws at the node: for each ``(i, kernel, intended)``, a level drawn from
    the kernel's column at the intended level with key ``i`` of
    ``split(fold_in(k, ACTUATOR_TAG), n)``, keyed by ``i``. No draw, no key."""
    if not draws:
        return {}
    keys = jax.random.split(jax.random.fold_in(k, ACTUATOR_TAG), n)
    out = {}
    for i, kernel, level in draws:
        log_p = jnp.log(jnp.asarray(kernel, jnp.float32))
        column = jnp.moveaxis(jnp.take(log_p, jnp.asarray(level, jnp.int32), axis=-1), 0, -1)
        out[i] = jax.random.categorical(keys[i], column).astype(jnp.int32)
    return out


def _check_action(
    chosen: Mapping[str, Array],
    ports: tuple[str, ...],
    name: str,
    action_of: Mapping[str, str] | None = None,
    types: Mapping[str, tuple[str, int | None]] | None = None,
) -> dict[str, Array]:
    """The mind's action, checked: exactly its action nodes, each one integer level (shape
    ``()``), or with ``types`` naming a node continuous, a float vector of its width; a key
    naming a control node is refused naming its action node."""
    if not isinstance(chosen, Mapping):
        raise TypeError(
            f"mind {name!r} returned a positional action; a RespondFn returns its action keyed "
            f"by action node, {list(ports)} (convert with view.from_positional('actions', ...))"
        )
    crossed = [k for k in chosen if k in (action_of or {}) and action_of[k] in ports]
    if crossed:
        raise ValueError(
            f"mind {name!r} returned actions[{crossed[0]!r}], a control node; key the action by "
            f"its action node {action_of[crossed[0]]!r}"
        )
    missing = [k for k in ports if k not in chosen]
    unknown = [k for k in chosen if k not in ports]
    if missing or unknown:
        raise ValueError(
            f"mind {name!r} returned an action keyed {list(chosen)}, missing {missing} and "
            f"unknown {unknown}; its action nodes are {list(ports)}"
        )
    out = {}
    for k in ports:
        level = jnp.asarray(chosen[k])
        if types is not None and types[k][0] == FLOAT:
            width = types[k][1]
            if level.shape != (width,) or not jnp.issubdtype(level.dtype, jnp.floating):
                raise ValueError(
                    f"mind {name!r} returned actions[{k!r}] of dtype {level.dtype} and shape "
                    f"{tuple(level.shape)}; a continuous action node takes a float vector of "
                    f"its width, shape ({width},)"
                )
            out[k] = level
            continue
        if level.shape != ():
            raise ValueError(
                f"mind {name!r} returned actions[{k!r}] of shape {tuple(level.shape)}; an action "
                "node takes one level per tick, shape ()"
            )
        if not jnp.issubdtype(level.dtype, jnp.integer):
            raise ValueError(
                f"mind {name!r}: actions [{k!r}] are not integer ({level.dtype}); a discrete "
                "control's action is a level index"
            )
        if types is not None:
            size = types[k][1]
            if not 0 <= int(level) < size:
                raise ValueError(
                    f"mind {name!r} returned actions[{k!r}] = {int(level)}; action node {k!r} has "
                    f"{size} levels, so emit a level in [0, {size})"
                )
        out[k] = level.astype(jnp.int32)
    return out
