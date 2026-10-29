"""Two candidate models of one recorded trace, written as documents and priced in nats.

The world is the T-maze of the demo notebook's Section 1. A scripted agent runs episodes of
five ticks: centre, cue, centre, middle, then the arm the cue named. Two models of that record
differ in one parent set. In model A the cue reads the context, as the world does. In model B
the cue reads the location alone, so in B the cue says nothing about the context and the
reward at the arm is a coin flip.

Each model is a document: :func:`hybrid_aif.describe` writes it, :func:`hybrid_aif.build`
reads it back bitwise, and the two documents differ in the cue's parent list alone. Each is
priced as a two-part code by :func:`~hybrid_aif.description.mdl.description_length`
with ``replay=``, both parts read from its document. The model part is the document's own
description length, each table priced at the data it is judged against along the trace, the
structure of a model with point tables (graph and family links). The data part is the trace's
code length under the model ``build`` makes of the document, the sum of
:func:`~hybrid_aif.replay.replay`'s per-tick free energies; the replay does not learn, so the
family links are what pays for the parameters. A has more structure, so it costs more to
state; it codes the record more cheaply once the cue has paid for itself, and
the two-part total crosses over as the record grows.

Design: docs/design/description-length.md (D1–D4, D6); docs/design/replay.md
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_aif import Categorical, ConditionalCategorical, build, describe
from hybrid_aif.description.mdl import description_length
from hybrid_aif.reference.envs.tmaze import CENTRE, CUE, LEFT, MIDDLE, RIGHT, tmaze
from hybrid_aif.replay import replay
from hybrid_aif.system import System, SystemParams

MODALITIES = ("location_obs", "reward", "cue")
FACTORS = ("location", "context")
TICKS_PER_EPISODE = 5

_ENV, _TABLES = tmaze(reward_probability=0.98, punishment_probability=0.98, cue_validity=0.98)


def _cc(table, parent_ndims: int) -> ConditionalCategorical:
    return ConditionalCategorical(
        log_table=jnp.log(jnp.clip(table, 1e-16)), parent_ndims=parent_ndims
    )


def make_model(cue_reads_context: bool = True) -> tuple[System, SystemParams]:
    """The T-maze model with the world's tables. ``cue_reads_context=False`` gives the cue the
    location alone as its parent, its table the world's averaged over the context."""
    A_deps = ((0,), (0, 1), (0, 1)) if cue_reads_context else ((0,), (0, 1), (0,))
    system = System(
        A_deps=A_deps,
        B_deps=_ENV.B_deps,
        controlled=(True, False),
        policy_len=2,
        modality_names=MODALITIES,
        factor_names=FACTORS,
    )
    tables = list(_TABLES.likelihood)
    if not cue_reads_context:
        tables[2] = tables[2].mean(-1)
    params = SystemParams(
        likelihood={m: _cc(a, len(d)) for m, a, d in zip(MODALITIES, tables, A_deps, strict=True)},
        transition={
            f: _cc(b, len(d) + 1)
            for f, b, d in zip(FACTORS, _TABLES.transition, _ENV.B_deps, strict=True)
        },
        initial_state_prior={
            f: Categorical.from_probs(d)
            for f, d in zip(FACTORS, _TABLES.initial_state_prior, strict=True)
        },
        obs_targets={"reward": Categorical(logits=jnp.asarray([0.0, 4.0, -6.0]))},
    )
    return system, params


def round_trip(system: System, params: SystemParams):
    """``describe`` then ``build``: the document, and whether every array came back bitwise."""
    document, arrays = describe(system, params)
    rebuilt_system, rebuilt = build(document, arrays=arrays)
    same = jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda a, b: bool(jnp.array_equal(a, b)), params, rebuilt)
    )
    return document, (rebuilt_system, rebuilt), bool(same)


def record(n_episodes: int, seed: int = 0) -> tuple[dict, dict, np.ndarray]:
    """A recorded trace from the world: ``(observations, actions, contexts)``, the first two
    ``(episodes, 5)`` and the hidden context of each episode ``(episodes,)``. The scripted
    agent visits the cue and then takes the arm the cue named."""
    rng = np.random.default_rng(seed)
    A_reward, A_cue = (np.asarray(a) for a in _TABLES.likelihood[1:])
    obs = {m: [] for m in MODALITIES}
    actions, contexts = [], []
    for _ in range(n_episodes):
        context = rng.integers(2)
        contexts.append(context)
        reading = rng.choice(3, p=A_cue[:, CUE, context])
        arm = LEFT if reading == 1 else RIGHT
        route = [CENTRE, CUE, CENTRE, MIDDLE, arm]
        obs["location_obs"].append(route)
        obs["reward"].append([rng.choice(3, p=A_reward[:, loc, context]) for loc in route])
        obs["cue"].append(
            [reading if loc == CUE else rng.choice(3, p=A_cue[:, loc, context]) for loc in route]
        )
        actions.append(route[1:] + [arm])  # the last tick's action stays put
    as_int = lambda rows: jnp.asarray(np.asarray(rows), jnp.int32)  # noqa: E731
    observations = {m: as_int(v) for m, v in obs.items()}
    return observations, {"a_location": as_int(actions)}, np.asarray(contexts)


def replays(system: System, params: SystemParams, observations, actions):
    """The episodes replayed independently, each from the model's initial priors: one
    :class:`~hybrid_aif.replay.Replay` whose leaves lead with ``(episodes, 5)``."""
    one = lambda o, a: replay(system, params, o, a, metrics=False)  # noqa: E731
    return jax.vmap(one)(observations, actions)


def free_energies(system: System, params: SystemParams, observations, actions) -> np.ndarray:
    """``(episodes, 5)``: each tick's free energy, the episodes replayed independently."""
    return np.asarray(replays(system, params, observations, actions).free_energy, np.float64)


class Comparison(NamedTuple):
    episodes: np.ndarray  # (E,) 1..E
    model_nats: dict[str, np.ndarray]  # (E,) the model part at each record length
    data_nats: dict[str, np.ndarray]  # (E,) the data part, cumulative
    energies: dict[str, np.ndarray]  # (E, 5) each tick's free energy
    wrong_cue: np.ndarray  # (E,) bool: the cue named the punished arm
    null_outcome: np.ndarray  # (E,) bool: the arm returned neither reward nor punishment

    @property
    def per_tick(self) -> dict[str, np.ndarray]:
        """``(5,)`` per model: the mean free energy of each tick of an episode."""
        return {label: e.mean(axis=0) for label, e in self.energies.items()}


def compare(n_episodes: int = 120, seed: int = 0) -> Comparison:
    """The two-part code of each model as the record grows by an episode at a time. Each model
    is written as a document, and ``description_length(..., replay=)`` reads both parts from
    it: the model part is the document's own description length, each table at the data it is
    judged against in the record, the record part the replay of the model ``build`` makes of
    it."""
    observations, actions, contexts = record(n_episodes, seed)
    episodes = np.arange(1, n_episodes + 1)
    model_nats, data_nats, energies = {}, {}, {}
    for label, reads in (("A", True), ("B", False)):
        document, arrays = describe(*make_model(reads))
        system, params = build(document, arrays=arrays)
        trace = replays(system, params, observations, actions)
        energies[label] = np.asarray(trace.free_energy, np.float64)
        lengths = [
            description_length(
                document, replay=jax.tree_util.tree_map(lambda x, e=e: x[:e], trace), arrays=arrays
            )
            for e in episodes
        ]
        model_nats[label] = np.asarray([length.model for length in lengths])
        data_nats[label] = np.asarray([length.data for length in lengths])
    reading = np.asarray(observations["cue"][:, 1])
    wrong_cue = reading != contexts + 1
    null_outcome = np.asarray(observations["reward"][:, -1]) == 0
    return Comparison(episodes, model_nats, data_nats, energies, wrong_cue, null_outcome)


def crossover(result: Comparison) -> int | None:
    """The first record length, in episodes, from which A's two-part total stays below B's."""
    gap = (result.model_nats["A"] + result.data_nats["A"]) - (
        result.model_nats["B"] + result.data_nats["B"]
    )
    below = gap < 0.0
    for i in range(len(below)):
        if below[i:].all():
            return int(result.episodes[i])
    return None


def main() -> None:
    document, _, same = round_trip(*make_model(True))
    print("bitwise round trip:", same)
    print("cue record (A):", document["factors"]["cue"]["likelihood"]["parents"])
    result = compare()
    for label in ("A", "B"):
        print(
            f"model {label}: {result.model_nats[label][-1]:.1f} nats to state, "
            f"{result.data_nats[label][-1]:.1f} nats of data over {result.episodes[-1]} episodes"
        )
    print("A wins from episode", crossover(result))


if __name__ == "__main__":
    main()
