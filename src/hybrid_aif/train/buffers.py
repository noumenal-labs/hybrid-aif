"""Replay: one episode format, three buffers for two execution regimes.

Replay is agent-coupled, so it lives beside the trainers; it reads
:mod:`hybrid_aif.data`, never the reverse.

============================  ====================  ==========================
                              :class:`ReplayBuffer`  :class:`StaticBuffer`,
                                                     :class:`StreamBuffer`
============================  ====================  ==========================
sampled                       in **Python**, between  **inside** jit /
                              jitted calls            ``lax.scan``
shapes                        dynamic; schema from    static, declared at
                              the first episode       allocation
episodes                      ragged, as collected    fitted to ``seq_len`` /
                                                      spliced into a stream
storage                       numpy ring, one array   a **pytree** of JAX
                              per column              arrays
columns                       whatever was written    the :class:`Episode`
                                                      format
extras                        ``Dataset`` windowing,  rides a ``lax.scan``
                              pluggable samplers,     carry
                              ``dump()`` to any
                              registered format
============================  ====================  ==========================

Use :class:`ReplayBuffer` unless you need in-jit sampling. ``num_steps`` ≈
``seq_len`` and ``max_steps`` ≈ ``capacity·seq_len``; :func:`as_episode` and
:func:`episode_columns` bridge the regimes. The device-side buffers differ in their
sequence contract:

============================  ======================  ==========================
                              :class:`StaticBuffer`   :class:`StreamBuffer`
============================  ======================  ==========================
a stored unit is              one whole episode,      one **step**
                              fitted to ``seq_len``
``capacity`` counts           sequences               steps
an episode shorter than       padded by freezing      nothing — it is
``seq_len``                   its final step          ``length`` steps of stream
a sampled window              starts at an episode's  starts anywhere, and may
                              first step              cross episode boundaries
``is_first``                  true only at step 0     true wherever an episode
                                                      begins inside the window
needs                         nothing of the RSSM     ``RSSMCore.observe``'s
                                                      ``is_first`` reset
============================  ======================  ==========================

Buffer policy (several buffers, routing, fallback) stays in the caller's loop.

Design: docs/design/amortization.md A12 chunk 2c §(d′)
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hybrid_aif.data.dataset import Dataset
from hybrid_aif.data.format import check_schema, episode_schema, get_format, public_items

# ============================================================== the episode format


class ColumnSpec(NamedTuple):
    """One task-specific column of an :class:`Episode`.

    Its per-step trailing shape, its dtype by name, and how :func:`fit_length` pads
    it: ``"zero"`` or ``"hold"`` (repeat the last value).
    """

    shape: tuple[int, ...] = ()
    dtype: str = "float32"
    pad: str = "zero"


class Episode(NamedTuple):
    """One episode's per-step columns — the format the whole training side speaks.

    Leaves are ``(T, ...)`` for one episode, ``(B, T, ...)`` for a batch. Row ``t``
    pairs ``obs[t]`` with the action then drawn at it, so ``reward[t]`` is the
    reward of ``action[t−1]`` and every flag is a predicate on ``obs[t]``. ``T``
    applied actions make ``T + 1`` rows: a reset row (``is_first``, ``reward = 0``)
    and a terminal row with the action zeroed. ``cont`` is ``1 − terminal``, as
    recorded by the collector. ``extras`` holds the task's own per-step columns,
    carried through untouched.
    """

    obs: Array  # (T, obs_dim) float32
    action: Array  # (T, action_dim) float32 — a_t, the RAW draw taken AT obs[t]
    reward: Array  # (T,) float32 — the reward that PRODUCED obs[t], i.e. a_{t−1}'s
    cont: Array  # (T,) float32 — 1 − terminal, a predicate on obs[t]
    is_first: Array  # (T,) bool
    is_last: Array  # (T,) bool
    extras: dict[str, Array] = {}  # (T, ...) each — the task's own columns

    @property
    def length(self) -> int:
        return self.obs.shape[-2]

    def prev_actions(self) -> Array:
        """The actions re-indexed so entry ``t`` is ``a_{t−1}``; step 0 gets zeros.

        :func:`~hybrid_aif.train.world.fit_world`'s convention.
        """
        zero = jnp.zeros_like(self.action[..., :1, :])
        return jnp.concatenate([zero, self.action[..., :-1, :]], axis=-2)


#: The engine's columns: what every episode carries, whatever the task.
EPISODE_FIELDS: tuple[str, ...] = ("obs", "action", "reward", "cont", "is_first", "is_last")
#: How :func:`fit_length` pads each engine column.
_CORE_PAD: dict[str, str] = {
    "obs": "hold",
    "action": "zero",
    "reward": "zero",
    "cont": "hold",
    "is_first": "zero",
}
Schema = Mapping[str, ColumnSpec]


def episode(
    obs: Array,
    action: Array,
    reward: Array,
    *,
    is_first: Array,
    is_last: Array,
    cont: Array | None = None,
    **extras: Array,
) -> Episode:
    """An :class:`Episode` from its columns, cast to the engine dtypes and checked.

    ``cont`` defaults to all ones. Any further keyword is a task column, stored in
    ``extras`` with its own dtype and a leading ``T`` axis checked.
    """
    t = int(jnp.asarray(obs).shape[0])
    core = {
        "obs": jnp.asarray(obs, jnp.float32),
        "action": jnp.asarray(action, jnp.float32),
        "reward": jnp.asarray(reward, jnp.float32),
        "cont": jnp.ones((t,), jnp.float32) if cont is None else jnp.asarray(cont, jnp.float32),
        "is_first": jnp.asarray(is_first, jnp.bool_),
        "is_last": jnp.asarray(is_last, jnp.bool_),
    }
    for name in ("obs", "action"):
        if core[name].ndim != 2:
            raise ValueError(
                f"episode(): {name} has shape {core[name].shape}, expected (T, dim) — a "
                "per-step vector even where the dimension is 1"
            )
    for name in ("reward", "cont", "is_first", "is_last"):
        if core[name].ndim != 1:
            raise ValueError(
                f"episode(): {name} has shape {core[name].shape}, expected (T,) — one number "
                "per step"
            )
    cast_extras = {name: jnp.asarray(value) for name, value in extras.items()}
    lengths = {
        name: int(value.shape[0]) if value.ndim else -1
        for name, value in {**core, **cast_extras}.items()
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"episode(): columns disagree on T: {lengths}")
    return Episode(**core, extras=cast_extras)


def fit_length(ep: Episode, seq_len: int, pad: Mapping[str, str] | None = None) -> Episode:
    """One episode at exactly ``seq_len`` steps: truncated, or padded by freezing.

    Padding repeats the final observation and ``cont`` with zero action and zero
    reward; ``is_last`` marks the final stored step. Task columns pad by
    ``pad[name]`` (``"hold"`` or ``"zero"``; zeros when not given).
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    t = ep.length
    if t >= seq_len:
        out = jax.tree_util.tree_map(lambda x: x[:seq_len], ep)
        return out._replace(is_last=jnp.zeros((seq_len,), jnp.bool_).at[-1].set(True))
    n = seq_len - t
    extra_rules = dict(pad or {})
    if bad := sorted(set(extra_rules) & set(EPISODE_FIELDS)):
        raise ValueError(
            f"fit_length: pad names engine columns {bad}; only task columns take a rule"
        )
    rules = {**_CORE_PAD, **extra_rules}

    def padded(name: str, x: Array) -> Array:
        rule = rules.get(name, "zero")
        if rule == "hold":
            return jnp.concatenate([x, jnp.repeat(x[-1:], n, axis=0)], axis=0)
        if rule == "zero":
            return jnp.concatenate([x, jnp.zeros((n, *x.shape[1:]), x.dtype)], axis=0)
        raise ValueError(
            f"fit_length: pad rule for column {name!r} is {rule!r}; the rules are 'hold' and 'zero'"
        )

    core = {name: padded(name, getattr(ep, name)) for name in EPISODE_FIELDS if name != "is_last"}
    return Episode(
        **core,
        is_last=jnp.zeros((seq_len,), jnp.bool_).at[-1].set(True),
        extras={name: padded(name, x) for name, x in ep.extras.items()},
    )


# ======================================================= host-side, dynamic schema

Sampler = Callable[["ReplayBuffer", int, np.random.Generator], np.ndarray]
"""``sampler(buffer, batch_size, rng) -> flat clip indices in [0, len(buffer))``."""


def uniform_sampler(buffer: ReplayBuffer, batch_size: int, rng: np.random.Generator):
    n = len(buffer)
    if n == 0:
        raise RuntimeError("ReplayBuffer.sample: no clips available")
    return rng.integers(0, n, size=batch_size)


class ReplayBuffer(Dataset):
    """Ring-storage replay buffer with whole-episode FIFO eviction. Host-side.

    Both a :class:`~hybrid_aif.data.dataset.Dataset` and a
    :class:`~hybrid_aif.data.format.Writer`. Storage is one numpy ring per column
    of ``max_steps`` rows, allocated from the first episode's schema; clips never
    cross episode boundaries. ``dump(path, format)`` persists through any
    registered format.

    Args:
        max_steps: capacity budget in raw steps.
        num_steps: observation frames per clip (Dataset windowing).
        frameskip: stride between observation frames within a clip.
        sampler: pluggable :data:`Sampler` used by :meth:`sample`; uniform
            over valid clips by default.
        seed: seed for the default rng handed to the sampler.
    """

    def __init__(
        self,
        max_steps: int,
        num_steps: int = 1,
        frameskip: int = 1,
        sampler: Sampler | None = None,
        seed: int | None = None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        if num_steps <= 0 or frameskip <= 0:
            raise ValueError(
                f"num_steps and frameskip must be positive, got {num_steps=} {frameskip=}"
            )
        self.max_steps = int(max_steps)
        self.num_steps = num_steps
        self.frameskip = frameskip
        self.sampler: Sampler = sampler if sampler is not None else uniform_sampler
        self.rng = np.random.default_rng(seed)
        self._cols: dict[str, np.ndarray] = {}
        self._schema: dict[str, tuple[tuple[int, ...], np.dtype]] | None = None
        self._episodes: deque[tuple[int, int]] = deque()  # (ring_start, ep_len)
        self._head = 0
        self._size = 0
        self._clip_cum: np.ndarray | None = None
        self._clips: list[tuple[int, int]] | None = None

    # -- Writer side -------------------------------------------------------

    def __enter__(self) -> ReplayBuffer:
        return self

    def __exit__(self, *exc) -> None:
        return None

    def write_episode(self, ep_data: dict) -> None:
        if self._schema is None:
            schema = episode_schema(ep_data)
        else:
            check_schema(self._schema, ep_data, "ReplayBuffer.write_episode")
            schema = self._schema
        stored = dict(public_items(ep_data))
        ep_len = next(iter(stored.values())).shape[0]
        if ep_len == 0:
            return
        if ep_len > self.max_steps:
            raise ValueError(f"episode of length {ep_len} exceeds max_steps={self.max_steps}")
        if self._schema is None:
            self._schema = schema
            for col, (shape, dtype) in schema.items():
                self._cols[col] = np.empty((self.max_steps, *shape), dtype=dtype)

        while self._size + ep_len > self.max_steps:
            _, evicted = self._episodes.popleft()
            self._size -= evicted
        positions = (self._head + np.arange(ep_len)) % self.max_steps
        for col, arr in stored.items():
            self._cols[col][positions] = arr
        self._episodes.append((self._head, ep_len))
        self._head = (self._head + ep_len) % self.max_steps
        self._size += ep_len
        self._clip_cum = None
        self._clips = None

    def write_episodes(self, episodes: Iterable[dict]) -> None:
        for ep in episodes:
            self.write_episode(ep)

    # -- Dataset side ------------------------------------------------------

    @property
    def column_names(self) -> list[str]:
        return list(self._cols)

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def num_steps_stored(self) -> int:
        return self._size

    @property
    def lengths(self) -> np.ndarray:
        return np.asarray([ln for _, ln in self._episodes], dtype=np.int64)

    @property
    def offsets(self) -> np.ndarray:
        lens = self.lengths
        return np.concatenate(([0], np.cumsum(lens[:-1]))).astype(np.int64)

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict[str, np.ndarray]:
        ring_start, ep_len = self._episodes[ep_idx]
        if not 0 <= start <= end <= ep_len:
            raise IndexError(f"slice [{start}:{end}] out of episode of length {ep_len}")
        positions = (ring_start + np.arange(start, end)) % self.max_steps
        return {col: arr[positions] for col, arr in self._cols.items()}

    def sample(
        self, batch_size: int, rng: np.random.Generator | None = None
    ) -> dict[str, np.ndarray]:
        """Draw a batch of clips via the sampler; arrays are stacked to
        ``(batch_size, num_steps, ...)``."""
        idx = np.asarray(self.sampler(self, batch_size, rng if rng is not None else self.rng))
        eps, starts = self.locate(idx)
        clips = [self._clip(int(e), int(s)) for e, s in zip(eps, starts, strict=True)]
        return {col: np.stack([c[col] for c in clips]) for col in self.column_names}

    # -- Persistence -------------------------------------------------------

    def iter_episodes(self) -> Iterator[dict[str, np.ndarray]]:
        """Yield stored episodes oldest-first as contiguous column dicts."""
        for ring_start, ep_len in list(self._episodes):
            positions = (ring_start + np.arange(ep_len)) % self.max_steps
            yield {col: arr[positions] for col, arr in self._cols.items()}

    def dump(self, path, format: str, mode: str = "overwrite", **kwargs: Any) -> None:
        """Persist current contents through the registered writer for ``format``."""
        import hybrid_aif.data.formats  # noqa: F401  (registers built-in formats)

        with get_format(format).open_writer(path, mode=mode, **kwargs) as writer:
            writer.write_episodes(self.iter_episodes())


# ====================================================== device-side, static shapes


class StaticBuffer(eqx.Module):
    """A uniform replay buffer of fixed-length episode-aligned sequences. A pytree.

    ``capacity`` sequences of ``seq_len`` steps, one JAX array per column, written
    FIFO into a ring. ``size`` (live slots) and ``cursor`` (next slot) are traced
    scalars, so :func:`add` and :func:`sample` are pure and the buffer can ride a
    ``lax.scan`` carry.
    """

    data: Episode
    size: Array
    cursor: Array
    capacity: int = eqx.field(static=True)
    seq_len: int = eqx.field(static=True)
    schema: tuple[tuple[str, ColumnSpec], ...] = eqx.field(static=True, default=())

    @property
    def pad_rules(self) -> dict[str, str]:
        """How :func:`fit_length` pads each task column, from the schema."""
        return {name: spec.pad for name, spec in self.schema}

    @classmethod
    def empty(
        cls,
        capacity: int,
        seq_len: int,
        obs_dim: int,
        action_dim: int,
        extras: Schema | None = None,
    ) -> StaticBuffer:
        """An allocated, empty buffer. Shapes are fixed here and never change.

        ``extras`` declares the task's columns (:class:`ColumnSpec` per name); every
        episode written must carry exactly those.
        """
        n, t = int(capacity), int(seq_len)
        if capacity < 1:
            raise ValueError(f"capacity must be positive, got {capacity}")
        schema = _schema(extras)
        lead = (n, t) if cls is StaticBuffer else (n,)
        return cls(
            data=Episode(
                obs=jnp.zeros((*lead, int(obs_dim)), jnp.float32),
                action=jnp.zeros((*lead, int(action_dim)), jnp.float32),
                reward=jnp.zeros(lead, jnp.float32),
                cont=jnp.ones(lead, jnp.float32),
                is_first=jnp.zeros(lead, jnp.bool_),
                is_last=jnp.zeros(lead, jnp.bool_),
                extras={
                    name: jnp.zeros((*lead, *spec.shape), jnp.dtype(spec.dtype))
                    for name, spec in schema
                },
            ),
            size=jnp.zeros((), jnp.int32),
            cursor=jnp.zeros((), jnp.int32),
            capacity=n,
            seq_len=t,
            schema=schema,
        )

    def __len__(self) -> int:
        """Live sequences, as a Python int (host-side); inside jit read :attr:`size`."""
        return int(self.size)

    def ready(self, batch_size: int) -> bool:
        """Whether a full batch can be drawn without repeating the whole buffer. Host-side."""
        return len(self) >= int(batch_size)


def _check_extras(buffer: StaticBuffer | StreamBuffer, ep: Episode) -> None:
    declared = dict(buffer.schema)
    if set(ep.extras) != set(declared):
        raise ValueError(
            f"the episode carries task columns {sorted(ep.extras)} but the buffer was "
            f"allocated for {sorted(declared)}: declare them in .empty(extras=...) — a buffer "
            "holds exactly the columns its schema names"
        )
    for name, spec in declared.items():
        column = ep.extras[name]
        if column.dtype != jnp.dtype(spec.dtype) or tuple(column.shape[1:]) != tuple(spec.shape):
            raise ValueError(
                f"task column {name!r} is {column.dtype} with per-step shape "
                f"{tuple(column.shape[1:])} but the buffer's schema says {spec.dtype} "
                f"{tuple(spec.shape)}; cast or reshape it before writing — a store would cast "
                "silently"
            )


def _schema(extras: Schema | None) -> tuple[tuple[str, ColumnSpec], ...]:
    """A buffer's static schema from ``extras``: names to :class:`ColumnSpec`, validated."""
    out = []
    for name, spec in dict(extras or {}).items():
        spec = ColumnSpec(*spec)
        if name in EPISODE_FIELDS:
            raise ValueError(f"{name!r} is an engine column, not a task column")
        if spec.pad not in ("hold", "zero"):
            raise ValueError(f"column {name!r}: pad must be 'hold' or 'zero', got {spec.pad!r}")
        out.append((name, ColumnSpec(tuple(int(d) for d in spec.shape), str(spec.dtype), spec.pad)))
    return tuple(out)


def add(buffer: StaticBuffer, ep: Episode) -> StaticBuffer:
    """One episode written into a :class:`StaticBuffer` ring: a new buffer, nothing mutated.

    ``ep`` is fitted to ``seq_len`` first. Eviction is FIFO by slot. Compute any
    column derived from the whole episode before adding.
    """
    _check_extras(buffer, ep)
    fitted = fit_length(ep, buffer.seq_len, pad=buffer.pad_rules)
    slot = buffer.cursor % buffer.capacity
    data = jax.tree_util.tree_map(lambda store, row: store.at[slot].set(row), buffer.data, fitted)
    return StaticBuffer(
        data=data,
        size=jnp.minimum(buffer.size + 1, buffer.capacity),
        cursor=slot + 1,
        capacity=buffer.capacity,
        seq_len=buffer.seq_len,
        schema=buffer.schema,
    )


def sample(buffer: StaticBuffer, key: Array, batch_size: int) -> Episode:
    """A uniform batch of whole sequences: leaves ``(batch_size, seq_len, ...)``.

    Samples with replacement over the live slots. An empty buffer draws slot 0, so
    guard with :meth:`StaticBuffer.ready`.
    """
    index = jax.random.randint(key, (int(batch_size),), 0, jnp.maximum(buffer.size, 1))
    return jax.tree_util.tree_map(lambda store: store[index], buffer.data)


class StreamBuffer(eqx.Module):
    """A uniform replay buffer over a flat step stream. Device-side, a pytree.

    Static shapes and traced counters like :class:`StaticBuffer`, but ``capacity``
    counts steps: :func:`extend` splices each episode onto one ring in completion
    order, and :func:`sample_windows` draws ``seq_len`` contiguous steps starting
    anywhere, so a window may cross episode boundaries; its ``is_first`` column
    marks the joins for :meth:`~hybrid_aif.inference.rssm.RSSMCore.observe`.
    Nothing is padded or truncated; eviction is FIFO by step.

    Design: docs/design/amortization.md A12 chunk 2c §(d′)
    """

    data: Episode
    size: Array
    cursor: Array
    capacity: int = eqx.field(static=True)
    seq_len: int = eqx.field(static=True)
    schema: tuple[tuple[str, ColumnSpec], ...] = eqx.field(static=True, default=())

    @property
    def pad_rules(self) -> dict[str, str]:
        """How :func:`fit_length` pads each task column, from the schema."""
        return {name: spec.pad for name, spec in self.schema}

    @classmethod
    def empty(
        cls,
        capacity: int,
        seq_len: int,
        obs_dim: int,
        action_dim: int,
        extras: Schema | None = None,
    ) -> StreamBuffer:
        """An allocated, empty buffer. Shapes are fixed here and never change.

        ``extras`` declares the task's columns (:class:`ColumnSpec` per name); every
        episode written must carry exactly those.
        """
        n, t = int(capacity), int(seq_len)
        if t < 1:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if n < t:
            raise ValueError(
                f"capacity must be at least seq_len ({t} steps), got {capacity} — capacity is "
                "in STEPS here, not sequences, and a stream shorter than one window can never "
                "be sampled"
            )
        schema = _schema(extras)
        lead = (n, t) if cls is StaticBuffer else (n,)
        return cls(
            data=Episode(
                obs=jnp.zeros((*lead, int(obs_dim)), jnp.float32),
                action=jnp.zeros((*lead, int(action_dim)), jnp.float32),
                reward=jnp.zeros(lead, jnp.float32),
                cont=jnp.ones(lead, jnp.float32),
                is_first=jnp.zeros(lead, jnp.bool_),
                is_last=jnp.zeros(lead, jnp.bool_),
                extras={
                    name: jnp.zeros((*lead, *spec.shape), jnp.dtype(spec.dtype))
                    for name, spec in schema
                },
            ),
            size=jnp.zeros((), jnp.int32),
            cursor=jnp.zeros((), jnp.int32),
            capacity=n,
            seq_len=t,
            schema=schema,
        )

    def __len__(self) -> int:
        """Live steps, as a Python int (host-side); inside jit read :attr:`size`."""
        return int(self.size)

    @property
    def windows(self) -> int:
        """How many distinct ``seq_len``-step windows the live stream offers.

        ``size − seq_len + 1``, floored at 0: one window per valid start offset.
        """
        return max(int(self.size) - self.seq_len + 1, 0)

    def ready(self, batch_steps: int) -> bool:
        """Whether the stream offers at least ``batch_steps`` windows yet. Host-side."""
        return self.windows >= int(batch_steps)


def extend(buffer: StreamBuffer, ep: Episode) -> StreamBuffer:
    """One episode's steps spliced onto the stream: a new buffer, nothing mutated.

    Every column, ``extras`` included, is written at the same ring positions; the
    episode keeps its own ``is_first``/``is_last``. Compute any column derived from
    the whole episode before extending. Jit-able, with ``ep.length`` static per
    trace.
    """
    if jnp.ndim(ep.obs) != 2:
        raise ValueError(
            f"extend takes ONE episode: obs (T, obs_dim), got {jnp.shape(ep.obs)} — a batch of "
            "episodes is a loop over this call, since each has its own length"
        )
    _check_extras(buffer, ep)
    t = ep.length
    if t > buffer.capacity:
        raise ValueError(
            f"episode of length {t} exceeds the stream's capacity of {buffer.capacity} steps"
        )
    positions = (buffer.cursor + jnp.arange(t, dtype=jnp.int32)) % buffer.capacity
    data = jax.tree_util.tree_map(
        lambda store, column: store.at[positions].set(column), buffer.data, ep
    )
    return StreamBuffer(
        data=data,
        size=jnp.minimum(buffer.size + t, buffer.capacity),
        cursor=(buffer.cursor + t) % buffer.capacity,
        capacity=buffer.capacity,
        seq_len=buffer.seq_len,
        schema=buffer.schema,
    )


def sample_windows(buffer: StreamBuffer, key: Array, batch_size: int) -> Episode:
    """A uniform batch of ``seq_len``-step windows: leaves ``(batch_size, seq_len, ...)``.

    Start offsets are uniform over valid starts, with replacement; ``is_first``
    marks every episode boundary in the window, and step 0's flag is forced true.
    A stream shorter than one window floors the offset range at one slot, so guard
    with :meth:`StreamBuffer.ready`.
    """
    oldest = (buffer.cursor - buffer.size) % buffer.capacity
    starts = jnp.maximum(buffer.size - buffer.seq_len + 1, 1)
    offset = jax.random.randint(key, (int(batch_size), 1), 0, starts)
    index = (oldest + offset + jnp.arange(buffer.seq_len, dtype=jnp.int32)) % buffer.capacity
    window = jax.tree_util.tree_map(lambda store: store[index], buffer.data)
    return window._replace(is_first=window.is_first.at[:, 0].set(True))


# ================================================================== the two, bridged


def as_episode(columns: Mapping[str, np.ndarray]) -> Episode:
    """A :class:`ReplayBuffer` episode dict → an :class:`Episode`.

    ``obs`` and ``action`` are required; ``reward`` defaults to zeros, ``cont`` to
    ones, ``is_first``/``is_last`` to the first and last row. Every other column
    becomes a task column in ``extras``, with the dtype it arrived in.
    """
    if "obs" not in columns or "action" not in columns:
        raise ValueError(
            f"as_episode: no 'obs'/'action' column (got {sorted(columns)}); those two carry "
            "the episode's shape, so the format cannot be filled in without them"
        )
    t = int(np.asarray(columns["obs"]).shape[0])
    core = {
        "obs": columns["obs"],
        "action": columns["action"],
        "reward": columns.get("reward", jnp.zeros((t,), jnp.float32)),
        "cont": columns.get("cont", jnp.ones((t,), jnp.float32)),
        "is_first": columns.get("is_first", jnp.zeros((t,), jnp.bool_).at[0].set(True)),
        "is_last": columns.get("is_last", jnp.zeros((t,), jnp.bool_).at[-1].set(True)),
    }
    extras = {name: value for name, value in columns.items() if name not in EPISODE_FIELDS}
    return episode(
        core["obs"],
        core["action"],
        core["reward"],
        is_first=core["is_first"],
        is_last=core["is_last"],
        cont=core["cont"],
        **extras,
    )


def episode_columns(ep: Episode) -> dict[str, np.ndarray]:
    """An :class:`Episode` → a column dict for :meth:`ReplayBuffer.write_episode`:
    the engine columns and the task columns, flat, as numpy arrays."""
    out = {name: np.asarray(getattr(ep, name)) for name in EPISODE_FIELDS}
    out.update({name: np.asarray(value) for name, value in ep.extras.items()})
    return out


def stack_episodes(
    episodes: Sequence[Episode], seq_len: int, pad: Mapping[str, str] | None = None
) -> Episode:
    """A list of episodes as one batched :class:`Episode`, each fitted to ``seq_len``
    (:func:`fit_length`, with ``pad`` the task columns' rules)."""
    if not episodes:
        raise ValueError("stack_episodes: no episodes")
    fitted = [fit_length(ep, seq_len, pad=pad) for ep in episodes]
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *fitted)


__all__ = [
    "EPISODE_FIELDS",
    "ColumnSpec",
    "Episode",
    "ReplayBuffer",
    "Sampler",
    "Schema",
    "SeedStats",
    "StaticBuffer",
    "StreamBuffer",
    "add",
    "as_episode",
    "episode",
    "episode_columns",
    "extend",
    "fit_length",
    "sample",
    "sample_windows",
    "seed_stats",
    "stack_episodes",
    "uniform_sampler",
]


# ------------------------------------------------------------- seed statistics


class SeedStats(NamedTuple):
    """Seed-policy quality over a set of episodes.

    ``mean_return`` is the summed reward per episode; ``mean_length`` the steps to
    the first success when a success column is named, else the whole episode;
    ``success_rate`` the fraction with a success step, or ``None`` without a column.
    """

    n_episodes: int
    mean_return: float
    success_rate: float | None
    mean_length: float


def seed_stats(episodes: Sequence[Episode], success: str | None = None) -> SeedStats:
    """Mean return / success rate / mean length over a list of episodes.

    ``success`` names the task column in ``extras`` holding the per-step success
    flag; without it the success rate is ``None`` and lengths are whole episodes.
    """
    if not episodes:
        raise ValueError("seed_stats: no episodes")
    returns = [float(jnp.sum(ep.reward)) for ep in episodes]
    if success is None:
        return SeedStats(
            len(episodes),
            float(np.mean(returns)),
            None,
            float(np.mean([ep.length for ep in episodes])),
        )
    flags = [np.asarray(ep.extras[success]).astype(bool) for ep in episodes]
    lengths = [
        int(np.flatnonzero(f)[0]) + 1 if f.any() else int(ep.length)
        for f, ep in zip(flags, episodes, strict=True)
    ]
    return SeedStats(
        n_episodes=len(episodes),
        mean_return=float(np.mean(returns)),
        success_rate=float(np.mean([bool(f.any()) for f in flags])),
        mean_length=float(np.mean(lengths)),
    )
