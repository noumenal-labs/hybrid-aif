"""Format registry: a declarative read/write abstraction over dataset backends.

A ``Format`` ties together three concerns for one on-disk layout:

- ``name`` — the registry key, also usable as an explicit ``format=`` argument.
- ``detect(path)`` — does this path look like this format?
- ``open_reader(path)`` / ``open_writer(path, mode)`` — entry points.

To add a backend, subclass :class:`Format` in a module under ``data/formats/``
and decorate it with :func:`register_format`. Read-only formats omit
``open_writer``; write-only formats omit ``open_reader``.

Column convention
-----------------
An *episode* is a ``dict[str, np.ndarray]`` of named columns sharing a leading
time axis (``column.shape[0] == episode_length``). Keys starting with ``_``
are transient (e.g. debug info) and are skipped by all writers. Temporal
alignment of ``action`` relative to observations (whether ``action[t]`` was
taken *at* or *produced* ``obs[t]``) is the caller's concern: the pipeline
stores and windows columns positionally and never reinterprets them.

Writers are streaming: ``write_episodes`` consumes its iterable one episode at
a time with bounded memory. All writers share the modes in :data:`WRITE_MODES`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol, runtime_checkable

import numpy as np

FORMATS: dict[str, type[Format]] = {}

WRITE_MODES = ("append", "overwrite", "error")
"""Writer modes shared by all formats.

- ``append``: extend an existing dataset (schema-validated) or create one.
- ``overwrite``: drop any existing dataset and start fresh.
- ``error``: raise :class:`FileExistsError` if a dataset already exists.
"""


def validate_write_mode(mode: str) -> str:
    if mode not in WRITE_MODES:
        raise ValueError(f"write mode must be one of {WRITE_MODES}, got {mode!r}")
    return mode


def register_format(cls: type[Format]) -> type[Format]:
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} must set a non-empty `name` class attribute")
    if name in FORMATS:
        raise ValueError(f"format {name!r} is already registered")
    FORMATS[name] = cls
    return cls


def list_formats() -> list[str]:
    return list(FORMATS)


def get_format(name: str) -> type[Format]:
    try:
        return FORMATS[name]
    except KeyError:
        raise ValueError(f"unknown format {name!r}; available: {list_formats()}") from None


def detect_format(path) -> type[Format] | None:
    """Return the first registered format whose ``detect`` matches, else None."""
    for fmt in FORMATS.values():
        if fmt.detect(path):
            return fmt
    return None


class Format:
    """Declarative format spec. Subclasses set ``name`` and implement ``detect``,
    plus ``open_reader`` and/or ``open_writer``."""

    name: str = ""

    @classmethod
    def detect(cls, path) -> bool:
        raise NotImplementedError(f"{cls.__name__}.detect must be implemented")

    @classmethod
    def open_reader(cls, path, **kwargs) -> Reader:
        raise NotImplementedError(f"format {cls.name or cls.__name__!r} does not support reading")

    @classmethod
    def open_writer(cls, path, *, mode: str = "append", **kwargs) -> Writer:
        raise NotImplementedError(f"format {cls.name or cls.__name__!r} does not support writing")


class Reader(Protocol):
    """Random-access episode reader over flat per-column storage."""

    @property
    def lengths(self) -> np.ndarray: ...
    @property
    def offsets(self) -> np.ndarray: ...
    @property
    def column_names(self) -> list[str]: ...
    def read_slice(self, ep_idx: int, start: int, end: int) -> dict[str, np.ndarray]: ...


@runtime_checkable
class Writer(Protocol):
    """Streaming episode writer — append episodes, finalize on exit.

    ``write_episodes`` must consume its iterable lazily (one episode resident
    at a time); ``write_episode`` pushes a single episode. Keys starting with
    ``_`` are skipped.
    """

    def __enter__(self) -> Writer: ...
    def __exit__(self, *exc) -> None: ...
    def write_episode(self, ep_data: dict[str, np.ndarray]) -> None: ...
    def write_episodes(self, episodes: Iterable[dict[str, np.ndarray]]) -> None: ...


def public_items(ep_data: dict) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (column, array) pairs, skipping ``_``-prefixed keys and coercing to numpy."""
    for col, vals in ep_data.items():
        if not col.startswith("_"):
            yield col, np.asarray(vals)


def episode_schema(ep_data: dict) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    """Map each public column to its (per-step shape, dtype); validates the time axis."""
    schema: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    ep_len: int | None = None
    for col, arr in public_items(ep_data):
        if arr.ndim == 0:
            raise ValueError(f"column {col!r} is scalar; expected a leading time axis")
        if ep_len is None:
            ep_len = arr.shape[0]
        elif arr.shape[0] != ep_len:
            raise ValueError(
                f"column {col!r} has length {arr.shape[0]}, but other columns have {ep_len}"
            )
        schema[col] = (arr.shape[1:], arr.dtype)
    if not schema:
        raise ValueError("episode has no public (non-underscore) columns")
    return schema


def check_schema(
    expected: dict[str, tuple[tuple[int, ...], np.dtype]],
    ep_data: dict,
    context: str,
) -> None:
    """Raise if ``ep_data``'s public columns don't match ``expected`` exactly."""
    incoming = episode_schema(ep_data)
    if set(incoming) != set(expected):
        raise ValueError(
            f"{context}: schema mismatch — missing columns "
            f"{sorted(set(expected) - set(incoming))}, "
            f"unexpected columns {sorted(set(incoming) - set(expected))}"
        )
    for col, (shape, dtype) in incoming.items():
        exp_shape, exp_dtype = expected[col]
        if shape != exp_shape or dtype != exp_dtype:
            raise ValueError(
                f"{context}: column {col!r} mismatch — expected per-step shape "
                f"{exp_shape} dtype {exp_dtype}, got shape {shape} dtype {dtype}"
            )


__all__ = [
    "FORMATS",
    "WRITE_MODES",
    "Format",
    "Reader",
    "Writer",
    "check_schema",
    "detect_format",
    "episode_schema",
    "get_format",
    "list_formats",
    "public_items",
    "register_format",
    "validate_write_mode",
]
