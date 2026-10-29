"""Folder backend: a directory of one ``.npz`` file per episode.

Layout::

    <root>/
      ep_len.npy            # int64 episode lengths (index, rewritten on close)
      episode_0000000.npz   # one uncompressed npz per episode, one member per column

This diverges from stable-worldmodel's folder format (one whole-dataset ``.npz``
per column, buffered in memory by the writer): per-episode files keep the
writer streaming with bounded memory and preserve dtypes/shapes exactly with
no image-specific special cases. Only the folder and HDF5 backends are registered.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from hybrid_aif.data.format import (
    Format,
    check_schema,
    episode_schema,
    public_items,
    register_format,
    validate_write_mode,
)

_INDEX = "ep_len.npy"


def _episode_files(root: Path) -> list[Path]:
    return sorted(root.glob("episode_*.npz"))


class FolderReader:
    """Random-access reader over a directory of per-episode ``.npz`` files."""

    def __init__(self, path) -> None:
        self.path = Path(path)
        self.lengths = np.load(self.path / _INDEX).astype(np.int64)
        self.offsets = np.concatenate(([0], np.cumsum(self.lengths[:-1]))).astype(np.int64)
        self._files = _episode_files(self.path)
        if len(self._files) != len(self.lengths):
            raise ValueError(
                f"folder dataset '{self.path}' is inconsistent: {len(self._files)} episode "
                f"files but {len(self.lengths)} index entries"
            )
        self._columns: list[str] = []
        if self._files:
            with np.load(self._files[0]) as f:
                self._columns = sorted(f.files)
        self._cache: tuple[int, dict[str, np.ndarray]] | None = None

    @property
    def column_names(self) -> list[str]:
        return self._columns

    def _episode(self, ep_idx: int) -> dict[str, np.ndarray]:
        if self._cache is None or self._cache[0] != ep_idx:
            with np.load(self._files[ep_idx]) as f:
                self._cache = (ep_idx, {col: f[col] for col in f.files})
        return self._cache[1]

    def read_slice(self, ep_idx: int, start: int, end: int) -> dict[str, np.ndarray]:
        ep = self._episode(ep_idx)
        return {col: arr[start:end] for col, arr in ep.items()}


class FolderWriter:
    """Streaming writer: one ``.npz`` per episode, index rewritten on exit."""

    def __init__(self, path, *, mode: str = "append") -> None:
        validate_write_mode(mode)
        self.path = Path(path)
        self.mode = mode
        self._schema: dict[str, tuple[tuple[int, ...], np.dtype]] | None = None
        self._lengths: list[int] = []
        self._entered = False

    def __enter__(self) -> FolderWriter:
        exists = (self.path / _INDEX).exists()
        if exists and self.mode == "error":
            raise FileExistsError(
                f"{self.path} already contains a dataset; use mode='overwrite' or mode='append'"
            )
        if exists and self.mode == "overwrite":
            shutil.rmtree(self.path)
            exists = False
        self.path.mkdir(parents=True, exist_ok=True)
        if exists and self.mode == "append":
            self._lengths = np.load(self.path / _INDEX).astype(np.int64).tolist()
            files = _episode_files(self.path)
            if files:
                with np.load(files[0]) as f:
                    self._schema = {col: (f[col].shape[1:], f[col].dtype) for col in f.files}
        self._entered = True
        return self

    def __exit__(self, *exc) -> None:
        np.save(self.path / _INDEX, np.asarray(self._lengths, dtype=np.int64))
        self._entered = False

    def write_episode(self, ep_data: dict) -> None:
        if not self._entered:
            raise RuntimeError("FolderWriter must be used as a context manager")
        if self._schema is None:
            self._schema = episode_schema(ep_data)
        else:
            check_schema(self._schema, ep_data, f"folder append to '{self.path}'")
        episode = dict(public_items(ep_data))
        ep_len = next(iter(episode.values())).shape[0]
        np.savez(self.path / f"episode_{len(self._lengths):07d}.npz", **episode)
        self._lengths.append(ep_len)

    def write_episodes(self, episodes) -> None:
        for ep in episodes:
            self.write_episode(ep)


@register_format
class Folder(Format):
    name = "folder"

    @classmethod
    def detect(cls, path) -> bool:
        p = Path(path)
        return p.is_dir() and (p / _INDEX).exists()

    @classmethod
    def open_reader(cls, path, **kwargs) -> FolderReader:
        return FolderReader(path, **kwargs)

    @classmethod
    def open_writer(cls, path, *, mode: str = "append", **kwargs) -> FolderWriter:
        return FolderWriter(path, mode=mode, **kwargs)


__all__ = ["Folder", "FolderReader", "FolderWriter"]
