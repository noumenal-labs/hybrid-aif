"""HDF5 backend: a single ``.h5`` file with one resizable dataset per column.

Layout: episodes are stored as contiguous rows in flat per-column datasets,
indexed by two metadata datasets ``ep_len`` / ``ep_offset``. We follow
stable-worldmodel here rather than one-group-per-episode: flat rows give O(1)
slice reads through global offsets (exactly matching the Dataset windowing
math), resizable appends for streaming writers, and none of the per-group
metadata overhead that thousands of episode groups would incur.

Requires the ``data`` extra (``pip install "hybrid-aif[data]"``).
"""

from __future__ import annotations

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

_META_KEYS = ("ep_len", "ep_offset")


def _h5py():
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required for the hdf5 format; install the 'data' extra: "
            'pip install "hybrid-aif[data]"'
        ) from e
    return h5py


def _file_schema(f) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    return {k: (f[k].shape[1:], f[k].dtype) for k in f.keys() if k not in _META_KEYS}


class HDF5Reader:
    """Random-access reader over a single ``.h5`` file (lazy file handle)."""

    def __init__(self, path) -> None:
        self.path = Path(path)
        self._file = None
        with _h5py().File(self.path, "r") as f:
            self.lengths = np.asarray(f["ep_len"][:], dtype=np.int64)
            self.offsets = np.asarray(f["ep_offset"][:], dtype=np.int64)
            self._columns = [k for k in f.keys() if k not in _META_KEYS]

    @property
    def column_names(self) -> list[str]:
        return self._columns

    def _open(self):
        if self._file is None:
            self._file = _h5py().File(self.path, "r")
        return self._file

    def read_slice(self, ep_idx: int, start: int, end: int) -> dict[str, np.ndarray]:
        f = self._open()
        g0 = int(self.offsets[ep_idx])
        return {col: f[col][g0 + start : g0 + end] for col in self._columns}

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class HDF5Writer:
    """Streaming writer appending episodes to a single ``.h5`` file.

    Schema is inferred from the first episode and locked; every episode is
    validated against it. In append mode the lock comes from the existing file.
    """

    def __init__(self, path, *, mode: str = "append") -> None:
        validate_write_mode(mode)
        self.path = Path(path)
        self.mode = mode
        self._f = None
        self._schema: dict[str, tuple[tuple[int, ...], np.dtype]] | None = None
        self._ptr = 0

    def __enter__(self) -> HDF5Writer:
        h5py = _h5py()
        exists = self.path.exists()
        if exists and self.mode == "error":
            raise FileExistsError(
                f"{self.path} already exists; use mode='overwrite' or mode='append'"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if exists and self.mode == "append":
            self._f = h5py.File(self.path, "a")
            lengths = self._f["ep_len"][:]
            self._ptr = int(lengths.sum()) if len(lengths) else 0
            self._schema = _file_schema(self._f)
        else:
            self._f = h5py.File(self.path, "w")
        return self

    def __exit__(self, *exc) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def write_episode(self, ep_data: dict) -> None:
        if self._f is None:
            raise RuntimeError("HDF5Writer must be used as a context manager")
        if self._schema is None:
            self._schema = episode_schema(ep_data)
            for col, (shape, dtype) in self._schema.items():
                self._f.create_dataset(col, shape=(0, *shape), maxshape=(None, *shape), dtype=dtype)
            for key in _META_KEYS:
                self._f.create_dataset(key, shape=(0,), maxshape=(None,), dtype=np.int64)
        else:
            check_schema(self._schema, ep_data, f"hdf5 append to '{self.path}'")

        ep_len = None
        for col, arr in public_items(ep_data):
            ep_len = arr.shape[0]
            ds = self._f[col]
            ds.resize(self._ptr + ep_len, axis=0)
            ds[self._ptr : self._ptr + ep_len] = arr
        n = self._f["ep_len"].shape[0]
        for key, val in zip(_META_KEYS, (ep_len, self._ptr), strict=True):
            self._f[key].resize(n + 1, axis=0)
            self._f[key][n] = val
        self._ptr += ep_len

    def write_episodes(self, episodes) -> None:
        for ep in episodes:
            self.write_episode(ep)


@register_format
class HDF5(Format):
    name = "hdf5"

    @classmethod
    def detect(cls, path) -> bool:
        return Path(path).suffix in (".h5", ".hdf5")

    @classmethod
    def open_reader(cls, path, **kwargs) -> HDF5Reader:
        return HDF5Reader(path, **kwargs)

    @classmethod
    def open_writer(cls, path, *, mode: str = "append", **kwargs) -> HDF5Writer:
        return HDF5Writer(path, mode=mode, **kwargs)


__all__ = ["HDF5", "HDF5Reader", "HDF5Writer"]
