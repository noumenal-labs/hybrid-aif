"""Episode datasets with fixed-size clip windowing.

:class:`Dataset` turns per-episode ``lengths``/``offsets`` into a flat index
of all valid clips — ``clip_indices`` lists every ``(episode, start)`` whose
window of ``num_steps * frameskip`` raw steps fits inside the episode (shorter
episodes contribute nothing). Subclasses implement ``_load_slice`` to return
the dense raw slice; ``__getitem__`` applies frameskip and returns a dict of
numpy arrays. Converting to device arrays (``jnp.asarray``) is the caller's
job at the device boundary.
"""

from __future__ import annotations

import numpy as np

from hybrid_aif.data.format import Reader, detect_format, get_format


class Dataset:
    """Base class for episode-based datasets.

    Args:
        lengths: per-episode step counts.
        offsets: per-episode start offsets in the flat underlying storage.
        num_steps: observation frames per clip.
        frameskip: stride between observation frames within a clip.

    A clip spans ``num_steps * frameskip`` raw steps. ``__getitem__`` strides
    every column by ``frameskip`` except ``action``, which stays dense and is
    reshaped to ``(num_steps, frameskip * action_dim)`` when ``frameskip > 1``
    so each observation frame carries the actions taken until the next one.
    """

    def __init__(self, lengths, offsets, num_steps: int = 1, frameskip: int = 1) -> None:
        if num_steps <= 0 or frameskip <= 0:
            raise ValueError(
                f"num_steps and frameskip must be positive, got {num_steps=} {frameskip=}"
            )
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.num_steps = num_steps
        self.frameskip = frameskip
        self._clip_cum: np.ndarray | None = None
        self._clips: list[tuple[int, int]] | None = None

    @property
    def span(self) -> int:
        return self.num_steps * self.frameskip

    @property
    def _windows(self) -> np.ndarray:
        """Cumulative valid-window counts over episodes, ``(num_episodes,)``, built once
        per change of ``lengths`` (a subclass that appends resets the cache)."""
        if self._clip_cum is None:
            self._clip_cum = np.cumsum(clip_window_counts(self.lengths, self.span))
        return self._clip_cum

    @property
    def clip_indices(self) -> list[tuple[int, int]]:
        """Every valid ``(episode, start)`` in flat index order, as a list."""
        if self._clips is None:
            self._clips = compute_clip_indices(self.lengths, self.span)
        return self._clips

    def locate(self, idx) -> tuple[np.ndarray, np.ndarray]:
        """Flat clip indices → ``(episode, start)`` arrays, vectorised; negative
        indices count from the end, and an index outside ``[-len, len)`` raises."""
        return locate_clips(self._windows, np.asarray(idx))

    @property
    def column_names(self) -> list[str]:
        raise NotImplementedError

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict[str, np.ndarray]:
        """Return the dense raw slice ``[start:end)`` of one episode."""
        raise NotImplementedError

    def __len__(self) -> int:
        cum = self._windows
        return int(cum[-1]) if cum.size else 0

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        ep_idx, start = self.locate(idx)
        return self._clip(int(ep_idx), int(start))

    def _clip(self, ep_idx: int, start: int) -> dict[str, np.ndarray]:
        steps = self._load_slice(ep_idx, start, start + self.span)
        if self.frameskip == 1:
            return steps
        out = {}
        for col, arr in steps.items():
            if col == "action":
                out[col] = arr.reshape(self.num_steps, -1)
            else:
                out[col] = arr[:: self.frameskip]
        return out

    def load_episode(self, ep_idx: int) -> dict[str, np.ndarray]:
        return self._load_slice(ep_idx, 0, int(self.lengths[ep_idx]))


def clip_window_counts(lengths: np.ndarray, span: int) -> np.ndarray:
    """How many windows of ``span`` raw steps each episode holds, ``(num_episodes,)``
    int64; an episode shorter than ``span`` holds none."""
    lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
    return np.maximum(lengths - int(span) + 1, 0)


def locate_clips(cum: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flat clip indices → ``(episode, start)`` under cumulative window counts ``cum``
    (:func:`clip_window_counts` summed), in the order :func:`compute_clip_indices`
    lists them. Negative indices count from the end; out-of-range ones raise."""
    total = int(cum[-1]) if cum.size else 0
    idx = np.asarray(idx, dtype=np.int64)
    flat = np.where(idx < 0, idx + total, idx)
    if flat.size and (flat.min() < 0 or flat.max() >= total):
        raise IndexError(f"clip index out of range for a dataset of {total} clips")
    eps = np.searchsorted(cum, flat, side="right")
    before = np.where(eps > 0, cum[np.maximum(eps - 1, 0)], 0)
    return eps, flat - before


def compute_clip_indices(lengths: np.ndarray, span: int) -> list[tuple[int, int]]:
    """All valid ``(episode, start)`` windows of ``span`` raw steps, episodes in order
    and starts ascending within each."""
    cum = np.cumsum(clip_window_counts(lengths, span))
    total = int(cum[-1]) if cum.size else 0
    eps, starts = locate_clips(cum, np.arange(total))
    return list(zip(eps.tolist(), starts.tolist(), strict=True))


class ReaderDataset(Dataset):
    """Dataset over any registry :class:`Reader`."""

    def __init__(self, reader: Reader, num_steps: int = 1, frameskip: int = 1) -> None:
        self.reader = reader
        super().__init__(reader.lengths, reader.offsets, num_steps, frameskip)

    @property
    def column_names(self) -> list[str]:
        return self.reader.column_names

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict[str, np.ndarray]:
        return self.reader.read_slice(ep_idx, start, end)


class HDF5Dataset(ReaderDataset):
    def __init__(self, path, num_steps: int = 1, frameskip: int = 1) -> None:
        from hybrid_aif.data.formats.hdf5 import HDF5Reader

        super().__init__(HDF5Reader(path), num_steps, frameskip)


class FolderDataset(ReaderDataset):
    def __init__(self, path, num_steps: int = 1, frameskip: int = 1) -> None:
        from hybrid_aif.data.formats.folder import FolderReader

        super().__init__(FolderReader(path), num_steps, frameskip)


def load_dataset(
    path, *, format: str | None = None, num_steps: int = 1, frameskip: int = 1
) -> ReaderDataset:
    """Open ``path`` as a Dataset, autodetecting the format unless given."""
    import hybrid_aif.data.formats  # noqa: F401  (registers built-in formats)

    fmt = get_format(format) if format is not None else detect_format(path)
    if fmt is None:
        raise ValueError(f"could not detect a registered dataset format at '{path}'")
    return ReaderDataset(fmt.open_reader(path), num_steps=num_steps, frameskip=frameskip)


__all__ = [
    "Dataset",
    "FolderDataset",
    "HDF5Dataset",
    "ReaderDataset",
    "clip_window_counts",
    "compute_clip_indices",
    "load_dataset",
    "locate_clips",
]
