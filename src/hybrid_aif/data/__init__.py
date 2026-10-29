"""Torch-free data pipeline: format registry, episode datasets, readers and writers.

Formats and datasets only; replay lives in :mod:`hybrid_aif.train.buffers`, which
imports this package, never the reverse.

Pure numpy end to end — ``jnp.asarray`` at the device boundary is the
caller's job.
"""

from hybrid_aif.data import formats  # noqa: F401  (registers built-in formats)
from hybrid_aif.data.dataset import (
    Dataset,
    FolderDataset,
    HDF5Dataset,
    ReaderDataset,
    load_dataset,
)
from hybrid_aif.data.format import (
    FORMATS,
    WRITE_MODES,
    Format,
    Reader,
    Writer,
    detect_format,
    get_format,
    list_formats,
    register_format,
)

__all__ = [
    "FORMATS",
    "WRITE_MODES",
    "Dataset",
    "FolderDataset",
    "Format",
    "HDF5Dataset",
    "Reader",
    "ReaderDataset",
    "Writer",
    "detect_format",
    "get_format",
    "list_formats",
    "load_dataset",
    "register_format",
]
