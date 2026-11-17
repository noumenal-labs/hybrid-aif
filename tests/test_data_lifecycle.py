"""Tests for empty datasets, non-destructive FolderWriter overwrite, and reader context managers."""

import numpy as np

from hybrid_aif.data import load_dataset
from hybrid_aif.data.formats.folder import FolderReader, FolderWriter
from hybrid_aif.data.formats.hdf5 import HDF5Reader, HDF5Writer


def test_empty_hdf5_dataset_can_be_read_and_appended(tmp_path):
    path = tmp_path / "episodes.h5"
    with HDF5Writer(path, mode="overwrite"):
        pass

    with HDF5Reader(path) as reader:
        assert reader.lengths.shape == (0,)
        assert reader.offsets.shape == (0,)
        assert reader.column_names == []

    with load_dataset(path) as ds:
        assert len(ds) == 0

    ep = {"obs": np.arange(6, dtype=np.float32).reshape(3, 2)}
    with HDF5Writer(path, mode="append") as writer:
        writer.write_episode(ep)

    with load_dataset(path) as ds:
        assert len(ds) == 3
        np.testing.assert_array_equal(ds[0]["obs"], ep["obs"][:1])
    assert ds.reader._file is None


def test_empty_folder_dataset_offsets_and_safe_overwrite(tmp_path):
    root = tmp_path / "folder_ds"
    with FolderWriter(root, mode="overwrite"):
        pass

    with FolderReader(root) as reader:
        assert reader.lengths.shape == (0,)
        assert reader.offsets.shape == (0,)
        assert reader.column_names == []

    ep = {"obs": np.ones((4, 2), dtype=np.float32)}
    with FolderWriter(root, mode="append") as writer:
        writer.write_episode(ep)

    sidecar = root / "metadata.json"
    sidecar.write_text('{"note": "keep"}', encoding="utf-8")

    with FolderWriter(root, mode="overwrite") as writer:
        writer.write_episode({"obs": np.zeros((2, 2), dtype=np.float32)})

    assert sidecar.exists()
    with load_dataset(root) as ds:
        assert len(ds) == 2
