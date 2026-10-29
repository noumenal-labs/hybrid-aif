"""Tests for the data pipeline: format registry, backends, datasets, replay buffer."""

import numpy as np
import pytest

from hybrid_aif.data import (
    Dataset,
    FolderDataset,
    HDF5Dataset,
    Writer,
    detect_format,
    get_format,
    list_formats,
    load_dataset,
)
from hybrid_aif.data.dataset import compute_clip_indices
from hybrid_aif.data.formats.folder import Folder
from hybrid_aif.data.formats.hdf5 import HDF5
from hybrid_aif.train.buffers import ReplayBuffer

FORMAT_NAMES = ["hdf5", "folder"]


def make_episode(length, obs_dim=3, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "obs": rng.normal(size=(length, obs_dim)).astype(np.float32),
        "action": rng.integers(0, 4, size=(length, 2)).astype(np.int32),
        "done": np.zeros(length, dtype=bool),
        "_debug": np.arange(length),
    }


def make_episodes(lengths, seed=0):
    return [make_episode(ln, seed=seed + i) for i, ln in enumerate(lengths)]


def target(tmp_path, fmt_name):
    return tmp_path / ("data.h5" if fmt_name == "hdf5" else "data_dir")


def write(path, fmt_name, episodes, mode="append"):
    with get_format(fmt_name).open_writer(path, mode=mode) as w:
        w.write_episodes(episodes)


def assert_episode_equal(actual, expected):
    expected = {k: np.asarray(v) for k, v in expected.items() if not k.startswith("_")}
    assert set(actual) == set(expected)
    for col, arr in expected.items():
        assert actual[col].dtype == arr.dtype, col
        assert actual[col].shape == arr.shape, col
        np.testing.assert_array_equal(actual[col], arr)


# -- registry ---------------------------------------------------------------


def test_registry_lists_builtin_formats():
    assert {"hdf5", "folder"} <= set(list_formats())
    assert get_format("hdf5") is HDF5
    assert get_format("folder") is Folder


def test_get_format_unknown_raises():
    with pytest.raises(ValueError, match="unknown format"):
        get_format("no-such-format")


def test_detect_format(tmp_path):
    eps = make_episodes([4])
    h5 = target(tmp_path, "hdf5")
    folder = target(tmp_path, "folder")
    write(h5, "hdf5", eps)
    write(folder, "folder", eps)
    assert detect_format(h5) is HDF5
    assert detect_format(folder) is Folder
    assert detect_format(tmp_path / "nothing.xyz") is None


# -- round-trip ---------------------------------------------------------------


@pytest.mark.parametrize("fmt_name", FORMAT_NAMES)
def test_round_trip_multi_episode_mixed_dtypes(tmp_path, fmt_name):
    episodes = make_episodes([5, 3, 7])
    path = target(tmp_path, fmt_name)
    write(path, fmt_name, episodes)

    ds = load_dataset(path)
    np.testing.assert_array_equal(ds.lengths, [5, 3, 7])
    np.testing.assert_array_equal(ds.offsets, [0, 5, 8])
    assert set(ds.column_names) == {"obs", "action", "done"}  # _debug skipped
    for i, ep in enumerate(episodes):
        assert_episode_equal(ds.load_episode(i), ep)


@pytest.mark.parametrize("fmt_name", FORMAT_NAMES)
def test_append_mode_extends(tmp_path, fmt_name):
    episodes = make_episodes([4, 6, 5])
    path = target(tmp_path, fmt_name)
    write(path, fmt_name, episodes[:2])
    write(path, fmt_name, episodes[2:], mode="append")

    ds = load_dataset(path)
    np.testing.assert_array_equal(ds.lengths, [4, 6, 5])
    for i, ep in enumerate(episodes):
        assert_episode_equal(ds.load_episode(i), ep)


@pytest.mark.parametrize("fmt_name", FORMAT_NAMES)
def test_append_schema_validation(tmp_path, fmt_name):
    path = target(tmp_path, fmt_name)
    write(path, fmt_name, make_episodes([4]))

    bad_cols = {"obs": np.zeros((3, 3), np.float32)}  # missing action/done
    with pytest.raises(ValueError, match="schema mismatch"):
        write(path, fmt_name, [bad_cols], mode="append")

    bad_shape = make_episode(3, obs_dim=5)
    with pytest.raises(ValueError, match="obs"):
        write(path, fmt_name, [bad_shape], mode="append")

    bad_dtype = make_episode(3)
    bad_dtype["obs"] = bad_dtype["obs"].astype(np.float64)
    with pytest.raises(ValueError, match="obs"):
        write(path, fmt_name, [bad_dtype], mode="append")


@pytest.mark.parametrize("fmt_name", FORMAT_NAMES)
def test_mid_stream_schema_validation(tmp_path, fmt_name):
    path = target(tmp_path, fmt_name)
    with pytest.raises(ValueError, match="schema mismatch"):
        with get_format(fmt_name).open_writer(path) as w:
            w.write_episode(make_episode(4))
            w.write_episode({"other": np.zeros((3, 2), np.float32)})


@pytest.mark.parametrize("fmt_name", FORMAT_NAMES)
def test_error_and_overwrite_modes(tmp_path, fmt_name):
    path = target(tmp_path, fmt_name)
    write(path, fmt_name, make_episodes([4, 5]))

    with pytest.raises(FileExistsError):
        get_format(fmt_name).open_writer(path, mode="error").__enter__()

    fresh = make_episodes([2], seed=99)
    write(path, fmt_name, fresh, mode="overwrite")
    ds = load_dataset(path)
    np.testing.assert_array_equal(ds.lengths, [2])
    assert_episode_equal(ds.load_episode(0), fresh[0])


def test_invalid_write_mode(tmp_path):
    with pytest.raises(ValueError, match="write mode"):
        get_format("folder").open_writer(tmp_path / "d", mode="clobber")


# -- streaming ----------------------------------------------------------------


def test_streaming_write_bounded_memory(tmp_path):
    path = tmp_path / "stream_dir"
    consumed = 0

    def episode_gen():
        nonlocal consumed
        for i in range(100):
            # All previously yielded episodes must already be on disk: the
            # writer pulls one episode at a time instead of materializing all.
            assert len(list(path.glob("episode_*.npz"))) == i
            consumed += 1
            yield make_episode(3, seed=i)

    write(path, "folder", episode_gen())
    assert consumed == 100
    ds = load_dataset(path)
    assert len(ds.lengths) == 100


def test_streaming_write_hdf5_100_episodes(tmp_path):
    path = tmp_path / "stream.h5"
    gen = (make_episode(3, seed=i) for i in range(100))
    write(path, "hdf5", gen)
    ds = load_dataset(path)
    np.testing.assert_array_equal(ds.lengths, np.full(100, 3))
    assert_episode_equal(ds.load_episode(42), make_episode(3, seed=42))


# -- dataset windowing ---------------------------------------------------------


def test_clip_indices_basic():
    assert compute_clip_indices(np.array([5, 2, 4]), span=2) == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
        (2, 0),
        (2, 1),
        (2, 2),
    ]


def test_clip_indices_frameskip_and_short_episodes():
    # num_steps=2, frameskip=2 -> span 4; episode of length 2 is excluded.
    assert compute_clip_indices(np.array([5, 2, 4]), span=4) == [
        (0, 0),
        (0, 1),
        (2, 0),
    ]


def _clip_indices_by_comprehension(lengths, span):
    """The clip table as a Python list comprehension: the reference for the
    cumulative-count table."""
    return [
        (ep, start)
        for ep, length in enumerate(lengths)
        if length >= span
        for start in range(int(length) - span + 1)
    ]


@pytest.mark.parametrize("span", [1, 2, 4, 7])
def test_clip_indices_match_the_list_comprehension(span):
    for lengths in ([5, 2, 4, 0, 3, 6], [3], [], [1, 1, 1]):
        expected = _clip_indices_by_comprehension(lengths, span)
        assert compute_clip_indices(np.array(lengths, dtype=np.int64), span) == expected


def test_buffer_samples_the_clips_the_list_index_would(monkeypatch):
    for num_steps, frameskip in ((2, 1), (2, 2), (3, 1)):
        buf = ReplayBuffer(max_steps=40, num_steps=num_steps, frameskip=frameskip)
        buf.write_episodes(make_episodes([5, 2, 4, 3, 6]))
        span = num_steps * frameskip
        table = _clip_indices_by_comprehension(buf.lengths, span)
        assert len(buf) == len(table)
        assert buf.clip_indices == table

        def by_table(i, table=table, buf=buf):
            ep, start = table[i]
            steps = buf._load_slice(ep, start, start + buf.span)
            if buf.frameskip == 1:
                return steps
            return {
                col: arr.reshape(buf.num_steps, -1) if col == "action" else arr[:: buf.frameskip]
                for col, arr in steps.items()
            }

        for i in list(range(len(table))) + [-1, -len(table)]:  # every start, both ends
            assert_episode_equal(buf[i], by_table(i))
        idx = np.array([0, len(table) - 1, 3, 3, 1, len(table) // 2, -1])
        buf.sampler = lambda b, n, rng, idx=idx: idx
        batch = buf.sample(len(idx))
        for b, i in enumerate(idx):
            want = by_table(int(i))
            for col in buf.column_names:
                np.testing.assert_array_equal(batch[col][b], want[col])
        with pytest.raises(IndexError):
            buf[len(table)]


def test_buffer_clip_table_follows_eviction():
    buf = ReplayBuffer(max_steps=10, num_steps=3)
    for i in range(5):
        buf.write_episode(make_episode(4, seed=i))
        table = _clip_indices_by_comprehension(buf.lengths, buf.span)
        assert len(buf) == len(table) and buf.clip_indices == table
        assert [tuple(map(int, buf.locate(i))) for i in range(len(buf))] == table


@pytest.mark.parametrize("fmt_name", FORMAT_NAMES)
def test_getitem_returns_numpy_clip(tmp_path, fmt_name):
    episodes = make_episodes([6, 2])
    path = target(tmp_path, fmt_name)
    write(path, fmt_name, episodes)

    ds = load_dataset(path, num_steps=3)
    assert isinstance(ds, Dataset)
    assert len(ds) == 4  # episode 1 (length 2) too short for span 3
    clip = ds[1]
    assert isinstance(clip["obs"], np.ndarray)
    np.testing.assert_array_equal(clip["obs"], episodes[0]["obs"][1:4])
    np.testing.assert_array_equal(clip["action"], episodes[0]["action"][1:4])


@pytest.mark.parametrize("fmt_name", FORMAT_NAMES)
def test_getitem_frameskip_strides_obs_keeps_action_dense(tmp_path, fmt_name):
    episodes = make_episodes([8])
    path = target(tmp_path, fmt_name)
    write(path, fmt_name, episodes)

    ds = load_dataset(path, num_steps=2, frameskip=3)
    assert len(ds) == 3  # 8 - 6 + 1
    clip = ds[0]
    np.testing.assert_array_equal(clip["obs"], episodes[0]["obs"][0:6:3])
    assert clip["action"].shape == (2, 6)  # dense, reshaped (num_steps, frameskip * act_dim)
    np.testing.assert_array_equal(clip["action"], episodes[0]["action"][:6].reshape(2, 6))


def test_concrete_dataset_classes(tmp_path):
    episodes = make_episodes([4])
    h5 = target(tmp_path, "hdf5")
    folder = target(tmp_path, "folder")
    write(h5, "hdf5", episodes)
    write(folder, "folder", episodes)
    for ds in (HDF5Dataset(h5), FolderDataset(folder)):
        assert_episode_equal(ds.load_episode(0), episodes[0])


def test_load_dataset_autodetect_and_explicit(tmp_path):
    episodes = make_episodes([4])
    path = target(tmp_path, "hdf5")
    write(path, "hdf5", episodes)
    assert_episode_equal(load_dataset(path).load_episode(0), episodes[0])
    assert_episode_equal(load_dataset(path, format="hdf5").load_episode(0), episodes[0])
    with pytest.raises(ValueError, match="could not detect"):
        load_dataset(tmp_path / "unknown.xyz")


# -- replay buffer --------------------------------------------------------------


def test_buffer_is_dataset_and_writer():
    buf = ReplayBuffer(max_steps=20)
    assert isinstance(buf, Dataset)
    assert isinstance(buf, Writer)
    with buf as w:
        w.write_episodes(make_episodes([4, 3]))
    assert buf.num_episodes == 2
    np.testing.assert_array_equal(buf.lengths, [4, 3])
    np.testing.assert_array_equal(buf.offsets, [0, 4])
    assert len(buf) == 7  # num_steps=1: every step is a clip
    assert_episode_equal(buf.load_episode(0), make_episode(4, seed=0))


def test_buffer_skips_private_keys_and_validates_schema():
    buf = ReplayBuffer(max_steps=20)
    buf.write_episode(make_episode(4))
    assert set(buf.column_names) == {"obs", "action", "done"}
    with pytest.raises(ValueError, match="schema mismatch"):
        buf.write_episode({"obs": np.zeros((2, 3), np.float32)})


def test_buffer_whole_episode_fifo_eviction():
    buf = ReplayBuffer(max_steps=10)
    buf.write_episodes(make_episodes([4, 4], seed=0))
    buf.write_episode(make_episode(4, seed=2))  # evicts episode 0 whole
    assert buf.num_episodes == 2
    assert buf.num_steps_stored == 8 <= buf.max_steps
    assert_episode_equal(buf.load_episode(0), make_episode(4, seed=1))
    assert_episode_equal(buf.load_episode(1), make_episode(4, seed=2))

    buf.write_episode(make_episode(9, seed=3))  # evicts both remaining episodes
    assert buf.num_episodes == 1
    assert buf.num_steps_stored == 9
    assert_episode_equal(buf.load_episode(0), make_episode(9, seed=3))


def test_buffer_rejects_episode_over_budget():
    buf = ReplayBuffer(max_steps=5)
    with pytest.raises(ValueError, match="exceeds max_steps"):
        buf.write_episode(make_episode(6))


def test_buffer_getitem_after_ring_wraparound():
    buf = ReplayBuffer(max_steps=10, num_steps=3)
    for i in range(5):
        buf.write_episode(make_episode(4, seed=i))
    assert buf.num_episodes == 2
    clip = buf[0]  # first clip of the oldest surviving episode
    np.testing.assert_array_equal(clip["obs"], make_episode(4, seed=3)["obs"][:3])


def test_buffer_sampler_hook():
    buf = ReplayBuffer(max_steps=20, num_steps=2, sampler=lambda b, n, rng: np.zeros(n, int))
    buf.write_episodes(make_episodes([5, 4]))
    batch = buf.sample(3)
    assert batch["obs"].shape == (3, 2, 3)
    expected = buf[0]["obs"]
    for b in range(3):
        np.testing.assert_array_equal(batch["obs"][b], expected)


def test_buffer_default_sampler_shapes():
    buf = ReplayBuffer(max_steps=50, num_steps=2, seed=0)
    buf.write_episodes(make_episodes([6, 7]))
    batch = buf.sample(8)
    assert batch["obs"].shape == (8, 2, 3)
    assert batch["action"].shape == (8, 2, 2)
    assert batch["done"].dtype == np.bool_


@pytest.mark.parametrize("fmt_name", FORMAT_NAMES)
def test_buffer_dump_reload_equality(tmp_path, fmt_name):
    episodes = make_episodes([4, 6, 3])
    buf = ReplayBuffer(max_steps=20)
    buf.write_episodes(episodes)
    path = target(tmp_path, fmt_name)
    buf.dump(path, format=fmt_name)

    ds = load_dataset(path)
    np.testing.assert_array_equal(ds.lengths, buf.lengths)
    for i, ep in enumerate(episodes):
        assert_episode_equal(ds.load_episode(i), ep)
