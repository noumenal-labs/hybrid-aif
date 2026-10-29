"""Replay in both regimes, ``alternate``, the fixture format and the fallback seeder.

Design: ``docs/design/amortization.md`` A10 disposition 9 (alternation stays in the
caller's loop; a three-line ``alternate`` ships) and A12 item 6 (expert seeding by
oracle-fixture export, with the bang-bang controller as the documented fallback and
seed stats reported either way), plus the 2026-08-28 consolidation that put both
buffers in :mod:`hybrid_aif.train.buffers` — host-side/dynamic ``ReplayBuffer`` and
device-side/static ``StaticBuffer``, bridged by ``as_episode``/``episode_columns``.
The third, ``StreamBuffer``, is the device-side buffer's other *sequence* contract:
a flat step stream whose windows cross episode boundaries, which is R-AIF's own
replay and what closed FLAG 7 (the padding fraction).
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.reference.envs.mountain_car import MountainCar
from hybrid_aif.reference.raif.columns import EXTRAS, RECORDED, raif_episode
from hybrid_aif.reference.raif.fixtures import load_fixture, provenance_path, save_fixture
from hybrid_aif.reference.raif.run import alternate
from hybrid_aif.reference.raif.seed import bang_bang, bang_bang_episodes, collect_episode
from hybrid_aif.train.buffers import (
    EPISODE_FIELDS,
    ReplayBuffer,
    StaticBuffer,
    StreamBuffer,
    add,
    as_episode,
    episode_columns,
    extend,
    fit_length,
    sample,
    sample_windows,
    seed_stats,
    stack_episodes,
)

KEY = jax.random.PRNGKey(0)
SHORT_ENV = MountainCar(time_limit=90)
PAD = {name: spec.pad for name, spec in EXTRAS.items()}


def _episode(t: int, first_action: float = 0.0):
    return raif_episode(
        obs=jnp.arange(2 * t, dtype=jnp.float32).reshape(t, 2),
        action=jnp.full((t, 1), first_action),
        reward=jnp.full((t,), -2.0),
        is_success=jnp.zeros((t,), bool).at[-1].set(True),
        has_succeeded=jnp.zeros((t,), bool).at[-1].set(True),
        can_self_imitate=jnp.ones((t,), bool),
        is_first=jnp.zeros((t,), bool).at[0].set(True),
        is_last=jnp.zeros((t,), bool).at[-1].set(True),
    )


# ------------------------------------------------------------------ the episode format


def test_the_episode_format_is_the_fixture_format_plus_derived_rho():
    """The recorded columns *are* the on-disk arrays; ``pref_label`` is the one
    derived column, defaulted to zeros so a loader or a seeder need not know about
    self-revision at all."""
    assert EPISODE_FIELDS == ("obs", "action", "reward", "cont", "is_first", "is_last")
    ep = _episode(4)
    assert ep._fields == (*EPISODE_FIELDS, "extras")
    assert set(ep.extras) == set(RECORDED) - set(EPISODE_FIELDS) | {"pref_label"}
    np.testing.assert_allclose(np.asarray(ep.extras["pref_label"]), np.zeros(4))


def test_the_constructor_catches_the_scalar_reward_mistake():
    """``(T,)`` versus ``(T, 1)`` is the mistake an env's scalar reward makes, so it
    is checked once, in the constructor every producer goes through."""
    with pytest.raises(ValueError, match=r"expected \(T,\)"):
        raif_episode(
            obs=jnp.zeros((3, 2)),
            action=jnp.zeros((3, 1)),
            reward=jnp.zeros((3, 1)),
            is_success=jnp.zeros((3,), bool),
            has_succeeded=jnp.zeros((3,), bool),
            can_self_imitate=jnp.zeros((3,), bool),
            is_first=jnp.zeros((3,), bool),
            is_last=jnp.zeros((3,), bool),
        )


def test_cont_is_one_before_the_goal_and_zero_at_and_after_it():
    """``cont = 1 − is_success``: in their Mountain Car the goal predicate *is* the
    termination predicate and the state freezes at the flag, so the padded tail
    keeps the predicate true (addendum §1's ``easy_mode``)."""
    ep = _episode(4)
    np.testing.assert_allclose(np.asarray(ep.cont), [1.0, 1.0, 1.0, 0.0])


def test_prev_actions_shifts_once_into_the_recurrent_trainers_convention():
    """Replay records ``a_t`` (taken AT ``obs[t]``); the recurrent trainers want
    ``a_{t−1}`` (the action that *produced* ``obs[t]``). One shift, one place."""
    ep = _episode(3)._replace(action=jnp.asarray([[1.0], [2.0], [3.0]]))
    np.testing.assert_allclose(np.asarray(ep.prev_actions()), [[0.0], [1.0], [2.0]])


# ------------------------------------------------------------------------ the buffer


def test_a_short_episode_is_padded_by_freezing_its_final_step():
    """Padding is what continuing to step the env past the flag actually returns —
    the state frozen, the reward zero — so the tail is not fiction. The action and
    the self-imitation flag *are* zeroed: there is no action there to imitate."""
    fitted = fit_length(_episode(3, first_action=1.0), 5)
    assert fitted.length == 5
    np.testing.assert_allclose(np.asarray(fitted.obs[3]), np.asarray(fitted.obs[2]))
    np.testing.assert_allclose(np.asarray(fitted.reward[3:]), [0.0, 0.0])
    np.testing.assert_allclose(np.asarray(fitted.action[3:]), [[0.0], [0.0]])
    assert not bool(fitted.extras["can_self_imitate"][4])
    assert bool(fitted.is_last[-1]) and not bool(fitted.is_last[-2])


def test_a_long_episode_is_truncated_to_the_stored_window():
    """The episode-aligned contract's other edge, asserted so it is visible: a stored
    window starts at step 0, so steps past ``seq_len`` are dropped. A caller that
    cannot afford either edge wants ``StreamBuffer`` (below), which pads and
    truncates nothing."""
    fitted = fit_length(_episode(9), 4)
    assert fitted.length == 4
    np.testing.assert_allclose(np.asarray(fitted.obs), np.asarray(_episode(9).obs[:4]))
    assert bool(fitted.is_last[-1])  # the window's end is marked, not the episode's


def test_the_buffer_is_a_pytree_with_static_shapes_and_a_traced_size():
    buf = StaticBuffer.empty(4, 6, 2, 1, extras=EXTRAS)
    assert buf.data.obs.shape == (4, 6, 2) and buf.data.reward.shape == (4, 6)
    assert len(buf) == 0 and not buf.ready(1)
    buf = add(buf, _episode(6))
    assert len(buf) == 1 and buf.ready(1)
    leaves = jax.tree_util.tree_leaves(buf)
    assert all(hasattr(leaf, "shape") for leaf in leaves)


def test_the_ring_evicts_whole_sequences_fifo():
    """A slot holds exactly one whole sequence, so a clip never straddles two
    episodes — which is what lets ``fit_world`` skip the ``is_first`` reset."""
    buf = StaticBuffer.empty(2, 3, 2, 1, extras=EXTRAS)
    for i in range(3):
        buf = add(buf, _episode(3)._replace(reward=jnp.full((3,), float(i))))
    assert len(buf) == 2
    stored = {float(row[0]) for row in np.asarray(buf.data.reward)}
    assert stored == {1.0, 2.0}  # episode 0 evicted


def test_sampling_is_jit_able_with_a_static_batch_shape():
    buf = StaticBuffer.empty(4, 6, 2, 1, extras=EXTRAS)
    for _ in range(3):
        buf = add(buf, _episode(6))
    draw = jax.jit(lambda b, k: sample(b, k, 3))(buf, KEY)
    assert draw.obs.shape == (3, 6, 2) and draw.extras["pref_label"].shape == (3, 6)


def test_add_and_sample_are_pure():
    buf = StaticBuffer.empty(4, 6, 2, 1, extras=EXTRAS)
    after = add(buf, _episode(6))
    assert len(buf) == 0 and len(after) == 1


def test_alternate_is_strict_and_the_fallback_is_the_callers():
    """A10 disposition 9 exactly: the iterator alternates, and *nothing else* — the
    fallback while M⁺ is under one batch is one ``if`` in the recipe, because whose
    turn it is, is data policy."""
    a, b = object(), object()
    it = alternate(a, b)
    assert [next(it) for _ in range(5)] == [a, b, a, b, a]


def test_stack_episodes_fits_each_to_the_window():
    stacked = stack_episodes([_episode(3), _episode(7)], 5, pad=PAD)
    assert stacked.obs.shape == (2, 5, 2)


# ------------------------------------------------------------------- the flat stream


def _tagged(t: int, tag: float):
    """One episode whose every column is stamped with ``tag``, so a spliced stream
    can be read back column by column and attributed to the episode it came from."""
    return _episode(t)._replace(
        obs=jnp.full((t, 2), tag) + jnp.arange(t, dtype=jnp.float32)[:, None] / 100.0,
        action=jnp.full((t, 1), tag),
        reward=jnp.full((t,), tag),
        extras={
            **_episode(t).extras,
            "can_self_imitate": jnp.full((t,), tag),
            "pref_label": jnp.full((t,), tag),
        },
    )


def test_splicing_keeps_every_column_aligned_and_marks_the_boundaries():
    """The stream is the episodes' steps end to end: no padding, no truncation, no
    loss, no duplication, and the columns still line up row by row.

    Column alignment is the property a per-column ring can silently break (each
    column written at its own offset), so it is asserted across *all* of them rather
    than on ``obs`` alone — and ``pref_label``/``can_self_imitate``, which
    :func:`hybrid_aif.reference.raif.revise.label` computed per *finished* episode, must
    arrive per step exactly as they were written.
    """
    buf = StreamBuffer.empty(64, 4, 2, 1, extras=EXTRAS)
    lengths, tags = (3, 5, 2), (10.0, 20.0, 30.0)
    for t, tag in zip(lengths, tags, strict=True):
        buf = extend(buf, _tagged(t, tag))
    assert len(buf) == sum(lengths)  # steps, not sequences: nothing padded away

    live = jax.tree_util.tree_map(lambda x: np.asarray(x[: sum(lengths)]), buf.data)
    expected = np.concatenate([np.full(t, tag) for t, tag in zip(lengths, tags, strict=True)])
    np.testing.assert_array_equal(live.reward, expected)
    for column in ("can_self_imitate", "pref_label"):
        np.testing.assert_array_equal(live.extras[column], expected)
    np.testing.assert_array_equal(live.action[:, 0], expected)
    within = np.concatenate([np.arange(t) / 100.0 for t in lengths])
    np.testing.assert_allclose(live.obs[:, 0], expected + within, atol=1e-6)
    np.testing.assert_allclose(live.obs[:, 1], expected + within, atol=1e-6)
    #  the boundaries are exactly the episode starts and ends, and nowhere else
    assert np.flatnonzero(live.is_first).tolist() == [0, 3, 8]
    assert np.flatnonzero(live.is_last).tolist() == [2, 7, 9]


def test_a_sampled_window_spans_an_episode_boundary():
    """The whole point: a window is ``seq_len`` contiguous steps of the stream, so it
    may hold the tail of one episode and the head of the next, and ``is_first`` says
    where. Their replay samples stride-1 windows the same way, and forces step 0's
    flag true on every draw (``generic.py::_sample``)."""
    buf = StreamBuffer.empty(64, 4, 2, 1, extras=EXTRAS)
    buf = extend(buf, _tagged(3, 10.0))
    buf = extend(buf, _tagged(5, 20.0))
    assert buf.windows == 8 - 4 + 1  # their len(replay): one stride-1 item per step

    windows = jax.jit(lambda b, k: sample_windows(b, k, 32))(buf, KEY)
    assert windows.obs.shape == (32, 4, 2) and windows.extras["pref_label"].shape == (32, 4)
    tags = np.asarray(windows.reward)
    assert bool(np.all(windows.is_first[:, 0]))  # forced, as theirs is
    crossing = np.flatnonzero((tags == 10.0).any(axis=1) & (tags == 20.0).any(axis=1))
    assert crossing.size, "no window crossed the boundary — the stream is not flat"
    for row in crossing:
        #  inside a crossing window the join is marked, and every step still belongs
        #  to exactly one episode: the tags never interleave
        join = int(np.argmax(tags[row] == 20.0))
        assert bool(windows.is_first[row, join])
        assert tags[row].tolist() == [10.0] * join + [20.0] * (4 - join)


def test_the_stream_evicts_the_oldest_steps_and_wraps_without_seams():
    """Capacity is in **steps** and eviction is a single global FIFO over them —
    their ``selectors.Fifo`` over stride-1 items. A stream that has wrapped must
    still read out as one contiguous run of the most recent ``capacity`` steps, with
    no duplicated and no dropped step at the seam."""
    buf = StreamBuffer.empty(10, 3, 2, 1, extras=EXTRAS)
    for tag in (1.0, 2.0, 3.0, 4.0):  # 4 × 4 steps into a 10-step ring
        buf = extend(buf, _tagged(4, tag))
    assert len(buf) == 10 == buf.capacity and buf.windows == 8

    #  read the live stream back in order through a window at every valid offset
    draws = np.asarray(
        jax.vmap(lambda k: sample_windows(buf, k, 1).reward[0])(jax.random.split(KEY, 256))
    )
    seen = {tuple(row) for row in draws}
    stream = np.concatenate([np.full(4, t) for t in (2.0, 3.0, 4.0)])[-10:]
    valid = {tuple(stream[i : i + 3]) for i in range(len(stream) - 3 + 1)}
    assert seen <= valid, seen - valid
    assert seen == valid, valid - seen  # every offset is reachable, none is repeated
    #  the six evicted steps are gone: the oldest episode left no trace
    assert 1.0 not in {value for row in draws for value in row}


def test_no_training_window_holds_frozen_tail_padding():
    """FLAG 7's property, pinned: every step of every window is a real transition.

    The regression this exists to catch is the one the episode-aligned buffer had by
    construction — a short episode padded to ``seq_len`` by repeating its final
    observation with a zero action, so at converged Mountain Car lengths (≈38–45
    steps against ``batch_length`` 64) roughly a third of every training window was
    copies of one frozen state, *rising* as the agent improved. A frozen tail is
    recognisable without knowing where it came from: consecutive identical
    observations with no action.
    """
    seq_len, lengths = 64, (38, 45, 41, 39, 44, 40)
    episodes = [_tagged(t, float(i)) for i, t in enumerate(lengths)]

    padded = stack_episodes(episodes, seq_len, pad=PAD)  # what the episode-aligned buffer stores
    frozen = np.asarray(
        jnp.all(padded.obs[:, 1:] == padded.obs[:, :-1], axis=-1)
        & jnp.all(padded.action[:, 1:] == 0.0, axis=-1)
    )
    assert frozen.mean() > 0.25  # the bias this change removes, still there in `add`

    stream = StreamBuffer.empty(4096, seq_len, 2, 1, extras=EXTRAS)
    for ep in episodes * 4:
        stream = extend(stream, ep)
    windows = sample_windows(stream, jax.random.PRNGKey(11), 64)
    frozen = np.asarray(
        jnp.all(windows.obs[:, 1:] == windows.obs[:, :-1], axis=-1)
        & jnp.all(windows.action[:, 1:] == 0.0, axis=-1)
    )
    assert not frozen.any()
    #  and the arithmetic that follows: batch_size × seq_len REAL steps per update
    assert windows.obs.shape[0] * windows.obs.shape[1] == 64 * seq_len


def test_the_stream_buffer_is_a_pytree_with_static_shapes_and_traced_counters():
    """Same execution regime as :class:`StaticBuffer` — the counters are traced, the
    shapes are not, so ``extend``/``sample_windows`` ride a jitted scan carry."""
    buf = StreamBuffer.empty(32, 8, 2, 1, extras=EXTRAS)
    assert buf.data.obs.shape == (32, 2) and buf.data.reward.shape == (32,)
    assert len(buf) == 0 and not buf.ready(1)
    after = jax.jit(extend)(buf, _episode(8))
    assert len(buf) == 0 and len(after) == 8  # pure: nothing mutated
    assert after.ready(1) and not after.ready(2)  # one window at exactly seq_len steps
    assert all(hasattr(leaf, "shape") for leaf in jax.tree_util.tree_leaves(after))

    with pytest.raises(ValueError, match="at least seq_len"):
        StreamBuffer.empty(4, 8, 2, 1, extras=EXTRAS)
    with pytest.raises(ValueError, match="exceeds the stream's capacity"):
        extend(StreamBuffer.empty(8, 8, 2, 1, extras=EXTRAS), _episode(9))


# ---------------------------------------------------------------- the fixture format


def _provenance(**overrides):
    record = {
        "repo": "raif-paper",
        "commit_sha": "0" * 40,
        "command": "python raif/train_aif.py --configs gym_mtc tiny",
        "seed": 0,
        "collected_at": "2026-08-28T00:00:00Z",
        "n_episodes": 0,
        "n_steps": 0,
    }
    record.update(overrides)
    return record


def test_a_fixture_round_trips_through_the_declared_format(tmp_path):
    """The format is defined here because the oracle-side export will be written to
    match it: eight arrays concatenated over episodes, boundaries recoverable from
    ``is_first``/``is_last``, provenance beside the file."""
    episodes = [_episode(4), _episode(6), _episode(3)]
    path = save_fixture(tmp_path / "mtc_seed_0.npz", episodes, _provenance())
    with np.load(path) as data:
        assert set(data.files) == set(RECORDED)  # rho is derived, never stored
        assert data["obs"].shape == (13, 2) and data["action"].shape == (13, 1)
    loaded, record = load_fixture(path)
    assert [ep.length for ep in loaded] == [4, 6, 3]
    assert record["n_episodes"] == 3 and record["n_steps"] == 13
    np.testing.assert_allclose(np.asarray(loaded[1].obs), np.asarray(episodes[1].obs))


def test_a_missing_provenance_file_raises(tmp_path):
    """A12 item 6: "recorded with the fixture or it rots into magic data"."""
    path = save_fixture(tmp_path / "f.npz", [_episode(3)], _provenance())
    provenance_path(path).unlink()
    with pytest.raises(FileNotFoundError, match="no provenance beside"):
        load_fixture(path)


def test_a_provenance_missing_a_required_key_raises(tmp_path):
    path = save_fixture(tmp_path / "f.npz", [_episode(3)], _provenance())
    record = json.loads(provenance_path(path).read_text())
    del record["commit_sha"]
    provenance_path(path).write_text(json.dumps(record))
    with pytest.raises(ValueError, match="missing provenance key"):
        load_fixture(path)


def test_a_stale_provenance_beside_a_re_export_raises(tmp_path):
    path = save_fixture(tmp_path / "f.npz", [_episode(3)], _provenance())
    record = json.loads(provenance_path(path).read_text())
    record["n_episodes"] = 7
    provenance_path(path).write_text(json.dumps(record))
    with pytest.raises(ValueError, match="provenance says"):
        load_fixture(path)


def test_broken_episode_boundaries_are_caught_at_load(tmp_path):
    """The flags *are* the episode structure, so an export that breaks them fails
    at load rather than becoming a silently mis-split batch."""
    path = save_fixture(tmp_path / "f.npz", [_episode(4), _episode(4)], _provenance())
    with np.load(path) as data:
        columns = {k: np.asarray(data[k]) for k in RECORDED}
    columns["is_last"][3] = False  # one start too many
    np.savez(path, **columns)
    with pytest.raises(ValueError, match="is_first flag"):
        load_fixture(path)


def test_a_fixture_that_does_not_begin_at_an_episode_start_raises(tmp_path):
    path = save_fixture(tmp_path / "f.npz", [_episode(4)], _provenance())
    with np.load(path) as data:
        columns = {k: np.asarray(data[k]) for k in RECORDED}
    columns["is_first"][0] = False
    columns["is_first"][1] = True
    np.savez(path, **columns)
    with pytest.raises(ValueError, match="begin at an episode start"):
        load_fixture(path)


def test_a_wrong_obs_width_raises(tmp_path):
    """A pixel export is a different fixture; the width is checked, not assumed."""
    path = save_fixture(tmp_path / "f.npz", [_episode(4)], _provenance())
    with np.load(path) as data:
        columns = {k: np.asarray(data[k]) for k in RECORDED}
    columns["obs"] = np.zeros((4, 3), np.float32)
    np.savez(path, **columns)
    with pytest.raises(ValueError, match="obs is"):
        load_fixture(path)


# ---------------------------------------------------------------- the fallback seeder


def test_bang_bang_is_the_sign_of_the_velocity():
    """~15 lines and no tuning knob: full throttle in the direction of travel."""
    assert float(bang_bang(KEY, jnp.asarray([-0.5, 0.01]))[0]) == 1.0
    assert float(bang_bang(KEY, jnp.asarray([-0.5, -0.01]))[0]) == -1.0
    assert float(bang_bang(KEY, jnp.asarray([-0.5, 0.0]))[0]) == 0.0  # the reset tie, untouched


def test_the_fallback_seeder_solves_mountain_car_and_flags_every_step_expert():
    """A12 item 6's fallback, doing its job: energy pumping walks the car to the
    flag, and every step is marked ``can_self_imitate`` as
    ``run/collect_prior.py:105-109`` marks it — **except the terminal row**, whose
    action their driver zeroes (``driver.py:58-60``, which zeroes every action key,
    ``can_self_imitate`` included). Re-aimed 2026-09-07 with the row layout: an
    episode now ends at the flag rather than running the full ``time_limit``, so its
    length is variable and bounded by ``time_limit + 1``, exactly as the checked-in
    oracle fixtures are (39–41 rows against their ``TimeLimit(100)``)."""
    episodes = bang_bang_episodes(jax.random.PRNGKey(3), 3, SHORT_ENV)
    assert all(bool(ep.extras["has_succeeded"][-1]) for ep in episodes)
    for ep in episodes:
        csi = np.asarray(ep.extras["can_self_imitate"])
        assert (csi[:-1] == 1.0).all() and csi[-1] == 0.0
        #  their row layout: a reset row at reward 0, the goal row last and flagged
        assert float(ep.reward[0]) == 0.0
        assert (
            bool(ep.extras["is_success"][-1])
            and int(np.asarray(ep.extras["is_success"]).sum()) == 1
        )
        assert 1 < ep.length <= SHORT_ENV.time_limit + 1


def test_seed_stats_reports_the_three_numbers_a12_asks_for():
    """ "Seed stats are reported next to the acceptance numbers either way — under
    the fixture that is a parity check, not a confession" (A12 item 6)."""
    episodes = bang_bang_episodes(jax.random.PRNGKey(3), 3, SHORT_ENV)
    assert seed_stats(episodes).success_rate is None  # success is the task's column
    stats = seed_stats(episodes, success="is_success")
    assert stats.n_episodes == 3
    assert stats.success_rate == 1.0
    assert 0 < stats.mean_length <= SHORT_ENV.time_limit
    assert stats.mean_return < 0.0  # sparse −1 per raw step, ×2 for the action repeat


def test_collect_episode_records_the_flags_the_wrapper_records():
    """``is_success`` per step, ``has_succeeded`` its running OR, and
    ``can_self_imitate`` the caller's claim about the policy (``wrappers.py:685-722``)."""
    ep = collect_episode(SHORT_ENV, bang_bang, jax.random.PRNGKey(4))
    success = np.asarray(ep.extras["is_success"])
    np.testing.assert_array_equal(
        np.asarray(ep.extras["has_succeeded"]), np.maximum.accumulate(success)
    )
    assert not bool(jnp.any(ep.extras["can_self_imitate"]))  # not claimed expert by default
    assert bool(ep.is_first[0]) and bool(ep.is_last[-1])
    assert int(np.sum(ep.is_first)) == 1 and int(np.sum(ep.is_last)) == 1


# ----------------------------------------------------- the two regimes, bridged


def test_collect_into_a_replay_buffer_and_train_from_a_static_one():
    """The whole reason both buffers live in one module: the bridge is two lines.

    Collect with the host-side, dynamic-schema ``ReplayBuffer`` — ragged episodes,
    whatever columns the loop happens to record — then convert each stored episode
    and add it to a ``StaticBuffer``, which is what a jitted training step samples
    from. The ragged episode goes through :func:`fit_length` on the way in, which is
    where the two regimes' shape contracts meet.
    """
    host = ReplayBuffer(max_steps=64)
    for t in (3, 7):  # ragged: one shorter than the window, one longer
        host.write_episode(episode_columns(_episode(t)))
    assert host.num_episodes == 2
    assert set(host.column_names) == set(EPISODE_FIELDS) | set(
        EXTRAS
    )  # the dynamic schema took ours

    device = StaticBuffer.empty(4, 5, 2, 1, extras=EXTRAS)
    for stored in host.iter_episodes():
        device = add(device, as_episode(stored))

    assert len(device) == 2
    batch = jax.jit(lambda b, k: sample(b, k, 2))(device, KEY)
    assert batch.obs.shape == (2, 5, 2)  # fitted to seq_len, sampled inside jit
    #  the short episode was padded by freezing and the long one truncated
    stored = list(host.iter_episodes())
    short, long_ = as_episode(stored[0]), as_episode(stored[1])
    np.testing.assert_allclose(np.asarray(fit_length(short, 5).obs[3]), np.asarray(short.obs[2]))
    np.testing.assert_allclose(np.asarray(fit_length(long_, 5).obs), np.asarray(long_.obs[:5]))


def test_the_bridge_round_trips_an_episode_exactly():
    """``episode_columns`` then ``as_episode`` is the identity on every column,
    ``pref_label`` included — the host-side buffer has a dynamic schema, so unlike
    the fixture format it can carry derived columns."""
    ep = _episode(4)
    ep = ep._replace(extras={**ep.extras, "pref_label": jnp.asarray([0.0, 0.25, 0.5, 1.0])})
    back = as_episode(episode_columns(ep))
    cols, back_cols = episode_columns(ep), episode_columns(back)
    assert set(cols) == set(back_cols)
    for column, value in cols.items():
        np.testing.assert_allclose(back_cols[column], value, atol=0)


def test_as_episode_fills_in_the_columns_a_plain_collection_loop_lacks():
    """A loop that records only ``obs``/``action`` still converts: the reward
    defaults to zeros, ``cont`` to ones, the boundaries to the episode's own; any
    further column a ``ReplayBuffer`` carries becomes a task column."""
    got = as_episode(
        {
            "obs": np.zeros((3, 2), np.float32),
            "action": np.ones((3, 1), np.float32),
            "tag": np.arange(3, dtype=np.int32),  # a task column the engine does not know
        }
    )
    assert got.length == 3
    np.testing.assert_array_equal(np.asarray(got.cont), [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(np.asarray(got.reward), [0.0, 0.0, 0.0])
    assert set(got.extras) == {"tag"} and got.extras["tag"].dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(got.is_first), [True, False, False])
    np.testing.assert_array_equal(np.asarray(got.is_last), [False, False, True])
    with pytest.raises(ValueError, match="no 'obs'/'action' column"):
        as_episode({"reward": np.zeros((3,))})
