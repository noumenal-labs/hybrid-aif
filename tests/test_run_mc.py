"""P1′ chunk 2d's harness: the pieces that must be right *before* a 40-hour run.

Design: ``docs/design/amortization.md`` A12 ("**2d** = the 4-seed 1M-step run
against the oracle — compute, not code") and
``docs/design/amortization-p0-addendum.md`` §§5–7. The run itself is not a test;
what is testable is everything that would waste the run's wall clock if it were
wrong:

    the boundary        an episode that a chunk cut in half must reassemble
    the routing         success → both buffers, failure → M only, ONE code path
    the replay          flat step streams, so a training window crosses episode
                        boundaries and carries no frozen-tail padding (FLAG 7)
    the batch source    M only by default (FLAG 18, adjudicated 2026-09-02: the
                        oracle's own alternation is inert); ``--alternate`` keeps
                        the paper's strict i % 2 with the fallback, per seed
    the lanes           four env streams per seed, independent and contiguous
    resume              a reboot must cost at most one checkpoint interval
    the metrics         §6/§7's table must come out of ``scores.jsonl`` alone

All of it at ``--small`` sizes (``run_mc.SMALL``): the population claim and the
harness's arithmetic do not depend on the widths, and the parity-size versions of
these loops are the ``slow`` twins at the bottom.
"""

import json
import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.reference.envs.mountain_car import MountainCar
from hybrid_aif.reference.raif import metrics as ev
from hybrid_aif.reference.raif import run as run_mc
from hybrid_aif.reference.raif.agent import to_world_batch
from hybrid_aif.reference.raif.columns import EXTRAS
from hybrid_aif.reference.raif.revise import label
from hybrid_aif.reference.raif.seed import bang_bang, collect_episode
from hybrid_aif.train.buffers import StreamBuffer, extend, sample_windows

SEEDS = (0, 1, 2, 3)


def _config(tmp_path, **overrides) -> run_mc.Config:
    """A small, fast, fixture-seeded run configuration."""
    base = dict(
        out=str(tmp_path / "run"),
        seeds=SEEDS,
        steps=256,
        chunk_steps=128,
        batch_size=8,
        capacity=4096,  # replay STEPS per buffer (their 5e5, scaled down)
        small=True,
        checkpoint_every=1e9,  # only the shutdown checkpoint, unless a test asks
        time_limit=30,  # their TimeLimit is 100; shorter means more episodes per chunk
    )
    base.update(overrides)
    return run_mc.Config(**base)


# ================================================================== the boundary


def test_split_stream_reassembles_an_episode_a_chunk_boundary_cut():
    """Two chunks, one episode straddling them: the pending carry is what makes it whole.

    The single failure this guards is the one that would be invisible in a loss
    curve: a truncated episode is a *valid* episode, it just has the wrong
    ``has_succeeded[-1]``, so it would route to the wrong buffer and get ρ from the
    failure branch. Their ``on_episode`` callback cannot have this bug because their
    driver never cuts an episode; ours does, at every chunk.
    """
    rng = np.random.default_rng(0)
    n = 25
    is_last = np.zeros(n, bool)
    is_last[[7, 19]] = True  # episodes of length 8, 12, then a 5-step tail
    is_first = np.zeros(n, bool)
    is_first[[0, 8, 20]] = True
    cols = {
        "obs": rng.normal(size=(n, 2)).astype(np.float32),
        "action": rng.normal(size=(n, 1)).astype(np.float32),
        "reward": -2.0 * np.ones(n, np.float32),
        "is_success": is_last.copy(),
        "has_succeeded": is_last.copy(),
        "is_first": is_first,
        "is_last": is_last,
    }
    #  cut at 10, mid-second-episode
    first, pending = run_mc.split_stream(None, {k: v[:10] for k, v in cols.items()})
    second, tail = run_mc.split_stream(pending, {k: v[10:] for k, v in cols.items()})
    assert [len(e["reward"]) for e in first] == [8]
    assert [len(e["reward"]) for e in second] == [12]
    assert len(tail["reward"]) == 5  # still in flight, and it is what a checkpoint keeps
    #  the reassembled episode is bit-identical to the uncut one
    whole, _ = run_mc.split_stream(None, cols)
    for name, value in whole[1].items():
        assert np.array_equal(value, second[0][name]), name


def test_the_collected_episodes_flags_and_labels_are_the_recorded_ones(tmp_path):
    """A real chunk's stream splits into episodes whose flags obey the env's own rule.

    Written against the *device* path rather than synthetic flags, because the
    in-scan auto-reset is where ``is_first``/``is_last``/``has_succeeded`` are
    actually produced. What is asserted is the invariant the fixture loader also
    checks — one ``is_first`` and one ``is_last`` per episode, first step first —
    plus ``as_collected``'s two labels, plus **their row layout** (adopted
    2026-09-07): a reset row at reward 0 carrying a real action, a terminal row
    whose action is zeroed, ``T + 1`` rows for ``T`` applied actions, and
    ``is_success`` a predicate on the row's own observation. See
    :func:`hybrid_aif.reference.raif.run.make_collect_fn`.
    """
    cfg = _config(tmp_path, steps=256, chunk_steps=256)
    system, crspp, env, state, buffers, _, _, horizon = run_mc.build(cfg)
    chunk_fn = run_mc.make_chunk_fn(
        system,
        crspp,
        env,
        chunk_steps=cfg.chunk_steps,
        train_every=cfg.train_every,
        lanes=cfg.lanes,
        batch_size=cfg.batch_size,
        horizon=horizon,
    )
    ups = run_mc.alternation(0, cfg.chunk_steps // cfg.train_every, np.ones(len(SEEDS), bool))
    _, stream, losses = chunk_fn(
        state,
        run_mc._stack([b[0] for b in buffers]),
        run_mc._stack([b[1] for b in buffers]),
        ups,
    )
    assert bool(jnp.all(jnp.isfinite(losses.world.total)))
    columns = {name: np.asarray(getattr(stream, name)) for name in run_mc.Stream._fields}
    #  (seeds, lanes, chunk_steps // lanes, ...): the POOLED chunk split across lanes
    assert columns["reward"].shape == (len(SEEDS), cfg.lanes, cfg.chunk_steps // cfg.lanes)
    for lane in range(cfg.lanes):
        episodes, tail = run_mc.split_stream(None, {k: v[0, lane] for k, v in columns.items()})
        assert episodes, f"lane {lane}: 64 steps at TimeLimit(30) must finish 2 episodes"
        for cols in episodes:
            assert cols["is_first"][0] and not cols["is_first"][1:].any()
            assert cols["is_last"][-1] and not cols["is_last"][:-1].any()
            #  has_succeeded is the running OR of is_success, by construction
            assert np.array_equal(cols["has_succeeded"], np.maximum.accumulate(cols["is_success"]))
            #  their driver's row layout: T applied actions -> T + 1 rows, so a
            #  timeout episode is `time_limit + 1` long and never longer
            assert len(cols["reward"]) <= cfg.time_limit + 1
            #  the RESET row: a complete observation at reward 0 with a real action
            assert cols["reward"][0] == 0.0
            assert not cols["is_success"][0]  # U(-0.6, -0.4) is never at the flag
            #  the TERMINAL row: their `acts *= 1 - is_last`
            np.testing.assert_array_equal(cols["action"][-1], np.zeros_like(cols["action"][-1]))
            #  ...and no other row is zeroed by the driver, so the episode ends exactly
            #  once. `is_success` is the predicate on THIS row's obs, so a successful
            #  episode's only flagged row is its last
            assert cols["is_success"].sum() in (0, 1)
            if cols["is_success"].any():
                assert cols["is_success"][-1] and cols["obs"][-1, 0] >= 0.45
                assert len(cols["reward"]) <= cfg.time_limit  # it stopped early
            else:
                assert len(cols["reward"]) == cfg.time_limit + 1
            ep = run_mc.as_collected(cols)
            assert ep.extras["pref_label"].shape == (len(cols["reward"]),)
            assert float(ep.extras["pref_label"][0]) == 0.0  # their `pref_label[0] = 0`
            assert bool(jnp.all(jnp.isfinite(ep.extras["pref_label"])))
        assert len(tail["reward"]) <= cfg.time_limit


def test_the_lanes_are_four_independent_streams_driven_by_one_agent(tmp_path):
    """Their ``envs: {amount: 4}``: four env streams per seed, one ``params`` above them.

    The two things a lane axis can silently get wrong, both asserted here:

    1. **The lanes must not be copies.** Four identical streams would cost 4× the
       compute for one stream's information — the failure mode of broadcasting the
       env reset or the key instead of drawing one per lane.
    2. **A lane's steps must stay contiguous.** ``split_stream`` finds episodes as
       runs of consecutive steps, so a stream layout that interleaved the lanes would
       splice unrelated steps into one "episode" and the flags would still look
       plausible. The reassembly invariant above is what would catch it; this asserts
       the layout directly.
    """
    cfg = _config(tmp_path, steps=64, chunk_steps=64)
    system, crspp, env, state, buffers, _, _, horizon = run_mc.build(cfg)
    #  (1) the initial states differ lane to lane, and so do the keys
    assert state.env_state[0].shape == (len(SEEDS), cfg.lanes, 2)
    assert len(np.unique(np.asarray(state.env_state[0])[0, :, 0])) == cfg.lanes
    assert len(np.unique(np.asarray(state.key)[0], axis=0)) == cfg.lanes
    #  the trained things carry ONLY the seed axis: one agent per seed, above the lanes
    assert np.asarray(state.train_key).shape == (len(SEEDS), 2)

    chunk_fn = run_mc.make_chunk_fn(
        system,
        crspp,
        env,
        chunk_steps=64,
        train_every=16,
        lanes=cfg.lanes,
        batch_size=8,
        horizon=horizon,
    )
    ups = run_mc.alternation(0, 4, np.ones(len(SEEDS), bool))
    _, stream, _ = chunk_fn(
        state, run_mc._stack([b[0] for b in buffers]), run_mc._stack([b[1] for b in buffers]), ups
    )
    obs = np.asarray(stream.obs)
    assert obs.shape == (len(SEEDS), cfg.lanes, 64 // cfg.lanes, 2)
    #  (1) again, on the collected stream: no two lanes traced the same trajectory
    for a in range(cfg.lanes):
        for b in range(a + 1, cfg.lanes):
            assert not np.array_equal(obs[0, a], obs[0, b]), (a, b)
    #  (2) contiguity: within a lane, position moves by one Mountain Car step at a
    #  time, so consecutive recorded observations are close; an interleaved layout
    #  would put four different lanes' positions side by side
    for lane in range(cfg.lanes):
        jumps = np.abs(np.diff(obs[0, lane, :, 0]))
        assert jumps.max() < 0.1, (lane, jumps.max())


def test_lanes_must_divide_train_every(tmp_path):
    """Their ``Ratio(64/1024)`` counts pooled steps, so the division must be exact."""
    system, crspp, env, *_ = run_mc.build(_config(tmp_path, steps=0))
    with pytest.raises(ValueError, match="positive divisor of train_every"):
        run_mc.make_chunk_fn(system, crspp, env, chunk_steps=64, train_every=16, lanes=3)


# =================================================================== the routing


def test_routing_sends_successes_to_both_buffers_and_failures_to_m_only():
    """§4's rule, on both branches, through the one function the seeding path uses too."""
    env = MountainCar(time_limit=90)  # bang-bang needs ≈48 steps to the flag
    won = label(collect_episode(env, bang_bang, jax.random.PRNGKey(0), can_self_imitate=True))
    lost = label(collect_episode(env, lambda k, o: jnp.zeros((1,)), jax.random.PRNGKey(1)))
    assert bool(won.extras["has_succeeded"][-1]) and not bool(lost.extras["has_succeeded"][-1])
    positive = general = StreamBuffer.empty(512, 16, 2, 1, extras=EXTRAS)
    positive, general = run_mc.route(positive, general, won)
    #  the buffers count STEPS now, so the routing shows up as the stream's length
    assert (len(positive), len(general)) == (won.length, won.length)
    positive, general = run_mc.route(positive, general, lost)
    assert (len(positive), len(general)) == (won.length, won.length + lost.length)


def test_the_oracle_fixture_is_seeded_through_that_same_rule(tmp_path):
    """Seeding is the routing rule applied to the oracle's episodes — not a special case.

    A12 item 6's parity check, as a test: the fixtures' ``seed_stats`` must reproduce
    §7's exported numbers (48 episodes, mean return ≈−76, SR 1.00), and because every
    one of them succeeded, the rule puts all 48 in **both** buffers. That the counts
    are equal is a *consequence* of the rule here; a seeding path with its own
    branch could not tell you that.
    """
    cfg = _config(tmp_path)
    _, _, _, _, buffers, _, reports, _ = run_mc.build(cfg)
    assert len(reports) == len(SEEDS)
    for _seed, stats, provenance in reports:
        assert stats.n_episodes == 48
        assert stats.success_rate == 1.0
        assert -77.0 < stats.mean_return < -75.0, stats
        assert "raif-paper" in provenance["repo"]
        assert len(provenance["commit_sha"]) == 40
    for (_seed, stats, _provenance), (positive, general) in zip(reports, buffers, strict=True):
        #  in STEPS, because the buffers are flat streams: every one of the 48 episodes
        #  went to both, and the stream is exactly their steps spliced end to end
        steps = int(round(stats.n_episodes * stats.mean_length))
        assert len(positive) == len(general) == steps


# ================================================================== the flat stream


def test_the_training_windows_are_flat_stream_windows_with_no_padding(tmp_path):
    """FLAG 7, closed, end to end through the harness's own buffers.

    Three properties in one draw, because they are one change: the buffers count
    **steps** (their ``replay_size`` verbatim, not ``5e5 / seq_len`` sequences), a
    sampled window crosses episode boundaries and says where with ``is_first``, and
    no step of any window is frozen-tail padding — so an update sees
    ``batch_size · seq_len`` real transitions, their ``batch_steps``.
    """
    assert run_mc.CAPACITY == 500_000  # STEPS per buffer
    cfg = _config(tmp_path, seq_len=32)
    _, _, _, _, buffers, _, _, _ = run_mc.build(cfg)
    general = buffers[0][1]
    assert general.capacity == cfg.capacity and len(general) > cfg.seq_len

    batch = sample_windows(general, jax.random.PRNGKey(5), 64)
    assert batch.obs.shape == (64, cfg.seq_len, 2)
    assert bool(np.all(np.asarray(batch.is_first)[:, 0]))
    interior = np.asarray(batch.is_first)[:, 1:].sum(axis=1)
    assert interior.max() >= 1, "no sampled window crossed an episode boundary"

    obs, action = np.asarray(batch.obs), np.asarray(batch.action)
    frozen = (obs[:, 1:] == obs[:, :-1]).all(axis=-1) & (action[:, 1:] == 0.0).all(axis=-1)
    assert not frozen.any()

    #  and the flags reach the recurrent scan: to_world_batch carries them through
    world = to_world_batch(batch)
    np.testing.assert_array_equal(np.asarray(world.is_first), np.asarray(batch.is_first))


# ============================================================== the alternation


def test_the_parity_default_never_samples_m_plus(tmp_path):
    """FLAG 18 (adjudicated 2026-09-02): the default schedule is all-``False``.

    The oracle's ``i % 2`` alternation is inert (``Ratio(64/1024)`` returns only 0
    or 1 per driver transition, so ``next(positive_dataset)`` is unreachable), and
    parity matches the code — so ``Config.alternate`` defaults ``False`` and the
    ``use_positive`` schedule is :func:`run_mc.m_only`. The paper-described
    schedule survives behind ``--alternate``, as a labeled A/B.
    """
    assert run_mc.Config(out=str(tmp_path / "x")).alternate is False
    ups = run_mc.m_only(4, 8)
    assert ups.shape == (4, 8) and not bool(jnp.any(ups))
    cfg, _ = run_mc.parse(["--out", str(tmp_path / "y")])
    assert cfg.alternate is False
    cfg, _ = run_mc.parse(["--out", str(tmp_path / "y"), "--alternate"])
    assert cfg.alternate is True


def test_alternation_is_strict_per_seed_and_falls_back_while_m_plus_is_short():
    """The ``--alternate`` A/B's schedule (the PAPER's ``i % 2``, not the parity
    default — see above): continued across chunks, with the fallback per seed."""
    ready = np.asarray([True, False, True, True])
    first = np.asarray(run_mc.alternation(0, 4, ready))
    assert first.shape == (4, 4)
    assert first[0].tolist() == [True, False, True, False]  # ready: strict alternation
    assert first[1].tolist() == [False] * 4  # short M+: always M (the fallback)
    #  the next chunk continues the parity rather than restarting it
    second = np.asarray(run_mc.alternation(4, 4, ready))
    assert second[0].tolist() == [True, False, True, False]
    third = np.asarray(run_mc.alternation(5, 4, ready))
    assert third[0].tolist() == [False, True, False, True]


# =================================================================== resume


def test_a_run_resumes_and_continues_rather_than_restarting(tmp_path):
    """The checkpoint round trip: stop, resume, and the run goes *on*.

    What is asserted is what the module docstring claims and no more — the counters
    continue, the episode stream continues, the replay contents survive, and the
    losses stay finite past the boundary. **Bitwise** continuation across a process
    boundary is deliberately not asserted: the PRNG keys are saved, so it should
    hold at a fixed chunk size and JAX version, but asserting it would turn an XLA
    version bump into a data-corruption failure. Here both halves run in one
    process, so the stronger check *is* available and is made where it is cheap:
    the restored state's leaves are bit-identical to the ones written.
    """
    cfg = _config(tmp_path, steps=256, chunk_steps=128)
    first = run_mc.train(cfg, log=lambda *a: None)
    assert first.step == 256 and first.chunk == 2
    assert first.grad_index == 256 // cfg.train_every
    assert sum(first.episodes) > 0

    #  the restored leaves are the written ones, exactly
    system, crspp, env, fresh, fresh_buffers, _, _, _ = run_mc.build(cfg)
    state, buffers, pending, counters = run_mc.load_checkpoint(cfg, fresh, fresh_buffers)
    assert counters == first
    assert len(buffers) == len(SEEDS)
    assert all(len(b[1]) >= 48 for b in buffers)  # the fixture's 48 plus what was collected

    #  and the run continues from there
    scores_before = run_mc.read_scores(cfg.path / "scores.jsonl")
    extended = run_mc.train(
        run_mc.Config(**{**vars(cfg), "steps": 512}), resume=True, log=lambda *a: None
    )
    assert extended.step == 512 and extended.chunk == 4
    assert extended.grad_index == 512 // cfg.train_every
    assert sum(extended.episodes) > sum(first.episodes)
    scores_after = run_mc.read_scores(cfg.path / "scores.jsonl")
    assert len(scores_after) > len(scores_before)
    #  the appended episodes carry the CONTINUED step counter, not a restarted one
    assert min(r["step"] for r in scores_after[len(scores_before) :]) > 256
    #  every recorded return is inside the env's own bounds, before and after
    for row in scores_after:
        assert -2 * cfg.time_limit <= row["return"] <= 0.0
        #  T applied actions -> T + 1 recorded rows (their driver's reset row and its
        #  terminal row), so a timeout episode is `time_limit + 1` long
        assert 1 <= row["length"] <= cfg.time_limit + 1
        #  R-S's C is its own logged channel, and NOT `return + offset·length`: theirs
        #  counts one offset per row EXCEPT the closing one and DROPS that row's
        #  reward, which on this env is in [-2, 0] — so C sits at or above the naive
        #  reconstruction, by exactly the reward theirs excluded
        naive = row["return"] + ev.MC_OFFSET * (row["length"] - 1)
        assert naive - 1e-5 <= row["stability"] <= naive + 2.0 + 1e-5
        if not row["success"]:  # a timeout's closing row is worth exactly -2
            assert row["stability"] == pytest.approx(naive + 2.0, abs=1e-5)

    metrics = [
        json.loads(line)
        for line in (cfg.path / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert metrics
    for row in metrics:
        for name in ("world", "actor", "critic", "ensemble", "crspp", "distill_gap"):
            assert np.isfinite(row[name]), (name, row)


def test_a_resume_refuses_to_change_a_shape(tmp_path):
    """``--resume`` takes the run's own configuration; only the schedule knobs move.

    A resume that silently accepted a different ``--batch-size`` would deserialise
    the checkpoint onto the wrong shapes — loudly if lucky, into a differently
    trained agent if not. Tested through ``parse``, which is where the rule lives.
    """
    #  `steps=0` writes run.json, seeds the buffers and checkpoints without ever
    #  entering the loop — so the rule is tested without paying for a chunk compile
    cfg = _config(tmp_path, steps=0)
    run_mc.train(cfg, log=lambda *a: None)
    resumed, _ = run_mc.parse(
        ["--out", cfg.out, "--resume", "--steps", "4096", "--batch-size", "64", "--capacity", "9"]
    )
    assert resumed.steps == 4096  # the schedule knob wins
    assert resumed.batch_size == cfg.batch_size  # the shape knobs do not
    assert resumed.capacity == cfg.capacity
    assert resumed.seeds == cfg.seeds and resumed.small is True


def test_a_resume_writes_where_it_was_told_not_where_the_run_once_lived(tmp_path):
    """``--out`` is this invocation's directory. A renamed run keeps the ``out`` it
    was launched with in ``run.json``, and a resume that honoured that would load
    and checkpoint into the OTHER path — in practice, into whatever live run now
    occupies it."""
    cfg = _config(tmp_path, steps=0)
    run_mc.train(cfg, log=lambda *a: None)
    moved = tmp_path / "archived_run"
    (tmp_path / cfg.out).rename(moved) if not pathlib.Path(cfg.out).is_absolute() else pathlib.Path(
        cfg.out
    ).rename(moved)
    resumed, _ = run_mc.parse(["--out", str(moved), "--resume"])
    assert resumed.out == str(moved)
    assert resumed.batch_size == cfg.batch_size  # everything else still comes from the run


def test_a_resume_refuses_a_knob_it_cannot_account_for(tmp_path):
    """A knob missing from a stored ``run.json`` is refused, not defaulted.

    The current default describes what a run launched today would build; the
    checkpoint was built by whatever the code did then. `winit` cost a failed resume
    exactly this way. Absence is tolerated only where `_ABSENT_MEANS` records what
    the code did before the knob existed.
    """
    cfg = _config(tmp_path, steps=0)
    run_mc.train(cfg, log=lambda *a: None)
    stored = pathlib.Path(cfg.out) / "run.json"
    config = json.loads(stored.read_text())

    #  a knob with a recorded historical meaning resumes, and takes that meaning
    for knob, before in run_mc._ABSENT_MEANS.items():
        thinned = {k: v for k, v in config.items() if k != knob}
        stored.write_text(json.dumps(thinned))
        resumed, _ = run_mc.parse(["--out", cfg.out, "--resume"])
        assert getattr(resumed, knob) == before, knob

    #  one without it refuses, naming the knob
    thinned = {k: v for k, v in config.items() if k != "batch_size"}
    stored.write_text(json.dumps(thinned))
    with pytest.raises(ValueError, match=r"records no \['batch_size'\]"):
        run_mc.parse(["--out", cfg.out, "--resume"])


# ======================================================================= the report


def test_the_report_computes_the_addendum_7_table_from_scores_alone(tmp_path):
    """§6/§7's numbers out of ``scores.jsonl``, with a synthetic curve of known shape.

    ``scores.jsonl`` stores returns and lengths rather than per-step rewards, which
    is sufficient because R-S's offset applies *per step*: ``C = return + 1.1·length``.
    The synthetic run below is the oracle's shape in miniature — the floor, then a
    dip, then convergence — so the assertions are about which window each statistic
    reads, which is §7's whole ruling.
    """
    directory = tmp_path / "synthetic"
    directory.mkdir()
    rows = []
    step = 0
    for phase, value, count in (("floor", -200.0, 200), ("dip", -120.0, 100), ("conv", -68.0, 700)):
        for _ in range(count):
            step += 1000
            rows.append(
                {
                    "step": step,
                    "seed": 0,
                    "return": value,
                    "length": int(-value / 2),
                    "success": phase != "floor",
                }
            )
    (directory / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = run_mc.report(run_mc.Config(out=str(directory)), log=lambda *a: None)
    assert len(out) == 1
    got = out[0]
    assert got.episodes == 1000 and got.steps == 1_000_000
    assert got.final_acr == pytest.approx(-68.0)  # the last 100 episodes
    assert got.best_window == pytest.approx(-68.0)  # some 200k window is all-converged
    assert got.last_500k == pytest.approx(-68.0)
    assert got.sr_window == 1.0
    assert got.sr_oracle == 1.0  # every converged episode beats -100
    assert 0.0 <= got.rs < 1.0
    #  the floor is what the FIRST window reads, which is why an endpoint is not the target
    early = run_mc.seed_report(rows[:200], 0)
    assert early.final_acr == pytest.approx(-200.0)
    assert early.sr_oracle == 0.0


def test_a_partial_trailing_window_does_not_count_as_the_best_200k():
    """Only a complete window competes: a few episodes past the last full window do not.

    Four complete windows, the last (600–800k) at −70 and the rest at −80, then three
    episodes at −60 just past 800k. Tiled to the last step, the partial 800–1000k window
    would read −60 and win; complete windows only, the best is 600–800k at −70. A log that
    ends a few dozen steps short of a window's end still counts that window.
    """
    rows = [
        {
            "step": s,
            "seed": 0,
            "return": -70.0 if s >= 600_000 else -80.0,
            "length": 40,
            "success": True,
        }
        for s in range(1000, 800_001, 1000)
    ]
    rows += [
        {"step": 800_000 + 300 * i, "seed": 0, "return": -60.0, "length": 30, "success": True}
        for i in (1, 2, 3)
    ]
    got = run_mc.seed_report(rows, 0)
    assert got.best_window == pytest.approx(-70.0)
    assert got.best_window_at == "600–800k"
    short = [r for r in rows if r["step"] <= 799_000]  # the log ends 1k short of 800k
    assert run_mc.seed_report(short, 0).best_window_at == "600–800k"


# ==================================================================== the slow twins


@pytest.mark.slow
def test_the_throughput_benchmark_reports_a_vmap_gain(tmp_path):
    """Deliverable 1 as a test, at ``--small`` sizes: the ratio must be a *gain*.

    The parity-size numbers are the ones that matter and they are in the chunk's
    report; what a test can pin is the sign of the effect — vmapping the seeds must
    beat running them one after another, or the whole 4-seed-in-one-process design
    is unjustified. Deselected by default (``pytest -m slow``) because it compiles
    two chunk functions.
    """
    #  A wall-clock ratio is only meaningful on a machine that is not otherwise busy,
    #  and this project routinely runs multi-hour training alongside its own test
    #  suite. Under that load both arms are throttled unequally and the ratio says
    #  nothing about vmap. Skip rather than fail: a red gate that means "the laptop was
    #  busy" trains everyone to ignore it.
    per_core = os.getloadavg()[0] / (os.cpu_count() or 1)
    if per_core > 0.6:
        pytest.skip(f"machine is loaded (1-minute load {per_core:.2f} per core); timings are noise")
    out = run_mc.benchmark(_config(tmp_path, chunk_steps=64), reps=1, log=lambda *a: None)
    assert out["ratio"] > 1.0, out
    assert out["projected_hours"] > 0.0
    for n in (1, len(SEEDS)):
        assert 0.0 < out[n]["collect_fraction"] < 1.0


@pytest.mark.slow
def test_a_chunk_runs_at_parity_settings(tmp_path):
    """One chunk of the real thing: their ``tiny``, ``16 × 64``, ``imag_horizon 8``.

    Shape parity and finiteness over the whole cycle — the collect scan, the
    auto-reset, sixteen-step blocks, the four trainers, the probe — at the sizes the
    40-hour run uses. Not learning: that is the run.
    """
    cfg = run_mc.Config(
        out=str(tmp_path / "parity"), seeds=(0, 1), steps=64, chunk_steps=64, capacity=64
    )
    counters = run_mc.train(cfg, log=lambda *a: None)
    assert counters.step == 64 and counters.grad_index == 4
    metrics = [
        json.loads(line)
        for line in (cfg.path / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(metrics) == 2
    for row in metrics:
        for name, value in row.items():
            assert np.isfinite(value), (name, value)


def test_add_is_the_engines_extend(tmp_path):
    """The harness routes through :func:`hybrid_aif.train.buffers.extend`, nothing else.

    Stated as a test because the temptation 2d had to resist was writing a
    seed-indexed splice of its own (the buffers are stacked on the device and
    unstacked on the host for exactly this reason). A private copy would drift from
    the engine's ring/eviction semantics silently.
    """
    env = MountainCar(time_limit=90)
    ep = label(collect_episode(env, bang_bang, jax.random.PRNGKey(2), can_self_imitate=True))
    empty = StreamBuffer.empty(256, 64, 2, 1, extras=EXTRAS)
    mine, _ = run_mc.route(empty, empty, ep)
    theirs = extend(empty, ep)
    for a, b in zip(
        jax.tree_util.tree_leaves(mine), jax.tree_util.tree_leaves(theirs), strict=True
    ):
        assert np.array_equal(np.asarray(a), np.asarray(b))


# ============================================================= the act-time telemetry


def test_act_entropy_is_a_key_free_read_of_the_actor(tmp_path):
    """Their ``outs['action'].entropy()`` — and nothing that could move the stream.

    Two claims, and the second is the one the isolation control depends on
    (``series-sweep.md`` §5.3 asked for act-time telemetry precisely so the *next*
    run would answer a question it otherwise could not, at zero marginal cost — a
    measurement that perturbed the run would have cost the run instead).

    1. **The quantity is theirs.** Their actor is ``dist: normal``
       (``aif/configs.yaml:87``) over a 1-D box, so the entropy their
       ``agent.py:73`` logs is the differential entropy of a diagonal Gaussian,
       ``sum_d [log sigma_d + 0.5 log(2 pi e)]``. Asserted in closed form against
       the actor's own scale, so a change of head family or of reduction fails here
       rather than in a 5-hour run's logs.
    2. **It takes no key.** Asserted structurally (the signature has no key
       parameter) and behaviourally (repeated calls agree bitwise, and a collection
       chunk run twice from one carry is bit-identical with the column in it).
    """
    import inspect

    cfg = _config(tmp_path, steps=128, chunk_steps=128)
    system, _, env, state, _, _, _, _ = run_mc.build(cfg)
    params = jax.tree_util.tree_map(lambda x: x[0], state.params)
    beliefs = jax.tree_util.tree_map(lambda x: x[0, 0], state.beliefs)

    assert "key" not in inspect.signature(run_mc.act_entropy).parameters

    got = run_mc.act_entropy(system, params, beliefs)
    assert float(got) == float(run_mc.act_entropy(system, params, beliefs))

    #  the closed form, off the same conditional the draw is taken from
    from hybrid_aif.objectives.amortized import features as _features

    feats = _features(system.to_positional("beliefs", beliefs))
    chol = np.asarray(system._habit_module(params).condition_on((feats,)).chol)
    scale = np.diag(chol)
    expected = float(np.sum(np.log(scale) + 0.5 * np.log(2 * np.pi * np.e)))
    np.testing.assert_allclose(float(got), expected, rtol=1e-5)


def test_the_telemetry_columns_do_not_move_the_collection_stream(tmp_path):
    """A chunk is a pure function of ``(params, carry)`` with the new column in it.

    The strong form of this — the same four chunks with the telemetry *removed*,
    diffed bitwise over the stream, all four trainers' losses and every parameter
    leaf — was run against a pristine checkout of this commit on 2026-09-15: 652
    arrays, zero mismatches, the only new key ``ImagineMetrics.bc_term``. What is
    reproducible *in* the tree is that determinism, which is what would break if a
    measurement ever reached for a key.
    """
    cfg = _config(tmp_path, steps=128, chunk_steps=128)
    system, crspp, env, state, buffers, _, _, horizon = run_mc.build(cfg)
    chunk_fn = run_mc.make_chunk_fn(
        system,
        crspp,
        env,
        chunk_steps=cfg.chunk_steps,
        train_every=cfg.train_every,
        lanes=cfg.lanes,
        batch_size=cfg.batch_size,
        horizon=horizon,
    )
    ups = run_mc.m_only(len(SEEDS), cfg.chunk_steps // cfg.train_every)
    args = (state, run_mc._stack([b[0] for b in buffers]), run_mc._stack([b[1] for b in buffers]))
    first = chunk_fn(*args, ups)
    second = chunk_fn(*args, ups)
    for a, b in zip(
        jax.tree_util.tree_leaves(first), jax.tree_util.tree_leaves(second), strict=True
    ):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    #  and the column is real: a finite entropy on every recorded row
    assert np.all(np.isfinite(np.asarray(first[1].log_entropy)))
    assert np.asarray(first[1].log_entropy).shape == np.asarray(first[1].reward).shape


def test_coverage_is_their_ten_by_ten_grid_over_position_and_velocity():
    """:class:`~hybrid_aif.reference.raif.run.Coverage` against their arithmetic, by hand.

    Their ``MeasureCoverage`` for ``gym_mtc`` (``wrappers.py:765-904``) takes the
    observation space's declared box, cuts each axis into ``bin_size = 10``, keys a
    cell by ``10 * bin(position) + bin(velocity)`` and reports ``len(counter) / 100``.
    Three things are worth pinning: the **interior** edges (``linspace(...,
    bins + 1)[1:-1]`` — nine of them, so the outermost bins are half-open to the
    box's own clipping), the cell-id packing, and that the counter is **cumulative**
    rather than per episode, which is what makes their series a saturating 0.338 →
    0.777 rather than a per-episode ratio no 45-step episode could reach.
    """
    env = MountainCar()
    cov = run_mc.Coverage(env)
    np.testing.assert_allclose(cov.edges[:, 0], np.linspace(-1.2, 0.6, 11)[1:-1])
    np.testing.assert_allclose(cov.edges[:, 1], np.linspace(-0.07, 0.07, 11)[1:-1])

    #  one cell, however many rows land in it
    corner = np.array([[-1.2, -0.07]] * 5)
    assert cov.update(corner) == 0.01
    #  the far corner is cell 99: bin 9 on both axes
    assert cov.update(np.array([[0.6, 0.07]])) == 0.02
    assert cov.seen == {0, 99}
    #  the packing: position bin 3, velocity bin 0 is cell 30
    pos = np.linspace(-1.2, 0.6, 11)
    assert cov.update(np.array([[0.5 * (pos[3] + pos[4]), -0.07]])) == 0.03
    assert 30 in cov.seen
    #  cumulative: re-counting the first cell adds nothing
    assert cov.update(corner) == 0.03


def test_the_telemetry_columns_reach_scores_and_metrics(tmp_path):
    """``log_entropy`` and ``coverage`` per episode; ``bc_term`` per chunk.

    Where each lands is its semantics: coverage and act-time entropy are *episode*
    quantities on their side too (``episode/coverage_mean``,
    ``stats/mean_log_entropy``, both written by their ``per_episode`` callback), and
    ``bc_term`` is a term of the actor loss, so it rides with the rest of the actor
    channel in ``metrics.jsonl``. Also checks the coverage counter survives a
    checkpoint — it is cumulative over a lane's whole life, so a resume that
    restarted it would put a discontinuity in a monotone series.
    """
    cfg = _config(tmp_path, steps=256, chunk_steps=128, seeds=(0,), lanes=2)
    run_mc.train(cfg, log=lambda *a: None)
    rows = run_mc.read_scores(cfg.path / "scores.jsonl")
    assert rows
    for row in rows:
        assert np.isfinite(row["log_entropy"])
        assert 0.0 < row["coverage"] <= 1.0
    #  cumulative per lane, so it never falls
    for lane in (0, 1):
        series = [r["coverage"] for r in rows if r["lane"] == lane]
        assert series == sorted(series)
    metrics = [
        json.loads(line)
        for line in (cfg.path / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert metrics and all(np.isfinite(row["bc_term"]) for row in metrics)
    #  the two-path actor read: both means and scales finite, scales inside the
    #  head's [minstd, maxstd], the gap non-negative
    for row in metrics:
        for col in ("probe_mu_act", "probe_mu_train", "probe_sigma_act", "probe_sigma_train"):
            assert np.isfinite(row[col])
        assert 0.1 - 1e-6 <= row["probe_sigma_act"] <= 1.0 + 1e-6
        assert 0.1 - 1e-6 <= row["probe_sigma_train"] <= 1.0 + 1e-6
        assert row["probe_mu_gap"] >= 0.0
    saved = json.loads((cfg.path / "checkpoint" / "checkpoint.json").read_text())
    for lane in (0, 1):
        last = [r["coverage"] for r in rows if r["lane"] == lane][-1]
        assert len(saved["coverage"][0][lane]) == int(round(last * 100))
    restored = run_mc.load_coverage(cfg, MountainCar())
    assert restored[0][0].seen == set(saved["coverage"][0][0])


# ------------------------------------------------------------- the act-time sample (D6)


def _sample_codes_reference(system, params, beliefs, key):
    """The lane's first form of the draw (2026-09-28, before ``infer_states(key=)``): one
    categorical per code factor under a per-factor split; carries untouched. Kept as the
    reference the face is pinned to, so the runs launched on it stay comparable."""
    from hybrid_aif.inference import amortized as _amortized

    families = system._amortized_structure(params).families
    keys = jax.random.split(key, len(beliefs))
    out = []
    for belief, family, k in zip(beliefs, families, keys, strict=True):
        if family == _amortized.CODE:
            draw = jax.random.categorical(k, jnp.log(jnp.maximum(belief, 1e-30)), axis=-1)
            out.append(jax.nn.one_hot(draw, belief.shape[-1], dtype=belief.dtype))
        else:
            out.append(belief)
    return tuple(out)


def test_the_keyed_recognition_face_is_the_lanes_draw_bitwise(tmp_path):
    """``infer_states(key=)`` on the reference agent: one-hot codes, carries untouched,
    the same draw the lane's first helper made under the same key (bitwise), and the
    keyless face unchanged."""
    from hybrid_aif.inference import amortized as _amortized

    cfg = _config(tmp_path, steps=128, chunk_steps=128)
    system, _, env, state, _, _, _, _ = run_mc.build(cfg)
    params = jax.tree_util.tree_map(lambda x: x[0], state.params)
    priors = jax.tree_util.tree_map(lambda x: x[0, 0], state.beliefs)
    obs = env.observe(jax.tree_util.tree_map(lambda x: x[0, 0], state.env_state))[0]
    families = system._amortized_structure(params).families
    observed = system.from_positional("observations", (obs, None, None))
    expected, _ = system.infer_states(params, observed, priors)
    again, _ = system.infer_states(params, observed, priors, key=None)
    keyed, _ = system.infer_states(params, observed, priors, key=jax.random.PRNGKey(3))
    expected, again, keyed = (system.to_positional("beliefs", b) for b in (expected, again, keyed))
    reference = _sample_codes_reference(system, params, expected, jax.random.PRNGKey(3))
    for a, b in zip(expected, again, strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    for mean, draw, ref, family in zip(expected, keyed, reference, families, strict=True):
        assert draw.shape == mean.shape and draw.dtype == mean.dtype
        assert np.array_equal(np.asarray(draw), np.asarray(ref))
        if family == _amortized.CODE:
            d = np.asarray(draw)
            assert np.all((d == 0) | (d == 1)) and np.all(d.sum(-1) == 1)
            assert not np.array_equal(d, np.asarray(mean))
        else:
            assert np.array_equal(np.asarray(draw), np.asarray(mean))


def test_act_sampling_is_the_default_and_the_ablation_reproduces_a_distinct_stream(tmp_path):
    """The lane acts on a sampled code by default; ``act_sampling=False`` is the
    pre-2026-09-28 lane, a different stream from the same carry; each is a pure
    function of ``(params, carry)`` (run twice, bitwise); a stored ``run.json``
    without the knob means the old lane."""
    cfg = _config(tmp_path, steps=128, chunk_steps=128)
    assert cfg.act_sampling is True
    assert run_mc._ABSENT_MEANS["act_sampling"] is False
    system, crspp, env, state, buffers, _, _, horizon = run_mc.build(cfg)
    args = (state, run_mc._stack([b[0] for b in buffers]), run_mc._stack([b[1] for b in buffers]))
    ups = run_mc.m_only(len(SEEDS), cfg.chunk_steps // cfg.train_every)
    streams = {}
    for sampling in (True, False):
        chunk_fn = run_mc.make_chunk_fn(
            system,
            crspp,
            env,
            chunk_steps=cfg.chunk_steps,
            train_every=cfg.train_every,
            lanes=cfg.lanes,
            batch_size=cfg.batch_size,
            horizon=horizon,
            act_sampling=sampling,
        )
        first = chunk_fn(*args, ups)
        second = chunk_fn(*args, ups)
        for a, b in zip(
            jax.tree_util.tree_leaves(first), jax.tree_util.tree_leaves(second), strict=True
        ):
            assert np.array_equal(np.asarray(a), np.asarray(b))
        assert np.all(np.isfinite(np.asarray(first[1].log_entropy)))
        streams[sampling] = first[1]
    assert not np.array_equal(np.asarray(streams[True].action), np.asarray(streams[False].action))
    assert not np.array_equal(
        np.asarray(streams[True].log_entropy), np.asarray(streams[False].log_entropy)
    )


def test_the_acting_start_is_the_training_restart(tmp_path):
    """An episode's first act-time step reads the state the world model's training restart
    reads: the learned carry and the prior's mode (a one-hot code), one GRU step on a zero
    action. Its carry equals the first filtered step of ``observe`` at ``is_first`` bitwise,
    whatever action came in and whatever that step observes."""
    from hybrid_aif.train.world import assemble

    system, _, _, state, *_ = run_mc.build(_config(tmp_path))
    params = jax.tree_util.tree_map(lambda x: x[0], state.params)
    core, _ = assemble(system, params)
    restart = core.initial_state()
    assert jnp.all((restart.stoch == 0) | (restart.stoch == 1))
    assert jnp.all(jnp.sum(restart.stoch, axis=-1) == 1)
    start = run_mc.episode_start_beliefs(system, params)
    states, _, _ = core.observe(
        jax.random.PRNGKey(0),
        jnp.full((1, 1), 0.7),  # zeroed at is_first
        jnp.asarray([[-0.5, 0.01]]),
        is_first=jnp.ones((1,), bool),
    )
    assert jnp.array_equal(start["deter"], states.deter[0])
