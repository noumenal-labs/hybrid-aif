"""Oracle-export seed episodes: the ``.npz`` fixture format, its loader and its writer.

The fixtures beside this file, ``oracle_mc_seed<N>.npz`` with a
``oracle_mc_seed<N>.provenance.json`` of the same stem, are the oracle's ``collect_prior``
episodes for seed ``N``. Their rows follow the driver convention the collection loop also
uses, so fixture and collected episodes share a buffer without conversion.

Eight arrays, concatenated over episodes along axis 0 (``N`` total steps)::

    obs               (N, 2)  float32   (position, velocity)
    action            (N, 1)  float32   the raw actor sample, unclipped; zero on the
                                        terminal row
    reward            (N,)    float32   the reward of action[i−1]; 0 on a reset row
    is_success        (N,)    bool      the goal predicate holds on obs[i]
    has_succeeded     (N,)    bool      it has held at some step of this episode
    can_self_imitate  (N,)    bool      the recorded expert flag (the imitation mask
                                        is derived, like pref_label, and not stored)
    is_first          (N,)    bool      first step of an episode
    is_last           (N,)    bool      last step of an episode

Episode boundaries come from ``is_first``/``is_last`` alone, and the loader checks that
they agree. The provenance file must carry every key in :data:`PROVENANCE_KEYS`; a missing
file raises, and its ``n_episodes``/``n_steps`` are checked against the arrays.

Design: ``docs/design/amortization.md`` A12 item 6.
"""

import json
import pathlib

import numpy as np

from hybrid_aif.reference.raif.columns import RECORDED, raif_episode, recorded_columns
from hybrid_aif.train.buffers import Episode

#: The keys ``provenance.json`` must carry: A12 item 6's ``{SHA, command, seed}`` plus
#: the counts that check the file against its arrays.
PROVENANCE_KEYS: tuple[str, ...] = (
    "repo",
    "commit_sha",
    "command",
    "seed",
    "collected_at",
    "n_episodes",
    "n_steps",
)

#: The declared per-step widths of the two vector columns, state-based Mountain Car
#: (addendum §1). Checked, not assumed: a pixel export would be a different fixture.
OBS_DIM = 2
ACTION_DIM = 1


def provenance_path(path: str | pathlib.Path) -> pathlib.Path:
    """The ``provenance.json`` that must sit beside a fixture: ``<stem>.provenance.json``."""
    path = pathlib.Path(path)
    return path.with_suffix("").with_suffix(".provenance.json")


def load_provenance(path: str | pathlib.Path) -> dict:
    """The fixture's provenance record, with every required key present."""
    beside = provenance_path(path)
    if not beside.exists():
        raise FileNotFoundError(
            f"no provenance beside {pathlib.Path(path).name}: expected {beside}. A12 item 6 "
            "requires it — {repo, commit_sha, command, seed, collected_at, n_episodes, "
            "n_steps} — because an undocumented fixture rots into magic data, and the whole "
            "point of exporting from the oracle is that its numbers are traceable to a run."
        )
    record = json.loads(beside.read_text())
    missing = sorted(set(PROVENANCE_KEYS) - set(record))
    if missing:
        raise ValueError(f"{beside.name}: missing provenance key(s) {missing}")
    return record


def load_fixture(path: str | pathlib.Path) -> tuple[list[Episode], dict]:
    """Load an oracle-export fixture: ``(episodes, provenance)``.

    Validates names, dtypes, widths and one common ``N``, checks the episode boundaries and the
    provenance counts, then splits on ``is_first``. Raises ``ValueError`` on any mismatch.
    """
    path = pathlib.Path(path)
    record = load_provenance(path)
    with np.load(path) as data:
        present = set(data.files)
        missing = sorted(set(RECORDED) - present)
        extra = sorted(present - set(RECORDED))
        if missing or extra:
            raise ValueError(
                f"{path.name}: missing array(s) {missing}, unexpected array(s) {extra}; the "
                f"fixture format is {list(RECORDED)} (see this module's docstring)"
            )
        columns = {name: np.asarray(data[name]) for name in RECORDED}
    _check_columns(path.name, columns)
    bounds = _episode_bounds(path.name, columns["is_first"], columns["is_last"])
    episodes = [
        raif_episode(**{name: column[start : stop + 1] for name, column in columns.items()})
        for start, stop in bounds
    ]
    n_steps = columns["obs"].shape[0]
    if (record["n_episodes"], record["n_steps"]) != (len(episodes), n_steps):
        raise ValueError(
            f"{path.name}: provenance says {record['n_episodes']} episode(s) / "
            f"{record['n_steps']} step(s) but the arrays hold {len(episodes)} / {n_steps}; "
            "re-export the pair together"
        )
    return episodes, record


def _check_columns(name: str, columns: dict[str, np.ndarray]) -> None:
    n = columns["obs"].shape[0]
    if columns["obs"].shape != (n, OBS_DIM):
        raise ValueError(f"{name}: obs is {columns['obs'].shape}, expected (N, {OBS_DIM})")
    if columns["action"].shape != (n, ACTION_DIM):
        raise ValueError(f"{name}: action is {columns['action'].shape}, expected (N, {ACTION_DIM})")
    for field in RECORDED[2:]:
        if columns[field].shape != (n,):
            raise ValueError(f"{name}: {field} is {columns[field].shape}, expected (N,) with N={n}")
    for field in RECORDED[3:]:
        if columns[field].dtype != np.bool_:
            raise ValueError(f"{name}: {field} has dtype {columns[field].dtype}, expected bool")
    #   `can_self_imitate` is the bool recorded flag on disk; the Episode constructor casts
    #   it to the float the imitation mask replaces it with.
    if n == 0:
        raise ValueError(f"{name}: the fixture is empty")


def _episode_bounds(name: str, is_first: np.ndarray, is_last: np.ndarray) -> list[tuple[int, int]]:
    """Episode ``(start, stop)`` index pairs, with the flags checked against each other."""
    starts = np.flatnonzero(is_first)
    stops = np.flatnonzero(is_last)
    if starts.size == 0 or stops.size == 0:
        raise ValueError(
            f"{name}: no episode boundaries — is_first has {starts.size} and is_last "
            f"{stops.size} set. Boundaries are what makes the concatenation lossless."
        )
    if starts.size != stops.size:
        raise ValueError(
            f"{name}: {starts.size} is_first flag(s) but {stops.size} is_last — every episode "
            "has exactly one of each"
        )
    if starts[0] != 0 or stops[-1] != is_last.size - 1:
        raise ValueError(
            f"{name}: the fixture must begin at an episode start and end at an episode end; "
            f"is_first[0]={bool(is_first[0])}, is_last[-1]={bool(is_last[-1])} (a partial "
            "episode at either end would be a silently truncated trajectory)"
        )
    if not np.all(starts <= stops) or not np.all(stops[:-1] < starts[1:]):
        raise ValueError(
            f"{name}: is_first/is_last do not interleave — starts {starts.tolist()[:8]}, stops "
            f"{stops.tolist()[:8]}. Episodes may not overlap or nest."
        )
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def save_fixture(
    path: str | pathlib.Path, episodes: list[Episode], provenance: dict
) -> pathlib.Path:
    """Write episodes and their provenance in the fixture format; return the ``.npz`` path.

    A float ``can_self_imitate`` (after :func:`~hybrid_aif.reference.raif.revise.label`) is
    stored as ``!= 0``; ``pref_label`` is not stored.
    """
    path = pathlib.Path(path)
    missing = sorted(set(PROVENANCE_KEYS) - set(provenance))
    if missing:
        raise ValueError(f"save_fixture: missing provenance key(s) {missing}")
    columns = {
        name: np.concatenate([recorded_columns(ep)[name] for ep in episodes], axis=0)
        for name in RECORDED
    }
    #   the format stores flags: a float imitation mask is narrowed to `!= 0`, and
    #   `label` recomputes both derived columns
    columns["can_self_imitate"] = columns["can_self_imitate"] != 0
    record = dict(provenance)
    record["n_episodes"] = len(episodes)
    record["n_steps"] = int(columns["obs"].shape[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **columns)
    provenance_path(path).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path
