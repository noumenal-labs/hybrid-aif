"""Export the oracle's ``collect_prior`` expert stream as checked-in fixtures.

The one implementation of A12 decision 6's export, run once per oracle seed on
the machine holding the ``raif-paper`` worktree (see the provenance JSON beside
each ``oracle_mc_seed<N>.npz`` for the exact run). It reads the DreamerV3-style
chunk streams under ``logs/<exp>/replay`` — one chain per env worker, ordered by
the filename's ``<time>-<id>-<successor>-<length>`` fields, each chunk trimmed
to its recorded length (a final chunk's tail is uninitialized) — takes the
chains stamped within 60 s of the first (the collect run; training's chunks
flush minutes later), drops leading/trailing partial episodes at chain
boundaries, and writes whole episodes through
:func:`hybrid_aif.reference.raif.fixtures.save_fixture`. The COMPLETE collect stream
is exported, successes and failures alike: M⁺ is reproduced at seed time by the
recipe's own routing rule, never baked into the file. Recorded truth on
``can_self_imitate``: 1 on expert-authored steps, 0 on the terminal step (the
goal step's action is the reset filler) and 0 throughout success-gated
failures.

Usage::

    python -m hybrid_aif.reference.raif.fixtures.export_oracle_mc /path/to/raif-paper
"""

import datetime as dt
import json
import pathlib
import re
import sys

import numpy as np

from hybrid_aif.reference.raif.columns import raif_episode
from hybrid_aif.reference.raif.fixtures import load_fixture, save_fixture
from hybrid_aif.train.buffers import seed_stats

OUT = pathlib.Path(__file__).parent
COMMIT = "9b1864e8fab9d1becd6dd248f2c590d06ff22848"
NAME = re.compile(r"^(\d{8}T\d{6})F(\d+)-([A-Za-z0-9]+)-([A-Za-z0-9]+)-(\d+)\.npz$")
KEYS = (
    "state",
    "action",
    "reward",
    "is_success",
    "has_succeeded",
    "can_self_imitate",
    "is_first",
    "is_last",
)


def fields(f: pathlib.Path) -> tuple[float, str, str, int]:
    m = NAME.match(f.name)
    assert m, f.name
    ts, frac, cid, succ, ln = m.groups()
    t = dt.datetime.strptime(ts, "%Y%m%dT%H%M%S").timestamp() + int(frac) / 1e6
    return t, cid, succ, int(ln)


def collect_episodes(replay_dir: pathlib.Path, seed: int) -> list:
    files = sorted(replay_dir.glob("*.npz"), key=lambda f: fields(f)[0])
    t0 = fields(files[0])[0]
    # the collect chains share a sub-second stamp; training's chunks flush minutes later
    window = [f for f in files if fields(f)[0] < t0 + 60.0]
    by_id = {fields(f)[1]: f for f in window}
    succ_of = {fields(f)[1]: fields(f)[2] for f in window}
    referenced = set(succ_of.values())
    heads = [cid for cid in by_id if cid not in referenced]
    assert 1 <= len(heads) <= 8, (seed, len(heads))  # one chain per env worker
    episodes = []
    for head in sorted(heads, key=lambda c: fields(by_id[c])[0]):
        chain, cid = [], head
        while cid in by_id:
            chain.append(by_id[cid])
            cid = succ_of[cid]
        cols = {k: np.concatenate([np.load(f)[k][: fields(f)[3]] for f in chain]) for k in KEYS}
        assert np.isfinite(cols["state"]).all() and np.isfinite(cols["reward"]).all(), seed
        firsts = list(np.flatnonzero(cols["is_first"]))
        lasts = list(np.flatnonzero(cols["is_last"]))
        if lasts and firsts and lasts[0] < firsts[0]:
            lasts = lasts[1:]  # chain starts mid-episode: drop the leading partial
        if len(firsts) == len(lasts) + 1 and (not lasts or firsts[-1] > lasts[-1]):
            firsts = firsts[:-1]  # in-progress episode at shutdown: drop the trailing partial
        pairs = list(zip(firsts, lasts, strict=True))
        assert all(b >= a for a, b in pairs), (seed, len(firsts), len(lasts))
        csi = cols["can_self_imitate"]
        assert np.isin(csi, (0.0, 1.0)).all(), f"seed {seed}: fractional csi in collect stream"
        for a, b in pairs:
            sl = slice(a, b + 1)
            episodes.append(
                raif_episode(
                    obs=cols["state"][sl].astype(np.float32),
                    action=cols["action"][sl].astype(np.float32),
                    reward=cols["reward"][sl].astype(np.float32),
                    is_success=cols["is_success"][sl],
                    has_succeeded=cols["has_succeeded"][sl],
                    can_self_imitate=csi[sl].astype(np.float32),
                    is_first=cols["is_first"][sl],
                    is_last=cols["is_last"][sl],
                )
            )
    return episodes


def main() -> None:
    raif = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "../raif-paper").resolve()
    for seed in range(4):
        replay_dir = raif / f"logs/oracle_state_mc_seed{seed}/replay"
        eps = collect_episodes(replay_dir, seed)
        n_steps = sum(int(np.asarray(e.obs).shape[0]) for e in eps)
        # completed episodes only: episodes in progress at shutdown are dropped above
        assert 1500 <= n_steps <= 2400, (seed, n_steps)
        t0 = fields(sorted(replay_dir.glob("*.npz"), key=lambda f: fields(f)[0])[0])[0]
        prov = {
            "repo": "NACLab/robust-active-inference (local checkout: raif-paper)",
            "commit_sha": COMMIT,
            "command": (
                "python scripts/collect_prior.py --configs collect gym_mtc --run.steps 2000 "
                f"--expname oracle_state_mc_seed{seed} --seed {seed}"
            ),
            "seed": seed,
            "note": (
                "complete collect stream from replay/ (per-env-worker chains; partial "
                "episodes at chain boundaries dropped); M+ is reproduced by the recipe "
                "routing successes"
            ),
            "collected_at": dt.datetime.fromtimestamp(t0).isoformat(),
            # validated then re-stamped by save_fixture:
            "n_episodes": len(eps),
            "n_steps": n_steps,
        }
        path = save_fixture(OUT / f"oracle_mc_seed{seed}.npz", eps, prov)
        back, record = load_fixture(path)
        stats = seed_stats(back)
        print(
            f"seed {seed}: episodes {record['n_episodes']}, steps {record['n_steps']}, "
            f"return {stats.mean_return:.1f}, SR {stats.success_rate:.2f}, "
            f"mean len {stats.mean_length:.1f}"
        )
        print(json.dumps({k: record[k] for k in ("commit_sha", "collected_at")}))


if __name__ == "__main__":
    main()
