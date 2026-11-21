"""Trim a survey bundle for the project page.

    python examples/survey_trim.py IN.json OUT_DIR
    python examples/survey_trim.py IN/bundle_v2.json OUT_DIR --name HYBRID_AIF_BUNDLE_V2 \
        --file bundle_v2.js

Writes OUT_DIR/bundle.js (``window.HYBRID_AIF_BUNDLE = ...``), the one data file the page loads,
or the file ``--file`` defining ``window.<--name>``. Every tick is kept, every heavy field too.
Every float is rounded to four significant digits, each tick (and each slow step of a two-tier
run, under ``slow``) keeps at most ``MAX_POLICIES`` policies (the best by the midpoint of G,
always including the chosen one, in their original order with ``chosen`` re-indexed), and each
image (``img``, ``recon``) and the field's table (``map``) becomes one base64 string of its bytes,
in the layouts ``meta.image.layout`` and ``meta.map_layout`` describe. A multi-rover run's rows
of each rover (``run["rovers"][name]``) are trimmed and packed as its ticks are; its joint
decisions (``slow`` with ``chosen`` one record per rover) carry no policy list to cap, and are
rounded.
"""

import argparse
import base64
import json
from pathlib import Path

MAX_POLICIES = 12
PACKED = ("img", "recon", "map")


def rounded(obj):
    if isinstance(obj, float):
        return float(f"{obj:.4g}")
    if isinstance(obj, list):
        return [rounded(v) for v in obj]
    if isinstance(obj, dict):
        return {k: rounded(v) for k, v in obj.items()}
    return obj


def trim_policies(tick, limit=MAX_POLICIES, per_option=False):
    """Keep the ``limit`` best policies of ``tick`` (a rover tick or a slow step), the chosen one
    among them, in their original order with ``chosen`` re-indexed. With ``per_option`` (a slow
    step) the best sequence of every first option is kept as well."""
    policies = tick.get("policies")
    if not policies or len(policies) <= limit or not isinstance(policies[0], dict):
        return  # a joint decision's ``policies`` counts each rover's sequences: nothing to cap
    chosen = tick["chosen"]
    order = sorted(range(len(policies)), key=lambda i: sum(policies[i]["G"]) / 2)
    keep = {chosen}
    for i in order:
        if len(keep) >= limit:
            break
        keep.add(i)
    if per_option:  # the best sequence of each first option, for the slow tier's bars
        best = {}
        for i in order:
            best.setdefault(policies[i]["options"][0], i)
        keep |= set(best.values())
    kept = sorted(keep)
    tick["policies"] = [policies[i] for i in kept]
    tick["chosen"] = kept.index(chosen)


def pack(tick):
    for key in PACKED:
        rows = tick.get(key)
        if isinstance(rows, list):
            tick[key] = base64.b64encode(bytes(v for row in rows for v in row)).decode("ascii")


def trim_run(run):
    """Trim and pack every tick of a run, and every row of each of its rovers."""
    for rows in [run["ticks"], *run.get("rovers", {}).values()]:
        for tick in rows:
            trim_policies(tick)
            if tick.get("slow", {}).get("decision"):
                trim_policies(tick["slow"], per_option=True)
            pack(tick)


def main(src, out, name="HYBRID_AIF_BUNDLE", file="bundle.js"):
    bundle = json.loads(Path(src).read_text())
    meta = bundle.setdefault("meta", {})
    for run in bundle["runs"].values():
        trim_run(run)
    meta.setdefault("image", {})["encoding"] = "base64 of the pixel bytes"
    meta["map_encoding"] = "base64 of the cell bytes, three per cell"
    text = json.dumps(rounded(bundle), separators=(",", ":"), allow_nan=False)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / file).write_text(f"window.{name} = " + text + ";\n")
    print(f"{Path(src).stat().st_size / 1e6:.2f} MB -> {len(text) / 1e6:.2f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--name", default="HYBRID_AIF_BUNDLE", help="the window variable defined")
    ap.add_argument("--file", default="bundle.js", help="the file written in OUT_DIR")
    a = ap.parse_args()
    main(a.src, a.out, a.name, a.file)
