"""``examples/ensemble_leaf_variants.py`` runs end to end on untrained agents.

The diagnostic is loaded from ``examples/`` the way the other demo suites load
theirs (``importlib`` off the file path) and driven at its smallest settings, so a
seam it reads through (the stream buffers' sampler, the control layout, the
trainer's rollout step) cannot drift without this test noticing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "examples" / "ensemble_leaf_variants.py"
_spec = importlib.util.spec_from_file_location("ensemble_leaf_variants", _PATH)
elv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(elv)


def test_the_diagnostic_runs_on_untrained_agents(capsys):
    assert (
        elv.main(["--seeds", "0", "--batch", "2", "--keys", "2", "--sigma-scales", "0.0,1.0"]) == 0
    )
    out = capsys.readouterr().out
    assert "seed 0:" in out
    #  at sigma x 0 the sampled reading has no noise to add: the two leaves coincide
    row = next(line for line in out.splitlines() if line.strip().startswith("0.00 "))
    ours, theirs = row.split()[1:3]
    assert ours == theirs
