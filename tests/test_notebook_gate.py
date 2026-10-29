"""The notebook-execution adoption gate, as an opt-in marked test.

The notebooks in `examples/` are living exhibits and an adoption gate (CLAUDE.md): a
change that touches public API must leave every one of them executing end to end.
This test enforces it — it runs ``nbconvert --execute`` on a *temp copy* of each
notebook (never in place) and asserts zero error outputs. It is slow (a full
kernel run of every cell), so it carries the ``notebook`` marker and is
deselected by default; run it explicitly with ``pytest -m notebook``.

The convention was recorded after the v0.2 API rewrite silently broke a notebook
section while every unit test stayed green: the notebook is the integration test
the unit suite is not. Notebooks are enumerated from the directory rather than
listed here, so a new exhibit is gated the moment it is added.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
_NOTEBOOKS = sorted(p.name for p in _EXAMPLES.glob("*.ipynb"))


def test_the_gate_covers_every_shipped_notebook():
    """The enumeration is the gate's coverage claim, so it is asserted rather than
    assumed: a notebook nobody runs is exactly the failure this file exists for."""
    assert _NOTEBOOKS, f"no notebooks found in {_EXAMPLES}"


@pytest.mark.notebook
@pytest.mark.parametrize("name", _NOTEBOOKS)
def test_notebook_executes_without_errors(tmp_path, name):
    work = tmp_path / name  # execute a copy, never touch the checked-in notebook
    shutil.copy(_EXAMPLES / name, work)

    # nbconvert runs the kernel in the copied notebook's directory, so the
    # notebook's cwd-relative ``examples/`` import shim cannot see the demo
    # modules — put the real examples dir on PYTHONPATH for the child process.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_EXAMPLES), env.get("PYTHONPATH", "")]).strip(
        os.pathsep
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--output",
            str(work),
            "--ExecutePreprocessor.timeout=600",
            str(work),
        ],
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
    )
    assert result.returncode == 0, f"nbconvert failed on {name}:\n{result.stderr}"

    executed = json.loads(work.read_text())
    errors = [
        (i, o.get("ename"), o.get("evalue"))
        for i, cell in enumerate(executed["cells"])
        if cell["cell_type"] == "code"
        for o in cell.get("outputs", [])
        if o.get("output_type") == "error"
    ]
    assert not errors, f"{name} produced error outputs: " + "; ".join(
        f"cell {i}: {ename}: {value}" for i, ename, value in errors
    )
