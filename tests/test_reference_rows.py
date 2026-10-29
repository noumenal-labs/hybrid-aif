"""Objective rows a reference subtree defines register when the tier is imported —
and a document that names one says so when the tier was not."""

from __future__ import annotations

import subprocess
import sys

import jax

from hybrid_aif.description import Description, describe, save
from hybrid_aif.reference.raif.agent import raif_mountain_car
from hybrid_aif.reference.raif.run import SMALL

KEY = jax.random.PRNGKey(0)


def test_a_saved_agent_builds_in_a_fresh_interpreter_once_the_tier_is_imported(tmp_path):
    agent = raif_mountain_car(KEY, **SMALL)
    document, arrays = describe(agent.system, agent.params, arrays="file")
    path, _ = save(Description(document, arrays), tmp_path / "agent")
    code = (
        "import sys, jax; from hybrid_aif.description import load, build\n"
        "import hybrid_aif.reference  # registers the tier's rows\n"
        f"doc = load({str(path)!r})\n"
        "build(doc.document, arrays=doc.arrays, key=jax.random.PRNGKey(1))\n"
        "print('built')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0 and "built" in out.stdout, out.stderr[-2000:]


def test_without_the_tier_the_document_names_the_missing_import(tmp_path):
    agent = raif_mountain_car(KEY, **SMALL)
    document, arrays = describe(agent.system, agent.params, arrays="file")
    path, _ = save(Description(document, arrays), tmp_path / "agent")
    code = (
        "import sys, jax; from hybrid_aif.description import load, build\n"
        f"doc = load({str(path)!r})\n"
        "try:\n"
        "    build(doc.document, arrays=doc.arrays, key=jax.random.PRNGKey(1))\n"
        "except Exception as e:\n"
        "    print('REFUSED', str(e))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert "REFUSED" in out.stdout and "import hybrid_aif.reference" in out.stdout, (
        out.stdout[-2000:] + out.stderr[-2000:]
    )
