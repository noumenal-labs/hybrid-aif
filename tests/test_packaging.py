"""Packaging metadata and PEP 561 marker checks."""

import importlib.resources
import tomllib
from pathlib import Path


def test_py_typed_marker_is_present():
    marker = importlib.resources.files("hybrid_aif").joinpath("py.typed")
    assert marker.is_file()


def test_pyproject_python_floor_matches_jax_floor():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data["project"]
    assert project["requires-python"] == ">=3.12"
    assert "Repository" in project["urls"]
    assert "Typing :: Typed" in project["classifiers"]
