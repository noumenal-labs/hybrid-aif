"""The one-way rule between the two package tiers.

``hybrid_aif.reference`` is the second tier (see its ``__init__`` docstring):
content pinned to external targets, free to churn, and **never imported by the
engine**. The rule has one direction, so it has one test: parse every engine
module's imports and assert none of them names the reference tier. Reference
content imports the engine freely — asserted once, positively, at the bottom.

Parsing is by ``ast``, not by string search: a path inside a docstring or an
error message is not an import. The tier has subtrees (``reference/raif/`` for
the R-AIF replication); the prefix test covers every one of them.
"""

import ast
import pathlib

import hybrid_aif

ENGINE_ROOT = pathlib.Path(hybrid_aif.__file__).parent
REFERENCE_ROOT = ENGINE_ROOT / "reference"
TIER = "hybrid_aif.reference"


def _imported_names(source: str) -> list[str]:
    """Every module name a source file imports, absolute form only.

    ``import a.b`` yields ``a.b``; ``from a.b import c`` yields ``a.b.c`` as
    well as ``a.b``, so a re-exported symbol is caught by the same prefix test.
    Relative imports (``level > 0``) are not used in this package and are
    skipped rather than resolved.
    """
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
            names.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _runtime_imported_names(source: str) -> list[str]:
    """:func:`_imported_names` without the imports under ``if TYPE_CHECKING:``, which name a type
    for annotations and are never executed."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            node.body = [ast.Pass()]
    return _imported_names(ast.unparse(tree))


def _engine_modules() -> list[pathlib.Path]:
    return sorted(path for path in ENGINE_ROOT.rglob("*.py") if REFERENCE_ROOT not in path.parents)


def test_the_engine_tier_never_imports_the_reference_tier():
    modules = _engine_modules()
    assert modules, f"no engine modules found under {ENGINE_ROOT}"

    violations = [
        (path.relative_to(ENGINE_ROOT.parent), name)
        for path in modules
        for name in _imported_names(path.read_text())
        if name == TIER or name.startswith(f"{TIER}.")
    ]
    assert not violations, "engine modules import the reference tier: " + "; ".join(
        f"{module} imports {name}" for module, name in violations
    )


def test_the_reference_tier_imports_the_engine_freely():
    """The permitted direction, so the rule above reads as one-way, not two-way."""
    from hybrid_aif.reference.raif import world

    assert "hybrid_aif.system" in _imported_names(pathlib.Path(world.__file__).read_text())


#: The engine packages: algorithms over arrays and declarations, which never import a System at
#: run time (an import under ``if TYPE_CHECKING:``, naming a type for annotations, is not one).
#: The faces of ``System`` (``hybrid_aif/system/faces``) assemble calls into them, so the
#: dependency runs one way.
ALGORITHM_PACKAGES = ("inference", "learning", "planning", "objectives")


def test_the_algorithm_packages_never_import_a_system():
    modules = [
        path
        for package in ALGORITHM_PACKAGES
        for path in sorted((ENGINE_ROOT / package).rglob("*.py"))
    ]
    assert modules, f"no modules found under {ALGORITHM_PACKAGES}"

    violations = [
        (path.relative_to(ENGINE_ROOT.parent), name)
        for path in modules
        for name in _runtime_imported_names(path.read_text())
        if name == "hybrid_aif.system" or name.startswith("hybrid_aif.system.")
    ]
    assert not violations, "algorithm modules import the system package: " + "; ".join(
        f"{module} imports {name}" for module, name in violations
    )
