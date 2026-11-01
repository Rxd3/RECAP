"""Dependencies are exact pins, declared once in pyproject.toml, mirrored in requirements.txt, and complete."""

from __future__ import annotations

import ast
import importlib.metadata
import re
import sys
import tomllib
from collections.abc import Iterable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*==[0-9][A-Za-z0-9.+!_-]*$")
EDITABLE_INSTALL = "-e .[dev]"
IMPORT_TO_DISTRIBUTION = {
    "yaml": "PyYAML",
    "joblib": "joblib",
    "sklearn_crfsuite": "sklearn-crfsuite",
    "pycrfsuite": "python-crfsuite",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
}
PROJECT_PACKAGE = "punct_restore"


def _pyproject_pins() -> dict[str, list[str]]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    return {"runtime": list(project["dependencies"]), "dev": list(project["optional-dependencies"]["dev"])}


def _requirements_lines() -> list[str]:
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def _distribution(name: str) -> str:
    """PEP 503 normalised distribution name, so 'PyYAML' and 'pyyaml' compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _first_party_names() -> set[str]:
    """Names of the script files and of the folders holding them; the scripts are stubs that import punct_restore."""
    scripts = ROOT / "scripts"
    return {path.stem for path in scripts.rglob("*.py")} | {path.parent.name for path in scripts.rglob("*.py")}


def _top_level_imports(files: Iterable[Path]) -> set[str]:
    """Top-level module of every absolute import in the files, including imports inside functions."""
    found: set[str] = set()
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=path.name)):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


def _third_party(imports: set[str], first_party: set[str]) -> set[str]:
    skipped = set(sys.stdlib_module_names) | first_party | {"__future__", PROJECT_PACKAGE}
    return imports - skipped


def _shipped_files() -> list[Path]:
    return [path for folder in ("src", "scripts") for path in (ROOT / folder).rglob("*.py")]


def test_pyproject_dependencies_are_pinned_exactly() -> None:
    pins = _pyproject_pins()
    unpinned = [spec for specs in pins.values() for spec in specs if not PIN.match(spec)]
    assert unpinned == []
    assert pins["runtime"] and pins["dev"]


@pytest.mark.parametrize("spec", [spec for specs in _pyproject_pins().values() for spec in specs])
def test_installed_version_equals_the_pin(spec: str) -> None:
    name, _, pinned = spec.partition("==")
    try:
        installed = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        pytest.skip(f"{name} is not installed")
    assert installed == pinned, f"{name}: installed {installed}, pinned {pinned}"


def test_requirements_txt_pins_exactly_and_keeps_the_editable_install() -> None:
    lines = _requirements_lines()
    assert EDITABLE_INSTALL in lines
    unpinned = [line for line in lines if line != EDITABLE_INSTALL and not PIN.match(line)]
    assert unpinned == []


def test_requirements_txt_lists_exactly_the_pyproject_pins() -> None:
    pins = _pyproject_pins()
    listed = [line for line in _requirements_lines() if line != EDITABLE_INSTALL]
    assert len(listed) == len(set(listed))
    assert set(listed) == set(pins["runtime"]) | set(pins["dev"])


def _lock_pins() -> dict[str, str]:
    """PEP 503 normalised name -> version for every line of requirements-lock.txt (comments skipped)."""
    lines = (ROOT / "requirements-lock.txt").read_text(encoding="utf-8").splitlines()
    pins = [line.strip().partition("==") for line in lines if line.strip() and not line.strip().startswith("#")]
    return {_distribution(name): version for name, _, version in pins}


def test_requirements_lock_is_plain_name_equals_version_lines() -> None:
    lines = (ROOT / "requirements-lock.txt").read_text(encoding="utf-8").splitlines()
    pins = [line for line in lines if line.strip() and not line.startswith("#")]
    assert pins and [line for line in pins if not PIN.match(line)] == []
    assert len(_lock_pins()) == len(pins), "a package is pinned twice in requirements-lock.txt"


def test_requirements_lock_pins_every_direct_dependency_to_the_same_version() -> None:
    wanted = {_distribution(name): version
              for specs in _pyproject_pins().values() for name, _, version in (s.partition("==") for s in specs)}
    locked = _lock_pins()
    assert {name: locked.get(name) for name in wanted} == wanted


def test_requirements_lock_pins_what_the_pickled_crf_models_load_through() -> None:
    """The CRF .joblib models unpickle through these; the lock exists to fix their versions."""
    assert {"scikit-learn", "scipy", "numpy"} <= set(_lock_pins())


def test_every_third_party_import_is_declared() -> None:
    declared = {_distribution(re.match(r"[A-Za-z0-9._-]+", spec)[0]) for spec in _pyproject_pins()["runtime"]}
    imports = _third_party(_top_level_imports(_shipped_files()), _first_party_names())
    assert imports, "the import scan found nothing, so it cannot be trusted"
    unmapped = sorted(name for name in imports if name not in IMPORT_TO_DISTRIBUTION)
    assert unmapped == [], f"third-party imports with no declared distribution: {unmapped}"
    undeclared = sorted(name for name in imports if _distribution(IMPORT_TO_DISTRIBUTION[name]) not in declared)
    assert undeclared == [], f"third-party imports missing from pyproject dependencies: {undeclared}"


def test_import_scan_finds_nested_and_from_imports_and_skips_relative_and_stdlib(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text(
        "from __future__ import annotations\n"
        "import os\nimport numpy as np\nfrom yaml import safe_load\nfrom . import sibling\n"
        "from .pkg import thing\nfrom punct_restore.data import labeling\nimport helper_script\n"
        "def lazy():\n    import joblib.parallel\n",
        encoding="utf-8",
    )
    found = _third_party(_top_level_imports([source]), {"helper_script"})
    assert found == {"numpy", "yaml", "joblib"}
