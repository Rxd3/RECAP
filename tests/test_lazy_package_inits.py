"""Round 3, item 9 (course N2): importing punct_restore.data or punct_restore.evaluation loads only the submodule a
name comes from, so the HMM provenance closure stops hashing the evaluator and the release builder.

Every name the two packages exported before stays importable (PEP 562 module __getattr__)."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from punct_restore.lazy_exports import lazy_exports
from punct_restore.models import hmm_experiment_identity
from punct_restore.paths import REPO_ROOT

DATA_NAMES = {"ReleaseSources": "release", "build_release": "release", "build_release_from": "release",
              "join_labels": "labeling", "load_label_sets": "labeling", "load_records": "release",
              "read_ids": "release", "validate_file": "validation", "validate_record": "validation"}
EVALUATION_NAMES = {"BootstrapSpec": "uncertainty", "EvaluationInputError": "inputs", "check_manifest": "manifest",
                    "compare": "comparison", "comparison_markdown": "comparison", "error_examples": "error_analysis",
                    "errors_markdown": "error_analysis", "evaluate": "report", "load_bundle": "inputs",
                    "load_scored_gold": "inputs", "load_vocab": "inputs", "shortcut_predictions": "baseline",
                    "to_markdown": "report"}
PACKAGES = {"punct_restore.data": DATA_NAMES, "punct_restore.evaluation": EVALUATION_NAMES}


def _fresh(code: str) -> list[str]:
    """What a fresh interpreter prints for ``code`` (one item per line), with the worktree's code on the path."""
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    return done.stdout.split()


def _loaded_after(statement: str, prefix: str) -> list[str]:
    code = f"import sys\n{statement}\nprint(*sorted(m for m in sys.modules if m.startswith({prefix!r})))"
    return _fresh(code)


# ---------------------------------------------------------------- every old name still works
@pytest.mark.parametrize(("package", "name"), [(p, n) for p, names in PACKAGES.items() for n in names])
def test_a_name_the_package_exported_is_the_object_of_its_submodule(package: str, name: str) -> None:
    module = importlib.import_module(f"{package}.{PACKAGES[package][name]}")
    exported = getattr(importlib.import_module(package), name)
    assert exported is getattr(module, name)


@pytest.mark.parametrize("package", PACKAGES)
def test_from_import_and_hasattr_and_star_import_find_every_exported_name(package: str) -> None:
    names = sorted(PACKAGES[package])
    code = (f"import {package} as p\n"
            f"print(*(int(hasattr(p, n)) for n in {names!r}))\n"
            f"from {package} import {', '.join(names)}\n"
            f"namespace = {{}}\nexec('from {package} import *', namespace)\n"
            f"print(*(int(n in namespace) for n in {names!r}))")
    assert set(_fresh(code)) == {"1"}


@pytest.mark.parametrize("package", PACKAGES)
def test_all_and_dir_list_every_exported_name(package: str) -> None:
    module = importlib.import_module(package)
    assert set(PACKAGES[package]) <= set(module.__all__)
    assert set(PACKAGES[package]) <= set(dir(module))


@pytest.mark.parametrize("package", PACKAGES)
def test_an_unknown_name_is_an_attribute_error_and_hasattr_says_false(package: str) -> None:
    module = importlib.import_module(package)
    unknown = "no_such_name"
    assert not hasattr(module, unknown)
    with pytest.raises(AttributeError, match=unknown):
        getattr(module, unknown)


@pytest.mark.parametrize("package", PACKAGES)
def test_a_submodule_is_still_an_attribute_after_importing_only_the_package(package: str) -> None:
    submodule = sorted(set(PACKAGES[package].values()))[0]
    code = f"import {package} as p\nprint(p.{submodule}.__name__)"
    assert _fresh(code) == [f"{package}.{submodule}"]


# ---------------------------------------------------------------- nothing is loaded before it is asked for
def test_importing_the_data_package_loads_none_of_its_submodules() -> None:
    assert _loaded_after("import punct_restore.data", "punct_restore.data") == ["punct_restore.data"]


def test_importing_the_evaluation_package_loads_none_of_its_submodules() -> None:
    assert _loaded_after("import punct_restore.evaluation", "punct_restore.evaluation") == ["punct_restore.evaluation"]


def test_asking_for_one_name_loads_its_submodule_and_what_that_needs_not_the_rest() -> None:
    loaded = _loaded_after("from punct_restore.data import validate_record", "punct_restore.data")
    assert loaded == ["punct_restore.data", "punct_restore.data.validation"]


def test_the_second_access_does_not_go_through_the_lookup_again() -> None:
    code = ("import punct_restore.data as p\np.load_records\n"
            "print(int('load_records' in vars(p)))")
    assert _fresh(code) == ["1"]


# ---------------------------------------------------------------- the helper itself
def test_lazy_exports_gives_the_getattr_and_dir_of_a_package(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    package = tmp_path / "lazy_pkg_for_test"
    package.mkdir()
    (package / "__init__.py").write_text(
        "from punct_restore.lazy_exports import lazy_exports\n"
        "__all__ = ['thing']\n"
        "__getattr__, __dir__ = lazy_exports(__name__, {'thing': 'inner'})\n", encoding="utf-8")
    (package / "inner.py").write_text("thing = 42\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("lazy_pkg_for_test")
    assert "lazy_pkg_for_test.inner" not in sys.modules
    assert module.thing == 42  # type: ignore[attr-defined]
    assert "thing" in dir(module) and "thing" in vars(module)
    unknown = "no_such"
    with pytest.raises(AttributeError, match=unknown):
        getattr(module, unknown)
    for name in ("lazy_pkg_for_test", "lazy_pkg_for_test.inner"):
        sys.modules.pop(name, None)


def test_lazy_exports_is_a_pair_of_functions() -> None:
    getter, lister = lazy_exports("punct_restore.data", {})
    assert callable(getter) and callable(lister)


# ---------------------------------------------------------------- the HMM closure
def test_the_hmm_commands_load_neither_the_evaluator_nor_the_release_builder() -> None:
    statement = ("import punct_restore.cli.predict_hmm, punct_restore.cli.run_hmm_experiments\n"
                 "import punct_restore.cli.train_hmm, punct_restore.cli.verify_hmm_experiments")
    loaded = _loaded_after(statement, "punct_restore.")
    heavy = [m for m in loaded if m.startswith("punct_restore.evaluation")
             or m.split(".")[:3] in (["punct_restore", "data", "release"], ["punct_restore", "data", "extraction"],
                                     ["punct_restore", "data", "splits"])
             or m.startswith(("punct_restore.data.release", "punct_restore.data.gold_noise"))]
    assert heavy == []


def test_source_files_hold_no_evaluator_and_no_release_builder_file() -> None:
    names = hmm_experiment_identity.SOURCE_FILES
    assert [n for n in names if "/evaluation/" in n or "/data/release" in n or n.endswith(("extraction.py",
                                                                                          "splits.py"))] == []
    assert len(names) < 54  # 54 entries before the lazy package inits and the file_identity move
