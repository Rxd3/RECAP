"""Round 3, item 8: `crf_io.load_artifact` is the one joblib loader. The CRF reader, the inspection, the gap classifier,
the model pipeline and the convergence report open an artifact through it, so they refuse alike: the same wording for a
file that is not a readable artifact and for a missing Python package, and the same pass-through of an OS error.

The pipeline words every unreadable model file its own way (`unsupported model file (unreadable)`, for JSON too)."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from punct_restore import pipeline
from punct_restore.models import crf_inspection, crf_io
from punct_restore.models.gap_classifier import GapClassifier
from punct_restore.paths import REPO_ROOT
from punct_restore.reporting import crf_convergence

PACKAGE = REPO_ROOT / "src" / "punct_restore"

Opener = Callable[[Path], Any]
OPENERS: dict[str, Opener] = {
    "crf_io.read_bundle": crf_io.read_bundle,
    "crf_inspection.inspect_artifact": crf_inspection.inspect_artifact,
    "gap_classifier.load_model": GapClassifier.load_model,
    "pipeline.load_model": pipeline.load_model,
    "crf_convergence.convergence_entry": crf_convergence.convergence_entry,
}
PIPELINE = "pipeline.load_model"


@pytest.fixture(params=list(OPENERS))
def opener_name(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def _refusal(opener_name: str, path: Path) -> str:
    with pytest.raises(ValueError) as caught:
        OPENERS[opener_name](path)
    return str(caught.value)


def test_a_file_that_is_not_a_readable_artifact_is_refused_in_one_wording(tmp_path: Path, opener_name: str) -> None:
    path = tmp_path / "junk.joblib"
    path.write_bytes(b"this is not a joblib file")
    expected = "unsupported model file (unreadable)" if opener_name == PIPELINE else "not a readable artifact"
    assert _refusal(opener_name, path) == f"junk.joblib: {expected}"


def test_a_missing_python_package_is_named_by_every_opener(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, opener_name: str
) -> None:
    path = tmp_path / "m.joblib"
    path.write_bytes(b"x")

    def need_package(_path: Path) -> None:
        raise ModuleNotFoundError("No module named 'some_pkg'", name="some_pkg")

    monkeypatch.setattr(crf_io.joblib, "load", need_package)
    expected = "unsupported model file (unreadable)" if opener_name == PIPELINE \
        else "needs the Python package some_pkg to open it"
    assert _refusal(opener_name, path) == f"m.joblib: {expected}"


def test_an_import_error_that_names_no_package_is_an_unreadable_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "m.joblib"
    path.write_bytes(b"x")

    def anonymous(_path: Path) -> None:
        raise ImportError("something")

    monkeypatch.setattr(crf_io.joblib, "load", anonymous)
    with pytest.raises(ValueError, match=r"^m\.joblib: not a readable artifact$"):
        crf_io.load_artifact(path, "model artifact")


def test_an_os_error_while_reading_stays_an_os_error_for_every_opener(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, opener_name: str
) -> None:
    path = tmp_path / "locked.joblib"
    path.write_bytes(b"x")

    def denied(p: Path) -> None:
        raise PermissionError(13, "Permission denied", str(p))

    monkeypatch.setattr(crf_io.joblib, "load", denied)
    with pytest.raises(OSError) as caught:
        OPENERS[opener_name](path)
    assert not isinstance(caught.value, ValueError)


@pytest.mark.parametrize("name", [n for n in OPENERS if n != "crf_convergence.convergence_entry"])
def test_a_missing_file_and_a_folder_are_named_before_anything_is_loaded(tmp_path: Path, name: str) -> None:
    (tmp_path / "adir.joblib").mkdir()
    with pytest.raises(FileNotFoundError, match=r"^ghost\.joblib: .* not found$"):
        OPENERS[name](tmp_path / "ghost.joblib")
    with pytest.raises(IsADirectoryError, match=r"^adir\.joblib: is a folder, not a file$"):
        OPENERS[name](tmp_path / "adir.joblib")


def test_the_convergence_report_names_a_missing_file_and_a_folder_as_a_value_error(tmp_path: Path) -> None:
    """Its boundary hides a message-only OSError, so it words these two itself (as asr_noise_table does)."""
    (tmp_path / "adir.joblib").mkdir()
    with pytest.raises(ValueError, match=r"^ghost\.joblib: .* not found$"):
        crf_convergence.convergence_entry(tmp_path / "ghost.joblib")
    with pytest.raises(ValueError, match=r"^adir\.joblib: is a folder, not a file$"):
        crf_convergence.convergence_entry(tmp_path / "adir.joblib")


def test_joblib_load_is_called_in_one_place_only() -> None:
    callers = sorted(path.relative_to(PACKAGE).as_posix() for path in PACKAGE.rglob("*.py")
                     if re.search(r"\bjoblib\.load\(", path.read_text(encoding="utf-8")))
    assert callers == ["models/crf_io.py"]


def test_the_convergence_report_no_longer_has_a_read_bundle_that_shadows_the_crf_one() -> None:
    assert not hasattr(crf_convergence, "read_bundle")
    assert callable(crf_convergence.read_convergence_bundle)
    assert crf_io.read_bundle is not crf_convergence.read_convergence_bundle


def test_the_convergence_report_still_refuses_a_pickle_that_is_no_crf_bundle(tmp_path: Path) -> None:
    import joblib  # this test writes the file a line before it reads it

    path = tmp_path / "list.joblib"
    joblib.dump(["not", "a", "bundle"], path)
    with pytest.raises(ValueError, match=r"^list\.joblib: not a CRF model file$"):
        crf_convergence.read_convergence_bundle(path)
