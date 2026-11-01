"""package-hmm-handoff removes what a failed run built, and never touches a folder it did not create."""

from __future__ import annotations

import errno
from pathlib import Path
from typing import Any

import pytest

from punct_restore.cli import package_hmm_handoff


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """(handoff folder, its zip) under a pretend repository whose artifacts/ folder is empty; the checks pass."""
    monkeypatch.setattr(package_hmm_handoff, "ROOT", tmp_path)
    monkeypatch.setattr(
        package_hmm_handoff, "verified_frozen_selection", lambda run, evaluator: ({"config": {}}, {"modes": {}})
    )
    monkeypatch.setattr(package_hmm_handoff, "read_object", lambda path, schema: {})
    monkeypatch.setattr(package_hmm_handoff, "_check_verification", lambda run, verified: None)
    monkeypatch.setattr(package_hmm_handoff, "_selected_models", lambda *args: {})
    monkeypatch.setattr(package_hmm_handoff, "_model_index", lambda *args: {"fake": True})
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    return artifacts / "handoff", artifacts / "handoff.zip"


def _package(output: Path, tmp_path: Path) -> dict[str, Any]:
    return package_hmm_handoff.package(tmp_path / "run", tmp_path / "evaluator", output)


def _copy_some_files_then(error: BaseException) -> Any:
    def copy(run: Path, output: Path, config: dict[str, Any]) -> None:
        output.mkdir(exist_ok=True)
        (output / "half.txt").write_text("half built", encoding="utf-8")
        raise error

    return copy


def test_a_failure_while_copying_removes_the_half_built_folder(
        paths: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, archive = paths
    full_disk = OSError(errno.ENOSPC, "No space left on device", "half.txt")
    monkeypatch.setattr(package_hmm_handoff, "_copy_bundle_files", _copy_some_files_then(full_disk))
    with pytest.raises(OSError, match="No space left"):
        _package(output, tmp_path)
    assert not output.exists() and not archive.exists()


def test_a_bug_while_copying_also_removes_the_half_built_folder_and_still_propagates(
        paths: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, archive = paths
    monkeypatch.setattr(package_hmm_handoff, "_copy_bundle_files", _copy_some_files_then(KeyError("config")))
    with pytest.raises(KeyError):
        _package(output, tmp_path)
    assert not output.exists() and not archive.exists()


def test_a_failed_archive_check_removes_the_folder_and_the_zip(
        paths: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, archive = paths

    def failing_check(archive_path: Path, hashes: dict[str, str], sums: str) -> None:
        assert archive_path.is_file(), "the zip was written before it was checked"
        raise ValueError("archive member hash mismatch: x")

    monkeypatch.setattr(package_hmm_handoff, "_copy_bundle_files", lambda run, out, config: out.mkdir(exist_ok=True))
    monkeypatch.setattr(package_hmm_handoff, "_verify_archive", failing_check)
    with pytest.raises(ValueError, match="hash mismatch"):
        _package(output, tmp_path)
    assert not output.exists() and not archive.exists()
    assert not archive.with_suffix(".zip.sha256").exists()


def test_a_folder_that_already_exists_is_refused_and_left_untouched(
        paths: tuple[Path, Path], tmp_path: Path) -> None:
    output, _ = paths
    output.mkdir()
    (output / "mine.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        _package(output, tmp_path)
    assert (output / "mine.txt").read_text(encoding="utf-8") == "keep"


def test_a_zip_that_already_exists_is_refused_and_left_untouched(
        paths: tuple[Path, Path], tmp_path: Path) -> None:
    output, archive = paths
    archive.write_bytes(b"keep")
    with pytest.raises(ValueError, match="already exists"):
        _package(output, tmp_path)
    assert archive.read_bytes() == b"keep" and not output.exists()


def test_a_folder_that_appears_after_the_check_is_not_removed(
        paths: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, archive = paths
    real = package_hmm_handoff._bundle_paths

    def checked_then_taken(run: Path, out: Path) -> tuple[Path, Path, Path]:
        result = real(run, out)
        out.mkdir()  # another run wins the race
        (out / "theirs.txt").write_text("keep", encoding="utf-8")
        return result

    monkeypatch.setattr(package_hmm_handoff, "_bundle_paths", checked_then_taken)
    with pytest.raises(FileExistsError):
        _package(output, tmp_path)
    assert (output / "theirs.txt").read_text(encoding="utf-8") == "keep"


def test_a_checksum_file_that_already_exists_is_refused_and_left_untouched(
        paths: tuple[Path, Path], tmp_path: Path) -> None:
    """The cleanup after a failure deletes the checksum file, so a run must never start over one it did not write."""
    output, archive = paths
    checksum = archive.with_suffix(".zip.sha256")
    checksum.write_bytes(b"keep")
    with pytest.raises(ValueError, match="already exists"):
        _package(output, tmp_path)
    assert checksum.read_bytes() == b"keep" and not output.exists() and not archive.exists()


def test_a_cleanup_that_fails_does_not_replace_the_original_error_and_still_tries_every_file(
        paths: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, archive = paths
    checksum = archive.with_suffix(".zip.sha256")

    def copy_then_leave_both_files(run: Path, out: Path, config: dict[str, Any]) -> None:
        out.mkdir(exist_ok=True)
        archive.write_bytes(b"half")
        checksum.write_bytes(b"half")
        raise ValueError("the real problem")

    real_unlink = Path.unlink
    tried: list[str] = []

    def locked_zip(self: Path, missing_ok: bool = False) -> None:
        tried.append(self.name)
        if self == archive:
            raise PermissionError(errno.EACCES, "Permission denied", str(self))
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(package_hmm_handoff, "_copy_bundle_files", copy_then_leave_both_files)
    monkeypatch.setattr(Path, "unlink", locked_zip)
    with pytest.raises(ValueError, match="the real problem"):
        _package(output, tmp_path)
    assert tried == ["handoff.zip", "handoff.zip.sha256"] and not checksum.exists() and not output.exists()
