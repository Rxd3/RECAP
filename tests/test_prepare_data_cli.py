"""punct_restore.cli.prepare_data: main(argv) returns 0 on success and keeps its one-line failure exit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from punct_restore.cli import prepare_data


def _fake_extraction(monkeypatch: pytest.MonkeyPatch, result: tuple[list[str], list[str]]) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(prepare_data.extraction, "extract_ranges", calls.append)
    monkeypatch.setattr(prepare_data.extraction, "verify_extraction", lambda root, manifest: result)
    return calls


def test_extract_passes_argv_through_and_returns_zero(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls = _fake_extraction(monkeypatch, ([], []))
    assert prepare_data.main(["extract", "--raw-root", str(tmp_path)]) == 0
    assert calls == [str(tmp_path)]
    assert capsys.readouterr().out == "extraction matches MANIFEST.sha256\n"


def test_a_mismatched_extraction_still_exits_with_its_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_extraction(monkeypatch, (["a.txt"], []))
    with pytest.raises(SystemExit) as stop:
        prepare_data.main(["extract", "--raw-root", str(tmp_path)])
    assert str(stop.value.code).startswith("extraction does not match MANIFEST.sha256: 1 missing/different, 0 extra")


def _one_error_line(err: str) -> str:
    lines = err.splitlines()
    assert len(lines) == 1 and lines[0].startswith("error: "), err
    return lines[0]


def test_a_missing_archive_is_one_error_line_naming_the_file_only(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "deep" / "missing.tar.gz"
    assert prepare_data.main(["extract", "--raw-root", str(tmp_path), "--archive", str(missing)]) == 1
    captured = capsys.readouterr()
    line = _one_error_line(captured.err)
    assert line.startswith("error: missing.tar.gz: ")
    assert str(tmp_path) not in line and captured.out == ""


def test_an_archive_that_is_a_folder_is_one_error_line_naming_the_folder_only(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = tmp_path / "adir"
    folder.mkdir()
    assert prepare_data.main(["extract", "--raw-root", str(tmp_path), "--archive", str(folder)]) == 1
    line = _one_error_line(capsys.readouterr().err)
    assert line.startswith("error: adir: ") and str(tmp_path) not in line


def test_a_changed_source_is_one_error_line(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def changed(_root: str) -> None:
        raise prepare_data.extraction.SourceChanged("archive changed: size 1, generation 2")

    monkeypatch.setattr(prepare_data.extraction, "extract_ranges", changed)
    assert prepare_data.main(["extract", "--raw-root", str(tmp_path)]) == 1
    assert _one_error_line(capsys.readouterr().err) == "error: archive changed: size 1, generation 2"


def test_a_bug_in_extraction_is_not_reported_as_bad_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def bug(_root: str) -> None:
        raise KeyError("oops")

    monkeypatch.setattr(prepare_data.extraction, "extract_ranges", bug)
    with pytest.raises(KeyError):
        prepare_data.main(["extract", "--raw-root", str(tmp_path)])


def test_build_without_the_manifest_exits_1_with_one_line_naming_the_file_only(tmp_path: Path) -> None:
    done = subprocess.run(
        [sys.executable, "-m", "punct_restore.cli.prepare_data", "build", "--raw-root", str(tmp_path / "raw"),
         "--release-id", "r", "--processed-dir", str(tmp_path / "p"), "--splits-dir", str(tmp_path / "s")],
        capture_output=True, text=True, check=False)
    assert done.returncode == 1
    assert done.stdout == "" and len(done.stderr.splitlines()) == 1
    assert done.stderr.startswith("missing MANIFEST.sha256") and str(tmp_path) not in done.stderr
    assert "Traceback" not in done.stderr
