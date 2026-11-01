"""punct_restore.cli.prepare_data: main(argv) returns 0 on success and keeps its one-line failure exit."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from punct_restore.cli import prepare_data
from punct_restore.data.release import ReleaseSources


def _fake_extraction(monkeypatch: pytest.MonkeyPatch, result: tuple[list[str], list[str]]) -> list[str]:
    calls: list[str] = []

    def extract_ranges(raw_root: str, log: Callable[[str], None]) -> None:
        calls.append(raw_root)

    monkeypatch.setattr(prepare_data.extraction, "extract_ranges", extract_ranges)
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
    def changed(_root: str, log: Callable[[str], None]) -> None:
        raise prepare_data.extraction.SourceChanged("archive changed: size 1, generation 2")

    monkeypatch.setattr(prepare_data.extraction, "extract_ranges", changed)
    assert prepare_data.main(["extract", "--raw-root", str(tmp_path)]) == 1
    assert _one_error_line(capsys.readouterr().err) == "error: archive changed: size 1, generation 2"


def test_a_bug_in_extraction_is_not_reported_as_bad_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def bug(_root: str, log: Callable[[str], None]) -> None:
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


def test_extract_ranges_progress_lines_are_printed_before_the_verdict(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _fake_extraction(monkeypatch, ([], []))

    def extract_ranges(raw_root: str, log: Callable[[str], None]) -> None:
        log("range train: 2 files")

    monkeypatch.setattr(prepare_data.extraction, "extract_ranges", extract_ranges)
    assert prepare_data.main(["extract", "--raw-root", str(tmp_path)]) == 0
    assert capsys.readouterr().out == "range train: 2 files\nextraction matches MANIFEST.sha256\n"


def test_archive_extraction_progress_line_is_printed_before_the_verdict(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _fake_extraction(monkeypatch, ([], []))
    seen: list[tuple[str, str]] = []

    def extract_archive(archive: str, raw_root: str, log: Callable[[str], None]) -> int:
        seen.append((archive, raw_root))
        log("archive: 3 files")
        return 3

    monkeypatch.setattr(prepare_data.extraction, "extract_archive", extract_archive)
    argv = ["extract", "--raw-root", str(tmp_path), "--archive", "sponspeech.tar.gz"]
    assert prepare_data.main(argv) == 0
    assert seen == [("sponspeech.tar.gz", str(tmp_path))]
    assert capsys.readouterr().out == "archive: 3 files\nextraction matches MANIFEST.sha256\n"


def test_build_hands_one_release_sources_and_its_progress_lines_to_the_user(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "MANIFEST.sha256").write_text("", encoding="utf-8")
    (tmp_path / "video_metadata_oembed.jsonl").write_text("", encoding="utf-8")
    received: list[ReleaseSources] = []

    def build_release(sources: ReleaseSources, *, log: Callable[[str], None]) -> dict:
        received.append(sources)
        log("loaded train=4")
        return {"records": {"source": 9, "released_per_view": 8, "rejected": 1}}

    monkeypatch.setattr(prepare_data, "build_release", build_release)
    argv = ["build", "--raw-root", str(tmp_path), "--release-id", "r1", "--processed-dir", str(tmp_path / "p"),
            "--splits-dir", str(tmp_path / "s")]
    assert prepare_data.main(argv) == 0
    assert received == [ReleaseSources(
        raw_dir=str(tmp_path / "sponspeech_text"), processed_dir=str(tmp_path / "p"), splits_dir=str(tmp_path / "s"),
        release_id="r1", raw_manifest=str(tmp_path / "MANIFEST.sha256"),
        titles_path=str(tmp_path / "video_metadata_oembed.jsonl"))]
    assert capsys.readouterr().out == 'loaded train=4\n{"source": 9, "released_per_view": 8, "rejected": 1}\n'
