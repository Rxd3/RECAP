"""data: build_release takes one ReleaseSources; the library says nothing unless a `log` is passed."""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import pytest
from test_data_contract import fake_fetch, make_archive, pin, write_fixture

from punct_restore import data
from punct_restore.data import labeling, release


def _sources(tmp_path: Path, name: str = "run", **over: object) -> release.ReleaseSources:
    raw = tmp_path / "raw"
    manifest = write_fixture(raw) if not raw.exists() else raw / "MANIFEST.sha256"
    fields: dict[str, object] = {"raw_dir": str(raw), "processed_dir": str(tmp_path / name / "processed"),
                                 "splits_dir": str(tmp_path / name / "splits"), "release_id": "fixture",
                                 "raw_manifest": str(manifest)}
    return release.ReleaseSources(**(fields | over))


def test_release_sources_has_the_documented_defaults(tmp_path: Path) -> None:
    sources = release.ReleaseSources("raw", "processed", "splits", "id")
    assert (sources.raw_dir, sources.processed_dir, sources.splits_dir, sources.release_id) == (
        "raw", "processed", "splits", "id")
    assert sources.label_sets_path == labeling.LABEL_SETS_PATH
    assert sources.raw_manifest is None and sources.titles_path is None


def test_release_sources_is_frozen() -> None:
    sources = release.ReleaseSources("raw", "processed", "splits", "id")
    with pytest.raises(dataclasses.FrozenInstanceError):
        sources.release_id = "other"  # type: ignore[misc]


def test_the_package_exports_both_build_release_and_release_sources() -> None:
    assert data.build_release is release.build_release and data.ReleaseSources is release.ReleaseSources
    assert {"build_release", "ReleaseSources"} <= set(data.__all__)


def test_build_release_takes_the_sources_and_only_a_keyword_log() -> None:
    parameters = inspect.signature(release.build_release).parameters
    assert list(parameters) == ["sources", "log"]
    assert parameters["log"].kind is inspect.Parameter.KEYWORD_ONLY


def test_build_release_builds_from_the_sources_and_reports_progress_to_the_log(tmp_path: Path) -> None:
    lines: list[str] = []
    manifest = release.build_release(_sources(tmp_path), log=lines.append)
    assert manifest["release_id"] == "fixture"
    assert (tmp_path / "run" / "processed" / "fixture" / "release_manifest.json").is_file()
    assert lines[0].startswith("loaded train=") and lines[-1].startswith("checks: ")


def test_build_release_prints_nothing_when_no_log_is_given(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    release.build_release(_sources(tmp_path))
    assert capsys.readouterr().out == ""


def test_extract_archive_prints_nothing_when_no_log_is_given(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    gz, _ = make_archive(tmp_path)
    ex = pin(monkeypatch, gz)
    archive = tmp_path / "sponspeech.tar.gz"
    archive.write_bytes(gz)
    out = tmp_path / "raw"
    out.mkdir()
    monkeypatch.setattr(ex, "ARCHIVE_BYTES", len(gz))
    assert ex.extract_archive(str(archive), str(out)) == 3
    assert capsys.readouterr().out == ""


def test_extract_ranges_prints_nothing_when_no_log_is_given(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    gz, _ = make_archive(tmp_path)
    ex = pin(monkeypatch, gz)
    out = tmp_path / "raw"
    out.mkdir()
    assert ex.extract_ranges(str(out), fetch=fake_fetch(gz)) == 3
    assert capsys.readouterr().out == ""
