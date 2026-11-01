"""data: build_release_from takes one ReleaseSources, build_release keeps the call form of main; the library says
nothing unless a `log` is passed."""

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


def test_the_package_exports_the_old_and_the_new_build_function_and_release_sources() -> None:
    assert data.build_release is release.build_release and data.build_release_from is release.build_release_from
    assert data.ReleaseSources is release.ReleaseSources
    assert {"build_release", "build_release_from", "ReleaseSources"} <= set(data.__all__)


def test_build_release_from_takes_the_sources_and_only_a_keyword_log() -> None:
    parameters = inspect.signature(release.build_release_from).parameters
    assert list(parameters) == ["sources", "log"]
    assert parameters["log"].kind is inspect.Parameter.KEYWORD_ONLY


def test_build_release_from_builds_from_the_sources_and_reports_progress_to_the_log(tmp_path: Path) -> None:
    lines: list[str] = []
    manifest = release.build_release_from(_sources(tmp_path), log=lines.append)
    assert manifest["release_id"] == "fixture"
    assert (tmp_path / "run" / "processed" / "fixture" / "release_manifest.json").is_file()
    assert lines[0].startswith("loaded train=") and lines[-1].startswith("checks: ")


def test_build_release_from_prints_nothing_when_no_log_is_given(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    release.build_release_from(_sources(tmp_path))
    assert capsys.readouterr().out == ""


# --- build_release keeps the call form of main 72ebfd1 (restructure spec section 2) ----------------


def test_build_release_has_the_parameters_of_main_with_the_last_four_keyword_only() -> None:
    parameters = inspect.signature(release.build_release).parameters
    assert [(p.name, p.kind.name, p.default is not p.empty) for p in parameters.values()] == [
        ("raw_dir", "POSITIONAL_OR_KEYWORD", False), ("processed_dir", "POSITIONAL_OR_KEYWORD", False),
        ("splits_dir", "POSITIONAL_OR_KEYWORD", False), ("release_id", "POSITIONAL_OR_KEYWORD", False),
        ("label_sets_path", "KEYWORD_ONLY", True), ("raw_manifest", "KEYWORD_ONLY", True),
        ("titles_path", "KEYWORD_ONLY", True), ("log", "KEYWORD_ONLY", True)]


def test_build_release_called_as_on_main_builds_and_returns_the_manifest_of_build_release_from(
        tmp_path: Path) -> None:
    sources = _sources(tmp_path, "new")
    expected = release.build_release_from(sources)
    old = tmp_path / "old"
    manifest = release.build_release(sources.raw_dir, str(old / "processed"), str(old / "splits"), "fixture",
                                     raw_manifest=sources.raw_manifest)
    assert manifest == expected
    assert (old / "processed" / "fixture" / "release_manifest.json").is_file()
    assert (old / "splits" / "fixture" / "SHA256SUMS").is_file()


def test_build_release_called_with_every_argument_by_keyword_passes_each_one_on(tmp_path: Path) -> None:
    sources = _sources(tmp_path, "new")
    lines: list[str] = []
    manifest = release.build_release(
        raw_dir=sources.raw_dir, processed_dir=str(tmp_path / "kw" / "processed"),
        splits_dir=str(tmp_path / "kw" / "splits"), release_id="by-keyword", label_sets_path=labeling.LABEL_SETS_PATH,
        raw_manifest=sources.raw_manifest, titles_path=None, log=lines.append)
    assert manifest["release_id"] == "by-keyword"
    assert (tmp_path / "kw" / "processed" / "by-keyword" / "release_manifest.json").is_file()
    assert lines[0].startswith("loaded train=") and lines[-1].startswith("checks: ")


def test_build_release_called_as_on_main_stops_before_any_output_for_an_unapproved_projection(
        tmp_path: Path) -> None:
    unapproved = tmp_path / "label_sets.yaml"
    text = Path(labeling.LABEL_SETS_PATH).read_text(encoding="utf-8")
    unapproved.write_text(text.replace("approved: true", "approved: false"), encoding="utf-8")
    with pytest.raises(ValueError, match="not approved"):
        release.build_release("raw", str(tmp_path / "processed"), str(tmp_path / "splits"), "id",
                              label_sets_path=str(unapproved))
    assert not (tmp_path / "processed").exists()


def test_build_release_prints_nothing_when_no_log_is_given(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sources = _sources(tmp_path)
    release.build_release(sources.raw_dir, sources.processed_dir, sources.splits_dir, "fixture",
                          raw_manifest=sources.raw_manifest)
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
