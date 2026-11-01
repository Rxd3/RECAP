"""data/: helpers return new values instead of changing their arguments; release writers share one write_lines."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest
from test_data_contract import LS, build

from punct_restore.data import labeling, preprocessing, release, release_outputs
from punct_restore.data.release_checks import ReleaseReadback, verify_release
from punct_restore.data.release_constants import RELEASE_FILES
from punct_restore.data.release_manifest import ManifestInputs, build_manifest
from punct_restore.data.release_stats import StatsInputs, build_stats

# --- _flag_source_artifacts ---------------------------------------------------------------------------------


def test_source_artifact_flags_are_returned_as_a_new_set_and_the_argument_is_left_alone() -> None:
    flags = {"uncertain_words"}

    result = preprocessing._flag_source_artifacts("a *starred* word, Hebdom?konta and ?what", flags)

    assert flags == {"uncertain_words"}
    assert result == {"uncertain_words", "asterisk_text", "inverted_mark_folded", "question_inside_word"}
    assert result is not flags


def test_clean_text_gets_the_same_flags_back_as_a_copy() -> None:
    flags = {"kept_bracket_or_paren"}

    result = preprocessing._flag_source_artifacts("plain words only", flags)

    assert result == flags and result is not flags


# --- write_lines --------------------------------------------------------------------------------------------


def test_write_lines_creates_the_folder_and_writes_the_lines_with_lf_endings(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "er" / "out.tsv"

    release_outputs.write_lines(str(target), (f"{i}\n" for i in range(3)))

    assert target.read_bytes() == b"0\n1\n2\n"


# --- _preflight ---------------------------------------------------------------------------------------------


def test_preflight_returns_what_the_build_needs_and_nothing_else(tmp_path: Path) -> None:
    sources = release.ReleaseSources("raw", str(tmp_path / "processed"), str(tmp_path / "splits"), "rel")

    result = release._preflight(sources)

    assert result._fields == ("titles", "label_sets", "out", "split_out")
    assert result.titles is None and result.label_sets["projection"]["approved"]
    assert Path(result.out) == tmp_path / "processed" / "rel" and Path(result.split_out) == tmp_path / "splits" / "rel"


# --- verify_release -----------------------------------------------------------------------------------------


def _readback(tmp_path: Path, *, raw_manifest: bool) -> tuple[ReleaseReadback, dict[str, dict[str, int]]]:
    processed, _ = build(tmp_path, "built")
    manifest = json.loads((processed / "release_manifest.json").read_text(encoding="utf-8"))
    views = {path: {"records": info["records"]} for path, info in manifest["files"].items() if "records" in info}
    raw = str(tmp_path / "raw" / "MANIFEST.sha256") if raw_manifest else None
    readback = ReleaseReadback(out=str(processed), files=views, results=[], kept=[], label_sets=LS, raw_manifest=raw)
    return readback, views


def test_verify_release_returns_the_validity_of_every_view_and_leaves_the_files_argument_alone(
        tmp_path: Path) -> None:
    readback, views = _readback(tmp_path, raw_manifest=False)
    before = copy.deepcopy(views)

    verified = verify_release(readback)

    assert views == before and all("valid" not in info for info in views.values())
    assert set(verified.files) == set(views) and len(views) == 16
    assert all(info["valid"] == views[path]["records"] for path, info in verified.files.items())
    assert verified.checks["file_errors"] == {} and verified.checks["view_mismatches"] == {}


@pytest.mark.parametrize(("with_raw", "before"), [(True, []), (False, None)])
def test_the_raw_manifest_check_before_the_build_is_an_empty_list_or_none(
        tmp_path: Path, with_raw: bool, before: list | None) -> None:
    readback, _ = _readback(tmp_path, raw_manifest=with_raw)

    checks = verify_release(readback).checks

    assert checks["raw_mismatch_before"] == before and checks["raw_mismatch_after"] == before


def test_a_manifest_lists_every_view_as_valid_for_all_its_records(tmp_path: Path) -> None:
    processed, _ = build(tmp_path, "run1")
    manifest = json.loads((processed / "release_manifest.json").read_text(encoding="utf-8"))

    views = {path: info for path, info in manifest["files"].items() if "records" in info}

    assert len(views) == 16
    assert all(info["valid"] == info["records"] for info in views.values())


# --- the dataclasses that replace long parameter lists ------------------------------------------------------


def test_the_manifest_is_built_from_one_frozen_set_of_inputs() -> None:
    inputs = ManifestInputs(release_id="rel-1", label_sets_path=labeling.LABEL_SETS_PATH, raw_manifest=None,
                            titles_used=True, code_sha256=lambda: "abc123", source_count=10, released_count=7)

    manifest = build_manifest(inputs)

    assert (manifest["release_id"], manifest["code_sha256"], manifest["titles_used_for_tts_check"]) == (
        "rel-1", "abc123", True)
    assert manifest["records"] == {"source": 10, "released_per_view": 7, "rejected": 3}
    assert manifest["raw_manifest_sha256"] is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        inputs.release_id = "other"  # type: ignore[misc]


def test_the_stats_are_built_from_one_frozen_set_of_inputs() -> None:
    empty = {name: [] for name in RELEASE_FILES}
    inputs = StatsInputs(results=[], kept=[], by_file=empty, default=empty, controlled_report={}, overlap={},
                         label_sets=LS)

    stats = build_stats(inputs)

    assert stats["leakage_controlled"] == {} and stats["overlap"] == {}
    assert stats["subset_sizes"]["A"] == {"default_train": 0, "primary_validation": 0, "primary_test": 0,
                                          "primary_test_amb": 0}
    assert set(stats["labels"]) == set(RELEASE_FILES) and "checks" not in stats
    with pytest.raises(dataclasses.FrozenInstanceError):
        inputs.kept = [{}]  # type: ignore[misc]
