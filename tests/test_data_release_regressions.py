"""Regression checks for the Windows and data-policy release audit findings."""

import json
from pathlib import Path

import pytest
from test_data_contract import FIXTURE, LS, build

from punct_restore.data import labeling, release, release_outputs, splits, validation


@pytest.mark.parametrize("boundary", ["...?", "?...", "...!", "! ...", ". . . ?", ",..."])
@pytest.mark.parametrize("last", [False, True])
@pytest.mark.parametrize("scheme", ["extended", "native"])
def test_mixed_ellipsis_boundaries_are_flagged(boundary, last, scheme):
    _, flags = labeling.gap_label(boundary, last, scheme)
    assert "ellipsis" in flags
    assert flags.count("mark_cluster") == 1


@pytest.mark.parametrize("last,expected", [(False, "O"), (True, "PERIOD")])
def test_ellipsis_only_policy_is_preserved(last, expected):
    label, flags = labeling.gap_label("...", last, "extended")
    assert label == expected and "mark_cluster" not in flags


@pytest.mark.parametrize(
    "source_split,file_name",
    [("train", "train"), ("dev", "validation"), ("test", "test"), ("test-amb", "test_amb")],
)
def test_unresolved_ellipsis_is_retained_but_excluded_from_selection(
    tmp_path, monkeypatch, source_split, file_name
):
    monkeypatch.setitem(FIXTURE[source_split], "fffffffffff-0", "Really...?")
    processed, split_dir = build(tmp_path, "ellipsis")
    sid = f"sponspeech:{source_split}:fffffffffff-0"
    audit = {
        r["sample_id"]: r
        for r in release.load_records(processed / "mode_A" / (file_name + ".jsonl"), LS)
    }
    assert sid in audit
    with (processed / "audit/unresolved.jsonl").open(encoding="utf-8") as stream:
        reasons = {r["sample_id"]: r for r in map(json.loads, stream)}
    for mode in "ABC":
        assert "mark_cluster" in reasons[sid]["unresolved_by_mode"][mode]
        ids_path = (
            split_dir / f"default_train/mode_{mode}.ids"
            if source_split == "train"
            else split_dir / f"primary_eval/{file_name}_mode_{mode}.ids"
        )
        assert sid not in release.read_ids(ids_path)


@pytest.mark.parametrize("tokens", [["hello"], ["thank", "you"], ["a", "b", "c", "d", "e"]])
def test_short_normalized_inputs_are_detected_without_six_grams(tokens):
    reference = {"group_id": "train-video", "original_text": "ORIGINAL TEXT.", "tokens": tokens}
    candidate = {
        "group_id": "other-video",
        "original_text": "Different casing/punctuation?",
        "tokens": tokens,
    }
    assert not splits.ngram_keys(tokens)
    assert (
        splits.overlap_reason(candidate, splits.Reference([reference]))
        == "identical_model_input_other_video"
    )
    report = splits.overlap_counts([candidate], [reference])
    assert report["identical_texts"] == 0 and report["near_duplicates"] == 0
    assert report["identical_model_inputs"] == 1


def test_matching_words_with_different_token_boundaries_are_not_exact_duplicates():
    reference = {"group_id": "v1", "original_text": "ab c", "tokens": ["ab", "c"]}
    candidate = {"group_id": "v2", "original_text": "a bc", "tokens": ["a", "bc"]}
    assert splits.overlap_reason(candidate, splits.Reference([reference])) is None


def test_release_filters_normalized_duplicates_and_reports_them(tmp_path, monkeypatch):
    monkeypatch.setitem(FIXTURE["train"], "fffffffffff-0", "Hello there.")
    monkeypatch.setitem(FIXTURE["dev"], "ggggggggggg-0", "HELLO THERE?")
    _, split_dir = build(tmp_path, "normalized")
    sid = "sponspeech:dev:ggggggggggg-0"
    excluded = (split_dir / "leakage_controlled/validation_excluded.tsv").read_text(
        encoding="utf-8"
    )
    assert f"{sid}\tidentical_model_input_other_video" in excluded
    report = json.loads((split_dir / "leakage_controlled/report.json").read_text(encoding="utf-8"))
    assert report["overlap"]["official:validation_vs_train"]["identical_model_inputs"] >= 1
    assert (
        report["overlap"]["leakage_controlled:validation_vs_train"]["identical_model_inputs"] == 0
    )
    for mode in "ABC":
        assert sid not in release.read_ids(split_dir / f"primary_eval/validation_mode_{mode}.ids")


def test_release_manifests_and_serialization_are_portable(tmp_path):
    processed, split_dir = build(tmp_path, "portable")
    manifest = json.loads((processed / "release_manifest.json").read_text(encoding="utf-8"))
    for key in [*manifest["files"], *manifest["split_files"]]:
        assert "\\" not in key
    for folder in [processed, split_dir]:
        for path in folder.rglob("*"):
            if path.is_file() and (
                path.suffix in [".json", ".jsonl", ".md", ".tsv", ".ids"]
                or path.name == "SHA256SUMS"
            ):
                assert b"\r\n" not in path.read_bytes(), path


def test_release_id_is_immutable(tmp_path):
    processed, _ = build(tmp_path, "immutable")
    checksum_before = (processed / "SHA256SUMS").read_bytes()
    with pytest.raises(FileExistsError, match="immutable release ID"):
        build(tmp_path, "immutable")
    assert (processed / "SHA256SUMS").read_bytes() == checksum_before


def test_an_unapproved_projection_stops_the_build_with_a_value_error_before_any_output(tmp_path):
    unapproved = tmp_path / "label_sets.yaml"
    text = Path(labeling.LABEL_SETS_PATH).read_text(encoding="utf-8")
    unapproved.write_text(text.replace("approved: true", "approved: false"), encoding="utf-8")
    with pytest.raises(ValueError, match="not approved"):
        sources = release.ReleaseSources(str(tmp_path / "raw"), str(tmp_path / "processed"), str(tmp_path / "splits"),
                                         "fixture", label_sets_path=str(unapproved))
        release.build_release(sources, log=lambda _: None)
    assert not (tmp_path / "processed").exists() and not (tmp_path / "splits").exists()


def test_raw_mismatch_stops_build_before_outputs(tmp_path):
    build(tmp_path, "first")
    source = tmp_path / "raw/train/text/aaaaaaaaaaa-0.txt"
    source.write_text("Changed source.", encoding="utf-8")
    with pytest.raises(ValueError, match="raw manifest mismatch"):
        build(tmp_path, "must-not-publish")
    assert not (tmp_path / "must-not-publish").exists()


def test_failed_validation_cannot_publish_release_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        validation, "validate_file", lambda *args, **kwargs: (0, [(1, "forced validation failure")])
    )
    with pytest.raises(ValueError, match="no release manifest or checksums"):
        build(tmp_path, "invalid")
    for path in (tmp_path / "invalid").rglob("*"):
        assert path.name not in ["release_manifest.json", "SHA256SUMS"]


def test_joint_targets_do_not_silently_truncate():
    with pytest.raises(ValueError):
        labeling.join_labels(["O", "PERIOD"], ["LOWER"])


def test_truncated_view_cannot_publish_a_release(tmp_path, monkeypatch):
    original_write = release_outputs.write_lines

    def truncate_one_view(path, lines):
        if str(path).replace("\\", "/").endswith("mode_B/train.jsonl"):
            lines = list(lines)[:-1]
        original_write(path, lines)

    monkeypatch.setattr(release_outputs, "write_lines", truncate_one_view)
    with pytest.raises(ValueError, match="release validation failed"):
        build(tmp_path, "truncated-view")
    assert not (tmp_path / "truncated-view/processed/fixture/release_manifest.json").exists()


def test_code_fingerprint_hashes_the_data_package_sources():
    """The fingerprint covers exactly the .py files next to release.py (the folder is derived from REPO_ROOT)."""
    import hashlib
    from pathlib import Path

    folder = Path(release.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(folder.glob("*.py"), key=lambda p: p.name):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    assert len(list(folder.glob("*.py"))) > 1 and release.code_fingerprint() == digest.hexdigest()
