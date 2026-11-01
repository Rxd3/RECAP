"""models/crf_inspection: the text `recap train-crf inspect` shows for a saved artifact; the library never prints."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest
import sklearn_crfsuite

from punct_restore.models.crf_inspection import RULE, inspect_artifact

HYPERPARAMETERS = {"algorithm": "lbfgs", "c1": 0.1, "c2": 0.2, "max_iterations": 5, "all_possible_transitions": True}
FEATURE_CONFIG = {
    "preset": "basic", "window_size": 1, "use_affixes": True, "use_shape": True, "use_internal_punct": True,
    "use_token_ngrams": False, "use_question_cues": False, "use_clause_cues": False, "use_casing_cues": False,
    "include_utterance_boundaries": True, "use_positional_buckets": False, "version": "v1-basic",
}
BUNDLE = {
    "schema_version": 1, "model_name": "crf_probe", "label_mode": "A", "vocabulary": ["O", "COMMA"],
    "hyperparameters": HYPERPARAMETERS, "feature_config": FEATURE_CONFIG,
    "training_metadata": {"num_sequences": 3}, "environment": {"platform": "test-os"},
    "backend_state": sklearn_crfsuite.CRF(),  # the kind is checked, the model inside is not looked at
}
SIDECAR = {
    "run_id": "run-7", "provenance": {"dataset_release": "rel-1", "policy_version": "pol-1"},
    "code_fingerprint": "abc", "fingerprinted_files": ["a.py", "b.py"], "training_duration_seconds": 1.5,
    "reproducibility_warning": "",
}


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    path = tmp_path / "m.joblib"
    joblib.dump(BUNDLE, path)
    return path


# --- the summary -------------------------------------------------------------------------------------------


def test_the_summary_lists_the_header_and_every_section_in_order(artifact: Path) -> None:
    summary = inspect_artifact(artifact).summary

    assert summary == [
        RULE,
        "RECAP CRF Model Artifact: m.joblib",
        RULE,
        "Schema Version:     1",
        "Model Name:         crf_probe",
        "Label Mode:         A",
        "Vocabulary Size:    2 classes",
        "Vocabulary:         ['O', 'COMMA']",
        "",
        "Hyperparameters:",
        "  algorithm               : lbfgs",
        "  c1                      : 0.1",
        "  c2                      : 0.2",
        "  max_iterations          : 5",
        "  all_possible_transitions: True",
        "",
        "Feature Configuration:",
        "  preset                  : basic",
        "  window_size             : 1",
        "  use_affixes             : True",
        "  use_shape               : True",
        "  use_internal_punct      : True",
        "  use_token_ngrams        : False",
        "  use_question_cues       : False",
        "  use_clause_cues         : False",
        "  use_casing_cues         : False",
        "  include_utterance_boundaries: True",
        "  use_positional_buckets  : False",
        "  version                 : v1-basic",
        "",
        "Training Metadata:",
        "  num_sequences           : 3",
        "",
        "Environment:",
        "  platform                : test-os",
    ]
    assert RULE == "=" * 70


@pytest.mark.parametrize("field", ["vocabulary", "hyperparameters", "feature_config", "training_metadata",
                                   "environment", "backend_state"])
def test_a_bundle_without_a_field_is_refused_naming_it(field: str, tmp_path: Path) -> None:
    path = tmp_path / "bare.joblib"
    joblib.dump({k: v for k, v in BUNDLE.items() if k != field}, path)

    with pytest.raises(ValueError) as caught:
        inspect_artifact(path)

    assert str(caught.value) == f"bare.joblib: corrupt artifact: missing {field}"


@pytest.mark.parametrize(("field", "value", "reason"), [
    ("hyperparameters", None, "hyperparameters is not a JSON object"),
    ("vocabulary", 5, "vocabulary is not a JSON list"),
    ("environment", [], "environment is not a JSON object"),
    ("backend_state", b"x", "backend_state is not a CRF model"),
])
def test_a_field_of_the_wrong_kind_is_refused_naming_it(field: str, value: object, reason: str,
                                                        tmp_path: Path) -> None:
    path = tmp_path / "odd.joblib"
    joblib.dump({**BUNDLE, field: value}, path)

    with pytest.raises(ValueError) as caught:
        inspect_artifact(path)

    assert str(caught.value) == f"odd.joblib: corrupt artifact: {reason}"


# --- the companion manifest --------------------------------------------------------------------------------


def test_without_a_companion_manifest_there_is_nothing_to_add(artifact: Path) -> None:
    report = inspect_artifact(artifact)

    assert report.manifest == [] and report.notes == []


def test_the_manifest_next_to_the_artifact_is_shown(artifact: Path) -> None:
    (artifact.parent / "m.joblib.manifest.json").write_text(json.dumps(SIDECAR), encoding="utf-8")

    report = inspect_artifact(artifact)

    assert report.notes == []
    assert report.manifest == [
        "",
        "Companion Manifest:",
        "  Run ID:                 run-7",
        "  Dataset Release:        rel-1",
        "  Policy Version:         pol-1",
        "  Code Fingerprint:       abc",
        "  Fingerprinted Files:    2 files",
        "  Training Duration:      1.5s",
        "  Reproducibility Note:   ",
    ]


def test_an_unreadable_manifest_is_noted_by_name_and_the_next_candidate_is_used(artifact: Path) -> None:
    (artifact.parent / "m.joblib.manifest.json").write_text("{not json", encoding="utf-8")
    (artifact.parent / "m.manifest.json").write_text(json.dumps(SIDECAR), encoding="utf-8")

    report = inspect_artifact(artifact)

    assert report.notes == ["Note: companion manifest unreadable: m.joblib.manifest.json"]
    assert report.manifest[1:3] == ["Companion Manifest:", "  Run ID:                 run-7"]


@pytest.mark.parametrize("content", [b"[1, 2]", b'{"provenance": 5}', b"\xff\xfe", b"[" * 100_000], ids=[
    "not-an-object", "provenance-not-an-object", "not-utf8", "nested-too-deeply"])
def test_every_kind_of_unreadable_manifest_is_noted(artifact: Path, content: bytes) -> None:
    (artifact.parent / "m.joblib.manifest.json").write_bytes(content)

    report = inspect_artifact(artifact)

    assert report.notes == ["Note: companion manifest unreadable: m.joblib.manifest.json"]
    assert report.manifest == []


# --- unreadable artifacts ----------------------------------------------------------------------------------


def test_a_missing_artifact_is_a_file_not_found_error_naming_the_file_only(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as caught:
        inspect_artifact(tmp_path / "deep" / "ghost.joblib")

    assert str(caught.value) == "ghost.joblib: artifact not found"


def test_a_file_that_is_not_an_artifact_is_a_value_error(tmp_path: Path) -> None:
    junk = tmp_path / "junk.joblib"
    junk.write_bytes(b"this is not a joblib file")
    with pytest.raises(ValueError, match=r"^junk\.joblib: not a readable CRF artifact$"):
        inspect_artifact(junk)

    listing = tmp_path / "list.joblib"
    joblib.dump([1, 2], listing)
    with pytest.raises(ValueError) as caught:
        inspect_artifact(listing)
    assert str(caught.value) == "list.joblib: invalid artifact format: root is list"


def test_the_library_prints_nothing(artifact: Path, capsys: pytest.CaptureFixture[str]) -> None:
    inspect_artifact(artifact)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
