"""models/crf_training: optimizer diagnostics, the training provenance and the sidecar manifest of a trained model."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from punct_restore.features.crf_features import FeatureConfig
from punct_restore.hashing import sha256_file
from punct_restore.models import crf_training
from punct_restore.models.crf import CRFModel
from punct_restore.models.crf_training import (
    TrainingRun,
    TrainingSource,
    build_sidecar_manifest,
    build_training_provenance,
    collect_training_diagnostics,
)

RECORDS = [
    {"sample_id": "m1", "tokens": ["hello", "everyone"], "punct_labels": ["O", "PERIOD"]},
    {"sample_id": "m2", "tokens": ["yes", "we", "can"], "punct_labels": ["COMMA", "O", "PERIOD"]},
]
MANIFEST_KEYS = [
    "schema_version", "run_id", "label_mode", "model_name", "vocabulary", "feature_preset", "feature_preset_version",
    "feature_config", "hyperparameters", "training_diagnostics", "memory_diagnostics", "provenance",
    "training_command", "training_duration_seconds", "artifact", "backend", "environment", "code_fingerprint",
    "fingerprinted_files", "reproducibility_warning", "created_at",
]


# --- collect_training_diagnostics --------------------------------------------------------------------------


def test_diagnostics_come_from_the_public_training_log_of_the_model() -> None:
    log = SimpleNamespace(iterations=[{}, {}, {}], lines=["start", " L-BFGS terminated: converged ", "tail"])
    model = SimpleNamespace(training_log=log)  # no private _backend at all: the function may not need one

    assert collect_training_diagnostics(model) == {
        "num_iterations": 3, "termination_status": "L-BFGS terminated: converged"}


def test_diagnostics_are_empty_for_a_model_without_a_training_log() -> None:
    assert collect_training_diagnostics(SimpleNamespace(training_log=None)) == {}
    assert collect_training_diagnostics(CRFModel()) == {}


def test_diagnostics_of_a_fitted_model_count_its_iterations() -> None:
    model = CRFModel(max_iterations=6).fit(RECORDS)

    assert collect_training_diagnostics(model)["num_iterations"] == len(model.training_log.iterations)


# --- build_training_provenance -----------------------------------------------------------------------------


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    data, ids = tmp_path / "train.jsonl", tmp_path / "train.ids"
    data.write_text('{"sample_id": "m1"}\n', encoding="utf-8")
    ids.write_text("m1\n", encoding="utf-8")
    return data, ids


def test_provenance_describes_the_labels_and_the_input_files(tmp_path: Path) -> None:
    data, ids = _inputs(tmp_path)
    source = TrainingSource(input_path=data, ids_path=ids, num_sequences=2, num_tokens=5,
                            dataset_release="rel-1", policy_version="pol-1", run_id="run-7")

    provenance = build_training_provenance(source)

    assert list(provenance) == ["dataset_release", "policy_version", "run_id", "input_file", "input_sha256",
                                "ids_file", "ids_sha256", "selected_sequences", "selected_tokens"]
    assert provenance == {
        "dataset_release": "rel-1", "policy_version": "pol-1", "run_id": "run-7",
        "input_file": data.as_posix(), "input_sha256": sha256_file(data),
        "ids_file": ids.as_posix(), "ids_sha256": sha256_file(ids),
        "selected_sequences": 2, "selected_tokens": 5}


def test_provenance_names_the_labels_it_was_not_given(tmp_path: Path) -> None:
    data, ids = _inputs(tmp_path)

    provenance = build_training_provenance(TrainingSource(input_path=data, ids_path=ids, num_sequences=1, num_tokens=2))

    assert (provenance["dataset_release"], provenance["policy_version"], provenance["run_id"]) == (
        "unknown", "unknown", "unnamed_run")


# --- build_sidecar_manifest --------------------------------------------------------------------------------


def _run(tmp_path: Path) -> TrainingRun:
    model = CRFModel(max_iterations=5).fit(RECORDS)
    artifact = tmp_path / "m.joblib"
    model.save_model(artifact)
    return TrainingRun(
        run_id="run-7", mode="A", model=model, preset="basic", feature_config=FeatureConfig.basic(),
        hyperparameters={"c1": 0.1}, training_diagnostics={"num_iterations": 5}, memory_diagnostics={},
        provenance={"run_id": "run-7"}, training_command=["recap", "train-crf", "train"], duration=1.23456,
        output_path=artifact, artifact_hash=sha256_file(artifact))


def test_the_sidecar_manifest_records_the_run_the_artifact_and_the_code(tmp_path: Path) -> None:
    run = _run(tmp_path)

    manifest = build_sidecar_manifest(run)

    assert list(manifest) == MANIFEST_KEYS
    assert (manifest["run_id"], manifest["label_mode"], manifest["feature_preset"]) == ("run-7", "A", "basic")
    assert manifest["vocabulary"] == run.model.vocabulary and manifest["model_name"] == run.model.model_name
    assert manifest["hyperparameters"] == {"c1": 0.1} and manifest["provenance"] == {"run_id": "run-7"}
    assert manifest["training_command"] == ["recap", "train-crf", "train"]
    assert manifest["training_duration_seconds"] == 1.235
    assert manifest["artifact"] == {"path": run.output_path.as_posix(), "sha256": run.artifact_hash,
                                    "size_bytes": run.output_path.stat().st_size}
    assert manifest["code_fingerprint"] == crf_training.compute_code_fingerprint()
    assert manifest["fingerprinted_files"] == crf_training.get_fingerprinted_files()


def test_the_run_and_the_source_cannot_be_changed_after_they_are_built(tmp_path: Path) -> None:
    data, ids = _inputs(tmp_path)
    source = TrainingSource(input_path=data, ids_path=ids, num_sequences=1, num_tokens=2)

    with pytest.raises(dataclasses.FrozenInstanceError):
        source.run_id = "later"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        _run(tmp_path).mode = "B"  # type: ignore[misc]


def test_a_run_without_memory_figures_records_null_and_why(tmp_path: Path) -> None:
    run = dataclasses.replace(_run(tmp_path), memory_diagnostics=None, memory_unavailable="not available on linux")

    manifest = build_sidecar_manifest(run)

    assert manifest["memory_diagnostics"] is None
    assert manifest["memory_diagnostics_unavailable"] == "not available on linux"
    assert set(manifest) == {*MANIFEST_KEYS, "memory_diagnostics_unavailable"}


def test_a_run_with_memory_figures_has_no_reason_key(tmp_path: Path) -> None:
    figures = {"peak_working_set_mb": 1.5}
    manifest = build_sidecar_manifest(dataclasses.replace(_run(tmp_path), memory_diagnostics=figures))

    assert manifest["memory_diagnostics"] == figures
    assert list(manifest) == MANIFEST_KEYS
