"""Validated file selection and provenance for Member 2's HMM commands."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from punct_restore.hashing import sha256_file
from punct_restore.models.hmm import HMMModel
from punct_restore.models.hmm_records import SelectedRecords
from punct_restore.paths import REPO_ROOT as ROOT
from punct_restore.prediction_io import parse_json_object, read_ids, write_new_file


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def file_identity(path: Path) -> dict[str, str]:
    return {"path": portable_path(path), "sha256": sha256_file(path)}


def code_identity(root: Path = ROOT) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def read_json_object(path: Path) -> dict[str, Any]:
    """The JSON object a UTF-8 file holds.

    Raises:
        OSError: If the file cannot be read.
        ValueError: One line that starts with the file name only: not valid UTF-8, not valid JSON (also when it
            is nested too deeply) or not a JSON object.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name}: not valid UTF-8") from exc
    return parse_json_object(text, path.name)


def checked_release_file(
    path: Path, root: Path, entries: dict[str, Any], *, expected: str | None = None
) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"{path.name}: file must be inside its declared release directory"
        ) from exc
    if relative not in entries or (expected is not None and relative != expected):
        raise ValueError(f"{relative}: not the expected release file {expected or ''}")
    entry = entries[relative]
    if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
        raise ValueError(f"{relative}: release checksum mismatch")
    return relative


def _check_split_manifest(split_manifest: Path, manifest: dict[str, Any]) -> None:
    """Authenticate the supplied split checksum index against the release manifest."""
    lines = split_manifest.read_text(encoding="utf-8").splitlines()
    expected_sums = {name: entry["sha256"] for name, entry in manifest["split_files"].items()}
    actual_sums = {}
    for line in lines:
        digest, name = line.split("  ", 1)
        if name in actual_sums:
            raise ValueError("duplicate path in split checksum manifest")
        actual_sums[name] = digest
    if actual_sums != expected_sums:
        raise ValueError("split checksum manifest does not match release")


def _check_prediction_files(
    manifest: dict[str, Any],
    processed: Path,
    splits: Path,
    mode: str,
    prediction: Path,
    prediction_ids: Path | None,
) -> None:
    """Check the prediction input (and its ID selection) against the release."""
    relative = checked_release_file(prediction, processed, manifest["files"])
    if not relative.startswith(f"mode_{mode}/"):
        raise ValueError("prediction input mode does not match model")
    if prediction_ids is not None:
        name = Path(relative).stem
        checked_release_file(
            prediction_ids,
            splits,
            manifest["split_files"],
            expected=f"primary_eval/{name}_mode_{mode}.ids",
        )


@dataclass(frozen=True)
class ReleaseInputs:
    """The files and identities a training command was given, to be checked against the data release."""

    train: Path
    train_ids: Path
    split_manifest: Path
    label_config: Path
    mode: str
    release_id: str
    policy_version: str
    prediction: Path | None = None
    prediction_ids: Path | None = None


_MANIFEST_TEXT_FIELDS = ("release_id", "policy_version", "label_sets_sha256")
_MANIFEST_FILE_TABLES = ("files", "split_files")


def read_release_manifest(path: Path) -> dict[str, Any]:
    """A release manifest with the fields the HMM commands read; a wrong shape is a ValueError naming the file."""
    manifest = read_json_object(path)
    for field in _MANIFEST_TEXT_FIELDS:
        if not isinstance(manifest.get(field), str):
            raise ValueError(f"{path.name}: release manifest has no text field {field}")
    for table in _MANIFEST_FILE_TABLES:
        entries = manifest.get(table)
        if not isinstance(entries, dict) or not all(
            isinstance(entry, dict) and isinstance(entry.get("sha256"), str) and isinstance(entry.get("bytes"), int)
            for entry in entries.values()
        ):
            raise ValueError(f"{path.name}: release manifest {table} must map names to sha256 and bytes")
    source = manifest.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("revision"), str):
        raise ValueError(f"{path.name}: release manifest has no source revision")
    return manifest


def verify_training_release(manifest_path: Path, inputs: ReleaseInputs) -> dict[str, Any]:
    """Check the command's inputs against the release manifest; return the manifest."""
    manifest = read_release_manifest(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("release_id") != inputs.release_id
        or manifest.get("policy_version") != inputs.policy_version
    ):
        raise ValueError("release identity/schema/policy does not match command")
    if sha256_file(inputs.label_config) != manifest["label_sets_sha256"]:
        raise ValueError("label configuration does not match the data release")
    processed, splits, mode = manifest_path.parent, inputs.split_manifest.parent, inputs.mode
    checked_release_file(inputs.train, processed, manifest["files"], expected=f"mode_{mode}/train.jsonl")
    checked_release_file(
        inputs.train_ids, splits, manifest["split_files"], expected=f"default_train/mode_{mode}.ids"
    )
    _check_split_manifest(inputs.split_manifest, manifest)
    if inputs.prediction is not None:
        _check_prediction_files(manifest, processed, splits, mode, inputs.prediction, inputs.prediction_ids)
    return manifest


def write_json(path: Path, value: Any) -> None:
    write_new_file(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def _check_frozen_selection(selection_record: Path, model: Any, inputs: dict[str, Any]) -> None:
    """Require a valid frozen selection that names this artifact."""
    selection = read_json_object(selection_record)
    modes = selection.get("modes")
    winner = modes.get(model.label_mode) if isinstance(modes, dict) else None
    if (
        selection.get("used_test_labels") is not False
        or selection.get("training_includes_validation") is not False
        or not isinstance(winner, dict)
        or winner.get("model_sha256") != inputs["model"]["sha256"]
    ):
        raise ValueError(f"{selection_record.name}: artifact does not match a valid frozen selection")


def _check_release_compatibility(
    release_manifest: Path, input_path: Path, ids_path: Path | None, model: Any, inputs: dict[str, Any]
) -> None:
    """Require the artifact, input file and ID selection to belong to the release."""
    release = read_release_manifest(release_manifest)
    meta = model.metadata
    if (
        release["release_id"] != meta.get("dataset_release_id")
        or release["policy_version"] != meta.get("label_policy_version")
        or release["label_sets_sha256"] != meta.get("label_configuration", {}).get("sha256")
    ):
        raise ValueError("artifact is incompatible with this data release")
    relative = checked_release_file(input_path, release_manifest.parent, release["files"])
    if not relative.startswith(f"mode_{model.label_mode}/"):
        raise ValueError("prediction input mode does not match the artifact")
    if ids_path:
        # Split selections are authenticated through their content, even if copied.
        allowed = {
            entry["sha256"]
            for name, entry in release["split_files"].items()
            if name.endswith(f"_mode_{model.label_mode}.ids")
            or name == f"default_train/mode_{model.label_mode}.ids"
        }
        if inputs["ids"]["sha256"] not in allowed:
            raise ValueError("prediction IDs are not an approved release selection")


def _check_inputs_unchanged(inputs: dict[str, Any], paths: list[tuple[str, Path | None]]) -> None:
    """Reject the run if any input file changed while inference ran."""
    for key, path in paths:
        if path is not None and sha256_file(path) != inputs[key]["sha256"]:
            raise ValueError(f"{path.name}: input changed during inference")


def _write_predictions(output_dir: Path, predictions: list[dict[str, Any]]) -> Path:
    """Create output_dir and write predictions.jsonl; return its path."""
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return predictions_path


@dataclass(frozen=True)
class PredictionOptions:
    """What binds a prediction run to a release or a frozen selection, and the command line to record."""

    ids_path: Path | None = None
    release_manifest: Path | None = None
    selection_record: Path | None = None
    command: list[str] | None = None


@dataclass(frozen=True)
class _RunFacts:
    """What one prediction run read and measured, to be written as its manifest."""

    model: HMMModel
    inputs: dict[str, Any]
    started_at: str
    predictions_path: Path
    counts: dict[str, int]
    num_sequences: int
    load_seconds: float
    prediction_seconds: float
    command: list[str] | None


def _run_manifest(run: _RunFacts) -> dict[str, Any]:
    """Build the prediction run manifest."""
    return {
        "schema_version": 1,
        "model_name": "hmm",
        "label_mode": run.model.label_mode,
        "run_id": run.model.metadata.get("run_id"),
        "ordered_labels": list(run.model.labels),
        "started_at_utc": run.started_at,
        "inputs": run.inputs,
        "model_sha256": run.inputs["model"]["sha256"],
        "predictions_sha256": sha256_file(run.predictions_path),
        "selection_counts": run.counts,
        "num_sequences": run.num_sequences,
        "num_tokens": run.counts["selected_tokens"],
        "model_load_seconds": run.load_seconds,
        "prediction_seconds_including_input": run.prediction_seconds,
        "timing_scope": "JSONL reading, observable-input validation, selection, predict; excludes writes",
        "training_provenance": run.model.metadata,
        "code": code_identity(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "command": run.command,
    }


def _input_identities(model_path: Path, input_path: Path, options: PredictionOptions) -> dict[str, Any]:
    """The identity of every file a prediction run reads, keyed by its role."""
    inputs = {"model": file_identity(model_path), "input": file_identity(input_path)}
    for key, path in [
        ("ids", options.ids_path),
        ("release_manifest", options.release_manifest),
        ("selection_record", options.selection_record),
    ]:
        if path:
            inputs[key] = file_identity(path)
    return inputs


def predict_run(
    model_path: Path, input_path: Path, output_dir: Path, options: PredictionOptions | None = None
) -> dict[str, Any]:
    """Load an artifact and export predictions without refitting or reading gold fields."""
    options = options or PredictionOptions()
    ids_path, release_manifest, selection_record = options.ids_path, options.release_manifest, options.selection_record
    if output_dir.exists():
        raise ValueError("output-dir already exists; choose a new prediction run")
    inputs = _input_identities(model_path, input_path, options)
    started_at = datetime.now(UTC).isoformat()
    start = perf_counter()
    model = HMMModel.load_model(model_path)
    load_seconds = perf_counter() - start
    if selection_record:
        _check_frozen_selection(selection_record, model, inputs)
    if release_manifest:
        _check_release_compatibility(release_manifest, input_path, ids_path, model, inputs)
    ids = tuple(read_ids(ids_path, strict=True)) if ids_path else None
    start = perf_counter()
    selection = SelectedRecords(input_path, ids)
    predictions = model.predict(selection)
    prediction_seconds = perf_counter() - start
    checked = [
        ("model", model_path),
        ("input", input_path),
        ("ids", ids_path),
        ("release_manifest", release_manifest),
        ("selection_record", selection_record),
    ]
    _check_inputs_unchanged(inputs, checked)
    predictions_path = _write_predictions(output_dir, predictions)
    facts = _RunFacts(
        model=model,
        inputs=inputs,
        started_at=started_at,
        predictions_path=predictions_path,
        counts=selection.counts,
        num_sequences=len(predictions),
        load_seconds=load_seconds,
        prediction_seconds=prediction_seconds,
        command=options.command,
    )
    manifest = _run_manifest(facts)
    write_json(output_dir / "predictions.manifest.json", manifest)
    return manifest
