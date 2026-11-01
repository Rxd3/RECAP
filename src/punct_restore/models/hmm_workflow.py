"""Validated file selection and provenance for Member 2's HMM commands."""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from punct_restore.data.labeling import load_label_sets
from punct_restore.data.validation import check_model_input, validate_record
from punct_restore.hashing import sha256_file
from punct_restore.paths import REPO_ROOT as ROOT
from punct_restore.prediction_io import read_ids, write_new_file


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path.name}:{number}: blank JSONL record")
            try:
                record = json.loads(line)
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f"{path.name}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path.name}:{number}: record must be a JSON object")
            yield record


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


def _check_record_basics(record: dict[str, Any], path: Path, seen: set[str]) -> tuple[str, list[Any]]:
    """Validate schema_version, sample_id and tokens; remember the sample_id."""
    sid, tokens = record.get("sample_id"), record.get("tokens")
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise ValueError(f"{path.name}: unsupported schema_version")
    if not isinstance(sid, str) or not sid.strip() or sid in seen:
        raise ValueError(f"{path.name}: invalid or duplicate sample_id {sid!r}")
    seen.add(sid)
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(not isinstance(word, str) or not word.strip() for word in tokens)
    ):
        raise ValueError(f"{sid}: tokens must be a nonempty list of nonempty strings")
    return sid, tokens


def _check_training_record(
    record: dict[str, Any],
    sid: str,
    tokens: list[Any],
    label_sets: Any,
    training_mode: str,
    policy_version: str | None,
    source_revision: str | None,
) -> None:
    """Validate a training row, including the split, mode and release policy."""
    errors = validate_record(record, label_sets) + check_model_input(tokens)
    if errors:
        raise ValueError(f"{sid}: {errors[0]}")
    if record["split"] != "train" or record["label_mode"] != training_mode:
        raise ValueError(f"{sid}: training input must contain only train/{training_mode}")
    if policy_version is not None:
        meta = record.get("metadata", {})
        if (
            meta.get("policy_version") != policy_version
            or meta.get("source_revision") != source_revision
            or meta.get("annotation_view") != "canonical"
        ):
            raise ValueError(f"{sid}: record does not match the canonical release policy")


def _check_selection_complete(
    path: Path, wanted: set[str] | None, selected: set[str], totals: dict[str, int]
) -> None:
    """Reject missing selection IDs and empty selections."""
    if wanted is not None and selected != wanted:
        missing = sorted(wanted - selected)
        raise ValueError(f"{path.name}: {len(missing)} selection IDs missing, first: {missing[:3]}")
    if not totals["selected_records"]:
        raise ValueError(f"{path.name}: selection is empty")


def selected_records(
    path: Path,
    ids: tuple[str, ...] | None,
    *,
    training_mode: str | None = None,
    policy_version: str | None = None,
    source_revision: str | None = None,
    counts: dict[str, int] | None = None,
) -> Iterator[dict[str, Any]]:
    """Keep source-file order and reject missing IDs, even after the last yielded row.

    Training validates every row, including excluded rows, to catch mixed splits.
    Prediction checks observable fields only and removes gold and source metadata.
    """
    wanted = None if ids is None else set(ids)
    seen: set[str] = set()
    selected: set[str] = set()
    totals = counts if counts is not None else {}
    totals.update(input_records=0, selected_records=0, selected_tokens=0)
    label_sets = load_label_sets() if training_mode is not None else None
    for record in read_jsonl(path):
        sid, tokens = _check_record_basics(record, path, seen)
        if training_mode is not None:
            _check_training_record(
                record, sid, tokens, label_sets, training_mode, policy_version, source_revision
            )
        totals["input_records"] += 1
        if wanted is not None and sid not in wanted:
            continue
        selected.add(sid)
        totals["selected_records"] += 1
        totals["selected_tokens"] += len(tokens)
        yield (
            record
            if training_mode is not None
            else {"schema_version": 1, "sample_id": sid, "tokens": tokens}
        )
    _check_selection_complete(path, wanted, selected, totals)


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


def verify_training_release(
    manifest_path: Path,
    *,
    train: Path,
    train_ids: Path,
    split_manifest: Path,
    label_config: Path,
    mode: str,
    release_id: str,
    policy_version: str,
    prediction: Path | None = None,
    prediction_ids: Path | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("release_id") != release_id
        or manifest.get("policy_version") != policy_version
    ):
        raise ValueError("release identity/schema/policy does not match command")
    if sha256_file(label_config) != manifest["label_sets_sha256"]:
        raise ValueError("label configuration does not match the data release")
    processed, splits = manifest_path.parent, split_manifest.parent
    checked_release_file(train, processed, manifest["files"], expected=f"mode_{mode}/train.jsonl")
    checked_release_file(
        train_ids, splits, manifest["split_files"], expected=f"default_train/mode_{mode}.ids"
    )
    _check_split_manifest(split_manifest, manifest)
    if prediction is not None:
        _check_prediction_files(manifest, processed, splits, mode, prediction, prediction_ids)
    return manifest


def write_json(path: Path, value: Any) -> None:
    write_new_file(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def _check_frozen_selection(selection_record: Path, model: Any, inputs: dict[str, Any]) -> None:
    """Require a valid frozen selection that names this artifact."""
    selection = json.loads(selection_record.read_text(encoding="utf-8"))
    if (
        selection.get("used_test_labels") is not False
        or selection.get("training_includes_validation") is not False
        or selection["modes"][model.label_mode]["model_sha256"] != inputs["model"]["sha256"]
    ):
        raise ValueError("artifact does not match a valid frozen selection")


def _check_release_compatibility(
    release_manifest: Path, input_path: Path, ids_path: Path | None, model: Any, inputs: dict[str, Any]
) -> None:
    """Require the artifact, input file and ID selection to belong to the release."""
    release = json.loads(release_manifest.read_text(encoding="utf-8"))
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


def _run_manifest(
    model: Any,
    inputs: dict[str, Any],
    started_at: str,
    predictions_path: Path,
    counts: dict[str, int],
    num_sequences: int,
    timings: tuple[float, float],
    command: list[str] | None,
) -> dict[str, Any]:
    """Build the prediction run manifest; timings is (load_seconds, prediction_seconds)."""
    return {
        "schema_version": 1,
        "model_name": "hmm",
        "label_mode": model.label_mode,
        "run_id": model.metadata.get("run_id"),
        "ordered_labels": list(model.labels),
        "started_at_utc": started_at,
        "inputs": inputs,
        "model_sha256": inputs["model"]["sha256"],
        "predictions_sha256": sha256_file(predictions_path),
        "selection_counts": counts,
        "num_sequences": num_sequences,
        "num_tokens": counts["selected_tokens"],
        "model_load_seconds": timings[0],
        "prediction_seconds_including_input": timings[1],
        "timing_scope": "JSONL reading, observable-input validation, selection, predict; excludes writes",
        "training_provenance": model.metadata,
        "code": code_identity(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "command": command,
    }


def predict_run(
    model_path: Path,
    input_path: Path,
    output_dir: Path,
    *,
    ids_path: Path | None = None,
    release_manifest: Path | None = None,
    selection_record: Path | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Load an artifact and export predictions without refitting or reading gold fields."""
    from .hmm import HMMModel

    if output_dir.exists():
        raise ValueError("output-dir already exists; choose a new prediction run")
    inputs = {"model": file_identity(model_path), "input": file_identity(input_path)}
    if ids_path:
        inputs["ids"] = file_identity(ids_path)
    if release_manifest:
        inputs["release_manifest"] = file_identity(release_manifest)
    if selection_record:
        inputs["selection_record"] = file_identity(selection_record)
    started_at = datetime.now(UTC).isoformat()
    start = perf_counter()
    model = HMMModel.load_model(model_path)
    load_seconds = perf_counter() - start
    if selection_record:
        _check_frozen_selection(selection_record, model, inputs)
    if release_manifest:
        _check_release_compatibility(release_manifest, input_path, ids_path, model, inputs)
    ids = tuple(read_ids(ids_path, strict=True)) if ids_path else None
    counts: dict[str, int] = {}
    start = perf_counter()
    predictions = model.predict(selected_records(input_path, ids, counts=counts))
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
    manifest = _run_manifest(
        model, inputs, started_at, predictions_path, counts, len(predictions),
        (load_seconds, prediction_seconds), command,
    )
    write_json(output_dir / "predictions.manifest.json", manifest)
    return manifest
