"""Validated file selection and provenance for Member 2's HMM commands."""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from punct_restore.evaluation.inputs import file_identity_with
from punct_restore.hashing import sha256_file
from punct_restore.models.hmm import HMMModel
from punct_restore.models.hmm_records import SelectedRecords
from punct_restore.models.hmm_release import checked_release_file, read_release_manifest
from punct_restore.paths import REPO_ROOT as ROOT
from punct_restore.prediction_io import read_ids, write_new_file

# isort: split
# old import path kept (restructure spec §2)
from punct_restore.data.labeling import load_label_sets as load_label_sets
from punct_restore.data.validation import check_model_input as check_model_input
from punct_restore.data.validation import validate_record as validate_record
from punct_restore.file_io import read_json_object as read_json_object
from punct_restore.models.hmm_records import read_jsonl as read_jsonl
from punct_restore.models.hmm_release import ReleaseInputs as ReleaseInputs
from punct_restore.models.hmm_release import verify_release_inputs as verify_release_inputs
from punct_restore.models.hmm_release import verify_training_release as verify_training_release


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def file_identity(path: Path) -> dict[str, str]:
    """The evaluator's file identity, with the file name alone for a file outside the repository."""
    return file_identity_with(path, ROOT, 1)


def code_identity(root: Path = ROOT) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file: main's name for `hashing.sha256_file` (old import path kept, restructure spec §2)."""
    return sha256_file(path)


def read_id_selection(path: Path) -> tuple[str, ...]:
    """The IDs of a selection file as a tuple; blank lines, padded IDs, duplicates and an empty list are refused.

    Main's reader (old import path kept, restructure spec §2), now `prediction_io.read_ids` in strict mode.
    """
    return tuple(read_ids(path, strict=True))


def selected_records(
    path: Path,
    ids: tuple[str, ...] | None,
    *,
    training_mode: str | None = None,
    policy_version: str | None = None,
    source_revision: str | None = None,
    counts: dict[str, int] | None = None,
) -> Iterator[dict[str, Any]]:
    """Main's generator over `SelectedRecords` (old import path kept, restructure spec §2).

    The old API's contract, and the one place an argument is deliberately changed: a caller-supplied `counts` dict is
    filled in place. Its three keys are reset when the first row is asked for and hold the totals of everything the
    file held once the iteration has finished (`SelectedRecords` only knows them then, so they are not live). If the
    iteration stops early or raises, the counts stay at zero, where main kept partial totals.
    """
    selection = SelectedRecords(
        path, ids, training_mode=training_mode, policy_version=policy_version, source_revision=source_revision
    )
    if counts is not None:
        counts.update(input_records=0, selected_records=0, selected_tokens=0)
    yield from selection
    if counts is not None:
        counts.update(selection.counts)


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
    label_configuration = meta.get("label_configuration")  # the artifact's own note: any JSON value
    if (
        release["release_id"] != meta.get("dataset_release_id")
        or release["policy_version"] != meta.get("label_policy_version")
        or release["label_sets_sha256"]
        != (label_configuration.get("sha256") if isinstance(label_configuration, dict) else None)
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


def write_predictions_file(path: Path, predictions: Iterable[Mapping[str, Any]]) -> None:
    """Create ``path`` holding one JSON object per prediction: UTF-8, LF line ends, no NaN, never overwriting.

    Raises:
        FileExistsError: If ``path`` exists (it is left alone).
        ValueError: If a prediction holds a NaN or infinite number; no file is made then.
    """
    write_new_file(path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in predictions))


def _write_predictions(output_dir: Path, predictions: list[dict[str, Any]]) -> Path:
    """Create output_dir and write predictions.jsonl; return its path."""
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "predictions.jsonl"
    write_predictions_file(predictions_path, predictions)
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


def predict_run_with(
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
    """Main's call signature of `predict_run_with` (old import path and signature kept, restructure spec §2)."""
    options = PredictionOptions(
        ids_path=ids_path, release_manifest=release_manifest, selection_record=selection_record, command=command
    )
    return predict_run_with(model_path, input_path, output_dir, options)
