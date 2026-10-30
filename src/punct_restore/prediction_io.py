"""Gold-free prediction input/output shared by train_crf.py and tune_threshold.py."""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .evaluation.inputs import file_identity, sha256_file


def read_frozen_ids(ids_path: Path) -> list[str]:
    """Read and validate a frozen ID manifest file.

    Raises:
        FileNotFoundError: If the ID file does not exist.
        ValueError: If the ID list is empty or contains duplicate IDs.
    """
    if not ids_path.is_file():
        raise FileNotFoundError(f"Frozen ID list not found: {ids_path}")

    ids: list[str] = []
    seen: set[str] = set()

    with open(ids_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            s_id = line.strip()
            if not s_id:
                continue
            if s_id in seen:
                raise ValueError(
                    f"Duplicate ID {s_id!r} found at line {line_num} in {ids_path}"
                )
            seen.add(s_id)
            ids.append(s_id)

    if not ids:
        raise ValueError(f"Frozen ID list is empty: {ids_path}")

    return ids


def _parse_observation_line(line_str: str, line_num: int, input_path: Path) -> dict[str, Any]:
    """Parse one JSONL line into a record with a valid sample_id."""
    try:
        rec = json.loads(line_str)
    except ValueError as e:
        raise ValueError(f"Corrupt JSON at line {line_num}: {e}") from e
    if not isinstance(rec, dict):
        raise ValueError(f"Line {line_num} in {input_path} is not a JSON object")
    s_id = rec.get("sample_id")
    if not isinstance(s_id, str) or not s_id.strip():
        raise ValueError(f"Line {line_num} has missing or empty sample_id")
    return rec


def _gold_free_observation(rec: Mapping[str, Any], line_num: int) -> dict[str, Any]:
    """Keep only sample_id and validated observable tokens (strictly gold-free)."""
    s_id = rec["sample_id"]
    tokens = rec.get("tokens")
    if not tokens or not isinstance(tokens, list):
        raise ValueError(f"Invalid tokens at line {line_num} for ID {s_id!r}")
    if any(not isinstance(t, str) or not t.strip() for t in tokens):
        raise ValueError(f"Tokens contain invalid elements at line {line_num} for ID {s_id!r}")
    return {"sample_id": s_id, "tokens": list(tokens)}


def read_observations(input_path: Path, ids_path: Path | None) -> list[dict[str, Any]]:
    """Read gold-free ``{"sample_id", "tokens"}`` records in INPUT FILE order.

    When ``ids_path`` is given, only the listed IDs are kept and every one must be present.

    Raises:
        FileNotFoundError: If the input JSONL or the ID list does not exist.
        ValueError: On corrupt, duplicate or invalid records, or IDs missing from the input.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL not found at {input_path}")
    target_ids = read_frozen_ids(ids_path) if ids_path is not None else None
    target_set = set(target_ids) if target_ids is not None else None

    observations: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            rec = _parse_observation_line(line_str, line_num, input_path)
            s_id = rec["sample_id"]
            if s_id in seen_ids:
                raise ValueError(f"Duplicate sample_id {s_id!r} encountered at line {line_num}")
            seen_ids.add(s_id)
            if target_set is None or s_id in target_set:
                observations.append(_gold_free_observation(rec, line_num))

    if target_ids is not None:
        missing_ids = set(target_ids) - {o["sample_id"] for o in observations}
        if missing_ids:
            raise ValueError(
                f"Missing {len(missing_ids)} selected IDs from input file {input_path}. "
                f"Examples: {sorted(missing_ids)[:5]}"
            )
        if len(observations) != len(target_ids):
            raise ValueError(
                f"Found {len(observations)} matching records, expected {len(target_ids)} from ID list."
            )
    return observations


def write_predictions_jsonl(predictions: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write one JSON object per line with the platform newline (byte-stable across the team)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(pred, ensure_ascii=False) + "\n" for pred in predictions)


def write_prediction_manifest(
    manifest_path: Path,
    *,
    predictions_path: Path,
    input_path: Path,
    model_name: str,
    label_mode: str,
    ordered_labels: Sequence[str],
) -> None:
    """Write the evaluator-facing sidecar manifest for a predictions file."""
    manifest: dict[str, Any] = {
        "model_name": model_name,
        "label_mode": label_mode,
        "ordered_labels": list(ordered_labels),
        "predictions_sha256": sha256_file(predictions_path),
        "input": file_identity(input_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def write_prediction_outputs(
    predictions: Sequence[Mapping[str, Any]],
    output_path: Path,
    manifest_path: Path | None,
    *,
    input_path: Path,
    model_name: str,
    label_mode: str,
    ordered_labels: Sequence[str],
) -> None:
    """Write the predictions and, when given, their manifest; on any failure try to delete both.

    Cleanup is best effort: it deletes each file this call started writing (including a file it
    was overwriting), ignores a delete that itself fails, and leaves any folders it created.
    The original error is always the one re-raised.
    """
    started: list[Path] = []
    done = False
    try:
        started.append(output_path)
        write_predictions_jsonl(predictions, output_path)
        if manifest_path is not None:
            started.append(manifest_path)
            write_prediction_manifest(
                manifest_path,
                predictions_path=output_path,
                input_path=input_path,
                model_name=model_name,
                label_mode=label_mode,
                ordered_labels=ordered_labels,
            )
        done = True
    finally:
        if not done:
            for path in started:
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)


def same_path(a: str | Path, b: str | Path) -> bool:
    """True when both paths name the same file once resolved (case-blind where the OS is)."""
    return _comparable(a) == _comparable(b)


def _comparable(path: str | Path) -> str:
    """Resolved path as the OS sees it: abspath drops Windows' trailing dots and spaces."""
    return os.path.normcase(os.path.abspath(Path(path).resolve()))


def _check_creatable(flag: str, path: Path) -> None:
    """Raise ValueError if ``path`` is a folder or its nearest existing ancestor is not a writable folder.

    An OSError while looking (e.g. a parent that cannot be read) also becomes a one-line ValueError.
    """
    try:
        if path.is_dir():
            raise ValueError(f"{flag}: {path.resolve().name or path.resolve()} is a folder, not a file")
        ancestor = path.resolve().parent
        while not ancestor.exists() and ancestor.parent != ancestor:
            ancestor = ancestor.parent
        shown = ancestor.name or ancestor.anchor
        if not ancestor.is_dir():
            raise ValueError(f"{flag}: cannot create {path.name}: {shown} is not a folder")
        if not os.access(ancestor, os.W_OK):
            raise ValueError(f"{flag}: cannot create {path.name}: {shown} is not writable")
    except OSError as e:
        raise ValueError(f"{flag}: cannot check {path.name or path}: {e.strerror or e}") from e


def check_new_output_paths(
    outputs: Mapping[str, Path | None], inputs: Mapping[str, Path | None]
) -> None:
    """Raise a one-line ValueError if an output is another output or an input, or cannot be created.

    Keys are the flag names (``"--output"``) used in the message; ``None`` entries are skipped.
    """
    targets = [(flag, Path(path)) for flag, path in outputs.items() if path is not None]
    sources = [(flag, Path(path)) for flag, path in inputs.items() if path is not None]
    for index, (flag, path) in enumerate(targets):
        for other_flag, other in [*targets[index + 1:], *sources]:
            if same_path(path, other):
                raise ValueError(f"{flag} and {other_flag} are the same file")
    for flag, path in targets:
        _check_creatable(flag, path)
