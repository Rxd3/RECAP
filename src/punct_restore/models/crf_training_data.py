"""Frozen-ID loading of CRF training records: the JSONL records of a frozen ID list, checked for split and labels.

Used by punct_restore.cli.train_crf; the other CRF training support (fingerprint, provenance, manifest) is in
punct_restore.models.crf_training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from punct_restore.file_io import iter_jsonl, require_file
from punct_restore.prediction_io import parse_sample_record, read_ids


def _check_selected_record(rec: dict[str, Any], sample_id: str, expected_mode: str) -> None:
    """Raise ValueError unless a selected record is a train record of the expected mode with aligned labels."""
    # Strict split validation: cannot train on non-train splits
    split = rec.get("split")
    if split != "train":
        raise ValueError(
            f"Selected record {sample_id!r} has split={split!r}, expected 'train'. "
            "Non-train records cannot enter training."
        )

    # Label mode validation
    mode = rec.get("label_mode")
    if mode != expected_mode:
        raise ValueError(
            f"Selected record {sample_id!r} has label_mode={mode!r}, expected {expected_mode!r}."
        )

    # Token & label alignment validation
    tokens = rec.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(f"Selected record {sample_id!r} has empty or non-list tokens")

    punct_labels = rec.get("punct_labels")
    if not isinstance(punct_labels, list) or len(punct_labels) != len(tokens):
        raise ValueError(
            f"Selected record {sample_id!r} has mismatched punct_labels length "
            f"({len(punct_labels) if isinstance(punct_labels, list) else None} vs {len(tokens)})"
        )

    if expected_mode == "C":
        case_labels = rec.get("case_labels")
        if not isinstance(case_labels, list) or len(case_labels) != len(tokens):
            raise ValueError(
                f"Selected record {sample_id!r} has mismatched case_labels length "
                f"({len(case_labels) if isinstance(case_labels, list) else None} vs {len(tokens)})"
            )


def _scan_selected_records(
    input_path: Path,
    target_id_set: set[str],
    expected_mode: str,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Read the JSONL file once; return the validated records whose sample_id is selected and the IDs found."""
    selected_records: list[dict[str, Any]] = []
    found_ids: set[str] = set()

    for _, rec in iter_jsonl(input_path, parse_sample_record):
        sample_id = rec["sample_id"]
        if sample_id in target_id_set:  # selected
            if sample_id in found_ids:
                raise ValueError(f"Duplicate sample_id {sample_id!r} encountered in source file {input_path.name}")
            found_ids.add(sample_id)
            _check_selected_record(rec, sample_id, expected_mode)
            selected_records.append(rec)

    return selected_records, found_ids


def load_and_filter_training_records(
    input_path: Path,
    ids_path: Path,
    expected_mode: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load JSONL training records filtered strictly by membership in frozen IDs.

    Enforces:
    - Every ID in ids_path exists in input_path.
    - Exact cardinality match between ID list and selected records.
    - Records have split == 'train'.
    - Records have label_mode == expected_mode.
    - Token and label sequence lengths match exactly.
    - Source sequence order is preserved.
    """
    require_file(input_path, "input JSONL")

    frozen_ids = read_ids(ids_path)
    target_id_set = set(frozen_ids)

    selected_records, found_ids = _scan_selected_records(input_path, target_id_set, expected_mode)

    # Validate that every requested ID was found in the input file
    missing_ids = target_id_set - found_ids
    if missing_ids:
        sample_missing = sorted(missing_ids)[:5]
        raise ValueError(
            f"Missing {len(missing_ids)} selected IDs from input file {input_path.name}. "
            f"Examples: {sample_missing}"
        )

    if len(selected_records) != len(frozen_ids):
        raise ValueError(
            f"Selected record count ({len(selected_records)}) does not match ID-list cardinality ({len(frozen_ids)})"
        )

    return selected_records, frozen_ids
