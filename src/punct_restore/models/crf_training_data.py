"""Frozen-ID loading of CRF training records: the JSONL records of a frozen ID list, checked for split and labels.

Used by punct_restore.cli.train_crf; the other CRF training support (fingerprint, provenance, manifest) is in
punct_restore.models.crf_training. The token and label checks are the ones CRFModel.fit makes
(punct_restore.models.crf_records), made here on the line each record came from.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from punct_restore.file_io import iter_jsonl, require_file
from punct_restore.models.crf_modes import get_mode_vocabulary
from punct_restore.models.crf_records import checked_targets
from punct_restore.prediction_io import parse_sample_record, read_ids


def _check_selected_record(
    rec: dict[str, Any], sample_id: str, expected_mode: str, vocabulary: Sequence[str], where: str
) -> None:
    """Raise ValueError unless a selected record is a train record of the expected mode with valid tokens and labels.

    ``where`` is ``<file name> line N``, the start of every message.
    """
    # Strict split validation: cannot train on non-train splits
    split = rec.get("split")
    if split != "train":
        raise ValueError(
            f"{where}: {sample_id!r} has split {split!r}, expected 'train'. Non-train records cannot enter training."
        )

    # Label mode validation
    mode = rec.get("label_mode")
    if mode != expected_mode:
        raise ValueError(f"{where}: {sample_id!r} has label_mode {mode!r}, expected {expected_mode!r}")

    # Token and label types, label vocabulary and token/label alignment: the check CRFModel.fit makes
    checked_targets(rec, where, expected_mode, vocabulary)


def _scan_selected_records(
    input_path: Path,
    target_id_set: set[str],
    expected_mode: str,
    vocabulary: Sequence[str],
) -> tuple[list[dict[str, Any]], set[str]]:
    """Read the JSONL file once; return the validated records whose sample_id is selected and the IDs found."""
    selected_records: list[dict[str, Any]] = []
    found_ids: set[str] = set()

    for number, rec in iter_jsonl(input_path, parse_sample_record):
        sample_id = rec["sample_id"]
        if sample_id in target_id_set:  # selected
            where = f"{input_path.name} line {number}"
            if sample_id in found_ids:
                raise ValueError(f"{where}: duplicate sample_id {sample_id!r}")
            found_ids.add(sample_id)
            _check_selected_record(rec, sample_id, expected_mode, vocabulary, where)
            selected_records.append(rec)

    return selected_records, found_ids


def load_and_filter_training_records(
    input_path: Path,
    ids_path: Path,
    expected_mode: str,
    config_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load JSONL training records filtered strictly by membership in frozen IDs.

    Enforces:
    - Every ID in ids_path exists in input_path.
    - Exact cardinality match between ID list and selected records.
    - Records have split == 'train'.
    - Records have label_mode == expected_mode.
    - Tokens are non-empty strings and labels are strings of the mode's vocabulary (the label file ``config_path``,
      default configs/label_sets.yaml), as ``CRFModel.fit`` requires; each problem names the file and the line.
    - Token and label sequence lengths match exactly.
    - Source sequence order is preserved.
    """
    require_file(input_path, "input JSONL")

    frozen_ids = read_ids(ids_path)
    target_id_set = set(frozen_ids)
    vocabulary = get_mode_vocabulary(expected_mode, config_path=config_path)

    selected_records, found_ids = _scan_selected_records(input_path, target_id_set, expected_mode, vocabulary)

    # Validate that every requested ID was found in the input file
    missing_ids = target_id_set - found_ids
    if missing_ids:
        sample_missing = sorted(missing_ids)[:5]
        raise ValueError(
            f"{input_path.name}: missing {len(missing_ids)} selected IDs. Examples: {sample_missing}"
        )

    if len(selected_records) != len(frozen_ids):
        raise ValueError(
            f"{input_path.name}: selected record count ({len(selected_records)}) "
            f"does not match ID-list cardinality ({len(frozen_ids)})"
        )

    return selected_records, frozen_ids
