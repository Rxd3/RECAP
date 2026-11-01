"""Factored mode C prediction: mode A punctuation labels joined with CASE casing labels.

A punctuation model (mode A) and a casing model (CRFModel label mode "CASE") each predict their own half;
``combine`` zips the two prediction lists into ordinary mode C predictions (``PUNCT|CASE``) that the
evaluator scores like the joint model's output.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from punct_restore.data.labeling import join_labels, load_label_sets


def _labels_by_id(
    records: Sequence[Mapping[str, Any]], kind: str, mode: str, vocabulary: Sequence[str]
) -> dict[str, list[str]]:
    """sample_id -> labels (in input order) after checking the mode, the labels and duplicate IDs."""
    index: dict[str, list[str]] = {}
    for number, record in enumerate(records, 1):
        sample_id = record.get("sample_id") if isinstance(record, Mapping) else None
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{kind} prediction {number}: sample_id must be a non-empty string")
        if sample_id in index:
            raise ValueError(f"{kind} predictions: duplicate sample_id {sample_id!r}")
        if record.get("label_mode") != mode:
            raise ValueError(
                f"{kind} sample_id {sample_id!r}: label_mode is {record.get('label_mode')!r}, expected {mode!r}"
            )
        labels = record.get("predicted_labels")
        if not isinstance(labels, list):
            raise ValueError(f"{kind} sample_id {sample_id!r}: predicted_labels must be a list")
        unknown = [label for label in labels if label not in vocabulary]
        if unknown:
            raise ValueError(f"{kind} sample_id {sample_id!r}: unknown mode {mode} label {unknown[0]!r}")
        index[sample_id] = list(labels)
    return index


def _require_same_ids(punct: Mapping[str, Any], case: Mapping[str, Any]) -> None:
    """Raise ValueError naming the first sample_id that only one of the two prediction sets has."""
    for sample_id in punct:
        if sample_id not in case:
            raise ValueError(f"sample_id {sample_id!r} is in the punct predictions but not in the case predictions")
    for sample_id in case:
        if sample_id not in punct:
            raise ValueError(f"sample_id {sample_id!r} is in the case predictions but not in the punct predictions")


def combine(
    punct_predictions: Sequence[Mapping[str, Any]],
    case_predictions: Sequence[Mapping[str, Any]],
    *,
    model_name: str,
) -> list[dict[str, Any]]:
    """Join mode A and CASE predictions into mode C predictions, in the order of ``punct_predictions``.

    Raises:
        ValueError: One line naming the first offending sample_id: a wrong label_mode, an unknown label,
            a duplicate ID, an ID only one side has, or labels of different lengths.
    """
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty string")
    sets = load_label_sets()
    punct = _labels_by_id(punct_predictions, "punct", "A", sets["A"])
    case = _labels_by_id(case_predictions, "case", "CASE", sets["casing"])
    _require_same_ids(punct, case)
    joint_vocabulary = set(sets["C"])
    combined: list[dict[str, Any]] = []
    for sample_id, punct_labels in punct.items():
        case_labels = case[sample_id]
        if len(punct_labels) != len(case_labels):
            raise ValueError(
                f"sample_id {sample_id!r}: {len(punct_labels)} punct labels but {len(case_labels)} case labels"
            )
        labels = join_labels(punct_labels, case_labels, separator=sets["separator"])
        outside = [label for label in labels if label not in joint_vocabulary]
        if outside:
            raise ValueError(f"sample_id {sample_id!r}: {outside[0]!r} is not a mode C label")
        combined.append(
            {"sample_id": sample_id, "model_name": model_name, "label_mode": "C", "predicted_labels": labels}
        )
    return combined
