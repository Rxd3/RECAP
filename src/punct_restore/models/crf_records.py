"""CRF record validation: training records to (features, targets), prediction records to features."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from punct_restore.data.labeling import join_labels
from punct_restore.features.crf_features import FeatureConfig
from punct_restore.models.crf_modes import CASING_VOCABULARY

FeatureExtractor = Callable[..., list[dict[str, Any]]]


def _checked_case_labels(rec: Mapping[str, Any], where: str, sample_id: str, seq_len: int) -> list[str]:
    """Return the record's case_labels as a new list after checking length and vocabulary."""
    case_labels = rec.get("case_labels")
    if not isinstance(case_labels, list) or len(case_labels) != seq_len:
        raise ValueError(
            f"{where} ({sample_id}): case_labels length "
            f"({len(case_labels) if isinstance(case_labels, list) else None}) "
            f"does not match tokens length ({seq_len})."
        )
    unknown_cases = [c for c in case_labels if c not in CASING_VOCABULARY]
    if unknown_cases:
        raise ValueError(f"{where} ({sample_id}): unknown case_labels {unknown_cases!r}.")
    return list(case_labels)


def _target_sequence(
    rec: Mapping[str, Any], where: str, sample_id: str, seq_len: int, label_mode: str, vocabulary: Sequence[str]
) -> list[str]:
    """Derive the per-token training targets of the given label mode from one record."""
    punct_labels = rec.get("punct_labels")
    if not isinstance(punct_labels, list) or len(punct_labels) != seq_len:
        raise ValueError(
            f"{where} ({sample_id}): punct_labels length "
            f"({len(punct_labels) if isinstance(punct_labels, list) else None}) "
            f"does not match tokens length ({seq_len})."
        )
    vocab_set = set(vocabulary)

    if label_mode in ("A", "B"):
        unknown_labels = [lbl for lbl in punct_labels if lbl not in vocab_set]
        if unknown_labels:
            raise ValueError(
                f"{where} ({sample_id}): unknown punct_labels {unknown_labels!r} "
                f"for Mode {label_mode}."
            )
        return list(punct_labels)

    if label_mode == "CASE":
        return _checked_case_labels(rec, where, sample_id, seq_len)

    if label_mode == "C":
        case_labels = _checked_case_labels(rec, where, sample_id, seq_len)
        try:
            target_sequence = join_labels(punct_labels, case_labels, separator="|")
        except Exception as e:
            raise ValueError(f"{where} ({sample_id}): failed to derive Mode C targets: {e}") from e
        unknown_joint = [lbl for lbl in target_sequence if lbl not in vocab_set]
        if unknown_joint:
            raise ValueError(
                f"{where} ({sample_id}): derived joint labels {unknown_joint!r} "
                f"outside Mode C vocabulary."
            )
        return target_sequence

    raise ValueError(f"Unsupported label_mode: {label_mode}")


def _checked_training_record(rec: Any, where: str, seen_ids: set[str]) -> tuple[str, list[str]]:
    """sample_id (recorded in ``seen_ids``) and tokens of one training record; leakage-safe split check."""
    if not isinstance(rec, Mapping):
        raise TypeError(f"{where} must be a Mapping, got {type(rec).__name__}")
    sample_id = rec.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError(f"{where}: sample_id must be a non-empty string.")
    if sample_id in seen_ids:
        raise ValueError(f"{where}: duplicate sample_id {sample_id!r} detected.")
    seen_ids.add(sample_id)
    split = rec.get("split")  # prevent training on non-train data (leakage prevention)
    if split is not None and split != "train":
        raise ValueError(
            f"{where} ({sample_id}): cannot train on record with split={split!r}. "
            "Training data must have split='train'."
        )
    tokens = rec.get("tokens")
    if not isinstance(tokens, list) or len(tokens) == 0:
        raise ValueError(f"{where} ({sample_id}): tokens must be a non-empty list.")
    if any(not isinstance(t, str) or not t.strip() for t in tokens):
        raise ValueError(f"{where} ({sample_id}): tokens must contain non-empty strings.")
    return sample_id, tokens


def validate_training_records(
    records: Sequence[Mapping[str, Any]],
    *,
    label_mode: str,
    vocabulary: Sequence[str],
    feature_extractor: FeatureExtractor,
    feature_config: FeatureConfig,
) -> tuple[list[list[dict[str, Any]]], list[list[str]]]:
    """Validate input training records against schema and extract features and targets."""
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError(f"training_records must be a Sequence, got {type(records).__name__}")
    if len(records) == 0:
        raise ValueError("training_records sequence cannot be empty.")

    seen_ids: set[str] = set()
    X: list[list[dict[str, Any]]] = []
    y: list[list[str]] = []

    for idx, rec in enumerate(records):
        where = f"Record at index {idx}"
        sample_id, tokens = _checked_training_record(rec, where, seen_ids)
        seq_len = len(tokens)
        target_sequence = _target_sequence(rec, where, sample_id, seq_len, label_mode, vocabulary)

        # Extract features strictly using observable tokens (gold leakage prevention)
        features = feature_extractor(tokens, config=feature_config)
        if len(features) != seq_len:
            raise ValueError(
                f"{where} ({sample_id}): feature extractor returned {len(features)} dicts "
                f"for {seq_len} tokens."
            )

        X.append(features)
        y.append(target_sequence)

    return X, y


def prediction_features(
    records: Sequence[Mapping[str, Any]], *, feature_extractor: FeatureExtractor, feature_config: FeatureConfig
) -> tuple[list[str], list[int], list[list[dict[str, Any]]]]:
    """Validate gold-free records and extract features.

    Returns:
        (sample ids, token lengths, feature sequences); all empty for empty records.
    """
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError(f"records must be a Sequence, got {type(records).__name__}")

    X: list[list[dict[str, Any]]] = []
    sample_ids: list[str] = []
    token_lengths: list[int] = []

    for idx, rec in enumerate(records):
        where = f"Prediction record at index {idx}"
        if not isinstance(rec, Mapping):
            raise TypeError(f"{where} must be a Mapping, got {type(rec).__name__}")

        sample_id = rec.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"{where}: sample_id must be a non-empty string.")

        tokens = rec.get("tokens")
        if not isinstance(tokens, list) or len(tokens) == 0:
            raise ValueError(f"{where} ({sample_id}): tokens must be a non-empty list.")
        if any(not isinstance(t, str) or not t.strip() for t in tokens):
            raise ValueError(f"{where} ({sample_id}): tokens must contain non-empty strings.")

        features = feature_extractor(tokens, config=feature_config)
        X.append(features)
        sample_ids.append(sample_id)
        token_lengths.append(len(tokens))

    return sample_ids, token_lengths, X
