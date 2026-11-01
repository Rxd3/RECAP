"""Validated, ordered record selection from JSONL files for the HMM commands."""

from __future__ import annotations

from collections.abc import Generator, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from punct_restore.data.labeling import load_label_sets
from punct_restore.data.validation import check_model_input, validate_record
from punct_restore.prediction_io import parse_json_object


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield each line of a JSONL file as an object, rejecting blank lines, bad JSON, non-objects and non-UTF-8 text.

    Every rejection is a ValueError of one line that names the file by name only.
    """
    try:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(f"{path.name} line {number}: blank JSONL record")
                yield parse_json_object(line, f"{path.name} line {number}")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name}: not valid UTF-8") from exc


def _check_record_basics(record: dict[str, Any], path: Path, seen: set[str]) -> tuple[str, list[Any]]:
    """Validate schema_version, sample_id and tokens; seen is only read, the caller remembers the sample_id."""
    sid, tokens = record.get("sample_id"), record.get("tokens")
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise ValueError(f"{path.name}: unsupported schema_version")
    if not isinstance(sid, str) or not sid.strip() or sid in seen:
        raise ValueError(f"{path.name}: invalid or duplicate sample_id {sid!r}")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(not isinstance(word, str) or not word.strip() for word in tokens)
    ):
        raise ValueError(f"{sid}: tokens must be a nonempty list of nonempty strings")
    return sid, tokens


@dataclass(frozen=True)
class _TrainingRules:
    """What a training row must satisfy: the label sets, its mode and, for a release, its policy and source."""

    label_sets: Mapping[str, Any]
    mode: str
    policy_version: str | None
    source_revision: str | None


def _check_training_record(record: dict[str, Any], sid: str, tokens: list[Any], rules: _TrainingRules) -> None:
    """Validate a training row, including the split, mode and release policy."""
    errors = validate_record(record, rules.label_sets) + check_model_input(tokens)
    if errors:
        raise ValueError(f"{sid}: {errors[0]}")
    if record["split"] != "train" or record["label_mode"] != rules.mode:
        raise ValueError(f"{sid}: training input must contain only train/{rules.mode}")
    if rules.policy_version is not None:
        meta = record.get("metadata", {})
        if (
            meta.get("policy_version") != rules.policy_version
            or meta.get("source_revision") != rules.source_revision
            or meta.get("annotation_view") != "canonical"
        ):
            raise ValueError(f"{sid}: record does not match the canonical release policy")


def _check_selection_complete(path: Path, wanted: set[str] | None, selected: set[str]) -> None:
    """Reject missing selection IDs and empty selections."""
    if wanted is not None and selected != wanted:
        missing = sorted(wanted - selected)
        raise ValueError(f"{path.name}: {len(missing)} selection IDs missing, first: {missing[:3]}")
    if not selected:
        raise ValueError(f"{path.name}: selection is empty")


def _scan(
    path: Path,
    ids: tuple[str, ...] | None,
    training_mode: str | None,
    policy_version: str | None,
    source_revision: str | None,
) -> Generator[dict[str, Any], None, dict[str, int]]:
    """Yield the selected records in file order; the generator's result is the counts of everything it read."""
    wanted = None if ids is None else set(ids)
    rules = None
    if training_mode is not None:
        rules = _TrainingRules(
            label_sets=load_label_sets(),
            mode=training_mode,
            policy_version=policy_version,
            source_revision=source_revision,
        )
    seen: set[str] = set()
    selected: set[str] = set()
    read = tokens_selected = 0
    for record in read_jsonl(path):
        sid, tokens = _check_record_basics(record, path, seen)
        seen.add(sid)
        if rules is not None:
            _check_training_record(record, sid, tokens, rules)
        read += 1
        if wanted is not None and sid not in wanted:
            continue
        selected.add(sid)
        tokens_selected += len(tokens)
        yield record if rules is not None else {"schema_version": 1, "sample_id": sid, "tokens": tokens}
    _check_selection_complete(path, wanted, selected)
    return {"input_records": read, "selected_records": len(selected), "selected_tokens": tokens_selected}


class SelectedRecords:
    """The selected records of a JSONL file, in source-file order, read lazily.

    Missing IDs are rejected even after the last yielded row. Training validates every row, including excluded
    rows, to catch mixed splits; prediction checks observable fields only and removes gold and source metadata.
    `counts` is complete once the iteration has finished; each iteration reads the file again and replaces it.
    """

    def __init__(
        self,
        path: Path,
        ids: tuple[str, ...] | None,
        *,
        training_mode: str | None = None,
        policy_version: str | None = None,
        source_revision: str | None = None,
    ) -> None:
        self._scan_args = (path, ids, training_mode, policy_version, source_revision)
        self.counts: dict[str, int] = {}

    def __iter__(self) -> Iterator[dict[str, Any]]:
        self.counts = yield from _scan(*self._scan_args)
