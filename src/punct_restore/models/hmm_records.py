"""Validated, ordered record selection from JSONL files for the HMM commands."""

from __future__ import annotations

from collections.abc import Generator, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from punct_restore.data.labeling import load_label_sets
from punct_restore.data.validation import check_model_input, validate_record
from punct_restore.file_io import iter_jsonl
from punct_restore.prediction_io import parse_json_object, select_records


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield each line of a JSONL file as an object, rejecting blank lines, bad JSON, non-objects and non-UTF-8 text.

    Every rejection is a ValueError of one line that names the file by name only.
    """
    for _, record in iter_jsonl(path, parse_json_object, skip_blank=False):
        yield record


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


def _checked_rows(
    path: Path, rules: _TrainingRules | None, counts: dict[str, int]
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Every row of the file as ``(line number, record)`` once checked, selected or not; counts the rows read."""
    seen: set[str] = set()
    for number, record in iter_jsonl(path, parse_json_object, skip_blank=False):
        sid, tokens = _check_record_basics(record, path, seen)
        seen.add(sid)
        if rules is not None:
            _check_training_record(record, sid, tokens, rules)
        counts["input_records"] += 1
        yield number, record


def _scan(
    path: Path,
    ids: tuple[str, ...] | None,
    training_mode: str | None,
    policy_version: str | None,
    source_revision: str | None,
    counts: dict[str, int] | None,
) -> Generator[dict[str, Any], None, dict[str, int]]:
    """Yield the selected records in file order; the generator's result is the counts of everything it read.

    ``counts`` is the dict to fill, in place and row by row (so a stopped iteration leaves partial totals), or None
    for a new one. Its three keys are reset when the first row is asked for.
    """
    counts = {} if counts is None else counts
    counts.update(input_records=0, selected_records=0, selected_tokens=0)
    rules = None
    if training_mode is not None:
        rules = _TrainingRules(
            label_sets=load_label_sets(),
            mode=training_mode,
            policy_version=policy_version,
            source_revision=source_revision,
        )

    def keep(record: dict[str, Any], _where: str) -> dict[str, Any]:
        counts["selected_records"] += 1
        counts["selected_tokens"] += len(record["tokens"])
        return record if rules is not None else {"schema_version": 1, "sample_id": record["sample_id"],
                                                  "tokens": record["tokens"]}

    yield from select_records(_checked_rows(path, rules, counts), path.name, ids, keep)
    if not counts["selected_records"]:
        raise ValueError(f"{path.name}: selection is empty")
    return counts


class SelectedRecords:
    """The selected records of a JSONL file, in source-file order, read lazily.

    Missing IDs are rejected even after the last yielded row. Training validates every row, including excluded
    rows, to catch mixed splits; prediction checks observable fields only and removes gold and source metadata.
    `counts` is complete once the iteration has finished; each iteration reads the file again and replaces it. A
    ``counts`` dict given to the constructor is the one filled, in place and row by row, so the caller sees partial
    totals when it stops early or the file is refused (the old `hmm_workflow.selected_records` contract).
    """

    def __init__(
        self,
        path: Path,
        ids: tuple[str, ...] | None,
        *,
        training_mode: str | None = None,
        policy_version: str | None = None,
        source_revision: str | None = None,
        counts: dict[str, int] | None = None,
    ) -> None:
        self._scan_args = (path, ids, training_mode, policy_version, source_revision, counts)
        self.counts: dict[str, int] = {} if counts is None else counts

    def __iter__(self) -> Iterator[dict[str, Any]]:
        self.counts = yield from _scan(*self._scan_args)
