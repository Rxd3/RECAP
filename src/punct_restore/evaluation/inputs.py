"""Evaluation inputs and checks, owned by Member 5.

Reads a release view (gold), an .ids list, model predictions and the mode C casing masks,
checks them against docs/EVALUATION_PROTOCOL.md and returns aligned, immutable records.
Nothing is repaired: every violation raises EvaluationInputError naming the file and the
line or sample_id.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .metrics import join_joint

ROOT = Path(__file__).resolve().parents[3]
LABEL_SETS_PATH = ROOT / "configs" / "label_sets.yaml"
MODES = ("A", "B", "C")
NO_PUNCT = "O"
BOM = "\ufeff"
GOLD_FIELDS = ("sample_id", "label_mode", "tokens", "punct_labels", "case_labels")
PREDICTION_FIELDS = ("sample_id", "model_name", "label_mode", "predicted_labels")
MASK_FIELDS = ("sample_id", "excluded_positions", "reason")


class EvaluationInputError(ValueError):
    """Input that cannot be scored as given."""


@dataclass(frozen=True)
class Vocab:
    A: tuple[str, ...]
    B: tuple[str, ...]
    C: tuple[str, ...]
    casing: tuple[str, ...]

    def labels(self, mode: str) -> tuple[str, ...]:
        return {"A": self.A, "B": self.B, "C": self.C}[mode]


@dataclass(frozen=True)
class GoldRecord:
    sample_id: str
    group_id: str | None
    tokens: tuple[str, ...]
    labels: tuple[str, ...]  # mode target; "PUNCT|CASE" for mode C


@dataclass(frozen=True)
class Aligned:
    sample_id: str
    group_id: str | None
    gold: tuple[str, ...]
    pred: tuple[str, ...]
    masked: frozenset[int]
    tokens: tuple[str, ...] = ()  # model input words, for error examples


@dataclass(frozen=True)
class Bundle:
    label_mode: str
    model_name: str
    records: tuple[Aligned, ...]
    ignored_prediction_ids: int
    inputs: Mapping[str, Mapping[str, str] | None]  # read-only views (MappingProxyType)
    policy_version: str | None = None  # the gold file's data release policy, e.g. "m1-v2"


def load_vocab(path: str | Path = LABEL_SETS_PATH) -> Vocab:
    """Ordered vocabularies from configs/label_sets.yaml (agreed 4/8/3/12 structure)."""
    name = Path(path).name
    try:
        import yaml  # PyYAML is a declared runtime dependency (pyproject)
    except ModuleNotFoundError as exc:
        raise EvaluationInputError('PyYAML is required: python -m pip install -e ".[dev]"') from exc
    try:
        with open(path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        standard = tuple(config["punctuation"]["standard"])
        extended = tuple(config["punctuation"]["extended"])
        casing = tuple(config["casing"])
        joint = tuple(config["modes"]["C"]["labels"])
    except (KeyError, TypeError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise EvaluationInputError(f"{name}: not a valid label configuration") from exc
    expected = tuple(join_joint(p, c) for p in standard for c in casing)
    sizes = (len(standard), len(extended), len(casing))
    if (sizes != (4, 8, 3) or standard[0] != NO_PUNCT or extended[:4] != standard
            or joint != expected):
        raise EvaluationInputError(f"{name}: label sets must be 4/8/3 with C = standard x casing")
    return Vocab(A=standard, B=extended, C=joint, casing=casing)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict[str, str]:
    """Repo-relative POSIX path (folder/name outside the repo) and SHA-256; no absolute paths."""
    resolved = Path(path).resolve()
    try:
        shown = resolved.relative_to(ROOT).as_posix()
    except ValueError:
        shown = "/".join(resolved.parts[-2:])
    return {"path": shown, "sha256": sha256_file(resolved)}


def _text_lines(path: Path) -> Iterator[tuple[int, str]]:
    """(line number, text) of a UTF-8 file; a leading BOM is dropped, bad bytes name the line."""
    with path.open("rb") as handle:
        for number, raw in enumerate(handle, 1):
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise EvaluationInputError(f"{path.name} line {number}: not UTF-8 text") from exc
            yield number, text.removeprefix(BOM) if number == 1 else text


def _read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """(line number, object) for every non-blank line; a file without records is an error."""
    found = False
    for number, line in _text_lines(path):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as exc:  # JSONDecodeError, or an integer beyond Python's digit limit
            raise EvaluationInputError(
                f"{path.name} line {number}: invalid JSON ({getattr(exc, 'msg', exc)})") from exc
        if not isinstance(record, dict):
            raise EvaluationInputError(f"{path.name} line {number}: record must be a JSON object")
        found = True
        yield number, record
    if not found:
        raise EvaluationInputError(f"{path.name}: file is empty")


def _require(record: Mapping[str, Any], fields: Sequence[str], where: str) -> None:
    missing = [field for field in fields if field not in record]
    if missing:
        raise EvaluationInputError(f"{where}: missing field(s) {', '.join(missing)}")


def _sample_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvaluationInputError(f"{where}: sample_id must be a nonempty string")
    return value


def _new_known_id(record: Mapping[str, Any], gold: Mapping[str, GoldRecord],
                  seen: Mapping[str, Any], where: str) -> str:
    """The record's sample_id, checked to be a gold ID not already seen in this file."""
    sample_id = _sample_id(record["sample_id"], where)
    if sample_id not in gold:
        raise EvaluationInputError(f"{where}: sample_id {sample_id!r} is not in the gold file")
    if sample_id in seen:
        raise EvaluationInputError(f"{where}: duplicate sample_id {sample_id!r}")
    return sample_id


def _require_covered(scored: Sequence[str], found: Mapping[str, Any], path: Path,
                     what: str) -> None:
    missing = [sample_id for sample_id in scored if sample_id not in found]
    if missing:
        raise EvaluationInputError(
            f"{path.name}: no {what} for {len(missing)} scored ID(s), first {missing[0]!r}")


def _checked_labels(value: Any, allowed: Sequence[str], length: int, where: str,
                    field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(label, str) for label in value):
        raise EvaluationInputError(f"{where}: {field} must be a list of strings")
    if len(value) != length:
        raise EvaluationInputError(f"{where}: {field} has {len(value)} labels for {length} tokens")
    unknown = next((label for label in value if label not in allowed), None)
    if unknown is not None:
        raise EvaluationInputError(
            f"{where}: {field} label {unknown!r} is not one of {list(allowed)}")
    return tuple(value)


def _metadata_text(record: Mapping[str, Any], field: str, where: str) -> str | None:
    """metadata.<field> as given; no metadata object or null gives None, non-strings raise."""
    metadata = record.get("metadata")
    value = metadata.get(field) if isinstance(metadata, dict) else None
    if value is not None and (not isinstance(value, str) or not value):
        raise EvaluationInputError(f"{where}: metadata.{field} must be a nonempty string")
    return value


def _gold_record(record: Mapping[str, Any], mode: str, vocab: Vocab, where: str) -> GoldRecord:
    sample_id = _sample_id(record["sample_id"], where)
    tokens = record["tokens"]
    if (not isinstance(tokens, list) or not tokens
            or any(not isinstance(token, str) or not token for token in tokens)):
        raise EvaluationInputError(f"{where}: tokens must be a nonempty list of nonempty strings")
    punct_vocab = vocab.B if mode == "B" else vocab.A
    punct = _checked_labels(record["punct_labels"], punct_vocab, len(tokens), where, "punct_labels")
    case = _checked_labels(record["case_labels"], vocab.casing, len(tokens), where, "case_labels")
    labels = tuple(map(join_joint, punct, case)) if mode == "C" else punct
    return GoldRecord(sample_id, _metadata_text(record, "group_id", where), tuple(tokens), labels)


def _read_gold(path: Path, vocab: Vocab) -> tuple[str, dict[str, GoldRecord], str | None]:
    """Mode, records by ID and the single policy_version shared by every line (None if absent)."""
    mode: str | None = None
    policy_version: str | None = None
    records: dict[str, GoldRecord] = {}
    for number, record in _read_jsonl(path):
        where = f"{path.name} line {number}"
        _require(record, GOLD_FIELDS, where)
        label_mode = record["label_mode"]
        if label_mode not in MODES:
            raise EvaluationInputError(
                f"{where}: label_mode {label_mode!r} is not one of {list(MODES)}")
        mode = mode or label_mode
        if label_mode != mode:
            raise EvaluationInputError(
                f"{where}: label_mode {label_mode} differs from earlier {mode}")
        version = _metadata_text(record, "policy_version", where)
        if records and version != policy_version:
            raise EvaluationInputError(
                f"{where}: metadata.policy_version {version!r} differs from earlier "
                f"{policy_version!r}")
        policy_version = version
        gold = _gold_record(record, mode, vocab, where)
        if gold.sample_id in records:
            raise EvaluationInputError(f"{where}: duplicate sample_id {gold.sample_id!r}")
        records[gold.sample_id] = gold
    return mode, records, policy_version


def _read_ids(path: Path, gold: Mapping[str, GoldRecord], gold_name: str) -> tuple[str, ...]:
    ids = tuple(line.strip() for _, line in _text_lines(path) if line.strip())
    if not ids:
        raise EvaluationInputError(f"{path.name}: ID list is empty")
    seen: set[str] = set()
    for sample_id in ids:
        if sample_id in seen:
            raise EvaluationInputError(f"{path.name}: duplicate ID {sample_id!r}")
        if sample_id not in gold:
            raise EvaluationInputError(f"{path.name}: ID {sample_id!r} is not in {gold_name}")
        seen.add(sample_id)
    return ids


def _model_name(record: Mapping[str, Any], earlier: str | None, where: str) -> str:
    name = record["model_name"]
    if not isinstance(name, str) or not name:
        raise EvaluationInputError(f"{where}: model_name must be a nonempty string")
    if earlier is not None and name != earlier:
        raise EvaluationInputError(f"{where}: model_name {name!r} differs from earlier {earlier!r}")
    return name


def _read_predictions(path: Path, mode: str, vocab: Vocab, gold: Mapping[str, GoldRecord],
                      scored: Sequence[str]) -> tuple[str, dict[str, tuple[str, ...]], int]:
    """(model name, labels by ID, number of valid predictions outside the scored list)."""
    model: str | None = None
    predictions: dict[str, tuple[str, ...]] = {}
    for number, record in _read_jsonl(path):
        where = f"{path.name} line {number}"
        _require(record, PREDICTION_FIELDS, where)
        if record["label_mode"] != mode:
            raise EvaluationInputError(
                f"{where}: label_mode {record['label_mode']!r} does not match gold mode {mode}")
        model = _model_name(record, model, where)
        sample_id = _new_known_id(record, gold, predictions, where)
        predictions[sample_id] = _checked_labels(record["predicted_labels"], vocab.labels(mode),
                                                 len(gold[sample_id].tokens), where,
                                                 "predicted_labels")
    _require_covered(scored, predictions, path, "prediction")
    return model, predictions, len(predictions.keys() - set(scored))


def _read_masks(path: Path, gold: Mapping[str, GoldRecord],
                scored: Sequence[str]) -> dict[str, frozenset[int]]:
    masks: dict[str, frozenset[int]] = {}
    for number, record in _read_jsonl(path):
        where = f"{path.name} line {number}"
        _require(record, MASK_FIELDS, where)
        sample_id = _new_known_id(record, gold, masks, where)
        positions, length = record["excluded_positions"], len(gold[sample_id].tokens)
        if (not isinstance(positions, list)
                or any(type(i) is not int or not 0 <= i < length for i in positions)):
            raise EvaluationInputError(
                f"{where}: excluded_positions must be integers from 0 to {length - 1}")
        if len(set(positions)) != len(positions):
            raise EvaluationInputError(f"{where}: excluded_positions has duplicates")
        masks[sample_id] = frozenset(positions)
    _require_covered(scored, masks, path, "mask entry")
    return masks


def load_scored_gold(gold_path: str | Path, ids_path: str | Path,
                     vocab: Vocab) -> tuple[str, tuple[GoldRecord, ...]]:
    """Mode and the gold records named by the ID list, in list order."""
    gold_path = Path(gold_path)
    mode, gold, _ = _read_gold(gold_path, vocab)
    scored = _read_ids(Path(ids_path), gold, gold_path.name)
    return mode, tuple(gold[sample_id] for sample_id in scored)


def _identities(**paths: Path | None) -> Mapping[str, Mapping[str, str] | None]:
    return MappingProxyType({
        key: MappingProxyType(file_identity(path)) if path is not None else None
        for key, path in paths.items()})


def load_bundle(gold_path: str | Path, ids_path: str | Path, predictions_path: str | Path,
                masks_path: str | Path | None, vocab: Vocab) -> Bundle:
    """Checked, aligned records for the scored IDs; masks are required for mode C only."""
    gold_path, ids_path, predictions_path = Path(gold_path), Path(ids_path), Path(predictions_path)
    masks_file = Path(masks_path) if masks_path is not None else None
    mode, gold, policy_version = _read_gold(gold_path, vocab)
    if mode == "C" and masks_file is None:
        raise EvaluationInputError(
            "mode C needs the casing masks file (masks/case_joint/<file>.jsonl)")
    if mode != "C" and masks_file is not None:
        raise EvaluationInputError(
            f"casing masks apply to mode C only; {gold_path.name} is mode {mode}")
    scored = _read_ids(ids_path, gold, gold_path.name)
    model, predictions, ignored = _read_predictions(predictions_path, mode, vocab, gold, scored)
    masks = _read_masks(masks_file, gold, scored) if masks_file is not None else {}
    records = tuple(Aligned(sample_id, gold[sample_id].group_id, gold[sample_id].labels,
                            predictions[sample_id], masks.get(sample_id, frozenset()),
                            gold[sample_id].tokens)
                    for sample_id in scored)
    inputs = _identities(gold=gold_path, ids=ids_path, predictions=predictions_path,
                         masks=masks_file)
    return Bundle(mode, model, records, ignored, inputs, policy_version)
