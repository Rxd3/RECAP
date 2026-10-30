"""Project mode C predictions (PUNCT|CASE) onto mode A punctuation labels: the joint-vs-punctuation-only study (H1).

Usage (study order):
  python scripts/project_joint.py --predictions C_PRED.jsonl --gold MODE_A_GOLD.jsonl \\
      --out A_PRED.jsonl --manifest-out A_PRED.manifest.json
  python scripts/evaluate_models.py compare ... --ids primary_eval/validation_mode_C.ids

``--gold`` is the mode A gold file the projection will be scored against: it becomes the manifest's input
identity, which the evaluator compares with the gold file it scores. All predictions are read and validated
before anything is written, and an existing output is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# Support the documented source-checkout command before an editable install.
sys.path.insert(0, str(ROOT / "src"))
from punct_restore.data.labeling import load_label_sets
from punct_restore.prediction_io import check_new_output_paths, write_prediction_outputs

SEPARATOR = "|"
MODEL_SUFFIX = "_joint_C"


def project(record: Mapping[str, Any]) -> dict[str, Any]:
    """A new mode A record from a validated mode C record (the key order is part of the file format)."""
    return {
        "sample_id": record["sample_id"],
        "model_name": f"{record['model_name']}{MODEL_SUFFIX}",
        "label_mode": "A",
        "predicted_labels": [label.split(SEPARATOR, 1)[0] for label in record["predicted_labels"]],
    }


def _check_record(record: Any, where: str, joint_labels: frozenset[str]) -> None:
    """Raise ValueError unless ``record`` is a well-formed mode C prediction."""
    if not isinstance(record, dict):
        raise ValueError(f"{where}: not a JSON object")  # noqa: TRY004 - main() reports ValueError only
    for field in ("sample_id", "model_name"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise ValueError(f"{where}: {field} must be a non-empty string")
    if record.get("label_mode") != "C":
        raise ValueError(f"{where}: label_mode is {record.get('label_mode')!r}, expected 'C'")
    labels = record.get("predicted_labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"{where}: predicted_labels must be a non-empty list")
    for label in labels:
        if not isinstance(label, str) or label not in joint_labels:
            raise ValueError(f"{where}: a mode C label must be PUNCT{SEPARATOR}CASE, got {label!r}")


def _parse_line(line: str, where: str) -> Any:
    try:
        return json.loads(line)
    except ValueError as exc:
        raise ValueError(f"{where}: not valid JSON") from exc


def read_predictions(path: Path, joint_labels: frozenset[str]) -> list[dict[str, Any]]:
    """Every record of a mode C predictions file, validated; ValueError names the file and line."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name}: not valid UTF-8") from exc
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for number, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        where = f"{path.name} line {number}"
        record = _parse_line(line, where)
        _check_record(record, where, joint_labels)
        if record["sample_id"] in seen:
            raise ValueError(f"{where}: duplicate sample_id {record['sample_id']!r}")
        if records and record["model_name"] != records[0]["model_name"]:
            raise ValueError(f"{where}: model_name {record['model_name']!r} differs from "
                             f"{records[0]['model_name']!r}")
        seen.add(record["sample_id"])
        records.append(record)
    if not records:
        raise ValueError(f"{path.name}: no predictions found")
    return records


def _check_paths(out: Path, man: Path, preds: Path, gold: Path) -> None:
    """Refuse collisions, unwritable outputs and existing outputs, then require both inputs to exist."""
    check_new_output_paths({"--out": out, "--manifest-out": man}, {"--predictions": preds, "--gold": gold})
    for flag, path in (("--out", out), ("--manifest-out", man)):
        if os.path.lexists(path):
            raise ValueError(f"{flag}: {path.name} already exists, refusing to overwrite it")
    for flag, path in (("--predictions", preds), ("--gold", gold)):
        if not path.is_file():
            raise ValueError(f"{flag}: {path.name} is not an existing file")


def _run(args: argparse.Namespace) -> str:
    preds, gold, out, man = Path(args.predictions), Path(args.gold), Path(args.out), Path(args.manifest_out)
    _check_paths(out, man, preds, gold)
    label_sets = load_label_sets()
    records = read_predictions(preds, frozenset(label_sets["C"]))
    projected = [project(record) for record in records]
    write_prediction_outputs(projected, out, man, input_path=gold, model_name=projected[0]["model_name"],
                             label_mode="A", ordered_labels=label_sets["A"])
    return f"projected {len(projected)} predictions to {out.name}"


def _one_line(exc: ValueError | OSError) -> str:
    """The message on one line; an OSError names its file by name only (its text carries the full path)."""
    text = str(exc)
    if isinstance(exc, OSError) and exc.strerror:
        text = exc.strerror
        if exc.filename is not None:
            text = f"{Path(os.fsdecode(exc.filename)).name}: {text}"
    return " ".join(text.splitlines())


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project mode C predictions onto mode A punctuation labels.")
    parser.add_argument("--predictions", required=True, help="mode C predictions JSONL")
    parser.add_argument("--gold", required=True, help="mode A gold JSONL the projection will be scored against")
    parser.add_argument("--out", required=True, help="new mode A predictions JSONL (never overwritten)")
    parser.add_argument("--manifest-out", required=True, help="new manifest JSON for --out (never overwritten)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(errors="backslashreplace")
    args = _parse_args(argv)
    try:
        summary = _run(args)
    except (ValueError, OSError) as exc:
        print(f"error: {_one_line(exc)}", file=sys.stderr)
        return 1
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
