"""Combine mode A punctuation and CASE casing predictions into mode C predictions (the factored model).

Usage:
  recap factor-predictions --punct A_PRED.jsonl --case CASE_PRED.jsonl \\
      --gold MODE_C_GOLD.jsonl --out C_PRED.jsonl --manifest-out C_PRED.manifest.json [--model-name factored]

``--punct`` holds mode A predictions and ``--case`` holds CRFModel(label_mode="CASE") predictions for the same
sample IDs and lengths. Each token gets the label ``PUNCT|CASE``. ``--gold`` is the mode C gold file the result
will be scored against: it becomes the manifest's input identity, which the evaluator compares with the gold
file it scores. Everything is read and checked before anything is written, and an existing output is never
overwritten.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from punct_restore.data.labeling import load_label_sets
from punct_restore.models.factored import combine
from punct_restore.prediction_io import (
    check_script_paths,
    one_line_error,
    read_prediction_lines,
    reconfigure_stdout,
    write_prediction_outputs,
)


def read_predictions(path: Path) -> list[dict[str, Any]]:
    """Every JSON object of a predictions file; ValueError names the file (and line) only."""
    return [record for _, record in read_prediction_lines(path)]


def _run(args: argparse.Namespace) -> str:
    punct, case, gold = Path(args.punct), Path(args.case), Path(args.gold)
    out, manifest = Path(args.out), Path(args.manifest_out)
    check_script_paths({"--out": out, "--manifest-out": manifest},
                       {"--punct": punct, "--case": case, "--gold": gold})
    combined = combine(read_predictions(punct), read_predictions(case), model_name=args.model_name)
    write_prediction_outputs(combined, out, manifest, input_path=gold, model_name=args.model_name,
                             label_mode="C", ordered_labels=load_label_sets()["C"])
    return f"factored {len(combined)} predictions to {out.name}"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="recap factor-predictions",
                                     description="Combine mode A and CASE predictions into mode C predictions.")
    parser.add_argument("--punct", required=True, help="mode A predictions JSONL")
    parser.add_argument("--case", required=True, help="CASE predictions JSONL (CRFModel label_mode CASE)")
    parser.add_argument("--gold", required=True, help="mode C gold JSONL the result will be scored against")
    parser.add_argument("--out", required=True, help="new mode C predictions JSONL (never overwritten)")
    parser.add_argument("--manifest-out", required=True, help="new manifest JSON for --out (never overwritten)")
    parser.add_argument("--model-name", default="factored", help="model_name of the output (default factored)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    reconfigure_stdout()
    args = _parse_args(argv)
    try:
        summary = _run(args)
    except (ValueError, OSError) as exc:
        print(f"error: {one_line_error(exc)}", file=sys.stderr)
        return 1
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
