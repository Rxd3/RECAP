"""Mode A errors by utterance length and QUESTION recall by position (docs/EVALUATION_RESULTS.md section 6).

Usage: recap length-position --gold GOLD.jsonl --ids IDS [--masks M]
           --predictions NAME=PRED.jsonl [NAME=PRED.jsonl ...] --out OUT_JSON
Every prediction file is validated by the evaluator's ``load_bundle`` and scored with its ``score_block`` over
all positions of the listed IDs. Per model the report holds:
  * the micro F1 over COMMA, PERIOD and QUESTION (O excluded) by utterance length in tokens: 1-10, 11-25,
    26-50 and > 50, with the number of sequences in each bucket (null F1 for an empty bucket);
  * the QUESTION recall at the utterance-final position and at every other ("inside") position, with the gold
    QUESTION count of each (null recall without any gold QUESTION).
Fractions have 6 decimals like the evaluator's; the ``*_pct`` strings are the docs' half-up 2-decimal percentages.
Only mode A is handled. OUT_JSON must not exist: the script never replaces a file.
"""

from __future__ import annotations

import argparse
import json
import sys
from bisect import bisect_left
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from punct_restore.evaluation.inputs import (
    Aligned,
    Bundle,
    Vocab,
    load_bundle,
    load_vocab,
)
from punct_restore.evaluation.metrics import score_block
from punct_restore.evaluation.report import pct, rounded
from punct_restore.evaluation.views import BlockSpec, block_pairs, blocks_for, non_final
from punct_restore.reporting._cli import check_out, run, write_new_text

UPPER_EDGES = (10, 25, 50)  # a sequence of 10 tokens is in the first bucket, one of 51 in the last
BOUNDS = list(zip((1, *(edge + 1 for edge in UPPER_EDGES)), (*UPPER_EDGES, None), strict=True))
QUESTION = "QUESTION"


def bucket_index(length: int) -> int:
    """Index into BOUNDS of an utterance of ``length`` tokens."""
    return bisect_left(UPPER_EDGES, length)


def bucket_label(low: int, high: int | None) -> str:
    return f"{low}-{high}" if high is not None else f">{low - 1}"


def final_position(record: Aligned) -> tuple[int]:
    return (len(record.gold) - 1,)


def fraction(value: float) -> tuple[float, str]:
    """The 6-decimal fraction and its half-up percentage text."""
    stored = rounded(value)
    return stored, pct(stored)


def bucket_entry(records: Sequence[Aligned], spec: BlockSpec, low: int, high: int | None) -> dict[str, Any]:
    entry: dict[str, Any] = {"label": bucket_label(low, high), "min_tokens": low, "max_tokens": high,
                             "sequences": len(records), "micro_f1": None, "micro_f1_pct": None}
    if records:
        block = score_block(block_pairs(records, spec), spec.labels, spec.macro_over)
        entry["micro_f1"], entry["micro_f1_pct"] = fraction(block["micro"]["f1"])
    return entry


def question_recall(
    records: Sequence[Aligned], spec: BlockSpec, positions: Callable[[Aligned], Iterable[int]]
) -> dict[str, Any]:
    """QUESTION support and recall at the positions chosen by ``positions`` (record -> indices)."""
    block = score_block(block_pairs(records, replace(spec, positions=positions)), spec.labels, spec.macro_over)
    scores = block["per_class"][QUESTION]
    recall, text = fraction(scores["recall"]) if scores["support"] else (None, None)
    return {"support": scores["support"], "recall": recall, "recall_pct": text}


def model_report(bundle: Bundle, spec: BlockSpec) -> dict[str, Any]:
    groups: list[list[Aligned]] = [[] for _ in BOUNDS]
    for record in bundle.records:
        groups[bucket_index(len(record.gold))].append(record)
    return {"model_name": bundle.model_name, "predictions": dict(bundle.identity("predictions")),
            "length_buckets": [
                bucket_entry(group, spec, low, high) for group, (low, high) in zip(groups, BOUNDS, strict=False)
            ],
            "question_recall": {"final": question_recall(bundle.records, spec, final_position),
                                "inside": question_recall(bundle.records, spec, non_final)}}


def load_mode_a(args: argparse.Namespace, path: Path, vocab: Vocab) -> Bundle:
    bundle = load_bundle(args.gold, args.ids, path, args.masks, vocab)
    if bundle.label_mode != "A":
        raise ValueError(f"{args.gold.name} is mode {bundle.label_mode}: this report handles mode A only")
    return bundle


def write_report(args: argparse.Namespace) -> None:
    inputs = {"--gold": args.gold, "--ids": args.ids, "--masks": args.masks}
    inputs |= {f"--predictions ({name})": path for name, path in args.predictions}
    check_out(args.out, inputs)  # before reading anything
    vocab = load_vocab()
    spec = blocks_for("A", vocab)[0]
    bundles = {name: load_mode_a(args, path, vocab) for name, path in args.predictions}
    shared = next(iter(bundles.values())).inputs
    identities = {key: shared[key] for key in ("gold", "ids", "masks")}
    doc = {"label_mode": "A", "inputs": {key: dict(identity) if identity else None
                                         for key, identity in identities.items()},
           "models": {name: model_report(bundle, spec) for name, bundle in bundles.items()}}
    write_new_text(args.out, json.dumps(doc, indent=1) + "\n")
    print(f"wrote {len(bundles)} model report(s) to {args.out.name}")


def name_and_path(text: str) -> tuple[str, Path]:
    name, separator, path = text.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("each --predictions entry must look like NAME=PREDICTIONS.jsonl")
    return name, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recap length-position",
        description="Mode A micro F1 by utterance length and QUESTION recall by position, per model.")
    parser.add_argument("--gold", type=Path, required=True, help="mode A gold JSONL")
    parser.add_argument("--ids", type=Path, required=True, help="ID list of the scored sequences")
    parser.add_argument("--masks", type=Path, help="casing masks (the evaluator refuses them for mode A)")
    parser.add_argument("--predictions", type=name_and_path, nargs="+", required=True, metavar="NAME=PRED.jsonl",
                        help="one or more prediction files, each with a name for the report")
    parser.add_argument("--out", type=Path, required=True, help="JSON file to create (must not exist)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    names = [name for name, _ in args.predictions]
    if len(set(names)) != len(names):
        parser.error("the --predictions names must be distinct")
    return run(lambda: write_report(args), json_errors=False)  # the evaluator names bad input itself


if __name__ == "__main__":
    sys.exit(main())
