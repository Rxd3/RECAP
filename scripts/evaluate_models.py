"""Member 5 evaluation command line: score a prediction file, or write the shortcut baseline.

score    --gold mode_X/<split>.jsonl --ids <list>.ids --predictions <file>.jsonl
         [--masks masks/case_joint/<split>.jsonl] --out results/tables/<run>.json
         (also writes <run>.md)
baseline --gold mode_X/<split>.jsonl --ids <list>.ids --out <predictions>.jsonl
Existing outputs are never overwritten and folders are never created.
"""

from __future__ import annotations

import argparse
import errno
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Support the documented source-checkout command before an editable install.
sys.path.insert(0, str(ROOT / "src"))
from punct_restore.evaluation import (
    EvaluationInputError,
    evaluate,
    load_bundle,
    load_scored_gold,
    load_vocab,
    shortcut_predictions,
    to_markdown,
)

OS_MESSAGES = {errno.EEXIST: "{} already exists; refusing to overwrite",
               errno.ENOENT: "file or folder not found: {}",
               errno.EISDIR: "is a folder, not a file: {}",
               errno.EACCES: "permission denied: {}",
               errno.EPERM: "permission denied: {}"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score RECAP predictions or write the shortcut baseline.")
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="validate and score one prediction file")
    score.add_argument("--gold", type=Path, required=True,
                       help="release view file, e.g. mode_A/validation.jsonl")
    score.add_argument("--ids", type=Path, required=True,
                       help="scored ID list, e.g. primary_eval/validation_mode_A.ids")
    score.add_argument("--predictions", type=Path, required=True,
                       help="prediction JSONL (docs/MODEL_CONTRACT.md)")
    score.add_argument("--masks", type=Path,
                       help="mode C only (required there): masks/case_joint/<split>.jsonl")
    score.add_argument("--out", type=Path, required=True,
                       help="report .json path; a .md summary is written next to it")
    baseline = commands.add_parser("baseline",
                                   help="write shortcut_baseline predictions for the listed IDs")
    baseline.add_argument("--gold", type=Path, required=True)
    baseline.add_argument("--ids", type=Path, required=True)
    baseline.add_argument("--out", type=Path, required=True, help="prediction JSONL to create")
    return parser


def _printable(text: str) -> str:
    return "".join(char if char.isprintable() else "?" for char in text)


def _refuse_existing(paths: tuple[Path, ...]) -> None:
    for path in paths:
        if path.exists():
            raise FileExistsError(errno.EEXIST, "already exists", str(path))


def _write_new(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _score(args: argparse.Namespace) -> None:
    if args.out.suffix.lower() != ".json":
        raise EvaluationInputError(
            f"--out must be a .json path (the .md summary goes next to it), got {args.out.name}")
    outputs = (args.out, args.out.with_suffix(".md"))
    _refuse_existing(outputs)
    vocab = load_vocab()
    report = evaluate(load_bundle(args.gold, args.ids, args.predictions, args.masks, vocab), vocab)
    summary = to_markdown(report)  # built first so a failure cannot leave RUN.json without RUN.md
    _write_new(outputs[0], json.dumps(report, indent=1, sort_keys=True) + "\n")
    _write_new(outputs[1], summary)
    primary = report["primary"]
    print(f"{_printable(report['model_name'])} mode {report['label_mode']}: {primary['block']} "
          f"macro F1 = {100 * primary['value']:.2f}% -> {outputs[0].name}, {outputs[1].name}")


def _baseline(args: argparse.Namespace) -> None:
    _refuse_existing((args.out,))
    mode, records = load_scored_gold(args.gold, args.ids, load_vocab())
    lines = (json.dumps(record, sort_keys=True) + "\n"
             for record in shortcut_predictions(records, mode))
    _write_new(args.out, "".join(lines))
    print(f"wrote {len(records)} shortcut_baseline predictions (mode {mode}) -> {args.out.name}")


def _os_message(exc: OSError) -> str:
    path = exc.filename if exc.filename is not None else "(unknown path)"
    code = exc.errno
    if code in (errno.EACCES, errno.EPERM) and Path(str(path)).is_dir():
        code = errno.EISDIR  # Windows reports opening a folder as permission denied
    fallback = "cannot access {}: " + (exc.strerror or type(exc).__name__)
    return OS_MESSAGES.get(code, fallback).format(path)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        # cp1252 consoles cannot print every model or file name
        sys.stdout.reconfigure(errors="backslashreplace")
    try:
        (_score if args.command == "score" else _baseline)(args)
    except EvaluationInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {_os_message(exc)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
