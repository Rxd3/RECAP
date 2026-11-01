"""Score, compare and analyse predictions against gold labels (score, compare, errors, baseline).

score    --gold mode_X/<split>.jsonl --ids <list>.ids --predictions <file>.jsonl
         [--masks masks/case_joint/<split>.jsonl] [--manifest <predictions.manifest.json>]
         [--bootstrap N [--seed S]] --out results/tables/<run>.json   (also writes <run>.md)
compare  --gold ... --ids ... --predictions-a A.jsonl --predictions-b B.jsonl [--masks ...]
         [--bootstrap N [--seed S]] [--class LABEL ...] --out results/tables/<comparison>.json
errors   --gold ... --ids ... --predictions ... [--masks ...] [--block NAME] [--per-pair K]
         [--seed S] --out results/tables/<errors>.json
baseline --gold mode_X/<split>.jsonl --ids <list>.ids --out <predictions>.jsonl
Existing outputs are never overwritten and folders are never created.
"""

from __future__ import annotations

import argparse
import errno
import json
import sys
from collections.abc import Callable
from pathlib import Path

from punct_restore.evaluation import (
    BootstrapSpec,
    EvaluationInputError,
    check_manifest,
    compare,
    comparison_markdown,
    error_examples,
    errors_markdown,
    evaluate,
    load_bundle,
    load_scored_gold,
    load_vocab,
    shortcut_predictions,
    to_markdown,
)
from punct_restore.evaluation.error_analysis import DEFAULT_BLOCK, DEFAULT_PER_PAIR
from punct_restore.evaluation.inputs import file_identity
from punct_restore.evaluation.report import printable
from punct_restore.evaluation.uncertainty import (
    DEFAULT_SEED,
    MAX_RESAMPLES,
    MIN_RESAMPLES,
)
from punct_restore.prediction_io import (
    OS_REASONS,
    os_error_code,
    reconfigure_stdout,
    refuse_existing,
    write_new_file,
)


def _bounded(minimum: int, maximum: int | None = None) -> Callable[[str], int]:
    def integer(text: str) -> int:
        value = int(text)
        if value < minimum or (maximum is not None and value > maximum):
            limit = f"{minimum} to {maximum}" if maximum is not None else f"at least {minimum}"
            raise argparse.ArgumentTypeError(f"must be {limit}")
        return value
    return integer


def _add_common(parser: argparse.ArgumentParser, *, predictions: bool = True) -> None:
    parser.add_argument("--gold", type=Path, required=True,
                        help="release view file, e.g. mode_A/validation.jsonl")
    parser.add_argument("--ids", type=Path, required=True,
                        help="scored ID list, e.g. primary_eval/validation_mode_A.ids")
    if predictions:
        parser.add_argument("--predictions", type=Path, required=True,
                            help="prediction JSONL (docs/MODEL_CONTRACT.md)")
    parser.add_argument("--masks", type=Path,
                        help="mode C only (required there): masks/case_joint/<split>.jsonl")
    parser.add_argument("--out", type=Path, required=True,
                        help="report .json path; a .md summary is written next to it")


def _add_bootstrap(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bootstrap", type=_bounded(MIN_RESAMPLES, MAX_RESAMPLES), metavar="N",
                        help=f"grouped bootstrap resamples for 95%% CIs "
                             f"({MIN_RESAMPLES} to {MAX_RESAMPLES})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"random seed for resampling or sampling (default {DEFAULT_SEED})")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recap evaluate-models",
                                     description="Score, compare and analyse RECAP predictions.")
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="validate and score one prediction file")
    _add_common(score)
    score.add_argument("--manifest", type=Path, help="the model's predictions.manifest.json")
    _add_bootstrap(score)
    pair = commands.add_parser("compare", help="paired comparison of two prediction files")
    _add_common(pair, predictions=False)
    pair.add_argument("--predictions-a", type=Path, required=True)
    pair.add_argument("--predictions-b", type=Path, required=True)
    _add_bootstrap(pair)
    pair.add_argument("--class", dest="classes", action="append", default=None, metavar="LABEL",
                      help="per-class F1 difference for LABEL (repeatable; "
                           "mode C uses joint labels like COMMA|LOWER)")
    errors = commands.add_parser("errors", help="error-pair counts and random examples")
    _add_common(errors)
    errors.add_argument("--block", default=DEFAULT_BLOCK)
    errors.add_argument("--per-pair", type=_bounded(1), default=DEFAULT_PER_PAIR)
    errors.add_argument("--seed", type=int, default=DEFAULT_SEED)
    baseline = commands.add_parser("baseline",
                                   help="write shortcut_baseline predictions for the listed IDs")
    baseline.add_argument("--gold", type=Path, required=True)
    baseline.add_argument("--ids", type=Path, required=True)
    baseline.add_argument("--out", type=Path, required=True, help="prediction JSONL to create")
    return parser


def _report_outputs(out: Path) -> tuple[Path, Path]:
    if out.suffix.lower() != ".json":
        raise EvaluationInputError(
            f"--out must be a .json path (the .md summary goes next to it), got {out.name}")
    outputs = (out, out.with_suffix(".md"))
    refuse_existing(*outputs)
    return outputs


def _write_report(outputs: tuple[Path, Path], report: dict, summary: str) -> None:
    write_new_file(outputs[0], json.dumps(report, indent=1, sort_keys=True) + "\n")
    write_new_file(outputs[1], summary)


def _bootstrap(args: argparse.Namespace) -> BootstrapSpec | None:
    return BootstrapSpec(args.bootstrap, args.seed) if args.bootstrap else None


def _score(args: argparse.Namespace) -> None:
    outputs = _report_outputs(args.out)
    vocab = load_vocab()
    bundle = load_bundle(args.gold, args.ids, args.predictions, args.masks, vocab)
    if args.manifest is not None:
        check_manifest(args.manifest, bundle, vocab)  # before the slow bootstrap
    report = evaluate(bundle, vocab, _bootstrap(args))
    if args.manifest is not None:
        inputs = {**report["inputs"], "manifest": file_identity(args.manifest)}
        report = {**report, "inputs": inputs}
    _write_report(outputs, report, to_markdown(report))
    primary = report["primary"]
    print(f"{printable(report['model_name'])} mode {report['label_mode']}: {primary['block']} "
          f"macro F1 = {100 * primary['value']:.2f}% -> {outputs[0].name}, {outputs[1].name}")


def _compare(args: argparse.Namespace) -> None:
    outputs = _report_outputs(args.out)
    vocab = load_vocab()
    bundle_a = load_bundle(args.gold, args.ids, args.predictions_a, args.masks, vocab)
    bundle_b = load_bundle(args.gold, args.ids, args.predictions_b, args.masks, vocab)
    report = compare(bundle_a, bundle_b, vocab, _bootstrap(args), tuple(args.classes or ()))
    _write_report(outputs, report, comparison_markdown(report))
    primary = report["blocks"][report["primary_block"]]
    print(f"{printable(report['model_b'])} vs {printable(report['model_a'])}: "
          f"{report['primary_block']} difference {100 * primary['difference']:+.2f} points "
          f"-> {outputs[0].name}, {outputs[1].name}")


def _errors(args: argparse.Namespace) -> None:
    outputs = _report_outputs(args.out)
    vocab = load_vocab()
    bundle = load_bundle(args.gold, args.ids, args.predictions, args.masks, vocab)
    report = error_examples(bundle, vocab, args.block, args.per_pair, args.seed)
    _write_report(outputs, report, errors_markdown(report))
    print(f"{report['errors_total']} errors, {len(report['examples'])} examples "
          f"-> {outputs[0].name}, {outputs[1].name}")


def _baseline(args: argparse.Namespace) -> None:
    refuse_existing(args.out)
    mode, records = load_scored_gold(args.gold, args.ids, load_vocab())
    lines = (json.dumps(record, sort_keys=True) + "\n"
             for record in shortcut_predictions(records, mode))
    write_new_file(args.out, "".join(lines))
    print(f"wrote {len(records)} shortcut_baseline predictions (mode {mode}) -> {args.out.name}")


COMMANDS = {"score": _score, "compare": _compare, "errors": _errors, "baseline": _baseline}


def _os_message(exc: OSError) -> str:
    """The OS error with its path as given (this command's output wording is pinned by its tests)."""
    path = exc.filename if exc.filename is not None else "(unknown path)"
    code = os_error_code(exc)
    reason = OS_REASONS.get(code)
    if reason is None:
        return f"cannot access {path}: {exc.strerror or type(exc).__name__}"
    return f"{path} {reason}" if code == errno.EEXIST else f"{reason}: {path}"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    reconfigure_stdout()  # cp1252 consoles cannot print every model or file name
    try:
        COMMANDS[args.command](args)
    except EvaluationInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {_os_message(exc)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
