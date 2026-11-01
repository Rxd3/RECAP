"""Tune and apply the CRF decoding threshold tau and prior exponent gamma.

Usage:
    recap tune-threshold tune --model M.joblib --gold GOLD.jsonl --ids VAL.ids \
        [--masks MASKS.jsonl] [--grid 0.05:0.95:0.05] [--split NAME] \
        [--train TRAIN.jsonl --train-ids TRAIN.ids] --out tau.json
    recap tune-threshold predict --model M.joblib --input IN.jsonl --ids IDS \
        --tau-file tau.json --out PRED.jsonl --manifest-out PRED.manifest.json

``tune`` scores every tau of the grid with the project evaluator (primary macro F1) on the
validation gold and writes the best one next to the model's SHA-256, the split name (``--split``, default
the gold file's stem) and the identity (repo-relative path and SHA-256) of every input file: gold, ids,
masks, train and train ids (null when not given). ``predict`` ignores those two keys, so tau files written
before they existed still work. With ``--train`` and
``--train-ids`` (given together) it also searches gamma over ``GAMMAS``: the label priors come
from the listed training records, and a token that passes the tau gate gets the label with the
best ``P(c|x) / prior(c) ** gamma``. Without them gamma is 0 and nothing changes. ``predict``
refuses a tau file that was tuned for a different model. Existing output files are never
overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from punct_restore.evaluation.inputs import (
    file_identity,
    load_bundle,
    load_scored_gold,
    load_vocab,
    sha256_file,
)
from punct_restore.evaluation.report import evaluate
from punct_restore.models.crf import CRFModel, require_scoring_mode
from punct_restore.models.threshold import (
    GAMMAS,
    Marginals,
    decode_predictions,
    pick_tau_gamma,
    read_tau_file,
    score_grid,
    tau_model_name,
    tau_result,
    train_priors,
)
from punct_restore.prediction_io import (
    check_new_output_paths,
    one_line_error_naming,
    read_observations,
    refuse_existing,
    write_new_file,
    write_prediction_outputs,
)

DEFAULT_GRID = "0.05:0.95:0.05"
MAX_GRID_VALUES = 1000
PATH_OPTIONS = ("model", "gold", "ids", "masks", "train", "train_ids", "out", "input", "tau_file", "manifest_out")


def parse_grid(text: str) -> tuple[float, ...]:
    """Parse ``start:stop:step`` into the tau values from start to stop (inclusive)."""
    try:
        start, stop, step = map(float, text.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"grid must be start:stop:step (three numbers), got {text!r}"
        ) from None
    if not (0 <= start <= stop <= 1 and step > 0 and math.isfinite(step)):
        raise argparse.ArgumentTypeError(
            f"grid needs 0 <= start <= stop <= 1 and step > 0, got {text!r}"
        )
    span = (stop - start) / step
    if not math.isfinite(span) or span + 1e-9 >= MAX_GRID_VALUES:
        raise argparse.ArgumentTypeError(
            f"grid may have at most {MAX_GRID_VALUES} values, got {text!r}"
        )
    count = math.floor(span + 1e-9) + 1
    return tuple(round(start + i * step, 6) for i in range(count))


def _identity_or_none(path: str | None) -> dict[str, str] | None:
    return file_identity(path) if path is not None else None


def _provenance(args: argparse.Namespace) -> dict[str, Any]:
    """The split name and the identity (repo-relative path and SHA-256) of every input file of a tune run."""
    inputs = {"gold": file_identity(args.gold), "ids": file_identity(args.ids)}
    inputs |= {key: _identity_or_none(getattr(args, key)) for key in ("masks", "train", "train_ids")}
    return {"split": args.split or Path(args.gold).stem, "inputs": inputs}


def cmd_tune(args: argparse.Namespace) -> int:
    """Pick the best tau (and gamma, given --train) on the validation gold and write it as JSON."""
    out = Path(args.out)
    train, train_ids = (Path(p) if p is not None else None for p in (args.train, args.train_ids))
    check_new_output_paths(
        {"--out": out},
        {"--model": Path(args.model), "--gold": Path(args.gold), "--ids": Path(args.ids),
         "--masks": Path(args.masks) if args.masks else None,
         "--train": train, "--train-ids": train_ids},
    )
    refuse_existing(out)
    model = CRFModel.load_model(args.model)
    require_scoring_mode(model.label_mode)
    vocab = load_vocab()
    mode, gold = load_scored_gold(args.gold, args.ids, vocab)
    if mode != model.label_mode:
        raise ValueError(f"gold is mode {mode} but the model is mode {model.label_mode}")
    priors = None
    if train is not None and train_ids is not None:
        priors = train_priors(train, train_ids, model.label_mode, model.vocabulary)
    sample_ids = [record.sample_id for record in gold]
    marginals = model.predict_marginals(
        [{"sample_id": record.sample_id, "tokens": list(record.tokens)} for record in gold]
    )

    def score_file(predictions_path: Path) -> float:
        bundle = load_bundle(args.gold, args.ids, predictions_path, args.masks, vocab)
        return evaluate(bundle, vocab)["primary"]["value"]

    source = Marginals(base_name=model.model_name, mode=mode, sample_ids=sample_ids, marginals=marginals,
                       priors=priors)
    scores = score_grid(source, args.grid, score_file, out.parent, gammas=GAMMAS if priors is not None else (0.0,))
    best = pick_tau_gamma(scores)
    result = tau_result(best, scores, priors, sha256_file(args.model), _provenance(args))  # before out exists
    write_new_file(out, json.dumps(result, indent=2), newline=None)  # the platform newline, as json.dump to a text file
    best_f1 = next(f1 for tau, gamma, f1 in scores if (tau, gamma) == best)
    print(f"tau={best[0]:g} gamma={best[1]:g} macro_f1={best_f1:.6f} ({len(scores)} values scored)")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Write thresholded predictions plus the evaluator's sidecar manifest."""
    model_path, input_path, ids_path = Path(args.model), Path(args.input), Path(args.ids)
    tau_path, out, manifest_out = Path(args.tau_file), Path(args.out), Path(args.manifest_out)
    check_new_output_paths(
        {"--out": out, "--manifest-out": manifest_out},
        {"--model": model_path, "--input": input_path, "--ids": ids_path,
         "--tau-file": tau_path},
    )
    refuse_existing(out, manifest_out)
    model = CRFModel.load_model(model_path)
    require_scoring_mode(model.label_mode)
    settings = read_tau_file(tau_path, model_path)
    observations = read_observations(input_path, ids_path)
    marginals = model.predict_marginals(observations)
    sample_ids = [observation["sample_id"] for observation in observations]
    source = Marginals(base_name=model.model_name, mode=model.label_mode, sample_ids=sample_ids,
                       marginals=marginals, priors=settings.priors)
    predictions = decode_predictions(source, settings.tau, gamma=settings.gamma)
    write_prediction_outputs(
        predictions,
        out,
        manifest_out,
        input_path=input_path,
        model_name=tau_model_name(model.model_name, settings.tau, settings.gamma),
        label_mode=model.label_mode,
        ordered_labels=model.vocabulary,
    )
    return 0


def _split_name(text: str) -> str:
    if not text.strip():
        raise argparse.ArgumentTypeError("--split must not be blank")
    return text


def build_parser() -> argparse.ArgumentParser:
    """Top-level parser with the ``tune`` and ``predict`` subcommands."""
    parser = argparse.ArgumentParser(
        prog="recap tune-threshold",
        description="Tune and apply the CRF marginal-threshold decoding tau.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    tune = commands.add_parser("tune", help="Pick the best tau on the validation gold.")
    tune.add_argument("--model", required=True, help="Trained CRF artifact (.joblib).")
    tune.add_argument("--gold", required=True, help="Validation gold JSONL.")
    tune.add_argument("--ids", required=True, help="Validation ID list.")
    tune.add_argument("--masks", default=None, help="Casing masks JSONL (mode C only).")
    tune.add_argument("--grid", type=parse_grid, default=DEFAULT_GRID,
                      help=f"start:stop:step, stop inclusive (default: {DEFAULT_GRID}).")
    tune.add_argument("--split", type=_split_name, default=None,
                      help="Split name recorded in the tau file (default: the gold file's name without extension).")
    tune.add_argument("--train", default=None,
                      help="Training JSONL for the label priors; also searches gamma (needs --train-ids).")
    tune.add_argument("--train-ids", default=None, help="ID list selecting the --train records.")
    tune.add_argument("--out", required=True, help="New tau JSON file (never overwritten).")
    tune.set_defaults(handler=cmd_tune)

    predict = commands.add_parser("predict", help="Predict with a tuned tau.")
    predict.add_argument("--model", required=True, help="Trained CRF artifact (.joblib).")
    predict.add_argument("--input", required=True, help="Input JSONL with sample_id and tokens.")
    predict.add_argument("--ids", required=True, help="ID list restricting the input records.")
    predict.add_argument("--tau-file", required=True, help="Result file written by 'tune'.")
    predict.add_argument("--out", required=True, help="New predictions JSONL (never overwritten).")
    predict.add_argument("--manifest-out", required=True,
                         help="New prediction manifest JSON (never overwritten).")
    predict.set_defaults(handler=cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: 0 on success, 1 with one ``error:`` line on stderr, 2 on usage errors."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "tune" and (args.train is None) != (args.train_ids is None):
        parser.error("--train and --train-ids must be given together")
    try:
        return args.handler(args)
    except (ValueError, OSError) as exc:
        print("error: " + one_line_error_naming(exc, *(getattr(args, option, None) for option in PATH_OPTIONS)),
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
