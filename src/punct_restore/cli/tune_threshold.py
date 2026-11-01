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
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
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
    check_gamma,
    check_priors,
    check_tau,
    decode_with_threshold,
    label_priors,
)
from punct_restore.prediction_io import (
    check_new_output_paths,
    parse_sample_record,
    read_ids,
    read_observations,
    write_prediction_outputs,
    write_predictions_jsonl,
)

DEFAULT_GRID = "0.05:0.95:0.05"
MAX_GRID_VALUES = 1000
GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0)  # searched only when --train gives the priors


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


def pick_tau_gamma(grid_scores: Sequence[tuple[float, float, float]]) -> tuple[float, float]:
    """``(tau, gamma)`` with the highest macro F1 among ``(tau, gamma, f1)`` rows.

    Ties go to the smallest gamma, then to the tau nearest 0.5, then to the lower tau.
    """
    if not grid_scores:
        raise ValueError("cannot pick a tau from an empty grid")
    best = max(score for _, _, score in grid_scores)
    tied = [(tau, gamma) for tau, gamma, score in grid_scores if score == best]
    return min(tied, key=lambda pair: (pair[1], round(abs(pair[0] - 0.5), 9), pair[0]))


def _tau_model_name(base_name: str, tau: float, gamma: float = 0.0) -> str:
    name = f"{base_name}_tau{tau:g}"
    return f"{name}_gamma{gamma:g}" if gamma > 0 else name


@dataclass(frozen=True)
class Marginals:
    """A model's per-token marginals for some sequences, the name and mode of its predictions, the label priors."""

    base_name: str
    mode: str
    sample_ids: Sequence[str]
    marginals: Sequence[Sequence[Mapping[str, float]]]
    priors: Mapping[str, float] | None = None


def _predictions(source: Marginals, tau: float, *, gamma: float = 0.0) -> list[dict[str, Any]]:
    """Prediction records (model contract) for every sequence, decoded at ``tau`` and ``gamma``."""
    name = _tau_model_name(source.base_name, tau, gamma)
    return [
        {"sample_id": sample_id, "model_name": name, "label_mode": source.mode,
         "predicted_labels": decode_with_threshold(sequence, tau, priors=source.priors, gamma=gamma)}
        for sample_id, sequence in zip(source.sample_ids, source.marginals, strict=True)
    ]


def _refuse_existing(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            raise FileExistsError(f"{path} already exists; refusing to overwrite it")


def _score_grid(
    source: Marginals,
    grid: Sequence[float],
    score_file: Callable[[Path], float],
    work_dir: Path,
    *,
    gammas: Sequence[float] = (0.0,),
) -> list[tuple[float, float, float]]:
    """``(tau, gamma, primary macro F1)`` of the decoded predictions for every gamma and tau."""
    work_dir.mkdir(parents=True, exist_ok=True)
    scores: list[tuple[float, float, float]] = []
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        predictions_path = Path(tmp) / "predictions.jsonl"
        for gamma in gammas:
            for tau in grid:
                write_predictions_jsonl(_predictions(source, tau, gamma=gamma), predictions_path)
                scores.append((tau, gamma, score_file(predictions_path)))
    return scores


def _stream_train_labels(train_path: Path, ids: Sequence[str], mode: str) -> Iterator[dict[str, Any]]:
    """Yield the gold label lists of the listed records, one line at a time; all IDs must be found."""
    wanted, found = set(ids), set()
    with open(train_path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            where = f"{train_path.name} line {number}"
            record = parse_sample_record(line, where)
            if record["sample_id"] not in wanted:
                continue
            if record["sample_id"] in found:
                raise ValueError(f"{where}: a listed sample_id appears twice")
            if record.get("label_mode") != mode:
                raise ValueError(f"{where}: a listed record is not label mode {mode}")
            found.add(record["sample_id"])
            yield {key: record.get(key) for key in ("punct_labels", "case_labels")}
    if len(found) < len(wanted):
        raise ValueError(f"{len(wanted) - len(found)} listed ID(s) are missing from {train_path.name}")


def _train_priors(train: Path, train_ids: Path, model: CRFModel) -> dict[str, float]:
    """Add-one label priors of the model's mode from the listed training records."""
    if not train.is_file():
        raise FileNotFoundError(f"training file not found at {train.name}")
    try:
        ids = read_ids(train_ids)
        return label_priors(_stream_train_labels(train, ids, model.label_mode), model.vocabulary)
    except UnicodeDecodeError:
        raise ValueError("--train and --train-ids must be UTF-8 text files") from None


def _identity_or_none(path: str | None) -> dict[str, str] | None:
    return file_identity(path) if path is not None else None


def _provenance(args: argparse.Namespace) -> dict[str, Any]:
    """The split name and the identity (repo-relative path and SHA-256) of every input file of a tune run."""
    inputs = {"gold": file_identity(args.gold), "ids": file_identity(args.ids)}
    inputs |= {key: _identity_or_none(getattr(args, key)) for key in ("masks", "train", "train_ids")}
    return {"split": args.split or Path(args.gold).stem, "inputs": inputs}


def _tau_result(
    best: tuple[float, float],
    scores: Sequence[tuple[float, float, float]],
    priors: Mapping[str, float] | None,
    model_sha256: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {"tau": best[0], "gamma": best[1]}
    if priors is not None:
        result["priors"] = dict(priors)
    result["grid"] = [{"tau": tau, "gamma": gamma, "macro_f1": f1} for tau, gamma, f1 in scores]
    result["model_sha256"] = model_sha256
    return {**result, **provenance}


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
    _refuse_existing(out)
    model = CRFModel.load_model(args.model)
    require_scoring_mode(model.label_mode)
    vocab = load_vocab()
    mode, gold = load_scored_gold(args.gold, args.ids, vocab)
    if mode != model.label_mode:
        raise ValueError(f"gold is mode {mode} but the model is mode {model.label_mode}")
    priors = None
    if train is not None and train_ids is not None:
        priors = _train_priors(train, train_ids, model)
    sample_ids = [record.sample_id for record in gold]
    marginals = model.predict_marginals(
        [{"sample_id": record.sample_id, "tokens": list(record.tokens)} for record in gold]
    )

    def score_file(predictions_path: Path) -> float:
        bundle = load_bundle(args.gold, args.ids, predictions_path, args.masks, vocab)
        return evaluate(bundle, vocab)["primary"]["value"]

    scores = _score_grid(Marginals(model.model_name, mode, sample_ids, marginals, priors), args.grid, score_file,
                         out.parent, gammas=GAMMAS if priors is not None else (0.0,))
    best = pick_tau_gamma(scores)
    result = _tau_result(best, scores, priors, sha256_file(args.model), _provenance(args))  # before out exists
    with open(out, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    best_f1 = next(f1 for tau, gamma, f1 in scores if (tau, gamma) == best)
    print(f"tau={best[0]:g} gamma={best[1]:g} macro_f1={best_f1:.6f} ({len(scores)} values scored)")
    return 0


@dataclass(frozen=True)
class TauSettings:
    """What ``predict`` needs from a tau file; ``priors`` is None when ``gamma`` is 0."""

    tau: float
    gamma: float
    priors: dict[str, float] | None


def _gamma_settings(data: Mapping[str, Any]) -> tuple[float, dict[str, float] | None]:
    """Gamma (0 when the file predates it) and, when gamma > 0, the validated priors."""
    gamma = data.get("gamma", 0.0)
    check_gamma(gamma)
    gamma = float(gamma) + 0.0  # + 0.0: a gamma of -0.0 is a plain gamma of 0
    if gamma == 0:
        return 0.0, None
    priors = data.get("priors")
    check_priors(priors)
    return gamma, dict(priors)


def _read_tau_file(path: Path, model_path: Path) -> TauSettings:
    """Tau, gamma and priors from a ``tune`` result, provided it was tuned for exactly this model file."""
    if not path.is_file():
        raise FileNotFoundError(f"tau file not found at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"tau file {path} is not valid JSON") from exc
    if not isinstance(data, dict) or not isinstance(data.get("model_sha256"), str):
        raise ValueError(f"tau file {path} must be a JSON object with a tau and a model_sha256 string")
    try:
        check_tau(data.get("tau"))
        gamma, priors = _gamma_settings(data)
    except ValueError as exc:
        raise ValueError(f"tau file {path}: {exc}") from exc
    if data["model_sha256"] != sha256_file(model_path):
        raise ValueError("tau file was tuned for a different model")
    # + 0.0: a tau of -0.0 must not name the model "_tau-0"
    return TauSettings(float(data["tau"]) + 0.0, gamma, priors)


def cmd_predict(args: argparse.Namespace) -> int:
    """Write thresholded predictions plus the evaluator's sidecar manifest."""
    model_path, input_path, ids_path = Path(args.model), Path(args.input), Path(args.ids)
    tau_path, out, manifest_out = Path(args.tau_file), Path(args.out), Path(args.manifest_out)
    check_new_output_paths(
        {"--out": out, "--manifest-out": manifest_out},
        {"--model": model_path, "--input": input_path, "--ids": ids_path,
         "--tau-file": tau_path},
    )
    _refuse_existing(out, manifest_out)
    model = CRFModel.load_model(model_path)
    require_scoring_mode(model.label_mode)
    settings = _read_tau_file(tau_path, model_path)
    observations = read_observations(input_path, ids_path)
    marginals = model.predict_marginals(observations)
    sample_ids = [observation["sample_id"] for observation in observations]
    predictions = _predictions(Marginals(model.model_name, model.label_mode, sample_ids, marginals, settings.priors),
                               settings.tau, gamma=settings.gamma)
    write_prediction_outputs(
        predictions,
        out,
        manifest_out,
        input_path=input_path,
        model_name=_tau_model_name(model.model_name, settings.tau, settings.gamma),
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
        print("error: " + " ".join(str(exc).splitlines()), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
