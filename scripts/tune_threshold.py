"""Tune and apply the CRF decoding threshold tau (team extension; tau is tuned on validation only).

Usage:
    python scripts/tune_threshold.py tune --model M.joblib --gold GOLD.jsonl --ids VAL.ids \
        [--masks MASKS.jsonl] [--grid 0.05:0.95:0.05] --out tau.json
    python scripts/tune_threshold.py predict --model M.joblib --input IN.jsonl --ids IDS \
        --tau-file tau.json --out PRED.jsonl --manifest-out PRED.manifest.json

``tune`` scores every tau of the grid with the project evaluator (primary macro F1) on the
validation gold and writes the best one next to the model's SHA-256. ``predict`` refuses a tau
file that was tuned for a different model. Existing output files are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# Support the documented source-checkout command before an editable install.
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_crf import (  # noqa: E402
    read_observations,
    sha256_file,
    write_prediction_manifest,
    write_predictions_jsonl,
)

from punct_restore.evaluation.inputs import load_bundle, load_scored_gold, load_vocab  # noqa: E402
from punct_restore.evaluation.report import evaluate  # noqa: E402
from punct_restore.models.crf import CRFModel  # noqa: E402
from punct_restore.models.threshold import check_tau, decode_with_threshold  # noqa: E402

DEFAULT_GRID = "0.05:0.95:0.05"


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
    count = math.floor((stop - start) / step + 1e-9) + 1
    return tuple(round(start + i * step, 6) for i in range(count))


def pick_tau(grid_scores: Sequence[tuple[float, float]]) -> float:
    """Tau with the highest macro F1; ties go to the tau nearest 0.5, then to the lower tau."""
    if not grid_scores:
        raise ValueError("cannot pick a tau from an empty grid")
    best = max(score for _, score in grid_scores)
    tied = [tau for tau, score in grid_scores if score == best]
    return min(tied, key=lambda tau: (round(abs(tau - 0.5), 9), tau))


def _tau_model_name(base_name: str, tau: float) -> str:
    return f"{base_name}_tau{tau:g}"


def _predictions(
    base_name: str,
    mode: str,
    sample_ids: Sequence[str],
    marginals: Sequence[Sequence[Mapping[str, float]]],
    tau: float,
) -> list[dict[str, Any]]:
    """Prediction records (model contract) for every sequence, decoded at ``tau``."""
    name = _tau_model_name(base_name, tau)
    return [
        {"sample_id": sample_id, "model_name": name, "label_mode": mode,
         "predicted_labels": decode_with_threshold(sequence, tau)}
        for sample_id, sequence in zip(sample_ids, marginals, strict=True)
    ]


def _refuse_existing(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            raise FileExistsError(f"{path} already exists; refusing to overwrite it")


def _score_grid(
    base_name: str,
    mode: str,
    sample_ids: Sequence[str],
    marginals: Sequence[Sequence[Mapping[str, float]]],
    grid: Sequence[float],
    score_file: Callable[[Path], float],
    work_dir: Path,
) -> list[tuple[float, float]]:
    """Primary macro F1 of the thresholded predictions for every tau of the grid."""
    work_dir.mkdir(parents=True, exist_ok=True)
    scores: list[tuple[float, float]] = []
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        predictions_path = Path(tmp) / "predictions.jsonl"
        for tau in grid:
            write_predictions_jsonl(
                _predictions(base_name, mode, sample_ids, marginals, tau), predictions_path
            )
            scores.append((tau, score_file(predictions_path)))
    return scores


def cmd_tune(args: argparse.Namespace) -> int:
    """Pick the best tau on the validation gold and write it as JSON."""
    out = Path(args.out)
    _refuse_existing(out)
    model = CRFModel.load_model(args.model)
    vocab = load_vocab()
    mode, gold = load_scored_gold(args.gold, args.ids, vocab)
    if mode != model.label_mode:
        raise ValueError(f"gold is mode {mode} but the model is mode {model.label_mode}")
    sample_ids = [record.sample_id for record in gold]
    marginals = model.predict_marginals(
        [{"sample_id": record.sample_id, "tokens": list(record.tokens)} for record in gold]
    )

    def score_file(predictions_path: Path) -> float:
        bundle = load_bundle(args.gold, args.ids, predictions_path, args.masks, vocab)
        return evaluate(bundle, vocab)["primary"]["value"]

    scores = _score_grid(model.model_name, mode, sample_ids, marginals, args.grid,
                         score_file, out.parent)
    best = pick_tau(scores)
    result = {
        "tau": best,
        "grid": [{"tau": tau, "macro_f1": f1} for tau, f1 in scores],
        "model_sha256": sha256_file(args.model),
    }
    with open(out, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"tau={best:g} macro_f1={dict(scores)[best]:.6f} ({len(scores)} values scored)")
    return 0


def _read_tau_file(path: Path, model_path: Path) -> float:
    """Tau from a ``tune`` result, provided it was tuned for exactly this model file."""
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
    except ValueError as exc:
        raise ValueError(f"tau file {path}: {exc}") from exc
    if data["model_sha256"] != sha256_file(model_path):
        raise ValueError("tau file was tuned for a different model")
    return float(data["tau"])


def cmd_predict(args: argparse.Namespace) -> int:
    """Write thresholded predictions plus the evaluator's sidecar manifest."""
    model_path, input_path = Path(args.model), Path(args.input)
    out, manifest_out = Path(args.out), Path(args.manifest_out)
    _refuse_existing(out, manifest_out)
    model = CRFModel.load_model(model_path)
    tau = _read_tau_file(Path(args.tau_file), model_path)
    observations = read_observations(input_path, Path(args.ids))
    marginals = model.predict_marginals(observations)
    sample_ids = [observation["sample_id"] for observation in observations]
    predictions = _predictions(model.model_name, model.label_mode, sample_ids, marginals, tau)
    write_predictions_jsonl(predictions, out)
    write_prediction_manifest(
        manifest_out,
        predictions_path=out,
        input_path=input_path,
        model_name=_tau_model_name(model.model_name, tau),
        label_mode=model.label_mode,
        ordered_labels=model.vocabulary,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Top-level parser with the ``tune`` and ``predict`` subcommands."""
    parser = argparse.ArgumentParser(
        prog="python scripts/tune_threshold.py",
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


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: 0 on success, 1 with one ``error:`` line on stderr, 2 on usage errors."""
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, OSError) as exc:
        print("error: " + " ".join(str(exc).splitlines()), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
