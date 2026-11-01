"""Time HMM or CRF prediction (and HMM fitting) on this machine under identical conditions.

Usage: recap time-models --kind hmm|crf --model MODEL --mode A|B|C --out OUT_JSON
           [--data-root DIR] [--release sponspeech-m1-v2]
           [--fit-train TRAIN_JSONL --fit-ids IDS --smoothing S --min-token-count K]   (HMM fit timing)
Prediction input: the primary validation records of the mode in DIR/processed/RELEASE (same IDs for both
models). One warm-up pass on 200 records, then 3 timed passes over all records; the median is reported.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from punct_restore.paths import REPO_ROOT as ROOT
from punct_restore.prediction_io import read_ids
from punct_restore.reporting._cli import check_out, name_only, run, write_new_text

REPS, WARMUP = 3, 200
FIT_OPTIONS = ("fit_train", "fit_ids", "smoothing", "min_token_count")


def usable_tokens(tokens: Any) -> bool:
    return isinstance(tokens, list) and bool(tokens) and all(isinstance(t, str) and t for t in tokens)


def records(path: str | Path, ids_path: str | Path) -> list[dict[str, Any]]:
    keep = set(read_ids(Path(ids_path)))
    try:
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    except ValueError as exc:
        raise ValueError(f"{Path(path).name}: {exc}") from exc
    if any(not isinstance(r, dict) or not isinstance(r.get("sample_id"), str) for r in rows):
        raise ValueError(f"{Path(path).name}: every record must be an object with a sample_id")
    if any(not usable_tokens(r.get("tokens")) for r in rows):
        raise ValueError(f"{Path(path).name}: every record needs a non-empty tokens list")
    return [r for r in rows if r["sample_id"] in keep]


def model_class(kind: str) -> Any:
    if kind == "hmm":
        from punct_restore.models.hmm import HMMModel
        return HMMModel
    from punct_restore.models.crf import CRFModel
    return CRFModel


def load_model(model_cls: Any, path: str) -> Any:
    model_path = Path(path)
    if model_path.is_dir():
        raise ValueError(f"{model_path.name}: is a folder, not a file")
    if not model_path.is_file():
        raise ValueError(f"{model_path.name}: model file not found")
    try:
        return model_cls.load_model(path)
    except (OSError, ValueError) as exc:  # the loaders quote the whole path: keep the file name only
        raise ValueError(f"{model_path.name}: {name_only(str(exc), model_path)}") from exc


def time_predictions(model: Any, recs: list[dict[str, Any]]) -> list[float]:
    model.predict(recs[:WARMUP])
    passes = []
    for _ in range(REPS):
        start = time.perf_counter()
        model.predict(recs)
        passes.append(time.perf_counter() - start)
    return passes


def time_fit(model_cls: Any, model: Any, args: argparse.Namespace) -> dict[str, Any]:
    train = records(args.fit_train, args.fit_ids)
    fresh = model_cls(label_mode=args.mode, labels=model.labels, smoothing=args.smoothing,
                      min_token_count=args.min_token_count)
    start = time.perf_counter()
    fresh.fit(train)
    return {"fit_seconds": time.perf_counter() - start, "fit_sequences": len(train)}


def measure(args: argparse.Namespace) -> dict[str, Any]:
    model_cls = model_class(args.kind)
    start = time.perf_counter()
    model = load_model(model_cls, args.model)
    load_s = time.perf_counter() - start
    proc, spl = args.data_root / "processed" / args.release, args.data_root / "splits" / args.release
    ids = spl / f"primary_eval/validation_mode_{args.mode}.ids"
    recs = records(proc / f"mode_{args.mode}/validation.jsonl", ids)
    if not recs:
        raise ValueError(f"{ids.name}: no validation record matches these ids")
    tokens = sum(len(r["tokens"]) for r in recs)
    passes = time_predictions(model, recs)
    median = statistics.median(passes)
    out = {"kind": args.kind, "mode": args.mode, "model": Path(args.model).name, "load_seconds": load_s,
           "sequences": len(recs), "tokens": tokens, "predict_seconds": passes, "median_seconds": median,
           "tokens_per_second": tokens / median, "ms_per_sequence": 1000 * median / len(recs),
           "hardware": {"processor": platform.processor(), "python": platform.python_version()}}
    if args.fit_train:  # HMM only: the CRF fit time comes from its run.json, measured on this machine
        out.update(time_fit(model_cls, model, args))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recap time-models", description="Time HMM or CRF prediction (and HMM fitting).")
    parser.add_argument("--kind", choices=("hmm", "crf"), required=True)
    parser.add_argument("--model", required=True, help="saved model: HMM .json or CRF .joblib")
    parser.add_argument("--mode", choices=("A", "B", "C"), required=True)
    parser.add_argument("--out", required=True, help="JSON file to create (must not exist)")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--release", default="sponspeech-m1-v2")
    parser.add_argument("--fit-train", help="HMM only: training records to time a fit on")
    parser.add_argument("--fit-ids", help="IDs of the training records to fit on")
    parser.add_argument("--smoothing", type=float)
    parser.add_argument("--min-token-count", type=int)
    return parser


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.fit_train and args.kind != "hmm":
        parser.error("--fit-train times an HMM fit, so it needs --kind hmm")
    if args.fit_train and any(getattr(args, name) is None for name in FIT_OPTIONS):
        parser.error("--fit-train needs --fit-ids, --smoothing and --min-token-count")
    return args


def write_report(args: argparse.Namespace) -> None:
    fit_files = {"--fit-train": args.fit_train, "--fit-ids": args.fit_ids}
    inputs = {"--model": Path(args.model)} | {flag: Path(name) for flag, name in fit_files.items() if name}
    out = Path(args.out)
    check_out(out, inputs)  # before loading or timing anything
    result = measure(args)
    write_new_text(out, json.dumps(result, indent=1) + "\n")
    print(f"{args.kind} {args.mode}: {result['tokens_per_second']:,.0f} tok/s, fit {result.get('fit_seconds', '-')}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(lambda: write_report(args), json_errors=False)  # it runs model code: no input-JSON relabelling


if __name__ == "__main__":
    sys.exit(main())
