"""Train a CRF model, predict with it, or inspect a saved model artifact.

Owned by Member 3 (Omar Alzyod).
Supports:
- train: Train a linear-chain CRF model with frozen ID filtering, hyperparameter tuning,
         and comprehensive provenance manifest generation.
- predict: Export gold-free predictions conforming to docs/MODEL_CONTRACT.md.
- inspect: Inspect metadata and hyperparameters of saved model artifacts.

Adheres strictly to docs/DATA_CONTRACT.md, docs/MODEL_CONTRACT.md, and docs/DECISIONS.md.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from punct_restore.features.crf_features import FeatureConfig
from punct_restore.hashing import sha256_file
from punct_restore.models.crf import CRFModel
from punct_restore.models.crf_inspection import RULE, inspect_artifact
from punct_restore.models.crf_prediction import (
    PredictionFailed,
    check_predict_paths,
    inference_summary,
    load_scoring_model,
    predict_records,
    read_input_records,
    write_model_predictions,
)
from punct_restore.models.crf_training import (
    TrainingFailed,
    TrainingRun,
    TrainingSource,
    build_sidecar_manifest,
    build_training_provenance,
    collect_training_diagnostics,
    process_memory,
)
from punct_restore.models.crf_training_data import load_and_filter_training_records
from punct_restore.prediction_io import (
    check_new_output_paths,
    one_line_error,
    one_line_error_naming,
    reconfigure_stdout,
)

__all__ = ["main"]


def recorded_command(argv: Sequence[str]) -> list[str]:
    """The command line to record: the `recap` launcher (an absolute path on Windows) is written as `recap`."""
    if argv and Path(argv[0]).stem == "recap":
        return ["recap", *argv[1:]]
    return list(argv)


def _error(message: str) -> None:
    """Write the one user-facing error line (no folder, no stack trace) to stderr."""
    sys.stderr.write(f"error: {message}\n")


def _prepare_train_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """Check the input and output paths; return (input, ids, output, manifest). TrainingFailed says why not."""
    input_path = Path(args.input)
    ids_path = Path(args.ids)
    output_path = Path(args.output)

    try:  # even with --overwrite, the manifest must never replace the model or an input
        if not output_path.name:
            raise ValueError(f"--output must name a file, got {args.output!r}")
        manifest_path = (
            Path(args.manifest_out) if args.manifest_out else output_path.with_suffix(".joblib.manifest.json")
        )
        check_new_output_paths({"--output": output_path, "--manifest-out": manifest_path},
                               {"--input": input_path, "--ids": ids_path,
                                "--config-path": Path(args.config_path) if args.config_path else None})
    except ValueError as e:
        raise TrainingFailed(str(e)) from e

    # Safety: do not silently overwrite existing files
    if output_path.exists() and not args.overwrite:
        raise TrainingFailed(f"{output_path.name}: output artifact already exists; use --overwrite to replace")
    if manifest_path.exists() and not args.overwrite:
        raise TrainingFailed(f"{manifest_path.name}: manifest already exists; use --overwrite to replace")
    return input_path, ids_path, output_path, manifest_path


def _load_records(input_path: Path, ids_path: Path, mode: str) -> tuple[list[dict[str, Any]], int]:
    """Load and verify the frozen training records; return (records, token count). TrainingFailed says why not."""
    print(f"[RECAP CRF] Loading and verifying training records for Mode {mode}...")
    try:
        records, frozen_ids = load_and_filter_training_records(input_path, ids_path, mode)
    except (OSError, ValueError) as e:  # a missing or unreadable file, or a record that breaks the frozen-data rules
        raise TrainingFailed(one_line_error_naming(e, input_path, ids_path)) from e

    total_tokens = sum(len(r["tokens"]) for r in records)
    print(
        f"[RECAP CRF] Verified {len(records):,} sequences ({total_tokens:,} tokens) strictly matching {ids_path.name}."
    )
    return records, total_tokens


def _feature_config(args: argparse.Namespace) -> FeatureConfig:
    """Build the feature configuration of the requested preset."""
    if args.preset == "improved":
        return FeatureConfig.improved(
            include_utterance_boundaries=not args.no_utterance_boundaries,
            window_size=args.window_size or 2,
        )
    return FeatureConfig.basic(
        include_utterance_boundaries=not args.no_utterance_boundaries,
        window_size=args.window_size or 1,
    )


def _configure_model(args: argparse.Namespace, mode: str) -> tuple[CRFModel, FeatureConfig]:
    """Configure the CRF model; return (model, feature config). TrainingFailed says why not."""
    feature_config = _feature_config(args)

    print(
        f"[RECAP CRF] Configuring CRFModel (algorithm={args.algorithm}, c1={args.c1}, c2={args.c2}, "
        f"max_iter={args.max_iterations})..."
    )
    try:
        feature_model_name = getattr(args, "model_name", None) or f"crf_{args.preset}"
        model = CRFModel(
            label_mode=mode,
            algorithm=args.algorithm,
            c1=args.c1,
            c2=args.c2,
            max_iterations=args.max_iterations,
            all_possible_transitions=True,
            feature_config=feature_config,
            model_name=feature_model_name,
            config_path=args.config_path,
        )
    except (OSError, ValueError) as e:  # a bad hyperparameter, or a --config-path file that cannot be read
        raise TrainingFailed(one_line_error_naming(e, args.config_path)) from e
    return model, feature_config


def _fit_model(model: CRFModel, records: list[dict[str, Any]], provenance: dict[str, Any]) -> float:
    """Fit the model; return the training duration in seconds. TrainingFailed says why a fit failed."""
    start_time = time.perf_counter()
    print("[RECAP CRF] Training model...")
    try:
        model.fit(records, config={"provenance": provenance})
    except (OSError, ValueError) as e:  # the backend refuses its settings or the records, or has no disk for its files
        raise TrainingFailed(f"training failed: {one_line_error(e)}") from e
    duration = time.perf_counter() - start_time
    print(f"[RECAP CRF] Training completed in {duration:.2f} seconds.")
    return duration


def _report_memory() -> tuple[dict[str, float] | None, str | None]:
    """Collect the OS-level process memory diagnostics: print the peak figures, or one note saying why not."""
    mem_stats, unavailable = process_memory()
    if mem_stats is None:
        print(f"[RECAP CRF] Note: process memory figures are unavailable ({unavailable}); the manifest records none.")
    else:
        print(
            f"[RECAP CRF] Process Memory (Kernel): Peak Working Set = {mem_stats['peak_working_set_mb']} MB, "
            f"Peak Private Commit = {mem_stats['peak_private_commit_mb']} MB"
        )
    return mem_stats, unavailable


def _report_optimizer(model: CRFModel) -> dict[str, Any]:
    """Collect the optimizer convergence diagnostics and print the termination status when known."""
    training_diagnostics = collect_training_diagnostics(model)
    if "termination_status" in training_diagnostics:
        print(
            f"[RECAP CRF] Optimization Status: {training_diagnostics['termination_status']} "
            f"({training_diagnostics.get('num_iterations', 'unknown')} iterations)"
        )
    return training_diagnostics


def _save_artifact(model: CRFModel, output_path: Path) -> str:
    """Save the model artifact and return its SHA-256; TrainingFailed says why it could not be saved.

    A failure deletes the half-written file only when this run created it. With ``--overwrite`` the file was already
    there: it is left in place (the failed write may have cut it short, but it is never deleted).
    """
    created_here = not os.path.lexists(output_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(output_path)
        artifact_hash = sha256_file(output_path)
    except OSError as e:
        if created_here:
            with contextlib.suppress(OSError):  # best effort; the save error is the one reported
                output_path.unlink(missing_ok=True)
        raise TrainingFailed(one_line_error(e)) from e
    print(f"[RECAP CRF] Saved artifact: {output_path} (SHA-256: {artifact_hash})")
    return artifact_hash


def _write_manifest(sidecar_manifest: Mapping[str, Any], manifest_path: Path) -> None:
    """Write the sidecar manifest JSON (sorted keys) and report where; TrainingFailed says why it failed."""
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(sidecar_manifest, f, indent=2, sort_keys=True)
    except OSError as e:
        raise TrainingFailed(one_line_error(e)) from e
    print(f"[RECAP CRF] Saved manifest: {manifest_path}")


def _hyperparameters(args: argparse.Namespace) -> dict[str, Any]:
    """The optimizer hyperparameters recorded in the sidecar manifest."""
    return {
        "algorithm": args.algorithm,
        "c1": args.c1,
        "c2": args.c2,
        "max_iterations": args.max_iterations,
        "all_possible_transitions": True,
    }


def _train(args: argparse.Namespace) -> None:
    """Run the training steps in order; TrainingFailed from any step stops the run."""
    input_path, ids_path, output_path, manifest_path = _prepare_train_paths(args)
    mode = args.mode.upper()
    records, total_tokens = _load_records(input_path, ids_path, mode)
    model, feature_config = _configure_model(args, mode)
    provenance = build_training_provenance(TrainingSource(
        input_path=input_path, ids_path=ids_path, num_sequences=len(records), num_tokens=total_tokens,
        dataset_release=args.dataset_release, policy_version=args.policy_version, run_id=args.run_id,
    ))
    duration = _fit_model(model, records, provenance)
    mem_stats, memory_unavailable = _report_memory()
    training_diagnostics = _report_optimizer(model)
    artifact_hash = _save_artifact(model, output_path)

    # Generate comprehensive sidecar manifest
    sidecar_manifest = build_sidecar_manifest(TrainingRun(
        run_id=args.run_id or output_path.stem, mode=mode, model=model, preset=args.preset,
        feature_config=feature_config, hyperparameters=_hyperparameters(args),
        training_diagnostics=training_diagnostics, memory_diagnostics=mem_stats, provenance=provenance,
        training_command=recorded_command(sys.argv), duration=duration, output_path=output_path,
        artifact_hash=artifact_hash, memory_unavailable=memory_unavailable,
    ))
    _write_manifest(sidecar_manifest, manifest_path)


def cmd_train(args: argparse.Namespace) -> int:
    """Execute CRF training with frozen ID enforcement and provenance recording; 1 after one `error:` line."""
    try:
        _train(args)
    except TrainingFailed as e:
        _error(str(e))
        return 1
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Generate contract-compliant predictions from a saved CRF model artifact."""
    model_path, input_path, output_path = Path(args.model), Path(args.input), Path(args.output)
    manifest_path = Path(args.manifest_out) if args.manifest_out else None
    ids_path = Path(args.ids) if args.ids else None
    try:
        check_predict_paths(model_path, input_path, output_path, manifest_path, ids_path, args.overwrite)
        print(f"[RECAP CRF] Loading model from {model_path}...")
        model = load_scoring_model(model_path)
        records = read_input_records(input_path, ids_path)
        if ids_path is not None:
            print(f"[RECAP CRF] Restricting inference to {len(records):,} IDs from {ids_path.name}...")
        print(f"[RECAP CRF] Running batch inference on {len(records):,} sequences...")
        start_time = time.perf_counter()
        predictions = predict_records(model, records)
        print(f"[RECAP CRF] {inference_summary(predictions, time.perf_counter() - start_time)}")
        write_model_predictions(model, predictions, input_path, output_path, manifest_path)
    except PredictionFailed as e:
        _error(str(e))
        return 1
    print(f"[RECAP CRF] Predictions exported to {output_path}")
    if manifest_path is not None:
        print(f"[RECAP CRF] Manifest written to {manifest_path}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect and display metadata of a trained CRF model artifact."""
    try:
        report = inspect_artifact(Path(args.model))
    except (OSError, ValueError) as e:  # a missing artifact or a folder (named by name only), or one that is unreadable
        _error(one_line_error(e))
        return 1
    for line in report.summary:
        print(line)
    for note in report.notes:
        sys.stderr.write(note + "\n")
    for line in report.manifest:
        print(line)
    print(RULE)
    return 0


def _add_train_arguments(p_train: argparse.ArgumentParser) -> None:
    """Options of the `train` subcommand."""
    p_train.add_argument("--mode", "-m", choices=["A", "B", "C"], required=True, help="Target label mode (A, B, or C).")
    p_train.add_argument("--input", "-i", required=True, help="Path to input JSONL dataset file.")
    p_train.add_argument("--ids", required=True, help="Path to official frozen training ID list (.ids file).")
    p_train.add_argument("--output", "-o", required=True, help="Path to save the trained model (.joblib).")
    p_train.add_argument(
        "--preset", "-p", choices=["basic", "improved"], default="basic", help="Feature preset (default: basic)."
    )
    p_train.add_argument("--algorithm", default="lbfgs", help="CRF optimization algorithm (default: lbfgs).")
    p_train.add_argument("--c1", type=float, default=0.1, help="L1 regularization coefficient (default: 0.1).")
    p_train.add_argument("--c2", type=float, default=0.1, help="L2 regularization coefficient (default: 0.1).")
    p_train.add_argument("--max-iterations", type=int, default=100, help="Maximum training iterations (default: 100).")
    p_train.add_argument("--window-size", type=int, default=None, help="Context window radius override.")
    p_train.add_argument("--no-utterance-boundaries", action="store_true", help="Ablate BOS/EOS boundary features.")
    p_train.add_argument("--run-id", default=None, help="Identifier for experiment tracking.")
    p_train.add_argument(
        "--model-name",
        default=None,
        help="Model name identifier emitted in prediction JSONL (default: crf_basic or crf_improved).",
    )
    p_train.add_argument(
        "--dataset-release", default="sponspeech-m1-v2", help="Dataset release ID (default: sponspeech-m1-v2)."
    )
    p_train.add_argument("--policy-version", default="m1-v2", help="Annotation policy version (default: m1-v2).")
    p_train.add_argument("--config-path", default=None, help="Path to custom configs/label_sets.yaml.")
    p_train.add_argument("--manifest-out", default=None, help="Optional custom path for output manifest JSON.")
    p_train.add_argument("--overwrite", action="store_true", help="Overwrite existing output artifact and manifest.")


def _add_predict_arguments(p_pred: argparse.ArgumentParser) -> None:
    """Options of the `predict` subcommand."""
    p_pred.add_argument("--model", "-m", required=True, help="Path to trained model artifact (.joblib).")
    p_pred.add_argument("--input", "-i", required=True, help="Path to input JSONL records.")
    p_pred.add_argument("--output", "-o", required=True, help="Path to write prediction JSONL.")
    p_pred.add_argument("--ids", default=None, help="Optional frozen evaluation ID list to filter input records.")
    p_pred.add_argument(
        "--manifest-out",
        default=None,
        help="Optional path for the prediction sidecar manifest JSON (checked by the evaluator).",
    )
    p_pred.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output prediction file and manifest."
    )


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="recap train-crf",
        description="RECAP Project 2 — Conditional Random Field (CRF) Training, Inference & Inspection CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: train
    p_train = subparsers.add_parser("train", help="Train a CRF model on frozen training records.")
    _add_train_arguments(p_train)
    p_train.set_defaults(handler=cmd_train)

    # Subcommand: predict
    p_pred = subparsers.add_parser("predict", help="Generate predictions using a trained CRF model artifact.")
    _add_predict_arguments(p_pred)
    p_pred.set_defaults(handler=cmd_predict)

    # Subcommand: inspect
    p_insp = subparsers.add_parser("inspect", help="Inspect metadata of a saved model artifact.")
    p_insp.add_argument("--model", "-m", required=True, help="Path to trained model artifact (.joblib).")
    p_insp.set_defaults(handler=cmd_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    # A path the console code page cannot encode must not crash the run after files are written.
    reconfigure_stdout()
    args = build_cli_parser().parse_args(argv)
    return args.handler(args)  # every subcommand sets its handler, and the subparsers are required


if __name__ == "__main__":
    sys.exit(main())
