"""Reproducible command-line interface for CRF training, prediction, and inspection.

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
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib

from punct_restore.features.crf_features import FeatureConfig
from punct_restore.models.crf import (
    CRFModel,
    require_scoring_mode,
)
from punct_restore.paths import REPO_ROOT as ROOT
from punct_restore.prediction_io import (
    check_new_output_paths,
    read_frozen_ids,
    read_observations,
    write_prediction_outputs,
    write_predictions_jsonl,  # noqa: F401 - re-exported: tests call train_crf.write_predictions_jsonl
)


def sha256_file(path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a local file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def get_package_version(pkg_name: str) -> str:
    """Safely obtain installed package version or return 'unknown'."""
    try:
        import importlib.metadata
        return importlib.metadata.version(pkg_name)
    except Exception:
        return "unknown"


def get_process_memory_stats() -> dict[str, float]:
    """Retrieve process memory metrics directly from the OS kernel."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
            GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
                wintypes.DWORD,
            ]
            GetProcessMemoryInfo.restype = wintypes.BOOL

            pmc = PROCESS_MEMORY_COUNTERS_EX()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
            hProcess = ctypes.windll.kernel32.GetCurrentProcess()
            if GetProcessMemoryInfo(hProcess, ctypes.byref(pmc), pmc.cb):
                return {
                    "peak_working_set_mb": round(pmc.PeakWorkingSetSize / (1024 * 1024), 2),
                    "current_working_set_mb": round(pmc.WorkingSetSize / (1024 * 1024), 2),
                    "peak_private_commit_mb": round(pmc.PeakPagefileUsage / (1024 * 1024), 2),
                    "current_private_commit_mb": round(pmc.PrivateUsage / (1024 * 1024), 2),
                }
        except Exception:
            pass
    return {}


def get_fingerprinted_files() -> list[str]:
    """Return deterministically sorted repository-relative POSIX paths of model-defining files."""
    repo_root = ROOT
    files_to_hash: set[str] = set()

    # Core execution entry point, configuration, and dependencies
    essential_files = [
        "src/punct_restore/cli/train_crf.py",
        "scripts/train_crf.py",
        "configs/label_sets.yaml",
        "pyproject.toml",
    ]
    for rel_path in essential_files:
        full_path = repo_root / rel_path
        if not full_path.is_file():
            raise FileNotFoundError(f"Required model-defining file missing for fingerprinting: {rel_path}")
        files_to_hash.add(rel_path)

    # Core package source files
    pkg_dir = repo_root / "src" / "punct_restore"
    if not pkg_dir.is_dir():
        raise FileNotFoundError(f"Source package directory missing: {pkg_dir}")

    for py_file in pkg_dir.rglob("*.py"):
        rel_posix = py_file.relative_to(repo_root).as_posix()
        files_to_hash.add(rel_posix)

    return sorted(files_to_hash)


def compute_code_fingerprint() -> str:
    """Compute deterministic SHA-256 over all model-defining source and configuration files.

    Includes src/punct_restore/cli/train_crf.py and its scripts/train_crf.py stub,
    configs/label_sets.yaml, pyproject.toml, and all Python files under src/punct_restore/.
    Uses repository-relative POSIX paths and hashes full file byte contents.
    """
    repo_root = ROOT
    file_list = get_fingerprinted_files()

    h = hashlib.sha256()
    for rel_posix in file_list:
        content = (repo_root / rel_posix).read_bytes()
        h.update(rel_posix.encode("utf-8") + b"\0" + content)

    return h.hexdigest()


def load_and_filter_training_records(
    input_path: Path,
    ids_path: Path,
    expected_mode: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load JSONL training records filtered strictly by membership in frozen IDs.

    Enforces:
    - Every ID in ids_path exists in input_path.
    - Exact cardinality match between ID list and selected records.
    - Records have split == 'train'.
    - Records have label_mode == expected_mode.
    - Token and label sequence lengths match exactly.
    - Source sequence order is preserved.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL file not found: {input_path}")

    frozen_ids = read_frozen_ids(ids_path)
    target_id_set = set(frozen_ids)

    selected_records: list[dict[str, Any]] = []
    found_ids: set[str] = set()

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue

            try:
                rec = json.loads(line_str)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Corrupt JSON at line {line_num} in {input_path}: {e}"
                ) from e

            if not isinstance(rec, dict):
                raise ValueError(f"Line {line_num} in {input_path} is not a JSON object")

            sample_id = rec.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"Line {line_num} in {input_path} has missing or empty sample_id")

            # Check if this record is selected
            if sample_id in target_id_set:
                if sample_id in found_ids:
                    raise ValueError(
                        f"Duplicate sample_id {sample_id!r} encountered in source file {input_path}"
                    )
                found_ids.add(sample_id)

                # Strict split validation: cannot train on non-train splits
                split = rec.get("split")
                if split != "train":
                    raise ValueError(
                        f"Selected record {sample_id!r} has split={split!r}, expected 'train'. "
                        "Non-train records cannot enter training."
                    )

                # Label mode validation
                mode = rec.get("label_mode")
                if mode != expected_mode:
                    raise ValueError(
                        f"Selected record {sample_id!r} has label_mode={mode!r}, expected {expected_mode!r}."
                    )

                # Token & label alignment validation
                tokens = rec.get("tokens")
                if not isinstance(tokens, list) or not tokens:
                    raise ValueError(f"Selected record {sample_id!r} has empty or non-list tokens")

                punct_labels = rec.get("punct_labels")
                if not isinstance(punct_labels, list) or len(punct_labels) != len(tokens):
                    raise ValueError(
                        f"Selected record {sample_id!r} has mismatched punct_labels length "
                        f"({len(punct_labels) if isinstance(punct_labels, list) else None} vs {len(tokens)})"
                    )

                if expected_mode == "C":
                    case_labels = rec.get("case_labels")
                    if not isinstance(case_labels, list) or len(case_labels) != len(tokens):
                        raise ValueError(
                            f"Selected record {sample_id!r} has mismatched case_labels length "
                            f"({len(case_labels) if isinstance(case_labels, list) else None} vs {len(tokens)})"
                        )

                selected_records.append(rec)

    # Validate that every requested ID was found in the input file
    missing_ids = target_id_set - found_ids
    if missing_ids:
        sample_missing = sorted(missing_ids)[:5]
        raise ValueError(
            f"Missing {len(missing_ids)} selected IDs from input file {input_path}. "
            f"Examples: {sample_missing}"
        )

    if len(selected_records) != len(frozen_ids):
        raise ValueError(
            f"Selected record count ({len(selected_records)}) does not match ID-list cardinality ({len(frozen_ids)})"
        )

    return selected_records, frozen_ids


def recorded_command(argv: Sequence[str]) -> list[str]:
    """The command line to record: the `recap` launcher (an absolute path on Windows) is written as `recap`."""
    if argv and Path(argv[0]).stem == "recap":
        return ["recap", *argv[1:]]
    return list(argv)


def git_status_porcelain() -> str:
    """Return ``git status --porcelain`` of the repository, or ``"unknown"`` if unavailable."""
    repo_root = ROOT
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if result.returncode != 0 or result.stdout is None:
        return "unknown"
    return result.stdout


def cmd_train(args: argparse.Namespace) -> int:
    """Execute CRF training with frozen ID enforcement and provenance recording."""
    input_path = Path(args.input)
    ids_path = Path(args.ids)
    output_path = Path(args.output)

    try:  # even with --overwrite, the manifest must never replace the model or an input
        if not output_path.name:
            raise ValueError(f"--output must name a file, got {args.output!r}")
        manifest_path = Path(args.manifest_out) if args.manifest_out else output_path.with_suffix(".joblib.manifest.json")
        check_new_output_paths({"--output": output_path, "--manifest-out": manifest_path},
                               {"--input": input_path, "--ids": ids_path,
                                "--config-path": Path(args.config_path) if args.config_path else None})
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1

    # Safety: do not silently overwrite existing files
    if output_path.exists() and not args.overwrite:
        sys.stderr.write(
            f"Error: Output artifact already exists at {output_path}. Use --overwrite to replace.\n"
        )
        return 1
    if manifest_path.exists() and not args.overwrite:
        sys.stderr.write(
            f"Error: Manifest already exists at {manifest_path}. Use --overwrite to replace.\n"
        )
        return 1

    mode = args.mode.upper()
    print(f"[RECAP CRF] Loading and verifying training records for Mode {mode}...")
    try:
        records, frozen_ids = load_and_filter_training_records(input_path, ids_path, mode)
    except Exception as e:
        sys.stderr.write(f"Error loading training data: {e}\n")
        return 1

    total_tokens = sum(len(r["tokens"]) for r in records)
    print(f"[RECAP CRF] Verified {len(records):,} sequences ({total_tokens:,} tokens) strictly matching {ids_path.name}.")

    # Configure features
    if args.preset == "improved":
        feature_config = FeatureConfig.improved(
            include_utterance_boundaries=not args.no_utterance_boundaries,
            window_size=args.window_size or 2,
        )
    else:
        feature_config = FeatureConfig.basic(
            include_utterance_boundaries=not args.no_utterance_boundaries,
            window_size=args.window_size or 1,
        )

    print(f"[RECAP CRF] Configuring CRFModel (algorithm={args.algorithm}, c1={args.c1}, c2={args.c2}, max_iter={args.max_iterations})...")
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
    except Exception as e:
        sys.stderr.write(f"Configuration error: {e}\n")
        return 1

    provenance: dict[str, Any] = {
        "dataset_release": args.dataset_release or "unknown",
        "policy_version": args.policy_version or "unknown",
        "run_id": args.run_id or "unnamed_run",
        "input_file": input_path.as_posix(),
        "input_sha256": sha256_file(input_path),
        "ids_file": ids_path.as_posix(),
        "ids_sha256": sha256_file(ids_path),
        "selected_sequences": len(records),
        "selected_tokens": total_tokens,
    }

    start_time = time.perf_counter()
    print("[RECAP CRF] Training model...")
    try:
        model.fit(records, config={"provenance": provenance})
    except Exception as e:
        sys.stderr.write(f"Training failed: {e}\n")
        return 1
    duration = time.perf_counter() - start_time
    print(f"[RECAP CRF] Training completed in {duration:.2f} seconds.")

    # Collect OS-level process memory diagnostics
    mem_stats = get_process_memory_stats()
    if mem_stats:
        print(
            f"[RECAP CRF] Process Memory (Kernel): Peak Working Set = {mem_stats['peak_working_set_mb']} MB, "
            f"Peak Private Commit = {mem_stats['peak_private_commit_mb']} MB"
        )

    # Collect optimizer convergence diagnostics
    training_diagnostics: dict[str, Any] = {}
    if hasattr(model, "_backend") and hasattr(model._backend, "training_log_"):
        tl = getattr(model._backend, "training_log_", None)
        if hasattr(tl, "iterations"):
            training_diagnostics["num_iterations"] = len(tl.iterations)
        if hasattr(tl, "lines") and tl.lines:
            for line in reversed(tl.lines):
                s = line.strip()
                if "terminated" in s.lower() or "converged" in s.lower():
                    training_diagnostics["termination_status"] = s
                    break
        if "termination_status" in training_diagnostics:
            print(
                f"[RECAP CRF] Optimization Status: {training_diagnostics['termination_status']} "
                f"({training_diagnostics.get('num_iterations', 'unknown')} iterations)"
            )

    # Save model artifact
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(output_path)
    artifact_hash = sha256_file(output_path)
    print(f"[RECAP CRF] Saved artifact: {output_path} (SHA-256: {artifact_hash})")

    # Generate comprehensive sidecar manifest
    sidecar_manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": args.run_id or output_path.stem,
        "label_mode": mode,
        "model_name": model.model_name,
        "vocabulary": model.vocabulary,
        "feature_preset": args.preset,
        "feature_preset_version": feature_config.version,
        "feature_config": feature_config.to_dict(),
        "hyperparameters": {
            "algorithm": args.algorithm,
            "c1": args.c1,
            "c2": args.c2,
            "max_iterations": args.max_iterations,
            "all_possible_transitions": True,
        },
        "training_diagnostics": training_diagnostics,
        "memory_diagnostics": mem_stats,
        "provenance": provenance,
        "training_command": recorded_command(sys.argv),
        "training_duration_seconds": round(duration, 3),
        "artifact": {
            "path": output_path.as_posix(),
            "sha256": artifact_hash,
            "size_bytes": output_path.stat().st_size,
        },
        "backend": {
            "name": "sklearn-crfsuite (python-crfsuite)",
            "sklearn_crfsuite_version": get_package_version("sklearn-crfsuite"),
            "python_crfsuite_version": get_package_version("python-crfsuite"),
        },
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "joblib_version": get_package_version("joblib"),
            "numpy_version": get_package_version("numpy"),
            "scipy_version": get_package_version("scipy"),
            "recap_version": get_package_version("recap-nlp"),
        },
        "code_fingerprint": compute_code_fingerprint(),
        "fingerprinted_files": get_fingerprinted_files(),
        "reproducibility_warning": git_status_porcelain(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(sidecar_manifest, f, indent=2, sort_keys=True)
    print(f"[RECAP CRF] Saved manifest: {manifest_path}")

    return 0


def _predict_path_error(args: argparse.Namespace, manifest_path: Path | None) -> str | None:
    """Return a one-line error for missing, unreadable or clashing paths, else None."""
    model_path, input_path, output_path = Path(args.model), Path(args.input), Path(args.output)
    try:
        if not model_path.is_file():
            return f"Error: Model artifact not found at {model_path}"
        if not input_path.is_file():
            return f"Error: Input JSONL not found at {input_path}"
        check_new_output_paths(
            {"--output": output_path, "--manifest-out": manifest_path},
            {"--input": input_path, "--model": model_path,
             "--ids": Path(args.ids) if args.ids else None},
        )
        if output_path.exists() and not args.overwrite:
            return f"Error: Output file already exists at {output_path}. Use --overwrite to replace."
        if manifest_path is not None and manifest_path.exists() and not args.overwrite:
            return f"Error: Manifest already exists at {manifest_path}. Use --overwrite to replace."
    except ValueError as e:
        return f"Error: {e}"
    except OSError as e:  # e.g. a folder on the way cannot be read
        shown = Path(e.filename).name if e.filename else "a path"
        return f"Error: cannot check {shown}: {e.strerror or e}"
    return None


def _export_predictions(
    model: CRFModel,
    predictions: Sequence[Mapping[str, Any]],
    input_path: Path,
    output_path: Path,
    manifest_path: Path | None,
) -> None:
    """Write the predictions JSONL and, when requested, its sidecar manifest; report both after."""
    write_prediction_outputs(
        predictions,
        output_path,
        manifest_path,
        input_path=input_path,
        model_name=model.model_name,
        label_mode=model.label_mode,
        ordered_labels=model.vocabulary,
    )
    print(f"[RECAP CRF] Predictions exported to {output_path}")
    if manifest_path is not None:
        print(f"[RECAP CRF] Manifest written to {manifest_path}")


def cmd_predict(args: argparse.Namespace) -> int:
    """Generate contract-compliant predictions from a saved CRF model artifact."""
    model_path, input_path, output_path = Path(args.model), Path(args.input), Path(args.output)
    manifest_path = Path(args.manifest_out) if args.manifest_out else None

    path_error = _predict_path_error(args, manifest_path)
    if path_error:
        sys.stderr.write(path_error + "\n")
        return 1

    print(f"[RECAP CRF] Loading model from {model_path}...")
    try:
        model = CRFModel.load_model(model_path)
    except Exception as e:
        sys.stderr.write(f"Failed to load model: {e}\n")
        return 1
    try:
        require_scoring_mode(model.label_mode)  # a CASE helper artifact loads fine but must not be scored here
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1

    ids_path = Path(args.ids) if args.ids else None
    try:
        records_to_predict = read_observations(input_path, ids_path)
    except (OSError, ValueError) as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1
    if ids_path is not None:
        print(f"[RECAP CRF] Restricting inference to {len(records_to_predict):,} IDs from {ids_path.name}...")

    print(f"[RECAP CRF] Running batch inference on {len(records_to_predict):,} sequences...")
    start_time = time.perf_counter()
    try:
        predictions = model.predict(records_to_predict)
    except Exception as e:
        sys.stderr.write(f"Prediction failed: {e}\n")
        return 1
    duration = time.perf_counter() - start_time
    total_tokens = sum(len(p["predicted_labels"]) for p in predictions)
    tok_per_sec = total_tokens / duration if duration > 0 else 0.0
    print(f"[RECAP CRF] Inferred {len(predictions):,} sequences ({total_tokens:,} tokens) in {duration:.2f}s ({tok_per_sec:.1f} tok/s).")

    try:
        _export_predictions(model, predictions, input_path, output_path, manifest_path)
    except (OSError, ValueError) as e:
        sys.stderr.write(f"Error: Could not write prediction outputs: {e}\n")
        return 1
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect and display metadata of a trained CRF model artifact."""
    model_path = Path(args.model)
    if not model_path.is_file():
        sys.stderr.write(f"Error: Artifact not found at {model_path}\n")
        return 1

    try:
        bundle = joblib.load(model_path)
    except Exception as e:
        sys.stderr.write(f"Failed to read artifact: {e}\n")
        return 1

    if not isinstance(bundle, dict):
        sys.stderr.write(f"Invalid artifact format: root is {type(bundle).__name__}\n")
        return 1

    print("=" * 70)
    print(f"RECAP CRF Model Artifact: {model_path.name}")
    print("=" * 70)
    print(f"Schema Version:     {bundle.get('schema_version')}")
    print(f"Model Name:         {bundle.get('model_name')}")
    print(f"Label Mode:         {bundle.get('label_mode')}")
    print(f"Vocabulary Size:    {len(bundle.get('vocabulary', []))} classes")
    print(f"Vocabulary:         {bundle.get('vocabulary')}")
    print("\nHyperparameters:")
    for k, v in bundle.get("hyperparameters", {}).items():
        print(f"  {k:24}: {v}")
    print("\nFeature Configuration:")
    for k, v in bundle.get("feature_config", {}).items():
        print(f"  {k:24}: {v}")
    print("\nTraining Metadata:")
    for k, v in bundle.get("training_metadata", {}).items():
        print(f"  {k:24}: {v}")
    print("\nEnvironment:")
    for k, v in bundle.get("environment", {}).items():
        print(f"  {k:24}: {v}")

    # Display companion manifest if present
    manifest_candidates = [
        model_path.with_name(f"{model_path.name}.manifest.json"),
        model_path.with_suffix(".joblib.manifest.json"),
        model_path.with_suffix(".manifest.json"),
    ]
    for mc in manifest_candidates:
        if mc.is_file():
            try:
                with open(mc, "r", encoding="utf-8") as f:
                    m_data = json.load(f)
                print("\nCompanion Manifest:")
                print(f"  Run ID:                 {m_data.get('run_id')}")
                print(f"  Dataset Release:        {m_data.get('provenance', {}).get('dataset_release')}")
                print(f"  Policy Version:         {m_data.get('provenance', {}).get('policy_version')}")
                print(f"  Code Fingerprint:       {m_data.get('code_fingerprint')}")
                print(f"  Fingerprinted Files:    {len(m_data.get('fingerprinted_files', []))} files")
                print(f"  Training Duration:      {m_data.get('training_duration_seconds')}s")
                print(f"  Reproducibility Note:   {m_data.get('reproducibility_warning')}")
                break
            except Exception:
                pass

    print("=" * 70)
    return 0


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="recap train-crf",
        description="RECAP Project 2 — Conditional Random Field (CRF) Training, Inference & Inspection CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: train
    p_train = subparsers.add_parser("train", help="Train a CRF model on frozen training records.")
    p_train.add_argument("--mode", "-m", choices=["A", "B", "C"], required=True, help="Target label mode (A, B, or C).")
    p_train.add_argument("--input", "-i", required=True, help="Path to input JSONL dataset file.")
    p_train.add_argument("--ids", required=True, help="Path to official frozen training ID list (.ids file).")
    p_train.add_argument("--output", "-o", required=True, help="Path to save the trained model (.joblib).")
    p_train.add_argument("--preset", "-p", choices=["basic", "improved"], default="basic", help="Feature preset (default: basic).")
    p_train.add_argument("--algorithm", default="lbfgs", help="CRF optimization algorithm (default: lbfgs).")
    p_train.add_argument("--c1", type=float, default=0.1, help="L1 regularization coefficient (default: 0.1).")
    p_train.add_argument("--c2", type=float, default=0.1, help="L2 regularization coefficient (default: 0.1).")
    p_train.add_argument("--max-iterations", type=int, default=100, help="Maximum training iterations (default: 100).")
    p_train.add_argument("--window-size", type=int, default=None, help="Context window radius override.")
    p_train.add_argument("--no-utterance-boundaries", action="store_true", help="Ablate BOS/EOS boundary features.")
    p_train.add_argument("--run-id", default=None, help="Identifier for experiment tracking.")
    p_train.add_argument("--model-name", default=None, help="Model name identifier emitted in prediction JSONL (default: crf_basic or crf_improved).")
    p_train.add_argument("--dataset-release", default="sponspeech-m1-v2", help="Dataset release ID (default: sponspeech-m1-v2).")
    p_train.add_argument("--policy-version", default="m1-v2", help="Annotation policy version (default: m1-v2).")
    p_train.add_argument("--config-path", default=None, help="Path to custom configs/label_sets.yaml.")
    p_train.add_argument("--manifest-out", default=None, help="Optional custom path for output manifest JSON.")
    p_train.add_argument("--overwrite", action="store_true", help="Overwrite existing output artifact and manifest.")

    # Subcommand: predict
    p_pred = subparsers.add_parser("predict", help="Generate predictions using a trained CRF model artifact.")
    p_pred.add_argument("--model", "-m", required=True, help="Path to trained model artifact (.joblib).")
    p_pred.add_argument("--input", "-i", required=True, help="Path to input JSONL records.")
    p_pred.add_argument("--output", "-o", required=True, help="Path to write prediction JSONL.")
    p_pred.add_argument("--ids", default=None, help="Optional frozen evaluation ID list to filter input records.")
    p_pred.add_argument("--manifest-out", default=None, help="Optional path for the prediction sidecar manifest JSON (checked by the evaluator).")
    p_pred.add_argument("--overwrite", action="store_true", help="Overwrite existing output prediction file and manifest.")

    # Subcommand: inspect
    p_insp = subparsers.add_parser("inspect", help="Inspect metadata of a saved model artifact.")
    p_insp.add_argument("--model", "-m", required=True, help="Path to trained model artifact (.joblib).")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    # A path the console code page cannot encode must not crash the run after files are written.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(errors="backslashreplace")
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "train":
        return cmd_train(args)
    if args.subcommand == "predict":
        return cmd_predict(args)
    if args.subcommand == "inspect":
        return cmd_inspect(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
