"""CRF training support that is not CLI-specific: frozen-record loading, code fingerprint and manifest building.

Used by punct_restore.cli.train_crf, which keeps the command-line handling (arguments, console output, exit codes).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from punct_restore.features.crf_features import FeatureConfig
from punct_restore.models.crf import CRFModel
from punct_restore.paths import REPO_ROOT as ROOT
from punct_restore.prediction_io import read_frozen_ids


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
        return importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
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
        except (OSError, AttributeError):
            # Memory figures are optional diagnostics: a Windows without psapi (OSError) or a ctypes build without
            # windll (AttributeError) just reports none; any other error is a bug and propagates.
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


def _parse_record_line(line_str: str, line_num: int, input_path: Path) -> dict[str, Any]:
    """Parse one non-blank JSONL line into a record that has a non-empty string sample_id."""
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
    return rec


def _check_selected_record(rec: dict[str, Any], sample_id: str, expected_mode: str) -> None:
    """Raise ValueError unless a selected record is a train record of the expected mode with aligned labels."""
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


def _scan_selected_records(
    input_path: Path,
    target_id_set: set[str],
    expected_mode: str,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Read the JSONL file once; return the validated records whose sample_id is selected and the IDs found."""
    selected_records: list[dict[str, Any]] = []
    found_ids: set[str] = set()

    with open(input_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue

            rec = _parse_record_line(line_str, line_num, input_path)
            sample_id = rec.get("sample_id")

            # Check if this record is selected
            if sample_id in target_id_set:
                if sample_id in found_ids:
                    raise ValueError(
                        f"Duplicate sample_id {sample_id!r} encountered in source file {input_path}"
                    )
                found_ids.add(sample_id)
                _check_selected_record(rec, sample_id, expected_mode)
                selected_records.append(rec)

    return selected_records, found_ids


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

    selected_records, found_ids = _scan_selected_records(input_path, target_id_set, expected_mode)

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


def collect_training_diagnostics(model: CRFModel) -> dict[str, Any]:
    """Collect optimizer convergence diagnostics (iteration count, termination status) from a fitted model."""
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
    return training_diagnostics


def build_training_provenance(
    *,
    dataset_release: str | None,
    policy_version: str | None,
    run_id: str | None,
    input_path: Path,
    ids_path: Path,
    num_sequences: int,
    num_tokens: int,
) -> dict[str, Any]:
    """Describe the training inputs (paths, SHA-256 digests, sizes) recorded in the model and its manifest."""
    return {
        "dataset_release": dataset_release or "unknown",
        "policy_version": policy_version or "unknown",
        "run_id": run_id or "unnamed_run",
        "input_file": input_path.as_posix(),
        "input_sha256": sha256_file(input_path),
        "ids_file": ids_path.as_posix(),
        "ids_sha256": sha256_file(ids_path),
        "selected_sequences": num_sequences,
        "selected_tokens": num_tokens,
    }


def _environment_info() -> dict[str, Any]:
    """Python, platform and key package versions of the training environment."""
    return {
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
    }


def build_sidecar_manifest(
    *,
    run_id: str,
    mode: str,
    model: CRFModel,
    preset: str,
    feature_config: FeatureConfig,
    hyperparameters: Mapping[str, Any],
    training_diagnostics: Mapping[str, Any],
    memory_diagnostics: Mapping[str, Any],
    provenance: Mapping[str, Any],
    training_command: Sequence[str],
    duration: float,
    output_path: Path,
    artifact_hash: str,
) -> dict[str, Any]:
    """Build the comprehensive sidecar manifest that is written next to a saved CRF artifact."""
    return {
        "schema_version": 1,
        "run_id": run_id,
        "label_mode": mode,
        "model_name": model.model_name,
        "vocabulary": model.vocabulary,
        "feature_preset": preset,
        "feature_preset_version": feature_config.version,
        "feature_config": feature_config.to_dict(),
        "hyperparameters": hyperparameters,
        "training_diagnostics": training_diagnostics,
        "memory_diagnostics": memory_diagnostics,
        "provenance": provenance,
        "training_command": training_command,
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
        "environment": _environment_info(),
        "code_fingerprint": compute_code_fingerprint(),
        "fingerprinted_files": get_fingerprinted_files(),
        "reproducibility_warning": git_status_porcelain(),
        "created_at": datetime.now(UTC).isoformat(),
    }
