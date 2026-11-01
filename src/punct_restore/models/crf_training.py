"""CRF training support that is not CLI-specific: code fingerprint, provenance and sidecar-manifest building.

Used by punct_restore.cli.train_crf, which keeps the command-line handling (arguments, console output, exit codes).
The frozen-ID record loading is in punct_restore.models.crf_training_data.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from punct_restore.features.crf_features import FeatureConfig
from punct_restore.hashing import sha256_file
from punct_restore.models.crf import CRFModel
from punct_restore.paths import REPO_ROOT as ROOT


class TrainingFailed(Exception):
    """A training step failed for a reason the user can fix; the message is the one line for the user (`error: ...`).

    Not an OSError or ValueError, so the `except (OSError, ValueError)` clauses of the steps never swallow it.
    """


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


def process_memory() -> tuple[dict[str, float] | None, str | None]:
    """(figures, None) when the OS reports the process memory; (None, why) when it does not."""
    stats = get_process_memory_stats()
    if stats:
        return stats, None
    if sys.platform != "win32":
        return None, f"not available on {sys.platform}"
    return None, "the Windows memory API gave no figures"


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

    digest = hashlib.sha256()
    for rel_posix in file_list:
        content = (repo_root / rel_posix).read_bytes()
        digest.update(rel_posix.encode("utf-8") + b"\0" + content)

    return digest.hexdigest()


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


def collect_training_diagnostics(model: CRFModel) -> dict[str, Any]:
    """Collect optimizer convergence diagnostics (iteration count, termination status) from a fitted model."""
    training_diagnostics: dict[str, Any] = {}
    log = model.training_log
    if hasattr(log, "iterations"):
        training_diagnostics["num_iterations"] = len(log.iterations)
    if hasattr(log, "lines") and log.lines:
        for line in reversed(log.lines):
            text = line.strip()
            if "terminated" in text.lower() or "converged" in text.lower():
                training_diagnostics["termination_status"] = text
                break
    return training_diagnostics


@dataclass(frozen=True)
class TrainingSource:
    """What one training run read: the frozen record file and ID list, their sizes and the release labels."""

    input_path: Path
    ids_path: Path
    num_sequences: int
    num_tokens: int
    dataset_release: str | None = None
    policy_version: str | None = None
    run_id: str | None = None


def build_training_provenance(source: TrainingSource) -> dict[str, Any]:
    """Describe the training inputs (paths, SHA-256 digests, sizes) recorded in the model and its manifest."""
    return {
        "dataset_release": source.dataset_release or "unknown",
        "policy_version": source.policy_version or "unknown",
        "run_id": source.run_id or "unnamed_run",
        "input_file": source.input_path.as_posix(),
        "input_sha256": sha256_file(source.input_path),
        "ids_file": source.ids_path.as_posix(),
        "ids_sha256": sha256_file(source.ids_path),
        "selected_sequences": source.num_sequences,
        "selected_tokens": source.num_tokens,
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


@dataclass(frozen=True)
class TrainingRun:
    """Everything the sidecar manifest of a trained and saved CRF model records."""

    run_id: str
    mode: str
    model: CRFModel
    preset: str
    feature_config: FeatureConfig
    hyperparameters: Mapping[str, Any]
    training_diagnostics: Mapping[str, Any]
    memory_diagnostics: Mapping[str, Any] | None
    provenance: Mapping[str, Any]
    training_command: Sequence[str]
    duration: float
    output_path: Path
    artifact_hash: str
    memory_unavailable: str | None = None  # why memory_diagnostics is None


def build_sidecar_manifest(run: TrainingRun) -> dict[str, Any]:
    """Build the comprehensive sidecar manifest that is written next to a saved CRF artifact.

    Without memory figures (`memory_diagnostics` None) it also holds `memory_diagnostics_unavailable`, the reason.
    """
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run.run_id,
        "label_mode": run.mode,
        "model_name": run.model.model_name,
        "vocabulary": run.model.vocabulary,
        "feature_preset": run.preset,
        "feature_preset_version": run.feature_config.version,
        "feature_config": run.feature_config.to_dict(),
        "hyperparameters": run.hyperparameters,
        "training_diagnostics": run.training_diagnostics,
        "memory_diagnostics": run.memory_diagnostics,
        "provenance": run.provenance,
        "training_command": run.training_command,
        "training_duration_seconds": round(run.duration, 3),
        "artifact": {
            "path": run.output_path.as_posix(),
            "sha256": run.artifact_hash,
            "size_bytes": run.output_path.stat().st_size,
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
    if run.memory_diagnostics is None:
        manifest["memory_diagnostics_unavailable"] = run.memory_unavailable or "unknown"
    return manifest
