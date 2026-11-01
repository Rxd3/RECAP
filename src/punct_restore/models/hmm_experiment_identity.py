"""Which code a recorded HMM experiment was run with, and which shared evaluator checkout it was scored by."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from punct_restore.hashing import sha256_file
from punct_restore.models.hmm_workflow import code_identity
from punct_restore.paths import REPO_ROOT as ROOT

# Every punct_restore file that is loaded when the four HMM commands are imported, package `__init__.py` files
# included (Python runs those first, and they import more modules); this is checked against a fresh interpreter. The
# order: the HMM modules, then every other file that gets loaded, sorted, then each command's module and the stub that
# still runs it. This module is listed too: a change to the code that computes the record must change the record.
SOURCE_FILES = (
    "src/punct_restore/models/hmm.py",
    "src/punct_restore/models/hmm_distributions.py",
    "src/punct_restore/models/hmm_validation.py",
    "src/punct_restore/models/hmm_records.py",
    "src/punct_restore/models/hmm_workflow.py",
    "src/punct_restore/models/hmm_release.py",
    "src/punct_restore/models/hmm_experiment_protocol.py",
    "src/punct_restore/models/hmm_experiment_commands.py",
    "src/punct_restore/models/hmm_experiment_runtime.py",
    "src/punct_restore/models/hmm_experiment_audit.py",
    "src/punct_restore/models/hmm_experiment_identity.py",
    "src/punct_restore/models/hmm_experiment_tune.py",
    "src/punct_restore/models/hmm_experiment_frozen.py",
    "src/punct_restore/models/hmm_experiment_finalize.py",
    "src/punct_restore/models/hmm_experiment_verify.py",
    "src/punct_restore/__init__.py",
    "src/punct_restore/cli/__init__.py",
    "src/punct_restore/data/__init__.py",
    "src/punct_restore/data/extraction.py",
    "src/punct_restore/data/labeling.py",
    "src/punct_restore/data/preprocessing.py",
    "src/punct_restore/data/release.py",
    "src/punct_restore/data/release_checks.py",
    "src/punct_restore/data/release_constants.py",
    "src/punct_restore/data/release_manifest.py",
    "src/punct_restore/data/release_notes.py",
    "src/punct_restore/data/release_outputs.py",
    "src/punct_restore/data/release_stats.py",
    "src/punct_restore/data/splits.py",
    "src/punct_restore/data/validation.py",
    "src/punct_restore/evaluation/__init__.py",
    "src/punct_restore/evaluation/baseline.py",
    "src/punct_restore/evaluation/comparison.py",
    "src/punct_restore/evaluation/error_analysis.py",
    "src/punct_restore/evaluation/inputs.py",
    "src/punct_restore/evaluation/manifest.py",
    "src/punct_restore/evaluation/metrics.py",
    "src/punct_restore/evaluation/report.py",
    "src/punct_restore/evaluation/uncertainty.py",
    "src/punct_restore/evaluation/views.py",
    "src/punct_restore/file_io.py",
    "src/punct_restore/hashing.py",
    "src/punct_restore/models/__init__.py",
    "src/punct_restore/paths.py",
    "src/punct_restore/prediction_io.py",
    "src/punct_restore/progress.py",
    "src/punct_restore/cli/train_hmm.py",
    "scripts/train_hmm.py",
    "src/punct_restore/cli/predict_hmm.py",
    "scripts/predict_hmm.py",
    "src/punct_restore/cli/run_hmm_experiments.py",
    "scripts/run_hmm_experiments.py",
    "src/punct_restore/cli/verify_hmm_experiments.py",
    "scripts/verify_hmm_experiments.py",
)


def implementation() -> dict[str, str]:
    """The SHA-256 of every file of the HMM study's code, keyed by its repository path."""
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def check_evaluator(root: Path, expected_commit: str) -> dict[str, Any]:
    """Require a clean evaluator checkout at the pinned commit; return its commit and the hashes of its files."""
    identity = code_identity(root)
    if identity != {"commit": expected_commit, "dirty": False}:
        raise ValueError("evaluator checkout must be clean at the pinned Member 5 commit")
    names = sorted(
        p.relative_to(root).as_posix() for p in (root / "src/punct_restore/evaluation").glob("*.py")
    )
    names += ["scripts/evaluate_models.py", "configs/label_sets.yaml"]
    return {"commit": expected_commit, "files": {name: sha256_file(root / name) for name in names}}
