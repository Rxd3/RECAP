"""Protocol rules of the HMM experiment: config validation, release authentication and winner selection."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from punct_restore.models.hmm_workflow import checked_release_file, file_identity, sha256
from punct_restore.paths import REPO_ROOT as ROOT


def now() -> str:
    """The current UTC time as an ISO string."""
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> Any:
    """Parse a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Reject any experiment config that is not the frozen validation-only protocol."""
    if config["schema_version"] != 1 or not config["modes"]:
        raise ValueError("invalid experiment schema/modes")
    if len(set(config["modes"])) != len(config["modes"]) or set(config["modes"]) - set("ABC"):
        raise ValueError("modes must be unique members of A/B/C")
    search = config["search"]
    if search["selection_split"] != "validation" or search["retrain_on_validation"]:
        raise ValueError("settings must be selected on validation; training stays train-only")
    if search["metric"] != "macro_f1" or search["score_precision"] != 6:
        raise ValueError("selection must use the shared scorer's six-decimal primary macro F1")
    if search["tie_break"] != ["lower_min_token_count", "lower_smoothing"]:
        raise ValueError("unsupported tie-break policy")
    if config["evaluation"]["test_sets"] != ["test", "test_amb"]:
        raise ValueError("report test and test_amb separately")
    for mode in config["modes"]:
        expected = "joint/unmasked" if mode == "C" else "punctuation/all"
        if search["selection_blocks"][mode] != expected:
            raise ValueError(f"mode {mode}: wrong primary selection block")
    for values, kind in [(search["smoothing"], float), (search["min_token_count"], int)]:
        if not values or len(set(values)) != len(values):
            raise ValueError("search values must be nonempty and unique")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError("search parameters must be numeric")
            if not math.isfinite(value) or value <= 0 or (kind is int and type(value) is not int):
                raise ValueError("smoothing/cutoffs must be positive; cutoffs must be integers")
    runtime = config["runtime"]
    if runtime["repetitions"] < 2 or runtime["warmup_sequences"] < 1:
        raise ValueError("runtime measurement requires warmup and repetitions")
    if runtime["model_threads"] != 1 or config["evaluation"]["zero_division"] != 0:
        raise ValueError("this implementation uses one model thread and zero_division=0")
    return config


def authenticate(config: dict[str, Any], phase: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check the release manifest, split lists and masks the given phase may read."""
    processed = ROOT / config["processed_dir"]
    splits = ROOT / config["splits_dir"]
    manifest_path = processed / "release_manifest.json"
    manifest = read_json(manifest_path)
    if (
        manifest["release_id"] != config["dataset_release_id"]
        or manifest["policy_version"] != config["policy_version"]
    ):
        raise ValueError("experiment and release versions differ")
    if sha256(ROOT / config["label_config"]) != manifest["label_sets_sha256"]:
        raise ValueError("label configuration mismatch")
    names = ["train", "validation"] if phase == "tune" else ["validation", "test", "test_amb"]
    verified = {"release_manifest": file_identity(manifest_path)}
    for mode in config["modes"]:
        for name in names:
            relative = f"mode_{mode}/{name}.jsonl"
            checked_release_file(
                processed / relative, processed, manifest["files"], expected=relative
            )
            id_name = (
                f"default_train/mode_{mode}.ids"
                if name == "train"
                else f"primary_eval/{name}_mode_{mode}.ids"
            )
            checked_release_file(
                splits / id_name, splits, manifest["split_files"], expected=id_name
            )
            verified[relative] = manifest["files"][relative]["sha256"]
            verified[id_name] = manifest["split_files"][id_name]["sha256"]
            if mode == "C" and name != "train":
                relative = f"masks/case_joint/{name}.jsonl"
                checked_release_file(
                    processed / relative, processed, manifest["files"], expected=relative
                )
                verified[relative] = manifest["files"][relative]["sha256"]
    return manifest, verified


def choose(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """The winning candidate: best validation score, then lower cutoff, then lower smoothing."""
    if not candidates:
        raise ValueError("no completed validation candidates")
    if any(not math.isfinite(c["validation_primary"]) for c in candidates):
        raise ValueError("non-finite validation score")
    return min(
        candidates, key=lambda c: (-c["validation_primary"], c["min_token_count"], c["smoothing"])
    )
