"""Protocol rules of the HMM experiment: config validation, release authentication and winner selection."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from punct_restore.file_io import require_object
from punct_restore.hashing import sha256_file
from punct_restore.models.hmm_workflow import checked_release_file, file_identity, read_release_manifest
from punct_restore.paths import REPO_ROOT as ROOT

# isort: split
# old import path kept: the one JSON reader is in file_io
from punct_restore.file_io import read_json as read_json

# What an experiment config must hold for the steps after validate_config to index it, whatever the file says:
# every key that any step reads, with the kind of value it needs (see file_io.require_object).
_CONFIG_SCHEMA = {
    "schema_version": object,
    "experiment_id": str,
    "dataset_release_id": str,
    "policy_version": str,
    "processed_dir": str,
    "splits_dir": str,
    "label_config": str,
    "modes": list[str],
    "search": {
        "smoothing": list[float],
        "min_token_count": list[float],
        "selection_split": object,
        "selection_blocks": {},
        "metric": object,
        "score_precision": object,
        "tie_break": list[str],
        "retrain_on_validation": object,
    },
    "evaluation": {"repository_commit": str, "test_sets": list[str], "zero_division": object},
    "runtime": {"comparison_ids": str, "warmup_sequences": int, "repetitions": int, "model_threads": object},
}


def now() -> str:
    """The current UTC time as an ISO string."""
    return datetime.now(UTC).isoformat()


def _check_modes_and_search(config: dict[str, Any]) -> None:
    """The protocol rules for the modes and the selection settings (the shape of the config is already checked)."""
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
        if search["selection_blocks"].get(mode) != expected:
            raise ValueError(f"mode {mode}: wrong primary selection block")


def _check_search_values(values: list[Any], kind: type) -> None:
    """Require a search grid to be nonempty, unique, positive and finite (cutoffs: whole numbers)."""
    if not values or len(set(values)) != len(values):
        raise ValueError("search values must be nonempty and unique")
    for value in values:
        if (isinstance(value, float) and not math.isfinite(value)) or value <= 0 or (
            kind is int and type(value) is not int
        ):
            raise ValueError("smoothing/cutoffs must be positive; cutoffs must be integers")


def validate_config(config: Any, source: str = "config") -> dict[str, Any]:
    """Reject any experiment config that is not the frozen validation-only protocol.

    `source` names the file the config came from. The shape comes first: a config that is not an object, or lacks
    a key a later step indexes, or holds a value of another kind (a mode that is no text, a search value that is
    no number) is a ValueError of one line, `<source>: invalid experiment config: <what is wrong>`, that names
    the key. Only then are the protocol rules checked.
    """
    require_object(f"{source}: invalid experiment config", config, _CONFIG_SCHEMA)
    _check_modes_and_search(config)
    search = config["search"]
    _check_search_values(search["smoothing"], float)
    _check_search_values(search["min_token_count"], int)
    runtime = config["runtime"]
    if runtime["repetitions"] < 2 or runtime["warmup_sequences"] < 1:
        raise ValueError("runtime measurement requires warmup and repetitions")
    if runtime["model_threads"] != 1 or config["evaluation"]["zero_division"] != 0:
        raise ValueError("this implementation uses one model thread and zero_division=0")
    return config


def authenticate(config: dict[str, Any], phase: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check the release manifest, split lists and masks the given phase may read.

    A release manifest of the wrong shape is a ValueError that starts with its file name.
    """
    processed = ROOT / config["processed_dir"]
    splits = ROOT / config["splits_dir"]
    manifest_path = processed / "release_manifest.json"
    manifest = read_release_manifest(manifest_path)
    if (
        manifest["release_id"] != config["dataset_release_id"]
        or manifest["policy_version"] != config["policy_version"]
    ):
        raise ValueError("experiment and release versions differ")
    if sha256_file(ROOT / config["label_config"]) != manifest["label_sets_sha256"]:
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
