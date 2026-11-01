"""Protocol rules of the HMM experiment: config validation, the shape check of its files, release authentication and
winner selection."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from punct_restore.hashing import sha256_file
from punct_restore.models.hmm_workflow import checked_release_file, file_identity, read_release_manifest
from punct_restore.paths import REPO_ROOT as ROOT

# isort: split
# old import path kept: the one JSON reader is in file_io
from punct_restore.file_io import read_json as read_json

# What an experiment config must hold for the steps after validate_config to index it without a KeyError.
_CONFIG_SCHEMA = {
    "experiment_id": str,
    "dataset_release_id": str,
    "policy_version": str,
    "processed_dir": str,
    "splits_dir": str,
    "label_config": str,
    "modes": list,
    "evaluation": {"repository_commit": str},
}
_KIND_NAMES = {str: "nonempty text", list: "a JSON list", float: "a number"}


def now() -> str:
    """The current UTC time as an ISO string."""
    return datetime.now(UTC).isoformat()


def _has_kind(value: Any, kind: type) -> bool:
    """Whether a JSON value is of the kind a schema asks for (`object`: anything; `float`: a number but not a bool)."""
    if kind is str:
        return isinstance(value, str) and bool(value.strip())
    if kind is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return kind is object or isinstance(value, kind)


def require_object(name: str, value: Any, schema: Mapping[str, Any], at: str = "") -> dict[str, Any]:
    """Require `value`, the content of the file `name`, to be a JSON object that holds every key of `schema`.

    A schema value says what its key must hold: `object` anything (the key only has to be there), `str` nonempty
    text, `float` a number, `list` a JSON list, or a schema of its own for a nested object (`{}`: any object).
    Other keys are not looked at. `at` is where `value` sits inside the file (a dotted path), for a nested call.

    Raises:
        ValueError: One line, `<name>: <what is wrong>`, that names the key by its dotted path.
    """
    if not isinstance(value, dict):
        raise ValueError(f"{name}: {at} is not a JSON object" if at else f"{name}: not a JSON object")
    for key, want in schema.items():
        place = f"{at}.{key}" if at else key
        if key not in value:
            raise ValueError(f"{name}: missing {place}")
        if isinstance(want, Mapping):
            require_object(name, value[key], want, place)
        elif not _has_kind(value[key], want):
            raise ValueError(f"{name}: {place} is not {_KIND_NAMES[want]}")
    return value


def read_object(path: Path, schema: Mapping[str, Any]) -> dict[str, Any]:
    """The JSON object a study file holds, checked against `schema` (see `read_json` and `require_object`).

    Every message starts with the file name only.
    """
    return require_object(path.name, read_json(path), schema)


def validate_config(config: dict[str, Any], source: str = "config") -> dict[str, Any]:
    """Reject any experiment config that is not the frozen validation-only protocol.

    `source` names the file the config came from. A text value the later steps index that is missing, empty or of
    another type is a ValueError that starts `<source>: invalid experiment config:` and names the key.
    """
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
    require_object(f"{source}: invalid experiment config", config, _CONFIG_SCHEMA)
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
