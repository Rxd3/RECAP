"""Checking a training command's inputs against the data release manifest (split out of `hmm_workflow`)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from punct_restore.data.preprocessing import parse_manifest_line
from punct_restore.file_io import read_object, require_object
from punct_restore.hashing import sha256_file


def checked_release_file(
    path: Path, root: Path, entries: dict[str, Any], *, expected: str | None = None
) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"{path.name}: file must be inside its declared release directory"
        ) from exc
    if relative not in entries or (expected is not None and relative != expected):
        raise ValueError(f"{relative}: not the expected release file {expected or ''}")
    entry = entries[relative]
    if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
        raise ValueError(f"{relative}: release checksum mismatch")
    return relative


def _check_split_manifest(split_manifest: Path, manifest: dict[str, Any]) -> None:
    """Authenticate the supplied split checksum index against the release manifest.

    Raises:
        ValueError: One line that starts with the file name: not valid UTF-8, or a line that is not
            ``<sha256>  <path>``; or the index does not match the release.
    """
    try:
        lines = split_manifest.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{split_manifest.name}: not valid UTF-8") from exc
    expected_sums = {name: entry["sha256"] for name, entry in manifest["split_files"].items()}
    actual_sums = {}
    for number, line in enumerate(lines, 1):
        digest, name = parse_manifest_line(line, number, split_manifest.name)
        if name in actual_sums:
            raise ValueError(f"{split_manifest.name}: duplicate path in split checksum manifest")
        actual_sums[name] = digest
    if actual_sums != expected_sums:
        raise ValueError(f"{split_manifest.name}: split checksum manifest does not match release")


def _check_prediction_files(
    manifest: dict[str, Any],
    processed: Path,
    splits: Path,
    mode: str,
    prediction: Path,
    prediction_ids: Path | None,
) -> None:
    """Check the prediction input (and its ID selection) against the release."""
    relative = checked_release_file(prediction, processed, manifest["files"])
    if not relative.startswith(f"mode_{mode}/"):
        raise ValueError("prediction input mode does not match model")
    if prediction_ids is not None:
        name = Path(relative).stem
        checked_release_file(
            prediction_ids,
            splits,
            manifest["split_files"],
            expected=f"primary_eval/{name}_mode_{mode}.ids",
        )


@dataclass(frozen=True)
class ReleaseInputs:
    """The files and identities a training command was given, to be checked against the data release."""

    train: Path
    train_ids: Path
    split_manifest: Path
    label_config: Path
    mode: str
    release_id: str
    policy_version: str
    prediction: Path | None = None
    prediction_ids: Path | None = None


# What the HMM commands read from a release manifest; `{}` is a table whose entries `read_release_manifest` checks.
_RELEASE_MANIFEST = {
    "release_id": str,
    "policy_version": str,
    "label_sets_sha256": str,
    "files": {},
    "split_files": {},
    "source": {"revision": str},
}
_FILE_ENTRY = {"sha256": str, "bytes": int}


def read_release_manifest(path: Path) -> dict[str, Any]:
    """A release manifest with the fields the HMM commands read; a wrong shape is a ValueError naming the file.

    The message is `<file name>: missing <key>` or `<file name>: <key> is not <kind>` (`file_io.require_object`);
    an entry of `files` or `split_files` is placed by its name (`files.mode_A/train.jsonl.bytes`).
    """
    manifest = read_object(path, _RELEASE_MANIFEST)
    for table in ("files", "split_files"):
        require_object(path.name, manifest[table], dict.fromkeys(manifest[table], _FILE_ENTRY), table)
    return manifest


def verify_release_inputs(manifest_path: Path, inputs: ReleaseInputs) -> dict[str, Any]:
    """Check the command's inputs against the release manifest; return the manifest."""
    manifest = read_release_manifest(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("release_id") != inputs.release_id
        or manifest.get("policy_version") != inputs.policy_version
    ):
        raise ValueError("release identity/schema/policy does not match command")
    if sha256_file(inputs.label_config) != manifest["label_sets_sha256"]:
        raise ValueError("label configuration does not match the data release")
    processed, splits, mode = manifest_path.parent, inputs.split_manifest.parent, inputs.mode
    checked_release_file(inputs.train, processed, manifest["files"], expected=f"mode_{mode}/train.jsonl")
    checked_release_file(
        inputs.train_ids, splits, manifest["split_files"], expected=f"default_train/mode_{mode}.ids"
    )
    _check_split_manifest(inputs.split_manifest, manifest)
    if inputs.prediction is not None:
        _check_prediction_files(manifest, processed, splits, mode, inputs.prediction, inputs.prediction_ids)
    return manifest


def verify_training_release(
    manifest_path: Path,
    *,
    train: Path,
    train_ids: Path,
    split_manifest: Path,
    label_config: Path,
    mode: str,
    release_id: str,
    policy_version: str,
    prediction: Path | None = None,
    prediction_ids: Path | None = None,
) -> dict[str, Any]:
    """Main's call signature of `verify_release_inputs` (old import path and signature kept, restructure spec §2)."""
    inputs = ReleaseInputs(
        train=train,
        train_ids=train_ids,
        split_manifest=split_manifest,
        label_config=label_config,
        mode=mode,
        release_id=release_id,
        policy_version=policy_version,
        prediction=prediction,
        prediction_ids=prediction_ids,
    )
    return verify_release_inputs(manifest_path, inputs)
