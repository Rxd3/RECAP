"""CRF inspection support that is not CLI-specific: the text shown for a saved model artifact and its manifest.

Used by punct_restore.cli.train_crf (`inspect`), which prints the lines and chooses the exit code. Nothing here prints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

from punct_restore.models.crf_io import load_artifact

RULE = "=" * 70
_SECTIONS = (
    ("hyperparameters", "Hyperparameters:"),
    ("feature_config", "Feature Configuration:"),
    ("training_metadata", "Training Metadata:"),
    ("environment", "Environment:"),
)


class Inspection(NamedTuple):
    """What to show: the artifact summary, notes about unreadable manifests (for stderr) and the manifest block."""

    summary: list[str]
    notes: list[str]
    manifest: list[str]


def _load_artifact(model_path: Path) -> dict[str, Any]:
    """The bundle dict of the artifact; FileNotFoundError if absent, IsADirectoryError if a folder, ValueError if
    unreadable or not a dict."""
    bundle = load_artifact(model_path, "artifact")
    if not isinstance(bundle, dict):
        raise ValueError(f"{model_path.name}: invalid artifact format: root is {type(bundle).__name__}")
    return bundle


def _summary_lines(bundle: dict[str, Any], file_name: str) -> list[str]:
    """The header, vocabulary and the hyperparameter, feature, training and environment sections."""
    lines = [
        RULE,
        f"RECAP CRF Model Artifact: {file_name}",
        RULE,
        f"Schema Version:     {bundle.get('schema_version')}",
        f"Model Name:         {bundle.get('model_name')}",
        f"Label Mode:         {bundle.get('label_mode')}",
        f"Vocabulary Size:    {len(bundle.get('vocabulary', []))} classes",
        f"Vocabulary:         {bundle.get('vocabulary')}",
    ]
    for key, title in _SECTIONS:
        lines += ["", title]
        lines += [f"  {k:24}: {v}" for k, v in bundle.get(key, {}).items()]
    return lines


def _read_companion_manifest(path: Path) -> dict[str, Any] | None:
    """The manifest object at ``path``, or None when the file is unreadable, not UTF-8 or not a JSON object."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError, RecursionError):  # unreadable, not UTF-8, not JSON or nested too deeply
        return None
    if not isinstance(data, dict) or not isinstance(data.get("provenance", {}), dict):
        return None
    return data


def _manifest_lines(m_data: dict[str, Any]) -> list[str]:
    """The block shown for a companion manifest."""
    provenance = m_data.get("provenance", {})
    return [
        "",
        "Companion Manifest:",
        f"  Run ID:                 {m_data.get('run_id')}",
        f"  Dataset Release:        {provenance.get('dataset_release')}",
        f"  Policy Version:         {provenance.get('policy_version')}",
        f"  Code Fingerprint:       {m_data.get('code_fingerprint')}",
        f"  Fingerprinted Files:    {len(m_data.get('fingerprinted_files', []))} files",
        f"  Training Duration:      {m_data.get('training_duration_seconds')}s",
        f"  Reproducibility Note:   {m_data.get('reproducibility_warning')}",
    ]


def _companion_manifest(model_path: Path) -> tuple[list[str], list[str]]:
    """(notes, manifest block) of the first readable manifest next to the artifact; notes name unreadable ones."""
    candidates = [
        model_path.with_name(f"{model_path.name}.manifest.json"),
        model_path.with_suffix(".joblib.manifest.json"),
        model_path.with_suffix(".manifest.json"),
    ]
    notes: list[str] = []
    for path in dict.fromkeys(candidates):  # for a .joblib artifact the first two are the same file
        if not path.is_file():
            continue
        data = _read_companion_manifest(path)
        if data is not None:
            return notes, _manifest_lines(data)
        notes.append(f"Note: companion manifest unreadable: {path.name}")
    return notes, []


def inspect_artifact(model_path: Path) -> Inspection:
    """Read a saved CRF artifact and its companion manifest; FileNotFoundError or ValueError if it cannot be read."""
    bundle = _load_artifact(model_path)
    notes, manifest = _companion_manifest(model_path)
    return Inspection(_summary_lines(bundle, model_path.name), notes, manifest)
