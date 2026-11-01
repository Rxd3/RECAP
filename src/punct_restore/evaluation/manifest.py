"""Prediction-manifest check, owned by Member 5 (MODEL_CONTRACT.md "sidecar manifest").

Field names follow the HMM's predictions.manifest.json (Member 2), which other models should
mirror: model_name, label_mode, ordered_labels, predictions_sha256 and input.sha256. The HMM
handoff bundle nests the input identity as inputs.input.sha256; either place is accepted, and
both must agree when both are present. The check proves the predictions file is the one the
model wrote, for the gold file being scored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .inputs import Bundle, EvaluationInputError, Vocab

REQUIRED = ("model_name", "label_mode", "ordered_labels", "predictions_sha256")


def _input_candidates(manifest: dict[str, Any]) -> list[Any]:
    """The input identity at top level and/or under inputs.input (HMM handoff bundle shape)."""
    nested = manifest.get("inputs")
    found = [manifest["input"]] if "input" in manifest else []
    if isinstance(nested, dict) and "input" in nested:
        found.append(nested["input"])
    return found


def _load(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """The manifest and its checked input identity."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:  # RecursionError: nested too deeply
        raise EvaluationInputError(f"{path.name}: not a JSON manifest") from exc
    if not isinstance(manifest, dict):
        raise EvaluationInputError(f"{path.name}: manifest must be a JSON object")
    candidates = _input_candidates(manifest)
    missing = [field for field in REQUIRED if field not in manifest] + ([] if candidates else ["input"])
    if missing:
        raise EvaluationInputError(f"{path.name}: missing field(s) {', '.join(missing)}")
    if any(not isinstance(value, dict) or not isinstance(value.get("sha256"), str)
           for value in candidates):
        raise EvaluationInputError(f"{path.name}: input must be an object with sha256")
    if candidates[0]["sha256"] != candidates[-1]["sha256"]:
        raise EvaluationInputError(f"{path.name}: input and inputs.input name different files")
    return manifest, candidates[0]


def check_manifest(manifest_path: str | Path, bundle: Bundle, vocab: Vocab) -> None:
    """Raise EvaluationInputError unless the manifest matches the scored predictions and gold."""
    path = Path(manifest_path)
    manifest, input_identity = _load(path)
    expected = {
        "model_name": bundle.model_name,
        "label_mode": bundle.label_mode,
        "ordered_labels": list(vocab.labels(bundle.label_mode)),
        "predictions_sha256": bundle.identity("predictions")["sha256"],
    }
    for field, value in expected.items():
        if manifest[field] != value:
            raise EvaluationInputError(
                f"{path.name}: {field} is {manifest[field]!r}, expected {value!r}")
    if input_identity["sha256"] != bundle.identity("gold")["sha256"]:
        raise EvaluationInputError(
            f"{path.name}: predictions were made from a different input file than the gold file")
