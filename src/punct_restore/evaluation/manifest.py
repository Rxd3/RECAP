"""Prediction-manifest check, owned by Member 5 (MODEL_CONTRACT.md "sidecar manifest").

Field names follow the HMM's predictions.manifest.json (Member 2), which other models should
mirror: model_name, label_mode, ordered_labels, predictions_sha256 and input.sha256. The check
proves the predictions file is the one the model wrote, for the gold file being scored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .inputs import Bundle, EvaluationInputError, Vocab

REQUIRED = ("model_name", "label_mode", "ordered_labels", "predictions_sha256", "input")


def _load(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise EvaluationInputError(f"{path.name}: not a JSON manifest") from exc
    if not isinstance(manifest, dict):
        raise EvaluationInputError(f"{path.name}: manifest must be a JSON object")
    missing = [field for field in REQUIRED if field not in manifest]
    if missing:
        raise EvaluationInputError(f"{path.name}: missing field(s) {', '.join(missing)}")
    if not isinstance(manifest["input"], dict) or "sha256" not in manifest["input"]:
        raise EvaluationInputError(f"{path.name}: input must be an object with sha256")
    return manifest


def check_manifest(manifest_path: str | Path, bundle: Bundle, vocab: Vocab) -> None:
    """Raise EvaluationInputError unless the manifest matches the scored predictions and gold."""
    path = Path(manifest_path)
    manifest = _load(path)
    expected = {
        "model_name": bundle.model_name,
        "label_mode": bundle.label_mode,
        "ordered_labels": list(vocab.labels(bundle.label_mode)),
        "predictions_sha256": bundle.inputs["predictions"]["sha256"],
    }
    for field, value in expected.items():
        if manifest[field] != value:
            raise EvaluationInputError(
                f"{path.name}: {field} is {manifest[field]!r}, expected {value!r}")
    if manifest["input"]["sha256"] != bundle.inputs["gold"]["sha256"]:
        raise EvaluationInputError(
            f"{path.name}: predictions were made from a different input file than the gold file")
