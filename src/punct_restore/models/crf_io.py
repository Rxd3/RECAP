"""CRF artifact persistence: write, read and validate the joblib bundle of a fitted CRFModel."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import sklearn_crfsuite

from punct_restore.models.crf_modes import LABEL_MODES
from punct_restore.prediction_io import require_file

if TYPE_CHECKING:  # crf.py imports this module at run time
    from punct_restore.models.crf import CRFModel

MODEL_SCHEMA_VERSION = 1


def _build_bundle(model: CRFModel) -> dict[str, Any]:
    """The artifact dict of a fitted CRFModel: metadata, hyperparameters, features and the backend."""
    state = model._to_bundle()  # RuntimeError if unfitted
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_family": "crf",
        "model_name": model.model_name,
        "label_mode": model.label_mode,
        "vocabulary": list(model.vocabulary),
        "hyperparameters": {
            "algorithm": model.algorithm,
            "c1": model.c1,
            "c2": model.c2,
            "max_iterations": model.max_iterations,
            "all_possible_transitions": model.all_possible_transitions,
        },
        "feature_config": model.feature_config.to_dict(),
        "training_metadata": state["training_metadata"],
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "sklearn_crfsuite": getattr(sklearn_crfsuite, "__version__", "unknown"),
            "joblib": getattr(joblib, "__version__", "unknown"),
        },
        "backend_state": state["backend_state"],
    }


def write_model(model: CRFModel, path: str | Path) -> None:
    """Save a fitted CRFModel and its complete compatibility metadata to ``path``; RuntimeError if unfitted."""
    bundle = _build_bundle(model)  # before any folder is created
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, save_path)


def load_artifact(path: str | Path, what: str) -> Any:
    """The object the joblib artifact file at ``path`` holds; the caller checks its type. ``what`` names the file in
    a not-found message.

    Raises:
        FileNotFoundError: ``<name>: <what> not found``.
        IsADirectoryError: ``<name>: is a folder, not a file``.
        ValueError: ``<name>: not a readable CRF artifact`` (the unpickling text is not shown).
    """
    load_path = Path(path)
    require_file(load_path, what)
    try:  # joblib is pickle: artifacts are this project's own saved models, or a local file the user chose to inspect
        return joblib.load(load_path)
    except Exception as e:  # broad on purpose: unpickling a corrupt or foreign file can raise any exception type
        raise ValueError(f"{load_path.name}: not a readable CRF artifact") from e


def read_bundle(path: str | Path) -> dict[str, Any]:
    """Load the artifact at ``path`` and check root type, schema, family, label mode and backend presence.

    Raises:
        FileNotFoundError: If the artifact path does not exist.
        IsADirectoryError: If the artifact path is a folder.
        ValueError: If the artifact is unreadable or its schema, family or metadata is incompatible.
    """
    bundle = load_artifact(path, "model artifact")

    if not isinstance(bundle, dict):
        raise ValueError(f"Corrupt artifact: expected dict root, got {type(bundle).__name__}")

    schema_ver = bundle.get("schema_version")
    if schema_ver != MODEL_SCHEMA_VERSION:
        raise ValueError(f"Unsupported model schema_version: {schema_ver!r}")

    family = bundle.get("model_family")
    if family != "crf":
        raise ValueError(f"Incompatible model family: {family!r}, expected 'crf'")

    label_mode = bundle.get("label_mode")
    if label_mode not in LABEL_MODES:
        raise ValueError(f"Invalid label_mode in artifact: {label_mode!r}")

    if bundle.get("backend_state") is None:
        raise ValueError("Corrupt artifact: missing backend_state.")

    return bundle


def restore_model(cls: type[CRFModel], bundle: dict[str, Any]) -> CRFModel:
    """Rebuild a fitted ``cls`` (CRFModel) from a bundle checked by ``read_bundle``; verify its vocabulary."""
    model = cls._from_bundle(bundle)

    # Verify restored vocabulary
    restored_vocab = bundle.get("vocabulary")
    if restored_vocab != model.vocabulary:
        raise ValueError(
            f"Artifact vocabulary {restored_vocab} does not match expected {model.vocabulary}"
        )

    return model
