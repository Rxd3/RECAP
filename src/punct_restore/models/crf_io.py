"""CRF artifact persistence: write, read and validate the joblib bundle of a fitted CRFModel."""

from __future__ import annotations

import copy
import platform
import sys
from pathlib import Path
from typing import Any

import joblib
import sklearn_crfsuite

from punct_restore.features.crf_features import FeatureConfig
from punct_restore.models.crf_modes import LABEL_MODES

MODEL_SCHEMA_VERSION = 1


def _build_bundle(model: Any) -> dict[str, Any]:
    """The artifact dict of a fitted CRFModel: metadata, hyperparameters, features and the backend."""
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
        "training_metadata": copy.deepcopy(model._training_metadata),
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "sklearn_crfsuite": getattr(sklearn_crfsuite, "__version__", "unknown"),
            "joblib": getattr(joblib, "__version__", "unknown"),
        },
        "backend_state": model._backend,
    }


def write_model(model: Any, path: str | Path) -> None:
    """Save a fitted CRFModel and its complete compatibility metadata to ``path``; RuntimeError if unfitted."""
    if not model._is_fitted or model._backend is None:
        raise RuntimeError("Cannot save an unfitted CRFModel.")

    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = _build_bundle(model)

    joblib.dump(bundle, save_path)


def read_bundle(path: str | Path) -> dict[str, Any]:
    """Load the artifact at ``path`` and check root type, schema, family, label mode and backend presence.

    Raises:
        FileNotFoundError: If the artifact path does not exist.
        ValueError: If the artifact is unreadable or its schema, family or metadata is incompatible.
    """
    load_path = Path(path)
    if not load_path.is_file():
        raise FileNotFoundError(f"Model artifact not found at {load_path}")

    try:
        bundle = joblib.load(load_path)  # joblib is pickle: artifacts are this project's own saved models only
    except Exception as e:
        raise ValueError(f"Corrupt or unreadable model artifact at {load_path}: {e}") from e

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


def restore_model(cls: Any, bundle: dict[str, Any]) -> Any:
    """Rebuild a fitted ``cls`` (CRFModel) from a bundle checked by ``read_bundle``; verify its vocabulary."""
    label_mode = bundle.get("label_mode")
    hp = bundle.get("hyperparameters", {})
    feat_dict = bundle.get("feature_config", {})
    backend_state = bundle.get("backend_state")

    # Reconstruct model instance
    feature_config = FeatureConfig.from_dict(feat_dict)
    model = cls(
        label_mode=label_mode,
        algorithm=hp.get("algorithm", "lbfgs"),
        c1=hp.get("c1", 0.1),
        c2=hp.get("c2", 0.1),
        max_iterations=hp.get("max_iterations", 100),
        all_possible_transitions=hp.get("all_possible_transitions", True),
        feature_config=feature_config,
        model_name=bundle.get("model_name"),
    )

    model._backend = backend_state
    model._is_fitted = True
    model._training_metadata = bundle.get("training_metadata", {})

    # Verify restored vocabulary
    restored_vocab = bundle.get("vocabulary")
    if restored_vocab != model.vocabulary:
        raise ValueError(
            f"Artifact vocabulary {restored_vocab} does not match expected {model.vocabulary}"
        )

    return model
