"""CRF artifact persistence: write, read and validate the joblib bundle of a fitted CRFModel."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import sklearn_crfsuite

from punct_restore.file_io import require_file, require_object
from punct_restore.models.crf_modes import LABEL_MODES

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


# What a saved bundle holds besides the checked schema_version, model_family and label_mode (see
# `check_bundle_fields`). The flags are `object` here and checked as true or false there (file_io has no such kind).
_FLAGS = ("use_affixes", "use_shape", "use_internal_punct", "use_token_ngrams", "use_question_cues",
          "use_clause_cues", "use_casing_cues", "include_utterance_boundaries", "use_positional_buckets")
# the team-extension switches: an artifact saved before they existed lacks them
_EXTENSION_FLAGS = ("use_filler_cues", "use_question_lookback", "use_word_clusters")
_FIELDS: dict[str, Any] = {
    "model_name": str,
    "vocabulary": list[str],
    "hyperparameters": {"algorithm": str, "c1": float, "c2": float, "max_iterations": int,
                        "all_possible_transitions": object},
    "feature_config": {"preset": str, "window_size": int, "version": str, **dict.fromkeys(_FLAGS, object)},
    "training_metadata": {},
    "environment": {},
}


def _require_flags(prefix: str, section: dict[str, Any], place: str, keys: tuple[str, ...]) -> None:
    """Raise ValueError for the first of ``keys`` that ``section`` holds and that is not true or false."""
    for key in keys:
        if key in section and not isinstance(section[key], bool):
            raise ValueError(f"{prefix}: {place}.{key} is not true or false")


def check_bundle_fields(name: str, bundle: dict[str, Any]) -> None:
    """Check that every field of a CRF artifact bundle that the loader or ``inspect`` reads is there and of its kind.

    ``name`` is the file name. The team-extension feature switches may be absent (older artifacts) but, if present,
    must be of their kind. A bundle's backend must be an ``sklearn_crfsuite.CRF``.

    Raises:
        ValueError: ``<name>: corrupt artifact: missing <field>`` or ``... <field> is not <kind>`` (a dotted path
            names a nested field or a list item).
    """
    prefix = f"{name}: corrupt artifact"
    if bundle.get("backend_state") is None:
        raise ValueError(f"{prefix}: missing backend_state")
    if not isinstance(bundle["backend_state"], sklearn_crfsuite.CRF):
        raise ValueError(f"{prefix}: backend_state is not a CRF model")
    require_object(prefix, bundle, _FIELDS)
    config = bundle["feature_config"]
    _require_flags(prefix, bundle["hyperparameters"], "hyperparameters", ("all_possible_transitions",))
    _require_flags(prefix, config, "feature_config", _FLAGS + _EXTENSION_FLAGS)
    if config.get("word_clusters") is not None:
        require_object(prefix, config, {"word_clusters": {}}, "feature_config")


def read_bundle(path: str | Path) -> dict[str, Any]:
    """Load the artifact at ``path`` and check root type, schema, family, label mode, backend and field kinds.

    Raises:
        FileNotFoundError: If the artifact path does not exist.
        IsADirectoryError: If the artifact path is a folder.
        ValueError: ``<name>: <reason>`` if the artifact is unreadable or its schema, family, metadata or a field
            is incompatible (see ``check_bundle_fields``).
    """
    bundle = load_artifact(path, "model artifact")
    name = Path(path).name

    if not isinstance(bundle, dict):
        raise ValueError(f"{name}: corrupt artifact: expected dict root, got {type(bundle).__name__}")

    schema_ver = bundle.get("schema_version")
    if schema_ver != MODEL_SCHEMA_VERSION:
        raise ValueError(f"{name}: unsupported model schema_version {schema_ver!r}")

    family = bundle.get("model_family")
    if family != "crf":
        raise ValueError(f"{name}: incompatible model family {family!r}, expected 'crf'")

    label_mode = bundle.get("label_mode")
    if label_mode not in LABEL_MODES:
        raise ValueError(f"{name}: invalid label_mode in artifact: {label_mode!r}")

    check_bundle_fields(name, bundle)
    return bundle


def restore_model(cls: type[CRFModel], bundle: dict[str, Any]) -> CRFModel:
    """Rebuild a fitted ``cls`` (CRFModel) from a bundle checked by ``read_bundle``; verify its vocabulary."""
    model = cls._from_bundle(bundle)

    # Verify restored vocabulary
    restored_vocab = bundle.get("vocabulary")
    if restored_vocab != model.vocabulary:
        raise ValueError(
            f"artifact vocabulary {restored_vocab} does not match expected {model.vocabulary}"
        )

    return model


def load_fitted(cls: type[CRFModel], path: str | Path) -> CRFModel:
    """A fitted ``cls`` from the artifact at ``path``; every ValueError starts with the artifact's file name."""
    bundle = read_bundle(path)  # names the file itself
    try:
        return restore_model(cls, bundle)
    except ValueError as exc:
        raise ValueError(f"{Path(path).name}: {exc}") from exc
