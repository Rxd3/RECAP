"""CRF without transitions: a per-token multinomial logistic regression on the CRF's own features.

GapClassifier trains on EXACTLY the per-token feature dicts that CRFModel uses (same validation, same
extractor, same FeatureConfig) and predicts every token independently of its neighbours' labels. Its
role is a comparison against the ``improved`` CRF on identical features and data: the two models differ
in whether labels are scored jointly with label-to-label transitions and in their training (L2 logistic
regression, C = 1.0, 200 lbfgs iterations vs the CRF's c1 = c2 = 0.1, 100 iterations), so the gap between
them is the value of the transitions plus those training differences.

Persistence follows the CRF's bundle style: a joblib file holding a plain dict. joblib is pickle, so
load only artifacts you produced yourself.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from punct_restore.features.crf_features import FeatureConfig
from punct_restore.file_io import require_file
from punct_restore.models.crf import (
    SCORING_MODES,
    CRFModel,
    get_mode_vocabulary,
    require_scoring_mode,
)

SCHEMA_VERSION = 1
MODEL_FAMILY = "gap_classifier"
SOLVER = "lbfgs"  # deterministic, and handles the sparse multinomial problem
# Records validated, featurised and vectorised at a time. Holding every token's feature dict at once
# (plus DictVectorizer's own per-value lists) cost several GB on the full data; the sparse matrix is small.
CHUNK_RECORDS = 2000
DEFAULT_FEATURE_CONFIG = FeatureConfig.improved()  # frozen, so safe as a shared default argument
_BUNDLE_PARTS: dict[str, type] = {
    "model_name": str,
    "label_mode": str,
    "vocabulary": list,
    "feature_config": dict,
    "hyperparameters": dict,
    "training_summary": dict,
    "vectorizer": DictVectorizer,
    "classifier": LogisticRegression,
}
# Opt-in switches added to FeatureConfig after the first gap models were saved. An older bundle lacks
# them, which means "off" (the field default), so a missing key is tolerated for these only.
_OPTIONAL_CONFIG_KEYS = frozenset({"use_question_lookback", "use_word_clusters", "word_clusters"})


def _flat(sequences: Iterable[Iterable[dict[str, Any]]]) -> Iterator[dict[str, Any]]:
    """Stream the token feature dicts of all sequences, without a second full list."""
    return (token_features for sequence in sequences for token_features in sequence)


def _chunks(records: Any) -> Iterator[Any]:
    """Slices of at most CHUNK_RECORDS records (one empty slice for no records).

    Anything that is not a Sequence goes through whole, so the CRF validator rejects it with its own message.
    """
    if isinstance(records, Sequence):
        for start in range(0, max(len(records), 1), CHUNK_RECORDS):
            yield records[start:start + CHUNK_RECORDS]
    else:
        yield records


def _check_unique_sample_ids(records: Any) -> None:
    """Reject a sample_id repeated anywhere in the input; the CRF validator only sees one chunk at a time.

    Same message as the CRF's. Records the CRF would reject for other reasons are skipped here and left to it.
    """
    if not isinstance(records, Sequence):
        return
    seen: set[str] = set()
    for idx, record in enumerate(records):
        sample_id = record.get("sample_id") if isinstance(record, Mapping) else None
        if not isinstance(sample_id, str) or not sample_id.strip():
            continue
        if sample_id in seen:
            raise ValueError(f"Record at index {idx}: duplicate sample_id {sample_id!r} detected.")
        seen.add(sample_id)


def _check_hyperparameters(regularization: Any, max_iter: Any, model_name: Any) -> None:
    if (isinstance(regularization, bool) or not isinstance(regularization, (int, float))
            or not math.isfinite(regularization) or regularization <= 0):
        raise ValueError(f"C must be a positive finite number, got {regularization!r}")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter!r}")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string.")


def _config_is_sound(config: Mapping[str, Any]) -> bool:
    """True when config has no unknown key, lacks only optional keys, and every present key round-trips."""
    fields = FeatureConfig.__dataclass_fields__.keys()  # type: ignore[attr-defined]
    if config.keys() - fields or (fields - config.keys()) - _OPTIONAL_CONFIG_KEYS:
        return False
    try:
        rebuilt = FeatureConfig.from_dict(config).to_dict()
    except (TypeError, ValueError):  # e.g. the cluster switch on without a usable map
        return False
    return all(rebuilt[key] == value for key, value in config.items())


def _bundle_problem(bundle: Any) -> str | None:
    """Describe the first inconsistency in a loaded bundle, or None when it is sound."""
    if not isinstance(bundle, dict):
        return f"expected a dict, got {type(bundle).__name__}"
    if bundle.get("schema_version") != SCHEMA_VERSION:
        return f"unsupported schema_version {bundle.get('schema_version')!r}"
    if bundle.get("model_family") != MODEL_FAMILY:
        return f"model family {bundle.get('model_family')!r}, expected {MODEL_FAMILY!r}"
    for key, kind in _BUNDLE_PARTS.items():
        if not isinstance(bundle.get(key), kind):
            return f"{key!r} is missing or invalid"
    if bundle["label_mode"] not in SCORING_MODES:
        return f"unknown label mode {bundle['label_mode']!r}"
    if bundle["vocabulary"] != get_mode_vocabulary(bundle["label_mode"]):
        return "vocabulary does not match the label mode"
    if not _config_is_sound(bundle["feature_config"]):
        return "feature_config has unknown, missing or invalid keys"
    hyper = bundle["hyperparameters"]
    try:
        _check_hyperparameters(hyper.get("C"), hyper.get("max_iter"), bundle["model_name"])
    except ValueError as exc:
        return str(exc)
    classes = getattr(bundle["classifier"], "classes_", None)
    if classes is None or not hasattr(bundle["vectorizer"], "vocabulary_"):
        return "the vectorizer or classifier is not fitted"
    if not set(classes) <= set(bundle["vocabulary"]):
        return "classifier labels are outside the vocabulary"
    return None


class GapClassifier:
    """Per-token logistic regression over CRF feature dicts (no label transitions).

    Attributes:
        label_mode: 'A', 'B' or 'C'.
        vocabulary: Ordered label vocabulary of the mode (from get_mode_vocabulary).
        feature_config: FeatureConfig handed to the CRF feature extractor.
        C: Inverse L2 regularisation strength of the logistic regression.
        max_iter: Maximum lbfgs iterations.
        model_name: Identifier written into every prediction.
    """

    def __init__(
        self,
        label_mode: str,
        *,
        feature_config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
        C: float = 1.0,
        max_iter: int = 200,
        model_name: str = "gap_improved",
    ) -> None:
        require_scoring_mode(label_mode)  # get_mode_vocabulary also accepts the casing helper "CASE"
        self.vocabulary = get_mode_vocabulary(label_mode)
        if not isinstance(feature_config, FeatureConfig):
            raise TypeError(f"feature_config must be a FeatureConfig, got {type(feature_config).__name__}")
        _check_hyperparameters(C, max_iter, model_name)
        self.label_mode = label_mode
        self.feature_config = feature_config
        self.C = float(C)
        self.max_iter = max_iter
        self.model_name = model_name
        self._vectorizer: DictVectorizer | None = None
        self._classifier: LogisticRegression | None = None
        self._summary: dict[str, int] = {}

    @property
    def training_summary(self) -> dict[str, int]:
        """Copy of {num_sequences, num_tokens, num_features, iterations}; empty before fit()."""
        return dict(self._summary)

    def _crf_view(self) -> CRFModel:
        # Deliberate: reuse the CRF's own record validation and feature extraction instead of copying
        # them, so the CRF and this model see byte-identical inputs and the comparison stays controlled.
        return CRFModel(label_mode=self.label_mode, feature_config=self.feature_config)

    def _fitted(self) -> tuple[DictVectorizer, LogisticRegression]:
        if self._vectorizer is None or self._classifier is None:
            raise RuntimeError("GapClassifier is not fitted. Call fit() first.")
        return self._vectorizer, self._classifier

    def _validated_chunks(self, records: Any) -> Iterator[tuple[list[list[dict[str, Any]]], list[list[str]]]]:
        """Lazily validate and featurise one chunk at a time with the CRF's own validator."""
        crf = self._crf_view()
        for chunk in _chunks(records):
            yield crf._validate_training_records(chunk)

    def _learn_vocabulary(self, records: Any) -> tuple[DictVectorizer, list[str], int]:
        """Pass 1: fit the vectorizer on a stream of feature dicts; also collect the (small) labels."""
        targets: list[str] = []
        sequences = 0

        def token_features() -> Iterator[dict[str, Any]]:
            nonlocal sequences
            for features, labels in self._validated_chunks(records):
                sequences += len(features)
                targets.extend(label for sequence in labels for label in sequence)
                yield from _flat(features)

        return DictVectorizer(sparse=True).fit(token_features()), targets, sequences

    def _vectorize(self, vectorizer: DictVectorizer, records: Any) -> Any:
        """Pass 2: vectorise chunk by chunk and stack the blocks, so only one chunk's dicts are alive."""
        blocks = [vectorizer.transform(_flat(features)) for features, _ in self._validated_chunks(records)]
        return sparse.vstack(blocks, format="csr")

    def fit(self, records: Sequence[Mapping[str, Any]]) -> GapClassifier:
        """Train on records with the same contract as CRFModel.fit; returns self.

        The records are validated and featurised CHUNK_RECORDS at a time, twice (once to learn the feature
        vocabulary, once to build the matrix), instead of holding every feature dict at once. The record
        index in a validation error counts from the start of its chunk, not of ``records``.

        Raises:
            ValueError: Invalid records (as in CRFModel) or fewer than two distinct labels.
        """
        _check_unique_sample_ids(records)
        vectorizer, targets, num_sequences = self._learn_vocabulary(records)
        if len(set(targets)) < 2:
            raise ValueError(f"Training data needs at least two labels, found {sorted(set(targets))}.")
        matrix = self._vectorize(vectorizer, records)
        # Convergence warnings are left visible on purpose.
        classifier = LogisticRegression(C=self.C, max_iter=self.max_iter, solver=SOLVER).fit(matrix, targets)
        self._vectorizer, self._classifier = vectorizer, classifier
        self._summary = {
            "num_sequences": num_sequences,
            "num_tokens": len(targets),
            "num_features": int(matrix.shape[1]),
            "iterations": int(max(classifier.n_iter_)),
        }
        return self

    def predict(self, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Predict labels for gold-free records; same output format as CRFModel.predict.

        Records are featurised and classified CHUNK_RECORDS at a time; the output keeps the input order.
        The record index in a validation error counts from the start of its chunk, not of ``records``.
        """
        vectorizer, classifier = self._fitted()  # RuntimeError when unfitted, before any record is looked at
        crf = self._crf_view()
        return [
            prediction
            for chunk in _chunks(records)
            for prediction in self._predict_chunk(crf, vectorizer, classifier, chunk)
        ]

    def _predict_chunk(
        self, crf: CRFModel, vectorizer: DictVectorizer, classifier: LogisticRegression, records: Any
    ) -> list[dict[str, Any]]:
        sample_ids, lengths, features = crf._prediction_features(records)
        if not sample_ids:
            return []
        flat = [str(label) for label in classifier.predict(vectorizer.transform(_flat(features)))]
        crf._require_known_labels(flat, set(self.vocabulary))
        predictions: list[dict[str, Any]] = []
        start = 0
        for sample_id, length in zip(sample_ids, lengths, strict=True):
            predictions.append({
                "sample_id": sample_id,
                "model_name": self.model_name,
                "label_mode": self.label_mode,
                "predicted_labels": flat[start:start + length],
            })
            start += length
        return predictions

    def save_model(self, path: str | Path) -> None:
        """Write the fitted model as a joblib bundle (RuntimeError when unfitted)."""
        vectorizer, classifier = self._fitted()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "schema_version": SCHEMA_VERSION,
            "model_family": MODEL_FAMILY,
            "model_name": self.model_name,
            "label_mode": self.label_mode,
            "vocabulary": list(self.vocabulary),
            "feature_config": self.feature_config.to_dict(),
            "hyperparameters": {"C": self.C, "max_iter": self.max_iter, "solver": SOLVER},
            "training_summary": self.training_summary,
            "vectorizer": vectorizer,
            "classifier": classifier,
        }, target)

    @classmethod
    def load_model(cls, path: str | Path) -> GapClassifier:
        """Load a model written by save_model. joblib is pickle: load trusted files only.

        Raises:
            FileNotFoundError: ``<name>: model artifact not found``.
            IsADirectoryError: ``<name>: is a folder, not a file``.
            OSError: The file cannot be read (permissions, I/O); passed through as is.
            ValueError: An unreadable file, wrong family or schema, or inconsistent or missing parts.
        """
        source = Path(path)
        require_file(source, "model artifact")
        try:
            bundle = joblib.load(source)  # pickle: trusted files only (see docstring)
        except OSError:
            raise  # permission and I/O errors are not a corrupt artifact
        except Exception as exc:  # any unpickling failure means a corrupt or foreign file
            raise ValueError(f"{source.name}: corrupt or unreadable model artifact") from exc
        problem = _bundle_problem(bundle)
        if problem is not None:
            raise ValueError(f"{source.name}: corrupt model artifact: {problem}")
        hyper = bundle["hyperparameters"]
        model = cls(
            bundle["label_mode"],
            feature_config=FeatureConfig.from_dict(bundle["feature_config"]),
            C=hyper.get("C"),
            max_iter=hyper.get("max_iter"),
            model_name=bundle["model_name"],
        )
        model._vectorizer = bundle["vectorizer"]
        model._classifier = bundle["classifier"]
        model._summary = dict(bundle["training_summary"])
        return model
