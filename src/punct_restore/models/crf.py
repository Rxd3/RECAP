"""Linear-chain Conditional Random Field (CRF) model for punctuation and casing restoration.

Owned by Member 3 (Omar Alzyod).
Adheres strictly to the shared model contract (docs/MODEL_CONTRACT.md):
- Common interface: fit(training_records, *, label_mode, config, feature_extractor=None)
- Record-oriented prediction: predict(records) -> list[dict[str, Any]]
- Per-token posteriors: predict_marginals(records) -> list[list[dict[str, float]]]
- Invariant model persistence: save_model(path), load_model(path)
- Full support for:
    * Mode A (4 standard punctuation classes)
    * Mode B (8 extended punctuation classes)
    * Mode C (12 joint punctuation and casing classes: 4 punct x 3 casing)
    * Helper mode CASE (3 casing classes, trained from a mode C record's case_labels; it is a
      building block for factored joint prediction, not a scoring mode)
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import sklearn_crfsuite

from punct_restore.features.crf_features import (
    FeatureConfig,
    extract_sequence_features,
)
from punct_restore.models.crf_io import read_bundle, restore_model, write_model
from punct_restore.models.crf_modes import (
    LABEL_MODES,
    SCORING_MODES,
    get_mode_vocabulary,
    require_scoring_mode,
)
from punct_restore.models.crf_records import prediction_features, validate_training_records

__all__ = ["CRFModel", "SCORING_MODES", "get_mode_vocabulary", "require_scoring_mode"]


class CRFModel:
    """Production-quality Linear-chain Conditional Random Field for sequence labeling.

    Attributes:
        label_mode: Active label mode ('A', 'B', 'C', or the casing helper 'CASE').
        vocabulary: Exact ordered list of permitted target labels.
        algorithm: Optimization algorithm (default: 'lbfgs').
        c1: L1 regularization coefficient.
        c2: L2 regularization coefficient.
        max_iterations: Maximum optimization iterations.
        all_possible_transitions: Whether to learn transitions between all label pairs.
        feature_config: FeatureConfig instance controlling feature extraction.
        model_name: Descriptive model identifier (e.g. 'crf_basic', 'crf_improved').
    """

    def __init__(
        self,
        *,
        label_mode: str = "A",
        algorithm: str = "lbfgs",
        c1: float = 0.1,
        c2: float = 0.1,
        max_iterations: int = 100,
        all_possible_transitions: bool = True,
        feature_config: FeatureConfig | Mapping[str, Any] | None = None,
        feature_extractor: Callable[..., list[dict[str, Any]]] | None = None,
        model_name: str | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self._validate_init_params(label_mode, algorithm, c1, c2, max_iterations)

        self._config_path = config_path
        self.label_mode = label_mode
        self.vocabulary = get_mode_vocabulary(label_mode, config_path=config_path)
        self.algorithm = algorithm
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.max_iterations = int(max_iterations)
        self.all_possible_transitions = bool(all_possible_transitions)

        # Feature extraction setup
        if feature_config is None:
            self.feature_config = FeatureConfig.basic()
        elif isinstance(feature_config, FeatureConfig):
            self.feature_config = feature_config
        elif isinstance(feature_config, Mapping):
            self.feature_config = FeatureConfig.from_dict(feature_config)
        else:
            raise TypeError(f"Invalid feature_config type: {type(feature_config).__name__}")

        self._feature_extractor = feature_extractor or extract_sequence_features
        self.model_name = model_name or f"crf_{self.feature_config.preset}"

        # State tracking
        self._is_fitted: bool = False
        self._backend: sklearn_crfsuite.CRF | None = None
        self._training_metadata: dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        """Return True if model parameters have been estimated from training data."""
        return self._is_fitted

    @property
    def training_log(self) -> Any:
        """The optimizer's log of the last fit (``iterations`` and ``lines``); None before a fit."""
        return getattr(self._backend, "training_log_", None)

    @staticmethod
    def _validate_init_params(
        label_mode: str,
        algorithm: str,
        c1: float,
        c2: float,
        max_iterations: int,
    ) -> None:
        if label_mode not in LABEL_MODES:
            raise ValueError(f"Invalid label_mode: {label_mode!r}. Must be 'A', 'B', 'C', or 'CASE'.")
        if not isinstance(algorithm, str) or not algorithm.strip():
            raise ValueError("algorithm must be a non-empty string.")
        if not isinstance(c1, (int, float)) or math.isnan(c1) or math.isinf(c1) or c1 < 0:
            raise ValueError(f"c1 must be a non-negative finite float, got {c1!r}")
        if not isinstance(c2, (int, float)) or math.isnan(c2) or math.isinf(c2) or c2 < 0:
            raise ValueError(f"c2 must be a non-negative finite float, got {c2!r}")
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise ValueError(f"max_iterations must be a positive integer, got {max_iterations!r}")

    def _validate_training_records(
        self, records: Sequence[Mapping[str, Any]]
    ) -> tuple[list[list[dict[str, Any]]], list[list[str]]]:
        """Validate input training records against schema and extract features and targets."""
        return validate_training_records(
            records,
            label_mode=self.label_mode,
            vocabulary=self.vocabulary,
            feature_extractor=self._feature_extractor,
            feature_config=self.feature_config,
        )

    def _apply_fit_overrides(self, label_mode: str | None, config: Mapping[str, Any] | None) -> None:
        """Apply the runtime label_mode and config overrides given to fit()."""
        if label_mode is not None:
            if label_mode not in LABEL_MODES:
                raise ValueError(f"Invalid label_mode: {label_mode!r}")
            self.label_mode = label_mode
            self.vocabulary = get_mode_vocabulary(label_mode, config_path=self._config_path)

        if config is not None:
            if "algorithm" in config:
                self.algorithm = config["algorithm"]
            if "c1" in config:
                self.c1 = float(config["c1"])
            if "c2" in config:
                self.c2 = float(config["c2"])
            if "max_iterations" in config:
                self.max_iterations = int(config["max_iterations"])
            if "all_possible_transitions" in config:
                self.all_possible_transitions = bool(config["all_possible_transitions"])
            if "feature_config" in config:
                fc = config["feature_config"]
                self.feature_config = fc if isinstance(fc, FeatureConfig) else FeatureConfig.from_dict(fc)
                self.model_name = f"crf_{self.feature_config.preset}"
            self._validate_init_params(self.label_mode, self.algorithm, self.c1, self.c2, self.max_iterations)

    def _record_training_metadata(
        self, X: list[list[dict[str, Any]]], y: list[list[str]], config: Mapping[str, Any] | None
    ) -> None:
        """Record training provenance (counts, observed labels, feature preset) after a fit."""
        unique_targets = sorted({lbl for seq in y for lbl in seq})
        self._training_metadata = {
            "num_sequences": len(X),
            "num_tokens": sum(len(seq) for seq in X),
            "observed_labels": unique_targets,
            "label_mode": self.label_mode,
            "feature_preset": self.feature_config.preset,
            "feature_version": self.feature_config.version,
        }
        if config is not None and "provenance" in config and isinstance(config["provenance"], Mapping):
            self._training_metadata["provenance"] = dict(config["provenance"])

    def fit(
        self,
        training_records: Sequence[Mapping[str, Any]],
        *,
        label_mode: str | None = None,
        config: Mapping[str, Any] | None = None,
        feature_extractor: Callable[..., list[dict[str, Any]]] | None = None,
    ) -> CRFModel:
        """Train the CRF model on annotated sequence records.

        Args:
            training_records: Sequence of validated training records.
            label_mode: Optional label mode override ('A', 'B', 'C', or 'CASE').
            config: Optional hyperparameters/feature configuration overrides.
            feature_extractor: Optional callable for feature extraction.

        Returns:
            The fitted CRFModel instance.
        """
        self._apply_fit_overrides(label_mode, config)

        if feature_extractor is not None:
            self._feature_extractor = feature_extractor

        # Validate records and extract X, y
        X, y = self._validate_training_records(training_records)

        # Initialize backend CRF estimator
        self._backend = sklearn_crfsuite.CRF(
            algorithm=self.algorithm,
            c1=self.c1,
            c2=self.c2,
            max_iterations=self.max_iterations,
            all_possible_transitions=self.all_possible_transitions,
            keep_tempfiles=False,
        )

        # Fit model parameters
        self._backend.fit(X, y)
        self._is_fitted = True

        # Record training provenance
        self._record_training_metadata(X, y, config)

        return self

    def _fitted_backend(self) -> sklearn_crfsuite.CRF:
        """Return the fitted backend estimator or raise the standard not-fitted error."""
        if not self._is_fitted or self._backend is None:
            raise RuntimeError("CRFModel is not fitted. Call fit() before predict().")
        return self._backend

    def _prediction_features(
        self, records: Sequence[Mapping[str, Any]]
    ) -> tuple[list[str], list[int], list[list[dict[str, Any]]]]:
        """Validate gold-free records and extract features.

        Returns:
            (sample ids, token lengths, feature sequences); all empty for empty records.
        """
        return prediction_features(
            records, feature_extractor=self._feature_extractor, feature_config=self.feature_config
        )

    def predict(self, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Predict labels for gold-free input observation records.

        Args:
            records: Sequence of input observation mappings, each containing:
                - 'sample_id': Non-empty unique identifier string.
                - 'tokens': Non-empty list of non-empty lowercased strings.

        Returns:
            List of prediction dictionaries conforming to docs/MODEL_CONTRACT.md:
            {
                "sample_id": str,
                "model_name": str,
                "label_mode": str,
                "predicted_labels": list[str]
            }
        """
        backend = self._fitted_backend()
        sample_ids, token_lengths, X = self._prediction_features(records)
        if not X:
            return []

        # Perform backend batch prediction
        raw_preds = backend.predict(X)

        predictions: list[dict[str, Any]] = []
        vocab_set = set(self.vocabulary)

        for s_id, expected_len, pred_seq in zip(sample_ids, token_lengths, raw_preds, strict=True):
            seq_list = [str(lbl) for lbl in pred_seq]
            if len(seq_list) != expected_len:
                raise ValueError(
                    f"Model predicted {len(seq_list)} labels for sequence {s_id} with {expected_len} tokens."
                )
            self._require_known_labels(seq_list, vocab_set)

            predictions.append({
                "sample_id": s_id,
                "model_name": self.model_name,
                "label_mode": self.label_mode,
                "predicted_labels": seq_list,
            })

        return predictions

    def predict_marginals(
        self, records: Sequence[Mapping[str, Any]]
    ) -> list[list[dict[str, float]]]:
        """Per-token posterior label probabilities for gold-free input records.

        Args:
            records: Same input records as predict().

        Returns:
            One list per record, one dict per token mapping EVERY label of the active mode
            vocabulary (in vocabulary order) to its marginal probability. Labels never seen
            in training get 0.0. Empty records give [].
        """
        backend = self._fitted_backend()
        sample_ids, token_lengths, X = self._prediction_features(records)
        if not X:
            return []

        raw_marginals = backend.predict_marginals(X)
        vocab_set = set(self.vocabulary)
        results: list[list[dict[str, float]]] = []

        for s_id, expected_len, seq in zip(sample_ids, token_lengths, raw_marginals, strict=True):
            if len(seq) != expected_len:
                raise ValueError(
                    f"Model produced {len(seq)} marginal distributions for sequence {s_id} "
                    f"with {expected_len} tokens."
                )
            results.append([self._token_marginals(probs, vocab_set) for probs in seq])

        return results

    def _token_marginals(
        self, token_probs: Mapping[str, float], vocab_set: set[str]
    ) -> dict[str, float]:
        """Build a new full-vocabulary probability dict for one token."""
        self._require_known_labels(token_probs, vocab_set)
        return {label: float(token_probs.get(label, 0.0)) for label in self.vocabulary}

    def _require_known_labels(self, labels: Iterable[str], vocab_set: set[str]) -> None:
        """Raise ValueError when a predicted label is outside the declared Mode vocabulary."""
        for lbl in labels:
            if lbl not in vocab_set:
                raise ValueError(
                    f"Model predicted label {lbl!r} outside declared Mode {self.label_mode} vocabulary."
                )

    def predict_tokens(self, tokens: Sequence[str], *, sample_id: str = "seq_0") -> list[str]:
        """Convenience method predicting a single sequence of string tokens.

        Acts as an adapter for single-sequence pipeline integration (e.g., Member 6).
        """
        rec = {"sample_id": sample_id, "tokens": list(tokens)}
        res = self.predict([rec])
        return res[0]["predicted_labels"] if res else []

    def save_model(self, path: str | Path) -> None:
        """Save fitted model and complete compatibility metadata to disk.

        Args:
            path: Target file path (.joblib or .pkl).
        """
        write_model(self, path)

    @classmethod
    def load_model(cls, path: str | Path) -> CRFModel:
        """Load and reconstruct a fitted CRFModel from a verified disk artifact.

        Args:
            path: Path to serialized artifact file.

        Returns:
            Equivalent fitted CRFModel instance.

        Raises:
            FileNotFoundError: If the artifact path does not exist.
            ValueError: If artifact schema, model family, or metadata is corrupted/incompatible.
        """
        return restore_model(cls, read_bundle(path))

    def _to_bundle(self) -> dict[str, Any]:
        """The private state a saved artifact keeps beside the public settings; the counterpart of ``_from_bundle``."""
        if not self._is_fitted or self._backend is None:
            raise RuntimeError("Cannot save an unfitted CRFModel.")
        return {"training_metadata": copy.deepcopy(self._training_metadata), "backend_state": self._backend}

    @classmethod
    def _from_bundle(cls, bundle: Mapping[str, Any]) -> CRFModel:
        """A fitted model rebuilt from the bundle of a saved artifact (a bundle checked by ``read_bundle``)."""
        hp = bundle.get("hyperparameters", {})
        model = cls(
            label_mode=bundle.get("label_mode"),
            algorithm=hp.get("algorithm", "lbfgs"),
            c1=hp.get("c1", 0.1),
            c2=hp.get("c2", 0.1),
            max_iterations=hp.get("max_iterations", 100),
            all_possible_transitions=hp.get("all_possible_transitions", True),
            feature_config=FeatureConfig.from_dict(bundle.get("feature_config", {})),
            model_name=bundle.get("model_name"),
        )
        model._backend = bundle.get("backend_state")
        model._is_fitted = True
        model._training_metadata = bundle.get("training_metadata", {})
        return model
