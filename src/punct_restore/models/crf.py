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
"""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import joblib
import sklearn_crfsuite

from punct_restore.data.labeling import join_labels, load_label_sets
from punct_restore.features.crf_features import (
    FeatureConfig,
    extract_sequence_features,
)

CASING_VOCABULARY: tuple[str, ...] = ("LOWER", "FIRST_UPPER", "ALL_UPPER")

MODEL_SCHEMA_VERSION = 1


def get_mode_vocabulary(label_mode: str, *, config_path: str | Path | None = None) -> list[str]:
    """Retrieve the authoritative ordered label vocabulary for a given mode from configs/label_sets.yaml.

    Args:
        label_mode: 'A', 'B', or 'C'.
        config_path: Optional path to label_sets.yaml (defaults to repository configuration).

    Returns:
        List of valid label strings in configuration order.

    Raises:
        ValueError: If label_mode is unsupported or configuration structure is invalid.
        FileNotFoundError: If configuration file does not exist.
    """
    if label_mode not in ("A", "B", "C"):
        raise ValueError(f"Unsupported label_mode: {label_mode!r}. Must be 'A', 'B', or 'C'.")

    label_sets = load_label_sets(str(config_path)) if config_path is not None else load_label_sets()
    return list(label_sets[label_mode])


class CRFModel:
    """Production-quality Linear-chain Conditional Random Field for sequence labeling.

    Attributes:
        label_mode: Active label mode ('A', 'B', or 'C').
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

    @staticmethod
    def _validate_init_params(
        label_mode: str,
        algorithm: str,
        c1: float,
        c2: float,
        max_iterations: int,
    ) -> None:
        if label_mode not in ("A", "B", "C"):
            raise ValueError(f"Invalid label_mode: {label_mode!r}. Must be 'A', 'B', or 'C'.")
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
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise TypeError(f"training_records must be a Sequence, got {type(records).__name__}")
        if len(records) == 0:
            raise ValueError("training_records sequence cannot be empty.")

        seen_ids: set[str] = set()
        X: list[list[dict[str, Any]]] = []
        y: list[list[str]] = []

        vocab_set = set(self.vocabulary)
        casing_set = set(CASING_VOCABULARY)

        for idx, rec in enumerate(records):
            where = f"Record at index {idx}"
            if not isinstance(rec, Mapping):
                raise TypeError(f"{where} must be a Mapping, got {type(rec).__name__}")

            # Validate sample_id
            sample_id = rec.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"{where}: sample_id must be a non-empty string.")
            if sample_id in seen_ids:
                raise ValueError(f"{where}: duplicate sample_id {sample_id!r} detected.")
            seen_ids.add(sample_id)

            # Prevent training on non-train data (leakage prevention)
            split = rec.get("split")
            if split is not None and split != "train":
                raise ValueError(
                    f"{where} ({sample_id}): cannot train on record with split={split!r}. "
                    "Training data must have split='train'."
                )

            # Validate tokens
            tokens = rec.get("tokens")
            if not isinstance(tokens, list) or len(tokens) == 0:
                raise ValueError(f"{where} ({sample_id}): tokens must be a non-empty list.")
            if any(not isinstance(t, str) or not t.strip() for t in tokens):
                raise ValueError(f"{where} ({sample_id}): tokens must contain non-empty strings.")

            seq_len = len(tokens)

            # Derive targets based on mode
            punct_labels = rec.get("punct_labels")
            if not isinstance(punct_labels, list) or len(punct_labels) != seq_len:
                raise ValueError(
                    f"{where} ({sample_id}): punct_labels length ({len(punct_labels) if isinstance(punct_labels, list) else None}) "
                    f"does not match tokens length ({seq_len})."
                )

            if self.label_mode in ("A", "B"):
                unknown_labels = [lbl for lbl in punct_labels if lbl not in vocab_set]
                if unknown_labels:
                    raise ValueError(
                        f"{where} ({sample_id}): unknown punct_labels {unknown_labels!r} "
                        f"for Mode {self.label_mode}."
                    )
                target_sequence = list(punct_labels)

            elif self.label_mode == "C":
                case_labels = rec.get("case_labels")
                if not isinstance(case_labels, list) or len(case_labels) != seq_len:
                    raise ValueError(
                        f"{where} ({sample_id}): case_labels length ({len(case_labels) if isinstance(case_labels, list) else None}) "
                        f"does not match tokens length ({seq_len})."
                    )
                unknown_cases = [c for c in case_labels if c not in casing_set]
                if unknown_cases:
                    raise ValueError(
                        f"{where} ({sample_id}): unknown case_labels {unknown_cases!r}."
                    )

                try:
                    target_sequence = join_labels(punct_labels, case_labels, separator="|")
                except Exception as e:
                    raise ValueError(f"{where} ({sample_id}): failed to derive Mode C targets: {e}") from e

                unknown_joint = [lbl for lbl in target_sequence if lbl not in vocab_set]
                if unknown_joint:
                    raise ValueError(
                        f"{where} ({sample_id}): derived joint labels {unknown_joint!r} "
                        f"outside Mode C vocabulary."
                    )
            else:
                raise ValueError(f"Unsupported label_mode: {self.label_mode}")

            # Extract features strictly using observable tokens (gold leakage prevention)
            features = self._feature_extractor(tokens, config=self.feature_config)
            if len(features) != seq_len:
                raise ValueError(
                    f"{where} ({sample_id}): feature extractor returned {len(features)} dicts "
                    f"for {seq_len} tokens."
                )

            X.append(features)
            y.append(target_sequence)

        return X, y

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
            label_mode: Optional label mode override ('A', 'B', or 'C').
            config: Optional hyperparameters/feature configuration overrides.
            feature_extractor: Optional callable for feature extraction.

        Returns:
            The fitted CRFModel instance.
        """
        # Apply runtime overrides if provided
        if label_mode is not None:
            if label_mode not in ("A", "B", "C"):
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
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise TypeError(f"records must be a Sequence, got {type(records).__name__}")

        X: list[list[dict[str, Any]]] = []
        sample_ids: list[str] = []
        token_lengths: list[int] = []

        for idx, rec in enumerate(records):
            where = f"Prediction record at index {idx}"
            if not isinstance(rec, Mapping):
                raise TypeError(f"{where} must be a Mapping, got {type(rec).__name__}")

            sample_id = rec.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"{where}: sample_id must be a non-empty string.")

            tokens = rec.get("tokens")
            if not isinstance(tokens, list) or len(tokens) == 0:
                raise ValueError(f"{where} ({sample_id}): tokens must be a non-empty list.")
            if any(not isinstance(t, str) or not t.strip() for t in tokens):
                raise ValueError(f"{where} ({sample_id}): tokens must contain non-empty strings.")

            features = self._feature_extractor(tokens, config=self.feature_config)
            X.append(features)
            sample_ids.append(sample_id)
            token_lengths.append(len(tokens))

        return sample_ids, token_lengths, X

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
        if not self._is_fitted or self._backend is None:
            raise RuntimeError("Cannot save an unfitted CRFModel.")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        bundle: dict[str, Any] = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "model_family": "crf",
            "model_name": self.model_name,
            "label_mode": self.label_mode,
            "vocabulary": list(self.vocabulary),
            "hyperparameters": {
                "algorithm": self.algorithm,
                "c1": self.c1,
                "c2": self.c2,
                "max_iterations": self.max_iterations,
                "all_possible_transitions": self.all_possible_transitions,
            },
            "feature_config": self.feature_config.to_dict(),
            "training_metadata": copy.deepcopy(self._training_metadata),
            "environment": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "sklearn_crfsuite": getattr(sklearn_crfsuite, "__version__", "unknown"),
                "joblib": getattr(joblib, "__version__", "unknown"),
            },
            "backend_state": self._backend,
        }

        joblib.dump(bundle, save_path)

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
        load_path = Path(path)
        if not load_path.is_file():
            raise FileNotFoundError(f"Model artifact not found at {load_path}")

        try:
            bundle = joblib.load(load_path)
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
        if label_mode not in ("A", "B", "C"):
            raise ValueError(f"Invalid label_mode in artifact: {label_mode!r}")

        hp = bundle.get("hyperparameters", {})
        feat_dict = bundle.get("feature_config", {})
        backend_state = bundle.get("backend_state")

        if backend_state is None:
            raise ValueError("Corrupt artifact: missing backend_state.")

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
