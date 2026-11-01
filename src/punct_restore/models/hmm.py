"""Supervised HMM baseline owned by Member 2.

States label punctuation after each word (or punctuation/case in mode C).
Training uses counts, additive smoothing and a train-only rare-word bucket.
No dataset loader, annotation projection or text reconstruction lives here.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from punct_restore.file_io import write_new_file

from .hmm_distributions import HMMDistributions
from .hmm_validation import positive_integer as _positive_integer
from .hmm_validation import read_artifact, validate_counts
from .hmm_validation import validated_tokens as _tokens

__all__ = ["HMMModel"]

_STANDARD = ("O", "COMMA", "PERIOD", "QUESTION")
_EXTENDED = _STANDARD + ("EXCLAMATION", "COLON", "SEMICOLON", "DASH")
_CASING = ("LOWER", "FIRST_UPPER", "ALL_UPPER")
_FORMAT_VERSION = 1


def _json_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is not None and not isinstance(value, Mapping):
        raise ValueError("metadata must be a JSON-serializable mapping")
    try:
        return json.loads(json.dumps(dict(value or {}), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain finite JSON-serializable values") from exc


def _observation(record: Any, seen_ids: set[str]) -> tuple[str, list[str]]:
    """Validate the observable fields of one record; seen_ids is only read, the caller remembers the ID."""
    if not isinstance(record, Mapping):
        raise ValueError("each record must be a mapping")
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise ValueError("record schema_version must be integer 1")
    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("sample_id must be a nonempty string")
    if sample_id in seen_ids:
        raise ValueError(f"duplicate sample_id: {sample_id}")
    return sample_id, _tokens(record.get("tokens"))


@dataclass(frozen=True)
class _Counts:
    start: list[int]
    transition: list[list[int]]
    end: list[int]
    state: list[int]
    emissions: list[dict[str, int]]
    unknown: list[int]
    vocabulary: list[str]
    num_sequences: int


def _count_sequences(
    sequences: Iterable[tuple[list[str], list[int]]], k: int, min_token_count: int
) -> _Counts:
    """Count k-state (tokens, states) sequences; words seen fewer than min_token_count times fold into unknown."""
    start, end, state = [0] * k, [0] * k, [0] * k
    transition = [[0] * k for _ in range(k)]
    raw_emissions: list[Counter[str]] = [Counter() for _ in range(k)]
    word_frequency: Counter[str] = Counter()
    num_sequences = 0
    for tokens, states in sequences:
        start[states[0]] += 1
        end[states[-1]] += 1
        for position, (word, current) in enumerate(zip(tokens, states, strict=True)):
            state[current] += 1
            raw_emissions[current][word] += 1
            word_frequency[word] += 1
            if position:
                transition[states[position - 1]][current] += 1
        num_sequences += 1
    vocabulary = sorted(word for word, frequency in word_frequency.items() if frequency >= min_token_count)
    kept = set(vocabulary)
    return _Counts(
        start=start,
        transition=transition,
        end=end,
        state=state,
        emissions=[{word: n for word, n in emissions.items() if word in kept} for emissions in raw_emissions],
        unknown=[sum(n for word, n in emissions.items() if word not in kept) for emissions in raw_emissions],
        vocabulary=vocabulary,
        num_sequences=num_sequences,
    )


class HMMModel:
    """Count-trained first-order HMM with an explicit end transition.

    Supply ordered labels from configs/label_sets.yaml. No normalization or
    cross-mode collapse is performed. Hyperparameter defaults are starting
    settings to select on validation, not experimentally chosen values.
    """

    def __init__(
        self,
        *,
        label_mode: str,
        labels: Sequence[str],
        smoothing: float = 0.1,
        min_token_count: int = 2,
    ) -> None:
        universes = {
            "A": set(_STANDARD),
            "B": set(_EXTENDED),
            "C": {f"{punct}|{case}" for punct in _STANDARD for case in _CASING},
        }
        if not isinstance(label_mode, str) or label_mode not in universes:
            raise ValueError("label_mode must be A, B or C")
        if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
            raise ValueError("labels must be an ordered sequence of label strings")
        if any(not isinstance(label, str) for label in labels):
            raise ValueError("labels must contain strings")
        if len(labels) != len(set(labels)) or set(labels) != universes[label_mode]:
            raise ValueError(f"labels must be the exact unique vocabulary for mode {label_mode}")
        if type(smoothing) not in (int, float):
            raise ValueError("smoothing must be a finite number greater than zero")
        try:
            alpha = float(smoothing)
        except OverflowError as exc:
            raise ValueError("smoothing must be a finite number greater than zero") from exc
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("smoothing must be a finite number greater than zero")
        if not _positive_integer(min_token_count):
            raise ValueError("min_token_count must be a positive integer")
        self._label_mode = label_mode
        self._labels = tuple(labels)
        self._label_index = {label: index for index, label in enumerate(labels)}
        self._smoothing = alpha
        self._min_token_count = min_token_count
        self._counts: _Counts | None = None
        self._metadata: dict[str, Any] = {}

    @property
    def label_mode(self) -> str:
        return self._label_mode

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a detached provenance copy, never mutable model state."""
        return _json_metadata(self._metadata)

    @property
    def training_summary(self) -> dict[str, int]:
        counts = self._require_fitted()
        return {
            "num_sequences": counts.num_sequences,
            "num_tokens": sum(counts.state),
            "vocabulary_size": len(counts.vocabulary),
            "unknown_training_tokens": sum(counts.unknown),
        }

    @property
    def distributions(self) -> HMMDistributions:
        """A read-only copy of the log-probability tables, for checks; never the model's own tables."""
        self._require_fitted()
        return HMMDistributions.copy_of(
            self._log_start, self._log_transition, self._log_end, self._vocabulary,
            self._log_emissions, self._log_default, self._log_unknown,
        )

    def fit(
        self,
        training_records: Iterable[Mapping[str, Any]],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> HMMModel:
        """Fit annotated train records; a failed refit preserves old state."""
        provenance = _json_metadata(metadata)
        counts = _count_sequences(
            self._train_sequences(training_records), len(self.labels), self._min_token_count
        )
        if not counts.num_sequences:
            raise ValueError("training_records must contain at least one nonempty train sequence")
        self._install(counts, provenance)
        return self

    def _train_sequences(
        self, training_records: Iterable[Mapping[str, Any]]
    ) -> Iterator[tuple[list[str], list[int]]]:
        """Validate the train records one by one, in order, and yield each one's tokens and state indices."""
        seen_ids: set[str] = set()
        for record in training_records:
            sample_id, tokens = _observation(record, seen_ids)
            seen_ids.add(sample_id)
            yield tokens, self._record_states(record, sample_id, tokens)

    def _record_states(self, record: Mapping[str, Any], sample_id: str, tokens: list[str]) -> list[int]:
        """Validate one train record and return its state indices."""
        if record.get("split") != "train":
            raise ValueError(f"{sample_id}: fit accepts only split=train records")
        if record.get("label_mode") != self.label_mode:
            raise ValueError(
                f"{sample_id}: label_mode does not match model; no implicit collapse"
            )
        if not isinstance(record.get("source"), str) or not record["source"].strip():
            raise ValueError(f"{sample_id}: source must be a nonempty string")
        punct, casing = record.get("punct_labels"), record.get("case_labels")
        if not isinstance(punct, list) or not isinstance(casing, list):
            raise ValueError(f"{sample_id}: punct_labels and case_labels must be lists")
        if len(punct) != len(tokens) or len(casing) != len(tokens):
            raise ValueError(f"{sample_id}: token and label lengths must match")
        punctuation = _EXTENDED if self.label_mode == "B" else _STANDARD
        if any(label not in punctuation for label in punct):
            raise ValueError(
                f"{sample_id}: invalid punctuation label for mode {self.label_mode}"
            )
        if any(label not in _CASING for label in casing):
            raise ValueError(f"{sample_id}: invalid casing label")
        targets = (
            [f"{p}|{c}" for p, c in zip(punct, casing, strict=True)]
            if self.label_mode == "C"
            else punct
        )
        return [self._label_index[label] for label in targets]

    def _require_fitted(self) -> _Counts:
        if self._counts is None:
            raise RuntimeError("HMMModel is not fitted; call fit or load_model first")
        return self._counts

    def _install(self, counts: _Counts, metadata: dict[str, Any]) -> None:
        alpha, k = self._smoothing, len(self.labels)

        def denominator(total: int, size: int) -> float:
            value = total + alpha * size
            if not math.isfinite(value):
                raise ValueError("smoothing is too large for this vocabulary")
            return math.log(value)

        start_denominator = denominator(counts.num_sequences, k)
        log_start = [math.log(count + alpha) - start_denominator for count in counts.start]
        log_transition, log_end, log_default, log_unknown, log_emissions = [], [], [], [], []
        for state in range(k):
            transition_denominator = denominator(counts.state[state], k + 1)
            log_transition.append(
                [
                    math.log(count + alpha) - transition_denominator
                    for count in counts.transition[state]
                ]
            )
            log_end.append(math.log(counts.end[state] + alpha) - transition_denominator)
            emission_denominator = denominator(counts.state[state], len(counts.vocabulary) + 1)
            log_default.append(math.log(alpha) - emission_denominator)
            log_unknown.append(math.log(counts.unknown[state] + alpha) - emission_denominator)
            log_emissions.append(
                {
                    word: math.log(count + alpha) - emission_denominator
                    for word, count in counts.emissions[state].items()
                }
            )
        self._counts, self._metadata = counts, metadata
        self._vocabulary = frozenset(counts.vocabulary)
        self._log_start, self._log_transition, self._log_end = log_start, log_transition, log_end
        self._log_default, self._log_unknown = log_default, log_unknown
        self._log_emissions = log_emissions

    def _emission(self, state: int, word: str) -> float:
        if word not in self._vocabulary:
            return self._log_unknown[state]
        return self._log_emissions[state].get(word, self._log_default[state])

    def decode(self, tokens: Sequence[str]) -> list[str]:
        """Return the best path including END; ties follow configured label order.

        Complexity is O(T*K^2) time and O(T*K) backpointer space.
        """
        self._require_fitted()
        words, k = _tokens(tokens), len(self.labels)
        scores = [self._log_start[state] + self._emission(state, words[0]) for state in range(k)]
        backpointers: list[list[int]] = []
        for word in words[1:]:
            next_scores, pointers = [], []
            for state in range(k):
                previous = max(
                    range(k), key=lambda prev: scores[prev] + self._log_transition[prev][state]
                )
                next_scores.append(
                    scores[previous]
                    + self._log_transition[previous][state]
                    + self._emission(state, word)
                )
                pointers.append(previous)
            scores = next_scores
            backpointers.append(pointers)
        final = max(range(k), key=lambda state: scores[state] + self._log_end[state])
        path = [final]
        for pointers in reversed(backpointers):
            path.append(pointers[path[-1]])
        return [self.labels[state] for state in reversed(path)]

    def sequence_log_probability(self, tokens: Sequence[str], labels: Sequence[str]) -> float:
        """Return log P(tokens, specified states, END), not confidence or log P(tokens)."""
        self._require_fitted()
        words = _tokens(tokens)
        if (
            isinstance(labels, (str, bytes))
            or not isinstance(labels, Sequence)
            or len(labels) != len(words)
        ):
            raise ValueError("one state label is required per token")
        if any(not isinstance(label, str) or label not in self._label_index for label in labels):
            raise ValueError("state labels must belong to the model vocabulary")
        states = [self._label_index[label] for label in labels]
        score = self._log_start[states[0]] + self._emission(states[0], words[0])
        for position in range(1, len(words)):
            score += self._log_transition[states[position - 1]][states[position]]
            score += self._emission(states[position], words[position])
        return score + self._log_end[states[-1]]

    def predict(self, records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Preserve IDs/order; read only schema_version, sample_id and tokens.

        Extra gold labels, original text and metadata are ignored, never copied
        into output or consulted when decoding. No caller records are mutated.
        """
        self._require_fitted()
        predictions: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for record in records:
            sample_id, tokens = _observation(record, seen_ids)
            seen_ids.add(sample_id)
            predictions.append(
                {
                    "sample_id": sample_id,
                    "model_name": "hmm",
                    "label_mode": self.label_mode,
                    "predicted_labels": self.decode(tokens),
                }
            )
        return predictions

    def save_model(self, path: str | Path) -> None:
        """Write versioned JSON; refuse to overwrite an existing file, and remove a file whose write failed."""
        counts = self._require_fitted()
        payload = {
            "format_version": _FORMAT_VERSION,
            "model_name": "hmm",
            "config": {
                "label_mode": self.label_mode,
                "labels": list(self.labels),
                "smoothing": self._smoothing,
                "min_token_count": self._min_token_count,
            },
            "counts": asdict(counts),
            "metadata": self.metadata,
        }
        serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_new_file(destination, serialized + "\n")

    @classmethod
    def load_model(cls, path: str | Path) -> HMMModel:
        """Validate saved counts and rebuild probabilities without pickle.

        Raises:
            OSError: If the file cannot be read.
            ValueError: One line that starts with the file name: not valid UTF-8 or JSON (also one nested too
                deeply), the wrong shape, or counts that do not add up.
        """
        payload = read_artifact(Path(path))
        try:
            if (
                type(payload["format_version"]) is not int
                or payload["format_version"] != _FORMAT_VERSION
                or payload["model_name"] != "hmm"
            ):
                raise ValueError("unsupported HMM artifact version or family")
            model = cls(**payload["config"])
            counts = _Counts(**payload["counts"])
            model._validate_counts(counts)
            model._install(counts, _json_metadata(payload["metadata"]))
            return model
        except ValueError as exc:
            raise ValueError(f"{Path(path).name}: invalid HMM artifact: {exc}") from exc
        except OverflowError as exc:  # a count or a smoothing value too large for a float
            raise ValueError(f"{Path(path).name}: invalid HMM artifact: a number is out of range") from exc

    def _validate_counts(self, counts: _Counts) -> None:
        validate_counts(counts, len(self.labels), self._min_token_count)
