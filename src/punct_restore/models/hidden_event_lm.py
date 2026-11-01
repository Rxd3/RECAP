"""Hidden-event language model for punctuation restoration (comparison method H; Stolcke & Shriberg style).

Punctuation is modelled as hidden events inserted in the word stream. Training turns each utterance into

    <s> ... <s>  w1 [e1]  w2 [e2]  ...  wn [en]  </s>

where ``ei`` is the event token of the punctuation after word i (``<COMMA>``, ``<PERIOD>``, ...) and is absent
when that label is ``O``. An n-gram language model over this mixed stream (interpolated absolute discounting, see
``ngram_lm``) learns which events tend to follow which words, and the end symbol ``</s>`` is scored together
with the last word's event. Prediction is exact Viterbi over the hidden choice after each word (no event, or one
of the mode's events): a path's score is the sum of the log probabilities of every emitted token, and the DP state
is the tuple of the last ``order - 1`` emitted tokens, which is all an order-n model can see, so paths that leave
the same history are merged without losing the optimum.

Mode C (``PUNCT|CASE``) is a FACTORED pipeline, not a joint model: the event LM predicts the punctuation part
(the mode A punctuation set) and a ``UnigramTruecaser`` (see ``truecaser``) assigns the casing part from per-word
training counts.

Artifact: versioned JSON (``format_version`` 1, ``model_family`` "hidden_event_lm"). Token ids are 0 ``<s>``,
1 ``</s>``, 2 ``<unk>``, then one id per event in vocabulary order, then the sorted ``words``.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from punct_restore.data.labeling import load_label_sets
from punct_restore.models.hidden_event_lm_viterbi import Back, History, LogProb, viterbi
from punct_restore.models.ngram_lm import (
    NgramLM,
    count_ngrams,
    counts_from_payload,
    counts_to_payload,
)
from punct_restore.models.truecaser import UnigramTruecaser
from punct_restore.prediction_io import write_new_file

__all__ = [
    "BOS_TOKEN",
    "EOS_TOKEN",
    "FORMAT_VERSION",
    "MODEL_FAMILY",
    "UNK_TOKEN",
    "Back",
    "HiddenEventLM",
    "History",
    "LogProb",
    "UnigramTruecaser",
]

FORMAT_VERSION = 1
MODEL_FAMILY = "hidden_event_lm"
BOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<s>", "</s>", "<unk>"
_BOS_ID, _EOS_ID, _UNK_ID, _FIRST_EVENT_ID = 0, 1, 2, 3
_MODES = ("A", "B", "C")
_NO_EVENT = "O"
_CONFIG_KEYS = ("label_mode", "order", "discount", "min_count", "model_name")
_SUMMARY_KEYS = ("num_sequences", "num_tokens", "vocabulary_size")


@dataclass(frozen=True)
class _ModeVocab:
    labels: tuple[str, ...]
    punctuation: tuple[str, ...]
    casing: tuple[str, ...]
    separator: str


@cache
def _mode_vocab(mode: str) -> _ModeVocab:
    """Output labels, punctuation set and casing set of a mode, read from configs/label_sets.yaml."""
    sets = load_label_sets()
    punctuation = tuple(sets["B" if mode == "B" else "A"])
    return _ModeVocab(tuple(sets[mode]), punctuation, tuple(sets["casing"]), sets["separator"])


# ------------------------------------------------------------------ input validation


def _check_config(label_mode: Any, order: Any, discount: Any, min_count: Any, model_name: Any) -> None:
    if not isinstance(label_mode, str) or label_mode not in _MODES:
        raise ValueError("label_mode must be A, B or C")
    if type(order) is not int or order < 2:
        raise ValueError("order must be an integer of at least 2")
    if type(discount) not in (int, float) or not 0 < discount < 1:
        raise ValueError("discount must be a number with 0 < discount < 1")
    if type(min_count) is not int or min_count < 1:
        raise ValueError("min_count must be an integer of at least 1")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a nonempty string")


def _claim_sample_id(record: Any, seen: set[str]) -> str:
    if not isinstance(record, Mapping):
        raise ValueError("each record must be a mapping")
    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("sample_id must be a nonempty string")
    if sample_id in seen:
        raise ValueError(f"duplicate sample_id {sample_id!r}")
    seen.add(sample_id)
    return sample_id


def _word_ids(words: Sequence[str], num_events: int) -> tuple[dict[str, int], int]:
    """Ids of the sorted words (they follow the specials and the events) and the total number of ids."""
    first = _FIRST_EVENT_ID + num_events
    return {word: first + index for index, word in enumerate(words)}, first + len(words)


def _check_tokens(value: Any, where: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{where}: tokens must be a nonempty list")
    if any(not isinstance(token, str) or not token.strip() for token in value):
        raise ValueError(f"{where}: tokens must be nonempty, non-whitespace strings")
    return list(value)


def _check_labels(value: Any, length: int, allowed: Sequence[str], field: str, where: str) -> list[str]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{where}: {field} must be a list with one label per token")
    unknown = next((label for label in value if label not in allowed), None)
    if unknown is not None:
        raise ValueError(f"{where}: unknown {field} label {unknown!r}")
    return value


# ------------------------------------------------------------------ the model


@dataclass(frozen=True)
class _Fitted:
    words: tuple[str, ...]
    word_ids: dict[str, int]
    lm: NgramLM
    truecaser: UnigramTruecaser | None
    summary: dict[str, int]


class HiddenEventLM:
    """Hidden-event LM for mode A, B or C; ``fit`` / ``predict`` follow the shared model contract."""

    def __init__(
        self,
        label_mode: str,
        *,
        order: int = 3,
        discount: float = 0.75,
        min_count: int = 1,
        model_name: str = "hidden_event_lm",
    ) -> None:
        _check_config(label_mode, order, discount, min_count, model_name)
        self._mode, self._order, self._discount = label_mode, order, float(discount)
        self._min_count, self._model_name = min_count, model_name
        self._vocab = _mode_vocab(label_mode)
        self._events = tuple(label for label in self._vocab.punctuation if label != _NO_EVENT)
        self._event_ids = tuple(range(_FIRST_EVENT_ID, _FIRST_EVENT_ID + len(self._events)))
        self._special = {BOS_TOKEN: _BOS_ID, EOS_TOKEN: _EOS_ID, UNK_TOKEN: _UNK_ID}
        self._special.update({f"<{event}>": index for event, index in zip(self._events, self._event_ids, strict=True)})
        self._fitted: _Fitted | None = None

    @property
    def label_mode(self) -> str:
        return self._mode

    @property
    def order(self) -> int:
        return self._order

    @property
    def discount(self) -> float:
        return self._discount

    @property
    def min_count(self) -> int:
        return self._min_count

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def vocabulary(self) -> tuple[str, ...]:
        """The ordered output labels of the mode (what ``predict`` may emit)."""
        return self._vocab.labels

    @property
    def predicted_tokens(self) -> tuple[str, ...]:
        """Every token the LM can predict (the set V): ``</s>``, ``<unk>``, the events, then the words."""
        fitted = self._require_fitted()
        return (EOS_TOKEN, UNK_TOKEN, *(f"<{event}>" for event in self._events), *fitted.words)

    @property
    def training_summary(self) -> dict[str, int]:
        return dict(self._require_fitted().summary)

    def _require_fitted(self) -> _Fitted:
        if self._fitted is None:
            raise RuntimeError("HiddenEventLM is not fitted; call fit or load_model first")
        return self._fitted

    # -------------------------------------------------------------- training

    def fit(self, records: Iterable[Mapping[str, Any]]) -> HiddenEventLM:
        """Fit train records (sample_id, tokens, punct_labels, case_labels for C); a failed refit changes nothing."""
        rows = list(records)
        if not rows:
            raise ValueError("training records must contain at least one record")
        word_counts = self._validated_word_counts(rows)
        words = tuple(sorted(word for word, count in word_counts.items() if count >= self._min_count))
        word_ids, num_ids = _word_ids(words, len(self._events))
        counts = count_ngrams(self._streams(rows, word_ids), self._order, num_ids)
        lm = NgramLM(counts, discount=self._discount, num_ids=num_ids)
        truecaser = None
        if self._mode == "C":
            truecaser = UnigramTruecaser.fit(
                ((row["tokens"], row["case_labels"]) for row in rows), self._vocab.casing
            )
        summary = {"num_sequences": len(rows), "num_tokens": sum(word_counts.values()), "vocabulary_size": len(words)}
        self._fitted = _Fitted(words, word_ids, lm, truecaser, summary)
        return self

    def _validated_word_counts(self, rows: Sequence[Any]) -> Counter[str]:
        seen: set[str] = set()
        counts: Counter[str] = Counter()
        for record in rows:
            counts.update(token.lower() for token in self._check_training_record(record, seen))
        return counts

    def _check_training_record(self, record: Any, seen: set[str]) -> list[str]:
        sample_id = _claim_sample_id(record, seen)
        where = repr(sample_id)
        if record.get("split", "train") != "train":
            raise ValueError(f"{where}: fit accepts only split=train records, got {record.get('split')!r}")
        if record.get("label_mode") != self._mode:
            raise ValueError(f"{where}: label_mode {record.get('label_mode')!r} does not match mode {self._mode}")
        tokens = _check_tokens(record.get("tokens"), where)
        _check_labels(record.get("punct_labels"), len(tokens), self._vocab.punctuation, "punct_labels", where)
        if self._mode == "C":
            _check_labels(record.get("case_labels"), len(tokens), self._vocab.casing, "case_labels", where)
        return tokens

    def _streams(self, rows: Sequence[Mapping[str, Any]], word_ids: Mapping[str, int]) -> Iterator[list[int]]:
        """``<s>`` padding, then each word and its event (if any), then ``</s>``, as token ids."""
        padding = [_BOS_ID] * (self._order - 1)
        event_ids = dict(zip(self._events, self._event_ids, strict=True))
        for record in rows:
            stream = list(padding)
            for token, label in zip(record["tokens"], record["punct_labels"], strict=True):
                stream.append(word_ids.get(token.lower(), _UNK_ID))
                if label != _NO_EVENT:
                    stream.append(event_ids[label])
            stream.append(_EOS_ID)
            yield stream

    # -------------------------------------------------------------- probabilities and prediction

    def _token_id(self, fitted: _Fitted, token: Any) -> int:
        if not isinstance(token, str):
            raise ValueError("tokens must be strings")
        special = self._special.get(token)
        return special if special is not None else fitted.word_ids.get(token.lower(), _UNK_ID)

    def logprob(self, token: str, history: Sequence[str]) -> float:
        """Natural-log P(token | last order-1 tokens of history); OOV words count as ``<unk>``."""
        fitted = self._require_fitted()
        token_id = self._token_id(fitted, token)
        if token_id == _BOS_ID:
            raise ValueError("the start symbol <s> is never predicted")
        ids = [self._token_id(fitted, item) for item in history][-(self._order - 1):]
        return math.log(fitted.lm.prob(token_id, fitted.lm.contexts(ids)))

    def decode(self, tokens: Sequence[str]) -> list[str]:
        """Labels for one utterance: the Viterbi punctuation, joined with the truecased casing in mode C."""
        fitted = self._require_fitted()
        words = _check_tokens(tokens, "tokens")
        ids = [fitted.word_ids.get(word.lower(), _UNK_ID) for word in words]
        choices = viterbi(fitted.lm, ids, self._event_ids, bos_id=_BOS_ID, eos_id=_EOS_ID)
        punctuation = [_NO_EVENT if choice == 0 else self._events[choice - 1] for choice in choices]
        if fitted.truecaser is None:
            return punctuation
        casing = fitted.truecaser.predict(words, punctuation)
        return [f"{punct}{self._vocab.separator}{case}" for punct, case in zip(punctuation, casing, strict=True)]

    def predict(self, records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Contract records in input order; only ``sample_id`` and ``tokens`` of each record are read."""
        self._require_fitted()
        seen: set[str] = set()
        predictions = []
        for record in records:
            sample_id = _claim_sample_id(record, seen)
            predictions.append({
                "sample_id": sample_id,
                "model_name": self._model_name,
                "label_mode": self._mode,
                "predicted_labels": self.decode(_check_tokens(record.get("tokens"), repr(sample_id))),
            })
        return predictions

    # -------------------------------------------------------------- persistence

    def save_model(self, path: str | Path) -> None:
        """Write the versioned JSON artifact; never overwrites, and removes its own partial file if the write fails."""
        fitted = self._require_fitted()
        payload = {
            "format_version": FORMAT_VERSION,
            "model_family": MODEL_FAMILY,
            "config": {
                "label_mode": self._mode, "order": self._order, "discount": self._discount,
                "min_count": self._min_count, "model_name": self._model_name,
            },
            "training": fitted.summary,
            "words": list(fitted.words),
            "ngrams": counts_to_payload(fitted.lm.counts),
            "truecaser": fitted.truecaser.to_rows() if fitted.truecaser is not None else None,
        }
        text = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_new_file(destination, text + "\n")  # FileExistsError leaves the old file alone

    @classmethod
    def load_model(cls, path: str | Path) -> HiddenEventLM:
        """Read an artifact written by ``save_model``; ValueError (one line) if it is malformed."""
        try:
            return cls._from_payload(json.loads(Path(path).read_text(encoding="utf-8")))
        except (ValueError, RecursionError, OverflowError) as exc:  # OverflowError: a backstop for huge numbers
            name = Path(path).name
            raise ValueError(f"{name}: invalid hidden-event LM artifact: {' '.join(str(exc).splitlines())}") from exc

    @classmethod
    def _from_payload(cls, payload: object) -> HiddenEventLM:
        if not isinstance(payload, dict):
            raise ValueError("the file does not hold a JSON object")
        if type(payload.get("format_version")) is not int or payload["format_version"] != FORMAT_VERSION:
            raise ValueError("unsupported format_version")
        if payload.get("model_family") != MODEL_FAMILY:
            raise ValueError("unsupported model_family")
        config = payload.get("config")
        if not isinstance(config, dict) or set(config) != set(_CONFIG_KEYS):
            raise ValueError(f"config must hold exactly {', '.join(_CONFIG_KEYS)}")
        model = cls(**config)
        words = _artifact_words(payload.get("words"))
        word_ids, num_ids = _word_ids(words, len(model._events))
        counts = counts_from_payload(payload.get("ngrams"), model._order, num_ids)
        model._fitted = _Fitted(
            words=words,
            word_ids=word_ids,
            lm=NgramLM(counts, discount=model._discount, num_ids=num_ids),
            truecaser=model._artifact_truecaser(payload.get("truecaser")),
            summary=_artifact_summary(payload.get("training"), len(words)),
        )
        return model

    def _artifact_truecaser(self, rows: object) -> UnigramTruecaser | None:
        if self._mode != "C":
            if rows is not None:
                raise ValueError("a truecaser belongs to mode C only")
            return None
        return UnigramTruecaser.from_rows(rows, self._vocab.casing)


def _artifact_words(value: object) -> tuple[str, ...]:
    if (not isinstance(value, list) or any(not isinstance(word, str) or not word for word in value)
            or len(set(value)) != len(value)):
        raise ValueError("words must be a list of unique nonempty strings")
    return tuple(value)


def _artifact_summary(value: object, num_words: int) -> dict[str, int]:
    if (
        not isinstance(value, dict) or set(value) != set(_SUMMARY_KEYS)
        or any(type(count) is not int or count < 0 for count in value.values())
        or value["vocabulary_size"] != num_words
    ):
        raise ValueError("training must hold nonnegative num_sequences, num_tokens and vocabulary_size")
    return dict(value)
