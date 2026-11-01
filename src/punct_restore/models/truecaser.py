"""Unigram truecaser: the casing half of the factored mode C pipeline of the hidden-event LM."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from typing import Any

from punct_restore.data.labeling import load_label_sets

_TERMINALS = frozenset({"PERIOD", "QUESTION", "EXCLAMATION"})


@cache
def _default_casing() -> tuple[str, ...]:
    """The casing labels in vocabulary order, read from configs/label_sets.yaml."""
    return tuple(load_label_sets()["casing"])


class UnigramTruecaser:
    """Casing from per-word training counts (the casing half of the factored mode C pipeline).

    A word seen in training gets its most frequent casing label (ties go to the casing vocabulary order
    LOWER, FIRST_UPPER, ALL_UPPER). An unseen word is FIRST_UPPER when it starts the utterance or the previous
    token's predicted punctuation ends a sentence (PERIOD, QUESTION, EXCLAMATION), otherwise LOWER.
    """

    def __init__(self, counts: Mapping[str, Sequence[int]], casing: Sequence[str] | None = None) -> None:
        self._casing = tuple(casing) if casing is not None else _default_casing()
        self._counts = {word: tuple(row) for word, row in counts.items()}
        self._best = {
            word: self._casing[max(range(len(row)), key=lambda index: (row[index], -index))]
            for word, row in self._counts.items()
        }

    @classmethod
    def fit(
        cls, sequences: Iterable[tuple[Sequence[str], Sequence[str]]], casing: Sequence[str] | None = None
    ) -> UnigramTruecaser:
        """Count (lowercased token, casing label) over ``(tokens, labels)`` pairs."""
        labels = tuple(casing) if casing is not None else _default_casing()
        counts: dict[str, list[int]] = {}
        for tokens, case_labels in sequences:
            for token, label in zip(tokens, case_labels, strict=True):
                if label not in labels:
                    raise ValueError(f"unknown casing label {label!r}")
                counts.setdefault(token.lower(), [0] * len(labels))[labels.index(label)] += 1
        return cls(counts, labels)

    def predict(self, tokens: Sequence[str], punctuation: Sequence[str]) -> list[str]:
        """One casing label per token, given the punctuation predicted after each token."""
        labels: list[str] = []
        starts_sentence = True
        for token, punct in zip(tokens, punctuation, strict=True):
            best = self._best.get(token.lower())
            labels.append(best if best is not None else ("FIRST_UPPER" if starts_sentence else "LOWER"))
            starts_sentence = punct in _TERMINALS
        return labels

    def to_rows(self) -> list[list[Any]]:
        return [[word, *row] for word, row in sorted(self._counts.items())]

    @classmethod
    def from_rows(cls, rows: object, casing: Sequence[str]) -> UnigramTruecaser:
        """Rebuild from ``to_rows`` output; ValueError for anything else."""
        if not isinstance(rows, list):
            raise ValueError("truecaser must be a list of [word, counts...] rows")
        counts: dict[str, list[int]] = {}
        for row in rows:
            valid = (
                isinstance(row, list) and len(row) == 1 + len(casing) and isinstance(row[0], str)
                and bool(row[0]) and all(type(count) is int and count >= 0 for count in row[1:])
                and sum(row[1:]) > 0
            )
            if not valid or row[0] in counts:
                raise ValueError("truecaser rows must be unique words with nonnegative integer counts")
            counts[row[0]] = row[1:]
        return cls(counts, casing)
