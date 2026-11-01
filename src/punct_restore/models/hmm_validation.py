"""Input and saved-count validation for the HMM baseline (no model state lives here)."""

from __future__ import annotations

from typing import Any


def positive_integer(value: Any) -> bool:
    """Return True for a real int (not bool) above zero."""
    return type(value) is int and value > 0


def validated_tokens(value: Any) -> list[str]:
    """Return a copy of a nonempty token list/tuple, rejecting blank tokens."""
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("tokens must be a nonempty list or tuple")
    if any(not isinstance(token, str) or not token.strip() for token in value):
        raise ValueError("each token must be a nonempty, non-whitespace string")
    return list(value)


def _is_vector(value: Any, k: int) -> bool:
    """Return True for a list of exactly k nonnegative integers."""
    return (
        isinstance(value, list)
        and len(value) == k
        and all(type(item) is int and item >= 0 for item in value)
    )


def _check_shapes(counts: Any, k: int) -> None:
    """Check count vectors, the transition matrix and the sequence totals."""
    if not all(_is_vector(value, k) for value in [counts.start, counts.end, counts.state, counts.unknown]):
        raise ValueError("count vectors must contain K nonnegative integers")
    if (
        not isinstance(counts.transition, list)
        or len(counts.transition) != k
        or not all(_is_vector(row, k) for row in counts.transition)
    ):
        raise ValueError("transition counts must be a K-by-K integer matrix")
    if not positive_integer(counts.num_sequences):
        raise ValueError("num_sequences must be a positive integer")
    if sum(counts.start) != counts.num_sequences or sum(counts.end) != counts.num_sequences:
        raise ValueError("start/end counts disagree with num_sequences")
    if sum(counts.state) < counts.num_sequences:
        raise ValueError("too few tokens for the sequence count")


def _check_vocabulary(counts: Any) -> set[str]:
    """Validate the vocabulary list and return it as a set."""
    if not isinstance(counts.vocabulary, list):
        raise ValueError("vocabulary must be a list")
    if counts.vocabulary:
        validated_tokens(counts.vocabulary)
    if len(set(counts.vocabulary)) != len(counts.vocabulary):
        raise ValueError("duplicate vocabulary entry")
    return set(counts.vocabulary)


def _check_state(counts: Any, state: int, vocabulary: set[str]) -> None:
    """Check one state's emission table and its in/out flow totals."""
    emission = counts.emissions[state]
    if not isinstance(emission, dict) or not set(emission) <= vocabulary:
        raise ValueError("emission keys must belong to the vocabulary")
    if not all(positive_integer(count) for count in emission.values()):
        raise ValueError("sparse emission counts must be positive integers")
    if sum(emission.values()) + counts.unknown[state] != counts.state[state]:
        raise ValueError("emission totals disagree with state counts")
    if sum(counts.transition[state]) + counts.end[state] != counts.state[state]:
        raise ValueError("outgoing transitions/end counts disagree with state counts")
    incoming = sum(row[state] for row in counts.transition) + counts.start[state]
    if incoming != counts.state[state]:
        raise ValueError("incoming transitions/start counts disagree with state counts")


def validate_counts(counts: Any, k: int, min_token_count: int) -> None:
    """Raise ValueError unless saved counts are consistent for k states."""
    _check_shapes(counts, k)
    vocabulary = _check_vocabulary(counts)
    if not isinstance(counts.emissions, list) or len(counts.emissions) != k:
        raise ValueError("one sparse emission table is required per state")
    for state in range(k):
        _check_state(counts, state, vocabulary)
    for word in vocabulary:
        if sum(table.get(word, 0) for table in counts.emissions) < min_token_count:
            raise ValueError("retained vocabulary violates min_token_count")
