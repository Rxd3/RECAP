"""Threshold decoding over CRF marginals (team extension; tau and gamma are tuned on validation only).

Plain Viterbi decoding picks the most probable label sequence. Here every token is decoded
on its own from its posterior marginals: a token gets punctuation only when the total
probability of "some punctuation" (1 - P(O)) is strictly greater than ``tau``. A small tau
favours recall, a large tau favours precision.

Once the gate fires, the label is the best non-O one. With ``gamma > 0`` each non-O label is
first divided by ``prior ** gamma`` (its training frequency), so rare classes are not drowned
out by common ones; ``gamma == 0`` is the plain argmax.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

NO_PUNCT = "O"


def is_o_label(label: str) -> bool:
    """True for the no-punctuation label of every mode: ``O`` or a joint ``O|<CASE>``."""
    return label == NO_PUNCT or label.startswith(f"{NO_PUNCT}|")


def _is_finite_number(value: object) -> bool:
    """True for a finite int or float (not a bool); an int too big for a float is not finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def check_tau(tau: object) -> None:
    """Raise ValueError unless ``tau`` is a finite number (not a bool) in [0, 1]."""
    if not (_is_finite_number(tau) and 0 <= tau <= 1):  # _is_finite_number also refuses ints too big for a float
        raise ValueError(f"tau must be a number between 0 and 1, got {tau!r}")


def check_gamma(gamma: object) -> None:
    """Raise ValueError unless ``gamma`` is a finite number (not a bool) that is >= 0."""
    if not (_is_finite_number(gamma) and gamma >= 0):
        raise ValueError(f"gamma must be a finite number >= 0, got {gamma!r}")


def _as_priors(priors: object) -> Mapping[str, float]:
    if isinstance(priors, Mapping):
        return priors
    raise ValueError("priors must be a mapping of label to a finite number > 0")


def _valid_prior(priors: Mapping[str, float], label: str) -> float:
    prior = priors.get(label)
    if not (_is_finite_number(prior) and prior > 0):
        raise ValueError(f"gamma > 0 needs a finite prior > 0 for label {label!r}")
    return prior


def check_priors(priors: object) -> None:
    """Raise ValueError unless ``priors`` maps every label to a finite number > 0."""
    mapping = _as_priors(priors)
    for label in mapping:
        _valid_prior(mapping, label)


def _label_shifts(
    marginals: Sequence[Mapping[str, float]], priors: object, gamma: float
) -> dict[str, float]:
    """``gamma * log(prior)`` of every non-O label that appears in ``marginals``.

    Raises:
        ValueError: ``priors`` is not a mapping, or lacks a finite prior > 0 for an appearing label.
    """
    mapping = _as_priors(priors)
    labels = {label for probs in marginals for label in probs if not is_o_label(label)}
    return {label: gamma * math.log(_valid_prior(mapping, label)) for label in labels}


def _adjusted(prob: float, shift: float) -> float:
    """``log(P / prior ** gamma)``; log space, so a huge gamma cannot overflow or divide by zero."""
    return -math.inf if prob <= 0 else math.log(prob) - shift


def _decode_token(
    probs: Mapping[str, float], index: int, tau: float, shifts: Mapping[str, float] | None
) -> str:
    if not probs:
        raise ValueError(f"token {index} has no label probabilities")
    o_labels = [label for label in probs if is_o_label(label)]
    if not o_labels:
        raise ValueError(f"token {index} has no O label to compare against")
    punct_labels = [label for label in probs if not is_o_label(label)]
    p_o = sum(probs[label] for label in o_labels)
    if punct_labels and 1 - p_o > tau:
        if shifts is None:
            return max(punct_labels, key=probs.__getitem__)
        return max(punct_labels, key=lambda label: _adjusted(probs[label], shifts[label]))
    return max(o_labels, key=probs.__getitem__)


def decode_with_threshold(
    marginals: Sequence[Mapping[str, float]],
    tau: float,
    *,
    priors: Mapping[str, float] | None = None,
    gamma: float = 0.0,
) -> list[str]:
    """One label per token: the best punctuation label if ``1 - P(O) > tau``, else the best O label.

    ``P(O)`` sums every O label (``O`` or ``O|*``), so mode C casing never splits the O mass.
    With ``gamma > 0`` the punctuation label is ``argmax P(c|x) / priors[c] ** gamma`` over the
    token's non-O labels; the gate is unchanged. With ``gamma == 0`` (the default) ``priors`` is
    ignored and the output is the plain argmax. Ties go to the first label in the mapping's
    order. Returns a new list.

    Raises:
        ValueError: ``tau`` is not a finite number in [0, 1], ``gamma`` is not a finite number
            >= 0, ``gamma > 0`` without a priors mapping that has a finite prior > 0 for every
            non-O label that appears, or a token has no probabilities or no O label.
    """
    check_tau(tau)
    check_gamma(gamma)
    shifts = _label_shifts(marginals, priors, gamma) if gamma > 0 else None
    return [_decode_token(probs, index, tau, shifts) for index, probs in enumerate(marginals)]


def _label_list(record: Mapping[str, Any], key: str) -> list[str]:
    labels = record.get(key)
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise ValueError(f"a record has no {key} list of strings")
    return labels


def _gold_labels(record: Mapping[str, Any], joint: bool) -> list[str]:
    punct = _label_list(record, "punct_labels")
    if not joint:
        return punct
    case = _label_list(record, "case_labels")
    if len(case) != len(punct):
        raise ValueError("a record's case_labels and punct_labels differ in length")
    return [f"{p}|{c}" for p, c in zip(punct, case, strict=True)]


def label_priors(records: Iterable[Mapping[str, Any]], labels: Sequence[str]) -> dict[str, float]:
    """Add-one-smoothed gold frequency of every non-O label: ``(count + 1) / (tokens + K)``.

    ``K = len(labels)`` is the mode's label count, so no prior is 0. Counts come from the
    records' ``punct_labels``; when ``labels`` are joint (``PUNCT|CASE``, mode C) they come from
    the ``f"{punct}|{case}"`` pairs instead. ``records`` may be a one-shot iterable.

    Raises:
        ValueError: a record lacks its label lists, they differ in length, or a gold label is
            outside ``labels``.
    """
    joint = any("|" in label for label in labels)
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(_gold_labels(record, joint))
    if not counts.keys() <= set(labels):
        raise ValueError("gold labels outside the mode's label set")
    denominator = sum(counts.values()) + len(labels)
    return {label: (counts[label] + 1) / denominator for label in labels if not is_o_label(label)}
