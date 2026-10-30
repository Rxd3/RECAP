"""Threshold decoding over CRF marginals (team extension; tau is tuned on validation only).

Plain Viterbi decoding picks the most probable label sequence. Here every token is decoded
on its own from its posterior marginals: a token gets punctuation only when the total
probability of "some punctuation" (1 - P(O)) is strictly greater than ``tau``. A small tau
favours recall, a large tau favours precision.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

NO_PUNCT = "O"


def is_o_label(label: str) -> bool:
    """True for the no-punctuation label of every mode: ``O`` or a joint ``O|<CASE>``."""
    return label == NO_PUNCT or label.startswith(f"{NO_PUNCT}|")


def check_tau(tau: object) -> None:
    """Raise ValueError unless ``tau`` is a finite number (not a bool) in [0, 1]."""
    is_number = isinstance(tau, (int, float)) and not isinstance(tau, bool)
    if not (is_number and math.isfinite(tau) and 0 <= tau <= 1):
        raise ValueError(f"tau must be a number between 0 and 1, got {tau!r}")


def _decode_token(probs: Mapping[str, float], index: int, tau: float) -> str:
    if not probs:
        raise ValueError(f"token {index} has no label probabilities")
    o_labels = [label for label in probs if is_o_label(label)]
    if not o_labels:
        raise ValueError(f"token {index} has no O label to compare against")
    punct_labels = [label for label in probs if not is_o_label(label)]
    p_o = sum(probs[label] for label in o_labels)
    if punct_labels and 1 - p_o > tau:
        return max(punct_labels, key=probs.__getitem__)
    return max(o_labels, key=probs.__getitem__)


def decode_with_threshold(marginals: Sequence[Mapping[str, float]], tau: float) -> list[str]:
    """One label per token: the best punctuation label if ``1 - P(O) > tau``, else the best O label.

    ``P(O)`` sums every O label (``O`` or ``O|*``), so mode C casing never splits the O mass.
    Ties go to the first label in the mapping's order. Returns a new list.

    Raises:
        ValueError: ``tau`` is not a finite number in [0, 1], or a token has no probabilities
            or no O label.
    """
    check_tau(tau)
    return [_decode_token(probs, index, tau) for index, probs in enumerate(marginals)]
