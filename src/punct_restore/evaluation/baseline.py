"""Shortcut reference baseline, owned by Member 5. A reference row, not a model.

It shows how much a score owes to two dataset shortcuts: every utterance ends in PERIOD or
QUESTION, and the first word is capitalized. It reads tokens only, never gold labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .inputs import NO_PUNCT, GoldRecord
from .metrics import join_joint

MODEL_NAME = "shortcut_baseline"
WH = frozenset({"what", "why", "how", "where", "when", "who", "whom", "whose", "which"})
AUX = frozenset({"do", "does", "did", "is", "are", "am", "was", "were", "can", "could", "will",
                 "would", "shall", "should", "may", "might", "must", "have", "has", "had"})
QUESTION_STARTERS = WH | AUX


def shortcut_labels(tokens: Sequence[str], mode: str) -> list[str]:
    """O, then QUESTION (wh-word or auxiliary first) or PERIOD; mode C adds first-word casing."""
    if not tokens:
        raise ValueError("tokens must be nonempty")
    final = "QUESTION" if tokens[0] in QUESTION_STARTERS else "PERIOD"
    punct = [NO_PUNCT] * (len(tokens) - 1) + [final]
    if mode in ("A", "B"):
        return punct
    if mode == "C":
        return [join_joint(p, "FIRST_UPPER" if i == 0 else "LOWER") for i, p in enumerate(punct)]
    raise ValueError(f"unknown label mode {mode!r}")


def shortcut_predictions(records: Sequence[GoldRecord], mode: str) -> list[dict[str, Any]]:
    """Prediction records in the shared prediction format, in input order."""
    return [{"sample_id": record.sample_id, "model_name": MODEL_NAME, "label_mode": mode,
             "predicted_labels": shortcut_labels(record.tokens, mode)} for record in records]
