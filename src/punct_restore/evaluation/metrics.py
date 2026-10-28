"""Token-level classification metrics, owned by Member 5 (docs/EVALUATION_PROTOCOL.md).

Every score derives from one confusion matrix (rows gold, columns predicted). Any zero
denominator gives 0.0 (zero_division=0), and classes without support stay in macro averages
so their absence stays visible.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

SEPARATOR = "|"


def _div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _div(2 * precision * recall, precision + recall)


def confusion(pairs: Iterable[tuple[str, str]],
              labels: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    """Counts of (gold, predicted) pairs over the ordered vocabulary; unknown labels raise."""
    index = {label: i for i, label in enumerate(labels)}
    counts = [[0] * len(labels) for _ in labels]
    for gold, predicted in pairs:
        unknown = next((label for label in (gold, predicted) if label not in index), None)
        if unknown is not None:
            raise ValueError(f"label {unknown!r} is not one of {list(labels)}")
        counts[index[gold]][index[predicted]] += 1
    return tuple(tuple(row) for row in counts)


def _class_scores(tp: int, support: int, predicted: int) -> dict[str, float | int]:
    precision, recall = _div(tp, predicted), _div(tp, support)
    return {"precision": precision, "recall": recall, "f1": _f1(precision, recall),
            "support": support, "predicted": predicted}


def score_block(pairs: Iterable[tuple[str, str]], labels: Sequence[str],
                macro_over: Sequence[str]) -> dict[str, Any]:
    """Per-class, macro and micro (both over `macro_over`) and accuracy for one block."""
    labels, macro_over = tuple(labels), tuple(macro_over)
    if not macro_over or any(label not in labels for label in macro_over):
        raise ValueError(f"macro_over must be a nonempty subset of the labels: {list(macro_over)}")
    matrix = confusion(pairs, labels)
    support = [sum(row) for row in matrix]
    predicted = [sum(column) for column in zip(*matrix)]
    tp_by_class = [matrix[i][i] for i in range(len(labels))]
    per_class = {label: _class_scores(tp_by_class[i], support[i], predicted[i])
                 for i, label in enumerate(labels)}
    macro_indices = [labels.index(label) for label in macro_over]
    micro_tp = sum(tp_by_class[i] for i in macro_indices)
    micro_p = _div(micro_tp, sum(predicted[i] for i in macro_indices))
    micro_r = _div(micro_tp, sum(support[i] for i in macro_indices))
    macro_f1 = _div(sum(per_class[label]["f1"] for label in macro_over), len(macro_over))
    return {"labels": list(labels), "macro_over": list(macro_over),
            "scored_positions": sum(support), "per_class": per_class, "macro_f1": macro_f1,
            "micro": {"precision": micro_p, "recall": micro_r, "f1": _f1(micro_p, micro_r)},
            "accuracy": _div(sum(tp_by_class), sum(support)),
            "confusion": [list(row) for row in matrix],
            "empty_support": not any(support[i] or predicted[i] for i in macro_indices)}


def join_joint(punct: str, case: str) -> str:
    """'PERIOD|FIRST_UPPER' from ('PERIOD', 'FIRST_UPPER'); the inverse of split_joint()."""
    return f"{punct}{SEPARATOR}{case}"


def split_joint(label: str) -> tuple[str, str]:
    """('PERIOD', 'FIRST_UPPER') from 'PERIOD|FIRST_UPPER'; no separator raises ValueError."""
    punct, separator, case = label.partition(SEPARATOR)
    if not separator:
        raise ValueError(f"{label!r} is not a joint PUNCT{SEPARATOR}CASE label")
    return punct, case
