"""Hand-computed and scikit-learn-checked tests for the evaluation metrics (spec section 4)."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from types import ModuleType
from typing import Any

import pytest

from punct_restore.evaluation.inputs import Vocab
from punct_restore.evaluation.metrics import (
    class_counts,
    confusion,
    macro_f1_from_counts,
    score_block,
    split_joint,
)

LABELS = ("O", "COMMA", "PERIOD", "QUESTION")
NON_O = LABELS[1:]
Pairs = list[tuple[str, str]]

# 13 (gold, predicted) pairs; the hand-computed numbers in the tests below come from this table.
PAIRS: Pairs = (
    [("O", "O")] * 5
    + [("O", "COMMA")]
    + [("COMMA", "COMMA")] * 2
    + [("COMMA", "O")]
    + [("COMMA", "PERIOD")]
    + [("PERIOD", "PERIOD")] * 2
    + [("QUESTION", "PERIOD")]
)


def _scored(pairs: Iterable[tuple[str, str]]) -> dict[str, Any]:
    return score_block(pairs, LABELS, NON_O)


def _floats(node: Any) -> Iterator[float]:
    """Every float found anywhere inside nested dicts, lists and tuples."""
    if isinstance(node, float):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _floats(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _floats(value)


def test_hand_computed_block() -> None:
    block = _scored(PAIRS)
    per_class = block["per_class"]
    expected = {
        "COMMA": (2 / 3, 1 / 2, 4 / 7, 4, 3),
        "PERIOD": (1 / 2, 1.0, 2 / 3, 2, 4),
        "QUESTION": (0.0, 0.0, 0.0, 1, 0),
        "O": (5 / 6, 5 / 6, 5 / 6, 6, 6),
    }
    for label, (precision, recall, f1, support, predicted) in expected.items():
        assert per_class[label]["precision"] == pytest.approx(precision)
        assert per_class[label]["recall"] == pytest.approx(recall)
        assert per_class[label]["f1"] == pytest.approx(f1)
        assert per_class[label]["support"] == support
        assert per_class[label]["predicted"] == predicted
    assert block["macro_f1"] == pytest.approx((4 / 7 + 2 / 3 + 0) / 3)
    for name in ("precision", "recall", "f1"):
        assert block["micro"][name] == pytest.approx(4 / 7)
    assert block["accuracy"] == pytest.approx(9 / 13)
    assert block["scored_positions"] == 13
    assert block["confusion"] == [[5, 1, 0, 0], [1, 2, 1, 0], [0, 0, 2, 0], [0, 0, 1, 0]]
    assert block["labels"] == list(LABELS)
    assert block["macro_over"] == list(NON_O)
    assert block["empty_support"] is False


def test_hand_computed_block_shape() -> None:
    block = _scored(PAIRS)
    assert set(block) == {"labels", "macro_over", "scored_positions", "per_class", "macro_f1",
                          "micro", "accuracy", "confusion", "empty_support"}
    assert list(block["per_class"]) == list(LABELS)
    for entry in block["per_class"].values():
        assert set(entry) == {"precision", "recall", "f1", "support", "predicted"}
        assert type(entry["support"]) is int
        assert type(entry["predicted"]) is int
    assert set(block["micro"]) == {"precision", "recall", "f1"}


def test_block_independent_of_pair_order() -> None:
    assert _scored(list(reversed(PAIRS))) == _scored(PAIRS)


def test_o_to_comma_is_false_positive() -> None:
    block = _scored([("O", "COMMA"), ("COMMA", "COMMA")])
    assert block["per_class"]["COMMA"]["precision"] == pytest.approx(1 / 2)
    assert block["per_class"]["COMMA"]["recall"] == pytest.approx(1.0)
    assert block["micro"]["precision"] == pytest.approx(1 / 2)
    assert block["micro"]["recall"] == pytest.approx(1.0)


def test_lone_false_positive_scores_zero_but_is_not_empty() -> None:
    block = _scored([("O", "COMMA")])
    assert block["per_class"]["COMMA"]["predicted"] == 1
    assert block["per_class"]["COMMA"]["support"] == 0
    assert block["per_class"]["COMMA"]["f1"] == 0.0
    assert block["micro"]["precision"] == 0.0
    assert block["micro"]["recall"] == 0.0
    assert block["empty_support"] is False


def test_comma_to_o_is_false_negative() -> None:
    block = _scored([("COMMA", "O"), ("COMMA", "COMMA")])
    assert block["per_class"]["COMMA"]["precision"] == pytest.approx(1.0)
    assert block["per_class"]["COMMA"]["recall"] == pytest.approx(1 / 2)
    assert block["micro"]["precision"] == pytest.approx(1.0)
    assert block["micro"]["recall"] == pytest.approx(1 / 2)


def test_lone_false_negative_is_not_empty() -> None:
    block = _scored([("COMMA", "O")])
    assert block["per_class"]["COMMA"]["support"] == 1
    assert block["per_class"]["COMMA"]["predicted"] == 0
    assert block["empty_support"] is False


def test_comma_period_swap_counts_fp_and_fn() -> None:
    block = _scored([("COMMA", "PERIOD"), ("PERIOD", "PERIOD")])
    assert block["per_class"]["COMMA"]["recall"] == 0.0
    assert block["per_class"]["COMMA"]["support"] == 1
    assert block["per_class"]["PERIOD"]["precision"] == pytest.approx(1 / 2)
    assert block["per_class"]["PERIOD"]["recall"] == pytest.approx(1.0)
    assert block["micro"]["precision"] == pytest.approx(1 / 2)
    assert block["micro"]["recall"] == pytest.approx(1 / 2)


def test_zero_support_class_in_macro() -> None:
    block = _scored([("COMMA", "COMMA"), ("PERIOD", "PERIOD"), ("O", "O")])
    assert block["per_class"]["QUESTION"]["support"] == 0
    assert block["per_class"]["QUESTION"]["predicted"] == 0
    assert block["per_class"]["QUESTION"]["f1"] == 0.0
    assert block["macro_f1"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(("pairs", "scored_positions", "accuracy"),
                         [([], 0, 0.0), ([("O", "O"), ("O", "O")], 2, 1.0)],
                         ids=["no_pairs", "only_o"])
def test_empty_block_flagged(pairs: Pairs, scored_positions: int, accuracy: float) -> None:
    block = _scored(pairs)
    assert block["empty_support"] is True
    assert block["macro_f1"] == 0.0
    assert set(block["micro"].values()) == {0.0}
    for label in NON_O:
        entry = block["per_class"][label]
        assert (entry["precision"], entry["recall"], entry["f1"]) == (0.0, 0.0, 0.0)
    assert block["scored_positions"] == scored_positions
    assert block["accuracy"] == accuracy
    json.dumps(block, allow_nan=False)
    floats = list(_floats(block))
    assert floats
    assert not any(math.isnan(value) for value in floats)


@pytest.mark.parametrize(("label", "expected"), [
    ("PERIOD|FIRST_UPPER", ("PERIOD", "FIRST_UPPER")),
    ("O|LOWER", ("O", "LOWER")),
    ("A|B|C", ("A", "B|C")),
])
def test_split_joint(label: str, expected: tuple[str, str]) -> None:
    assert split_joint(label) == expected


@pytest.mark.parametrize("label", ["PERIOD", ""])
def test_split_joint_without_separator_raises(label: str) -> None:
    with pytest.raises(ValueError):
        split_joint(label)


def test_unknown_label_raises() -> None:
    with pytest.raises(ValueError, match="EXCLAMATION"):
        confusion([("O", "EXCLAMATION")], LABELS)
    with pytest.raises(ValueError, match="'X'"):
        confusion([("X", "O")], LABELS)
    with pytest.raises(ValueError, match="DASH"):
        score_block([("O", "DASH")], LABELS, NON_O)


@pytest.mark.parametrize("macro_over", [["EXCLAMATION"], []], ids=["not_a_subset", "empty"])
def test_macro_over_must_be_nonempty_subset(macro_over: list[str]) -> None:
    with pytest.raises(ValueError, match="macro_over"):
        score_block([], LABELS, macro_over)


def test_confusion_rows_are_gold() -> None:
    assert confusion([("O", "COMMA")], LABELS) == ((0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))


def test_score_block_accepts_generator() -> None:
    assert _scored(pair for pair in PAIRS) == _scored(PAIRS)


# --- scikit-learn cross-checks ------------------------------------------------------------------------------------

@pytest.fixture
def sk_metrics() -> ModuleType:
    return pytest.importorskip("sklearn.metrics")


def _assert_matches_sklearn(sk: ModuleType, pairs: Pairs, labels: Sequence[str], macro_over: Sequence[str]) -> None:
    """score_block against sklearn with zero_division=0: every class, micro and macro over macro_over, accuracy."""
    block = score_block(pairs, labels, macro_over)
    gold, pred = [g for g, _ in pairs], [p for _, p in pairs]
    precision, recall, f1, support = sk.precision_recall_fscore_support(
        gold, pred, labels=list(labels), zero_division=0)
    predicted = Counter(pred)
    for i, label in enumerate(labels):
        entry = block["per_class"][label]
        assert entry["precision"] == pytest.approx(precision[i])
        assert entry["recall"] == pytest.approx(recall[i])
        assert entry["f1"] == pytest.approx(f1[i])
        assert entry["support"] == support[i]
        assert entry["predicted"] == predicted[label]
    micro = sk.precision_recall_fscore_support(gold, pred, labels=list(macro_over), average="micro", zero_division=0)
    for name, value in zip(("precision", "recall", "f1"), micro[:3], strict=False):
        assert block["micro"][name] == pytest.approx(value)
    macro = sk.precision_recall_fscore_support(gold, pred, labels=list(macro_over), average="macro", zero_division=0)
    assert block["macro_f1"] == pytest.approx(macro[2])
    assert block["accuracy"] == pytest.approx(sk.accuracy_score(gold, pred))


def test_matches_sklearn(sk_metrics: ModuleType) -> None:
    _assert_matches_sklearn(sk_metrics, PAIRS, LABELS, NON_O)


def _random_punctuation_pairs(rng: random.Random) -> Pairs:
    weights = [6, 2, 2, 1]
    return [
        (g, p)
        for g, p in zip(rng.choices(LABELS, weights, k=300), rng.choices(LABELS, weights, k=300), strict=False)
    ]


def _random_joint_pairs(rng: random.Random, labels: Sequence[str]) -> Pairs:
    """Skewed gold; predictions only from the first 10 labels, so the last 2 are never predicted (zero division)."""
    gold = rng.choices(labels, weights=range(len(labels), 0, -1), k=300)
    pool = labels[:10]
    return [(g, g if g in pool and rng.random() < 0.5 else rng.choice(pool)) for g in gold]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_sklearn_random(sk_metrics: ModuleType, vocab: Vocab, seed: int) -> None:
    rng = random.Random(seed)
    _assert_matches_sklearn(sk_metrics, _random_punctuation_pairs(rng), LABELS, NON_O)
    _assert_matches_sklearn(sk_metrics, _random_joint_pairs(rng, vocab.C), vocab.C, vocab.C)


# --- count helpers used by the grouped bootstrap -----------------------------------------------

def test_class_counts() -> None:
    assert class_counts(((5, 1), (2, 3))) == ((5, 3), (6, 5), (7, 4))


def test_macro_f1_from_counts() -> None:
    assert macro_f1_from_counts((0, 1), (1, 1), (0, 1)) == 0.5
    assert macro_f1_from_counts((1, 0), (1, 0), (1, 0)) == 0.5  # zero-support class stays in the mean
    assert macro_f1_from_counts((), (), ()) == 0.0
