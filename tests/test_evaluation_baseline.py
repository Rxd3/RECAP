"""Shortcut baseline of punct_restore.evaluation (spec section 6): word lists, labels, records, public API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from eval_fixtures import SCORED, all_masks, gold_rows, write_ids, write_jsonl

from punct_restore import evaluation
from punct_restore.evaluation import EvaluationInputError, load_bundle, load_scored_gold
from punct_restore.evaluation.baseline import (
    AUX,
    MODEL_NAME,
    WH,
    shortcut_labels,
    shortcut_predictions,
)
from punct_restore.evaluation.inputs import GoldRecord, Vocab

QUESTION_STARTERS = ("what", "why", "how", "where", "when", "who", "whom", "whose", "which",
                     "do", "does", "did", "is", "are", "am", "was", "were", "can", "could", "will", "would",
                     "shall", "should", "may", "might", "must", "have", "has", "had")


class TokensOnly:
    """Stands in for a GoldRecord whose gold fields must never be touched."""

    def __init__(self, sample_id: str, tokens: tuple[str, ...]) -> None:
        self.sample_id = sample_id
        self.tokens = tokens

    @property
    def labels(self) -> tuple[str, ...]:
        raise AssertionError("gold read")

    @property
    def group_id(self) -> str | None:
        raise AssertionError("gold read")


def test_word_lists_exact() -> None:
    assert WH == frozenset({"what", "why", "how", "where", "when", "who", "whom", "whose", "which"})
    assert AUX == frozenset({"do", "does", "did", "is", "are", "am", "was", "were", "can", "could", "will",
                             "would", "shall", "should", "may", "might", "must", "have", "has", "had"})
    assert MODEL_NAME == "shortcut_baseline"


@pytest.mark.parametrize("mode", ["A", "B"])
@pytest.mark.parametrize(("tokens", "expected"), [
    (["what", "is", "it"], ["O", "O", "QUESTION"]),
    (["is", "it", "ok"], ["O", "O", "QUESTION"]),
    (["it", "is", "ok"], ["O", "O", "PERIOD"]),
])
def test_shortcut_wh_aux_plain(mode: str, tokens: list[str], expected: list[str]) -> None:
    assert shortcut_labels(tokens, mode) == expected


@pytest.mark.parametrize(("tokens", "expected"), [
    (["it", "is", "ok"], ["O|FIRST_UPPER", "O|LOWER", "PERIOD|LOWER"]),
    (["what", "is", "it"], ["O|FIRST_UPPER", "O|LOWER", "QUESTION|LOWER"]),
])
def test_shortcut_mode_c_joint_labels(tokens: list[str], expected: list[str]) -> None:
    assert shortcut_labels(tokens, "C") == expected


def test_question_starters_list_has_29_distinct_words() -> None:
    assert len(set(QUESTION_STARTERS)) == 29


@pytest.mark.parametrize("word", QUESTION_STARTERS)
def test_every_question_starter_gives_question(word: str) -> None:
    assert shortcut_labels([word, "x"], "A") == ["O", "QUESTION"]


@pytest.mark.parametrize(("tokens", "expected"), [
    (["it", "is", "what"], ["O", "O", "PERIOD"]),
    (["ok", "do"], ["O", "PERIOD"]),
    (["whatever", "x"], ["O", "PERIOD"]),
    (["What", "is", "it"], ["O", "O", "PERIOD"]),
])
def test_only_exact_first_token_decides(tokens: list[str], expected: list[str]) -> None:
    assert shortcut_labels(tokens, "A") == expected


@pytest.mark.parametrize(("mode", "tokens", "expected"), [
    ("A", ["what"], ["QUESTION"]),
    ("A", ["ok"], ["PERIOD"]),
    ("C", ["what"], ["QUESTION|FIRST_UPPER"]),
    ("C", ["ok"], ["PERIOD|FIRST_UPPER"]),
])
def test_single_token(mode: str, tokens: list[str], expected: list[str]) -> None:
    assert shortcut_labels(tokens, mode) == expected


def test_labels_are_in_vocab(vocab: Vocab) -> None:
    token_lists = [["what", "is", "it"], ["it", "is", "ok"], ["ok"], ["do"], ["well", "it", "rains", "today"]]
    for tokens in token_lists:
        assert set(shortcut_labels(tokens, "A")) <= set(vocab.A)
        assert set(shortcut_labels(tokens, "C")) <= set(vocab.C)


@pytest.mark.parametrize(("tokens", "mode"), [([], "A"), (["a"], "D")])
def test_invalid_arguments(tokens: list[str], mode: str) -> None:
    with pytest.raises(ValueError):
        shortcut_labels(tokens, mode)


def test_shortcut_predictions_records() -> None:
    records = (GoldRecord("s2", None, ("it", "is", "ok"), ("O", "O", "PERIOD")),
               GoldRecord("s1", "g", ("what",), ("QUESTION",)))
    assert shortcut_predictions(records, "A") == [
        {"sample_id": "s2", "model_name": "shortcut_baseline", "label_mode": "A",
         "predicted_labels": ["O", "O", "PERIOD"]},
        {"sample_id": "s1", "model_name": "shortcut_baseline", "label_mode": "A",
         "predicted_labels": ["QUESTION"]},
    ]


def test_shortcut_never_reads_gold() -> None:
    tokens = ("what", "is", "it")
    tokens_only: list[Any] = [TokensOnly("s1", tokens)]
    real = [GoldRecord("s1", "g1", tokens, ("O", "O", "QUESTION|LOWER"))]
    impossible = [GoldRecord("s1", "g1", tokens, ("NOT_A_LABEL",) * 3)]
    expected = shortcut_predictions(real, "C")
    assert shortcut_predictions(tokens_only, "C") == expected
    assert shortcut_predictions(impossible, "C") == expected


@pytest.mark.parametrize("mode", ["A", "C"])
def test_baseline_predictions_pass_input_checks(tmp_path: Path, vocab: Vocab, mode: str) -> None:
    gold = write_jsonl(tmp_path / "gold.jsonl", gold_rows(mode))
    ids = write_ids(tmp_path / "list.ids", SCORED)
    loaded_mode, records = load_scored_gold(gold, ids, vocab)
    assert loaded_mode == mode
    predictions = write_jsonl(tmp_path / "preds.jsonl", shortcut_predictions(records, mode))
    masks = write_jsonl(tmp_path / "masks.jsonl", all_masks()) if mode == "C" else None
    bundle = load_bundle(gold, ids, predictions, masks, vocab)
    assert bundle.model_name == "shortcut_baseline"
    assert bundle.ignored_prediction_ids == 0


def test_public_api() -> None:
    names = {"EvaluationInputError", "evaluate", "load_bundle", "load_scored_gold", "load_vocab",
             "shortcut_predictions", "to_markdown"}
    assert set(evaluation.__all__) == names
    for name in names:
        assert hasattr(evaluation, name), name
    assert issubclass(EvaluationInputError, ValueError)
