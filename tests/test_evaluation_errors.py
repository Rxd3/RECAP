"""Tests for the error-examples module (design F3): rendering, windows, error-pair counts, seeded examples."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from eval_fixtures import (
    SCORED,
    TOKENS,
    Case,
    gold_row,
    gold_rows,
    joint,
    mask_row,
    perfect_preds,
    pred_row,
    valid_case,
    write_case,
)

from punct_restore.evaluation import error_examples, errors_markdown, load_bundle
from punct_restore.evaluation.error_analysis import DEFAULT_BLOCK, DEFAULT_PER_PAIR, MARKS, SCHEMA, WINDOW, render
from punct_restore.evaluation.inputs import Aligned, Bundle, EvaluationInputError, Vocab

ELLIPSIS = "…"
SIX_TOKENS = ("a", "b", "c", "d", "e", "f")
SIX_LOWER = ["LOWER"] * 6
EXAMPLE_KEYS = {"gold", "pred", "sample_id", "group_id", "position", "gold_text", "pred_text"}
REPORT_KEYS = {"schema", "label_mode", "model_name", "block", "seed", "per_pair", "inputs", "errors_total",
               "pair_counts", "examples"}

# (sample_id, group_id, gold punctuation, predicted punctuation), six tokens a..f each.
ERROR_ROWS = (
    ("r1", "g1", ["O", "COMMA", "O", "COMMA", "O", "PERIOD"], ["O", "O", "COMMA", "O", "O", "PERIOD"]),
    ("r2", "g1", ["O", "O", "O", "O", "O", "QUESTION"], ["COMMA", "O", "O", "O", "O", "PERIOD"]),
    ("r3", "g2", ["O", "O", "PERIOD", "O", "O", "PERIOD"], ["O", "COMMA", "QUESTION", "O", "O", "PERIOD"]),
    ("r4", None, ["O", "PERIOD", "O", "O", "O", "PERIOD"], ["O", "QUESTION", "O", "O", "O", "PERIOD"]),
)
ERROR_IDS = ("r3", "r1", "r4", "r2")  # the scored order differs from the gold file order
ERROR_GROUPS = {sample_id: group for sample_id, group, _, _ in ERROR_ROWS}
ERROR_PAIR_COUNTS = [
    {"gold": "O", "pred": "COMMA", "count": 3},
    {"gold": "COMMA", "pred": "O", "count": 2},
    {"gold": "PERIOD", "pred": "QUESTION", "count": 2},
    {"gold": "QUESTION", "pred": "PERIOD", "count": 1},
]


def words(start: int, end: int) -> str:
    """Plain tokens t{start}..t{end - 1} joined by spaces."""
    return " ".join(f"t{i}" for i in range(start, end))


def bundle_of(case: Case, vocab: Vocab) -> Bundle:
    return load_bundle(case.gold, case.ids, case.predictions, case.masks, vocab)


def mode_a_errors_case(folder: Path) -> Case:
    """Mode A case with 8 wrong positions over four records (see ERROR_PAIR_COUNTS)."""
    gold = [gold_row(sid, "A", SIX_TOKENS, punct, SIX_LOWER, group) for sid, group, punct, _ in ERROR_ROWS]
    predictions = [pred_row(sid, "A", labels) for sid, _, _, labels in ERROR_ROWS]
    return write_case(folder, gold, ERROR_IDS, predictions)


def comma_errors_case(folder: Path) -> Case:
    """Two records with five O->COMMA errors each (ten in one pair)."""
    punct = ["O"] * 5 + ["PERIOD"]
    predictions = ["COMMA"] * 5 + ["PERIOD"]
    gold = [gold_row(sid, "A", SIX_TOKENS, punct, SIX_LOWER) for sid in ("s1", "s2")]
    return write_case(folder, gold, ("s1", "s2"), [pred_row(sid, "A", predictions) for sid in ("s1", "s2")])


def masked_mode_c_case(folder: Path) -> Case:
    """One mode C record with a casing-only error at 0-based position 1 (masked) and a punctuation error at 2."""
    gold = [gold_row("s1", "C", ["hello", "world", "now"], ["O", "O", "PERIOD"], ["FIRST_UPPER", "LOWER", "LOWER"])]
    predictions = [pred_row("s1", "C", ["O|FIRST_UPPER", "O|ALL_UPPER", "QUESTION|LOWER"])]
    return write_case(folder, gold, ["s1"], predictions, [mask_row("s1", [1])])


def pair_of(example: dict[str, Any]) -> tuple[str, str]:
    return example["gold"], example["pred"]


def drawn(report: dict[str, Any]) -> list[tuple[str, int]]:
    return [(example["sample_id"], example["position"]) for example in report["examples"]]


def test_constants() -> None:
    assert MARKS == {"O": "", "COMMA": ",", "PERIOD": ".", "QUESTION": "?", "EXCLAMATION": "!",
                     "COLON": ":", "SEMICOLON": ";", "DASH": " -"}
    assert WINDOW == 12
    assert DEFAULT_PER_PAIR == 3
    assert DEFAULT_BLOCK == "punctuation/all"
    assert SCHEMA == "recap-errors-v1"


def test_render_marks() -> None:
    labels = ["O", "COMMA", "PERIOD", "QUESTION", "EXCLAMATION", "COLON", "SEMICOLON", "DASH"]
    assert render(list("abcdefgh"), labels) == "a b, c. d? e! f: g; h -"


def test_render_joint_casing() -> None:
    tokens = ["hello", "nasa", "ok", "don't"]
    labels = ["O|FIRST_UPPER", "COMMA|ALL_UPPER", "PERIOD|LOWER", "QUESTION|FIRST_UPPER"]
    assert render(tokens, labels) == "Hello NASA, ok. Don't?"
    assert render(["eBay"], ["O|FIRST_UPPER"]) == "EBay"
    assert render(["NASA"], ["PERIOD"]) == "NASA."


WINDOW_CASES = [
    (0, "O", "COMMA", f"{words(0, 13)} {ELLIPSIS}", f"t0, {words(1, 13)} {ELLIPSIS}"),
    (12, "O", "COMMA", f"{words(0, 25)} {ELLIPSIS}", f"{words(0, 12)} t12, {words(13, 25)} {ELLIPSIS}"),
    (13, "O", "COMMA", f"{ELLIPSIS} {words(1, 26)} {ELLIPSIS}",
     f"{ELLIPSIS} {words(1, 13)} t13, {words(14, 26)} {ELLIPSIS}"),
    (16, "O", "COMMA", f"{ELLIPSIS} {words(4, 29)} {ELLIPSIS}",
     f"{ELLIPSIS} {words(4, 16)} t16, {words(17, 29)} {ELLIPSIS}"),
    (17, "O", "COMMA", f"{ELLIPSIS} {words(5, 29)} t29.",
     f"{ELLIPSIS} {words(5, 17)} t17, {words(18, 29)} t29."),
    (29, "PERIOD", "QUESTION", f"{ELLIPSIS} {words(17, 29)} t29.", f"{ELLIPSIS} {words(17, 29)} t29?"),
]


@pytest.mark.parametrize(("position", "gold", "pred", "gold_text", "pred_text"), WINDOW_CASES,
                         ids=[f"position{case[0]}" for case in WINDOW_CASES])
def test_window_cuts_at_12_tokens(tmp_path: Path, vocab: Vocab, position: int, gold: str, pred: str,
                                  gold_text: str, pred_text: str) -> None:
    tokens = [f"t{i}" for i in range(30)]
    gold_labels = ["O"] * 29 + ["PERIOD"]
    pred_labels = [pred if i == position else label for i, label in enumerate(gold_labels)]
    case = write_case(tmp_path, [gold_row("s1", "A", tokens, gold_labels, ["LOWER"] * 30)], ["s1"],
                      [pred_row("s1", "A", pred_labels)])
    report = error_examples(bundle_of(case, vocab), vocab, per_pair=3)
    assert report["examples"] == [{"gold": gold, "pred": pred, "sample_id": "s1", "group_id": None,
                                   "position": position, "gold_text": gold_text, "pred_text": pred_text}]
    assert report["pair_counts"] == [{"gold": gold, "pred": pred, "count": 1}]


def test_pair_counts_sum_and_order(tmp_path: Path, vocab: Vocab) -> None:
    report = error_examples(bundle_of(mode_a_errors_case(tmp_path), vocab), vocab)
    assert report["pair_counts"] == ERROR_PAIR_COUNTS
    assert report["errors_total"] == 8
    assert report["errors_total"] == sum(row["count"] for row in report["pair_counts"])


def test_examples_capped_per_pair_and_grouped(tmp_path: Path, vocab: Vocab) -> None:
    bundle = bundle_of(mode_a_errors_case(tmp_path), vocab)
    report = error_examples(bundle, vocab, per_pair=2)
    pairs = [pair_of(example) for example in report["examples"]]
    assert pairs == ([("O", "COMMA")] * 2 + [("COMMA", "O")] * 2 + [("PERIOD", "QUESTION")] * 2
                     + [("QUESTION", "PERIOD")])
    assert len(report["examples"]) == 7
    assert report["per_pair"] == 2
    scored_order = {sample_id: index for index, sample_id in enumerate(ERROR_IDS)}
    for pair in set(pairs):
        keys = [(scored_order[e["sample_id"]], e["position"]) for e in report["examples"] if pair_of(e) == pair]
        assert keys == sorted(set(keys))
    single = error_examples(bundle, vocab, per_pair=1)
    assert [pair_of(example) for example in single["examples"]] == [("O", "COMMA"), ("COMMA", "O"),
                                                                    ("PERIOD", "QUESTION"), ("QUESTION", "PERIOD")]
    assert error_examples(bundle, vocab)["per_pair"] == 3


def test_default_per_pair_lists_every_error_in_scored_order(tmp_path: Path, vocab: Vocab) -> None:
    report = error_examples(bundle_of(mode_a_errors_case(tmp_path), vocab), vocab)
    shown = [(e["gold"], e["pred"], e["sample_id"], e["position"]) for e in report["examples"]]
    assert shown == [("O", "COMMA", "r3", 1), ("O", "COMMA", "r1", 2), ("O", "COMMA", "r2", 0),
                     ("COMMA", "O", "r1", 1), ("COMMA", "O", "r1", 3),
                     ("PERIOD", "QUESTION", "r3", 2), ("PERIOD", "QUESTION", "r4", 1),
                     ("QUESTION", "PERIOD", "r2", 5)]


def test_examples_deterministic_from_seed(tmp_path: Path, vocab: Vocab) -> None:
    bundle = bundle_of(comma_errors_case(tmp_path), vocab)
    first = error_examples(bundle, vocab, per_pair=2, seed=5)
    again = error_examples(bundle, vocab, per_pair=2, seed=5)
    assert first == again
    assert first["seed"] == 5
    assert first["errors_total"] == 10
    assert len(first["examples"]) == 2
    selections = {tuple(drawn(error_examples(bundle, vocab, per_pair=2, seed=seed))) for seed in range(10)}
    assert len(selections) >= 2
    assert error_examples(bundle, vocab, per_pair=2)["seed"] == 2026


def test_examples_are_real_errors_with_context(tmp_path: Path, vocab: Vocab) -> None:
    bundle = bundle_of(mode_a_errors_case(tmp_path), vocab)
    report = error_examples(bundle, vocab, per_pair=10)
    records = {record.sample_id: record for record in bundle.records}
    assert len(report["examples"]) == 8
    for example in report["examples"]:
        record = records[example["sample_id"]]
        assert set(example) == EXAMPLE_KEYS
        assert example["gold"] != example["pred"]
        assert example["gold"] == record.gold[example["position"]]
        assert example["pred"] == record.pred[example["position"]]
        assert example["group_id"] == ERROR_GROUPS[example["sample_id"]]
        assert example["gold_text"] != example["pred_text"]
    last = next(example for example in report["examples"] if example["sample_id"] == "r4")
    assert last["group_id"] is None
    assert last["gold_text"] == "a b. c d e f."
    assert last["pred_text"] == "a b? c d e f."


MASKED_CASES = [
    ("joint/unmasked", [{"gold": "PERIOD|LOWER", "pred": "QUESTION|LOWER", "count": 1}]),
    ("punctuation/all", [{"gold": "PERIOD", "pred": "QUESTION", "count": 1}]),
    ("punctuation/non_final", []),
    ("casing/unmasked", []),
    ("casing/unmasked_non_initial", []),
    ("casing/all", [{"gold": "LOWER", "pred": "ALL_UPPER", "count": 1}]),
]


@pytest.mark.parametrize(("block", "expected"), MASKED_CASES, ids=[case[0] for case in MASKED_CASES])
def test_only_scored_positions_count(tmp_path: Path, vocab: Vocab, block: str, expected: list[dict[str, Any]]) -> None:
    bundle = bundle_of(masked_mode_c_case(tmp_path), vocab)
    report = error_examples(bundle, vocab, block=block)
    assert report["block"] == block
    assert report["pair_counts"] == expected
    assert report["errors_total"] == len(expected)
    assert len(report["examples"]) == len(expected)


def test_report_shape_and_defaults(tmp_path: Path, vocab: Vocab) -> None:
    predictions = [pred_row("s1", "A", ["O", "O", "PERIOD"]), *perfect_preds("A", ("s2",))]
    bundle = bundle_of(write_case(tmp_path, gold_rows("A"), SCORED, predictions), vocab)
    report = error_examples(bundle, vocab)
    assert set(report) == REPORT_KEYS
    assert report["schema"] == "recap-errors-v1"
    assert report["label_mode"] == "A"
    assert report["model_name"] == "m"
    assert report["block"] == "punctuation/all"
    assert report["per_pair"] == 3
    assert report["seed"] == 2026
    assert report["errors_total"] == 1
    assert report["inputs"] == {key: dict(value) if value is not None else None for key, value in bundle.inputs.items()}
    assert type(report["inputs"]["gold"]) is dict
    assert report["inputs"]["masks"] is None
    assert json.loads(json.dumps(report)) == report


def test_mode_c_default_block_is_punctuation_all(tmp_path: Path, vocab: Vocab) -> None:
    report = error_examples(bundle_of(valid_case(tmp_path, "C"), vocab), vocab)
    assert report["label_mode"] == "C"
    assert report["block"] == "punctuation/all"


def test_no_errors_gives_empty_lists(tmp_path: Path, vocab: Vocab) -> None:
    report = error_examples(bundle_of(valid_case(tmp_path, "A"), vocab), vocab)
    assert report["errors_total"] == 0
    assert report["pair_counts"] == []
    assert report["examples"] == []


@pytest.mark.parametrize(("arguments", "message"), [
    ({"block": "joint/unmasked"}, "block 'joint/unmasked' is not one of"),
    ({"block": "nope"}, "block 'nope' is not one of"),
    ({"per_pair": 0}, "per_pair must be at least 1"),
    ({"per_pair": -1}, "per_pair must be at least 1"),
])
def test_rejects_bad_arguments(tmp_path: Path, vocab: Vocab, arguments: dict[str, Any], message: str) -> None:
    bundle = bundle_of(valid_case(tmp_path, "A"), vocab)
    with pytest.raises(EvaluationInputError, match=message):
        error_examples(bundle, vocab, **arguments)


def test_rejects_bundle_without_tokens(vocab: Vocab) -> None:
    bundle = Bundle("A", "m", (Aligned("s1", None, ("O",), ("O",), frozenset()),), 0, {})
    with pytest.raises(EvaluationInputError, match="error examples need the gold tokens"):
        error_examples(bundle, vocab)


@pytest.mark.parametrize("mode", ["A", "B", "C"])
def test_load_bundle_fills_tokens(tmp_path: Path, vocab: Vocab, mode: str) -> None:
    bundle = bundle_of(valid_case(tmp_path, mode), vocab)
    assert [record.tokens for record in bundle.records] == [TOKENS["s1"], TOKENS["s2"]]
    assert all(isinstance(record.tokens, tuple) for record in bundle.records)


def test_aligned_tokens_default_to_empty() -> None:
    assert Aligned("x", None, (), (), frozenset()).tokens == ()


def mode_c_markdown_case(folder: Path) -> Case:
    """Mode C case with PERIOD|LOWER -> QUESTION|LOWER twice and O|FIRST_UPPER -> COMMA|FIRST_UPPER once."""
    gold = [gold_row("s1", "C", ["hello", "world", "now"], ["O", "O", "PERIOD"], ["FIRST_UPPER", "LOWER", "LOWER"]),
            gold_row("s2", "C", ["it", "rains", "now"], ["O", "O", "PERIOD"], ["LOWER"] * 3)]
    predictions = [pred_row("s1", "C", joint(["COMMA", "O", "QUESTION"], ["FIRST_UPPER", "LOWER", "LOWER"])),
                   pred_row("s2", "C", joint(["O", "O", "QUESTION"], ["LOWER"] * 3))]
    return write_case(folder, gold, ["s1", "s2"], predictions, [mask_row("s1", []), mask_row("s2", [])])


def test_errors_markdown(tmp_path: Path, vocab: Vocab) -> None:
    bundle = bundle_of(mode_c_markdown_case(tmp_path), vocab)
    report = error_examples(bundle, vocab, block="joint/unmasked", per_pair=2, seed=7)
    lines = errors_markdown(report).splitlines()
    assert lines[0] == "# Errors: m (mode C, block `joint/unmasked`)"
    assert "3 wrong positions. Up to 2 examples per pair, drawn at random with seed 7 (not hand-picked)." in lines
    table = [line for line in lines if line.startswith("|")]
    assert table == ["| Gold | Predicted | Count |", "| --- | --- | ---: |",
                     "| PERIOD\\|LOWER | QUESTION\\|LOWER | 2 |",
                     "| O\\|FIRST_UPPER | COMMA\\|FIRST_UPPER | 1 |"]
    cell_counts = {len(re.split(r"(?<!\\)\|", line)) for line in table}
    assert cell_counts == {len(re.split(r"(?<!\\)\|", table[0]))}
    assert [line for line in lines if line.startswith("- gold: ")] == [
        "- gold: Hello world now.", "- gold: it rains now.", "- gold: Hello world now."]
    assert [line for line in lines if line.startswith("- pred: ")] == [
        "- pred: Hello, world now?", "- pred: it rains now?", "- pred: Hello, world now?"]
