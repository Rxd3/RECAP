"""Report schema, values, rounding and Markdown view of punct_restore.evaluation (spec sections 5 and 7)."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from itertools import takewhile
from pathlib import Path
from typing import Any

import pytest
from eval_fixtures import (
    SCORED,
    Case,
    all_masks,
    gold_row,
    gold_rows,
    perfect_preds,
    pred_row,
    valid_case,
    write_case,
)

from punct_restore.evaluation import evaluate, load_bundle, to_markdown
from punct_restore.evaluation.inputs import Bundle, Vocab, sha256_file
from punct_restore.evaluation.metrics import score_block
from punct_restore.evaluation.views import block_pairs, blocks_for

BLOCK_KEYS = {"labels", "macro_over", "scored_positions", "per_class", "macro_f1", "micro", "accuracy",
              "confusion", "empty_support"}
MODE_C_BLOCKS = ["joint/unmasked", "punctuation/all", "punctuation/non_final", "casing/unmasked",
                 "casing/unmasked_non_initial", "casing/all"]
UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")
ESCAPED_PIPE = r"\|"  # how a Markdown table cell shows a literal '|'


def _folder(tmp_path: Path) -> Path:
    folder = tmp_path / "case"
    folder.mkdir()
    return folder


def _bundle(case: Case, vocab: Vocab) -> Bundle:
    return load_bundle(case.gold, case.ids, case.predictions, case.masks, vocab)


def _report(case: Case, vocab: Vocab) -> dict[str, Any]:
    return evaluate(_bundle(case, vocab), vocab)


def _identity(folder: Path, name: str, path: Path | None) -> dict[str, str]:
    assert path is not None
    return {"path": f"{folder.name}/{name}", "sha256": sha256_file(path)}


def _mode_c_case(folder: Path) -> Case:
    """Mode C with s1 index 1 masked."""
    return write_case(folder, gold_rows("C"), SCORED, perfect_preds("C"), all_masks({"s1": [1]}))


def _rounding_case(folder: Path) -> Case:
    """Mode A: s1 final QUESTION is predicted PERIOD, s2 is perfect."""
    predictions = [pred_row("s1", "A", ["O", "O", "PERIOD"])] + perfect_preds("A", ("s2",))
    return write_case(folder, gold_rows("A"), SCORED, predictions)


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    return value


def _floats_in(value: Any) -> Iterator[float]:
    if isinstance(value, float):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _floats_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from _floats_in(item)


def _pct(value: float) -> str:
    return f"{100 * value:.2f}"


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _header_text(markdown: str) -> str:
    return "\n".join(takewhile(lambda line: not line.startswith("|"), markdown.splitlines()))


def _lines_starting(markdown: str, prefix: str) -> list[str]:
    return [line for line in markdown.splitlines() if line.startswith(prefix)]


def test_report_schema_mode_a(tmp_path: Path, vocab: Vocab) -> None:
    folder = _folder(tmp_path)
    case = valid_case(folder, "A")
    report = _report(case, vocab)
    assert set(report) == {"schema", "label_mode", "model_name", "inputs", "counts", "primary", "blocks"}
    assert report["schema"] == "recap-eval-v1"
    assert report["label_mode"] == "A"
    assert report["model_name"] == "m"
    assert report["inputs"] == {"gold": _identity(folder, "gold.jsonl", case.gold),
                                "ids": _identity(folder, "list.ids", case.ids),
                                "predictions": _identity(folder, "preds.jsonl", case.predictions),
                                "masks": None}
    assert report["counts"] == {"sequences": 2, "tokens": 6, "ignored_prediction_ids": 0}
    assert report["primary"] == {"block": "punctuation/all", "metric": "macro_f1", "value": 1.0}
    assert list(report["blocks"]) == ["punctuation/all", "punctuation/non_final"]
    for block in report["blocks"].values():
        assert set(block) == BLOCK_KEYS
        assert list(block["per_class"]) == ["O", "COMMA", "PERIOD", "QUESTION"]
    assert report["blocks"]["punctuation/non_final"]["scored_positions"] == 4


def test_report_counts_scored_sequences_only(tmp_path: Path, vocab: Vocab) -> None:
    predictions = perfect_preds("A", ("s1", "s2", "s3"))
    case = write_case(_folder(tmp_path), gold_rows("A"), SCORED, predictions)
    report = _report(case, vocab)
    assert report["counts"] == {"sequences": 2, "tokens": 6, "ignored_prediction_ids": 1}


def test_report_mode_c_primary_is_joint(tmp_path: Path, vocab: Vocab) -> None:
    folder = _folder(tmp_path)
    case = _mode_c_case(folder)
    report = _report(case, vocab)
    assert report["primary"]["block"] == "joint/unmasked"
    assert list(report["blocks"]) == MODE_C_BLOCKS
    assert report["inputs"]["masks"] == _identity(folder, "masks.jsonl", case.masks)
    scored = {name: block["scored_positions"] for name, block in report["blocks"].items()}
    assert scored == {"joint/unmasked": 5, "punctuation/all": 6, "punctuation/non_final": 4,
                      "casing/unmasked": 5, "casing/unmasked_non_initial": 3, "casing/all": 6}


def test_report_values_match_metrics(tmp_path: Path, vocab: Vocab) -> None:
    predictions = [pred_row("s1", "A", ["O", "COMMA", "PERIOD"]), pred_row("s2", "A", ["O", "O", "PERIOD"])]
    case = write_case(_folder(tmp_path), gold_rows("A"), SCORED, predictions)
    bundle = _bundle(case, vocab)
    report = evaluate(bundle, vocab)
    for spec in blocks_for("A", vocab):
        expected = score_block(block_pairs(bundle.records, spec), spec.labels, spec.macro_over)
        actual = report["blocks"][spec.name]
        for key in BLOCK_KEYS:
            assert actual[key] == _round_floats(expected[key]), f"{spec.name}: {key}"
    assert report["primary"]["value"] == report["blocks"]["punctuation/all"]["macro_f1"]


def test_report_floats_rounded_to_6_decimals(tmp_path: Path, vocab: Vocab) -> None:
    report = _report(_rounding_case(_folder(tmp_path)), vocab)
    block = report["blocks"]["punctuation/all"]
    assert block["per_class"]["PERIOD"]["f1"] == 0.666667
    assert block["macro_f1"] == 0.555556
    assert report["primary"]["value"] == 0.555556
    floats = list(_floats_in(report))
    assert floats
    assert all(value == round(value, 6) for value in floats)


def test_report_empty_support_has_no_nan(tmp_path: Path, vocab: Vocab) -> None:
    gold = [gold_row("s1", "A", ["a", "b"], ["O", "O"], ["LOWER", "LOWER"])]
    case = write_case(_folder(tmp_path), gold, ["s1"], [pred_row("s1", "A", ["O", "O"])])
    report = _report(case, vocab)
    for block in report["blocks"].values():
        assert block["empty_support"] is True
        assert block["macro_f1"] == 0.0
    assert report["primary"]["value"] == 0.0
    json.dumps(report, allow_nan=False)


def test_report_has_no_absolute_paths(tmp_path: Path, vocab: Vocab) -> None:
    report = _report(_mode_c_case(_folder(tmp_path)), vocab)
    forbidden = {str(tmp_path), tmp_path.as_posix(), str(tmp_path.resolve()), tmp_path.resolve().as_posix(),
                 json.dumps(str(tmp_path))[1:-1]}
    for text in (json.dumps(report), to_markdown(report)):
        for path_text in forbidden:
            assert path_text not in text


@pytest.mark.parametrize("mode", ["A", "C"])
def test_markdown_contains_rows(tmp_path: Path, vocab: Vocab, mode: str) -> None:
    folder = _folder(tmp_path)
    case = _rounding_case(folder) if mode == "A" else valid_case(folder, "C")
    report = _report(case, vocab)
    markdown = to_markdown(report)
    header = _header_text(markdown)
    assert re.search(r"\bm\b", header)
    assert re.search(rf"\b{mode}\b", header)
    for name, block in report["blocks"].items():
        rows = _lines_starting(markdown, f"| {name} |")
        assert len(rows) == 1, name
        assert _table_cells(rows[0])[1:5] == [str(block["scored_positions"]), _pct(block["macro_f1"]),
                                              _pct(block["micro"]["f1"]), _pct(block["accuracy"])], name


def test_markdown_per_class_table_for_primary_block_only(tmp_path: Path, vocab: Vocab) -> None:
    report = _report(_rounding_case(_folder(tmp_path)), vocab)
    markdown = to_markdown(report)
    assert _table_cells(_lines_starting(markdown, "| punctuation/all |")[0])[2] == "55.56"
    assert _lines_starting(markdown, "| COMMA |") == ["| COMMA | 100.00 | 100.00 | 100.00 | 1 | 1 |"]
    for label in ("O", "PERIOD", "QUESTION"):
        assert len(_lines_starting(markdown, f"| {label} |")) == 1, label


def test_markdown_header_shows_full_input_hashes(tmp_path: Path, vocab: Vocab) -> None:
    report = _report(valid_case(_folder(tmp_path), "C"), vocab)
    markdown = to_markdown(report)
    shown = {key: value["sha256"] for key, value in report["inputs"].items() if value is not None}
    assert set(shown) == {"gold", "ids", "predictions", "masks"}
    for key, digest in shown.items():
        assert len(digest) == 64
        assert digest in markdown, f"full SHA-256 of {key} is missing from the Markdown"


def test_markdown_mode_c_per_class_table_keeps_columns(tmp_path: Path, vocab: Vocab) -> None:
    report = _report(valid_case(_folder(tmp_path), "C"), vocab)
    lines = to_markdown(report).splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("| Label |"))
    table = list(takewhile(lambda line: line.startswith("|"), lines[start:]))
    width = len(UNESCAPED_PIPE.split(table[0]))
    rows = table[2:]  # skip the header and the alignment row
    assert len(rows) == len(vocab.C)
    for row in rows:
        assert len(UNESCAPED_PIPE.split(row)) == width, f"unescaped '|' in label breaks the table: {row}"


@pytest.mark.parametrize(("raw", "shown"),
                         [("m\nx|y", "m?x" + ESCAPED_PIPE + "y"), ("a\tb\rc\x1bd", "a?b?c?d"),
                          ("a|b|c", "a" + ESCAPED_PIPE + "b" + ESCAPED_PIPE + "c"), ("模型", "模型"),
                          ("a\x7fb\x85c", "a?b?c")],
                         ids=["newline_and_pipe", "tab_cr_escape", "pipes", "non_ascii_kept", "del_and_c1_nel"])
def test_markdown_sanitises_model_name(tmp_path: Path, vocab: Vocab, raw: str, shown: str) -> None:
    report = _report(valid_case(_folder(tmp_path), "A"), vocab)
    markdown = to_markdown({**report, "model_name": raw})
    assert markdown.split("\n")[0] == f"# Evaluation: {shown} (mode A)"
    assert markdown.count("\n") == to_markdown(report).count("\n")
    if "\n" in raw or "\r" in raw:
        assert raw not in markdown
