"""Paired comparison of two prediction sets: schema, B - A values, paired bootstrap, input checks, Markdown."""

from __future__ import annotations

import re
from collections.abc import Callable
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
    write_jsonl,
)

from punct_restore.evaluation import (
    BootstrapSpec,
    EvaluationInputError,
    compare,
    comparison_markdown,
    load_bundle,
)
from punct_restore.evaluation.inputs import Bundle, Vocab
from punct_restore.evaluation.metrics import score_block
from punct_restore.evaluation.report import pct, rounded
from punct_restore.evaluation.uncertainty import paired_differences
from punct_restore.evaluation.views import BlockSpec, block_pairs, blocks_for

Pair = tuple[Bundle, Bundle]
PairBuilder = Callable[[Path, Vocab], Pair]

TOKENS = ("w1", "w2", "w3")
CASING = ("LOWER", "LOWER", "LOWER")
GOLD_PUNCT = {"s0": ("O", "COMMA", "PERIOD"), "s1": ("O", "O", "QUESTION"),
              "s2": ("COMMA", "O", "PERIOD"), "s3": ("O", "O", "PERIOD")}
GROUP_OF = {"s0": "g0", "s1": "g1", "s2": "g2", "s3": "g3"}
IDS = tuple(GOLD_PUNCT)
# A misses both commas, turns the question into a period and invents a comma in s3; B only makes the s3 error.
PREDS_A = {"s0": ("O", "O", "PERIOD"), "s1": ("O", "O", "PERIOD"), "s2": ("O", "O", "PERIOD"),
           "s3": ("O", "COMMA", "PERIOD")}
PREDS_B = {**GOLD_PUNCT, "s3": ("O", "COMMA", "PERIOD")}
MODE_C_BLOCKS = ["joint/unmasked", "punctuation/all", "punctuation/non_final", "casing/unmasked",
                 "casing/unmasked_non_initial", "casing/all"]
CI_KEYS = {"low", "high", "level", "resamples", "seed", "unit", "groups", "share_b_better"}
BOOTSTRAP = BootstrapSpec(100, 11)
MINUS = "−"
HEAD_NO_CI = f"| Block | A: model_a % | B: model_b % | B {MINUS} A (points) |"
HEAD_CI = (f"| Block | A: model_a % | B: model_b % | B {MINUS} A (points) | 95% CI of B {MINUS} A | "
           "Share of resamples B > A |")
SAME_LABEL_MODE = "compared prediction sets must have the same label mode"
SAME_INPUTS = "compared prediction sets must use the same gold file, ID list and masks"


def _dir(path: Path) -> Path:
    path.mkdir(parents=True)
    return path


def _grouped_case(folder: Path, gold_punct: dict[str, tuple[str, ...]] = GOLD_PUNCT,
                  ids: tuple[str, ...] = IDS) -> Case:
    """Mode A over four groups (one record each), model_a predictions in preds.jsonl."""
    gold = [gold_row(sid, "A", TOKENS, gold_punct[sid], CASING, GROUP_OF[sid]) for sid in IDS]
    return write_case(_dir(folder), gold, ids, _predictions(PREDS_A, "model_a"))


def _predictions(labels: dict[str, tuple[str, ...]], model: str) -> list[dict[str, Any]]:
    return [pred_row(sid, "A", labels[sid], model) for sid in IDS]


def _bundle(case: Case, vocab: Vocab, predictions: Path | None = None) -> Bundle:
    return load_bundle(case.gold, case.ids, predictions or case.predictions, case.masks, vocab)


def _raw_macro_f1(bundle: Bundle, spec: BlockSpec) -> float:
    return score_block(block_pairs(bundle.records, spec), spec.labels, spec.macro_over)["macro_f1"]


def _row(name: str, block: dict[str, Any]) -> str:
    return (f"| {name} | {pct(block['macro_f1_a'])} | {pct(block['macro_f1_b'])} | "
            f"{pct(block['difference'])} |")


@pytest.fixture
def pair(tmp_path: Path, vocab: Vocab) -> Pair:
    """Bundles of model_a (preds.jsonl) and model_b (preds_b.jsonl) over the same grouped mode A case."""
    case = _grouped_case(tmp_path / "case")
    preds_b = write_jsonl(case.predictions.parent / "preds_b.jsonl", _predictions(PREDS_B, "model_b"))
    return _bundle(case, vocab), _bundle(case, vocab, preds_b)


@pytest.fixture
def report(pair: Pair, vocab: Vocab) -> dict[str, Any]:
    return compare(pair[0], pair[1], vocab)


@pytest.fixture
def ci_report(pair: Pair, vocab: Vocab) -> dict[str, Any]:
    return compare(pair[0], pair[1], vocab, BOOTSTRAP)


def test_compare_report_schema_without_bootstrap(pair: Pair, report: dict[str, Any], vocab: Vocab) -> None:
    assert set(report) == {"schema", "label_mode", "policy_version", "model_a", "model_b", "inputs_a",
                           "inputs_b", "sequences", "primary_block", "blocks"}
    assert report["schema"] == "recap-compare-v1"
    assert report["label_mode"] == "A"
    assert report["policy_version"] == pair[0].policy_version
    assert report["model_a"] == "model_a"
    assert report["model_b"] == "model_b"
    assert report["sequences"] == len(IDS) == len(pair[0].records)
    assert report["primary_block"] == "punctuation/all"
    assert list(report["blocks"]) == [spec.name for spec in blocks_for("A", vocab)]
    assert [set(block) for block in report["blocks"].values()] == [
        {"macro_f1_a", "macro_f1_b", "difference"}] * len(report["blocks"])


def test_compare_records_data_policy_version(tmp_path: Path, vocab: Vocab) -> None:
    gold = [{**row, "metadata": {"policy_version": "m1-v2"}} for row in gold_rows("A")]
    case = write_case(tmp_path, gold, SCORED, perfect_preds("A"))
    preds_b = write_jsonl(tmp_path / "preds_b.jsonl", perfect_preds("A", model="model_b"))
    versioned = compare(_bundle(case, vocab), _bundle(case, vocab, preds_b), vocab)
    assert versioned["policy_version"] == "m1-v2"
    assert "Data policy `m1-v2`." in comparison_markdown(versioned)


def test_compare_report_inputs_identify_both_prediction_files(report: dict[str, Any]) -> None:
    assert report["inputs_a"]["predictions"]["path"].endswith("preds.jsonl")
    assert report["inputs_b"]["predictions"]["path"].endswith("preds_b.jsonl")
    assert report["inputs_a"]["gold"] == report["inputs_b"]["gold"]
    assert type(report["inputs_a"]["gold"]) is dict


def test_difference_is_b_minus_a(pair: Pair, report: dict[str, Any], vocab: Vocab) -> None:
    expected = {}
    for spec in blocks_for("A", vocab):
        raw_a, raw_b = _raw_macro_f1(pair[0], spec), _raw_macro_f1(pair[1], spec)
        expected[spec.name] = {"macro_f1_a": round(raw_a, 6), "macro_f1_b": round(raw_b, 6),
                               "difference": round(raw_b - raw_a, 6)}
    assert report["blocks"] == expected
    assert report["blocks"]["punctuation/all"]["difference"] > 0


def test_swapping_a_and_b_negates_every_difference(pair: Pair, report: dict[str, Any], vocab: Vocab) -> None:
    swapped = compare(pair[1], pair[0], vocab)
    assert swapped["model_a"] == "model_b" and swapped["model_b"] == "model_a"
    assert {name: block["difference"] for name, block in swapped["blocks"].items()} == {
        name: -block["difference"] for name, block in report["blocks"].items()}


def test_compare_mode_c_primary_is_joint(tmp_path: Path, vocab: Vocab) -> None:
    case = write_case(_dir(tmp_path / "case"), gold_rows("C"), SCORED,
                      perfect_preds("C", model="model_a"), all_masks({"s1": [1]}))
    preds_b = write_jsonl(case.predictions.parent / "preds_b.jsonl", perfect_preds("C", model="model_b"))
    mode_c = compare(_bundle(case, vocab), _bundle(case, vocab, preds_b), vocab)
    assert mode_c["label_mode"] == "C"
    assert mode_c["primary_block"] == "joint/unmasked"
    assert list(mode_c["blocks"]) == MODE_C_BLOCKS


def test_compare_with_bootstrap_adds_paired_ci(ci_report: dict[str, Any]) -> None:
    for name, block in ci_report["blocks"].items():
        ci = block["difference_ci"]
        assert set(block) == {"macro_f1_a", "macro_f1_b", "difference", "difference_ci"}, name
        assert set(ci) == CI_KEYS, name
        assert (ci["level"], ci["resamples"], ci["seed"], ci["unit"], ci["groups"]) == (
            0.95, 100, 11, "group_id", 4), name
        assert ci["low"] <= ci["high"], name
        assert 0.0 <= ci["share_b_better"] <= 1.0, name


def test_compare_ci_is_b_minus_a(pair: Pair, ci_report: dict[str, Any], vocab: Vocab) -> None:
    """The paired CI and share are for B - A (A passed first), not A - B."""
    expected = paired_differences(pair[0].records, pair[1].records, blocks_for("A", vocab), BOOTSTRAP)
    assert {name: block["difference_ci"] for name, block in ci_report["blocks"].items()} == rounded(expected)
    primary = ci_report["blocks"]["punctuation/all"]["difference_ci"]
    assert primary["share_b_better"] > 0.5 and primary["high"] > 0.0  # B is the better model in this fixture


def test_comparing_a_bundle_with_itself_has_zero_difference(pair: Pair, vocab: Vocab) -> None:
    same = compare(pair[0], pair[0], vocab, BOOTSTRAP)
    summary = {name: (block["difference"], block["difference_ci"]["low"], block["difference_ci"]["high"],
                      block["difference_ci"]["share_b_better"])
               for name, block in same["blocks"].items()}
    assert summary == {name: (0.0, 0.0, 0.0, 0.0) for name in same["blocks"]}


def _different_label_mode(root: Path, vocab: Vocab) -> Pair:
    return (_bundle(valid_case(_dir(root / "a"), "A"), vocab),
            _bundle(valid_case(_dir(root / "b"), "B"), vocab))


def _different_gold(root: Path, vocab: Vocab) -> Pair:
    """Same relative paths (case/...), so only the gold content differs."""
    changed = {**GOLD_PUNCT, "s0": ("O", "O", "PERIOD")}
    return (_bundle(_grouped_case(root / "left" / "case"), vocab),
            _bundle(_grouped_case(root / "right" / "case", gold_punct=changed), vocab))


def _different_id_order(root: Path, vocab: Vocab) -> Pair:
    return (_bundle(_grouped_case(root / "left" / "case"), vocab),
            _bundle(_grouped_case(root / "right" / "case", ids=tuple(reversed(IDS))), vocab))


def _different_masks(root: Path, vocab: Vocab) -> Pair:
    cases = [write_case(_dir(root / side / "case"), gold_rows("C"), SCORED, perfect_preds("C"),
                        all_masks(positions))
             for side, positions in (("left", {}), ("right", {"s1": [1]}))]
    return _bundle(cases[0], vocab), _bundle(cases[1], vocab)


@pytest.mark.parametrize(("build", "message"), [
    pytest.param(_different_label_mode, SAME_LABEL_MODE, id="label_mode"),
    pytest.param(_different_gold, SAME_INPUTS, id="gold_file"),
    pytest.param(_different_id_order, SAME_INPUTS, id="id_list_order"),
    pytest.param(_different_masks, SAME_INPUTS, id="masks_file"),
])
def test_compare_rejects_mismatched_inputs(build: PairBuilder, message: str, tmp_path: Path,
                                           vocab: Vocab) -> None:
    bundle_a, bundle_b = build(tmp_path, vocab)
    with pytest.raises(EvaluationInputError, match=rf"\A{re.escape(message)}\Z"):
        compare(bundle_a, bundle_b, vocab)


def test_comparison_markdown_without_ci(report: dict[str, Any]) -> None:
    markdown = comparison_markdown(report)
    lines = markdown.splitlines()
    assert lines[0] == "# Comparison: model_b vs model_a (mode A)"
    assert HEAD_NO_CI in lines
    assert "95% CI" not in markdown
    for name, block in report["blocks"].items():
        assert _row(name, block) in lines


def test_comparison_markdown_with_ci(ci_report: dict[str, Any]) -> None:
    lines = comparison_markdown(ci_report).splitlines()
    assert HEAD_CI in lines
    for name, block in ci_report["blocks"].items():
        ci = block["difference_ci"]
        row = f"{_row(name, block)} {pct(ci['low'])} to {pct(ci['high'])} | {pct(ci['share_b_better'])}% |"
        assert row in lines


def test_comparison_markdown_sanitises_model_names(report: dict[str, Any]) -> None:
    markdown = comparison_markdown({**report, "model_a": "m\nx|y"})
    lines = markdown.splitlines()
    assert lines[0] == "# Comparison: model_b vs m?x\\|y (mode A)"
    assert next(line for line in lines if line.startswith("| Block")) == (
        f"| Block | A: m?x\\|y % | B: model_b % | B {MINUS} A (points) |")
    assert "m\nx" not in markdown
