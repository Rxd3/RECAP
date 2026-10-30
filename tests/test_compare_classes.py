"""compare --class LABEL: per-class F1 difference (B - A), validation and the unchanged default path."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from cli_analysis_cases import (
    assert_nothing_written,
    assert_one_line,
    bundle_of,
    compare_argv,
    predictions_b,
    read_json,
)
from eval_fixtures import (
    Case,
    assert_one_error_line,
    gold_row,
    pred_row,
    run_cli,
    valid_case,
    write_case,
    write_jsonl,
)

from punct_restore.evaluation import (
    BootstrapSpec,
    EvaluationInputError,
    compare,
    comparison_markdown,
)
from punct_restore.evaluation import comparison as comparison_module
from punct_restore.evaluation.inputs import Vocab
from punct_restore.evaluation.report import rounded
from punct_restore.evaluation.uncertainty import paired_differences
from punct_restore.evaluation.views import blocks_for

TOKENS = ("w1", "w2", "w3")
CASING = ("LOWER", "LOWER", "LOWER")
GOLD = {"s0": ("O", "COMMA", "PERIOD"), "s1": ("COMMA", "O", "PERIOD"),
        "s2": ("O", "COMMA", "QUESTION"), "s3": ("O", "O", "PERIOD")}
PREDS_A = {"s0": GOLD["s0"], "s1": ("O", "O", "PERIOD"), "s2": ("O", "O", "QUESTION"),
           "s3": ("O", "O", "COMMA")}
PREDS_B = {"s0": GOLD["s0"], "s1": GOLD["s1"], "s2": ("O", "O", "QUESTION"),
           "s3": ("O", "COMMA", "PERIOD")}
# Every record is one video (group "g"), so each bootstrap resample draws that video only and the
# interval of the difference collapses to the point difference.
#
# COMMA, punctuation/all (12 positions): gold commas s0[1], s1[0], s2[1] -> support 3.
#   A predicts commas at s0[1] (right) and s3[2] (wrong): tp 1, predicted 2, P = 1/2, R = 1/3,
#     F1 = 2 * (1/2 * 1/3) / (1/2 + 1/3) = 2/5 = 0.4
#   B predicts commas at s0[1], s1[0] (right) and s3[1] (wrong): tp 2, predicted 3, P = R = 2/3,
#     F1 = 2/3 = 0.666667
#   B - A = 2/3 - 2/5 = 4/15 = 0.266667
# COMMA, punctuation/non_final (positions 0 and 1 only): A's wrong comma sits at s3[2], the final
#   position, so it drops out: tp 1, predicted 1, P = 1, R = 1/3, F1 = 2 * (1/3) / (4/3) = 0.5.
#   B's three commas and the three gold commas are all non-final: F1 = 2/3 = 0.666667.
#   B - A = 2/3 - 1/2 = 1/6 = 0.166667
# QUESTION: gold s2[2], found by both models -> all: F1 1.0 for A and B, difference 0.0.
#   It is the final position, so non_final has no support and no predictions: F1 0.0 (zero division).
HAND = {"COMMA": {"punctuation/all": (0.4, 0.666667, 0.266667),
                  "punctuation/non_final": (0.5, 0.666667, 0.166667)},
        "QUESTION": {"punctuation/all": (1.0, 1.0, 0.0), "punctuation/non_final": (0.0, 0.0, 0.0)}}
CI_KEYS = {"low", "high", "level", "resamples", "seed", "unit", "groups", "share_b_better"}
MINUS = "−"
CLASS_HEAD = "| Class | Block | A % | B % | B − A (points) |"
SECTION = "\n## Per-class F1 (B − A)\n\n"


def _predictions(labels: dict[str, tuple[str, ...]], model: str) -> list[dict[str, Any]]:
    return [pred_row(sid, "A", labels[sid], model) for sid in GOLD]


@pytest.fixture
def hand_case(tmp_path: Path) -> Case:
    """Mode A, four 3-token records of one video; model_a in preds.jsonl, model_b in preds_b.jsonl."""
    folder = tmp_path / "hand"
    folder.mkdir()
    gold = [gold_row(sid, "A", TOKENS, GOLD[sid], CASING, "g") for sid in GOLD]
    case = write_case(folder, gold, tuple(GOLD), _predictions(PREDS_A, "model_a"))
    write_jsonl(folder / "preds_b.jsonl", _predictions(PREDS_B, "model_b"))
    return case


def _class_args(*labels: str) -> list[str]:
    return [part for label in labels for part in ("--class", label)]


def _cli_report(cli: ModuleType, capsys: pytest.CaptureFixture[str], case: Case, out: Path,
                *extra: str) -> dict[str, Any]:
    code, _, err = run_cli(cli, capsys, compare_argv(case, out, *extra))
    assert (code, err) == (0, "")
    return read_json(out)


# ---------------------------------------------------------------- hand-checked values (mode A)
def test_class_f1_matches_hand_computation(cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path,
                                           hand_case: Case) -> None:
    report = _cli_report(cli, capsys, hand_case, tmp_path / "cmp.json", *_class_args("COMMA", "QUESTION"))

    assert list(report["classes"]) == ["COMMA", "QUESTION"]
    assert report["classes"] == {
        label: {block: {"f1_a": a, "f1_b": b, "difference": difference}
                for block, (a, b, difference) in blocks.items()}
        for label, blocks in HAND.items()}


def test_class_interval_collapses_on_a_single_video(cli: ModuleType, capsys: pytest.CaptureFixture[str],
                                                    tmp_path: Path, hand_case: Case) -> None:
    report = _cli_report(cli, capsys, hand_case, tmp_path / "cmp.json", "--bootstrap", "100",
                         *_class_args("COMMA", "QUESTION"))

    for label, blocks in HAND.items():
        for block, (a, b, difference) in blocks.items():
            row = report["classes"][label][block]
            ci = row["difference_ci"]
            assert (row["f1_a"], row["f1_b"], row["difference"]) == (a, b, difference), (label, block)
            assert set(row) == {"f1_a", "f1_b", "difference", "difference_ci"}
            assert set(ci) == CI_KEYS
            assert ci["low"] == ci["high"] == difference, (label, block)
            assert ci["share_b_better"] == (1.0 if difference > 0 else 0.0), (label, block)
            assert (ci["resamples"], ci["groups"]) == (100, 1)


def test_markdown_section_without_ci(hand_case: Case, vocab: Vocab) -> None:
    plain = compare(bundle_of(hand_case, vocab), bundle_of(hand_case, vocab, predictions_b(hand_case)), vocab)
    report = compare(bundle_of(hand_case, vocab), bundle_of(hand_case, vocab, predictions_b(hand_case)), vocab,
                     classes=("COMMA",))

    assert comparison_markdown(report) == comparison_markdown(plain) + (
        f"{SECTION}{CLASS_HEAD}\n| --- | --- | ---: | ---: | ---: |\n"
        "| COMMA | punctuation/all | 40.00 | 66.67 | 26.67 |\n"
        "| COMMA | punctuation/non_final | 50.00 | 66.67 | 16.67 |\n")


def test_markdown_section_with_ci(hand_case: Case, vocab: Vocab) -> None:
    bootstrap = BootstrapSpec(100, 5)
    bundles = (bundle_of(hand_case, vocab), bundle_of(hand_case, vocab, predictions_b(hand_case)))
    plain = compare(*bundles, vocab, bootstrap)
    report = compare(*bundles, vocab, bootstrap, classes=("COMMA",))

    assert comparison_markdown(report) == comparison_markdown(plain) + (
        f"{SECTION}{CLASS_HEAD} 95% CI of B {MINUS} A | Share of resamples B > A |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |\n"
        "| COMMA | punctuation/all | 40.00 | 66.67 | 26.67 | 26.67 to 26.67 | 100.00% |\n"
        "| COMMA | punctuation/non_final | 50.00 | 66.67 | 16.67 | 16.67 to 16.67 | 100.00% |\n")


def test_mode_b_accepts_its_own_labels(tmp_path: Path, vocab: Vocab) -> None:
    case = valid_case(tmp_path, "B")
    bundle = bundle_of(case, vocab)

    report = compare(bundle, bundle, vocab, classes=("COLON",))

    assert report["classes"] == {"COLON": {block: {"f1_a": 1.0, "f1_b": 1.0, "difference": 0.0}
                                           for block in ("punctuation/all", "punctuation/non_final")}}


# ---------------------------------------------------------------- mode C
def test_mode_c_class_uses_the_joint_block_only(cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path,
                                                mode_c_case: Case) -> None:
    out = tmp_path / "cmp.json"

    report = _cli_report(cli, capsys, mode_c_case, out, *_class_args("COMMA|LOWER"))

    # COMMA|LOWER: gold at s1[1], s3[1], s5[1], s7[1] (support 4).
    #   model_a misses s5[1]: tp 3, predicted 3, P = 1, R = 3/4, F1 = 6/7 = 0.857143
    #   model_b misses s3[1] and adds a wrong comma at s7[2]: tp 3, predicted 4, P = R = 3/4, F1 = 0.75
    #   B - A = 0.75 - 6/7 = -0.107143
    assert report["classes"] == {"COMMA|LOWER": {
        "joint/unmasked": {"f1_a": 0.857143, "f1_b": 0.75, "difference": -0.107143}}}
    lines = out.with_suffix(".md").read_text(encoding="utf-8").splitlines()
    assert "| COMMA\\|LOWER | joint/unmasked | 85.71 | 75.00 | -10.71 |" in lines
    assert not any(line.startswith("| COMMA|LOWER") for line in lines)


# ---------------------------------------------------------------- refused labels
def _refused(cli: ModuleType, capsys: pytest.CaptureFixture[str], case: Case, out: Path, *extra: str) -> str:
    code, stdout, err = run_cli(cli, capsys, compare_argv(case, out, *extra))
    assert code == 1 and stdout == ""
    assert_one_error_line(err)
    assert_nothing_written(out)
    return err


@pytest.mark.parametrize("extra", [_class_args("BANG"), _class_args("COMMA", "BANG"),
                                   ["--bootstrap", "100", *_class_args("BANG")]],
                         ids=["unknown", "unknown_after_valid", "unknown_with_bootstrap"])
def test_unknown_label_in_mode_a_is_refused(cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path,
                                            vocab: Vocab, grouped_case: Case, extra: list[str]) -> None:
    err = _refused(cli, capsys, grouped_case, tmp_path / "cmp.json", *extra)

    assert err == f"error: --class 'BANG' is not a mode A label; choose from {list(vocab.A)}\n"


def test_plain_punctuation_label_is_refused_in_mode_c(cli: ModuleType, capsys: pytest.CaptureFixture[str],
                                                      tmp_path: Path, vocab: Vocab, mode_c_case: Case) -> None:
    err = _refused(cli, capsys, mode_c_case, tmp_path / "cmp.json", *_class_args("COMMA"))

    assert err == f"error: --class 'COMMA' is not a mode C label; choose from {list(vocab.C)}\n"


@pytest.mark.parametrize("labels", [("COMMA", "COMMA"), ("COMMA", "PERIOD", "COMMA")],
                         ids=["adjacent", "apart"])
def test_repeated_label_is_refused(cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path,
                                   grouped_case: Case, labels: tuple[str, ...]) -> None:
    err = _refused(cli, capsys, grouped_case, tmp_path / "cmp.json", *_class_args(*labels))

    assert err == "error: --class 'COMMA' was given more than once\n"


def test_labels_are_checked_before_any_bootstrap_work(monkeypatch: pytest.MonkeyPatch, grouped_case: Case,
                                                      vocab: Vocab) -> None:
    def must_not_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("bootstrap started before the labels were checked")

    monkeypatch.setattr(comparison_module, "paired_differences", must_not_run)
    bundles = (bundle_of(grouped_case, vocab), bundle_of(grouped_case, vocab, predictions_b(grouped_case)))

    with pytest.raises(EvaluationInputError, match="--class 'BANG'"):
        compare(*bundles, vocab, BootstrapSpec(100), classes=("COMMA", "BANG"))


# ---------------------------------------------------------------- the default path does not change
@pytest.mark.parametrize("extra", [[], ["--bootstrap", "100", "--seed", "3"]], ids=["plain", "bootstrap"])
def test_without_class_the_files_are_the_unchanged_compare_output(
        cli: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path, vocab: Vocab, grouped_case: Case,
        extra: list[str]) -> None:
    out = tmp_path / "cmp.json"
    bootstrap = BootstrapSpec(100, 3) if extra else None

    _cli_report(cli, capsys, grouped_case, out, *extra)

    expected = compare(bundle_of(grouped_case, vocab), bundle_of(grouped_case, vocab, predictions_b(grouped_case)),
                       vocab, bootstrap)
    assert "classes" not in expected and "classes" not in read_json(out)
    assert out.read_bytes() == (json.dumps(expected, indent=1, sort_keys=True) + "\n").encode("utf-8")
    assert out.with_suffix(".md").read_bytes() == comparison_markdown(expected).encode("utf-8")


@pytest.mark.parametrize("bootstrap", [None, BootstrapSpec(100, 3)], ids=["plain", "bootstrap"])
def test_empty_classes_return_the_same_report(grouped_case: Case, vocab: Vocab,
                                              bootstrap: BootstrapSpec | None) -> None:
    bundles = (bundle_of(grouped_case, vocab), bundle_of(grouped_case, vocab, predictions_b(grouped_case)))

    explicit = compare(*bundles, vocab, bootstrap, classes=())

    assert explicit == compare(*bundles, vocab, bootstrap)
    assert "classes" not in explicit


def test_class_option_leaves_blocks_and_summary_untouched(cli: ModuleType, capsys: pytest.CaptureFixture[str],
                                                          tmp_path: Path, grouped_case: Case) -> None:
    runs = {}
    for name, extra in (("plain", []), ("classed", _class_args("COMMA", "QUESTION"))):
        folder = tmp_path / name
        folder.mkdir()
        out = folder / "cmp.json"
        code, stdout, err = run_cli(cli, capsys, compare_argv(grouped_case, out, "--bootstrap", "100", *extra))
        assert (code, err) == (0, "")
        assert_one_line(stdout)
        runs[name] = (read_json(out), out.with_suffix(".md").read_text(encoding="utf-8"), stdout)

    (plain, plain_md, plain_out), (classed, classed_md, classed_out) = runs["plain"], runs["classed"]
    assert {key: value for key, value in classed.items() if key != "classes"} == plain
    assert classed["blocks"] == plain["blocks"]
    assert list(classed["classes"]) == ["COMMA", "QUESTION"]
    assert classed_md.startswith(plain_md) and len(classed_md) > len(plain_md)
    assert classed_out == plain_out


def test_class_interval_is_the_paired_interval_on_the_block_resamples(grouped_case: Case, vocab: Vocab) -> None:
    bootstrap = BootstrapSpec(100, 7)
    bundle_a = bundle_of(grouped_case, vocab)
    bundle_b = bundle_of(grouped_case, vocab, predictions_b(grouped_case))

    report = compare(bundle_a, bundle_b, vocab, bootstrap, classes=("COMMA",))

    for spec in blocks_for("A", vocab):
        single = replace(spec, macro_over=("COMMA",))
        expected = paired_differences(bundle_a.records, bundle_b.records, [single], bootstrap)[spec.name]
        row = report["classes"]["COMMA"][spec.name]
        assert row["difference_ci"] == rounded(expected)
        assert row["difference_ci"]["groups"] == report["blocks"][spec.name]["difference_ci"]["groups"] == 4
