"""Grouped bootstrap of punct_restore.evaluation.uncertainty: resampling units, shared draws,
interval method, report/Markdown integration and the paired comparison (design section F1)."""

from __future__ import annotations

import dataclasses
import random
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pytest
from eval_fixtures import gold_row, joint, mask_row, pred_row, write_case

from punct_restore.evaluation import evaluate, load_bundle, to_markdown
from punct_restore.evaluation.inputs import Aligned, Bundle, EvaluationInputError, Vocab
from punct_restore.evaluation.metrics import score_block, split_joint
from punct_restore.evaluation.report import pct
from punct_restore.evaluation.uncertainty import (
    DEFAULT_SEED,
    MAX_RESAMPLES,
    MIN_RESAMPLES,
    BootstrapSpec,
    block_intervals,
    draw_weights,
    group_counts,
    group_order,
    interval,
    paired_differences,
    resampled_macro_f1,
)
from punct_restore.evaluation.views import BlockSpec, block_pairs, blocks_for

CI_KEYS = {"low", "high", "level", "resamples", "seed", "unit", "groups"}
HEADER_PLAIN = "| Block | Scored positions | Macro F1 % | Micro F1 % | Accuracy % | Flag |"
HEADER_CI = HEADER_PLAIN + " Macro F1 95% CI |"
CI_SEPARATOR = " to "  # report.ci_text(); the first brief expected an EN DASH here

GOLD_ONE = ("COMMA", "O", "PERIOD")
GOLD_TWO = ("O", "QUESTION", "COMMA")
WRONG_ONE = ("PERIOD", "COMMA", "O")  # differs from GOLD_ONE at every position
WRONG_TWO = ("COMMA", "O", "QUESTION")  # differs from GOLD_TWO at every position

# (group_id, gold, pred) rows for the punctuation-only modes
PUNCT_ROWS = {
    "A": (("g1", ("O", "COMMA", "PERIOD"), ("O", "O", "PERIOD")),
          ("g1", ("COMMA", "O", "QUESTION"), ("COMMA", "PERIOD", "QUESTION")),
          ("g2", ("O", "O", "PERIOD"), ("O", "COMMA", "PERIOD")),
          ("g3", ("COMMA", "QUESTION"), ("PERIOD", "QUESTION"))),
    "B": (("g1", ("O", "EXCLAMATION", "COLON"), ("O", "PERIOD", "COLON")),
          ("g1", ("SEMICOLON", "O", "QUESTION"), ("SEMICOLON", "DASH", "QUESTION")),
          ("g2", ("O", "DASH", "PERIOD"), ("O", "DASH", "COMMA")),
          ("g3", ("COMMA", "EXCLAMATION"), ("PERIOD", "EXCLAMATION"))),
}


def rec(sample_id: str, group_id: str | None, gold: Iterable[str], pred: Iterable[str],
        masked: Iterable[int] = ()) -> Aligned:
    return Aligned(sample_id, group_id, tuple(gold), tuple(pred), frozenset(masked))


def joint_records() -> list[Aligned]:
    """Mode C records over three groups: joint labels, some wrong, one masked position."""
    return [
        rec("c1", "g1", joint(("O", "COMMA", "PERIOD"), ("FIRST_UPPER", "LOWER", "LOWER")),
            joint(("O", "O", "PERIOD"), ("FIRST_UPPER", "LOWER", "ALL_UPPER")), masked={2}),
        rec("c2", "g1", joint(("COMMA", "O", "QUESTION"), ("LOWER", "LOWER", "FIRST_UPPER")),
            joint(("COMMA", "PERIOD", "QUESTION"), ("LOWER", "FIRST_UPPER", "FIRST_UPPER"))),
        rec("c3", "g2", joint(("O", "O", "PERIOD"), ("FIRST_UPPER", "LOWER", "LOWER")),
            joint(("O", "COMMA", "PERIOD"), ("FIRST_UPPER", "LOWER", "LOWER"))),
        rec("c4", "g3", joint(("COMMA", "QUESTION"), ("LOWER", "ALL_UPPER")),
            joint(("PERIOD", "QUESTION"), ("LOWER", "ALL_UPPER"))),
    ]


def records_for(mode: str) -> list[Aligned]:
    if mode == "C":
        return joint_records()
    return [rec(f"{mode.lower()}{number}", group, gold, pred)
            for number, (group, gold, pred) in enumerate(PUNCT_ROWS[mode], 1)]


def quality_records() -> list[Aligned]:
    """Mode A records over six groups: two perfect, two all wrong, two mixed."""
    return [
        rec("a1", "g1", GOLD_ONE, GOLD_ONE),
        rec("a2", "g2", GOLD_TWO, GOLD_TWO),
        rec("a3", "g3", GOLD_ONE, WRONG_ONE),
        rec("a4", "g4", GOLD_TWO, WRONG_TWO),
        rec("a5", "g5", GOLD_ONE, ("COMMA", "O", "O")),
        rec("a6", "g6", GOLD_TWO, ("O", "QUESTION", "PERIOD")),
    ]


def identical_group_records() -> list[Aligned]:
    """Four groups that are exact copies of each other, each with two imperfect records."""
    records = []
    for group in ("g1", "g2", "g3", "g4"):
        records.append(rec(f"{group}-1", group, GOLD_ONE, ("COMMA", "O", "O")))
        records.append(rec(f"{group}-2", group, GOLD_TWO, ("O", "QUESTION", "PERIOD")))
    return records


def final_o_records() -> list[Aligned]:
    """Mode A records over five groups; every record ends in gold O predicted O."""
    rows = (("g1", ("COMMA", "PERIOD", "O"), ("COMMA", "PERIOD", "O")),
            ("g2", ("QUESTION", "COMMA", "O"), ("O", "COMMA", "O")),
            ("g3", ("PERIOD", "O", "O"), ("COMMA", "O", "O")),
            ("g4", ("O", "COMMA", "O"), ("O", "PERIOD", "O")),
            ("g5", ("COMMA", "QUESTION", "O"), ("COMMA", "QUESTION", "O")))
    return [rec(f"{group}-1", group, gold, pred) for group, gold, pred in rows]


def spec_named(specs: Sequence[BlockSpec], name: str) -> BlockSpec:
    return next(spec for spec in specs if spec.name == name)


def point_estimate(records: Sequence[Aligned], spec: BlockSpec) -> float:
    return score_block(block_pairs(records, spec), spec.labels, spec.macro_over)["macro_f1"]


def write_bundle(folder: Path, records: Sequence[Aligned], mode: str, vocab: Vocab) -> Bundle:
    """Write the records as gold/ids/predictions (+ masks for mode C) and load them back."""
    gold, preds, masks = [], [], []
    for record in records:
        tokens = [f"w{i}" for i in range(len(record.gold))]
        if mode == "C":
            punct, case = zip(*map(split_joint, record.gold))
        else:
            punct, case = record.gold, ("LOWER",) * len(record.gold)
        gold.append(gold_row(record.sample_id, mode, tokens, punct, case, record.group_id))
        preds.append(pred_row(record.sample_id, mode, record.pred))
        masks.append(mask_row(record.sample_id, sorted(record.masked)))
    case_files = write_case(folder, gold, [r.sample_id for r in records], preds,
                            masks if mode == "C" else None)
    return load_bundle(case_files.gold, case_files.ids, case_files.predictions,
                       case_files.masks, vocab)


def block_header(lines: Iterable[str]) -> str:
    return next(line for line in lines if line.startswith("| Block |"))


# --- BootstrapSpec, interval, draws, group_order -----------------------------------------------

def test_bootstrap_spec_minimum() -> None:
    spec = BootstrapSpec(100)
    assert spec.resamples == 100
    assert spec.seed == DEFAULT_SEED == 2026
    assert MIN_RESAMPLES == 100
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.seed = 1  # type: ignore[misc]


def test_bootstrap_spec_bounds() -> None:
    message = "bootstrap resamples must be 100 to 100000"
    assert MAX_RESAMPLES == 100_000
    assert BootstrapSpec(100_000).resamples == 100_000
    for resamples in (99, 100_001):
        with pytest.raises(ValueError, match=r"\A" + re.escape(message) + r"\Z"):
            BootstrapSpec(resamples)


def test_interval_is_inclusive_percentile_interval() -> None:
    values = list(range(101))
    assert interval(values) == (2.5, 97.5)
    assert interval([0.5] * 100) == (0.5, 0.5)
    shuffled = list(values)
    random.Random(0).shuffle(shuffled)
    assert interval(shuffled) == (2.5, 97.5)


def test_draws_are_deterministic_and_complete() -> None:
    first = draw_weights(5, BootstrapSpec(100, 7))
    assert draw_weights(5, BootstrapSpec(100, 7)) == first
    assert len(first) == 100
    for drawn in first:
        assert sum(drawn.values()) == 5
        assert set(drawn) <= set(range(5))
    assert draw_weights(5, BootstrapSpec(100, 8)) != first


def test_group_order_first_seen_and_required() -> None:
    records = [rec("s1", "g2", ("O",), ("O",)), rec("s2", "g1", ("O",), ("O",)),
               rec("s3", "g2", ("O",), ("O",))]
    assert group_order(records) == ("g2", "g1")
    missing = [rec("s1", "g1", ("O",), ("O",)), rec("s2", None, ("O",), ("O",))]
    with pytest.raises(EvaluationInputError, match="'s2'"):
        group_order(missing)


# --- counts and resampled values ---------------------------------------------------------------

def test_group_counts_restricted_to_macro_classes(vocab: Vocab) -> None:
    spec = spec_named(blocks_for("A", vocab), "punctuation/all")
    assert spec.macro_over == ("COMMA", "PERIOD", "QUESTION")
    records = [rec("s1", "g1", ("O", "COMMA", "PERIOD"), ("O", "O", "PERIOD")),
               rec("s2", "g2", ("O", "QUESTION"), ("COMMA", "QUESTION"))]
    assert group_counts(records, spec, ("g1", "g2")) == (
        ((0, 1, 0), (1, 1, 0), (0, 1, 0)),
        ((0, 0, 1), (0, 0, 1), (1, 0, 1)))


def test_resampled_macro_f1_hand_values() -> None:
    counts = (((0, 1, 0), (1, 1, 0), (0, 1, 0)),
              ((0, 0, 1), (0, 0, 1), (1, 0, 1)))
    weights = [Counter({0: 2}), Counter({1: 2}), Counter({0: 1, 1: 1})]
    assert resampled_macro_f1(counts, weights) == pytest.approx([1 / 3, 1 / 3, 2 / 3])


@pytest.mark.parametrize("mode", ["A", "B", "C"])
def test_identity_resample_equals_point_estimate(vocab: Vocab, mode: str) -> None:
    records = records_for(mode)
    groups = group_order(records)
    points = []
    for spec in blocks_for(mode, vocab):
        expected = point_estimate(records, spec)
        counts = group_counts(records, spec, groups)
        for times in (1, 2):
            weights = Counter({index: times for index in range(len(groups))})
            assert resampled_macro_f1(counts, [weights]) == [expected], (spec.name, times)
        points.append(expected)
    assert max(points) > 0.0


# --- block intervals ---------------------------------------------------------------------------

def test_same_seed_same_ci_and_seed_matters(vocab: Vocab) -> None:
    records, specs = quality_records(), blocks_for("A", vocab)
    first = block_intervals(records, specs, BootstrapSpec(200, 1))
    assert block_intervals(records, specs, BootstrapSpec(200, 1)) == first
    bounds = set()
    for seed in range(1, 6):
        ci = block_intervals(records, specs, BootstrapSpec(200, seed))["punctuation/all"]
        bounds.add((ci["low"], ci["high"]))
    assert len(bounds) > 1


def test_identical_groups_give_point_ci(tmp_path: Path, vocab: Vocab) -> None:
    records, specs = identical_group_records(), blocks_for("A", vocab)
    intervals = block_intervals(records, specs, BootstrapSpec(100))
    for spec in specs:
        point = point_estimate(records, spec)
        assert 0.0 < point < 1.0
        assert intervals[spec.name]["low"] == intervals[spec.name]["high"] == point
    report = evaluate(write_bundle(tmp_path, records, "A", vocab), vocab, BootstrapSpec(100))
    for name, block in report["blocks"].items():
        ci = block["macro_f1_ci"]
        assert ci["low"] == ci["high"] == block["macro_f1"], name


def test_resampling_is_by_group_not_by_record(vocab: Vocab) -> None:
    spec = spec_named(blocks_for("A", vocab), "punctuation/all")
    same = ("COMMA", "PERIOD")
    records = [rec("good-1", "good", same, same), rec("good-2", "good", same, same),
               rec("good-3", "good", ("O", "QUESTION"), ("O", "QUESTION")),
               rec("good-4", "good", ("O", "QUESTION"), ("O", "QUESTION")),
               rec("bad-1", "bad", same, ("PERIOD", "COMMA"))]
    for seed in (0, 1):
        ci = block_intervals(records, [spec], BootstrapSpec(400, seed))[spec.name]
        assert (ci["low"], ci["high"], ci["groups"]) == (0.0, 1.0, 2), seed


def test_all_blocks_share_the_same_draws(tmp_path: Path, vocab: Vocab) -> None:
    records, specs = final_o_records(), blocks_for("A", vocab)
    cis = block_intervals(records, specs, BootstrapSpec(300, 4))
    assert cis["punctuation/all"] == cis["punctuation/non_final"]
    assert cis["punctuation/all"]["low"] < cis["punctuation/all"]["high"]
    report = evaluate(write_bundle(tmp_path, records, "A", vocab), vocab, BootstrapSpec(300, 4))
    blocks = report["blocks"]
    assert blocks["punctuation/all"]["macro_f1_ci"] == blocks["punctuation/non_final"]["macro_f1_ci"]


# --- report and Markdown integration -----------------------------------------------------------

def test_ci_dict_shape_and_bounds(tmp_path: Path, vocab: Vocab) -> None:
    bundle = write_bundle(tmp_path, joint_records(), "C", vocab)
    report = evaluate(bundle, vocab, BootstrapSpec(150, 9))
    assert len(report["blocks"]) == 6
    for name, block in report["blocks"].items():
        ci = block["macro_f1_ci"]
        assert set(ci) == CI_KEYS, name
        assert (ci["level"], ci["resamples"], ci["seed"], ci["unit"], ci["groups"]) == (
            0.95, 150, 9, "group_id", 3), name
        assert 0.0 <= ci["low"] <= ci["high"] <= 1.0, name
        assert ci["low"] == round(ci["low"], 6) and ci["high"] == round(ci["high"], 6), name
    assert all("macro_f1_ci" not in block for block in evaluate(bundle, vocab)["blocks"].values())


def test_missing_group_id_raises_naming_sample(tmp_path: Path, vocab: Vocab) -> None:
    records = [rec("s1", "g1", ("O", "PERIOD"), ("O", "PERIOD")),
               rec("s3", None, ("COMMA", "O"), ("COMMA", "O"))]
    bundle = write_bundle(tmp_path, records, "A", vocab)
    with pytest.raises(EvaluationInputError, match="'s3'"):
        evaluate(bundle, vocab, BootstrapSpec(100))
    assert "blocks" in evaluate(bundle, vocab)


def test_markdown_ci_column(tmp_path: Path, vocab: Vocab) -> None:
    bundle = write_bundle(tmp_path, quality_records(), "A", vocab)
    report = evaluate(bundle, vocab, BootstrapSpec(100))
    lines = to_markdown(report).splitlines()
    assert block_header(lines) == HEADER_CI
    for name, block in report["blocks"].items():
        ci = block["macro_f1_ci"]
        row = next(line for line in lines if line.startswith(f"| {name} |"))
        assert row.endswith(f" {pct(ci['low'])}{CI_SEPARATOR}{pct(ci['high'])} |"), name
    plain = to_markdown(evaluate(bundle, vocab))
    assert block_header(plain.splitlines()) == HEADER_PLAIN
    assert "95% CI" not in plain


# --- paired differences ------------------------------------------------------------------------

def test_paired_identical_predictions_give_zero(vocab: Vocab) -> None:
    records, specs = quality_records(), blocks_for("A", vocab)
    result = paired_differences(records, records, specs, BootstrapSpec(100, 3))
    assert set(result) == {spec.name for spec in specs}
    for name, entry in result.items():
        assert set(entry) == CI_KEYS | {"share_b_better"}, name
        assert (entry["low"], entry["high"], entry["share_b_better"]) == (0.0, 0.0, 0.0), name


def test_paired_b_better_in_every_group(vocab: Vocab) -> None:
    golds = (("COMMA", "PERIOD"), ("O", "QUESTION"), ("PERIOD", "O", "PERIOD"),
             ("COMMA", "COMMA", "QUESTION"), ("O", "PERIOD"))
    records_a = [rec(f"s{i}", f"g{i}", gold, ("O",) * len(gold)) for i, gold in enumerate(golds)]
    records_b = [rec(f"s{i}", f"g{i}", gold, gold) for i, gold in enumerate(golds)]
    result = paired_differences(records_a, records_b, blocks_for("A", vocab), BootstrapSpec(100, 3))
    entry: dict[str, Any] = result["punctuation/all"]
    assert entry["low"] > 0.0
    assert entry["share_b_better"] == 1.0


@pytest.mark.parametrize("change", ["reordered", "missing_one"])
def test_paired_requires_same_scored_ids(vocab: Vocab, change: str) -> None:
    records = quality_records()
    other = list(reversed(records)) if change == "reordered" else records[:-1]
    with pytest.raises(EvaluationInputError, match="same scored IDs"):
        paired_differences(records, other, blocks_for("A", vocab), BootstrapSpec(100))
