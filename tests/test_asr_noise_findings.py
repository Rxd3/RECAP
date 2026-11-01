"""Tests that every finding in docs/ASR_NOISE_V2.md follows from the data and flips when the data changes.

Each test bends a synthetic scores tree (see asr_noise_table_fixtures.py) so that one claim stops being true and
checks that the text no longer makes it. Numbers are worked out by hand from the fixture formulas.
"""

from __future__ import annotations

import copy
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from asr_noise_table_fixtures import (
    CLEAN_F1,
    MODELS,
    PRIMARY,
    SEEDS,
    WERS,
    patch_json,
    report,
    row,
    section,
    set_f1,
    v1_run,
    v2_run,
    write_asr_scores,
)

MINUS, ARROW, DASH = "−", "→", "—"  # U+2212 minus sign, U+2192 arrow, U+2014 em dash
WER_PCT = {"0.05": 5, "0.10": 10, "0.15": 15}
TARGET = {"substitution": 0.45, "deletion": 0.30, "insertion": 0.15, "swap": 0.10}


@pytest.fixture
def asr_scores(tmp_path: Path) -> Path:
    return write_asr_scores(tmp_path / "scores")


def _finding(md: str, number: int) -> str:
    lines = [line for line in md.splitlines() if line.startswith(f"{number}. **")]
    assert len(lines) == 1, number
    return lines[0]


def _set_v2_shares(root: Path, shares: dict[str, float], wers: tuple[str, ...] = WERS) -> None:
    for wer in wers:
        for seed in SEEDS:
            patch_json(v2_run(root, wer, seed) / "summary.json", cost_share=shares)


def _means_ordered(summary: dict[str, Any]) -> bool:
    """True if, in every condition, the seed means keep the order improved CRF > basic CRF > HMM."""
    v2, v1 = summary["v2"]["modes"], summary["v1"]["wer0.15"]["modes"]
    cells = [[v2[m]["models"][k]["by_wer"][w] for k in MODELS] for m in v2 for w in WERS]
    cells += [[v1[m]["models"][k] for k in MODELS] for m in v1]
    return all(c[0]["mean"] < c[1]["mean"] < c[2]["mean"] for c in cells)


def _break_order(run: Path, mode: str, *, tie: bool = False) -> None:
    """Put the improved CRF just below (or exactly at) the basic CRF in one run."""
    path = run / f"score_crf_basic_{mode}.json"
    basic = json.loads(path.read_text(encoding="utf-8"))["blocks"][PRIMARY[mode]]["macro_f1"]
    set_f1(run, mode, "crf_improved", basic if tie else round(basic - 0.0005, 6))


def test_findings_of_the_default_tree_follow_the_hand_computed_numbers(asr_scores: Path, tmp_path: Path) -> None:
    _, md = report(asr_scores, tmp_path)
    one, two, three = _finding(md, 1), _finding(md, 2), _finding(md, 3)

    assert one.startswith("1. **Does the improved CRF still degrade faster than the HMM?** Yes. At 15% WER under `v2`")
    assert (f"the improved CRF loses 4.48 points in mode A (50.00 {ARROW} 45.52), compared with 2.38 points for "
            f"the basic CRF (43.00 {ARROW} 40.62) and 0.58 points for the HMM (35.00 {ARROW} 34.42)") in one
    assert (f"The same pattern holds in mode B (improved CRF {MINUS}4.48 pts vs HMM {MINUS}0.58 pts) and mode C "
            f"(improved CRF {MINUS}4.48 pts vs HMM {MINUS}0.58 pts).") in one
    assert "A likely reason, not tested here" in one

    assert (f"Yes {DASH} `CRF improved > CRF basic > HMM` holds for every individual seed (2026, 2027, 2028) at "
            "every WER (5%, 10%, 15%) in all three modes (`A`, `B`, `C`), as well as across all `v1` 15% WER "
            "seeds.") in two
    assert ("Seed spread: across all 27 `(mode, model, WER)` cells in `v2`, the min–max spread over seeds "
            "2026, 2027, and 2028 is between 0.25 and 0.25 macro F1 points (and between 0.15 and 0.15 points "
            "for `v1` at 15% WER).") in two
    assert ("Under `v2`, the largest seed spread (HMM, mode A, 5% WER: 0.25 points) is smaller than the smallest "
            "gap between adjacent models (CRF improved vs CRF basic, mode B, 15% WER: 0.90 points); the ratio is "
            "0.28.") in two
    assert ("Under `v1`, the largest seed spread (HMM, mode A, 15% WER: 0.15 points) is smaller than the smallest "
            "gap between adjacent models (CRF improved vs CRF basic, mode B, 15% WER: 0.90 points); the ratio is "
            "0.17.") in two
    assert "order of magnitude smaller than the gaps between model families" not in two

    assert ("`v1` spends 50.00% of its error budget on adjacent-word swaps, 25.00% on function-word deletions, "
            "and 25.00% on 5-group homophone substitutions, with no insertions.") in three
    assert ("`v2` matches the target mix to within 0.50 points per edit type on all three WERs (45.00% "
            "substitution, 30.00% deletion, 15.00% insertion, and 10.00% swap at 15% WER).") in three
    assert (f"shifts mean macro F1 from 34.75 to 34.42 ({MINUS}0.33 pts) for the HMM, from 40.95 to 40.62 "
            f"({MINUS}0.33 pts) for CRF basic, and from 45.85 to 45.52 ({MINUS}0.33 pts) for CRF improved (and for "
            f"CRF improved, {MINUS}0.33 pts in mode B and {MINUS}0.33 pts in mode C)") in three
    # the v1 HMM lost 0.25 points: that is not "barely moved"
    assert (f"Under `v1` the HMM moved {MINUS}0.25 pts from clean in mode A; under `v2` it moved {MINUS}0.58 pts."
            in three)
    assert "barely moved" not in three


def test_the_hmm_is_said_to_barely_move_only_when_it_barely_moves(asr_scores: Path, tmp_path: Path) -> None:
    for seed in SEEDS:
        set_f1(v1_run(asr_scores, seed), "A", "hmm", 0.3505)  # +0.05 points from its clean 0.35
    _, md = report(asr_scores, tmp_path)
    assert (f"Under `v1` the HMM barely moved (+0.05 pts from clean in mode A), and 50.00% of `v1`'s budget was "
            "swaps of non-final words, which leave the utterance-final word in place; under `v2` the HMM also "
            f"loses ({MINUS}0.58 pts).") in _finding(md, 3)


@pytest.mark.parametrize(
    ("hmm_modes", "verdict"),
    [("ABC", "No."), ("A", "Only in mode B and mode C."), ("BC", "Only in mode A.")],
)
def test_finding_one_flips_when_the_hmm_falls_faster(
    hmm_modes: str, verdict: str, asr_scores: Path, tmp_path: Path
) -> None:
    for mode in hmm_modes:
        for seed in SEEDS:  # the HMM now falls 15 points at 15% WER
            set_f1(v2_run(asr_scores, "0.15", seed), mode, "hmm", round(CLEAN_F1[(mode, "hmm")] - 0.15, 6))
    _, md = report(asr_scores, tmp_path)
    one = _finding(md, 1)
    assert one.startswith(f"1. **Does the improved CRF still degrade faster than the HMM?** {verdict} At 15% WER")
    assert "A likely reason" not in one
    assert f"HMM {MINUS}15.00 pts" in one


def test_finding_one_says_gains_when_a_model_improves_under_noise(asr_scores: Path, tmp_path: Path) -> None:
    for seed in SEEDS:
        set_f1(v2_run(asr_scores, "0.15", seed), "A", "crf_improved", 0.52)  # clean is 0.50
    _, md = report(asr_scores, tmp_path)
    assert f"the improved CRF gains 2.00 points in mode A (50.00 {ARROW} 52.00)" in _finding(md, 1)


@pytest.mark.parametrize(
    ("edit_model", "mode", "wer", "seed"),
    [
        ("v2", "B", "0.10", 2027),  # not the first mode, not the first seed, not the extreme WER
        ("v2", "C", "0.05", 2028),
        ("v1", "C", "0.15", 2028),  # the v1 seeds count too
        ("v1", "A", "0.15", 2027),
        ("v1", "B", "0.15", 2026),
    ],
)
def test_ranking_is_not_stable_if_any_single_seed_crosses(
    edit_model: str, mode: str, wer: str, seed: int, asr_scores: Path, tmp_path: Path
) -> None:
    run = v2_run(asr_scores, wer, seed) if edit_model == "v2" else v1_run(asr_scores, seed)
    _break_order(run, mode)
    summary, md = report(asr_scores, tmp_path)
    assert _means_ordered(summary)  # the means still look fine: only the single seed crosses
    two = _finding(md, 2)
    assert f"Is the ranking stable across seeds?** No {DASH} `CRF improved > CRF basic > HMM`" in two
    assert f"breaks in 1 of 36 runs: `{edit_model}` mode {mode}, {WER_PCT[wer]}% WER, seed {seed}." in two
    assert f"Yes {DASH}" not in two


def test_a_tie_between_models_is_not_a_stable_ranking(asr_scores: Path, tmp_path: Path) -> None:
    _break_order(v2_run(asr_scores, "0.15", 2026), "A", tie=True)
    _, md = report(asr_scores, tmp_path)
    assert f"No {DASH}" in _finding(md, 2)


def test_the_ranking_order_is_read_from_the_data(tmp_path: Path) -> None:
    flipped = {key: CLEAN_F1[(key[0], {"hmm": "crf_improved", "crf_improved": "hmm"}.get(key[1], key[1]))]
               for key in CLEAN_F1}  # the HMM best, the improved CRF worst
    root = write_asr_scores(tmp_path / "scores", clean=flipped)
    _, md = report(root, tmp_path)
    assert f"Yes {DASH} `HMM > CRF basic > CRF improved` holds for every individual seed" in _finding(md, 2)


def test_a_seed_spread_larger_than_the_gap_is_reported_as_larger(asr_scores: Path, tmp_path: Path) -> None:
    for seed, value in zip(SEEDS, (0.465, 0.520, 0.490), strict=True):  # spread 5.50 points, still above the basic CRF
        set_f1(v2_run(asr_scores, "0.10", seed), "A", "crf_improved", value)
    _, md = report(asr_scores, tmp_path)
    assert ("Under `v2`, the largest seed spread (CRF improved, mode A, 10% WER: 5.50 points) is larger than the "
            "smallest gap between adjacent models (CRF improved vs CRF basic, mode B, 15% WER: 0.90 points); the "
            "ratio is 6.11.") in _finding(md, 2)


def test_a_tiny_seed_spread_is_an_order_of_magnitude_smaller(tmp_path: Path) -> None:
    root = write_asr_scores(tmp_path / "scores", offsets={2026: -0.00005, 2027: 0.0, 2028: 0.00005})
    _, md = report(root, tmp_path)
    assert ("Under `v2`, the largest seed spread (HMM, mode A, 5% WER: 0.01 points) is an order of magnitude smaller "
            "than the smallest gap between adjacent models (CRF improved vs CRF basic, mode B, 15% WER: 0.90 "
            "points); the ratio is 0.01.") in _finding(md, 2)


@pytest.mark.parametrize(
    ("ratio", "words"),
    [
        ("0.05", "an order of magnitude smaller than"),
        ("0.1", "an order of magnitude smaller than"),
        ("0.11", "smaller than"),
        ("0.49", "smaller than"),
        ("0.5", "comparable to"),
        ("0.91", "comparable to"),
        ("1", "comparable to"),
        ("1.05", "larger than"),
        (None, "larger than"),  # no positive gap at all
    ],
)
def test_describe_ratio_rule(ratio: str | None, words: str) -> None:
    from punct_restore.reporting import asr_noise_findings

    assert asr_noise_findings.describe_ratio(None if ratio is None else Decimal(ratio)) == words


def test_the_mix_finding_flips_when_the_mix_misses_the_target(asr_scores: Path, tmp_path: Path) -> None:
    _set_v2_shares(asr_scores, {"substitution": 0.0, "deletion": 0.0, "insertion": 0.986667, "swap": 0.013333})
    _, md = report(asr_scores, tmp_path)
    three = _finding(md, 3)
    assert "matches the target mix" not in three
    assert ("`v2` misses the target mix (tolerance 0.50 points per edit type) at 5%, 10%, 15% WER: at 5% WER it "
            "spends 0.00% on substitution, 0.00% on deletion, 98.67% on insertion, and 1.33% on swap, against "
            "targets of 45.00%, 30.00%, 15.00%, and 10.00%.") in three


def test_the_mix_finding_names_only_the_wers_that_miss(asr_scores: Path, tmp_path: Path) -> None:
    _set_v2_shares(asr_scores, {"substitution": 0.30, "deletion": 0.45, "insertion": 0.15, "swap": 0.10}, ("0.10",))
    _, md = report(asr_scores, tmp_path)
    assert ("`v2` misses the target mix (tolerance 0.50 points per edit type) at 10% WER, and matches it at "
            "5%, 15% WER: at 10% WER it spends 30.00% on substitution, 45.00% on deletion") in _finding(md, 3)


def test_the_mix_tolerance_is_one_constant(asr_scores: Path, tmp_path: Path) -> None:
    from punct_restore.reporting import asr_noise_findings

    tolerance = asr_noise_findings.MIX_TOLERANCE
    for deviation, verdict in ((tolerance / 2, "matches"), (tolerance * 2, "misses")):
        shares = {**TARGET, "substitution": round(0.45 + deviation, 6), "deletion": round(0.30 - deviation, 6)}
        _set_v2_shares(asr_scores, shares)
        out = tmp_path / verdict
        out.mkdir()
        _, md = report(asr_scores, out)
        assert f"`v2` {verdict} the target mix" in _finding(md, 3)


def test_one_seed_off_target_misses_even_when_the_mean_is_within_tolerance(
    asr_scores: Path, tmp_path: Path
) -> None:
    # Seeds 2026/2027 sit on target; 2028 spends 0.462 on substitution (1.2 points off), so the mean
    # (0.454) is within the 0.50-point tolerance but one seed is not: the rule is per seed, not on the mean.
    for seed in SEEDS:
        off = seed == SEEDS[-1]
        shares = {**TARGET, "substitution": 0.462 if off else 0.45, "deletion": 0.288 if off else 0.30}
        patch_json(v2_run(asr_scores, WERS[0], seed) / "summary.json", cost_share=shares)
    _, md = report(asr_scores, tmp_path)
    assert "`v2` misses the target mix" in _finding(md, 3)


def test_a_small_v1_insertion_share_on_one_seed_is_not_called_insertion_free(
    asr_scores: Path, tmp_path: Path
) -> None:
    path = v1_run(asr_scores, SEEDS[0]) / "summary.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    patch_json(path, edits={**document["edits"], "insertion": 1},
               cost_share={**document["cost_share"], "insertion": 0.004})
    _, md = report(asr_scores, tmp_path)
    assert "with no insertions" not in _finding(md, 3)


def test_the_target_mix_comes_from_the_simulator_weights(asr_scores: Path, tmp_path: Path) -> None:
    from punct_restore.data.asr_noise import V2_WEIGHTS

    summary, _ = report(asr_scores, tmp_path)
    total = sum(V2_WEIGHTS.values())
    assert summary["v2"]["target_cost_share"] == {
        "substitution": V2_WEIGHTS["replace"] / total, "deletion": V2_WEIGHTS["delete"] / total,
        "insertion": V2_WEIGHTS["insert"] / total, "swap": V2_WEIGHTS["swap"] / total}


def test_a_v1_summary_with_insertions_is_not_called_insertion_free(asr_scores: Path, tmp_path: Path) -> None:
    for seed in SEEDS:
        path = v1_run(asr_scores, seed) / "summary.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        patch_json(path, edits={**document["edits"], "insertion": 10},
                   cost_share={**document["cost_share"], "insertion": 0.1})
    _, md = report(asr_scores, tmp_path)
    three = _finding(md, 3)
    assert "with no insertions" not in three
    assert "and 10.00% on insertions." in three


def test_counts_and_labels_in_the_report_come_from_the_summary(asr_scores: Path, tmp_path: Path) -> None:
    from punct_restore.reporting import asr_noise_table

    summary, _ = report(asr_scores, tmp_path)
    fewer = {**summary, "wers": ["0.05", "0.15"]}
    md = asr_noise_table.render_markdown(fewer)
    assert "across all 18 `(mode, model, WER)` cells" in md and "across all 27" not in md
    assert f"| Model | Clean (0%) | WER 5% | WER 15% | Drop (0% {ARROW} 15%) |" in md
    assert "WER 10%" not in md and "at every WER (5%, 15%)" in _finding(md, 2)

    other = copy.deepcopy(summary)
    other["v1"]["wer0.15"]["achieved"]["wer"] = 0.2
    other_md = asr_noise_table.render_markdown(other)
    assert row(section(other_md, "## 2. Achieved Edit Mix (`v2` and `v1`)"), "`v1`")[1] == "20.00%"


def test_render_markdown_does_not_change_the_summary(asr_scores: Path, tmp_path: Path) -> None:
    from punct_restore.reporting import asr_noise_table

    summary: dict[str, Any]
    summary, _ = report(asr_scores, tmp_path)
    before = copy.deepcopy(summary)
    asr_noise_table.render_markdown(summary)
    assert summary == before
