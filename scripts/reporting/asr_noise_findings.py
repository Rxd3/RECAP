"""Section 5 (Findings) of the ASR noise report: every verdict and every number is computed from the summary dict.

Rules, so that made-up scores flip the text:
- *Degrades faster:* the improved CRF's mean drop from clean at the highest WER under ``v2`` is larger than the
  HMM's, per mode. Yes if in every mode, No if in none, otherwise the modes where it is.
- *Ranking:* the observed order is the models sorted by their mean macro F1 over every noisy run (``v2`` and
  ``v1``, all modes, WERs and seeds). It is stable only if, in every single run (edit model, mode, WER, seed),
  every model scores strictly above the next one in that order. A tie is a break.
- *Seed spread vs gap:* per edit model, the largest seed spread (max - min over seeds, one model, mode and WER)
  is compared with the smallest gap between the seed means of models adjacent in the observed order (same mode
  and WER). Ratio = spread / gap, worded by ``describe_ratio``. Ties go to the first cell in mode, WER, model order.
- *Mix:* ``v2`` matches the target mix at a WER if the min and the max over seeds of every edit type's cost share
  are within ``MIX_TOLERANCE`` (absolute, as a share) of the target share.
- *Barely moved:* the HMM's mean moves from clean by at most ``BARELY_MOVED`` (0.10 points).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from itertools import pairwise
from typing import Any

from asr_noise_stats import (
    ARROW,
    MODEL_LABELS,
    V1_KEY,
    V1_WER,
    cell_mean,
    count_word,
    dec,
    join_and,
    pts,
    signed_pts,
    wer_label,
)
from build_tables import half_up, pct

from punct_restore.data.asr_noise import HOMOPHONE_GROUPS

EM = "—"  # em dash
EDIT_MODELS = ("v2", "v1")
MIX_TOLERANCE = 0.005  # absolute cost-share tolerance (0.50 percentage points) for "matches the target mix"
BARELY_MOVED = Decimal("0.001")  # 0.10 points
ORDER_OF_MAGNITUDE = Decimal("0.1")  # ratio at or below: "an order of magnitude smaller than"
COMPARABLE_FROM = Decimal("0.5")  # ratio from here up to 1: "comparable to"
SHOWN_BREAKS = 3
MODEL_FOR = {"hmm": "the HMM", "crf_basic": "CRF basic", "crf_improved": "CRF improved"}


@dataclass(frozen=True)
class SpreadVsGap:
    """The largest seed spread and the smallest gap between adjacent models, with where each was found."""

    spread: Decimal
    spread_at: str
    gap: Decimal
    gap_at: str

    @property
    def ratio(self) -> Decimal | None:
        """Spread divided by gap, or ``None`` if there is no positive gap."""
        return self.spread / self.gap if self.gap > 0 else None


def describe_ratio(ratio: Decimal | None) -> str:
    """How a spread compares with a gap: at most 0.1 of it, under half, up to equal, or above it."""
    if ratio is None or ratio > 1:
        return "larger than"
    if ratio <= ORDER_OF_MAGNITUDE:
        return "an order of magnitude smaller than"
    if ratio < COMPARABLE_FROM:
        return "smaller than"
    return "comparable to"


def _cell(summary: Mapping[str, Any], edit_model: str, mode: str, model: str, wer: str) -> Mapping[str, Any]:
    if edit_model == "v2":
        return summary["v2"]["modes"][mode]["models"][model]["by_wer"][wer]
    return summary["v1"][V1_KEY]["modes"][mode]["models"][model]


def _conditions(summary: Mapping[str, Any], edit_model: str) -> list[tuple[str, str]]:
    """The (mode, WER) pairs one edit model was run at: every WER for ``v2``, the one WER for ``v1``."""
    wers = summary["wers"] if edit_model == "v2" else [V1_WER]
    return [(mode, wer) for mode in summary["modes"] for wer in wers]


def observed_order(summary: Mapping[str, Any]) -> list[str]:
    """The models from best to worst by mean macro F1 over every noisy run, in the summary's model order on a tie."""

    def total(model: str) -> Decimal:
        scores = (
            dec(score)
            for em in EDIT_MODELS
            for mode, wer in _conditions(summary, em)
            for score in _cell(summary, em, mode, model, wer)["per_seed"].values()
        )
        return sum(scores, Decimal(0))

    return sorted(summary["models"], key=total, reverse=True)


def ranking_breaks(summary: Mapping[str, Any], order: list[str]) -> tuple[int, list[str]]:
    """(number of runs, the runs where ``order`` is not kept strictly); a run is an edit model, mode, WER and seed."""
    runs, broken = 0, []
    for em in EDIT_MODELS:
        for mode, wer in _conditions(summary, em):
            per_seed = [_cell(summary, em, mode, model, wer)["per_seed"] for model in order]
            for seed in summary["seeds"]:
                runs += 1
                scores = [dec(scores_of[str(seed)]) for scores_of in per_seed]
                if not all(higher > lower for higher, lower in pairwise(scores)):
                    broken.append(f"`{em}` mode {mode}, {wer_label(wer)} WER, seed {seed}")
    return runs, broken


def _spread(cell: Mapping[str, Any]) -> Decimal:
    scores = [dec(score) for score in cell["per_seed"].values()]
    return max(scores) - min(scores)


def spread_vs_gap(summary: Mapping[str, Any], edit_model: str, order: list[str]) -> SpreadVsGap:
    """The largest seed spread and the smallest adjacent-model gap of one edit model (see the module rules)."""
    conditions = _conditions(summary, edit_model)
    spreads, gaps = [], []
    for mode, wer in conditions:
        cells = {m: _cell(summary, edit_model, mode, m, wer) for m in summary["models"]}
        spreads += [(_spread(cells[m]), f"{MODEL_LABELS[m]}, mode {mode}, {wer_label(wer)} WER") for m in cells]
        for high, low in pairwise(order):
            gap = cell_mean(cells[high]) - cell_mean(cells[low])
            gaps.append((gap, f"{MODEL_LABELS[high]} vs {MODEL_LABELS[low]}, mode {mode}, {wer_label(wer)} WER"))
    spread, spread_at = max(spreads, key=lambda item: item[0])
    gap, gap_at = min(gaps, key=lambda item: item[0])
    return SpreadVsGap(spread, spread_at, gap, gap_at)


def mix_misses(summary: Mapping[str, Any]) -> list[str]:
    """The ``v2`` WERs whose cost shares, for any seed, are further than ``MIX_TOLERANCE`` from the target."""
    target = summary["v2"]["target_cost_share"]
    tolerance = dec(MIX_TOLERANCE)
    return [
        wer
        for wer in summary["wers"]
        if any(
            abs(dec(summary["v2"]["achieved_by_wer"][wer]["cost_share"][key][edge]) - dec(share)) > tolerance
            for key, share in target.items()
            for edge in ("min", "max")
        )
    ]


def _points(value: Decimal) -> str:
    return signed_pts(value) if value < 0 else pts(value)


def _ratio_clause(comparison: SpreadVsGap) -> str:
    ratio = comparison.ratio
    return f"the ratio is {half_up(ratio, 2)}" if ratio is not None else "the gap is not positive, so there is no ratio"


def _spread_sentence(edit_model: str, comparison: SpreadVsGap) -> str:
    return (
        f"Under `{edit_model}`, the largest seed spread ({comparison.spread_at}: {_points(comparison.spread)} "
        f"points) is {describe_ratio(comparison.ratio)} the smallest gap between adjacent models "
        f"({comparison.gap_at}: {_points(comparison.gap)} points); {_ratio_clause(comparison)}."
    )


def _ranking_sentence(summary: Mapping[str, Any], order: list[str]) -> str:
    names = " > ".join(MODEL_LABELS[model] for model in order)
    runs, broken = ranking_breaks(summary, order)
    if not broken:
        modes = ", ".join(f"`{m}`" for m in summary["modes"])
        return (
            f"Yes {EM} `{names}` holds for every individual seed ({', '.join(map(str, summary['seeds']))}) at every "
            f"WER ({', '.join(wer_label(w) for w in summary['wers'])}) in all {count_word(len(summary['modes']))} "
            f"modes ({modes}), as well as across all `v1` {wer_label(V1_WER)} WER seeds."
        )
    more = f"; and {len(broken) - SHOWN_BREAKS} more" if len(broken) > SHOWN_BREAKS else ""
    return (
        f"No {EM} `{names}` (the order of the mean scores over all noisy runs) breaks in {len(broken)} of {runs} "
        f"runs: {'; '.join(broken[:SHOWN_BREAKS])}{more}."
    )


def _spread_bounds(summary: Mapping[str, Any], edit_model: str) -> tuple[str, str]:
    spreads = [
        _spread(_cell(summary, edit_model, mode, model, wer))
        for mode, wer in _conditions(summary, edit_model)
        for model in summary["models"]
    ]
    return pts(min(spreads)), pts(max(spreads))


def _finding_ranking(summary: Mapping[str, Any]) -> str:
    order = observed_order(summary)
    cells = len(summary["modes"]) * len(summary["models"]) * len(summary["wers"])
    (v2_low, v2_high), (v1_low, v1_high) = _spread_bounds(summary, "v2"), _spread_bounds(summary, "v1")
    comparisons = " ".join(_spread_sentence(em, spread_vs_gap(summary, em, order)) for em in EDIT_MODELS)
    return (
        f"2. **Is the ranking stable across seeds?** {_ranking_sentence(summary, order)} Seed spread: across all "
        f"{cells} `(mode, model, WER)` cells in `v2`, the min–max spread over seeds "
        f"{join_and([str(s) for s in summary['seeds']])} is between {v2_low} and {v2_high} macro F1 points (and "
        f"between {v1_low} and {v1_high} points for `v1` at {wer_label(V1_WER)} WER). {comparisons}"
    )


def _change(cell: Mapping[str, Any]) -> str:
    """The signed change from clean in points: a loss is negative."""
    return signed_pts(dec(-cell["drop_from_clean"]))


def _span(summary: Mapping[str, Any], mode: str, model: str, cell: Mapping[str, Any]) -> str:
    return f"({summary['clean'][mode][model]['macro_f1_pct']} {ARROW} {cell['mean_pct']})"


def _amount(cell: Mapping[str, Any]) -> str:
    """The drop from clean as a size: ``4.48 points``, or ``a gain of 2.00 points`` if the score rose."""
    drop = dec(cell["drop_from_clean"])
    return f"{pts(drop)} points" if drop >= 0 else f"a gain of {pts(-drop)} points"


def _loses(cell: Mapping[str, Any]) -> str:
    """The drop from clean as a verb phrase: ``loses 4.48 points`` or ``gains 2.00 points``."""
    drop = dec(cell["drop_from_clean"])
    return f"loses {pts(drop)} points" if drop >= 0 else f"gains {pts(-drop)} points"


def _degradation_verdict(faster: list[str], modes: list[str]) -> str:
    """Yes if the improved CRF degrades faster than the HMM in every mode, No if in none, else which modes."""
    if len(faster) == len(modes):
        return "Yes."
    if not faster:
        return "No."
    return f"Only in {join_and([f'mode {m}' for m in faster])}."


def _degradation_modes(faster: list[str], cells: Mapping[str, Mapping[str, Any]], modes: list[str]) -> str:
    """The sentence after the mode-A numbers, and the untested explanation only when the pattern holds everywhere."""
    parts = [
        f"mode {m} (improved CRF {_change(cells[m]['crf_improved'])} pts vs HMM {_change(cells[m]['hmm'])} pts)"
        for m in modes
    ]
    if len(faster) != len(modes):
        return f" By mode: {join_and(parts)}."
    same = f" The same pattern holds in {join_and(parts[1:])}." if len(modes) > 1 else ""
    return same + (
        " A likely reason, not tested here, is that the improved CRF's word-pair and window features depend on "
        "the exact neighbouring words, so each edit changes more of its active features."
    )


def _finding_degradation(summary: Mapping[str, Any]) -> str:
    modes, top = summary["modes"], summary["wers"][-1]
    cells = {m: {k: _cell(summary, "v2", m, k, top) for k in summary["models"]} for m in modes}
    faster = [m for m in modes if cells[m]["crf_improved"]["drop_from_clean"] > cells[m]["hmm"]["drop_from_clean"]]
    first = modes[0]
    imp, bas, hmm = (cells[first][k] for k in ("crf_improved", "crf_basic", "hmm"))
    return (
        f"1. **Does the improved CRF still degrade faster than the HMM?** {_degradation_verdict(faster, modes)} "
        f"At {wer_label(top)} WER under `v2`, the improved CRF {_loses(imp)} in mode {first} "
        f"{_span(summary, first, 'crf_improved', imp)}, compared with {_amount(bas)} for the basic CRF "
        f"{_span(summary, first, 'crf_basic', bas)} and {_amount(hmm)} for the HMM "
        f"{_span(summary, first, 'hmm', hmm)}.{_degradation_modes(faster, cells, modes)}"
    )


def _shares_text(summary: Mapping[str, Any], wer: str, template: str) -> str:
    """The mean cost share of every edit type at ``wer``, each formatted with ``template`` (``{pct}``, ``{key}``)."""
    shares = summary["v2"]["achieved_by_wer"][wer]["cost_share"]
    keys = summary["v2"]["target_cost_share"]
    return join_and([template.format(pct=shares[key]["mean_pct"], key=key) for key in keys])


def _mix_sentence(summary: Mapping[str, Any]) -> str:
    wers, missed = summary["wers"], mix_misses(summary)
    tolerance = pct(MIX_TOLERANCE)
    if not missed:
        return (
            f"By contrast, `v2` matches the target mix to within {tolerance} points per edit type on all "
            f"{count_word(len(wers))} WERs ({_shares_text(summary, wers[-1], '{pct}% {key}')} at "
            f"{wer_label(wers[-1])} WER)."
        )
    targets = join_and([f"{pct(share)}%" for share in summary["v2"]["target_cost_share"].values()])
    kept = [w for w in wers if w not in missed]
    matched = f", and matches it at {', '.join(wer_label(w) for w in kept)} WER" if kept else ""
    return (
        f"By contrast, `v2` misses the target mix (tolerance {tolerance} points per edit type) at "
        f"{', '.join(wer_label(w) for w in missed)} WER{matched}: at {wer_label(missed[0])} WER it spends "
        f"{_shares_text(summary, missed[0], '{pct}% on {key}')}, against targets of {targets}."
    )


def _v1_composition(summary: Mapping[str, Any]) -> str:
    shares = summary["v1"][V1_KEY]["achieved"]["cost_share"]
    items = [
        f"{shares['swap']['mean_pct']}% of its error budget on adjacent-word swaps",
        f"{shares['deletion']['mean_pct']}% on function-word deletions",
        f"{shares['substitution']['mean_pct']}% on {len(HOMOPHONE_GROUPS)}-group homophone substitutions",
    ]
    has_insertions = dec(shares["insertion"]["max"]) > 0  # any seed with an insertion: never "insertion-free"
    if has_insertions:
        items.append(f"{shares['insertion']['mean_pct']}% on insertions")
    return (
        f"In edit composition at {wer_label(V1_WER)} WER, `v1` spends {join_and(items)}"
        f"{'' if has_insertions else ', with no insertions'}."
    )


def _shift_sentence(summary: Mapping[str, Any]) -> str:
    modes, top = summary["modes"], V1_WER
    swap_share = summary["v1"][V1_KEY]["achieved"]["cost_share"]["swap"]["mean_pct"]

    def shift(mode: str, model: str) -> tuple[Mapping[str, Any], Mapping[str, Any], str]:
        v1, v2 = _cell(summary, "v1", mode, model, top), _cell(summary, "v2", mode, model, top)
        return v1, v2, signed_pts(cell_mean(v2) - cell_mean(v1))

    parts = []
    for model in summary["models"]:
        v1, v2, diff = shift(modes[0], model)
        parts.append(f"from {v1['mean_pct']} to {v2['mean_pct']} ({diff} pts) for {MODEL_FOR[model]}")
    others = [f"{shift(m, 'crf_improved')[2]} pts in mode {m}" for m in modes[1:]]
    tail = f" (and for CRF improved, {join_and(others)})" if others else ""
    return (
        f"In mode {modes[0]} at {wer_label(top)} WER, replacing `v1`'s {swap_share}% swap share with confusion "
        "substitutions, short-word deletions, and filler/function-word insertions shifts mean macro F1 "
        f"{join_and(parts)}{tail}."
    )


def _hmm_sentence(summary: Mapping[str, Any]) -> str:
    mode = summary["modes"][0]
    clean = dec(summary["clean"][mode]["hmm"]["macro_f1"])
    v1_move = cell_mean(_cell(summary, "v1", mode, "hmm", V1_WER)) - clean
    v2_move = cell_mean(_cell(summary, "v2", mode, "hmm", V1_WER)) - clean
    swap_share = summary["v1"][V1_KEY]["achieved"]["cost_share"]["swap"]["mean_pct"]
    if abs(v1_move) <= BARELY_MOVED and v2_move < 0:
        return (
            f"Under `v1` the HMM barely moved ({signed_pts(v1_move)} pts from clean in mode {mode}), and "
            f"{swap_share}% of `v1`'s budget was swaps of non-final words, which leave the utterance-final word "
            f"in place; under `v2` the HMM also loses ({signed_pts(v2_move)} pts)."
        )
    return (
        f"Under `v1` the HMM moved {signed_pts(v1_move)} pts from clean in mode {mode}; under `v2` it moved "
        f"{signed_pts(v2_move)} pts."
    )


def _finding_versions(summary: Mapping[str, Any]) -> str:
    return " ".join([
        "3. **How do `v1` and `v2` differ?**",
        _v1_composition(summary),
        _mix_sentence(summary),
        _shift_sentence(summary),
        _hmm_sentence(summary),
        "Which `v2` edit type causes which loss was not measured separately.",
    ])


def findings_section(summary: Mapping[str, Any]) -> list[str]:
    """The Markdown lines of section 5, computed from ``summary`` only."""
    return [
        "## 5. Findings",
        "",
        _finding_degradation(summary),
        _finding_ranking(summary),
        _finding_versions(summary),
        "",
    ]
