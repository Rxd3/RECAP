"""The Markdown of docs/ASR_NOISE_V2.md, rendered strictly from the summary dict (no file is read here).

Sections 1 to 4 and 6 are in this module; section 5 (the findings) is computed in asr_noise_findings.py.
Numbers about the simulator that the text repeats (counts, the swap cap, the insertion words) are imported from
``punct_restore.data.asr_noise`` so the report cannot drift from it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from punct_restore.data.asr_noise import INSERT_WORDS, MIN_TRAIN_COUNT, SWAP_CAP_SHARE
from punct_restore.reporting.asr_noise_findings import findings_section
from punct_restore.reporting.asr_noise_stats import (
    ARROW,
    EN,
    MINUS,
    MODEL_LABELS,
    V1_KEY,
    V1_WER,
    cell_mean,
    count_word,
    dec,
    join_and,
    signed_pts,
    wer_label,
)
from punct_restore.reporting.build_tables import half_up, pct

Summary = Mapping[str, Any]


def _method_intro(summary: Summary) -> list[str]:
    seeds = ", ".join(str(s) for s in summary["seeds"])
    return [
        "# Simulated ASR Noise Robustness (`v2` Edit Model and Seed Variance)",
        "",
        "## 1. Method",
        "",
        (
            "This study extends RECAP's exploratory simulated-ASR robustness check (`docs/EVALUATION_RESULTS_CRF.md` "
            "§13) with a substitution-dominant edit model (`--edit-model v2` in `scripts/simulate_asr_noise.py`, "
            "implemented in `src/punct_restore/data/asr_noise.py`) and evaluates seed variance across "
            f"{count_word(len(summary['seeds']))} random seeds ({seeds}). The original `v1` edit model remains the "
            "default (`--edit-model v1`) and produces byte-identical outputs."
        ),
        "",
    ]


def _method_edits(summary: Summary) -> list[str]:
    words = ", ".join(f"`{word}`" for word in INSERT_WORDS)
    cap = half_up(SWAP_CAP_SHARE * 100, 0)
    return [
        "### Edit operations in `v2`",
        "",
        (
            "- **Confusion substitution (cost 1):** replaces a token with a different word at character Levenshtein "
            "edit distance 1 that shares the same first letter. Candidate words come exclusively from the "
            f"`{summary['v2']['vocabulary_source']}` split vocabulary with a minimum count of {MIN_TRAIN_COUNT} "
            "(never from `validation` or `test`). The token keeps its punctuation and casing labels. Homophone "
            "substitution (`their/there/they're`, `your/you're`, `its/it's`, `to/too/two`, `then/than`) stays "
            "available as a second substitution source."
        ),
        (
            "- **Deletion (cost 1):** deletes a function word or any word of 3 characters or fewer, never the last "
            "remaining token of an utterance. As in `v1`, a deleted token's punctuation label moves to the previous "
            "surviving token if that token has `O`, and is dropped otherwise."
        ),
        (
            f"- **Insertion (cost 1):** inserts a filler or short function word ({words}) before a token. The "
            "inserted token receives punctuation `O`, casing `LOWER`, mode C casing mask `0` (included in "
            "`excluded_positions` so its casing is not scored), and an empty character offset span `[pos, pos]` at "
            "the start offset of the following token."
        ),
        (
            f"- **Swap (cost 2):** swaps two adjacent non-final tokens as in `v1`, capped at {cap}% of the error "
            f"budget (`2 * swaps <= int({SWAP_CAP_SHARE:.2f} * budget)`)."
        ),
        "",
    ]


def _method_protocol(summary: Summary) -> list[str]:
    target = summary["v2"]["target_cost_share"]
    seeds = ", ".join(str(s) for s in summary["seeds"])
    tokens = summary["v2"]["achieved_by_wer"][summary["wers"][0]]["tokens_before"]
    mix = join_and([f"{pct(share)}% {key}" for key, share in target.items()])
    wers = join_and([wer_label(w) for w in summary["wers"]])
    modes = join_and([f"`{m}`" for m in summary["modes"]])
    models = ", ".join(f"`{MODEL_LABELS[m]}`" for m in summary["models"])
    return [
        (
            f"**Target cost mix and budget:** target cost shares are {mix}. As in `v1`, the error budget is "
            f"`round(wer * tokens_before)` across the {half_up(tokens, 0, grouping=True)} primary test tokens, and "
            "edits are drawn with a single seeded `random.Random(seed)` so identical seeds yield identical outputs."
        ),
        "",
        (
            f"**Evaluation protocol:** each frozen model ({models}) in modes {modes} predicted each noisy primary "
            f"`test` set once without retuning (`v2` at WER {wers} across seeds {seeds}; `v1` at WER "
            f"{wer_label(V1_WER)} across seeds {seeds}, reusing `results/scores/asr_noise/{V1_KEY}/` for seed "
            f"{summary['seeds'][0]}). Every run was scored with `scripts/evaluate_models.py score --bootstrap "
            "1000`, using the remapped mode C casing masks."
        ),
        "",
    ]


def _mix_row(label: str, wer: str, info: Mapping[str, Any]) -> str:
    shares = info["cost_share"]
    return (
        f"| {label} | {wer}% | {info['achieved_wer_pct']} | {half_up(info['budget'], 0, grouping=True)} | "
        f"{half_up(info['tokens_before'], 0, grouping=True)} | "
        f"{half_up(info['tokens_after_mean'], 1, grouping=True)} | {shares['substitution']['cell']} | "
        f"{shares['deletion']['cell']} | {shares['insertion']['cell']} | {shares['swap']['cell']} |"
    )


def _sec_achieved_mix(summary: Summary) -> list[str]:
    target = summary["v2"]["target_cost_share"]
    seeds = join_and([str(s) for s in summary["seeds"]])
    lines = [
        "## 2. Achieved Edit Mix (`v2` and `v1`)",
        "",
        (
            f"Cost share % per edit type across seeds {seeds} (mean and min{EN}max range across seeds; "
            "swaps have cost 2, all other edits cost 1):"
        ),
        "",
        (
            "| Edit model | Target WER | Achieved WER % | Budget | Tokens before | Tokens after (mean) "
            "| Substitution % | Deletion % | Insertion % | Swap % |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
        (
            f"| `v2` target | — | — | — | — | — | {pct(target['substitution'])} | {pct(target['deletion'])} | "
            f"{pct(target['insertion'])} | {pct(target['swap'])} |"
        ),
    ]
    for wer in summary["wers"]:
        info = summary["v2"]["achieved_by_wer"][wer]
        lines.append(_mix_row("`v2`", pct(float(wer)), info))
    v1_info = summary["v1"][V1_KEY]["achieved"]
    lines += [_mix_row("`v1`", pct(v1_info["wer"]), v1_info), ""]
    return lines


def _mode_table(summary: Summary, mode: str) -> list[str]:
    wers = summary["wers"]
    top = wers[-1]
    columns = "".join(f" WER {wer_label(w)} |" for w in wers)
    lines = [
        f"### Mode {mode} (`{summary['primary_blocks'][mode]}`)",
        "",
        f"| Model | Clean (0%) |{columns} Drop (0% {ARROW} {wer_label(top)}) |",
        "|---" * (len(wers) + 3) + "|",
    ]
    for model in summary["models"]:
        entry = summary["v2"]["modes"][mode]["models"][model]
        cells = " | ".join(entry["by_wer"][w]["cell"] for w in wers)
        change = signed_pts(dec(-entry["by_wer"][top]["drop_from_clean"]))
        lines.append(f"| {MODEL_LABELS[model]} | {entry['clean']['macro_f1_pct']} | {cells} | {change} |")
    return lines + [""]


def _sec_mode_tables(summary: Summary) -> list[str]:
    seeds = join_and([str(s) for s in summary["seeds"]])
    lines = [
        "## 3. Primary Macro F1 by Mode (`v2`)",
        "",
        (
            "Primary macro F1 % on the primary `test` lists: clean (0% WER) alongside the mean over seeds "
            f"{seeds} with the min{EN}max seed range in parentheses, plus the mean drop from 0% to "
            f"{wer_label(summary['wers'][-1])} WER in percentage points."
        ),
        "",
    ]
    for mode in summary["modes"]:
        lines += _mode_table(summary, mode)
    return lines


def _sec_v1_vs_v2(summary: Summary) -> list[str]:
    label = wer_label(V1_WER)
    seeds = join_and([str(s) for s in summary["seeds"]])
    lines = [
        f"## 4. Comparison: `v1` vs `v2` at {label} WER",
        "",
        (
            f"Primary macro F1 % at {label} WER averaged across seeds {seeds} (mean and min{EN}max range) for "
            f"both edit models, with the mean difference `v2 {MINUS} v1` in percentage points:"
        ),
        "",
        (
            f"| Mode | Model | Clean (0%) | `v1` {label} WER | `v2` {label} WER | `v2` {MINUS} `v1` (pts) "
            "| `v1` seed range | `v2` seed range |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for mode in summary["modes"]:
        for model in summary["models"]:
            v1 = summary["v1"][V1_KEY]["modes"][mode]["models"][model]
            v2 = summary["v2"]["modes"][mode]["models"][model]["by_wer"][V1_WER]
            lines.append(
                f"| {mode} | {MODEL_LABELS[model]} | {summary['clean'][mode][model]['macro_f1_pct']} | {v1['cell']} | "
                f"{v2['cell']} | {signed_pts(cell_mean(v2) - cell_mean(v1))} | {v1['range_pct']} | {v2['range_pct']} |"
            )
    return lines + [""]


def _sec_limits(summary: Summary) -> list[str]:
    words = ", ".join(f"`{word}`" for word in INSERT_WORDS)
    return [
        "## 6. Limits",
        "",
        (
            "- **Synthetic token-level edits, not acoustic ASR:** `v2` corrupts clean reference transcripts "
            "uniformly at random among eligible positions rather than decoding audio through an acoustic and "
            "language model, where errors cluster around fast speech, rare names, and acoustic confusions."
        ),
        (
            "- **Orthographic proxy for confusion substitutions:** confusion candidates use character Levenshtein "
            f"distance 1 with the same initial letter from `default_train` (count >= {MIN_TRAIN_COUNT}) rather than "
            "phonetic or subword acoustic confusion matrices."
        ),
        (
            f"- **Fixed insertion list and neutral labels:** insertions draw from {len(INSERT_WORDS)} common fillers "
            f"and short function words ({words}) and always receive `O` punctuation and `LOWER` casing (masked in "
            "mode C), so false sentence boundaries induced by multi-word ASR hallucinations are not tested."
        ),
        (
            "- **Single-token edits only:** multi-token phrase substitutions, word merges/splits (e.g., compound "
            "words), and timing/prosody cues remain outside the scope of this text-only check."
        ),
        "",
        (
            "Files: `results/scores/asr_noise_v2/summary.json`, "
            "`results/scores/asr_noise_v2/v2/wer<rate>/seed<seed>/`, and "
            f"`results/scores/asr_noise_v2/v1/{V1_KEY}/seed<seed>/`."
        ),
        "",
    ]


def render_markdown(summary: Summary) -> str:
    """Render the complete docs/ASR_NOISE_V2.md text strictly from ``summary``."""
    lines = (
        _method_intro(summary)
        + _method_edits(summary)
        + _method_protocol(summary)
        + _sec_achieved_mix(summary)
        + _sec_mode_tables(summary)
        + _sec_v1_vs_v2(summary)
        + findings_section(summary)
        + _sec_limits(summary)
    )
    return "\n".join(lines)
