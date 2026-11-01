"""A synthetic ``results/scores`` tree for the ASR noise report tests (no real data), and helpers to bend it.

``write_asr_scores`` writes the clean, ``v2`` and ``v1`` evaluator and simulator JSON files that
``punct_restore.reporting.asr_noise_table`` reads. Every number follows a formula (see the module constants), so a
test can work out the expected table cell by hand. ``set_f1`` and ``patch_json`` change single files afterwards.
Not a test module: pytest does not collect it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reporting_fixtures import dump

MODES = ("A", "B", "C")
PRIMARY = {"A": "punctuation/all", "B": "punctuation/all", "C": "joint/unmasked"}
WERS = ("0.05", "0.10", "0.15")
SEEDS = (2026, 2027, 2028)
MODELS = ("hmm", "crf_basic", "crf_improved")
CLEAN_F1 = {
    ("A", "hmm"): 0.35, ("A", "crf_basic"): 0.43, ("A", "crf_improved"): 0.50,
    ("B", "hmm"): 0.17, ("B", "crf_basic"): 0.20, ("B", "crf_improved"): 0.23,
    ("C", "hmm"): 0.37, ("C", "crf_basic"): 0.44, ("C", "crf_improved"): 0.47,
}
DROPS_PER_5PCT = {"hmm": 0.002, "crf_basic": 0.008, "crf_improved": 0.015}
SEED_OFFSETS = {2026: -0.001, 2027: 0.0, 2028: 0.0015}
TOKENS = 2000


def score_doc(mode: str, model_name: str, macro_f1: float) -> dict[str, Any]:
    block = PRIMARY[mode]
    return {
        "schema": "recap-eval-v1",
        "label_mode": mode,
        "policy_version": "m1-v2",
        "model_name": model_name,
        "counts": {"sequences": 100, "tokens": 1000, "ignored_prediction_ids": 0},
        "primary": {"block": block, "metric": "macro_f1", "value": macro_f1},
        "blocks": {
            block: {
                "scored_positions": 1000,
                "macro_f1": macro_f1,
                "macro_f1_ci": {"low": round(macro_f1 - 0.02, 6), "high": round(macro_f1 + 0.02, 6)},
            }
        },
    }


def v2_summary_doc(wer: str, seed: int) -> dict[str, Any]:
    """A v2 summary whose cost shares are exactly the 45/30/15/10 target (budgets 100, 200 and 300)."""
    budget = round(float(wer) * TOKENS)
    ins, dele, swaps = budget * 3 // 20, budget * 6 // 20, budget // 20
    sub = budget - ins - dele - 2 * swaps
    return {
        "release": "sponspeech-m1-v2", "split": "test", "seed": seed, "edit_model": "v2",
        "vocabulary_source": "default_train", "wer": float(wer), "achieved_wer": round(budget / TOKENS, 6),
        "budget": budget, "samples": 100, "tokens_before": TOKENS, "tokens_after": TOKENS - dele + ins,
        "edits": {"deletion": dele, "insertion": ins, "substitution": sub, "swap": swaps},
        "cost_share": {
            "deletion": round(dele / budget, 6), "insertion": round(ins / budget, 6),
            "substitution": round(sub / budget, 6), "swap": round(2 * swaps / budget, 6),
        },
    }


def v1_summary_doc(wer: str, seed: int) -> dict[str, Any]:
    """A v1 summary: 25% substitution, 25% deletion, 50% swap by cost, no insertions."""
    budget = round(float(wer) * TOKENS)
    dele = sub = budget // 4
    swaps = (budget - dele - sub) // 2
    return {
        "release": "sponspeech-m1-v2", "split": "test", "seed": seed, "wer": float(wer),
        "achieved_wer": round(budget / TOKENS, 6), "budget": budget, "samples": 100,
        "tokens_before": TOKENS, "tokens_after": TOKENS - dele,
        "edits": {"deletion": dele, "substitution": sub, "swap": swaps},
        "cost_share": {
            "deletion": round(dele / budget, 6), "substitution": round(sub / budget, 6),
            "swap": round(2 * swaps / budget, 6),
        },
    }


def v2_run(root: Path, wer: str, seed: int) -> Path:
    return root / "asr_noise_v2" / "v2" / f"wer{wer}" / f"seed{seed}"


def v1_run(root: Path, seed: int, wer: str = "0.15") -> Path:
    """Seed 2026 of ``v1`` lives in ``asr_noise/``, the other seeds in ``asr_noise_v2/v1/``."""
    if seed == SEEDS[0]:
        return root / "asr_noise" / f"wer{wer}"
    return root / "asr_noise_v2" / "v1" / f"wer{wer}" / f"seed{seed}"


def _write_scores(folder: Path, f1_of: Mapping[tuple[str, str], float]) -> None:
    for (mode, model), value in f1_of.items():
        dump(folder / f"score_{model}_{mode}.json", score_doc(mode, model, round(value, 6)))


def write_asr_scores(
    root: Path,
    *,
    clean: Mapping[tuple[str, str], float] = CLEAN_F1,
    drops: Mapping[str, float] = DROPS_PER_5PCT,
    offsets: Mapping[int, float] = SEED_OFFSETS,
) -> Path:
    """Write the whole tree under ``root``: v2 F1 = clean - step * drop + offset, v1 drops 0.001 less per step."""
    for mode in MODES:
        dump(root / "hmm" / f"hmm_{mode}_test.json", score_doc(mode, "hmm", clean[(mode, "hmm")]))
        for preset in ("basic", "improved"):
            dump(root / "crf_test" / f"m3_{preset}_{mode}_test.json",
                 score_doc(mode, f"crf_{preset}", clean[(mode, f"crf_{preset}")]))
    for step, wer in enumerate(WERS, start=1):
        for seed in SEEDS:
            dump(v2_run(root, wer, seed) / "summary.json", v2_summary_doc(wer, seed))
            _write_scores(v2_run(root, wer, seed), {
                (m, k): clean[(m, k)] - step * drops[k] + offsets[seed] for m in MODES for k in MODELS})
        dump(v1_run(root, SEEDS[0], wer) / "summary.json", v1_summary_doc(wer, SEEDS[0]))
        _write_scores(v1_run(root, SEEDS[0], wer), {
            (m, k): clean[(m, k)] - step * (drops[k] - 0.001) for m in MODES for k in MODELS})
    for seed in SEEDS[1:]:
        dump(v1_run(root, seed) / "summary.json", v1_summary_doc("0.15", seed))
        _write_scores(v1_run(root, seed), {
            (m, k): clean[(m, k)] - 3 * (drops[k] - 0.001) + offsets[seed] for m in MODES for k in MODELS})
    return root


def set_f1(run: Path, mode: str, model: str, value: float) -> None:
    """Replace one score file of a run folder by one holding ``value`` as its primary macro F1."""
    dump(run / f"score_{model}_{mode}.json", score_doc(mode, model, value))


def patch_json(path: Path, **updates: Any) -> None:
    """Merge ``updates`` into the top level of a JSON file."""
    document = json.loads(path.read_text(encoding="utf-8"))
    dump(path, {**document, **updates})


def run_main(root: Path, json_out: Path, md_out: Path) -> int:
    from punct_restore.reporting import asr_noise_table

    return asr_noise_table.main(["--scores-root", str(root), "--json-out", str(json_out), "--out", str(md_out)])


def report(root: Path, out_dir: Path) -> tuple[dict[str, Any], str]:
    """Run the report on ``root`` and return the summary JSON and the Markdown it wrote into ``out_dir``."""
    json_out, md_out = out_dir / "summary.json", out_dir / "ASR_NOISE_V2.md"
    assert run_main(root, json_out, md_out) == 0
    return json.loads(json_out.read_text(encoding="utf-8")), md_out.read_text(encoding="utf-8")


def section(md: str, heading: str) -> str:
    """The text after ``heading`` up to the next heading."""
    body = md.split(heading + "\n", 1)[1]
    next_heading = re.search(r"\n#", body)
    return body[: next_heading.start()] if next_heading else body


def row(text: str, *first_cells: str) -> list[str]:
    """The stripped cells of the one table row of ``text`` that starts with ``first_cells``."""
    start = "| " + " | ".join(first_cells) + " |"
    rows = [line for line in text.splitlines() if line.startswith(start)]
    assert len(rows) == 1, (start, rows)
    return [cell.strip() for cell in rows[0].strip("|").split("|")]
