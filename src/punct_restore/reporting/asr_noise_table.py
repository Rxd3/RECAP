"""Build the ASR noise v2 summary JSON and Markdown report from evaluator and simulator JSON files.

Usage: recap asr-noise-table [--scores-root R] --json-out OUT_JSON --out OUT_MD
Reads clean test scores from R/hmm/hmm_<M>_test.json and R/crf_test/m3_{basic,improved}_<M>_test.json,
v2 noisy test scores and summaries from R/asr_noise_v2/v2/wer{0.05,0.10,0.15}/seed{2026,2027,2028}/,
and v1 noisy test scores and summaries from R/asr_noise/wer{0.05,0.10,0.15}/ (seed 2026) plus
R/asr_noise_v2/v1/wer0.15/seed{2027,2028}/ (default R: results/scores). Both OUT_JSON and OUT_MD must not exist.
Both files are written with LF line endings; if the second cannot be written the first is removed again.

The statistics are in asr_noise_stats.py, the Markdown in asr_noise_report.py, and the findings (every verdict and
number computed from the summary) in asr_noise_findings.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from punct_restore.paths import REPO_ROOT as ROOT
from punct_restore.prediction_io import write_new_file
from punct_restore.reporting._cli import check_out, run
from punct_restore.reporting.asr_noise_report import render_markdown
from punct_restore.reporting.asr_noise_stats import (
    MODELS,
    SEEDS,
    V1_KEY,
    V1_SHARE_KEYS,
    V1_WER,
    V2_SHARE_KEYS,
    V2_TARGET_SHARE,
    WERS,
    SeedStats,
    compute_seed_stats,
    dec,
    mean_of,
    spread_summary,
)
from punct_restore.reporting.build_tables import MODES, PRIMARY, load, pct

Run = tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]  # (simulator summary, entries[mode][model])
__all__ = ["SeedStats", "build", "build_summary", "compute_seed_stats", "main", "render_markdown"]


@contextmanager
def _named(path: Path) -> Iterator[None]:
    """Turn a missing key or a wrong container type in an input file into a one-line error naming the file."""
    try:
        yield
    except KeyError as exc:
        raise ValueError(f"{path.name}: missing field {exc}") from exc
    except TypeError as exc:
        raise ValueError(f"{path.name}: a field has the wrong type") from exc


def _finite_float(value: Any, filename: str, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{filename}: {field} must be a finite number")
    return float(value)


def _unit_float(value: Any, filename: str, field: str) -> float:
    number = _finite_float(value, filename, field)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{filename}: {field} must be between 0 and 1")
    return number


def _positive_int(value: Any, filename: str, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{filename}: {field} must be a positive integer")
    return value


def _nonneg_int(value: Any, filename: str, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{filename}: {field} must be a non-negative integer")
    return value


def check_outputs(json_out: Path, out_md: Path) -> None:
    """Refuse outputs that collide, are folders, have missing parents, or already exist, before reading inputs."""
    check_out(json_out)
    check_out(out_md, {"--json-out": json_out})


def write_outputs(json_out: Path, json_text: str, out_md: Path, md_text: str) -> None:
    """Write both files or neither: if the second cannot be written, the first is removed again."""
    write_new_file(json_out, json_text)
    try:
        write_new_file(out_md, md_text)
    except Exception:
        json_out.unlink(missing_ok=True)
        raise


def _read_primary_f1(path: Path, mode: str) -> dict[str, Any]:
    doc = load(path)
    with _named(path):
        if doc["label_mode"] != mode:
            raise ValueError(f"{path.name}: label_mode must be {mode}, found {doc['label_mode']}")
        block = doc["blocks"][PRIMARY[mode]]
        f1 = _unit_float(block["macro_f1"], path.name, "macro_f1")
        entry: dict[str, Any] = {"macro_f1": f1, "macro_f1_pct": pct(f1)}
        if isinstance(block.get("macro_f1_ci"), dict):
            low = _unit_float(block["macro_f1_ci"]["low"], path.name, "macro_f1_ci.low")
            high = _unit_float(block["macro_f1_ci"]["high"], path.name, "macro_f1_ci.high")
            entry["macro_f1_ci"] = {"low": low, "high": high, "low_pct": pct(low), "high_pct": pct(high)}
    return entry


def _clean_path(scores_root: Path, mode: str, model: str) -> Path:
    if model == "hmm":
        return scores_root / "hmm" / f"hmm_{mode}_test.json"
    preset = model.removeprefix("crf_")
    return scores_root / "crf_test" / f"m3_{preset}_{mode}_test.json"


def _score_filename(model: str, mode: str) -> str:
    return f"score_{model}_{mode}.json"


def _share_keys(doc: Mapping[str, Any], edit_model: str) -> tuple[str, ...]:
    """The edit types to read: all four for ``v2``; for ``v1`` three, plus insertions if its summary has them."""
    if edit_model == "v2":
        return V2_SHARE_KEYS
    return V1_SHARE_KEYS + tuple(k for k in V2_SHARE_KEYS if k not in V1_SHARE_KEYS and k in doc["cost_share"])


def _check_sim_identity(doc: Mapping[str, Any], path: Path, expected_seed: int, edit_model: str) -> None:
    if doc["seed"] != expected_seed:
        raise ValueError(f"{path.name}: expected seed {expected_seed}, found {doc['seed']}")
    if edit_model == "v2":
        if doc.get("edit_model") != "v2":
            raise ValueError(f"{path.name}: edit_model must be 'v2'")
        if doc.get("vocabulary_source") != "default_train":
            raise ValueError(f"{path.name}: vocabulary_source must be 'default_train'")


def _read_sim_summary(path: Path, *, expected_seed: int, expected_wer: str, edit_model: str) -> dict[str, Any]:
    doc = load(path)
    with _named(path):
        _check_sim_identity(doc, path, expected_seed, edit_model)
        tokens_before = _positive_int(doc["tokens_before"], path.name, "tokens_before")
        tokens_after = _positive_int(doc["tokens_after"], path.name, "tokens_after")
        budget = _positive_int(doc["budget"], path.name, "budget")
        target_wer = _finite_float(doc["wer"], path.name, "wer")
        if round(target_wer, 2) != round(float(expected_wer), 2):
            raise ValueError(f"{path.name}: expected wer {expected_wer}, found {target_wer}")
        achieved_wer = _finite_float(doc["achieved_wer"], path.name, "achieved_wer")
        keys = _share_keys(doc, edit_model)
        edits = {k: _nonneg_int(doc["edits"][k], path.name, f"edits.{k}") for k in keys}
        cost_share = {k: _finite_float(doc["cost_share"][k], path.name, f"cost_share.{k}") for k in keys}
    return {
        "seed": expected_seed,
        "wer": target_wer,
        "achieved_wer": achieved_wer,
        "achieved_wer_pct": pct(achieved_wer),
        "budget": budget,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "edits": edits,
        "cost_share": cost_share,
    }


def _aggregate_sim_summaries(per_seed: Mapping[int, dict[str, Any]]) -> dict[str, Any]:
    """Mean and range over seeds of the simulator summaries; an edit type a summary lacks counts as 0
    (v1 inserts none)."""
    first = per_seed[SEEDS[0]]
    for s in SEEDS[1:]:
        if per_seed[s]["tokens_before"] != first["tokens_before"] or per_seed[s]["budget"] != first["budget"]:
            raise ValueError("summary.json files across seeds disagree on tokens_before or budget")
    tokens_after = [per_seed[s]["tokens_after"] for s in SEEDS]
    return {
        "wer": first["wer"],
        "tokens_before": first["tokens_before"],
        "budget": first["budget"],
        "achieved_wer": first["achieved_wer"],
        "achieved_wer_pct": pct(first["achieved_wer"]),
        "tokens_after_mean": float(mean_of([dec(t) for t in tokens_after])),
        "tokens_after_min": min(tokens_after),
        "tokens_after_max": max(tokens_after),
        "cost_share": {
            k: spread_summary([dec(per_seed[s]["cost_share"].get(k, 0)) for s in SEEDS]) for k in V2_SHARE_KEYS
        },
        "edits_mean": {
            k: float(mean_of([dec(per_seed[s]["edits"].get(k, 0)) for s in SEEDS])) for k in V2_SHARE_KEYS
        },
        "per_seed": {str(s): per_seed[s] for s in SEEDS},
    }


def _v2_folder(root: Path, wer: str, seed: int) -> Path:
    return root / "asr_noise_v2" / "v2" / f"wer{wer}" / f"seed{seed}"


def _v1_folder(root: Path, wer: str, seed: int) -> Path:
    """Seed 2026 of v1 is the original asr_noise run; the other seeds are in asr_noise_v2/v1."""
    if seed == SEEDS[0]:
        return root / "asr_noise" / f"wer{wer}"
    return root / "asr_noise_v2" / "v1" / f"wer{wer}" / f"seed{seed}"


def _read_run(folder: Path, *, seed: int, wer: str, edit_model: str) -> Run:
    sim = _read_sim_summary(folder / "summary.json", expected_seed=seed, expected_wer=wer, edit_model=edit_model)
    entries = {
        mode: {model: _read_primary_f1(folder / _score_filename(model, mode), mode) for model in MODELS}
        for mode in MODES
    }
    return sim, entries


def _stats(clean_f1: float, runs: Mapping[int, Run], mode: str, model: str) -> dict[str, Any]:
    scores = {seed: runs[seed][1][mode][model]["macro_f1"] for seed in SEEDS}
    return asdict(compute_seed_stats(clean_f1, scores))


def _read_clean(scores_root: Path) -> dict[str, dict[str, Any]]:
    return {
        mode: {model: _read_primary_f1(_clean_path(scores_root, mode, model), mode) for model in MODELS}
        for mode in MODES
    }


def _v2_summary(scores_root: Path, clean: Mapping[str, Any]) -> dict[str, Any]:
    runs = {
        wer: {s: _read_run(_v2_folder(scores_root, wer, s), seed=s, wer=wer, edit_model="v2") for s in SEEDS}
        for wer in WERS
    }
    modes = {
        mode: {
            "primary_block": PRIMARY[mode],
            "models": {
                model: {
                    "clean": clean[mode][model],
                    "by_wer": {wer: _stats(clean[mode][model]["macro_f1"], runs[wer], mode, model) for wer in WERS},
                }
                for model in MODELS
            },
        }
        for mode in MODES
    }
    return {
        "edit_model": "v2",
        "vocabulary_source": "default_train",
        "target_cost_share": dict(V2_TARGET_SHARE),
        "achieved_by_wer": {wer: _aggregate_sim_summaries({s: runs[wer][s][0] for s in SEEDS}) for wer in WERS},
        "modes": modes,
    }


def _v1_summary(scores_root: Path, clean: Mapping[str, Any]) -> dict[str, Any]:
    first = {
        wer: _read_run(_v1_folder(scores_root, wer, SEEDS[0]), seed=SEEDS[0], wer=wer, edit_model="v1")
        for wer in WERS
    }
    runs = {SEEDS[0]: first[V1_WER]} | {
        s: _read_run(_v1_folder(scores_root, V1_WER, s), seed=s, wer=V1_WER, edit_model="v1") for s in SEEDS[1:]
    }
    modes = {
        mode: {
            "primary_block": PRIMARY[mode],
            "models": {model: _stats(clean[mode][model]["macro_f1"], runs, mode, model) for model in MODELS},
        }
        for mode in MODES
    }
    return {
        "edit_model": "v1",
        "seed_2026_by_wer": {wer: {"summary": first[wer][0], "modes": first[wer][1]} for wer in WERS},
        V1_KEY: {"achieved": _aggregate_sim_summaries({s: runs[s][0] for s in SEEDS}), "modes": modes},
    }


def build_summary(scores_root: Path) -> dict[str, Any]:
    """Read clean, v2, and v1 evaluator and simulator JSON files and return the full structured summary."""
    clean = _read_clean(scores_root)
    return {
        "split": "test",
        "modes": list(MODES),
        "primary_blocks": dict(PRIMARY),
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "wers": list(WERS),
        "clean": clean,
        "v2": _v2_summary(scores_root, clean),
        "v1": _v1_summary(scores_root, clean),
    }


def build(scores_root: Path, json_out: Path, out_md: Path) -> None:
    check_outputs(json_out, out_md)  # before reading anything
    summary = build_summary(scores_root)
    json_text = json.dumps(summary, indent=2) + "\n"
    markdown = render_markdown(summary)
    write_outputs(json_out, json_text, out_md, markdown)
    print(f"wrote {json_out.name} and {out_md.name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recap asr-noise-table",
        description="Build the ASR noise v2 summary JSON and Markdown report from evaluator JSON files.",
    )
    parser.add_argument(
        "--scores-root",
        type=Path,
        default=ROOT / "results" / "scores",
        help="folder holding hmm/, crf_test/, asr_noise/ and asr_noise_v2/",
    )
    parser.add_argument("--json-out", type=Path, required=True, help="JSON summary file to create (must not exist)")
    parser.add_argument("--out", type=Path, required=True, help="Markdown report file to create (must not exist)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(lambda: build(args.scores_root, args.json_out, args.out))


if __name__ == "__main__":
    sys.exit(main())
