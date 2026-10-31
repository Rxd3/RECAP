"""Tests for scripts/reporting/asr_noise_table.py using synthetic score fixtures only (no real data)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from reporting_fixtures import assert_one_error_line, dump

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "reporting"
EN = "\u2013"
MODES = ("A", "B", "C")
PRIMARY = {"A": "punctuation/all", "B": "punctuation/all", "C": "joint/unmasked"}
WERS = ("0.05", "0.10", "0.15")
SEEDS = (2026, 2027, 2028)


def _score_doc(mode: str, model_name: str, macro_f1: float) -> dict[str, Any]:
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


def _v2_summary_doc(wer: str, seed: int) -> dict[str, Any]:
    w = float(wer)
    budget = round(w * 1000)
    ins = round(0.15 * budget)
    dele = round(0.30 * budget)
    swap = budget // 20
    sub = budget - ins - dele - 2 * swap
    return {
        "release": "sponspeech-m1-v2",
        "split": "test",
        "seed": seed,
        "edit_model": "v2",
        "vocabulary_source": "default_train",
        "wer": w,
        "achieved_wer": round(budget / 1000, 6),
        "budget": budget,
        "samples": 100,
        "tokens_before": 1000,
        "tokens_after": 1000 - dele + ins,
        "edits": {
            "deletion": dele,
            "insertion": ins,
            "substitution": sub,
            "swap": swap,
        },
        "cost_share": {
            "deletion": round(dele / budget, 6),
            "insertion": round(ins / budget, 6),
            "substitution": round(sub / budget, 6),
            "swap": round(2 * swap / budget, 6),
        },
    }


def _v1_summary_doc(wer: str, seed: int) -> dict[str, Any]:
    w = float(wer)
    budget = round(w * 1000)
    dele = budget // 4
    sub = budget // 4
    swap = (budget - dele - sub) // 2
    return {
        "release": "sponspeech-m1-v2",
        "split": "test",
        "seed": seed,
        "wer": w,
        "achieved_wer": round(budget / 1000, 6),
        "budget": budget,
        "samples": 100,
        "tokens_before": 1000,
        "tokens_after": 1000 - dele,
        "edits": {
            "deletion": dele,
            "substitution": sub,
            "swap": swap,
        },
        "cost_share": {
            "deletion": round(dele / budget, 6),
            "substitution": round(sub / budget, 6),
            "swap": round(2 * swap / budget, 6),
        },
    }


@pytest.fixture
def asr_scores(tmp_path: Path) -> Path:
    """Build a complete synthetic scores tree for asr_noise_table."""
    root = tmp_path / "scores"
    base_clean = {
        ("A", "hmm"): 0.350000,
        ("A", "crf_basic"): 0.430000,
        ("A", "crf_improved"): 0.500000,
        ("B", "hmm"): 0.170000,
        ("B", "crf_basic"): 0.200000,
        ("B", "crf_improved"): 0.230000,
        ("C", "hmm"): 0.370000,
        ("C", "crf_basic"): 0.440000,
        ("C", "crf_improved"): 0.470000,
    }
    for mode in MODES:
        dump(root / "hmm" / f"hmm_{mode}_test.json", _score_doc(mode, "hmm", base_clean[(mode, "hmm")]))
        for preset in ("basic", "improved"):
            model = f"crf_{preset}"
            dump(
                root / "crf_test" / f"m3_{preset}_{mode}_test.json",
                _score_doc(mode, model, base_clean[(mode, model)]),
            )

    drops_per_5pct = {"hmm": 0.002000, "crf_basic": 0.008000, "crf_improved": 0.015000}
    seed_offsets = {2026: -0.001000, 2027: 0.000000, 2028: 0.001500}

    for wer_idx, wer in enumerate(WERS, start=1):
        for seed in SEEDS:
            folder = root / "asr_noise_v2" / "v2" / f"wer{wer}" / f"seed{seed}"
            dump(folder / "summary.json", _v2_summary_doc(wer, seed))
            for mode in MODES:
                for model, fname in (
                    ("hmm", f"score_hmm_{mode}.json"),
                    ("crf_basic", f"score_crf_basic_{mode}.json"),
                    ("crf_improved", f"score_crf_improved_{mode}.json"),
                ):
                    f1 = round(base_clean[(mode, model)] - wer_idx * drops_per_5pct[model] + seed_offsets[seed], 6)
                    dump(folder / fname, _score_doc(mode, model, f1))

        # v1 seed 2026 under asr_noise/wer<rate>/
        v1_folder = root / "asr_noise" / f"wer{wer}"
        dump(v1_folder / "summary.json", _v1_summary_doc(wer, 2026))
        for mode in MODES:
            for model, fname in (
                ("hmm", f"score_hmm_{mode}.json"),
                ("crf_basic", f"score_crf_basic_{mode}.json"),
                ("crf_improved", f"score_crf_improved_{mode}.json"),
            ):
                f1 = round(base_clean[(mode, model)] - wer_idx * (drops_per_5pct[model] - 0.001000), 6)
                dump(v1_folder / fname, _score_doc(mode, model, f1))

    # v1 wer0.15 seeds 2027 and 2028 under asr_noise_v2/v1/wer0.15/seed<seed>/
    for seed in (2027, 2028):
        v1_seed_folder = root / "asr_noise_v2" / "v1" / "wer0.15" / f"seed{seed}"
        dump(v1_seed_folder / "summary.json", _v1_summary_doc("0.15", seed))
        for mode in MODES:
            for model, fname in (
                ("hmm", f"score_hmm_{mode}.json"),
                ("crf_basic", f"score_crf_basic_{mode}.json"),
                ("crf_improved", f"score_crf_improved_{mode}.json"),
            ):
                f1 = round(base_clean[(mode, model)] - 3 * (drops_per_5pct[model] - 0.001000) + seed_offsets[seed], 6)
                dump(v1_seed_folder / fname, _score_doc(mode, model, f1))

    return root


def test_asr_noise_table_end_to_end(
    asr_scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.reporting import asr_noise_table

    json_out = tmp_path / "summary.json"
    md_out = tmp_path / "ASR_NOISE_V2.md"
    code = asr_noise_table.main(
        ["--scores-root", str(asr_scores), "--json-out", str(json_out), "--out", str(md_out)]
    )
    captured = capsys.readouterr()
    assert (code, captured.err) == (0, "")
    assert json_out.is_file() and md_out.is_file()

    summary = json.loads(json_out.read_text(encoding="utf-8"))
    md_text = md_out.read_text(encoding="utf-8")

    # Markdown is generated purely from the JSON summary dict
    assert asr_noise_table.render_markdown(summary) == md_text

    # Check Mode A CRF improved at WER 0.05:
    # clean = 0.50, wer 0.05 seeds: 2026 -> 0.484000, 2027 -> 0.485000, 2028 -> 0.486500
    # mean = (0.484 + 0.485 + 0.4865) / 3 = 0.4851666... -> 0.485167 (48.52%), min 48.40, max 48.65
    imp_a_05 = summary["v2"]["modes"]["A"]["models"]["crf_improved"]["by_wer"]["0.05"]
    assert imp_a_05["mean"] == 0.485167
    assert imp_a_05["mean_pct"] == "48.52"
    assert imp_a_05["min_pct"] == "48.40"
    assert imp_a_05["max_pct"] == "48.65"
    assert imp_a_05["cell"] == f"48.52 (48.40{EN}48.65)"

    # Check per-mode Markdown tables exist and contain expected formatted cells
    assert "### Mode A (`punctuation/all`)" in md_text
    assert "### Mode B (`punctuation/all`)" in md_text
    assert "### Mode C (`joint/unmasked`)" in md_text
    assert f"| CRF improved | 50.00 | 48.52 (48.40{EN}48.65) |" in md_text

    # Check required sections and findings questions are covered
    assert "## 1. Method" in md_text
    assert "## 2. Achieved Edit Mix (`v2` and `v1`)" in md_text
    assert "## 3. Primary Macro F1 by Mode (`v2`)" in md_text
    assert "## 4. Comparison: `v1` vs `v2` at 15% WER" in md_text
    assert "## 5. Findings" in md_text
    assert "Does the improved CRF still degrade faster than the HMM?" in md_text
    assert "Is the ranking stable across seeds?" in md_text
    assert "How do `v1` and `v2` differ?" in md_text
    assert "## 6. Limits" in md_text


def test_asr_noise_table_rejects_malformed_json_and_bad_values(
    asr_scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.reporting import asr_noise_table

    bad_file = asr_scores / "hmm" / "hmm_A_test.json"
    bad_file.write_text("{broken", encoding="utf-8")
    code = asr_noise_table.main(
        ["--scores-root", str(asr_scores), "--json-out", str(tmp_path / "s.json"), "--out", str(tmp_path / "o.md")]
    )
    assert code == 1
    assert_one_error_line(capsys.readouterr().err, "hmm_A_test.json")

    dump(bad_file, _score_doc("A", "hmm", 0.35))
    doc = json.loads(bad_file.read_text(encoding="utf-8"))
    doc["blocks"]["punctuation/all"]["macro_f1"] = "nan_str"
    dump(bad_file, doc)
    code = asr_noise_table.main(
        ["--scores-root", str(asr_scores), "--json-out", str(tmp_path / "s.json"), "--out", str(tmp_path / "o.md")]
    )
    assert code == 1
    assert_one_error_line(capsys.readouterr().err, "hmm_A_test.json", "macro_f1")


def test_asr_noise_table_usage_error_exits_with_2() -> None:
    from scripts.reporting import asr_noise_table

    with pytest.raises(SystemExit) as raised:
        asr_noise_table.main([])
    assert raised.value.code == 2


def test_asr_noise_table_refuses_existing_outputs_before_reading(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.reporting import asr_noise_table

    existing = tmp_path / "exists.json"
    existing.write_text("keep", encoding="utf-8")
    code = asr_noise_table.main(
        ["--scores-root", str(tmp_path / "missing"), "--json-out", str(existing), "--out", str(tmp_path / "out.md")]
    )
    assert code == 1
    assert_one_error_line(capsys.readouterr().err, "exists.json", "already exists")
    assert existing.read_text(encoding="utf-8") == "keep"


def test_asr_noise_table_refuses_same_path_for_json_and_md(
    asr_scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.reporting import asr_noise_table

    same = tmp_path / "same.out"
    code = asr_noise_table.main(
        ["--scores-root", str(asr_scores), "--json-out", str(same), "--out", str(same)]
    )
    assert code == 1
    assert_one_error_line(capsys.readouterr().err, "--json-out", "--out", "same file")
    assert not same.exists()


def test_asr_noise_table_refuses_folder_or_missing_parent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.reporting import asr_noise_table

    folder = tmp_path / "a_folder"
    folder.mkdir()
    code = asr_noise_table.main(
        ["--scores-root", str(tmp_path / "missing"), "--json-out", str(tmp_path / "s.json"), "--out", str(folder)]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert_one_error_line(err, "a_folder", "is a folder")
    assert str(tmp_path) not in err

    missing_parent = tmp_path / "no_such_dir" / "out.md"
    code = asr_noise_table.main(
        [
            "--scores-root",
            str(tmp_path / "missing"),
            "--json-out",
            str(tmp_path / "s.json"),
            "--out",
            str(missing_parent),
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert_one_error_line(err, "out.md: folder no_such_dir not found")
    assert str(tmp_path) not in err


def test_asr_noise_table_reports_missing_input_without_folder_path(
    asr_scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.reporting import asr_noise_table

    victim = asr_scores / "asr_noise_v2" / "v2" / "wer0.10" / "seed2027" / "score_hmm_B.json"
    victim.unlink()
    code = asr_noise_table.main(
        [
            "--scores-root",
            str(asr_scores),
            "--json-out",
            str(tmp_path / "s.json"),
            "--out",
            str(tmp_path / "out.md"),
        ]
    )
    err = capsys.readouterr().err
    assert code == 1
    assert_one_error_line(err, "score_hmm_B.json")
    assert str(asr_scores) not in err
    assert not (tmp_path / "s.json").exists() and not (tmp_path / "out.md").exists()


def test_asr_noise_table_rejects_wrong_vocabulary_source(
    asr_scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.reporting import asr_noise_table

    summary_path = asr_scores / "asr_noise_v2" / "v2" / "wer0.05" / "seed2026" / "summary.json"
    doc = json.loads(summary_path.read_text(encoding="utf-8"))
    doc["vocabulary_source"] = "validation"
    dump(summary_path, doc)

    code = asr_noise_table.main(
        [
            "--scores-root",
            str(asr_scores),
            "--json-out",
            str(tmp_path / "s.json"),
            "--out",
            str(tmp_path / "out.md"),
        ]
    )
    err = capsys.readouterr().err
    assert code == 1
    assert_one_error_line(err, "summary.json", "default_train")


def test_asr_noise_table_runs_as_script_on_cp1252_console(asr_scores: Path, tmp_path: Path) -> None:
    json_out = tmp_path / "\u03a9summary.json"
    md_out = tmp_path / "\u03a9report.md"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"PYTHONIOENCODING": "cp1252"}
    done = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "asr_noise_table.py"),
            "--scores-root",
            str(asr_scores),
            "--json-out",
            str(json_out),
            "--out",
            str(md_out),
        ],
        env=env,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert (done.returncode, done.stderr) == (0, b"")
    assert json_out.is_file() and md_out.is_file()
