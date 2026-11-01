"""Ablation table from the evaluator's compare files (full improved CRF vs one group removed, validation).

Usage: recap build-ablation-table [--scores-root R] --out OUT_MD [--setup TEXT]
Reads R/ablations/mode_<M>/ (default R: results/scores); OUT_MD must not exist. Groups are read from the
improved_minus_<group>.json files present, in a fixed order when known.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from punct_restore.paths import REPO_ROOT as ROOT
from punct_restore.reporting._cli import check_out, run, write_new_text
from punct_restore.reporting.build_tables import MODES, PRIMARY, cell, diff_cell, load

ORDER = ("window2", "bigrams", "lexcls", "qcue", "ngrams", "question", "clause", "casing", "position")
DEFAULT_SETUP = "50k-sequence sample, 100 iterations, frozen c1/c2"


def groups(scores: Path) -> list[str]:
    found = [p.stem.removeprefix("improved_minus_") for p in scores.glob("improved_minus_*.json")]
    return sorted(found, key=lambda g: (ORDER.index(g) if g in ORDER else len(ORDER), g))


def mode_lines(scores: Path, mode: str) -> list[str]:
    full = load(scores / "improved.json")["blocks"]
    lines = [(f"**Mode {mode}**: full improved {cell(full[PRIMARY[mode]])} primary, "
              f"{cell(full['punctuation/non_final'])} non_final"), "",
             "| Without | Primary | Δ primary | non_final | Δ non_final |", "|---|---|---|---|---|"]
    for group in groups(scores):
        variant = f"improved_minus_{group}"
        score = load(scores / f"{variant}.json")["blocks"]
        comp = load(scores / f"full_vs_{variant}.json")["blocks"]
        lines.append(f"| {group} | {cell(score[PRIMARY[mode]])} | {diff_cell(comp[PRIMARY[mode]])} | "
                     f"{cell(score['punctuation/non_final'])} | {diff_cell(comp['punctuation/non_final'])} |")
    return lines + [""]


def build(root: Path, out_md: Path, setup: str = DEFAULT_SETUP) -> None:
    check_out(out_md)  # before reading anything
    lines = [f"## Feature ablations (validation; {setup})", "",
             ("Each row removes one feature group from the improved CRF. Difference = without − with, in points "
              "(95% paired CI, grouped bootstrap by video, 1,000 resamples, seed 2026; * = interval excludes 0). "
              "Negative means the group helps."), ""]
    for mode in MODES:
        lines += mode_lines(root / "ablations" / f"mode_{mode}", mode)
    write_new_text(out_md, "\n".join(lines))
    print(f"wrote {out_md}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recap build-ablation-table", description="Write the feature-ablation table as Markdown.")
    parser.add_argument("--scores-root", type=Path, default=ROOT / "results" / "scores",
                        help="folder holding ablations/mode_A, mode_B and mode_C")
    parser.add_argument("--out", type=Path, required=True, help="Markdown file to create (must not exist)")
    parser.add_argument("--setup", default=DEFAULT_SETUP, help="setup text for the table heading")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(lambda: build(args.scores_root, args.out, args.setup))


if __name__ == "__main__":
    sys.exit(main())
