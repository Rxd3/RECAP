"""Write the CRF result tables for one split (basic and improved CRF vs HMM and shortcut) as Markdown.

Usage: recap build-m3-tables [--scores-root R] [--split validation|test] [--tau] --out OUT_MD
Every number comes from evaluator JSON (score / compare / errors) under the scores root R
(default results/scores): the CRF files from R/crf_validation or R/crf_test (validation files are
named m3_<preset>_<M>.json, test files m3_<preset>_<M>_test.json), the shortcut and HMM files from
R/hmm, and with --tau the threshold files from R/threshold.
OUT_MD must not exist: the script never replaces a file.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

from punct_restore.file_io import read_object, require_object
from punct_restore.paths import REPO_ROOT as ROOT
from punct_restore.reporting._cli import check_out, run, write_new_text
from punct_restore.reporting.build_tables import (
    CELL,
    CONFUSION,
    DIFF,
    MODES,
    PER_CLASS,
    PRIMARY,
    ROW,
    cell,
    diff_cell,
    half_up,
    pct,
    read_block,
    read_scores,
)

NON_FINAL = "punctuation/non_final"
C_VIEWS = ("punctuation/all", "casing/unmasked", "casing/unmasked_non_initial", "casing/all")  # mode C, after non_final
SPLITS = {"validation": "", "test": "_test"}
CRF_DIRS = {"validation": "crf_validation", "test": "crf_test"}
Models = dict[str, Callable[[str], Path]]


def models(scores: Path, refs: Path, split: str, tau: Path | None = None) -> Models:
    sfx = SPLITS[split]
    out: Models = {
        "Shortcut": lambda m: refs / f"shortcut_{m}_{split}.json",
        "HMM (M2)": lambda m: refs / f"hmm_{m}_{split}.json",
        "CRF basic (M3)": lambda m: scores / f"m3_basic_{m}{sfx}.json",
        "CRF improved (M3)": lambda m: scores / f"m3_improved_{m}{sfx}.json",
    }
    if tau is not None:
        out["CRF improved + threshold (M5)"] = lambda m: tau / f"tau_improved_{m}{sfx}.json"
    return out


def prf(block: dict) -> str:
    mi = block["micro"]
    return f"{pct(mi['precision'])} / {pct(mi['recall'])} / {pct(mi['f1'])}"


def sec_main(ms: Models, split: str) -> list[str]:
    ids = "primary ID lists" if split == "validation" else "primary test ID lists, each model predicted once"
    out = [f"## 1. Main result ({split}, {ids})", "",
           ("Primary = macro F1 % with 95% CI (grouped bootstrap by video, 1,000 resamples, seed 2026). "
            "Micro P / R / F1 % over the same classes. A/B: non-O punctuation classes; C: all 12 joint classes "
            "on unmasked positions (case_joint mask applied)."), ""]
    for mode in MODES:
        blk = PRIMARY[mode]
        out += [f"**Mode {mode}** (`{blk}`)", "", "| Model | Macro F1 (95% CI) | Micro P / R / F1 | Accuracy |",
                "|---|---|---|---|"]
        for name, f in ms.items():
            b = read_block(f(mode), blk, ROW)
            out.append(f"| {name} | {cell(b)} | {prf(b)} | {pct(b['accuracy'])} |")
        out.append("")
    return out


def sec_paired(scores: Path, split: str, tau: Path | None = None) -> list[str]:
    sfx = SPLITS[split]
    rows = [("m3_improved_vs_basic", "M3 improved − M3 basic", scores), ("m3_basic_vs_hmm", "M3 basic − HMM", scores),
            ("m3_improved_vs_hmm", "M3 improved − HMM", scores)]
    if tau is not None:
        rows.append(("tau_vs_improved", "threshold − M3 improved", tau))
    out = [f"## 2. Paired differences ({split}, points, 95% CI; * = interval excludes 0)", "",
           "| Mode | Comparison | Primary | non_final |", "|---|---|---|---|"]
    for mode in MODES:
        for stem, label, folder in rows:
            shown = dict.fromkeys((PRIMARY[mode], NON_FINAL), DIFF)
            b = read_scores(folder / f"{stem}_{mode}{sfx}.json", shown)["blocks"]
            primary, non_final = diff_cell(b[PRIMARY[mode]]), diff_cell(b[NON_FINAL])
            out.append(f"| {mode} | {label} | {primary} | {non_final} |")
    return out + [""]


def sec_views(ms: Models, split: str) -> list[str]:
    out = [f"## 3. Views and mode C sub-blocks ({split}, macro F1 %, 95% CI)", "",
           ("`non_final` drops each utterance's last token. Mode C casing uses the case_joint mask; "
            "`casing/all` is unmasked, for reference only."), "",
           ("| Model | A non_final | B non_final | C non_final | C punctuation/all | C casing (masked) | "
            "C casing non-initial | C casing/all |"), "|---|---|---|---|---|---|---|---|"]
    for name, f in ms.items():
        nf = [cell(read_block(f(m), NON_FINAL, CELL)) for m in MODES]
        c = read_scores(f("C"), dict.fromkeys(C_VIEWS, CELL))["blocks"]
        rest = [cell(c[k]) for k in C_VIEWS]
        out.append(f"| {name} | " + " | ".join(nf + rest) + " |")
    return out + [""]


def class_table(ms: Models, mode: str, blk: str, title: str) -> list[str]:
    files = [f(mode) for f in ms.values()]
    labels = read_block(files[0], blk, {"labels": list[str]})["labels"]
    shape = {"per_class": {lab: PER_CLASS for lab in labels}}
    stats = [read_block(path, blk, shape)["per_class"] for path in files]
    out = [f"**{title}**", "", "| Class | Support | " + " | ".join(f"{n} P / R / F1" for n in ms) + " |",
           "|---" * (2 + len(ms)) + "|"]
    for lab in labels:
        cells = [f"{pct(pc[lab]['precision'])} / {pct(pc[lab]['recall'])} / {pct(pc[lab]['f1'])}" for pc in stats]
        name = lab.replace("|", r"\|")  # joint labels contain a pipe, which would split the Markdown cell
        out.append(f"| {name} | {stats[0][lab]['support']:,} | " + " | ".join(cells) + " |")
    return out + [""]


def sec_classes(ms: Models, split: str) -> list[str]:
    out = [f"## 4. Per-class precision / recall / F1 % ({split})", "",
           "Classes marked O are not in the A/B primary average; in mode C all 12 joint classes are.", ""]
    out += class_table(ms, "A", "punctuation/all", "Mode A")
    out += class_table(ms, "B", "punctuation/all", "Mode B")
    out += class_table(ms, "C", "joint/unmasked", "Mode C joint (masked)")
    out += class_table(ms, "C", "casing/unmasked", "Mode C casing (masked)")
    return out


def read_errors(path: Path) -> dict[str, Any]:
    """The errors file at ``path``: its positive ``errors_total`` and the five most frequent ``pair_counts``."""
    errors = read_object(path, {"errors_total": int, "pair_counts": list})
    if errors["errors_total"] <= 0:
        raise ValueError(f"{path.name}: errors_total must be positive")
    for number, pair in enumerate(errors["pair_counts"][:5]):
        require_object(path.name, pair, {"gold": str, "pred": str, "count": int}, f"pair_counts.{number}")
    return errors


def sec_errors(scores: Path, split: str) -> list[str]:
    sfx = SPLITS[split]
    out = [f"## Most frequent errors, M3 improved ({split}, punctuation labels; mode C projected to punctuation)", "",
           "| Mode | Errors | Top pairs (count, share) |", "|---|---|---|"]
    for mode in MODES:
        e = read_errors(scores / f"m3_improved_{mode}{sfx}.errors.json")
        total = e["errors_total"]
        top = "; ".join(f"{p['gold']}→{p['pred']} {p['count']:,} ({half_up(Decimal(100 * p['count']) / total)}%)"
                        for p in e["pair_counts"][:5])
        out.append(f"| {mode} | {e['errors_total']:,} | {top} |")
    return out + [""]


def matrix(block: dict, title: str) -> list[str]:
    labels = block["labels"]
    rows = [f"*{title}* (rows = gold, columns = predicted)", "", "| gold \\ pred | " + " | ".join(labels) + " |",
            "|---" * (len(labels) + 1) + "|"]
    rows += [f"| {lab} | " + " | ".join(f"{n:,}" for n in row) + " |"
             for lab, row in zip(labels, block["confusion"], strict=True)]
    return rows + [""]


def sec_confusion(ms: Models, split: str) -> list[str]:
    out = [f"## Confusion matrices, M3 CRFs ({split})", "",
           ("Mode C is shown as its punctuation projection and its masked casing projection; the 12 × 12 joint "
            "matrix is in each score JSON (`blocks.joint/unmasked.confusion`)."), ""]
    crfs = {n: f for n, f in ms.items() if "M3" in n}
    for mode in MODES:
        out += [f"### Mode {mode}", ""]
        for name, f in crfs.items():
            shown = ("punctuation/all", "casing/unmasked") if mode == "C" else ("punctuation/all",)
            blocks = read_scores(f(mode), dict.fromkeys(shown, CONFUSION))["blocks"]
            out += matrix(blocks["punctuation/all"], f"{name}, punctuation")
            if mode == "C":
                out += matrix(blocks["casing/unmasked"], f"{name}, casing (masked)")
    return out


def build(root: Path, out_md: Path, split: str = "validation", tau: bool = False) -> None:
    check_out(out_md)  # before reading anything
    scores, tau_dir = root / CRF_DIRS[split], (root / "threshold" if tau else None)
    ms = models(scores, root / "hmm", split, tau_dir)
    body = sec_main(ms, split) + sec_paired(scores, split, tau_dir) + sec_views(ms, split) + sec_classes(ms, split)
    body += sec_errors(scores, split) + sec_confusion(ms, split)
    write_new_text(out_md, "\n".join(body))
    print(f"wrote {out_md}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recap build-m3-tables", description="Write the CRF result tables for one split as Markdown.")
    parser.add_argument("--scores-root", type=Path, default=ROOT / "results" / "scores",
                        help="folder holding hmm/, crf_validation/, crf_test/ and threshold/")
    parser.add_argument("--split", choices=tuple(SPLITS), default="validation")
    parser.add_argument("--tau", action="store_true", help="add the threshold-decoding rows (needs R/threshold)")
    parser.add_argument("--out", type=Path, required=True, help="Markdown file to create (must not exist)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(lambda: build(args.scores_root, args.out, args.split, args.tau))


if __name__ == "__main__":
    sys.exit(main())
