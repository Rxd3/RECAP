"""Sections 7-10 of docs/EVALUATION_RESULTS_CRF.md (threshold grid, leakage, test_amb, runtime) from evaluator JSON.

Usage: python scripts/reporting/build_m3_final_sections.py [--scores-root R] --manifests DIR --out OUT_MD
Scores come from R (default results/scores): threshold/, leakage/, crf_test/, hmm/ and runtime/. The CRF fit
times come from Member 3's training manifests crf_mode_<m>_<preset>_full_01.manifest.json in DIR.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _cli import run
from build_tables import MODES, PRIMARY, cell, diff_cell, load, pct

ROOT = Path(__file__).resolve().parents[2]


def sec_threshold(tau_dir: Path) -> list[str]:
    grids = {m: {round(g["tau"], 2): g["macro_f1"] for g in load(tau_dir / f"tau_{m}.json")["grid"]} for m in MODES}
    chosen = {m: load(tau_dir / f"tau_{m}.json")["tau"] for m in MODES}
    out = ["## 7. Threshold decoding (Member 5 extension)", "",
           ("For each token, the improved CRF's marginals give P(O); in mode C, P(O) sums every `O|*` label. If "
            "1 − P(O) > τ the token gets its most probable non-O label, otherwise its most probable O label "
            "(`src/punct_restore/models/threshold.py`). τ was tuned per mode on the primary validation list "
            "(`scripts/tune_threshold.py tune`, grid 0.05–0.95), then test was predicted once with the chosen τ: "
            f"A {chosen['A']}, B {chosen['B']}, C {chosen['C']}. Test effects are in sections 1–2."), "",
           "Validation primary macro F1 % per τ (bold = chosen; standard Viterbi decoding scores A 51.18, B 25.10, C 51.52):",
           "", "| τ | A | B | C |", "|---|---|---|---|"]
    for tau in sorted(grids["A"]):
        row = [f"**{pct(grids[m][tau])}**" if tau == round(chosen[m], 2) else pct(grids[m][tau]) for m in MODES]
        out.append(f"| {tau:.2f} | " + " | ".join(row) + " |")
    return out + [""]


def sec_leakage(supp: Path) -> list[str]:
    rows = [("Official test, all utterances", "canonical", "official_all"),
            ("Leakage-controlled test (kept)", "canonical", "leakfree"),
            ("Excluded: video also in the training set", "canonical", "train_video"),
            ("Excluded: other leakage (video shared with validation only, or duplicate text)", "canonical",
             "other_leak"),
            ("Excluded: quality filter (low comma density or TTS source)", "canonical", "quality"),
            ("Official test, all utterances", "SponSpeech native", "native_all")]
    out = ["## 8. Leakage: official vs leakage-controlled test (mode A)", "",
           ("Supplementary. The frozen improved CRF and the HMM predicted all 17,695 official test utterances once "
            "(the CRF's predictions on the primary list are identical to section 1's). The excluded utterances are "
            "the official test minus `data/splits/sponspeech-m1-v2/leakage_controlled/test.ids`, split by why they "
            "were excluded (`leakage_controlled/test_excluded.tsv`). \"Training set\" means a video with an "
            "utterance in `default_train`, which both models were trained on. Macro F1 % (95% CI; the quality group "
            "is one video, so its interval is degenerate)."), "",
           "| Scored set | Gold labels | Utterances | HMM | CRF improved |", "|---|---|---|---|---|"]
    for label, gold, key in rows:
        h = load(supp / f"hmm_A_test_{key}.json")
        c = load(supp / f"crf_A_test_{key}.json")
        n = h["counts"]["sequences"]
        out.append(f"| {label} | {gold} | {n:,} | {cell(h['blocks']['punctuation/all'])} | "
                   f"{cell(c['blocks']['punctuation/all'])} |")
    return out + ["", ("On utterances from videos it was trained on, the improved CRF scores about 18 points "
                       "higher than on the leak-free set; the HMM does not gain at all. Reporting the official test "
                       "as-is would overstate the CRF by about 3 points."), ""]


def sec_test_amb(refs: Path, test: Path, tau_dir: Path) -> list[str]:
    names: list[tuple[str, Callable[[str], Path]]] = [
        ("Shortcut", lambda m: refs / f"shortcut_{m}_test_amb.json"),
        ("HMM (M2)", lambda m: refs / f"hmm_{m}_test_amb.json"),
        ("CRF basic (M3)", lambda m: test / f"m3_basic_{m}_test_amb.json"),
        ("CRF improved (M3)", lambda m: test / f"m3_improved_{m}_test_amb.json"),
        ("CRF improved + threshold (M5)", lambda m: tau_dir / f"tau_improved_{m}_test_amb.json")]
    out = ["## 9. test_amb (second held-out set)", "",
           ("Primary macro F1 % (95% CI) on the primary test_amb lists. test_amb shares 25 videos with test, so the "
            "two are reported separately and never pooled."), "", "| Model | A | B | C |", "|---|---|---|---|"]
    for name, f in names:
        out.append(f"| {name} | " + " | ".join(cell(load(f(m))["blocks"][PRIMARY[m]]) for m in MODES) + " |")
    out += ["", "| Paired difference (test_amb) | A | B | C |", "|---|---|---|---|"]
    for stem, label, folder in (("m3_improved_vs_basic", "M3 improved − M3 basic", test),
                                ("m3_basic_vs_hmm", "M3 basic − HMM", test),
                                ("m3_improved_vs_hmm", "M3 improved − HMM", test),
                                ("tau_vs_improved", "threshold − M3 improved", tau_dir)):
        out.append(f"| {label} | " + " | ".join(
            diff_cell(load(folder / f"{stem}_{m}_test_amb.json")["blocks"][PRIMARY[m]]) for m in MODES) + " |")
    return out + [""]


def fit_text(seconds: float) -> str:
    """Seconds with one decimal, rounded half-up like every other number in the report."""
    value = Decimal(repr(seconds)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return f"{value:,}"


def sec_runtime(runtime: Path, manifests: Path) -> list[str]:
    out = ["## 10. Runtime", "",
           ("Prediction speed on one laptop (Intel64 Family 6 Model 158, Python 3.13.1, one process, no other jobs): "
            "median of 3 passes over the primary validation set after a 200-record warm-up, model loading excluded "
            "(the same procedure and machine as the HMM rows in EVALUATION_RESULTS.md §7). Fit time is from each "
            "model's training manifest, measured on Member 3's machine (AMD64, 12 logical CPUs), so it is not "
            "comparable with the HMM's fit time."), "",
           "| Model | Mode | Fit s (M3 machine) | Predict s | Tokens/s | ms/utterance |", "|---|---|---|---|---|---|"]
    for preset in ("basic", "improved"):
        for m in MODES:
            r = load(runtime / f"crf_{preset}_{m}.json")
            fit = load(manifests / f"crf_mode_{m.lower()}_{preset}_full_01.manifest.json")["training_duration_seconds"]
            out.append(f"| CRF {preset} | {m} | {fit_text(fit)} | {r['median_seconds']:.2f} | "
                       f"{r['tokens_per_second']:,.0f} | {r['ms_per_sequence']:.3f} |")
    return out + ["", ("The improved preset predicts at about half the basic preset's speed: under 0.32 ms per "
                       "utterance for basic, under 0.67 ms for improved."), ""]


def build(root: Path, manifests: Path, out_md: Path) -> None:
    body = (sec_threshold(root / "threshold") + sec_leakage(root / "leakage")
            + sec_test_amb(root / "hmm", root / "crf_test", root / "threshold")
            + sec_runtime(root / "runtime", manifests))
    out_md.write_text("\n".join(body), encoding="utf-8")
    print(f"wrote {out_md}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write sections 7-10 of the CRF results report as Markdown.")
    parser.add_argument("--scores-root", type=Path, default=ROOT / "results" / "scores",
                        help="folder holding threshold/, leakage/, crf_test/, hmm/ and runtime/")
    parser.add_argument("--manifests", type=Path, required=True,
                        help="folder with Member 3's crf_mode_<m>_<preset>_full_01.manifest.json files")
    parser.add_argument("--out", type=Path, required=True, help="Markdown file to write")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(lambda: build(args.scores_root, args.manifests, args.out))


if __name__ == "__main__":
    sys.exit(main())
