"""Sections 7-10 of docs/EVALUATION_RESULTS_CRF.md (threshold grid, leakage, test_amb, runtime) from evaluator JSON.

Usage: recap build-m3-final-sections [--scores-root R] [--manifests DIR] --out OUT_MD
Scores come from R (default results/scores): threshold/, crf_validation/, leakage/, crf_test/, hmm/ and
runtime/. The CRF fit times and the training machine come from Member 3's training manifests
crf_mode_<m>_<preset>_full_01.manifest.json in DIR (default R/crf_training_manifests). OUT_MD must not exist.

The figures quoted in the prose are derived from those files (the standard-decoding scores, the utterance
count, the leakage gains, the laptop and the speed bounds), except three fixed descriptions written into the
text: the τ grid range, the quality group being one video, and test_amb sharing 25 videos with test.
Files that must agree (the utterance counts of the two leakage runs, the hardware of the six runtime files, the
machine of the six manifests) are checked.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Iterable
from decimal import Decimal
from pathlib import Path
from typing import Any

from punct_restore.file_io import Number, is_finite_number, read_object, require_object
from punct_restore.paths import REPO_ROOT as ROOT
from punct_restore.reporting._cli import check_out, check_scores_folder, run, write_new_text
from punct_restore.reporting.build_tables import (
    CELL,
    DIFF,
    MODES,
    PRIMARY,
    SCORE,
    cell,
    diff_cell,
    half_up,
    load,
    pct,
    read_block,
    read_scores,
)

PRESETS = ("basic", "improved")
LEAK_BLOCK = "punctuation/all"
SECONDS = Number("a non-negative number", 0.0)  # a fit time


def read_tau(path: Path) -> dict[str, Any]:
    """A tau_<mode>.json file: the chosen ``tau`` and the ``grid`` of {tau, macro_f1} points."""
    doc = read_object(path, {"tau": float, "grid": list})
    for number, point in enumerate(doc["grid"]):
        require_object(path.name, point, {"tau": float, "macro_f1": SCORE}, f"grid.{number}")
    return doc


def sec_threshold(root: Path) -> list[str]:
    tau_dir = root / "threshold"
    files = {m: read_tau(tau_dir / f"tau_{m}.json") for m in MODES}
    grids = {m: {round(g["tau"], 2): g["macro_f1"] for g in files[m]["grid"]} for m in MODES}
    for m in MODES[1:]:
        if grids[m].keys() != grids[MODES[0]].keys():
            raise ValueError(f"tau_{m}.json: the grid does not have the taus of tau_{MODES[0]}.json")
    chosen = {m: files[m]["tau"] for m in MODES}
    viterbi = {m: pct(read_block(root / "crf_validation" / f"m3_improved_{m}.json", PRIMARY[m], CELL)["macro_f1"])
               for m in MODES}
    out = ["## 7. Threshold decoding (Member 5 extension)", "",
           ("For each token, the improved CRF's marginals give P(O); in mode C, P(O) sums every `O|*` label. If "
            "1 − P(O) > τ the token gets its most probable non-O label, otherwise its most probable O label "
            "(`src/punct_restore/models/threshold.py`). τ was tuned per mode on the primary validation list "
            "(`scripts/tune_threshold.py tune`, grid 0.05–0.95), then test was predicted once with the chosen τ: "
            f"A {chosen['A']}, B {chosen['B']}, C {chosen['C']}. Test effects are in sections 1–2."), "",
           ("Validation primary macro F1 % per τ (bold = chosen; standard Viterbi decoding scores "
            f"A {viterbi['A']}, B {viterbi['B']}, C {viterbi['C']}):"),
           "", "| τ | A | B | C |", "|---|---|---|---|"]
    for tau in sorted(grids["A"]):
        row = [f"**{pct(grids[m][tau])}**" if tau == round(chosen[m], 2) else pct(grids[m][tau]) for m in MODES]
        out.append(f"| {tau:.2f} | " + " | ".join(row) + " |")
    return out + [""]


def leak_gain(docs: dict[tuple[str, str], Any], model: str, key: str) -> Decimal:
    """Points by which ``model`` scores higher on leakage subset ``key`` than on the leak-free set, exactly."""
    def macro(subset: str) -> Decimal:
        return Decimal(str(docs[(model, subset)]["blocks"][LEAK_BLOCK]["macro_f1"]))

    return 100 * (macro(key) - macro("leakfree"))


def whole_gain(docs: dict[tuple[str, str], Any], model: str, key: str) -> int:
    """``leak_gain`` rounded half-up to whole points (0 when a small loss rounds to 0)."""
    return int(half_up(leak_gain(docs, model, key)))


def points(n: int) -> str:
    return f"{n} point" if n == 1 else f"{n} points"


def trained_phrase(n: int) -> str:
    """The CRF on videos it was trained on versus the leak-free set, ``n`` whole points apart."""
    if n > 0:
        return f"scores about {points(n)} higher than on the leak-free set"
    if n == 0:
        return "scores about the same as on the leak-free set"
    return f"scores about {points(-n)} lower than on the leak-free set"


def official_phrase(n: int) -> str:
    """What scoring the official test as-is does to the CRF, given its ``n``-point gain over the leak-free set."""
    if n > 0:
        return f"would overstate the CRF by about {points(n)}"
    if n == 0:
        return "would not change the CRF's score"
    return f"would understate the CRF by about {points(-n)}"


def leakage_prose(docs: dict[tuple[str, str], Any]) -> str:
    hmm_trained = whole_gain(docs, "hmm", "train_video")
    hmm = "the HMM does not gain at all" if hmm_trained <= 0 else f"the HMM gains about {points(hmm_trained)}"
    return ("On utterances from videos it was trained on, the improved CRF "
            f"{trained_phrase(whole_gain(docs, 'crf', 'train_video'))}; {hmm}. "
            f"Reporting the official test as-is {official_phrase(whole_gain(docs, 'crf', 'official_all'))}.")


def leak_file(model: str, key: str) -> str:
    """The name of the leakage file of ``model`` (hmm or crf) for the scored set ``key``."""
    return f"{model}_A_test_{key}.json"


def official_count(docs: dict[tuple[str, str], Any]) -> int:
    counts = {model: docs[(model, "official_all")]["counts"]["sequences"] for model in ("hmm", "crf")}
    if counts["hmm"] != counts["crf"]:
        raise ValueError(f"{leak_file('crf', 'official_all')}: counts.sequences differs from "
                         f"{leak_file('hmm', 'official_all')}")
    return counts["hmm"]


def sec_leakage(supp: Path) -> list[str]:
    rows = [("Official test, all utterances", "canonical", "official_all"),
            ("Leakage-controlled test (kept)", "canonical", "leakfree"),
            ("Excluded: video also in the training set", "canonical", "train_video"),
            ("Excluded: other leakage (video shared with validation only, or duplicate text)", "canonical",
             "other_leak"),
            ("Excluded: quality filter (low comma density or TTS source)", "canonical", "quality"),
            ("Official test, all utterances", "SponSpeech native", "native_all")]
    docs = {(model, key): read_scores(supp / leak_file(model, key), {LEAK_BLOCK: CELL},
                                      {"counts": {"sequences": int}})
            for _, _, key in rows for model in ("hmm", "crf")}
    out = ["## 8. Leakage: official vs leakage-controlled test (mode A)", "",
           ("Supplementary. The frozen improved CRF and the HMM predicted all "
            f"{official_count(docs):,} official test utterances once "
            "(the CRF's predictions on the primary list are identical to section 1's). The excluded utterances are "
            "the official test minus `data/splits/sponspeech-m1-v2/leakage_controlled/test.ids`, split by why they "
            "were excluded (`leakage_controlled/test_excluded.tsv`). \"Training set\" means a video with an "
            "utterance in `default_train`, which both models were trained on. Macro F1 % (95% CI; the quality group "
            "is one video, so its interval is degenerate)."), "",
           "| Scored set | Gold labels | Utterances | HMM | CRF improved |", "|---|---|---|---|---|"]
    for label, gold, key in rows:
        h, c = docs[("hmm", key)], docs[("crf", key)]
        out.append(f"| {label} | {gold} | {h['counts']['sequences']:,} | {cell(h['blocks'][LEAK_BLOCK])} | "
                   f"{cell(c['blocks'][LEAK_BLOCK])} |")
    return out + ["", leakage_prose(docs), ""]


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
        out.append(f"| {name} | " + " | ".join(cell(read_block(f(m), PRIMARY[m], CELL)) for m in MODES) + " |")
    out += ["", "| Paired difference (test_amb) | A | B | C |", "|---|---|---|---|"]
    for stem, label, folder in (("m3_improved_vs_basic", "M3 improved − M3 basic", test),
                                ("m3_basic_vs_hmm", "M3 basic − HMM", test),
                                ("m3_improved_vs_hmm", "M3 improved − HMM", test),
                                ("tau_vs_improved", "threshold − M3 improved", tau_dir)):
        out.append(f"| {label} | " + " | ".join(
            diff_cell(read_block(folder / f"{stem}_{m}_test_amb.json", PRIMARY[m], DIFF)) for m in MODES) + " |")
    return out + [""]


def fit_text(seconds: float) -> str:
    """Seconds with one decimal, rounded half-up like every other number in the report."""
    return half_up(seconds, 1, grouping=True)


def agreed[T](values: Iterable[T], names: Iterable[str], what: str) -> T:
    """The one value all files hold for ``what`` (``values[i]`` is in the file ``names[i]``).

    Raises:
        ValueError: ``<file>: <what> differs from <first file>`` for the first file that holds another value.
    """
    pairs = list(zip(names, values, strict=True))
    first_name, first_value = pairs[0]
    for name, value in pairs[1:]:
        if value != first_value:
            raise ValueError(f"{name}: {what} differs from {first_name}")
    return first_value


def is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def facts(docs: dict[tuple[str, str], Any], names: dict[tuple[str, str], str], section: str, key: str,
          valid: Callable[[Any], bool], expected: str) -> list[Any]:
    """``doc[section][key]`` of every file; a one-line ValueError names the first file where it is not ``expected``."""
    found = []
    for which, doc in docs.items():
        holder = doc.get(section) if isinstance(doc, dict) else None  # a missing section is named below
        value = holder.get(key) if isinstance(holder, dict) else None
        if not valid(value):
            raise ValueError(f"{names[which]}: {section}.{key} must be {expected}")
        found.append(value)
    return found


def check_speeds(runs: dict[tuple[str, str], Any], names: dict[tuple[str, str], str]) -> None:
    """Every runtime file's speeds are positive numbers (a zero speed would divide by zero below), and each improved
    speed over the basic one of its mode is a number (1e300 over 1e-300 overflows)."""
    for which, record in runs.items():
        for field in ("median_seconds", "tokens_per_second", "ms_per_sequence"):
            value: Any = record.get(field) if isinstance(record, dict) else None
            if not (is_finite_number(value) and value > 0):  # NaN, Infinity and 1e999 are no speed either
                raise ValueError(f"{names[which]}: {field} must be a positive number")
    for mode in MODES:
        better, basic = ("improved", mode), ("basic", mode)
        if not math.isfinite(runs[better]["tokens_per_second"] / runs[basic]["tokens_per_second"]):
            raise ValueError(f"{names[better]}: tokens_per_second cannot be compared with that of {names[basic]}")


def ms_bound(values: Iterable[Any]) -> str:
    """A strict upper bound with margin on the slowest mode's ms/utterance, for "under X ms".

    Rounding the slowest value half-up to 2 decimals moves it by at most 0.005; adding 0.01 keeps the bound above it.
    """
    slowest = max(Decimal(repr(v)) for v in values)
    return f"{Decimal(half_up(slowest, 2)) + Decimal('0.01'):.2f}"


def speed_phrase(runs: dict[tuple[str, str], Any]) -> str:
    ratios = [runs[("improved", m)]["tokens_per_second"] / runs[("basic", m)]["tokens_per_second"] for m in MODES]
    if all(0.4 <= ratio <= 0.6 for ratio in ratios):
        return "about half the basic preset's speed"
    low, high = (half_up(Decimal(repr(ratio)) * 100) for ratio in (min(ratios), max(ratios)))
    return f"{low}%–{high}% of the basic preset's speed"


def sec_runtime(runtime: Path, manifests: Path) -> list[str]:
    run_names = {(p, m): f"crf_{p}_{m}.json" for p in PRESETS for m in MODES}
    sheet_names = {(p, m): f"crf_mode_{m.lower()}_{p}_full_01.manifest.json" for p in PRESETS for m in MODES}
    runs = {key: load(runtime / name) for key, name in run_names.items()}
    sheets = {key: load(manifests / name, {"training_duration_seconds": SECONDS}) for key, name in sheet_names.items()}
    check_speeds(runs, run_names)
    run_files, sheet_files = list(run_names.values()), list(sheet_names.values())
    processors = facts(runs, run_names, "hardware", "processor", is_text, "a string")
    cpu = agreed([text.split(" Stepping")[0] for text in processors], run_files, "processor")
    python = agreed(facts(runs, run_names, "hardware", "python", is_text, "a string"), run_files, "Python version")
    machine = agreed(facts(sheets, sheet_names, "environment", "machine", is_text, "a string"), sheet_files, "machine")
    cpus = agreed(facts(sheets, sheet_names, "environment", "cpu_count", is_count, "a positive integer"),
                  sheet_files, "CPU count")
    out = ["## 10. Runtime", "",
           (f"Prediction speed on one laptop ({cpu}, Python {python}, one process, no other jobs): "
            "median of 3 passes over the primary validation set after a 200-record warm-up, model loading excluded "
            "(the same procedure and machine as the HMM rows in EVALUATION_RESULTS.md §7). Fit time is from each "
            f"model's training manifest, measured on Member 3's machine ({machine}, {cpus} logical CPUs), so it is "
            "not comparable with the HMM's fit time."), "",
           "| Model | Mode | Fit s (M3 machine) | Predict s | Tokens/s | ms/utterance |", "|---|---|---|---|---|---|"]
    for preset in PRESETS:
        for m in MODES:
            r, fit = runs[(preset, m)], sheets[(preset, m)]["training_duration_seconds"]
            out.append(f"| CRF {preset} | {m} | {fit_text(fit)} | {half_up(r['median_seconds'], 2)} | "
                       f"{half_up(r['tokens_per_second'], 0, grouping=True)} | {half_up(r['ms_per_sequence'], 3)} |")
    bounds = {p: ms_bound(runs[(p, m)]["ms_per_sequence"] for m in MODES) for p in PRESETS}
    return out + ["", (f"The improved preset predicts at {speed_phrase(runs)}: under {bounds['basic']} ms per "
                       f"utterance for basic, under {bounds['improved']} ms for improved."), ""]


def build(root: Path, manifests: Path, out_md: Path) -> None:
    check_out(out_md)  # before reading anything
    check_scores_folder(root)
    body = (sec_threshold(root) + sec_leakage(root / "leakage")
            + sec_test_amb(root / "hmm", root / "crf_test", root / "threshold")
            + sec_runtime(root / "runtime", manifests))
    write_new_text(out_md, "\n".join(body))
    print(f"wrote {out_md}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recap build-m3-final-sections",
        description="Write sections 7-10 of the CRF results report as Markdown.")
    parser.add_argument("--scores-root", type=Path, default=ROOT / "results" / "scores",
                        help="folder holding threshold/, crf_validation/, leakage/, crf_test/, hmm/ and runtime/")
    parser.add_argument("--manifests", type=Path,
                        help="folder with the CRF training manifests crf_mode_<m>_<preset>_full_01.manifest.json "
                             "(default: <scores-root>/crf_training_manifests)")
    parser.add_argument("--out", type=Path, required=True, help="Markdown file to create (must not exist)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifests = args.manifests or args.scores_root / "crf_training_manifests"
    return run(lambda: build(args.scores_root, manifests, args.out))


if __name__ == "__main__":
    sys.exit(main())
