"""A tiny generated ``results/scores`` tree for the reporting-script tests (no real data).

``write_scores_root`` writes, for modes A, B and C, only the blocks and keys that the scripts in
``scripts/reporting`` read, in the committed layout. ``write_manifests`` writes the CRF training
manifests that the runtime section reads. Not a test module: pytest does not collect it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODES = ("A", "B", "C")
PRIMARY = {"A": "punctuation/all", "B": "punctuation/all", "C": "joint/unmasked"}
PUNCT = {"A": ["O", "COMMA"], "B": ["O", "COMMA", "DASH"], "C": ["O", "COMMA"]}
JOINT = ["O|LOWER", "COMMA|FIRST_UPPER"]
CASING = ["LOWER", "FIRST_UPPER"]
LEVEL = {"shortcut": 0.1, "hmm": 0.2, "basic": 0.3, "improved": 0.4, "tau": 0.5}
SPLIT_SHIFT = {"validation": 0.0, "test": 0.01, "test_amb": 0.02}
EXTRA_C = {"joint/unmasked": (JOINT, 0.0), "casing/unmasked": (CASING, 0.01),
           "casing/unmasked_non_initial": (CASING, 0.02), "casing/all": (CASING, 0.03)}
# (difference, ci low, ci high) of the primary block; non_final is shifted up by 0.01
COMPARES = {"m3_improved_vs_basic": (0.1, 0.05, 0.15), "m3_basic_vs_hmm": (-0.02, -0.05, 0.01),
            "m3_improved_vs_hmm": (0.2, 0.1, 0.3), "tau_vs_improved": (0.0, -0.01, 0.01)}
GROUPS = ("window2", "casing", "zeta")  # zeta is not a known group, so it sorts last
LEAK_KEYS = ("official_all", "leakfree", "train_video", "other_leak", "quality", "native_all")
TAU_CHOSEN = {"A": 0.5, "B": 0.05, "C": 0.95}
FIT_SECONDS = {"basic": {"A": 1234.55, "B": 2000.0, "C": 3000.25},
               "improved": {"A": 2469.1, "B": 4000.0, "C": 6000.5}}
RUNTIME = {"basic": {"median_seconds": 1.234, "tokens_per_second": 123456.7, "ms_per_sequence": 0.2341},
           "improved": {"median_seconds": 2.5, "tokens_per_second": 61728.4, "ms_per_sequence": 0.5}}


def dump(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def block(level: float, labels: list[str]) -> dict[str, Any]:
    """One evaluator block with every key the tables read; each class scores `level`."""
    size = len(labels)
    return {
        "macro_f1": level,
        "macro_f1_ci": {"low": round(level - 0.1, 6), "high": round(level + 0.1, 6)},
        "micro": {"precision": round(level + 0.01, 6), "recall": round(level + 0.02, 6),
                  "f1": round(level + 0.03, 6)},
        "accuracy": 0.9,
        "labels": labels,
        "macro_over": labels,
        "per_class": {label: {"precision": level, "recall": level, "f1": level, "support": 1000 * (i + 1)}
                      for i, label in enumerate(labels)},
        "confusion": [[10 if i == j else 1 for j in range(size)] for i in range(size)],
    }


def score_doc(mode: str, level: float) -> dict[str, Any]:
    blocks = {"punctuation/all": block(level, PUNCT[mode]),
              "punctuation/non_final": block(round(level - 0.05, 6), PUNCT[mode])}
    if mode == "C":
        for name, (labels, shift) in EXTRA_C.items():
            blocks[name] = block(round(level + shift, 6), labels)
    return {"blocks": blocks}


def compare_doc(mode: str, difference: float, low: float, high: float) -> dict[str, Any]:
    def diff(shift: float) -> dict[str, Any]:
        return {"difference": round(difference + shift, 6),
                "difference_ci": {"low": round(low + shift, 6), "high": round(high + shift, 6)}}
    return {"blocks": {PRIMARY[mode]: diff(0.0), "punctuation/non_final": diff(0.01)}}


def errors_doc() -> dict[str, Any]:
    return {"errors_total": 200,
            "pair_counts": [{"gold": "O", "pred": "COMMA", "count": 50},
                            {"gold": "COMMA", "pred": "O", "count": 30}]}


def _level(model: str, split: str) -> float:
    return round(LEVEL[model] + SPLIT_SHIFT[split], 6)


def _write_refs(folder: Path, mode: str) -> None:
    for split in SPLIT_SHIFT:
        for model in ("shortcut", "hmm"):
            dump(folder / f"{model}_{mode}_{split}.json", score_doc(mode, _level(model, split)))


def _write_crf(folder: Path, mode: str, split: str) -> None:
    suffix = "" if split == "validation" else f"_{split}"
    for preset in ("basic", "improved"):
        dump(folder / f"m3_{preset}_{mode}{suffix}.json", score_doc(mode, _level(preset, split)))
    for stem, (difference, low, high) in COMPARES.items():
        if stem != "tau_vs_improved":
            dump(folder / f"{stem}_{mode}{suffix}.json", compare_doc(mode, difference, low, high))
    if split != "test_amb":
        dump(folder / f"m3_improved_{mode}{suffix}.errors.json", errors_doc())


def _write_threshold(folder: Path, mode: str) -> None:
    index = MODES.index(mode)
    grid = [{"tau": tau, "macro_f1": round(0.3 + tau * 0.4 + 0.01 * index, 6)} for tau in (0.05, 0.5, 0.95)]
    dump(folder / f"tau_{mode}.json", {"tau": TAU_CHOSEN[mode], "grid": grid})
    for split in ("test", "test_amb"):
        dump(folder / f"tau_improved_{mode}_{split}.json", score_doc(mode, _level("tau", split)))
        difference, low, high = COMPARES["tau_vs_improved"]
        dump(folder / f"tau_vs_improved_{mode}_{split}.json", compare_doc(mode, difference, low, high))


def _write_ablations(folder: Path, mode: str) -> None:
    dump(folder / "improved.json", score_doc(mode, LEVEL["improved"]))
    for index, group in enumerate(GROUPS, start=1):
        gap = round(0.02 * index, 6)
        dump(folder / f"improved_minus_{group}.json", score_doc(mode, round(LEVEL["improved"] - gap, 6)))
        dump(folder / f"full_vs_improved_minus_{group}.json",
             compare_doc(mode, gap, round(gap - 0.01, 6), round(gap + 0.01, 6)))


def _write_leakage(folder: Path) -> None:
    for index, key in enumerate(LEAK_KEYS):
        for name, model in (("hmm", "hmm"), ("crf", "improved")):
            doc = score_doc("A", round(LEVEL[model] + 0.01 * index, 6))
            dump(folder / f"{name}_A_test_{key}.json",
                 {"counts": {"sequences": 1000 * (index + 1) + 234}, "blocks": doc["blocks"]})


def write_scores_root(root: Path) -> Path:
    """Write the whole tree under `root` and return it."""
    for mode in MODES:
        _write_refs(root / "hmm", mode)
        _write_crf(root / "crf_validation", mode, "validation")
        _write_crf(root / "crf_test", mode, "test")
        _write_crf(root / "crf_test", mode, "test_amb")
        _write_threshold(root / "threshold", mode)
        _write_ablations(root / "ablations" / f"mode_{mode}", mode)
        for preset in ("basic", "improved"):
            dump(root / "runtime" / f"crf_{preset}_{mode}.json", RUNTIME[preset])
    _write_leakage(root / "leakage")
    return root


def write_manifests(folder: Path) -> Path:
    """Write the six CRF training manifests (`training_duration_seconds` only) into `folder`."""
    for preset, seconds in FIT_SECONDS.items():
        for mode, value in seconds.items():
            dump(folder / f"crf_mode_{mode.lower()}_{preset}_full_01.manifest.json",
                 {"training_duration_seconds": value})
    return folder
