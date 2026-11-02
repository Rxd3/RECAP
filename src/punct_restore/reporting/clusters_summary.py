"""The K choice of the word-cluster comparison, from the six compare files in results/scores/clusters.

``clusters_summary.json`` repeats the numbers of ``compare_<mode>_k<K>.json`` (improved CRF with vs without the
cluster features, K = 100 and 300, modes A, B and C) and adds the one choice the compare files do not hold: the K
with the higher mean primary macro-F1 difference over the three modes (a tie goes to the smaller K).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from punct_restore.file_io import read_object, require_object

MODES = ("A", "B", "C")
KS = (100, 300)
WHAT = "improved_plus_clusters vs improved (same seeded 50k default_train sample, primary validation lists)"
BASELINE = "RECAP_m3_ablations/mode_<M>/improved/validation.jsonl"  # where the baseline predictions were made
CAVEAT = ("K is chosen on the same validation lists the comparison is reported on, so the chosen K's difference is "
          "optimistically biased; no test data was used")
RULE = ("one K for all modes: the K with the higher mean primary macro-F1 difference over modes A, B, C on "
        "validation; ties go to the smaller K")
_BOOTSTRAP = {"resamples": int, "seed": int, "unit": str}
_INTERVAL = {"low": float, "high": float}
_PRIMARY = {"difference": float, "macro_f1_a": float, "macro_f1_b": float,
            "difference_ci": {**_INTERVAL, "share_b_better": float, **_BOOTSTRAP}}
_CLASS = {"difference": float, "difference_ci": _INTERVAL}


def compare_name(mode: str, k: int) -> str:
    return f"compare_{mode}_k{k}.json"


def class_labels(mode: str) -> tuple[str, ...]:
    """The three per-class blocks the summary keeps: mode C classes carry their casing."""
    return tuple(f"{mark}|LOWER" if mode == "C" else mark for mark in ("COMMA", "PERIOD", "QUESTION"))


def lf_sha256(path: Path) -> str:
    """SHA-256 of the file with CRLF line ends read as LF, as git stores it (a Windows checkout may hold CRLF)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _interval(cell: dict[str, Any]) -> dict[str, Any]:
    return {"ci_high": cell["difference_ci"]["high"], "ci_low": cell["difference_ci"]["low"],
            "difference": cell["difference"]}


def read_result(path: Path, mode: str, k: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """One entry of ``results_by_k`` read from a compare file, and the bootstrap settings that file records.

    Raises:
        OSError: If the file cannot be read.
        ValueError: One line that starts with the file name: not valid JSON, or a missing or mistyped key.
    """
    doc = read_object(path, {"primary_block": str, "blocks": {}, "classes": {}})
    block = doc["primary_block"]
    primary = require_object(path.name, doc["blocks"], {block: _PRIMARY}, "blocks")[block]
    by_class = {label: require_object(path.name, doc["classes"], {label: {block: _CLASS}}, "classes")[label][block]
                for label in class_labels(mode)}
    entry = {
        "classes": {label: _interval(cell) for label, cell in by_class.items()},
        "compare_file": path.name, "compare_sha256": lf_sha256(path), "map_file": f"maps/clusters_{mode}_k{k}.json",
        "primary": {**_interval(primary), "macro_f1_a": primary["macro_f1_a"], "macro_f1_b": primary["macro_f1_b"],
                    "share_b_better": primary["difference_ci"]["share_b_better"]},
        "primary_block": block,
    }
    return entry, {key: primary["difference_ci"][key] for key in _BOOTSTRAP}


def read_all(folder: Path) -> tuple[dict[int, dict[str, dict[str, Any]]], dict[str, Any]]:
    """The ``results_by_k`` tree of the six compare files, and the one set of bootstrap settings they all share."""
    results: dict[int, dict[str, dict[str, Any]]] = {k: {} for k in KS}
    settings: dict[str, Any] | None = None
    for k in KS:
        for mode in MODES:
            path = folder / compare_name(mode, k)
            results[k][mode], found = read_result(path, mode, k)
            if settings not in (None, found):
                raise ValueError(f"{path.name}: bootstrap settings differ from {compare_name(MODES[0], KS[0])}")
            settings = found
    return results, settings or {}


def clusters_summary_text(scores_root: Path) -> str:
    """The text of clusters_summary.json (LF line ends, keys sorted, one trailing newline).

    Reads ``clusters/compare_<mode>_k<K>.json`` under ``scores_root``.
    """
    results, bootstrap = read_all(scores_root / "clusters")
    mean = {k: sum(r["primary"]["difference"] for r in by_mode.values()) / len(MODES) for k, by_mode in results.items()}
    summary = {
        "baseline_predictions": BASELINE, "bootstrap": bootstrap,
        "k_selection": {"caveat": CAVEAT, "chosen_k": max(KS, key=lambda k: (mean[k], -k)),
                        "mean_primary_difference": {str(k): value for k, value in mean.items()}, "rule": RULE},
        "results_by_k": {str(k): by_mode for k, by_mode in results.items()}, "what": WHAT,
    }
    return json.dumps(summary, indent=1, sort_keys=True) + "\n"
