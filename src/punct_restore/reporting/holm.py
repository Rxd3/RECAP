"""Holm's step-down correction over the paired bootstrap comparisons in results/scores.

This is the "Multiple comparisons (Holm)" check of docs/EVALUATION_RESULTS_CRF.md. Each p-value is two-sided and
comes from the share of bootstrap resamples in which B beats A in the primary block of a compare file:
``p = 2 * min(share, 1 - share)``. A p of 0 means that no resample went the other way. Holm's step-down
correction is applied at alpha 0.05 within two families: the 21 test comparisons and the 18 ablations.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from punct_restore.file_io import read_object, require_object

ALPHA = 0.05
MODES = ("A", "B", "C")
# (label, compare file under the scores root); every file holds B minus A, as its model_a and model_b say
TEST_COMPARISONS = (
    ("improved - basic", "crf_test/m3_improved_vs_basic_{mode}_test.json"),
    ("basic - hmm", "crf_test/m3_basic_vs_hmm_{mode}_test.json"),
    ("improved - hmm", "crf_test/m3_improved_vs_hmm_{mode}_test.json"),
    ("threshold - improved", "threshold/tau_vs_improved_{mode}_test.json"),
    ("prior-adjusted threshold - improved", "threshold_prior/tg_vs_improved_{mode}_test.json"),
    ("hidden-event LM - hmm", "hidden_event_lm/hel_vs_hmm_{mode}_test.json"),
    ("hidden-event LM - improved", "hidden_event_lm/hel_vs_crf_improved_{mode}_test.json"),
)
ABLATION_GROUPS = ("casing", "clause", "ngrams", "position", "question", "window2")
ABLATION_FILE = "ablations/mode_{mode}/full_vs_improved_minus_{group}.json"
METHOD = (
    "p = 2 * min(share_b_better, 1 - share_b_better) from the primary block of each compare file (difference = B "
    "minus A; the shares have a resolution of 0.001, and p = 0 means no resample went the other way); p_holm is "
    "Holm's step-down adjustment within the family; survives = p_holm <= alpha"
)
# what a compare file must hold for a Holm entry (the primary block is named by the file itself)
_COMPARE = {"primary_block": str, "blocks": {}}


def two_sided_p(share_b_better: float) -> float:
    """The two-sided bootstrap p-value of a share of resamples in which B beats A."""
    return round(2 * min(share_b_better, 1 - share_b_better), 6)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm's step-down adjusted p-values, in the order of ``p_values`` (each at most 1)."""
    count = len(p_values)
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(sorted(range(count), key=p_values.__getitem__)):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = round(min(1.0, running), 6)
    return adjusted


def read_primary(path: Path) -> tuple[str, float, float]:
    """(primary block name, difference, share of resamples in which B beats A) of a compare file.

    Raises:
        OSError: If the file cannot be read.
        ValueError: One line that starts with the file name: not valid JSON, a missing or mistyped key, or a share
            outside 0 to 1.
    """
    doc = read_object(path, _COMPARE)
    name = doc["primary_block"]
    schema = {name: {"difference": float, "difference_ci": {"share_b_better": float}}}
    block = require_object(path.name, doc["blocks"], schema, "blocks")[name]
    share = float(block["difference_ci"]["share_b_better"])
    if not 0 <= share <= 1:
        raise ValueError(f"{path.name}: share_b_better is not between 0 and 1")
    return name, float(block["difference"]), share


def _entry(scores_root: Path, label: dict[str, str], rel: str) -> dict[str, Any]:
    block, difference, share = read_primary(scores_root / rel)
    return {**label, "file": rel, "block": block, "difference": difference, "share_b_better": share,
            "p": two_sided_p(share)}


def _family(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """The family document: each entry gets p_holm and survives."""
    adjusted = holm_adjust([entry["p"] for entry in entries])
    rows = [{**entry, "p_holm": p, "survives": p <= ALPHA} for entry, p in zip(entries, adjusted, strict=True)]
    return {"count": len(rows), "survivors": sum(row["survives"] for row in rows), "comparisons": rows}


def comparisons_family(scores_root: Path) -> dict[str, Any]:
    """The 21 test comparisons (seven comparisons in each of modes A, B and C)."""
    return _family([_entry(scores_root, {"comparison": label, "mode": mode}, rel.format(mode=mode))
                    for label, rel in TEST_COMPARISONS for mode in MODES])


def ablation_family(scores_root: Path) -> dict[str, Any]:
    """The 18 ablations (six feature groups in each of modes A, B and C), on validation."""
    return _family([_entry(scores_root, {"group": group, "mode": mode}, ABLATION_FILE.format(mode=mode, group=group))
                    for group in ABLATION_GROUPS for mode in MODES])


def holm_text(scores_root: Path) -> str:
    """The text of holm.json (LF line ends, one trailing newline), computed from the compare files in
    ``scores_root``."""
    families = {"test_comparisons": comparisons_family(scores_root), "ablations": ablation_family(scores_root)}
    document = {"method": METHOD, "alpha": ALPHA, "families": families}
    return json.dumps(document, indent=1) + "\n"
