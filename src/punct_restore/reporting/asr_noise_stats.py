"""Constants, exact seed statistics and number formatting shared by the ASR noise report modules.

Seed means are kept exact (``Decimal``, from the six-digit scores in the evaluator files) and are rounded once,
half-up, when a number is displayed: a mean is never rounded to six digits first, so a difference of two means
cannot pick up a second rounding. Used by asr_noise_table.py, asr_noise_report.py and asr_noise_findings.py, which
sit beside it in punct_restore.reporting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from punct_restore.data.asr_noise import V2_WEIGHTS
from punct_restore.reporting.build_tables import half_up, pct

WERS = ("0.05", "0.10", "0.15")
SEEDS = (2026, 2027, 2028)
MODELS = ("hmm", "crf_basic", "crf_improved")
MODEL_LABELS = {"hmm": "HMM", "crf_basic": "CRF basic", "crf_improved": "CRF improved"}
V1_WER = "0.15"  # the one WER at which v1 was run for every seed
V1_KEY = f"wer{V1_WER}"
V2_SHARE_KEYS = ("substitution", "deletion", "insertion", "swap")
V1_SHARE_KEYS = ("substitution", "deletion", "swap")  # v1 makes no insertions, so its summaries have no such entry
SHARE_OF_KIND = {"replace": "substitution", "delete": "deletion", "insert": "insertion", "swap": "swap"}
V2_TARGET_SHARE = {SHARE_OF_KIND[kind]: weight / sum(V2_WEIGHTS.values()) for kind, weight in V2_WEIGHTS.items()}
EN, MINUS, ARROW = "–", "−", "→"  # en dash, minus sign, right arrow
NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


@dataclass(frozen=True)
class SeedStats:
    """Mean, min-max range, and clean-to-noisy drop across seeds for one model, mode, and WER."""

    per_seed: dict[str, float]
    mean: float
    min: float
    max: float
    range: float
    drop_from_clean: float
    mean_pct: str
    min_pct: str
    max_pct: str
    range_pct: str
    drop_from_clean_pct: str
    cell: str


def dec(value: float) -> Decimal:
    """The shortest decimal of a number, so 0.67985 stays 0.67985 and not its binary value."""
    return Decimal(repr(value))


def mean_of(values: Sequence[Decimal]) -> Decimal:
    """The exact mean (28 significant digits, so exact for scores with six decimals and three seeds)."""
    return sum(values, Decimal(0)) / len(values)


def pts(value: Decimal) -> str:
    """``value`` (a fraction) as percentage points, rounded once half-up to two decimals: 0.12345 -> 12.35."""
    return pct(str(value))


def spread_summary(values: Sequence[Decimal]) -> dict[str, Any]:
    """Mean, min and max of ``values`` as floats, their displayed percentages, and the ``mean (min-max)`` cell."""
    mean, low, high = mean_of(values), min(values), max(values)
    mean_p, low_p, high_p = pts(mean), pts(low), pts(high)
    return {
        "mean": float(mean),
        "min": float(low),
        "max": float(high),
        "mean_pct": mean_p,
        "min_pct": low_p,
        "max_pct": high_p,
        "cell": f"{mean_p} ({low_p}{EN}{high_p})",
    }


def compute_seed_stats(clean_f1: float, seed_scores: Mapping[int, float]) -> SeedStats:
    """Aggregate per-seed macro F1 scores into mean, min, max, range, and drop from clean."""
    values = [dec(seed_scores[s]) for s in SEEDS]
    base = spread_summary(values)
    spread = max(values) - min(values)
    drop = dec(clean_f1) - mean_of(values)
    return SeedStats(
        per_seed={str(s): seed_scores[s] for s in SEEDS},
        mean=base["mean"],
        min=base["min"],
        max=base["max"],
        range=float(spread),
        drop_from_clean=float(drop),
        mean_pct=base["mean_pct"],
        min_pct=base["min_pct"],
        max_pct=base["max_pct"],
        range_pct=pts(spread),
        drop_from_clean_pct=pts(drop),
        cell=base["cell"],
    )


def cell_mean(cell: Mapping[str, Any]) -> Decimal:
    """The exact mean of a ``SeedStats`` dict, from its per-seed scores."""
    return mean_of([dec(score) for score in cell["per_seed"].values()])


def signed_pts(value: Decimal | float) -> str:
    """``value`` (a fraction) as percentage points: rounded once, half-up, with a sign and U+2212 for minus."""
    points = Decimal(pct(value))
    return f"{MINUS if points < 0 else '+'}{abs(points)}"  # a rounded -0.00 is not negative: it prints +0.00


def wer_label(wer: str) -> str:
    """``"0.05"`` -> ``"5%"``."""
    return f"{half_up(Decimal(wer) * 100, 0)}%"


def count_word(count: int) -> str:
    """``3`` -> ``"three"``; numbers above ten stay digits."""
    return NUMBER_WORDS[count] if count < len(NUMBER_WORDS) else str(count)


def join_and(items: Sequence[str]) -> str:
    """``a``; ``a and b``; ``a, b, and c``."""
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]
