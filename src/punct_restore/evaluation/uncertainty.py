"""Bootstrap confidence intervals grouped by video, owned by Member 5.

Records of one video are not independent, so resampling draws whole videos (metadata.group_id)
with replacement. Every block and, for a comparison, both models share the same draws, so the
intervals are paired. The interval is the 95% percentile interval of the resampled macro F1.
"""

from __future__ import annotations

import random
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from .inputs import Aligned, EvaluationInputError
from .metrics import class_counts, confusion, macro_f1_from_counts
from .views import BlockSpec, block_pairs

DEFAULT_SEED = 2026
MIN_RESAMPLES = 100
MAX_RESAMPLES = 100_000  # every resample's draws are kept in memory
LEVEL = 0.95
QUANTILE_CUTS = 40  # 39 cut points; the first and last are the 2.5% and 97.5% quantiles
UNIT = "group_id"

Counts = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]  # tp, support, predicted


@dataclass(frozen=True)
class BootstrapSpec:
    resamples: int
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        if not MIN_RESAMPLES <= self.resamples <= MAX_RESAMPLES:
            raise ValueError(f"bootstrap resamples must be {MIN_RESAMPLES} to {MAX_RESAMPLES}")


def group_order(records: Sequence[Aligned]) -> tuple[str, ...]:
    """Distinct group IDs in first-seen order; every record must have one."""
    missing = next((record.sample_id for record in records if record.group_id is None), None)
    if missing is not None:
        raise EvaluationInputError(
            f"bootstrap needs metadata.group_id on every scored record; {missing!r} has none")
    return tuple(dict.fromkeys(record.group_id for record in records if record.group_id is not None))


def group_counts(records: Sequence[Aligned], spec: BlockSpec,
                 groups: Sequence[str]) -> tuple[Counts, ...]:
    """Per group: tp/support/predicted for the block's macro classes only."""
    indices = [spec.labels.index(label) for label in spec.macro_over]
    by_group: dict[str, list[Aligned]] = {group: [] for group in groups}
    for record in records:
        by_group[cast(str, record.group_id)].append(record)  # group_order checked that every record has one
    counts: list[Counts] = []
    for group in groups:
        per_class = class_counts(confusion(block_pairs(by_group[group], spec), spec.labels))
        tp, support, predicted = (tuple(vector[i] for i in indices) for vector in per_class)
        counts.append((tp, support, predicted))
    return tuple(counts)


def draw_weights(n_groups: int, bootstrap: BootstrapSpec) -> tuple[Counter[int], ...]:
    """For each resample, how often each group index was drawn (with replacement)."""
    rng = random.Random(bootstrap.seed)
    return tuple(Counter(rng.choices(range(n_groups), k=n_groups))
                 for _ in range(bootstrap.resamples))


def resampled_macro_f1(counts: Sequence[Counts], weights: Sequence[Counter[int]]) -> list[float]:
    """Macro F1 of every resample, summing each drawn group's counts times its draw count."""
    width = len(counts[0][0])
    values = []
    for drawn in weights:
        totals = [[0] * width for _ in range(3)]
        for group, times in drawn.items():
            for total, vector in zip(totals, counts[group], strict=False):
                for i in range(width):
                    total[i] += times * vector[i]
        values.append(macro_f1_from_counts(*totals))
    return values


def interval(values: Sequence[float]) -> tuple[float, float]:
    """95% percentile interval (inclusive quantile method)."""
    cuts = statistics.quantiles(values, n=QUANTILE_CUTS, method="inclusive")
    return cuts[0], cuts[-1]


def _ci(low: float, high: float, bootstrap: BootstrapSpec, n_groups: int) -> dict[str, Any]:
    return {"low": low, "high": high, "level": LEVEL, "resamples": bootstrap.resamples,
            "seed": bootstrap.seed, "unit": UNIT, "groups": n_groups}


def block_intervals(records: Sequence[Aligned], specs: Sequence[BlockSpec],
                    bootstrap: BootstrapSpec) -> dict[str, dict[str, Any]]:
    """Macro F1 confidence interval for every block, all blocks sharing the same draws."""
    groups = group_order(records)
    weights = draw_weights(len(groups), bootstrap)
    intervals = {}
    for spec in specs:
        values = resampled_macro_f1(group_counts(records, spec, groups), weights)
        intervals[spec.name] = _ci(*interval(values), bootstrap, len(groups))
    return intervals


def paired_differences(records_a: Sequence[Aligned], records_b: Sequence[Aligned],
                       specs: Sequence[BlockSpec],
                       bootstrap: BootstrapSpec) -> dict[str, dict[str, Any]]:
    """Per block: CI of macro F1 (B - A) and the share of resamples where B beats A."""
    if [r.sample_id for r in records_a] != [r.sample_id for r in records_b]:
        raise EvaluationInputError("compared prediction sets must cover the same scored IDs")
    groups = group_order(records_a)
    weights = draw_weights(len(groups), bootstrap)
    result = {}
    for spec in specs:
        a = resampled_macro_f1(group_counts(records_a, spec, groups), weights)
        b = resampled_macro_f1(group_counts(records_b, spec, groups), weights)
        differences = [y - x for x, y in zip(a, b, strict=False)]
        share = sum(d > 0 for d in differences) / len(differences)
        result[spec.name] = {**_ci(*interval(differences), bootstrap, len(groups)),
                             "share_b_better": share}
    return result
