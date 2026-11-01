"""Statistics of a data release (owned by Member 1): label and length distributions, overlap checks, stats.json content.

release.py re-exports `label_distribution`, `length_stats` and `mode_labels`.
"""
import collections
import statistics

from . import labeling, splits
from .release_constants import RELEASE_FILES


def label_distribution(label_lists, vocab):
    c = collections.Counter(label for labels in label_lists for label in labels)
    n = sum(c.values())
    return {"records": len(label_lists), "tokens": n, "counts": {label: c.get(label, 0) for label in vocab},
            "percent": {label: round(100 * c.get(label, 0) / n, 3) if n else 0.0 for label in vocab}}


def length_stats(records):
    lens = sorted(len(r["tokens"]) for r in records)
    if not lens:
        return {}

    def q(p):
        return lens[min(len(lens) - 1, int(p * len(lens)))]
    return {"records": len(lens), "tokens": sum(lens), "min": lens[0], "median": statistics.median(lens),
            "mean": round(statistics.fmean(lens), 2), "p95": q(0.95), "max": lens[-1]}


def mode_labels(r, mode):
    return r["ext"] if mode == "B" else r["canonical"]


def pairwise_overlap(by_file: dict, controlled: dict) -> dict:
    """Independent string-set overlap check between the release files, official and leakage-controlled."""
    overlap = {}
    for view, sets in (("official", {n: by_file[n] for n in RELEASE_FILES}),
                       ("leakage_controlled", {"train": by_file["train"], **controlled})):
        for i, a in enumerate(RELEASE_FILES):
            for b in RELEASE_FILES[i + 1:]:
                overlap[f"{view}:{b}_vs_{a}"] = splits.overlap_counts(sets[b], sets[a])
    return overlap


def _select(recs: list, mode: str) -> list:
    """Records whose labels are fully resolved for `mode`."""
    return [r for r in recs if not r["unresolved"][mode]]


def _overview_stats(results: list, kept: list, by_file: dict) -> dict:
    """Status, reject, unresolved, info, length, group and normalization counts."""
    return {"status_by_file": {n: dict(collections.Counter(r["status"] for r in results if r["file"] == n))
                               for n in RELEASE_FILES},
            "reject_reasons": dict(collections.Counter(
                x for r in results if r["status"] == "rejected" for x in r["reasons"])),
            "unresolved_reasons_by_mode": {
                m: dict(collections.Counter(x for r in kept for x in r["unresolved"][m]).most_common())
                for m in ("A", "B", "C")},
            "info_flags": dict(collections.Counter(x for r in kept for x in r["info"]).most_common()),
            "lengths": {n: length_stats(by_file[n]) for n in RELEASE_FILES},
            "groups": {n: len({r["group_id"] for r in by_file[n]}) for n in RELEASE_FILES},
            "normalization_removals": dict(collections.Counter(
                op for r in results for op, _ in r["removals"]).most_common())}


def _subset_sizes(default: dict) -> dict:
    """Sample counts of the default training and primary evaluation subsets per mode."""
    return {m: {"default_train": len(_select(default["train"], m)),
                **{f"primary_{n}": len(_select(default[n], m)) for n in ("validation", "test", "test_amb")}}
            for m in ("A", "B", "C")}


def _label_stats(recs: list, label_sets: dict) -> dict:
    """Label distributions of one release file for every view."""
    return {"A": label_distribution([r["canonical"] for r in recs], label_sets["A"]),
            "B": label_distribution([r["ext"] for r in recs], label_sets["B"]),
            "native_A": label_distribution([r["native"] for r in recs], label_sets["A"]),
            "case": label_distribution([r["case"] for r in recs], label_sets["casing"]),
            "C": label_distribution([labeling.join_labels(r["canonical"], r["case"]) for r in recs],
                                    label_sets["C"])}


def _subset_label_stats(subset: list, label_sets: dict) -> dict:
    """Label distributions of the resolved records of a default/primary subset."""
    stats = {m: label_distribution([mode_labels(r, m) for r in _select(subset, m)], label_sets[m])
             for m in ("A", "B")}
    stats["C"] = label_distribution(
        [labeling.join_labels(r["canonical"], r["case"]) for r in _select(subset, "C")], label_sets["C"])
    return stats


def _native_change_stats(recs: list) -> dict:
    """How many labels the A projection changed relative to the native SponSpeech labels."""
    pairs = [(a, b) for r in recs for a, b in zip(r["native"], r["canonical"], strict=False)]
    return {"tokens": len(pairs), "changed": sum(a != b for a, b in pairs),
            "records_changed": sum(r["native"] != r["canonical"] for r in recs),
            "changes": dict(collections.Counter(f"{a}->{b}" for a, b in pairs if a != b))}


def _case_mask_stats(recs: list) -> dict:
    """Masked casing positions of one release file."""
    masked = sum(len(r["case_masked"]) for r in recs)
    tokens = sum(len(r["tokens"]) for r in recs)
    return {"masked_positions": masked, "records_with_masks": sum(bool(r["case_masked"]) for r in recs),
            "tokens": tokens, "masked_percent": round(100 * masked / max(1, tokens), 3)}


def _final_token_stats(by_file: dict) -> dict:
    """Final-label counts and the share of PERIOD labels that sit on the last token."""
    return {n: {"final_label_counts": dict(collections.Counter(r["canonical"][-1] for r in by_file[n])),
                "share_of_PERIOD_that_is_final": round(
                    100 * sum(r["canonical"][-1] == "PERIOD" for r in by_file[n])
                    / max(1, sum(r["canonical"].count("PERIOD") for r in by_file[n])), 2)}
            for n in RELEASE_FILES}


def build_stats(results: list, kept: list, by_file: dict, default: dict, controlled_report: dict,
                overlap: dict, label_sets: dict) -> dict:
    """The stats.json content (without `checks`), computed from the processed records."""
    stats = {**_overview_stats(results, kept, by_file),
             "leakage_controlled": controlled_report, "overlap": overlap,
             "subset_sizes": _subset_sizes(default),
             "labels": {}, "labels_default_and_primary": {}, "canonical_vs_native": {}, "case_masks": {},
             "case_categories": {n: dict(collections.Counter(c for r in by_file[n] for c in r["case_cats"]))
                                 for n in RELEASE_FILES}}
    for n in RELEASE_FILES:
        recs = by_file[n]
        stats["labels"][n] = _label_stats(recs, label_sets)
        subset = default[n] if n in default else []
        stats["labels_default_and_primary"][n] = _subset_label_stats(subset, label_sets)
        stats["canonical_vs_native"][n] = _native_change_stats(recs)
        stats["case_masks"][n] = _case_mask_stats(recs)
    stats["final_token"] = _final_token_stats(by_file)
    return stats
