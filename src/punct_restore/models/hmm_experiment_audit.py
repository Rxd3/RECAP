"""Independent recount of scored HMM reports and probability checks, written without the Member 5 formulas."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from punct_restore.data.labeling import load_label_sets
from punct_restore.hashing import sha256_file
from punct_restore.models.hmm_experiment_protocol import read_json
from punct_restore.models.hmm_records import read_jsonl
from punct_restore.paths import REPO_ROOT as ROOT
from punct_restore.prediction_io import read_ids


def assert_close(value: float, expected: float) -> None:
    """Require two metric values to agree to the six-decimal score precision."""
    if not math.isclose(value, expected, abs_tol=5.01e-7, rel_tol=0):
        raise ValueError(f"independent metric mismatch: {value} != {expected}")


def independent_scores(pairs, labels, macro_over):
    """Compute P/R/F1 directly from counts, independently of the Member 5 formulas."""
    counts = Counter(pairs)
    matrix = [[counts[g, p] for p in labels] for g in labels]
    support = {g: sum(counts[g, p] for p in labels) for g in labels}
    predicted = {p: sum(counts[g, p] for g in labels) for p in labels}
    per_class = {}
    for label in labels:
        tp, gold_n, pred_n = counts[label, label], support[label], predicted[label]
        per_class[label] = {
            "support": gold_n,
            "predicted": pred_n,
            "precision": tp / pred_n if pred_n else 0.0,
            "recall": tp / gold_n if gold_n else 0.0,
            "f1": 2 * tp / (gold_n + pred_n) if gold_n + pred_n else 0.0,
        }
    tp = sum(counts[label, label] for label in macro_over)
    gold_n = sum(support[label] for label in macro_over)
    pred_n = sum(predicted[label] for label in macro_over)
    total = sum(support.values())
    return {
        "confusion": matrix,
        "per_class": per_class,
        "scored_positions": total,
        "macro_f1": sum(per_class[label]["f1"] for label in macro_over) / len(macro_over),
        "accuracy": sum(counts[label, label] for label in labels) / total if total else 0.0,
        "micro": {
            "precision": tp / pred_n if pred_n else 0.0,
            "recall": tp / gold_n if gold_n else 0.0,
            "f1": 2 * tp / (gold_n + pred_n) if gold_n + pred_n else 0.0,
        },
    }


def assert_block(actual, expected):
    """Require a recounted metric block to match the shared scorer's block."""
    if (
        actual["confusion"] != expected["confusion"]
        or actual["scored_positions"] != expected["scored_positions"]
    ):
        raise ValueError("confusion matrix/support mismatch")
    for key in ["macro_f1", "accuracy"]:
        assert_close(actual[key], expected[key])
    for key in ["precision", "recall", "f1"]:
        assert_close(actual["micro"][key], expected["micro"][key])
    for label, values in expected["per_class"].items():
        for key, value in values.items():
            if key in ["support", "predicted"]:
                if actual["per_class"][label][key] != value:
                    raise ValueError("per-class support mismatch")
            else:
                assert_close(actual["per_class"][label][key], value)


def probability_checks(model):
    """Require every start, transition and emission distribution of the model to sum to one."""
    assert_close(sum(math.exp(v) for v in model._log_start), 1.0)
    for state in range(len(model.labels)):
        assert_close(
            sum(math.exp(v) for v in model._log_transition[state])
            + math.exp(model._log_end[state]),
            1.0,
        )
        missing = len(model._vocabulary) - len(model._log_emissions[state])
        assert_close(
            sum(math.exp(v) for v in model._log_emissions[state].values())
            + missing * math.exp(model._log_default[state])
            + math.exp(model._log_unknown[state]),
            1.0,
        )


def _report_paths(config: dict[str, Any], mode: str, split: str) -> tuple[Path, Path, Path]:
    """The primary-ID list, gold split and case-mask paths of one scored split."""
    ids_path = ROOT / config["splits_dir"] / f"primary_eval/{split}_mode_{mode}.ids"
    gold_path = ROOT / config["processed_dir"] / f"mode_{mode}/{split}.jsonl"
    mask_path = ROOT / config["processed_dir"] / f"masks/case_joint/{split}.jsonl"
    return ids_path, gold_path, mask_path


def _read_gold_and_predictions(
    ids_path: Path, gold_path: Path, predictions: Path, baseline: bool
) -> tuple[set[str], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Read the gold split and predictions; require the exact IDs, order and coverage."""
    wanted = set(read_ids(ids_path, strict=True))
    gold = list(read_jsonl(gold_path))
    gold_ids = [row["sample_id"] for row in gold]
    if len(gold_ids) != len(set(gold_ids)) or not wanted <= set(gold_ids):
        raise ValueError("gold/selection ID violation")
    pred = list(read_jsonl(predictions))
    pred_ids = [row["sample_id"] for row in pred]
    expected_ids = [sid for sid in gold_ids if sid in wanted] if baseline else gold_ids
    if pred_ids != expected_ids:
        raise ValueError("predictions do not preserve exact required IDs/order/coverage")
    return wanted, gold, pred, pred_ids


def _check_label_vocabulary(pred: list[dict[str, Any]], label_sets: dict[str, Any], mode: str) -> None:
    """Require every prediction to use the mode's own label set."""
    for row in pred:
        if row["label_mode"] != mode or any(
            label not in label_sets[mode] for label in row["predicted_labels"]
        ):
            raise ValueError("prediction label vocabulary/mode violation")


def _read_masks(mode: str, mask_path: Path, gold: list[dict[str, Any]]) -> dict[str, set[int]]:
    """The case-joint exclusion masks (mode C only), which must cover the whole gold split."""
    masks = {}
    if mode == "C":
        masks = {row["sample_id"]: set(row["excluded_positions"]) for row in read_jsonl(mask_path)}
        if set(masks) != {row["sample_id"] for row in gold}:
            raise ValueError("mask IDs do not cover the full gold split")
    return masks


def _check_input_hashes(
    report: dict[str, Any], mode: str, predictions: Path, paths: tuple[Path, Path, Path]
) -> None:
    """Require the shared report to cite the exact gold, ID, prediction and mask files."""
    ids_path, gold_path, mask_path = paths
    for key, path in [("gold", gold_path), ("ids", ids_path), ("predictions", predictions)]:
        if report["inputs"][key]["sha256"] != sha256_file(path):
            raise ValueError("shared report input hash mismatch")
    if mode == "C" and report["inputs"]["masks"]["sha256"] != sha256_file(mask_path):
        raise ValueError("shared report mask hash mismatch")


def _targets(row: dict[str, Any], mode: str) -> list[str]:
    """The gold labels of a record in the scoring mode's label space."""
    return (
        row["punct_labels"]
        if mode != "C"
        else [f"{p}|{c}" for p, c in zip(row["punct_labels"], row["case_labels"], strict=True)]
    )


def _error_sample(sid: str, row: dict[str, Any], i: int, g: str, p: str) -> dict[str, Any]:
    """One error example with its surrounding tokens."""
    return {
        "sample_id": sid,
        "position": i,
        "gold": g,
        "predicted": p,
        "context_tokens": row["tokens"][max(0, i - 3) : i + 4],
        "selection_rule": "first two occurrences per ordered confusion pair",
    }


@dataclass(frozen=True)
class _PairSpec:
    """What to collect from a scored split: the mode, the metric blocks, the primary block and the exclusions."""

    mode: str
    block_names: list[str]
    primary: str
    masks: dict[str, set[int]]
    examples: bool


def _collect_pairs(
    gold: list[dict[str, Any]], wanted: set[str], lookup: dict[str, dict[str, Any]], spec: _PairSpec
) -> tuple[dict[str, list], int, list[dict[str, Any]]]:
    """Per-block (gold, predicted) pairs over the scored IDs, with the token count and error samples."""
    mode, masks = spec.mode, spec.masks
    pairs = {name: [] for name in spec.block_names}
    selected_tokens, samples, sample_counts = 0, [], Counter()
    for row in gold:
        sid = row["sample_id"]
        if sid not in wanted:
            continue
        targets = _targets(row, mode)
        labels = lookup[sid]["predicted_labels"]
        if len(labels) != len(row["tokens"]) or len(targets) != len(labels):
            raise ValueError("prediction/gold token alignment violation")
        selected_tokens += len(targets)
        for i, (g, p) in enumerate(zip(targets, labels, strict=True)):
            for name in spec.block_names:
                if name.startswith(("joint/", "casing/unmasked")) and i in masks.get(sid, set()):
                    continue
                if name.endswith("non_final") and i == len(targets) - 1:
                    continue
                if name.endswith("non_initial") and i == 0:
                    continue
                pair = (g, p)
                if mode == "C" and name.startswith("punctuation/"):
                    pair = (g.split("|")[0], p.split("|")[0])
                elif mode == "C" and name.startswith("casing/"):
                    pair = (g.split("|")[1], p.split("|")[1])
                pairs[name].append(pair)
                if spec.examples and name == spec.primary and g != p and sample_counts[g, p] < 2:
                    samples.append(_error_sample(sid, row, i, g, p))
                    sample_counts[g, p] += 1
    return pairs, selected_tokens, samples


def _check_counts_and_blocks(
    report: dict[str, Any],
    pairs: dict[str, list],
    wanted: set[str],
    selected_tokens: int,
    pred_ids: list[str],
) -> None:
    """Require the shared report's counts and every metric block to match the recount."""
    if (
        report["counts"]["sequences"] != len(wanted)
        or report["counts"]["tokens"] != selected_tokens
        or report["counts"]["ignored_prediction_ids"] != len(pred_ids) - len(wanted)
    ):
        raise ValueError("scored-set counts disagree")
    for name, values in pairs.items():
        block = report["blocks"][name]
        assert_block(block, independent_scores(values, block["labels"], block["macro_over"]))


@dataclass(frozen=True)
class ReportOptions:
    """How much of a scored report to recount: only its primary block, as a baseline, with error examples."""

    primary_only: bool = False
    baseline: bool = False
    examples: bool = False


def check_report(
    config: dict[str, Any],
    mode: str,
    split: str,
    predictions: Path,
    report_path: Path,
    options: ReportOptions,
) -> dict[str, Any]:
    """Recount a shared scorer report from the raw gold and prediction files."""
    label_sets = load_label_sets(ROOT / config["label_config"])
    ids_path, gold_path, mask_path = _report_paths(config, mode, split)
    wanted, gold, pred, pred_ids = _read_gold_and_predictions(ids_path, gold_path, predictions, options.baseline)
    lookup = {row["sample_id"]: row for row in pred}
    _check_label_vocabulary(pred, label_sets, mode)
    masks = _read_masks(mode, mask_path, gold)
    report = read_json(report_path)
    _check_input_hashes(report, mode, predictions, (ids_path, gold_path, mask_path))
    primary = config["search"]["selection_blocks"][mode]
    block_names = [primary] if options.primary_only else list(report["blocks"])
    spec = _PairSpec(mode, block_names, primary, masks, options.examples)
    pairs, selected_tokens, samples = _collect_pairs(gold, wanted, lookup, spec)
    _check_counts_and_blocks(report, pairs, wanted, selected_tokens, pred_ids)
    assert_close(report["primary"]["value"], report["blocks"][primary]["macro_f1"])
    return {
        "blocks_checked": block_names,
        "sequences": len(wanted),
        "tokens": selected_tokens,
        "sample_errors": samples,
        "largest_confusions": [
            {"gold": g, "predicted": p, "count": n}
            for (g, p), n in Counter(pairs[primary]).most_common()
            if g != p
        ][:10],
    }
