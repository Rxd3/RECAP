"""Independently audit HMM runs, refit frozen winners and reproduce every prediction."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from punct_restore.data.labeling import load_label_sets  # noqa: E402
from punct_restore.models.hmm import HMMModel  # noqa: E402
from punct_restore.models.hmm_workflow import (  # noqa: E402
    file_identity,
    read_id_selection,
    read_jsonl,
    selected_records,
    sha256,
    write_json,
)
from run_hmm_experiments import authenticate, frozen, now, read_json  # noqa: E402


def assert_close(value, expected):
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


def check_report(
    config,
    mode,
    split,
    predictions,
    report_path,
    *,
    primary_only=False,
    baseline=False,
    examples=False,
):
    label_sets = load_label_sets(ROOT / config["label_config"])
    ids_path = ROOT / config["splits_dir"] / f"primary_eval/{split}_mode_{mode}.ids"
    wanted = set(read_id_selection(ids_path))
    gold_path = ROOT / config["processed_dir"] / f"mode_{mode}/{split}.jsonl"
    gold = list(read_jsonl(gold_path))
    gold_ids = [row["sample_id"] for row in gold]
    if len(gold_ids) != len(set(gold_ids)) or not wanted <= set(gold_ids):
        raise ValueError("gold/selection ID violation")
    pred = list(read_jsonl(predictions))
    pred_ids = [row["sample_id"] for row in pred]
    expected_ids = [sid for sid in gold_ids if sid in wanted] if baseline else gold_ids
    if pred_ids != expected_ids:
        raise ValueError("predictions do not preserve exact required IDs/order/coverage")
    lookup = {row["sample_id"]: row for row in pred}
    for row in pred:
        if row["label_mode"] != mode or any(
            label not in label_sets[mode] for label in row["predicted_labels"]
        ):
            raise ValueError("prediction label vocabulary/mode violation")
    masks = {}
    mask_path = ROOT / config["processed_dir"] / f"masks/case_joint/{split}.jsonl"
    if mode == "C":
        masks = {row["sample_id"]: set(row["excluded_positions"]) for row in read_jsonl(mask_path)}
        if set(masks) != set(gold_ids):
            raise ValueError("mask IDs do not cover the full gold split")
    report = read_json(report_path)
    for key, path in [("gold", gold_path), ("ids", ids_path), ("predictions", predictions)]:
        if report["inputs"][key]["sha256"] != sha256(path):
            raise ValueError("shared report input hash mismatch")
    if mode == "C" and report["inputs"]["masks"]["sha256"] != sha256(mask_path):
        raise ValueError("shared report mask hash mismatch")
    primary = config["search"]["selection_blocks"][mode]
    block_names = [primary] if primary_only else list(report["blocks"])
    pairs = {name: [] for name in block_names}
    selected_tokens, samples, sample_counts = 0, [], Counter()
    for row in gold:
        sid = row["sample_id"]
        if sid not in wanted:
            continue
        targets = (
            row["punct_labels"]
            if mode != "C"
            else [f"{p}|{c}" for p, c in zip(row["punct_labels"], row["case_labels"], strict=True)]
        )
        labels = lookup[sid]["predicted_labels"]
        if len(labels) != len(row["tokens"]) or len(targets) != len(labels):
            raise ValueError("prediction/gold token alignment violation")
        selected_tokens += len(targets)
        for i, (g, p) in enumerate(zip(targets, labels, strict=True)):
            for name in block_names:
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
                if examples and name == primary and g != p and sample_counts[g, p] < 2:
                    samples.append(
                        {
                            "sample_id": sid,
                            "position": i,
                            "gold": g,
                            "predicted": p,
                            "context_tokens": row["tokens"][max(0, i - 3) : i + 4],
                            "selection_rule": "first two occurrences per ordered confusion pair",
                        }
                    )
                    sample_counts[g, p] += 1
    if (
        report["counts"]["sequences"] != len(wanted)
        or report["counts"]["tokens"] != selected_tokens
        or report["counts"]["ignored_prediction_ids"] != len(pred_ids) - len(wanted)
    ):
        raise ValueError("scored-set counts disagree")
    for name, values in pairs.items():
        block = report["blocks"][name]
        assert_block(block, independent_scores(values, block["labels"], block["macro_over"]))
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


def probability_checks(model):
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


def verify(run, evaluator, output):
    plan, selection = frozen(run, evaluator)
    config = plan["config"]
    if output.exists():
        raise ValueError("verification output already exists; use a new verification directory")
    if not (run / "results.json").exists():
        raise ValueError("finalization is incomplete")
    authenticate(config, "tune")
    authenticate(config, "finalize")
    results = read_json(run / "results.json")
    if results["selection_sha256"] != sha256(run / "selection.json"):
        raise ValueError("final results are not bound to frozen selection")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "recap-hmm-verification-v1",
        "at_utc": now(),
        "plan": file_identity(run / "plan.json"),
        "selection": file_identity(run / "selection.json"),
        "python": sys.version.split()[0],
        "modes": {},
    }
    stats = read_json(ROOT / config["processed_dir"] / "stats.json")
    for mode, winner in selection["modes"].items():
        mode_report = {"candidates_independently_checked": 0, "final_reports": {}}
        candidates = read_json(run / f"mode_{mode}/selection.json")["candidates"]
        for candidate in candidates:
            folder = (run / candidate["model"]).parent
            check_report(
                config,
                mode,
                "validation",
                folder / "predictions.jsonl",
                folder / "validation.json",
                primary_only=True,
            )
            mode_report["candidates_independently_checked"] += 1
        model_path = run / winner["model"]
        model = HMMModel.load_model(model_path)
        probability_checks(model)
        saved = read_json(model_path)
        expected_support = stats["labels_default_and_primary"]["train"][mode]["counts"]
        if dict(zip(model.labels, saved["counts"]["state"], strict=True)) != expected_support:
            raise ValueError("learned state counts do not match the approved training subset")
        ids = read_id_selection(ROOT / config["splits_dir"] / f"default_train/mode_{mode}.ids")
        if model.training_summary["num_sequences"] != len(ids):
            raise ValueError("model trained on the wrong sequence count")
        fresh = HMMModel(**saved["config"]).fit(
            selected_records(
                ROOT / config["processed_dir"] / f"mode_{mode}/train.jsonl",
                ids,
                training_mode=mode,
                policy_version=config["policy_version"],
                source_revision=read_json(ROOT / config["processed_dir"] / "release_manifest.json")[
                    "source"
                ]["revision"],
            ),
            metadata=saved["metadata"],
        )
        reproduced = output / f"mode_{mode}.reproduced.json"
        fresh.save_model(reproduced)
        if sha256(reproduced) != sha256(model_path):
            raise ValueError("independent refit does not reproduce the frozen artifact bytes")
        mode_report["independent_refit"] = {
            "result": "PASS",
            "model_sha256": sha256(reproduced),
            "metadata_policy": "original frozen provenance retained for byte comparison",
        }
        for split in ["validation", "test", "test_amb"]:
            folder = run / f"final/mode_{mode}/{split}"
            manifest = read_json(folder / "predictions.manifest.json")
            if (
                manifest["model_sha256"] != sha256(model_path)
                or manifest["predictions_sha256"] != sha256(folder / "predictions.jsonl")
                or manifest["inputs"]["selection_record"]["sha256"]
                != sha256(run / "selection.json")
            ):
                raise ValueError("prediction sidecar provenance/hash mismatch")
            gold_path = ROOT / config["processed_dir"] / f"mode_{mode}/{split}.jsonl"
            observations = list(selected_records(gold_path, None))
            predictions = fresh.predict(observations)
            actual = list(read_jsonl(folder / "predictions.jsonl"))
            if actual != predictions:
                raise ValueError("independent refit changed full-split predictions")
            poison = [
                {
                    **row,
                    "punct_labels": ["INVALID"],
                    "case_labels": ["INVALID"],
                    "metadata": {"original_text": "NEVER USE GOLD"},
                }
                for row in observations[:20]
            ]
            if fresh.predict(poison) != predictions[:20]:
                raise ValueError("predictions depend on privileged fields")
            checked = check_report(
                config,
                mode,
                split,
                folder / "predictions.jsonl",
                folder / "evaluation.json",
                examples=split != "validation",
            )
            checked["shortcut"] = check_report(
                config,
                mode,
                split,
                folder / "shortcut.jsonl",
                folder / "shortcut.evaluation.json",
                baseline=True,
            )
            mode_report["final_reports"][split] = checked
        report["modes"][mode] = mode_report
        print(
            f"Verified mode {mode}: search, counts, probabilities, full refit/predictions, scores.",
            flush=True,
        )
    runtime = read_json(run / "runtime.json")
    if set(runtime["modes"]) != set(config["modes"]):
        raise ValueError("runtime measurements are incomplete")
    for mode in config["modes"]:
        values = runtime["modes"][mode]["prediction_seconds"]
        if len(values) != config["runtime"]["repetitions"] or any(v <= 0 for v in values):
            raise ValueError("invalid runtime repetitions")
    report["runtime_checked"] = True
    report["result"] = "PASS"
    write_json(output / "verification.json", report)
    print(f"Complete HMM verification: {output / 'verification.json'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        verify(args.run_dir, args.evaluator_root, args.output_dir)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
