"""Member 2's frozen validation search, held-out evaluation and reproducible HMM handoff.

The evaluator is Member 5's implementation in a separate checkout pinned by config.
Tuning never opens test records. Finalization requires every configured mode's selection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import yaml

from punct_restore.models.hmm import HMMModel
from punct_restore.models.hmm_workflow import (
    checked_release_file,
    code_identity,
    file_identity,
    portable_path,
    read_id_selection,
    selected_records,
    sha256,
    write_json,
)
from punct_restore.paths import REPO_ROOT as ROOT

SOURCE_FILES = (
    "src/punct_restore/models/hmm.py",
    "src/punct_restore/models/hmm_workflow.py",
    "src/punct_restore/cli/train_hmm.py",
    "scripts/train_hmm.py",
    "src/punct_restore/cli/predict_hmm.py",
    "scripts/predict_hmm.py",
    "src/punct_restore/cli/run_hmm_experiments.py",
    "scripts/run_hmm_experiments.py",
    "src/punct_restore/cli/verify_hmm_experiments.py",
    "scripts/verify_hmm_experiments.py",
)


def now():
    return datetime.now(UTC).isoformat()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def implementation():
    return {name: sha256(ROOT / name) for name in SOURCE_FILES}


def hardware():
    details = {"processor": platform.processor(), "logical_cpus": os.cpu_count()}
    if platform.system() == "Windows":
        command = (
            "$hmmCpuInfo = Get-CimInstance Win32_Processor | Select-Object -First 1; "
            "$hmmOsInfo = Get-CimInstance Win32_OperatingSystem; "
            "[pscustomobject]@{processor=$hmmCpuInfo.Name; "
            "physical_cores=$hmmCpuInfo.NumberOfCores; "
            "logical_cpus=$hmmCpuInfo.NumberOfLogicalProcessors; "
            "memory_bytes=1024*$hmmOsInfo.TotalVisibleMemorySize} | ConvertTo-Json -Compress"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            check=True,
        )
        details.update(json.loads(result.stdout))
    elif hasattr(os, "sysconf"):
        try:
            details["memory_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (OSError, ValueError):
            pass
    return details


def validate_config(config):
    if config["schema_version"] != 1 or not config["modes"]:
        raise ValueError("invalid experiment schema/modes")
    if len(set(config["modes"])) != len(config["modes"]) or set(config["modes"]) - set("ABC"):
        raise ValueError("modes must be unique members of A/B/C")
    search = config["search"]
    if search["selection_split"] != "validation" or search["retrain_on_validation"]:
        raise ValueError("settings must be selected on validation; training stays train-only")
    if search["metric"] != "macro_f1" or search["score_precision"] != 6:
        raise ValueError("selection must use the shared scorer's six-decimal primary macro F1")
    if search["tie_break"] != ["lower_min_token_count", "lower_smoothing"]:
        raise ValueError("unsupported tie-break policy")
    if config["evaluation"]["test_sets"] != ["test", "test_amb"]:
        raise ValueError("report test and test_amb separately")
    for mode in config["modes"]:
        expected = "joint/unmasked" if mode == "C" else "punctuation/all"
        if search["selection_blocks"][mode] != expected:
            raise ValueError(f"mode {mode}: wrong primary selection block")
    for values, kind in [(search["smoothing"], float), (search["min_token_count"], int)]:
        if not values or len(set(values)) != len(values):
            raise ValueError("search values must be nonempty and unique")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError("search parameters must be numeric")
            if not math.isfinite(value) or value <= 0 or (kind is int and type(value) is not int):
                raise ValueError("smoothing/cutoffs must be positive; cutoffs must be integers")
    runtime = config["runtime"]
    if runtime["repetitions"] < 2 or runtime["warmup_sequences"] < 1:
        raise ValueError("runtime measurement requires warmup and repetitions")
    if runtime["model_threads"] != 1 or config["evaluation"]["zero_division"] != 0:
        raise ValueError("this implementation uses one model thread and zero_division=0")
    return config


def check_evaluator(root, expected_commit):
    identity = code_identity(root)
    if identity != {"commit": expected_commit, "dirty": False}:
        raise ValueError("evaluator checkout must be clean at the pinned Member 5 commit")
    names = sorted(
        p.relative_to(root).as_posix() for p in (root / "src/punct_restore/evaluation").glob("*.py")
    )
    names += ["scripts/evaluate_models.py", "configs/label_sets.yaml"]
    return {"commit": expected_commit, "files": {name: sha256(root / name) for name in names}}


def authenticate(config, phase):
    processed = ROOT / config["processed_dir"]
    splits = ROOT / config["splits_dir"]
    manifest_path = processed / "release_manifest.json"
    manifest = read_json(manifest_path)
    if (
        manifest["release_id"] != config["dataset_release_id"]
        or manifest["policy_version"] != config["policy_version"]
    ):
        raise ValueError("experiment and release versions differ")
    if sha256(ROOT / config["label_config"]) != manifest["label_sets_sha256"]:
        raise ValueError("label configuration mismatch")
    names = ["train", "validation"] if phase == "tune" else ["validation", "test", "test_amb"]
    verified = {"release_manifest": file_identity(manifest_path)}
    for mode in config["modes"]:
        for name in names:
            relative = f"mode_{mode}/{name}.jsonl"
            checked_release_file(
                processed / relative, processed, manifest["files"], expected=relative
            )
            id_name = (
                f"default_train/mode_{mode}.ids"
                if name == "train"
                else f"primary_eval/{name}_mode_{mode}.ids"
            )
            checked_release_file(
                splits / id_name, splits, manifest["split_files"], expected=id_name
            )
            verified[relative] = manifest["files"][relative]["sha256"]
            verified[id_name] = manifest["split_files"][id_name]["sha256"]
            if mode == "C" and name != "train":
                relative = f"masks/case_joint/{name}.jsonl"
                checked_release_file(
                    processed / relative, processed, manifest["files"], expected=relative
                )
                verified[relative] = manifest["files"][relative]["sha256"]
    return manifest, verified


def execute(args, run, log):
    start = perf_counter()
    result = subprocess.run(
        [str(x) for x in args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout + result.stderr, encoding="utf-8", newline="\n")
    record = {
        "command": ["python", *[str(x) for x in args[1:]]],
        "at_utc": now(),
        "seconds": perf_counter() - start,
        "returncode": result.returncode,
        "log": log.relative_to(run).as_posix(),
    }
    with (run / "commands.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record) + "\n")
    if result.returncode:
        raise RuntimeError(f"command failed; see {portable_path(log)}: {result.stderr[-1000:]}")
    return record


def score(evaluator, config, mode, split, predictions, out, run):
    args = [
        sys.executable,
        portable_path(evaluator / "scripts/evaluate_models.py"),
        "score",
        "--gold",
        f"{config['processed_dir']}/mode_{mode}/{split}.jsonl",
        "--ids",
        f"{config['splits_dir']}/primary_eval/{split}_mode_{mode}.ids",
        "--predictions",
        portable_path(predictions),
        "--out",
        portable_path(out),
    ]
    if mode == "C":
        args += ["--masks", f"{config['processed_dir']}/masks/case_joint/{split}.jsonl"]
    execute(args, run, out.with_suffix(".log"))
    report = read_json(out)
    if report["primary"]["block"] != config["search"]["selection_blocks"][mode]:
        raise ValueError("shared scorer returned an unexpected primary block")
    return report


def choose(candidates):
    if not candidates:
        raise ValueError("no completed validation candidates")
    if any(not math.isfinite(c["validation_primary"]) for c in candidates):
        raise ValueError("non-finite validation score")
    return min(
        candidates, key=lambda c: (-c["validation_primary"], c["min_token_count"], c["smoothing"])
    )


def tune(config_path, evaluator, run):
    config = validate_config(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    evaluator_identity = check_evaluator(evaluator, config["evaluation"]["repository_commit"])
    _, verified = authenticate(config, "tune")
    if run.exists():
        raise ValueError("run directory exists; choose a new experiment ID")
    plan = {
        "schema": "recap-hmm-experiment-v1",
        "created_at_utc": now(),
        "config": config,
        "configuration": file_identity(config_path),
        "code": code_identity(),
        "implementation": implementation(),
        "evaluator": evaluator_identity,
        "tuning_inputs": verified,
        "hardware": hardware(),
        "environment": {
            "python": platform.python_version(),
            "pyyaml": yaml.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "model_threads": 1,
        },
        "test_policy": "No test JSONL is opened in tuning; all modes freeze before finalization.",
    }
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "plan.json", plan)
    winners, mode_selection_hashes = {}, {}
    for mode in config["modes"]:
        candidates = []
        for cutoff in config["search"]["min_token_count"]:
            for smoothing in config["search"]["smoothing"]:
                candidate = f"s{smoothing:g}-c{cutoff}"
                out = run / f"mode_{mode}/candidates/{candidate}"
                out.parent.mkdir(parents=True, exist_ok=True)
                args = [
                    sys.executable,
                    "scripts/train_hmm.py",
                    "--train",
                    f"{config['processed_dir']}/mode_{mode}/train.jsonl",
                    "--train-ids",
                    f"{config['splits_dir']}/default_train/mode_{mode}.ids",
                    "--output-dir",
                    portable_path(out),
                    "--label-mode",
                    mode,
                    "--run-id",
                    f"{config['experiment_id']}-{mode}-{candidate}",
                    "--label-config",
                    config["label_config"],
                    "--smoothing",
                    str(smoothing),
                    "--min-token-count",
                    str(cutoff),
                    "--dataset-release-id",
                    config["dataset_release_id"],
                    "--preprocessing-version",
                    config["policy_version"],
                    "--tokenizer-version",
                    config["policy_version"],
                    "--label-policy-version",
                    config["policy_version"],
                    "--split-manifest",
                    f"{config['splits_dir']}/SHA256SUMS",
                    "--release-manifest",
                    f"{config['processed_dir']}/release_manifest.json",
                    "--predict-input",
                    f"{config['processed_dir']}/mode_{mode}/validation.jsonl",
                ]
                execute(args, run, out.parent / f"{candidate}.training.log")
                report = score(
                    evaluator,
                    config,
                    mode,
                    "validation",
                    out / "predictions.jsonl",
                    out / "validation.json",
                    run,
                )
                record = {
                    "candidate": candidate,
                    "smoothing": smoothing,
                    "min_token_count": cutoff,
                    "validation_primary": report["primary"]["value"],
                    "model": (out / "model.json").relative_to(run).as_posix(),
                    "model_sha256": sha256(out / "model.json"),
                    "validation_report_sha256": sha256(out / "validation.json"),
                }
                candidates.append(record)
                print(json.dumps({"mode": mode, **record}), flush=True)
        winner = choose(candidates)
        record = {
            "mode": mode,
            "frozen_at_utc": now(),
            "candidates": candidates,
            "selected": winner,
            "selection_block": config["search"]["selection_blocks"][mode],
            "train_only": True,
            "used_test_labels": False,
        }
        write_json(run / f"mode_{mode}/selection.json", record)
        winners[mode] = winner
        mode_selection_hashes[mode] = sha256(run / f"mode_{mode}/selection.json")
    # A single immutable gate freezes all configured modes before any held-out prediction.
    write_json(
        run / "selection.json",
        {
            "schema": "recap-hmm-selection-v1",
            "frozen_at_utc": now(),
            "plan_sha256": sha256(run / "plan.json"),
            "modes": winners,
            "mode_selection_sha256": mode_selection_hashes,
            "used_test_labels": False,
            "training_includes_validation": False,
        },
    )
    print(f"All mode selections frozen: {portable_path(run / 'selection.json')}", flush=True)


def frozen(run, evaluator):
    plan = read_json(run / "plan.json")
    selection = read_json(run / "selection.json")
    config = validate_config(plan["config"])
    if selection["plan_sha256"] != sha256(run / "plan.json"):
        raise ValueError("selection does not match its frozen plan")
    if (
        set(selection["modes"]) != set(config["modes"])
        or selection["used_test_labels"]
        or selection["training_includes_validation"]
    ):
        raise ValueError("all mode selections must freeze on validation before test access")
    if implementation() != plan["implementation"]:
        raise ValueError("HMM implementation changed since selection")
    if check_evaluator(evaluator, config["evaluation"]["repository_commit"]) != plan["evaluator"]:
        raise ValueError("shared evaluator changed since selection")
    for mode, winner in selection["modes"].items():
        if sha256(run / f"mode_{mode}/selection.json") != selection["mode_selection_sha256"][mode]:
            raise ValueError("mode selection changed after freezing")
        mode_selection = read_json(run / f"mode_{mode}/selection.json")
        expected = {
            (a, c)
            for a in config["search"]["smoothing"]
            for c in config["search"]["min_token_count"]
        }
        actual = {(c["smoothing"], c["min_token_count"]) for c in mode_selection["candidates"]}
        if actual != expected or len(actual) != len(mode_selection["candidates"]):
            raise ValueError("validation search is incomplete")
        for candidate in mode_selection["candidates"]:
            model_path = run / candidate["model"]
            report_path = model_path.parent / "validation.json"
            if (
                sha256(model_path) != candidate["model_sha256"]
                or sha256(report_path) != candidate["validation_report_sha256"]
                or read_json(report_path)["primary"]["value"] != candidate["validation_primary"]
            ):
                raise ValueError("candidate model/report differs from frozen validation evidence")
        if choose(mode_selection["candidates"]) != winner or mode_selection["selected"] != winner:
            raise ValueError("winner is inconsistent with the validation search")
        model = run / winner["model"]
        if sha256(model) != winner["model_sha256"]:
            raise ValueError("selected artifact checksum mismatch")
    return plan, selection


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def runtime_benchmark(config, run, winners):
    ids_path = ROOT / config["splits_dir"] / config["runtime"]["comparison_ids"]
    ids = read_id_selection(ids_path)
    observations = list(
        selected_records(ROOT / config["processed_dir"] / "mode_A/validation.jsonl", ids)
    )
    # Verify the exact same observable workload exists in all modes.
    for mode in config["modes"]:
        other = list(
            selected_records(ROOT / config["processed_dir"] / f"mode_{mode}/validation.jsonl", ids)
        )
        if other != observations:
            raise ValueError("runtime comparison tokens differ across label modes")
    tokens = sum(len(row["tokens"]) for row in observations)
    report = {
        "comparison_ids": file_identity(ids_path),
        "sequences": len(observations),
        "tokens": tokens,
        "repetitions": config["runtime"]["repetitions"],
        "warmup_sequences": min(config["runtime"]["warmup_sequences"], len(observations)),
        "scope": "in-memory model.predict; excludes disk IO, model loading, scoring and writes",
        "latency_scope": "one additional per-sequence model.decode pass after warmup",
        "model_threads": 1,
        "gc": "Python default, enabled",
        "modes": {},
    }
    for mode in config["modes"]:
        model = HMMModel.load_model(run / winners[mode]["model"])
        model.predict(observations[: report["warmup_sequences"]])
        times, reference = [], None
        for _ in range(report["repetitions"]):
            start = perf_counter()
            predicted = model.predict(observations)
            times.append(perf_counter() - start)
            if reference is not None and predicted != reference:
                raise ValueError("HMM inference is not deterministic")
            reference = predicted
        latencies = []
        for row, prediction in zip(observations, reference, strict=True):
            start = perf_counter()
            labels = model.decode(row["tokens"])
            latencies.append(perf_counter() - start)
            if labels != prediction["predicted_labels"]:
                raise ValueError("decode/predict disagree")
        median = statistics.median(times)
        report["modes"][mode] = {
            "prediction_seconds": times,
            "median_seconds": median,
            "tokens_per_second": tokens / median,
            "sequences_per_second": len(ids) / median,
            "sequence_latency_p50_ms": percentile(latencies, 0.5) * 1000,
            "sequence_latency_p95_ms": percentile(latencies, 0.95) * 1000,
        }
    write_json(run / "runtime.json", report)
    return report


def finalize(run, evaluator):
    plan, selection = frozen(run, evaluator)
    config = plan["config"]
    _, verified = authenticate(config, "finalize")
    if verified["release_manifest"] != plan["tuning_inputs"]["release_manifest"]:
        raise ValueError("release changed after model selection")
    final = run / "final"
    if final.exists():
        raise ValueError("finalization already started; final outputs are immutable")
    final.mkdir()
    write_json(
        final / "test_access.json",
        {
            "at_utc": now(),
            "selection_sha256": sha256(run / "selection.json"),
            "inputs": verified,
            "policy": "Frozen settings; no test-driven retuning.",
        },
    )
    results = {}
    for mode, winner in selection["modes"].items():
        results[mode] = {}
        for split in ["validation", *config["evaluation"]["test_sets"]]:
            out = final / f"mode_{mode}/{split}"
            out.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "scripts/predict_hmm.py",
                "--model",
                portable_path(run / winner["model"]),
                "--input",
                f"{config['processed_dir']}/mode_{mode}/{split}.jsonl",
                "--output-dir",
                portable_path(out),
                "--all-records",
                "--selection-record",
                portable_path(run / "selection.json"),
                "--release-manifest",
                f"{config['processed_dir']}/release_manifest.json",
            ]
            execute(command, run, out.parent / f"{split}.prediction.log")
            measured = score(
                evaluator,
                config,
                mode,
                split,
                out / "predictions.jsonl",
                out / "evaluation.json",
                run,
            )
            baseline = out / "shortcut.jsonl"
            execute(
                [
                    sys.executable,
                    portable_path(evaluator / "scripts/evaluate_models.py"),
                    "baseline",
                    "--gold",
                    f"{config['processed_dir']}/mode_{mode}/{split}.jsonl",
                    "--ids",
                    f"{config['splits_dir']}/primary_eval/{split}_mode_{mode}.ids",
                    "--out",
                    portable_path(baseline),
                ],
                run,
                out / "shortcut.log",
            )
            reference = score(
                evaluator, config, mode, split, baseline, out / "shortcut.evaluation.json", run
            )
            results[mode][split] = {
                "primary": measured["primary"],
                "counts": measured["counts"],
                "shortcut_primary": reference["primary"],
                "evaluation_sha256": sha256(out / "evaluation.json"),
            }
            print(
                json.dumps({"final": mode, "split": split, "primary": measured["primary"]}),
                flush=True,
            )
    runtime_benchmark(config, run, selection["modes"])
    write_json(
        run / "results.json",
        {
            "schema": "recap-hmm-results-v1",
            "finished_at_utc": now(),
            "selection_sha256": sha256(run / "selection.json"),
            "results": results,
            "note": "Mode A/B punctuation and mode C joint F1 measure different tasks.",
        },
    )
    print("Final predictions, shared scores and runtime measurements complete.", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recap run-hmm-experiments", description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    tune_parser = commands.add_parser("tune")
    tune_parser.add_argument("--config", type=Path, required=True)
    tune_parser.add_argument("--output-dir", type=Path, required=True)
    tune_parser.add_argument("--evaluator-root", type=Path, required=True)
    final_parser = commands.add_parser("finalize")
    final_parser.add_argument("--run-dir", type=Path, required=True)
    final_parser.add_argument("--evaluator-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.phase == "tune":
            if code_identity()["dirty"]:
                raise ValueError("commit reviewed code/config before starting an official search")
            tune(args.config, args.evaluator_root, args.output_dir)
        else:
            finalize(args.run_dir, args.evaluator_root)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
