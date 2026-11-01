"""Command lines and subprocess execution for the HMM experiment's train, predict and score steps."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from punct_restore.models.hmm_experiment_protocol import now, read_json
from punct_restore.models.hmm_workflow import portable_path
from punct_restore.paths import REPO_ROOT as ROOT


def execute(args: list[Any], run: Path, log: Path) -> dict[str, Any]:
    """Run one command from the repository root, log its output and append it to commands.jsonl."""
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


def score(
    evaluator: Path,
    config: dict[str, Any],
    mode: str,
    split: str,
    predictions: Path,
    out: Path,
    run: Path,
) -> dict[str, Any]:
    """Score predictions with the pinned shared evaluator and check the primary block."""
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


def train_command(
    config: dict[str, Any], mode: str, candidate: str, smoothing: float, cutoff: int, out: Path
) -> list[str]:
    """The scripts/train_hmm.py command line for one validation-search candidate."""
    return [
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


def predict_command(
    config: dict[str, Any], mode: str, split: str, run: Path, winner: dict[str, Any], out: Path
) -> list[str]:
    """The scripts/predict_hmm.py command line for one frozen winner on one split."""
    return [
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


def baseline_command(
    config: dict[str, Any], evaluator: Path, mode: str, split: str, baseline: Path
) -> list[str]:
    """The shared evaluator's shortcut-baseline command line for one mode and split."""
    return [
        sys.executable,
        portable_path(evaluator / "scripts/evaluate_models.py"),
        "baseline",
        "--gold",
        f"{config['processed_dir']}/mode_{mode}/{split}.jsonl",
        "--ids",
        f"{config['splits_dir']}/primary_eval/{split}_mode_{mode}.ids",
        "--out",
        portable_path(baseline),
    ]
