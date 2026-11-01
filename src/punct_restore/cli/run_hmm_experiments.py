"""Member 2's frozen validation search, held-out evaluation and reproducible HMM handoff.

The evaluator is Member 5's implementation in a separate checkout pinned by config.
Tuning never opens test records. Finalization requires every configured mode's selection.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import yaml

from punct_restore.hashing import sha256_file
from punct_restore.models.hmm import HMMModel
from punct_restore.models.hmm_experiment_commands import (
    baseline_command,
    execute,
    predict_command,
    score,
    train_command,
)
from punct_restore.models.hmm_experiment_protocol import (
    authenticate,
    choose,
    now,
    read_json,
    validate_config,
)
from punct_restore.models.hmm_experiment_runtime import hardware, percentile, runtime_benchmark
from punct_restore.models.hmm_workflow import (
    checked_release_file,
    code_identity,
    file_identity,
    portable_path,
    selected_records,
    write_json,
)
from punct_restore.paths import REPO_ROOT as ROOT
from punct_restore.prediction_io import one_line_error

__all__ = [
    "HMMModel",
    "ROOT",
    "SOURCE_FILES",
    "authenticate",
    "check_evaluator",
    "checked_release_file",
    "choose",
    "code_identity",
    "execute",
    "file_identity",
    "finalize",
    "frozen",
    "hardware",
    "implementation",
    "main",
    "now",
    "percentile",
    "portable_path",
    "read_json",
    "runtime_benchmark",
    "score",
    "selected_records",
    "sha256_file",
    "tune",
    "validate_config",
    "write_json",
]

SOURCE_FILES = (
    "src/punct_restore/models/hmm.py",
    "src/punct_restore/models/hmm_validation.py",
    "src/punct_restore/models/hmm_workflow.py",
    "src/punct_restore/models/hmm_experiment_protocol.py",
    "src/punct_restore/models/hmm_experiment_commands.py",
    "src/punct_restore/models/hmm_experiment_runtime.py",
    "src/punct_restore/models/hmm_experiment_audit.py",
    "src/punct_restore/hashing.py",
    "src/punct_restore/prediction_io.py",
    "src/punct_restore/cli/train_hmm.py",
    "scripts/train_hmm.py",
    "src/punct_restore/cli/predict_hmm.py",
    "scripts/predict_hmm.py",
    "src/punct_restore/cli/run_hmm_experiments.py",
    "scripts/run_hmm_experiments.py",
    "src/punct_restore/cli/verify_hmm_experiments.py",
    "scripts/verify_hmm_experiments.py",
)


def implementation():
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def check_evaluator(root, expected_commit):
    identity = code_identity(root)
    if identity != {"commit": expected_commit, "dirty": False}:
        raise ValueError("evaluator checkout must be clean at the pinned Member 5 commit")
    names = sorted(
        p.relative_to(root).as_posix() for p in (root / "src/punct_restore/evaluation").glob("*.py")
    )
    names += ["scripts/evaluate_models.py", "configs/label_sets.yaml"]
    return {"commit": expected_commit, "files": {name: sha256_file(root / name) for name in names}}


def _tune_plan(
    config: dict[str, Any], config_path: Path, evaluator_identity: dict[str, Any], verified: dict[str, Any]
) -> dict[str, Any]:
    """The plan recorded before any candidate is trained."""
    return {
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


def _search_candidate(
    config: dict[str, Any], evaluator: Path, run: Path, mode: str, smoothing: float, cutoff: int
) -> dict[str, Any]:
    """Train one candidate, score it on validation and describe it."""
    candidate = f"s{smoothing:g}-c{cutoff}"
    out = run / f"mode_{mode}/candidates/{candidate}"
    out.parent.mkdir(parents=True, exist_ok=True)
    args = train_command(config, mode, candidate, smoothing, cutoff, out)
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
    return {
        "candidate": candidate,
        "smoothing": smoothing,
        "min_token_count": cutoff,
        "validation_primary": report["primary"]["value"],
        "model": (out / "model.json").relative_to(run).as_posix(),
        "model_sha256": sha256_file(out / "model.json"),
        "validation_report_sha256": sha256_file(out / "validation.json"),
    }


def _tune_mode(
    config: dict[str, Any], evaluator: Path, run: Path, mode: str
) -> tuple[dict[str, Any], str]:
    """Search one mode's grid, write its selection file and return the winner and that file's hash."""
    candidates = []
    for cutoff in config["search"]["min_token_count"]:
        for smoothing in config["search"]["smoothing"]:
            record = _search_candidate(config, evaluator, run, mode, smoothing, cutoff)
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
    return winner, sha256_file(run / f"mode_{mode}/selection.json")


def _freeze_selection(
    run: Path, winners: dict[str, Any], mode_selection_hashes: dict[str, str]
) -> None:
    """Write the single immutable gate that freezes all configured modes."""
    write_json(
        run / "selection.json",
        {
            "schema": "recap-hmm-selection-v1",
            "frozen_at_utc": now(),
            "plan_sha256": sha256_file(run / "plan.json"),
            "modes": winners,
            "mode_selection_sha256": mode_selection_hashes,
            "used_test_labels": False,
            "training_includes_validation": False,
        },
    )


def tune(config_path, evaluator, run):
    config = validate_config(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    evaluator_identity = check_evaluator(evaluator, config["evaluation"]["repository_commit"])
    _, verified = authenticate(config, "tune")
    if run.exists():
        raise ValueError("run directory exists; choose a new experiment ID")
    plan = _tune_plan(config, config_path, evaluator_identity, verified)
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "plan.json", plan)
    winners, mode_selection_hashes = {}, {}
    for mode in config["modes"]:
        winners[mode], mode_selection_hashes[mode] = _tune_mode(config, evaluator, run, mode)
    # A single immutable gate freezes all configured modes before any held-out prediction.
    _freeze_selection(run, winners, mode_selection_hashes)
    print(f"All mode selections frozen: {portable_path(run / 'selection.json')}", flush=True)


def frozen(run, evaluator):
    plan = read_json(run / "plan.json")
    selection = read_json(run / "selection.json")
    config = validate_config(plan["config"])
    if selection["plan_sha256"] != sha256_file(run / "plan.json"):
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
        if sha256_file(run / f"mode_{mode}/selection.json") != selection["mode_selection_sha256"][mode]:
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
                sha256_file(model_path) != candidate["model_sha256"]
                or sha256_file(report_path) != candidate["validation_report_sha256"]
                or read_json(report_path)["primary"]["value"] != candidate["validation_primary"]
            ):
                raise ValueError("candidate model/report differs from frozen validation evidence")
        if choose(mode_selection["candidates"]) != winner or mode_selection["selected"] != winner:
            raise ValueError("winner is inconsistent with the validation search")
        model = run / winner["model"]
        if sha256_file(model) != winner["model_sha256"]:
            raise ValueError("selected artifact checksum mismatch")
    return plan, selection


def _final_split(
    config: dict[str, Any],
    evaluator: Path,
    run: Path,
    final: Path,
    mode: str,
    winner: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    """Predict and score one mode and split with its shortcut baseline; return its result summary."""
    out = final / f"mode_{mode}/{split}"
    out.parent.mkdir(parents=True, exist_ok=True)
    command = predict_command(config, mode, split, run, winner, out)
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
    execute(baseline_command(config, evaluator, mode, split, baseline), run, out / "shortcut.log")
    reference = score(
        evaluator, config, mode, split, baseline, out / "shortcut.evaluation.json", run
    )
    summary = {
        "primary": measured["primary"],
        "counts": measured["counts"],
        "shortcut_primary": reference["primary"],
        "evaluation_sha256": sha256_file(out / "evaluation.json"),
    }
    print(
        json.dumps({"final": mode, "split": split, "primary": measured["primary"]}),
        flush=True,
    )
    return summary


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
            "selection_sha256": sha256_file(run / "selection.json"),
            "inputs": verified,
            "policy": "Frozen settings; no test-driven retuning.",
        },
    )
    results = {}
    for mode, winner in selection["modes"].items():
        results[mode] = {}
        for split in ["validation", *config["evaluation"]["test_sets"]]:
            results[mode][split] = _final_split(config, evaluator, run, final, mode, winner, split)
    runtime_benchmark(config, run, selection["modes"])
    write_json(
        run / "results.json",
        {
            "schema": "recap-hmm-results-v1",
            "finished_at_utc": now(),
            "selection_sha256": sha256_file(run / "selection.json"),
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
    except (OSError, ValueError) as exc:
        print(f"error: {one_line_error(exc)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
