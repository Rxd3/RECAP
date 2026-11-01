"""The HMM experiment's validation search: train and score every candidate, then freeze the winners."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any

import yaml

from punct_restore.hashing import sha256_file
from punct_restore.models.hmm_experiment_commands import Study, execute, score, train_command
from punct_restore.models.hmm_experiment_identity import check_evaluator, implementation
from punct_restore.models.hmm_experiment_protocol import authenticate, choose, now, validate_config
from punct_restore.models.hmm_experiment_runtime import hardware
from punct_restore.models.hmm_workflow import code_identity, file_identity, portable_path, write_json
from punct_restore.progress import Log, discard_log


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


def _search_candidate(study: Study, mode: str, smoothing: float, cutoff: int) -> dict[str, Any]:
    """Train one candidate, score it on validation and describe it."""
    candidate = f"s{smoothing:g}-c{cutoff}"
    out = study.run / f"mode_{mode}/candidates/{candidate}"
    out.parent.mkdir(parents=True, exist_ok=True)
    args = train_command(study, mode, candidate, smoothing, cutoff, out)
    execute(args, study.run, out.parent / f"{candidate}.training.log")
    report = score(study, mode, "validation", out / "predictions.jsonl", out / "validation.json")
    return {
        "candidate": candidate,
        "smoothing": smoothing,
        "min_token_count": cutoff,
        "validation_primary": report["primary"]["value"],
        "model": (out / "model.json").relative_to(study.run).as_posix(),
        "model_sha256": sha256_file(out / "model.json"),
        "validation_report_sha256": sha256_file(out / "validation.json"),
    }


def _tune_mode(study: Study, mode: str, log: Log) -> tuple[dict[str, Any], str]:
    """Search one mode's grid, write its selection file and return the winner and that file's hash."""
    search = study.config["search"]
    candidates = []
    for cutoff in search["min_token_count"]:
        for smoothing in search["smoothing"]:
            record = _search_candidate(study, mode, smoothing, cutoff)
            candidates.append(record)
            log(json.dumps({"mode": mode, **record}))
    winner = choose(candidates)
    record = {
        "mode": mode,
        "frozen_at_utc": now(),
        "candidates": candidates,
        "selected": winner,
        "selection_block": search["selection_blocks"][mode],
        "train_only": True,
        "used_test_labels": False,
    }
    write_json(study.run / f"mode_{mode}/selection.json", record)
    return winner, sha256_file(study.run / f"mode_{mode}/selection.json")


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


def _load_config(config_path: Path) -> dict[str, Any]:
    """Read and validate the experiment config.

    Text that is not UTF-8, bad or too deeply nested YAML, a missing key or a wrong type is a ValueError that
    starts with the file name.
    """
    try:
        return validate_config(yaml.safe_load(config_path.read_text(encoding="utf-8")), config_path.name)
    except UnicodeDecodeError as exc:
        raise ValueError(f"{config_path.name}: not valid UTF-8") from exc
    except (KeyError, TypeError, yaml.YAMLError, RecursionError) as exc:  # RecursionError: nested too deeply
        raise ValueError(f"{config_path.name}: invalid experiment config: {exc}") from exc


def tune(config_path: Path, evaluator: Path, run: Path, *, log: Log = discard_log) -> None:
    """Run the validation-only search of every configured mode in a new run folder and freeze the winners.

    Refuses a checkout with uncommitted changes. Progress lines go to `log`.
    """
    if code_identity()["dirty"]:
        raise ValueError("commit reviewed code/config before starting an official search")
    config = _load_config(config_path)
    evaluator_identity = check_evaluator(evaluator, config["evaluation"]["repository_commit"])
    _, verified = authenticate(config, "tune")
    if run.exists():
        raise ValueError("run directory exists; choose a new experiment ID")
    plan = _tune_plan(config, config_path, evaluator_identity, verified)
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "plan.json", plan)
    study = Study(config=config, evaluator=evaluator, run=run)
    winners, mode_selection_hashes = {}, {}
    for mode in config["modes"]:
        winners[mode], mode_selection_hashes[mode] = _tune_mode(study, mode, log)
    # A single immutable gate freezes all configured modes before any held-out prediction.
    _freeze_selection(run, winners, mode_selection_hashes)
    log(f"All mode selections frozen: {portable_path(run / 'selection.json')}")
