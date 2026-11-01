"""The gate every held-out step passes: the frozen validation selection, checked against everything on disk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from punct_restore.hashing import sha256_file
from punct_restore.models.hmm_experiment_identity import check_evaluator, implementation
from punct_restore.models.hmm_experiment_protocol import choose, read_json, validate_config


def _check_gate(run: Path, selection: dict[str, Any], config: dict[str, Any]) -> None:
    """Require the selection to be bound to this plan and to cover every mode without test access."""
    if selection["plan_sha256"] != sha256_file(run / "plan.json"):
        raise ValueError("selection does not match its frozen plan")
    if (
        set(selection["modes"]) != set(config["modes"])
        or selection["used_test_labels"]
        or selection["training_includes_validation"]
    ):
        raise ValueError("all mode selections must freeze on validation before test access")


def _check_candidate(run: Path, candidate: dict[str, Any]) -> None:
    """Require one candidate's model and validation report to be the frozen evidence."""
    model_path = run / candidate["model"]
    report_path = model_path.parent / "validation.json"
    if (
        sha256_file(model_path) != candidate["model_sha256"]
        or sha256_file(report_path) != candidate["validation_report_sha256"]
        or read_json(report_path)["primary"]["value"] != candidate["validation_primary"]
    ):
        raise ValueError("candidate model/report differs from frozen validation evidence")


def _check_search_complete(config: dict[str, Any], mode_selection: dict[str, Any]) -> None:
    """Require exactly one candidate for every smoothing and cutoff of the configured grid."""
    expected = {
        (a, c) for a in config["search"]["smoothing"] for c in config["search"]["min_token_count"]
    }
    actual = {(c["smoothing"], c["min_token_count"]) for c in mode_selection["candidates"]}
    if actual != expected or len(actual) != len(mode_selection["candidates"]):
        raise ValueError("validation search is incomplete")


def _check_mode(
    run: Path, config: dict[str, Any], selection: dict[str, Any], mode: str, winner: dict[str, Any]
) -> None:
    """Require one mode's selection file, candidates and winner to be what was frozen."""
    if sha256_file(run / f"mode_{mode}/selection.json") != selection["mode_selection_sha256"][mode]:
        raise ValueError("mode selection changed after freezing")
    mode_selection = read_json(run / f"mode_{mode}/selection.json")
    _check_search_complete(config, mode_selection)
    for candidate in mode_selection["candidates"]:
        _check_candidate(run, candidate)
    if choose(mode_selection["candidates"]) != winner or mode_selection["selected"] != winner:
        raise ValueError("winner is inconsistent with the validation search")
    if sha256_file(run / winner["model"]) != winner["model_sha256"]:
        raise ValueError("selected artifact checksum mismatch")


def verified_frozen_selection(run: Path, evaluator: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check a run's frozen plan and selection against the code, evaluator and files on disk.

    Returns (plan, selection) once every part is as it was when the selection was frozen; raises ValueError for
    the first part that is not.
    """
    plan = read_json(run / "plan.json")
    selection = read_json(run / "selection.json")
    config = validate_config(plan["config"])
    _check_gate(run, selection, config)
    if implementation() != plan["implementation"]:
        raise ValueError("HMM implementation changed since selection")
    if check_evaluator(evaluator, config["evaluation"]["repository_commit"]) != plan["evaluator"]:
        raise ValueError("shared evaluator changed since selection")
    for mode, winner in selection["modes"].items():
        _check_mode(run, config, selection, mode, winner)
    return plan, selection
