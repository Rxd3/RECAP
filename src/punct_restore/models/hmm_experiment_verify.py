"""Independent audit of an HMM run: recount every score, refit the frozen winners and reproduce every prediction."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from punct_restore.hashing import sha256_file
from punct_restore.models.hmm import HMMModel
from punct_restore.models.hmm_experiment_audit import ReportOptions, check_report, probability_checks
from punct_restore.models.hmm_experiment_frozen import verified_frozen_selection
from punct_restore.models.hmm_experiment_protocol import authenticate, now, read_json
from punct_restore.models.hmm_records import SelectedRecords, read_jsonl
from punct_restore.models.hmm_workflow import file_identity, write_json
from punct_restore.paths import REPO_ROOT as ROOT
from punct_restore.prediction_io import read_ids


def _check_candidates(run: Path, config: dict[str, Any], mode: str) -> int:
    """Recount every validation candidate of one mode; returns how many were checked."""
    checked = 0
    candidates = read_json(run / f"mode_{mode}/selection.json")["candidates"]
    for candidate in candidates:
        folder = (run / candidate["model"]).parent
        check_report(
            config,
            mode,
            "validation",
            folder / "predictions.jsonl",
            folder / "validation.json",
            ReportOptions(primary_only=True),
        )
        checked += 1
    return checked


def _refit_winner(
    run: Path,
    config: dict[str, Any],
    output: Path,
    mode: str,
    winner: dict[str, Any],
    stats: dict[str, Any],
) -> tuple[Path, HMMModel, dict[str, str]]:
    """Check the frozen winner, refit it from the release and require identical artifact bytes."""
    model_path = run / winner["model"]
    model = HMMModel.load_model(model_path)
    probability_checks(model)
    saved = read_json(model_path)
    expected_support = stats["labels_default_and_primary"]["train"][mode]["counts"]
    if dict(zip(model.labels, saved["counts"]["state"], strict=True)) != expected_support:
        raise ValueError("learned state counts do not match the approved training subset")
    ids = tuple(read_ids(ROOT / config["splits_dir"] / f"default_train/mode_{mode}.ids", strict=True))
    if model.training_summary["num_sequences"] != len(ids):
        raise ValueError("model trained on the wrong sequence count")
    fresh = HMMModel(**saved["config"]).fit(
        SelectedRecords(
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
    if sha256_file(reproduced) != sha256_file(model_path):
        raise ValueError("independent refit does not reproduce the frozen artifact bytes")
    refit = {
        "result": "PASS",
        "model_sha256": sha256_file(reproduced),
        "metadata_policy": "original frozen provenance retained for byte comparison",
    }
    return model_path, fresh, refit


def _check_sidecar(run: Path, folder: Path, model_path: Path) -> None:
    """Require the prediction sidecar to cite the exact model, predictions and selection record."""
    manifest = read_json(folder / "predictions.manifest.json")
    if (
        manifest["model_sha256"] != sha256_file(model_path)
        or manifest["predictions_sha256"] != sha256_file(folder / "predictions.jsonl")
        or manifest["inputs"]["selection_record"]["sha256"] != sha256_file(run / "selection.json")
    ):
        raise ValueError("prediction sidecar provenance/hash mismatch")


def _check_predictions(
    config: dict[str, Any], mode: str, split: str, folder: Path, fresh: HMMModel
) -> None:
    """Require the refit model to reproduce the saved predictions without reading privileged fields."""
    gold_path = ROOT / config["processed_dir"] / f"mode_{mode}/{split}.jsonl"
    observations = list(SelectedRecords(gold_path, None))
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


def _verify_split(
    run: Path,
    config: dict[str, Any],
    mode: str,
    split: str,
    model_path: Path,
    fresh: HMMModel,
) -> dict[str, Any]:
    """Audit one final split: sidecar, reproduced predictions, recounted report and baseline."""
    folder = run / f"final/mode_{mode}/{split}"
    _check_sidecar(run, folder, model_path)
    _check_predictions(config, mode, split, folder, fresh)
    checked = check_report(
        config,
        mode,
        split,
        folder / "predictions.jsonl",
        folder / "evaluation.json",
        ReportOptions(examples=split != "validation"),
    )
    checked["shortcut"] = check_report(
        config,
        mode,
        split,
        folder / "shortcut.jsonl",
        folder / "shortcut.evaluation.json",
        ReportOptions(baseline=True),
    )
    return checked


def _verify_mode(
    run: Path,
    config: dict[str, Any],
    output: Path,
    stats: dict[str, Any],
    mode: str,
    winner: dict[str, Any],
) -> dict[str, Any]:
    """Audit one mode: its search candidates, refit winner and final splits."""
    mode_report = {"candidates_independently_checked": 0, "final_reports": {}}
    mode_report["candidates_independently_checked"] = _check_candidates(run, config, mode)
    model_path, fresh, refit = _refit_winner(run, config, output, mode, winner, stats)
    mode_report["independent_refit"] = refit
    for split in ["validation", "test", "test_amb"]:
        mode_report["final_reports"][split] = _verify_split(run, config, mode, split, model_path, fresh)
    return mode_report


def _check_runtime(run: Path, config: dict[str, Any]) -> None:
    """Require complete, positive runtime measurements for every configured mode."""
    runtime = read_json(run / "runtime.json")
    if set(runtime["modes"]) != set(config["modes"]):
        raise ValueError("runtime measurements are incomplete")
    for mode in config["modes"]:
        values = runtime["modes"][mode]["prediction_seconds"]
        if len(values) != config["runtime"]["repetitions"] or any(v <= 0 for v in values):
            raise ValueError("invalid runtime repetitions")


def verify(run: Path, evaluator: Path, output: Path) -> None:
    """Audit a finished run into a new output folder, writing verification.json when every check passes."""
    plan, selection = verified_frozen_selection(run, evaluator)
    config = plan["config"]
    if output.exists():
        raise ValueError("verification output already exists; use a new verification directory")
    if not (run / "results.json").exists():
        raise ValueError("finalization is incomplete")
    authenticate(config, "tune")
    authenticate(config, "finalize")
    results = read_json(run / "results.json")
    if results["selection_sha256"] != sha256_file(run / "selection.json"):
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
        report["modes"][mode] = _verify_mode(run, config, output, stats, mode, winner)
        print(
            f"Verified mode {mode}: search, counts, probabilities, full refit/predictions, scores.",
            flush=True,
        )
    _check_runtime(run, config)
    report["runtime_checked"] = True
    report["result"] = "PASS"
    write_json(output / "verification.json", report)
    print(f"Complete HMM verification: {output / 'verification.json'}", flush=True)
