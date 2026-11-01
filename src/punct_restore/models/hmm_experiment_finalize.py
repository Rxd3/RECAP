"""The HMM experiment's held-out step: predict and score every frozen winner, then measure its runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from punct_restore.hashing import sha256_file
from punct_restore.models.hmm_experiment_commands import Study, baseline_command, execute, predict_command, score
from punct_restore.models.hmm_experiment_frozen import verified_frozen_selection
from punct_restore.models.hmm_experiment_protocol import authenticate, now
from punct_restore.models.hmm_experiment_runtime import runtime_benchmark
from punct_restore.models.hmm_workflow import write_json
from punct_restore.progress import Log, discard_log


def _final_split(
    study: Study, final: Path, mode: str, winner: dict[str, Any], split: str, log: Log
) -> dict[str, Any]:
    """Predict and score one mode and split with its shortcut baseline; return its result summary."""
    run = study.run
    out = final / f"mode_{mode}/{split}"
    out.parent.mkdir(parents=True, exist_ok=True)
    command = predict_command(study, mode, split, winner, out)
    execute(command, run, out.parent / f"{split}.prediction.log")
    measured = score(study, mode, split, out / "predictions.jsonl", out / "evaluation.json")
    baseline = out / "shortcut.jsonl"
    execute(baseline_command(study, mode, split, baseline), run, out / "shortcut.log")
    reference = score(study, mode, split, baseline, out / "shortcut.evaluation.json")
    summary = {
        "primary": measured["primary"],
        "counts": measured["counts"],
        "shortcut_primary": reference["primary"],
        "evaluation_sha256": sha256_file(out / "evaluation.json"),
    }
    log(json.dumps({"final": mode, "split": split, "primary": measured["primary"]}))
    return summary


def finalize(run: Path, evaluator: Path, *, log: Log = discard_log) -> None:
    """Open the held-out data once, with the frozen settings, and write the final results of every mode.

    Progress lines go to `log`.
    """
    plan, selection = verified_frozen_selection(run, evaluator)
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
    study = Study(config=config, evaluator=evaluator, run=run)
    results: dict[str, dict[str, Any]] = {}
    for mode, winner in selection["modes"].items():
        results[mode] = {}
        for split in ["validation", *config["evaluation"]["test_sets"]]:
            results[mode][split] = _final_split(study, final, mode, winner, split, log)
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
    log("Final predictions, shared scores and runtime measurements complete.")
