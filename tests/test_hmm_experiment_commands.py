"""The command lines of the HMM experiment's steps: each builder takes the Study, and the lines never change."""

from __future__ import annotations

import sys
from typing import Any

from test_hmm_cli import ROOT

from punct_restore.models.hmm_experiment_commands import Study, baseline_command, predict_command, train_command

CONFIG: dict[str, Any] = {
    "experiment_id": "exp-1",
    "processed_dir": "data/processed/rel",
    "splits_dir": "data/splits/rel",
    "label_config": "configs/label_sets.yaml",
    "dataset_release_id": "rel-1",
    "policy_version": "pv-1",
}
RUN = ROOT / "artifacts/exp-1"
STUDY = Study(config=CONFIG, evaluator=ROOT / "evaluator", run=RUN)


def test_train_command_is_the_train_hmm_line_for_one_candidate() -> None:
    out = RUN / "mode_A/candidates/s0.1-c2"
    assert train_command(STUDY, "A", "s0.1-c2", 0.1, 2, out) == [
        sys.executable, "scripts/train_hmm.py",
        "--train", "data/processed/rel/mode_A/train.jsonl",
        "--train-ids", "data/splits/rel/default_train/mode_A.ids",
        "--output-dir", "artifacts/exp-1/mode_A/candidates/s0.1-c2",
        "--label-mode", "A",
        "--run-id", "exp-1-A-s0.1-c2",
        "--label-config", "configs/label_sets.yaml",
        "--smoothing", "0.1",
        "--min-token-count", "2",
        "--dataset-release-id", "rel-1",
        "--preprocessing-version", "pv-1",
        "--tokenizer-version", "pv-1",
        "--label-policy-version", "pv-1",
        "--split-manifest", "data/splits/rel/SHA256SUMS",
        "--release-manifest", "data/processed/rel/release_manifest.json",
        "--predict-input", "data/processed/rel/mode_A/validation.jsonl",
    ]


def test_predict_command_is_the_predict_hmm_line_for_one_winner_and_split() -> None:
    winner = {"model": "mode_C/candidates/s1-c1/model.json"}
    out = RUN / "final/mode_C/test"
    assert predict_command(STUDY, "C", "test", winner, out) == [
        sys.executable, "scripts/predict_hmm.py",
        "--model", "artifacts/exp-1/mode_C/candidates/s1-c1/model.json",
        "--input", "data/processed/rel/mode_C/test.jsonl",
        "--output-dir", "artifacts/exp-1/final/mode_C/test",
        "--all-records",
        "--selection-record", "artifacts/exp-1/selection.json",
        "--release-manifest", "data/processed/rel/release_manifest.json",
    ]


def test_baseline_command_is_the_pinned_evaluators_baseline_line() -> None:
    baseline = RUN / "final/mode_B/test_amb/shortcut.jsonl"
    assert baseline_command(STUDY, "B", "test_amb", baseline) == [
        sys.executable, "evaluator/scripts/evaluate_models.py", "baseline",
        "--gold", "data/processed/rel/mode_B/test_amb.jsonl",
        "--ids", "data/splits/rel/primary_eval/test_amb_mode_B.ids",
        "--out", "artifacts/exp-1/final/mode_B/test_amb/shortcut.jsonl",
    ]

