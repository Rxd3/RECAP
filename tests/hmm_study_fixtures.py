"""A tiny but complete HMM study on disk, for in-process tests of the study code (tune, finalize, verify, package).

The study lives in a fake repository root inside a temporary folder:
- a 10-utterance release built from data/sample (modes A and C), with its split lists, manifests and stats;
- the experiment config;
- a real git checkout that stands in for the pinned shared evaluator.

Every module that reads REPO_ROOT is pointed at the fake root. The commands the study starts as subprocesses
(train_hmm, predict_hmm and the evaluator's evaluate_models) run in-process behind the same `subprocess.run`
call, so the real command code runs and counts towards coverage. Any other subprocess (git, the hardware
probe) still runs for real.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from punct_restore.cli import evaluate_models, package_hmm_handoff, predict_hmm, train_hmm
from punct_restore.data import load_label_sets
from punct_restore.models import (
    hmm_experiment_audit,
    hmm_experiment_commands,
    hmm_experiment_protocol,
    hmm_experiment_runtime,
    hmm_experiment_tune,
    hmm_experiment_verify,
    hmm_workflow,
)
from punct_restore.models.hmm_experiment_finalize import finalize
from punct_restore.models.hmm_experiment_tune import tune
from punct_restore.models.hmm_experiment_verify import verify

REPO = Path(__file__).resolve().parents[1]
SAMPLE = REPO / "data" / "sample"
MODES = ("A", "C")
SPLITS = ("train", "validation", "test", "test_amb")
PRIMARY_IDS = 8  # the first 8 sample IDs are scored; the other 2 are predicted but ignored by the scorer
RELEASE_ID = "unit-hmm-study-release"
POLICY = "m1-v2"  # the policy_version the sample rows carry
PROCESSED = "data/processed/unit"
SPLITS_DIR = "data/splits/unit"
CLEAN_CHECKOUT = {"commit": "0" * 40, "dirty": False}
SMOOTHING = (1.0, 0.1)  # the best candidate, smoothing 0.1 with cutoff 1, is then not the first one in grid order
CUTOFFS = (1, 2)

# modules that resolve study paths against REPO_ROOT; the fake root replaces it in each
ROOT_MODULES = (
    hmm_workflow,
    hmm_experiment_protocol,
    hmm_experiment_commands,
    hmm_experiment_audit,
    hmm_experiment_runtime,
    hmm_experiment_verify,
    package_hmm_handoff,
)
COMMAND_MAINS = {
    "scripts/train_hmm.py": train_hmm.main,
    "scripts/predict_hmm.py": predict_hmm.main,
    "scripts/evaluate_models.py": evaluate_models.main,
}


@dataclass(frozen=True)
class Study:
    """Where one study's fake repository, evaluator checkout, config and run folder are."""

    root: Path
    evaluator: Path
    config_path: Path
    run: Path


# ---- writing the fake repository ----


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8", newline="\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def sample_rows(name: str) -> list[dict[str, Any]]:
    """The rows of data/sample/<name>_validation.jsonl (name: mode_A, mode_C or case_joint_mask)."""
    lines = (SAMPLE / f"{name}_validation.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _train_counts(mode: str, rows: list[dict[str, Any]]) -> dict[str, int]:
    """Gold training labels per state, as the release's stats.json reports them (zero counts included)."""
    if mode == "C":
        pairs = (zip(row["punct_labels"], row["case_labels"], strict=True) for row in rows)
        targets = Counter(f"{p}|{c}" for row_pairs in pairs for p, c in row_pairs)
    else:
        targets = Counter(label for row in rows for label in row["punct_labels"])
    return {label: targets[label] for label in load_label_sets()[mode]}


def _write_split_files(root: Path) -> None:
    processed, splits = root / PROCESSED, root / SPLITS_DIR
    masks = sample_rows("case_joint_mask")
    stats: dict[str, Any] = {"labels_default_and_primary": {"train": {}}}
    for mode in MODES:
        rows = sample_rows(f"mode_{mode}")
        ids = [row["sample_id"] for row in rows]
        for split in SPLITS:
            write_jsonl(processed / f"mode_{mode}/{split}.jsonl", [{**row, "split": split} for row in rows])
        write_text(splits / f"default_train/mode_{mode}.ids", "".join(f"{i}\n" for i in ids))
        for split in SPLITS[1:]:
            write_text(splits / f"primary_eval/{split}_mode_{mode}.ids", "".join(f"{i}\n" for i in ids[:PRIMARY_IDS]))
        stats["labels_default_and_primary"]["train"][mode] = {"counts": _train_counts(mode, rows)}
    for split in SPLITS[1:]:
        write_jsonl(processed / f"masks/case_joint/{split}.jsonl", masks)
    write_text(processed / "stats.json", json.dumps(stats, indent=2) + "\n")
    write_text(processed / "SHA256SUMS", "unit fixture: the release files are listed in release_manifest.json\n")
    write_text(processed / "SPONSPEECH_LICENSE_AND_ATTRIBUTION.txt", "CC BY-NC 4.0 (unit fixture)\n")


def _entries(folder: Path, suffix: str) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(folder).as_posix(): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(folder.rglob(f"*{suffix}"))
    }


def _write_manifests(root: Path) -> None:
    processed, splits = root / PROCESSED, root / SPLITS_DIR
    split_files = _entries(splits, ".ids")
    write_text(splits / "SHA256SUMS", "".join(f"{e['sha256']}  {name}\n" for name, e in split_files.items()))
    manifest = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "policy_version": POLICY,
        "label_sets_sha256": sha256(root / "configs/label_sets.yaml"),
        "source": {"revision": sample_rows("mode_A")[0]["metadata"]["source_revision"]},
        "files": _entries(processed, ".jsonl"),
        "split_files": split_files,
    }
    write_text(processed / "release_manifest.json", json.dumps(manifest, indent=2) + "\n")


def _write_repo_files(root: Path) -> None:
    for name in ("configs/label_sets.yaml", "configs/hmm_sponspeech_v2.yaml", "requirements-hmm.txt"):
        write_text(root / name, (REPO / name).read_text(encoding="utf-8"))
    write_text(root / "experiments/HMM_RESULTS.md", "# HMM results (unit fixture)\n")
    write_text(root / "experiments/HMM_RUNBOOK.md", "# HMM runbook (unit fixture)\n")
    (root / "artifacts").mkdir()


def git(folder: Path, *args: str) -> str:
    """Run git in `folder` with no user or system config, so a signing or hook setting cannot interfere."""
    empty = folder.parent / "empty.gitconfig"
    empty.touch()
    env = {
        **os.environ, "GIT_CONFIG_GLOBAL": str(empty), "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "unit", "GIT_AUTHOR_EMAIL": "unit@example.invalid",
        "GIT_COMMITTER_NAME": "unit", "GIT_COMMITTER_EMAIL": "unit@example.invalid",
    }
    done = subprocess.run(["git", "-C", str(folder), *args], capture_output=True, text=True, env=env, check=True)
    return done.stdout.strip()


def make_evaluator(folder: Path) -> str:
    """A clean one-commit checkout with the files the evaluator identity hashes; returns its commit."""
    write_text(folder / "src/punct_restore/evaluation/metrics.py", "# unit evaluator\n")
    write_text(folder / "scripts/evaluate_models.py", "# unit evaluator entry point\n")
    write_text(folder / "configs/label_sets.yaml", (REPO / "configs/label_sets.yaml").read_text(encoding="utf-8"))
    git(folder, "init", "-q")
    git(folder, "add", ".")
    git(folder, "commit", "-q", "-m", "unit evaluator")
    return git(folder, "rev-parse", "HEAD")


def study_config(commit: str) -> dict[str, Any]:
    """The real study config, narrowed to modes A and C, a 2 x 2 grid and the unit release."""
    config = yaml.safe_load((REPO / "configs/hmm_sponspeech_v2.yaml").read_text(encoding="utf-8"))
    config.update(experiment_id="unit-hmm-study", dataset_release_id=RELEASE_ID, policy_version=POLICY,
                  processed_dir=PROCESSED, splits_dir=SPLITS_DIR, modes=list(MODES))
    config["search"].update(smoothing=list(SMOOTHING), min_token_count=list(CUTOFFS))
    config["evaluation"]["repository_commit"] = commit
    config["runtime"].update(warmup_sequences=1, repetitions=2)
    return config


def make_repo(base: Path) -> Study:
    """A fake repository with the unit release, config and evaluator, but no run yet."""
    root = base / "repo"
    _write_repo_files(root)
    _write_split_files(root)
    _write_manifests(root)
    commit = make_evaluator(root / "tmp/evaluator")
    config_path = root / "configs/hmm_unit.yaml"
    write_text(config_path, yaml.safe_dump(study_config(commit), sort_keys=False))
    return Study(root=root, evaluator=root / "tmp/evaluator", config_path=config_path, run=root / "artifacts/hmm/unit")


# ---- running the study in-process ----


def _in_process(real_run: Any) -> Any:
    """A subprocess.run that runs the study's own commands in this process and passes anything else on."""

    def run(args: list[Any], **kwargs: Any) -> Any:
        script = str(args[1]).replace("\\", "/") if len(args) > 1 else ""
        main = next((m for name, m in COMMAND_MAINS.items() if script.endswith(name)), None)
        if main is None or str(args[0]) != sys.executable:
            return real_run(args, **kwargs)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.chdir(kwargs["cwd"]), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main([str(arg) for arg in args[2:]])
            except SystemExit as stop:  # argparse usage errors; a bare exit is 0, as in a real interpreter
                code = stop.code if stop.code is None or isinstance(stop.code, int) else 1
        return subprocess.CompletedProcess(args, code or 0, out.getvalue(), err.getvalue())

    return run


def patch_study(monkeypatch: pytest.MonkeyPatch, study: Study) -> None:
    """Point the study code at `study`'s fake root, run its commands in-process and report a clean checkout.

    The code's own git calls also run without user or system config, so a setting such as
    status.showUntrackedFiles cannot change whether the evaluator checkout counts as clean.
    """
    empty = study.root.parent / "empty.gitconfig"
    empty.touch()
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for module in ROOT_MODULES:
        monkeypatch.setattr(module, "ROOT", study.root)
    monkeypatch.setattr(hmm_experiment_commands.subprocess, "run", _in_process(subprocess.run))
    monkeypatch.setattr(hmm_experiment_tune, "code_identity", lambda: dict(CLEAN_CHECKOUT))


def run_study(study: Study) -> None:
    """Tune, finalize and verify `study` (its modules must already be patched)."""
    tune(study.config_path, study.evaluator, study.run)
    finalize(study.run, study.evaluator)
    verify(study.run, study.evaluator, study.run / "verification")


_FINISHED: list[Study] = []


def finished_study(factory: pytest.TempPathFactory) -> Study:
    """One tuned, finalized and verified study for the whole session; tests use copies of it."""
    if not _FINISHED:
        study = make_repo(factory.mktemp("finished"))
        with pytest.MonkeyPatch.context() as monkeypatch:
            patch_study(monkeypatch, study)
            run_study(study)
        _FINISHED.append(study)
    return _FINISHED[0]


def copy_study(study: Study, base: Path) -> Study:
    """A private copy of a study's whole fake repository (run, evaluator checkout and release included)."""
    root = base / "repo"
    shutil.copytree(study.root, root)
    moved = {name: root / getattr(study, name).relative_to(study.root) for name in ("evaluator", "config_path", "run")}
    return replace(study, root=root, **moved)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rewrite_json(path: Path, change: Any) -> None:
    """Apply `change` to a JSON file's content and write it back (the study writes JSON with indent 2)."""
    text = json.dumps(change(read_json(path)), indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
