"""tune, finalize and verify report progress to a `log` callable: silent for a library caller, `print` for a CLI."""

from __future__ import annotations

import io
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import pytest
import yaml
from test_hmm_cli import ROOT

from punct_restore.cli import run_hmm_experiments, verify_hmm_experiments
from punct_restore.hashing import sha256_file
from punct_restore.models import hmm_experiment_finalize as finalize_module
from punct_restore.models import hmm_experiment_tune as tune_module
from punct_restore.models import hmm_experiment_verify as verify_module
from punct_restore.models.hmm_experiment_tune import tune


class Case(NamedTuple):
    """One of the three steps, wired to fakes: how to call the library, how to call the CLI and the lines it logs."""

    library: Callable[..., object]
    main: Callable[[list[str]], int]
    argv: list[str]
    lines: list[str]


def _candidate(study: Any, mode: str, smoothing: float, cutoff: int) -> dict[str, Any]:
    (study.run / f"mode_{mode}").mkdir(parents=True, exist_ok=True)
    name = f"s{smoothing:g}-c{cutoff}"
    return {
        "candidate": name, "smoothing": smoothing, "min_token_count": cutoff,
        "validation_primary": 0.5 if smoothing == 1.0 else 0.4,
        "model": f"mode_{mode}/candidates/{name}/model.json", "model_sha256": "m", "validation_report_sha256": "v",
    }


def _tune_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    config = yaml.safe_load((ROOT / "configs/hmm_sponspeech_v2.yaml").read_text(encoding="utf-8"))
    config["modes"] = ["A"]
    config["search"]["smoothing"] = [0.1, 1.0]
    config["search"]["min_token_count"] = [1]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(tune_module, "code_identity", lambda: {"commit": "c", "dirty": False})
    monkeypatch.setattr(tune_module, "check_evaluator", lambda root, commit: {"commit": commit, "files": {}})
    monkeypatch.setattr(tune_module, "authenticate", lambda cfg, phase: ({}, {}))
    monkeypatch.setattr(tune_module, "_tune_plan", lambda *args: {"plan": True})
    monkeypatch.setattr(tune_module, "_search_candidate", _candidate)
    lines = [json.dumps({"mode": "A", **_candidate(_Study(tmp_path / "x"), "A", s, 1)}) for s in (0.1, 1.0)]
    lines.append("All mode selections frozen: selection.json")
    run = tmp_path / "run"
    argv = ["tune", "--config", str(path), "--output-dir", str(run), "--evaluator-root", str(tmp_path / "ev")]
    return Case(lambda **kw: tune(path, tmp_path / "ev", run, **kw), run_hmm_experiments.main, argv, lines)


class _Study:
    def __init__(self, run: Path) -> None:
        self.run = run


def _finalize_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    config = {"processed_dir": "p", "splits_dir": "s", "evaluation": {"test_sets": ["test"]}}
    run = tmp_path / "run"
    run.mkdir()
    (run / "selection.json").write_text("{}", encoding="utf-8")
    plan = {"config": config, "tuning_inputs": {"release_manifest": "r"}}
    selection = {"modes": {"A": {"model": "mode_A/candidates/x/model.json"}}}
    primary = {"value": 0.5, "block": "punctuation/all"}

    def score(study: Any, mode: str, split: str, predictions: Path, out: Path) -> dict[str, Any]:
        out.parent.mkdir(parents=True, exist_ok=True)  # the predict step would have made it
        out.write_text("{}", encoding="utf-8")
        return {"primary": primary, "counts": {"n": 1}}

    monkeypatch.setattr(finalize_module, "verified_frozen_selection", lambda run, evaluator: (plan, selection))
    monkeypatch.setattr(finalize_module, "authenticate", lambda cfg, phase: ({}, {"release_manifest": "r"}))
    monkeypatch.setattr(finalize_module, "execute", lambda *args: {})
    monkeypatch.setattr(finalize_module, "score", score)
    monkeypatch.setattr(finalize_module, "runtime_benchmark", lambda *args: {})
    lines = [json.dumps({"final": "A", "split": split, "primary": primary}) for split in ("validation", "test")]
    lines.append("Final predictions, shared scores and runtime measurements complete.")
    argv = ["finalize", "--run-dir", str(run), "--evaluator-root", str(tmp_path / "ev")]
    return Case(
        lambda **kw: finalize_module.finalize(run, tmp_path / "ev", **kw), run_hmm_experiments.main, argv, lines
    )


def _verify_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    run, output = tmp_path / "run", tmp_path / "verification"
    run.mkdir()
    for name in ("plan.json", "selection.json"):
        (run / name).write_text("{}", encoding="utf-8")
    results = {"selection_sha256": sha256_file(run / "selection.json")}
    (run / "results.json").write_text(json.dumps(results), encoding="utf-8")
    plan, selection = {"config": {"processed_dir": "p", "modes": ["A"]}}, {"modes": {"A": {}, "B": {}}}
    monkeypatch.setattr(verify_module, "verified_frozen_selection", lambda run, evaluator: (plan, selection))
    monkeypatch.setattr(verify_module, "authenticate", lambda cfg, phase: ({}, {}))
    monkeypatch.setattr(verify_module, "read_json", lambda path: results if path.name == "results.json" else {})
    monkeypatch.setattr(verify_module, "_verify_mode", lambda *args: {})
    monkeypatch.setattr(verify_module, "_check_runtime", lambda *args: None)
    lines = [
        f"Verified mode {mode}: search, counts, probabilities, full refit/predictions, scores." for mode in "AB"
    ]
    lines.append(f"Complete HMM verification: {output / 'verification.json'}")
    argv = ["--run-dir", str(run), "--evaluator-root", str(tmp_path / "ev"), "--output-dir", str(output)]
    return Case(
        lambda **kw: verify_module.verify(run, tmp_path / "ev", output, **kw), verify_hmm_experiments.main, argv, lines
    )


CASES = {"tune": _tune_case, "finalize": _finalize_case, "verify": _verify_case}


@pytest.mark.parametrize("name", CASES)
def test_the_library_prints_nothing_unless_it_is_given_a_log(
        name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    CASES[name](tmp_path, monkeypatch).library()
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


@pytest.mark.parametrize("name", CASES)
def test_the_library_hands_each_progress_line_to_the_log_in_order(
        name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    case = CASES[name](tmp_path, monkeypatch)
    logged: list[str] = []
    case.library(log=logged.append)
    assert logged == case.lines
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("name", CASES)
def test_the_command_prints_the_same_lines_and_flushes_each_one(
        name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = CASES[name](tmp_path, monkeypatch)

    class CountingStdout(io.StringIO):
        flushes = 0

        def flush(self) -> None:
            type(self).flushes += 1

    stdout = CountingStdout()
    monkeypatch.setattr(sys, "stdout", stdout)
    assert case.main(case.argv) == 0
    assert stdout.getvalue() == "".join(f"{line}\n" for line in case.lines)
    assert CountingStdout.flushes >= len(case.lines)


def test_tune_refuses_a_dirty_checkout_when_called_as_a_library(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tune_module, "code_identity", lambda: {"commit": "c", "dirty": True})
    run = tmp_path / "run"
    with pytest.raises(ValueError, match="commit reviewed code/config before starting an official search"):
        tune(ROOT / "configs/hmm_sponspeech_v2.yaml", tmp_path / "ev", run)
    assert not run.exists()


def test_tune_does_not_refuse_a_checkout_whose_state_is_unknown(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Outside a git checkout `dirty` is None: that is not a refusal (the copy the golden runs uses it)."""
    case = _tune_case(tmp_path, monkeypatch)
    monkeypatch.setattr(tune_module, "code_identity", lambda: {"commit": None, "dirty": None})
    case.library()
    assert (tmp_path / "run" / "selection.json").is_file()
