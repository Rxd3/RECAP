"""verified_frozen_selection is the gate every held-out step passes: each tampering is refused, in a fixed order."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_hmm_cli import ROOT

from punct_restore.hashing import sha256_file
from punct_restore.models import hmm_experiment_frozen
from punct_restore.models.hmm_experiment_frozen import verified_frozen_selection
from punct_restore.models.hmm_workflow import write_json

IMPLEMENTATION = {"models/hmm.py": "a" * 64}
EVALUATOR = {"commit": "pinned", "files": {}}
GRID = [(0.1, 1), (1.0, 1)]  # (smoothing, min_token_count)
SCORES = {(0.1, 1): 0.4, (1.0, 1): 0.6}  # validation primary score of each grid point


def _candidate(run: Path, smoothing: float, cutoff: int) -> dict[str, Any]:
    name = f"s{smoothing:g}-c{cutoff}"
    folder = run / f"mode_A/candidates/{name}"
    folder.mkdir(parents=True)
    score = SCORES[smoothing, cutoff]
    (folder / "model.json").write_text(f"model {name}", encoding="utf-8")
    (folder / "validation.json").write_text(json.dumps({"primary": {"value": score}}), encoding="utf-8")
    return {
        "candidate": name, "smoothing": smoothing, "min_token_count": cutoff, "validation_primary": score,
        "model": f"mode_A/candidates/{name}/model.json", "model_sha256": sha256_file(folder / "model.json"),
        "validation_report_sha256": sha256_file(folder / "validation.json"),
    }


def _study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, searched: list[tuple[float, int]] = GRID) -> Path:
    """A one-mode frozen run on disk, consistent in every part; the code and evaluator identities are pinned."""
    config = yaml.safe_load((ROOT / "configs/hmm_sponspeech_v2.yaml").read_text(encoding="utf-8"))
    config["modes"] = ["A"]
    config["search"]["smoothing"] = [0.1, 1.0]
    config["search"]["min_token_count"] = [1]
    run = tmp_path / "run"
    candidates = [_candidate(run, *point) for point in searched]
    winner = max(candidates, key=lambda c: c["validation_primary"])
    write_json(run / "mode_A/selection.json", {"candidates": candidates, "selected": winner})
    write_json(run / "plan.json", {"config": config, "implementation": IMPLEMENTATION, "evaluator": EVALUATOR})
    write_json(run / "selection.json", {
        "plan_sha256": sha256_file(run / "plan.json"), "modes": {"A": winner},
        "mode_selection_sha256": {"A": sha256_file(run / "mode_A/selection.json")},
        "used_test_labels": False, "training_includes_validation": False,
    })
    monkeypatch.setattr(hmm_experiment_frozen, "implementation", lambda: IMPLEMENTATION)
    monkeypatch.setattr(hmm_experiment_frozen, "check_evaluator", lambda root, commit: EVALUATOR)
    return run


def _rewrite(path: Path, change: Any) -> None:
    """Replace a JSON file (write_json never overwrites) with `change` applied to its content."""
    value = json.loads(path.read_text(encoding="utf-8"))
    path.unlink()
    write_json(path, change(value))


def test_a_consistent_run_returns_its_plan_and_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _study(tmp_path, monkeypatch)
    plan, selection = verified_frozen_selection(run, tmp_path / "evaluator")
    assert plan["evaluator"] == EVALUATOR and selection["modes"]["A"]["candidate"] == "s1-c1"


def test_a_plan_edited_after_selection_is_refused_before_the_code_and_evaluator_are_compared(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _study(tmp_path, monkeypatch)
    (run / "plan.json").write_text((run / "plan.json").read_text(encoding="utf-8") + " ", encoding="utf-8")
    monkeypatch.setattr(hmm_experiment_frozen, "implementation", lambda: {"changed": "b" * 64})
    with pytest.raises(ValueError, match="selection does not match its frozen plan"):
        verified_frozen_selection(run, tmp_path / "evaluator")


def test_a_selection_that_used_test_labels_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _study(tmp_path, monkeypatch)
    _rewrite(run / "selection.json", lambda selection: {**selection, "used_test_labels": True})
    with pytest.raises(ValueError, match="must freeze on validation before test access"):
        verified_frozen_selection(run, tmp_path / "evaluator")


def test_code_that_changed_since_selection_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _study(tmp_path, monkeypatch)
    monkeypatch.setattr(hmm_experiment_frozen, "implementation", lambda: {"models/hmm.py": "b" * 64})
    with pytest.raises(ValueError, match="HMM implementation changed since selection"):
        verified_frozen_selection(run, tmp_path / "evaluator")


def test_an_evaluator_that_changed_since_selection_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _study(tmp_path, monkeypatch)
    monkeypatch.setattr(hmm_experiment_frozen, "check_evaluator", lambda root, commit: {"commit": "moved", "files": {}})
    with pytest.raises(ValueError, match="shared evaluator changed since selection"):
        verified_frozen_selection(run, tmp_path / "evaluator")


def test_a_mode_selection_edited_after_freezing_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _study(tmp_path, monkeypatch)
    path = run / "mode_A/selection.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="mode selection changed after freezing"):
        verified_frozen_selection(run, tmp_path / "evaluator")


def test_a_search_that_skipped_a_grid_point_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _study(tmp_path, monkeypatch, searched=[(1.0, 1)])
    with pytest.raises(ValueError, match="validation search is incomplete"):
        verified_frozen_selection(run, tmp_path / "evaluator")


def test_a_candidate_model_replaced_after_freezing_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _study(tmp_path, monkeypatch)
    (run / "mode_A/candidates/s0.1-c1/model.json").write_text("another model", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate model/report differs from frozen validation evidence"):
        verified_frozen_selection(run, tmp_path / "evaluator")


def test_a_candidate_report_whose_score_disagrees_with_the_record_is_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The report file and every hash around it are consistent: only the score differs from the recorded one."""
    run = _study(tmp_path, monkeypatch)
    report = run / "mode_A/candidates/s0.1-c1/validation.json"
    report.write_text(json.dumps({"primary": {"value": 0.9}}), encoding="utf-8")

    def rehash(selection: dict[str, Any]) -> dict[str, Any]:
        selection["candidates"][0]["validation_report_sha256"] = sha256_file(report)
        return selection

    _rewrite(run / "mode_A/selection.json", rehash)
    _rewrite(run / "selection.json", lambda frozen: {
        **frozen, "mode_selection_sha256": {"A": sha256_file(run / "mode_A/selection.json")}})
    with pytest.raises(ValueError, match="candidate model/report differs from frozen validation evidence"):
        verified_frozen_selection(run, tmp_path / "evaluator")


def test_a_winner_that_is_not_the_best_validation_candidate_is_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _study(tmp_path, monkeypatch)
    worse = _rewrite_winner_to_the_worse_candidate(run)
    assert worse["candidate"] == "s0.1-c1"
    with pytest.raises(ValueError, match="winner is inconsistent with the validation search"):
        verified_frozen_selection(run, tmp_path / "evaluator")


def _rewrite_winner_to_the_worse_candidate(run: Path) -> dict[str, Any]:
    """Name the lower-scoring candidate as the winner in both selection files, keeping every hash consistent."""
    path = run / "mode_A/selection.json"
    worse = json.loads(path.read_text(encoding="utf-8"))["candidates"][0]
    _rewrite(path, lambda selection: {**selection, "selected": worse})
    _rewrite(run / "selection.json", lambda frozen: {
        **frozen, "modes": {"A": worse}, "mode_selection_sha256": {"A": sha256_file(path)}})
    return worse
