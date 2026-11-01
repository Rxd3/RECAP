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
CODE = {"commit": "c", "dirty": False}
TUNING_INPUTS = {"release_manifest": {"path": "release_manifest.json", "sha256": "0" * 64}}
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
    write_json(run / "plan.json", {
        "config": config, "implementation": IMPLEMENTATION, "evaluator": EVALUATOR, "code": CODE,
        "tuning_inputs": TUNING_INPUTS,
    })
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


@pytest.mark.parametrize(
    ("item", "message"),
    [
        ({}, "selection.json: missing candidates.0.smoothing"),
        (["O"], "selection.json: candidates.0 is not a JSON object"),
        (1, "selection.json: candidates.0 is not a JSON object"),
        (None, "selection.json: candidates.0 is not a JSON object"),
        ("text", "selection.json: candidates.0 is not a JSON object"),
    ],
    ids=["object", "list", "number", "null", "text"],
)
def test_a_candidate_that_is_no_candidate_record_is_refused_naming_its_place(
        item: Any, message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The mode selection is hash-bound to the frozen one, so only a consistent forgery reaches the shape check."""
    run = _study(tmp_path, monkeypatch)
    path = run / "mode_A/selection.json"
    _rewrite(path, lambda selection: {**selection, "candidates": [item, *selection["candidates"][1:]]})
    _rewrite(run / "selection.json", lambda frozen: {**frozen, "mode_selection_sha256": {"A": sha256_file(path)}})
    with pytest.raises(ValueError) as refused:
        verified_frozen_selection(run, tmp_path / "evaluator")
    assert str(refused.value) == message


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


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b'{"a": 1}', r"^plan\.json: missing config$"),
        (b"[1]", r"^plan\.json: not a JSON object$"),
        (b"null", r"^plan\.json: not a JSON object$"),
        (b'{"config": [], "implementation": {}, "evaluator": {}}', r"^plan\.json: config is not a JSON object$"),
        (b'{"config": {}, "evaluator": {}}', r"^plan\.json: missing implementation$"),
        (b'{"config": {}, "implementation": {}}', r"^plan\.json: missing evaluator$"),
        (b"[" * 100_000, r"^plan\.json: not valid JSON$"),
        (b"\xff\xfe", r"^plan\.json: not valid UTF-8$"),
        (b'{"config": {}, "implementation": {}, "evaluator": {}, "code": {}, "tuning_inputs": {"release_manifest": 1}}',
         r"^plan\.json: invalid experiment config: missing schema_version$"),
        (b'{"config": {}, "implementation": {}, "evaluator": {}, "tuning_inputs": {"release_manifest": {}}}',
         r"^plan\.json: missing code$"),
        (b'{"config": {}, "implementation": {}, "evaluator": {}, "code": {}}',
         r"^plan\.json: missing tuning_inputs$"),
        (b'{"config": {}, "implementation": {}, "evaluator": {}, "code": {}, "tuning_inputs": []}',
         r"^plan\.json: tuning_inputs is not a JSON object$"),
        (b'{"config": {}, "implementation": {}, "evaluator": {}, "code": {}, "tuning_inputs": {}}',
         r"^plan\.json: missing tuning_inputs\.release_manifest$"),
    ],
    ids=["no-keys", "array", "null", "config-array", "no-implementation", "no-evaluator", "nested", "not-utf8",
         "config-lacks-a-key", "no-code", "no-tuning-inputs", "tuning-inputs-array", "no-release-manifest-identity"],
)
def test_a_plan_of_the_wrong_shape_is_one_value_error_that_starts_with_its_file_name(
        content: bytes, message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _study(tmp_path, monkeypatch)
    (run / "plan.json").write_bytes(content)
    with pytest.raises(ValueError, match=message):
        verified_frozen_selection(run, tmp_path / "evaluator")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b'{"a": 1}', r"^selection\.json: missing plan_sha256$"),
        (b"[1]", r"^selection\.json: not a JSON object$"),
        (b'{"plan_sha256": "x", "modes": [], "used_test_labels": false, "training_includes_validation": false, '
         b'"mode_selection_sha256": {}}', r"^selection\.json: modes is not a JSON object$"),
        (b'{"plan_sha256": "x", "modes": {}, "used_test_labels": false, "training_includes_validation": false}',
         r"^selection\.json: missing mode_selection_sha256$"),
        (b"[" * 100_000, r"^selection\.json: not valid JSON$"),
        (b"\xff\xfe", r"^selection\.json: not valid UTF-8$"),
    ],
    ids=["no-keys", "array", "modes-array", "no-mode-hashes", "nested", "not-utf8"],
)
def test_a_selection_of_the_wrong_shape_is_one_value_error_that_starts_with_its_file_name(
        content: bytes, message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _study(tmp_path, monkeypatch)
    (run / "selection.json").write_bytes(content)
    with pytest.raises(ValueError, match=message):
        verified_frozen_selection(run, tmp_path / "evaluator")


def test_a_key_error_after_the_files_were_read_is_still_a_bug_and_propagates(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the shape of plan.json and selection.json is relabelled; a KeyError further on is not a file problem."""
    run = _study(tmp_path, monkeypatch)

    def broken(root: Path, commit: str) -> dict[str, Any]:
        raise KeyError("bug")

    monkeypatch.setattr(hmm_experiment_frozen, "check_evaluator", broken)
    with pytest.raises(KeyError):
        verified_frozen_selection(run, tmp_path / "evaluator")
