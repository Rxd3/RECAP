"""The frozen-selection gate (verified_frozen_selection) and finalize, on copies of the shared tiny study."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from hmm_study_fixtures import (
    MODES,
    PROCESSED,
    Study,
    copy_study,
    finished_study,
    git,
    patch_study,
    read_json,
    rewrite_json,
    sha256,
    write_text,
)

from punct_restore.models.hmm_experiment_finalize import finalize
from punct_restore.models.hmm_experiment_frozen import verified_frozen_selection

Json = dict[str, Any]
Mutation = Callable[[Study], None]
ModeMutation = Callable[[Study, str], None]

SPLITS = ("validation", "test", "test_amb")
COMPLETE = "Final predictions, shared scores and runtime measurements complete."
FROZEN_ON_VALIDATION = "all mode selections must freeze on validation before test access"
EVIDENCE_DIFFERS = "candidate model/report differs from frozen validation evidence"
PLAN_MISMATCH = "selection does not match its frozen plan"
INCONSISTENT_WINNER = "winner is inconsistent with the validation search"


# ---- fixtures ----


@pytest.fixture
def study(tmp_path_factory: pytest.TempPathFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Study:
    """A private copy of the finished study with the study code patched to it."""
    copy = copy_study(finished_study(tmp_path_factory), tmp_path)
    patch_study(monkeypatch, copy)
    return copy


@pytest.fixture
def tuned(study: Study) -> Study:
    """The copy as it stood after tuning: no final outputs, results, runtime or verification yet."""
    for name in ("final", "verification"):
        shutil.rmtree(study.run / name)
    for name in ("results.json", "runtime.json"):
        (study.run / name).unlink()
    return study


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory) -> Study:
    """The shared finished study itself; tests only read it."""
    return finished_study(tmp_path_factory)


# ---- helpers ----


def _snapshot(folder: Path) -> dict[str, tuple[int, int]]:
    """Every file under `folder` with its size and modification time, to prove a refusal wrote nothing."""
    stats = {p: p.stat() for p in folder.rglob("*") if p.is_file()}
    return {p.relative_to(folder).as_posix(): (s.st_size, s.st_mtime_ns) for p, s in stats.items()}


def _append(path: Path, data: bytes = b" ") -> None:
    """Add bytes to a file's end (a trailing space keeps JSON valid)."""
    path.write_bytes(path.read_bytes() + data)


def _drop(content: Json, key: str) -> Json:
    return {k: v for k, v in content.items() if k != key}


def _edit_plan(study: Study, change: Callable[[Json], Any]) -> None:
    rewrite_json(study.run / "plan.json", change)


def _edit_selection(study: Study, change: Callable[[Json], Any]) -> None:
    rewrite_json(study.run / "selection.json", change)


def _mode_selection(study: Study, mode: str) -> Json:
    return read_json(study.run / f"mode_{mode}/selection.json")


def _other(mode_selection: Json) -> Json:
    """The first candidate that is not the selected one."""
    return next(c for c in mode_selection["candidates"] if c != mode_selection["selected"])


def _loser(study: Study, mode: str) -> Json:
    return _other(_mode_selection(study, mode))


def _edit_mode(study: Study, mode: str, change: Callable[[Json], Json]) -> None:
    """Change a mode's selection.json and re-freeze its checksum in selection.json, so earlier checks still pass."""
    path = study.run / f"mode_{mode}/selection.json"
    rewrite_json(path, change)
    digest = sha256(path)
    _edit_selection(study, lambda s: {**s, "mode_selection_sha256": {**s["mode_selection_sha256"], mode: digest}})


def _evaluator_unpinned(study: Study) -> str:
    """The refusal of the evaluator folder: its name and the first 12 characters of the commit the plan pins."""
    pinned = yaml.safe_load(study.config_path.read_text(encoding="utf-8"))["evaluation"]["repository_commit"]
    return f"{study.evaluator.name}: evaluator checkout must be clean at the pinned commit {pinned[:12]}"


def _assert_refused(study: Study, message: str) -> None:
    """The gate raises exactly `message` and writes nothing into the run."""
    before = _snapshot(study.run)
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        verified_frozen_selection(study.run, study.evaluator)
    assert _snapshot(study.run) == before


# ---- the gate: a consistent run ----


def test_consistent_run_returns_its_plan_and_selection_and_writes_nothing(study: Study) -> None:
    before = _snapshot(study.run)

    plan, selection = verified_frozen_selection(study.run, study.evaluator)

    assert plan == read_json(study.run / "plan.json")
    assert selection == read_json(study.run / "selection.json")
    assert _snapshot(study.run) == before


def test_missing_plan_is_a_file_not_found_error(study: Study) -> None:
    (study.run / "plan.json").unlink()

    with pytest.raises(FileNotFoundError) as refused:
        verified_frozen_selection(study.run, study.evaluator)

    assert Path(refused.value.filename).name == "plan.json"


# ---- the gate: files of the wrong shape ----

SHAPE_CASES = [
    pytest.param(lambda s: write_text(s.run / "plan.json", "{"), "plan.json: not valid JSON", id="plan-not-json"),
    pytest.param(lambda s: write_text(s.run / "plan.json", "[]"), "plan.json: not a JSON object", id="plan-list"),
    pytest.param(
        lambda s: _edit_plan(s, lambda p: _drop(p, "implementation")),
        "plan.json: missing implementation",
        id="plan-no-implementation",
    ),
    pytest.param(
        lambda s: _edit_plan(s, lambda p: {**p, "config": []}),
        "plan.json: config is not a JSON object",
        id="plan-config-list",
    ),
    pytest.param(
        lambda s: write_text(s.run / "selection.json", "[]"), "selection.json: not a JSON object", id="selection-list"
    ),
    pytest.param(
        lambda s: _edit_selection(s, lambda x: _drop(x, "plan_sha256")),
        "selection.json: missing plan_sha256",
        id="selection-no-plan-hash",
    ),
    pytest.param(
        lambda s: _edit_selection(s, lambda x: {**x, "modes": []}),
        "selection.json: modes is not a JSON object",
        id="selection-modes-list",
    ),
    pytest.param(
        lambda s: _edit_selection(s, lambda x: {**x, "mode_selection_sha256": []}),
        "selection.json: mode_selection_sha256 is not a JSON object",
        id="selection-mode-hashes-list",
    ),
    pytest.param(
        lambda s: _edit_plan(s, lambda p: {**p, "config": _drop(p["config"], "search")}),
        "plan.json: invalid experiment config: missing search",
        id="plan-config-no-search",
    ),
    pytest.param(
        lambda s: _edit_plan(s, lambda p: {**p, "config": {**p["config"], "search": 5}}),
        "plan.json: invalid experiment config: search is not a JSON object",
        id="plan-config-search-not-an-object",
    ),
    pytest.param(
        lambda s: _edit_plan(s, lambda p: {**p, "config": {**p["config"], "modes": []}}),
        "plan.json: invalid experiment config: schema_version must be 1 and modes must not be empty",
        id="plan-config-no-modes",
    ),
]


@pytest.mark.parametrize(("mutate", "message"), SHAPE_CASES)
def test_gate_refuses_a_plan_or_selection_of_the_wrong_shape(study: Study, mutate: Mutation, message: str) -> None:
    mutate(study)

    _assert_refused(study, message)


# ---- the gate: plan binding, test access and code identity ----

BINDING_CASES = [
    pytest.param(lambda s: _append(s.run / "plan.json"), PLAN_MISMATCH, id="plan-edited-by-a-space"),
    pytest.param(
        lambda s: _edit_selection(s, lambda x: {**x, "used_test_labels": True}),
        FROZEN_ON_VALIDATION,
        id="test-labels-used",
    ),
    pytest.param(
        lambda s: _edit_selection(s, lambda x: {**x, "training_includes_validation": True}),
        FROZEN_ON_VALIDATION,
        id="training-includes-validation",
    ),
    pytest.param(
        lambda s: _edit_selection(s, lambda x: {**x, "modes": _drop(x["modes"], "C")}),
        FROZEN_ON_VALIDATION,
        id="only-mode-A-selected",
    ),
]


@pytest.mark.parametrize(("mutate", "message"), BINDING_CASES)
def test_gate_refuses_a_selection_not_bound_to_its_plan_or_not_frozen_on_validation(
    study: Study, mutate: Mutation, message: str
) -> None:
    mutate(study)

    _assert_refused(study, message)


def _edit_bound_plan(study: Study, change: Callable[[Json], Any]) -> None:
    """Edit the plan and re-bind the selection to it, so only the edited part differs from the code on disk."""
    _edit_plan(study, change)
    _edit_selection(study, lambda s: {**s, "plan_sha256": sha256(study.run / "plan.json")})


def test_gate_refuses_changed_hmm_code(study: Study) -> None:
    hmm_file = "src/punct_restore/models/hmm.py"
    _edit_bound_plan(study, lambda p: {**p, "implementation": {**p["implementation"], hmm_file: "0" * 64}})

    _assert_refused(study, "HMM implementation changed since selection")


def test_gate_refuses_changed_evaluator_files(study: Study) -> None:
    _edit_bound_plan(study, lambda p: {**p, "evaluator": {**p["evaluator"], "files": {}}})

    _assert_refused(study, "shared evaluator changed since selection")


EVALUATOR_CASES = [
    pytest.param(
        lambda s: (write_text(s.evaluator / "NOTES.txt", "later\n"), git(s.evaluator, "add", "."),
                   git(s.evaluator, "commit", "-q", "-m", "later")),
        id="new-commit-clean-tree",
    ),
    pytest.param(lambda s: write_text(s.evaluator / "stray.txt", "x\n"), id="untracked-file"),
    pytest.param(
        lambda s: write_text(s.evaluator / "scripts/evaluate_models.py", "# edited\n"), id="tracked-file-edited"
    ),
]


@pytest.mark.parametrize("mutate", EVALUATOR_CASES)
def test_gate_refuses_an_evaluator_that_is_not_clean_at_the_pinned_commit(study: Study, mutate: Mutation) -> None:
    mutate(study)

    _assert_refused(study, _evaluator_unpinned(study))


# ---- the gate: per-mode evidence ----


def _edit_mode_selection_file(study: Study, mode: str) -> None:
    _append(study.run / f"mode_{mode}/selection.json")


def _change_loser_model(study: Study, mode: str) -> None:
    _append(study.run / _loser(study, mode)["model"])


def _change_loser_report(study: Study, mode: str) -> None:
    _append((study.run / _loser(study, mode)["model"]).parent / "validation.json")


def _change_winner_model(study: Study, mode: str) -> None:
    _append(study.run / _mode_selection(study, mode)["selected"]["model"])


def _misrecord_loser_primary(study: Study, mode: str) -> None:
    loser = _loser(study, mode)
    halved = {**loser, "validation_primary": loser["validation_primary"] / 2}
    _edit_mode(study, mode, lambda m: {**m, "candidates": [halved if c == loser else c for c in m["candidates"]]})


def _drop_last_candidate(study: Study, mode: str) -> None:
    _edit_mode(study, mode, lambda m: {**m, "candidates": m["candidates"][:-1]})


def _repeat_last_candidate(study: Study, mode: str) -> None:
    _edit_mode(study, mode, lambda m: {**m, "candidates": [*m["candidates"], m["candidates"][-1]]})


def _select_other_in_mode_file(study: Study, mode: str) -> None:
    _edit_mode(study, mode, lambda m: {**m, "selected": _other(m)})


def _select_other_in_selection(study: Study, mode: str) -> None:
    other = _loser(study, mode)
    _edit_selection(study, lambda s: {**s, "modes": {**s["modes"], mode: other}})


def _select_other_in_both_files(study: Study, mode: str) -> None:
    """The two files agree on a winner, but it is not the best candidate on validation."""
    other = _loser(study, mode)
    _edit_mode(study, mode, lambda m: {**m, "selected": other})
    _edit_selection(study, lambda s: {**s, "modes": {**s["modes"], mode: other}})


MODE_CASES = [
    ("mode-selection-edited", _edit_mode_selection_file, "mode selection changed after freezing"),
    ("loser-model-changed", _change_loser_model, EVIDENCE_DIFFERS),
    ("loser-report-changed", _change_loser_report, EVIDENCE_DIFFERS),
    ("winner-model-changed", _change_winner_model, EVIDENCE_DIFFERS),
    ("loser-primary-differs-from-report", _misrecord_loser_primary, EVIDENCE_DIFFERS),
    ("candidate-missing", _drop_last_candidate, "validation search is incomplete"),
    ("candidate-duplicated", _repeat_last_candidate, "validation search is incomplete"),
    ("mode-file-selects-other", _select_other_in_mode_file, INCONSISTENT_WINNER),
    ("selection-names-other", _select_other_in_selection, INCONSISTENT_WINNER),
    ("both-files-name-other", _select_other_in_both_files, INCONSISTENT_WINNER),
]
# every case runs on the first mode; these also run on the others, which proves the gate checks each mode
ON_EVERY_MODE = {"mode-selection-edited", "loser-model-changed", "selection-names-other"}
MODE_PARAMS = [
    pytest.param(mode, mutate, message, id=f"{name}-{mode}")
    for name, mutate, message in MODE_CASES
    for mode in MODES
    if mode == MODES[0] or name in ON_EVERY_MODE
]


@pytest.mark.parametrize(("mode", "mutate", "message"), MODE_PARAMS)
def test_gate_refuses_a_mode_whose_evidence_changed(
    study: Study, mode: str, mutate: ModeMutation, message: str
) -> None:
    mutate(study, mode)

    _assert_refused(study, message)


# ---- finalize: what the finished study wrote ----


def test_test_access_record_binds_the_selection_and_lists_the_opened_inputs(finished: Study) -> None:
    access = read_json(finished.run / "final/test_access.json")

    assert access["selection_sha256"] == sha256(finished.run / "selection.json")
    assert {"mode_A/test.jsonl", "mode_C/test_amb.jsonl", "masks/case_joint/test.jsonl"} <= set(access["inputs"])
    assert not [name for name in access["inputs"] if "train" in name]
    tuning_inputs = read_json(finished.run / "plan.json")["tuning_inputs"]
    assert access["inputs"]["release_manifest"] == tuning_inputs["release_manifest"]


def test_results_file_binds_the_selection_and_covers_every_mode_and_split(finished: Study) -> None:
    results = read_json(finished.run / "results.json")

    assert results["schema"] == "recap-hmm-results-v1"
    assert results["selection_sha256"] == sha256(finished.run / "selection.json")
    assert set(results["results"]) == set(MODES)
    assert all(set(results["results"][mode]) == set(SPLITS) for mode in MODES)


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("mode", MODES)
def test_result_summary_repeats_the_shared_scorer_output(finished: Study, mode: str, split: str) -> None:
    folder = finished.run / f"final/mode_{mode}/{split}"
    summary = read_json(finished.run / "results.json")["results"][mode][split]
    evaluation = read_json(folder / "evaluation.json")

    assert set(summary) == {"primary", "counts", "shortcut_primary", "evaluation_sha256"}
    assert summary["primary"] == evaluation["primary"]
    assert summary["counts"] == evaluation["counts"]
    assert summary["shortcut_primary"] == read_json(folder / "shortcut.evaluation.json")["primary"]
    assert summary["evaluation_sha256"] == sha256(folder / "evaluation.json")


def test_runtime_file_times_every_mode_for_each_repetition(finished: Study) -> None:
    modes = read_json(finished.run / "runtime.json")["modes"]

    assert set(modes) == set(MODES)
    for mode in MODES:
        assert len(modes[mode]["prediction_seconds"]) == 2
        assert all(seconds > 0 for seconds in modes[mode]["prediction_seconds"])


# ---- finalize: running and refusing ----


def test_finalize_reports_one_line_per_mode_and_split_then_completion(tuned: Study) -> None:
    lines: list[str] = []

    finalize(tuned.run, tuned.evaluator, log=lines.append)

    assert len(lines) == 7
    progress = [json.loads(line) for line in lines[:6]]
    assert all(set(item) == {"final", "split", "primary"} for item in progress)
    assert [(item["final"], item["split"]) for item in progress] == [(m, s) for m in MODES for s in SPLITS]
    assert lines[6] == COMPLETE
    results = read_json(tuned.run / "results.json")["results"]
    assert all(item["primary"] == results[item["final"]][item["split"]]["primary"] for item in progress)


def test_finalize_twice_is_refused_and_leaves_the_outputs_untouched(study: Study) -> None:
    results_before = (study.run / "results.json").read_bytes()
    before = _snapshot(study.run)

    with pytest.raises(ValueError, match=f"^{re.escape('finalization already started; final outputs are immutable')}$"):
        finalize(study.run, study.evaluator)

    assert (study.run / "results.json").read_bytes() == results_before
    assert _snapshot(study.run) == before


def _extend_release_manifest(study: Study) -> None:
    rewrite_json(study.root / PROCESSED / "release_manifest.json", lambda m: {**m, "note": "still valid JSON"})


def _append_to_held_out_file(study: Study) -> None:
    _append(study.root / PROCESSED / "mode_A/test.jsonl", b'{"sample_id": "extra"}\n')


FINALIZE_CASES = [
    pytest.param(_extend_release_manifest, "release changed after model selection", id="release-manifest-changed"),
    pytest.param(_append_to_held_out_file, "mode_A/test.jsonl: release checksum mismatch", id="held-out-file-changed"),
    pytest.param(lambda s: _append(s.run / "plan.json"), PLAN_MISMATCH, id="plan-edited-the-gate-runs-first"),
]


@pytest.mark.parametrize(("mutate", "message"), FINALIZE_CASES)
def test_finalize_refuses_before_creating_any_final_output(tuned: Study, mutate: Mutation, message: str) -> None:
    mutate(tuned)
    before = _snapshot(tuned.run)

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        finalize(tuned.run, tuned.evaluator)

    assert not (tuned.run / "final").exists()
    assert _snapshot(tuned.run) == before
