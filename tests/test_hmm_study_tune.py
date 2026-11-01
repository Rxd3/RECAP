"""The HMM study's validation search: tune(), its refusals, the evaluator/code identity and the command helpers.

The study runs for real on a tiny fake repository (see hmm_study_fixtures); the happy-path tests share one
tuned run and only read it, every other test builds its own.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from hmm_study_fixtures import (
    CLEAN_CHECKOUT,
    CUTOFFS,
    MODES,
    POLICY,
    PROCESSED,
    RELEASE_ID,
    SMOOTHING,
    SPLITS_DIR,
    Study,
    copy_study,
    git,
    make_repo,
    patch_study,
    read_json,
    sha256,
    write_text,
)

from punct_restore.models import hmm_experiment_tune
from punct_restore.models.hmm_experiment_commands import (
    HmmCommandFailed,
    baseline_command,
    execute,
    predict_command,
    score,
    train_command,
)
from punct_restore.models.hmm_experiment_commands import Study as CommandStudy
from punct_restore.models.hmm_experiment_identity import SOURCE_FILES, check_evaluator, implementation
from punct_restore.models.hmm_experiment_tune import tune

REAL_ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ["s1-c1", "s0.1-c1", "s1-c2", "s0.1-c2"]  # cutoff is the outer loop, smoothing the inner one
EVALUATOR_FILES = ["src/punct_restore/evaluation/metrics.py", "scripts/evaluate_models.py", "configs/label_sets.yaml"]
DIRTY = "commit reviewed code/config before starting an official search"
RUN_EXISTS = "run directory exists; choose a new experiment ID"


@dataclass(frozen=True)
class Tuned:
    """One finished tune() call: the study it ran on and the progress lines it logged."""

    study: Study
    lines: list[str]


@pytest.fixture(scope="module")
def tuned(tmp_path_factory: pytest.TempPathFactory) -> Tuned:
    """One tuned run for the happy-path tests; they only read it."""
    study = make_repo(tmp_path_factory.mktemp("tuned"))
    lines: list[str] = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        patch_study(monkeypatch, study)
        tune(study.config_path, study.evaluator, study.run, log=lines.append)
    return Tuned(study, lines)


@pytest.fixture
def fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Study:
    """A fake repository with no run yet, with the study code pointed at it."""
    study = make_repo(tmp_path)
    patch_study(monkeypatch, study)
    return study


@pytest.fixture
def study(tuned: Tuned, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Study:
    """A private copy of the tuned study (scoring needs only its candidates' predictions)."""
    copy = copy_study(tuned.study, tmp_path)
    patch_study(monkeypatch, copy)
    return copy


def _config(study: Study) -> dict[str, Any]:
    return yaml.safe_load(study.config_path.read_text(encoding="utf-8"))


def _edit_config(study: Study, change: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
    write_text(study.config_path, yaml.safe_dump(change(_config(study)), sort_keys=False))


def _selection(tuned: Tuned, mode: str) -> dict[str, Any]:
    return read_json(tuned.study.run / f"mode_{mode}/selection.json")


def _refuse_matching(study: Study, pattern: str, evaluator: Path | None = None) -> None:
    """tune() must raise a ValueError matching `pattern` and leave no run (nor anything else) behind."""
    with pytest.raises(ValueError, match=pattern):
        tune(study.config_path, evaluator or study.evaluator, study.run)
    assert not study.run.exists()
    assert list((study.root / "artifacts").rglob("*")) == []


def _evaluator_refused(study: Study, evaluator: Path | None = None) -> str:
    """The refusal of an evaluator folder: its name and the first 12 characters of the commit the config pins."""
    pinned = yaml.safe_load(study.config_path.read_text(encoding="utf-8"))["evaluation"]["repository_commit"]
    return f"{(evaluator or study.evaluator).name}: evaluator checkout must be clean at the pinned commit {pinned[:12]}"


def _refuse(study: Study, message: str, evaluator: Path | None = None) -> None:
    _refuse_matching(study, f"^{re.escape(message)}$", evaluator)


# ---- A. the happy path ----


def test_each_mode_searches_the_grid_with_cutoff_outer_and_smoothing_inner(tuned: Tuned) -> None:
    for mode in MODES:
        candidates = _selection(tuned, mode)["candidates"]
        assert [c["candidate"] for c in candidates] == CANDIDATES
        assert [(c["min_token_count"], c["smoothing"]) for c in candidates] == [
            (cutoff, smoothing) for cutoff in CUTOFFS for smoothing in SMOOTHING
        ]


def test_each_candidate_records_the_hashes_and_score_of_the_files_on_disk(tuned: Tuned) -> None:
    run = tuned.study.run
    for mode in MODES:
        for candidate in _selection(tuned, mode)["candidates"]:
            folder = run / f"mode_{mode}/candidates/{candidate['candidate']}"
            assert candidate["model"] == f"mode_{mode}/candidates/{candidate['candidate']}/model.json"
            assert candidate["model_sha256"] == sha256(folder / "model.json")
            assert candidate["validation_report_sha256"] == sha256(folder / "validation.json")
            assert candidate["validation_primary"] == read_json(folder / "validation.json")["primary"]["value"]


def test_selection_freezes_every_mode_on_the_best_validation_candidate(tuned: Tuned) -> None:
    run = tuned.study.run
    frozen = read_json(run / "selection.json")
    assert frozen["schema"] == "recap-hmm-selection-v1"
    assert frozen["plan_sha256"] == sha256(run / "plan.json")
    assert frozen["used_test_labels"] is False
    assert frozen["training_includes_validation"] is False
    assert list(frozen["modes"]) == list(MODES)
    assert list(frozen["mode_selection_sha256"]) == list(MODES)
    for mode in MODES:
        candidates = _selection(tuned, mode)["candidates"]
        best = sorted(candidates, key=lambda c: (-c["validation_primary"], c["min_token_count"], c["smoothing"]))[0]
        assert frozen["modes"][mode] == best
        assert frozen["mode_selection_sha256"][mode] == sha256(run / f"mode_{mode}/selection.json")


def test_mode_selection_files_say_how_the_winner_was_picked(tuned: Tuned) -> None:
    blocks = {"A": "punctuation/all", "C": "joint/unmasked"}
    frozen = read_json(tuned.study.run / "selection.json")
    for mode in MODES:
        record = _selection(tuned, mode)
        assert record["mode"] == mode
        assert record["selection_block"] == blocks[mode]
        assert record["train_only"] is True
        assert record["used_test_labels"] is False
        assert record["selected"] == frozen["modes"][mode]


def test_plan_records_the_code_evaluator_and_config_before_training(tuned: Tuned) -> None:
    study = tuned.study
    plan = read_json(study.run / "plan.json")
    commit = git(study.evaluator, "rev-parse", "HEAD")
    assert plan["schema"] == "recap-hmm-experiment-v1"
    assert plan["code"] == CLEAN_CHECKOUT
    assert plan["implementation"] == implementation()
    assert plan["evaluator"] == check_evaluator(study.evaluator, commit)
    assert plan["config"] == _config(study)
    assert plan["configuration"] == {"path": "configs/hmm_unit.yaml", "sha256": sha256(study.config_path)}
    assert plan["test_policy"].startswith("No test JSONL is opened in tuning")
    assert set(plan) == {
        "schema", "created_at_utc", "config", "configuration", "code", "implementation", "evaluator",
        "tuning_inputs", "hardware", "environment", "test_policy",
    }  # fmt: skip


def test_plan_lists_exactly_the_tuning_inputs_with_their_hashes(tuned: Tuned) -> None:
    study = tuned.study
    expected = {"release_manifest"}
    for mode in MODES:
        expected |= {
            f"mode_{mode}/train.jsonl", f"default_train/mode_{mode}.ids",
            f"mode_{mode}/validation.jsonl", f"primary_eval/validation_mode_{mode}.ids",
        }  # fmt: skip
    expected.add("masks/case_joint/validation.jsonl")
    inputs = read_json(study.run / "plan.json")["tuning_inputs"]
    assert set(inputs) == expected
    assert not [key for key in inputs if "test" in key]
    manifest = study.root / PROCESSED / "release_manifest.json"
    assert inputs["release_manifest"] == {"path": f"{PROCESSED}/release_manifest.json", "sha256": sha256(manifest)}
    for key, digest in inputs.items():
        if key != "release_manifest":
            folder = SPLITS_DIR if key.endswith(".ids") else PROCESSED
            assert digest == sha256(study.root / folder / key)


def test_tuning_never_touches_test_data_and_makes_no_final_run(tuned: Tuned) -> None:
    run = tuned.study.run
    records = [json.loads(line) for line in (run / "commands.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 16
    assert [r["command"][1] for r in records].count("scripts/train_hmm.py") == 8
    assert [r["command"][1] for r in records].count("tmp/evaluator/scripts/evaluate_models.py") == 8
    for record in records:
        assert record["command"][0] == "python"
        assert record["returncode"] == 0
        assert not [arg for arg in record["command"] if "test" in arg]  # no test.jsonl, test_amb or test ID list
    assert not (run / "final").exists()
    assert not [path for path in run.rglob("*") if "test" in path.name]
    top = {"plan.json", "selection.json", "commands.jsonl", "mode_A", "mode_C"}
    assert {path.name for path in run.iterdir()} == top


def test_tune_logs_one_json_line_per_candidate_then_the_freeze_message(tuned: Tuned) -> None:
    *progress, last = tuned.lines
    assert last == "All mode selections frozen: artifacts/hmm/unit/selection.json"
    parsed = [json.loads(line) for line in progress]
    assert [(p["mode"], p["candidate"]) for p in parsed] == [(m, c) for m in MODES for c in CANDIDATES]
    for mode in MODES:
        by_name = {c["candidate"]: c for c in _selection(tuned, mode)["candidates"]}
        for line in (p for p in parsed if p["mode"] == mode):
            assert {k: v for k, v in line.items() if k != "mode"} == by_name[line["candidate"]]


# ---- B. tune refusals ----


def test_tune_refuses_an_uncommitted_checkout(fresh: Study, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hmm_experiment_tune, "code_identity", lambda: {"commit": "x", "dirty": True})
    _refuse(fresh, DIRTY)


def test_tune_refuses_a_config_that_is_not_utf8(fresh: Study) -> None:
    fresh.config_path.write_bytes(b"\xff\xfe")
    _refuse(fresh, "hmm_unit.yaml: not valid UTF-8")


def test_tune_refuses_a_config_that_is_not_yaml(fresh: Study) -> None:
    write_text(fresh.config_path, "a: [1, 2")
    _refuse(fresh, "hmm_unit.yaml: not valid YAML")


def test_tune_refuses_a_config_without_a_search_section(fresh: Study) -> None:
    _edit_config(fresh, lambda c: {k: v for k, v in c.items() if k != "search"})
    _refuse(fresh, "hmm_unit.yaml: invalid experiment config: missing search")


def test_tune_refuses_a_config_that_selects_on_test(fresh: Study) -> None:
    _edit_config(fresh, lambda c: {**c, "search": {**c["search"], "selection_split": "test"}})
    _refuse(
        fresh,
        "hmm_unit.yaml: invalid experiment config: settings must be selected on validation; training stays train-only",
    )


def test_tune_refuses_an_evaluator_pinned_to_another_commit(fresh: Study) -> None:
    _edit_config(fresh, lambda c: {**c, "evaluation": {**c["evaluation"], "repository_commit": "f" * 40}})
    _refuse(fresh, _evaluator_refused(fresh))


def test_tune_refuses_an_evaluator_checkout_with_uncommitted_files(fresh: Study) -> None:
    write_text(fresh.evaluator / "untracked.txt", "not committed\n")
    _refuse(fresh, _evaluator_refused(fresh))


def test_tune_refuses_an_evaluator_folder_that_is_not_a_git_checkout(
    fresh: Study, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = tmp_path / "not-a-checkout"
    plain.mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))  # never find a repository above the test folder
    _refuse(fresh, _evaluator_refused(fresh, plain), evaluator=plain)


def test_tune_refuses_a_release_with_another_id(fresh: Study) -> None:
    _edit_config(fresh, lambda c: {**c, "dataset_release_id": "another-release"})
    _refuse(fresh, "experiment and release versions differ")


def test_tune_refuses_a_label_configuration_that_differs_from_the_release(fresh: Study) -> None:
    original = (fresh.root / "configs/label_sets.yaml").read_text(encoding="utf-8")
    write_text(fresh.root / "configs/label_sets_copy.yaml", original + "# one more comment line\n")
    _edit_config(fresh, lambda c: {**c, "label_config": "configs/label_sets_copy.yaml"})
    _refuse(fresh, "label configuration mismatch")


def test_tune_refuses_a_release_file_that_was_edited(fresh: Study) -> None:
    edited = fresh.root / PROCESSED / "mode_A/train.jsonl"
    with edited.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write('{"extra": "line"}\n')
    _refuse(fresh, "mode_A/train.jsonl: release checksum mismatch")


def test_tune_refuses_an_existing_run_folder_and_leaves_it_untouched(fresh: Study) -> None:
    marker = fresh.run / "marker.txt"
    write_text(marker, "mine\n")
    with pytest.raises(ValueError, match=f"^{re.escape(RUN_EXISTS)}$"):
        tune(fresh.config_path, fresh.evaluator, fresh.run)
    assert [path.name for path in fresh.run.iterdir()] == ["marker.txt"]
    assert marker.read_text(encoding="utf-8") == "mine\n"
    assert list(fresh.run.parent.iterdir()) == [fresh.run]


# ---- C. identity ----


def test_source_files_are_unique_and_exist_in_the_repository() -> None:
    assert len(set(SOURCE_FILES)) == len(SOURCE_FILES)
    assert [name for name in SOURCE_FILES if not (REAL_ROOT / name).is_file()] == []


def test_implementation_hashes_every_source_file_in_order() -> None:
    result = implementation()
    assert list(result) == list(SOURCE_FILES)
    for name, digest in result.items():
        assert digest == hashlib.sha256((REAL_ROOT / name).read_bytes()).hexdigest()


def test_check_evaluator_returns_the_commit_and_the_hash_of_each_evaluator_file(fresh: Study) -> None:
    commit = git(fresh.evaluator, "rev-parse", "HEAD")
    identity = check_evaluator(fresh.evaluator, commit)
    assert identity == {"commit": commit, "files": {name: sha256(fresh.evaluator / name) for name in EVALUATOR_FILES}}
    assert list(identity["files"]) == EVALUATOR_FILES


def test_check_evaluator_hashes_every_evaluation_module_sorted_before_the_script_and_config(fresh: Study) -> None:
    for name in ("metrics.py", "inputs.py", "__init__.py", "notes.txt"):
        write_text(fresh.evaluator / "src/punct_restore/evaluation" / name, f"# {name}\n")
    git(fresh.evaluator, "add", ".")
    git(fresh.evaluator, "commit", "-q", "-m", "more evaluator files")
    identity = check_evaluator(fresh.evaluator, git(fresh.evaluator, "rev-parse", "HEAD"))
    assert list(identity["files"]) == [
        "src/punct_restore/evaluation/__init__.py",
        "src/punct_restore/evaluation/inputs.py",
        "src/punct_restore/evaluation/metrics.py",
        "scripts/evaluate_models.py",
        "configs/label_sets.yaml",
    ]


# ---- D. commands ----


def test_execute_logs_records_and_then_raises_for_a_failing_command(fresh: Study, tmp_path: Path) -> None:
    run, code = tmp_path / "exec-run", "import sys; print('out'); sys.exit(3)"
    with pytest.raises(HmmCommandFailed) as caught:
        execute([sys.executable, "-c", code], run, run / "logs/x.log")
    assert str(caught.value) == "command failed; see x.log"
    assert caught.value.log == run / "logs/x.log"
    assert (run / "logs/x.log").read_text(encoding="utf-8") == "out" + chr(10)
    records = [json.loads(line) for line in (run / "commands.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["command"] == ["python", "-c", code]
    assert records[0]["returncode"] == 3
    assert records[0]["log"] == "logs/x.log"


def test_execute_returns_the_record_and_appends_it_to_the_command_log(fresh: Study, tmp_path: Path) -> None:
    run, code = tmp_path / "exec-run", "import sys; print(sys.argv[1]); print('err', file=sys.stderr)"
    first = execute([sys.executable, "-c", code, 7], run, run / "logs/first.log")
    second = execute([sys.executable, "-c", code, "8"], run, run / "logs/second.log")
    assert first["command"] == ["python", "-c", code, "7"]  # every argument is passed and recorded as text
    assert (first["returncode"], first["log"]) == (0, "logs/first.log")
    assert first["seconds"] >= 0
    assert datetime.fromisoformat(first["at_utc"]).utcoffset() is not None
    assert (run / "logs/first.log").read_text(encoding="utf-8") == "7\nerr\n"  # stdout, then stderr
    lines = (run / "commands.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [first, second]


def test_train_command_is_the_exact_train_hmm_line(fresh: Study) -> None:
    study = CommandStudy(_config(fresh), fresh.evaluator, fresh.run)
    assert train_command(study, "A", "cand", 0.1, 1, fresh.run / "cand") == [
        sys.executable, "scripts/train_hmm.py",
        "--train", "data/processed/unit/mode_A/train.jsonl",
        "--train-ids", "data/splits/unit/default_train/mode_A.ids",
        "--output-dir", "artifacts/hmm/unit/cand",
        "--label-mode", "A",
        "--run-id", "unit-hmm-study-A-cand",
        "--label-config", "configs/label_sets.yaml",
        "--smoothing", "0.1",
        "--min-token-count", "1",
        "--dataset-release-id", RELEASE_ID,
        "--preprocessing-version", POLICY,
        "--tokenizer-version", POLICY,
        "--label-policy-version", POLICY,
        "--split-manifest", "data/splits/unit/SHA256SUMS",
        "--release-manifest", "data/processed/unit/release_manifest.json",
        "--predict-input", "data/processed/unit/mode_A/validation.jsonl",
    ]  # fmt: skip


def test_predict_command_is_the_exact_predict_hmm_line(fresh: Study) -> None:
    study = CommandStudy(_config(fresh), fresh.evaluator, fresh.run)
    winner = {"model": "mode_C/candidates/s1-c1/model.json"}
    assert predict_command(study, "C", "test", winner, fresh.run / "final/mode_C/test") == [
        sys.executable, "scripts/predict_hmm.py",
        "--model", "artifacts/hmm/unit/mode_C/candidates/s1-c1/model.json",
        "--input", "data/processed/unit/mode_C/test.jsonl",
        "--output-dir", "artifacts/hmm/unit/final/mode_C/test",
        "--all-records",
        "--selection-record", "artifacts/hmm/unit/selection.json",
        "--release-manifest", "data/processed/unit/release_manifest.json",
    ]  # fmt: skip


def test_baseline_command_is_the_exact_pinned_evaluator_line(fresh: Study) -> None:
    study = CommandStudy(_config(fresh), fresh.evaluator, fresh.run)
    assert baseline_command(study, "C", "test_amb", fresh.run / "final/mode_C/baseline.json") == [
        sys.executable, "tmp/evaluator/scripts/evaluate_models.py", "baseline",
        "--gold", "data/processed/unit/mode_C/test_amb.jsonl",
        "--ids", "data/splits/unit/primary_eval/test_amb_mode_C.ids",
        "--out", "artifacts/hmm/unit/final/mode_C/baseline.json",
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("mode", "block", "masks"),
    [
        ("A", "punctuation/all", []),
        ("C", "joint/unmasked", ["--masks", "data/processed/unit/masks/case_joint/validation.jsonl"]),
    ],
)
def test_score_runs_the_pinned_scorer_and_returns_its_report(
    study: Study, mode: str, block: str, masks: list[str]
) -> None:
    predictions = study.run / f"mode_{mode}/candidates/s0.1-c1/predictions.jsonl"
    out = study.run / "rescore/validation.json"
    out.parent.mkdir()  # the shared scorer writes its report into an existing folder
    report = score(CommandStudy(_config(study), study.evaluator, study.run), mode, "validation", predictions, out)
    assert report == read_json(out)
    assert report["primary"]["block"] == block
    last = json.loads((study.run / "commands.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert last["command"] == [
        "python", "tmp/evaluator/scripts/evaluate_models.py", "score",
        "--gold", f"data/processed/unit/mode_{mode}/validation.jsonl",
        "--ids", f"data/splits/unit/primary_eval/validation_mode_{mode}.ids",
        "--predictions", f"artifacts/hmm/unit/mode_{mode}/candidates/s0.1-c1/predictions.jsonl",
        "--out", "artifacts/hmm/unit/rescore/validation.json",
        *masks,
    ]  # fmt: skip
    assert last["log"] == "rescore/validation.log"


def test_score_refuses_a_report_whose_primary_block_is_not_the_configured_one(study: Study) -> None:
    config = _config(study)
    blocks = {**config["search"]["selection_blocks"], "A": "punctuation/non_final"}
    wrong = {**config, "search": {**config["search"], "selection_blocks": blocks}}
    commands = study.run / "commands.jsonl"
    before = commands.read_text(encoding="utf-8").splitlines()
    out = study.run / "rescore/validation.json"
    out.parent.mkdir()  # the shared scorer writes its report into an existing folder
    predictions = study.run / "mode_A/candidates/s0.1-c1/predictions.jsonl"
    with pytest.raises(ValueError, match="^shared scorer returned an unexpected primary block$"):
        score(CommandStudy(wrong, study.evaluator, study.run), "A", "validation", predictions, out)
    after = commands.read_text(encoding="utf-8").splitlines()
    assert after[:-1] == before  # the scorer ran once, and its run is on record
    assert json.loads(after[-1])["returncode"] == 0
    assert read_json(out)["primary"]["block"] == "punctuation/all"  # the scorer's own report is left as it wrote it
