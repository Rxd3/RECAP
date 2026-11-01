"""Independent workflow gates and inference/metric checks on tiny fixtures."""

import json
import sys
from copy import deepcopy

import pytest
import yaml
from import_closure import import_closure
from test_hmm_cli import ROOT, jsonl, setup_run
from test_hmm_workflow import run

from punct_restore.hashing import sha256_file
from punct_restore.models import hmm_experiment_audit as audit
from punct_restore.models import hmm_experiment_finalize, hmm_experiment_identity
from punct_restore.models.hmm_experiment_commands import HmmCommandFailed, execute
from punct_restore.models.hmm_experiment_finalize import finalize
from punct_restore.models.hmm_experiment_identity import SOURCE_FILES, check_evaluator, implementation
from punct_restore.models.hmm_experiment_protocol import choose, validate_config
from punct_restore.models.hmm_workflow import PredictionOptions, predict_run

HMM_COMMANDS = ("train_hmm", "predict_hmm", "run_hmm_experiments", "verify_hmm_experiments")


def config():
    return yaml.safe_load((ROOT / "configs/hmm_sponspeech_v2.yaml").read_text())


def test_source_files_list_each_hmm_command_as_module_then_stub():
    """The hashed code names the study's modules, the shared code they use, then each command's file and its stub."""
    models = ("hmm", "hmm_validation", "hmm_records", "hmm_workflow", "hmm_experiment_protocol",
              "hmm_experiment_commands", "hmm_experiment_runtime", "hmm_experiment_audit",
              "hmm_experiment_identity", "hmm_experiment_tune", "hmm_experiment_frozen",
              "hmm_experiment_finalize", "hmm_experiment_verify")
    expected = [f"src/punct_restore/models/{name}.py" for name in models]
    shared = ("data/labeling", "data/preprocessing", "data/validation", "evaluation/inputs", "evaluation/metrics",
              "hashing", "paths", "prediction_io", "progress")  # what the study's modules import, sorted
    expected += [f"src/punct_restore/{name}.py" for name in shared]
    for command in HMM_COMMANDS:
        expected += [f"src/punct_restore/cli/{command}.py", f"scripts/{command}.py"]
    assert list(SOURCE_FILES) == expected
    assert [name for name in expected if not (ROOT / name).is_file()] == []
    assert list(implementation()) == expected


def test_source_files_are_exactly_what_the_hmm_study_imports_inside_the_package():
    """A file the study imports (directly or not) decides its results, so a change to it must change the record."""
    package = ROOT / "src"
    roots = [package / "punct_restore/cli" / f"{command}.py" for command in HMM_COMMANDS]
    roots += sorted((package / "punct_restore/models").glob("hmm*.py"))
    imported = {path.relative_to(ROOT).as_posix() for path in import_closure(roots, package, "punct_restore")}
    listed = {name for name in SOURCE_FILES if name.startswith("src/")}
    assert sorted(imported - listed) == [], "imported by the study but not hashed"
    assert sorted(listed - imported) == [], "hashed but not imported by the study"


def test_the_module_that_computes_the_implementation_record_is_itself_hashed():
    assert "src/punct_restore/models/hmm_experiment_identity.py" in SOURCE_FILES


def test_check_evaluator_still_hashes_scripts_evaluate_models_in_the_pinned_checkout(
    tmp_path, monkeypatch
):
    """The pinned evaluator commit keeps its scorer at scripts/evaluate_models.py, so that path stays."""
    names = [
        "src/punct_restore/evaluation/report.py",
        "scripts/evaluate_models.py",
        "configs/label_sets.yaml",
    ]
    for name in names:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(name, encoding="utf-8")
    pinned = {"commit": "pinned", "dirty": False}
    monkeypatch.setattr(hmm_experiment_identity, "code_identity", lambda _root: pinned)
    files = check_evaluator(tmp_path, "pinned")["files"]
    assert sorted(files) == sorted(names)
    scorer = tmp_path / "scripts/evaluate_models.py"
    assert files["scripts/evaluate_models.py"] == sha256_file(scorer)


def test_check_evaluator_rejects_a_dirty_or_unpinned_checkout(tmp_path, monkeypatch):
    """The patched identity must be the one check_evaluator reads: a different commit is refused."""
    monkeypatch.setattr(hmm_experiment_identity, "code_identity", lambda _root: {"commit": "other", "dirty": False})
    with pytest.raises(ValueError, match="clean at the pinned"):
        check_evaluator(tmp_path, "pinned")


def test_selection_uses_only_validation_score_and_predeclared_tie_break():
    records = [
        {"validation_primary": 0.2, "min_token_count": 1, "smoothing": 0.01, "test_score": 1.0},
        {"validation_primary": 0.3, "min_token_count": 5, "smoothing": 0.01, "test_score": 1.0},
        {"validation_primary": 0.3, "min_token_count": 1, "smoothing": 0.1, "test_score": 0.0},
        {"validation_primary": 0.3, "min_token_count": 1, "smoothing": 0.01, "test_score": 0.0},
    ]
    assert choose(records) is records[3]
    for record in records:
        record["test_score"] = 1 - record["test_score"]
    assert choose(records) is records[3]


@pytest.mark.parametrize(
    "bad",
    [
        "test-selection",
        "retrain-validation",
        "C-wrong-metric",
        "negative-alpha",
        "fractional-cutoff",
        "repeat-grid",
        "missing-test-amb",
        "wrong-tie",
    ],
)
def test_experiment_rejects_invalid_or_test_driven_protocols(bad):
    c = config()
    if bad == "test-selection":
        c["search"]["selection_split"] = "test"
    elif bad == "retrain-validation":
        c["search"]["retrain_on_validation"] = True
    elif bad == "C-wrong-metric":
        c["search"]["selection_blocks"]["C"] = "punctuation/all"
    elif bad == "negative-alpha":
        c["search"]["smoothing"] = [-0.1]
    elif bad == "fractional-cutoff":
        c["search"]["min_token_count"] = [1.1]
    elif bad == "repeat-grid":
        c["search"]["smoothing"] = [0.1, 0.1]
    elif bad == "missing-test-amb":
        c["evaluation"]["test_sets"] = ["test"]
    else:
        c["search"]["tie_break"] = ["test_score"]
    with pytest.raises(ValueError):
        validate_config(c)


def test_finalize_cannot_touch_test_data_before_global_selection_exists(tmp_path, monkeypatch):
    (tmp_path / "plan.json").write_text(json.dumps({"config": config()}))
    called = []
    monkeypatch.setattr(hmm_experiment_finalize, "authenticate", lambda *args: called.append(args))
    with pytest.raises(FileNotFoundError):
        finalize(tmp_path, tmp_path / "evaluator")
    assert not called and not (tmp_path / "final").exists()


def test_a_failing_command_raises_hmm_command_failed_naming_only_its_log(tmp_path):
    log = tmp_path / "deep" / "c1.training.log"
    failing = [sys.executable, "-c", "import sys; print('boom', file=sys.stderr); sys.exit(3)"]
    with pytest.raises(HmmCommandFailed) as caught:
        execute(failing, tmp_path, log)
    assert str(caught.value) == "command failed; see c1.training.log"
    assert isinstance(caught.value, RuntimeError)
    assert "boom" in log.read_text(encoding="utf-8")  # the detail the message leaves out is in the log
    assert json.loads((tmp_path / "commands.jsonl").read_text(encoding="utf-8"))["returncode"] == 3


def test_a_succeeding_command_is_logged_and_returned(tmp_path):
    record = execute([sys.executable, "-c", "print('fine')"], tmp_path, tmp_path / "ok.log")
    assert record["returncode"] == 0 and record["log"] == "ok.log"
    assert (tmp_path / "ok.log").read_text(encoding="utf-8").strip() == "fine"


def test_independent_f1_counts_gold_O_errors_and_absent_classes():
    result = audit.independent_scores(
        [("O", "COMMA"), ("COMMA", "O"), ("PERIOD", "PERIOD")],
        ["O", "COMMA", "PERIOD", "QUESTION"],
        ["COMMA", "PERIOD", "QUESTION"],
    )
    assert result["confusion"] == [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
    assert result["per_class"]["COMMA"]["predicted"] == 1
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["micro"]["f1"] == pytest.approx(0.5)
    assert result["per_class"]["QUESTION"]["f1"] == 0


@pytest.mark.parametrize("mode", ["A", "B", "C"])
def test_saved_model_inference_is_reproducible_and_does_not_read_gold(tmp_path, mode):
    args, _, prediction, trained = setup_run(tmp_path, mode)
    assert run(args).returncode == 0
    rows = [
        {"schema_version": 1, "sample_id": "p1", "tokens": ["hello", "unseen"]},
        {"schema_version": 1, "sample_id": "p2", "tokens": ["nasa"]},
    ]
    jsonl(prediction, rows)
    first = tmp_path / "pred1"
    before = predict_run(trained / "model.json", prediction, first)
    poisoned = [
        {
            **row,
            "punct_labels": ["INVALID"],
            "case_labels": [],
            "metadata": {"original_text": "A GOLD LEAK"},
        }
        for row in rows
    ]
    jsonl(prediction, poisoned)
    second = tmp_path / "pred2"
    after = predict_run(trained / "model.json", prediction, second)
    assert before["predictions_sha256"] == after["predictions_sha256"]
    assert before["num_sequences"] == 2 and before["num_tokens"] == 3
    assert before["inputs"]["input"]["sha256"] != after["inputs"]["input"]["sha256"]
    with pytest.raises(ValueError, match="already exists"):
        predict_run(trained / "model.json", prediction, first)


def test_artifact_cannot_be_substituted_for_another_frozen_winner(tmp_path):
    args, _, prediction, trained = setup_run(tmp_path)
    assert run(args).returncode == 0
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "used_test_labels": False,
                "training_includes_validation": False,
                "modes": {"A": {"model_sha256": "incorrect"}},
            }
        )
    )
    out = tmp_path / "prediction-output"
    with pytest.raises(ValueError, match="frozen selection"):
        predict_run(trained / "model.json", prediction, out, PredictionOptions(selection_record=selection))
    assert not out.exists()


def test_metric_checker_rejects_a_green_looking_incorrect_report():
    expected = audit.independent_scores([("O", "COMMA")], ["O", "COMMA"], ["COMMA"])
    wrong = deepcopy(expected)
    wrong["macro_f1"] = 1.0
    with pytest.raises(ValueError, match="metric mismatch"):
        audit.assert_block(wrong, expected)
