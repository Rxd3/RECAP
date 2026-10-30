"""Independent workflow gates and inference/metric checks on tiny fixtures."""

import json
import sys
from copy import deepcopy

import pytest
import yaml

from punct_restore.models.hmm_workflow import predict_run
from test_hmm_cli import ROOT, jsonl, setup_run
from test_hmm_workflow import run

sys.path.insert(0, str(ROOT / "scripts"))
import run_hmm_experiments as workflow  # noqa: E402
import verify_hmm_experiments as audit  # noqa: E402


def config():
    return yaml.safe_load((ROOT / "configs/hmm_sponspeech_v2.yaml").read_text())


def test_selection_uses_only_validation_score_and_predeclared_tie_break():
    records = [
        {"validation_primary": 0.2, "min_token_count": 1, "smoothing": 0.01, "test_score": 1.0},
        {"validation_primary": 0.3, "min_token_count": 5, "smoothing": 0.01, "test_score": 1.0},
        {"validation_primary": 0.3, "min_token_count": 1, "smoothing": 0.1, "test_score": 0.0},
        {"validation_primary": 0.3, "min_token_count": 1, "smoothing": 0.01, "test_score": 0.0},
    ]
    assert workflow.choose(records) is records[3]
    for record in records:
        record["test_score"] = 1 - record["test_score"]
    assert workflow.choose(records) is records[3]


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
        workflow.validate_config(c)


def test_finalize_cannot_touch_test_data_before_global_selection_exists(tmp_path, monkeypatch):
    (tmp_path / "plan.json").write_text(json.dumps({"config": config()}))
    called = []
    monkeypatch.setattr(workflow, "authenticate", lambda *args: called.append(args))
    with pytest.raises(FileNotFoundError):
        workflow.finalize(tmp_path, tmp_path / "evaluator")
    assert not called and not (tmp_path / "final").exists()


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
        predict_run(trained / "model.json", prediction, out, selection_record=selection)
    assert not out.exists()


def test_metric_checker_rejects_a_green_looking_incorrect_report():
    expected = audit.independent_scores([("O", "COMMA")], ["O", "COMMA"], ["COMMA"])
    wrong = deepcopy(expected)
    wrong["macro_f1"] = 1.0
    with pytest.raises(ValueError, match="metric mismatch"):
        audit.assert_block(wrong, expected)
