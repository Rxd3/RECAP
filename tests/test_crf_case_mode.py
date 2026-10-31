"""CRFModel label mode "CASE": a casing-only helper mode trained from mode C records' case_labels."""

from __future__ import annotations

import copy
from pathlib import Path

import joblib
import pytest

from punct_restore.models.crf import CRFModel, get_mode_vocabulary

CASING = ["LOWER", "FIRST_UPPER", "ALL_UPPER"]


@pytest.fixture
def mode_c_records() -> list[dict]:
    return [
        {"sample_id": "c1", "label_mode": "C", "tokens": ["today", "we", "discuss", "nlp"],
         "punct_labels": ["O", "O", "O", "PERIOD"], "case_labels": ["FIRST_UPPER", "LOWER", "LOWER", "ALL_UPPER"]},
        {"sample_id": "c2", "label_mode": "C", "tokens": ["did", "you", "agree"],
         "punct_labels": ["O", "O", "QUESTION"], "case_labels": ["FIRST_UPPER", "LOWER", "LOWER"]},
        {"sample_id": "c3", "label_mode": "C", "tokens": ["yes", "indeed"],
         "punct_labels": ["COMMA", "PERIOD"], "case_labels": ["FIRST_UPPER", "LOWER"]},
    ]


def test_case_vocabulary_is_the_casing_labels_in_config_order():
    assert get_mode_vocabulary("CASE") == CASING
    assert CRFModel(label_mode="CASE").vocabulary == CASING


@pytest.mark.parametrize("bad_mode", ["case", "D", "CASING"])
def test_other_mode_names_are_still_rejected(bad_mode):
    with pytest.raises(ValueError):
        get_mode_vocabulary(bad_mode)
    with pytest.raises(ValueError):
        CRFModel(label_mode=bad_mode)


def test_fit_label_mode_override_rejects_other_mode_names(mode_c_records):
    model = CRFModel(label_mode="A", max_iterations=5)
    with pytest.raises(ValueError, match="Invalid label_mode: 'case'"):
        model.fit(mode_c_records, label_mode="case")
    assert model.label_mode == "A"


def test_fit_trains_on_case_labels_and_predicts_casing_labels(mode_c_records):
    model = CRFModel(label_mode="CASE", max_iterations=20, model_name="crf_case_x")
    assert model.fit(mode_c_records) is model
    assert model._training_metadata["label_mode"] == "CASE"
    assert set(model._training_metadata["observed_labels"]) == set(CASING)  # the fixture uses all three
    preds = model.predict([{"sample_id": "p1", "tokens": ["today", "we", "agree", "now"]}])
    assert len(preds) == 1
    assert preds[0]["sample_id"] == "p1" and preds[0]["model_name"] == "crf_case_x"
    assert preds[0]["label_mode"] == "CASE"
    assert len(preds[0]["predicted_labels"]) == 4
    assert set(preds[0]["predicted_labels"]) <= set(CASING)


def test_fit_label_mode_override_switches_to_case(mode_c_records):
    model = CRFModel(label_mode="A", max_iterations=5)
    model.fit(mode_c_records, label_mode="CASE")
    assert model.label_mode == "CASE" and model.vocabulary == CASING


def test_fit_does_not_check_the_record_label_mode_field(mode_c_records):
    mislabelled = [{**r, "label_mode": "A"} for r in mode_c_records]
    CRFModel(label_mode="CASE", max_iterations=5).fit(mislabelled)


def test_fit_does_not_mutate_the_records(mode_c_records):
    before = copy.deepcopy(mode_c_records)
    CRFModel(label_mode="CASE", max_iterations=5).fit(mode_c_records)
    assert mode_c_records == before


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        pytest.param(lambda r: r.pop("case_labels"), "case_labels length", id="missing"),
        pytest.param(lambda r: r.update(case_labels="LOWER"), "case_labels length", id="not-a-list"),
        pytest.param(lambda r: r.update(case_labels=["LOWER"]), "case_labels length", id="short"),
        pytest.param(lambda r: r.update(case_labels=["LOWER"] * 9), "case_labels length", id="long"),
        pytest.param(lambda r: r.update(case_labels=["LOWER", "MIXED", "LOWER", "LOWER"]), "unknown case_labels",
                     id="unknown"),
        pytest.param(lambda r: r.update(case_labels=["LOWER", 1, "LOWER", "LOWER"]), "unknown case_labels",
                     id="non-string"),
    ],
)
def test_bad_case_labels_are_a_value_error(mode_c_records, edit, message):
    edit(mode_c_records[0])
    with pytest.raises(ValueError, match=message):
        CRFModel(label_mode="CASE", max_iterations=5).fit(mode_c_records)


def test_save_load_round_trip_keeps_the_mode_and_the_predictions(mode_c_records, tmp_path: Path):
    model = CRFModel(label_mode="CASE", max_iterations=20, model_name="crf_case_x").fit(mode_c_records)
    path = tmp_path / "case.joblib"
    model.save_model(path)
    loaded = CRFModel.load_model(path)
    assert loaded.label_mode == "CASE" and loaded.vocabulary == CASING and loaded.model_name == "crf_case_x"
    probe = [{"sample_id": "p1", "tokens": ["did", "you", "agree"]}]
    assert loaded.predict(probe) == model.predict(probe)


@pytest.mark.parametrize("bad_mode", ["case", "D", None])
def test_load_rejects_an_artifact_with_an_unknown_label_mode(mode_c_records, tmp_path: Path, bad_mode):
    path = tmp_path / "case.joblib"
    CRFModel(label_mode="CASE", max_iterations=5).fit(mode_c_records).save_model(path)
    bundle = joblib.load(path)
    joblib.dump({**bundle, "label_mode": bad_mode}, path)
    with pytest.raises(ValueError, match="Invalid label_mode in artifact"):
        CRFModel.load_model(path)


def test_modes_a_and_c_still_train_on_their_own_targets(mode_c_records):
    punct = CRFModel(label_mode="A", max_iterations=5).fit(mode_c_records)
    assert set(punct._training_metadata["observed_labels"]) <= {"O", "COMMA", "PERIOD", "QUESTION"}
    joint = CRFModel(label_mode="C", max_iterations=5).fit(mode_c_records)
    assert all("|" in label for label in joint._training_metadata["observed_labels"])
    assert joint.predict([{"sample_id": "p", "tokens": ["yes"]}])[0]["label_mode"] == "C"
