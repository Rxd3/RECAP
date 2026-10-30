"""punct_restore.models.factored.combine: mode A punctuation + CASE casing predictions -> mode C predictions."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from punct_restore.data.labeling import load_label_sets
from punct_restore.models import factored
from punct_restore.models.factored import combine


def _pred(sample_id: str, mode: str, labels: list[str], model: str = "m") -> dict[str, Any]:
    return {"sample_id": sample_id, "model_name": model, "label_mode": mode, "predicted_labels": labels}


@pytest.fixture
def punct() -> list[dict[str, Any]]:
    return [
        _pred("b", "A", ["COMMA", "O", "QUESTION"], "crf_a"),
        _pred("a", "A", ["O", "PERIOD"], "crf_a"),
    ]


@pytest.fixture
def case() -> list[dict[str, Any]]:  # another order on purpose: the output follows the punct order
    return [
        _pred("a", "CASE", ["FIRST_UPPER", "ALL_UPPER"], "crf_case"),
        _pred("b", "CASE", ["FIRST_UPPER", "LOWER", "LOWER"], "crf_case"),
    ]


def test_hand_example_joins_labels_in_punct_order_with_the_new_model_name(punct, case):
    assert combine(punct, case, model_name="crf_factored") == [
        {"sample_id": "b", "model_name": "crf_factored", "label_mode": "C",
         "predicted_labels": ["COMMA|FIRST_UPPER", "O|LOWER", "QUESTION|LOWER"]},
        {"sample_id": "a", "model_name": "crf_factored", "label_mode": "C",
         "predicted_labels": ["O|FIRST_UPPER", "PERIOD|ALL_UPPER"]},
    ]


def test_output_key_order_is_the_prediction_file_format(punct, case):
    row = combine(punct, case, model_name="x")[0]
    assert list(row) == ["sample_id", "model_name", "label_mode", "predicted_labels"]


def test_inputs_are_not_mutated(punct, case):
    before = copy.deepcopy((punct, case))
    combine(punct, case, model_name="x")
    assert (punct, case) == before


def test_every_punct_and_casing_pair_is_a_mode_c_label():
    sets = load_label_sets()
    pairs = [(p, c) for p in sets["A"] for c in sets["casing"]]
    punct = [_pred("s", "A", [p for p, _ in pairs])]
    case = [_pred("s", "CASE", [c for _, c in pairs])]
    labels = combine(punct, case, model_name="x")[0]["predicted_labels"]
    assert labels == sets["C"]


def test_empty_inputs_give_no_predictions():
    assert combine([], [], model_name="x") == []


def _err(punct, case, **kwargs) -> str:
    with pytest.raises(ValueError) as stop:
        combine(punct, case, model_name=kwargs.get("model_name", "x"))
    message = str(stop.value)
    assert "\n" not in message
    return message


def test_id_only_in_punct_is_named(punct, case):
    assert "'b'" in _err(punct, case[:1]) and "case" in _err(punct, case[:1])


def test_id_only_in_case_is_named(punct, case):
    message = _err(punct[:1], case)
    assert "'a'" in message and "punct" in message


def test_duplicate_id_in_punct_is_named(punct, case):
    message = _err([*punct, punct[1]], case)
    assert "duplicate" in message and "'a'" in message and "punct" in message


def test_duplicate_id_in_case_is_named(punct, case):
    message = _err(punct, [*case, case[0]])
    assert "duplicate" in message and "'a'" in message and "case" in message


def test_length_mismatch_names_the_first_offending_id(punct, case):
    case[1] = _pred("b", "CASE", ["LOWER"] * 2)
    case[0] = _pred("a", "CASE", ["LOWER"] * 3)
    message = _err(punct, case)
    assert "'b'" in message and "3" in message and "2" in message


def test_wrong_punct_mode_is_refused(punct, case):
    punct[0] = {**punct[0], "label_mode": "B"}
    message = _err(punct, case)
    assert "'b'" in message and "label_mode" in message and "'A'" in message


@pytest.mark.parametrize("wrong", ["A", "C", None])
def test_wrong_case_mode_is_refused(punct, case, wrong):
    case[0] = {**case[0], "label_mode": wrong}
    message = _err(punct, case)
    assert "'a'" in message and "label_mode" in message and "'CASE'" in message


def test_unknown_punct_label_is_refused(punct, case):
    punct[0]["predicted_labels"][1] = "EXCLAMATION"
    message = _err(punct, case)
    assert "'b'" in message and "'EXCLAMATION'" in message


def test_joint_label_in_punct_is_refused(punct, case):
    punct[1]["predicted_labels"][0] = "O|LOWER"
    assert "'O|LOWER'" in _err(punct, case)


def test_unknown_case_label_is_refused(punct, case):
    case[1]["predicted_labels"][2] = "MIXED"
    message = _err(punct, case)
    assert "'b'" in message and "'MIXED'" in message


@pytest.mark.parametrize("labels", ["O", None, [["O"]], [1]])
def test_labels_that_are_not_a_list_of_strings_are_refused(punct, case, labels):
    punct[0] = {**punct[0], "predicted_labels": labels}
    assert "'b'" in _err(punct, case)


@pytest.mark.parametrize("record", [None, "x", {"predicted_labels": []}, {"sample_id": "", "label_mode": "A"}])
def test_records_without_a_usable_sample_id_are_refused(punct, case, record):
    message = _err([record, *punct], case)
    assert "punct" in message


@pytest.mark.parametrize("name", ["", None, 3])
def test_model_name_must_be_a_non_empty_string(punct, case, name):
    assert "model_name" in _err(punct, case, model_name=name)


def test_a_joint_label_outside_the_mode_c_vocabulary_is_refused(punct, case, monkeypatch):
    real = load_label_sets()
    trimmed = {**real, "C": [label for label in real["C"] if label != "QUESTION|LOWER"]}
    monkeypatch.setattr(factored, "load_label_sets", lambda: trimmed)
    message = _err(punct, case)
    assert "'b'" in message and "QUESTION|LOWER" in message
