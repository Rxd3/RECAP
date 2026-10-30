"""Tests for GapClassifier, "CRF without transitions" (per-token logistic regression on CRF features)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from punct_restore.features.crf_features import FeatureConfig, extract_sequence_features
from punct_restore.models import gap_classifier
from punct_restore.models.crf import CRFModel, get_mode_vocabulary
from punct_restore.models.gap_classifier import GapClassifier


def _rec(
    sample_id: str,
    tokens: list[str],
    punct: list[str],
    case: list[str] | None = None,
    split: str = "train",
) -> dict[str, Any]:
    record: dict[str, Any] = {"sample_id": sample_id, "split": split, "tokens": tokens, "punct_labels": punct}
    if case is not None:
        record["case_labels"] = case
    return record


MODE_A = [
    _rec("a1", ["hello", "everyone"], ["O", "PERIOD"]),
    _rec("a2", ["did", "you", "understand"], ["O", "O", "QUESTION"]),
    _rec("a3", ["yes", "we", "can"], ["COMMA", "O", "PERIOD"]),
    _rec("a4", ["well", "why", "not"], ["COMMA", "O", "QUESTION"]),
]
MODE_B = [
    _rec("b1", ["look", "out"], ["COMMA", "EXCLAMATION"]),
    _rec("b2", ["chapter", "one", "notes"], ["COLON", "DASH", "PERIOD"]),
    _rec("b3", ["however", "it", "failed"], ["SEMICOLON", "O", "PERIOD"]),
]
MODE_C = [
    _rec("c1", ["today", "we", "discuss", "nlp"], ["O", "O", "O", "PERIOD"],
         ["FIRST_UPPER", "LOWER", "LOWER", "ALL_UPPER"]),
    _rec("c2", ["did", "you", "agree"], ["O", "O", "QUESTION"], ["FIRST_UPPER", "LOWER", "LOWER"]),
    _rec("c3", ["yes", "indeed"], ["COMMA", "PERIOD"], ["FIRST_UPPER", "LOWER"]),
]
GOLD_FREE = [
    {"sample_id": "p1", "tokens": ["did", "you", "see", "zebras"]},
    {"sample_id": "p2", "tokens": ["hello"]},
]


def _fit_a(**kwargs: Any) -> GapClassifier:
    return GapClassifier("A", **kwargs).fit(MODE_A)


# --- the one thing that distinguishes it from the CRF: no label transitions ---------------------

GOLD_ALTERNATING = ["O", "COMMA", "O", "COMMA", "O", "COMMA", "O", "PERIOD"]


def _alternating(prefix: str, count: int) -> list[dict[str, Any]]:
    return [_rec(f"{prefix}{i}", ["x"] * 8, GOLD_ALTERNATING) for i in range(count)]


def _accuracy(predictions: list[dict[str, Any]], records: list[dict[str, Any]]) -> float:
    pairs = [(p, g) for pred, rec in zip(predictions, records, strict=True)
             for p, g in zip(pred["predicted_labels"], rec["punct_labels"], strict=True)]
    return sum(p == g for p, g in pairs) / len(pairs)


def test_gap_model_cannot_reproduce_a_pattern_that_only_transitions_define():
    # With the basic preset (window +-1, no positional buckets) every INTERIOR token of "x x x ..."
    # has the same feature dict: only BOS/EOS tell the two ends apart. The gold labels alternate
    # O, COMMA, O, COMMA, ... so the only evidence for "COMMA now" is "the previous label was O",
    # which is a label transition. A per-token model sees six identical rows with three COMMA and
    # three O labels, so it gives all six the same label and can get at most 3 of them right
    # (plus the two ends) = 5/8. A CRF learns the O<->COMMA transitions and reproduces it exactly.
    config = FeatureConfig.basic()
    features = extract_sequence_features(["x"] * 8, config=config)
    assert all(feats == features[1] for feats in features[1:7])
    assert features[0] != features[1] and features[7] != features[1]

    train, held_out = _alternating("train", 6), _alternating("held", 3)
    gap = GapClassifier("A", feature_config=config).fit(train)
    crf = CRFModel(label_mode="A", feature_config=config, max_iterations=100).fit(train)

    gap_preds, crf_preds = gap.predict(held_out), crf.predict(held_out)

    for pred in gap_preds:
        assert len(set(pred["predicted_labels"][1:7])) == 1
    assert _accuracy(crf_preds, held_out) == 1.0
    assert _accuracy(gap_preds, held_out) < 1.0
    assert _accuracy(gap_preds, held_out) <= 5 / 8


# --- constructor ---------------------------------------------------------------------------------


@pytest.mark.parametrize("label_mode", ["Z", "a", "", None, 1])
def test_bad_label_mode_is_a_value_error(label_mode):
    with pytest.raises(ValueError):
        GapClassifier(label_mode)


@pytest.mark.parametrize("value", [0, -1.0, float("nan"), float("inf"), "1", None, True])
def test_bad_C_is_a_value_error(value):
    with pytest.raises(ValueError, match="C"):
        GapClassifier("A", C=value)


@pytest.mark.parametrize("value", [0, -3, 1.5, "10", None, True])
def test_bad_max_iter_is_a_value_error(value):
    with pytest.raises(ValueError, match="max_iter"):
        GapClassifier("A", max_iter=value)


@pytest.mark.parametrize("value", ["", "   ", 5, None])
def test_bad_model_name_is_a_value_error(value):
    with pytest.raises(ValueError, match="model_name"):
        GapClassifier("A", model_name=value)


def test_feature_config_must_be_a_feature_config():
    with pytest.raises(TypeError, match="feature_config"):
        GapClassifier("A", feature_config={"preset": "improved"})  # type: ignore[arg-type]


def test_defaults_and_vocabulary_come_from_the_label_sets():
    model = GapClassifier("B")
    assert model.label_mode == "B"
    assert model.vocabulary == get_mode_vocabulary("B")
    assert model.feature_config == FeatureConfig.improved()
    assert model.model_name == "gap_improved"


# --- fit -----------------------------------------------------------------------------------------


def test_fit_returns_self_and_summarises_the_training_data():
    model = GapClassifier("A")
    assert model.fit(MODE_A) is model
    summary = model.training_summary
    assert summary["num_sequences"] == 4
    assert summary["num_tokens"] == 11
    assert summary["num_features"] > 11
    assert summary["iterations"] >= 1


def test_training_summary_is_a_copy():
    model = _fit_a()
    model.training_summary["num_tokens"] = -1
    assert model.training_summary["num_tokens"] == 11


def test_single_label_training_data_is_rejected():
    only_o = [_rec("s1", ["a", "b"], ["O", "O"]), _rec("s2", ["c"], ["O"])]
    with pytest.raises(ValueError, match="at least two labels"):
        GapClassifier("A").fit(only_o)


def test_non_convergence_is_warned_not_hidden():
    with pytest.warns(ConvergenceWarning):
        GapClassifier("A", max_iter=1).fit(MODE_A)


@pytest.mark.parametrize(
    "bad_records",
    [
        [],
        [_rec("v1", ["a"], ["PERIOD"], split="validation")],
        [_rec("d", ["a"], ["PERIOD"]), _rec("d", ["b"], ["O"])],
        [_rec("u", ["a"], ["EXCLAMATION"])],
        [_rec("l", ["a", "b"], ["O"])],
        [_rec("t", [], [])],
    ],
)
def test_fit_applies_the_crf_validation(bad_records):
    with pytest.raises(ValueError):
        GapClassifier("A").fit(bad_records)


def test_fit_rejects_a_non_sequence():
    with pytest.raises(TypeError):
        GapClassifier("A").fit("not records")  # type: ignore[arg-type]


# --- predict -------------------------------------------------------------------------------------


@pytest.mark.parametrize(("mode", "records"), [("A", MODE_A), ("B", MODE_B), ("C", MODE_C)])
def test_predict_output_contract_in_every_mode(mode, records):
    model = GapClassifier(mode, model_name="gap_x").fit(records)
    predictions = model.predict(GOLD_FREE)
    vocabulary = set(get_mode_vocabulary(mode))
    assert [p["sample_id"] for p in predictions] == ["p1", "p2"]
    for pred, rec in zip(predictions, GOLD_FREE, strict=True):
        assert set(pred) == {"sample_id", "model_name", "label_mode", "predicted_labels"}
        assert pred["model_name"] == "gap_x" and pred["label_mode"] == mode
        assert len(pred["predicted_labels"]) == len(rec["tokens"])
        assert set(pred["predicted_labels"]) <= vocabulary


def test_mode_c_predicts_joint_labels():
    labels = {lbl for p in GapClassifier("C").fit(MODE_C).predict(GOLD_FREE) for lbl in p["predicted_labels"]}
    assert labels and all("|" in lbl for lbl in labels)


def test_predict_is_deterministic():
    model = _fit_a()
    assert model.predict(GOLD_FREE) == model.predict(GOLD_FREE)


def test_predict_empty_records_gives_empty_list():
    assert _fit_a().predict([]) == []


def test_predict_unfitted_is_a_runtime_error():
    with pytest.raises(RuntimeError):
        GapClassifier("A").predict(GOLD_FREE)


@pytest.mark.parametrize("bad", ["text", [{"sample_id": "", "tokens": ["a"]}], [{"sample_id": "q", "tokens": []}],
                                 [{"sample_id": "q", "tokens": ["a", " "]}]])
def test_predict_applies_the_crf_input_validation(bad):
    with pytest.raises((TypeError, ValueError)):
        _fit_a().predict(bad)


def test_predicting_a_label_outside_the_vocabulary_is_a_value_error():
    model = _fit_a()
    model._classifier = SimpleNamespace(predict=lambda matrix: ["BOGUS"] * matrix.shape[0])
    # the CRF's own wording, because the check is the CRF's _require_known_labels
    with pytest.raises(ValueError, match=r"Model predicted label 'BOGUS' outside declared Mode A vocabulary\."):
        model.predict(GOLD_FREE)


def test_predict_checks_the_fit_once_not_once_per_chunk(monkeypatch):
    model = _fit_a()
    monkeypatch.setattr(gap_classifier, "CHUNK_RECORDS", 1)
    calls: list[None] = []
    original = GapClassifier._fitted

    def counting(self):
        calls.append(None)
        return original(self)

    monkeypatch.setattr(GapClassifier, "_fitted", counting)
    assert len(model.predict(MORE_GOLD_FREE)) == 3  # three chunks of one record
    assert len(calls) == 1


def test_features_are_the_crfs_under_a_non_default_config():
    config = FeatureConfig.improved(window_size=1, use_filler_cues=True)
    model = GapClassifier("A", feature_config=config).fit(MODE_A)

    crf_features, _ = CRFModel(label_mode="A", feature_config=config)._validate_training_records(MODE_A)
    expected = DictVectorizer().fit([token for sequence in crf_features for token in sequence]).vocabulary_
    default_features, _ = CRFModel(label_mode="A")._validate_training_records(MODE_A)
    default = DictVectorizer().fit([token for sequence in default_features for token in sequence]).vocabulary_

    assert set(model._vectorizer.vocabulary_) == set(expected)
    assert set(expected) != set(default)  # the config really changes the features, so the equality means something


# --- chunking: bounded memory, identical results ---------------------------------------------------

MORE_GOLD_FREE = [*GOLD_FREE, {"sample_id": "p3", "tokens": ["well", "why", "not"]}]


def _fitted_parts(model: GapClassifier) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    return model._vectorizer.vocabulary_, model._classifier.coef_, model._classifier.intercept_


@pytest.mark.parametrize("chunk", [1, 2, 3])
@pytest.mark.parametrize(("mode", "records"), [("A", MODE_A), ("C", MODE_C)])
def test_chunked_fit_and_predict_equal_the_unchunked_ones(monkeypatch, chunk, mode, records):
    whole = GapClassifier(mode).fit(records)
    monkeypatch.setattr(gap_classifier, "CHUNK_RECORDS", chunk)
    chunked = GapClassifier(mode).fit(records)

    vocabulary, coef, intercept = _fitted_parts(whole)
    chunked_vocabulary, chunked_coef, chunked_intercept = _fitted_parts(chunked)
    assert chunked_vocabulary == vocabulary
    assert np.allclose(chunked_coef, coef) and np.allclose(chunked_intercept, intercept)
    assert chunked.training_summary == whole.training_summary
    assert chunked.predict(MORE_GOLD_FREE) == whole.predict(MORE_GOLD_FREE)


def _spy_chunk_sizes(monkeypatch, name: str) -> list[int]:
    sizes: list[int] = []
    original = getattr(CRFModel, name)

    def spy(self, records):
        sizes.append(len(records))
        return original(self, records)

    monkeypatch.setattr(CRFModel, name, spy)
    return sizes


def test_fit_validates_and_extracts_at_most_one_chunk_at_a_time(monkeypatch):
    monkeypatch.setattr(gap_classifier, "CHUNK_RECORDS", 3)
    sizes = _spy_chunk_sizes(monkeypatch, "_validate_training_records")
    GapClassifier("A").fit(MODE_A)  # 4 records: chunks of 3 and 1, once to learn the features, once to use them
    assert sizes == [3, 1, 3, 1]


def test_predict_extracts_at_most_one_chunk_at_a_time_and_keeps_the_order(monkeypatch):
    model = _fit_a()
    expected = model.predict(MORE_GOLD_FREE)
    monkeypatch.setattr(gap_classifier, "CHUNK_RECORDS", 2)
    sizes = _spy_chunk_sizes(monkeypatch, "_prediction_features")
    assert model.predict(MORE_GOLD_FREE) == expected
    assert sizes == [2, 1]
    assert [p["sample_id"] for p in expected] == ["p1", "p2", "p3"]


def test_duplicate_sample_ids_in_different_chunks_are_rejected(monkeypatch):
    monkeypatch.setattr(gap_classifier, "CHUNK_RECORDS", 1)
    records = [_rec("d", ["a"], ["PERIOD"]), _rec("e", ["b"], ["O"]), _rec("d", ["c"], ["O"])]
    with pytest.raises(ValueError, match=r"Record at index 2: duplicate sample_id 'd' detected\."):
        GapClassifier("A").fit(records)


@pytest.mark.parametrize(
    ("bad", "error", "message"),
    [
        (_rec("", ["a"], ["PERIOD"]), ValueError, "sample_id must be a non-empty string"),
        (_rec("v", ["a"], ["PERIOD"], split="validation"), ValueError, "cannot train on record with split"),
        (_rec("u", ["a"], ["EXCLAMATION"]), ValueError, "unknown punct_labels"),
        ("not a record", TypeError, "must be a Mapping"),
    ],
)
def test_a_bad_record_in_a_later_chunk_keeps_the_crf_error(monkeypatch, bad, error, message):
    monkeypatch.setattr(gap_classifier, "CHUNK_RECORDS", 1)
    with pytest.raises(error, match=message):
        GapClassifier("A").fit([*MODE_A, bad])


def test_a_bad_prediction_record_in_a_later_chunk_keeps_the_crf_error(monkeypatch):
    model = _fit_a()
    monkeypatch.setattr(gap_classifier, "CHUNK_RECORDS", 1)
    with pytest.raises(ValueError, match="tokens must be a non-empty list"):
        model.predict([*GOLD_FREE, {"sample_id": "q", "tokens": []}])


@pytest.mark.parametrize("not_a_sequence", [None, 5, iter(MODE_A)])
def test_fit_and_predict_reject_a_non_sequence_with_the_crf_type_error(not_a_sequence):
    with pytest.raises(TypeError, match="Sequence"):
        GapClassifier("A").fit(not_a_sequence)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Sequence"):
        _fit_a().predict(not_a_sequence)  # type: ignore[arg-type]


# --- persistence ---------------------------------------------------------------------------------


def _saved(tmp_path: Path, **kwargs: Any) -> tuple[GapClassifier, Path]:
    model = _fit_a(**kwargs)
    path = tmp_path / "nested" / "gap.joblib"
    model.save_model(path)
    return model, path


def _rewrite(path: Path, **changes: Any) -> None:
    bundle = joblib.load(path)
    for key, value in changes.items():
        if value is KeyError:
            del bundle[key]
        else:
            bundle[key] = value
    joblib.dump(bundle, path)


def test_save_load_round_trip(tmp_path):
    config = FeatureConfig.improved(window_size=1, use_filler_cues=True)
    model, path = _saved(tmp_path, feature_config=config, C=0.5, max_iter=77, model_name="gap_custom")

    loaded = GapClassifier.load_model(path)

    assert isinstance(loaded, GapClassifier)
    assert loaded.predict(GOLD_FREE) == model.predict(GOLD_FREE)
    assert loaded.feature_config == config
    assert (loaded.label_mode, loaded.model_name) == ("A", "gap_custom")
    assert loaded.vocabulary == model.vocabulary
    assert loaded.training_summary == model.training_summary
    bundle = joblib.load(path)
    assert bundle["schema_version"] == 1 and bundle["model_family"] == "gap_classifier"
    assert bundle["hyperparameters"] == {"C": 0.5, "max_iter": 77, "solver": "lbfgs"}
    assert bundle["feature_config"] == config.to_dict()


def test_save_unfitted_is_a_runtime_error(tmp_path):
    with pytest.raises(RuntimeError):
        GapClassifier("A").save_model(tmp_path / "x.joblib")


def test_load_missing_file_is_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        GapClassifier.load_model(tmp_path / "absent.joblib")


def test_load_folder_is_not_reported_as_missing(tmp_path):
    folder = tmp_path / "gap.joblib"
    folder.mkdir()
    with pytest.raises(ValueError, match=r"gap\.joblib: is a folder, not a model file") as exc:
        GapClassifier.load_model(folder)
    assert str(tmp_path) not in str(exc.value)


def test_load_os_error_is_not_relabelled_as_a_corrupt_artifact(tmp_path, monkeypatch):
    _, path = _saved(tmp_path)

    def denied(_: Path) -> None:
        raise PermissionError("access denied")

    monkeypatch.setattr(gap_classifier.joblib, "load", denied)
    with pytest.raises(PermissionError):
        GapClassifier.load_model(path)


def test_load_corrupt_file_is_one_friendly_value_error(tmp_path):
    path = tmp_path / "bad.joblib"
    path.write_bytes(b"this is not a joblib file")
    with pytest.raises(ValueError, match="bad.joblib") as exc:
        GapClassifier.load_model(path)
    assert "\n" not in str(exc.value) and str(tmp_path) not in str(exc.value)


def test_load_non_dict_is_a_value_error(tmp_path):
    path = tmp_path / "list.joblib"
    joblib.dump(["not", "a", "bundle"], path)
    with pytest.raises(ValueError):
        GapClassifier.load_model(path)


@pytest.mark.parametrize(
    "changes",
    [
        {"model_family": "crf"},
        {"schema_version": 2},
        {"label_mode": "Z"},
        {"vocabulary": ["O", "COMMA"]},
        {"model_name": ""},
        {"feature_config": "improved"},
        {"hyperparameters": {"C": -1.0, "max_iter": 10, "solver": "lbfgs"}},
        {"hyperparameters": KeyError},
        {"training_summary": KeyError},
        {"vectorizer": KeyError},
        {"classifier": KeyError},
        {"vectorizer": "not a vectorizer"},
        {"classifier": "not a classifier"},
        {"classifier": LogisticRegression()},  # right type, but never fitted
    ],
    ids=lambda changes: ",".join(f"{k}={'<missing>' if v is KeyError else v}" for k, v in changes.items()),
)
def test_load_rejects_a_bad_bundle(tmp_path, changes):
    _, path = _saved(tmp_path)
    _rewrite(path, **changes)
    with pytest.raises(ValueError):
        GapClassifier.load_model(path)


_IMPROVED = FeatureConfig.improved().to_dict()  # what _saved() writes by default


@pytest.mark.parametrize(
    ("changes", "detail"),
    [
        ({"label_mode": "Z"}, "label mode"),
        ({"hyperparameters": {"C": -1.0, "max_iter": 10, "solver": "lbfgs"}}, "C must be a positive finite number"),
        ({"hyperparameters": {"C": 1.0, "max_iter": 0, "solver": "lbfgs"}}, "max_iter must be a positive integer"),
        ({"hyperparameters": {"max_iter": 10, "solver": "lbfgs"}}, "C must be a positive finite number"),
        ({"model_name": " "}, "model_name must be a non-empty string"),
        ({"feature_config": {**_IMPROVED, "bogus": True}}, "feature_config"),
        ({"feature_config": {k: v for k, v in _IMPROVED.items() if k != "version"}}, "feature_config"),
    ],
    ids=["label_mode", "bad_C", "bad_max_iter", "missing_C", "blank_model_name", "extra_key", "missing_key"],
)
def test_load_reports_every_inconsistent_part_as_a_corrupt_artifact(tmp_path, changes, detail):
    _, path = _saved(tmp_path)
    _rewrite(path, **changes)
    with pytest.raises(ValueError, match=rf"Corrupt model artifact gap\.joblib: .*{detail}"):
        GapClassifier.load_model(path)


_LEGACY = {k: v for k, v in _IMPROVED.items() if k != "use_question_lookback"}  # saved before the switch


def test_load_accepts_a_feature_config_saved_before_the_question_lookback_switch(tmp_path):
    model, path = _saved(tmp_path)
    _rewrite(path, feature_config=_LEGACY)

    loaded = GapClassifier.load_model(path)

    assert loaded.feature_config == FeatureConfig.improved()
    assert loaded.feature_config.use_question_lookback is False
    assert loaded.predict(GOLD_FREE) == model.predict(GOLD_FREE)


def test_load_still_refuses_an_unknown_key_next_to_a_legacy_feature_config(tmp_path):
    _, path = _saved(tmp_path)
    _rewrite(path, feature_config={**_LEGACY, "bogus": True})
    with pytest.raises(ValueError, match=r"Corrupt model artifact gap\.joblib: .*feature_config"):
        GapClassifier.load_model(path)


def test_load_refuses_a_feature_config_missing_a_key_that_is_not_a_later_addition(tmp_path):
    _, path = _saved(tmp_path)
    _rewrite(path, feature_config={k: v for k, v in _LEGACY.items() if k != "window_size"})
    with pytest.raises(ValueError, match=r"Corrupt model artifact gap\.joblib: .*feature_config"):
        GapClassifier.load_model(path)


def test_load_refuses_a_present_key_that_does_not_round_trip(tmp_path):
    _, path = _saved(tmp_path)
    _rewrite(path, feature_config={**_IMPROVED, "window_size": float("nan")})  # nan != nan after rebuild
    with pytest.raises(ValueError, match=r"Corrupt model artifact gap\.joblib: .*feature_config"):
        GapClassifier.load_model(path)


def test_save_load_round_trip_with_the_question_lookback_switch_on(tmp_path):
    config = FeatureConfig.improved(use_question_lookback=True)
    _, path = _saved(tmp_path, feature_config=config)

    loaded = GapClassifier.load_model(path)

    assert loaded.feature_config == config
    assert joblib.load(path)["feature_config"]["use_question_lookback"] is True


def test_load_rejects_classes_outside_the_vocabulary(tmp_path):
    path = tmp_path / "b.joblib"
    GapClassifier("B").fit(MODE_B).save_model(path)  # trained on EXCLAMATION, COLON, ...
    _rewrite(path, label_mode="A", vocabulary=get_mode_vocabulary("A"))
    with pytest.raises(ValueError, match="vocabulary"):
        GapClassifier.load_model(path)
