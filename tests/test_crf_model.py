"""Comprehensive unit and contract tests for CRFModel.

Covers:
A. Initialization and parameter validation.
B. Training validation, target derivation, and gold leakage prevention.
C. Prediction contract, schema compliance, length preservation, and unseen word handling.
D. Feature integration (basic vs improved presets with same CRF engine).
E. Model persistence (save/load roundtrip, prediction equivalence, corruption rejection).
F. Modes A (4), B (8), and C (12 joint classes).
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import yaml

from punct_restore.features.crf_features import FeatureConfig
from punct_restore.models.crf import CRFModel, get_mode_vocabulary

# -------------------------------------------------------------------------------------------------
# Synthetic Test Fixtures (Explicitly synthetic, not real SponSpeech data)
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def synth_mode_a_data() -> list[dict]:
    return [
        {"sample_id": "synth_a_1", "tokens": ["hello", "everyone"], "punct_labels": ["O", "PERIOD"]},
        {"sample_id": "synth_a_2", "tokens": ["did", "you", "understand"], "punct_labels": ["O", "O", "QUESTION"]},
        {"sample_id": "synth_a_3", "tokens": ["yes", "we", "can"], "punct_labels": ["COMMA", "O", "PERIOD"]},
        {"sample_id": "synth_a_4", "tokens": ["well", "why", "not"], "punct_labels": ["COMMA", "O", "QUESTION"]},
    ]


@pytest.fixture
def synth_mode_b_data() -> list[dict]:
    return [
        {"sample_id": "synth_b_1", "tokens": ["look", "out"], "punct_labels": ["COMMA", "EXCLAMATION"]},
        {"sample_id": "synth_b_2", "tokens": ["chapter", "one", "notes"], "punct_labels": ["COLON", "DASH", "PERIOD"]},
        {"sample_id": "synth_b_3", "tokens": ["however", "it", "failed"], "punct_labels": ["SEMICOLON", "O", "PERIOD"]},
    ]


@pytest.fixture
def synth_mode_c_data() -> list[dict]:
    return [
        {
            "sample_id": "synth_c_1",
            "tokens": ["today", "we", "discuss", "nlp"],
            "punct_labels": ["O", "O", "O", "PERIOD"],
            "case_labels": ["FIRST_UPPER", "LOWER", "LOWER", "ALL_UPPER"],
        },
        {
            "sample_id": "synth_c_2",
            "tokens": ["did", "you", "agree"],
            "punct_labels": ["O", "O", "QUESTION"],
            "case_labels": ["FIRST_UPPER", "LOWER", "LOWER"],
        },
        {
            "sample_id": "synth_c_3",
            "tokens": ["yes", "indeed"],
            "punct_labels": ["COMMA", "PERIOD"],
            "case_labels": ["FIRST_UPPER", "LOWER"],
        },
    ]


# -------------------------------------------------------------------------------------------------
# Category A: Initialization & Configuration Validation
# -------------------------------------------------------------------------------------------------

def test_valid_initialization():
    for mode, expected_len in [("A", 4), ("B", 8), ("C", 12)]:
        model = CRFModel(label_mode=mode)
        assert model.label_mode == mode
        assert len(model.vocabulary) == expected_len
        assert not model.is_fitted


@pytest.mark.parametrize("bad_mode", ["D", "", "X", "12", None])
def test_invalid_mode_rejected(bad_mode):
    with pytest.raises(ValueError, match="Invalid label_mode|Unsupported label_mode"):
        CRFModel(label_mode=bad_mode)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_c1", [-0.1, -1.0, math.nan, math.inf, "0.1"])
def test_invalid_c1_rejected(bad_c1):
    with pytest.raises(ValueError):
        CRFModel(c1=bad_c1)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_c2", [-0.1, -5.0, math.nan, math.inf, None])
def test_invalid_c2_rejected(bad_c2):
    with pytest.raises(ValueError):
        CRFModel(c2=bad_c2)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_iter", [0, -10, 1.5, "100"])
def test_invalid_max_iterations_rejected(bad_iter):
    with pytest.raises(ValueError):
        CRFModel(max_iterations=bad_iter)  # type: ignore[arg-type]


def test_invalid_feature_config_type_rejected():
    with pytest.raises(TypeError, match="Invalid feature_config type"):
        CRFModel(feature_config=[1, 2, 3])  # type: ignore[arg-type]


# -------------------------------------------------------------------------------------------------
# Category B: Training Validation & Target Derivation
# -------------------------------------------------------------------------------------------------

def test_fit_mode_a_success(synth_mode_a_data):
    model = CRFModel(label_mode="A", max_iterations=10)
    res = model.fit(synth_mode_a_data)
    assert res is model
    assert model.is_fitted
    assert model._training_metadata["num_sequences"] == 4
    assert model._training_metadata["num_tokens"] == 11


def test_fit_rejects_empty_records():
    model = CRFModel(label_mode="A")
    with pytest.raises(ValueError, match="cannot be empty"):
        model.fit([])


def test_fit_rejects_duplicate_sample_ids():
    model = CRFModel(label_mode="A")
    dupe_data = [
        {"sample_id": "dupe_1", "tokens": ["hello"], "punct_labels": ["PERIOD"]},
        {"sample_id": "dupe_1", "tokens": ["world"], "punct_labels": ["PERIOD"]},
    ]
    with pytest.raises(ValueError, match="duplicate sample_id"):
        model.fit(dupe_data)


def test_fit_rejects_mismatched_token_label_lengths():
    model = CRFModel(label_mode="A")
    bad_data = [
        {"sample_id": "bad_len", "tokens": ["hello", "world"], "punct_labels": ["PERIOD"]}
    ]
    with pytest.raises(ValueError, match="does not match tokens length"):
        model.fit(bad_data)


def test_fit_rejects_unknown_punctuation_label_for_mode():
    model = CRFModel(label_mode="A")
    # EXCLAMATION belongs to Mode B, not Mode A
    bad_data = [
        {"sample_id": "bad_label", "tokens": ["look", "out"], "punct_labels": ["O", "EXCLAMATION"]}
    ]
    with pytest.raises(ValueError, match="unknown punct_labels"):
        model.fit(bad_data)


def test_fit_rejects_test_split_records():
    model = CRFModel(label_mode="A")
    test_leak = [
        {"sample_id": "leak_1", "split": "test", "tokens": ["hello"], "punct_labels": ["PERIOD"]}
    ]
    with pytest.raises(ValueError, match="cannot train on record with split='test'"):
        model.fit(test_leak)


def test_fit_preserves_input_immutability(synth_mode_a_data):
    original = copy.deepcopy(synth_mode_a_data)
    model = CRFModel(label_mode="A", max_iterations=5)
    model.fit(synth_mode_a_data)
    assert synth_mode_a_data == original


# -------------------------------------------------------------------------------------------------
# Category C: Prediction Contract & Schema Verification
# -------------------------------------------------------------------------------------------------

def test_predict_before_fit_raises_error():
    model = CRFModel(label_mode="A")
    with pytest.raises(RuntimeError, match="CRFModel is not fitted"):
        model.predict([{"sample_id": "s1", "tokens": ["hello"]}])


def test_predict_schema_and_length_preservation(synth_mode_a_data):
    model = CRFModel(label_mode="A", max_iterations=10)
    model.fit(synth_mode_a_data)

    test_inputs = [
        {"sample_id": "test_1", "tokens": ["hello", "everyone", "today"]},
        {"sample_id": "test_2", "tokens": ["did", "you"]},
    ]
    preds = model.predict(test_inputs)

    assert len(preds) == 2
    for p, orig in zip(preds, test_inputs, strict=False):
        assert p["sample_id"] == orig["sample_id"]
        assert p["label_mode"] == "A"
        assert p["model_name"] == "crf_basic"
        assert len(p["predicted_labels"]) == len(orig["tokens"])
        assert all(lbl in model.vocabulary for lbl in p["predicted_labels"])


def test_predict_empty_sequence_returns_empty_list(synth_mode_a_data):
    model = CRFModel(label_mode="A", max_iterations=5).fit(synth_mode_a_data)
    assert model.predict([]) == []


def test_predict_unseen_words_does_not_crash(synth_mode_a_data):
    model = CRFModel(label_mode="A", max_iterations=10).fit(synth_mode_a_data)
    unseen = [{"sample_id": "oov_1", "tokens": ["quantum", "fluctuation", "entanglement"]}]
    preds = model.predict(unseen)
    assert len(preds[0]["predicted_labels"]) == 3
    assert all(lbl in model.vocabulary for lbl in preds[0]["predicted_labels"])


def test_predict_tokens_adapter_convenience(synth_mode_a_data):
    model = CRFModel(label_mode="A", max_iterations=10).fit(synth_mode_a_data)
    tokens = ["hello", "everyone"]
    labels = model.predict_tokens(tokens)
    assert len(labels) == 2
    assert all(lbl in model.vocabulary for lbl in labels)


def test_predict_marginals_sums_to_one_and_matches_predict(synth_mode_a_data, synth_mode_c_data):
    model = CRFModel(label_mode="A", max_iterations=50).fit(synth_mode_a_data)
    records = [{"sample_id": r["sample_id"], "tokens": r["tokens"]} for r in synth_mode_a_data]
    marginals = model.predict_marginals(records)
    predicted = model.predict(records)

    assert len(marginals) == len(predicted) == len(records)
    for seq, pred in zip(marginals, predicted, strict=True):
        assert len(seq) == len(pred["predicted_labels"])
        for token_probs, label in zip(seq, pred["predicted_labels"], strict=True):
            assert abs(sum(token_probs.values()) - 1.0) <= 1e-6
            assert list(token_probs) == model.vocabulary
            assert max(token_probs, key=token_probs.get) == label

    # Mode C: 12 joint labels, labels never seen in training get 0.0 but stay present.
    model_c = CRFModel(label_mode="C", max_iterations=20).fit(synth_mode_c_data)
    c_records = [{"sample_id": "c1", "tokens": ["today", "we", "learn"]}]
    for token_probs in model_c.predict_marginals(c_records)[0]:
        assert abs(sum(token_probs.values()) - 1.0) <= 1e-6
        assert list(token_probs) == model_c.vocabulary
        assert len(token_probs) == 12


def test_predict_marginals_requires_fitted_model():
    model = CRFModel(label_mode="A")
    records = [{"sample_id": "s1", "tokens": ["hello"]}]
    with pytest.raises(RuntimeError) as predict_error:
        model.predict(records)
    with pytest.raises(type(predict_error.value), match="CRFModel is not fitted"):
        model.predict_marginals(records)


# -------------------------------------------------------------------------------------------------
# Category D: Feature Integration (Basic vs Improved)
# -------------------------------------------------------------------------------------------------

def test_basic_and_improved_presets_with_same_crf_engine(synth_mode_a_data):
    # Train with basic preset
    model_basic = CRFModel(
        label_mode="A",
        feature_config=FeatureConfig.basic(),
        max_iterations=10,
    ).fit(synth_mode_a_data)

    # Train with improved preset
    model_improved = CRFModel(
        label_mode="A",
        feature_config=FeatureConfig.improved(),
        max_iterations=10,
    ).fit(synth_mode_a_data)

    test_rec = [{"sample_id": "t1", "tokens": ["did", "you", "understand"]}]
    preds_b = model_basic.predict(test_rec)
    preds_i = model_improved.predict(test_rec)

    assert len(preds_b[0]["predicted_labels"]) == 3
    assert len(preds_i[0]["predicted_labels"]) == 3
    assert model_basic.model_name == "crf_basic"
    assert model_improved.model_name == "crf_improved"


# -------------------------------------------------------------------------------------------------
# Category E: Persistence (Save / Load Equivalence)
# -------------------------------------------------------------------------------------------------

def test_save_load_equivalence(synth_mode_a_data, tmp_path):
    model = CRFModel(
        label_mode="A",
        c1=0.2,
        c2=0.3,
        feature_config=FeatureConfig.improved(),
        max_iterations=15,
    ).fit(synth_mode_a_data)

    test_input = [
        {"sample_id": "eval_1", "tokens": ["hello", "world"]},
        {"sample_id": "eval_2", "tokens": ["why", "not", "today"]},
    ]
    original_preds = model.predict(test_input)

    artifact_path = tmp_path / "crf_model.joblib"
    model.save_model(artifact_path)
    assert artifact_path.is_file()

    loaded_model = CRFModel.load_model(artifact_path)
    assert loaded_model.is_fitted
    assert loaded_model.label_mode == model.label_mode
    assert loaded_model.vocabulary == model.vocabulary
    assert loaded_model.c1 == model.c1
    assert loaded_model.c2 == model.c2
    assert loaded_model.feature_config == model.feature_config

    loaded_preds = loaded_model.predict(test_input)
    assert original_preds == loaded_preds


def test_save_unfitted_model_raises_error(tmp_path):
    model = CRFModel(label_mode="A")
    with pytest.raises(RuntimeError, match="Cannot save an unfitted"):
        model.save_model(tmp_path / "model.joblib")


def test_load_nonexistent_file_raises_error():
    with pytest.raises(FileNotFoundError):
        CRFModel.load_model(Path("nonexistent_path_123.joblib"))


def test_load_corrupt_file_raises_error(tmp_path):
    corrupt_file = tmp_path / "corrupt.joblib"
    corrupt_file.write_text("not a valid joblib file", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt or unreadable"):
        CRFModel.load_model(corrupt_file)


# -------------------------------------------------------------------------------------------------
# Category F: Mode Handling (Modes A, B, and C)
# -------------------------------------------------------------------------------------------------

def test_mode_b_training_and_prediction(synth_mode_b_data):
    model_b = CRFModel(label_mode="B", max_iterations=10).fit(synth_mode_b_data)
    assert len(model_b.vocabulary) == 8
    assert "DASH" in model_b.vocabulary
    assert "COLON" in model_b.vocabulary

    preds = model_b.predict([{"sample_id": "b1", "tokens": ["chapter", "one"]}])
    assert len(preds[0]["predicted_labels"]) == 2
    assert all(lbl in model_b.vocabulary for lbl in preds[0]["predicted_labels"])


def test_mode_c_joint_training_and_prediction(synth_mode_c_data):
    model_c = CRFModel(label_mode="C", max_iterations=10).fit(synth_mode_c_data)
    assert len(model_c.vocabulary) == 12

    # Verify joint label format
    for lbl in model_c.vocabulary:
        assert "|" in lbl
        punct, case = lbl.split("|")
        assert punct in ["O", "COMMA", "PERIOD", "QUESTION"]
        assert case in ["LOWER", "FIRST_UPPER", "ALL_UPPER"]

    test_input = [{"sample_id": "c1", "tokens": ["today", "we", "learn"]}]
    preds = model_c.predict(test_input)
    assert len(preds[0]["predicted_labels"]) == 3
    for lbl in preds[0]["predicted_labels"]:
        assert lbl in model_c.vocabulary
        punct, case = lbl.split("|")
        assert punct in get_mode_vocabulary("A")


def test_mode_c_rejects_missing_case_labels():
    model_c = CRFModel(label_mode="C")
    missing_case = [
        {"sample_id": "c_bad", "tokens": ["hello"], "punct_labels": ["PERIOD"]}
    ]
    with pytest.raises(ValueError, match="case_labels length"):
        model_c.fit(missing_case)


def test_mode_c_rejects_mismatched_casing_array_length():
    model_c = CRFModel(label_mode="C")
    mismatched = [
        {
            "sample_id": "c_bad_len",
            "tokens": ["hello", "world"],
            "punct_labels": ["O", "PERIOD"],
            "case_labels": ["LOWER"],  # only 1 case label for 2 tokens
        }
    ]
    with pytest.raises(ValueError, match="does not match tokens length"):
        model_c.fit(mismatched)


# -------------------------------------------------------------------------------------------------
# Category G: Configuration Hardening & Regression Tests (Phase 4.1)
# -------------------------------------------------------------------------------------------------

def test_get_mode_vocabulary_missing_file_raises_error(tmp_path):
    """Missing config file must raise FileNotFoundError, not silently fall back."""
    missing = tmp_path / "nonexistent_label_sets.yaml"
    with pytest.raises(FileNotFoundError):
        get_mode_vocabulary("A", config_path=missing)


def test_get_mode_vocabulary_malformed_yaml_raises_error(tmp_path):
    """Malformed YAML syntax must raise error, not silently fall back."""
    bad_yaml = tmp_path / "malformed.yaml"
    bad_yaml.write_text("punctuation: [unclosed list", encoding="utf-8")
    with pytest.raises(yaml.parser.ParserError):
        get_mode_vocabulary("A", config_path=bad_yaml)


def test_get_mode_vocabulary_inconsistent_structure_raises_error(tmp_path):
    """Inconsistent structure (e.g. 5 standard labels instead of 4) must raise ValueError."""
    bad_struct = tmp_path / "bad_structure.yaml"
    content = """
punctuation:
  standard: [O, COMMA, PERIOD, QUESTION, EXTRA]
  extended: [O, COMMA, PERIOD, QUESTION, EXCLAMATION, COLON, SEMICOLON, DASH]
casing: [LOWER, FIRST_UPPER, ALL_UPPER]
modes:
  C:
    separator: "|"
    labels: []
extended_to_standard: {}
"""
    bad_struct.write_text(content.strip(), encoding="utf-8")
    with pytest.raises(ValueError, match="agreed 4/8/12 structure"):
        get_mode_vocabulary("A", config_path=bad_struct)


def test_get_mode_vocabulary_authoritative_config():
    """Verify authoritative label sets loaded without fallback."""
    vocab_a = get_mode_vocabulary("A")
    vocab_b = get_mode_vocabulary("B")
    vocab_c = get_mode_vocabulary("C")

    assert vocab_a == ["O", "COMMA", "PERIOD", "QUESTION"]
    assert len(vocab_b) == 8
    assert len(vocab_c) == 12


def test_fit_rejects_validation_split_records():
    """Ensure validation split records cannot be used for training (leakage prevention)."""
    model = CRFModel(label_mode="A")
    val_record = [
        {"sample_id": "val_1", "split": "validation", "tokens": ["hello"], "punct_labels": ["PERIOD"]}
    ]
    with pytest.raises(ValueError, match="Training data must have split='train'"):
        model.fit(val_record)


def test_fit_records_optional_provenance_metadata(synth_mode_a_data):
    """Ensure optional provenance metadata is attached to model training metadata."""
    model = CRFModel(label_mode="A", max_iterations=5)
    provenance = {
        "dataset_release": "sponspeech-m1-v2",
        "split_manifest_hash": "abc123hash",
        "run_id": "exp_01",
    }
    model.fit(synth_mode_a_data, config={"provenance": provenance})

    assert "provenance" in model._training_metadata
    assert model._training_metadata["provenance"]["dataset_release"] == "sponspeech-m1-v2"
    assert model._training_metadata["provenance"]["run_id"] == "exp_01"
