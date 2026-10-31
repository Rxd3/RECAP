"""Member 2's HMM tests: synthetic fixtures only, never research results."""

import json
import math
from copy import deepcopy
from itertools import product
from pathlib import Path

import pytest
import yaml

from punct_restore.models.hmm import HMMModel

ROOT = Path(__file__).resolve().parents[1]
LABEL_CONFIG = yaml.safe_load((ROOT / "configs/label_sets.yaml").read_text(encoding="utf-8"))


def labels(mode="A"):
    config = LABEL_CONFIG["modes"][mode]
    return (
        config["labels"] if mode == "C" else LABEL_CONFIG["punctuation"][config["punctuation_set"]]
    )


def record(sample_id, tokens, punct, *, mode="A", casing=None, split="train"):
    return {
        "schema_version": 1,
        "sample_id": sample_id,
        "source": "synthetic_unit_fixture",
        "split": split,
        "label_mode": mode,
        "tokens": list(tokens),
        "punct_labels": list(punct),
        "case_labels": list(casing or ["LOWER"] * len(tokens)),
    }


@pytest.fixture
def training():
    return [
        record("s1", ["a", "b"], ["O", "PERIOD"]),
        record("s2", ["a", "a", "b"], ["O", "COMMA", "PERIOD"]),
        record("s3", ["c"], ["QUESTION"]),
    ]


@pytest.fixture
def fitted(training):
    return HMMModel(label_mode="A", labels=labels(), smoothing=1, min_token_count=1).fit(training)


def test_supervised_counts_start_end_and_emissions_by_hand(fitted, tmp_path):
    path = tmp_path / "model.json"
    fitted.save_model(path)
    counts = json.loads(path.read_text(encoding="utf-8"))["counts"]
    assert counts["start"] == [2, 0, 0, 1]
    assert counts["end"] == [0, 0, 2, 1]
    assert counts["state"] == [2, 1, 2, 1]
    assert counts["transition"] == [[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    assert counts["emissions"] == [{"a": 2}, {"a": 1}, {"b": 2}, {"c": 1}]
    # P(start=O)*P(a|O)*P(PERIOD|O)*P(b|PERIOD)*P(END|PERIOD).
    expected = (3 / 7) * (3 / 6) * (2 / 7) * (3 / 6) * (3 / 7)
    assert fitted.sequence_log_probability(["a", "b"], ["O", "PERIOD"]) == pytest.approx(
        math.log(expected)
    )


@pytest.mark.parametrize("words", [["a", "b"], ["a", "a", "b"], ["new", "new"], ["c"]])
def test_viterbi_matches_exhaustive_independent_probability_table(fitted, words):
    # Hand-derived probabilities for the fixture, including smoothed absent states.
    initial = [3 / 7, 1 / 7, 1 / 7, 2 / 7]
    transition = [
        [1 / 7, 2 / 7, 2 / 7, 1 / 7],
        [1 / 6, 1 / 6, 2 / 6, 1 / 6],
        [1 / 7] * 4,
        [1 / 6] * 4,
    ]
    end = [1 / 7, 1 / 6, 3 / 7, 2 / 6]
    emissions = [
        {"a": 3 / 6, "b": 1 / 6, "c": 1 / 6, "new": 1 / 6},
        {"a": 2 / 5, "b": 1 / 5, "c": 1 / 5, "new": 1 / 5},
        {"a": 1 / 6, "b": 3 / 6, "c": 1 / 6, "new": 1 / 6},
        {"a": 1 / 5, "b": 1 / 5, "c": 2 / 5, "new": 1 / 5},
    ]

    def probability(states):
        value = initial[states[0]] * emissions[states[0]][words[0]]
        for position in range(1, len(words)):
            value *= transition[states[position - 1]][states[position]]
            value *= emissions[states[position]][words[position]]
        return value * end[states[-1]]

    best = max(probability(states) for states in product(range(4), repeat=len(words)))
    predicted = fitted.decode(words)
    assert math.exp(fitted.sequence_log_probability(words, predicted)) == pytest.approx(best)


def test_end_transition_can_change_the_single_token_decision():
    training = [record(str(i), ["a"] * 4, ["O", "O", "O", "PERIOD"]) for i in range(10)]
    model = HMMModel(label_mode="A", labels=labels(), min_token_count=1).fit(training)
    # Start and emission terms alone favor O; learned termination favors PERIOD.
    assert model.decode(["a"]) == ["PERIOD"]


@pytest.mark.parametrize("mode,size", [("A", 4), ("B", 8), ("C", 12)])
def test_all_modes_obey_config_and_preserve_prediction_alignment(mode, size, tmp_path):
    punct = ["O", "COLON"] if mode == "B" else ["O", "PERIOD"]
    training = [
        record("train", ["hello", "nasa"], punct, mode=mode, casing=["FIRST_UPPER", "ALL_UPPER"])
    ]
    model = HMMModel(label_mode=mode, labels=labels(mode), min_token_count=1).fit(training)
    inputs = [
        {"schema_version": 1, "sample_id": "z-last", "tokens": ["unseen"]},
        {"schema_version": 1, "sample_id": "a-first", "tokens": ["hello", "nasa"]},
    ]
    predictions = model.predict(iter(inputs))
    assert len(model.labels) == size
    assert [p["sample_id"] for p in predictions] == ["z-last", "a-first"]
    for prediction, source in zip(predictions, inputs, strict=True):
        assert set(prediction) == {"sample_id", "model_name", "label_mode", "predicted_labels"}
        assert prediction["model_name"] == "hmm" and prediction["label_mode"] == mode
        assert len(prediction["predicted_labels"]) == len(source["tokens"])
        assert set(prediction["predicted_labels"]) <= set(model.labels)
    path = tmp_path / f"{mode}.json"
    model.save_model(path)
    assert HMMModel.load_model(path).predict(inputs) == predictions
    if mode == "C":
        assert predictions[1]["predicted_labels"] == ["O|FIRST_UPPER", "PERIOD|ALL_UPPER"]


def test_rare_words_pool_into_train_only_unknown_bucket(tmp_path):
    model = HMMModel(label_mode="A", labels=labels(), min_token_count=2).fit(
        [record("s1", ["keep", "keep", "rare"], ["O", "O", "QUESTION"])]
    )
    assert model.training_summary == {
        "num_sequences": 1,
        "num_tokens": 3,
        "vocabulary_size": 1,
        "unknown_training_tokens": 1,
    }
    for target in model.labels:
        assert model.sequence_log_probability(["rare"], [target]) == model.sequence_log_probability(
            ["never_seen"], [target]
        )
    before = model.training_summary
    model.decode(["never_seen", "another_unseen"])
    assert model.training_summary == before
    path = tmp_path / "model.json"
    model.save_model(path)
    counts = json.loads(path.read_text(encoding="utf-8"))["counts"]
    assert counts["vocabulary"] == ["keep"]
    assert counts["unknown"] == [0, 0, 0, 1]


def test_literal_unknown_token_does_not_collide_with_the_unknown_bucket():
    model = HMMModel(label_mode="A", labels=labels(), min_token_count=2).fit(
        [record("s1", ["<UNK>", "<UNK>", "rare"], ["O", "O", "QUESTION"])]
    )
    assert model.sequence_log_probability(["<UNK>"], ["O"]) != model.sequence_log_probability(
        ["new"], ["O"]
    )


def test_all_words_can_be_rare_and_absent_classes_still_decode():
    model = HMMModel(label_mode="A", labels=labels(), min_token_count=10).fit(
        [record("single", ["one"], ["PERIOD"])]
    )
    assert model.training_summary["vocabulary_size"] == 0
    for target in model.labels:
        assert math.isfinite(model.sequence_log_probability(["anything"], [target]))
    assert model.decode(["anything"]) == ["PERIOD"]


def test_long_sequence_does_not_underflow(fitted):
    words = ["a", "b", "new"] * 1000
    prediction = fitted.decode(words)
    assert len(prediction) == len(words)
    assert math.isfinite(fitted.sequence_log_probability(words, prediction))


def test_no_gold_or_original_text_leakage_and_no_input_mutation(fitted):
    clean = [{"schema_version": 1, "sample_id": "p", "tokens": ["a", "b"]}]
    poison = deepcopy(clean)
    poison[0].update(
        punct_labels=["QUESTION"] * 2,
        case_labels=None,
        label_mode="C",
        metadata={"original_text": "SECRET GOLD"},
    )
    original = deepcopy(poison)
    assert fitted.predict(poison) == fitted.predict(clean)
    assert poison == original


def test_fit_is_deterministic_and_does_not_modify_records(training, fitted):
    original = deepcopy(training)
    second = HMMModel(label_mode="A", labels=labels(), smoothing=1, min_token_count=1).fit(
        iter(training)
    )
    assert second.predict(training) == fitted.predict(training)
    assert training == original


def test_ties_follow_supplied_label_order():
    balanced = [record(str(i), ["a"], [label]) for i, label in enumerate(labels())]
    reverse = list(reversed(labels()))
    model = HMMModel(label_mode="A", labels=reverse, min_token_count=1).fit(balanced)
    assert model.decode(["a"]) == [reverse[0]]


@pytest.mark.parametrize("split", ["validation", "test"])
def test_fit_rejects_held_out_data(split):
    model = HMMModel(label_mode="A", labels=labels())
    with pytest.raises(ValueError, match="split=train"):
        model.fit([record("s", ["a"], ["O"], split=split)])


@pytest.mark.parametrize(
    "updates",
    [
        {"tokens": []},
        {"tokens": [" "]},
        {"tokens": [5]},
        {"schema_version": True},
        {"sample_id": ""},
        {"source": ""},
        {"punct_labels": []},
        {"case_labels": []},
        {"case_labels": ["MIXED"]},
        {"punct_labels": ["COLON"]},
        {"label_mode": "B"},
    ],
)
def test_malformed_training_is_rejected(updates):
    sample = record("s", ["a"], ["O"])
    sample.update(updates)
    with pytest.raises(ValueError):
        HMMModel(label_mode="A", labels=labels()).fit([sample])


def test_empty_and_duplicate_records_are_rejected(fitted, training):
    with pytest.raises(ValueError, match="at least one"):
        HMMModel(label_mode="A", labels=labels()).fit([])
    with pytest.raises(ValueError, match="duplicate"):
        HMMModel(label_mode="A", labels=labels()).fit([training[0], training[0]])
    with pytest.raises(ValueError, match="duplicate"):
        fitted.predict([training[0], training[0]])


def test_failed_refit_keeps_the_existing_model(fitted, training):
    before = fitted.predict(training)
    bad = record("bad", ["a"], ["O"], split="test")
    with pytest.raises(ValueError):
        fitted.fit([training[0], bad])
    assert fitted.predict(training) == before


@pytest.mark.parametrize(
    "kwargs",
    [
        {"smoothing": 0},
        {"smoothing": -1},
        {"smoothing": float("nan")},
        {"smoothing": float("inf")},
        {"smoothing": True},
        {"min_token_count": 0},
        {"min_token_count": True},
        {"min_token_count": 1.5},
        {"label_mode": "D"},
        {"labels": ["O"] * 4},
    ],
)
def test_invalid_hyperparameters_and_vocabularies(kwargs):
    arguments = {"label_mode": "A", "labels": labels(), **kwargs}
    with pytest.raises(ValueError):
        HMMModel(**arguments)


def test_unfitted_operations_fail_clearly(tmp_path):
    model = HMMModel(label_mode="A", labels=labels())
    for operation in [
        lambda: model.decode(["a"]),
        lambda: model.predict([]),
        lambda: model.save_model(tmp_path / "x.json"),
    ]:
        with pytest.raises(RuntimeError, match="not fitted"):
            operation()


def test_save_load_preserves_metadata_and_refuses_overwrite(training, tmp_path):
    metadata = {"release": "unit-fixture", "versions": {"tokenizer": "test-only"}}
    model = HMMModel(label_mode="A", labels=labels()).fit(training, metadata=metadata)
    metadata["versions"]["tokenizer"] = "changed"
    detached = model.metadata
    detached["versions"]["tokenizer"] = "changed-again"
    path = tmp_path / "nested" / "model.json"
    model.save_model(path)
    restored = HMMModel.load_model(path)
    assert restored.metadata["versions"]["tokenizer"] == "test-only"
    assert restored.training_summary == model.training_summary
    assert restored.predict(training) == model.predict(training)
    with pytest.raises(FileExistsError):
        model.save_model(path)


@pytest.mark.parametrize(
    "kind", ["version", "transition", "emission", "start", "unknown", "metadata"]
)
def test_corrupt_artifacts_fail_validation(kind, fitted, tmp_path):
    path = tmp_path / "model.json"
    fitted.save_model(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if kind == "version":
        payload["format_version"] = 999
    elif kind == "transition":
        payload["counts"]["transition"][0][0] += 1
    elif kind == "emission":
        payload["counts"]["emissions"][0]["a"] = -1
    elif kind == "start":
        payload["counts"]["start"][0] = True
    elif kind == "unknown":
        payload["counts"]["unknown"] = []
    else:
        payload["metadata"] = {"invalid": float("nan")}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid HMM artifact"):
        HMMModel.load_model(path)
