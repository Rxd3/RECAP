"""Member 2's HMM tests: synthetic fixtures only, never research results."""

import gc
import json
import math
from copy import deepcopy
from itertools import product
from pathlib import Path

import pytest
import yaml

from punct_restore.models.hmm import HMMModel, _observation

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


def test_the_observation_check_returns_id_and_tokens_and_leaves_the_seen_set_alone():
    seen = {"earlier"}
    assert _observation(record("later", ["a", "b"], ["O", "PERIOD"]), seen) == ("later", ["a", "b"])
    assert seen == {"earlier"}
    with pytest.raises(ValueError, match="duplicate sample_id: earlier"):
        _observation(record("earlier", ["a"], ["O"]), seen)


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


def _saved(tmp_path, content):
    path = tmp_path / "deep" / "model.json"
    path.parent.mkdir()
    path.write_bytes(content)
    return path


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"[" * 100_000, "model.json: not valid JSON"),  # nested far deeper than the JSON reader allows
        (b"[1]", "model.json: invalid HMM artifact: not a JSON object"),
        (b"null", "model.json: invalid HMM artifact: not a JSON object"),
        (b"{}", "model.json: invalid HMM artifact: missing format_version"),
        (b'{"format_version": 1}', "model.json: invalid HMM artifact: missing model_name"),
        (b"{oops", "model.json: not valid JSON"),
        (b"\xff\xfe", "model.json: not valid UTF-8"),
    ],
    ids=["nested", "array", "null", "empty-object", "missing-key", "bad-json", "not-utf8"],
)
def test_an_unusable_artifact_is_one_value_error_that_starts_with_the_file_name(tmp_path, content, message):
    path = _saved(tmp_path, content)
    with pytest.raises(ValueError) as raised:
        HMMModel.load_model(path)
    text = str(raised.value)
    assert text == message
    assert str(tmp_path) not in text


PYTHON_TEXT = ("subscriptable", "unhashable", "indices", "NoneType", "KeyError", "TypeError", "__init__", "argument",
               "positional", "keyword", "must be a mapping", "not iterable", "has no attribute", "punct_restore.")
MISSING = object()


def _artifact_damaged(fitted, tmp_path, keys, new):
    """The saved artifact of `fitted` with the value at `keys` replaced (MISSING: removed); returns its path."""
    path = tmp_path / "model.json"
    fitted.save_model(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not keys:
        payload = new
    else:
        holder = payload
        for key in keys[:-1]:
            holder = holder[key]
        if new is MISSING:
            del holder[keys[-1]]
        else:
            holder[keys[-1]] = new
    damaged = tmp_path / "damaged" / "model.json"
    damaged.parent.mkdir()
    damaged.write_text(json.dumps(payload), encoding="utf-8")
    return damaged


COUNT_KEYS = ("start", "transition", "end", "state", "emissions", "unknown", "vocabulary", "num_sequences")
ARTIFACT_PROBLEMS = [  # (the place in the artifact, its new value, the one reason load_model gives)
    ((), None, "not a JSON object"),
    ((), [], "not a JSON object"),
    ((), {}, "missing format_version"),
    (("model_name",), MISSING, "missing model_name"),
    (("config",), MISSING, "missing config"),
    *[(("config",), value, "config is not a JSON object") for value in (None, [], 5, "text")],
    (("config",), {}, "missing config.label_mode"),
    (("config", "label_mode"), MISSING, "missing config.label_mode"),
    (("config", "labels"), MISSING, "missing config.labels"),
    (("config", "extra"), 1, "unknown key config.extra"),
    (("counts",), MISSING, "missing counts"),
    *[(("counts",), value, "counts is not a JSON object") for value in (None, [], 5, "text")],
    (("counts",), {}, "missing counts.start"),
    *[(("counts", key), MISSING, f"missing counts.{key}") for key in COUNT_KEYS],
    (("counts", "extra"), 1, "unknown key counts.extra"),
    (("metadata",), MISSING, "missing metadata"),
]


@pytest.mark.parametrize(
    ("keys", "new", "reason"), ARTIFACT_PROBLEMS,
    ids=[f"{'.'.join(keys) or 'root'}-{'missing' if new is MISSING else type(new).__name__}-{n}"
         for n, (keys, new, _) in enumerate(ARTIFACT_PROBLEMS)],
)
def test_an_artifact_that_lacks_a_key_or_holds_the_wrong_kind_is_refused_naming_the_key(
        fitted, tmp_path, keys, new, reason):
    """The message is the artifact's own wording, not Python's (`'config'`, `__init__() missing 1 required...`)."""
    path = _artifact_damaged(fitted, tmp_path, keys, new)
    with pytest.raises(ValueError) as raised:
        HMMModel.load_model(path)
    assert str(raised.value) == f"model.json: invalid HMM artifact: {reason}"


def _places(value, prefix=()):
    """Every place in a parsed artifact: dict keys, and the first two items of a list."""
    yield prefix
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _places(item, (*prefix, key))
    elif isinstance(value, list):
        for index, item in enumerate(value[:2]):
            yield from _places(item, (*prefix, index))


@pytest.mark.parametrize("new", [MISSING, None, [], {}, "x", 5, [{}], [["O"]], [1], [None]],
                         ids=["missing", "null", "list", "object", "text", "number", "items-object", "items-list",
                              "items-number", "items-null"])
def test_no_damage_to_an_artifact_shows_python_text_or_escapes_as_another_error(fitted, tmp_path, new):
    """Every place of a saved artifact, damaged in every way: a ValueError of one line, or a model that loads."""
    saved = tmp_path / "saved.json"
    fitted.save_model(saved)
    for keys in _places(json.loads(saved.read_text(encoding="utf-8"))):
        if not keys or (new is MISSING and isinstance(keys[-1], int)):
            continue
        folder = tmp_path / ("-".join(map(str, keys)) or "root")
        folder.mkdir()
        try:
            HMMModel.load_model(_artifact_damaged(fitted, folder, keys, new))
        except ValueError as raised:
            text = str(raised)
            assert "\n" not in text and not [bit for bit in PYTHON_TEXT if bit in text], (keys, text)


def test_a_save_that_fails_while_writing_leaves_no_half_written_file(training, tmp_path):
    """A lone surrogate passes the metadata check but cannot be written as UTF-8: the file it started is removed."""
    model = HMMModel(label_mode="A", labels=labels()).fit(training, metadata={"note": "\ud800"})
    path = tmp_path / "model.json"
    with pytest.raises(UnicodeEncodeError):
        model.save_model(path)
    assert not path.exists()


def test_the_saved_bytes_are_sorted_json_with_one_final_line_feed(fitted, tmp_path):
    path = tmp_path / "model.json"
    fitted.save_model(path)
    raw = path.read_bytes()
    expected = json.dumps(json.loads(raw), ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
    assert raw == expected.encode("utf-8") and b"\r" not in raw


def test_distributions_are_the_log_tables_of_the_fixture_by_hand(fitted):
    # smoothing 1, K=4: states O/COMMA/PERIOD/QUESTION hold 2/1/2/1 tokens, the vocabulary is {a, b, c}.
    tables = fitted.distributions
    assert tables.log_start == pytest.approx([math.log(n / 7) for n in (3, 1, 1, 2)])
    assert tables.log_transition[0] == pytest.approx([math.log(n / 7) for n in (1, 2, 2, 1)])
    assert tables.log_transition[1] == pytest.approx([math.log(n / 6) for n in (1, 1, 2, 1)])
    assert tables.log_end == pytest.approx([math.log(n / d) for n, d in ((1, 7), (1, 6), (3, 7), (2, 6))])
    assert tables.vocabulary == {"a", "b", "c"}
    assert dict(tables.log_emissions[0]) == pytest.approx({"a": math.log(3 / 6)})
    assert tables.log_default == pytest.approx([math.log(1 / n) for n in (6, 5, 6, 5)])
    assert tables.log_unknown == pytest.approx(tables.log_default)  # nothing was folded into the unknown bucket


def test_the_distributions_reproduce_sequence_log_probability(fitted):
    tables, index = fitted.distributions, {label: n for n, label in enumerate(fitted.labels)}
    words, states = ["a", "a", "new"], ["O", "COMMA", "PERIOD"]
    score = tables.log_start[index[states[0]]] + tables.log_end[index[states[-1]]]
    for position, (word, state) in enumerate(zip(words, states, strict=True)):
        table = tables.log_emissions[index[state]]
        score += table.get(word, tables.log_default[index[state]]) if word in tables.vocabulary else (
            tables.log_unknown[index[state]])
        if position:
            score += tables.log_transition[index[states[position - 1]]][index[state]]
    assert score == pytest.approx(fitted.sequence_log_probability(words, states))


def test_the_distributions_of_an_unfitted_model_are_refused():
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = HMMModel(label_mode="A", labels=labels()).distributions


def test_the_distributions_cannot_be_changed_and_hold_no_live_table_of_the_model(fitted):
    words, states = ["a", "b"], ["O", "PERIOD"]
    before, tables = fitted.sequence_log_probability(words, states), fitted.distributions
    with pytest.raises(AttributeError):
        tables.log_start = ()
    with pytest.raises(TypeError):
        tables.log_start[0] = 0.0
    with pytest.raises(TypeError):
        tables.log_transition[0][0] = 0.0
    with pytest.raises(TypeError):
        tables.log_emissions[0]["a"] = 0.0
    with pytest.raises(AttributeError):
        tables.vocabulary.add("x")
    # Even the dict under a read-only emission table is a copy: writing to it does not reach the model.
    gc.get_referents(tables.log_emissions[0])[0]["a"] = 0.0
    assert fitted.sequence_log_probability(words, states) == before
    assert fitted.distributions.log_emissions[0]["a"] == pytest.approx(math.log(3 / 6))
