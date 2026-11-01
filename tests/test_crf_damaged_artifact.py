"""A CRF joblib whose fields are of the wrong kind is one `error: <file>: ...` line in every command that reads it.

The files are real joblib files: a model that was saved by the code, with one field damaged. Each of the four damages
is run through the five commands that open a CRF artifact, in-process through the command's `main`. A real bug in
our own code is not covered here: it must still propagate as a traceback.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import pytest
from test_threshold import Fixture, _build_fixture

from punct_restore.cli import run_demo, train_crf, tune_threshold
from punct_restore.models.crf import CRFModel
from punct_restore.reporting import time_models

NAME = "damaged.joblib"
PREFIX = f"{NAME}: corrupt artifact: "


@pytest.fixture(scope="module")
def good(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    return _build_fixture(tmp_path_factory.mktemp("damaged_source"), "A")


@pytest.fixture(scope="module")
def bundle(good: Fixture) -> dict[str, Any]:
    """The bundle of a model this code saved a moment ago (joblib is pickle: the file is our own)."""
    loaded: dict[str, Any] = joblib.load(good.model)
    return loaded


def _without(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    return {k: v for k, v in mapping.items() if k != key}


# --- the four damages of the five commands ------------------------------------------------------------------

DAMAGES: dict[str, tuple[Callable[[dict[str, Any]], dict[str, Any]], str]] = {
    "wrong-type-field": (lambda b: {**b, "feature_config": 5}, "feature_config is not a JSON object"),
    "missing-nested-field": (
        lambda b: {**b, "hyperparameters": _without(b["hyperparameters"], "c1")}, "missing hyperparameters.c1"),
    "backend-wrong-type": (lambda b: {**b, "backend_state": b"x"}, "backend_state is not a CRF model"),
    "vocabulary-not-strings": (lambda b: {**b, "vocabulary": [1, 2, 3, 4]}, "vocabulary.0 is not nonempty text"),
}


def _predict(fx: Fixture, model: Path, tmp_path: Path) -> int:
    source = tmp_path / "eval.jsonl"
    source.write_text(json.dumps({"sample_id": "e1", "tokens": ["hello", "world"]}) + "\n", encoding="utf-8")
    return train_crf.main(["predict", "-m", str(model), "-i", str(source), "-o", str(tmp_path / "preds.jsonl")])


def _inspect(fx: Fixture, model: Path, tmp_path: Path) -> int:
    return train_crf.main(["inspect", "-m", str(model)])


def _demo(fx: Fixture, model: Path, tmp_path: Path) -> int:
    return run_demo.main(["--model-path", str(model), "--text", "hello world"])


def _tune(fx: Fixture, model: Path, tmp_path: Path) -> int:
    return tune_threshold.main(["tune", "--model", str(model), "--gold", str(fx.gold), "--ids", str(fx.ids),
                                "--grid", "0.1:0.9:0.4", "--out", str(tmp_path / "tau.json")])


def _time(fx: Fixture, model: Path, tmp_path: Path) -> int:
    return time_models.main(["--kind", "crf", "--model", str(model), "--mode", "A", "--out", str(tmp_path / "t.json")])


COMMANDS = {"train-crf predict": _predict, "train-crf inspect": _inspect, "run-demo": _demo,
            "tune-threshold tune": _tune, "time-models": _time}


@pytest.mark.parametrize("command", COMMANDS)
@pytest.mark.parametrize("damage", DAMAGES)
def test_a_damaged_artifact_is_one_error_line_in_every_command(
        command: str, damage: str, good: Fixture, bundle: dict[str, Any], tmp_path: Path,
        capsys: pytest.CaptureFixture[str]) -> None:
    change, reason = DAMAGES[damage]
    model = tmp_path / NAME
    joblib.dump(change(bundle), model)

    code = COMMANDS[command](good, model, tmp_path)

    assert code == 1
    assert capsys.readouterr().err == f"error: {PREFIX}{reason}\n"


# --- every field the loader reads has its kind checked --------------------------------------------------------

FIELDS = [
    ("model_name", 5, "model_name is not nonempty text"),
    ("vocabulary", 5, "vocabulary is not a JSON list"),
    ("vocabulary", ["O", "COMMA", None, "QUESTION"], "vocabulary.2 is not nonempty text"),
    ("hyperparameters", [], "hyperparameters is not a JSON object"),
    ("hyperparameters", None, "hyperparameters is not a JSON object"),
    ("feature_config", "basic", "feature_config is not a JSON object"),
    ("training_metadata", [], "training_metadata is not a JSON object"),
    ("environment", None, "environment is not a JSON object"),
    ("backend_state", {}, "backend_state is not a CRF model"),
]


@pytest.mark.parametrize(("field", "value", "reason"), FIELDS)
def test_a_top_level_field_of_the_wrong_kind_is_named(
        field: str, value: object, reason: str, bundle: dict[str, Any], tmp_path: Path) -> None:
    joblib.dump({**bundle, field: value}, tmp_path / NAME)

    with pytest.raises(ValueError) as caught:
        CRFModel.load_model(tmp_path / NAME)

    assert str(caught.value) == PREFIX + reason


@pytest.mark.parametrize("field", ["model_name", "vocabulary", "hyperparameters", "feature_config",
                                   "training_metadata", "environment"])
def test_a_top_level_field_that_is_missing_is_named(
        field: str, bundle: dict[str, Any], tmp_path: Path) -> None:
    joblib.dump(_without(bundle, field), tmp_path / NAME)

    with pytest.raises(ValueError) as caught:
        CRFModel.load_model(tmp_path / NAME)

    assert str(caught.value) == f"{PREFIX}missing {field}"


NESTED = [
    ("hyperparameters", "algorithm", 5, "hyperparameters.algorithm is not nonempty text"),
    ("hyperparameters", "c1", "0.1", "hyperparameters.c1 is not a number"),
    ("hyperparameters", "c2", None, "hyperparameters.c2 is not a number"),
    ("hyperparameters", "max_iterations", "100", "hyperparameters.max_iterations is not a whole number"),
    ("hyperparameters", "all_possible_transitions", "yes",
     "hyperparameters.all_possible_transitions is not true or false"),
    ("feature_config", "preset", 1, "feature_config.preset is not nonempty text"),
    ("feature_config", "window_size", "1", "feature_config.window_size is not a whole number"),
    ("feature_config", "use_shape", "yes", "feature_config.use_shape is not true or false"),
    ("feature_config", "version", None, "feature_config.version is not nonempty text"),
]


@pytest.mark.parametrize(("section", "key", "value", "reason"), NESTED)
def test_a_nested_field_of_the_wrong_kind_is_named(
        section: str, key: str, value: object, reason: str, bundle: dict[str, Any], tmp_path: Path) -> None:
    joblib.dump({**bundle, section: {**bundle[section], key: value}}, tmp_path / NAME)

    with pytest.raises(ValueError) as caught:
        CRFModel.load_model(tmp_path / NAME)

    assert str(caught.value) == PREFIX + reason


@pytest.mark.parametrize(("section", "key"), [
    ("hyperparameters", "algorithm"), ("hyperparameters", "c2"), ("hyperparameters", "max_iterations"),
    ("hyperparameters", "all_possible_transitions"), ("feature_config", "preset"), ("feature_config", "window_size"),
    ("feature_config", "use_affixes"), ("feature_config", "version"),
])
def test_a_nested_field_that_is_missing_is_named(
        section: str, key: str, bundle: dict[str, Any], tmp_path: Path) -> None:
    joblib.dump({**bundle, section: _without(bundle[section], key)}, tmp_path / NAME)

    with pytest.raises(ValueError) as caught:
        CRFModel.load_model(tmp_path / NAME)

    assert str(caught.value) == f"{PREFIX}missing {section}.{key}"


def test_the_team_extension_fields_of_the_feature_config_may_be_absent_but_not_of_the_wrong_kind(
        bundle: dict[str, Any], tmp_path: Path) -> None:
    older = {**bundle, "feature_config": {k: v for k, v in bundle["feature_config"].items()
                                          if k not in {"use_filler_cues", "use_question_lookback",
                                                       "use_word_clusters", "word_clusters"}}}
    joblib.dump(older, tmp_path / "older.joblib")
    assert CRFModel.load_model(tmp_path / "older.joblib").is_fitted
    for key, value, reason in (("use_filler_cues", 1, "use_filler_cues is not true or false"),
                               ("word_clusters", 5, "word_clusters is not a JSON object")):
        joblib.dump({**bundle, "feature_config": {**older["feature_config"], key: value}}, tmp_path / NAME)
        with pytest.raises(ValueError) as caught:
            CRFModel.load_model(tmp_path / NAME)
        assert str(caught.value) == f"{PREFIX}feature_config.{reason}"


def test_a_sound_artifact_still_loads_and_predicts(good: Fixture, bundle: dict[str, Any], tmp_path: Path) -> None:
    joblib.dump(dict(bundle), tmp_path / NAME)
    observations = [{"sample_id": "p1", "tokens": ["we", "can", "go"]}]

    assert CRFModel.load_model(tmp_path / NAME).predict(observations) == CRFModel.load_model(good.model).predict(
        observations)
