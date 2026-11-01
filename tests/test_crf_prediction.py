"""models/crf_prediction: the steps behind `recap train-crf predict`; a user error is PredictionFailed."""

from __future__ import annotations

import errno
import json
from pathlib import Path
from typing import Any

import pytest

from punct_restore.models import crf_prediction
from punct_restore.models.crf import CRFModel
from punct_restore.models.crf_prediction import (
    PredictionFailed,
    check_predict_paths,
    inference_summary,
    load_scoring_model,
    predict_records,
    read_input_records,
    write_model_predictions,
)

RECORDS = [
    {"sample_id": "t1", "tokens": ["hello", "everyone"], "punct_labels": ["O", "PERIOD"]},
    {"sample_id": "t2", "tokens": ["yes", "we", "can"], "punct_labels": ["COMMA", "O", "PERIOD"]},
]


def _raising(error: BaseException) -> Any:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise error

    return fail


class _Paths:
    """A model file and an input file that exist, and an output and a manifest that do not."""

    def __init__(self, folder: Path) -> None:
        self.model, self.input = folder / "m.joblib", folder / "in.jsonl"
        self.output, self.manifest = folder / "out.jsonl", folder / "out.man.json"
        self.model.write_bytes(b"model")
        self.input.write_text('{"sample_id": "s", "tokens": ["a"]}\n', encoding="utf-8")

    def check(self, *, overwrite: bool = False, manifest: bool = True, ids: Path | None = None) -> None:
        check_predict_paths(self.model, self.input, self.output, self.manifest if manifest else None, ids, overwrite)


@pytest.fixture
def paths(tmp_path: Path) -> _Paths:
    return _Paths(tmp_path)


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "fitted.joblib"
    CRFModel(max_iterations=5).fit(RECORDS).save_model(path)
    return path


# --- check_predict_paths -----------------------------------------------------------------------------------


def test_free_output_paths_and_existing_inputs_pass(paths: _Paths) -> None:
    assert paths.check() is None
    assert paths.check(manifest=False) is None


def test_a_missing_model_or_input_is_named_by_file_name_only(paths: _Paths) -> None:
    paths.model.unlink()
    with pytest.raises(PredictionFailed) as caught:
        paths.check()
    assert str(caught.value) == "m.joblib: model artifact not found"

    paths.model.write_bytes(b"model")
    paths.input.unlink()
    with pytest.raises(PredictionFailed) as caught:
        paths.check()
    assert str(caught.value) == "in.jsonl: input JSONL not found"


def test_an_existing_output_or_manifest_needs_overwrite(paths: _Paths) -> None:
    paths.output.write_text("keep", encoding="utf-8")
    with pytest.raises(PredictionFailed) as caught:
        paths.check()
    assert str(caught.value) == "out.jsonl: output file already exists; use --overwrite to replace"
    assert paths.check(overwrite=True) is None

    paths.output.unlink()
    paths.manifest.write_text("keep", encoding="utf-8")
    with pytest.raises(PredictionFailed) as caught:
        paths.check()
    assert str(caught.value) == "out.man.json: manifest already exists; use --overwrite to replace"
    assert paths.check(overwrite=True) is None
    assert paths.check(manifest=False) is None  # no manifest requested, so the existing file is not in the way


def test_an_output_that_is_also_an_input_is_refused(paths: _Paths) -> None:
    with pytest.raises(PredictionFailed, match="^--output"):
        check_predict_paths(paths.model, paths.input, paths.input, None, None, True)


def test_a_path_that_cannot_be_checked_is_one_line_naming_it_by_name(
        paths: _Paths, monkeypatch: pytest.MonkeyPatch) -> None:
    locked = paths.model.parent / "deep" / "locked"
    monkeypatch.setattr(crf_prediction, "check_new_output_paths",
                        _raising(OSError(errno.EACCES, "Permission denied", str(locked))))

    with pytest.raises(PredictionFailed) as caught:
        paths.check()

    assert str(caught.value) == "locked: Permission denied"


# --- load_scoring_model ------------------------------------------------------------------------------------


def test_a_fitted_artifact_loads_as_a_model(model_file: Path) -> None:
    assert load_scoring_model(model_file).label_mode == "A"


def test_a_corrupt_artifact_is_one_line_naming_the_file_only(tmp_path: Path) -> None:
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"this is not a joblib file")

    with pytest.raises(PredictionFailed) as caught:
        load_scoring_model(bad)

    message = str(caught.value)
    assert message == "bad.joblib: not a readable CRF artifact"
    assert "\n" not in message and str(tmp_path) not in message


def test_a_casing_helper_artifact_is_not_a_scoring_model(tmp_path: Path) -> None:
    records = [{**record, "label_mode": "C", "case_labels": ["LOWER"] * len(record["tokens"])} for record in RECORDS]
    helper = tmp_path / "case.joblib"
    CRFModel(label_mode="CASE", max_iterations=5).fit(records).save_model(helper)

    with pytest.raises(PredictionFailed, match="^case.joblib: .*CASE.*scoring mode"):
        load_scoring_model(helper)


def test_a_bug_while_loading_is_not_reported_as_a_user_error(
        model_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CRFModel, "load_model", classmethod(_raising(KeyError("backend_state"))))

    with pytest.raises(KeyError):
        load_scoring_model(model_file)


# --- read_input_records ------------------------------------------------------------------------------------


def test_the_records_come_back_gold_free_and_restricted_to_the_id_list(tmp_path: Path) -> None:
    source, ids = tmp_path / "eval.jsonl", tmp_path / "wanted.ids"
    rows = [{"sample_id": f"e{i}", "tokens": ["a", "b"], "punct_labels": ["O", "O"]} for i in range(3)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    ids.write_text("e2\ne0\n", encoding="utf-8")

    records = read_input_records(source, ids)

    assert records == [{"sample_id": "e0", "tokens": ["a", "b"]}, {"sample_id": "e2", "tokens": ["a", "b"]}]
    assert len(read_input_records(source, None)) == 3


def test_an_id_missing_from_the_input_is_one_line_naming_the_file_only(tmp_path: Path) -> None:
    source, ids = tmp_path / "eval.jsonl", tmp_path / "wanted.ids"
    source.write_text('{"sample_id": "e0", "tokens": ["a"]}\n', encoding="utf-8")
    ids.write_text("e0\nnope\n", encoding="utf-8")

    with pytest.raises(PredictionFailed) as caught:
        read_input_records(source, ids)

    assert str(caught.value).startswith("Missing 1 selected IDs from input file eval.jsonl")
    assert str(tmp_path) not in str(caught.value)


# --- predict_records ---------------------------------------------------------------------------------------


def test_the_predictions_are_the_models_own(model_file: Path) -> None:
    model = load_scoring_model(model_file)
    observations = [{"sample_id": "p1", "tokens": ["we", "can"]}]

    assert predict_records(model, observations) == model.predict(observations)


def test_a_label_sequence_the_contract_rejects_is_a_prediction_failure(
        model_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = load_scoring_model(model_file)
    monkeypatch.setattr(CRFModel, "predict", _raising(ValueError("Model predicted 1 labels for sequence e1")))

    with pytest.raises(PredictionFailed) as caught:
        predict_records(model, [])

    assert str(caught.value) == "prediction failed: Model predicted 1 labels for sequence e1"


def test_a_bug_while_predicting_is_not_reported_as_a_user_error(
        model_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = load_scoring_model(model_file)
    monkeypatch.setattr(CRFModel, "predict", _raising(TypeError("tokens must be a Sequence")))

    with pytest.raises(TypeError):
        predict_records(model, [])


# --- inference_summary -------------------------------------------------------------------------------------


def test_the_summary_counts_sequences_and_tokens_and_gives_the_rate() -> None:
    predictions = [{"predicted_labels": ["O", "O"]}, {"predicted_labels": ["O"]}]

    assert inference_summary(predictions, 0.5) == "Inferred 2 sequences (3 tokens) in 0.50s (6.0 tok/s)."


def test_the_summary_groups_thousands_and_survives_a_zero_duration() -> None:
    predictions = [{"predicted_labels": ["O"] * 5} for _ in range(1200)]

    assert inference_summary(predictions, 0.0) == "Inferred 1,200 sequences (6,000 tokens) in 0.00s (0.0 tok/s)."


# --- write_model_predictions -------------------------------------------------------------------------------


def test_predictions_and_manifest_are_written_with_the_model_labels(tmp_path: Path) -> None:
    model = CRFModel(max_iterations=5).fit(RECORDS)
    observations = [{"sample_id": "p1", "tokens": ["we", "can"]}]
    source, out, manifest = tmp_path / "in.jsonl", tmp_path / "p.jsonl", tmp_path / "p.man.json"
    source.write_text(json.dumps(observations[0]) + "\n", encoding="utf-8")

    write_model_predictions(model, model.predict(observations), source, out, manifest)

    assert [json.loads(line)["sample_id"] for line in out.read_text(encoding="utf-8").splitlines()] == ["p1"]
    written = json.loads(manifest.read_text(encoding="utf-8"))
    assert (written["model_name"], written["label_mode"]) == (model.model_name, "A")
    assert written["ordered_labels"] == model.vocabulary


def test_without_a_manifest_path_only_the_predictions_are_written(tmp_path: Path) -> None:
    model = CRFModel(max_iterations=5).fit(RECORDS)
    source, out = tmp_path / "in.jsonl", tmp_path / "p.jsonl"
    source.write_text('{"sample_id": "p1", "tokens": ["we"]}\n', encoding="utf-8")

    write_model_predictions(model, model.predict([{"sample_id": "p1", "tokens": ["we"]}]), source, out, None)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["in.jsonl", "p.jsonl"]


def test_a_full_disk_while_writing_is_one_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = CRFModel(max_iterations=5).fit(RECORDS)
    monkeypatch.setattr(crf_prediction, "write_prediction_outputs",
                        _raising(OSError(errno.ENOSPC, "No space left on device", "p.jsonl")))

    with pytest.raises(PredictionFailed) as caught:
        write_model_predictions(model, [], tmp_path / "in.jsonl", tmp_path / "p.jsonl", None)

    assert str(caught.value) == "could not write prediction outputs: p.jsonl: No space left on device"


def test_the_library_prints_nothing(paths: _Paths, model_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths.check()
    predict_records(load_scoring_model(model_file), [{"sample_id": "p1", "tokens": ["we"]}])
    inference_summary([{"predicted_labels": ["O"]}], 1.0)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
