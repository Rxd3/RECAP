"""Main's (72ebfd1) call signatures of two HMM workflow functions still work (restructure spec section 2).

`verify_training_release` and `predict_run` take main's separate parameters and build the dataclass of the new
`verify_release_inputs` and `predict_run_with`; an old call returns exactly what the new function returns.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from test_hmm_cli import jsonl
from test_hmm_workflow import release_args

from punct_restore.models.hmm import HMMModel
from punct_restore.models.hmm_workflow import (
    PredictionOptions,
    ReleaseInputs,
    predict_run,
    predict_run_with,
    verify_release_inputs,
    verify_training_release,
)

POSITIONAL, KEYWORD = "POSITIONAL_OR_KEYWORD", "KEYWORD_ONLY"
VOLATILE = ("started_at_utc", "model_load_seconds", "prediction_seconds_including_input")


def _shape(function: Callable[..., Any]) -> list[tuple[str, str, bool]]:
    return [(p.name, p.kind.name, p.default is not p.empty) for p in inspect.signature(function).parameters.values()]


def test_predict_run_has_the_parameters_of_main() -> None:
    assert _shape(predict_run) == [
        ("model_path", POSITIONAL, False),
        ("input_path", POSITIONAL, False),
        ("output_dir", POSITIONAL, False),
        ("ids_path", KEYWORD, True),
        ("release_manifest", KEYWORD, True),
        ("selection_record", KEYWORD, True),
        ("command", KEYWORD, True),
    ]


def test_verify_training_release_has_the_parameters_of_main() -> None:
    assert _shape(verify_training_release) == [
        ("manifest_path", POSITIONAL, False),
        ("train", KEYWORD, False),
        ("train_ids", KEYWORD, False),
        ("split_manifest", KEYWORD, False),
        ("label_config", KEYWORD, False),
        ("mode", KEYWORD, False),
        ("release_id", KEYWORD, False),
        ("policy_version", KEYWORD, False),
        ("prediction", KEYWORD, True),
        ("prediction_ids", KEYWORD, True),
    ]


def test_the_dataclass_versions_take_the_dataclass() -> None:
    assert list(inspect.signature(verify_release_inputs).parameters) == ["manifest_path", "inputs"]
    assert list(inspect.signature(predict_run_with).parameters) == ["model_path", "input_path", "output_dir", "options"]


@pytest.fixture
def artifact(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A saved tiny model, an input file with two sequences and the ID list that selects the second one."""
    model = HMMModel(label_mode="A", labels=["O", "COMMA", "PERIOD", "QUESTION"], min_token_count=1)
    train = {
        "schema_version": 1, "sample_id": "t1", "source": "unit", "split": "train", "label_mode": "A",
        "tokens": ["hello", "nasa"], "punct_labels": ["O", "PERIOD"], "case_labels": ["FIRST_UPPER", "ALL_UPPER"],
    }
    model.fit([train])
    model.save_model(tmp_path / "model.json")
    rows = [{"schema_version": 1, "sample_id": f"p{n}", "tokens": ["hello", "new"]} for n in (1, 2)]
    jsonl(tmp_path / "input.jsonl", rows)
    (tmp_path / "pick.ids").write_text("p2\n", encoding="utf-8")
    return tmp_path / "model.json", tmp_path / "input.jsonl", tmp_path / "pick.ids"


def _stable(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key not in VOLATILE}


def test_an_old_positional_predict_run_equals_predict_run_with(
    artifact: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    model, data, _ = artifact
    old = predict_run(model, data, tmp_path / "old")
    new = predict_run_with(model, data, tmp_path / "new", PredictionOptions())
    assert _stable(old) == _stable(new) and old["num_sequences"] == 2
    assert (tmp_path / "old/predictions.jsonl").read_bytes() == (tmp_path / "new/predictions.jsonl").read_bytes()


def test_an_old_keyword_predict_run_equals_predict_run_with(artifact: tuple[Path, Path, Path], tmp_path: Path) -> None:
    model, data, ids = artifact
    command = ["python", "scripts/predict_hmm.py"]
    old = predict_run(model, data, tmp_path / "old", ids_path=ids, command=command)
    new = predict_run_with(model, data, tmp_path / "new", PredictionOptions(ids_path=ids, command=command))
    assert _stable(old) == _stable(new)
    assert old["num_sequences"] == 1 and old["command"] == command and old["inputs"]["ids"]["sha256"]
    assert json.loads((tmp_path / "old/predictions.jsonl").read_text())["sample_id"] == "p2"


@pytest.mark.parametrize("keyword", ["release_manifest", "selection_record"])
def test_an_old_predict_run_passes_the_release_and_selection_files_on(
    artifact: tuple[Path, Path, Path], tmp_path: Path, keyword: str
) -> None:
    model, data, _ = artifact
    (tmp_path / "other.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        predict_run(model, data, tmp_path / "out", **{keyword: tmp_path / "other.json"})
    assert not (tmp_path / "out").exists()


def test_an_old_keyword_verify_training_release_equals_verify_release_inputs(tmp_path: Path) -> None:
    _, processed, splits, _, opts = release_args(tmp_path)
    manifest = processed / "release_manifest.json"
    old = verify_training_release(manifest, **opts)
    new = verify_release_inputs(manifest, ReleaseInputs(**opts))
    assert old == new and old["release_id"] == "fixture"


def test_an_old_verify_training_release_checks_the_prediction_files_too(tmp_path: Path) -> None:
    _, processed, splits, _, opts = release_args(tmp_path)
    manifest = processed / "release_manifest.json"
    checked = {"prediction": processed / "mode_A/validation.jsonl",
               "prediction_ids": splits / "primary_eval/validation_mode_A.ids"}
    assert verify_training_release(manifest, **opts, **checked) == verify_release_inputs(
        manifest, ReleaseInputs(**opts, **checked)
    )
    wrong = {"prediction": processed / "mode_C/validation.jsonl"}
    with pytest.raises(ValueError, match="mode does not match"):
        verify_training_release(manifest, **opts, **wrong)


def test_an_old_verify_training_release_refuses_what_verify_release_inputs_refuses(tmp_path: Path) -> None:
    _, processed, _, _, opts = release_args(tmp_path)
    opts["release_id"] = "some-other-release"
    with pytest.raises(ValueError, match="release identity") as old:
        verify_training_release(processed / "release_manifest.json", **opts)
    with pytest.raises(ValueError) as new:
        verify_release_inputs(processed / "release_manifest.json", ReleaseInputs(**opts))
    assert str(old.value) == str(new.value)
