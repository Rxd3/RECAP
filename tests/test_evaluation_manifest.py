"""Prediction-manifest check: matching, mismatching and malformed predictions.manifest.json files."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from eval_fixtures import BOM, Case, valid_case

from punct_restore.evaluation import EvaluationInputError, check_manifest, load_bundle
from punct_restore.evaluation.inputs import Bundle, Vocab, sha256_file

MANIFEST_NAME = "predictions.manifest.json"
PREFIX = f"{MANIFEST_NAME}: "
REQUIRED_FIELDS = ("model_name", "label_mode", "ordered_labels", "predictions_sha256", "input")
NOT_JSON = f"{PREFIX}not a JSON manifest"
NOT_OBJECT = f"{PREFIX}manifest must be a JSON object"
BAD_INPUT = f"{PREFIX}input must be an object with sha256"
DIFFERENT_INPUT = f"{PREFIX}predictions were made from a different input file than the gold file"

Change = Callable[[Case, Vocab], Any]
Render = Callable[[dict[str, Any]], bytes]


def valid_manifest(case: Case, vocab: Vocab, mode: str) -> dict[str, Any]:
    """A manifest that matches the case's predictions and gold (extra fields are allowed)."""
    return {"model_name": "m", "label_mode": mode, "ordered_labels": list(vocab.labels(mode)),
            "predictions_sha256": sha256_file(case.predictions),
            "input": {"path": f"mode_{mode}/validation.jsonl", "sha256": sha256_file(case.gold)},
            "runtime": {"fit_seconds": 1.5}}


def _bundle(case: Case, vocab: Vocab) -> Bundle:
    return load_bundle(case.gold, case.ids, case.predictions, case.masks, vocab)


def _manifest_path(case: Case) -> Path:
    return case.predictions.parent / MANIFEST_NAME


def _write_manifest(case: Case, manifest: dict[str, Any]) -> Path:
    path = _manifest_path(case)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _exact(message: str) -> str:
    """A match= pattern that accepts the whole message and nothing else."""
    return rf"\A{re.escape(message)}\Z"


@pytest.fixture
def case_a(tmp_path: Path) -> Case:
    return valid_case(tmp_path, "A")


@pytest.mark.parametrize("mode", ["A", "C"])
def test_valid_manifest_passes(mode: str, tmp_path: Path, vocab: Vocab) -> None:
    case = valid_case(tmp_path, mode)
    path = _write_manifest(case, valid_manifest(case, vocab, mode))
    assert check_manifest(path, _bundle(case, vocab), vocab) is None


@pytest.mark.parametrize("mode", ["A", "C"])
def test_valid_manifest_with_bom_passes(mode: str, tmp_path: Path, vocab: Vocab) -> None:
    case = valid_case(tmp_path, mode)
    path = _manifest_path(case)
    path.write_bytes(BOM + json.dumps(valid_manifest(case, vocab, mode)).encode("utf-8"))
    assert check_manifest(path, _bundle(case, vocab), vocab) is None


@pytest.mark.parametrize(("field", "changed"), [
    pytest.param("model_name", lambda case, vocab: "other", id="model_name"),
    pytest.param("label_mode", lambda case, vocab: "B", id="label_mode"),
    pytest.param("ordered_labels", lambda case, vocab: list(reversed(vocab.A)), id="labels_reversed"),
    pytest.param("ordered_labels", lambda case, vocab: list(vocab.A)[:-1], id="labels_missing_last"),
    pytest.param("ordered_labels", lambda case, vocab: list(vocab.B), id="labels_of_mode_B"),
    pytest.param("predictions_sha256", lambda case, vocab: sha256_file(case.gold), id="sha_of_gold"),
    pytest.param("predictions_sha256", lambda case, vocab: "0" * 64, id="sha_all_zeros"),
])
def test_manifest_mismatch_raises(field: str, changed: Change, case_a: Case, vocab: Vocab) -> None:
    manifest = {**valid_manifest(case_a, vocab, "A"), field: changed(case_a, vocab)}
    path = _write_manifest(case_a, manifest)
    with pytest.raises(EvaluationInputError) as error:
        check_manifest(path, _bundle(case_a, vocab), vocab)
    assert str(error.value).startswith(f"{PREFIX}{field} is ")


def test_manifest_mismatch_message_names_got_and_expected(case_a: Case, vocab: Vocab) -> None:
    path = _write_manifest(case_a, {**valid_manifest(case_a, vocab, "A"), "label_mode": "B"})
    with pytest.raises(EvaluationInputError, match=_exact(f"{PREFIX}label_mode is 'B', expected 'A'")):
        check_manifest(path, _bundle(case_a, vocab), vocab)


def test_manifest_from_a_different_input_file_raises(case_a: Case, vocab: Vocab) -> None:
    manifest = valid_manifest(case_a, vocab, "A")
    manifest["input"] = {**manifest["input"], "sha256": sha256_file(case_a.predictions)}
    path = _write_manifest(case_a, manifest)
    with pytest.raises(EvaluationInputError, match=_exact(DIFFERENT_INPUT)):
        check_manifest(path, _bundle(case_a, vocab), vocab)


def _json_bytes(manifest: dict[str, Any]) -> bytes:
    return json.dumps(manifest).encode("utf-8")


def _fixed(raw: bytes) -> Render:
    return lambda manifest: raw


def _without(field: str) -> Render:
    return lambda manifest: _json_bytes({key: value for key, value in manifest.items() if key != field})


def _with_input(value: Any) -> Render:
    return lambda manifest: _json_bytes({**manifest, "input": value})


@pytest.mark.parametrize(("render", "message"), [
    pytest.param(_fixed(b"{oops"), NOT_JSON, id="invalid_json"),
    pytest.param(_fixed(b'{"model_name": "\xe9\xff"}'), NOT_JSON, id="invalid_utf8"),
    pytest.param(_fixed(b"[]"), NOT_OBJECT, id="list"),
    pytest.param(_fixed(b'"x"'), NOT_OBJECT, id="string"),
    *[pytest.param(_without(field), f"{PREFIX}missing field(s) {field}", id=f"missing_{field}")
      for field in REQUIRED_FIELDS],
    pytest.param(_with_input("abc"), BAD_INPUT, id="input_is_string"),
    pytest.param(_with_input({}), BAD_INPUT, id="input_is_empty_object"),
    pytest.param(_with_input({"path": "x"}), BAD_INPUT, id="input_without_sha256"),
])
def test_malformed_manifest_raises(render: Render, message: str, case_a: Case, vocab: Vocab) -> None:
    path = _manifest_path(case_a)
    path.write_bytes(render(valid_manifest(case_a, vocab, "A")))
    with pytest.raises(EvaluationInputError, match=_exact(message)):
        check_manifest(path, _bundle(case_a, vocab), vocab)


@pytest.mark.parametrize("name", ["m\nx", "m\r\nx"])
def test_mismatch_message_is_one_line(name: str, case_a: Case, vocab: Vocab) -> None:
    path = _write_manifest(case_a, {**valid_manifest(case_a, vocab, "A"), "model_name": name})
    with pytest.raises(EvaluationInputError) as error:
        check_manifest(path, _bundle(case_a, vocab), vocab)
    message = str(error.value)
    assert message.startswith(f"{PREFIX}model_name is ")
    assert "\n" not in message and "\r" not in message


def test_missing_manifest_file_propagates_oserror(tmp_path: Path, case_a: Case, vocab: Vocab) -> None:
    with pytest.raises(FileNotFoundError):
        check_manifest(tmp_path / "nope.json", _bundle(case_a, vocab), vocab)
