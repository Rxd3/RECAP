"""The tau/gamma grid search, the label priors and the tau files, in punct_restore.models.threshold.

They were the middle of cli/tune_threshold.py; the command keeps the argument parsing, the file orchestration and the
printing. The CLI end to end is covered by test_threshold.py, test_threshold_priors.py and
test_tune_threshold_errors.py; these tests call the library directly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from punct_restore.cli import tune_threshold
from punct_restore.hashing import sha256_file
from punct_restore.models import threshold
from punct_restore.models.threshold import (
    GAMMAS,
    Marginals,
    TauSettings,
    decode_predictions,
    pick_tau_gamma,
    read_tau_file,
    score_grid,
    stream_train_labels,
    tau_model_name,
    tau_result,
    train_priors,
)

SURE_COMMA = {"O": 0.2, "COMMA": 0.8}
SURE_O = {"O": 0.9, "COMMA": 0.1}


def test_the_command_module_still_offers_the_names_it_moved_out(tmp_path: Path) -> None:
    assert tune_threshold.pick_tau_gamma is threshold.pick_tau_gamma
    assert tune_threshold.GAMMAS is threshold.GAMMAS
    assert tune_threshold.Marginals is threshold.Marginals


def test_the_gammas_searched_with_priors_are_zero_to_one_in_quarters() -> None:
    assert GAMMAS == (0.0, 0.25, 0.5, 0.75, 1.0)


def test_pick_tau_gamma_takes_the_best_score_then_the_smallest_gamma_then_the_tau_nearest_half() -> None:
    rows = [(0.3, 0.5, 0.9), (0.8, 0.0, 0.9), (0.6, 0.0, 0.9), (0.2, 0.0, 0.1)]
    assert pick_tau_gamma(rows) == (0.6, 0.0)


def test_pick_tau_gamma_refuses_an_empty_grid() -> None:
    with pytest.raises(ValueError, match="empty grid"):
        pick_tau_gamma([])


# --- naming and decoding ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("tau", "gamma", "name"), [(0.5, 0.0, "m_tau0.5"), (0.85, 0.25, "m_tau0.85_gamma0.25")])
def test_tau_model_name_names_the_gamma_only_when_it_is_above_zero(tau: float, gamma: float, name: str) -> None:
    assert tau_model_name("m", tau, gamma) == name


def test_decode_predictions_gives_one_model_contract_record_per_sequence() -> None:
    source = Marginals("m", "A", ["s1", "s2"], [[SURE_O, SURE_COMMA], [SURE_COMMA]])
    assert decode_predictions(source, 0.5) == [
        {"sample_id": "s1", "model_name": "m_tau0.5", "label_mode": "A", "predicted_labels": ["O", "COMMA"]},
        {"sample_id": "s2", "model_name": "m_tau0.5", "label_mode": "A", "predicted_labels": ["COMMA"]},
    ]


def test_decode_predictions_refuses_ids_and_marginals_of_different_lengths() -> None:
    with pytest.raises(ValueError):
        decode_predictions(Marginals("m", "A", ["s1", "s2"], [[SURE_O]]), 0.5)


# --- score_grid ------------------------------------------------------------------------------------------------------


def test_score_grid_scores_every_gamma_and_tau_and_leaves_nothing_in_the_work_folder(tmp_path: Path) -> None:
    work = tmp_path / "new" / "work"
    source = Marginals("m", "A", ["s"], [[SURE_COMMA]])
    seen: list[str] = []

    def score_file(path: Path) -> float:
        seen.append(json.loads(path.read_text(encoding="utf-8"))["model_name"])
        return float(len(seen))

    scores = score_grid(source, (0.5, 0.9), score_file, work, gammas=(0.0,))
    assert scores == [(0.5, 0.0, 1.0), (0.9, 0.0, 2.0)]
    assert seen == ["m_tau0.5", "m_tau0.9"]
    assert work.is_dir() and list(work.iterdir()) == []


# --- label priors from the training file -----------------------------------------------------------------------------


def _train_file(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows) + "\n", encoding="utf-8")
    return path


def _row(sample_id: str, punct: list[str], mode: str = "A") -> dict[str, Any]:
    return {"sample_id": sample_id, "label_mode": mode, "tokens": ["w"] * len(punct), "punct_labels": punct,
            "case_labels": ["LOWER"] * len(punct)}


def _ids(tmp_path: Path, *ids: str) -> Path:
    path = tmp_path / "train.ids"
    path.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    return path


def test_stream_train_labels_yields_the_label_lists_of_the_listed_records_only_in_file_order(tmp_path: Path) -> None:
    train = _train_file(tmp_path, [_row("a", ["O"]), _row("skip", ["COMMA"]), _row("b", ["COMMA", "O"])])
    streamed = list(stream_train_labels(train, ["b", "a"], "A"))
    assert streamed == [{"punct_labels": ["O"], "case_labels": ["LOWER"]},
                        {"punct_labels": ["COMMA", "O"], "case_labels": ["LOWER", "LOWER"]}]


@pytest.mark.parametrize(
    ("rows", "message"),
    [([_row("a", ["O"])], r"^1 listed ID\(s\) are missing from train\.jsonl$"),
     ([_row("a", ["O"]), _row("b", ["O"]), _row("b", ["O"])],
      r"^train\.jsonl line 3: a listed sample_id appears twice$"),
     ([_row("a", ["O"]), _row("b", ["O"], "B")],
      r"^train\.jsonl line 2: a listed record is not label mode A$")],
    ids=["missing-id", "repeated-id", "other-mode"],
)
def test_stream_train_labels_refuses_a_listed_id_that_is_missing_repeated_or_of_another_mode(
        tmp_path: Path, rows: list[dict[str, Any]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        list(stream_train_labels(_train_file(tmp_path, rows), ["a", "b"], "A"))


def test_train_priors_are_the_add_one_frequencies_of_the_listed_records(tmp_path: Path) -> None:
    train = _train_file(tmp_path, [_row("a", ["O", "COMMA", "O", "O"])])
    priors = train_priors(train, _ids(tmp_path, "a"), "A", ["O", "COMMA", "PERIOD"])
    assert priors == {"COMMA": 2 / 7, "PERIOD": 1 / 7}


def test_train_priors_names_a_missing_training_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"^training file not found at ghost\.jsonl$"):
        train_priors(tmp_path / "ghost.jsonl", _ids(tmp_path, "a"), "A", ["O", "COMMA"])


def test_train_priors_names_a_training_file_that_is_not_utf8_but_not_a_bad_id_list(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    train.write_bytes(b"\xff\xfe\n")
    with pytest.raises(ValueError, match=r"^train\.jsonl: not valid UTF-8$"):
        train_priors(train, _ids(tmp_path, "a"), "A", ["O", "COMMA"])
    ids = tmp_path / "bad.ids"
    ids.write_bytes(b"\xff\n")
    with pytest.raises(ValueError, match=r"^bad\.ids: not valid UTF-8$"):
        train_priors(_train_file(tmp_path, [_row("a", ["O"])]), ids, "A", ["O", "COMMA"])


# --- tau files -------------------------------------------------------------------------------------------------------


def _model(tmp_path: Path) -> Path:
    path = tmp_path / "m.joblib"
    path.write_bytes(b"model bytes")
    return path


def _tau_file(tmp_path: Path, **fields: Any) -> Path:
    data = {"tau": 0.4, "model_sha256": sha256_file(_model(tmp_path)), **fields}
    path = tmp_path / "tau.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_read_tau_file_reads_a_tau_file_that_predates_gamma_as_gamma_zero_without_priors(tmp_path: Path) -> None:
    assert read_tau_file(_tau_file(tmp_path), _model(tmp_path)) == TauSettings(0.4, 0.0, None)


def test_read_tau_file_reads_gamma_and_priors_and_ignores_the_grid_and_provenance(tmp_path: Path) -> None:
    path = _tau_file(tmp_path, gamma=0.5, priors={"COMMA": 0.1}, grid=[], split="validation", inputs={})
    assert read_tau_file(path, _model(tmp_path)) == TauSettings(0.4, 0.5, {"COMMA": 0.1})


def test_read_tau_file_turns_a_negative_zero_tau_into_zero_so_the_model_is_not_named_tau_minus_zero(
        tmp_path: Path) -> None:
    settings = read_tau_file(_tau_file(tmp_path, tau=-0.0), _model(tmp_path))
    assert math.copysign(1.0, settings.tau) == 1.0


@pytest.mark.parametrize(
    ("fields", "message"),
    [({"gamma": 0.5}, r"^tau file tau\.json: priors must be a mapping"),
     ({"tau": 1.5}, r"^tau file tau\.json: tau must be a number between 0 and 1"),
     ({"model_sha256": "0" * 64}, r"^tau file was tuned for a different model$")],
    ids=["gamma-without-priors", "tau-out-of-range", "other-model"],
)
def test_read_tau_file_refuses_a_file_that_cannot_decode_with_this_model(
        tmp_path: Path, fields: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        read_tau_file(_tau_file(tmp_path, **fields), _model(tmp_path))


@pytest.mark.parametrize(
    ("content", "message"),
    [(b"\xff", r"^tau file tau\.json is not valid UTF-8$"),
     (b"{", r"^tau file tau\.json is not valid JSON$"),
     (b"[1]", r"^tau file tau\.json must be a JSON object with a tau and a model_sha256 string$")],
    ids=["not-utf8", "not-json", "not-an-object"],
)
def test_read_tau_file_names_a_file_that_is_not_a_json_object(tmp_path: Path, content: bytes, message: str) -> None:
    path = tmp_path / "tau.json"
    path.write_bytes(content)
    with pytest.raises(ValueError, match=message):
        read_tau_file(path, _model(tmp_path))


def test_read_tau_file_names_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"^tau file not found at ghost\.json$"):
        read_tau_file(tmp_path / "ghost.json", _model(tmp_path))


def test_tau_result_lists_tau_gamma_priors_grid_and_model_before_the_provenance() -> None:
    result = tau_result((0.5, 0.25), [(0.5, 0.25, 0.75)], {"COMMA": 0.1}, "abc", {"split": "validation"})
    assert list(result) == ["tau", "gamma", "priors", "grid", "model_sha256", "split"]
    assert result["grid"] == [{"tau": 0.5, "gamma": 0.25, "macro_f1": 0.75}]


def test_tau_result_leaves_out_the_priors_when_there_are_none() -> None:
    assert list(tau_result((0.5, 0.0), [], None, "abc", {})) == ["tau", "gamma", "grid", "model_sha256"]


def test_the_library_prints_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    train = _train_file(tmp_path, [_row("a", ["O", "COMMA"])])
    train_priors(train, _ids(tmp_path, "a"), "A", ["O", "COMMA"])
    read_tau_file(_tau_file(tmp_path), _model(tmp_path))
    score_grid(Marginals("m", "A", ["s"], [[SURE_O]]), (0.5,), lambda _: 1.0, tmp_path / "work")
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
