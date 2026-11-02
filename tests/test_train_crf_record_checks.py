"""train-crf train: a record whose tokens or labels are damaged is refused while the frozen file is read.

Each damage gives one ``error: data.jsonl line N: <what is wrong>`` line, the file named by name only and the line
that holds the record, before any model is configured or trained. The checks are the ones ``CRFModel.fit`` makes
(punct_labels / case_labels / vocabulary / token types), run by the loader on the line they came from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from crf_cli_fixtures import create_synthetic_train_jsonl, write_id_file

import punct_restore.cli.train_crf as train_crf
from punct_restore.data.labeling import LABEL_SETS_PATH

BAD_LINE = 2  # the record damaged below is sample_0001: its tokens are how / are / you


def _argv(tmp_path: Path, mode: str = "A", *extra: str) -> list[str]:
    data, ids = tmp_path / "data.jsonl", tmp_path / "train.ids"
    write_id_file(ids, create_synthetic_train_jsonl(data, mode=mode, num_samples=3))
    return ["train", "-m", mode, "-i", str(data), "--ids", str(ids), "-o", str(tmp_path / "out.joblib"),
            "--max-iterations", "5", *extra]


def _damage(tmp_path: Path, **change: Any) -> None:
    """Change the fields of the record on line ``BAD_LINE`` of data.jsonl."""
    data = tmp_path / "data.jsonl"
    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines()]
    rows[BAD_LINE - 1].update(change)
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _refused(tmp_path: Path, capsys: pytest.CaptureFixture[str], argv: list[str]) -> str:
    """Run the command; it must exit 1 with exactly one stderr line, no folder shown and no model written."""
    assert train_crf.main(argv) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and err.startswith("error: ")
    assert str(tmp_path) not in err
    assert not (tmp_path / "out.joblib").exists()
    return err


MODE_A_DAMAGE = [
    pytest.param({"tokens": "how are you"}, "tokens must be a non-empty list.", id="tokens-not-a-list"),
    pytest.param({"tokens": []}, "tokens must be a non-empty list.", id="tokens-empty"),
    pytest.param({"tokens": ["how", 5, "you"]}, "tokens must contain non-empty strings.", id="token-not-a-string"),
    pytest.param({"tokens": ["how", "", "you"]}, "tokens must contain non-empty strings.", id="token-empty"),
    pytest.param({"tokens": ["how", "  ", "you"]}, "tokens must contain non-empty strings.", id="token-blank"),
    pytest.param({"punct_labels": "O"}, "punct_labels must be a list of strings.", id="labels-not-a-list"),
    pytest.param({"punct_labels": None}, "punct_labels must be a list of strings.", id="labels-null"),
    pytest.param({"punct_labels": ["O", 7, "O"]}, "punct_labels must be a list of strings.", id="label-not-a-string"),
    pytest.param({"punct_labels": ["O", None, "O"]}, "punct_labels must be a list of strings.", id="label-null"),
    pytest.param({"punct_labels": ["O", ["COMMA"], "O"]}, "punct_labels must be a list of strings.", id="label-list"),
    pytest.param({"punct_labels": ["O", "FOO", "O"]}, "unknown punct_labels ['FOO'] for Mode A.", id="label-unknown"),
    pytest.param({"punct_labels": ["O", "EXCLAMATION", "O"]}, "unknown punct_labels ['EXCLAMATION'] for Mode A.",
                 id="label-of-mode-b"),
    pytest.param({"punct_labels": ["O", "O"]}, "punct_labels length (2) does not match tokens length (3).",
                 id="labels-short"),
]


@pytest.mark.parametrize(("change", "reason"), MODE_A_DAMAGE)
def test_a_damaged_mode_a_record_is_one_line_naming_the_file_and_line(
        tmp_path: Path, capsys: pytest.CaptureFixture[str], change: dict[str, Any], reason: str) -> None:
    argv = _argv(tmp_path)
    _damage(tmp_path, **change)
    assert _refused(tmp_path, capsys, argv) == f"error: data.jsonl line {BAD_LINE}: {reason}\n"


def test_an_unknown_label_in_mode_b_names_mode_b(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = _argv(tmp_path, "B")
    _damage(tmp_path, punct_labels=["O", "FOO", "O"])
    reason = "unknown punct_labels ['FOO'] for Mode B."
    assert _refused(tmp_path, capsys, argv) == f"error: data.jsonl line {BAD_LINE}: {reason}\n"


MODE_C_DAMAGE = [
    pytest.param({"case_labels": ["LOWER", "MIXED", "LOWER"]}, "unknown case_labels ['MIXED'].", id="case-unknown"),
    pytest.param({"case_labels": ["LOWER", 3, "LOWER"]}, "case_labels must be a list of strings.",
                 id="case-not-a-string"),
    pytest.param({"case_labels": ["LOWER"]}, "case_labels length (1) does not match tokens length (3).",
                 id="case-short"),
    pytest.param({"punct_labels": ["O", 7, "O"]}, "punct_labels must be a list of strings.", id="punct-not-a-string"),
]


@pytest.mark.parametrize(("change", "reason"), MODE_C_DAMAGE)
def test_a_damaged_mode_c_record_is_one_line_naming_the_file_and_line(
        tmp_path: Path, capsys: pytest.CaptureFixture[str], change: dict[str, Any], reason: str) -> None:
    argv = _argv(tmp_path, "C")
    _damage(tmp_path, **change)
    assert _refused(tmp_path, capsys, argv) == f"error: data.jsonl line {BAD_LINE}: {reason}\n"


def test_the_first_damaged_line_is_the_one_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = _argv(tmp_path)
    data = tmp_path / "data.jsonl"
    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines()]
    rows[2]["punct_labels"] = ["O", "FOO", "O"]
    rows[1]["tokens"] = ["how", 5, "you"]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert _refused(tmp_path, capsys, argv) == "error: data.jsonl line 2: tokens must contain non-empty strings.\n"


def test_a_damaged_record_that_the_id_list_does_not_select_is_not_looked_at(tmp_path: Path) -> None:
    argv = _argv(tmp_path)
    _damage(tmp_path, punct_labels=["O", "FOO", "O"])
    (tmp_path / "train.ids").write_text("sample_0000\nsample_0002\n", encoding="utf-8")
    assert train_crf.main(argv) == 0


def test_a_good_file_still_trains(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert train_crf.main(_argv(tmp_path)) == 0
    assert capsys.readouterr().err == ""
    assert (tmp_path / "out.joblib").exists()


# --- the vocabulary is the one of the label file the model will use ---------------------------------------------------


def _renamed_comma_config(tmp_path: Path) -> Path:
    """A copy of the label file whose label COMMA is called KOMMA (the 4/8/12 structure is kept)."""
    config = tmp_path / "custom_labels.yaml"
    text = Path(LABEL_SETS_PATH).read_text(encoding="utf-8").replace("COMMA", "KOMMA")
    config.write_text(text, encoding="utf-8", newline="\n")
    return config


def test_a_label_of_the_given_config_path_is_accepted(tmp_path: Path) -> None:
    argv = _argv(tmp_path)
    data = tmp_path / "data.jsonl"
    data.write_text(data.read_text(encoding="utf-8").replace("COMMA", "KOMMA"), encoding="utf-8")
    assert train_crf.main([*argv, "--config-path", str(_renamed_comma_config(tmp_path))]) == 0


def test_the_same_label_is_refused_under_the_default_label_file(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = _argv(tmp_path)
    data = tmp_path / "data.jsonl"
    data.write_text(data.read_text(encoding="utf-8").replace("COMMA", "KOMMA"), encoding="utf-8")
    reason = "unknown punct_labels ['KOMMA'] for Mode A."
    assert _refused(tmp_path, capsys, argv) == f"error: data.jsonl line 2: {reason}\n"


def test_a_missing_config_path_is_one_line_naming_the_file_only(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = _argv(tmp_path, "A", "--config-path", str(tmp_path / "nolabels.yaml"))
    err = _refused(tmp_path, capsys, argv)
    assert err.startswith("error: nolabels.yaml")
