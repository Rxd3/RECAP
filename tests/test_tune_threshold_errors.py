"""recap tune-threshold: a failure is one stderr line that names every file by file name only, never a folder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_threshold import Fixture, _build_fixture, _predict_argv, _sha256, _tune_argv

from punct_restore.cli import tune_threshold

FOLDER = "secretfolder"  # a folder name that must never be shown


@pytest.fixture(scope="module")
def mode_a(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    return _build_fixture(tmp_path_factory.mktemp("errors_mode_a"), "A")


def _one_line(capsys: pytest.CaptureFixture[str]) -> str:
    err = capsys.readouterr().err
    assert err.startswith("error: ") and err.count("\n") == 1, err
    assert FOLDER not in err, err
    return err


def _in_folder(tmp_path: Path, name: str) -> Path:
    folder = tmp_path / FOLDER
    folder.mkdir(exist_ok=True)
    return folder / name


def test_tune_with_a_missing_model_names_the_file_only(
    mode_a: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _tune_argv(mode_a, tmp_path / "tau.json")
    argv[argv.index("--model") + 1] = str(tmp_path / FOLDER / "ghost.joblib")
    assert tune_threshold.main(argv) == 1
    assert _one_line(capsys) == "error: Model artifact not found at ghost.joblib\n"


def test_predict_with_a_missing_model_names_the_file_only(
    mode_a: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tau_file = tmp_path / "tau.json"
    tau_file.write_text(json.dumps({"tau": 0.3, "model_sha256": _sha256(mode_a.model)}), encoding="utf-8")
    argv = _predict_argv(mode_a, tau_file, tmp_path / "p.jsonl", tmp_path / "p.man.json")
    argv[argv.index("--model") + 1] = str(tmp_path / FOLDER / "ghost.joblib")
    assert tune_threshold.main(argv) == 1
    assert _one_line(capsys) == "error: Model artifact not found at ghost.joblib\n"


def test_predict_with_a_corrupt_model_names_the_file_only(
    mode_a: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = _in_folder(tmp_path, "bad.joblib")
    bad.write_bytes(b"this is not a joblib file")
    tau_file = tmp_path / "tau.json"
    tau_file.write_text(json.dumps({"tau": 0.3, "model_sha256": "x"}), encoding="utf-8")
    argv = _predict_argv(mode_a, tau_file, tmp_path / "p.jsonl", tmp_path / "p.man.json")
    argv[argv.index("--model") + 1] = str(bad)
    assert tune_threshold.main(argv) == 1
    assert _one_line(capsys).startswith("error: Corrupt or unreadable model artifact at bad.joblib: ")


def test_predict_with_a_missing_tau_file_names_the_file_only(
    mode_a: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _predict_argv(mode_a, tmp_path / FOLDER / "ghost.json", tmp_path / "p.jsonl", tmp_path / "p.man.json")
    assert tune_threshold.main(argv) == 1
    assert _one_line(capsys) == "error: tau file not found at ghost.json\n"


@pytest.mark.parametrize(
    ("content", "message"),
    [("{not json", "tau file bad.json is not valid JSON"),
     ("[1, 2]", "tau file bad.json must be a JSON object"),
     (json.dumps({"tau": 5, "model_sha256": "x"}), "tau file bad.json: tau must be")],
    ids=["not-json", "not-an-object", "bad-tau"],
)
def test_predict_with_a_bad_tau_file_names_the_file_only(
    mode_a: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str, message: str
) -> None:
    tau_file = _in_folder(tmp_path, "bad.json")
    tau_file.write_text(content, encoding="utf-8")
    assert tune_threshold.main(_predict_argv(mode_a, tau_file, tmp_path / "p.jsonl", tmp_path / "p.man.json")) == 1
    assert message in _one_line(capsys)


def test_predict_with_a_tau_file_for_another_model_shows_no_folder(
    mode_a: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tau_file = _in_folder(tmp_path, "tau.json")
    tau_file.write_text(json.dumps({"tau": 0.3, "model_sha256": "0" * 64}), encoding="utf-8")
    assert tune_threshold.main(_predict_argv(mode_a, tau_file, tmp_path / "p.jsonl", tmp_path / "p.man.json")) == 1
    assert _one_line(capsys) == "error: tau file was tuned for a different model\n"


def test_tune_refuses_an_existing_out_and_names_it_by_file_name(
    mode_a: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _in_folder(tmp_path, "tau.json")
    out.write_text("keep me", encoding="utf-8")
    assert tune_threshold.main(_tune_argv(mode_a, out)) == 1
    assert out.read_text(encoding="utf-8") == "keep me"
    assert _one_line(capsys) == "error: tau.json: already exists; refusing to overwrite\n"


@pytest.mark.parametrize("which", ["out", "manifest"])
def test_predict_refuses_an_existing_output_and_names_it_by_file_name(
    mode_a: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str], which: str
) -> None:
    tau_file = tmp_path / "tau.json"
    tau_file.write_text(json.dumps({"tau": 0.3, "model_sha256": _sha256(mode_a.model)}), encoding="utf-8")
    out, man = _in_folder(tmp_path, "preds.jsonl"), _in_folder(tmp_path, "preds.man.json")
    taken = out if which == "out" else man
    taken.write_text("keep me", encoding="utf-8")
    assert tune_threshold.main(_predict_argv(mode_a, tau_file, out, man)) == 1
    assert taken.read_text(encoding="utf-8") == "keep me"
    assert _one_line(capsys) == f"error: {taken.name}: already exists; refusing to overwrite\n"


def test_a_missing_gold_file_names_the_file_only(
    mode_a: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    gold = tmp_path / FOLDER / "ghost.jsonl"
    argv = _tune_argv(Fixture(mode_a.model, gold, mode_a.ids, None), tmp_path / "tau.json")
    assert tune_threshold.main(argv) == 1
    assert "ghost.jsonl" in _one_line(capsys)



def test_predict_with_a_deeply_nested_input_line_is_one_error_line_not_a_traceback(
    mode_a: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deep = _in_folder(tmp_path, "deep.jsonl")
    deep.write_text("[" * 100_000 + "\n", encoding="utf-8")  # json.loads raises RecursionError for it
    tau_file = tmp_path / "tau.json"
    tau_file.write_text(json.dumps({"tau": 0.3, "model_sha256": _sha256(mode_a.model)}), encoding="utf-8")
    argv = _predict_argv(Fixture(mode_a.model, deep, mode_a.ids, None), tau_file, tmp_path / "p.jsonl",
                         tmp_path / "p.man.json")
    assert tune_threshold.main(argv) == 1
    assert _one_line(capsys) == "error: Corrupt JSON at line 1: not valid JSON\n"
    assert not (tmp_path / "p.jsonl").exists() and not (tmp_path / "p.man.json").exists()
