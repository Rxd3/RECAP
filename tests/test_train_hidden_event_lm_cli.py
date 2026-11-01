"""punct_restore.cli.train_hidden_event_lm: train and predict CLI for the hidden-event language model (method H)."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from eval_fixtures import assert_one_error_line, gold_row, write_ids, write_jsonl

from punct_restore import prediction_io
from punct_restore.cli import train_hidden_event_lm as cli
from punct_restore.evaluation import load_vocab
from punct_restore.evaluation.inputs import load_bundle
from punct_restore.evaluation.manifest import check_manifest
from punct_restore.models.hidden_event_lm import HiddenEventLM

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_hidden_event_lm.py"
TRAIN_ROWS = [
    ("t1", ["the", "cat", "sat"], ["O", "O", "PERIOD"]),
    ("t2", ["the", "dog", "sat"], ["O", "O", "PERIOD"]),
    ("t3", ["well", "i", "think", "so"], ["COMMA", "O", "O", "PERIOD"]),
    ("t4", ["how", "are", "you"], ["O", "O", "QUESTION"]),
    ("t5", ["the", "cat", "ran"], ["O", "O", "PERIOD"]),
]
VALID_ROWS = [
    ("v1", ["the", "cat", "sat"], ["O", "O", "PERIOD"]),
    ("v2", ["how", "are", "you"], ["O", "O", "QUESTION"]),
    ("v3", ["well", "the", "end"], ["COMMA", "O", "PERIOD"]),
]


def _rows(mode: str, rows: list[tuple[str, list[str], list[str]]], split: str) -> list[dict[str, Any]]:
    result = []
    for sample_id, tokens, punct in rows:
        row = gold_row(sample_id, mode, tokens, punct, ["LOWER"] * len(tokens))
        row.update(split=split, schema_version=1, metadata={"offsets": [[0, 1]] * len(tokens), "group_id": "g"})
        result.append(row)
    return result


@dataclass(frozen=True)
class Case:
    folder: Path
    train: Path
    train_ids: Path
    valid: Path
    valid_ids: Path
    model: Path
    preds: Path
    man: Path

    def train_argv(self, mode: str = "A", **override: Path | str) -> list[str]:
        values = {"input": self.train, "ids": self.train_ids, "output": self.model, **override}
        return ["train", "--mode", mode, "--input", str(values["input"]), "--ids", str(values["ids"]),
                "--output", str(values["output"])]

    def predict_argv(self, *extra: str, **override: Path) -> list[str]:
        values = {"model": self.model, "input": self.valid, "output": self.preds, **override}
        return ["predict", "--model", str(values["model"]), "--input", str(values["input"]),
                "--output", str(values["output"]), *extra]


def _build(folder: Path, mode: str = "A") -> Case:
    folder.mkdir(parents=True, exist_ok=True)
    train = _rows(mode, TRAIN_ROWS, "train")
    # records that must be ignored: a train record and a validation record the ID list does not name
    extra = _rows(mode, [("unlisted_train", ["x", "y"], ["O", "PERIOD"])], "train")
    extra += _rows(mode, [("unlisted_valid", ["x", "y"], ["O", "PERIOD"])], "validation")
    return Case(
        folder=folder,
        train=write_jsonl(folder / "train.jsonl", [*extra[:1], *train, *extra[1:]]),
        train_ids=write_ids(folder / "train.ids", [sample_id for sample_id, _, _ in TRAIN_ROWS]),
        valid=write_jsonl(folder / "validation.jsonl", _rows(mode, VALID_ROWS, "validation")),
        valid_ids=write_ids(folder / "validation.ids", [sample_id for sample_id, _, _ in VALID_ROWS]),
        model=folder / "model.json", preds=folder / "pred.jsonl", man=folder / "pred.manifest.json")


@pytest.fixture
def case(tmp_path: Path) -> Case:
    return _build(tmp_path / "case")


def _run(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, str, str]:
    code = cli.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _refused(capsys: pytest.CaptureFixture[str], tmp_path: Path, argv: list[str]) -> str:
    code, out, err = _run(capsys, argv)
    assert code == 1 and out == ""
    assert_one_error_line(err)
    assert str(tmp_path) not in err, "an error must name files by name only"
    return err


# ------------------------------------------------------------------ the happy path


def test_train_then_predict_passes_the_evaluator_manifest_check(case: Case, capsys) -> None:
    code, out, err = _run(capsys, case.train_argv())
    assert (code, err) == (0, "")
    assert out == "trained mode A order 3: 5 sequences, 16 tokens, 12 word types -> model.json\n"
    code, out, err = _run(capsys, case.predict_argv("--ids", str(case.valid_ids), "--manifest-out", str(case.man)))
    assert (code, err) == (0, "")
    assert out == "predicted 3 sequences -> pred.jsonl\n"
    vocab = load_vocab()
    bundle = load_bundle(case.valid, case.valid_ids, case.preds, None, vocab)
    check_manifest(case.man, bundle, vocab)
    manifest = json.loads(case.man.read_text(encoding="utf-8"))
    assert (manifest["model_name"], manifest["label_mode"]) == ("hidden_event_lm", "A")
    rows = [json.loads(line) for line in case.preds.read_text(encoding="utf-8").splitlines()]
    assert [row["sample_id"] for row in rows] == ["v1", "v2", "v3"]
    assert rows[0]["predicted_labels"] == ["O", "O", "PERIOD"]


def test_only_the_listed_records_are_trained_on_and_options_reach_the_model(case: Case, capsys) -> None:
    argv = [*case.train_argv(), "--order", "2", "--discount", "0.5", "--min-count", "2", "--model-name", "h2"]
    assert _run(capsys, argv)[0] == 0
    model = HiddenEventLM.load_model(case.model)
    assert (model.order, model.discount, model.min_count, model.model_name) == (2, 0.5, 2, "h2")
    assert model.training_summary["num_sequences"] == 5
    assert "x" not in model.predicted_tokens, "an unlisted record leaked into training"


def test_predict_without_ids_reads_every_record_and_manifest_is_optional(case: Case, capsys) -> None:
    assert _run(capsys, case.train_argv())[0] == 0
    code, out, err = _run(capsys, case.predict_argv())
    assert (code, err) == (0, "")
    assert out == "predicted 3 sequences -> pred.jsonl\n"
    assert not case.man.exists()


@pytest.mark.parametrize("mode", ["B", "C"])
def test_modes_b_and_c_train_and_predict_in_their_label_vocabularies(tmp_path: Path, capsys, mode: str) -> None:
    other = _build(tmp_path / "other", mode)
    assert _run(capsys, other.train_argv(mode))[0] == 0
    assert _run(capsys, other.predict_argv("--ids", str(other.valid_ids)))[0] == 0
    labels = [label for line in other.preds.read_text(encoding="utf-8").splitlines()
              for label in json.loads(line)["predicted_labels"]]
    assert set(labels) <= set(load_vocab().labels(mode))
    assert all(("|" in label) == (mode == "C") for label in labels)


# ------------------------------------------------------------------ refusals


def test_existing_model_output_is_never_overwritten(case: Case, capsys, tmp_path: Path) -> None:
    case.model.write_text("keep me", encoding="utf-8")
    assert "already exists" in _refused(capsys, tmp_path, case.train_argv())
    assert case.model.read_text(encoding="utf-8") == "keep me"


@pytest.mark.parametrize("clash", ["input", "ids"])
def test_train_output_equal_to_an_input_is_refused_before_reading(
        case: Case, capsys, tmp_path: Path, clash: str) -> None:
    target = {"input": case.train, "ids": case.train_ids}[clash]
    before = target.read_bytes()
    err = _refused(capsys, tmp_path, case.train_argv(output=target))
    assert "same file" in err
    assert target.read_bytes() == before


def test_a_listed_id_missing_from_the_input_is_refused(case: Case, capsys, tmp_path: Path) -> None:
    write_ids(case.train_ids, ["t1", "t2", "ghost"])
    err = _refused(capsys, tmp_path, case.train_argv())
    assert err == "error: train.jsonl: 1 listed ID(s) are missing, first 'ghost'\n"
    assert not case.model.exists()


def test_a_listed_record_that_is_not_train_is_refused(case: Case, capsys, tmp_path: Path) -> None:
    write_ids(case.train_ids, ["t1", "unlisted_valid"])
    err = _refused(capsys, tmp_path, case.train_argv())
    assert "unlisted_valid" in err and "validation" in err
    assert not case.model.exists()


def test_a_listed_record_of_another_label_mode_is_refused(case: Case, capsys, tmp_path: Path) -> None:
    err = _refused(capsys, tmp_path, case.train_argv("B"))
    assert "label_mode" in err and not case.model.exists()


@pytest.mark.parametrize("argv_change", [
    {"input": "absent.jsonl"}, {"ids": "absent.ids"}])
def test_a_missing_input_file_names_the_file_only(case: Case, capsys, tmp_path: Path, argv_change: dict) -> None:
    name = next(iter(argv_change.values()))
    err = _refused(capsys, tmp_path, case.train_argv(**{k: case.folder / v for k, v in argv_change.items()}))
    assert name in err and not case.model.exists()


@pytest.mark.parametrize(("option", "what"), [("input", "input JSONL"), ("ids", "ID list")])
def test_train_says_which_file_is_not_found(case: Case, capsys, tmp_path: Path, option: str, what: str) -> None:
    err = _refused(capsys, tmp_path, case.train_argv(**{option: case.folder / "ghost.txt"}))
    assert err == f"error: ghost.txt: {what} not found\n" and not case.model.exists()


@pytest.mark.parametrize("option", ["input", "ids"])
def test_train_says_a_folder_given_as_a_file_is_a_folder(case: Case, capsys, tmp_path: Path, option: str) -> None:
    (case.folder / "adir").mkdir()
    err = _refused(capsys, tmp_path, case.train_argv(**{option: case.folder / "adir"}))
    assert err == "error: adir: is a folder, not a file\n" and not case.model.exists()


def test_corrupt_and_duplicate_input_lines_are_one_line_errors(case: Case, capsys, tmp_path: Path) -> None:
    lines = case.train.read_text(encoding="utf-8").splitlines()
    case.train.write_text("\n".join([*lines, "{not json"]) + "\n", encoding="utf-8")
    assert "line 8" in _refused(capsys, tmp_path, case.train_argv())
    case.train.write_text("\n".join([*lines, lines[1]]) + "\n", encoding="utf-8")
    assert "duplicate" in _refused(capsys, tmp_path, case.train_argv())
    case.train.write_text("\n".join([*lines, "[1, 2]"]) + "\n", encoding="utf-8")
    assert "not a JSON object" in _refused(capsys, tmp_path, case.train_argv())
    assert not case.model.exists()


@pytest.mark.parametrize("label", [None, 1, {}, ["O"]], ids=["null", "number", "object", "list"])
@pytest.mark.parametrize(("mode", "field"), [("A", "punct_labels"), ("C", "punct_labels"), ("C", "case_labels")])
def test_a_training_label_that_is_not_text_is_one_error_line_naming_the_sample(
        tmp_path: Path, capsys, mode: str, field: str, label: object) -> None:
    """A `null` label reached the event table as `KeyError: None` and printed a traceback."""
    made = _build(tmp_path / "case", mode)
    rows = [json.loads(line) for line in made.train.read_text(encoding="utf-8").splitlines()]
    target = next(row for row in rows if row["sample_id"] == "t1")
    target[field][0] = label
    write_jsonl(made.train, rows)
    err = _refused(capsys, tmp_path, made.train_argv(mode))
    assert err.startswith(f"error: train.jsonl: 't1': unknown {field} label")
    assert not made.model.exists()


def test_bad_hyperparameters_are_one_line_errors(case: Case, capsys, tmp_path: Path) -> None:
    err = _refused(capsys, tmp_path, [*case.train_argv(), "--discount", "1.5"])
    assert "discount" in err and not case.model.exists()


def test_empty_ids_and_non_utf8_input_are_one_line_errors(case: Case, capsys, tmp_path: Path) -> None:
    case.train_ids.write_text("\n", encoding="utf-8")
    assert "empty" in _refused(capsys, tmp_path, case.train_argv())
    write_ids(case.train_ids, ["t1"])
    case.train.write_bytes(b'\xe9\xff\n{"sample_id": "t1"}\n')
    assert "train.jsonl" in _refused(capsys, tmp_path, case.train_argv())


DEEP_LINE = "[" * 5000 + "]" * 5000  # json.loads gives up on it with RecursionError, which is not a ValueError


def test_a_deeply_nested_training_line_is_a_one_line_error(case: Case, capsys, tmp_path: Path) -> None:
    lines = case.train.read_text(encoding="utf-8").splitlines()
    case.train.write_text("\n".join([*lines, DEEP_LINE]) + "\n", encoding="utf-8")
    assert "train.jsonl line 8: not valid JSON" in _refused(capsys, tmp_path, case.train_argv())
    assert not case.model.exists()


def test_a_non_utf8_training_id_list_names_the_ids_file(case: Case, capsys, tmp_path: Path) -> None:
    case.train_ids.write_bytes(b"t1\n\xff\n")
    assert _refused(capsys, tmp_path, case.train_argv()) == "error: train.ids: not valid UTF-8\n"
    assert not case.model.exists()


@pytest.fixture
def trained(case: Case, capsys) -> Case:
    assert _run(capsys, case.train_argv())[0] == 0
    capsys.readouterr()
    return case


@pytest.mark.parametrize("existing", ["output", "manifest"])
def test_predict_never_overwrites_an_existing_output(trained: Case, capsys, tmp_path: Path, existing: str) -> None:
    target = {"output": trained.preds, "manifest": trained.man}[existing]
    target.write_text("keep me", encoding="utf-8")
    err = _refused(capsys, tmp_path, trained.predict_argv("--manifest-out", str(trained.man)))
    assert "already exists" in err
    assert target.read_text(encoding="utf-8") == "keep me"


@pytest.mark.parametrize("clash", ["manifest-is-output", "output-is-model", "output-is-input", "manifest-is-model",
                                   "manifest-is-input", "output-is-ids"])
def test_predict_outputs_may_not_collide_with_each_other_or_with_inputs(
        trained: Case, capsys, tmp_path: Path, clash: str) -> None:
    ids = ["--ids", str(trained.valid_ids)]
    argv = {
        "manifest-is-output": trained.predict_argv(*ids, "--manifest-out", str(trained.preds)),
        "output-is-model": trained.predict_argv(*ids, output=trained.model),
        "output-is-input": trained.predict_argv(*ids, output=trained.valid),
        "manifest-is-model": trained.predict_argv(*ids, "--manifest-out", str(trained.model)),
        "manifest-is-input": trained.predict_argv(*ids, "--manifest-out", str(trained.valid)),
        "output-is-ids": trained.predict_argv(*ids, output=trained.valid_ids),
    }[clash]
    watched = [trained.model, trained.valid, trained.valid_ids]
    before = [path.read_bytes() for path in watched]
    assert "same file" in _refused(capsys, tmp_path, argv)
    assert [path.read_bytes() for path in watched] == before
    assert not trained.preds.exists()


@pytest.mark.parametrize(("option", "what"), [("model", "model artifact"), ("input", "input JSONL")])
def test_predict_says_which_file_is_not_found(
        trained: Case, capsys, tmp_path: Path, option: str, what: str) -> None:
    err = _refused(capsys, tmp_path, trained.predict_argv(**{option: trained.folder / "ghost.txt"}))
    assert err == f"error: ghost.txt: {what} not found\n" and not trained.preds.exists()


def test_predict_says_an_ids_file_is_not_found(trained: Case, capsys, tmp_path: Path) -> None:
    err = _refused(capsys, tmp_path, trained.predict_argv("--ids", str(trained.folder / "ghost.txt")))
    assert err == "error: ghost.txt: ID list not found\n" and not trained.preds.exists()


@pytest.mark.parametrize("option", ["model", "input", "ids"])
def test_predict_says_a_folder_given_as_a_file_is_a_folder(
        trained: Case, capsys, tmp_path: Path, option: str) -> None:
    (trained.folder / "adir").mkdir()
    folder = trained.folder / "adir"
    argv = trained.predict_argv("--ids", str(folder)) if option == "ids" else trained.predict_argv(**{option: folder})
    err = _refused(capsys, tmp_path, argv)
    assert err == "error: adir: is a folder, not a file\n" and not trained.preds.exists()


def test_predict_checks_its_outputs_before_loading_the_model(trained: Case, capsys, tmp_path: Path) -> None:
    trained.model.write_text("{not json", encoding="utf-8")
    trained.preds.write_text("keep me", encoding="utf-8")
    assert "already exists" in _refused(capsys, tmp_path, trained.predict_argv())


@pytest.mark.parametrize(("fault", "fragment"), [
    ("missing model", "absent_model.json"), ("corrupt model", "not valid JSON"),
    ("missing input", "absent_input.jsonl"), ("missing id", "ghost"), ("missing ids file", "absent.ids")])
def test_predict_faults_are_one_line_errors_and_write_nothing(
        trained: Case, capsys, tmp_path: Path, fault: str, fragment: str) -> None:
    argv_extra: list[str] = ["--manifest-out", str(trained.man)]
    overrides: dict[str, Path] = {}
    if fault == "missing model":
        overrides["model"] = trained.folder / "absent_model.json"
    elif fault == "corrupt model":
        trained.model.write_text("{not json", encoding="utf-8")
    elif fault == "missing input":
        overrides["input"] = trained.folder / "absent_input.jsonl"
    elif fault == "missing id":
        write_ids(trained.valid_ids, ["v1", "ghost"])
        argv_extra += ["--ids", str(trained.valid_ids)]
    else:
        argv_extra += ["--ids", str(trained.folder / "absent.ids")]
    err = _refused(capsys, tmp_path, trained.predict_argv(*argv_extra, **overrides))
    assert fragment in err
    assert not trained.preds.exists() and not trained.man.exists()


@pytest.mark.parametrize(("content", "fragment"), [("{not json", "not valid JSON"),
                                                   ("[1, 2]", "invalid hidden-event LM artifact")])
def test_a_corrupt_model_reports_the_artifact_error(
        trained: Case, capsys, tmp_path: Path, content: str, fragment: str) -> None:
    trained.model.write_text(content, encoding="utf-8")
    assert fragment in _refused(capsys, tmp_path, trained.predict_argv())


@pytest.mark.parametrize("command", ["train", "predict"])
def test_a_recursion_error_from_our_own_code_is_not_reported_as_bad_input(
        trained: Case, monkeypatch: pytest.MonkeyPatch, command: str) -> None:
    def bug(self: HiddenEventLM, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raise RecursionError("a bug in the model")

    monkeypatch.setattr(HiddenEventLM, "fit" if command == "train" else "predict", bug)
    argv = trained.train_argv(output=trained.folder / "again.json") if command == "train" else trained.predict_argv()
    with pytest.raises(RecursionError, match="a bug in the model"):
        cli.main(argv)


def test_a_deeply_nested_predict_input_line_is_a_one_line_error(trained: Case, capsys, tmp_path: Path) -> None:
    lines = trained.valid.read_text(encoding="utf-8").splitlines()
    trained.valid.write_text("\n".join([*lines, DEEP_LINE]) + "\n", encoding="utf-8")
    line = _refused(capsys, tmp_path, trained.predict_argv("--manifest-out", str(trained.man)))
    assert trained.valid.name in line and "not valid JSON" in line
    assert not trained.preds.exists() and not trained.man.exists()


def test_a_corrupt_predict_input_line_names_the_file_and_line_once(trained: Case, capsys, tmp_path: Path) -> None:
    lines = trained.valid.read_text(encoding="utf-8").splitlines()
    trained.valid.write_text("".join(f"{line}\n" for line in [*lines, "{not json"]), encoding="utf-8")
    line = _refused(capsys, tmp_path, trained.predict_argv())
    assert line == f"error: {trained.valid.name} line {len(lines) + 1}: not valid JSON\n"


def test_a_model_with_an_oversized_count_is_a_one_line_error(trained: Case, capsys, tmp_path: Path) -> None:
    payload = json.loads(trained.model.read_text(encoding="utf-8"))
    payload["ngrams"][1]["counts"][0] = 10**400  # a bigram count: it would only overflow while decoding
    trained.model.write_text(json.dumps(payload), encoding="utf-8")
    assert "invalid hidden-event LM artifact" in _refused(capsys, tmp_path, trained.predict_argv())


@pytest.mark.parametrize("broken", ["validation.ids", "validation.jsonl"])
def test_a_non_utf8_predict_file_is_named_in_a_one_line_error(
        trained: Case, capsys, tmp_path: Path, broken: str) -> None:
    path = {"validation.ids": trained.valid_ids, "validation.jsonl": trained.valid}[broken]
    path.write_bytes(path.read_bytes() + b"\xff\n")
    err = _refused(capsys, tmp_path, trained.predict_argv("--ids", str(trained.valid_ids)))
    assert err == f"error: {broken}: not valid UTF-8\n"
    assert not trained.preds.exists()


def test_predict_reads_the_id_list_once(trained: Case, capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    reads: list[Path] = []
    real_read_ids = cli.read_ids

    def counting_read_ids(path: Path, **options: Any) -> list[str]:
        reads.append(Path(path))
        return real_read_ids(path, **options)

    monkeypatch.setattr(cli, "read_ids", counting_read_ids)
    monkeypatch.setattr(prediction_io, "read_ids", counting_read_ids)
    assert _run(capsys, trained.predict_argv("--ids", str(trained.valid_ids)))[0] == 0
    assert reads == [trained.valid_ids]


@pytest.mark.parametrize("appears", ["output", "manifest"])
def test_an_output_created_while_decoding_is_not_overwritten(
        trained: Case, capsys, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, appears: str) -> None:
    target, other = (trained.preds, trained.man) if appears == "output" else (trained.man, trained.preds)
    real_predict = HiddenEventLM.predict

    def predict_while_someone_else_writes(self: HiddenEventLM, records: Any) -> list[dict[str, Any]]:
        predictions = real_predict(self, records)
        target.write_text("keep me", encoding="utf-8")
        return predictions

    monkeypatch.setattr(HiddenEventLM, "predict", predict_while_someone_else_writes)
    assert "already exists" in _refused(capsys, tmp_path, trained.predict_argv("--manifest-out", str(trained.man)))
    assert target.read_text(encoding="utf-8") == "keep me"
    assert not other.exists()


# ------------------------------------------------------------------ usage and process behaviour


@pytest.mark.parametrize("argv", [[], ["train"], ["predict", "--model", "m.json"], ["fit"]])
def test_usage_errors_exit_with_two(capsys, argv: list[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        cli.main(argv)
    assert stop.value.code == 2
    assert capsys.readouterr().err


def test_stdout_without_reconfigure_still_prints_the_summary(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)
    assert cli.main(case.train_argv()) == 0
    assert buffer.getvalue().startswith("trained mode A order 3")


def test_an_os_error_is_reported_without_the_full_path(case: Case, capsys, tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    def full_disk(self: HiddenEventLM, path: Any) -> None:
        raise OSError(28, "No space left on device", str(case.model))

    monkeypatch.setattr(HiddenEventLM, "save_model", full_disk)
    assert _refused(capsys, tmp_path, case.train_argv()) == "error: model.json: No space left on device\n"


def test_the_cli_runs_as_a_subprocess_without_pythonpath(case: Case) -> None:
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=case.folder, capture_output=True,
                              text=True, encoding="utf-8", timeout=120, env=env, check=False)

    trained = run(*case.train_argv())
    assert (trained.returncode, trained.stderr) == (0, "")
    predicted = run(*case.predict_argv("--ids", str(case.valid_ids), "--manifest-out", str(case.man)))
    assert (predicted.returncode, predicted.stderr) == (0, "")
    assert case.preds.is_file() and case.man.is_file()
    refused = run(*case.train_argv())
    assert refused.returncode == 1 and refused.stdout == ""
    assert_one_error_line(refused.stderr)
    assert run("fit").returncode == 2
