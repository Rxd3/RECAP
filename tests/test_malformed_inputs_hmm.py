"""A malformed input file of an HMM command is one `error: <file name>: <reason>` line and exit 1, never a traceback.

Every case runs the command's `main` in-process on a tiny valid fixture with ONE input replaced by a malformed file.
The command stops before it prints anything, so stdout stays empty. The damaged-study cases at the end run the long
commands (verify may log a mode before it reaches a damaged runtime.json), so they allow progress lines on stdout.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from hmm_study_fixtures import PROCESSED, Study, copy_study, finished_study, patch_study, read_json
from reporting_fixtures import RELEASE, time_argv, tiny_data_root, tiny_hmm
from test_hmm_cli import ROOT, setup_run
from test_hmm_workflow import release_args

from punct_restore.cli import package_hmm_handoff, predict_hmm, run_hmm_experiments, train_hmm, verify_hmm_experiments
from punct_restore.models import hmm_experiment_tune
from punct_restore.reporting import time_models

# What a malformed file holds: a line nested far deeper than the JSON reader allows, bytes that are no UTF-8,
# a JSON array where an object is expected, and an object that lacks every key.
BAD = {
    "nested": b"[" * 100_000 + b"\n",
    "not-utf8": b"\xff\xfe not text\n",
    "array": b"[1]\n",
    "object": b'{"a": 1}\n',
    "manifest-files-not-an-object": (b'{"schema_version": 1, "release_id": "r", "policy_version": "p", '
                                     b'"label_sets_sha256": "s", "files": [], "split_files": {}, '
                                     b'"source": {"revision": "x"}}\n'),
    "selection-without-modes": b'{"used_test_labels": false, "training_includes_validation": false}\n',
}
SHAPES = ["object", "manifest-files-not-an-object", "selection-without-modes"]
KINDS = ["nested", "not-utf8", "array", "object"]
VALID_SELECTION = {
    "plan_sha256": "0" * 64, "modes": {}, "mode_selection_sha256": {},
    "used_test_labels": False, "training_includes_validation": False,
}
VALID_LOOKING_PLAN = {"config": {}, "implementation": {}, "evaluator": {}}


def assert_one_error_line(
        capsys: pytest.CaptureFixture[str], tmp_path: Path, file_name: str, *, quiet: bool = True) -> str:
    """Stdout is empty (unless `quiet` is off: a long command may have logged progress); stderr is one
    `error: <file name>...` line with no folder, no traceback and no usage text. Returns that line."""
    captured = capsys.readouterr()
    err = captured.err
    assert captured.out == "" or not quiet
    assert err.startswith(f"error: {file_name}") and err.endswith("\n") and err.count("\n") == 1, err
    assert str(tmp_path) not in err and tmp_path.name not in err
    assert "Traceback" not in err and "usage:" not in err
    return err


def trained_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    """A valid saved HMM, made by the real train-hmm in a folder of its own (what it printed is dropped)."""
    folder = tmp_path / "trained"
    folder.mkdir()
    argv = setup_run(folder)[0][2:]  # drop the interpreter and the script
    assert train_hmm.main(argv) == 0
    capsys.readouterr()
    return folder / "new-run" / "model.json"


@pytest.mark.parametrize("kind", KINDS)
def test_train_hmm_with_a_malformed_train_file_is_one_error_line(
        kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv, train, _, output = setup_run(tmp_path)
    train.write_bytes(BAD[kind])
    assert train_hmm.main(argv[2:]) == 1
    assert_one_error_line(capsys, tmp_path, "train.jsonl")
    assert not output.exists()


def predict_argv(tmp_path: Path, model: Path, source: Path) -> list[str]:
    return ["--model", str(model), "--input", str(source),
            "--output-dir", str(tmp_path / "predicted"), "--all-records"]


@pytest.mark.parametrize("kind", KINDS)
def test_predict_hmm_with_a_malformed_model_is_one_error_line(
        kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = setup_run(tmp_path)[2]
    model = tmp_path / "model.json"
    model.write_bytes(BAD[kind])
    assert predict_hmm.main(predict_argv(tmp_path, model, source)) == 1
    assert_one_error_line(capsys, tmp_path, "model.json")
    assert not (tmp_path / "predicted").exists()


@pytest.mark.parametrize("kind", KINDS)
def test_predict_hmm_with_a_malformed_input_file_is_one_error_line(
        kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model = trained_model(tmp_path, capsys)
    source = tmp_path / "records.jsonl"
    source.write_bytes(BAD[kind])
    assert predict_hmm.main(predict_argv(tmp_path, model, source)) == 1
    assert_one_error_line(capsys, tmp_path, "records.jsonl")
    assert not (tmp_path / "predicted").exists()


@pytest.mark.parametrize("kind", KINDS)
def test_time_models_with_a_malformed_hmm_model_is_one_error_line(
        kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model = tmp_path / "hmm.json"
    model.write_bytes(BAD[kind])
    argv = time_argv(model, tmp_path / "timing.json", tiny_data_root(tmp_path / "data"))
    assert time_models.main(argv) == 1
    assert_one_error_line(capsys, tmp_path, "hmm.json")
    assert not (tmp_path / "timing.json").exists()


def test_time_models_still_times_a_valid_hmm_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The fixture of the cases above is sound: with an unbroken model the same command succeeds."""
    argv = time_argv(tiny_hmm(tmp_path), tmp_path / "timing.json", tiny_data_root(tmp_path / "data"))
    assert time_models.main(argv) == 0
    assert capsys.readouterr().err == ""


def frozen_run(tmp_path: Path, plan: bytes | dict[str, object], selection: bytes | dict[str, object]) -> Path:
    """A run folder holding only a plan.json and a selection.json, each given as bytes or as a JSON value."""
    run = tmp_path / "run"
    run.mkdir()
    for name, content in (("plan.json", plan), ("selection.json", selection)):
        (run / name).write_bytes(content if isinstance(content, bytes) else json.dumps(content).encode("utf-8"))
    return run


def _finalize(run: Path, tmp_path: Path) -> int:
    return run_hmm_experiments.main(["finalize", "--run-dir", str(run), "--evaluator-root", str(tmp_path / "ev")])


def _audit(command: Callable[[list[str] | None], int]) -> Callable[[Path, Path], int]:
    def call(run: Path, tmp_path: Path) -> int:
        return command(["--run-dir", str(run), "--evaluator-root", str(tmp_path / "ev"),
                        "--output-dir", str(tmp_path / "out")])

    return call


GATE_COMMANDS = {
    "finalize": _finalize,
    "verify": _audit(verify_hmm_experiments.main),
    "package": _audit(package_hmm_handoff.main),
}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", GATE_COMMANDS)
def test_a_malformed_plan_is_one_error_line_in_every_command_that_passes_the_frozen_gate(
        name: str, kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = frozen_run(tmp_path, BAD[kind], VALID_SELECTION)
    assert GATE_COMMANDS[name](run, tmp_path) == 1
    assert_one_error_line(capsys, tmp_path, "plan.json")
    assert not (tmp_path / "out").exists() and not (run / "final").exists()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", GATE_COMMANDS)
def test_a_malformed_selection_is_one_error_line_in_every_command_that_passes_the_frozen_gate(
        name: str, kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = frozen_run(tmp_path, VALID_LOOKING_PLAN, BAD[kind])
    assert GATE_COMMANDS[name](run, tmp_path) == 1
    assert_one_error_line(capsys, tmp_path, "selection.json")
    assert not (tmp_path / "out").exists() and not (run / "final").exists()


@pytest.fixture
def clean_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real `tune` runs, on a checkout `git status` calls clean (the one under test has uncommitted changes)."""
    monkeypatch.setattr(hmm_experiment_tune, "code_identity", lambda: {"commit": "c", "dirty": False})


@pytest.mark.parametrize("kind", ["not-utf8", "nested"])
def test_tune_with_a_malformed_config_is_one_error_line(
        kind: str, tmp_path: Path, clean_checkout: None, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_bytes(BAD[kind])
    argv = ["tune", "--config", str(config), "--output-dir", str(tmp_path / "out"),
            "--evaluator-root", str(tmp_path / "ev")]
    assert run_hmm_experiments.main(argv) == 1
    assert_one_error_line(capsys, tmp_path, "experiment.yaml")
    assert not (tmp_path / "out").exists()


def test_time_models_names_a_corrupt_hmm_model_once(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The HMM loader starts its message with the file name; the command must not put it there a second time."""
    bad = tmp_path / "broken.json"
    bad.write_bytes(b"{oops")
    assert time_models.main(time_argv(bad, tmp_path / "t.json", tiny_data_root(tmp_path / "data"))) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: broken.json: invalid HMM artifact: ") and err.count("broken.json") == 1


@pytest.mark.parametrize("kind", ["nested", "not-utf8"])
def test_time_models_with_a_malformed_validation_file_is_one_error_line(
        kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = tiny_data_root(tmp_path / "data")
    records = data / "processed" / RELEASE / "mode_A" / "validation.jsonl"
    records.write_bytes(BAD[kind])
    assert time_models.main(time_argv(tiny_hmm(tmp_path), tmp_path / "t.json", data)) == 1
    assert_one_error_line(capsys, tmp_path, "validation.jsonl")
    assert not (tmp_path / "t.json").exists()


@pytest.mark.parametrize("kind", SHAPES)
def test_train_hmm_with_a_wrong_shape_release_manifest_is_one_error_line(
        kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv, _, _, output = setup_run(tmp_path)
    broken = tmp_path / "broken.input"
    broken.write_bytes(BAD[kind])
    assert train_hmm.main([*argv[2:], "--release-manifest", str(broken)]) == 1
    assert_one_error_line(capsys, tmp_path, "broken.input")
    assert not output.exists()


@pytest.mark.parametrize("flag", ["--release-manifest", "--selection-record"])
@pytest.mark.parametrize("kind", SHAPES)
def test_predict_hmm_with_a_wrong_shape_manifest_or_selection_record_is_one_error_line(
        flag: str, kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model = trained_model(tmp_path, capsys)
    broken = tmp_path / "broken.input"
    broken.write_bytes(BAD[kind])
    argv = predict_argv(tmp_path, model, model.parents[1] / "predict.jsonl")
    assert predict_hmm.main([*argv, flag, str(broken)]) == 1
    assert_one_error_line(capsys, tmp_path, "broken.input")
    assert not (tmp_path / "predicted").exists()


@pytest.mark.parametrize("flag", ["--label-config", "--release-manifest"])
@pytest.mark.parametrize("kind", ["nested", "not-utf8", "array"])
def test_train_hmm_with_a_malformed_config_or_manifest_is_one_error_line(
        flag: str, kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv, _, _, output = setup_run(tmp_path)
    broken = tmp_path / "broken.input"
    broken.write_bytes(BAD[kind])
    assert train_hmm.main([*argv[2:], flag, str(broken)]) == 1
    assert_one_error_line(capsys, tmp_path, "broken.input")
    assert not output.exists()


@pytest.mark.parametrize("flag", ["--release-manifest", "--selection-record"])
@pytest.mark.parametrize("kind", ["nested", "not-utf8", "array"])
def test_predict_hmm_with_a_malformed_manifest_or_selection_record_is_one_error_line(
        flag: str, kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model = trained_model(tmp_path, capsys)
    broken = tmp_path / "broken.input"
    broken.write_bytes(BAD[kind])
    argv = predict_argv(tmp_path, model, model.parents[1] / "predict.jsonl")
    assert predict_hmm.main([*argv, flag, str(broken)]) == 1
    assert_one_error_line(capsys, tmp_path, "broken.input")
    assert not (tmp_path / "predicted").exists()


# ---- a config or a plan whose config lacks a value that a later step indexes ----

CONFIG_KEYS = ("processed_dir", "splits_dir", "label_config", "dataset_release_id", "policy_version",
               "experiment_id", "evaluation.repository_commit")
CONFIG_DAMAGE = [(key, how) for key in CONFIG_KEYS for how in ("missing", "empty", "number")] + [("modes", "string")]


def damaged_config(key: str, how: str) -> dict[str, Any]:
    """The real study config with one value missing, emptied, a number or (modes) a string of letters."""
    content = yaml.safe_load((ROOT / "configs/hmm_sponspeech_v2.yaml").read_text(encoding="utf-8"))
    section, _, name = key.rpartition(".")
    holder = content[section] if section else content
    if how == "missing":
        del holder[name]
    else:
        holder[name] = {"empty": "", "number": 5, "string": "ABC"}[how]
    return content


@pytest.mark.parametrize(("key", "how"), CONFIG_DAMAGE)
def test_tune_with_a_config_that_lacks_a_required_value_is_one_error_line_naming_the_key(
        key: str, how: str, tmp_path: Path, clean_checkout: None, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump(damaged_config(key, how)), encoding="utf-8")
    argv = ["tune", "--config", str(config), "--output-dir", str(tmp_path / "out"),
            "--evaluator-root", str(tmp_path / "ev")]
    assert run_hmm_experiments.main(argv) == 1
    assert key in assert_one_error_line(capsys, tmp_path, "experiment.yaml")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(("key", "how"), CONFIG_DAMAGE)
@pytest.mark.parametrize("name", GATE_COMMANDS)
def test_a_plan_whose_config_lacks_a_required_value_is_one_error_line_naming_the_key(
        name: str, key: str, how: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plan = {"config": damaged_config(key, how), "implementation": {}, "evaluator": {}}
    run = frozen_run(tmp_path, plan, VALID_SELECTION)
    assert GATE_COMMANDS[name](run, tmp_path) == 1
    assert key in assert_one_error_line(capsys, tmp_path, "plan.json")
    assert not (tmp_path / "out").exists() and not (run / "final").exists()


# ---- a split checksum manifest that is not UTF-8 or has a line of another shape ----


@pytest.mark.parametrize("kind", ["not-utf8", "bad-line"])
def test_train_hmm_with_a_malformed_split_checksum_manifest_is_one_error_line(
        kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args, _, _, out, options = release_args(tmp_path)
    lines = options["split_manifest"].read_bytes().splitlines(keepends=True)
    options["split_manifest"].write_bytes(b"\xff\xfe" if kind == "not-utf8" else b"".join(lines) + b"oops\n")
    assert train_hmm.main(args[2:]) == 1
    err = assert_one_error_line(capsys, tmp_path, "SHA256SUMS")
    assert err == ("error: SHA256SUMS: not valid UTF-8\n" if kind == "not-utf8"
                   else f"error: SHA256SUMS line {len(lines) + 1}: expected '<sha256>  <path>'\n")
    assert not out.exists()


# ---- a damaged study: its release manifest and its own JSON files ----

RUN = "artifacts/hmm/unit"
WINNER_RUN = f"{RUN}/{{winner_A}}/run.json"  # the folder of mode A's winner is looked up in the study
MANIFEST = f"{PROCESSED}/release_manifest.json"
FINAL = f"{RUN}/final/mode_A/test"
GATED = ("verify", "package", "finalize")
# (file under the study root, the command that reads it, the key whose loss that command must report)
STUDY_FILES = [
    (f"{RUN}/results.json", "verify", ("selection_sha256",)),
    (f"{RUN}/results.json", "package", ("results", "A", "test", "primary")),
    (f"{RUN}/runtime.json", "verify", ("modes", "A", "prediction_seconds")),
    (f"{RUN}/verification/verification.json", "package", ("modes", "A", "candidates_independently_checked")),
    (f"{RUN}/verification/verification.json", "package", ("plan", "sha256")),
    (f"{FINAL}/evaluation.json", "verify", ("inputs", "gold", "sha256")),
    (f"{FINAL}/evaluation.json", "package", ("primary",)),
    (f"{FINAL}/shortcut.evaluation.json", "verify", ("counts", "sequences")),
    (f"{FINAL}/shortcut.evaluation.json", "package", ("primary",)),
    (f"{FINAL}/predictions.manifest.json", "verify", ("inputs", "selection_record", "sha256")),
    (WINNER_RUN, "package", ("training_summary",)),
    (f"{PROCESSED}/stats.json", "verify", ("labels_default_and_primary", "train", "A", "counts")),
    *[(f"{RUN}/selection.json", command, ("mode_selection_sha256", "A")) for command in GATED],
    *[(f"{RUN}/selection.json", command, ("modes", "A", "model")) for command in GATED],
    *[(MANIFEST, command, (key,)) for command in ("tune", "finalize", "verify")
      for key in ("files", "split_files", "source", "release_id")],
]
HOWS = ("object", "array", "null", "missing", "emptied")  # {} [] null; the key gone; the object that holds it empty
STUDY_CASES = [
    pytest.param(relative, command, key_path, how, id=f"{Path(relative).name}-{'.'.join(key_path)}-{command}-{how}")
    for relative, command, key_path in STUDY_FILES for how in HOWS if not (how == "emptied" and len(key_path) == 1)
]


@pytest.fixture
def study(tmp_path_factory: pytest.TempPathFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Study:
    """A private copy of the finished study with the study code patched to it."""
    copy = copy_study(finished_study(tmp_path_factory), tmp_path)
    patch_study(monkeypatch, copy)
    return copy


def without_key(content: dict[str, Any], key_path: tuple[str, ...], how: str) -> dict[str, Any]:
    """A copy of `content` with the value at `key_path` removed (or, for "emptied", the object holding it)."""
    head, *rest = key_path
    if not rest:
        return {k: v for k, v in content.items() if k != head}
    if how == "emptied" and len(rest) == 1:
        return {**content, head: {}}
    return {**content, head: without_key(content[head], tuple(rest), how)}


def damage(path: Path, key_path: tuple[str, ...], how: str) -> None:
    """Replace a JSON file by {}, [] or null, or take a key out of it (or empty the object that holds the key)."""
    whole = {"object": b"{}", "array": b"[]", "null": b"null"}
    if how in whole:
        path.write_bytes(whole[how])
    else:
        path.write_text(json.dumps(without_key(read_json(path), key_path, how)), encoding="utf-8")


def run_study_command(command: str, study: Study) -> int:
    """The command's `main`, run in-process on the study; each writes a new folder at a fixed place."""
    run, evaluator = str(study.run), str(study.evaluator)
    if command == "tune":
        output = str(study.root / "artifacts/hmm/new")
        return run_hmm_experiments.main(
            ["tune", "--config", str(study.config_path), "--output-dir", output, "--evaluator-root", evaluator])
    if command == "finalize":
        return run_hmm_experiments.main(["finalize", "--run-dir", run, "--evaluator-root", evaluator])
    main, output = ((verify_hmm_experiments.main, study.run / "audit-new") if command == "verify"
                    else (package_hmm_handoff.main, study.root / "artifacts/handoff-new"))
    return main(["--run-dir", run, "--evaluator-root", evaluator, "--output-dir", str(output)])


def strip_to_tuned(study: Study) -> None:
    """Take away what finalize would write, leaving the run as tuning left it."""
    for name in ("final", "verification"):
        shutil.rmtree(study.run / name)
    for name in ("results.json", "runtime.json"):
        (study.run / name).unlink()


@pytest.mark.parametrize(("relative", "command", "key_path", "how"), STUDY_CASES)
def test_a_damaged_study_file_is_one_error_line_naming_the_file_and_leaves_no_output(
        relative: str, command: str, key_path: tuple[str, ...], how: str,
        study: Study, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    winner = Path(read_json(study.run / "selection.json")["modes"]["A"]["model"]).parent.as_posix()
    target = study.root / relative.replace("{winner_A}", winner)
    if command == "finalize":
        strip_to_tuned(study)
    damage(target, key_path, how)
    assert run_study_command(command, study) == 1
    err = assert_one_error_line(capsys, tmp_path, target.name, quiet=False)
    if how in ("missing", "emptied"):
        assert ".".join(key_path) in err
    elif how != "object":
        assert "not a JSON object" in err
    left = [study.root / "artifacts/hmm/new", study.run / "audit-new", study.root / "artifacts/handoff-new",
            study.root / "artifacts/handoff-new.zip"]
    assert [path for path in left if path.exists()] == []
    assert command != "finalize" or not (study.run / "final").exists()


# A prediction file is JSON lines, not a JSON object, but the audit indexes its rows the same way.
PREDICTION_FILES = [f"{FINAL}/shortcut.jsonl", f"{RUN}/{{winner_A}}/predictions.jsonl"]
ROW_DAMAGE = {  # what the first row of the file becomes, and the reason the audit gives for it
    "no-id": (lambda row: {k: v for k, v in row.items() if k != "sample_id"}, "missing sample_id"),
    "no-mode": (lambda row: {k: v for k, v in row.items() if k != "label_mode"}, "missing label_mode"),
    "no-labels": (lambda row: {k: v for k, v in row.items() if k != "predicted_labels"}, "missing predicted_labels"),
    "labels-text": (lambda row: {**row, "predicted_labels": "O"}, "predicted_labels is not a JSON list"),
    "array": (lambda row: [], "not a JSON object"),
    "null": (lambda row: None, "not a JSON object"),
}


@pytest.mark.parametrize("damage_name", ROW_DAMAGE)
@pytest.mark.parametrize("relative", PREDICTION_FILES)
def test_a_damaged_prediction_row_is_one_error_line_naming_the_file_and_its_line(
        relative: str, damage_name: str, study: Study, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    winner = Path(read_json(study.run / "selection.json")["modes"]["A"]["model"]).parent.as_posix()
    target = study.root / relative.replace("{winner_A}", winner)
    change, reason = ROW_DAMAGE[damage_name]
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    lines = [json.dumps(row) + "\n" for row in [change(rows[0]), *rows[1:]]]
    target.write_text("".join(lines), encoding="utf-8")
    assert run_study_command("verify", study) == 1
    err = assert_one_error_line(capsys, tmp_path, f"{target.name}:1: {reason}", quiet=False)
    assert err.endswith(f"{reason}\n") and not (study.run / "audit-new").exists()
