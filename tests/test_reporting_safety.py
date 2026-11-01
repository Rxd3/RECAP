"""Safety tests for punct_restore.reporting: no overwriting, no folder names in errors, no crash on odd consoles.

Covers the table builders and time_models. Runs on the tiny generated scores tree (tests/reporting_fixtures.py).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import reporting_fixtures
from reporting_fixtures import (
    RELEASE,
    assert_one_error_line,
    run_main,
    set_json,
    time_argv,
    tiny_data_root,
    tiny_hmm,
    train_rows,
    write_jsonl,
    write_manifests,
)

from punct_restore.models.hmm import HMMModel
from punct_restore.reporting import (
    _cli,
    build_ablation_table,
    build_m3_final_sections,
    build_m3_tables,
    time_models,
)

scores = reporting_fixtures.scores  # the fixture is defined once, there; pytest finds it by this name
# (builder, one of the evaluator JSON files it reads, relative to the scores root)
BUILDERS = [(build_m3_tables, "hmm/shortcut_A_validation.json"),
            (build_m3_final_sections, "threshold/tau_A.json"),
            (build_ablation_table, "ablations/mode_A/improved_minus_window2.json")]
BUILDER_IDS = [module.__name__.rsplit(".", 1)[-1] for module, _ in BUILDERS]


def builder_argv(module: ModuleType, scores: Path, tmp_path: Path, out: Path) -> list[str]:
    extra = ["--manifests", str(write_manifests(tmp_path / "manifests"))] if module is build_m3_final_sections else []
    return ["--scores-root", str(scores), *extra, "--out", str(out)]


# ---------------------------------------------------------------- builders never overwrite
@pytest.mark.parametrize(("module", "target"), BUILDERS, ids=BUILDER_IDS)
def test_a_builder_refuses_to_replace_one_of_its_own_input_files(
        module: ModuleType, target: str, scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    victim = scores / target
    before = victim.read_bytes()
    code, stdout, err = run_main(module, capsys, *builder_argv(module, scores, tmp_path, victim))
    assert (code, stdout) == (1, "")
    assert_one_error_line(err, victim.name, "already exists")
    assert victim.read_bytes() == before


@pytest.mark.parametrize("module", [m for m, _ in BUILDERS], ids=BUILDER_IDS)
def test_an_existing_out_file_is_refused_before_anything_is_read(
        module: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "report.md"
    out.write_text("keep me", encoding="utf-8")
    code, stdout, err = run_main(module, capsys, *builder_argv(module, tmp_path / "nothing", tmp_path, out))
    assert (code, stdout) == (1, "")
    assert_one_error_line(err, "report.md", "already exists")  # not "file not found": nothing was read
    assert out.read_text(encoding="utf-8") == "keep me"


@pytest.mark.parametrize("module", [m for m, _ in BUILDERS], ids=BUILDER_IDS)
def test_an_out_that_is_a_folder_is_refused_before_anything_is_read(
        module: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "a_folder"
    out.mkdir()
    code, stdout, err = run_main(module, capsys, *builder_argv(module, tmp_path / "nothing", tmp_path, out))
    assert (code, stdout) == (1, "")
    assert_one_error_line(err, "a_folder", "is a folder")  # not "file not found": nothing was read
    assert str(tmp_path) not in err
    assert out.is_dir()


@pytest.mark.parametrize("module", [m for m, _ in BUILDERS], ids=BUILDER_IDS)
def test_an_out_in_a_missing_folder_is_refused_before_anything_is_read(
        module: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "no_such_folder" / "report.md"
    code, stdout, err = run_main(module, capsys, *builder_argv(module, tmp_path / "nothing", tmp_path, out))
    assert (code, stdout) == (1, "")
    assert_one_error_line(err, "report.md: folder no_such_folder not found")  # not "file not found": nothing was read
    assert str(tmp_path) not in err
    assert not out.parent.exists()


# (folder, file, entry to change, new value, what the one-line error must name)
DISAGREE = [
    ("scores", "runtime/crf_improved_C.json", ("hardware", "python"), "3.12.0", "Python version"),
    ("scores", "runtime/crf_basic_B.json", ("hardware", "processor"), "Other CPU Family 1", "processor"),
    ("manifests", "crf_mode_c_improved_full_01.manifest.json", ("environment", "cpu_count"), 64, "CPU count"),
    ("manifests", "crf_mode_a_basic_full_01.manifest.json", ("environment", "machine"), "OTHER", "machine"),
    ("scores", "leakage/crf_A_test_official_all.json", ("counts", "sequences"), 1235, "different numbers"),
]


@pytest.mark.parametrize(("folder", "name", "keys", "value", "needle"), DISAGREE, ids=[d[4] for d in DISAGREE])
def test_final_sections_refuse_inputs_that_disagree_on_a_quoted_fact(
        folder: str, name: str, keys: tuple[str, ...], value: object, needle: str, scores: Path, tmp_path: Path,
        capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "final.md"
    manifests = write_manifests(tmp_path / "manifests")
    set_json((scores if folder == "scores" else manifests) / name, keys, value)
    code, stdout, err = run_main(build_m3_final_sections, capsys, "--scores-root", scores,
                                 "--manifests", manifests, "--out", out)
    assert (code, stdout) == (1, "")
    assert_one_error_line(err, needle)
    assert str(tmp_path) not in err and not out.exists()


PROCESSOR, CPUS = ("hardware", "processor"), ("environment", "cpu_count")
MANIFEST_A = "crf_mode_a_basic_full_01.manifest.json"
# (folder, file, entry to change, bad value, the one-line error must say)
BAD_FACTS = [
    ("scores", "runtime/crf_basic_A.json", PROCESSOR, None, "crf_basic_A.json: hardware.processor must be a string"),
    ("scores", "runtime/crf_basic_B.json", PROCESSOR, "", "crf_basic_B.json: hardware.processor must be a string"),
    ("scores", "runtime/crf_improved_C.json", PROCESSOR, 7, "crf_improved_C.json: hardware.processor must be a string"),
    ("scores", "runtime/crf_improved_B.json", ("hardware", "python"), None,
     "crf_improved_B.json: hardware.python must be a string"),
    ("scores", "runtime/crf_improved_A.json", ("hardware",), None,
     "crf_improved_A.json: hardware.processor must be a string"),
    ("manifests", MANIFEST_A, ("environment", "machine"), None, f"{MANIFEST_A}: environment.machine must be a string"),
    ("manifests", MANIFEST_A, CPUS, "8", f"{MANIFEST_A}: environment.cpu_count must be a positive integer"),
    ("manifests", MANIFEST_A, CPUS, 8.5, f"{MANIFEST_A}: environment.cpu_count must be a positive integer"),
    ("manifests", MANIFEST_A, CPUS, 0, f"{MANIFEST_A}: environment.cpu_count must be a positive integer"),
    ("manifests", MANIFEST_A, CPUS, True, f"{MANIFEST_A}: environment.cpu_count must be a positive integer"),
]


@pytest.mark.parametrize(("folder", "name", "keys", "value", "message"), BAD_FACTS,
                         ids=[f"{d[2][-1]}={d[3]!r}" for d in BAD_FACTS])
def test_final_sections_refuse_a_quoted_fact_of_the_wrong_type(
        folder: str, name: str, keys: tuple[str, ...], value: object, message: str, scores: Path,
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "final.md"
    manifests = write_manifests(tmp_path / "manifests")
    set_json((scores if folder == "scores" else manifests) / name, keys, value)
    code, stdout, err = run_main(build_m3_final_sections, capsys, "--scores-root", scores,
                                 "--manifests", manifests, "--out", out)
    assert (code, stdout) == (1, "")
    assert_one_error_line(err, message)  # a named complaint, not an AttributeError or "wrong type"
    assert str(tmp_path) not in err and not out.exists()


def test_write_new_text_keeps_the_platform_newline_and_never_replaces_a_file(tmp_path: Path) -> None:
    out = tmp_path / "t.md"
    _cli.write_new_text(out, "a\nb\n")
    assert out.read_bytes() == b"a" + os.linesep.encode() + b"b" + os.linesep.encode()
    with pytest.raises(FileExistsError):
        _cli.write_new_text(out, "x")
    assert out.read_bytes().startswith(b"a")


# ---------------------------------------------------------------- time_models never overwrites
def fit_options(train: Path, ids: Path) -> list[str]:
    return ["--fit-train", str(train), "--fit-ids", str(ids), "--smoothing", "0.1", "--min-token-count", "1"]


@pytest.mark.parametrize("victim_flag", ["--model", "--fit-train", "--fit-ids"])
def test_time_models_refuses_an_out_that_is_one_of_its_inputs(
        victim_flag: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model, data = tiny_hmm(tmp_path), tiny_data_root(tmp_path / "data")
    train = write_jsonl(tmp_path / "train.jsonl", train_rows())
    ids = tmp_path / "train.ids"
    ids.write_text("t1\nt2\n", encoding="utf-8")
    victim = {"--model": model, "--fit-train": train, "--fit-ids": ids}[victim_flag]
    before = victim.read_bytes()
    code = time_models.main(time_argv(model, victim, data, *fit_options(train, ids)))
    err = capsys.readouterr().err
    assert code == 1
    assert_one_error_line(err, "--out", victim_flag, "same file")
    assert victim.read_bytes() == before
    assert str(tmp_path) not in err


def test_time_models_refuses_an_existing_out_before_loading_the_model(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "t.json"
    out.write_text("keep me", encoding="utf-8")
    assert time_models.main(time_argv(tmp_path / "gone.json", out, tmp_path)) == 1
    assert_one_error_line(capsys.readouterr().err, "t.json", "already exists")  # not "model file not found"
    assert out.read_text(encoding="utf-8") == "keep me"


def test_time_models_refuses_an_out_that_is_a_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "a_folder"
    out.mkdir()
    assert time_models.main(time_argv(tiny_hmm(tmp_path), out, tiny_data_root(tmp_path / "data"))) == 1
    assert_one_error_line(capsys.readouterr().err, "a_folder", "is a folder")
    assert out.is_dir()


def test_time_models_refuses_an_out_in_a_missing_folder_before_loading_or_timing_anything(
        tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    load, predict = time_models.load_model, HMMModel.predict
    monkeypatch.setattr(time_models, "load_model", lambda cls, path: events.append("load") or load(cls, path))
    monkeypatch.setattr(HMMModel, "predict", lambda self, recs: events.append("predict") or predict(self, recs))
    out = tmp_path / "no_such_folder" / "t.json"
    assert time_models.main(time_argv(tiny_hmm(tmp_path), out, tiny_data_root(tmp_path / "data"))) == 1
    err = capsys.readouterr().err
    assert_one_error_line(err, "t.json: folder no_such_folder not found")
    assert str(tmp_path) not in err and events == []


# ---------------------------------------------------------------- time_models validation records
@pytest.mark.parametrize("record", [
    {"sample_id": "v1"}, {"sample_id": "v1", "tokens": None}, {"sample_id": "v1", "tokens": []},
    {"sample_id": "v1", "tokens": "hello there"}, {"sample_id": "v1", "tokens": ["ok", ""]},
    {"sample_id": "v1", "tokens": ["ok", 3]}],
    ids=["missing", "null", "empty", "a-string", "empty-token", "number-token"])
def test_time_models_reports_a_validation_record_without_usable_tokens(
        record: dict[str, object], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = tiny_data_root(tmp_path / "data")
    write_jsonl(data / "processed" / RELEASE / "mode_A" / "validation.jsonl", [record])
    assert time_models.main(time_argv(tiny_hmm(tmp_path), tmp_path / "t.json", data)) == 1
    assert_one_error_line(capsys.readouterr().err, "validation.jsonl", "non-empty tokens list")
    assert not (tmp_path / "t.json").exists()


# ---------------------------------------------------------------- no folder names in errors
def crf_argv(model: Path, tmp_path: Path) -> list[str]:
    return time_argv(model, tmp_path / "t.json", tiny_data_root(tmp_path / "data"), kind="crf")


def assert_names_only(err: str, name: str, folder: Path) -> None:
    assert_one_error_line(err, name)
    assert str(folder) not in err and str(folder).replace("\\", "\\\\") not in err


def test_a_missing_crf_model_is_named_without_its_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert time_models.main(crf_argv(tmp_path / "gone.joblib", tmp_path)) == 1
    assert_names_only(capsys.readouterr().err, "gone.joblib", tmp_path)


def test_a_missing_hmm_model_is_named_without_its_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = time_argv(tmp_path / "gone.json", tmp_path / "t.json", tiny_data_root(tmp_path / "data"))
    assert time_models.main(argv) == 1
    assert_names_only(capsys.readouterr().err, "gone.json", tmp_path)


def test_a_corrupt_crf_model_is_named_without_its_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"this is not a joblib file")
    assert time_models.main(crf_argv(bad, tmp_path)) == 1
    assert_names_only(capsys.readouterr().err, "bad.joblib", tmp_path)


def test_a_loader_error_that_quotes_the_path_is_shortened_to_the_file_name(
        tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    model = tiny_hmm(tmp_path)

    def denied(cls: type[HMMModel], path: str | Path) -> None:  # an OSError repr()s its filename: backslashes double
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(HMMModel, "load_model", classmethod(denied))
    assert time_models.main(time_argv(model, tmp_path / "t.json", tiny_data_root(tmp_path / "data"))) == 1
    assert_names_only(capsys.readouterr().err, "hmm.json", tmp_path)


def test_run_never_prints_an_oserror_text_that_has_no_file_name(capsys: pytest.CaptureFixture[str]) -> None:
    def task() -> None:
        raise OSError("C:\\secret\\x.json is gone")

    assert _cli.run(task) == 1
    err = capsys.readouterr().err
    assert_one_error_line(err, "a file could not be read or written")
    assert "secret" not in err


def test_run_prints_the_reason_of_an_oserror_that_has_no_file_name(capsys: pytest.CaptureFixture[str]) -> None:
    def task() -> None:
        raise OSError(28, "No space left on device")

    assert _cli.run(task) == 1
    assert capsys.readouterr().err == "error: No space left on device\n"


# ---------------------------------------------------------------- odd consoles and python -m runs
def run_module(
        module: str, argv: list[str], extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[bytes]:
    """Run a reporting module as ``python -m`` the way a user does: no PYTHONPATH, so the package comes from the
    install."""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"} | (extra_env or {})
    return subprocess.run([sys.executable, "-m", f"punct_restore.reporting.{module}", *argv], env=env,
                          capture_output=True, check=False, timeout=120)


def test_a_non_ascii_out_path_does_not_crash_a_cp1252_console(scores: Path, tmp_path: Path) -> None:
    out = tmp_path / "\u03a9tables.md"
    done = run_module("build_m3_tables", ["--scores-root", str(scores), "--out", str(out)],
                      {"PYTHONIOENCODING": "cp1252"})
    assert (done.returncode, done.stderr) == (0, b"")
    assert out.is_file() and b"\\u03a9tables.md" in done.stdout


def test_time_models_runs_as_a_module_without_pythonpath(tmp_path: Path) -> None:
    out = tmp_path / "t.json"
    done = run_module("time_models", time_argv(tiny_hmm(tmp_path), out, tiny_data_root(tmp_path / "data")))
    assert (done.returncode, done.stderr) == (0, b"")
    assert out.is_file() and done.stdout.startswith(b"hmm A: ")


def test_final_sections_name_the_file_when_a_hardware_section_is_missing(
        scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runtime = scores / "runtime" / "crf_basic_C.json"
    document = json.loads(runtime.read_text(encoding="utf-8"))
    del document["hardware"]
    runtime.write_text(json.dumps(document), encoding="utf-8")
    out = tmp_path / "final.md"
    code, _, err = run_main(build_m3_final_sections, capsys, "--scores-root", scores,
                            "--manifests", write_manifests(tmp_path / "manifests"), "--out", out)
    assert code == 1 and not out.exists()
    assert_one_error_line(err, "crf_basic_C.json: hardware.processor must be a string")


@pytest.mark.parametrize("field", ["tokens_per_second", "median_seconds", "ms_per_sequence"])
@pytest.mark.parametrize("value", [0, None, "fast"])
def test_final_sections_refuse_a_speed_that_is_not_a_positive_number(
        scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], field: str, value: object) -> None:
    set_json(scores / "runtime" / "crf_basic_A.json", (field,), value)
    out = tmp_path / "final.md"
    code, _, err = run_main(build_m3_final_sections, capsys, "--scores-root", scores,
                            "--manifests", write_manifests(tmp_path / "manifests"), "--out", out)
    assert code == 1 and not out.exists()
    assert_one_error_line(err, f"crf_basic_A.json: {field} must be a positive number")


def test_cli_reports_a_folder_given_as_an_input_file_as_a_folder(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = tmp_path / "fitdir"
    folder.mkdir()

    def task() -> None:
        open(folder, encoding="utf-8").close()  # Windows: PermissionError; POSIX: IsADirectoryError

    assert _cli.run(task) == 1
    assert_one_error_line(capsys.readouterr().err, "fitdir: is a folder, not a file")
