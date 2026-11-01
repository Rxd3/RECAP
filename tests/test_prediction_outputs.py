"""Safe output handling of the prediction CLIs: path collisions, partial output, stdout encoding.

Covers src/punct_restore/prediction_io.py and its two users, punct_restore.cli.train_crf predict and
punct_restore.cli.tune_threshold predict/tune.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from punct_restore import prediction_io
from punct_restore.cli import train_crf, tune_threshold
from punct_restore.evaluation.inputs import Vocab, load_bundle, sha256_file
from punct_restore.evaluation.manifest import check_manifest
from punct_restore.models import crf_training
from punct_restore.models.crf import CRFModel
from punct_restore.paths import REPO_ROOT

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_crf.py"
PREDICTION = {"sample_id": "s1", "model_name": "m", "label_mode": "A", "predicted_labels": ["O"]}
_ROWS = [
    (["well", "i", "think", "so"], ["COMMA", "O", "O", "PERIOD"]),
    (["how", "are", "you"], ["O", "O", "QUESTION"]),
    (["this", "is", "a", "test"], ["O", "O", "O", "PERIOD"]),
    (["hello", "world", "again"], ["COMMA", "O", "PERIOD"]),
]


@dataclass(frozen=True)
class Env:
    model: Path
    gold: Path
    ids: Path
    tau_file: Path


def _records(prefix: str, split: str) -> list[dict[str, Any]]:
    return [
        {"sample_id": f"{prefix}{index}", "split": split, "label_mode": "A", "tokens": tokens,
         "punct_labels": punct, "case_labels": ["LOWER"] * len(tokens)}
        for index, (tokens, punct) in enumerate(_ROWS)
    ]


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A mode A model plus gold JSONL and ids file, built once and copied per test."""
    root = tmp_path_factory.mktemp("template")
    model = CRFModel(label_mode="A", max_iterations=50)
    model.fit(_records("train", "train"))
    model.save_model(root / "m.joblib")
    gold = _records("gold", "val")
    (root / "gold.jsonl").write_text("".join(json.dumps(r) + "\n" for r in gold), encoding="utf-8")
    (root / "gold.ids").write_text("".join(r["sample_id"] + "\n" for r in gold), encoding="utf-8")
    return root


@pytest.fixture
def env(template: Path, tmp_path: Path) -> Env:
    """Fresh copies of the template files, so a test that damages one cannot hurt the next."""
    inputs = tmp_path / "inputs"
    shutil.copytree(template, inputs)
    model = inputs / "m.joblib"
    tau_file = inputs / "tau.json"
    tau_file.write_text(
        json.dumps({"tau": 0.35, "grid": [], "model_sha256": sha256_file(model)}), encoding="utf-8"
    )
    return Env(model, inputs / "gold.jsonl", inputs / "gold.ids", tau_file)


def _crf_argv(env: Env, output: Path, manifest: Path | None = None) -> list[str]:
    argv = ["predict", "--model", str(env.model), "--input", str(env.gold),
            "--ids", str(env.ids), "--output", str(output)]
    return argv + (["--manifest-out", str(manifest)] if manifest is not None else [])


def _tune_predict_argv(env: Env, out: Path, manifest: Path) -> list[str]:
    return ["predict", "--model", str(env.model), "--input", str(env.gold), "--ids", str(env.ids),
            "--tau-file", str(env.tau_file), "--out", str(out), "--manifest-out", str(manifest)]


def _one_error_line(captured_err: str) -> str:
    lines = captured_err.splitlines()
    assert len(lines) == 1, captured_err
    return lines[0]


# ---------------------------------------------------------------- train_crf predict collisions


def test_crf_predict_refuses_output_equal_to_manifest(
    env: Env, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    same = tmp_path / "x.json"
    assert train_crf.main(_crf_argv(env, same, same)) == 1
    line = _one_error_line(capsys.readouterr().err)
    assert line.startswith("error: ") and "--output" in line and "--manifest-out" in line
    assert not same.exists()


@pytest.mark.parametrize("slot", ["output", "manifest"])
@pytest.mark.parametrize("victim", ["gold", "model", "ids"])
def test_crf_predict_never_overwrites_its_inputs_even_with_overwrite(
    env: Env, tmp_path: Path, capsys: pytest.CaptureFixture[str], slot: str, victim: str
) -> None:
    target = getattr(env, victim)
    before = target.read_bytes()
    out = target if slot == "output" else tmp_path / "p.jsonl"
    man = target if slot == "manifest" else tmp_path / "p.man.json"
    assert train_crf.main([*_crf_argv(env, out, man), "--overwrite"]) == 1
    assert target.read_bytes() == before
    assert _one_error_line(capsys.readouterr().err).startswith("error: ")
    assert not [p for p in (tmp_path / "p.jsonl", tmp_path / "p.man.json") if p.exists()]


@pytest.mark.parametrize("clash", ["manifest", "input", "ids"])
def test_crf_train_never_lets_one_output_overwrite_another_or_an_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], clash: str
) -> None:
    train = tmp_path / "train.jsonl"
    ids = tmp_path / "train.ids"
    rows = _records("t", "train")
    train.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ids.write_text("".join(r["sample_id"] + "\n" for r in rows), encoding="utf-8")
    model = tmp_path / "m.joblib"
    target = {"manifest": model, "input": train, "ids": ids}[clash]
    before = target.read_bytes() if target.exists() else None
    output = model if clash == "manifest" else target
    argv = ["train", "--mode", "A", "--input", str(train), "--ids", str(ids),
            "--output", str(output), "--manifest-out", str(model), "--max-iterations", "5",
            "--overwrite"]
    assert train_crf.main(argv) == 1
    line = _one_error_line(capsys.readouterr().err)
    assert line.startswith("error: ") and "same file" in line
    assert (target.read_bytes() if target.exists() else None) == before


@pytest.mark.skipif(os.name != "nt", reason="only Windows drops trailing dots and spaces")
@pytest.mark.parametrize("alias", ["p.jsonl.", "p.jsonl "])
def test_trailing_dot_or_space_aliases_are_one_file_on_windows(
    env: Env, tmp_path: Path, capsys: pytest.CaptureFixture[str], alias: str
) -> None:
    with pytest.raises(ValueError, match="same file"):
        prediction_io.check_new_output_paths(
            {"--out": tmp_path / "p.jsonl", "--manifest-out": tmp_path / alias}, {})
    out_dir = tmp_path / "cli"
    out_dir.mkdir()
    argv = _tune_predict_argv(env, out_dir / "p.jsonl", out_dir / alias)
    assert tune_threshold.main(argv) == 1
    assert "same file" in _one_error_line(capsys.readouterr().err)
    assert list(out_dir.iterdir()) == []


def test_crf_train_never_overwrites_the_label_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "label_sets.yaml"
    config.write_bytes(REPO_ROOT.joinpath("configs", "label_sets.yaml").read_bytes())
    before = config.read_bytes()
    argv = [*_train_argv(tmp_path, config, tmp_path / "m.json"),
            "--config-path", str(config), "--overwrite"]
    assert train_crf.main(argv) == 1
    assert "--config-path" in _one_error_line(capsys.readouterr().err)
    assert config.read_bytes() == before


def test_empty_output_path_is_named_in_the_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"--out: \S+ is a folder"):
        prediction_io.check_new_output_paths({"--out": Path("")}, {})


@pytest.mark.skipif(os.name != "nt", reason="only Windows file names ignore case")
def test_crf_predict_treats_case_variants_as_one_file_on_windows(
    env: Env, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert train_crf.main(_crf_argv(env, tmp_path / "P.jsonl", tmp_path / "p.jsonl")) == 1
    assert "--manifest-out" in _one_error_line(capsys.readouterr().err)
    assert not (tmp_path / "p.jsonl").exists()


# ---------------------------------------------------------------- tune_threshold collisions


def test_tune_predict_collision_is_refused_before_loading_anything(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    same = tmp_path / "x.json"
    ghost = str(tmp_path / "ghost")
    argv = ["predict", "--model", ghost + ".joblib", "--input", ghost + ".jsonl", "--ids", ghost,
            "--tau-file", ghost + ".json", "--out", str(same), "--manifest-out", str(same)]
    assert tune_threshold.main(argv) == 1
    line = _one_error_line(capsys.readouterr().err)
    assert "--out" in line and "--manifest-out" in line and "same file" in line
    assert not same.exists()


@pytest.mark.parametrize("slot", ["out", "manifest-out"])
@pytest.mark.parametrize("victim", ["model", "gold", "ids", "tau_file"])
def test_tune_predict_never_overwrites_its_inputs(
    env: Env, tmp_path: Path, capsys: pytest.CaptureFixture[str], slot: str, victim: str
) -> None:
    target = getattr(env, victim)
    before = target.read_bytes()
    out = target if slot == "out" else tmp_path / "p.jsonl"
    man = target if slot == "manifest-out" else tmp_path / "p.man.json"
    assert tune_threshold.main(_tune_predict_argv(env, out, man)) == 1
    line = _one_error_line(capsys.readouterr().err)
    assert f"--{slot}" in line and "same file" in line
    assert target.read_bytes() == before
    assert not [p for p in (tmp_path / "p.jsonl", tmp_path / "p.man.json") if p.exists()]


@pytest.mark.parametrize("victim", ["--model", "--gold", "--ids", "--masks"])
def test_tune_never_overwrites_its_inputs(
    env: Env, tmp_path: Path, capsys: pytest.CaptureFixture[str], victim: str
) -> None:
    masks = tmp_path / "masks.jsonl"
    masks.write_text("{}\n", encoding="utf-8")
    paths = {"--model": env.model, "--gold": env.gold, "--ids": env.ids, "--masks": masks}
    before = paths[victim].read_bytes()
    argv = ["tune", *(arg for flag, path in paths.items() for arg in (flag, str(path))),
            "--out", str(paths[victim])]
    assert tune_threshold.main(argv) == 1
    line = _one_error_line(capsys.readouterr().err)
    assert "--out" in line and victim in line and "same file" in line
    assert paths[victim].read_bytes() == before


# ---------------------------------------------------------------- outputs that are folders


def _train_argv(tmp_path: Path, output: Path, manifest: Path) -> list[str]:
    train, ids = tmp_path / "train.jsonl", tmp_path / "train.ids"
    rows = _records("t", "train")
    train.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ids.write_text("".join(r["sample_id"] + "\n" for r in rows), encoding="utf-8")
    return ["train", "--mode", "A", "--input", str(train), "--ids", str(ids),
            "--output", str(output), "--manifest-out", str(manifest), "--max-iterations", "5"]


@pytest.mark.parametrize("command", ["train", "predict"])
@pytest.mark.parametrize("flag", ["output", "manifest"])
def test_crf_refuses_an_output_that_is_a_folder_even_with_overwrite(
    env: Env, tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str, flag: str
) -> None:
    folder, other = tmp_path / "taken", tmp_path / "other.out"
    folder.mkdir()
    out = folder if flag == "output" else other
    man = folder if flag == "manifest" else other.with_suffix(".man.json")
    argv = _train_argv(tmp_path, out, man) if command == "train" else _crf_argv(env, out, man)
    assert train_crf.main([*argv, "--overwrite"]) == 1
    line = _one_error_line(capsys.readouterr().err)
    assert line.startswith("error: ") and "taken is a folder, not a file" in line
    assert folder.is_dir() and not list(folder.iterdir())
    assert not other.exists() and not other.with_suffix(".man.json").exists()


@pytest.mark.parametrize("flag", ["out", "manifest-out"])
def test_tune_predict_refuses_an_output_that_is_a_folder(
    env: Env, tmp_path: Path, capsys: pytest.CaptureFixture[str], flag: str
) -> None:
    folder = tmp_path / "taken"
    folder.mkdir()
    out = folder if flag == "out" else tmp_path / "p.jsonl"
    man = folder if flag == "manifest-out" else tmp_path / "p.man.json"
    assert tune_threshold.main(_tune_predict_argv(env, out, man)) == 1
    line = _one_error_line(capsys.readouterr().err)
    assert line.startswith("error: ") and f"--{flag}: taken is a folder, not a file" in line
    assert folder.is_dir() and not list(folder.iterdir())
    assert not (tmp_path / "p.jsonl").exists() and not (tmp_path / "p.man.json").exists()


# ---------------------------------------------------------------- no partial output


def test_crf_predict_manifest_under_a_file_leaves_no_predictions(
    env: Env, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    notes, out = tmp_path / "notes.txt", tmp_path / "p.jsonl"
    notes.write_text("a file, not a folder", encoding="utf-8")
    assert train_crf.main(_crf_argv(env, out, notes / "m.json")) == 1
    assert not out.exists()
    line = _one_error_line(capsys.readouterr().err)
    assert "--manifest-out: cannot create m.json: notes.txt is not a folder" in line


def test_tune_predict_manifest_under_a_file_leaves_no_predictions_and_allows_a_rerun(
    env: Env, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    notes, out = tmp_path / "notes.txt", tmp_path / "p.jsonl"
    notes.write_text("a file, not a folder", encoding="utf-8")
    assert tune_threshold.main(_tune_predict_argv(env, out, notes / "m.json")) == 1
    assert not out.exists()
    line = _one_error_line(capsys.readouterr().err)
    assert "--manifest-out: cannot create m.json: notes.txt is not a folder" in line

    assert tune_threshold.main(_tune_predict_argv(env, out, tmp_path / "p.man.json")) == 0
    assert out.is_file() and (tmp_path / "p.man.json").is_file()


def test_tune_out_under_a_file_is_refused_before_loading(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("a file, not a folder", encoding="utf-8")
    argv = ["tune", "--model", str(tmp_path / "ghost.joblib"), "--gold", "g", "--ids", "i",
            "--out", str(notes / "tau.json")]
    assert tune_threshold.main(argv) == 1
    assert "--out: cannot create tau.json: notes.txt is not a folder" in _one_error_line(
        capsys.readouterr().err
    )


def test_crf_predict_prints_success_lines_only_after_both_files_are_written(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def disk_full(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(prediction_io, "write_prediction_manifest", disk_full)
    out, man = tmp_path / "p.jsonl", tmp_path / "p.man.json"
    assert train_crf.main(_crf_argv(env, out, man)) == 1
    captured = capsys.readouterr()
    assert "Predictions exported" not in captured.out and "Manifest written" not in captured.out
    assert _one_error_line(captured.err) == "error: could not write prediction outputs: disk full"
    assert not out.exists() and not man.exists()


def test_write_prediction_outputs_removes_its_files_when_the_manifest_fails(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, man = tmp_path / "p.jsonl", tmp_path / "p.man.json"

    def fail_after_starting(manifest_path: Path, **_kwargs: Any) -> None:
        manifest_path.write_text("{", encoding="utf-8")
        raise OSError("disk full")

    monkeypatch.setattr(prediction_io, "write_prediction_manifest", fail_after_starting)
    with pytest.raises(OSError, match="disk full"):
        prediction_io.write_prediction_outputs(
            [PREDICTION], out, man, input_path=env.gold, model_name="m", label_mode="A",
            ordered_labels=["O"],
        )
    assert not out.exists() and not man.exists()


def test_write_prediction_outputs_keeps_an_untouched_manifest_when_predictions_fail(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, man = tmp_path / "p.jsonl", tmp_path / "p.man.json"
    man.write_text("keep me", encoding="utf-8")

    def disk_full(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(prediction_io, "write_predictions_jsonl", disk_full)
    with pytest.raises(OSError, match="disk full"):
        prediction_io.write_prediction_outputs(
            [PREDICTION], out, man, input_path=env.gold, model_name="m", label_mode="A",
            ordered_labels=["O"],
        )
    assert man.read_text(encoding="utf-8") == "keep me"


_WRITE_KWARGS: dict[str, Any] = {"model_name": "m", "label_mode": "A", "ordered_labels": ["O"]}


def test_write_prediction_outputs_removes_the_predictions_when_the_manifest_fails_with_any_error(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, man = tmp_path / "p.jsonl", tmp_path / "p.man.json"

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("boom")

    monkeypatch.setattr(prediction_io, "write_prediction_manifest", boom)
    with pytest.raises(ValueError, match="boom"):
        prediction_io.write_prediction_outputs(
            [PREDICTION], out, man, input_path=env.gold, **_WRITE_KWARGS
        )
    assert not out.exists() and not man.exists()


def test_write_prediction_outputs_leaves_no_file_when_a_prediction_cannot_be_encoded(
    env: Env, tmp_path: Path
) -> None:
    out = tmp_path / "p.jsonl"
    with pytest.raises(UnicodeEncodeError):
        prediction_io.write_prediction_outputs(
            [{**PREDICTION, "sample_id": "\ud800"}], out, None, input_path=env.gold,
            **_WRITE_KWARGS,
        )
    assert not out.exists()


def test_write_prediction_outputs_reraises_the_original_error_when_cleanup_fails(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "taken"
    folder.mkdir()

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("boom")

    monkeypatch.setattr(prediction_io, "write_predictions_jsonl", boom)
    with pytest.raises(OSError, match="boom"):  # not the PermissionError of unlinking a folder
        prediction_io.write_prediction_outputs(
            [PREDICTION], folder, None, input_path=env.gold, **_WRITE_KWARGS
        )
    assert folder.is_dir()


def test_crf_predict_reports_an_unencodable_sample_id_as_one_line_and_leaves_no_file(
    env: Env, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lone = tmp_path / "lone.jsonl"
    lone.write_text('{"sample_id": "\\ud800", "tokens": ["well", "so"]}\n', encoding="utf-8")
    out, man = tmp_path / "p.jsonl", tmp_path / "p.man.json"
    argv = ["predict", "--model", str(env.model), "--input", str(lone),
            "--output", str(out), "--manifest-out", str(man)]
    assert train_crf.main(argv) == 1
    captured = capsys.readouterr()
    assert _one_error_line(captured.err).startswith("error: could not write prediction outputs: ")
    assert "Predictions exported" not in captured.out
    assert not out.exists() and not man.exists()


def test_tune_predict_reports_an_unencodable_sample_id_as_one_line_and_leaves_no_file(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    read_observations = tune_threshold.read_observations

    def with_lone_surrogate(input_path: Path, ids_path: Path | None) -> list[dict[str, Any]]:
        first, *rest = read_observations(input_path, ids_path)  # an ids file cannot hold one
        return [{**first, "sample_id": "\ud800"}, *rest]

    monkeypatch.setattr(tune_threshold, "read_observations", with_lone_surrogate)
    out, man = tmp_path / "p.jsonl", tmp_path / "p.man.json"
    assert tune_threshold.main(_tune_predict_argv(env, out, man)) == 1
    assert _one_error_line(capsys.readouterr().err).startswith("error: ")
    assert not out.exists() and not man.exists()


def test_write_prediction_outputs_writes_both_files_or_only_the_predictions(
    env: Env, tmp_path: Path
) -> None:
    both, man = tmp_path / "both.jsonl", tmp_path / "both.man.json"
    kwargs: dict[str, Any] = {"input_path": env.gold, "model_name": "m", "label_mode": "A",
                              "ordered_labels": ["O"]}
    prediction_io.write_prediction_outputs([PREDICTION], both, man, **kwargs)
    prediction_io.write_prediction_outputs([PREDICTION], tmp_path / "only.jsonl", None, **kwargs)

    expected = (json.dumps(PREDICTION) + "\n").replace("\n", os.linesep).encode("utf-8")
    assert both.read_bytes() == expected == (tmp_path / "only.jsonl").read_bytes()
    assert json.loads(man.read_text(encoding="utf-8"))["predictions_sha256"] == sha256_file(both)
    assert sorted(p.name for p in tmp_path.glob("*.man.json")) == ["both.man.json"]


# ---------------------------------------------------------------- stdout encoding


def test_crf_predict_survives_an_output_path_the_console_cannot_encode(
    env: Env, tmp_path: Path, vocab: Vocab
) -> None:
    environment = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    environment["PYTHONIOENCODING"] = "cp1252"
    out, man = tmp_path / "Ωout.jsonl", tmp_path / "Ωout.manifest.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *_crf_argv(env, out, man)],
        cwd=tmp_path, env=environment, capture_output=True, encoding="utf-8", errors="replace",
        timeout=300, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert out.is_file() and man.is_file()
    check_manifest(man, load_bundle(env.gold, env.ids, out, None, vocab), vocab)


def test_main_tolerates_a_stdout_without_reconfigure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert train_crf.main(["inspect", "--model", str(tmp_path / "ghost.joblib")]) == 1
    assert "not found" in capsys.readouterr().err


# ---------------------------------------------------------------- git_status_porcelain


def test_git_status_porcelain_decodes_utf8_without_failing_on_bad_bytes(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_run(args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs)
        stdout = b"\xff M x.py\n".decode(kwargs["encoding"], kwargs["errors"])
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(crf_training.subprocess, "run", fake_run)
    assert crf_training.git_status_porcelain() == "� M x.py\n"
    assert seen["encoding"] == "utf-8" and seen["errors"] == "replace"
    assert "text" not in seen


def test_git_status_porcelain_never_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    empty = subprocess.CompletedProcess(["git"], 0, stdout=None, stderr=None)
    monkeypatch.setattr(crf_training.subprocess, "run", lambda *_a, **_k: empty)
    assert crf_training.git_status_porcelain() == "unknown"


# ---------------------------------------------------------------- prediction_io helpers


def test_same_path_resolves_dots_and_ignores_case_only_where_the_platform_does(
    tmp_path: Path,
) -> None:
    assert prediction_io.same_path(tmp_path / "a" / ".." / "x.json", tmp_path / "x.json")
    assert not prediction_io.same_path(tmp_path / "x.json", tmp_path / "y.json")
    assert prediction_io.same_path(tmp_path / "X.json", tmp_path / "x.json") == (os.name == "nt")


def test_check_new_output_paths_skips_none_and_accepts_new_files_in_new_folders(
    tmp_path: Path,
) -> None:
    prediction_io.check_new_output_paths(
        {"--output": tmp_path / "new" / "deeper" / "p.jsonl", "--manifest-out": None},
        {"--input": tmp_path / "in.jsonl", "--ids": None},
    )


def test_check_new_output_paths_names_both_flags_of_a_collision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--output and --model are the same file"):
        prediction_io.check_new_output_paths(
            {"--output": tmp_path / "m.joblib", "--manifest-out": tmp_path / "m.json"},
            {"--model": tmp_path / "m.joblib"},
        )
    with pytest.raises(ValueError, match="--output and --manifest-out are the same file"):
        prediction_io.check_new_output_paths(
            {"--output": tmp_path / "x", "--manifest-out": tmp_path / "." / "x"}, {}
        )


def test_check_new_output_paths_rejects_an_output_that_is_an_existing_folder(
    tmp_path: Path,
) -> None:
    (tmp_path / "taken").mkdir()
    with pytest.raises(ValueError, match=r"^--output: taken is a folder, not a file$"):
        prediction_io.check_new_output_paths({"--output": tmp_path / "taken"}, {})


def test_check_new_output_paths_rejects_a_folder_that_is_not_writable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prediction_io.os, "access", lambda *_a, **_k: False)
    with pytest.raises(ValueError, match=r"--output: cannot create p.jsonl: .+ is not writable"):
        prediction_io.check_new_output_paths({"--output": tmp_path / "p.jsonl"}, {})


# ---------------------------------------------------------------- gold-free readers (moved)

_ONE = '{"sample_id": "s", "tokens": ["a"]}'


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (["{not json"], "^Corrupt JSON at line 1: not valid JSON$"),
        (["[" * 100_000], "^Corrupt JSON at line 1: not valid JSON$"),  # RecursionError in json, not a JSON error
        (["[1]"], "^Corrupt JSON at line 1: not a JSON object$"),
        (['{"tokens": ["a"]}'], "^Corrupt JSON at line 1: sample_id must be a nonempty string$"),
        (['{"sample_id": "s", "tokens": []}'], "Invalid tokens at line 1"),
        (['{"sample_id": "s", "tokens": ["a", " "]}'], "Tokens contain invalid elements"),
        ([_ONE, _ONE], "Duplicate sample_id 's' encountered at line 2"),
    ],
)
def test_read_observations_rejects_bad_records(
    tmp_path: Path, lines: list[str], message: str
) -> None:
    path = tmp_path / "in.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        prediction_io.read_observations(path, None)


def test_read_observations_skips_blank_lines_and_reports_a_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "in.jsonl"
    path.write_text("\n" + _ONE + "\n\n", encoding="utf-8")
    assert prediction_io.read_observations(path, None) == [{"sample_id": "s", "tokens": ["a"]}]
    with pytest.raises(FileNotFoundError, match="Input JSONL not found"):
        prediction_io.read_observations(tmp_path / "ghost.jsonl", None)


def test_read_observations_reports_ids_missing_from_the_input(tmp_path: Path) -> None:
    path, ids = tmp_path / "in.jsonl", tmp_path / "x.ids"
    path.write_text(_ONE + "\n", encoding="utf-8")
    ids.write_text("s\nt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing 1 selected IDs"):
        prediction_io.read_observations(path, ids)


def test_read_ids_skips_blanks_and_rejects_missing_empty_and_duplicate_lists(
    tmp_path: Path,
) -> None:
    ids = tmp_path / "x.ids"
    ids.write_text("a\n\nb\n", encoding="utf-8")
    assert prediction_io.read_ids(ids) == ["a", "b"]
    ids.write_text("a\na\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate ID 'a' found at line 2"):
        prediction_io.read_ids(ids)
    ids.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        prediction_io.read_ids(ids)
    with pytest.raises(FileNotFoundError):
        prediction_io.read_ids(tmp_path / "ghost.ids")


# ---------------------------------------------------------------- path errors stay one line


@pytest.mark.parametrize("output", [".", ""])
def test_crf_train_output_without_a_file_name_is_one_error_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], output: str
) -> None:
    argv = _train_argv(tmp_path, tmp_path / "placeholder", tmp_path / "unused.json")
    argv = argv[: argv.index("--manifest-out")] + argv[argv.index("--manifest-out") + 2:]
    argv[argv.index("--output") + 1] = output  # str(Path("")) would be ".", so set it directly
    assert "--manifest-out" not in argv
    assert train_crf.main(argv) == 1
    line = _one_error_line(capsys.readouterr().err)
    assert line.startswith("error: ") and "--output" in line


def _lock(monkeypatch: pytest.MonkeyPatch, locked: Path) -> None:
    """Path.stat raises PermissionError for anything under ``locked`` (an unreadable folder)."""
    real_stat = Path.stat
    prefix = os.path.normcase(str(locked))

    def stat(self: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        if os.path.normcase(str(self)).startswith(prefix):
            raise PermissionError(13, "Permission denied", str(self))
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)


def test_check_new_output_paths_turns_a_permission_error_into_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    locked = tmp_path / "locked"
    _lock(monkeypatch, locked)
    with pytest.raises(ValueError, match="--output") as raised:
        prediction_io.check_new_output_paths({"--output": locked / "p.jsonl"}, {})
    assert "\n" not in str(raised.value)


def test_crf_train_and_predict_report_an_unreadable_output_folder_in_one_line(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    locked = tmp_path / "locked"
    train_argv = _train_argv(tmp_path, locked / "m.joblib", tmp_path / "m.json")
    _lock(monkeypatch, locked)
    assert train_crf.main(train_argv) == 1
    assert "--output" in _one_error_line(capsys.readouterr().err)
    assert train_crf.main(_crf_argv(env, locked / "p.jsonl")) == 1
    assert "--output" in _one_error_line(capsys.readouterr().err)


def test_crf_predict_reports_an_unreadable_model_folder_in_one_line(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    locked = tmp_path / "locked"
    _lock(monkeypatch, locked)
    argv = _crf_argv(env, tmp_path / "p.jsonl")
    argv[argv.index("--model") + 1] = str(locked / "m.joblib")
    assert train_crf.main(argv) == 1
    line = _one_error_line(capsys.readouterr().err)
    assert line == "error: m.joblib: Permission denied"


@pytest.mark.parametrize("flag", ["--model", "--input"])
def test_crf_predict_reports_a_missing_model_or_input_in_one_line(
    env: Env, tmp_path: Path, capsys: pytest.CaptureFixture[str], flag: str
) -> None:
    argv = _crf_argv(env, tmp_path / "p.jsonl")
    argv[argv.index(flag) + 1] = str(tmp_path / "ghost")
    assert train_crf.main(argv) == 1
    assert "not found" in _one_error_line(capsys.readouterr().err)
    assert not (tmp_path / "p.jsonl").exists()
