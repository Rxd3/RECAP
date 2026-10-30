"""Tests for scripts/reporting/crf_convergence.py: loss trace of saved CRF models (tiny fitted models and fakes)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import pytest
from reporting_fixtures import TRAIN, assert_one_error_line, run_main

from punct_restore.evaluation.inputs import sha256_file
from punct_restore.models.crf import CRFModel
from scripts.reporting import crf_convergence

KEYS = ["file", "model_sha256", "model_name", "label_mode", "iterations", "final_loss", "loss_10_before",
        "relative_drop_last_10"]
RECORDS = [{"sample_id": sid, "tokens": tokens, "punct_labels": punct} for sid, tokens, punct in TRAIN]
# 12 iterations: the loss at iteration 12 - 10 = 2 is 100.0, the final loss is 10.0, so the drop is 90%
LOSSES = [200.0, 100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 25.0, 20.0, 10.0]


def fake_model(path: Path, losses: list[Any], *, family: str = "crf", name: str = "fake") -> Path:
    """A saved bundle shaped like CRFModel.save_model, with a hand-written training log."""
    log = SimpleNamespace(iterations=[{"num": number, "loss": loss} for number, loss in enumerate(losses, 1)])
    bundle = {"model_family": family, "model_name": name, "label_mode": "A",
              "backend_state": SimpleNamespace(training_log_=log)}
    joblib.dump(bundle, path)
    return path


def logged_losses(path: Path) -> list[float]:
    """The losses straight from the saved model's own sklearn-crfsuite training log."""
    return [it["loss"] for it in joblib.load(path)["backend_state"].training_log_.iterations]


def run_report(capsys: pytest.CaptureFixture[str], tmp_path: Path, *models: Path) -> list[dict[str, Any]]:
    out = tmp_path / "convergence.json"
    code, _, err = run_main(crf_convergence, capsys, "--models", *models, "--out", out)
    assert (code, err) == (0, "")
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tiny_models(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Two really fitted models: one stopped at 15 iterations, one at 5."""
    folder = tmp_path_factory.mktemp("tiny")
    paths = {"long": folder / "long.joblib", "short": folder / "short.joblib"}
    CRFModel(label_mode="A", max_iterations=15, model_name="tiny_long").fit(RECORDS).save_model(paths["long"])
    CRFModel(label_mode="A", max_iterations=5, model_name="tiny_short").fit(RECORDS).save_model(paths["short"])
    return paths


def test_entry_values_equal_the_models_own_training_log(
    tiny_models: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tiny_models["long"]
    losses = logged_losses(path)
    assert len(losses) >= 11  # the fixture must be long enough for the 10-iteration comparison

    (entry,) = run_report(capsys, tmp_path, path)
    assert list(entry) == KEYS
    assert entry["file"] == "long.joblib"  # the file name, not the path
    assert entry["model_sha256"] == sha256_file(path)
    assert (entry["model_name"], entry["label_mode"]) == ("tiny_long", "A")
    assert entry["iterations"] == len(losses)
    assert entry["final_loss"] == losses[-1]
    assert entry["loss_10_before"] == losses[-11]
    assert entry["relative_drop_last_10"] == (losses[-11] - losses[-1]) / losses[-11]


def test_a_model_with_fewer_than_eleven_iterations_gets_nulls(
    tiny_models: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tiny_models["short"]
    losses = logged_losses(path)
    assert len(losses) < 11

    (entry,) = run_report(capsys, tmp_path, path)
    assert (entry["iterations"], entry["final_loss"]) == (len(losses), losses[-1])
    assert entry["loss_10_before"] is None and entry["relative_drop_last_10"] is None


def test_relative_drop_by_hand(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (entry,) = run_report(capsys, tmp_path, fake_model(tmp_path / "m.joblib", LOSSES))
    assert (entry["iterations"], entry["loss_10_before"], entry["final_loss"]) == (12, 100.0, 10.0)
    assert entry["relative_drop_last_10"] == pytest.approx(0.9)


@pytest.mark.parametrize(("count", "before"), [(10, None), (11, 1000.0), (12, 990.0)])
def test_the_comparison_needs_eleven_iterations(
    count: int, before: float | None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    losses = [1000.0 - 10 * i for i in range(count)]  # 1000, 990, ...: the loss at iteration count - 10 is `before`
    (entry,) = run_report(capsys, tmp_path, fake_model(tmp_path / "m.joblib", losses))
    assert (entry["iterations"], entry["loss_10_before"]) == (count, before)


def test_a_zero_loss_ten_iterations_before_has_no_relative_drop(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (entry,) = run_report(capsys, tmp_path, fake_model(tmp_path / "m.joblib", [0.0] * 12))
    assert entry["loss_10_before"] == 0.0 and entry["relative_drop_last_10"] is None


def test_a_folder_reads_every_joblib_file_sorted(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = tmp_path / "models"
    folder.mkdir()
    fake_model(folder / "b.joblib", LOSSES, name="second")
    fake_model(folder / "a.joblib", LOSSES, name="first")
    (folder / "notes.txt").write_text("not a model", encoding="utf-8")
    entries = run_report(capsys, tmp_path, folder)
    assert [(e["file"], e["model_name"]) for e in entries] == [("a.joblib", "first"), ("b.joblib", "second")]


def test_a_folder_skips_a_sub_folder_named_like_a_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = tmp_path / "models"
    (folder / "sub.joblib").mkdir(parents=True)
    fake_model(folder / "a.joblib", LOSSES)
    entries = run_report(capsys, tmp_path, folder)
    assert [e["file"] for e in entries] == ["a.joblib"]


def test_files_and_folders_can_be_mixed_in_the_given_order(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = tmp_path / "models"
    folder.mkdir()
    fake_model(folder / "x.joblib", LOSSES)
    single = fake_model(tmp_path / "solo.joblib", LOSSES)
    entries = run_report(capsys, tmp_path, single, folder)
    assert [e["file"] for e in entries] == ["solo.joblib", "x.joblib"]


def test_the_report_ends_with_a_newline_and_prints_one_summary_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, out = fake_model(tmp_path / "m.joblib", LOSSES), tmp_path / "c.json"
    code, stdout, _ = run_main(crf_convergence, capsys, "--models", model, "--out", out)
    assert code == 0 and out.read_text(encoding="utf-8").endswith("]\n")
    assert stdout == "wrote 1 model entry to c.json\n"


def test_the_help_and_the_docstring_warn_that_joblib_files_are_pickles() -> None:
    assert "trusted files only" in (crf_convergence.__doc__ or "")
    assert "trusted files only" in crf_convergence.build_parser().format_help()


# ---------------------------------------------------------------- refusals (one line, exit 1, nothing written)
def refuse(capsys: pytest.CaptureFixture[str], out: Path, *models: Path, needle: str) -> None:
    code, stdout, err = run_main(crf_convergence, capsys, "--models", *models, "--out", out)
    assert (code, stdout) == (1, "")
    assert_one_error_line(err, needle)
    assert out.parent.name not in err  # file names only, never a folder
    assert not out.exists()


def test_a_non_crf_bundle_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    hmm = fake_model(tmp_path / "hmm.joblib", LOSSES, family="hmm")
    refuse(capsys, tmp_path / "o.json", hmm, needle="hmm.joblib: not a CRF model file")


def test_a_pickle_that_is_not_a_bundle_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    joblib.dump([1, 2, 3], tmp_path / "list.joblib")
    refuse(capsys, tmp_path / "o.json", tmp_path / "list.joblib", needle="list.joblib: not a CRF model file")


@pytest.mark.parametrize("state", [SimpleNamespace(), SimpleNamespace(training_log_=None),
                                   SimpleNamespace(training_log_=SimpleNamespace()),
                                   SimpleNamespace(training_log_=SimpleNamespace(iterations=[])), None])
def test_a_missing_training_log_is_refused(
    state: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    joblib.dump({"model_family": "crf", "model_name": "n", "label_mode": "A", "backend_state": state},
                tmp_path / "nolog.joblib")
    refuse(capsys, tmp_path / "o.json", tmp_path / "nolog.joblib", needle="nolog.joblib: the model holds no training log")


@pytest.mark.parametrize("bad", [None, "1.5", True, float("nan"), float("inf")])
def test_an_iteration_without_a_finite_loss_is_refused(
    bad: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_model(tmp_path / "bad.joblib", [*LOSSES[:-1], bad])
    refuse(capsys, tmp_path / "o.json", tmp_path / "bad.joblib", needle="bad.joblib: the training log has")


def test_an_unreadable_file_is_refused_by_name_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "junk.joblib").write_bytes(b"this is not a pickle")
    refuse(capsys, tmp_path / "o.json", tmp_path / "junk.joblib", needle="junk.joblib: not a readable joblib file")


def test_a_missing_python_package_is_named_not_called_unreadable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = fake_model(tmp_path / "m.joblib", LOSSES)

    def need_package(_path: Path) -> None:
        raise ModuleNotFoundError("No module named 'sklearn_crfsuite'", name="sklearn_crfsuite")

    monkeypatch.setattr(crf_convergence.joblib, "load", need_package)
    refuse(capsys, tmp_path / "o.json", model, needle="m.joblib: needs the Python package sklearn_crfsuite to open it")


def test_a_system_error_while_opening_reports_the_system_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = fake_model(tmp_path / "m.joblib", LOSSES)

    def denied(path: Path) -> None:
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(crf_convergence.joblib, "load", denied)
    refuse(capsys, tmp_path / "o.json", model, needle="m.joblib: Permission denied")


def test_a_folder_handed_to_the_loader_is_reported_as_a_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "sub.joblib"
    folder.mkdir()
    with pytest.raises(OSError):  # the system error is not relabelled "not a readable joblib file"
        crf_convergence.read_bundle(folder)
    assert crf_convergence.run(lambda: crf_convergence.read_bundle(folder), json_errors=False) == 1
    assert_one_error_line(capsys.readouterr().err, "sub.joblib: is a folder, not a file")


def test_a_folder_named_like_a_model_is_refused_when_passed_directly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "sub.joblib"
    folder.mkdir()
    refuse(capsys, tmp_path / "o.json", folder, needle="sub.joblib: no .joblib files in this folder")


def test_a_missing_model_is_refused_by_name_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run_main(crf_convergence, capsys, "--models", tmp_path / "gone.joblib", "--out", tmp_path / "o.json")
    assert code == 1
    assert_one_error_line(err, "gone.joblib: model file not found")
    assert tmp_path.name not in err


def test_a_folder_without_joblib_files_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "empty").mkdir()
    refuse(capsys, tmp_path / "o.json", tmp_path / "empty", needle="empty: no .joblib files")


def test_an_existing_output_is_never_replaced(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "o.json"
    out.write_text("keep me", encoding="utf-8")
    model = fake_model(tmp_path / "m.joblib", LOSSES)
    code, _, err = run_main(crf_convergence, capsys, "--models", model, "--out", out)
    assert code == 1
    assert_one_error_line(err, "o.json already exists")
    assert out.read_text(encoding="utf-8") == "keep me"


def test_an_output_in_a_missing_folder_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model = fake_model(tmp_path / "m.joblib", LOSSES)
    code, _, err = run_main(crf_convergence, capsys, "--models", model, "--out", tmp_path / "nope" / "o.json")
    assert code == 1
    assert_one_error_line(err, "o.json", "folder nope not found")


def test_an_output_that_is_a_model_is_refused_and_the_model_survives(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = fake_model(tmp_path / "m.joblib", LOSSES)
    before = sha256_file(model)
    code, _, err = run_main(crf_convergence, capsys, "--models", model, "--out", model)
    assert code == 1
    assert_one_error_line(err, "same file")
    assert sha256_file(model) == before


def test_no_arguments_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as raised:
        crf_convergence.main([])
    assert raised.value.code == 2
