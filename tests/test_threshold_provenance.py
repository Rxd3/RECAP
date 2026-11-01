"""Tests for the provenance that ``recap tune-threshold tune`` records: the split name and the input identities."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from test_threshold import Fixture, _build_fixture, _predict_argv, _sha256, _tune_argv
from test_threshold_priors import _train_argv, _write_train

from punct_restore.cli import tune_threshold

INPUT_KEYS = {"gold", "ids", "masks", "train", "train_ids"}


@pytest.fixture(scope="module")
def mode_a(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    return _build_fixture(tmp_path_factory.mktemp("prov_mode_a"), "A")


@pytest.fixture(scope="module")
def mode_c(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    return _build_fixture(tmp_path_factory.mktemp("prov_mode_c"), "C")


def _tune(argv: list[str], out: Path) -> dict[str, Any]:
    assert tune_threshold.main(argv) == 0
    return json.loads(out.read_text(encoding="utf-8"))


def _identity_of(path: Path) -> dict[str, str]:
    """What file_identity gives a file outside the repository: its folder and file name, and its SHA-256."""
    return {"path": "/".join(path.resolve().parts[-2:]), "sha256": _sha256(path)}


def test_tune_records_the_split_and_the_identity_of_every_input(mode_a: Fixture, tmp_path: Path) -> None:
    out = tmp_path / "tau.json"
    written = _tune(_tune_argv(mode_a, out), out)

    assert set(written) == {"tau", "gamma", "grid", "model_sha256", "split", "inputs"}
    assert written["split"] == "gold"  # the stem of gold.jsonl
    assert set(written["inputs"]) == INPUT_KEYS
    assert written["inputs"]["gold"] == _identity_of(mode_a.gold)
    assert written["inputs"]["ids"] == _identity_of(mode_a.ids)
    assert (written["inputs"]["masks"], written["inputs"]["train"], written["inputs"]["train_ids"]) == (None,) * 3
    assert tmp_path.name not in out.read_text(encoding="utf-8")  # no folder of the output, no absolute path


def test_tune_split_defaults_to_the_gold_file_stem(mode_a: Fixture, tmp_path: Path) -> None:
    gold = tmp_path / "validation.jsonl"
    shutil.copyfile(mode_a.gold, gold)
    fx = Fixture(mode_a.model, gold, mode_a.ids, None)
    out = tmp_path / "tau.json"
    written = _tune(_tune_argv(fx, out), out)
    assert written["split"] == "validation"
    assert written["inputs"]["gold"]["path"].endswith("/validation.jsonl")


def test_tune_split_can_be_named_explicitly(mode_a: Fixture, tmp_path: Path) -> None:
    out = tmp_path / "tau.json"
    written = _tune([*_tune_argv(mode_a, out), "--split", "validation_v2"], out)
    assert written["split"] == "validation_v2"  # not the gold stem "gold"


@pytest.mark.parametrize("name", ["", "   "])
def test_tune_split_must_not_be_blank(
    name: str, mode_a: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "tau.json"
    with pytest.raises(SystemExit) as raised:
        tune_threshold.main([*_tune_argv(mode_a, out), "--split", name])
    assert raised.value.code == 2 and "--split must not be blank" in capsys.readouterr().err
    assert not out.exists()


def test_tune_records_the_masks_identity_in_mode_c(mode_c: Fixture, tmp_path: Path) -> None:
    out = tmp_path / "tau.json"
    written = _tune(_tune_argv(mode_c, out, "0.2:0.8:0.3"), out)
    assert mode_c.masks is not None
    assert written["inputs"]["masks"] == _identity_of(mode_c.masks)
    assert written["inputs"]["train"] is None


def test_tune_records_the_training_file_and_its_id_list(mode_a: Fixture, tmp_path: Path) -> None:
    train, train_ids = _write_train(tmp_path, "A")
    out = tmp_path / "tau.json"
    written = _tune(_train_argv(mode_a, out, train, train_ids), out)
    assert written["inputs"]["train"] == _identity_of(train)
    assert written["inputs"]["train_ids"] == _identity_of(train_ids)
    assert written["inputs"]["gold"] == _identity_of(mode_a.gold)


@pytest.mark.parametrize("failing", ["_provenance", "sha256_file"])
def test_tune_that_cannot_describe_its_inputs_leaves_no_tau_file(
    failing: str, monkeypatch: pytest.MonkeyPatch, mode_a: Fixture, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def broken(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("disk went away")

    monkeypatch.setattr(tune_threshold, failing, broken)
    out = tmp_path / "tau.json"
    assert tune_threshold.main(_tune_argv(mode_a, out)) == 1
    assert capsys.readouterr().err.startswith("error: ")
    assert not out.exists()  # not even an empty file


def test_predict_reads_new_and_old_tau_files_alike(mode_a: Fixture, tmp_path: Path) -> None:
    new_tau = tmp_path / "tau_new.json"
    data = _tune(_tune_argv(mode_a, new_tau), new_tau)
    assert {"split", "inputs"} <= set(data)  # the new file really carries the extra keys
    old_tau = tmp_path / "tau_old.json"  # the same file as written before provenance existed
    old_tau.write_text(json.dumps({k: v for k, v in data.items() if k not in ("split", "inputs")}), encoding="utf-8")

    results = {}
    for tag, tau_file in (("new", new_tau), ("old", old_tau)):
        out, manifest = tmp_path / f"{tag}.jsonl", tmp_path / f"{tag}.man.json"
        assert tune_threshold.main(_predict_argv(mode_a, tau_file, out, manifest)) == 0
        results[tag] = out.read_text(encoding="utf-8")
    assert results["new"] == results["old"] != ""
