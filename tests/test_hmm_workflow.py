"""Data selections must not silently add, drop, reorder or leak training examples."""

import json
import subprocess
import sys
from copy import deepcopy

import pytest
from test_data_contract import build
from test_hmm_cli import ROOT, SCRIPT, jsonl, setup_run

from punct_restore.models.hmm import HMMModel
from punct_restore.models.hmm_workflow import (
    read_id_selection,
    selected_records,
    verify_training_release,
)


def run(args):
    return subprocess.run(args, text=True, capture_output=True, check=False)


def test_training_selection_changes_counts_and_excludes_unselected_vocabulary(tmp_path):
    args, train, _, out = setup_run(tmp_path)
    keep = json.loads(train.read_text())
    excluded = {**keep, "sample_id": "excluded", "tokens": ["never", "learn"]}
    jsonl(train, [excluded, keep])
    result = run(args)
    assert result.returncode == 0, result.stderr
    model = json.loads((out / "model.json").read_text())
    assert model["counts"]["num_sequences"] == 1
    assert "never" not in model["counts"]["vocabulary"]
    report = json.loads((out / "run.json").read_text())
    assert report["selection_counts"] == {
        "input_records": 2,
        "selected_records": 1,
        "selected_tokens": 2,
    }
    assert report["provenance"]["training_ids"]["sha256"]


@pytest.mark.parametrize("text", ["", "\n", "unit-train\n\n", " unit-train\n", "x\nx\n"])
def test_invalid_id_lists_fail_before_any_model_output(tmp_path, text):
    args, _, _, out = setup_run(tmp_path)
    (tmp_path / "train.ids").write_text(text)
    assert run(args).returncode == 2
    assert not out.exists()


def test_missing_id_is_rejected_after_stream_exhaustion(tmp_path):
    args, _, _, out = setup_run(tmp_path)
    (tmp_path / "train.ids").write_text("unit-train\nnot-in-data\n")
    result = run(args)
    assert result.returncode == 2 and "missing" in result.stderr
    assert not out.exists()


@pytest.mark.parametrize("bad", ["duplicate", "held-out", "bad-labels"])
def test_invalid_unselected_record_cannot_hide_behind_a_selection(tmp_path, bad):
    args, train, _, out = setup_run(tmp_path)
    keep = json.loads(train.read_text())
    other = deepcopy(keep)
    other["sample_id"] = "excluded"
    if bad == "duplicate":
        jsonl(train, [keep, other, other])
    else:
        other["split" if bad == "held-out" else "punct_labels"] = (
            "test" if bad == "held-out" else ["INVALID"]
        )
        jsonl(train, [keep, other])
    assert run(args).returncode == 2
    assert not out.exists()


def test_prediction_selection_preserves_input_order_and_strips_privileged_fields(tmp_path):
    source = tmp_path / "predict.jsonl"
    records = [
        {
            "schema_version": 1,
            "sample_id": sid,
            "tokens": ["hi"],
            "punct_labels": ["INVALID"],
            "case_labels": None,
            "metadata": {"original_text": "NEVER EXPOSE"},
        }
        for sid in ["first", "skip", "last"]
    ]
    jsonl(source, records)
    selected = list(selected_records(source, ("last", "first")))
    assert [r["sample_id"] for r in selected] == ["first", "last"]
    assert all(set(r) == {"schema_version", "sample_id", "tokens"} for r in selected)
    assert records[0]["metadata"]["original_text"] == "NEVER EXPOSE"


def test_prediction_id_filter_is_applied_and_recorded_by_cli(tmp_path):
    args, _, prediction, out = setup_run(tmp_path)
    ids = tmp_path / "predict.ids"
    ids.write_text("unit-predict\n")
    other = {"schema_version": 1, "sample_id": "excluded", "tokens": ["ignore"]}
    jsonl(
        prediction,
        [
            other,
            {
                "schema_version": 1,
                "sample_id": "unit-predict",
                "tokens": ["hello"],
                "punct_labels": ["INVALID"],
            },
        ],
    )
    result = run(args + ["--predict-input", str(prediction), "--predict-ids", str(ids)])
    assert result.returncode == 0, result.stderr
    preds = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    assert [r["sample_id"] for r in preds] == ["unit-predict"]
    sidecar = json.loads((out / "predictions.manifest.json").read_text())
    assert sidecar["ids"]["sha256"] and sidecar["selection_counts"]["input_records"] == 2


def test_id_flag_without_prediction_input_is_an_error(tmp_path):
    args, _, _, out = setup_run(tmp_path)
    result = run(args + ["--predict-ids", str(tmp_path / "train.ids")])
    assert result.returncode == 2 and not out.exists()


def release_args(tmp_path, mode="A"):
    processed, splits = build(tmp_path, "release")
    out = tmp_path / "model-run"
    options = {
        "train": processed / f"mode_{mode}/train.jsonl",
        "train_ids": splits / f"default_train/mode_{mode}.ids",
        "split_manifest": splits / "SHA256SUMS",
        "label_config": ROOT / "configs/label_sets.yaml",
        "mode": mode,
        "release_id": "fixture",
        "policy_version": "m1-v2",
    }
    args = [sys.executable, str(SCRIPT)]
    for flag, value in {
        "train": options["train"],
        "train-ids": options["train_ids"],
        "output-dir": out,
        "split-manifest": options["split_manifest"],
        "label-mode": mode,
        "release-manifest": processed / "release_manifest.json",
        "dataset-release-id": "fixture",
        "preprocessing-version": "m1-v2",
        "tokenizer-version": "m1-v2",
        "label-policy-version": "m1-v2",
    }.items():
        args.extend(["--" + flag, str(value)])
    return args, processed, splits, out, options


@pytest.mark.parametrize("mode", ["A", "B", "C"])
def test_real_release_contract_cli_training_and_reload_on_tiny_sources(tmp_path, mode):
    args, _, splits, out, _ = release_args(tmp_path, mode)
    result = run(args)
    assert result.returncode == 0, result.stderr
    ids = read_id_selection(splits / f"default_train/mode_{mode}.ids")
    loaded = HMMModel.load_model(out / "model.json")
    assert loaded.training_summary["num_sequences"] == len(ids)
    assert loaded.metadata["release_manifest"]["sha256"]


@pytest.mark.parametrize(
    "bad",
    [
        "release-id",
        "policy",
        "train-checksum",
        "ids-checksum",
        "wrong-ids-mode",
        "split-index",
        "label-config",
    ],
)
def test_release_identity_and_checksum_failures_stop_training(tmp_path, bad):
    _, processed, splits, _, opts = release_args(tmp_path)
    if bad == "release-id":
        opts["release_id"] = "some-other-release"
    elif bad == "policy":
        opts["policy_version"] = "m1-v1"
    elif bad == "train-checksum":
        opts["train"].write_bytes(opts["train"].read_bytes() + b"\n")
    elif bad == "ids-checksum":
        opts["train_ids"].write_text("wrong\n")
    elif bad == "wrong-ids-mode":
        opts["train_ids"] = splits / "default_train/mode_C.ids"
    elif bad == "split-index":
        opts["split_manifest"].write_text("0" * 64 + "  invalid\n")
    else:
        opts["label_config"] = tmp_path / "unmatched.yaml"
        opts["label_config"].write_text("not-the-release-config")
    with pytest.raises(ValueError):
        verify_training_release(processed / "release_manifest.json", **opts)


def test_old_cli_without_training_selection_fails_instead_of_training_everything(tmp_path):
    args, _, _, out = setup_run(tmp_path)
    index = args.index("--train-ids")
    del args[index : index + 2]
    result = run(args)
    assert result.returncode == 2 and "--train-ids" in result.stderr
    assert not out.exists()
