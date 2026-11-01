"""A label or token that is not text, in any input file of a command, is ONE stderr line and exit 1, no traceback.

Every case runs a command's ``main(argv)`` in-process on a tiny valid fixture (the fixtures of the command's own tests)
and corrupts one field of every record of the file(s) it reads: the first item of the ``tokens`` / ``punct_labels`` /
``case_labels`` / ``predicted_labels`` list becomes ``null``, a number, an object or a list. A command that does not
validate the field before using it raises KeyError, AttributeError or TypeError instead, and the case fails with it.
A real bug in our own code is not covered here: it must still propagate as a traceback.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from crf_cli_fixtures import create_synthetic_train_jsonl, write_id_file
from eval_fixtures import PUNCT, SCORED, cli_argv, pred_row, valid_case, write_jsonl
from run_ablations_fixtures import RELEASE as ABLATION_RELEASE
from run_ablations_fixtures import _argv as ablation_argv
from run_ablations_fixtures import build_root as ablation_root
from test_build_leakage_subsets import REL as LEAK_REL
from test_build_leakage_subsets import make_root as leakage_root
from test_build_word_clusters import args_for as cluster_args
from test_build_word_clusters import make_root as cluster_root
from test_find_gold_noise import RELEASE as NOISE_RELEASE
from test_find_gold_noise import argv_for as noise_argv
from test_find_gold_noise import build_root as noise_root
from test_hmm_cli import setup_run
from test_simulate_asr_noise import DATA as SIM_DATA
from test_simulate_asr_noise import LISTED_C, SAMPLES, make_sample, write_root
from test_simulate_asr_noise import argv_for as sim_argv
from test_threshold import Fixture, _build_fixture, _predict_argv
from test_train_hidden_event_lm_cli import _build as hidden_event_case

from punct_restore.cli import (
    build_leakage_subsets,
    build_word_clusters,
    evaluate_models,
    factor_predictions,
    find_gold_noise,
    predict_hmm,
    project_joint,
    run_ablations,
    simulate_asr_noise,
    train_crf,
    train_hidden_event_lm,
    train_hmm,
    tune_threshold,
)
from punct_restore.hashing import sha256_file

NOT_A_LIST = [pytest.param(None, id="null"), pytest.param(5, id="number"), pytest.param("a b", id="text"),
              pytest.param({"a": 1}, id="object"), pytest.param([], id="empty")]
NOT_TEXT = [pytest.param(None, id="null"), pytest.param(1, id="number"), pytest.param({}, id="object"),
            pytest.param(["x"], id="list")]


@dataclass(frozen=True)
class Shared(Fixture):
    """The tiny mode A CRF with its gold file and ID list, and an HMM trained by the real ``train-hmm``."""

    hmm_model: Path


class Run(NamedTuple):
    main: Callable[[list[str]], int]
    argv: list[str]
    files: tuple[Path, ...]  # the JSONL files the command reads; the corruption goes into every record of each


Builder = Callable[[Path, Shared], Run]


def corrupt(path: Path, field: str, value: Any, *, whole: bool = False) -> None:
    """Put ``value`` in place of the first item of ``field`` (of the whole field) in every record of the JSONL file."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if field not in row:
            continue
        if whole:
            row[field] = value
        elif isinstance(row[field], list) and row[field]:
            row[field][0] = value
    write_jsonl(path, rows)


# ------------------------------------------------------------------------------------------------- the commands


def _hidden_event_train(mode: str) -> Builder:
    def build(folder: Path, _shared: Shared) -> Run:
        case = hidden_event_case(folder, mode)
        return Run(train_hidden_event_lm.main, case.train_argv(mode), (case.train,))
    return build


def _hidden_event_predict(folder: Path, _shared: Shared) -> Run:
    case = hidden_event_case(folder)
    assert train_hidden_event_lm.main(case.train_argv()) == 0
    return Run(train_hidden_event_lm.main, case.predict_argv("--ids", str(case.valid_ids)), (case.valid,))


def _crf_train(mode: str) -> Builder:
    def build(folder: Path, _shared: Shared) -> Run:
        data, ids = folder / "data.jsonl", folder / "train.ids"
        write_id_file(ids, create_synthetic_train_jsonl(data, mode, 4))
        argv = ["train", "--mode", mode, "--input", str(data), "--ids", str(ids), "--output", str(folder / "m.joblib")]
        return Run(train_crf.main, argv, (data,))
    return build


def _crf_predict(folder: Path, crf: Shared) -> Run:
    gold = Path(shutil.copy(crf.gold, folder / "in.jsonl"))
    argv = ["predict", "--model", str(crf.model), "--input", str(gold), "--output", str(folder / "p.jsonl")]
    return Run(train_crf.main, argv, (gold,))


def _tune_gold(folder: Path, crf: Shared) -> Run:
    gold = Path(shutil.copy(crf.gold, folder / "gold.jsonl"))
    argv = ["tune", "--model", str(crf.model), "--gold", str(gold), "--ids", str(crf.ids),
            "--grid", "0.1:0.9:0.4", "--out", str(folder / "tau.json")]
    return Run(tune_threshold.main, argv, (gold,))


def _tune_priors(folder: Path, crf: Shared) -> Run:
    train = Path(shutil.copy(crf.gold, folder / "train.jsonl"))  # the gold is untouched: only the priors read this
    argv = ["tune", "--model", str(crf.model), "--gold", str(crf.gold), "--ids", str(crf.ids),
            "--train", str(train), "--train-ids", str(crf.ids), "--grid", "0.1:0.9:0.4",
            "--out", str(folder / "tau.json")]
    return Run(tune_threshold.main, argv, (train,))


def _tune_predict(folder: Path, crf: Shared) -> Run:
    tau = folder / "tau.json"
    tau.write_text(json.dumps({"tau": 0.3, "model_sha256": sha256_file(crf.model)}), encoding="utf-8")
    source = Path(shutil.copy(crf.gold, folder / "in.jsonl"))
    argv = _predict_argv(crf, tau, folder / "p.jsonl", folder / "p.manifest.json")
    argv[argv.index("--input") + 1] = str(source)
    return Run(tune_threshold.main, argv, (source,))


def _simulate(folder: Path, _shared: Shared) -> Run:
    root = write_root(folder / "data", [make_sample(i) for i in range(SAMPLES)], LISTED_C)
    files = tuple(root / path for path in SIM_DATA.values())
    return Run(simulate_asr_noise.main, sim_argv(root, folder / "out"), files)


def _noise(folder: Path, _shared: Shared) -> Run:
    root = noise_root(folder)
    return Run(find_gold_noise.main, noise_argv(root, folder / "out"),
               (root / "processed" / NOISE_RELEASE / "mode_B" / "validation.jsonl",))


def _clusters(folder: Path, _shared: Shared) -> Run:
    root = cluster_root(folder)
    return Run(build_word_clusters.main, cluster_args(root, folder / "map.json"),
               (root / "processed" / "rel" / "mode_A" / "train.jsonl",))


def _ablations(folder: Path, _shared: Shared) -> Run:
    root = ablation_root(folder)
    mode_a = root / "processed" / ABLATION_RELEASE / "mode_A"
    return Run(run_ablations.main, ablation_argv(root, folder / "out", "--sample", "3"),
               (mode_a / "train.jsonl", mode_a / "validation.jsonl"))


def _hmm_train(folder: Path, _shared: Shared) -> Run:
    argv, train, _, _ = setup_run(folder)
    return Run(train_hmm.main, argv[2:], (train,))  # drop the interpreter and the script


def _hmm_predict(folder: Path, _shared: Shared) -> Run:
    trained = folder / "trained"
    trained.mkdir()
    assert train_hmm.main(setup_run(trained)[0][2:]) == 0
    source = setup_run(folder)[2]
    argv = ["--model", str(trained / "new-run" / "model.json"), "--input", str(source),
            "--output-dir", str(folder / "predicted"), "--all-records"]
    return Run(predict_hmm.main, argv, (source,))


def _evaluate(command: str, damaged: str, mode: str = "A") -> Builder:
    def build(folder: Path, _shared: Shared) -> Run:
        case = valid_case(folder, mode)
        argv = cli_argv(command, case, folder / "out.json", predictions=command != "baseline")
        return Run(evaluate_models.main, argv, (getattr(case, damaged),))
    return build


def _factor(damaged: str) -> Builder:
    def build(folder: Path, _shared: Shared) -> Run:
        gold = valid_case(folder, "C").gold
        files = {
            "punct": write_jsonl(folder / "punct.jsonl", [pred_row(s, "A", PUNCT["A"][s]) for s in SCORED]),
            "case": write_jsonl(folder / "case.jsonl", [pred_row(s, "CASE", ["LOWER"] * len(PUNCT["A"][s]))
                                                       for s in SCORED]),
        }
        argv = ["--punct", str(files["punct"]), "--case", str(files["case"]), "--gold", str(gold),
                "--out", str(folder / "c.jsonl"), "--manifest-out", str(folder / "c.manifest.json")]
        return Run(factor_predictions.main, argv, (files[damaged],))
    return build


def _project(folder: Path, _shared: Shared) -> Run:
    gold = valid_case(folder, "A").gold
    preds = write_jsonl(folder / "c_pred.jsonl", [pred_row(s, "C", [f"{p}|LOWER" for p in PUNCT["A"][s]])
                                                  for s in SCORED])
    argv = ["--predictions", str(preds), "--gold", str(gold),
            "--out", str(folder / "a.jsonl"), "--manifest-out", str(folder / "a.manifest.json")]
    return Run(project_joint.main, argv, (preds,))


# (id, builder, the fields of its files that the command reads)
TOKEN_AND_LABEL_CASES: list[tuple[str, Builder, tuple[str, ...]]] = [
    ("hidden-event-train-A", _hidden_event_train("A"), ("tokens", "punct_labels")),
    ("hidden-event-train-C", _hidden_event_train("C"), ("tokens", "punct_labels", "case_labels")),
    ("hidden-event-predict", _hidden_event_predict, ("tokens",)),
    ("hmm-train", _hmm_train, ("tokens", "punct_labels")),
    ("hmm-predict", _hmm_predict, ("tokens",)),
    ("crf-train-A", _crf_train("A"), ("tokens", "punct_labels")),
    ("crf-train-C", _crf_train("C"), ("tokens", "punct_labels", "case_labels")),
    ("crf-predict", _crf_predict, ("tokens",)),
    ("tune-gold", _tune_gold, ("tokens", "punct_labels", "case_labels")),
    ("tune-priors", _tune_priors, ("punct_labels",)),
    ("tune-predict", _tune_predict, ("tokens",)),
    ("simulate-asr-noise", _simulate, ("tokens", "punct_labels", "case_labels")),
    ("find-gold-noise", _noise, ("tokens", "punct_labels")),
    ("build-word-clusters", _clusters, ("tokens",)),
    ("run-ablations", _ablations, ("tokens", "punct_labels")),
    ("evaluate-score-gold", _evaluate("score", "gold"), ("tokens", "punct_labels", "case_labels")),
    ("evaluate-baseline-gold", _evaluate("baseline", "gold"), ("tokens", "punct_labels", "case_labels")),
    ("evaluate-score-predictions", _evaluate("score", "predictions"), ("predicted_labels",)),
    ("factor-punct", _factor("punct"), ("predicted_labels",)),
    ("factor-case", _factor("case"), ("predicted_labels",)),
    ("project-joint", _project, ("predicted_labels",)),
]
PARAMS = [pytest.param(build, field, id=f"{name}-{field}")
          for name, build, fields in TOKEN_AND_LABEL_CASES for field in fields]


@pytest.fixture(scope="module")
def crf(tmp_path_factory: pytest.TempPathFactory) -> Shared:
    tiny_crf = _build_fixture(tmp_path_factory.mktemp("crf"), "A")
    trained = tmp_path_factory.mktemp("hmm")
    assert train_hmm.main(setup_run(trained)[0][2:]) == 0  # drop the interpreter and the script
    return Shared(tiny_crf.model, tiny_crf.gold, tiny_crf.ids, tiny_crf.masks, trained / "new-run" / "model.json")


def _run_corrupted(
    build: Builder, field: str, value: Any, crf: Shared, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    *, whole: bool,
) -> None:
    folder = tmp_path / "secretfolder"
    folder.mkdir()
    run = build(folder, crf)
    capsys.readouterr()
    for path in run.files:
        corrupt(path, field, value, whole=whole)
    assert run.main(run.argv) == 1
    err = capsys.readouterr().err
    assert len(err.splitlines()) == 1 and err.startswith("error: ") and err.endswith("\n"), err[:300]
    assert "Traceback" not in err and "secretfolder" not in err and str(tmp_path) not in err, err[:300]


@pytest.mark.parametrize("value", NOT_TEXT)
@pytest.mark.parametrize(("build", "field"), PARAMS)
def test_a_value_that_is_not_text_is_one_error_line(
    build: Builder, field: str, value: Any, crf: Shared, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run_corrupted(build, field, value, crf, tmp_path, capsys, whole=False)


@pytest.mark.parametrize("value", NOT_A_LIST)
@pytest.mark.parametrize(("build", "field"), PARAMS)
def test_a_field_that_is_not_a_list_is_one_error_line(
    build: Builder, field: str, value: Any, crf: Shared, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run_corrupted(build, field, value, crf, tmp_path, capsys, whole=True)


@pytest.mark.parametrize("value", NOT_TEXT)
def test_a_group_id_that_is_not_text_is_one_error_line(
    value: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """build-leakage-subsets reads no tokens or labels, only the ``metadata.group_id`` of each record."""
    root = leakage_root(tmp_path / "secretfolder")
    train = root / "processed" / LEAK_REL / "mode_A" / "train.jsonl"
    rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["metadata"]["group_id"] = value
    write_jsonl(train, rows)
    argv = ["--data-root", str(root), "--release", LEAK_REL, "--out", str(tmp_path / "out")]
    assert build_leakage_subsets.main(argv) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: train.jsonl: ID ") and "has no string metadata.group_id" in err, err[:300]
    assert len(err.splitlines()) == 1 and "secretfolder" not in err
