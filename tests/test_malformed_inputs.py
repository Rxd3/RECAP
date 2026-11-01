"""Regression net: a malformed input file gives ONE stderr line ``error: <file name>: <reason>`` and exit 1.

Every case runs a command's ``main(argv)`` in-process on a tiny valid fixture (the fixtures of the command's own
tests) with exactly one input replaced by a malformed file: JSON nested 100,000 levels deep, bytes that are not UTF-8,
a model file that is not a model, no file at all, or a folder in the file's place. The line must name the file and must
not show the folder or echo the file's text.
A real bug in our own code is not covered here: it must still propagate as a traceback.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pytest
from crf_cli_fixtures import create_synthetic_train_jsonl, write_id_file
from eval_fixtures import CASE, PUNCT, SCORED, cli_argv, pred_row, valid_case, write_jsonl
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
from test_simulate_asr_noise import DATA as SIM_DATA
from test_simulate_asr_noise import LISTED_C, MASKS, SAMPLES, make_sample, write_root
from test_simulate_asr_noise import argv_for as sim_argv
from test_threshold import Fixture, _build_fixture, _predict_argv, _sha256
from test_train_hidden_event_lm_cli import _build as hidden_event_case

from punct_restore.cli import (
    build_leakage_subsets,
    build_word_clusters,
    evaluate_models,
    factor_predictions,
    find_gold_noise,
    prepare_data,
    project_joint,
    run_ablations,
    simulate_asr_noise,
    train_crf,
    train_hidden_event_lm,
    tune_threshold,
)

FOLDER = "secretfolder"  # the folder every fixture lives in; it must never be shown
NESTED = b"[" * 100_000 + b"\n"  # the standard json parser gives up on it with a RecursionError
NOT_UTF8 = b"\xe9\xff not utf-8\n"  # 0xE9 starts a 3-byte sequence that 0xFF breaks
NOT_A_MODEL = b"123"
NESTED_YAML = b"a: " + b"[" * 100_000 + b"\n"  # PyYAML gives up on it with a RecursionError
NOT_YAML = b"a: [unclosed\n"
JSON_ARRAY = b"[1]\n"  # valid JSON, but not an object
NO_VIDEO_ID = b'{"status": 200}\n'  # a metadata row without the video it describes
BAD_MANIFEST_LINE = b"0" * 64 + b" sponspeech_text/x.txt\n"  # one space where two belong
TITLE_NOT_TEXT = b'{"status": 200, "video_id": "v1", "title": 5}\n'


class Env(NamedTuple):
    folder: Path  # a fresh folder named FOLDER for this case
    payload: bytes | str  # the malformed bytes, or "missing" (no file at all) or "folder" (a folder in its place)
    crf: Fixture  # a fitted tiny mode A CRF with its gold file and ID list


class Run(NamedTuple):
    main: Callable[[list[str]], int]
    argv: list[str]
    damaged: Path  # the file the error must name


Builder = Callable[[Env], Run]


def _damage(path: Path, env: Env) -> Path:
    if isinstance(env.payload, str):
        path.unlink(missing_ok=True)
        if env.payload == "folder":
            path.mkdir()
    else:
        path.write_bytes(env.payload)
    return path


# ---------------------------------------------------------------------------------------------------- evaluate-models


def _evaluate(command: str, damaged: str, mode: str = "A") -> Builder:
    def build(env: Env) -> Run:
        case = valid_case(env.folder, mode)
        path = _damage(getattr(case, damaged), env)
        argv = cli_argv(command, case, env.folder / "out.json", predictions=command != "baseline")
        return Run(evaluate_models.main, argv, path)
    return build


def _evaluate_manifest(env: Env) -> Run:
    case = valid_case(env.folder, "A")
    manifest = _damage(env.folder / "preds.manifest.json", env)
    argv = cli_argv("score", case, env.folder / "out.json", "--manifest", str(manifest))
    return Run(evaluate_models.main, argv, manifest)


# ------------------------------------------------------------------------------------------------------ CRF commands


def _tune(damaged: str) -> Builder:
    def build(env: Env) -> Run:
        gold = _damage(env.folder / "gold.jsonl", env) if damaged == "gold" else env.crf.gold
        ids = _damage(env.folder / "val.ids", env) if damaged == "ids" else env.crf.ids
        argv = ["tune", "--model", str(env.crf.model), "--gold", str(gold), "--ids", str(ids),
                "--grid", "0.1:0.9:0.4", "--out", str(env.folder / "tau.json")]
        return Run(tune_threshold.main, argv, gold if damaged == "gold" else ids)
    return build


def _tune_predict(damaged: str) -> Builder:
    def build(env: Env) -> Run:
        tau = env.folder / "tau.json"
        tau.write_text(json.dumps({"tau": 0.3, "model_sha256": _sha256(env.crf.model)}), encoding="utf-8")
        argv = _predict_argv(env.crf, tau, env.folder / "p.jsonl", env.folder / "p.manifest.json")
        path = _damage(env.folder / f"{damaged}.dat", env)
        argv[argv.index("--input" if damaged == "input" else "--tau-file") + 1] = str(path)
        return Run(tune_threshold.main, argv, path)
    return build


def _crf_train(damaged: str) -> Builder:
    def build(env: Env) -> Run:
        data, ids = env.folder / "data.jsonl", env.folder / "train.ids"
        write_id_file(ids, create_synthetic_train_jsonl(data, "A", 4))
        path = _damage(data if damaged == "input" else ids, env)
        argv = ["train", "--mode", "A", "--input", str(data), "--ids", str(ids),
                "--output", str(env.folder / "m.joblib")]
        return Run(train_crf.main, argv, path)
    return build


def _crf_predict(damaged: str) -> Builder:
    def build(env: Env) -> Run:
        model = env.crf.model
        if damaged == "model":
            model = _damage(env.folder / "notamodel.joblib", env)
        data = _damage(env.folder / "in.jsonl", env) if damaged == "input" else env.crf.gold
        argv = ["predict", "--model", str(model), "--input", str(data), "--output", str(env.folder / "p.jsonl")]
        return Run(train_crf.main, argv, model if damaged == "model" else data)
    return build


def _crf_inspect(env: Env) -> Run:
    model = _damage(env.folder / "notamodel.joblib", env)
    return Run(train_crf.main, ["inspect", "--model", str(model)], model)


# --------------------------------------------------------------------------------------- hidden-event language model


def _hidden_event(command: str) -> Builder:
    def build(env: Env) -> Run:
        case = hidden_event_case(env.folder)
        if command == "train":
            return Run(train_hidden_event_lm.main, case.train_argv(input=_damage(case.train, env)), case.train)
        assert train_hidden_event_lm.main(case.train_argv()) == 0
        return Run(train_hidden_event_lm.main, case.predict_argv(input=_damage(case.valid, env)), case.valid)
    return build


# ----------------------------------------------------------------------------------------------- data-root commands


def _leakage(name: str) -> Builder:
    def build(env: Env) -> Run:
        root = leakage_root(env.folder)
        path = _damage(root / "processed" / LEAK_REL / "mode_A" / name, env)
        argv = ["--data-root", str(root), "--release", LEAK_REL, "--out", str(env.folder / "out")]
        return Run(build_leakage_subsets.main, argv, path)
    return build


def _noise(env: Env) -> Run:
    root = noise_root(env.folder)
    path = _damage(root / "processed" / NOISE_RELEASE / "mode_B" / "validation.jsonl", env)
    return Run(find_gold_noise.main, noise_argv(root, env.folder / "out"), path)


def _clusters(env: Env) -> Run:
    root = cluster_root(env.folder)
    path = _damage(root / "processed" / "rel" / "mode_A" / "train.jsonl", env)
    return Run(build_word_clusters.main, cluster_args(root, env.folder / "map.json"), path)


def _ablations(name: str) -> Builder:
    def build(env: Env) -> Run:
        root = ablation_root(env.folder)
        argv = ablation_argv(root, env.folder / "out", "--sample", "3")
        if name == "cluster-map":
            path = _damage(env.folder / "clusters.json", env)
            return Run(run_ablations.main, [*argv, "--cluster-map", str(path)], path)
        path = _damage(root / "processed" / ABLATION_RELEASE / "mode_A" / name, env)
        return Run(run_ablations.main, argv, path)
    return build


def _simulate(target: str) -> Builder:
    def build(env: Env) -> Run:
        root = write_root(env.folder / "data", [make_sample(i) for i in range(SAMPLES)], LISTED_C)
        path = _damage(root / (MASKS if target == "masks" else SIM_DATA[target]), env)
        return Run(simulate_asr_noise.main, sim_argv(root, env.folder / "out"), path)
    return build


# ------------------------------------------------------------------------------- prediction files and prepare-data


def _factor(damaged: str) -> Builder:
    def build(env: Env) -> Run:
        gold = valid_case(env.folder, "C").gold
        files = {
            "punct": write_jsonl(env.folder / "punct.jsonl", [pred_row(s, "A", PUNCT["A"][s]) for s in SCORED]),
            "case": write_jsonl(env.folder / "case.jsonl", [pred_row(s, "CASE", CASE[s]) for s in SCORED]),
        }
        files[damaged] = _damage(files[damaged], env)
        argv = ["--punct", str(files["punct"]), "--case", str(files["case"]), "--gold", str(gold),
                "--out", str(env.folder / "c.jsonl"), "--manifest-out", str(env.folder / "c.manifest.json")]
        return Run(factor_predictions.main, argv, files[damaged])
    return build


def _project(env: Env) -> Run:
    gold = valid_case(env.folder, "A").gold
    preds = _damage(env.folder / "c_pred.jsonl", env)
    argv = ["--predictions", str(preds), "--gold", str(gold),
            "--out", str(env.folder / "a.jsonl"), "--manifest-out", str(env.folder / "a.manifest.json")]
    return Run(project_joint.main, argv, preds)


def _prepare_data_argv(env: Env) -> list[str]:
    return ["build", "--raw-root", str(env.folder / "raw"), "--release-id", "r",
            "--processed-dir", str(env.folder / "p"), "--splits-dir", str(env.folder / "s")]


def _prepare_data_without_manifest(env: Env) -> Run:
    (env.folder / "raw").mkdir(parents=True)  # the raw root exists; only its MANIFEST.sha256 is missing
    return Run(prepare_data.main, _prepare_data_argv(env), Path("MANIFEST.sha256"))


def _prepare_data_raw(env: Env, damaged: str) -> Run:
    """A raw root whose MANIFEST.sha256 and video_metadata_oembed.jsonl are empty, except the one named `damaged`."""
    raw = env.folder / "raw"
    raw.mkdir()
    (raw / "MANIFEST.sha256").write_bytes(b"")
    (raw / "video_metadata_oembed.jsonl").write_bytes(b"")
    (raw / "sponspeech_text").mkdir()  # the extracted transcripts; without them the build stops before it reads these
    return Run(prepare_data.main, _prepare_data_argv(env), _damage(raw / damaged, env))


def _prepare_data_titles(env: Env) -> Run:
    return _prepare_data_raw(env, "video_metadata_oembed.jsonl")


def _prepare_data_manifest(env: Env) -> Run:
    return _prepare_data_raw(env, "MANIFEST.sha256")


def _crf_config(env: Env) -> Run:
    data, ids = env.folder / "data.jsonl", env.folder / "train.ids"
    write_id_file(ids, create_synthetic_train_jsonl(data, "A", 4))
    config = _damage(env.folder / "label_sets.yaml", env)
    argv = ["train", "--mode", "A", "--input", str(data), "--ids", str(ids),
            "--output", str(env.folder / "m.joblib"), "--config-path", str(config)]
    return Run(train_crf.main, argv, config)


# ---------------------------------------------------------------------------------------------------------- the cases

NESTED_CASES: dict[str, Builder] = {
    "evaluate-score-gold": _evaluate("score", "gold"),
    "evaluate-score-predictions": _evaluate("score", "predictions"),
    "evaluate-score-masks": _evaluate("score", "masks", "C"),
    "evaluate-score-manifest": _evaluate_manifest,
    "evaluate-baseline-gold": _evaluate("baseline", "gold"),
    "tune-threshold-tune-gold": _tune("gold"),
    "tune-threshold-predict-input": _tune_predict("input"),
    "tune-threshold-predict-tau-file": _tune_predict("tau"),
    "train-crf-train-input": _crf_train("input"),
    "train-crf-predict-input": _crf_predict("input"),
    "train-hidden-event-lm-train-input": _hidden_event("train"),
    "train-hidden-event-lm-predict-input": _hidden_event("predict"),
    "build-leakage-subsets-split": _leakage("test.jsonl"),
    "build-leakage-subsets-train": _leakage("train.jsonl"),
    "find-gold-noise": _noise,
    "build-word-clusters": _clusters,
    "run-ablations-train": _ablations("train.jsonl"),
    "run-ablations-validation": _ablations("validation.jsonl"),
    "run-ablations-cluster-map": _ablations("cluster-map"),
    "simulate-asr-noise-data": _simulate("B"),
    "simulate-asr-noise-masks": _simulate("masks"),
    "factor-predictions-punct": _factor("punct"),
    "factor-predictions-case": _factor("case"),
    "project-joint-predictions": _project,
    "prepare-data-build-titles": _prepare_data_titles,
}
NOT_UTF8_CASES: dict[str, Builder] = {  # one input per reader family
    "evaluate-score-gold": _evaluate("score", "gold"),
    "evaluate-score-ids": _evaluate("score", "ids"),
    "tune-threshold-tune-ids": _tune("ids"),
    "tune-threshold-predict-input": _tune_predict("input"),
    "tune-threshold-predict-tau-file": _tune_predict("tau"),
    "train-crf-train-input": _crf_train("input"),
    "train-crf-train-ids": _crf_train("ids"),
    "train-crf-predict-input": _crf_predict("input"),
    "train-hidden-event-lm-train-input": _hidden_event("train"),
    "train-hidden-event-lm-predict-input": _hidden_event("predict"),
    "build-leakage-subsets-split": _leakage("test.jsonl"),
    "find-gold-noise": _noise,
    "build-word-clusters": _clusters,
    "run-ablations-train": _ablations("train.jsonl"),
    "run-ablations-cluster-map": _ablations("cluster-map"),
    "simulate-asr-noise-data": _simulate("B"),
    "factor-predictions-punct": _factor("punct"),
    "project-joint-predictions": _project,
    "train-crf-train-config-path": _crf_config,
    "prepare-data-build-titles": _prepare_data_titles,
}
NOT_A_MODEL_CASES: dict[str, Builder] = {
    "train-crf-inspect-model": _crf_inspect,
    "train-crf-predict-model": _crf_predict("model"),
}
TITLES_CASES: dict[str, Builder] = {"prepare-data-build-titles": _prepare_data_titles}
YAML_CASES: dict[str, Builder] = {"train-crf-train-config-path": _crf_config}
MANIFEST_CASES: dict[str, Builder] = {"prepare-data-build-manifest": _prepare_data_manifest}
ALL_CASES: dict[str, Builder] = {**NESTED_CASES, **NOT_UTF8_CASES, **NOT_A_MODEL_CASES, **YAML_CASES, **MANIFEST_CASES}
PARAMS = [pytest.param(build, payload, id=f"{kind}-{name}")
          for kind, payload, cases in (("nested", NESTED, NESTED_CASES), ("not-utf8", NOT_UTF8, NOT_UTF8_CASES),
                                       ("not-a-model", NOT_A_MODEL, NOT_A_MODEL_CASES),
                                       ("nested-yaml", NESTED_YAML, YAML_CASES), ("not-yaml", NOT_YAML, YAML_CASES),
                                       ("json-array", JSON_ARRAY, TITLES_CASES),
                                       ("no-video-id", NO_VIDEO_ID, TITLES_CASES),
                                       ("bad-manifest-line", BAD_MANIFEST_LINE, MANIFEST_CASES),
                                       ("not-utf8", NOT_UTF8, MANIFEST_CASES),
                                       ("title-not-text", TITLE_NOT_TEXT, TITLES_CASES))
          for name, build in cases.items()]
PARAMS += [pytest.param(build, kind, id=f"{kind}-{name}")
           for kind in ("missing", "folder") for name, build in ALL_CASES.items()]


@pytest.fixture(scope="module")
def crf(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    return _build_fixture(tmp_path_factory.mktemp("crf"), "A")


def _error_line(capsys: pytest.CaptureFixture[str]) -> str:
    err = capsys.readouterr().err
    assert len(err.splitlines()) == 1 and err.startswith("error: ") and err.endswith("\n"), err[:300]
    return err


@pytest.mark.parametrize(("build", "payload"), PARAMS)
def test_a_malformed_input_is_one_error_line_naming_the_file(
    build: Builder, payload: bytes | str, crf: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / FOLDER
    folder.mkdir()
    run = build(Env(folder, payload, crf))
    assert run.main(run.argv) == 1
    line = _error_line(capsys)
    assert run.damaged.name in line, line
    assert FOLDER not in line and str(tmp_path) not in line, line
    assert len(line) < 400, "the message must not echo the file's text"


@pytest.mark.parametrize("build", [_crf_inspect, _crf_predict("model")], ids=["inspect", "predict"])
def test_a_file_that_is_not_a_model_is_named_and_its_text_is_not_echoed(
    build: Builder, crf: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / FOLDER
    folder.mkdir()
    run = build(Env(folder, NOT_A_MODEL, crf))
    assert run.main(run.argv) == 1
    assert _error_line(capsys) == "error: notamodel.joblib: not a readable CRF artifact\n"


def test_prepare_data_build_without_the_manifest_is_one_error_line(
    crf: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = _prepare_data_without_manifest(Env(tmp_path / FOLDER, b"", crf))
    assert run.main(run.argv) == 1
    line = _error_line(capsys)
    assert line.startswith("error: MANIFEST.sha256: not found; copy it from") and FOLDER not in line


def test_prepare_data_build_without_the_raw_root_names_the_folder_not_the_manifest(
    crf: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = Run(prepare_data.main, _prepare_data_argv(Env(tmp_path / FOLDER, b"", crf)), Path("raw"))
    assert run.main(run.argv) == 1
    assert _error_line(capsys) == "error: raw: folder not found\n"


def test_an_unreadable_companion_manifest_is_a_note_not_a_traceback(
    crf: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = tmp_path / "m.joblib"
    model.write_bytes(crf.model.read_bytes())
    (tmp_path / "m.joblib.manifest.json").write_bytes(NESTED)
    assert train_crf.main(["inspect", "--model", str(model)]) == 0
    captured = capsys.readouterr()
    assert captured.err == "Note: companion manifest unreadable: m.joblib.manifest.json\n"
    assert "Companion Manifest:" not in captured.out
