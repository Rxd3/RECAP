"""Contract tests: ModelAdapter loads real saved artifacts and enforces the model contract."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import pytest

from punct_restore import pipeline
from punct_restore.data.labeling import load_label_sets
from punct_restore.models.crf import CRFModel
from punct_restore.models.gap_classifier import GapClassifier
from punct_restore.models.hidden_event_lm import HiddenEventLM
from punct_restore.models.hmm import HMMModel
from punct_restore.pipeline import ModelAdapter, RestorationPipeline, load_model
from scripts import run_demo

_SENTENCES = [
    ["hello", "world", "how", "are", "you"],
    ["this", "is", "a", "test"],
    ["we", "like", "tiny", "models"],
]


def tiny_records() -> list[dict[str, Any]]:
    """Mode A training records in which every sequence ends in PERIOD."""
    return [
        {
            "schema_version": 1,
            "sample_id": f"fixture-{index}",
            "split": "train",
            "label_mode": "A",
            "source": "fixture",
            "tokens": list(tokens),
            "punct_labels": ["O"] * (len(tokens) - 1) + ["PERIOD"],
            "case_labels": ["LOWER"] * len(tokens),
        }
        for index, tokens in enumerate(_SENTENCES)
    ]


@pytest.fixture(scope="module")
def fitted_hmm() -> HMMModel:
    model = HMMModel(label_mode="A", labels=load_label_sets()["A"], min_token_count=1)
    return model.fit(tiny_records())


@pytest.fixture(scope="module")
def fitted_crf() -> CRFModel:
    return CRFModel(label_mode="A", max_iterations=50).fit(tiny_records())


class StubModel:
    """Minimal model double whose predict output is scripted by the test."""

    def __init__(self, label_mode: str, respond: Callable[[list[dict[str, Any]]], Any]) -> None:
        self.label_mode = label_mode
        self._respond = respond

    def predict(self, records: list[dict[str, Any]]) -> Any:
        return self._respond(records)


def constant_labels(label: str) -> Callable[[list[dict[str, Any]]], list[dict[str, Any]]]:
    def respond(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        count = len(records[0]["tokens"])
        return [
            {
                "sample_id": "demo-0",
                "model_name": "stub",
                "label_mode": "A",
                "predicted_labels": [label] * count,
            }
        ]

    return respond


def stub_adapter(
    monkeypatch: pytest.MonkeyPatch,
    label_mode: str,
    respond: Callable[[list[dict[str, Any]]], Any],
) -> ModelAdapter:
    monkeypatch.setattr(pipeline, "load_model", lambda path: StubModel(label_mode, respond))
    return ModelAdapter(Path("stub.joblib"))


@pytest.mark.parametrize("name", ["missing.joblib", "missing.pkl", "missing.json"])
def test_model_adapter_raises_on_missing_model_file(tmp_path: Path, name: str) -> None:
    with pytest.raises(FileNotFoundError):
        ModelAdapter(tmp_path / name)


def test_model_adapter_raises_on_missing_config(tmp_path: Path, fitted_hmm: HMMModel) -> None:
    model_path = tmp_path / "hmm.json"
    fitted_hmm.save_model(model_path)

    with pytest.raises(FileNotFoundError):
        ModelAdapter(model_path, config_path=tmp_path / "missing.yaml")


def test_demo_hmm_predicts_some_non_o(tmp_path: Path, fitted_hmm: HMMModel) -> None:
    model_path = tmp_path / "hmm.json"
    fitted_hmm.save_model(model_path)

    adapter = ModelAdapter(model_path)
    labels = adapter.predict_tokens(["Hello,", "world."])

    assert len(labels) == 2
    assert any(label != "O" for label in labels)
    assert adapter.vocabulary == tuple(load_label_sets()["A"])


@pytest.mark.parametrize("suffix", [".joblib", ".pkl"])
def test_demo_crf_round_trip(tmp_path: Path, fitted_crf: CRFModel, suffix: str) -> None:
    model_path = tmp_path / f"crf{suffix}"
    fitted_crf.save_model(model_path)
    tokens = ["hello", "world", "how"]
    expected = fitted_crf.predict([{"sample_id": "demo-0", "tokens": tokens}])[0]["predicted_labels"]

    labels = ModelAdapter(model_path).predict_tokens(["Hello,", "World", "how."])

    assert labels == expected


def test_load_model_dispatches_on_suffix(tmp_path: Path, fitted_hmm: HMMModel) -> None:
    model_path = tmp_path / "hmm.json"
    fitted_hmm.save_model(model_path)

    assert isinstance(load_model(model_path), HMMModel)


def test_load_model_suffix_is_case_insensitive(
    tmp_path: Path, fitted_hmm: HMMModel, fitted_crf: CRFModel
) -> None:
    hmm_path = tmp_path / "M.JSON"
    fitted_hmm.save_model(hmm_path)
    crf_paths = [tmp_path / "M.JOBLIB", tmp_path / "M.PKL"]
    for crf_path in crf_paths:
        fitted_crf.save_model(crf_path)

    for model_path in (hmm_path, *crf_paths):
        labels = ModelAdapter(model_path).predict_tokens(["hello", "world"])
        assert len(labels) == 2
        assert set(labels) <= set(load_label_sets()["A"])
    assert isinstance(load_model(hmm_path), HMMModel)
    assert all(isinstance(load_model(p), CRFModel) for p in crf_paths)

    unsupported = tmp_path / "M.TXT"
    unsupported.write_text("not a model")
    with pytest.raises(ValueError, match="unsupported model file"):
        ModelAdapter(unsupported)


def test_rejects_unknown_extension(tmp_path: Path) -> None:
    model_path = tmp_path / "model.txt"
    model_path.write_text("not a model")

    with pytest.raises(ValueError, match="unsupported model file"):
        ModelAdapter(model_path)


def test_rejects_labels_outside_vocabulary(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = stub_adapter(monkeypatch, "A", constant_labels("BANG"))

    with pytest.raises(ValueError, match="outside the mode A vocabulary"):
        adapter.predict_tokens(["hello", "world"])


def test_rejects_joint_label_in_mode_a(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = stub_adapter(monkeypatch, "A", constant_labels("PERIOD|FIRST_UPPER"))

    with pytest.raises(ValueError, match="outside the mode A vocabulary"):
        adapter.predict_tokens(["hello"])


def test_mode_c_adapter_accepts_joint_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = stub_adapter(monkeypatch, "C", constant_labels("PERIOD|FIRST_UPPER"))

    assert adapter.predict_tokens(["hello", "world"]) == ["PERIOD|FIRST_UPPER"] * 2
    assert len(adapter.vocabulary) == 12


def test_rejects_unknown_label_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="label mode"):
        stub_adapter(monkeypatch, "Z", constant_labels("O"))


@pytest.mark.parametrize("result", [[], [{"predicted_labels": ["O"]}] * 2, "nope"])
def test_rejects_malformed_prediction_list(monkeypatch: pytest.MonkeyPatch, result: Any) -> None:
    adapter = stub_adapter(monkeypatch, "A", lambda records: result)

    with pytest.raises(ValueError, match="exactly one prediction"):
        adapter.predict_tokens(["hello"])


def test_rejects_prediction_without_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = stub_adapter(monkeypatch, "A", lambda records: [{"sample_id": "demo-0"}])

    with pytest.raises(ValueError, match="predicted_labels"):
        adapter.predict_tokens(["hello"])


def test_rejects_label_count_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = stub_adapter(monkeypatch, "A", lambda records: [{"predicted_labels": ["O"]}])

    with pytest.raises(ValueError, match="1 labels for 3 tokens"):
        adapter.predict_tokens(["a", "b", "c"])


def test_predict_tokens_does_not_mutate_input(fitted_hmm: HMMModel, tmp_path: Path) -> None:
    model_path = tmp_path / "hmm.json"
    fitted_hmm.save_model(model_path)
    tokens = ["Hello,", "World."]

    ModelAdapter(model_path).predict_tokens(tokens)

    assert tokens == ["Hello,", "World."]


# --- hidden-event LM and gap classifier artifacts -----------------------------------------------

_WORDS = ("hello", "world", "how", "are", "you")


def tiny_c_records() -> list[dict[str, Any]]:
    """Mode C records in which the first word of every sequence is FIRST_UPPER."""
    return [
        {**record, "label_mode": "C", "case_labels": ["FIRST_UPPER", *["LOWER"] * (len(record["tokens"]) - 1)]}
        for record in tiny_records()
    ]


@pytest.mark.parametrize(
    ("filename", "model_class", "build"),
    [
        ("hel.json", HiddenEventLM, lambda: HiddenEventLM("A").fit(tiny_records())),
        ("HEL.JSON", HiddenEventLM, lambda: HiddenEventLM("A").fit(tiny_records())),
        ("gap.joblib", GapClassifier, lambda: GapClassifier("A").fit(tiny_records())),
        ("GAP.PKL", GapClassifier, lambda: GapClassifier("A").fit(tiny_records())),
    ],
)
def test_demo_restores_text_with_the_new_model_families(
    tmp_path: Path, filename: str, model_class: type, build: Callable[[], Any]
) -> None:
    model_path = tmp_path / filename
    build().save_model(model_path)

    adapter = ModelAdapter(model_path)
    result = RestorationPipeline(adapter).process_text("Hello world how are you")

    assert isinstance(load_model(model_path), model_class)
    assert adapter.vocabulary == tuple(load_label_sets()["A"])
    assert len(result["predicted_labels"]) == len(_WORDS)
    assert set(result["predicted_labels"]) <= set(adapter.vocabulary)
    assert all(word in result["restored_text"] for word in _WORDS)


def test_demo_hidden_event_lm_mode_c_restores_casing(tmp_path: Path) -> None:
    model_path = tmp_path / "hel_c.json"
    HiddenEventLM("C").fit(tiny_c_records()).save_model(model_path)

    adapter = ModelAdapter(model_path)
    result = RestorationPipeline(adapter).process_text("hello world how are you")

    assert adapter.vocabulary == tuple(load_label_sets()["C"])
    assert set(result["predicted_labels"]) <= set(adapter.vocabulary)
    assert result["predicted_labels"][0].endswith("|FIRST_UPPER")
    assert result["restored_text"].startswith("Hello world")


def test_run_demo_cli_works_with_a_hidden_event_lm(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_path = tmp_path / "hel.json"
    HiddenEventLM("A").fit(tiny_records()).save_model(model_path)

    code = run_demo.main(["--model-path", str(model_path), "--text", "hello world"])

    out = capsys.readouterr().out
    assert code == 0
    assert "Input:    hello world" in out
    assert "Restored: hello world" in out


def _assert_one_friendly_line(error: pytest.ExceptionInfo[ValueError], tmp_path: Path) -> None:
    message = str(error.value)
    assert len(message.splitlines()) == 1
    assert tmp_path.name not in message  # a path fragment, not just the file name


@pytest.mark.parametrize(
    "content",
    [
        json.dumps({"model_family": "mystery"}),
        json.dumps({"model_family": ["hidden_event_lm"]}),
        json.dumps({"model_family": "crf"}),  # a joblib family inside a .json
        json.dumps({"format_version": 1}),
        json.dumps({"model_family": "mystery", "model_name": "hmm"}),  # the family field wins over model_name
        json.dumps([1, 2]),
        "{not json",
        "",
    ],
)
def test_load_model_rejects_unrecognised_json(tmp_path: Path, content: str) -> None:
    model_path = tmp_path / "bad.json"
    model_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported model file: bad.json") as error:
        load_model(model_path)

    _assert_one_friendly_line(error, tmp_path)


@pytest.mark.parametrize(
    "bundle",
    [
        {"model_family": "mystery"},
        {"model_family": ["crf"]},
        {"model_family": "hidden_event_lm"},  # a json family inside a .joblib
        {"schema_version": 1},
        [1, 2],
        "just text",
    ],
)
def test_load_model_rejects_unrecognised_joblib(tmp_path: Path, bundle: Any) -> None:
    model_path = tmp_path / "bad.joblib"
    joblib.dump(bundle, model_path)

    with pytest.raises(ValueError, match="unsupported model file: bad.joblib") as error:
        load_model(model_path)

    _assert_one_friendly_line(error, tmp_path)


def test_load_model_rejects_an_unpicklable_file(tmp_path: Path) -> None:
    model_path = tmp_path / "garbage.pkl"
    model_path.write_bytes(b"this is not a pickle")

    with pytest.raises(ValueError, match="unsupported model file: garbage.pkl") as error:
        load_model(model_path)

    _assert_one_friendly_line(error, tmp_path)


@pytest.mark.parametrize("name", ["folder.json", "folder.joblib"])
def test_load_model_reports_a_folder_as_a_folder(tmp_path: Path, name: str) -> None:
    (tmp_path / name).mkdir()
    with pytest.raises(IsADirectoryError, match=f"model path is a folder, not a file: {name}") as error:
        load_model(tmp_path / name)
    assert tmp_path.name not in str(error.value)


@pytest.mark.parametrize("name", ["locked.json", "locked.joblib"])
def test_load_model_keeps_an_os_read_error_an_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    model_path = tmp_path / name
    model_path.write_text("{}", encoding="utf-8")

    def denied(*_args: Any, **_kwargs: Any) -> Any:
        raise PermissionError(13, "Permission denied", str(model_path))

    monkeypatch.setattr(Path, "read_text", denied)
    monkeypatch.setattr(pipeline.joblib, "load", denied)
    with pytest.raises(OSError, match=f"cannot read model file {name}: Permission denied") as error:
        load_model(model_path)
    assert "unsupported model file" not in str(error.value) and tmp_path.name not in str(error.value)


@pytest.mark.parametrize("name", ["missing.json", "missing.joblib", "missing.pkl"])
def test_load_model_missing_file_error_names_no_path(tmp_path: Path, name: str) -> None:
    with pytest.raises(FileNotFoundError) as error:
        load_model(tmp_path / name)

    assert name in str(error.value)
    assert tmp_path.name not in str(error.value)


def test_run_demo_prints_one_error_line_for_a_casing_helper_crf(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = [
        {"sample_id": f"c{index}", "split": "train", "tokens": ["we", "agree"],
         "punct_labels": ["O", "PERIOD"], "case_labels": ["FIRST_UPPER", "LOWER"]}
        for index in range(3)
    ]
    model_path = tmp_path / "case.joblib"
    CRFModel(label_mode="CASE", max_iterations=5).fit(records).save_model(model_path)

    code = run_demo.main(["--model-path", str(model_path), "--text", "we agree"])

    err = capsys.readouterr().err
    assert code == 1
    assert len(err.strip().splitlines()) == 1
    assert "CASE" in err
