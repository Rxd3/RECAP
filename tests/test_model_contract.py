"""Contract tests: ModelAdapter loads real saved artifacts and enforces the model contract."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from punct_restore import pipeline
from punct_restore.data.labeling import load_label_sets
from punct_restore.models.crf import CRFModel
from punct_restore.models.hmm import HMMModel
from punct_restore.pipeline import ModelAdapter, load_model

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
