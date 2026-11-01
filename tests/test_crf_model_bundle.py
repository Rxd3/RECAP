"""CRFModel: the read-only training log and the rebuild of a fitted model from a saved artifact bundle."""

from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from punct_restore.models.crf import CRFModel

RECORDS = [
    {"sample_id": "b1", "tokens": ["hello", "everyone"], "punct_labels": ["O", "PERIOD"]},
    {"sample_id": "b2", "tokens": ["did", "you", "understand"], "punct_labels": ["O", "O", "QUESTION"]},
    {"sample_id": "b3", "tokens": ["yes", "we", "can"], "punct_labels": ["COMMA", "O", "PERIOD"]},
]
OBSERVATIONS = [{"sample_id": "p1", "tokens": ["we", "can", "go"]}, {"sample_id": "p2", "tokens": ["hello"]}]


@pytest.fixture
def fitted() -> CRFModel:
    return CRFModel(label_mode="A", c1=0.3, c2=0.2, max_iterations=7, model_name="bundle_probe").fit(RECORDS)


def _saved_bundle(model: CRFModel, path: Path) -> dict:
    """Save ``model`` and read the bundle back; joblib is pickle, but the file was written one line earlier."""
    model.save_model(path)
    return joblib.load(path)


# --- training_log -------------------------------------------------------------------------------------------


def test_the_training_log_is_none_before_a_fit_and_the_optimizer_log_after(fitted: CRFModel) -> None:
    assert CRFModel(max_iterations=5).training_log is None
    assert len(fitted.training_log.iterations) >= 1
    assert fitted.training_log is fitted._backend.training_log_


def test_the_training_log_cannot_be_assigned(fitted: CRFModel) -> None:
    with pytest.raises(AttributeError):
        fitted.training_log = None  # type: ignore[misc]


# --- _from_bundle -------------------------------------------------------------------------------------------


def test_from_bundle_rebuilds_the_fitted_model_of_a_saved_artifact(fitted: CRFModel, tmp_path: Path) -> None:
    bundle = _saved_bundle(fitted, tmp_path / "m.joblib")

    rebuilt = CRFModel._from_bundle(bundle)

    assert rebuilt.is_fitted
    assert (rebuilt.label_mode, rebuilt.model_name) == ("A", "bundle_probe")
    assert (rebuilt.algorithm, rebuilt.c1, rebuilt.c2, rebuilt.max_iterations) == ("lbfgs", 0.3, 0.2, 7)
    assert rebuilt.feature_config.to_dict() == fitted.feature_config.to_dict()
    assert rebuilt.predict(OBSERVATIONS) == fitted.predict(OBSERVATIONS)


def test_from_bundle_keeps_the_training_metadata_so_a_resave_is_identical(fitted: CRFModel, tmp_path: Path) -> None:
    bundle = _saved_bundle(fitted, tmp_path / "m.joblib")

    again = _saved_bundle(CRFModel._from_bundle(bundle), tmp_path / "again.joblib")

    assert again["training_metadata"] == bundle["training_metadata"]
    assert {k: v for k, v in again.items() if k != "backend_state"} == {
        k: v for k, v in bundle.items() if k != "backend_state"}


def test_load_model_still_rejects_an_artifact_whose_vocabulary_does_not_match_its_mode(
        fitted: CRFModel, tmp_path: Path) -> None:
    bundle = _saved_bundle(fitted, tmp_path / "m.joblib")
    joblib.dump({**bundle, "vocabulary": ["O", "COMMA"]}, tmp_path / "tampered.joblib")

    with pytest.raises(ValueError, match="does not match expected"):
        CRFModel.load_model(tmp_path / "tampered.joblib")
