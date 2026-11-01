"""CRFModel: the read-only training log and the rebuild of a fitted model from a saved artifact bundle."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import joblib
import pytest

from punct_restore.features.crf_features import FeatureConfig
from punct_restore.models.crf import CRFModel
from punct_restore.models.crf_io import MODEL_SCHEMA_VERSION, _build_bundle, load_artifact

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


# --- _to_bundle ---------------------------------------------------------------------------------------------


def test_to_bundle_hands_over_the_private_state_the_artifact_keeps(fitted: CRFModel) -> None:
    state = fitted._to_bundle()

    assert state == {"training_metadata": fitted._training_metadata, "backend_state": fitted._backend}
    assert state["backend_state"] is fitted._backend


def test_to_bundle_copies_the_training_metadata_so_the_model_cannot_be_changed_through_it(fitted: CRFModel) -> None:
    before = {**fitted._training_metadata}

    fitted._to_bundle()["training_metadata"]["num_sequences"] = -1

    assert fitted._training_metadata == before


def test_to_bundle_refuses_a_model_that_was_never_fitted() -> None:
    with pytest.raises(RuntimeError, match="Cannot save an unfitted CRFModel"):
        CRFModel(max_iterations=5)._to_bundle()


def test_the_bundle_is_built_from_to_bundle_and_the_public_settings_only() -> None:
    """`_build_bundle` may not reach into the model's private attributes: this stand-in has none."""
    stand_in = SimpleNamespace(
        model_name="stand_in", label_mode="A", vocabulary=["O", "COMMA"], algorithm="lbfgs", c1=0.1, c2=0.2,
        max_iterations=5, all_possible_transitions=True, feature_config=FeatureConfig.basic(),
        _to_bundle=lambda: {"training_metadata": {"num_sequences": 3}, "backend_state": "backend"},
    )

    bundle = _build_bundle(stand_in)

    assert (bundle["schema_version"], bundle["training_metadata"], bundle["backend_state"]) == (
        MODEL_SCHEMA_VERSION, {"num_sequences": 3}, "backend")
    assert list(bundle) == [
        "schema_version", "model_family", "model_name", "label_mode", "vocabulary", "hyperparameters",
        "feature_config", "training_metadata", "environment", "backend_state"]


# --- an artifact path that is not a file ---------------------------------------------------------------------


def test_load_model_names_a_missing_artifact_by_file_name_only(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as caught:
        CRFModel.load_model(tmp_path / "deep" / "ghost.joblib")
    assert str(caught.value) == "ghost.joblib: model artifact not found"


def test_load_model_says_a_folder_is_a_folder(tmp_path: Path) -> None:
    (tmp_path / "m.joblib").mkdir()
    with pytest.raises(IsADirectoryError) as caught:
        CRFModel.load_model(tmp_path / "m.joblib")
    assert str(caught.value) == "m.joblib: is a folder, not a file"


# --- the one loader of a joblib artifact ----------------------------------------------------------------------


def test_load_artifact_names_a_missing_file_by_what_it_is_and_a_folder_as_a_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as missing:
        load_artifact(tmp_path / "deep" / "ghost.joblib", "thing")
    assert str(missing.value) == "ghost.joblib: thing not found"
    (tmp_path / "m.joblib").mkdir()
    with pytest.raises(IsADirectoryError) as folder:
        load_artifact(tmp_path / "m.joblib", "thing")
    assert str(folder.value) == "m.joblib: is a folder, not a file"


def test_load_artifact_gives_back_whatever_the_file_holds(tmp_path: Path) -> None:
    path = tmp_path / "list.joblib"
    joblib.dump([1, 2], path)
    assert load_artifact(path, "thing") == [1, 2]  # the caller checks the type


def test_load_artifact_does_not_show_the_unpickling_error_of_a_file_that_is_no_artifact(tmp_path: Path) -> None:
    path = tmp_path / "junk.joblib"
    path.write_bytes(b"not a pickle at all")
    with pytest.raises(ValueError) as caught:
        load_artifact(path, "thing")
    assert str(caught.value) == "junk.joblib: not a readable CRF artifact"
