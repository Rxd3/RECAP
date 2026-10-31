"""Tests for the opt-in word-cluster CRF features (use_word_clusters / word_clusters).

Fixtures only: the cluster maps here are plain dicts, so nothing depends on the map builder module.
"""

from __future__ import annotations

import copy
import json
import pickle
from types import MappingProxyType
from typing import Any

import joblib
import pytest

from punct_restore.features.crf_features import (
    UNKNOWN_CLUSTER_LABEL,
    FeatureConfig,
    _cluster_label,
    extract_sequence_features,
)
from punct_restore.models.crf import CRFModel

CLUSTERS = {"did": 5, "you": 12, "see": 0, "it": 12, "hello": 3, "everyone": 7}
NEW_KEYS = ("use_word_clusters", "word_clusters")
RECORDS = [
    {"sample_id": "a1", "tokens": ["hello", "everyone"], "punct_labels": ["O", "PERIOD"]},
    {"sample_id": "a2", "tokens": ["did", "you", "see"], "punct_labels": ["O", "O", "QUESTION"]},
    {"sample_id": "a3", "tokens": ["you", "see", "it"], "punct_labels": ["COMMA", "O", "PERIOD"]},
    {"sample_id": "a4", "tokens": ["did", "it", "see"], "punct_labels": ["COMMA", "O", "QUESTION"]},
]
GOLD_FREE = [
    {"sample_id": "p1", "tokens": ["did", "you", "see", "zebras"]},
    {"sample_id": "p2", "tokens": ["hello"]},
]


def _clustered(**overrides: Any) -> FeatureConfig:
    return FeatureConfig.improved(use_word_clusters=True, word_clusters=CLUSTERS, **overrides)


def _cluster_keys(features: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in features.items() if key.startswith("cl")}


# --- features -----------------------------------------------------------------------------------

def test_cluster_features_on_a_hand_case():
    feats = extract_sequence_features(["did", "you", "see", "zebras"], config=_clustered())

    assert _cluster_keys(feats[0]) == {"cl0": "c5", "cl+1": "c12", "cl0|cl+1": "c5|c12"}
    assert _cluster_keys(feats[1]) == {"cl-1": "c5", "cl0": "c12", "cl+1": "c0", "cl0|cl+1": "c12|c0"}
    assert _cluster_keys(feats[2]) == {"cl-1": "c12", "cl0": "c0", "cl+1": "unk", "cl0|cl+1": "c0|unk"}
    assert _cluster_keys(feats[3]) == {"cl-1": "c0", "cl0": "unk"}


def test_a_single_token_gets_only_its_own_cluster():
    assert _cluster_keys(extract_sequence_features(["hello"], config=_clustered())[0]) == {"cl0": "c3"}
    assert _cluster_keys(extract_sequence_features(["zebras"], config=_clustered())[0]) == {"cl0": "unk"}


def test_cluster_id_zero_is_a_real_cluster_not_unknown():
    feats = extract_sequence_features(["see"], config=_clustered())
    assert feats[0]["cl0"] == "c0"


def test_lookup_uses_the_lowercase_form_of_the_word():
    feats = extract_sequence_features(["Did", "YOU"], config=_clustered())
    assert _cluster_keys(feats[0]) == {"cl0": "c5", "cl+1": "c12", "cl0|cl+1": "c5|c12"}
    assert _cluster_keys(feats[1]) == {"cl-1": "c5", "cl0": "c12"}


def test_cluster_label_known_unknown_and_case():
    clusters = {"cat": 3, "dog": 10}
    assert _cluster_label(clusters, "cat") == "c3"
    assert _cluster_label(clusters, "DOG") == "c10"
    assert _cluster_label(clusters, "emu") == UNKNOWN_CLUSTER_LABEL == "unk"


def test_unknown_word_between_known_words_makes_unknown_bigram_halves():
    feats = extract_sequence_features(["did", "zebras", "it"], config=_clustered())
    assert _cluster_keys(feats[1]) == {"cl-1": "c5", "cl0": "unk", "cl+1": "c12", "cl0|cl+1": "unk|c12"}


@pytest.mark.parametrize("preset", [FeatureConfig.basic, FeatureConfig.improved])
def test_switch_off_emits_no_cluster_keys_even_when_a_map_is_present(preset):
    tokens = ["did", "you", "see", "zebras"]
    for config in (preset(), preset(word_clusters=CLUSTERS)):
        assert all(not _cluster_keys(f) for f in extract_sequence_features(tokens, config=config))


def test_switch_off_features_are_identical_to_the_presets_without_the_fields():
    tokens = ["did", "you", "see", "it"]
    plain = extract_sequence_features(tokens, config=FeatureConfig.improved())
    with_map = extract_sequence_features(tokens, config=FeatureConfig.improved(word_clusters=CLUSTERS))
    assert plain == with_map


@pytest.mark.parametrize("preset", [FeatureConfig.basic, FeatureConfig.improved])
def test_switch_works_on_any_preset_and_keeps_the_other_features(preset):
    tokens = ["did", "you", "see"]
    off = extract_sequence_features(tokens, config=preset())
    on = extract_sequence_features(tokens, config=preset(use_word_clusters=True, word_clusters=CLUSTERS))
    for before, after in zip(off, on, strict=True):
        assert {k: v for k, v in after.items() if not k.startswith("cl")} == before
        assert after["cl0"].startswith("c")


def test_mapping_config_with_a_cluster_map_extracts_like_the_object():
    tokens = ["did", "you", "see"]
    config = _clustered()
    assert extract_sequence_features(tokens, config=config.to_dict()) == extract_sequence_features(
        tokens, config=config
    )


def test_extraction_does_not_change_the_map():
    snapshot = dict(CLUSTERS)
    extract_sequence_features(["did", "zebras", "see"], config=_clustered())
    assert CLUSTERS == snapshot


# --- configuration ------------------------------------------------------------------------------

def test_presets_default_to_off_and_no_map():
    for config in (FeatureConfig.basic(), FeatureConfig.improved(), FeatureConfig()):
        assert config.use_word_clusters is False and config.word_clusters is None


@pytest.mark.parametrize("empty", [None, {}])
@pytest.mark.parametrize("preset", [FeatureConfig.basic, FeatureConfig.improved])
def test_switch_on_without_a_map_is_a_one_line_value_error(preset, empty):
    with pytest.raises(ValueError, match="word_clusters") as exc:
        preset(use_word_clusters=True, word_clusters=empty)
    assert "\n" not in str(exc.value)


def test_from_dict_without_a_map_but_with_the_switch_on_is_a_value_error():
    data = FeatureConfig.improved().to_dict()
    with pytest.raises(ValueError, match="word_clusters"):
        FeatureConfig.from_dict({**data, "use_word_clusters": True})


def test_to_dict_emits_a_plain_dict_copy_and_none_when_absent():
    config = FeatureConfig.improved(use_word_clusters=True, word_clusters=MappingProxyType(CLUSTERS))
    data = config.to_dict()

    assert type(data["word_clusters"]) is dict and data["word_clusters"] == CLUSTERS
    assert data["word_clusters"] is not CLUSTERS
    assert json.loads(json.dumps(data)) == data
    assert FeatureConfig.improved().to_dict()["word_clusters"] is None


def test_to_dict_from_dict_round_trip_keeps_the_map():
    config = _clustered(window_size=1)
    rebuilt = FeatureConfig.from_dict(config.to_dict())

    assert rebuilt == config
    assert rebuilt.word_clusters == CLUSTERS and rebuilt.use_word_clusters is True


def test_config_keeps_its_own_plain_dict_copy_of_a_read_only_map():
    source = MappingProxyType(dict(CLUSTERS))
    config = FeatureConfig.improved(use_word_clusters=True, word_clusters=source)

    assert type(config.word_clusters) is dict and config.word_clusters == CLUSTERS
    assert config.word_clusters is not source
    assert copy.deepcopy(config) == config and pickle.loads(pickle.dumps(config)) == config


def test_repr_leaves_out_the_cluster_map():
    text = repr(_clustered())
    assert "use_word_clusters=True" in text and "did" not in text and len(text) < 600
    assert _clustered() == _clustered() != FeatureConfig.improved(use_word_clusters=True, word_clusters={"x": 1})


def test_config_with_a_map_is_still_hashable_and_frozen():
    config = _clustered()
    assert isinstance(hash(config), int)
    with pytest.raises(AttributeError):
        config.use_word_clusters = False  # type: ignore[misc]


def test_equality_compares_the_map():
    assert _clustered() == _clustered()
    other = FeatureConfig.improved(use_word_clusters=True, word_clusters={**CLUSTERS, "did": 99})
    assert _clustered() != other


# --- CRF bundles --------------------------------------------------------------------------------

def _fit(config: FeatureConfig) -> CRFModel:
    return CRFModel(label_mode="A", max_iterations=25, feature_config=config).fit(RECORDS)


def test_crf_trained_with_clusters_round_trips_from_one_file_with_identical_predictions(tmp_path):
    model = _fit(_clustered())
    path = tmp_path / "only" / "crf.joblib"
    model.save_model(path)

    assert [p.name for p in path.parent.iterdir()] == ["crf.joblib"]  # no side file holds the map
    assert joblib.load(path)["feature_config"]["word_clusters"] == CLUSTERS
    loaded = CRFModel.load_model(path)

    assert loaded.feature_config == model.feature_config
    assert loaded.predict(GOLD_FREE) == model.predict(GOLD_FREE)
    assert extract_sequence_features(["did", "you"], config=loaded.feature_config)[0]["cl0"] == "c5"


def test_the_clusters_change_what_the_crf_sees():
    tokens = ["did", "you", "see"]
    plain = extract_sequence_features(tokens, config=_fit(FeatureConfig.improved()).feature_config)
    clustered = extract_sequence_features(tokens, config=_fit(_clustered()).feature_config)
    assert clustered[1].keys() - plain[1].keys() == {"cl-1", "cl0", "cl+1", "cl0|cl+1"}


def test_crf_bundle_saved_before_the_cluster_switch_still_loads_and_predicts_the_same(tmp_path):
    model = _fit(FeatureConfig.improved())
    expected = model.predict(GOLD_FREE)
    path = tmp_path / "old.joblib"
    model.save_model(path)
    bundle = joblib.load(path)
    for key in NEW_KEYS:
        bundle["feature_config"].pop(key, None)  # what a bundle from before the switch looks like
    joblib.dump(bundle, path)

    loaded = CRFModel.load_model(path)

    assert loaded.feature_config == FeatureConfig.improved()
    assert loaded.feature_config.use_word_clusters is False
    assert loaded.predict(GOLD_FREE) == expected


def test_crf_bundle_with_the_switch_on_but_no_map_is_a_value_error(tmp_path):
    path = tmp_path / "broken.joblib"
    _fit(_clustered()).save_model(path)
    bundle = joblib.load(path)
    bundle["feature_config"]["word_clusters"] = None
    joblib.dump(bundle, path)

    with pytest.raises(ValueError, match="word_clusters"):
        CRFModel.load_model(path)


def test_crf_save_does_not_mutate_the_config_map(tmp_path):
    snapshot = dict(CLUSTERS)
    _fit(_clustered()).save_model(tmp_path / "m.joblib")
    assert CLUSTERS == snapshot
