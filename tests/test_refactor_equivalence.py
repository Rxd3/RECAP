"""Outputs the 2026-11-01 restructure must not change, pinned from main 72ebfd1 (course finding C5).

The tests feed the committed data/sample utterances (10, all three label modes) and one fixed sentence
to the public feature, HMM, hidden-event LM, CRF and demo-pipeline APIs, and compare what comes out
with what main 72ebfd1 produced from the same inputs. The EXPECTED_* values were computed by running
this module's helpers with PYTHONPATH set to the src/ of a 72ebfd1 checkout. Never regenerate them
from the current code: a pin taken from the code under test proves nothing.

Digests are SHA-256 over a canonical JSON dump (sorted keys, no spaces, UTF-8). Saved HMM and
hidden-event LM files are hashed byte for byte; a saved CRF bundle is hashed without its platform
record and its crfsuite weights, which the predictions pin instead. The models are trained on copies of
the (validation) sample rows re-marked split=train.
"""

from __future__ import annotations

import copy
import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import joblib
import pytest

from punct_restore.data import load_label_sets
from punct_restore.features.crf_features import FeatureConfig, extract_sequence_features
from punct_restore.models.crf import CRFModel
from punct_restore.models.hidden_event_lm import HiddenEventLM
from punct_restore.models.hmm import HMMModel
from punct_restore.pipeline import ModelAdapter, RestorationPipeline

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"
MODES = ("A", "B", "C")
FAMILIES = ("hmm", "hidden_event_lm", "crf_basic", "crf_improved")
SAVED_JSON = ("hmm", "hidden_event_lm")
CRF_FAMILIES = ("crf_basic", "crf_improved")
CRF_UNPINNED_KEYS = ("environment", "backend_state")
DEMO_TEXT = "so what do you think about that i mean we should go to london next week right"

# ---- expected values, computed with main 72ebfd1 (see the module docstring) ----

EXPECTED_FEATURES: dict[str, str] = {
    "basic": "5c3a80e5fd511db9e19a594785ec106ee3e6fb661a2f89c3130bfa5d2d288c7e",
    "improved": "2d342176840aac541b32debac4338c647d85cee01eb2cfd22d332193c8539c92",
    "all_switches": "b1cb49948aa9687acf5a8b96bfd31e5f12825f2270b58ac6e8b9fb9336b0ae4e",
}
EXPECTED_PREDICTIONS: dict[str, str] = {
    "hmm/A": "752fa51ab9723ac35b99deef33b71676145129efd109a11018f899c2fba8c842",
    "hmm/B": "970b01deb9c62fe720f41b269b25f604b4158ea2fae69426843eda9a8f5b5bab",
    "hmm/C": "5d55e0057a0a61625f9f4539e4ec790ee9928014e65132aba4a3eafbaa1df1f5",
    "hidden_event_lm/A": "2b7ffa54fcc3e182956fd98a24f6c0537f4d779f5962cfb0abd5ed4cf22b6921",
    "hidden_event_lm/B": "7d0f6ff1de9e89a34bc8b888ab98851ed2ede6d9259d55c35cbcbb4c9b6802a9",
    "hidden_event_lm/C": "ad9ab13d39ef54f929289f2d9d24121642fb7469336ebde54cba9f4e5ffa5d44",
    "crf_basic/A": "5b0925301a3cb6c55d31af699d4f9e87c62e0af8024dd3948a64118c2c6ed6e5",
    "crf_basic/B": "1a2c44d0013023d57b8afdf5d96d849ce64727d14251ab1710d08fd5552379f2",
    "crf_basic/C": "c74784ddb00fe3d522569574dade3dd849f377350babb5b614fe64cbb8599e85",
    "crf_improved/A": "99bf3fb93ec9f12f7d73a2426bd97ff0cb666d51a0db546f729eeb595fd62938",
    "crf_improved/B": "9ea3e8bdf54cda811cebe085afb1a84949552cb2fe049e47d11077e4cdad4216",
    "crf_improved/C": "0b7e4db188a4c61224ddc924512b5e671cb1e60cd9860d7b8b3529d467e60b05",
}
EXPECTED_SAVED_MODELS: dict[str, str] = {
    "hmm/A": "525b9a2464d7c197bacf4976d9ba87f442848cad3b8c310aab77e9b6c4b53a66",
    "hmm/B": "ab4d6c330876f4b0851c2c7762c0c3bfa65c9a3bf09460945acd02c61a361979",
    "hmm/C": "4a746b0133f40b31be585785b85bf2dfa65bfb3d04fa1a4f27e0cd6d04750eef",
    "hidden_event_lm/A": "3a8be1e85b8e959b7e81f6e40dea8c0f9a50b196944a1590ab8dbfab863fcd72",
    "hidden_event_lm/B": "68e0db93ad23d9db12b55e0f3e6c862b92ed2d0c4466a2ffc8c46f9cc9ad91b6",
    "hidden_event_lm/C": "b735976c0173b2dba2a9a8e93bad20c029a65604825446f6abd5a43d27086ed5",
}
EXPECTED_CRF_BUNDLES: dict[str, str] = {
    "crf_basic/A": "f8a8136fe41c1e3f624484cb5ac99f533987929f5b90f68e0ab51d5bd8f9c610",
    "crf_basic/B": "bad889d1030fba14681961935ebe5aa86349325838b78470953b96a3e79d1955",
    "crf_basic/C": "c3cfe6ba02635d33581566dc0057296b6cd49d8894f4b0f4886eb1b844d2c500",
    "crf_improved/A": "6a0a3123cf6f17fb8d27fa00a51a22ee4d7a4364755a1461a81788804b5a3b0b",
    "crf_improved/B": "275b227d972624ce7c4ce266703b75ef99c2e66f28124f8951f3b0cc4b67adfa",
    "crf_improved/C": "e883e53e81e6149c2af939021ddb9c680d22965d9be0737a5c7b9fd78102fb99",
}
EXPECTED_DEMO: dict[str, str] = {
    "hmm/A": "so what do you think about that i mean we should go to london next week right.",
    "hmm/C": "So what do you think about that? I mean we should go to london next week right.",
    "hidden_event_lm/A": "so what do you think about that i mean we should go to london next week right",
    "hidden_event_lm/C": "So what do you think about that I mean we should go to london next week right",
    "crf_basic/A": "so what do you think about that i mean we should go to london next week right.",
    "crf_basic/C": "So what do you think about that I mean we should go to london next week right.",
    "crf_improved/A": "so what do you think about that i mean we should go to london next week right.",
    "crf_improved/C": "So what do you think about that I mean we should go to london next week right.",
}


# ---- helpers (also used, unchanged, to compute the expected values on 72ebfd1) ----


def canonical_sha256(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@cache
def sample_records(mode: str) -> tuple[dict[str, Any], ...]:
    lines = (SAMPLE / f"mode_{mode}_validation.jsonl").read_text(encoding="utf-8").splitlines()
    return tuple(json.loads(line) for line in lines if line.strip())


def training_copies(mode: str) -> list[dict[str, Any]]:
    return [{**record, "split": "train"} for record in sample_records(mode)]


def feature_configs() -> dict[str, FeatureConfig]:
    """The two course presets, plus every switch on (team extensions and a fixed word-cluster map)."""
    vocabulary = sorted({token for record in sample_records("A") for token in record["tokens"]})
    clusters = {token: index % 7 for index, token in enumerate(vocabulary)}
    everything = FeatureConfig.improved(
        use_filler_cues=True,
        use_question_lookback=True,
        use_word_clusters=True,
        word_clusters=clusters,
    )
    return {"basic": FeatureConfig.basic(), "improved": FeatureConfig.improved(), "all_switches": everything}


def feature_digest(config_name: str) -> str:
    sequences = [list(record["tokens"]) for record in sample_records("A")] + [DEMO_TEXT.split()]
    config = feature_configs()[config_name]
    return canonical_sha256([extract_sequence_features(tokens, config=config) for tokens in sequences])


def new_model(family: str, mode: str) -> Any:
    if family == "hmm":
        return HMMModel(label_mode=mode, labels=load_label_sets()[mode])
    if family == "hidden_event_lm":
        return HiddenEventLM(mode)
    if family == "crf_basic":
        return CRFModel(label_mode=mode, feature_config=FeatureConfig.basic())
    if family == "crf_improved":
        return CRFModel(label_mode=mode, feature_config=FeatureConfig.improved())
    raise ValueError(f"unknown model family {family!r}")


@cache
def fitted_model(family: str, mode: str) -> Any:
    return new_model(family, mode).fit(training_copies(mode))


def prediction_digest(family: str, mode: str) -> str:
    records = copy.deepcopy(list(sample_records(mode)))  # the cached rows stay untouched whatever predict does
    return canonical_sha256(fitted_model(family, mode).predict(records))


def saved_model_digest(family: str, mode: str, folder: Path) -> str:
    path = folder / f"{family}_{mode}.json"
    fitted_model(family, mode).save_model(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def crf_bundle_digest(family: str, mode: str, folder: Path) -> str:
    path = folder / f"{family}_{mode}.joblib"
    fitted_model(family, mode).save_model(path)
    bundle = joblib.load(path)  # a pickle, but this test wrote it a line above
    return canonical_sha256({key: value for key, value in bundle.items() if key not in CRF_UNPINNED_KEYS})


def demo_output(family: str, mode: str, folder: Path) -> str:
    """What `recap run-demo --model-path <saved model> --text DEMO_TEXT` restores (the same pipeline call)."""
    path = folder / f"demo_{family}_{mode}{'.json' if family in SAVED_JSON else '.joblib'}"
    fitted_model(family, mode).save_model(path)
    return RestorationPipeline(ModelAdapter(path)).process_text(DEMO_TEXT)["restored_text"]


def pin_key(family: str, mode: str) -> str:
    return f"{family}/{mode}"


# ---- tests ----


@pytest.mark.parametrize("config_name", ["basic", "improved", "all_switches"])
def test_crf_feature_dicts_match_main(config_name: str) -> None:
    assert feature_digest(config_name) == EXPECTED_FEATURES[config_name]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("family", FAMILIES)
def test_predictions_on_the_sample_match_main(family: str, mode: str) -> None:
    assert prediction_digest(family, mode) == EXPECTED_PREDICTIONS[pin_key(family, mode)]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("family", SAVED_JSON)
def test_saved_json_models_match_main_byte_for_byte(family: str, mode: str, tmp_path: Path) -> None:
    assert saved_model_digest(family, mode, tmp_path) == EXPECTED_SAVED_MODELS[pin_key(family, mode)]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("family", CRF_FAMILIES)
def test_saved_crf_bundles_match_main(family: str, mode: str, tmp_path: Path) -> None:
    assert crf_bundle_digest(family, mode, tmp_path) == EXPECTED_CRF_BUNDLES[pin_key(family, mode)]


@pytest.mark.parametrize("mode", ["A", "C"])
@pytest.mark.parametrize("family", FAMILIES)
def test_demo_restores_the_fixed_sentence_like_main(family: str, mode: str, tmp_path: Path) -> None:
    assert demo_output(family, mode, tmp_path) == EXPECTED_DEMO[pin_key(family, mode)]


def test_the_pins_cover_every_case() -> None:
    assert set(EXPECTED_FEATURES) == {"basic", "improved", "all_switches"}
    assert set(EXPECTED_PREDICTIONS) == {pin_key(f, m) for f in FAMILIES for m in MODES}
    assert set(EXPECTED_SAVED_MODELS) == {pin_key(f, m) for f in SAVED_JSON for m in MODES}
    assert set(EXPECTED_CRF_BUNDLES) == {pin_key(f, m) for f in CRF_FAMILIES for m in MODES}
    assert set(EXPECTED_DEMO) == {pin_key(f, m) for f in FAMILIES for m in ("A", "C")}


def test_an_unknown_model_family_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown model family 'crf'"):
        new_model("crf", "A")
