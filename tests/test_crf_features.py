"""Unit and acceptance tests for CRF feature extraction module.

Verifies:
- Length preservation: Exactly one feature dict per token.
- Input validation: Rejection of empty sequences, non-strings, and whitespace.
- Gold-free integrity: No target punctuation or gold casing in features.
- Determinism and immutability: Identical inputs produce identical outputs without mutation.
- Boundary handling: BOS and EOS behavior across varied sequence lengths, including single tokens.
- Presets and ablations: Basic vs improved presets and utterance-final shortcut ablation.
"""

from __future__ import annotations

import copy
import pytest

from punct_restore.features.crf_features import (
    FeatureConfig,
    extract_sequence_features,
    WH_WORDS,
    AUXILIARY_VERBS,
    CONJUNCTIONS,
    DISCOURSE_MARKERS,
)


@pytest.fixture
def sample_tokens() -> list[str]:
    return ["hello", "everyone", "welcome", "to", "our", "presentation", "today"]


@pytest.fixture
def question_tokens() -> list[str]:
    return ["did", "you", "understand", "the", "results"]


def test_feature_count_equals_token_count(sample_tokens):
    """Output list length must strictly equal the input token count."""
    for preset in [FeatureConfig.basic(), FeatureConfig.improved()]:
        features = extract_sequence_features(sample_tokens, config=preset)
        assert len(features) == len(sample_tokens)
        assert all(isinstance(f, dict) for f in features)


@pytest.mark.parametrize("length", [1, 2, 3, 5, 15, 30])
def test_various_sequence_lengths(length):
    """Test length preservation across varied sequence sizes."""
    tokens = [f"word{i}" for i in range(length)]
    features = extract_sequence_features(tokens)
    assert len(features) == length
    assert all(f["token"] == f"word{i}" for i, f in enumerate(features))


def test_single_token_boundary_handling():
    """A single-token sequence must have both BOS and EOS flags."""
    tokens = ["hello"]
    features = extract_sequence_features(tokens)
    assert len(features) == 1
    assert features[0]["BOS"] is True
    assert features[0]["EOS"] is True


def test_multi_token_boundary_handling(sample_tokens):
    """BOS only at index 0, EOS only at index T-1."""
    features = extract_sequence_features(sample_tokens)
    total = len(sample_tokens)
    for i, feat in enumerate(features):
        if i == 0:
            assert feat.get("BOS") is True
            assert feat.get("EOS") is None
        elif i == total - 1:
            assert feat.get("BOS") is None
            assert feat.get("EOS") is True
        else:
            assert feat.get("BOS") is None
            assert feat.get("EOS") is None


def test_empty_sequence_raises_value_error():
    """Empty token sequences must raise ValueError per data contract."""
    with pytest.raises(ValueError, match="cannot be empty"):
        extract_sequence_features([])


def test_string_as_tokens_raises_type_error():
    """Passing a single string instead of a sequence must raise TypeError."""
    with pytest.raises(TypeError, match="must be a sequence of strings"):
        extract_sequence_features("hello everyone")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_input", [
    ["valid", ""],
    ["valid", "   "],
    ["valid", None],
    ["valid", 123],
])
def test_malformed_tokens_raise_error(bad_input):
    """Empty strings, whitespace-only strings, or non-strings must raise error."""
    with pytest.raises((ValueError, TypeError)):
        extract_sequence_features(bad_input)  # type: ignore[arg-type]


def test_gold_free_integrity(sample_tokens):
    """Feature keys and values must never contain gold target names or metadata."""
    forbidden_terms = {
        "punct_labels", "case_labels", "gold", "period", "comma", "question",
        "exclamation", "colon", "semicolon", "dash", "first_upper", "all_upper", "lower"
    }
    for preset in [FeatureConfig.basic(), FeatureConfig.improved()]:
        features = extract_sequence_features(sample_tokens, config=preset)
        for feat_dict in features:
            for k, v in feat_dict.items():
                k_clean = k.lower().replace("-", "").replace("+", "").replace(":", "")
                assert k_clean not in forbidden_terms, f"Suspicious key {k} in features"
                if isinstance(v, str):
                    assert v not in {"FIRST_UPPER", "ALL_UPPER", "PERIOD", "COMMA", "QUESTION"}


def test_deterministic_extraction(sample_tokens):
    """Consecutive extractions on the same input must yield identical output."""
    cfg = FeatureConfig.improved()
    first = extract_sequence_features(sample_tokens, config=cfg)
    second = extract_sequence_features(sample_tokens, config=cfg)
    assert first == second


def test_input_immutability(sample_tokens):
    """Input sequence must not be modified in place."""
    original = copy.deepcopy(sample_tokens)
    extract_sequence_features(sample_tokens, config=FeatureConfig.improved())
    assert sample_tokens == original


def test_basic_preset_contains_expected_features():
    """Basic preset must include token, affixes, shape, and +/-1 context."""
    tokens = ["apple", "pie"]
    features = extract_sequence_features(tokens, config=FeatureConfig.basic())

    assert features[0]["token"] == "apple"
    assert features[0]["len"] == 5
    assert features[0]["len_bucket"] == "medium"
    assert features[0]["pref1"] == "a"
    assert features[0]["suff1"] == "e"
    assert features[0]["suff3"] == "ple"
    assert features[0]["+1:token"] == "pie"
    assert features[0]["BOS"] is True
    assert "bigram_next" not in features[0]  # bigrams belong to improved


def test_improved_preset_richer_features(question_tokens):
    """Improved preset must include +/-2 window, bigrams, and question cues."""
    features = extract_sequence_features(question_tokens, config=FeatureConfig.improved())

    # Token 0: 'did'
    assert features[0]["token"] == "did"
    assert features[0]["is_aux"] is True
    assert features[0]["utt_starts_aux"] is True
    assert features[0]["bigram_next"] == "did_you"
    assert features[0]["q_inversion_next"] is True
    assert features[0]["+2:token"] == "understand"

    # Token 1: 'you'
    assert features[1]["q_inversion_curr"] is True
    assert features[1]["-1:token"] == "did"
    assert features[1]["+1:token"] == "understand"


def test_utterance_boundary_ablation_switch(sample_tokens):
    """Ablation switch: include_utterance_boundaries=False must strip BOS and EOS."""
    with_boundaries = extract_sequence_features(
        sample_tokens,
        config=FeatureConfig.basic(include_utterance_boundaries=True)
    )
    without_boundaries = extract_sequence_features(
        sample_tokens,
        config=FeatureConfig.basic(include_utterance_boundaries=False)
    )

    assert "BOS" in with_boundaries[0]
    assert "EOS" in with_boundaries[-1]
    assert "-1:BOS" in with_boundaries[0]
    assert "+1:EOS" in with_boundaries[-1]

    assert "BOS" not in without_boundaries[0]
    assert "EOS" not in without_boundaries[-1]
    assert "-1:BOS" not in without_boundaries[0]
    assert "+1:EOS" not in without_boundaries[-1]


def test_config_serialization():
    """FeatureConfig must serialize to and deserialize from dictionaries."""
    orig = FeatureConfig.improved(window_size=3, use_token_ngrams=False)
    d = orig.to_dict()
    assert d["window_size"] == 3
    assert d["use_token_ngrams"] is False
    restored = FeatureConfig.from_dict(d)
    assert restored == orig


def test_feature_value_types(sample_tokens):
    """Feature values must strictly be str, int, float, or bool."""
    features = extract_sequence_features(sample_tokens, config=FeatureConfig.improved())
    allowed_types = (str, int, float, bool)
    for feat_dict in features:
        for k, v in feat_dict.items():
            assert isinstance(v, allowed_types), f"Feature {k} has unexpected type {type(v)}"
