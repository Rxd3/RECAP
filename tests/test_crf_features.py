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


FILLER_KEYS = (
    "is_filler",
    "+1:is_filler",
    "starts_filler_bigram",
    "+1:starts_filler_bigram",
    "is_repeat",
)


def test_improved_and_basic_presets_unchanged():
    """Presets keep today's values plus the opt-in switches off; no filler keys leak when off.

    Deliberate rule change: the snapshot gains use_word_clusters=False and word_clusters=None (the
    word-cluster switch), and nothing else.
    """
    assert FeatureConfig.basic().to_dict() == {
        "preset": "basic",
        "window_size": 1,
        "use_affixes": True,
        "use_shape": True,
        "use_internal_punct": True,
        "use_token_ngrams": False,
        "use_question_cues": False,
        "use_clause_cues": False,
        "use_casing_cues": False,
        "include_utterance_boundaries": True,
        "use_positional_buckets": False,
        "use_filler_cues": False,
        "use_question_lookback": False,
        "use_word_clusters": False,
        "word_clusters": None,
        "version": "v1-basic",
    }
    assert FeatureConfig.improved().to_dict() == {
        "preset": "improved",
        "window_size": 2,
        "use_affixes": True,
        "use_shape": True,
        "use_internal_punct": True,
        "use_token_ngrams": True,
        "use_question_cues": True,
        "use_clause_cues": True,
        "use_casing_cues": True,
        "include_utterance_boundaries": True,
        "use_positional_buckets": True,
        "use_filler_cues": False,
        "use_question_lookback": False,
        "use_word_clusters": False,
        "word_clusters": None,
        "version": "v1-improved",
    }
    tokens = ["so", "um", "i", "i", "mean", "you", "know"]
    for cfg in (FeatureConfig.basic(), FeatureConfig.improved()):
        for feats in extract_sequence_features(tokens, config=cfg):
            assert not any(key in feats for key in FILLER_KEYS)


def test_old_saved_config_without_filler_key_loads_with_default_off():
    """Configs saved before the flag existed must still load, with the flag off."""
    legacy = FeatureConfig.improved().to_dict()
    del legacy["use_filler_cues"]
    assert FeatureConfig.from_dict(legacy).use_filler_cues is False


def test_filler_features():
    """Filler flags fire only where true, and are absent elsewhere."""
    tokens = ["so", "um", "i", "i", "mean", "you", "know"]
    feats = extract_sequence_features(tokens, config=FeatureConfig.improved(use_filler_cues=True))

    assert feats[1]["is_filler"] is True
    assert feats[0]["+1:is_filler"] is True
    assert feats[3]["starts_filler_bigram"] is True
    assert feats[5]["starts_filler_bigram"] is True
    assert feats[3]["is_repeat"] is True
    assert feats[2]["+1:starts_filler_bigram"] is True
    assert feats[4]["+1:starts_filler_bigram"] is True

    # Flags that are false are absent, not stored as False.
    assert "is_filler" not in feats[0]
    assert "+1:is_filler" not in feats[1]
    assert "starts_filler_bigram" not in feats[2]
    assert "is_repeat" not in feats[2]
    assert "is_repeat" not in feats[0]
    assert "+1:starts_filler_bigram" not in feats[5]
    assert "+1:starts_filler_bigram" not in feats[6]
    assert "starts_filler_bigram" not in feats[6]


def test_is_repeat_never_wraps_to_the_last_token():
    """Index 0 has no previous token; tokens[-1] must not count as one."""
    tokens = ["um", "so", "um"]
    feats = extract_sequence_features(tokens, config=FeatureConfig.improved(use_filler_cues=True))

    assert "is_repeat" not in feats[0]
    assert "is_repeat" not in feats[2]


# --- team extension: question lookback (wh_aux_lookback, is_tag_question_cue) ----------------------

LOOKBACK_KEYS = ("wh_aux_lookback", "is_tag_question_cue")


def _fired(tokens: list[str], key: str) -> list[int]:
    """Indices at which the flag is present (flags are only stored when true)."""
    feats = extract_sequence_features(tokens, config=FeatureConfig.improved(use_question_lookback=True))
    assert all(f[key] is True for f in feats if key in f)
    return [i for i, f in enumerate(feats) if key in f]


def test_old_saved_config_without_lookback_key_loads_with_default_off():
    """Configs saved before the switch existed must still load, with the switch off."""
    legacy = FeatureConfig.improved().to_dict()
    del legacy["use_question_lookback"]
    assert FeatureConfig.from_dict(legacy).use_question_lookback is False


def test_lookback_keys_absent_when_the_switch_is_off():
    """Basic and improved presets never emit the lookback keys, even on tokens that would fire."""
    tokens = ["what", "is", "it", "you", "know", "right", "isn't", "it"]
    for cfg in (FeatureConfig.basic(), FeatureConfig.improved()):
        for feats in extract_sequence_features(tokens, config=cfg):
            assert not any(key in feats for key in LOOKBACK_KEYS)


def test_one_switch_turns_on_both_lookback_features_for_any_preset():
    tokens = ["what", "is", "it", "isn't", "it"]
    for cfg in (FeatureConfig.basic(use_question_lookback=True), FeatureConfig.improved(use_question_lookback=True)):
        feats = extract_sequence_features(tokens, config=cfg)
        assert feats[1]["wh_aux_lookback"] is True
        assert feats[4]["is_tag_question_cue"] is True


def test_wh_aux_lookback_fires_from_the_aux_onward():
    """The (what, is) pair at (0, 1) is seen from index 1 on, never at the wh-word itself."""
    tokens = ["what", "is", "it", "you", "know", "really"]
    assert _fired(tokens, "wh_aux_lookback") == [1, 2, 3, 4, 5]


def test_wh_aux_lookback_window_is_the_previous_twelve_tokens_including_the_current():
    """Window = tokens[i-11 .. i]: a pair at (0, 1) is seen at i = 1..11 but not at i = 12."""
    tokens = ["what", "is"] + ["x"] * 12
    assert _fired(tokens, "wh_aux_lookback") == list(range(1, 12))


def test_wh_aux_lookback_last_position_inside_the_window():
    """A pair whose wh-word is exactly 11 tokens back still counts; 12 back does not."""
    tokens = ["x", "what", "is"] + ["x"] * 11
    assert _fired(tokens, "wh_aux_lookback") == list(range(2, 13))


@pytest.mark.parametrize(
    "tokens",
    [
        ["what", "time", "is", "it"],  # wh-word not followed by an auxiliary
        ["what", "the", "is"],  # not adjacent
        ["how", "really", "can", "we"],  # not adjacent
        ["is", "what", "now"],  # aux before wh
        ["what"],  # wh-word alone
        ["is", "it"],  # auxiliary alone
    ],
    ids=["no_aux", "gap_the", "gap_really", "aux_first", "wh_alone", "aux_alone"],
)
def test_wh_aux_lookback_needs_an_adjacent_wh_then_aux_pair(tokens):
    assert _fired(tokens, "wh_aux_lookback") == []


def test_tag_question_cue_pronoun_after_negative_contraction():
    assert _fired(["it", "is", "nice", "isn't", "it"], "is_tag_question_cue") == [4]


def test_tag_question_cue_trailing_word():
    assert _fired(["you", "know", "right"], "is_tag_question_cue") == [2]


@pytest.mark.parametrize("word", ["right", "correct", "huh", "eh"])
def test_tag_question_cue_words_fire_anywhere(word):
    assert _fired(["so", word, "then"], "is_tag_question_cue") == [1]


@pytest.mark.parametrize("pronoun", ["it", "they", "you", "we", "he", "she", "there"])
def test_tag_question_cue_every_listed_pronoun_after_nt(pronoun):
    assert _fired(["don't", pronoun], "is_tag_question_cue") == [1]


@pytest.mark.parametrize(
    "tokens",
    [
        ["it", "is", "nice", "you"],  # pronoun not after n't
        ["he", "is"],  # pronoun at index 0
        ["you"],  # single-token pronoun
        ["she", "went", "isn't"],  # index 0 must not read tokens[-1]
        ["isn't", "i"],  # i is not a tag pronoun
        ["isn't", "nice"],  # n't followed by a non-pronoun
        ["not", "it"],  # previous token only resembles n't
    ],
    ids=["no_nt", "index0", "single", "no_wraparound", "pronoun_i", "non_pronoun", "not_nt"],
)
def test_tag_question_cue_absent_otherwise(tokens):
    assert _fired(tokens, "is_tag_question_cue") == []
