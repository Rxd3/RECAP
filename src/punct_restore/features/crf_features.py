"""CRF feature extraction module for punctuation and casing restoration.

Owned by Member 3 (expanded scope covering CRF model and feature engineering).
Follows the shared feature contract from docs/MODEL_CONTRACT.md:
    extract_sequence_features(tokens, *, config) -> list[dict[str, str | int | float | bool]]

Guarantees:
- Output length exactly matches input token count (one dictionary per token).
- Completely gold-free: inputs are lowercased tokens; no access to gold punctuation or casing.
- Deterministic, side-effect free extraction without input mutation.
- Configurable feature presets ('basic' baseline and 'improved' enhanced feature sets).
- Controlled ablation switches to isolate contextual signals from positional shortcuts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

from .crf_cues import (
    _cluster_features,
    _filler_features,
    _length_bucket,
    _position_bucket,
    _question_lookback_features,
)
from .crf_lexicons import (
    AUXILIARY_VERBS,
    CONJUNCTIONS,
    DAYS_AND_MONTHS,
    DISCOURSE_MARKERS,
    HONORIFICS,
    SUBJECT_PRONOUNS,
    WH_WORDS,
)

__all__ = ["FeatureConfig", "extract_sequence_features"]

# old import path kept (restructure spec §2)
from .crf_lexicons import FILLER_BIGRAMS as FILLER_BIGRAMS
from .crf_lexicons import FILLERS as FILLERS
from .crf_lexicons import LOOKBACK_WINDOW as LOOKBACK_WINDOW
from .crf_lexicons import TAG_CUE_PRONOUNS as TAG_CUE_PRONOUNS
from .crf_lexicons import TAG_CUE_WORDS as TAG_CUE_WORDS
from .crf_lexicons import UNKNOWN_CLUSTER_LABEL as UNKNOWN_CLUSTER_LABEL

_Feats = dict[str, str | int | float | bool]


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for CRF sequence feature extraction.

    Attributes:
        preset: 'basic' or 'improved'.
        window_size: Context window radius (1 for basic [+/-1], 2 for improved [+/-2]).
        use_affixes: Extract character n-gram prefixes and suffixes (len 1..3).
        use_shape: Extract token length and digit/numeric properties.
        use_internal_punct: Check for internal hyphens, apostrophes, etc.
        use_token_ngrams: Extract contextual token bigrams (w-1_w0, w0_w+1).
        use_question_cues: Extract wh-words and auxiliary-subject inversion markers.
        use_clause_cues: Extract coordinating/subordinating conjunctions and discourse markers.
        use_casing_cues: Extract lexical triggers for Mode C capitalization (I, days, titles).
        include_utterance_boundaries: Include BOS and EOS flags (allows shortcut ablation).
        use_positional_buckets: Include relative position in utterance (early/mid/late).
        use_filler_cues: Extract spoken-filler flags (um/uh/like, "you know", "i mean") and
            immediate word repeats. Team extension.
        use_question_lookback: Extract wh_aux_lookback (wh-word + auxiliary within the last 12
            tokens) and is_tag_question_cue (tag-question endings). Team extension.
        use_word_clusters: Extract the cluster label of the current, previous and next word and of the
            current+next pair (cl0, cl-1, cl+1, cl0|cl+1). Needs word_clusters. Team extension.
        word_clusters: Word -> cluster id map read by use_word_clusters; None when the switch is off.
            Stored as a plain-dict copy. Part of ==, left out of the hash (a dict is unhashable) and of the repr
            (it is huge).
            Team extension.
        version: Version identifier for reproducibility.

    Raises:
        ValueError: use_word_clusters is on but word_clusters is None or empty.
    """
    preset: str = "basic"
    window_size: int = 1
    use_affixes: bool = True
    use_shape: bool = True
    use_internal_punct: bool = True
    use_token_ngrams: bool = False
    use_question_cues: bool = False
    use_clause_cues: bool = False
    use_casing_cues: bool = False
    include_utterance_boundaries: bool = True
    use_positional_buckets: bool = False
    # --- team extensions (off in both course presets) ---
    use_filler_cues: bool = False
    use_question_lookback: bool = False
    use_word_clusters: bool = False
    word_clusters: Mapping[str, int] | None = field(default=None, hash=False, repr=False)
    version: str = "v1"

    def __post_init__(self) -> None:
        if self.use_word_clusters and not self.word_clusters:
            raise ValueError("use_word_clusters=True needs a non-empty word_clusters map")
        if self.word_clusters is not None:  # own plain-dict copy: detached from the caller, picklable
            object.__setattr__(self, "word_clusters", dict(self.word_clusters))

    @classmethod
    def basic(cls, **overrides: Any) -> FeatureConfig:
        """Create the verified basic baseline preset (window +/-1, affixes, shape)."""
        defaults: dict[str, Any] = {
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
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def improved(cls, **overrides: Any) -> FeatureConfig:
        """Create the improved feature preset (window +/-2, ngrams, question/clause cues)."""
        defaults: dict[str, Any] = {
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
        defaults.update(overrides)
        return cls(**defaults)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to a serializable dictionary (the cluster map becomes a plain dict copy)."""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        if self.word_clusters is not None:
            data["word_clusters"] = dict(self.word_clusters)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FeatureConfig:
        """Construct config from dictionary."""
        valid_keys = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


def _extract_token_features(token: str, config: FeatureConfig) -> dict[str, str | int | float | bool]:
    """Extract individual token features without sequence context."""
    features: dict[str, str | int | float | bool] = {
        "token": token,
    }

    if config.use_shape:
        token_len = len(token)
        features["len"] = token_len
        features["len_bucket"] = _length_bucket(token_len)
        features["is_numeric"] = token.isnumeric()
        features["has_digit"] = any(ch.isdigit() for ch in token)

    if config.use_internal_punct:
        features["has_hyphen"] = "-" in token
        features["has_apostrophe"] = "'" in token

    if config.use_affixes:
        t_len = len(token)
        # Prefixes (len 1..3)
        features["pref1"] = token[:1]
        if t_len >= 2:
            features["pref2"] = token[:2]
        if t_len >= 3:
            features["pref3"] = token[:3]
        # Suffixes (len 1..3)
        features["suff1"] = token[-1:]
        if t_len >= 2:
            features["suff2"] = token[-2:]
        if t_len >= 3:
            features["suff3"] = token[-3:]

    if config.use_question_cues:
        features["is_wh"] = token in WH_WORDS
        features["is_aux"] = token in AUXILIARY_VERBS

    if config.use_clause_cues:
        features["is_conjunction"] = token in CONJUNCTIONS
        features["is_discourse"] = token in DISCOURSE_MARKERS

    if config.use_casing_cues:
        features["is_pronoun_i"] = token == "i"
        features["is_honorific"] = token in HONORIFICS
        features["is_calendar"] = token in DAYS_AND_MONTHS

    return features


def _validate_tokens(tokens: Sequence[str]) -> None:
    """Raise TypeError or ValueError unless tokens is a non-empty sequence of non-blank strings."""
    if isinstance(tokens, (str, bytes)):
        raise TypeError("tokens must be a sequence of strings, not a single string")
    if not isinstance(tokens, Sequence):
        raise TypeError(f"tokens must be a Sequence, got {type(tokens).__name__}")
    if len(tokens) == 0:
        raise ValueError("tokens sequence cannot be empty")

    # Validate all tokens are non-empty strings
    for i, tok in enumerate(tokens):
        if not isinstance(tok, str):
            raise TypeError(f"Token at index {i} must be a str, got {type(tok).__name__}")
        if not tok or not tok.strip():
            raise ValueError(f"Token at index {i} is empty or whitespace-only")


def _resolve_config(config: FeatureConfig | Mapping[str, Any] | None) -> FeatureConfig:
    """Return a FeatureConfig from None (basic preset), an instance or a mapping; TypeError otherwise."""
    if config is None:
        return FeatureConfig.basic()
    if isinstance(config, FeatureConfig):
        return config
    if isinstance(config, Mapping):
        return FeatureConfig.from_dict(config)
    raise TypeError(f"Invalid config type: {type(config).__name__}")


def _left_context(tokens: Sequence[str], i: int, offset: int, cfg: FeatureConfig) -> _Feats:
    """Features of the token `offset` positions before i, or the -1:BOS marker when there is none."""
    feats: _Feats = {}
    left_idx = i - offset
    if left_idx >= 0:
        left_tok = tokens[left_idx]
        feats[f"-{offset}:token"] = left_tok
        if cfg.use_affixes and len(left_tok) >= 3:
            feats[f"-{offset}:suff3"] = left_tok[-3:]
        if cfg.use_question_cues:
            feats[f"-{offset}:is_wh"] = left_tok in WH_WORDS
            feats[f"-{offset}:is_aux"] = left_tok in AUXILIARY_VERBS
        if cfg.use_clause_cues:
            feats[f"-{offset}:is_conj"] = left_tok in CONJUNCTIONS
    elif cfg.include_utterance_boundaries and offset == 1:
        feats["-1:BOS"] = True
    return feats


def _right_context(tokens: Sequence[str], i: int, offset: int, cfg: FeatureConfig) -> _Feats:
    """Features of the token `offset` positions after i, or the +1:EOS marker when there is none."""
    feats: _Feats = {}
    right_idx = i + offset
    if right_idx < len(tokens):
        right_tok = tokens[right_idx]
        feats[f"+{offset}:token"] = right_tok
        if cfg.use_affixes and len(right_tok) >= 3:
            feats[f"+{offset}:pref3"] = right_tok[:3]
        if cfg.use_question_cues:
            feats[f"+{offset}:is_subj"] = right_tok in SUBJECT_PRONOUNS
    elif cfg.include_utterance_boundaries and offset == 1:
        feats["+1:EOS"] = True
    return feats


def _window_features(tokens: Sequence[str], i: int, cfg: FeatureConfig) -> _Feats:
    """Contextual window features (negative offsets: past, positive offsets: future), left then right per offset."""
    feats: _Feats = {}
    radius = max(1, cfg.window_size)
    for offset in range(1, radius + 1):
        feats.update(_left_context(tokens, i, offset, cfg))
        feats.update(_right_context(tokens, i, offset, cfg))
    return feats


def _ngram_features(tokens: Sequence[str], i: int) -> _Feats:
    """Token bigram combinations (w-1_w0 and w0_w+1)."""
    feats: _Feats = {}
    if i > 0:
        feats["bigram_prev"] = f"{tokens[i - 1]}_{tokens[i]}"
    if i < len(tokens) - 1:
        feats["bigram_next"] = f"{tokens[i]}_{tokens[i + 1]}"
    return feats


def _question_inversion_features(tokens: Sequence[str], i: int) -> _Feats:
    """Syntactic question cues: aux + subject pronoun inversion and an utterance opening with wh-word or aux."""
    feats: _Feats = {}
    # Question inversion: aux + pronoun pattern at current and previous token
    if i > 0 and tokens[i - 1] in AUXILIARY_VERBS and tokens[i] in SUBJECT_PRONOUNS:
        feats["q_inversion_curr"] = True
    if i < len(tokens) - 1 and tokens[i] in AUXILIARY_VERBS and tokens[i + 1] in SUBJECT_PRONOUNS:
        feats["q_inversion_next"] = True
    # Utterance starts with Wh-word or Auxiliary verb
    if tokens[0] in WH_WORDS:
        feats["utt_starts_wh"] = True
    if tokens[0] in AUXILIARY_VERBS:
        feats["utt_starts_aux"] = True
    return feats


def _features_at(tokens: Sequence[str], i: int, cfg: FeatureConfig) -> _Feats:
    """Feature dict of the token at position i; the key insertion order is kept stable for crfsuite."""
    total_tokens = len(tokens)
    token_feats: _Feats = _extract_token_features(tokens[i], cfg)

    # Positional buckets
    if cfg.use_positional_buckets:
        token_feats["pos_bucket"] = _position_bucket(i, total_tokens)

    # Utterance boundaries (BOS / EOS)
    if cfg.include_utterance_boundaries:
        if i == 0:
            token_feats["BOS"] = True
        if i == total_tokens - 1:
            token_feats["EOS"] = True

    token_feats.update(_window_features(tokens, i, cfg))
    if cfg.use_token_ngrams:
        token_feats.update(_ngram_features(tokens, i))
    if cfg.use_question_cues:
        token_feats.update(_question_inversion_features(tokens, i))

    # Team extensions (off unless switched on)
    if cfg.use_filler_cues:
        token_feats.update(_filler_features(tokens, i))
    if cfg.use_question_lookback:
        token_feats.update(_question_lookback_features(tokens, i))
    if cfg.use_word_clusters and cfg.word_clusters is not None:  # __post_init__ guarantees a non-empty map
        token_feats.update(_cluster_features(tokens, i, cfg.word_clusters))
    return token_feats


def extract_sequence_features(
    tokens: Sequence[str],
    *,
    config: FeatureConfig | Mapping[str, Any] | None = None,
) -> list[dict[str, str | int | float | bool]]:
    """Extract CRF feature dictionaries for an ordered sequence of tokens.

    Args:
        tokens: Sequence of non-empty lowercased string tokens.
        config: FeatureConfig instance or mapping with configuration options.
            Defaults to FeatureConfig.basic() if None.

    Returns:
        A list of feature dictionaries, exactly one per token.

    Raises:
        TypeError: If tokens is not a sequence or contains non-string items.
        ValueError: If tokens is empty or contains whitespace-only/empty tokens.
    """
    _validate_tokens(tokens)
    cfg = _resolve_config(config)
    return [_features_at(tokens, i, cfg) for i in range(len(tokens))]
