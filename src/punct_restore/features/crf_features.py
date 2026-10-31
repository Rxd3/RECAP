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

from dataclasses import dataclass, field, fields
import json
from typing import Any, Mapping, Sequence

# =============================================================================================
# COURSE presets (basic, improved): Member 3's original feature groups.
# =============================================================================================
# Lexical word lists for linguistic feature extraction (all lowercased)
WH_WORDS: frozenset[str] = frozenset({
    "what", "why", "when", "where", "who", "whom", "whose", "how", "which"
})

AUXILIARY_VERBS: frozenset[str] = frozenset({
    "do", "did", "does", "is", "are", "was", "were", "am",
    "can", "could", "will", "would", "should", "shall",
    "may", "might", "must", "have", "has", "had"
})

SUBJECT_PRONOUNS: frozenset[str] = frozenset({
    "you", "we", "they", "he", "she", "it", "i", "there"
})

CONJUNCTIONS: frozenset[str] = frozenset({
    "and", "but", "or", "nor", "for", "yet", "so",
    "because", "although", "though", "since", "unless",
    "while", "whereas", "if"
})

DISCOURSE_MARKERS: frozenset[str] = frozenset({
    "however", "therefore", "moreover", "furthermore", "meanwhile",
    "actually", "basically", "well", "anyway", "first", "second", "finally"
})

HONORIFICS: frozenset[str] = frozenset({
    "mr", "mrs", "ms", "dr", "prof"
})

DAYS_AND_MONTHS: frozenset[str] = frozenset({
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december"
})

# =============================================================================================
# TEAM EXTENSIONS (filler cues, question lookback): opt-in switches, off in both course presets.
# =============================================================================================
FILLERS: frozenset[str] = frozenset({"um", "uh", "er", "erm", "like"})

FILLER_BIGRAMS: frozenset[tuple[str, str]] = frozenset({("you", "know"), ("i", "mean")})

# Question lookback: a wh-word + auxiliary pair anywhere in the last LOOKBACK_WINDOW tokens
# (current one included), and tag-question endings ("..., isn't it", "..., right").
LOOKBACK_WINDOW = 12
TAG_CUE_WORDS: frozenset[str] = frozenset({"right", "correct", "huh", "eh"})
TAG_CUE_PRONOUNS: frozenset[str] = frozenset({"it", "they", "you", "we", "he", "she", "there"})

# Word clusters: a word -> cluster id map built offline. It travels inside the config, hence inside every
# saved model, so a trained model needs no side file. A word missing from the map gets this label.
UNKNOWN_CLUSTER_LABEL = "unk"


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
            Stored as a plain-dict copy. Part of ==, left out of the hash (a dict is unhashable) and of the repr (it is huge).
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


def _length_bucket(length: int) -> str:
    if length <= 1:
        return "1"
    if length <= 3:
        return "short"
    if length <= 7:
        return "medium"
    return "long"


def _position_bucket(index: int, total: int) -> str:
    if total <= 1:
        return "single"
    if index == 0:
        return "start"
    if index == total - 1:
        return "end"
    rel = index / total
    if rel < 0.33:
        return "early"
    if rel > 0.66:
        return "late"
    return "middle"


def _filler_features(tokens: Sequence[str], i: int) -> dict[str, bool]:
    """Spoken-disfluency flags for position i; only the true flags are returned."""
    n = len(tokens)
    flags = {
        "is_filler": tokens[i] in FILLERS,
        "+1:is_filler": i + 1 < n and tokens[i + 1] in FILLERS,
        "starts_filler_bigram": i + 1 < n and (tokens[i], tokens[i + 1]) in FILLER_BIGRAMS,
        "+1:starts_filler_bigram": i + 2 < n and (tokens[i + 1], tokens[i + 2]) in FILLER_BIGRAMS,
        "is_repeat": i > 0 and tokens[i] == tokens[i - 1],
    }
    return {name: True for name, on in flags.items() if on}


def _question_lookback_features(tokens: Sequence[str], i: int) -> dict[str, bool]:
    """Question-lookback flags for position i; only the true flags are returned."""
    start = max(0, i - (LOOKBACK_WINDOW - 1))
    wh_aux = any(tokens[j] in WH_WORDS and tokens[j + 1] in AUXILIARY_VERBS for j in range(start, i))
    tag_pronoun = tokens[i] in TAG_CUE_PRONOUNS and i > 0 and tokens[i - 1].endswith("n't")
    flags = {
        "wh_aux_lookback": wh_aux,
        "is_tag_question_cue": tokens[i] in TAG_CUE_WORDS or tag_pronoun,
    }
    return {name: True for name, on in flags.items() if on}


def _cluster_label(clusters: Mapping[str, int], word: str) -> str:
    """'c<id>' for a clustered word (looked up in lowercase), UNKNOWN_CLUSTER_LABEL otherwise (id 0 is real)."""
    cluster_id = clusters.get(word.lower())
    return UNKNOWN_CLUSTER_LABEL if cluster_id is None else f"c{cluster_id}"


def _cluster_features(tokens: Sequence[str], i: int, clusters: Mapping[str, int]) -> dict[str, str]:
    """Cluster labels of the word at i, its left neighbour and its right neighbour, and the current+next pair."""
    current = _cluster_label(clusters, tokens[i])
    feats = {"cl0": current}
    if i > 0:
        feats["cl-1"] = _cluster_label(clusters, tokens[i - 1])
    if i < len(tokens) - 1:
        following = _cluster_label(clusters, tokens[i + 1])
        feats["cl+1"] = following
        feats["cl0|cl+1"] = f"{current}|{following}"
    return feats


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

    # Resolve configuration
    cfg: FeatureConfig
    if config is None:
        cfg = FeatureConfig.basic()
    elif isinstance(config, FeatureConfig):
        cfg = config
    elif isinstance(config, Mapping):
        cfg = FeatureConfig.from_dict(config)
    else:
        raise TypeError(f"Invalid config type: {type(config).__name__}")

    total_tokens = len(tokens)
    seq_features: list[dict[str, str | int | float | bool]] = []

    for i in range(total_tokens):
        # Base features for current token
        token_feats = _extract_token_features(tokens[i], cfg)

        # Positional buckets
        if cfg.use_positional_buckets:
            token_feats["pos_bucket"] = _position_bucket(i, total_tokens)

        # Utterance boundaries (BOS / EOS)
        if cfg.include_utterance_boundaries:
            if i == 0:
                token_feats["BOS"] = True
            if i == total_tokens - 1:
                token_feats["EOS"] = True

        # Contextual window features (negative offsets: past, positive offsets: future)
        radius = max(1, cfg.window_size)
        for offset in range(1, radius + 1):
            # Left context (past tokens)
            left_idx = i - offset
            if left_idx >= 0:
                left_tok = tokens[left_idx]
                token_feats[f"-{offset}:token"] = left_tok
                if cfg.use_affixes and len(left_tok) >= 3:
                    token_feats[f"-{offset}:suff3"] = left_tok[-3:]
                if cfg.use_question_cues:
                    token_feats[f"-{offset}:is_wh"] = left_tok in WH_WORDS
                    token_feats[f"-{offset}:is_aux"] = left_tok in AUXILIARY_VERBS
                if cfg.use_clause_cues:
                    token_feats[f"-{offset}:is_conj"] = left_tok in CONJUNCTIONS
            elif cfg.include_utterance_boundaries and offset == 1:
                token_feats["-1:BOS"] = True

            # Right context (future tokens)
            right_idx = i + offset
            if right_idx < total_tokens:
                right_tok = tokens[right_idx]
                token_feats[f"+{offset}:token"] = right_tok
                if cfg.use_affixes and len(right_tok) >= 3:
                    token_feats[f"+{offset}:pref3"] = right_tok[:3]
                if cfg.use_question_cues:
                    token_feats[f"+{offset}:is_subj"] = right_tok in SUBJECT_PRONOUNS
            elif cfg.include_utterance_boundaries and offset == 1:
                token_feats["+1:EOS"] = True

        # Token bigram combinations (w-1_w0 and w0_w+1)
        if cfg.use_token_ngrams:
            if i > 0:
                token_feats["bigram_prev"] = f"{tokens[i - 1]}_{tokens[i]}"
            if i < total_tokens - 1:
                token_feats["bigram_next"] = f"{tokens[i]}_{tokens[i + 1]}"

        # Syntactic question cue: auxiliary verb at start or near start followed by subject pronoun
        if cfg.use_question_cues:
            # Question inversion: aux + pronoun pattern at current and previous token
            if i > 0 and tokens[i - 1] in AUXILIARY_VERBS and tokens[i] in SUBJECT_PRONOUNS:
                token_feats["q_inversion_curr"] = True
            if i < total_tokens - 1 and tokens[i] in AUXILIARY_VERBS and tokens[i + 1] in SUBJECT_PRONOUNS:
                token_feats["q_inversion_next"] = True
            # Utterance starts with Wh-word or Auxiliary verb
            if tokens[0] in WH_WORDS:
                token_feats["utt_starts_wh"] = True
            if tokens[0] in AUXILIARY_VERBS:
                token_feats["utt_starts_aux"] = True

        # Team extensions (off unless switched on)
        if cfg.use_filler_cues:
            token_feats.update(_filler_features(tokens, i))
        if cfg.use_question_lookback:
            token_feats.update(_question_lookback_features(tokens, i))
        if cfg.use_word_clusters:  # __post_init__ guarantees a non-empty map
            token_feats.update(_cluster_features(tokens, i, cfg.word_clusters))

        seq_features.append(token_feats)

    return seq_features
