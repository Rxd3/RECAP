"""Word lists and constants for CRF feature extraction (all lowercased); re-exported by ``crf_features``."""

from __future__ import annotations

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
