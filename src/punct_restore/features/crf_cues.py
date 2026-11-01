"""Per-position cue helpers for CRF feature extraction (buckets, filler, question lookback, word clusters).

They depend only on the word lists in ``crf_lexicons``; ``crf_features`` composes them into the feature dicts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .crf_lexicons import (
    AUXILIARY_VERBS,
    FILLER_BIGRAMS,
    FILLERS,
    LOOKBACK_WINDOW,
    TAG_CUE_PRONOUNS,
    TAG_CUE_WORDS,
    UNKNOWN_CLUSTER_LABEL,
    WH_WORDS,
)


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
