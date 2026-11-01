"""Data-driven word clusters for CRF features: PPMI co-occurrence vectors, truncated SVD, k-means.

``induce_clusters`` maps every frequent (lowercased) word to a cluster id. The map travels as a small JSON file
(``map_to_json`` / ``load_cluster_map``); the CRF turns an id into its feature value in ``crf_features``.

The corpus is held as sparse position-by-word incidence matrices, so only ``scipy.sparse`` and scikit-learn are
used here (numpy arrives through them and is never imported directly).
"""

from __future__ import annotations

import json
from array import array
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

FORMAT_VERSION = 1

_KMEANS_N_INIT = 4
_MAX_SEED = 2**32 - 1  # scikit-learn / numpy random_state limit
_CHUNK_ROWS = 1_000_000  # corpus positions turned into word pairs at a time, which bounds the working memory
_INT_FIELDS = {"k": 2, "seed": 0, "min_count": 0, "dims": 0, "window": 0, "vocabulary_size": 0}
_STR_FIELDS = ("train_ids_sha256", "release", "mode")


@dataclass(frozen=True)
class ClusterMap:
    """An induced word clustering with the settings and the training IDs it came from."""

    k: int
    seed: int
    min_count: int
    dims: int
    window: int
    vocabulary_size: int
    train_ids_sha256: str
    release: str
    mode: str
    clusters: Mapping[str, int]  # lowercase word -> cluster id in 0..k-1, always a read-only view

    def __post_init__(self) -> None:
        object.__setattr__(self, "clusters", MappingProxyType(dict(self.clusters)))


# ---------------------------------------------------------------- induction

def _require_int(name: str, value: object, minimum: int, maximum: int | None = None) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        bound = f">= {minimum}" if maximum is None else f"in {minimum}..{maximum}"
        raise ValueError(f"{name} must be an integer {bound}")


def _lowercase(sentence: Sequence[str]) -> list[str]:
    if isinstance(sentence, str) or not all(isinstance(word, str) for word in sentence):
        raise ValueError("tokens must be a sequence of strings per sentence")
    return [word.lower() for word in sentence]


def _encode(token_lists: Iterable[Sequence[str]], window: int) -> tuple[dict[str, int], sparse.csr_array]:
    """Map each lowercase word to a raw id and build the (position x raw id) one-hot incidence of the corpus.

    Sentences are laid end to end with ``window`` empty rows between them, so two positions at most ``window``
    apart never belong to different sentences. The corpus is read once.
    """
    raw: dict[str, int] = {}
    positions, ids = array("q"), array("i")
    cursor = 0
    for sentence in token_lists:
        words = _lowercase(sentence)
        ids.extend([raw.setdefault(word, len(raw)) for word in words])
        positions.extend(range(cursor, cursor + len(words)))
        cursor += len(words) + window
    ones = array("d", [1.0]) * len(ids)
    incidence = sparse.coo_array((ones, (positions, ids)), shape=(cursor, len(raw)))
    return raw, incidence.tocsr()


def _cooccurrence(incidence: sparse.csr_array, window: int) -> sparse.csr_array:
    """Symmetric counts of word pairs at most ``window`` positions apart (empty rows pair with nothing)."""
    size = incidence.shape[1]
    total = sparse.csr_array((size, size), dtype=float)
    for start in range(0, incidence.shape[0], _CHUNK_ROWS):
        left = incidence[start : start + _CHUNK_ROWS]
        for distance in range(1, window + 1):
            right = incidence[start + distance : start + _CHUNK_ROWS + distance]
            if right.shape[0] == 0:
                break
            pairs = left[: right.shape[0]].T @ right
            total = total + pairs + pairs.T
    return total


def _count_cooccurrence(
    token_lists: Iterable[Sequence[str]], *, min_count: int, window: int
) -> tuple[list[str], sparse.csr_array]:
    """Sorted vocabulary (lowercase words seen ``min_count`` times) and its symmetric co-occurrence counts.

    The window runs over the original positions: a rare neighbour is skipped, not removed.
    """
    raw, incidence = _encode(token_lists, window)
    totals = incidence.sum(axis=0)
    vocabulary = sorted(word for word, index in raw.items() if totals[index] >= min_count)
    if not vocabulary:
        return [], sparse.csr_array((0, 0), dtype=float)
    return vocabulary, _cooccurrence(incidence[:, [raw[word] for word in vocabulary]], window)


def _ppmi(counts: sparse.csr_array) -> sparse.csr_array:
    """Positive pointwise mutual information: max(0, log(n_ij * N / (n_i * n_j))) on the stored pairs."""
    marginal = counts.sum(axis=1)
    marginal[marginal == 0] = 1.0  # a word without neighbours has no stored pair, so this only avoids 1/0
    inverse = sparse.diags_array(1.0 / marginal)
    ratio = (inverse @ counts @ inverse) * counts.sum()
    ratio.data -= 1.0
    pmi = ratio.log1p()  # log(1 + (ratio - 1)) on the stored entries only
    pmi.data[pmi.data < 0] = 0.0
    pmi.eliminate_zeros()
    return pmi


def _cluster_vectors(ppmi: sparse.csr_array, *, k: int, seed: int, dims: int) -> list[int]:
    components = min(dims, ppmi.shape[0] - 1)
    vectors = TruncatedSVD(n_components=components, random_state=seed).fit_transform(ppmi)
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=_KMEANS_N_INIT)
    return kmeans.fit_predict(normalize(vectors)).tolist()


def induce_clusters(
    token_lists: Iterable[Sequence[str]],
    *,
    k: int,
    seed: int,
    min_count: int = 5,
    dims: int = 100,
    window: int = 2,
) -> dict[str, int]:
    """Cluster the lowercase words seen at least ``min_count`` times into ``k`` groups.

    The same input and seed always give the same map. Raises ValueError for unusable settings (the checks
    that need only the settings run before the corpus is read) or when ``k`` exceeds the vocabulary size.
    """
    _require_int("k", k, 2)
    _require_int("seed", seed, 0, _MAX_SEED)
    _require_int("min_count", min_count, 1)
    _require_int("dims", dims, 2)
    _require_int("window", window, 1)
    vocabulary, counts = _count_cooccurrence(token_lists, min_count=min_count, window=window)
    if k > len(vocabulary):
        raise ValueError(f"k={k} exceeds the vocabulary size {len(vocabulary)} (min_count={min_count})")
    labels = _cluster_vectors(_ppmi(counts), k=k, seed=seed, dims=dims)
    return dict(zip(vocabulary, labels, strict=True))


# ---------------------------------------------------------------- JSON file

def map_to_json(cluster_map: ClusterMap) -> str:
    """The map as JSON text with sorted keys, so equal maps give equal bytes; ends with a newline."""
    payload = {
        "format_version": FORMAT_VERSION,
        "k": cluster_map.k,
        "seed": cluster_map.seed,
        "min_count": cluster_map.min_count,
        "dims": cluster_map.dims,
        "window": cluster_map.window,
        "vocabulary_size": cluster_map.vocabulary_size,
        "train_ids_sha256": cluster_map.train_ids_sha256,
        "release": cluster_map.release,
        "mode": cluster_map.mode,
        "clusters": dict(cluster_map.clusters),
    }
    return json.dumps(payload, sort_keys=True, indent=1) + "\n"


# A malformed map file raises ValueError, not TypeError, by the loader's contract.
def _check_fields(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    expected = {"format_version", "clusters", *_INT_FIELDS, *_STR_FIELDS}
    if missing := sorted(expected - payload.keys()):
        raise ValueError(f"missing field(s): {', '.join(missing)}")
    if extra := sorted(payload.keys() - expected):
        raise ValueError(f"unexpected field(s): {', '.join(extra)}")
    version = payload["format_version"]
    if type(version) is not int or version != FORMAT_VERSION:
        raise ValueError(f"unsupported format_version (expected {FORMAT_VERSION})")
    for field, minimum in _INT_FIELDS.items():
        _require_int(field, payload[field], minimum)
    for field in _STR_FIELDS:
        if not isinstance(payload[field], str):
            raise ValueError(f"{field} must be a string")
    return payload


def _check_clusters(clusters: object, k: int, vocabulary_size: int) -> dict[str, int]:
    if not isinstance(clusters, dict):
        raise ValueError("clusters must be an object of word -> cluster id")
    for cluster in clusters.values():
        if isinstance(cluster, bool) or not isinstance(cluster, int):
            raise ValueError("clusters must map each word to an integer cluster id")
        if not 0 <= cluster < k:
            raise ValueError(f"a cluster id is outside 0..{k - 1}")
    if vocabulary_size != len(clusters):
        raise ValueError(f"vocabulary_size {vocabulary_size} does not match the {len(clusters)} clusters entries")
    return clusters


def load_cluster_map(path: str | Path) -> ClusterMap:
    """Read and check a cluster-map file.

    Raises FileNotFoundError for a missing file and a one-line ValueError (starting with the file name) for
    anything malformed: bad JSON, wrong version, wrong field types, cluster ids outside 0..k-1, or a
    vocabulary_size that differs from the number of words.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name}: not valid UTF-8") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name}: invalid JSON ({exc.msg})") from exc
    try:
        payload = _check_fields(payload)
        clusters = _check_clusters(payload["clusters"], payload["k"], payload["vocabulary_size"])
    except ValueError as exc:
        raise ValueError(f"{path.name}: {exc}") from None
    return ClusterMap(clusters=clusters, **{key: payload[key] for key in (*_INT_FIELDS, *_STR_FIELDS)})
