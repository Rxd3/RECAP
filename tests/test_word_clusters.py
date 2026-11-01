"""punct_restore.features.word_clusters on tiny fixture corpora (no real data)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import MappingProxyType

import pytest
from scipy import sparse

from punct_restore.features import word_clusters as wc

# "cat"/"dog" share every context, and so do "run"/"walk"; the two groups never mix.
SENTENCES = [
    ["the", "cat", "sat"],
    ["the", "dog", "sat"],
    ["we", "run", "fast"],
    ["we", "walk", "fast"],
]
CORPUS = SENTENCES * 10


def induce(corpus=CORPUS, **overrides):
    options = {"k": 2, "seed": 7, **overrides}
    return wc.induce_clusters(corpus, **options)


def make_map(**overrides) -> wc.ClusterMap:
    fields = {
        "k": 3, "seed": 1, "min_count": 5, "dims": 10, "window": 2, "vocabulary_size": 4,
        "train_ids_sha256": "a" * 64, "release": "rel", "mode": "A",
        "clusters": {"b": 2, "a": 0, "c": 1, "d": 1},
    }
    return wc.ClusterMap(**{**fields, **overrides})


def count_matrix(corpus, *, window, min_count=1):
    vocabulary, matrix = wc._count_cooccurrence(corpus, min_count=min_count, window=window)
    return vocabulary, matrix.toarray().tolist()


# ---------------------------------------------------------------- induce_clusters

def test_words_with_identical_contexts_share_a_cluster():
    clusters = induce()
    assert clusters["cat"] == clusters["dog"]
    assert clusters["run"] == clusters["walk"]
    assert clusters["cat"] != clusters["run"]


def test_ids_are_plain_ints_inside_the_requested_range():
    clusters = induce(k=3)
    assert set(clusters) == {"the", "cat", "dog", "sat", "we", "run", "walk", "fast"}
    assert all(type(cluster) is int and 0 <= cluster < 3 for cluster in clusters.values())


def test_same_input_and_seed_give_identical_json_bytes():
    first = induce(k=3, seed=11)
    second = induce([list(sentence) for sentence in CORPUS], k=3, seed=11)
    assert wc.map_to_json(make_map(clusters=first)) == wc.map_to_json(make_map(clusters=second))


def test_a_different_seed_still_gives_a_valid_map():
    clusters = induce(k=3, seed=12)
    assert set(clusters) == set(induce(k=3, seed=11))
    assert all(0 <= cluster < 3 for cluster in clusters.values())


def test_a_one_shot_generator_is_accepted():
    assert induce(iter(CORPUS)) == induce()


def test_min_count_drops_rare_words():
    clusters = induce(CORPUS + [["the", "emu", "sat"]] * 2, min_count=5)
    assert "emu" not in clusters
    assert "emu" in induce(CORPUS + [["the", "emu", "sat"]] * 2, min_count=2)


def test_words_are_lowercased_and_counted_together():
    shouty = [["The", "CAT", "Sat"] if index % 2 else ["the", "cat", "sat"] for index in range(30)]
    assert set(induce(shouty + CORPUS)) == {"the", "cat", "dog", "sat", "we", "run", "walk", "fast"}
    assert "CAT" not in induce(shouty + CORPUS)


@pytest.mark.parametrize(("overrides", "fragment"), [
    ({"k": 1}, "k must be"),
    ({"k": 9}, "vocabulary"),
    ({"k": "2"}, "k must be"),
    ({"k": True}, "k must be"),
    ({"dims": 1}, "dims must be"),
    ({"window": 0}, "window must be"),
    ({"min_count": 0}, "min_count must be"),
    ({"seed": -1}, "seed must be"),
    ({"seed": 1.5}, "seed must be"),
    ({"seed": 2**32}, "seed must be an integer in 0..4294967295"),  # numpy / scikit-learn seed limit
])
def test_invalid_parameters_raise_one_line_value_errors(overrides, fragment):
    with pytest.raises(ValueError, match=fragment) as caught:
        induce(**overrides)
    assert "\n" not in str(caught.value)


@pytest.mark.parametrize(("overrides", "fragment"), [({"k": 1}, "k must be"), ({"seed": 2**32}, "seed must be")])
def test_parameters_are_checked_before_the_corpus_is_read(overrides, fragment):
    def exploding():
        raise AssertionError("the corpus was read")
        yield  # pragma: no cover

    with pytest.raises(ValueError, match=fragment):
        induce(exploding(), **overrides)


def test_the_largest_allowed_seed_works():
    assert set(induce(seed=2**32 - 1)) == set(induce())


def test_an_empty_corpus_has_no_vocabulary_for_k():
    with pytest.raises(ValueError, match="vocabulary"):
        induce([])


@pytest.mark.parametrize("corpus", ["cat", ["the cat", "sat"], [["the", 3]], [["the", None]]])
def test_non_token_input_is_rejected(corpus):
    with pytest.raises(ValueError, match="tokens"):
        induce(corpus)


# ---------------------------------------------------------------- co-occurrence counts

def test_window_one_counts_neighbours_symmetrically_and_stops_at_sentence_ends():
    vocabulary, dense = count_matrix([["a", "b", "a"], ["b", "c"]], window=1)
    assert vocabulary == ["a", "b", "c"]
    assert dense == [[0, 2, 0], [2, 0, 1], [0, 1, 0]]


def test_window_two_reaches_two_positions_but_not_across_sentences():
    _, dense = count_matrix([["a", "b", "c"]], window=2)
    assert dense == [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    _, apart = count_matrix([["a"], ["b"]], window=2)
    assert apart == [[0, 0], [0, 0]]


def test_out_of_vocabulary_neighbours_are_skipped_not_removed():
    # "r" is rare: a-r-b are not neighbours at window 1, only the a-b of the second sentence is.
    vocabulary, dense = count_matrix([["a", "r", "b"], ["a", "b"]], window=1, min_count=2)
    assert vocabulary == ["a", "b"]
    assert dense == [[0, 1], [1, 0]]


def test_vocabulary_is_sorted_lowercase_and_filtered_by_min_count():
    vocabulary, _ = count_matrix([["Zed", "ant", "zed", "Ant"], ["mid"]], window=1, min_count=2)
    assert vocabulary == ["ant", "zed"]


def test_chunking_does_not_change_the_counts(monkeypatch):
    corpus = [["a", "b", "c", "d"], ["c", "a", "b"], ["d", "d", "a", "c", "b"]] * 3
    whole = count_matrix(corpus, window=3)
    monkeypatch.setattr(wc, "_CHUNK_ROWS", 4)
    assert count_matrix(corpus, window=3) == whole


# ---------------------------------------------------------------- PPMI and k-means input

def test_ppmi_matches_a_hand_computed_matrix_and_clips_negative_pmi_to_zero():
    counts = sparse.csr_array([[0, 1, 5], [1, 0, 3], [5, 3, 0]], dtype=float)
    # row sums 6, 4, 8 and total N = 18, so PMI(i, j) = log(n_ij * 18 / (n_i * n_j)):
    assert math.log(1 * 18 / (6 * 4)) < 0  # the (0, 1) pair is below chance, its PMI is clipped to 0
    expected = [
        [0.0, 0.0, math.log(5 * 18 / (6 * 8))],
        [0.0, 0.0, math.log(3 * 18 / (4 * 8))],
        [math.log(5 * 18 / (6 * 8)), math.log(3 * 18 / (4 * 8)), 0.0],
    ]
    result = wc._ppmi(counts)
    for got_row, want_row in zip(result.toarray().tolist(), expected, strict=True):
        assert got_row == pytest.approx(want_row)
    assert result.nnz == 4  # the clipped pair is not stored


def test_k_means_gets_unit_length_vectors(monkeypatch):
    seen = []

    class Spy(wc.KMeans):
        def fit_predict(self, X, *args, **kwargs):
            seen.append(X)
            return super().fit_predict(X, *args, **kwargs)

    monkeypatch.setattr(wc, "KMeans", Spy)
    induce()
    (vectors,) = seen
    assert (vectors**2).sum(axis=1).tolist() == pytest.approx([1.0] * len(vectors))


# ---------------------------------------------------------------- JSON format

def test_json_has_the_pinned_fields_sorted_and_ends_with_a_newline():
    text = wc.map_to_json(make_map())
    payload = json.loads(text)
    assert text.endswith("\n") and not text.endswith("\n\n")
    assert list(payload) == sorted(payload)
    assert payload["format_version"] == wc.FORMAT_VERSION == 1
    assert set(payload) == {
        "format_version", "k", "seed", "min_count", "dims", "window", "vocabulary_size",
        "train_ids_sha256", "release", "mode", "clusters",
    }
    assert list(payload["clusters"]) == ["a", "b", "c", "d"]
    assert payload["clusters"] == {"a": 0, "b": 2, "c": 1, "d": 1}


def test_round_trip_through_a_file(tmp_path):
    original = make_map(clusters={"café": 1, "naïve": 0, "x": 2, "y": 2}, vocabulary_size=4)
    path = tmp_path / "map.json"
    path.write_text(wc.map_to_json(original), encoding="utf-8", newline="\n")
    loaded = wc.load_cluster_map(path)
    assert loaded == original
    assert wc.map_to_json(loaded) == path.read_text(encoding="utf-8")


def test_loaded_clusters_are_read_only(tmp_path):
    assert isinstance(make_map().clusters, MappingProxyType)
    path = tmp_path / "map.json"
    path.write_text(wc.map_to_json(make_map()), encoding="utf-8")
    with pytest.raises(TypeError):
        wc.load_cluster_map(path).clusters["a"] = 1  # type: ignore[index]


def test_the_cluster_map_cannot_be_reassigned():
    with pytest.raises(AttributeError):
        make_map().k = 5  # type: ignore[misc]


def test_a_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        wc.load_cluster_map(tmp_path / "absent.json")


# ---------------------------------------------------------------- malformed files

def _payload() -> dict:
    return json.loads(wc.map_to_json(make_map()))


def _edit(change):
    payload = _payload()
    change(payload)
    return json.dumps(payload)


MALFORMED = {
    "wrong version": (_edit(lambda p: p.update(format_version=2)), "format_version"),
    "id below zero": (_edit(lambda p: p["clusters"].update(a=-1)), "outside"),
    "id at k": (_edit(lambda p: p["clusters"].update(a=3)), "outside"),
    "id is bool": (_edit(lambda p: p["clusters"].update(a=True)), "clusters"),
    "id is str": (_edit(lambda p: p["clusters"].update(a="0")), "clusters"),
    "id is float": (_edit(lambda p: p["clusters"].update(a=1.5)), "clusters"),
    "k is str": (_edit(lambda p: p.update(k="3")), "k"),
    "k is bool": (_edit(lambda p: p.update(k=True)), "k"),
    "k is zero": (_edit(lambda p: p.update(k=0)), "k"),
    "k is one": (_edit(lambda p: p.update(k=1)), "k must be an integer >= 2"),
    "seed is float": (_edit(lambda p: p.update(seed=1.0)), "seed"),
    "window is null": (_edit(lambda p: p.update(window=None)), "window"),
    "negative dims": (_edit(lambda p: p.update(dims=-1)), "dims"),
    "sha is int": (_edit(lambda p: p.update(train_ids_sha256=5)), "train_ids_sha256"),
    "release is null": (_edit(lambda p: p.update(release=None)), "release"),
    "mode is list": (_edit(lambda p: p.update(mode=["A"])), "mode"),
    "clusters is list": (_edit(lambda p: p.update(clusters=[0, 1])), "clusters"),
    "vocabulary_size mismatch": (_edit(lambda p: p.update(vocabulary_size=5)), "vocabulary_size"),
    "missing field": (_edit(lambda p: p.pop("mode")), "missing"),
    "extra field": (_edit(lambda p: p.update(extra=1)), "unexpected"),
    "not an object": ("[1, 2]", "object"),
    "invalid json": ('{"k": ', "JSON"),
    "nested too deeply": ("[" * 100_000, "invalid JSON"),  # the json module gives up with a RecursionError
    "empty file": ("", "JSON"),
}


@pytest.mark.parametrize(("text", "fragment"), list(MALFORMED.values()), ids=list(MALFORMED))
def test_malformed_files_raise_one_line_value_errors_naming_the_file(tmp_path, text, fragment):
    path = tmp_path / "bad_map.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=fragment) as caught:
        wc.load_cluster_map(path)
    message = str(caught.value)
    assert message.startswith("bad_map.json: ") and "\n" not in message
    assert str(tmp_path) not in message


def test_a_file_that_is_not_utf8_is_rejected(tmp_path: Path):
    path = tmp_path / "bad_map.json"
    path.write_bytes(b"\xff\xfe\x00{")
    with pytest.raises(ValueError, match="bad_map.json: not valid UTF-8"):
        wc.load_cluster_map(path)
