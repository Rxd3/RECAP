"""Hidden-event language model (comparison method H): probabilities, Viterbi, mode C, persistence."""

from __future__ import annotations

import copy
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any, Self

import pytest

from punct_restore import file_io
from punct_restore.models.hidden_event_lm import HiddenEventLM, UnigramTruecaser

ARTIFACT_ERROR = "invalid hidden-event LM artifact"


def rec(sample_id: str, tokens: list[str], punct: list[str], case: list[str] | None = None,
        mode: str = "A", split: str = "train") -> dict[str, Any]:
    return {"sample_id": sample_id, "split": split, "label_mode": mode, "tokens": list(tokens),
            "punct_labels": list(punct), "case_labels": list(case or ["LOWER"] * len(tokens))}


def obs(sample_id: str, tokens: list[str]) -> dict[str, Any]:
    return {"sample_id": sample_id, "tokens": list(tokens)}


def repeated(tokens: list[str], punct: list[str], times: int = 3, mode: str = "A") -> list[dict[str, Any]]:
    return [rec(f"r{i}", tokens, punct, mode=mode) for i in range(times)]


# ------------------------------------------------------------------ hand-computed probabilities
#
# Corpus (mode A, order 3, discount D = 0.5), streams with <s> padding:
#   u1: <s> <s> a b <PERIOD> </s>            u2: <s> <s> b <COMMA> a <PERIOD> </s>
# Predicted tokens (never <s>): u1 gives a, b, <PERIOD>, </s>; u2 gives b, <COMMA>, a, <PERIOD>, </s>.
# Unigram counts: a=2 b=2 <PERIOD>=2 </s>=2 <COMMA>=1, so N=9 and N1+()=5.
# Prediction vocabulary V = {a, b, <unk>, <COMMA>, <PERIOD>, <QUESTION>, </s>}, |V| = 7.
# Base floor D*N1+/N/|V| = 0.5*5/9/7 = 5/126.
#   P(a)          = (2-.5)/9 + 5/126 = 21/126 + 5/126 = 26/126 = 13/63
#   P(<unk>)      = 0 + 5/126                              P(<COMMA>) = .5/9 + 5/126 = 12/126
# Bigram contexts (total c(h), types N1+): (a): b,<PERIOD> -> 2,2   (b): <PERIOD>,<COMMA> -> 2,2
#   (<s>): a,b -> 2,2   (<PERIOD>): </s> x2 -> 2,1   (<COMMA>): a -> 1,1
# Trigram contexts: (<s>,<s>): a,b -> 2,2   (a,b): <PERIOD> -> 1,1   (<s>,a): b -> 1,1 ...
#   P(b | a)            = (1-.5)/2 + .5*2/2*P(b)               = 1/4 + 13/126 = 89/252
#   P(a | a)  (unseen)  = 0 + .5*2/2*P(a)                      = 13/126
#   P(<PERIOD> | b)     = 1/4 + 13/126                         = 89/252
#   P(<PERIOD> | a b)   = (1-.5)/1 + .5*1/1*P(<PERIOD> | b)    = 1/2 + 89/504 = 341/504
#   P(</s> | b)  (unseen) = .5*2/2*P(</s>) = 13/126;  P(</s> | a b) (unseen) = .5*1/1*13/126 = 13/252
#   P(a | <s> <s>)      = (1-.5)/2 + .5*2/2*P(a | <s>), P(a | <s>) = 1/4 + 13/126 = 89/252
#                       = 1/4 + 89/504 = 215/504
#   P(a | a a): context (a,a) unseen, so P(a | a a) = P(a | a) = 13/126
#   P(b | zzz zzz): zzz is OOV -> <unk>; contexts (<unk>,<unk>) and (<unk>) unseen -> P(b) = 13/63
#   P(zzz) = P(<unk>) = 5/126;   P(<unk> | a) = .5*2/2*5/126 = 5/252
TINY = [rec("u1", ["a", "b"], ["O", "PERIOD"]), rec("u2", ["b", "a"], ["COMMA", "PERIOD"])]
HAND = [
    ("a", (), 13 / 63),
    ("<unk>", (), 5 / 126),
    ("<COMMA>", (), 12 / 126),
    ("<QUESTION>", (), 5 / 126),
    ("zzz", (), 5 / 126),
    ("b", ("a",), 89 / 252),
    ("a", ("a",), 13 / 126),
    ("<PERIOD>", ("b",), 89 / 252),
    ("<PERIOD>", ("a", "b"), 341 / 504),
    ("</s>", ("b",), 13 / 126),
    ("</s>", ("a", "b"), 13 / 252),
    ("a", ("<s>", "<s>"), 215 / 504),
    ("a", ("a", "a"), 13 / 126),
    ("b", ("zzz", "zzz"), 13 / 63),
    ("<unk>", ("a",), 5 / 252),
]


@pytest.fixture
def tiny() -> HiddenEventLM:
    return HiddenEventLM("A", order=3, discount=0.5).fit(TINY)


@pytest.mark.parametrize(("token", "history", "expected"), HAND)
def test_logprob_matches_the_hand_computation(tiny, token, history, expected) -> None:
    assert tiny.logprob(token, history) == pytest.approx(math.log(expected), abs=1e-12)


@pytest.mark.parametrize("history", [(), ("a",), ("a", "b"), ("<s>", "<s>"), ("b", "<COMMA>"), ("zzz", "a")])
def test_the_distribution_over_the_prediction_vocabulary_sums_to_one(tiny, history) -> None:
    assert tiny.predicted_tokens == ("</s>", "<unk>", "<COMMA>", "<PERIOD>", "<QUESTION>", "a", "b")
    total = sum(math.exp(tiny.logprob(token, history)) for token in tiny.predicted_tokens)
    assert total == pytest.approx(1.0, abs=1e-12)


def test_the_start_symbol_is_never_predicted(tiny) -> None:
    with pytest.raises(ValueError, match="<s>"):
        tiny.logprob("<s>", ("a",))


def test_min_count_maps_rare_training_words_to_unk() -> None:
    # u1: a rare / u2: a a.  words a=3 rare=1.  Streams give N=8; a=3 <unk>=1 <PERIOD>=2 </s>=2.
    corpus = [rec("u1", ["a", "rare"], ["O", "PERIOD"]), rec("u2", ["a", "a"], ["O", "PERIOD"])]
    # min_count=2: |V| = a, <unk>, 3 events, </s> = 6, N1+ = 4, D = .5 -> P(<unk>) = .5/8 + .5*4/8/6 = 5/48
    rare = HiddenEventLM("A", discount=0.5, min_count=2).fit(corpus)
    assert "rare" not in rare.predicted_tokens
    assert rare.logprob("rare", ()) == rare.logprob("<unk>", ())
    assert rare.logprob("<unk>", ()) == pytest.approx(math.log(5 / 48), abs=1e-12)
    # min_count=1: |V| = 7, N1+ = 4 -> P(rare) = .5/8 + .5*4/8/7 = 11/112 and P(<unk>) = .5*4/8/7 = 1/28
    kept = HiddenEventLM("A", discount=0.5, min_count=1).fit(corpus)
    assert "rare" in kept.predicted_tokens
    assert kept.logprob("rare", ()) == pytest.approx(math.log(11 / 112), abs=1e-12)
    assert kept.logprob("<unk>", ()) == pytest.approx(math.log(1 / 28), abs=1e-12)


def test_unseen_words_are_decoded_as_unk_and_input_is_lowercased() -> None:
    model = HiddenEventLM("A").fit(repeated(["the", "cat", "sat"], ["O", "O", "PERIOD"]))
    assert model.decode(["THE", "Cat", "SAT"]) == ["O", "O", "PERIOD"]
    assert model.decode(["the", "zzz", "sat"]) == model.decode(["the", "qqq", "sat"])


def test_training_tokens_are_lowercased_too() -> None:
    upper = HiddenEventLM("A").fit([rec("u", ["The", "Cat"], ["O", "PERIOD"])])
    lower = HiddenEventLM("A").fit([rec("u", ["the", "cat"], ["O", "PERIOD"])])
    assert upper.predicted_tokens == lower.predicted_tokens == ("</s>", "<unk>", "<COMMA>", "<PERIOD>",
                                                                "<QUESTION>", "cat", "the")


# ------------------------------------------------------------------ Viterbi decisions


def test_viterbi_prefers_period_before_the_end_and_no_event_between_co_occurring_words() -> None:
    model = HiddenEventLM("A").fit(repeated(["the", "cat", "sat"], ["O", "O", "PERIOD"]))
    assert model.decode(["the", "cat", "sat"]) == ["O", "O", "PERIOD"]


def test_viterbi_inserts_a_comma_and_a_question_where_the_corpus_has_them() -> None:
    corpus = (repeated(["well", "i", "think", "so"], ["COMMA", "O", "O", "PERIOD"])
              + [rec(f"q{i}", ["how", "are", "you"], ["O", "O", "QUESTION"]) for i in range(3)])
    model = HiddenEventLM("A").fit(corpus)
    assert model.decode(["well", "i", "think", "so"]) == ["COMMA", "O", "O", "PERIOD"]
    assert model.decode(["how", "are", "you"]) == ["O", "O", "QUESTION"]


def test_viterbi_prefers_no_event_at_all_when_the_corpus_has_none() -> None:
    model = HiddenEventLM("A").fit(repeated(["the", "cat", "sat"], ["O", "O", "O"]))
    assert model.decode(["the", "cat", "sat"]) == ["O", "O", "O"]


def test_the_final_event_is_scored_together_with_the_end_symbol() -> None:
    always_period = HiddenEventLM("A").fit(repeated(["go", "now"], ["O", "PERIOD"]))
    never_final = HiddenEventLM("A").fit(repeated(["go", "now"], ["O", "O"]))
    assert always_period.decode(["go", "now"])[-1] == "PERIOD"
    assert never_final.decode(["go", "now"])[-1] == "O"


def test_mode_b_uses_its_extra_events() -> None:
    corpus = (repeated(["stop", "now"], ["EXCLAMATION", "O"], mode="B")
              + [rec(f"d{i}", ["wait", "here"], ["DASH", "O"], mode="B") for i in range(3)])
    model = HiddenEventLM("B").fit(corpus)
    assert model.decode(["stop", "now"]) == ["EXCLAMATION", "O"]
    assert model.decode(["wait", "here"]) == ["DASH", "O"]
    assert model.vocabulary == ("O", "COMMA", "PERIOD", "QUESTION", "EXCLAMATION", "COLON", "SEMICOLON", "DASH")


def test_ties_prefer_none_then_the_events_in_vocabulary_order() -> None:
    # COMMA and PERIOD are exactly symmetric in this corpus, so both paths score the same.
    model = HiddenEventLM("A").fit([rec("t1", ["x", "y"], ["COMMA", "O"]), rec("t2", ["x", "y"], ["PERIOD", "O"])])

    def score(first: str) -> float:
        return (model.logprob("x", ("<s>", "<s>")) + model.logprob(first, ("<s>", "x"))
                + model.logprob("y", ("x", first)) + model.logprob("</s>", (first, "y")))

    assert score("<COMMA>") == score("<PERIOD>")
    assert model.decode(["x", "y"]) == ["COMMA", "O"]


@pytest.mark.parametrize(("tokens", "rest"), [(["x", "y"], ["QUESTION"]), (["x", "y", "z"], ["O", "PERIOD"])])
def test_tied_paths_that_merge_keep_the_earlier_choice(tokens: list[str], rest: list[str]) -> None:
    # The first label is symmetric (COMMA or PERIOD) and the rest is fixed, so both first events reach the same
    # history with the same score: whichever merge keeps the path, the vocabulary-order COMMA must survive.
    corpus = [rec("m1", tokens, ["COMMA", *rest]), rec("m2", tokens, ["PERIOD", *rest])]
    assert HiddenEventLM("A").fit(corpus).decode(tokens) == ["COMMA", *rest]


# ------------------------------------------------------------------ brute force


def _random_corpus(mode: str, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    labels = ["O", "O", "O", "COMMA", "PERIOD", "QUESTION"] + (["DASH", "COLON"] if mode == "B" else [])
    words = ["a", "b", "c", "d"]
    corpus = []
    for index in range(16):
        tokens = [rng.choice(words) for _ in range(rng.randint(1, 4))]
        corpus.append(rec(f"s{index}", tokens, [rng.choice(labels) for _ in tokens], mode=mode))
    return corpus


def _path_score(model: HiddenEventLM, words: list[str], labels: tuple[str, ...]) -> float:
    history = ["<s>"] * (model.order - 1)
    total = 0.0
    stream = []
    for word, label in zip(words, labels, strict=True):
        stream.append(word)
        if label != "O":
            stream.append(f"<{label}>")
    stream.append("</s>")
    for token in stream:
        total += model.logprob(token, tuple(history))
        history = (history + [token])[1:]
    return total


@pytest.mark.parametrize(("mode", "order"), [("A", 2), ("A", 3), ("A", 4), ("B", 2), ("B", 3)])
def test_viterbi_equals_exhaustive_search(mode: str, order: int) -> None:
    model = HiddenEventLM(mode, order=order, discount=0.7).fit(_random_corpus(mode, seed=order))
    rng = random.Random(100 + order)
    events = model.vocabulary
    checked = 0
    for _ in range(14):
        words = [rng.choice(["a", "b", "c", "d", "zz"]) for _ in range(rng.randint(1, 3))]
        scored = sorted(((_path_score(model, words, path), path)
                         for path in itertools.product(events, repeat=len(words))), reverse=True)
        if scored[0][0] - scored[1][0] < 1e-9:
            continue  # a genuine tie: the tie-break test covers it
        assert tuple(model.decode(words)) == scored[0][1], words
        checked += 1
    assert checked >= 8


# ------------------------------------------------------------------ mode C and the truecaser


def test_the_truecaser_has_its_own_module_and_is_still_importable_from_the_lm_module() -> None:
    from punct_restore.models import truecaser

    assert truecaser.UnigramTruecaser is UnigramTruecaser


def test_truecaser_most_frequent_case_and_tie_order() -> None:
    truecaser = UnigramTruecaser.fit([
        (["hello", "it", "us", "nasa"], ["FIRST_UPPER", "LOWER", "FIRST_UPPER", "ALL_UPPER"]),
        (["Hello", "IT", "US", "nasa"], ["FIRST_UPPER", "FIRST_UPPER", "ALL_UPPER", "ALL_UPPER"]),
        (["hello", "x"], ["LOWER", "LOWER"]),
    ])
    labels = truecaser.predict(["hello", "it", "us", "nasa", "x"], ["O"] * 5)
    # hello: FIRST 2 vs LOWER 1; it: LOWER 1 = FIRST 1 -> LOWER; us: FIRST 1 = ALL 1 -> FIRST; nasa ALL 2
    assert labels == ["FIRST_UPPER", "LOWER", "FIRST_UPPER", "ALL_UPPER", "LOWER"]


def test_truecaser_unseen_words_follow_sentence_boundaries() -> None:
    truecaser = UnigramTruecaser.fit([(["seen"], ["LOWER"])])
    tokens = ["new1", "seen", "new2", "new3", "new4", "new5", "new6"]
    punct = ["O", "PERIOD", "QUESTION", "COMMA", "O", "EXCLAMATION", "O"]
    # new1: utterance start; new2: after PERIOD; new3: after QUESTION; new4: after COMMA; new5: after O;
    # new6: after EXCLAMATION; the seen word keeps its own case whatever precedes it
    assert truecaser.predict(tokens, punct) == [
        "FIRST_UPPER", "LOWER", "FIRST_UPPER", "FIRST_UPPER", "LOWER", "LOWER", "FIRST_UPPER"]


def test_truecaser_seen_word_ignores_the_sentence_boundary() -> None:
    truecaser = UnigramTruecaser.fit([(["seen"], ["LOWER"])])
    assert truecaser.predict(["seen", "seen"], ["PERIOD", "O"]) == ["LOWER", "LOWER"]


C_CORPUS = [
    rec("c1", ["hello", "nasa", "it"], ["COMMA", "O", "PERIOD"], ["FIRST_UPPER", "ALL_UPPER", "LOWER"], "C"),
    rec("c2", ["hello", "nasa", "it"], ["COMMA", "O", "PERIOD"], ["FIRST_UPPER", "ALL_UPPER", "LOWER"], "C"),
    rec("c3", ["is", "it", "you"], ["O", "O", "QUESTION"], ["FIRST_UPPER", "LOWER", "LOWER"], "C"),
]


def test_mode_c_labels_are_punctuation_joined_with_the_truecased_case() -> None:
    model = HiddenEventLM("C").fit(C_CORPUS)
    assert len(model.vocabulary) == 12 and "COMMA|ALL_UPPER" in model.vocabulary
    labels = model.decode(["hello", "nasa", "it"])
    assert labels == ["COMMA|FIRST_UPPER", "O|ALL_UPPER", "PERIOD|LOWER"]
    assert set(labels) <= set(model.vocabulary)


def test_mode_c_punctuation_part_equals_the_mode_a_model_on_the_same_records() -> None:
    punctuation_only = HiddenEventLM("A").fit([{**r, "label_mode": "A"} for r in C_CORPUS])
    joint = HiddenEventLM("C").fit(C_CORPUS)
    for tokens in (["hello", "nasa", "it"], ["is", "it", "you"], ["zzz", "hello"]):
        assert [label.split("|")[0] for label in joint.decode(tokens)] == punctuation_only.decode(tokens)


def test_mode_c_unseen_words_are_capitalised_only_at_the_start_or_after_a_predicted_terminal() -> None:
    model = HiddenEventLM("C").fit(C_CORPUS)
    seen = {"hello", "nasa", "it", "is", "you"}
    checked = 0
    for tokens in (["zzz", "nasa"], ["hello", "nasa", "it", "zzz"], ["is", "it", "qqq", "zzz", "you"],
                   ["zzz", "zzz", "zzz"], ["it", "zzz", "hello", "qqq"]):
        labels = model.decode(tokens)
        for index, token in enumerate(tokens):
            if token in seen:
                continue
            terminal = index == 0 or labels[index - 1].split("|")[0] in ("PERIOD", "QUESTION", "EXCLAMATION")
            assert labels[index].split("|")[1] == ("FIRST_UPPER" if terminal else "LOWER"), (tokens, index)
            checked += 1
    assert checked >= 8


# ------------------------------------------------------------------ predict contract


def test_predict_returns_contract_records_in_input_order_and_ignores_gold() -> None:
    model = HiddenEventLM("A", model_name="my_h").fit(repeated(["the", "cat", "sat"], ["O", "O", "PERIOD"]))
    records = [{**obs("z", ["the", "cat", "sat"]), "punct_labels": ["COMMA", "COMMA", "COMMA"], "label_mode": "B"},
               obs("a", ["cat"])]
    before = copy.deepcopy(records)
    out = model.predict(records)
    assert records == before
    assert [list(row) for row in out] == [["sample_id", "model_name", "label_mode", "predicted_labels"]] * 2
    assert [row["sample_id"] for row in out] == ["z", "a"]
    assert out[0] == {"sample_id": "z", "model_name": "my_h", "label_mode": "A",
                      "predicted_labels": ["O", "O", "PERIOD"]}
    assert all(len(row["predicted_labels"]) == n for row, n in zip(out, (3, 1), strict=False))


def test_predict_of_nothing_is_empty_and_an_unfitted_model_refuses() -> None:
    assert HiddenEventLM("A").fit(TINY).predict([]) == []
    fresh = HiddenEventLM("A")
    for call in (lambda: fresh.predict([]), lambda: fresh.decode(["a"]), lambda: fresh.logprob("a", ()),
                 lambda: fresh.predicted_tokens, lambda: fresh.training_summary):
        with pytest.raises(RuntimeError, match="not fitted"):
            call()


@pytest.mark.parametrize(("records", "fragment"), [
    ([obs("x", ["a"]), obs("x", ["b"])], "duplicate sample_id 'x'"),
    ([obs("", ["a"])], "sample_id"),
    ([{"tokens": ["a"]}], "sample_id"),
    ([obs("x", [])], "tokens"),
    ([{"sample_id": "x", "tokens": "a b"}], "tokens"),
    ([obs("x", ["a", " "])], "tokens"),
    ([obs("x", ["a", 3])], "tokens"),
    (["nope"], "mapping"),
])
def test_predict_rejects_bad_observations_with_a_one_line_error(tiny, records, fragment) -> None:
    with pytest.raises(ValueError, match=fragment) as error:
        tiny.predict(records)
    assert "\n" not in str(error.value)


# ------------------------------------------------------------------ validation errors


@pytest.mark.parametrize(("kwargs", "fragment"), [
    ({"label_mode": "D"}, "label_mode"), ({"label_mode": "a"}, "label_mode"), ({"label_mode": None}, "label_mode"),
    ({"order": 1}, "order"), ({"order": 2.5}, "order"), ({"order": "3"}, "order"), ({"order": True}, "order"),
    ({"discount": 0}, "discount"), ({"discount": 1}, "discount"), ({"discount": -0.1}, "discount"),
    ({"discount": float("nan")}, "discount"), ({"discount": "x"}, "discount"), ({"discount": True}, "discount"),
    ({"min_count": 0}, "min_count"), ({"min_count": 1.5}, "min_count"), ({"min_count": True}, "min_count"),
    ({"model_name": ""}, "model_name"), ({"model_name": "  "}, "model_name"), ({"model_name": 3}, "model_name"),
])
def test_bad_constructor_arguments_give_a_one_line_value_error(kwargs, fragment) -> None:
    arguments = {"label_mode": "A", **kwargs}
    with pytest.raises(ValueError, match=fragment) as error:
        HiddenEventLM(**arguments)
    assert "\n" not in str(error.value)


def _bad(**changes: Any) -> dict[str, Any]:
    return {**rec("bad", ["a", "b"], ["O", "PERIOD"]), **changes}


BAD_TRAINING = [
    ("not a mapping", ["nope"], "mapping"),
    ("no sample id", [{"tokens": ["a"]}], "sample_id"),
    ("blank sample id", [_bad(sample_id=" ")], "sample_id"),
    ("duplicate id", [_bad(), _bad()], "'bad'"),
    ("validation split", [_bad(split="validation")], "'bad'"),
    ("test split", [_bad(split="test")], "split"),
    ("mode mismatch", [_bad(label_mode="B")], "label_mode"),
    ("no mode", [{k: v for k, v in _bad().items() if k != "label_mode"}], "label_mode"),
    ("empty tokens", [_bad(tokens=[], punct_labels=[])], "tokens"),
    ("tokens not a list", [_bad(tokens="a b")], "tokens"),
    ("blank token", [_bad(tokens=["a", "  "])], "tokens"),
    ("non-string token", [_bad(tokens=["a", None])], "tokens"),
    ("labels missing", [{k: v for k, v in _bad().items() if k != "punct_labels"}], "punct_labels"),
    ("labels not a list", [_bad(punct_labels="OO")], "punct_labels"),
    ("labels too short", [_bad(punct_labels=["O"])], "punct_labels"),
    ("unknown label", [_bad(punct_labels=["O", "BANG"])], "BANG"),
    ("mode B label in mode A", [_bad(punct_labels=["O", "DASH"])], "DASH"),
    ("no records", [], "at least one"),
]


@pytest.mark.parametrize(("records", "fragment"), [pytest.param(r, f, id=i) for i, r, f in BAD_TRAINING])
def test_bad_training_records_give_a_one_line_value_error(records, fragment) -> None:
    with pytest.raises(ValueError, match=fragment) as error:
        HiddenEventLM("A").fit(records)
    assert "\n" not in str(error.value)


@pytest.mark.parametrize(("changes", "fragment"), [
    ({"case_labels": None}, "case_labels"),
    ({"case_labels": ["LOWER"]}, "case_labels"),
    ({"case_labels": ["LOWER", "TITLE"]}, "TITLE"),
    ({"punct_labels": ["O", "EXCLAMATION"]}, "EXCLAMATION"),
])
def test_bad_mode_c_records_name_the_sample(changes, fragment) -> None:
    good = rec("c", ["a", "b"], ["O", "PERIOD"], ["LOWER", "LOWER"], mode="C")
    row = {k: v for k, v in {**good, **changes}.items() if v is not None}
    with pytest.raises(ValueError, match=fragment) as error:
        HiddenEventLM("C").fit([row])
    assert "'c'" in str(error.value) and "\n" not in str(error.value)


def test_a_sample_id_with_a_newline_stays_on_one_line() -> None:
    with pytest.raises(ValueError) as error:
        HiddenEventLM("A").fit([_bad(sample_id="x\ny", split="test")])
    assert "\n" not in str(error.value)


def test_fit_leaves_its_records_untouched_and_a_failed_refit_keeps_the_old_model() -> None:
    model = HiddenEventLM("A").fit(TINY)
    before = copy.deepcopy(TINY)
    expected = model.logprob("a", ("b",))
    model.fit(TINY)
    assert TINY == before
    with pytest.raises(ValueError):
        model.fit([_bad(split="test")])
    assert model.logprob("a", ("b",)) == expected


def test_training_summary_counts_sequences_tokens_and_word_types() -> None:
    assert HiddenEventLM("A").fit(TINY).training_summary == {
        "num_sequences": 2, "num_tokens": 4, "vocabulary_size": 2}


# ------------------------------------------------------------------ save and load


def _fitted(mode: str) -> HiddenEventLM:
    if mode == "C":
        return HiddenEventLM("C", order=2, model_name="h_c").fit(C_CORPUS)
    return HiddenEventLM(mode, order=3, discount=0.6, min_count=2,
                         model_name="h_x").fit(_random_corpus(mode, seed=5))


@pytest.mark.parametrize("mode", ["A", "B", "C"])
def test_save_load_round_trip_gives_identical_predictions(tmp_path: Path, mode: str) -> None:
    model = _fitted(mode)
    path = tmp_path / "model.json"
    model.save_model(path)
    loaded = HiddenEventLM.load_model(path)
    inputs = [obs("1", ["a", "b", "c"]), obs("2", ["hello", "nasa", "it", "zzz"]), obs("3", ["d"])]
    assert loaded.predict(inputs) == model.predict(inputs)
    assert (loaded.label_mode, loaded.order, loaded.discount, loaded.min_count, loaded.model_name) == (
        model.label_mode, model.order, model.discount, model.min_count, model.model_name)
    assert loaded.training_summary == model.training_summary
    assert loaded.predicted_tokens == model.predicted_tokens
    assert loaded.logprob("a", ("b", "c")) == model.logprob("a", ("b", "c"))


def test_refitting_the_same_records_writes_the_same_bytes(tmp_path: Path) -> None:
    _fitted("B").save_model(tmp_path / "one.json")
    _fitted("B").save_model(tmp_path / "two.json")
    assert (tmp_path / "one.json").read_bytes() == (tmp_path / "two.json").read_bytes()


def test_the_artifact_is_compact_json_utf8_with_one_lf_at_the_end(tmp_path: Path) -> None:
    _fitted("B").save_model(tmp_path / "m.json")
    raw = (tmp_path / "m.json").read_bytes()
    text = raw.decode("utf-8")
    assert raw.endswith(b"}\n") and raw.count(b"\n") == 1 and b"\r" not in raw
    assert text == json.dumps(json.loads(text), ensure_ascii=False, separators=(",", ":")) + "\n"


def test_the_artifact_names_its_family_and_version(tmp_path: Path) -> None:
    _fitted("A").save_model(tmp_path / "m.json")
    payload = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert (payload["format_version"], payload["model_family"]) == (1, "hidden_event_lm")
    assert payload["config"] == {"label_mode": "A", "order": 3, "discount": 0.6, "min_count": 2,
                                 "model_name": "h_x"}


def test_save_refuses_to_overwrite_and_an_unfitted_model_cannot_be_saved(tmp_path: Path) -> None:
    target = tmp_path / "keep.json"
    target.write_text("keep me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _fitted("A").save_model(target)
    assert target.read_text(encoding="utf-8") == "keep me"
    with pytest.raises(RuntimeError, match="not fitted"):
        HiddenEventLM("A").save_model(tmp_path / "never.json")
    assert not (tmp_path / "never.json").exists()


class _FullDiskHandle:
    """A real file handle whose write puts part of the text on disk and then fails like a full disk."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __enter__(self) -> Self:
        self._inner.__enter__()
        return self

    def __exit__(self, *exc: object) -> Any:
        return self._inner.__exit__(*exc)

    def write(self, text: str) -> int:
        self._inner.write(text[:20])
        raise OSError(28, "No space left on device")


def test_a_failed_save_removes_the_partial_file_so_a_retry_succeeds(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model, path = _fitted("A"), tmp_path / "model.json"

    def full_disk_open(file: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        return _FullDiskHandle(open(file, mode, *args, **kwargs))  # the one shared writer calls open()

    with monkeypatch.context() as patch:
        patch.setattr(file_io, "open", full_disk_open, raising=False)
        with pytest.raises(OSError, match="No space left"):
            model.save_model(path)
    assert not path.exists()
    model.save_model(path)
    inputs = [obs("1", ["a", "b", "c"])]
    assert HiddenEventLM.load_model(path).predict(inputs) == model.predict(inputs)


def test_a_missing_artifact_is_a_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        HiddenEventLM.load_model(tmp_path / "absent.json")


def _saved_payload(tmp_path: Path, mode: str = "A") -> dict[str, Any]:
    _fitted(mode).save_model(tmp_path / "good.json")
    return json.loads((tmp_path / "good.json").read_text(encoding="utf-8"))


def _break(payload: dict[str, Any], how: str) -> None:
    ngrams = payload["ngrams"]
    edits = {
        "version": lambda: payload.update(format_version=2),
        "family": lambda: payload.update(model_family="hmm"),
        "no config": lambda: payload.pop("config"),
        "bad config": lambda: payload["config"].update(order=1),
        "extra config key": lambda: payload["config"].update(extra=1),
        "no words": lambda: payload.pop("words"),
        "duplicate words": lambda: payload.update(words=payload["words"] + payload["words"][:1]),
        "word not a string": lambda: payload["words"].append(7),
        "ngrams wrong length": lambda: payload.update(ngrams=ngrams[:-1]),
        "keys and counts differ": lambda: ngrams[0]["counts"].pop(),
        "zero count": lambda: ngrams[1]["counts"].__setitem__(0, 0),
        "float count": lambda: ngrams[1]["counts"].__setitem__(0, 1.5),
        "bool count": lambda: ngrams[1]["counts"].__setitem__(0, True),
        "key out of range": lambda: ngrams[1]["keys"].__setitem__(0, 10**12),
        "negative key": lambda: ngrams[1]["keys"].__setitem__(0, -1),
        "duplicate key": lambda: ngrams[1]["keys"].__setitem__(0, ngrams[1]["keys"][1]),
        "start symbol predicted": lambda: ngrams[0]["keys"].__setitem__(0, 0),
        "no unigrams": lambda: (ngrams[0]["keys"].clear(), ngrams[0]["counts"].clear()),
        "bad summary": lambda: payload.update(training={"num_sequences": -1}),
        "truecaser in mode A": lambda: payload.update(truecaser=[["a", 1, 0, 0]]),
        "huge unigram count": lambda: ngrams[0]["counts"].__setitem__(0, 10**400),
        "huge bigram count": lambda: ngrams[1]["counts"].__setitem__(0, 10**400),
    }
    edits[how]()


BREAKS = ["version", "family", "no config", "bad config", "extra config key", "no words", "duplicate words",
          "word not a string", "ngrams wrong length", "keys and counts differ", "zero count", "float count",
          "bool count", "key out of range", "negative key", "duplicate key", "start symbol predicted",
          "no unigrams", "bad summary", "truecaser in mode A", "huge unigram count", "huge bigram count"]


@pytest.mark.parametrize("how", BREAKS)
def test_a_tampered_artifact_is_a_one_line_value_error(tmp_path: Path, how: str) -> None:
    payload = _saved_payload(tmp_path)
    _break(payload, how)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=ARTIFACT_ERROR) as error:
        HiddenEventLM.load_model(bad)
    assert "\n" not in str(error.value)


@pytest.mark.parametrize(("count", "loads"), [(2**53, True), (2**53 + 1, False)])
def test_a_count_of_two_to_the_53_loads_and_one_more_is_refused(tmp_path: Path, count: int, loads: bool) -> None:
    payload = _saved_payload(tmp_path)
    payload["ngrams"][1]["counts"][0] = count
    path = tmp_path / "edge.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    if loads:
        assert HiddenEventLM.load_model(path).label_mode == "A"
    else:
        with pytest.raises(ValueError, match=ARTIFACT_ERROR):
            HiddenEventLM.load_model(path)


@pytest.mark.parametrize("how", ["missing truecaser", "bad truecaser row", "duplicate truecaser word",
                                 "negative truecaser count", "all-zero truecaser row"])
def test_a_tampered_mode_c_truecaser_is_a_value_error(tmp_path: Path, how: str) -> None:
    payload = _saved_payload(tmp_path, "C")
    table = payload["truecaser"]
    edits = {
        "missing truecaser": lambda: payload.update(truecaser=None),
        "bad truecaser row": lambda: table.__setitem__(0, ["x", 1]),
        "duplicate truecaser word": lambda: table.append(list(table[0])),
        "negative truecaser count": lambda: table[0].__setitem__(1, -1),
        "all-zero truecaser row": lambda: table.__setitem__(0, [table[0][0], 0, 0, 0]),
    }
    edits[how]()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=ARTIFACT_ERROR):
        HiddenEventLM.load_model(bad)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", r"^bad\.json: not valid JSON$"),
        (b"{not json", r"^bad\.json: not valid JSON$"),
        (b"[" * 100_000, r"^bad\.json: not valid JSON$"),
        (b"\xff\xfe\x00", r"^bad\.json: not valid UTF-8$"),
        (b"[1, 2]", rf"^bad\.json: {ARTIFACT_ERROR}: "),
        (b'{"format_version": 1}', rf"^bad\.json: {ARTIFACT_ERROR}: "),
    ],
)
def test_corrupt_artifact_bytes_are_one_named_line_without_python_text(
        tmp_path: Path, content: bytes, message: str) -> None:
    bad = tmp_path / "bad.json"
    bad.write_bytes(content)
    with pytest.raises(ValueError, match=message) as error:
        HiddenEventLM.load_model(bad)
    assert "\n" not in str(error.value) and "codec" not in str(error.value) and "Expecting" not in str(error.value)


@pytest.mark.parametrize("content", [b"{not json", b"\xff\xfe\x00", b"[1, 2]", b"[" * 100_000])
def test_the_artifact_error_starts_with_the_file_name_and_shows_no_folder(tmp_path: Path, content: bytes) -> None:
    folder = tmp_path / "secretfolder"
    folder.mkdir()
    bad = folder / "bad.json"
    bad.write_bytes(content)
    with pytest.raises(ValueError) as error:
        HiddenEventLM.load_model(bad)
    assert str(error.value).startswith("bad.json: ")  # the exact wording per kind is pinned above
    assert "secretfolder" not in str(error.value)
