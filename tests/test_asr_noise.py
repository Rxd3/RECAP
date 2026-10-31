"""Tests for punct_restore.data.asr_noise: the edit model, the edit plan and the position remapping."""

from __future__ import annotations

import json
from typing import Any

import pytest

from punct_restore.data import asr_noise as noise

WORDS = ["the", "cat", "is", "on", "a", "mat", "their", "dog", "and", "you're",
         "too", "fast", "then", "it", "ran", "to", "its", "home", "today", "now"]
SAMPLES = 30


def tokens_of(index: int) -> list[str]:
    """The 20 WORDS, rotated by the index: every candidate kind occurs in every utterance."""
    shift = index % len(WORDS)
    return WORDS[shift:] + WORDS[:shift]


def rec(tokens: list[str], punct: list[str], case: list[str] | None = None) -> dict[str, Any]:
    case = case or ["LOWER"] * len(tokens)
    return {"tokens": tokens, "punct_labels": punct, "case_labels": case,
            "metadata": {"offsets": [[k, k + 1] for k in range(len(tokens))]}}


def test_deleting_a_function_word_moves_its_comma_to_a_previous_o():
    out = noise.apply_edits(rec(["so", "the", "dog", "ran"], ["O", "COMMA", "O", "PERIOD"],
                                ["FIRST_UPPER", "LOWER", "ALL_UPPER", "LOWER"]),
                            noise.Edits(deleted=frozenset({1})))
    assert out["tokens"] == ["so", "dog", "ran"]
    assert out["punct_labels"] == ["COMMA", "O", "PERIOD"]
    assert out["case_labels"] == ["FIRST_UPPER", "ALL_UPPER", "LOWER"]
    assert out["metadata"]["offsets"] == [[0, 1], [2, 3], [3, 4]]


@pytest.mark.parametrize(("tokens", "punct", "deleted", "want_tokens", "want_punct"), [
    (["go", "now", "the", "end"], ["O", "PERIOD", "COMMA", "PERIOD"], {2}, ["go", "now", "end"],
     ["O", "PERIOD", "PERIOD"]),  # the previous token already has a PERIOD: the COMMA is dropped
    (["the", "dog"], ["COMMA", "PERIOD"], {0}, ["dog"], ["PERIOD"]),  # no previous token: dropped
    (["so", "the", "a", "dog"], ["O", "COMMA", "PERIOD", "PERIOD"], {1, 2}, ["so", "dog"],
     ["COMMA", "PERIOD"]),  # the first deleted COMMA lands on "so", so the second label is dropped
])
def test_a_deleted_label_that_has_no_previous_o_is_dropped(tokens, punct, deleted, want_tokens, want_punct):
    out = noise.apply_edits(rec(tokens, punct), noise.Edits(deleted=frozenset(deleted)))
    assert out["tokens"] == want_tokens and out["punct_labels"] == want_punct


def test_a_swap_moves_tokens_with_their_labels():
    out = noise.apply_edits(rec(["a", "b", "c", "d"], ["O", "COMMA", "O", "PERIOD"],
                                ["FIRST_UPPER", "ALL_UPPER", "LOWER", "LOWER"]),
                            noise.Edits(swapped=frozenset({0})))
    assert out["tokens"] == ["b", "a", "c", "d"]
    assert out["punct_labels"] == ["COMMA", "O", "O", "PERIOD"]
    assert out["case_labels"] == ["ALL_UPPER", "FIRST_UPPER", "LOWER", "LOWER"]
    assert out["metadata"]["offsets"] == [[1, 2], [0, 1], [2, 3], [3, 4]]


def test_a_substitution_keeps_the_labels_and_does_not_mutate_the_input():
    source = rec(["their", "dog", "ran"], ["COMMA", "O", "PERIOD"], ["FIRST_UPPER", "LOWER", "LOWER"])
    snapshot = json.dumps(source)
    out = noise.apply_edits(source, noise.Edits(replaced={0: "there"}))
    assert out["tokens"] == ["there", "dog", "ran"]
    assert out["punct_labels"] == source["punct_labels"] and out["case_labels"] == source["case_labels"]
    assert json.dumps(source) == snapshot


def test_a_record_without_offsets_is_edited_without_them():
    source = {"tokens": ["a", "b"], "punct_labels": ["O", "PERIOD"], "case_labels": ["LOWER", "LOWER"]}
    out = noise.apply_edits(source, noise.Edits(swapped=frozenset({0})))
    assert out["tokens"] == ["b", "a"] and "metadata" not in out


def test_a_deletion_after_a_swapped_pair_uses_the_last_output_token():
    out = noise.apply_edits(rec(["a", "b", "the", "d", "e"], ["O", "O", "COMMA", "O", "PERIOD"]),
                            noise.Edits(swapped=frozenset({0}), deleted=frozenset({2})))
    assert out["tokens"] == ["b", "a", "d", "e"] and out["punct_labels"] == ["O", "COMMA", "O", "PERIOD"]


def test_the_last_remaining_token_is_never_deleted():
    with pytest.raises(ValueError, match="not enough editable tokens for WER 0.5"):
        noise.build_plan({k: ["the"] for k in "abcd"}, 0.5, 1)
    plan = noise.build_plan({"s": ["the", "the", "the"]}, 0.5, 1)  # budget round(1.5) = 2
    assert len(plan["s"].deleted) == 2
    with pytest.raises(ValueError, match="for WER 0.25"):  # swapping equal words is not a candidate
        noise.build_plan({"s": ["no"] * 8}, 0.25, 1)


def test_plan_invariants_hold_for_every_candidate_kind():
    tokens = {f"s{i}": tokens_of(i) for i in range(SAMPLES)}
    plan = noise.build_plan(tokens, 0.15, 3)
    groups = {w: g for g in noise.HOMOPHONE_GROUPS for w in g}
    for sid, edits in plan.items():
        words, n = tokens[sid], len(tokens[sid])
        touched = [*edits.deleted, *edits.replaced, *(j for i in edits.swapped for j in (i, i + 1))]
        assert len(touched) == len(set(touched)) and len(edits.deleted) < n
        assert all(words[i] in noise.FUNCTION_WORDS for i in edits.deleted)
        assert all(i + 1 < n - 1 and words[i] != words[i + 1] for i in edits.swapped)
        assert all(new != words[i] and new in groups[words[i]] for i, new in edits.replaced.items())


def kind_counts(plan) -> tuple[int, int, int]:
    counts = noise.edit_counts(plan)
    return counts["deletion"], counts["substitution"], counts["swap"]


def hosts(deletions: int, substitutions: int, swaps: int) -> dict[str, list[str]]:
    """Utterances that each offer exactly one candidate, of one kind, so the outcome ignores the shuffle."""
    return {**{f"d{i}": ["the", "cat"] for i in range(deletions)},
            **{f"r{i}": ["their", "cat"] for i in range(substitutions)},
            **{f"w{i}": ["cat", "dog", "fox"] for i in range(swaps)}}


@pytest.mark.parametrize(("budget", "expected"), [
    (1, (1, 0, 0)),  # deletion goes first
    (2, (1, 1, 0)),  # then substitution
    (3, (2, 1, 0)),  # a swap (cost 2) does not fit in 1: deletion takes the turn again
    (4, (1, 1, 1)),  # one full round
    (7, (3, 2, 1)),  # round, then deletion + substitution, then the swap no longer fits
])
def test_round_robin_hand_case_has_exact_counts(budget, expected):
    tokens = hosts(3, 3, 3)  # 21 tokens
    assert noise.edit_budget(tokens, budget / 21) == budget
    plan = noise.build_plan(tokens, budget / 21, 2026)
    assert kind_counts(plan) == expected
    assert sum(c * noise.COST[kind] for c, kind in zip(expected, ("delete", "replace", "swap"), strict=True)) == budget


@pytest.mark.parametrize(("budget", "expected"), [(14, (4, 2, 4)), (15, (5, 2, 4))])
def test_when_substitutions_run_out_deletions_and_swaps_alternate(budget, expected):
    tokens = hosts(6, 2, 6)  # 34 tokens, only two substitution candidates
    assert noise.edit_budget(tokens, budget / 34) == budget
    assert kind_counts(noise.build_plan(tokens, budget / 34, 2026)) == expected


@pytest.mark.parametrize("budget", [56, 57, 58, 60])
def test_counts_alternate_in_rounds_on_a_plentiful_fixture(budget):
    tokens = {f"s{i}": tokens_of(i) for i in range(SAMPLES)}  # 600 tokens, candidates of every kind
    wer = budget / 600
    assert noise.edit_budget(tokens, wer) == budget
    rounds, rest = divmod(budget, 4)  # one round costs 1 + 1 + 2
    extra = {0: (0, 0), 1: (1, 0), 2: (1, 1)}[rest]
    plan = noise.build_plan(tokens, wer, 2026)
    assert kind_counts(plan) == (rounds + extra[0], rounds + extra[1], rounds)
    assert plan == noise.build_plan(tokens, wer, 2026)  # deterministic under the seed


def test_the_shuffle_inside_a_type_depends_on_the_seed():
    tokens = {f"s{i}": tokens_of(i) for i in range(SAMPLES)}
    one, other = noise.build_plan(tokens, 0.1, 1), noise.build_plan(tokens, 0.1, 2)
    assert kind_counts(one) == kind_counts(other) and one != other


@pytest.mark.parametrize(("excluded", "expected"), [([0, 1, 2, 3], [0, 1, 2]), ([2, 4], [2, 3]), ([3], [1]), ([1], [])])
def test_masks_are_remapped_through_the_edits(excluded, expected):
    edits = noise.Edits(deleted=frozenset({1}), swapped=frozenset({2}))  # output order is [0, 3, 2, 4]
    assert noise.align(5, edits) == [0, 3, 2, 4]
    assert noise.remap_positions(excluded, noise.align(5, edits)) == expected


def test_v1_plan_matches_golden_sha256():
    import hashlib
    tokens = {f"s{i}": tokens_of(i) for i in range(SAMPLES)}
    plan = noise.build_plan(tokens, 0.10, 2026)
    serialized = {sid: {"deleted": sorted(e.deleted), "swapped": sorted(e.swapped),
                        "replaced": dict(sorted(e.replaced.items()))}
                  for sid, e in sorted(plan.items())}
    digest = hashlib.sha256(json.dumps(serialized, sort_keys=True).encode()).hexdigest()
    assert digest == "eb1df4c89d28453a6ba33e151d49c61f8a3acd54eb96d6be30475a5c9d06a255"
    assert noise.build_plan(tokens, 0.10, 2026, edit_model="v1") == plan


def test_insertion_sets_punct_o_case_lower_empty_next_offset_and_masks_mode_c():
    source = {"tokens": ["hello", "world", "now"], "punct_labels": ["COMMA", "O", "PERIOD"],
              "case_labels": ["FIRST_UPPER", "ALL_UPPER", "LOWER"],
              "metadata": {"offsets": [[0, 5], [6, 11], [12, 15]]}}
    edits = noise.Edits(inserted={1: "um"})
    out = noise.apply_edits(source, edits)
    assert out["tokens"] == ["hello", "um", "world", "now"]
    assert out["punct_labels"] == ["COMMA", "O", "O", "PERIOD"]
    assert out["case_labels"] == ["FIRST_UPPER", "LOWER", "ALL_UPPER", "LOWER"]
    assert out["metadata"]["offsets"] == [[0, 5], [6, 6], [6, 11], [12, 15]]
    order = noise.align(3, edits)
    assert noise.remap_positions([2], order) == [1, 3]


def test_docstring_documents_all_three_insertion_rules():
    doc = noise.__doc__ or ""
    assert "O" in doc and "LOWER" in doc
    assert "mask" in doc and "0" in doc
    assert "offsets" in doc and "empty" in doc


def test_train_vocabulary_enforces_min_count_five():
    corpus = [["cat", "cap", "cab", "rare"] for _ in range(4)] + [["cat", "cap"]]
    vocab = noise.train_vocabulary(corpus)
    assert vocab == frozenset({"cat", "cap"})


def test_v2_confusion_substitution_uses_only_train_vocab_at_edit_distance_one_same_first_letter():
    # "cat" -> "cap" (sub, same first letter), "cats" (insert char, same first letter), "ca" (del char, same first).
    # Rejected: "bat" (different first letter), "chats" (edit distance 2), "cat" (same word).
    vocab = frozenset({"cap", "cats", "ca", "bat", "chats", "cat"})
    tokens = {f"s{i}": ["cat", "zzzz"] for i in range(10)}
    plan = noise.build_plan(tokens, 0.25, 2026, edit_model="v2", vocab=vocab)
    replaced = [word for edits in plan.values() for word in edits.replaced.values()]
    assert replaced and set(replaced) <= {"cap", "cats", "ca"}


def test_v2_keeps_homophone_substitution_as_second_source():
    tokens = {f"s{i}": ["their", "zzzz"] for i in range(10)}
    plan = noise.build_plan(tokens, 0.25, 2026, edit_model="v2", vocab=frozenset())
    replaced = [word for edits in plan.values() for word in edits.replaced.values()]
    assert replaced and set(replaced) <= {"there", "they're"}


def test_v2_deletes_short_words_and_function_words_never_last_token():
    tokens = {"a": ["cat", "elephant"], "b": ["would", "elephant"]}
    plan = noise.build_plan(tokens, 0.5, 2026, edit_model="v2", vocab=frozenset())
    assert plan["a"].deleted == frozenset({0}) and plan["b"].deleted == frozenset({0})
    # Single-token utterances cannot be deleted even though delete has higher priority than insert
    single = {k: ["cat"] for k in "abcd"}
    plan_single = noise.build_plan(single, 0.5, 2026, edit_model="v2", vocab=frozenset())
    assert all(not edits.deleted for edits in plan_single.values())


def test_v2_caps_swaps_at_ten_percent_of_budget_even_when_other_pools_run_dry():
    # Utterances of long identical-prefix-free words with no deletions/replacements and only 1 insertion token
    tokens = {f"s{i}": ["elephant", "giraffe", "hippopotamus", "crocodile"] for i in range(10)}  # 40 tokens, budget 20
    # Even if we only allow swaps by exhausting or limiting other types, swaps must never exceed 10% of budget.
    plan = noise.build_plan(tokens, 0.5, 2026, edit_model="v2", vocab=frozenset())
    counts = noise.edit_counts(plan, edit_model="v2")
    assert counts["swap"] * noise.COST["swap"] <= 0.10 * noise.edit_budget(tokens, 0.5)


def test_v2_achieves_target_mix_and_exact_budget_on_plentiful_fixture():
    vocab = frozenset({"cap", "map", "hat", "ran", "tan", "fan", "now", "new"})
    tokens = {f"s{i}": tokens_of(i) for i in range(SAMPLES)}  # 600 tokens
    plan = noise.build_plan(tokens, 0.10, 2026, edit_model="v2", vocab=vocab)  # budget 60 = 3 * 20
    counts = noise.edit_counts(plan, edit_model="v2")
    assert counts == {"deletion": 18, "insertion": 9, "substitution": 27, "swap": 3}
    cost = (counts["deletion"] + counts["insertion"] + counts["substitution"]
            + counts["swap"] * noise.COST["swap"])
    assert cost == 60
    assert plan == noise.build_plan(tokens, 0.10, 2026, edit_model="v2", vocab=vocab)
    assert plan != noise.build_plan(tokens, 0.10, 2027, edit_model="v2", vocab=vocab)


def _char_edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def test_v2_plan_invariants_hold_for_every_candidate_kind():
    vocab = frozenset({"cap", "map", "hat", "ran", "tan", "fan", "now", "new"})
    tokens = {f"s{i}": tokens_of(i) for i in range(SAMPLES)}
    plan = noise.build_plan(tokens, 0.15, 2026, edit_model="v2", vocab=vocab)
    for sid, edits in plan.items():
        words, n = tokens[sid], len(tokens[sid])
        touched = [*edits.deleted, *edits.replaced, *edits.inserted,
                   *(j for i in edits.swapped for j in (i, i + 1))]
        assert len(touched) == len(set(touched)) and len(edits.deleted) < n
        assert all(words[i] in noise.FUNCTION_WORDS or len(words[i]) <= 3 for i in edits.deleted)
        assert all(w in noise.INSERT_WORDS for w in edits.inserted.values())
        assert all(i + 1 < n - 1 and words[i] != words[i + 1] for i in edits.swapped)
        for i, new in edits.replaced.items():
            old = words[i]
            assert new != old
            in_homophones = old in noise.HOMOPHONES and new in noise.HOMOPHONES[old]
            in_confusion = new in vocab and new[0] == old[0] and _char_edit_distance(old, new) == 1
            assert in_homophones or in_confusion


def test_unknown_edit_model_raises():
    with pytest.raises(ValueError, match="unknown edit model"):
        noise.build_plan({"s": ["the", "cat"]}, 0.1, 2026, edit_model="v3")


