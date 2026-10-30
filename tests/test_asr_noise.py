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
