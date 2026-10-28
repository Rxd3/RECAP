"""Tests for the scored-position views and per-mode report blocks (spec section 5)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pytest

from punct_restore.evaluation.inputs import Aligned, Vocab
from punct_restore.evaluation.metrics import score_block
from punct_restore.evaluation.views import (
    BlockSpec,
    all_positions,
    block_pairs,
    blocks_for,
    casing_of,
    non_final,
    punctuation_of,
    unmasked,
    unmasked_non_initial,
)

NON_O = ("COMMA", "PERIOD", "QUESTION")
CASING = ("LOWER", "FIRST_UPPER", "ALL_UPPER")
EXTENDED_NON_O = NON_O + ("EXCLAMATION", "COLON", "SEMICOLON", "DASH")
MODE_C_NAMES = ["joint/unmasked", "punctuation/all", "punctuation/non_final",
                "casing/unmasked", "casing/unmasked_non_initial", "casing/all"]


def aligned(gold: Sequence[str], pred: Sequence[str], masked: Iterable[int] = (), sample_id: str = "x") -> Aligned:
    return Aligned(sample_id, None, tuple(gold), tuple(pred), frozenset(masked))


def spec_named(mode: str, vocab: Vocab, name: str) -> BlockSpec:
    return next(spec for spec in blocks_for(mode, vocab) if spec.name == name)


def tokens_of(count: int, masked: Iterable[int] = ()) -> Aligned:
    """A record with `count` positions; the labels do not matter for position views."""
    return aligned(["O"] * count, ["O"] * count, masked)


def masked_mode_c_record() -> Aligned:
    """Position 1 is masked and wrong in both punctuation and casing."""
    return aligned(("O|FIRST_UPPER", "COMMA|LOWER", "PERIOD|LOWER"),
                   ("O|FIRST_UPPER", "PERIOD|ALL_UPPER", "PERIOD|LOWER"), masked={1})


@pytest.mark.parametrize(("mode", "names"), [
    ("A", ["punctuation/all", "punctuation/non_final"]),
    ("B", ["punctuation/all", "punctuation/non_final"]),
    ("C", MODE_C_NAMES),
])
def test_block_names_per_mode(vocab: Vocab, mode: str, names: list[str]) -> None:
    assert [spec.name for spec in blocks_for(mode, vocab)] == names


def _expected_blocks(mode: str, vocab: Vocab) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Block name -> (labels, macro_over)."""
    if mode == "A":
        return {name: (vocab.A, NON_O) for name in ("punctuation/all", "punctuation/non_final")}
    if mode == "B":
        return {name: (vocab.B, EXTENDED_NON_O) for name in ("punctuation/all", "punctuation/non_final")}
    return {"joint/unmasked": (vocab.C, vocab.C),
            "punctuation/all": (vocab.A, NON_O),
            "punctuation/non_final": (vocab.A, NON_O),
            "casing/unmasked": (CASING, CASING),
            "casing/unmasked_non_initial": (CASING, CASING),
            "casing/all": (CASING, CASING)}


@pytest.mark.parametrize("mode", ["A", "B", "C"])
def test_block_labels_and_macro_over(vocab: Vocab, mode: str) -> None:
    specs = blocks_for(mode, vocab)
    expected = _expected_blocks(mode, vocab)
    assert [spec.name for spec in specs] == list(expected)
    for spec in specs:
        assert (spec.labels, spec.macro_over) == expected[spec.name], spec.name
    if mode != "C":
        assert all(spec.project is None for spec in specs)


def test_joint_block_covers_all_twelve_labels(vocab: Vocab) -> None:
    spec = spec_named("C", vocab, "joint/unmasked")
    assert len(spec.labels) == 12
    assert spec.macro_over == spec.labels


def test_unknown_mode_rejected(vocab: Vocab) -> None:
    with pytest.raises(ValueError, match="'D'"):
        blocks_for("D", vocab)


@pytest.mark.parametrize(("length", "expected"), [(3, [0, 1]), (2, [0]), (1, [])])
def test_non_final_drops_last_index_only(length: int, expected: list[int]) -> None:
    assert list(non_final(tokens_of(length))) == expected


def test_all_positions_ignores_masks() -> None:
    assert list(all_positions(tokens_of(3, masked={1}))) == [0, 1, 2]


def test_masks_affect_joint_and_casing_not_punctuation(vocab: Vocab) -> None:
    record = masked_mode_c_record()
    expected = {
        "joint/unmasked": [("O|FIRST_UPPER", "O|FIRST_UPPER"), ("PERIOD|LOWER", "PERIOD|LOWER")],
        "punctuation/all": [("O", "O"), ("COMMA", "PERIOD"), ("PERIOD", "PERIOD")],
        "punctuation/non_final": [("O", "O"), ("COMMA", "PERIOD")],
        "casing/unmasked": [("FIRST_UPPER", "FIRST_UPPER"), ("LOWER", "LOWER")],
        "casing/unmasked_non_initial": [("LOWER", "LOWER")],
        "casing/all": [("FIRST_UPPER", "FIRST_UPPER"), ("LOWER", "ALL_UPPER"), ("LOWER", "LOWER")],
    }
    assert list(expected) == MODE_C_NAMES
    for spec in blocks_for("C", vocab):
        assert block_pairs([record], spec) == expected[spec.name], spec.name


def test_casing_all_counts_masked_position(vocab: Vocab) -> None:
    record = masked_mode_c_record()
    everything, kept = spec_named("C", vocab, "casing/all"), spec_named("C", vocab, "casing/unmasked")
    all_pairs, kept_pairs = block_pairs([record], everything), block_pairs([record], kept)
    assert ("LOWER", "ALL_UPPER") in all_pairs
    assert ("LOWER", "ALL_UPPER") not in kept_pairs
    assert all_pairs[0] == ("FIRST_UPPER", "FIRST_UPPER")
    assert score_block(all_pairs, everything.labels, everything.macro_over)["scored_positions"] == 3
    assert score_block(kept_pairs, kept.labels, kept.macro_over)["scored_positions"] == 2


def test_non_initial_drops_zero_and_masks() -> None:
    record = tokens_of(4, masked={2})
    assert list(unmasked_non_initial(record)) == [1, 3]
    assert list(unmasked(record)) == [0, 1, 3]
    assert list(unmasked_non_initial(tokens_of(4, masked={0}))) == [1, 2, 3]
    assert list(unmasked_non_initial(tokens_of(1))) == []


def test_projections() -> None:
    assert punctuation_of("PERIOD|FIRST_UPPER") == "PERIOD"
    assert casing_of("PERIOD|FIRST_UPPER") == "FIRST_UPPER"


def test_mode_a_pairs_keep_record_then_position_order(vocab: Vocab) -> None:
    first = aligned(("O", "PERIOD"), ("O", "QUESTION"), sample_id="r1")
    second = aligned(("COMMA", "O", "PERIOD"), ("O", "O", "PERIOD"), sample_id="r2")
    records = [first, second]
    assert block_pairs(records, spec_named("A", vocab, "punctuation/all")) == [
        ("O", "O"), ("PERIOD", "QUESTION"), ("COMMA", "O"), ("O", "O"), ("PERIOD", "PERIOD")]
    assert block_pairs(records, spec_named("A", vocab, "punctuation/non_final")) == [
        ("O", "O"), ("COMMA", "O"), ("O", "O")]


def test_joint_micro_equals_accuracy(vocab: Vocab) -> None:
    first = masked_mode_c_record()
    second = aligned(("O|FIRST_UPPER", "QUESTION|LOWER", "O|LOWER"),
                     ("O|FIRST_UPPER", "PERIOD|LOWER", "O|ALL_UPPER"), sample_id="x2")
    spec = spec_named("C", vocab, "joint/unmasked")
    block = score_block(block_pairs([first, second], spec), spec.labels, spec.macro_over)
    assert block["scored_positions"] == 5
    assert block["accuracy"] == pytest.approx(3 / 5)
    for name in ("precision", "recall", "f1"):
        assert block["micro"][name] == pytest.approx(block["accuracy"])
