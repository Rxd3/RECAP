"""Scored-position views and per-mode report blocks, owned by Member 5.

The first block of each mode is the primary one. Utterance-final punctuation is nearly always
PERIOD or QUESTION and the first word is nearly always capitalized, so the non_final and
non_initial views show how much of a score comes from those shortcuts. Casing masks apply to
the joint and casing blocks only; punctuation is always scored.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from .inputs import NO_PUNCT, Aligned, Vocab
from .metrics import split_joint


@dataclass(frozen=True)
class BlockSpec:
    name: str
    labels: tuple[str, ...]
    macro_over: tuple[str, ...]
    positions: Callable[[Aligned], Iterable[int]]
    project: Callable[[str], str] | None = None


def all_positions(record: Aligned) -> Iterable[int]:
    return range(len(record.gold))


def non_final(record: Aligned) -> Iterable[int]:
    return range(len(record.gold) - 1)


def unmasked(record: Aligned) -> Iterable[int]:
    return (i for i in range(len(record.gold)) if i not in record.masked)


def unmasked_non_initial(record: Aligned) -> Iterable[int]:
    return (i for i in range(1, len(record.gold)) if i not in record.masked)


def punctuation_of(label: str) -> str:
    return split_joint(label)[0]


def casing_of(label: str) -> str:
    return split_joint(label)[1]


def blocks_for(mode: str, vocab: Vocab) -> tuple[BlockSpec, ...]:
    """Report blocks for a label mode, primary first (names and order are part of the schema)."""
    if mode in ("A", "B"):
        labels = vocab.labels(mode)
        non_o = tuple(label for label in labels if label != NO_PUNCT)
        return (BlockSpec("punctuation/all", labels, non_o, all_positions),
                BlockSpec("punctuation/non_final", labels, non_o, non_final))
    if mode == "C":
        non_o = tuple(label for label in vocab.A if label != NO_PUNCT)
        casing = vocab.casing
        return (BlockSpec("joint/unmasked", vocab.C, vocab.C, unmasked),
                BlockSpec("punctuation/all", vocab.A, non_o, all_positions, punctuation_of),
                BlockSpec("punctuation/non_final", vocab.A, non_o, non_final, punctuation_of),
                BlockSpec("casing/unmasked", casing, casing, unmasked, casing_of),
                BlockSpec("casing/unmasked_non_initial", casing, casing, unmasked_non_initial,
                          casing_of),
                BlockSpec("casing/all", casing, casing, all_positions, casing_of))  # secondary
    raise ValueError(f"unknown label mode {mode!r}")


def block_pairs(records: Sequence[Aligned], spec: BlockSpec) -> list[tuple[str, str]]:
    """(gold, predicted) pairs at the block's positions, projected to the block's labels."""
    project = spec.project or (lambda label: label)
    return [(project(record.gold[i]), project(record.pred[i]))
            for record in records for i in spec.positions(record)]
