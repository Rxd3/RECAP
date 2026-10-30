"""ASR-like word errors for gold records, with every label kept consistent (robustness study).

Used by scripts/simulate_asr_noise.py, which does the file reading and writing.

Edit model. One plan is drawn from the tokens of a set of samples and applied to each record. Candidates:
deletion of a function word (cost 1, never the last remaining token), substitution by a homophone of the same
group (cost 1), swap of two adjacent distinct non-final tokens (cost 2). The budget is round(wer * tokens).
The candidates of each type are shuffled separately with random.Random(seed), then taken round-robin by type in
the fixed order deletion, substitution, swap: each turn gives the next acceptable candidate of that type, and a
type with no candidates left is skipped. A candidate is skipped when it touches an edited token, overspends what
is left of the budget, or would delete an utterance's last token. The run stops when the budget is exactly spent
and fails if every candidate is used up first. So the mix is as balanced as the data allows: deletions and
substitutions take their turn first, and swaps (which alone would be most of the candidates) fill the rest.

Labels. A deleted token's case label is dropped; its punctuation label moves to the previous surviving token if
that carries O, else it is dropped. A substitution keeps all labels; a swap moves both tokens with their labels.
metadata.offsets (one span per token) is edited like the tokens. Other fields are left unchanged.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

NO_PUNCT = "O"
COST = {"delete": 1, "replace": 1, "swap": 2}
TURN_ORDER = tuple(COST)  # the round-robin order of the candidate types
FUNCTION_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "have", "has", "had",
    "be", "been", "will", "would", "can", "could", "to", "of", "in", "on", "at", "for", "and",
    "or", "that"})
HOMOPHONE_GROUPS = (("their", "there", "they're"), ("your", "you're"), ("its", "it's"),
                    ("to", "too", "two"), ("then", "than"))
HOMOPHONES = {word: group for group in HOMOPHONE_GROUPS for word in group}
Candidate = tuple[str, str, int, str]  # sample_id, kind, token index, replacement word ("" if none)


@dataclass(frozen=True)
class Edits:
    """Edits of one utterance, by index into its original tokens."""

    deleted: frozenset[int] = frozenset()
    swapped: frozenset[int] = frozenset()  # first index i of each swapped pair (i, i + 1)
    replaced: Mapping[int, str] = field(default_factory=dict)


NO_EDITS = Edits()


def align(n: int, edits: Edits) -> list[int]:
    """Original index of every output token: deleted ones are gone, swapped pairs are reversed."""
    order: list[int] = []
    i = 0
    while i < n:
        if i in edits.deleted:
            i += 1
        elif i in edits.swapped:
            order += [i + 1, i]
            i += 2
        else:
            order.append(i)
            i += 1
    return order


def remap_positions(positions: Iterable[int], order: Sequence[int]) -> list[int]:
    """Output positions of original positions; deleted ones are dropped."""
    new_at = {source: j for j, source in enumerate(order)}
    return sorted(new_at[p] for p in positions if p in new_at)


def _edited_punct(punct: Sequence[str], edits: Edits, order: Sequence[int]) -> list[str]:
    """Punctuation labels in output order; a deleted token hands its label to a previous O."""
    out = [punct[source] for source in order]
    for rank, index in enumerate(sorted(edits.deleted)):
        before = index - rank  # surviving tokens that precede the deleted one form a prefix of out
        if before and out[before - 1] == NO_PUNCT:
            out[before - 1] = punct[index]
    return out


def apply_edits(record: Mapping[str, Any], edits: Edits) -> dict[str, Any]:
    """A new record with tokens, labels and metadata.offsets edited; the input is not changed."""
    tokens, case = record["tokens"], record["case_labels"]
    order = align(len(tokens), edits)
    edited = {**record,
              "tokens": [edits.replaced.get(s, tokens[s]) for s in order],
              "punct_labels": _edited_punct(record["punct_labels"], edits, order),
              "case_labels": [case[s] for s in order]}
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and "offsets" in metadata:
        edited["metadata"] = {**metadata, "offsets": [metadata["offsets"][s] for s in order]}
    return edited


def _candidates(sample_id: str, tokens: Sequence[str], rng: random.Random) -> list[Candidate]:
    found: list[Candidate] = []
    for i, word in enumerate(tokens):
        if word in FUNCTION_WORDS:
            found.append((sample_id, "delete", i, ""))
        if word in HOMOPHONES:
            found.append((sample_id, "replace", i, rng.choice([w for w in HOMOPHONES[word] if w != word])))
        if i + 1 < len(tokens) - 1 and word != tokens[i + 1]:  # non-final, and not a no-op
            found.append((sample_id, "swap", i, ""))
    return found


class _Builder:
    """Accumulates accepted candidates per sample (internal; build_plan returns frozen Edits)."""

    def __init__(self, tokens_by_id: Mapping[str, Sequence[str]]) -> None:
        self.alive = {sid: len(tokens) for sid, tokens in tokens_by_id.items()}
        self.touched: dict[str, set[int]] = {sid: set() for sid in tokens_by_id}
        self.deleted: dict[str, set[int]] = {}
        self.swapped: dict[str, set[int]] = {}
        self.replaced: dict[str, dict[int, str]] = {}

    def try_add(self, candidate: Candidate, left: int) -> int:
        """Accept the candidate if it fits; return the cost spent (0 when skipped)."""
        sid, kind, i, word = candidate
        span = {i, i + 1} if kind == "swap" else {i}
        if COST[kind] > left or self.touched[sid] & span:
            return 0
        if kind == "delete":
            if self.alive[sid] <= 1:
                return 0
            self.alive[sid] -= 1
            self.deleted.setdefault(sid, set()).add(i)
        elif kind == "swap":
            self.swapped.setdefault(sid, set()).add(i)
        else:
            self.replaced.setdefault(sid, {})[i] = word
        self.touched[sid] |= span
        return COST[kind]

    def freeze(self) -> dict[str, Edits]:
        sids = self.deleted.keys() | self.swapped.keys() | self.replaced.keys()
        return {sid: Edits(frozenset(self.deleted.get(sid, ())), frozenset(self.swapped.get(sid, ())),
                           dict(self.replaced.get(sid, {}))) for sid in sids}


def edit_budget(tokens_by_id: Mapping[str, Sequence[str]], wer: float) -> int:
    return round(wer * sum(len(tokens) for tokens in tokens_by_id.values()))


def _next_accepted(builder: _Builder, candidates: Iterator[Candidate], left: int) -> int:
    """Cost of the next candidate the builder accepts (0 when the iterator runs dry).

    A skipped candidate never becomes acceptable later (edited tokens, the budget left and the
    tokens alive only shrink), so consuming it is safe.
    """
    return next(filter(None, (builder.try_add(c, left) for c in candidates)), 0)


def build_plan(tokens_by_id: Mapping[str, Sequence[str]], wer: float, seed: int) -> dict[str, Edits]:
    """Edits by sample_id (only samples with edits), with total cost exactly edit_budget()."""
    rng = random.Random(seed)
    found = [c for sid in sorted(tokens_by_id) for c in _candidates(sid, tokens_by_id[sid], rng)]
    pools = [[c for c in found if c[1] == kind] for kind in TURN_ORDER]
    for pool in pools:
        rng.shuffle(pool)
    queues = [iter(pool) for pool in pools]
    builder, left = _Builder(tokens_by_id), edit_budget(tokens_by_id, wer)
    progress = True
    while left and progress:  # a round that spends nothing means every candidate is used up
        progress = False
        for queue in queues:
            spent = _next_accepted(builder, queue, left)
            left, progress = left - spent, progress or spent > 0
            if not left:
                break
    if left:
        raise ValueError(f"not enough editable tokens for WER {wer:g}")
    return builder.freeze()


def edit_counts(plan: Mapping[str, Edits]) -> dict[str, int]:
    edits = list(plan.values())
    return {"deletion": sum(len(e.deleted) for e in edits),
            "substitution": sum(len(e.replaced) for e in edits),
            "swap": sum(len(e.swapped) for e in edits)}
