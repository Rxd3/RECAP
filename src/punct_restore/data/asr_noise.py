"""ASR-like word errors for gold records, with every label kept consistent (robustness study).

Used by scripts/simulate_asr_noise.py, which does the file reading and writing.

Edit model (v1, default). One plan is drawn from the tokens of a set of samples and applied to each record. Candidates:
deletion of a function word (cost 1, never the last remaining token), substitution by a homophone of the same
group (cost 1), swap of two adjacent distinct non-final tokens (cost 2). The budget is round(wer * tokens).
The candidates of each type are shuffled separately with random.Random(seed), then taken round-robin by type in
the fixed order deletion, substitution, swap: each turn gives the next acceptable candidate of that type, and a
type with no candidates left is skipped. A candidate is skipped when it touches an edited token, overspends what
is left of the budget, or would delete an utterance's last token. The run stops when the budget is exactly spent
and fails if every candidate is used up first. So the mix is as balanced as the data allows: deletions and
substitutions take their turn first, and swaps (which alone would be most of the candidates) fill the rest.

Edit model (v2). Selected with edit_model="v2" (--edit-model v2 in simulate_asr_noise.py). Uses one seeded
random.Random(seed) and the same round(wer * tokens) budget, targeting a substitution-dominant cost mix of about
45% substitution, 30% deletion, 15% insertion and 10% swap (weights 9 : 6 : 3 : 2, with swaps capped at 10% of
the budget):
  * Insertion (cost 1): insert a filler or short function word (um, uh, the, a, and, so, like, i) before a token.
    1. The inserted token gets punctuation O and casing LOWER.
    2. Its mode C casing mask is 0 (its output position is added to excluded_positions), so it is not scored.
    3. Its metadata.offsets span is empty ([start, start]), at the position of the next token.
  * Confusion substitution (cost 1): replace a word with a different word at character edit distance 1 that has
    the same first letter, drawn only from the default_train vocabulary with a minimum count of 5. Keeps all
    labels. Homophone substitution stays as a second substitution source.
  * Deletion (cost 1): a function word, or any word of 3 characters or fewer, never the last remaining token.
    The label rules are as in v1.
  * Swap (cost 2): as in v1, but capped at 10% of the budget.

Labels. A deleted token's case label is dropped; its punctuation label moves to the previous surviving token if
that carries O, else it is dropped. A substitution keeps all labels; a swap moves both tokens with their labels;
an insertion gets punctuation O, casing LOWER, mode C casing mask 0 (excluded from casing/joint scoring) and an
empty metadata.offsets span at the position of the next token. metadata.offsets (one span per token) is edited
like the tokens. Other fields are left unchanged.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

NO_PUNCT = "O"
LOWER = "LOWER"
EDIT_MODELS = ("v1", "v2")
MIN_TRAIN_COUNT = 5
MAX_SHORT_DELETE = 3
SWAP_CAP_SHARE = 0.10
COST = {"delete": 1, "replace": 1, "swap": 2, "insert": 1}
TURN_ORDER = ("delete", "replace", "swap")  # the v1 round-robin order of the candidate types
V2_TURN_ORDER = ("replace", "delete", "insert", "swap")
V2_WEIGHTS = {"replace": 9, "delete": 6, "insert": 3, "swap": 2}  # 45%, 30%, 15%, 10% of a 20-cost cycle
INSERT_WORDS = ("um", "uh", "the", "a", "and", "so", "like", "i")
FUNCTION_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "have", "has", "had",
    "be", "been", "will", "would", "can", "could", "to", "of", "in", "on", "at", "for", "and",
    "or", "that"})
HOMOPHONE_GROUPS = (("their", "there", "they're"), ("your", "you're"), ("its", "it's"),
                    ("to", "too", "two"), ("then", "than"))
HOMOPHONES = {word: group for group in HOMOPHONE_GROUPS for word in group}
Candidate = tuple[str, str, int, str]  # sample_id, kind, token index, replacement/inserted word ("" if none)


@dataclass(frozen=True)
class Edits:
    """Edits of one utterance, by index into its original tokens."""

    deleted: frozenset[int] = frozenset()
    swapped: frozenset[int] = frozenset()  # first index i of each swapped pair (i, i + 1)
    replaced: Mapping[int, str] = field(default_factory=dict)
    inserted: Mapping[int, str] = field(default_factory=dict)  # word inserted before index i


NO_EDITS = Edits()


def align(n: int, edits: Edits) -> list[int]:
    """Source of every output token: >= 0 for original index i, ~i (< 0) for a word inserted before i."""
    order: list[int] = []
    i = 0
    while i < n:
        if i in edits.inserted:
            order.append(~i)
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
    """Output positions of original excluded positions plus any inserted tokens; deleted ones are dropped."""
    excluded = set(positions)
    return [j for j, source in enumerate(order) if source < 0 or source in excluded]


def _edited_punct(punct: Sequence[str], edits: Edits, order: Sequence[int]) -> list[str]:
    """Punctuation labels in output order; insertions get O, and a deleted token hands its label to a previous O."""
    out = [NO_PUNCT if source < 0 else punct[source] for source in order]
    for index in sorted(edits.deleted):
        before = sum(1 for s in order if (~s <= index if s < 0 else s < index))
        if before and out[before - 1] == NO_PUNCT:
            out[before - 1] = punct[index]
    return out


def _edited_offsets(offsets: Sequence[Any], order: Sequence[int]) -> list[Any]:
    out = []
    for j, source in enumerate(order):
        if source >= 0:
            out.append(offsets[source])
        else:
            next_tok = next((k for k in order[j + 1:] if k >= 0), ~source)
            pos = offsets[next_tok][0]
            out.append([pos, pos])
    return out


def apply_edits(record: Mapping[str, Any], edits: Edits) -> dict[str, Any]:
    """A new record with tokens, labels and metadata.offsets edited; the input is not changed."""
    tokens, case = record["tokens"], record["case_labels"]
    order = align(len(tokens), edits)
    edited = {**record,
              "tokens": [edits.inserted[~s] if s < 0 else edits.replaced.get(s, tokens[s]) for s in order],
              "punct_labels": _edited_punct(record["punct_labels"], edits, order),
              "case_labels": [LOWER if s < 0 else case[s] for s in order]}
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and "offsets" in metadata:
        edited["metadata"] = {**metadata, "offsets": _edited_offsets(metadata["offsets"], order)}
    return edited


def train_vocabulary(token_lists: Iterable[Sequence[str]], min_count: int = MIN_TRAIN_COUNT) -> frozenset[str]:
    """Words occurring at least min_count times across the training token sequences."""
    counts: Counter[str] = Counter()
    for tokens in token_lists:
        counts.update(tokens)
    return frozenset(word for word, count in counts.items() if count >= min_count)


def _replacements_v2(words: Iterable[str], vocab: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Per word, sorted distinct replacements from homophones and edit-distance-1 same-initial vocab words."""
    vocab_set = set(vocab)
    by_del: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_sub: dict[tuple[str, int, str], list[str]] = defaultdict(list)
    for v in sorted(vocab_set):
        if not v:
            continue
        c0 = v[0]
        for k in range(len(v)):
            short = v[:k] + v[k + 1:]
            by_del[(c0, short)].append(v)
            if k > 0:
                by_sub[(c0, k, short)].append(v)
    out: dict[str, tuple[str, ...]] = {}
    for word in sorted(set(words)):
        if not word:
            continue
        c0 = word[0]
        found = {w for w in HOMOPHONES.get(word, ()) if w != word}
        found.update(by_del.get((c0, word), ()))
        for k in range(len(word)):
            short = word[:k] + word[k + 1:]
            if short and short[0] == c0 and short in vocab_set:
                found.add(short)
            if k > 0:
                found.update(by_sub.get((c0, k, short), ()))
        found.discard(word)
        if found:
            out[word] = tuple(sorted(found))
    return out


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


def _candidates_v2(
    sample_id: str,
    tokens: Sequence[str],
    replacements: Mapping[str, Sequence[str]],
    rng: random.Random,
) -> list[Candidate]:
    found: list[Candidate] = []
    for i, word in enumerate(tokens):
        found.append((sample_id, "insert", i, rng.choice(INSERT_WORDS)))
        if word in FUNCTION_WORDS or len(word) <= MAX_SHORT_DELETE:
            found.append((sample_id, "delete", i, ""))
        if word in replacements:
            found.append((sample_id, "replace", i, rng.choice(replacements[word])))
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
        self.inserted: dict[str, dict[int, str]] = {}

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
        elif kind == "insert":
            self.inserted.setdefault(sid, {})[i] = word
        elif kind == "swap":
            self.swapped.setdefault(sid, set()).add(i)
        else:
            self.replaced.setdefault(sid, {})[i] = word
        self.touched[sid] |= span
        return COST[kind]

    def freeze(self) -> dict[str, Edits]:
        sids = self.deleted.keys() | self.swapped.keys() | self.replaced.keys() | self.inserted.keys()
        return {sid: Edits(frozenset(self.deleted.get(sid, ())), frozenset(self.swapped.get(sid, ())),
                           dict(self.replaced.get(sid, {})), dict(self.inserted.get(sid, {}))) for sid in sids}


def edit_budget(tokens_by_id: Mapping[str, Sequence[str]], wer: float) -> int:
    return round(wer * sum(len(tokens) for tokens in tokens_by_id.values()))


def _next_accepted(builder: _Builder, candidates: Iterator[Candidate], left: int) -> int:
    """Cost of the next candidate the builder accepts (0 when the iterator runs dry).

    A skipped candidate never becomes acceptable later (edited tokens, the budget left and the
    tokens alive only shrink), so consuming it is safe.
    """
    return next(filter(None, (builder.try_add(c, left) for c in candidates)), 0)


def _build_plan_v2(
    tokens_by_id: Mapping[str, Sequence[str]],
    wer: float,
    rng: random.Random,
    vocab: Iterable[str],
) -> dict[str, Edits]:
    words = {w for tokens in tokens_by_id.values() for w in tokens}
    replacements = _replacements_v2(words, vocab)
    found = [c for sid in sorted(tokens_by_id) for c in _candidates_v2(sid, tokens_by_id[sid], replacements, rng)]
    pools = {kind: [c for c in found if c[1] == kind] for kind in V2_TURN_ORDER}
    for kind in V2_TURN_ORDER:
        rng.shuffle(pools[kind])
    active = {kind: iter(pools[kind]) for kind in V2_TURN_ORDER}
    builder, budget = _Builder(tokens_by_id), edit_budget(tokens_by_id, wer)
    left, swap_cap = budget, int(SWAP_CAP_SHARE * budget)
    spent_by_kind = {kind: 0 for kind in V2_TURN_ORDER}
    while left and active:
        eligible = [
            kind for kind in V2_TURN_ORDER
            if kind in active and COST[kind] <= left
            and (kind != "swap" or spent_by_kind["swap"] + COST["swap"] <= swap_cap)
        ]
        if not eligible:
            break
        kind = min(
            eligible,
            key=lambda k: ((spent_by_kind[k] + COST[k]) * (18 // V2_WEIGHTS[k]), V2_TURN_ORDER.index(k)),
        )
        spent = _next_accepted(builder, active[kind], left)
        if not spent:
            del active[kind]
        else:
            spent_by_kind[kind] += spent
            left -= spent
    if left:
        raise ValueError(f"not enough editable tokens for WER {wer:g}")
    return builder.freeze()


def build_plan(
    tokens_by_id: Mapping[str, Sequence[str]],
    wer: float,
    seed: int,
    *,
    edit_model: str = "v1",
    vocab: Iterable[str] = (),
) -> dict[str, Edits]:
    """Edits by sample_id (only samples with edits), with total cost exactly edit_budget()."""
    if edit_model not in EDIT_MODELS:
        raise ValueError(f"unknown edit model {edit_model!r}")
    rng = random.Random(seed)
    if edit_model == "v2":
        return _build_plan_v2(tokens_by_id, wer, rng, vocab)
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


def edit_counts(plan: Mapping[str, Edits], edit_model: str = "v1") -> dict[str, int]:
    edits = list(plan.values())
    counts = {"deletion": sum(len(e.deleted) for e in edits),
              "substitution": sum(len(e.replaced) for e in edits),
              "swap": sum(len(e.swapped) for e in edits)}
    if edit_model == "v2" or any(e.inserted for e in edits):
        counts["insertion"] = sum(len(e.inserted) for e in edits)
    return counts
