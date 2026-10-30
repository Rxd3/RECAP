"""Interpolated absolute-discounting n-gram counts and probabilities (the core of the hidden-event LM).

Tokens are small integers 0..num_ids-1; id 0 is the start symbol ``<s>``, which only ever appears as history
and is never predicted. An n-gram (t1, ..., tk) with t_k the predicted token is stored as the single integer
t1*M^(k-1) + ... + t_(k-1)*M + t_k with M = num_ids, one table per length, so a context key is the n-gram key
divided by M. Counts come from predicted positions only: for every position i >= order-1 of a stream, the n-grams
of length 1..order that end at i are counted.

For a history h (the last tokens emitted) and the history h' that drops its oldest token:

    P(w | h) = max(c(h,w) - D, 0) / c(h) + D * N1+(h.) / c(h) * P(w | h')      if c(h) > 0
    P(w | h) = P(w | h')                                                       if c(h) = 0
    P(w)     = max(c(w) - D, 0) / N + D * N1+(.) / N * 1/|V|                   (empty history)

where c(h) = sum over w of c(h,w), N1+(h.) is the number of distinct w with c(h,w) > 0, N = sum of c(w) and
|V| = num_ids - 1 (every id but ``<s>``). One discount D serves every order. Each P(. | h) sums to 1 over V.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

# (history length, context key, c(h), N1+(h.)) for one backoff level whose history was seen in training
Context = tuple[int, int, int, int]


def count_ngrams(streams: Iterable[Sequence[int]], order: int, num_ids: int) -> list[dict[int, int]]:
    """Counts of the n-grams of length 1..order ending at each predicted position of every stream.

    The first ``order - 1`` tokens of a stream are the ``<s>`` padding that serves as history only.
    """
    tables: list[dict[int, int]] = [{} for _ in range(order)]
    for stream in streams:
        for position in range(order - 1, len(stream)):
            key, multiplier = 0, 1
            for back, table in enumerate(tables):  # the n-gram of length back+1 ends at this position
                key += stream[position - back] * multiplier
                multiplier *= num_ids
                table[key] = table.get(key, 0) + 1
    return tables


class NgramLM:
    """The probabilities above, built once from the counts; read-only afterwards."""

    def __init__(self, counts: Sequence[Mapping[int, int]], *, discount: float, num_ids: int) -> None:
        if not counts or not counts[0]:
            raise ValueError("the language model needs at least one predicted token")
        self.order = len(counts)
        self.counts = counts
        self._discount = discount
        self._num_ids = num_ids
        self._unigram = _unigram_probabilities(counts[0], discount, num_ids)
        self._totals: list[dict[int, int]] = [{}]
        self._types: list[dict[int, int]] = [{}]
        for length in range(1, self.order):
            totals: dict[int, int] = {}
            types: dict[int, int] = {}
            for key, count in counts[length].items():
                history = key // num_ids
                totals[history] = totals.get(history, 0) + count
                types[history] = types.get(history, 0) + 1
            self._totals.append(totals)
            self._types.append(types)

    def contexts(self, history: Sequence[int]) -> tuple[Context, ...]:
        """The seen backoff levels of ``history`` (its last order-1 tokens), shortest first."""
        found: list[Context] = []
        key, multiplier = 0, 1
        for length in range(1, min(len(history), self.order - 1) + 1):
            key += history[-length] * multiplier
            multiplier *= self._num_ids
            total = self._totals[length].get(key)
            if total:
                found.append((length, key, total, self._types[length][key]))
        return tuple(found)

    def prob(self, token: int, contexts: tuple[Context, ...]) -> float:
        """P(token | history) for the contexts of a history, by interpolation from the unigram upwards."""
        discount, num_ids = self._discount, self._num_ids
        probability = self._unigram[token]
        for length, key, total, types in contexts:
            count = self.counts[length].get(key * num_ids + token, 0)
            probability = (max(count - discount, 0.0) + discount * types * probability) / total
        return probability


def _unigram_probabilities(counts: Mapping[int, int], discount: float, num_ids: int) -> list[float]:
    total = sum(counts.values())
    floor = discount * len(counts) / total / (num_ids - 1)
    probabilities = [floor] * num_ids
    probabilities[0] = 0.0
    for token, count in counts.items():
        probabilities[token] = max(count - discount, 0.0) / total + floor
    return probabilities


def counts_to_payload(counts: Sequence[Mapping[int, int]]) -> list[dict[str, list[int]]]:
    """One ``{"keys", "counts"}`` entry per n-gram length, keys ascending (so equal models write equal bytes)."""
    payload = []
    for table in counts:
        keys = sorted(table)
        payload.append({"keys": keys, "counts": [table[key] for key in keys]})
    return payload


def counts_from_payload(payload: object, order: int, num_ids: int) -> list[dict[int, int]]:
    """Validated count tables from ``counts_to_payload`` output; ValueError for anything else."""
    if not isinstance(payload, list) or len(payload) != order:
        raise ValueError(f"ngrams must hold {order} tables")
    tables = [_table(entry, length, num_ids) for length, entry in enumerate(payload, 1)]
    if not tables[0]:
        raise ValueError("ngrams hold no unigram counts")
    return tables


MAX_COUNT = 2**53  # larger counts cannot be turned into floats exactly (or at all), so a file holding one is corrupt


def _table(entry: object, length: int, num_ids: int) -> dict[int, int]:
    if not isinstance(entry, dict) or set(entry) != {"keys", "counts"}:
        raise ValueError(f"the {length}-gram table must have exactly keys and counts")
    keys, counts = entry["keys"], entry["counts"]
    if not isinstance(keys, list) or not isinstance(counts, list) or len(keys) != len(counts):
        raise ValueError(f"the {length}-gram keys and counts must be lists of equal length")
    limit = num_ids**length
    if any(type(key) is not int or not 0 <= key < limit or key % num_ids == 0 for key in keys):
        raise ValueError(f"the {length}-gram keys must be ids ending in a predicted token")
    if any(type(count) is not int or not 1 <= count <= MAX_COUNT for count in counts):
        raise ValueError(f"the {length}-gram counts must be integers from 1 to 2**53")
    table = dict(zip(keys, counts, strict=True))
    if len(table) != len(keys):
        raise ValueError(f"the {length}-gram table repeats a key")
    return table
