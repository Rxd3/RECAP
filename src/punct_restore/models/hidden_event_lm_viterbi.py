"""Exact Viterbi decoding of the hidden events of the hidden-event language model."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from punct_restore.models.ngram_lm import Context, NgramLM

History = tuple[int, ...]
LogProb = Callable[[int, History], float]
Back = dict[History, tuple[History, int]]  # new history -> (history it came from, choice made)


def _merge(
    best: dict[History, float], back: Back, states: dict[History, float], token: int, logprob: LogProb,
    span: int, choice: int,
) -> None:
    """Emit ``token`` from every state into ``best``/``back``; only a strictly better path replaces a stored one."""
    for history, score in states.items():
        new = (history + (token,))[-span:]
        total = score + logprob(token, history)
        if new not in best or total > best[new]:
            best[new], back[new] = total, (history, choice)


def _emit(states: dict[History, float], token: int, logprob: LogProb, span: int) -> tuple[dict[History, float], Back]:
    """Emit ``token`` from every state; paths that end in the same history merge, the best one is kept."""
    best: dict[History, float] = {}
    back: Back = {}
    _merge(best, back, states, token, logprob, span, 0)
    return best, back


def _choose(
    states: dict[History, float], events: Sequence[int], logprob: LogProb, span: int
) -> tuple[dict[History, float], Back]:
    """Choose no event (choice 0) or one event (choice i for events[i - 1]) after a word.

    "No event" is inserted first and events follow in vocabulary order, so ties prefer no event, then the
    earlier event.
    """
    best: dict[History, float] = dict(states)
    back: Back = {history: (history, 0) for history in states}
    for choice, event in enumerate(events, 1):
        _merge(best, back, states, event, logprob, span, choice)
    return best, back


def viterbi(lm: NgramLM, words: Sequence[int], events: Sequence[int], *, bos_id: int, eos_id: int) -> list[int]:
    """Best choice per word (0 = no event, i = events[i - 1]) under the exact order-n model, ``</s>`` included."""
    span = lm.order - 1
    contexts: dict[History, tuple[Context, ...]] = {}
    memo: dict[tuple[int, History], float] = {}

    def logprob(token: int, history: History) -> float:
        value = memo.get((token, history))
        if value is None:
            context = contexts.get(history)
            if context is None:
                context = contexts[history] = lm.contexts(history)
            value = memo[(token, history)] = math.log(lm.prob(token, context))
        return value

    states: dict[History, float] = {(bos_id,) * span: 0.0}
    trail: list[tuple[Back, Back]] = []
    for word in words:
        states, emitted = _emit(states, word, logprob, span)
        states, chosen = _choose(states, events, logprob, span)
        trail.append((emitted, chosen))
    states, ended = _emit(states, eos_id, logprob, span)
    state = ended[max(states, key=states.__getitem__)][0]
    choices = [0] * len(words)
    for index in range(len(words) - 1, -1, -1):
        emitted, chosen = trail[index]
        state, choices[index] = chosen[state]
        state = emitted[state][0]
    return choices
