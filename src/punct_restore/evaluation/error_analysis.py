"""Error examples for analysis and slides, owned by Member 5.

Every wrong position of one block is counted per (gold, predicted) pair, and up to K examples
per pair are drawn at random with a fixed seed, never picked by hand (EVALUATION_PROTOCOL.md:
"Do not select only favorable examples"). Each example shows the utterance around the error,
rebuilt once with gold and once with predicted labels.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from .inputs import Aligned, Bundle, EvaluationInputError, Vocab
from .metrics import SEPARATOR, split_joint
from .report import inputs_line, md_text, plain_inputs
from .uncertainty import DEFAULT_SEED
from .views import BlockSpec, blocks_for

SCHEMA = "recap-errors-v1"
DEFAULT_BLOCK = "punctuation/all"
DEFAULT_PER_PAIR = 3
WINDOW = 12  # tokens shown on each side of the error
MARKS = {"O": "", "COMMA": ",", "PERIOD": ".", "QUESTION": "?", "EXCLAMATION": "!",
         "COLON": ":", "SEMICOLON": ";", "DASH": " -"}


def _cased(token: str, case: str) -> str:
    if case == "ALL_UPPER":
        return token.upper()
    if case == "FIRST_UPPER":
        return token[:1].upper() + token[1:]
    return token


def render(tokens: Sequence[str], labels: Sequence[str]) -> str:
    """Text rebuilt from words and labels; joint labels also restore casing."""
    words = []
    for token, label in zip(tokens, labels, strict=False):
        punct, case = split_joint(label) if SEPARATOR in label else (label, "LOWER")
        words.append(_cased(token, case) + MARKS[punct])
    return " ".join(words)


def _window(record: Aligned, position: int, labels: Sequence[str]) -> str:
    start, end = max(0, position - WINDOW), min(len(record.tokens), position + WINDOW + 1)
    text = render(record.tokens[start:end], labels[start:end])
    return ("… " if start else "") + text + (" …" if end < len(record.tokens) else "")


def error_positions(records: Sequence[Aligned],
                    spec: BlockSpec) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """(gold, predicted) -> [(record index, position)] for every wrong position of the block."""
    found: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for index, record in enumerate(records):
        for position in spec.positions(record):
            gold = spec.projected(record.gold[position])
            predicted = spec.projected(record.pred[position])
            if gold != predicted:
                found[(gold, predicted)].append((index, position))
    return dict(found)


def _example(record: Aligned, position: int, pair: tuple[str, str]) -> dict[str, Any]:
    return {"gold": pair[0], "pred": pair[1], "sample_id": record.sample_id,
            "group_id": record.group_id, "position": position,
            "gold_text": _window(record, position, record.gold),
            "pred_text": _window(record, position, record.pred)}


def error_examples(bundle: Bundle, vocab: Vocab, block: str = DEFAULT_BLOCK,
                   per_pair: int = DEFAULT_PER_PAIR, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Counts of every error pair and up to `per_pair` randomly drawn examples of each."""
    specs = {spec.name: spec for spec in blocks_for(bundle.label_mode, vocab)}
    if block not in specs:
        raise EvaluationInputError(f"block {block!r} is not one of {list(specs)}")
    if per_pair < 1:
        raise EvaluationInputError("per_pair must be at least 1")
    if any(not record.tokens for record in bundle.records):
        raise EvaluationInputError("error examples need the gold tokens (use load_bundle)")
    found = error_positions(bundle.records, specs[block])
    pairs = sorted(found, key=lambda pair: (-len(found[pair]), pair))
    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    for pair in pairs:
        chosen = rng.sample(found[pair], min(per_pair, len(found[pair])))
        examples += [_example(bundle.records[i], position, pair) for i, position in sorted(chosen)]
    return {"schema": SCHEMA, "label_mode": bundle.label_mode, "model_name": bundle.model_name,
            "block": block, "seed": seed, "per_pair": per_pair, "inputs": plain_inputs(bundle),
            "errors_total": sum(len(found[pair]) for pair in pairs),
            "pair_counts": [{"gold": g, "pred": p, "count": len(found[(g, p)])} for g, p in pairs],
            "examples": examples}


def errors_markdown(report: dict[str, Any]) -> str:
    """Error-pair counts and the drawn examples, ready for analysis notes or slides."""
    lines = [(f"# Errors: {md_text(report['model_name'])} (mode {report['label_mode']}, "
              f"block `{report['block']}`)\n\n"),
             inputs_line(report["inputs"]),
             (f"{report['errors_total']} wrong positions. Up to {report['per_pair']} examples per "
              f"pair, drawn at random with seed {report['seed']} (not hand-picked).\n\n"),
             "| Gold | Predicted | Count |\n", "| --- | --- | ---: |\n"]
    lines += [f"| {md_text(row['gold'])} | {md_text(row['pred'])} | {row['count']} |\n"
              for row in report["pair_counts"]]
    for example in report["examples"]:
        lines += [(f"\n**{md_text(example['gold'])} → {md_text(example['pred'])}** "
                   f"(`{md_text(example['sample_id'])}`, word {example['position']})\n\n"),
                  f"- gold: {md_text(example['gold_text'])}\n",
                  f"- pred: {md_text(example['pred_text'])}\n"]
    return "".join(lines)
