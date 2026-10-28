"""Evaluation report (JSON dict and Markdown summary), owned by Member 5.

Schema "recap-eval-v1": inputs with SHA-256, counts, the primary score and every block of the
mode. Floats are rounded to 6 decimals; the Markdown view shows percentages with 2 decimals.
"""

from __future__ import annotations

from typing import Any

from .inputs import Bundle, Vocab
from .metrics import score_block
from .views import block_pairs, blocks_for

SCHEMA = "recap-eval-v1"
DECIMALS = 6
PRIMARY_METRIC = "macro_f1"


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, DECIMALS)
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    return value


def evaluate(bundle: Bundle, vocab: Vocab) -> dict[str, Any]:
    """Score every block of the bundle's mode; the first block is primary."""
    specs = blocks_for(bundle.label_mode, vocab)
    blocks = {spec.name: score_block(block_pairs(bundle.records, spec), spec.labels,
                                     spec.macro_over)
              for spec in specs}
    primary = specs[0].name
    inputs = {key: dict(value) if value is not None else None
              for key, value in bundle.inputs.items()}
    report = {"schema": SCHEMA, "label_mode": bundle.label_mode, "model_name": bundle.model_name,
              "inputs": inputs,
              "counts": {"sequences": len(bundle.records),
                         "tokens": sum(len(record.gold) for record in bundle.records),
                         "ignored_prediction_ids": bundle.ignored_prediction_ids},
              "primary": {"block": primary, "metric": PRIMARY_METRIC,
                          "value": blocks[primary][PRIMARY_METRIC]},
              "blocks": blocks}
    return _rounded(report)


def _pct(value: float) -> str:
    return f"{100 * value:.2f}"


def _md_text(text: str) -> str:
    """One-line Markdown-safe text: control characters become '?', table pipes are escaped."""
    printable = "".join(char if char.isprintable() else "?" for char in text)
    return printable.replace("|", "\\|")


def _header(report: dict[str, Any]) -> list[str]:
    counts, primary = report["counts"], report["primary"]
    inputs = ", ".join(f"{key} `{_md_text(value['path'])}` (sha256 {value['sha256']})"
                       for key, value in report["inputs"].items() if value is not None)
    return [f"# Evaluation: {_md_text(report['model_name'])} (mode {report['label_mode']})\n\n",
            f"Inputs: {inputs}.\n\n",
            (f"Sequences {counts['sequences']}, tokens {counts['tokens']}, "
             f"ignored prediction IDs {counts['ignored_prediction_ids']}.\n\n"),
            f"**Primary:** `{primary['block']}` macro F1 = {_pct(primary['value'])}%\n\n"]


def to_markdown(report: dict[str, Any]) -> str:
    """Short human-readable summary of a report produced by evaluate()."""
    lines = _header(report) + [
        "| Block | Scored positions | Macro F1 % | Micro F1 % | Accuracy % | Flag |\n",
        "| --- | ---: | ---: | ---: | ---: | --- |\n"]
    for name, block in report["blocks"].items():
        flag = "empty support" if block["empty_support"] else ""
        lines.append(f"| {name} | {block['scored_positions']} | {_pct(block['macro_f1'])} | "
                     f"{_pct(block['micro']['f1'])} | {_pct(block['accuracy'])} | {flag} |\n")
    primary = report["primary"]["block"]
    lines += [f"\n## Per class: {primary}\n\n",
              "| Label | Precision % | Recall % | F1 % | Support | Predicted |\n",
              "| --- | ---: | ---: | ---: | ---: | ---: |\n"]
    for label, scores in report["blocks"][primary]["per_class"].items():
        lines.append(f"| {_md_text(label)} | {_pct(scores['precision'])} | "
                     f"{_pct(scores['recall'])} | {_pct(scores['f1'])} | "
                     f"{scores['support']} | {scores['predicted']} |\n")
    return "".join(lines)
