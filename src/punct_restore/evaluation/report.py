"""Evaluation report (JSON dict and Markdown summary), owned by Member 5.

Schema "recap-eval-v1": inputs with SHA-256, counts, the primary score and every block of the
mode, plus a grouped bootstrap interval per block when requested. Floats are rounded to 6
decimals; the Markdown view shows percentages with 2 decimals.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from .inputs import Bundle, Vocab
from .metrics import score_block
from .uncertainty import BootstrapSpec, block_intervals
from .views import block_pairs, blocks_for

SCHEMA = "recap-eval-v1"
DECIMALS = 6
PRIMARY_METRIC = "macro_f1"
PCT_STEP = Decimal("0.01")


def rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, DECIMALS)
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return value


def plain_inputs(bundle: Bundle) -> dict[str, dict[str, str] | None]:
    """The bundle's read-only input identities as plain JSON-ready dicts."""
    return {key: dict(value) if value is not None else None
            for key, value in bundle.inputs.items()}


def evaluate(bundle: Bundle, vocab: Vocab,
             bootstrap: BootstrapSpec | None = None) -> dict[str, Any]:
    """Score every block of the bundle's mode (first block primary); optional bootstrap CIs."""
    specs = blocks_for(bundle.label_mode, vocab)
    blocks = {spec.name: score_block(block_pairs(bundle.records, spec), spec.labels,
                                     spec.macro_over)
              for spec in specs}
    if bootstrap is not None:
        intervals = block_intervals(bundle.records, specs, bootstrap)
        blocks = {name: {**block, "macro_f1_ci": intervals[name]} for name, block in blocks.items()}
    primary = specs[0].name
    report = {"schema": SCHEMA, "label_mode": bundle.label_mode,
              "policy_version": bundle.policy_version, "model_name": bundle.model_name,
              "inputs": plain_inputs(bundle),
              "counts": {"sequences": len(bundle.records),
                         "tokens": sum(len(record.gold) for record in bundle.records),
                         "ignored_prediction_ids": bundle.ignored_prediction_ids},
              "primary": {"block": primary, "metric": PRIMARY_METRIC,
                          "value": blocks[primary][PRIMARY_METRIC]},
              "blocks": blocks}
    return rounded(report)


def pct(value: float) -> str:
    """Percentage with 2 decimals, rounded half-up from the stored decimal value.

    Formatting the float directly would print 0.67985 as 67.98, because 100 * 0.67985 is
    67.98499... in binary floating point.
    """
    return str((Decimal(repr(value)) * 100).quantize(PCT_STEP, rounding=ROUND_HALF_UP))


def printable(text: str) -> str:
    """One-line text: control characters (newlines, escape codes) become '?'."""
    return "".join(char if char.isprintable() else "?" for char in text)


def md_text(text: str) -> str:
    """One-line Markdown-safe text: printable() plus escaped table pipes."""
    return printable(text).replace("|", "\\|")


def ci_text(ci: dict[str, Any]) -> str:
    """'low to high' in percent (reads well for negative differences too)."""
    return f"{pct(ci['low'])} to {pct(ci['high'])}"


def policy_text(version: str | None) -> str:
    """' Data policy `m1-v2`.' for a versioned gold file, '' when the gold carries no version."""
    return f" Data policy `{md_text(version)}`." if version is not None else ""


def inputs_line(inputs: dict[str, Any]) -> str:
    listed = ", ".join(f"{key} `{md_text(value['path'])}` (sha256 {value['sha256']})"
                       for key, value in inputs.items() if value is not None)
    return f"Inputs: {listed}.\n\n"


def _header(report: dict[str, Any]) -> list[str]:
    counts, primary = report["counts"], report["primary"]
    return [f"# Evaluation: {md_text(report['model_name'])} (mode {report['label_mode']})\n\n",
            inputs_line(report["inputs"]),
            (f"Sequences {counts['sequences']}, tokens {counts['tokens']}, "
             f"ignored prediction IDs {counts['ignored_prediction_ids']}."
             f"{policy_text(report['policy_version'])}\n\n"),
            f"**Primary:** `{primary['block']}` macro F1 = {pct(primary['value'])}%\n\n"]


def _block_rows(report: dict[str, Any]) -> list[str]:
    with_ci = any("macro_f1_ci" in block for block in report["blocks"].values())
    extra_head, extra_rule = (" Macro F1 95% CI |", " ---: |") if with_ci else ("", "")
    rows = [("| Block | Scored positions | Macro F1 % | Micro F1 % | Accuracy % | Flag |"
             f"{extra_head}\n"),
            f"| --- | ---: | ---: | ---: | ---: | --- |{extra_rule}\n"]
    for name, block in report["blocks"].items():
        flag = "empty support" if block["empty_support"] else ""
        ci = f" {ci_text(block['macro_f1_ci'])} |" if with_ci else ""
        rows.append(f"| {name} | {block['scored_positions']} | {pct(block['macro_f1'])} | "
                    f"{pct(block['micro']['f1'])} | {pct(block['accuracy'])} | {flag} |{ci}\n")
    return rows


def to_markdown(report: dict[str, Any]) -> str:
    """Short human-readable summary of a report produced by evaluate()."""
    lines = _header(report) + _block_rows(report)
    primary = report["primary"]["block"]
    lines += [f"\n## Per class: {primary}\n\n",
              "| Label | Precision % | Recall % | F1 % | Support | Predicted |\n",
              "| --- | ---: | ---: | ---: | ---: | ---: |\n"]
    for label, scores in report["blocks"][primary]["per_class"].items():
        lines.append(f"| {md_text(label)} | {pct(scores['precision'])} | "
                     f"{pct(scores['recall'])} | {pct(scores['f1'])} | "
                     f"{scores['support']} | {scores['predicted']} |\n")
    return "".join(lines)
