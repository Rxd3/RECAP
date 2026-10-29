"""Paired comparison of two prediction sets on the same scored records, owned by Member 5.

For every block: macro F1 of A and of B, the difference B - A and, with a bootstrap, its paired
95% interval (both models scored on the same resampled videos) and the share of resamples in
which B beats A. Compare models within one label mode only (EVALUATION_PROTOCOL.md).
"""

from __future__ import annotations

from typing import Any

from .inputs import Bundle, EvaluationInputError, Vocab
from .metrics import score_block
from .report import (
    ci_text,
    inputs_line,
    md_text,
    pct,
    plain_inputs,
    policy_text,
    rounded,
)
from .uncertainty import BootstrapSpec, paired_differences
from .views import BlockSpec, block_pairs, blocks_for

SCHEMA = "recap-compare-v1"


def _require_same_scored_set(bundle_a: Bundle, bundle_b: Bundle) -> None:
    if bundle_a.label_mode != bundle_b.label_mode:
        raise EvaluationInputError("compared prediction sets must have the same label mode")
    if any(bundle_a.inputs[key] != bundle_b.inputs[key] for key in ("gold", "ids", "masks")):
        raise EvaluationInputError(
            "compared prediction sets must use the same gold file, ID list and masks")


def _macro_f1(bundle: Bundle, spec: BlockSpec) -> float:
    return score_block(block_pairs(bundle.records, spec), spec.labels, spec.macro_over)["macro_f1"]


def compare(bundle_a: Bundle, bundle_b: Bundle, vocab: Vocab,
            bootstrap: BootstrapSpec | None = None) -> dict[str, Any]:
    """Comparison report of B against A over every block of their shared label mode."""
    _require_same_scored_set(bundle_a, bundle_b)
    specs = blocks_for(bundle_a.label_mode, vocab)
    intervals = ({} if bootstrap is None
                 else paired_differences(bundle_a.records, bundle_b.records, specs, bootstrap))
    blocks = {}
    for spec in specs:
        a, b = _macro_f1(bundle_a, spec), _macro_f1(bundle_b, spec)
        blocks[spec.name] = {"macro_f1_a": a, "macro_f1_b": b, "difference": b - a,
                             **({"difference_ci": intervals[spec.name]} if intervals else {})}
    return rounded({"schema": SCHEMA, "label_mode": bundle_a.label_mode,
                    "policy_version": bundle_a.policy_version,  # same gold file for A and B
                    "model_a": bundle_a.model_name, "model_b": bundle_b.model_name,
                    "inputs_a": plain_inputs(bundle_a), "inputs_b": plain_inputs(bundle_b),
                    "sequences": len(bundle_a.records), "primary_block": specs[0].name,
                    "blocks": blocks})


def comparison_markdown(report: dict[str, Any]) -> str:
    """Short table of a compare() report."""
    a, b = md_text(report["model_a"]), md_text(report["model_b"])
    with_ci = any("difference_ci" in block for block in report["blocks"].values())
    head = f"| Block | A: {a} % | B: {b} % | B − A (points) |"
    head += " 95% CI of B − A | Share of resamples B > A |\n" if with_ci else "\n"
    rule = "| --- | ---: | ---: | ---: |" + (" ---: | ---: |\n" if with_ci else "\n")
    lines = [f"# Comparison: {b} vs {a} (mode {report['label_mode']})\n\n",
             "A " + inputs_line(report["inputs_a"]), "B " + inputs_line(report["inputs_b"]),
             (f"Sequences {report['sequences']}. Primary block `{report['primary_block']}`."
              f"{policy_text(report['policy_version'])}\n\n"),
             head, rule]
    for name, block in report["blocks"].items():
        row = (f"| {name} | {pct(block['macro_f1_a'])} | {pct(block['macro_f1_b'])} | "
               f"{pct(block['difference'])} |")
        if with_ci:
            ci = block["difference_ci"]
            row += f" {ci_text(ci)} | {pct(ci['share_b_better'])}% |"
        lines.append(row + "\n")
    return "".join(lines)
