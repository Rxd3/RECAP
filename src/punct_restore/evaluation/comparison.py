"""Paired comparison of two prediction sets on the same scored records, owned by Member 5.

For every block: macro F1 of A and of B, the difference B - A and, with a bootstrap, its paired
95% interval (both models scored on the same resampled videos) and the share of resamples in
which B beats A. Compare models within one label mode only (EVALUATION_PROTOCOL.md).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
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


def _check_classes(classes: Sequence[str], labels: Sequence[str], mode: str) -> None:
    seen: set[str] = set()
    for label in classes:
        if label not in labels:
            raise EvaluationInputError(
                f"--class {label!r} is not a mode {mode} label; choose from {list(labels)}")
        if label in seen:
            raise EvaluationInputError(f"--class {label!r} was given more than once")
        seen.add(label)


def _class_report(bundle_a: Bundle, bundle_b: Bundle, specs: Sequence[BlockSpec],
                  classes: Sequence[str], bootstrap: BootstrapSpec | None) -> dict[str, Any]:
    """Per requested label and applicable block: F1 of A and B, B - A and its paired interval.

    A class F1 is the macro F1 over that one class. Mode C keeps only the primary joint block:
    the other blocks are in punctuation or casing space, where a joint label has no meaning.
    """
    applicable = specs[:1] if bundle_a.label_mode == "C" else specs
    class_specs = [(label, spec, replace(spec, name=f"{label} @ {spec.name}", macro_over=(label,)))
             for label in classes for spec in applicable]
    intervals = ({} if bootstrap is None else paired_differences(
        bundle_a.records, bundle_b.records, [single for _, _, single in class_specs], bootstrap))
    report: dict[str, dict[str, Any]] = {label: {} for label in classes}
    for label, spec, single in class_specs:
        a, b = _macro_f1(bundle_a, single), _macro_f1(bundle_b, single)
        report[label][spec.name] = {
            "f1_a": a, "f1_b": b, "difference": b - a,
            **({"difference_ci": intervals[single.name]} if intervals else {})}
    return report


def compare(bundle_a: Bundle, bundle_b: Bundle, vocab: Vocab,
            bootstrap: BootstrapSpec | None = None,
            classes: Sequence[str] = ()) -> dict[str, Any]:
    """Comparison report of B against A over every block of their shared label mode.

    `classes` adds a top-level "classes" entry with the F1 difference of each listed label.
    """
    _require_same_scored_set(bundle_a, bundle_b)
    specs = blocks_for(bundle_a.label_mode, vocab)
    _check_classes(classes, specs[0].labels, bundle_a.label_mode)  # before the slow bootstrap
    intervals = ({} if bootstrap is None
                 else paired_differences(bundle_a.records, bundle_b.records, specs, bootstrap))
    blocks = {}
    for spec in specs:
        a, b = _macro_f1(bundle_a, spec), _macro_f1(bundle_b, spec)
        blocks[spec.name] = {"macro_f1_a": a, "macro_f1_b": b, "difference": b - a,
                             **({"difference_ci": intervals[spec.name]} if intervals else {})}
    per_class = ({"classes": _class_report(bundle_a, bundle_b, specs, classes, bootstrap)}
                 if classes else {})
    return rounded({"schema": SCHEMA, "label_mode": bundle_a.label_mode,
                    "policy_version": bundle_a.policy_version,  # same gold file for A and B
                    "model_a": bundle_a.model_name, "model_b": bundle_b.model_name,
                    "inputs_a": plain_inputs(bundle_a), "inputs_b": plain_inputs(bundle_b),
                    "sequences": len(bundle_a.records), "primary_block": specs[0].name,
                    "blocks": blocks, **per_class})


def _class_lines(classes: dict[str, Any]) -> list[str]:
    """The 'Per-class F1' section: one row per label and block, CI columns only with a CI."""
    with_ci = any("difference_ci" in row for blocks in classes.values() for row in blocks.values())
    head = "| Class | Block | A % | B % | B − A (points) |"
    head += " 95% CI of B − A | Share of resamples B > A |\n" if with_ci else "\n"
    rule = "| --- | --- | ---: | ---: | ---: |" + (" ---: | ---: |\n" if with_ci else "\n")
    lines = ["\n## Per-class F1 (B − A)\n\n", head, rule]
    for label, blocks in classes.items():
        for name, block in blocks.items():
            row = (f"| {md_text(label)} | {name} | {pct(block['f1_a'])} | {pct(block['f1_b'])} | "
                   f"{pct(block['difference'])} |")
            if with_ci:
                ci = block["difference_ci"]
                row += f" {ci_text(ci)} | {pct(ci['share_b_better'])}% |"
            lines.append(row + "\n")
    return lines


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
    if "classes" in report:
        lines += _class_lines(report["classes"])
    return "".join(lines)
