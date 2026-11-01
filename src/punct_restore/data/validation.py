"""Validation against docs/DATA_CONTRACT.md (schema v1), owned by Member 1.

validate_record()  checks one record's fields, types, vocabularies and alignment;
validate_file()    checks a JSONL file: one object per physical line, no blank lines, unique sample IDs;
compare_views()    checks that the A/B/C views share IDs, tokens and splits.
Errors are returned as strings so a release build can count and report them; nothing is repaired.
"""
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

SCHEMA_VERSION = 1
SPLITS = ("train", "validation", "test")
MODES = ("A", "B", "C")
REQUIRED = ("schema_version", "sample_id", "source", "split", "label_mode", "tokens", "punct_labels", "case_labels")
OPTIONAL = ("metadata",)
TARGET_MARK_END = re.compile(r"[,?!;:]$")


def validate_record(rec: object, label_sets: Mapping[str, Any]) -> list[str]:
    """List of contract violations for one record (empty list = valid)."""
    if not isinstance(rec, dict):
        return ["record is not a JSON object"]
    errors = [f"missing field {k}" for k in REQUIRED if k not in rec]
    errors += [f"unknown field {k}" for k in rec if k not in REQUIRED + OPTIONAL]
    if errors:
        return errors
    if type(rec["schema_version"]) is not int or rec["schema_version"] != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version {rec['schema_version']!r}")
    for k in ("sample_id", "source"):
        if not isinstance(rec[k], str) or not rec[k]:
            errors.append(f"{k} must be a nonempty string")
    if rec["split"] not in SPLITS:
        errors.append(f"split {rec['split']!r} not in {SPLITS}")
    if rec["label_mode"] not in MODES:
        return errors + [f"label_mode {rec['label_mode']!r} not in {MODES}"]
    tokens = rec["tokens"]
    if not isinstance(tokens, list) or not tokens:
        return errors + ["tokens must be a nonempty list"]
    if any(not isinstance(t, str) or not t.strip() for t in tokens):
        errors.append("tokens must be nonempty, non-whitespace strings")
    punct_vocab = label_sets["B"] if rec["label_mode"] == "B" else label_sets["A"]
    for field, vocab in (("punct_labels", punct_vocab), ("case_labels", label_sets["casing"])):
        labels = rec[field]
        if not isinstance(labels, list) or len(labels) != len(tokens):
            errors.append(f"{field} must have exactly one label per token")
        elif any(label not in vocab for label in labels):
            errors.append(f"{field} contains a label outside the {rec['label_mode']} vocabulary")
    if "metadata" in rec and not isinstance(rec["metadata"], dict):
        errors.append("metadata must be an object")
    return errors


def check_model_input(tokens: Iterable[str]) -> list[str]:
    """Member 1 input policy: model tokens are lowercased and carry no trailing target punctuation."""
    errors = []
    for t in tokens:
        if t != t.lower():
            errors.append(f"token {t!r} is not lowercased")
        if TARGET_MARK_END.search(t) or re.search(r"\s", t):
            errors.append(f"token {t!r} contains whitespace or a trailing target mark")
    return errors


def validate_file(path: str | os.PathLike[str], label_sets: Mapping[str, Any], split: str | None = None,
                  mode: str | None = None) -> tuple[int, list[tuple[int, str]]]:
    """(number of valid records, list of (line number, error)) for one release file."""
    seen, errors, ok = set(), [], 0
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                errors.append((n, "blank line"))
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append((n, f"invalid JSON: {e.msg}"))
                continue
            errs = validate_record(rec, label_sets)
            if not errs:
                errs += check_model_input(rec["tokens"])
                if rec["sample_id"] in seen:
                    errs.append(f"duplicate sample_id {rec['sample_id']}")
                if split and rec["split"] != split:
                    errs.append(f"split {rec['split']} != file split {split}")
                if mode and rec["label_mode"] != mode:
                    errs.append(f"label_mode {rec['label_mode']} != file mode {mode}")
                seen.add(rec.get("sample_id"))
            if errs:
                errors += [(n, e) for e in errs]
            else:
                ok += 1
    return ok, errors


def compare_views(view_a: Sequence[Mapping[str, Any]], view_b: Sequence[Mapping[str, Any]]) -> list[str]:
    """Differences between two views of the same records (lists of dicts): IDs, order, tokens, split."""
    if [r["sample_id"] for r in view_a] != [r["sample_id"] for r in view_b]:
        return ["sample IDs or their order differ"]
    return [f"{a['sample_id']}: tokens or split differ" for a, b in zip(view_a, view_b, strict=False)
            if a["tokens"] != b["tokens"] or a["split"] != b["split"]]
