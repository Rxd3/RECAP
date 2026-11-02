"""The label audit of 100 primary-validation records, with 95% Wilson intervals, from the filled audit sheet.

``label_audit.json`` is computed from ``gold_noise/label_audit_sheet.csv``. Marks are ``, . ? ! : ; -`` in our
rendered labels. Label precision is (our marks - wrong) / our marks. Label recall is correct / (correct + missed); a
different mark at the right place counts only against precision, so recall is slightly optimistic. Casing accuracy
is 1 - casing errors / scored words (the words of our rendering; bracketed, masked words are not scored). The
Wilson intervals treat marks (words) as independent; marks inside one utterance are not, so the intervals are
somewhat narrow. Rows 51-100 were checked by hand; rows 1-50 are an AI agent's counts only. The file reports both
halves and all 100 rows, each pooled and per stratum (the sample over-represents rare marks), and rows 1-50 again
with the marks that our normalization removes by design (speaker names before a colon, leading turn dashes, a
trailing hyphen) not counted as missed: all 8 marks missed in rows 1-50 are such removals, as their notes say (a
row whose note does not say so is refused).
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

Z = 1.959964
ROWS = 100
HAND_FROM = 51  # rows 51-100 were checked by hand, rows 1-50 are the agent's counts
SHEET_PATH = "gold_noise/label_audit_sheet.csv"  # under the scores root
# how the notes of rows 1-50 name a removal that our normalization makes by design: a speaker name before a colon
# ("speaker tag removed", "NAME: removed"), a leading turn dash ("speaker-turn dash omitted"), a trailing hyphen
BY_DESIGN_NOTE = re.compile(r"\bremoved\b|speaker-turn dash|trailing hyphen", re.IGNORECASE)
STRATA = ("rare", "random")
COLUMNS = ("row", "stratum", "our_marks", "wrong_marks", "missed_marks", "casing_errors", "notes", "checked_by")
# the helper sheets that the committed file names as its inputs (the sheet in the repository holds all 100 rows)
SHEETS = {"hand": "RECAP_label_audit_sheet_hand.csv", "agent": "RECAP_label_audit_sheet_agent.csv",
          "all": "RECAP_label_audit_sheet_merged.csv", "by_design": "RECAP_label_audit_sheet_agent_bydesign.csv"}
WHAT = "label audit of 100 primary-validation records (seed 2026; 60 random, 40 rare-mark stratum)"
METHOD = "AI-assisted counts; rows 51-100 checked by hand by Abdullah Gamal; rows 1-50 agent counts only"
DEFINITIONS = {"label_precision": "correct / our marks", "label_recall": "correct / (correct + missed)",
               "casing_accuracy": "1 - casing errors / scored words", "interval": "95% Wilson"}


@dataclass(frozen=True)
class AuditRow:
    """One audited record: the counts of its row in the sheet."""

    number: int
    stratum: str
    rendered: str
    our_marks: int
    wrong_marks: int
    missed_marks: int
    casing_errors: int
    notes: str
    checked_by: str


def wilson(k: int, n: int) -> dict[str, Any]:
    """The share ``k / n`` with its 95% Wilson score interval (six decimals; all null when ``n`` is 0)."""
    if n == 0:
        return {"k": k, "n": 0, "p": None, "low": None, "high": None}
    share = k / n
    den = 1 + Z * Z / n
    mid = (share + Z * Z / (2 * n)) / den
    half = Z * math.sqrt(share * (1 - share) / n + Z * Z / (4 * n * n)) / den
    return {"k": k, "n": n, "p": round(share, 6), "low": round(mid - half, 6), "high": round(mid + half, 6)}


def words_scored(rendered: str) -> int:
    """The words of our rendering whose casing is scored: not bracketed (masked) and with a letter."""
    return sum(1 for word in rendered.split() if not word.startswith("[") and re.search(r"[A-Za-z]", word))


def _count(raw: dict[str, str], key: str, where: str) -> int:
    text = (raw.get(key) or "").strip()
    if not text:
        raise ValueError(f"{where}: {key} is empty")
    try:
        value = int(text)
    except ValueError:
        raise ValueError(f"{where}: {key} is not a whole number") from None
    if value < 0:
        raise ValueError(f"{where}: {key} is negative")
    return value


def _row(raw: dict[str, str], name: str) -> AuditRow:
    where = f"{name}: row {(raw.get('row') or '').strip()}"
    stratum = (raw.get("stratum") or "").strip()
    if stratum not in STRATA:
        raise ValueError(f"{where}: stratum is not {' or '.join(STRATA)}")
    counts = {key: _count(raw, key, where) for key in ("row", "our_marks", "wrong_marks", "missed_marks",
                                                      "casing_errors")}
    return AuditRow(counts["row"], stratum, raw.get("rendered") or "", counts["our_marks"], counts["wrong_marks"],
                    counts["missed_marks"], counts["casing_errors"], (raw.get("notes") or "").strip(),
                    (raw.get("checked_by") or "").strip())


def _by_first_word(item: dict[Any, Any]) -> dict[str, str]:
    """A sheet row keyed by the first word of each column name (the long names carry hints), plus ``rendered``."""
    row = {key.split(" ")[0]: value for key, value in item.items() if key}
    row["rendered"] = next((value for key, value in item.items() if key and key.startswith("our_labels_rendered")), "")
    return row


def _raw_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            names = [(field or "").split(" ")[0] for field in reader.fieldnames or []]
            rows = [_by_first_word(item) for item in reader]
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name}: not valid UTF-8") from exc
    except csv.Error as exc:
        raise ValueError(f"{path.name}: not a valid CSV file") from exc
    if missing := next((column for column in COLUMNS if column not in names), None):
        raise ValueError(f"{path.name}: missing column {missing}")
    return rows


def read_sheet(path: Path) -> list[AuditRow]:
    """The 100 audited rows of the sheet, checked.

    Raises:
        OSError: If the file cannot be read.
        ValueError: One line that starts with the file name: not valid UTF-8 or CSV, a missing column, not exactly
            the rows 1 to 100, a count that is empty, negative or not a whole number, an unknown stratum, or a row
            in 51-100 without a checker.
    """
    return _check_rows([_row(raw, path.name) for raw in _raw_rows(path)], path.name)


def _check_rows(rows: list[AuditRow], name: str) -> list[AuditRow]:
    if len(rows) != ROWS:
        raise ValueError(f"{name}: has {len(rows)} rows, not {ROWS}")
    if [row.number for row in rows] != list(range(1, ROWS + 1)):
        raise ValueError(f"{name}: the rows are not numbered 1 to {ROWS}")
    if not all(row.checked_by for row in rows[HAND_FROM - 1:]):
        raise ValueError(f"{name}: rows {HAND_FROM}-{ROWS} must all have checked_by")
    return rows


def _stats(rows: Sequence[AuditRow]) -> dict[str, Any]:
    our, wrong = sum(r.our_marks for r in rows), sum(r.wrong_marks for r in rows)
    missed, casing = sum(r.missed_marks for r in rows), sum(r.casing_errors for r in rows)
    words = sum(words_scored(r.rendered) for r in rows)
    correct = our - wrong
    return {"rows": len(rows), "our_marks": our, "wrong_marks": wrong, "missed_marks": missed,
            "casing_errors": casing, "scored_words": words,
            "label_precision": wilson(correct, our), "label_recall": wilson(correct, correct + missed),
            "casing_accuracy": wilson(words - casing, words)}


def _section(sheet: str, rows: Sequence[AuditRow]) -> dict[str, Any]:
    notes = [row.notes for row in rows]
    return {"sheet": sheet, "rows_total": len(rows), "rows_filled": len(rows),
            "rows_unchecked": [str(row.number) for row in rows if not row.checked_by],
            "checkers": sorted({row.checked_by for row in rows if row.checked_by}),
            "pooled": _stats(rows),
            "by_stratum": {s: _stats([row for row in rows if row.stratum == s]) for s in STRATA},
            "notes": {"speaker_label": sum("speaker" in n.lower() for n in notes),
                      "check": sum(n.upper().startswith("CHECK") for n in notes),
                      "any": sum(bool(n) for n in notes)}}


def _without_by_design_removals(agent: Sequence[AuditRow], name: str) -> list[AuditRow]:
    """``agent`` with no mark counted as missed; a missed mark whose note names no by-design removal is refused."""
    for row in agent:
        if row.missed_marks and not BY_DESIGN_NOTE.search(row.notes):
            raise ValueError(f"{name}: row {row.number}: missed marks without a by-design note")
    return [replace(row, missed_marks=0) for row in agent]


def label_audit_text(scores_root: Path) -> str:
    """The text of label_audit.json (LF line ends, two-space indent, one trailing newline).

    Reads ``gold_noise/label_audit_sheet.csv`` under ``scores_root``.

    Raises:
        OSError: If the sheet cannot be read.
        ValueError: As ``read_sheet``, or ``<sheet>: row N: missed marks without a by-design note`` for a row in
            1-50 with missed marks whose note does not name a removal that our normalization makes by design.
    """
    sheet = scores_root / SHEET_PATH
    rows = read_sheet(sheet)
    agent, hand = rows[:HAND_FROM - 1], rows[HAND_FROM - 1:]
    by_design = _without_by_design_removals(agent, sheet.name)
    document = {"what": WHAT, "method": METHOD, "definitions": DEFINITIONS,
                "hand_checked_rows_51_100": _section(SHEETS["hand"], hand),
                "agent_only_rows_1_50": _section(SHEETS["agent"], agent),
                "all_100": _section(SHEETS["all"], rows),
                "agent_only_rows_1_50_by_design_removals_not_missed": _section(SHEETS["by_design"], by_design)}
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"
