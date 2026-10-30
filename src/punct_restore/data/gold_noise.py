"""Pure rules that flag gold records carrying non-speech material (grader findings data-1 and data-4).

Three kinds of record are flagged:

* ``speaker_label``: a leftover speaker label ('Joe McNeely:', '-Deist:', 'felix91gr:', 'SPEAKER 1:') sits in the
  text. The preprocessing missed these: its name pattern is plain [A-Z][a-z]+ words, the turn dash is stripped only
  after label detection, and a label after ', ' is not a label position for it. A name counts at exactly four
  positions (a numbered role such as 'SPEAKER 2:' is the one name that counts anywhere):

  1. the utterance start;
  2. after a sentence end ('.', '?' or '!', an optional quote and a space);
  3. after ', ': only when the name is in the video's inventory;
  4. after a turn dash ('-Deist:', '- Deist:').

  Positions 1, 2 and 4 accept a name by its shape. Mid-sentence names and names after a transcript time
  ('21:12 Trevor Burrus:') are not labels, inventory or not.

  Known limit: matching by shape alone also flags some single-word headings and references at those positions
  ('Disclaimer:', 'Notes:', 'Abingdon: Routledge'), so ``speaker_label_colons`` is an upper estimate. A colon that
  follows a title is skipped only when a title phrase of capitalised words comes after it ('The Power Paradox: How We
  Gain', 'Star Wars: A New Hope'); a subtitle with a lowercase word ('Order of the Phoenix') is not excluded.
* ``cue_number``: SRT cue residue ('506 00:18:45,000 -- 00:18:44,999') that the normalization turned into a bare
  block number and a dash. Years before a dash ('in 2004 - but') are not cues.
* ``auto_caption_video``: the record comes from a video whose captions are machine-made and unpunctuated.

Text fields read, and why:

* Speaker labels are looked for in ``metadata.normalized_text``. ``metadata.offsets`` index that text (checked on the
  release: every token matches ``normalized_text[start:end]``, not ``original_text``), so a label found there is a
  label in the model input and the COLON gold label sits on its last token.
* The per-video inventory of speaker names reads both texts: the normalization already removed the recurring labels
  from ``normalized_text``, but they are still in ``original_text`` and prove the name is a speaker of the video.
* Cue residue is read from ``metadata.original_text``: the timestamps exist only there. The normalized text keeps just
  '506 --', which is indistinguishable from '2000 -- was it 2016'.

All functions are pure: no I/O, no mutation of their arguments. Records must have passed ``validate_record``.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from typing import Any, cast

from .preprocessing import (
    CAPS_NAME_PART,
    CAPS_NOT_LABEL,
    NOT_SPEAKER_LABELS,
    SPEAKER_MIN_RECUR,
)

RULES = ("speaker_label", "cue_number", "auto_caption_video")
AUTO_CAPTION_VIDEOS = frozenset({"732fql_jhJQ", "JNxcznsrRb8"})

# Discourse fillers: not part of a name that follows them ('So Henriette Danel:') and never a name themselves.
FILLER_WORDS = ("And", "But", "So", "Well", "Oh", "Okay", "OK", "Yeah", "Yes", "Now", "Or", "Then")
# Words that open a spoken list or heading, never a speaker: the preprocessing's list, the fillers (capitalised like
# the name they are compared with, so 'OK' is 'Ok'), numbers and 'Re'.
NOT_LABEL_WORDS = frozenset(NOT_SPEAKER_LABELS | {word.capitalize() for word in FILLER_WORDS} | {
    "Re", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
    "First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Tenth"})

# --- Label shapes -----------------------------------------------------------------------------------------------

QUOTE = "[\"'“”‘’]"
# One capitalised name part. Mc/Mac/O' prefixes and hyphenated surnames stay whole: McNeely, O'Brien, Katz-Rosene.
TITLE_PART = r"(?:Mc|Mac|O')?[A-Z][a-z]+(?:-[A-Z][a-z]+)*"
# A role and its number: 'SPEAKER 1', 'Audience member 2'. Other '<word> <number>:' ('Key 1:', 'Step 2:') are headings.
NUMBERED_ROLE = (r"(?i:speaker|audience member|questioner|participant|guest|caller|student|panelist|interviewer|host)"
                 r" ?\d+")
CAPS_PART = "(?:Mc|Mac)?" + CAPS_NAME_PART                                # McARDLE, WESLEY McGREW
NAME_SHAPES = [
    NUMBERED_ROLE,
    r"(?=[A-Za-z0-9_]*\d)(?=[A-Za-z0-9_]*[a-z])[A-Za-z][A-Za-z0-9_]{2,}",  # handle: letters, digits, a lowercase letter
    "(?!" + CAPS_NOT_LABEL + ")" + CAPS_PART + "(?: " + CAPS_PART + "){0,3}",   # STEPHAN KINSELLA
    TITLE_PART + "(?: " + TITLE_PART + "){0,2}",                          # Joe McNeely: one to three parts
    TITLE_PART + " [A-Z]",                                                # David M: a first name and an initial
    "[A-Z]",                                                              # single capital: A, T
]
# A filler before the name is not part of it, as in the preprocessing's CAPS lead.
FILLER = "(?:" + "|".join(FILLER_WORDS) + ") "
# The lookbehinds keep a match out of the middle of a word ('NerdCon', 'off-Jeff') but allow a glued turn dash.
LABEL = re.compile(r"(?<![\w'’])(?<!\w-)(?P<lead>" + FILLER + r")?(?P<name>" + "|".join(NAME_SHAPES)
                   + r"):(?= |$)")
ROLE_NUMBER = re.compile(NUMBERED_ROLE)
# Label positions 1, 2 and 4: the start of the text, after a sentence end (plus a quote), after a turn dash.
STRONG_BEFORE = re.compile(r"(?:^|[.?!]" + QUOTE + r"? |(?:^|\s)-{1,2} ?)" + QUOTE + r"?$")
# Label position 3: after ', '. The name needs the inventory there.
COMMA_BEFORE = re.compile(", " + QUOTE + r"?$")
# 'Global Diasporas: An Introduction', 'NerdCon: Stories ...': a colon followed by a title phrase is a title.
TITLE_PHRASE = re.compile(r" [A-Z][a-z]*(?: [A-Z][a-z]+)+\b(?![.?!,])")      # '[a-z]*': 'Star Wars: A New Hope'

# SRT cue residue: block number, start time, '-->' (or the '--' it was turned into), end time. A year has no timestamps.
TIME = r"\d{1,2}:\d{2}:\d{2}"
CUE = re.compile(r"(?<![\d:.,])\d{1,6}\s+" + TIME + r"[.,]\d{1,3}\s*-{1,2}>?\s*" + TIME + r"(?:[.,]\d{1,3})?(?![\d:])")


def _shape_is_label(name: str) -> bool:
    """A name with no evidence but its shape: not a heading word and not a two-letter ALL-CAPS word (US, OK, AI)."""
    if name.split(" ")[0].capitalize() in NOT_LABEL_WORDS:
        return False
    return not (len(name) == 2 and name.isupper())


def _known_start(name: str, start: int, inventory: frozenset[str]) -> int | None:
    """Start of the longest tail of ``name`` that is a known speaker ('Probably Jeff' -> 'Jeff'), else None."""
    words = name.split(" ")
    for i in range(len(words)):
        tail = " ".join(words[i:])
        if tail in inventory:
            return start + len(name) - len(tail)
    return None


def _label_start(text: str, match: re.Match[str], inventory: frozenset[str]) -> int | None:
    """Where the label name starts, or None when this colon does not follow a speaker label."""
    name, start = match.group("name"), match.start("name")
    if ROLE_NUMBER.fullmatch(name):
        return start                                      # a role with a number is a label anywhere
    before = text[:match.start("lead") if match.group("lead") else start]
    known = _known_start(name, start, inventory)
    if STRONG_BEFORE.search(before):                      # positions 1, 2 and 4: a known name, else the shape
        if known is not None:
            return known
        title = TITLE_PHRASE.match(text, match.end("name") + 1)
        return start if _shape_is_label(name) and not title else None
    if COMMA_BEFORE.search(before):
        return known                                      # position 3: only a name of the video's inventory
    return None                                           # mid-sentence, after a transcript time: no label


def speaker_label_spans(text: str, inventory: frozenset[str] = frozenset()) -> list[tuple[int, int, int]]:
    """Speaker labels in ``text`` as ``(start, end_of_name, colon_index)``; the colon follows the name directly.

    A name counts at four positions only: the text start, after '.?!' (a quote and a space), after ', ' and after a
    turn dash. After ', ' it counts when ``inventory``, the speaker names that recur in the same video, has it; at
    the other three its shape is enough. A capitalised word before a colon is never a label on that alone.
    """
    spans = []
    for match in LABEL.finditer(text):
        start = _label_start(text, match, inventory)
        if start is not None:
            end = match.end("name")
            spans.append((start, end, end))
    return spans


# --- Per-video inventory ----------------------------------------------------------------------------------------

def record_label_names(record: Mapping[str, Any]) -> frozenset[str]:
    """Names shown as labels at strong positions in the record's normalized or original text."""
    meta = record["metadata"]
    texts = (meta["normalized_text"], meta.get("original_text") or "")
    return frozenset(text[start:end] for text in texts for start, end, _colon in speaker_label_spans(text))


def build_inventories(records: Iterable[Mapping[str, Any]],
                      min_recur: int = SPEAKER_MIN_RECUR) -> dict[str, frozenset[str]]:
    """Per video (``metadata.group_id``), the names that are labels in at least ``min_recur`` of its records."""
    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        counts[record["metadata"]["group_id"]].update(record_label_names(record))
    inventories = {video: frozenset(name for name, n in counter.items() if n >= min_recur)
                   for video, counter in counts.items()}
    return {video: names for video, names in inventories.items() if names}


# --- The rules --------------------------------------------------------------------------------------------------

def speaker_label_colons(record: Mapping[str, Any], inventory: frozenset[str]) -> int:
    """COLON gold labels on a speaker label: the token whose span ends at the name end, right before the ':'."""
    meta = record["metadata"]
    ends = {end for _start, end, _colon in speaker_label_spans(meta["normalized_text"], inventory)}
    return sum(1 for label, (_start, end) in zip(record["punct_labels"], meta["offsets"], strict=True)
               if label == "COLON" and end in ends)


def is_cue_residue(record: Mapping[str, Any]) -> bool:
    """True when the original text holds an SRT cue ('<n> hh:mm:ss,mmm --> hh:mm:ss'); years before a dash do not."""
    original = record["metadata"].get("original_text")
    return isinstance(original, str) and CUE.search(original) is not None


def rules_for(record: Mapping[str, Any], inventory: frozenset[str]) -> list[str]:
    """The rules that flag ``record``, a subset of ``RULES`` in that order (empty when the record is clean)."""
    meta = record["metadata"]
    flags = (bool(speaker_label_spans(meta["normalized_text"], inventory)), is_cue_residue(record),
             meta["group_id"] in AUTO_CAPTION_VIDEOS)
    return [rule for rule, flagged in zip(RULES, flags, strict=True) if flagged]


# --- Input check ------------------------------------------------------------------------------------------------

def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _is_list_of(value: object, check: Callable[[Any], bool]) -> bool:
    return isinstance(value, list) and all(check(item) for item in value)


def _is_span(value: object) -> bool:
    return (isinstance(value, list) and len(value) == 2
            and all(isinstance(n, int) and not isinstance(n, bool) for n in value))


def validate_record(record: object) -> None:
    """Raise a one-line ValueError unless ``record`` has what the rules read (sample_id, labels, offsets, texts)."""
    _require(isinstance(record, dict), "record is not a JSON object")
    record = cast("dict[str, Any]", record)
    _require(isinstance(record.get("sample_id"), str) and bool(record["sample_id"]),
             "record without a string sample_id")
    meta = record.get("metadata")
    _require(isinstance(meta, dict), "record without metadata")
    meta = cast("dict[str, Any]", meta)
    for key in ("group_id", "normalized_text"):
        _require(isinstance(meta.get(key), str), f"record without a string metadata.{key}")
    _require(isinstance(meta.get("original_text", ""), str), "metadata.original_text is not a string")
    tokens, labels, offsets = record.get("tokens"), record.get("punct_labels"), meta.get("offsets")
    _require(_is_list_of(tokens, lambda t: isinstance(t, str)), "record without a list of string tokens")
    _require(_is_list_of(labels, lambda x: isinstance(x, str)), "record without a list of string punct_labels")
    _require(_is_list_of(offsets, _is_span), "record without a list of [start, end] metadata.offsets")
    _require(len(tokens) == len(labels) == len(offsets), "tokens, punct_labels and metadata.offsets differ in length")
