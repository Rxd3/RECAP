"""Punctuation and casing labels (owned by Member 1).

Each word gets the punctuation of the boundary AFTER it and the casing of the word itself. Labels are derived from
the gap text and the original-case word surface produced by preprocessing.tokenize(), before lowercasing.
Mode C is the standard 4 punctuation labels x 3 casing labels (12); it is composed at load time with join_labels(),
never stored as an independent array.

Approved policies (docs/DECISIONS.md, policy m1-v2):
  * D3: mode B maps '--' and standalone spaced hyphens to DASH; ellipsis is O inside and PERIOD at the end.
  * D5: canonical mode A (and the punctuation part of mode C) is the approved B -> A projection in
    configs/label_sets.yaml. The "native" SponSpeech four-class view (! ; -> PERIOD, : -> COMMA, dash -> O) is
    kept separately for comparison with the original benchmark only.
  * D4: casing precedence in case_label(); mixed-case approximations are masked per token position.
Mark clusters (several marks in one gap) are not covered by the approval: the strongest mark is used and the record
is flagged for review.
"""
import itertools
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..file_io import read_yaml as read_yaml  # old import path kept: the one YAML reader is in file_io
from ..file_io import require_object
from ..paths import REPO_ROOT
from .preprocessing import ABBREVIATIONS, TITLES, Token

LABEL_SETS_PATH = os.path.join(REPO_ROOT, "configs", "label_sets.yaml")


# What label_sets.yaml must hold for load_label_sets to build the label sets (see file_io.require_object).
_LABEL_SCHEMA = {
    "punctuation": {"standard": list[str], "extended": list[str]},
    "casing": list[str],
    "modes": {"C": {"separator": str, "labels": list[str]}},
    "extended_to_standard": {},
}


def read_label_config(path: str | os.PathLike[str] = LABEL_SETS_PATH) -> dict[str, Any]:
    """The raw label configuration mapping; a ValueError names the file (not UTF-8, not YAML, not a mapping)."""
    return require_object(os.path.basename(path), read_yaml(path), {})


def label_sets_from_config(config: Mapping[str, Any], file_name: str) -> dict[str, Any]:
    """The label sets of an already parsed label configuration, after checking its 4/8/12 structure.

    `file_name` is the name of the file `config` was read from; it starts the ValueError that
    `file_io.require_object` raises for a missing key or a value of the wrong kind, and the one for a vocabulary
    that is not the agreed 4/8/12 structure (`<file name>: does not have the agreed 4/8/12 structure`).
    """
    cfg = require_object(file_name, config, _LABEL_SCHEMA)
    std, ext, case = cfg["punctuation"]["standard"], cfg["punctuation"]["extended"], cfg["casing"]
    separator, joint_listed = cfg["modes"]["C"]["separator"], cfg["modes"]["C"]["labels"]
    projection = cfg["extended_to_standard"]
    joint = [f"{p}{separator}{c}" for p, c in itertools.product(std, case)]
    if (len(std), len(ext), len(case)) != (4, 8, 3) or ext[:4] != std or joint_listed != joint:
        raise ValueError(f"{file_name}: does not have the agreed 4/8/12 structure")
    return {"A": list(std), "B": list(ext), "C": joint, "casing": list(case), "separator": separator,
            "projection": projection}


def load_label_sets(path: str | os.PathLike[str] = LABEL_SETS_PATH) -> dict[str, Any]:
    """Read configs/label_sets.yaml (the vocabulary source of truth) and check its 4/8/12 structure.

    Raises ValueError("<file name>: <what is wrong>") for a file that is not valid UTF-8 or YAML, or that parses but
    lacks a key or has a value of the wrong type (`file_io.require_object` words it).
    """
    return label_sets_from_config(read_label_config(path), os.path.basename(path))


# --- Punctuation ---------------------------------------------------------------------------------------------------

ELLIPSIS = re.compile(r"\.{2,}|\.(?: \.){2,}")
# A dash is '--' or a hyphen with whitespace before it. A hyphen glued to the word ('plant- based', 'I just-')
# is a cut-off word, not a dash (draft D3 rule).
DASH = re.compile(r"--|\s-(?![\w-])")
CUTOFF_HYPHEN = re.compile(r"^-(?!-)")

EXTENDED_MARKS = {"?": "QUESTION", "!": "EXCLAMATION", ".": "PERIOD", ";": "SEMICOLON", ":": "COLON", ",": "COMMA"}
EXTENDED_PRECEDENCE = ("QUESTION", "EXCLAMATION", "PERIOD", "SEMICOLON", "COLON", "COMMA", "DASH")
# Native SponSpeech four-class view (audit policy v0.3), kept for benchmark comparison only. It differs from the
# canonical projection only at dash-only boundaries (O here, COMMA in canonical mode A).
STANDARD_MARKS = {"?": "QUESTION", ".": "PERIOD", "!": "PERIOD", ";": "PERIOD", ",": "COMMA", ":": "COMMA"}
STANDARD_PRECEDENCE = ("QUESTION", "PERIOD", "COMMA")


def gap_label(gap: str, is_last: bool, scheme: str) -> tuple[str, list[str]]:
    """Label for one gap under scheme 'extended' (mode B) or 'native' (SponSpeech four-class view), plus flags."""
    marks, precedence = (
        (EXTENDED_MARKS, EXTENDED_PRECEDENCE) if scheme == "extended" else (STANDARD_MARKS, STANDARD_PRECEDENCE)
    )
    flags = []
    if ELLIPSIS.search(gap):
        flags.append("ellipsis")
        gap = ELLIPSIS.sub(" ", gap)
        # Ellipsis-only normalization does not resolve mixed boundaries such as
        # '...?' or '?...'. Keep the fallback label, but exclude these cases via
        # the existing unresolved-cluster policy before the terminal early return.
        if any(c in EXTENDED_MARKS for c in gap) or DASH.search(gap):
            flags.append("mark_cluster")
        if is_last:
            return "PERIOD", flags          # every source utterance ends with a terminal mark by construction
    found = {marks[c] for c in gap if c in marks}
    has_dash = bool(DASH.search(gap))
    if has_dash:
        flags.append("dash")
        if scheme == "extended":
            found.add("DASH")
    if CUTOFF_HYPHEN.search(gap):
        flags.append("cutoff_hyphen")
    if len(found) > 1 and "mark_cluster" not in flags:
        flags.append("mark_cluster")
    if scheme == "native":
        flags += [
            f for mark, f in (("!", "exclam_mapped"), (";", "semicolon_mapped"), (":", "colon_mapped")) if mark in gap
        ]
    return next((p for p in precedence if p in found), "O"), flags


def punctuation_labels(tokens: Sequence[Token], scheme: str) -> tuple[list[str], list[list[str]]]:
    """One punctuation label per token, plus per-token flags."""
    labels, token_flags = [], []
    n = len(tokens)
    for i, tok in enumerate(tokens):
        last = i == n - 1
        label, flags = gap_label(tok.gap, last, scheme)
        nxt = tokens[i + 1].text if not last else ""
        if tok.abbrev:
            if last:
                label = "PERIOD" if label == "O" else label      # the absorbed dot also ends the utterance
                flags.append("abbrev_at_utt_end")
            elif label == "O" and nxt[:1].isupper() and tok.text.lower().rstrip(".") not in TITLES:
                flags.append("abbrev_boundary_ambiguous")         # '... in the U.S. The next ...'
        elif (label == "PERIOD" and not last and re.fullmatch(r"[A-Z]", tok.text) and tok.gap.startswith(". ")
              and re.match(r"[A-Z][a-z]", nxt)):
            flags.append("single_initial_ambiguous")              # 'Michael J. Fox' vs 'plan B. So'
        if last and label not in ("PERIOD", "QUESTION", "EXCLAMATION"):
            flags.append("final_not_terminal")
        labels.append(label)
        token_flags.append(flags)
    return labels, token_flags


def project_extended_to_standard(labels: Iterable[str], label_sets: Mapping[str, Any],
                                 allow_unapproved: bool = False) -> list[str]:
    """Map B labels to canonical A with the configs/label_sets.yaml mapping (refused if it is not approved)."""
    proj = label_sets["projection"]
    if not proj.get("approved") and not allow_unapproved:
        raise ValueError("extended_to_standard mapping is not approved (decision D5)")
    return [proj["mapping"][label] for label in labels]


# --- Casing --------------------------------------------------------------------------------------------------------

def case_category(surface: str) -> str:
    """Descriptive casing category of an original-case word, kept for audits and statistics."""
    letters = [c for c in surface if c.isalpha()]
    if not letters:
        return "no_cased_letters"                      # '70', '2017', '9:30'
    if all(c.islower() for c in letters):
        return "lower"
    if len(letters) == 1:
        return "one_letter_upper"                      # 'I', 'A', 'B12'
    if all(c.isupper() for c in letters):
        return "all_upper"                             # 'FDA', 'U.S.', 'CD36'
    if letters[0].isupper() and all(c.islower() for c in letters[1:]):
        return "first_upper"
    return "mixed"                                     # 'NetApp', 'iPhone', 'McDonald'


def case_label(surface: str) -> tuple[str, bool]:
    """(case label, approximated) under the approved D4 precedence, applied in this exact order:
    1. no cased letters (numbers) -> LOWER;  2. a single uppercase letter ('I', 'A') -> FIRST_UPPER;
    3. several cased letters, all uppercase -> ALL_UPPER;  4. otherwise the first cased letter decides:
    uppercase -> FIRST_UPPER, lowercase -> LOWER. Step 4 is an approximation when the remaining letters are not
    all lowercase ('iPhone', 'NetApp'); such positions are masked from casing/joint metrics.
    """
    letters = [c for c in surface if c.isalpha()]
    if not letters:
        return "LOWER", False
    if len(letters) == 1 and letters[0].isupper():
        return "FIRST_UPPER", False
    if len(letters) > 1 and all(c.isupper() for c in letters):
        return "ALL_UPPER", False
    approximated = not all(c.islower() for c in letters[1:])
    return ("FIRST_UPPER" if letters[0].isupper() else "LOWER"), approximated


def casing_labels(tokens: Sequence[Token], text: str) -> tuple[list[str], list[str], list[int], set[str]]:
    """Case labels, per-token categories, masked positions (approximations) and utterance-level casing flags."""
    labels, cats, masked = [], [], []
    for i, t in enumerate(tokens):
        label, approximated = case_label(t.text)
        labels.append(label)
        cats.append(case_category(t.text))
        if approximated:
            masked.append(i)
    flags = set()
    letters = re.sub(r"[^A-Za-z]", "", text)
    if letters and letters.islower():
        flags.add("case_source_no_capitals")           # transcriber did not use casing at all (not covered by D4)
    if len(letters) > 15 and letters.isupper():
        flags.add("case_source_all_caps")
    return labels, cats, masked, flags


def join_labels(punct_labels: Iterable[str], case_labels: Iterable[str], separator: str = "|") -> list[str]:
    """Mode C target at load time: 'PERIOD|FIRST_UPPER'. Never stored as a separate gold array."""
    return [f"{p}{separator}{c}" for p, c in zip(punct_labels, case_labels, strict=True)]


# --- Independent cross-check of standard punctuation ----------------------------------------------------------------

STANDARD_MARK_OF = {"O": "", "COMMA": ",", "PERIOD": ".", "QUESTION": "?"}
ALNUM = re.compile(r"[A-Za-z0-9]")


def _read_marks(text: str) -> list[tuple[str, str]]:
    """[(word, mark)] from a whitespace reading of the cleaned text (no word regex), standard policy."""
    text = re.sub(r"(?<=[\w\"')])(--+|\.{2,})(?=[\w\"'(])", r" \1 ", text)
    text = re.sub(r"(?<=[A-Za-z])([,;?!])(?=[A-Za-z])", r"\1 ", text)
    text = re.sub(r"([.,;?!])([\"'-])(?=[A-Za-z])", r"\1\2 ", text)
    text = " ".join(c if re.search(r"@|www\.|://", c) else re.sub(r"(?<=[a-z])\.(?=[A-Z][a-z])", ". ", c)
                    for c in text.split())
    words: list[str] = []
    trails: list[str] = []
    for ch in text.split():
        if (first_match := ALNUM.search(ch)) is None:
            if trails:
                trails[-1] += " " + ch
            continue
        first = first_match.start()
        lead = ch[:first]
        if lead.endswith(".") and ch[first].islower():
            lead, first = lead[:-1], first - 1
        if trails:
            trails[-1] += lead
        if first > 0 and ch[first - 1] in "$£€#@":
            first -= 1
        last = max(m.end() for m in ALNUM.finditer(ch))
        if ch[last:last + 1] == "%":
            last += 1
        words.append(ch[first:last])
        trails.append(ch[last:])
    out = []
    n = len(words)
    for i, (w, tr) in enumerate(zip(words, trails, strict=False)):
        low = w.lower()
        nxt = words[i + 1] if i + 1 < n else ""
        prev = words[i - 1] if i else ""
        abbr = (low in ABBREVIATIONS or re.fullmatch(r"(?:[a-z]\.)+[a-z]", low)
                or (low == "no" and re.match(r"\d", nxt)) or (low == "al" and prev.lower() == "et"))
        if abbr and tr.startswith(".") and not tr.startswith(".."):
            w, tr = w + ".", tr[1:]
        is_last = i == n - 1
        if re.search(r"\.\.|\. \. \.", tr):
            tr = re.sub(r"\.{2,}|\.(?: \.)+", " ", tr)
            if is_last:
                out.append((w.lower(), "."))
                continue
        found = {STANDARD_MARKS[c] for c in tr if c in STANDARD_MARKS}
        lab = next((p for p in STANDARD_PRECEDENCE if p in found), "O")
        if abbr and w.endswith(".") and is_last and lab == "O":
            lab = "PERIOD"
        out.append((w.lower(), STANDARD_MARK_OF[lab]))
    return out


def _signature(pairs: Iterable[tuple[str, str]]) -> str:
    return "".join(re.sub(r"[^a-z0-9]", "", w) + m for w, m in pairs)


def cross_check_standard(tokens: Sequence[Token], labels: Sequence[str], text: str) -> bool:
    """True if an independent reading of the text puts the same standard marks after the same letters/digits."""
    from_labels = _signature(
        (token.text.lower(), STANDARD_MARK_OF[label]) for token, label in zip(tokens, labels, strict=False))
    return from_labels == _signature(_read_marks(text))
