"""Source loading, cleaning and tokenization for Member 1's data preparation (owned by Member 1).

Implements SponSpeech, the project dataset (D1). Tokenization follows the approved D2 policy (docs/DECISIONS.md):
contractions, numbers, URLs, abbreviations and meaningful internal punctuation stay inside tokens; boundary
punctuation is moved to the gap after each word; one record per utterance.

Pipeline for one transcript:
    normalize(raw_text)  -> cleaned text plus a log of every removal
    tokenize(cleaned)    -> words with character offsets and the punctuation "gap" after each word
Labels are derived from the gaps and the original-case word surfaces in labeling.py, before any lowercasing.
"""
import collections
import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import chain, islice

# --- Source identity (SponSpeech; D1 approved) ------------------------------------------------------------------

SOURCE_ID = "sponspeech"
# Object generation of https://storage.googleapis.com/sponspeech/sponspeech.tar.gz (verified size 63,547,617,096 B).
SOURCE_REVISION = "gcs-generation-1716975130398953"
SOURCE_SPLITS = ("train", "dev", "test-amb", "test")
POLICY_VERSION = "m1-v2"               # D3 mixed-ellipsis flags and D6 normalized-input deduplication
UTT_ID_RE = re.compile(r"^([A-Za-z0-9_-]{11})-(\d+)$")


@dataclass(frozen=True)
class SourceUtterance:
    source_split: str      # SponSpeech directory name: train / dev / test-amb / test
    upstream_id: str       # file stem, e.g. '-1ER_f6AD4o-14'
    group_id: str          # YouTube video ID (talk/document-equivalent group)
    index: int
    text: str              # raw transcript, unchanged
    sha256: str            # of the raw file bytes


class SourceError(ValueError):
    """A raw file breaks the input expectations (encoding, single line, ID format, utt2spk agreement)."""


def _map_source_files(function, items):
    """Read small source files concurrently, preserving order with bounded pending work.

    Windows file-open latency dominates the 206k-file source check. Batches avoid
    queuing a future for every transcript and keep memory independent of file count.
    """
    pending = iter(items)
    first = list(islice(pending, 257))
    if len(first) <= 256:
        yield from map(function, first)
        return
    pending = chain(first, pending)
    with ThreadPoolExecutor(max_workers=16) as pool:
        while batch := list(islice(pending, 256)):
            yield from pool.map(function, batch)


def read_utt2spk(raw_dir, source_split):
    """{utterance id: video id} for one SponSpeech split. Lines are '<utt-id> <video-id>'."""
    mapping = {}
    with open(os.path.join(raw_dir, source_split, "utt2spk"), encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            parts = line.split()
            if len(parts) != 2:
                raise SourceError(f"{source_split}/utt2spk line {n}: expected 2 fields")
            mapping[parts[0]] = parts[1]
    return mapping


def read_source_split(raw_dir, source_split):
    """All utterances of one SponSpeech split, sorted by upstream ID, with input checks."""
    def read_one(item):
        utt_id, video_id = item
        with open(os.path.join(raw_dir, source_split, "text", utt_id + ".txt"), "rb") as f:
            raw = f.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            raise SourceError(f"{source_split}/{utt_id}: not UTF-8") from e
        m = UTT_ID_RE.match(utt_id)
        if not m or m.group(1) != video_id:
            raise SourceError(f"{source_split}/{utt_id}: ID does not match utt2spk video {video_id}")
        if not text.strip() or "\n" in text or "\r" in text:
            raise SourceError(f"{source_split}/{utt_id}: empty or multi-line transcript")
        return SourceUtterance(source_split, utt_id, video_id, int(m.group(2)), text, hashlib.sha256(raw).hexdigest())
    return list(_map_source_files(read_one, sorted(read_utt2spk(raw_dir, source_split).items())))


def verify_raw_manifest(manifest_path):
    """Paths (relative to the manifest) whose SHA-256 no longer matches. Empty list means raw data is unchanged."""
    base = os.path.dirname(os.path.abspath(manifest_path))
    def check_one(line):
        digest, rel = line.rstrip("\n").split("  ", 1)
        with open(os.path.join(base, rel), "rb") as fh:
            return rel if hashlib.sha256(fh.read()).hexdigest() != digest else None
    with open(manifest_path, encoding="utf-8") as f:
        return [bad for bad in _map_source_files(check_one, f) if bad is not None]


# --- Normalization: remove non-lexical material, log every removal ------------------------------------------------

NONSPEECH_TAG = re.compile(
    r"\b(music|laugh\w*|chuckl\w*|giggl\w*|applause|applaud\w*|clap\w*|cheer\w*|cough\w*|sigh\w*|sniff\w*|"
    r"breath\w*|gasp\w*|groan\w*|noise|sound\w*|silence|pause|static|beep\w*|bleep\w*|blip|whistl\w*|"
    r"background|intro|outro|jingle|upbeat|crying|scream\w*|yell\w*|humm\w*|inhales|exhales|gestur\w*|"
    r"mumbl\w*|clears? \w*\s?throat|stutter\w*|whisper\w*|inaudible|indiscernible|unintelligible|indistinct|"
    r"unclear|crosstalk|cross talk|overlapping|phonetic|silent|no audio|blank_audio)\b", re.I)
# Words commonly followed by ':' in speech; never treated as speaker labels.
NOT_SPEAKER_LABELS = {"Note", "Example", "Step", "Tip", "Warning", "Update", "Source", "Rule", "Lesson", "Chapter",
                      "Part", "Episode", "Number", "Option", "Fact", "Bonus", "Reminder", "Hint", "Question",
                      "Answer", "Yes", "No", "So", "Now", "And", "But", "Well", "Okay", "Here", "This", "That",
                      "All", "Because", "In", "The", "It", "If", "What", "When", "We", "You", "I"}
SPEAKER_MIN_RECUR = 3      # a non-CAPS label must recur this often in one video before it is removed

TIMECODE_TAG = re.compile(r"\[\s*\d{1,2}(?::\d{2}){1,3}(?:[.,]\d+)?\s*\]")
# hh:mm:ss with optional frames/fraction, optionally a range. Clock times such as 9:30 do not match.
TIMECODE_INLINE = re.compile(
    r"\b\d{1,2}:\d{2}:\d{2}(?::\d{2}|[.,]\d+)?(?:\s*-\s*\d{1,2}:\d{2}:\d{2}(?::\d{2}|[.,]\d+)?)?\b")
SPEAKER_N = re.compile(r"\b[Ss]peaker ?\d+\b:?")
CHEVRONS = re.compile(r">>")
TAG = re.compile(r"\[[^\[\]]{0,80}\]|\([^()]{0,40}\)")
BRACKET_NAME = re.compile(r"^[A-Z][A-Za-z]*(?: [A-Z][A-Za-z]*){0,2}:?$")
# A label sits at the start of the text, after a sentence end, or after a turn dash.
LABEL_POS = r"(?:(?<=^)|(?<=[.?!][\"'] )|(?<=[.?!] )|(?<=- ))"
# ALL-CAPS label: dots only in initials/titles (J.C., MR.), so 'OK. SURMA:' keeps 'OK.'; a leading filler
# ('OK JASON MILLER:') stays; spoken headings ('ONE: Washing dishes') are not labels.
CAPS_NAME_PART = r"(?:MRS?\.|MS\.|DR\.|(?:[A-Z]\.)+|[A-Z][A-Z'-]+)"
CAPS_NOT_LABEL = (r"(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|FIRST|SECOND|THIRD|PS|P\.S\.|NOTE|STEP|TIP|"
                  r"UPDATE|WARNING|EXAMPLE|QUESTION|ANSWER)(?= *:)")
CAPS_LABEL = re.compile(LABEL_POS + r"(?P<lead>(?:OK|YES|SO|AND|BUT|WELL) )?(?P<name>(?!" + CAPS_NOT_LABEL + r")"
                        + CAPS_NAME_PART + r"(?: " + CAPS_NAME_PART + r"){0,3}):(?= |$)")
TITLE_LABEL = re.compile(LABEL_POS + r"([A-Z][a-z]+(?: [A-Z][a-z]+){0,2}|[A-Z]):(?= )")
GLUED_LABEL = re.compile(r"(?<![\w:])([A-Z][a-z]+|[A-Z][A-Z.'-]*):(?=[A-Za-z])")   # 'ES:He', 'Ben:so'
NAME_THEN_TIMECODE = re.compile(r"^((?:[A-Z][a-z]*\.? ){0,2}[A-Z][a-z]+) \[\d{1,2}(?::\d{2}){1,3}\]")
UNCERTAIN_WORDS = re.compile(r"\[\?([^\[\]]*?)\?\]")       # '[? vitals ?]': transcriber unsure of a spoken word
TURN_DASH_START = re.compile(r"^-+ ?")
ASTERISK_TEXT = re.compile(r"\*[A-Za-z][^*]{0,40}\*")
INVERTED_FOLD = re.compile(r"(?:^|[\s\"(])[?!][A-Za-z]")   # upstream folded '¿'/'¡' into '?'/'!'
QUESTION_IN_WORD = re.compile(r"[A-Za-z]\?[a-z]")           # upstream replaced a letter by '?': 'Hebdom?konta'


def _squash(text):
    """Collapse whitespace and re-attach marks stranded by a removal ('word ,' -> 'word,')."""
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"(?<=\w) +(?=[.,?!;:](?:\s|$))", "", text)


def _strip_markers(text):
    """Timecodes and turn markers removed without logging; speaker labels are counted on this view."""
    for rx in (TIMECODE_TAG, TIMECODE_INLINE, SPEAKER_N, CHEVRONS):
        text = rx.sub(" ", text)
    return _squash(TURN_DASH_START.sub("", _squash(text)))


def build_speaker_inventory(texts, min_recur=SPEAKER_MIN_RECUR):
    """Speaker names that recur >= min_recur times among one video's transcripts."""
    labels, tc_names, caps, glued = (collections.Counter() for _ in range(4))
    for t in texts:
        m = NAME_THEN_TIMECODE.match(t)
        if m:
            tc_names[m.group(1)] += 1
        clean = _strip_markers(t)
        for m in TITLE_LABEL.finditer(clean):
            labels[m.group(1)] += 1
        for m in CAPS_LABEL.finditer(clean):
            caps[m.group("name")] += 1
        for m in GLUED_LABEL.finditer(clean):
            glued[m.group(1)] += 1
    keep = lambda c: {x for x, n in c.items() if n >= min_recur and x.split()[0] not in NOT_SPEAKER_LABELS}
    return {"labels": keep(labels), "timecode_names": keep(tc_names), "glued_ok": keep(labels + caps + glued)}


def normalize(text, speakers=None):
    """Return (cleaned_text, removals, flags). Spoken words are never removed; unclear tags are kept and flagged."""
    speakers = {"labels": set(), "timecode_names": set(), "glued_ok": set(), **(speakers or {})}
    ops, flags = [], set()

    def remove(pattern, rule, s, keep=None):
        def repl(m):
            if keep and keep(m):
                return m.group()
            ops.append((rule, m.group()))
            return " "
        return _squash(pattern.sub(repl, s))

    tc_name = NAME_THEN_TIMECODE.match(text)

    def unsure(m):
        ops.append(("uncertain_word_marker", m.group()))
        return " " + m.group(1).replace("?", " ") + " "
    t = _squash(UNCERTAIN_WORDS.sub(unsure, text))
    if t != _squash(text):
        flags.add("uncertain_words")
    t = remove(TIMECODE_TAG, "timecode_tag", t)
    t = remove(TIMECODE_INLINE, "timecode_inline", t)
    t = remove(SPEAKER_N, "speaker_number", t)
    t = remove(CHEVRONS, "chevrons", t)

    def tag_keep(m):
        inner = m.group()[1:-1].strip()
        if NONSPEECH_TAG.search(inner) or inner in ("", "...", "?", "!!!"):
            return False
        if m.group().startswith("[") and BRACKET_NAME.match(inner):
            return False                               # [Student], [Instructor], [HANK:]
        flags.add("kept_bracket_or_paren")
        return True
    t = remove(TAG, "nonspeech_or_name_tag", t, keep=tag_keep)

    def caps_label(m):
        ops.append(("speaker_label_caps", m.group("name") + ":"))
        return " " + (m.group("lead") or "")
    t = _squash(CAPS_LABEL.sub(caps_label, t))
    t = remove(GLUED_LABEL, "speaker_label_glued", t, keep=lambda m: m.group(1) not in speakers["glued_ok"])
    t = remove(TITLE_LABEL, "speaker_label_recurring", t, keep=lambda m: m.group(1) not in speakers["labels"])
    if TITLE_LABEL.search(t):
        flags.add("possible_speaker_label")            # label-shaped but not recurring in this video: kept
    t = remove(TURN_DASH_START, "turn_dash_start", t)
    if tc_name and tc_name.group(1) in speakers["timecode_names"] and t.startswith(tc_name.group(1) + " "):
        flags.add("possible_speaker_label")            # 'Dr. Kalu [00:20:58] ...': flagged, never removed
    if ASTERISK_TEXT.search(t):
        flags.add("asterisk_text")
    if INVERTED_FOLD.search(t):
        flags.add("inverted_mark_folded")
    if QUESTION_IN_WORD.search(t):
        flags.add("question_inside_word")
    return t, ops, flags


# --- Tokenization ------------------------------------------------------------------------------------------------

ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "rev", "sgt", "capt", "lt", "vs", "etc",
                 "e.g", "i.e", "cf", "approx", "inc", "ltd", "vol", "a.m", "p.m", "u.s", "u.k", "u.n", "d.c", "ph.d"}
TITLES = {"mr", "mrs", "ms", "dr", "prof", "st", "rev", "sgt", "capt", "lt"}   # always precede a name
TLD = r"(?:com|org|net|io|edu|gov|co|uk|tv|ai|fm|me|info|us|ca|au|de)"
# Negative sign (D3): a '-' at the start of a word, directly before a digit, belongs to the number ('-12 dB').
NEGATIVE = r"(?:(?<![^\s(\[\"'])-(?=\d))?"
# Ordered word types: the first alternative that matches wins. Joiners ('.', ',', ':') are allowed only where the
# type needs them, so missing spaces such as 'work,but' or 'today?James' split into two words.
WORD = re.compile("|".join([
    r"(?:https?://|www\.)[^\s\"]+?(?=[.,?!;:)\"]*(?:\s|$))",                  # URL
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+",                   # email
    r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\." + TLD + r"\b(?:/[\w/-]*)?",        # bare domain, optional path
    r"[Pp]h\.[Dd](?![A-Za-z])",                                                # Ph.D
    r"[Pp]p?\.\d+(?:-\d+)?",                                                   # page reference p.300, pp.12-14
    r"(?<![^\s\"(])\.[a-z][A-Za-z0-9_]*",                                      # .com .wav: leading dot is part of the word
    r"(?:[A-Za-z]\.)+[A-Za-z](?![A-Za-z])",                                    # initialism: U.S, e.g, a.m
    NEGATIVE + r"(?:[$£€]|PS)?\d+(?:-\d+)?(?:[.,:]\d+)+(?:-\d+(?:[.,:]\d+)*[a-z]?)?%?",   # 28.5 10,000 9:30 -2.5 PS1.14 (£)
    r"[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+",                   # Rack::Attack
    r"[a-z][A-Za-z0-9_]*:[A-Za-z][A-Za-z0-9_]*",                               # re:Invent, method:save
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[a-z_][A-Za-z0-9_]*)+",                       # build.rs, object.assign, e.coli
    r"(?:[$£€#@]|" + NEGATIVE + r")?[A-Za-z0-9]+(?:(?:['\-_&+/]|\*+|(?<=\d)\.(?=\d))[A-Za-z0-9]+)*%?",   # don't age-related -12 50%
]))
INITIALISM = re.compile(r"(?:[a-z]\.)+[a-z]")


@dataclass
class Token:
    text: str        # surface form in the cleaned text, original casing kept
    start: int       # character offsets in the cleaned text
    end: int
    gap: str         # text between this word and the next word (or the end): source of the punctuation label
    abbrev: bool     # a trailing abbreviation '.' was absorbed into the word


def _is_abbreviation(word, prev, nxt):
    low = word.lower()
    if low in ABBREVIATIONS or INITIALISM.fullmatch(low):
        return True
    if low == "no":
        return bool(re.match(r"\d", nxt))       # 'No. 5' but not '... is no.'
    if low == "al":
        return prev.lower() == "et"             # 'et al.'
    return False


def tokenize(text):
    """Words of the cleaned text. Quotes, brackets, dashes and marks stay in the gaps, never in tokens."""
    matches = list(WORD.finditer(text))
    out = []
    for i, m in enumerate(matches):
        start, end = m.start(), m.end()
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        word, gap = m.group(), text[end:next_start]
        prev = matches[i - 1].group() if i else ""
        nxt = matches[i + 1].group() if i + 1 < len(matches) else ""
        absorbed = gap.startswith(".") and not gap.startswith("..") and _is_abbreviation(word, prev, nxt)
        if absorbed:
            word, end, gap = word + ".", end + 1, gap[1:]
        out.append(Token(word, start, end, gap, absorbed))
    return out


def model_token(surface):
    """Model input form of a word: Python str.lower(), as proposed in DATA_CONTRACT.md (not casefold)."""
    return surface.lower()


# --- Per-video context (computed from all of a video's transcripts) ------------------------------------------------

LOW_COMMA_PER_100W = 1.0       # below this comma rate (and >= LOW_COMMA_MIN_WORDS) a video looks ASR-transcribed
LOW_COMMA_MIN_WORDS = 300
TTS_TITLE = re.compile(r"\|\s*Wikipedia audio article\s*$", re.I)


def video_context(texts, title=None):
    """Speaker inventory and source-quality flags for one video."""
    words = sum(len(t.split()) for t in texts)
    commas = sum(t.count(",") for t in texts)
    flags = set()
    if words >= LOW_COMMA_MIN_WORDS and 100 * commas / words < LOW_COMMA_PER_100W:
        flags.add("low_comma_video")
    if title and TTS_TITLE.search(title):
        flags.add("tts_source")
    return {"speakers": build_speaker_inventory(texts), "flags": flags}
