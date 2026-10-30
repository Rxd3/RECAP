"""Tests for punct_restore.data.gold_noise (the pure gold-noise rules) on invented texts: no real data."""

from __future__ import annotations

import copy
import re
from typing import Any

import pytest

from punct_restore.data.gold_noise import (
    AUTO_CAPTION_VIDEOS,
    build_inventories,
    is_cue_residue,
    rules_for,
    speaker_label_colons,
    speaker_label_spans,
    validate_record,
)

TOKEN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
MARKS = {",": "COMMA", ".": "PERIOD", "?": "QUESTION", "!": "EXCLAMATION", ":": "COLON", ";": "SEMICOLON"}
CUE = "Oh, I thought that was great. 506 00:18:45,000 -- 00:18:44,999 Oh, no. I just went home."


def make_record(text: str, *, original: str | None = None, group: str = "vidA", sid: str = "s:1") -> dict[str, Any]:
    """A mode B record built like the release: tokens are the words of ``text``, the label is the mark after."""
    found = list(TOKEN.finditer(text))
    return {
        "sample_id": sid, "label_mode": "B", "schema_version": 1, "source": "sponspeech", "split": "validation",
        "tokens": [m.group().lower() for m in found],
        "punct_labels": [MARKS.get(text[m.end():m.end() + 1], "O") for m in found],
        "case_labels": ["LOWER"] * len(found),
        "metadata": {"group_id": group, "normalized_text": text, "original_text": original or text,
                     "offsets": [[m.start(), m.end()] for m in found]},
    }


def names(text: str, inventory: frozenset[str] = frozenset()) -> list[str]:
    return [text[start:end] for start, end, _colon in speaker_label_spans(text, inventory)]


# --- speaker_label_spans: what is a label ---------------------------------------------------------------------------

def test_span_triple_is_start_name_end_and_colon_index() -> None:
    assert speaker_label_spans("Joe McNeely: hi") == [(0, 11, 11)]


@pytest.mark.parametrize(("text", "expected"), [
    ("Joe McNeely: hi", ["Joe McNeely"]),                      # Mc surname, two parts
    ("Ryan Katz-Rosene: yes", ["Ryan Katz-Rosene"]),           # hyphenated surname
    ("felix91gr: lol", ["felix91gr"]),                         # handle: letters plus digits
    ("-Deist: We have got a doctor.", ["Deist"]),              # turn dash glued to the name
    ("- Deist: We have got a doctor.", ["Deist"]),             # turn dash and a space
    ("somebody else - Paul: Just one", ["Paul"]),              # turn dash inside the text
    ("O'Brien: hello", ["O'Brien"]),
    ("A: hello", ["A"]),                                       # single capital
    ("Aaron Ross Powell: hello", ["Aaron Ross Powell"]),       # three parts is the maximum
    ("He left. Jeff: ok", ["Jeff"]),                           # after a sentence end
    ('He said "go." Jeff: ok', ["Jeff"]),                      # after a sentence end and a closing quote
    ("Why? Jeff: ok", ["Jeff"]),
    ("Wow! Jeff: ok", ["Jeff"]),
    ('"Sadhguru: ok', ["Sadhguru"]),                           # opening quote at the start
    ("Audience Member: Hello, my name", ["Audience Member"]),
    ("Edward Snowden: So there are a lot", ["Edward Snowden"]),
    ("Isha Kid: Namaskaram Sadhguru. Can you", ["Isha Kid"]),  # a name after the colon is no title phrase
    ("Q: why? A: because", ["Q", "A"]),
    ("for sure. David F: I would at least", ["David F"]),      # a first name and an initial
    ("David M: Next we have Greg", ["David M"]),
    ("Audience member 1: Hi. Thanks, guys.", ["Audience member 1"]),   # a role and its number
    ("Questioner 2: Sadhguru", ["Questioner 2"]),
])
def test_label_positions_and_name_shapes(text: str, expected: list[str]) -> None:
    assert names(text) == expected


@pytest.mark.parametrize(("text", "expected"), [
    ("SPEAKER 1: Something", ["SPEAKER 1"]),
    ("it was cut off-- SPEAKER 2: Things", ["SPEAKER 2"]),     # an explicit speaker number is a label anywhere
    ("Speaker 3: Hello", ["Speaker 3"]),
    ("STEPHAN KINSELLA: Yeah", ["STEPHAN KINSELLA"]),          # the ALL-CAPS labels the preprocessing knows
    ("OK. SURMA: yes", ["SURMA"]),
    ("McARDLE: You talked about", ["McARDLE"]),                # an Mc prefix before ALL-CAPS letters
    ("give him a big hand! WESLEY McGREW: Thank you.", ["WESLEY McGREW"]),
    ("ONE: Washing dishes", []),                               # spoken heading, not a label
    ("US: it", []),                                            # a two-letter ALL-CAPS word is no name
])
def test_speaker_numbers_and_caps_names(text: str, expected: list[str]) -> None:
    assert names(text) == expected


def test_a_leading_filler_word_is_not_part_of_the_name() -> None:
    text = "So Henriette Danel: it is"
    assert names(text) == ["Henriette Danel"]
    assert text[speaker_label_spans(text)[0][0]:].startswith("Henriette")


# --- speaker_label_spans: what is not a label -----------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "NerdCon: Stories",                        # camel-case brand, then a title word
    "Hank, NerdCon: Stories is",
    "Key 1: the",
    "Key 4: keep food",
    "Step 1: We admitted",
    "at 9 Sam went on",
    "In 1 Thessalonians 2: 19 it says",        # Bible-style reference
    "1 Thessalonians 2: verse",
    "The Power Paradox: How We Gain",          # a title and its subtitle
    "New York: Oxford University Press",
    "Note: this is important",                 # words the preprocessing never treats as labels
    "Question: why",
    "Two: you set unrealistic standards",      # spoken list markers
    "Re: the mail",
    "KOL361: x",                               # file-code shape: digits, no lowercase
    "9:30 is late",
    "21:12 Trevor Burrus: Probably, yeah.",    # a transcript time is not a label position
    "00:01:24 Dan Ikenson: So the President",
    "",
])
def test_non_labels_do_not_match(text: str) -> None:
    assert names(text) == []


@pytest.mark.parametrize("text", ["Yeah: right.", "Oh: no", "Okay: fine", "Or: maybe", "Then: we left", "Ok: x"])
def test_a_filler_word_before_a_colon_is_not_a_label(text: str) -> None:
    assert names(text) == []
    assert rules_for(make_record(text), frozenset()) == []


def test_a_known_name_after_a_filler_word_and_comma_still_counts() -> None:
    assert names("Yeah, Jeff: x", frozenset({"Jeff"})) == ["Jeff"]
    record = make_record("Yeah, Jeff: x")
    assert speaker_label_colons(record, frozenset({"Jeff"})) == 1


@pytest.mark.parametrize("text", [
    "Star Wars: A New Hope",                   # a one-letter capital word opens the title phrase
    "Star Wars: A New Hope is long",
])
def test_a_title_phrase_that_starts_with_a_one_letter_word_is_not_a_label(text: str) -> None:
    assert names(text) == []


def test_a_one_letter_word_and_a_word_that_end_a_sentence_are_not_a_title() -> None:
    assert names("Isha Kid: A Good. Next") == ["Isha Kid"]


@pytest.mark.parametrize("text", [
    "We read Joe: and left",                   # capitalised word before a colon, but mid-sentence
    "Yeah, Jeff: it was fine",                 # after ', ': needs the inventory
    "I like Plan A: B",
    "Wow. This pulled Jeff: up.",
])
def test_weak_positions_need_the_inventory(text: str) -> None:
    assert names(text) == []
    assert names(text, frozenset({"Bob"})) == []


@pytest.mark.parametrize("text", [
    "robot Pong: kit vector",                  # mid-sentence: not one of the four label positions
    "We read Joe: and left",
    "Wow. This pulled Jeff: up.",
    "21:12 Trevor Burrus: Probably, yeah.",    # after a transcript time
    "00:01:24 Dan Ikenson: So the President",
])
def test_an_inventory_name_outside_the_four_positions_is_not_a_label(text: str) -> None:
    inventory = frozenset({"Pong", "Joe", "Jeff", "Trevor Burrus", "Dan Ikenson"})
    assert names(text, inventory) == []


# --- speaker_label_spans: the inventory ------------------------------------------------------------------------------

def test_inventory_names_match_after_a_comma_but_not_in_mid_sentence() -> None:
    text = "Yeah, Jeff: it was Shea: no"
    assert names(text, frozenset({"Jeff", "Shea"})) == ["Jeff"]      # Shea is mid-sentence
    assert names(text) == []
    assert names(text, frozenset({"Jeff"})) == ["Jeff"]
    assert names(text, frozenset({"Shea"})) == []


def test_inventory_picks_the_known_name_out_of_a_capitalised_run() -> None:
    assert names("Probably Jeff: that was a year ago") == ["Probably Jeff"]    # shape and position only
    assert names("Probably Jeff: that was a year ago", frozenset({"Jeff"})) == ["Jeff"]
    assert names("Well, Drew Linsalata: the", frozenset({"Drew Linsalata"})) == ["Drew Linsalata"]
    assert names("Well, because Drew Linsalata: the", frozenset({"Drew Linsalata"})) == []    # mid-sentence


def test_inventory_covers_caps_initials_and_handles() -> None:
    inventory = frozenset({"STEPHAN KINSELLA", "T", "felix91gr"})
    assert names("within the binary, STEPHAN KINSELLA: Yeah", inventory) == ["STEPHAN KINSELLA"]
    assert names("please come, T: so", inventory) == ["T"]
    assert names("that I see, felix91gr: mean", inventory) == ["felix91gr"]


def test_a_recurring_label_is_kept_even_before_a_title_phrase() -> None:
    text = "Joe McNeely: Chris Depineda said so"
    assert names(text) == []                                    # unknown name + title phrase: conservative
    assert names(text, frozenset({"Joe McNeely"})) == ["Joe McNeely"]


def test_the_inventory_argument_is_not_modified() -> None:
    inventory = frozenset({"Jeff"})
    speaker_label_spans("Yeah, Jeff: it", inventory)
    assert inventory == frozenset({"Jeff"})


# --- build_inventories -----------------------------------------------------------------------------------------------

def test_inventory_is_the_names_that_recur_in_three_records_of_a_video() -> None:
    records = [make_record("Jeff: a. Bob: b", group="v1", sid=f"s:{i}") for i in range(3)]
    records += [make_record("Jeff: one", group="v2", sid="s:9"), make_record("Jeff: hi", group="v2", sid="s:10")]
    assert build_inventories(records) == {"v1": frozenset({"Jeff", "Bob"})}
    assert build_inventories(records, min_recur=2) == {"v1": frozenset({"Jeff", "Bob"}),
                                                       "v2": frozenset({"Jeff"})}


def test_a_name_counts_once_per_record() -> None:
    records = [make_record("Jeff: a. Jeff: b. Jeff: c.", sid="s:1"), make_record("Jeff: d", sid="s:2")]
    assert build_inventories(records) == {}


def test_inventory_reads_the_original_text_too() -> None:
    """The preprocessing removes a recurring label from the normalized text: the original still shows it."""
    records = [make_record("hello there", original="Jeff: hello there", sid=f"s:{i}") for i in range(3)]
    assert build_inventories(records) == {"vidA": frozenset({"Jeff"})}


def test_inventory_ignores_weak_positions_and_non_labels() -> None:
    records = [make_record("Yeah, Jeff: it. Note: x", sid=f"s:{i}") for i in range(4)]
    assert build_inventories(records) == {}


def test_inventory_ignores_a_name_after_a_transcript_time_or_in_mid_sentence() -> None:
    records = [make_record("21:12 Trevor Burrus: yes. We saw Shea: no", sid=f"s:{i}") for i in range(4)]
    assert build_inventories(records) == {}


# --- speaker_label_colons --------------------------------------------------------------------------------------------

def test_counts_colon_labels_that_sit_on_a_speaker_label() -> None:
    record = make_record("Joe McNeely: hi there. Note: here and Plan A: B")
    assert record["punct_labels"].count("COLON") == 3
    assert speaker_label_colons(record, frozenset()) == 1


def test_counts_an_inventory_label_after_a_comma_but_not_in_mid_sentence() -> None:
    record = make_record("Wow. This pulled Jeff: up. We saw, Shea: no")
    assert speaker_label_colons(record, frozenset()) == 0
    assert speaker_label_colons(record, frozenset({"Jeff", "Shea"})) == 1       # only Shea, after ', '


def test_a_name_after_a_transcript_time_is_not_counted() -> None:
    record = make_record("21:12 Trevor Burrus: Probably, yeah.")
    assert record["punct_labels"].count("COLON") == 2                 # the clock colon and the one after the name
    assert speaker_label_colons(record, frozenset({"Trevor Burrus"})) == 0
    assert rules_for(record, frozenset({"Trevor Burrus"})) == []


def test_a_speaker_number_label_counts_on_its_number_token() -> None:
    record = make_record("SPEAKER 1: Something. SPEAKER 2: Else")
    assert speaker_label_colons(record, frozenset()) == 2


def test_no_colon_gold_label_no_count() -> None:
    record = make_record("Jeff: hi")
    record["punct_labels"][0] = "O"
    assert speaker_label_colons(record, frozenset()) == 0


# --- is_cue_residue --------------------------------------------------------------------------------------------------

def test_real_cue_residue_is_found_in_the_original_text() -> None:
    record = make_record("Oh, I thought that was great. 506 -- Oh, no. I just went home.", original=CUE)
    assert is_cue_residue(record) is True


@pytest.mark.parametrize("original", [
    "We need to imagine. 41 0:03:21.990 -- 0:03:27.710 To give you a metaphor",    # one-digit hour, dot
    "A thing. 506 00:18:45,000 --> 00:18:44,999 Oh, no.",                           # the SRT arrow
    "Well. 61 00:03:10,8 -- 00:03:13,66 so",                                        # short milliseconds
    "Then. 839 00:29:34,400 -- 00:29:34 Yeah.",                                     # no milliseconds on the end
])
def test_cue_shapes(original: str) -> None:
    assert is_cue_residue(make_record("x", original=original)) is True


@pytest.mark.parametrize("text", [
    "in 2004 - but",
    "2004-'05",
    "by 1990 - or",
    "2023- just",
    "2000 -- was it 2016",
])
def test_years_before_a_dash_are_not_cues(text: str) -> None:
    assert is_cue_residue(make_record(text)) is False


@pytest.mark.parametrize("original", [
    "at [00:01:02] we start",                    # a spoken timecode, no range
    "from 10:30 -- sharp",                       # a clock time
    "the 506 -- Oh, no",                         # the number and dash left in the normalized text only
    "00:18:45,000 -- 00:18:44,999 without a block number",
    "see 12:00:01 and 12:00:05",
])
def test_other_numbers_and_timecodes_are_not_cues(original: str) -> None:
    assert is_cue_residue(make_record("x", original=original)) is False


def test_cue_rule_needs_the_original_text() -> None:
    record = make_record("506 -- Oh, no")
    del record["metadata"]["original_text"]
    assert is_cue_residue(record) is False


# --- rules_for / auto-caption ----------------------------------------------------------------------------------------

def test_auto_caption_videos() -> None:
    assert AUTO_CAPTION_VIDEOS == frozenset({"732fql_jhJQ", "JNxcznsrRb8"})
    assert rules_for(make_record("hello there", group="732fql_jhJQ"), frozenset()) == ["auto_caption_video"]
    assert rules_for(make_record("hello there", group="JNxcznsrRb8"), frozenset()) == ["auto_caption_video"]
    assert rules_for(make_record("hello there", group="other"), frozenset()) == []


def test_clean_record_has_no_rule() -> None:
    assert rules_for(make_record("It is fine, really. Go on."), frozenset()) == []


def test_speaker_label_rule() -> None:
    assert rules_for(make_record("Joe McNeely: hi"), frozenset()) == ["speaker_label"]
    assert rules_for(make_record("Yeah, Jeff: it"), frozenset({"Jeff"})) == ["speaker_label"]
    assert rules_for(make_record("Yeah, Jeff: it"), frozenset()) == []
    assert rules_for(make_record("Yeah we saw Jeff: it"), frozenset({"Jeff"})) == []     # mid-sentence


def test_cue_rule_in_rules_for() -> None:
    record = make_record("Oh, I thought that was great. 506 -- Oh, no.", original=CUE)
    assert rules_for(record, frozenset()) == ["cue_number"]


def test_a_record_with_several_rules_lists_them_in_a_fixed_order() -> None:
    text = "Pat: hi. 506 -- Oh, no."
    record = make_record(text, original="Pat: hi. 506 00:18:45,000 -- 00:18:44,999 Oh, no.", group="732fql_jhJQ")
    assert rules_for(record, frozenset()) == ["speaker_label", "cue_number", "auto_caption_video"]


# --- validate_record -------------------------------------------------------------------------------------------------

def test_a_well_formed_record_passes() -> None:
    validate_record(make_record("Joe McNeely: hi"))
    record = make_record("Joe McNeely: hi")
    del record["metadata"]["original_text"]
    validate_record(record)                                    # original_text is optional


def _drop(*path: str) -> Any:
    def mutate(record: dict[str, Any]) -> None:
        target = record
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]
    return mutate


def _set(path: tuple[str, ...], value: Any) -> Any:
    def mutate(record: dict[str, Any]) -> None:
        target = record
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return mutate


@pytest.mark.parametrize(("mutate", "fragment"), [
    (_drop("sample_id"), "sample_id"),
    (_set(("sample_id",), ""), "sample_id"),
    (_drop("metadata"), "metadata"),
    (_set(("metadata",), []), "metadata"),
    (_set(("metadata", "group_id"), 7), "group_id"),
    (_drop("metadata", "normalized_text"), "normalized_text"),
    (_set(("metadata", "original_text"), 3), "original_text"),
    (_drop("metadata", "offsets"), "offsets"),
    (_set(("metadata", "offsets"), [[0, 1]]), "length"),
    (_set(("metadata", "offsets"), [[0, "x"]] * 3), "offsets"),
    (_set(("punct_labels",), "OOO"), "punct_labels"),
    (_set(("punct_labels",), ["O"]), "length"),
    (_set(("tokens",), "abc"), "tokens"),
])
def test_malformed_records_are_refused_with_one_line(mutate: Any, fragment: str) -> None:
    record = make_record("Joe McNeely: hi")
    broken = copy.deepcopy(record)
    mutate(broken)
    with pytest.raises(ValueError, match=fragment) as caught:
        validate_record(broken)
    assert "\n" not in str(caught.value)


def test_a_record_that_is_not_an_object_is_refused() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        validate_record(["not", "a", "record"])
