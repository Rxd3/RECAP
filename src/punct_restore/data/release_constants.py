"""Constants shared by the release modules (owned by Member 1): file names, status sets, JSON style, source citation."""

RELEASE_FILES = ("train", "validation", "test", "test_amb")
REJECT = {"no_tokens", "validation_error", "inverted_mark_folded", "question_inside_word"}
# Unresolved for every mode: source, normalization and cross-check problems not covered by the approved rules.
UNRESOLVED_COMMON = {"possible_speaker_label", "kept_bracket_or_paren", "asterisk_text", "uncertain_words",
                     "low_comma_video", "tts_source", "reconstruction_mismatch"}
# Unresolved punctuation positions (canonical A, B and C share them because A is projected from B).
UNRESOLVED_TOKEN = {"mark_cluster", "abbrev_boundary_ambiguous", "single_initial_ambiguous", "final_not_terminal"}
# Unresolved for mode C only: transcripts whose casing is unusable as a whole (not covered by D4).
UNRESOLVED_CASE = {"case_source_no_capitals", "case_source_all_caps"}
JSON = dict(ensure_ascii=False, sort_keys=True, separators=(",", ":"))
SOURCE_REFERENCE = ("Liu, X. Y. and Beigi, H. (2024). Spontaneous Informal Speech Dataset for Punctuation "
                    "Restoration. arXiv:2409.11241. https://storage.googleapis.com/sponspeech/sponspeech.tar.gz")
