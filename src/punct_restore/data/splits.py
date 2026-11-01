"""Sample identity, split assignment and duplicate/leakage audits (owned by Member 1).

The official SponSpeech splits are kept (DATA_CONTRACT: respect benchmark split semantics):
    train -> train,  dev -> validation,  test -> test,  test-amb -> test (written to its own file)
`test-amb` has no contract split value of its own. Its records use split "test", keep their identity in the sample
ID and metadata.upstream_id, and are released in a separate file, so they are never merged into `test` silently.
Whether test-amb stays a second test set is part of decision D6.

The official splits share videos and near-identical text (an upstream property). A video-disjoint evaluation
subset is therefore offered as separate ID lists; it does not replace the official assignment (D6).
"""
import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

SPLIT_OF = {"train": "train", "dev": "validation", "test": "test", "test-amb": "test"}
FILE_OF = {"train": "train", "dev": "validation", "test": "test", "test-amb": "test_amb"}
# Video-disjoint subsets: validation is compared with train; both test sets with train + validation.
DISJOINT_REFERENCE = {"dev": ("train",), "test": ("train", "dev"), "test-amb": ("train", "dev")}
NGRAM = 6
NEAR_DUP = 0.5        # share of an utterance's word 6-grams found in the reference


def sample_id(source: str, source_split: str, upstream_id: str) -> str:
    """Stable, release-unique ID. SponSpeech reuses upstream IDs across splits, so the split is part of the ID."""
    return f"{source}:{source_split}:{upstream_id}"


def ngram_keys(tokens: Sequence[str], n: int = NGRAM) -> set[int]:
    """Hashed word n-grams; blake2b keeps results independent of Python's string-hash seed."""
    return {int.from_bytes(hashlib.blake2b(" ".join(tokens[i:i + n]).encode(), digest_size=8).digest(), "big")
            for i in range(len(tokens) - n + 1)}


class Reference:
    """Groups, original text, normalized token tuples and 6-grams of reference records."""

    def __init__(self, records: Iterable[Mapping[str, Any]]) -> None:
        self.groups, self.texts, self.model_inputs, self.grams = set(), set(), set(), set()
        for r in records:
            self.groups.add(r["group_id"])
            self.texts.add(r["original_text"])
            self.model_inputs.add(tuple(r["tokens"]))
            self.grams |= ngram_keys(r["tokens"])


def overlap_reason(rec: Mapping[str, Any], ref: Reference) -> str | None:
    """Why a record overlaps the reference (checked in this order), or None."""
    if rec["group_id"] in ref.groups:
        return "shared_video"
    if rec["original_text"] in ref.texts:
        return "identical_text_other_video"
    if tuple(rec["tokens"]) in ref.model_inputs:
        return "identical_model_input_other_video"
    keys = ngram_keys(rec["tokens"])
    if keys and sum(g in ref.grams for g in keys) / len(keys) >= NEAR_DUP:
        return "near_duplicate_other_video"
    return None


def overlap_counts(eval_recs: Sequence[Mapping[str, Any]], ref_recs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Independently re-check videos, raw text, model-input tuples and six-grams."""
    ref_groups = {r["group_id"] for r in ref_recs}
    ref_texts = {r["original_text"] for r in ref_recs}
    ref_inputs = {tuple(r["tokens"]) for r in ref_recs}
    ref_grams = set()
    for r in ref_recs:
        t = r["tokens"]
        ref_grams.update(" ".join(t[i:i + NGRAM]) for i in range(len(t) - NGRAM + 1))
    near = 0
    for r in eval_recs:
        t = r["tokens"]
        grams = {" ".join(t[i:i + NGRAM]) for i in range(len(t) - NGRAM + 1)}
        if grams and sum(g in ref_grams for g in grams) / len(grams) >= NEAR_DUP:
            near += 1
    return {"eval_records": len(eval_recs),
            "shared_videos": len({r["group_id"] for r in eval_recs} & ref_groups),
            "records_in_shared_videos": sum(r["group_id"] in ref_groups for r in eval_recs),
            "identical_texts": sum(r["original_text"] in ref_texts for r in eval_recs),
            "identical_model_inputs": sum(tuple(r["tokens"]) in ref_inputs for r in eval_recs),
            "near_duplicates": near}
