"""Build and load a versioned SponSpeech data release (owned by Member 1). Policies: docs/DECISIONS.md D1-D6.

Each transcript is processed once; every view shares the same sample IDs, order, tokens and splits:
    mode_A/      canonical standard punctuation (approved B -> A projection, D5) + case_labels
    mode_B/      extended punctuation (D3) + case_labels
    mode_C/      canonical standard punctuation + case_labels; 12-class target = join_labels() at load time
    benchmark_native/mode_A/   native SponSpeech four-class labels, for original-benchmark comparison only
Files per view: train, validation, test, test_amb (test-amb keeps split "test" and is never merged with test).

Records whose labels depend on cases the approved policies do not cover ("unresolved") stay in the views and are
listed with reasons; they are left out of the default training and primary evaluation ID lists (D3). Casing
approximations are masked per token position in masks/case_joint/ (D4). Primary evaluation uses the frozen
leakage-controlled subsets (D6); the full official splits are for supplementary reporting.
"""
import collections
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

from .. import prediction_io
from ..paths import REPO_ROOT
from ..progress import Log, discard_log
from . import labeling, preprocessing, splits, validation
from .release_checks import ReleaseReadback, verify_release
from .release_constants import REJECT, UNRESOLVED_COMMON, UNRESOLVED_TOKEN
from .release_manifest import (
    ManifestInputs,
    build_manifest,
    build_provenance,
    copy_license,
    with_file_inventory,
    write_manifest,
)
from .release_outputs import (
    write_audit,
    write_checksums,
    write_id_lists,
    write_json_reports,
    write_leakage_controlled,
    write_masks,
    write_release_notes,
    write_split_map,
    write_views,
)
from .release_stats import StatsInputs, build_stats, pairwise_overlap

__all__ = ["ReleaseSources", "build_release", "build_release_from", "load_records", "read_ids"]

# old import path kept (restructure spec §2)
from ..hashing import sha256_file as sha256_file
from .release_constants import JSON as JSON
from .release_constants import RELEASE_FILES as RELEASE_FILES
from .release_constants import SOURCE_REFERENCE as SOURCE_REFERENCE
from .release_constants import UNRESOLVED_CASE as UNRESOLVED_CASE
from .release_manifest import describe_files as describe_files
from .release_outputs import write_lines as write_lines
from .release_stats import label_distribution as label_distribution
from .release_stats import length_stats as length_stats
from .release_stats import mode_labels as mode_labels


def code_fingerprint() -> str:
    """SHA-256 over this package's source files, recorded in the manifest."""
    here = os.path.join(REPO_ROOT, "src", "punct_restore", "data")
    digest = hashlib.sha256()
    for name in sorted(os.listdir(here)):
        if name.endswith(".py"):
            with open(os.path.join(here, name), "rb") as source:
                digest.update(name.encode() + b"\0" + source.read())
    return digest.hexdigest()


def _flag_positions(token_flags: Sequence[Sequence[str]],
                    case_masked: list[int]) -> collections.defaultdict[str, list[int]]:
    """Token positions per flag; masked casing positions are listed under case_mixed_approximation."""
    positions: collections.defaultdict[str, list[int]] = collections.defaultdict(list)
    for position, flags in enumerate(token_flags):
        for flag in flags:
            positions[flag].append(position)
    if case_masked:
        positions["case_mixed_approximation"] = case_masked
    return positions


def _view_metadata(utt: preprocessing.SourceUtterance, words: Sequence[preprocessing.Token],
                   cleaned: str) -> dict[str, Any]:
    """The metadata every view of one utterance shares (without `annotation_view`)."""
    return {"upstream_id": f"{utt.source_split}/{utt.upstream_id}", "source_split": utt.source_split,
            "group_id": utt.group_id, "original_text": utt.text, "normalized_text": cleaned,
            "offsets": [[word.start, word.end] for word in words], "source_revision": preprocessing.SOURCE_REVISION,
            "policy_version": preprocessing.POLICY_VERSION, "condition": "clean"}


def process(utt: preprocessing.SourceUtterance, context: preprocessing.VideoContext,
            label_sets: Mapping[str, Any]) -> dict[str, Any]:
    """One transcript -> views, status, reasons, masks and audit information."""
    cleaned, removals, utterance_flags = preprocessing.normalize(utt.text, context["speakers"])
    utterance_flags |= context["flags"]
    words = preprocessing.tokenize(cleaned)
    base = {"sample_id": splits.sample_id(preprocessing.SOURCE_ID, utt.source_split, utt.upstream_id),
            "source_split": utt.source_split, "file": splits.FILE_OF[utt.source_split], "group_id": utt.group_id,
            "original_text": utt.text, "removals": removals, "views": {}, "tokens": []}
    if not words:
        return {**base, "status": "rejected", "reasons": sorted(utterance_flags | {"no_tokens"}),
                "validation_errors": []}
    extended, extended_flags = labeling.punctuation_labels(words, "extended")
    native, _ = labeling.punctuation_labels(words, "native")
    canonical = labeling.project_extended_to_standard(extended, label_sets)
    case, case_categories, case_masked, case_flags = labeling.casing_labels(words, cleaned)
    if not labeling.cross_check_standard(words, native, cleaned):
        utterance_flags.add("reconstruction_mismatch")
    if any(word.text.startswith("PS") and word.text[2:3].isdigit() for word in words):
        utterance_flags.add("currency_folded")         # '£' became 'PS' upstream (information only)
    if any(word.text.startswith("-") for word in words):
        utterance_flags.add("leading_minus")           # kept as a negative sign (D3); dialogue dashes may look alike
    tokens = [preprocessing.model_token(word.text) for word in words]
    meta = _view_metadata(utt, words, cleaned)

    def make_view(mode: str, punct: list[str], annotation: str) -> dict[str, Any]:
        return {"schema_version": validation.SCHEMA_VERSION, "sample_id": base["sample_id"],
                "source": preprocessing.SOURCE_ID, "split": splits.SPLIT_OF[utt.source_split], "label_mode": mode,
                "tokens": tokens, "punct_labels": punct, "case_labels": case,
                "metadata": {**meta, "annotation_view": annotation}}
    views = {"A": make_view("A", canonical, "canonical"), "B": make_view("B", extended, "canonical"),
             "C": make_view("C", canonical, "canonical"), "native_A": make_view("A", native, "native")}
    errors = [error for view in views.values() for error in validation.validate_record(view, label_sets)]
    errors += validation.check_model_input(tokens)
    reject = sorted((utterance_flags & REJECT) | ({"validation_error"} if errors else set()))
    positions = _flag_positions(extended_flags, case_masked)
    token_unresolved = UNRESOLVED_TOKEN & set(positions)
    punct_unresolved = sorted((utterance_flags & UNRESOLVED_COMMON) | token_unresolved)
    unresolved = {"A": punct_unresolved, "B": punct_unresolved, "C": sorted(set(punct_unresolved) | case_flags)}
    status = "rejected" if reject else ("flagged" if any(unresolved.values()) else "accepted")
    info = sorted((utterance_flags - REJECT - UNRESOLVED_COMMON)
                  | ({"case_mixed_approximation"} if case_masked else set())
                  | {flag for flag in positions if flag in ("ellipsis", "dash", "cutoff_hyphen", "abbrev_at_utt_end")})
    return {**base, "status": status, "reasons": reject or sorted(set().union(*unresolved.values())),
            "unresolved": unresolved, "positions": dict(positions), "info": info, "validation_errors": errors,
            "views": views, "tokens": tokens, "canonical": canonical, "native": native, "ext": extended, "case": case,
            "case_cats": case_categories, "case_masked": case_masked}


def load_records(path: str | os.PathLike[str], label_sets: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Read one release file for model training/evaluation. Raises on any contract violation; never repairs."""
    label_sets = label_sets or labeling.load_label_sets()
    _, errors = validation.validate_file(path, label_sets)
    if errors:
        raise ValueError(f"{path}: {len(errors)} contract errors, first: line {errors[0][0]}: {errors[0][1]}")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def read_ids(path: str | os.PathLike[str]) -> list[str]:
    """Sample IDs from an .ids list (e.g. default_train/mode_A.ids or primary_eval/test_mode_A.ids), in file order.

    Exactly main 72ebfd1's reader (old name kept, restructure spec section 2): each line stripped, blank lines
    skipped, a repeated ID kept, a leading BOM left on the first ID, a list without IDs read as []. Bytes that are not
    UTF-8 raise UnicodeDecodeError. The commands refuse such lists with `prediction_io.read_ids` instead.
    """
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _title_entry(line: str, where: str) -> tuple[str, str] | None:
    """(video ID, title) of one metadata line with status 200; None for any other status."""
    row = prediction_io.parse_json_object(line, where)
    if row.get("status") != 200:
        return None
    video_id = row.get("video_id")
    if not isinstance(video_id, str):
        raise ValueError(f"{where}: video_id must be a string")
    title = row.get("title") or ""
    if not isinstance(title, str):
        raise ValueError(f"{where}: title must be a string")
    return video_id, title


def read_titles(path: str | os.PathLike[str]) -> dict[str, str]:
    """video ID -> title from the dated metadata snapshot (JSONL: video_id, status, title).

    Raises ValueError("<file name> line N: <what is wrong>") for a line that is not a JSON object or has a status 200
    without a string video_id, and ValueError("<file name>: not valid UTF-8") for bytes that are not UTF-8.
    """
    name = os.path.basename(path)
    try:
        with open(path, encoding="utf-8") as f:
            entries = (_title_entry(line, f"{name} line {number}") for number, line in enumerate(f, 1))
            return dict(entry for entry in entries if entry is not None)
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name}: not valid UTF-8") from exc


class Preflight(NamedTuple):
    """What the build needs once its preflight checks passed: the TTS titles, label sets and the two output folders."""

    titles: dict[str, str] | None
    label_sets: dict[str, Any]
    out: str
    split_out: str


@dataclass(frozen=True)
class ReleaseSources:
    """Where one release build reads from and writes to: the raw transcripts, the output folders and the release ID.

    `raw_manifest` (MANIFEST.sha256 of the raw data) and `titles_path` (the metadata snapshot used for the TTS flags)
    are optional; a build without them is not the published release.
    """

    raw_dir: str
    processed_dir: str
    splits_dir: str
    release_id: str
    label_sets_path: str = labeling.LABEL_SETS_PATH
    raw_manifest: str | None = None
    titles_path: str | None = None


def _preflight(sources: ReleaseSources) -> Preflight:
    """Checks before anything is read or written; raises if the build must not start."""
    titles = read_titles(sources.titles_path) if sources.titles_path else None
    label_sets = labeling.load_label_sets(sources.label_sets_path)
    if not label_sets["projection"].get("approved"):
        raise ValueError("the B -> A projection in configs/label_sets.yaml is not approved")
    out = os.path.join(sources.processed_dir, sources.release_id)
    split_out = os.path.join(sources.splits_dir, sources.release_id)
    if os.path.exists(out) or os.path.exists(split_out):
        raise FileExistsError(f"release already exists: {sources.release_id}; use a new immutable release ID")
    raw_bad_before = preprocessing.verify_raw_manifest(sources.raw_manifest) if sources.raw_manifest else None
    if raw_bad_before:
        raise ValueError(f"raw manifest mismatch before build: {raw_bad_before[:5]}")
    return Preflight(titles, label_sets, out, split_out)


def _load_sources(raw_dir: str, titles: Mapping[str, str] | None, log: Log) -> tuple[
        dict[str, list[preprocessing.SourceUtterance]], dict[str, preprocessing.VideoContext]]:
    """Read every source split; returns (utterances per split, video context per video ID)."""
    utterances = {split: preprocessing.read_source_split(raw_dir, split) for split in preprocessing.SOURCE_SPLITS}
    log("loaded " + ", ".join(f"{split}={len(found)}" for split, found in utterances.items()))
    video_texts = collections.defaultdict(list)
    for split in preprocessing.SOURCE_SPLITS:
        for utterance in sorted(utterances[split], key=lambda utterance: utterance.index):
            video_texts[utterance.group_id].append(utterance.text)
    contexts = {video: preprocessing.video_context(texts, (titles or {}).get(video))
                for video, texts in sorted(video_texts.items())}
    return utterances, contexts


def _group_kept(kept: Sequence[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]],
                                                          dict[str, list[dict[str, Any]]]]:
    """Released records grouped by release file and by source split."""
    by_file = collections.defaultdict(list)
    by_source = collections.defaultdict(list)
    for record in kept:
        by_file[record["file"]].append(record)
        by_source[record["source_split"]].append(record)
    return by_file, by_source


def _log_checks(checks: Mapping[str, Any], log: Log) -> None:
    """One summary line of the read-back checks."""
    log("checks: " + json.dumps({k: v for k, v in checks.items() if k not in ("file_errors", "view_mismatches")})
        + f" file_errors={len(checks['file_errors'])} view_mismatches={len(checks['view_mismatches'])}")


def build_release_from(sources: ReleaseSources, *, log: Log = discard_log) -> dict[str, Any]:
    """Build the release; returns the manifest dict. `log` gets one line per step (the default says nothing)."""
    titles, label_sets, out, split_out = _preflight(sources)
    utterances, contexts = _load_sources(sources.raw_dir, titles, log)
    results = [process(utterance, contexts[utterance.group_id], label_sets)
               for split in preprocessing.SOURCE_SPLITS for utterance in utterances[split]]
    log("processed " + json.dumps(collections.Counter(result["status"] for result in results)))
    kept = [result for result in results if result["status"] != "rejected"]
    by_file, by_source = _group_kept(kept)

    files = write_views(out, by_file)       # rejected records are never written to a view
    write_masks(out, by_file)
    write_audit(out, results, kept)
    write_split_map(split_out, results)
    controlled, controlled_report = write_leakage_controlled(split_out, by_source)
    default = {"train": by_file["train"], **controlled}
    write_id_lists(split_out, default)
    kept_per_subset = {name: report["kept"] for name, report in controlled_report.items()}
    log("leakage-controlled subsets: " + json.dumps(kept_per_subset))

    overlap = pairwise_overlap(by_file, controlled)
    stats = build_stats(StatsInputs(results=results, kept=kept, by_file=by_file, default=default,
                                    controlled_report=controlled_report, overlap=overlap, label_sets=label_sets))
    verified = verify_release(ReleaseReadback(out=out, files=files, results=results, kept=kept,
                                              label_sets=label_sets, raw_manifest=sources.raw_manifest))
    stats["checks"] = verified.checks
    write_json_reports(out, split_out, stats, controlled_report, overlap)

    copy_license(sources.raw_dir, out)
    manifest = build_manifest(ManifestInputs(
        release_id=sources.release_id, label_sets_path=sources.label_sets_path, raw_manifest=sources.raw_manifest,
        titles_used=titles is not None, code_sha256=code_fingerprint, source_count=len(results),
        released_count=len(kept)))
    manifest["provenance"] = build_provenance(manifest["raw_manifest_sha256"], sources.titles_path, contexts, by_file)
    write_release_notes(out, sources.release_id, stats, manifest)
    manifest = with_file_inventory(manifest, verified.files, out, split_out)
    write_manifest(out, manifest)
    write_checksums((out, split_out))
    _log_checks(stats["checks"], log)
    return manifest


def build_release(raw_dir: str, processed_dir: str, splits_dir: str, release_id: str, *,
                  label_sets_path: str = labeling.LABEL_SETS_PATH, raw_manifest: str | None = None,
                  titles_path: str | None = None, log: Log = discard_log) -> dict[str, Any]:
    """`build_release_from` with the call form of main 72ebfd1 (old name kept, restructure spec section 2).

    Returns the manifest dict. Main printed its progress by default; `log` now says nothing unless given one.
    """
    sources = ReleaseSources(raw_dir, processed_dir, splits_dir, release_id, label_sets_path=label_sets_path,
                             raw_manifest=raw_manifest, titles_path=titles_path)
    return build_release_from(sources, log=log)
