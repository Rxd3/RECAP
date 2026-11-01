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
from collections.abc import Callable

from ..paths import REPO_ROOT
from . import labeling, preprocessing, splits, validation
from .release_checks import verify_release
from .release_constants import (
    JSON,
    REJECT,
    RELEASE_FILES,
    SOURCE_REFERENCE,
    UNRESOLVED_CASE,
    UNRESOLVED_COMMON,
    UNRESOLVED_TOKEN,
)
from .release_manifest import (
    build_manifest,
    build_provenance,
    copy_license,
    describe_files,
    sha256_file,
    with_file_inventory,
    write_checksums,
    write_manifest,
)
from .release_notes import release_notes_lines
from .release_outputs import (
    write_audit,
    write_id_lists,
    write_json_reports,
    write_leakage_controlled,
    write_masks,
    write_split_map,
    write_views,
)
from .release_stats import build_stats, label_distribution, length_stats, mode_labels, pairwise_overlap

__all__ = [
    "JSON", "REJECT", "RELEASE_FILES", "SOURCE_REFERENCE", "UNRESOLVED_CASE", "UNRESOLVED_COMMON", "UNRESOLVED_TOKEN",
    "build_release", "code_fingerprint", "describe_files", "label_distribution", "length_stats", "load_records",
    "mode_labels", "process", "read_ids", "read_titles", "sha256_file", "write_lines", "write_release_notes",
]


def code_fingerprint():
    """SHA-256 over this package's source files, recorded in the manifest."""
    here = os.path.join(REPO_ROOT, "src", "punct_restore", "data")
    h = hashlib.sha256()
    for name in sorted(os.listdir(here)):
        if name.endswith(".py"):
            with open(os.path.join(here, name), "rb") as f:
                h.update(name.encode() + b"\0" + f.read())
    return h.hexdigest()


def _flag_positions(ext_flags: list, case_masked: list) -> collections.defaultdict:
    """Token positions per flag; masked casing positions are listed under case_mixed_approximation."""
    positions = collections.defaultdict(list)
    for i, fl in enumerate(ext_flags):
        for f in fl:
            positions[f].append(i)
    if case_masked:
        positions["case_mixed_approximation"] = case_masked
    return positions


def process(utt, context, label_sets):
    """One transcript -> views, status, reasons, masks and audit information."""
    cleaned, removals, uflags = preprocessing.normalize(utt.text, context["speakers"])
    uflags |= context["flags"]
    toks = preprocessing.tokenize(cleaned)
    out = {"sample_id": splits.sample_id(preprocessing.SOURCE_ID, utt.source_split, utt.upstream_id),
           "source_split": utt.source_split, "file": splits.FILE_OF[utt.source_split], "group_id": utt.group_id,
           "original_text": utt.text, "removals": removals, "views": {}, "tokens": []}
    if not toks:
        return {**out, "status": "rejected", "reasons": sorted(uflags | {"no_tokens"}), "validation_errors": []}
    ext, ext_flags = labeling.punctuation_labels(toks, "extended")
    native, native_flags = labeling.punctuation_labels(toks, "native")
    canonical = labeling.project_extended_to_standard(ext, label_sets)
    case, case_cats, case_masked, case_flags = labeling.casing_labels(toks, cleaned)
    if not labeling.cross_check_standard(toks, native, cleaned):
        uflags.add("reconstruction_mismatch")
    if any(t.text.startswith("PS") and t.text[2:3].isdigit() for t in toks):
        uflags.add("currency_folded")                  # '£' became 'PS' upstream (information only)
    if any(t.text.startswith("-") for t in toks):
        uflags.add("leading_minus")                    # kept as a negative sign (D3); dialogue dashes may look alike
    tokens = [preprocessing.model_token(t.text) for t in toks]
    meta = {"upstream_id": f"{utt.source_split}/{utt.upstream_id}", "source_split": utt.source_split,
            "group_id": utt.group_id, "original_text": utt.text, "normalized_text": cleaned,
            "offsets": [[t.start, t.end] for t in toks], "source_revision": preprocessing.SOURCE_REVISION,
            "policy_version": preprocessing.POLICY_VERSION, "condition": "clean"}

    def view(mode, punct, annotation):
        return {"schema_version": validation.SCHEMA_VERSION, "sample_id": out["sample_id"],
                "source": preprocessing.SOURCE_ID, "split": splits.SPLIT_OF[utt.source_split], "label_mode": mode,
                "tokens": tokens, "punct_labels": punct, "case_labels": case,
                "metadata": {**meta, "annotation_view": annotation}}
    views = {"A": view("A", canonical, "canonical"), "B": view("B", ext, "canonical"),
             "C": view("C", canonical, "canonical"), "native_A": view("A", native, "native")}
    errors = [e for v in views.values() for e in validation.validate_record(v, label_sets)]
    errors += validation.check_model_input(tokens)
    reject = sorted((uflags & REJECT) | ({"validation_error"} if errors else set()))
    positions = _flag_positions(ext_flags, case_masked)
    token_unresolved = UNRESOLVED_TOKEN & set(positions)
    punct = sorted((uflags & UNRESOLVED_COMMON) | token_unresolved)
    unresolved = {"A": punct, "B": punct, "C": sorted(set(punct) | case_flags)}
    status = "rejected" if reject else ("flagged" if any(unresolved.values()) else "accepted")
    info = sorted((uflags - REJECT - UNRESOLVED_COMMON) | ({"case_mixed_approximation"} if case_masked else set())
                  | ({f for f in positions if f in ("ellipsis", "dash", "cutoff_hyphen", "abbrev_at_utt_end")}))
    return {**out, "status": status, "reasons": reject or sorted(set().union(*unresolved.values())),
            "unresolved": unresolved, "positions": dict(positions), "info": info, "validation_errors": errors,
            "views": views, "tokens": tokens, "canonical": canonical, "native": native, "ext": ext, "case": case,
            "case_cats": case_cats, "case_masked": case_masked}


def load_records(path, label_sets=None):
    """Read one release file for model training/evaluation. Raises on any contract violation; never repairs."""
    label_sets = label_sets or labeling.load_label_sets()
    ok, errors = validation.validate_file(path, label_sets)
    if errors:
        raise ValueError(f"{path}: {len(errors)} contract errors, first: line {errors[0][0]}: {errors[0][1]}")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def read_ids(path):
    """Sample IDs from an .ids list (e.g. default_train/mode_A.ids or primary_eval/test_mode_A.ids)."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)


def read_titles(path):
    """video ID -> title from the dated metadata snapshot (JSONL: video_id, status, title)."""
    titles = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") == 200:
                titles[r["video_id"]] = r.get("title") or ""
    return titles


def _preflight(release_id: str, processed_dir: str, splits_dir: str, label_sets_path: str,
               raw_manifest: str | None, titles_path: str | None) -> tuple:
    """Checks before anything is read or written; returns (titles, label_sets, out, split_out, raw_bad_before)."""
    titles = read_titles(titles_path) if titles_path else None
    label_sets = labeling.load_label_sets(label_sets_path)
    if not label_sets["projection"].get("approved"):
        raise PermissionError("the B -> A projection in configs/label_sets.yaml is not approved")
    out = os.path.join(processed_dir, release_id)
    split_out = os.path.join(splits_dir, release_id)
    if os.path.exists(out) or os.path.exists(split_out):
        raise FileExistsError(f"release already exists: {release_id}; use a new immutable release ID")
    raw_bad_before = preprocessing.verify_raw_manifest(raw_manifest) if raw_manifest else None
    if raw_bad_before:
        raise ValueError(f"raw manifest mismatch before build: {raw_bad_before[:5]}")
    return titles, label_sets, out, split_out, raw_bad_before


def _load_sources(raw_dir: str, titles: dict | None, log: Callable) -> tuple[dict, dict]:
    """Read every source split; returns (utterances per split, video context per video ID)."""
    utts = {s: preprocessing.read_source_split(raw_dir, s) for s in preprocessing.SOURCE_SPLITS}
    log("loaded " + ", ".join(f"{s}={len(u)}" for s, u in utts.items()))
    texts = collections.defaultdict(list)
    for s in preprocessing.SOURCE_SPLITS:
        for u in sorted(utts[s], key=lambda u: u.index):
            texts[u.group_id].append(u.text)
    contexts = {v: preprocessing.video_context(t, (titles or {}).get(v)) for v, t in sorted(texts.items())}
    return utts, contexts


def _group_kept(kept: list) -> tuple[dict, dict]:
    """Released records grouped by release file and by source split."""
    by_file = collections.defaultdict(list)
    by_source = collections.defaultdict(list)
    for r in kept:
        by_file[r["file"]].append(r)
        by_source[r["source_split"]].append(r)
    return by_file, by_source


def _log_checks(checks: dict, log: Callable) -> None:
    """One summary line of the read-back checks."""
    log("checks: " + json.dumps({k: v for k, v in checks.items() if k not in ("file_errors", "view_mismatches")})
        + f" file_errors={len(checks['file_errors'])} view_mismatches={len(checks['view_mismatches'])}")


def build_release(raw_dir, processed_dir, splits_dir, release_id, *, label_sets_path=labeling.LABEL_SETS_PATH,
                  raw_manifest=None, titles_path=None, log=print):
    """Build the release; returns the manifest dict. `titles_path` is the metadata snapshot used for TTS flags."""
    titles, label_sets, out, split_out, raw_bad_before = _preflight(
        release_id, processed_dir, splits_dir, label_sets_path, raw_manifest, titles_path)
    utts, contexts = _load_sources(raw_dir, titles, log)
    results = [process(u, contexts[u.group_id], label_sets) for s in preprocessing.SOURCE_SPLITS for u in utts[s]]
    log("processed " + json.dumps(collections.Counter(r["status"] for r in results)))
    kept = [r for r in results if r["status"] != "rejected"]
    by_file, by_source = _group_kept(kept)

    files = write_views(out, by_file, write_lines)       # rejected records are never written to a view
    write_masks(out, by_file, write_lines)
    write_audit(out, results, kept, write_lines)
    write_split_map(split_out, results, write_lines)
    controlled, controlled_report = write_leakage_controlled(split_out, by_source, write_lines)
    default = {"train": by_file["train"], **controlled}
    write_id_lists(split_out, default, write_lines)
    log("leakage-controlled subsets: " + json.dumps({k: v["kept"] for k, v in controlled_report.items()}))

    overlap = pairwise_overlap(by_file, controlled)
    stats = build_stats(results, kept, by_file, default, controlled_report, overlap, label_sets)
    stats["checks"] = verify_release(out, files, results, kept, label_sets, raw_manifest, raw_bad_before)
    write_json_reports(out, split_out, stats, controlled_report, overlap)

    copy_license(raw_dir, out)
    manifest = build_manifest(release_id, label_sets_path, raw_manifest, titles is not None, code_fingerprint,
                              len(results), len(kept))
    manifest["provenance"] = build_provenance(manifest["raw_manifest_sha256"], titles_path, contexts, by_file)
    write_release_notes(out, release_id, stats, manifest)
    manifest = with_file_inventory(manifest, files, out, split_out)
    write_manifest(out, manifest)
    write_checksums((out, split_out), write_lines)
    _log_checks(stats["checks"], log)
    return manifest


def write_release_notes(out, release_id, stats, manifest):
    """Short human-readable notes stored with the release (all numbers come from stats)."""
    write_lines(os.path.join(out, "RELEASE_NOTES.md"), release_notes_lines(release_id, stats, manifest))
