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
import platform
import shutil
import statistics
from contextlib import ExitStack
from itertools import zip_longest
from pathlib import Path, PurePosixPath

from . import extraction, labeling, preprocessing, splits, validation

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


def code_fingerprint():
    """SHA-256 over this package's source files, recorded in the manifest."""
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for name in sorted(os.listdir(here)):
        if name.endswith(".py"):
            with open(os.path.join(here, name), "rb") as f:
                h.update(name.encode() + b"\0" + f.read())
    return h.hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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
    positions = collections.defaultdict(list)
    for i, fl in enumerate(ext_flags):
        for f in fl:
            positions[f].append(i)
    if case_masked:
        positions["case_mixed_approximation"] = case_masked
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


def label_distribution(label_lists, vocab):
    c = collections.Counter(l for labels in label_lists for l in labels)
    n = sum(c.values())
    return {"records": len(label_lists), "tokens": n, "counts": {l: c.get(l, 0) for l in vocab},
            "percent": {l: round(100 * c.get(l, 0) / n, 3) if n else 0.0 for l in vocab}}


def length_stats(records):
    lens = sorted(len(r["tokens"]) for r in records)
    if not lens:
        return {}
    q = lambda p: lens[min(len(lens) - 1, int(p * len(lens)))]
    return {"records": len(lens), "tokens": sum(lens), "min": lens[0], "median": statistics.median(lens),
            "mean": round(statistics.fmean(lens), 2), "p95": q(0.95), "max": lens[-1]}


def write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)


def mode_labels(r, mode):
    return r["ext"] if mode == "B" else r["canonical"]


def read_titles(path):
    """video ID -> title from the dated metadata snapshot (JSONL: video_id, status, title)."""
    titles = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") == 200:
                titles[r["video_id"]] = r.get("title") or ""
    return titles


def describe_files(folder, skip=("release_manifest.json", "SHA256SUMS")):
    """{relative path: {sha256, bytes, lines}} for every file below `folder`."""
    out = {}
    for d, _, fs in os.walk(folder):
        for x in fs:
            p = os.path.join(d, x)
            rel = Path(p).relative_to(folder).as_posix()
            if rel in skip:
                continue
            with open(p, "rb") as f:
                lines = sum(1 for _ in f)
            out[rel] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p), "lines": lines}
    return dict(sorted(out.items()))


def build_release(raw_dir, processed_dir, splits_dir, release_id, *, label_sets_path=labeling.LABEL_SETS_PATH,
                  raw_manifest=None, titles_path=None, log=print):
    """Build the release; returns the manifest dict. `titles_path` is the metadata snapshot used for TTS flags."""
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
    utts = {s: preprocessing.read_source_split(raw_dir, s) for s in preprocessing.SOURCE_SPLITS}
    log("loaded " + ", ".join(f"{s}={len(u)}" for s, u in utts.items()))
    texts = collections.defaultdict(list)
    for s in preprocessing.SOURCE_SPLITS:
        for u in sorted(utts[s], key=lambda u: u.index):
            texts[u.group_id].append(u.text)
    contexts = {v: preprocessing.video_context(t, (titles or {}).get(v)) for v, t in sorted(texts.items())}

    results = [process(u, contexts[u.group_id], label_sets) for s in preprocessing.SOURCE_SPLITS for u in utts[s]]
    log("processed " + json.dumps(collections.Counter(r["status"] for r in results)))
    kept = [r for r in results if r["status"] != "rejected"]
    by_file = collections.defaultdict(list)
    by_source = collections.defaultdict(list)
    for r in kept:
        by_file[r["file"]].append(r)
        by_source[r["source_split"]].append(r)

    # views (rejected records are never written to a view)
    files = {}
    for folder, key in (("mode_A", "A"), ("mode_B", "B"), ("mode_C", "C"), ("benchmark_native/mode_A", "native_A")):
        for name in RELEASE_FILES:
            path = os.path.join(out, folder, name + ".jsonl")
            write_lines(path, (json.dumps(r["views"][key], **JSON) + "\n" for r in by_file[name]))
            files[Path(path).relative_to(out).as_posix()] = {"records": len(by_file[name])}

    # casing/joint evaluation masks: one line per released record, positions excluded from casing and joint metrics
    for name in RELEASE_FILES:
        write_lines(os.path.join(out, "masks", "case_joint", name + ".jsonl"),
                    (json.dumps({"sample_id": r["sample_id"], "excluded_positions": r["case_masked"],
                                 "reason": "case_mixed_approximation" if r["case_masked"] else None}, **JSON) + "\n"
                     for r in by_file[name]))

    # audit trail: rejected, unresolved (kept in the views) and every normalization removal
    write_lines(os.path.join(out, "audit", "rejected.jsonl"),
                (json.dumps({k: r.get(k) for k in ("sample_id", "source_split", "group_id", "reasons",
                                                    "validation_errors", "original_text")}, **JSON) + "\n"
                 for r in results if r["status"] == "rejected"))
    write_lines(os.path.join(out, "audit", "unresolved.jsonl"),
                (json.dumps({"sample_id": r["sample_id"], "source_split": r["source_split"],
                             "unresolved_by_mode": r["unresolved"], "token_positions": r["positions"]}, **JSON) + "\n"
                 for r in kept if r["status"] == "flagged"))
    write_lines(os.path.join(out, "audit", "normalization.jsonl"),
                (json.dumps({"sample_id": r["sample_id"], "removals": r["removals"]}, **JSON) + "\n"
                 for r in results if r["removals"]))

    # split map, frozen leakage-controlled subsets, default training and primary evaluation ID lists
    write_lines(os.path.join(split_out, "split_map.tsv"),
                ["sample_id\tsplit\tsource_split\tfile\tgroup_id\tstatus\n"] +
                [f"{r['sample_id']}\t{splits.SPLIT_OF[r['source_split']]}\t{r['source_split']}\t{r['file']}\t"
                 f"{r['group_id']}\t{r['status']}\n" for r in results])
    controlled, controlled_report = {}, {}
    for s, refs in splits.DISJOINT_REFERENCE.items():
        ref = splits.Reference([r for name in refs for r in by_source[name]])
        reasons, keep, dropped = collections.Counter(), [], []
        for r in by_source[s]:
            why = splits.overlap_reason(r, ref)
            if why is None and {"low_comma_video", "tts_source"} & set(r["unresolved"]["A"]):
                why = "low_comma_or_tts_source"
            if why:
                reasons[why] += 1
                dropped.append(f"{r['sample_id']}\t{why}\n")
            else:
                keep.append(r)
        name = splits.FILE_OF[s]
        controlled[name] = keep
        controlled_report[name] = {"reference": [splits.FILE_OF[x] for x in refs], "official_records": len(by_source[s]),
                                   "kept": len(keep), "kept_videos": len({r["group_id"] for r in keep}),
                                   "dropped_by_reason": dict(sorted(reasons.items()))}
        write_lines(os.path.join(split_out, "leakage_controlled", name + ".ids"), (r["sample_id"] + "\n" for r in keep))
        write_lines(os.path.join(split_out, "leakage_controlled", name + "_excluded.tsv"),
                    ["sample_id\treason\n"] + dropped)
    default = {"train": by_file["train"], **controlled}
    for mode in ("A", "B", "C"):
        write_lines(os.path.join(split_out, "default_train", f"mode_{mode}.ids"),
                    (r["sample_id"] + "\n" for r in default["train"] if not r["unresolved"][mode]))
        for name in ("validation", "test", "test_amb"):
            write_lines(os.path.join(split_out, "primary_eval", f"{name}_mode_{mode}.ids"),
                        (r["sample_id"] + "\n" for r in default[name] if not r["unresolved"][mode]))
    log("leakage-controlled subsets: " + json.dumps({k: v["kept"] for k, v in controlled_report.items()}))

    # pairwise overlap verification (independent string-set check), official and leakage-controlled
    overlap = {}
    for view, sets in (("official", {n: by_file[n] for n in RELEASE_FILES}),
                       ("leakage_controlled", {"train": by_file["train"], **controlled})):
        for i, a in enumerate(RELEASE_FILES):
            for b in RELEASE_FILES[i + 1:]:
                overlap[f"{view}:{b}_vs_{a}"] = splits.overlap_counts(sets[b], sets[a])

    # statistics
    sel = lambda recs, mode: [r for r in recs if not r["unresolved"][mode]]
    stats = {"status_by_file": {n: dict(collections.Counter(r["status"] for r in results if r["file"] == n)) for n in RELEASE_FILES},
             "reject_reasons": dict(collections.Counter(x for r in results if r["status"] == "rejected" for x in r["reasons"])),
             "unresolved_reasons_by_mode": {m: dict(collections.Counter(x for r in kept for x in r["unresolved"][m]).most_common())
                                            for m in ("A", "B", "C")},
             "info_flags": dict(collections.Counter(x for r in kept for x in r["info"]).most_common()),
             "lengths": {n: length_stats(by_file[n]) for n in RELEASE_FILES},
             "groups": {n: len({r["group_id"] for r in by_file[n]}) for n in RELEASE_FILES},
             "normalization_removals": dict(collections.Counter(op for r in results for op, _ in r["removals"]).most_common()),
             "leakage_controlled": controlled_report, "overlap": overlap,
             "subset_sizes": {m: {"default_train": len(sel(default["train"], m)),
                                  **{f"primary_{n}": len(sel(default[n], m)) for n in ("validation", "test", "test_amb")}}
                              for m in ("A", "B", "C")},
             "labels": {}, "labels_default_and_primary": {}, "canonical_vs_native": {}, "case_masks": {},
             "case_categories": {n: dict(collections.Counter(c for r in by_file[n] for c in r["case_cats"])) for n in RELEASE_FILES}}
    for n in RELEASE_FILES:
        recs = by_file[n]
        stats["labels"][n] = {"A": label_distribution([r["canonical"] for r in recs], label_sets["A"]),
                              "B": label_distribution([r["ext"] for r in recs], label_sets["B"]),
                              "native_A": label_distribution([r["native"] for r in recs], label_sets["A"]),
                              "case": label_distribution([r["case"] for r in recs], label_sets["casing"]),
                              "C": label_distribution([labeling.join_labels(r["canonical"], r["case"]) for r in recs], label_sets["C"])}
        subset = default[n] if n in default else []
        stats["labels_default_and_primary"][n] = {m: label_distribution([mode_labels(r, m) for r in sel(subset, m)], label_sets[m])
                                                  for m in ("A", "B")}
        stats["labels_default_and_primary"][n]["C"] = label_distribution(
            [labeling.join_labels(r["canonical"], r["case"]) for r in sel(subset, "C")], label_sets["C"])
        pairs = [(a, b) for r in recs for a, b in zip(r["native"], r["canonical"])]
        stats["canonical_vs_native"][n] = {"tokens": len(pairs), "changed": sum(a != b for a, b in pairs),
                                           "records_changed": sum(r["native"] != r["canonical"] for r in recs),
                                           "changes": dict(collections.Counter(f"{a}->{b}" for a, b in pairs if a != b))}
        masked = sum(len(r["case_masked"]) for r in recs)
        tokens = sum(len(r["tokens"]) for r in recs)
        stats["case_masks"][n] = {"masked_positions": masked, "records_with_masks": sum(bool(r["case_masked"]) for r in recs),
                                  "tokens": tokens, "masked_percent": round(100 * masked / max(1, tokens), 3)}
    stats["final_token"] = {n: {"final_label_counts": dict(collections.Counter(r["canonical"][-1] for r in by_file[n])),
                                "share_of_PERIOD_that_is_final": round(100 * sum(r["canonical"][-1] == "PERIOD" for r in by_file[n])
                                                                       / max(1, sum(r["canonical"].count("PERIOD") for r in by_file[n])), 2)}
                            for n in RELEASE_FILES}

    # read back every view: contract validation and cross-view consistency
    checks = {"source_records": len(results), "rejected": len(results) - len(kept), "file_errors": {}, "view_mismatches": {}}
    for path in files:
        mode = PurePosixPath(path).parent.name.removeprefix("mode_")
        name = PurePosixPath(path).stem
        ok, errs = validation.validate_file(os.path.join(out, path), label_sets, split=splits.SPLIT_OF[{"validation": "dev", "test_amb": "test-amb"}.get(name, name)], mode=mode)
        files[path]["valid"] = ok
        if errs:
            checks["file_errors"][path] = errs[:20]
    for name in RELEASE_FILES:
        # Read aligned rows together instead of retaining four full JSONL views.
        # The train views contain millions of tokens; full reloads multiply peak RAM.
        diffs = []
        with ExitStack() as stack:
            streams = [stack.enter_context(open(os.path.join(out, folder, name + ".jsonl"), encoding="utf-8"))
                       for folder in ("mode_A", "mode_B", "mode_C", "benchmark_native/mode_A")]
            for lines in zip_longest(*streams):
                if any(line is None for line in lines):
                    diffs.append("sample IDs or their order differ")
                    break
                rows = [json.loads(line) for line in lines]
                diffs.extend(d for row in rows[1:] for d in validation.compare_views(rows[:1], [row]))
                if len(diffs) >= 20:
                    break
        if diffs:
            checks["view_mismatches"][name] = diffs[:20]
    checks["unique_sample_ids"] = len({r["sample_id"] for r in results}) == len(results)
    checks["raw_mismatch_before"] = raw_bad_before
    checks["raw_mismatch_after"] = preprocessing.verify_raw_manifest(raw_manifest) if raw_manifest else None
    if (checks["file_errors"] or checks["view_mismatches"] or not checks["unique_sample_ids"]
            or checks["raw_mismatch_after"]):
        raise ValueError("release validation failed; no release manifest or checksums will be published")
    stats["checks"] = checks
    with open(os.path.join(out, "stats.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(stats, f, indent=1, sort_keys=True)
    with open(os.path.join(split_out, "leakage_controlled", "report.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"subsets": controlled_report, "overlap": overlap}, f, indent=1, sort_keys=True)

    # license and attribution travel with the release (CC BY-NC 4.0 requires attribution)
    license_src = os.path.join(raw_dir, "README.txt")
    if os.path.exists(license_src):
        shutil.copyfile(license_src, os.path.join(out, "SPONSPEECH_LICENSE_AND_ATTRIBUTION.txt"))
    manifest = {"release_id": release_id, "status": "final candidate: policies D1-D6 approved by the team lead",
                "schema_version": validation.SCHEMA_VERSION, "policy_version": preprocessing.POLICY_VERSION,
                "source": {"id": preprocessing.SOURCE_ID, "revision": preprocessing.SOURCE_REVISION,
                           "license": "CC BY-NC 4.0", "reference": SOURCE_REFERENCE, "decision": "D1 approved"},
                "annotation_views": {"canonical": "mode_A, mode_B, mode_C (D3, D5 approved projection)",
                                     "native": "benchmark_native/mode_A (SponSpeech four-class view, comparison only)"},
                "split_mapping": splits.SPLIT_OF, "label_sets_sha256": sha256_file(label_sets_path),
                "code_sha256": code_fingerprint(), "python": platform.python_version(),
                "raw_manifest_sha256": sha256_file(raw_manifest) if raw_manifest else None,
                "titles_used_for_tts_check": titles is not None,
                "records": {"source": len(results), "released_per_view": len(kept), "rejected": len(results) - len(kept)},
                "manifest_revision": 2}
    tts_videos = sorted(v for v, c in contexts.items() if "tts_source" in c["flags"])
    manifest["provenance"] = {
        "archive": {"url": extraction.ARCHIVE_URL, "bytes": extraction.ARCHIVE_BYTES,
                    "gcs_generation": extraction.ARCHIVE_GENERATION, "extraction": "punct_restore.data.extraction",
                    "head_range": extraction.HEAD_RANGE, "text_ranges": extraction.TEXT_RANGES},
        "raw_manifest": {"file": "MANIFEST.sha256", "sha256": manifest["raw_manifest_sha256"]},
        "metadata_snapshot": {
            "file": "video_metadata_oembed.jsonl", "sha256": sha256_file(titles_path) if titles_path else None,
            "fetched": "2026-10-28 (YouTube oEmbed; do not re-fetch, titles can change)",
            "use": "a video whose title ends with '| Wikipedia audio article' is flagged tts_source",
            "tts_videos": tts_videos,
            "tts_records_by_file": {n: sum("tts_source" in r["unresolved"]["A"] for r in by_file[n]) for n in RELEASE_FILES}}}
    write_release_notes(out, release_id, stats, manifest)
    view_info = {p: {"records": v["records"], "valid": v.get("valid")} for p, v in files.items()}
    manifest["files"] = {p: {**d, **view_info.get(p, {})} for p, d in describe_files(out).items()}
    manifest["split_files"] = describe_files(split_out)
    with open(os.path.join(out, "release_manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    for folder in (out, split_out):
        paths = sorted(Path(d, x).relative_to(folder).as_posix() for d, _, fs in os.walk(folder) for x in fs if x != "SHA256SUMS")
        write_lines(os.path.join(folder, "SHA256SUMS"), (f"{sha256_file(os.path.join(folder, p))}  {p}\n" for p in paths))
    log("checks: " + json.dumps({k: v for k, v in checks.items() if k not in ("file_errors", "view_mismatches")})
        + f" file_errors={len(checks['file_errors'])} view_mismatches={len(checks['view_mismatches'])}")
    return manifest


def write_release_notes(out, release_id, stats, manifest):
    """Short human-readable notes stored with the release (all numbers come from stats)."""
    sizes = stats["subset_sizes"]
    lines = [f"# {release_id}\n\n",
             f"Source: SponSpeech, {manifest['source']['reference']}\n",
             "License: CC BY-NC 4.0 (non-commercial use only). Attribution list: SPONSPEECH_LICENSE_AND_ATTRIBUTION.txt\n",
             f"Source revision: {manifest['source']['revision']}. Policy: {manifest['policy_version']}. Schema: v{manifest['schema_version']}.\n",
             f"Code SHA-256: {manifest['code_sha256']}\n\n",
             "## Records\n\n| file | released | flagged (unresolved) | rejected |\n|---|---:|---:|---:|\n"]
    for n in RELEASE_FILES:
        s = stats["status_by_file"][n]
        lines.append(f"| {n} | {s.get('accepted', 0) + s.get('flagged', 0)} | {s.get('flagged', 0)} | {s.get('rejected', 0)} |\n")
    lines.append("\n## Default and primary subsets (sample counts)\n\n| mode | default train | validation | test | test_amb |\n|---|---:|---:|---:|---:|\n")
    for m in ("A", "B", "C"):
        z = sizes[m]
        lines.append(f"| {m} | {z['default_train']} | {z['primary_validation']} | {z['primary_test']} | {z['primary_test_amb']} |\n")
    snap = manifest["provenance"]["metadata_snapshot"]
    lines += ["\n## Reproducing from the original source\n\n",
              f"- Archive: {manifest['provenance']['archive']['url']} ({manifest['provenance']['archive']['bytes']} bytes, "
              f"GCS generation {manifest['provenance']['archive']['gcs_generation']}).\n",
              "- `python scripts/prepare_data.py extract` recreates the transcripts from pinned byte ranges (or from a fully "
              "downloaded archive with `--archive`) and checks them against `MANIFEST.sha256` "
              f"(SHA-256 {manifest['provenance']['raw_manifest']['sha256']}).\n",
              f"- The dated metadata snapshot (SHA-256 {snap['sha256']}) flags {len(snap['tts_videos'])} text-to-speech "
              f"videos; flagged records per file: {snap['tts_records_by_file']}. They are excluded from the default and "
              "primary ID lists, so building without this exact snapshot gives different lists.\n",
              "\n## Use\n\n",
              "- Train on `mode_<M>/train.jsonl` restricted to `default_train/mode_<M>.ids`.\n",
              "- Select on `primary_eval/validation_mode_<M>.ids`; report on `primary_eval/test_mode_<M>.ids` and, separately, "
              "`primary_eval/test_amb_mode_<M>.ids`. Full official files are supplementary results.\n",
              "- Mode C: target = `join_labels(punct_labels, case_labels)`; exclude `masks/case_joint` positions from "
              "casing and joint metrics.\n",
              "- `benchmark_native/` is for comparison with the original SponSpeech four-class benchmark only. Never compare "
              "its scores with canonical mode A scores as if the gold labels were the same.\n",
              "- Models receive `sample_id` and `tokens` only.\n"]
    write_lines(os.path.join(out, "RELEASE_NOTES.md"), lines)
