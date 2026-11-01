"""Data-contract tests for Member 1's preparation code. Tiny inline fixtures only: no downloads, no real data."""
import hashlib
import json
import os

import pytest

from punct_restore.data import labeling, preprocessing, release, splits, validation

LS = labeling.load_label_sets()


def words(text):
    return [t.text for t in preprocessing.tokenize(text)]


def labels(text, scheme="extended"):
    toks = preprocessing.tokenize(preprocessing.normalize(text)[0])
    return [
        (t.text.lower(), label) for t, label in zip(toks, labeling.punctuation_labels(toks, scheme)[0], strict=False)
    ]


def canonical(text):
    projected = labeling.project_extended_to_standard([label for _, label in labels(text)], LS)
    return [(t, label) for (t, _), label in zip(labels(text), projected, strict=False)]


def record(**over):
    rec = {"schema_version": 1, "sample_id": "sponspeech:train:abcdefghijk-0", "source": "sponspeech",
           "split": "train", "label_mode": "A", "tokens": ["hello", "everyone"], "punct_labels": ["O", "PERIOD"],
           "case_labels": ["FIRST_UPPER", "LOWER"]}
    rec.update(over)
    return rec


# --- label configuration and the approved D5 projection ------------------------------------------------------------

def test_label_sets_have_4_8_12_classes_in_config_order():
    assert LS["A"] == ["O", "COMMA", "PERIOD", "QUESTION"]
    assert LS["B"] == LS["A"] + ["EXCLAMATION", "COLON", "SEMICOLON", "DASH"]
    assert len(LS["C"]) == 12 and LS["C"][:3] == ["O|LOWER", "O|FIRST_UPPER", "O|ALL_UPPER"]


def test_mode_c_is_standard_times_casing_never_extended():
    joint = labeling.join_labels(["PERIOD", "O"], ["FIRST_UPPER", "LOWER"])
    assert joint == ["PERIOD|FIRST_UPPER", "O|LOWER"] and set(joint) <= set(LS["C"])
    assert "DASH|LOWER" not in LS["C"]


def test_approved_projection_is_exactly_the_d5_mapping():
    assert LS["projection"]["approved"] is True
    assert labeling.project_extended_to_standard(LS["B"], LS) == [
        "O", "COMMA", "PERIOD", "QUESTION", "PERIOD", "COMMA", "PERIOD", "COMMA"]


def test_unapproved_projection_is_still_refused():
    unapproved = {**LS, "projection": {**LS["projection"], "approved": False}}
    with pytest.raises(ValueError, match="not approved"):
        labeling.project_extended_to_standard(["DASH"], unapproved)


# --- D2 tokenization and cleaning ---------------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("I don't think it's y'all's.", ["I", "don't", "think", "it's", "y'all's"]),
    ("It was 28.5 percent, 10,000 people, at 9:30.",
     ["It", "was", "28.5", "percent", "10,000", "people", "at", "9:30"]),
    ("Visit www.radioenglish.net or mail a.b@c.org now.",
     ["Visit", "www.radioenglish.net", "or", "mail", "a.b@c.org", "now"]),
    ("Dr. Brown lives in the U.S. now.", ["Dr.", "Brown", "lives", "in", "the", "U.S.", "now"]),
    ("it works,but today?James", ["it", "works", "but", "today", "James"]),
    ("a whole-food plant- based diet -- yes", ["a", "whole-food", "plant", "based", "diet", "yes"]),
    ("call object.assign at re:Invent with Rack::Attack",
     ["call", "object.assign", "at", "re:Invent", "with", "Rack::Attack"]),
    ("between -12 and -3 dB, a well-known -2.5% drop",
     ["between", "-12", "and", "-3", "dB", "a", "well-known", "-2.5%", "drop"]),
    ("from 20-30 to 15-20,000", ["from", "20-30", "to", "15-20,000"]),
])
def test_tokenization_keeps_words_and_internal_punctuation(text, expected):
    assert words(text) == expected


def test_offsets_point_into_cleaned_text():
    text = '"Well," she said -- U.S. 9:30.'
    assert all(text[t.start:t.end] == t.text for t in preprocessing.tokenize(text))


def test_normalization_removes_only_non_lexical_material_and_logs_it():
    text, ops, flags = preprocessing.normalize("[00:04:30] JAKE: So, [Music] it's 9:30 at [the] end.")
    assert text == "So, it's 9:30 at [the] end."
    assert [o[0] for o in ops] == ["timecode_tag", "nonspeech_or_name_tag", "speaker_label_caps"]
    assert "kept_bracket_or_paren" in flags


def test_title_case_label_needs_recurrence_in_the_video():
    inv = preprocessing.build_speaker_inventory(["L: So.", "G: Yes. L: Hm.", "L: Right.", "Moldova: the record."])
    assert preprocessing.normalize("L: Okay then.", inv)[0] == "Okay then."
    assert preprocessing.normalize("Moldova: Department of History.", inv)[0].startswith("Moldova:")


def test_model_tokens_use_str_lower():
    assert preprocessing.model_token("İSTANBUL") == "İSTANBUL".lower()


# --- D3 punctuation: DASH, hyphens, negatives, ellipsis -------------------------------------------------------------

def test_extended_labels_and_canonical_projection():
    text = "Wow! Note this: one; two -- three, done."
    assert [label for _, label in labels(text)] == ["EXCLAMATION", "O", "COLON", "SEMICOLON", "DASH", "COMMA", "PERIOD"]
    assert [label for _, label in canonical(text)] == ["PERIOD", "O", "COMMA", "PERIOD", "COMMA", "COMMA", "PERIOD"]


@pytest.mark.parametrize("text, expected", [
    ("one -- two.", ["DASH", "PERIOD"]),            # double hyphen
    ("one - two.", ["DASH", "PERIOD"]),             # standalone spaced hyphen
    ("one--two.", ["DASH", "PERIOD"]),              # glued double hyphen
    ("well-known fact.", ["O", "PERIOD"]),          # word-internal hyphen stays in the token
    ("going plant- based.", ["O", "O", "PERIOD"]),  # cut-off hyphen is not a dash
    ("it is -12 today.", ["O", "O", "O", "PERIOD"]),   # negative sign stays in the token
])
def test_dash_rules(text, expected):
    assert [label for _, label in labels(text)] == expected


def test_ellipsis_inside_is_o_and_at_the_end_is_period():
    assert [label for _, label in labels("You think... I do. And then...")] == ["O", "O", "O", "PERIOD", "O", "PERIOD"]


def test_native_view_differs_from_canonical_only_at_dash_boundaries():
    toks = preprocessing.tokenize("So: yes; no! Maybe -- well, done?")
    native, _ = labeling.punctuation_labels(toks, "native")
    canon = labeling.project_extended_to_standard(labeling.punctuation_labels(toks, "extended")[0], LS)
    assert [(t.text, a, b) for t, a, b in zip(toks, native, canon, strict=False) if a != b] == [("Maybe", "O", "COMMA")]


def test_mark_clusters_are_flagged_not_resolved():
    toks = preprocessing.tokenize("Really?! Yes.")
    assert "mark_cluster" in labeling.punctuation_labels(toks, "extended")[1][0]


# --- D4 casing precedence ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("surface, label, approximated", [
    ("2017", "LOWER", False), ("9:30", "LOWER", False),                            # 1. no cased letters
    ("I", "FIRST_UPPER", False), ("A", "FIRST_UPPER", False), ("B12", "FIRST_UPPER", False),   # 2. single uppercase
    ("FDA", "ALL_UPPER", False), ("U.S.", "ALL_UPPER", False),                     # 3. all uppercase
    ("hello", "LOWER", False), ("a", "LOWER", False), ("Hello", "FIRST_UPPER", False), ("I'm", "FIRST_UPPER", False),
    ("iPhone", "LOWER", True), ("NetApp", "FIRST_UPPER", True), ("McDonald", "FIRST_UPPER", True),   # 4. first letter
])
def test_casing_precedence(surface, label, approximated):
    assert labeling.case_label(surface) == (label, approximated)


def test_casing_mask_keeps_positions_and_flags_source_casing():
    toks = preprocessing.tokenize("we met NetApp and iPhone people")
    labels_, cats, masked, flags = labeling.casing_labels(toks, "we met NetApp and iPhone people")
    assert len(labels_) == len(toks) and masked == [2, 4]
    assert "case_source_no_capitals" not in flags
    assert "case_source_no_capitals" in labeling.casing_labels(toks[:2], "we met")[3]


# --- contract validation --------------------------------------------------------------------------------------------

def test_valid_record_passes():
    assert validation.validate_record(record(), LS) == []
    assert validation.validate_record(record(label_mode="B", punct_labels=["DASH", "PERIOD"]), LS) == []


@pytest.mark.parametrize("change", [
    {"schema_version": "1"}, {"schema_version": True}, {"schema_version": 2}, {"split": "dev"}, {"split": "test-amb"},
    {"label_mode": "D"}, {"tokens": []}, {"tokens": ["hello", " "]}, {"punct_labels": ["O"]},
    {"punct_labels": ["DASH", "PERIOD"]}, {"case_labels": ["UPPER", "LOWER"]}, {"sample_id": ""}, {"extra": 1},
    {"metadata": "x"},
])
def test_invalid_records_are_rejected(change):
    assert validation.validate_record(record(**change), LS)


def test_missing_field_is_rejected():
    rec = record()
    del rec["case_labels"]
    assert "missing field case_labels" in validation.validate_record(rec, LS)


def test_file_validation_catches_blank_lines_bad_json_and_duplicates(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps(record()) + "\n\n{bad\n" + json.dumps(record()) + "\n", encoding="utf-8")
    ok, errors = validation.validate_file(p, LS)
    kinds = " ".join(e for _, e in errors)
    assert ok == 1 and "blank line" in kinds and "invalid JSON" in kinds and "duplicate sample_id" in kinds


def test_model_input_check_flags_casing_and_marks():
    assert validation.check_model_input(["Hello", "world,"])
    assert validation.check_model_input(["hello", "u.s.", "9:30", "-12"]) == []


def test_views_must_share_ids_and_tokens():
    assert validation.compare_views([record()], [record(label_mode="B")]) == []
    assert validation.compare_views([record()], [record(tokens=["hi", "everyone"])])


# --- D6 identities and leakage --------------------------------------------------------------------------------------

def test_sample_ids_and_split_mapping():
    assert splits.sample_id("sponspeech", "train", "x-1") != splits.sample_id("sponspeech", "dev", "x-1")
    assert splits.SPLIT_OF == {"train": "train", "dev": "validation", "test": "test", "test-amb": "test"}


def test_overlap_reasons():
    ref = splits.Reference([{
        "group_id": "v1", "original_text": "I'm Bryan Lynn.",
        "tokens": "we went to the big market and bought apples".split(),
    }])

    def mk(g, text, toks):
        return {"group_id": g, "original_text": text, "tokens": toks.split()}

    assert splits.overlap_reason(mk("v1", "x", "a b"), ref) == "shared_video"
    assert splits.overlap_reason(mk("v2", "I'm Bryan Lynn.", "i'm bryan lynn"), ref) == "identical_text_other_video"
    assert (
        splits.overlap_reason(mk("v3", "y", "so we went to the big market and bought apples"), ref)
        == "near_duplicate_other_video"
    )
    assert splits.overlap_reason(mk("v4", "z", "a completely different sentence about cars"), ref) is None


# --- end-to-end release on a tiny synthetic fixture -----------------------------------------------------------------

FIXTURE = {
    "train": {"aaaaaaaaaaa-0": "JAKE: Hello, everyone. Welcome to the show!",
              "aaaaaaaaaaa-1": "We talk about the U.S. economy; it is big -- really big.",
              "aaaaaaaaaaa-2": "Is NetApp really?! (Laughs) Okay.",
              "bbbbbbbbbbb-0": 'In our draft we say "?Es suyo?" here.'},
    "dev": {"aaaaaaaaaaa-0": "Is this the same video as in train?", "ddddddddddd-0": "A brand new video, right?"},
    "test": {"ccccccccccc-0": "The FDA met on Monday, I think."},
    "test-amb": {"ccccccccccc-0": "Same video, different split... okay.", "eeeeeeeeeee-0": "Something else entirely."},
}


def write_fixture(root):
    for split, utts in FIXTURE.items():
        os.makedirs(root / split / "text")
        with open(root / split / "utt2spk", "w") as f:
            f.writelines(f"{u} {u.rsplit('-', 1)[0]}\n" for u in utts)
        for u, text in utts.items():
            (root / split / "text" / f"{u}.txt").write_text(text, encoding="utf-8")
    (root / "README.txt").write_text("LICENSE\nCC BY-NC 4.0\n", encoding="utf-8")
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}\n"
             for p in sorted(root.rglob("*")) if p.is_file()]
    (root / "MANIFEST.sha256").write_text("".join(lines), encoding="utf-8", newline="\n")
    return root / "MANIFEST.sha256"


def build(tmp_path, name, titles=None):
    raw = tmp_path / "raw"
    manifest = raw / "MANIFEST.sha256" if raw.exists() else write_fixture(raw)
    out = tmp_path / name
    sources = release.ReleaseSources(str(raw), str(out / "processed"), str(out / "splits"), "fixture",
                                     raw_manifest=str(manifest), titles_path=titles)
    release.build_release_from(sources, log=lambda _: None)
    return out / "processed" / "fixture", out / "splits" / "fixture"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_release_views_are_aligned_valid_and_complete(tmp_path):
    rel, _ = build(tmp_path, "run1")
    stats = json.loads((rel / "stats.json").read_text())
    assert stats["checks"]["file_errors"] == {} and stats["checks"]["view_mismatches"] == {}
    assert stats["checks"]["raw_mismatch_before"] == [] and stats["checks"]["raw_mismatch_after"] == []
    views = {m: release.load_records(rel / f"mode_{m}" / "train.jsonl", LS) for m in "ABC"}
    native = release.load_records(rel / "benchmark_native" / "mode_A" / "train.jsonl", LS)
    assert [r["sample_id"] for r in views["A"]] == [r["sample_id"] for r in native]
    assert all(r["tokens"] == t["tokens"] for r, t in zip(views["A"], views["B"], strict=False))
    a = {r["sample_id"]: r for r in views["A"]}
    b = {r["sample_id"]: r["punct_labels"] for r in views["B"]}
    n = {r["sample_id"]: r["punct_labels"] for r in native}
    sid = "sponspeech:train:aaaaaaaaaaa-1"
    assert b[sid] == ["O", "O", "O", "O", "O", "SEMICOLON", "O", "O", "DASH", "O", "PERIOD"]
    assert a[sid]["punct_labels"][8] == "COMMA" and n[sid][8] == "O"          # canonical vs native at the dash
    assert a[sid]["metadata"]["annotation_view"] == "canonical" and native[0]["metadata"]["annotation_view"] == "native"
    first = views["C"][0]
    assert first["tokens"][0] == "hello" and first["case_labels"][0] == "FIRST_UPPER"   # 'JAKE:' was removed
    amb = release.load_records(rel / "mode_A" / "test_amb.jsonl", LS)
    assert {r["split"] for r in amb} == {"test"} and {r["metadata"]["source_split"] for r in amb} == {"test-amb"}


def test_rejected_and_unresolved_records_are_kept_out_of_default_lists(tmp_path):
    rel, spl = build(tmp_path, "run1")
    rejected = read_jsonl(rel / "audit" / "rejected.jsonl")
    assert [r["sample_id"] for r in rejected] == ["sponspeech:train:bbbbbbbbbbb-0"]
    assert len((spl / "split_map.tsv").read_text().splitlines()) == 1 + sum(len(v) for v in FIXTURE.values())
    unresolved = {r["sample_id"]: r for r in read_jsonl(rel / "audit" / "unresolved.jsonl")}
    default_a = set(release.read_ids(spl / "default_train" / "mode_A.ids"))
    assert "mark_cluster" in unresolved["sponspeech:train:aaaaaaaaaaa-2"]["unresolved_by_mode"]["A"]  # '?!' not covered
    assert "sponspeech:train:aaaaaaaaaaa-2" not in default_a
    assert "sponspeech:train:aaaaaaaaaaa-1" in default_a                         # ';' and '--' are approved rules


def test_case_masks_cover_mixed_case_positions_only(tmp_path):
    rel, spl = build(tmp_path, "run1")
    masks = {r["sample_id"]: r["excluded_positions"] for r in read_jsonl(rel / "masks" / "case_joint" / "train.jsonl")}
    assert masks["sponspeech:train:aaaaaaaaaaa-2"] == [1]                        # 'NetApp'
    assert masks["sponspeech:train:aaaaaaaaaaa-0"] == []
    assert len(masks) == 3                                                       # one line per released record


def test_primary_eval_uses_frozen_leakage_controlled_subsets(tmp_path):
    rel, spl = build(tmp_path, "run1")
    assert release.read_ids(spl / "leakage_controlled" / "validation.ids") == ["sponspeech:dev:ddddddddddd-0"]
    # test-amb is checked against train + validation; sharing a video with test is reported, not filtered (D6)
    assert release.read_ids(spl / "leakage_controlled" / "test_amb.ids") == ["sponspeech:test-amb:ccccccccccc-0",
                                                                            "sponspeech:test-amb:eeeeeeeeeee-0"]
    report = json.loads((spl / "leakage_controlled" / "report.json").read_text())
    assert report["overlap"]["leakage_controlled:test_amb_vs_test"]["shared_videos"] == 1
    excluded = (spl / "leakage_controlled" / "validation_excluded.tsv").read_text()
    assert "sponspeech:dev:aaaaaaaaaaa-0\tshared_video" in excluded
    assert release.read_ids(spl / "primary_eval" / "validation_mode_A.ids") == ["sponspeech:dev:ddddddddddd-0"]


def test_release_is_byte_reproducible(tmp_path):
    rel1, spl1 = build(tmp_path, "run1")
    rel2, spl2 = build(tmp_path, "run2")
    assert (rel1 / "SHA256SUMS").read_text() == (rel2 / "SHA256SUMS").read_text()
    assert (spl1 / "SHA256SUMS").read_text() == (spl2 / "SHA256SUMS").read_text()


def test_manifest_records_policy_license_and_views(tmp_path):
    rel, _ = build(tmp_path, "run1")
    manifest = json.loads((rel / "release_manifest.json").read_text())
    assert manifest["policy_version"] == "m1-v2" and manifest["source"]["license"] == "CC BY-NC 4.0"
    assert set(manifest["annotation_views"]) == {"canonical", "native"}
    assert (rel / "SPONSPEECH_LICENSE_AND_ATTRIBUTION.txt").exists() and (rel / "RELEASE_NOTES.md").exists()


def test_manifest_lists_every_release_and_split_file(tmp_path):
    rel, spl = build(tmp_path, "run1")
    manifest = json.loads((rel / "release_manifest.json").read_text())
    on_disk = (
        {p.relative_to(rel).as_posix() for p in rel.rglob("*") if p.is_file()} - {"release_manifest.json", "SHA256SUMS"}
    )
    assert set(manifest["files"]) == on_disk
    assert "masks/case_joint/train.jsonl" in manifest["files"] and "audit/unresolved.jsonl" in manifest["files"]
    for path, info in manifest["files"].items():
        assert info["sha256"] == hashlib.sha256((rel / path).read_bytes()).hexdigest()
    assert set(manifest["split_files"]) == (
        {p.relative_to(spl).as_posix() for p in spl.rglob("*") if p.is_file()} - {"SHA256SUMS"}
    )


def test_metadata_snapshot_flags_tts_videos_and_changes_id_lists(tmp_path):
    titles = tmp_path / "titles.jsonl"
    titles.write_text(json.dumps({"video_id": "aaaaaaaaaaa", "status": 200,
                                  "title": "Some Topic | Wikipedia audio article"}) + "\n", encoding="utf-8")
    rel, spl = build(tmp_path, "with_titles", titles=str(titles))
    _, spl_plain = build(tmp_path, "without_titles")
    snap = json.loads((rel / "release_manifest.json").read_text())["provenance"]["metadata_snapshot"]
    assert snap["tts_videos"] == ["aaaaaaaaaaa"] and snap["sha256"] == hashlib.sha256(titles.read_bytes()).hexdigest()
    assert not any(
        i.startswith("sponspeech:train:aaaaaaaaaaa") for i in release.read_ids(spl / "default_train" / "mode_A.ids")
    )
    assert (
        (spl / "default_train" / "mode_A.ids").read_text() != (spl_plain / "default_train" / "mode_A.ids").read_text()
    )


# --- source extraction (synthetic archive; the real ranges are pinned in extraction.py) -----------------------------

def make_archive(tmp_path):
    """A small tar.gz laid out like SponSpeech: utt2spk first, then >32 KiB of audio, then text files at the end."""
    import gzip
    import io
    import tarfile
    members = [("sponspeech/train/utt2spk", b"aaaaaaaaaaa-0 aaaaaaaaaaa\n"),
               ("sponspeech/train/audio/aaaaaaaaaaa-0.wav", os.urandom(40000)),
               ("sponspeech/train/text/aaaaaaaaaaa-0.txt", b"Hello, everyone."),
               ("sponspeech/README.txt", b"LICENSE CC BY-NC 4.0")]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size, info.mtime = len(data), 0
            tf.addfile(info, io.BytesIO(data))
    gz = gzip.compress(buf.getvalue(), mtime=0)          # 10-byte header, deflate data starts at byte 10, bit 0
    raw = tmp_path / "expected"
    for name, data in members:
        if not name.endswith(".wav"):
            path = raw / "sponspeech_text" / name[len("sponspeech/"):]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    manifest = "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(raw).as_posix()}\n"
                       for p in sorted((raw / "sponspeech_text").rglob("*")) if p.is_file())
    return gz, manifest


def pin(monkeypatch, gz):
    sha = hashlib.sha256(gz).hexdigest()
    from punct_restore.data import extraction
    monkeypatch.setattr(extraction, "HEAD_RANGE", {"start": 0, "length": len(gz), "sha256": sha})
    monkeypatch.setattr(extraction, "TEXT_RANGES", [{"name": "all", "start": 0, "length": len(gz), "deflate_byte": 10,
                                                     "bit": 0, "sha256": sha}])
    return extraction


def fake_fetch(gz):
    def fetch(start, length, sha):
        body = gz[start:start + length]
        if hashlib.sha256(body).hexdigest() != sha:
            from punct_restore.data.extraction import SourceChanged
            raise SourceChanged("checksum differs")
        return body
    return fetch


def test_range_extraction_recreates_files_that_match_the_manifest(tmp_path, monkeypatch):
    gz, manifest = make_archive(tmp_path)
    ex = pin(monkeypatch, gz)
    out = tmp_path / "raw"
    out.mkdir()
    (out / "MANIFEST.sha256").write_text(manifest)
    assert ex.extract_ranges(str(out), fetch=fake_fetch(gz), log=lambda _: None) == 3
    assert ex.verify_extraction(str(out), str(out / "MANIFEST.sha256")) == ([], [])
    assert not (out / "sponspeech_text" / "train" / "audio").exists()


def test_range_extraction_stops_when_the_source_changes(tmp_path, monkeypatch):
    gz, _ = make_archive(tmp_path)
    ex = pin(monkeypatch, gz)
    with pytest.raises(ex.SourceChanged):
        ex.extract_ranges(str(tmp_path / "raw"), fetch=fake_fetch(gz[:-1] + b"x"), log=lambda _: None)
    monkeypatch.setattr(
        ex, "_http", lambda url, headers=None, method="GET": ({"content-length": "1", "x-goog-generation": "2"}, b"")
    )
    with pytest.raises(ex.SourceChanged):
        ex.check_remote()


def test_full_archive_fallback_and_manifest_check(tmp_path, monkeypatch):
    gz, manifest = make_archive(tmp_path)
    from punct_restore.data import extraction as ex
    archive = tmp_path / "sponspeech.tar.gz"
    archive.write_bytes(gz)
    out = tmp_path / "raw"
    out.mkdir()
    (out / "MANIFEST.sha256").write_text(manifest)
    with pytest.raises(ex.SourceChanged):                # wrong size: not the official archive
        ex.extract_archive(str(archive), str(out), log=lambda _: None)
    monkeypatch.setattr(ex, "ARCHIVE_BYTES", len(gz))
    assert ex.extract_archive(str(archive), str(out), log=lambda _: None) == 3
    assert ex.verify_extraction(str(out), str(out / "MANIFEST.sha256")) == ([], [])
    (out / "sponspeech_text" / "README.txt").write_bytes(b"changed")
    assert ex.verify_extraction(str(out), str(out / "MANIFEST.sha256"))[0] == ["sponspeech_text/README.txt"]
