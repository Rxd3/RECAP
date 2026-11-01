"""Packaging a verified HMM study: the handoff folder, its checksums and ZIP, the refusals and the cleanup."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from hmm_study_fixtures import (
    POLICY,
    PROCESSED,
    RELEASE_ID,
    SPLITS_DIR,
    Study,
    copy_study,
    finished_study,
    patch_study,
    read_json,
    rewrite_json,
    sha256,
    write_text,
)

from punct_restore.cli import package_hmm_handoff

HANDOFF_ID = "handoff-1"
OUTSIDE = "handoff must be a separate directory inside artifacts/"
EXISTS = "handoff/archive already exists; use a new immutable handoff ID"
UNVERIFIED = "a matching complete independent verification is required"
SUMMARY = "summary does not match the actual scored reports"
UNCHECKED = "not every search candidate was independently checked"
SUM_LINE = re.compile(r"([0-9a-f]{64})  (\S.*)")
TOP_LEVEL = {"experiment", "data", "config", "tools", "HMM_RESULTS.md", "HMM_RUNBOOK.md", "HANDOFF.md",
             "MODEL_INDEX.json", "SHA256SUMS"}
DATA_ENTRIES = {"splits", "case_masks", "release_manifest.json", "stats.json", "SHA256SUMS",
                "SPONSPEECH_LICENSE_AND_ATTRIBUTION.txt"}
Edit = Callable[[dict[str, Any]], dict[str, Any]]


@pytest.fixture
def study(tmp_path_factory: pytest.TempPathFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Study:
    """A private copy of the finished study with the study code patched to it."""
    copy = copy_study(finished_study(tmp_path_factory), tmp_path)
    patch_study(monkeypatch, copy)
    return copy


@dataclass(frozen=True)
class Built:
    """One finished packaging run: where things are, what the run folder held before and what package() returned."""

    study: Study
    output: Path
    archive: Path
    checksum: Path
    run_hashes: dict[str, str]
    result: dict[str, Any]


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Built:
    """One packaging run of a private copy of the finished study; the tests that use it only read the result."""
    copy = copy_study(finished_study(tmp_path_factory), tmp_path_factory.mktemp("built"))
    with pytest.MonkeyPatch.context() as monkeypatch:
        patch_study(monkeypatch, copy)
        mark_training_summaries(copy)
        before = hashes(copy.run)
        return Built(copy, *paths(copy), before, build(copy))


# ---- helpers ----


def paths(study: Study) -> tuple[Path, Path, Path]:
    """The handoff folder, its ZIP and the ZIP's checksum file."""
    output = study.root / "artifacts" / HANDOFF_ID
    return output, output.with_name(f"{HANDOFF_ID}.zip"), output.with_name(f"{HANDOFF_ID}.zip.sha256")


def mark_training_summaries(study: Study) -> None:
    """Give each mode's winner a training summary of its own (the unit modes' real ones are identical)."""
    for mode, winner in read_json(study.run / "selection.json")["modes"].items():
        run_json = (study.run / winner["model"]).parent / "run.json"
        rewrite_json(run_json, lambda run, mode=mode: {**run, "training_summary": {"marker": mode}})


def build(study: Study, output: Path | None = None) -> dict[str, Any]:
    return package_hmm_handoff.package(study.run, study.evaluator, output or paths(study)[0])


def cli_args(study: Study, output: Path) -> list[str]:
    return ["--run-dir", str(study.run), "--evaluator-root", str(study.evaluator), "--output-dir", str(output)]


def tree(folder: Path) -> set[str]:
    """Every file and folder under `folder`, as relative paths."""
    return {path.relative_to(folder).as_posix() for path in folder.rglob("*")}


def hashes(folder: Path) -> dict[str, str]:
    """{relative path: SHA-256} of every file under `folder`."""
    return {path.relative_to(folder).as_posix(): sha256(path) for path in folder.rglob("*") if path.is_file()}


def parse_sums(text: str) -> dict[str, str]:
    """{relative path: SHA-256} of a SHA256SUMS text, where every line must read '<64 hex digits>  <path>'."""
    assert text.endswith("\n")
    entries: dict[str, str] = {}
    for line in text.splitlines():
        match = SUM_LINE.fullmatch(line)
        assert match, line
        assert match[2] not in entries, f"{match[2]} is listed twice"
        entries[match[2]] = match[1]
    return entries


def refuse(study: Study, message: str, output: Path | None = None, *, fresh: bool = True) -> None:
    """package() raises exactly `message` and leaves the whole fake repository as it was."""
    before = tree(study.root)
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        build(study, output)
    assert tree(study.root) == before
    if fresh:
        assert not any(path.exists() for path in paths(study))


def refuse_each(study: Study, subtests: pytest.Subtests, file: Path, edits: dict[str, Edit], message: str) -> None:
    """Each named edit of the JSON `file`, applied to its original content, is refused with `message`."""
    original = file.read_bytes()
    for name, edit in edits.items():
        with subtests.test(msg=name):
            rewrite_json(file, edit)
            refuse(study, message)
        file.write_bytes(original)


def in_summary(mode: str, split: str, field: str, change: Callable[[Any], Any]) -> Edit:
    """An edit of results.json that applies `change` to one field of the summary of (mode, split)."""

    def edit(results: dict[str, Any]) -> dict[str, Any]:
        summary = results["results"][mode][split]
        new = {**summary, field: change(summary[field])}
        return {**results, "results": {**results["results"], mode: {**results["results"][mode], split: new}}}

    return edit


def write_zip(archive_path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)


# ---- A. the happy path ----


def test_result_describes_the_archive(built: Built) -> None:
    sums = (built.output / "SHA256SUMS").read_text(encoding="utf-8")
    assert built.result == {
        "archive": {"path": "artifacts/handoff-1.zip", "sha256": sha256(built.archive)},
        "bytes": built.archive.stat().st_size,
        "checksummed_files": len(sums.splitlines()),
        "archive_roundtrip": "PASS",
    }
    assert built.result["checksummed_files"] > 100


def test_handoff_folder_holds_exactly_the_expected_parts(built: Built) -> None:
    assert {path.name for path in built.output.iterdir()} == TOP_LEVEL
    assert {path.name for path in (built.output / "data").iterdir()} == DATA_ENTRIES
    assert (built.output / "HANDOFF.md").read_text(encoding="utf-8").startswith("# Member 2 HMM handoff\n")


def test_experiment_folder_is_a_byte_identical_copy_and_the_run_is_left_unchanged(built: Built) -> None:
    assert "plan.json" in built.run_hashes and "verification/verification.json" in built.run_hashes
    assert hashes(built.output / "experiment") == built.run_hashes
    assert hashes(built.study.run) == built.run_hashes


def test_data_config_notes_and_tool_are_copies_of_their_repository_sources(built: Built) -> None:
    root = built.study.root
    assert hashes(built.output / "data/splits") == hashes(root / SPLITS_DIR) != {}
    assert hashes(built.output / "data/case_masks") == hashes(root / PROCESSED / "masks/case_joint") != {}
    sources = {f"data/{name}": root / PROCESSED / name for name in DATA_ENTRIES - {"splits", "case_masks"}}
    sources |= {f"config/{name}": root / "configs" / name for name in ("hmm_sponspeech_v2.yaml", "label_sets.yaml")}
    sources |= {
        "config/requirements-hmm.txt": root / "requirements-hmm.txt",
        "HMM_RESULTS.md": root / "experiments/HMM_RESULTS.md",
        "HMM_RUNBOOK.md": root / "experiments/HMM_RUNBOOK.md",
        "tools/package_hmm_handoff.py": Path(package_hmm_handoff.__file__),
    }
    assert len(sources) == 10
    for name, source in sources.items():
        assert (built.output / name).read_bytes() == source.read_bytes(), name


def test_sha256sums_covers_every_file_but_itself_with_matching_hashes(built: Built) -> None:
    listed = parse_sums((built.output / "SHA256SUMS").read_text(encoding="utf-8"))
    actual = hashes(built.output)
    assert "SHA256SUMS" in actual and "SHA256SUMS" not in listed
    assert listed == {name: digest for name, digest in actual.items() if name != "SHA256SUMS"}
    assert "data/SHA256SUMS" in listed


def test_zip_members_are_the_folder_files_under_one_top_folder(built: Built) -> None:
    with zipfile.ZipFile(built.archive) as archive:
        names = archive.namelist()
        in_zip = {n.removeprefix(f"{HANDOFF_ID}/"): hashlib.sha256(archive.read(n)).hexdigest() for n in names}
    assert len(names) == len(set(names))
    assert set(names) == {f"{HANDOFF_ID}/{name}" for name in hashes(built.output)}
    assert in_zip == hashes(built.output)
    listed = parse_sums((built.output / "SHA256SUMS").read_text(encoding="utf-8"))
    assert {name: digest for name, digest in in_zip.items() if name != "SHA256SUMS"} == listed


def test_zip_checksum_file_names_the_zip_and_matches_it(built: Built) -> None:
    expected = f"{sha256(built.archive)}  handoff-1.zip\n"
    assert built.checksum.read_bytes() == expected.encode("utf-8")


def test_model_index_names_the_experiment_and_each_selected_model(built: Built) -> None:
    run = built.study.run
    index = read_json(built.output / "MODEL_INDEX.json")
    assert index["schema"] == "recap-hmm-handoff-v1"
    assert (index["experiment_id"], index["dataset_release"], index["policy_version"]) == (
        "unit-hmm-study", RELEASE_ID, POLICY)
    assert index["license"].startswith("CC BY-NC 4.0")
    assert set(index["selected_models"]) == {"A", "C"}
    for mode, winner in read_json(run / "selection.json")["modes"].items():
        entry = index["selected_models"][mode]
        assert winner.items() <= entry.items()
        assert entry["path"] == "experiment/" + winner["model"]
        assert (built.output / entry["path"]).is_file()
        assert entry["training_summary"] == {"marker": mode}


def test_model_index_records_where_every_part_came_from(built: Built) -> None:
    run = built.study.run
    index = read_json(built.output / "MODEL_INDEX.json")
    plan = read_json(run / "plan.json")
    assert index["verification"] == {
        "path": "experiment/verification/verification.json",
        "sha256": sha256(run / "verification/verification.json"),
    }
    assert index["implementation"] == plan["implementation"]
    assert index["shared_evaluator"] == plan["evaluator"]
    assert index["training_code"] == plan["code"]
    assert index["packager"]["sha256"] == sha256(Path(package_hmm_handoff.__file__))
    assert set(index["packaging_code"]) == {"commit", "dirty"}
    assert index["packaging_code"] != index["training_code"]
    assert isinstance(index["created_at_utc"], str)


def test_main_prints_the_result_as_json_and_returns_zero(study: Study, capsys: pytest.CaptureFixture[str]) -> None:
    output, archive, _ = paths(study)
    code = package_hmm_handoff.main(cli_args(study, output))
    captured = capsys.readouterr()
    assert code == 0 and captured.err == ""
    printed = json.loads(captured.out)
    assert printed["archive_roundtrip"] == "PASS"
    assert printed["archive"] == {"path": "artifacts/handoff-1.zip", "sha256": sha256(archive)}


# ---- B. refusals that create nothing ----


def test_missing_verification_file_is_refused_and_main_reports_one_error_line(
        study: Study, capsys: pytest.CaptureFixture[str]) -> None:
    (study.run / "verification/verification.json").unlink()
    before = tree(study.root)
    with pytest.raises(FileNotFoundError) as caught:
        build(study)
    code = package_hmm_handoff.main(cli_args(study, paths(study)[0]))
    captured = capsys.readouterr()
    assert tree(study.root) == before
    assert code == 1 and captured.out == ""
    assert captured.err == f"error: verification.json: {caught.value.strerror}\n"


def test_a_verification_that_failed_or_does_not_match_the_run_is_refused(
        study: Study, subtests: pytest.Subtests) -> None:
    def wrong_sha(key: str) -> Edit:
        return lambda verified: {**verified, key: {**verified[key], "sha256": "0" * 64}}

    edits = {
        "result FAIL": lambda verified: {**verified, "result": "FAIL"},
        "plan sha256 changed": wrong_sha("plan"),
        "selection sha256 changed": wrong_sha("selection"),
    }
    refuse_each(study, subtests, study.run / "verification/verification.json", edits, UNVERIFIED)


def test_a_mode_with_unchecked_search_candidates_is_refused(study: Study, subtests: pytest.Subtests) -> None:
    def three_checked(mode: str) -> Edit:
        return lambda verified: {
            **verified,
            "modes": {**verified["modes"], mode: {**verified["modes"][mode], "candidates_independently_checked": 3}},
        }

    edits = {"mode A": three_checked("A"), "mode C": three_checked("C")}
    refuse_each(study, subtests, study.run / "verification/verification.json", edits, UNCHECKED)


def test_a_changed_field_of_any_split_summary_is_refused(study: Study, subtests: pytest.Subtests) -> None:
    def changed(_: Any) -> str:
        return "changed"

    edits = {
        "A test primary value": in_summary("A", "test", "primary", lambda p: {**p, "value": p["value"] + 0.25}),
        "A test counts": in_summary("A", "test", "counts", changed),
        "A test shortcut_primary": in_summary("A", "test", "shortcut_primary", changed),
        "A test evaluation_sha256": in_summary("A", "test", "evaluation_sha256", changed),
        "A validation primary": in_summary("A", "validation", "primary", changed),
        "A test_amb primary": in_summary("A", "test_amb", "primary", changed),
        "C test primary": in_summary("C", "test", "primary", changed),
    }
    refuse_each(study, subtests, study.run / "results.json", edits, SUMMARY)


@pytest.mark.parametrize("where", ["outside", "inside-run", "contains-run", "dotdot"])
def test_a_handoff_folder_that_is_not_separate_and_inside_artifacts_is_refused(study: Study, where: str) -> None:
    bad = {
        "outside": study.root / "elsewhere",
        "inside-run": study.run / "handoff",
        "contains-run": study.run.parent,
        "dotdot": study.root / "artifacts" / ".." / "elsewhere",
    }[where]
    refuse(study, OUTSIDE, bad, fresh=False)
    assert (study.run / "plan.json").is_file()


def test_main_reports_an_outside_handoff_folder_as_the_exact_error_line(
        study: Study, capsys: pytest.CaptureFixture[str]) -> None:
    code = package_hmm_handoff.main(cli_args(study, study.root / "elsewhere"))
    captured = capsys.readouterr()
    assert code == 1 and captured.out == ""
    assert captured.err == f"error: {OUTSIDE}\n"
    assert not (study.root / "elsewhere").exists()


@pytest.mark.parametrize("which", ["folder", "zip", "checksum"])
def test_an_existing_handoff_item_is_refused_and_left_untouched(study: Study, which: str) -> None:
    output, archive, checksum = paths(study)
    item = {"folder": output / "marker.txt", "zip": archive, "checksum": checksum}[which]
    write_text(item, "keep me\n")
    refuse(study, EXISTS, fresh=False)
    assert item.read_text(encoding="utf-8") == "keep me\n"


# ---- C. a failure part way removes the half-built handoff ----


def test_a_workspace_path_in_a_run_log_stops_the_build_and_the_id_stays_usable(study: Study) -> None:
    log = study.run / "mode_A/candidates/s0.1-c1.training.log"
    original = log.read_bytes()
    log.write_bytes(original + f"see {study.root}\n".encode())
    refuse(study, f"personal absolute workspace path in {log.name}")
    log.write_bytes(original)
    assert build(study)["archive_roundtrip"] == "PASS"
    assert all(path.exists() for path in paths(study))


@pytest.mark.parametrize(
    ("suffix", "spelling"),
    [(".md", "native"), (".md", "posix"), (".md", "escaped"), (".json", "native"), (".jsonl", "native"),
     (".yaml", "native"), (".txt", "native"), (".log", "native")],
)
def test_every_spelling_of_the_workspace_path_is_rejected_in_every_text_file_type(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str, spelling: str) -> None:
    root = tmp_path / "personal repo"
    monkeypatch.setattr(package_hmm_handoff, "ROOT", root)
    forms = {"native": str(root), "posix": root.as_posix(), "escaped": str(root).replace("\\", "\\\\")}
    write_text(tmp_path / f"handoff/sub/a{suffix}", f"see {forms[spelling]}/x\n")
    with pytest.raises(ValueError, match=f"^personal absolute workspace path in a{re.escape(suffix)}$"):
        package_hmm_handoff._bundle_files(tmp_path / "handoff")


def test_bundle_files_lists_every_file_sorted_and_does_not_decode_other_file_types(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(package_hmm_handoff, "ROOT", tmp_path / "personal repo")
    folder = tmp_path / "handoff"
    write_text(folder / "b/notes.md", "clean\n")
    write_text(folder / "a.json", "{}\n")
    (folder / "model.bin").write_bytes(b"\xff\xfe\x00 not utf-8")
    assert package_hmm_handoff._bundle_files(folder) == [folder / "a.json", folder / "b/notes.md", folder / "model.bin"]


def test_a_failed_archive_check_removes_the_folder_and_the_zip(study: Study, monkeypatch: pytest.MonkeyPatch) -> None:
    output, archive, checksum = paths(study)
    zip_was_written: list[bool] = []

    def failing_check(archive_path: Path, digests: dict[str, str], sums: str) -> None:
        zip_was_written.append(archive_path.is_file())
        raise ValueError("injected")

    monkeypatch.setattr(package_hmm_handoff, "_verify_archive", failing_check)
    with pytest.raises(ValueError, match="^injected$"):
        build(study)
    assert zip_was_written == [True]
    assert not (output.exists() or archive.exists() or checksum.exists())


def test_an_interrupt_after_the_zip_was_written_removes_everything_and_propagates(
        study: Study, monkeypatch: pytest.MonkeyPatch) -> None:
    output, archive, checksum = paths(study)
    real_write = package_hmm_handoff._write_archive
    zip_was_written: list[bool] = []

    def write_then_interrupt(folder: Path, archive_path: Path) -> None:
        real_write(folder, archive_path)
        zip_was_written.append(archive_path.is_file())
        raise KeyboardInterrupt

    monkeypatch.setattr(package_hmm_handoff, "_write_archive", write_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        build(study)
    assert zip_was_written == [True]
    assert not (output.exists() or archive.exists() or checksum.exists())


def test_a_failure_after_the_checksum_file_was_written_removes_it_too(
        study: Study, monkeypatch: pytest.MonkeyPatch) -> None:
    output, archive, checksum = paths(study)
    real_identity = package_hmm_handoff.file_identity
    checksum_was_written: list[bool] = []

    def failing_identity(path: Path) -> dict[str, str]:
        if path.suffix != ".zip":
            return real_identity(path)
        checksum_was_written.append(checksum.is_file())
        raise RuntimeError("late failure")

    monkeypatch.setattr(package_hmm_handoff, "file_identity", failing_identity)
    with pytest.raises(RuntimeError, match="^late failure$"):
        build(study)
    assert checksum_was_written == [True]
    assert not (output.exists() or archive.exists() or checksum.exists())


# ---- the archive round trip, on a tiny two-file handoff ----


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("intact", None),
        ("extra-member", "archive has missing or extra members"),
        ("missing-member", "archive has missing or extra members"),
        ("member-bytes", "archive member hash mismatch: a.txt"),
        ("checksum-index", "archive checksum index mismatch"),
    ],
)
def test_verify_archive_accepts_an_intact_archive_and_catches_every_corruption(
        tmp_path: Path, kind: str, message: str | None) -> None:
    payload = {"a.txt": b"alpha", "b/c.txt": b"gamma"}
    digests = {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}
    sums = "".join(f"{digest}  {name}\n" for name, digest in digests.items())
    members = {f"{HANDOFF_ID}/{name}": data for name, data in payload.items()}
    members[f"{HANDOFF_ID}/SHA256SUMS"] = sums.encode()
    members = {
        "intact": members,
        "extra-member": {**members, f"{HANDOFF_ID}/extra.txt": b"x"},
        "missing-member": {k: v for k, v in members.items() if not k.endswith("/c.txt")},
        "member-bytes": {**members, f"{HANDOFF_ID}/a.txt": b"changed"},
        "checksum-index": {**members, f"{HANDOFF_ID}/SHA256SUMS": members[f"{HANDOFF_ID}/SHA256SUMS"] + b"\n"},
    }[kind]
    write_zip(tmp_path / "h.zip", members)
    if message is None:
        assert package_hmm_handoff._verify_archive(tmp_path / "h.zip", digests, sums) is None
        return
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        package_hmm_handoff._verify_archive(tmp_path / "h.zip", digests, sums)
