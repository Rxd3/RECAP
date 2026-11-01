"""punct_restore.cli.build_leakage_subsets on a tiny fake data root (release ``rel``, mode A)."""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from punct_restore.cli import build_leakage_subsets as subsets

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_leakage_subsets.py"
REL = "rel"
QUALITY = "low_comma_or_tts_source"

# test.jsonl order; the video is the part before "-". Train has videos A, A, Z (listed) and C (not listed).
KEYS = ["B-0", "A-0", "C-0", "A-1", "D-0", "B-1", "C-1", "A-2", "D-1", "D-2"]
KEPT_KEYS = ["D-1", "B-0", "D-0", "C-0"]  # file order differs from jsonl order on purpose
EXCLUDED_ROWS = [
    ("D-2", QUALITY),  # video D is not in train -> quality
    ("A-0", "shared_video"),  # video A is in train -> train video
    ("C-1", "shared_video"),  # video C only appears in an unlisted train record -> other leak
    ("A-2", QUALITY),  # video A is in train, but quality wins
    ("B-1", "near_duplicate_other_video"),  # -> other leak
    ("A-1", "identical_text_other_video"),  # -> train video
]
EXPECTED = {
    "all": KEYS,
    "kept": ["B-0", "C-0", "D-0", "D-1"],
    "excluded_quality": ["A-2", "D-2"],
    "excluded_train_video": ["A-0", "A-1"],
    "excluded_other_leak": ["B-1", "C-1"],
}


def _record(sample_id: str, group: str | None) -> dict:
    meta = {} if group is None else {"group_id": group}
    return {"sample_id": sample_id, "tokens": ["x"], "metadata": meta}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    _write(path, "".join(json.dumps(record) + "\n" for record in records))


def _write_lines(path: Path, rows: list[str]) -> None:
    _write(path, "".join(row + "\n" for row in rows))


def make_root(tmp_path: Path, split: str = "test") -> Path:
    """Write the fake data root (idempotent for the train files) and return it."""
    root = tmp_path / "data"
    processed = root / "processed" / REL / "mode_A"
    controlled = root / "splits" / REL / "leakage_controlled"
    _write_jsonl(processed / f"{split}.jsonl", [_record(f"{split}:{k}", k[0]) for k in KEYS])
    _write_jsonl(processed / "train.jsonl", [
        _record("train:1", "A"), _record("train:2", "A"), _record("train:3", "Z"), _record("train:4", "C")])
    _write_lines(root / "splits" / REL / "default_train" / "mode_A.ids", ["train:1", "train:2", "train:3"])
    _write_lines(controlled / f"{split}.ids", [f"{split}:{k}" for k in KEPT_KEYS])
    _write_lines(controlled / f"{split}_excluded.tsv",
                 ["sample_id\treason", *[f"{split}:{k}\t{why}" for k, why in EXCLUDED_ROWS]])
    return root


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    _write(path, text.replace(old, new, 1))


def jsonl_path(root: Path, name: str = "test") -> Path:
    return root / "processed" / REL / "mode_A" / f"{name}.jsonl"


def tsv_path(root: Path, split: str = "test") -> Path:
    return root / "splits" / REL / "leakage_controlled" / f"{split}_excluded.tsv"


def kept_path(root: Path, split: str = "test") -> Path:
    return root / "splits" / REL / "leakage_controlled" / f"{split}.ids"


def run(root: Path, out: Path, *extra: str) -> int:
    return subsets.main(["--data-root", str(root), "--release", REL, "--out", str(out), *extra])


def read_ids(out: Path, name: str) -> list[str]:
    return (out / f"{name}.ids").read_text(encoding="utf-8").splitlines()


def assert_one_error_line(capsys: pytest.CaptureFixture[str], *needles: str) -> str:
    err = capsys.readouterr().err
    assert err.startswith("error: ") and err.count("\n") == 1
    for needle in needles:
        assert needle in err
    return err


# --- success ------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(EXPECTED))
def test_each_ids_file_has_the_right_ids_in_all_order(tmp_path: Path, name: str) -> None:
    out = tmp_path / "out"
    assert run(make_root(tmp_path), out) == 0
    assert read_ids(out, name) == [f"test:{k}" for k in EXPECTED[name]]


def test_ids_files_are_lf_terminated_utf8_and_counts_json_is_exact(tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert run(make_root(tmp_path), out) == 0
    for name in EXPECTED:
        raw = (out / f"{name}.ids").read_bytes()
        assert raw.endswith(b"\n") and b"\r" not in raw
        raw.decode("utf-8")
    expected = {"release": REL, "mode": "A", "split": "test", "all": 10, "kept": 4,
                "excluded_train_video": 2, "excluded_other_leak": 2, "excluded_quality": 2}
    assert (out / "counts.json").read_bytes() == (json.dumps(expected, indent=2) + "\n").encode("utf-8")
    assert sorted(path.name for path in out.iterdir()) == sorted([*(f"{n}.ids" for n in EXPECTED), "counts.json"])


def test_prints_a_single_summary_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(make_root(tmp_path), tmp_path / "out") == 0
    captured = capsys.readouterr()
    assert captured.err == "" and captured.out.count("\n") == 1
    assert "all=10" in captured.out and "excluded_quality=2" in captured.out


def test_an_empty_bucket_writes_an_empty_file(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    edit(tsv_path(root), f"test:A-2\t{QUALITY}", "test:A-2\tshared_video")
    edit(tsv_path(root), f"test:D-2\t{QUALITY}", "test:D-2\tshared_video")
    out = tmp_path / "out"
    assert run(root, out) == 0
    assert (out / "excluded_quality.ids").read_bytes() == b""
    assert read_ids(out, "excluded_train_video") == ["test:A-0", "test:A-1", "test:A-2"]


def test_validation_split_uses_the_validation_files(tmp_path: Path) -> None:
    root = make_root(tmp_path, "test")
    make_root(tmp_path, "validation")
    out = tmp_path / "out"
    assert run(root, out, "--split", "validation") == 0
    assert read_ids(out, "kept") == [f"validation:{k}" for k in EXPECTED["kept"]]
    assert json.loads((out / "counts.json").read_text(encoding="utf-8"))["split"] == "validation"


def test_a_missing_group_id_is_fine_when_the_group_is_not_needed(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    edit(jsonl_path(root), '"sample_id": "test:B-0", "tokens": ["x"], "metadata": {"group_id": "B"}',
         '"sample_id": "test:B-0", "tokens": ["x"], "metadata": {}')  # kept
    edit(jsonl_path(root), '"sample_id": "test:D-2", "tokens": ["x"], "metadata": {"group_id": "D"}',
         '"sample_id": "test:D-2", "tokens": ["x"]')  # quality
    assert run(root, tmp_path / "out") == 0


def test_the_stub_runs_as_a_subprocess_without_pythonpath(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    out = tmp_path / "out"
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--data-root", str(root), "--release", REL, "--out", str(out)],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", timeout=60, env=env, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert read_ids(out, "kept") == [f"test:{k}" for k in EXPECTED["kept"]]


# --- refusals -----------------------------------------------------------------------------------


def test_existing_out_is_refused_before_anything_is_read(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "marker.txt").write_text("keep", encoding="utf-8")
    assert run(tmp_path / "no-such-data-root", out) == 1
    assert_one_error_line(capsys, "already exists")
    assert [path.name for path in out.iterdir()] == ["marker.txt"]
    assert (out / "marker.txt").read_text(encoding="utf-8") == "keep"


def test_an_existing_out_file_is_refused_too(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out"
    out.write_text("keep", encoding="utf-8")
    assert run(make_root(tmp_path), out) == 1
    assert_one_error_line(capsys, "already exists")
    assert out.read_text(encoding="utf-8") == "keep"


def test_tsv_missing_an_excluded_id_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = make_root(tmp_path)
    edit(tsv_path(root), "test:B-1\tnear_duplicate_other_video\n", "")
    out = tmp_path / "out"
    assert run(root, out) == 1
    assert_one_error_line(capsys, "test:B-1", "test_excluded.tsv")
    assert not out.exists()


def test_tsv_listing_a_kept_id_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = make_root(tmp_path)
    tsv_path(root).write_text(tsv_path(root).read_text(encoding="utf-8") + "test:B-0\tshared_video\n",
                              encoding="utf-8", newline="\n")
    out = tmp_path / "out"
    assert run(root, out) == 1
    assert_one_error_line(capsys, "test:B-0", "test_excluded.tsv")
    assert not out.exists()


def test_tsv_listing_an_unknown_id_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = make_root(tmp_path)
    tsv_path(root).write_text(tsv_path(root).read_text(encoding="utf-8") + "test:ZZ-9\tshared_video\n",
                              encoding="utf-8", newline="\n")
    assert run(root, tmp_path / "out") == 1
    assert_one_error_line(capsys, "test:ZZ-9", "test_excluded.tsv")


def test_tsv_listing_an_id_twice_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = make_root(tmp_path)
    tsv_path(root).write_text(tsv_path(root).read_text(encoding="utf-8") + "test:A-0\tshared_video\n",
                              encoding="utf-8", newline="\n")
    out = tmp_path / "out"
    assert run(root, out) == 1
    assert_one_error_line(capsys, "test:A-0", "test_excluded.tsv")
    assert not out.exists()


@pytest.mark.parametrize("bad_text", ["id\treason\n", "", "sample_id\treason\ntest:A-0\n", "sample_id\treason\n\t\n"])
def test_a_malformed_tsv_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str], bad_text: str) -> None:
    root = make_root(tmp_path)
    _write(tsv_path(root), bad_text)
    assert run(root, tmp_path / "out") == 1
    assert_one_error_line(capsys, "test_excluded.tsv")


def test_a_kept_id_missing_from_the_split_file_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = make_root(tmp_path)
    kept_path(root).write_text(kept_path(root).read_text(encoding="utf-8") + "test:ZZ-9\n",
                               encoding="utf-8", newline="\n")
    out = tmp_path / "out"
    assert run(root, out) == 1
    assert_one_error_line(capsys, "test:ZZ-9", "test.ids")
    assert not out.exists()


def test_a_train_id_missing_from_train_jsonl_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = make_root(tmp_path)
    ids = root / "splits" / REL / "default_train" / "mode_A.ids"
    ids.write_text(ids.read_text(encoding="utf-8") + "train:99\n", encoding="utf-8", newline="\n")
    out = tmp_path / "out"
    assert run(root, out) == 1
    assert_one_error_line(capsys, "train:99", "train.jsonl")
    assert not out.exists()


def test_a_needed_test_group_id_that_is_missing_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = make_root(tmp_path)
    edit(jsonl_path(root), '"sample_id": "test:A-0", "tokens": ["x"], "metadata": {"group_id": "A"}',
         '"sample_id": "test:A-0", "tokens": ["x"], "metadata": {"group_id": 7}')
    out = tmp_path / "out"
    assert run(root, out) == 1
    assert_one_error_line(capsys, "test:A-0", "group_id", "test.jsonl")
    assert not out.exists()


def test_a_listed_train_record_without_a_group_id_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = make_root(tmp_path)
    edit(jsonl_path(root, "train"), '"sample_id": "train:3", "tokens": ["x"], "metadata": {"group_id": "Z"}',
         '"sample_id": "train:3", "tokens": ["x"], "metadata": null')
    assert run(root, tmp_path / "out") == 1
    assert_one_error_line(capsys, "train:3", "group_id", "train.jsonl")


@pytest.mark.parametrize("bad_line, needle", [
    ("{not json", "line 3"),
    ('{"tokens": ["x"]}', "sample_id"),
    ("[1, 2]", "sample_id"),
    ('{"sample_id": "test:A-0", "tokens": []}', "duplicate"),
])
def test_a_bad_jsonl_line_fails_with_file_name(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                               bad_line: str, needle: str) -> None:
    root = make_root(tmp_path)
    lines = jsonl_path(root).read_text(encoding="utf-8").splitlines()
    _write_lines(jsonl_path(root), [*lines[:2], bad_line, *lines[2:]])
    out = tmp_path / "out"
    assert run(root, out) == 1
    assert_one_error_line(capsys, "test.jsonl", needle)
    assert not out.exists()


def test_a_missing_input_file_is_a_friendly_one_line_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out"
    assert run(tmp_path / "no-such-root", out) == 1
    err = assert_one_error_line(capsys, "test.jsonl")
    assert str(tmp_path) not in err
    assert not out.exists()


@pytest.mark.parametrize("target, name", [
    (lambda root: jsonl_path(root), "test.jsonl"),
    (lambda root: kept_path(root), "test.ids"),
    (lambda root: tsv_path(root), "test_excluded.tsv"),
    (lambda root: jsonl_path(root, "train"), "train.jsonl"),
    (lambda root: root / "splits" / REL / "default_train" / "mode_A.ids", "mode_A.ids"),
])
def test_an_input_that_is_not_utf8_fails_with_file_name(
        tmp_path: Path, capsys: pytest.CaptureFixture[str], target, name: str) -> None:
    root = make_root(tmp_path)
    target(root).write_bytes(b"\xff\xfe not utf-8\n")
    out = tmp_path / "out"
    assert run(root, out) == 1
    assert_one_error_line(capsys, f"{name}: not valid UTF-8")
    assert not out.exists()


def test_a_failed_write_leaves_no_partial_out_and_a_rerun_works(
        tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    root, out = make_root(tmp_path), tmp_path / "out"
    real_write = Path.write_text

    def fail_on_kept(self: Path, *args: object, **kwargs: object) -> int:
        if self.name == "kept.ids":  # all.ids is already written: the failure is midway
            raise OSError(errno.ENOSPC, "No space left on device", str(self))
        return real_write(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", fail_on_kept)
        assert run(root, out) == 1
    assert_one_error_line(capsys, "kept.ids", "No space left on device")
    assert not out.exists()
    assert [path.name for path in tmp_path.iterdir()] == ["data"]  # no temp folder left behind
    assert run(root, out) == 0
    assert read_ids(out, "kept") == [f"test:{k}" for k in EXPECTED["kept"]]


def test_the_module_docstring_usage_keeps_its_line_break() -> None:
    assert "\\\n" in subsets.__doc__  # a lone backslash-newline in the source would be swallowed


# --- usage --------------------------------------------------------------------------------------


@pytest.mark.parametrize("argv", [[], ["--out", "x", "--mode", "D"], ["--out", "x", "--split", "train"]])
def test_usage_errors_exit_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        subsets.main(argv)
    assert raised.value.code == 2
