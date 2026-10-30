"""Shared script helpers of src/punct_restore/prediction_io.py: one-line errors, path checks, JSONL reads, stdout."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from punct_restore import prediction_io

# --- one_line_error ------------------------------------------------------------------------------


def test_one_line_error_joins_a_multi_line_message_with_single_spaces() -> None:
    assert prediction_io.one_line_error(ValueError("first\nsecond\r\nthird")) == "first second third"


def test_one_line_error_names_an_os_error_file_by_name_only() -> None:
    exc = OSError(28, "No space left on device", str(Path("some") / "deep" / "out.jsonl"))
    assert prediction_io.one_line_error(exc) == "out.jsonl: No space left on device"


def test_one_line_error_accepts_a_bytes_file_name() -> None:
    assert prediction_io.one_line_error(OSError(2, "No such file", b"dir/x.json")) == "x.json: No such file"


def test_one_line_error_uses_the_reason_alone_when_the_os_error_has_no_file() -> None:
    assert prediction_io.one_line_error(OSError(13, "Permission denied")) == "Permission denied"


def test_one_line_error_falls_back_to_the_text_of_an_os_error_without_a_reason() -> None:
    assert prediction_io.one_line_error(OSError("plain\ntext")) == "plain text"


# --- check_script_paths --------------------------------------------------------------------------


def _inputs(tmp_path: Path) -> dict[str, Path]:
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    first.write_text("{}\n", encoding="utf-8")
    second.write_text("{}\n", encoding="utf-8")
    return {"--first": first, "--second": second}


def test_check_script_paths_accepts_new_outputs_and_existing_inputs(tmp_path: Path) -> None:
    prediction_io.check_script_paths({"--out": tmp_path / "new" / "o.jsonl"}, _inputs(tmp_path))


def test_check_script_paths_refuses_an_existing_output_and_names_it_by_name(tmp_path: Path) -> None:
    out = tmp_path / "o.jsonl"
    out.write_text("keep me", encoding="utf-8")
    with pytest.raises(ValueError, match=r"^--out: o\.jsonl already exists, refusing to overwrite it$"):
        prediction_io.check_script_paths({"--out": out}, _inputs(tmp_path))
    assert out.read_text(encoding="utf-8") == "keep me"


def test_check_script_paths_refuses_a_dangling_symlink_as_existing(tmp_path: Path) -> None:
    link = tmp_path / "link.jsonl"
    try:
        link.symlink_to(tmp_path / "nowhere.jsonl")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available here")
    with pytest.raises(ValueError, match=r"link\.jsonl already exists"):
        prediction_io.check_script_paths({"--out": link}, _inputs(tmp_path))


def test_check_script_paths_requires_every_input_to_be_an_existing_file(tmp_path: Path) -> None:
    inputs = {**_inputs(tmp_path), "--gold": tmp_path / "absent.jsonl"}
    with pytest.raises(ValueError, match=r"^--gold: absent\.jsonl is not an existing file$"):
        prediction_io.check_script_paths({"--out": tmp_path / "o.jsonl"}, inputs)


def test_check_script_paths_rejects_a_folder_input(tmp_path: Path) -> None:
    folder = tmp_path / "folder"
    folder.mkdir()
    with pytest.raises(ValueError, match=r"--first: folder is not an existing file"):
        prediction_io.check_script_paths({"--out": tmp_path / "o.jsonl"}, {"--first": folder})


def test_check_script_paths_reports_collisions_before_anything_else(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    with pytest.raises(ValueError, match="--out and --first are the same file"):
        prediction_io.check_script_paths({"--out": inputs["--first"]}, inputs)


def test_check_script_paths_reports_an_existing_output_before_a_missing_input(tmp_path: Path) -> None:
    out = tmp_path / "o.jsonl"
    out.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        prediction_io.check_script_paths({"--out": out}, {"--gold": tmp_path / "absent.jsonl"})


# --- read_prediction_lines -----------------------------------------------------------------------


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def test_read_prediction_lines_yields_line_numbers_and_objects_skipping_blank_lines(tmp_path: Path) -> None:
    path = _write(tmp_path / "p.jsonl", '\n{"a": 1}\n   \n{"b": 2}\n')
    assert list(prediction_io.read_prediction_lines(path)) == [(2, {"a": 1}), (4, {"b": 2})]


def test_read_prediction_lines_accepts_a_leading_byte_order_mark_like_the_evaluator(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + b'{"a": 1}\n')
    assert list(prediction_io.read_prediction_lines(path)) == [(1, {"a": 1})]


def test_read_prediction_lines_is_lazy_so_a_caller_can_stop_at_its_own_error(tmp_path: Path) -> None:
    path = _write(tmp_path / "p.jsonl", '{"a": 1}\n{not json\n')
    lines = prediction_io.read_prediction_lines(path)
    assert next(lines) == (1, {"a": 1})
    with pytest.raises(ValueError, match=r"^p\.jsonl line 2: not valid JSON$"):
        next(lines)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("{not json\n", r"^p\.jsonl line 1: not valid JSON$"),
        ("[" * 5000 + "]" * 5000 + "\n", r"^p\.jsonl line 1: not valid JSON$"),  # RecursionError in json
        ('{"a": 1}\n\n[1, 2]\n', r"^p\.jsonl line 3: not a JSON object$"),
        ("null\n", r"^p\.jsonl line 1: not a JSON object$"),
        ("", r"^p\.jsonl: no predictions found$"),
        ("\n   \n", r"^p\.jsonl: no predictions found$"),
    ],
)
def test_read_prediction_lines_names_the_file_and_line_in_one_line_errors(
    tmp_path: Path, text: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        list(prediction_io.read_prediction_lines(_write(tmp_path / "p.jsonl", text)))


def test_read_prediction_lines_rejects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    path.write_bytes(b'{"a": 1}\n\xe9\xff\n')
    with pytest.raises(ValueError, match=r"^p\.jsonl: not valid UTF-8$"):
        list(prediction_io.read_prediction_lines(path))


def test_read_prediction_lines_reports_a_missing_file_as_an_os_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(prediction_io.read_prediction_lines(tmp_path / "ghost.jsonl"))


# --- reconfigure_stdout --------------------------------------------------------------------------


def test_reconfigure_stdout_makes_unencodable_text_print_as_escapes(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="ascii", errors="strict", newline="\n")
    monkeypatch.setattr(sys, "stdout", stream)
    prediction_io.reconfigure_stdout()
    print("café")
    stream.flush()
    assert raw.getvalue() == b"caf\\xe9\n"


def test_reconfigure_stdout_leaves_a_stdout_without_reconfigure_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)
    prediction_io.reconfigure_stdout()
    print("ok")
    assert buffer.getvalue() == "ok\n"
