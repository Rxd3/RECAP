"""punct_restore.cli.prepare_data: main(argv) returns 0 on success and keeps its one-line failure exit."""

from __future__ import annotations

from pathlib import Path

import pytest

from punct_restore.cli import prepare_data


def _fake_extraction(monkeypatch: pytest.MonkeyPatch, result: tuple[list[str], list[str]]) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(prepare_data.extraction, "extract_ranges", calls.append)
    monkeypatch.setattr(prepare_data.extraction, "verify_extraction", lambda root, manifest: result)
    return calls


def test_extract_passes_argv_through_and_returns_zero(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls = _fake_extraction(monkeypatch, ([], []))
    assert prepare_data.main(["extract", "--raw-root", str(tmp_path)]) == 0
    assert calls == [str(tmp_path)]
    assert capsys.readouterr().out == "extraction matches MANIFEST.sha256\n"


def test_a_mismatched_extraction_still_exits_with_its_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_extraction(monkeypatch, (["a.txt"], []))
    with pytest.raises(SystemExit) as stop:
        prepare_data.main(["extract", "--raw-root", str(tmp_path)])
    assert str(stop.value.code).startswith("extraction does not match MANIFEST.sha256: 1 missing/different, 0 extra")
