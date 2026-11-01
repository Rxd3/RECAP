"""Tests for the `recap` dispatcher (punct_restore.cli): command table, help listing and routing."""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from pathlib import Path

import pytest

from punct_restore.cli import COMMANDS, dispatch, main

CLI_MODULES = (
    "train_crf", "build_leakage_subsets", "build_word_clusters", "evaluate_models", "factor_predictions",
    "find_gold_noise", "package_hmm_handoff", "predict_hmm", "prepare_data", "project_joint", "run_ablations",
    "run_demo", "run_hmm_experiments", "simulate_asr_noise", "train_hidden_event_lm", "train_hmm",
    "tune_threshold", "verify_hmm_experiments",
)
REPORTING_MODULES = (
    "asr_noise_table", "build_ablation_table", "build_m3_final_sections", "build_m3_tables", "crf_convergence",
    "length_position", "time_models",
)
EXPECTED_COMMANDS = {
    **{name.replace("_", "-"): f"punct_restore.cli.{name}" for name in CLI_MODULES},
    **{name.replace("_", "-"): f"punct_restore.reporting.{name}" for name in REPORTING_MODULES},
}
NOT_AVAILABLE = "(not available)"


def test_no_argument_lists_every_command(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in COMMANDS)


def test_help_flags_match_no_argument(capsys):
    assert main(["-h"]) == 0 and main(["--help"]) == 0


def test_unknown_command_exits_2_with_one_stderr_line(capsys):
    assert main(["nope"]) == 2
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "nope" in err and "recap --help" in err
    assert err.startswith("error: unknown command ")


def test_dispatch_passes_args_and_returns_the_code(monkeypatch):
    fake = types.SimpleNamespace(main=lambda argv: 7 if argv == ["--x", "1"] else 0, __doc__="Fake.")
    monkeypatch.setitem(sys.modules, "fake_cmd", fake)
    assert dispatch(["fake", "--x", "1"], {"fake": "fake_cmd"}) == 7


def test_the_command_map_is_read_only():
    with pytest.raises(TypeError):
        COMMANDS["x"] = "y"


def test_command_names_follow_the_module_names():
    assert all(name == mod.rsplit(".", 1)[1].replace("_", "-") for name, mod in COMMANDS.items())


# ---- extra coverage --------------------------------------------------------------------------------------------


@pytest.fixture
def make_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[str, str], str]:
    """Write `<name>.py` into a folder put on the import path and return the module name."""
    monkeypatch.syspath_prepend(str(tmp_path))

    def write(name: str, source: str) -> str:
        (tmp_path / f"{name}.py").write_text(source, encoding="utf-8")
        return name

    return write


def _listing(capsys: pytest.CaptureFixture[str], commands: dict[str, str]) -> list[str]:
    assert dispatch([], commands) == 0
    return capsys.readouterr().out.splitlines()


def _line_for(lines: list[str], name: str) -> str:
    return next(line for line in lines if line.split()[:1] == [name])


def test_there_are_25_commands_named_after_their_modules():
    assert len(COMMANDS) == 25 and dict(COMMANDS) == EXPECTED_COMMANDS


def test_the_listing_has_a_usage_line(capsys):
    assert "usage: recap <command> [args]" in _listing(capsys, dict(COMMANDS))[0]


def test_every_help_form_prints_the_same_listing(capsys):
    outputs = []
    for argv in ([], ["-h"], ["--help"]):
        assert main(argv) == 0
        outputs.append(capsys.readouterr().out)
    assert outputs[0] == outputs[1] == outputs[2]


def test_a_command_whose_module_is_missing_shows_not_available(capsys):
    lines = _listing(capsys, {"ghost": "punct_restore.cli.no_such_module", "orphan": "no_such_pkg.mod"})
    assert NOT_AVAILABLE in _line_for(lines, "ghost") and NOT_AVAILABLE in _line_for(lines, "orphan")


def test_the_listing_shows_the_first_docstring_line_without_importing_the_module(capsys, make_module):
    name = make_module("lazy_doc_mod", '"""First line.\n\nSecond paragraph."""\nraise RuntimeError("imported")\n')
    lines = _listing(capsys, {"lazy": name})
    assert "First line." in _line_for(lines, "lazy") and "Second paragraph" not in "\n".join(lines)
    assert name not in sys.modules


def test_a_module_without_a_docstring_is_listed_with_an_empty_summary(capsys, make_module):
    lines = _listing(capsys, {"bare": make_module("bare_mod", "VALUE = 1\n")})
    assert _line_for(lines, "bare").strip() == "bare" and NOT_AVAILABLE not in "\n".join(lines)


def test_a_module_with_unparsable_source_shows_not_available(capsys, make_module):
    lines = _listing(capsys, {"broken": make_module("broken_mod", "def (:\n")})
    assert NOT_AVAILABLE in _line_for(lines, "broken")


def test_a_namespace_package_without_a_source_file_shows_not_available(capsys, tmp_path, monkeypatch):
    (tmp_path / "nspkg").mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    assert NOT_AVAILABLE in _line_for(_listing(capsys, {"ns": "nspkg"}), "ns")


def test_a_module_without_a_spec_shows_not_available(capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "spec_less_cmd", types.SimpleNamespace(main=lambda argv: 0))
    assert NOT_AVAILABLE in _line_for(_listing(capsys, {"specless": "spec_less_cmd"}), "specless")


def test_only_the_chosen_command_is_imported(monkeypatch):
    fake = types.SimpleNamespace(main=lambda argv: 0)
    monkeypatch.setitem(sys.modules, "fake_cmd", fake)
    assert dispatch(["fake"], {"fake": "fake_cmd", "never": "no_such_pkg.never_imported"}) == 0


def test_system_exit_from_a_command_passes_through(monkeypatch):
    def leave(argv: list[str]) -> int:
        raise SystemExit(0)

    monkeypatch.setitem(sys.modules, "exiting_cmd", types.SimpleNamespace(main=leave))
    with pytest.raises(SystemExit) as caught:
        dispatch(["exit"], {"exit": "exiting_cmd"})
    assert caught.value.code == 0


def test_main_without_argv_reads_sys_argv(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["recap", "nope"])
    assert main() == 2 and "nope" in capsys.readouterr().err


def test_main_with_an_empty_list_ignores_sys_argv(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["recap", "nope"])
    assert main([]) == 0 and capsys.readouterr().err == ""


@pytest.mark.parametrize("name", ["nope", "-x", "--helpme"])
def test_an_unknown_command_prints_nothing_on_stdout(capsys, name):
    assert main([name]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and name in captured.err and captured.err.count("\n") == 1


@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_every_command_answers_help(name, capsys):
    with pytest.raises(SystemExit) as stop:
        main([name, "--help"])
    assert stop.value.code == 0 and capsys.readouterr().out.startswith(f"usage: recap {name}")
