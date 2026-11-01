"""Every script under scripts/reporting/ is an exact compatibility stub that runs from any folder."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from punct_restore.cli import COMMANDS, main
from punct_restore.paths import REPO_ROOT as ROOT

STUB = (
    '"""Compatibility stub: the code lives in {pkg}.{mod}; prefer `recap {name}`."""\n'
    "from {pkg}.{mod} import main\n\nraise SystemExit(main())\n"
)
PACKAGES = {"scripts": "punct_restore.cli", "reporting": "punct_restore.reporting"}
# Task 4 widens this to every script under scripts/ (rglob).
STUBS = sorted((ROOT / "scripts" / "reporting").glob("*.py"))
REPORTING_COMMANDS = (
    "asr_noise_table", "build_ablation_table", "build_m3_final_sections", "build_m3_tables", "crf_convergence",
    "length_position", "time_models",
)
DASHED = [name.replace("_", "-") for name in REPORTING_COMMANDS]


def _usage_line(name: str) -> str:
    return f"usage: recap {name}"


def _run_from(folder: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    """Run the current interpreter with ``argv`` inside ``folder``, so nothing can resolve against the repository."""
    return subprocess.run([sys.executable, *argv], check=False, cwd=folder, capture_output=True, text=True)


@pytest.mark.parametrize("path", STUBS, ids=str)
def test_every_script_is_an_exact_stub(path: Path) -> None:
    mod = path.stem
    expected = STUB.format(pkg=PACKAGES[path.parent.name], mod=mod, name=mod.replace("_", "-"))
    assert path.read_text(encoding="utf-8") == expected


@pytest.mark.parametrize("path", STUBS, ids=str)
def test_every_stub_answers_help_from_another_directory(path: Path, tmp_path: Path) -> None:
    done = _run_from(tmp_path, str(path), "--help")
    assert done.returncode == 0 and "usage: recap " in done.stdout
    assert _usage_line(path.stem.replace("_", "-")) in done.stdout


def test_the_reporting_stubs_are_exactly_the_reporting_commands() -> None:
    commands = {mod.rsplit(".", 1)[1] for mod in COMMANDS.values() if mod.startswith("punct_restore.reporting.")}
    assert {path.stem for path in STUBS} == commands == set(REPORTING_COMMANDS)


@pytest.mark.parametrize("name", DASHED)
def test_every_reporting_command_answers_help_through_recap(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main([name, "--help"])
    assert caught.value.code == 0
    assert _usage_line(name) in capsys.readouterr().out


@pytest.mark.parametrize("mod", REPORTING_COMMANDS)
def test_every_reporting_module_runs_with_python_dash_m(mod: str, tmp_path: Path) -> None:
    done = _run_from(tmp_path, "-m", f"punct_restore.reporting.{mod}", "--help")
    assert done.returncode == 0
    assert _usage_line(mod.replace("_", "-")) in done.stdout


def test_a_stub_run_from_another_directory_reads_the_repo_defaults(tmp_path: Path) -> None:
    """The default --scores-root is the repository's results/scores, not a folder under the current directory."""
    out = tmp_path / "ablation.md"
    stub = ROOT / "scripts" / "reporting" / "build_ablation_table.py"
    done = _run_from(tmp_path, str(stub), "--out", str(out))
    assert done.returncode == 0 and done.stderr == ""
    assert out.is_file() and out.stat().st_size > 0
