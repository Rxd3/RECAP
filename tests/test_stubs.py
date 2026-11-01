"""Every script under scripts/ is an exact compatibility stub that runs from any folder."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from eval_fixtures import cli_argv, valid_case

from punct_restore.cli import COMMANDS
from punct_restore.paths import REPO_ROOT as ROOT

STUB = (
    '"""Compatibility stub: the code lives in {pkg}.{mod}; prefer `recap {name}`."""\n'
    "from {pkg}.{mod} import main\n\nraise SystemExit(main())\n"
)
PACKAGES = {"scripts": "punct_restore.cli", "reporting": "punct_restore.reporting"}
STUBS = sorted((ROOT / "scripts").rglob("*.py"))


def _usage_line(name: str) -> str:
    return f"usage: recap {name}"


def _module_of(path: Path) -> str:
    return f"{PACKAGES[path.parent.name]}.{path.stem}"


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


def test_there_is_one_stub_per_command() -> None:
    assert len(STUBS) == len(COMMANDS) == 25
    assert sorted(_module_of(path) for path in STUBS) == sorted(COMMANDS.values())


def test_scripts_holds_only_stubs_the_readme_and_the_reporting_folder() -> None:
    entries = {path.name for path in (ROOT / "scripts").iterdir() if path.name != "__pycache__"}
    assert entries == {"README.md", "reporting"} | {path.name for path in STUBS if path.parent.name == "scripts"}


@pytest.mark.parametrize("module", sorted(COMMANDS.values()))
def test_every_command_module_runs_with_python_dash_m(module: str, tmp_path: Path) -> None:
    done = _run_from(tmp_path, "-m", module, "--help")
    assert done.returncode == 0
    assert _usage_line(module.rsplit(".", 1)[1].replace("_", "-")) in done.stdout


def test_a_stub_run_from_another_directory_reads_the_repo_defaults(tmp_path: Path) -> None:
    """The default --scores-root is the repository's results/scores, not a folder under the current directory."""
    out = tmp_path / "ablation.md"
    stub = ROOT / "scripts" / "reporting" / "build_ablation_table.py"
    done = _run_from(tmp_path, str(stub), "--out", str(out))
    assert done.returncode == 0 and done.stderr == ""
    assert out.is_file() and out.stat().st_size > 0


def test_a_command_stub_run_from_another_directory_reads_the_repo_defaults(tmp_path: Path) -> None:
    """`evaluate_models score` finds the default configs/label_sets.yaml in the repository, not under the cwd."""
    case = valid_case(tmp_path, "A")
    out = tmp_path / "report.json"
    stub = ROOT / "scripts" / "evaluate_models.py"
    done = _run_from(tmp_path, str(stub), *cli_argv("score", case, out))
    assert done.returncode == 0 and done.stderr == ""
    assert out.is_file() and out.with_suffix(".md").is_file()
