"""The `recap` command: routes `recap <command> [args]` to the module that implements the command."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

_CLI_MODULES = (
    "build_leakage_subsets", "build_word_clusters", "evaluate_models", "factor_predictions", "find_gold_noise",
    "package_hmm_handoff", "predict_hmm", "prepare_data", "project_joint", "run_ablations", "run_demo",
    "run_hmm_experiments", "simulate_asr_noise", "train_crf", "train_hidden_event_lm", "train_hmm",
    "tune_threshold", "verify_hmm_experiments",
)
_REPORTING_MODULES = (
    "asr_noise_table", "build_ablation_table", "build_m3_final_sections", "build_m3_tables", "crf_convergence",
    "length_position", "time_models",
)
_DOTTED_MODULES = (
    *(f"punct_restore.cli.{name}" for name in _CLI_MODULES),
    *(f"punct_restore.reporting.{name}" for name in _REPORTING_MODULES),
)

# Subcommand -> dotted module; the subcommand is the module name with "_" written as "-".
COMMANDS: Mapping[str, str] = MappingProxyType(
    {dotted.rsplit(".", 1)[1].replace("_", "-"): dotted for dotted in _DOTTED_MODULES}
)

_USAGE = "usage: recap <command> [args]"
_NOT_AVAILABLE = "(not available)"
_HELP_FLAGS = ("-h", "--help")


def _summary(dotted: str) -> str:
    """First docstring line of a module, read from its source so the module is never imported."""
    try:
        spec = importlib.util.find_spec(dotted)
        if spec is None or not spec.origin:
            return _NOT_AVAILABLE
        tree = ast.parse(Path(spec.origin).read_bytes())
    except (ImportError, ValueError, OSError, SyntaxError):
        return _NOT_AVAILABLE
    return (ast.get_docstring(tree) or "").partition("\n")[0].strip()


def _print_listing(commands: Mapping[str, str]) -> None:
    width = max(map(len, commands), default=0)
    print(_USAGE, "", "commands:", sep="\n")
    for name, dotted in commands.items():
        print(f"  {name:<{width}}  {_summary(dotted)}".rstrip())


def dispatch(argv: list[str], commands: Mapping[str, str]) -> int:
    """Run `argv[0]` as a command with `argv[1:]`; no argument or -h/--help lists the commands (exit 0)."""
    if not argv or argv[0] in _HELP_FLAGS:
        _print_listing(commands)
        return 0
    name = argv[0]
    if name not in commands:
        print(f"error: unknown command {name!r}; run recap --help for the list", file=sys.stderr)
        return 2
    module = importlib.import_module(commands[name])
    return module.main(argv[1:])


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point (`recap`): dispatch `sys.argv[1:]` unless `argv` is given."""
    return dispatch(sys.argv[1:] if argv is None else argv, COMMANDS)
