"""Regenerate the statistics files of results/scores from their committed inputs, byte for byte.

Usage: recap result-stats {holm,clusters,label-audit} [--scores-root R] --out OUT_JSON
- ``holm`` writes the Holm-adjusted p-values of the 21 test comparisons and the 18 ablations (``holm.json``; it
  backs "Multiple comparisons (Holm)" in docs/EVALUATION_RESULTS_CRF.md). It reads the compare files in
  R/crf_test, R/threshold, R/threshold_prior, R/hidden_event_lm and R/ablations.
- ``clusters`` writes the K choice of the word-cluster comparison (``clusters/clusters_summary.json``) from the
  six R/clusters/compare_*.json files.
- ``label-audit`` writes the label audit with 95% Wilson intervals (``gold_noise/label_audit.json``) from the
  filled sheet R/gold_noise/label_audit_sheet.csv.
R defaults to results/scores. OUT_JSON must not exist; it is written with LF line ends. The text comes from the pure
functions in holm.py, clusters_summary.py and label_audit.py.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from punct_restore.file_io import write_new_file
from punct_restore.paths import REPO_ROOT as ROOT
from punct_restore.reporting._cli import check_out, run
from punct_restore.reporting.clusters_summary import clusters_summary_text
from punct_restore.reporting.holm import holm_text
from punct_restore.reporting.label_audit import label_audit_text

# subcommand -> (the function that returns the file's text, what the file holds)
REPORTS: Mapping[str, tuple[Callable[[Path], str], str]] = {
    "holm": (holm_text, "Holm-adjusted p-values of the 21 test comparisons and the 18 ablations"),
    "clusters": (clusters_summary_text, "the K choice of the word-cluster comparison"),
    "label-audit": (label_audit_text, "label-audit precision, recall and casing accuracy with 95% Wilson intervals"),
}


def regenerate(report: str, scores_root: Path, out: Path) -> None:
    """Write the text of ``report`` to the new file ``out``; nothing is written if an input cannot be read."""
    check_out(out)  # before reading anything
    text = REPORTS[report][0](scores_root)
    write_new_file(out, text, newline="\n")
    print(f"wrote {out.name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recap result-stats",
        description="Regenerate a statistics file of results/scores from its committed inputs.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--scores-root", type=Path, default=ROOT / "results" / "scores",
                        help="folder that holds the committed score files (default: results/scores)")
    common.add_argument("--out", type=Path, required=True, help="JSON file to create (must not exist)")
    commands = parser.add_subparsers(dest="report", required=True, metavar="{" + ",".join(REPORTS) + "}")
    for name, (_, what) in REPORTS.items():
        commands.add_parser(name, parents=[common], help=what.replace("%", "%%"), description=f"Write {what}.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(lambda: regenerate(args.report, args.scores_root, args.out))


if __name__ == "__main__":
    sys.exit(main())
