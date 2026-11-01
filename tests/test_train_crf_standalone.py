"""The scripts/train_crf.py stub must run from any folder, with no PYTHONPATH.

The stub runs in a subprocess with PYTHONPATH removed and the cwd outside the repo. It then
imports the `punct_restore` package installed in the venv; no path bootstrap is involved, so
nothing resolves against the working directory.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_crf.py"


def test_train_crf_help_runs_without_pythonpath(tmp_path: Path) -> None:
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage: recap train-crf" in result.stdout
