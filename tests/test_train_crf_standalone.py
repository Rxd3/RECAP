"""scripts/train_crf.py must run from a plain checkout, without `pip install -e .`.

The script runs in a subprocess with PYTHONPATH removed and the cwd outside the repo, so only
the script's own sys.path bootstrap can make `punct_restore` importable. (If the package is
also installed in the interpreter's site-packages, this test cannot tell the two apart.)
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
    assert "usage:" in result.stdout
