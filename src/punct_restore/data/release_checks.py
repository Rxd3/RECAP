"""Read-back checks of a written release (owned by Member 1): contract validation and cross-view consistency.

Nothing is published (no manifest, no checksums) unless every check passes.
"""
import json
import os
from contextlib import ExitStack
from itertools import zip_longest
from pathlib import PurePosixPath

from . import preprocessing, splits, validation
from .release_constants import RELEASE_FILES


def _validate_view_files(out: str, files: dict, label_sets: dict, checks: dict) -> None:
    """Validate every written view file against the contract; sets files[path]["valid"], collects file_errors."""
    for path in files:
        mode = PurePosixPath(path).parent.name.removeprefix("mode_")
        name = PurePosixPath(path).stem
        split = splits.SPLIT_OF[{"validation": "dev", "test_amb": "test-amb"}.get(name, name)]
        ok, errs = validation.validate_file(os.path.join(out, path), label_sets, split=split, mode=mode)
        files[path]["valid"] = ok
        if errs:
            checks["file_errors"][path] = errs[:20]


def _view_diffs(out: str, name: str) -> list:
    """Differences between the four views of one release file (at most about 20), read row by row."""
    # Read aligned rows together instead of retaining four full JSONL views.
    # The train views contain millions of tokens; full reloads multiply peak RAM.
    diffs = []
    with ExitStack() as stack:
        streams = [stack.enter_context(open(os.path.join(out, folder, name + ".jsonl"), encoding="utf-8"))
                   for folder in ("mode_A", "mode_B", "mode_C", "benchmark_native/mode_A")]
        for lines in zip_longest(*streams):
            if any(line is None for line in lines):
                diffs.append("sample IDs or their order differ")
                break
            rows = [json.loads(line) for line in lines]
            diffs.extend(d for row in rows[1:] for d in validation.compare_views(rows[:1], [row]))
            if len(diffs) >= 20:
                break
    return diffs


def verify_release(out: str, files: dict, results: list, kept: list, label_sets: dict, raw_manifest: str | None,
                   raw_bad_before: list | None) -> dict:
    """Read back every view; returns the `checks` dict, raises ValueError if the release must not be published.

    It also sets `files[path]["valid"]` for every view it validates, which `build_release` relies on later.
    """
    checks = {"source_records": len(results), "rejected": len(results) - len(kept), "file_errors": {},
              "view_mismatches": {}}
    _validate_view_files(out, files, label_sets, checks)
    for name in RELEASE_FILES:
        diffs = _view_diffs(out, name)
        if diffs:
            checks["view_mismatches"][name] = diffs[:20]
    checks["unique_sample_ids"] = len({r["sample_id"] for r in results}) == len(results)
    checks["raw_mismatch_before"] = raw_bad_before
    checks["raw_mismatch_after"] = preprocessing.verify_raw_manifest(raw_manifest) if raw_manifest else None
    if (checks["file_errors"] or checks["view_mismatches"] or not checks["unique_sample_ids"]
            or checks["raw_mismatch_after"]):
        raise ValueError("release validation failed; no release manifest or checksums will be published")
    return checks
