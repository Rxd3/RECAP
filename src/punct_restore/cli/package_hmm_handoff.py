"""Package a verified HMM study into a checksummed handoff folder and ZIP."""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from punct_restore.hashing import sha256_file, sha256_stream
from punct_restore.models.hmm_experiment_frozen import verified_frozen_selection
from punct_restore.models.hmm_experiment_protocol import now, read_object
from punct_restore.models.hmm_workflow import code_identity, file_identity, write_json
from punct_restore.paths import REPO_ROOT as ROOT
from punct_restore.prediction_io import one_line_error

__all__ = ["main"]

_HANDOFF_INSTRUCTIONS = (
    "# Member 2 HMM handoff\n\n"
    "This bundle contains the complete verified SponSpeech v2 HMM study: all 27 "
    "validation candidates, three frozen winners, full validation/test/test_amb "
    "predictions and sidecars, shared reports/confusions, baselines, error examples, "
    "runtime measurements, raw commands and independent refit evidence.\n\n"
    "MODEL_INDEX.json identifies each selected model. Paths are relative to this "
    "handoff. Every file except SHA256SUMS is checksummed in SHA256SUMS; the ZIP "
    "has a separate checksum next to it. Data records are distributed separately "
    "as sponspeech-m1-v2. Keep the supplied CC BY-NC 4.0 attribution with copies.\n\n"
    "Member 3: compare within a label mode on the same primary IDs.\n"
    "Member 5: raw prediction JSONL, masks/IDs and scorer input hashes are included.\n"
    "Member 6: HMMModel.load_model(model_path) returns the fitted model; predict "
    "accepts dictionaries containing schema_version, sample_id and tokens. "
    "Do not use an unimplemented generic adapter or substitute O predictions.\n\n"
    "For reproduction, check out the training implementation commit in plan.json "
    "(or matching implementation hashes), install config/requirements-hmm.txt, "
    "place the v2 data at the documented repository paths, and prepare the pinned "
    "Member 5 worktree from HMM_RUNBOOK.md. Run scripts/verify_hmm_experiments.py "
    "with --run-dir pointing to this handoff's experiment directory and a fresh "
    "--output-dir. The verifier independently refits models and checks every prediction. "
    "All metric values in HMM_RESULTS.md come from measured runs, not templates. "
    "A/B punctuation and C joint metrics are different tasks.\n"
)


_SPLITS = ("validation", "test", "test_amb")
# What the packager reads from the study files besides the frozen plan and selection.
_SCORED = {"primary": object, "counts": object}
_RUN_RECORD: dict[str, Any] = {"training_summary": {}}


def _verification_schema(modes: Iterable[str]) -> dict[str, Any]:
    """The keys of verification.json that the packager reads, with one audit entry for each mode."""
    return {
        "result": object,
        "plan": {"sha256": str},
        "selection": {"sha256": str},
        "modes": {mode: {"candidates_independently_checked": object} for mode in modes},
    }


def _results_schema(modes: Iterable[str]) -> dict[str, Any]:
    """The keys of results.json that the packager reads, with one summary for each mode and split."""
    summary = {"primary": object, "counts": object, "shortcut_primary": object, "evaluation_sha256": str}
    return {"results": {mode: dict.fromkeys(_SPLITS, summary) for mode in modes}}


def _check_verification(run: Path, verified: dict[str, Any]) -> None:
    """Require a passing independent verification bound to this plan and selection."""
    if (
        verified["result"] != "PASS"
        or verified["plan"]["sha256"] != sha256_file(run / "plan.json")
        or verified["selection"]["sha256"] != sha256_file(run / "selection.json")
    ):
        raise ValueError("a matching complete independent verification is required")


def _check_split_summaries(run: Path, results: dict[str, Any], mode: str) -> None:
    """Require the results summary of every split to match the actual scored reports."""
    for split in _SPLITS:
        folder = run / f"final/mode_{mode}/{split}"
        metric = read_object(folder / "evaluation.json", _SCORED)
        baseline = read_object(folder / "shortcut.evaluation.json", _SCORED)
        summary = results["results"][mode][split]
        if (
            summary["primary"] != metric["primary"]
            or summary["counts"] != metric["counts"]
            or summary["shortcut_primary"] != baseline["primary"]
            or summary["evaluation_sha256"] != sha256_file(folder / "evaluation.json")
        ):
            raise ValueError("summary does not match the actual scored reports")


def _selected_models(
    run: Path,
    config: dict[str, Any],
    selection: dict[str, Any],
    verified: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, Any]:
    """Check every frozen winner against the verification and results; describe it for the index."""
    selected = {}
    for mode, winner in selection["modes"].items():
        if verified["modes"][mode]["candidates_independently_checked"] != len(
            config["search"]["smoothing"]
        ) * len(config["search"]["min_token_count"]):
            raise ValueError("not every search candidate was independently checked")
        if sha256_file(run / winner["model"]) != winner["model_sha256"]:
            raise ValueError("selected model checksum mismatch")
        _check_split_summaries(run, results, mode)
        selected[mode] = {
            **winner,
            "path": "experiment/" + winner["model"],
            "training_summary": read_object((run / winner["model"]).parent / "run.json", _RUN_RECORD)[
                "training_summary"
            ],
        }
    return selected


def _bundle_paths(run: Path, output: Path) -> tuple[Path, Path, Path]:
    """Resolve the run and handoff folders; require a new separate folder inside artifacts/."""
    output = output.resolve()
    run = run.resolve()
    if (
        not output.is_relative_to((ROOT / "artifacts").resolve())
        or output.is_relative_to(run)
        or run.is_relative_to(output)
    ):
        raise ValueError("handoff must be a separate directory inside artifacts/")
    archive_path = output.parent / f"{output.name}.zip"
    # the cleanup after a failed run removes all three, so a run may only start when none of them exists
    if output.exists() or archive_path.exists() or archive_path.with_suffix(".zip.sha256").exists():
        raise ValueError("handoff/archive already exists; use a new immutable handoff ID")
    return run, output, archive_path


def _copy_bundle_files(run: Path, output: Path, config: dict[str, Any]) -> None:
    """Copy the experiment, data, configuration, results notes and this packager into the new handoff folder."""
    shutil.copytree(run, output / "experiment")
    processed = ROOT / config["processed_dir"]
    shutil.copytree(ROOT / config["splits_dir"], output / "data/splits")
    (output / "data").mkdir(exist_ok=True)
    for name in [
        "release_manifest.json",
        "SHA256SUMS",
        "stats.json",
        "SPONSPEECH_LICENSE_AND_ATTRIBUTION.txt",
    ]:
        shutil.copyfile(processed / name, output / "data" / name)
    shutil.copytree(processed / "masks/case_joint", output / "data/case_masks")
    (output / "config").mkdir()
    for name in [
        "configs/hmm_sponspeech_v2.yaml",
        "configs/label_sets.yaml",
        "requirements-hmm.txt",
    ]:
        shutil.copyfile(ROOT / name, output / "config" / Path(name).name)
    shutil.copyfile(ROOT / "experiments/HMM_RESULTS.md", output / "HMM_RESULTS.md")
    shutil.copyfile(ROOT / "experiments/HMM_RUNBOOK.md", output / "HMM_RUNBOOK.md")
    (output / "tools").mkdir()
    shutil.copyfile(Path(__file__), output / "tools/package_hmm_handoff.py")


def _model_index(
    output: Path, config: dict[str, Any], plan: dict[str, Any], selected: dict[str, Any]
) -> dict[str, Any]:
    """The MODEL_INDEX.json content: selected models and the provenance of every part."""
    return {
        "schema": "recap-hmm-handoff-v1",
        "created_at_utc": now(),
        "experiment_id": config["experiment_id"],
        "dataset_release": config["dataset_release_id"],
        "policy_version": config["policy_version"],
        "selected_models": selected,
        "training_code": plan["code"],
        "implementation": plan["implementation"],
        "shared_evaluator": plan["evaluator"],
        "packaging_code": code_identity(),
        "packager": file_identity(Path(__file__)),
        "verification": {
            "path": "experiment/verification/verification.json",
            "sha256": sha256_file(output / "experiment/verification/verification.json"),
        },
        "license": "CC BY-NC 4.0; see data/SPONSPEECH_LICENSE_AND_ATTRIBUTION.txt",
    }


def _bundle_files(output: Path) -> list[Path]:
    """Every file of the handoff, after rejecting any that names the personal workspace path."""
    files = sorted(path for path in output.rglob("*") if path.is_file())
    forbidden = [str(ROOT), ROOT.as_posix(), str(ROOT).replace("\\", "\\\\")]
    for path in files:
        if path.suffix in [".json", ".jsonl", ".md", ".yaml", ".txt", ".log"]:
            content = path.read_text(encoding="utf-8")
            if any(root in content for root in forbidden):
                raise ValueError(f"personal absolute workspace path in {path.name}")
    return files


def _write_archive(output: Path, archive_path: Path) -> None:
    """Write the handoff folder into a new deflated ZIP."""
    with zipfile.ZipFile(
        archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, output.name + "/" + path.relative_to(output).as_posix())


def _verify_archive(archive_path: Path, hashes: dict[str, str], sums: str) -> None:
    """Round-trip every archived byte through SHA-256, not just the ZIP directory/CRC."""
    with zipfile.ZipFile(archive_path) as archive:
        members = {entry.filename.split("/", 1)[1]: entry for entry in archive.infolist()}
        if set(members) != set(hashes) | {"SHA256SUMS"}:
            raise ValueError("archive has missing or extra members")
        for relative, expected in hashes.items():
            with archive.open(members[relative]) as handle:
                actual = sha256_stream(handle)
            if actual != expected:
                raise ValueError(f"archive member hash mismatch: {relative}")
        if archive.read(members["SHA256SUMS"]).decode("utf-8") != sums:
            raise ValueError("archive checksum index mismatch")


def package(run: Path, evaluator: Path, output: Path) -> dict[str, Any]:
    plan, selection = verified_frozen_selection(run, evaluator)
    modes = list(selection["modes"])
    verified = read_object(run / "verification/verification.json", _verification_schema(modes))
    _check_verification(run, verified)
    config = plan["config"]
    results = read_object(run / "results.json", _results_schema(modes))
    selected = _selected_models(run, config, selection, verified, results)
    run, output, archive_path = _bundle_paths(run, output)
    output.mkdir(parents=True)  # FileExistsError if it appeared since the check: then it is not ours to remove
    try:
        return _build_handoff(run, output, archive_path, plan, selected)
    except BaseException:
        # best effort: leave neither the half-built folder nor its zip, so the handoff ID can be used again;
        # a cleanup that fails must not replace the error that is being raised
        shutil.rmtree(output, ignore_errors=True)
        for leftover in (archive_path, archive_path.with_suffix(".zip.sha256")):
            with contextlib.suppress(OSError):
                leftover.unlink(missing_ok=True)
        raise


def _build_handoff(
    run: Path, output: Path, archive_path: Path, plan: dict[str, Any], selected: dict[str, Any]
) -> dict[str, Any]:
    """Fill the new handoff folder, zip it, verify the zip and write its checksum; describe the archive."""
    config = plan["config"]
    _copy_bundle_files(run, output, config)
    (output / "HANDOFF.md").write_text(_HANDOFF_INSTRUCTIONS, encoding="utf-8", newline="\n")
    write_json(output / "MODEL_INDEX.json", _model_index(output, config, plan, selected))
    files = _bundle_files(output)
    hashes = {p.relative_to(output).as_posix(): sha256_file(p) for p in files}
    sums = "".join(f"{digest}  {name}\n" for name, digest in hashes.items())
    (output / "SHA256SUMS").write_text(sums, encoding="utf-8", newline="\n")
    _write_archive(output, archive_path)
    _verify_archive(archive_path, hashes, sums)
    digest = sha256_file(archive_path)
    checksum = archive_path.with_suffix(".zip.sha256")
    checksum.write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8", newline="\n")
    return {
        "archive": file_identity(archive_path),
        "bytes": archive_path.stat().st_size,
        "checksummed_files": len(hashes),
        "archive_roundtrip": "PASS",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recap package-hmm-handoff", description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = package(args.run_dir, args.evaluator_root, args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"error: {one_line_error(exc)}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
