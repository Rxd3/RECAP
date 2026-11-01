"""Package an independently verified HMM study for Members 3, 5 and 6."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

from punct_restore.cli.run_hmm_experiments import frozen, now, read_json
from punct_restore.models.hmm_workflow import (
    code_identity,
    file_identity,
    sha256,
    write_json,
)
from punct_restore.paths import REPO_ROOT as ROOT


def package(run, evaluator, output):
    plan, selection = frozen(run, evaluator)
    verified = read_json(run / "verification/verification.json")
    if (
        verified["result"] != "PASS"
        or verified["plan"]["sha256"] != sha256(run / "plan.json")
        or verified["selection"]["sha256"] != sha256(run / "selection.json")
    ):
        raise ValueError("a matching complete independent verification is required")
    config = plan["config"]
    results = read_json(run / "results.json")
    selected = {}
    for mode, winner in selection["modes"].items():
        if verified["modes"][mode]["candidates_independently_checked"] != len(
            config["search"]["smoothing"]
        ) * len(config["search"]["min_token_count"]):
            raise ValueError("not every search candidate was independently checked")
        if sha256(run / winner["model"]) != winner["model_sha256"]:
            raise ValueError("selected model checksum mismatch")
        for split in ["validation", "test", "test_amb"]:
            folder = run / f"final/mode_{mode}/{split}"
            metric = read_json(folder / "evaluation.json")
            baseline = read_json(folder / "shortcut.evaluation.json")
            summary = results["results"][mode][split]
            if (
                summary["primary"] != metric["primary"]
                or summary["counts"] != metric["counts"]
                or summary["shortcut_primary"] != baseline["primary"]
                or summary["evaluation_sha256"] != sha256(folder / "evaluation.json")
            ):
                raise ValueError("summary does not match the actual scored reports")
        selected[mode] = {
            **winner,
            "path": "experiment/" + winner["model"],
            "training_summary": read_json((run / winner["model"]).parent / "run.json")[
                "training_summary"
            ],
        }
    output = output.resolve()
    run = run.resolve()
    if (
        not output.is_relative_to((ROOT / "artifacts").resolve())
        or output.is_relative_to(run)
        or run.is_relative_to(output)
    ):
        raise ValueError("handoff must be a separate directory inside artifacts/")
    archive_path = output.parent / f"{output.name}.zip"
    if output.exists() or archive_path.exists():
        raise ValueError("handoff/archive already exists; use a new immutable handoff ID")
    output.mkdir(parents=True, exist_ok=False)
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
    instructions = (
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
    (output / "HANDOFF.md").write_text(instructions, encoding="utf-8", newline="\n")
    index = {
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
            "sha256": sha256(output / "experiment/verification/verification.json"),
        },
        "license": "CC BY-NC 4.0; see data/SPONSPEECH_LICENSE_AND_ATTRIBUTION.txt",
    }
    write_json(output / "MODEL_INDEX.json", index)
    files = sorted(path for path in output.rglob("*") if path.is_file())
    forbidden = [str(ROOT), ROOT.as_posix(), str(ROOT).replace("\\", "\\\\")]
    for path in files:
        if path.suffix in [".json", ".jsonl", ".md", ".yaml", ".txt", ".log"]:
            content = path.read_text(encoding="utf-8")
            if any(root in content for root in forbidden):
                raise ValueError(f"personal absolute workspace path in {path.name}")
    hashes = {p.relative_to(output).as_posix(): sha256(p) for p in files}
    sums = "".join(f"{digest}  {name}\n" for name, digest in hashes.items())
    (output / "SHA256SUMS").write_text(sums, encoding="utf-8", newline="\n")
    with zipfile.ZipFile(
        archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, output.name + "/" + path.relative_to(output).as_posix())
    # Round-trip every archived byte through SHA-256, not just the ZIP directory/CRC.
    import hashlib

    with zipfile.ZipFile(archive_path) as archive:
        members = {entry.filename.split("/", 1)[1]: entry for entry in archive.infolist()}
        if set(members) != set(hashes) | {"SHA256SUMS"}:
            raise ValueError("archive has missing or extra members")
        for relative, expected in hashes.items():
            digest = hashlib.sha256()
            with archive.open(members[relative]) as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"archive member hash mismatch: {relative}")
        if archive.read(members["SHA256SUMS"]).decode("utf-8") != sums:
            raise ValueError("archive checksum index mismatch")
    digest = sha256(archive_path)
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
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
