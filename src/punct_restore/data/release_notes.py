"""Text of the human-readable RELEASE_NOTES.md stored with a release (owned by Member 1).

release_outputs.write_release_notes() writes these lines, so the file goes through release_outputs.write_lines like
every other output.
"""
from .release_constants import RELEASE_FILES


def _header_lines(release_id: str, manifest: dict) -> list[str]:
    """Title, source, license, revision and code hash, then the head of the records table."""
    return [f"# {release_id}\n\n",
            f"Source: SponSpeech, {manifest['source']['reference']}\n",
            "License: CC BY-NC 4.0 (non-commercial use only). "
            "Attribution list: SPONSPEECH_LICENSE_AND_ATTRIBUTION.txt\n",
            f"Source revision: {manifest['source']['revision']}. Policy: {manifest['policy_version']}. "
            f"Schema: v{manifest['schema_version']}.\n",
            f"Code SHA-256: {manifest['code_sha256']}\n\n",
            "## Records\n\n| file | released | flagged (unresolved) | rejected |\n|---|---:|---:|---:|\n"]


def _table_lines(stats: dict) -> list[str]:
    """The record-count rows per file, then the default and primary subset table."""
    lines = []
    for name in RELEASE_FILES:
        status = stats["status_by_file"][name]
        lines.append(f"| {name} | {status.get('accepted', 0) + status.get('flagged', 0)} "
                     f"| {status.get('flagged', 0)} | {status.get('rejected', 0)} |\n")
    lines.append("\n## Default and primary subsets (sample counts)\n\n"
                 "| mode | default train | validation | test | test_amb |\n|---|---:|---:|---:|---:|\n")
    for m in ("A", "B", "C"):
        z = stats["subset_sizes"][m]
        lines.append(f"| {m} | {z['default_train']} | {z['primary_validation']} | {z['primary_test']} "
                     f"| {z['primary_test_amb']} |\n")
    return lines


def _reproduce_lines(manifest: dict) -> list[str]:
    """How to rebuild the transcripts from the pinned source archive."""
    snap = manifest["provenance"]["metadata_snapshot"]
    return ["\n## Reproducing from the original source\n\n",
            f"- Archive: {manifest['provenance']['archive']['url']} "
            f"({manifest['provenance']['archive']['bytes']} bytes, "
            f"GCS generation {manifest['provenance']['archive']['gcs_generation']}).\n",
            "- `python scripts/prepare_data.py extract` recreates the transcripts from pinned byte ranges (or from a "
            "fully downloaded archive with `--archive`) and checks them against `MANIFEST.sha256` "
            f"(SHA-256 {manifest['provenance']['raw_manifest']['sha256']}).\n",
            f"- The dated metadata snapshot (SHA-256 {snap['sha256']}) flags {len(snap['tts_videos'])} text-to-speech "
            f"videos; flagged records per file: {snap['tts_records_by_file']}. They are excluded from the default and "
            "primary ID lists, so building without this exact snapshot gives different lists.\n"]


def _use_lines() -> list[str]:
    """How to train and evaluate with the release files."""
    return ["\n## Use\n\n",
            "- Train on `mode_<M>/train.jsonl` restricted to `default_train/mode_<M>.ids`.\n",
            "- Select on `primary_eval/validation_mode_<M>.ids`; report on `primary_eval/test_mode_<M>.ids` and, "
            "separately, `primary_eval/test_amb_mode_<M>.ids`. Full official files are supplementary results.\n",
            "- Mode C: target = `join_labels(punct_labels, case_labels)`; exclude `masks/case_joint` positions from "
            "casing and joint metrics.\n",
            "- `benchmark_native/` is for comparison with the original SponSpeech four-class benchmark only. Never "
            "compare its scores with canonical mode A scores as if the gold labels were the same.\n",
            "- Models receive `sample_id` and `tokens` only.\n"]


def release_notes_lines(release_id: str, stats: dict, manifest: dict) -> list[str]:
    """Lines of RELEASE_NOTES.md (all numbers come from stats and the manifest)."""
    return _header_lines(release_id, manifest) + _table_lines(stats) + _reproduce_lines(manifest) + _use_lines()
