"""The `inspect` subcommand of train_crf: print the metadata of a saved CRF model artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib


def _print_artifact_summary(bundle: dict[str, Any], model_path: Path) -> None:
    """Print the header, vocabulary and the hyperparameter, feature, training and environment sections."""
    print("=" * 70)
    print(f"RECAP CRF Model Artifact: {model_path.name}")
    print("=" * 70)
    print(f"Schema Version:     {bundle.get('schema_version')}")
    print(f"Model Name:         {bundle.get('model_name')}")
    print(f"Label Mode:         {bundle.get('label_mode')}")
    print(f"Vocabulary Size:    {len(bundle.get('vocabulary', []))} classes")
    print(f"Vocabulary:         {bundle.get('vocabulary')}")
    print("\nHyperparameters:")
    for k, v in bundle.get("hyperparameters", {}).items():
        print(f"  {k:24}: {v}")
    print("\nFeature Configuration:")
    for k, v in bundle.get("feature_config", {}).items():
        print(f"  {k:24}: {v}")
    print("\nTraining Metadata:")
    for k, v in bundle.get("training_metadata", {}).items():
        print(f"  {k:24}: {v}")
    print("\nEnvironment:")
    for k, v in bundle.get("environment", {}).items():
        print(f"  {k:24}: {v}")


def _print_companion_manifest(model_path: Path) -> None:
    """Display the first readable companion manifest that sits next to the artifact, if any."""
    manifest_candidates = [
        model_path.with_name(f"{model_path.name}.manifest.json"),
        model_path.with_suffix(".joblib.manifest.json"),
        model_path.with_suffix(".manifest.json"),
    ]
    for mc in manifest_candidates:
        if mc.is_file():
            try:
                with open(mc, encoding="utf-8") as f:
                    m_data = json.load(f)
                print("\nCompanion Manifest:")
                print(f"  Run ID:                 {m_data.get('run_id')}")
                print(f"  Dataset Release:        {m_data.get('provenance', {}).get('dataset_release')}")
                print(f"  Policy Version:         {m_data.get('provenance', {}).get('policy_version')}")
                print(f"  Code Fingerprint:       {m_data.get('code_fingerprint')}")
                print(f"  Fingerprinted Files:    {len(m_data.get('fingerprinted_files', []))} files")
                print(f"  Training Duration:      {m_data.get('training_duration_seconds')}s")
                print(f"  Reproducibility Note:   {m_data.get('reproducibility_warning')}")
                break
            except Exception:
                pass


def cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect and display metadata of a trained CRF model artifact."""
    model_path = Path(args.model)
    if not model_path.is_file():
        sys.stderr.write(f"Error: Artifact not found at {model_path}\n")
        return 1

    try:  # joblib is unpickling: the artifact is a local file the user chose to inspect
        bundle = joblib.load(model_path)
    except Exception as e:
        sys.stderr.write(f"Failed to read artifact: {e}\n")
        return 1

    if not isinstance(bundle, dict):
        sys.stderr.write(f"Invalid artifact format: root is {type(bundle).__name__}\n")
        return 1

    _print_artifact_summary(bundle, model_path)

    # Display companion manifest if present
    _print_companion_manifest(model_path)

    print("=" * 70)
    return 0
