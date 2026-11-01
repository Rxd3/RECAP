"""Hardware description and in-memory inference timing for the HMM experiment's runtime report."""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import subprocess
from pathlib import Path
from time import perf_counter
from typing import Any

from punct_restore.models.hmm import HMMModel
from punct_restore.models.hmm_workflow import (
    file_identity,
    selected_records,
    write_json,
)
from punct_restore.paths import REPO_ROOT as ROOT
from punct_restore.prediction_io import read_ids


def hardware() -> dict[str, Any]:
    """The processor, core and memory details of this machine."""
    details = {"processor": platform.processor(), "logical_cpus": os.cpu_count()}
    if platform.system() == "Windows":
        command = (
            "$hmmCpuInfo = Get-CimInstance Win32_Processor | Select-Object -First 1; "
            "$hmmOsInfo = Get-CimInstance Win32_OperatingSystem; "
            "[pscustomobject]@{processor=$hmmCpuInfo.Name; "
            "physical_cores=$hmmCpuInfo.NumberOfCores; "
            "logical_cpus=$hmmCpuInfo.NumberOfLogicalProcessors; "
            "memory_bytes=1024*$hmmOsInfo.TotalVisibleMemorySize} | ConvertTo-Json -Compress"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            check=True,
        )
        details.update(json.loads(result.stdout))
    elif hasattr(os, "sysconf"):
        try:
            details["memory_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (OSError, ValueError):
            pass  # the memory size is optional detail: a platform that cannot report it just leaves the field out
    return details


def percentile(values: list[float], fraction: float) -> float:
    """The nearest-rank percentile of the values."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def _comparison_observations(
    config: dict[str, Any], ids: tuple[str, ...]
) -> list[dict[str, Any]]:
    """The comparison records, verified identical across every label mode."""
    observations = list(
        selected_records(ROOT / config["processed_dir"] / "mode_A/validation.jsonl", ids)
    )
    # Verify the exact same observable workload exists in all modes.
    for mode in config["modes"]:
        other = list(
            selected_records(ROOT / config["processed_dir"] / f"mode_{mode}/validation.jsonl", ids)
        )
        if other != observations:
            raise ValueError("runtime comparison tokens differ across label modes")
    return observations


def _time_predictions(
    model: HMMModel, observations: list[dict[str, Any]], repetitions: int
) -> tuple[list[float], Any]:
    """Time repeated model.predict passes and require identical predictions each time."""
    times, reference = [], None
    for _ in range(repetitions):
        start = perf_counter()
        predicted = model.predict(observations)
        times.append(perf_counter() - start)
        if reference is not None and predicted != reference:
            raise ValueError("HMM inference is not deterministic")
        reference = predicted
    return times, reference


def _sequence_latencies(
    model: HMMModel, observations: list[dict[str, Any]], reference: Any
) -> list[float]:
    """Time one model.decode pass per sequence and require it to agree with model.predict."""
    latencies = []
    for row, prediction in zip(observations, reference, strict=True):
        start = perf_counter()
        labels = model.decode(row["tokens"])
        latencies.append(perf_counter() - start)
        if labels != prediction["predicted_labels"]:
            raise ValueError("decode/predict disagree")
    return latencies


def runtime_benchmark(
    config: dict[str, Any], run: Path, winners: dict[str, Any]
) -> dict[str, Any]:
    """Measure in-memory inference speed of every frozen winner and write runtime.json."""
    ids_path = ROOT / config["splits_dir"] / config["runtime"]["comparison_ids"]
    ids = tuple(read_ids(ids_path, strict=True))
    observations = _comparison_observations(config, ids)
    tokens = sum(len(row["tokens"]) for row in observations)
    report = {
        "comparison_ids": file_identity(ids_path),
        "sequences": len(observations),
        "tokens": tokens,
        "repetitions": config["runtime"]["repetitions"],
        "warmup_sequences": min(config["runtime"]["warmup_sequences"], len(observations)),
        "scope": "in-memory model.predict; excludes disk IO, model loading, scoring and writes",
        "latency_scope": "one additional per-sequence model.decode pass after warmup",
        "model_threads": 1,
        "gc": "Python default, enabled",
        "modes": {},
    }
    for mode in config["modes"]:
        model = HMMModel.load_model(run / winners[mode]["model"])
        model.predict(observations[: report["warmup_sequences"]])
        times, reference = _time_predictions(model, observations, report["repetitions"])
        latencies = _sequence_latencies(model, observations, reference)
        median = statistics.median(times)
        report["modes"][mode] = {
            "prediction_seconds": times,
            "median_seconds": median,
            "tokens_per_second": tokens / median,
            "sequences_per_second": len(ids) / median,
            "sequence_latency_p50_ms": percentile(latencies, 0.5) * 1000,
            "sequence_latency_p95_ms": percentile(latencies, 0.95) * 1000,
        }
    write_json(run / "runtime.json", report)
    return report
