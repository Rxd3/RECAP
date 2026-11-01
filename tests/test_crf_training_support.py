"""models/crf_training: the two optional provenance probes fail soft on the expected errors and nothing else."""

from __future__ import annotations

import ctypes
import importlib.metadata
import sys
from typing import Any

import pytest

from punct_restore.models import crf_training

# --- get_package_version -----------------------------------------------------------------------------------


def test_the_version_of_an_installed_package_is_reported() -> None:
    assert crf_training.get_package_version("pytest") == importlib.metadata.version("pytest")


def test_the_version_of_a_package_that_is_not_installed_is_unknown() -> None:
    assert crf_training.get_package_version("no-such-package-for-recap") == "unknown"


def test_an_error_other_than_a_missing_package_is_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(_name: str) -> str:
        raise RuntimeError("metadata is broken")

    monkeypatch.setattr(importlib.metadata, "version", broken)
    with pytest.raises(RuntimeError):
        crf_training.get_package_version("pytest")


# --- get_process_memory_stats ------------------------------------------------------------------------------


class _Windll:
    """A stand-in for ctypes.windll whose ``psapi`` raises the given error when it is looked up."""

    def __init__(self, error: BaseException) -> None:
        self._error = error

    @property
    def psapi(self) -> Any:
        raise self._error


def _pretend_windows(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", _Windll(error), raising=False)


@pytest.mark.parametrize("error", [AttributeError("no psapi"), OSError("cannot load psapi")],
                         ids=["attribute-error", "os-error"])
def test_memory_stats_are_empty_when_the_windows_api_is_unavailable(
        monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    _pretend_windows(monkeypatch, error)
    assert crf_training.get_process_memory_stats() == {}


def test_a_bug_in_the_memory_probe_is_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    _pretend_windows(monkeypatch, RuntimeError("a bug"))
    with pytest.raises(RuntimeError):
        crf_training.get_process_memory_stats()


def test_memory_stats_are_empty_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert crf_training.get_process_memory_stats() == {}


@pytest.mark.skipif(sys.platform != "win32", reason="reads the Windows kernel counters")
def test_memory_stats_on_windows_report_the_four_counters() -> None:
    stats = crf_training.get_process_memory_stats()
    assert set(stats) == {"peak_working_set_mb", "current_working_set_mb",
                          "peak_private_commit_mb", "current_private_commit_mb"}
    assert all(isinstance(value, float) and value > 0 for value in stats.values())

