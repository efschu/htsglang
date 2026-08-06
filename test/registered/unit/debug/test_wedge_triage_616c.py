"""Unit tests for wedge_triage - GPU-rank-wedge evidence collector.

Hermetic: all external commands (nvidia-smi, py-spy, pgrep) are
monkeypatched through the module's _run helper.
"""

from __future__ import annotations

import importlib
import os
import subprocess
from unittest.mock import patch


# Resolve the module under test.
# PYTHONPATH must include the project's python/ dir.
_wedge = importlib.import_module("sglang.srt.debug_utils.wedge_triage")


class TestCollectCreatesOutDirAndKeys:
    """collect() creates out_dir and returns a dict with the expected keys."""

    def test_creates_out_dir_and_keys(self, tmp_path, monkeypatch) -> None:
        out_dir = str(tmp_path / "triage")
        log_file = tmp_path / "server.log"
        log_file.write_text("nothing special here\n")

        # _run always reports failure (no GPU tools in test env).
        monkeypatch.setattr(
            _wedge,
            "_run",
            lambda cmd, timeout=30: (-1, "", "no tool"),
        )

        result = _wedge.collect(str(log_file), out_dir)

        assert os.path.isdir(out_dir)
        # expected top-level keys
        for key in ("timestamp", "files", "markers", "nvidia_smi", "py_spy", "errors"):
            assert key in result, f"missing key: {key}"


class TestMarkerCounts:
    """Marker counts are correct for a synthetic log with known markers."""

    def test_counts_match(self, tmp_path, monkeypatch) -> None:
        out_dir = str(tmp_path / "triage")
        log_file = tmp_path / "server.log"
        log_file.write_text(
            "2026-01-01 index out of bounds at tensor 0\n"
            "2026-01-01 index out of bounds at tensor 1\n"
            "2026-01-01 Bar1CollectiveAborted\n"
            "2026-01-01 Health check failed on rank 2\n"
            "2026-01-01 Health check failed on rank 3\n"
            "2026-01-01 Health check failed on rank 4\n"
            "2026-01-01 abort flag snapshot for rank 1\n"
            "2026-01-01 collective history (rank 0): 3 steps\n"
            "2026-01-01 collective history (rank 1): 5 steps\n"
            "2026-01-01 collective census (rank 0): 2/3 alive\n"
            "2026-01-01 some other noise\n",
        )

        monkeypatch.setattr(
            _wedge,
            "_run",
            lambda cmd, timeout=30: (-1, "", "no tool"),
        )

        result = _wedge.collect(str(log_file), out_dir)

        assert result["markers"]["index out of bounds"] == 2
        assert result["markers"]["Bar1CollectiveAborted"] == 1
        assert result["markers"]["Health check failed"] == 3
        assert result["markers"]["abort flag snapshot"] == 1
        assert result["markers"]["collective history (rank"] == 2
        assert result["markers"]["collective census (rank"] == 1


class TestFailingCommandRecorded:
    """A failing / missing external command is recorded as an error and does not raise."""

    def test_missing_binary_recorded(self, tmp_path, monkeypatch) -> None:
        out_dir = str(tmp_path / "triage")
        log_file = tmp_path / "server.log"
        log_file.write_text("empty\n")

        # Simulate FileNotFoundError from subprocess.
        def _fail_run(cmd, timeout=30):
            return -1, "", "command not found: [Errno 2] no such file"

        monkeypatch.setattr(_wedge, "_run", _fail_run)

        # Must NOT raise
        result = _wedge.collect(str(log_file), out_dir)

        assert len(result["errors"]) >= 1  # at least nvidia-smi failure
        assert any("nvidia-smi" in e for e in result["errors"])


class TestEmptyPidsAndDiscovery:
    """When pids is empty and discovery finds none, collect() still succeeds."""

    def test_no_pids_ok(self, tmp_path, monkeypatch) -> None:
        out_dir = str(tmp_path / "triage")
        log_file = tmp_path / "server.log"
        log_file.write_text("quiet log\n")

        monkeypatch.setattr(
            _wedge,
            "_run",
            lambda cmd, timeout=30: (-1, "", "no tool"),
        )

        result = _wedge.collect(str(log_file), out_dir, pids=[])

        assert isinstance(result, dict)
        assert result["py_spy"] == []


class TestReturnedFilesExist:
    """Files listed in the returned dict actually exist on disk."""

    def test_files_on_disk(self, tmp_path, monkeypatch) -> None:
        out_dir = str(tmp_path / "triage")
        log_file = tmp_path / "server.log"
        log_file.write_text("Bar1CollectiveAborted\n")

        monkeypatch.setattr(
            _wedge,
            "_run",
            lambda cmd, timeout=30: (-1, "", "no tool"),
        )

        result = _wedge.collect(str(log_file), out_dir)

        for f in result["files"]:
            assert os.path.isfile(f), f"missing file: {f}"


class TestMainPrintsSummary:
    """main() prints a short human-readable summary (capsys)."""

    def test_main_output(self, tmp_path, capsys) -> None:
        log_file = tmp_path / "server.log"
        log_file.write_text("nothing here\n")
        out_dir = str(tmp_path / "triage")

        # Patch subprocess.run inside _run so no real commands fire.
        mock_proc = subprocess.CompletedProcess(
            args=[], returncode=-1, stdout="", stderr="no nvidia-smi"
        )

        with patch.object(_wedge.subprocess, "run", return_value=mock_proc):
            _wedge.main(["--log", str(log_file), "--out", out_dir, "--pids"])

        captured = capsys.readouterr()
        assert "wedge_triage" in captured.out
        assert "files collected" in captured.out
