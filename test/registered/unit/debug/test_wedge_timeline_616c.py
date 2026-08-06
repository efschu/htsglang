"""Tests for sglang.srt.debug_utils.wedge_timeline."""

from __future__ import annotations

from datetime import timedelta

import pytest
from _pytest.capture import CaptureFixture

from sglang.srt.debug_utils.wedge_timeline import (
    find_freeze_point,
    main,
    summarize,
    timeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _line(ts: str, text: str, tag: str = "TP0") -> str:
    return f"[{ts} {tag}] {text}"


def _plain_line(ts: str, text: str) -> str:
    return f"[{ts}] {text}"


# ---------------------------------------------------------------------------
# 1. find_freeze_point returns last progress timestamp before first trouble
# ---------------------------------------------------------------------------

def test_find_freeze_point_returns_last_progress_before_trouble() -> None:
    lines = [
        _line("2026-08-06 10:00:00", "Prefill batch, #new-seq: 1"),
        _line("2026-08-06 10:00:05", "Decode batch, #running-req: 2"),
        _line("2026-08-06 10:00:10", "POST /v1/chat/completions HTTP/1.1"),
        _line("2026-08-06 10:00:15", "Bar1CollectiveAborted: something broke"),
    ]
    result = find_freeze_point(lines)
    assert result == "2026-08-06 10:00:10"


# ---------------------------------------------------------------------------
# 2. returns None when no trouble lines exist
# ---------------------------------------------------------------------------

def test_find_freeze_point_no_trouble_returns_none() -> None:
    lines = [
        _line("2026-08-06 10:00:00", "Prefill batch, #new-seq: 1"),
        _line("2026-08-06 10:00:05", "Decode batch, #running-req: 2"),
    ]
    assert find_freeze_point(lines) is None


# ---------------------------------------------------------------------------
# 3. returns None when no progress lines before trouble
# ---------------------------------------------------------------------------

def test_find_freeze_point_no_progress_returns_none() -> None:
    lines = [
        _line("2026-08-06 10:00:00", "Just some random log"),
        _line("2026-08-06 10:00:05", "Bar1CollectiveAborted: oops"),
    ]
    assert find_freeze_point(lines) is None


# ---------------------------------------------------------------------------
# 4. timeline includes lines inside the window and excludes older ones
# ---------------------------------------------------------------------------

def test_timeline_window_filtering(tmp_path) -> None:
    from sglang.srt.debug_utils.wedge_timeline import timeline

    freeze_ts = "2026-08-06 10:05:00"
    lines = [
        # 10 minutes before — outside default 60-s window
        _line("2026-08-06 09:55:00", "Prefill batch, old"),
        # 2 minutes before — outside 60-s window
        _line("2026-08-06 10:03:00", "Prefill batch, too old"),
        # 30 seconds before — inside window
        _line("2026-08-06 10:04:30", "Decode batch, in window"),
        # freeze point
        _line(freeze_ts, "Prefill batch, last progress"),
        # trouble
        _line("2026-08-06 10:05:10", "Bar1CollectiveAborted: crash"),
        _line("2026-08-06 10:05:11", "index out of bounds"),
    ]
    result = timeline(lines, window_s=60)
    # Must include the line at 30s before and the freeze line.
    assert any("in window" in r for r in result), "Window line missing"
    assert any("last progress" in r for r in result), "Freeze line missing"
    # Must NOT include lines outside the window.
    assert not any("old" in r for r in result), "Too-old line leaked in"
    assert not any("too old" in r for r in result), "2-min-ago line leaked in"
    # Must include trouble lines after freeze.
    assert any("Bar1CollectiveAborted" in r for r in result), "Trouble line missing"


# ---------------------------------------------------------------------------
# 5. summarize reports correct gap in seconds
# ---------------------------------------------------------------------------

def test_summarize_gap_in_seconds() -> None:
    lines = [
        _line("2026-08-06 10:00:00", "Prefill batch, #new-seq: 1"),
        _line("2026-08-06 10:00:30", "Decode batch, #running-req: 1"),
        _line("2026-08-06 10:01:00", "Bar1CollectiveAborted: crash"),
    ]
    report = summarize(lines)
    # Freeze at 10:00:30, trouble at 10:01:00 → gap = 30.0 s
    assert "Gap           : 30.0 s" in report, (
        f"Expected gap 30.0 s in report:\n{report}"
    )


# ---------------------------------------------------------------------------
# 6. Trouble marker counts are correct
# ---------------------------------------------------------------------------

def test_summarize_trouble_counts() -> None:
    lines = [
        _line("2026-08-06 10:00:00", "Prefill batch, #new-seq: 1"),
        _line("2026-08-06 10:00:10", "Bar1CollectiveAborted: first"),
        _line("2026-08-06 10:00:11", "Bar1CollectiveAborted: second"),
        _line("2026-08-06 10:00:12", "Health check failed"),
        _line("2026-08-06 10:00:13", "index out of bounds"),
    ]
    report = summarize(lines)
    assert "Bar1CollectiveAborted" in report
    # There should be 2 Bar1CollectiveAborted entries.
    # Parse the count from the report.
    for raw_line in report.splitlines():
        if "Bar1CollectiveAborted" in raw_line and "markers" not in raw_line.lower():
            assert " 2" in raw_line, f"Expected count 2 for Bar1CollectiveAborted, got: {raw_line}"
            break
    for raw_line in report.splitlines():
        if "Health check failed" in raw_line and "markers" not in raw_line.lower():
            assert " 1" in raw_line, f"Expected count 1 for Health check failed, got: {raw_line}"
            break
    for raw_line in report.splitlines():
        if "index out of bounds" in raw_line and "markers" not in raw_line.lower():
            assert " 1" in raw_line, f"Expected count 1 for index out of bounds, got: {raw_line}"
            break


# ---------------------------------------------------------------------------
# 7. main prints a summary (capsys)
# ---------------------------------------------------------------------------

def test_main_prints_summary(tmp_path, capsys: CaptureFixture) -> None:
    log_file = tmp_path / "test.log"
    log_file.write_text(
        "\n".join([
            _line("2026-08-06 12:00:00", "Prefill batch, #new-seq: 1"),
            _line("2026-08-06 12:00:05", "Decode batch, #running-req: 3"),
            _line("2026-08-06 12:00:20", "Bar1CollectiveAborted: crash"),
        ])
    )
    main([str(log_file)])
    captured = capsys.readouterr()
    assert "WEDGE TIMELINE REPORT" in captured.out
    assert "Freeze point" in captured.out
    assert "Bar1CollectiveAborted" in captured.out
