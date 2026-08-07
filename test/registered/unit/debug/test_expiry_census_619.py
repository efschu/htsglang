# SPDX-License-Identifier: Apache-2.0
"""Hermetic tests for the expiry-path capture-census dump (#619).

The abort handler in ``_handle_abort`` already dumps the capture census, but
the rank whose compute stream is stuck never reaches the abort handler -- it
lives in the bounded-poll warning path. These tests verify that after N
expiries the same census dump fires, at most once per process.

Methods are invoked unbound against stubs, following the documented pattern
in ``test_barlink_bar1_abort_poll_616f.py``.
"""

from __future__ import annotations

import logging
import sys
import types
import unittest.mock

import pytest
import torch

from sglang.srt.distributed.device_communicators.barlink_abort_gate import (
    ENV_SYNC_DEADLINE_MS,
)
from sglang.srt.distributed.device_communicators.barlink_bar1 import (
    BarlinkBar1Transport,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _Event:
    """Stand-in for a CUDA event that never completes."""

    def __init__(self):
        self.queries = 0
        self.synchronized = 0

    def query(self) -> bool:
        self.queries += 1
        return False  # never ready

    def synchronize(self) -> None:
        self.synchronized += 1


def _stub(event: _Event, **extra) -> types.SimpleNamespace:
    base = dict(
        _ctl_event=event,
        _ctl_sync_timeouts=0,
        _ctl_stall_run=0,
        group="tp",
        rank=2,
        _last_op="all_reduce",
        _last_nbytes=12584960,
        _deferred_launches=3,
        _ctl_build_deferred_s=0.0,
        _peer_table=None,
        _expiry_census_fired=False,
    )
    base.update(extra)
    return types.SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# Capture-census mock
# ---------------------------------------------------------------------------


class _CaptureCensusMock:
    """Fake ``barlink_capture_census`` module for the test."""

    def __init__(self):
        self.format_call_count = 0
        self._enabled = True

    def capture_census_enabled(self) -> bool:
        return self._enabled

    def format_local_capture_census(self, rank: int) -> str:
        self.format_call_count += 1
        return f"barlink capture census (rank {rank}): mock data"


@pytest.fixture()
def mock_census(monkeypatch):
    """Patch sys.modules so the lazy import in _wait_ctl_event reaches the
    mock instead of the real module (which needs CUDA)."""
    mock = _CaptureCensusMock()
    mp = unittest.mock.MagicMock(wraps=mock)
    monkeypatch.setitem(sys.modules,
                        "sglang.srt.distributed.device_communicators.barlink_capture_census",
                        mp)
    return mock, mp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExpiryCensusAfter:
    """After N expiries the census dump fires exactly once."""

    def test_default_threshold_is_three(self, monkeypatch, mock_census):
        """SGLANG_BARLINK_EXPIRY_CENSUS_AFTER defaults to 3."""
        mock, mp = mock_census
        monkeypatch.delenv("SGLANG_BARLINK_EXPIRY_CENSUS_AFTER", raising=False)
        ev = _Event()
        s = _stub(ev)

        monkeypatch.setenv(ENV_SYNC_DEADLINE_MS, "1")
        monkeypatch.setenv("SGLANG_BARLINK_BAR1_STALL_RAISE_AFTER", "0")

        # Two expiries -- below threshold, no dump.
        BarlinkBar1Transport._wait_ctl_event(s)
        BarlinkBar1Transport._wait_ctl_event(s)
        assert s._ctl_sync_timeouts == 2
        assert mock.format_call_count == 0
        assert s._expiry_census_fired is False

        # Third expiry -- at threshold, dump fires.
        BarlinkBar1Transport._wait_ctl_event(s)
        assert s._ctl_sync_timeouts == 3
        assert mock.format_call_count == 1
        assert s._expiry_census_fired is True

    def test_disabled_via_zero(self, monkeypatch, mock_census):
        """Threshold 0 disables the expiry dump entirely."""
        mock, mp = mock_census
        monkeypatch.setenv("SGLANG_BARLINK_EXPIRY_CENSUS_AFTER", "0")
        ev = _Event()
        s = _stub(ev)

        monkeypatch.setenv(ENV_SYNC_DEADLINE_MS, "1")
        monkeypatch.setenv("SGLANG_BARLINK_BAR1_STALL_RAISE_AFTER", "0")

        for _ in range(10):
            BarlinkBar1Transport._wait_ctl_event(s)

        assert s._ctl_sync_timeouts == 10
        assert mock.format_call_count == 0
        assert s._expiry_census_fired is False

    def test_dump_fires_exactly_once(self, monkeypatch, mock_census):
        """The bool latch prevents re-dumping after repeated expiries."""
        mock, mp = mock_census
        monkeypatch.setenv("SGLANG_BARLINK_EXPIRY_CENSUS_AFTER", "3")
        ev = _Event()
        s = _stub(ev)

        monkeypatch.setenv(ENV_SYNC_DEADLINE_MS, "1")
        monkeypatch.setenv("SGLANG_BARLINK_BAR1_STALL_RAISE_AFTER", "0")

        # Run well past the threshold.
        for _ in range(10):
            BarlinkBar1Transport._wait_ctl_event(s)

        assert s._ctl_sync_timeouts == 10
        assert mock.format_call_count == 1  # exactly once
        assert s._expiry_census_fired is True

    def test_census_disabled_skips_format(self, monkeypatch, mock_census):
        """When capture_census_enabled() returns False, format is not called."""
        mock, mp = mock_census
        mock._enabled = False
        monkeypatch.setenv("SGLANG_BARLINK_EXPIRY_CENSUS_AFTER", "3")
        ev = _Event()
        s = _stub(ev)

        monkeypatch.setenv(ENV_SYNC_DEADLINE_MS, "1")
        monkeypatch.setenv("SGLANG_BARLINK_BAR1_STALL_RAISE_AFTER", "0")

        for _ in range(5):
            BarlinkBar1Transport._wait_ctl_event(s)

        assert s._ctl_sync_timeouts == 5
        assert mock.format_call_count == 0

    def test_custom_threshold(self, monkeypatch, mock_census):
        """A user-set threshold of 1 means dump on the first expiry."""
        mock, mp = mock_census
        monkeypatch.setenv("SGLANG_BARLINK_EXPIRY_CENSUS_AFTER", "1")
        ev = _Event()
        s = _stub(ev)

        monkeypatch.setenv(ENV_SYNC_DEADLINE_MS, "1")
        monkeypatch.setenv("SGLANG_BARLINK_BAR1_STALL_RAISE_AFTER", "0")

        BarlinkBar1Transport._wait_ctl_event(s)
        assert s._ctl_sync_timeouts == 1
        assert mock.format_call_count == 1

        # Second expiry -- latch prevents re-dump.
        BarlinkBar1Transport._wait_ctl_event(s)
        assert s._ctl_sync_timeouts == 2
        assert mock.format_call_count == 1  # still exactly once

    def test_dump_uses_correct_rank(self, monkeypatch, mock_census, caplog):
        """The dump logs the rank from the stub, not a hardcoded value."""
        mock, mp = mock_census
        monkeypatch.setenv("SGLANG_BARLINK_EXPIRY_CENSUS_AFTER", "1")
        ev = _Event()
        s = _stub(ev, rank=7)

        monkeypatch.setenv(ENV_SYNC_DEADLINE_MS, "1")
        monkeypatch.setenv("SGLANG_BARLINK_BAR1_STALL_RAISE_AFTER", "0")

        with caplog.at_level(logging.ERROR):
            BarlinkBar1Transport._wait_ctl_event(s)

        assert "rank 7" in caplog.text


# ---------------------------------------------------------------------------
# Verify shared function (no duplication)
# ---------------------------------------------------------------------------


class TestSharedAbortExpiryPath:
    """The expiry path calls the SAME function as the abort path."""

    def test_both_paths_call_same_function(self):
        """Both sites call format_local_capture_census, not a duplicate."""
        import inspect

        src = inspect.getsource(BarlinkBar1Transport)
        count = src.count('format_local_capture_census(')
        # Exactly 2: one in the expiry path, one in the abort path.
        assert count == 2, f"expected 2 calls to format_local_capture_census, got {count}"
