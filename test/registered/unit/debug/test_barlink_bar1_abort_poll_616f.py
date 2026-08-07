# SPDX-License-Identifier: Apache-2.0
"""Hermetic tests for the BAR1 transport's abort visibility (#616f).

The production wedge these cover: three ranks sat inside
``_read_status_for_check`` -> ``Event.synchronize()`` for over ten minutes and
emitted no abort report at all.  Two independent defects produced that
silence, and both are exercised here without a GPU:

* the staged read waited on an event ordered BEHIND the wedged collective, so
  the guard joined the fault instead of reporting it;
* ``BarlinkBar1Transport`` never defined ``poll_status_word``, so the
  watchdog's private-stream read -- the only reader that still resolves while
  the compute stream is stuck -- silently skipped the transport entirely
  (``barlink_abort_gate.poll_status_words`` looks the method up with
  ``getattr(..., None)``).

Methods are invoked unbound against stubs: constructing a real transport
needs BAR1, peers and a device, none of which belong in a unit test.
"""

from __future__ import annotations

import types

import pytest
import torch

from sglang.srt.distributed.device_communicators.barlink_abort_gate import (
    ENV_SYNC_DEADLINE_MS,
    sync_deadline_s,
)
from sglang.srt.distributed.device_communicators.barlink_bar1 import (
    Bar1CollectiveStalled,
    BarlinkBar1Transport,
)


# ---------------------------------------------------------------------------
# sync_deadline_s -- the knob
# ---------------------------------------------------------------------------


class TestSyncDeadline:
    def test_default_is_two_seconds(self, monkeypatch):
        monkeypatch.delenv(ENV_SYNC_DEADLINE_MS, raising=False)
        assert sync_deadline_s() == pytest.approx(2.0)

    def test_env_is_milliseconds(self, monkeypatch):
        monkeypatch.setenv(ENV_SYNC_DEADLINE_MS, "250")
        assert sync_deadline_s() == pytest.approx(0.25)

    def test_zero_disables_the_bound(self, monkeypatch):
        monkeypatch.setenv(ENV_SYNC_DEADLINE_MS, "0")
        assert sync_deadline_s() == 0.0

    def test_negative_clamps_to_zero(self, monkeypatch):
        monkeypatch.setenv(ENV_SYNC_DEADLINE_MS, "-5")
        assert sync_deadline_s() == 0.0

    def test_garbage_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(ENV_SYNC_DEADLINE_MS, "soon")
        assert sync_deadline_s() == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# _wait_ctl_event -- the bound that the wedge proved was missing
# ---------------------------------------------------------------------------


class _Event:
    """Stand-in for a CUDA event with a scripted query() sequence."""

    def __init__(self, ready: bool = False):
        self.ready = ready
        self.queries = 0
        self.synchronized = 0

    def query(self) -> bool:
        self.queries += 1
        return self.ready

    def synchronize(self) -> None:
        self.synchronized += 1


def _stub(event: _Event) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        _ctl_event=event,
        _ctl_sync_timeouts=0,
        _ctl_stall_run=0,
        group="tp",
        rank=0,
        _last_op="all_reduce",
        _last_nbytes=12584960,
        _deferred_launches=3,
        # #615: the stub mirrors the real transport's fields so the
        # build-window deferral path is actually EXERCISED here rather than
        # short-circuited by an AttributeError. With no peer table there is no
        # peer that could be building, so every assertion below keeps its
        # pre-#615 meaning: these tests pin the escalation, not the extension.
        _ctl_build_deferred_s=0.0,
        _peer_table=None,
    )


class TestWaitCtlEvent:
    def test_resolved_event_returns_true_without_blocking(self, monkeypatch):
        monkeypatch.setenv(ENV_SYNC_DEADLINE_MS, "50")
        ev = _Event(ready=True)
        s = _stub(ev)
        assert BarlinkBar1Transport._wait_ctl_event(s) is True
        assert ev.synchronized == 0
        assert s._ctl_sync_timeouts == 0

    def test_stuck_event_gives_up_at_the_deadline(self, monkeypatch):
        """THE REGRESSION. Pre-#616f this call never returned."""
        monkeypatch.setenv(ENV_SYNC_DEADLINE_MS, "20")
        ev = _Event(ready=False)
        s = _stub(ev)
        assert BarlinkBar1Transport._wait_ctl_event(s) is False
        # It must never fall back to the unbounded wait.
        assert ev.synchronized == 0
        assert s._ctl_sync_timeouts == 1

    def test_stuck_event_is_bounded_in_wall_clock(self, monkeypatch):
        import time as _time

        monkeypatch.setenv(ENV_SYNC_DEADLINE_MS, "30")
        s = _stub(_Event(ready=False))
        t0 = _time.monotonic()
        BarlinkBar1Transport._wait_ctl_event(s)
        elapsed = _time.monotonic() - t0
        # Generous upper bound: the point is that it terminates at all.
        assert elapsed < 2.0

    def test_zero_deadline_restores_the_blocking_wait(self, monkeypatch):
        monkeypatch.setenv(ENV_SYNC_DEADLINE_MS, "0")
        ev = _Event(ready=False)
        s = _stub(ev)
        assert BarlinkBar1Transport._wait_ctl_event(s) is True
        assert ev.synchronized == 1

    def test_timeouts_accumulate(self, monkeypatch):
        monkeypatch.setenv(ENV_SYNC_DEADLINE_MS, "1")
        s = _stub(_Event(ready=False))
        for _ in range(3):
            BarlinkBar1Transport._wait_ctl_event(s)
        assert s._ctl_sync_timeouts == 3

    def test_expiry_is_logged_with_attribution(self, monkeypatch, caplog):
        monkeypatch.setenv(ENV_SYNC_DEADLINE_MS, "1")
        s = _stub(_Event(ready=False))
        with caplog.at_level("WARNING"):
            BarlinkBar1Transport._wait_ctl_event(s)
        text = caplog.text
        # Silence was the whole defect: the line must name the op and size.
        assert "all_reduce" in text
        assert "12584960" in text


class TestStallEscalation:
    """A run of expiries must reach the serving path as an exception."""

    def test_default_threshold(self, monkeypatch):
        from sglang.srt.distributed.device_communicators.barlink_abort_gate import (
            ENV_STALL_RAISE_AFTER,
            stall_raise_after,
        )

        monkeypatch.delenv(ENV_STALL_RAISE_AFTER, raising=False)
        assert stall_raise_after() == 30
        monkeypatch.setenv(ENV_STALL_RAISE_AFTER, "3")
        assert stall_raise_after() == 3
        monkeypatch.setenv(ENV_STALL_RAISE_AFTER, "0")
        assert stall_raise_after() == 0

    def test_a_run_of_expiries_raises(self, monkeypatch):
        monkeypatch.setenv(ENV_SYNC_DEADLINE_MS, "1")
        monkeypatch.setenv("SGLANG_BARLINK_BAR1_STALL_RAISE_AFTER", "3")
        s = _stub(_Event(ready=False))
        assert BarlinkBar1Transport._wait_ctl_event(s) is False
        assert BarlinkBar1Transport._wait_ctl_event(s) is False
        with pytest.raises(Bar1CollectiveStalled) as ei:
            BarlinkBar1Transport._wait_ctl_event(s)
        # Structured attribution, not just English.
        assert ei.value.op == "all_reduce"
        assert ei.value.nbytes == 12584960
        assert ei.value.expiries == 3
        assert ei.value.rank == 0
        # And it says the thing the wedge needed said: nothing tripped.
        assert "CLEAN" in str(ei.value)

    def test_zero_never_raises(self, monkeypatch):
        monkeypatch.setenv(ENV_SYNC_DEADLINE_MS, "1")
        monkeypatch.setenv("SGLANG_BARLINK_BAR1_STALL_RAISE_AFTER", "0")
        s = _stub(_Event(ready=False))
        for _ in range(8):
            assert BarlinkBar1Transport._wait_ctl_event(s) is False

    def test_a_resolved_read_breaks_the_run(self, monkeypatch):
        """A slow step must not accumulate toward a stall."""
        monkeypatch.setenv(ENV_SYNC_DEADLINE_MS, "1")
        monkeypatch.setenv("SGLANG_BARLINK_BAR1_STALL_RAISE_AFTER", "3")
        ev = _Event(ready=False)
        s = _stub(ev)
        for _ in range(20):
            BarlinkBar1Transport._wait_ctl_event(s)
            ev.ready = True
            assert BarlinkBar1Transport._wait_ctl_event(s) is True
            assert s._ctl_stall_run == 0
            ev.ready = False

    def test_stall_is_catchable_as_a_liveness_failure(self):
        """Existing handlers of the barlink abort family must still catch it."""
        from sglang.srt.distributed.device_communicators.barlink_bar1 import (
            Bar1KernelAborted,
        )

        assert issubclass(Bar1CollectiveStalled, Bar1KernelAborted)


class _WedgedEvent(_Event):
    """An event that models the production fault: it never completes.

    ``synchronize()`` blocks until explicitly released, which is what a real
    CUDA event does when the copy behind it is stream-ordered after a spin
    kernel that will not retire.  Three ranks were observed in exactly this
    state for over ten minutes.
    """

    def __init__(self):
        super().__init__(ready=False)
        import threading

        self.released = threading.Event()

    def synchronize(self) -> None:
        self.synchronized += 1
        # Capped so a broken test cannot wedge the suite the way it wedged
        # the server; the assertion below fires long before this expires.
        self.released.wait(timeout=10.0)


class TestWedgeFalsifier:
    """The bound must be the difference between returning and not returning."""

    def test_bounded_wait_returns_while_unbounded_wait_does_not(self, monkeypatch):
        import threading

        # 1. The unbounded wait -- the pre-#616f code path, still reachable
        #    with the bound disabled. It must NOT return.
        monkeypatch.setenv(ENV_SYNC_DEADLINE_MS, "0")
        stuck = _WedgedEvent()
        s_old = _stub(stuck)
        done = threading.Event()

        def _old():
            BarlinkBar1Transport._wait_ctl_event(s_old)
            done.set()

        t = threading.Thread(target=_old, daemon=True)
        t.start()
        assert done.wait(timeout=0.5) is False, (
            "the unbounded wait returned; this test no longer models the wedge"
        )
        stuck.released.set()
        t.join(timeout=5.0)

        # 2. The same never-completing event, with the bound in force.
        monkeypatch.setenv(ENV_SYNC_DEADLINE_MS, "20")
        s_new = _stub(_WedgedEvent())
        assert BarlinkBar1Transport._wait_ctl_event(s_new) is False
        assert s_new._ctl_sync_timeouts == 1


# ---------------------------------------------------------------------------
# poll_status_word -- the method whose ABSENCE made the watchdog a no-op
# ---------------------------------------------------------------------------


class _Stream:
    def __init__(self):
        self.synchronized = 0

    def synchronize(self) -> None:
        self.synchronized += 1


def _poll_stub(**kw) -> types.SimpleNamespace:
    base = dict(
        _abort_poll_active=True,
        _abort_code_seen=0,
        _ctl_dev=torch.zeros(2, dtype=torch.int32),
        _abort_poll_dst=torch.zeros(1, dtype=torch.int32),
        _abort_poll_stream=_Stream(),
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


@pytest.fixture()
def _no_cuda_stream(monkeypatch):
    """torch.cuda.stream(...) as a no-op context; the copy stays on CPU."""
    import contextlib

    monkeypatch.setattr(
        torch.cuda, "stream", lambda s: contextlib.nullcontext(), raising=False
    )


class TestPollStatusWordExists:
    def test_transport_defines_the_method(self):
        """The bug in one assertion: the walk used getattr(..., None)."""
        assert callable(getattr(BarlinkBar1Transport, "poll_status_word", None))

    def test_inactive_transport_reports_no_trip(self):
        s = _poll_stub(_abort_poll_active=False)
        assert BarlinkBar1Transport.poll_status_word(s) is False

    def test_inactive_transport_still_reports_a_remembered_trip(self):
        s = _poll_stub(_abort_poll_active=False, _abort_code_seen=1)
        assert BarlinkBar1Transport.poll_status_word(s) is True

    def test_missing_word_reports_no_trip(self):
        s = _poll_stub(_ctl_dev=None)
        assert BarlinkBar1Transport.poll_status_word(s) is False

    def test_clean_word_reports_no_trip(self, _no_cuda_stream):
        s = _poll_stub()
        assert BarlinkBar1Transport.poll_status_word(s) is False
        assert s._abort_code_seen == 0
        # It waited only for its own copy.
        assert s._abort_poll_stream.synchronized == 1

    def test_tripped_word_is_seen(self, _no_cuda_stream):
        s = _poll_stub(_ctl_dev=torch.tensor([1, 0], dtype=torch.int32))
        assert BarlinkBar1Transport.poll_status_word(s) is True
        assert s._abort_code_seen == 1

    def test_trip_is_sticky_and_stops_reading(self, _no_cuda_stream):
        s = _poll_stub(_ctl_dev=torch.tensor([1, 0], dtype=torch.int32))
        assert BarlinkBar1Transport.poll_status_word(s) is True
        reads = s._abort_poll_stream.synchronized
        # The word going back to 0 must not un-trip the mirror.
        s._ctl_dev = torch.zeros(2, dtype=torch.int32)
        assert BarlinkBar1Transport.poll_status_word(s) is True
        assert s._abort_poll_stream.synchronized == reads

    def test_the_gate_now_reaches_this_transport(self, _no_cuda_stream, monkeypatch):
        """End of the chain: poll_status_words must count a BAR1 trip."""
        from sglang.srt.distributed.device_communicators import barlink_abort_gate as m

        m.reset_for_test()
        monkeypatch.setenv("SGLANG_BARLINK_BAR1_ABORT_ENABLE", "1")

        stub = _poll_stub(_ctl_dev=torch.tensor([1, 0], dtype=torch.int32))
        # Bind the real method to the stub, exactly as a transport exposes it.
        stub.poll_status_word = lambda: BarlinkBar1Transport.poll_status_word(stub)
        m.register(stub)
        try:
            assert m.poll_status_words() == 1
        finally:
            m.reset_for_test()


# ---------------------------------------------------------------------------
# _read_status_for_check -- the watchdog mirror outranks the staged word
# ---------------------------------------------------------------------------


class TestReadStatusPrefersMirror:
    def test_mirror_short_circuits_the_staged_read(self):
        ev = _Event(ready=False)
        s = _stub(ev)
        s._ctl_defer = True
        s._abort_code_seen = 1
        s._ctl_inflight = True
        assert BarlinkBar1Transport._read_status_for_check(s) == 1
        # No wait of any kind was needed.
        assert ev.queries == 0
        assert ev.synchronized == 0

    def test_unresolved_read_returns_none_and_keeps_the_copy_in_flight(
        self, monkeypatch
    ):
        """A second copy would queue behind the same stuck kernel."""
        monkeypatch.setenv(ENV_SYNC_DEADLINE_MS, "1")
        ev = _Event(ready=False)
        s = _stub(ev)
        s._ctl_defer = True
        s._abort_code_seen = 0
        s._ctl_inflight = True
        s._ctl_lag = 99
        s._ctl_stage = torch.zeros(1, dtype=torch.int32)
        s._ctl_src = torch.zeros(1, dtype=torch.int32)
        s._wait_ctl_event = lambda: BarlinkBar1Transport._wait_ctl_event(s)
        assert BarlinkBar1Transport._read_status_for_check(s) is None
        assert s._ctl_inflight is True
        assert ev.synchronized == 0
