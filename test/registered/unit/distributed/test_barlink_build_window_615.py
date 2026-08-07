# SPDX-License-Identifier: Apache-2.0
"""Hermetic tests for the group-visible build window (#615).

THE FAILURE THESE PIN
---------------------
A rank that enters a lazy JIT cold build blocks for tens of seconds to
minutes. Its peers sit in a barlink collective with a steady-state deadline
and no way to tell a compiler from a wedge. Since #616f the BAR1 abort guard
RAISES on that condition (``Bar1CollectiveStalled`` after 30 expiries of a 2 s
deadline, ~60 s), so a healthy cold boot on an empty kernel cache now aborts
the group. Before the guard it was a multi-minute silent stall. Neither
outcome is a diagnosis.

WHAT IS PINNED HERE, IN THE ORDER THAT MATTERS
----------------------------------------------
 1. A waiter whose peer has published a build does NOT raise at the deadline
    it would have raised at -- and the PAIRED control, with no marker, does.
    A test that only asserted the extension would keep passing if the
    extension became unconditional.
 2. It DOES raise at the absolute cap, marker or no marker. That is the
    can-fail proof for the bound: a rank that publishes "building" and then
    genuinely wedges is still caught.
 3. The marker is cleared on exit, INCLUDING on the exception path, at every
    production hook -- ``cold_build_window``, ``jit_build_guard``, and the
    context manager itself.
 4. The default path does no work at all: no marker directory, no stat, no
    collective, no CUDA, and the waiter's deadline arithmetic is unchanged
    when nothing ever publishes.
 5. A LEAKED marker, from a process that has died, extends nothing. Without
    this the mechanism would be a way to wedge the group permanently by
    dying at the wrong moment.

CPU only, no GPU, no torch.distributed: the transport methods are invoked
unbound against stubs, which is the pattern
``test_barlink_bar1_abort_poll_616f.py`` established for exactly this guard.
"""

from __future__ import annotations

import inspect
import os
import socket
import subprocess
import time
import types

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# THE REAL MODULES, imported -- never re-implemented here. A private copy of
# the decision in the test would keep these green across a revert of the fix.
from sglang.srt.distributed.device_communicators import (  # noqa: E402
    barlink_build_window as bw,
)
from sglang.srt.distributed.device_communicators import barlink_liveness  # noqa: E402
from sglang.srt.distributed.device_communicators.barlink_bar1 import (  # noqa: E402
    Bar1CollectiveStalled,
    BarlinkBar1Transport,
)
from sglang.srt.distributed.device_communicators.barlink_liveness import (  # noqa: E402
    CollectiveTimeoutError,
    PeerIdentity,
    PeerTable,
    bounded_poll,
)

_ENV_STALL_AFTER = "SGLANG_BARLINK_BAR1_STALL_RAISE_AFTER"
_ENV_SYNC_MS = "SGLANG_BARLINK_BAR1_ABORT_SYNC_DEADLINE_MS"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_marker_dir(tmp_path, monkeypatch):
    """Every test gets its own marker directory and a clean scan cache."""
    monkeypatch.setenv(bw.ENV_DIR, str(tmp_path / "markers"))
    monkeypatch.delenv(bw.ENV_ENABLE, raising=False)
    monkeypatch.delenv(bw.ENV_CAP_S, raising=False)
    bw.reset_for_test()
    barlink_liveness.reset_for_test()
    yield
    bw.reset_for_test()
    barlink_liveness.reset_for_test()


@pytest.fixture
def live_peer():
    """A real, live, foreign pid on this host.

    A real process rather than a mocked one on purpose: the mechanism's whole
    claim is that it needs NO cooperation from the peer -- the peer publishes
    once and then disappears into a compiler -- and the reader's only check on
    it is ``os.kill(pid, 0)``. A mock would test the mock.
    """
    proc = subprocess.Popen(["sleep", "60"])
    try:
        yield proc.pid
    finally:
        proc.kill()
        proc.wait()


@pytest.fixture
def dead_peer():
    """A pid that certainly no longer exists."""
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def _table(peer_pid: int, rank: int = 1) -> PeerTable:
    host = socket.gethostname()
    me = PeerIdentity(
        rank=0, host=host, pid=os.getpid(), boot=barlink_liveness._boot_marker(os.getpid())
    )
    peer = PeerIdentity(
        rank=rank, host=host, pid=peer_pid, boot=barlink_liveness._boot_marker(peer_pid)
    )
    return PeerTable([me, peer], self_rank=0)


def _publish_for(pid: int, reason: str = "nvcc gptq_marlin") -> None:
    """Write the marker a PEER would have written. Same writer as production.

    Uses ``marker_path`` -- the real function -- so a change to the naming
    scheme breaks these tests instead of silently making them test nothing.
    """
    path = bw.marker_path(socket.gethostname(), pid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{socket.gethostname()}\n{pid}\n{time.time()}\n{reason}\n")


# ---------------------------------------------------------------------------
# 1. publication and withdrawal (requirement c)
# ---------------------------------------------------------------------------


class TestPublication:
    def test_context_manager_publishes_then_clears(self):
        assert bw.publishing() is False
        with bw.barlink_build_window("a build"):
            assert bw.publishing() is True
            assert bw.marker_path(socket.gethostname(), os.getpid()).exists()
        assert bw.publishing() is False
        assert not bw.marker_path(socket.gethostname(), os.getpid()).exists()

    def test_marker_is_cleared_on_the_exception_path(self):
        """THE LEAK THAT WOULD MATTER MOST: a failed build must not keep
        extending its peers' deadlines for the rest of the run."""
        with pytest.raises(RuntimeError):
            with bw.barlink_build_window("a build that fails"):
                raise RuntimeError("nvcc died")
        assert bw.publishing() is False
        assert not bw.marker_path(socket.gethostname(), os.getpid()).exists()

    def test_nested_windows_clear_only_at_the_outermost(self):
        with bw.barlink_build_window("outer"):
            with bw.barlink_build_window("inner"):
                assert bw.publishing() is True
            # The inner exit must NOT withdraw: the outer build is still
            # running and the peers still need to know.
            assert bw.publishing() is True
            assert bw.marker_path(socket.gethostname(), os.getpid()).exists()
        assert bw.publishing() is False

    def test_marker_carries_host_pid_and_reason(self):
        with bw.barlink_build_window("flashinfer sampling JIT"):
            info = bw._read_marker(bw.marker_path(socket.gethostname(), os.getpid()))
        assert info is not None
        host, pid, started, reason = info
        assert host == socket.gethostname()
        assert pid == os.getpid()
        assert started > 0
        # Without the reason a log line says "a peer is building" and nothing
        # an operator can act on.
        assert reason == "flashinfer sampling JIT"

    def test_disabled_publishes_nothing(self, monkeypatch):
        monkeypatch.setenv(bw.ENV_ENABLE, "0")
        with bw.barlink_build_window("a build"):
            assert bw.publishing() is False
            assert not bw.marker_path(socket.gethostname(), os.getpid()).exists()


# ---------------------------------------------------------------------------
# 2. the production hooks (requirement c, at the sites that exist)
# ---------------------------------------------------------------------------


class TestProductionHooks:
    """The two shared wrappers every real build in this tree goes through.

    Hooking these rather than each build site is what makes the coverage
    checkable: ``cold_build_window`` carries the capture warmups, the sampler
    warmup and the grouped BAR1 build; ``jit_build_guard`` carries
    ``hicache_hash_cpp``, the KV arena stub, ``radix_tree_cpp``,
    ``hf3fs_utils``, the NCCL allocator and the render kernels.
    """

    def test_cold_build_window_publishes(self):
        from sglang.srt.utils.jit_cold_build import cold_build_window

        with cold_build_window("capture warmup"):
            assert bw.publishing() is True
        assert bw.publishing() is False

    def test_cold_build_window_clears_on_exception(self):
        from sglang.srt.utils.jit_cold_build import cold_build_window

        with pytest.raises(ValueError):
            with cold_build_window("capture warmup"):
                raise ValueError("boom")
        assert bw.publishing() is False

    def test_jit_build_guard_publishes(self):
        from sglang.jit_kernel.baton_health import jit_build_guard

        with jit_build_guard("hicache_hash_cpp_avx2"):
            assert bw.publishing() is True
        assert bw.publishing() is False

    def test_jit_build_guard_clears_on_exception(self):
        from sglang.jit_kernel.baton_health import jit_build_guard

        with pytest.raises(ValueError):
            with jit_build_guard("hicache_hash_cpp_avx2"):
                raise ValueError("compile failed")
        assert bw.publishing() is False

    def test_the_hooks_are_actually_wired_in_the_source(self):
        """Structural pin: a revert that removes the call must fail HERE too.

        The behavioural tests above would also fail, but they would fail with
        an unhelpful "publishing is False"; this one names the missing call.
        """
        from sglang.jit_kernel import baton_health
        from sglang.srt.utils import jit_cold_build

        assert "_publish_build_window" in inspect.getsource(
            jit_cold_build.cold_build_window
        )
        assert "_withdraw_build_window" in inspect.getsource(
            jit_cold_build.cold_build_window
        )
        assert "_publish_build_window" in inspect.getsource(
            baton_health.jit_build_guard
        )
        assert "_withdraw_build_window" in inspect.getsource(
            baton_health.jit_build_guard
        )


# ---------------------------------------------------------------------------
# 3. observation
# ---------------------------------------------------------------------------


class TestPeersBuilding:
    def test_a_live_peers_marker_is_seen(self, live_peer):
        table = _table(live_peer)
        assert bw.peers_building(table) == []
        _publish_for(live_peer)
        builds = bw.peers_building(table)
        assert len(builds) == 1
        assert builds[0].pid == live_peer
        assert builds[0].rank == 1
        assert "gptq_marlin" in builds[0].describe()

    def test_a_dead_peers_marker_is_ignored(self, dead_peer):
        """THE FALSIFIER FOR THE LEAK. A marker outlives a hard-killed
        process; if it still counted, one badly timed SIGKILL would extend
        every peer's deadline to the cap forever after."""
        table = _table(dead_peer)
        _publish_for(dead_peer)
        assert bw.marker_path(socket.gethostname(), dead_peer).exists()
        assert bw.peers_building(table) == []

    def test_own_marker_is_not_a_peer(self):
        table = _table(os.getpid(), rank=1)
        with bw.barlink_build_window("my own build"):
            # This rank building is not a reason for this rank to forgive its
            # own deadline -- it is not the one waiting.
            assert bw.peers_building(table) == []

    def test_cross_host_peer_is_not_guessed_at(self, live_peer):
        """Consistent with PeerTable.state, which reports UNKNOWN rather than
        guessing: the marker directory is local, so absence is not evidence."""
        peer = PeerIdentity(rank=1, host="some-other-host", pid=live_peer, boot="")
        me = PeerIdentity(rank=0, host=socket.gethostname(), pid=os.getpid(), boot="")
        table = PeerTable([me, peer], self_rank=0)
        _publish_for(live_peer)
        assert bw.peers_building(table) == []

    def test_disabled_sees_nothing(self, live_peer, monkeypatch):
        table = _table(live_peer)
        _publish_for(live_peer)
        assert len(bw.peers_building(table)) == 1
        monkeypatch.setenv(bw.ENV_ENABLE, "0")
        assert bw.peers_building(table) == []

    def test_no_table_no_scan(self):
        """With no peer table registered there is nobody to be building."""
        assert bw.peers_building() == []


# ---------------------------------------------------------------------------
# 4. the BAR1 abort guard: extension and cap (requirements a and b)
# ---------------------------------------------------------------------------


class _Event:
    """A CUDA event that never resolves -- the wedged-stream stand-in."""

    def __init__(self, ready: bool = False):
        self.ready = ready
        self.synchronized = 0

    def query(self) -> bool:
        return self.ready

    def synchronize(self) -> None:
        self.synchronized += 1


def _stub(table=None, event=None) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        _ctl_event=event if event is not None else _Event(ready=False),
        _ctl_sync_timeouts=0,
        _ctl_stall_run=0,
        _ctl_build_deferred_s=0.0,
        _peer_table=table,
        group="tp",
        rank=0,
        _last_op="all_reduce",
        _last_nbytes=12584960,
        _deferred_launches=3,
    )


def _expire(stub, times: int):
    """Run the bounded poll ``times`` times. Returns nothing; may raise."""
    for _ in range(times):
        BarlinkBar1Transport._wait_ctl_event(stub)


class TestBar1StallExtension:
    def test_control_no_marker_still_raises_at_the_old_deadline(
        self, live_peer, monkeypatch
    ):
        """THE PAIRED CONTROL, and it comes first on purpose.

        Without it, a change that made the extension unconditional -- i.e.
        that disabled the #616f guard outright -- would pass every other test
        in this class.
        """
        monkeypatch.setenv(_ENV_SYNC_MS, "1")
        monkeypatch.setenv(_ENV_STALL_AFTER, "3")
        stub = _stub(table=_table(live_peer))
        # No marker published.
        with pytest.raises(Bar1CollectiveStalled):
            _expire(stub, 3)

    def test_a_building_peer_prevents_the_raise(self, live_peer, monkeypatch):
        """REQUIREMENT (a). Same stub, same deadline, same expiry count -- the
        only difference from the control above is the peer's marker."""
        monkeypatch.setenv(_ENV_SYNC_MS, "1")
        monkeypatch.setenv(_ENV_STALL_AFTER, "3")
        _publish_for(live_peer)
        stub = _stub(table=_table(live_peer))
        _expire(stub, 3)  # must not raise
        assert stub._ctl_build_deferred_s > 0.0
        # The run is reset so the next stall run starts clean, exactly as a
        # resolved read would leave it.
        assert stub._ctl_stall_run == 0

    def test_the_extension_is_logged_loudly_and_names_the_peer(
        self, live_peer, monkeypatch, caplog
    ):
        monkeypatch.setenv(_ENV_SYNC_MS, "1")
        monkeypatch.setenv(_ENV_STALL_AFTER, "3")
        _publish_for(live_peer, reason="nvcc hicache_hash_cpp")
        stub = _stub(table=_table(live_peer))
        with caplog.at_level("WARNING"):
            _expire(stub, 3)
        text = caplog.text
        # Silence is what made #616 a black hole; an extension that is not
        # logged makes a 12-minute boot look like a hang that got lucky.
        assert "building" in text
        assert str(live_peer) in text
        assert "nvcc hicache_hash_cpp" in text
        assert bw.ENV_CAP_S in text

    def test_the_hard_cap_raises_anyway(self, live_peer, monkeypatch):
        """REQUIREMENT (b), the can-fail proof for the bound.

        The peer's marker stays in place for the whole test. If the cap were
        not applied -- or were applied per extension instead of to the total
        -- this loop would never raise and the test would hang rather than
        fail, so the iteration count is bounded and the assertion is on the
        raise having happened.
        """
        monkeypatch.setenv(_ENV_SYNC_MS, "1")
        monkeypatch.setenv(_ENV_STALL_AFTER, "3")
        # Three expiries at 1 ms each is a ~3 ms increment; a 4 ms cap is
        # crossed on the second extension.
        monkeypatch.setenv(bw.ENV_CAP_S, "0.004")
        _publish_for(live_peer)
        stub = _stub(table=_table(live_peer))
        with pytest.raises(Bar1CollectiveStalled) as ei:
            _expire(stub, 60)
        # The report must account for the time the build already bought,
        # otherwise the operator reads a 60 s stall that was really minutes.
        assert ei.value.waited_s > 0.0
        assert ei.value.op == "all_reduce"

    def test_a_zero_cap_disables_extension_but_not_publication(
        self, live_peer, monkeypatch
    ):
        """The bisect switch: markers keep being written, nothing honours
        them. This is what separates "the marker is not being written" from
        "the marker is not being read" when a wedge is under investigation."""
        monkeypatch.setenv(_ENV_SYNC_MS, "1")
        monkeypatch.setenv(_ENV_STALL_AFTER, "3")
        monkeypatch.setenv(bw.ENV_CAP_S, "0")
        _publish_for(live_peer)
        stub = _stub(table=_table(live_peer))
        with pytest.raises(Bar1CollectiveStalled):
            _expire(stub, 3)
        assert bw.peers_building(_table(live_peer))  # still visible

    def test_a_resolved_read_clears_the_forgiven_total(self, live_peer, monkeypatch):
        """The cap is per STALL, not per process lifetime: a stream that
        recovers must not carry a spent budget into the next incident."""
        monkeypatch.setenv(_ENV_SYNC_MS, "1")
        monkeypatch.setenv(_ENV_STALL_AFTER, "3")
        _publish_for(live_peer)
        event = _Event(ready=False)
        stub = _stub(table=_table(live_peer), event=event)
        _expire(stub, 3)
        assert stub._ctl_build_deferred_s > 0.0
        event.ready = True
        BarlinkBar1Transport._wait_ctl_event(stub)
        assert stub._ctl_build_deferred_s == 0.0

    def test_extension_adds_no_device_wait(self, live_peer, monkeypatch):
        """The asymmetry the wedge census made load-bearing: bounded polls
        recovered, unbounded host syncs were fatal. The extension must never
        turn this bounded poll into a blocking wait."""
        monkeypatch.setenv(_ENV_SYNC_MS, "1")
        monkeypatch.setenv(_ENV_STALL_AFTER, "3")
        _publish_for(live_peer)
        event = _Event(ready=False)
        stub = _stub(table=_table(live_peer), event=event)
        _expire(stub, 9)
        assert event.synchronized == 0


# ---------------------------------------------------------------------------
# 5. the host waiter (requirement a and b, second caller)
# ---------------------------------------------------------------------------


class TestBoundedPollExtension:
    def test_control_no_marker_raises_at_its_budget(self, live_peer):
        table = _table(live_peer)
        t0 = time.monotonic()
        with pytest.raises(CollectiveTimeoutError):
            bounded_poll(lambda: False, "all_reduce", timeout_s=0.05, table=table)
        assert time.monotonic() - t0 < 0.5

    def test_a_building_peer_extends_past_that_budget(self, live_peer, monkeypatch):
        """REQUIREMENT (a) for the host waiter, and (b) in the same run.

        Run on a THREAD with a bounded join, deliberately. The failure mode
        this test guards against -- a cap that is not applied -- does not make
        the wait raise late, it makes it never raise at all, and a test that
        called ``bounded_poll`` inline would HANG instead of failing. A hang
        is not a test result; the join turns it into one.
        """
        import threading

        monkeypatch.setenv(bw.ENV_CAP_S, "0.4")
        _publish_for(live_peer)
        table = _table(live_peer)
        outcome = {}

        def run():
            t0 = time.monotonic()
            try:
                bounded_poll(lambda: False, "all_reduce", timeout_s=0.05, table=table)
                outcome["raised"] = None
            except CollectiveTimeoutError:
                outcome["raised"] = CollectiveTimeoutError
            outcome["elapsed"] = time.monotonic() - t0

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(10.0)
        # (b) THE CAP. Without it this thread is still spinning.
        assert not t.is_alive(), "bounded_poll never returned: the cap did not bind"
        assert outcome["raised"] is CollectiveTimeoutError
        # (a) THE EXTENSION. Several increments past the single 50 ms budget.
        assert outcome["elapsed"] > 0.2

    def test_a_ready_predicate_never_consults_the_mechanism(self, live_peer, tmp_path):
        """The extension lives on the failure path only."""
        _publish_for(live_peer)
        bw.reset_for_test()
        bounded_poll(lambda: True, "all_reduce", timeout_s=0.05, table=_table(live_peer))
        # Nothing was scanned, so nothing was cached.
        assert bw._scan_at == 0.0


# ---------------------------------------------------------------------------
# 6. the default path (requirement d)
# ---------------------------------------------------------------------------


class TestDefaultPathUnchanged:
    def test_no_window_means_no_directory_at_all(self):
        """Nothing is created, so nothing is stat'ed, so a process that never
        builds pays literally zero."""
        assert not bw.marker_dir().exists()

    def test_resolve_timeout_cycles_is_still_the_identity_outside_a_window(self):
        """The pre-#615 backward-compatibility claim of jit_cold_build, which
        this change must not have disturbed."""
        from sglang.srt.utils.jit_cold_build import (
            in_cold_build_window,
            resolve_timeout_cycles,
        )

        assert in_cold_build_window() is False
        assert resolve_timeout_cycles(60_000_000_000) == 60_000_000_000

    def test_wait_ctl_event_is_unchanged_when_nobody_builds(self, monkeypatch):
        """The #616f behaviour verbatim: expiry count, return value, raise."""
        monkeypatch.setenv(_ENV_SYNC_MS, "1")
        monkeypatch.setenv(_ENV_STALL_AFTER, "3")
        stub = _stub(table=None)
        assert BarlinkBar1Transport._wait_ctl_event(stub) is False
        assert BarlinkBar1Transport._wait_ctl_event(stub) is False
        with pytest.raises(Bar1CollectiveStalled) as ei:
            BarlinkBar1Transport._wait_ctl_event(stub)
        assert ei.value.expiries == 3
        assert stub._ctl_build_deferred_s == 0.0
        # And the reported wait is the pre-#615 arithmetic exactly.
        assert ei.value.waited_s == pytest.approx(3 * 0.001)

    def test_the_mechanism_touches_no_cuda_and_no_collective(self):
        """STRUCTURAL. The requirement is 'pure peer-table writes plus local
        deadline arithmetic', and the way that requirement fails in practice
        is somebody adding a barrier to agree on the build state. Pin it at
        the source: this module may not reference torch, CUDA or a collective
        at all."""
        src = inspect.getsource(bw)
        for forbidden in (
            "import torch",
            "torch.",
            "all_gather",
            "all_reduce",
            "barrier",
            "broadcast",
            "cuda",
        ):
            assert forbidden not in src, f"{forbidden!r} appeared in barlink_build_window"

    def test_the_scan_is_not_on_any_hot_path(self):
        """``peers_building`` is called from exactly two places, both of them
        a wait that has already hit its deadline. A third caller is not
        forbidden, but it must be a deliberate act, not a merge accident."""
        from sglang.srt.distributed.device_communicators import barlink_bar1

        assert "extension_for" in inspect.getsource(
            barlink_bar1.defer_stall_for_building_peer
        )
        assert "extension_for" in inspect.getsource(
            barlink_liveness._extend_for_building_peers
        )
        # And the BAR1 caller sits BELOW the stall threshold test, i.e. after
        # a minute of stall, not on every check.
        wait_src = inspect.getsource(BarlinkBar1Transport._wait_ctl_event)
        assert wait_src.index("stall_raise_after") < wait_src.index(
            "defer_stall_for_building_peer"
        )
