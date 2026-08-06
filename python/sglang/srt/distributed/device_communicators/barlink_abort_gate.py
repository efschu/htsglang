# SPDX-License-Identifier: Apache-2.0
"""Where a tripped BAR1 spin kernel becomes a raised exception (task #431).

THE DEFECT
----------
``barlink_bar1_ext`` kernels break out of their flag-equality wait when
``clock64() - t0 > A.capCycles``, write ``*A.ctlStatus = 1u`` and RETURN --
leaving the output buffer partially written. The word they write is read by
``BarlinkBar1Transport.status()``, and until this module existed nothing on a
production path ever called it: ``raise_if_aborted`` was reachable only from
three bring-up proofs. The #431 window measured what that costs. A run that
tripped the cap on essentially every collective for 22 minutes produced an
``abort_*.txt`` with ZERO matching lines
(``/spinning/gpu-battery-results/2026-08-02_431_repro/abort_fp8_bar1_decode.txt``):
nothing in the process could distinguish "no kernel tripped" from "every
kernel tripped", and the stream kept running over the partial buffers.

WHY A SEPARATE MODULE
---------------------
The check has to fire at two places that must not import each other:

* the barlink dispatch sites, which already hold the transport, and
* the CUDA-graph replay boundary in ``model_executor``, which holds nothing
  and must not grow an import of ``barlink_bar1`` (ctypes, mmap, dma-buf) in
  a process that never asked for barlink.

So the registry lives here, in a module whose entire import cost is
``threading`` + ``weakref``, and the graph backends import only this.

CAPTURE SAFETY
--------------
Reading ``ctlStatus`` is a device read and therefore synchronizes; inside a
stream capture it is not merely expensive but illegal. Every check in this
family is therefore skipped while ``graph_capture_running()`` is true, and
the transport records that the launch was CAPTURED instead. The kernels those
captures contain execute on replay, and the replay boundary -- host code, no
capture in progress -- is where they get checked.

DEFAULT PATH
------------
With no BAR1 transport in the process the registry is empty and
``check_aborts`` returns after one truth test on a list. Nothing else in this
module runs.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, List

logger = logging.getLogger(__name__)

#: Environment kill switch for the whole family. "0" restores the pre-#431
#: behaviour exactly: a tripped kernel stays silent and the stream continues
#: over the partially written buffer. It exists because a check that cannot
#: be switched off is a check nobody can bisect around, not because running
#: without it is a reasonable operating point.
ENV_ENABLE = "SGLANG_BARLINK_BAR1_ABORT_CHECK"

#: How many BAR1 collectives may run on the host path between two device
#: reads of the status word. 1 = check after every collective, which is the
#: default: the word is 4 bytes, but reading it synchronizes the current
#: stream, so an operator who measures a regression on an eager path has a
#: knob instead of an all-or-nothing switch. Values <= 0 read as 1.
#:
#: REACH (#517). Until #517 this knob could not throttle the CUDA-graph
#: replay boundary at all: at a boundary nothing was launched on the host
#: path, so ``_unchecked_launches == 0`` and the gate at
#: ``barlink_bar1.check_aborted`` was entered through ``_captured_launches``,
#: below the interval test. It now counts boundary entries in their own
#: counter and applies the same interval, so K > 1 throttles BOTH seams.
#: K = 1 (the default) is unchanged in behaviour: every boundary still
#: checks.
ENV_EVERY = "SGLANG_BARLINK_BAR1_ABORT_CHECK_EVERY"

#: Whether the status word is read ASYNCHRONOUSLY (#517). ``1`` (default)
#: stages the word into pinned host memory with a non-blocking D2H on the
#: current stream and reads the PREVIOUS staged value, so a check costs one
#: copy dispatch and one ``cudaEventQuery`` instead of a stream
#: synchronization. ``0`` restores the pre-#517 blocking ``.item()`` at every
#: check exactly -- which is the bisect switch for the #476 measurement, and
#: the strictest possible reporting latency (zero checks).
#:
#: The guard does not get weaker: ``ctlStatus`` is STICKY (written ``1u`` by a
#: tripped kernel, never cleared -- ``barlink_bar1.py:2172`` is the only
#: writer on the host side and it runs once at bring-up), so a value that
#: arrives one check late is the same value. What changes is WHEN the raise
#: happens, and that is bounded by ``ENV_MAX_LAG``.
ENV_DEFER = "SGLANG_BARLINK_BAR1_ABORT_DEFER"

#: How many consecutive checks a staged read may stay unresolved before ONE
#: blocking wait is forced (#517). This is the bound that keeps the deferred
#: read a guard rather than a hope: an overlap-scheduled host can run
#: arbitrarily far ahead of the device, and without a bound the staged value
#: could remain in flight for as long as the host keeps queueing work.
#: Values <= 0 read as 1. ``1`` means "resolve at the very next check or
#: block", which is still cheaper than the pre-#517 behaviour (block at EVERY
#: check).
#:
#: Default 4: one NEXTN decode round on the measured recipe passes eight
#: checks (three CUDA-graph replay boundaries plus five host-path speculative
#: rank syncs -- see docs/dev/NOTE_517_bar1_guard_desk.md), so 4 bounds the
#: report to well inside the round that produced the abort, while being loose
#: enough that the copy resolves on its own in the steady state. It BINDS
#: whenever the host is more than four checks ahead of the device; the
#: falsifier for that is a never-ready event in
#: ``test_barlink_bar1_abort_deferred_517.py``.
#:
#: SUPERSEDED IN THE STEADY STATE (#517 phase 2), and the reason is the
#: measurement this bound produced. A check-COUNT bound presumes that "the
#: host is far ahead of the device" is a fault precursor. Under the overlap
#: scheduler at bs=1 it is the DESIGN: the host is always more than four
#: checks ahead, so this bound bound EVERY round, and the forced
#: ``event.synchronize()`` became the round's dominant host cost (11/14 py-spy
#: samples, task #600). The bound now lives in TIME (``ENV_POLL_MS``) and on
#: the watchdog thread. This knob still governs the legacy hot-path deferred
#: read, which is what a transport falls back to when the watchdog is not
#: running -- see ``should_poll_status``.
ENV_MAX_LAG = "SGLANG_BARLINK_BAR1_ABORT_MAX_LAG"

#: How often the WATCHDOG thread reads the status word, in milliseconds
#: (#517 phase 2). This is the bound that replaced ``ENV_MAX_LAG`` in the
#: steady state, and the unit change is the whole point: a check-COUNT bound
#: asks "how many collectives may run unverified", which at bs=1 is the wrong
#: question, because the overlap scheduler runs the host ahead of the device
#: BY DESIGN. Being N checks ahead is the healthy state there, so a
#: check-count bound fires on every round and turns the guard into a
#: per-round stream synchronization -- measured at bs=1 as 11 of 14 py-spy
#: samples inside ``check_aborted`` and ~7 ms of a 46.5 ms round (task #600).
#:
#: A TIME bound asks the question the guard actually promises to answer: how
#: long may a tripped kernel stay unreported. That is independent of how far
#: ahead the host is, so the steady state costs the hot path nothing at all.
#: Default 10 ms: two orders of magnitude below the round time, so a report
#: still lands inside the round that produced it, and far above the cost of
#: the poll itself (one 4-byte D2H on a private stream).
ENV_POLL_MS = "SGLANG_BARLINK_BAR1_ABORT_POLL_MS"

#: Milliseconds the hot-path staged read may block on its OWN event before it
#: gives up for this check (#616f). The legacy deferred read issues its D2H on
#: the compute stream, so the event is ordered behind the collective that just
#: ran; ``event.synchronize()`` on it is bounded by that collective, and when
#: the collective is the fault it is not bounded at all. Three ranks were
#: captured inside that exact call for over ten minutes, emitting nothing.
#: Default 2000 ms -- far above any healthy step (~46 ms) and far below the
#: supervisor's patience. ``0`` restores the pre-#616f unbounded wait.
ENV_SYNC_DEADLINE_MS = "SGLANG_BARLINK_BAR1_ABORT_SYNC_DEADLINE_MS"

#: Consecutive ``ENV_SYNC_DEADLINE_MS`` expiries before the rank raises a
#: stall (#616f). ``0`` never raises and keeps the condition a log line.
#: Default 30, i.e. a minute of a compute stream that will not retire four
#: bytes, which no healthy step produces.
ENV_STALL_RAISE_AFTER = "SGLANG_BARLINK_BAR1_STALL_RAISE_AFTER"

#: Whether the CUDA-graph replay boundary checks. Separate from ENV_ENABLE
#: because it is the one check that can cost a synchronization the host would
#: not otherwise have paid (an overlap-scheduled decode runs ahead of the
#: device on purpose). On by default -- a captured decode is exactly the case
#: where nothing else looks -- and documented so it can be turned off with a
#: reason rather than by disabling the whole family.
ENV_REPLAY = "SGLANG_BARLINK_BAR1_ABORT_CHECK_REPLAY"

_FALSE = ("0", "false", "no", "off", "")

_lock = threading.Lock()
_transports: List[Any] = []


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSE


def abort_check_enabled() -> bool:
    """Read per call, not at import: tests and operators change it in place."""
    return _env_flag(ENV_ENABLE, True)


def replay_check_enabled() -> bool:
    return abort_check_enabled() and _env_flag(ENV_REPLAY, True)


def check_every() -> int:
    raw = os.environ.get(ENV_EVERY)
    if raw is None:
        return 1
    try:
        return max(int(raw), 1)
    except ValueError:
        logger.warning("%s=%r is not an integer; using 1.", ENV_EVERY, raw)
        return 1


def defer_enabled() -> bool:
    """Whether the status read is staged asynchronously. Read per call."""
    return _env_flag(ENV_DEFER, True)


def max_lag() -> int:
    """Checks a staged read may stay unresolved before one blocking wait."""
    raw = os.environ.get(ENV_MAX_LAG)
    if raw is None:
        return 4
    try:
        return max(int(raw), 1)
    except ValueError:
        logger.warning("%s=%r is not an integer; using 4.", ENV_MAX_LAG, raw)
        return 4


def sync_deadline_s() -> float:
    """Seconds the staged read may block on its own event before giving up.

    The staged D2H is issued on the COMPUTE stream, so the event that signals
    it is ordered behind whatever collective just ran. ``event.synchronize()``
    on that event is therefore bounded by the collective -- and when the
    collective is the fault, it is not bounded at all. #616f measured three
    ranks sitting in exactly that call for over ten minutes with no report.

    So the wait gets a wall-clock deadline. Exceeding it is not itself an
    error: the read simply stays unresolved for this check, the accumulated
    window is preserved, and the watchdog's private-stream poll -- which does
    not queue behind the model -- remains the path that can still see a trip.
    0 disables the bound and restores the pre-#616f blocking wait.
    """
    raw = os.environ.get(ENV_SYNC_DEADLINE_MS)
    if raw is None:
        return 2.0
    try:
        return max(float(raw), 0.0) / 1000.0
    except ValueError:
        logger.warning(
            "%s=%r is not a number; using 2000.", ENV_SYNC_DEADLINE_MS, raw
        )
        return 2.0


def stall_raise_after() -> int:
    """Consecutive deadline expiries before the stall is raised, 0 to never.

    A single expiry is a slow step; an unbroken run of them is a compute
    stream that is not retiring a four-byte copy, which no healthy step
    produces -- a decode round on this rig is ~46 ms. The default of 30 at
    the default 2 s deadline means a minute of that before the rank gives up.

    Raising matters because the alternative is what #616f observed: the host
    thread sat in the guard, the supervisor's health probe saw nothing it
    could attribute, and the wedge stayed a black hole for ten minutes. A
    raise reaches the serving path, names the op and the byte count, and lets
    the supervisor restart a rank that is definitively not coming back.
    """
    raw = os.environ.get(ENV_STALL_RAISE_AFTER)
    if raw is None:
        return 30
    try:
        return max(int(raw), 0)
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; using 30.", ENV_STALL_RAISE_AFTER, raw
        )
        return 30


def poll_interval_s() -> float:
    """Seconds between two watchdog reads of the status word."""
    raw = os.environ.get(ENV_POLL_MS)
    if raw is None:
        return 0.010
    try:
        return max(float(raw), 0.0) / 1000.0
    except ValueError:
        logger.warning("%s=%r is not a number; using 10 ms.", ENV_POLL_MS, raw)
        return 0.010


#: Guards the watchdog's device reads against a CUDA-graph capture running in
#: ANOTHER thread. Torch captures with ``cudaStreamCaptureModeGlobal``, under
#: which a synchronizing CUDA call in any thread of the process invalidates
#: the capture -- so the poll may not merely be skipped on a best-effort
#: check, it has to be EXCLUDED for the duration. The lock does that: a
#: capture cannot begin until an in-flight poll has finished, and no poll can
#: begin while one is open.
_capture_lock = threading.Lock()
_capture_depth = 0


class _PausePolling:
    """Context manager form of the capture exclusion. Reentrant per process."""

    def __enter__(self) -> "_PausePolling":
        global _capture_depth
        with _capture_lock:
            _capture_depth += 1
        return self

    def __exit__(self, *exc) -> None:
        global _capture_depth
        with _capture_lock:
            _capture_depth -= 1
        return None


def pause_polling() -> _PausePolling:
    """Exclude the watchdog's device reads for the duration of a capture.

    Entered by ``parallel_state.graph_capture`` -- the ONE context manager
    that surrounds every CUDA-graph capture in this process, so there is one
    definition of "a capture is happening" rather than a per-backend guess.
    ``barlink.graph_capture_running()`` cannot serve here: it asks whether the
    CALLING THREAD's current stream is capturing, and the watchdog is a
    different thread, where the honest answer is always False.
    """
    return _PausePolling()


def polling_paused() -> bool:
    with _capture_lock:
        return _capture_depth > 0


def poll_status_words() -> int:
    """One watchdog pass over every live BAR1 transport's status word.

    Returns the number of transports that reported a trip on THIS pass. Each
    transport reads its own word on its own private stream, so the read is
    ordered against nothing the compute stream is doing -- which is what makes
    it free for the hot path. The word is sticky, so a poll that observes it
    late observes the same bit.

    Never raises: the raise belongs on the serving path, where the op, the
    byte count and the rank context are known. The poll only flips a flag.
    """
    if not _transports:
        return 0
    if not abort_check_enabled():
        return 0
    if polling_paused():
        return 0
    tripped = 0
    for transport in registered():
        poll = getattr(transport, "poll_status_word", None)
        if poll is None:
            continue
        try:
            if poll():
                tripped += 1
        except Exception:  # pragma: no cover - a watchdog must not die
            logger.exception("barlink-BAR1 status poll failed")
    return tripped


def should_poll_status(status_is_cuda: bool, watchdog_running: bool) -> bool:
    """Whether THIS transport hands its status read to the watchdog thread.

    Module-level and importable on purpose -- the single definition of the
    decision, the same pattern as ``should_defer_status`` and
    ``parallel_state.should_build_pynccl``.

    Both inputs are load-bearing. A status word that is not on a device has no
    synchronization to avoid, so the direct read stays (cheaper AND stricter).
    A watchdog that is NOT running would leave the word unread by anyone, so
    the transport must keep the deferred hot-path read instead -- the guard
    degrades to the #517-phase-1 behaviour rather than to blindness.
    """
    return bool(status_is_cuda) and bool(watchdog_running)


def should_defer_status(status_is_cuda: bool, defer_on: bool) -> bool:
    """Whether THIS transport stages its status read instead of blocking.

    Module-level and importable on purpose -- the single definition of the
    decision, the same pattern as ``parallel_state.should_build_pynccl``. A
    test that re-implemented the condition would keep passing after the real
    one was reverted.

    ``status_is_cuda`` is the whole point of the feature: deferring exists to
    avoid a device synchronization, and a status word that is NOT on a device
    has no synchronization to avoid. Reading such a word directly is both
    cheaper and STRICTER (zero reporting latency), so the CPU case must not
    take the deferred path -- which is also why every pre-#517 hermetic test,
    all of which carry a CPU ``_ctl_dev``, keeps its exact meaning.
    """
    return bool(status_is_cuda) and bool(defer_on)


def register(transport: Any) -> None:
    """Publish a live BAR1 transport. Called once per successful bring-up."""
    with _lock:
        if transport not in _transports:
            _transports.append(transport)


def unregister(transport: Any) -> None:
    """Withdraw a transport. Called from ``close()``, before the tensors go."""
    with _lock:
        if transport in _transports:
            _transports.remove(transport)


def registered() -> List[Any]:
    with _lock:
        return list(_transports)


def reset_for_test() -> None:
    """Drop the registry. Tests only."""
    with _lock:
        _transports.clear()


def check_aborts(where: str) -> None:
    """Raise if any live BAR1 transport's kernels took their abort path.

    The FIRST statement is the empty-registry test, and it is the whole
    default path: a process without a BAR1 transport pays one truth test on a
    list per call, which is why this is safe to put on a per-forward-step
    boundary.

    Raises the transport's own ``Bar1CollectiveAborted``. Deliberately not
    wrapped or aggregated: the first transport to report is the one whose
    buffers are corrupt, and a group-level summary would cost the reader the
    rank/op/rounds context that is the entire point.
    """
    if not _transports:
        return
    if not abort_check_enabled():
        return
    for transport in registered():
        check = getattr(transport, "check_aborted", None)
        if check is not None:
            check(where)


def check_after_graph_replay(where: str = "cuda-graph replay") -> None:
    """The replay-boundary check. See the module docstring for why it exists.

    A captured graph contains the BAR1 kernels but no host code, so the
    per-collective check cannot fire during a replay; this is the next host
    point after it. Separately switchable via ``ENV_REPLAY`` because it is
    the one place where the read can cost a synchronization that the host
    would not otherwise have paid.
    """
    if not _transports:
        return
    if not replay_check_enabled():
        return
    check_aborts(where)


__all__ = [
    "ENV_DEFER",
    "ENV_ENABLE",
    "ENV_EVERY",
    "ENV_MAX_LAG",
    "ENV_POLL_MS",
    "ENV_REPLAY",
    "abort_check_enabled",
    "check_aborts",
    "check_after_graph_replay",
    "check_every",
    "defer_enabled",
    "max_lag",
    "sync_deadline_s",
    "stall_raise_after",
    "pause_polling",
    "poll_interval_s",
    "poll_status_words",
    "polling_paused",
    "register",
    "registered",
    "replay_check_enabled",
    "reset_for_test",
    "should_defer_status",
    "should_poll_status",
    "unregister",
]
