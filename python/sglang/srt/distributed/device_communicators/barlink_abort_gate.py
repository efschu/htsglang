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
import time
from typing import Any, List, Optional

from sglang.srt.distributed.device_communicators import lockstep_sentinel

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


# --- #867: a device fault the watchdog cannot survive ------------------------
# `poll_status_words` swallows every exception so the watchdog thread cannot
# die. That is right for a poll that hiccups and WRONG for a CUDA illegal
# memory access: once a context takes one, it is unusable, and EVERY later CUDA
# call in the process raises the same error at whatever site happens to run
# next. Swallowing it therefore does not keep serving alive -- it only decides
# that the crash will be reported somewhere innocent.
#
# W40 (2026-08-25 21:17:13) is the specimen. The first fault in the whole log is
# this poll, logged as "barlink-BAR1 status poll failed" and continued past; the
# scheduler then died in `get_cpu_copy` inside the seam capture, and the boot
# before it died in `load_cpu_copy` inside the seam RESTORE. Three sites, one
# fault, and two of them innocent -- which cost this shift two wrong roots.
#
# THIS MODULE ALREADY OWNS THIS CLASS. Its own docstring opens on #431: a run
# that tripped the cap on every collective produced a file with ZERO matching
# lines, so "nothing tripped" and "everything tripped" were indistinguishable.
# The handler below had made "a poll hiccuped" and "the CUDA context is dead"
# indistinguishable in exactly the same way.
_poison_lock = threading.Lock()
_poison: Optional[dict] = None

#: Substrings of CUDA errors that leave the context UNUSABLE. Matched on the
#: message because the driver's class does not distinguish them: torch raises
#: `AcceleratorError` for a recoverable OOM and for an illegal access alike.
_POISON_MARKERS = (
    "illegal memory access",
    "an illegal instruction",
    "misaligned address",
    "unspecified launch failure",
    "device-side assert",
)


def is_poison_error(exc: BaseException) -> bool:
    """Does this exception mean the CUDA context is gone?

    Deliberately NOT `isinstance(exc, AcceleratorError)`: torch raises that for
    an out-of-memory too, which is recoverable and must keep being swallowed.
    The message is what carries the distinction the driver makes.
    """
    return any(m in str(exc) for m in _POISON_MARKERS)


def record_poison(source: str, exc: BaseException) -> bool:
    """Record the FIRST poison-class fault. Returns True if this call recorded it.

    First wins, not last: the point of the record is the ORIGIN, and every
    later CUDA call in a poisoned process reports the same error. A record that
    tracked the newest would name the innocent site, which is the defect.
    """
    global _poison
    with _poison_lock:
        if _poison is not None:
            return False
        _poison = {
            "source": source,
            "error": str(exc),
            "type": type(exc).__name__,
            "monotonic": time.monotonic(),
        }
        return True


def poison_record() -> Optional[dict]:
    """The first poison-class fault seen in this process, or None.

    The serving path reads this to attribute its own failure: a CUDA error
    raised while this is set did not originate where it was raised.
    """
    with _poison_lock:
        return dict(_poison) if _poison is not None else None


def clear_poison_record() -> None:
    """Test-only. A real process does not recover from a poisoned context."""
    global _poison
    with _poison_lock:
        _poison = None


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
        except Exception as exc:  # a watchdog must not die
            source = f"barlink poll_status_word({type(transport).__name__})"
            if is_poison_error(exc):
                # #867: NOT survivable, so do not pretend it was. The watchdog
                # still does not raise -- raising here kills the thread and
                # loses the record -- but it stops reading devices for the rest
                # of this round (every further read returns this same error and
                # multiplies the misattribution) and it names itself as the
                # origin so the crash that lands later can point back here.
                if record_poison(source, exc):
                    logger.error(
                        "#867 barlink-BAR1 status poll hit an UNSURVIVABLE CUDA "
                        "fault: %s. The context is now unusable and every later "
                        "CUDA call in this process will raise this same error at "
                        "an unrelated site -- so treat THIS as the origin, not "
                        "the traceback that lands next. Device polling stops "
                        "here. (source=%s)",
                        exc,
                        source,
                    )
                break
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


def rearm_inline_reads() -> int:
    """Give the abort-word read back to the transports (#673 Option B).

    Clears ``_abort_poll_active`` on every registered transport, so
    ``check_aborted`` resumes the pre-#517 in-line device read. Returns how
    many transports were changed.

    WHY THIS EXISTS. ``_arm_status_poll`` hands the read to the watchdog ONCE
    at bring-up and the latch is one-way, so stopping the watchdog afterwards
    would leave the word read by nobody -- ``check_aborted`` short-circuits on
    the latch and would answer "not aborted" forever. ``should_poll_status``
    above already declares the intended degradation, *"the guard degrades to
    the #517-phase-1 behaviour rather than to blindness"*; it is evaluated only
    at bring-up, and this is what makes that sentence true at teardown too.

    THE ABORT-WORD SEMANTICS ARE NOT TOUCHED. The word stays sticky, nothing
    writes it, no code is cleared -- only WHO READS IT changes, back to the
    reader it had before #517 phase 2.

    Cost note: the in-line read is the pre-#517 per-collective device read, so
    it is paid only AFTER a stop -- i.e. during teardown, never during serving.

    Never raises: it runs on the teardown path, and a transport that objects to
    being asked must not take the others down with it.
    """
    changed = 0
    for transport in registered():
        try:
            if getattr(transport, "_abort_poll_active", False):
                transport._abort_poll_active = False
                changed += 1
        except Exception:  # noqa: BLE001 - teardown must not raise
            logger.exception("barlink abort gate: could not re-arm an in-line read")
    return changed


def registered() -> List[Any]:
    with _lock:
        return list(_transports)


def reset_for_test() -> None:
    """Drop the registry. Tests only."""
    with _lock:
        _transports.clear()


# ---------------------------------------------------------------------------
# #622: REPLAY-WINDOW ATTRIBUTION
# ---------------------------------------------------------------------------
# The gap. ``Bar1CollectiveAborted`` raised at a replay boundary says, in its
# own words, that "the kernel that aborted is inside the replayed graph and is
# NOT named here". That is honest and it is also the end of the trail: the
# 2026-08-05 21:10 and 2026-08-07 03:25 specimens both stop exactly there,
# with the only named collective being an 8-byte control-plane launch from a
# window that had already closed.
#
# Nothing in the process knew which graph was on the stream, because nobody
# recorded it. A replay makes no host calls, so the collective census (#583)
# and the launch record both stay silent through it -- but the HOST still
# knows which graph it just asked for, one frame up, at the replay call site.
# This is that fact, kept where the abort handler can read it.
#
# WHAT IT IS. Four module globals, rank-local, written at every replay launch
# and read only on the already-broken abort path.
#
# ``kind``   which replay site: "full", "breakable", "breakable/seg",
#           "draft/rung". Distinguishes the decode graph from a draft rung
#           without either site having to know about the other.
# ``key``    the runner's own key object, stored BY REFERENCE and never
#           formatted here -- formatting on the hot path would allocate a
#           string per decode step for a value almost never read. It is
#           ``repr``'d once, defensively, at abort time.
# ``index``  the segment / rung ordinal within that graph, or -1.
# ``seq``    a monotonic replay counter. Its only job is to let the reader
#           distinguish "this tag describes the replay we just aborted in"
#           from a stale tag left by a process that stopped replaying.
#
# GRAPH SAFETY. Every write happens on the HOST, at replay LAUNCH time, i.e.
# outside any capture and outside the graph itself. No device read, no
# synchronization, no collective, no ``.item()``. A captured graph cannot
# contain any of this because none of it is a CUDA op.
#
# COST. One function call and five stores per replay. No allocation: the key
# is an existing reference, the ordinals are small ints, the kind strings are
# module constants. Deliberately NOT gated on a live transport -- the gate
# test would cost about what the stores cost, and an instrument that is only
# armed in the configuration you already suspect explains nothing about the
# one you do not.
#
# NOT A LOCK. Concurrent writers would interleave, and that is accepted: the
# value is a rank-local hint on a path that is already failing, and a lock on
# the decode replay path would be a real cost paid every step to protect a
# read that happens once per crash.
_REPLAY_UNSET = "<no replay recorded>"

_replay_kind: str = ""
_replay_key: Any = None
_replay_index: int = -1
_replay_seq: int = 0


def note_replay(kind: str, key: Any = None, index: int = -1) -> None:
    """Record which captured segment this rank is replaying RIGHT NOW.

    Called from the replay call sites immediately BEFORE the launch, so that
    if the graph's kernels abort, the next host point -- the abort check at
    the very same boundary -- can name the graph they were in.
    """
    global _replay_kind, _replay_key, _replay_index, _replay_seq
    _replay_kind = kind
    _replay_key = key
    _replay_index = index
    _replay_seq += 1
    # #622: feed the lockstep sentinel's positional ring. Replays make no
    # host calls, so divergence enters exactly through WHICH graph each rank
    # selects per step — this is the only point that knows the selection.
    lockstep_sentinel.note_replay(kind, key, index)


def current_replay():
    """``(kind, key, index, seq)`` as last recorded. Diagnostics only."""
    return (_replay_kind, _replay_key, _replay_index, _replay_seq)


def format_current_replay() -> str:
    """One line naming the replay window, for the abort message.

    Never raises. This runs on an already-broken run and inside an exception
    path; a diagnostic that throws would replace the real error with its own,
    which is the failure mode #583's attribution note was written about. The
    key is arbitrary caller data, so even its ``repr`` is guarded.
    """
    try:
        if not _replay_kind:
            return _REPLAY_UNSET
        try:
            key = repr(_replay_key)
        except Exception:  # noqa: BLE001 - a repr must not break triage
            key = "<unrepresentable key>"
        if len(key) > 200:
            key = key[:200] + "..."
        part = f"{_replay_kind} key={key}"
        if _replay_index >= 0:
            part += f" index={_replay_index}"
        return f"{part} (replay #{_replay_seq} on this rank)"
    except Exception:  # noqa: BLE001 - never mask the abort
        return "<replay tag unreadable>"


def reset_replay_tag_for_test() -> None:
    """Clear the tag. Tests only."""
    global _replay_kind, _replay_key, _replay_index, _replay_seq
    _replay_kind = ""
    _replay_key = None
    _replay_index = -1
    _replay_seq = 0


def format_sibling_transports(raising: Any) -> Optional[str]:
    """The OTHER live transports' flag state, read host-only, at abort time.

    #622b, and the gap is not hypothetical. A serving process on this fork
    brings up THREE independent BAR1 transports -- ``world:0``, ``tp:0`` and
    ``dcp:0`` (2026-08-07 06:12 boot log lines 162-164, 213-215, 270-273) --
    and each owns a SEPARATE flag region and a SEPARATE round counter
    (``round_dev`` is a per-transport tensor: ``barlink_bar1_ext.cpp``
    launchers at 1249 and 1461 both take the caller's). ``check_aborts``
    raises at the FIRST transport that reports, and every instrument on the
    abort path -- capture census, own-flag snapshot, peer-flag snapshot -- is
    emitted by ``self``, i.e. by that one transport only.

    So the 2026-08-07 06:12 specimen dumped ``tp:0`` and nothing else, and the
    dump it produced is internally contradictory: every kernel writes its flag
    BEFORE entering its deadline-bearing spin, yet ``tp:0``'s snapshot shows
    every cell already at the round its spin was waiting for. A transport
    whose flags are satisfied cannot be the one that ran out its deadline on
    them. ``dcp:0`` -- 40544 all-gathers and 12768 all-reduces on that run, all
    inside the same replayed decode graph -- was never read.

    HOST-ONLY, DELIBERATELY. This runs on the abort path of an already-broken
    run, where a device read can queue behind a spin that is still going (in
    the 06:12 specimen ``_abort_flag_snapshot``'s ``cuMemcpy`` took 55 s to
    return). So this touches no device: ``_abort_peer_flag_snapshot`` is an
    ordinary host load from an already-mapped BAR window, and the status word
    is taken from the pinned STAGED copy rather than re-read. The staged value
    can therefore be one check old -- it is labelled as such rather than
    refreshed, because refreshing it is the thing that can hang.

    COVERAGE. The peer snapshot omits the reporting rank's OWN region, by
    construction. It does not need it: across the ranks of a group, every
    rank's own region appears in both peers' dumps, so the three lines
    together still cover every (block, sender) cell.

    Returns None when there is nothing to say -- one transport in the process,
    or no sibling that can be read.
    """
    parts = []
    for transport in registered():
        if transport is raising:
            continue
        group = str(getattr(transport, "group", "") or "<unnamed>")
        # Pinned host memory, not a device read. May be one staged copy old;
        # `None` means this transport never staged one.
        staged = "<never staged>"
        try:
            stage = getattr(transport, "_ctl_stage", None)
            if stage is not None:
                staged = str(int(stage[0]))
        except Exception:  # noqa: BLE001 - a diagnostic must not raise
            staged = "<unreadable>"
        snapshot = None
        try:
            fn = getattr(transport, "_abort_peer_flag_snapshot", None)
            if fn is not None:
                snapshot = fn()
        except Exception:  # noqa: BLE001 - a diagnostic must not raise
            snapshot = None
        parts.append(
            f"group {group}: staged status word {staged} "
            f"(possibly one check old, NOT re-read); "
            f"{snapshot if snapshot else 'no peer flag mapping to read'}"
        )
    if not parts:
        return None
    return (
        "barlink-BAR1 SIBLING TRANSPORT state at abort (#622b), read host-only "
        "from the rank that is about to raise. The raising transport's own "
        "dump above covers only ITS flag region and round counter; these are "
        "the other live transports in this process, whose regions and counters "
        "are independent. A sibling whose staged status word is 1, or whose "
        "flag cells disagree across (block, sender), is where the replay "
        "actually stalled -- " + " || ".join(parts)
    )


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

    #622b: what IS aggregated, just before that raise leaves this function, is
    the host-readable flag state of the transports the raise skips. See
    ``format_sibling_transports`` for why the raising transport's own dump is
    not sufficient evidence, and why this cannot be a device read.
    """
    if not _transports:
        return
    if not abort_check_enabled():
        return
    for transport in registered():
        check = getattr(transport, "check_aborted", None)
        if check is None:
            continue
        try:
            check(where)
        except Exception:
            # ``Exception``, not ``BaseException``: a KeyboardInterrupt or an
            # interpreter shutdown is not a collective fault and must not pay
            # for a flag dump. Warn-never-raise below, same discipline as every
            # other instrument on this path -- the sibling dump must not
            # replace the real error.
            try:
                line = format_sibling_transports(transport)
                if line:
                    logger.error("%s", line)
            except Exception:  # noqa: BLE001 - an instrument must not mask the abort
                pass
            raise


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
    "current_replay",
    "format_current_replay",
    "note_replay",
    "reset_replay_tag_for_test",
    "defer_enabled",
    "max_lag",
    "sync_deadline_s",
    "stall_raise_after",
    "clear_poison_record",
    "is_poison_error",
    "pause_polling",
    "poison_record",
    "record_poison",
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
