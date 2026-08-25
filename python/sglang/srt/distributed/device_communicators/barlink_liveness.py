"""Bounded waits with peer liveness for the barlink collective family.

THE DEFECT
----------
A rank that dies -- SIGKILL from the OOM killer during CUDA-graph capture is
the observed case -- leaves every surviving rank spinning. Two distinct
mechanisms produce the same picture of 100 % SM at zero PCIe traffic and an
unbounded burn of card time:

1. HOST side. Every ``dist.*`` call in this family runs on the group's gloo
   ``cpu_group``, which ``GroupCoordinator.__init__`` builds with a hardcoded
   ``timedelta(seconds=120 * 60)``. ``--dist-timeout`` reaches the NCCL
   ``device_group`` only. Two hours is not a bound anybody waits out; the
   symptom is indistinguishable from a hang. The same shape appears in
   ``torch.cuda.synchronize()`` after a spinning kernel, which has no bound
   at all.
2. DEVICE side. The BAR1 spin kernels do carry a cycle deadline
   (``SGLANG_BARLINK_BAR1_CAP_CYCLES``, ~30 s at 2 GHz), but it is
   rank-local, it is multiplied by up to 40x inside the JIT cold-build window
   (``jit_cold_build``) -- which is exactly the capture window where the OOM
   kill happens -- and its expiry writes ``ctlStatus`` into rank-local VRAM.

   Both halves of that sentence were, until #431, statements about the
   DEVICE transport only. BAR1 passed its cap constant to the kernels raw
   and never called ``resolve_timeout_cycles``, so the 40x extension did not
   in fact apply to it; and no production path read the status word, so a
   tripped kernel was silent and the stream continued over a partially
   written output buffer. ``BarlinkBar1Transport._deadline_cycles`` and
   ``check_aborted`` close both, and ``barlink_abort_gate`` carries the
   check to the one place a per-collective host check cannot reach -- the
   CUDA-graph replay boundary.

Neither mechanism can see that a peer process no longer exists. That is the
one fact that turns an indefinite wait into a decidable one.

THE SHAPE OF THE FIX
--------------------
One fact, published once, consulted only while a wait is already making no
progress:

* ``PeerTable`` exchanges ``(hostname, pid, boot marker)`` per rank at
  transport bring-up -- the same tuple shape ``jit_kernel/cache_health.py``
  already writes for build-lock ownership. Same-host peers are then decidable
  with ``os.kill(pid, 0)``; a cross-host peer is not, and is deliberately
  reported as UNKNOWN rather than guessed at (a false "dead" would abort a
  healthy group).
* ``bounded_poll`` / ``bounded_collective`` / ``bounded_barrier`` /
  ``bounded_device_sync`` replace the unbounded host waits. They raise
  ``PeerLostError`` (naming the rank, host and pid) as soon as a peer is
  provably gone, and ``CollectiveTimeoutError`` at the deadline otherwise.
* ``AbortWindow`` is one 32-bit word in pinned, device-mapped host memory.
  ``PeerWatchdog`` writes 1 into it when a peer is provably gone; the BAR1
  spin kernels read it inside their spin loops and take the abort path they
  already have. That converts the device-side wait from "up to 40 x 30 s,
  then silent" into "within one watchdog probe, then the existing loud abort
  path".

HOT-PATH COST
-------------
Nothing here runs on a collective that completes.

* ``bounded_poll`` evaluates the caller's predicate first and returns before
  it reads a clock. It then spins ``_SPIN_BUDGET`` further iterations of the
  bare predicate before the first ``time.monotonic()``. A wait that was
  already satisfied costs one extra Python call.
* The liveness probe is rate-limited to ``SGLANG_BARLINK_PEER_PROBE_S``
  (default 1 s) and only fires from a loop that is not progressing. Its cost
  is ``world_size`` ``kill(pid, 0)`` syscalls, ~1 us each.
* ``PeerWatchdog`` is one daemon thread per process. It sleeps
  ``SGLANG_BARLINK_PEER_PROBE_S`` between probes and touches no CUDA state on
  the healthy path.
* The device-side probe is a masked volatile load of one host word every
  1024 spin iterations, inside a loop that is already issuing ``R`` volatile
  BAR1 loads per iteration. It is never reached by a collective that
  completes.

DEFAULT PATH
------------
With ``SGLANG_BARLINK_PEER_LIVENESS=0`` every helper degrades to the exact
blocking call it replaced, and the abort window is never allocated, so the
kernels see a null pointer and skip the probe. With no barlink transport in
the process nothing imports this module.
"""

from __future__ import annotations

import ctypes
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# --- environment ----------------------------------------------------------

#: Kill switch. "0" restores the previous, unbounded blocking calls exactly.
ENV_ENABLE = "SGLANG_BARLINK_PEER_LIVENESS"
#: Seconds a host-side wait may make no progress before it gives up. 120 s is
#: the first cut asked for by task #312: long enough that a slow but healthy
#: rank is not cut off, short enough that the answer arrives while somebody is
#: still watching. Scaled by the JIT cold-build multiplier while that window
#: is open, so a first boot on an empty kernel cache does not trip it.
ENV_TIMEOUT_S = "SGLANG_BARLINK_PEER_TIMEOUT_S"
#: How often a stalled wait may ask whether the peers still exist.
ENV_PROBE_S = "SGLANG_BARLINK_PEER_PROBE_S"
#: Whether the watchdog thread runs. Separate from ENV_ENABLE so the bounded
#: host waits can be kept while the device-side abort is investigated.
ENV_WATCHDOG = "SGLANG_BARLINK_PEER_WATCHDOG"

_DEFAULT_TIMEOUT_S = 120.0
_DEFAULT_PROBE_S = 1.0

#: #673: how long teardown waits for the watchdog before detaching it.
#: DERIVED FROM THIS THREAD'S OWN CADENCE, not copied from the kvso-dest-io and
#: dual-group-lane siblings: ``_run`` waits ``poll_interval_s()`` (10 ms by
#: default) between passes, so a cooperative exit lands within about one tick
#: plus one poll pass. 250 ms is ~25 ticks -- ample for the loop that exists,
#: and 8x tighter than the 2 s the siblings use, which wait on a host tier
#: write and a CUDA kernel launch respectively.
WATCHDOG_JOIN_TIMEOUT_S: float = 0.25
WATCHDOG_LOG_PREFIX = "[#673 barlink-watchdog]"

#: Iterations of the bare predicate before the first clock read. Mirrors the
#: HiCache bounded-collective fix: a wait that resolves in microseconds must
#: not pay for a ``time.monotonic()``.
_SPIN_BUDGET = 512
#: Sleep backoff once the spin budget is spent, in seconds.
_SLEEP_MIN_S = 0.0005
_SLEEP_MAX_S = 0.05

_FALSE = ("0", "false", "no", "off", "")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSE


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %g.", name, raw, default)
        return default


def liveness_enabled() -> bool:
    """Read per call, not at import: tests and operators change it in place."""
    return _env_flag(ENV_ENABLE, True)


def watchdog_enabled() -> bool:
    return liveness_enabled() and _env_flag(ENV_WATCHDOG, True)


def probe_interval_s() -> float:
    return max(_env_float(ENV_PROBE_S, _DEFAULT_PROBE_S), 0.0)


def wait_timeout_s() -> float:
    """The deadline a host wait should use right now.

    Scaled by the JIT cold-build multiplier while that window is open, for
    the same reason ``resolve_timeout_cycles`` scales the device deadline:
    inside the pre-capture warmup the ranks are minutes apart in nvcc, and a
    steady-state bound would fire on a healthy group. Outside the window this
    is the configured value unchanged.
    """
    base = _env_float(ENV_TIMEOUT_S, _DEFAULT_TIMEOUT_S)
    if base <= 0:
        return 0.0
    try:
        from sglang.srt.utils import jit_cold_build

        if jit_cold_build.in_cold_build_window():
            return base * max(jit_cold_build.cold_build_timeout_mult(), 1)
    except Exception:  # pragma: no cover - the window is optional context
        pass
    return base


# --- errors ---------------------------------------------------------------


class PeerLivenessError(RuntimeError):
    """Base of the two decidable outcomes of a bounded wait."""


class PeerLostError(PeerLivenessError):
    """A peer rank is provably gone. Named, so nobody has to infer it."""


class CollectiveTimeoutError(PeerLivenessError):
    """The deadline expired with every peer still alive (or undecidable)."""


# --- peer identity --------------------------------------------------------

#: A cross-host peer's pid is meaningless locally. Rather than guess, the
#: table reports it undecidable and the deadline carries the case.
ALIVE = "alive"
DEAD = "dead"
UNKNOWN = "unknown"


def pid_alive(pid: int) -> bool:
    """Does a process with this pid exist on THIS host?

    ``PermissionError`` means it exists and belongs to somebody else, which
    is still "exists". Same rule as ``utils/stale_shm_cleanup.py``.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _hostname() -> str:
    return socket.gethostname()


def _boot_marker(pid: int) -> str:
    """Distinguish a recycled pid from the original process.

    ``/proc/<pid>`` carries the start time in field 22 of ``stat``. A pid that
    got recycled between rendezvous and probe therefore reads as a DIFFERENT
    process rather than as "still alive", which is the failure mode that
    would make the whole check useless on a busy box. Empty string where
    ``/proc`` is unavailable; the check then degrades to the plain pid test.
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            fields = f.read().rsplit(b")", 1)[-1].split()
        # fields[0] is state; the process start time is field 22 overall,
        # i.e. index 19 after the comm field has been split off.
        return fields[19].decode()
    except Exception:
        return ""


@dataclass(frozen=True)
class PeerIdentity:
    """What one rank publishes about itself, once, at bring-up."""

    rank: int
    host: str
    pid: int
    boot: str

    def describe(self) -> str:
        return f"rank {self.rank} ({self.host}, pid {self.pid})"


def local_identity(rank: int) -> PeerIdentity:
    pid = os.getpid()
    return PeerIdentity(rank=rank, host=_hostname(), pid=pid, boot=_boot_marker(pid))


class PeerTable:
    """Who the peers are, and whether they still exist.

    Built once per transport bring-up from an ``all_gather_object`` on the
    ``cpu_group`` -- the only collective in this module, and it is not on any
    hot path. Consulting it afterwards is purely local: no collective, no
    lock held across a syscall, no CUDA.
    """

    def __init__(self, entries: Sequence[PeerIdentity], self_rank: int):
        self.entries: Tuple[PeerIdentity, ...] = tuple(entries)
        self.self_rank = self_rank
        self.host = _hostname()

    # -- construction ------------------------------------------------------

    @classmethod
    def exchange(cls, cpu_group: Any, timeout_s: Optional[float] = None) -> PeerTable:
        """Publish this rank's identity and collect everyone else's.

        Bounded like every other collective here, but WITHOUT a peer check --
        there is no table yet. A peer that never shows up at bring-up is a
        deadline case by construction.
        """
        import torch.distributed as dist

        rank = dist.get_rank(cpu_group)
        world = dist.get_world_size(cpu_group)
        mine = local_identity(rank)
        gathered: List[Any] = [None] * world
        work = dist.all_gather_object(gathered, mine, group=cpu_group)
        if work is not None:  # all_gather_object returns None in current torch
            _poll_work(work, "peer-table exchange", timeout_s=timeout_s, table=None)
        entries = []
        for i, item in enumerate(gathered):
            if isinstance(item, PeerIdentity):
                entries.append(item)
            else:  # a peer on an older build: keep the slot, mark it opaque
                entries.append(PeerIdentity(rank=i, host="", pid=0, boot=""))
        return cls(entries, self_rank=rank)

    # -- queries -----------------------------------------------------------

    def state(self, entry: PeerIdentity) -> str:
        if entry.rank == self.self_rank:
            return ALIVE
        if not entry.host or entry.pid <= 0:
            return UNKNOWN
        if entry.host != self.host:
            return UNKNOWN
        if not pid_alive(entry.pid):
            return DEAD
        if entry.boot and _boot_marker(entry.pid) != entry.boot:
            # Same pid, different process: the original is gone.
            return DEAD
        return ALIVE

    def dead_peers(self) -> List[PeerIdentity]:
        return [e for e in self.entries if self.state(e) == DEAD]

    def describe_dead(self) -> str:
        dead = self.dead_peers()
        if not dead:
            return ""
        return ", ".join(e.describe() for e in dead)

    def describe(self) -> str:
        return "; ".join(f"{e.describe()} = {self.state(e)}" for e in self.entries)


# --- process-global registration -----------------------------------------
#
# The deep wait sites (a spin loop three frames inside a transport) must not
# have to thread a table through every signature. One table per process is
# also the truth: a rank is one process, and every group it belongs to has
# the same set of peer PROCESSES on this host.

_registry_lock = threading.Lock()
_tables: List[PeerTable] = []
_windows: List[AbortWindow] = []
_watchdog: Optional[PeerWatchdog] = None


def register_peer_table(table: PeerTable) -> None:
    with _registry_lock:
        if table not in _tables:
            _tables.append(table)
    logger.info("barlink peer liveness: %s", table.describe())


def registered_tables() -> List[PeerTable]:
    with _registry_lock:
        return list(_tables)


def any_dead_peers() -> List[PeerIdentity]:
    dead: List[PeerIdentity] = []
    seen = set()
    for table in registered_tables():
        for entry in table.dead_peers():
            key = (entry.host, entry.pid, entry.rank)
            if key not in seen:
                seen.add(key)
                dead.append(entry)
    return dead


def reset_for_test() -> None:
    """Drop all process-global state. Tests only."""
    global _watchdog
    with _registry_lock:
        _tables.clear()
        _windows.clear()
        wd, _watchdog = _watchdog, None
    if wd is not None:
        wd.stop()


# --- the abort window -----------------------------------------------------


class AbortWindow:
    """One 32-bit word the spin kernels can read and the host can set.

    Pinned + portable + mapped host memory, so the device address is valid
    for every context of this process and the kernels reach it with a plain
    volatile load. That is the same mechanism the host transport's data plane
    already uses; nothing new is being asked of the driver.

    CAPTURE SAFETY. The kernel side is a pure load of a fixed address that is
    baked into the graph as a kernel argument like every other pointer. No
    allocation, no host callback, no stream operation, no launch-time branch
    -- so a captured graph replays it unchanged, and the word can be set
    between replays or during one.

    Degrades rather than raises: if the runtime will not map the word, the
    device pointer stays 0, the kernels see ``nullptr`` and skip the probe,
    and only the host-side bounds remain. Losing the fast device abort is not
    a reason to refuse to boot.
    """

    def __init__(self) -> None:
        self._buf: Any = None
        self._host_ptr = 0
        self.device_ptr = 0
        self._registered = False
        self._tripped_reason: Optional[str] = None
        self._build()

    def _build(self) -> None:
        try:
            import torch

            # A 64-byte tensor, not 4: the word must own its cache line, so a
            # host store cannot false-share with anything the kernels poll.
            self._buf = torch.zeros(16, dtype=torch.int32, pin_memory=True)
            self._host_ptr = int(self._buf.data_ptr())
            self.device_ptr = _map_host_word(self._host_ptr, self._buf.numel() * 4)
            self._registered = self.device_ptr != 0
        except Exception as e:
            logger.warning(
                "barlink peer liveness: no device-mapped abort word (%s). The "
                "spin kernels keep only their cycle deadline; the host-side "
                "bounds are unaffected.",
                e,
            )
            self._buf = None
            self._host_ptr = 0
            self.device_ptr = 0

    @property
    def tripped(self) -> bool:
        return self._tripped_reason is not None

    @property
    def reason(self) -> Optional[str]:
        return self._tripped_reason

    def value(self) -> int:
        if self._buf is None:
            return 0
        return int(self._buf[0].item())

    def trip(self, reason: str) -> None:
        """Tell every spinning kernel of this process to take its abort path.

        Idempotent, and safe from any thread: a single aligned 32-bit store
        into pinned memory is what the kernels poll for, and they only ever
        compare it against zero.
        """
        if self._tripped_reason is not None:
            return
        self._tripped_reason = reason
        if self._buf is not None:
            self._buf[0] = 1
        logger.error(
            "barlink peer liveness: abort window tripped -- %s. Spinning "
            "collective kernels on this rank will take their abort path.",
            reason,
        )

    def close(self) -> None:
        if self._registered and self._host_ptr:
            try:
                _unmap_host_word(self._host_ptr)
            except Exception:
                pass
        self._registered = False
        self._buf = None
        self._host_ptr = 0
        self.device_ptr = 0


def _cuda_runtime() -> Any:
    import torch

    hip = getattr(torch.version, "hip", None) is not None
    return ctypes.CDLL("libamdhip64.so" if hip else "libcudart.so"), hip


def _drain_last_error(lib: Any, hip: bool) -> None:
    """Clear the runtime's sticky "last error" slot.

    A failed ``cudaHostRegister`` is recorded there and would be returned by
    the next unrelated ``cudaGetLastError()`` in the process -- which is how
    a benign probe here turns into somebody else's mystery error later. The
    expected failure below (the memory is already registered) is precisely
    such a probe.
    """
    fn = getattr(lib, "hipGetLastError" if hip else "cudaGetLastError", None)
    if fn is not None:
        fn.restype = ctypes.c_int
        fn()


def _map_host_word(ptr: int, nbytes: int) -> int:
    """Return the device address of the word, page-locking it if needed.

    A torch pinned tensor is allocated with ``cudaHostAlloc``, and under UVA
    that memory is already device-accessible -- so the device pointer query
    is tried FIRST and the register call is the fallback, not the other way
    round. Registering memory the allocator already pinned fails with
    "already registered", and doing that on the common path would leave a
    stale error in the runtime for an unrelated caller to trip over.

    Flags 1|2 (Portable|Mapped) have the same values in both runtimes, as the
    host transport's ``_register_portable_mapped`` documents.
    """
    lib, hip = _cuda_runtime()
    register = lib.hipHostRegister if hip else lib.cudaHostRegister
    get_dev = lib.hipHostGetDevicePointer if hip else lib.cudaHostGetDevicePointer
    register.restype = ctypes.c_int
    get_dev.restype = ctypes.c_int

    dptr = ctypes.c_void_p()
    rc = get_dev(ctypes.byref(dptr), ctypes.c_void_p(ptr), ctypes.c_uint(0))
    if rc == 0:
        return int(dptr.value or 0)
    _drain_last_error(lib, hip)

    rc = register(ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes), ctypes.c_uint(3))
    if rc != 0:
        _drain_last_error(lib, hip)
        raise RuntimeError(f"host-register of the abort word failed (rc={rc})")
    rc = get_dev(ctypes.byref(dptr), ctypes.c_void_p(ptr), ctypes.c_uint(0))
    if rc != 0:
        _drain_last_error(lib, hip)
        raise RuntimeError(f"no device pointer for the abort word (rc={rc})")
    return int(dptr.value or 0)


def _unmap_host_word(ptr: int) -> None:
    """Undo a registration this module made. A no-op for allocator-pinned
    memory, which was never registered here -- the runtime answers
    "not registered" and the error is drained rather than raised, because a
    teardown path must not be the thing that fails."""
    lib, hip = _cuda_runtime()
    fn = lib.hipHostUnregister if hip else lib.cudaHostUnregister
    fn.restype = ctypes.c_int
    if fn(ctypes.c_void_p(ptr)) != 0:
        _drain_last_error(lib, hip)


def register_abort_window(window: AbortWindow) -> None:
    with _registry_lock:
        if window not in _windows:
            _windows.append(window)


def unregister_abort_window(window: AbortWindow) -> None:
    with _registry_lock:
        if window in _windows:
            _windows.remove(window)


def trip_all_abort_windows(reason: str) -> None:
    with _registry_lock:
        windows = list(_windows)
    for window in windows:
        window.trip(reason)


# --- the watchdog ---------------------------------------------------------


class PeerWatchdog:
    """One daemon thread that turns "a peer died" into an immediate abort.

    It exists because of the device side. A host wait can check liveness in
    its own loop; a spin KERNEL cannot, and the CUDA-graph replay case has no
    host code in the collective at all. Somebody has to be awake and outside
    the collective to write the abort word, and this is that somebody.

    Deliberately narrow. It trips ONLY on a provably dead peer -- a pid that
    no longer exists on this host. It never trips on a deadline: "slow" is a
    judgement the wait sites make with their own context, "gone" is a fact.
    """

    def __init__(self, interval_s: Optional[float] = None):
        self._interval = interval_s if interval_s is not None else probe_interval_s()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.trips = 0

    def start(self) -> PeerWatchdog:
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._run, name="barlink-peer-watchdog", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, timeout_s: float = WATCHDOG_JOIN_TIMEOUT_S) -> str:
        """Stop the watchdog and GIVE THE ABORT-WORD READ BACK (#673).

        THE RE-ARM COMES FIRST, and it is not hygiene. This thread is the only
        reader of a device transport's abort word once ``_arm_status_poll`` has
        latched ``_abort_poll_active``, and that latch is one-way -- so a stop
        without the re-arm leaves the word unread by anyone and
        ``check_aborted`` answers "not aborted" forever. Clearing before the
        join also closes the window: at no point is the reader gone while the
        latch still claims it reads.

        THE HANDLE IS CLEARED ONLY ON A REAL JOIN. The previous body did
        ``thread, self._thread = self._thread, None`` BEFORE joining, so a
        timed-out join left a live thread with no record of it and a second
        stop reported success for a thread it had abandoned.

        The deadline is derived from THIS loop's cadence, not copied from the
        siblings: ``_run`` waits ``poll_interval_s()`` (10 ms by default)
        between passes, so a cooperative exit is observed within roughly one
        tick plus one poll pass. 250 ms is ~25 ticks -- an order of magnitude
        above the cadence and far below the 2 s the kvso and lane workers use,
        which wait on much slower things.

        Never raises, idempotent.
        """
        self._stop.set()
        try:
            from sglang.srt.distributed.device_communicators import barlink_abort_gate

            barlink_abort_gate.rearm_inline_reads()
        except Exception:  # noqa: BLE001 - teardown must not raise
            logger.exception(
                "%s could not re-arm the in-line abort reads; the guard may be "
                "blind from here on",
                WATCHDOG_LOG_PREFIX,
            )
        thread = self._thread
        if thread is None or not thread.is_alive():
            self._thread = None
            return "already stopped"
        thread.join(timeout=float(timeout_s))
        if thread.is_alive():
            logger.warning(
                "%s barlink-peer-watchdog did not stop within %.3fs and is "
                "being DETACHED deliberately. The in-line abort reads have "
                "already been re-armed, so the guard is not blind -- but the "
                "thread is still polling, and the transports it polls must not "
                "be closed while that is true. The handle is KEPT so the leak "
                "stays visible.",
                WATCHDOG_LOG_PREFIX,
                float(timeout_s),
            )
            return "detached"
        self._thread = None
        return "joined"

    def probe_once(self) -> List[PeerIdentity]:
        """One pass. Split out so the test drives it without a thread."""
        dead = any_dead_peers()
        if dead:
            reason = "peer rank gone: " + ", ".join(e.describe() for e in dead)
            trip_all_abort_windows(reason)
            self.trips += 1
        return dead

    def poll_abort_words(self) -> int:
        """The #517 phase-2 duty: read every BAR1/device transport's sticky
        abort word so the SERVING path never has to.

        It lives on this thread because this thread already exists, is
        already awake on a timer, and is already the process's answer to
        "somebody has to be outside the collective". The read itself is a
        four-byte D2H on the transport's private stream, so waiting for it
        waits for nothing the model is doing.

        Split out from ``probe_once`` deliberately: peer liveness is a
        host-only fact on a slow cadence, the abort word is a device read on a
        fast one, and folding them would tie the report latency of each to the
        other's natural interval.
        """
        from sglang.srt.distributed.device_communicators import barlink_abort_gate

        return barlink_abort_gate.poll_status_words()

    def _stop_on_poison(self, what: str, exc: BaseException) -> bool:
        """#867: the OUTER swallow layer, and the trap it was leaving behind.

        This thread wrapped `probe_once()` and `poll_abort_words()` in
        `except Exception: logger.exception(...)` with the same "a watchdog must
        not die" comment as the inner handler #867 fixed -- and
        `poll_abort_words` calls straight into that inner one. So the moment
        anyone made the inner layer re-raise on a poisoned context, THIS layer
        would have swallowed it again and quietly restored the original defect.
        It did not mask the fix on the day it landed; it was a trap set for the
        next editor, which is the shape this tree lost to three times in one
        night.

        A poisoned CUDA context cannot be polled out of. Every further tick
        re-reads a dead device, logs the same error and buries the origin
        deeper, at ~10 ms per tick. So: record it as the origin (first wins),
        say once that the thread is standing down, and return True so the
        caller LEAVES THE LOOP. The thread still does not raise -- raising here
        kills it and loses the record, which is what the original comment was
        right about.
        """
        from sglang.srt.distributed.device_communicators import barlink_abort_gate

        if not barlink_abort_gate.is_poison_error(exc):
            return False
        source = f"barlink liveness {what}"
        if barlink_abort_gate.record_poison(source, exc):
            logger.error(
                "#867 barlink liveness thread standing down: %s hit an "
                "UNSURVIVABLE CUDA fault (%s). The context is unusable, so "
                "every further tick would re-read a dead device and bury the "
                "origin. Treat THIS as the origin, not the traceback that "
                "lands next.",
                what,
                exc,
            )
        else:
            logger.error(
                "#867 barlink liveness thread standing down after a poisoned "
                "context (origin already recorded elsewhere): %s",
                what,
            )
        return True

    def _run(self) -> None:
        # Two duties, two cadences. The loop ticks at the FASTER one (the
        # abort poll, ~10 ms) and runs the peer probe every Nth tick, so the
        # abort report stays bounded in time without turning the liveness
        # probe -- which reads /proc -- into a 100 Hz job.
        from sglang.srt.distributed.device_communicators import barlink_abort_gate

        next_probe = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            try:
                if now >= next_probe:
                    next_probe = now + max(self._interval, 0.05)
                    if self.probe_once():
                        # The fact does not change back. Keep the thread alive
                        # so a window registered later still gets tripped, but
                        # stop re-logging: trip() is idempotent.
                        pass
            except Exception as exc:  # a watchdog must not die
                if self._stop_on_poison("peer watchdog probe", exc):
                    break
                logger.exception("barlink peer watchdog probe failed")
            try:
                self.poll_abort_words()
            except Exception as exc:  # a watchdog must not die
                if self._stop_on_poison("abort-word poll", exc):
                    break
                logger.exception("barlink abort-word poll failed")
            tick = barlink_abort_gate.poll_interval_s()
            self._stop.wait(max(min(tick, max(self._interval, 0.05)), 0.001))


def ensure_watchdog() -> Optional[PeerWatchdog]:
    """Start the process's watchdog on first use. Idempotent."""
    global _watchdog
    if not watchdog_enabled():
        return None
    with _registry_lock:
        if _watchdog is None:
            _watchdog = PeerWatchdog()
            started = _watchdog
        else:
            return _watchdog
    return started.start()


def stop_watchdog(timeout_s: float = WATCHDOG_JOIN_TIMEOUT_S) -> Optional[str]:
    """Stop the process's watchdog if one is running (#673).

    The counterpart ``ensure_watchdog`` never had. Returns the outcome, or
    None when no watchdog was ever started -- which is the common case, since
    the watchdog only exists where barlink liveness is on.

    Never raises: it runs during teardown, and it is also called from
    ``release_distributed`` as an ordering precondition, where an exception
    would abort the teardown it exists to make safe.
    """
    global _watchdog
    with _registry_lock:
        watchdog = _watchdog
    if watchdog is None:
        return None
    try:
        return watchdog.stop(timeout_s)
    except Exception:  # noqa: BLE001 - teardown must not raise
        logger.exception("%s stopping the watchdog failed", WATCHDOG_LOG_PREFIX)
        return None


def install(cpu_group: Any) -> Optional[PeerTable]:
    """Bring-up entry point: exchange identities, arm the watchdog.

    One call per transport constructor, right after the group exists. Returns
    ``None`` when the feature is switched off, so the caller's default path
    stays a plain ``if table is not None``.
    """
    if not liveness_enabled():
        return None
    try:
        table = PeerTable.exchange(cpu_group)
    except Exception as e:
        logger.warning(
            "barlink peer liveness: identity exchange failed (%s); bounded "
            "waits keep their deadline but cannot name a dead peer.",
            e,
        )
        return None
    register_peer_table(table)
    ensure_watchdog()
    return table


# --- the bounded waits ----------------------------------------------------


def _dead_peer_error(label: str, dead: Sequence[PeerIdentity], waited: float) -> str:
    who = ", ".join(e.describe() for e in dead)
    return (
        f"barlink collective '{label}' cannot complete: {who} no longer "
        f"exists (checked after {waited:.1f}s of no progress). This rank "
        f"aborts instead of spinning until the gloo process-group timeout "
        f"(7200s) expires. Set {ENV_ENABLE}=0 to restore the previous "
        f"unbounded behaviour."
    )


def _timeout_error(label: str, timeout_s: float, table: Optional[PeerTable]) -> str:
    census = f" Peer census: {table.describe()}." if table is not None else ""
    return (
        f"barlink collective '{label}' made no progress for {timeout_s:g}s and "
        f"no peer could be proven dead -- a peer is wedged, or the ranks "
        f"disagree about the collective sequence.{census} Raise "
        f"{ENV_TIMEOUT_S} if this is a legitimately slow step, or set "
        f"{ENV_ENABLE}=0 to restore the previous unbounded behaviour."
    )


def _dead_for(table: Optional[PeerTable]) -> List[PeerIdentity]:
    if table is not None:
        return table.dead_peers()
    return any_dead_peers()


def _extend_for_building_peers(
    label: str,
    waited_s: float,
    increment_s: float,
    table: Optional[PeerTable],
) -> bool:
    """Whether this wait may run one more budget because a peer is building.

    The decision itself lives in ``barlink_build_window.extension_for`` -- one
    definition, shared with the BAR1 abort guard, so a revert of the mechanism
    flips both callers rather than leaving one silently on a private copy.
    Import is lazy and failure is False: a wait must raise on its own terms if
    the visibility mechanism is absent.
    """
    try:
        from sglang.srt.distributed.device_communicators.barlink_build_window import (
            extension_for,
        )

        return extension_for(waited_s, increment_s, label, table=table) is not None
    except Exception:  # noqa: BLE001 - never let a diagnostic hold a wait open
        return False


def bounded_poll(
    ready: Callable[[], bool],
    label: str,
    *,
    timeout_s: Optional[float] = None,
    table: Optional[PeerTable] = None,
    on_abort: Optional[Callable[[], None]] = None,
    sleep: bool = True,
) -> None:
    """Spin on ``ready`` with a deadline and a peer-liveness escape.

    The success path is the predicate and nothing else: ``ready()`` is called
    before any clock is read, and ``_SPIN_BUDGET`` more times before the
    first ``time.monotonic()``. A wait that resolves immediately therefore
    costs one Python call over the bare ``while not ready(): pass`` it
    replaces.

    ``on_abort`` runs once, before the exception, for callers that must trip
    a device-side flag so their own kernels stop too.
    """
    if ready():
        return
    if not liveness_enabled():
        while not ready():
            if sleep:
                time.sleep(0)
        return

    budget = timeout_s if timeout_s is not None else wait_timeout_s()
    if budget <= 0:
        while not ready():
            if sleep:
                time.sleep(0)
        return

    # The spin budget is spent BEFORE the first clock read, not after it: a
    # wait that resolves in microseconds must not pay for a
    # ``time.monotonic()`` it will never need.
    for _ in range(_SPIN_BUDGET):
        if ready():
            return

    start = time.monotonic()
    deadline = start + budget
    probe_every = probe_interval_s()
    next_probe = start + probe_every
    nap = _SLEEP_MIN_S
    extended = False  # #615: whether a peer's build window moved the deadline

    while not ready():
        now = time.monotonic()
        if probe_every >= 0 and now >= next_probe:
            next_probe = now + max(probe_every, 0.05)
            dead = _dead_for(table)
            if dead:
                message = _dead_peer_error(label, dead, now - start)
                trip_all_abort_windows(message)
                if on_abort is not None:
                    on_abort()
                raise PeerLostError(message)
        if now >= deadline:
            if ready():
                return
            # #615: a peer that is legitimately inside a lazy JIT build has
            # published that fact, and a deadline that fires on it turns a
            # slow compiler into a wedge report. Extend by one further
            # budget, loudly, until the absolute cap -- past which this
            # raises anyway, so a rank that published a build and then wedged
            # is still caught. The extension is consulted HERE and nowhere
            # else: a collective that completes never reaches this branch.
            if _extend_for_building_peers(label, now - start, budget, table):
                deadline = now + budget
                extended = True
                continue
            # Report the TOTAL wait once a build extended it. Reporting the
            # configured budget after twelve minutes of forgiven build time
            # would send the reader to raise a knob that was never the bound.
            message = _timeout_error(
                label, (now - start) if extended else budget, table
            )
            if on_abort is not None:
                on_abort()
            raise CollectiveTimeoutError(message)
        if sleep:
            time.sleep(nap)
            nap = min(nap * 2, _SLEEP_MAX_S)


def _poll_work(
    work: Any,
    label: str,
    *,
    timeout_s: Optional[float] = None,
    table: Optional[PeerTable] = None,
) -> None:
    """Wait for a c10d ``Work`` under the same bound."""
    bounded_poll(work.is_completed, label, timeout_s=timeout_s, table=table)
    work.wait()


def bounded_collective(
    issue: Callable[[], Any],
    label: str,
    *,
    timeout_s: Optional[float] = None,
    table: Optional[PeerTable] = None,
) -> None:
    """Run one ``dist.*`` call asynchronously and wait for it under the bound.

    ``issue`` must call the collective with ``async_op=True`` and return the
    ``Work``. A ``None`` return means the backend ran it inline (torch does
    this for the object-based collectives); there is nothing left to bound
    and the call has already completed.
    """
    if not liveness_enabled():
        work = issue()
        if work is not None:
            work.wait()
        return
    work = issue()
    if work is None:
        return
    _poll_work(work, label, timeout_s=timeout_s, table=table)


def bounded_barrier(
    group: Any,
    label: str = "barrier",
    *,
    timeout_s: Optional[float] = None,
    table: Optional[PeerTable] = None,
) -> None:
    """``dist.barrier(group)`` that names the rank that did not arrive."""
    import torch.distributed as dist

    if not liveness_enabled():
        dist.barrier(group=group)
        return
    bounded_collective(
        lambda: dist.barrier(group=group, async_op=True),
        label,
        timeout_s=timeout_s,
        table=table,
    )


def bounded_device_sync(
    label: str = "device sync",
    *,
    device: Any = None,
    timeout_s: Optional[float] = None,
    table: Optional[PeerTable] = None,
) -> None:
    """``torch.cuda.synchronize()`` with a bound.

    The plain call blocks the host thread in the driver with no timeout at
    all, which is how a spinning collective kernel turns into a wedged
    PROCESS rather than a wedged stream. An event on the current stream can
    be polled instead, so the host keeps the ability to notice that a peer
    has gone and to trip the abort word that will end the kernel.
    """
    import torch

    if not torch.cuda.is_available():
        return
    if not liveness_enabled():
        torch.cuda.synchronize(device)
        return
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(device))
    bounded_poll(event.query, label, timeout_s=timeout_s, table=table)
    # The event is complete; this returns immediately and keeps the exact
    # post-condition the plain call had.
    torch.cuda.synchronize(device)


def check_peers(label: str, table: Optional[PeerTable] = None) -> None:
    """One-shot: raise if a peer is already known to be gone.

    For the points that are about to enter something unbounded and cannot be
    polled -- a blocking third-party call, a kernel launch whose result
    nobody waits for. Costs ``world_size`` syscalls.
    """
    if not liveness_enabled():
        return
    dead = _dead_for(table)
    if dead:
        message = _dead_peer_error(label, dead, 0.0)
        trip_all_abort_windows(message)
        raise PeerLostError(message)


def describe_config() -> str:
    return (
        f"{ENV_ENABLE}={int(liveness_enabled())} "
        f"{ENV_TIMEOUT_S}={wait_timeout_s():g} "
        f"{ENV_PROBE_S}={probe_interval_s():g} "
        f"{ENV_WATCHDOG}={int(watchdog_enabled())}"
    )


__all__ = [
    "ALIVE",
    "DEAD",
    "UNKNOWN",
    "AbortWindow",
    "CollectiveTimeoutError",
    "ENV_ENABLE",
    "ENV_PROBE_S",
    "ENV_TIMEOUT_S",
    "ENV_WATCHDOG",
    "PeerIdentity",
    "PeerLivenessError",
    "PeerLostError",
    "PeerTable",
    "PeerWatchdog",
    "any_dead_peers",
    "bounded_barrier",
    "bounded_collective",
    "bounded_device_sync",
    "bounded_poll",
    "check_peers",
    "describe_config",
    "ensure_watchdog",
    "install",
    "liveness_enabled",
    "local_identity",
    "pid_alive",
    "probe_interval_s",
    "register_abort_window",
    "register_peer_table",
    "registered_tables",
    "reset_for_test",
    "trip_all_abort_windows",
    "unregister_abort_window",
    "wait_timeout_s",
    "watchdog_enabled",
]
