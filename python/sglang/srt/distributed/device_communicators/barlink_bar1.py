# SPDX-License-Identifier: Apache-2.0
"""barlink sub-path: BAR1 direct transport.

The source card writes via DMA directly into the destination card's BAR1
aperture. No host memory, no NIC, no NCCL. The path has byte-level proof
(see ``BYTE_BELEG_DMABUF.md``, four directions, ``bad_bytes = 0``), but
**there is not a single timing measurement** for it -- every rate in this
repo comes from the static-window path. This module builds the path; whether
it pays off is for the measurement it enables to decide.

Setup sequence, run exactly once at startup
--------------------------------------------
1. **Receive buffer via CUDA VMM** (``cuMemCreate`` + ``cuMemAddressReserve``
   + ``cuMemMap`` + ``cuMemSetAccess``). Not ``cudaMalloc``: only a VMM
   allocation can be exported as a dma-buf.
2. **Create the dma-buf fd** (``cuMemGetHandleForAddressRange``,
   ``CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD``). The fd stays open.
3. **Hand the fd to every peer** via ``SCM_RIGHTS`` on an AF_UNIX socket. An
   fd is process-local; it cannot travel over gloo.
4. **Every peer attaches** -- ``/dev/dmabuf_holder``, ``dma_buf_attach`` +
   ``dma_buf_map_attachment``. **Only this step actually programs the
   destination card's BAR1 pages** (``nv-dmabuf.c:1066``). An open fd
   without an importer is demonstrably not enough: the pattern scan found
   nothing across 65,536 probes, but the hit appeared immediately once the
   attach happened. The attach runs as the **source card** -- it is the
   device that will later write, which keeps the topology check and the
   IOMMU domain consistent.
5. **BAR1 offset from the sg-table** the holder returns -- no pattern scan
   needed. The offset is the difference from the BAR1 base read out of
   sysfs.
6. **Map and register the window**: the destination card's ``resource1_wc``,
   only the needed slice (an mmap over a 32-GiB window fails with
   ``EINVAL``), then ``cudaHostRegister(..., cudaHostRegisterIoMemory)`` and
   ``cudaHostGetDevicePointer`` on the **source card**.
7. **Byte-level proof for each directed pair.** Write a pattern, read it back
   on the destination card through its **own** VMM pointer, compare every
   byte. Not read back through the aperture -- otherwise a broken path could
   mask its own failure.

After this, the peer-pointer table is fixed. **Nothing is mapped and nothing
is registered on the hot path** -- that is the expensive part, and it costs
about 7 us here.

Non-negotiable requirements
----------------------------
* **Clean fallback.** If the patch, the holder module, or peer capability is
  missing, the transport withdraws: ``handles(...)`` returns ``False``. It
  does not fail outright and it changes nothing without an explicit choice.
  A transport that REQUIRES a patch would be unusable on most machines.
* **No silent stand-in.** Where a driver capability that does not yet exist
  is missing, a ``NotImplementedError`` with a reason is raised -- not a
  stub that appears to work.
* **Window limit.** Only what is mapped at the same time is reachable. The
  requirement is computed explicitly (``window_requirement``) and checked at
  startup against what can actually be exported -- not against the gross
  size reported by sysfs.
* **Phase-separated only.** The relaxed driver guard exists because of a
  documented full-duplex deadlock over BAR1 (bug 1571948). Until
  counter-traffic has been verified over a full collective's duration, the
  decomposition must not write in both directions at the same time.

The collectives
----------------
``all_reduce`` runs over two kernels ported from the probe
``/spinning/nvidia-open-595/bar1_kollektiv.cu`` (source and translation:
``barlink_bar1_ext.py``):

``mesh``
    Reduce-scatter + all-gather over ALL pairs, **two** barriers.
``ring``
    Ring-reduce-scatter + ring-all-gather, **2(R-1)** barriers.

Both have byte-level proof in the probe and are measured against NCCL
(three ranks, p50, full operation duration, uncached):

====== ========== ======== ========= =======
Size    best arm      us   NCCL us  Factor
====== ========== ======== ========= =======
 20 KiB hub          28.22     41.75   1.48x
 80 KiB mesh         50.81     73.58   1.45x
  1 MiB ring        328.60    372.79   1.13x
  4 MiB mesh       1301.05   1356.69   1.04x
 16 MiB ring       4077.43   5172.83   1.27x
====== ========== ======== ========= =======

**The 20-KiB winner ``hub`` is NOT ported here.** It is not a decomposition
but a role (one rank collects everything and redistributes the result), it
needs R full buffers per hub instead of chunk slots, and the planner in
``barlink_matrix.py`` already has its own algorithm ``star`` for that. At
20 KiB, ``mesh`` sits at 31.67 us in the same measurement -- the loss
against ``hub`` is small, but the cost of a second memory geometry (R full
buffers instead of chunk slots) is not. Anyone who wants ``star`` today
falls back to the non-BAR1 path; ``handles`` says so with ``False``, it is
never silently substituted with ``mesh``.

Which of the two kernels runs at which size is decided **not** by this
module but by the plan from ``barlink_matrix.py``. That is the conclusion of
the measurement itself: between 80 KiB and 16 MiB, ``MESSUNG_ALLES_IM_SELBEN_LAUF.md``
reports "no clean threshold" (mesh 330.30 vs. ring 326.57 us at 1 MiB), and
the ring advantage at 1 MiB falls within what cannot be distinguished from
noise without repeated runs. Without a plan, the emergency threshold
``SGLANG_BARLINK_BAR1_RING_THRESHOLD`` applies (default 1 MiB) -- a default,
not a measured conclusion.

**Data types.** Only ``float32`` is measured. The kernels additionally
handle ``float16`` and ``bfloat16``, because the access width (128 bits) --
and thus the measured part of the path -- stays the same, and only the
interpretation of the 16 bytes changes. There is no timing measurement for
these.

``all_to_all``
----------------
``all_to_all_single`` runs over a **third** kernel (``a2a``) that has no
counterpart in the probe and was therefore written from scratch rather than
ported. Rank r writes its block for rank j directly into that rank's
receive slot -- one step, all sends in the same flat index space, then
**one** barrier. No host detour, no re-mapping.

Three properties that set it apart from ``all_reduce``:

* **No reduction, hence no data type.** The kernel moves bytes. fp8, bf16,
  int32, uint8 -- one path. That sm_86 cards lack fp8 conversion
  instructions (those start at sm_89) is irrelevant here.
* **Unequal block sizes are the normal case.** The number of tokens per
  expert varies; ``send_bytes``/``recv_bytes`` arrive per rank. If a
  block does not fit in a slot, the transport withdraws via
  ``supports_a2a`` instead of failing.
* **Double slots instead of a second barrier.** ``2(R-1)`` slots, with the
  round number selecting the half. Rationale under ``geometry`` and in the
  kernel.

**Unmeasured.** To date there is **not a single timing measurement** for
this kernel -- only the byte-level proof (``byte_proof_a2a``, both
uniform and skewed, every byte, every directed pair). The table above
applies to ``all_reduce`` and only to it.

On the same kernel, without its own timing measurement
---------------------------------------------------------
``all_gather`` and ``broadcast``. Both are special cases of the same a2a
table -- all_gather with a constant send offset, broadcast with exactly one
sender -- and therefore need no second kernel. Neither is measured; each has
proof (``byte_proof_a2a``, ``byte_proof_broadcast``). The timing table
above continues to apply exclusively to ``all_reduce``.

What is NOT in here
----------------------
``reduce_scatter``. It needs a REDUCTION, and the a2a kernel moves bytes.
The RS phase of the all_reduce kernels could do it, but only as its own
entry point with its own slot set -- not as a side effect. ``handles``
returns ``False`` for it and ``_no_collective`` explains what is missing.
"""

from __future__ import annotations

import ctypes
import fcntl
import logging
import mmap
import os
import struct
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

from sglang.srt.distributed.device_communicators import (
    barlink_abort_gate,
    barlink_env_guard,  # noqa: F401  (rejects retired SGLANG_HTCCL* vars)
    barlink_liveness,
)
from sglang.srt.distributed.device_communicators.barlink_liveness import (
    PeerLivenessError,
    bounded_barrier,
    bounded_device_sync,
    check_peers,
)

# NOT a module-level `from sglang.srt.utils.jit_cold_build import ...`, for the
# reason barlink_device.py:43-49 spells out: importing the `sglang.srt.utils`
# PACKAGE runs its __init__, which creates a CUDA context, and this file must
# stay importable on a rank that has none. By the time a collective is
# launched the module is long since imported, so the lookup below is a cached
# global read.
_resolve_timeout_cycles = None


def resolve_timeout_cycles(base_cycles: int) -> int:
    """Deadline for a device collective right now -- see utils/jit_cold_build.

    Byte-for-byte the same lazy-lookup shape as ``barlink_device.py:53``, and
    that is the point of #431 fix 1: BAR1 is the transport whose kernels spin
    on a cycle deadline, and it was the one transport that did not consult
    this resolver. Outside the cold-build window the function is the
    identity, so the steady-state path is unchanged.
    """
    global _resolve_timeout_cycles
    if _resolve_timeout_cycles is None:
        from sglang.srt.utils.jit_cold_build import (
            resolve_timeout_cycles as _resolver,
        )

        _resolve_timeout_cycles = _resolver
    return _resolve_timeout_cycles(base_cycles)


logger = logging.getLogger(__name__)


class Bar1Unavailable(RuntimeError):
    """The BAR1 path is not available on this machine.

    Raised during setup and translated by the caller into
    ``handles() == False``. ALWAYS carries the reason, because in this
    project "doesn't work" is worthless without evidence.
    """


class Bar1KernelAborted(PeerLivenessError):
    """A spin kernel took its abort path, and somebody finally asked.

    Derived from ``PeerLivenessError`` because it is the device-side half of
    the same fact the bounded host waits report: the collective did not
    complete. The distinction matters to a caller that wants to treat "this
    group is finished" uniformly, whichever side noticed first.
    """


class Bar1CollectiveStalled(Bar1KernelAborted):
    """The compute stream stopped retiring work, and no kernel tripped (#616f).

    A DIFFERENT fact from ``Bar1CollectiveAborted``, which reports a kernel
    that took its abort path and left a partial buffer behind. Here nothing
    tripped: the status word stayed clean and the watchdog's private-stream
    poll kept saying so, while the compute stream simply stopped making
    progress -- the shape of the #616 wedge, where three ranks spun at 100 %
    SM with a clean abort word for over ten minutes.

    It derives from ``Bar1KernelAborted`` so that a caller which already
    treats "this group is finished" uniformly keeps working, and carries the
    same structured attribution, because the question is still WHICH
    collective stopped.
    """

    def __init__(self, group, rank, op: str, nbytes: int, waited_s: float,
                 expiries: int):
        self.group = group
        self.rank = rank
        self.op = op
        self.nbytes = int(nbytes)
        self.waited_s = float(waited_s)
        self.expiries = int(expiries)
        super().__init__(
            f"barlink-BAR1 group {group} rank {rank}: collective stalled. "
            f"The staged status read failed to resolve {expiries} consecutive "
            f"times over ~{waited_s:.1f} s, so the compute stream has not "
            f"retired a four-byte copy in that window. Last launch: "
            f"op={op} nbytes={nbytes}. The abort word is CLEAN -- no kernel "
            f"took its abort path -- so this is a stall, not a trip: the "
            f"collective is waiting on a peer contribution that is not "
            f"arriving. Compare last_op/nbytes across ranks; they must match "
            f"pairwise for a collective to complete."
        )


class Bar1PeerLost(Bar1KernelAborted):
    """A peer process is GONE, and this rank was waiting on its contribution.

    A THIRD fact, distinct from both siblings, and the distinction is the
    whole point of #818:

      ``Bar1CollectiveAborted``  a kernel took its abort path -- something
                                 tripped, and there is a partial buffer.
      ``Bar1CollectiveStalled``  nothing tripped and nobody could be proven
                                 dead; a peer is merely not arriving.
      ``Bar1PeerLost``           nobody tripped either, but a peer's PROCESS
                                 no longer exists, so the contribution this
                                 rank is waiting for can never arrive.

    Reporting the third case as the second is what cost the instance on
    2026-08-22/23: the survivors' only escalation was ``Bar1CollectiveStalled``
    after N CONSECUTIVE expiries, and the message they actually emitted said
    "the compute stream has not retired the copy" -- true, and useless, because
    the reason it had not was that PP1 was dead. A stall says "wait longer"; a
    lost peer says "stop now". They must not share a name.

    Derived from ``Bar1KernelAborted`` (hence from ``PeerLivenessError``) so
    that every existing handler of the family -- ``barlink.py``'s
    ``except barlink_liveness.PeerLivenessError``, and the phase-flip runtime's
    ``except (CollectiveTimeoutError, PeerLostError)`` -- keeps working without
    a new except clause anywhere.

    It carries the peer IDENTITY, which neither sibling does. ``waited_s`` is
    how long this rank had made no progress when the peer was proven gone, not
    how long the peer had been dead: nothing here can know the latter.
    """

    def __init__(self, group, rank, op: str, nbytes: int, waited_s: float,
                 peers: str):
        self.group = group
        self.rank = rank
        self.op = op
        self.nbytes = int(nbytes)
        self.waited_s = float(waited_s)
        self.peers = peers
        super().__init__(
            f"barlink-BAR1 group {group} rank {rank}: peer lost. {peers} no "
            f"longer exists, and this rank has been waiting {waited_s:.1f} s "
            f"for a staged status word that the dead peer's contribution will "
            f"never let the compute stream retire. Last launch: op={op} "
            f"nbytes={nbytes}. This rank aborts NOW instead of waiting for "
            f"the consecutive-expiry ceiling, which a dead peer need never "
            f"reach: every resolved read in between clears the run. Set "
            f"SGLANG_BARLINK_PEER_LIVENESS=0 to restore the previous "
            f"wait-forever behaviour."
        )


def raise_if_peer_lost(transport, waited_s: float) -> None:
    """Abort this wait if a peer's process is provably gone (#818).

    MODULE-LEVEL and taking the transport, for the same load-bearing reason
    ``defer_stall_for_building_peer`` is: the guard's methods are invoked
    UNBOUND against stubs by their tests, so a method call from inside
    ``_wait_ctl_event`` would resolve against the stub and raise
    ``AttributeError`` on the very path the stub exists to exercise.

    Costs ``world_size`` ``os.kill(pid, 0)`` syscalls, and only at the cadence
    the liveness module already defines for its own bounded waits. No device
    read, no sync, no allocation -- deliberately, for the reason
    ``defer_stall_for_building_peer`` records: on this path the only
    fatal-capable events found by the wedge census were the unbounded host
    syncs, so a guard against a wedge must not introduce one.

    Silent no-op when liveness is disabled or no peer table was installed, so
    the pre-#818 behaviour is recoverable by one env var and this returns
    control to the existing stall ladder untouched.
    """
    table = getattr(transport, "_peer_table", None)
    if table is None:
        return
    try:
        if not barlink_liveness.liveness_enabled():
            return
        dead = table.dead_peers()
    except Exception:  # noqa: BLE001 - a diagnostic must not hold a wedge open
        return
    if not dead:
        return
    who = ", ".join(e.describe() for e in dead)
    message = (
        f"barlink-BAR1 group {getattr(transport, 'group', '?')} rank "
        f"{getattr(transport, 'rank', '?')}: peer lost -- {who}"
    )
    # Trip the device-mapped abort words FIRST, exactly as
    # ``barlink_liveness.check_peers`` does. The OTHER survivor may be spinning
    # inside a device kernel where no Python exception can reach it; the abort
    # word is the only channel that does. Raising without tripping would fail
    # this rank fast and leave the third one hanging -- which is the bug, moved
    # rather than fixed.
    try:
        barlink_liveness.trip_all_abort_windows(message)
    except Exception:  # noqa: BLE001 - never let the notify path eat the raise
        pass
    raise Bar1PeerLost(
        getattr(transport, "group", "?"),
        getattr(transport, "rank", "?"),
        getattr(transport, "_last_op", None) or "unknown",
        int(getattr(transport, "_last_nbytes", 0) or 0),
        waited_s,
        who,
    )


def defer_stall_for_building_peer(transport, run: int, deadline_s: float) -> bool:
    """Forgive one stall run because a PEER published a build window (#615).

    Returns True when the escalation is deferred -- the caller then leaves the
    read in flight and returns False, exactly as it does for any unresolved
    read -- and False when it must raise.

    MODULE-LEVEL, not a method, and that is load-bearing. The guard's methods
    are invoked UNBOUND against stubs by their tests
    (``test_barlink_bar1_abort_poll_616f.py``, "constructing a real transport
    needs BAR1, peers and a device"), so a method call from inside
    ``_wait_ctl_event`` would resolve against the stub and raise
    ``AttributeError`` on the very path the stub exists to exercise. A plain
    function takes the transport as an argument and works for both.

    THE BOUND. ``_ctl_build_deferred_s`` accumulates every forgiven run and is
    checked against ``build_cap_s()`` BEFORE the next one is granted, so the
    total a building peer can buy from this guard is the cap -- not the cap
    per extension. It is cleared only by a RESOLVED read (the stream actually
    retiring the copy), never by a forgiven run: otherwise a peer that
    republishes its marker between two runs would reset its own ceiling and
    the bound would not exist.

    WHAT THIS DOES NOT DO. No device read, no event wait, no ``.item()``. The
    whole decision is one directory ``stat`` per same-host peer plus float
    arithmetic, on a path that has already stalled for a minute. The wedge
    census that motivated the asymmetry found 195 of 239 events parked in this
    very bounded poll and recovering, while the only fatal-capable events were
    the unbounded host syncs -- so the guard's extension must not become one.
    """
    try:
        from sglang.srt.distributed.device_communicators import barlink_build_window

        builds = barlink_build_window.extension_for(
            transport._ctl_build_deferred_s,
            run * deadline_s,
            f"barlink-BAR1 group {transport.group} rank {transport.rank} "
            f"status read",
            table=transport._peer_table,
        )
    except Exception:  # noqa: BLE001 - a diagnostic must not hold a wedge open
        return False
    if builds is None:
        return False
    transport._ctl_build_deferred_s += run * deadline_s
    transport._ctl_stall_run = 0
    return True


class Bar1CollectiveAborted(Bar1KernelAborted):
    """The same fact, raised from a PRODUCTION path with its context (#431).

    ``Bar1KernelAborted`` is what the three bring-up proofs raise; they know
    which proof they are in, so "a kernel aborted" is a complete statement
    there. On the serving path it is not: by the time anybody notices, the
    only useful question is WHICH collective produced the corrupt buffer, and
    the answer has to be carried, because ``ctlStatus`` is one sticky bit
    with no history.

    The attributes are structured, not only formatted into the message, so a
    caller (a test, a supervisor, a future retry policy) can dispatch on them
    without parsing English:

    ``rank`` / ``world`` / ``group``
        who reports it. Rank first because the #431 evidence showed three
        ranks in three different host frames while their collective
        sequences were identical -- the rank is the only part of that picture
        that was ever load-bearing.
    ``op`` / ``nbytes`` / ``rounds``
        the last BAR1 collective this transport launched before the check.
        ``rounds`` is recomputed at raise time from the transport's own
        planner, never on the hot path.
    ``launches``
        how many collectives ran since the previous successful check. ``1``
        means the aborting collective is exactly the one named; more means
        the name is the most recent of several candidates and the abort is
        somewhere in that window.
    """

    def __init__(
        self,
        message: str,
        *,
        rank: int = -1,
        world: int = -1,
        group: str = "",
        op: str = "",
        nbytes: int = 0,
        rounds: int = 0,
        launches: int = 0,
    ):
        super().__init__(message)
        self.rank = rank
        self.world = world
        self.group = group
        self.op = op
        self.nbytes = nbytes
        self.rounds = rounds
        self.launches = launches


#: The values that count as "off" in this transport. Word-for-word the same
#: as ``parallel_state.graph_enable_set`` -- both decide the same
#: thing and must never read differently.
_OFF = ("0", "no", "off", "false", "")


def _is_on(value: Optional[str]) -> bool:
    """Whether an environment variable counts as set."""
    return value is not None and value not in _OFF


def _is_on_with_default(value: Optional[str], default: bool = True) -> bool:
    """Whether an environment variable counts as on, with a configurable
    default when unset. Used to mirror graph_enable_set() which defaults
    to True (release ON by default since #369)."""
    if value is None:
        return default
    return value not in _OFF


def graph_grid_default(env=None) -> bool:
    """May the cooperative launch fire WHILE a graph is being captured?

    **Derived, not independent.** Whether ``cudaLaunchCooperativeKernel`` can
    be captured on this rig is the same question that decides
    ``SGLANG_BARLINK_GRAPH_ENABLE`` -- and it is answered by the same gate
    (``benchmark/bar1_graph_check.py``, case ``grid``). A separate opt-in
    switch next to it would have meant: the gate passes, the release is set,
    and the kernel still falls back to ``1blk``. That is exactly what cost
    the entire BAR1 lead in the lever measurement for #293 once prefill was
    captured (1334.5 -> 1151.6 tok/s at eight sessions; the falsifier with
    ``SGLANG_BARLINK_BAR1_GRAPH_GRID=1`` recovered 1337.2, i.e. +16.1%).

    ``SGLANG_BARLINK_BAR1_GRAPH_GRID`` remains an override in BOTH directions:
    set, it allows the cooperative launch even without the release (this is
    how the gate case ``grid`` itself runs); set to ``0``, it restores the
    old restriction (this is how the gate case ``reservation`` runs). Only
    when it is NOT SET AT ALL does the release decide.
    """
    import os as _os

    if env is None:
        # Live os.environ, not a caller-supplied dict: re-run the retired-name
        # check on every call (not just once at import time), since a caller
        # may export a retired name at runtime, after this module was already
        # imported. See barlink_env_guard.check_retired_env_vars().
        from sglang.srt.distributed.device_communicators import (
            barlink_env_guard as _barlink_env_guard,
        )

        _barlink_env_guard.check_retired_env_vars()
        env = _os.environ
    explicit = env.get("SGLANG_BARLINK_BAR1_GRAPH_GRID")
    if explicit is not None:
        return explicit not in _OFF
    # Mirror graph_enable_set(): default to True when unset (release ON
    # since #369). Using _is_on_with_default instead of _is_on so the
    # unset case returns True, not False.
    return _is_on_with_default(env.get("SGLANG_BARLINK_GRAPH_ENABLE"), default=True)


# ===========================================================================
# CUDA bindings (ctypes) -- only what setup needs
# ===========================================================================

CU_MEM_ALLOCATION_TYPE_PINNED = 0x1
CU_MEM_LOCATION_TYPE_DEVICE = 0x1
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 0x1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3
CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD = 0x1
# Forces a PCIe mapping of the dma-buf instead of a possible shortcut
# mapping. Without this, the driver can hand back a mapping that is not
# reachable over PCIe for a given peer.
CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE = 0x1

CUDA_HOST_REGISTER_IO_MEMORY = 0x04
CUDA_MEMCPY_DEFAULT = 4
CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 0x1
CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CUmemAllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _CUmemAllocFlags),
    ]


class _CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _CUmemLocation), ("flags", ctypes.c_int)]


class _Cuda:
    """Thin wrapper around libcuda/libcudart. Loads lazily, never on import."""

    def __init__(self):
        try:
            self.drv = ctypes.CDLL("libcuda.so.1")
        except OSError as e:
            raise Bar1Unavailable(f"libcuda.so.1 could not be loaded: {e}") from e
        try:
            self.rt = ctypes.CDLL("libcudart.so")
        except OSError:
            try:
                self.rt = ctypes.CDLL("libcudart.so.12")
            except OSError as e:
                raise Bar1Unavailable(f"libcudart could not be loaded: {e}") from e

    def _d(self, name: str, *args) -> None:
        fn = getattr(self.drv, name, None)
        if fn is None:
            raise Bar1Unavailable(
                f"{name} is missing from libcuda -- the driver is too old for "
                f"the VMM/dma-buf path."
            )
        rc = fn(*args)
        if rc != 0:
            text = ctypes.c_char_p()
            if hasattr(self.drv, "cuGetErrorString"):
                self.drv.cuGetErrorString(ctypes.c_int(rc), ctypes.byref(text))
            raise Bar1Unavailable(
                f"{name} -> {rc} "
                f"({text.value.decode() if text.value else 'no text'})"
            )

    def _r(self, name: str, *args) -> None:
        fn = getattr(self.rt, name)
        rc = fn(*args)
        if rc != 0:
            raise Bar1Unavailable(f"{name} -> cudaError {rc}")

    # -- VMM ---------------------------------------------------------------

    def _prop(self, ordinal: int) -> _CUmemAllocationProp:
        p = _CUmemAllocationProp()
        ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
        p.type = CU_MEM_ALLOCATION_TYPE_PINNED
        p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        p.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        p.location.id = ordinal
        return p

    def granularitaet(self, ordinal: int) -> int:
        gran = ctypes.c_size_t(0)
        p = self._prop(ordinal)
        self._d("cuMemGetAllocationGranularity", ctypes.byref(gran),
                ctypes.byref(p), ctypes.c_int(CU_MEM_ALLOC_GRANULARITY_RECOMMENDED))
        return int(gran.value) or (2 << 20)

    def vmm_alloc(self, ordinal: int, size: int) -> tuple[int, int, int]:
        """``(dptr, handle, size)`` -- exportfaehige Geraeteallokation."""
        gran = self.granularitaet(ordinal)
        size = ((size + gran - 1) // gran) * gran
        handle = ctypes.c_ulonglong(0)
        p = self._prop(ordinal)
        self._d("cuMemCreate", ctypes.byref(handle), ctypes.c_size_t(size),
                ctypes.byref(p), ctypes.c_ulonglong(0))
        dptr = ctypes.c_ulonglong(0)
        self._d("cuMemAddressReserve", ctypes.byref(dptr),
                ctypes.c_size_t(size), ctypes.c_size_t(gran),
                ctypes.c_ulonglong(0), ctypes.c_ulonglong(0))
        self._d("cuMemMap", ctypes.c_ulonglong(dptr.value),
                ctypes.c_size_t(size), ctypes.c_size_t(0), handle,
                ctypes.c_ulonglong(0))
        desc = _CUmemAccessDesc()
        desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        desc.location.id = ordinal
        desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self._d("cuMemSetAccess", ctypes.c_ulonglong(dptr.value),
                ctypes.c_size_t(size), ctypes.byref(desc), ctypes.c_size_t(1))
        return int(dptr.value), int(handle.value), size

    def vmm_free(self, dptr: int, handle: int, size: int) -> None:
        for name, args in (
            ("cuMemUnmap", (ctypes.c_ulonglong(dptr), ctypes.c_size_t(size))),
            ("cuMemRelease", (ctypes.c_ulonglong(handle),)),
            ("cuMemAddressFree", (ctypes.c_ulonglong(dptr),
                                  ctypes.c_size_t(size))),
        ):
            try:
                getattr(self.drv, name)(*args)
            except Exception:      # teardown must never mask the real error
                pass

    def pci_bus(self, ordinal: int) -> int:
        """``CU_DEVICE_ATTRIBUTE_PCI_BUS_ID``. The RM ioctl path looks up the
        card by bus number, not by ordinal."""
        dev = ctypes.c_int(0)
        self._d("cuDeviceGet", ctypes.byref(dev), ctypes.c_int(ordinal))
        value = ctypes.c_int(0)
        self._d("cuDeviceGetAttribute", ctypes.byref(value),
                ctypes.c_int(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID), dev)
        return int(value.value)

    def export_shareable(self, handle: int) -> int:
        """``cuMemExportToShareableHandle`` -- the object fd for the ioctl path."""
        fd = ctypes.c_int(-1)
        self._d("cuMemExportToShareableHandle", ctypes.byref(fd),
                ctypes.c_ulonglong(handle),
                ctypes.c_int(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
                ctypes.c_ulonglong(0))
        return int(fd.value)

    def memset_d8(self, dptr: int, value: int, n: int) -> None:
        # ``cuMemsetD8_v2``, NOT ``cuMemsetD8``. In cuda.h, the short name is
        # a macro for the _v2 form; going through dlsym/ctypes instead gets
        # you the old ABI entry point with a 32-bit CUdeviceptr, and that one
        # answers 201 (invalid device context) on a current driver -- even
        # when a context is current (measured directly: cuCtxGetCurrent
        # returns a valid context, cuMemsetD8 -> 201, cuMemsetD8_v2 -> 0).
        # This applies to every _v2 function of the driver API.
        self._d("cuMemsetD8_v2", ctypes.c_ulonglong(dptr), ctypes.c_ubyte(value),
                ctypes.c_size_t(n))

    def dmabuf_fd(self, dptr: int, handle: int, size: int,
                  ordinal: int) -> tuple[int, list[int], str]:
        """``(dmabuf_fd, fds_to_hold, path)``.

        First the convenient path, ``cuMemGetHandleForAddressRange``. On
        GeForce, the driver reports ``CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED
        = 0`` and returns ``CUDA_ERROR_INVALID_VALUE`` (=1), even though the
        kernel module can do the export (``nv->dma_buf_supported = 1``,
        osinit.c:671). Then the ioctl path via the native extension --
        ported from ``sonden/dmabuf_p2p_probe.cpp::nvExportToDmabuf()``.

        ``fds_to_hold`` are ``/dev/nvidiactl`` and ``/dev/nvidia<N>`` from
        the ioctl path. The RM client that owns the imported memory object
        is attached to the former; if it is closed, RM releases the object
        and the dma-buf points into nothing. They are therefore handed back
        and closed at teardown instead of being leaked.
        """
        from sglang.srt.distributed.device_communicators import barlink_bar1_ext

        fd = ctypes.c_int(-1)
        fn = getattr(self.drv, "cuMemGetHandleForAddressRange", None)
        rc = -1
        if fn is not None:
            rc = fn(ctypes.byref(fd), ctypes.c_ulonglong(dptr),
                    ctypes.c_size_t(size),
                    ctypes.c_int(CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD),
                    ctypes.c_ulonglong(
                        CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE))
            if rc == 0:
                return int(fd.value), [], "cuMemGetHandleForAddressRange"

        ext = barlink_bar1_ext.load_dmabuf_ext()
        if ext is None:
            raise Bar1Unavailable(
                f"dma-buf export not possible. "
                f"cuMemGetHandleForAddressRange -> "
                f"{'missing from libcuda' if fn is None else rc}, and the "
                f"fallback path via NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD "
                f"+ NV_ESC_EXPORT_TO_DMABUF_FD is not available either: "
                f"{barlink_bar1_ext.dmabuf_reason()}"
            )
        objfd = self.export_shareable(handle)
        try:
            handed = ext.bar1_export_dmabuf(int(objfd), int(self.pci_bus(ordinal)),
                                         int(size))
        except Exception as e:
            raise Bar1Unavailable(
                f"NV_ESC_EXPORT_TO_DMABUF_FD failed: {e}"
            ) from e
        finally:
            # The object fd is no longer needed after the import -- RM now
            # holds the object through its own client.
            try:
                os.close(objfd)
            except OSError:
                pass
        return int(handed[0]), [int(handed[1]), int(handed[2])], "NV_ESC_EXPORT_TO_DMABUF_FD"

    # -- Runtime -----------------------------------------------------------

    def register_io(self, address: int, length: int) -> None:
        self._r("cudaHostRegister", ctypes.c_void_p(address),
                ctypes.c_size_t(length),
                ctypes.c_uint(CUDA_HOST_REGISTER_IO_MEMORY))

    def unregister(self, address: int) -> None:
        try:
            self.rt.cudaHostUnregister(ctypes.c_void_p(address))
        except Exception:
            pass

    def dev_ptr(self, host_address: int) -> int:
        p = ctypes.c_void_p(0)
        self._r("cudaHostGetDevicePointer", ctypes.byref(p),
                ctypes.c_void_p(host_address), ctypes.c_uint(0))
        return int(p.value or 0)

    def memcpy_async(self, dst: int, source: int, n: int, stream: int) -> None:
        self._r("cudaMemcpyAsync", ctypes.c_void_p(dst), ctypes.c_void_p(source),
                ctypes.c_size_t(n), ctypes.c_int(CUDA_MEMCPY_DEFAULT),
                ctypes.c_void_p(stream))

    def memcpy(self, dst: int, source: int, n: int) -> None:
        self._r("cudaMemcpy", ctypes.c_void_p(dst), ctypes.c_void_p(source),
                ctypes.c_size_t(n), ctypes.c_int(CUDA_MEMCPY_DEFAULT))


# ===========================================================================
# /dev/dmabuf_holder
# ===========================================================================

HOLDER_PATH = os.environ.get("SGLANG_BARLINK_BAR1_HOLDER", "/dev/dmabuf_holder")

#: Driver registry keys that switch on the widened peer-BAR1 guard, in
#: preference order. The second is the name the key carried before the
#: transport was renamed from HTCCL to barlink (#358); the driver patch still
#: reads it as a fallback, so a module loaded before the rename stays usable
#: and this probe has to accept either spelling. Index 0 is the one to name
#: in diagnostics.
PEER_BAR1_REGKEYS = ("BarlinkPeerBar1", "RMSmallBarP2PPeerBar1")

_HOLD_FMT = "=iIIBBBBIIQIIQQ"      # struct dmabuf_holder_hold
_HOLD_SIZE = struct.calcsize(_HOLD_FMT)
_REL_FMT = "=II"
_REL_SIZE = struct.calcsize(_REL_FMT)
_MAGIC = 0xDB
_F_BDF_VALID = 1 << 0


def _ioc(direction: int, magic: int, nr: int, size: int) -> int:
    return (direction << 30) | (size << 16) | (magic << 8) | nr


_IOC_WRITE, _IOC_READ = 1, 2
IOC_HOLD = _ioc(_IOC_READ | _IOC_WRITE, _MAGIC, 1, _HOLD_SIZE)
IOC_RELEASE = _ioc(_IOC_WRITE, _MAGIC, 2, _REL_SIZE)


def _ioc_arg(op: int) -> int:
    """``_IOWR`` sets the top bit; ``fcntl.ioctl`` wants it signed.

    ``_IOC_READ|_IOC_WRITE`` is 3, so bit 31 is set -- the number does not
    fit in an ``int`` and CPython rejects it depending on version. The
    two's-complement value is the same 32-bit value the kernel sees.
    """
    return op - (1 << 32) if op >= (1 << 31) else op


@dataclass
class SgEntry:
    dma_address: int
    length: int


class Holder:
    """Keeps foreign dma-bufs alive and returns their BAR1 addresses.

    Without an importer, the NVIDIA driver does not program the BAR1 pages
    at all -- an open fd alone is demonstrably not enough. This used to
    require an RDMA card (``ibv_reg_dmabuf_mr``); the GPL module
    ``dmabuf_holder`` takes over that role without a NIC and additionally
    returns the sg-table, which makes the pattern scan unnecessary.
    """

    def __init__(self, path: str = HOLDER_PATH):
        if not os.path.exists(path):
            raise Bar1Unavailable(
                f"{path} is missing. Without an importer, the driver does "
                f"not map the BAR1 pages of the exported buffer (proven: "
                f"the pattern scan found nothing across 65,536 probes, but "
                f"the hit appeared immediately once the attach happened). "
                f"Load the module from nvidia-smallbar-p2p/dmabuf_holder/. "
                f"NO silent fallback to an RDMA card -- that would be a "
                f"different mode of operation, not the same one."
            )
        try:
            self.fd = os.open(path, os.O_RDWR)
        except OSError as e:
            raise Bar1Unavailable(f"{path} could not be opened: {e}") from e
        self._handles: list[int] = []

    def hold(self, dmabuf_fd: int, bdf: str,
              max_entries: int = 1024) -> tuple[int, list[SgEntry], int]:
        """``dma_buf_attach`` + ``dma_buf_map_attachment`` as ``bdf``.

        ``bdf`` is the **source card** -- the device that will later write.
        There is deliberately no dummy device: ``nv_dma_buf_attach`` calls
        ``to_pci_dev(attachment->dev)`` without checking (nv-dmabuf.c:1033)
        and ``nv_dma_map_peer`` accesses ``->resource[]`` (nv-dma.c:749); a
        device without an embedded ``struct pci_dev`` would read outside
        the object there.

        Two passes if needed: the module writes the TRUE number of
        sg-entries into ``arg.nents``, but copies out at most
        ``max_entries`` of them (dmabuf_holder.c:216). A truncated table
        looks like a short contiguous stretch -- setup would then fail on a
        window limit that does not actually exist. Whoever reports more
        entries than fit gets a second hold with a matching buffer; the
        first is kept alive in the meantime so the BAR1 mapping never drops
        in between.
        """
        handle_, entries, total_len, nents = self._hold_once(
            dmabuf_fd, bdf, max_entries
        )
        if nents > max_entries:
            old = handle_
            try:
                handle_, entries, total_len, nents2 = self._hold_once(
                    dmabuf_fd, bdf, nents
                )
            finally:
                self.release(old)
            if nents2 > nents:
                self.release(handle_)
                raise Bar1Unavailable(
                    f"The holder reports {nents2} sg-entries, {nents} were "
                    f"requested -- the table grows between two hold calls. "
                    f"Without a complete table, the contiguous length "
                    f"cannot be determined."
                )
        if not entries:
            raise Bar1Unavailable(
                "The holder reports 0 sg-entries -- the mapping is empty. "
                "Without an sg-address, the BAR1 offset cannot be "
                "determined; the pattern scan would be the fallback, but it "
                "does not belong in a transport."
            )
        return handle_, entries, total_len

    def _hold_once(self, dmabuf_fd: int, bdf: str,
                      max_entries: int) -> tuple[int, list[SgEntry], int, int]:
        dom, bus, slot, func = _split_bdf(bdf)
        buffer = ctypes.create_string_buffer(16 * max_entries)
        arg = bytearray(struct.pack(
            _HOLD_FMT,
            dmabuf_fd, _F_BDF_VALID, dom, bus, slot, func, 0,
            max_entries, 0, ctypes.addressof(buffer),
            0, 0, 0, 0,
        ))
        try:
            fcntl.ioctl(self.fd, _ioc_arg(IOC_HOLD), arg, True)
        except OSError as e:
            raise Bar1Unavailable(
                f"DMABUF_HOLDER_IOC_HOLD failed ({e}). Without a held "
                f"attachment there is no BAR1 mapping and thus no direct "
                f"path."
            ) from e
        values = struct.unpack(_HOLD_FMT, bytes(arg))
        handle_, nents, _dmabuf_size, total_len = values[10], values[11], values[12], values[13]
        self._handles.append(handle_)
        entries = []
        valid = min(nents, max_entries)
        raw = bytes(buffer.raw[: 16 * valid])
        for i in range(valid):
            addr, length = struct.unpack_from("=QQ", raw, 16 * i)
            entries.append(SgEntry(addr, length))
        return handle_, entries, int(total_len), int(nents)

    def release(self, handle_: int) -> None:
        # Remove from the list BEFORE running the ioctl: otherwise `close()`
        # releases the same handle a second time and logs a warning about an
        # error that does not exist.
        if handle_ in self._handles:
            self._handles.remove(handle_)
        try:
            fcntl.ioctl(self.fd, _ioc_arg(IOC_RELEASE),
                        struct.pack(_REL_FMT, handle_, 0))
        except OSError as e:
            logger.warning("barlink-BAR1: RELEASE(%d) failed: %s", handle_, e)

    def close(self) -> None:
        for h in list(self._handles):
            self.release(h)
        self._handles.clear()
        try:
            os.close(self.fd)
        except OSError:
            pass


def _split_bdf(bdf: str) -> tuple[int, int, int, int]:
    s = bdf.strip().lower()
    if s.count(":") == 1:
        s = "0000:" + s
    dom, rest = s.split(":", 1)
    bus, rest = rest.split(":", 1)
    slot, func = rest.split(".", 1)
    return int(dom, 16), int(bus, 16), int(slot, 16), int(func, 16)


# ===========================================================================
# sysfs: BAR1 location and size
# ===========================================================================


@dataclass(frozen=True)
class Bar1Window:
    bdf: str
    base: int
    size: int          # gross, per sysfs

    @property
    def end(self) -> int:
        return self.base + self.size


def bar1_window(bdf: str) -> Bar1Window:
    """BAR1 from ``/sys/bus/pci/devices/<bdf>/resource``, line 1.

    NOTE: this is the **gross size** of the aperture. How much of it is
    actually available for peer mappings is unmeasured -- RM reserves part
    of it for itself. ``check_window_requirement`` therefore checks against
    what could actually be exported, not against this number.
    """
    path = f"/sys/bus/pci/devices/{bdf}/resource"
    try:
        with open(path) as f:
            lines = f.read().strip().split("\n")
    except OSError as e:
        raise Bar1Unavailable(f"{path} could not be read: {e}") from e
    if len(lines) < 2:
        raise Bar1Unavailable(f"{path}: no BAR1 line")
    start_s, end_s, _flags = lines[1].split()
    start, end = int(start_s, 16), int(end_s, 16)
    if end <= start:
        raise Bar1Unavailable(f"{bdf}: BAR1 is empty ({start_s}..{end_s})")
    return Bar1Window(bdf=bdf, base=start, size=end - start + 1)


PAGE_SIZE = 4096

#: Largest group for which the kernel arguments have room.
MAX_RANGE = 8


def window_requirement(algorithm: str, nbytes: int, world: int) -> int:
    """How much BAR1 the decomposition needs mapped at the same time.

    **Counted against the ported kernels, not estimated.** Both need
    ``2(R-1)`` slots of ``ceil(N/R)`` each, for different reasons:

    * **Chunked mesh**: ``R-1`` slots for the reduce-scatter and another
      ``R-1`` for the all-gather. The two sets must **not** be shared:
      there is no ordering between "I'm reading my RS slots" and "the other
      side is writing its AG chunk". A shared set would need a third
      barrier, and that costs more than the slots do.
    * **Ring**: one per step, and there are ``2(R-1)`` steps. Alternating
      between two slots would only work if the sender knew that the
      receiver had already read the slot from two steps ago -- but it only
      observes its PREDECESSOR, never its successor.

    This module's earlier numbers (mesh ``R-1``, ring ``2``, each doubled
    for double-buffering) happened to give the same value at ``R=3`` and
    that is the only reason the discrepancy was never noticed. From ``R=4``
    on they diverge, and on the low side -- a window that does not suffice
    would have looked like one that does.

    * **Star**: ``R-1`` full buffers on the hub -- the reason it is the
      first to hit the limit on 256-MiB cards. Not ported, see the module
      docstring; the branch stays here so the planner computes the same
      value.
    """
    if world < 2:
        return 0
    share = -(-nbytes // world)          # round up
    # ``mesh_pipe`` is grouped with mesh and ring here because this function
    # asks about the requirement of ONE payload, and the pipe moves the same
    # 2(R-1)*ceil(N/R) bytes for that. What it actually OCCUPIES in the
    # window is something different: ``2 T (R-1)`` slots of one chunk-piece
    # size each, computed in ``barlink_bar1_pipe_ext.pipe_window_requirement``
    # and additionally checked in ``handles``. That number depends on
    # ``pipe_chunk_bytes``, not on the payload.
    if algorithm in ("mesh", "mesh_pipe", "ring", "hierarchical"):
        return 2 * (world - 1) * share
    if algorithm == "star":
        return 2 * (world - 1) * nbytes
    raise ValueError(f"unknown algorithm {algorithm!r}")


def geometry(world: int, max_bytes: int, with_a2a: bool = True,
              with_pipe: bool = False, result_ring: int = 0,
              pipe_range: int = 0) -> dict:
    """The memory layout of ONE receive region, for arbitrary R.

    It carries all schemes **simultaneously**, so a plan can switch per size
    and per operation without anything being re-mapped:

    ===========  =================  ========================================
    Offset       Content            Size
    ===========  =================  ========================================
    ``0``        mesh RS slots      ``(R-1) * chunk_max``
    ...          mesh AG slots      ``(R-1) * chunk_max``
    ``off_ring`` ring slots         ``2(R-1) * chunk_max``
    ``off_a2a``  a2a slots          ``2(R-1) * chunk_max``
    ``off_pipe`` mesh_pipe slots    ``pipe_range`` (absolute)
    ===========  =================  ========================================

    ``chunk_max`` is rounded up to a page -- a slot that begins on a page
    boundary can never share a page with its neighboring slot, so an
    overlong write hits its own page rather than someone else's payload.

    **Why a2a needs 2(R-1) slots and not (R-1).** One slot per sender would
    be enough if the sender knew the receiver had already read the previous
    content. But the flag only says "written". The alternatives are a
    second barrier (half the latency at MoE sizes) or two halves, between
    which the round number alternates. It is two halves; the rationale for
    why two suffice is in the kernel, in ``barlink_bar1_ext.py``.

    **What this costs the all_reduce path.** The region grows from
    ``4(R-1)`` to ``6(R-1)`` slots, so the largest all_reduce payload for a
    given window drops to two-thirds. No measured number changes because of
    this -- only the ceiling above which ``handles`` says False. Anyone who
    wants it back sets ``SGLANG_BARLINK_BAR1_A2A=0``; then ``with_a2a`` is
    False and the layout is byte-for-byte the old one.

    **Why mesh_pipe gets its OWN region and not mesh's.** The regions of
    the different schemes must be pairwise disjoint, and not only within a
    single call. When rank A finishes its round ``n``, rank B may still be
    reading that round's all-gather slot -- before A finishes, it only
    waits on B's flag, not on B's read. This does not show up with
    ``mesh``, because A's next write goes into the RS half while B reads
    from the AG half. A ``mesh_pipe`` that used the whole mesh region would
    immediately hit the AG half. A dedicated region makes the question
    moot.

    The region is only created when ``with_pipe`` is set
    (``SGLANG_BARLINK_BAR1_PIPE=1``); without it, the layout is byte-for-byte
    the measured one.

    **Why the pipe region arrives as an absolute byte count.** It used to
    be a full slot set like mesh, ring, and a2a, i.e.
    ``2(R-1) * chunk_max``. That was too generous: a pipe slot carries a
    piece of a CHUNK, and how big a chunk becomes is decided by
    ``pipe_chunk_bytes``, not by the largest payload. The full slot set
    pushed the all_reduce slot at R=3 down from 8188 to 6140 KiB and, with
    it, the tipping point from 2456 to 1842 tokens -- below the working
    point of 2048; that is the 7.5% loss the lever measurement for #293
    attributed to the pipe arm (there still attributed to the result ring,
    which was wrong: that arm ran ``PIPE_DIRECT=0`` and thus
    ``result_ring = 0``).

    The region's requirement depends on ``pipe_chunk_bytes``, ``T``, and
    ``R`` -- on nothing that follows from ``max_bytes``. That makes it a
    constant in the fixed-point computation of :func:`max_payload` rather
    than another denominator term. ``pipe_range = 0`` keeps the old
    sizing, so a geometry without this argument stays byte-for-byte the
    old one.
    """
    if world < 2:
        raise ValueError("world < 2")
    n4_max = max_bytes // 16
    chunk4 = -(-n4_max // world)
    chunk_max = ((chunk4 * 16 + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
    slots = 2 * (world - 1)
    off_mesh = 0
    off_ring = slots * chunk_max
    off_a2a = 2 * slots * chunk_max
    from sglang.srt.distributed.device_communicators.barlink_bar1_pipe_ext import (
        result_stride_bytes,
    )

    sets = 2 + (1 if with_a2a else 0)
    off_pipe = sets * slots * chunk_max
    # The pipe region: the absolute number passed in, otherwise the old full
    # slot set. Rounded up to a page, so the result ring behind it again
    # begins on a page boundary.
    pipe_range = int(pipe_range) if with_pipe else 0
    if with_pipe and pipe_range <= 0:
        pipe_range = slots * chunk_max
    pipe_range = ((pipe_range + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE if pipe_range else 0
    off_result = off_pipe + pipe_range
    ring = int(result_ring) if with_pipe else 0
    result_stride = result_stride_bytes(max_bytes) if ring > 0 else 0
    region = off_result + ring * result_stride + PAGE_SIZE
    return {
        "chunk_max": chunk_max,
        "off_mesh": off_mesh,
        "off_ring": off_ring,
        # -1 explicitly means "does not exist", not "is at 0" -- an offset
        # of 0 would be the mesh region.
        "off_a2a": off_a2a if with_a2a else -1,
        "a2a_slot": chunk_max if with_a2a else 0,
        "off_pipe": off_pipe if with_pipe else -1,
        "pipe_range": pipe_range,
        "off_result": off_result if ring > 0 else -1,
        "result_stride": result_stride,
        "result_ring": ring,
        "region_bytes": region,
        "max_bytes": max_bytes,
        "with_a2a": bool(with_a2a),
        "with_pipe": bool(with_pipe),
    }


def _flags_base(world: int, with_a2a: bool, with_pipe: bool) -> int:
    """Everything in the flag region BEFORE the #622 acknowledgment banks.

    The one arithmetic source for that size. ``flags_requirement`` adds the
    two ack banks on top, ``ackbase_mesh`` returns exactly this value as the
    offset of the first of them -- a second copy of the formula would be
    precisely the place where the allocation and the offset drift apart.
    """
    from sglang.srt.distributed.device_communicators.barlink_bar1_pipe_ext import (
        pipe_flags_extra,
    )

    base = (2 + 2 * (world - 1) + (1 if with_a2a else 0)) * world * 256
    return base + (pipe_flags_extra(world) if with_pipe else 0)


def flags_requirement(world: int, with_a2a: bool = True,
                   with_pipe: bool = False) -> int:
    """``(2 + 2(R-1) [+ 1]) * R * 256`` bytes, plus ``5 R * 256`` for the pipe,
    plus ``2 R * 256`` for the two acknowledgment banks (#622).

    One 256-byte line per (topology, step, sender): no false sharing between
    senders, none between steps, none between topologies. Mesh has 2 steps,
    ring ``2(R-1)``, a2a exactly **one**. At R=8 that is 34 KiB, well below
    an allocation granularity.

    ``mesh_pipe`` appends five lines per rank at the end (``tailRS``,
    ``tailAG``, ``headRS``, ``headAG``, ``resultReady``) -- **independent of
    K and T**, because it is a sliding window with one counter per
    connection, not a flag per chunk. Appended at the end so every existing
    line offset stays byte-for-byte the same.

    #622 appends TWO more banks of one line per rank, behind the pipe rows
    and thus behind everything that existed before: the mesh consumption
    acknowledgment and the a2a one. Same discipline -- appended, never
    inserted, so no line that a peer already addresses can move. They are
    allocated unconditionally (independent of ``with_a2a``/``with_pipe``),
    because the entry barrier they carry belongs to the transport, not to a
    topology that may or may not be enabled; their OFFSET does depend on the
    two switches, which is why ``ackbase_mesh`` takes the same arguments.
    """
    return _flags_base(world, with_a2a, with_pipe) + 2 * world * 256


def ackbase_mesh(world: int, with_a2a: bool = True,
                 with_pipe: bool = False) -> int:
    """Offset of the MESH acknowledgment bank -- one line per rank (#622).

    Behind every pre-#622 line, i.e. exactly the flag-region size as it was
    before the acknowledgment banks existed. Rank ``r`` publishes into line
    ``r`` of every PEER's bank and reads line ``z`` of its OWN bank to learn
    which round peer ``z`` has finished consuming.

    Depends on ``with_a2a`` and ``with_pipe`` exactly as the region size
    does; both are forwarded to ``_flags_base``, the single place the
    pre-ack arithmetic lives.
    """
    return _flags_base(world, with_a2a, with_pipe)


def ackbase_a2a(world: int, with_a2a: bool = True,
                with_pipe: bool = False) -> int:
    """Offset of the A2A acknowledgment bank -- one line per rank (#622).

    Directly behind the mesh bank. Two separate banks and not one shared
    one: mesh and a2a advance independently (the round counter is global,
    the CONSUMPTION watermark is per topology), and a shared line would let
    an a2a acknowledgment satisfy a mesh entry wait.
    """
    return ackbase_mesh(world, with_a2a, with_pipe) + world * 256


def fbase_a2a(world: int) -> int:
    """Offset of the a2a flag lines within the flag region.

    Behind mesh and ring, so the two measured topologies stay byte-for-byte
    where they were. The same computation does NOT appear a second time in
    the kernel -- it is passed in as an argument, because a second version
    would be exactly the place where sender and receiver end up pointing at
    different lines.
    """
    return (2 + 2 * (world - 1)) * world * 256


def ag_plan(lengths, slot: int) -> list:
    """The round decomposition of an ``all_gather``. Pure arithmetic.

    ``lengths[i]`` is rank ``i``'s shard in **bytes**; the result is their
    concatenation, i.e. ``sum(lengths)`` bytes, with rank ``i`` at offset
    ``sum(lengths[:i])``.

    Delivered is, per round, a list of ``(send_offset, length,
    receive_offset)`` per rank -- all in bytes, all absolute, nothing left
    to be guessed as a prefix sum:

    * ``send_offset`` points into the CALLER'S OWN shard (the same slice for
      every destination -- that is exactly what distinguishes all_gather
      from all_to_all),
    * ``receive_offset`` points into the result, i.e. ``base[i] + k*slot``.

    **Why rounds at all.** A shard can be larger than a slot. The failure
    case from the handoff is exactly that: 10,600,448 bytes of all_gather
    against an a2a slot of just under 8 MiB with a 96-MiB window. Instead of
    withdrawing via ``handles`` -- which aborts the run during a CUDA graph
    capture, because there is no fallback path -- the shard runs in
    ``ceil(max(lengths)/slot)`` rounds.

    **Why this survives a capture.** The round count depends only on
    ``lengths`` and ``slot``. Both are group-wide identical and constant
    for a captured shape, so the number of kernel launches is baked in and
    the same on every replay -- the same argument that lets
    ``barlink_device.all_reduce`` capture its slot loop. No host code here
    decides anything per round that could change between capture and
    replay. That is the difference from the pipe's direct mode, whose
    host-side ring index fails on exactly this point (see
    ``_result_slot``).

    **Rank-uniform.** Every rank computes from the SAME ``lengths`` vector,
    so all of them end up with the same number of rounds. If a rank counted
    differently, that would not be an error but a hang: the others would
    wait in the barrier of a round it no longer runs.

    **Unequal shards** are arithmetic here, not a rewrite. Today's seam
    (``BarlinkCommunicator.all_gather``) is uniform -- its result is
    ``(R,) + shape``, which CANNOT be uneven, and the uneven form is called
    ``all_gatherv`` in sglang and explicitly not covered under barlink. This
    function nonetheless takes a vector: under uneven TP, unequal shards are
    the normal case, and the place where a uniform distribution is ASSUMED
    is the place where a later ``all_gatherv`` silently gets wrong offsets.
    A rank whose shard ends earlier gets length 0 in the remaining rounds --
    it rides along in the barrier without moving any bytes.
    """
    lengths = [int(x) for x in lengths]
    if not lengths:
        return []
    if slot <= 0:
        raise ValueError(f"slot size {slot} is not positive")
    if any(n < 0 for n in lengths):
        raise ValueError(f"negative shard length in {lengths}")
    base, acc = [], 0
    for n in lengths:
        base.append(acc)
        acc += n
    rounds = max(1, -(-max(lengths) // slot))
    plan = []
    for k in range(rounds):
        row = []
        for i, n in enumerate(lengths):
            a = min(k * slot, n)
            b = min((k + 1) * slot, n)
            row.append((a, b - a, base[i] + a))
        plan.append(row)
    return plan


def ar_plan(nbytes: int, chunk_max: int, world: int) -> list:
    """The round decomposition of an ``all_reduce``. Pure arithmetic.

    Delivered is, per round, an ``(offset, length)`` in **bytes**. Each
    round is a complete all_reduce over a slice of the buffer -- the same
    kernel, the same decomposition into ``world`` shards, just on fewer
    bytes.

    **Why this exists.** The kernel decomposes a payload into ``world``
    equally sized shards (reduce-scatter, then all-gather), and the shard
    has to fit into a slot. Up to now, "does not fit" simply meant
    ``handles() == False``, and the payload fell back to the base
    transport. The tipping point in the standard run is 2456 tokens per
    batch (shard 6.67 MiB against a 7.996 MiB slot, 20% headroom); with the
    ``chunked_prefill_size`` values of 4096 or 8192 common in sglang, the
    direct path would have silently disappeared during prefill. The same
    answer as with all_gather and broadcast: decompose into rounds instead
    of declining.

    **Evenly distributed, not filled to the brim.** The obvious approach
    would be to fill every round up to ``chunk_max*world`` and put the
    remainder in the last one. That produces a tail that can become
    arbitrarily small -- and the extension insists on ``n4 >= R`` (one
    128-bit packet per rank, ``TORCH_CHECK`` in the host). A leftover round
    of 16 bytes across three ranks would not be a slow case but an abort.
    Evenly distributed, the smallest round can lie at most ONE packet below
    the largest.

    **Rank-uniform and capture-safe.** The round count depends solely on
    ``nbytes``, ``chunk_max``, and ``world``. All three are group-wide
    identical and constant for a captured shape, so the number of kernel
    launches is baked in -- the same argument as for :func:`ag_plan` and
    :func:`bc_plan`.
    """
    nbytes = int(nbytes)
    if world < 2:
        raise ValueError(f"world {world} is smaller than 2")
    if chunk_max < 16:
        raise ValueError(f"chunk_max {chunk_max} cannot carry a packet")
    if nbytes < 0:
        raise ValueError(f"negative payload {nbytes}")
    if nbytes % 16:
        raise ValueError(f"payload {nbytes} is not a multiple of 16")
    if nbytes == 0:
        return []
    packets = nbytes // 16
    # Packets per rank and round -- the size the slot depends on.
    per_rank_max = chunk_max // 16
    max_packets = per_rank_max * world
    rounds = -(-packets // max_packets)
    base, rest = divmod(packets, rounds)
    plan = []
    offset = 0
    for k in range(rounds):
        p = base + (1 if k < rest else 0)
        length = p * 16
        plan.append((offset, length))
        offset += length
    return plan


def a2a_rounds(largest_block: int, slot: int) -> int:
    """Round count for an ``all_to_all``, from the LARGEST block.

    ``largest_block`` is the maximum over all ``R*R`` blocks, not over the
    caller's own row -- the seam computes it group-wide before asking
    (``BarlinkCommunicator.all_to_all_single``). That is exactly why the round
    count can be derived from it and still be the same on every rank:
    computed from the caller's own row it would not be, and a rank running
    one round fewer would leave the others waiting in the barrier.

    Each round moves at most one slot's worth out of every block, in one
    piece and at a constant offset ``k*slot``. Blocks that finish earlier
    carry length 0 in the remaining rounds -- they ride along in the
    barrier without moving any bytes. Same pattern as in :func:`ag_plan`.
    """
    if slot <= 0:
        raise ValueError(f"slot size {slot} is not positive")
    if largest_block < 0:
        raise ValueError(f"negative block {largest_block}")
    return max(1, -(-int(largest_block) // int(slot)))


def bc_plan(nbytes: int, slot: int) -> list:
    """The round decomposition of a ``broadcast``. Pure arithmetic.

    Delivered is, per round, an ``(offset, length)`` in **bytes**. Unlike
    :func:`ag_plan`, there is no per-rank vector: in a broadcast, exactly
    ONE rank moves bytes, and every other rank gets the same slice at the
    same offset. Send offset and receive offset therefore coincide -- the
    slice sits at the same place in the source as in the result.

    **Why rounds at all.** The same answer as for all_gather: the payload
    can be larger than a slot, and a decline would not be a slower path
    under a CUDA graph capture but the abort of the run. Hence
    ``ceil(nbytes/slot)`` rounds instead of a refusal.

    **Rank-uniform even though only one rank sends.** This is the one place
    where a broadcast looks different from an all_gather and yet must not
    be. The round count depends solely on ``nbytes`` and ``slot``; both are
    group-wide identical, so every rank counts the same number of rounds --
    including the ones that send zero bytes in each of them. They ride
    along in the barrier. If a non-source rank counted fewer rounds (say,
    "I'm sending nothing, so there's nothing to do"), that would not be an
    error but a hang: the source would be left waiting in the barrier of a
    round nobody else runs anymore.

    **Why this survives a capture.** The same rationale as for
    :func:`ag_plan`: the number of kernel launches is baked in, and no host
    code decides anything per round that could change between capture and
    replay. ``src`` is fixed at capture time and travels along as a kernel
    argument.
    """
    nbytes = int(nbytes)
    if slot <= 0:
        raise ValueError(f"slot size {slot} is not positive")
    if nbytes < 0:
        raise ValueError(f"negative payload {nbytes}")
    if nbytes == 0:
        return []
    rounds = -(-nbytes // slot)
    return [
        (k * slot, min((k + 1) * slot, nbytes) - k * slot)
        for k in range(rounds)
    ]


def pipe_on_group_verdict(gathered) -> bool:
    """The group's answer for ``pipe_on``: on only if EVERY rank has it (#728).

    Split out as a pure function so the reduction can be falsified without a
    process group -- the gather that feeds it needs one, the decision does not.

    AND, not OR or majority, and the direction matters. ``pipe_on`` feeds the
    payload ceiling and the slot offsets, so the group must agree on the layout
    the WEAKEST rank can build. A rank that could not compile the pipelined
    extension cannot run it however many peers could.

    An empty or all-``None`` carrier answers False: "nobody reported that they
    can" is not "everybody can", and defaulting the other way would enable a
    path on the strength of a failed exchange.
    """
    values = [v for v in gathered if v is not None]
    if not values:
        return False
    return all(bool(v) for v in values)


def max_payload(world: int, region_bytes: int, with_a2a: bool = True,
                 with_pipe: bool = False, result_ring: int = 0,
                 pipe_range: int = 0) -> int:
    """Largest payload whose slots fit into a region of this size.

    Inverse of :func:`geometry`. Deliberately rounded conservatively and
    then checked by re-computing forward -- an inverse that is off by one
    page would otherwise only surface on the hot path. This forward check
    is exactly why there does not need to be a second version of the
    factor computation here: ``geometry`` itself has the final say.
    """
    if world < 2 or region_bytes <= PAGE_SIZE:
        return 0
    # The pipe region is an ABSOLUTE number as soon as it is passed in -- it
    # depends on `pipe_chunk_bytes`, T, and R, not on `chunk_max`. It is
    # therefore subtracted and does not appear in the denominator. Without
    # it, the old sizing remains (a full slot set, i.e. 2(R-1) in the
    # denominator).
    absolute = int(pipe_range) if (with_pipe and pipe_range > 0) else 0
    if absolute:
        absolute = ((absolute + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
    # 2 sets for mesh, 2 for ring, 2 each for a2a and the pipe -- so 4 as
    # the base, not 2. Spelled out instead of "(6 if a2a else 4)", so the
    # fourth term does not disappear back into a single number.
    slots = (4 + (2 if with_a2a else 0)
                + (2 if (with_pipe and not absolute) else 0)) * (world - 1)
    ring = int(result_ring) if with_pipe else 0
    # The result ring costs ``L * roundup(N, PAGE_SIZE)``, and ``N`` is
    # ``chunk_max * R``. In units of chunk_max that is ``L * R`` additional
    # units on top of ``slots`` -- which is why the ring appears here IN
    # THE DENOMINATOR and not as a subtraction. A subtraction would have
    # placed the initial value far enough off that the forward check below
    # would have had to search downward in 32-byte steps.
    denominator = slots + ring * world
    rest = region_bytes - PAGE_SIZE - absolute
    if rest <= 0:
        return 0
    chunk_max = (rest // denominator // PAGE_SIZE) * PAGE_SIZE
    if chunk_max <= 0:
        return 0
    n = (chunk_max // 16) * world * 16
    while n > 0 and geometry(world, n, with_a2a, with_pipe, ring,
                              absolute)["region_bytes"] > region_bytes:
        n -= world * 16
    return n


# ===========================================================================
# fd exchange via SCM_RIGHTS
# ===========================================================================


def _exchange_fds(cpu_group, rank: int, world: int,
                 own_fds: list[int]) -> list[list[int]]:
    """Every rank hands its dma-buf fds to all the others.

    There are TWO per rank: the payload region and the flag region. They
    live in separate VMM allocations, because that is how the probe
    measured them -- the same layout, the same numbers. Both travel in ONE
    ``SCM_RIGHTS`` message, so the exchange does not run twice and a
    half-succeeded pass does not leave a rank with half a table.

    ``SCM_RIGHTS`` over AF_UNIX, because an fd is process-local: gloo can
    transfer a number, but not an access right. The sequence is
    deliberately round-based and serial -- it runs exactly once at startup,
    and a stuck bootstrap costs more than a pair of milliseconds of startup
    time.
    """
    import socket

    import torch.distributed as dist

    carrier = [None]
    if rank == 0:
        carrier = [tempfile.mkdtemp(prefix="barlink-bar1-")]
    # torch runs the object collectives inline and ignores async_op, so there
    # is no Work to bound. A one-shot check before the call names a peer that
    # is already gone instead of entering the 7200 s gloo wait for it.
    check_peers("bar1 fd exchange: broadcast of the socket directory")
    dist.broadcast_object_list(
        carrier, src=dist.get_global_rank(cpu_group, 0), group=cpu_group
    )
    directory = str(carrier[0])
    path = os.path.join(directory, f"r{rank}.sock")

    count = len(own_fds)
    fds: list[list[int]] = [[] for _ in range(world)]
    fds[rank] = list(own_fds)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        if os.path.exists(path):
            os.unlink(path)
        listener.bind(path)
        listener.listen(world)
        bounded_barrier(cpu_group, "bar1 fd exchange: sockets bound")

        for owner in range(world):
            if owner == rank:
                for _ in range(world - 1):
                    conn, _ = listener.accept()
                    with conn:
                        socket.send_fds(conn, [b"x"], list(own_fds))
            else:
                dst = os.path.join(directory, f"r{owner}.sock")
                conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                last_error: Optional[Exception] = None
                for _ in range(200):        # the peer may still be binding
                    try:
                        conn.connect(dst)
                        last_error = None
                        break
                    except OSError as e:
                        last_error = e
                        time.sleep(0.01)
                if last_error is not None:
                    # The 2 s cap above already bounds this loop. What it
                    # cannot do is say WHY: a peer that died before binding
                    # its socket looks exactly like one that is merely slow.
                    check_peers(f"bar1 fd exchange: connect to rank {owner}")
                    raise Bar1Unavailable(
                        f"fd exchange: {dst} unreachable ({last_error})"
                    )
                with conn:
                    _data, received, _fl, _addr = socket.recv_fds(
                        conn, 1, count
                    )
                if len(received) != count:
                    raise Bar1Unavailable(
                        f"fd exchange: rank {owner} sent {len(received)} "
                        f"fds instead of {count}"
                    )
                fds[owner] = list(received)
            bounded_barrier(
                cpu_group, f"bar1 fd exchange: round {owner} complete"
            )
    finally:
        listener.close()
        try:
            os.unlink(path)
        except OSError:
            pass
    return fds


# ===========================================================================
# The transport
# ===========================================================================


@dataclass
class Mapping:
    """A mapped and registered foreign BAR1 region."""

    bar1_base: int
    bar1_offset: int          # region's offset within BAR1
    length: int                # ACTUALLY mapped, contiguous length
    mmap_obj: object           # held so the mapping stays alive
    reg_address: int           # address under which REGISTRATION happened
    host_address: int          # user-space address of the region (reg + lead-in)
    dev_ptr: int               # THIS card's device pointer to the foreign BAR
    holder_handle: int


@dataclass
class PeerTarget:
    """What setup established for this peer -- immutable from then on.

    Two regions per peer, in separate VMM allocations and thus exported and
    mapped separately: the payload slots and the flag lines. Exactly the
    arrangement the probe measured.
    """

    rank: int
    bdf: str
    payload: Mapping
    flag: Mapping
    byte_proof: bool = False

    # The two names under which the point-to-point path (put/pair) knows
    # the payload region. Kept so the measurement probe stays unchanged.
    @property
    def dev_ptr(self) -> int:
        return self.payload.dev_ptr

    @property
    def length(self) -> int:
        return self.payload.length


class BarlinkBar1Transport:
    """BAR1 direct transport.

    Implements the transport seam from ``barlink.py`` (lines 67-80):
    ``handles(op, nbytes) -> bool`` plus ``barlink_<op>(comm, ...)`` for each
    operation offered.

    What it offers, and which parts of it are measured:

    * ``barlink_all_reduce`` over the ported kernels ``mesh`` and ``ring``.
      Fully measured in the probe (float32, three ranks, rig 1); the table
      is in the module docstring.
    * ``put(dst, source_ptr, nbytes, offset)`` -- a single write into the
      destination card's BAR.
    * ``pair``/``pair_receive`` -- the measurement probe that
      ``barlink_matrix.BarlinkMatrixPlanner`` needs for real edge capacities
      (instead of the self-load estimate).
    * ``byte_proof_all()`` -- the byte-level proof for each directed pair.
      If it fails, the edge is struck, no matter what the driver reports.
      On this rig, the driver reported peer access for one pair and
      delivered 4096 of 1,048,576 bytes.

    ``handles`` returns ``True`` **only** if the peer pointers are in place,
    every byte-level proof held, the size fits into the region, and the
    window requirement fits into the ACTUALLY mapped length. Otherwise
    ``False`` -- no exception, no emergency path.
    """

    #: all_reduce (measured), all_to_all (own kernel, unmeasured), all_gather
    #: (on the same kernel, unmeasured -- see :meth:`barlink_all_gather`), and
    #: broadcast (the same kernel again, see :meth:`barlink_broadcast`).
    #:
    #: ``reduce_scatter`` is still missing, and for a reason, not by
    #: accident: it needs a REDUCTION. The a2a kernel moves bytes and knows
    #: no data type; it therefore carries all_gather and broadcast for free
    #: and reduce_scatter not at all. The RS phase of the two all_reduce
    #: kernels could do it, but only as its own entry point with its own
    #: slot set -- not as a side effect.
    #:
    #: ``broadcast`` was absent for a second reason for a long time: it is
    #: IN-PLACE in sglang (``broadcast(tensor, src)`` returns the same
    #: tensor), and the extension rejects ``in is out``. The price for that
    #: is ONE intermediate buffer plus a local copy -- no new kernel. At the
    #: sizes on this path (128 bytes in the acceptance case), that is
    #: nothing, and the alternative was aborting the graph capture.
    #:
    #: For reduce_scatter, the loud guard in ``barlink._select`` remains
    #: responsible. It names the operation, and because this set here is the
    #: single source of truth about what is covered, the message cannot go
    #: stale.
    #:
    #: Both spellings of all_to_all appear here because the seam in
    #: barlink.py asks for the operation under the name ``all_to_all``, while
    #: the only real caller in sglang (GroupCoordinator, equal split) is
    #: named ``all_to_all_single``. Two names, one path -- better than a
    #: rename at the seam that gets overlooked when reading.
    BARLINK_OPS: frozenset = frozenset(
        {"all_reduce", "all_gather", "all_to_all", "all_to_all_single",
         "broadcast"}
    )

    def __init__(self, cpu_group, device, window_bytes: int,
                 enabled: Optional[bool] = None, group: str = ""):
        import torch
        import torch.distributed as dist

        self.cpu_group = cpu_group
        self.device = device
        #: Name of the communicator group ("tp", "dcp", ...). It lives here
        #: because BAR1 is a PROCESS-WIDE resource: whatever this group
        #: pins down is unavailable to the next one. Without the name,
        #: there would be no way to either book it or say who holds the
        #: space.
        self.group = group
        self.rank = dist.get_rank(cpu_group)
        self.world = dist.get_world_size(cpu_group)
        self.window_bytes = int(window_bytes)
        self._up = False
        self._peers: dict[int, PeerTarget] = {}
        self._holder: Optional[Holder] = None
        self._cuda: Optional[_Cuda] = None
        self._own = (0, 0, 0)          # payload: dptr, handle, size
        self._own_flag = (0, 0, 0)     # flags:  dptr, handle, size
        self._own_sensor = None
        self._dmabuf_fds: list[int] = []       # own, exported
        self._hold_fds: list[int] = []        # /dev/nvidiactl, /dev/nvidiaN
        self._foreign_fds: list[list[int]] = []
        self._ext = None
        self._geo: dict = {}
        self._plan = None                      # optional plan from barlink_matrix
        # Capability, group-wide uniform. Only valid after _build_up.
        self._window_minimum = 0
        self._proofs_hold = False
        self._round_dev = None
        self._ctl_dev = None
        # #622: acknowledgment-bank offsets. Only valid after _build_up, like
        # the geometry they are derived from.
        self._ackbase_mesh = 0
        self._ackbase_a2a = 0
        # Peer liveness. Both stay None when SGLANG_BARLINK_PEER_LIVENESS=0 or
        # when the identity exchange fails; every use site then falls back to
        # the behaviour this transport had before task #312.
        self._peer_table = None
        self._abort_window = None
        # #431 fix 2: what the loud abort check needs, and nothing more.
        # Three plain attribute stores per collective on the hot path; every
        # derived quantity (rounds, message) is computed at raise time, where
        # the run is already broken and cost no longer matters.
        self._last_op = ""
        self._last_nbytes = 0
        #: Whether ``_last_op`` was recorded under stream capture (#583). It
        #: decides whether the raise may present the named collective as a
        #: member of the abort window at all: a captured launch is RECORDED,
        #: not executed, so it does not belong to any host-path window and
        #: can predate the abort by the whole run. Crash 2026-08-06 05:53:59
        #: named an 8-byte collective from capture as "its most recent
        #: member" and sent the investigation after a collective that never
        #: ran in that window.
        self._last_op_captured = False
        #: Collectives launched on the HOST path since the last device read of
        #: the status word. Zero means there is nothing new to look at, which
        #: is what makes a check at a boundary free when no traffic happened.
        self._unchecked_launches = 0
        #: True once a kernel of this transport has been recorded into a CUDA
        #: graph. A replay executes those kernels with no host code in
        #: between, so this is the flag that arms the replay-boundary check --
        #: the per-collective counter cannot see a replay at all.
        self._captured_launches = False
        self._registered_in_gate = False
        # #603b: make this transport's launch record readable from OUTSIDE, on
        # every rank at once. `_note_launch` already records the position, but
        # only the rank that RAISES ever prints it -- and exactly one rank
        # raises per crash, so the cross-rank comparison that would name a
        # POSITION offset (as opposed to the COUNT divergence the census
        # covers) has never been available. Registration is a weakref append;
        # the handler is host-side only and takes no device read.
        from sglang.srt.distributed.device_communicators import (
            barlink_launch_dump as _launch_dump,
        )

        _launch_dump.register(self)
        _launch_dump.install_sigusr1_handler()
        # The sampler, not the signal, is what actually observes a wedge: a
        # Python signal handler cannot run while the main thread sits in a
        # C-level CUDA sync, which is precisely the wedge state (verified
        # on-card: SIGUSR1 delivered to all three ranks, nothing logged).
        try:
            _launch_dump.start_sampler(int(getattr(self, "rank", -1)))
        except Exception:  # noqa: BLE001 - diagnostic must not break a boot
            pass
        # #517: the deferred status read. All of it stays None/0 until
        # bring-up decides (via `barlink_abort_gate.should_defer_status`)
        # that this transport's status word is a DEVICE word and therefore
        # worth staging; a host word is read directly, which is both cheaper
        # and stricter.
        self._ctl_defer = False
        self._ctl_src = None          # persistent 1-element view of _ctl_dev
        self._ctl_stage = None        # pinned host destination of the D2H
        self._ctl_event = None        # completion of the staged copy
        self._ctl_inflight = False
        self._ctl_lag = 0
        #: The watchdog-owned read of the same word, on a PRIVATE stream
        #: (#616f). The staged read above is ordered behind the compute
        #: stream by construction; when that stream is the thing that hung,
        #: the read hangs with it and the guard becomes part of the fault.
        #: This pair is ordered against nothing the model is doing, so it
        #: still observes the word while the compute stream is wedged. The
        #: device transport has carried this shape since #517 phase 2; the
        #: BAR1 transport did not, which is why a BAR1 wedge reported
        #: nothing at all.
        self._abort_poll_stream = None  # private stream for the watchdog read
        self._abort_poll_dst = None     # pinned host destination of that read
        self._round_mirror = None       # pinned (round, mesh wm, a2a wm) mirror
        self._abort_poll_active = False
        self._abort_code_seen = 0
        #: How often the staged read hit its deadline with the copy still in
        #: flight. Non-zero means the compute stream is not retiring work.
        self._ctl_sync_timeouts = 0
        #: The CONSECUTIVE run of those, reset by any resolved read. This is
        #: what escalates: one expiry is a slow step, an unbroken run is a
        #: stream that has stopped.
        self._ctl_stall_run = 0
        #: Seconds of stall this transport has already FORGIVEN because a peer
        #: published a build window (#615). Accumulated across extensions and
        #: never reset by a forgiven run, because it is what the absolute cap
        #: is applied to: a peer that keeps republishing a marker must not be
        #: able to buy unbounded time. Reset only by a RESOLVED read, i.e. by
        #: the stream actually retiring the copy.
        self._ctl_build_deferred_s = 0.0
        #: Collectives (host-path and captured) that have run since the last
        #: RESOLVED read of the status word. With a blocking read this always
        #: equals the current window; with a staged read it accumulates over
        #: the checks whose value had not arrived yet, so the raise still
        #: names the true size of the unverified window.
        self._deferred_launches = 0
        #: Replay-boundary entries since the last boundary check, so
        #: ``..._CHECK_EVERY`` reaches Seam B too (#517). The host-path
        #: counter cannot serve here: at a boundary it is zero by
        #: construction, which is exactly why the knob used to miss.
        self._boundary_checks = 0
        #: Latch for the expiry-path capture-census dump (#619). Once this
        #: fires the dump runs at most ONCE per process; re-entering the
        #: expiry path after the dump must not repeat it.
        self._expiry_census_fired = False

        if enabled is None:
            enabled = os.environ.get("SGLANG_BARLINK_MATRIX_DIRECT", "1") not in (
                "0", "no", "off", "false"
            )
        if not enabled:
            raise Bar1Unavailable(
                "disabled via SGLANG_BARLINK_MATRIX_DIRECT=0"
            )
        if self.world > MAX_RANGE:
            raise Bar1Unavailable(
                f"{self.world} ranks, but the kernel arguments hold at most "
                f"{MAX_RANGE}. The limit lives in barlink_bar1_ext.py "
                f"(BARLINK_BAR1_MAX_RANKS) and should be raised there in a "
                f"traceable way -- not worked around here."
            )
        self.ordinal = device.index if getattr(device, "index", None) is not None \
            else torch.cuda.current_device()
        # Operating parameters of the kernels. All rank-uniform, like every
        # other SGLANG_BARLINK* variable.
        self.threads = int(os.environ.get("SGLANG_BARLINK_BAR1_THREADS", "256"))
        # ~30 s at 2 GHz -- a stalled peer gets caught by a deadline in the
        # kernel instead of occupying the card indefinitely. Same order of
        # magnitude as BarlinkDeviceTransport._TIMEOUT_CYCLES, for the same
        # reason.
        self.cap_cycles = int(
            os.environ.get("SGLANG_BARLINK_BAR1_CAP_CYCLES", "60000000000")
        )
        # Flag load shape: 2 = ld.mmio.relaxed.sys (the only genuine
        # cache-bypass, the probe's default), 0 = ld.global.cv.
        self.load_shape = int(os.environ.get("SGLANG_BARLINK_BAR1_LOAD_SHAPE", "2"))
        # Read fence: only needed when payload and flag sit on different
        # PCIe targets. Here they do not; default off.
        self.read_flush = int(os.environ.get("SGLANG_BARLINK_BAR1_FLOW", "0"))
        # Payload size from which the cooperative multi-block launch kicks
        # in. 4 MiB, because in MESSUNG_ALLES_IM_SELBEN_LAUF.md the 'grid'
        # variant wins from 4 MiB up and '1blk' wins below it.
        self.grid_from = int(
            os.environ.get("SGLANG_BARLINK_BAR1_GRID_THRESHOLD", str(4 << 20))
        )
        # May the cooperative variant be launched WHILE a CUDA graph is being
        # captured? The default comes from SGLANG_BARLINK_GRAPH_ENABLE -- the
        # same release, the same gate (`bar1_graph_check.py`, case `grid`).
        # The derivation lives in `graph_grid_default`.
        #
        # What the headers on this rig provide (CUDA 12.9):
        # `CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2` is explicitly "Valid for
        # graph nodes, launches" (cuda.h:2043, driver_types.h:3800) -- a
        # cooperative launch is thus REPRESENTABLE as a graph node. That the
        # driver also accepts it out of a stream capture is proven twice on
        # this rig: the gate case `grid` passed, and the lever measurement
        # for #293 (phase A) with the same result.
        self.graph_grid = graph_grid_default()
        self._graph_grid_reported = False
        # Emergency mesh->ring threshold, if no plan is passed in.
        self.ring_from = int(
            os.environ.get("SGLANG_BARLINK_BAR1_RING_THRESHOLD", str(1 << 20))
        )
        self.min_bytes = int(os.environ.get("SGLANG_BARLINK_BAR1_MIN_BYTES", "4096"))
        self.max_bytes = 0
        # all_to_all occupies a third slot set in the same region and thus
        # costs a third of the largest all_reduce payload (see `geometry`).
        # Rank-uniform like every other SGLANG_BARLINK* variable; 0 restores
        # the old memory layout byte-for-byte.
        self.a2a_on = os.environ.get("SGLANG_BARLINK_BAR1_A2A", "1") not in (
            "0", "no", "off", "false"
        )
        #: Only valid after `byte_proof_a2a`. Without a passed proof,
        #: all_to_all withdraws -- all_reduce is unaffected by this.
        self._a2a_proof = False

        # -- mesh_pipe (pipelined mesh, barlink_bar1_pipe_ext) ----------------
        # OFF by default. Enabled, it occupies another slot set and four
        # flag lines per rank; disabled, every number and every offset in
        # this module is byte-for-byte the measured one.
        self.pipe_on = os.environ.get("SGLANG_BARLINK_BAR1_PIPE", "0") not in (
            "0", "no", "off", "false"
        )
        # RING DEPTH T -- slots per phase and connection. 4, from NCCL:
        # NCCL_STEPS 8 (src/include/device.h:26) divided by
        # ALLREDUCE_SLICESTEPS 2 (src/include/collectives.h:19).
        self.pipe_t = int(os.environ.get("SGLANG_BARLINK_BAR1_PIPE_T", "4"))
        # SCHEDULE LEAD P -- by how many loop rounds sending runs ahead of
        # reducing. SEPARATE from the ring depth, and that separation is
        # exactly the timing decoupling: the receiver may lag behind by
        # `T - P + 1` loop rounds before the sender blocks. With P = T that
        # would be exactly ONE round, i.e. effectively lockstep -- and on a
        # rig with x4, x8, and x8 links and three different card models,
        # the skew between unevenly fast ranks is exactly what the window
        # is meant to absorb. P = 2 is the minimum that pipelines at all;
        # with T = 4 that is three rounds of skew.
        self.pipe_lead = int(
            os.environ.get("SGLANG_BARLINK_BAR1_PIPE_LEAD", "2")
        )
        # Chunk count K. 0 = automatic, from `pipe_chunk_bytes`.
        self.pipe_k = int(os.environ.get("SGLANG_BARLINK_BAR1_PIPE_K", "0"))
        # Target size of a chunk under automatic K.
        #
        # 1 MiB, computed rather than guessed. Two barriers occur per loop
        # round; for the cooperative variant that is `grid.sync()`. Idle
        # time is `2*t_sync*(K+P)` (barriers) plus `(P-1)*T_leitung/K`
        # (pipeline ramp-up), minimized at
        # `K = sqrt((P-1)*T_leitung/(2*t_sync))`. At 16 MiB and two ranks,
        # `T_leitung` is around 1985 us (33.55 MB over the measured
        # full-duplex ceiling of 16.90 GB/s); with P=2 and t_sync=3 us that
        # gives K ~ 18 -- around 1 MiB per chunk.
        #
        # UNMEASURED: `t_sync` for `grid.sync()` has not been measured on
        # this rig; 3 us is an assumption. It applies to the 1blk variant --
        # and KORREKTUR_BANDBREITE.md measures 12.64 GB/s already with 256
        # threads in ONE block, i.e. the full write rate --, in which case
        # the barrier costs around 0.1 us and the optimal K is about five
        # times larger. That is the first axis of the measurement series.
        self.pipe_chunk_bytes = int(
            os.environ.get("SGLANG_BARLINK_BAR1_PIPE_CHUNK_BYTES", str(1 << 20))
        )
        self.pipe_k_max = int(os.environ.get("SGLANG_BARLINK_BAR1_PIPE_K_MAX", "64"))
        # Size of ONE pipe slot in KiB. 0 = computed from `pipe_chunk_bytes`
        # (`pipe_slot_default`), and that is the default.
        #
        # The region the pipe occupies in the BAR1 window is
        # `2 T (R-1) * slot`. It used to be a full slot set like mesh, ring,
        # and a2a, i.e. derived from `chunk_max` -- and thus from the
        # largest payload instead of from the chunk size a slot actually
        # needs to carry. At R=3, T=4, and a 96-MiB window, that was 32 MiB
        # of region for a 5.4-MiB requirement, and the difference was
        # missing from the all_reduce slot (8188 -> 6140 KiB). Anyone who
        # explicitly wants a larger slot thereby covers larger payloads at
        # small K and pays for it out of the same window.
        self.pipe_slot_kib = int(
            os.environ.get("SGLANG_BARLINK_BAR1_PIPE_SLOT_KIB", "0")
        )
        #: Only fixed during setup (from `pipe_slot_kib` or computed).
        self.pipe_slot = 0
        # SEPARATE 1blk/grid threshold for the pipe.
        #
        # Separate from `grid_from`, because the calculation for the pipe
        # comes out differently: `mesh` pays two `grid.sync()` per
        # collective, the pipe pays two per LOOP ROUND, i.e. 2(K+P) instead
        # of 2. And the reason the cooperative variant wins for `mesh` from
        # 4 MiB up does not apply to the data path at all:
        # KORREKTUR_BANDBREITE.md measures the write rate into the peer BAR
        # across 1 to 16 blocks and 32 to 1024 threads at 12.6-12.7 GB/s --
        # without variance. 256 threads in ONE block already reach the full
        # rate. More blocks do not help the transfer; they only help the
        # local reduction, whose cache-bypassing loads need concurrency.
        #
        # Default therefore unchanged (no silent difference), but as its
        # own lever: per the calculation above, the pipe should do better
        # with 1blk, and that is the first question of the measurement
        # series.
        self.pipe_grid_from = int(
            os.environ.get("SGLANG_BARLINK_BAR1_PIPE_GRID_THRESHOLD",
                           str(self.grid_from))
        )
        # Receiver acknowledgment (head). 1 = on. Turned off, it shows what
        # the sliding window costs; turned ON, it may only be used by
        # someone who has read the schedule proof in barlink_bar1_pipe_ext.
        self.pipe_ack = int(
            os.environ.get("SGLANG_BARLINK_BAR1_PIPE_ACK", "1")
        )
        # Payload size from which mesh_pipe runs instead of mesh. 256 KiB,
        # because below that a single chunk would remain and the pipe would
        # then just be the mesh's bookkeeping.
        self.pipe_from = int(
            os.environ.get("SGLANG_BARLINK_BAR1_PIPE_THRESHOLD", str(256 << 10))
        )
        # Direct mode: the all-gather writes into the receiver's result
        # buffer instead of into a slot the receiver would then have to read
        # out and copy over. Default ON as soon as the pipe is on -- that is
        # the whole point of the pipe. 0 is the control run with the same
        # memory layout.
        self.pipe_direct = os.environ.get(
            "SGLANG_BARLINK_BAR1_PIPE_DIRECT", "1"
        ) not in ("0", "no", "off", "false")
        # Direct mode WHILE a graph is being captured. Default OFF -- not
        # because it wouldn't work, but because it changes the memory layout
        # (flag family 4) and requires a larger result ring. Enabled, this
        # holds: per captured call site, ONE reserved ring slot from the
        # pool above the eager slots, plus the release handshake in the
        # kernel. The derivation lives in `_result_slot` and in
        # `barlink_bar1_pipe_ext.result_slot_split`.
        self.pipe_direct_graph = os.environ.get(
            "SGLANG_BARLINK_BAR1_PIPE_DIRECT_GRAPH", "0"
        ) not in ("0", "no", "off", "false")
        self._direct_graph_reported = False
        # How many result buffers the ring holds. Costs L*max_bytes in the
        # BAR window; 2 is the minimum with which round n does not write
        # into the buffer the caller still holds from round n-1.
        self.pipe_result_ring = int(
            os.environ.get("SGLANG_BARLINK_BAR1_PIPE_RESULT_RING", "2")
        )
        # How many ring slots the EAGER path retains. Default 2, i.e.
        # unchanged. The knob exists because the number is a property of
        # the caller: it must be as large as the number of results the
        # caller keeps alive at the same time. Under the graph-safe direct
        # mode, the failure occurs during the capture WARMUP, and that runs
        # eager -- a larger SGLANG_BARLINK_BAR1_PIPE_RESULT_RING there
        # allocates exclusively graph slots and does not help.
        from sglang.srt.distributed.device_communicators.barlink_bar1_pipe_ext import (  # noqa: E501
            RESULT_EAGER_SLOTS,
        )

        self.pipe_result_eager = int(
            os.environ.get("SGLANG_BARLINK_BAR1_PIPE_RESULT_EAGER",
                           str(RESULT_EAGER_SLOTS))
        )
        #: Only valid after `byte_proof_pipe`.
        self._pipe_proof = False
        self._pipe_ext = None
        self._step_dev = None
        self._result_gen_dev = None
        #: Running index into the result ring. HOST-SIDE and rank-uniform,
        #: because every rank sees the same sequence of collectives (SPMD)
        #: -- the same assumption `algorithm_for` already relies on. The
        #: kernel CANNOT choose it itself: the host must build the result
        #: tensor before the kernel runs.
        self._result_i = -1
        #: Weak references to the most recently handed-out result tensors,
        #: per ring slot. They are the liveness check.
        self._result_alive: list = []
        #: Running number of eager direct calls, and the number at which a
        #: slot was last assigned. From that falls out the reuse distance
        #: the release handshake needs as ``resultSlack`` -- under strict
        #: rotation it is the number of eager slots, fewer after skipping
        #: an occupied slot.
        self._result_counter = 0
        self._result_last: list = []
        #: How often an eager call found no free slot and therefore ran
        #: ``direct=0``. Reported once per rank; the counter itself keeps
        #: going, so "once at warmup" and "on every call" can be told
        #: apart.
        self._result_eager_full = 0
        self._result_eager_full_reported = False
        #: Split of the result ring into eager slots and graph slots. Fixed
        #: before the first call runs (`result_slot_split`), so the graph
        #: pool can never grab a slot whose eager tensor the caller still
        #: holds.
        self._result_eager_slots = 0
        self._result_graph_slots = 0
        #: How many graph slots have already been assigned. Only grows; a
        #: slot once assigned never comes back, because from here there is
        #: no way to tell whether the graph it belongs to is still alive.
        self._result_graph_assigned = 0
        self._result_graph_empty_reported = False
        #: Lower bound for all_to_all. Deliberately NOT `min_bytes` (4096):
        #: the whole appeal of a2a over BAR1 lies precisely in the small
        #: MoE dispatch blocks. 16 bytes = one packet.
        self.a2a_min_bytes = int(
            os.environ.get("SGLANG_BARLINK_BAR1_A2A_MIN_BYTES", "16")
        )
        #: all_gather over the a2a kernel. DEFAULT ON, and that is
        #: deliberate: without it, the standard run aborts during graph
        #: capture (the guard in barlink._select, correct and loud). The
        #: switch exists so a benchmarker can pit it against the gloo tier
        #: -- and because a new hot-path route needs an off switch that
        #: does not require a recompile. It only takes effect within
        #: SGLANG_BARLINK_TRANSPORT=bar1|matrix; without barlink it changes
        #: nothing.
        self.ag_on = os.environ.get("SGLANG_BARLINK_BAR1_AG", "1") not in (
            "0", "no", "off", "false"
        )
        #: Lower bound: **1 byte**. This used to be 16 ("one packet",
        #: carried over from a2a) for the same reason as with broadcast,
        #: and it is gone for the same reason. The finding came up with
        #: broadcast (12 bytes from the standard run, rejected by the 16);
        #: all_gather has the same structure, the same capture situation,
        #: and the same tail path in the kernel -- a 12-byte shard would
        #: have aborted the same way under capture. Leaving the threshold
        #: here would have meant knowingly leaving the twin of a crash that
        #: had just been proven in place.
        #:
        #: Not `min_bytes` (4096), and certainly not higher: the all_gather
        #: calls on the speculation and DP-attention paths are small, and
        #: that is exactly where the gloo tier's latency is most costly --
        #: if it even exists there, which it does not under capture.
        self.ag_min_bytes = int(
            os.environ.get("SGLANG_BARLINK_BAR1_AG_MIN_BYTES", "1")
        )
        #: How many rounds a shard may cost at most. Not a window limit but
        #: a round limit: each round is one kernel launch with one barrier.
        #: 16 carries a shard of ~128 MiB at a slot of just under 8 MiB
        #: (96-MiB window, R=3), and thus every size that occurs in this
        #: model -- the largest measured is 10.6 MB. Above that, the path
        #: withdraws instead of presenting a loop as a transport.
        self.ag_max_rounds = int(
            os.environ.get("SGLANG_BARLINK_BAR1_AG_MAX_ROUNDS", "16")
        )
        #: broadcast over the same a2a kernel. DEFAULT ON, for the same
        #: reason as all_gather: without it, the standard run aborts while
        #: capturing the draft graph (eagle_worker_v2.init_cuda_graphs ->
        #: parallel_state broadcast -> the guard in barlink._select). The
        #: switch exists so a benchmarker can pit it against the gloo tier.
        self.bc_on = os.environ.get("SGLANG_BARLINK_BAR1_BC", "1") not in (
            "0", "no", "off", "false"
        )
        #: Lower bound: **1 byte**, i.e. none at all. This used to be 16
        #: ("one packet"), copied from a2a, and that was wrong -- the
        #: standard run sends broadcast with 12 BYTES, and it failed on
        #: that right after the 128-byte case had just gotten through.
        #:
        #: The 16 had no technical reason. The kernel does write in 16-byte
        #: packets, but assembles the last, incomplete one from the
        #: available bytes in a register (``packBytes``) -- a 12-byte
        #: block is ONE such packet, not a special case. The slot begins on
        #: a page boundary and is a multiple of 16, so the four padding
        #: bytes land inside its own slot; the receive phase writes exactly
        #: ``rest`` bytes byte-by-byte, so it never runs past the output
        #: buffer. For a2a, the 16 is a statement about the packet
        #: granularity of the blocks; here it was a number carried over
        #: without justification.
        #:
        #: And it was the wrong KIND of limit: a lower bound is a
        #: worth-it threshold against the gloo tier, but under a CUDA graph
        #: capture there is no gloo tier, only the abort. That was exactly
        #: the rationale already written here, and it was disproved by the
        #: number sitting right next to it.
        #:
        #: Covered is thus everything from 1 byte to
        #: ``a2a_slot * bc_max_rounds``. Whoever turns the knob up is
        #: declining on purpose; it no longer happens silently.
        self.bc_min_bytes = int(
            os.environ.get("SGLANG_BARLINK_BAR1_BC_MIN_BYTES", "1")
        )
        #: Round limit as with all_gather, for the same reason: each round
        #: is one kernel launch with one barrier.
        self.bc_max_rounds = int(
            os.environ.get("SGLANG_BARLINK_BAR1_BC_MAX_ROUNDS", "16")
        )
        #: Round limit for all_reduce and all_to_all -- the same kind of
        #: limit as ag/bc and for the same reason: each round is one kernel
        #: launch with one barrier, and arbitrarily many of those per
        #: collective would not be a transport but a loop. 16 carries an
        #: all_reduce payload of ~384 MiB at an 8188-KiB slot and R=3, and
        #: thus every size that occurs in this model -- the standard run's
        #: working point is 20 MiB.
        self.ar_max_rounds = int(
            os.environ.get("SGLANG_BARLINK_BAR1_AR_MAX_ROUNDS", "16")
        )
        self.a2a_max_rounds = int(
            os.environ.get("SGLANG_BARLINK_BAR1_A2A_MAX_ROUNDS", "16")
        )
        #: Only valid after `byte_proof_broadcast`. Its own flag even
        #: though the same kernel runs: if the broadcast proof fails, that
        #: is not a verdict on all_to_all -- the table is a different one
        #: (exactly one sender), and a failure there carries no conclusion
        #: over here.
        self._bc_proof = False
        try:
            self._build_up()
        except BaseException:
            # A half-built transport is not left standing: the peers
            # already bound hold mappings, registrations, and attachments,
            # and those would otherwise survive until process exit --
            # along with the BAR1 pages they occupy.
            try:
                self.close()
            except Exception:
                pass
            raise

    # -- Capability ----------------------------------------------------------

    @staticmethod
    def patch_state() -> dict:
        """What the driver reveals about itself -- without any interpretation.

        ``BarlinkPeerBar1`` widens the guard from "within the static
        window of another GPU" to "within the BAR1 aperture of another
        GPU". The default is **0**; without the reg key, the guard behaves
        exactly as before, and ``cudaHostRegister(..., IoMemory)`` on a
        foreign BAR fails.

        Either spelling in ``PEER_BAR1_REGKEYS`` counts as set:
        ``RMSmallBarP2PPeerBar1`` is the name the key carried before #358,
        and the driver patch still reads it, so a module loaded before the
        rename is just as usable as one loaded after it.

        Only what is in ``/proc/driver/nvidia/params`` is reported. An
        empty ``RegistryDwords`` entry means: the reg key is not set. That
        is not proof that "the path is dead" -- the proof is the failed
        ``cudaHostRegister``, and that is exactly what setup waits for.
        """
        info = {"regkeys": "", "driver": "", "holder": os.path.exists(HOLDER_PATH)}
        try:
            with open("/proc/driver/nvidia/params") as f:
                for line in f:
                    # Exactly "RegistryDwords:" -- not
                    # "RegistryDwordsPerDevice:", which would otherwise
                    # overwrite the real one as a later line and report a
                    # reg key that is actually set as empty.
                    if line.startswith("RegistryDwords:"):
                        info["regkeys"] = line.strip()
        except OSError:
            pass
        try:
            with open("/proc/driver/nvidia/version") as f:
                info["driver"] = f.readline().strip()
        except OSError:
            pass
        info["peer_bar1_regkey"] = any(
            k in info["regkeys"] for k in PEER_BAR1_REGKEYS
        )
        return info

    # -- Setup -----------------------------------------------------------

    def _build_up(self) -> None:
        import torch
        import torch.distributed as dist

        from sglang.srt.distributed.device_communicators.barlink_matrix import (
            bdf_of_card,
        )

        if self.world < 2:
            raise Bar1Unavailable("fewer than two ranks -- nothing to do")

        # Peer liveness, before the first collective of the bring-up. From
        # here on every host wait in this transport can decide whether a peer
        # that has not arrived still exists, and the spin kernels get a host
        # word they can be told to abort through. Returns None when the
        # feature is off; every use site is guarded on that.
        self._peer_table = barlink_liveness.install(self.cpu_group)
        self._install_abort_window()

        t0 = time.perf_counter()
        self._cuda = _Cuda()
        self._holder = Holder()

        own_bdf = bdf_of_card(self.device)
        # BDF and window proposal in ONE all_gather. The proposal has to
        # travel along because the cards in the group have differently
        # sized apertures (3080: 256 MiB gross) and, in a process with two
        # groups, a different amount of it may already be spoken for. A
        # region that differs per rank would mean a different slot layout
        # per rank -- not an error, but writes landing at the wrong place.
        # Hence: a group-wide MINIMUM, and that decides.
        gathered: list[object] = [None] * self.world
        # torch runs the object collectives inline, so there is no Work to
        # bound; the one-shot check names a peer that is already gone instead
        # of letting gloo wait 7200 s for it.
        check_peers("bar1 bring-up: BDF and window exchange", self._peer_table)
        dist.all_gather_object(
            gathered, (own_bdf, int(self.window_bytes)),
            group=self.cpu_group,
        )
        self.bdfs = [str(x[0]) for x in gathered]      # type: ignore[index]
        proposals = [int(x[1]) for x in gathered]    # type: ignore[index]
        common = min(proposals)
        if common != self.window_bytes:
            logger.warning(
                "barlink-BAR1: per-rank window proposals %s MiB -- the "
                "group-wide minimum of %d MiB governs. This rank could "
                "have done %d MiB. The region is rank-uniform because the "
                "slot offsets in both kernels are computed from it.",
                [v // 2**20 for v in proposals], common // 2**20,
                self.window_bytes // 2**20,
            )
        if common <= 0:
            raise Bar1Unavailable(
                "0 bytes of BAR1 window remain group-wide. Another "
                "communicator in this process has claimed the aperture; "
                "the calculation is in the warning from "
                "barlink_matrix_transport.window_for. Either give the other "
                "group less (SGLANG_BARLINK_BAR1_WINDOW_MIB_<NAME>) or run "
                "this group explicitly over NCCL."
            )
        self.window_bytes = common

        # 0. The kernels. First, because a failed build is cheaper to abort
        # than a half-built peer table.
        from sglang.srt.distributed.device_communicators import barlink_bar1_ext

        try:
            self._ext = barlink_bar1_ext.load_collective_ext(self.cpu_group)
        except Exception as e:
            raise Bar1Unavailable(
                f"The collective extension could not be compiled: {e}"
            ) from e

        # 0b. The pipelined kernel, if enabled. A failed build disables it
        # instead of losing the whole transport -- mesh and ring are
        # unaffected by this.
        if self.pipe_on:
            from sglang.srt.distributed.device_communicators import (
                barlink_bar1_pipe_ext,
            )

            try:
                self._pipe_ext = barlink_bar1_pipe_ext.load_pipe_ext(self.cpu_group)
            except Exception as e:
                logger.warning(
                    "barlink-BAR1: the pipelined extension could not be "
                    "compiled (%s). mesh_pipe drops out; mesh and ring "
                    "continue unchanged.", e,
                )
                self.pipe_on = False
                self._pipe_ext = None

        # #728: RECONCILE pipe_on ACROSS THE GROUP, here, before anything is
        # derived from it.
        #
        # The build above is a RANK-LOCAL try/except: a compile that fails on
        # one rank's disk or ccache state and succeeds on its neighbour used to
        # leave the two with different pipe_on. That is not a local matter,
        # because pipe_on feeds max_payload() below, which feeds BOTH
        # ``max_bytes`` -- the ceiling ``handles()`` compares against, so two
        # ranks answer differently for the same payload and one enters a
        # collective the other does not -- AND ``geometry()``, which fixes the
        # slot OFFSETS.
        #
        # Reconciling here rather than later is the whole point. A late group
        # minimum on max_bytes would be exactly the "silent shrinking of the
        # payload" that the window check below refuses by name: "the slot
        # offsets are fixed in both kernels, and a rank with a different layout
        # would write to the wrong place". Fixing the INPUT keeps every derived
        # quantity uniform by construction and leaves that check untouched.
        #
        # AND-reduce, so the weakest rank decides: if any rank could not build
        # the extension, the whole group runs without it. Same shape as
        # ``parallel_state._harmonize_ca_comm_enablement``.
        if self.world > 1:
            local_pipe_on = bool(self.pipe_on)
            carrier_pipe: list[object] = [None] * self.world
            dist.all_gather_object(carrier_pipe, local_pipe_on, group=self.cpu_group)
            group_pipe_on = pipe_on_group_verdict(carrier_pipe)
            if group_pipe_on != local_pipe_on:
                logger.warning(
                    "barlink-BAR1: the pipelined extension is available on "
                    "this rank but NOT on every rank of the group (%s), so it "
                    "is disabled group-wide. One rank's failed build lowers "
                    "the payload ceiling for the whole session -- which is the "
                    "correct trade: a smaller ceiling is slow, a per-rank "
                    "ceiling hangs.",
                    [bool(x) for x in carrier_pipe],
                )
            self.pipe_on = group_pipe_on
            if not self.pipe_on:
                self._pipe_ext = None

        # 1. Memory layout. The largest payload follows from the window the
        # caller grants -- not the other way around.
        if self.pipe_on and not (2 <= self.pipe_lead <= self.pipe_t):
            raise Bar1Unavailable(
                f"SGLANG_BARLINK_BAR1_PIPE_LEAD={self.pipe_lead} does not "
                f"fit T={self.pipe_t}: 2 <= P <= T is required. P=1 "
                f"deadlocks (sending and consuming a chunk would fall into "
                f"the same loop round), P>T would let the schedule overtake "
                f"the slots."
            )
        pipe_range = 0
        if self.pipe_on:
            from sglang.srt.distributed.device_communicators.barlink_bar1_pipe_ext import (  # noqa: E501
                pipe_range_bytes,
                pipe_slot_default,
            )

            if self.pipe_slot_kib > 0:
                self.pipe_slot = (self.pipe_slot_kib * 1024 // 16) * 16
            else:
                self.pipe_slot = pipe_slot_default(
                    self.world, self.pipe_chunk_bytes
                )
            if self.pipe_slot <= 0:
                raise Bar1Unavailable(
                    f"Pipe slot size of {self.pipe_slot} bytes is not "
                    f"usable (from SGLANG_BARLINK_BAR1_PIPE_SLOT_KIB="
                    f"{self.pipe_slot_kib} resp. PIPE_CHUNK_BYTES="
                    f"{self.pipe_chunk_bytes} at {self.world} ranks)."
                )
            pipe_range = pipe_range_bytes(
                self.world, self.pipe_t, self.pipe_slot
            )
            logger.info(
                "barlink-BAR1-PIPE: ring depth T=%d, lead P=%d -- a peer may "
                "lag behind by %d loop rounds before the sender blocks. "
                "Direct mode %s, result ring L=%d, pipe slot %d KiB (%s), "
                "pipe region %.1f MiB.",
                self.pipe_t, self.pipe_lead,
                self.pipe_t - self.pipe_lead + 1,
                "on" if self.pipe_direct else "off", self.pipe_result_ring,
                self.pipe_slot // 1024,
                "set" if self.pipe_slot_kib > 0
                else f"from chunk target {self.pipe_chunk_bytes // 1024} KiB",
                pipe_range / 2**20,
            )
        if not self.pipe_on or not self.pipe_direct:
            self.pipe_result_ring = 0
        elif self.pipe_result_eager < 2:
            raise Bar1Unavailable(
                f"SGLANG_BARLINK_BAR1_PIPE_RESULT_EAGER={self.pipe_result_eager}: "
                f"direct mode needs at least two eager result buffers. With "
                f"only one, round n would write into exactly the buffer the "
                f"caller still holds from round n-1 -- a silently "
                f"overwritten result tensor, i.e. wrong numbers without a "
                f"crash. Whoever doesn't want the ring can disable direct "
                f"mode with SGLANG_BARLINK_BAR1_PIPE_DIRECT=0."
            )
        elif self.pipe_result_ring < self.pipe_result_eager:
            raise Bar1Unavailable(
                f"SGLANG_BARLINK_BAR1_PIPE_RESULT_RING={self.pipe_result_ring} is "
                f"smaller than SGLANG_BARLINK_BAR1_PIPE_RESULT_EAGER="
                f"{self.pipe_result_eager}. The ring holds the eager slots and "
                f"the graph pool together; it cannot be smaller than its "
                f"eager part."
            )
        max_bytes = max_payload(self.world, self.window_bytes, self.a2a_on,
                                 self.pipe_on, self.pipe_result_ring,
                                 pipe_range)
        if max_bytes < self.min_bytes:
            raise Bar1Unavailable(
                f"Window of {self.window_bytes // 1024} KiB carries only "
                f"{max_bytes} bytes of payload at {self.world} ranks, "
                f"minimum size is {self.min_bytes}. 4(R-1) slots of "
                f"ceil(N/R) each must fit."
            )
        self._geo = geometry(self.world, max_bytes, self.a2a_on, self.pipe_on,
                              self.pipe_result_ring, pipe_range)
        from sglang.srt.distributed.device_communicators.barlink_bar1_pipe_ext import (  # noqa: E501
            result_slot_split,
        )

        self._result_eager_slots, self._result_graph_slots = result_slot_split(
            self.pipe_result_ring, self.pipe_direct_graph, self.pipe_result_eager
        )
        self._result_alive = [None] * max(0, self._result_eager_slots)
        self._result_last = [None] * max(0, self._result_eager_slots)
        if self.pipe_direct_graph:
            logger.info(
                "barlink-BAR1-PIPE: graph-safe direct mode, result ring L=%d "
                "split into %d eager slots and %d graph slots. Each "
                "captured call site takes ONE graph slot and does not give "
                "it back; if the pool is empty, the capture runs the "
                "direct=0 path. More graphs need a larger "
                "SGLANG_BARLINK_BAR1_PIPE_RESULT_RING.",
                self.pipe_result_ring, self._result_eager_slots,
                self._result_graph_slots,
            )
        self.max_bytes = max_bytes
        region = self._geo["region_bytes"]
        flag_bytes = flags_requirement(self.world, self.a2a_on, self.pipe_on)
        # #622: the offsets of the two acknowledgment banks, computed ONCE
        # from the same (world, a2a_on, pipe_on) that sized the region above.
        # Kept as attributes and passed into the kernels as arguments -- the
        # C++ side must not recompute them, for the same reason `fbase_a2a`
        # is passed in: a second version of the formula is exactly where
        # sender and receiver end up pointing at different lines.
        self._ackbase_mesh = ackbase_mesh(self.world, self.a2a_on, self.pipe_on)
        self._ackbase_a2a = ackbase_a2a(self.world, self.a2a_on, self.pipe_on)

        # 2. Two receive regions, two exports. Separate, because the probe
        # measured them separately.
        dptr, handle, size = self._cuda.vmm_alloc(self.ordinal, region)
        self._own = (dptr, handle, size)
        fptr, fhandle, fsize = self._cuda.vmm_alloc(self.ordinal, flag_bytes)
        self._own_flag = (fptr, fhandle, fsize)
        # Flags to 0. Rounds start at 1, so no old marker can pass as a
        # valid acknowledgment.
        self._cuda.memset_d8(fptr, 0, fsize)

        route = ""
        for addr, hnd, nbytes in ((dptr, handle, size), (fptr, fhandle, fsize)):
            fd, hold, route = self._cuda.dmabuf_fd(addr, hnd, nbytes, self.ordinal)
            self._dmabuf_fds.append(fd)
            self._hold_fds.extend(hold)

        # 3. Exchange fds -- both in one message.
        self._foreign_fds = _exchange_fds(
            self.cpu_group, self.rank, self.world, self._dmabuf_fds
        )

        # 4.-6. attach each peer, offset from the sg-table, map, register.
        # This happens EXACTLY HERE and never again.
        for peer in range(self.world):
            if peer == self.rank:
                continue
            self._peers[peer] = self._bind_peer(peer, self._foreign_fds[peer])

        # 7. What is ACTUALLY mapped -- the group-wide minimum. Not the
        # gross size from sysfs and not the requested one: what governs is
        # the contiguous length the holder actually reported per peer. A
        # rank whose smallest window is smaller decides for everyone --
        # otherwise `handles` would answer differently per rank and the
        # SPMD assumption would be violated.
        local_min = min(z.payload.length for z in self._peers.values())
        local_flag_min = min(z.flag.length for z in self._peers.values())
        carrier: list[object] = [None] * self.world
        check_peers("bar1 bring-up: window minimum exchange", self._peer_table)
        dist.all_gather_object(
            carrier, (local_min, local_flag_min), group=self.cpu_group
        )
        self._window_minimum = min(int(x[0]) for x in carrier)   # type: ignore[index]
        flag_minimum = min(int(x[1]) for x in carrier)            # type: ignore[index]
        if self._window_minimum < region:
            raise Bar1Unavailable(
                f"{region} bytes of payload region were requested, at most "
                f"{self._window_minimum} bytes are mapped contiguously "
                f"group-wide. No silent shrinking of the payload: the slot "
                f"offsets are fixed in both kernels, and a rank with a "
                f"different layout would write to the wrong place."
            )
        if flag_minimum < flag_bytes:
            raise Bar1Unavailable(
                f"Flag region: {flag_bytes} bytes needed, at most "
                f"{flag_minimum} mapped group-wide."
            )

        # 8. Round counter and status word. Both LOCAL in VRAM -- they are
        # never touched by a peer.
        # #622: THREE words, not one. Word 0 is the running round counter as
        # before -- every kernel reads it at start and advances it at end, and
        # `bar1_all_reduce`/`bar1_all_to_all`/the pipe kernel all take the
        # tensor's base pointer, so word 0 keeps its meaning byte-for-byte.
        # Word 1 is the last MESH round this rank finished, word 2 the last
        # A2A round. They are the local half of the consumption acknowledgment:
        # a kernel entering mesh (a2a) waits until every peer has acknowledged
        # the round in word 1 (word 2), which is the round whose payload lines
        # it is about to overwrite. Local VRAM, never touched by a peer -- what
        # crosses the aperture are the ack lines in the flag region.
        self._round_dev = torch.zeros(3, dtype=torch.int64, device=self.device)
        self._ctl_dev = torch.zeros(2, dtype=torch.int32, device=self.device)
        # #517: arm the deferred status read now that the word exists.
        self._arm_status_stage()
        # #616f: and the watchdog's private-stream read of the same word.
        self._arm_abort_poll()
        # Absolute chunk counter of the sliding window. Separate from the
        # round counter, because it grows by K per call and is only
        # advanced by mesh_pipe -- it is the reference against which the
        # peers' head/tail lines are compared, and must therefore remain
        # absolute across calls. Rank-uniform, because every rank sees the
        # same sequence of calls with the same K.
        self._step_dev = torch.zeros(1, dtype=torch.int64, device=self.device)
        # Generation counter of the graph-safe direct mode. Also LOCAL in
        # VRAM: local accesses are coherent with one's own reads, and only
        # the PEER's view of the counter value needs the flag protocol --
        # that is carried by flag family 4 in the window. On the device and
        # not on the host, because it must keep counting on every graph
        # replay; a host counter gets baked in at capture time and then
        # sits still.
        self._result_gen_dev = torch.zeros(1, dtype=torch.int64,
                                        device=self.device)

        bounded_barrier(
            self.cpu_group,
            "bar1 bring-up: peer targets bound",
            table=self._peer_table,
        )
        self._up = True
        # #431 fix 2: only a transport that is UP can have a kernel abort, and
        # only now does `_ctl_dev` exist for the check to read. Registering
        # here rather than in __init__ keeps a transport that failed bring-up
        # out of the registry entirely -- `close()` withdraws it again.
        barlink_abort_gate.register(self)
        self._registered_in_gate = True
        # Into the ledger. Only NOW, because only now is it established that
        # the aperture actually gave up the space -- booking before the
        # holder would be a promise on spec, and the second group's ENOMEM
        # would then come from a reservation that does not actually exist.
        from sglang.srt.distributed.device_communicators import (
            barlink_matrix_transport as _ledger,
        )

        _ledger.ledger_credit(self.device, self.group, region + flag_bytes)
        duration = time.perf_counter() - t0
        logger.info(
            "barlink-BAR1: setup in %.0f ms, %d peer targets, region %.1f MiB "
            "per rank (%s), slot %d KiB, largest payload %d KiB, flags %d "
            "bytes, export via %s. From here on, nothing is mapped anymore "
            "on the hot path.",
            duration * 1000, len(self._peers), region / 2**20,
            f"{(6 if self.a2a_on else 4) * (self.world - 1)} slots"
            + (" (of which 2(R-1) for all_to_all)" if self.a2a_on
               else ", all_to_all disabled"),
            self._geo["chunk_max"] // 1024,
            max_bytes // 1024, flag_bytes, route,
        )
        # And log the ledger too. Without it, the next group that fails
        # with ENOMEM is again reduced to guessing.
        logger.info(
            "barlink-BAR1: BAR1 ledger of this card after group %r: %s.",
            self.group or "<unnamed>",
            ", ".join(f"{g or '<unnamed>'}: {b / 2**20:.1f} MiB"
                      for g, b in _ledger.ledger_balance(self.device)),
        )

    def _bind_peer(self, peer: int, foreign_fds: list) -> PeerTarget:
        """Attach, map, and register both regions of a peer."""
        dst_bdf = self.bdfs[peer]
        payload = self._bind_region(peer, dst_bdf, foreign_fds[0], "payload",
                                  self._geo["region_bytes"])
        try:
            flag = self._bind_region(peer, dst_bdf, foreign_fds[1], "flag",
                                      flags_requirement(self.world, self.a2a_on,
                                                     self.pipe_on))
        except BaseException:
            # Release the already-bound payload region again: it is not yet
            # recorded in any PeerTarget and would not be found by close().
            self._resolve_region(payload)
            raise
        return PeerTarget(rank=peer, bdf=dst_bdf, payload=payload, flag=flag)

    def _resolve_region(self, a: Mapping) -> None:
        if self._cuda is not None:
            self._cuda.unregister(a.reg_address)
        try:
            a.mmap_obj.close()              # type: ignore[attr-defined]
        except Exception:
            pass
        if self._holder is not None:
            self._holder.release(a.holder_handle)

    def _bind_region(self, peer: int, dst_bdf: str, foreign_fd: int,
                      kind: str, minimum: int) -> Mapping:
        assert self._cuda is not None and self._holder is not None
        window = bar1_window(dst_bdf)

        # Attach happens as THIS card -- it will write later.
        handle_, sg, total = self._holder.hold(foreign_fd, self.bdfs[self.rank])

        hits = [e for e in sg if window.base <= e.dma_address < window.end]
        if not hits:
            self._holder.release(handle_)
            raise Bar1Unavailable(
                f"None of the {len(sg)} sg-addresses of rank {peer} lie "
                f"within its BAR1 [{window.base:#x}, {window.end:#x}). "
                f"First address {sg[0].dma_address:#x}. This means either "
                f"that the IOMMU does not map identically (in which case "
                f"the offset derived from the sg-table does not hold and "
                f"the pattern scan would be needed), or that the driver did "
                f"not map into BAR1 at all. No guessing -- this edge is "
                f"dropped."
            )
        hits.sort(key=lambda e: e.dma_address)
        start = hits[0].dma_address
        # Contiguous? Only the contiguous beginning can be mapped as one
        # piece; the rest would be a second window.
        length = 0
        expected = start
        for e in hits:
            if e.dma_address != expected:
                break
            length += e.length
            expected += e.length
        offset = start - window.base
        if length < minimum:
            self._holder.release(handle_)
            raise Bar1Unavailable(
                f"{kind} region of rank {peer} ({dst_bdf}): "
                f"{minimum} bytes needed, but only {length} bytes are "
                f"mapped CONTIGUOUSLY in BAR1 (from {len(hits)} "
                f"sg-entries starting at {start:#x}). This is the length "
                f"that is checked against -- not the gross size from sysfs "
                f"({window.size} bytes), of which RM helps itself first."
            )

        page = mmap.PAGESIZE
        m_offset = (offset // page) * page
        lead_in = offset - m_offset
        m_length = length + lead_in
        res = f"/sys/bus/pci/devices/{dst_bdf}/resource1_wc"
        try:
            res_fd = os.open(res, os.O_RDWR | os.O_SYNC)
        except OSError as e:
            self._holder.release(handle_)
            raise Bar1Unavailable(
                f"{res} could not be opened ({e}). Without a "
                f"write-combining aperture there is no direct path."
            ) from e
        try:
            # ONLY the needed slice: an mmap over a 32-GiB window fails with
            # EINVAL (measured on the 5090).
            mapped = mmap.mmap(res_fd, m_length, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE, offset=m_offset)
        except (OSError, ValueError) as e:
            self._holder.release(handle_)
            raise Bar1Unavailable(
                f"mmap({res}, length={m_length}, offset={m_offset:#x}) "
                f"failed: {e}"
            ) from e
        finally:
            os.close(res_fd)

        host = ctypes.addressof(ctypes.c_char.from_buffer(mapped)) + lead_in
        try:
            self._cuda.register_io(host - lead_in, m_length)
        except Bar1Unavailable as e:
            mapped.close()
            self._holder.release(handle_)
            ps = self.patch_state()
            if not ps["peer_bar1_regkey"]:
                reason = (
                    f"This is the expected outcome WITHOUT the widened "
                    f"guard: the reg key {PEER_BAR1_REGKEYS[0]} defaults to "
                    f"0 and must be set via NVreg_RegistryDwords. "
                    f"Found: {ps['regkeys'] or '<empty>'}."
                )
            else:
                # The reg key is set. In that case, the guard is NO LONGER
                # the likely culprit -- and that mix-up was costly. Ahead
                # of the range guard, osCreateOsDescriptorFromIoMemory has
                # a second hurdle: the _PEER_MAP_OVERRIDE_REQUIRED branch
                # requires either peerMappingOverride or
                # osIsAdministrator(), and osIsAdministrator() on Linux is
                # capable(CAP_SYS_ADMIN) (os-interface.c:380 ->
                # nv-linux.h:499). A Docker container runs as root but does
                # NOT have CAP_SYS_ADMIN in the default set -- the call then
                # fails as NV_ERR_INSUFFICIENT_PERMISSIONS, which arrives
                # here as cudaError 800 (cudaErrorNotPermitted).
                reason = (
                    f"The reg key is set ({ps['regkeys']}), so it is not the "
                    f"range guard. Next suspect, in this order: (1) "
                    f"CAP_SYS_ADMIN is missing from the process -- "
                    f"osCreateOsDescriptorFromIoMemory requires, in the "
                    f"_PEER_MAP_OVERRIDE_REQUIRED branch, either "
                    f"peerMappingOverride or osIsAdministrator(), and that "
                    f"is capable(CAP_SYS_ADMIN); in a container, add "
                    f"'--cap-add SYS_ADMIN' or add PeerMappingOverride=1 to "
                    f"NVreg_RegistryDwords. (2) the range does not lie "
                    f"entirely within the peer's BAR1 aperture. The kernel "
                    f"log says unambiguously which of the two it was: "
                    f"'permission denied, allowPeermapping=0' versus "
                    f"'SMALLBAR_P2P: DENY ...'."
                )
            raise Bar1Unavailable(
                f"cudaHostRegister(IoMemory) on the {kind} region of rank "
                f"{peer} ({dst_bdf}) failed: {e}. {reason} "
                f"The transport withdraws, it does not force anything."
            ) from e
        dev = self._cuda.dev_ptr(host - lead_in)
        return Mapping(
            bar1_base=window.base, bar1_offset=offset, length=length,
            mmap_obj=mapped, reg_address=host - lead_in, host_address=host,
            dev_ptr=dev + lead_in, holder_handle=handle_,
        )

    # -- Window computation --------------------------------------------------

    def check_window_requirement(self, algorithm: str, nbytes: int) -> None:
        """Requirement against what could ACTUALLY be exported.

        Not against the gross size from sysfs: the 3080s report 256 MiB
        BAR1 gross, but how much of that is net available for peer mappings
        is unmeasured -- RM reserves part of it for itself. What governs is
        the length the holder actually mapped contiguously per peer.
        """
        needed = window_requirement(algorithm, nbytes, self.world)
        for peer, z in sorted(self._peers.items()):
            gross = bar1_window(z.bdf).size
            if needed > z.payload.length:
                raise Bar1Unavailable(
                    f"Window too small for '{algorithm}' at "
                    f"{nbytes // 1024} KiB and {self.world} ranks: needed "
                    f"{needed // 1024} KiB, but only {z.payload.length // 1024} "
                    f"KiB mapped at rank {peer} ({z.bdf}) (BAR1 gross "
                    f"{gross // 2**20} MiB). Either chunk smaller or "
                    f"exclude this edge. Mesh and ring BOTH need 2(R-1) "
                    f"slots -- the ring is no way out here."
                )

    def window_minimum(self) -> int:
        """Smallest actually mapped payload region in the GROUP.

        This is the number that belongs in the planner as
        ``window_bytes``: a **capability**, determined from what the
        holder actually mapped contiguously per peer, minimized across all
        ranks. A value that differs per rank would yield a different plan
        per rank, and the collectives' SPMD assumption depends on that not
        happening.
        """
        return self._window_minimum

    def algorithm_for(self, nbytes: int) -> str:
        """``mesh``, ``mesh_pipe``, or ``ring`` for this size.

        The plan from ``barlink_matrix.py`` takes precedence if one was
        passed in (``set_plan``). It is group-wide identical -- checked via
        the plan checksum -- and thus the only source that can keep this
        choice rank-uniform.

        Without a plan, the emergency threshold
        ``SGLANG_BARLINK_BAR1_RING_THRESHOLD``. It is a default, not a
        measured conclusion: between 1 and 16 MiB, mesh and ring differ by
        1 to 7 percent in the probe, and the finding says explicitly "no
        clean threshold -- the planner should measure this, not hard-code
        it".

        **THE ONE PLACE where mesh_pipe is chosen.** The planner does not
        know ``mesh_pipe`` and should not, for now: its cost models are
        calibrated against the two measured topologies. As long as
        ``SGLANG_BARLINK_BAR1_PIPE`` is off -- and that is the default --
        this method gives byte-for-byte the same answer as before.
        """
        if self._plan is not None:
            a = self._plan.algorithm_for(nbytes)
            # 'star' and 'hierarchical' are not ported here; they never
            # reach this point via handles() in the first place.
        else:
            a = "ring" if nbytes >= self.ring_from else "mesh"
        if (self.pipe_on and a == "mesh" and nbytes >= self.pipe_from
                and self._pipe_k(nbytes) is not None):
            return "mesh_pipe"
        return a

    def set_plan(self, plan) -> None:
        """Adopt the matrix planner's plan.

        Only once and only before the first collective: the choice must be
        fixed by the time a CUDA graph is captured.
        """
        self._plan = plan

    # -- Byte-level proof ------------------------------------------------

    def byte_proof_all(self, probe_bytes: int = 65536) -> dict[tuple[int, int], bool]:
        """For each directed pair: write a pattern, read it back on the target.

        The read-back goes through the **destination card's own VMM
        pointer**, not through the aperture -- otherwise a broken path
        could mask its own failure. That is exactly what caught the
        mailbox path on this rig: the driver reported peer access, and out
        of 1 MiB, 4096 bytes arrived.
        """
        import torch
        import torch.distributed as dist

        assert self._cuda is not None
        n = min(probe_bytes, self._own[2])
        result: dict[tuple[int, int], bool] = {}
        back = torch.empty(n, dtype=torch.uint8, pin_memory=True)
        for source in range(self.world):
            for dst in range(self.world):
                if source == dst:
                    continue
                marker = (source * 251 + dst * 37 + 1) & 0xFF
                pair_name = f"{source}->{dst}"
                bounded_barrier(
                    self.cpu_group,
                    f"bar1 byte proof {pair_name}: before clearing",
                    table=self._peer_table,
                )
                if self.rank == dst:
                    # Clear the destination first, so a buffer that was NOT
                    # written does not accidentally look like a hit.
                    blank = torch.full((n,), (marker ^ 0xFF) & 0xFF,
                                      dtype=torch.uint8, pin_memory=True)
                    self._cuda.memcpy(self._own[0], blank.data_ptr(), n)
                bounded_barrier(
                    self.cpu_group,
                    f"bar1 byte proof {pair_name}: destination cleared",
                    table=self._peer_table,
                )
                if self.rank == source:
                    pattern = torch.full((n,), marker, dtype=torch.uint8,
                                        device=self.device)
                    self.put(dst, pattern.data_ptr(), n, 0)
                    bounded_device_sync(
                        f"bar1 byte proof {pair_name}: pattern written",
                        device=self.device,
                        table=self._peer_table,
                    )
                bounded_barrier(
                    self.cpu_group,
                    f"bar1 byte proof {pair_name}: pattern in place",
                    table=self._peer_table,
                )
                ok = True
                if self.rank == dst:
                    self._cuda.memcpy(back.data_ptr(), self._own[0], n)
                    bad = int((back != marker).sum().item())
                    ok = bad == 0
                    if ok:
                        # The passed proof belongs in the log too: "0 of N
                        # bytes wrong" is the statement every later timing
                        # measurement rests on.
                        logger.info(
                            "barlink-BAR1: byte-level proof %d->%d passed: 0 "
                            "of %d bytes wrong.", source, dst, n,
                        )
                    else:
                        logger.warning(
                            "barlink-BAR1: byte-level proof %d->%d FAILED: %d "
                            "of %d bytes wrong. This edge is struck, "
                            "regardless of what the driver reports.",
                            source, dst, bad, n,
                        )
                carrier: list[object] = [ok if self.rank == dst else None]
                check_peers(
                    f"bar1 byte proof {pair_name}: verdict broadcast",
                    self._peer_table,
                )
                dist.broadcast_object_list(
                    carrier, src=dist.get_global_rank(self.cpu_group, dst),
                    group=self.cpu_group,
                )
                result[(source, dst)] = bool(carrier[0])
        for (q, z), ok in result.items():
            if q == self.rank and z in self._peers:
                self._peers[z].byte_proof = ok
        # ONE answer, group-wide. `result` is identical on every rank
        # (every entry was distributed from the destination), so this is
        # also rank-uniform -- exactly what `handles` needs.
        self._proofs_hold = all(result.values())
        if not self._proofs_hold:
            failed = sorted(k for k, v in result.items() if not v)
            logger.warning(
                "barlink-BAR1: byte-level proof failed for %s. The "
                "collectives withdraw (handles -> False); a collective over "
                "an edge that loses bytes would not be a collective.",
                failed,
            )
        return result

    # -- Data path -------------------------------------------------------

    def put(self, dst: int, source_ptr: int, nbytes: int, offset: int = 0,
            stream: Optional[int] = None) -> None:
        """A write into the destination card's BAR. Posted, hence fast.

        There is deliberately **no** ``get``: reading from a foreign BAR is
        non-posted and expensive (measured on the 2080 Ti at 1132 MB/s out
        versus 3254 MB/s in). Hence the rule "everyone pushes for
        themselves".
        """
        if not self._up:
            raise Bar1Unavailable("transport not set up")
        z = self._peers.get(dst)
        if z is None:
            raise Bar1Unavailable(f"no peer target for rank {dst}")
        if offset + nbytes > z.length:
            raise Bar1Unavailable(
                f"put({dst}): {offset}+{nbytes} exceeds the mapped "
                f"window of {z.length} bytes. The caller must chunk; "
                f"automatic re-mapping on the hot path is excluded -- it is "
                f"exactly the expensive part."
            )
        assert self._cuda is not None
        if stream is None:
            import torch

            stream = torch.cuda.current_stream(self.device).cuda_stream
        self._cuda.memcpy_async(z.dev_ptr + offset, source_ptr, nbytes, stream)

    # -- Measurement probe for barlink_matrix -------------------------------

    def name(self) -> str:
        return "bar1"

    def self_load(self, nbytes: int, direction: str) -> float:
        from sglang.srt.distributed.device_communicators.barlink_matrix import (
            SelfLoadSensor,
        )

        if getattr(self, "_own_sensor", None) is None:
            self._own_sensor = SelfLoadSensor(self.device, max_bytes=4 << 20)
        return self._own_sensor.self_load(nbytes, direction)

    def self_load_duplex(self, nbytes: int) -> Optional[float]:
        """Deliberately ``None``.

        Full duplex over the direct path is NOT measurable via host memory,
        and the relaxed driver guard exists because of a documented
        full-duplex deadlock (bug 1571948). As long as counter-traffic has
        not been verified over a full collective's duration, nothing is
        reported here that looks like a green light.
        """
        return None

    def pair(self, dst: int, nbytes: int) -> Optional[float]:
        """Directed edge GB/s -- only if the byte-level proof holds."""
        import torch

        if not self._up:
            return None
        if dst < 0:                       # planner's capability query
            return 0.0 if self._peers else None
        z = self._peers.get(dst)
        if z is None or not z.byte_proof:
            return None
        n = min(nbytes, z.length)
        source = torch.empty(n, dtype=torch.uint8, device=self.device)
        for _ in range(8):
            self.put(dst, source.data_ptr(), n, 0)
        bounded_device_sync(
            f"bar1 pair probe {self.rank}->{dst}: warm-up",
            device=self.device,
            table=self._peer_table,
        )
        rounds = 64 if n <= 65536 else 16
        t0 = time.perf_counter()
        for _ in range(rounds):
            self.put(dst, source.data_ptr(), n, 0)
        # Inside the timed section, so NOT bounded_device_sync: that one naps
        # up to 50 ms between polls, and a 150 ms transfer would report a
        # third less bandwidth than it delivers. Spinning on the bare event
        # predicate keeps the number honest and still ends on a dead peer.
        # With the feature off this is the plain synchronize it always was.
        if barlink_liveness.liveness_enabled():
            marker = f"bar1 pair probe {self.rank}->{dst}: timed writes"
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            barlink_liveness.bounded_poll(
                event.query,
                marker,
                table=self._peer_table,
                sleep=False,
                on_abort=self._wait_abort(marker),
            )
        torch.cuda.synchronize(self.device)
        dt = time.perf_counter() - t0
        return (rounds * n) / dt / 1e9 if dt > 0 else 0.0

    def pair_receive(self, source: int, nbytes: int) -> None:
        """Nothing to do -- the writes are one-sided.

        The destination card takes no active part in the transfer; its BAR
        absorbs the bytes without doing anything. That is exactly why the
        path costs so little.
        """
        return None

    # -- Transport seam ----------------------------------------------------

    def handles(self, op: str, nbytes: int) -> bool:
        """``True`` only if the path can really carry this operation.

        Every condition is **rank-uniform**: it depends only on group-wide
        reconciled state (``_proofs_hold`` comes from a distribution per
        directed pair, ``_window_minimum`` from an ``all_gather``, the
        thresholds from rank-uniform environment variables). Two ranks must
        never answer differently here -- one would run into the collective and
        the other would not, and the result would be a hang instead of an
        error.

        ``max_bytes`` (#728) is uniform BY CONSTRUCTION rather than by a
        gather of its own: it is ``max_payload()`` of inputs that are each
        already reconciled -- ``window_bytes`` from the window exchange, and
        ``pipe_on`` AND-reduced across the group right after its build. This
        docstring previously claimed ``max_bytes`` came "from an all_gather".
        It did not, and it does not now; it comes from reconciled inputs, which
        is the stronger property because the SLOT OFFSETS are derived from the
        same value and a value reconciled after the fact would leave the layout
        divergent.

        The data type is NOT a factor: ``handles`` does not see it. The
        extension accepts float32/float16/bfloat16 and rejects everything
        else with a reason. This is the same state as in ``barlink_device``.
        """
        if op not in self.BARLINK_OPS:
            return False
        if not self._up or self._ext is None:
            return False
        if not self._proofs_hold:
            return False
        if op in ("all_to_all", "all_to_all_single"):
            return self._handles_a2a(nbytes)
        if op == "all_gather":
            return self._handles_all_gather(nbytes)
        if op == "broadcast":
            return self._handles_broadcast(nbytes)
        return self._handles_all_reduce(nbytes)

    def _handles_all_reduce(self, nbytes: int) -> bool:
        """``all_reduce``, now decomposed into rounds instead of using a cap.

        What changed and why: up to here, this answer ended in False at
        ``largest_chunk > chunk_max``, and the payload fell back to the
        base transport. The tipping point in the standard run is 2456
        tokens per batch -- ``chunked_prefill_size`` 4096 or 8192, both
        common in sglang, would have turned off the direct path during
        prefill without a single line reporting it. Instead of a refusal,
        the payload now runs in ``ceil`` rounds (:func:`ar_plan`), the way
        all_gather and broadcast have long done.

        Every condition is rank-uniform, for the same reason as in
        :meth:`handles`: it depends only on group-wide identical sizes.
        """
        if nbytes < self.min_bytes:
            return False
        if nbytes % 16 != 0:
            # The access width is 128 bits; a remainder would be a second,
            # unmeasured path in the kernel. The extension insists on this
            # (TORCH_CHECK in the host), this is not a caution rule here.
            return False
        if nbytes // 16 < self.world:
            # Fewer than one packet per rank -- the chunk decomposition
            # would leave ranks empty-handed, and the host rejects it.
            return False
        chunk_max = int(self._geo.get("chunk_max", 0))
        if chunk_max < 16:
            return False
        # The round limit replaces the old size cap. It is not a window
        # limit but a limit on kernel launches.
        rounds = ar_plan(nbytes, chunk_max, self.world)
        if len(rounds) > self.ar_max_rounds:
            return False
        # The LARGEST round is checked. It is the only one that could
        # fail -- and it is the same group-wide.
        largest_round = max(length for _, length in rounds)
        if largest_round > self.max_bytes:
            return False
        # Does the LARGEST chunk of ONE ROUND fit into a slot? This is the
        # condition the mapping actually depends on -- checked, not
        # inferred from `nbytes <= max_bytes`. The extension computes it a
        # second time, there with its own chunk bounds rather than this
        # formula: a seam checked on both sides with the same wrong formula
        # would not stand out.
        largest_chunk = -(-(largest_round // 16) // self.world) * 16
        if largest_chunk > chunk_max:
            return False
        # The algorithm is decided PER ROUND, so it is checked per round.
        # For a single round, this is byte-for-byte the old question.
        for _, length in rounds:
            algo = self.algorithm_for(length)
            if algo not in ("mesh", "mesh_pipe", "ring"):
                # 'star' and 'hierarchical' are not ported here. No silent
                # fallback to 'mesh'.
                return False
            if algo == "mesh_pipe" and not self._pipe_supports(length):
                return False
            # And the same requirement again, in the planner's currency,
            # against the group-wide SMALLEST actually mapped length.
            # Redundant as long as setup completed successfully -- and
            # exactly for that reason cheap: this line catches anyone who
            # touches the region size without carrying the window concept
            # along.
            if window_requirement(algo, length, self.world) > self._window_minimum:
                return False
        return True

    def ar_rounds(self, nbytes: int) -> int:
        """Round count for ``nbytes`` -- for logging and tests."""
        chunk_max = int(self._geo.get("chunk_max", 0))
        if chunk_max < 16 or nbytes <= 0 or nbytes % 16:
            return 0
        return len(ar_plan(nbytes, chunk_max, self.world))

    def why_not(self, op: str, nbytes: int) -> str:
        """Why ``handles`` says False for this size -- in words.

        For the log message only, never for a decision. That is also why
        it may be coarse: it names the FIRST condition that fails, which
        answers the question the log really asks -- "is this about the
        size, the window, or the fact that the path isn't set up at all".

        It is deliberately NOT a second version of ``handles``: it decides
        nothing, and if it is ever wrong, the damage is an imprecise
        sentence in the log. A silent fallback, by contrast, costs a
        measurement.
        """
        if op not in self.BARLINK_OPS:
            return f"{op} is not in BARLINK_OPS"
        if not self._up or self._ext is None:
            return "the direct path is not set up"
        if not self._proofs_hold:
            return "the byte-level proof per pair does not hold"
        slot = int(self._geo.get("a2a_slot", 0))
        chunk_max = int(self._geo.get("chunk_max", 0))
        if op in ("all_to_all", "all_to_all_single"):
            if not self.a2a_on:
                return "all_to_all is disabled via SGLANG_BARLINK_BAR1_A2A=0"
            if not self._a2a_proof:
                return "the a2a byte-level proof does not hold"
            if nbytes < self.a2a_min_bytes:
                return f"{nbytes} bytes are below a2a_min_bytes ({self.a2a_min_bytes})"
            n = a2a_rounds(-(-nbytes // self.world), slot) if slot else 0
            if n > self.a2a_max_rounds:
                return (f"would need {n} rounds at a {slot}-byte slot, "
                        f"{self.a2a_max_rounds} are allowed")
        elif op == "all_gather":
            if not self.ag_on:
                return "all_gather is disabled via SGLANG_BARLINK_BAR1_AG=0"
            if not self._a2a_proof:
                return "the a2a byte-level proof does not hold (all_gather rides on it)"
            if nbytes < self.ag_min_bytes:
                return f"{nbytes} bytes are below ag_min_bytes ({self.ag_min_bytes})"
            if slot and -(-nbytes // slot) > self.ag_max_rounds:
                return (f"would need {-(-nbytes // slot)} rounds at a "
                        f"{slot}-byte slot, {self.ag_max_rounds} are allowed")
        elif op == "broadcast":
            if not self.bc_on:
                return "broadcast is disabled via SGLANG_BARLINK_BAR1_BC=0"
            if not self._bc_proof:
                return "the broadcast byte-level proof does not hold"
            if nbytes < self.bc_min_bytes:
                return f"{nbytes} bytes are below bc_min_bytes ({self.bc_min_bytes})"
            if slot and -(-nbytes // slot) > self.bc_max_rounds:
                return (f"would need {-(-nbytes // slot)} rounds at a "
                        f"{slot}-byte slot, {self.bc_max_rounds} are allowed")
        else:
            if nbytes < self.min_bytes:
                return f"{nbytes} bytes are below min_bytes ({self.min_bytes})"
            if nbytes % 16:
                return (f"{nbytes} bytes are not a multiple of 16 -- the "
                        f"kernel's access width is 128 bits")
            if nbytes // 16 < self.world:
                return (f"{nbytes} bytes are fewer than one 128-bit packet "
                        f"per rank ({self.world})")
            if chunk_max >= 16:
                n = len(ar_plan(nbytes, chunk_max, self.world))
                if n > self.ar_max_rounds:
                    return (f"would need {n} rounds at a chunk bound of "
                            f"{chunk_max} bytes, {self.ar_max_rounds} are "
                            f"allowed")
        if self._geo.get("region_bytes", 0) > self._window_minimum:
            return (f"the region ({self._geo.get('region_bytes')} bytes) does "
                    f"not fit into the group-wide smallest mapped window "
                    f"({self._window_minimum} bytes)")
        return ""

    def _kernel(self, moved: int, threshold: int, where: str) -> int:
        """``1`` = cooperative multi-block launch (``grid``), ``0`` = ``1blk``.

        **The one place where this choice is made** -- previously, each of
        the three collectives computed `moved >= threshold` for itself, and
        a capture rule could have been added at three places and forgotten
        at one of them.

        Two inputs, both rank-uniform: the size (group-wide identical) and
        the threshold (environment variable). Plus, if a capture is
        currently in progress, the restriction against the cooperative
        launch.

        On the rank-uniformity of the capture: it holds because the graph
        runner captures the same shapes in the same order on every rank.
        If it did not hold, we would already have a bigger problem than the
        kernel variant -- one rank in the collective, the other not, i.e. a
        hang.
        """
        if moved < threshold:
            return 0
        if self.graph_grid:
            return 1
        from sglang.srt.distributed.device_communicators.barlink import (
            graph_capture_running,
        )

        if not graph_capture_running():
            return 1
        if not self._graph_grid_reported:
            self._graph_grid_reported = True
            logger.warning(
                "barlink-BAR1: %s with %d bytes would be above the grid "
                "threshold (%d bytes), but is placed on the 1blk variant "
                "while a CUDA graph is being captured -- because the "
                "restriction is either explicitly set via "
                "SGLANG_BARLINK_BAR1_GRAPH_GRID=0 or SGLANG_BARLINK_GRAPH_ENABLE "
                "is not set. This costs: in the lever measurement for #293 "
                "it was 16.1%% prefill throughput once prefill was "
                "captured. The cooperative launch IS capturable on this "
                "rig (benchmark/bar1_graph_check.py, case 'grid'); with the "
                "release set, the restriction drops away on its own. This "
                "notice appears once per rank.",
                where, moved, threshold,
            )
        return 0

    def barlink_all_reduce(self, comm, inp):
        """Sum-allreduce over ``mesh`` or ``ring``, out of place.

        Out of place is not a convenience: the ring still reads ``in``
        while it is already writing forward into ``out`` (step s+1 sends
        the partial sum formed in step s). The extension checks this and
        rejects identical pointers.
        """
        import torch

        if not self._up or self._ext is None:
            raise Bar1Unavailable(
                "barlink_all_reduce without a transport set up -- reachable "
                "only if someone bypassed handles()."
            )
        inp = inp.contiguous()
        nbytes = inp.numel() * inp.element_size()
        chunk_max = int(self._geo.get("chunk_max", 0))
        plan = ar_plan(nbytes, chunk_max, self.world) if chunk_max >= 16 else []
        if len(plan) > 1:
            # MULTIPLE ROUNDS. Each is a complete all_reduce over a slice --
            # no new kernel, no different decomposition, just fewer bytes
            # per launch. The round count comes from `ar_plan` and depends
            # solely on group-wide identical sizes; it is thus baked in for
            # a captured shape.
            #
            # The slices are views into the flat buffer, not copies:
            # `offset` and `length` are multiples of 16, so every view
            # stays 16-byte-aligned, which the host insists on.
            result = torch.empty_like(inp)
            flat_in = inp.view(-1)
            flat_out = result.view(-1)
            eg = inp.element_size()
            for offset, length in plan:
                part = self._all_reduce_one_round(
                    flat_in[offset // eg:(offset + length) // eg]
                )
                flat_out[offset // eg:(offset + length) // eg].copy_(part)
            return result
        return self._all_reduce_one_round(inp)

    def _all_reduce_one_round(self, inp):
        """ONE kernel launch. The previous body, unchanged.

        Factored out so the round loop above it does not need to know
        anything about the algorithm choice, the pipe, or the pointer
        table -- and so the single-round case remains byte-for-byte the
        same path as before.
        """
        import torch

        inp = inp.contiguous()
        nbytes = inp.numel() * inp.element_size()
        algo = self.algorithm_for(nbytes)
        if algo == "mesh_pipe":
            k = self._pipe_k(nbytes)
            if k is None:
                raise Bar1Unavailable(
                    "mesh_pipe without a matching chunk count -- reachable "
                    "only if someone bypassed handles()."
                )
            return self._pipe_all_reduce(inp, k)
        out = torch.empty_like(inp)
        # 'grid' is the cooperative multi-block launch. The threshold is
        # measured (it wins from 4 MiB up), but it is a number from ONE
        # rig -- hence it lives in an environment variable. Under graph
        # capture, `_kernel` additionally decides.
        kernel_variant = self._kernel(nbytes, self.grid_from, "all_reduce")
        peer_payload = [0] * self.world
        peer_flag = [0] * self.world
        for r, z in self._peers.items():
            peer_payload[r] = z.payload.dev_ptr
            peer_flag[r] = z.flag.dev_ptr
        peer_payload[self.rank] = self._own[0]
        peer_flag[self.rank] = self._own_flag[0]
        self._note_launch("all_reduce", nbytes, kernel_variant)
        self._ext.bar1_all_reduce(
            inp, out, int(self.rank), int(self.world),
            0 if algo == "mesh" else 1,
            peer_payload, peer_flag,
            int(self._own[0]), int(self._own_flag[0]),
            int(self._geo["chunk_max"]), int(self._geo["off_mesh"]),
            int(self._geo["off_ring"]),
            self._round_dev, self._ctl_dev,
            self._deadline_cycles(), int(self.threads), int(kernel_variant),
            int(self.load_shape), int(self.read_flush),
            int(self._abort_host),
            # #622: appended, not inserted. Every existing positional index
            # into this call stays where it was -- the same discipline the
            # flag region follows for its new banks.
            int(self._ackbase_mesh),
        )
        return out

    # -- mesh_pipe ---------------------------------------------------------
    #
    # Everything beyond the choice itself lives in barlink_bar1_pipe_ext: the
    # kernel, the slot and counter geometry, the chunk planning, and the
    # byte-level proof. Here there are only the three lines with which the
    # transport puts it to use.

    def _pipe_k(self, nbytes: int):
        """Chunk count for this payload, or ``None``.

        Rank-uniform: every input is group-wide identical. ``None`` means
        "the pipelined path cannot carry this" and leads to False in
        ``handles`` -- not to a silent fallback.

        **Memoized per size.** ``pipe_plan`` computes the decomposition
        over all (chunk, rank) pairs -- deliberately, rather than with a
        closed-form second version --, and that does not belong on the hot
        path. ``handles`` and ``barlink_all_reduce`` ask per collective; the
        sizes repeat.
        """
        if not self.pipe_on or self._pipe_ext is None:
            return None
        if self._geo.get("off_pipe", -1) < 0:
            return None
        memo = getattr(self, "_pipe_k_memo", None)
        if memo is None:
            memo = {}
            self._pipe_k_memo = memo
        if nbytes in memo:
            return memo[nbytes]
        from sglang.srt.distributed.device_communicators.barlink_bar1_pipe_ext import (
            pipe_plan,
        )

        k = pipe_plan(
            int(nbytes), int(self.world), int(self.pipe_slot),
            int(self.pipe_t), int(self.pipe_k), int(self.pipe_chunk_bytes),
            int(self.pipe_k_max),
        )
        memo[nbytes] = k
        return k

    def _pipe_supports(self, nbytes: int) -> bool:
        """Window limit for ``mesh_pipe`` -- computed, not assumed.

        The requirement is ``2 T (R-1)`` slots of ``chunk_max/T`` each,
        computed in ``pipe_window_requirement``. Checked against the
        group-wide SMALLEST **actually mapped** length (``_window_minimum``),
        not against the gross size from sysfs and not against the requested
        region: what governs is what the holder actually found mapped
        contiguously in BAR1 per peer. If it does not fit, this path
        withdraws via ``handles``.
        """
        if not self._pipe_proof:
            return False
        from sglang.srt.distributed.device_communicators.barlink_bar1_pipe_ext import (
            result_ring_bytes,
            pipe_window_requirement,
        )

        off = int(self._geo.get("off_pipe", -1))
        if off < 0:
            return False
        needed = off + pipe_window_requirement(
            self.world, int(self.pipe_t), int(self.pipe_slot)
        )
        # And the result ring on top: L * roundup(max_bytes, PAGE_SIZE). It sits
        # BEHIND the slots, so the requirement is the ring's offset plus its
        # length -- not the maximum of the two.
        if int(self._geo.get("off_result", -1)) >= 0:
            needed = max(needed, int(self._geo["off_result"]) + result_ring_bytes(
                int(self.max_bytes), int(self._geo["result_ring"])
            ))
        return needed <= self._window_minimum

    def result_window(self) -> Optional[tuple[int, int]]:
        """``(start, length)`` of the result ring within the caller's own window.

        For proofs that want to check whether a result tensor REALLY lies
        within the exported window -- i.e. whether direct mode actually
        ran. Without this information, a proof would have to reach into
        the internals, and a proof that silently measures the control path
        proves nothing (measurement discipline, rule 5).

        ``None`` means: there is no result ring, so no call could have run
        in direct mode.
        """
        off = int(self._geo.get("off_result", -1)) if self._geo else -1
        if off < 0 or not self._own[0]:
            return None
        from sglang.srt.distributed.device_communicators.barlink_bar1_pipe_ext import (  # noqa: E501
            result_ring_bytes,
        )

        length = result_ring_bytes(int(self.max_bytes),
                                int(self._geo["result_ring"]))
        return int(self._own[0]) + off, int(length)

    def _result_slot(self, inp):
        """Result buffer and ownership info -- or ``None``.

        Returns ``(tensor, slot, slack)``:

        ``tensor``  the result buffer, a tensor OVER the window,
        ``slot``   its ring slot -- the caller needs it to build the
                    ``peer_result`` table,
        ``slack``   after how many generations this slot is overwritten
                    again; ``0`` means "handshake off".

        **Why a return value and not a field.** The slot belongs to exactly
        this buffer. As an object field, it would be a returned buffer
        with an ordering assumption that lives only in a comment -- the
        bug family that has already hit this transport twice. Whoever
        holds the tensor holds its slot along with it.

        ``None`` means "no direct mode"; the caller then falls back to
        ``torch.empty_like``. Visible, not silent.

        **The liveness check.** The ring assigns slot ``i`` round-robin.
        Before slot ``i`` is overwritten, the tensor this slot handed out
        ``L`` rounds ago must be dead. This is checked with a weak
        reference: if it is still alive, someone holds it, and overwriting
        it would mean silently writing different numbers into a tensor the
        caller believes is finished -- wrong results without a crash, the
        worst conceivable failure mode of this project. That is why it
        aborts here with a reason rather than silently working around it.

        **What this check does NOT cover**, and why that is stated here: it
        sees Python references, not running kernels. As long as the result
        and the following consumer sit on the SAME stream -- and they do in
        sglang -- the stream orders the accesses. Anyone who further
        processes the result tensor on a different stream must synchronize
        themselves; direct mode should then be turned off
        (``SGLANG_BARLINK_BAR1_PIPE_DIRECT=0``).
        """
        import weakref

        if not self.pipe_direct or self._geo.get("off_result", -1) < 0:
            return None
        # -- Capture -------------------------------------------------------
        #
        # This method is HOST CODE. It runs exactly once during capture and
        # never again on replay. The chosen ring slot, the pointer computed
        # from it, and the kernel's `peer_result` table get baked into the
        # graph. This has three consequences, and they are the reason for
        # splitting the ring:
        #
        # 1. The slot of a captured call is fixed. On its own, that is just
        #    graph-standard -- a graph has fixed output buffers.
        # 2. The liveness check below -- the weak reference -- no longer
        #    applies on replay, because no host code runs. For a slot
        #    reserved permanently for a graph, it does not need to: the
        #    slot belongs to exactly this one call site and is touched by
        #    no other. What it does NOT replace is the distance between
        #    two replays of the SAME slot -- that is carried by the
        #    release handshake in the kernel (flag family 4,
        #    `resultSlack = 1`).
        # 3. The silent bug was this: with a freely running ring index,
        #    multiple captures (sglang captures one per batch size) run
        #    over the same slots. Two graphs then share one BAR1 slot, and
        #    whoever replays them alternately gets the other one's numbers.
        #    No crash. That is exactly what the pool eliminates: a graph
        #    slot is assigned ONCE and never again.
        #
        # If the pool is empty, the call falls back to `direct=0` --
        # reported, not silent, and correct: `direct=0` is the same
        # measured control path, its `torch.empty_like` during capture
        # comes from the graph's private memory pool and thus already has
        # a fixed address regardless. It costs the saved VRAM pass, not
        # correctness.
        from sglang.srt.distributed.device_communicators.barlink import (
            graph_capture_running,
        )
        from sglang.srt.distributed.device_communicators.barlink_bar1_pipe_ext import (  # noqa: E501
            result_eager_free_slot,
            result_eager_slack,
            result_graph_slot,
        )

        if graph_capture_running():
            if not self.pipe_direct_graph:
                if not self._direct_graph_reported:
                    self._direct_graph_reported = True
                    logger.warning(
                        "barlink-BAR1-PIPE: direct mode disabled while a CUDA "
                        "graph is being captured (default). mesh_pipe runs "
                        "the captured direct=0 path. "
                        "SGLANG_BARLINK_BAR1_PIPE_DIRECT_GRAPH=1 enables the "
                        "graph-safe path: a reserved ring slot per call "
                        "site plus a release handshake in the kernel. This "
                        "notice appears once per rank."
                    )
                return None
            i = result_graph_slot(self._result_graph_assigned,
                                self._result_eager_slots,
                                self._result_graph_slots)
            if i is None:
                if not self._result_graph_empty_reported:
                    self._result_graph_empty_reported = True
                    logger.warning(
                        "barlink-BAR1-PIPE: the graph pool of the result ring "
                        "is exhausted (%d of %d slots assigned, L=%d). This "
                        "and every further captured call site runs the "
                        "direct=0 path -- correct, but without the saved "
                        "VRAM pass. More slots are available with a larger "
                        "SGLANG_BARLINK_BAR1_PIPE_RESULT_RING, and each slot "
                        "costs %d bytes in the BAR1 window.",
                        self._result_graph_assigned, self._result_graph_slots,
                        int(self._geo["result_ring"]),
                        int(self._geo["result_stride"]),
                    )
                return None
            # NO weak reference: from now on, this slot belongs to exactly
            # this call site. Assigning it again later would be the bug --
            # and that is exactly why the counter only grows.
            self._result_graph_assigned += 1
            ptr = (self._own[0] + int(self._geo["off_result"])
                   + i * int(self._geo["result_stride"]))
            out = self._pipe_ext.bar1_result_tensor(int(ptr), inp)
            return out, i, 1

        # -- eager -----------------------------------------------------------
        #
        # What is sought is the next FREE slot, not just the next one
        # checked. This used to be a hard abort as soon as the one
        # successor slot was still held -- and that is exactly where the
        # graph-safe direct mode failed to get off the ground in the lever
        # measurement for #293: the failure occurred during the capture
        # WARMUP, which runs eager, and `SGLANG_BARLINK_BAR1_PIPE_RESULT_RING`
        # did not help, because a larger ring only assigns graph slots.
        #
        # The abort was the wrong answer to the right concern. What must
        # not happen is overwriting a buffer that is still held. If ALL
        # eager slots are held, the correct answer is the same as for the
        # exhausted graph pool a couple of lines up: `direct=0`, reported
        # and counted. That is the measured control path and costs the
        # saved VRAM pass, not correctness. How many slots a given caller
        # needs is set by `SGLANG_BARLINK_BAR1_PIPE_RESULT_EAGER`.
        if self._result_eager_slots < 2:
            return None
        busy = [
            v is not None and v() is not None for v in self._result_alive
        ]
        i = result_eager_free_slot(
            self._result_i, self._result_eager_slots, busy
        )
        if i is None:
            self._result_eager_full += 1
            if not self._result_eager_full_reported:
                self._result_eager_full_reported = True
                logger.warning(
                    "barlink-BAR1-PIPE: all %d eager result slots are still "
                    "being held by the caller. This call runs the "
                    "direct=0 path -- correct, but without the saved VRAM "
                    "pass. The caller is keeping more results alive at the "
                    "same time than the ring's eager part has slots; more "
                    "are available with SGLANG_BARLINK_BAR1_PIPE_RESULT_EAGER "
                    "(and a SGLANG_BARLINK_BAR1_PIPE_RESULT_RING of at least "
                    "that size). Each slot costs %d bytes in the BAR1 "
                    "window. This notice appears once per rank; how often "
                    "it actually happens is recorded in _result_eager_full.",
                    self._result_eager_slots,
                    int(self._geo["result_stride"]),
                )
            return None
        ptr = (self._own[0] + int(self._geo["off_result"])
               + i * int(self._geo["result_stride"]))
        out = self._pipe_ext.bar1_result_tensor(int(ptr), inp)
        # The handshake only runs along in eager mode if graph-safe mode is
        # on. Without it, the kernel stays byte-for-byte the measured one --
        # flag family 4 is then never touched. With it, the slack is the
        # ACTUAL reuse distance of this slot: under strict rotation, the
        # number of eager slots as before, fewer after a skipped slot. A
        # slack that is too large would be the weaker wait condition, i.e.
        # the dangerous direction.
        slack = (
            result_eager_slack(i, self._result_counter, self._result_last,
                            self._result_eager_slots)
            if self.pipe_direct_graph else 0
        )
        self._result_alive[i] = weakref.ref(out)
        # The number recorded is that of THIS call, not the next one: the
        # distance to the next use of the same slot is the difference of
        # two call numbers, and under strict rotation over L slots that is
        # thus exactly L.
        self._result_last[i] = self._result_counter
        self._result_counter += 1
        self._result_i = i
        return out, i, slack

    def _pipe_all_reduce(self, inp, k: int):
        """One call of the pipelined kernel. Out of place, like mesh/ring."""
        import torch

        inp = inp.contiguous()
        nbytes = inp.numel() * inp.element_size()
        reserved = self._result_slot(inp)
        direct = reserved is not None
        if direct:
            out, result_slot, result_slack = reserved
        else:
            out, result_slot, result_slack = torch.empty_like(inp), -1, 0
        kernel_variant = self._kernel(nbytes, self.pipe_grid_from, "mesh_pipe")
        peer_payload = [0] * self.world
        peer_flag = [0] * self.world
        peer_result = [0] * self.world
        for r, z in self._peers.items():
            peer_payload[r] = z.payload.dev_ptr
            peer_flag[r] = z.flag.dev_ptr
        peer_payload[self.rank] = self._own[0]
        peer_flag[self.rank] = self._own_flag[0]
        if direct:
            # THE SAME ring slot on every rank. This is not an assumption
            # about the neighbor but the same SPMD precondition every
            # collective in this module rests on: all ranks see the same
            # sequence of calls. The kernel additionally checks that its
            # own entry really is `out`.
            offset = (int(self._geo["off_result"])
                       + result_slot * int(self._geo["result_stride"]))
            for r in range(self.world):
                peer_result[r] = peer_payload[r] + offset
        from sglang.srt.distributed.device_communicators.barlink_bar1_pipe_ext import (
            pipe_fbase,
        )

        self._note_launch("all_reduce/mesh_pipe", nbytes, kernel_variant)
        self._pipe_ext.bar1_mesh_pipe(
            inp, out, int(self.rank), int(self.world),
            peer_payload, peer_flag, peer_result,
            int(self._own[0]), int(self._own_flag[0]),
            int(self.pipe_slot),
            int(self._geo["off_pipe"]),
            int(pipe_fbase(self.world, self.a2a_on)),
            int(k), int(self.pipe_t), int(self.pipe_lead),
            int(self.pipe_ack), 1 if direct else 0, int(result_slack),
            self._round_dev, self._step_dev, self._result_gen_dev,
            self._ctl_dev,
            self._deadline_cycles(), int(self.threads), int(kernel_variant),
            int(self.load_shape),
            int(self._abort_host),
        )
        return out

    def byte_proof_pipe(self, rounds: int = 0) -> bool:
        """Byte-level proof for ``mesh_pipe``. Without it, the path withdraws.

        Separate from ``byte_proof_all``, because it checks something
        different: the pair proof shows that bytes arrive; this one shows
        that slot reuse holds up across multiple rounds. The second point
        is the more dangerous one, and it does not show up in a single
        round.
        """
        from sglang.srt.distributed.device_communicators import (
            barlink_bar1_pipe_ext,
        )

        if not self.pipe_on or self._pipe_ext is None:
            self._pipe_proof = False
            return False
        # Provisionally let it through so the proof itself can run; the
        # final answer is below. Nobody asks `handles` in the meantime --
        # the proof runs during setup, before the first collective.
        self._pipe_proof = True
        try:
            ok = barlink_bar1_pipe_ext.byte_proof_pipe(self, rounds)
        except Exception as e:
            logger.warning(
                "barlink-BAR1-PIPE: the byte-level proof aborted with %r. "
                "mesh_pipe withdraws; mesh and ring are unaffected.",
                e,
            )
            ok = False
        self._pipe_proof = bool(ok)
        if not ok:
            logger.warning(
                "barlink-BAR1-PIPE: byte-level proof not passed -- mesh_pipe "
                "withdraws via handles()."
            )
        return self._pipe_proof

    # -- all_gather --------------------------------------------------------
    #
    # THE STOPPER. Before this change, BARLINK_OPS did not cover all_gather,
    # and the standard run aborted:
    #
    #     RuntimeError: barlink: 'all_gather' with 10600448 bytes during a
    #     CUDA graph capture, but bar1 reports handles(...) -> False.
    #
    # Correctly aborted -- under barlink, PyNccl is not built, the fallback
    # path would be the host-staged gloo tier, and that runs once during a
    # capture and never again on replay. It just did not work.
    #
    # WHY NO NEW KERNEL. An all_gather is the AG phase of the mesh allreduce
    # without the reduction, and the a2a kernel can already do exactly that:
    # it moves bytes, knows no data type, and receives offsets and lengths
    # SEPARATELY PER RANK. An all_gather is an all_to_all in which every
    # destination gets the same slice -- i.e. the same table with
    # ``send_offsets[z] = const``. This is not a trick but the promise
    # already spelled out in barlink_bar1_ext.py ("it never assumed they were
    # contiguous").
    #
    # What comes along for free, without building it: the byte-level proof
    # per directed pair (``byte_proof_a2a``), the half-selection by round
    # parity (which makes slot reuse safe without a third barrier), the
    # local path for one's own block, the tail path for lengths that are
    # not a multiple of 16, and the extension's bounds checks. A second
    # kernel would have needed every one of these pieces a second time --
    # and each would be a place where the two versions could drift apart.
    #
    # WHAT THIS COSTS, honestly: one intermediate slot. The receiver reads
    # from its slot into the output buffer, instead of the sender writing
    # directly into the output buffer. Without a reduction, the latter
    # would be possible -- but only with a mapped result buffer, i.e. with
    # the pipe's result ring. Since the graph-safe direct mode
    # (``_result_slot``: a reserved ring slot per captured call site,
    # release handshake in the kernel), this is no longer a hard exclusion
    # but an open extension -- the a2a kernel does not know the handshake,
    # and an all_gather needs its own result ring sized for the FULL
    # result, not the shard. Until then: slot instead of direct.

    def _handles_all_gather(self, nbytes: int) -> bool:
        """``nbytes`` is the CALLER'S OWN shard, not the result.

        The seam asks with ``input_.numel() * element_size()``
        (``barlink.BarlinkCommunicator.all_gather``), i.e. with the shard. The
        result is ``R`` times as large and is NOT checked here: it lives
        in local VRAM, not in the window.

        Every condition is rank-uniform, for the same reason as in
        :meth:`handles`.

        Beyond the slot size, this is NOT rejected but decomposed into
        rounds (:func:`ag_plan`) -- a rejection would be an abort with no
        fallback under capture, and that is exactly what the stopper hung
        on. Only what does not work even in rounds is rejected.

        ``nbytes % 16 != 0`` is explicitly NOT rejected, unlike with
        all_reduce. The a2a kernel has the tail path for that (``VEC=0``,
        packet assembled byte by byte): correct, slower, unmeasured. That
        is the right choice, because under capture the alternative is not
        a slower path but no path at all.

        **What a crooked shard costs, precisely stated:** not the last 15
        bytes, but everything. Rank ``i``'s result offset is ``i *
        shard``; if ``shard`` is not a multiple of 16, every offset except
        rank 0's is off-alignment, and the extension switches to
        ``VEC=0`` for the WHOLE call (it checks all offsets jointly,
        barlink_bar1_ext.py: "there is no such thing as 'mostly aligned'").
        Whoever sees a slow number should check this first before
        attributing it to the transport.
        """
        if not self.ag_on:
            return False
        # Same region, same kernel, same byte-level proof. Without the a2a
        # region (SGLANG_BARLINK_BAR1_A2A=0), there is also no all_gather --
        # stated, not silently assumed.
        if not self.a2a_on or not self._a2a_proof:
            return False
        geo = self._geo
        if geo.get("off_a2a", -1) < 0 or geo.get("a2a_slot", 0) <= 0:
            return False
        if nbytes <= 0:
            return False
        if nbytes < self.ag_min_bytes:
            return False
        # The same window concept as with all_reduce and a2a: against the
        # group-wide SMALLEST actually mapped length.
        if geo["region_bytes"] > self._window_minimum:
            return False
        # There is still a ceiling, though, and it is not a window limit
        # but a round limit: every round is one kernel launch with one
        # barrier, and arbitrarily many of those per collective would not
        # be a transport but a loop. Rank-uniform, because nbytes is.
        if -(-nbytes // int(geo["a2a_slot"])) > self.ag_max_rounds:
            return False
        return True

    def ag_rounds(self, nbytes: int) -> int:
        """Round count for a shard of ``nbytes`` -- for logging/tests."""
        slot = int(self._geo.get("a2a_slot", 0))
        if slot <= 0:
            return 0
        return max(1, -(-int(nbytes) // slot))

    def barlink_all_gather(self, comm, inp, dim: int = -1):
        """``all_gather`` over the direct path, in several rounds if needed.

        Result shape and axis handling are byte-for-byte those of the seam
        (``barlink.BarlinkCommunicator.all_gather``) and of
        ``barlink_device.all_gather``: first ``(R,) + shape``, then
        ``movedim(0, dim)``, then merge. Not reinvented -- the same
        expression, so a transport switch changes nothing about the
        numbers.
        """
        import torch

        if not self._up or self._ext is None or not self.a2a_on:
            raise Bar1Unavailable(
                "barlink_all_gather without an a2a region set up -- reachable "
                "only if someone bypassed handles()."
            )
        if dim < 0:
            dim += inp.dim()
        inp = inp.contiguous()
        shape = tuple(inp.size())
        shard = inp.numel() * inp.element_size()
        out = torch.empty((self.world,) + shape, dtype=inp.dtype,
                          device=inp.device)
        # Uniform, because the seam is -- but passed as a VECTOR to
        # ag_plan, not as an assumption in the arithmetic. The rationale is
        # in ag_plan.
        plan = ag_plan([shard] * self.world, int(self._geo["a2a_slot"]))
        flat = out.view(-1)
        for round in plan:
            s_off = [round[self.rank][0]] * self.world
            s_len = [round[self.rank][1]] * self.world
            e_off = [x[2] for x in round]
            e_len = [x[1] for x in round]
            self.barlink_all_to_all_single(
                comm, flat, inp, s_len, e_len, s_off, e_off,
                op_label="all_gather",
            )
        out = out.movedim(0, dim)
        return out.reshape(shape[:dim] + (self.world * shape[dim],) + shape[dim + 1:])

    # -- broadcast ---------------------------------------------------------
    #
    # THE NEXT STOPPER, and it sat one level below the all_gather one. The
    # standard run aborted while capturing the DRAFT graph
    # (eagle_worker_v2.init_cuda_graphs -> parallel_state broadcast ->
    # barlink._select):
    #
    #     RuntimeError: barlink: 'broadcast' with 128 bytes during a CUDA
    #     graph capture, but bar1 reports handles('broadcast', 128) ->
    #     False.
    #
    # 128 bytes. It was not bandwidth that was missing but coverage -- and
    # under capture, missing coverage is not a slower path but the abort:
    # the fallback path (gloo, host-staged) would run ONCE during capture
    # and never again on replay.
    #
    # AGAIN NO NEW KERNEL, and this time the table is even simpler than for
    # all_gather. A broadcast is an all_to_all in which exactly ONE rank
    # fills a row: the source sets the same slice for every destination
    # (``send_offsets[z] = const``, ``send_bytes[z] = n``), and every
    # other rank sends 0 bytes everywhere. On the receive side, exactly one
    # block is non-zero, namely the one from ``src``. The kernel already
    # knows this: it receives offsets and lengths separately per rank, and a
    # length of 0 is not a special case for it, just an empty loop.
    #
    # THE ONE CONDITION THAT IS STRICTER HERE THAN FOR ALL_GATHER: the
    # round count must be the same on ALL ranks, even though only one
    # sends. It is, because it falls out solely from ``nbytes`` and the
    # slot (see bc_plan) -- not from how much this particular rank has to
    # do. A non-source rank that shortened "I'm sending nothing" to "so no
    # round" would leave the source waiting in the barrier.
    #
    # WHAT THIS COSTS, honestly: ONE intermediate buffer and ONE local
    # copy. The seam is IN-PLACE (``broadcast(tensor, src)`` returns the
    # same tensor), but the extension rejects ``in is out`` -- rightly so:
    # the kernel's send and receive phases run around ONE barrier, and a
    # buffer that is both would already be overwritten in the second phase.
    # The intermediate buffer costs n bytes of VRAM and one on-device copy
    # at HBM bandwidth, while the collective itself goes over PCIe; at the
    # sizes on this path (128 bytes), that is not measurable.
    # `torch.empty_like` under capture comes from the graph's private pool
    # and sits at the same address on every replay -- the same rationale
    # barlink_device rests on.

    def _handles_broadcast(self, nbytes: int) -> bool:
        """``nbytes`` is the full payload -- it is the same group-wide.

        The seam asks with ``tensor.numel() * element_size()``
        (``barlink.BarlinkCommunicator.broadcast``). Unlike with all_gather,
        there is no difference here between shard and result: every rank
        ends up holding exactly these bytes.

        Every condition is rank-uniform, for the same reason as in
        :meth:`handles` -- and here that stands out especially, because the
        operation itself is not: only ``src`` sends. Precisely because of
        that, nothing in this answer may depend on ``self.rank``.

        Beyond the slot size, this is NOT rejected but decomposed into
        rounds (:func:`bc_plan`). ``nbytes % 16 != 0`` is likewise not
        rejected -- the a2a kernel has the tail path. Both for the same
        reason as in :meth:`_handles_all_gather`: under capture, a decline
        is an abort.

        **The covered range is ``1 .. a2a_slot * bc_max_rounds``**, with
        no gaps. This is not a rewrite but the lesson from the second
        attempt: the first covered broadcast but rejected 12 bytes (lower
        bound 16, copied from a2a), and the standard run sends exactly
        those 12 bytes. Silently answering ``False`` for a size in this
        range means, under capture, aborting the run -- if a threshold is
        ever reintroduced here, it needs the same rationale
        ``_no_collective`` also states for reduce_scatter.
        """
        if not self.bc_on:
            return False
        # Same region, same kernel -- but an OWN byte-level proof
        # (`_bc_proof`), because the table is a different one. Without the
        # a2a region there is also no broadcast; stated, not silently
        # assumed.
        if not self.a2a_on or not self._a2a_proof or not self._bc_proof:
            return False
        geo = self._geo
        if geo.get("off_a2a", -1) < 0 or geo.get("a2a_slot", 0) <= 0:
            return False
        if nbytes <= 0:
            return False
        if nbytes < self.bc_min_bytes:
            return False
        # The same window concept as everywhere: against the group-wide
        # SMALLEST actually mapped length.
        if geo["region_bytes"] > self._window_minimum:
            return False
        if -(-nbytes // int(geo["a2a_slot"])) > self.bc_max_rounds:
            return False
        return True

    def bc_rounds(self, nbytes: int) -> int:
        """Round count for ``nbytes`` -- for logging/tests."""
        slot = int(self._geo.get("a2a_slot", 0))
        if slot <= 0:
            return 0
        return max(1, -(-int(nbytes) // slot))

    def barlink_broadcast(self, comm, tensor, src: int = 0):
        """``broadcast`` over the direct path, in several rounds if needed.

        IN-PLACE and with the same return value as every other version of
        this seam (``barlink.BarlinkCommunicator.broadcast``,
        ``barlink_device.barlink_broadcast``): the tensor handed in is filled
        and returned. ``src`` is a GROUP-LOCAL rank -- that is how the seam
        asks, and it is also what ``self.rank`` means.
        """
        import torch

        if not self._up or self._ext is None or not self.a2a_on:
            raise Bar1Unavailable(
                "barlink_broadcast without an a2a region set up -- reachable "
                "only if someone bypassed handles()."
            )
        R = self.world
        if not 0 <= int(src) < R:
            raise Bar1Unavailable(
                f"broadcast with src={src} at {R} ranks -- the rank is to "
                f"be understood as group-local."
            )
        src = int(src)
        # `reshape(-1)` instead of `view(-1)`: a non-contiguous tensor gets
        # a contiguous copy here as the SOURCE, and the return below writes
        # the result back into the original storage with `copy_`. Both are
        # needed because the seam promises in-place, but the kernel
        # requires contiguous buffers.
        source = tensor.reshape(-1)
        nbytes = source.numel() * source.element_size()
        if nbytes == 0:
            return tensor
        dst = torch.empty_like(source)
        plan = bc_plan(nbytes, int(self._geo["a2a_slot"]))
        # Group-wide identical, because `plan` is -- and that is why the
        # kernel variant (grid/1blk) also comes out the same on every rank,
        # instead of depending on the rank-dependent question "how much do
        # I send".
        for offset, length in plan:
            sends = (self.rank == src)
            s_len = [length if sends else 0] * R
            s_off = [offset] * R
            e_len = [length if i == src else 0 for i in range(R)]
            e_off = [offset if i == src else 0 for i in range(R)]
            self.barlink_all_to_all_single(
                comm, dst, source, s_len, e_len, s_off, e_off,
                kernel_bytes=length * (R - 1),
                op_label="broadcast",
            )
        tensor.copy_(dst.view(tensor.shape))
        return tensor

    def byte_proof_broadcast(self) -> bool:
        """Byte-level proof for ``broadcast``: each source once, over rounds.

        Its own proof, even though the same kernel runs. What needs proving
        here is not the kernel -- that is covered by ``byte_proof_a2a`` --
        but the TABLE: exactly one sender, all other lengths 0, and a
        receive block whose offset is not the prefix sum of the lengths. A
        swapped row does not show up in an all_to_all with uniformly sized
        blocks; it does here.

        What runs is
        (a) per rank, ONE pass with this rank as the source -- so every
            directed edge ``src -> r`` has run once, the same coverage
            ``byte_proof_a2a`` has over the pairs --, and
        (b) one pass over MULTIPLE rounds (payload > slot), because the
            half-selection by round parity never even gets exercised in a
            single round.

        **Read back over a different path than it was written.** The
        counterpart card's BAR1 aperture did the writing; the check runs on
        the CPU, after an ordinary device-to-host copy. And the EXPECTED
        VALUE does not come from the transport: it is computed locally
        from the source rank number (``_a2a_marker(src, src)``). A rank
        thus never learns the expected bytes through the path currently
        under test -- otherwise the proof would be proving itself.

        If it fails, **only** ``broadcast`` withdraws.
        """
        import torch
        import torch.distributed as dist

        self._bc_proof = False
        if not self.bc_on:
            logger.info(
                "barlink-BAR1: broadcast is disabled via SGLANG_BARLINK_BAR1_BC=0 "
                "-- no byte-level proof, handles() says False."
            )
            return False
        if not self.a2a_on or not self._a2a_proof:
            return False
        if not self._up or self._ext is None:
            return False

        R, r = self.world, self.rank
        slot = int(self._geo.get("a2a_slot", 0))
        if slot <= 0:
            return False
        small = min(4096, slot)
        # Beyond the slot size, but not by a hair: 16 leftover bytes in the
        # second round additionally hit the kernel's tail path.
        large = slot + 16
        # BELOW one packet. This is the case the first attempt walked right
        # past: the lower bound stood at 16, the standard run sends 12, and
        # the proof only ran sizes the bound accepted anyway -- so it could
        # never have seen the bug. 12 bytes exercise the one path no other
        # size here does: a single, incomplete packet, assembled in a
        # register and read back out byte by byte.
        tiny = 12

        ok_local = True
        for src, n in ([(s, small) for s in range(R)]
                       + [(s, tiny) for s in range(R)]
                       + [(0, large)]):
            # Every rank starts with ITS OWN marker. For the source, that is
            # already the expected value; for everyone else, a byte
            # distinguishable from it -- a broadcast that moves nothing at
            # all thus stands out instead of accidentally looking correct.
            expected = self._a2a_marker(src, src)
            buffer = torch.full((n,), self._a2a_marker(r, r), dtype=torch.uint8,
                                device=self.device)
            bounded_barrier(
                self.cpu_group,
                f"bar1 broadcast proof src={src} n={n}: before the round",
                table=self._peer_table,
            )
            ran = True
            try:
                self.barlink_broadcast(None, buffer, src)
                bounded_device_sync(
                    f"bar1 broadcast proof src={src} n={n}",
                    device=self.device,
                    table=self._peer_table,
                )
                # A tripped kernel is otherwise silent: the stream carries on
                # over a half-written buffer, and the byte comparison below
                # would report a data bug where the cause was an abort.
                self.raise_if_aborted(f"broadcast proof src={src} n={n}")
            except Exception as ex:            # noqa: BLE001 -- reason goes into the log
                logger.warning(
                    "barlink-BAR1: broadcast byte-level proof (src=%d, %d "
                    "bytes) could not run: %r", src, n, ex,
                )
                ok_local = False
                ran = False
            if not ran:
                continue
            back = buffer.cpu()
            bad = int((back != expected).sum().item())
            if bad:
                ok_local = False
                logger.warning(
                    "barlink-BAR1: broadcast byte-level proof %d->%d FAILED: "
                    "%d of %d bytes wrong (%d rounds). broadcast withdraws.",
                    src, r, bad, n, len(bc_plan(n, slot)),
                )
            else:
                logger.info(
                    "barlink-BAR1: broadcast byte-level proof %d->%d passed: "
                    "0 of %d bytes wrong (%d rounds).",
                    src, r, n, len(bc_plan(n, slot)),
                )

        carrier: list[object] = [None] * R
        check_peers("bar1 broadcast proof: verdict exchange", self._peer_table)
        dist.all_gather_object(carrier, bool(ok_local), group=self.cpu_group)
        self._bc_proof = all(bool(x) for x in carrier)
        if not self._bc_proof:
            logger.warning(
                "barlink-BAR1: broadcast byte-level proof failed group-wide "
                "(ranks %s). handles('broadcast') returns False; "
                "all_reduce, all_to_all, and all_gather are unaffected.",
                [i for i, x in enumerate(carrier) if not x],
            )
        return self._bc_proof

    # -- all_to_all --------------------------------------------------------

    def _handles_a2a(self, nbytes: int) -> bool:
        """The COARSE answer of the seam for ``all_to_all``.

        It only knows the total size, not the individual block sizes -- so
        it checks the uniform case. For unequal block sizes (the normal
        case for MoE), :meth:`supports_a2a` decides once the actual counts
        are known. Both answers are rank-uniform: this one depends only on
        group-wide reconciled sizes, that one on a value the caller passes
        in after maximizing it group-wide.
        """
        if not self.a2a_on or not self._a2a_proof:
            return False
        geo = self._geo
        if geo.get("off_a2a", -1) < 0 or geo.get("a2a_slot", 0) <= 0:
            return False
        if nbytes < self.a2a_min_bytes:
            return False
        # Beyond the slot size, this is NOT rejected but decomposed into
        # rounds -- the same answer as for all_reduce, all_gather, and
        # broadcast. The coarse number here is the uniform case; the exact
        # round count falls out in `supports_a2a`, once the group-wide
        # largest block is known.
        if a2a_rounds(-(-nbytes // self.world),
                      int(geo["a2a_slot"])) > self.a2a_max_rounds:
            return False
        # The same window concept as with all_reduce: against the
        # group-wide SMALLEST actually mapped length, not against the
        # requested one. Redundant as long as setup completed
        # successfully -- and exactly for that reason cheap.
        if geo["region_bytes"] > self._window_minimum:
            return False
        return True

    def a2a_slot_bytes(self) -> int:
        """Largest block ONE directed pair can carry. 0 = no a2a."""
        if not self.a2a_on or not self._a2a_proof:
            return 0
        return int(self._geo.get("a2a_slot", 0))

    def supports_a2a(self, largest_block: int) -> bool:
        """Does the largest block over ALL pairs fit into a slot?

        ``largest_block`` must be a **group-wide identical** value -- the
        maximum over all R*R blocks, not over the caller's own row. If
        every rank computed it only from its own block sizes, one rank
        could run into the collective and another into the fallback, and
        the result would be a hang instead of an error. The caller
        (``BarlinkCommunicator.all_to_all_single``) maximizes over the group
        beforehand; that is exactly why this check does not live in
        ``handles``.
        """
        if not self.a2a_on or not self._a2a_proof or not self._up:
            return False
        if largest_block < 0:
            return False
        slot = int(self._geo.get("a2a_slot", 0))
        if slot <= 0:
            return False
        # If it does not fit into ONE slot, it runs in several rounds --
        # only what does not work even in rounds is rejected.
        return a2a_rounds(largest_block, slot) <= self.a2a_max_rounds

    def a2a_rounds_for(self, largest_block: int) -> int:
        """Round count the caller passes to ``barlink_all_to_all_single``.

        It falls out of the GROUP-WIDE largest block, which the seam has
        already computed anyway -- not out of the caller's own row. From
        its own row it would be rank-dependent, and a rank running one
        round fewer would leave the others waiting in the barrier.
        """
        slot = int(self._geo.get("a2a_slot", 0))
        if slot <= 0:
            return 0
        return a2a_rounds(largest_block, slot)

    def barlink_all_to_all_single(self, comm, output, inp,
                                send_bytes, recv_bytes,
                                send_offsets=None, recv_offsets=None,
                                kernel_bytes=None, rounds=None,
                                op_label="all_to_all"):
        """Wrapper with a round loop. One step or several, depending on the block.

        ``rounds`` comes from the caller and is GROUP-WIDE identical -- the
        seam computes it from the largest block over all pairs
        (:meth:`a2a_rounds_for`). ``None`` means one round and is thus
        byte-for-byte the previous path; that is exactly how
        :meth:`barlink_all_gather` and :meth:`barlink_broadcast` call it, since
        they have already sliced their own segments themselves.

        What one round moves: at most one slot's worth out of every block,
        in one piece, starting at offset ``k*slot``. Blocks that finish
        earlier carry length 0 -- they ride along in the barrier without
        moving any bytes. Same pattern as in ag_plan, and for the same
        reason: the round count must not depend on how much THIS rank has
        to do.
        """
        n = 1 if rounds is None else max(1, int(rounds))
        if n == 1:
            return self._a2a_one_round(
                comm, output, inp, send_bytes, recv_bytes,
                send_offsets, recv_offsets, kernel_bytes,
                op_label=op_label,
            )
        slot = int(self._geo["a2a_slot"])
        s_base = list(send_offsets) if send_offsets is not None else None
        e_base = list(recv_offsets) if recv_offsets is not None else None
        if s_base is None:
            s_base, acc = [], 0
            for length in send_bytes:
                s_base.append(acc)
                acc += int(length)
        if e_base is None:
            e_base, acc = [], 0
            for length in recv_bytes:
                e_base.append(acc)
                acc += int(length)
        for k in range(n):
            s_len = [
                min(slot, max(0, int(length) - k * slot))
                for length in send_bytes
            ]
            e_len = [
                min(slot, max(0, int(length) - k * slot))
                for length in recv_bytes
            ]
            self._a2a_one_round(
                comm, output, inp, s_len, e_len,
                [b + k * slot for b in s_base],
                [b + k * slot for b in e_base],
                kernel_bytes,
                op_label=op_label,
            )
        return output

    def _a2a_one_round(self, comm, output, inp,
                        send_bytes, recv_bytes,
                        send_offsets=None, recv_offsets=None,
                        kernel_bytes=None, op_label="all_to_all"):
        """``all_to_all_single`` over the direct path. One step, one barrier.

        ``send_bytes[j]`` is the block going to rank ``j``,
        ``recv_bytes[i]`` the one coming from rank ``i`` -- both in
        **bytes**, not in rows and not in elements. The kernel moves bytes:
        there is no reduction, hence no data type. fp8, bf16, int32, uint8
        take the same path, and the sm_86 cards' missing fp8 conversion
        instructions are irrelevant here.

        ``send_offsets`` / ``recv_offsets`` are **optional** and in
        bytes. ``None`` means: the prefix sum of the lengths, i.e. the
        gap-free concatenation that ``torch.distributed.
        all_to_all_single`` means -- byte-for-byte the previous path. They
        are supplied by a caller who wants to move only a piece per block
        out of ONE existing buffer without copying it over first: that is
        exactly what the MoE dispatcher needs when a block is larger than
        the slot and therefore spans several rounds. The kernel itself
        already knows both cases -- it receives offsets and lengths
        separately and has never assumed they were contiguous; only this
        seam used to compute the offsets itself.

        ``kernel_bytes`` is the byte count that decides the kernel variant
        (grid/1blk). ``None`` means: what this rank actually sends over the
        aperture -- correct for any table in which all ranks move similar
        amounts. A broadcast is precisely not that: there, ONE rank sends
        everything and all others send nothing, and a choice computed from
        the caller's own row would come out differently per rank.
        Correctness does not depend on this (both variants run the same
        flag protocol), but comparability of measurements and the
        rank-uniformity of the capture do. The caller therefore passes in a
        group-wide identical value.

        The caller is responsible for ensuring that ``recv_bytes[i]``
        on this rank equals ``send_bytes[rank]`` on rank ``i``. The
        extension checks what it can check locally (buffer bounds, slot
        bound, its own block), but not agreement across the group -- that
        would require running a collective, and that would be exactly the
        host sync this path avoids.
        """
        if not self._up or self._ext is None or not self.a2a_on:
            raise Bar1Unavailable(
                "barlink_all_to_all_single without an a2a region set up -- "
                "reachable only if someone bypassed handles()."
            )
        R = self.world
        if len(send_bytes) != R or len(recv_bytes) != R:
            raise Bar1Unavailable(
                f"block sizes have length {len(send_bytes)}/"
                f"{len(recv_bytes)}, {R} expected."
            )
        inp = inp.contiguous()
        if not output.is_contiguous():
            raise Bar1Unavailable("output buffer is not contiguous")

        if send_offsets is None:
            send_off, s = [], 0
            for n in send_bytes:
                send_off.append(s)
                s += int(n)
        else:
            if len(send_offsets) != R:
                raise Bar1Unavailable(
                    f"send_offsets has length {len(send_offsets)}, "
                    f"{R} expected."
                )
            send_off = [int(x) for x in send_offsets]
        if recv_offsets is None:
            recv_off, e = [], 0
            for n in recv_bytes:
                recv_off.append(e)
                e += int(n)
        else:
            if len(recv_offsets) != R:
                raise Bar1Unavailable(
                    f"recv_offsets has length {len(recv_offsets)}, "
                    f"{R} expected."
                )
            recv_off = [int(x) for x in recv_offsets]

        # The cooperative multi-block launch pays off at the same threshold
        # as with all_reduce -- it is measured THERE and only there; here it
        # is carried over, not confirmed. What governs is what actually
        # goes over PCIe, i.e. excluding one's own block.
        if kernel_bytes is None:
            moved = sum(
                int(n) for j, n in enumerate(send_bytes) if j != self.rank
            )
        else:
            moved = int(kernel_bytes)
        kernel_variant = self._kernel(moved, self.grid_from, "all_to_all_single")

        peer_payload = [0] * R
        peer_flag = [0] * R
        for rr, z in self._peers.items():
            peer_payload[rr] = z.payload.dev_ptr
            peer_flag[rr] = z.flag.dev_ptr
        peer_payload[self.rank] = self._own[0]
        peer_flag[self.rank] = self._own_flag[0]

        # #583: the label names the collective the CALLER asked for, not the
        # kernel that serves it. ``barlink_broadcast`` and
        # ``barlink_all_gather`` both run this a2a kernel, so recording a bare
        # "all_to_all" made an 8-byte broadcast be reported as an 8-byte
        # all_to_all -- a collective no seam in the decode loop issues, which
        # is why the 2026-08-06 crash triage went looking for one.
        self._note_launch(op_label, int(moved), kernel_variant)
        self._ext.bar1_all_to_all(
            inp, output, int(self.rank), int(R),
            [int(x) for x in send_off], [int(x) for x in send_bytes],
            [int(x) for x in recv_off], [int(x) for x in recv_bytes],
            peer_payload, peer_flag,
            int(self._own[0]), int(self._own_flag[0]),
            int(self._geo["a2a_slot"]), int(self._geo["off_a2a"]),
            int(fbase_a2a(R)),
            self._round_dev, self._ctl_dev,
            self._deadline_cycles(), int(self.threads), int(kernel_variant),
            int(self.load_shape),
            int(self._abort_host),
            # #622: appended, not inserted -- see _all_reduce_one_round.
            int(self._ackbase_a2a),
        )
        return output

    @staticmethod
    def _a2a_marker(source: int, dst: int) -> int:
        """A byte that differs per directed pair, never 0x00 and never 0xFF.

        ``0x40 | (source*8 + dst)`` -- for R <= 8, ``source*8+dst`` is
        injective and fits in 6 bits. 0xFF is the output buffer's
        pre-fill value, 0x00 the receive slot's; both are thus
        distinguishable from the pattern, and a block that was NOT written
        stands out as such instead of accidentally looking like a hit.
        """
        return 0x40 | ((source * 8 + dst) & 0x3F)

    def byte_proof_a2a(self) -> bool:
        """Byte-level proof for ``all_to_all``: every byte, every directed pair.

        Two passes over the REAL kernel -- not over ``put``, because what
        needs proving is the path the collectives take, including slot
        selection, half selection, and the barrier:

        1. **uniform** -- every block the same size, all offsets
           16-byte-aligned. This is the kernel's vector path.
        2. **skewed and crooked** -- block lengths
           ``block*(1+((q+z)%3)) + ((q*5+z*3)%7)``. The factor makes the
           block sizes unequal (the MoE normal case), the addend turns
           them into non-multiples of 16 and thereby pushes every
           subsequent offset out of alignment -- exactly the tail path
           that would otherwise never run, and where the last bytes of a
           block live.

        Checked on the RECEIVING card against the local output buffer,
        byte by byte, separately for each sender -- including one's own
        block, which does not even go through the aperture (otherwise a
        swapped offset would not stand out).

        If the proof fails, **only** ``all_to_all`` withdraws.
        ``all_reduce`` uses different slots, different flag lines, and a
        measured kernel; disabling it too would be a conclusion this
        probe does not support.
        """
        import torch.distributed as dist

        self._a2a_proof = False
        if not self.a2a_on:
            logger.info(
                "barlink-BAR1: all_to_all is disabled via "
                "SGLANG_BARLINK_BAR1_A2A=0 -- no byte-level proof, handles() "
                "says False."
            )
            return False
        if not self._up or self._ext is None:
            return False

        R = self.world
        slot = int(self._geo.get("a2a_slot", 0))
        # The largest block of the skewed pass is 3*block+6.
        block = min(8192, (slot - 6) // 3)
        block = (block // 16) * 16
        if block <= 0:
            logger.warning(
                "barlink-BAR1: a2a slot of %d bytes is too small for the "
                "byte-level proof. all_to_all withdraws.", slot,
            )
            return False

        # From here on, the group-wide reconciliation runs in EVERY case. A
        # rank that bails out before the all_gather_object because of a
        # local exception would leave the others waiting in it -- a failed
        # proof would turn into a hang.
        ok_local = True
        try:
            ok_local = self._a2a_proof_passes(block)
        except Exception as ex:                # noqa: BLE001
            ok_local = False
            logger.warning("barlink-BAR1: a2a byte-level proof aborted: %r", ex)

        carrier: list[object] = [None] * R
        check_peers("bar1 a2a proof: verdict exchange", self._peer_table)
        dist.all_gather_object(carrier, bool(ok_local), group=self.cpu_group)
        self._a2a_proof = all(bool(x) for x in carrier)
        if not self._a2a_proof:
            logger.warning(
                "barlink-BAR1: a2a byte-level proof failed group-wide (ranks "
                "%s). handles('all_to_all') returns False; all_reduce is "
                "unaffected.",
                [i for i, x in enumerate(carrier) if not x],
            )
        return self._a2a_proof

    def _a2a_proof_passes(self, block: int) -> bool:
        """The two probe passes. Purely local, without group reconciliation.

        **Both passes always run**, even if the first one failed. Each
        contains a barrier; a rank that cuts short after a failure would
        leave the others waiting in the next barrier. A failed proof would
        turn into a hang -- and a hang does not say what is broken. The
        barrier is bounded, so that hang is now decidable rather than
        merely documented as a risk.
        """
        import torch

        R, r = self.world, self.rank
        ok_local = True
        for skewed in (False, True):

            def length(q: int, z: int) -> int:
                if not skewed:
                    return block
                return block * (1 + ((q + z) % 3)) + ((q * 5 + z * 3) % 7)

            send = [length(r, z) for z in range(R)]
            recv = [length(q, r) for q in range(R)]
            inp = torch.empty(sum(send), dtype=torch.uint8, device=self.device)
            o = 0
            for z in range(R):
                inp[o:o + send[z]] = self._a2a_marker(r, z)
                o += send[z]
            out = torch.full((sum(recv),), 0xFF, dtype=torch.uint8,
                             device=self.device)
            run_name = "skewed" if skewed else "uniform"
            bounded_barrier(
                self.cpu_group,
                f"bar1 a2a proof ({run_name}): before the round",
                table=self._peer_table,
            )
            ran = True
            try:
                self.barlink_all_to_all_single(None, out, inp, send, recv)
                bounded_device_sync(
                    f"bar1 a2a proof ({run_name})",
                    device=self.device,
                    table=self._peer_table,
                )
                # Same reason as in the broadcast proof: without this, an
                # aborted kernel is reported as a byte mismatch.
                self.raise_if_aborted(f"a2a proof ({run_name})")
            except Exception as ex:            # noqa: BLE001 -- reason goes into the log
                logger.warning(
                    "barlink-BAR1: a2a byte-level proof (%s) could not run: %r",
                    "skewed" if skewed else "uniform", ex,
                )
                ok_local = False
                ran = False
            if not ran:
                continue
            back = out.cpu()
            o = 0
            bad_total = 0
            for q in range(R):
                expected = self._a2a_marker(q, r)
                piece = back[o:o + recv[q]]
                bad = int((piece != expected).sum().item())
                if bad:
                    ok_local = False
                    bad_total += bad
                    logger.warning(
                        "barlink-BAR1: a2a byte-level proof %d->%d (%s) "
                        "FAILED: %d of %d bytes wrong. all_to_all "
                        "withdraws.",
                        q, r, "skewed" if skewed else "uniform", bad,
                        recv[q],
                    )
                o += recv[q]
            if not bad_total:
                # The passed proof belongs in the log too -- it is the
                # statement every later timing measurement rests on.
                logger.info(
                    "barlink-BAR1: a2a byte-level proof (%s) passed: 0 of %d "
                    "bytes wrong across %d senders.",
                    "skewed" if skewed else "uniform", sum(recv), R,
                )
        return ok_local

    # -- Peer liveness -------------------------------------------------------

    def _install_abort_window(self) -> None:
        """One host word the spin kernels poll, so a dead peer can end them.

        Only built when the feature is on. When the runtime refuses to map
        the word, ``AbortWindow`` degrades to ``device_ptr == 0`` on its own
        and the kernels see ``nullptr``; nothing here has to special-case
        that.
        """
        if not barlink_liveness.liveness_enabled():
            return
        try:
            window = barlink_liveness.AbortWindow()
        except Exception as e:            # pragma: no cover - degrade, do not refuse to boot
            logger.warning("barlink-BAR1: no abort window (%s).", e)
            return
        barlink_liveness.register_abort_window(window)
        self._abort_window = window

    def _release_abort_window(self) -> None:
        window, self._abort_window = self._abort_window, None
        if window is None:
            return
        barlink_liveness.unregister_abort_window(window)
        window.close()

    def _wait_abort(self, label: str):
        """``on_abort`` hook for the bounded waits of this transport.

        It trips the abort word and nothing else. Deliberately NOT
        ``raise_if_aborted``: that reads ``ctlStatus`` off the device, the
        read queues behind whatever is on the stream, and on the timeout
        branch -- where the module does not trip the windows itself -- the
        thing on the stream is exactly the spin kernel nobody can get past.
        The abort handler would then hang in the same way the wait it is
        handling did. Tripping first is what makes that read finish, so the
        ordering is: hook trips, the wait raises with the peer named, and
        whoever catches it can ask ``raise_if_aborted`` afterwards.
        """

        def _hook() -> None:
            window = self._abort_window
            if window is not None:
                window.trip(f"host wait '{label}' gave up")

        return _hook

    @property
    def _abort_host(self) -> int:
        """Device address of the abort word, ``0`` when there is none.

        Every extension call passes this. ``0`` is the pre-#312 behaviour
        exactly: the kernels keep their cycle deadline and probe nothing.
        """
        window = self._abort_window
        return window.device_ptr if window is not None else 0

    # -- The device deadline (#431 fix 1) ------------------------------------

    def _deadline_cycles(self) -> int:
        """The cycle budget THIS launch carries. The only producer of it.

        ``self.cap_cycles`` is the steady-state constant
        (``SGLANG_BARLINK_BAR1_CAP_CYCLES``, ~30 s at 2 GHz). What actually
        reaches the kernel is that constant put through
        ``resolve_timeout_cycles``, which is the identity outside the JIT
        cold-build window and multiplies by ``cold_build_timeout_mult()``
        (default 40) inside it.

        #431: this indirection is the fix. ``barlink_liveness.py`` has
        documented since #312 that the BAR1 device cap "is multiplied by up
        to 40x inside the JIT cold-build window", and
        ``barlink_device.py:1239`` does exactly that -- but BAR1 passed
        ``int(self.cap_cycles)`` raw at all three of its launch sites and
        never called the resolver at all. The mechanism existed, was
        documented, was used by the sibling transport, and did not reach the
        one transport whose kernels spin on a device deadline. The #431
        window measured the consequence: the cold-build window was open for
        the entire 22-minute stall (6 opens, ``jit/coldwindow_*.txt``) while
        the arm advanced at one collective per ~30-40 s -- the RAW cap, to
        the digit.

        Two properties this must keep:

        * OUTSIDE the window it is byte-identical to the previous behaviour.
          ``resolve_timeout_cycles`` returns its argument unchanged there.
        * The CAPTURED value is the steady-state one. The pass that records
          a CUDA graph runs outside the window by construction
          (``full_cuda_graph_backend.capture_one``: ``run_capture_warmups``
          owns the window, the recorded ``forward_fn()`` below it does not),
          so the deadline baked into a graph is the unmultiplied constant --
          which is what every replay for the rest of the process's life then
          uses. Extending the deadline for the warmups does not extend it for
          serving.
        """
        return int(resolve_timeout_cycles(int(self.cap_cycles)))

    # -- The loud abort (#431 fix 2) -----------------------------------------

    def _note_launch(
        self, op: str, nbytes: int, variant: Optional[int] = None
    ) -> None:
        """Record what is about to be launched. Three stores, no allocation.

        Under stream capture the kernel is RECORDED, not executed, so there
        is nothing for a status read to find and the unchecked counter must
        not advance -- otherwise the first check after capture would report a
        stale window of "collectives" that never ran. What capture does mean
        is that this transport's kernels are now inside a graph and will run
        on every replay with no host code between them; that is what
        ``_captured_launches`` arms.

        ``variant`` is the kernel choice from ``_kernel`` (1 = cooperative
        grid, 0 = 1blk). It is passed only so the capture census can record
        it: the choice is derived from ``moved`` bytes, and where ``moved`` is
        not rank-uniform the ranks bake DIFFERENT kernels into the same
        collective -- a divergence no other instrument in this tree can see
        (#603b). Not stored on the hot-path attributes; the census call below
        is already gated on capture, so a serving replay pays nothing.
        """
        self._last_op = op
        self._last_nbytes = int(nbytes)
        from sglang.srt.distributed.device_communicators.barlink import (
            graph_capture_running,
        )

        if graph_capture_running():
            self._captured_launches = True
            self._last_op_captured = True
            # #603b: this launch is being BAKED INTO A GRAPH. Everything the
            # replay will do is decided here, and nothing observes it later --
            # so it is recorded here or not at all.
            from sglang.srt.distributed.device_communicators import (
                barlink_capture_census,
            )

            barlink_capture_census.note(op, int(nbytes), variant)
        else:
            self._unchecked_launches += 1
            self._last_op_captured = False

    def _rounds_for(self, op: str, nbytes: int) -> int:
        """Round count of the named collective, computed at RAISE time only.

        Never on the hot path: ``ar_rounds``/``ag_rounds`` run the planner,
        and the launch sites would pay for a number that is only ever read
        once a run is already broken.
        """
        try:
            if op.startswith("all_reduce"):
                return int(self.ar_rounds(nbytes))
            if op == "all_gather":
                return int(self.ag_rounds(nbytes))
        except Exception:  # noqa: BLE001 - a diagnostic must not be the cause
            pass
        return 0

    # -- The deferred status read (#517) -------------------------------------

    def _arm_status_stage(self) -> None:
        """Build the staging buffer for the asynchronous status read.

        Called once, from bring-up, right after ``_ctl_dev`` exists. Whether
        the deferred path is used at all is
        ``barlink_abort_gate.should_defer_status`` -- one definition, tested
        directly -- and it says no for a status word that is not on a device:
        there is no synchronization to avoid there, and a direct read is both
        cheaper and stricter.

        The 1-element view is built ONCE and kept. ``self._ctl_dev[0]`` is an
        aten dispatch, and the pre-#517 read paid it plus ``.item()`` on
        every single check; the staged path pays one ``copy_`` dispatch and
        nothing else.
        """
        ctl = self._ctl_dev
        if ctl is None:
            return
        if not barlink_abort_gate.should_defer_status(
            bool(getattr(ctl, "is_cuda", False)),
            barlink_abort_gate.defer_enabled(),
        ):
            return
        import torch

        try:
            src = ctl[0:1]
            stage = torch.zeros(1, dtype=ctl.dtype, pin_memory=True)
            event = torch.cuda.Event()
        except Exception as e:  # noqa: BLE001 - degrade to the blocking read
            logger.warning(
                "barlink-BAR1 group %s: no staged status read (%s); falling "
                "back to the blocking read.",
                self.group,
                e,
            )
            return
        self._ctl_src = src
        self._ctl_stage = stage
        self._ctl_event = event
        self._ctl_inflight = False
        self._ctl_lag = 0
        self._ctl_defer = True

    def _arm_abort_poll(self) -> None:
        """Build the watchdog's private-stream read of the status word.

        Why this exists (#616f). ``barlink_abort_gate.poll_status_words``
        walks every registered transport and calls ``poll_status_word`` on
        each -- but it looks the method up with ``getattr(..., None)`` and
        skips the transport when it is absent. Only the DEVICE transport ever
        defined it. The BAR1 transport therefore had no watchdog read at all:
        the walk reached it, found nothing, and moved on, silently. Every
        abort report on the BAR1 path consequently depended on the in-line
        staged read, which is ordered behind the compute stream -- so a
        wedged compute stream produced no report from either side.

        Failure to build the pair degrades to exactly the previous
        behaviour (no watchdog read), never to a raise: a guard that can
        break bring-up is worse than the gap it closes.
        """
        ctl = self._ctl_dev
        if ctl is None:
            return
        if not bool(getattr(ctl, "is_cuda", False)):
            # A host word needs no private stream; the direct read is already
            # unordered with respect to the compute stream.
            return
        if not barlink_abort_gate.abort_check_enabled():
            return
        import torch

        try:
            self._abort_poll_stream = torch.cuda.Stream(device=self.device)
            self._abort_poll_dst = torch.zeros(1, dtype=ctl.dtype, pin_memory=True)
            # #622: pinned mirror of (round, mesh watermark, a2a watermark),
            # refreshed by the same watchdog poll; read by the launch dump.
            self._round_mirror = torch.zeros(3, dtype=torch.int64, pin_memory=True)
        except Exception as e:  # noqa: BLE001 - degrade to the in-line read
            logger.warning(
                "barlink-BAR1 group %s: no watchdog abort poll (%s); keeping "
                "the in-line read.",
                self.group,
                e,
            )
            self._abort_poll_stream = None
            self._abort_poll_dst = None
            self._round_mirror = None
            return
        self._abort_poll_active = True

    def poll_status_word(self) -> bool:
        """One watchdog read of the abort word. Returns True once tripped.

        Runs on the WATCHDOG thread, never on the serving path. The copy is
        issued on a private stream and only that stream is synchronized, so
        the read waits for its own four bytes and for nothing the model is
        doing. That is the entire point: it is the one read that still
        resolves while the compute stream is wedged.

        Sticky in both directions of the word's life: the device word only
        ever goes 0 -> non-zero, and this mirror only ever follows it once,
        so a poll after a trip is a no-op and the hot path's view can never
        go backwards.
        """
        if not self._abort_poll_active or self._ctl_dev is None:
            return bool(self._abort_code_seen)
        if self._abort_code_seen:
            return True
        import torch

        with torch.cuda.stream(self._abort_poll_stream):
            self._abort_poll_dst.copy_(self._ctl_dev[0:1], non_blocking=True)
            # #622: stage the three round words (round, mesh watermark, a2a
            # watermark) alongside the abort word, on the same private
            # stream. This gives the SIGUSR1 launch dump a host-resident
            # mirror to print — the live monotonicity probe for the ack
            # barrier's capture-safety proof — without ever violating the
            # dump's no-device-sync constraint.
            if self._round_dev is not None and self._round_mirror is not None:
                n = min(3, self._round_dev.numel())
                self._round_mirror[:n].copy_(self._round_dev[:n], non_blocking=True)
        # Waits for THIS copy on THIS stream only -- not for the model.
        self._abort_poll_stream.synchronize()
        code = int(self._abort_poll_dst[0])
        if code:
            self._abort_code_seen = code
            return True
        return False

    def _wait_ctl_event(self) -> bool:
        """Wait for the staged copy, but never past the deadline.

        Returns True when the event resolved (the staged word is readable),
        False when the deadline expired with the copy still in flight.

        A deadline of 0 restores the pre-#616f blocking wait verbatim, for
        bisecting against this change.

        The expiry is LOGGED, once per transport per ramp of ten, because it
        is the first host-visible symptom of a wedged compute stream that
        this path can emit at all. Before #616f the same condition produced
        no line anywhere: the thread was inside ``event.synchronize()`` and
        stayed there.
        """
        deadline_s = barlink_abort_gate.sync_deadline_s()
        if deadline_s <= 0.0:
            # #818: this branch is genuinely unbounded -- it blocks in the CUDA
            # driver with no timeout at all. That is exactly the shape
            # ``check_peers`` exists for, so ask ONCE before entering it. It
            # cannot help a peer that dies after this point, and it does not
            # pretend to: the deadline-0 mode is a bisecting aid, not a
            # supported serving configuration.
            raise_if_peer_lost(self, 0.0)
            self._ctl_event.synchronize()
            return True
        start = time.monotonic()
        end = start + deadline_s
        next_probe = start + barlink_liveness.probe_interval_s()
        while True:
            if self._ctl_event.query():
                self._ctl_stall_run = 0
                # #615: the stream retired the copy, so whatever a build
                # window forgave is spent history and the next stall starts
                # from a full cap. This is the ONLY thing that clears it.
                self._ctl_build_deferred_s = 0.0
                return True
            now = time.monotonic()
            if now >= end:
                break
            # #818: probe peer liveness at the cadence the liveness module
            # already defines for its own bounded waits -- NOT once per 0.5 ms
            # iteration, which would be world_size syscalls at 2 kHz. With the
            # shipped defaults (2 s deadline, 1 s probe) this fires about twice
            # per wait; with an operator-raised deadline it is what keeps the
            # gate's response time bounded by the probe interval instead of by
            # the deadline.
            if now >= next_probe:
                next_probe = now + barlink_liveness.probe_interval_s()
                raise_if_peer_lost(self, now - start)
            time.sleep(0.0005)
        self._ctl_sync_timeouts += 1
        self._ctl_stall_run += 1
        n = self._ctl_sync_timeouts
        if n == 1 or n % 10 == 0:
            logger.warning(
                "barlink-BAR1 group %s rank %s: staged status read did not "
                "resolve within %.1f ms (%d time(s)); the compute stream has "
                "not retired the copy. last_op=%s last_nbytes=%d "
                "unchecked_window=%d. The watchdog's private-stream poll is "
                "now the only reader that can still see a trip.",
                self.group,
                self.rank,
                deadline_s * 1000.0,
                n,
                self._last_op or "unknown",
                int(self._last_nbytes),
                int(self._deferred_launches),
            )
        # #619: expiry-path capture-census dump. The warning above fires on
        # the rank whose compute stream is stuck, and that rank never reaches
        # the abort handler -- so it dumps no census there. Dump the SAME
        # capture census the abort path uses, after N expiries. Collective-free,
        # device-sync-free, no new allocation (one bool read for the latch).
        threshold = int(os.environ.get(
            'SGLANG_BARLINK_EXPIRY_CENSUS_AFTER', '3'))
        if (threshold > 0 and n >= threshold
                and not self._expiry_census_fired):
            self._expiry_census_fired = True
            try:
                from sglang.srt.distributed.device_communicators import (
                    barlink_capture_census,
                )

                if barlink_capture_census.capture_census_enabled():
                    logger.error(
                        '%s',
                        barlink_capture_census.format_local_capture_census(
                            self.rank),
                    )
            except Exception:  # noqa: BLE001 - never mask the path below
                pass
        # #818: THE fix. Before consulting the consecutive-expiry ceiling and
        # before letting a build window forgive the run, ask whether a peer is
        # simply GONE.
        #
        # Both of the gates below are unreachable for a dead peer, and that is
        # what cost the instance:
        #   * the ceiling counts CONSECUTIVE expiries (``_ctl_stall_run`` is
        #     reset to 0 by every resolved read above), and a dead peer
        #     produces INTERMITTENT expiries -- the 2026-08-23 specimen logged
        #     20 CUMULATIVE expiries in ~36 s while PP1 was already dead, and
        #     never escalated;
        #   * ``defer_stall_for_building_peer`` can extend the wait to
        #     ``SGLANG_BARLINK_BUILD_WINDOW_CAP_S`` (900 s default) off a
        #     marker a dead peer may have left behind -- a process that no
        #     longer exists is not building anything, and must never be
        #     forgiven for it.
        # A stall says "wait longer". A lost peer says "stop now". This is the
        # line that tells them apart.
        raise_if_peer_lost(self, self._ctl_stall_run * deadline_s)
        limit = barlink_abort_gate.stall_raise_after()
        if limit and self._ctl_stall_run >= limit:
            # #615: ...unless a PEER is legitimately inside a lazy JIT build
            # and has published it. The escalation below is ~60 s of stall by
            # default (30 expiries x 2 s), and an nvcc build of gptq_marlin or
            # a flashinfer JIT runs for minutes -- so without this check the
            # guard aborts the group for a healthy cold boot. The extension is
            # bounded by ``SGLANG_BARLINK_BUILD_WINDOW_CAP_S`` measured over
            # the whole stall, so a rank that publishes a build and then
            # actually wedges is still caught, just later and by name.
            #
            # This costs the waiter one ``stat`` per same-host peer, on a path
            # that has already stalled for a minute. It adds no sync, no
            # ``.item()`` and no device read -- deliberately: the wedge census
            # (239 events) found that every bounded poll recovered and only
            # the unbounded host syncs were fatal-capable, so the guard's
            # extension must not be the thing that introduces one.
            if defer_stall_for_building_peer(self, self._ctl_stall_run, deadline_s):
                return False
            # The escalation. Logging alone left #616 a black hole: the host
            # thread stayed in the guard and the only actor that ever ended
            # the wedge was an external supervisor, minutes later, with no
            # attribution. A raise reaches the serving path with the op and
            # the byte count in hand.
            run = self._ctl_stall_run
            self._ctl_stall_run = 0
            raise Bar1CollectiveStalled(
                self.group,
                self.rank,
                self._last_op or "unknown",
                int(self._last_nbytes),
                # #615: the time a published build already bought is part of
                # how long this stall has run. Zero unless an extension
                # happened, so the pre-#615 arithmetic is unchanged on every
                # path that never saw a building peer. ``getattr`` because
                # this suite's documented pattern is to invoke these methods
                # unbound against stubs, and a diagnostic must not be what
                # turns a stall report into an AttributeError.
                run * deadline_s + getattr(self, "_ctl_build_deferred_s", 0.0),
                run,
            )
        return False

    def _read_status_for_check(self):
        """The status value for one check, or ``None`` if it is not in yet.

        Blocking mode (``_ctl_defer`` false) is the pre-#517 read verbatim:
        one D2H plus a stream synchronization, and never ``None``.

        Staged mode issues a non-blocking D2H of the sticky word onto the
        CURRENT stream -- the same stream the collective or the replayed
        graph just ran on, so the copy is ordered after it -- and reads the
        value staged by an EARLIER check. Three properties make that a guard
        rather than a hope:

        * ``ctlStatus`` is sticky (only ever written ``1u`` by a tripped
          kernel, only ever zeroed by the ``torch.zeros`` at bring-up), so a
          value seen late is the same value seen on time;
        * the report is bounded: after ``max_lag()`` consecutive unresolved
          checks one ``event.synchronize()`` is forced, which is a wait the
          blocking mode paid at EVERY check;
        * the size of the unverified window is accumulated
          (``_deferred_launches``) rather than reset, so the raise still
          names how many collectives are implicated.
        """
        if not self._ctl_defer:
            return self.status()
        # #616f: the watchdog's private-stream mirror outranks the staged
        # word. It is sticky and it resolves while the compute stream is
        # wedged, which is precisely the case the staged read cannot serve.
        if self._abort_code_seen:
            return int(self._abort_code_seen)
        value = None
        if self._ctl_inflight:
            if self._ctl_event.query():
                self._ctl_inflight = False
                self._ctl_stall_run = 0
                self._ctl_build_deferred_s = 0.0  # #615: see _wait_ctl_event
                value = int(self._ctl_stage[0])
            else:
                self._ctl_lag += 1
                if self._ctl_lag >= barlink_abort_gate.max_lag():
                    # The bound. An overlap-scheduled host can queue work
                    # indefinitely; without this the staged word could stay
                    # in flight for as long as it keeps queueing, and a
                    # check that never resolves is not a check.
                    #
                    # #616f: but the wait itself must be bounded too. This
                    # event is ordered behind the collective that just ran,
                    # so waiting on it without a deadline makes the guard
                    # hang on the very fault it reports. A deadline turns
                    # "never returns" into "unresolved this time"; the
                    # window is preserved (``_deferred_launches`` is only
                    # cleared by a RESOLVED clean read), so nothing is lost
                    # but the latency.
                    if self._wait_ctl_event():
                        self._ctl_inflight = False
                        value = int(self._ctl_stage[0])
                    else:
                        # Still in flight. Leave it in flight -- re-recording
                        # the event would queue a SECOND copy behind the same
                        # stuck kernel and lose the one already staged.
                        return None
        if not self._ctl_inflight:
            self._ctl_stage.copy_(self._ctl_src, non_blocking=True)
            self._ctl_event.record()
            self._ctl_inflight = True
            self._ctl_lag = 0
        return value

    def _abort_flag_snapshot(self, max_lines: int = 64) -> Optional[str]:
        """This rank's flag words, read AFTER a spin kernel took its abort path.

        Why this is the one instrument the wedge still lacks (#616c). Every
        HOST-visible signal says the ranks agree: the #583 census counts match,
        the launch sampler shows identical last ops and identical
        captured/eager flags, and the host kernel logs zero Xid / AER / IOMMU
        faults over 23 h. Yet all three GPUs sit at 100 % SM occupancy with
        0 % memory utilisation -- spinning, not computing. What nothing reads
        is the DEVICE flag state the spin actually waits on, which is where the
        remaining explanations live (a sequence/generation mismatch, or a flag
        written but never observed).

        Why reading it HERE is safe, when the 1 Hz launch sampler deliberately
        refuses to: the sampler runs while a collective may still be in flight,
        so a device read would queue behind the wedged spin and hang exactly
        when the evidence is wanted. By the time this runs the kernel has
        ALREADY taken its abort path, so there is nothing left to queue behind.

        One 256-byte line per (topology, step, sender); the flag/generation
        word is the first dword of each line, so only that is reported.
        Diffing the three ranks' snapshots is what names a generation mismatch
        -- a value one rank still waits for that its peers have already passed.
        """
        fptr, _handle, fsize = self._own_flag
        if not fptr or not fsize or self._cuda is None:
            return None
        line = 256
        n_lines = min(int(fsize) // line, int(max_lines))
        if n_lines <= 0:
            return None
        nbytes = n_lines * line
        buf = (ctypes.c_ubyte * nbytes)()
        self._cuda.memcpy(ctypes.addressof(buf), int(fptr), nbytes)
        raw = bytes(buf)
        words = [
            int.from_bytes(raw[i * line : i * line + 4], "little")
            for i in range(n_lines)
        ]
        shown = " ".join(f"{i}:{w}" for i, w in enumerate(words))
        # #622/#649: the round counter, WITHOUT which this whole dump cannot be
        # interpreted.
        #
        # Every reading of these snapshots so far has had to assume which round
        # the spin was waiting for, because roundDev is never printed anywhere
        # in this file. That assumption is load-bearing: the claim "the
        # aborting spin's exit condition was already satisfied in its own flag
        # region" (ANALYSE_622_replay_abort.md:151-160) is derived from it, and
        # it has never been measured. It is the same class of error as the
        # retracted per-rank-maximum reading (41d76e7513), which this very
        # function's closing sentence still warns against.
        #
        # With the counter printed, "was the exit condition satisfied?" stops
        # being an inference and becomes a subtraction: compare round against
        # the per-topology watermarks above. A spin waiting on R with peers
        # published at R is satisfied; waiting on R with peers at R-1 is a
        # genuine missing flag.
        #
        # Read through the SAME ctypes memcpy path as the flag region above,
        # deliberately NOT via .item(): a tensor read would enqueue on the
        # compute stream and, on the very wedge this runs inside, would queue
        # behind the stuck kernel and hang exactly when the evidence is wanted.
        # Eight bytes from a local VRAM word the peers never touch.
        #
        # This belongs HERE and not in the host-only sibling dump: the sibling
        # path deliberately takes no device access at all, and the existing
        # device copy already cost 55 s in the 06:12 specimen
        # (barlink_abort_gate.py:544-548). Adding device reads per sibling
        # transport would multiply that.
        round_txt = "unavailable"
        try:
            rd = getattr(self, "_round_dev", None)
            if rd is not None and self._cuda is not None:
                # #622: THREE words now -- the counter and the two consumption
                # watermarks (mesh, a2a). The sentence below already told the
                # reader to compare the counter against the per-topology
                # watermarks; before the acknowledgment protocol there were
                # none to compare against. Read only as many words as the
                # tensor actually has, so a transport built by an older path
                # still yields its counter instead of an unreadable dump.
                words = min(3, int(rd.numel()))
                rbuf = (ctypes.c_ubyte * (8 * words))()
                self._cuda.memcpy(ctypes.addressof(rbuf), int(rd.data_ptr()),
                                  8 * words)
                raw = bytes(rbuf)
                vals = [int.from_bytes(raw[8 * i:8 * i + 8], "little")
                        for i in range(words)]
                names = ("round", "mesh_consumed", "a2a_consumed")
                round_txt = ", ".join(f"{names[i]}={vals[i]}"
                                      for i in range(words))
        except Exception as exc:  # noqa: BLE001 - never lose the flag dump
            # The flag words above are the primary evidence; failing to read an
            # 8-byte counter must never suppress them.
            round_txt = f"unreadable ({type(exc).__name__})"
        return (
            f"barlink-BAR1 abort flag snapshot rank {self.rank}/{self.world} "
            f"group {self.group or '<unnamed>'}: {n_lines} lines of "
            f"{int(fsize)} bytes, roundDev[{round_txt}], first dword per line "
            f"-- {shown}. "
            "Compare against the peers': a rank waiting on a generation its "
            "peers have already passed is a sequence mismatch, whereas "
            "all-equal values mean the flags agree and the wait is elsewhere. "
            "roundDev is this rank's own counter at dump time; compare it "
            "against the per-topology watermarks rather than assuming which "
            "round the spin awaited. mesh_consumed/a2a_consumed (#622) are "
            "the last rounds of those topologies this rank COMPLETED; a "
            "watermark far behind the counter means the rank is waiting to be "
            "let into a collective, not inside one."
        )

    def _abort_peer_flag_snapshot(self, max_lines: int = 64) -> Optional[str]:
        """Every PEER's flag region, read from THIS process at abort time.

        Closes the gap that made the first three wedges only partly readable
        (#616c). The abort path dumps the flag words of whichever rank reaches
        it, and the last rank to abort tends to die before it gets there -- so
        every incident so far yielded at most 2 of 3 ranks, and any
        (block, sender) cell involving the missing rank could not be compared.
        One incident even showed a sender-2 disagreement that the next could
        neither confirm nor refute, precisely because rank 2 was the one that
        never emitted.

        This needs no device sync and no cooperation from the peer: setup
        already mapped each peer's flag region into THIS process's address
        space (``PeerTarget.flag.host_address``), so the read is an ordinary
        host load from a mapped BAR window. That is what makes it safe on the
        abort path, where a device read would queue behind the wedged spin.

        Reading a write-combined BAR mapping from the host is slow per access;
        it is bounded here to ``max_lines`` lines of 256 bytes per peer, which
        is a few kB total and only ever runs once, after the collective has
        already failed.
        """
        peers = getattr(self, "_peers", None)
        if not peers:
            return None
        parts = []
        for peer_rank in sorted(peers):
            mapping = getattr(peers[peer_rank], "flag", None)
            host_addr = getattr(mapping, "host_address", 0) if mapping else 0
            length = int(getattr(mapping, "length", 0) or 0) if mapping else 0
            if not host_addr or length <= 0:
                continue
            line = 256
            n_lines = min(length // line, int(max_lines))
            if n_lines <= 0:
                continue
            raw = bytes(
                (ctypes.c_ubyte * (n_lines * line)).from_address(int(host_addr))
            )
            words = " ".join(
                f"{i}:{int.from_bytes(raw[i * line : i * line + 4], 'little')}"
                for i in range(n_lines)
            )
            parts.append(f"peer {peer_rank} [{n_lines} lines]: {words}")
        if not parts:
            return None
        return (
            f"barlink-BAR1 abort PEER flag snapshot, observed by rank "
            f"{self.rank}/{self.world} group {self.group or '<unnamed>'} -- "
            + " || ".join(parts)
            + ". These are the peers' OWN flag regions read from this process, "
            "so one surviving rank publishes every rank's view; compare cells "
            "by (block, sender), never by per-rank maximum."
        )

    def check_aborted(self, where: str) -> None:
        """The production check: raise if a kernel took its abort path.

        This is what #431 fix 2 adds. Before it, ``ctlStatus`` was written by
        every tripped kernel and read by nothing on a serving path, so a run
        in which every collective aborted looked exactly like a run in which
        none did -- and the stream kept going over partially written output
        buffers. The #431 window's ``abort_*.txt`` is empty for a 22-minute
        stall for precisely this reason.

        WHERE IT MAY RUN. Reading the status word is a device read: it
        synchronizes the current stream, and inside a stream capture it is
        not just expensive but illegal. So:

        * inside a capture this returns immediately (``graph_capture_running``
          is the single definition of that question, ``barlink.py:414``);
        * on the host path it runs after the collective, from the barlink
          dispatch sites;
        * for kernels that only ever run inside a replayed graph -- where
          there is no host code per collective at all -- the next host point
          is the replay boundary, and ``barlink_abort_gate`` is called from
          there.

        A background watchdog THREAD was considered for the captured case and
        rejected: it would read the same device word, that read queues behind
        whatever is on the stream, and the thing on the stream is exactly the
        kernel it is meant to report on. ``_wait_abort`` above records the
        same reasoning for the same read; a thread does not change it.

        COST (#517). Free when nothing has been launched since the last
        check. Otherwise one 4-byte D2H, STAGED: a non-blocking copy onto the
        current stream plus a ``cudaEventQuery``, reading the value an
        earlier check staged. No stream synchronization in the steady state.
        The pre-#517 blocking read -- which the #476 window measured at 6.64
        pp of code decode_TPS on Seam A and 5.26 pp on Seam B -- is restored
        exactly by ``SGLANG_BARLINK_BAR1_ABORT_DEFER=0``, and a staged value
        that does not arrive within ``..._ABORT_MAX_LAG`` checks forces one
        wait, so the report stays bounded.

        WHY THIS DOES NOT WEAKEN THE GUARD. ``ctlStatus`` is sticky: the
        kernels only ever write ``1u`` and the only host write is the
        ``torch.zeros`` at bring-up. A late read of a sticky bit is the same
        bit. What the deferral changes is the reporting latency, and #476 §3
        is the case it has to keep catching -- an intermittent abort observed
        at a replay boundary, which under the overlap scheduler is still many
        checks before the round's tokens reach the host at
        ``batch_result_processor.py:217`` (``copy_done.synchronize()``).
        """
        if self._ctl_dev is None:
            return
        if not barlink_abort_gate.abort_check_enabled():
            return
        pending = self._unchecked_launches
        if pending <= 0 and not self._captured_launches:
            return
        every = barlink_abort_gate.check_every()
        if pending > 0:
            if pending < every:
                return
        else:
            # Replay-boundary entry. Its own counter, because the host-path
            # one is zero here by construction -- which is why ..._CHECK_EVERY
            # could not reach this seam before #517. K = 1 keeps every
            # boundary checking, i.e. the default is behaviour-identical.
            self._boundary_checks += 1
            if self._boundary_checks < every:
                return
            self._boundary_checks = 0
        from sglang.srt.distributed.device_communicators.barlink import (
            graph_capture_running,
        )

        if graph_capture_running():
            return
        self._unchecked_launches = 0
        self._deferred_launches += pending
        status = self._read_status_for_check()
        # #622: the kernels now write TWO non-zero codes. 1 is the historical
        # one (a payload barrier gave up); 2 is the entry acknowledgment wait.
        # Both mean "a spin kernel took its abort path", so both must raise --
        # a `!= 1` test here would have made the new code an invisible abort,
        # which is worse than the deadlock it reports on.
        if status not in (1, 2):
            if status is not None:
                # A RESOLVED clean read closes the window it covered; an
                # unresolved one (None) must not, or the raise would
                # understate how many collectives are implicated.
                self._deferred_launches = 0
            return
        pending = self._deferred_launches
        op = self._last_op or "unknown"
        nbytes = int(self._last_nbytes)
        rounds = self._rounds_for(op, nbytes)
        window = self._abort_window
        reason = window.reason if window is not None and window.tripped else None
        staged = (
            " The status word was read from the STAGED copy (#517), so the "
            "abort may have happened one check earlier than this line; the "
            "word is sticky, so nothing is lost, only delayed."
            if self._ctl_defer
            else ""
        )
        # #583: ATTRIBUTION. ``_last_op`` is stored by ``_note_launch`` on
        # every launch INCLUDING captured ones, while ``_unchecked_launches``
        # is deliberately not advanced under capture. So when the window is
        # empty the named collective is not in it -- it is simply the last
        # launch this transport ever saw, which at a replay boundary in the
        # steady state is whatever was recorded at graph-capture time. The
        # 2026-08-06 05:53:59 crash presented exactly that as "its most
        # recent member" and cost a triage cycle chasing a collective that
        # had not run for the whole serving run.
        named = f"{op} ({nbytes} bytes, {rounds} rounds)"
        if pending > 0:
            attribution = (
                f"Last collective launched: {named}; {pending} collective(s) "
                f"ran on the host path since the previous check, so the abort "
                f"is in that window and the named one is its most recent "
                f"member."
            )
        else:
            # getattr, not attribute access: this runs on an already-broken
            # run, and a diagnostic must never be the thing that raises. Some
            # bring-up and test paths build the transport without __init__.
            origin = (
                "recorded under CUDA-graph capture"
                if getattr(self, "_last_op_captured", False)
                else "from an earlier, already-closed window"
            )
            # #622: the sentence above is true and was, until now, the end
            # of the trail -- both the 2026-08-05 21:10 and 2026-08-07 03:25
            # specimens stop exactly here. The kernel still cannot be named
            # (no host code runs per collective inside a replay), but the
            # GRAPH it is in can be: the host chose that graph one frame above
            # the launch, and barlink_abort_gate records it there. Diff this
            # line across the ranks -- ranks in different windows is host-path
            # divergence, ranks in the SAME window sends you to that graph's
            # capture census.
            replay_tag = "<unavailable>"
            try:
                replay_tag = barlink_abort_gate.format_current_replay()
            except Exception:  # noqa: BLE001 - a diagnostic must not mask this
                pass
            attribution = (
                "No collective ran on the host path since the previous check, "
                "so this is a GRAPH-REPLAY window: the kernel that aborted is "
                "inside the replayed graph and is NOT named here. The last "
                f"launch this transport recorded is {named}, {origin}; it can "
                "predate this abort by the whole run and must not be read as "
                f"the culprit. REPLAY WINDOW (#622): {replay_tag}."
            )
        # #583: CAUSE. The two causes the message used to offer as a
        # disjunction are not equally unknowable -- the host abort word's
        # state is local and exact. Reporting "either A or B" when B is
        # decidable here forced the reader to reconstruct it from the absence
        # of an unrelated log line. Decide it.
        deadline = (
            f"the kernel exceeded its cycle deadline "
            f"(SGLANG_BARLINK_BAR1_CAP_CYCLES={self.cap_cycles}, effective "
            f"this launch {self._deadline_cycles()}) waiting for a peer's flag"
        )
        if reason is not None:
            cause = f"Cause: the host abort word was set on this rank -- {reason}."
        elif window is None:
            cause = (
                f"Cause: {deadline}. This rank has no device-mapped host abort "
                f"word, so the cycle deadline is the only path a kernel has "
                f"into its abort branch."
            )
        else:
            cause = (
                f"Cause: {deadline}. The host abort word exists on this rank "
                f"and was NOT set, which excludes it: no peer was declared "
                f"dead and no host wait gave up. A peer did not arrive."
            )
        # #622: WHERE in the kernel. Status 1 is a payload barrier -- this
        # rank had already published its own data and was waiting for the
        # peers' contribution to the collective it is IN. Status 2 is the
        # entry acknowledgment wait, which is a different sentence about a
        # different collective: this rank had not yet sent anything for the
        # current one, it was waiting for the peers to finish consuming the
        # payload of the PREVIOUS same-topology collective before overwriting
        # the slots. A rank stuck there is downstream of a peer that is
        # behind, not the origin of the stall -- look at the peer whose ack
        # line is short, not at this rank's own last collective.
        if status == 2:
            cause += (
                " PHASE: the abort fired in the ENTRY ack wait (#622): this "
                "rank waited for peers to consume its PREVIOUS collective's "
                "payload, before publishing anything for the collective named "
                "above. The peer that did not acknowledge is the one to chase; "
                "its own report, if it produced one, names the phase it was in."
            )
        # #583: dump THIS rank's collective census before raising. No
        # collective is taken here on purpose -- by now a peer is very likely
        # already wedged or dead (in crashes 9/10/11 the third rank aborted
        # ~30 s after the first two), so any cross-rank exchange would hang
        # exactly when the evidence is wanted. Each rank logs its own counts;
        # diffing the three lines afterwards names the family that diverged.
        # Warn-never-raise: an instrument must not replace the real error.
        try:
            from sglang.srt.distributed.collective_census import (
                census_enabled,
                format_local_census,
            )

            if census_enabled():
                logger.error("%s", format_local_census(self.rank))
                # #616c: the COUNTS say which family diverged, but not where.
                # The flag snapshot below proved the ranks sit one generation
                # apart; this recent-sequence dump is what turns that into a
                # named culprit -- diff the three ranks' history lines and the
                # first differing entry is the collective that caused the skew.
                from sglang.srt.distributed.collective_census import (
                    format_local_history,
                )

                logger.error("%s", format_local_history(self.rank))
        except Exception:  # noqa: BLE001 - never mask the abort below
            pass
        # #622 (composes with #619): the collective census above counts HOST
        # calls, and a replay makes none -- so on a GRAPH-REPLAY abort it is
        # silent by construction, which is precisely the window the replay tag
        # just named. The CAPTURE census is the instrument that can speak
        # about that window: it recorded, at capture time, the ordered kernel
        # list baked into each graph. It was previously dumped only from the
        # scheduler's periodic tick (scheduler.py _capture_census_once), which
        # a rank that dies first never reaches -- the 2026-08-05 specimen has
        # rank 0 raising 10 s before the others for exactly that reason.
        # Dumping it here pairs "which graph was replaying" with "what is in
        # that graph", per rank, on the one rank that is certain to get here.
        # Scope note: this ADDS a dump site for the abort path only. #619
        # owns the expiry-path dump; nothing about that path is changed here.
        # Warn-never-raise, same discipline as the census above.
        try:
            from sglang.srt.distributed.device_communicators import (
                barlink_capture_census,
            )

            if barlink_capture_census.capture_census_enabled():
                logger.error(
                    "%s", barlink_capture_census.format_local_capture_census(self.rank)
                )
        except Exception:  # noqa: BLE001 - an instrument must not mask the abort
            pass
        # #616c: and the device flag words this rank's spin was waiting on.
        # Same warn-never-raise discipline as the census above, and safe only
        # because the kernel has already aborted -- see _abort_flag_snapshot.
        try:
            snapshot = self._abort_flag_snapshot()
            if snapshot:
                logger.error("%s", snapshot)
        except Exception:  # noqa: BLE001 - an instrument must not mask the abort
            pass
        # #616c: and every PEER's region, read from here. The rank that dies
        # first never reaches its own dump, so without this an incident yields
        # at most 2 of 3 views -- which is exactly why the sender-2 reading of
        # wedge 2 could not be confirmed or refuted by wedge 3.
        try:
            peer_snapshot = self._abort_peer_flag_snapshot()
            if peer_snapshot:
                logger.error("%s", peer_snapshot)
        except Exception:  # noqa: BLE001 - an instrument must not mask the abort
            pass
        raise Bar1CollectiveAborted(
            f"barlink-BAR1 rank {self.rank}/{self.world} group "
            f"{self.group or '<unnamed>'}: a spin kernel took its abort path, "
            f"observed at {where}. {attribution}{staged} {cause}"
            " The output buffer of that collective is partially written; "
            "every result computed from it is garbage, which is why this "
            "raises instead of logging. Set "
            f"{barlink_abort_gate.ENV_ENABLE}=0 to restore the previous, "
            "silent behaviour.",
            rank=int(self.rank),
            world=int(self.world),
            group=str(self.group or ""),
            op=op,
            nbytes=nbytes,
            rounds=rounds,
            launches=int(pending),
        )

    def raise_if_aborted(self, label: str) -> None:
        """Raise if a spin kernel took its abort path. NOT for the hot path.

        The bring-up form of the check. ``check_aborted`` is the production
        one (#431 fix 2) and carries rank/op/rounds context; this one is for
        the three byte-level proofs, which know which proof they are in and
        need no context beyond ``label``. Both read the same sticky word.

        Reading the word synchronizes -- that cost is precisely what the
        direct path exists to avoid -- so this belongs at bring-up and on
        wait paths that have already failed, never inside a collective.

        Not gated on ``SGLANG_BARLINK_PEER_LIVENESS``. The kill switch restores
        the previous behaviour of the liveness machinery; it is not a request
        to go back to accepting a half-written buffer as a result. The only
        way to reach this raise is a run that was already broken.
        """
        code = self.status()
        # #622: 1 = a payload barrier gave up, 2 = the entry acknowledgment
        # wait did. Both are abort paths; only the phase differs.
        if code not in (1, 2):
            return
        window = self._abort_window
        reason = window.reason if window is not None and window.tripped else None
        phase = (
            " The abort fired in the ENTRY ack wait (#622): this rank waited "
            "for the peers to consume its PREVIOUS collective's payload."
            if code == 2
            else ""
        )
        raise Bar1KernelAborted(
            f"barlink-BAR1 {label}: a spin kernel took its abort path. Either "
            f"it exceeded SGLANG_BARLINK_BAR1_CAP_CYCLES "
            f"({self.cap_cycles} cycles) waiting for a peer's flag, or "
            f"the host abort word was set"
            + (f" -- {reason}" if reason else "")
            + ". The output buffer of that call is partially written and "
            "must not be used." + phase
        )

    def status(self) -> int:
        """``1`` if a kernel ever hit the time limit, ``2`` (#622) if it hit
        it in the entry acknowledgment wait.

        Both are non-zero and both are sticky; the value only says WHICH spin
        gave up, so every caller that used to test ``== 1`` must test
        membership instead.

        Deliberately a separate query and not on the hot path: it reads a
        device word and thereby synchronizes -- exactly what the direct
        path avoids. Whoever calls it wants to know and pays for it.
        """
        if self._ctl_dev is None:
            return 0
        return int(self._ctl_dev[0].item())

    def _no_collective(self, comm, *args, **kwargs):
        """The stand-in for the collectives still not covered.

        ``*args`` is not a convenience gesture: the seams call with
        different signatures (``barlink_reduce_scatter(comm, inp, dim)``,
        formerly also ``barlink_broadcast(comm, tensor, src)``). With the
        earlier fixed ``(self, comm, inp)``, both produced a ``TypeError``
        before this message even got a chance to run -- so the text was
        unreachable and the cause was recorded nowhere.
        """
        raise NotImplementedError(
            f"The BAR1 transport covers {', '.join(sorted(self.BARLINK_OPS))}. "
            f"reduce_scatter needs a reduction, and the a2a kernel moves "
            f"bytes: it carries all_gather and broadcast for free, and "
            f"reduce_scatter not at all. This line is reachable only if "
            f"someone bypassed handles()."
        )

    # all_gather and broadcast are NO LONGER here. Both assignments used to
    # sit in this list until barlink_all_gather resp. barlink_broadcast were
    # introduced, and would have overwritten the new method -- an
    # assignment in the class body silently wins against a `def` of the
    # same name higher up. Ruff (F811) caught it for all_gather; without
    # that run, every all_gather would have raised NotImplementedError even
    # though handles() had promised otherwise, and the guard would have
    # looked like a transport bug. The same trap was set up for broadcast
    # and is now additionally pinned down by
    # test_barlink_bar1_broadcast.py.
    barlink_reduce_scatter = _no_collective

    # -- Teardown ----------------------------------------------------------

    def close(self) -> None:
        """Tear everything back down. Order matters.

        First the registrations and mappings of the foreign BARs, then the
        attachments (which hold the BAR1 pages), then one's own
        allocation. Reversed, this would pull pages out from under the
        driver while a mapping is still live.
        """
        self._up = False
        # Before anything else: no kernel of this transport will run again,
        # so the abort word has nobody left to talk to. Unregistering it here
        # keeps the watchdog from holding a reference to a closed window.
        # Same argument for the abort gate: `_ctl_dev` is about to be dropped,
        # and a boundary check that reached a torn-down transport would fail
        # in the teardown path instead of reporting on the run.
        if self._registered_in_gate:
            barlink_abort_gate.unregister(self)
            self._registered_in_gate = False
        self._release_abort_window()
        self._peer_table = None
        # Deregister first: the space is on its way back from here on, and
        # a ledger that still reports it occupied after a `close` would
        # groundlessly shortchange a group built later.
        try:
            from sglang.srt.distributed.device_communicators import (
                barlink_matrix_transport as _ledger,
            )

            _ledger.ledger_debit(self.device, self.group)
        except Exception:
            pass
        for z in self._peers.values():
            for a in (z.payload, z.flag):
                if self._cuda is not None:
                    # UNREGISTER AT THE SAME ADDRESS UNDER WHICH IT WAS
                    # REGISTERED. cudaHostUnregister on a pointer INSIDE the
                    # registration fails, and the unregister would silently
                    # not happen -- the aperture would remain registered on
                    # the next run.
                    self._cuda.unregister(a.reg_address)
                try:
                    a.mmap_obj.close()      # type: ignore[attr-defined]
                except Exception:
                    pass
        if self._holder is not None:
            for z in self._peers.values():
                for a in (z.payload, z.flag):
                    self._holder.release(a.holder_handle)
            self._holder.close()
            self._holder = None
        self._peers.clear()
        own_set = set(self._dmabuf_fds)
        for items in self._foreign_fds:
            for fd in items or ():
                if fd is not None and fd >= 0 and fd not in own_set:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
        self._foreign_fds = []
        for fd in self._dmabuf_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self._dmabuf_fds = []
        # ONLY NOW: /dev/nvidiactl is where the RM client that owns the
        # exported memory object is attached. Had it been closed earlier,
        # RM would have released the object out from under the peers'
        # still-live mappings.
        for fd in self._hold_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self._hold_fds = []
        if self._cuda is not None:
            for attr in ("_own", "_own_flag"):
                w = getattr(self, attr)
                if w[2]:
                    self._cuda.vmm_free(*w)
                    setattr(self, attr, (0, 0, 0))
        self._round_dev = None
        self._ctl_dev = None
        # #517: the staged read holds a VIEW of `_ctl_dev`, so dropping the
        # device tensor alone would not free it. The pinned host page and the
        # event go with it.
        self._ctl_defer = False
        self._ctl_src = None
        self._ctl_stage = None
        self._ctl_event = None
        self._ctl_inflight = False
        # #616f: the watchdog poll holds a pinned page and a stream, and it
        # reads `_ctl_dev` directly -- it has to stand down with the word.
        self._abort_poll_active = False
        self._abort_poll_stream = None
        self._abort_poll_dst = None
        self._step_dev = None
        self._result_gen_dev = None


def build_bar1(cpu_group, device, window_bytes: int,
              report: Optional[dict] = None,
              group: str = "") -> Optional[BarlinkBar1Transport]:
    """Factory with a clean fallback.

    ``None`` means: this machine cannot do the direct path, with a logged
    reason. No raising, no silent fallback to another path -- the choice
    of a replacement path is made by the caller, not by this module.

    ``window_bytes`` is the **requested** size of the receive region per
    rank. What actually comes of it is reported afterward by
    ``transport.window_minimum()`` -- and only that belongs in the
    planner.

    ``report`` is the REASON, and it is not an afterthought. Previously,
    every failure ended in a ``logger.info`` and a ``None``, and the
    caller then went on to log "transport=bar1" regardless. That is
    exactly how a measurement was devalued once: the tp group ran over
    BAR1, the dcp group over gloo, and the log looked the same in both
    cases. Whoever passes ``report`` gets ``reason`` and ``stage``
    ("setup", "byte_proof") written into it here and can turn that into a
    loud message.
    """
    if report is None:
        report = {}

    def _aus(stage: str, text: str):
        report["stage"] = stage
        report["reason"] = text
        return None

    try:
        t = BarlinkBar1Transport(cpu_group, device, window_bytes, group=group)
    except Bar1Unavailable as e:
        logger.info("barlink-BAR1: direct path not available -- %s", e)
        return _aus("setup", str(e))
    except NotImplementedError as e:
        logger.info("barlink-BAR1: direct path needs driver work -- %s", e)
        return _aus("setup", f"driver work needed: {e}")
    except Exception as e:                 # a half-built setup is not left standing
        logger.info("barlink-BAR1: setup failed -- %r", e)
        return _aus("setup", f"{type(e).__name__}: {e}")
    # The byte-level proof is part of setup, not an optional extra: without
    # it, `handles` is locked. On this rig, the driver reported peer
    # access for one pair and delivered 4096 of 1,048,576 bytes.
    try:
        proofs = t.byte_proof_all()
    except Exception as e:
        logger.info("barlink-BAR1: byte-level proof could not run -- %r", e)
        t.close()
        return _aus("byte_proof", f"could not run: {type(e).__name__}: {e}")
    if not t._proofs_hold:
        # Previously, the transport came out of here UNSCATHED and only
        # withdrew later via `handles` -- so every collective silently ran
        # over the gloo tier while the log said "transport=bar1". The
        # reason belongs reported to the caller, not withheld.
        failed = sorted(k for k, v in proofs.items() if not v)
        report["stage"] = "byte_proof"
        report["reason"] = (
            f"Byte-level proof failed for the directed pairs {failed}. "
            f"handles() says False for everything; every collective in "
            f"this group runs over the gloo tier."
        )
        report["holds_space"] = True
    # And the same principle for all_to_all -- its own kernel, its own
    # slots, its own flag lines, hence its own proof. It is ONLY attempted
    # if the all_reduce proof holds: a collective over an edge that has
    # already lost bytes there needs no second probe. If it fails,
    # all_reduce remains available regardless; that is why the transport
    # is not torn down here either.
    if t._proofs_hold:
        try:
            t.byte_proof_a2a()
        except Exception as e:
            logger.info(
                "barlink-BAR1: a2a byte-level proof could not run -- %r. "
                "all_to_all withdraws, all_reduce continues.", e,
            )
        # And once more for broadcast: the same kernel, but a different
        # table (exactly one sender, receive offset is not the prefix
        # sum), hence its own proof. It runs ONLY if the a2a proof holds --
        # without that, everything is closed anyway. If it fails,
        # all_reduce, all_to_all, and all_gather remain available.
        if t._a2a_proof:
            try:
                t.byte_proof_broadcast()
            except Exception as e:
                logger.info(
                    "barlink-BAR1: broadcast byte-level proof could not run "
                    "-- %r. broadcast withdraws, the rest continues.", e,
                )
        # And the same principle once more for mesh_pipe: its own kernel,
        # its own slots, its own counter lines, hence its own proof -- and
        # one over MULTIPLE rounds at that, because slot reuse never even
        # gets exercised in a single round. If it fails, mesh and ring
        # remain available.
        if t.pipe_on:
            try:
                t.byte_proof_pipe()
            except Exception as e:
                logger.info(
                    "barlink-BAR1: pipe byte-level proof could not run -- "
                    "%r. mesh_pipe withdraws, mesh and ring continue.", e,
                )
    return t
