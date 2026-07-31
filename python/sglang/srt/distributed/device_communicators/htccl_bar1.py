# SPDX-License-Identifier: Apache-2.0
"""HTCCL sub-path: BAR1 direct transport.

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
``htccl_bar1_ext.py``):

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
``htccl_matrix.py`` already has its own algorithm ``star`` for that. At
20 KiB, ``mesh`` sits at 31.67 us in the same measurement -- the loss
against ``hub`` is small, but the cost of a second memory geometry (R full
buffers instead of chunk slots) is not. Anyone who wants ``star`` today
falls back to the non-BAR1 path; ``handles`` says so with ``False``, it is
never silently substituted with ``mesh``.

Which of the two kernels runs at which size is decided **not** by this
module but by the plan from ``htccl_matrix.py``. That is the conclusion of
the measurement itself: between 80 KiB and 16 MiB, ``MESSUNG_ALLES_IM_SELBEN_LAUF.md``
reports "no clean threshold" (mesh 330.30 vs. ring 326.57 us at 1 MiB), and
the ring advantage at 1 MiB falls within what cannot be distinguished from
noise without repeated runs. Without a plan, the emergency threshold
``SGLANG_HTCCL_BAR1_RING_THRESHOLD`` applies (default 1 MiB) -- a default,
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
  expert varies; ``sende_bytes``/``empfangs_bytes`` arrive per rank. If a
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
    htccl_env_compat,  # noqa: F401  (resolves deprecated env var aliases)
    htccl_liveness,
)
from sglang.srt.distributed.device_communicators.htccl_liveness import (
    PeerLivenessError,
    bounded_barrier,
    bounded_device_sync,
    check_peers,
)

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


#: The values that count as "off" in this transport. Word-for-word the same
#: as ``parallel_state.graph_freigabe_gesetzt`` -- both decide the same
#: thing and must never read differently.
_AUS = ("0", "nein", "aus", "false", "")


def _an(wert: Optional[str]) -> bool:
    """Whether an environment variable counts as set."""
    return wert is not None and wert not in _AUS


def graph_grid_default(umgebung=None) -> bool:
    """May the cooperative launch fire WHILE a graph is being captured?

    **Derived, not independent.** Whether ``cudaLaunchCooperativeKernel`` can
    be captured on this rig is the same question that decides
    ``SGLANG_HTCCL_GRAPH_ENABLE`` -- and it is answered by the same gate
    (``benchmark/bar1_graph_check.py``, case ``grid``). A separate opt-in
    switch next to it would have meant: the gate passes, the release is set,
    and the kernel still falls back to ``1blk``. That is exactly what cost
    the entire BAR1 lead in the lever measurement for #293 once prefill was
    captured (1334.5 -> 1151.6 tok/s at eight sessions; the falsifier with
    ``SGLANG_HTCCL_BAR1_GRAPH_GRID=1`` recovered 1337.2, i.e. +16.1%).

    ``SGLANG_HTCCL_BAR1_GRAPH_GRID`` remains an override in BOTH directions:
    set, it allows the cooperative launch even without the release (this is
    how the gate case ``grid`` itself runs); set to ``0``, it restores the
    old restriction (this is how the gate case ``vorbehalt`` runs). Only
    when it is NOT SET AT ALL does the release decide.
    """
    import os as _os

    if umgebung is None:
        # Live os.environ, not a caller-supplied dict: re-resolve deprecated
        # aliases on every call (not just once at import time), since a
        # caller may set the OLD env var name at runtime, after this module
        # was already imported. See htccl_env_compat.resolve_env_aliases().
        from sglang.srt.distributed.device_communicators import (
            htccl_env_compat as _htccl_env_compat,
        )

        _htccl_env_compat.resolve_env_aliases()
        umgebung = _os.environ
    eigen = umgebung.get("SGLANG_HTCCL_BAR1_GRAPH_GRID")
    if eigen is not None:
        return eigen not in _AUS
    return _an(umgebung.get("SGLANG_HTCCL_GRAPH_ENABLE"))


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

    def vmm_alloc(self, ordinal: int, groesse: int) -> tuple[int, int, int]:
        """``(dptr, handle, groesse)`` -- exportfaehige Geraeteallokation."""
        gran = self.granularitaet(ordinal)
        groesse = ((groesse + gran - 1) // gran) * gran
        handle = ctypes.c_ulonglong(0)
        p = self._prop(ordinal)
        self._d("cuMemCreate", ctypes.byref(handle), ctypes.c_size_t(groesse),
                ctypes.byref(p), ctypes.c_ulonglong(0))
        dptr = ctypes.c_ulonglong(0)
        self._d("cuMemAddressReserve", ctypes.byref(dptr),
                ctypes.c_size_t(groesse), ctypes.c_size_t(gran),
                ctypes.c_ulonglong(0), ctypes.c_ulonglong(0))
        self._d("cuMemMap", ctypes.c_ulonglong(dptr.value),
                ctypes.c_size_t(groesse), ctypes.c_size_t(0), handle,
                ctypes.c_ulonglong(0))
        desc = _CUmemAccessDesc()
        desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        desc.location.id = ordinal
        desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self._d("cuMemSetAccess", ctypes.c_ulonglong(dptr.value),
                ctypes.c_size_t(groesse), ctypes.byref(desc), ctypes.c_size_t(1))
        return int(dptr.value), int(handle.value), groesse

    def vmm_free(self, dptr: int, handle: int, groesse: int) -> None:
        for name, args in (
            ("cuMemUnmap", (ctypes.c_ulonglong(dptr), ctypes.c_size_t(groesse))),
            ("cuMemRelease", (ctypes.c_ulonglong(handle),)),
            ("cuMemAddressFree", (ctypes.c_ulonglong(dptr),
                                  ctypes.c_size_t(groesse))),
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
        wert = ctypes.c_int(0)
        self._d("cuDeviceGetAttribute", ctypes.byref(wert),
                ctypes.c_int(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID), dev)
        return int(wert.value)

    def export_shareable(self, handle: int) -> int:
        """``cuMemExportToShareableHandle`` -- the object fd for the ioctl path."""
        fd = ctypes.c_int(-1)
        self._d("cuMemExportToShareableHandle", ctypes.byref(fd),
                ctypes.c_ulonglong(handle),
                ctypes.c_int(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
                ctypes.c_ulonglong(0))
        return int(fd.value)

    def memset_d8(self, dptr: int, wert: int, n: int) -> None:
        # ``cuMemsetD8_v2``, NOT ``cuMemsetD8``. In cuda.h, the short name is
        # a macro for the _v2 form; going through dlsym/ctypes instead gets
        # you the old ABI entry point with a 32-bit CUdeviceptr, and that one
        # answers 201 (invalid device context) on a current driver -- even
        # when a context is current (measured directly: cuCtxGetCurrent
        # returns a valid context, cuMemsetD8 -> 201, cuMemsetD8_v2 -> 0).
        # This applies to every _v2 function of the driver API.
        self._d("cuMemsetD8_v2", ctypes.c_ulonglong(dptr), ctypes.c_ubyte(wert),
                ctypes.c_size_t(n))

    def dmabuf_fd(self, dptr: int, handle: int, groesse: int,
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
        from sglang.srt.distributed.device_communicators import htccl_bar1_ext

        fd = ctypes.c_int(-1)
        fn = getattr(self.drv, "cuMemGetHandleForAddressRange", None)
        rc = -1
        if fn is not None:
            rc = fn(ctypes.byref(fd), ctypes.c_ulonglong(dptr),
                    ctypes.c_size_t(groesse),
                    ctypes.c_int(CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD),
                    ctypes.c_ulonglong(
                        CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE))
            if rc == 0:
                return int(fd.value), [], "cuMemGetHandleForAddressRange"

        ext = htccl_bar1_ext.load_dmabuf_ext()
        if ext is None:
            raise Bar1Unavailable(
                f"dma-buf export not possible. "
                f"cuMemGetHandleForAddressRange -> "
                f"{'missing from libcuda' if fn is None else rc}, and the "
                f"fallback path via NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD "
                f"+ NV_ESC_EXPORT_TO_DMABUF_FD is not available either: "
                f"{htccl_bar1_ext.dmabuf_reason()}"
            )
        objfd = self.export_shareable(handle)
        try:
            aus = ext.bar1_export_dmabuf(int(objfd), int(self.pci_bus(ordinal)),
                                         int(groesse))
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
        return int(aus[0]), [int(aus[1]), int(aus[2])], "NV_ESC_EXPORT_TO_DMABUF_FD"

    # -- Runtime -----------------------------------------------------------

    def register_io(self, adresse: int, length: int) -> None:
        self._r("cudaHostRegister", ctypes.c_void_p(adresse),
                ctypes.c_size_t(length),
                ctypes.c_uint(CUDA_HOST_REGISTER_IO_MEMORY))

    def unregister(self, adresse: int) -> None:
        try:
            self.rt.cudaHostUnregister(ctypes.c_void_p(adresse))
        except Exception:
            pass

    def dev_ptr(self, host_adresse: int) -> int:
        p = ctypes.c_void_p(0)
        self._r("cudaHostGetDevicePointer", ctypes.byref(p),
                ctypes.c_void_p(host_adresse), ctypes.c_uint(0))
        return int(p.value or 0)

    def memcpy_async(self, ziel: int, source: int, n: int, stream: int) -> None:
        self._r("cudaMemcpyAsync", ctypes.c_void_p(ziel), ctypes.c_void_p(source),
                ctypes.c_size_t(n), ctypes.c_int(CUDA_MEMCPY_DEFAULT),
                ctypes.c_void_p(stream))

    def memcpy(self, ziel: int, source: int, n: int) -> None:
        self._r("cudaMemcpy", ctypes.c_void_p(ziel), ctypes.c_void_p(source),
                ctypes.c_size_t(n), ctypes.c_int(CUDA_MEMCPY_DEFAULT))


# ===========================================================================
# /dev/dmabuf_holder
# ===========================================================================

HALTER_PFAD = os.environ.get("SGLANG_HTCCL_BAR1_HOLDER", "/dev/dmabuf_holder")

_HOLD_FMT = "=iIIBBBBIIQIIQQ"      # struct dmabuf_holder_hold
_HOLD_SIZE = struct.calcsize(_HOLD_FMT)
_REL_FMT = "=II"
_REL_SIZE = struct.calcsize(_REL_FMT)
_MAGIC = 0xDB
_F_BDF_VALID = 1 << 0


def _ioc(richtung: int, typ: int, nr: int, groesse: int) -> int:
    return (richtung << 30) | (groesse << 16) | (typ << 8) | nr


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
    dma_adresse: int
    length: int


class Holder:
    """Keeps foreign dma-bufs alive and returns their BAR1 addresses.

    Without an importer, the NVIDIA driver does not program the BAR1 pages
    at all -- an open fd alone is demonstrably not enough. This used to
    require an RDMA card (``ibv_reg_dmabuf_mr``); the GPL module
    ``dmabuf_holder`` takes over that role without a NIC and additionally
    returns the sg-table, which makes the pattern scan unnecessary.
    """

    def __init__(self, pfad: str = HALTER_PFAD):
        if not os.path.exists(pfad):
            raise Bar1Unavailable(
                f"{pfad} is missing. Without an importer, the driver does "
                f"not map the BAR1 pages of the exported buffer (proven: "
                f"the pattern scan found nothing across 65,536 probes, but "
                f"the hit appeared immediately once the attach happened). "
                f"Load the module from nvidia-smallbar-p2p/dmabuf_holder/. "
                f"NO silent fallback to an RDMA card -- that would be a "
                f"different mode of operation, not the same one."
            )
        try:
            self.fd = os.open(pfad, os.O_RDWR)
        except OSError as e:
            raise Bar1Unavailable(f"{pfad} could not be opened: {e}") from e
        self._handles: list[int] = []

    def hold(self, dmabuf_fd: int, bdf: str,
              max_eintraege: int = 1024) -> tuple[int, list[SgEntry], int]:
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
        handle_, eintraege, total_len, nents = self._hold_once(
            dmabuf_fd, bdf, max_eintraege
        )
        if nents > max_eintraege:
            alt = handle_
            try:
                handle_, eintraege, total_len, nents2 = self._hold_once(
                    dmabuf_fd, bdf, nents
                )
            finally:
                self.release(alt)
            if nents2 > nents:
                self.release(handle_)
                raise Bar1Unavailable(
                    f"The holder reports {nents2} sg-entries, {nents} were "
                    f"requested -- the table grows between two hold calls. "
                    f"Without a complete table, the contiguous length "
                    f"cannot be determined."
                )
        if not eintraege:
            raise Bar1Unavailable(
                "The holder reports 0 sg-entries -- the mapping is empty. "
                "Without an sg-address, the BAR1 offset cannot be "
                "determined; the pattern scan would be the fallback, but it "
                "does not belong in a transport."
            )
        return handle_, eintraege, total_len

    def _hold_once(self, dmabuf_fd: int, bdf: str,
                      max_eintraege: int) -> tuple[int, list[SgEntry], int, int]:
        dom, bus, slot, func = _split_bdf(bdf)
        puffer = ctypes.create_string_buffer(16 * max_eintraege)
        arg = bytearray(struct.pack(
            _HOLD_FMT,
            dmabuf_fd, _F_BDF_VALID, dom, bus, slot, func, 0,
            max_eintraege, 0, ctypes.addressof(puffer),
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
        werte = struct.unpack(_HOLD_FMT, bytes(arg))
        handle_, nents, _dmabuf_size, total_len = werte[10], werte[11], werte[12], werte[13]
        self._handles.append(handle_)
        eintraege = []
        gueltig = min(nents, max_eintraege)
        roh = bytes(puffer.raw[: 16 * gueltig])
        for i in range(gueltig):
            a, l = struct.unpack_from("=QQ", roh, 16 * i)
            eintraege.append(SgEntry(a, l))
        return handle_, eintraege, int(total_len), int(nents)

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
            logger.warning("HTCCL-BAR1: RELEASE(%d) failed: %s", handle_, e)

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
    basis: int
    groesse: int          # gross, per sysfs

    @property
    def end(self) -> int:
        return self.basis + self.groesse


def bar1_window(bdf: str) -> Bar1Window:
    """BAR1 from ``/sys/bus/pci/devices/<bdf>/resource``, line 1.

    NOTE: this is the **gross size** of the aperture. How much of it is
    actually available for peer mappings is unmeasured -- RM reserves part
    of it for itself. ``check_window_requirement`` therefore checks against
    what could actually be exported, not against this number.
    """
    pfad = f"/sys/bus/pci/devices/{bdf}/resource"
    try:
        with open(pfad) as f:
            zeilen = f.read().strip().split("\n")
    except OSError as e:
        raise Bar1Unavailable(f"{pfad} could not be read: {e}") from e
    if len(zeilen) < 2:
        raise Bar1Unavailable(f"{pfad}: no BAR1 line")
    start_s, ende_s, _flags = zeilen[1].split()
    start, end = int(start_s, 16), int(ende_s, 16)
    if end <= start:
        raise Bar1Unavailable(f"{bdf}: BAR1 is empty ({start_s}..{ende_s})")
    return Bar1Window(bdf=bdf, basis=start, groesse=end - start + 1)


SEITE = 4096

#: Largest group for which the kernel arguments have room.
MAX_RANGE = 8


def window_requirement(algorithmus: str, nbytes: int, welt: int) -> int:
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
    if welt < 2:
        return 0
    anteil = -(-nbytes // welt)          # round up
    # ``netz_pipe`` is grouped with mesh and ring here because this function
    # asks about the requirement of ONE payload, and the pipe moves the same
    # 2(R-1)*ceil(N/R) bytes for that. What it actually OCCUPIES in the
    # window is something different: ``2 T (R-1)`` slots of one chunk-piece
    # size each, computed in ``htccl_bar1_pipe_ext.pipe_window_requirement``
    # and additionally checked in ``handles``. That number depends on
    # ``pipe_chunk_bytes``, not on the payload.
    if algorithmus in ("mesh", "netz_pipe", "ring", "hierarchisch"):
        return 2 * (welt - 1) * anteil
    if algorithmus == "star":
        return 2 * (welt - 1) * nbytes
    raise ValueError(f"unknown algorithm {algorithmus!r}")


def geometry(welt: int, max_bytes: int, mit_a2a: bool = True,
              mit_pipe: bool = False, erg_ring: int = 0,
              pipe_bereich: int = 0) -> dict:
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
    ``off_pipe`` netz_pipe slots    ``pipe_bereich`` (absolute)
    ===========  =================  ========================================

    ``chunk_max`` is rounded up to a page -- a slot that begins on a page
    boundary can never share a page with its neighboring slot, so an
    overlong write hits its own page rather than someone else's payload.

    **Why a2a needs 2(R-1) slots and not (R-1).** One slot per sender would
    be enough if the sender knew the receiver had already read the previous
    content. But the flag only says "written". The alternatives are a
    second barrier (half the latency at MoE sizes) or two halves, between
    which the round number alternates. It is two halves; the rationale for
    why two suffice is in the kernel, in ``htccl_bar1_ext.py``.

    **What this costs the all_reduce path.** The region grows from
    ``4(R-1)`` to ``6(R-1)`` slots, so the largest all_reduce payload for a
    given window drops to two-thirds. No measured number changes because of
    this -- only the ceiling above which ``handles`` says False. Anyone who
    wants it back sets ``SGLANG_HTCCL_BAR1_A2A=0``; then ``mit_a2a`` is
    False and the layout is byte-for-byte the old one.

    **Why netz_pipe gets its OWN region and not mesh's.** The regions of
    the different schemes must be pairwise disjoint, and not only within a
    single call. When rank A finishes its round ``n``, rank B may still be
    reading that round's all-gather slot -- before A finishes, it only
    waits on B's flag, not on B's read. This does not show up with
    ``mesh``, because A's next write goes into the RS half while B reads
    from the AG half. A ``netz_pipe`` that used the whole mesh region would
    immediately hit the AG half. A dedicated region makes the question
    moot.

    The region is only created when ``mit_pipe`` is set
    (``SGLANG_HTCCL_BAR1_PIPE=1``); without it, the layout is byte-for-byte
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
    which was wrong: that arm ran ``PIPE_DIREKT=0`` and thus
    ``erg_ring = 0``).

    The region's requirement depends on ``pipe_chunk_bytes``, ``T``, and
    ``R`` -- on nothing that follows from ``max_bytes``. That makes it a
    constant in the fixed-point computation of :func:`max_payload` rather
    than another denominator term. ``pipe_bereich = 0`` keeps the old
    sizing, so a geometry without this argument stays byte-for-byte the
    old one.
    """
    if welt < 2:
        raise ValueError("welt < 2")
    n4_max = max_bytes // 16
    chunk4 = -(-n4_max // welt)
    chunk_max = ((chunk4 * 16 + SEITE - 1) // SEITE) * SEITE
    schlitze = 2 * (welt - 1)
    off_netz = 0
    off_ring = schlitze * chunk_max
    off_a2a = 2 * schlitze * chunk_max
    from sglang.srt.distributed.device_communicators.htccl_bar1_pipe_ext import (
        result_stride_bytes,
    )

    saetze = 2 + (1 if mit_a2a else 0)
    off_pipe = saetze * schlitze * chunk_max
    # The pipe region: the absolute number passed in, otherwise the old full
    # slot set. Rounded up to a page, so the result ring behind it again
    # begins on a page boundary.
    pipe_bereich = int(pipe_bereich) if mit_pipe else 0
    if mit_pipe and pipe_bereich <= 0:
        pipe_bereich = schlitze * chunk_max
    pipe_bereich = ((pipe_bereich + SEITE - 1) // SEITE) * SEITE if pipe_bereich else 0
    off_erg = off_pipe + pipe_bereich
    ring = int(erg_ring) if mit_pipe else 0
    erg_stride = result_stride_bytes(max_bytes) if ring > 0 else 0
    region = off_erg + ring * erg_stride + SEITE
    return {
        "chunk_max": chunk_max,
        "off_netz": off_netz,
        "off_ring": off_ring,
        # -1 explicitly means "does not exist", not "is at 0" -- an offset
        # of 0 would be the mesh region.
        "off_a2a": off_a2a if mit_a2a else -1,
        "a2a_schlitz": chunk_max if mit_a2a else 0,
        "off_pipe": off_pipe if mit_pipe else -1,
        "pipe_bereich": pipe_bereich,
        "off_erg": off_erg if ring > 0 else -1,
        "erg_stride": erg_stride,
        "erg_ring": ring,
        "region_bytes": region,
        "max_bytes": max_bytes,
        "mit_a2a": bool(mit_a2a),
        "mit_pipe": bool(mit_pipe),
    }


def flags_requirement(welt: int, mit_a2a: bool = True,
                   mit_pipe: bool = False) -> int:
    """``(2 + 2(R-1) [+ 1]) * R * 256`` bytes, plus ``5 R * 256`` for the pipe.

    One 256-byte line per (topology, step, sender): no false sharing between
    senders, none between steps, none between topologies. Mesh has 2 steps,
    ring ``2(R-1)``, a2a exactly **one**. At R=8 that is 34 KiB, well below
    an allocation granularity.

    ``netz_pipe`` appends five lines per rank at the end (``tailRS``,
    ``tailAG``, ``headRS``, ``headAG``, ``ergBereit``) -- **independent of
    K and T**, because it is a sliding window with one counter per
    connection, not a flag per chunk. Appended at the end so every existing
    line offset stays byte-for-byte the same.
    """
    from sglang.srt.distributed.device_communicators.htccl_bar1_pipe_ext import (
        pipe_flags_extra,
    )

    grund = (2 + 2 * (welt - 1) + (1 if mit_a2a else 0)) * welt * 256
    return grund + (pipe_flags_extra(welt) if mit_pipe else 0)


def fbasis_a2a(welt: int) -> int:
    """Offset of the a2a flag lines within the flag region.

    Behind mesh and ring, so the two measured topologies stay byte-for-byte
    where they were. The same computation does NOT appear a second time in
    the kernel -- it is passed in as an argument, because a second version
    would be exactly the place where sender and receiver end up pointing at
    different lines.
    """
    return (2 + 2 * (welt - 1)) * welt * 256


def ag_plan(laengen, slot: int) -> list:
    """The round decomposition of an ``all_gather``. Pure arithmetic.

    ``laengen[i]`` is rank ``i``'s shard in **bytes**; the result is their
    concatenation, i.e. ``sum(laengen)`` bytes, with rank ``i`` at offset
    ``sum(laengen[:i])``.

    Delivered is, per round, a list of ``(send_offset, length,
    receive_offset)`` per rank -- all in bytes, all absolute, nothing left
    to be guessed as a prefix sum:

    * ``send_offset`` points into the CALLER'S OWN shard (the same slice for
      every destination -- that is exactly what distinguishes all_gather
      from all_to_all),
    * ``receive_offset`` points into the result, i.e. ``basis[i] + k*slot``.

    **Why rounds at all.** A shard can be larger than a slot. The failure
    case from the handoff is exactly that: 10,600,448 bytes of all_gather
    against an a2a slot of just under 8 MiB with a 96-MiB window. Instead of
    withdrawing via ``handles`` -- which aborts the run during a CUDA graph
    capture, because there is no fallback path -- the shard runs in
    ``ceil(max(laengen)/slot)`` rounds.

    **Why this survives a capture.** The round count depends only on
    ``laengen`` and ``slot``. Both are group-wide identical and constant
    for a captured shape, so the number of kernel launches is baked in and
    the same on every replay -- the same argument that lets
    ``htccl_device.all_reduce`` capture its slot loop. No host code here
    decides anything per round that could change between capture and
    replay. That is the difference from the pipe's direct mode, whose
    host-side ring index fails on exactly this point (see
    ``_result_slot``).

    **Rank-uniform.** Every rank computes from the SAME ``laengen`` vector,
    so all of them end up with the same number of rounds. If a rank counted
    differently, that would not be an error but a hang: the others would
    wait in the barrier of a round it no longer runs.

    **Unequal shards** are arithmetic here, not a rewrite. Today's seam
    (``HTCCLCommunicator.all_gather``) is uniform -- its result is
    ``(R,) + form``, which CANNOT be uneven, and the uneven form is called
    ``all_gatherv`` in sglang and explicitly not covered under HTCCL. This
    function nonetheless takes a vector: under uneven TP, unequal shards are
    the normal case, and the place where a uniform distribution is ASSUMED
    is the place where a later ``all_gatherv`` silently gets wrong offsets.
    A rank whose shard ends earlier gets length 0 in the remaining rounds --
    it rides along in the barrier without moving any bytes.
    """
    laengen = [int(x) for x in laengen]
    if not laengen:
        return []
    if slot <= 0:
        raise ValueError(f"slot size {slot} is not positive")
    if any(n < 0 for n in laengen):
        raise ValueError(f"negative shard length in {laengen}")
    basis, acc = [], 0
    for n in laengen:
        basis.append(acc)
        acc += n
    runden = max(1, -(-max(laengen) // slot))
    plan = []
    for k in range(runden):
        eine = []
        for i, n in enumerate(laengen):
            a = min(k * slot, n)
            b = min((k + 1) * slot, n)
            eine.append((a, b - a, basis[i] + a))
        plan.append(eine)
    return plan


def ar_plan(nbytes: int, chunk_max: int, welt: int) -> list:
    """The round decomposition of an ``all_reduce``. Pure arithmetic.

    Delivered is, per round, an ``(offset, length)`` in **bytes**. Each
    round is a complete all_reduce over a slice of the buffer -- the same
    kernel, the same decomposition into ``welt`` shards, just on fewer
    bytes.

    **Why this exists.** The kernel decomposes a payload into ``welt``
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
    would be to fill every round up to ``chunk_max*welt`` and put the
    remainder in the last one. That produces a tail that can become
    arbitrarily small -- and the extension insists on ``n4 >= R`` (one
    128-bit packet per rank, ``TORCH_CHECK`` in the host). A leftover round
    of 16 bytes across three ranks would not be a slow case but an abort.
    Evenly distributed, the smallest round can lie at most ONE packet below
    the largest.

    **Rank-uniform and capture-safe.** The round count depends solely on
    ``nbytes``, ``chunk_max``, and ``welt``. All three are group-wide
    identical and constant for a captured shape, so the number of kernel
    launches is baked in -- the same argument as for :func:`ag_plan` and
    :func:`bc_plan`.
    """
    nbytes = int(nbytes)
    if welt < 2:
        raise ValueError(f"welt {welt} is smaller than 2")
    if chunk_max < 16:
        raise ValueError(f"chunk_max {chunk_max} cannot carry a packet")
    if nbytes < 0:
        raise ValueError(f"negative payload {nbytes}")
    if nbytes % 16:
        raise ValueError(f"payload {nbytes} is not a multiple of 16")
    if nbytes == 0:
        return []
    pakete = nbytes // 16
    # Packets per rank and round -- the size the slot depends on.
    je_rang_max = chunk_max // 16
    max_pakete = je_rang_max * welt
    runden = -(-pakete // max_pakete)
    basis, rest = divmod(pakete, runden)
    plan = []
    versatz = 0
    for k in range(runden):
        p = basis + (1 if k < rest else 0)
        length = p * 16
        plan.append((versatz, length))
        versatz += length
    return plan


def a2a_rounds(groesster_block: int, slot: int) -> int:
    """Round count for an ``all_to_all``, from the LARGEST block.

    ``groesster_block`` is the maximum over all ``R*R`` blocks, not over the
    caller's own row -- the seam computes it group-wide before asking
    (``HTCCLCommunicator.all_to_all_single``). That is exactly why the round
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
    if groesster_block < 0:
        raise ValueError(f"negative block {groesster_block}")
    return max(1, -(-int(groesster_block) // int(slot)))


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
    runden = -(-nbytes // slot)
    return [
        (k * slot, min((k + 1) * slot, nbytes) - k * slot)
        for k in range(runden)
    ]


def max_payload(welt: int, region_bytes: int, mit_a2a: bool = True,
                 mit_pipe: bool = False, erg_ring: int = 0,
                 pipe_bereich: int = 0) -> int:
    """Largest payload whose slots fit into a region of this size.

    Inverse of :func:`geometry`. Deliberately rounded conservatively and
    then checked by re-computing forward -- an inverse that is off by one
    page would otherwise only surface on the hot path. This forward check
    is exactly why there does not need to be a second version of the
    factor computation here: ``geometry`` itself has the final say.
    """
    if welt < 2 or region_bytes <= SEITE:
        return 0
    # The pipe region is an ABSOLUTE number as soon as it is passed in -- it
    # depends on `pipe_chunk_bytes`, T, and R, not on `chunk_max`. It is
    # therefore subtracted and does not appear in the denominator. Without
    # it, the old sizing remains (a full slot set, i.e. 2(R-1) in the
    # denominator).
    absolut = int(pipe_bereich) if (mit_pipe and pipe_bereich > 0) else 0
    if absolut:
        absolut = ((absolut + SEITE - 1) // SEITE) * SEITE
    # 2 sets for mesh, 2 for ring, 2 each for a2a and the pipe -- so 4 as
    # the base, not 2. Spelled out instead of "(6 if a2a else 4)", so the
    # fourth term does not disappear back into a single number.
    schlitze = (4 + (2 if mit_a2a else 0)
                + (2 if (mit_pipe and not absolut) else 0)) * (welt - 1)
    ring = int(erg_ring) if mit_pipe else 0
    # The result ring costs ``L * roundup(N, SEITE)``, and ``N`` is
    # ``chunk_max * R``. In units of chunk_max that is ``L * R`` additional
    # units on top of ``schlitze`` -- which is why the ring appears here IN
    # THE DENOMINATOR and not as a subtraction. A subtraction would have
    # placed the initial value far enough off that the forward check below
    # would have had to search downward in 32-byte steps.
    nenner = schlitze + ring * welt
    rest = region_bytes - SEITE - absolut
    if rest <= 0:
        return 0
    chunk_max = (rest // nenner // SEITE) * SEITE
    if chunk_max <= 0:
        return 0
    n = (chunk_max // 16) * welt * 16
    while n > 0 and geometry(welt, n, mit_a2a, mit_pipe, ring,
                              absolut)["region_bytes"] > region_bytes:
        n -= welt * 16
    return n


# ===========================================================================
# fd exchange via SCM_RIGHTS
# ===========================================================================


def _exchange_fds(cpu_group, rank: int, welt: int,
                 eigene_fds: list[int]) -> list[list[int]]:
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

    traeger = [None]
    if rank == 0:
        traeger = [tempfile.mkdtemp(prefix="htccl-bar1-")]
    # torch runs the object collectives inline and ignores async_op, so there
    # is no Work to bound. A one-shot check before the call names a peer that
    # is already gone instead of entering the 7200 s gloo wait for it.
    check_peers("bar1 fd exchange: broadcast of the socket directory")
    dist.broadcast_object_list(
        traeger, src=dist.get_global_rank(cpu_group, 0), group=cpu_group
    )
    verz = str(traeger[0])
    pfad = os.path.join(verz, f"r{rank}.sock")

    anzahl = len(eigene_fds)
    fds: list[list[int]] = [[] for _ in range(welt)]
    fds[rank] = list(eigene_fds)
    horcher = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        if os.path.exists(pfad):
            os.unlink(pfad)
        horcher.bind(pfad)
        horcher.listen(welt)
        bounded_barrier(cpu_group, "bar1 fd exchange: sockets bound")

        for besitzer in range(welt):
            if besitzer == rank:
                for _ in range(welt - 1):
                    verb, _ = horcher.accept()
                    with verb:
                        socket.send_fds(verb, [b"x"], list(eigene_fds))
            else:
                ziel = os.path.join(verz, f"r{besitzer}.sock")
                verb = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                letzter: Optional[Exception] = None
                for _ in range(200):        # the peer may still be binding
                    try:
                        verb.connect(ziel)
                        letzter = None
                        break
                    except OSError as e:
                        letzter = e
                        time.sleep(0.01)
                if letzter is not None:
                    # The 2 s cap above already bounds this loop. What it
                    # cannot do is say WHY: a peer that died before binding
                    # its socket looks exactly like one that is merely slow.
                    check_peers(f"bar1 fd exchange: connect to rank {besitzer}")
                    raise Bar1Unavailable(
                        f"fd exchange: {ziel} unreachable ({letzter})"
                    )
                with verb:
                    _daten, empfangen, _fl, _adr = socket.recv_fds(
                        verb, 1, anzahl
                    )
                if len(empfangen) != anzahl:
                    raise Bar1Unavailable(
                        f"fd exchange: rank {besitzer} sent {len(empfangen)} "
                        f"fds instead of {anzahl}"
                    )
                fds[besitzer] = list(empfangen)
            bounded_barrier(
                cpu_group, f"bar1 fd exchange: round {besitzer} complete"
            )
    finally:
        horcher.close()
        try:
            os.unlink(pfad)
        except OSError:
            pass
    return fds


# ===========================================================================
# The transport
# ===========================================================================


@dataclass
class Mapping:
    """A mapped and registered foreign BAR1 region."""

    bar1_basis: int
    bar1_versatz: int          # region's offset within BAR1
    length: int                # ACTUALLY mapped, contiguous length
    mmap_obj: object           # held so the mapping stays alive
    reg_adresse: int           # address under which REGISTRATION happened
    host_adresse: int          # user-space address of the region (reg + lead-in)
    dev_ptr: int               # THIS card's device pointer to the foreign BAR
    halter_handle: int


@dataclass
class PeerTarget:
    """What setup established for this peer -- immutable from then on.

    Two regions per peer, in separate VMM allocations and thus exported and
    mapped separately: the payload slots and the flag lines. Exactly the
    arrangement the probe measured.
    """

    rang: int
    bdf: str
    nutz: Mapping
    flag: Mapping
    byte_proof: bool = False

    # The two names under which the point-to-point path (put/pair) knows
    # the payload region. Kept so the measurement probe stays unchanged.
    @property
    def dev_ptr(self) -> int:
        return self.nutz.dev_ptr

    @property
    def length(self) -> int:
        return self.nutz.length


class HTCCLBar1Transport:
    """BAR1 direct transport.

    Implements the transport seam from ``htccl.py`` (lines 67-80):
    ``handles(op, nbytes) -> bool`` plus ``htccl_<op>(comm, ...)`` for each
    operation offered.

    What it offers, and which parts of it are measured:

    * ``htccl_all_reduce`` over the ported kernels ``mesh`` and ``ring``.
      Fully measured in the probe (float32, three ranks, rig 1); the table
      is in the module docstring.
    * ``put(ziel, quell_ptr, nbytes, versatz)`` -- a single write into the
      destination card's BAR.
    * ``pair``/``pair_receive`` -- the measurement probe that
      ``htccl_matrix.HTCCLMatrixPlanner`` needs for real edge capacities
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
    #: (on the same kernel, unmeasured -- see :meth:`htccl_all_gather`), and
    #: broadcast (the same kernel again, see :meth:`htccl_broadcast`).
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
    #: For reduce_scatter, the loud guard in ``htccl._select`` remains
    #: responsible. It names the operation, and because this set here is the
    #: single source of truth about what is covered, the message cannot go
    #: stale.
    #:
    #: Both spellings of all_to_all appear here because the seam in
    #: htccl.py asks for the operation under the name ``all_to_all``, while
    #: the only real caller in sglang (GroupCoordinator, equal split) is
    #: named ``all_to_all_single``. Two names, one path -- better than a
    #: rename at the seam that gets overlooked when reading.
    HTCCL_OPS: frozenset = frozenset(
        {"all_reduce", "all_gather", "all_to_all", "all_to_all_single",
         "broadcast"}
    )

    def __init__(self, cpu_group, device, fenster_bytes: int,
                 aktiviert: Optional[bool] = None, gruppe: str = ""):
        import torch
        import torch.distributed as dist

        self.cpu_group = cpu_group
        self.device = device
        #: Name of the communicator group ("tp", "dcp", ...). It lives here
        #: because BAR1 is a PROCESS-WIDE resource: whatever this group
        #: pins down is unavailable to the next one. Without the name,
        #: there would be no way to either book it or say who holds the
        #: space.
        self.gruppe = gruppe
        self.rank = dist.get_rank(cpu_group)
        self.welt = dist.get_world_size(cpu_group)
        self.fenster_bytes = int(fenster_bytes)
        self._auf = False
        self._peers: dict[int, PeerTarget] = {}
        self._halter: Optional[Holder] = None
        self._cuda: Optional[_Cuda] = None
        self._eigen = (0, 0, 0)          # payload: dptr, handle, size
        self._eigen_flag = (0, 0, 0)     # flags:  dptr, handle, size
        self._eigen_fuehler = None
        self._dmabuf_fds: list[int] = []       # own, exported
        self._halte_fds: list[int] = []        # /dev/nvidiactl, /dev/nvidiaN
        self._fremde_fds: list[list[int]] = []
        self._ext = None
        self._geo: dict = {}
        self._plan = None                      # optional plan from htccl_matrix
        # Capability, group-wide uniform. Only valid after _build_up.
        self._fenster_minimum = 0
        self._belege_stehen = False
        self._runde_dev = None
        self._ctl_dev = None
        # Peer liveness. Both stay None when SGLANG_HTCCL_PEER_LIVENESS=0 or
        # when the identity exchange fails; every use site then falls back to
        # the behaviour this transport had before task #312.
        self._peer_table = None
        self._abort_window = None

        if aktiviert is None:
            aktiviert = os.environ.get("SGLANG_HTCCL_MATRIX_DIRECT", "1") not in (
                "0", "nein", "aus", "false"
            )
        if not aktiviert:
            raise Bar1Unavailable(
                "disabled via SGLANG_HTCCL_MATRIX_DIRECT=0"
            )
        if self.welt > MAX_RANGE:
            raise Bar1Unavailable(
                f"{self.welt} ranks, but the kernel arguments hold at most "
                f"{MAX_RANGE}. The limit lives in htccl_bar1_ext.py "
                f"(HTCCL_BAR1_MAX_RANKS) and should be raised there in a "
                f"traceable way -- not worked around here."
            )
        self.ordinal = device.index if getattr(device, "index", None) is not None \
            else torch.cuda.current_device()
        # Operating parameters of the kernels. All rank-uniform, like every
        # other SGLANG_HTCCL* variable.
        self.threads = int(os.environ.get("SGLANG_HTCCL_BAR1_THREADS", "256"))
        # ~30 s at 2 GHz -- a stalled peer gets caught by a deadline in the
        # kernel instead of occupying the card indefinitely. Same order of
        # magnitude as HTCCLDeviceTransport._TIMEOUT_CYCLES, for the same
        # reason.
        self.deckel_zyklen = int(
            os.environ.get("SGLANG_HTCCL_BAR1_CAP_CYCLES", "60000000000")
        )
        # Flag load shape: 2 = ld.mmio.relaxed.sys (the only genuine
        # cache-bypass, the probe's default), 0 = ld.global.cv.
        self.ladeform = int(os.environ.get("SGLANG_HTCCL_BAR1_LOAD_SHAPE", "2"))
        # Read fence: only needed when payload and flag sit on different
        # PCIe targets. Here they do not; default off.
        self.fluss = int(os.environ.get("SGLANG_HTCCL_BAR1_FLOW", "0"))
        # Payload size from which the cooperative multi-block launch kicks
        # in. 4 MiB, because in MESSUNG_ALLES_IM_SELBEN_LAUF.md the 'grid'
        # variant wins from 4 MiB up and '1blk' wins below it.
        self.gitter_ab = int(
            os.environ.get("SGLANG_HTCCL_BAR1_GRID_THRESHOLD", str(4 << 20))
        )
        # May the cooperative variant be launched WHILE a CUDA graph is being
        # captured? The default comes from SGLANG_HTCCL_GRAPH_ENABLE -- the
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
        self.graph_gitter = graph_grid_default()
        self._graph_gitter_gemeldet = False
        # Emergency mesh->ring threshold, if no plan is passed in.
        self.ring_ab = int(
            os.environ.get("SGLANG_HTCCL_BAR1_RING_THRESHOLD", str(1 << 20))
        )
        self.min_bytes = int(os.environ.get("SGLANG_HTCCL_BAR1_MIN_BYTES", "4096"))
        self.max_bytes = 0
        # all_to_all occupies a third slot set in the same region and thus
        # costs a third of the largest all_reduce payload (see `geometry`).
        # Rank-uniform like every other SGLANG_HTCCL* variable; 0 restores
        # the old memory layout byte-for-byte.
        self.a2a_an = os.environ.get("SGLANG_HTCCL_BAR1_A2A", "1") not in (
            "0", "nein", "aus", "false"
        )
        #: Only valid after `byte_proof_a2a`. Without a passed proof,
        #: all_to_all withdraws -- all_reduce is unaffected by this.
        self._a2a_beleg = False

        # -- netz_pipe (pipelined mesh, htccl_bar1_pipe_ext) ----------------
        # OFF by default. Enabled, it occupies another slot set and four
        # flag lines per rank; disabled, every number and every offset in
        # this module is byte-for-byte the measured one.
        self.pipe_an = os.environ.get("SGLANG_HTCCL_BAR1_PIPE", "0") not in (
            "0", "nein", "aus", "false"
        )
        # RING DEPTH T -- slots per phase and connection. 4, from NCCL:
        # NCCL_STEPS 8 (src/include/device.h:26) divided by
        # ALLREDUCE_SLICESTEPS 2 (src/include/collectives.h:19).
        self.pipe_t = int(os.environ.get("SGLANG_HTCCL_BAR1_PIPE_T", "4"))
        # SCHEDULE LEAD P -- by how many loop rounds sending runs ahead of
        # reducing. SEPARATE from the ring depth, and that separation is
        # exactly the timing decoupling: the receiver may lag behind by
        # `T - P + 1` loop rounds before the sender blocks. With P = T that
        # would be exactly ONE round, i.e. effectively lockstep -- and on a
        # rig with x4, x8, and x8 links and three different card models,
        # the skew between unevenly fast ranks is exactly what the window
        # is meant to absorb. P = 2 is the minimum that pipelines at all;
        # with T = 4 that is three rounds of skew.
        self.pipe_vorlauf = int(
            os.environ.get("SGLANG_HTCCL_BAR1_PIPE_LEAD", "2")
        )
        # Chunk count K. 0 = automatic, from `pipe_chunk_bytes`.
        self.pipe_k = int(os.environ.get("SGLANG_HTCCL_BAR1_PIPE_K", "0"))
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
            os.environ.get("SGLANG_HTCCL_BAR1_PIPE_CHUNK_BYTES", str(1 << 20))
        )
        self.pipe_k_max = int(os.environ.get("SGLANG_HTCCL_BAR1_PIPE_K_MAX", "64"))
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
        self.pipe_schlitz_kib = int(
            os.environ.get("SGLANG_HTCCL_BAR1_PIPE_SLOT_KIB", "0")
        )
        #: Only fixed during setup (from `pipe_schlitz_kib` or computed).
        self.pipe_schlitz = 0
        # SEPARATE 1blk/grid threshold for the pipe.
        #
        # Separate from `gitter_ab`, because the calculation for the pipe
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
        self.pipe_gitter_ab = int(
            os.environ.get("SGLANG_HTCCL_BAR1_PIPE_GRID_THRESHOLD",
                           str(self.gitter_ab))
        )
        # Receiver acknowledgment (head). 1 = on. Turned off, it shows what
        # the sliding window costs; turned ON, it may only be used by
        # someone who has read the schedule proof in htccl_bar1_pipe_ext.
        self.pipe_quittung = int(
            os.environ.get("SGLANG_HTCCL_BAR1_PIPE_ACK", "1")
        )
        # Payload size from which netz_pipe runs instead of mesh. 256 KiB,
        # because below that a single chunk would remain and the pipe would
        # then just be the mesh's bookkeeping.
        self.pipe_ab = int(
            os.environ.get("SGLANG_HTCCL_BAR1_PIPE_THRESHOLD", str(256 << 10))
        )
        # Direct mode: the all-gather writes into the receiver's result
        # buffer instead of into a slot the receiver would then have to read
        # out and copy over. Default ON as soon as the pipe is on -- that is
        # the whole point of the pipe. 0 is the control run with the same
        # memory layout.
        self.pipe_direkt = os.environ.get(
            "SGLANG_HTCCL_BAR1_PIPE_DIRECT", "1"
        ) not in ("0", "nein", "aus", "false")
        # Direct mode WHILE a graph is being captured. Default OFF -- not
        # because it wouldn't work, but because it changes the memory layout
        # (flag family 4) and requires a larger result ring. Enabled, this
        # holds: per captured call site, ONE reserved ring slot from the
        # pool above the eager slots, plus the release handshake in the
        # kernel. The derivation lives in `_result_slot` and in
        # `htccl_bar1_pipe_ext.result_slot_split`.
        self.pipe_direkt_graph = os.environ.get(
            "SGLANG_HTCCL_BAR1_PIPE_DIRECT_GRAPH", "0"
        ) not in ("0", "nein", "aus", "false")
        self._direkt_graph_gemeldet = False
        # How many result buffers the ring holds. Costs L*max_bytes in the
        # BAR window; 2 is the minimum with which round n does not write
        # into the buffer the caller still holds from round n-1.
        self.pipe_erg_ring = int(
            os.environ.get("SGLANG_HTCCL_BAR1_PIPE_RESULT_RING", "2")
        )
        # How many ring slots the EAGER path retains. Default 2, i.e.
        # unchanged. The knob exists because the number is a property of
        # the caller: it must be as large as the number of results the
        # caller keeps alive at the same time. Under the graph-safe direct
        # mode, the failure occurs during the capture WARMUP, and that runs
        # eager -- a larger SGLANG_HTCCL_BAR1_PIPE_RESULT_RING there
        # allocates exclusively graph slots and does not help.
        from sglang.srt.distributed.device_communicators.htccl_bar1_pipe_ext import (  # noqa: E501
            ERG_EAGER_PLAETZE,
        )

        self.pipe_erg_eager = int(
            os.environ.get("SGLANG_HTCCL_BAR1_PIPE_RESULT_EAGER",
                           str(ERG_EAGER_PLAETZE))
        )
        #: Only valid after `byte_proof_pipe`.
        self._pipe_beleg = False
        self._pipe_ext = None
        self._schritt_dev = None
        self._erg_gen_dev = None
        #: Running index into the result ring. HOST-SIDE and rank-uniform,
        #: because every rank sees the same sequence of collectives (SPMD)
        #: -- the same assumption `algorithm_for` already relies on. The
        #: kernel CANNOT choose it itself: the host must build the result
        #: tensor before the kernel runs.
        self._erg_i = -1
        #: Weak references to the most recently handed-out result tensors,
        #: per ring slot. They are the liveness check.
        self._erg_lebt: list = []
        #: Running number of eager direct calls, and the number at which a
        #: slot was last assigned. From that falls out the reuse distance
        #: the release handshake needs as ``ergSlack`` -- under strict
        #: rotation it is the number of eager slots, fewer after skipping
        #: an occupied slot.
        self._erg_zaehler = 0
        self._erg_zuletzt: list = []
        #: How often an eager call found no free slot and therefore ran
        #: ``direkt=0``. Reported once per rank; the counter itself keeps
        #: going, so "once at warmup" and "on every call" can be told
        #: apart.
        self._erg_eager_voll = 0
        self._erg_eager_voll_gemeldet = False
        #: Split of the result ring into eager slots and graph slots. Fixed
        #: before the first call runs (`result_slot_split`), so the graph
        #: pool can never grab a slot whose eager tensor the caller still
        #: holds.
        self._erg_eager_plaetze = 0
        self._erg_graph_plaetze = 0
        #: How many graph slots have already been assigned. Only grows; a
        #: slot once assigned never comes back, because from here there is
        #: no way to tell whether the graph it belongs to is still alive.
        self._erg_graph_vergeben = 0
        self._erg_graph_leer_gemeldet = False
        #: Lower bound for all_to_all. Deliberately NOT `min_bytes` (4096):
        #: the whole appeal of a2a over BAR1 lies precisely in the small
        #: MoE dispatch blocks. 16 bytes = one packet.
        self.a2a_min_bytes = int(
            os.environ.get("SGLANG_HTCCL_BAR1_A2A_MIN_BYTES", "16")
        )
        #: all_gather over the a2a kernel. DEFAULT ON, and that is
        #: deliberate: without it, the standard run aborts during graph
        #: capture (the guard in htccl._select, correct and loud). The
        #: switch exists so a benchmarker can pit it against the gloo tier
        #: -- and because a new hot-path route needs an off switch that
        #: does not require a recompile. It only takes effect within
        #: SGLANG_HTCCL_TRANSPORT=bar1|matrix; without HTCCL it changes
        #: nothing.
        self.ag_an = os.environ.get("SGLANG_HTCCL_BAR1_AG", "1") not in (
            "0", "nein", "aus", "false"
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
            os.environ.get("SGLANG_HTCCL_BAR1_AG_MIN_BYTES", "1")
        )
        #: How many rounds a shard may cost at most. Not a window limit but
        #: a round limit: each round is one kernel launch with one barrier.
        #: 16 carries a shard of ~128 MiB at a slot of just under 8 MiB
        #: (96-MiB window, R=3), and thus every size that occurs in this
        #: model -- the largest measured is 10.6 MB. Above that, the path
        #: withdraws instead of presenting a loop as a transport.
        self.ag_max_runden = int(
            os.environ.get("SGLANG_HTCCL_BAR1_AG_MAX_ROUNDS", "16")
        )
        #: broadcast over the same a2a kernel. DEFAULT ON, for the same
        #: reason as all_gather: without it, the standard run aborts while
        #: capturing the draft graph (eagle_worker_v2.init_cuda_graphs ->
        #: parallel_state broadcast -> the guard in htccl._select). The
        #: switch exists so a benchmarker can pit it against the gloo tier.
        self.bc_an = os.environ.get("SGLANG_HTCCL_BAR1_BC", "1") not in (
            "0", "nein", "aus", "false"
        )
        #: Lower bound: **1 byte**, i.e. none at all. This used to be 16
        #: ("one packet"), copied from a2a, and that was wrong -- the
        #: standard run sends broadcast with 12 BYTES, and it failed on
        #: that right after the 128-byte case had just gotten through.
        #:
        #: The 16 had no technical reason. The kernel does write in 16-byte
        #: packets, but assembles the last, incomplete one from the
        #: available bytes in a register (``packeBytes``) -- a 12-byte
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
        #: ``a2a_schlitz * bc_max_runden``. Whoever turns the knob up is
        #: declining on purpose; it no longer happens silently.
        self.bc_min_bytes = int(
            os.environ.get("SGLANG_HTCCL_BAR1_BC_MIN_BYTES", "1")
        )
        #: Round limit as with all_gather, for the same reason: each round
        #: is one kernel launch with one barrier.
        self.bc_max_runden = int(
            os.environ.get("SGLANG_HTCCL_BAR1_BC_MAX_ROUNDS", "16")
        )
        #: Round limit for all_reduce and all_to_all -- the same kind of
        #: limit as ag/bc and for the same reason: each round is one kernel
        #: launch with one barrier, and arbitrarily many of those per
        #: collective would not be a transport but a loop. 16 carries an
        #: all_reduce payload of ~384 MiB at an 8188-KiB slot and R=3, and
        #: thus every size that occurs in this model -- the standard run's
        #: working point is 20 MiB.
        self.ar_max_runden = int(
            os.environ.get("SGLANG_HTCCL_BAR1_AR_MAX_ROUNDS", "16")
        )
        self.a2a_max_runden = int(
            os.environ.get("SGLANG_HTCCL_BAR1_A2A_MAX_ROUNDS", "16")
        )
        #: Only valid after `byte_proof_broadcast`. Its own flag even
        #: though the same kernel runs: if the broadcast proof fails, that
        #: is not a verdict on all_to_all -- the table is a different one
        #: (exactly one sender), and a failure there carries no conclusion
        #: over here.
        self._bc_beleg = False
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

        ``RMSmallBarP2PPeerBar1`` widens the guard from "within the static
        window of another GPU" to "within the BAR1 aperture of another
        GPU". The default is **0**; without the reg key, the guard behaves
        exactly as before, and ``cudaHostRegister(..., IoMemory)`` on a
        foreign BAR fails.

        Only what is in ``/proc/driver/nvidia/params`` is reported. An
        empty ``RegistryDwords`` entry means: the reg key is not set. That
        is not proof that "the path is dead" -- the proof is the failed
        ``cudaHostRegister``, and that is exactly what setup waits for.
        """
        aus = {"regkeys": "", "treiber": "", "halter": os.path.exists(HALTER_PFAD)}
        try:
            with open("/proc/driver/nvidia/params") as f:
                for z in f:
                    # Exactly "RegistryDwords:" -- not
                    # "RegistryDwordsPerDevice:", which would otherwise
                    # overwrite the real one as a later line and report a
                    # reg key that is actually set as empty.
                    if z.startswith("RegistryDwords:"):
                        aus["regkeys"] = z.strip()
        except OSError:
            pass
        try:
            with open("/proc/driver/nvidia/version") as f:
                aus["treiber"] = f.readline().strip()
        except OSError:
            pass
        aus["smallbar_p2p_peerbar1"] = "RMSmallBarP2PPeerBar1" in aus["regkeys"]
        return aus

    # -- Setup -----------------------------------------------------------

    def _build_up(self) -> None:
        import torch
        import torch.distributed as dist

        from sglang.srt.distributed.device_communicators.htccl_matrix import (
            bdf_of_card,
        )

        if self.welt < 2:
            raise Bar1Unavailable("fewer than two ranks -- nothing to do")

        # Peer liveness, before the first collective of the bring-up. From
        # here on every host wait in this transport can decide whether a peer
        # that has not arrived still exists, and the spin kernels get a host
        # word they can be told to abort through. Returns None when the
        # feature is off; every use site is guarded on that.
        self._peer_table = htccl_liveness.install(self.cpu_group)
        self._install_abort_window()

        t0 = time.perf_counter()
        self._cuda = _Cuda()
        self._halter = Holder()

        eigener_bdf = bdf_of_card(self.device)
        # BDF and window proposal in ONE all_gather. The proposal has to
        # travel along because the cards in the group have differently
        # sized apertures (3080: 256 MiB gross) and, in a process with two
        # groups, a different amount of it may already be spoken for. A
        # region that differs per rank would mean a different slot layout
        # per rank -- not an error, but writes landing at the wrong place.
        # Hence: a group-wide MINIMUM, and that decides.
        gesammelt: list[object] = [None] * self.welt
        # torch runs the object collectives inline, so there is no Work to
        # bound; the one-shot check names a peer that is already gone instead
        # of letting gloo wait 7200 s for it.
        check_peers("bar1 bring-up: BDF and window exchange", self._peer_table)
        dist.all_gather_object(
            gesammelt, (eigener_bdf, int(self.fenster_bytes)),
            group=self.cpu_group,
        )
        self.bdfs = [str(x[0]) for x in gesammelt]      # type: ignore[index]
        vorschlaege = [int(x[1]) for x in gesammelt]    # type: ignore[index]
        gemeinsam = min(vorschlaege)
        if gemeinsam != self.fenster_bytes:
            logger.warning(
                "HTCCL-BAR1: per-rank window proposals %s MiB -- the "
                "group-wide minimum of %d MiB governs. This rank could "
                "have done %d MiB. The region is rank-uniform because the "
                "slot offsets in both kernels are computed from it.",
                [v // 2**20 for v in vorschlaege], gemeinsam // 2**20,
                self.fenster_bytes // 2**20,
            )
        if gemeinsam <= 0:
            raise Bar1Unavailable(
                "0 bytes of BAR1 window remain group-wide. Another "
                "communicator in this process has claimed the aperture; "
                "the calculation is in the warning from "
                "htccl_matrix_transport.window_for. Either give the other "
                "group less (SGLANG_HTCCL_BAR1_FENSTER_MIB_<NAME>) or run "
                "this group explicitly over NCCL."
            )
        self.fenster_bytes = gemeinsam

        # 0. The kernels. First, because a failed build is cheaper to abort
        # than a half-built peer table.
        from sglang.srt.distributed.device_communicators import htccl_bar1_ext

        try:
            self._ext = htccl_bar1_ext.load_collective_ext(self.cpu_group)
        except Exception as e:
            raise Bar1Unavailable(
                f"The collective extension could not be compiled: {e}"
            ) from e

        # 0b. The pipelined kernel, if enabled. A failed build disables it
        # instead of losing the whole transport -- mesh and ring are
        # unaffected by this.
        if self.pipe_an:
            from sglang.srt.distributed.device_communicators import (
                htccl_bar1_pipe_ext,
            )

            try:
                self._pipe_ext = htccl_bar1_pipe_ext.load_pipe_ext(self.cpu_group)
            except Exception as e:
                logger.warning(
                    "HTCCL-BAR1: the pipelined extension could not be "
                    "compiled (%s). netz_pipe drops out; mesh and ring "
                    "continue unchanged.", e,
                )
                self.pipe_an = False
                self._pipe_ext = None

        # 1. Memory layout. The largest payload follows from the window the
        # caller grants -- not the other way around.
        if self.pipe_an and not (2 <= self.pipe_vorlauf <= self.pipe_t):
            raise Bar1Unavailable(
                f"SGLANG_HTCCL_BAR1_PIPE_LEAD={self.pipe_vorlauf} does not "
                f"fit T={self.pipe_t}: 2 <= P <= T is required. P=1 "
                f"deadlocks (sending and consuming a chunk would fall into "
                f"the same loop round), P>T would let the schedule overtake "
                f"the slots."
            )
        pipe_bereich = 0
        if self.pipe_an:
            from sglang.srt.distributed.device_communicators.htccl_bar1_pipe_ext import (  # noqa: E501
                pipe_range_bytes,
                pipe_slot_default,
            )

            if self.pipe_schlitz_kib > 0:
                self.pipe_schlitz = (self.pipe_schlitz_kib * 1024 // 16) * 16
            else:
                self.pipe_schlitz = pipe_slot_default(
                    self.welt, self.pipe_chunk_bytes
                )
            if self.pipe_schlitz <= 0:
                raise Bar1Unavailable(
                    f"Pipe slot size of {self.pipe_schlitz} bytes is not "
                    f"usable (from SGLANG_HTCCL_BAR1_PIPE_SLOT_KIB="
                    f"{self.pipe_schlitz_kib} resp. PIPE_CHUNK_BYTES="
                    f"{self.pipe_chunk_bytes} at {self.welt} ranks)."
                )
            pipe_bereich = pipe_range_bytes(
                self.welt, self.pipe_t, self.pipe_schlitz
            )
            logger.info(
                "HTCCL-BAR1-PIPE: ring depth T=%d, lead P=%d -- a peer may "
                "lag behind by %d loop rounds before the sender blocks. "
                "Direct mode %s, result ring L=%d, pipe slot %d KiB (%s), "
                "pipe region %.1f MiB.",
                self.pipe_t, self.pipe_vorlauf,
                self.pipe_t - self.pipe_vorlauf + 1,
                "on" if self.pipe_direkt else "off", self.pipe_erg_ring,
                self.pipe_schlitz // 1024,
                "set" if self.pipe_schlitz_kib > 0
                else f"from chunk target {self.pipe_chunk_bytes // 1024} KiB",
                pipe_bereich / 2**20,
            )
        if not self.pipe_an or not self.pipe_direkt:
            self.pipe_erg_ring = 0
        elif self.pipe_erg_eager < 2:
            raise Bar1Unavailable(
                f"SGLANG_HTCCL_BAR1_PIPE_RESULT_EAGER={self.pipe_erg_eager}: "
                f"direct mode needs at least two eager result buffers. With "
                f"only one, round n would write into exactly the buffer the "
                f"caller still holds from round n-1 -- a silently "
                f"overwritten result tensor, i.e. wrong numbers without a "
                f"crash. Whoever doesn't want the ring can disable direct "
                f"mode with SGLANG_HTCCL_BAR1_PIPE_DIRECT=0."
            )
        elif self.pipe_erg_ring < self.pipe_erg_eager:
            raise Bar1Unavailable(
                f"SGLANG_HTCCL_BAR1_PIPE_RESULT_RING={self.pipe_erg_ring} is "
                f"smaller than SGLANG_HTCCL_BAR1_PIPE_RESULT_EAGER="
                f"{self.pipe_erg_eager}. The ring holds the eager slots and "
                f"the graph pool together; it cannot be smaller than its "
                f"eager part."
            )
        max_bytes = max_payload(self.welt, self.fenster_bytes, self.a2a_an,
                                 self.pipe_an, self.pipe_erg_ring,
                                 pipe_bereich)
        if max_bytes < self.min_bytes:
            raise Bar1Unavailable(
                f"Window of {self.fenster_bytes // 1024} KiB carries only "
                f"{max_bytes} bytes of payload at {self.welt} ranks, "
                f"minimum size is {self.min_bytes}. 4(R-1) slots of "
                f"ceil(N/R) each must fit."
            )
        self._geo = geometry(self.welt, max_bytes, self.a2a_an, self.pipe_an,
                              self.pipe_erg_ring, pipe_bereich)
        from sglang.srt.distributed.device_communicators.htccl_bar1_pipe_ext import (  # noqa: E501
            result_slot_split,
        )

        self._erg_eager_plaetze, self._erg_graph_plaetze = result_slot_split(
            self.pipe_erg_ring, self.pipe_direkt_graph, self.pipe_erg_eager
        )
        self._erg_lebt = [None] * max(0, self._erg_eager_plaetze)
        self._erg_zuletzt = [None] * max(0, self._erg_eager_plaetze)
        if self.pipe_direkt_graph:
            logger.info(
                "HTCCL-BAR1-PIPE: graph-safe direct mode, result ring L=%d "
                "split into %d eager slots and %d graph slots. Each "
                "captured call site takes ONE graph slot and does not give "
                "it back; if the pool is empty, the capture runs the "
                "direct=0 path. More graphs need a larger "
                "SGLANG_HTCCL_BAR1_PIPE_RESULT_RING.",
                self.pipe_erg_ring, self._erg_eager_plaetze,
                self._erg_graph_plaetze,
            )
        self.max_bytes = max_bytes
        region = self._geo["region_bytes"]
        flaggen = flags_requirement(self.welt, self.a2a_an, self.pipe_an)

        # 2. Two receive regions, two exports. Separate, because the probe
        # measured them separately.
        dptr, handle, groesse = self._cuda.vmm_alloc(self.ordinal, region)
        self._eigen = (dptr, handle, groesse)
        fptr, fhandle, fgroesse = self._cuda.vmm_alloc(self.ordinal, flaggen)
        self._eigen_flag = (fptr, fhandle, fgroesse)
        # Flags to 0. Rounds start at 1, so no old marker can pass as a
        # valid acknowledgment.
        self._cuda.memset_d8(fptr, 0, fgroesse)

        weg = ""
        for adr, hnd, gr in ((dptr, handle, groesse), (fptr, fhandle, fgroesse)):
            fd, hold, weg = self._cuda.dmabuf_fd(adr, hnd, gr, self.ordinal)
            self._dmabuf_fds.append(fd)
            self._halte_fds.extend(hold)

        # 3. Exchange fds -- both in one message.
        self._fremde_fds = _exchange_fds(
            self.cpu_group, self.rank, self.welt, self._dmabuf_fds
        )

        # 4.-6. attach each peer, offset from the sg-table, map, register.
        # This happens EXACTLY HERE and never again.
        for peer in range(self.welt):
            if peer == self.rank:
                continue
            self._peers[peer] = self._bind_peer(peer, self._fremde_fds[peer])

        # 7. What is ACTUALLY mapped -- the group-wide minimum. Not the
        # gross size from sysfs and not the requested one: what governs is
        # the contiguous length the holder actually reported per peer. A
        # rank whose smallest window is smaller decides for everyone --
        # otherwise `handles` would answer differently per rank and the
        # SPMD assumption would be violated.
        lokal_min = min(z.nutz.length for z in self._peers.values())
        lokal_flag_min = min(z.flag.length for z in self._peers.values())
        traeger: list[object] = [None] * self.welt
        check_peers("bar1 bring-up: window minimum exchange", self._peer_table)
        dist.all_gather_object(
            traeger, (lokal_min, lokal_flag_min), group=self.cpu_group
        )
        self._fenster_minimum = min(int(x[0]) for x in traeger)   # type: ignore[index]
        flag_minimum = min(int(x[1]) for x in traeger)            # type: ignore[index]
        if self._fenster_minimum < region:
            raise Bar1Unavailable(
                f"{region} bytes of payload region were requested, at most "
                f"{self._fenster_minimum} bytes are mapped contiguously "
                f"group-wide. No silent shrinking of the payload: the slot "
                f"offsets are fixed in both kernels, and a rank with a "
                f"different layout would write to the wrong place."
            )
        if flag_minimum < flaggen:
            raise Bar1Unavailable(
                f"Flag region: {flaggen} bytes needed, at most "
                f"{flag_minimum} mapped group-wide."
            )

        # 8. Round counter and status word. Both LOCAL in VRAM -- they are
        # never touched by a peer.
        self._runde_dev = torch.zeros(1, dtype=torch.int64, device=self.device)
        self._ctl_dev = torch.zeros(2, dtype=torch.int32, device=self.device)
        # Absolute chunk counter of the sliding window. Separate from the
        # round counter, because it grows by K per call and is only
        # advanced by netz_pipe -- it is the reference against which the
        # peers' head/tail lines are compared, and must therefore remain
        # absolute across calls. Rank-uniform, because every rank sees the
        # same sequence of calls with the same K.
        self._schritt_dev = torch.zeros(1, dtype=torch.int64, device=self.device)
        # Generation counter of the graph-safe direct mode. Also LOCAL in
        # VRAM: local accesses are coherent with one's own reads, and only
        # the PEER's view of the counter value needs the flag protocol --
        # that is carried by flag family 4 in the window. On the device and
        # not on the host, because it must keep counting on every graph
        # replay; a host counter gets baked in at capture time and then
        # sits still.
        self._erg_gen_dev = torch.zeros(1, dtype=torch.int64,
                                        device=self.device)

        bounded_barrier(
            self.cpu_group,
            "bar1 bring-up: peer targets bound",
            table=self._peer_table,
        )
        self._auf = True
        # Into the ledger. Only NOW, because only now is it established that
        # the aperture actually gave up the space -- booking before the
        # holder would be a promise on spec, and the second group's ENOMEM
        # would then come from a reservation that does not actually exist.
        from sglang.srt.distributed.device_communicators import (
            htccl_matrix_transport as _kasse,
        )

        _kasse.ledger_credit(self.device, self.gruppe, region + flaggen)
        dauer = time.perf_counter() - t0
        logger.info(
            "HTCCL-BAR1: setup in %.0f ms, %d peer targets, region %.1f MiB "
            "per rank (%s), slot %d KiB, largest payload %d KiB, flags %d "
            "bytes, export via %s. From here on, nothing is mapped anymore "
            "on the hot path.",
            dauer * 1000, len(self._peers), region / 2**20,
            f"{(6 if self.a2a_an else 4) * (self.welt - 1)} slots"
            + (" (of which 2(R-1) for all_to_all)" if self.a2a_an
               else ", all_to_all disabled"),
            self._geo["chunk_max"] // 1024,
            max_bytes // 1024, flaggen, weg,
        )
        # And log the ledger too. Without it, the next group that fails
        # with ENOMEM is again reduced to guessing.
        logger.info(
            "HTCCL-BAR1: BAR1 ledger of this card after group %r: %s.",
            self.gruppe or "<unnamed>",
            ", ".join(f"{g or '<unnamed>'}: {b / 2**20:.1f} MiB"
                      for g, b in _kasse.ledger_balance(self.device)),
        )

    def _bind_peer(self, peer: int, fremde_fds: list) -> PeerTarget:
        """Attach, map, and register both regions of a peer."""
        ziel_bdf = self.bdfs[peer]
        nutz = self._bind_region(peer, ziel_bdf, fremde_fds[0], "payload",
                                  self._geo["region_bytes"])
        try:
            flag = self._bind_region(peer, ziel_bdf, fremde_fds[1], "flag",
                                      flags_requirement(self.welt, self.a2a_an,
                                                     self.pipe_an))
        except BaseException:
            # Release the already-bound payload region again: it is not yet
            # recorded in any PeerTarget and would not be found by close().
            self._resolve_region(nutz)
            raise
        return PeerTarget(rang=peer, bdf=ziel_bdf, nutz=nutz, flag=flag)

    def _resolve_region(self, a: Mapping) -> None:
        if self._cuda is not None:
            self._cuda.unregister(a.reg_adresse)
        try:
            a.mmap_obj.close()              # type: ignore[attr-defined]
        except Exception:
            pass
        if self._halter is not None:
            self._halter.release(a.halter_handle)

    def _bind_region(self, peer: int, ziel_bdf: str, fremder_fd: int,
                      kind: str, mindestens: int) -> Mapping:
        assert self._cuda is not None and self._halter is not None
        window = bar1_window(ziel_bdf)

        # Attach happens as THIS card -- it will write later.
        handle_, sg, total = self._halter.hold(fremder_fd, self.bdfs[self.rank])

        treffer = [e for e in sg if window.basis <= e.dma_adresse < window.end]
        if not treffer:
            self._halter.release(handle_)
            raise Bar1Unavailable(
                f"None of the {len(sg)} sg-addresses of rank {peer} lie "
                f"within its BAR1 [{window.basis:#x}, {window.end:#x}). "
                f"First address {sg[0].dma_adresse:#x}. This means either "
                f"that the IOMMU does not map identically (in which case "
                f"the offset derived from the sg-table does not hold and "
                f"the pattern scan would be needed), or that the driver did "
                f"not map into BAR1 at all. No guessing -- this edge is "
                f"dropped."
            )
        treffer.sort(key=lambda e: e.dma_adresse)
        start = treffer[0].dma_adresse
        # Contiguous? Only the contiguous beginning can be mapped as one
        # piece; the rest would be a second window.
        length = 0
        erwartet = start
        for e in treffer:
            if e.dma_adresse != erwartet:
                break
            length += e.length
            erwartet += e.length
        versatz = start - window.basis
        if length < mindestens:
            self._halter.release(handle_)
            raise Bar1Unavailable(
                f"{kind} region of rank {peer} ({ziel_bdf}): "
                f"{mindestens} bytes needed, but only {length} bytes are "
                f"mapped CONTIGUOUSLY in BAR1 (from {len(treffer)} "
                f"sg-entries starting at {start:#x}). This is the length "
                f"that is checked against -- not the gross size from sysfs "
                f"({window.groesse} bytes), of which RM helps itself first."
            )

        seite = mmap.PAGESIZE
        m_versatz = (versatz // seite) * seite
        vorlauf = versatz - m_versatz
        m_laenge = length + vorlauf
        res = f"/sys/bus/pci/devices/{ziel_bdf}/resource1_wc"
        try:
            res_fd = os.open(res, os.O_RDWR | os.O_SYNC)
        except OSError as e:
            self._halter.release(handle_)
            raise Bar1Unavailable(
                f"{res} could not be opened ({e}). Without a "
                f"write-combining aperture there is no direct path."
            ) from e
        try:
            # ONLY the needed slice: an mmap over a 32-GiB window fails with
            # EINVAL (measured on the 5090).
            abb = mmap.mmap(res_fd, m_laenge, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE, offset=m_versatz)
        except (OSError, ValueError) as e:
            self._halter.release(handle_)
            raise Bar1Unavailable(
                f"mmap({res}, length={m_laenge}, offset={m_versatz:#x}) "
                f"failed: {e}"
            ) from e
        finally:
            os.close(res_fd)

        host = ctypes.addressof(ctypes.c_char.from_buffer(abb)) + vorlauf
        try:
            self._cuda.register_io(host - vorlauf, m_laenge)
        except Bar1Unavailable as e:
            abb.close()
            self._halter.release(handle_)
            ps = self.patch_state()
            if not ps["smallbar_p2p_peerbar1"]:
                grund = (
                    f"This is the expected outcome WITHOUT the widened "
                    f"guard: the reg key RMSmallBarP2PPeerBar1 defaults to "
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
                grund = (
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
                f"{peer} ({ziel_bdf}) failed: {e}. {grund} "
                f"The transport withdraws, it does not force anything."
            ) from e
        dev = self._cuda.dev_ptr(host - vorlauf)
        return Mapping(
            bar1_basis=window.basis, bar1_versatz=versatz, length=length,
            mmap_obj=abb, reg_adresse=host - vorlauf, host_adresse=host,
            dev_ptr=dev + vorlauf, halter_handle=handle_,
        )

    # -- Window computation --------------------------------------------------

    def check_window_requirement(self, algorithmus: str, nbytes: int) -> None:
        """Requirement against what could ACTUALLY be exported.

        Not against the gross size from sysfs: the 3080s report 256 MiB
        BAR1 gross, but how much of that is net available for peer mappings
        is unmeasured -- RM reserves part of it for itself. What governs is
        the length the holder actually mapped contiguously per peer.
        """
        noetig = window_requirement(algorithmus, nbytes, self.welt)
        for peer, z in sorted(self._peers.items()):
            brutto = bar1_window(z.bdf).groesse
            if noetig > z.nutz.length:
                raise Bar1Unavailable(
                    f"Window too small for '{algorithmus}' at "
                    f"{nbytes // 1024} KiB and {self.welt} ranks: needed "
                    f"{noetig // 1024} KiB, but only {z.nutz.length // 1024} "
                    f"KiB mapped at rank {peer} ({z.bdf}) (BAR1 gross "
                    f"{brutto // 2**20} MiB). Either chunk smaller or "
                    f"exclude this edge. Mesh and ring BOTH need 2(R-1) "
                    f"slots -- the ring is no way out here."
                )

    def window_minimum(self) -> int:
        """Smallest actually mapped payload region in the GROUP.

        This is the number that belongs in the planner as
        ``fenster_bytes``: a **capability**, determined from what the
        holder actually mapped contiguously per peer, minimized across all
        ranks. A value that differs per rank would yield a different plan
        per rank, and the collectives' SPMD assumption depends on that not
        happening.
        """
        return self._fenster_minimum

    def algorithm_for(self, nbytes: int) -> str:
        """``mesh``, ``netz_pipe``, or ``ring`` for this size.

        The plan from ``htccl_matrix.py`` takes precedence if one was
        passed in (``set_plan``). It is group-wide identical -- checked via
        the plan checksum -- and thus the only source that can keep this
        choice rank-uniform.

        Without a plan, the emergency threshold
        ``SGLANG_HTCCL_BAR1_RING_THRESHOLD``. It is a default, not a
        measured conclusion: between 1 and 16 MiB, mesh and ring differ by
        1 to 7 percent in the probe, and the finding says explicitly "no
        clean threshold -- the planner should measure this, not hard-code
        it".

        **THE ONE PLACE where netz_pipe is chosen.** The planner does not
        know ``netz_pipe`` and should not, for now: its cost models are
        calibrated against the two measured topologies. As long as
        ``SGLANG_HTCCL_BAR1_PIPE`` is off -- and that is the default --
        this method gives byte-for-byte the same answer as before.
        """
        if self._plan is not None:
            a = self._plan.algorithm_for(nbytes)
            # 'star' and 'hierarchisch' are not ported here; they never
            # reach this point via handles() in the first place.
        else:
            a = "ring" if nbytes >= self.ring_ab else "mesh"
        if (self.pipe_an and a == "mesh" and nbytes >= self.pipe_ab
                and self._pipe_k(nbytes) is not None):
            return "netz_pipe"
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
        n = min(probe_bytes, self._eigen[2])
        ergebnis: dict[tuple[int, int], bool] = {}
        rueck = torch.empty(n, dtype=torch.uint8, pin_memory=True)
        for source in range(self.welt):
            for ziel in range(self.welt):
                if source == ziel:
                    continue
                marke = (source * 251 + ziel * 37 + 1) & 0xFF
                paar = f"{source}->{ziel}"
                bounded_barrier(
                    self.cpu_group,
                    f"bar1 byte proof {paar}: before clearing",
                    table=self._peer_table,
                )
                if self.rank == ziel:
                    # Clear the destination first, so a buffer that was NOT
                    # written does not accidentally look like a hit.
                    leer = torch.full((n,), (marke ^ 0xFF) & 0xFF,
                                      dtype=torch.uint8, pin_memory=True)
                    self._cuda.memcpy(self._eigen[0], leer.data_ptr(), n)
                bounded_barrier(
                    self.cpu_group,
                    f"bar1 byte proof {paar}: destination cleared",
                    table=self._peer_table,
                )
                if self.rank == source:
                    muster = torch.full((n,), marke, dtype=torch.uint8,
                                        device=self.device)
                    self.put(ziel, muster.data_ptr(), n, 0)
                    bounded_device_sync(
                        f"bar1 byte proof {paar}: pattern written",
                        device=self.device,
                        table=self._peer_table,
                    )
                bounded_barrier(
                    self.cpu_group,
                    f"bar1 byte proof {paar}: pattern in place",
                    table=self._peer_table,
                )
                ok = True
                if self.rank == ziel:
                    self._cuda.memcpy(rueck.data_ptr(), self._eigen[0], n)
                    schlecht = int((rueck != marke).sum().item())
                    ok = schlecht == 0
                    if ok:
                        # The passed proof belongs in the log too: "0 of N
                        # bytes wrong" is the statement every later timing
                        # measurement rests on.
                        logger.info(
                            "HTCCL-BAR1: byte-level proof %d->%d passed: 0 "
                            "of %d bytes wrong.", source, ziel, n,
                        )
                    else:
                        logger.warning(
                            "HTCCL-BAR1: byte-level proof %d->%d FAILED: %d "
                            "of %d bytes wrong. This edge is struck, "
                            "regardless of what the driver reports.",
                            source, ziel, schlecht, n,
                        )
                traeger: list[object] = [ok if self.rank == ziel else None]
                check_peers(
                    f"bar1 byte proof {paar}: verdict broadcast",
                    self._peer_table,
                )
                dist.broadcast_object_list(
                    traeger, src=dist.get_global_rank(self.cpu_group, ziel),
                    group=self.cpu_group,
                )
                ergebnis[(source, ziel)] = bool(traeger[0])
        for (q, z), ok in ergebnis.items():
            if q == self.rank and z in self._peers:
                self._peers[z].byte_proof = ok
        # ONE answer, group-wide. `ergebnis` is identical on every rank
        # (every entry was distributed from the destination), so this is
        # also rank-uniform -- exactly what `handles` needs.
        self._belege_stehen = all(ergebnis.values())
        if not self._belege_stehen:
            gefallen = sorted(k for k, v in ergebnis.items() if not v)
            logger.warning(
                "HTCCL-BAR1: byte-level proof failed for %s. The "
                "collectives withdraw (handles -> False); a collective over "
                "an edge that loses bytes would not be a collective.",
                gefallen,
            )
        return ergebnis

    # -- Data path -------------------------------------------------------

    def put(self, ziel: int, quell_ptr: int, nbytes: int, versatz: int = 0,
            stream: Optional[int] = None) -> None:
        """A write into the destination card's BAR. Posted, hence fast.

        There is deliberately **no** ``get``: reading from a foreign BAR is
        non-posted and expensive (measured on the 2080 Ti at 1132 MB/s out
        versus 3254 MB/s in). Hence the rule "everyone pushes for
        themselves".
        """
        if not self._auf:
            raise Bar1Unavailable("transport not set up")
        z = self._peers.get(ziel)
        if z is None:
            raise Bar1Unavailable(f"no peer target for rank {ziel}")
        if versatz + nbytes > z.length:
            raise Bar1Unavailable(
                f"put({ziel}): {versatz}+{nbytes} exceeds the mapped "
                f"window of {z.length} bytes. The caller must chunk; "
                f"automatic re-mapping on the hot path is excluded -- it is "
                f"exactly the expensive part."
            )
        assert self._cuda is not None
        if stream is None:
            import torch

            stream = torch.cuda.current_stream(self.device).cuda_stream
        self._cuda.memcpy_async(z.dev_ptr + versatz, quell_ptr, nbytes, stream)

    # -- Measurement probe for htccl_matrix -------------------------------

    def name(self) -> str:
        return "bar1"

    def self_load(self, nbytes: int, richtung: str) -> float:
        from sglang.srt.distributed.device_communicators.htccl_matrix import (
            SelfLoadSensor,
        )

        if getattr(self, "_eigen_fuehler", None) is None:
            self._eigen_fuehler = SelfLoadSensor(self.device, max_bytes=4 << 20)
        return self._eigen_fuehler.self_load(nbytes, richtung)

    def self_load_duplex(self, nbytes: int) -> Optional[float]:
        """Deliberately ``None``.

        Full duplex over the direct path is NOT measurable via host memory,
        and the relaxed driver guard exists because of a documented
        full-duplex deadlock (bug 1571948). As long as counter-traffic has
        not been verified over a full collective's duration, nothing is
        reported here that looks like a green light.
        """
        return None

    def pair(self, ziel: int, nbytes: int) -> Optional[float]:
        """Directed edge GB/s -- only if the byte-level proof holds."""
        import torch

        if not self._auf:
            return None
        if ziel < 0:                       # planner's capability query
            return 0.0 if self._peers else None
        z = self._peers.get(ziel)
        if z is None or not z.byte_proof:
            return None
        n = min(nbytes, z.length)
        source = torch.empty(n, dtype=torch.uint8, device=self.device)
        for _ in range(8):
            self.put(ziel, source.data_ptr(), n, 0)
        bounded_device_sync(
            f"bar1 pair probe {self.rank}->{ziel}: warm-up",
            device=self.device,
            table=self._peer_table,
        )
        runden = 64 if n <= 65536 else 16
        t0 = time.perf_counter()
        for _ in range(runden):
            self.put(ziel, source.data_ptr(), n, 0)
        # Inside the timed section, so NOT bounded_device_sync: that one naps
        # up to 50 ms between polls, and a 150 ms transfer would report a
        # third less bandwidth than it delivers. Spinning on the bare event
        # predicate keeps the number honest and still ends on a dead peer.
        # With the feature off this is the plain synchronize it always was.
        if htccl_liveness.liveness_enabled():
            marke = f"bar1 pair probe {self.rank}->{ziel}: timed writes"
            ereignis = torch.cuda.Event()
            ereignis.record(torch.cuda.current_stream(self.device))
            htccl_liveness.bounded_poll(
                ereignis.query,
                marke,
                table=self._peer_table,
                sleep=False,
                on_abort=self._wait_abort(marke),
            )
        torch.cuda.synchronize(self.device)
        dt = time.perf_counter() - t0
        return (runden * n) / dt / 1e9 if dt > 0 else 0.0

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
        reconciled sizes (``_belege_stehen`` comes from a distribution per
        directed pair, ``_fenster_minimum`` and ``max_bytes`` from an
        ``all_gather``, the thresholds from rank-uniform environment
        variables). Two ranks must never answer differently here -- one
        would run into the collective and the other would not, and the
        result would be a hang instead of an error.

        The data type is NOT a factor: ``handles`` does not see it. The
        extension accepts float32/float16/bfloat16 and rejects everything
        else with a reason. This is the same state as in ``htccl_device``.
        """
        if op not in self.HTCCL_OPS:
            return False
        if not self._auf or self._ext is None:
            return False
        if not self._belege_stehen:
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
        ``groesster_chunk > chunk_max``, and the payload fell back to the
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
        if nbytes // 16 < self.welt:
            # Fewer than one packet per rank -- the chunk decomposition
            # would leave ranks empty-handed, and the host rejects it.
            return False
        chunk_max = int(self._geo.get("chunk_max", 0))
        if chunk_max < 16:
            return False
        # The round limit replaces the old size cap. It is not a window
        # limit but a limit on kernel launches.
        runden = ar_plan(nbytes, chunk_max, self.welt)
        if len(runden) > self.ar_max_runden:
            return False
        # The LARGEST round is checked. It is the only one that could
        # fail -- and it is the same group-wide.
        groesste = max(length for _, length in runden)
        if groesste > self.max_bytes:
            return False
        # Does the LARGEST chunk of ONE ROUND fit into a slot? This is the
        # condition the mapping actually depends on -- checked, not
        # inferred from `nbytes <= max_bytes`. The extension computes it a
        # second time, there with its own chunk bounds rather than this
        # formula: a seam checked on both sides with the same wrong formula
        # would not stand out.
        groesster_chunk = -(-(groesste // 16) // self.welt) * 16
        if groesster_chunk > chunk_max:
            return False
        # The algorithm is decided PER ROUND, so it is checked per round.
        # For a single round, this is byte-for-byte the old question.
        for _, length in runden:
            algo = self.algorithm_for(length)
            if algo not in ("mesh", "netz_pipe", "ring"):
                # 'star' and 'hierarchisch' are not ported here. No silent
                # fallback to 'mesh'.
                return False
            if algo == "netz_pipe" and not self._pipe_supports(length):
                return False
            # And the same requirement again, in the planner's currency,
            # against the group-wide SMALLEST actually mapped length.
            # Redundant as long as setup completed successfully -- and
            # exactly for that reason cheap: this line catches anyone who
            # touches the region size without carrying the window concept
            # along.
            if window_requirement(algo, length, self.welt) > self._fenster_minimum:
                return False
        return True

    def ar_rounds(self, nbytes: int) -> int:
        """Round count for ``nbytes`` -- for logging and tests."""
        chunk_max = int(self._geo.get("chunk_max", 0))
        if chunk_max < 16 or nbytes <= 0 or nbytes % 16:
            return 0
        return len(ar_plan(nbytes, chunk_max, self.welt))

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
        if op not in self.HTCCL_OPS:
            return f"{op} is not in HTCCL_OPS"
        if not self._auf or self._ext is None:
            return "the direct path is not set up"
        if not self._belege_stehen:
            return "the byte-level proof per pair does not hold"
        slot = int(self._geo.get("a2a_schlitz", 0))
        chunk_max = int(self._geo.get("chunk_max", 0))
        if op in ("all_to_all", "all_to_all_single"):
            if not self.a2a_an:
                return "all_to_all is disabled via SGLANG_HTCCL_BAR1_A2A=0"
            if not self._a2a_beleg:
                return "the a2a byte-level proof does not hold"
            if nbytes < self.a2a_min_bytes:
                return f"{nbytes} bytes are below a2a_min_bytes ({self.a2a_min_bytes})"
            n = a2a_rounds(-(-nbytes // self.welt), slot) if slot else 0
            if n > self.a2a_max_runden:
                return (f"would need {n} rounds at a {slot}-byte slot, "
                        f"{self.a2a_max_runden} are allowed")
        elif op == "all_gather":
            if not self.ag_an:
                return "all_gather is disabled via SGLANG_HTCCL_BAR1_AG=0"
            if not self._a2a_beleg:
                return "the a2a byte-level proof does not hold (all_gather rides on it)"
            if nbytes < self.ag_min_bytes:
                return f"{nbytes} bytes are below ag_min_bytes ({self.ag_min_bytes})"
            if slot and -(-nbytes // slot) > self.ag_max_runden:
                return (f"would need {-(-nbytes // slot)} rounds at a "
                        f"{slot}-byte slot, {self.ag_max_runden} are allowed")
        elif op == "broadcast":
            if not self.bc_an:
                return "broadcast is disabled via SGLANG_HTCCL_BAR1_BC=0"
            if not self._bc_beleg:
                return "the broadcast byte-level proof does not hold"
            if nbytes < self.bc_min_bytes:
                return f"{nbytes} bytes are below bc_min_bytes ({self.bc_min_bytes})"
            if slot and -(-nbytes // slot) > self.bc_max_runden:
                return (f"would need {-(-nbytes // slot)} rounds at a "
                        f"{slot}-byte slot, {self.bc_max_runden} are allowed")
        else:
            if nbytes < self.min_bytes:
                return f"{nbytes} bytes are below min_bytes ({self.min_bytes})"
            if nbytes % 16:
                return (f"{nbytes} bytes are not a multiple of 16 -- the "
                        f"kernel's access width is 128 bits")
            if nbytes // 16 < self.welt:
                return (f"{nbytes} bytes are fewer than one 128-bit packet "
                        f"per rank ({self.welt})")
            if chunk_max >= 16:
                n = len(ar_plan(nbytes, chunk_max, self.welt))
                if n > self.ar_max_runden:
                    return (f"would need {n} rounds at a chunk bound of "
                            f"{chunk_max} bytes, {self.ar_max_runden} are "
                            f"allowed")
        if self._geo.get("region_bytes", 0) > self._fenster_minimum:
            return (f"the region ({self._geo.get('region_bytes')} bytes) does "
                    f"not fit into the group-wide smallest mapped window "
                    f"({self._fenster_minimum} bytes)")
        return ""

    def _kernel(self, bewegt: int, schwelle: int, wo: str) -> int:
        """``1`` = cooperative multi-block launch (``grid``), ``0`` = ``1blk``.

        **The one place where this choice is made** -- previously, each of
        the three collectives computed `bewegt >= schwelle` for itself, and
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
        if bewegt < schwelle:
            return 0
        if self.graph_gitter:
            return 1
        from sglang.srt.distributed.device_communicators.htccl import (
            graph_capture_running,
        )

        if not graph_capture_running():
            return 1
        if not self._graph_gitter_gemeldet:
            self._graph_gitter_gemeldet = True
            logger.warning(
                "HTCCL-BAR1: %s with %d bytes would be above the grid "
                "threshold (%d bytes), but is placed on the 1blk variant "
                "while a CUDA graph is being captured -- because the "
                "restriction is either explicitly set via "
                "SGLANG_HTCCL_BAR1_GRAPH_GRID=0 or SGLANG_HTCCL_GRAPH_ENABLE "
                "is not set. This costs: in the lever measurement for #293 "
                "it was 16.1%% prefill throughput once prefill was "
                "captured. The cooperative launch IS capturable on this "
                "rig (benchmark/bar1_graph_check.py, case 'grid'); with the "
                "release set, the restriction drops away on its own. This "
                "notice appears once per rank.",
                wo, bewegt, schwelle,
            )
        return 0

    def htccl_all_reduce(self, comm, inp):
        """Sum-allreduce over ``mesh`` or ``ring``, out of place.

        Out of place is not a convenience: the ring still reads ``in``
        while it is already writing forward into ``out`` (step s+1 sends
        the partial sum formed in step s). The extension checks this and
        rejects identical pointers.
        """
        import torch

        if not self._auf or self._ext is None:
            raise Bar1Unavailable(
                "htccl_all_reduce without a transport set up -- reachable "
                "only if someone bypassed handles()."
            )
        inp = inp.contiguous()
        nbytes = inp.numel() * inp.element_size()
        chunk_max = int(self._geo.get("chunk_max", 0))
        plan = ar_plan(nbytes, chunk_max, self.welt) if chunk_max >= 16 else []
        if len(plan) > 1:
            # MULTIPLE ROUNDS. Each is a complete all_reduce over a slice --
            # no new kernel, no different decomposition, just fewer bytes
            # per launch. The round count comes from `ar_plan` and depends
            # solely on group-wide identical sizes; it is thus baked in for
            # a captured shape.
            #
            # The slices are views into the flat buffer, not copies:
            # `versatz` and `length` are multiples of 16, so every view
            # stays 16-byte-aligned, which the host insists on.
            ergebnis = torch.empty_like(inp)
            flach_ein = inp.view(-1)
            flach_aus = ergebnis.view(-1)
            eg = inp.element_size()
            for versatz, length in plan:
                teil = self._all_reduce_one_round(
                    flach_ein[versatz // eg:(versatz + length) // eg]
                )
                flach_aus[versatz // eg:(versatz + length) // eg].copy_(teil)
            return ergebnis
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
        if algo == "netz_pipe":
            k = self._pipe_k(nbytes)
            if k is None:
                raise Bar1Unavailable(
                    "netz_pipe without a matching chunk count -- reachable "
                    "only if someone bypassed handles()."
                )
            return self._pipe_all_reduce(inp, k)
        out = torch.empty_like(inp)
        # 'grid' is the cooperative multi-block launch. The threshold is
        # measured (it wins from 4 MiB up), but it is a number from ONE
        # rig -- hence it lives in an environment variable. Under graph
        # capture, `_kernel` additionally decides.
        kern = self._kernel(nbytes, self.gitter_ab, "all_reduce")
        peer_nutz = [0] * self.welt
        peer_flag = [0] * self.welt
        for r, z in self._peers.items():
            peer_nutz[r] = z.nutz.dev_ptr
            peer_flag[r] = z.flag.dev_ptr
        peer_nutz[self.rank] = self._eigen[0]
        peer_flag[self.rank] = self._eigen_flag[0]
        self._ext.bar1_all_reduce(
            inp, out, int(self.rank), int(self.welt),
            0 if algo == "mesh" else 1,
            peer_nutz, peer_flag,
            int(self._eigen[0]), int(self._eigen_flag[0]),
            int(self._geo["chunk_max"]), int(self._geo["off_netz"]),
            int(self._geo["off_ring"]),
            self._runde_dev, self._ctl_dev,
            int(self.deckel_zyklen), int(self.threads), int(kern),
            int(self.ladeform), int(self.fluss),
            int(self._abbruch_wirt),
        )
        return out

    # -- netz_pipe ---------------------------------------------------------
    #
    # Everything beyond the choice itself lives in htccl_bar1_pipe_ext: the
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
        path. ``handles`` and ``htccl_all_reduce`` ask per collective; the
        sizes repeat.
        """
        if not self.pipe_an or self._pipe_ext is None:
            return None
        if self._geo.get("off_pipe", -1) < 0:
            return None
        merk = getattr(self, "_pipe_k_merk", None)
        if merk is None:
            merk = {}
            self._pipe_k_merk = merk
        if nbytes in merk:
            return merk[nbytes]
        from sglang.srt.distributed.device_communicators.htccl_bar1_pipe_ext import (
            pipe_plan,
        )

        k = pipe_plan(
            int(nbytes), int(self.welt), int(self.pipe_schlitz),
            int(self.pipe_t), int(self.pipe_k), int(self.pipe_chunk_bytes),
            int(self.pipe_k_max),
        )
        merk[nbytes] = k
        return k

    def _pipe_supports(self, nbytes: int) -> bool:
        """Window limit for ``netz_pipe`` -- computed, not assumed.

        The requirement is ``2 T (R-1)`` slots of ``chunk_max/T`` each,
        computed in ``pipe_window_requirement``. Checked against the
        group-wide SMALLEST **actually mapped** length (``_fenster_minimum``),
        not against the gross size from sysfs and not against the requested
        region: what governs is what the holder actually found mapped
        contiguously in BAR1 per peer. If it does not fit, this path
        withdraws via ``handles``.
        """
        if not self._pipe_beleg:
            return False
        from sglang.srt.distributed.device_communicators.htccl_bar1_pipe_ext import (
            result_ring_bytes,
            pipe_window_requirement,
        )

        off = int(self._geo.get("off_pipe", -1))
        if off < 0:
            return False
        noetig = off + pipe_window_requirement(
            self.welt, int(self.pipe_t), int(self.pipe_schlitz)
        )
        # And the result ring on top: L * roundup(max_bytes, SEITE). It sits
        # BEHIND the slots, so the requirement is the ring's offset plus its
        # length -- not the maximum of the two.
        if int(self._geo.get("off_erg", -1)) >= 0:
            noetig = max(noetig, int(self._geo["off_erg"]) + result_ring_bytes(
                int(self.max_bytes), int(self._geo["erg_ring"])
            ))
        return noetig <= self._fenster_minimum

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
        off = int(self._geo.get("off_erg", -1)) if self._geo else -1
        if off < 0 or not self._eigen[0]:
            return None
        from sglang.srt.distributed.device_communicators.htccl_bar1_pipe_ext import (  # noqa: E501
            result_ring_bytes,
        )

        length = result_ring_bytes(int(self.max_bytes),
                                int(self._geo["erg_ring"]))
        return int(self._eigen[0]) + off, int(length)

    def _result_slot(self, inp):
        """Result buffer and ownership info -- or ``None``.

        Returns ``(tensor, platz, slack)``:

        ``tensor``  the result buffer, a tensor OVER the window,
        ``platz``   its ring slot -- the caller needs it to build the
                    ``peer_erg`` table,
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
        (``SGLANG_HTCCL_BAR1_PIPE_DIRECT=0``).
        """
        import weakref

        if not self.pipe_direkt or self._geo.get("off_erg", -1) < 0:
            return None
        # -- Capture -------------------------------------------------------
        #
        # This method is HOST CODE. It runs exactly once during capture and
        # never again on replay. The chosen ring slot, the pointer computed
        # from it, and the kernel's `peer_erg` table get baked into the
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
        #    `ergSlack = 1`).
        # 3. The silent bug was this: with a freely running ring index,
        #    multiple captures (sglang captures one per batch size) run
        #    over the same slots. Two graphs then share one BAR1 slot, and
        #    whoever replays them alternately gets the other one's numbers.
        #    No crash. That is exactly what the pool eliminates: a graph
        #    slot is assigned ONCE and never again.
        #
        # If the pool is empty, the call falls back to `direkt=0` --
        # reported, not silent, and correct: `direkt=0` is the same
        # measured control path, its `torch.empty_like` during capture
        # comes from the graph's private memory pool and thus already has
        # a fixed address regardless. It costs the saved VRAM pass, not
        # correctness.
        from sglang.srt.distributed.device_communicators.htccl import (
            graph_capture_running,
        )
        from sglang.srt.distributed.device_communicators.htccl_bar1_pipe_ext import (  # noqa: E501
            result_eager_free_slot,
            erg_eager_slack,
            result_graph_slot,
        )

        if graph_capture_running():
            if not self.pipe_direkt_graph:
                if not self._direkt_graph_gemeldet:
                    self._direkt_graph_gemeldet = True
                    logger.warning(
                        "HTCCL-BAR1-PIPE: direct mode disabled while a CUDA "
                        "graph is being captured (default). netz_pipe runs "
                        "the captured direct=0 path. "
                        "SGLANG_HTCCL_BAR1_PIPE_DIRECT_GRAPH=1 enables the "
                        "graph-safe path: a reserved ring slot per call "
                        "site plus a release handshake in the kernel. This "
                        "notice appears once per rank."
                    )
                return None
            i = result_graph_slot(self._erg_graph_vergeben,
                                self._erg_eager_plaetze,
                                self._erg_graph_plaetze)
            if i is None:
                if not self._erg_graph_leer_gemeldet:
                    self._erg_graph_leer_gemeldet = True
                    logger.warning(
                        "HTCCL-BAR1-PIPE: the graph pool of the result ring "
                        "is exhausted (%d of %d slots assigned, L=%d). This "
                        "and every further captured call site runs the "
                        "direct=0 path -- correct, but without the saved "
                        "VRAM pass. More slots are available with a larger "
                        "SGLANG_HTCCL_BAR1_PIPE_RESULT_RING, and each slot "
                        "costs %d bytes in the BAR1 window.",
                        self._erg_graph_vergeben, self._erg_graph_plaetze,
                        int(self._geo["erg_ring"]),
                        int(self._geo["erg_stride"]),
                    )
                return None
            # NO weak reference: from now on, this slot belongs to exactly
            # this call site. Assigning it again later would be the bug --
            # and that is exactly why the counter only grows.
            self._erg_graph_vergeben += 1
            ptr = (self._eigen[0] + int(self._geo["off_erg"])
                   + i * int(self._geo["erg_stride"]))
            out = self._pipe_ext.bar1_erg_tensor(int(ptr), inp)
            return out, i, 1

        # -- eager -----------------------------------------------------------
        #
        # What is sought is the next FREE slot, not just the next one
        # checked. This used to be a hard abort as soon as the one
        # successor slot was still held -- and that is exactly where the
        # graph-safe direct mode failed to get off the ground in the lever
        # measurement for #293: the failure occurred during the capture
        # WARMUP, which runs eager, and `SGLANG_HTCCL_BAR1_PIPE_RESULT_RING`
        # did not help, because a larger ring only assigns graph slots.
        #
        # The abort was the wrong answer to the right concern. What must
        # not happen is overwriting a buffer that is still held. If ALL
        # eager slots are held, the correct answer is the same as for the
        # exhausted graph pool a couple of lines up: `direkt=0`, reported
        # and counted. That is the measured control path and costs the
        # saved VRAM pass, not correctness. How many slots a given caller
        # needs is set by `SGLANG_HTCCL_BAR1_PIPE_RESULT_EAGER`.
        if self._erg_eager_plaetze < 2:
            return None
        belegt = [
            v is not None and v() is not None for v in self._erg_lebt
        ]
        i = result_eager_free_slot(
            self._erg_i, self._erg_eager_plaetze, belegt
        )
        if i is None:
            self._erg_eager_voll += 1
            if not self._erg_eager_voll_gemeldet:
                self._erg_eager_voll_gemeldet = True
                logger.warning(
                    "HTCCL-BAR1-PIPE: all %d eager result slots are still "
                    "being held by the caller. This call runs the "
                    "direct=0 path -- correct, but without the saved VRAM "
                    "pass. The caller is keeping more results alive at the "
                    "same time than the ring's eager part has slots; more "
                    "are available with SGLANG_HTCCL_BAR1_PIPE_RESULT_EAGER "
                    "(and a SGLANG_HTCCL_BAR1_PIPE_RESULT_RING of at least "
                    "that size). Each slot costs %d bytes in the BAR1 "
                    "window. This notice appears once per rank; how often "
                    "it actually happens is recorded in erg_eager_voll.",
                    self._erg_eager_plaetze,
                    int(self._geo["erg_stride"]),
                )
            return None
        ptr = (self._eigen[0] + int(self._geo["off_erg"])
               + i * int(self._geo["erg_stride"]))
        out = self._pipe_ext.bar1_erg_tensor(int(ptr), inp)
        # The handshake only runs along in eager mode if graph-safe mode is
        # on. Without it, the kernel stays byte-for-byte the measured one --
        # flag family 4 is then never touched. With it, the slack is the
        # ACTUAL reuse distance of this slot: under strict rotation, the
        # number of eager slots as before, fewer after a skipped slot. A
        # slack that is too large would be the weaker wait condition, i.e.
        # the dangerous direction.
        slack = (
            erg_eager_slack(i, self._erg_zaehler, self._erg_zuletzt,
                            self._erg_eager_plaetze)
            if self.pipe_direkt_graph else 0
        )
        self._erg_lebt[i] = weakref.ref(out)
        # The number recorded is that of THIS call, not the next one: the
        # distance to the next use of the same slot is the difference of
        # two call numbers, and under strict rotation over L slots that is
        # thus exactly L.
        self._erg_zuletzt[i] = self._erg_zaehler
        self._erg_zaehler += 1
        self._erg_i = i
        return out, i, slack

    def _pipe_all_reduce(self, inp, k: int):
        """One call of the pipelined kernel. Out of place, like mesh/ring."""
        import torch

        inp = inp.contiguous()
        nbytes = inp.numel() * inp.element_size()
        platz = self._result_slot(inp)
        direkt = platz is not None
        if direkt:
            out, erg_slot, erg_slack = platz
        else:
            out, erg_slot, erg_slack = torch.empty_like(inp), -1, 0
        kern = self._kernel(nbytes, self.pipe_gitter_ab, "netz_pipe")
        peer_nutz = [0] * self.welt
        peer_flag = [0] * self.welt
        peer_erg = [0] * self.welt
        for r, z in self._peers.items():
            peer_nutz[r] = z.nutz.dev_ptr
            peer_flag[r] = z.flag.dev_ptr
        peer_nutz[self.rank] = self._eigen[0]
        peer_flag[self.rank] = self._eigen_flag[0]
        if direkt:
            # THE SAME ring slot on every rank. This is not an assumption
            # about the neighbor but the same SPMD precondition every
            # collective in this module rests on: all ranks see the same
            # sequence of calls. The kernel additionally checks that its
            # own entry really is `out`.
            versatz = (int(self._geo["off_erg"])
                       + erg_slot * int(self._geo["erg_stride"]))
            for r in range(self.welt):
                peer_erg[r] = peer_nutz[r] + versatz
        from sglang.srt.distributed.device_communicators.htccl_bar1_pipe_ext import (
            pipe_fbasis,
        )

        self._pipe_ext.bar1_netz_pipe(
            inp, out, int(self.rank), int(self.welt),
            peer_nutz, peer_flag, peer_erg,
            int(self._eigen[0]), int(self._eigen_flag[0]),
            int(self.pipe_schlitz),
            int(self._geo["off_pipe"]),
            int(pipe_fbasis(self.welt, self.a2a_an)),
            int(k), int(self.pipe_t), int(self.pipe_vorlauf),
            int(self.pipe_quittung), 1 if direkt else 0, int(erg_slack),
            self._runde_dev, self._schritt_dev, self._erg_gen_dev,
            self._ctl_dev,
            int(self.deckel_zyklen), int(self.threads), int(kern),
            int(self.ladeform),
            int(self._abbruch_wirt),
        )
        return out

    def byte_proof_pipe(self, runden: int = 0) -> bool:
        """Byte-level proof for ``netz_pipe``. Without it, the path withdraws.

        Separate from ``byte_proof_all``, because it checks something
        different: the pair proof shows that bytes arrive; this one shows
        that slot reuse holds up across multiple rounds. The second point
        is the more dangerous one, and it does not show up in a single
        round.
        """
        from sglang.srt.distributed.device_communicators import (
            htccl_bar1_pipe_ext,
        )

        if not self.pipe_an or self._pipe_ext is None:
            self._pipe_beleg = False
            return False
        # Provisionally let it through so the proof itself can run; the
        # final answer is below. Nobody asks `handles` in the meantime --
        # the proof runs during setup, before the first collective.
        self._pipe_beleg = True
        try:
            ok = htccl_bar1_pipe_ext.byte_proof_pipe(self, runden)
        except Exception as e:
            logger.warning(
                "HTCCL-BAR1-PIPE: the byte-level proof aborted with %r. "
                "netz_pipe withdraws; mesh and ring are unaffected.",
                e,
            )
            ok = False
        self._pipe_beleg = bool(ok)
        if not ok:
            logger.warning(
                "HTCCL-BAR1-PIPE: byte-level proof not passed -- netz_pipe "
                "withdraws via handles()."
            )
        return self._pipe_beleg

    # -- all_gather --------------------------------------------------------
    #
    # THE STOPPER. Before this change, HTCCL_OPS did not cover all_gather,
    # and the standard run aborted:
    #
    #     RuntimeError: HTCCL: 'all_gather' with 10600448 bytes during a
    #     CUDA graph capture, but bar1 reports handles(...) -> False.
    #
    # Correctly aborted -- under HTCCL, PyNccl is not built, the fallback
    # path would be the host-staged gloo tier, and that runs once during a
    # capture and never again on replay. It just did not work.
    #
    # WHY NO NEW KERNEL. An all_gather is the AG phase of the mesh allreduce
    # without the reduction, and the a2a kernel can already do exactly that:
    # it moves bytes, knows no data type, and receives offsets and lengths
    # SEPARATELY PER RANK. An all_gather is an all_to_all in which every
    # destination gets the same slice -- i.e. the same table with
    # ``sende_versatz[z] = const``. This is not a trick but the promise
    # already spelled out in htccl_bar1_ext.py ("it never assumed they were
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
        (``htccl.HTCCLCommunicator.all_gather``), i.e. with the shard. The
        result is ``R`` times as large and is NOT checked here: it lives
        in local VRAM, not in the window.

        Every condition is rank-uniform, for the same reason as in
        :meth:`handles`.

        Beyond the slot size, this is NOT rejected but decomposed into
        rounds (:func:`ag_plan`) -- a rejection would be an abort with no
        fallback under capture, and that is exactly what the stopper hung
        on. Only what does not work even in rounds is rejected.

        ``nbytes % 16 != 0`` is explicitly NOT rejected, unlike with
        all_reduce. The a2a kernel has the tail path for that (``VEK=0``,
        packet assembled byte by byte): correct, slower, unmeasured. That
        is the right choice, because under capture the alternative is not
        a slower path but no path at all.

        **What a crooked shard costs, precisely stated:** not the last 15
        bytes, but everything. Rank ``i``'s result offset is ``i *
        shard``; if ``shard`` is not a multiple of 16, every offset except
        rank 0's is off-alignment, and the extension switches to
        ``VEK=0`` for the WHOLE call (it checks all offsets jointly,
        htccl_bar1_ext.py: "there is no such thing as 'mostly aligned'").
        Whoever sees a slow number should check this first before
        attributing it to the transport.
        """
        if not self.ag_an:
            return False
        # Same region, same kernel, same byte-level proof. Without the a2a
        # region (SGLANG_HTCCL_BAR1_A2A=0), there is also no all_gather --
        # stated, not silently assumed.
        if not self.a2a_an or not self._a2a_beleg:
            return False
        geo = self._geo
        if geo.get("off_a2a", -1) < 0 or geo.get("a2a_schlitz", 0) <= 0:
            return False
        if nbytes <= 0:
            return False
        if nbytes < self.ag_min_bytes:
            return False
        # The same window concept as with all_reduce and a2a: against the
        # group-wide SMALLEST actually mapped length.
        if geo["region_bytes"] > self._fenster_minimum:
            return False
        # There is still a ceiling, though, and it is not a window limit
        # but a round limit: every round is one kernel launch with one
        # barrier, and arbitrarily many of those per collective would not
        # be a transport but a loop. Rank-uniform, because nbytes is.
        if -(-nbytes // int(geo["a2a_schlitz"])) > self.ag_max_runden:
            return False
        return True

    def ag_rounds(self, nbytes: int) -> int:
        """Round count for a shard of ``nbytes`` -- for logging/tests."""
        slot = int(self._geo.get("a2a_schlitz", 0))
        if slot <= 0:
            return 0
        return max(1, -(-int(nbytes) // slot))

    def htccl_all_gather(self, comm, inp, dim: int = -1):
        """``all_gather`` over the direct path, in several rounds if needed.

        Result shape and axis handling are byte-for-byte those of the seam
        (``htccl.HTCCLCommunicator.all_gather``) and of
        ``htccl_device.all_gather``: first ``(R,) + form``, then
        ``movedim(0, dim)``, then merge. Not reinvented -- the same
        expression, so a transport switch changes nothing about the
        numbers.
        """
        import torch

        if not self._auf or self._ext is None or not self.a2a_an:
            raise Bar1Unavailable(
                "htccl_all_gather without an a2a region set up -- reachable "
                "only if someone bypassed handles()."
            )
        if dim < 0:
            dim += inp.dim()
        inp = inp.contiguous()
        form = tuple(inp.size())
        scherbe = inp.numel() * inp.element_size()
        out = torch.empty((self.welt,) + form, dtype=inp.dtype,
                          device=inp.device)
        # Uniform, because the seam is -- but passed as a VECTOR to
        # ag_plan, not as an assumption in the arithmetic. The rationale is
        # in ag_plan.
        plan = ag_plan([scherbe] * self.welt, int(self._geo["a2a_schlitz"]))
        flach = out.view(-1)
        for round in plan:
            s_off = [round[self.rank][0]] * self.welt
            s_len = [round[self.rank][1]] * self.welt
            e_off = [x[2] for x in round]
            e_len = [x[1] for x in round]
            self.htccl_all_to_all_single(
                comm, flach, inp, s_len, e_len, s_off, e_off,
            )
        out = out.movedim(0, dim)
        return out.reshape(form[:dim] + (self.welt * form[dim],) + form[dim + 1:])

    # -- broadcast ---------------------------------------------------------
    #
    # THE NEXT STOPPER, and it sat one level below the all_gather one. The
    # standard run aborted while capturing the DRAFT graph
    # (eagle_worker_v2.init_cuda_graphs -> parallel_state broadcast ->
    # htccl._select):
    #
    #     RuntimeError: HTCCL: 'broadcast' with 128 bytes during a CUDA
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
    # (``sende_versatz[z] = const``, ``sende_bytes[z] = n``), and every
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
    # htccl_device rests on.

    def _handles_broadcast(self, nbytes: int) -> bool:
        """``nbytes`` is the full payload -- it is the same group-wide.

        The seam asks with ``tensor.numel() * element_size()``
        (``htccl.HTCCLCommunicator.broadcast``). Unlike with all_gather,
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

        **The covered range is ``1 .. a2a_schlitz * bc_max_runden``**, with
        no gaps. This is not a rewrite but the lesson from the second
        attempt: the first covered broadcast but rejected 12 bytes (lower
        bound 16, copied from a2a), and the standard run sends exactly
        those 12 bytes. Silently answering ``False`` for a size in this
        range means, under capture, aborting the run -- if a threshold is
        ever reintroduced here, it needs the same rationale
        ``_no_collective`` also states for reduce_scatter.
        """
        if not self.bc_an:
            return False
        # Same region, same kernel -- but an OWN byte-level proof
        # (`_bc_beleg`), because the table is a different one. Without the
        # a2a region there is also no broadcast; stated, not silently
        # assumed.
        if not self.a2a_an or not self._a2a_beleg or not self._bc_beleg:
            return False
        geo = self._geo
        if geo.get("off_a2a", -1) < 0 or geo.get("a2a_schlitz", 0) <= 0:
            return False
        if nbytes <= 0:
            return False
        if nbytes < self.bc_min_bytes:
            return False
        # The same window concept as everywhere: against the group-wide
        # SMALLEST actually mapped length.
        if geo["region_bytes"] > self._fenster_minimum:
            return False
        if -(-nbytes // int(geo["a2a_schlitz"])) > self.bc_max_runden:
            return False
        return True

    def bc_rounds(self, nbytes: int) -> int:
        """Round count for ``nbytes`` -- for logging/tests."""
        slot = int(self._geo.get("a2a_schlitz", 0))
        if slot <= 0:
            return 0
        return max(1, -(-int(nbytes) // slot))

    def htccl_broadcast(self, comm, tensor, src: int = 0):
        """``broadcast`` over the direct path, in several rounds if needed.

        IN-PLACE and with the same return value as every other version of
        this seam (``htccl.HTCCLCommunicator.broadcast``,
        ``htccl_device.htccl_broadcast``): the tensor handed in is filled
        and returned. ``src`` is a GROUP-LOCAL rank -- that is how the seam
        asks, and it is also what ``self.rank`` means.
        """
        import torch

        if not self._auf or self._ext is None or not self.a2a_an:
            raise Bar1Unavailable(
                "htccl_broadcast without an a2a region set up -- reachable "
                "only if someone bypassed handles()."
            )
        R = self.welt
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
        ziel = torch.empty_like(source)
        plan = bc_plan(nbytes, int(self._geo["a2a_schlitz"]))
        # Group-wide identical, because `plan` is -- and that is why the
        # kernel variant (grid/1blk) also comes out the same on every rank,
        # instead of depending on the rank-dependent question "how much do
        # I send".
        for versatz, length in plan:
            sendet = (self.rank == src)
            s_len = [length if sendet else 0] * R
            s_off = [versatz] * R
            e_len = [length if i == src else 0 for i in range(R)]
            e_off = [versatz if i == src else 0 for i in range(R)]
            self.htccl_all_to_all_single(
                comm, ziel, source, s_len, e_len, s_off, e_off,
                kern_last=length * (R - 1),
            )
        tensor.copy_(ziel.view(tensor.shape))
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

        self._bc_beleg = False
        if not self.bc_an:
            logger.info(
                "HTCCL-BAR1: broadcast is disabled via SGLANG_HTCCL_BAR1_BC=0 "
                "-- no byte-level proof, handles() says False."
            )
            return False
        if not self.a2a_an or not self._a2a_beleg:
            return False
        if not self._auf or self._ext is None:
            return False

        R, r = self.welt, self.rank
        slot = int(self._geo.get("a2a_schlitz", 0))
        if slot <= 0:
            return False
        klein = min(4096, slot)
        # Beyond the slot size, but not by a hair: 16 leftover bytes in the
        # second round additionally hit the kernel's tail path.
        gross = slot + 16
        # BELOW one packet. This is the case the first attempt walked right
        # past: the lower bound stood at 16, the standard run sends 12, and
        # the proof only ran sizes the bound accepted anyway -- so it could
        # never have seen the bug. 12 bytes exercise the one path no other
        # size here does: a single, incomplete packet, assembled in a
        # register and read back out byte by byte.
        winzig = 12

        ok_lokal = True
        for src, n in ([(s, klein) for s in range(R)]
                       + [(s, winzig) for s in range(R)]
                       + [(0, gross)]):
            # Every rank starts with ITS OWN marker. For the source, that is
            # already the expected value; for everyone else, a byte
            # distinguishable from it -- a broadcast that moves nothing at
            # all thus stands out instead of accidentally looking correct.
            soll = self._a2a_marker(src, src)
            puffer = torch.full((n,), self._a2a_marker(r, r), dtype=torch.uint8,
                                device=self.device)
            bounded_barrier(
                self.cpu_group,
                f"bar1 broadcast proof src={src} n={n}: before the round",
                table=self._peer_table,
            )
            gelaufen = True
            try:
                self.htccl_broadcast(None, puffer, src)
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
                    "HTCCL-BAR1: broadcast byte-level proof (src=%d, %d "
                    "bytes) could not run: %r", src, n, ex,
                )
                ok_lokal = False
                gelaufen = False
            if not gelaufen:
                continue
            rueck = puffer.cpu()
            schlecht = int((rueck != soll).sum().item())
            if schlecht:
                ok_lokal = False
                logger.warning(
                    "HTCCL-BAR1: broadcast byte-level proof %d->%d FAILED: "
                    "%d of %d bytes wrong (%d rounds). broadcast withdraws.",
                    src, r, schlecht, n, len(bc_plan(n, slot)),
                )
            else:
                logger.info(
                    "HTCCL-BAR1: broadcast byte-level proof %d->%d passed: "
                    "0 of %d bytes wrong (%d rounds).",
                    src, r, n, len(bc_plan(n, slot)),
                )

        traeger: list[object] = [None] * R
        check_peers("bar1 broadcast proof: verdict exchange", self._peer_table)
        dist.all_gather_object(traeger, bool(ok_lokal), group=self.cpu_group)
        self._bc_beleg = all(bool(x) for x in traeger)
        if not self._bc_beleg:
            logger.warning(
                "HTCCL-BAR1: broadcast byte-level proof failed group-wide "
                "(ranks %s). handles('broadcast') returns False; "
                "all_reduce, all_to_all, and all_gather are unaffected.",
                [i for i, x in enumerate(traeger) if not x],
            )
        return self._bc_beleg

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
        if not self.a2a_an or not self._a2a_beleg:
            return False
        geo = self._geo
        if geo.get("off_a2a", -1) < 0 or geo.get("a2a_schlitz", 0) <= 0:
            return False
        if nbytes < self.a2a_min_bytes:
            return False
        # Beyond the slot size, this is NOT rejected but decomposed into
        # rounds -- the same answer as for all_reduce, all_gather, and
        # broadcast. The coarse number here is the uniform case; the exact
        # round count falls out in `supports_a2a`, once the group-wide
        # largest block is known.
        if a2a_rounds(-(-nbytes // self.welt),
                      int(geo["a2a_schlitz"])) > self.a2a_max_runden:
            return False
        # The same window concept as with all_reduce: against the
        # group-wide SMALLEST actually mapped length, not against the
        # requested one. Redundant as long as setup completed
        # successfully -- and exactly for that reason cheap.
        if geo["region_bytes"] > self._fenster_minimum:
            return False
        return True

    def a2a_slot_bytes(self) -> int:
        """Largest block ONE directed pair can carry. 0 = no a2a."""
        if not self.a2a_an or not self._a2a_beleg:
            return 0
        return int(self._geo.get("a2a_schlitz", 0))

    def supports_a2a(self, groesster_block: int) -> bool:
        """Does the largest block over ALL pairs fit into a slot?

        ``groesster_block`` must be a **group-wide identical** value -- the
        maximum over all R*R blocks, not over the caller's own row. If
        every rank computed it only from its own block sizes, one rank
        could run into the collective and another into the fallback, and
        the result would be a hang instead of an error. The caller
        (``HTCCLCommunicator.all_to_all_single``) maximizes over the group
        beforehand; that is exactly why this check does not live in
        ``handles``.
        """
        if not self.a2a_an or not self._a2a_beleg or not self._auf:
            return False
        if groesster_block < 0:
            return False
        slot = int(self._geo.get("a2a_schlitz", 0))
        if slot <= 0:
            return False
        # If it does not fit into ONE slot, it runs in several rounds --
        # only what does not work even in rounds is rejected.
        return a2a_rounds(groesster_block, slot) <= self.a2a_max_runden

    def a2a_rounds_for(self, groesster_block: int) -> int:
        """Round count the caller passes to ``htccl_all_to_all_single``.

        It falls out of the GROUP-WIDE largest block, which the seam has
        already computed anyway -- not out of the caller's own row. From
        its own row it would be rank-dependent, and a rank running one
        round fewer would leave the others waiting in the barrier.
        """
        slot = int(self._geo.get("a2a_schlitz", 0))
        if slot <= 0:
            return 0
        return a2a_rounds(groesster_block, slot)

    def htccl_all_to_all_single(self, comm, output, inp,
                                sende_bytes, empfangs_bytes,
                                sende_versatz=None, empfangs_versatz=None,
                                kern_last=None, runden=None):
        """Wrapper with a round loop. One step or several, depending on the block.

        ``runden`` comes from the caller and is GROUP-WIDE identical -- the
        seam computes it from the largest block over all pairs
        (:meth:`a2a_rounds_for`). ``None`` means one round and is thus
        byte-for-byte the previous path; that is exactly how
        :meth:`htccl_all_gather` and :meth:`htccl_broadcast` call it, since
        they have already sliced their own segments themselves.

        What one round moves: at most one slot's worth out of every block,
        in one piece, starting at offset ``k*slot``. Blocks that finish
        earlier carry length 0 -- they ride along in the barrier without
        moving any bytes. Same pattern as in ag_plan, and for the same
        reason: the round count must not depend on how much THIS rank has
        to do.
        """
        n = 1 if runden is None else max(1, int(runden))
        if n == 1:
            return self._a2a_one_round(
                comm, output, inp, sende_bytes, empfangs_bytes,
                sende_versatz, empfangs_versatz, kern_last,
            )
        slot = int(self._geo["a2a_schlitz"])
        s_basis = list(sende_versatz) if sende_versatz is not None else None
        e_basis = list(empfangs_versatz) if empfangs_versatz is not None else None
        if s_basis is None:
            s_basis, acc = [], 0
            for length in sende_bytes:
                s_basis.append(acc)
                acc += int(length)
        if e_basis is None:
            e_basis, acc = [], 0
            for length in empfangs_bytes:
                e_basis.append(acc)
                acc += int(length)
        for k in range(n):
            s_len = [
                min(slot, max(0, int(length) - k * slot))
                for length in sende_bytes
            ]
            e_len = [
                min(slot, max(0, int(length) - k * slot))
                for length in empfangs_bytes
            ]
            self._a2a_one_round(
                comm, output, inp, s_len, e_len,
                [b + k * slot for b in s_basis],
                [b + k * slot for b in e_basis],
                kern_last,
            )
        return output

    def _a2a_one_round(self, comm, output, inp,
                        sende_bytes, empfangs_bytes,
                        sende_versatz=None, empfangs_versatz=None,
                        kern_last=None):
        """``all_to_all_single`` over the direct path. One step, one barrier.

        ``sende_bytes[j]`` is the block going to rank ``j``,
        ``empfangs_bytes[i]`` the one coming from rank ``i`` -- both in
        **bytes**, not in rows and not in elements. The kernel moves bytes:
        there is no reduction, hence no data type. fp8, bf16, int32, uint8
        take the same path, and the sm_86 cards' missing fp8 conversion
        instructions are irrelevant here.

        ``sende_versatz`` / ``empfangs_versatz`` are **optional** and in
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

        ``kern_last`` is the byte count that decides the kernel variant
        (grid/1blk). ``None`` means: what this rank actually sends over the
        aperture -- correct for any table in which all ranks move similar
        amounts. A broadcast is precisely not that: there, ONE rank sends
        everything and all others send nothing, and a choice computed from
        the caller's own row would come out differently per rank.
        Correctness does not depend on this (both variants run the same
        flag protocol), but comparability of measurements and the
        rank-uniformity of the capture do. The caller therefore passes in a
        group-wide identical value.

        The caller is responsible for ensuring that ``empfangs_bytes[i]``
        on this rank equals ``sende_bytes[rang]`` on rank ``i``. The
        extension checks what it can check locally (buffer bounds, slot
        bound, its own block), but not agreement across the group -- that
        would require running a collective, and that would be exactly the
        host sync this path avoids.
        """
        if not self._auf or self._ext is None or not self.a2a_an:
            raise Bar1Unavailable(
                "htccl_all_to_all_single without an a2a region set up -- "
                "reachable only if someone bypassed handles()."
            )
        R = self.welt
        if len(sende_bytes) != R or len(empfangs_bytes) != R:
            raise Bar1Unavailable(
                f"block sizes have length {len(sende_bytes)}/"
                f"{len(empfangs_bytes)}, {R} expected."
            )
        inp = inp.contiguous()
        if not output.is_contiguous():
            raise Bar1Unavailable("output buffer is not contiguous")

        if sende_versatz is None:
            sende_off, s = [], 0
            for n in sende_bytes:
                sende_off.append(s)
                s += int(n)
        else:
            if len(sende_versatz) != R:
                raise Bar1Unavailable(
                    f"sende_versatz has length {len(sende_versatz)}, "
                    f"{R} expected."
                )
            sende_off = [int(x) for x in sende_versatz]
        if empfangs_versatz is None:
            empf_off, e = [], 0
            for n in empfangs_bytes:
                empf_off.append(e)
                e += int(n)
        else:
            if len(empfangs_versatz) != R:
                raise Bar1Unavailable(
                    f"empfangs_versatz has length {len(empfangs_versatz)}, "
                    f"{R} expected."
                )
            empf_off = [int(x) for x in empfangs_versatz]

        # The cooperative multi-block launch pays off at the same threshold
        # as with all_reduce -- it is measured THERE and only there; here it
        # is carried over, not confirmed. What governs is what actually
        # goes over PCIe, i.e. excluding one's own block.
        if kern_last is None:
            bewegt = sum(
                int(n) for j, n in enumerate(sende_bytes) if j != self.rank
            )
        else:
            bewegt = int(kern_last)
        kern = self._kernel(bewegt, self.gitter_ab, "all_to_all_single")

        peer_nutz = [0] * R
        peer_flag = [0] * R
        for rr, z in self._peers.items():
            peer_nutz[rr] = z.nutz.dev_ptr
            peer_flag[rr] = z.flag.dev_ptr
        peer_nutz[self.rank] = self._eigen[0]
        peer_flag[self.rank] = self._eigen_flag[0]

        self._ext.bar1_all_to_all(
            inp, output, int(self.rank), int(R),
            [int(x) for x in sende_off], [int(x) for x in sende_bytes],
            [int(x) for x in empf_off], [int(x) for x in empfangs_bytes],
            peer_nutz, peer_flag,
            int(self._eigen[0]), int(self._eigen_flag[0]),
            int(self._geo["a2a_schlitz"]), int(self._geo["off_a2a"]),
            int(fbasis_a2a(R)),
            self._runde_dev, self._ctl_dev,
            int(self.deckel_zyklen), int(self.threads), int(kern),
            int(self.ladeform),
            int(self._abbruch_wirt),
        )
        return output

    @staticmethod
    def _a2a_marker(source: int, ziel: int) -> int:
        """A byte that differs per directed pair, never 0x00 and never 0xFF.

        ``0x40 | (source*8 + ziel)`` -- for R <= 8, ``source*8+ziel`` is
        injective and fits in 6 bits. 0xFF is the output buffer's
        pre-fill value, 0x00 the receive slot's; both are thus
        distinguishable from the pattern, and a block that was NOT written
        stands out as such instead of accidentally looking like a hit.
        """
        return 0x40 | ((source * 8 + ziel) & 0x3F)

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

        self._a2a_beleg = False
        if not self.a2a_an:
            logger.info(
                "HTCCL-BAR1: all_to_all is disabled via "
                "SGLANG_HTCCL_BAR1_A2A=0 -- no byte-level proof, handles() "
                "says False."
            )
            return False
        if not self._auf or self._ext is None:
            return False

        R, r = self.welt, self.rank
        slot = int(self._geo.get("a2a_schlitz", 0))
        # The largest block of the skewed pass is 3*block+6.
        block = min(8192, (slot - 6) // 3)
        block = (block // 16) * 16
        if block <= 0:
            logger.warning(
                "HTCCL-BAR1: a2a slot of %d bytes is too small for the "
                "byte-level proof. all_to_all withdraws.", slot,
            )
            return False

        # From here on, the group-wide reconciliation runs in EVERY case. A
        # rank that bails out before the all_gather_object because of a
        # local exception would leave the others waiting in it -- a failed
        # proof would turn into a hang.
        ok_lokal = True
        try:
            ok_lokal = self._a2a_proof_passes(block)
        except Exception as ex:                # noqa: BLE001
            ok_lokal = False
            logger.warning("HTCCL-BAR1: a2a byte-level proof aborted: %r", ex)

        traeger: list[object] = [None] * R
        check_peers("bar1 a2a proof: verdict exchange", self._peer_table)
        dist.all_gather_object(traeger, bool(ok_lokal), group=self.cpu_group)
        self._a2a_beleg = all(bool(x) for x in traeger)
        if not self._a2a_beleg:
            logger.warning(
                "HTCCL-BAR1: a2a byte-level proof failed group-wide (ranks "
                "%s). handles('all_to_all') returns False; all_reduce is "
                "unaffected.",
                [i for i, x in enumerate(traeger) if not x],
            )
        return self._a2a_beleg

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

        R, r = self.welt, self.rank
        ok_lokal = True
        for schief in (False, True):

            def length(q: int, z: int) -> int:
                if not schief:
                    return block
                return block * (1 + ((q + z) % 3)) + ((q * 5 + z * 3) % 7)

            sende = [length(r, z) for z in range(R)]
            empf = [length(q, r) for q in range(R)]
            inp = torch.empty(sum(sende), dtype=torch.uint8, device=self.device)
            o = 0
            for z in range(R):
                inp[o:o + sende[z]] = self._a2a_marker(r, z)
                o += sende[z]
            out = torch.full((sum(empf),), 0xFF, dtype=torch.uint8,
                             device=self.device)
            lauf = "skewed" if schief else "uniform"
            bounded_barrier(
                self.cpu_group,
                f"bar1 a2a proof ({lauf}): before the round",
                table=self._peer_table,
            )
            gelaufen = True
            try:
                self.htccl_all_to_all_single(None, out, inp, sende, empf)
                bounded_device_sync(
                    f"bar1 a2a proof ({lauf})",
                    device=self.device,
                    table=self._peer_table,
                )
                # Same reason as in the broadcast proof: without this, an
                # aborted kernel is reported as a byte mismatch.
                self.raise_if_aborted(f"a2a proof ({lauf})")
            except Exception as ex:            # noqa: BLE001 -- reason goes into the log
                logger.warning(
                    "HTCCL-BAR1: a2a byte-level proof (%s) could not run: %r",
                    "skewed" if schief else "uniform", ex,
                )
                ok_lokal = False
                gelaufen = False
            if not gelaufen:
                continue
            rueck = out.cpu()
            o = 0
            schlecht_ges = 0
            for q in range(R):
                soll = self._a2a_marker(q, r)
                stueck = rueck[o:o + empf[q]]
                schlecht = int((stueck != soll).sum().item())
                if schlecht:
                    ok_lokal = False
                    schlecht_ges += schlecht
                    logger.warning(
                        "HTCCL-BAR1: a2a byte-level proof %d->%d (%s) "
                        "FAILED: %d of %d bytes wrong. all_to_all "
                        "withdraws.",
                        q, r, "skewed" if schief else "uniform", schlecht,
                        empf[q],
                    )
                o += empf[q]
            if not schlecht_ges:
                # The passed proof belongs in the log too -- it is the
                # statement every later timing measurement rests on.
                logger.info(
                    "HTCCL-BAR1: a2a byte-level proof (%s) passed: 0 of %d "
                    "bytes wrong across %d senders.",
                    "skewed" if schief else "uniform", sum(empf), R,
                )
        return ok_lokal

    # -- Peer liveness -------------------------------------------------------

    def _install_abort_window(self) -> None:
        """One host word the spin kernels poll, so a dead peer can end them.

        Only built when the feature is on. When the runtime refuses to map
        the word, ``AbortWindow`` degrades to ``device_ptr == 0`` on its own
        and the kernels see ``nullptr``; nothing here has to special-case
        that.
        """
        if not htccl_liveness.liveness_enabled():
            return
        try:
            window = htccl_liveness.AbortWindow()
        except Exception as e:            # pragma: no cover - degrade, do not refuse to boot
            logger.warning("HTCCL-BAR1: no abort window (%s).", e)
            return
        htccl_liveness.register_abort_window(window)
        self._abort_window = window

    def _release_abort_window(self) -> None:
        window, self._abort_window = self._abort_window, None
        if window is None:
            return
        htccl_liveness.unregister_abort_window(window)
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

        def _haken() -> None:
            window = self._abort_window
            if window is not None:
                window.trip(f"host wait '{label}' gave up")

        return _haken

    @property
    def _abbruch_wirt(self) -> int:
        """Device address of the abort word, ``0`` when there is none.

        Every extension call passes this. ``0`` is the pre-#312 behaviour
        exactly: the kernels keep their cycle deadline and probe nothing.
        """
        window = self._abort_window
        return window.device_ptr if window is not None else 0

    def raise_if_aborted(self, label: str) -> None:
        """Raise if a spin kernel took its abort path. NOT for the hot path.

        ``status()`` has always held this answer and nothing in production
        ever asked for it, which is why a tripped kernel was silent: the
        stream simply continued over a partially written output buffer.
        Reading the word synchronizes -- that cost is precisely what the
        direct path exists to avoid -- so this belongs at bring-up and on
        wait paths that have already failed, never inside a collective.

        Not gated on ``SGLANG_HTCCL_PEER_LIVENESS``. The kill switch restores
        the previous behaviour of the liveness machinery; it is not a request
        to go back to accepting a half-written buffer as a result. The only
        way to reach this raise is a run that was already broken.
        """
        if self.status() != 1:
            return
        window = self._abort_window
        grund = window.reason if window is not None and window.tripped else None
        raise Bar1KernelAborted(
            f"HTCCL-BAR1 {label}: a spin kernel took its abort path. Either "
            f"it exceeded SGLANG_HTCCL_BAR1_CAP_CYCLES "
            f"({self.deckel_zyklen} cycles) waiting for a peer's flag, or "
            f"the host abort word was set"
            + (f" -- {grund}" if grund else "")
            + ". The output buffer of that call is partially written and "
            "must not be used."
        )

    def status(self) -> int:
        """``1`` if a kernel ever hit the time limit.

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
        different signatures (``htccl_reduce_scatter(comm, inp, dim)``,
        formerly also ``htccl_broadcast(comm, tensor, src)``). With the
        earlier fixed ``(self, comm, inp)``, both produced a ``TypeError``
        before this message even got a chance to run -- so the text was
        unreachable and the cause was recorded nowhere.
        """
        raise NotImplementedError(
            f"The BAR1 transport covers {', '.join(sorted(self.HTCCL_OPS))}. "
            f"reduce_scatter needs a reduction, and the a2a kernel moves "
            f"bytes: it carries all_gather and broadcast for free, and "
            f"reduce_scatter not at all. This line is reachable only if "
            f"someone bypassed handles()."
        )

    # all_gather and broadcast are NO LONGER here. Both assignments used to
    # sit in this list until htccl_all_gather resp. htccl_broadcast were
    # introduced, and would have overwritten the new method -- an
    # assignment in the class body silently wins against a `def` of the
    # same name higher up. Ruff (F811) caught it for all_gather; without
    # that run, every all_gather would have raised NotImplementedError even
    # though handles() had promised otherwise, and the guard would have
    # looked like a transport bug. The same trap was set up for broadcast
    # and is now additionally pinned down by
    # test_htccl_bar1_broadcast.py.
    htccl_reduce_scatter = _no_collective

    # -- Teardown ----------------------------------------------------------

    def close(self) -> None:
        """Tear everything back down. Order matters.

        First the registrations and mappings of the foreign BARs, then the
        attachments (which hold the BAR1 pages), then one's own
        allocation. Reversed, this would pull pages out from under the
        driver while a mapping is still live.
        """
        self._auf = False
        # Before anything else: no kernel of this transport will run again,
        # so the abort word has nobody left to talk to. Unregistering it here
        # keeps the watchdog from holding a reference to a closed window.
        self._release_abort_window()
        self._peer_table = None
        # Deregister first: the space is on its way back from here on, and
        # a ledger that still reports it occupied after a `close` would
        # groundlessly shortchange a group built later.
        try:
            from sglang.srt.distributed.device_communicators import (
                htccl_matrix_transport as _kasse,
            )

            _kasse.ledger_debit(self.device, self.gruppe)
        except Exception:
            pass
        for z in self._peers.values():
            for a in (z.nutz, z.flag):
                if self._cuda is not None:
                    # UNREGISTER AT THE SAME ADDRESS UNDER WHICH IT WAS
                    # REGISTERED. cudaHostUnregister on a pointer INSIDE the
                    # registration fails, and the unregister would silently
                    # not happen -- the aperture would remain registered on
                    # the next run.
                    self._cuda.unregister(a.reg_adresse)
                try:
                    a.mmap_obj.close()      # type: ignore[attr-defined]
                except Exception:
                    pass
        if self._halter is not None:
            for z in self._peers.values():
                for a in (z.nutz, z.flag):
                    self._halter.release(a.halter_handle)
            self._halter.close()
            self._halter = None
        self._peers.clear()
        eigene = set(self._dmabuf_fds)
        for liste in self._fremde_fds:
            for fd in liste or ():
                if fd is not None and fd >= 0 and fd not in eigene:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
        self._fremde_fds = []
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
        for fd in self._halte_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self._halte_fds = []
        if self._cuda is not None:
            for eig in ("_eigen", "_eigen_flag"):
                w = getattr(self, eig)
                if w[2]:
                    self._cuda.vmm_free(*w)
                    setattr(self, eig, (0, 0, 0))
        self._runde_dev = None
        self._ctl_dev = None
        self._schritt_dev = None
        self._erg_gen_dev = None


def build_bar1(cpu_group, device, fenster_bytes: int,
              bericht: Optional[dict] = None,
              gruppe: str = "") -> Optional[HTCCLBar1Transport]:
    """Factory with a clean fallback.

    ``None`` means: this machine cannot do the direct path, with a logged
    reason. No raising, no silent fallback to another path -- the choice
    of a replacement path is made by the caller, not by this module.

    ``fenster_bytes`` is the **requested** size of the receive region per
    rank. What actually comes of it is reported afterward by
    ``transport.window_minimum()`` -- and only that belongs in the
    planner.

    ``bericht`` is the REASON, and it is not an afterthought. Previously,
    every failure ended in a ``logger.info`` and a ``None``, and the
    caller then went on to log "transport=bar1" regardless. That is
    exactly how a measurement was devalued once: the tp group ran over
    BAR1, the dcp group over gloo, and the log looked the same in both
    cases. Whoever passes ``bericht`` gets ``grund`` and ``stufe``
    ("aufbau", "byte_proof") written into it here and can turn that into a
    loud message.
    """
    if bericht is None:
        bericht = {}

    def _aus(stufe: str, text: str):
        bericht["stufe"] = stufe
        bericht["grund"] = text
        return None

    try:
        t = HTCCLBar1Transport(cpu_group, device, fenster_bytes, gruppe=gruppe)
    except Bar1Unavailable as e:
        logger.info("HTCCL-BAR1: direct path not available -- %s", e)
        return _aus("aufbau", str(e))
    except NotImplementedError as e:
        logger.info("HTCCL-BAR1: direct path needs driver work -- %s", e)
        return _aus("aufbau", f"driver work needed: {e}")
    except Exception as e:                 # a half-built setup is not left standing
        logger.info("HTCCL-BAR1: setup failed -- %r", e)
        return _aus("aufbau", f"{type(e).__name__}: {e}")
    # The byte-level proof is part of setup, not an optional extra: without
    # it, `handles` is locked. On this rig, the driver reported peer
    # access for one pair and delivered 4096 of 1,048,576 bytes.
    try:
        belege = t.byte_proof_all()
    except Exception as e:
        logger.info("HTCCL-BAR1: byte-level proof could not run -- %r", e)
        t.close()
        return _aus("byte_proof", f"could not run: {type(e).__name__}: {e}")
    if not t._belege_stehen:
        # Previously, the transport came out of here UNSCATHED and only
        # withdrew later via `handles` -- so every collective silently ran
        # over the gloo tier while the log said "transport=bar1". The
        # reason belongs reported to the caller, not withheld.
        gefallen = sorted(k for k, v in belege.items() if not v)
        bericht["stufe"] = "byte_proof"
        bericht["grund"] = (
            f"Byte-level proof failed for the directed pairs {gefallen}. "
            f"handles() says False for everything; every collective in "
            f"this group runs over the gloo tier."
        )
        bericht["haelt_belegt"] = True
    # And the same principle for all_to_all -- its own kernel, its own
    # slots, its own flag lines, hence its own proof. It is ONLY attempted
    # if the all_reduce proof holds: a collective over an edge that has
    # already lost bytes there needs no second probe. If it fails,
    # all_reduce remains available regardless; that is why the transport
    # is not torn down here either.
    if t._belege_stehen:
        try:
            t.byte_proof_a2a()
        except Exception as e:
            logger.info(
                "HTCCL-BAR1: a2a byte-level proof could not run -- %r. "
                "all_to_all withdraws, all_reduce continues.", e,
            )
        # And once more for broadcast: the same kernel, but a different
        # table (exactly one sender, receive offset is not the prefix
        # sum), hence its own proof. It runs ONLY if the a2a proof holds --
        # without that, everything is closed anyway. If it fails,
        # all_reduce, all_to_all, and all_gather remain available.
        if t._a2a_beleg:
            try:
                t.byte_proof_broadcast()
            except Exception as e:
                logger.info(
                    "HTCCL-BAR1: broadcast byte-level proof could not run "
                    "-- %r. broadcast withdraws, the rest continues.", e,
                )
        # And the same principle once more for netz_pipe: its own kernel,
        # its own slots, its own counter lines, hence its own proof -- and
        # one over MULTIPLE rounds at that, because slot reuse never even
        # gets exercised in a single round. If it fails, mesh and ring
        # remain available.
        if t.pipe_an:
            try:
                t.byte_proof_pipe()
            except Exception as e:
                logger.info(
                    "HTCCL-BAR1: pipe byte-level proof could not run -- "
                    "%r. netz_pipe withdraws, mesh and ring continue.", e,
                )
    return t
