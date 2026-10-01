# SPDX-License-Identifier: Apache-2.0
"""VA-stable BAR1 detach / re-attach for a sleeping barlink group (prototype).

WHY. A 3080 has 256 MiB of BAR1 and no Resizable BAR. One model's two groups
already take ~224 MiB (P 24 + 96, D 16 + 32 + 40 MiB per rank, plus the flip
lanes). With two models booted, a sleeping group must give its BAR1 back
and take it again at wake (DUAL-MODEL-FLIP-KONZEPT-1001.md section 8.3).
Today it keeps it (``weg2_memory_saver.py`` "untouchable", no rebuild path).

THE CONSTRAINT. The peer pointers are baked into the captured CUDA graphs
(``barlink_bar1.py`` pipe-direct capture comment), so a plain ``close()`` +
new transport would need a re-capture. What has to survive is every DEVICE
address a kernel argument can hold:

* the owner's receive buffer -- a VMM allocation with its own VA
  reservation; untouched here;
* each writer's view of a peer region -- ``cudaHostRegister(IoMemory)`` of
  an mmap of the peer's ``resource1_wc`` slice, device pointer from
  ``cudaHostGetDevicePointer``. THIS is what a detach drops.

WHAT HOLDS THE BAR1 PAGES. The writer's dma-buf attachment through
``/dev/dmabuf_holder`` (``dma_buf_map_attachment`` programs the destination
card's BAR1 pages, ``barlink_bar1.py`` module header step 4). Releasing the
hold unprograms them; the exported fd itself stays open in the writer
(``t._foreign_fds``), so a re-attach needs no fd exchange, no collective and
no JIT -- one HOLD ioctl per region.

TWO WAYS BACK, tried in this order at re-attach:

1. **in place** -- the new hold lands on the SAME BAR1 offset: the old mmap
   and registration are still valid, only the holder handle is swapped.
2. **fixed remap** -- the offset moved (another group took the range while
   this one slept): unregister, mmap the new ``resource1_wc`` slice over the
   SAME host VA (``MAP_FIXED``), register again and REQUIRE the device
   pointer to come back unchanged. If it does not, the graphs would point at
   the old range: the re-attach refuses by name and never runs a kernel.

``mode="keep_map"`` releases only the holds (way 1 is then free when the
offset is unchanged). ``mode="remap"`` also unregisters and parks a PROT_NONE
placeholder on the VA so nothing else can be mapped there while the group
sleeps; the re-attach then always takes way 2.

Callers must quiesce first: no collective of this transport may be in flight
on any rank between detach and re-attach (host barrier before both).

PROTOTYPE LIMIT (M1 work item): ``_bind_region`` maps with Python's
``mmap.mmap``; that object would munmap its range when collected, so the
remap path keeps the original objects alive in ``t._detach_keepalive``.
The M1 version moves ``_bind_region`` onto :class:`FixedMap` and drops that
list.
"""
from __future__ import annotations

import ctypes
import dataclasses
import mmap
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

PROT_NONE = 0
PROT_READ = 1
PROT_WRITE = 2
MAP_SHARED = 0x01
MAP_PRIVATE = 0x02
MAP_FIXED = 0x10
MAP_ANONYMOUS = 0x20
_MAP_FAILED = ctypes.c_void_p(-1).value


class Bar1ReattachRefused(RuntimeError):
    """A re-attach that would leave a captured graph pointing elsewhere."""


# ---------------------------------------------------------------------------
# pure helpers (unit-tested without a GPU)
# ---------------------------------------------------------------------------


def contiguous_bar1_slice(sg: Sequence[Tuple[int, int]], base: int, end: int) -> Tuple[int, int]:
    """(offset in BAR1, contiguous length) of the sg entries inside [base, end).

    The same rule ``_bind_region`` applies: only the contiguous beginning of
    the in-aperture entries is mappable as one piece. (0, 0) when no entry
    lies inside the aperture.
    """
    hits = sorted((a, n) for a, n in sg if base <= a < end)
    if not hits:
        return 0, 0
    start = hits[0][0]
    length = 0
    expected = start
    for a, n in hits:
        if a != expected:
            break
        length += n
        expected += n
    return start - base, length


@dataclass(frozen=True)
class RegionState:
    """What a captured graph depends on for one mapped peer region."""

    peer: int
    kind: str              # "payload" | "flag"
    bar1_offset: int
    length: int            # contiguous length that was mapped
    lead_in: int           # host address - page-aligned mmap start
    reg_address: int       # page-aligned host VA the registration used
    dev_ptr: int           # device pointer kernels were captured with


def reattach_plan(old: RegionState, new_offset: int, new_length: int, page: int) -> Tuple[str, str]:
    """('in_place' | 'remap' | 'refuse', reason) for one region. Pure.

    A shorter contiguous slice than before refuses (the captured geometry
    would overrun it); so does a different sub-page lead-in, because the
    region would then start at a different host -- and device -- address
    inside the same pages.
    """
    if new_length < old.length:
        return "refuse", (f"peer {old.peer} {old.kind}: contiguous BAR1 slice shrank "
                          f"{old.length} -> {new_length} B")
    if new_offset == old.bar1_offset:
        return "in_place", "same BAR1 offset"
    if new_offset % page != old.lead_in:
        return "refuse", (f"peer {old.peer} {old.kind}: lead-in {new_offset % page} != "
                          f"{old.lead_in} -- the region would move inside its pages")
    return "remap", f"offset moved {old.bar1_offset:#x} -> {new_offset:#x}"


# ---------------------------------------------------------------------------
# libc mmap with a fixed address
# ---------------------------------------------------------------------------

_libc = None


def _c():
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(None, use_errno=True)
        _libc.mmap.restype = ctypes.c_void_p
        _libc.mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int, ctypes.c_long)
        _libc.munmap.restype = ctypes.c_int
        _libc.munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    return _libc


class FixedMap:
    """An mmap at a caller-chosen address; ``close()`` unmaps exactly it."""

    def __init__(self, addr: int, length: int, prot: int, flags: int, fd: int = -1, offset: int = 0):
        p = _c().mmap(ctypes.c_void_p(addr), length, prot, flags | MAP_FIXED, fd, offset)
        if p is None or p == _MAP_FAILED or int(p) != int(addr):
            err = ctypes.get_errno()
            raise OSError(err, f"mmap(MAP_FIXED, addr={addr:#x}, len={length}, off={offset:#x}) "
                               f"-> {p!r}: {os.strerror(err)}")
        self.addr = int(addr)
        self.length = int(length)
        self._open = True

    def close(self) -> None:
        if self._open:
            _c().munmap(ctypes.c_void_p(self.addr), self.length)
            self._open = False


# ---------------------------------------------------------------------------
# detach / re-attach on a live BarlinkBar1Transport
# ---------------------------------------------------------------------------


@dataclass
class DetachRecord:
    mode: str
    regions: List[RegionState]
    ms: float


def _regions(t):
    for peer, z in sorted(t._peers.items()):
        for idx, kind in ((0, "payload"), (1, "flag")):
            yield peer, idx, kind, (z.payload if idx == 0 else z.flag)


def detach(t, mode: str = "keep_map") -> DetachRecord:
    """Drop this rank's BAR1 attachments of every peer region (see module doc)."""
    if mode not in ("keep_map", "remap"):
        raise ValueError(f"mode {mode!r}")
    if getattr(t, "_bar1_detached", None):
        raise RuntimeError("transport already detached")
    import torch

    torch.cuda.synchronize(t.device)
    page = mmap.PAGESIZE
    t0 = time.time()
    regions: List[RegionState] = []
    keep = getattr(t, "_detach_keepalive", None)
    if keep is None:
        keep = t._detach_keepalive = []
    for peer, idx, kind, a in _regions(t):
        regions.append(RegionState(
            peer=peer, kind=kind, bar1_offset=a.bar1_offset, length=a.length,
            lead_in=a.host_address - a.reg_address, reg_address=a.reg_address,
            dev_ptr=a.dev_ptr))
        if mode == "remap":
            t._cuda.unregister(a.reg_address)
            m_len = a.length + (a.host_address - a.reg_address)
            m_len = (m_len + page - 1) // page * page
            # Replace the aperture mapping by an inaccessible placeholder at
            # the same VA: atomically, so the range is never free to others.
            ph = FixedMap(a.reg_address, m_len, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS)
            keep.append(a.mmap_obj)   # see PROTOTYPE LIMIT in the module doc
            a.mmap_obj = ph
        t._holder.release(a.holder_handle)
    t._bar1_detached = mode
    return DetachRecord(mode=mode, regions=regions, ms=(time.time() - t0) * 1000)


def reattach(t, rec: DetachRecord) -> List[dict]:
    """Take the BAR1 attachments back; device pointers must not move."""
    from sglang.srt.distributed.device_communicators.barlink_bar1 import bar1_window

    if getattr(t, "_bar1_detached", None) != rec.mode:
        raise RuntimeError(f"transport not detached in mode {rec.mode!r}")
    page = mmap.PAGESIZE
    out = []
    old_by = {(r.peer, r.kind): r for r in rec.regions}
    for peer, idx, kind, a in _regions(t):
        t0 = time.time()
        old = old_by[(peer, kind)]
        dst_bdf = t.bdfs[peer]
        win = bar1_window(dst_bdf)
        handle, sg, _total = t._holder.hold(t._foreign_fds[peer][idx], t.bdfs[t.rank])
        off, length = contiguous_bar1_slice([(e.dma_address, e.length) for e in sg], win.base, win.end)
        how, why = reattach_plan(old, off, length, page)
        if how == "in_place" and rec.mode == "remap":
            how = "remap"  # the registration was dropped: map again, same VA
        if how == "refuse":
            t._holder.release(handle)
            raise Bar1ReattachRefused(why)
        if how == "remap":
            if rec.mode == "keep_map":
                t._cuda.unregister(a.reg_address)
            m_off = off - old.lead_in
            m_len = (length + old.lead_in + page - 1) // page * page
            res_fd = os.open(f"/sys/bus/pci/devices/{dst_bdf}/resource1_wc", os.O_RDWR | os.O_SYNC)
            try:
                fm = FixedMap(old.reg_address, m_len, PROT_READ | PROT_WRITE, MAP_SHARED, res_fd, m_off)
            finally:
                os.close(res_fd)
            if rec.mode == "keep_map":
                if getattr(t, "_detach_keepalive", None) is None:
                    t._detach_keepalive = []
                t._detach_keepalive.append(a.mmap_obj)
            else:
                # The PROT_NONE placeholder: MAP_FIXED above already replaced its
                # pages. Closing it would munmap the NEW aperture mapping.
                a.mmap_obj._open = False
            t._cuda.register_io(old.reg_address, m_len)
            dev = t._cuda.dev_ptr(old.reg_address) + old.lead_in
            if dev != old.dev_ptr:
                raise Bar1ReattachRefused(
                    f"peer {peer} {kind}: device pointer moved {old.dev_ptr:#x} -> {dev:#x} "
                    f"after re-registration at the same host VA -- captured graphs would "
                    f"write to the old address")
            new = dataclasses.replace(a, bar1_offset=off, length=length, mmap_obj=fm,
                                      holder_handle=handle)
        else:
            new = dataclasses.replace(a, holder_handle=handle)
        z = t._peers[peer]
        if idx == 0:
            z.payload = new
        else:
            z.flag = new
        out.append({"peer": peer, "kind": kind, "how": how, "why": why,
                    "old_offset": old.bar1_offset, "new_offset": off,
                    "ms": round((time.time() - t0) * 1000, 2)})
    t._bar1_detached = None
    return out
