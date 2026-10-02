"""L15-10 S3b: P reads D's held KV pages directly (hot handover = L15-15).

L1.5 keeps D's held prefix PHYSICALLY on the card through the P phase
(KEEP-SPLIT: the hold extents stay mapped and unreleased across the kv
pause). So the window L15-09 found missing -- "D pool still mapped while P's
kv is resumed" -- exists for exactly the rows a hot follow-up needs: D's hold
extents. P does not need D's pause moved (S4 of the design collapses for the
held part); it maps D's hold extents into its own address space and copies
the planned rows (l15_handover_copy) into its pool.

Pieces:

* D side -- :func:`export_hold_extents`: the hold extents of one base as
  POSIX fds (``tms_export_extent``; needs SGLANG_WEG2_VMM_EXPORTABLE=1 at boot,
  else the export refuses by name), and :func:`send_hold` / :func:`recv_hold`
  -- one JSON header plus the fds over a unix socket (SCM_RIGHTS, the BAR1
  lanes' vmm_utils transport).
* P side -- :func:`map_hold_extent`: import one fd, map it at a fresh VA
  (vmm_utils.import_and_map_alloc) and wrap it as a uint8 tensor through the
  CUDA array interface (no extension build); the caller views it with D's row
  layout.

Every failure raises L15ShareError; the caller falls back to today's store
read (hot_handover ``fallen_back``).
"""

from __future__ import annotations

import ctypes
import json
import os
import socket
import struct
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


class L15ShareError(RuntimeError):
    """The held pages cannot be shared with P (named); nothing was mapped."""


@dataclass(frozen=True)
class HoldExtent:
    base_index: int
    offset: int
    size: int


# -- D side ------------------------------------------------------------------

def _export_symbol():
    from sglang.srt.utils.torch_memory_saver_adapter import _weg2_ring_symbol

    fn = _weg2_ring_symbol("tms_export_extent")
    if fn is None:
        return None
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                   ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint64)]
    return fn


def export_hold_extents(base_ptr: int, extents: Sequence[Tuple[int, int]],
                        export=None) -> List[Tuple[int, int, int]]:
    """``[(offset, size, fd), ...]`` for the hold extents of one base. The
    caller owns the fds. ``export`` is the tms_export_extent callable (tests
    pass a fake)."""
    fn = export if export is not None else _export_symbol()
    if fn is None:
        raise L15ShareError("tms_export_extent missing (old saver build)")
    out: List[Tuple[int, int, int]] = []
    try:
        for off, _hi in extents:
            fd = ctypes.c_int(-1)
            size = ctypes.c_uint64(0)
            rc = int(fn(ctypes.c_void_p(int(base_ptr)), ctypes.c_uint64(int(off)),
                        ctypes.byref(fd), ctypes.byref(size)))
            if rc != 0:
                raise L15ShareError(
                    "export of extent @%d of base %#x refused rc=%d (-2: not a "
                    "span extent; CUresult: not exportable, set "
                    "SGLANG_WEG2_VMM_EXPORTABLE=1)" % (off, base_ptr, rc))
            out.append((int(off), int(size.value), int(fd.value)))
    except BaseException:
        for _o, _s, f in out:
            try:
                os.close(f)
            except OSError:
                pass
        raise
    return out


def send_hold(sock: socket.socket, header: dict, fds: Sequence[int]) -> None:
    """One message: 4-byte length + JSON header, then every fd (one
    SCM_RIGHTS message each, vmm_utils framing)."""
    from sglang.srt.distributed.device_communicators.vmm_utils import _send_fd

    blob = json.dumps(header, sort_keys=True).encode()
    sock.sendall(struct.pack("<I", len(blob)) + blob)
    for i, fd in enumerate(fds):
        _send_fd(sock, int(fd), int(header.get("rank", 0)), i)


def recv_hold(sock: socket.socket) -> Tuple[dict, List[int]]:
    from sglang.srt.distributed.device_communicators.vmm_utils import _recv_fd

    n_raw = _recv_exactly(sock, 4)
    (n,) = struct.unpack("<I", n_raw)
    header = json.loads(_recv_exactly(sock, n).decode())
    fds: List[int] = []
    try:
        for _ in range(int(header.get("n_fds", 0))):
            got = _recv_fd(sock)
            if got is None:
                raise L15ShareError("hold share: peer closed before all fds")
            fds.append(got[2])
    except BaseException:
        for f in fds:
            try:
                os.close(f)
            except OSError:
                pass
        raise
    return header, fds


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise L15ShareError("hold share: connection closed mid-header")
        buf += chunk
    return buf


# -- P side ------------------------------------------------------------------

class _CAI:
    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (int(nbytes),), "typestr": "|u1",
            "data": (int(ptr), False), "version": 3}


def map_hold_extent(fd: int, size: int, device_id: int, peer_rank: int):
    """Import D's extent fd, map it at a fresh VA, return a uint8 CUDA tensor
    over it (the mapping stays as long as the process; P unmaps nothing here,
    the extent is D's and D releases it at its wake)."""
    import torch

    from sglang.srt.distributed.device_communicators.vmm_utils import (
        import_and_map_alloc,
    )

    try:
        va = import_and_map_alloc(None, int(fd), int(size), int(device_id),
                                  use_fabric=False, peer_rank=int(peer_rank))
    except Exception as exc:  # noqa: BLE001 -- named for the fallback
        raise L15ShareError("hold share: import/map of %d bytes failed: %r"
                            % (size, exc)) from exc
    return torch.as_tensor(_CAI(va, size), device="cuda:%d" % int(device_id))


class HoldMapper:
    """P side: every import of one take/deposit, released together.

    An imported extent stays PHYSICALLY alive while P maps it (the import
    handle is released right after cuMemMap, the mapping is the reference) and
    while P holds the received fd. D releases its kept extents at a later
    sleep; a mapping P never undoes pins that VRAM on the shared card for the
    rest of the process -- one whole hold-extent set per hot admission. So a
    take maps through this object (one mapping per fd, reused across the KV
    and the anchor pieces) and :meth:`close` -- after the stream has finished
    the copies -- unmaps every VA and closes every fd it was handed.
    """

    def __init__(self, device_id: int, map_fn=None, unmap_fn=None,
                 sync_fn=None):
        self.device_id = int(device_id)
        self._map_fn = map_fn or (lambda fd, size: map_hold_extent(
            fd, size, self.device_id, 0))
        self._unmap_fn = unmap_fn or _unmap_va
        self._sync_fn = sync_fn or self._sync
        self._maps = {}      # fd -> (tensor, size)
        self._fds: List[int] = []

    def own_fds(self, fds: Sequence[int]) -> Sequence[int]:
        self._fds.extend(int(f) for f in fds)
        return fds

    def fetch(self, fetch_fn, rank: int):
        d, fds = fetch_fn(rank)
        self.own_fds(fds)
        return d, fds

    def __call__(self, fd: int, size: int):
        got = self._maps.get(int(fd))
        if got is not None:
            if got[1] != int(size):
                raise L15ShareError("fd %d mapped as %d bytes, asked %d"
                                    % (fd, got[1], size))
            return got[0]
        t = self._map_fn(int(fd), int(size))
        self._maps[int(fd)] = (t, int(size))
        return t

    @property
    def mapped(self) -> int:
        return len(self._maps)

    def _sync(self) -> None:
        import torch

        torch.cuda.current_stream(self.device_id).synchronize()

    def close(self) -> Tuple[int, int]:
        """(unmapped, fds_closed); never raises (a failed unmap is counted
        out, the rest is still released)."""
        unmapped = 0
        if self._maps:
            try:
                self._sync_fn()
            except Exception:  # noqa: BLE001 -- release what we can
                pass
        while self._maps:
            _fd, (t, size) = self._maps.popitem()
            try:
                self._unmap_fn(int(t.data_ptr()), size)
                unmapped += 1
            except Exception:  # noqa: BLE001
                pass
        closed = 0
        while self._fds:
            try:
                os.close(self._fds.pop())
                closed += 1
            except OSError:
                pass
        return unmapped, closed


def _unmap_va(va: int, size: int) -> None:
    from sglang.srt.distributed.device_communicators.vmm_utils import (
        release_mappings,
    )

    release_mappings([(int(va), int(size), [(0, int(size))])])
