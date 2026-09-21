"""Slice 2 of the union weights arena: ONE physical allocation, two processes.

Slice 1 (``union_arena``) settled WHAT may be shared and at which offset.
This module is the physics: the owner creates the arena's pages with
exportable VMM handles, the peer imports them and maps the SAME physical
pages into its own address space. After this, a tensor bound at offset X in
the owner and a tensor bound at offset X in the peer are one set of bytes on
the card, not two copies.

WHY VMM AND NOT CUDA IPC. ``cudaIpcGetMemHandle`` works only on whole
``cudaMalloc`` allocations and cannot be rebased; the arena is one large
reservation whose interior we bind views into, which is exactly what the VMM
(``cuMemCreate`` / ``cuMemExportToShareableHandle`` / ``cuMemMap``) family is
for. The tree already uses it for the BAR1 lanes
(``distributed/device_communicators/vmm_utils.py``), and ``kv_vmm_backing``
already proves the "reserve VA, map handles, hand torch a bump pointer via a
pluggable allocator" shape on this rig. This module is that shape, minus the
partial-commit machinery the KV pool needs and plus the export.

WHY NOT THROUGH THE MEMORY SAVER. ``REUSE_DESIGN/understand_prior-art.md`` §5
records the blocker that stopped this before: TMS creates its pages in
``tms_csrc/utils.h`` ``cu_mem_create`` WITHOUT ``requestedHandleTypes``, so a
TMS page can never be exported. Under ``--flip-weights resident`` the weights
are not TMS-owned (06eb b5b32e), so the arena owns its own pages and the
blocker does not apply. Nothing here changes the saver.

LIFETIME, stated because a use-after-free here is a wrong-answer bug, not a
crash: the peer's mapping stays valid as long as the OWNER keeps its handles
retained -- CUDA reference-counts the physical allocation, and the peer's
import holds a reference of its own, so the pages survive the owner's death.
What does NOT survive is the owner's content: whoever writes the weights must
finish before the peer reads, which is the boot-time barrier slice 3 installs.
"""

from __future__ import annotations

import ctypes
import dataclasses
import logging
import os
import socket
import struct
import tempfile
import threading
from typing import List, Optional, Sequence

import torch
import torch.utils.cpp_extension
from torch.cuda.memory import CUDAPluggableAllocator

from sglang.srt.mem_cache.kv_vmm_backing import align_up, query_granularity
from sglang.srt.weg2.union_arena import UnionShareError

logger = logging.getLogger(__name__)

_drv = None
_instances = 0


def _driver():
    global _drv
    if _drv is None:
        from cuda.bindings import driver

        _drv = driver
    return _drv


def _check(result, label: str):
    drv = _driver()
    err = result[0] if isinstance(result, tuple) else result
    if err != drv.CUresult.CUDA_SUCCESS:
        raise UnionShareError(f"{label} failed: {err}")
    return result[1] if isinstance(result, tuple) and len(result) > 1 else None


def _stub_source(sfx: str) -> str:
    """A bump allocator over one reserved VA range (kv_vmm_backing's shape).

    torch has no public way to wrap a raw device pointer in a tensor. A
    pluggable allocator that hands out our own pointers is the supported
    route, and ``no_split`` keeps the caching allocator from carving them.
    """
    return f"""
#include <cstddef>
#include <cstdint>
#include <mutex>
extern "C" {{
static uintptr_t g_base_{sfx} = 0;
static size_t g_cursor_{sfx} = 0;
static size_t g_reserved_{sfx} = 0;
static std::mutex g_mu_{sfx};
void unionarena_set_{sfx}(uintptr_t b, size_t r){{
  std::lock_guard<std::mutex> lk(g_mu_{sfx}); g_base_{sfx}=b; g_reserved_{sfx}=r; g_cursor_{sfx}=0; }}
void* unionarena_malloc_{sfx}(size_t size, int device, void* stream){{
  std::lock_guard<std::mutex> lk(g_mu_{sfx});
  size_t need = g_cursor_{sfx} + ((size + 511) / 512 * 512);
  if (need > g_reserved_{sfx}) return 0;
  void* p = reinterpret_cast<void*>(g_base_{sfx} + g_cursor_{sfx});
  g_cursor_{sfx} = need;
  return p;
}}
void unionarena_free_{sfx}(void* ptr, size_t size, int device, void* stream){{}}
}}
"""


@dataclasses.dataclass
class _Mapping:
    offset: int
    size: int
    handle: object


class UnionVmmArena:
    """One VMM-backed byte range, mappable into a second process.

    ``tensor`` is a uint8 tensor over the whole arena; ``weights_arena.
    bind_arena_views`` binds parameter views into it at the manifest's
    offsets. Both processes see the same offsets over the same pages.
    """

    def __init__(self, device_id: int, total_bytes: int):
        global _instances
        _instances += 1
        drv = _driver()
        self.device_id = int(device_id)
        self.granularity = query_granularity(self.device_id)
        self.total_bytes = align_up(int(total_bytes), self.granularity)
        self._sfx = f"{os.getpid()}_{_instances}"
        self._mappings: List[_Mapping] = []
        self._owned_handles: List[object] = []
        self._closed = False
        with torch.cuda.device(self.device_id):
            self.base = int(
                _check(
                    drv.cuMemAddressReserve(self.total_bytes, self.granularity, 0, 0),
                    "cuMemAddressReserve",
                )
            )
        self._access = drv.CUmemAccessDesc()
        self._access.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        self._access.location.id = self.device_id
        self._access.flags = drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self._tensor: Optional[torch.Tensor] = None

    # ---------------------------------------------------------------- owner
    @classmethod
    def create(cls, device_id: int, total_bytes: int) -> "UnionVmmArena":
        """Allocate the arena's pages here, exportable to the peer.

        ONE handle for the whole range: the peer then needs one fd, not one
        per granule. ``kv_vmm_backing`` chunks because it commits and releases
        ranges at runtime; the union arena is allocated once and lives for the
        boot, so the chunked form would buy nothing and cost thousands of fds.
        """
        drv = _driver()
        self = cls(device_id, total_bytes)
        prop = drv.CUmemAllocationProp()
        prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = self.device_id
        # THE line the prior art called the blocker, on pages we own.
        prop.requestedHandleTypes = (
            drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
        with torch.cuda.device(self.device_id):
            handle = _check(
                drv.cuMemCreate(self.total_bytes, prop, 0), "cuMemCreate(exportable)"
            )
            self._owned_handles.append(handle)
            self._map(handle, 0, self.total_bytes)
        logger.info(
            "WEG2-UNION arena OWNER device=%d bytes=%d granularity=%d base=0x%x",
            self.device_id,
            self.total_bytes,
            self.granularity,
            self.base,
        )
        return self

    def export_fds(self) -> List[int]:
        """POSIX fds for this arena's handles; the caller owns and closes them."""
        drv = _driver()
        if not self._owned_handles:
            raise UnionShareError("this arena did not create its pages; nothing to export")
        fds = []
        for handle in self._owned_handles:
            fd = _check(
                drv.cuMemExportToShareableHandle(
                    handle,
                    drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                    0,
                ),
                "cuMemExportToShareableHandle(POSIX_FD)",
            )
            fds.append(int(fd))
        return fds

    # ----------------------------------------------------------------- peer
    @classmethod
    def attach(cls, device_id: int, total_bytes: int, fds: Sequence[int]) -> "UnionVmmArena":
        """Map the owner's pages here. ``fds`` in the owner's export order."""
        drv = _driver()
        self = cls(device_id, total_bytes)
        offset = 0
        with torch.cuda.device(self.device_id):
            for fd in fds:
                handle = _check(
                    drv.cuMemImportFromShareableHandle(
                        int(fd),
                        drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                    ),
                    "cuMemImportFromShareableHandle",
                )
                size = self.total_bytes if len(fds) == 1 else self.granularity
                self._map(handle, offset, size)
                offset += size
        logger.info(
            "WEG2-UNION arena PEER device=%d bytes=%d attached %d handle(s) base=0x%x",
            self.device_id,
            self.total_bytes,
            len(fds),
            self.base,
        )
        return self

    # --------------------------------------------------------------- shared
    def _map(self, handle, offset: int, size: int) -> None:
        drv = _driver()
        addr = self.base + offset
        _check(drv.cuMemMap(addr, size, 0, handle, 0), "cuMemMap")
        _check(drv.cuMemSetAccess(addr, size, [self._access], 1), "cuMemSetAccess")
        self._mappings.append(_Mapping(offset=offset, size=size, handle=handle))

    @property
    def tensor(self) -> torch.Tensor:
        """uint8 view of the whole arena (allocated once, cached)."""
        if self._tensor is None:
            self._tensor = self._make_tensor()
        return self._tensor

    def _make_tensor(self) -> torch.Tensor:
        out_dir = os.path.join(tempfile.gettempdir(), "sgl_weg2_union_arena", self._sfx)
        os.makedirs(out_dir, exist_ok=True)
        libname = f"sgl_weg2_union_arena_{self._sfx}"
        torch.utils.cpp_extension.load_inline(
            name=libname,
            cpp_sources=_stub_source(self._sfx),
            with_cuda=False,
            is_python_module=False,
            verbose=False,
            build_directory=out_dir,
        )
        so_path = f"{out_dir}/{libname}.so"
        lib = ctypes.CDLL(so_path)
        setter = getattr(lib, f"unionarena_set_{self._sfx}")
        setter.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        setter.restype = None
        setter(ctypes.c_void_p(self.base), ctypes.c_size_t(self.total_bytes))
        allocator = CUDAPluggableAllocator(
            so_path,
            f"unionarena_malloc_{self._sfx}",
            f"unionarena_free_{self._sfx}",
        ).allocator()
        self._pool = torch.cuda.MemPool(allocator, no_split=True)
        self._lib = lib
        with torch.cuda.device(self.device_id), torch.cuda.use_mem_pool(self._pool):
            t = torch.empty(self.total_bytes, dtype=torch.uint8, device="cuda")
        if int(t.data_ptr()) != self.base:
            raise UnionShareError(
                f"the arena tensor landed at 0x{int(t.data_ptr()):x}, not at the "
                f"reserved base 0x{self.base:x}; the pluggable allocator did not "
                f"hand back our bump pointer"
            )
        return t

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        drv = _driver()
        self._tensor = None
        for m in self._mappings:
            try:
                _check(drv.cuMemUnmap(self.base + m.offset, m.size), "cuMemUnmap")
            except Exception:  # a teardown must not mask the real error
                logger.warning("WEG2-UNION unmap failed at offset %d", m.offset)
        for m in self._mappings:
            try:
                drv.cuMemRelease(m.handle)
            except Exception:
                pass
        self._mappings = []
        self._owned_handles = []
        try:
            drv.cuMemAddressFree(self.base, self.total_bytes)
        except Exception:
            pass


# --------------------------------------------------------------------------
# The rendezvous: fds travel over a unix socket (SCM_RIGHTS), the manifest as
# plain bytes beside them. One socket per card per boot, under the boot's own
# /dev/shm directory, so a stale socket from a dead boot cannot be attached to.
# --------------------------------------------------------------------------

_HEADER = struct.Struct("<II")  # payload length, fd count


#: ``sockaddr_un.sun_path`` is 108 bytes including the terminator on Linux.
SUN_PATH_MAX = 107


def socket_path(union_dir: str, card: str) -> str:
    """One socket per card under the boot's own directory.

    The card UUID's tail is unique on this rig and keeps the name short. The
    LENGTH is checked here rather than left to ``bind()``: over the limit the
    kernel reports a bare ``OSError`` from a line that says nothing about path
    length, and the boot would look like a rendezvous failure instead of a
    misconfigured directory.
    """
    os.makedirs(union_dir, exist_ok=True)
    path = os.path.join(union_dir, f"u-{card[-12:]}.sock")
    if len(path.encode()) > SUN_PATH_MAX:
        raise UnionShareError(
            f"the rendezvous socket path is {len(path.encode())} bytes and a "
            f"unix socket takes at most {SUN_PATH_MAX}: {path!r}. Point "
            f"--union-dir at a short path (the boot uses /dev/shm/weg2-union-<tag>)."
        )
    return path


class UnionRendezvousServer:
    """Serves ``(manifest_text, fds)`` to peers until stopped.

    Threaded and repeatable on purpose: the peer group may restart, and a
    one-shot handoff would then leave a live owner beside a peer that can
    never attach -- which looks exactly like a hung boot.
    """

    def __init__(self, path: str, manifest_text: str, fds: Sequence[int]):
        self.path = path
        self._payload = manifest_text.encode()
        self._fds = [int(f) for f in fds]
        if os.path.exists(path):
            os.unlink(path)
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(path)
        self._sock.listen(8)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._serve, name="weg2-union-rendezvous", daemon=True
        )
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            try:
                conn.sendall(_HEADER.pack(len(self._payload), len(self._fds)))
                conn.sendall(self._payload)
                socket.send_fds(conn, [b"F"], self._fds)
            except OSError as e:
                logger.warning("WEG2-UNION rendezvous send failed: %s", e)
            finally:
                conn.close()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass
        try:
            os.unlink(self.path)
        except OSError:
            pass


def fetch_union(path: str, timeout_s: float = 120.0) -> tuple:
    """Connect to the owner and return ``(manifest_text, fds)``.

    Retries until ``timeout_s``: the peer group routinely reaches this point
    before the owner has finished loading its weights, and a refusal there
    would only move the wait into the launcher.
    """
    import time

    deadline = time.monotonic() + timeout_s
    last: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.connect(path)
                head = _recv_exact(s, _HEADER.size)
                length, n_fds = _HEADER.unpack(head)
                payload = _recv_exact(s, length)
                _msg, fds, _flags, _addr = socket.recv_fds(s, 1, n_fds)
                if len(fds) != n_fds:
                    raise UnionShareError(
                        f"the owner announced {n_fds} handle(s) and sent {len(fds)}"
                    )
                return payload.decode(), fds
        except (FileNotFoundError, ConnectionRefusedError, OSError) as e:
            last = e
            time.sleep(0.25)
    raise UnionShareError(
        f"no union arena owner at {path} within {timeout_s:.0f}s: {last}"
    )


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise UnionShareError("the owner closed the rendezvous mid-message")
        buf += chunk
    return buf
