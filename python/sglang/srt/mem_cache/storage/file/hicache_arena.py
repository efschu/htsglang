"""ONE HiCache for all six ranks: the shared-memory page arena (arena.c).

BAUPLAN_SHM_ARENA_0916 (user order 2026-09-16: "ich moechte EINEN L2 haben ...
jeder rank schreibt und liest in den gleichen hicache"). One file per slot
class under /dev/shm, mapped by every rank process of both groups; a canonical
page is written into it by the ranks that own its extents (PP stages, TP
shards) and read out of it by any rank that needs its own extents. The disk
store (HiCacheFile's directory) stays the cold tier behind it: a miss reads
from disk into the arena, eviction writes complete pages to disk first.

This module is the ctypes/mmap side; the protocol lives in arena.c. Nothing
here needs CUDA: the arena is a plain shared mapping until stage 3 (direct
DMA) pins it with cudaHostRegister.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import mmap
import os
import subprocess
import tempfile
import threading
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arena.c")
_lock = threading.Lock()
_lib: Optional[ctypes.CDLL] = None
_failed = False

#: every extent boundary must sit on this granule (arena.c GRANULE)
GRANULE = 256


def _load_lib() -> Optional[ctypes.CDLL]:
    global _lib, _failed
    if _lib is not None:
        return _lib
    if _failed:
        return None
    with _lock:
        if _lib is not None:
            return _lib
        if _failed:
            return None
        try:
            with open(_SRC, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()[:16]
            so = os.path.join(
                tempfile.gettempdir(), f"sglang_hicache_arena_{digest}_{os.getuid()}.so"
            )
            if not os.path.exists(so):
                tmp = f"{so}.{os.getpid()}.tmp"
                subprocess.run(
                    ["gcc", "-O2", "-shared", "-fPIC", "-o", tmp, _SRC],
                    check=True, capture_output=True, timeout=120,
                )
                os.replace(tmp, so)
            lib = ctypes.CDLL(so)
            i64 = ctypes.c_int64
            p_i64 = ctypes.POINTER(i64)
            p_u64 = ctypes.POINTER(ctypes.c_uint64)
            p_i8 = ctypes.POINTER(ctypes.c_int8)
            p_u8 = ctypes.c_void_p
            lib.arena_layout.restype = i64
            lib.arena_layout.argtypes = [i64, i64, p_i64]
            lib.arena_init.restype = ctypes.c_int
            lib.arena_init.argtypes = [p_u8, i64, i64]
            lib.arena_write.restype = i64
            lib.arena_write.argtypes = [p_u8, i64, p_u64, p_u64, p_i64, p_i64, p_i64, p_i64,
                                        ctypes.POINTER(ctypes.c_void_p), p_i8]
            lib.arena_read.restype = i64
            lib.arena_read.argtypes = [p_u8, i64, p_u64, p_u64, p_i64, p_i64, p_i64, p_i64,
                                       ctypes.POINTER(ctypes.c_void_p), p_i8]
            lib.arena_lookup.restype = i64
            lib.arena_lookup.argtypes = [p_u8, i64, p_u64, p_u64, p_i8]
            lib.arena_evict_candidates.restype = i64
            lib.arena_evict_candidates.argtypes = [p_u8, i64, p_i64, p_u64, p_u64, p_i64, p_u64, i64]
            lib.arena_slot_ptr.restype = ctypes.c_void_p
            lib.arena_slot_ptr.argtypes = [p_u8, i64]
            lib.arena_free_slots.restype = None
            lib.arena_free_slots.argtypes = [p_u8, i64, p_i64]
            lib.arena_reap_stale.restype = i64
            lib.arena_reap_stale.argtypes = [p_u8]
            lib.arena_stats.restype = None
            lib.arena_stats.argtypes = [p_u8, p_i64]
            _lib = lib
            return lib
        except Exception as e:  # noqa: BLE001 - the arena is optional
            _failed = True
            logger.warning("[arena] helper unavailable (%s: %s); no shared arena.",
                           type(e).__name__, str(e)[:200])
            return None


def key128(stem: str) -> tuple[int, int]:
    """The 128-bit key of a store stem; low word never 0 or ~0 (index sentinels)."""
    d = hashlib.blake2b(stem.encode("utf-8"), digest_size=16).digest()
    lo = int.from_bytes(d[:8], "little")
    hi = int.from_bytes(d[8:], "little")
    if lo in (0, (1 << 64) - 1):
        lo = 1
    return lo, hi


class ShmArena:
    """One arena file: `slots` slots of `slot_bytes` each.

    `path` is created (sparse, sized by arena_layout) when absent; every
    process that opens the same path shares the same pages. Statuses follow
    arena.c: write 0 partial / 1 completed / 2 already / 3 refused / 4 full;
    read 0 ok / 1 absent / 2 width / 3 refused.
    """

    def __init__(self, path: str, slot_bytes: int, slots: int):
        lib = _load_lib()
        if lib is None:
            raise RuntimeError("arena helper unavailable")
        self._lib = lib
        self.path = path
        self.slot_bytes = int(slot_bytes)
        self.slots = int(slots)
        out = (ctypes.c_int64 * 6)()
        total = lib.arena_layout(self.slots, self.slot_bytes, out)
        self.file_bytes = int(total)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            st = os.fstat(fd)
            if st.st_size < self.file_bytes:
                os.ftruncate(fd, self.file_bytes)
            self._mm = mmap.mmap(fd, self.file_bytes, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        self._base = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(self._mm)))
        rc = lib.arena_init(self._base, self.slots, self.slot_bytes)
        if rc < 0:
            raise RuntimeError(
                f"{path} holds an arena of another geometry (not {slots} x {slot_bytes})"
            )
        self.fresh = rc == 0

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _keys(stems: Sequence[str]):
        n = len(stems)
        lo = (ctypes.c_uint64 * n)()
        hi = (ctypes.c_uint64 * n)()
        for i, s in enumerate(stems):
            a, b = key128(s)
            lo[i] = a
            hi[i] = b
        return lo, hi

    @staticmethod
    def _extents(extents):
        n = len(extents)
        n_ext = (ctypes.c_int64 * n)(*[len(e) for e in extents])
        flat_off = [int(o) for e in extents for o, _ in e]
        flat_len = [int(l) for e in extents for _, l in e]
        m = max(1, len(flat_off))
        return n_ext, (ctypes.c_int64 * m)(*flat_off), (ctypes.c_int64 * m)(*flat_len)

    # -- protocol ----------------------------------------------------------
    def lookup(self, stems: Sequence[str]) -> list[bool]:
        n = len(stems)
        if n == 0:
            return []
        lo, hi = self._keys(stems)
        st = (ctypes.c_int8 * n)()
        self._lib.arena_lookup(self._base, n, lo, hi, st)
        return [bool(x) for x in st]

    def write(self, stems, totals, extents, payload_ptrs) -> list[int]:
        n = len(stems)
        if n == 0:
            return []
        lo, hi = self._keys(stems)
        n_ext, c_off, c_len = self._extents(extents)
        c_tot = (ctypes.c_int64 * n)(*[int(t) for t in totals])
        c_pay = (ctypes.c_void_p * n)(*[int(p) for p in payload_ptrs])
        st = (ctypes.c_int8 * n)()
        self._lib.arena_write(self._base, n, lo, hi, c_tot, n_ext, c_off, c_len, c_pay, st)
        return list(st)

    def read(self, stems, totals, extents, out_ptrs) -> list[int]:
        n = len(stems)
        if n == 0:
            return []
        lo, hi = self._keys(stems)
        n_ext, c_off, c_len = self._extents(extents)
        c_tot = (ctypes.c_int64 * n)(*[int(t) for t in totals])
        c_out = (ctypes.c_void_p * n)(*[int(p) for p in out_ptrs])
        st = (ctypes.c_int8 * n)()
        self._lib.arena_read(self._base, n, lo, hi, c_tot, n_ext, c_off, c_len, c_out, st)
        return list(st)

    def evict_candidates(self, want: int, keep_stems: Sequence[str] = ()):
        """(slot, key_lo, key_hi, total) of up to `want` slots now EVICTING."""
        want = int(want)
        if want <= 0:
            return []
        slots = (ctypes.c_int64 * want)()
        lo = (ctypes.c_uint64 * want)()
        hi = (ctypes.c_uint64 * want)()
        tot = (ctypes.c_int64 * want)()
        keep = [key128(s)[0] for s in keep_stems]
        c_keep = (ctypes.c_uint64 * max(1, len(keep)))(*keep)
        got = self._lib.arena_evict_candidates(self._base, want, slots, lo, hi, tot, c_keep, len(keep))
        return [(int(slots[i]), int(lo[i]), int(hi[i]), int(tot[i])) for i in range(got)]

    def slot_view(self, slot: int, nbytes: int) -> memoryview:
        off = int(self._lib.arena_slot_ptr(self._base, int(slot))) - int(self._base.value)
        return memoryview(self._mm)[off:off + int(nbytes)]

    def free_slots(self, slots: Sequence[int]) -> None:
        n = len(slots)
        if n == 0:
            return
        c = (ctypes.c_int64 * n)(*[int(s) for s in slots])
        self._lib.arena_free_slots(self._base, n, c)

    def reap_stale(self) -> int:
        return int(self._lib.arena_reap_stale(self._base))

    def stats(self) -> dict:
        out = (ctypes.c_int64 * 4)()
        self._lib.arena_stats(self._base, out)
        return {"slots": int(out[0]), "complete": int(out[1]), "claimed": int(out[2]),
                "slot_bytes": int(out[3]), "file_bytes": self.file_bytes}

    def close(self) -> None:
        try:
            self._mm.close()
        except Exception:  # noqa: BLE001
            pass
