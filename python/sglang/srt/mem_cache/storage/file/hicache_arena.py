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
import functools
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
GRANULE = 1  # coverage is an interval list; any byte range counts


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
                                        ctypes.POINTER(ctypes.c_void_p),
                                        ctypes.POINTER(ctypes.c_char_p), p_i8]
            lib.arena_slot_stem.restype = ctypes.c_char_p
            lib.arena_slot_stem.argtypes = [p_u8, i64]
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
            lib.arena_find_slots.restype = i64
            lib.arena_find_slots.argtypes = [p_u8, i64, p_u64, p_u64, p_i64, p_i8]
            lib.arena_find_stems.restype = i64
            lib.arena_find_stems.argtypes = [p_u8, i64, ctypes.POINTER(ctypes.c_char_p), p_i64, p_i8]
            lib.arena_claim.restype = i64
            lib.arena_claim.argtypes = [p_u8, i64, p_u64, p_u64, p_i64,
                                        ctypes.POINTER(ctypes.c_char_p), p_i64, p_i64, p_i8]
            lib.arena_complete.restype = i64
            lib.arena_complete.argtypes = [p_u8, i64, p_i64, p_i64, p_i64, p_i64, p_i64, p_i8]
            lib.arena_ref_slots.restype = i64
            lib.arena_ref_slots.argtypes = [p_u8, i64, p_i64, ctypes.c_int32]
            lib.arena_data_offset.restype = i64
            lib.arena_data_offset.argtypes = [p_u8]
            _lib = lib
            return lib
        except Exception as e:  # noqa: BLE001 - the arena is optional
            _failed = True
            logger.warning("[arena] helper unavailable (%s: %s); no shared arena.",
                           type(e).__name__, str(e)[:200])
            return None


@functools.lru_cache(maxsize=1 << 20)  # #1438: the same stems are hashed for find/ref/draft; cached
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
        c_stems = (ctypes.c_char_p * n)(*[s.encode("utf-8") for s in stems])
        st = (ctypes.c_int8 * n)()
        self._lib.arena_write(self._base, n, lo, hi, c_tot, n_ext, c_off, c_len, c_pay, c_stems, st)
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

    # -- Stufe 3 (#1424): address pages in place --------------------------
    def find_slots(self, stems: Sequence[str]) -> list[tuple[int, int]]:
        """(slot, state) per stem; slot -1 when absent. state 2 = COMPLETE.
        #1439: hashed in C (arena_find_stems) -- one call for a 100k prefix."""
        n = len(stems)
        if n == 0:
            return []
        c_stems = (ctypes.c_char_p * n)(*[s.encode("utf-8") for s in stems])
        slots = (ctypes.c_int64 * n)()
        st = (ctypes.c_int8 * n)()
        self._lib.arena_find_stems(self._base, n, c_stems, slots, st)
        return list(zip(slots, st))

    def find_slots_np(self, stems: Sequence[str]):
        """Posten 2 (18.09.): (slots int64[n], states int8[n]) as numpy views --
        no 520k-tuple Python list (xsn306: zip 95 ms + lead-list 26 ms + the
        per-element loops downstream)."""
        import numpy as np
        n = len(stems)
        if n == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int8)
        c_stems = (ctypes.c_char_p * n)(*[s.encode("utf-8") for s in stems])
        slots = np.empty((n,), dtype=np.int64)
        st = np.empty((n,), dtype=np.int8)
        self._lib.arena_find_stems(self._base, n,
                                   c_stems,
                                   slots.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                                   st.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)))
        return slots, st

    def ref_slots_np(self, slots, delta: int) -> int:
        """ref_slots over a numpy int64 array (zero-copy pointer)."""
        import numpy as np
        a = np.ascontiguousarray(np.asarray(slots, dtype=np.int64))
        n = int(a.shape[0])
        if n == 0:
            return 0
        return int(self._lib.arena_ref_slots(self._base, n,
                                             a.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                                             int(delta)))

    def find_states(self, stems: Sequence[str]) -> list[int]:
        """#1439: the states only (0 absent/free, 1 claimed, 2 complete), by stem, hashed in C."""
        n = len(stems)
        if n == 0:
            return []
        c_stems = (ctypes.c_char_p * n)(*[s.encode("utf-8") for s in stems])
        slots = (ctypes.c_int64 * n)()
        st = (ctypes.c_int8 * n)()
        self._lib.arena_find_stems(self._base, n, c_stems, slots, st)
        return list(st)

    def claim_slots(self, stems: Sequence[str], totals: Sequence[int]) -> list[tuple[int, int, int]]:
        """#1427 direct writes: (slot, status, generation) per stem. status
        0 = fresh claim, 1 = join an earlier writer's claim, 2 = already
        COMPLETE, 3 = too large, 4 = no free slot (slot -1)."""
        n = len(stems)
        if n == 0:
            return []
        lo, hi = self._keys(stems)
        c_tot = (ctypes.c_int64 * n)(*[int(t) for t in totals])
        c_stems = (ctypes.c_char_p * n)(*[s.encode("utf-8") for s in stems])
        slots = (ctypes.c_int64 * n)()
        gens = (ctypes.c_int64 * n)()
        st = (ctypes.c_int8 * n)()
        self._lib.arena_claim(self._base, n, lo, hi, c_tot, c_stems, slots, gens, st)
        return [(int(slots[i]), int(st[i]), int(gens[i])) for i in range(n)]

    def complete_slots(self, slots: Sequence[int], gens: Sequence[int], extents) -> list[int]:
        """#1427: merge this writer's extents (same shape for every slot) into
        the coverage and flip COMPLETE when the page is full. status per slot:
        1 completed now, 0 merged but not full, 2 already complete, 3 lost."""
        n = len(slots)
        if n == 0:
            return []
        ext = [tuple(extents)] * n
        n_ext, c_off, c_len = self._extents(ext)
        c_slots = (ctypes.c_int64 * n)(*[int(s) for s in slots])
        c_gens = (ctypes.c_int64 * n)(*[int(g) for g in gens])
        st = (ctypes.c_int8 * n)()
        self._lib.arena_complete(self._base, n, c_slots, c_gens, n_ext, c_off, c_len, st)
        return list(st)

    def ref_slots(self, slots: Sequence[int], delta: int) -> int:
        """Reader references: +1 pins COMPLETE slots against eviction, -1 releases.
        Returns how many slots took the delta."""
        n = len(slots)
        if n == 0:
            return 0
        c = (ctypes.c_int64 * n)(*[int(s) for s in slots])
        return int(self._lib.arena_ref_slots(self._base, n, c, int(delta)))

    def data_offset(self) -> int:
        """Byte offset of slot 0's data inside the mapping."""
        return int(self._lib.arena_data_offset(self._base))

    def slot_stem(self, slot: int) -> str:
        """The store stem recorded in the slot header (any rank may evict it)."""
        raw = self._lib.arena_slot_stem(self._base, int(slot))
        return (raw or b"").decode("utf-8", "replace")

    def slot_ptr(self, slot: int) -> int:
        """Address of slot data inside the mapping (for a C read straight into the slot)."""
        return int(self._lib.arena_slot_ptr(self._base, int(slot)))

    def slot_view(self, slot: int, nbytes: int) -> memoryview:
        off = int(self._lib.arena_slot_ptr(self._base, int(slot))) - int(self._base.value)
        return memoryview(self._mm)[off:off + int(nbytes)]

    def free_slots(self, slots: Sequence[int]) -> None:
        # xsn328: who frees which slots (D's dormant re-reads found P's pages FREE/CLAIMED again)
        _fn = getattr(type(self), "_free_log_n", 0) + 1
        type(self)._free_log_n = _fn
        if _fn <= 16 or _fn % 256 == 0:
            import traceback as _tb
            _caller = "".join(_tb.format_stack(limit=4)[:-1]).strip().replace("\n", " | ")[-300:]
            logger.info("ARENA-FREE n=%d slots=%d first=%s caller=%s", _fn, len(slots), (list(slots)[:3] if slots else []), _caller)
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
