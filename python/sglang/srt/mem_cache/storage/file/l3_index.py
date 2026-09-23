# SPDX-License-Identifier: Apache-2.0
"""#1459: the shared L3 STEM INDEX (arena.c ``l3idx_*``) -- one shm table for
all ranks answering "is this stem on the disk store?" without a stat per page.

Boot weg2xsn214: group P's intake probe for a fresh 100k prompt stat'ed ~94k
page files to learn that none exists -- 3 s per request with the GPU idle.
The evictor is the ONE bookkeeper of the disk store (commit / evict / clear),
so it maintains this table; ``HiCacheFile._stat_stems`` asks it first and
stats only the stems the table names.  A stem missing here is NOT on disk.

The table lives beside the arena (``<arena dir>/l3idx.bin``): 2^23 entries x
16 B = 134 MB of shm, room for ~8M page files at 50 % load.
"""
from __future__ import annotations

import ctypes
import logging
import mmap
import os
import threading
import time
from typing import List, Optional, Sequence

from sglang.srt.mem_cache.storage.file.hicache_arena import _load_lib

logger = logging.getLogger(__name__)

DEFAULT_CAP = 1 << 23
_MAGIC = 0x4C334944585F5731


def _declare(lib: ctypes.CDLL) -> None:
    if getattr(lib, "_l3idx_declared", False):
        return
    P = ctypes.c_void_p
    lib.l3idx_layout.restype, lib.l3idx_layout.argtypes = ctypes.c_int64, [ctypes.c_int64]
    lib.l3idx_init.restype, lib.l3idx_init.argtypes = ctypes.c_int, [P, ctypes.c_int64]
    lib.l3idx_cap.restype, lib.l3idx_cap.argtypes = ctypes.c_int64, [P]
    lib.l3idx_count.restype, lib.l3idx_count.argtypes = ctypes.c_int64, [P]
    SS = ctypes.POINTER(ctypes.c_char_p)
    lib.l3idx_add_stems.restype, lib.l3idx_add_stems.argtypes = ctypes.c_int64, [P, ctypes.c_int64, SS]
    lib.l3idx_del_stems.restype, lib.l3idx_del_stems.argtypes = ctypes.c_int64, [P, ctypes.c_int64, SS]
    lib.l3idx_has_stems.restype, lib.l3idx_has_stems.argtypes = ctypes.c_int64, [
        P, ctypes.c_int64, SS, ctypes.POINTER(ctypes.c_int8)]
    lib.l3idx_clear.restype, lib.l3idx_clear.argtypes = None, [P]
    lib._l3idx_declared = True


class L3Index:
    """One mapping of the shared table.  ``open`` creates it when absent; a
    second opener waits for the creator's magic (bounded), then maps it."""

    def __init__(self, path: str, cap: int = DEFAULT_CAP):
        lib = _load_lib()
        if lib is None:
            raise RuntimeError("arena.c did not build; no L3 index")
        _declare(lib)
        self._lib, self.path, self.cap = lib, path, int(cap)
        nbytes = int(lib.l3idx_layout(self.cap))
        if nbytes < 0:
            raise ValueError(f"cap must be a power of two >= 1024: {cap}")
        self._lock = threading.Lock()
        created = False
        try:
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            created = True
        except FileExistsError:
            fd = os.open(path, os.O_RDWR)
        try:
            if created:
                os.ftruncate(fd, nbytes)
            else:
                t0 = time.monotonic()
                while os.fstat(fd).st_size < nbytes:
                    if time.monotonic() - t0 > 30.0:
                        raise RuntimeError(f"L3 index {path} never reached {nbytes} bytes")
                    time.sleep(0.01)
            self._mm = mmap.mmap(fd, nbytes, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        self._base = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(self._mm)))
        if created:
            if int(lib.l3idx_init(self._base, self.cap)) != 0:
                raise RuntimeError("l3idx_init refused")
        else:
            t0 = time.monotonic()
            while int(lib.l3idx_cap(self._base)) != self.cap:
                if time.monotonic() - t0 > 30.0:
                    raise RuntimeError(f"L3 index {path}: magic/cap never appeared (cap {lib.l3idx_cap(self._base)})")
                time.sleep(0.01)
        self.created = created
        self._full_logged = False

    @staticmethod
    def _carr(stems: Sequence[str]):
        n = len(stems)
        arr = (ctypes.c_char_p * n)(*[s.encode() for s in stems])
        return n, arr

    def add(self, stems: Sequence[str]) -> int:
        if not stems:
            return 0
        n, arr = self._carr(stems)
        with self._lock:
            r = int(self._lib.l3idx_add_stems(self._base, n, arr))
        if r == -2 and not self._full_logged:
            self._full_logged = True
            logger.error("#1459 L3-INDEX FULL at %d entries (cap %d): stems beyond this are stat'ed the old way",
                         self.count(), self.cap)
        return max(r, 0)

    def remove(self, stems: Sequence[str]) -> int:
        if not stems:
            return 0
        n, arr = self._carr(stems)
        with self._lock:
            return int(self._lib.l3idx_del_stems(self._base, n, arr))

    def has(self, stems: Sequence[str]) -> List[bool]:
        if not stems:
            return []
        n, arr = self._carr(stems)
        out = (ctypes.c_int8 * n)()
        self._lib.l3idx_has_stems(self._base, n, arr, out)
        return [bool(v) for v in out]

    def clear(self) -> None:
        with self._lock:
            self._lib.l3idx_clear(self._base)

    def count(self) -> int:
        return int(self._lib.l3idx_count(self._base))

    def close(self) -> None:
        try:
            self._mm.close()
        except Exception:  # noqa: BLE001
            pass


def open_index(path: Optional[str], cap: int = DEFAULT_CAP) -> Optional[L3Index]:
    """``L3Index`` or None (no path / build failed / env off) -- never raises."""
    if not path or os.environ.get("SGLANG_HICACHE_L3_INDEX", "1") == "0":
        return None
    try:
        return L3Index(path, cap)
    except Exception as exc:  # noqa: BLE001
        logger.warning("#1459 L3-INDEX n/a (%s: %s) -- stat per page as before", type(exc).__name__, exc)
        return None
