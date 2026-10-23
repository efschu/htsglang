"""#1402: ctypes loader for the batched page I/O helper (``pageio.c``).

The helper is compiled on first use with the system ``gcc`` into a shared
object under the temp directory, keyed by the source's hash, so a changed
source never runs stale code and the ranks of one boot share one build (each
compiles to a private name and ``os.replace``s it into place -- the last
writer wins with identical bytes). Anything that fails -- no compiler, no
writable temp dir, a load error -- leaves ``load()`` returning None and the
callers on their per-syscall Python path, with the reason logged once.

``FLLIPER_HICACHE_PAGEIO=0`` disables the helper outright.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import subprocess
import tempfile
import threading
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pageio.c")
_lock = threading.Lock()
_loaded: Optional["PageIO"] = None
_failed = False


class PageIO:
    """Thin typed wrapper over the two C entry points."""

    def __init__(self, lib: ctypes.CDLL):
        self._lib = lib
        lib.hicache_stat_sizes.restype = ctypes.c_int64
        lib.hicache_stat_sizes.argtypes = [
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_int64),
        ]
        lib.hicache_read_pages.restype = ctypes.c_int64
        lib.hicache_read_pages.argtypes = [
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int8),
        ]

        # STOREREAD-1023: the whole batch on C threads in one call (see pageio.c)
        self._has_read_mt = hasattr(lib, "hicache_read_pages_mt")
        if self._has_read_mt:
            lib.hicache_read_pages_mt.restype = ctypes.c_int64
            lib.hicache_read_pages_mt.argtypes = [
                ctypes.c_int64,
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_int8),
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_int64),
            ]

        # S36 (18.10.): KV-DURCHLAUF reads out of held fds, O_DIRECT writes
        self._has_fds = hasattr(lib, "hicache_read_fds")
        if self._has_fds:
            lib.hicache_read_fds.restype = ctypes.c_int64
            lib.hicache_read_fds.argtypes = [
                ctypes.c_int64,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int8),
                ctypes.POINTER(ctypes.c_int8),
            ]
        self._has_direct_write = hasattr(lib, "hicache_write_pages_direct")
        if self._has_direct_write:
            lib.hicache_write_pages_direct.restype = ctypes.c_int64
            lib.hicache_write_pages_direct.argtypes = [
                ctypes.c_int64,
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_int8),
                ctypes.POINTER(ctypes.c_int32),
            ]

        lib.hicache_write_pages.restype = ctypes.c_int64
        lib.hicache_write_pages.argtypes = [
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int8),
        ]

    def write_pages(
        self,
        finals: Sequence[str],
        totals: Sequence[int],
        extents: Sequence[Sequence[tuple[int, int]]],
        payload_ptrs: Sequence[int],
        fsync: bool,
        direct: bool = False,
        direct_counts: Optional[list] = None,
    ) -> list[int]:
        """the canonical extent writer of canonical_page_store for a whole batch, in C.

        S36 (18.10., Soll 2): ``direct=True`` writes every 4-KiB-aligned extent
        through an O_DIRECT descriptor of the locked part file (XFS: the L2 ->
        L3 copy bypasses the page cache); anything unaligned or a filesystem
        without O_DIRECT falls back to the buffered descriptor, page by page,
        with the same statuses. ``direct_counts`` (a list) receives, per page,
        how many extents went direct.

        Status per page: 0 completed and published, 1 partial (marker
        written), 2 already complete, 3 shape refused, 4 io/lock error.
        ``payload_ptrs[i]`` holds the page's extents back to back.
        """
        from flliper.srt.mem_cache.canonical_page_store import marker_path, part_path

        n = len(finals)
        if n == 0:
            return []
        c_final = (ctypes.c_char_p * n)(*[os.fsencode(p) for p in finals])
        c_part = (ctypes.c_char_p * n)(*[os.fsencode(part_path(p)) for p in finals])
        c_mark = (ctypes.c_char_p * n)(*[os.fsencode(marker_path(p)) for p in finals])
        c_total = (ctypes.c_int64 * n)(*[int(t) for t in totals])
        n_ext = (ctypes.c_int64 * n)(*[len(e) for e in extents])
        flat_off = [int(o) for e in extents for o, _ in e]
        flat_len = [int(l) for e in extents for _, l in e]
        m = max(1, len(flat_off))
        c_off = (ctypes.c_int64 * m)(*flat_off)
        c_len = (ctypes.c_int64 * m)(*flat_len)
        c_pay = (ctypes.c_void_p * n)(*[int(p) for p in payload_ptrs])
        status = (ctypes.c_int8 * n)()
        if direct and self._has_direct_write:
            dcount = (ctypes.c_int32 * n)()
            self._lib.hicache_write_pages_direct(
                n, c_final, c_part, c_mark, c_total, n_ext, c_off, c_len, c_pay,
                1 if fsync else 0, status, dcount,
            )
            if direct_counts is not None:
                direct_counts.extend(int(x) for x in dcount)
            return list(status)
        self._lib.hicache_write_pages(
            n, c_final, c_part, c_mark, c_total, n_ext, c_off, c_len, c_pay,
            1 if fsync else 0, status,
        )
        if direct_counts is not None:
            direct_counts.extend([0] * n)
        return list(status)

    def read_fds(
        self,
        fds: Sequence[int],
        offs: Sequence[int],
        lens: Sequence[int],
        out_ptrs: Sequence[int],
    ) -> tuple[list[int], list[int]]:
        """S36 KV-DURCHLAUF: ``(status, direct)`` per entry -- ``lens[i]`` bytes
        at ``offs[i]`` of the HELD descriptor ``fds[i]`` into ``out_ptrs[i]``.
        Status 0 ok, 3 short, 4 error; direct 1 = the read went O_DIRECT (a
        refused aligned read is repeated buffered on the same inode)."""
        n = len(fds)
        if n == 0:
            return [], []
        if not self._has_fds:
            raise RuntimeError("pageio helper without hicache_read_fds (stale build)")
        c_fd = (ctypes.c_int32 * n)(*[int(f) for f in fds])
        c_off = (ctypes.c_int64 * n)(*[int(o) for o in offs])
        c_len = (ctypes.c_int64 * n)(*[int(x) for x in lens])
        c_out = (ctypes.c_void_p * n)(*[int(p) for p in out_ptrs])
        status = (ctypes.c_int8 * n)()
        direct = (ctypes.c_int8 * n)()
        self._lib.hicache_read_fds(n, c_fd, c_off, c_len, c_out, status, direct)
        return list(status), list(direct)

    def stat_sizes(self, paths: Sequence[str]) -> list[int]:
        """``st_size`` per path, -1 where the path cannot be stat'ed."""
        n = len(paths)
        if n == 0:
            return []
        c_paths = (ctypes.c_char_p * n)(*[os.fsencode(p) for p in paths])
        sizes = (ctypes.c_int64 * n)()
        self._lib.hicache_stat_sizes(n, c_paths, sizes)
        return list(sizes)

    @property
    def has_read_mt(self) -> bool:
        return bool(self._has_read_mt)

    def read_pages_mt(
        self,
        paths: Sequence[str],
        expect_total: Sequence[int],
        extents: Sequence[Sequence[tuple[int, int]]],
        out_ptrs: Sequence[int],
        touch: bool,
        threads: int,
        progress=None,
        poll_s: float = 0.1,
    ) -> list[int]:
        """STOREREAD-1023: ``read_pages`` for the whole batch on ``threads`` C
        threads in ONE ctypes call (one GIL release, one re-acquire).

        ``progress`` (optional, a no-arg callable) is the S23 witness: while
        the C call runs on a helper thread, the caller calls it at most every
        ``poll_s`` and only when the C page counter moved since the last call
        -- a read that stops moving stops reporting, exactly like the pool
        path's per-sub-call report. Status per page as ``read_pages``.
        """
        n = len(paths)
        if n == 0:
            return []
        if not self._has_read_mt:
            raise RuntimeError("pageio helper without hicache_read_pages_mt (stale build)")
        c_paths = (ctypes.c_char_p * n)(*[os.fsencode(p) for p in paths])
        c_total = (ctypes.c_int64 * n)(*[int(t) for t in expect_total])
        n_ext = (ctypes.c_int64 * n)(*[len(e) for e in extents])
        flat_off = [int(o) for e in extents for o, _ in e]
        flat_len = [int(l) for e in extents for _, l in e]
        m = max(1, len(flat_off))
        c_off = (ctypes.c_int64 * m)(*flat_off)
        c_len = (ctypes.c_int64 * m)(*flat_len)
        c_out = (ctypes.c_void_p * n)(*[int(p) for p in out_ptrs])
        status = (ctypes.c_int8 * n)()
        done = ctypes.c_int64(0)
        k = max(1, min(64, int(threads)))

        def _call():
            self._lib.hicache_read_pages_mt(
                n, c_paths, c_total, n_ext, c_off, c_len, c_out, 1 if touch else 0, status, k,
                ctypes.byref(done),
            )

        if progress is None:
            _call()
            return list(status)
        err: list = []

        def _run():
            try:
                _call()
            except BaseException as e:  # noqa: BLE001 - re-raised in the caller
                err.append(e)

        t = threading.Thread(target=_run, name="pageio-read-mt", daemon=True)
        t.start()
        last = -1
        while True:
            t.join(max(0.001, float(poll_s)))
            cur = int(done.value)
            if cur != last:
                last = cur
                try:
                    progress()
                except Exception:  # noqa: BLE001 - a witness never breaks a read
                    pass
            if not t.is_alive():
                break
        if err:
            raise err[0]
        return list(status)

    def read_pages(
        self,
        paths: Sequence[str],
        expect_total: Sequence[int],
        extents: Sequence[Sequence[tuple[int, int]]],
        out_ptrs: Sequence[int],
        touch: bool,
    ) -> list[int]:
        """Status per page: 0 ok, 1 missing, 2 size mismatch, 3 short, 4 error.

        ``out_ptrs[i]`` must point at at least ``sum(len for _, len in
        extents[i])`` writable bytes that stay alive for the call.
        """
        n = len(paths)
        if n == 0:
            return []
        c_paths = (ctypes.c_char_p * n)(*[os.fsencode(p) for p in paths])
        c_total = (ctypes.c_int64 * n)(*[int(t) for t in expect_total])
        n_ext = (ctypes.c_int64 * n)(*[len(e) for e in extents])
        flat_off = [int(o) for e in extents for o, _ in e]
        flat_len = [int(l) for e in extents for _, l in e]
        m = max(1, len(flat_off))
        c_off = (ctypes.c_int64 * m)(*flat_off)
        c_len = (ctypes.c_int64 * m)(*flat_len)
        c_out = (ctypes.c_void_p * n)(*[int(p) for p in out_ptrs])
        status = (ctypes.c_int8 * n)()
        self._lib.hicache_read_pages(
            n, c_paths, c_total, n_ext, c_off, c_len, c_out, 1 if touch else 0, status
        )
        return list(status)


def _build_path() -> str:
    with open(_SRC, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:16]
    return os.path.join(
        tempfile.gettempdir(), f"flliper_hicache_pageio_{digest}_{os.getuid()}.so"
    )


def load() -> Optional[PageIO]:
    """The helper, or None (reason logged once) when it cannot be had."""
    global _loaded, _failed
    if _loaded is not None:
        return _loaded
    if _failed:
        return None
    raw = os.environ.get("FLLIPER_HICACHE_PAGEIO", "1").strip().lower()
    if raw in ("0", "false", "no", "off"):
        _failed = True
        return None
    with _lock:
        if _loaded is not None:
            return _loaded
        if _failed:
            return None
        try:
            so = _build_path()
            if not os.path.exists(so):
                tmp = f"{so}.{os.getpid()}.tmp"
                subprocess.run(
                    ["gcc", "-O2", "-shared", "-fPIC", "-pthread", "-o", tmp, _SRC],
                    check=True,
                    capture_output=True,
                    timeout=120,
                )
                os.replace(tmp, so)
            _loaded = PageIO(ctypes.CDLL(so))
            logger.info("[#1402 pageio] batched page I/O helper loaded from %s", so)
            return _loaded
        except Exception as e:  # noqa: BLE001 - the fallback is the old path
            _failed = True
            logger.warning(
                "[#1402 pageio] helper unavailable (%s: %s); the file backend "
                "stays on the per-syscall Python read path.",
                type(e).__name__,
                str(e)[:200],
            )
            return None
