"""27B INT8 metal 21.10. 01:26Z (rc29b, c77da494f9): P read L3 at up to 8.7 ms per one-token page (PP2: 4392 pages in
34.6 s, 14042 pages in 91.5 s; a P phase > 6 min, 11 clients gone). Two roots, both fixed:

1. the S23 progress witness touched ALL claims of the batch after every 16 pages per worker and rebuilt two ctypes
   arrays of the whole batch for each touch under the GIL -- quadratic in the page count, and 27B's KV pages are one
   token (page_size=1), so a fill is thousands of pages. Now the arrays are built once per read and the clock moves at
   most every L3_TOUCH_MIN_S (first and last touch always);
2. the S36 write-behind wrote every aligned page O_DIRECT, which leaves (on XFS: drops) the page cache -- every later
   read of a 32 KiB KV page went to the NVMe. O_DIRECT is now kept for pages of >= 1 MiB (NF's 64-token pages, the mamba
   anchors); small pages go buffered as before the merge.
"""
import os
import threading
import time
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-pdflip-unit")

from flliper.srt.mem_cache import hicache_storage as hs  # noqa: E402
from flliper.srt.mem_cache.hicache_storage import HiCacheFile  # noqa: E402


class _Lib:
    def __init__(self):
        self.calls = 0
        self.lock = threading.Lock()

    def arena_touch_progress(self, base, n, c_sl, c_g):
        with self.lock:
            self.calls += 1
        return n


class _Arena:
    def __init__(self):
        self._lib = _Lib()
        self._base = 1
        self.py_touches = 0

    def touch_progress(self, slots, gens):   # the per-call array rebuild: must not be used when the lib is there
        self.py_touches += 1
        return len(slots)


class _Pio:
    def __init__(self, sleep=0.0):
        self.sleep = sleep

    def read_pages(self, paths, sizes, exts, ptrs, *a):
        if self.sleep:
            time.sleep(self.sleep)
        return [0] * len(paths)


def _read(n, sleep=0.0, threads=16):
    be = object.__new__(HiCacheFile)
    todo = [(i, i, 1, "s%d" % i) for i in range(n)]
    arena = _Arena()
    with mock.patch.object(hs, "l3_read_threads", lambda: threads):
        t0 = time.monotonic()
        rc, k = be._l3_read_with_progress(arena, _Pio(sleep), todo, ["p"] * n, 32768, [0] * n)
        dt = time.monotonic() - t0
    assert all(r == 0 for r in rc)
    return arena, dt


def test_a_14042_page_fill_touches_a_bounded_number_of_times_with_one_array_build():
    arena, dt = _read(14042)
    assert arena.py_touches == 0                     # no per-touch rebuild of the batch arrays
    assert 2 <= arena._lib.calls <= 2 + int(dt / hs.L3_TOUCH_MIN_S) + 16, (arena._lib.calls, dt)   # not 14042/16 x 16


def test_a_slow_reader_still_touches_inside_the_join_idle_bound():
    # 0.15 s per sub-call: the witness moves at least every L3_TOUCH_MIN_S while the read lasts (S23 danger direction)
    arena, dt = _read(64, sleep=0.15, threads=1)
    assert arena._lib.calls >= int(dt / 0.5), (arena._lib.calls, dt)
    assert hs.L3_TOUCH_MIN_S < 1.0                   # well inside the 1 s JOIN-IDLE bound


def test_o_direct_only_for_big_pages():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("FLLIPER_PDFLIP_L3_WRITE_BEHIND_DIRECT_MIN_KIB", None)
        lim = HiCacheFile._l3wb_direct_min_bytes()
    assert lim == 1 << 20
    assert 32768 < lim                               # 27B KV page (one token): buffered, stays in the page cache
    assert 78446592 >= lim                           # 27B mamba anchor: O_DIRECT as NF
    src = open(hs.__file__, encoding="utf-8").read()
    assert "direct=(direct and int(total) >= self._l3wb_direct_min_bytes())," in src
