"""L3-FAST (28.09.): the L3 -> L2 fill reads a batch in one claim, parallel
page reads and one completion -- the same pages, the same prefix, faster.

27B boots of 28.09. (releasedraft 13:40, parkdraft 14:21): P read its told
prefix at 1.4-3.2k tokens/s (``#1416e PACED-TOLD told=16383 own_read=15.40s``,
``#1433 L3->L2 fill: 1 of 1 pages`` per page, ``#1436 ARENA-GET ref_ms~745``
per 1024 pages). Measured with this repo's pageio on the same store (445k
32-KiB pages, NVMe): one page per call 10.2k pages/s, one call 11.7k/s, 8
threads 62k/s, 16 threads 106k/s. The disk was never the limit -- the per-page
Python loop around it was. User: "das lesen muss schneller gehen. lesen kostet
fast keine energie. neurechnen hingegen sehr viel".

Guarded here (danger direction: a batch that yields a DIFFERENT prefix would
split the ranks' told):
* ``arena_fill_from_disk`` over n stems returns exactly what n single-stem
  calls return (slot per stem, None for a page not on disk / not readable),
  with ONE claim and ONE completion call, and the bytes land in the slots;
* the parallel read (16 threads) equals the serial read, status and bytes;
  ``SGLANG_HICACHE_L3_READ_THREADS=1`` is the serial single call;
* ``_arena_page_get`` returns the same page count as before for a mixed batch
  (complete lead, L3 pages, a page the L3 lacks) -- the prefix ends at the
  first page nothing can give -- and asks the L3 once per batch;
* a microbench (same sample, before/after) reads at least 2x faster here
  (hermetic); on the real store see the commit message.
Hermetic: the real pageio (gcc) on temp files, a recording arena double.
"""
from __future__ import annotations

import os
import shutil
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from sglang.srt.mem_cache import hicache_storage as hs  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

PAGE = 4096


class _Arena:
    """A slot table over one numpy buffer; records the batch calls."""

    def __init__(self, n_slots=4096):
        self.buf = np.zeros((n_slots, PAGE), dtype=np.uint8)
        self.state = {}          # stem -> slot (COMPLETE)
        self.free = list(range(n_slots))
        self.calls = {"claim": 0, "complete": 0, "free": 0}
        self.claimed = {}

    def claim_slots(self, stems, totals):
        self.calls["claim"] += 1
        out = []
        for st in stems:
            if st in self.state:
                out.append((self.state[st], 2, 0))
            elif not self.free:
                out.append((-1, 4, 0))
            else:
                s = self.free.pop(0)
                self.claimed[s] = st
                out.append((s, 0, 1))
        return out

    def slot_ptr(self, slot):
        return self.buf[slot].ctypes.data

    def complete_slots(self, slots, gens, extents):
        self.calls["complete"] += 1
        for s in slots:
            self.state[self.claimed.pop(s)] = s
        return [1] * len(slots)

    def free_slots(self, slots):
        self.calls["free"] += 1
        for s in slots:
            self.claimed.pop(s, None)
            self.free.append(s)


def _store(tmp_path, n, missing=()):
    """n pages on disk (page i filled with byte i%251), except ``missing``."""
    be = object.__new__(hs.HiCacheFile)
    root = tmp_path / "l3"
    root.mkdir()
    paths = {}
    for i in range(n):
        st = "s%05d" % i
        p = root / (st + ".bin")
        if i not in missing:
            p.write_bytes(bytes([i % 251]) * PAGE)
        paths[st] = str(p)
    be._stat_stems = lambda stems: {st: PAGE for st in stems if os.path.exists(paths[st])}
    be._existing_path = lambda st: paths[st]
    be._arena_evict_to_disk = lambda arena, want: 0
    return be, ["s%05d" % i for i in range(n)]


def test_one_batch_fill_equals_n_single_fills(tmp_path):
    be, stems = _store(tmp_path, 300, missing={7, 150})
    a1 = _Arena()
    single = [hs.HiCacheFile.arena_fill_from_disk(be, a1, [st], PAGE)[0] for st in stems]
    a2 = _Arena()
    batch = hs.HiCacheFile.arena_fill_from_disk(be, a2, stems, PAGE)
    assert [s is None for s in single] == [s is None for s in batch]
    assert batch[7] is None and batch[150] is None and batch[8] is not None
    assert a2.calls["claim"] == 1 and a2.calls["complete"] == 1
    for i, s in enumerate(batch):
        if s is not None:
            assert int(a2.buf[s][0]) == i % 251 and int(a2.buf[s][-1]) == i % 251


def test_the_parallel_read_equals_the_serial_read(tmp_path, monkeypatch):
    be, stems = _store(tmp_path, 500, missing={3})
    from sglang.srt.mem_cache.storage.file.pageio import load

    pio = load()
    paths = [be._existing_path(st) for st in stems]
    b1 = np.zeros((500, PAGE), dtype=np.uint8)
    b2 = np.zeros((500, PAGE), dtype=np.uint8)
    rc1, k1 = hs.l3_read_pages_parallel(pio, paths, PAGE, [b1[i].ctypes.data for i in range(500)], threads=1)
    rc2, k2 = hs.l3_read_pages_parallel(pio, paths, PAGE, [b2[i].ctypes.data for i in range(500)], threads=16)
    assert (k1, k2) == (1, 16)
    assert rc1 == rc2 and rc1[3] != 0 and rc1.count(0) == 499
    assert np.array_equal(b1, b2)
    monkeypatch.setenv(hs.L3_READ_THREADS_ENV, "1")
    assert hs.l3_read_threads() == 1
    monkeypatch.delenv(hs.L3_READ_THREADS_ENV)
    assert hs.l3_read_threads() == hs.L3_READ_THREADS_DEFAULT == 16


class _PoolArena:
    """The pool's arena as _arena_page_get sees it: find/ref over fixed state."""

    def __init__(self, found, evicted=()):
        self.found = found
        self.evicted = set(evicted)
        self.refs = []

    def find_slots_np(self, stems):
        return (np.asarray([s for s, _ in self.found], dtype=np.int64),
                np.asarray([t for _, t in self.found], dtype=np.int8))

    def ref_slots_np(self, a, delta):
        a = [int(x) for x in a]
        self.refs += [(x, delta) for x in a]
        return len(a)

    def ref_slots(self, slots, delta):
        s = int(slots[0])
        if s in self.evicted:
            return 0
        self.refs.append((s, delta))
        return 1


def _controller(found, fill_ok, evicted=()):
    from sglang.srt.managers import cache_controller as cc

    arena = _PoolArena(found, evicted)
    fills = []

    def fill(ar, stems, nbytes):
        fills.append(list(stems))
        return [1000 + int(st[1:]) if int(st[1:]) in fill_ok else None for st in stems]

    backend = types.SimpleNamespace(arena_fill_from_disk=fill)
    pool = types.SimpleNamespace(arena_read=True, arena=arena, _page_bytes=PAGE,
                                 ensure_bound=lambda be, role: True,
                                 resolve_rows=lambda hi, slots: None)
    ctl = types.SimpleNamespace(mem_pool_host=pool, storage_backend=backend, page_size=1)
    op = types.SimpleNamespace(probe_pins=None, completed_tokens=0, n=0)
    op.increment = lambda k: setattr(op, "n", op.n + k)
    orig = cc.weg2_suffixed_stems
    cc.weg2_suffixed_stems = lambda be, hv: ["s%d" % i for i in range(len(hv))]
    try:
        got = cc.HiCacheController._arena_page_get(ctl, op, list(range(len(found))), None)
    finally:
        cc.weg2_suffixed_stems = orig
    return got, fills, arena


def test_arena_page_get_returns_the_same_prefix_and_asks_the_l3_once():
    # lead 0..2 complete, 3..5 in L3, 6 complete, 7 in L3 but NOT on disk, 8 in L3
    found = [(10, 2), (11, 2), (12, 2), (-1, 0), (-1, 0), (-1, 0), (16, 2), (-1, 0), (-1, 0)]
    got, fills, _ = _controller(found, fill_ok={3, 4, 5, 8})
    assert got == 7                        # the prefix ends at page 7, as before
    assert fills == [["s3", "s4", "s5", "s7", "s8"]]   # ONE fill for the batch


def test_a_failed_reference_still_ends_the_prefix_there():
    found = [(10, 2), (-1, 0), (-1, 0), (13, 2)]
    got, _, _ = _controller(found, fill_ok={1, 2}, evicted={1002})
    assert got == 2                        # lead 1 + page 1; page 2's ref fails


def test_microbench_before_after_on_the_same_sample(tmp_path):
    """Same 2000 pages: the per-page path (one fill per page, as before) vs the
    batch (one fill, 16 threads). Hermetic (page cache, a trivial arena), so
    the bar is 2x; the real-store numbers are in the commit message."""
    be, stems = _store(tmp_path, 2000)
    a1 = _Arena(n_slots=2048)
    t = time.perf_counter()
    for st in stems:
        hs.HiCacheFile.arena_fill_from_disk(be, a1, [st], PAGE)
    before = time.perf_counter() - t
    a2 = _Arena(n_slots=2048)
    t = time.perf_counter()
    out = hs.HiCacheFile.arena_fill_from_disk(be, a2, stems, PAGE)
    after = time.perf_counter() - t
    assert all(s is not None for s in out)
    print("L3-FAST microbench: before %.0f pages/s, after %.0f pages/s (x%.1f)"
          % (2000 / before, 2000 / after, before / after))
    assert before / after >= 2.0
