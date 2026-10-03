"""WAKE-PARALLEL (28.09.): the wake's store reads of the parked requests run
on several aux threads instead of one -- same resume set and order on every
D rank.

THE SPECIMEN. NF rc12z22 D (boot dkrnfh91dprsavisnoadoptstbar1dauer09281447,
7d507357b9), TP0 14:56:23: ``#248 WAKE-READ issued=5 ['weg2-0-5',
'weg2-4-11', 'weg2-10-16', 'weg2-11-18', 'weg2-11-17']``, ``#1471 SETTLE 5
held``, then one read after the other on the single prefetch aux thread:

    weg2-0-5    read_ms=681  queue_ms=364
    weg2-4-11   read_ms=854  queue_ms=1028
    weg2-10-16  read_ms=557  queue_ms=1859
    weg2-11-18  read_ms=350  queue_ms=2402
    weg2-11-17  read_ms=378  queue_ms=2750

-- 2.82 s of reads in a row, the last one released 4.1 s after the wake.

Guarded here, hermetic (a stand-in ``_page_transfer`` sleeps the measured
read time x ``SCALE``; the real ``prefetch_io_aux_func`` and worker start):
* serial (``SGLANG_HICACHE_PREFETCH_IO_WORKERS=1``): the five reads end at
  the SUM of their read times -- the metal's shape;
* parallel (default 4): they end at about the LONGEST read (854 ms on the
  metal, < 1 s) -- the resume target;
* three D ranks (TP3) whose reads finish in DIFFERENT orders release the same
  requests in the same order in every settle tick (the real
  ``_weg2_post_wake_settle_tick`` with its group MIN over the three ranks);
* the census and the stop see every operation the aux threads hold.
"""

import os
import threading
import time
import types
import unittest
from queue import Queue
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import cache_controller as cc  # noqa: E402
from sglang.srt.managers import scheduler as sched_mod  # noqa: E402

S = sched_mod.Scheduler

#: TP0 14:56:23-27 (see the module docstring), in wake-read order
WAKE = [("weg2-0-5", 681), ("weg2-4-11", 854), ("weg2-10-16", 557),
        ("weg2-11-18", 350), ("weg2-11-17", 378)]
SCALE = 0.25


def _op(rid):
    return types.SimpleNamespace(request_id=rid, host_indices=list(range(4)), completed_tokens=4,
                                 done=threading.Event(), t_start=None, t_end=None,
                                 mark_terminate=lambda: None, is_terminated=lambda: False)


def _controller(read_ms, workers):
    """A HiCacheController with only what the aux loop touches; its
    _page_transfer sleeps the op's read time (x the rank's jitter)."""
    c = object.__new__(cc.HiCacheController)
    c.storage_stop_event = threading.Event()
    c.prefetch_buffer = Queue()
    c.prefetch_queue = Queue()
    c._prefetch_io_current = None
    c._prefetch_io_inflight = {}
    c._prefetch_current = None
    c._prefetch_io_drained_after_stop = 0
    c._prefetch_drained_after_stop = 0
    c.mem_pool_host = None
    c.append_host_mem_release = lambda *a, **k: None
    c.peak = 0

    def transfer(op):
        op.t_start = time.monotonic()
        c.peak = max(c.peak, c.storage_loads_pending())
        time.sleep(read_ms[op.request_id] / 1000.0 * SCALE)
        op.t_end = time.monotonic()
        op.done.set()

    c._page_transfer = transfer
    with mock.patch.dict(os.environ, {cc.PREFETCH_IO_WORKERS_ENV: str(workers)}):
        c._start_prefetch_io_workers()
    return c


def _stop(c):
    c.storage_stop_event.set()
    for _ in c.prefetch_io_aux_threads:
        c.prefetch_buffer.put(None)
    for t in c.prefetch_io_aux_threads:
        t.join(timeout=3)


def _wake(workers, jitter=None):
    """Issue the five wake reads in hold order; returns (ops, end - start)."""
    read_ms = {rid: ms * (jitter or {}).get(rid, 1.0) for rid, ms in WAKE}
    c = _controller(read_ms, workers)
    t0 = time.monotonic()
    ops = [_op(rid) for rid, _ in WAKE]
    for op in ops:
        c.prefetch_buffer.put(op)
    for op in ops:
        assert op.done.wait(10)
    dt = time.monotonic() - t0
    _stop(c)
    return c, ops, dt


class TheWakeReadsRunSideBySide(unittest.TestCase):
    def test_serial_is_the_metal_shape(self):
        _c, ops, dt = _wake(workers=1)
        total = sum(ms for _, ms in WAKE) / 1000.0 * SCALE
        self.assertGreaterEqual(dt, total * 0.95)
        starts = [op.t_start for op in ops]
        self.assertEqual(starts, sorted(starts))       # one after the other, in hold order

    def test_parallel_ends_at_the_longest_read(self):
        c, _ops, dt = _wake(workers=4)
        longest = max(ms for _, ms in WAKE) / 1000.0 * SCALE
        total = sum(ms for _, ms in WAKE) / 1000.0 * SCALE
        # metal units: ~854 ms (< 1 s) instead of 2.82 s
        self.assertLess(dt, longest + 0.5 * (total - longest))
        self.assertLess(dt / SCALE, 1.3)
        self.assertGreaterEqual(c.peak, 4)             # the census saw them in flight together
        self.assertEqual(c._prefetch_io_inflight, {})
        self.assertIsNone(c._prefetch_io_current)

    def test_default_workers_and_env(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(cc.PREFETCH_IO_WORKERS_ENV, None)
            self.assertEqual(cc.prefetch_io_workers(), 4)
        with mock.patch.dict(os.environ, {cc.PREFETCH_IO_WORKERS_ENV: "1"}):
            self.assertEqual(cc.prefetch_io_workers(), 1)
        with mock.patch.dict(os.environ, {cc.PREFETCH_IO_WORKERS_ENV: "x"}):
            self.assertEqual(cc.prefetch_io_workers(), 4)

    def test_stop_terminates_every_held_operation(self):
        c = object.__new__(cc.HiCacheController)
        c.prefetch_queue, c.prefetch_buffer = Queue(), Queue()
        c._prefetch_current = None
        marked = []
        ops = [types.SimpleNamespace(mark_terminate=lambda i=i: marked.append(i),
                                     is_terminated=lambda: False) for i in range(3)]
        c._prefetch_io_inflight = {id(o): o for o in ops}
        c._prefetch_io_current = ops[0]
        self.assertEqual(c._terminate_inflight_prefetch(), 3)
        self.assertEqual(sorted(marked), [0, 1, 2])
        self.assertEqual(c.storage_loads_pending(), 3)


class _Group:
    """The group MIN of three D ranks: every rank's call k meets the others'."""

    def __init__(self, n):
        self.n = n
        self.bar = threading.Barrier(n)
        self.slots = {}
        self.lock = threading.Lock()

    def min_flags(self, rank, k, flags):
        with self.lock:
            self.slots.setdefault(k, {})[rank] = [int(bool(f)) for f in flags]
        self.bar.wait(5)
        out = [min(v) for v in zip(*self.slots[k].values())] if flags else []
        self.bar.wait(5)
        return out


def _rank(rank, group, ops, releases, stop):
    h = types.SimpleNamespace()
    h.waiting_queue = []
    h.weg2_dormant = False
    h.tree_cache = types.SimpleNamespace(cache_controller=types.SimpleNamespace(
        weg2_hold_rids={rid for rid, _ in WAKE}))
    h.WEG2_POST_WAKE_SETTLE_S = S.WEG2_POST_WAKE_SETTLE_S
    h.ps = types.SimpleNamespace(tp_size=3)
    h._weg2_post_wake_settle_tick = types.MethodType(S._weg2_post_wake_settle_tick, h)
    calls = [0]

    def gmin(flags):
        calls[0] += 1
        return group.min_flags(rank, calls[0], flags)

    h._weg2_group_min_flags = gmin
    by_rid = {op.request_id: op for op in ops}
    h._weg2_refetch_one = lambda req, now, allow_reissue=True: (
        "complete" if by_rid[req.rid].done.is_set() else "reading")
    now = time.monotonic()
    h.weg2_post_wake_settle = [types.SimpleNamespace(rid=rid, _1471_since=now) for rid, _ in WAKE]
    tick = 0
    while not stop.is_set():
        before = len(h.waiting_queue)
        h._weg2_post_wake_settle_tick()
        releases[rank].append((tick, [r.rid for r in h.waiting_queue[before:]]))
        tick += 1
        if not h.weg2_post_wake_settle:
            break
        time.sleep(0.01)
    return h


class ThreeRanksResumeTheSameSetInTheSameOrder(unittest.TestCase):
    def test_different_completion_orders_same_releases(self):
        # each rank's reads end in a different order (per-rank jitter)
        jitters = [
            {},
            {"weg2-0-5": 1.6, "weg2-11-18": 0.4},
            {"weg2-4-11": 0.3, "weg2-11-17": 2.0},
        ]
        rank_ops, ctrls = [], []
        for j in jitters:
            read_ms = {rid: ms * j.get(rid, 1.0) for rid, ms in WAKE}
            c = _controller(read_ms, workers=4)
            ops = [_op(rid) for rid, _ in WAKE]
            for op in ops:
                c.prefetch_buffer.put(op)
            rank_ops.append(ops)
            ctrls.append(c)
        group = _Group(3)
        releases = {0: [], 1: [], 2: []}
        stop = threading.Event()
        ths = [threading.Thread(target=_rank, args=(k, group, rank_ops[k], releases, stop))
               for k in range(3)]
        t0 = time.monotonic()
        for t in ths:
            t.start()
        for t in ths:
            t.join(15)
        stop.set()
        dt = time.monotonic() - t0
        for c in ctrls:
            _stop(c)
        # the completion orders really differ between the ranks
        orders = [[op.request_id for op in sorted(ops, key=lambda o: o.t_end)] for ops in rank_ops]
        self.assertGreater(len({tuple(o) for o in orders}), 1)
        # ... and every tick released the same requests in the same order
        self.assertEqual(releases[0], releases[1])
        self.assertEqual(releases[0], releases[2])
        flat = [rid for _t, rids in releases[0] for rid in rids]
        self.assertEqual(sorted(flat), sorted(rid for rid, _ in WAKE))
        for _t, rids in releases[0]:   # within a tick: the hold order
            self.assertEqual(rids, [r for r, _ in WAKE if r in rids])
        # all resumed at about the slowest rank's longest read
        self.assertLess(dt / SCALE, 2.2)


if __name__ == "__main__":
    unittest.main()
