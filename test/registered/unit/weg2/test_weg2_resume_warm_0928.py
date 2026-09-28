"""RW (28.09.): the experts of the resumes are warmed AFTER the legs, while a
wake read settles -- never in a leg, never waited for.

NF rc12z22 D (dkrnfh91dprsavisnoadoptstbar1dauer09281447, 7d507357b9) TP0: every
eager extend after a wake paid "MoE expert pool layer N sync: eager pass wrote
23..58 rows" -- 1.6-2.0 s of compute for 8..229 tokens -- while an extend whose
experts were resident took 12-16 ms; the rearm prefetched nothing
(rows_from_store=0 prefetched=0 overlap=off). The resumes are the requests D
decoded before it slept, so the experts its LRU owned then are what their tail
extend routes to.

Pinned (hermetic, CPU pool tables, the copy kernel recorded):
* the snapshot reads the LRU owners, most recent first, capped;
* the warm fills only rows the rearm left free, from the store rows a miss
  copies from (the same (host_row, bank_row) pairs), and keeps the bijection;
* ticks warm LAYERS_PER_TICK layers; an eager forward that reaches a queued
  layer drops it (never waits); the settle over cancels the rest;
* the scheduler ticks only while a wake read settles AND the device is idle,
  and never while dormant;
* one RW WAKE-FIRST-TOKEN line per wake at the first decode forward with the
  three numbers NF asked for;
* switch off: no snapshot, nothing armed.
"""

import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.layers.moe import expert_offload as eo  # noqa: E402
from sglang.srt.layers.moe import expert_pool_device as epd  # noqa: E402
from sglang.srt.managers import scheduler as sched_mod  # noqa: E402

E, R, ROWS, STAGING = 16, 4, 12, 2        # residents 0..3, LRU rows 4..9, staging 10..11


def _tables():
    hot = {e: e for e in range(R)}
    host = [-1] * R + list(range(E - R))
    return epd.allocate_pool_tables("cpu", E, ROWS, R, STAGING, hot, host)


class _Cache:
    """A pool layer: real tables and the real lru_snapshot / warm_lru_local."""

    def __init__(self, lid):
        self.layer = types.SimpleNamespace(layer_id=lid)
        self._pool_ready = True
        self._pool_tables = _tables()
        self._pool_dsts = [torch.zeros(1)]
        self._pool_srcs = [torch.zeros(1)]
        self.copies = []

    lru_snapshot = eo.MoEExpertOffloadCache.lru_snapshot
    warm_lru_local = eo.MoEExpertOffloadCache.warm_lru_local


def _own(tables, rows_experts_use):
    for r, e, u in rows_experts_use:
        tables.row_key[r] = e
        tables.hot_phys[e] = r
        tables.row_use[r] = u


class _Model:
    def __init__(self, caches):
        self._mods = [types.SimpleNamespace(_expert_offload=c) for c in caches]

    def modules(self):
        return iter(self._mods)


def _record_copies(test):
    calls = []

    def fake_copy(srcs, dsts, src, dst, count):
        calls.append(list(zip(src.tolist(), dst.tolist())))

    p = mock.patch.object(epd, "copy_rows", fake_copy)
    p.start()
    test.addCleanup(p.stop)
    q = mock.patch.object(eo, "MoEExpertOffloadCache", _Cache)
    q.start()
    test.addCleanup(q.stop)
    return calls


class _Env(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        for k in (eo.RESUME_WARM_ENV, eo.RESUME_WARM_ROWS_ENV, eo.RESUME_WARM_LAYERS_ENV):
            os.environ.pop(k, None)
        eo._RESUME_WARM = None

    def tearDown(self):
        self._env.stop()
        eo._RESUME_WARM = None


class SnapshotAndWarm(_Env):
    def test_snapshot_is_the_lru_most_recent_first_and_capped(self):
        c = _Cache(0)
        _own(c._pool_tables, [(4, 9, 5), (5, 12, 9), (6, 7, 1)])
        self.assertEqual(c.lru_snapshot(), [12, 9, 7])
        self.assertEqual(c.lru_snapshot(2), [12, 9])

    def test_warm_fills_only_free_rows_from_the_store_rows(self):
        calls = _record_copies(self)
        c = _Cache(0)
        n = c.warm_lru_local([12, 9, 7, 2])      # 2 is resident: skipped
        self.assertEqual(n, 3)
        t = c._pool_tables
        (pairs,) = calls
        self.assertEqual([h for h, _r in pairs], [int(t.host_row[e]) for e in (12, 9, 7)])
        for e in (12, 9, 7):                      # bijection
            r = int(t.hot_phys[e])
            self.assertTrue(4 <= r < 10)
            self.assertEqual(int(t.row_key[r]), e)
        self.assertEqual(int(t.hot_phys[2]), 2)   # the resident row untouched

    def test_the_whole_cycle_sleep_wake_ticks(self):
        calls = _record_copies(self)
        caches = [_Cache(i) for i in range(5)]
        for c in caches:
            _own(c._pool_tables, [(4, 9, 3), (5, 12, 8)])
        rw = eo.resume_warm()
        self.assertEqual(rw.snapshot([_Model(caches)]), 10)
        for c in caches:                          # the rearm's reinit: LRU empty again
            c._pool_tables = _tables()
        self.assertEqual(rw.arm([_Model(caches)]), 5)
        os.environ[eo.RESUME_WARM_LAYERS_ENV] = "2"
        self.assertEqual(rw.tick(), 2)
        rw.eager_reached(caches[2])               # an eager forward got there first
        self.assertEqual(rw.pending(), 2)
        self.assertEqual(rw.tick(), 2)
        self.assertEqual(rw.pending(), 0)
        self.assertEqual(len(calls), 4)
        self.assertEqual(rw.warm_rows, 8)
        self.assertEqual(rw.skipped_eager, 1)
        self.assertEqual(int(caches[2]._pool_tables.hot_phys[12]), -1)   # the dropped layer: untouched

    def test_cancel_skips_the_rest(self):
        _record_copies(self)
        caches = [_Cache(i) for i in range(3)]
        for c in caches:
            _own(c._pool_tables, [(4, 9, 3)])
        rw = eo.resume_warm()
        rw.snapshot([_Model(caches)])
        rw.arm([_Model(caches)])
        self.assertEqual(rw.cancel("settle_done"), 3)
        self.assertEqual(rw.tick(), 0)
        self.assertIn("skipped_cancel=3 cancel=settle_done", rw.fields())

    def test_switch_off(self):
        _record_copies(self)
        os.environ[eo.RESUME_WARM_ENV] = "0"
        c = _Cache(0)
        _own(c._pool_tables, [(4, 9, 3)])
        rw = eo.resume_warm()
        self.assertEqual(rw.snapshot([_Model([c])]), 0)
        self.assertEqual(rw.arm([_Model([c])]), 0)

    def test_first_extend_census(self):
        rw = eo.resume_warm()
        for lid, rows in ((0, 23), (1, 0), (2, 17)):
            rw.note_eager_sync(lid, rows)
        rw.note_eager_sync(0, 99)                 # the next forward: closed
        rw.note_eager_sync(3, 99)
        self.assertIn("first_extend_eager_syncs=2 first_extend_rows=40", rw.fields())


class TheScheduler(_Env):
    def _sched(self, settle=True, running=False, dormant=False):
        rb = types.SimpleNamespace(is_empty=lambda: not running)
        return types.SimpleNamespace(
            weg2_post_wake_settle=[object()] if settle else [], running_batch=rb,
            last_batch=None, enable_overlap=True, result_queue=[], chunked_req=None,
            weg2_dormant=dormant)

    def _armed(self, n=3):
        _record_copies(self)
        caches = [_Cache(i) for i in range(n)]
        for c in caches:
            _own(c._pool_tables, [(4, 9, 3)])
        rw = eo.resume_warm()
        rw.snapshot([_Model(caches)])
        rw.arm([_Model(caches)])
        return rw

    def test_ticks_only_while_settling_and_idle(self):
        rw = self._armed()
        os.environ[eo.RESUME_WARM_LAYERS_ENV] = "1"
        self.assertEqual(sched_mod._weg2_resume_warm_tick(self._sched(running=True)), 0)
        self.assertEqual(sched_mod._weg2_resume_warm_tick(self._sched(dormant=True)), 0)
        self.assertEqual(rw.pending(), 3)
        self.assertEqual(sched_mod._weg2_resume_warm_tick(self._sched()), 1)
        self.assertEqual(rw.pending(), 2)
        self.assertEqual(sched_mod._weg2_resume_warm_tick(self._sched(settle=False)), 0)
        self.assertEqual(rw.pending(), 0)          # the settle is over: the rest is skipped
        self.assertIn("cancel=settle_done", rw.fields())

    def test_one_line_per_wake_at_the_first_decode(self):
        import time

        self._armed()
        s = types.SimpleNamespace(_rw_first_token_open=True, _weg2_resume_t0=time.perf_counter() - 1.0,
                                  _weg2_last_wake_t=time.perf_counter() - 0.25)
        ext = types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_decode=lambda: False))
        dec = types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_decode=lambda: True))
        with self.assertLogs(sched_mod.logger.name, level="INFO") as cm:
            self.assertFalse(sched_mod._weg2_resume_first_token_note(s, ext))
            self.assertTrue(sched_mod._weg2_resume_first_token_note(s, dec))
            self.assertFalse(sched_mod._weg2_resume_first_token_note(s, dec))
        lines = [m for m in cm.output if "RW WAKE-FIRST-TOKEN" in m]
        self.assertEqual(len(lines), 1)
        for f in ("wake_to_first_decode_ms=", "leg_ms=7", "first_extend_eager_syncs="):
            self.assertIn(f, lines[0])


class TheWiring(unittest.TestCase):
    def test_the_eager_path_never_waits_and_the_leg_copies_nothing(self):
        import inspect

        eager = inspect.getsource(eo.MoEExpertOffloadCache.run_eager_pool)
        self.assertIn("resume_warm().eager_reached(self)", eager)
        from sglang.srt.managers.scheduler_components import weight_updater as wu

        rel = inspect.getsource(wu.SchedulerWeightUpdaterManager.release_memory_occupation)
        self.assertIn("_rw().snapshot([_m])", rel)
        self.assertNotIn("warm_lru_local", rel)
        res = inspect.getsource(wu.SchedulerWeightUpdaterManager.resume_memory_occupation)
        self.assertIn("_rw().arm(_early)", res)
        self.assertNotIn("_rw().tick", res)          # nothing is copied in the leg


if __name__ == "__main__":
    unittest.main()
