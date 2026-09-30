# SPDX-License-Identifier: Apache-2.0
"""RW-FINISH (#287, 30.09., NF y4k): the resume warm runs to the end instead of
being cancelled at the first decode.

DER BEFUND (y4k dff1a7fed4, D TP0): ``RW WAKE-FIRST-TOKEN ... snap_layers=48
... warm_layers=8 ... skipped_cancel=40 cancel=first_decode`` in allen 14 Wakes;
das erste 64er-Fenster danach: ``MOE-POOL-DEMAND max_nonres_per_step=57
median_layer_max=29`` (danach 24/18, 22/17), ``DECODE-HOST-SPLIT`` wall 38,4 ms
(gpu_verify 35,1, max 186) gegen 28,5 / 27,0 im Steady.

Gepinnt (CPU-Pooltabellen, Kopierkernel aufgezeichnet, Stream-Attrappe):
* plan/reserve/commit/release auf der Tabelle: nur freie LRU-Zeilen unter dem
  Sitzblock, reservierte Zeilen sind OFF (nie frei, nie Opfer), der Commit
  haelt die Bijektion, auch wenn der Experte inzwischen woanders heiss wurde;
* Schalter an: der erste Leerlauf-Tick plant alle Layer, die Ticks waermen wie
  bisher, der Abbruchpunkt schickt den Rest in Layer-Reihenfolge auf den
  Seitenstrom (skipped_cancel=0), vor jedem Forward werden nur FERTIGE Layer
  eingetragen, RW FINISH-LANDED am Ende; Schlaf wartet laufende Kopien ab;
* Schalter aus: der alte Abbruch, Zeichen fuer Zeichen;
* KEINE Wartestelle im Decode-Pfad: promote_ready / _finish / commit /
  reserve / release / warm_copy / deferred_rows_tick enthalten keinen
  Host-Lesezugriff und kein synchronize (AST).
"""

import ast
import inspect
import logging
import os
import textwrap
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.layers.moe import expert_offload as eo  # noqa: E402
from sglang.srt.layers.moe import expert_pool_device as epd  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=15, suite="stage-a-test-cpu")

E, R, ROWS, STAGING = 16, 4, 12, 2        # residents 0..3, LRU rows 4..9, staging 10..11
FINISH_ENV = "SGLANG_WEG2_RESUME_WARM_FINISH"


def _tables():
    hot = {e: e for e in range(R)}
    host = [-1] * R + list(range(E - R))
    return epd.allocate_pool_tables("cpu", E, ROWS, R, STAGING, hot, host)


class _Cache:
    def __init__(self, lid):
        self.layer = types.SimpleNamespace(layer_id=lid)
        self._pool_ready = True
        self._pool_tables = _tables()
        self._pool_dsts = [torch.zeros(1)]
        self._pool_srcs = [torch.zeros(1)]

    lru_snapshot = eo.MoEExpertOffloadCache.lru_snapshot
    warm_lru_local = eo.MoEExpertOffloadCache.warm_lru_local
    pool_row_bytes = eo.MoEExpertOffloadCache.pool_row_bytes
    warm_plan = eo.MoEExpertOffloadCache.warm_plan
    warm_copy = eo.MoEExpertOffloadCache.warm_copy
    warm_commit = eo.MoEExpertOffloadCache.warm_commit
    warm_release = eo.MoEExpertOffloadCache.warm_release


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


class _Ev:
    def __init__(self, log):
        self.done = False
        self.synced = False
        self.log = log

    def query(self):
        return self.done

    def synchronize(self):
        self.synced = True
        self.done = True


class _Ops:
    """The stream attrappe: records what the finish does on which stream."""

    def __init__(self):
        self.log = []
        self.events = []

    def new_stream(self):
        return "side"

    def stream_ctx(self, stream):
        ops = self

        class _Ctx:
            def __enter__(self):
                ops.log.append(("enter", stream))

            def __exit__(self, *a):
                ops.log.append(("exit", stream))

        return _Ctx()

    def after_current(self, stream):
        self.log.append(("after_current", stream))

    def record(self, stream):
        ev = _Ev(self.log)
        self.events.append(ev)
        self.log.append(("record", stream))
        return ev

    def current_waits(self, ev):
        assert ev.done, "the forward stream must only wait on a COMPLETED event"
        self.log.append(("current_waits",))


class _Base(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        for k in (eo.RESUME_WARM_ENV, eo.RESUME_WARM_ROWS_ENV, eo.RESUME_WARM_LAYERS_ENV, FINISH_ENV):
            os.environ.pop(k, None)
        eo._RESUME_WARM = None
        self.copies = []

        def fake_copy(srcs, dsts, src, dst, count):
            self.copies.append(list(zip(src.tolist(), dst.tolist())))

        p = mock.patch.object(epd, "copy_rows", fake_copy)
        p.start()
        self.addCleanup(p.stop)
        q = mock.patch.object(eo, "MoEExpertOffloadCache", _Cache)
        q.start()
        self.addCleanup(q.stop)

    def tearDown(self):
        self._env.stop()
        eo._RESUME_WARM = None

    def _armed(self, n):
        caches = [_Cache(i) for i in range(n)]
        for c in caches:
            _own(c._pool_tables, [(4, 9, 3), (5, 12, 8)])
        rw = eo.resume_warm()
        rw.snapshot([_Model(caches)])
        for c in caches:
            c._pool_tables = _tables()          # the rearm's reinit
        rw.arm([_Model(caches)])
        return rw, caches


class Tables(unittest.TestCase):
    def test_plan_reserve_commit_keep_the_bijection(self):
        t = _tables()
        trip = epd.plan_warm_rows(t, [12, 9, 2, 99])     # 2 resident, 99 out of range
        self.assertEqual([(e, r) for _h, r, e in trip], [(12, 4), (9, 5)])
        self.assertEqual(int(t.row_key[4]), -1)          # planning writes nothing
        rows = torch.tensor([4, 5])
        epd.reserve_warm_rows(t, rows)
        self.assertEqual([int(t.row_key[r]) for r in (4, 5)], [epd.SEAT_OFF_KEY] * 2)
        self.assertEqual(int(t.row_use[4]), epd.ROW_USE_NEVER)
        self.assertEqual(epd.plan_warm_rows(t, [7])[0][1], 6)   # a reserved row is not free
        t.clock[0] = 41
        t.hot_phys[9] = 7                                  # a step made 9 hot elsewhere meanwhile
        t.row_key[7] = 9
        epd.commit_warm_rows(t, torch.tensor([12, 9]), rows)
        self.assertEqual((int(t.hot_phys[12]), int(t.row_key[4]), int(t.row_use[4])), (4, 12, 41))
        self.assertEqual((int(t.hot_phys[9]), int(t.row_key[5]), int(t.row_use[5])), (7, -1, 0))
        self.assertEqual(epd.bijection_breaks(t), 0)
        epd.reserve_warm_rows(t, torch.tensor([6]))
        epd.release_warm_rows(t, torch.tensor([6]))
        self.assertEqual((int(t.row_key[6]), int(t.row_use[6])), (-1, 0))

    def test_no_row_of_the_seat_block(self):
        t = _tables()
        t.seat_rows, t.seat_base = 2, 6                   # rows 6.. are seat rows
        trip = epd.plan_warm_rows(t, list(range(4, 16)))
        self.assertEqual(sorted(r for _h, r, _e in trip), [4, 5])


class Finish(_Base):
    def test_finish_runs_the_rest_on_the_side_stream_and_lands_only_completed_layers(self):
        os.environ[FINISH_ENV] = "1"
        os.environ[eo.RESUME_WARM_LAYERS_ENV] = "2"
        rw, caches = self._armed(5)
        ops = rw._ops = _Ops()
        self.assertEqual(rw.tick(), 2)                    # plans all 5, warms 2 now
        self.assertTrue(all(c._pool_tables.row_key[4] >= 0 for c in caches))   # reserved or owned
        self.assertEqual(rw.cancel("first_decode"), 0)   # nothing skipped
        self.assertEqual(rw.skipped_cancel, 0)
        self.assertEqual(ops.log[:2], [("enter", "side"), ("after_current", "side")])
        self.assertEqual([x[0] for x in ops.log].count("record"), 3)
        self.assertEqual(len(self.copies), 5)             # 2 in the ticks, 3 on the side stream
        # not landed: the tables still hold the reservation, no commit, no wait
        self.assertEqual(rw.promote_ready(), 0)
        self.assertEqual(int(caches[2]._pool_tables.row_key[4]), epd.SEAT_OFF_KEY)
        self.assertNotIn(("current_waits",), ops.log)
        ops.events[0].done = True
        self.assertEqual(rw.promote_ready(), 1)
        self.assertEqual(int(caches[2]._pool_tables.hot_phys[12]), 4)
        ops.events[2].done = True                         # out of order: one stream, waits for [1]
        self.assertEqual(rw.promote_ready(), 0)
        ops.events[1].done = True
        with self.assertLogs(eo.logger, level=logging.INFO) as cm:
            self.assertEqual(rw.promote_ready(), 2)
        self.assertTrue(any("RW FINISH-LANDED layers=3" in m for m in cm.output))
        f = rw.fields()
        self.assertIn("warm_layers=5", f)
        self.assertIn("skipped_cancel=0", f)
        self.assertIn("finish_layers=3", f)
        for c in caches:
            self.assertEqual(epd.bijection_breaks(c._pool_tables), 0)
            self.assertEqual(int(c._pool_tables.hot_phys[12]), 4)

    def test_the_hook_before_every_forward_lands_them(self):
        os.environ[FINISH_ENV] = "1"
        rw, caches = self._armed(3)
        ops = rw._ops = _Ops()
        rw.tick(layers=1)
        rw.cancel("first_decode")
        for ev in ops.events:
            ev.done = True
        batch = types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_decode=lambda: True))
        eo.deferred_rows_tick(batch)
        self.assertEqual(rw.finish_landed, 2)
        self.assertEqual(rw._inflight, [])

    def test_eager_before_the_cancel_gives_the_rows_back(self):
        os.environ[FINISH_ENV] = "1"
        rw, caches = self._armed(3)
        rw._ops = _Ops()
        rw.tick(layers=1)
        rw.eager_reached(caches[2])
        self.assertEqual(int(caches[2]._pool_tables.row_key[4]), -1)   # released
        self.assertEqual(rw.skipped_eager, 1)

    def test_sleep_waits_for_copies_in_flight(self):
        os.environ[FINISH_ENV] = "1"
        rw, caches = self._armed(3)
        ops = rw._ops = _Ops()
        rw.tick(layers=1)
        rw.cancel("first_decode")
        rw.settle()
        self.assertTrue(all(ev.synced for ev in ops.events))
        self.assertEqual(rw._inflight, [])

    def test_switch_off_is_the_old_cancel(self):
        rw, caches = self._armed(3)
        os.environ[eo.RESUME_WARM_LAYERS_ENV] = "1"
        self.assertEqual(rw.tick(), 1)
        self.assertEqual(rw.cancel("first_decode"), 2)
        self.assertIn("skipped_cancel=2 cancel=first_decode", rw.fields())
        self.assertNotIn("finish_layers", rw.fields())

    def test_switch_default_off(self):
        from sglang.srt.environ import envs

        self.assertFalse(envs.SGLANG_WEG2_RESUME_WARM_FINISH.get())


class NoWaitInTheDecodePath(unittest.TestCase):
    FORBIDDEN = ("cpu", "item", "tolist", "synchronize", "numpy")

    def _calls(self, fn):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        return {n.func.attr for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}

    def test_no_host_read_and_no_synchronize(self):
        for fn in (eo.ResumeWarm.promote_ready, eo.ResumeWarm._finish, eo.deferred_rows_tick,
                   eo.MoEExpertOffloadCache.warm_copy, eo.MoEExpertOffloadCache.warm_commit,
                   eo.MoEExpertOffloadCache.warm_release, epd.commit_warm_rows,
                   epd.reserve_warm_rows, epd.release_warm_rows):
            bad = self._calls(fn) & set(self.FORBIDDEN)
            self.assertFalse(bad, f"{fn.__qualname__} calls {bad}")
        # the only event read in the promote is the non-blocking query
        self.assertIn("query", self._calls(eo.ResumeWarm.promote_ready))


if __name__ == "__main__":
    unittest.main()
