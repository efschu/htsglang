# SPDX-License-Identifier: Apache-2.0
"""RW-AT-ARM (30.09., NF P->D): the resume warm is issued at the arm.

Metal (y4l 11e5db4370 n=12 / y4k dff1a7fed4 n=14 P->D wakes, D TP0): the
first decode round after every wake ``gpu-ms`` 148 / 143 (median) against a
steady round of 38 / 31 at the same bs, ``spec_verify:pool.fetch`` 104 / 108
ms -- 26 of 26 flips, with no extend in between (TAIL-SKIP-EXTEND). The warm
armed there ran 8 layers in one idle settle pass and was cancelled with the
settle (``warm_layers=8 skipped_cancel=40``, 12/12 wakes).

Pinned (CPU pool tables, copy kernel recorded, stream attrappe):
* off: the arm only queues (the settle-pass warm, unchanged);
* on: the arm plans + reserves every queued layer (free LRU rows below the
  seat block, capped by _AT_ARM_ROWS) and issues every copy on the side
  stream behind the current one; nothing is left for the ticks or the
  cancel; each layer lands only once its event completed (the hook before
  every forward); a sleep waits for copies still in flight.
"""

import logging
import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.layers.moe import expert_offload as eo  # noqa: E402
from flliper.srt.layers.moe import expert_pool_device as epd  # noqa: E402

E, R, ROWS, STAGING = 16, 4, 12, 2        # residents 0..3, LRU rows 4..9, staging 10..11
AT_ARM = "FLLIPER_PDFLIP_RESUME_WARM_AT_ARM"
AT_ARM_ROWS = "FLLIPER_PDFLIP_RESUME_WARM_AT_ARM_ROWS"


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


class _Model:
    def __init__(self, caches):
        self._mods = [types.SimpleNamespace(_expert_offload=c) for c in caches]

    def modules(self):
        return iter(self._mods)


class _Ev:
    def __init__(self):
        self.done = False

    def query(self):
        return self.done

    def synchronize(self):
        self.done = True


class _Ops:
    def __init__(self):
        self.log, self.events = [], []

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
        ev = _Ev()
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
        for k in (eo.RESUME_WARM_ENV, eo.RESUME_WARM_ROWS_ENV, eo.RESUME_WARM_LAYERS_ENV,
                  "FLLIPER_PDFLIP_RESUME_WARM_FINISH", AT_ARM, AT_ARM_ROWS):
            os.environ.pop(k, None)
        eo._RESUME_WARM = None
        self.copies = []

        def fake_copy(srcs, dsts, src, dst, count):
            self.copies.append(list(zip(src.tolist(), dst.tolist())))

        for target, name, repl in ((epd, "copy_rows", fake_copy), (eo, "MoEExpertOffloadCache", _Cache)):
            p = mock.patch.object(target, name, repl)
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self._env.stop()
        eo._RESUME_WARM = None

    def _armed(self, n):
        caches = [_Cache(i) for i in range(n)]
        for c in caches:  # the last D phase's LRU: experts 9 and 12 (and 13)
            t = c._pool_tables
            for r, e, u in ((4, 9, 3), (5, 12, 8), (6, 13, 5)):
                t.row_key[r], t.hot_phys[e], t.row_use[r] = e, r, u
        rw = eo.resume_warm()
        ops = rw._ops = _Ops()
        rw.snapshot([_Model(caches)])
        for c in caches:
            c._pool_tables = _tables()          # the rearm's reinit
        rw.arm([_Model(caches)])
        return rw, caches, ops


class AtArm(_Base):
    def test_defaults_off_and_off_only_queues(self):
        self.assertIs(envs.FLLIPER_PDFLIP_RESUME_WARM_AT_ARM.get(), False)
        self.assertEqual(envs.FLLIPER_PDFLIP_RESUME_WARM_AT_ARM_ROWS.get(), 16)
        rw, caches, ops = self._armed(3)
        self.assertEqual(rw.pending(), 3)       # the settle-pass warm, as before
        self.assertEqual(self.copies, [])
        self.assertEqual(rw._inflight, [])

    def test_on_the_arm_issues_every_layer_on_the_side_stream(self):
        os.environ[AT_ARM] = "1"
        rw, caches, ops = self._armed(4)
        self.assertEqual(rw.pending(), 0)       # nothing left for the settle ticks
        self.assertEqual(len(self.copies), 4)
        self.assertEqual(ops.log[:2], [("enter", "side"), ("after_current", "side")])
        self.assertEqual([x[0] for x in ops.log].count("record"), 4)
        # most recently used first, free LRU rows, reserved until landed
        self.assertEqual(self.copies[0], [(12 - R, 4), (13 - R, 5), (9 - R, 6)])
        for c in caches:
            self.assertEqual(int(c._pool_tables.row_key[4]), epd.SEAT_OFF_KEY)
        self.assertIn("at_arm=1", rw.fields())

    def test_the_row_cap(self):
        os.environ[AT_ARM] = "1"
        os.environ[AT_ARM_ROWS] = "1"
        rw, caches, ops = self._armed(2)
        self.assertEqual([len(c) for c in self.copies], [1, 1])

    def test_layers_land_before_a_forward_once_their_copy_completed(self):
        os.environ[AT_ARM] = "1"
        rw, caches, ops = self._armed(3)
        batch = types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_decode=lambda: True))
        eo.deferred_rows_tick(batch)            # nothing completed: no commit, no wait
        self.assertEqual(rw.finish_landed, 0)
        self.assertNotIn(("current_waits",), ops.log)
        for ev in ops.events:
            ev.done = True
        with self.assertLogs(eo.logger, level=logging.INFO) as cm:
            eo.deferred_rows_tick(batch)
        self.assertEqual(rw.finish_landed, 3)
        self.assertTrue(any("RW FINISH-LANDED layers=3" in m and "reason=arm" in m for m in cm.output))
        for c in caches:
            t = c._pool_tables
            self.assertEqual((int(t.hot_phys[12]), int(t.row_key[4])), (4, 12))
            self.assertEqual(epd.bijection_breaks(t), 0)

    def test_the_first_decode_cancel_loses_nothing(self):
        os.environ[AT_ARM] = "1"
        rw, caches, ops = self._armed(3)
        self.assertEqual(rw.cancel("first_decode"), 0)
        self.assertEqual(rw.skipped_cancel, 0)
        self.assertEqual(len(rw._inflight), 3)

    def test_a_sleep_waits_for_copies_in_flight(self):
        os.environ[AT_ARM] = "1"
        rw, caches, ops = self._armed(2)
        rw.settle()
        self.assertTrue(all(ev.done for ev in ops.events))
        self.assertEqual(rw._inflight, [])


if __name__ == "__main__":
    unittest.main()
