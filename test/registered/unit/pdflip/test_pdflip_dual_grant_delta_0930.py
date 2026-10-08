# SPDX-License-Identifier: Apache-2.0
"""DUAL-TP3PP3 (operator order after dual13): PP0 asks only for the
DIFFERENCE between the grant level and the mapping it already holds. A request
whose level the mapping already covers never waits.

DANGER DIRECTIONS guarded here:
* covered level, short card: the grant is taken and nothing is charged;
* a larger level charges exactly the difference and still waits when the
  difference does not fit. P never presses D;
* afterwards the ledger equals the mapping: never less, since the mapping
  must stay covered, and never more, since a double count starves the next
  grant;
* follower cards keep the full unit plus the excess return on adoption.
  PP0 cannot see a follower's mapping, and a follower may idle-release
  between PP0's grant and its adoption.
"""
from __future__ import annotations

import os
import tempfile
import types
import unittest.mock as mock
import uuid

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from flliper.srt.pdflip import card_kv_ledger as K
from flliper.srt.pdflip import dual_p_kv_stage as S
from flliper.srt.pdflip.d_seat_vram import AllocInfo
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

G = 2 << 20
ROW = 2048
ALLOC = (16384 + 64) * ROW
MIB = 1 << 20


class FakeSpans:
    available = True

    def info(self, ptr):
        return AllocInfo(size=ALLOC, mapped=0, planned=0, active=True)

    def set_spans(self, ptr, spans, now):
        return 0


def _stage(led):
    return S.PKvStage([(1, S._geom_for(torch.zeros(16448, 512), 16384, 64, "k", ALLOC))], led,
                      allocator=object(), pools=[], page_size=64, granule=G, top_tokens=16384,
                      spans=FakeSpans(), engage_cap=lambda *a: None)


def _req(rid, n):
    return types.SimpleNamespace(_dual_grant_untold=None, rid=rid, origin_input_ids=list(range(n)), output_ids=[])


class GrantDelta(CustomTestCase):
    def setUp(self):
        self.tag = "t-%s" % uuid.uuid4().hex[:8]
        self.env = mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_DUAL_KV_TAG": self.tag})
        self.env.start()
        S._reset_wait_log()

    def tearDown(self):
        self.env.stop()
        for r in range(3):
            try:
                os.unlink(S.stage_file(self.tag, r))
            except OSError:
                pass

    def _pp0(self, n_followers=0):
        root = tempfile.mkdtemp(prefix="wkvdl")
        paths = [os.path.join(root, "card%d" % i) for i in range(1 + n_followers)]
        ds = []
        for p in paths:
            d = K.CardKvLedger(p, "D")
            d.contribute(40 * MIB, committed=40 * MIB)          # D keeps its boot pool until P joined
            ds.append(d)
        stages = []
        for i, p in enumerate(paths):
            led = K.CardKvLedger(p, "P")
            led.contribute(30 * MIB)                             # budget 70 MiB per card
            st = _stage(led)
            S.publish_stage(st, self.tag, i)
            stages.append(st)
            ds[i].release(20 * MIB)                              # then shrinks: D holds 20 MiB
        sched = types.SimpleNamespace(tp_worker=types.SimpleNamespace(
            model_runner=types.SimpleNamespace(dual_p_kv=stages[0])),
            ps=types.SimpleNamespace(pp_rank=0, pp_size=len(paths)))
        return sched, stages, paths, ds

    def test_a_covered_level_never_waits(self):
        sched, (pp0,), (path,), (d,) = self._pp0()
        u = pp0.bytes_for(12288) - pp0.bytes_for(0)             # 24 MiB
        self.assertEqual(S.pp0_grant(sched, _req("r1", 12000)), 12288)
        self.assertEqual(K.peek(path).committed["P"], u)
        d.request(20 * MIB)                                     # D grows: free = 70 - 24 - 40 = 6 MiB < u
        self.assertLess(K.peek(path).free, u)
        self.assertEqual(S.pp0_grant(sched, _req("r2", 12000)), 12288, "P waited for a level it already maps")
        self.assertEqual(K.peek(path).committed["P"], u)          # nothing charged twice
        self.assertEqual(pp0._committed, u)

    def test_a_larger_level_charges_the_difference_and_waits_when_it_does_not_fit(self):
        sched, (pp0,), (path,), (d,) = self._pp0()
        S.pp0_grant(sched, _req("r1", 12000))                   # 12288 mapped, 24 MiB
        d.request(20 * MIB)                                     # free 6 MiB
        diff = pp0.bytes_for(16384) - pp0.bytes_for(12288)      # 8 MiB
        self.assertGreater(diff, K.peek(path).free)
        self.assertEqual(S.pp0_grant(sched, _req("r3", 16000)), 0)
        self.assertEqual(K.peek(path).committed["P"], pp0.bytes_for(12288) - pp0.bytes_for(0))
        self.assertEqual(K.peek(path).committed["D"], 40 * MIB)  # P never presses D
        d.release(2 * MIB)                                      # free 8 MiB: exactly the difference
        self.assertEqual(S.pp0_grant(sched, _req("r3", 16000)), 16384)
        self.assertEqual(K.peek(path).committed["P"], pp0.bytes_for(16384) - pp0.bytes_for(0))
        self.assertEqual(pp0.mapped_tokens, 16384)
        self.assertLessEqual(sum(K.peek(path).committed.values()), K.peek(path).budget)

    def test_follower_cards_keep_the_full_unit_and_return_it_on_adoption(self):
        sched, (pp0, f1), (p0, p1), _ds = self._pp0(n_followers=1)
        lvl = S.pp0_grant(sched, _req("r1", 12000))
        u1 = f1.bytes_for(12288) - f1.bytes_for(0)
        self.assertEqual(K.peek(p1).committed["P"], u1)          # charged in full for the follower
        f1.map_granted(lvl)                                      # adoption
        lvl2 = S.pp0_grant(sched, _req("r2", 12000))
        self.assertEqual(K.peek(p1).committed["P"], 2 * u1)      # in flight: still charged in full
        f1.map_granted(lvl2)
        self.assertEqual(K.peek(p1).committed["P"], u1)          # ... and returned on adoption
        self.assertEqual(K.peek(p0).committed["P"], pp0.bytes_for(12288) - pp0.bytes_for(0))

    def test_idle_release_then_new_grant_charges_the_full_level_again(self):
        sched, (pp0,), (path,), _ = self._pp0()
        S.pp0_grant(sched, _req("r1", 12000))
        with mock.patch.object(S, "max_live_id", lambda *a: 0):
            pp0.release_all()
        self.assertEqual(K.peek(path).committed["P"], 0)
        self.assertEqual(S.pp0_grant(sched, _req("r2", 12000)), 12288)
        self.assertEqual(K.peek(path).committed["P"], pp0.bytes_for(12288) - pp0.bytes_for(0))


if __name__ == "__main__":
    import unittest

    unittest.main()
