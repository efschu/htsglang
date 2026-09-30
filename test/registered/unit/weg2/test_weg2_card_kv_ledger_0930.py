# SPDX-License-Identifier: Apache-2.0
"""DUAL-TP3PP3 U1: the card KV ledger P and D share (unified KV per card).

User order 30.09. ~07:10Z: one KV pool per card, runtime arbitration between
the two processes, no boot split, no quota, no reserve.

DANGER DIRECTIONS guarded here:
* I1: committed[P] + committed[D] never exceeds the card budget, also under
  two processes hammering the ledger concurrently;
* no quota/reserve: a request takes what is free; the shortfall becomes
  pressure on the OTHER side's cache only (never more than it reports
  evictable), and a release answers that pressure;
* a dead process's bytes are reaped (the driver freed them);
* two processes stating different budgets for one card are refused.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import card_kv_ledger as K
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

MIB = 1 << 20


def _hammer(path, group, n, chunk, out):
    led = K.CardKvLedger(path, group)
    got = 0
    for i in range(n):
        g, _ = led.request(chunk)
        got += g
        if i % 3 == 2 and got:
            got -= led.release(chunk)
        st = led.state()
        assert sum(st.committed.values()) <= st.budget
    out.put((group, got, led.state().committed[group]))


class CardKvLedger(CustomTestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="wkv")
        self.path = os.path.join(self.dir, "card")

    def test_policy_no_quota_no_reserve(self):
        self.assertEqual(K.arbitrate(100, 300, 0), (100, 0))
        self.assertEqual(K.arbitrate(500, 300, 1000), (300, 200))
        self.assertEqual(K.arbitrate(500, 300, 50), (300, 50))  # never more than the other's cache
        self.assertEqual(K.arbitrate(500, 0, 0), (0, 0))

    def test_grant_pressure_release(self):
        d = K.CardKvLedger(self.path, "D")
        p = K.CardKvLedger(self.path, "P")
        d.join(1000 * MIB)
        p.join(1000 * MIB)
        self.assertEqual(d.request(900 * MIB)[0], 900 * MIB)          # D takes the idle card
        g, pr = p.request(300 * MIB, other_evictable=800 * MIB)       # P's prompt arrives
        self.assertEqual((g, pr), (100 * MIB, 200 * MIB))
        self.assertEqual(d.pressure_on_me(), 200 * MIB)
        d.release(200 * MIB)                                          # D evicts cache
        self.assertEqual(d.pressure_on_me(), 0)
        self.assertEqual(p.request(200 * MIB)[0], 200 * MIB)
        st = p.state()
        self.assertEqual(st.committed, {"P": 300 * MIB, "D": 700 * MIB})
        p.release(300 * MIB)                                          # P hands off, returns all
        self.assertEqual(d.request(300 * MIB)[0], 300 * MIB)          # D grows back
        self.assertEqual(d.state().committed["D"], 1000 * MIB)

    def test_budget_disagreement_refused(self):
        K.CardKvLedger(self.path, "D").join(1000 * MIB)
        with self.assertRaises(RuntimeError):
            K.CardKvLedger(self.path, "P").join(900 * MIB)

    def test_dead_process_reaped(self):
        alive = {os.getpid(): True, 999999: False}
        p = K.CardKvLedger(self.path, "P", pid_alive=lambda pid: alive.get(pid, False))
        p.join(1000 * MIB, pid=999999)
        p._pid_alive = lambda pid: True
        p.request(600 * MIB)
        d = K.CardKvLedger(self.path, "D", pid_alive=lambda pid: alive.get(pid, False))
        d.join(1000 * MIB)
        self.assertEqual(d.state().committed["P"], 0)                 # pid 999999 is gone
        self.assertEqual(d.request(1000 * MIB)[0], 1000 * MIB)

    def test_two_processes_never_break_i1(self):
        K.CardKvLedger(self.path, "D").join(512 * MIB)
        K.CardKvLedger(self.path, "P").join(512 * MIB)
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        ps = [ctx.Process(target=_hammer, args=(self.path, g, 400, 16 * MIB, q)) for g in ("P", "D")]
        [x.start() for x in ps]
        res = [q.get(timeout=120) for _ in ps]
        [x.join(60) for x in ps]
        for g, got, committed in res:
            self.assertEqual(got, committed, g)
        st = K.CardKvLedger(self.path, "D").state()
        self.assertLessEqual(sum(st.committed.values()), st.budget)
