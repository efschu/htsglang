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

from flliper.srt.pdflip import card_kv_ledger as K
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

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

    def test_policy_d_first_p_pauses(self):
        # user order 07:25Z: KV short -> P pauses and frees its context; D has priority
        self.assertEqual(K.arbitrate(100, 300, 0, requester="D", other_committed=900), (100, 0))
        self.assertEqual(K.arbitrate(500, 300, 0, requester="D", other_committed=900), (300, 200))
        self.assertEqual(K.arbitrate(500, 300, 0, requester="D", other_committed=50), (300, 50))
        self.assertEqual(K.arbitrate(500, 300, 999, requester="P", other_committed=900), (300, 0))
        self.assertEqual(K.arbitrate(500, 0, 0, requester="P"), (0, 0))

    def test_grant_pressure_release(self):
        d = K.CardKvLedger(self.path, "D")
        p = K.CardKvLedger(self.path, "P")
        d.join(1000 * MIB)
        p.join(1000 * MIB)
        self.assertEqual(d.request(900 * MIB)[0], 900 * MIB)          # D takes the idle card
        g, pr = p.request(300 * MIB)                                  # P's prompt arrives
        self.assertEqual((g, pr), (100 * MIB, 0))                     # P never presses D: it waits
        self.assertEqual(d.pressure_on_me(), 0)
        d.release(400 * MIB)                                          # D's seats finish
        self.assertEqual(p.request(200 * MIB)[0], 200 * MIB)
        self.assertEqual(p.state().committed, {"P": 300 * MIB, "D": 500 * MIB})
        g, pr = d.request(400 * MIB)                                  # D needs it back: priority
        self.assertEqual((g, pr), (200 * MIB, 200 * MIB))
        self.assertEqual(p.pressure_on_me(), 200 * MIB)               # P pauses ...
        p.release(300 * MIB)                                          # ... and frees its whole context
        self.assertEqual(p.pressure_on_me(), 0)
        self.assertEqual(d.request(200 * MIB)[0], 200 * MIB)
        self.assertEqual(d.state().committed, {"P": 0, "D": 900 * MIB})

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

    def test_pool_is_the_sum_of_both_boot_kv(self):
        d = K.CardKvLedger(self.path, "D")
        p = K.CardKvLedger(self.path, "P")
        d.contribute(2400 * MIB, committed=2400 * MIB)     # D keeps its boot stage mapped
        p.contribute(2000 * MIB, committed=0)              # P puts all of its KV into the pool
        st = p.state()
        self.assertEqual(st.budget, 4400 * MIB)
        self.assertEqual(st.free, 2000 * MIB)
        self.assertEqual(p.request(3000 * MIB), (2000 * MIB, 0))   # P waits for the rest
        p.release(2000 * MIB)
        self.assertEqual(d.request(2000 * MIB)[0], 2000 * MIB)     # D may grow into P's share
        p.contribute(2000 * MIB, committed=0)                     # idempotent
        self.assertEqual(p.state().budget, 4400 * MIB)

