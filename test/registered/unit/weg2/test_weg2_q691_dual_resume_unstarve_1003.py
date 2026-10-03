# SPDX-License-Identifier: Apache-2.0
"""Q-691 DUAL RESUME-UNSTARVE (27B dual y8x, boot dkr27bnvfp4dual1mpsleepbar1fs10031727, bc2bd121c0).

METAL (front log, 03.10. 17:52:30-17:54:10+): the paused head weg2-0-309
(dual_paused_n > 0, head_uncached=173) waited > 90 s in RESUME-WAIT on per card
(pressure, P committed, D demand) = [(0,1107296256,0),(0,201326592,0),
(0,301989888,0)] while 12 requests went past it ('WEG2 DUAL SHORT-BYPASS ...
past=weg2-0-309 head_wait_s=34.8..93.7'). P was never idle, so 'P committed'
never reached 0 and the Q-680 stale bound (P idle) never applied.

RED on bc2bd121c0:
  * a short paused head (uncached <= SHORT_BYPASS_TOKENS) on those rows stays held;
  * a long paused head that waited the stale bound past a bypass stays held.
GREEN on both (danger directions):
  * pressure or D demand on any card -> the head waits;
  * a long head nobody went past waits until all zeros (or Q-680);
  * flip form unchanged.
"""
from __future__ import annotations

import asyncio
import collections
import os
import tempfile
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import card_kv_ledger as K  # noqa: E402
from sglang.srt.weg2 import dual_parallel as DP  # noqa: E402
from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MIB = 1 << 20
METAL_COMMITTED = (1107296256, 201326592, 301989888)
UNSTARVE = "DUAL RESUME-UNSTARVE"


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())


def _front(dual=True):
    f = F.Front(prefill="http://p", decode="http://d", awake="D" if dual else "P", tag="dual",
                store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0, weight_chunks=2,
                flip_min_work_tokens=1, dual_layout=dual)

    async def rpc(g, path, body, timeout):
        return 200, "{}"

    f.rpc = rpc
    return f


def _pending(rid, uncached, paused=0):
    fut = asyncio.get_event_loop().create_future()
    p = F.Pending(rid, "/generate", {}, "x", time.time(), fut, est_prompt=uncached, est_uncached=uncached)
    p.dual_paused_n = paused
    return p


def _metal_ledgers(d_demand=False):
    """The metal rows: P committed on every card, no pressure, no D demand."""
    root = tempfile.mkdtemp(prefix="wkv691")
    paths = [os.path.join(root, "card%d" % i) for i in range(3)]
    for pth in paths:
        K.CardKvLedger(pth, "D").contribute(4000 * MIB, committed=0)
        K.CardKvLedger(pth, "P").contribute(0)
    for pth, c in zip(paths, METAL_COMMITTED):
        K.CardKvLedger(pth, "P").request(c)
    if d_demand:
        K.CardKvLedger(paths[1], "D").request(8000 * MIB)
    return paths


class _Base(CustomTestCase):
    def _held(self, f):
        return f._dual_resume_held()

    def _setup(self, head_uncached, *, dual=True, d_demand=False, behind=()):
        f = _front(dual)
        f.dual_kv_ledgers = _metal_ledgers(d_demand=d_demand)
        f.queue = collections.deque([_pending("weg2-0-309", head_uncached, paused=1)]
                                    + [_pending(r, u) for r, u in behind])
        return f


class ShortHeadResumes(_Base):
    """(a) the metal head: 173 tokens left, P committed > 0, no pressure -> resume."""

    def test_metal_short_paused_head_resumes_on_p_committed(self):
        async def run():
            f = self._setup(173)
            # metal: P was never idle -- a SHORT-BYPASS leg is in flight (Q-693:
            # without one, 'P committed' is the head's own previous instance)
            f._dual_inflight["weg2-0-320"] = _pending("weg2-0-320", 600)
            with self.assertLogs(F.logger.name, level="WARNING") as cm:
                held = self._held(f)
            return f, held, cm.output

        f, held, out = _run(run())
        self.assertFalse(held, "the short paused head stayed in RESUME-WAIT (metal: > 90 s past 12 bypasses)")
        self.assertTrue(any("%s rid=weg2-0-309 reason=short" % UNSTARVE in m and "head_uncached=173" in m
                            for m in out), out)
        self.assertEqual(f.counters["dual_resume_unstarve"], 1)
        self.assertIsNone(f._dual_resume_wait_since)


class BypassedHeadResumes(_Base):
    """(b) a long head that waited the stale bound while a SHORT-BYPASS went past it."""

    def test_long_head_bypassed_past_the_stale_bound_resumes(self):
        async def run():
            f = self._setup(90000, behind=[("weg2-0-310", 600)])
            r = [self._held(f)]
            moved = f._dual_short_reorder(head_blocked=True)       # the pump's SHORT-BYPASS
            head = f.queue[1]
            short = f.queue.popleft()                               # the short one went to P
            f._dual_inflight[short.rid] = short                     # ... and is in flight there
            r.append(self._held(f))                                 # young wait: still held
            f._dual_resume_wait_since = time.time() - 11.0          # >= RESUME_STALE_S (10)
            with self.assertLogs(F.logger.name, level="WARNING") as cm:
                r.append(self._held(f))
            return f, moved, head, r, cm.output

        f, moved, head, r, out = _run(run())
        self.assertTrue(moved)
        self.assertEqual(r[:2], [True, True], "a long head resumed before the bound")
        self.assertFalse(r[2], "the bypassed long head starved in RESUME-WAIT")
        self.assertTrue(any("%s rid=weg2-0-309 reason=bypassed" % UNSTARVE in m and "bypassed=1" in m
                            for m in out), out)
        self.assertEqual(head.dual_bypassed_n, 0, "the bypass count survives the resume")


class StillWaits(_Base):
    """Danger directions: green on both sides."""

    def test_pressure_on_a_card_keeps_the_short_head_waiting(self):
        orig = K.p_resume_ready
        K.p_resume_ready = lambda paths: (False, [(0, METAL_COMMITTED[0], 0), (4096, 0, 0), (0, 0, 0)])
        try:
            async def run():
                f = self._setup(173)
                return f, self._held(f)

            f, held = _run(run())
        finally:
            K.p_resume_ready = orig
        self.assertTrue(held, "a short head resumed into pressure")
        self.assertEqual(f.counters.get("dual_resume_unstarve", 0), 0)

    def test_d_demand_keeps_the_short_head_waiting(self):
        async def run():
            f = self._setup(173, d_demand=True)
            return self._held(f)

        self.assertTrue(_run(run()), "a short head resumed while D still asks for KV")

    def test_a_missing_ledger_row_keeps_it_waiting(self):
        self.assertIsNone(DP.resume_unstarve([(0, 1, 0), None], head_uncached=1, short_limit=8192,
                                             wait_s=99.0, stale_s=10.0, bypassed=3))
        self.assertIsNone(DP.resume_unstarve([], head_uncached=1, short_limit=8192,
                                             wait_s=99.0, stale_s=10.0, bypassed=3))

    def test_long_head_without_bypass_waits_until_all_zeros(self):
        async def run():
            f = self._setup(90000)
            r = [self._held(f)]
            f._dual_resume_wait_since = time.time() - 60.0
            r.append(self._held(f))
            for pth in f.dual_kv_ledgers:                       # every P stage released
                led = K.CardKvLedger(pth, "P")
                led.release(led.state().committed["P"])
            r.append(self._held(f))
            return f, r

        f, r = _run(run())
        self.assertEqual(r, [True, True, False])
        self.assertEqual(f.counters.get("dual_resume_unstarve", 0), 0)

    def test_bypassed_long_head_below_the_bound_waits(self):
        async def run():
            f = self._setup(90000, behind=[("weg2-0-310", 600)])
            self._held(f)
            f._dual_short_reorder(head_blocked=True)
            f.queue.popleft()
            f._dual_resume_wait_since = time.time() - 5.0
            return self._held(f)

        self.assertTrue(_run(run()))


class FlipUnchanged(_Base):
    """Flip form (no dual layout): no unstarve, no bypass count."""

    def test_flip_form_short_paused_head_is_not_unstarved(self):
        async def run():
            f = self._setup(173, dual=False, behind=[("weg2-0-310", 600)])
            held = self._held(f)
            moved = f._dual_short_reorder(head_blocked=True)
            return f, held, moved

        f, held, moved = _run(run())
        self.assertTrue(held, "the flip form's resume gate changed")
        self.assertFalse(moved)
        self.assertEqual(f.counters.get("dual_resume_unstarve", 0), 0)
        self.assertEqual(getattr(f.queue[0], "dual_bypassed_n", 0), 0)


if __name__ == "__main__":
    import unittest

    unittest.main()
