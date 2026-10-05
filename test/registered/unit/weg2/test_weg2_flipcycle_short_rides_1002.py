"""FLIPCYCLE H5 (02.10.): the P phase cap never strands a SHORT.

y6z boot 081355Z, epoch 1: the P phase ended at its cap (dispatched=6 cap=6,
queue_left=2). The 25-token SHORT weg2-1-7, queued while P was awake, stayed in
the queue, became a QUEUED-SHORT on D after the P->D flip and D prefilled it
(2.03 s pass, every decode seat stalled) -- the d_extend of the flip view and a
breach of E2 (no D prefill after a flip). With ``cap_exempt`` the pool still
dispatches such a SHORT past the cap; the LONG items keep waiting.
"""

import asyncio
import collections
import types
import unittest

from sglang.srt.environ import envs
from sglang.srt.weg2 import front


def _drain(items, cap, exempt):
    async def main():
        q = collections.deque(items)
        done = []

        async def one(i):
            await asyncio.sleep(0)
            return i

        stats = {}
        await front._p_drain_pool(q, 2, one, done.append, lambda: True,
                                  max_dispatch=cap, stats=stats, cap_exempt=exempt)
        return done, list(q), stats

    return asyncio.run(main())


def _short(name):
    return name.startswith("S")


class TheCapLeavesNoShortBehind(unittest.TestCase):
    def test_without_exempt_the_cap_strands_the_short(self):
        done, left, _ = _drain(["L1", "L2", "S1", "L3"], 2, None)
        self.assertEqual(done, ["L1", "L2"])
        self.assertEqual(left, ["S1", "L3"])

    def test_short_past_the_cap_rides_p(self):
        done, left, stats = _drain(["L1", "L2", "L3", "S1", "L4", "S2"], 2, _short)
        self.assertEqual(sorted(done), ["L1", "L2", "S1", "S2"])
        self.assertEqual(left, ["L3", "L4"])  # the LONG ones wait, in order
        self.assertEqual(stats["short_rides"], 2)

    def test_under_the_cap_nothing_changes(self):
        done, left, stats = _drain(["S1", "L1"], 0, _short)
        self.assertEqual(done, ["S1", "L1"])
        self.assertEqual(left, [])
        self.assertEqual(stats["short_rides"], 0)


class ThePredicateIsTheQueuedShortFilter(unittest.TestCase):
    def _front(self, x=4096):
        f = types.SimpleNamespace(tp_prefill_max_tokens=x)
        return f

    def _p(self, **kw):
        base = dict(d_eligible=True, intake_stalled=False, leg1_done=False, reroutes=0,
                    x_requeues=0, p_only=False, x_deferred=False, client_gone=False,
                    est_uncached=25)
        base.update(kw)
        return types.SimpleNamespace(**base)

    def test_short_rides(self):
        self.assertTrue(front.Front._p_phase_short_rides(self._front(), self._p()))

    def test_long_or_flagged_does_not(self):
        f = self._front(4096)
        self.assertFalse(front.Front._p_phase_short_rides(f, self._p(est_uncached=5000)))
        self.assertFalse(front.Front._p_phase_short_rides(f, self._p(d_eligible=False)))
        self.assertFalse(front.Front._p_phase_short_rides(f, self._p(p_only=True)))
        self.assertFalse(front.Front._p_phase_short_rides(f, self._p(x_requeues=1)))

    def test_switch_is_on_by_default(self):
        self.assertTrue(envs.SGLANG_WEG2_ENABLE_P_PHASE_SHORT_RIDES.get())

    def test_controller_passes_the_predicate(self):
        import inspect

        src = inspect.getsource(front)
        self.assertIn("cap_exempt=(self._p_phase_short_rides", src)
        self.assertIn("WEG2-FLIPCYCLE stage=d_prefill_after_flip", src)

    def test_seat_gate_cap_is_never_exceeded(self):
        # 27B head (DECODE-COLLECT SEAT GATE, NF e124d8f431): the P cap IS the free D
        # seat count; the H5 exemption stays off while it is set.
        import inspect

        src = inspect.getsource(front)
        self.assertIn("envs.SGLANG_WEG2_ENABLE_P_PHASE_SHORT_RIDES.get() and _dc_cap <= 0", src)


if __name__ == "__main__":
    unittest.main()
