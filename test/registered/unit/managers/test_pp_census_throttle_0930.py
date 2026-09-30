"""Census throttle (30.09., dual13): #998 / #997d / #1000 / #996 no longer flood the P log in a livelock.

Measured on dual13 (boot_weg2_dkr27bnvfp4dual1gbar1fs09301054, P log): 23776 lines EACH of #998 EXTEND-INVARIANT,
#997d OUTPUT-FILL, #1000 SLOT-OCCUPANT and #996, ~68 lines/s each during 11:00-11:04Z, the same state every time
(#998 seen=237 breaks=0; #997d seen=0; #1000 only its counters rising). Such lines keep every silence watcher awake.

Pinned here:
  * N unchanged passes give at most log2(N)+1 lines -- with fast passes (the time backoff 1, 2, 4 ... s binds)
    and with slow passes (the pass backoff 2, 4, 8 ... binds);
  * the first line comes at once, a CHANGE comes at once (and names how many lines were held back);
  * the held-back count rides on the emitted line (no extra lines);
  * pure pass counters (#1000 seen/reason counts, #996 admissions/site counts) are not "state";
  * SGLANG_PP_CENSUS_THROTTLE=0 restores every line;
  * pp_ring_note itself, driven with a fake holder through the dual13 shape, stays within the bound.
"""

import logging
import math
import os
import types
import unittest
from unittest import mock

from sglang.srt.managers import scheduler_pp_mixin as M


def _holder(rank=1):
    return types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=rank))


class TestThrottle(unittest.TestCase):
    def setUp(self):
        os.environ.pop(M.CENSUS_THROTTLE_ENV, None)

    def _count(self, n, dt):
        h, lines = _holder(), 0
        for i in range(n):
            if M._census_throttle(h, "#998", ("same",), now=1000.0 + i * dt) is not None:
                lines += 1
        return lines

    def test_unchanged_state_log2_bound_fast_and_slow_passes(self):
        for n in (1, 2, 3, 10, 100, 1000, 12198):
            bound = math.floor(math.log2(n)) + 1
            self.assertLessEqual(self._count(n, 0.0147), bound, n)   # dual13 cadence: 68 lines/s
            self.assertLessEqual(self._count(n, 10.0), bound, n)     # slow passes: the pass backoff binds
        self.assertGreaterEqual(self._count(1000, 0.0147), 2)        # it still speaks, with its count

    def test_first_line_and_changes_immediately(self):
        h = _holder()
        self.assertEqual(M._census_throttle(h, "#998", ("a",), now=0.0), "")          # first: at once
        held = sum(M._census_throttle(h, "#998", ("a",), now=0.001 * i) is None for i in range(1, 50))
        self.assertGreater(held, 0)
        s = M._census_throttle(h, "#998", ("b",), now=0.06)                            # change: at once
        self.assertIsNotNone(s)
        self.assertIn("held back", s)
        self.assertEqual(M._census_throttle(h, "#998", ("c",), now=0.061), "")         # next change: at once

    def test_markers_are_independent(self):
        h = _holder()
        M._census_throttle(h, "#998", ("a",), now=0.0)
        self.assertEqual(M._census_throttle(h, "#1000", ("a",), now=0.0), "")

    def test_env_off_restores_every_line(self):
        with mock.patch.dict(os.environ, {M.CENSUS_THROTTLE_ENV: "0"}):
            h = _holder()
            self.assertTrue(all(M._census_throttle(h, "#998", ("a",), now=i * 0.01) == "" for i in range(100)))

    def test_count_rides_on_the_same_line(self):
        h, out = _holder(), []
        with mock.patch.object(M.logger, "warning", side_effect=lambda *a: out.append(a[0] % a[1:])):
            for i in range(200):
                M._census_log(h, "#998", "#998 EXTEND-INVARIANT rank=%s seen=%d", 1, 237)
        self.assertLessEqual(len(out), math.floor(math.log2(200)) + 1)
        self.assertTrue(all(l.startswith("#998 EXTEND-INVARIANT rank=1 seen=237") for l in out))


class TestPpRingNoteDual13Shape(unittest.TestCase):
    """pp_ring_note with a fake holder: the #998/#997d state constant, #1000 seen and #996 admissions rising
    on every pass (exactly the dual13 P log) -> each marker stays within log2(N)+1 lines."""

    def test_livelock_shape(self):
        from sglang.srt.managers import schedule_batch as SB

        h = _holder(1)
        out = []
        n_passes = 400
        with mock.patch.dict(os.environ, {"SGLANG_947_RING_EVERY": "1"}), \
             mock.patch.object(M.logger, "warning", side_effect=lambda *a: out.append(a[0] % a[1:])), \
             mock.patch.object(M.time, "monotonic", side_effect=[1000.0 + 0.0147 * i for i in range(100000)]):
            for _ in range(n_passes):
                M._1000_SEEN[0] += 100                        # a pass counter, not state
                M._1000_REASONS["no-statement"] = M._1000_SEEN[0]
                M.pp_ring_note(h, "admit", False)
        bound = math.floor(math.log2(n_passes)) + 1
        for mk in ("#998 EXTEND-INVARIANT", "#997d OUTPUT-FILL", "#1000 SLOT-OCCUPANT", "#996 group_floor"):
            n = sum(1 for l in out if l.startswith(mk))
            self.assertGreaterEqual(n, 1, mk)
            self.assertLessEqual(n, bound, (mk, n))
        # a real state change of #998 (a break) goes out at once
        before = sum(1 for l in out if l.startswith("#998"))
        SB._998_BREAKS[0] += 1
        with mock.patch.dict(os.environ, {"SGLANG_947_RING_EVERY": "1"}), \
             mock.patch.object(M.logger, "warning", side_effect=lambda *a: out.append(a[0] % a[1:])):
            M.pp_ring_note(h, "admit", False)
        SB._998_BREAKS[0] -= 1
        self.assertEqual(sum(1 for l in out if l.startswith("#998")), before + 1)


if __name__ == "__main__":
    unittest.main()
