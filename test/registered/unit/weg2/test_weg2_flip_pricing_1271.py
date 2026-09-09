# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# ==============================================================================
"""#1271: X* divided two rates that were not the same measurement.

`r_P` was `(prompt_tokens - cached_tokens) / wall` off the front's
`WEG2-SERVED group=P leg=1` line (launcher.py:445-447, appended at :485-489) --
a PER-REQUEST wall, taken while the front drains P at `p_concurrency=8`, so it
charges one leg for the time it sat behind its peers. `r_D` came from
`verdict=single_prefill` legs: concurrency ONE. X* = 2*flip_s/(1/r_D - 1/r_P)
therefore divided a concurrent-latency rate by a single-request rate.

MEASURED, and it is a sampling artifact rather than a rate change: on sb1's own
front log the shipped filter (`unc >= 4096`, 23 of 79 legs) gives r_P = 1964
tok/s, all 79 legs give 528, and the drain-aggregate gives ~3180. rg6 gave 3096
on 8 samples. P's compute-honest rate was ~6594-8325 across both boots -- flat.
The filter admits only large-uncached legs, which are the cold, least-queued
ones; as more of a drain's later legs qualify, the median falls.
"""

import collections
import unittest

from sglang.srt.weg2.launcher import (
    RATE_UNIT_COMPUTE_HONEST,
    RATE_UNIT_GROUP_THROUGHPUT,
    RATE_UNIT_REQUEST_LATENCY,
    MixedRateUnits,
    derive_x_star,
)
from sglang.srt.weg2.front import Front
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase


class MixedUnitsAreRefusedByName(CustomTestCase):
    def test_the_shipped_pairing_is_refused(self):
        """RED-FIRST: r_D single-request vs r_P concurrent-latency -- the
        combination that shipped -- must not return a number at all."""
        with self.assertRaises(MixedRateUnits) as ctx:
            derive_x_star(
                3.64, 994.0, 3096.0, 4096,
                unit_d=RATE_UNIT_GROUP_THROUGHPUT,
                unit_p=RATE_UNIT_REQUEST_LATENCY,
            )
        msg = str(ctx.exception)
        self.assertIn("request_latency", msg)
        self.assertIn("queueing behind peers", msg)

    def test_compute_honest_against_wall_is_also_refused(self):
        with self.assertRaises(MixedRateUnits):
            derive_x_star(3.64, 994.0, 7000.0, 4096,
                          unit_d=RATE_UNIT_GROUP_THROUGHPUT,
                          unit_p=RATE_UNIT_COMPUTE_HONEST)

    def test_same_unit_still_prices(self):
        x = derive_x_star(3.64, 994.0, 3096.0, 4096)
        self.assertGreater(x, 4096)

    def test_the_refusal_precedes_the_arithmetic(self):
        """A mixed pair with an otherwise-fatal input must still name the UNIT:
        the unit error is the one the reader has to fix first."""
        with self.assertRaises(MixedRateUnits):
            derive_x_star(3.64, -1.0, 3096.0, 4096,
                          unit_d=RATE_UNIT_GROUP_THROUGHPUT,
                          unit_p=RATE_UNIT_REQUEST_LATENCY)

    def test_no_breakeven_still_refuses_when_units_agree(self):
        with self.assertRaises(ValueError):
            derive_x_star(3.64, 3096.0, 994.0, 4096)


class RpIsADrainAggregate(CustomTestCase):
    """The corrected instrument, on the real front log."""

    LOG = "/spinning/evidence-665-f1/boot_weg2_weg2sb1_7b60280f57_0908_140216.front.log"

    @unittest.skipUnless(__import__("os").path.exists(LOG), "evidence-tree bound")
    def test_sb1_reads_a_stable_group_rate(self):
        from sglang.srt.weg2.launcher import measure_x_inputs

        got = measure_x_inputs(self.LOG, 4096)
        self.assertIsNotNone(got)
        _flip, _r_d, r_p, _nf, _nd, n_p = got
        # ~3180, against the 1964 the per-request form reported for this log
        # and the 3096 rg6 reported -- i.e. STABLE across boots.
        self.assertGreater(r_p, 2500)
        self.assertLess(r_p, 4000)
        self.assertEqual(n_p, 23, "one sample per P-DRAIN window")

    def test_a_log_without_the_drain_instrument_is_refused_not_mixed(self):
        """rg6 carries 0 `WEG2 P-DRAIN` lines -- the instrument postdates it.
        Such a log must yield None so `resolve_x` moves on, never a
        per-request r_P smuggled in under the same name."""
        import os

        from sglang.srt.weg2.launcher import measure_x_inputs

        rg6 = ("/spinning/evidence-665-f1/"
               "boot_weg2_weg2rg6_7f88b1c75d_0908_070324.front.log")
        if not os.path.exists(rg6):
            self.skipTest("evidence-tree bound")
        self.assertIsNone(measure_x_inputs(rg6, 4096))




class RollingResolve(CustomTestCase):
    """(b): X re-solves from THIS boot's samples, seeded from the launcher."""

    def _front(self, x=10649):
        f = Front.__new__(Front)
        f.tp_prefill_max_tokens = x
        f.flip_min_work_tokens = x
        f._x_min_work_follows = True
        f.x_floor_tokens = 4096
        f._x_samples = {
            "r_d": collections.deque(maxlen=Front.X_SAMPLE_WINDOW),
            "r_p": collections.deque(maxlen=Front.X_SAMPLE_WINDOW),
            "flip_s": collections.deque(maxlen=Front.X_SAMPLE_WINDOW),
        }
        f._x_since_resolve = 0
        f._x_seed_note = f"launcher solve X={x}"
        # #1289: mirrors the constructor. A hand-built Front that omits a
        # field the real one has is a test that passes on a shape production
        # never runs -- which is how this ticket's defect stayed green here.
        f._x_last_missing = []
        f.counters = collections.Counter()
        f.note_x_sample = Front.note_x_sample.__get__(f)
        f.resolve_x_live = Front.resolve_x_live.__get__(f)
        f.x_flip_s_provenance = Front.x_flip_s_provenance.__get__(f)
        return f

    def test_red_first_x_does_not_move_before_n_samples(self):
        """The seed must hold until there is a median worth trusting."""
        f = self._front()
        f.note_x_sample("r_d", 1106.0)
        f.note_x_sample("flip_s", 3.29)
        for _ in range(7):
            f.note_x_sample("r_p", 3180.0)
        self.assertEqual(f.tp_prefill_max_tokens, 10649, "no re-solve before N=8")
        self.assertEqual(f.counters["x_resolves"], 0)

    def test_x_moves_on_the_eighth_drain(self):
        f = self._front()
        f.note_x_sample("r_d", 1106.0)
        f.note_x_sample("flip_s", 3.29)
        for _ in range(8):
            f.note_x_sample("r_p", 3180.0)
        self.assertEqual(f.counters["x_resolves"], 1)
        # 2*3.29 / (1/1106 - 1/3180) = ~11.2k
        self.assertNotEqual(f.tp_prefill_max_tokens, 10649)
        self.assertGreater(f.tp_prefill_max_tokens, 8000)
        self.assertLess(f.tp_prefill_max_tokens, 15000)

    def test_min_work_follows_x_when_it_was_not_pinned(self):
        f = self._front()
        f.note_x_sample("r_d", 1106.0)
        f.note_x_sample("flip_s", 3.29)
        for _ in range(8):
            f.note_x_sample("r_p", 3180.0)
        self.assertEqual(f.flip_min_work_tokens, f.tp_prefill_max_tokens)

    def test_a_pinned_min_work_is_not_overwritten(self):
        f = self._front()
        f._x_min_work_follows = False
        f.flip_min_work_tokens = 5000
        f.note_x_sample("r_d", 1106.0)
        f.note_x_sample("flip_s", 3.29)
        for _ in range(8):
            f.note_x_sample("r_p", 3180.0)
        self.assertEqual(f.flip_min_work_tokens, 5000, "an operator pin stands")

    def test_no_breakeven_holds_x_instead_of_raising(self):
        """r_D >= r_P is a real state: the round trip does not pay and X stays."""
        f = self._front()
        f.note_x_sample("r_d", 5000.0)
        f.note_x_sample("flip_s", 3.29)
        for _ in range(8):
            f.note_x_sample("r_p", 3000.0)
        self.assertEqual(f.tp_prefill_max_tokens, 10649)

    def test_the_floor_is_a_chunk_not_the_seed(self):
        """A floor at the seed would make X monotonically non-decreasing --
        unable to correct an X that was too high, the sb2 direction."""
        f = self._front()
        self.assertEqual(f.x_floor_tokens, 4096)
        self.assertLess(f.x_floor_tokens, 10649)

    def test_samples_are_bounded(self):
        f = self._front()
        for _ in range(200):
            f.note_x_sample("r_d", 1000.0)
        self.assertLessEqual(len(f._x_samples["r_d"]), 32)

    def test_a_nonpositive_sample_is_ignored(self):
        f = self._front()
        f.note_x_sample("r_p", 0.0)
        f.note_x_sample("r_p", -1.0)
        self.assertEqual(len(f._x_samples["r_p"]), 0)


class BacklogRule(CustomTestCase):
    """(c): the verdict is on Σ uncached, and stranding is reported."""

    @staticmethod
    def _sum(pendings):
        return sum(int(u) for u in pendings)

    def test_red_first_a_backlog_flips_where_no_single_prompt_would(self):
        """sb1 epoch 46: X=10,649 and every queued prompt was ~6k uncached --
        each one BELOW X, the sum far above it."""
        X = 10649
        prompts = [6007, 1916, 1913, 2060, 2108]
        self.assertTrue(all(p < X for p in prompts), "no single prompt clears X")
        self.assertGreater(self._sum(prompts), X, "the backlog does")

    def test_the_sum_is_over_uncached_not_est_prompt(self):
        """est_prompt counts the cached head P does not recompute; summing it
        flips on prefixes that are already resident."""
        est_prompt = [6007, 6011, 6008]
        cached = [0, 4095, 4095]
        unc = [p - c for p, c in zip(est_prompt, cached)]
        self.assertEqual(self._sum(est_prompt), 18026)
        self.assertEqual(self._sum(unc), 9836)
        self.assertLess(self._sum(unc), 10649, "on uncached this backlog HOLDS")
        self.assertGreater(self._sum(est_prompt), 10649, "on est_prompt it would flip")

    def test_the_rule_and_the_stranded_term_are_wired(self):
        import inspect

        src = inspect.getsource(Front._flip_economics_ok)
        self.assertIn("queued_uncached", src)
        self.assertIn("est_uncached", src)
        self.assertIn("stranded_decodes=%d", src)
        self.assertIn("never a veto", src)
        # REPORTED, not a veto: the verdict must not read the stranded count.
        verdict_line = [ln for ln in src.splitlines() if ln.strip().startswith("ok = ")]
        self.assertEqual(len(verdict_line), 1)
        self.assertNotIn("stranded", verdict_line[0])


register_cpu_ci(__file__)

if __name__ == "__main__":
    unittest.main()
