# SPDX-License-Identifier: Apache-2.0
"""#656: the corridor law and the gate's arming floor are ONE declared pair.

WHAT WENT WRONG. The acceptance boot armed its corridor gate at 1536 MiB and
read its corridor verdict against 1024 MiB. Both numbers were correct on
their own terms -- the separation is deliberate and a previous shift proved
it prevents a pp->tp deadlock -- but nothing anywhere declared what the 512
MiB BETWEEN them was for. It is the draw a seam is assumed to make while it
runs, and on this rig the corridor sampler measured that draw at 1814-1852
MiB. So five cutovers passed a gate with no objection and took GPU0 to 886
MiB, 138 below the law, and the breach lived precisely in the gap between
the gate's number and the verdict's number, where nothing looks.

Three properties are pinned here:

* the arming floor is DERIVED from the law plus a named reserve, so raising
  one cannot leave the other behind;
* an arming floor BELOW the law is refused, because such a gate would return
  "no reclaim needed" for an allocation that ends under the corridor -- it
  would launder a breach as a passed check, which the guard's own refusal
  message says it must never do;
* every module that needs the law reads the SAME declaration.
  ``corridor_trace.summary`` used to default to a literal 1024 of its own,
  which is how an instrument ends up reporting a different verdict from the
  gate it audits.
"""

import unittest

from sglang.srt.managers import corridor_guard as cg
from sglang.srt.mem_ledger import corridor_trace
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTheDeclaredPair(CustomTestCase):
    def test_the_arming_floor_is_the_law_plus_a_named_reserve(self):
        self.assertEqual(
            cg.arming_floor_mib(),
            cg.CORRIDOR_LAW_MIB + cg.DEFAULT_SEAM_ENTRY_RESERVE_MIB,
        )
        # The shipped pair, unchanged: 1024 + 512 = 1536 is what the
        # acceptance boot ran, so deriving it is not a behaviour change.
        self.assertEqual(cg.CORRIDOR_LAW_MIB, 1024)
        self.assertEqual(cg.arming_floor_mib(), 1536)

    def test_the_reserve_moves_the_floor_and_only_the_floor(self):
        """The measured draw is the input the term is meant to take."""
        self.assertEqual(cg.arming_floor_mib(1852), 1024 + 1852)
        self.assertEqual(cg.arming_floor_mib(0), cg.CORRIDOR_LAW_MIB)
        # A negative reserve cannot pull the floor under the law.
        self.assertEqual(cg.arming_floor_mib(-4096), cg.CORRIDOR_LAW_MIB)

    def test_can_fail_an_arming_floor_below_the_law_is_refused(self):
        with self.assertRaisesRegex(ValueError, "BELOW the corridor law"):
            cg.check_threshold_pair(1023, 1024)
        with self.assertRaisesRegex(ValueError, "BELOW the corridor law"):
            cg.check_threshold_pair(0, cg.CORRIDOR_LAW_MIB)

    def test_equality_and_above_are_accepted(self):
        cg.check_threshold_pair(cg.CORRIDOR_LAW_MIB, cg.CORRIDOR_LAW_MIB)
        cg.check_threshold_pair(cg.arming_floor_mib(), cg.CORRIDOR_LAW_MIB)
        cg.check_threshold_pair(cg.arming_floor_mib(1852), cg.CORRIDOR_LAW_MIB)

    def test_the_legacy_name_is_the_same_number(self):
        """Callers pass DEFAULT_FLOOR_MIB as the guard's law; it must not
        drift from the canonical name now that both exist."""
        self.assertEqual(cg.DEFAULT_FLOOR_MIB, cg.CORRIDOR_LAW_MIB)


class TestOneDeclaration(CustomTestCase):
    def test_the_trace_reads_the_guards_law(self):
        self.assertEqual(corridor_trace.corridor_law_mib(), cg.CORRIDOR_LAW_MIB)

    def test_the_trace_summary_defaults_to_the_declared_law(self):
        """The pin that stops a private literal creeping back in."""
        trace = corridor_trace.CorridorTrace(card_uuid="test")
        trace.samples.append(
            corridor_trace.Sample(
                monotonic=0.0,
                nvml_free_bytes=900 * corridor_trace.MIB,
                nvml_self_bytes=0,
                kv_arena_backed_bytes=0,
                torch_reserved_bytes=0,
                torch_allocated_bytes=0,
            )
        )
        trace.samples.append(
            corridor_trace.Sample(
                monotonic=1.0,
                nvml_free_bytes=4096 * corridor_trace.MIB,
                nvml_self_bytes=0,
                kv_arena_backed_bytes=0,
                torch_reserved_bytes=0,
                torch_allocated_bytes=0,
            )
        )
        summary = trace.summary()
        self.assertEqual(summary["corridor_mib"], cg.CORRIDOR_LAW_MIB)
        # 900 MiB is under the law, so the verdict must be a breach and the
        # margin must be the signed depth, not an absolute value.
        self.assertTrue(summary["breach"])
        self.assertEqual(summary["free_min_mib"], 900)
        self.assertEqual(summary["margin_mib"], 900 - cg.CORRIDOR_LAW_MIB)

    def test_an_explicit_corridor_still_wins(self):
        trace = corridor_trace.CorridorTrace(card_uuid="test")
        trace.samples.append(
            corridor_trace.Sample(
                monotonic=0.0,
                nvml_free_bytes=1500 * corridor_trace.MIB,
                nvml_self_bytes=0,
                kv_arena_backed_bytes=0,
                torch_reserved_bytes=0,
                torch_allocated_bytes=0,
            )
        )
        self.assertFalse(trace.summary()["breach"])
        self.assertTrue(trace.summary(corridor_mib=2048)["breach"])


if __name__ == "__main__":
    unittest.main()


class TestTheLawHasOneReader(CustomTestCase):
    """`SGLANG_CORRIDOR_LAW_FLOOR_MIB` was read in three places, each with
    its own `"1024"` fallback. The law could then be moved for one module
    and not the others -- a divergence with no symptom until a breach is
    judged twice and answered differently."""

    def setUp(self):
        import os

        self._saved = os.environ.pop(cg.LAW_ENV, None)

    def tearDown(self):
        import os

        os.environ.pop(cg.LAW_ENV, None)
        if self._saved is not None:
            os.environ[cg.LAW_ENV] = self._saved

    def test_unset_is_the_declared_constant(self):
        self.assertEqual(cg.corridor_law_mib(), cg.CORRIDOR_LAW_MIB)
        self.assertEqual(cg.corridor_law_bytes(), cg.CORRIDOR_LAW_MIB << 20)

    def test_every_consumer_moves_together(self):
        import os

        from sglang.srt.managers import phase_flip_seam_census as census
        from sglang.srt.mem_cache import kv_vmm_backing

        os.environ[cg.LAW_ENV] = "1500"
        self.assertEqual(cg.corridor_law_mib(), 1500)
        self.assertEqual(corridor_trace.corridor_law_mib(), 1500)
        self.assertEqual(census.law_floor_bytes(), 1500 << 20)
        self.assertEqual(kv_vmm_backing._corridor_law_floor_bytes(), 1500 << 20)
        # ... and the arming floor follows the law rather than staying put.
        self.assertEqual(
            cg.arming_floor_mib(law_mib=cg.corridor_law_mib()),
            1500 + cg.DEFAULT_SEAM_ENTRY_RESERVE_MIB,
        )

    def test_a_malformed_override_falls_back_to_the_constant(self):
        import os

        os.environ[cg.LAW_ENV] = "not-a-number"
        self.assertEqual(cg.corridor_law_mib(), cg.CORRIDOR_LAW_MIB)
