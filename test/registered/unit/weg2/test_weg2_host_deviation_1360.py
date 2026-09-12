# SPDX-License-Identifier: Apache-2.0
"""#1360 -- the NAMED deviation: the reap mark is soft, silence is not.

Boot weg2xsn26 was refused correctly (`run_peak=95.12 vs hard bound 94.43`,
`source=SERIES` binding weg2xsn25 8.697, W87 -> W20) and the launcher had NO
named override at all -- the boot seat grepped the refusal text, the argparse
and every SGLANG_*OVERRIDE/FORCE/ACCEPT and found zero. The user's standing rule
of 2026-09-12 (memory `host-schwelle-nie-uebertreten`) is that the reap mark is
SOFT: crossing it is allowed WITH a number and a runtime latch, and never
silently. This is that switch, and nothing more than that switch.

WHAT IT CONVERTS: the FUNDABILITY VERDICT of the arm, after it has been computed
in full. WHAT IT DOES NOT TOUCH: any term, the ring dimensioning, the manifest
guard (#1359), the store sizing, and the runtime guard W22 -- which still tears
down at the latch. `test_the_sizing_is_byte_identical_with_and_without_the_flag`
is the ratchet on that claim.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger as hl
from sglang.test.test_utils import CustomTestCase

GIB = hl.GIB
_KW = dict(
    # Sized so the specimen's predicted peak lands BETWEEN the hard bound and
    # the reap watermark -- which is the only window the deviation may act in,
    # and therefore the only specimen that tests it.
    ring_bytes=int(30.4 * GIB), ring_span1_bytes=int(15.0 * GIB),
    cg_current_bytes=int(9.0 * GIB), reclaimable_bytes=int(3.0 * GIB),
    cg_ceiling_bytes=int(123.78 * GIB), s_gb_d=4,
)


def _choose(**extra):
    return hl.choose(
        int(123.78 * GIB), int(110.0 * GIB), arms=[(1, 150)],
        flip_ratchet=hl.resolve_flip_ratchet_gib(None), **_KW, **extra,
    )


class DeviationIsBothOrNeither1360(CustomTestCase):
    def test_a_reason_without_a_latch_refuses_by_name(self):
        with self.assertRaises(hl.Weg2HostDeviationRefused) as cm:
            _choose(deviation_reason="xsn26 Zielfrage")
        self.assertIn("W97 Weg2HostDeviationRefused", str(cm.exception))
        self.assertIn("BOTH OR NEITHER", str(cm.exception))

    def test_a_latch_without_a_reason_refuses_by_name(self):
        with self.assertRaises(hl.Weg2HostDeviationRefused) as cm:
            _choose(riegel_gib=93.0)
        self.assertIn("without the why", str(cm.exception))

    def test_it_is_a_ledger_refusal_so_the_launch_path_already_stops(self):
        self.assertTrue(
            issubclass(hl.Weg2HostDeviationRefused, hl.Weg2HostLedgerRefused))

    def test_without_the_flag_the_arm_is_still_refused(self):
        # W21 is a plain RuntimeError, NOT a Weg2HostLedgerRefused subtype --
        # the peak refusal and the ladder refusal are deliberately two classes
        # so a log says WHICH quantity refused.
        with self.assertRaises(
                (hl.Weg2HostLedgerRefused, hl.Weg2HostRunPeakRefused)):
            _choose()


class TheTwoHardRefusalsStand1360(CustomTestCase):
    def test_a_latch_at_or_above_the_hard_bound_refuses(self):
        """A latch above the bound cannot fire before the bound is crossed."""
        with self.assertRaises(hl.Weg2HostDeviationRefused) as cm:
            _choose(deviation_reason="r", riegel_gib=99.0)
        msg = str(cm.exception)
        self.assertIn("--host-riegel-gib", msg)
        self.assertIn("AT OR ABOVE the hard bound", msg)

    def test_a_predicted_peak_at_or_above_the_reap_watermark_refuses(self):
        """The bound is soft; the WATERMARK is where the kernel reaped."""
        watermark = hl.OBSERVED_REAP_NONRECLAIM_BYTES / GIB
        with self.assertRaises(hl.Weg2HostDeviationRefused) as cm:
            hl.choose(
                int(123.78 * GIB), int(110.0 * GIB), arms=[(1, 2400)],
                flip_ratchet=hl.FlipRatchet(
                    per_flip_gib=40.0, flips_priced=1, source="test",
                    from_record=False),
                deviation_reason="r", riegel_gib=10.0, **_KW,
            )
        msg = str(cm.exception)
        self.assertIn("reap watermark", msg)
        self.assertIn(f"{watermark:.2f}", msg)
        self.assertIn("weg2xsn25", msg)


class TheHappyPathPrintsEveryNumber1360(CustomTestCase):
    def _line(self):
        _arm, _hr, lines = _choose(
            deviation_reason="xsn26 Zielfrage", riegel_gib=93.0)
        hits = [ln for ln in lines if "WEG2-HOST-LEDGER DEVIATION" in ln]
        self.assertEqual(len(hits), 1)
        return hits[0]

    def test_the_line_carries_the_ordered_fields(self):
        ln = self._line()
        for field in ("reason=", "predicted_run_peak=", "hard_bound=", "reap=",
                      "excess=", "riegel_gib=", "source="):
            self.assertIn(field, ln)
        self.assertIn('reason="xsn26 Zielfrage"', ln)
        self.assertIn("riegel_gib=93.00", ln)
        self.assertIn("SERIES", ln)

    def test_the_numbers_are_the_arms_own_and_excess_closes(self):
        import re
        ln = self._line()
        g = {k: float(v) for k, v in re.findall(
            r"(predicted_run_peak|hard_bound|reap|excess)=([-0-9.]+)", ln)}
        self.assertAlmostEqual(
            g["excess"], g["predicted_run_peak"] - g["hard_bound"], places=2)
        self.assertAlmostEqual(
            g["reap"], hl.OBSERVED_REAP_NONRECLAIM_BYTES / GIB, places=2)
        self.assertGreater(g["excess"], 0.0)       # it WAS refused
        self.assertLess(g["predicted_run_peak"], g["reap"])

    def test_the_line_says_what_it_did_not_do(self):
        ln = self._line()
        self.assertIn("NOTHING IS RE-PRICED", ln)
        self.assertIn("W22", ln)
        self.assertIn("2026-09-12", ln)

    def test_an_arm_comes_back_and_it_is_the_one_the_ladder_priced(self):
        arm, headroom, _l = _choose(deviation_reason="r", riegel_gib=93.0)
        self.assertEqual((arm.s_gb, arm.m_mib), (1, 150))
        self.assertIsNotNone(headroom)
        self.assertLess(headroom, 0.0)     # negative: it deviated, honestly


class SizingIsUntouched1360(CustomTestCase):
    """THE RATCHET ON THE CENTRAL CLAIM: the switch accepts, it never re-prices."""

    def test_the_sizing_is_byte_identical_with_and_without_the_flag(self):
        try:
            _choose()
            self.fail("the specimen must be refused without the flag")
        except (hl.Weg2HostLedgerRefused, hl.Weg2HostRunPeakRefused):
            pass
        arm, _h, _l = _choose(deviation_reason="r", riegel_gib=93.0)
        plain = hl.price(
            int(123.78 * GIB), int(110.0 * GIB), 1, 150,
            flip_ratchet=hl.resolve_flip_ratchet_gib(None), **_KW,
        )
        skip = {"run_origin_source", "base_source", "flip_ratchet_source",
                "launch_worst_case_margin_source"}
        self.assertEqual(set(arm.terms) - skip, set(plain.terms) - skip)
        for key in sorted(set(plain.terms) - skip):
            with self.subTest(term=key):
                self.assertEqual(arm.terms[key], plain.terms[key])
        self.assertEqual(arm.launch_leftover_gib, plain.launch_leftover_gib)
        self.assertEqual(arm.run_leftover_gib, plain.run_leftover_gib)

    def test_no_env_twin_exists(self):
        """#781 fossil class: ONE spelling, on the command line."""
        import inspect
        from sglang.srt.weg2 import launcher as lc
        src = inspect.getsource(lc)
        self.assertIn('"--host-ledger-deviation"', src)
        self.assertIn('"--host-riegel-gib"', src)
        for fossil in ("HOST_LEDGER_DEVIATION", "HOST_RIEGEL",
                       "SGLANG_WEG2_DEVIATION"):
            self.assertNotIn(fossil, src)


if __name__ == "__main__":
    unittest.main()
