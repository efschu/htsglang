# SPDX-License-Identifier: Apache-2.0
"""Weg 2 (#1233, train fix 3): the STORE is bounded by the reap point.

THE DEFECT this file pins, measured on the train tip a917eb404c
(`python -m sglang.srt.weg2.launcher --dry-run`, 2026-09-08 10:00Z, rc=2 on
BOTH idle layouts): the store was sized as the WHOLE run leftover and the
reap-point check was applied AFTERWARDS as a gate, so every arm of the ladder
carried its own refusal by construction::

    ARM S=1 M=2400: leftover run=10.24 -> store=10 -> run_peak 98.60 vs reap 95.90
    ARM S=1 M=1200: leftover run=15.18 -> store=15 -> run_peak 98.66
    ARM S=1 M=600 : leftover run=17.65 -> store=17 -> run_peak 98.19

and the lever the refusal named (``--store-min-gib``) is a MINIMUM, hence
inert against a peak that is too HIGH (fix 2's record: 4 and 2 gave identical
refusals).

THE CALIBRATION that says the peak MODEL is right and the STORE SIZING is the
defect -- boot weg2rg6 (base 7f88b1c75d, front log lines 24-28) chose
S=1 M=1200 store=10 GiB, and the train-tip model prices that same arm at
21.38 + 17.23 + 4.75 + 7.45 + 0.49 + 0.17 + 32.19 + 10 = 93.66 GiB while rg6
measured memory.current flat at 91.31-92.89 over 26 load samples, whole-boot
margin 2.35 GiB to the 95.90 GiB reap watermark = 93.55 GiB peak
(`/spinning/gpu-arb/weg2/BOOT_weg2rg6_0908.md` lines 330-345).  The model is
within 0.11 GiB of the metal.  What grew was the LEFTOVER, because fix 6
deleted the #1232 ``host_headroom`` compensation term (16 GiB) -- the store
then swallowed that room and the peak check refused the result.

THE FIX under test: the store is ``min(run leftover, reap_bound)`` with
``reap_bound = reap_point - unsampled_term - peak_without_store``, where the
unsampled term is the ``slab_reclaimable`` the reap watermark's own row LACKED
(the row has no slab column, so the watermark is an UPPER bound) -- read LIVE
from ``memory.stat``, never a constant.  ``--store-min-gib`` stays the floor
it is; an arm whose reap-bounded store falls below it is refused by W21 with
the bound printed.

Hermetic: ``CUDA_VISIBLE_DEVICES=""``, no server, no NVML, no checkpoint, no
real ``/proc`` and no real cgroup -- every reading is passed in.
"""

import math
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger
from sglang.test.test_utils import CustomTestCase

GIB = host_ledger.GIB

#: The watermark this whole file is bounded against, from the module itself --
#: boot weg2dk5's reap row, 95.90 GiB non-reclaimable.
REAP_GIB = host_ledger.OBSERVED_REAP_NONRECLAIM_BYTES / GIB

# --------------------------------------------------------- MEASURED: the arms
# The three ARM lines of the train-tip dry-run above, decomposed.  The
# peak-WITHOUT-store figure is that line's own run_peak minus its own store, so
# these are not new numbers: 98.66 - 15 = 83.66, 98.19 - 17 = 81.19.
M1200_LEFTOVER_GIB = 15.18
M1200_PEAK_WO_STORE_GIB = 83.66
M600_LEFTOVER_GIB = 17.65
M600_PEAK_WO_STORE_GIB = 81.19
#: Boot weg2rg6's own arm: the leftover that ALREADY fits under the bound.  The
#: fix must be byte-identical here -- rg6 ran store=10 and lived.
RG6_LEFTOVER_GIB = 10.45
#: ``/sys/fs/cgroup/memory.stat slab_reclaimable`` at the fix-6 reading, 0.73
#: GiB (0.53 GiB live 2026-09-07T23:57:35Z).  The reap row carries no slab
#: column, so this term is what the watermark over-states by.
UNSAMPLED_GIB = 0.73

# ------------------------------------------------- MEASURED: the train-tip box
# The TERMS line of the same dry-run, so the ladder below is that box.
BOX_MEMTOTAL_B = int(118.05 * GIB)
BOX_MEMAVAIL_B = int(103.46 * GIB)
BOX_CG_CURRENT_B = int(44.87 * GIB)
BOX_CG_RECLAIM_B = int(30.28 * GIB)        # -> non-reclaimable 14.58 GiB
BOX_SLAB_B = int(UNSAMPLED_GIB * GIB)
RING_BYTES = 32964 * 1024 * 1024           # Sigma H, boot weg2rg6's ring table
RING_SPAN1_BYTES = 29912 * 1024 * 1024     # Sigma image_P
STORE_MIN_GIB = 8.0


def _ladder(**over):
    kw = dict(
        store_min_gib=STORE_MIN_GIB,
        ring_bytes=RING_BYTES,
        ring_span1_bytes=RING_SPAN1_BYTES,
        ring_provenance="test: boot weg2rg6 ring table",
        cg_current_bytes=BOX_CG_CURRENT_B,
        reclaimable_bytes=BOX_CG_RECLAIM_B,
        slab_reclaimable_bytes=BOX_SLAB_B,
        cg_ceiling_bytes=BOX_MEMTOTAL_B,
        cg_ceiling_source="test ceiling",
        cg_oom_kill=42,
        measured_record=None,
    )
    kw.update(over)
    return host_ledger.choose(BOX_MEMTOTAL_B, BOX_MEMAVAIL_B, **kw)


class TestStoreReapBound(CustomTestCase):
    """The pure sizing: three numbers in, one store and its winning bound out."""

    def test_m1200_arm_is_bounded_by_the_reap_point(self):
        s = host_ledger.size_store_gib(
            M1200_LEFTOVER_GIB, M1200_PEAK_WO_STORE_GIB, UNSAMPLED_GIB
        )
        # 95.90 - 0.73 - 83.66 = 11.51 -> floored to 11, and 11 < 15.18, so the
        # reap point is what bounds this arm, not the leftover.
        self.assertAlmostEqual(s.reap_bound_gib, REAP_GIB - UNSAMPLED_GIB - M1200_PEAK_WO_STORE_GIB, places=6)
        self.assertAlmostEqual(s.reap_bound_gib, 11.51, places=2)
        self.assertEqual(s.gib, 11)
        self.assertEqual(s.bound, "reap")
        self.assertAlmostEqual(s.leftover_gib, M1200_LEFTOVER_GIB, places=6)

    def test_m600_arm_is_bounded_by_the_reap_point(self):
        s = host_ledger.size_store_gib(
            M600_LEFTOVER_GIB, M600_PEAK_WO_STORE_GIB, UNSAMPLED_GIB
        )
        self.assertAlmostEqual(s.reap_bound_gib, 13.98, places=2)
        self.assertEqual(s.gib, 13)
        self.assertEqual(s.bound, "reap")

    def test_the_bounded_store_predicts_a_peak_under_the_watermark(self):
        """The point of the bound: peak_without_store + store stays below reap
        by AT LEAST the unsampled term the watermark over-states by."""
        for leftover, peak_wo in (
            (M1200_LEFTOVER_GIB, M1200_PEAK_WO_STORE_GIB),
            (M600_LEFTOVER_GIB, M600_PEAK_WO_STORE_GIB),
        ):
            s = host_ledger.size_store_gib(leftover, peak_wo, UNSAMPLED_GIB)
            self.assertLessEqual(peak_wo + s.gib, REAP_GIB - UNSAMPLED_GIB)

    def test_rg6_arm_unchanged_where_the_leftover_already_fits(self):
        """Boot weg2rg6's own arm: leftover 10.45 under a bound of 11.51, so the
        LEFTOVER wins and the store is the 10 GiB that boot actually ran."""
        s = host_ledger.size_store_gib(
            RG6_LEFTOVER_GIB, M1200_PEAK_WO_STORE_GIB, UNSAMPLED_GIB
        )
        self.assertEqual(s.gib, 10)
        self.assertEqual(s.bound, "leftover")
        self.assertEqual(s.gib, math.floor(RG6_LEFTOVER_GIB))

    def test_reap_bound_below_the_floor_is_a_refusal_not_a_smaller_store(self):
        """peak_without_store 90 GiB leaves 5.17 GiB of room -- below the 8 GiB
        floor, so this arm is refused (and W21 names the bound)."""
        s = host_ledger.size_store_gib(15.0, 90.0, UNSAMPLED_GIB)
        self.assertAlmostEqual(s.reap_bound_gib, 5.17, places=2)
        self.assertEqual(s.gib, 5)
        self.assertEqual(s.bound, "reap")
        self.assertLess(s.gib, STORE_MIN_GIB)

    def test_the_unsampled_term_is_subtracted_not_assumed_zero(self):
        """Dropping the term would hand the store the 0.73 GiB the watermark
        over-states by -- 12 instead of 11 on the train-tip arm."""
        without = host_ledger.size_store_gib(
            M1200_LEFTOVER_GIB, M1200_PEAK_WO_STORE_GIB, 0.0
        )
        self.assertEqual(without.gib, 12)
        with_term = host_ledger.size_store_gib(
            M1200_LEFTOVER_GIB, M1200_PEAK_WO_STORE_GIB, UNSAMPLED_GIB
        )
        self.assertEqual(with_term.gib, 11)

    def test_an_unreadable_slab_term_is_named_never_priced_as_zero(self):
        s = host_ledger.size_store_gib(M1200_LEFTOVER_GIB, M1200_PEAK_WO_STORE_GIB, None)
        self.assertIsNone(s.unsampled_gib)
        self.assertIn("unsampled", s.note)
        self.assertIn("unreadable", s.note)

    def test_no_run_peak_prediction_falls_back_to_the_leftover_and_says_so(self):
        s = host_ledger.size_store_gib(M1200_LEFTOVER_GIB, None, UNSAMPLED_GIB)
        self.assertIsNone(s.reap_bound_gib)
        self.assertEqual(s.gib, 15)
        self.assertEqual(s.bound, "leftover")

    def test_a_negative_run_leftover_stores_nothing(self):
        s = host_ledger.size_store_gib(-3.0, M1200_PEAK_WO_STORE_GIB, UNSAMPLED_GIB)
        self.assertEqual(s.gib, 0)


class TestLadderOnTheTrainTipBox(CustomTestCase):
    """The same three arms through :func:`host_ledger.choose`, on the readings
    the refusing dry-run itself printed."""

    def test_the_train_tip_box_now_funds_an_arm(self):
        arm, store, lines = _ladder()
        self.assertEqual((arm.s_gb, arm.m_mib), (1, 1200))
        self.assertEqual(store, 11)
        chosen = [ln for ln in lines if "WEG2-HOST-LEDGER CHOSEN" in ln]
        self.assertEqual(len(chosen), 1)
        # The CHOSEN line carries the store's PROVENANCE: which bound won.
        self.assertIn("bound=reap", chosen[0])
        self.assertIn("reap-bound", chosen[0])

    def test_every_arm_line_prints_all_three_numbers_and_the_winner(self):
        _, _, lines = _ladder()
        arms = [ln for ln in lines if "WEG2-HOST-LEDGER ARM" in ln]
        self.assertEqual(len(arms), 3)
        for ln in arms:
            self.assertIn("leftover ", ln)
            self.assertIn("reap-bound ", ln)
            self.assertIn("floor ", ln)
            self.assertIn("bound=", ln)

    def test_the_chosen_arms_predicted_peak_is_below_the_watermark(self):
        arm, store, _ = _ladder()
        self.assertLess(arm.predicted_run_peak_gib(store), REAP_GIB)
        # ... and below it by at least the unsampled term the watermark
        # over-states by, which is the whole reason that term is subtracted.
        self.assertLessEqual(arm.predicted_run_peak_gib(store), REAP_GIB - UNSAMPLED_GIB)

    def test_the_floor_still_refuses_and_it_is_W21_that_names_the_bound(self):
        """A floor above every arm's reap bound: the binding quantity is the
        reap point, so the refusal is W21 (not W20) and prints the bound."""
        with self.assertRaises(host_ledger.Weg2HostRunPeakRefused) as cm:
            _ladder(store_min_gib=14.0)
        text = str(cm.exception)
        self.assertIn("reap-bound", text)
        self.assertIn("bound=reap", text)

    def test_an_unreadable_slab_reading_still_bounds_the_store(self):
        """With no slab reading the bound is computed WITHOUT the term and says
        so -- it is never silently priced as a whole-leftover store again."""
        arm, store, lines = _ladder(slab_reclaimable_bytes=None)
        self.assertLess(arm.predicted_run_peak_gib(store), REAP_GIB)
        self.assertTrue(
            any("unsampled unreadable" in ln for ln in lines if "ARM" in ln)
        )


if __name__ == "__main__":
    unittest.main()
