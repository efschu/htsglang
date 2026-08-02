"""A measurement from one rig may only be applied to a rig it matches (#434).

Three gates, one rule. Each of them used to hand the development rig's own
measured numbers to a machine that merely resembled it, and in each case the
resemblance was checked on the wrong axis:

``flags._match_calibration``
    matched on CARD MODEL NAMES alone. The reference rig's 3080s are the
    20 GB variant and the stock RTX 3080 is a 10 GB card
    (``planner/card_library.py``), so a stock 5090 + 2x 3080 10 GB box was
    handed a KV token vector and a 3000 MiB reserve solved against cards with
    twice the memory. A token vector and a reserve are BUDGET-space
    quantities; the name is not the budget.

``lever_profiles._speed_tie_tol``
    presented the reference rig's boot-to-boot noise floor as *"this rig's
    measured noise floor"* on every rig. A noise floor is a property of a
    machine, so on any other machine that sentence is false.

``uneven_perf.PerfCalibration``
    reported only the OVERRIDDEN case, so silence meant "reference fit" and
    nobody was told. The borrowed case is now the one the plan log names.

FALSIFIER for all three: every test below fails on the pre-#434 tree.

Nothing here touches NVML, CUDA, a model cache or the local machine.
"""

import os
import sys
import tempfile
import time
import unittest

# The synthetic foreign rig lives next door and is deliberately not duplicated:
# a second copy would drift, and the point of the last class here is that the
# borrowed notice reaches the log of the SAME plan the generality suite runs.
# There is no package __init__ in this directory, so the sibling is imported by
# path the way pytest's own rootdir insertion would.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sglang.srt.planner import crossover, flags
from sglang.srt.planner import lever_profiles as lp
from sglang.srt.uneven_perf import PerfCalibration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _gpu(name, total_mib):
    return {"name": name, "total_mib": total_mib}


#: The rig the calibration was solved on.
_CALIBRATED = [
    _gpu("NVIDIA GeForce RTX 5090", 32607),
    _gpu("NVIDIA GeForce RTX 3080", 20480),
    _gpu("NVIDIA GeForce RTX 3080", 20480),
]
#: The same card MODELS, one VRAM tier down on the two small cards. A real,
#: purchasable machine -- the 10 GB 3080 is the stock part.
_SAME_NAMES_HALF_THE_VRAM = [
    _gpu("NVIDIA GeForce RTX 5090", 32607),
    _gpu("NVIDIA GeForce RTX 3080", 10240),
    _gpu("NVIDIA GeForce RTX 3080", 10240),
]


class TestTheCalibrationGateChecksTheBudget(CustomTestCase):
    def test_the_calibrated_rig_still_gets_its_calibration(self):
        """Anti-vacuity: the gate must not have been tightened into always
        refusing. The rig it was measured on still matches."""
        self.assertIsNotNone(flags._match_calibration(_CALIBRATED, "fp8"))

    def test_the_same_names_at_half_the_vram_get_nothing(self):
        self.assertIsNone(
            flags._match_calibration(_SAME_NAMES_HALF_THE_VRAM, "fp8"),
            "a measured KV vector and reserve were applied to cards with half "
            "the memory they were solved against",
        )

    def test_a_small_nvml_difference_still_matches(self):
        """NVML totals of nominally identical cards move by a few MiB with ECC
        state, driver and vBIOS. The gate must survive a driver upgrade of the
        rig it describes -- it is a VRAM-TIER check, not an equality check."""
        drifted = [
            _gpu("NVIDIA GeForce RTX 5090", 32600),
            _gpu("NVIDIA GeForce RTX 3080", 20470),
            _gpu("NVIDIA GeForce RTX 3080", 20490),
        ]
        self.assertIsNotNone(flags._match_calibration(drifted, "fp8"))

    def test_an_inventory_without_totals_is_refused_not_assumed(self):
        """A manual or offline inventory carries no NVML total. The gate
        cannot check the budget it is about to hand out, so it declines --
        guessing here is the failure it exists to prevent."""
        nameless_budget = [
            _gpu("NVIDIA GeForce RTX 5090", 0),
            _gpu("NVIDIA GeForce RTX 3080", 0),
            _gpu("NVIDIA GeForce RTX 3080", 0),
        ]
        self.assertIsNone(flags._match_calibration(nameless_budget, "fp8"))


def _finding(provenance, *, floor=(0.4, 2.6), bypass=True, age_s=0.0):
    return crossover.CrossoverFinding(
        rig=crossover.RigDescriptor(
            cards=("SYNTH Accel A", "SYNTH Accel B"),
            model="SynthDense-27B",
            quant="fp8",
            tp_size=2,
        ),
        points=[],
        provenance=provenance,
        measured_at=time.time() - age_s,
        noise_floor_pct={"ms_per_spec_step": floor},
        cache_bypass_proven=bypass,
    )


class TestTheTieToleranceNamesWhereItCameFrom(CustomTestCase):
    def test_without_a_local_measurement_the_borrowed_number_says_so(self):
        tol, provenance = lp._speed_tie_tol(None)
        self.assertEqual(tol, lp._SPEED_TIE_TOL_PCT_FALLBACK)
        self.assertIn("REFERENCE", provenance)
        self.assertNotIn("this rig's measured", provenance)

    def test_a_local_a_vs_a_measurement_replaces_it(self):
        tol, provenance = lp._speed_tie_tol(_finding(crossover.MEASURED_HERE))
        # The UPPER end of the band: the tolerance has to clear the worst
        # spread the rig showed, not the best one.
        self.assertAlmostEqual(tol, 2.6)
        self.assertIn("this rig's measured", provenance)

    def test_another_rigs_finding_is_not_a_local_measurement(self):
        for finding in (
            _finding(crossover.MEASURED_ELSEWHERE),
            _finding(crossover.MODELLED),
            _finding(crossover.MEASURED_HERE, bypass=False),
        ):
            with self.subTest(provenance=finding.provenance):
                tol, provenance = lp._speed_tie_tol(finding)
                self.assertEqual(tol, lp._SPEED_TIE_TOL_PCT_FALLBACK)
                self.assertIn("REFERENCE", provenance)

    def test_the_shipped_reference_finding_is_refused_by_construction(self):
        """``crossover.REFERENCE_FINDING`` is the fork's own measurement of
        its development rig, shipped with provenance MEASURED_ELSEWHERE so
        that every consumer refuses it without having to remember to. Pin
        that, because it is the pattern the rest of this file argues for."""
        self.assertEqual(
            crossover.REFERENCE_FINDING.provenance, crossover.MEASURED_ELSEWHERE
        )
        self.assertFalse(crossover.REFERENCE_FINDING.usable_for_advice())
        tol, provenance = lp._speed_tie_tol(crossover.REFERENCE_FINDING)
        self.assertEqual(tol, lp._SPEED_TIE_TOL_PCT_FALLBACK)
        self.assertIn("REFERENCE", provenance)


class TestTheCostModelSaysWhichScalarsAreBorrowed(CustomTestCase):
    def test_an_unrefit_calibration_reports_every_field_as_borrowed(self):
        cal = PerfCalibration()
        self.assertEqual(cal.overridden_fields(), [])
        borrowed = cal.borrowed_fields()
        self.assertTrue(borrowed, "an unrefit calibration borrows everything")
        for name in (
            "decode_gemv_residual_exp",
            "decode_peak_compression_exp",
            "decode_nonweight_fraction",
            "prefill_invariant_fraction",
        ):
            self.assertIn(name, borrowed)

    def test_the_two_sets_partition_the_fields(self):
        """A field is borrowed or overridden, never both and never neither --
        otherwise the plan log can report a scalar twice or not at all."""
        cal = PerfCalibration(decode_nonweight_fraction=0.31)
        borrowed = set(cal.borrowed_fields())
        overridden = set(cal.overridden_fields())
        self.assertEqual(overridden, {"decode_nonweight_fraction"})
        self.assertEqual(borrowed & overridden, set())
        self.assertEqual(
            borrowed | overridden,
            {f for f in vars(cal)},
        )

    def test_a_fully_refit_calibration_borrows_nothing(self):
        cal = PerfCalibration(
            decode_gemv_residual_exp=0.6,
            decode_peak_compression_exp=0.5,
            decode_nonweight_fraction=0.3,
            prefill_invariant_fraction=0.2,
        )
        self.assertEqual(cal.borrowed_fields(), [])


class TestTheBorrowedNoticeReachesThePlanLog(CustomTestCase):
    """The methods above are only useful if the boot prints them. Assert the
    plan log carries the borrowed line on a foreign rig -- which is every rig
    that has not refitted, i.e. all of them today."""

    def test_the_plan_log_names_the_borrowed_scalars(self):
        from test_planner_generality_434 import (
            _mix_cards,
            _plan,
            _write_checkpoint,
        )

        with tempfile.TemporaryDirectory() as root:
            model = _write_checkpoint(root, "Synth-fp8", "fp8")
            _sa, log = _plan(model, _mix_cards())
        self.assertIn("calibration BORROWED", log)
        self.assertIn("decode_gemv_residual_exp", log)
        self.assertIn("SGLANG_PERF_", log)


if __name__ == "__main__":
    unittest.main()
