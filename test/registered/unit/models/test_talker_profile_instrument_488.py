# SPDX-License-Identifier: Apache-2.0
"""#488 precursor: the profiler's own logic, exercised off-GPU.

The measurement arms need a card. Everything that DECIDES -- the spread
precondition, the decomposition arithmetic, the verdict wording and the RTF
projection -- is pure and runs here, so the script is not desk-written code
that first executes inside a scarce GPU window.

The arms below include the two refusals, because a gate that has never been
seen to fire is not known to be a gate.
"""

import os
import sys
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "99")

import importlib.util
import pathlib

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[4]
    / "scripts"
    / "dev"
    / "488_talker_profile"
    / "profile_talker_steps.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("talker_profile_488", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: the script uses `from __future__ import
    # annotations`, and @dataclass resolves those string annotations through
    # sys.modules[cls.__module__]. Without this line the import dies inside
    # dataclasses with an unrelated-looking AttributeError.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prof = _load_module()


def _arm(name, wall_ms, kernel_ms, kernels=100, iterations=10, syncs=0):
    return prof.ArmResult(
        name=name,
        iterations=iterations,
        wall_s=iterations * wall_ms / 1000.0,
        kernel_s=iterations * kernel_ms / 1000.0,
        kernel_count=iterations * kernels,
        sync_count=syncs,
    )


class TestDecomposition(CustomTestCase):
    def test_gap_is_wall_minus_kernels(self):
        arm = _arm("x", wall_ms=6.4, kernel_ms=0.5)
        self.assertAlmostEqual(arm.wall_per_iter_ms, 6.4, places=6)
        self.assertAlmostEqual(arm.kernel_per_iter_ms, 0.5, places=6)
        self.assertAlmostEqual(arm.gap_ms, 5.9, places=6)
        self.assertAlmostEqual(arm.gap_fraction, 5.9 / 6.4, places=6)
        self.assertAlmostEqual(arm.kernels_per_iter, 100.0, places=6)

    def test_gap_never_goes_negative(self):
        """Kernels can overlap and outlast the wall clock on paper; a negative
        gap would be reported as a negative overhead, which is nonsense."""
        arm = _arm("x", wall_ms=1.0, kernel_ms=1.4)
        self.assertEqual(arm.gap_ms, 0.0)
        self.assertEqual(arm.gap_fraction, 0.0)

    def test_zero_wall_does_not_divide_by_zero(self):
        arm = _arm("x", wall_ms=0.0, kernel_ms=0.0)
        self.assertEqual(arm.gap_fraction, 0.0)


class TestDiscriminationGate(CustomTestCase):
    """The spread precondition. Both refusals are executed."""

    def test_clean_separation_passes(self):
        gpu = _arm("calib_gpu_bound", wall_ms=10.0, kernel_ms=9.5)
        launch = _arm("calib_launch_bound", wall_ms=10.0, kernel_ms=0.5)
        verdict = prof.check_discrimination(gpu, launch)
        self.assertTrue(verdict.ok, verdict.reason)
        self.assertGreater(verdict.separation, prof._MIN_SEPARATION)

    def test_refuses_when_the_arms_do_not_separate(self):
        """CAN-FAIL ARM: a profiler that reads both regimes the same way.

        The gpu-bound arm is kept UNDER the contention ceiling on purpose, so
        this arm exercises the separation branch and not the contention one.
        """
        gpu = _arm("calib_gpu_bound", wall_ms=10.0, kernel_ms=7.0)
        launch = _arm("calib_launch_bound", wall_ms=10.0, kernel_ms=6.5)
        verdict = prof.check_discrimination(gpu, launch)
        self.assertFalse(verdict.ok)
        self.assertIn("cannot distinguish", verdict.reason)

    def test_refuses_on_a_contended_card(self):
        """CAN-FAIL ARM, and the important one: on a busy box every arm looks
        overhead-bound, which would manufacture the expected answer."""
        gpu = _arm("calib_gpu_bound", wall_ms=10.0, kernel_ms=2.0)
        launch = _arm("calib_launch_bound", wall_ms=10.0, kernel_ms=0.1)
        verdict = prof.check_discrimination(gpu, launch)
        self.assertFalse(verdict.ok)
        self.assertIn("GPU-BOUND arm", verdict.reason)
        # Note the trap this closes: the SEPARATION is fine here (0.79 - 0.80
        # is within noise of the passing case), so a separation-only gate
        # would have let a contended box through.
        self.assertGreater(verdict.separation, 0.0)


class TestHeadroomGate(CustomTestCase):
    """The corridor precondition. This runs INSIDE a process serving a live
    conversation, so an OOM here is a dropped turn in front of the user."""

    def test_the_measured_2026_08_04_headroom_passes(self):
        """3605 MiB free on the 5090, rank 0 at 22436 and the tenant at 5910."""
        self.assertIsNone(prof.check_headroom(3605.0))

    def test_the_calibration_transient_is_under_100_mib(self):
        """A precondition on the precondition: a gate that needed gigabytes
        would never pass on this card and would silently never run."""
        self.assertLess(prof._CALIB_MIB, 100.0)

    def test_refuses_when_the_corridor_would_be_broken(self):
        """CAN-FAIL ARM. Derived from the constants rather than hardcoded: the
        transient was resized once already (96 -> 24 MiB) and a fixture pinned
        to the old value stopped exercising this branch silently."""
        refusal = prof.check_headroom(prof._CALIB_MIB + prof._MIN_FREE_MIB_AFTER - 1.0)
        self.assertIsNotNone(refusal)
        self.assertIn("corridor", refusal)
        self.assertIn("dropped turn", refusal)

    def test_the_floor_is_what_binds_not_merely_fitting(self):
        """REACH INCLUDES PARAMETERS: a gate that only checked 'does it fit'
        would pass at 100 MiB free, which is exactly the corridor violation
        the rule exists to prevent."""
        self.assertGreater(prof._CALIB_MIB, 0.0)
        self.assertIsNone(prof.check_headroom(prof._CALIB_MIB + 401.0))
        self.assertIsNotNone(prof.check_headroom(prof._CALIB_MIB + 399.0))


class TestVerdictWording(CustomTestCase):
    """The script must be able to report the premise FALSIFIED, not only
    confirmed -- otherwise it is not a test of anything."""

    def _arms(self, wall_ms, kernel_ms):
        return {"frame": _arm("frame", wall_ms=wall_ms, kernel_ms=kernel_ms)}

    def test_overhead_bound_reads_as_confirmed(self):
        text = prof._verdict(self._arms(6.4, 0.6))
        self.assertIn("OVERHEAD-BOUND confirmed", text)
        self.assertIn("TP is not", text)

    def test_kernel_bound_reads_as_falsified(self):
        text = prof._verdict(self._arms(6.4, 5.2))
        self.assertIn("PREMISE FALSIFIED", text)
        self.assertIn("revisiting", text)

    def test_the_middle_refuses_to_pick_a_side(self):
        text = prof._verdict(self._arms(6.4, 3.2))
        self.assertIn("MIXED", text)


class TestRtfProjection(CustomTestCase):
    def test_frame_cost_to_rtf(self):
        # 12 frames per audio-second: an 83.33 ms frame is exactly RTF 1.0.
        self.assertAlmostEqual(prof.project_rtf(83.3333, 12.0), 1.0, places=4)

    def test_recoverable_factor_against_the_analysis_floors(self):
        arms = {"frame": _arm("frame", wall_ms=102.5, kernel_ms=1.85)}
        out = prof.recoverable_rtf(arms, frame_hz=12.0)
        self.assertAlmostEqual(out["measured_rtf"], 1.23, places=2)
        self.assertAlmostEqual(out["kernel_only_rtf"], 0.0222, places=3)
        self.assertGreater(out["recoverable_factor"], 50.0)
        # The floors carried alongside are ANALYSE_488's, so the two models can
        # be compared in one place rather than in someone's head.
        self.assertEqual(out["bandwidth_floor_rtf_5090_from_analyse_488"], 0.022)

    def test_no_frame_arm_means_no_projection(self):
        self.assertIsNone(prof.recoverable_rtf({}))


if __name__ == "__main__":
    unittest.main()
