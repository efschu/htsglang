"""The boot-matrix arm list is well-formed data (#349).

Cheap invariants a typo in the matrix would break, checked before any GPU time
is spent driving it. Nothing here boots or checks; it reads the tuple.
"""

import unittest

from sglang.srt.boot_matrix.arms import (
    ARMS,
    BASE_EXPECT,
    Arm,
    arm_by_name,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestArmList(CustomTestCase):
    def test_names_are_unique(self):
        names = [a.name for a in ARMS]
        self.assertEqual(len(names), len(set(names)))

    def test_reject_arms_name_their_markers(self):
        for arm in ARMS:
            if arm.kind == "reject":
                self.assertTrue(
                    arm.reject_markers,
                    f"{arm.name}: a reject arm with no markers cannot tell a "
                    "clean refusal from an unrelated crash",
                )
                self.assertFalse(
                    arm.expect, f"{arm.name}: a reject arm never boots"
                )

    def test_boot_arms_declare_an_effective_config(self):
        for arm in ARMS:
            if arm.kind == "boot":
                self.assertTrue(
                    arm.expect,
                    f"{arm.name}: a boot arm with no expected config cannot "
                    "catch the #340 silent-env class",
                )

    def test_the_108_reject_surface_is_covered(self):
        """Every #108 reject reason must have an arm, or the guard it protects
        can rot unnoticed."""
        reject_axes = " ".join(a.axis for a in ARMS if a.kind == "reject")
        for needle in (
            "topk",
            "multi-layer",
            "off the weighted",
            "cross-algorithm",
            "kv-session-offload",
        ):
            self.assertIn(needle, reject_axes, f"no reject arm for {needle!r}")

    def test_the_combined_arm_exists(self):
        """Arm G -- 'all axes together' -- is the one that catches the
        #132 x weightless class, so its absence is a hole in the net."""
        g = arm_by_name("G_all_axes")
        self.assertEqual(g.kind, "boot")
        self.assertIn("132", g.catches)

    def test_bar1_arm_carries_the_capture_caveat(self):
        k = arm_by_name("K_bar1_graphs")
        self.assertTrue(k.capture_note)
        self.assertGreater(
            k.expected_seconds,
            600.0,
            "the bar1 graph arm must budget for the cold-cache capture #366 saw",
        )

    def test_base_expect_is_inherited(self):
        """A_default's declared config is the base one, unmodified."""
        a = arm_by_name("A_default")
        for key, val in BASE_EXPECT.items():
            self.assertEqual(a.expect[key], val)

    def test_arm_validation_rejects_a_markerless_reject(self):
        with self.assertRaises(ValueError):
            Arm(name="x", axis="", catches="", kind="reject")

    def test_arm_validation_rejects_a_bad_coherence_tier(self):
        with self.assertRaises(ValueError):
            Arm(name="x", axis="", catches="", kind="boot", coherence="sometimes")

    def test_arm_by_name_raises_on_unknown(self):
        with self.assertRaises(KeyError):
            arm_by_name("nope")


if __name__ == "__main__":
    unittest.main()
