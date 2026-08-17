# SPDX-License-Identifier: Apache-2.0
"""#709 -- the A/B judged without a GPU or a server.

Desk prep. The headline pin is the one that makes the whole ticket worth
running: an acceptance rule written on the END-TO-END decode round cannot
resolve the effect it is testing for, so the discriminator is the per-rank WAIT
SPREAD instead. That is asserted here with the arithmetic, not asserted in
prose.
"""

import os
import sys
import unittest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "bench", "709")
)

import acceptance as A  # noqa: E402


def _round(arm, rank, batch, ms, compute, wait, secs=12.0):
    return A.RankRound(arm, rank, f"card{rank}", batch, ms, compute, wait, secs)


def _arm(name, ratio, boot, waits, floor=0.05, digest="d0", rounds_ms=30.0, batch=1):
    return A.ArmRun(
        arm=name,
        ratio_flag=ratio,
        boot_id=boot,
        own_noise_floor=floor,
        rounds=[
            _round(name, i, batch, rounds_ms, rounds_ms - w, w)
            for i, w in enumerate(waits)
        ],
        answer_digest=digest,
    )


class TestTheEndToEndRuleCannotWork(unittest.TestCase):
    """The finding that reshaped the acceptance, pinned as arithmetic."""

    def test_the_predicted_gain_is_far_under_the_rig_floor(self):
        frac = A.PREDICTED_GAIN_MS / A.TYPICAL_BS1_ROUND_MS
        self.assertLess(frac, A.RIG_NOISE_FLOOR)
        self.assertGreater(A.RIG_NOISE_FLOOR / frac, 5.0)  # 5.4x

    def test_the_same_gain_IS_resolvable_on_the_family_slice(self):
        frac = A.PREDICTED_GAIN_MS / A.EQUAL_SHARD_MS
        self.assertGreater(frac, A.RIG_NOISE_FLOOR)
        self.assertAlmostEqual(frac, 0.311, places=2)

    def test_the_report_says_the_end_to_end_null_is_expected(self):
        a = _arm("A_equal", "None", "boot1", [0.0, 3.0, 3.0])
        b = _arm("B_proportional", "2,1,1", "boot2", [0.2, 0.4, 0.3])
        rep = A.evaluate(a, b)
        self.assertFalse(rep["secondary_not_the_discriminator"]["end_to_end_is_resolvable"])
        self.assertIn("not evidence against", rep["secondary_not_the_discriminator"]["note"])


class TestTheDiscriminator(unittest.TestCase):
    def test_a_collapsing_wait_spread_CONFIRMS(self):
        a = _arm("A_equal", "None", "boot1", [0.0, 3.0, 3.0])
        b = _arm("B_proportional", "2,1,1", "boot2", [0.2, 0.4, 0.3])
        rep = A.evaluate(a, b)
        self.assertTrue(rep["confirm"])
        self.assertEqual(rep["verdict"], "CONFIRM")

    def test_an_unchanged_spread_DECLINES_with_numbers(self):
        a = _arm("A_equal", "None", "boot1", [0.0, 3.0, 3.0])
        b = _arm("B_proportional", "2,1,1", "boot2", [0.0, 2.95, 2.95])
        rep = A.evaluate(a, b)
        self.assertFalse(rep["confirm"])
        self.assertIn("did not clear the per-boot floor", rep["verdict"])

    def test_a_gain_under_the_floor_is_not_a_win(self):
        a = _arm("A_equal", "None", "boot1", [0.0, 3.0, 3.0], floor=0.30)
        b = _arm("B_proportional", "2,1,1", "boot2", [0.0, 2.4, 2.4], floor=0.30)
        self.assertFalse(A.evaluate(a, b)["confirm"])  # 20% gain, 30% floor


class TestCoherenceIsAGate(unittest.TestCase):
    def test_a_changed_answer_VOIDS_a_speed_win(self):
        a = _arm("A_equal", "None", "boot1", [0.0, 3.0, 3.0], digest="d0")
        b = _arm("B_proportional", "2,1,1", "boot2", [0.1, 0.1, 0.1], digest="CHANGED")
        rep = A.evaluate(a, b)
        self.assertFalse(rep["confirm"])
        self.assertIn("lossless", rep["verdict"])


class TestTheFloorIsPerBoot(unittest.TestCase):
    def test_the_LARGER_of_the_two_boot_floors_is_used(self):
        a = _arm("A_equal", "None", "boot1", [0.0, 3.0, 3.0], floor=0.05)
        b = _arm("B_proportional", "2,1,1", "boot2", [0.2, 0.4, 0.3], floor=0.22)
        self.assertAlmostEqual(A.evaluate(a, b)["floor_used"], 0.22)

    def test_an_arm_without_its_own_floor_is_refused(self):
        a = _arm("A_equal", "None", "boot1", [0.0, 3.0], floor=0.0)
        with self.assertRaises(A.AcceptanceError) as e:
            A.validate(a)
        self.assertIn("per boot", str(e.exception))

    def test_two_arms_from_ONE_boot_are_refused(self):
        a = _arm("A_equal", "None", "boot1", [0.0, 3.0])
        b = _arm("B_proportional", "2,1,1", "boot1", [0.1, 0.2])
        with self.assertRaises(A.AcceptanceError) as e:
            A.evaluate(a, b)
        self.assertIn("boot flag", str(e.exception))

    def test_short_runs_are_refused(self):
        a = _arm("A_equal", "None", "boot1", [0.0, 3.0])
        a = A.ArmRun(a.arm, a.ratio_flag, a.boot_id, a.own_noise_floor,
                     [A.RankRound("A", 0, "c", 1, 30.0, 27.0, 3.0, 2.0)], a.answer_digest)
        with self.assertRaises(A.AcceptanceError) as e:
            A.validate(a)
        self.assertIn("bounds runs by TIME", str(e.exception))

    def test_an_unrecorded_ratio_flag_is_refused(self):
        a = _arm("A_equal", "", "boot1", [0.0, 3.0])
        with self.assertRaises(A.AcceptanceError) as e:
            A.validate(a)
        self.assertIn("CAPACITY-first", str(e.exception))


class TestTheRatioVectorIsConstrained(unittest.TestCase):
    """'Just enable proportional' is not a thing: sum(weights) must divide
    every sharded dimension, and 'auto' is capacity-first, not speed."""

    def test_admissible_vectors_respect_divisibility(self):
        got = A.admissible_ratios([5120, 4096], n_ranks=3, max_sum=16)
        self.assertTrue(got)
        for v in got:
            self.assertEqual(5120 % sum(v), 0)
            self.assertEqual(4096 % sum(v), 0)

    def test_an_indivisible_sum_is_excluded(self):
        got = A.admissible_ratios([5120], n_ranks=3, max_sum=16)
        self.assertNotIn((5, 2, 2), got)  # sum 9 does not divide 5120

    def test_the_practical_vector_FALLS_SHORT_of_the_bandwidth_ideal(self):
        """2,1,1 is ratio 2.0 against an ideal 2.36, so it cannot deliver the
        full +0.780 -- the run must not be judged as though it should."""
        self.assertAlmostEqual(A.ideal_bandwidth_ratio(), 2.355, places=2)
        short = A.ratio_shortfall((2, 1, 1))
        self.assertGreater(short, 0.10)
        self.assertLess(short, 0.20)

    def test_no_sharded_dims_is_refused_rather_than_guessed(self):
        with self.assertRaises(A.AcceptanceError):
            A.admissible_ratios([])


class TestTheRenderNamesBothMetrics(unittest.TestCase):
    def test_it_labels_the_secondary_as_not_the_discriminator(self):
        a = _arm("A_equal", "None", "boot1", [0.0, 3.0, 3.0])
        b = _arm("B_proportional", "2,1,1", "boot2", [0.2, 0.4, 0.3])
        text = A.render(A.evaluate(a, b))
        self.assertIn("NOT the discriminator", text)
        self.assertIn("PRIMARY", text)
        self.assertIn("coherence clean", text)


if __name__ == "__main__":
    unittest.main()
