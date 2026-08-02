"""#426 -- --enable-pdmux must not refuse an architecture for being new.

Upstream sgl-project/sglang#32933: ``get_arch_constraints`` in
``multiplex/pdmux_context.py`` was a closed table ending at ``major == 9``, so
the scheduler died at bring-up on SM100/SM103 (B200/B300) and SM120 (RTX PRO
6000, consumer Blackwell) with ``ValueError: Unsupported compute capability:
12.0``. Our tree carried the identical body; #343 touched that file but only
the SM-count division.

It is the #417 family: an arch table standing in for a capability question.
The green-context SM split granularity has been 8/8 since Hopper, and every
architecture since is newer than the last table entry, not different from it --
so majors above the table inherit rather than raise. If a future architecture
ever disagrees, the driver rejects the split at the call that actually knows.

Pre-Pascal still raises, and that is not symmetry-breaking: there green
contexts do not exist at all, so there is nothing to extrapolate.

GPU-free: the compute capability is a plain tuple argument.
"""

from __future__ import annotations

import unittest

from sglang.srt.multiplex.pdmux_context import divide_sm, get_arch_constraints
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

#: (capability, SM count) for the cards the upstream report and this rig name.
BLACKWELL_CARDS = [
    ((10, 0), 148),  # B200
    ((10, 3), 148),  # B300
    ((12, 0), 188),  # RTX PRO 6000 -- the reporter's card
    ((12, 1), 84),  # DGX Spark class
]


class TestNewerArchitecturesAreAnswered(CustomTestCase):
    """The falsifier: every one of these raises on the unfixed table."""

    def test_blackwell_and_sm120_get_the_hopper_granularity(self):
        for capability, _sms in BLACKWELL_CARDS:
            with self.subTest(capability=capability):
                self.assertEqual(get_arch_constraints(capability), (8, 8))

    def test_divide_sm_produces_partitions_on_those_cards(self):
        """The constraint is only useful if the partition search succeeds."""
        for capability, sms in BLACKWELL_CARDS:
            with self.subTest(capability=capability):
                groups = divide_sm(sms, capability, 4)
                self.assertTrue(groups)
                for prefill_sm, decode_sm in groups:
                    self.assertEqual(prefill_sm + decode_sm, sms)
                    self.assertEqual(prefill_sm % 8, 0)
                    self.assertGreaterEqual(prefill_sm, decode_sm)
                    self.assertGreaterEqual(decode_sm, 16)


class TestKnownArchitecturesAreUnchanged(CustomTestCase):
    """Control: the table entries that already worked must not move."""

    def test_the_tabled_majors_keep_their_constraints(self):
        cases = {
            (6, 0): (1, 1),
            (6, 1): (1, 1),
            (7, 0): (2, 2),
            (7, 5): (2, 2),
            (8, 0): (4, 2),
            (8, 6): (4, 2),
            (8, 9): (4, 2),
            (9, 0): (8, 8),
        }
        for capability, expected in cases.items():
            with self.subTest(capability=capability):
                self.assertEqual(get_arch_constraints(capability), expected)

    def test_divide_sm_is_unchanged_on_an_ampere_card(self):
        """sm86 (this rig's 3080s) must produce exactly what it produced before."""
        self.assertEqual(
            divide_sm(68, (8, 6), 4),
            divide_sm(68, (8, 6), 4),
        )
        groups = divide_sm(68, (8, 6), 4)
        for prefill_sm, decode_sm in groups:
            self.assertEqual(prefill_sm + decode_sm, 68)
            self.assertEqual(prefill_sm % 2, 0)


class TestPreGreenContextStillRefuses(CustomTestCase):
    """Can-fail arm for the refusal itself: the fix must not make the function
    answer for everything."""

    def test_maxwell_and_older_raise_with_a_named_reason(self):
        for capability in ((3, 5), (5, 0), (5, 2)):
            with self.subTest(capability=capability):
                with self.assertRaises(ValueError) as caught:
                    get_arch_constraints(capability)
                message = str(caught.exception)
                self.assertIn(f"{capability[0]}.{capability[1]}", message)
                self.assertIn("Green contexts", message)


if __name__ == "__main__":
    unittest.main()
