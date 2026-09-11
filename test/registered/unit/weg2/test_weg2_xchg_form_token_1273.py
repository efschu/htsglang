# SPDX-License-Identifier: Apache-2.0
"""B4e: the launcher hands solve() the XCHG_FORM_TOKEN when the arm is armed.

``ring_table`` wrote the requirement itself, at the name it defined for it::

    THIS tree's launcher never arms xchg, so THIS boot never carries the token
    and an xchg source is never same-form for it.  When the xchg slice lands on
    the line its launcher must add the same token to the argv it hands
    ``solve`` (one line, ``p_argv + [XCHG_FORM_TOKEN]``), or its own boots will
    rank a serving source as their form's twin -- the mirror of this defect.

The slice has landed and the launcher did not add the token, so #1305 item 4
EXCLUDED every boot of this form from the ring table's own selection -- the
exclusion is written for a boot that does not arm the exchange, and it was
being applied to one that does.  MEASURED consequence, 2026-09-11: the census
B4c produced took its group-P per-card rows from a boot that ran the PP layer
split ``[42, 11, 11]`` while the armed form runs ``[39, 13, 12]``, because
every boot that ran the armed split carries the marker and was excluded.

TWO HALVES, and the second is why this is not a one-liner:

* the token goes on, under EXACTLY the condition that makes a future reader
  classify this boot as an xchg source -- i.e. ``WEIGHT_SOURCE_ARMED``, the
  same derived tuple that decides whether the region is published at all, so
  the marker this boot emits and the token it claims cannot disagree;
* and with the token on, ``solve`` no longer EXCLUDES the xchg sources, so if
  none exists it falls through to the distance ranking and picks a boot of
  another form -- silently, which is the foreign-split defect one layer down.
  So the launcher REFUSES by name and PRINTS THE CLOSEST TWIN it rejected.

Hermetic: pure functions over a fake table, no NVML, no evidence tree.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher, ring_table
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

BASE_ARGV = ["/venv/bin/python", "-m", "sglang.launch_server",
             "--model-path", "/m", "--pp-size", "3", "--pp-stage-ratio", "39,13,12"]


class FakeTable:
    def __init__(self, form_same, boot="boot_weg2_weg2sn5b_x_0909_185632",
                 source_form_key="e3fe683c5d40", form_key="f003de5bc71a",
                 form_diff="THIS boot has ['--weg2-xchg-region=armed'] that the source has not"):
        self.form_same = form_same
        self.boot = boot
        self.source_form_key = source_form_key
        self.form_key = form_key
        self.form_diff = form_diff


class TheTokenGoesOnExactlyWhenTheArmIsArmed(CustomTestCase):
    def test_every_armed_arm_carries_it(self):
        self.assertEqual(launcher.WEIGHT_SOURCE_ARMED, ("exchange", "shadow"))
        for arm in launcher.WEIGHT_SOURCE_ARMED:
            out = launcher.xchg_form_argv(BASE_ARGV, arm)
            self.assertEqual(out[-1], ring_table.XCHG_FORM_TOKEN, arm)
            self.assertEqual(out[:-1], BASE_ARGV, arm)

    def test_the_default_arm_is_byte_identical(self):
        """Acceptance (iii): the serving form's selection must not move."""
        out = launcher.xchg_form_argv(BASE_ARGV, launcher.WEIGHT_SOURCE_DEFAULT)
        self.assertEqual(out, BASE_ARGV)
        self.assertNotIn(ring_table.XCHG_FORM_TOKEN, out)

    def test_an_unknown_arm_adds_nothing(self):
        self.assertEqual(launcher.xchg_form_argv(BASE_ARGV, "nonsense"), BASE_ARGV)

    def test_the_input_is_not_mutated(self):
        argv = list(BASE_ARGV)
        launcher.xchg_form_argv(argv, "exchange")
        self.assertEqual(argv, BASE_ARGV)

    def test_the_token_moves_the_form_key(self):
        """The whole point: the key must stop matching the serving twin's."""
        serving = ring_table.p_form_key(launcher.xchg_form_argv(BASE_ARGV, "ring"))[0]
        armed = ring_table.p_form_key(launcher.xchg_form_argv(BASE_ARGV, "exchange"))[0]
        self.assertNotEqual(serving, armed)

    def test_the_token_is_visible_in_the_printed_form(self):
        """'Provenance line prints the token' -- the FORM string is that line."""
        _key, norm = ring_table.p_form_key(launcher.xchg_form_argv(BASE_ARGV, "exchange"))
        self.assertIn(ring_table.XCHG_FORM_TOKEN, norm)


class WithoutASameFormSourceTheLaunchRefuses(CustomTestCase):
    def test_it_reuses_W48_rather_than_minting_a_code(self):
        """NO NEW W-CODE, and that is a judgement worth stating.

        W48 ``Weg2RingFormMismatch`` IS this event: "form key X != source Y".
        The launcher already raises that class for its OWN un-priceable argv
        (``ring_table.py``), it is a ``Weg2RingRefused`` and therefore already
        in ``REFUSALS``, and the W-code census keys on (code, NAME) pairs -- so
        a second name under W48 would be a fresh collision, and a fresh number
        for a form mismatch would be a second spelling of one event.  The only
        genuinely free numbers on this tip are W5, W39 and W90 (the census also
        NAMES W19 free and it is not -- ``W19 DormantResidueRefused`` lives in
        front.py under a non-``Weg2`` name the scan cannot see), and B4d needs
        one of them for a disagreement that is NOT a form mismatch.
        """
        self.assertIs(launcher.Weg2XchgFormSourceMissing,
                      ring_table.Weg2RingFormMismatch)
        self.assertTrue(issubclass(ring_table.Weg2RingFormMismatch,
                                   ring_table.Weg2RingRefused))

    def test_a_same_form_source_passes(self):
        self.assertIsNone(
            launcher.refuse_unless_same_form_source(FakeTable(True), "exchange"))

    def test_another_form_refuses_by_name_and_prints_the_closest_twin(self):
        for arm in launcher.WEIGHT_SOURCE_ARMED:
            with self.assertRaises(launcher.Weg2XchgFormSourceMissing) as caught:
                launcher.refuse_unless_same_form_source(FakeTable(False), arm)
            msg = str(caught.exception)
            self.assertIn("W48 Weg2RingFormMismatch", msg)
            self.assertIn("weg2sn5b", msg)            # the closest twin, printed
            self.assertIn("e3fe683c5d40", msg)        # its form key
            self.assertIn("f003de5bc71a", msg)        # and ours
            self.assertIn("--weg2-xchg-region=armed", msg)   # the form_diff

    def test_the_refusal_is_in_the_launchers_own_funnel(self):
        """Exit 2 and one named line, not a traceback (#1275 fix 2)."""
        self.assertTrue(issubclass(launcher.Weg2XchgFormSourceMissing, launcher.REFUSALS))

    def test_the_default_arm_never_refuses(self):
        """Acceptance (iii) again: a serving boot has no same-form requirement."""
        self.assertIsNone(
            launcher.refuse_unless_same_form_source(
                FakeTable(False), launcher.WEIGHT_SOURCE_DEFAULT))

    def test_no_table_is_someone_elses_refusal(self):
        """prepare_host_ring already refuses R22/W20; do not double-refuse."""
        self.assertIsNone(launcher.refuse_unless_same_form_source(None, "exchange"))

    def test_an_unarmed_form_gate_is_not_read_as_a_pass(self):
        """``form_same is None`` means the gate never ran -- that is not 'same'."""
        with self.assertRaises(launcher.Weg2XchgFormSourceMissing):
            launcher.refuse_unless_same_form_source(FakeTable(None), "exchange")


class BothHalvesAreWiredIntoMain(CustomTestCase):
    """A pin, because a pure function nothing calls is a pure function.

    Asserted over the NORMALISED source of ``main`` (whitespace collapsed), so
    a line wrap cannot break it -- that text-scan trap has bitten this campaign
    four times.
    """

    def _main_src(self):
        import inspect
        import re

        return re.sub(r"\s+", " ", inspect.getsource(launcher.main))

    def test_main_builds_the_form_argv_through_the_helper(self):
        self.assertIn("form_argv_p = xchg_form_argv(", self._main_src())

    def test_main_checks_the_source_form_after_the_ring_is_solved(self):
        src = self._main_src()
        self.assertIn("refuse_unless_same_form_source(ring_plan.table,", src)
        self.assertLess(src.index("prepare_host_ring("),
                        src.index("refuse_unless_same_form_source("))


if __name__ == "__main__":
    unittest.main()
