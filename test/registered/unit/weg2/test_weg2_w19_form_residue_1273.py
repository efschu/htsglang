# SPDX-License-Identifier: Apache-2.0
"""B4h: W19's dormant reserve is FORM-KEYED, following the launcher's own precedent.

`launcher.py`'s NCCL arm already does exactly this, and says why::

    MEASURED (A/B boot weg2ab0, ARM 0): D's dormant residue on the 5090 rises
    2230 -> 2310 MiB under NCCL, because libnccl's buffers are not
    memory-saver-tagged and therefore survive the sleep. ... The extra slack
    lives HERE, behind the switch, and not in the constant.

The exchange arm gets the same treatment one level MORE honest: not a slack
added to a serving number, but the whole per-card measurement, because boot
weg2xsn14 produced it -- 2588 / 3084 / 2588 MiB at EPOCH 0, where the serving
reserve (1986 / 2292 / 1986) tripped W19 DormantResidueRefused.

WHAT THIS TEST GUARDS, and the second is the danger direction:

* with the xchg constant, XSN14's own numbers PASS W19; with the serving
  constant they FAIL.  Both asserted, on the measured triple.
* the serving path is BYTE-IDENTICAL -- `DC_MEASURED_D_*` untouched and the
  serving reserve still 2292 / 1986 / 1986.  The xchg value leaking into the
  serving path would shrink every serving boot's P budget by 666-856 MiB per
  card for residency that form does not have.

ONE CONSUMER, FORM-SWITCHED (upstream-minimal law): whatever reads
`DC_MEASURED_D_*` today reads the xchg value under the arm.  No parallel
reserve object, no second bookkeeping -- which is also why P's per-card budget
follows automatically.

Hermetic: pure selector + AST wiring pins.  No NVML, no boot.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

BIG = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
SM1 = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
SM2 = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"

CARDS = [
    launcher.Card(1, BIG, "NVIDIA GeForce RTX 5090", 32607),
    launcher.Card(0, SM1, "NVIDIA GeForce RTX 3080", 20480),
    launcher.Card(2, SM2, "NVIDIA GeForce RTX 3080", 20480),
]
#: weg2xsn14's W19 line at epoch 0, nvml order 0/1/2.
XSN14_MEASURED = {SM1: 2588, BIG: 3084, SM2: 2588}


class TheMeasuredTripleIsTheConstant(CustomTestCase):
    def test_the_triple_is_the_boots_own_numbers(self):
        self.assertEqual(launcher.DC_MEASURED_D_XCHG_MIB, (2588, 3084, 2588))

    def test_it_is_keyed_by_CARD_IDENTITY_not_by_index(self):
        """The same keying `DC_MEASURED_D_*` uses -- the card's NAME.  An index
        is not an identity on this rig (NVML enumeration is not stable across
        boots, #589), and the two 3080s measured IDENTICALLY at 2588, so the
        name is exact rather than a convenient collapse."""
        self.assertEqual(launcher.DC_MEASURED_D_XCHG_5090_MIB, 3084)
        self.assertEqual(launcher.DC_MEASURED_D_XCHG_3080_MIB, 2588)
        self.assertEqual(launcher.DC_MEASURED_D_XCHG_MIB[0],
                         launcher.DC_MEASURED_D_XCHG_MIB[2])

    def test_the_selector_answers_the_xchg_value_under_the_arm(self):
        for arm in ("exchange",):
            for c in CARDS:
                self.assertEqual(launcher.dc_measured_d_mib(c, arm),
                                 XSN14_MEASURED[c.uuid], f"{arm}/{c.name}")

    def test_ANY_inject_mode_selects_it_because_that_is_where_it_was_measured(self):
        """The residency is present under `shadow` -- weg2xsn14 measured it
        there -- so the selector keys on the WEIGHT SOURCE alone.  This is the
        one place in this campaign where the inject arm does NOT enter a
        predicate, and B4f's conjunction is about a different question (who
        owns the bytes), not about what is resident."""
        import inspect

        src = inspect.getsource(launcher.dc_measured_d_mib)
        self.assertNotIn("inject", src)

    def test_the_serving_path_is_byte_identical(self):
        for arm in ("ring", "shadow"):
            self.assertEqual(launcher.dc_measured_d_mib(CARDS[0], arm),
                             launcher.DC_MEASURED_D_5090_MIB, arm)
            self.assertEqual(launcher.dc_measured_d_mib(CARDS[1], arm),
                             launcher.DC_MEASURED_D_3080_MIB, arm)

    def test_the_serving_constants_are_untouched(self):
        self.assertEqual(launcher.DC_MEASURED_D_5090_MIB, 2228)
        self.assertEqual(launcher.DC_MEASURED_D_3080_MIB, 1922)
        self.assertEqual(launcher.DC_RESERVE_SLACK_MIB, 64)

    def test_an_unknown_board_still_refuses_to_be_guessed(self):
        odd = launcher.Card(3, "GPU-cccc", "NVIDIA GeForce GTX 780", 3072)
        with self.assertRaises(launcher.Weg2LaunchRefused):
            launcher.dc_measured_d_mib(odd, "exchange")


class W19PassesOnTheFormAndFailsOnTheServingConstant(CustomTestCase):
    """Red-first, on XSN14's own numbers.  W19 fires when measured > reserve."""

    def _reserve(self, arm):
        slack = launcher.reserve_slack_mib("bar1")
        return {c.uuid: launcher.dc_measured_d_mib(c, arm) + slack for c in CARDS}

    def test_with_the_xchg_constant_xsn14s_numbers_PASS(self):
        resv = self._reserve("exchange")
        for u, meas in XSN14_MEASURED.items():
            self.assertLessEqual(meas, resv[u], f"{u} would trip W19")
        self.assertEqual(resv[BIG], 3084 + 64)

    def test_with_the_serving_constant_they_FAIL(self):
        """The regression weg2xsn14 hit, asserted so it cannot come back."""
        resv = self._reserve("ring")
        over = {u: m - resv[u] for u, m in XSN14_MEASURED.items() if m > resv[u]}
        self.assertEqual(len(over), 3, over)
        self.assertEqual(resv[BIG], 2292)
        self.assertEqual(resv[SM1], 1986)

    def test_the_serving_reserve_triple_is_pinned(self):
        resv = self._reserve("ring")
        self.assertEqual(sorted(resv.values()), [1986, 1986, 2292])


class OneConsumerFormSwitched(CustomTestCase):
    def _main_kwargs(self, callee):
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(launcher.main)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                if name == callee:
                    return {k.arg: ast.unparse(k.value) for k in node.keywords if k.arg}
        return None

    def test_main_builds_dc_expect_d_through_the_selector(self):
        """Structural: the ONE consumer, so P's budget follows automatically
        and no parallel reserve object exists."""
        import ast
        import inspect
        import textwrap

        src = textwrap.dedent(inspect.getsource(launcher.main))
        tree = ast.parse(src)
        found = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", "") == "dc_expect_d" for t in node.targets):
                found = ast.unparse(node.value)
                break
        self.assertIsNotNone(found, "main no longer assigns dc_expect_d")
        self.assertIn("dc_measured_d_mib(c, ns.weg2_weight_source)", found)
        # and the serving constants are no longer selected inline there
        self.assertNotIn("DC_MEASURED_D_5090_MIB", found)

    def test_there_is_exactly_one_selector_call_site_in_main(self):
        import inspect

        src = inspect.getsource(launcher.main)
        self.assertEqual(src.count("dc_measured_d_mib("), 1)


if __name__ == "__main__":
    unittest.main()


class TheExcessIsAttributedOnOneInstrument(CustomTestCase):
    """B4i REPLACES B4h's ``TheResidualIsPrintedNotAbsorbed``, and the
    replacement is a RETRACTION, not a rename.

    B4h printed ``residual_unattributed_mib=`` 773 / 935 / 773 beside the
    measurement.  That number does not exist: it came from subtracting
    ``/dev/shm`` byte counts (the staging region 385 + the on-card slots 96)
    from a per-process NVML reading that never contained them, so the
    subtraction produced the gap exactly (1254-481, 1416-481).  On ONE
    instrument the excess is the RESIDENT ``weights_draft`` tag, to within the
    instrument margin, and the field is gone rather than re-derived.
    """

    def test_the_line_carries_the_reserve_and_the_other_reading_of_ONE_instrument(self):
        import json
        import tempfile

        family = [f"weights_{k}" for k in range(8)] + ["weights"]
        dormant = {SM1: 1334, BIG: 1668, SM2: 1334}
        blob = {"cards": {u: {"tags": {g: {t: 100 for t in family} for g in ("P", "D")},
                              "dormant_proc_used_mib": dormant[u],
                              "dormant_source": "READING: boot weg2sn5b front WEG2-DC"}
                          for u in (BIG, SM1, SM2)},
                "waves": [family], "provenance": "test"}
        path = os.path.join(tempfile.mkdtemp(), "c.json")
        with open(path, "w") as fh:
            json.dump(blob, fh)
        out, lines = launcher.xchg_form_dormant_reserve(CARDS, path)
        self.assertEqual(out[BIG], 3084)
        self.assertEqual(out[SM1], 2588)
        big = next(ln for ln in lines if "5090" in ln)
        self.assertIn("reserved_mib=3084", big)
        self.assertIn("measured_mib=1668", big)
        self.assertIn("instrument=WEG2-DC-at-sleep", big)
        self.assertIn("delta_mib=1416", big)
        # RETRACTED, and pinned so a copy-paste cannot bring it back.
        self.assertNotIn("residual_unattributed_mib=", big)
        self.assertNotIn("priced_mib=", big)
        self.assertNotIn("region_mib=", big)

    def test_the_host_side_terms_are_deleted_from_the_module(self):
        """The two B4g helpers existed only for the retracted subtraction; the
        region's size is still derived where it is USED (prepare_region)."""
        self.assertFalse(hasattr(launcher, "XCHG_RESIDENT_ONCARD_SLOTS"))
        self.assertFalse(hasattr(launcher, "xchg_resident_region_mib"))

    def test_the_constant_says_it_is_not_final_and_names_its_attribution(self):
        """Nobody may read the triple as the answer -- and nobody may read the
        old residual as an open question either."""
        import inspect

        src = inspect.getsource(launcher)
        i = src.index("DC_MEASURED_D_XCHG_MIB = (")
        head = src[max(0, i - 2600):i]
        self.assertIn("RETRACTED", head)
        self.assertIn("weights_draft", head)
        self.assertIn("B4k", head)
        self.assertIn("PREFLIGHT MEASUREMENT REPLACES THIS", head)
        self.assertNotIn("UNATTRIBUTED:", head)
