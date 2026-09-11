# SPDX-License-Identifier: Apache-2.0
"""B4g wall 2: W19's reserve is PRICED on the exchange form, not borrowed.

Boot weg2xsn14 tripped ``W19 DormantResidueRefused`` at EPOCH 0 -- before the
first flip -- against a reserve built from ``DC_MEASURED_D_*``, constants
MEASURED ON A SERVING BOOT (weg2ls1b2) which therefore contain none of the
exchange lane's own device residency.  Those constants are never hand-raised
(operator ruling, and the VRAM-corridor law), so the exchange form is priced
instead: the census's per-card ``dormant_proc_used_mib`` plus the lane's own
terms, each NAMED on its own line.

THE SUBTRACTION THE OPERATOR ASKED FOR, and it REFUTED THIS SEAT'S OWN FIRST
READING -- which is why the numbers are asserted here rather than described:

    measured   2588 / 3084 / 2588      (weg2xsn14, epoch 0)
    reserved   1986 / 2292 / 1986      (serving constants + slack)
    excess      602 /  792 /  602
    census     1334 / 1668 / 1334      (the front's own uuid-keyed WEG2-DC)
    m - census 1254 / 1416 / 1254
    - named     481 = region 385 + on-card slots 3x32
    residual    773 /  935 /  773  UNEXPLAINED

The asymmetry is **162 MiB on the 5090 alone** and it SURVIVES every uniform
subtraction.  This seat had predicted the census's foreign-split bound (source
``[42, 11, 11]`` vs this form's ``[39, 13, 12]``, over-pricing stage 0 = the
5090) -- but an OVER-priced census term makes that card's residual SMALLER,
and the observed sign is the opposite.  The bound is not the explanation and
that reading is withdrawn.

So this half prices what can be named and REPORTS the rest: W19 stays a
refusal with a number attached.  A reserve widened to swallow an unexplained
773-935 MiB would be the defect the host-threshold law forbids.
"""

import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher, xchg_residency
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

BIG = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
SM1 = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
SM2 = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"

#: weg2xsn14's own epoch-0 readings, and the census's own dormant numbers.
MEASURED = {SM1: 2588, BIG: 3084, SM2: 2588}
CENSUS_DORMANT = {SM1: 1334, BIG: 1668, SM2: 1334}
SERVING_RESERVE = {SM1: 1986, BIG: 2292, SM2: 1986}

CARDS = [
    launcher.Card(1, BIG, "NVIDIA GeForce RTX 5090", 32607),
    launcher.Card(0, SM1, "NVIDIA GeForce RTX 3080", 20480),
    launcher.Card(2, SM2, "NVIDIA GeForce RTX 3080", 20480),
]
FAMILY = [f"weights_{k}" for k in range(8)] + ["weights"]


def _census(path, dormant=None, drop=()):
    dormant = dormant or CENSUS_DORMANT
    cards = {}
    for u in (BIG, SM1, SM2):
        if u in drop:
            continue
        cards[u] = {
            "tags": {g: {t: 100 for t in FAMILY} for g in ("P", "D")},
            "dormant_proc_used_mib": dormant[u],
            "dormant_source": f"READING: boot weg2sn5b front WEG2-DC peak P/D for {u[:12]}",
        }
    blob = {"cards": cards, "waves": [FAMILY], "provenance": "test"}
    with open(path, "w") as fh:
        json.dump(blob, fh)
    return path


class TheReserveIsPricedFromTheCensusAndNamedTerms(CustomTestCase):
    def _run(self, **kw):
        p = _census(os.path.join(tempfile.mkdtemp(), "c.json"), **kw)
        return launcher.xchg_form_dormant_reserve(CARDS, p, oncard_slot_mib=32)

    def test_the_reserve_is_the_census_plus_the_named_terms(self):
        out, lines, _res = self._run()
        named = launcher.xchg_resident_region_mib(len(CARDS)) + 3 * 32
        self.assertEqual(named, 481)
        for u in (BIG, SM1, SM2):
            self.assertEqual(out[u], CENSUS_DORMANT[u] + named)
        self.assertEqual(out[BIG], 2149)
        self.assertEqual(out[SM1], 1815)

    def test_it_is_SMALLER_than_the_serving_reserve_on_every_card(self):
        """AND THAT REFUTES THE SWAP AS A FIX -- the second refutation of this
        half, measured rather than argued.

        One would expect pricing the exchange form to RAISE the reserve, since
        the serving constants contain none of the lane's residency.  It does
        the opposite: the census's own dormant readings (1668 / 1334 / 1334,
        from boot weg2sn5b's front) are BELOW ``DC_MEASURED_D_*`` (2228 / 1922,
        from weg2ls1b2), so census+named comes to 2149 / 1815 / 1815 against
        the serving reserve's 2292 / 1986 / 1986.

        CONSEQUENCE: simply re-sourcing W19's reserve from the census makes the
        refusal STRICTER, not satisfiable -- the excess grows from 602/792/602
        to 773/935/773.  So the census is the right authority for the census's
        question (what a sleeping rank held on ITS boot) and the WRONG one for
        this one (what a sleeping rank holds on THE EXCHANGE FORM).  What W19
        needs is a measurement ON THIS FORM, which is the operator's second
        option, and these numbers are why it is the one that can work.
        """
        out, _l, _r = self._run()
        for u in (BIG, SM1, SM2):
            self.assertLess(out[u], SERVING_RESERVE[u], u)
        self.assertEqual(out[BIG], 2149)
        self.assertEqual(SERVING_RESERVE[BIG], 2292)

    def test_every_term_is_printed_by_name(self):
        _out, lines, _r = self._run()
        self.assertEqual(len(lines), 3)
        for ln in lines:
            self.assertIn("WEG2-XCHG-RESERVE", ln)
            self.assertIn("dormant_census_mib=", ln)
            self.assertIn(f"region_mib={launcher.xchg_resident_region_mib(len(CARDS))}", ln)
            self.assertIn("oncard_slots_mib=96", ln)
            self.assertIn("(3x32)", ln)
            # RENAMED BY B4h: this function is the PRICER and the MEASURED
            # value now sits beside it on the same line, so `priced_mib` says
            # which of the two it is.  The residual between them is printed
            # rather than left to the reader.
            self.assertIn("priced_mib=", ln)
            self.assertIn("measured_mib=", ln)
            self.assertIn("source=measured:weg2xsn14", ln)
            self.assertIn("residual_unattributed_mib=", ln)
            self.assertIn("census_source=READING", ln)

    def test_the_slot_term_follows_the_published_slot(self):
        p = _census(os.path.join(tempfile.mkdtemp(), "c.json"))
        out, lines, _r = launcher.xchg_form_dormant_reserve(
            CARDS, p, oncard_slot_mib=128)
        self.assertEqual(out[BIG], 1668 + launcher.xchg_resident_region_mib(len(CARDS)) + 3 * 128)
        self.assertTrue(any("oncard_slots_mib=384" in ln for ln in lines))

    def test_a_card_missing_from_the_census_refuses(self):
        """The serving constants must NOT stand in -- they contain none of the
        lane's residency, so borrowing them is how weg2xsn14 got its reserve."""
        with self.assertRaises(xchg_residency.Weg2XchgResidencyUnarmable) as caught:
            self._run(drop=(BIG,))
        self.assertIn("31d7ef41", str(caught.exception))
        self.assertIn("serving constants must not stand in", str(caught.exception))

    def test_the_serving_constants_are_not_hand_raised(self):
        """Operator ruling + the VRAM-corridor law: launcher.py:216 is untouched."""
        self.assertEqual(launcher.DC_MEASURED_D_5090_MIB, 2228)
        self.assertEqual(launcher.DC_MEASURED_D_3080_MIB, 1922)
        self.assertEqual(launcher.DC_RESERVE_SLACK_MIB, 64)


class TheUnexplainedResidualIsReportedNotAbsorbed(CustomTestCase):
    """The arithmetic, asserted, so the next reader inherits the refutation."""

    def test_the_named_terms_are_uniform_and_cannot_explain_the_asymmetry(self):
        named = launcher.xchg_resident_region_mib(len(CARDS)) + 3 * 32
        gaps = {u: MEASURED[u] - CENSUS_DORMANT[u] - named for u in MEASURED}
        self.assertEqual(gaps[SM1], 773)
        self.assertEqual(gaps[SM2], 773)
        self.assertEqual(gaps[BIG], 935)
        # 162 MiB on the 5090 alone, surviving every uniform subtraction
        self.assertEqual(gaps[BIG] - gaps[SM1], 162)

    def test_the_foreign_split_bound_has_the_WRONG_SIGN_for_this_gap(self):
        """This seat's first reading, refuted by its own subtraction.

        The census source ran [42,11,11] against this form's [39,13,12], which
        OVER-prices stage 0 (the 5090).  An over-priced census term makes that
        card's residual SMALLER; the 5090's residual is the LARGER one.  So the
        bound cannot be the explanation.
        """
        named = launcher.xchg_resident_region_mib(len(CARDS)) + 3 * 32
        over = 3 * 387   # three layers at the measured mean, the bound's size
        corrected = MEASURED[BIG] - (CENSUS_DORMANT[BIG] - over) - named
        self.assertGreater(corrected, MEASURED[BIG] - CENSUS_DORMANT[BIG] - named,
                           "removing an over-price must GROW the residual")

    def test_the_reserve_does_not_swallow_the_residual(self):
        """A reserve widened to make W19 pass would be the host-threshold
        law's forbidden 'accept the risk' branch."""
        p = _census(os.path.join(tempfile.mkdtemp(), "c.json"))
        out, _l, _r = launcher.xchg_form_dormant_reserve(
            CARDS, p, oncard_slot_mib=32)
        for u in (BIG, SM1, SM2):
            self.assertLess(out[u], MEASURED[u],
                            "the priced reserve must stay BELOW the measured "
                            "residue, so W19 still refuses and the gap is "
                            "reported rather than absorbed")


if __name__ == "__main__":
    unittest.main()
