# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# ==============================================================================
"""#1299: X's SEED was selected by a position cap, so a live measurement aged out.

MEASURED, two boots of the same lineage, same rig, same day:

    dec1  22d68d0839  X = 8,742   source=boot:...weg2sb5f...front.log
                                  flip_s=2.90 (n76) r_D=1138 (n17) r_P=4628 (n38)
    dec2b f929987a9c  X = 22,556  source=recorded PRE-BARLINK
                                  flip_s=13.247   r_D=690     r_P=3640

The X derivation path is BYTE-IDENTICAL between the two trees (launcher.py
lines 420-640 diff clean across `git diff 22d68d0839..f929987a9c`), and no
term was re-priced: all three seeds moved AT ONCE because the whole tuple
switched source, from this rig's own measurement to the hardcoded fallback
pair.  What changed was the DIRECTORY.  `resolve_x` scanned `logs[:8]` -- the
eight newest `.front.log` files by mtime -- and a NON-qualifying log consumed
a slot exactly like a qualifying one.  Between the two boots nine newer front
logs accumulated (dec1's own, three refused dec2/dec3 attempts, two sb5h, one
sb5g, dec2b's own), every one of them from a boot that never drained P, so
every one parsed to ``None``.  The measurement was still on disk, still valid,
one slot past the cap.

Replayed against the real evidence tree at dec2b's launch mtime, the scan the
launcher actually ran:

    0..7  IN logs[:8]      all None (barren: refused or decode-only boots)
    8     OUT of logs[:8]  X=8742   sb5f      <- the seed dec1 used
    10    OUT of logs[:8]  X=12944  sb5e
    11    OUT of logs[:8]  X=10053  sb5d
    12    OUT of logs[:8]  X=9926   sb5c
          fallback (recorded PRE-BARLINK) X=22556

FOUR measured logs sat past the cap and every one of them prices X in the
8.7k-12.9k band.  Not one measurement on this rig produces anything near the
22,556 the fallback asserts -- so the tripling is a SEED-SELECTION DEFECT, not
a rig that got slower.

Downstream it closed the carrier interval: `route_floor` prices the floor at
1.25x X (carrier_census.py:274-304 -- CHARS_PER_TOKEN 3.0 over
CARRIER_CHARS_PER_TOKEN 2.4, a principled ratio, not a hand number), the
carrier bound is 0.9x D's hicache host size (CARRIER_PREFETCH_FRACTION,
launcher.py:3365), and 1.25 x 22,556 = 28,195 > 27,466 = 0.9 x 30,518.  W45
refused the boot pre-READY, correctly: the launcher was right, its seed was
not.

THE FIX, and why this shape: the same launcher already scans `.P.log` files
for two other derived numbers (`newest_p_log_with_bubble` at launcher.py:4776
and the mean-prefix scan at :4885) and BOTH walk the whole sorted list,
skipping non-qualifying logs until one qualifies -- their own docstrings say
"such a log is skipped and an older one that carries the line is preferred".
`resolve_x` was the odd one out.  Removing the cap makes it match its own
siblings and removes a hand number rather than adding one.

NOTE ON THE TWO X VALUES IN THIS FILE.  The hermetic fixtures below derive
**8752**, not 8742: the provenance line prints r_D and r_P rounded to whole
tok/s, so synthetic logs built from the PRINTED 1138/4628 cannot reproduce the
unrounded medians the real log carries.  8742 is asserted only against the
real evidence tree, where the unrounded values live.
"""

import os
import tempfile
import unittest

from sglang.srt.weg2.launcher import (
    X_RECORDED_FLIP_S,
    X_RECORDED_R_D_TOKS,
    X_RECORDED_R_P_TOKS,
    derive_x_star,
    resolve_x,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

FLOOR = 4096

#: The dec1 seed, as the provenance line printed it.
DEC1_FLIP_S, DEC1_R_D, DEC1_R_P = 2.90, 1138.0, 4628.0
#: What those PRINTED (rounded) seeds derive -- see the note in the docstring.
DEC1_X_FROM_PRINTED_SEEDS = 8752
#: What the real sb5f log's unrounded medians derived on the dec1 boot.
DEC1_X_ON_METAL = 8742
#: The dec2b value: the hardcoded fallback, reached because the scan gave up.
DEC2B_X = 22556

SB5F = ("/spinning/evidence-665-f1/"
        "boot_weg2_weg2sb5f_4f762260ba_0909_050638.front.log")


def _qualifying(flip_s=DEC1_FLIP_S, r_d=DEC1_R_D, r_p=DEC1_R_P) -> str:
    """A front log carrying all three instruments, in the front's own shapes."""
    return "\n".join([
        f"[t] INFO weg2.front: WEG2-FLIP epoch=1 flip_total={int(flip_s * 1000)} ms",
        "[t] INFO weg2.front: WEG2-SERVED group=D leg=2 rid=a "
        f"uncached={int(r_d * 10)} verdict=single_prefill wall=10.0s",
        "[t] INFO weg2.front: WEG2-SERVED group=P leg=1 rid=b "
        f"prompt_tokens={int(r_p * 10)} cached_tokens=0 wall=1.0s",
        "[t] INFO weg2.front: WEG2 P-DRAIN epoch=1 prefilled=8 drain_s=10.0",
        "",
    ])


def _barren() -> str:
    """A refused or decode-only boot: a front log, no complete instrument set.

    This is the real shape of all eight logs that filled the cap at dec2b --
    the front came up and printed, but P never drained, so there is no r_P.
    """
    return "\n".join([
        "[t] INFO weg2.front: WEG2-FRONT up tag=refused awake=D",
        "[t] INFO weg2.front: WEG2-FLIP epoch=1 flip_total=2900 ms",
        "[t] INFO weg2.front: WEG2 X-ROUTE rid=r-1 est_uncached=124 X=8742",
        "",
    ])


class _Dir:
    """A temp evidence dir whose front logs have a controlled mtime ORDER."""

    def __init__(self, test):
        self.tmp = tempfile.TemporaryDirectory()
        test.addCleanup(self.tmp.cleanup)
        self._t = 1_700_000_000

    def add(self, name: str, body: str) -> str:
        """Files are added OLDEST FIRST; each new one is newer than the last."""
        path = os.path.join(self.tmp.name, f"boot_{name}.front.log")
        with open(path, "w") as f:
            f.write(body)
        self._t += 60
        os.utime(path, (self._t, self._t))
        return path

    @property
    def path(self) -> str:
        return self.tmp.name


class TheScanMustNotGiveUpOnAMeasurementItStillHas(CustomTestCase):
    """RED-FIRST: the exact dec2b directory shape, in miniature."""

    def test_eight_barren_logs_must_not_bury_the_ninth(self):
        """The shipped defect: eight non-qualifying logs consumed the whole
        scan budget and the launcher fell through to the PRE-BARLINK pair,
        with a valid measurement one slot past the cap."""
        d = _Dir(self)
        d.add("sb5f", _qualifying())          # oldest, and the only measurement
        for i in range(8):                    # eight newer, all barren
            d.add(f"refused{i}", _barren())

        seed = resolve_x(None, d.path, FLOOR)

        self.assertEqual(seed.tokens, DEC1_X_FROM_PRINTED_SEEDS)
        self.assertTrue(seed.measured)
        self.assertIn("source=boot:", seed.provenance)
        self.assertIn("sb5f", seed.provenance)
        self.assertNotEqual(seed.tokens, DEC2B_X, "the value the cap produced")
        self.assertNotIn("PRE-BARLINK", seed.provenance)

    def test_a_barren_log_between_two_measurements_does_not_stop_the_scan(self):
        """A gap is a skip, never a stop."""
        d = _Dir(self)
        d.add("older", _qualifying(r_d=1000.0))
        d.add("gap", _barren())
        seed = resolve_x(None, d.path, FLOOR)
        self.assertIn("source=boot:", seed.provenance)
        self.assertIn("older", seed.provenance)
        self.assertGreater(seed.tokens, FLOOR)

    def test_the_newest_qualifying_log_still_wins(self):
        """Removing the cap must not change the ORDER: newest measurement wins."""
        d = _Dir(self)
        d.add("old", _qualifying(flip_s=5.80))     # would derive ~2x
        d.add("barren", _barren())
        d.add("new", _qualifying(flip_s=2.90))
        seed = resolve_x(None, d.path, FLOOR)
        self.assertIn("new", seed.provenance)
        self.assertEqual(seed.tokens, DEC1_X_FROM_PRINTED_SEEDS)


class TheProvenanceCarriesItsDenominator(CustomTestCase):
    """DENOMINATOR LAW: a seed chosen by a scan must say what the scan saw."""

    def test_the_boot_source_names_how_many_logs_it_skipped(self):
        d = _Dir(self)
        d.add("sb5f", _qualifying())
        for i in range(8):
            d.add(f"refused{i}", _barren())
        self.assertIn("skipped 8", resolve_x(None, d.path, FLOOR).provenance)

    def test_a_first_hit_reports_zero_skipped_rather_than_staying_silent(self):
        d = _Dir(self)
        d.add("sb5f", _qualifying())
        self.assertIn("skipped 0", resolve_x(None, d.path, FLOOR).provenance)

    def test_the_fallback_names_how_many_it_examined(self):
        """When NOTHING qualifies the fallback is still right -- but it must
        say so with a count, so a reader can tell 'no measurement exists' from
        'the scan stopped early', which is precisely what dec2b could not."""
        d = _Dir(self)
        for i in range(9):
            d.add(f"refused{i}", _barren())
        seed = resolve_x(None, d.path, FLOOR)
        self.assertEqual(seed.tokens, DEC2B_X)
        self.assertFalse(seed.measured, "a table must declare itself unmeasured")
        self.assertIn("PRE-BARLINK", seed.provenance)
        self.assertIn("examined 9", seed.provenance)

    def test_an_empty_evidence_dir_reports_zero_examined(self):
        d = _Dir(self)
        seed = resolve_x(None, d.path, FLOOR)
        self.assertEqual(seed.tokens, DEC2B_X)
        self.assertFalse(seed.measured)
        self.assertIn("examined 0", seed.provenance)


class TheTwoFixtureNumbersArePinned(CustomTestCase):
    """The arithmetic behind both provenance lines, so a term that moves shows."""

    def test_the_recorded_fallback_is_exactly_the_dec2b_number(self):
        self.assertEqual(
            derive_x_star(X_RECORDED_FLIP_S, X_RECORDED_R_D_TOKS,
                          X_RECORDED_R_P_TOKS, FLOOR),
            DEC2B_X,
        )
        self.assertEqual(
            (X_RECORDED_FLIP_S, X_RECORDED_R_D_TOKS, X_RECORDED_R_P_TOKS),
            (13.247, 690.0, 3640.0),
        )

    def test_the_dec1_seeds_price_the_dec1_band_not_the_fallback(self):
        x = derive_x_star(DEC1_FLIP_S, DEC1_R_D, DEC1_R_P, FLOOR)
        self.assertEqual(x, DEC1_X_FROM_PRINTED_SEEDS)
        self.assertLess(x, DEC2B_X * 0.6, "the fallback is 2.6x the measurement")

    def test_the_fallback_sits_outside_every_measured_seed_on_this_rig(self):
        """sb5f 8742, sb5e 12944, sb5d 10053, sb5c 9926 -- every measured log
        past the cap prices X in one band, and the fallback is not in it."""
        for measured in (8742, 12944, 10053, 9926):
            self.assertLess(measured, DEC2B_X * 0.6)


class OverridePrecedenceIsUntouched(CustomTestCase):
    def test_an_explicit_flag_still_wins_over_any_scan(self):
        d = _Dir(self)
        d.add("sb5f", _qualifying())
        seed = resolve_x(9999, d.path, FLOOR)
        self.assertEqual(seed.tokens, 9999)
        self.assertTrue(seed.measured, "an operator number is an instruction, not a table")
        self.assertIn("source=flag", seed.provenance)

    def test_the_flag_is_still_floored(self):
        d = _Dir(self)
        self.assertEqual(resolve_x(10, d.path, FLOOR).tokens, FLOOR)


class AnUnmeasuredXMayNotRefuseABoot(CustomTestCase):
    """The standing law in its exact shape: a fallback is never an actuator.

    Both refused trees priced the SAME real carrier bound, 27,466 -- 0.9 x D's
    30,518-token hicache host pool.  What refused them was the floor beside it,
    1.25 x 22,556 = 28,195, and 22,556 is a table.  One measured term, one
    table term, and the comparison decided whether a boot could start.
    """

    BOUND, HOST = 27466, 30518
    X_MEASURED, X_FALLBACK = 8742, 22556

    def _census(self, x):
        from sglang.srt.weg2 import carrier_census as cc

        floor, why = cc.route_floor(x)
        cen = cc.CarrierCensus(
            bound=self.BOUND,
            floor=floor,
            per_rank={0: self.BOUND, 1: self.BOUND, 2: self.BOUND},
            role="decode",
            fraction=0.9,
            host_size=self.HOST,
            site="hicache",
            expected_ranks=3,
            lines=("CARRIER source line",),
            verdict=("below_floor" if self.BOUND <= floor else "ok"),
            detail=(f"the agreed bound {self.BOUND} is at or below the floor {floor}"
                    if self.BOUND <= floor else f"3 TP ranks agree on {self.BOUND} tokens"),
        )
        return cc, cen, floor, why

    def test_the_two_boots_arithmetic_is_reproduced_exactly(self):
        cc, _cen, floor, _why = self._census(self.X_FALLBACK)
        self.assertEqual(floor, 28195)
        self.assertGreater(floor, self.BOUND, "this is what refused dec2b and shadowB")
        self.assertEqual(cc.route_floor(self.X_MEASURED)[0], 10927)
        self.assertLess(10927, self.BOUND, "the measured X leaves the interval open")

    def test_red_first_a_fallback_x_must_not_produce_a_w45(self):
        cc, cen, _floor, why = self._census(self.X_FALLBACK)
        dec = cc.decide_bound(cen, None, log_path="D.log", floor_why=why,
                              x_measured=False)
        self.assertFalse(dec.refused, "an unmeasured floor refused two boots")
        self.assertEqual(dec.bound, self.BOUND, "the MEASURED bound still ships")
        self.assertIn("UNGRADED", dec.note)
        self.assertIn("did NOT measure", dec.note)

    def test_a_measured_x_below_the_floor_still_refuses(self):
        """The guard must not become an off switch: with both terms measured
        the refusal is a real finding and stands."""
        cc, cen, _floor, why = self._census(self.X_FALLBACK)
        dec = cc.decide_bound(cen, None, log_path="D.log", floor_why=why,
                              x_measured=True)
        self.assertTrue(dec.refused)
        self.assertEqual(dec.reason, "below_floor")

    def test_the_default_is_still_to_grade(self):
        """Callers that do not pass the flag keep the old behaviour."""
        cc, cen, _floor, why = self._census(self.X_FALLBACK)
        self.assertTrue(cc.decide_bound(cen, None, log_path="D.log",
                                        floor_why=why).refused)

    def test_a_census_defect_refuses_whatever_x_is(self):
        """Only `below_floor` depends on X. `missing` is about the census and
        must refuse even on a fallback X, or the exemption becomes a bypass."""
        from sglang.srt.weg2 import carrier_census as cc

        cen = cc.CarrierCensus(
            bound=0, floor=28195, per_rank={}, role="?", fraction=0.0,
            host_size=0, site="", expected_ranks=3, lines=(),
            verdict="missing", detail="no rank emitted the line",
        )
        dec = cc.decide_bound(cen, None, log_path="D.log", x_measured=False)
        self.assertTrue(dec.refused)
        self.assertEqual(dec.reason, "missing")

    def test_an_ok_census_is_untouched_by_the_flag(self):
        cc, cen, _floor, why = self._census(self.X_MEASURED)
        self.assertEqual(cen.verdict, "ok")
        for measured in (True, False):
            dec = cc.decide_bound(cen, None, log_path="D.log", floor_why=why,
                                  x_measured=measured)
            self.assertFalse(dec.refused)
            self.assertEqual(dec.bound, self.BOUND)


class OnTheRealEvidenceTree(CustomTestCase):
    """Evidence-tree bound: skipped anywhere the boot logs do not live."""

    @unittest.skipUnless(os.path.exists(SB5F), "evidence-tree bound")
    def test_the_measurement_dec2b_missed_is_still_readable(self):
        from sglang.srt.weg2.launcher import measure_x_inputs

        got = measure_x_inputs(SB5F, FLOOR)
        self.assertIsNotNone(got, "the seed dec2b fell back past")
        flip_s, r_d, r_p, n_f, n_d, n_p = got
        self.assertEqual(derive_x_star(flip_s, r_d, r_p, FLOOR), DEC1_X_ON_METAL)
        self.assertEqual((round(flip_s, 2), round(r_d), round(r_p)), (2.90, 1138, 4628))
        self.assertEqual((n_f, n_d, n_p), (76, 17, 38))

    @unittest.skipUnless(os.path.exists(SB5F), "evidence-tree bound")
    def test_the_live_rig_seeds_x_from_a_boot_not_from_the_table(self):
        """Whatever has since landed in the evidence dir, the scan must reach a
        MEASUREMENT: this rig has four, and the newest one wins."""
        from sglang.srt.weg2.launcher import EVIDENCE_DIR

        seed = resolve_x(None, EVIDENCE_DIR, FLOOR)
        self.assertIn("source=boot:", seed.provenance)
        self.assertNotIn("PRE-BARLINK", seed.provenance)
        self.assertTrue(seed.measured)
        self.assertLess(seed.tokens, DEC2B_X * 0.6, "inside the measured band")


register_cpu_ci(__file__)

if __name__ == "__main__":
    unittest.main()
