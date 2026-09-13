# SPDX-License-Identifier: Apache-2.0
"""#1361 -- a refusal that hides its cause, and a W-code with no raise site.

(1) W97 PRINTED A VERDICT AND NOT A CAUSE. The train's dry run
    (arm_def.log:112 @ 19c7502642) showed only `predicted 98.67 vs 95.90`.
    True, and useless: nothing in it says WHICH term carried the 98.67, so an
    honest arm and a mispriced one read identically and the reader has to
    re-derive the ladder by hand. The refusal now prints the ARM line's own
    term fields, from ONE producer (`arm_terms_line`), so the two cannot drift
    -- the rule `_TERM_FIELDS` already enforces one layer down.

(2) W102 HAD NO RAISE SITE ANYWHERE IN THE TREE (`git grep
    Weg2XchgLanesUnmeasured 19c7502642 -- python/` returned the class
    definition and nothing else), while a comment in
    host_ledger.py:1755 @ 19c7502642 claimed the leg driver refused with it.
    The leg driver did refuse -- with W68 Weg2XchgPlanDisagree, the code for a
    wrong PHASE or lane KEY, not for a lane COUNT priced too low.

    An intention written as a state is exactly what a W-code census reads as
    covered, and this rig has now hit that shape in a reader, a guard, a
    comment and a probe on the same day. The leg driver raises W102 here.
"""

import inspect
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger as hl
from sglang.test.test_utils import CustomTestCase

GIB = hl.GIB
#: The #1360 specimen, verbatim: its predicted peak lands between the hard
#: bound and the reap watermark, which is the only window W97 acts in.
_KW = dict(
    ring_bytes=int(30.4 * GIB), ring_span1_bytes=int(15.0 * GIB),
    cg_current_bytes=int(9.0 * GIB), reclaimable_bytes=int(3.0 * GIB),
    cg_ceiling_bytes=int(123.78 * GIB), s_gb_d=4,
)


class _Arm:
    s_gb, m_mib = 1, 150
    terms = {
        "anchors_gib": 0.59, "rings_gib": 7.24, "overhead_gib": 0.31,
        "xchg_bounce_gib": 7.79, "host_ring_gib": 46.40,
        "flip_ratchet_charged_gib": 8.70,
        "flip_ratchet_source": "SERIES (measured over six boots)",
        "s_gb_d": 2,
    }


class TheRefusalNamesItsCause(CustomTestCase):
    def test_the_term_line_carries_every_arm_field(self):
        line = hl.arm_terms_line(_Arm())
        for token in ("anchors=0.59", "rings=7.24", "overhead=0.31",
                      "xchg_bounce=7.79", "host_weights=46.40",
                      "ratchet_charged=8.70", "source=SERIES"):
            self.assertIn(token, line, f"the term line omits {token}")

    def test_an_absent_term_says_n_a_not_zero(self):
        """An unpriced term and a term priced at zero must not share a spelling."""
        class _Partial(_Arm):
            terms = dict(_Arm.terms, rings_gib=None)

        self.assertIn("rings=n/a", hl.arm_terms_line(_Partial()))

    def test_w97_prints_the_terms_and_not_only_the_verdict(self):
        """EXECUTION, not a source pin: the refusal that really fires must
        carry the cause. Specimen is the #1360 watermark case -- the same
        shape the train's dry run refused with `predicted 98.67 vs 95.90`
        and nothing else."""
        with self.assertRaises(hl.Weg2HostDeviationRefused) as cm:
            hl.choose(
                int(123.78 * GIB), int(110.0 * GIB), arms=[(1, 2400)],
                flip_ratchet=hl.FlipRatchet(
                    per_flip_gib=40.0, flips_priced=1, source="test",
                    from_record=False),
                deviation_reason="r", riegel_gib=10.0, **_KW,
            )
        msg = str(cm.exception)
        self.assertIn("reap watermark", msg, "the verdict half must stay")
        self.assertIn("TERMS (S=", msg)
        for field in ("anchors=", "rings=", "overhead=", "xchg_bounce=",
                      "host_weights=", "ratchet_charged=", "source="):
            self.assertIn(field, msg, f"the refusal hides {field}")

    def test_the_refusal_terms_are_the_arms_terms_field_by_field(self):
        """The point of ONE producer: the refusal's numbers must be the arm's
        own, not a second derivation that can drift."""
        arm, _hr, _lines = hl.choose(
            int(123.78 * GIB), int(110.0 * GIB), arms=[(1, 150)],
            flip_ratchet=hl.resolve_flip_ratchet_gib(None),
            deviation_reason="xsn26 Zielfrage", riegel_gib=93.0, **_KW,
        )
        line = hl.arm_terms_line(arm)
        t = arm.terms or {}
        self.assertIn(f"rings={float(t['rings_gib']):.2f}", line)
        self.assertIn(f"xchg_bounce={float(t['xchg_bounce_gib']):.2f}", line)


class TheLanesCodeIsRaised(CustomTestCase):
    def test_w102_has_a_raise_site(self):
        """THE POINT: at 19c7502642 it had none, only a class and a comment."""
        from sglang.srt.managers.scheduler_components import weight_updater as wu

        src = inspect.getsource(wu)
        self.assertIn("raise hl.Weg2XchgLanesUnmeasured(", src)
        self.assertIn("W102 Weg2XchgLanesUnmeasured:", src)

    def test_the_leg_driver_no_longer_uses_the_phase_code_for_a_count(self):
        from sglang.srt.managers.scheduler_components import weight_updater as wu

        src = inspect.getsource(wu)
        i = src.index("the ledger priced ")
        window = src[max(0, i - 400):i + 200]
        self.assertNotIn("Weg2XchgBouncePhaseUnordered", window,
                         "a lane COUNT mismatch is not a phase/lane-key defect")

    def test_the_comment_and_the_code_agree(self):
        """The comment claimed a refusal that did not exist. Both are checked
        here so neither can drift back."""
        src = inspect.getsource(hl.resolve_xchg_lanes)
        self.assertIn("IT IS RAISED", src)
        self.assertIn("weight_updater.py", src)

    def test_it_is_still_a_ledger_refusal(self):
        self.assertTrue(
            issubclass(hl.Weg2XchgLanesUnmeasured, hl.Weg2HostLedgerRefused))


if __name__ == "__main__":
    unittest.main()
