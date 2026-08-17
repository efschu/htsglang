"""`prefill_in_tp`: the DEFAULT purity mode, and why the blanket rule went.

THE CORRECTION THIS PINS (user, 2026-08-14)
-------------------------------------------
The earlier rule "NOT A SINGLE TOKEN may be prefilled in the TP layout" is
WITHDRAWN, and the record that it was ordered as a hard user rule is struck:
it rested on wrong input data. Small prefills do not repay a seam round trip,
so the default is now the sensible setup -- prefill MAY run in TP and the
policy's measured break-even N decides when flipping to PP is worth it.

The asymmetry that justifies keeping the OTHER half:
  * misplacing PREFILL costs P x (1/r_tp - 1/r_pp) -- a ONE-OFF, bounded by
    prompt length;
  * misplacing DECODE costs O x (1/d_pp - 1/d_tp) -- RECURRING per output
    token, unbounded in generation length, and in PP it additionally forfeits
    CUDA graphs and speculation (measured here: accept length 3.24, 74.8%).
So decode in PP stays forbidden; prefill in TP does not.

WHAT FORCED IT, measured on this rig 2026-08-14
-----------------------------------------------
Under many small requests the blanket rule produced 882 flips in one boot,
arming `tp_to_pp` at 184 pending tokens against a live policy break-even of
N=7004, ~4.8 s of seam per request (tp_to_pp ~2.7 s + pp_to_tp ~2.1 s), and
TTFT ~2.9 s on a 65-CHARACTER prompt. The flip cost dominated the thing it
was meant to accelerate.

THE MACHINERY ALREADY EXISTED
-----------------------------
`phase_policy.break_even_tokens()` computes N and the policy flips on it.
`strict` collapses N to 0 (scheduler.py: "Tell the policy that the TP layout
cannot prefill, so its break-even N collapses to 0 in that direction"),
because under the prohibition a sub-N prompt would otherwise never run at
all. This mode adds no threshold -- it re-enables the one already there.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.managers.phase_purity import (
    MODE_OFF,
    MODE_PREFILL_IN_TP,
    MODE_STRICT,
    PhasePurity,
    parse_purity,
)
from sglang.test.test_utils import CustomTestCase


class TestTheDefault(CustomTestCase):
    def test_default_is_prefill_in_tp(self):
        """User correction 2026-08-14: the blanket prohibition is withdrawn."""
        self.assertEqual(parse_purity(None).mode, MODE_PREFILL_IN_TP)
        self.assertEqual(parse_purity("").mode, MODE_PREFILL_IN_TP)

    def test_the_default_lets_the_break_even_machinery_run(self):
        self.assertTrue(parse_purity(None).prefill_allowed_in_tp())

    def test_the_default_still_forbids_decode_in_pp(self):
        for bs in (0, 1, 4):
            with self.subTest(bs=bs):
                self.assertFalse(parse_purity(None).decode_allowed_in_pp(bs))

    def test_strict_still_forbids_both(self):
        p = parse_purity("strict")
        self.assertFalse(p.prefill_allowed_in_tp())
        self.assertFalse(p.decode_allowed_in_pp(0))
        self.assertFalse(p.decode_allowed_in_pp(1))


class TestPrefillInTpMode(CustomTestCase):
    def test_it_parses(self):
        self.assertEqual(parse_purity("prefill_in_tp").mode, MODE_PREFILL_IN_TP)

    def test_it_lifts_only_the_prefill_prohibition(self):
        p = parse_purity("prefill_in_tp")
        self.assertTrue(
            p.prefill_allowed_in_tp(),
            "the point of the mode: prefill may run in TP so the policy's "
            "break-even N governs the flip instead of collapsing to 0",
        )

    def test_decode_in_pp_stays_forbidden(self):
        """The half the 2026-08-09 starvation evidence actually indicts."""
        p = parse_purity("prefill_in_tp")
        for running_bs in (0, 1, 2, 8):
            with self.subTest(running_bs=running_bs):
                self.assertFalse(p.decode_allowed_in_pp(running_bs))

    def test_it_is_still_enforced_unlike_off(self):
        self.assertTrue(parse_purity("prefill_in_tp").enforced)
        self.assertFalse(parse_purity("off").enforced)

    def test_it_is_not_strict(self):
        self.assertFalse(parse_purity("prefill_in_tp").strict)

    def test_off_still_lifts_both(self):
        p = parse_purity(MODE_OFF)
        self.assertTrue(p.prefill_allowed_in_tp())
        self.assertTrue(p.decode_allowed_in_pp(99))

    def test_threshold_mode_is_untouched(self):
        """The sanctioned decode-in-PP escape hatch must not shift."""
        p = parse_purity("threshold:2")
        self.assertFalse(p.prefill_allowed_in_tp())
        self.assertTrue(p.decode_allowed_in_pp(2))
        self.assertFalse(p.decode_allowed_in_pp(3))

    def test_it_describes_itself(self):
        self.assertIn("prefill", parse_purity("prefill_in_tp").describe().lower())


if __name__ == "__main__":
    unittest.main()


class TestTheSemanticGuardProperty(CustomTestCase):
    """Consumers must test the PROPERTY, not the mode name.

    The spill machinery releases the draft weights for the whole PP phase on
    the guarantee "no decode ever runs in PP". It used to assert that by
    comparing the purity STRING to "strict", so adding `prefill_in_tp` --
    whose decode prohibition is identical -- crashed the boot with
    "spill depth >= 2 requires --phase-flip-purity strict, got
    'prefill_in_tp'". The guarantee was met; only the spelling differed.
    """

    def test_modes_that_forbid_pp_decode(self):
        for mode in ("strict", "prefill_in_tp"):
            with self.subTest(mode=mode):
                p = parse_purity(mode)
                self.assertTrue(p.decode_forbidden_in_pp)
                self.assertFalse(p.decode_allowed_in_pp(0))
                self.assertFalse(p.decode_allowed_in_pp(16))

    def test_modes_that_permit_pp_decode(self):
        for mode in ("off", "threshold:2"):
            with self.subTest(mode=mode):
                self.assertFalse(parse_purity(mode).decode_forbidden_in_pp)

    def test_threshold_zero_forbids_it_like_strict(self):
        p = parse_purity("threshold:0")
        self.assertTrue(p.decode_forbidden_in_pp)
