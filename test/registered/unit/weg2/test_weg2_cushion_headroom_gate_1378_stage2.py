# SPDX-License-Identifier: Apache-2.0
"""#1378 Stage 2 (W105) -- ``cushion_headroom_gib`` wired into the ARM point.

``test_weg2_cushion_gate_1377.py`` built and calibrated the arithmetic
(``cushion_headroom_gib``) and its own docstring proved the form already
wired into ``choose()`` -- ``predicted_run_peak + FLOOR <= bound``
(``cushion_ok``) -- funds BOTH weg2xsn31/2 and /3:

    xsn31/2   90.66 + 1.50 = 92.16 <= 94.43  -> funds
    xsn31/3   90.66 + 1.50 = 92.16 <= 94.43  -> funds   <-- W98 fires anyway

``cushion_headroom_gib`` itself was never called from ``choose()`` --
``git grep -c "cushion_headroom_gib("`` on host_ledger.py before this file
found exactly ONE occurrence, the ``def`` line. That is the wiring gap this
file closes.

THE ORDER'S FOUR HARD CONSTRAINTS, each with its own test class below:

1. NO FUNDING ON GREEN. The one-term model is empirically refuted in
   direction (weg2xsn31/5->6: bounce fell 16.905->12.00 and cushion fell
   WITH it, 0.85->0.20) -- a gate that greenlights and funds on this model
   would be a false green in the danger direction.
2. The number is PRINTED at the ARM point, on the SAME line as its own
   limitation in plain text, visible in the ladder's own output (which a
   dry run runs the same as a real boot) -- not only when it refuses.
3. RED MAY refuse (a negative headroom is a violated NECESSARY condition;
   this would have caught weg2xsn31/3 had the term existed then).
4. GREEN refuses nothing and FUNDS NOTHING; it is an indicator, never a
   Freibrief, and the printed text says so on the line itself.

Plus the required mutant (danger direction): green-despite-a-broken-cushion
must not exist -- i.e. the wiring must actually convert a negative headroom
into a refusal, not just print an unenforced number (the #1386 failure
class one layer down: an unenforced half of a switch reads as the whole
switch). That mutant is exercised manually (see the bottom of this file's
history in STEPS.md) rather than re-run automatically, the same convention
``test_weg2_hicache_disabled_1386.py`` documents for its own paired mutant.
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger as hl
from sglang.test.test_utils import CustomTestCase

GIB = hl.GIB

#: One roomy specimen (17.86 GiB of reap headroom with zero bounce priced)
#: so that adding weg2xsn31/3's own measured bounce (15.75 GiB) still leaves
#: `peak_ok` and the EXISTING single-boot `cushion_ok` both true -- proving
#: any refusal in this file is caused SOLELY by the new cross-boot term, not
#: by the two checks #1377(a) already wires.
_KW = dict(
    arms=[(1, 150)],
    ring_bytes=int(12.0 * GIB), ring_span1_bytes=int(4.0 * GIB),
    cg_current_bytes=int(9.0 * GIB), reclaimable_bytes=int(3.0 * GIB),
    cg_ceiling_bytes=int(123.78 * GIB), s_gb_d=4,
)
MEMTOTAL = int(123.78 * GIB)
MEMAVAIL = int(110.0 * GIB)

#: weg2xsn31/2's own recorded numbers (test_weg2_cushion_gate_1377.py's own
#: calibration): cushion_min 6.99 GiB, bounce 7.79 GiB.
PRIOR_CUSHION_MIN = 6.99
PRIOR_BOUNCE = 7.79
#: weg2xsn31/3's own recorded bounce: Option 1 sizing, 5 lanes x 24 slots x
#: 128 MiB.
XSN31_3_BOUNCE_GIB = 15.75


def _choose(**kw):
    return hl.choose(
        MEMTOTAL, MEMAVAIL,
        flip_ratchet=hl.resolve_flip_ratchet_gib(),
        **_KW, **kw,
    )


def _arm_line(lines):
    hit = [l for l in lines if l.startswith("WEG2-HOST-LEDGER ARM")]
    assert hit, "no ARM line printed"
    return hit[0]


class ThePriorFormStillFundsBothBoots(CustomTestCase):
    """Pins the #1377 finding this file's whole premise rests on: without the
    new term, this exact specimen funds despite carrying xsn31/3's own
    bounce. If this ever goes red, the fixture stopped isolating the new
    term and every other test in this file needs re-grounding."""

    def test_peak_ok_and_old_cushion_ok_both_hold_at_the_xsn31_3_bounce(self):
        arm, headroom, lines = _choose(
            xchg_bounce_host_bytes=int(XSN31_3_BOUNCE_GIB * GIB),
        )
        self.assertIsNotNone(arm)
        self.assertGreater(headroom, 0.0)


class NoFundingOnGreen(CustomTestCase):
    """Constraint 1 and 4: a GREEN headroom must never be the reason an arm
    funds, and must never read as clearance."""

    def test_a_zero_growth_green_reading_funds_but_is_not_a_clearance(self):
        arm, headroom, lines = _choose(
            xchg_bounce_host_bytes=int(PRIOR_BOUNCE * GIB),
            prior_cushion_min_gib=PRIOR_CUSHION_MIN,
            prior_bounce_gib=PRIOR_BOUNCE,
        )
        self.assertIsNotNone(arm, "zero growth must not itself refuse")
        line = _arm_line(lines)
        self.assertIn(f"cushion_headroom={PRIOR_CUSHION_MIN:.2f}", line)
        self.assertIn("GREEN refuses nothing and FUNDS NOTHING BY ITSELF", line,
                       "the disclaimer must be on the SAME printed line as "
                       "the number, not a separate one a reader can miss")

    def test_removing_the_gate_call_entirely_still_funds_this_specimen(self):
        """The gate is additive: an arm this file's OWN green case funds must
        also fund with no prior cited at all (constraint 4's other half --
        the absence of measurement is not itself a refusal)."""
        arm, _, _ = _choose(xchg_bounce_host_bytes=int(PRIOR_BOUNCE * GIB))
        self.assertIsNotNone(arm)


class ThePrintedLineCarriesTheNumberAndItsCaveatTogether(CustomTestCase):
    """Constraint 2: printed at the ARM point, same line as the caveat, and
    printed even when unmeasured -- visible in a dry run, where today
    nothing stands (the order's own words)."""

    def test_unmeasured_prints_a_named_absence_not_a_blank(self):
        arm, _, lines = _choose(xchg_bounce_host_bytes=int(1.0 * GIB))
        self.assertIsNotNone(arm)
        line = _arm_line(lines)
        self.assertIn(
            "cushion_headroom=not measured (no --prior-cushion-min-gib / "
            "--prior-bounce-gib passed)", line)

    def test_the_one_term_model_caveat_is_on_the_same_line_as_the_number(self):
        arm, _, lines = _choose(
            xchg_bounce_host_bytes=int(PRIOR_BOUNCE * GIB),
            prior_cushion_min_gib=PRIOR_CUSHION_MIN,
            prior_bounce_gib=PRIOR_BOUNCE,
        )
        line = _arm_line(lines)
        self.assertIn(f"cushion_headroom={PRIOR_CUSHION_MIN:.2f}", line)
        self.assertIn("ONE-TERM MODEL", line)
        self.assertIn("NON-MONOTONIC TWICE", line)
        self.assertIn("weg2xsn31/5->6", line,
                      "the boot that measured the non-monotonic direction "
                      "belongs on the line, not just in the docstring")

    def test_the_caveat_prints_even_when_the_arm_is_refused(self):
        """A refused arm's own line must still carry the number and caveat --
        a reader auditing a W20 must see the same evidence a funded arm's
        line would have shown."""
        with self.assertRaises(
                (hl.Weg2HostLedgerRefused, hl.Weg2HostRunPeakRefused)) as cm:
            _choose(
                xchg_bounce_host_bytes=int(XSN31_3_BOUNCE_GIB * GIB),
                prior_cushion_min_gib=PRIOR_CUSHION_MIN,
                prior_bounce_gib=PRIOR_BOUNCE,
            )
        msg = str(cm.exception)
        self.assertIn("cushion_headroom=-0.97", msg)
        self.assertIn("ONE-TERM MODEL", msg)


class RedMayRefuse(CustomTestCase):
    """Constraint 3, RED-FIRST: reproduces weg2xsn31/2->3's own transition on
    an arm the two EXISTING single-boot checks both pass, and asserts the
    NEW term is what refuses it."""

    def test_the_xsn31_2_to_3_transition_refuses_an_otherwise_fundable_arm(self):
        with self.assertRaises(
                (hl.Weg2HostLedgerRefused, hl.Weg2HostRunPeakRefused)) as cm:
            _choose(
                xchg_bounce_host_bytes=int(XSN31_3_BOUNCE_GIB * GIB),
                prior_cushion_min_gib=PRIOR_CUSHION_MIN,
                prior_bounce_gib=PRIOR_BOUNCE,
            )
        msg = str(cm.exception)
        self.assertIn("CUSHION HEADROOM", msg)
        got = hl.cushion_headroom_gib(
            PRIOR_CUSHION_MIN, XSN31_3_BOUNCE_GIB, PRIOR_BOUNCE)
        self.assertAlmostEqual(got, -0.97, places=2)
        self.assertIn(f"{got:.2f} < ", msg)

    def test_a_negative_headroom_binds_even_though_peak_and_old_cushion_pass(self):
        """The structural half of red-first: confirms this specific specimen
        was NOT already refused by `peak_ok`/`cushion_ok` before the new
        term existed (ThePriorFormStillFundsBothBoots), so this refusal is
        provably new, not a restatement of an old one."""
        arm, headroom, _ = _choose(xchg_bounce_host_bytes=int(XSN31_3_BOUNCE_GIB * GIB))
        self.assertIsNotNone(arm, "precondition: unmeasured, this arm funds")
        with self.assertRaises(
                (hl.Weg2HostLedgerRefused, hl.Weg2HostRunPeakRefused)):
            _choose(
                xchg_bounce_host_bytes=int(XSN31_3_BOUNCE_GIB * GIB),
                prior_cushion_min_gib=PRIOR_CUSHION_MIN,
                prior_bounce_gib=PRIOR_BOUNCE,
            )


class BothOrNeither(CustomTestCase):
    """W105: half a citation is worse than none -- it would print a number
    that looks measured but rests on a guessed second input."""

    def test_cushion_min_alone_is_refused(self):
        with self.assertRaises(hl.Weg2PriorCushionArgsIncomplete):
            _choose(xchg_bounce_host_bytes=int(1.0 * GIB),
                    prior_cushion_min_gib=PRIOR_CUSHION_MIN)

    def test_bounce_alone_is_refused(self):
        with self.assertRaises(hl.Weg2PriorCushionArgsIncomplete):
            _choose(xchg_bounce_host_bytes=int(1.0 * GIB),
                    prior_bounce_gib=PRIOR_BOUNCE)

    def test_the_refusal_is_checked_before_a_single_arm_is_priced(self):
        """Same discipline as the #1360 deviation pair: a half-armed call
        must never reach the ladder at all, let alone a verdict."""
        import inspect

        src = inspect.getsource(hl.choose)
        i_check = src.index("_prior_cited = ")
        i_loop = src.index("for arm in priced:")
        self.assertLess(i_check, i_loop,
                        "the both-or-neither check must run before pricing")


class TheGateCanOnlyTightenNeverLoosen(CustomTestCase):
    """Structural guard for constraint 4's mechanism: `headroom_ok` must be
    ANDed into `ok`, never ORed and never substituted for an existing term --
    an AND can only ever refuse an arm the other checks funded, matching
    'GREEN refuses nothing, RED may refuse'."""

    def test_ok_is_conjunctive_with_headroom_ok(self):
        import inspect

        src = inspect.getsource(hl.choose)
        i = src.index("ok = moments_ok and peak_ok and cushion_ok")
        line = src[i:i + 120].splitlines()[0]
        self.assertIn("and headroom_ok", line)

    def test_headroom_ok_is_true_when_unmeasured(self):
        """Absence must not refuse: a box with no prior boot cited funds
        exactly as it did before this file existed."""
        import inspect

        src = inspect.getsource(hl.choose)
        i = src.index("headroom_ok = (")
        window = src[i:i + 200]
        self.assertIn("cushion_headroom_this_arm_gib is None", window)


class TheCliReachesTheGateNotJustHlChoose(CustomTestCase):
    """Every test above calls `hl.choose` directly -- proves the arithmetic
    and the printed line, not that a boot can ever reach them. This class
    closes that gap the same way #1233's own seam tests do: source facts
    about the real call chain, since `main` mounts tmpfs and spawns servers
    and cannot be executed here."""

    def test_the_flags_exist_and_parse(self):
        from sglang.srt.weg2 import launcher

        ap = launcher.build_parser()
        help_text = ap.format_help()
        self.assertIn("--prior-cushion-min-gib", help_text)
        self.assertIn("--prior-bounce-gib", help_text)
        ns = ap.parse_args([
            "--tree", "x", "--tag", "y",
            "--prior-cushion-min-gib", "6.99", "--prior-bounce-gib", "7.79",
        ])
        self.assertAlmostEqual(ns.prior_cushion_min_gib, 6.99)
        self.assertAlmostEqual(ns.prior_bounce_gib, 7.79)

    def test_unset_stays_none_byte_identical_default(self):
        from sglang.srt.weg2 import launcher

        ap = launcher.build_parser()
        ns = ap.parse_args(["--tree", "x", "--tag", "y"])
        self.assertIsNone(ns.prior_cushion_min_gib)
        self.assertIsNone(ns.prior_bounce_gib)

    def test_choose_host_ledger_forwards_both_params_to_hl_choose(self):
        import inspect

        from sglang.srt.weg2 import launcher

        sig = inspect.signature(launcher.choose_host_ledger)
        self.assertIn("prior_cushion_min_gib", sig.parameters)
        self.assertIn("prior_bounce_gib", sig.parameters)
        self.assertIsNone(sig.parameters["prior_cushion_min_gib"].default)
        self.assertIsNone(sig.parameters["prior_bounce_gib"].default)
        src = inspect.getsource(launcher.choose_host_ledger)
        self.assertIn("prior_cushion_min_gib=prior_cushion_min_gib", src)
        self.assertIn("prior_bounce_gib=prior_bounce_gib", src)

    def test_main_reads_the_two_flags_into_the_seam(self):
        import inspect

        from sglang.srt.weg2 import launcher

        src = inspect.getsource(launcher.main)
        self.assertIn(
            'prior_cushion_min_gib=getattr(ns, "prior_cushion_min_gib", None)',
            src)
        self.assertIn(
            'prior_bounce_gib=getattr(ns, "prior_bounce_gib", None)', src)


if __name__ == "__main__":
    unittest.main()
