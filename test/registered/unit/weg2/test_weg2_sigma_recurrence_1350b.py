# SPDX-License-Identifier: Apache-2.0
"""#1350b -- Sigma H is self-referential, and the guard that stops it (#782 class).

THE MEASURED CHAIN, six consecutive boots of ONE form, from their own
`Sigma H <N> MiB = per card the LARGER of [per-card tag census <C> MiB,
measured dormant image <I> MiB]` provenance lines:

    boot         Sigma H   census   image   image-census   census - prev Sigma H
    xsn20-in      43996     42156   44716      +2560              --
    weg2xsn21b    44587     42492   44587      +2095            -1504
    weg2xsn22     45471     44586   45471       +885               -1
    weg2xsn23     46356     45470   46356       +886               -1
    weg2xsn24     47514     46354   47514      +1160               -2
    weg2xsn25(dry)48672     47512   48672      +1160               -2

READ THE LAST COLUMN. On four consecutive boots the per-card TAG CENSUS of boot
n is the Sigma H of boot n-1 minus ONE OR TWO MiB. The census is not an
independent anchor: the saver's `enable_cpu_backup` metadata reports the size of
the REGION each tag was backed into, and that region was sized from the previous
Sigma H. The measured dormant image then adds the ring's own preallocated slack
(+885..+1160 MiB) and BECOMES the next Sigma H.

Both instruments measure the ring the solve is sizing. Nothing physical moves --
the weights are the same checkpoint -- and Sigma H walks 43996 -> 48672 MiB,
**+4676 MiB = +4.57 GiB over six boots**. That is 16x what the entire M ladder
150 -> 80 buys (0.29 GiB), and it is what makes every arm of the xsn25 form
unfundable.

THE RULE, one-sided by design: a Sigma H may always SHRINK; it may RISE above
the SOURCE boot's own figure only within
`SIGMA_H_RECURRENCE_TOLERANCE_MIB`, and only while the form key matches. A rise
beyond that at matching form has no physical candidate and is refused -- the
raise itself where the census still anchors it (route falls back to the census),
by name (`W96 Weg2RingSigmaRecurrence`) where even the census has risen.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import ring_table as rt
from sglang.test.test_utils import CustomTestCase

#: (boot, Sigma H, per-card census, netted dormant image) -- verbatim, MiB.
CHAIN = (
    ("xsn20-in",   43996, 42156, 44716),
    ("weg2xsn21b", 44587, 42492, 44587),
    ("weg2xsn22",  45471, 44586, 45471),
    ("weg2xsn23",  46356, 45470, 46356),
    ("weg2xsn24",  47514, 46354, 47514),
    ("weg2xsn25",  48672, 47512, 48672),
)

#: The ledger's own posted non-backup host terms for the group whose RssShmem
#: was measured (anchors + rings at S=1 S_D=4 M=150), so the test feeds
#: `apportion_dormant` the same un-netted figure a boot does.
NON_BACKUP_MIB = 2079


def _apportion(census_mib, image_mib, source_sigma, form_matches):
    img = rt.DormantImage(
        group="P", rss_mib=image_mib + NON_BACKUP_MIB,
        weight_tags_mib=29525, extra_mib=0, arm={"s_gb": 1, "m_mib": 150},
    )
    rows, src, _bound = rt.apportion_dormant(
        "P", img, {"card": census_mib}, True, non_backup_mib=NON_BACKUP_MIB,
        source_sigma_h_mib=source_sigma, form_matches=form_matches,
    )
    return sum(rows.values()), src


class TheOldSamplerReproducesTheLadder1350b(CustomTestCase):
    """CAN-FAIL: without the guard, the measured ladder comes back exactly."""

    def test_the_unguarded_route_reproduces_every_measured_sigma_h(self):
        for boot, sigma, census, image in CHAIN:
            with self.subTest(boot=boot):
                total, src = _apportion(census, image, None, False)
                self.assertEqual(total, image)
                self.assertIn("route = netted RssShmem", src)
                # ...and the image IS that boot's recorded Sigma H from xsn21b
                # on (xsn20-in predates the apportionment this test pins).
                if boot != "xsn20-in":
                    self.assertEqual(total, sigma)

    def test_the_recurrence_is_in_the_data_not_in_the_guard(self):
        """census(n) == Sigma H(n-1) - 1..2 MiB on four consecutive boots."""
        for (_b0, sigma_prev, _c0, _i0), (b1, _s1, census, _i1) in zip(
            CHAIN[1:], CHAIN[2:]
        ):
            with self.subTest(boot=b1):
                self.assertIn(sigma_prev - census, (1, 2))
        self.assertEqual(CHAIN[-1][1] - CHAIN[0][1], 4676)   # +4.57 GiB


class TheGuardRefusesTheRise1350b(CustomTestCase):
    """(1) The raise is refused at matching form; the census stands."""

    def test_every_measured_row_falls_back_to_its_census(self):
        for boot, _sigma, census, image in CHAIN[1:]:
            with self.subTest(boot=boot):
                prev_sigma = dict(
                    (c[0], c[1]) for c in CHAIN
                )[CHAIN[[c[0] for c in CHAIN].index(boot) - 1][0]]
                total, src = _apportion(census, image, prev_sigma, True)
                self.assertEqual(total, census)
                self.assertIn("RAISE REFUSED", src)
                self.assertIn("REPORTED, NOT CHARGED", src)

    def test_a_shrink_is_always_allowed(self):
        """A genuinely smaller image is a finding, not a ratchet."""
        total, src = _apportion(42000, 41000, 47514, True)
        self.assertEqual(total, 42000)          # census still stands (>= usable)
        self.assertNotIn("RAISE REFUSED", src)
        # ...and a small rise inside the tolerance still charges.
        total, src = _apportion(42000, 47514 + 32, 47514, True)
        self.assertEqual(total, 47546)
        self.assertIn("route = netted RssShmem", src)

    def test_a_DIFFERENT_form_is_never_refused(self):
        """A different form may legitimately need a different ring."""
        total, src = _apportion(47512, 48672, 47514, False)
        self.assertEqual(total, 48672)
        self.assertIn("route = netted RssShmem", src)

    def test_a_missing_source_figure_disables_the_guard(self):
        """The guard refuses a rise it can PROVE, never one it cannot compare."""
        total, _src = _apportion(47512, 48672, None, True)
        self.assertEqual(total, 48672)


class TheSecondOrderRefusalIsNamed1350b(CustomTestCase):
    """(2) When even the CENSUS has risen, nothing non-self-referential is left."""

    def test_w96_by_name(self):
        with self.assertRaises(rt.Weg2RingSigmaRecurrence) as cm:
            _apportion(47512, 48672, 46354, True)
        msg = str(cm.exception)
        self.assertIn("W96 Weg2RingSigmaRecurrence", msg)
        self.assertIn("MATCHING form", msg)
        self.assertIn("SHRINK is always allowed", msg)

    def test_the_tolerance_separates_the_ratchet_from_the_jitter(self):
        """Smaller than every measured step, larger than every measured jitter."""
        self.assertLess(rt.SIGMA_H_RECURRENCE_TOLERANCE_MIB, 885)
        self.assertGreater(rt.SIGMA_H_RECURRENCE_TOLERANCE_MIB, 2)


class TheFixedChainStopsRising1350b(CustomTestCase):
    """(3) THE REPLAY: six values, before and after, in one simulation.

    The recurrence law the chain MEASURES is `census(n) = Sigma H(n-1) - d`
    with d = 1..2 MiB (four consecutive boots). Replaying the chain forward
    under that law with the guard armed therefore pins the series at the first
    census the guard bounds and lets it DECAY, instead of letting the ring's own
    slack compound:

        measured (no guard):  43996 -> 44587 -> 45471 -> 46356 -> 47514 -> 48672
        guarded:              43996 -> 42492 -> 42491 -> 42490 -> 42489 -> 42488

    i.e. every value lands at the WEIGHT-BYTE level of the first boot of the
    series (~42.2-42.5 k MiB = ~41.2-41.5 GiB) instead of walking to 47.5 GiB.
    Saving at the tip: 48672 - 42488 = 6184 MiB = 6.04 GiB.
    """

    def test_the_guarded_chain_never_rises(self):
        sigma = CHAIN[0][1]
        series = [sigma]
        for _i in range(5):
            census = sigma - 1          # the MEASURED recurrence law
            image = sigma + 1160        # the MEASURED slack
            total, _src = _apportion(census, image, sigma, True)
            sigma = total
            series.append(sigma)
        self.assertEqual(series[0], 43996)
        for a, b in zip(series, series[1:]):
            self.assertLessEqual(b, a)
        self.assertLessEqual(series[-1], 42492 + 1510)
        # the unguarded chain, for contrast, is the measured one
        self.assertEqual([c[1] for c in CHAIN][-1], 48672)
        self.assertGreater(48672 - series[-1], 4000)

    def test_the_saving_exceeds_what_the_whole_M_ladder_buys(self):
        """0.29 GiB from M = 150 -> 80, against >= 4 GiB here."""
        sigma = CHAIN[0][1]
        for _i in range(5):
            sigma, _ = _apportion(sigma - 1, sigma + 1160, sigma, True)
        self.assertGreater((48672 - sigma) / 1024.0, 0.29 * 4)


if __name__ == "__main__":
    unittest.main()


class TheLegIsNamedOnEveryNeedLine1350g(CustomTestCase):
    """#1350g: `set_leg` existed and NOTHING called it.

    Measured on boot weg2xsn25: 90 of 90 `WEG2-RING NEED` lines read
    `leg=unknown`, so the per-card sums cannot be split per sleep leg and the
    ring-anchor measuring boot produces nothing. The #1350d emitter fields were
    inert for want of one call.
    """

    def test_the_leg_bracket_calls_set_leg(self):
        """The call site itself -- desk-written-never-executed."""
        import inspect
        from sglang.srt.managers.scheduler_components import weight_updater as wu
        src = inspect.getsource(wu)
        self.assertIn("RingNeedGuard(", src)
        self.assertIn("weg2_ring_guard.set_leg(", src)
        # ...and it is INSIDE the same bracket that builds the guard.
        build = src.index("weg2_ring_guard = RingNeedGuard(")
        call = src.index("weg2_ring_guard.set_leg(")
        first_use = src.index("weg2_ring_guard.guard_tag(")
        self.assertLess(build, call)
        self.assertLess(call, first_use)   # named BEFORE the first NEED line

    def test_a_named_leg_reaches_the_line_and_an_unnamed_one_says_unknown(self):
        import logging
        from sglang.srt.weg2 import ring_guard as rg
        seen = []

        class _L(logging.Logger):
            def info(self, msg, *a):
                seen.append(msg % a if a else msg)

        g = rg.RingNeedGuard("card-x", group="P", rank=0, log=_L("t"))
        self.assertEqual(g.leg, "unknown")
        g.record("weights_0", 10, 99)
        self.assertIn("leg=unknown", seen[-1])
        self.assertEqual(g.set_leg(7, "P", "D"), "7/P->D")
        g.record("weights_1", 10, 99)
        self.assertIn("leg=7/P->D", seen[-1])
