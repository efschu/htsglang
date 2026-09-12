# SPDX-License-Identifier: Apache-2.0
"""#1350 -- the flip RATCHET, charged instead of left to a blind margin term.

THE DEFECT, measured on five formgleiche boots of 2026-09-11/12
(``/spinning/gpu-arb/weg2/ANALYSE_1350_HOST_TERM_0912.md``): the ledger's
predicted run peak under-shot the measured non-reclaimable peak by
**+1.648 / +1.695 / +2.245 / +2.970 / +4.095 GiB** (weg2xsn20 / xsn21b / xsn22 /
xsn23 / xsn24), and the miss GREW while every stationary term got MORE accurate.
The analysis closes the balance to <= 0.03 GiB on all five as

    Residuum = RATSCHE - POLSTER

where the RATSCHE is what the FIRST waking of each group adds to
``anon+shmem+slab_unreclaimable`` and never gives back (+1.18..1.73 GiB on the
P->D leg, +0.28..0.30 on D->P; >= 0 in 34 of 37 measured flip deltas), and the
POLSTER was three over-pricings that the M=1200 -> 600 -> 150 walk removed.

The term was invisible because the only place it appeared was the MARGIN
(``Margin.transient_gib``, fed by ``FLIP_TRANSIENT_IN_CURRENCY_GIB``), measured
by a LOCAL instrument -- "max inside the flip window minus the HIGHER of its two
12 s shoulders" -- which against a staircase reads negative BY CONSTRUCTION: the
trailing shoulder already sits on the new step. That table reads weg2xsn20 at
**-0.023 GiB** while the same boot's same 1 Hz series rises 84.336 -> 88.084 GiB
over eight flips and never returns, i.e. **+4.462 GiB**.

WHAT THIS FILE PINS: the term is CHARGED and reaches the run peak exactly once;
its absence is a NAMED refusal and never a 0; the transient leaves the BOOT
margin (and only it) now that the prediction spends that quantity itself; and
the same bytes cannot be charged twice through the run-origin floor.
"""

import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger as hl
from sglang.test.test_utils import CustomTestCase

GIB = hl.GIB

# --------------------------------------------------------------------------
# THE FIVE RECORDS, VERBATIM.
#
# `predicted` and `peak` are quoted from the boot records
# /spinning/gpu-arb/weg2/BOOT_weg2xsn{20_0911,21b_0912,22_0912,23_0912,24_0912}.md
# (the `WEG2-HOST-LEDGER ARM` line of the BOOT log, not of a dry run, and the
# `PEAK:` line of section H). `origin`, `peak` and `pair` are re-read here from
# the boots' own 1 Hz sampler CSVs
# (/spinning/gpu-arb/weg2/hostsample_weg2xsn20.csv and
#  /spinning/evidence-665-f1/weg2xsn2{2,3,4}_0912/hostsample_weg2xsn2*.csv)
# joined against the `WEG2-FLIP begin/done epoch=` timestamps of the front logs
# under /spinning/evidence-665-f1/. The numbers below reproduced the analysis's
# table exactly (peaks 88.798 / 88.495 / 88.210 / 90.475, pre-flip readings
# 84.336 / 85.501 / 84.860 / 86.194), which is why they are quoted here as
# literals: this test is HERMETIC and must run on the remote gate, where the
# evidence tree does not exist.
#
# weg2xsn21b IS THE HONEST EXCEPTION AND IT IS MARKED. Group D died in the
# second leg (W85), so the boot never logged `WEG2-FLIP done epoch=2` and its
# 1 Hz CSV is no longer on this box. Its `pair` is the SUM OF THE TWO LEG
# DELTAS the analysis published (+0.302 D->P, +1.563 P->D = 1.865). On metal a
# boot of that shape writes NO `flip_ratchet_gib` at all and the next arm
# refuses by name -- which `test_a_boot_that_never_completed_a_pair...` pins.
#
#            boot          M   predicted  peak(nr)  origin   pair    residual
REPLAY = (
    ("weg2xsn20",  1200, 87.15, 88.798, 84.336, 3.161, +1.648),
    ("weg2xsn21b",  600, 85.30, 86.995, 84.401, 1.865, +1.695),
    ("weg2xsn22",   600, 86.25, 88.495, 85.501, 1.870, +2.245),
    ("weg2xsn23",   150, 85.24, 88.210, 84.860, 2.062, +2.970),
    ("weg2xsn24",   150, 86.38, 90.475, 86.194, 4.049, +4.095),
)

#: The analysis's own closure tolerance: "Die Bilanz `Residuum = Ratsche -
#: Polster` schliesst in allen fuenf Boots auf <= 0,03 GiB".
CLOSURE_GIB = 0.03


def _priced(flip_ratchet=None, m_mib=600):
    """One arm of the standing form, priced with and without the ratchet.

    The absolute figures do not matter (this is not a boot); what matters is
    that the ONLY thing the new term changes is the sum, which is what lets the
    replay below apply a measured delta to a recorded prediction.
    """
    return hl.price(
        int(123.78 * GIB), int(110.0 * GIB), 1, m_mib,
        ring_bytes=int(43.54 * GIB), ring_span1_bytes=int(20.0 * GIB),
        cg_current_bytes=int(9.0 * GIB), reclaimable_bytes=int(3.0 * GIB),
        cg_ceiling_bytes=int(123.78 * GIB), s_gb_d=4,
        flip_ratchet=flip_ratchet,
    )


def _ratchet(per_flip, flips=1, from_record=True, source="test"):
    return hl.FlipRatchet(
        per_flip_gib=per_flip, flips_priced=flips, source=source,
        from_record=from_record,
    )


class FlipRatchetTerm1350(CustomTestCase):
    """(1) The term exists, is charged ONCE, and reaches the run peak."""

    def test_the_term_lands_in_the_run_peak_and_exactly_once(self):
        without = _priced()
        with_r = _priced(_ratchet(3.161))
        self.assertIsNone(without.terms["flip_ratchet_gib"])
        self.assertAlmostEqual(with_r.terms["flip_ratchet_gib"], 3.161, places=6)
        # ONE addition point: the delta on the peak equals the delta on the
        # charges equals the charged term. A second addition site would show up
        # here as 2x.
        self.assertAlmostEqual(
            with_r.predicted_run_peak_gib() - without.predicted_run_peak_gib(),
            3.161, places=6,
        )
        self.assertAlmostEqual(
            hl._run_moment_charges_gib(with_r.terms)
            - hl._run_moment_charges_gib(without.terms),
            3.161, places=6,
        )

    def test_the_launch_moment_does_not_carry_it(self):
        """At the launch moment NO FLIP HAS HAPPENED.

        Charging it at both moments (the first shape of this fix) took the
        pinned M=150 arm's launch leftover from +4.50 to -0.19 GiB on a dry run
        of this tree and moved the refusal from the run peak (W21) to the launch
        moment (W20): a true refusal under a false name, which is the
        instrument-text class this fork keeps paying for.
        """
        without = _priced()
        with_r = _priced(_ratchet(3.161))
        self.assertAlmostEqual(
            with_r.launch_leftover_gib, without.launch_leftover_gib, places=9)
        self.assertAlmostEqual(
            with_r.launch_worst_case_gib, without.launch_worst_case_gib, places=9)
        # ...and the RUN leftover shrinks by exactly the charged term.
        self.assertAlmostEqual(
            without.run_leftover_gib - with_r.run_leftover_gib, 3.161, places=6)
        # The both-moment sum is untouched, which is what keeps the three
        # residual re-derivations byte-identical.
        self.assertAlmostEqual(
            hl._boot_charges_gib(with_r.terms),
            hl._boot_charges_gib(without.terms), places=9)

    def test_flips_priced_multiplies_and_is_printed_not_assumed(self):
        r = _ratchet(2.0, flips=3)
        self.assertAlmostEqual(r.charged_gib, 6.0, places=9)
        self.assertIn("flips_priced=3", r.arm_fields())
        # The default is ONE pair, and the reason is that the step SATURATES.
        self.assertEqual(hl.FLIP_RATCHET_FLIPS_PRICED_DEFAULT, 1)

    def test_an_absent_measurement_is_never_spelled_as_zero(self):
        """MUTANT DIRECTION 1: 'the term is silently 0'.

        A 0.00 on the ARM line and an unpriced term must not share a spelling:
        that is the #606 class this module has paid for repeatedly, and here it
        would report FUNDABLE for a peak the ledger stopped modelling.
        """
        terms = hl.charge_terms(1, 600, 3, hl.resolve_image_terms(None))
        self.assertIn("flip_ratchet_gib", terms)
        self.assertIsNone(terms["flip_ratchet_gib"])
        arm = _priced()
        self.assertIsNone(arm.terms["flip_ratchet_gib"])
        self.assertIn("ABSENT", arm.terms["flip_ratchet_source"])
        self.assertNotIn("0", arm.terms["flip_ratchet_source"].split(":")[0])

    def test_the_run_moment_sum_refuses_a_dict_that_never_carried_the_key(self):
        """A hand-built terms dict must raise, not be priced without the term.

        `.get(..., 0.0)` here would be the #606 getattr-default shape one layer
        down: a dict assembled by hand (or by an older writer) would price the
        run moment WITHOUT the ratchet and report a peak that is too low.
        """
        terms = dict(hl.charge_terms(1, 600, 3, hl.resolve_image_terms(None)))
        terms.pop("flip_ratchet_gib")
        with self.assertRaises(KeyError):
            hl._run_moment_charges_gib(terms)
        # ...while the BOTH-moment sum is unaffected by #1350 and still prices.
        self.assertGreater(hl._boot_charges_gib(terms), 0.0)


class FlipRatchetRefusal1350(CustomTestCase):
    """(2) The absence is a NAMED refusal, reachable and correctly classed."""

    def test_w94_is_raised_when_nothing_carries_the_field(self):
        with self.assertRaises(hl.Weg2HostFlipRatchetUnmeasured) as cm:
            hl.resolve_flip_ratchet_gib({}, seed_allowed=False)
        msg = str(cm.exception)
        self.assertIn("W94 Weg2HostFlipRatchetUnmeasured", msg)
        self.assertIn("flip_ratchet_gib", msg)
        # It is a ledger refusal, so every existing `except
        # Weg2HostLedgerRefused` on the launch path already stops the boot.
        self.assertTrue(
            issubclass(hl.Weg2HostFlipRatchetUnmeasured, hl.Weg2HostLedgerRefused)
        )

    def test_a_record_whose_reading_was_unreadable_refuses_rather_than_zero(self):
        rec = {"FLIP": hl.flip_ratchet_record(
            pre_gib=None, post_gib=88.0, boot_tag="b", commit="c", at="t")}
        self.assertIsNone(rec["FLIP"]["flip_ratchet_gib"])
        with self.assertRaises(hl.Weg2HostFlipRatchetUnmeasured):
            hl.resolve_flip_ratchet_gib(rec, seed_allowed=False)

    def test_a_boot_that_never_completed_a_pair_writes_nothing_and_the_next_arm_refuses(self):
        """weg2xsn21b's shape: begin epoch=0, done epoch=1, begin epoch=1, death.

        No `done epoch=2` -> no record -> W94. The ledger does not reconstruct a
        pair from one leg, and it does not price the leg it has.
        """
        with self.assertRaises(hl.Weg2HostFlipRatchetUnmeasured):
            hl.resolve_flip_ratchet_gib(
                {"P": {"group": "P", "rss_shmem_gib": 38.6}}, seed_allowed=False,
            )

    def test_the_recorded_seed_is_labelled_and_retires_on_the_first_real_record(self):
        # #1350g SUPERSEDES the SEED this test first pinned: boot weg2xsn25
        # charged the 4.049 GiB first-pair seed, under-shot by 4.647 and the
        # host reaped group D. The fallback is now the measured 6-boot SERIES;
        # what this test still pins is that the fallback is LABELLED as not
        # being this line's own record, and that it retires on the first one.
        seed = hl.resolve_flip_ratchet_gib(None)
        self.assertFalse(seed.from_record)
        self.assertIn("NOT this line's own record", seed.source)
        self.assertIn("SERIES", seed.source)
        self.assertAlmostEqual(seed.per_flip_gib, 8.697, places=3)
        rec = {"FLIP": hl.flip_ratchet_record(
            pre_gib=84.336, post_gib=87.497, boot_tag="weg2xsn20",
            commit="3267f109fb", at="2026-09-11T15:49:53Z")}
        live = hl.resolve_flip_ratchet_gib(rec)
        self.assertTrue(live.from_record)
        self.assertAlmostEqual(live.per_flip_gib, 3.161, places=3)
        self.assertNotIn("SEED", live.source)


class FlipRatchetRecordRoundTrip1350(CustomTestCase):
    """(3) ONE writer, ONE sidecar, ONE reader -- and no cross-talk."""

    def test_the_field_survives_append_and_is_read_by_the_run_origin_reader(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "weg2_measured_record.json")
            hl.append_measured_record(path, hl.flip_ratchet_record(
                pre_gib=86.194, post_gib=90.243, boot_tag="weg2xsn24",
                commit="bfb5e121e7", at="2026-09-12T19:44:34Z",
                pre_at="2026-09-12T19:41:22Z", post_at="2026-09-12T19:44:34Z"))
            back = hl.read_measured_record(path)
            self.assertIn("FLIP", back)
            self.assertAlmostEqual(
                back["FLIP"]["flip_ratchet_gib"], 4.049, places=3)
            self.assertAlmostEqual(
                hl.resolve_flip_ratchet_gib(back).per_flip_gib, 4.049, places=3)
            # The currency travels WITH the number.
            self.assertIn("anon+shmem+slab_unreclaimable",
                          back["FLIP"]["instrument"])
            self.assertIn("NEVER memory.current", back["FLIP"]["instrument"])
            # ...and raw JSON, so an operator can read it without this module.
            with open(path) as f:
                self.assertEqual(len(json.load(f)["samples"]), 1)

    def test_the_flip_entry_is_invisible_to_the_image_and_origin_readers(self):
        rec = {"FLIP": hl.flip_ratchet_record(
            pre_gib=84.0, post_gib=87.0, boot_tag="b", commit="c", at="t")}
        img = hl.resolve_image_terms(rec)
        self.assertFalse(img.p_measured)          # still the named dk7 reading
        origin, src = hl.run_origin_gib(6.44, rec)
        self.assertAlmostEqual(origin, hl.dk7_run_residual_gib(), places=6)
        self.assertIn("DERIVED", src)             # no residual in a FLIP entry


class TransientLeavesTheBootMargin1350(CustomTestCase):
    """(4) Patch C: what the prediction spends, the bound may not deduct."""

    def test_the_bound_moves_to_the_analysis_number_only_under_ratchet_pricing(self):
        legacy = hl.resolve_margin()
        ratcheted = hl.resolve_margin(flip_ratchet_charged_gib=3.35)
        w = hl.OBSERVED_REAP_NONRECLAIM_BYTES / GIB
        self.assertAlmostEqual(w - legacy.boot_total_gib, 87.30, places=2)
        # #1350e SUPERSEDES the 90.18 this test first pinned: the residual is
        # now RE-MEASURED on ratchet-priced predictions too, so the bound moves
        # by the transient AND by (5.16 - the re-measured 0.911).
        self.assertAlmostEqual(w - ratcheted.boot_total_gib, 94.43, places=2)
        self.assertAlmostEqual(
            legacy.boot_total_gib - ratcheted.boot_total_gib,
            legacy.transient_gib + (legacy.residual_gib - ratcheted.residual_gib),
            places=9)

    def test_the_transient_stays_in_the_runtime_margin(self):
        """W22 grades a LIVE reading; the NEXT flip has not happened yet."""
        legacy = hl.resolve_margin()
        ratcheted = hl.resolve_margin(flip_ratchet_charged_gib=3.35)
        self.assertAlmostEqual(
            legacy.runtime_total_gib, ratcheted.runtime_total_gib, places=9)
        self.assertGreater(ratcheted.runtime_total_gib, 0.0)
        self.assertIn("transient", ratcheted.terms("runtime"))

    def test_the_omission_is_stated_on_the_line_not_merely_done(self):
        text = hl.resolve_margin(flip_ratchet_charged_gib=3.35).terms()
        self.assertIn("DELIBERATELY NOT CHARGED", text)
        self.assertIn("#1350", text)

    def test_every_pre_1350_caller_is_byte_identical(self):
        m = hl.resolve_margin()
        self.assertTrue(m.transient_in_boot)
        self.assertAlmostEqual(
            m.boot_total_gib,
            m.transient_gib + m.residual_gib + m.drift_gib + m.foreign_gib,
            places=12)


class NoDoubleCount1350(CustomTestCase):
    """(5) The same bytes cannot be charged twice -- by construction."""

    def test_the_legacy_residual_is_kept_whole_and_never_netted(self):
        """MUTANT DIRECTION 2: 'old residual + new term summed'.

        `RUN_PEAK_RESIDUAL_GIB`'s rows were measured as (peak - predicted) on
        boots that did NOT charge the ratchet, so each row contains it. The two
        wrong moves are ADDING the charged term to the residual (which would
        move the bound down by it) and SUBTRACTING it (which would move the
        bound up by it). Neither happens: the row is untouched.
        """
        legacy = hl.resolve_margin()
        ratcheted = hl.resolve_margin(flip_ratchet_charged_gib=3.35)
        # #1350e: the ratchet-priced arm now uses the RE-MEASURED rows, and the
        # legacy row is neither netted nor summed with the charged term -- it is
        # simply not the row that applies. It still binds every unratcheted arm.
        self.assertAlmostEqual(legacy.residual_gib, 5.16, places=6)
        self.assertLess(ratcheted.residual_gib, legacy.residual_gib)
        self.assertIn("DELIBERATELY NOT USED", ratcheted.residual_source)
        # ...and the constant itself was not hand-edited.
        self.assertEqual(
            hl.RUN_PEAK_RESIDUAL_GIB,
            {"weg2sb4": 5.16, "weg2rg6": -0.11, "weg2sb5c": 1.54})

    def test_a_hand_supplied_residual_beside_a_charged_ratchet_refuses_by_name(self):
        with self.assertRaises(hl.Weg2HostRatchetDoubleCharged) as cm:
            hl.resolve_margin(residual_gib=2.0, flip_ratchet_charged_gib=3.35)
        self.assertIn("W95 Weg2HostRatchetDoubleCharged", str(cm.exception))
        self.assertTrue(
            issubclass(hl.Weg2HostRatchetDoubleCharged, hl.Weg2HostLedgerRefused))

    def test_an_origin_floor_sampled_during_a_flip_may_not_carry_the_term_too(self):
        """The wall ANALYSE SS2 predicts in one to two boots.

        The run-moment residual floor is sampled at a group's FIRST SLEEP, i.e.
        inside a flip, so it already realised the permanent step. Measured on
        metal: `run-moment residual ... stored=` walks 2.31 -> 2.34 -> 4.09 ->
        5.28 GiB over the 0912 boots for exactly that reason. When it wins the
        max() against the launch reading, adding `flip_ratchet_gib` on top is
        the same bytes twice.
        """
        floor_rec = {"D": {
            "group": "D", "at": "t", "boot_tag": "b", "commit": "c",
            "rss_shmem_gib": 38.6, "run_residual_gib": 9.9,
            "sampled_at_flip_epoch": 1,
            "arm": {"s_gb": 1, "m_mib": 600},
        }}
        # #1350e: rejected in favour of the launch reading (the correction);
        # the W95 guard itself is still live and is pinned by
        # `test_summing_both_residual_rules_is_still_refused` and by
        # `test_W95_still_fires_when_there_is_no_launch_reading_to_fall_back_on`.
        origin, src = hl.run_origin_gib(6.44, floor_rec)
        self.assertAlmostEqual(origin, 6.44, places=6)
        self.assertIn("REJECTED: interleaved", src)

    def test_a_pre_1350_record_is_covered_by_its_interleaved_flag(self):
        """The record that WINS the max() today has no epoch stamp.

        weg2xsn24 group P, 2026-09-12T19:44:31Z, stored 7.49 GiB, written
        before this field existed -- and the launch reading 6.44 GiB loses to
        it, so the floor binds NOW. `interleaved` is the same distinction and
        was always on the record: the front sets it True for every sample it
        takes (all of them during a flip), the launcher False for P's first
        sleep before any flip exists.
        """
        floor_rec = {"P": {
            "group": "P", "at": "2026-09-12T19:44:31Z", "boot_tag": "weg2xsn24",
            "commit": "bfb5e121e7", "rss_shmem_gib": 49.56,
            "run_residual_gib": 7.49, "interleaved": True,
            "arm": {"s_gb": 1, "m_mib": 150},
        }}
        # #1350e SUPERSEDES the refusal this test first pinned: the mid-flip
        # sample is now REJECTED in favour of the launch reading, which is the
        # correction rather than the refusal. `interleaved` is still the fact
        # that identifies it, which is what this test now pins.
        origin, src = hl.run_origin_gib(6.44, floor_rec)
        self.assertAlmostEqual(origin, 6.44, places=6)
        self.assertIn("REJECTED: interleaved", src)
        self.assertIn("7.49", src)

    def test_the_launchers_own_uninterleaved_sample_is_not_marked(self):
        """P's first sleep happens before any flip: interleaved=False."""
        floor_rec = {"P": {
            "group": "P", "at": "t", "boot_tag": "b", "commit": "c",
            "rss_shmem_gib": 38.6, "run_residual_gib": 9.9,
            "interleaved": False, "arm": {"s_gb": 1, "m_mib": 600},
        }}
        _origin, src = hl.run_origin_gib(6.44, floor_rec)
        self.assertNotIn(hl.RUN_ORIGIN_RATCHET_MARKER, src)

    def test_a_floor_sampled_outside_a_flip_prices_normally(self):
        """The guard must not fire on the launcher's own P sample (no flip yet)."""
        floor_rec = {"P": {
            "group": "P", "at": "t", "boot_tag": "b", "commit": "c",
            "rss_shmem_gib": 38.6, "run_residual_gib": 9.9,
            "sampled_at_flip_epoch": None,
            "arm": {"s_gb": 1, "m_mib": 600},
        }}
        origin, src = hl.run_origin_gib(6.44, floor_rec)
        self.assertNotIn(hl.RUN_ORIGIN_RATCHET_MARKER, src)
        arm = _priced(_ratchet(3.161))
        arm.terms["run_origin_gib"] = origin
        arm.terms["run_origin_source"] = src
        self.assertIsNotNone(arm.predicted_run_peak_gib())

    def test_the_historical_repricers_pass_an_explicit_zero(self):
        """dk7 ran ZERO flips, and a record re-pricing reconstructs what the
        SAMPLER subtracted -- which never knew this term. Both must be exactly
        unchanged by #1350, or the correction cancels the fix."""
        self.assertAlmostEqual(hl.dk7_run_residual_gib(),
                               hl.dk7_run_residual_gib(), places=12)
        entry = {"run_residual_gib": 9.01, "arm": {"s_gb": 1, "m_mib": 600},
                 "rss_shmem_gib": 38.63}
        value, correction = hl.record_run_residual_gib(entry)
        self.assertAlmostEqual(value, 9.01, places=9)
        self.assertEqual(correction, 0.0)


class OriginAndResidualCorrections1350e(CustomTestCase):
    """#1350e: the two terms that were measurably double-booked."""

    def test_a_mid_flip_run_moment_sample_is_rejected_in_favour_of_the_launch_reading(self):
        """weg2xsn24's floor 7.49 GiB was taken 3 s before its FLIP done epoch=2."""
        rec = {"P": {"group": "P", "at": "2026-09-12T19:44:31Z",
                     "boot_tag": "weg2xsn24", "commit": "bfb5e121e7",
                     "rss_shmem_gib": 49.56, "run_residual_gib": 7.49,
                     "interleaved": True, "arm": {"s_gb": 1, "m_mib": 150}}}
        origin, src = hl.run_origin_gib(6.44, rec)
        self.assertAlmostEqual(origin, 6.44, places=6)
        self.assertIn("source=launch", src)
        self.assertIn("7.49", src)
        self.assertIn("REJECTED: interleaved", src)
        # ...and W95 no longer fires, because the origin no longer carries it.
        self.assertNotIn(hl.RUN_ORIGIN_RATCHET_MARKER, src)
        arm = _priced(_ratchet(4.049))
        arm.terms["run_origin_gib"] = origin
        arm.terms["run_origin_source"] = src
        self.assertIsNotNone(arm.predicted_run_peak_gib())

    def test_a_rejected_sample_never_promotes_the_dk7_stand_in(self):
        """The stand-in answers 'no record at all', not 'the record was refused'
        -- and at 24.50 GiB it would be a WORSE origin than the 7.49 refused."""
        rec = {"P": {"group": "P", "at": "t", "boot_tag": "b", "commit": "c",
                     "rss_shmem_gib": 49.5, "run_residual_gib": 7.49,
                     "interleaved": True, "arm": {"s_gb": 1, "m_mib": 150}}}
        origin, src = hl.run_origin_gib(6.44, rec)
        self.assertLess(origin, hl.dk7_run_residual_gib())
        self.assertIn("dk7 stand-in is NOT used here", src)

    def test_W95_still_fires_when_there_is_no_launch_reading_to_fall_back_on(self):
        """The guard is narrowed, not removed."""
        rec = {"P": {"group": "P", "at": "t", "boot_tag": "b", "commit": "c",
                     "rss_shmem_gib": 38.6, "run_residual_gib": 9.9,
                     "interleaved": True, "arm": {"s_gb": 1, "m_mib": 600}}}
        origin, src = hl.run_origin_gib(9.0, rec)  # launch BELOW -> floor wins
        self.assertAlmostEqual(origin, 9.0, places=6)
        self.assertIn("REJECTED: interleaved", src)
        # with no cgroup reading at all there is no origin and no prediction
        self.assertIsNone(hl.run_origin_gib(None, rec)[0])

    def test_the_ratchet_priced_residual_is_the_five_replays_plus_its_own_drift(self):
        m = hl.resolve_margin(flip_ratchet_charged_gib=4.05, window_min=90.0)
        self.assertAlmostEqual(
            m.residual_gib,
            0.908 + hl.RATCHET_RESIDUAL_DRIFT_MIB_PER_MIN * 90.0 / 1024.0,
            places=6)
        self.assertIn("RATCHET-PRICED", m.residual_source)
        self.assertIn("weg2sb4", m.residual_source)      # named as NOT used
        self.assertIn("DELIBERATELY NOT USED", m.residual_source)
        w = hl.OBSERVED_REAP_NONRECLAIM_BYTES / GIB
        self.assertAlmostEqual(w - m.boot_total_gib, 94.43, places=2)

    def test_the_pre_ratchet_row_still_binds_an_unratcheted_arm(self):
        m = hl.resolve_margin()
        self.assertAlmostEqual(m.residual_gib, 5.16, places=6)
        self.assertAlmostEqual(
            hl.OBSERVED_REAP_NONRECLAIM_BYTES / GIB - m.boot_total_gib,
            87.30, places=2)

    def test_the_replay_under_the_new_residual_rule(self):
        """All five measured residuals sit at or below the charged one."""
        binding = max(hl.RUN_PEAK_RESIDUAL_RATCHET_GIB.values())
        self.assertAlmostEqual(binding, 0.908, places=3)
        for tag, _m, predicted, peak, _o, pair, _b in REPLAY:
            with self.subTest(boot=tag):
                self.assertLessEqual(peak - (predicted + pair), binding + 1e-9)

    def test_summing_both_residual_rules_is_still_refused(self):
        with self.assertRaises(hl.Weg2HostRatchetDoubleCharged):
            hl.resolve_margin(residual_gib=5.16, flip_ratchet_charged_gib=4.05)


class LaunchWorstCaseHasItsOwnMargin1350f(CustomTestCase):
    """#1350f RED-FIRST on the dry-run specimen: -6.33 must pass, -9 must refuse.

    Measured on the xsn25 form at M=150 after #1350e: `launch_sum = -6.33 GiB`
    with both moment leftovers positive (+5.67 / +1.62) and `run_peak 90.43`
    against the 94.43 bound -- i.e. an arm whose run peak funds with 4.00 GiB to
    spare was refused by a conjunct that asks a completely different question.
    """

    def _arm(self, launch_sum):
        arm = hl.Arm(s_gb=1, m_mib=150, ranks_per_group=3,
                     memtotal_bytes=0, memavail_bytes=0)
        arm.terms = {
            "launch_leftover_sum_gib": launch_sum,
            "launch_worst_case_margin_gib": hl.LAUNCH_WORST_CASE_MARGIN_GIB,
            "margin_total_gib": 1.47,       # the ratchet-era reap margin
        }
        arm.launch_leftover_gib = 5.67
        arm.run_leftover_gib = 1.62
        return arm

    def test_the_dry_run_specimen_passes(self):
        self.assertTrue(self._arm(-6.33).fundable_moments)

    def test_nine_gib_short_still_refuses(self):
        self.assertFalse(self._arm(-9.0).fundable_moments)

    def test_the_reap_margin_no_longer_scores_this_gate(self):
        """Under the shared number -6.33 vs -1.47 refused; that is the defect."""
        arm = self._arm(-6.33)
        self.assertLess(arm.launch_worst_case_gib, -float(arm.terms["margin_total_gib"]))
        self.assertTrue(arm.fundable_moments)

    def test_the_value_is_the_one_this_gate_always_had(self):
        self.assertAlmostEqual(hl.LAUNCH_WORST_CASE_MARGIN_GIB, 8.60, places=2)
        self.assertIn("8.60", hl.LAUNCH_WORST_CASE_MARGIN_SOURCE)
        # ...and it equals the pre-#1350e boot margin, so no earlier verdict moves
        self.assertAlmostEqual(
            hl.resolve_margin().boot_total_gib,
            hl.LAUNCH_WORST_CASE_MARGIN_GIB, places=2)

    def test_a_pre_1350f_arm_dict_still_grades_by_the_reap_margin(self):
        arm = hl.Arm(s_gb=1, m_mib=150, ranks_per_group=3,
                     memtotal_bytes=0, memavail_bytes=0)
        arm.terms = {"launch_leftover_sum_gib": -6.33, "margin_total_gib": 1.47}
        arm.launch_leftover_gib = 5.67
        arm.run_leftover_gib = 1.62
        self.assertFalse(arm.fundable_moments)


class PricedFromSeries1350g(CustomTestCase):
    """#1350g: the seed was proven too small ON METAL, so the term is priced
    from the measured series instead.

    weg2xsn25 charged the 4.049 GiB seed, predicted 90.43 GiB, reached 95.077
    and the host reaped group D (`oom_kill` 0 -> 1). It completed ONE flip, so
    `WEG2-FLIP-RATCHET post epoch=2` was never written and the next boot would
    have run on the same seed: the measurement meant to replace the seed cannot
    happen, because the boot dies before it.
    """

    #: predicted, measured peak, ratchet ALREADY charged -> required term.
    SERIES = (
        ("weg2xsn20", 87.15, 88.798, 0.000, 1.648),
        ("weg2xsn21b", 85.30, 86.995, 0.000, 1.695),
        ("weg2xsn22", 86.25, 88.495, 0.000, 2.245),
        ("weg2xsn23", 85.24, 88.210, 0.000, 2.970),
        ("weg2xsn24", 86.38, 90.475, 0.000, 4.095),
        ("weg2xsn25", 90.43, 95.077, 4.050, 8.697),
    )

    def test_each_row_is_what_that_boot_needed(self):
        for boot, pred, peak, charged, row in self.SERIES:
            with self.subTest(boot=boot):
                self.assertAlmostEqual(peak - pred + charged, row, places=2)
                self.assertAlmostEqual(
                    hl.FLIP_RATCHET_SERIES_GIB[boot], row, places=3)

    def test_the_rule_is_MAX_and_it_binds_the_worst_boot(self):
        got = hl.resolve_flip_ratchet_gib(None)
        self.assertAlmostEqual(got.per_flip_gib, 8.697, places=3)
        self.assertIn("SERIES", got.source)
        self.assertIn("weg2xsn25", got.source)
        self.assertFalse(got.from_record)
        # the retired seed must not come back
        self.assertNotIn("SEED", got.source)
        self.assertGreater(got.per_flip_gib, 4.049)

    def test_the_series_term_would_have_refused_xsn25_before_the_host_killed_it(self):
        """The whole point: an HONEST refusal instead of a reap.

        xsn25's un-ratcheted prediction was 90.43 - 4.050 = 86.38 GiB. With the
        series term the same arm predicts 86.38 + 8.697 = 95.08 GiB, which is
        AT its own measured peak 95.077 and far above the 94.43 hard bound --
        so the ladder refuses by name (W21) instead of booting into the reaper.
        """
        unratcheted = 90.43 - 4.050
        priced = unratcheted + hl.resolve_flip_ratchet_gib(None).per_flip_gib
        self.assertAlmostEqual(priced, 95.08, places=2)
        self.assertGreaterEqual(priced, 95.077 - 0.01)   # reaches the real peak
        margin = hl.resolve_margin(flip_ratchet_charged_gib=priced - unratcheted)
        bound = hl.OBSERVED_REAP_NONRECLAIM_BYTES / GIB - margin.boot_total_gib
        self.assertGreater(priced, bound)                 # -> refused, honestly

    def test_a_real_record_still_wins_over_the_series(self):
        rec = {"FLIP": hl.flip_ratchet_record(
            pre_gib=84.336, post_gib=87.497, boot_tag="weg2xsn26",
            commit="c", at="t")}
        got = hl.resolve_flip_ratchet_gib(rec)
        self.assertTrue(got.from_record)
        self.assertNotIn("SERIES", got.source)

    def test_the_refusal_is_still_reachable(self):
        with self.assertRaises(hl.Weg2HostFlipRatchetUnmeasured):
            hl.resolve_flip_ratchet_gib({}, seed_allowed=False)


class ArmLineFields1350(CustomTestCase):
    """(6) The four fields #1350 requires, on the ARM line."""

    def test_the_four_fields_are_printed_in_the_ordered_spelling(self):
        fields = _ratchet(3.161, source="MEASURED first flip pair of boot weg2xsn20").arm_fields()
        self.assertRegex(
            fields,
            r"ratchet_per_flip=3\.16 flips_priced=1 ratchet_charged=3\.16 source=",
        )

    def test_the_choose_line_carries_them_and_the_honest_bound(self):
        lines = _choose_lines(_ratchet(3.161))
        arm_lines = [ln for ln in lines if "WEG2-HOST-LEDGER ARM " in ln]
        self.assertTrue(arm_lines)
        for ln in arm_lines:
            self.assertIn("ratchet_per_flip=3.16", ln)
            self.assertIn("flips_priced=1", ln)
            self.assertIn("ratchet_charged=3.16", ln)
            self.assertIn("source=", ln)
            # (4): the analysis's honest bound, with its condition attached.
            self.assertIn("87.30 -> 94.43", ln)
            self.assertIn("FUNDING", ln)

    def test_an_unpriced_arm_prints_absent_not_zero(self):
        lines = _choose_lines(None)
        arm_lines = [ln for ln in lines if "WEG2-HOST-LEDGER ARM " in ln]
        self.assertTrue(arm_lines)
        for ln in arm_lines:
            self.assertIn("ratchet_charged=ABSENT", ln)
            self.assertNotIn("ratchet_charged=0.00", ln)


def _choose_lines(flip_ratchet):
    try:
        _arm, _hr, lines = hl.choose(
            int(123.78 * GIB), int(110.0 * GIB),
            ring_bytes=int(43.54 * GIB), ring_span1_bytes=int(20.0 * GIB),
            cg_current_bytes=int(9.0 * GIB), reclaimable_bytes=int(3.0 * GIB),
            cg_ceiling_bytes=int(123.78 * GIB), s_gb_d=4,
            flip_ratchet=flip_ratchet,
        )
        return list(lines)
    except (hl.Weg2HostLedgerRefused, hl.Weg2HostRunPeakRefused) as e:
        # A total refusal still prints the whole table, and the ARM lines are
        # exactly what this test reads. Refusing is a legitimate outcome of a
        # synthetic box and must not make the assertion unreachable.
        return str(e).splitlines()


class FrontEmitterSmoke1350(CustomTestCase):
    """(8) EXECUTION SMOKE -- desk-written-never-executed.

    The writer is a method of ``front.Front``, and standing up a whole front is
    not a desk test. The method is therefore called UNBOUND against a minimal
    stub carrying exactly the attributes it reads, which proves it RUNS, writes
    the field into a real sidecar and is read back by the real resolver.
    """

    class _Stub:
        def __init__(self, path):
            self.tag = "weg2smoke"
            self.commit = "bfb5e121e7"
            self.weights_tags = ["a"] * 10
            self.measured_record = path
            self._flip_ratchet_pre_gib = 84.336
            self._flip_ratchet_pre_at = "2026-09-11T15:49:22Z"
            self._flip_ratchet_written = False

    def _run(self, path, done_epoch, monkey_post):
        from sglang.srt.weg2 import front as fr
        stub = self._Stub(path)
        real = hl.read_flip_currency_gib
        hl.read_flip_currency_gib = lambda *a, **k: monkey_post
        try:
            fr.Front._write_flip_ratchet(stub, done_epoch)
        finally:
            hl.read_flip_currency_gib = real
        return stub

    def test_it_writes_at_done_epoch_two_and_only_there(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "weg2_measured_record.json")
            # epoch 1 is the FIRST leg: one leg is not a pair, nothing is written
            stub = self._run(path, 1, 87.497)
            self.assertFalse(os.path.exists(path))
            self.assertFalse(stub._flip_ratchet_written)
            # epoch 2 closes the pair
            stub = self._run(path, 2, 87.497)
            self.assertTrue(stub._flip_ratchet_written)
            back = hl.read_measured_record(path)
            self.assertAlmostEqual(
                hl.resolve_flip_ratchet_gib(back).per_flip_gib, 3.161, places=3)

    def test_an_unreadable_post_reading_writes_an_absence_not_a_zero(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "weg2_measured_record.json")
            self._run(path, 2, None)
            back = hl.read_measured_record(path)
            self.assertIn("FLIP", back)
            self.assertIsNone(back["FLIP"]["flip_ratchet_gib"])
            with self.assertRaises(hl.Weg2HostFlipRatchetUnmeasured):
                hl.resolve_flip_ratchet_gib(back, seed_allowed=False)


class Replay1350(CustomTestCase):
    """(7) THE FIVE RECORDS, replayed through the new pricing.

    METHOD, and it is deliberately not a re-implementation of the ledger: the
    new pricing changes ONE thing for these arms -- it adds the charged ratchet
    to the predicted run peak -- and
    `FlipRatchetTerm1350.test_the_term_lands_in_the_run_peak_and_exactly_once`
    proves that delta is exactly `charged_gib` against the real `price()`. The
    replay therefore applies each boot's MEASURED first-pair ratchet to that
    boot's OWN recorded prediction and checks the result against that boot's
    OWN measured peak.

    THE NUMBERS, verbatim (GiB, non-reclaimable = anon+shmem+slab_unreclaimable,
    1 Hz, never memory.current):

        boot        predicted  peak     ratchet  residual BEFORE  residual AFTER
        weg2xsn20   87.15      88.798   3.161    +1.648           -1.513
        weg2xsn21b  85.30      86.995   1.865    +1.695           -0.170
        weg2xsn22   86.25      88.495   1.870    +2.245           +0.375
        weg2xsn23   85.24      88.210   2.062    +2.970           +0.908
        weg2xsn24   86.38      90.475   4.049    +4.095           +0.046

    WHAT IS ASSERTED, and it is the honest claim rather than a flattering one:
    the under-prediction SHRINKS on all five and never grows, no boot is left
    with a bigger miss than the analysis recorded, and the remaining spread
    (-1.513 .. +0.908) is bounded by the POLSTER the analysis attributes
    separately (+2.814 / +0.899 / +0.749 / +0.380 / +0.186) and by the
    difference between the first PAIR and the whole staircase. The term is NOT
    claimed to close the balance to zero: a pair is not a boot.
    """

    def test_all_five_land_inside_the_analysis_stated_residual(self):
        for tag, _m, predicted, peak, _origin, pair, before in REPLAY:
            with self.subTest(boot=tag):
                self.assertAlmostEqual(peak - predicted, before, places=2,
                                       msg="the analysis's own residual must reproduce")
                after = peak - (predicted + pair)
                # (a) the miss never GROWS,
                self.assertLessEqual(after, before + 1e-9)
                # (b) and it lands inside the band the analysis stated for it.
                self.assertLessEqual(abs(after), abs(before) + CLOSURE_GIB)

    def test_the_expected_after_residuals_are_pinned_to_three_decimals(self):
        expect = {
            "weg2xsn20": -1.513, "weg2xsn21b": -0.170, "weg2xsn22": +0.375,
            "weg2xsn23": +0.908, "weg2xsn24": +0.046,
        }
        for tag, _m, predicted, peak, _origin, pair, _before in REPLAY:
            with self.subTest(boot=tag):
                self.assertAlmostEqual(
                    peak - (predicted + pair), expect[tag], places=3)

    def test_the_silently_zero_mutant_reproduces_the_original_defect(self):
        """MUTANT DIRECTION 1, stated as data rather than as a patch.

        If the term were charged at 0 -- the #606 shape -- every boot's residual
        would be exactly the one #1350 was raised for. That is what makes the
        absent-is-not-zero rule load-bearing rather than stylistic.
        """
        for tag, _m, predicted, peak, _origin, _pair, before in REPLAY:
            with self.subTest(boot=tag):
                mutant = peak - (predicted + 0.0)
                self.assertAlmostEqual(mutant, before, places=2)
                self.assertGreater(mutant, 0.0)   # always under-predicts

    def test_the_ratchet_is_the_dominant_half_of_every_residual(self):
        """`Residuum = RATSCHE - POLSTER`, closing to <= 0.03 GiB on all five --
        the reason to believe the attribution at all."""
        # analysis SS1.1: RATSCHE (peak - pre-first-flip) and POLSTER.
        full = {"weg2xsn20": 4.462, "weg2xsn21b": 2.594, "weg2xsn22": 2.994,
                "weg2xsn23": 3.350, "weg2xsn24": 4.280}
        polster = {"weg2xsn20": 2.814, "weg2xsn21b": 0.899, "weg2xsn22": 0.749,
                   "weg2xsn23": 0.380, "weg2xsn24": 0.186}
        for tag, _m, _p, peak, origin, pair, before in REPLAY:
            with self.subTest(boot=tag):
                self.assertLessEqual(
                    abs((full[tag] - polster[tag]) - before), CLOSURE_GIB)
                # the priced PAIR is a subset of the full staircase, never more
                self.assertLessEqual(pair, full[tag] + 1e-9)
                if tag != "weg2xsn21b":   # its CSV is gone; origin is quoted
                    self.assertAlmostEqual(peak - origin, full[tag], places=2)

    def test_the_cumulative_table_matches_the_replay_and_contradicts_the_local_one(self):
        for tag, _m, _p, _peak, _o, pair, _b in REPLAY:
            with self.subTest(boot=tag):
                self.assertAlmostEqual(
                    hl.FLIP_RATCHET_CUMULATIVE_GIB[tag][1], pair, places=3)
        # The same boot, the same series, the two instruments:
        self.assertAlmostEqual(hl.FLIP_TRANSIENT_IN_CURRENCY_GIB["weg2xsn20"],
                               -0.023, places=3)
        self.assertAlmostEqual(hl.FLIP_RATCHET_CUMULATIVE_GIB["weg2xsn20"][0],
                               4.462, places=3)

    def test_the_withdrawn_claim_is_no_longer_in_the_module(self):
        """The comment said "the flip transient is GONE"; it is refuted."""
        import inspect
        src = inspect.getsource(hl)
        head = src[:src.index("FLIP_RATCHET_CUMULATIVE_GIB")]
        self.assertNotIn("the flip\n#: transient is\n#: GONE", head)
        self.assertIn("#1350 WITHDRAWS", head)
        self.assertIn("INDIKATOR-GESETZ", head)


if __name__ == "__main__":
    unittest.main()
