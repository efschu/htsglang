# SPDX-License-Identifier: Apache-2.0
"""Weg 2 (#1233, fix 8): the dormant image is MEASURED, the run peak REFUSES.

Boot weg2dk7 (2026-09-08 00:19-00:32Z, `/spinning/gpu-arb/weg2/BOOT_weg2dk7_0907.md`)
measured three things this slice is built on, and each one refutes a term the
ledger was charging:

* with group P ASLEEP and group D AWAKE, the sum of P's per-rank ``RssShmem``
  was **38.63 GiB** in both attributed quiet rows (00:29:02Z, 00:31:07Z), while
  the ledger charged the #809 weight-tag census sum, **28.83 GiB** -- a +9.80
  GiB (+34 %) under-charge on ONE image.  44.56 GiB of the box's 48.33 GiB of
  cgroup shmem had NO backing file at all (the TMS CPU backup is anonymous
  ``MAP_SHARED``), so an attribution that looks for the image as a file finds
  nothing and concludes it is not resident;
* the boot IDLED at ``memory.current`` 90.10-94.57 GiB with the store EMPTY and
  ZERO flips, against a RUN-PEAK ADVISORY that predicted 91.92 GiB.  The
  advisory was already exceeded before the first request;
* the arm selector was ANTI-CORRELATED with safety: ``base`` comes from the
  LAUNCH-moment reading, so weg2dk7's quiet launch (15.36 GiB) bought the
  BIGGER arm (M=1200, 9 GiB store) and produced the tighter boot, while
  weg2dk6's loaded launch (46.80 GiB) took M=600 and died at 96.06 GiB.

Every number in this file is one of those measurements, a figure from
weg2dk5/dk6's own records, or derived from them by the module under test.

Hermetic: ``CUDA_VISIBLE_DEVICES=""``, no server, no NVML, no checkpoint, no
real ``/dev/shm`` and no real ``/proc`` -- the sweep is driven against fake
trees written per test.
"""

import inspect
import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger, launcher
from sglang.test.test_utils import CustomTestCase

GIB = host_ledger.GIB

# ---------------------------------------------------- MEASURED, boot weg2dk7
# BOOT_weg2dk7_0907.md, TERMS line and the attributed quiet rows.
DK7_MEMTOTAL_B = int(118.05 * GIB)
DK7_MEMAVAIL_B = int(104.02 * GIB)
DK7_CG_CURRENT_B = int(15.36 * GIB)
DK7_CG_RECLAIM_B = int(1.33 * GIB)          # -> non-reclaimable 14.03 GiB
DK7_IMAGE_P_GIB = 38.63                      # P's RssShmem sum, P asleep
DK7_IDLE_CURRENT_GIB = 90.10                 # quiet row 00:29:02Z, store EMPTY
# ---------------------------------------------------- MEASURED, boot weg2dk6
# BOOT_weg2dk6_0907.md, TERMS line and the death row.
DK6_MEMTOTAL_B = int(118.05 * GIB)
DK6_MEMAVAIL_B = int(102.25 * GIB)
DK6_CG_CURRENT_B = int(17.83 * GIB)
DK6_CG_RECLAIM_B = int(2.03 * GIB)          # -> non-reclaimable 15.80 GiB
DK6_DEATH_CURRENT_GIB = 96.06

CHUNKS = 8
#: C19 (ring rebase 0908): the host weights term is the measured per-card ring
#: table (Sigma H / Sigma image_P), not a chunk count.  Same figures the s3s4,
#: ring_ledger and host_budget suites pin.
RING_BYTES = 32964 * 1024 * 1024
RING_SPAN1_BYTES = 29912 * 1024 * 1024
RING_KW = dict(ring_bytes=RING_BYTES, ring_span1_bytes=RING_SPAN1_BYTES)
def _ladder(memtotal, memavail, current, reclaim, *, record=None, ceiling=None,
            arms=None):
    """Price the whole ladder at one box's readings.

    #1236: returns ``(arm, reap headroom, lines)``. The middle used to be the
    store size the ledger handed out; the store is on disk now and this
    function no longer sizes one.
    """
    return host_ledger.choose(
        memtotal,
        memavail,
        **({} if arms is None else {"arms": arms}),
        **RING_KW,
        cg_current_bytes=current,
        reclaimable_bytes=reclaim,
        cg_ceiling_bytes=memtotal if ceiling is None else ceiling,
        cg_ceiling_source="FALLBACK lxcfs MemTotal",
        cg_oom_kill=36,
        measured_record=record,
    )


def _fundable_ladder():
    """A box on which an arm IS fundable, run peak included.

    It is not this rig: ``ranks_per_group=1`` (three ranks of heap on each side
    is what makes the real shape unfundable, see the honest outcome below).  It
    exists so every refusal in this file has a control -- a gate that can only
    ever refuse pins nothing.  Readings chosen so the run leftover lands just
    above the store floor and the predicted peak just below the 95.90 GiB reap
    point; both are asserted where it is used, never assumed.
    """
    return host_ledger.choose(
        DK7_MEMTOTAL_B,
        DK7_MEMAVAIL_B,
        arms=((1, 600),),
        ranks_per_group=1,
        **RING_KW,
        cg_current_bytes=int(22.45 * GIB),
        reclaimable_bytes=0,
        cg_ceiling_bytes=DK7_MEMTOTAL_B,
        cg_ceiling_source="FALLBACK lxcfs MemTotal",
        cg_oom_kill=0,
    )


#: #1236: THE LADDER'S OUTCOME ON THESE BOXES MOVED, and it moved because of
#: this branch and not because a term was trimmed. Fix 8 refused weg2dk6 and
#: weg2dk7 outright; both boots carried a 9 GiB tmpfs page store, and every
#: arm's predicted peak carried those 9 GiB with it. The store is a directory
#: on the ZFS dataset now (#1236), so the same readings predict ~9 GiB lower
#: and the full ladder FUNDS S=1/M=1200 with 3.64 GiB of reap headroom -- which
#: is consistent with the metal rather than in spite of it: weg2dk6 died at
#: memory.current 96.06 GiB WITH its store resident, and 96.06 - 9 = 87.06 sits
#: at the hard bound.
#:
#: The refusal-TEXT assertions below are about the MESSAGE, not about dk7, so
#: they are pointed at the same box with the ladder restricted to its TOP arm
#: (M=2400), which still predicts above the hard bound and still raises W21.
#: The dk6-vs-dk7 SELECTOR claim is restated in the fundable direction in its
#: own test rather than deleted.
REFUSING_ARMS = ((1, 2400),)


def _refusal(**kw) -> str:
    """The ladder's refusal text, or a failure if it did NOT refuse."""
    try:
        _ladder(**kw)
    except (host_ledger.Weg2HostLedgerRefused, getattr(
            host_ledger, "Weg2HostRunPeakRefused", host_ledger.Weg2HostLedgerRefused)) as e:
        return str(e)
    raise AssertionError("the ladder funded an arm where the record says it must refuse")


# =====================================================================
# 1. THE IMAGE TERM COMES FROM A MEASUREMENT
# =====================================================================


class TestTheImageTermIsMeasuredNotSummedFromTags(CustomTestCase):
    def test_without_a_record_p_is_the_named_dk7_reading_and_says_whose(self):
        # Absent this line's own measurement, the term is another boot's
        # RECORDED reading -- charged, and labelled as not this boot's.
        it = host_ledger.resolve_image_terms(None)
        self.assertAlmostEqual(it.p_gib, DK7_IMAGE_P_GIB, delta=0.005)
        self.assertFalse(it.p_measured)
        self.assertIn("weg2dk7", it.p_source)
        self.assertIn("00:29:02Z", it.p_source)
        self.assertIn("RssShmem", it.p_source)

    def test_the_extra_term_is_the_measured_gap_to_the_weight_tag_census(self):
        # +9.80 GiB (+34 %): the census counted `weights_*`, the image carries
        # every enable_cpu_backup buffer.  Both halves stay visible.
        it = host_ledger.resolve_image_terms(None)
        self.assertAlmostEqual(
            it.extra_p_gib, DK7_IMAGE_P_GIB - host_ledger.WEIGHT_TAGS_P_BYTES / GIB, delta=0.005
        )
        self.assertAlmostEqual(it.extra_p_gib, 9.80, delta=0.01)

    def test_ds_image_is_a_bound_and_never_smaller_than_ps_measured_one(self):
        # D's dormant image has NEVER been measured (dk7's 6.71 GiB is D
        # AWAKE).  The bound refuses to be smaller than the one measurement
        # that exists, and it SAYS it is a bound.
        it = host_ledger.resolve_image_terms(None)
        candidate = host_ledger.WEIGHT_TAGS_D_BYTES / GIB + it.extra_p_gib
        self.assertGreater(it.d_gib, candidate)          # 38.63 > 36.94
        self.assertAlmostEqual(it.d_gib, it.p_gib, delta=1e-9)
        self.assertFalse(it.d_measured)
        self.assertIn("BOUND, NOT A MEASUREMENT", it.d_source)

    def test_this_lines_own_measurement_wins_over_the_recorded_one(self):
        # The point of the sidecar: a boot hands its successor a number, and
        # the successor charges THAT, labelled as measured, with its provenance.
        rec = {"P": {"rss_shmem_gib": 41.5, "boot_tag": "weg2dk9", "commit": "deadbeef01",
                     "at": "2026-09-08T02:00:00Z", "pids": [1, 2, 3, 4, 5]}}
        it = host_ledger.resolve_image_terms(rec)
        self.assertAlmostEqual(it.p_gib, 41.5, delta=1e-9)
        self.assertTrue(it.p_measured)
        self.assertIn("weg2dk9", it.p_source)
        self.assertIn("deadbeef01", it.p_source)
        # And D's bound follows the newer measurement, not the old constant.
        self.assertAlmostEqual(it.d_gib, 41.5, delta=1e-9)

    def test_the_ledger_reports_the_measured_image_and_charges_the_ring(self):
        # RENAMED ON THE RING (rebase 0908).  Fix 8's finding -- the image is
        # MEASURED (38.63) and is NOT the weight-tag census (28.83) -- is
        # unchanged and is still pinned here.  What moved is which of the two
        # numbers price() SUBTRACTS: C19 charges Sigma H once, sized from this
        # very measurement by ring_table, so charging images.p as well would
        # charge the same RssShmem bytes twice.  The image therefore appears in
        # `terms` as PROVENANCE (FLIPCOST A1-2 requires both numbers to print)
        # and `host_ring_gib` / `host_ring_span1_gib` are what the two moments
        # actually pay.
        arm = host_ledger.price(DK7_MEMTOTAL_B, DK7_MEMAVAIL_B, 1, 1200, **RING_KW)
        self.assertAlmostEqual(arm.terms["image_p_gib"], DK7_IMAGE_P_GIB, delta=0.005)
        self.assertNotIn("backup_resident_gib", arm.terms)
        self.assertAlmostEqual(arm.terms["host_ring_gib"], 32.19, delta=0.01)
        self.assertAlmostEqual(arm.terms["host_ring_span1_gib"], 29.21, delta=0.01)
        # The census sums are KEPT, and they are no longer the image.
        self.assertAlmostEqual(arm.terms["weight_tags_p_gib"], 28.83, delta=0.01)
        self.assertNotAlmostEqual(arm.terms["image_p_gib"], arm.terms["weight_tags_p_gib"], delta=1.0)

    def test_the_terms_line_prints_both_numbers_and_which_is_a_bound(self):
        _arm, _store, lines = _fundable_ladder()
        terms = [ln for ln in lines if "WEG2-HOST-LEDGER TERMS" in ln][0]
        self.assertIn("image_P=38.63 GiB", terms)
        self.assertIn("weight_tags_P=28.83 GiB", terms)
        self.assertIn("extra_P=9.80 GiB", terms)
        self.assertIn("BOUND, NOT A MEASUREMENT", terms)


class TestTheDormantImageSampler(CustomTestCase):
    """The instrument that produces the measurement, and its two denominators."""

    def test_the_line_carries_both_instruments_and_names_the_confounded_one(self):
        rec = host_ledger.dormant_image_sample(
            group="P", shmem_before_bytes=int(9.7 * GIB), shmem_after_bytes=int(48.33 * GIB),
            pids=[], weight_tags_gib=host_ledger.WEIGHT_TAGS_P_BYTES / GIB,
            interleaved=True, boot_tag="weg2dk8", commit="cafe", at="2026-09-08T01:00:00Z",
        )
        line = host_ledger.format_dormant_image(rec)
        self.assertIn("WEG2 DORMANT-IMAGE group=P", line)
        self.assertIn("shmem_delta_gib=", line)
        self.assertIn("rss_shmem_gib=", line)
        self.assertIn("weight_tags_gib=28.83", line)
        self.assertIn("extra_gib=", line)
        # A flip's sleep is interleaved: the delta is two images, not one.
        self.assertIn("INTERLEAVED", line)

    def test_an_uninterleaved_sample_says_the_two_instruments_are_comparable(self):
        rec = host_ledger.dormant_image_sample(
            group="P", shmem_before_bytes=0, shmem_after_bytes=0, pids=[],
            weight_tags_gib=1.0, interleaved=False, boot_tag="t", commit="c",
        )
        self.assertIn("un-interleaved", host_ledger.format_dormant_image(rec))

    def test_rss_shmem_sums_the_pids_that_answered_and_names_them(self):
        # The denominator law on a sum: the pid list IS the denominator, and a
        # pid that has exited is skipped, never counted as a zero.
        pid = os.getpid()
        total, seen = host_ledger.rss_shmem_bytes([pid, 999999999])
        self.assertEqual(seen, [pid])
        self.assertGreaterEqual(total, 0)

    def test_the_residual_is_none_with_a_reason_outside_the_run_moment(self):
        rec = host_ledger.dormant_image_sample(
            group="P", shmem_before_bytes=0, shmem_after_bytes=0, pids=[],
            weight_tags_gib=1.0, interleaved=False, boot_tag="t", commit="c",
        )
        self.assertIsNone(rec["run_residual_gib"])
        self.assertIn("not 0", rec["run_residual_note"])

    def test_the_sidecar_round_trips_and_the_newest_entry_per_group_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", host_ledger.MEASURED_RECORD_NAME)
            for at, val in (("2026-09-08T01:00:00Z", 30.0), ("2026-09-08T03:00:00Z", 39.5)):
                host_ledger.append_measured_record(path, {
                    "group": "P", "at": at, "rss_shmem_gib": val,
                    "boot_tag": "b", "commit": "c", "pids": [1],
                })
            rec = host_ledger.read_measured_record(path)
            self.assertAlmostEqual(rec["P"]["rss_shmem_gib"], 39.5, delta=1e-9)
            # Append-only: the older sample is still on disk as evidence.
            with open(path) as f:
                self.assertEqual(len(json.load(f)["samples"]), 2)

    def test_a_missing_or_broken_sidecar_is_an_absence_not_a_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(host_ledger.read_measured_record(os.path.join(tmp, "nope.json")), {})
            bad = os.path.join(tmp, "bad.json")
            with open(bad, "w") as f:
                f.write("{not json")
            self.assertEqual(host_ledger.read_measured_record(bad), {})
            # And the ledger then prices the RECORDED dk7 reading, not 0.
            self.assertAlmostEqual(
                host_ledger.resolve_image_terms(host_ledger.read_measured_record(bad)).p_gib,
                DK7_IMAGE_P_GIB, delta=0.005,
            )


# =====================================================================
# 2. THE RUN PEAK SELECTS THE ARM
# =====================================================================


class TestTheRunPeakRefusesInsteadOfAdvising(CustomTestCase):
    def test_an_arm_that_funds_both_moments_is_refused_by_its_run_peak(self):
        # A box with room to spare at BOTH moments: the ONLY term that can
        # refuse here is the run peak, so this test cannot pass by accident of
        # another gate. #1236: the ladder is restricted to its top arm, because
        # with the store off the RAM books the lower arms of this box now fit
        # (see REFUSING_ARMS) -- the gate under test is the run peak, not which
        # arm happens to trip it.
        with self.assertRaises(host_ledger.Weg2HostRunPeakRefused) as cm:
            _ladder(int(200 * GIB), int(190 * GIB), int(20 * GIB), 0,
                    arms=REFUSING_ARMS)
        msg = str(cm.exception)
        self.assertIn("W21 Weg2HostRunPeakRefused", msg)
        self.assertIn("RUN PEAK", msg)
        self.assertIn("95.90", msg)

    def test_a_box_whose_peak_fits_still_funds_its_arm(self):
        # The control that keeps the refusal from being a tautology: a gate
        # that can only refuse pins nothing.
        arm, headroom, lines = _fundable_ladder()
        self.assertEqual((arm.s_gb, arm.m_mib), (1, 600))
        # #1236: the middle of the tuple is the REAP HEADROOM, not a store size.
        self.assertGreater(headroom, 0.0)
        peak = arm.predicted_run_peak_gib()
        self.assertLess(peak, host_ledger.OBSERVED_REAP_NONRECLAIM_BYTES / GIB)
        self.assertTrue(any("FUNDABLE" in ln for ln in lines))

    def test_every_arm_line_carries_its_predicted_peak_and_binding_term(self):
        msg = _refusal(memtotal=DK7_MEMTOTAL_B, memavail=DK7_MEMAVAIL_B,
                       current=DK7_CG_CURRENT_B, reclaim=DK7_CG_RECLAIM_B,
                       arms=REFUSING_ARMS)
        self.assertIn("run_peak=", msg)
        self.assertIn("binding:", msg)
        self.assertIn("RUN PEAK", msg)

    def test_the_host_weights_term_is_in_the_predicted_peak_exactly_once(self):
        # RE-DERIVED ON THE RING (rebase 0908).  This test used to pin the flip
        # transient as a real SUMMAND of the peak rather than a number printed
        # beside it.  That intent is exactly what still needs pinning -- a term
        # silently dropped from (or double-counted in) this sum is the defect
        # class -- but the term itself is now Sigma H: C19 removed the transient
        # by preallocating the region and copying the legs through it, so the
        # peak charges the host weights ONCE and 9.97 is no longer a term.
        arm = host_ledger.price(
            int(200 * GIB), int(190 * GIB), 1, 600, **RING_KW,
            cg_current_bytes=int(2 * GIB), reclaimable_bytes=0,
            cg_ceiling_bytes=int(200 * GIB),
        )
        peak = arm.predicted_run_peak_gib()
        origin = arm.terms["run_origin_gib"]
        self.assertAlmostEqual(
            peak,
            origin + host_ledger._boot_charges_gib(arm.terms)
            + arm.terms["host_ring_gib"],
            delta=1e-6,
        )
        # ONCE, not twice: the measured image sizes the ring, it is not a second
        # charge beside it (ring fix 1 finding 3).
        self.assertLess(peak, origin + host_ledger._boot_charges_gib(arm.terms)
                        + arm.terms["host_ring_gib"] + arm.terms["image_p_gib"])


class TestTheOriginIsTheRunMomentNotTheLaunchMoment(CustomTestCase):
    def test_the_launch_reading_is_only_a_floor(self):
        # Two readings 30 GiB apart, both BELOW the measured run residual: the
        # predicted peak is identical, because what the box holds at the run
        # moment does not depend on how quiet it was at launch.
        a = host_ledger.price(
            DK7_MEMTOTAL_B, DK7_MEMAVAIL_B, 1, 600, **RING_KW,
            cg_current_bytes=int(2 * GIB), reclaimable_bytes=0,
            cg_ceiling_bytes=DK7_MEMTOTAL_B,
        )
        b = host_ledger.price(
            DK7_MEMTOTAL_B, DK7_MEMAVAIL_B, 1, 600, **RING_KW,
            cg_current_bytes=int(14 * GIB), reclaimable_bytes=0,
            cg_ceiling_bytes=DK7_MEMTOTAL_B,
        )
        self.assertAlmostEqual(a.terms["run_origin_gib"], b.terms["run_origin_gib"], delta=1e-9)
        self.assertAlmostEqual(
            a.predicted_run_peak_gib(), b.predicted_run_peak_gib(), delta=1e-9
        )
        self.assertIn("RUN-MOMENT RESIDUAL FLOOR", a.terms["run_origin_source"])

    def test_a_loaded_box_still_charges_what_it_holds(self):
        # The floor is a floor, not a replacement: above it, the live reading
        # binds and says so.
        arm = host_ledger.price(
            DK7_MEMTOTAL_B, DK7_MEMAVAIL_B, 1, 600, **RING_KW,
            cg_current_bytes=int(60 * GIB), reclaimable_bytes=0,
            cg_ceiling_bytes=DK7_MEMTOTAL_B,
        )
        self.assertAlmostEqual(arm.terms["run_origin_gib"], 60.0, delta=0.01)
        self.assertIn("launch-moment", arm.terms["run_origin_source"])

    def test_the_residual_floor_is_derived_from_dk7s_own_measurement(self):
        # DERIVED, not a constant: dk7's quiet reading minus the charges of the
        # arm it ran minus the measured image minus the store's measured
        # content (0.00 GiB -- the store was empty).
        images = host_ledger.resolve_image_terms(None)
        charges = host_ledger.charge_terms(1, 1200, 3, images)
        expect = (
            DK7_IDLE_CURRENT_GIB
            - host_ledger._boot_charges_gib(charges)
            - DK7_IMAGE_P_GIB
        )
        self.assertAlmostEqual(host_ledger.dk7_run_residual_gib(), expect, delta=1e-9)
        # And it is the size the record names: 90.10 idle, ~68.7 of it charged.
        self.assertAlmostEqual(host_ledger.dk7_run_residual_gib(), 21.38, delta=0.01)

    def test_dk7s_quiet_launch_no_longer_buys_a_bigger_arm_than_dk6s(self):
        # THE TWO-POINT DEMONSTRATION, re-run under #1236.  At fix-7 pricing
        # weg2dk7 (launch 15.36 GiB) took S=1 M=1200 with a 9 GiB store while
        # weg2dk6 (launch 46.80 GiB) took M=600 -- the quieter launch bought the
        # bigger arm and produced the tighter boot.  Fix 8 made both REFUSE.
        # #1236 makes both FUND THE SAME ARM, which is this test's claim in its
        # cleanest form yet: the selector cannot reward a quiet launch, because
        # both readings sit below the measured run-moment residual and get the
        # SAME origin.  What changed is only that the outcome is now an arm
        # rather than a refusal -- the 9 GiB store left the peak.
        dk7_arm, dk7_head, _ = _ladder(
            DK7_MEMTOTAL_B, DK7_MEMAVAIL_B, DK7_CG_CURRENT_B, DK7_CG_RECLAIM_B)
        dk6_arm, dk6_head, _ = _ladder(
            DK6_MEMTOTAL_B, DK6_MEMAVAIL_B, DK6_CG_CURRENT_B, DK6_CG_RECLAIM_B)
        self.assertEqual((dk7_arm.s_gb, dk7_arm.m_mib), (dk6_arm.s_gb, dk6_arm.m_mib))
        self.assertAlmostEqual(dk7_head, dk6_head, delta=1e-9)
        # Same origin, so the selector can no longer reward a quiet launch.
        a = host_ledger.price(DK7_MEMTOTAL_B, DK7_MEMAVAIL_B, 1, 600, **RING_KW,
                              cg_current_bytes=DK7_CG_CURRENT_B, reclaimable_bytes=DK7_CG_RECLAIM_B,
                              cg_ceiling_bytes=DK7_MEMTOTAL_B)
        b = host_ledger.price(DK6_MEMTOTAL_B, DK6_MEMAVAIL_B, 1, 600, **RING_KW,
                              cg_current_bytes=DK6_CG_CURRENT_B, reclaimable_bytes=DK6_CG_RECLAIM_B,
                              cg_ceiling_bytes=DK6_MEMTOTAL_B)
        self.assertAlmostEqual(a.terms["run_origin_gib"], b.terms["run_origin_gib"], delta=1e-9)
        self.assertGreaterEqual(
            b.predicted_run_peak_gib(), a.predicted_run_peak_gib() - 1e-9
        )


# =====================================================================
# 3. THE HONEST OUTCOME
# =====================================================================


class TestTheHonestOutcomeIsPrintedInsteadOfShrinkingATerm(CustomTestCase):
    def test_the_refusal_names_the_ring_slice_as_the_term_that_has_to_move(self):
        msg = _refusal(memtotal=DK7_MEMTOTAL_B, memavail=DK7_MEMAVAIL_B,
                       current=DK7_CG_CURRENT_B, reclaim=DK7_CG_RECLAIM_B,
                       arms=REFUSING_ARMS)
        # RING REBASE 0908: fix 8 wrote this refusal while the ring was still
        # a FUTURE slice ("the flip transient is removed by the host ring slice
        # (weg2/ring-0907)").  The ring has since landed and is the base of this
        # branch, so the sentence that named it as pending would now be false.
        # The refusal still names the ring -- as the term that HAS moved, and as
        # the remaining lever -- which is the same job, told in the present.
        self.assertIn("no arm funds a flip on this host budget", msg)
        self.assertIn("the host ring landed (C19)", msg)
        self.assertIn("there is no transient left to cut", msg)
        self.assertIn("cut Sigma H itself", msg)
        self.assertIn("ring_table.solve", msg)

    def test_no_term_was_shrunk_to_get_an_arm_through(self):
        # The three terms the refusal would be tempting to trim.  Each is a
        # measurement with a named boot; the refusal quotes them rather than
        # moving them.
        self.assertAlmostEqual(host_ledger.FLIP_HOST_TRANSIENT_GIB, 9.97, delta=1e-9)
        self.assertAlmostEqual(host_ledger.LOAD_TRANSIENT_GIB, 12.0, delta=1e-9)
        self.assertAlmostEqual(host_ledger.FLOOR_GIB, 16.0, delta=1e-9)
        self.assertAlmostEqual(
            host_ledger.resolve_image_terms(None).p_gib, DK7_IMAGE_P_GIB, delta=0.005
        )

    def test_the_whole_ladder_is_printed_with_the_refusal(self):
        msg = _refusal(memtotal=DK7_MEMTOTAL_B, memavail=DK7_MEMAVAIL_B,
                       current=DK7_CG_CURRENT_B, reclaim=DK7_CG_RECLAIM_B,
                       arms=REFUSING_ARMS)
        for m in (2400, 1200, 600):
            self.assertIn(f"M={m}", msg)


# =====================================================================
# 4. THE STALE /dev/shm ORPHAN SWEEP
# =====================================================================


class _Log:
    def __init__(self):
        self.lines = []

    def __call__(self, msg):
        self.lines.append(msg)


def _fake_shm(tmp: str) -> str:
    """A /dev/shm with one dead page store, one dead flag file, and foreigners."""
    shm = os.path.join(tmp, "shm")
    os.makedirs(os.path.join(shm, "hicache-weg2-fix973", "0a"), exist_ok=True)
    with open(os.path.join(shm, "hicache-weg2-fix973", "0a", "page.bin"), "wb") as f:
        f.write(b"x" * 4096)
    os.makedirs(os.path.join(shm, "sglang-phase-flip-presence"), exist_ok=True)
    with open(os.path.join(shm, ".weg2-pcie-serialize-GPU-abc.lock"), "w") as f:
        f.write("")
    # FOREIGN: other people's processes live here too.
    with open(os.path.join(shm, "sem.mp-deadbeef"), "wb") as f:
        f.write(b"y" * 32)
    return shm


def _fake_proc(tmp: str, pid: int = 4242, mapped: str = "") -> str:
    proc = os.path.join(tmp, "proc")
    os.makedirs(os.path.join(proc, str(pid)), exist_ok=True)
    with open(os.path.join(proc, str(pid), "maps"), "w") as f:
        if mapped:
            f.write(f"7f0000000000-7f0000001000 rw-s 00000000 00:19 12345 {mapped}\n")
        else:
            f.write("7f0000000000-7f0000001000 rw-p 00000000 00:00 0 \n")
    return proc


class TestTheShmOrphanSweep(CustomTestCase):
    def test_a_dead_boots_page_store_is_swept_and_its_bytes_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            shm, proc = _fake_shm(tmp), _fake_proc(tmp)
            log = _Log()
            out = launcher.shm_residue_sweep(
                log, "weg2dk8", "0908_010203", False, shm_dir=shm, proc_root=proc,
                archive_root=os.path.join(tmp, "archive"),
            )
            self.assertIn("hicache-weg2-fix973", out["swept"])
            # ALLOCATED bytes, not apparent: on the real orphan the two differ
            # by 0.53 GiB (4,042,810,432 apparent vs 3,471,384,576 allocated,
            # measured 2026-09-08) and only the allocated half is host RAM.
            # ALLOCATED is filesystem-dependent (this fake tree is not a
            # tmpfs), so what is pinned is that it is reported and non-zero;
            # the APPARENT size is the 4096 B this test wrote.
            self.assertGreater(out["bytes_freed"], 0)
            self.assertGreaterEqual(out["bytes_apparent"], 4096)
            # The bytes are actually gone from the fake tmpfs.
            self.assertFalse(os.path.exists(os.path.join(shm, "hicache-weg2-fix973")))
            # A manifest is left behind: what existed, how big, when.
            with open(os.path.join(out["archive"], "MANIFEST.json")) as f:
                names = {e["name"] for e in json.load(f)["entries"]}
            self.assertIn("hicache-weg2-fix973", names)
        self.assertTrue(any("allocated bytes" in ln and "freed" in ln for ln in log.lines))

    def test_foreign_names_are_never_touched(self):
        with tempfile.TemporaryDirectory() as tmp:
            shm, proc = _fake_shm(tmp), _fake_proc(tmp)
            out = launcher.shm_residue_sweep(
                _Log(), "t", "s", False, shm_dir=shm, proc_root=proc,
                archive_root=os.path.join(tmp, "archive"),
            )
            self.assertTrue(os.path.exists(os.path.join(shm, "sem.mp-deadbeef")))
            self.assertNotIn("sem.mp-deadbeef", out["swept"])

    def test_a_live_holder_refuses_the_boot_and_sweeps_nothing(self):
        # THE RULE THAT MAKES THE SWEEP SAFE: a mapped entry is never moved,
        # never deleted and nothing is killed -- the BOOT refuses instead, with
        # the entry and the holding pid named.
        with tempfile.TemporaryDirectory() as tmp:
            shm = _fake_shm(tmp)
            proc = _fake_proc(tmp, pid=4242, mapped=os.path.join(shm, "hicache-weg2-fix973", "0a", "page.bin"))
            with self.assertRaises(launcher.Weg2LaunchRefused) as cm:
                launcher.shm_residue_sweep(
                    _Log(), "t", "s", False, shm_dir=shm, proc_root=proc,
                    archive_root=os.path.join(tmp, "archive"),
                )
            self.assertIn("LIVE HOLDER", str(cm.exception))
            self.assertIn("4242", str(cm.exception))
            self.assertTrue(os.path.exists(os.path.join(shm, "hicache-weg2-fix973")))

    def test_the_holder_match_has_a_boundary_and_is_not_a_substring(self):
        with tempfile.TemporaryDirectory() as tmp:
            shm = _fake_shm(tmp)
            os.makedirs(os.path.join(shm, "hicache-weg2-fix9730"), exist_ok=True)
            proc = _fake_proc(tmp, pid=77, mapped=os.path.join(shm, "hicache-weg2-fix9730"))
            self.assertEqual(
                launcher.shm_holder_pids(os.path.join(shm, "hicache-weg2-fix973"), proc), []
            )
            self.assertEqual(
                launcher.shm_holder_pids(os.path.join(shm, "hicache-weg2-fix9730"), proc), [77]
            )

    def test_a_deleted_but_mapped_region_still_counts_as_a_holder(self):
        with tempfile.TemporaryDirectory() as tmp:
            shm = _fake_shm(tmp)
            target = os.path.join(shm, "hicache-weg2-fix973")
            proc = _fake_proc(tmp, pid=99, mapped=f"{target}/0a/page.bin (deleted)")
            self.assertEqual(launcher.shm_holder_pids(target, proc), [99])

    def test_a_dry_run_touches_nothing_and_still_reports_the_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            shm, proc = _fake_shm(tmp), _fake_proc(tmp)
            out = launcher.shm_residue_sweep(
                _Log(), "t", "s", True, shm_dir=shm, proc_root=proc,
                archive_root=os.path.join(tmp, "archive"),
            )
            self.assertGreater(out["bytes_freed"], 0)
            self.assertGreaterEqual(out["bytes_apparent"], 4096)
            self.assertTrue(os.path.exists(os.path.join(shm, "hicache-weg2-fix973")))

    def test_the_preflight_calls_the_sweep(self):
        # A sweep main() does not call is a sweep nothing runs (the fix-7 class).
        src = inspect.getsource(launcher)
        main_src = src[src.index("\ndef main("):]
        self.assertIn("shm_residue_sweep(log, ns.tag, stamp, dry)", main_src)


# =====================================================================
# 5. THE FIX-7 OPEN ITEMS, CARRIED
# =====================================================================


class TestTheCarriedOpenItems(CustomTestCase):
    def test_the_parameter_no_longer_shadows_the_module_function(self):
        for fn in (host_ledger.price, host_ledger.choose):
            params = inspect.signature(fn).parameters
            self.assertNotIn("cg_reclaimable_bytes", params)
            self.assertIn("reclaimable_bytes", params)
        # The function of that name is still the one the docstrings mean.
        self.assertTrue(callable(host_ledger.cg_reclaimable_bytes))

    def test_the_unevictable_residual_is_named_in_the_terms_line(self):
        _arm, _store, lines = _fundable_ladder()
        terms = [ln for ln in lines if "WEG2-HOST-LEDGER TERMS" in ln][0]
        self.assertIn("UNEVICTABLE LRU", terms)
        self.assertIn("16,384 B", terms)          # cgroup unevictable, measured
        self.assertIn("309,682,176 B", terms)     # /proc/meminfo Mlocked, measured
        self.assertIn("UNPRICED RESIDUAL", terms)

    def test_the_advisory_carries_all_three_boots_residuals_and_the_dk7_cause(self):
        _arm, _store, lines = _fundable_ladder()
        advisory = [ln for ln in lines if "RUN-PEAK ADVISORY" in ln][0]
        self.assertIn("3.01", advisory)            # dk5
        self.assertIn("4.83", advisory)            # dk6, sampled peak
        self.assertIn("7.67", advisory)            # dk6, kernel peak
        self.assertIn("38.63", advisory)           # dk7's explanation of both
        self.assertIn("9.80", advisory)


if __name__ == "__main__":
    unittest.main()
