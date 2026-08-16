"""#677 phase 1: a carrier awaiting its decode window must stop COUNTING.

THE WEDGE WAS A COUNTING DEFECT, and the measurement is what says so. At
2026-08-16 06:04 the instance held:

    max_mamba_cache_size  12        <- 8 slots FREE
    max_running_requests   4        <- the cap, fully occupied by carriers
    pending prefill   403779 tok    <- frozen, nothing admissible

Admission is `min(pp_max_micro_batch_size, admission_limiter.current) -
running_bs`, then `min(..., req_to_token_pool.available_size())`.
`HybridReqToTokenPool` does not override `available_size`, so that second term
is the REQUEST-slot count; the mamba allocator is consulted only later, inside
`alloc_req_slots`. Neither term of the gate sees the GDN pool. With
`running_bs == 4 == cap` the gate returned 0 and nothing further was reached.

So freeing a GDN slot would not have unblocked anything -- eight were already
free. What blocked admission was that four requests which PP is FORBIDDEN to
decode (strict purity, `decode_allowed_in_pp` is False) were nonetheless
counted against the concurrency cap the whole time.

PHASE 1 CHANGES ONLY THE ARITHMETIC. Nothing moves: the carrier keeps its GDN
slot and its KV, which is exactly the KV that would have been resident anyway.
That is deliberate -- no state movement means no new correctness surface from
the #450/#444 verify-write family, the #461 DEVICE_BOUND law, or #551
GDN-Vacate x kvso. The blob park is phase 2 and is gated behind the #551 read.

EVERY BOUND IS SOLVED FROM BOOT DIMENSIONING, never a free parameter:

  * ``parked + running <= slot_pool`` (12). A prefill that would exceed the
    GDN slot pool is refused EARLY and by name, because `alloc_req_slots`
    would refuse it late anyway -- and a late refusal is the raise this chain
    has spent three tasks making survivable.
  * ``running_bs <= max_running_requests`` (4) at ALL times, TP included. At
    TP entry the parked set re-admits in capture-set-sized batches; the rest
    stay parked and re-admit as decodes complete. Every pool therefore stays
    inside the dimensioning it was built for -- phase 1 raises nothing.
"""

import unittest

from sglang.srt.managers import parked_decode_set as pds
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

SLOT_POOL = 12
MAX_RUNNING = 4


def _set(*ids):
    s = pds.ParkedDecodeSet(slot_pool=SLOT_POOL, max_running=MAX_RUNNING)
    for rid in ids:
        s.park(rid, running_bs=MAX_RUNNING)
    return s


class TheWedgeCannotForm(unittest.TestCase):
    def test_four_parked_carriers_leave_the_full_cap_free_for_prefill(self):
        """THE 06:04 SCENARIO. Four carriers park; admission recovers."""
        s = _set("a", "b", "c", "d")
        self.assertEqual(4, len(s))
        # running_bs no longer counts them, so the whole cap is admissible...
        self.assertEqual(
            MAX_RUNNING, s.admission_headroom(running_bs=0, requested=MAX_RUNNING)
        )
        # ...and the fifth prefill the wedge refused now has room.
        self.assertGreater(s.admission_headroom(running_bs=0, requested=1), 0)

    def test_the_caller_gate_is_what_the_parking_opens(self):
        """THE CONTROL, and it states the composition exactly.

        ``admission_headroom`` is an ADDITIONAL bound (the GDN slot pool), not
        a replacement for the scheduler's own ``limit - running_bs``. Parking
        does not bypass the concurrency cap -- it removes the carrier from
        ``running_bs`` so the scheduler's own gate opens again.

        Carriers counted (the wedge): the caller's gate is 0 and nothing this
        module returns can help. Carriers parked: the caller's gate is the
        full cap, and the slot pool is what then bounds it.
        """
        s = _set("a", "b", "c", "d")

        def caller_gate(running_bs):
            return max(0, MAX_RUNNING - running_bs)

        self.assertEqual(0, caller_gate(MAX_RUNNING), "the wedge, as it was")
        self.assertEqual(MAX_RUNNING, caller_gate(0), "after parking the four")
        self.assertEqual(
            MAX_RUNNING,
            min(caller_gate(0), s.admission_headroom(running_bs=0, requested=99)),
            "and the slot pool leaves room for the whole cap here",
        )


class TheSlotPoolIsTheHonestCapacityLimit(unittest.TestCase):
    def test_parked_plus_running_never_exceeds_the_slot_pool(self):
        s = pds.ParkedDecodeSet(slot_pool=SLOT_POOL, max_running=MAX_RUNNING)
        for i in range(SLOT_POOL):
            running = min(MAX_RUNNING, SLOT_POOL - len(s))
            head = s.admission_headroom(running_bs=running, requested=SLOT_POOL)
            self.assertLessEqual(len(s) + running + head, SLOT_POOL)
            if head <= 0:
                break
            s.park(f"r{i}", running_bs=running)
        self.assertLessEqual(len(s), SLOT_POOL)

    def test_a_full_slot_pool_refuses_EARLY_and_by_name(self):
        s = pds.ParkedDecodeSet(slot_pool=SLOT_POOL, max_running=MAX_RUNNING)
        for i in range(SLOT_POOL - MAX_RUNNING):
            s.park(f"r{i}", running_bs=MAX_RUNNING)
        self.assertEqual(0, s.admission_headroom(running_bs=MAX_RUNNING, requested=1))
        self.assertIn("slot pool", s.last_refusal.lower())
        self.assertIn(str(SLOT_POOL), s.last_refusal)

    def test_parking_beyond_the_slot_pool_is_refused_not_silently_dropped(self):
        s = pds.ParkedDecodeSet(slot_pool=2, max_running=1)
        self.assertTrue(s.park("a", running_bs=1))
        with self.assertRaises(pds.ParkedSetFull):
            s.park("b", running_bs=1)


class TpReadmissionRespectsTheCap(unittest.TestCase):
    def test_it_never_exceeds_max_running_concurrent_decodes(self):
        s = _set("a", "b", "c", "d", "e", "f")
        plan = s.readmit_plan(running_bs=0)
        self.assertEqual(MAX_RUNNING, len(plan))
        self.assertLessEqual(len(plan), MAX_RUNNING)

    def test_it_tops_up_only_the_free_part_of_the_cap(self):
        s = _set("a", "b", "c", "d")
        self.assertEqual(1, len(s.readmit_plan(running_bs=MAX_RUNNING - 1)))
        self.assertEqual(0, len(s.readmit_plan(running_bs=MAX_RUNNING)))

    def test_the_rest_stay_parked_and_re_admit_as_decodes_complete(self):
        s = _set("a", "b", "c", "d", "e", "f")
        first = s.readmit(running_bs=0)
        self.assertEqual(MAX_RUNNING, len(first))
        self.assertEqual(2, len(s), "the overflow must remain parked")
        second = s.readmit(running_bs=MAX_RUNNING - 2)
        self.assertEqual(2, len(second))
        self.assertEqual(0, len(s))

    def test_re_admission_is_fifo_so_a_carrier_cannot_starve(self):
        s = _set("a", "b", "c", "d", "e")
        self.assertEqual(["a", "b", "c", "d"], s.readmit(running_bs=0))


class TheReceiptsNameTheBindingBound(unittest.TestCase):
    def test_a_park_logs_the_id_the_set_size_and_the_bound(self):
        s = pds.ParkedDecodeSet(slot_pool=SLOT_POOL, max_running=MAX_RUNNING)
        s.park("req-7", running_bs=MAX_RUNNING)
        line = s.last_receipt
        self.assertIn("req-7", line)
        self.assertIn("parked", line.lower())
        self.assertIn(str(SLOT_POOL), line, "the binding bound must be named")

    def test_a_re_admission_logs_the_same_three_facts(self):
        s = _set("req-9")
        s.readmit(running_bs=0)
        line = s.last_receipt
        self.assertIn("req-9", line)
        self.assertIn("re-admit", line.lower())
        self.assertIn(str(MAX_RUNNING), line)


class TheSafetyNetIsUnderneath(unittest.TestCase):
    """(d) A park failure must degrade to the #677 progress exit, never to a
    wedge. Expressed as: with parking DISABLED the arithmetic is byte-identical
    to the pre-change gate, so the wedge path -- and therefore the exit that
    now breaks it -- is exactly what it was."""

    def test_disabled_parking_reproduces_the_old_gate_exactly(self):
        s = pds.ParkedDecodeSet(slot_pool=SLOT_POOL, max_running=MAX_RUNNING,
                                enabled=False)
        self.assertFalse(s.park("a", running_bs=MAX_RUNNING))
        self.assertEqual(0, len(s))
        self.assertEqual(0, s.admission_headroom(running_bs=MAX_RUNNING, requested=4))

    def test_a_refused_park_leaves_the_request_countable(self):
        """The caller must be able to tell that the request stays a carrier,
        so it keeps counting and the progress exit still sees the wedge."""
        s = pds.ParkedDecodeSet(slot_pool=2, max_running=1)
        s.park("a", running_bs=1)
        with self.assertRaises(pds.ParkedSetFull):
            s.park("b", running_bs=1)
        self.assertNotIn("b", s.ids)


class NothingIsStranded(unittest.TestCase):
    """(e) A parked set is process-local: a crash loses it. So the shutdown
    path must hand every parked request back to be failed LOUDLY rather than
    leaving it to look admitted and never complete."""

    def test_evacuate_returns_every_parked_request(self):
        s = _set("a", "b", "c")
        out = s.evacuate("shutdown")
        self.assertEqual(["a", "b", "c"], out)
        self.assertEqual(0, len(s))

    def test_evacuate_says_why_and_leaves_nothing_behind(self):
        s = _set("a")
        s.evacuate("crash-restore")
        self.assertIn("crash-restore", s.last_receipt)
        self.assertEqual([], s.ids)

    def test_a_parked_request_is_visible_to_bookkeeping(self):
        """THE ONE BOOKKEEPING EDGE. A parked request holds its KV and its
        req_to_token row, so it must be neither double-counted nor invisible:
        pressure ladders and retract paths ask for the resident set, and a
        parked carrier IS resident."""
        s = _set("a", "b")
        self.assertEqual(["a", "b"], s.resident_ids)
        self.assertEqual(2, s.resident_count)


if __name__ == "__main__":
    unittest.main()
