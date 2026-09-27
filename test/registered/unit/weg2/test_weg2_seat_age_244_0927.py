# SPDX-License-Identifier: Apache-2.0
"""SA (#244 rebuilt to the user's design): one priority by FRONT ARRIVAL (the rid's counter) over
running, parked and waiting requests; seats by age with KV backfill; the next older moves in when a
seat frees and displaces the youngest running one (whole park); every park victim is the youngest.
User: "der request der zuletzt ankam wird zuletzt bedient, außer er passt zufällig in einen sitz ...
sobald dann der älteste request fertig ist, rückt der zweitälteste nach ... und würde ggf. jüngere
aktuell noch laufende requests verdrängen ganz oder teilweise"."""

import asyncio
import collections
import os
import time
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from sglang.srt.weg2 import d_park_runtime as DPR  # noqa: E402
from sglang.srt.weg2 import d_seats as DS  # noqa: E402
from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.srt.weg2 import seat_age as SA  # noqa: E402


def R(n, e=12):
    return f"weg2-{e}-{n}"


class Order(unittest.TestCase):
    def test_rid_age_is_the_front_arrival_counter(self):
        self.assertEqual(SA.rid_age("weg2-16-46"), 46)
        self.assertLess(SA.rid_age(R(39, 12)), SA.rid_age(R(44, 13)))  # epoch does not matter
        self.assertEqual(SA.rid_age("other"), SA.UNKNOWN_AGE)

    def test_plan_oldest_first_and_backfill(self):
        # seats 3, KV 1000: the 2nd-oldest needs 900 and does not fit after the oldest (400)
        cands = [(R(1), 400), (R(2), 900), (R(3), 100), (R(4), 300), (R(5), 50)]
        self.assertEqual(SA.plan_seats(cands, 3, budget_tokens=1000), [R(1), R(3), R(4)])
        self.assertEqual(SA.plan_seats(cands, 3), [R(1), R(2), R(3)])

    def test_no_younger_ever_displaces_an_older(self):
        self.assertIsNone(SA.displace_victim([R(9)], [R(2), R(5)], True))
        self.assertEqual(SA.displace_victim([R(3)], [R(2), R(5), R(7)], True), (R(3), R(7)))
        self.assertIsNone(SA.displace_victim([R(3)], [R(2), R(5)], False), "a free seat: no displacement")


class FrontWakePlan(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()

    def tearDown(self):
        self.loop.close()

    def _front(self, d_bs):
        f = object.__new__(F.Front)
        f.epoch, f.d_bs = 18, d_bs
        f.counters = collections.Counter()
        f._d_seat = asyncio.Semaphore(0)
        f._d_seats_live = set()
        f._ready_for_d = collections.deque()
        return f

    def _w(self, rid):
        p = F.Pending(rid, "/v1/messages", {}, "t", time.time() - 300, self.loop.create_future(),
                      est_prompt=1000, est_uncached=1000, span_known=True)
        p.leg1_done, p.t_ready = True, time.time() - 250
        return p

    def test_older_waiters_take_the_seats_of_the_younger_parked(self):
        f = self._front(d_bs=3)
        parked = [R(5), R(9), R(10)]
        for r in parked:
            F.Seat(f, r, "batch")
        with self.assertLogs(F.logger, level="INFO") as cap:
            out = f._seat_rotate_plan(parked, [self._w(R(3)), self._w(R(7))])
        # by age: 3, 5, 7 get the seats; 9 and 10 wait
        self.assertEqual(out, [R(9), R(10)])
        lines = [m for m in cap.output if "SEAT-ROTATE" in m]
        self.assertIn(f"rid_in={R(3)} rid_out={R(9)} reason=age", lines[0])
        self.assertTrue(any(f"D-REFILL rid={R(9)} freed_by=rotate" in m for m in cap.output))

    def test_the_old_long_runs_keep_their_seats_against_younger_waiters(self):
        # the NF finding re-read: the five long runs are OLDER than the waiters -> they keep the
        # seats (the user's rule), the waiters come in as each long run finishes
        f = self._front(d_bs=5)
        parked = [R(5, 0), R(12, 1), R(21, 5), R(34, 10), R(41, 13)]
        self.assertEqual(f._seat_rotate_plan(parked, [self._w(R(39)), self._w(R(44))]), [R(41, 13)])

    def test_switch_off(self):
        f = self._front(d_bs=1)
        with mock.patch.dict(os.environ, {SA.ENV: "0"}):
            self.assertEqual(f._seat_rotate_plan([R(9)], [self._w(R(3))]), [])


class DSide(unittest.TestCase):
    def _req(self, rid, site=None):
        r = types.SimpleNamespace(rid=rid, origin_input_ids=[0] * 10, output_ids=[])
        if site:
            DS.mark_parked(r, site, now=1.0)
        return r

    def _sched(self, running, waiting, cap=3, spec=False):
        batch = types.SimpleNamespace(reqs=list(running), released=[],
                                      spec_algorithm=None if not spec else types.SimpleNamespace(is_none=lambda: False))
        batch.release_req = lambda idx, rem, sa, retain=False: batch.released.append((batch.reqs[idx].rid, retain))

        def filt(keep_indices):
            batch.reqs = [batch.reqs[i] for i in keep_indices]

        batch.filter_batch = filt
        sched = types.SimpleNamespace(waiting_queue=list(waiting), server_args=types.SimpleNamespace(
            max_running_requests=cap), running_batch=batch)
        sched._add_request_to_queue = lambda req, is_retracted=False: sched.waiting_queue.append(req)
        return sched, batch

    def _run(self, fn):
        with mock.patch.object(DS, "d_flip_park_active", lambda: True), \
             mock.patch.object(DPR, "seat_cap", lambda sched: None):
            return fn()

    def test_oldest_done_second_oldest_parked_moves_in_and_displaces_the_youngest(self):
        # running 5, 6, 7 (seats full); the second-oldest (2) had to park and waits
        second = self._req(R(2))
        sched, batch = self._sched([self._req(R(5)), self._req(R(6)), self._req(R(7))], [second])
        with self.assertLogs(DPR.logger, level="WARNING") as cap:
            got = self._run(lambda: DPR.displace_for_age(sched, batch))
        self.assertEqual(got, R(7))
        self.assertEqual(batch.released, [(R(7), True)])
        self.assertEqual([r.rid for r in batch.reqs], [R(5), R(6)])
        victim = sched.waiting_queue[-1]
        self.assertEqual((victim.rid, DS.park_site(victim)), (R(7), DS.SITE_PRESSURE))
        self.assertIn("SEAT-AGE DISPLACE rid_out=%s" % R(7), cap.output[0])

    def test_a_younger_waiter_never_displaces(self):
        sched, batch = self._sched([self._req(R(2)), self._req(R(3)), self._req(R(4))], [self._req(R(9))])
        self.assertIsNone(self._run(lambda: DPR.displace_for_age(sched, batch)))
        self.assertEqual(batch.released, [])

    def test_free_seat_no_displacement_the_older_just_moves_in(self):
        sched, batch = self._sched([self._req(R(5)), self._req(R(6))], [self._req(R(2))])
        self.assertIsNone(self._run(lambda: DPR.displace_for_age(sched, batch)))

    def test_spec_only_the_back_may_leave(self):
        sched, batch = self._sched([self._req(R(7)), self._req(R(5)), self._req(R(6))], [self._req(R(2))],
                                   spec=True)
        self.assertIsNone(self._run(lambda: DPR.displace_for_age(sched, batch)))
        sched, batch = self._sched([self._req(R(5)), self._req(R(6)), self._req(R(7))], [self._req(R(2))],
                                   spec=True)
        self.assertEqual(self._run(lambda: DPR.displace_for_age(sched, batch)), R(7))

    def test_the_barrier_lets_an_older_newcomer_pass(self):
        gate = DS.admission_gate([self._req(R(7), DS.SITE_PRESSURE), self._req(R(3)), self._req(R(9))],
                                 running=[self._req(R(5))])
        self.assertEqual(gate.oldest_parked_age, 7.0)
        self.assertIsNone(gate.skip(self._req(R(3))), "older than the parked one: goes first")
        self.assertEqual(gate.skip(self._req(R(9))), "weg2_d_park_first")

    def test_park_victim_is_the_youngest_by_arrival(self):
        reqs = [self._req(R(9)), self._req(R(2)), self._req(R(5))]
        order = DS.retraction_order(reqs, spec_active=False)
        self.assertEqual(reqs[order[-1]].rid, R(9))

    def test_park_victim_age_is_the_front_arrival_not_ds_intake_seq(self):
        # ONE age: D's kv_arrival_seq (intake order) disagrees with the front
        # arrival -- the front arrival decides who is youngest.
        a, b = self._req(R(2)), self._req(R(9))
        a.kv_arrival_seq, b.kv_arrival_seq = 50, 1  # R(2) reached D last
        order = DS.retraction_order([a, b], spec_active=False)
        self.assertEqual([a, b][order[-1]].rid, R(9))

    def test_park_victim_keeps_kvso_class_and_fast_lane_rank(self):
        # kvso's protection key stands above age: a fast-lane request is not
        # the victim though youngest, a 'preferred' spill class parks first
        # though oldest.
        old, young_fast = self._req(R(2)), self._req(R(9))
        young_fast.is_fast_lane = True
        order = DS.retraction_order([old, young_fast], spec_active=False)
        self.assertEqual([old, young_fast][order[-1]].rid, R(2))
        pref, young = self._req(R(1)), self._req(R(9))
        pref.spill_class = "preferred"
        order = DS.retraction_order([pref, young], spec_active=False)
        self.assertEqual([pref, young][order[-1]].rid, R(1))

    def test_deferred_parked_placed_by_age(self):
        a, b = self._req(R(4), DS.SITE_FLIP), self._req(R(8))
        new_old, new_young = self._req(R(3)), self._req(R(11))
        sched = types.SimpleNamespace(waiting_queue=[a, new_old, b, new_young], weg2_dormant_hold=[],
                                      weg2_park_defer={R(4)})
        DPR._apply_park_defer(sched)
        self.assertEqual([r.rid for r in sched.waiting_queue], [R(3), R(4), R(8), R(11)])

    def test_nf_form_seat_cap_from_the_phase(self):
        sched, batch = self._sched([self._req(R(5)), self._req(R(6))], [self._req(R(2))], cap=6)
        with mock.patch.object(DS, "d_flip_park_active", lambda: True), \
             mock.patch.object(DPR, "seat_cap", lambda s: 2):
            self.assertEqual(DPR.displace_for_age(sched, batch), R(6))


class BackfillProbe(unittest.TestCase):
    """SA backfill probes candidates with the gate's own terms, side-effect
    free; the one gate call (and the one Seat charge) stays with the seated
    request (FIX 7 C3e)."""

    def _front(self):
        import time as _t

        f = F.Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, d_bs=6,
                    carrier_max_tokens=27466)
        t0 = _t.time()
        f._d_seats_live = [types.SimpleNamespace(tokens=15047, t_taken=t0 + 1)]
        return f, {"t": t0, "available": 27466, "limit": 27466, "occupied": 0}

    def test_probe_agrees_with_the_gate_and_counts_nothing(self):
        f, reading = self._front()
        before = dict(f.counters)
        hold = f._d_token_hold_rid
        for est, realised in ((15308, 0), (15308, 8642), (100, 0)):
            fits = f._d_budget_fits(est, reading, realised)
            self.assertEqual(dict(f.counters), before)
            self.assertEqual(f._d_token_hold_rid, hold)
            self.assertEqual(fits, not f._d_token_budget_blocks("x", est, reading, realised))
            f.counters.clear(); f.counters.update(before); f._d_token_hold_rid = hold

    def test_probe_open_gate_cases(self):
        f, reading = self._front()
        self.assertTrue(f._d_budget_fits(10 ** 9, None, 0))
        f._d_seats_live = []
        self.assertTrue(f._d_budget_fits(10 ** 9, reading, 0))


class Kept244(unittest.TestCase):
    """The #244 transport kept from e1eb275a15 (wake field, D defer set, seat_wait_s)."""

    def test_wake_object_sets_the_defer_set_27b_and_nf_forms(self):
        sched = types.SimpleNamespace(server_args=types.SimpleNamespace(max_running_requests=6))
        req = types.SimpleNamespace(handoff_n=3, parked_n=4, park_defer_rids=["weg2-13-41"], epoch="e")
        with mock.patch.dict(os.environ, {DS.GROUP_ENV: "D"}):
            seats = DPR.note_wake_seats(sched, req)
        self.assertEqual(sched.weg2_park_defer, {"weg2-13-41"})
        self.assertEqual(seats.n, 6)  # 27B: capped by d_bs
        nf = DS.phase_seats(2, 1, cap=6)  # NF form: n = hand-offs + resumed parked
        self.assertEqual(nf.n, 3)

    def test_io_struct_field(self):
        from sglang.srt.managers.io_struct import ResumeMemoryOccupationReqInput as R

        self.assertIsNone(R().park_defer_rids)


    def test_seat_wait_in_d_admit(self):
        src = open(F.__file__).read()
        self.assertIn("source=%s seat_wait_s=%.1f", src)
        self.assertIn("t_ready=getattr(p, \"t_ready\", 0.0)", src)
        self.assertIn("p.t_ready = time.time()  # #244 seat_wait_s", src)



class Wiring(unittest.TestCase):
    def test_admitter_orders_by_age_and_backfills(self):
        src = open(F.__file__).read()
        i = src.index("    async def d_admitter(self)")
        blk = src[i:i + 9000]
        self.assertIn("_ordered = _sa.by_age(self._ready_for_d)", blk)
        self.assertIn("WEG2 SEAT-AGE BACKFILL", blk)
        self.assertLess(blk.index("SEAT-AGE BACKFILL"), blk.index("await self._d_seat.acquire()"))

    def test_d_admission_displaces_before_the_early_return(self):
        src = open(DPR.__file__).read()
        i = src.index("def admission(sched, running_batch):")
        blk = src[i:i + 1500]
        self.assertLess(blk.index("displace_for_age(sched, running_batch)"),
                        blk.index("immediate park, nothing flip-parked waits"))


if __name__ == "__main__":
    unittest.main()
