"""NF-C-SITZ 1021: front and D name the SAME seat-age victim (one rule, ``d_park_runtime.displaceable``), a credit
seat only for a displacement D can do, at most ONE open credit, never past n+1 owed seats.

Specimens (rc27 893d09fa3f):
  _1021_020616 front 2271 SEAT-AGE ON-CREDIT rid=pdflip-2-27 victims=['pdflip-2-29'] seats_live=6 d_bs=6
               front 2277 D-ADMIT rid=pdflip-2-27 seat=7/6; 2299 ARRIVAL-SEAT ... why=seat taken=7 n=6
               D 19679 SEAT-AGE DISPLACE-WAIT older=pdflip-2-27 youngest=pdflip-2-28: under speculative decoding only
               the back of the batch may leave; the youngest is not the back
  _1021_023225 02:44:53 pdflip-2-30 verdict=flip_now taken=11 > n=6 (credit stack); rc29 _022209 seat=8/6.
"""
import asyncio
import collections
import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from flliper.srt.pdflip import arrival_seat_rule as ASR  # noqa: E402
from flliper.srt.pdflip import d_park_runtime as DPR  # noqa: E402
from flliper.srt.pdflip import d_seats as DS  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402


def R(n, e=2):
    return f"pdflip-{e}-{n}"


def _req(rid):
    return types.SimpleNamespace(rid=rid, origin_input_ids=[0] * 10, output_ids=[])


def _sched(running, waiting, cap=6, spec=True):
    batch = types.SimpleNamespace(reqs=list(running), released=[],
                                  spec_algorithm=None if not spec else types.SimpleNamespace(is_none=lambda: False))
    batch.release_req = lambda idx, rem, sa, retain=False: batch.released.append((batch.reqs[idx].rid, retain))

    def filt(keep_indices):
        batch.reqs = [batch.reqs[i] for i in keep_indices]

    batch.filter_batch = filt
    sched = types.SimpleNamespace(waiting_queue=list(waiting), server_args=types.SimpleNamespace(
        max_running_requests=cap), running_batch=batch,
        token_to_kv_pool_allocator=types.SimpleNamespace(available_size=lambda: 1 << 30))
    sched._add_request_to_queue = lambda req, is_retracted=False: sched.waiting_queue.append(req)
    return sched, batch


def _run(fn):
    with mock.patch.object(DS, "d_flip_park_active", lambda: True), \
         mock.patch.object(DPR, "seat_cap", lambda sched: None):
        return fn()


class DSideOneRule(unittest.TestCase):
    # the metal shape: D's youngest sits in the MIDDLE of the batch, the back is another younger request
    # (live: youngest=pdflip-2-28 not the back, the front's victim pdflip-2-29) -- here 2-29 middle, 2-28 back
    METAL = [R(20), R(21), R(29), R(22), R(23), R(28)]

    def test_displaceable_is_the_back_under_spec_and_all_otherwise(self):
        _, b = _sched([_req(r) for r in self.METAL], [], spec=True)
        self.assertEqual(DPR.displaceable_rids(b), [R(28)])
        _, b = _sched([_req(r) for r in self.METAL], [], spec=False)
        self.assertEqual(DPR.displaceable_rids(b), self.METAL)
        self.assertEqual(DPR.displaceable_rids(types.SimpleNamespace(reqs=[])), [])

    def test_metal_case_d_parks_the_back_the_front_named(self):
        """pdflip-2-27 waits, six seats full: D parks the back (younger than 2-27) -- the victim the front plans on --
        instead of DISPLACE-WAIT on the youngest in the middle."""
        sched, batch = _sched([_req(r) for r in self.METAL], [_req(R(27))])
        with self.assertLogs(DPR.logger, level="INFO") as cap:
            got = _run(lambda: DPR.displace_for_age(sched, batch))
        self.assertEqual(got, R(28))
        self.assertFalse(any("DISPLACE-WAIT" in m for m in cap.output), cap.output)
        self.assertEqual(batch.released, [(R(28), True)])

    def test_a_back_older_than_the_waiter_is_no_victim(self):
        """Alter-Vorrang: the back is OLDER than the waiter -> nobody leaves (never an older one for a younger)."""
        order = [R(28), R(29), R(30), R(31), R(32), R(20)]
        sched, batch = _sched([_req(r) for r in order], [_req(R(27))])
        self.assertIsNone(_run(lambda: DPR.displace_for_age(sched, batch)))
        self.assertEqual(batch.released, [])


class FrontCredit(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def _front(self, taken, n=6, open_credit=False):
        f = object.__new__(F.Front)
        f.epoch, f.d_bs = 2, n
        f.counters = collections.Counter()
        f._d_seat = asyncio.Semaphore(0)       # every permit held: locked()
        f._d_seats_live = set()
        f._ready_for_d = collections.deque()
        st = f._asr_st()
        st["granted"][R(27)] = 1.0
        st["age_victims"] = {R(27): [R(29)]}
        if open_credit:
            F.Seat(f, R(26), "short", on_credit=True)
        f._arrival_seat_taken = lambda for_rid=None: (taken, n)
        return f

    def _credit(self, f):
        with mock.patch.object(F._asr, "enabled", lambda *a, **k: True):
            return F.Front._age_seat_on_credit(f, R(27))

    def test_one_credit_when_owed_seats_are_at_n(self):
        # taken counts the head's own grant: 7 with it = 6 without it = n -> one credit (seat 7/6), no more
        self.assertTrue(self._credit(self._front(taken=7)))

    def test_danger_no_second_open_credit(self):
        """DANGER DIRECTION (credit stack, seat 8/6 / taken=11): with one credit open no second one is granted."""
        f = self._front(taken=7, open_credit=True)
        with self.assertLogs(F.logger, level="INFO") as cap:
            self.assertFalse(self._credit(f))
        self.assertTrue(any("CREDIT-REFUSED" in m for m in cap.output))

    def test_danger_never_past_n_plus_one_owed(self):
        f = self._front(taken=9)      # 8 owed without the head > n=6
        self.assertFalse(self._credit(f))
        self.assertEqual(f.counters["arrival_seat_age_credit_refused"], 1)


class FrontPlansOnDsSet(unittest.TestCase):
    def test_the_front_victim_is_one_d_can_park(self):
        """The AGE PLAN only names victims from D's ``displaceable_rids``: a youngest D cannot park is never planned."""
        f = object.__new__(F.Front)
        f.counters = collections.Counter()
        f._d_seats_live = set()
        f.groups = {"D": object()}
        st = f._asr_st()
        running = [R(20), R(21), R(28), R(29)]
        f.__dict__["_asr_arrive"] = {r: float(int(r.split("-")[-1])) for r in running}
        basis = {"unread": False, "wall": False, "running": set(running), "displaceable": {R(21)}}
        seen = {}

        def plan(head_arrival, seat_is_free, deficit, run, arrivals, tokens, lane_of=None, head_lane=0):
            seen["running"] = list(run)
            return list(run)[:1] or None

        with mock.patch.object(F.Front, "_flip_ledger", lambda self, g: list(running)), \
             mock.patch.object(F.Front, "_prio_lane", lambda self, r: 0), \
             mock.patch.object(ASR, "displace_plan", plan):
            out = F.Front._arrival_seat_age_verdict(f, R(19), 19.0, False, 0, basis=basis)
        self.assertEqual(seen["running"], [R(21)])
        self.assertEqual(out, [R(21)])
        self.assertNotIn("parked", basis)
        del st


if __name__ == "__main__":
    unittest.main()
