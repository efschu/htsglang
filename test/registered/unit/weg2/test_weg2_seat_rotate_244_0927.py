# SPDX-License-Identifier: Apache-2.0
"""#244 SEAT-ROTATE (NF rc12r dkrnfh91dprbar1dauer09271632, 16:52-17:04).

WAIT-BOUND fired, but every PARK-RESUME gave the seats back to the same 5 long runs (weg2-0-5,
1-12, 5-21, 10-34, 13-41: 18685 / 21405 tokens, 1375-1618 s wall); every D-REFILL was
freed_by=leg2_finished; waiters with leg 1 done waited 255 -> 658 s, queued_d 4 -> 13.
Now: at the P->D wake the waiters past the bound (oldest first) take the seats of the YOUNGEST
parked (last admitted to D); dwell (no rotate-out in the resume phase or the next), fairness (a
rid deferred >= 2 phases is not deferred again); D treats deferred ones as ordinary waiting work.
"""

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

LONG = ["weg2-0-5", "weg2-1-12", "weg2-5-21", "weg2-10-34", "weg2-13-41"]


def _front(epoch=16):
    f = object.__new__(F.Front)
    f.epoch = epoch
    f.d_wait_bound_s = 60.0
    f.d_bs = 6
    f.counters = collections.Counter()
    f._d_seat = asyncio.Semaphore(0)
    f._d_seats_live = set()
    f._ready_for_d = collections.deque()
    now = time.time()
    # admitted 16:38 .. 16:45 -> weg2-13-41 is the youngest
    f._d_admit_t = {r: now - 1500 + i * 60 for i, r in enumerate(LONG)}
    return f


def _waiter(loop, rid, waited):
    p = F.Pending(rid, "/v1/messages", {}, "t", time.time() - waited - 30, loop.create_future(),
                  est_prompt=50000, est_uncached=50000, span_known=True)
    p.leg1_done = True
    p.t_ready = time.time() - waited
    return p


class Plan(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()

    def tearDown(self):
        self.loop.close()

    def test_nf_shape_waiters_take_the_youngest_seats(self):
        f = _front()
        for r in LONG:  # the parked runs hold their front seats
            F.Seat(f, r, "batch")
        ready = [_waiter(self.loop, "weg2-12-39", 255), _waiter(self.loop, "weg2-13-44", 190)]
        with self.assertLogs(F.logger, level="INFO") as cap:
            out = f._seat_rotate_plan(list(LONG), ready)
        self.assertEqual(out, ["weg2-13-41", "weg2-10-34"])  # youngest first
        rot = [m for m in cap.output if "SEAT-ROTATE" in m]
        self.assertIn("rid_in=weg2-12-39 rid_out=weg2-13-41 reason=wait-bound waited_s=255", rot[0])
        refill = [m for m in cap.output if "D-REFILL rid=weg2-13-41 freed_by=rotate" in m]
        self.assertTrue(refill, "the deferred one's front seat goes to the waiter")

    def test_wake_fields_carry_the_defer_list(self):
        f = _front()
        f._d_parked = {r: 0.0 for r in LONG}
        f.groups = {"D": types.SimpleNamespace(outstanding={r: 0.0 for r in LONG})}
        f._ready_for_d.append(_waiter(self.loop, "weg2-12-39", 255))
        f._handoff_in_flight = lambda: 0
        out = F.Front._wake_handoff_fields(f, "D")
        self.assertEqual(out["park_defer_rids"], ["weg2-13-41"])
        self.assertEqual((out["handoff_n"], out["parked_n"]), (1, 4))

    def test_no_waiter_past_the_bound_no_rotation(self):
        f = _front()
        self.assertEqual(f._seat_rotate_plan(list(LONG), [_waiter(self.loop, "w", 30)]), [])

    def test_dwell_and_fairness(self):
        f = _front(epoch=16)
        f._seat_resumed_epoch = {"weg2-13-41": 15}  # resumed in the previous phase: dwell
        f._seat_defer_n = {"weg2-10-34": 2}         # deferred twice: priority now
        out = f._seat_rotate_plan(list(LONG), [_waiter(self.loop, "w1", 300), _waiter(self.loop, "w2", 200)])
        self.assertEqual(out, ["weg2-5-21", "weg2-1-12"])

    def test_switch_off(self):
        f = _front()
        with mock.patch.dict(os.environ, {"SGLANG_WEG2_SEAT_ROTATE": "0"}):
            self.assertEqual(f._seat_rotate_plan(list(LONG), [_waiter(self.loop, "w", 300)]), [])

    def test_phases_waiters_in_within_two_no_pingpong(self):
        """The finding's sequence over phases: 5 long runs parked every phase, waiters arriving.
        Every waiter past the bound gets a seat at the next wake; no rid is rotated out twice
        in a row (dwell) and none is deferred more than twice."""
        f = _front(epoch=12)
        parked = list(LONG)
        waiting = [_waiter(self.loop, f"w{i}", 70 + i) for i in range(4)]
        admitted_at = {}
        out_history = collections.defaultdict(list)
        for phase in range(6):
            f.epoch = 12 + 2 * phase
            out = f._seat_rotate_plan(parked, list(waiting))
            f._seat_last_defer = tuple(out)
            f._seat_rotate_note_resume(parked)
            n_in = len(out)
            for w in sorted(waiting, key=lambda p: p.t_ready)[:n_in]:
                admitted_at[w.rid] = phase
                waiting.remove(w)
                f._d_admit_t[w.rid] = time.time() + phase  # youngest now
                parked.append(w.rid)
            for r in out:
                out_history[r].append(phase)
                parked.remove(r)  # waits as ordinary work, not in the parked list any more
            waiting.extend(_waiter(self.loop, f"n{phase}{i}", 61) for i in range(1))
        self.assertTrue(all(ph <= 1 for w, ph in admitted_at.items() if w.startswith("w")),
                        admitted_at)
        for r, phases in out_history.items():
            self.assertEqual(len(phases), 1, f"{r} rotated out more than once: {phases}")


class DSide(unittest.TestCase):
    def _req(self, rid, parked):
        r = types.SimpleNamespace(rid=rid)
        if parked:
            DS.mark_parked(r, DS.SITE_FLIP, now=1.0)
        return r

    def test_deferred_parked_become_ordinary_waiting_work_at_the_tail(self):
        a, b = self._req("weg2-0-5", True), self._req("weg2-13-41", True)
        h1, h2 = self._req("weg2-12-39", False), self._req("weg2-13-44", False)
        sched = types.SimpleNamespace(waiting_queue=[a, b, h1, h2], weg2_dormant_hold=[],
                                      weg2_park_defer={"weg2-13-41"})
        with self.assertLogs(DPR.logger, level="INFO") as cap:
            self.assertEqual(DPR._apply_park_defer(sched), 1)
        self.assertIsNone(DS.park_site(b))
        self.assertEqual([r.rid for r in DS.order_waiting(sched.waiting_queue)],
                         ["weg2-0-5", "weg2-12-39", "weg2-13-44", "weg2-13-41"])
        self.assertIn("seat-rotate", cap.output[0])
        self.assertEqual(DPR._apply_park_defer(sched), 0, "each rid once")

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


class Wiring(unittest.TestCase):
    def test_seat_wait_in_d_admit(self):
        src = open(F.__file__).read()
        self.assertIn("source=%s seat_wait_s=%.1f", src)
        self.assertIn("t_ready=getattr(p, \"t_ready\", 0.0)", src)
        self.assertIn("p.t_ready = time.time()  # #244 seat_wait_s", src)


if __name__ == "__main__":
    unittest.main()
