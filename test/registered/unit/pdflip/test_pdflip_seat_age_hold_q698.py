# SPDX-License-Identifier: Apache-2.0
"""Q-698 SEAT-AGE DISPLACE LIVELOCK (NF y9n abl 76163d3aef, boot ...10032328, D, 23:33:24-23:44:15Z).

The older hand-off pdflip-4-14 (extent 188864, uncached 384) was refused NO_TOKEN; SEAT-AGE (KV
trigger) parked the youngest running request for it (pdflip-4-16), and in the NEXT pass the
ARRIVAL-SEAT early resume (ResumeBook, margin 0) let that victim back in because its OWN 14781
tokens fit (``PDFLIP ARRIVAL-SEAT PRESSURE-RESUME rid=pdflip-4-16 avail=167744 need=14781``) -- it
took back the room it had been parked to give, pdflip-4-14 was refused again and pdflip-4-17 was
parked for it, and so on: 782 DISPLACE passes on TP0, every pass an extend of a resumed victim,
0 decode rounds for 11 min, pdflip-4-14 never admitted (client gone after 652 s).

Fix: a victim waits for the older request it was parked for (d_seats.DISPLACED_FOR_ATTR):
blocked in the gate (no early resume either) while that one waits; the hold ends when it is
admitted or gone, when it is refused NO_TOKEN with nothing running on D (group MIN), or when its
displacement budget (DISPLACE_MAX_PER_OLDER) is spent -- named lines, no rotation."""

import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import d_park_runtime as DPR  # noqa: E402
from flliper.srt.pdflip import d_seats as DS  # noqa: E402

OLDER = "pdflip-4-14"


def _req(rid, span=1000, prefix=0, out=0):
    return types.SimpleNamespace(rid=rid, origin_input_ids=[0] * span, output_ids=[0] * out,
                                 prefix_indices=[0] * prefix)


def _sched(running, waiting, cap=6, no_token=None, group_min=None, avail=100):
    batch = types.SimpleNamespace(reqs=list(running), released=[], spec_algorithm=None)
    batch.release_req = lambda idx, rem, sa, retain=False: batch.released.append(batch.reqs[idx].rid)
    batch.filter_batch = lambda keep_indices: setattr(batch, "reqs", [batch.reqs[i] for i in keep_indices])
    sched = types.SimpleNamespace(
        waiting_queue=list(waiting), server_args=types.SimpleNamespace(max_running_requests=cap),
        _pdflip_sa_no_token=no_token, calls=[],
        tree_cache=types.SimpleNamespace(page_size=64),
        token_to_kv_pool_allocator=types.SimpleNamespace(available_size=lambda: avail))
    sched._add_request_to_queue = lambda req, is_retracted=False: sched.waiting_queue.append(req)

    def gm(flags):
        sched.calls.append(list(flags))
        return group_min(flags) if group_min else flags

    sched._pdflip_group_min_flags = gm
    return sched, batch


def _run(fn):
    with mock.patch.object(DS, "d_flip_park_active", lambda: True), \
         mock.patch.object(DPR, "seat_cap", lambda s: None), \
         mock.patch.object(DPR, "_partial_keep", lambda s, v, o: "0"):
        return fn()


def _book():
    # the ARRIVAL-SEAT early resume as armed on NF (margin 0), one step for the test
    return DS.ResumeBook(margin_tokens=0, steps=1, source="arrival-seat")


def _victim(rid, held_for=OLDER, span=14781):
    v = _req(rid, span=span)
    DS.mark_parked(v, DS.SITE_PRESSURE)
    setattr(v, DS.DISPLACED_FOR_ATTR, held_for)
    return v


class GateHold(unittest.TestCase):
    """The metal sequence at the gate: the victim's own fit (avail 167744 >= 14781) let it back."""

    def test_victim_does_not_resume_while_the_older_it_was_parked_for_waits(self):
        running = [_req("pdflip-2-11"), _req("pdflip-3-12"), _req("pdflip-4-13")]
        v16 = _victim("pdflip-4-16")
        older = _req(OLDER, span=188864)
        gate = DS.admission_gate([v16, older], running=running, avail_tokens=167744,
                                 resume_book=_book())
        self.assertIn("pdflip-4-16", gate.blocked)
        self.assertEqual(gate.skip(v16, admitted=[]), "pdflip_d_park_older_live")
        self.assertIsNone(gate.skip(older, admitted=[]), "the older one goes first")
        self.assertIn("held_for_older=1", gate.note)

    def test_hold_holds_with_nothing_running_until_the_idle_lift(self):
        v16 = _victim("pdflip-4-16")
        gate = DS.admission_gate([v16, _req(OLDER)], running=[], avail_tokens=10 ** 6,
                                 resume_book=_book())
        self.assertIn("pdflip-4-16", gate.blocked)

    def test_older_admitted_or_gone_the_hold_ends(self):
        running = [_req("pdflip-2-11"), _req(OLDER)]  # the older one was admitted
        v16 = _victim("pdflip-4-16")
        gate = DS.admission_gate([v16], running=running, avail_tokens=167744, resume_book=_book())
        self.assertNotIn("pdflip-4-16", gate.blocked, "fits beside the older running one (ARRIVAL-SEAT)")
        self.assertIsNone(getattr(v16, DS.DISPLACED_FOR_ATTR))

    def test_older_in_the_settle_still_holds(self):
        settle = _req(OLDER)
        v16 = _victim("pdflip-4-16")
        gate = DS.admission_gate([v16], running=[_req("pdflip-2-11")], pending_outside=[settle],
                                 avail_tokens=167744, resume_book=_book())
        self.assertIn("pdflip-4-16", gate.blocked)

    def test_unmarked_pressure_park_unchanged(self):
        v = _req("pdflip-4-16", span=14781)
        DS.mark_parked(v, DS.SITE_PRESSURE)
        gate = DS.admission_gate([v, _req(OLDER)], running=[_req("pdflip-2-11")],
                                 avail_tokens=167744, resume_book=_book())
        self.assertNotIn("pdflip-4-16", gate.blocked, "no displacement mark: the early resume as before")


class Replay(unittest.TestCase):
    def test_two_victims_no_longer_rotate(self):
        # pass 1: pdflip-4-14 refused NO_TOKEN, running 2-11, 3-12, 4-16, 4-17 (KV: 4-17 must leave)
        running = [_req("pdflip-2-11"), _req("pdflip-3-12"), _req("pdflip-4-16", span=1000),
                   _req("pdflip-4-17", span=1000)]
        older = _req(OLDER, span=1500)
        sched, batch = _sched(running, [older], no_token=OLDER, avail=600)
        self.assertEqual(_run(lambda: DPR.displace_for_age(sched, batch)), "pdflip-4-17")
        v17 = sched.waiting_queue[-1]
        self.assertEqual(getattr(v17, DS.DISPLACED_FOR_ATTR), OLDER)
        # pass 2: the gate holds pdflip-4-17 although its own fit would pass the early resume
        gate = DS.admission_gate(DS.order_waiting(sched.waiting_queue), running=batch.reqs,
                                 avail_tokens=10 ** 6, resume_book=_book())
        self.assertIn("pdflip-4-17", gate.blocked)
        self.assertIsNone(gate.skip(older, admitted=[]))
        # pass 2 refused again (adder budget): the next youngest leaves, both held
        sched._pdflip_sa_no_token = OLDER
        self.assertEqual(_run(lambda: DPR.displace_for_age(sched, batch)), "pdflip-4-16")
        gate = DS.admission_gate(DS.order_waiting(sched.waiting_queue), running=batch.reqs,
                                 avail_tokens=10 ** 6, resume_book=_book())
        self.assertEqual(gate.blocked, frozenset({"pdflip-4-16", "pdflip-4-17"}))
        # the older one is admitted: both victims are free to resume
        sched.waiting_queue = [q for q in sched.waiting_queue if q is not older]
        gate = DS.admission_gate(DS.order_waiting(sched.waiting_queue), running=batch.reqs + [older],
                                 avail_tokens=10 ** 6, resume_book=_book())
        self.assertEqual(gate.blocked, frozenset())


class EndStates(unittest.TestCase):
    def test_displacement_budget_is_bounded_and_named(self):
        running = [_req("pdflip-2-11"), _req("pdflip-4-17", span=1000)]
        older = _req(OLDER, span=1000)
        v16 = _victim("pdflip-4-16")
        sched, batch = _sched(running, [older, v16], no_token=OLDER, avail=100)
        sched._pdflip_sa_displace_counts = {OLDER: DPR.DISPLACE_MAX_PER_OLDER}
        with self.assertLogs(DPR.logger, level="WARNING") as cap:
            self.assertIsNone(_run(lambda: DPR.displace_for_age(sched, batch)))
        self.assertEqual(batch.released, [])
        self.assertIn("Q-698 SEAT-AGE DISPLACE-EXHAUSTED older=pdflip-4-14", cap.output[0])
        self.assertIsNone(getattr(v16, DS.DISPLACED_FOR_ATTR), "its held victims resume")

    def test_counts_reset_when_the_older_leaves_the_queue(self):
        sched, batch = _sched([_req("pdflip-2-11")], [], avail=100)
        sched._pdflip_sa_displace_counts = {OLDER: 5}
        _run(lambda: DPR.displace_for_age(sched, batch))
        self.assertEqual(sched._pdflip_sa_displace_counts, {})

    def test_idle_lift_after_a_refusal_on_an_idle_d(self):
        v16 = _victim("pdflip-4-16")
        sched, batch = _sched([], [_req(OLDER), v16], no_token=OLDER)
        # first idle pass: the refusal may stem from a pass in which something ran
        _run(lambda: DPR.displace_for_age(sched, batch))
        self.assertEqual(getattr(v16, DS.DISPLACED_FOR_ATTR), OLDER)
        self.assertEqual(sched.calls, [])
        # refused again in the idle pass -> every rank agrees -> the hold lifts, named
        sched._pdflip_sa_no_token = OLDER
        with self.assertLogs(DPR.logger, level="WARNING") as cap:
            _run(lambda: DPR.displace_for_age(sched, batch))
        self.assertIsNone(getattr(v16, DS.DISPLACED_FOR_ATTR))
        self.assertIn("Q-698 SEAT-AGE HOLD-LIFT", cap.output[0])
        self.assertEqual(sched.calls, [[True]], "through the group MIN")

    def test_idle_lift_needs_every_rank(self):
        v16 = _victim("pdflip-4-16")
        sched, batch = _sched([], [_req(OLDER), v16], no_token=OLDER, group_min=lambda f: [False])
        _run(lambda: DPR.displace_for_age(sched, batch))
        sched._pdflip_sa_no_token = OLDER
        _run(lambda: DPR.displace_for_age(sched, batch))
        self.assertEqual(getattr(v16, DS.DISPLACED_FOR_ATTR), OLDER)


if __name__ == "__main__":
    unittest.main()
