# SPDX-License-Identifier: Apache-2.0
"""PRIORITY LANES, fix round 3 (NF metal 211536, image int23lanes3 @ d12614c53e, boot 21:15:36Z, probe 2b early).

The failure: 21:22:35 PP1 ``#1004 SLOT DISAGREEMENT`` -- "PP1 launching slot 0 (fwd_ct=20, rids=[weg2-6-17], extend=151) but the
upstream's proxy names slot 1 (seq=21 rows=16000 extent=('weg2-7-18', 0, 16000))".  Cause, from the logs (done/lanes-metall4-1011):

* the P group of this form has NO row carrier (P log 21:17:30 "#631 ROW AUTHORITY DISABLED: no pp_flip_counters side channel on
  this boot form"): every follower plans for ITSELF.  PP0 applied the floor (list m, 21:22:29), parked the lane-0 continuation
  weg2-6-16, held the lane-0 request weg2-6-17 by lane and planned nothing in that pass; PP1 applied the SAME stamp at 21:22:32 in the
  pass that planned its slot 0 (before its plan) -- and admitted weg2-6-17, because the lane skip of the admission loop belonged to
  PP0 alone (``owns_admission``).  => ``lanes_p.skips_by_lane``.
* the effective pass of the stamp was implicit.  It is now the batch index ``eff`` (``forward_ct``) PP0 names: every stage, PP0 too,
  applies the floor in the plan of batch ``eff``.

What is pinned here: (1) the three-stage plan-first harness (real ``PrefillAdder.add_chunked_req`` for the continuation, the real lane
predicates for the admission loop, the real stamp/absorb functions): no stage admits the floor lane before the common batch, all stages
park the lower lane at the same chunk boundary, the batches are identical -- with the stamp absorbed in the same pass and one pass late;
CAN-FAIL twins (no skip on followers / no effective-batch binding) render exactly the ``#1004`` text; (2) the predicates
(``executes_schedule`` / ``skips_by_lane`` / ``cap_applies``) and the LaneP queue; (3) the front: a rank death reaches the held
streams (one named error event, at once); (4) fix 2 meets the effective batch (the wait of a floor-lane arrival is bounded).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import pickle
import sys
import tempfile
import time
import types
import unittest
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_weg2_lanes_l3_1008 import (  # noqa: E402
    CHUNK, SPAN, _LanesOn, _adder, _req, _rpc, _stage,
)

from sglang.srt.weg2 import lane_ctl as LC  # noqa: E402
from sglang.srt.weg2 import lanes, lanes_p  # noqa: E402
from sglang.srt.weg2 import pp_room_vote as pr  # noqa: E402


# ---------------------------------------------------------------------------
# the plan-first harness: three stages, one pass index, every stage plans for itself
# ---------------------------------------------------------------------------

QUEUE_LANES = {"6-17": 0, "7-18": 1}


class _Stage:
    """One PP stage of the form WITHOUT a row carrier: it holds the lane-0 continuation ``6-16`` (``cont_left`` chunks still to
    run) and its own waiting queue; ``_pp_admission_incoming_effective`` stays None (it plans for itself)."""

    def __init__(self, rank, cont_left, arrivals):
        self.stage = _stage(rank)
        self.stage.forward_ct = 0
        self.stage.chunked_req = None
        self.stage.mbs = [None, None, None]          # the three microbatch slots: a launched batch occupies its slot (work in flight)
        self.stage._pp_admission_incoming_effective = None
        self.cont_left = cont_left
        self.arrivals = arrivals            # {pass: [rids]} -- the request chain is aligned by list index on every stage
        self.queue = []
        self.log = []                       # (pass, members)

    def receive(self, m):
        for rid in self.arrivals.get(m, ()):
            self.queue.append(_req(rid, QUEUE_LANES[rid], n_tokens=1000))

    def plan(self, m):
        """One pass of ``_get_new_batch_prefill_raw`` as far as the lane is concerned: the REAL adder for the continuation (it reads
        the floor ``configure_adder`` hands it), then the admission loop with the REAL lane predicates."""
        st = self.stage
        members = ()
        if self.cont_left > 0:
            adder = _adder()
            lanes_p.configure_adder(st, adder)
            req = _req("6-16", 0, prefix=SPAN)
            adder.add_chunked_req(req)
            if [r.rid for r in adder.can_run_list]:
                members = ("6-16",)
                self.cont_left -= 1
        if not members:
            floor = lanes_p.floor_for_pass(st) if lanes_p.skips_by_lane(st) else 0
            for req in list(self.queue):
                if floor and lanes_p.holds(lanes_p.req_lane(req), floor):
                    continue
                members = (req.rid,)
                self.queue.remove(req)
                break
        if members:
            st.forward_ct += 1
        st.mbs[m % 3] = object() if members else None
        st.chunked_req = object() if (self.cont_left > 0 and members == ("6-16",)) else None
        self.log.append((m, members))
        return members


def _run(*, passes=10, rpc_pass=2, delays=(0, 0, 0), cont_left=6, arrivals=None, floor=1, rpcs=None):
    """Pass by pass, lockstep on the pass index (a stage's pass m plans the batch PP0 planned in pass m).  ``rpc_pass``: the front's RPC
    reaches PP0 DURING that pass (after its hook; PP0 stamps at the top of the next); ``rpcs`` = {pass: (floor, epoch)} for several.
    ``delays[j]``: the stamp of list m is absorbed by stage j in its pass m + delays[j] (the real ``absorb_lane_floor`` /
    ``apply_follower``).  Returns (stages, lists)."""
    arrivals = {4: ["7-18"]} if arrivals is None else arrivals
    rpcs = {rpc_pass: (floor, 1)} if rpcs is None else rpcs
    stages = [_Stage(i, cont_left, arrivals) for i in range(3)]
    for s in stages:
        s.queue.append(_req("6-17", 0, n_tokens=1000))   # the lane-0 request PP0 told (store-told admit) long before
    lists = {}
    for m in range(passes):
        for s in stages:
            s.receive(m)
        stamp = lanes_p.pp0_pass(stages[0].stage)
        lists[m] = pr.stamp_lane_floor([], stamp)
        for j, s in enumerate(stages):
            if j and m - delays[j] in lists:
                relayed = list(lists[m - delays[j]])
                _rest, st = pr.absorb_lane_floor(relayed)
                lanes_p.apply_follower(s.stage, st)
        for s in stages:
            s.plan(m)
        if m in rpcs:
            lanes_p.on_rpc(stages[0].stage, _rpc(*rpcs[m]))
    return stages, lists


def _members(stages):
    return [tuple(s.log[m][1] for s in stages) for m in range(len(stages[0].log))]


def _split(stages):
    return [m for m, row in enumerate(_members(stages)) if len(set(row)) > 1]


def _message(stages, m):
    """The real ``#1004`` text for pass m: the stage that launched a batch PP0 did not (or no batch where PP0 named one)."""
    from sglang.srt.managers.scheduler_pp_mixin import pp_slot_disagreement_message

    row = _members(stages)[m]
    follower = next(j for j, mem in enumerate(row) if mem != row[0])
    batch = SimpleNamespace(reqs=[SimpleNamespace(rid=r) for r in row[follower]], extend_num_tokens=151)
    return pp_slot_disagreement_message(pp_rank=follower, mb_id=0, stamp=(1, m + 1, 16000, -1, m + 1, ("7-18", 0, 16000)),
                                        recv_fwd_ct=stages[follower].stage.forward_ct, batch=batch)


def _assert_uniform(tc, stages):
    """Every pass planned the same batch on every stage -- a split fails with the REAL ``#1004`` text naming the pass."""
    bad = _split(stages)
    if bad:
        tc.fail(f"pass {bad[0]}: {_members(stages)[bad[0]]}\n{_message(stages, bad[0])}")


class MetalShapeTest(unittest.TestCase):
    """The NF metal 211536 situation through the real functions."""

    def test_no_stage_admits_the_floor_lane_before_the_common_batch_and_all_park_at_one_boundary(self):
        with _LanesOn():
            stages, _ = _run()
            _assert_uniform(self, stages)
            pp0 = [mem for _m, mem in stages[0].log]
            # RPC reaches PP0 in pass 2 -> stamped at the top of pass 3 with fc=3 -> effective in the plan of batch 4.  Pass 3
            # still plans the in-flight lane-0 chunk on EVERY stage; from pass 4 the continuation is parked, 6-17 held by lane,
            # 7-18 (arrived in pass 4) admitted.
            self.assertEqual(pp0[:4], [("6-16",)] * 4)
            self.assertEqual(pp0[4], ("7-18",))
            self.assertEqual(pp0[5:], [()] * 5)
            for s in stages:
                self.assertEqual(s.cont_left, 2, "every stage parked the continuation after the same chunk")
                self.assertEqual([r.rid for r in s.queue], ["6-17"], "the held lane-0 request was admitted on no stage")
            self.assertEqual({lanes_p.echo(s.stage) for s in stages}, {(1, 1)})

    def test_the_higher_lane_waits_at_most_one_batch_after_the_stamp(self):
        """Wait bound of the lead: the floor stands in the plan of batch ``stamped_fwd_ct + FLOOR_LEAD_BATCHES`` -- one chunk."""
        with _LanesOn():
            stages, _ = _run(arrivals={0: ["7-18"]})
            first = next(m for m, mem in stages[0].log if mem == ("7-18",))
            stamped_pass = 3
            self.assertEqual(lanes_p.FLOOR_LEAD_BATCHES, 1)
            self.assertEqual(first - stamped_pass, lanes_p.FLOOR_LEAD_BATCHES,
                             "7-18 (queued behind nothing but the lane-0 chunk) runs exactly one chunk after the stamp")
            _assert_uniform(self, stages)

    def test_a_stamp_absorbed_one_pass_late_still_binds_every_stage_to_one_batch(self):
        with _LanesOn():
            stages, _ = _run(delays=(0, 1, 1))
            _assert_uniform(self, stages)
            self.assertEqual(stages[0].log[4][1], ("7-18",))

    def test_canfail_without_the_effective_batch_a_late_stamp_splits_the_stages_with_the_1004_text(self):
        """Twin: the lead removed (the floor applies in the pass PP0 stamps it) and one stage one pass late."""
        old = lanes_p.FLOOR_LEAD_BATCHES
        lanes_p.FLOOR_LEAD_BATCHES = 0
        try:
            with _LanesOn():
                stages, _ = _run(delays=(0, 1, 0))
                bad = _split(stages)
                self.assertTrue(bad, "without the binding a stage one pass late plans another batch")
                msg = _message(stages, bad[0])
                self.assertIn("#1004 SLOT DISAGREEMENT", msg)
                self.assertIn("DIFFERENT passes", msg)
        finally:
            lanes_p.FLOOR_LEAD_BATCHES = old

    def test_a_stamp_two_passes_late_is_named_late_not_applied_silently(self):
        with _LanesOn(), self.assertLogs("sglang.srt.weg2.lanes_p", level="WARNING") as cm:
            stages, _ = _run(delays=(0, 2, 0))
            late = [l for l in cm.output if lanes.MARK_PR_FLOOR + " LATE" in l]
            self.assertEqual(len(late), 1, cm.output)
            self.assertIn("rank=1", late[0])
            self.assertEqual(lanes_p.state_of(stages[1].stage).late_n, 1)
            self.assertTrue(_split(stages), "a stamp past its batch cannot be cured, it is the detector's case (#1004)")

    def test_canfail_followers_that_plan_for_themselves_must_skip_the_held_request_like_pp0(self):
        """THE metal defect: the lane skip on PP0 alone.  Twin = the old rule (``owns_admission``) -> PP1/PP2 admit 6-17 in the pass
        PP0 plans nothing, and the real #1004 text names it."""
        orig = lanes_p.skips_by_lane
        lanes_p.skips_by_lane = lanes_p.owns_admission
        try:
            with _LanesOn():
                stages, _ = _run(cont_left=6, arrivals={5: ["7-18"]})
                bad = _split(stages)
                self.assertTrue(bad, "PP1/PP2 admitted a lane-0 request PP0 held")
                row = _members(stages)[bad[0]]
                self.assertEqual(row[0], ())
                self.assertEqual(row[1], ("6-17",))
                msg = _message(stages, bad[0])
                self.assertIn("#1004 SLOT DISAGREEMENT", msg)
                self.assertIn("rids=[6-17]", msg)
        finally:
            lanes_p.skips_by_lane = orig
        with _LanesOn():  # and the fix: the same scenario, one batch sequence on all stages
            stages, _ = _run(cont_left=6, arrivals={5: ["7-18"]})
            _assert_uniform(self, stages)
            self.assertEqual([m for _p, m in stages[0].log if m == ("6-17",)], [], "the held lane-0 request never ran")
            self.assertEqual(stages[0].log[4][1], (), "the pass in which PP0 plans nothing (6-16 parked, 6-17 held) is empty on all stages")

    def test_the_floor_falls_at_once_while_every_request_of_the_group_is_held(self):
        """Everything is held (6-16 parked, 6-17 held): no batch launches, ``forward_ct`` does not move.  The release must still take
        effect -- with nothing in flight the stamp names the CURRENT batch, not the next one."""
        with _LanesOn():
            stages, _ = _run(passes=14, cont_left=3, arrivals={}, rpcs={1: (1, 1), 9: (0, 2)})
            _assert_uniform(self, stages)
            pp0 = [mem for _m, mem in stages[0].log]
            self.assertEqual(pp0[:3], [("6-16",)] * 3)
            self.assertEqual(pp0[3:10], [()] * 7, "held on every stage while the floor stands")
            self.assertEqual(pp0[10], ("6-17",), "the release RPC of pass 9 is stamped in pass 10 and stands in that very plan")
            for s in stages:
                self.assertEqual(lanes_p.echo(s.stage), (0, 2))
                self.assertEqual(lanes_p.state_of(s.stage).pending, [])

    def test_a_release_stamped_while_the_last_batches_still_drain_is_relaxed_when_the_group_is_idle(self):
        """The release RPC lands in the pass right after the last batch (the three slots still hold stale batches, work IS in flight:
        ``eff = fc + 1``).  Nothing ever launches again, so PP0 re-sends the same epoch with ``eff = fc`` as soon as its slots are empty,
        every follower takes the lower value, and all stages release in the same pass."""
        with _LanesOn(), self.assertLogs("sglang.srt.weg2.lanes_p", level="INFO") as cm:
            stages, _ = _run(passes=14, cont_left=3, arrivals={}, rpcs={1: (1, 1), 3: (0, 2)})
            _assert_uniform(self, stages)
            pp0 = [mem for _m, mem in stages[0].log]
            ran = [m for m, mem in stages[0].log if mem == ("6-17",)]
            self.assertEqual(len(ran), 1, pp0)
            self.assertTrue(ran[0] <= 3 + 1 + 3, "released within the three passes it takes the slots to empty")
            self.assertTrue(any("RELAX floor=0 epoch=2 rank=0" in l for l in cm.output), cm.output)
            for s in stages:
                self.assertEqual(lanes_p.echo(s.stage), (0, 2))

    def test_a_follower_takes_a_lower_effective_batch_for_an_epoch_that_is_still_waiting(self):
        with _LanesOn():
            s1 = _stage(1)
            s1.forward_ct = 0
            lanes_p.apply_follower(s1, pr.Weg2PpLaneFloor(floor=1, epoch=1, eff=5, stamped=4))
            self.assertEqual(lanes_p.state_of(s1).pending, [(1, 1, 5, 4)])
            lanes_p.apply_follower(s1, pr.Weg2PpLaneFloor(floor=1, epoch=1, eff=3, stamped=4))   # relaxed: lower
            self.assertEqual(lanes_p.state_of(s1).pending, [(1, 1, 3, 4)])
            lanes_p.apply_follower(s1, pr.Weg2PpLaneFloor(floor=1, epoch=1, eff=9, stamped=4))   # never raised again
            self.assertEqual(lanes_p.state_of(s1).pending, [(1, 1, 3, 4)])

    def test_canfail_a_lead_with_nothing_in_flight_would_hold_everything_for_good(self):
        """Twin: the lead applied even when idle -> the release waits for a batch that never launches."""
        orig = lanes_p.work_in_flight
        lanes_p.work_in_flight = lambda sched: True
        try:
            with _LanesOn():
                stages, _ = _run(passes=20, cont_left=3, arrivals={}, rpcs={1: (1, 1), 9: (0, 2)})
                self.assertEqual([m for m, mem in stages[0].log if mem == ("6-17",)], [], "6-17 never ran: the floor never fell")
                self.assertEqual(lanes_p.echo(stages[0].stage), (1, 1))
        finally:
            lanes_p.work_in_flight = orig


# ---------------------------------------------------------------------------
# the predicates and the queue of the effective batch
# ---------------------------------------------------------------------------

def _follower(rank, *, effective=None, pp_size=3):
    s = _stage(rank, pp_size)
    s._pp_admission_incoming_effective = effective
    return s


class PredicateTest(unittest.TestCase):
    def test_skips_by_lane_is_every_rank_that_decides_and_no_rank_that_executes_a_schedule(self):
        self.assertTrue(lanes_p.skips_by_lane(_follower(0)))                  # PP0
        self.assertTrue(lanes_p.skips_by_lane(_follower(0, pp_size=1)))       # non-PP boot
        self.assertTrue(lanes_p.skips_by_lane(_follower(1)))                  # follower, no row carrier: plans for itself
        self.assertTrue(lanes_p.skips_by_lane(_follower(2)))
        self.assertFalse(lanes_p.skips_by_lane(_follower(1, effective={"R": 5})))   # executes PP0's row
        self.assertFalse(lanes_p.skips_by_lane(_follower(2, effective={})))         # '{}' = admit nothing, still a schedule
        self.assertFalse(lanes_p.executes_schedule(_follower(0, effective={"R": 5})))  # PP0 builds the row, never executes one

    def test_the_chunk_cap_exists_only_where_a_schedule_carries_it(self):
        s0 = _stage(0)
        self.assertFalse(lanes_p.cap_applies(s0), "a P form without a row carrier: followers would not narrow the chunk")
        s0.pp_flip_counters = object()
        self.assertTrue(lanes_p.cap_applies(s0))
        self.assertFalse(lanes_p.cap_applies(_stage(1)))
        self.assertTrue(lanes_p.cap_applies(_stage(0, pp_size=1)))

    def test_an_inert_cap_is_named_once_and_the_adder_keeps_the_normal_chunk(self):
        with _LanesOn() as on:
            on.chunk(4096)
            s = _stage(0)
            lanes_p.state_of(s, create=True)
            adder = _adder(page_size=64)
            with self.assertLogs("sglang.srt.weg2.lanes_p", level="WARNING") as cm:
                lanes_p.configure_adder(s, adder)
                lanes_p.configure_adder(s, _adder(page_size=64))
            self.assertEqual(sum("CHUNK-CAP INERT" in l for l in cm.output), 1)
            self.assertEqual(adder.lane_chunk_cap_tokens, 0)

    def test_the_scheduler_loop_and_the_adder_read_the_same_effective_floor(self):
        from sglang.srt.managers import scheduler as sch

        src = open(sch.__file__).read()
        self.assertIn("_lane_floor = _lanes_p.floor_for_pass(self) if _lanes_p.skips_by_lane(self) else 0", src)
        self.assertNotIn("_lanes_p.floor_for_pass(self) if _lanes_p.owns_admission(self) else 0", src)
        self.assertIn("adder.lane_floor = floor_for_pass(sched)", open(lanes_p.__file__).read())


class EffectiveBatchTest(unittest.TestCase):
    def _stage_with_ct(self, rank, fc):
        s = _stage(rank)
        s.forward_ct = fc
        return s

    def test_pp0_with_nothing_in_flight_names_the_current_batch(self):
        with _LanesOn():
            s0 = self._stage_with_ct(0, 7)             # no chunked continuation, no occupied slot
            self.assertFalse(lanes_p.work_in_flight(s0))
            lanes_p.on_rpc(s0, _rpc(2, 1))
            stamp = lanes_p.pp0_pass(s0)
            self.assertEqual((stamp.eff, stamp.stamped), (7, 7))
            self.assertEqual(lanes_p.floor_for_pass(s0), 2, "idle: the very next plan stands on the floor")

    def test_work_in_flight_is_a_continuation_or_an_occupied_slot(self):
        s = _stage(0)
        self.assertFalse(lanes_p.work_in_flight(s))
        s.chunked_req = object()
        self.assertTrue(lanes_p.work_in_flight(s))
        s.chunked_req = None
        s.mbs = [None, None, None]
        self.assertFalse(lanes_p.work_in_flight(s))
        s.mbs[1] = object()
        self.assertTrue(lanes_p.work_in_flight(s))

    def test_pp0_names_the_batch_and_applies_the_floor_only_in_its_plan(self):
        with _LanesOn():
            s0 = self._stage_with_ct(0, 7)
            s0.chunked_req = object()                  # a lower-lane chunk is in flight
            lanes_p.on_rpc(s0, _rpc(2, 1))
            stamp = lanes_p.pp0_pass(s0)
            self.assertEqual((stamp.floor, stamp.epoch, stamp.eff, stamp.stamped), (2, 1, 8, 7))
            self.assertEqual(lanes_p.floor_for_pass(s0), 0, "batch 7 is planned before the floor stands")
            s0.forward_ct = 8
            self.assertEqual(lanes_p.floor_for_pass(s0), 2)

    def test_the_standing_stamp_repeats_the_same_batch_every_pass(self):
        with _LanesOn():
            s0 = self._stage_with_ct(0, 3)
            s0.chunked_req = object()
            lanes_p.on_rpc(s0, _rpc(1, 1))
            first = lanes_p.pp0_pass(s0)
            s0.forward_ct = 9
            again = lanes_p.pp0_pass(s0)
            self.assertEqual((first.eff, first.stamped), (again.eff, again.stamped), "never recomputed")

    def test_a_follower_applies_at_the_named_batch_not_before_not_when_the_stamp_arrives(self):
        with _LanesOn():
            s1 = self._stage_with_ct(1, 5)
            moved = lanes_p.apply_follower(s1, pr.Weg2PpLaneFloor(floor=1, epoch=1, eff=8, stamped=7))
            self.assertFalse(moved)
            self.assertEqual(lanes_p.floor_for_pass(s1), 0)
            s1.forward_ct = 7
            self.assertEqual(lanes_p.floor_for_pass(s1), 0)
            s1.forward_ct = 8
            self.assertEqual(lanes_p.floor_for_pass(s1), 1)
            self.assertEqual(lanes_p.echo(s1), (1, 1))

    def test_two_epochs_in_flight_apply_in_order_each_at_its_own_batch(self):
        with _LanesOn():
            s1 = self._stage_with_ct(1, 0)
            lanes_p.apply_follower(s1, pr.Weg2PpLaneFloor(floor=2, epoch=1, eff=4, stamped=3))
            lanes_p.apply_follower(s1, pr.Weg2PpLaneFloor(floor=0, epoch=2, eff=6, stamped=5))
            seen = []
            for fc in range(8):
                s1.forward_ct = fc
                seen.append(lanes_p.floor_for_pass(s1))
            self.assertEqual(seen, [0, 0, 0, 0, 2, 2, 0, 0])

    def test_a_standing_stamp_of_a_known_epoch_is_ignored(self):
        with _LanesOn():
            s1 = self._stage_with_ct(1, 0)
            stamp = pr.Weg2PpLaneFloor(floor=1, epoch=3, eff=2, stamped=1)
            lanes_p.apply_follower(s1, stamp)
            s1.forward_ct = 2
            self.assertEqual(lanes_p.floor_for_pass(s1), 1)
            lanes_p.apply_follower(s1, stamp)
            self.assertEqual(lanes_p.state_of(s1).pending, [])
            self.assertEqual(lanes_p.state_of(s1).changes, 1)

    def test_a_stand_in_without_a_batch_counter_and_a_legacy_stamp_apply_at_once(self):
        with _LanesOn():
            s = _stage(1)                                   # no forward_ct
            lanes_p.apply_follower(s, pr.Weg2PpLaneFloor(floor=1, epoch=1, eff=9, stamped=8))
            self.assertEqual(lanes_p.floor_for_pass(s), 1)
            s2 = self._stage_with_ct(1, 0)
            lanes_p.apply_follower(s2, pr.Weg2PpLaneFloor(floor=1, epoch=1))   # eff = -1: a stamp of the old wire
            self.assertEqual(lanes_p.floor_for_pass(s2), 1)

    def test_the_marker_carries_the_effective_batch(self):
        with _LanesOn(), self.assertLogs("sglang.srt.weg2.lanes_p", level="WARNING") as cm:
            s0 = self._stage_with_ct(0, 4)
            s0.chunked_req = object()
            lanes_p.on_rpc(s0, _rpc(1, 1))
            lanes_p.pp0_pass(s0)
            s0.forward_ct = 5
            lanes_p.floor_for_pass(s0)
        line = next(l for l in cm.output if lanes.MARK_PR_FLOOR in l)
        self.assertIn("PR LANE-FLOOR floor=1 epoch=1 rank=0 prev_floor=0 eff_fwd_ct=5 at_fwd_ct=5 stamped_fwd_ct=4 lead=1", line)

    def test_the_wire_struct_pickles_and_keeps_its_old_two_field_form_valid(self):
        s = pr.Weg2PpLaneFloor(floor=2, epoch=7, eff=12, stamped=11)
        self.assertEqual(pickle.loads(pickle.dumps(s)), s)
        old = pr.Weg2PpLaneFloor(floor=2, epoch=7)
        self.assertEqual((old.eff, old.stamped), (-1, -1))

    def test_a_single_rank_boot_applies_the_rpc_at_once(self):
        with _LanesOn():
            s = _stage(0, pp_size=1)
            s.forward_ct = 4
            lanes_p.on_rpc(s, _rpc(1, 1))
            self.assertEqual(lanes_p.floor_for_pass(s), 1)


# ---------------------------------------------------------------------------
# the front: a rank death reaches the held streams
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, fail=False):
        self.written, self.eof, self.prepared, self.fail = [], False, False, fail

    async def prepare(self, request):
        self.prepared = True

    async def write(self, b):
        if self.fail:
            raise ConnectionResetError("gone")
        self.written.append(bytes(b))

    async def write_eof(self):
        self.eof = True


class _Req(dict):
    path = "/v1/chat/completions"


def _front():
    from test_weg2_lanes_l4_1008 import _front as f4
    return f4("D")


def _run_on(coro):
    from test_weg2_lanes_l4_1008 import _switch
    with _switch(True):
        return asyncio.run(coro)


class GroupDeathTest(unittest.TestCase):
    def test_a_rank_death_answers_every_held_stream_with_one_named_error_event_at_once(self):
        async def body():
            from test_weg2_lanes_l4_1008 import _note
            f = _front()
            f._lane_state().set_floor(1)
            _note(f, "weg2-6-16", 0)
            _note(f, "weg2-6-17", 0)
            _note(f, "weg2-7-18", 1)
            lc = LC.ctl(f)
            r16, r17 = _Resp(), _Resp()
            f._lane_stream_open("weg2-6-16", _Req(), r16)           # opened early (keepalive), no group answered
            f._lane_stream_open("weg2-6-17", _Req(), r17)
            orig = LC.boot_death_reason
            LC.boot_death_reason = lambda fr: "group P died (RANK_EXCEPTION: RuntimeError: #1004 SLOT DISAGREEMENT)"
            try:
                t0 = time.time()
                with self.assertLogs("weg2.front", level="WARNING") as cm:
                    await LC.step(f, t0)
                for r in (r16, r17):
                    self.assertEqual(len(r.written), 1)
                    ev = json.loads(r.written[0].decode().split("data: ", 1)[1])
                    self.assertEqual(ev["error"]["code"], 503)
                    self.assertIn("group P died (RANK_EXCEPTION", ev["error"]["message"])
                    self.assertIn("WEG2 lane hold ended", ev["error"]["message"])
                    self.assertTrue(r.eof, "the stream is closed: the client is not left on an empty 200")
                self.assertEqual(sum(lanes_marker in l for l in cm.output for lanes_marker in (LC.MARK_GROUP_DEAD,)), 1)
                self.assertEqual(f.counters["lane_group_dead_streams"], 2)
                self.assertEqual(lc.streams, {})
                # once per death: the next tick writes nothing again
                await LC.step(f, t0 + 5)
                self.assertEqual((len(r16.written), len(r17.written)), (1, 1))
            finally:
                LC.boot_death_reason = orig
        _run_on(body())

    def test_an_early_stream_not_yet_opened_is_opened_and_answered_too(self):
        async def body():
            from test_weg2_lanes_l4_1008 import _note
            f = _front()
            f._lane_state().set_floor(1)
            _note(f, "weg2-6-16", 0)
            _note(f, "weg2-7-18", 1)
            lc = LC.ctl(f)
            resp, req = _Resp(), _Req()
            lc.pre["weg2-6-16"] = {"request": req, "resp": None, "lock": None, "taken": False, "held_t0": None}
            orig_sse, orig_dead = LC.new_sse_response, LC.boot_death_reason
            LC.new_sse_response = lambda: resp
            LC.boot_death_reason = lambda fr: "group P died (RANK_EXCEPTION)"
            try:
                await LC.step(f, time.time())
                self.assertTrue(resp.prepared)
                self.assertEqual(len(resp.written), 2)                  # the first keepalive of the opening, then the error event
                self.assertIn(b"WEG2 lane hold ended", resp.written[-1])
                self.assertTrue(resp.eof)
            finally:
                LC.new_sse_response, LC.boot_death_reason = orig_sse, orig_dead
        _run_on(body())

    def test_a_client_that_left_costs_nothing_and_no_floor_asks_for_no_read(self):
        async def body():
            from test_weg2_lanes_l4_1008 import _note
            f = _front()
            _note(f, "weg2-6-16", 0)
            lc = LC.ctl(f)
            calls = []
            orig = LC.boot_death_reason
            LC.boot_death_reason = lambda fr: calls.append(1) or "dead"
            try:
                f._lane_stream_open("weg2-6-16", _Req(), _Resp(fail=True))
                await LC.step(f, time.time())                          # floor 0: nothing held, the lifecycle is not read
                self.assertEqual(calls, [])
                _note(f, "weg2-7-18", 1)                               # keeps the floor up through the loop's own reconcile
                f._lane_state().set_floor(1)
                await LC.step(f, time.time() + 5)                      # the dead client's write fails: swallowed
                self.assertEqual(calls, [1])
                self.assertEqual(f.counters["lane_group_dead_streams"], 0)
            finally:
                LC.boot_death_reason = orig
        _run_on(body())

    def test_the_death_is_read_from_the_lifecycle_the_rank_wrote(self):
        from sglang.srt.environ import envs
        from sglang.srt.weg2 import state_file as sf

        with tempfile.TemporaryDirectory() as root:
            d = sf.init(root, "b1", "boot", {})
            sf.transition(d, "launching")
            sf.transition(d, "serving")
            old = os.environ.get("WEG2_STATE_DIR")
            os.environ["WEG2_STATE_DIR"] = d
            try:
                self.assertIsNone(LC.boot_death_reason(None), "serving is no death")
                exc = RuntimeError("#1004 SLOT DISAGREEMENT (one layout per process): PP1 is launching slot 0")
                self.assertTrue(sf.note_rank_death(d, group="P", tp_rank=0, pp_rank=1, exc=exc, detail=str(exc)))
                why = LC.boot_death_reason(None)
                self.assertIn("group P died (RANK_EXCEPTION", why)
                self.assertIn("#1004 SLOT DISAGREEMENT", why)
                os.environ.pop("WEG2_STATE_DIR")
                self.assertIsNone(LC.boot_death_reason(None), "no state directory: no reading, no death")
                os.environ["WEG2_STATE_DIR"] = os.path.join(root, "nowhere")
                self.assertIsNone(LC.boot_death_reason(None), "no file: no death")
            finally:
                if old is None:
                    os.environ.pop("WEG2_STATE_DIR", None)
                else:
                    os.environ["WEG2_STATE_DIR"] = old

    def test_the_switch_off_front_never_reads_the_lifecycle(self):
        async def body():
            from test_weg2_lanes_l4_1008 import _note, _switch
            f = _front()
            calls = []
            orig = LC.boot_death_reason
            LC.boot_death_reason = lambda fr: calls.append(1) or "dead"
            try:
                with _switch(False):
                    await LC.step(f, time.time())
                self.assertEqual(calls, [])
            finally:
                LC.boot_death_reason = orig
        asyncio.run(body())

    def test_the_step_calls_the_death_tick_before_the_keepalive(self):
        src = open(LC.__file__).read()
        i = src.index("async def step(")
        blk = src[i:i + 2400]
        self.assertLess(blk.index("await group_death_tick(fr, lc, now)"), blk.index("await keepalive_tick(fr, now)"))


# ---------------------------------------------------------------------------
# fix 2 meets the effective batch
# ---------------------------------------------------------------------------

class FixTwoMeetsTheEffectiveBatchTest(unittest.TestCase):
    def test_the_arrival_wait_bound_covers_the_wait_for_the_effective_batch(self):
        """Fix 2 keeps the P phase open for a floor-lane arrival (``take_wait``) for at most ``ARRIVING_MAX_S``.  The arrival now waits for
        (a) PP0's pass top to take the floor RPC (its ack, unchanged: <= one chunk) and (b) the plan of batch ``stamped + lead`` (<= ``lead`` further
        chunks).  Metal chunk time 16384 tokens ~ 2.2-4.6 s (P log 21:22:16-21:22:32): two chunks stay far below the bound.  Pinned as the
        arithmetic the report names, with the longest chunk seen on metal."""
        longest_chunk_s = 5.0   # PASS-STALL / PP-BUBBLE lines of boot 211536: gpu-ms 2169-2488 + bubble <= 1.4 s; probe 2b 16k ~ 4-5 s
        wait = (1 + lanes_p.FLOOR_LEAD_BATCHES) * longest_chunk_s
        self.assertLess(wait * 2, LC.ARRIVING_MAX_S, "even at twice the longest chunk")

    def test_the_hold_queued_rule_of_fix_2_is_untouched_by_the_lead(self):
        """Fix 2's D rule (``hold_queued``) and the front's sync sweep read the FRONT's floor, not P's batch binding: nothing of fix 3
        sits on those paths."""
        import sglang.srt.weg2.d_park_runtime as dpr

        src = open(dpr.__file__).read()
        self.assertNotIn("FLOOR_LEAD_BATCHES", src)
        self.assertNotIn("lanes_p", src)


if __name__ == "__main__":
    unittest.main()
