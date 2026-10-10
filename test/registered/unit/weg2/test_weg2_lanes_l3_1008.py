# SPDX-License-Identifier: Apache-2.0
"""PRIORITY LANES, part L3 (P side) -- plan deskq/PLAN-PRIO-LANES-1008.md section 2 "P (Prefill, PP=3)".

There is no commandable in-flight park on P. The lane park is the pressure park (#679 ->
``weg2_pool_parked`` -> ``process_pending_weg2_park`` -> ``release_kv_cache(is_insert=True)``) with the lane
floor as the trigger, and it must act on ONE floor in the SAME pass on all three PP stages, or the stages admit
different batches (``#1004 SLOT DISAGREEMENT``).

What is pinned here, in the order of the risk:

1. RANK-UNIFORMITY. PP0 takes the floor from the RPC at the top of its next pass, applies it THEN and stamps it on
   list m; a follower takes it off list m after relaying and applies it in pass m (the ``Weg2PpRoomCap`` convention).
   Driven through the real functions (``lanes_p.on_rpc / pp0_pass / apply_follower``, ``pp_room_vote.stamp_lane_floor /
   absorb_lane_floor``) and the REAL ``PrefillAdder.add_chunked_req`` of every stage. CAN-FAIL: the same RPC applied by each
   stage when it reaches it (different delays) makes the stages admit different batches -- the #1004 shape, named by the
   real ``pp_slot_disagreement_message`` -- and the stamp removes it.
2. The park: a running lower-lane prefill gets no next chunk (in place), its rows go back with ``is_insert=True`` at the
   head of the next step, with or without a waiter; a floor that fell first voids the park.
3. The resume admits only the rest (prefix kept, no recomputation of the anchor).
4. Smaller chunks for a lower lane while a higher lane waits; 0 = off.
5. Switch off / no RPC: no state, no stamp, byte-identical lists, adder at its class defaults.
"""
from __future__ import annotations

import os
import pickle
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.schedule_batch import Req  # noqa: E402
from sglang.srt.managers.schedule_policy import PrefillAdder  # noqa: E402
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefResult, IncLockRefResult  # noqa: E402
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler  # noqa: E402
from sglang.srt.utils.common import Range  # noqa: E402
from sglang.srt.weg2 import lanes, lanes_p  # noqa: E402
from sglang.srt.weg2 import park as pk  # noqa: E402
from sglang.srt.weg2 import pp_room_vote as pr  # noqa: E402

AVAIL = 200000
CHUNK = 16384
SPAN = 40960  # the computed span of the running prefill at the chunk boundary
N_TOK = 99000


# ---------------------------------------------------------------------------
# harness: the real adder, a PP stage stand-in, the lane switch
# ---------------------------------------------------------------------------

def _tree_cache():
    tc = MagicMock()
    tc.full_evictable_size.return_value = 0
    tc.swa_evictable_size.return_value = 0
    tc.evictable_size.return_value = 0
    tc.disable = False
    tc.inc_lock_ref.return_value = IncLockRefResult()
    tc.dec_lock_ref.return_value = DecLockRefResult()
    return tc


def _allocator(avail):
    al = MagicMock()
    al.full_available_size.return_value = avail
    al.swa_available_size.return_value = avail
    al.available_size.return_value = avail
    return al


def _req(rid, priority, n_tokens=N_TOK, prefix=0, max_new_tokens=8):
    r = MagicMock(spec=Req)
    r.rid = str(rid)
    r.priority = priority
    r.prefix_indices = list(range(prefix))
    r.full_untruncated_fill_ids = list(range(n_tokens))
    r.output_ids = []
    r.host_hit_length = 0
    r.swa_host_hit_length = 0
    r.sampling_params = SimpleNamespace(max_new_tokens=max_new_tokens, ignore_eos=False)
    r.time_stats = SimpleNamespace(wait_queue_entry_time=0)
    r.retracted_stain = False
    r.born_spilled = False
    r.born_spilled_deep = False
    r.last_node = None
    r.mamba_pool_idx = None
    r.finished.return_value = False
    r.needs_host_load_back.return_value = False
    r.set_extend_range = MagicMock(side_effect=lambda a, b: setattr(r, "extend_range", Range(a, b)))
    return r


def _adder(**kw):
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    rb = MagicMock()
    rb.reqs = []
    return PrefillAdder(
        page_size=kw.pop("page_size", 1),
        tree_cache=_tree_cache(),
        token_to_kv_pool_allocator=_allocator(AVAIL),
        running_batch=rb,
        new_token_ratio=1.0,
        rem_input_tokens=1_000_000,
        rem_chunk_tokens=kw.pop("rem_chunk_tokens", CHUNK),
        num_mixed_decode_tokens=0,
        priority_scheduling_preemption_threshold=0,
        **kw,
    )


def _stage(rank, pp_size=3):
    """One scheduler as the lane code sees it: ``ps`` and nothing else (the state attribute appears on demand)."""
    return SimpleNamespace(ps=SimpleNamespace(pp_size=pp_size, pp_rank=rank), waiting_queue=[])


class _LanesOn:
    """SGLANG_WEG2_LANES=1 on group P with the park on (what a lane boot has)."""

    def __enter__(self):
        self.keys = ("SGLANG_WEG2_LANES", "SGLANG_WEG2_GROUP", "SGLANG_WEG2_PARK", "SGLANG_WEG2_LANE_PREEMPT_CHUNK_TOKENS")
        self.old = {k: os.environ.get(k) for k in self.keys}
        os.environ["SGLANG_WEG2_LANES"] = "1"
        os.environ["SGLANG_WEG2_GROUP"] = "P"
        os.environ.pop("SGLANG_WEG2_PARK", None)
        os.environ.pop("SGLANG_WEG2_LANE_PREEMPT_CHUNK_TOKENS", None)
        return self

    def chunk(self, n):
        os.environ["SGLANG_WEG2_LANE_PREEMPT_CHUNK_TOKENS"] = str(n)

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _rpc(floor, epoch):
    return SimpleNamespace(floor=floor, epoch=epoch)


def _run_chunk_decision(stage, rid="R", priority=0):
    """What one stage's adder does with its chunked continuation `rid` in this pass: the REAL add_chunked_req with the
    floor the stage applies. Returns (admitted rids, the continuation object)."""
    adder = _adder()
    lanes_p.configure_adder(stage, adder)
    req = _req(rid, priority, prefix=SPAN)
    out = adder.add_chunked_req(req)
    return [r.rid for r in adder.can_run_list], req, out


# ---------------------------------------------------------------------------
# 1. rank-uniformity
# ---------------------------------------------------------------------------

def _pass_with_stamp(stages, m, rpc_at):
    """One PP pass m through the real stamp/absorb functions: PP0's pass top, the wire list, each follower's relay-then-absorb,
    then every stage's chunk decision. `rpc_at` = {pass: (floor, epoch)} the front's RPC that reaches PP0 AFTER its pass-m top
    (it is a request handled in process_input_requests, which follows the hook)."""
    s0 = stages[0]
    stamp = lanes_p.pp0_pass(s0)
    wire = pr.stamp_lane_floor([], stamp)
    relayed = wire
    for s in stages[1:]:
        sent_on = list(relayed)  # the follower relays BEFORE it absorbs (the next stage still sees the stamp)
        rest, st = pr.absorb_lane_floor(relayed)
        lanes_p.apply_follower(s, st)
        relayed = sent_on
    members = [tuple(_run_chunk_decision(s)[0]) for s in stages]
    if m in rpc_at:  # the RPC reaches PP0's scheduler during this pass (after the hook)
        lanes_p.on_rpc(s0, _rpc(*rpc_at[m]))
    return members


def _pass_without_stamp(stages, m, rpc_at):
    """CAN-FAIL twin: no stamp. The RPC (one per stage, as a broadcast RPC would be) is applied by each stage when it reaches it:
    PP0 at pass `at`, PP1 `d1` passes later, PP2 `d2` passes later."""
    for i, s in enumerate(stages):
        for at, (floor, epoch) in rpc_at.items():
            if m == at + DELAY[i]:
                st = lanes_p.state_of(s, create=True)
                st.offer(floor, epoch)
                st.promote()
    return [tuple(_run_chunk_decision(s)[0]) for s in stages]


DELAY = (0, 1, 3)  # PP1 learns one pass later, PP2 three passes later (a broadcast RPC has no common pass)


class RankUniformityTest(unittest.TestCase):
    def test_the_stamp_puts_all_three_stages_on_one_floor_in_the_same_pass(self):
        with _LanesOn():
            stages = [_stage(i) for i in range(3)]
            rpc_at = {5: (1, 1)}
            seen = [_pass_with_stamp(stages, m, rpc_at) for m in range(12)]
            for m, members in enumerate(seen):
                self.assertEqual(len(set(members)), 1, f"pass {m}: the stages admitted different batches {members}")
            # before the floor the continuation runs (a chunk is admitted); after it, on every stage, it does not
            self.assertEqual(seen[5][0], ("R",))
            self.assertEqual(seen[6][0], ())
            self.assertEqual(seen[11][0], ())
            # every stage applies the same (floor, epoch)
            self.assertEqual({lanes_p.echo(s) for s in stages}, {(1, 1)})

    def test_canfail_without_the_stamp_the_stages_admit_different_batches_the_1004_shape(self):
        with _LanesOn():
            stages = [_stage(i) for i in range(3)]
            rpc_at = {5: (1, 1)}
            seen = [_pass_without_stamp(stages, m, rpc_at) for m in range(12)]
            split = [m for m, members in enumerate(seen) if len(set(members)) > 1]
            self.assertEqual(split, [5, 6, 7], "PP0 parks at 5, PP1 at 6, PP2 at 8: passes 5-7 split the group")
            # and the real #1004 message names exactly such a pass
            from sglang.srt.managers.scheduler_pp_mixin import pp_slot_disagreement_message

            batch = SimpleNamespace(reqs=[SimpleNamespace(rid="R")], extend_num_tokens=CHUNK)
            msg = pp_slot_disagreement_message(pp_rank=1, mb_id=split[0] % 3, stamp=(0, 7, 0, 0, 5, ("R", SPAN, SPAN)),
                                               recv_fwd_ct=split[0], batch=batch)
            self.assertIn("#1004 SLOT DISAGREEMENT", msg)
            self.assertIn("DIFFERENT passes", msg)

    def test_the_stamp_cures_the_same_delays(self):
        """Same delays the CAN-FAIL twin used: with the stamp the RPC reaching PP0 late or early moves all stages together."""
        with _LanesOn():
            stages = [_stage(i) for i in range(3)]
            rpc_at = {5: (2, 1), 9: (1, 2), 13: (0, 3)}
            seen = [_pass_with_stamp(stages, m, rpc_at) for m in range(20)]
            for m, members in enumerate(seen):
                self.assertEqual(len(set(members)), 1, f"pass {m}: {members}")
            # floor 2 holds lane 0 (6..), floor 1 still holds lane 0 (10..), floor 0 releases it (14..)
            self.assertEqual([seen[m][0] for m in (5, 6, 10, 13, 14, 19)],
                             [("R",), (), (), (), ("R",), ("R",)])

    def test_pp0_applies_in_the_pass_it_stamps_not_when_the_rpc_arrives(self):
        with _LanesOn():
            s0 = _stage(0)
            self.assertIsNone(lanes_p.pp0_pass(s0))  # no RPC yet: nothing to stamp
            lanes_p.on_rpc(s0, _rpc(3, 1))
            self.assertEqual(lanes_p.floor_for_pass(s0), 0, "PP0 must not apply before its next pass top (followers could not)")
            stamp = lanes_p.pp0_pass(s0)
            self.assertEqual((stamp.floor, stamp.epoch), (3, 1))
            self.assertEqual(lanes_p.floor_for_pass(s0), 3)

    def test_the_standing_value_is_stamped_every_pass_after_the_first_epoch(self):
        with _LanesOn():
            s0 = _stage(0)
            lanes_p.on_rpc(s0, _rpc(2, 4))
            first = lanes_p.pp0_pass(s0)
            again = lanes_p.pp0_pass(s0)
            self.assertEqual((first.floor, first.epoch), (again.floor, again.epoch))
            self.assertEqual((again.floor, again.epoch), (2, 4))

    def test_followers_ignore_the_rpc_itself_and_wait_for_the_stamp(self):
        with _LanesOn():
            s1 = _stage(1)
            lanes_p.on_rpc(s1, _rpc(3, 1))  # the control request also travels the chain: PP1 sees it
            self.assertIsNone(lanes_p.state_of(s1))
            self.assertEqual(lanes_p.floor_for_pass(s1), 0)

    def test_a_late_older_rpc_cannot_move_the_floor_back(self):
        with _LanesOn():
            s0 = _stage(0)
            lanes_p.on_rpc(s0, _rpc(2, 3))
            lanes_p.on_rpc(s0, _rpc(1, 2))  # epoch 2 overtaken by epoch 3 on the HTTP side
            stamp = lanes_p.pp0_pass(s0)
            self.assertEqual((stamp.floor, stamp.epoch), (2, 3))
            lanes_p.on_rpc(s0, _rpc(1, 2))  # and again after it was applied
            self.assertEqual((lanes_p.pp0_pass(s0).floor), 2)
            self.assertEqual(lanes_p.state_of(s0).stale_n, 2)

    def test_the_newest_of_two_rpcs_between_passes_wins(self):
        with _LanesOn():
            s0 = _stage(0)
            lanes_p.on_rpc(s0, _rpc(1, 1))
            lanes_p.on_rpc(s0, _rpc(2, 2))
            stamp = lanes_p.pp0_pass(s0)
            self.assertEqual((stamp.floor, stamp.epoch), (2, 2))

    def test_the_marker_is_logged_by_every_stage_with_floor_and_epoch(self):
        with _LanesOn():
            stages = [_stage(i) for i in range(3)]
            with self.assertLogs("sglang.srt.weg2.lanes_p", level="WARNING") as cm:
                for m in range(8):
                    _pass_with_stamp(stages, m, {2: (1, 1)})
            lines = [l for l in cm.output if lanes.MARK_PR_FLOOR in l]
            self.assertEqual(len(lines), 3, "one line per stage, on the change only")
            for rank in range(3):
                self.assertTrue(any(f"{lanes.MARK_PR_FLOOR} floor=1 epoch=1 rank={rank}" in l for l in lines), lines)

    def test_the_fact_echoes_the_applied_value_so_pp0_can_read_one_epoch(self):
        with _LanesOn():
            stages = [_stage(i) for i in range(3)]
            for m in range(6):
                _pass_with_stamp(stages, m, {1: (2, 1)})
            book = pr.RoomBook()
            book.begin_pass()
            facts = []
            for s in stages[1:]:
                lf, le = lanes_p.echo(s)
                facts.append(pr.Weg2PpRoomFactLane(rank=s.ps.pp_rank, executed=3, room=1000, available=900, payable=100,
                                                   reported=100, lane_floor=lf, lane_epoch=le))
            self.assertEqual(book.absorb(facts), 2)
            self.assertEqual(book.lane_echo(), {1: (2, 1), 2: (2, 1)})
            # a plain fact (a boot without lanes) is no echo
            book.absorb([pr.Weg2PpRoomFact(rank=1, executed=4, room=1, available=1, payable=0, reported=0)])
            self.assertEqual(book.lane_echo(), {2: (2, 1)})

    def test_the_stamp_survives_the_wire_pickle(self):
        s = pr.Weg2PpLaneFloor(floor=2, epoch=7)
        self.assertEqual(pickle.loads(pickle.dumps(s)), s)
        f = pr.Weg2PpRoomFactLane(rank=2, executed=1, room=2, available=1, payable=1, reported=1, lane_floor=2, lane_epoch=7)
        self.assertEqual(pickle.loads(pickle.dumps(f)), f)
        self.assertIsInstance(f, pr.Weg2PpRoomFact)


# ---------------------------------------------------------------------------
# 1b. the REAL follower path: row authority, the #992 gate, the park on PP1/PP2 (L3 review, findings 1 and 2)
# ---------------------------------------------------------------------------

def _park_stage(rank, pp_size=3):
    """A PP stage that can run the real head-of-step park: the lane-state stage plus what `process_pending_weg2_park` reads."""
    s = _stage(rank, pp_size)
    s.chunked_req = None
    s.tree_cache = object()
    s.queued = []
    s._add_request_to_queue = lambda req, is_retracted=False: s.queued.append((req.rid, is_retracted))
    return s


def _continuation_on(stage, rid="R", priority=0):
    """The chunked continuation this stage holds at the chunk boundary: its OWN Req object (each rank has its own rows)."""
    req = _req(rid, priority, prefix=SPAN)
    req.req_pool_idx = 3
    req.inflight_middle_chunks = 0
    req.origin_input_ids = list(range(N_TOK))
    req.reset_for_retract = MagicMock()
    stage.chunked_req = req
    return req


def _stage_pass(stage, incoming, *, old_gate=False):
    """What `Scheduler._get_new_batch_prefill_raw` does with `self.chunked_req` on this stage: the REAL #992 predicate
    (`Scheduler`'s `_pp_chunked_continuation_not_named`, the function the scheduler calls at that site), then -- when the
    seat is not refused -- the REAL adder with the floor this stage applies. `old_gate` = the expression before the L3 fix
    (CAN-FAIL twin). Returns (adder, not_named)."""
    from sglang.srt.managers import scheduler as sch

    req = stage.chunked_req
    adder = _adder()
    lanes_p.configure_adder(stage, adder)
    if old_gate:
        not_named = incoming is not None and incoming.get(req.rid) is None
    else:
        not_named = sch._pp_chunked_continuation_not_named(stage, incoming, req)
    if not not_named:
        with lanes_p.chunk_scope(adder, req):
            stage.chunked_req = adder.add_chunked_req(req)
    return adder, not_named


def _row_authority_incoming(stage):
    """PP0 decides (`None`: its own derivation); under the default row authority every follower holds an effective map for
    every pass -- `{}` when the decision names nothing (scheduler_pp_mixin.py: '{} is the admit nothing spelling')."""
    return None if stage.ps.pp_rank == 0 else {}


class FollowerPathTest(unittest.TestCase):
    def setUp(self):
        from sglang.srt.managers import scheduler as sch

        self.sch = sch
        self.released = []
        self.orig = sch.release_kv_cache
        sch.release_kv_cache = lambda req, tree, is_insert=True: self.released.append((req.rid, is_insert))

    def tearDown(self):
        self.sch.release_kv_cache = self.orig

    def _stamped_stages(self, floor=1):
        """Three stages with a continuation each, the floor stamped and applied by the real functions in one pass."""
        stages = [_park_stage(i) for i in range(3)]
        reqs = [_continuation_on(s) for s in stages]
        lanes_p.on_rpc(stages[0], _rpc(floor, 1))
        stamp = lanes_p.pp0_pass(stages[0])
        wire = pr.stamp_lane_floor([], stamp)
        for s in stages[1:]:
            sent_on = list(wire)
            _rest, st = pr.absorb_lane_floor(wire)
            lanes_p.apply_follower(s, st)
            wire = sent_on
        self.assertEqual({lanes_p.echo(s) for s in stages}, {(floor, 1)})
        return stages, reqs

    def test_every_stage_parks_in_place_under_the_row_authority(self):
        with _LanesOn():
            stages, reqs = self._stamped_stages()
            for s, req in zip(stages, reqs):
                adder, not_named = _stage_pass(s, _row_authority_incoming(s))
                self.assertFalse(not_named, f"rank {s.ps.pp_rank}: the #992 gate must not swallow a lane-held continuation")
                self.assertEqual(adder.can_run_list, [], "no seat: no chunk, nothing in the batch")
                self.assertIs(s.chunked_req, req, "it stays the chunked request (nothing leaks)")
                self.assertEqual(req.extend_range, Range(SPAN, SPAN))
                self.assertTrue(req.weg2_pool_parked and req.weg2_lane_parked, f"rank {s.ps.pp_rank} did not park")

    def test_every_stage_gives_its_rows_back_to_the_tree_with_is_insert(self):
        with _LanesOn():
            stages, reqs = self._stamped_stages()
            for s in stages:
                _stage_pass(s, _row_authority_incoming(s))
            with self.assertLogs("sglang.srt.weg2.lanes_p", level="INFO") as cm:
                for s in stages:
                    self.sch.Scheduler.process_pending_weg2_park(s)
            self.assertEqual(self.released, [("R", True)] * 3, "PP0, PP1 and PP2 each return the rows, inserted into the tree")
            for s, req in zip(stages, reqs):
                self.assertIsNone(s.chunked_req)
                self.assertEqual(s.queued, [("R", True)])
                self.assertEqual(req.weg2_parked_span, SPAN)
                self.assertTrue(req.weg2_lane_parked)
            self.assertEqual(len([l for l in cm.output if "WEG2-PARK (lane) n=1" in l]), 3)

    def test_canfail_the_old_992_gate_keeps_the_follower_rows_locked(self):
        """The expression before the fix: under the row authority a follower's continuation that the decision does not name
        is refused its seat and `add_chunked_req` is never reached -- PP0 parks, PP1/PP2 hold the device rows of the lower
        lane while the hold lasts (the review's finding 1), and the higher lane's PP-room vote carries that smaller room."""
        with _LanesOn():
            stages, reqs = self._stamped_stages()
            for s in stages:
                _adder_, not_named = _stage_pass(s, _row_authority_incoming(s), old_gate=True)
                self.assertEqual(not_named, s.ps.pp_rank != 0)
            for s in stages:
                self.sch.Scheduler.process_pending_weg2_park(s)
            self.assertEqual(self.released, [("R", True)], "only PP0 gave its rows back")
            self.assertIsNone(stages[0].chunked_req)
            for s, req in zip(stages[1:], reqs[1:]):
                self.assertIs(s.chunked_req, req, "PP1/PP2 still hold the continuation")
                self.assertFalse(getattr(req, "weg2_lane_parked", False))

    def test_the_predicate_is_the_old_992_expression_everywhere_the_lane_does_not_hold(self):
        """No lane state / floor 0 / a continuation at or above the floor: #992 behaves exactly as before the lanes."""
        sch = self.sch
        floor_free = _park_stage(1)
        held_free = _continuation_on(floor_free)
        for incoming in (None, {}, {"other": 5}, {"R": 7}):
            old = incoming is not None and incoming.get("R") is None
            self.assertEqual(sch._pp_chunked_continuation_not_named(floor_free, incoming, held_free), old)
        with _LanesOn():
            stages, _reqs = self._stamped_stages(floor=1)
            above = _continuation_on(stages[1], rid="H", priority=1)  # lane 1 at floor 1: not held
            for incoming in (None, {}, {"other": 5}, {"H": 7}):
                old = incoming is not None and incoming.get("H") is None
                self.assertEqual(sch._pp_chunked_continuation_not_named(stages[1], incoming, above), old)
            # held: never refused by #992, whatever the decision says (named: the schedule is executed by add_chunked_req)
            held = _continuation_on(stages[1], rid="L", priority=0)
            for incoming in (None, {}, {"other": 5}, {"L": 7}):
                self.assertFalse(sch._pp_chunked_continuation_not_named(stages[1], incoming, held))

    def test_a_lane_at_the_floor_is_still_refused_its_seat_by_992_when_unnamed(self):
        """The exemption is only for the held lane: a continuation of an admitted lane that the decision does not name keeps
        the #992 refusal (and its log), unchanged."""
        with _LanesOn():
            stages, _reqs = self._stamped_stages(floor=1)
            above = _continuation_on(stages[2], rid="H", priority=1)
            adder, not_named = _stage_pass(stages[2], {})
            self.assertTrue(not_named)
            self.assertIs(stages[2].chunked_req, above)
            self.assertFalse(getattr(above, "weg2_lane_parked", False))
            self.assertEqual(adder.can_run_list, [])

    def test_a_floor_that_falls_voids_the_follower_park_too(self):
        with _LanesOn():
            stages, reqs = self._stamped_stages(floor=1)
            for s in stages:
                _stage_pass(s, _row_authority_incoming(s))
            # LANE-EMPTY: floor 0, epoch 2, stamped and applied by every stage in the next pass before the rows went back
            lanes_p.on_rpc(stages[0], _rpc(0, 2))
            stamp = lanes_p.pp0_pass(stages[0])
            for s in stages[1:]:
                lanes_p.apply_follower(s, stamp)
            for s in stages:
                self.sch.Scheduler.process_pending_weg2_park(s)
            self.assertEqual(self.released, [], "the floor fell first: no rows go back, the continuation resumes")
            for s, req in zip(stages, reqs):
                self.assertIs(s.chunked_req, req)
                self.assertFalse(req.weg2_lane_parked or req.weg2_pool_parked)

    def test_the_scheduler_calls_the_predicate_at_the_992_site(self):
        """The one site the desk cannot drive (the 4000-line admission pass): pinned by text."""
        src = open(self.sch.__file__).read()
        i = src.index("incoming = getattr(self, \"_pp_admission_incoming_effective\", None)")
        blk = src[i:i + 400]
        self.assertIn("not_named = _pp_chunked_continuation_not_named(", blk)
        self.assertIn("self, incoming, self.chunked_req", blk)
        j = src.index("if not_named:", i)
        k = src.index("self.chunked_req = adder.add_chunked_req(self.chunked_req)", j)
        self.assertLess(j, k)  # the refusal branch comes first, the adder is in the else branch the predicate now routes to
        self.assertIn("_992_chunked_not_named", src[j:j + 400])


# ---------------------------------------------------------------------------
# 5. switch off / no RPC: byte-identical
# ---------------------------------------------------------------------------

class SwitchOffTest(unittest.TestCase):
    def test_no_rpc_no_state_no_stamp_and_the_list_is_untouched(self):
        s0 = _stage(0)
        self.assertIsNone(lanes_p.pp0_pass(s0))
        self.assertIsNone(lanes_p.state_of(s0))
        wire = [object(), object()]
        self.assertEqual(pr.stamp_lane_floor(wire, None), wire)
        rest, st = pr.absorb_lane_floor(wire)
        self.assertIs(rest, wire)  # untouched, the same object
        self.assertIsNone(st)
        self.assertFalse(lanes_p.apply_follower(_stage(1), None))
        self.assertEqual(lanes_p.echo(s0), (-1, -1))

    def test_the_switch_off_rpc_is_ignored(self):
        old = os.environ.pop("SGLANG_WEG2_LANES", None)
        try:
            os.environ["SGLANG_WEG2_GROUP"] = "P"
            s0 = _stage(0)
            lanes_p.on_rpc(s0, _rpc(3, 1))
            self.assertIsNone(lanes_p.state_of(s0))
            self.assertIsNone(lanes_p.pp0_pass(s0))
        finally:
            os.environ.pop("SGLANG_WEG2_GROUP", None)
            if old is not None:
                os.environ["SGLANG_WEG2_LANES"] = old

    def test_group_d_and_a_parkless_p_take_no_floor(self):
        with _LanesOn():
            os.environ["SGLANG_WEG2_GROUP"] = "D"
            s = _stage(0)
            lanes_p.on_rpc(s, _rpc(2, 1))
            self.assertIsNone(lanes_p.state_of(s), "group D is part L2's")
            os.environ["SGLANG_WEG2_GROUP"] = "P"
            os.environ["SGLANG_WEG2_PARK"] = "0"
            lanes_p.on_rpc(s, _rpc(2, 1))
            self.assertIsNone(lanes_p.state_of(s), "no park, no way to give rows back: inert, named in the log")

    def test_the_adder_keeps_its_class_defaults_without_state(self):
        adder = _adder()
        lanes_p.configure_adder(_stage(0), adder)
        self.assertEqual((adder.lane_floor, adder.lane_chunk_cap_tokens, adder.lane_higher_waiting), (0, 0, -1))
        self.assertEqual(PrefillAdder.lane_floor, 0)
        # and floor 0 changes nothing: the same chunk as an adder that never heard of lanes
        a, b = _adder(), _adder()
        ra, rb = _req("A", 0, prefix=SPAN), _req("A", 0, prefix=SPAN)
        a.add_chunked_req(ra)
        b.lane_floor = 0
        b.add_chunked_req(rb)
        self.assertEqual(ra.extend_range, rb.extend_range)
        self.assertEqual(ra.extend_range, Range(SPAN, SPAN + CHUNK))

    def test_a_single_rank_boot_applies_the_rpc_at_once(self):
        with _LanesOn():
            s = _stage(0, pp_size=1)
            lanes_p.on_rpc(s, _rpc(2, 1))
            self.assertEqual(lanes_p.floor_for_pass(s), 2)


# ---------------------------------------------------------------------------
# 2. the park
# ---------------------------------------------------------------------------

class AdderHoldTest(unittest.TestCase):
    def _stage_at(self, floor, epoch=1):
        s = _stage(0)
        with_state = lanes_p.state_of(s, create=True)
        with_state.offer(floor, epoch)
        with_state.promote()
        return s

    def test_a_lower_lane_continuation_gets_no_chunk_and_parks_in_place(self):
        with _LanesOn():
            members, req, out = _run_chunk_decision(self._stage_at(1))
            self.assertEqual(members, [])  # not in the batch
            self.assertIs(out, req)  # it stays the chunked request (the #679 shape: nothing leaks)
            self.assertEqual(req.extend_range, Range(SPAN, SPAN))  # no new rows
            self.assertTrue(req.weg2_pool_parked and req.weg2_lane_parked)

    def test_the_lane_at_the_floor_and_above_it_runs_its_chunk(self):
        with _LanesOn():
            for prio in (1, 2):
                members, req, out = _run_chunk_decision(self._stage_at(1), rid=f"L{prio}", priority=prio)
                self.assertEqual(members, [f"L{prio}"])
                self.assertEqual(req.extend_range, Range(SPAN, SPAN + CHUNK))
                self.assertFalse(getattr(req, "weg2_pool_parked", False))

    def test_a_floor_that_fell_before_the_rows_went_back_voids_the_park(self):
        with _LanesOn():
            s = self._stage_at(1)
            _members, req, _ = _run_chunk_decision(s)
            self.assertTrue(req.weg2_lane_parked)
            st = lanes_p.state_of(s)
            st.offer(0, 2)
            st.promote()
            adder = _adder()
            lanes_p.configure_adder(s, adder)
            adder.add_chunked_req(req)
            self.assertFalse(req.weg2_lane_parked)
            self.assertFalse(req.weg2_pool_parked)
            self.assertEqual(req.extend_range, Range(SPAN, SPAN + CHUNK))

    def test_a_follower_runs_what_the_schedule_names_even_when_its_floor_holds(self):
        """#791: membership is PP0's decision. The schedule branch comes BEFORE the lane check (source order), and a
        follower's admission loop never skips by lane."""
        from sglang.srt.managers import schedule_policy as sp

        src = open(sp.__file__).read()
        i = src.index("def add_chunked_req(self, req: Req):")
        body = src[i:i + 6000]
        self.assertLess(body.index("self._add_scheduled_req(req, scheduled, carried_chunk=True)"),
                        body.index("self._weg2_lane_hold_chunked(req)"))
        with _LanesOn():
            self.assertFalse(lanes_p.owns_admission(_stage(1)))
            self.assertFalse(lanes_p.owns_admission(_stage(2)))
            self.assertTrue(lanes_p.owns_admission(_stage(0)))
            self.assertTrue(lanes_p.owns_admission(_stage(0, pp_size=1)))

    def test_the_admission_loop_skips_a_held_request_on_the_deciding_rank_only(self):
        from sglang.srt.managers import scheduler as sch

        src = open(sch.__file__).read()
        # LANES FIX 3: the rank-local form skips too (every rank that decides membership itself), see test_weg2_lanes_fix3_1010
        i = src.index("_lane_floor = _lanes_p.floor_for_pass(self) if _lanes_p.skips_by_lane(self) else 0")
        blk = src[i:i + 900]
        self.assertIn("_lanes_p.holds(_lanes_p.req_lane(req), _lane_floor)", blk)
        self.assertIn('_note_skip("weg2_lane_held", req.rid)', blk)
        self.assertIn("continue", blk)
        # the skip sits before every other loop-head term
        self.assertLess(blk.index("weg2_lane_held"), blk.index("_burst_hold is not None") if "_burst_hold is not None" in blk else 10**9)
        self.assertTrue(lanes_p.holds(0, 1) and not lanes_p.holds(1, 1) and not lanes_p.holds(0, 0))


def _park_stand_in(*, waiting, inflight=0, lane=True, floor=1, epoch=1, pool_idx=3, prio=0):
    calls = []

    class _Req:
        rid = "weg2-1-7"
        priority = prio
        req_pool_idx = pool_idx
        weg2_pool_parked = True
        weg2_lane_parked = lane
        inflight_middle_chunks = inflight
        prefix_indices = list(range(SPAN))
        origin_input_ids = list(range(N_TOK))

        def finished(self):
            return False

        def reset_for_retract(self):
            calls.append("reset")

    class _S:
        chunked_req = _Req()
        tree_cache = object()
        waiting_queue = list(waiting)

        def _add_request_to_queue(self, req, is_retracted=False):
            calls.append(("queue", is_retracted))

    s = _S()
    s.ps = SimpleNamespace(pp_size=3, pp_rank=0)
    st = lanes_p.state_of(s, create=True)
    st.offer(floor, epoch)
    st.promote()
    return s, calls


class HeadOfStepParkTest(unittest.TestCase):
    def setUp(self):
        from sglang.srt.managers import scheduler as sch

        self.sch = sch
        self.seen = []
        self.orig = sch.release_kv_cache
        sch.release_kv_cache = lambda req, tree, is_insert=True: self.seen.append(is_insert)

    def tearDown(self):
        self.sch.release_kv_cache = self.orig

    def test_the_lane_park_gives_the_rows_back_with_is_insert_even_when_nobody_waits(self):
        with _LanesOn():
            s, calls = _park_stand_in(waiting=[])
            req = s.chunked_req
            with self.assertLogs("sglang.srt.weg2.lanes_p", level="INFO") as cm:
                self.sch.Scheduler.process_pending_weg2_park(s)
            self.assertEqual(self.seen, [True])  # inserted, not discarded: the KV stays in the tree
            self.assertIsNone(s.chunked_req)
            self.assertEqual(calls, ["reset", ("queue", True)])
            self.assertEqual(req.weg2_parked_span, SPAN)  # the resume is admitted WHOLE with this span behind it
            self.assertFalse(req.weg2_pool_parked)
            self.assertTrue(req.weg2_lane_parked)  # the loop skips it while the floor stands above its lane
            self.assertEqual(s._weg2_park_lane_n, 1)
            line = [l for l in cm.output if "WEG2-PARK (lane) n=1" in l]
            self.assertTrue(line, cm.output)
            self.assertIn("rid=weg2-1-7", line[0])
            self.assertIn(f"span={SPAN} of {N_TOK}", line[0])
            self.assertIn("lane=0 floor=1 epoch=1", line[0])
            self.assertFalse(hasattr(s, "_weg2_park_n"), "the pressure counter is not the lane counter")

    def test_the_pressure_park_is_unchanged_it_still_needs_a_waiter(self):
        with _LanesOn():
            s, calls = _park_stand_in(waiting=[], lane=False)
            self.sch.Scheduler.process_pending_weg2_park(s)
            self.assertEqual(self.seen, [])  # nobody to make room for: the in-place park stays (#679)
            s, calls = _park_stand_in(waiting=[object()], lane=False)
            self.sch.Scheduler.process_pending_weg2_park(s)
            self.assertEqual(self.seen, [True])
            self.assertEqual(s._weg2_park_n, 1)
            self.assertFalse(hasattr(s, "_weg2_park_lane_n"))

    def test_it_waits_for_the_launched_chunk_to_drain(self):
        with _LanesOn():
            s, calls = _park_stand_in(waiting=[], inflight=1)
            self.sch.Scheduler.process_pending_weg2_park(s)
            self.assertEqual(self.seen, [])
            self.assertIsNotNone(s.chunked_req)

    def test_a_floor_that_fell_voids_the_park_at_the_head_of_the_step(self):
        with _LanesOn():
            s, calls = _park_stand_in(waiting=[object()], floor=1)
            st = lanes_p.state_of(s)
            st.offer(0, 2)
            st.promote()
            req = s.chunked_req
            self.sch.Scheduler.process_pending_weg2_park(s)
            self.assertEqual(self.seen, [])
            self.assertIs(s.chunked_req, req)
            self.assertFalse(req.weg2_lane_parked or req.weg2_pool_parked)


# ---------------------------------------------------------------------------
# 3. the resume
# ---------------------------------------------------------------------------

class ResumeTest(unittest.TestCase):
    def test_the_resumed_request_computes_only_the_rest(self):
        """After the park the request sits at the head of its lane with `weg2_parked_span`; when the floor falls to its lane
        the adder admits it WHOLE (no per-chunk admission, no ping-pong) with the prefix the tree still holds (the match
        gives `prefix_indices` = the span): the extend starts at the span, the anchor is not recomputed."""
        with _LanesOn():
            floor_state = _stage(0)
            st = lanes_p.state_of(floor_state, create=True)
            st.offer(1, 1)
            st.promote()
            # while the floor stands above its lane: held (the loop skips it, `holds`), nothing admitted
            resumed = _req("weg2-1-7", 0, prefix=SPAN)
            resumed.weg2_parked_span = SPAN
            self.assertTrue(lanes_p.holds(lanes_p.req_lane(resumed), lanes_p.floor_for_pass(floor_state)))
            # the floor falls (LANE-EMPTY) -> admitted
            st.offer(0, 2)
            st.promote()
            self.assertFalse(lanes_p.holds(lanes_p.req_lane(resumed), lanes_p.floor_for_pass(floor_state)))
            adder = _adder()
            lanes_p.configure_adder(floor_state, adder)
            res = adder.add_one_req(resumed, truncation_align_size=None)
            self.assertEqual(len(resumed.prefix_indices), SPAN, "the prefix match is kept, nothing is dropped")
            self.assertEqual(resumed.extend_range.start, SPAN, "the extend starts AT the parked span: no recomputation")
            self.assertEqual(resumed.extend_range, Range(SPAN, SPAN + CHUNK))
            self.assertIn(resumed, adder.can_run_list)
            self.assertIsNotNone(res)

    def test_a_parked_request_is_admitted_whole_not_per_chunk(self):
        from sglang.srt.managers import schedule_policy as sp

        src = open(sp.__file__).read()
        i = src.index("if _weg2_chunk_admit() and not getattr(req, \"weg2_parked_span\", 0):")
        self.assertIn("chunk_admit_tokens as _cat", src[i:i + 300])  # a request with a span skips this charge


# ---------------------------------------------------------------------------
# 4. smaller chunks
# ---------------------------------------------------------------------------

class LaneChunkTest(unittest.TestCase):
    def _configured(self, cap, waiting_lane, *, pp_rank=0):
        s = _stage(pp_rank)
        s.pp_flip_counters = object()  # LANES FIX 3: the chunk cap exists only on a form with a row carrier (the schedule carries it)
        st = lanes_p.state_of(s, create=True)
        st.offer(0, 1)
        st.promote()
        s.waiting_queue = [SimpleNamespace(priority=waiting_lane)] if waiting_lane is not None else []
        adder = _adder(page_size=64)
        with _LanesOn() as on:
            if cap is not None:
                on.chunk(cap)
            lanes_p.configure_adder(s, adder)
        return adder

    def test_a_lower_lane_takes_the_small_chunk_while_a_higher_lane_waits(self):
        adder = self._configured(4096, waiting_lane=1)
        low = _req("low", 0)
        with lanes_p.chunk_scope(adder, low):
            adder.add_one_req(low, truncation_align_size=None)
        self.assertEqual(low.extend_range.end - low.extend_range.start, 4096)
        # the chunk budget is charged what was taken, not the cap
        self.assertEqual(adder.rem_chunk_tokens, CHUNK - 4096)

    def test_the_highest_waiting_lane_keeps_the_normal_chunk(self):
        adder = self._configured(4096, waiting_lane=1)
        high = _req("high", 1)
        with lanes_p.chunk_scope(adder, high):
            adder.add_one_req(high, truncation_align_size=None)
        self.assertEqual(high.extend_range.end - high.extend_range.start, CHUNK)

    def test_zero_is_the_normal_chunk(self):
        adder = self._configured(0, waiting_lane=1)
        low = _req("low", 0)
        with lanes_p.chunk_scope(adder, low):
            adder.add_one_req(low, truncation_align_size=None)
        self.assertEqual(low.extend_range.end - low.extend_range.start, CHUNK)
        self.assertEqual(adder.lane_chunk_cap_tokens, 0)

    def test_nobody_higher_waits_the_normal_chunk(self):
        adder = self._configured(4096, waiting_lane=None)
        low = _req("low", 0)
        with lanes_p.chunk_scope(adder, low):
            adder.add_one_req(low, truncation_align_size=None)
        self.assertEqual(low.extend_range.end - low.extend_range.start, CHUNK)

    def test_a_follower_executes_the_scheduled_chunk_and_never_caps_it(self):
        adder = self._configured(4096, waiting_lane=1, pp_rank=1)
        self.assertEqual(adder.lane_chunk_cap_tokens, 0, "only the rank that decides narrows the chunk")

    def test_the_cap_is_whole_pages_and_at_least_one(self):
        self.assertEqual(lanes_p.chunk_cap_tokens(4100, 64), 4096)
        self.assertEqual(lanes_p.chunk_cap_tokens(10, 64), 64)
        self.assertEqual(lanes_p.chunk_cap_tokens(0, 64), 0)

    def test_the_narrowed_chunk_continues_the_same_request_at_the_next_boundary(self):
        """Two passes: the small chunk, then the continuation's next small chunk -- the park window is one SMALL chunk."""
        adder = self._configured(2048, waiting_lane=2)
        low = _req("low", 0)
        with lanes_p.chunk_scope(adder, low):
            adder.add_one_req(low, truncation_align_size=None)
        self.assertEqual(low.extend_range, Range(0, 2048))


class FieldTest(unittest.TestCase):
    def test_the_lane_of_a_request(self):
        self.assertEqual(lanes_p.req_lane(SimpleNamespace(priority=None)), 0)
        self.assertEqual(lanes_p.req_lane(SimpleNamespace(priority=-5)), 0)
        self.assertEqual(lanes_p.req_lane(SimpleNamespace(priority=3)), 3)
        self.assertEqual(lanes_p.req_lane(SimpleNamespace(priority=True)), 0)
        self.assertEqual(lanes_p.req_lane(SimpleNamespace()), 0)
        self.assertEqual(lanes_p.highest_waiting_lane([SimpleNamespace(priority=0), SimpleNamespace(priority=4)]), 4)
        self.assertEqual(lanes_p.highest_waiting_lane([]), -1)


class WiringTest(unittest.TestCase):
    """The sites the desk cannot drive (the PP loop itself): pinned by text, like the park wiring (xsn303)."""

    def test_pp_mixin_stamps_after_the_cap_and_followers_absorb_after_the_cap(self):
        from sglang.srt.managers import scheduler_pp_mixin as m

        src = open(m.__file__).read()
        self.assertLess(src.index("_wire_reqs = _pr.stamp_cap(_wire_reqs, self._weg2_pr_verdict)"),
                        src.index("_wire_reqs = _pr.stamp_lane_floor(_wire_reqs, self._weg2_lane_stamp)"))
        self.assertLess(src.index("recv_reqs, _pr_cap = _pr.absorb_cap(recv_reqs)"),
                        src.index("recv_reqs, _lane_stamp = _pr.absorb_lane_floor(recv_reqs)"))
        # the follower takes the stamp off AFTER the relay: the relay (`_pp_send_pyobj_to_next_stage`) precedes the absorb
        self.assertLess(src.index("self.send_req_work = self._pp_send_pyobj_to_next_stage("),
                        src.index("recv_reqs, _lane_stamp = _pr.absorb_lane_floor(recv_reqs)"))
        # the hook runs right after the room hook, before the forward
        self.assertLess(src.index("self._weg2_pp_room_pass_hook()\n"), src.index("self._weg2_lane_floor_pass_hook()\n"))

    def test_the_rpc_is_wired_end_to_end(self):
        from sglang.srt.managers import io_struct as io
        from sglang.srt.managers import scheduler as sch

        self.assertEqual(lanes.RPC_LANE_FLOOR, "/weg2/lane_floor")
        # http_server imports audio libraries the desk venv lacks: read the file, do not import it
        hs_path = os.path.join(os.path.dirname(io.__file__), "..", "entrypoints", "http_server.py")
        hs_src = open(hs_path).read()
        self.assertIn('@app.api_route("/weg2/lane_floor", methods=["POST"])', hs_src)
        self.assertIn("Weg2LaneFloorReqInput", hs_src)
        req = io.Weg2LaneFloorReqInput(floor=2, epoch=5)
        self.assertEqual((req.floor, req.epoch), (2, 5))
        self.assertIn("(Weg2LaneFloorReqInput, self.handle_weg2_lane_floor)", open(sch.__file__).read())
        self.assertTrue(callable(sch.Scheduler.handle_weg2_lane_floor))
        with _LanesOn():
            s = _stage(0)
            sch.Scheduler.handle_weg2_lane_floor(s, req)
            self.assertEqual(lanes_p.pp0_pass(s).floor, 2)

    def test_the_markers_are_the_names_l1_fixed(self):
        self.assertEqual(lanes.MARK_PR_FLOOR, "PR LANE-FLOOR")
        self.assertEqual(lanes_p.MARK_PARK_LANE, "WEG2-PARK (lane)")


if __name__ == "__main__":
    unittest.main()
