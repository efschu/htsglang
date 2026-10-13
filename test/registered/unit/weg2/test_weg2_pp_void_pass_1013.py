# SPDX-License-Identifier: Apache-2.0
"""PP-VOID-PASS (FIXES-NACH-FREEZE Nr. 45, NF 13.10.2026): on the CARRIERLESS form (NF group P, ``#631 ROW AUTHORITY DISABLED``,
``SGLANG_WEG2_P_ROW_AUTHORITY`` off) the whole PP group discards ONE pass together when a stage cannot pay it.

The deaths (int24f4/f6, 13.10.): 11:59:14 ``PrefillOutOfPages`` (prefill), 18:30:21 / 19:36:02 / 20:11:43 ``AgreedRoomShort``
(load-back) -- PP0 raised, the typed exception propagated (Nr. 43 retreats only where the followers execute PP0's row), RANK-DEATH
-> W17 -> STOP. A PP0 that retreated ALONE would leave PP1/PP2 blocked in the proxy receive for a frame it never sends
(#969Z / #973 RING COMMIT TIMEOUT, #1233 width divergence). The carrier is the pass's own proxy frame: PP0 sends a VOID frame in
the slot of the frame it owes, every stage with a batch for the slot discards it at the receive (before its launch) and owes the
next stage the same statement.

RED on f5e3dc7b16 (Nr. 43): there is no ``pp_void_pass``; the carrierless seat answers ``NOT TAKEN (no_row_carrier)`` and the typed
exception propagates (``test_the_base_behaviour_is_death...`` pins that shape with the switch off). GREEN here: one pass, three
stages, same act, allocations back, next pass runs. Hermetic: stubs only, no CUDA; the event-loop segment is driven through the real
hooks (``after_plan`` / ``at_receive`` / ``send_owed_frame``) and the real ``_pp_recv_proxy_tensors`` / ``_pp_send_dict_to_next_stage``.
"""

import contextlib
import logging
import os
import pathlib
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.mem_cache import common as mc  # noqa: E402
from sglang.srt.weg2 import pp_slot_fidelity as SF  # noqa: E402
from sglang.srt.weg2 import pp_void_pass as VP  # noqa: E402
from sglang.srt.weg2 import prefill_oom_retreat as R  # noqa: E402
from sglang.srt.weg2.intake_origin_abort import IntakeOriginAborts  # noqa: E402
from sglang.srt.weg2.intake_stall import is_intake_stall  # noqa: E402
from test_weg2_prefill_oom_repro_1013 import _Alloc  # noqa: E402
from test_weg2_prefill_oom_retreat_1013 import _Guard, _Req, _fails  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[4]
PP = 3


class Stage:
    """One PP stage of the carrierless NF-P group: its own pool, queue, chunked request, slot ring."""

    def __init__(self, rank, free, relay=None):
        self.rank = rank
        self.free = free
        self.free0 = free
        self.held = {}
        self.launched = []
        self.voids = 0
        tree = types.SimpleNamespace(token_to_kv_pool_allocator=_Alloc(), ongoing_write_through={}, ongoing_load_back={})
        setattr(tree, SF.FLOOR_LOCAL_PP_ATTR, True)
        s = types.SimpleNamespace(
            ps=types.SimpleNamespace(pp_rank=rank, tp_size=1, pp_size=PP),
            pp_flip_counters=None,  # carrierless: no frame probe, followers plan for themselves
            tree_cache=tree, waiting_queue=[], chunked_req=None, anchor_tails=None, running_batch=None,
            max_total_num_tokens=262144, mbs=[None] * PP, forward_ct=121,
            _pp_admission_last_built_decision=types.SimpleNamespace(entries=[1]),
            _pp_load_back_wire={"x": 1}, _pp_admission_guard=_Guard(),
            _weg2_intake_origin_aborts=IntakeOriginAborts(relay=(rank == 0 if relay is None else relay)),
            ipc_channels=types.SimpleNamespace(send_to_tokenizer=types.SimpleNamespace(
                send_output=lambda obj, req: s.sent.append((obj, req)))),
            enable_hicache_storage=False, enable_hierarchical_cache=False, queued=[], sent=[], aborted_now=[],
        )
        s._add_request_to_queue = lambda req, is_retracted=False: (s.queued.append((req, is_retracted)), s.waiting_queue.append(req))
        s._abort_request_now = lambda a: s.aborted_now.append(a)
        s._pp_void_own_batch = self._own_void
        self.sched = s

    # -- the pool
    def alloc(self, rid, n):
        if n > self.free:
            raise mc.PrefillOutOfPages("Prefill out of memory", need=n, free=self.free, evictable_reported=150336,
                                       evictable_paid=320, evict_asked=n - self.free)
        self.free -= n
        self.held[rid] = self.held.get(rid, 0) + n

    def _own_void(self, mb):
        """The established own-void (``_pp_void_own_batch``): members parked unreset, the prepared chunk's rows given back."""
        batch = self.sched.mbs[mb]
        if batch is None:
            return False
        self.sched.mbs[mb] = None
        for r in batch.reqs:
            self.free += self.held.pop(r.rid, 0)
            if r is not self.sched.chunked_req and all(q is not r for q in self.sched.waiting_queue):
                self.sched.waiting_queue.append(r)
        self.voids += 1
        return True

    # -- planning (``get_new_batch_prefill`` + ``prepare_for_extend`` as the scheduler runs them)
    def _admit(self, reqs):
        """The admission loop takes its members out of the waiting queue (a carried continuation is not in it)."""
        self.sched.waiting_queue = [q for q in self.sched.waiting_queue if all(q is not r for r in reqs)]

    def plan(self, mb, reqs, need, carried=None):
        s = self.sched
        VP.snapshot_queue(s)
        self._admit(reqs)
        batch = types.SimpleNamespace(reqs=list(reqs))
        try:
            self.alloc(reqs[0].rid, need)
        except mc.PrefillOutOfPages as exc:
            adder = types.SimpleNamespace(new_chunked_req=None, new_anchor_tails=None)
            if not R.retreat(s, exc, batch, list(reqs), adder, carried_before=carried):
                raise
            s.mbs[mb] = None
            return None
        s.mbs[mb] = batch
        return batch

    def plan_loadback(self, mb, admitted, refused, *, kv_tokens, agreed, avail, evictable=99712, evicted=64, short=True):
        """The admission loop meets ``AgreedRoomShort`` at ``refused`` (this rank's room is below the agreed one); the loop ends
        (``NO_TOKEN``) and the pass is built from what was admitted before. ``short=False``: this rank can pay (a follower)."""
        s = self.sched
        VP.snapshot_queue(s)
        if short:
            exc = SF.AgreedRoomShort("PR AGREED-ROOM SHORT", rid=refused.rid, kv_tokens=kv_tokens, agreed=agreed, avail=avail,
                                     evictable=evictable, evicted=evicted, avail_after=avail + evicted)
            verdict = R.retreat_loadback(s, exc, refused, types.SimpleNamespace(can_run_list=list(admitted), new_anchor_tails=None))
            if verdict is None:
                raise exc
            members = list(admitted)
        else:
            self.alloc(refused.rid, kv_tokens)  # the load-back lands
            members = list(admitted) + [refused]
        self._admit([r for r in members if short is False or r is not refused])
        if members:
            s.mbs[mb] = types.SimpleNamespace(reqs=members)
            for r in admitted:
                self.alloc(r.rid, 4096)
        else:
            s.mbs[mb] = None

    # -- the event-loop segment, through the real hooks (mirrors ``_event_loop_pp_body``)
    def run_slot(self, mb, wire_in, wire_out):
        s = self.sched
        last = self.rank == PP - 1
        VP.after_plan(s, mb)
        cur = s.mbs[mb]
        if self.rank > 0 and (cur or VP.owes_frame(s)):
            def recv():
                assert wire_in, f"PP{self.rank}: HANGING RECV -- the upstream owes a frame in slot {mb} and posted none"
                raw = dict(wire_in.pop(0))
                stmt = VP.pop_from_frame(raw)
                if stmt is not None:
                    VP.note_incoming(s, stmt)
                    return None
                return raw

            cur, _proxy = VP.at_receive(s, mb, cur, recv)
        if cur:
            self.launched.append(mb)
            if not last:
                wire_out.append({"hidden_states": object(), "__real__": True})
        elif not last:
            VP.send_owed_frame(s, mb, lambda frame, _mb: wire_out.append(frame))
        else:
            VP.take_out(s)


class Group:
    def __init__(self, frees=(13568, 60000, 60000)):
        self.st = [Stage(i, f) for i, f in enumerate(frees)]
        self.w01, self.w12 = [], []

    def run(self, mb, order=(0, 1, 2)):
        wires = {0: (None, self.w01), 1: (self.w01, self.w12), 2: (self.w12, [])}
        for i in order:
            self.st[i].run_slot(mb, *wires[i])

    def reqs(self, rid, prefix=98240, total=148965):
        out = []
        for st in self.st:
            r = _Req(rid, prefix=prefix, total=total)
            st.sched.chunked_req = r
            out.append(r)
        return out

    def assert_aligned(self, tc, *, launched):
        tc.assertEqual([st.launched for st in self.st], [launched] * PP, "every rank did the same")
        tc.assertEqual(self.w01, [], "no stray frame on 0->1")
        tc.assertEqual(self.w12, [], "no stray frame on 1->2")


class _Base(unittest.TestCase):
    def setUp(self):
        keys = (R.ENV, R.ENV_RETRIES, R.ENV_HOLD, R.ENV_FAST_ABORT, R.GROUP_ENV, "SGLANG_PP_ROW_AUTHORITY", VP.ENV, "SGLANG_WEG2_PARK")
        self._env = {k: os.environ.get(k) for k in keys}
        for k in keys:
            os.environ.pop(k, None)
        os.environ[R.GROUP_ENV] = "P"
        os.environ["SGLANG_WEG2_PARK"] = "1"
        self._rel = mock.patch("sglang.srt.mem_cache.common.release_kv_cache")
        self._rel.start()

    def tearDown(self):
        self._rel.stop()
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# (A) PP0 cannot pay the prefill chunk: the incident of 11:59:14Z, on the carrierless form
# ---------------------------------------------------------------------------


class PrefillVoidTest(_Base):
    def _chunk7(self, g):
        reqs = g.reqs("weg2-587-1256")
        for st, r in zip(g.st, reqs):
            st.plan(1, [r], 16448, carried=r)
        return reqs

    def test_the_base_behaviour_is_death_when_the_switch_is_off(self):
        """f5e3dc7b16 / switch off: the carrierless seat is NOT TAKEN, the typed exception propagates (RANK-DEATH)."""
        os.environ[VP.ENV] = "0"
        g = Group()
        (r0, *_), st = g.reqs("weg2-587-1256"), g.st[0]
        with self.assertLogs(R.logger, "ERROR") as logs, self.assertRaises(mc.PrefillOutOfPages):
            st.plan(1, [r0], 16448, carried=r0)
        self.assertIn("NOT TAKEN (no_row_carrier)", "\n".join(logs.output))
        self.assertIsNone(getattr(st.sched, VP.OUT_ATTR, None), "nothing is owed to the followers")

    def test_the_whole_group_discards_the_pass_together(self):
        g = Group()
        with self.assertLogs(VP.logger, "WARNING") as logs:
            reqs = self._chunk7(g)
            g.run(1)
        g.assert_aligned(self, launched=[])
        self.assertEqual([st.sched.mbs[1] for st in g.st], [None] * PP)
        # allocations back on every rank: PP1/PP2 had paid the chunk when the void frame arrived
        self.assertEqual([st.free for st in g.st], [st.free0 for st in g.st])
        self.assertEqual([st.voids for st in g.st], [0, 1, 1], "PP0 unwound in the retreat, the followers by the own-void")
        # PP0 parked the continuation in place (Nr. 43), its in-flight count is given back
        self.assertEqual(reqs[0].extend_range, (98240, 98240))
        self.assertEqual(reqs[0].inflight_middle_chunks, 0)
        self.assertFalse(reqs[0].weg2_pool_parked, "no rows-back flag: the head-of-step park would requeue it on PP0 alone")
        self.assertTrue(all(st.sched.chunked_req is r for st, r in zip(g.st, reqs)), "the continuation stays the chunked request everywhere")
        # one marker per rank, the same facts
        lines = [m for m in logs.output if "PP-VOID-PASS rank=" in m]
        self.assertEqual(len(lines), 3, logs.output)
        for rank, line in enumerate(sorted(lines, key=lambda m: m.split("rank=")[1][0])):
            self.assertIn(f"rank={rank} cause=prefill_oom rid=weg2-587-1256 need=16448 free=13568 pass=121 tries=1", line)
        self.assertIn("role=origin", lines[0] if "rank=0" in lines[0] else [x for x in lines if "rank=0" in x][0])

    def test_the_next_pass_runs_on_every_rank(self):
        g = Group()
        self._chunk7(g)
        g.run(1)
        g.st[0].free = 30000  # the write-through finished, the wall moved
        g.st[0].free0 = 30000
        for st, r in zip(g.st, [st.sched.chunked_req for st in g.st]):
            st.plan(2, [r], 16448, carried=r)
        g.run(2)
        g.assert_aligned(self, launched=[2])
        self.assertTrue(all(st.sched.mbs[2] is not None for st in g.st))
        self.assertEqual(g.w01, [], "the real frames were taken")

    def test_the_void_frame_is_a_frame_without_tensors_that_the_receiver_cannot_compute_on(self):
        g = Group()
        self._chunk7(g)
        g.st[0].run_slot(1, None, g.w01)
        self.assertEqual(len(g.w01), 1)
        frame = g.w01[0]
        self.assertEqual(list(frame), [VP.KEY], "one non-tensor entry, no hidden states")
        self.assertEqual(VP.Statement.from_wire(frame[VP.KEY]).rid, "weg2-587-1256")

    def test_n_voids_for_the_same_request_end_in_a_503_for_that_request_only(self):
        g = Group()
        reqs = g.reqs("weg2-587-1256")
        other = [_Req("weg2-9-9", prefix=0, total=5000, inflight=0) for _ in g.st]
        for i in range(R.RETRIES_DEFAULT):
            for st, r in zip(g.st, reqs):
                r.inflight_middle_chunks = 1
                st.plan(1, [r], 16448, carried=r)
            g.run(1)
            g.assert_aligned(self, launched=[])
            aborts = g.st[0].sched._weg2_intake_origin_aborts.take()
            if i < R.RETRIES_DEFAULT - 1:
                self.assertEqual(aborts, [], f"void {i + 1}: no abort yet")
        self.assertEqual([a.rid for a in aborts], ["weg2-587-1256"], "ORIGIN-ABORT on the request wire, for that request")
        self.assertEqual(int(aborts[0].finished_reason["status_code"]), 503)
        self.assertTrue(is_intake_stall(aborts[0].finished_reason["message"]))
        self.assertEqual([st.sched.sent for st in g.st], [[], [], []], "nobody answers the tokenizer itself: the chunked abort does")
        # the followers hold the request in place until the abort lands (nothing requeues or advances it)
        self.assertTrue(all(getattr(r, "_weg2_oom_abort_pending", False) for r in reqs))
        # the abort lands on every stage (same list position): the rid is gone, the server lives and serves the next request
        for st, r in zip(g.st, reqs):
            st.sched.chunked_req = None
            led = getattr(st.sched, "_weg2_prefill_oom_ledger", None)
            if led is not None:
                led.forget("weg2-587-1256")
        for st, o in zip(g.st, other):
            st.plan(1, [o], 5000)
        g.run(1)
        g.assert_aligned(self, launched=[1])

    def test_a_follower_never_decides_the_503(self):
        """All three stages are short (the same wall): the followers unwind their own plan every pass, but only PP0's ledger
        ends in the abort -- a follower's own count never answers the tokenizer or relays anything."""
        g = Group(frees=(13568, 13568, 13568))
        reqs = g.reqs("weg2-587-1256")
        for _ in range(R.RETRIES_DEFAULT + 1):
            for st, r in zip(g.st, reqs):
                r.inflight_middle_chunks = 1
                st.plan(1, [r], 16448, carried=r)
            g.run(1)
            g.st[0].sched._weg2_intake_origin_aborts.take()
        for st in g.st[1:]:
            self.assertEqual(st.sched._weg2_intake_origin_aborts.take(), [])
            self.assertEqual(st.sched.sent, [])
            self.assertEqual(st.sched.aborted_now, [])


# ---------------------------------------------------------------------------
# (B) the load-back refusal: the 18:30:21 / 19:36:02 / 20:11:43 deaths
# ---------------------------------------------------------------------------


class LoadBackVoidTest(_Base):
    KW = dict(kv_tokens=92288, agreed=191744, avail=92032)

    def _pass(self, g, *, pp0_other=False):
        req = [_Req("weg2-302-1956", prefix=0, total=94288, inflight=0) for _ in g.st]
        other = [_Req("weg2-9-9", prefix=0, total=4096, inflight=0) for _ in g.st]
        for st, r in zip(g.st, req):
            st.sched.waiting_queue = [r]
        g.st[0].plan_loadback(1, other[:1] if pp0_other else [], req[0], **self.KW)
        for st, o, r in zip(g.st[1:], other[1:], req[1:]):
            st.plan_loadback(1, [], r, short=False, **self.KW)
        return req, other

    def test_the_whole_pass_is_discarded_not_the_one_request(self):
        """PP1/PP2 plan from their own queue and load the request back; PP0 skipped it. Without the group void they launch a
        pass PP0 never sends a frame for (the 18:30 skew)."""
        g = Group(frees=(92096, 200000, 200000))
        with self.assertLogs(VP.logger, "WARNING") as logs:
            req, _ = self._pass(g)
            g.run(1)
        g.assert_aligned(self, launched=[])
        self.assertEqual([st.free for st in g.st], [st.free0 for st in g.st], "the follower's load-back rows are back")
        self.assertEqual([[q.rid for q in st.sched.waiting_queue] for st in g.st], [["weg2-302-1956"]] * PP,
                         "the request stays queued on every rank")
        lines = [m for m in logs.output if "PP-VOID-PASS rank=" in m]
        self.assertEqual(len(lines), 3)
        self.assertTrue(all("cause=agreed_room_short rid=weg2-302-1956 need=92288 free=92096" in m for m in lines), lines)

    def test_a_pass_pp0_had_built_from_other_members_is_discarded_as_a_whole(self):
        g = Group(frees=(200000, 200000, 200000))
        req, other = self._pass(g, pp0_other=True)
        self.assertIsNotNone(g.st[0].sched.mbs[1], "PP0's loop went on with what it had admitted before the refusal")
        g.run(1)
        g.assert_aligned(self, launched=[])
        self.assertEqual(g.st[0].voids, 1, "PP0's batch goes through the established own-void")
        self.assertEqual([st.free for st in g.st], [st.free0 for st in g.st])

    def test_the_wall_that_stands_ends_in_a_503_for_the_request_and_the_pass_is_still_void_everywhere(self):
        g = Group(frees=(92096, 200000, 200000))
        os.environ[R.ENV_HOLD] = "0"
        for i in range(R.RETRIES_DEFAULT):
            req, _ = self._pass(g)
            g.run(1)
            g.assert_aligned(self, launched=[])
            aborts = g.st[0].sched._weg2_intake_origin_aborts.take()
        self.assertEqual([a.rid for a in aborts], ["weg2-302-1956"])
        self.assertTrue(is_intake_stall(aborts[0].finished_reason["message"]))
        self.assertEqual(g.st[0].sched.waiting_queue, [], "dropped from PP0's queue; the followers drop it at the abort's list position")

    def test_a_refusal_the_group_cannot_void_leaves_no_pending_state(self):
        s = Group().st[0].sched
        exc = SF.AgreedRoomShort("x", rid="r", kv_tokens=10, agreed=20, avail=1, evictable=2, evicted=0, avail_after=1)
        with mock.patch.object(R, "_answer_abort", side_effect=ValueError("boom")), \
                mock.patch.object(R, "verdict", lambda **k: (R.ABORT, "mutant")):
            self.assertIsNone(R.retreat_loadback(s, exc, _Req("r", prefix=0, inflight=0), types.SimpleNamespace(can_run_list=[], new_anchor_tails=None)))
        self.assertIsNone(getattr(s, VP.AFTER_PLAN_ATTR, None))


# ---------------------------------------------------------------------------
# (C) the exception comes up on a FOLLOWER first
# ---------------------------------------------------------------------------


class QueueOrderTest(_Base):
    """A void is "the pass never happened": the waiting queue is what it was before the plan on EVERY rank. The followers' own-void
    appends its members at the tail, PP0's give-back too, a refused load-back stays in place on PP0 only -- unrestored, one waiting
    request more than the pass touched makes the ranks admit in different orders next pass (W27)."""

    def _queues(self, g):
        return [[q.rid for q in st.sched.waiting_queue] for st in g.st]

    def test_a_refused_load_back_leaves_the_queue_as_it_was_on_every_rank(self):
        g = Group(frees=(92096, 400000, 400000))
        for st in g.st:
            st.sched.waiting_queue = [_Req(r, prefix=0, total=94288, inflight=0) for r in ("A", "B", "C")]
        # PP0 stops at A (NO_TOKEN); the followers admit A, B, C (they can pay)
        g.st[0].plan_loadback(1, [], g.st[0].sched.waiting_queue[0], kv_tokens=92288, agreed=191744, avail=92032)
        for st in g.st[1:]:
            q = list(st.sched.waiting_queue)
            VP.snapshot_queue(st.sched)
            st._admit(q)
            for r in q:
                st.alloc(r.rid, 1000)
            st.sched.mbs[1] = types.SimpleNamespace(reqs=q)
        g.run(1)
        g.assert_aligned(self, launched=[])
        self.assertEqual(self._queues(g), [["A", "B", "C"]] * PP)

    def test_fresh_members_given_back_by_pp0_return_to_their_places(self):
        g = Group()
        for st in g.st:
            st.sched.waiting_queue = [_Req(r, prefix=0, total=40000, inflight=0) for r in ("X", "Y", "Z")]
        for st in g.st:
            st.plan(1, [st.sched.waiting_queue[0]], 16448)  # X admitted; PP0 cannot pay
        g.run(1)
        g.assert_aligned(self, launched=[])
        self.assertEqual(self._queues(g), [["X", "Y", "Z"]] * PP)


class FollowerFirstTest(_Base):
    def test_pp1_raises_before_pp0_and_pp0_fails_the_same_pass_consistent(self):
        g = Group(frees=(13568, 13568, 60000))
        reqs = g.reqs("weg2-587-1256")
        with self.assertLogs(VP.logger, "WARNING") as logs:
            g.st[1].plan(1, [reqs[1]], 16448, carried=reqs[1])  # PP1 first, in time
            self.assertTrue(VP.owes_frame(g.st[1].sched), "the follower unwound its own plan and owes itself the frame of the slot")
            g.st[0].plan(1, [reqs[0]], 16448, carried=reqs[0])
            g.st[2].plan(1, [reqs[2]], 16448, carried=reqs[2])
            g.run(1)
        g.assert_aligned(self, launched=[])
        self.assertFalse(any(VP.owes_frame(st.sched) for st in g.st))
        self.assertEqual([st.free for st in g.st], [st.free0 for st in g.st])
        roles = sorted(m.split("role=")[1].split()[0] for m in logs.output if "PP-VOID-PASS rank=" in m)
        self.assertEqual(roles, ["follower-first", "origin", "relay"])

    def test_pp1_load_back_short_first_and_pp0_short_too_is_one_void_pass(self):
        g = Group(frees=(92096, 92096, 400000))
        req = [_Req("weg2-302-1956", prefix=0, total=94288, inflight=0) for _ in g.st]
        for st, r in zip(g.st, req):
            st.sched.waiting_queue = [r]
        kw = dict(kv_tokens=92288, agreed=191744, avail=92032)
        g.st[1].plan_loadback(1, [], req[1], **kw)  # PP1 in time first
        self.assertEqual(g.st[1].sched.mbs[1], None)
        g.st[0].plan_loadback(1, [], req[0], **kw)
        g.st[2].plan_loadback(1, [], req[2], short=False, **kw)
        g.run(1)
        g.assert_aligned(self, launched=[])
        self.assertEqual([st.free for st in g.st], [st.free0 for st in g.st])
        self.assertFalse(any(VP.owes_frame(st.sched) or getattr(st.sched, VP.AFTER_PLAN_ATTR, None) for st in g.st))

    def test_pp1_raises_while_pp0_launched_the_pass_for_real_stops_by_name(self):
        """A stage cannot take a launched pass back from the others: no silent single-rank void, no hang -- the named stop, with
        the terms, and the stage's own plan is unwound before it (nothing leaks)."""
        g = Group(frees=(60000, 13568, 60000))
        reqs = g.reqs("weg2-587-1256")
        g.st[0].plan(1, [reqs[0]], 16448, carried=reqs[0])
        g.st[1].plan(1, [reqs[1]], 16448, carried=reqs[1])
        g.st[2].plan(1, [reqs[2]], 16448, carried=reqs[2])
        g.st[0].run_slot(1, None, g.w01)
        with self.assertRaises(VP.VoidPassLate) as cm:
            g.st[1].run_slot(1, g.w01, g.w12)
        msg = str(cm.exception)
        self.assertIn("rid=weg2-587-1256", msg)
        self.assertIn("need=16448 free=13568", msg)
        self.assertIn("REAL", msg)
        self.assertEqual(g.st[1].free, g.st[1].free0, "the follower's partial plan was unwound")
        self.assertFalse(VP.owes_frame(g.st[1].sched))

    def test_a_follower_that_can_pay_follows_the_void_frame_of_pp0(self):
        g = Group()
        reqs = g.reqs("weg2-587-1256")
        for st, r in zip(g.st, reqs):
            st.plan(1, [r], 16448, carried=r)
        self.assertIsNotNone(g.st[1].sched.mbs[1])
        g.run(1)
        self.assertIsNone(g.st[1].sched.mbs[1])
        self.assertEqual(g.st[1].voids, 1)


# ---------------------------------------------------------------------------
# (D) the real receive / send of the mixin
# ---------------------------------------------------------------------------


class WireTest(_Base):
    def _holder(self, frame):
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin as M

        h = types.SimpleNamespace(
            pp_group=types.SimpleNamespace(is_first_rank=False), _pp_gapped_wire=False, require_attn_tp_allgather=False,
            attn_tp_group=None, mbs=[None] * PP, forward_ct=121,
            ps=types.SimpleNamespace(pp_rank=1, pp_size=PP, tp_size=1),
            _pp_wait_for_proxy_readiness=lambda mb: None, _pp_recv_typed_dict=lambda **kw: dict(frame),
        )
        h._pp_boundary_stats = lambda: None
        h._pp_flip_bump_attempted = lambda chan: None
        h._pp_flip_bump_sent = lambda chan: None
        sent = []
        h.pp_group.send_tensor_dict = lambda tensor_dict, all_gather_group, async_send: sent.append(dict(tensor_dict)) or []
        h._pp_proxy_stamp = lambda mb, result: M._pp_proxy_stamp(h, mb, result)
        return M, h, sent

    def test_the_real_send_puts_the_void_statement_on_the_proxy_wire_with_the_ordinary_stamp(self):
        M, h, sent = self._holder({})
        stmt = VP.Statement("prefill_oom", "weg2-587-1256", 16448, 13568, 121, 1, False, 0)
        M._pp_send_dict_to_next_stage(h, VP.frame_dict(stmt), async_send=True, msg_type="proxy", stamp=h._pp_proxy_stamp(1, None))
        (frame,) = sent
        self.assertEqual(frame["__msg_type__"], "proxy")
        self.assertEqual(frame["__stamp__"][0], 1, "slot")
        self.assertEqual(frame["__stamp__"][2], -1, "no rows")
        self.assertEqual(VP.Statement.from_wire(frame[VP.KEY]), stmt)

    def test_the_real_receive_hands_the_statement_over_and_returns_no_proxy(self):
        stmt = VP.Statement("agreed_room_short", "weg2-302-1956", 92288, 92096, 121, 2, True, 0)
        frame = {"__msg_type__": "proxy", "__stamp__": (1, 7, -1, -1, 121, (-1, -1, -1)), **VP.frame_dict(stmt)}
        M, h, _ = self._holder(frame)
        self.assertIsNone(M._pp_recv_proxy_tensors(h, 1))
        self.assertEqual(VP.take_incoming(h), stmt)
        self.assertIsNone(VP.take_incoming(h), "pass-scoped: taken once")

    def test_a_void_frame_for_another_slot_is_a_slot_disagreement_not_a_void(self):
        stmt = VP.Statement("prefill_oom", "r", 1, 1, 1, 1, False, 0)
        frame = {"__msg_type__": "proxy", "__stamp__": (2, 7, -1, -1, 121, (-1, -1, -1)), **VP.frame_dict(stmt)}
        M, h, _ = self._holder(frame)
        with self.assertRaises(RuntimeError) as cm:
            M._pp_recv_proxy_tensors(h, 1)
        self.assertIn("#1004 SLOT DISAGREEMENT", str(cm.exception))
        self.assertIsNone(VP.take_incoming(h))

    def test_an_unreadable_statement_is_an_error_never_a_frame_to_compute_on(self):
        frame = {"__msg_type__": "proxy", "__stamp__": (1, 7, -1, -1, 121, (-1, -1, -1)), VP.KEY: ("garbage",)}
        M, h, _ = self._holder(frame)
        with self.assertRaises(RuntimeError):
            M._pp_recv_proxy_tensors(h, 1)


# ---------------------------------------------------------------------------
# (E) scope and the event-loop wiring
# ---------------------------------------------------------------------------


class ScopeTest(_Base):
    def _fc(self, **kw):
        s = Group().st[0].sched
        for k, v in kw.items():
            setattr(s, k, v)
        return VP.frame_carrier(s)

    def test_the_frame_form_is_the_carrierless_multi_stage_group_p_only(self):
        self.assertEqual(self._fc(), (True, ""))
        os.environ[VP.ENV] = "0"
        self.assertEqual(self._fc(), (False, "switch_off"))
        os.environ.pop(VP.ENV)
        os.environ[R.GROUP_ENV] = "D"
        self.assertEqual(self._fc(), (False, "not_group_p"))
        os.environ[R.GROUP_ENV] = "P"
        self.assertEqual(self._fc(ps=types.SimpleNamespace(pp_rank=0, tp_size=2, pp_size=3)), (False, "tp_group"))
        self.assertEqual(self._fc(ps=types.SimpleNamespace(pp_rank=0, tp_size=1, pp_size=1)), (False, "single_stage"))
        self.assertEqual(self._fc(_pp_gapped_wire=True), (False, "gapped_wire"))
        self.assertEqual(self._fc(pp_flip_counters=object()), (False, "row_form"), "the row form needs no frame (Nr. 43 as is)")

    def test_the_other_scope_gates_of_the_retreat_still_apply_to_the_frame_form(self):
        g = Group()
        s = g.st[0].sched
        setattr(s.tree_cache, SF.FLOOR_LOCAL_PP_ATTR, False)
        self.assertEqual(R.applies(s), (False, "not_local_pp_floor"))
        setattr(s.tree_cache, SF.FLOOR_LOCAL_PP_ATTR, True)
        s.anchor_tails = [object()]
        self.assertEqual(R.applies(s), (False, "anchor_tails"))

    def test_nothing_changes_without_a_void_the_hooks_are_inert(self):
        g = Group(frees=(60000,) * 3)
        reqs = g.reqs("weg2-587-1256")
        for st, r in zip(g.st, reqs):
            st.plan(1, [r], 16448, carried=r)
        g.run(1)
        g.assert_aligned(self, launched=[1])
        self.assertEqual([st.voids for st in g.st], [0, 0, 0])
        self.assertTrue(all(not hasattr(st.sched, VP.SEEN_ATTR) for st in g.st))


class WiringTest(unittest.TestCase):
    """The mixin carries the hooks in the order the proof needs."""

    @classmethod
    def setUpClass(cls):
        cls.src = (REPO / "python/sglang/srt/managers/scheduler_pp_mixin.py").read_text()
        cls.body = cls.src[cls.src.index("def _event_loop_pp_body(self):"):cls.src.index("def event_loop_pp_disagg_prefill")]

    def test_the_pass_is_discarded_after_the_plan_and_before_the_launch(self):
        b = self.body
        i_plan = b.index("self._pp_void_pass_without_upstream_launch(mb_id)")
        i_after = b.index("_pp_void_pass.after_plan(self, mb_id)")
        i_cur = b.index("cur_batch: Optional[ScheduleBatch] = self.mbs[mb_id]")
        i_recv = b.index("_pp_void_pass.at_receive(")
        i_launch = b.index("self._pp_launch_batch(")
        self.assertLess(i_plan, i_after)
        self.assertLess(i_after, i_cur)
        self.assertLess(i_cur, i_recv)
        self.assertLess(i_recv, i_launch, "the common abort point is the frame receive, strictly before the launch")

    def test_the_void_frame_is_sent_where_the_real_frame_would_be_and_the_last_stage_sends_none(self):
        b = self.body
        i_real = b.index("send_proxy_dict_to_next_stage")
        i_void = b.index("_pp_void_pass.send_owed_frame(")
        i_last = b.index("_pp_void_pass.take_out(self)")
        self.assertLess(i_real, i_void)
        self.assertLess(i_void, i_last)
        self.assertIn("stamp=self._pp_proxy_stamp(_mb, None)", b)

    def test_the_receive_pops_the_void_statement_before_the_admission_row(self):
        r = self.src[self.src.index("def _pp_recv_proxy_tensors("):]
        self.assertLess(r.index("_pp_void_pass.pop_from_frame(raw)"), r.index("_ADMISSION_DECISION_PAYLOAD_KEY"))

    def test_the_unlaunched_slot_is_never_voided_by_the_hook_twice(self):
        self.assertEqual(self.body.count("_pp_void_pass.after_plan("), 1)


# ---------------------------------------------------------------------------
# (F) mutants: each danger direction must turn a test red
# ---------------------------------------------------------------------------


class MutantsTest(unittest.TestCase):
    def _killed(self, case, names, patches):
        self.assertFalse(_fails(case, names, []), "the unmutated tests must pass")
        self.assertTrue(_fails(case, names, patches), "the mutant survived")

    def test_v1_pp0_discards_alone_it_sends_no_frame(self):
        """The skew the carrierless form must not build: PP0 voids, nobody tells PP1/PP2 (they hang in the receive)."""
        self._killed(PrefillVoidTest, ["test_the_whole_group_discards_the_pass_together"],
                     [mock.patch.object(VP, "send_owed_frame", lambda *a, **k: False)])

    def test_v2_a_follower_ignores_the_void_frame_and_keeps_its_rows(self):
        self._killed(PrefillVoidTest, ["test_the_whole_group_discards_the_pass_together"],
                     [mock.patch.object(Stage, "_own_void", lambda self, mb: True)])

    def test_v3_a_follower_does_not_forward_the_statement(self):
        orig = VP.at_receive

        def no_forward(sched, mb_id, cur_batch, recv):
            out = orig(sched, mb_id, cur_batch, recv)
            if sched.ps.pp_rank == 1:
                VP.take_out(sched)  # PP2 is owed a frame and gets none
            return out

        self._killed(PrefillVoidTest, ["test_the_whole_group_discards_the_pass_together"],
                     [mock.patch.object(VP, "at_receive", no_forward)])

    def test_v4_the_load_back_refusal_voids_only_the_request(self):
        self._killed(LoadBackVoidTest, ["test_the_whole_pass_is_discarded_not_the_one_request"],
                     [mock.patch.object(VP, "after_plan", lambda *a, **k: False)])

    def test_v5_a_follower_that_unwound_its_plan_does_not_take_its_frame(self):
        self._killed(FollowerFirstTest, ["test_pp1_raises_before_pp0_and_pp0_fails_the_same_pass_consistent"],
                     [mock.patch.object(VP, "owe_frame", lambda *a, **k: None)])

    def test_v6_a_follower_swallows_the_late_case_and_continues_alone(self):
        orig = VP.at_receive

        def swallow(sched, mb_id, cur_batch, recv):
            try:
                return orig(sched, mb_id, cur_batch, recv)
            except VP.VoidPassLate:
                return None, None

        self._killed(FollowerFirstTest, ["test_pp1_raises_while_pp0_launched_the_pass_for_real_stops_by_name"],
                     [mock.patch.object(VP, "at_receive", swallow)])

    def test_v7_a_follower_decides_the_503(self):
        self._killed(PrefillVoidTest, ["test_a_follower_never_decides_the_503"],
                     [mock.patch.object(R, "_rank", lambda sched: 0)])

    def test_v8_the_abort_does_not_hold_the_request_on_the_followers(self):
        self._killed(PrefillVoidTest, ["test_n_voids_for_the_same_request_end_in_a_503_for_that_request_only"],
                     [mock.patch.object(VP, "_mark_abort_pending", lambda *a, **k: None)])

    def test_v9_the_void_frame_carries_hidden_states_shape_instead_of_the_statement_only(self):
        self._killed(PrefillVoidTest, ["test_the_void_frame_is_a_frame_without_tensors_that_the_receiver_cannot_compute_on"],
                     [mock.patch.object(VP, "frame_dict", lambda stmt: {VP.KEY: stmt.to_wire(), "hidden_states": 1})])

    def test_v11_the_queue_order_is_not_restored(self):
        self._killed(QueueOrderTest, ["test_a_refused_load_back_leaves_the_queue_as_it_was_on_every_rank",
                                      "test_fresh_members_given_back_by_pp0_return_to_their_places"],
                     [mock.patch.object(VP, "restore_queue_order", lambda sched: None)])

    def test_v10_the_switch_is_ignored(self):
        self._killed(PrefillVoidTest, ["test_the_base_behaviour_is_death_when_the_switch_is_off"],
                     [mock.patch.object(VP, "enabled", lambda env=None: True)])


if __name__ == "__main__":
    unittest.main()
