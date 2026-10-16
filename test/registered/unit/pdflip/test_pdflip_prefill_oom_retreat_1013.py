# SPDX-License-Identifier: Apache-2.0
"""PREFILL-OOM-RETREAT (NF 13.10.2026 11:59:14Z, fix 43): a prefill batch the pool cannot pay is taken back, the server lives.

The death (boot ..._b60520f01d_1013_073910, P.log 1214208-1214253): PP0 planned chunk 7 of pdflip-587-1256 (need 16384,
free 13568), the peel paid 320 of the 3200 it asked for (150336 reported evictable, 150016 of them behind the
un-backable leaf [6193], 135872 rows), ``alloc_paged_token_slots_extend`` raised ``RuntimeError`` -> RANK-DEATH -> W17.

RED on 0ea6d17efd (int24f6): (A) ``estimate_payable`` counts [6193] payable (150656 instead of 320: the two device
children the peel demotes first are host-locked, so UD-H cannot clear them and the drop never happens); (B) the raise is a
bare ``RuntimeError`` (no ``PrefillOutOfPages``), the scheduler has no retreat. GREEN: the walk matches the peel's order,
the allocation raises the typed exception, PP0 takes the pass back and the server lives; the wall that stands after N
retreats answers 503 for exactly that request.

Hermetic: stubs only, no CUDA.
"""

import ast
import logging
import os
import pathlib
import textwrap
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.mem_cache import common as mc  # noqa: E402
from test_pdflip_prefill_oom_repro_1013 import _Alloc, _incident  # noqa: E402
from flliper.srt.pdflip import pp_room_vote as PR  # noqa: E402
from flliper.srt.pdflip import pp_slot_fidelity as SF  # noqa: E402
from flliper.srt.pdflip import prefill_oom_retreat as R  # noqa: E402
from flliper.srt.pdflip.intake_origin_abort import IntakeOriginAborts  # noqa: E402
from flliper.srt.pdflip.intake_stall import is_intake_stall, is_too_large  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[4]


# ---------------------------------------------------------------------------
# (C) the retreat
# ---------------------------------------------------------------------------


class _Req:
    def __init__(self, rid, prefix=64, total=10000, inflight=1):
        self.rid = rid
        self.prefix_indices = torch.arange(prefix)
        self.full_untruncated_fill_ids = list(range(total))
        self.inflight_middle_chunks = inflight
        self.extend_range = None
        self.pdflip_pool_parked = False
        self.reset_calls = 0
        self.req_pool_idx = None

    def set_extend_range(self, a, b):
        self.extend_range = (a, b)

    def reset_for_retract(self):
        self.reset_calls += 1


class _Guard:
    def __init__(self):
        self.forgot = []

    def forget_offer(self, rid):
        self.forgot.append(rid)


def _exc(need=16448, free=13568, paid=320, reported=150336):
    return mc.PrefillOutOfPages("Prefill out of memory", need=need, free=free, evictable_reported=reported,
                                evictable_paid=paid, evict_asked=3200)


def _sched(*, pp_rank=0, tp_size=1, floor=True, relay=True, carrier=True, pp_size=3):
    tree = types.SimpleNamespace(token_to_kv_pool_allocator=_Alloc(), ongoing_write_through={}, ongoing_load_back={})
    setattr(tree, SF.FLOOR_LOCAL_PP_ATTR, floor)
    sent = []
    s = types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_rank=pp_rank, tp_size=tp_size, pp_size=pp_size),
        # the followers execute PP0's row (the #631 carrier: the frame probe's side channel); carrier=False is NF group P today
        pp_flip_counters=object() if carrier else None,
        tree_cache=tree, waiting_queue=[], chunked_req=None, anchor_tails=None, running_batch=None,
        max_total_num_tokens=262144,
        _pp_admission_last_built_decision=types.SimpleNamespace(entries=[1]),
        _pp_load_back_wire={"x": 1},
        _pp_admission_guard=_Guard(),
        _pdflip_intake_origin_aborts=IntakeOriginAborts(relay=relay),
        ipc_channels=types.SimpleNamespace(send_to_tokenizer=types.SimpleNamespace(
            send_output=lambda obj, req: sent.append((obj, req)))),
        enable_hicache_storage=False, enable_hierarchical_cache=False,
        queued=[], sent=sent, aborted_now=[],
    )
    s._add_request_to_queue = lambda req, is_retracted=False: s.queued.append((req, is_retracted))
    s._abort_request_now = lambda a: s.aborted_now.append(a)
    return s


def _adder(new_chunked=None):
    return types.SimpleNamespace(new_chunked_req=new_chunked, new_anchor_tails=None)


class RetreatTest(unittest.TestCase):
    def setUp(self):
        self._env = {k: os.environ.get(k) for k in (R.ENV, R.ENV_RETRIES, R.ENV_FAST_ABORT, R.GROUP_ENV, "FLLIPER_PDFLIP_PARK",
                                                    "FLLIPER_PP_ROW_AUTHORITY", "FLLIPER_PDFLIP_PP_VOID_PASS_ENABLE")}
        # Nr. 43 tests: the ROW form and the old refusals of the carrierless form. The void-pass FRAME form (Nr. 45) has its own file.
        os.environ["FLLIPER_PDFLIP_PP_VOID_PASS_ENABLE"] = "0"
        os.environ.pop("FLLIPER_PP_ROW_AUTHORITY", None)
        os.environ.pop(R.ENV_FAST_ABORT, None)
        os.environ[R.GROUP_ENV] = "P"
        os.environ.pop(R.ENV, None)
        os.environ.pop(R.ENV_RETRIES, None)
        os.environ["FLLIPER_PDFLIP_PARK"] = "1"

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # -- the incident: the carried continuation
    def test_the_continuation_is_parked_in_place_and_the_pass_is_voided(self):
        s = _sched()
        req = _Req("pdflip-587-1256")  # remaining 9936 <= free+paid: a retreat the pool can recover from
        s.chunked_req = req
        with self.assertLogs(R.logger, "WARNING") as logs:
            took = R.retreat(s, _exc(), None, [req], _adder())
        self.assertTrue(took)
        self.assertEqual(req.extend_range, (64, 64), "the #679 park form: no chunk this pass")
        self.assertTrue(req.pdflip_pool_parked)
        self.assertEqual(req.inflight_middle_chunks, 0, "counted for a chunk that never launched")
        self.assertIsNone(s._pp_admission_last_built_decision, "nothing of this pass reaches a follower")
        self.assertIsNone(s._pp_load_back_wire)
        self.assertEqual(s._pp_admission_guard.forgot, ["pdflip-587-1256"], "an unsent offer leaves no streak")
        self.assertIs(s.chunked_req, req)
        self.assertEqual(s.queued, [])
        self.assertEqual(s.sent, [])
        text = "\n".join(logs.output)
        self.assertIn("PREFILL-OOM-RETREAT rid=pdflip-587-1256 need=16448 free=13568 evictable_paid=320 retries=1", text)

    def test_the_wall_that_stands_after_n_retreats_answers_503_for_that_request_only(self):
        s = _sched()
        req = _Req("pdflip-587-1256", total=20000)  # remaining need fits free+paid -> only the counter ends it
        s.chunked_req = req
        for i in range(R.RETRIES_DEFAULT - 1):
            req.inflight_middle_chunks = 1
            self.assertTrue(R.retreat(s, _exc(free=20000), None, [req], _adder()))
            self.assertEqual(s._pdflip_intake_origin_aborts.take(), [], f"retry {i + 1}: no abort yet")
        req.inflight_middle_chunks = 1
        self.assertTrue(R.retreat(s, _exc(free=20000), None, [req], _adder()))
        aborts = s._pdflip_intake_origin_aborts.take()
        self.assertEqual([a.rid for a in aborts], ["pdflip-587-1256"])
        reason = aborts[0].finished_reason
        self.assertEqual(int(reason["status_code"]), 503)
        self.assertTrue(is_intake_stall(reason["message"]), "the front requeues on this mark")
        self.assertFalse(is_too_large(reason["message"]), "not the too-large refusal (that one is not requeued)")
        self.assertIn("PREFILL-OOM-RETREAT", reason["message"])
        self.assertEqual(s.sent, [], "the continuation is answered by the chunked abort of every stage, once")
        self.assertNotIn("pdflip-587-1256", s._pdflip_prefill_oom_ledger.counts)

    @mock.patch.dict(os.environ, {R.ENV_FAST_ABORT: "1"})
    def test_an_aborted_continuation_is_not_parked_rows_back_and_is_never_answered_twice(self):
        """The 503 rides the request wire and lands a few passes later. In between the head-of-step park must not requeue
        the request (a W27 split of PP0 against the followers who already dropped it), and a further failed pass does not
        send a second abort."""
        from flliper.srt.managers.scheduler import Scheduler

        s = _sched()
        req = _Req("pdflip-587-1256", prefix=98240, total=148965)
        req.finished = lambda: False
        req.req_pool_idx = 7
        s.chunked_req = req
        self.assertTrue(R.retreat(s, _exc(), None, [req], _adder()))  # impossible -> abort
        self.assertEqual(len(s._pdflip_intake_origin_aborts.take()), 1)
        self.assertFalse(req.pdflip_pool_parked, "no rows-back park for an aborted continuation")
        self.assertTrue(req._pdflip_oom_abort_pending)
        # the head-of-step park, even if add_chunked_req re-set the flag meanwhile, leaves it alone
        req.pdflip_pool_parked = True
        s.waiting_queue = [object()]
        with mock.patch("flliper.srt.managers.scheduler.release_kv_cache") as rel:
            Scheduler.process_pending_pdflip_park(s)
        rel.assert_not_called()
        self.assertIs(s.chunked_req, req)
        # a further failed pass of the same request: taken back, no second abort
        req.inflight_middle_chunks = 1
        self.assertTrue(R.retreat(s, _exc(), None, [req], _adder()))
        self.assertEqual(s._pdflip_intake_origin_aborts.take(), [])
        self.assertEqual(req.inflight_middle_chunks, 0)
        # the abort landing on this rank ends the history
        s._pdflip_prefill_oom_ledger.forget("pdflip-587-1256")
        self.assertNotIn("pdflip-587-1256", s._pdflip_prefill_oom_ledger.abort_pending)

    @mock.patch.dict(os.environ, {R.ENV_FAST_ABORT: "1"})
    def test_an_impossible_remaining_need_ends_at_the_first_retreat_only_with_the_opt_in_switch(self):
        """With FLLIPER_PDFLIP_PREFILL_OOM_FAST_ABORT=1: the wall will not move (nothing in flight, remaining 128901 > free+paid): no N passes into it."""
        s = _sched()
        req = _Req("pdflip-587-1256", prefix=98240, total=148965)
        s.chunked_req = req
        self.assertTrue(R.retreat(s, _exc(), None, [req], _adder()))
        self.assertEqual(len(s._pdflip_intake_origin_aborts.take()), 1)

    def test_without_the_switch_an_impossible_need_waits_for_the_n_retreats_as_ordered(self):
        """Review finding 7: the order says 503 'after N attempts'. The IMPOSSIBLE shortcut is opt-in
        (FLLIPER_PDFLIP_PREFILL_OOM_FAST_ABORT): it reads 'paid' from one peel."""
        self.assertFalse(R.fast_abort_enabled())
        s = _sched()
        req = _Req("pdflip-587-1256", prefix=98240, total=148965)  # remaining 50725 > free+paid: 'impossible' by the H91a verdict
        s.chunked_req = req
        for i in range(R.RETRIES_DEFAULT - 1):
            req.inflight_middle_chunks = 1
            self.assertTrue(R.retreat(s, _exc(), None, [req], _adder()))
            self.assertEqual(s._pdflip_intake_origin_aborts.take(), [], f"retreat {i + 1}: the wall is not yet declared")
        req.inflight_middle_chunks = 1
        self.assertTrue(R.retreat(s, _exc(), None, [req], _adder()))
        self.assertEqual([a.rid for a in s._pdflip_intake_origin_aborts.take()], ["pdflip-587-1256"])

    def test_the_fast_abort_switch_reads_only_an_explicit_on(self):
        self.assertFalse(R.fast_abort_enabled({}))
        self.assertFalse(R.fast_abort_enabled({R.ENV_FAST_ABORT: "0"}))
        self.assertTrue(R.fast_abort_enabled({R.ENV_FAST_ABORT: "1"}))
        self.assertTrue(R.fast_abort_enabled({R.ENV_FAST_ABORT: "true"}))

    def test_requests_given_back_return_to_the_head_in_the_order_the_pass_admitted_them(self):
        """Review finding 6: not to the tail (the N-in-a-row ledger would never see one rid twice, and FCFS would invert),
        and without the queue-limit / prefetch checks of a fresh intake deciding PP0 alone."""
        s = _sched()
        s._add_request_to_queue = lambda req, is_retracted=False: s.waiting_queue.append(req)
        other = _Req("other")
        f1 = _Req("fresh-1", prefix=0, total=3000, inflight=0)
        f2 = _Req("fresh-2", prefix=0, total=3000, inflight=0)
        s.waiting_queue = [other]
        with mock.patch("flliper.srt.mem_cache.common.release_kv_cache"):
            self.assertTrue(R.retreat(s, _exc(), None, [f1, f2], _adder()))
        self.assertEqual([q.rid for q in s.waiting_queue], ["fresh-1", "fresh-2", "other"])

    @mock.patch.dict(os.environ, {R.ENV_FAST_ABORT: "1"})
    def test_a_request_larger_than_the_whole_pool_is_named_too_large_not_requeued(self):
        s = _sched()
        req = _Req("huge", prefix=64, total=300000)
        s.chunked_req = req
        self.assertTrue(R.retreat(s, _exc(), None, [req], _adder()))
        msg = s._pdflip_intake_origin_aborts.take()[0].finished_reason["message"]
        self.assertTrue(is_too_large(msg), "weg2xsn291: the front refuses it by name, no requeue loop")

    def test_something_in_flight_keeps_the_request_waiting(self):
        s = _sched()
        s.tree_cache.ongoing_write_through = {1: 1}
        req = _Req("pdflip-587-1256", prefix=98240, total=148965)
        s.chunked_req = req
        self.assertTrue(R.retreat(s, _exc(), None, [req], _adder()))
        self.assertEqual(s._pdflip_intake_origin_aborts.take(), [])

    @mock.patch.dict(os.environ, {R.ENV_FAST_ABORT: "1"})
    def test_without_a_relay_the_abort_runs_locally(self):
        s = _sched(relay=False)
        req = _Req("pdflip-587-1256", prefix=98240, total=148965)
        s.chunked_req = req
        self.assertTrue(R.retreat(s, _exc(), None, [req], _adder()))
        self.assertEqual([a.rid for a in s.aborted_now], ["pdflip-587-1256"])

    # -- a request the pass admitted fresh
    def test_fresh_requests_go_back_by_the_retract_flow(self):
        s = _sched()
        fresh = _Req("fresh-1", prefix=0, total=5000, inflight=0)
        with mock.patch("flliper.srt.mem_cache.common.release_kv_cache") as rel:
            self.assertTrue(R.retreat(s, _exc(), None, [fresh], _adder()))
        rel.assert_called_once()
        self.assertFalse(rel.call_args.kwargs["is_insert"], "nothing was computed: no insert")
        self.assertEqual(fresh.reset_calls, 1)
        self.assertEqual(s.queued, [(fresh, True)])

    def test_a_fresh_new_chunked_request_is_no_longer_the_chunked_request(self):
        s = _sched()
        fresh = _Req("fresh-2", prefix=0, total=12000, inflight=1)
        s.chunked_req = fresh
        with mock.patch("flliper.srt.mem_cache.common.release_kv_cache"):
            self.assertTrue(R.retreat(s, _exc(), None, [fresh], _adder(new_chunked=fresh)))
        self.assertIsNone(s.chunked_req)
        self.assertEqual(s.queued, [(fresh, True)])

    def test_the_blamed_fresh_request_is_aborted_not_requeued(self):
        os.environ[R.ENV_RETRIES] = "1"
        s = _sched()
        fresh = _Req("fresh-3", prefix=0, total=5000, inflight=0)
        with mock.patch("flliper.srt.mem_cache.common.release_kv_cache") as rel:
            self.assertTrue(R.retreat(s, _exc(free=20000), None, [fresh], _adder()))
        rel.assert_called_once()
        self.assertEqual(s.queued, [])
        self.assertEqual(len(s.sent), 1, "answered 503 to the tokenizer directly (PP0 does not hold it any more)")
        self.assertEqual(int(s.sent[0][0].finished_reason["status_code"]), 503)
        self.assertEqual([a.rid for a in s._pdflip_intake_origin_aborts.take()], ["fresh-3"])

    # -- lanes
    def test_a_lane_held_continuation_is_not_touched_by_a_fresh_retreat(self):
        """Lanes-park x retreat: the lane-held continuation is not in the batch, so its hold, flags and count stay."""
        s = _sched()
        held = _Req("held", prefix=4096, inflight=0)
        held.pdflip_lane_parked = True
        held.pdflip_pool_parked = True
        held.extend_range = (4096, 4096)
        s.chunked_req = held
        fresh = _Req("fresh-4", prefix=0, total=3000, inflight=0)
        with mock.patch("flliper.srt.mem_cache.common.release_kv_cache"):
            self.assertTrue(R.retreat(s, _exc(), None, [fresh], _adder()))
        self.assertIs(s.chunked_req, held)
        self.assertEqual((held.extend_range, held.pdflip_pool_parked, held.pdflip_lane_parked, held.inflight_middle_chunks),
                         ((4096, 4096), True, True, 0))

    def test_the_in_place_park_is_the_state_process_pending_pdflip_park_expects(self):
        """After a retreat the head-of-step park (rows back to the tree) accepts the request like any #679 park."""
        from flliper.srt.managers.scheduler import Scheduler

        s = _sched()
        req = _Req("pdflip-587-1256", prefix=98240, total=100000)  # remaining 1760: recoverable
        req.finished = lambda: False
        req.req_pool_idx = 7
        s.chunked_req = req
        self.assertTrue(R.retreat(s, _exc(), None, [req], _adder()))
        s.waiting_queue = [object()]
        s._pdflip_park_n = 0
        with mock.patch("flliper.srt.managers.scheduler.release_kv_cache") as rel:
            Scheduler.process_pending_pdflip_park(s)
        rel.assert_called_once()
        self.assertIsNone(s.chunked_req)
        self.assertEqual(s.queued, [(req, True)])

    # -- scope: everything else dies as before
    def test_out_of_scope_seats_re_raise(self):
        with mock.patch("flliper.srt.mem_cache.common.release_kv_cache"):
            self._out_of_scope_seats_re_raise()

    def _out_of_scope_seats_re_raise(self):
        exc = _exc()
        req = _Req("r")
        cases = {
            "follower": dict(sched=_sched(pp_rank=1)),
            "tp_group": dict(sched=_sched(tp_size=2)),
            "no_floor": dict(sched=_sched(floor=False)),
            # review blocker 2: carrierless (NF group P, '#631 ROW AUTHORITY DISABLED'): the followers plan the same pass
            # themselves and would block in the proxy receive for a frame a retreating PP0 never sends
            "carrierless": dict(sched=_sched(carrier=False)),
        }
        for name, c in cases.items():
            with self.subTest(name):
                self.assertFalse(R.retreat(c["sched"], exc, None, [req], _adder()))
                self.assertIsNone(req.extend_range)
        with self.subTest("row_authority_killed"):
            os.environ["FLLIPER_PP_ROW_AUTHORITY"] = "0"
            try:
                self.assertFalse(R.retreat(_sched(), exc, None, [req], _adder()))
            finally:
                os.environ.pop("FLLIPER_PP_ROW_AUTHORITY")
        with self.subTest("switch_off"):
            os.environ[R.ENV] = "0"
            self.assertFalse(R.retreat(_sched(), exc, None, [req], _adder()))
            os.environ.pop(R.ENV)
        with self.subTest("group_d"):
            os.environ[R.GROUP_ENV] = "D"
            self.assertFalse(R.retreat(_sched(), exc, None, [req], _adder()))
            os.environ[R.GROUP_ENV] = "P"
        with self.subTest("anchor_tails"):
            s = _sched()
            s.anchor_tails = [object()]
            self.assertFalse(R.retreat(s, exc, None, [req], _adder()))
        with self.subTest("layer_split_dynamic"):
            with mock.patch("flliper.srt.pdflip.p_layer_split_runtime.active", return_value=object()):
                self.assertFalse(R.retreat(_sched(), exc, None, [req], _adder()))

    # -- review blocker 2: the form decides (the followers execute PP0's row, or PP0 retreats nowhere)
    def test_followers_execute_row_reads_the_carrier_not_a_guess(self):
        from flliper.srt.pdflip import p_row_authority as PRA

        self.assertTrue(R.followers_execute_row(_sched()), "legacy carrier: the pp_flip_counters side channel")
        self.assertFalse(R.followers_execute_row(_sched(carrier=False)), "NF group P today: followers plan rank-locally")
        row_only = _sched(carrier=False)
        setattr(row_only, PRA.ROW_ONLY_ATTR, True)
        self.assertTrue(R.followers_execute_row(row_only), "the re-armed P row form (frame probe + chain receiver built)")
        self.assertTrue(R.followers_execute_row(_sched(carrier=False, pp_size=1)), "nobody behind PP0")
        self.assertFalse(R.followers_execute_row(types.SimpleNamespace(ps=types.SimpleNamespace(pp_size="?"))),
                         "unreadable form: never 'PP0 may decide alone'")

    def test_carrierless_the_typed_exception_propagates_and_nothing_of_the_pass_is_touched(self):
        """The skew built into a PP0-only retreat: PP1/PP2 plan the same chunk from their own pool and block for a frame
        PP0 never sends. On that form the retreat is refused, the exception propagates, the pass state stays untouched."""
        s = _sched(carrier=False)
        req = _Req("pdflip-587-1256")
        s.chunked_req = req
        with self.assertLogs(R.logger, "ERROR") as logs:
            took = R.retreat(s, _exc(), None, [req], _adder(), carried_before=req)
        self.assertFalse(took)
        self.assertIn("NOT TAKEN (no_row_carrier)", "\n".join(logs.output))
        self.assertIsNone(req.extend_range)
        self.assertEqual(req.inflight_middle_chunks, 1)
        self.assertIsNotNone(s._pp_admission_last_built_decision, "an unretreated pass keeps its decision")
        self.assertEqual((s.queued, s.sent), ([], []))

    # -- review finding 3: the LAST chunk of a continuation
    def test_the_last_chunk_of_a_continuation_stays_the_continuation(self):
        """``add_chunked_req`` returns None for the final chunk: the attribute is None, the request is in can_run_list
        with its rows. It must be parked in place and kept as the chunked request -- not torn down by the retract flow
        while the followers keep it -- and no middle-chunk count is given back (none was added for this pass)."""
        s = _sched()
        req = _Req("pdflip-587-1256", prefix=98240, total=100000, inflight=2)  # 2 earlier middle chunks still in flight
        req.finished = lambda: False
        req.req_pool_idx = 7
        s.chunked_req = None  # what add_chunked_req left
        with mock.patch("flliper.srt.mem_cache.common.release_kv_cache") as rel:
            took = R.retreat(s, _exc(), None, [req], _adder(), carried_before=req)
        self.assertTrue(took)
        rel.assert_not_called()  # not the fresh give-back
        self.assertEqual(req.reset_calls, 0)
        self.assertEqual(s.queued, [], "it never left its place")
        self.assertIs(s.chunked_req, req, "still the continuation")
        self.assertEqual(req.extend_range, (98240, 98240))
        self.assertTrue(req.pdflip_pool_parked)
        self.assertEqual(req.inflight_middle_chunks, 2, "no count was added for the final chunk, none is given back")
        self.assertIsNone(s._pp_admission_last_built_decision)

    def test_a_truncated_continuation_gives_its_count_back_and_is_still_carried_with_carried_before(self):
        s = _sched()
        req = _Req("pdflip-587-1256", prefix=64, total=10000, inflight=1)
        s.chunked_req = req  # truncated chunk: still the chunked request, +1 added this pass
        self.assertTrue(R.retreat(s, _exc(), None, [req], _adder(), carried_before=req))
        self.assertEqual(req.inflight_middle_chunks, 0)
        self.assertIs(s.chunked_req, req)

    @mock.patch.dict(os.environ, {R.ENV_FAST_ABORT: "1"})
    def test_the_last_chunk_of_a_continuation_that_stands_at_the_wall_answers_503_through_the_chunked_abort(self):
        s = _sched()
        req = _Req("pdflip-587-1256", prefix=98240, total=148965, inflight=2)
        req.finished = lambda: False
        req.req_pool_idx = 7
        s.chunked_req = None
        with mock.patch("flliper.srt.mem_cache.common.release_kv_cache") as rel:
            self.assertTrue(R.retreat(s, _exc(), None, [req], _adder(), carried_before=req))  # impossible -> abort
        rel.assert_not_called()
        self.assertEqual([a.rid for a in s._pdflip_intake_origin_aborts.take()], ["pdflip-587-1256"])
        self.assertIs(s.chunked_req, req, "the abort machinery finds it as the chunked request on this rank too")
        self.assertTrue(req._pdflip_oom_abort_pending)
        self.assertFalse(req.pdflip_pool_parked)
        self.assertEqual(req.inflight_middle_chunks, 2)
        self.assertEqual(s.sent, [])
        self.assertEqual(s.queued, [])

    @mock.patch.dict(os.environ, {R.ENV_FAST_ABORT: "1"})
    def test_a_pending_abort_on_a_last_chunk_is_taken_back_without_a_second_answer(self):
        s = _sched()
        req = _Req("pdflip-587-1256", prefix=98240, total=148965, inflight=2)
        req.finished = lambda: False
        s.chunked_req = None
        with mock.patch("flliper.srt.mem_cache.common.release_kv_cache"):
            self.assertTrue(R.retreat(s, _exc(), None, [req], _adder(), carried_before=req))
        s._pdflip_intake_origin_aborts.take()
        s.chunked_req = None  # the next failed pass: the adder cleared it again
        with mock.patch("flliper.srt.mem_cache.common.release_kv_cache") as rel:
            self.assertTrue(R.retreat(s, _exc(), None, [req], _adder(), carried_before=req))
        rel.assert_not_called()
        self.assertEqual(s._pdflip_intake_origin_aborts.take(), [])
        self.assertIs(s.chunked_req, req)
        self.assertEqual(req.inflight_middle_chunks, 2)

    def test_a_fresh_request_is_not_taken_for_the_continuation(self):
        """carried_before names ANOTHER request (a lane-held continuation outside this batch): the fresh one is given back."""
        s = _sched()
        held = _Req("held", prefix=4096, inflight=0)
        fresh = _Req("fresh-6", prefix=0, total=3000, inflight=0)
        s.chunked_req = held
        with mock.patch("flliper.srt.mem_cache.common.release_kv_cache") as rel:
            self.assertTrue(R.retreat(s, _exc(), None, [fresh], _adder(), carried_before=held))
        rel.assert_called_once()
        self.assertEqual(s.queued, [(fresh, True)])
        self.assertIs(s.chunked_req, held)

    def test_a_retreat_that_cannot_complete_re_raises_instead_of_leaving_a_half_state(self):
        s = _sched()
        fresh = _Req("fresh-5", prefix=0, total=3000, inflight=0)
        with mock.patch("flliper.srt.mem_cache.common.release_kv_cache", side_effect=ValueError("boom")):
            self.assertFalse(R.retreat(s, _exc(), None, [fresh], _adder()))

    def test_a_rid_that_gets_through_has_no_history(self):
        led = R.RetreatLedger()
        led.note("a")
        led.note("a")
        led.clear_batch([types.SimpleNamespace(rid="a")])
        self.assertEqual(led.counts, {})
        led.note("abc-1")
        led.forget("abc")
        self.assertEqual(led.counts, {})


# ---------------------------------------------------------------------------
# the load-back entrance (NF 13.10.2026 18:30:21Z, int24f6: PR AGREED-ROOM SHORT rid=pdflip-302-1956 kv_tokens=92288
# agreed=191744 avail=92032 evictable=99712 evicted=64 avail_after=92096)
# ---------------------------------------------------------------------------


def _short(**kw):
    from flliper.srt.pdflip.pp_slot_fidelity import AgreedRoomShort

    d = dict(rid="pdflip-302-1956", kv_tokens=92288, agreed=191744, avail=92032, evictable=99712, evicted=64,
             avail_after=92096)
    d.update(kw)
    return AgreedRoomShort("PR AGREED-ROOM SHORT rid=%(rid)s kv_tokens=%(kv_tokens)d agreed=%(agreed)d" % d, **d)


class LoadBackRetreatTest(unittest.TestCase):
    def setUp(self):
        self._env = {k: os.environ.get(k) for k in (R.ENV, R.ENV_RETRIES, R.ENV_HOLD, R.ENV_FAST_ABORT, R.GROUP_ENV,
                                                    "FLLIPER_PP_ROW_AUTHORITY", "FLLIPER_PDFLIP_PP_VOID_PASS_ENABLE")}
        os.environ["FLLIPER_PDFLIP_PP_VOID_PASS_ENABLE"] = "0"  # Nr. 43 form; the void-pass FRAME form (Nr. 45) has its own file
        os.environ[R.GROUP_ENV] = "P"
        for k in (R.ENV, R.ENV_RETRIES, R.ENV_HOLD, R.ENV_FAST_ABORT, "FLLIPER_PP_ROW_AUTHORITY"):
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _incident(self, **kw):
        s = _sched(**kw)
        req = _Req("pdflip-302-1956", prefix=0, total=92288 + 2000, inflight=0)
        s.waiting_queue = [req]
        s.tree_cache.ongoing_write_through = {7: 1}  # the publish sweep issued write-throughs in that same second
        return s, req

    def test_the_request_is_skipped_for_the_pass_and_stays_queued(self):
        s, req = self._incident()
        with self.assertLogs(R.logger, "WARNING") as logs:
            v = R.retreat_loadback(s, _short(), req, _adder())
        self.assertEqual(v, R.RETRY)
        self.assertEqual(s.waiting_queue, [req])
        self.assertEqual((s.sent, s.queued, s._pdflip_intake_origin_aborts.take()), ([], [], []))
        text = "\n".join(logs.output)
        self.assertIn("PREFILL-OOM-RETREAT rid=pdflip-302-1956 need=92288 free=92096 evictable_paid=64 retries=1", text)
        self.assertIn("where=loadback", text)

    def test_the_stages_leave_the_load_back_together_the_row_names_nothing_of_it(self):
        """The skipped request is not in can_run_list, so the decision PP0 builds after the loop (the row the followers
        execute) cannot name it: no stage loads it back this pass, none refuses alone (#1004)."""
        s, req = self._incident()
        adder = _adder()
        adder.can_run_list = []
        self.assertEqual(R.retreat_loadback(s, _short(), req, adder), R.RETRY)
        self.assertNotIn(req, adder.can_run_list)
        self.assertIs(s.waiting_queue[0], req)

    def test_the_wall_that_stands_after_n_passes_answers_503_for_that_request_only(self):
        s, req = self._incident()
        s.tree_cache.ongoing_write_through = {}
        s.running_batch = types.SimpleNamespace(reqs=[object()])  # something runs: waiting may free it, not 'impossible'
        # a long running write-through excuses nothing once nothing is in flight: N in a row end it
        for _ in range(R.RETRIES_DEFAULT - 1):
            self.assertEqual(R.retreat_loadback(s, _short(), req, _adder()), R.RETRY)
        self.assertEqual(R.retreat_loadback(s, _short(), req, _adder()), R.ABORT)
        self.assertEqual(s.waiting_queue, [], "dropped from PP0's queue")
        aborts = s._pdflip_intake_origin_aborts.take()
        self.assertEqual([a.rid for a in aborts], ["pdflip-302-1956"], "relayed: every stage drops it in the same pass")
        msg = aborts[0].finished_reason["message"]
        self.assertTrue(is_intake_stall(msg))
        self.assertIn("PREFILL-OOM-RETREAT", msg)
        self.assertEqual(len(s.sent), 1, "the tokenizer is answered once, 503")
        self.assertEqual(int(s.sent[0][0].finished_reason["status_code"]), 503)
        self.assertNotIn("pdflip-302-1956", s._pdflip_prefill_oom_ledger.counts)

    def test_without_the_switch_an_impossible_load_back_waits_for_the_n_passes(self):
        """192 rows short at the first visit (the incident) is not 'impossible' on one peel's word: N passes first."""
        s, req = self._incident()
        s.tree_cache.ongoing_write_through = {}
        for _ in range(R.RETRIES_DEFAULT - 1):
            self.assertEqual(R.retreat_loadback(s, _short(), req, _adder()), R.RETRY)
        self.assertEqual(R.retreat_loadback(s, _short(), req, _adder()), R.ABORT)

    @mock.patch.dict(os.environ, {R.ENV_FAST_ABORT: "1"})
    def test_an_impossible_need_with_nothing_in_flight_ends_at_the_first_pass_only_with_the_opt_in_switch(self):
        s, req = self._incident()
        s.tree_cache.ongoing_write_through = {}
        self.assertEqual(R.retreat_loadback(s, _short(), req, _adder()), R.ABORT)

    def test_a_write_through_in_flight_holds_the_503_back_until_the_hold_has_passed(self):
        """The transient form: the leaf is freed by the write-through within ms -- shorter than N back-to-back passes."""
        s, req = self._incident()
        clock = [1000.0]
        with mock.patch.object(R.time, "monotonic", lambda: clock[0]):
            for _ in range(R.RETRIES_DEFAULT + 3):
                self.assertEqual(R.retreat_loadback(s, _short(), req, _adder()), R.RETRY)
                clock[0] += 0.1
            clock[0] += R.HOLD_DEFAULT_S
            self.assertEqual(R.retreat_loadback(s, _short(), req, _adder()), R.ABORT)

    def test_out_of_scope_seats_get_none_and_nothing_is_touched(self):
        for name, s in {
            "follower": self._incident(pp_rank=1)[0],
            "tp_group": self._incident(tp_size=2)[0],
            "no_floor": self._incident(floor=False)[0],
            "carrierless": self._incident(carrier=False)[0],
        }.items():
            with self.subTest(name):
                req = s.waiting_queue[0]
                self.assertIsNone(R.retreat_loadback(s, _short(), req, _adder()))
                self.assertEqual(s.waiting_queue, [req])
                self.assertEqual((s.sent, s._pdflip_intake_origin_aborts.take()), ([], []))
        with self.subTest("switch_off"):
            os.environ[R.ENV] = "0"
            s, req = self._incident()
            self.assertIsNone(R.retreat_loadback(s, _short(), req, _adder()))
        with self.subTest("group_d"):
            os.environ.pop(R.ENV, None)
            os.environ[R.GROUP_ENV] = "D"
            s, req = self._incident()
            self.assertIsNone(R.retreat_loadback(s, _short(), req, _adder()))

    @mock.patch.dict(os.environ, {R.ENV_FAST_ABORT: "1"})
    def test_a_failing_retreat_re_raises_instead_of_leaving_a_half_state(self):
        s, req = self._incident()
        s.tree_cache.ongoing_write_through = {}
        s.ipc_channels.send_to_tokenizer.send_output = mock.Mock(side_effect=ValueError("boom"))
        self.assertIsNone(R.retreat_loadback(s, _short(), req, _adder()))

    def test_a_request_that_gets_through_has_no_history(self):
        s, req = self._incident()
        self.assertEqual(R.retreat_loadback(s, _short(), req, _adder()), R.RETRY)
        s._pdflip_prefill_oom_ledger.clear_batch([req])  # what _get_new_batch_prefill_raw does after prepare_for_extend
        self.assertEqual(s._pdflip_prefill_oom_ledger.counts, {})

    def test_old_retreats_are_not_in_a_row(self):
        led = R.RetreatLedger()
        clock = [0.0]
        with mock.patch.object(R.time, "monotonic", lambda: clock[0]):
            led.note("a")
            led.note("a")
            clock[0] += R.CONSECUTIVE_S + 1
            self.assertEqual(led.note("a"), 1)

    def test_scheduler_glue_catches_the_short_room_and_re_raises_when_not_taken(self):
        src = (REPO / "python/flliper/srt/managers/scheduler.py").read_text()
        fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "_get_new_batch_prefill_raw")
        handler = None
        for n in ast.walk(fn):
            if isinstance(n, ast.ExceptHandler) and getattr(n.type, "id", None) == "AgreedRoomShort":
                handler = n
        self.assertIsNotNone(handler, "add_one_req must sit inside a try that catches AgreedRoomShort")
        calls = [ast.unparse(x) for x in ast.walk(handler) if isinstance(x, ast.Call)]
        self.assertTrue(any("_prefill_oom_retreat.retreat_loadback" in c for c in calls))
        self.assertTrue(any(isinstance(x, ast.Raise) and x.exc is None for x in ast.walk(handler)))
        self.assertIn("AddReqResult.NO_TOKEN", ast.unparse(handler))
        self.assertIn("AddReqResult.OTHER", ast.unparse(handler), "an aborted request must not reach the NO_TOKEN stall watch too")

    def test_load_back_unwinds_its_locks_before_the_exception_leaves(self):
        src = (REPO / "python/flliper/srt/mem_cache/unified_radix_cache.py").read_text()
        fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "load_back")
        handlers = [n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)
                    and "AgreedRoomShort" in ast.unparse(n.type or ast.Constant(None))]
        self.assertEqual(len(handlers), 1)
        body = ast.unparse(handlers[0])
        self.assertIn("dec_lock_ref", body)
        self.assertIn("dec_host_lock_ref", body)
        self.assertTrue(any(isinstance(x, ast.Raise) and x.exc is None for x in ast.walk(handlers[0])))


class GuardTest(unittest.TestCase):
    def test_retreats_do_not_escalate_the_request(self):
        """3 unsent offers of the same told must not exceed UNRESOLVED_DEFER_CAP and mark the premise dead."""
        from flliper.srt.managers.pp_admission_congruence import PPAdmissionCongruenceGuard

        g = PPAdmissionCongruenceGuard()
        for _ in range(10):
            g.note_offer("r", 98240)
            g.forget_offer("r")
        self.assertFalse(g.is_escalated("r"))
        g2 = PPAdmissionCongruenceGuard()
        for _ in range(10):
            g2.note_offer("r", 98240)
        self.assertTrue(g2.is_escalated("r") or g2._offer_streak["r"][1] > 3, "the control: sent offers still count")


# ---------------------------------------------------------------------------
# glue: the scheduler's call site and the 503 carried through the chunked abort
# ---------------------------------------------------------------------------


class SchedulerGlueTest(unittest.TestCase):
    def test_prepare_for_extend_is_guarded_by_the_typed_retreat(self):
        src = (REPO / "python/flliper/srt/managers/scheduler.py").read_text()
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_get_new_batch_prefill_raw")
        guarded = None
        for n in ast.walk(fn):
            if isinstance(n, ast.Try) and any(
                isinstance(c, ast.Expr) and isinstance(c.value, ast.Call)
                and getattr(c.value.func, "attr", "") == "prepare_for_extend" for c in n.body
            ):
                guarded = n
        self.assertIsNotNone(guarded, "prepare_for_extend() must sit inside a try")
        handler = guarded.handlers[0]
        self.assertEqual(getattr(handler.type, "id", None), "PrefillOutOfPages")
        calls = [ast.unparse(x) for x in ast.walk(handler) if isinstance(x, ast.Call)]
        self.assertTrue(any("_prefill_oom_retreat.retreat" in c for c in calls))
        # review finding 3: the continuation the pass STARTED with is handed over (add_chunked_req clears the attribute
        # for the last chunk of a continuation)
        self.assertTrue(any("carried_before=_oom_carried_before" in c for c in calls))
        snaps = [ast.unparse(n) for n in ast.walk(fn) if isinstance(n, ast.Assign)
                 and any(getattr(t, "id", "") == "_oom_carried_before" for t in n.targets)]
        self.assertEqual(snaps, ["_oom_carried_before = self.chunked_req"])
        # a failed retreat re-raises; a taken one returns the empty pass
        self.assertTrue(any(isinstance(x, ast.Raise) and x.exc is None for x in ast.walk(handler)))
        self.assertTrue(any(isinstance(x, ast.Return) for x in ast.walk(handler)))

    def test_the_abort_wire_records_the_reason_where_the_chunked_abort_reads_it(self):
        src = (REPO / "python/flliper/srt/managers/scheduler.py").read_text()
        fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "_abort_request_now")
        assigns = [ast.unparse(n) for n in ast.walk(fn) if isinstance(n, ast.Assign)]
        self.assertTrue(any(a.startswith("self._pending_chunked_abort_reason = ") and "finished_reason" in a
                            for a in assigns))
        self.assertTrue(any("_pdflip_prefill_oom_ledger" in a for a in assigns), "an abort ends the rid's retreat history")

    def test_the_chunked_abort_carries_the_503_reason_to_the_tokenizer(self):
        from flliper.srt.managers.io_struct import AbortReq
        from flliper.srt.managers.scheduler import Scheduler

        req = types.SimpleNamespace(rid="pdflip-587-1256", time_stats=mock.MagicMock(), req_pool_idx=3,
                                    kv_committed_freed=False, to_finish=None)
        sent = []
        reason = {"type": "abort", "status_code": 503, "message": "PDFLIP-INTAKE-STALL ..."}
        s = mock.MagicMock()
        s._pending_chunked_abort_req = req
        s.chunked_req = req
        s._pending_chunked_abort_reason = reason
        s._pending_chunked_abort_delay = 0
        s._pp_scheduled_extents.return_value = None
        s.disaggregation_mode = "null"
        s.enable_hicache_storage = False
        s.tree_cache.supports_mamba.return_value = False
        s.ipc_channels.send_to_tokenizer.send_output = lambda obj, r: sent.append(obj)
        with mock.patch("flliper.srt.managers.scheduler.prepare_abort"), \
                mock.patch("flliper.srt.managers.scheduler.release_kv_cache"), \
                mock.patch("flliper.srt.pdflip.pp_abort.follower_row_verdict", return_value=None), \
                mock.patch("flliper.srt.pdflip.pp_abort.countdown_step", return_value=(True, 0)):
            Scheduler.process_pending_chunked_abort(s)
        self.assertEqual(len(sent), 1)
        self.assertIsInstance(sent[0], AbortReq)
        self.assertEqual(sent[0].finished_reason, reason)
        # a plain abort answers as before
        sent.clear()
        s._pending_chunked_abort_req = req
        s.chunked_req = req
        s._pending_chunked_abort_reason = None
        with mock.patch("flliper.srt.managers.scheduler.prepare_abort"), \
                mock.patch("flliper.srt.managers.scheduler.release_kv_cache"), \
                mock.patch("flliper.srt.pdflip.pp_abort.follower_row_verdict", return_value=None), \
                mock.patch("flliper.srt.pdflip.pp_abort.countdown_step", return_value=(True, 0)):
            Scheduler.process_pending_chunked_abort(s)
        self.assertIsNone(sent[0].finished_reason)


# ---------------------------------------------------------------------------
# (b) PP0's own walk caps the pass when no follower fact is fresh
# ---------------------------------------------------------------------------


class OwnWalkTest(unittest.TestCase):
    def _pp0(self, root, carrier=True):
        tree = types.SimpleNamespace(
            root_node=root, ongoing_write_through={},
            cache_controller=types.SimpleNamespace(write_policy="write_back", mem_pool_host=None),
            uniform_avail_floor=None, evictable_size=lambda: 150336,
        )
        tree.token_to_kv_pool_allocator = types.SimpleNamespace(available_size=lambda: 13568)
        setattr(tree, SF.FLOOR_LOCAL_PP_ATTR, True)
        s = types.SimpleNamespace(
            ps=types.SimpleNamespace(pp_size=3, pp_rank=0, attn_tp_rank=0, attn_cp_rank=0),
            pp_flip_counters=object() if carrier else None,
            pp_group=types.SimpleNamespace(is_first_rank=True), tree_cache=tree,
            token_to_kv_pool_allocator=tree.token_to_kv_pool_allocator, forward_ct=5, mbs=[],
        )
        s._pdflip_vote_is_wire_rank = lambda: True
        s._pdflip_pr_channel = types.SimpleNamespace(harvest=lambda: [])
        return s, tree

    def test_pp0_without_a_follower_fact_caps_on_its_own_walk(self):
        from flliper.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        root, _, _ = _incident(child_host_lock=1)
        s, tree = self._pp0(root)
        os.environ.pop("FLLIPER_PDFLIP_PP_ROOM_OWN_WALK", None)
        SchedulerPPMixin._pdflip_pp_room_pass_hook(s)
        self.assertEqual(getattr(tree, mc.PP_ROOM_CAP_ATTR), 13568 + 320, "room = free + what the walk can pay")
        self.assertIsNone(s._pdflip_pr_verdict, "own-only: never stamped on list m")
        self.assertLess(mc.fundable_extend_tokens(tree), 16384, "PP0 no longer funds the 16384 chunk")

    def test_carrierless_pp0_publishes_no_cap_of_its_own(self):
        """Review blocker 1: PP1/PP2 plan their chunk width rank-locally on the carrierless form and see no cap; a PP0-only
        cap is a W27/#1233 width divergence (13568 + 320 against 16384) built into the default path."""
        from flliper.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        root, _, _ = _incident(child_host_lock=1)
        s, tree = self._pp0(root, carrier=False)
        os.environ.pop("FLLIPER_PDFLIP_PP_ROOM_OWN_WALK", None)
        SchedulerPPMixin._pdflip_pp_room_pass_hook(s)
        self.assertIsNone(getattr(tree, mc.PP_ROOM_CAP_ATTR, None), "no cap on PP0 alone")
        self.assertIsNone(s._pdflip_pr_verdict)

    def test_the_switch_restores_the_old_publish_nothing(self):
        from flliper.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        root, _, _ = _incident(child_host_lock=1)
        s, tree = self._pp0(root)
        os.environ["FLLIPER_PDFLIP_PP_ROOM_OWN_WALK"] = "0"
        try:
            SchedulerPPMixin._pdflip_pp_room_pass_hook(s)
        finally:
            os.environ.pop("FLLIPER_PDFLIP_PP_ROOM_OWN_WALK")
        self.assertIsNone(getattr(tree, mc.PP_ROOM_CAP_ATTR, None))


# ---------------------------------------------------------------------------
# mutants: each named danger direction, planted by monkeypatch, must turn a test of this file red
# ---------------------------------------------------------------------------


def _fails(case, names, patches):
    """True when at least one of ``names`` of ``case`` fails while ``patches`` are in force."""
    import contextlib

    suite = unittest.TestSuite(case(n) for n in names)
    result = unittest.TestResult()
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        suite.run(result)
    return bool(result.failures or result.errors)


class MutantsTest(unittest.TestCase):
    def _killed(self, case, names, patches):
        self.assertFalse(_fails(case, names, []), "the unmutated tests must pass")
        self.assertTrue(_fails(case, names, patches), "the mutant survived")

    def test_m1_the_voided_decision_stays_standing(self):
        self._killed(RetreatTest, ["test_the_continuation_is_parked_in_place_and_the_pass_is_voided"],
                     [mock.patch.object(R, "_void_decision", lambda *a, **k: None)])

    def test_m2_the_continuation_is_not_parked(self):
        self._killed(RetreatTest, ["test_the_continuation_is_parked_in_place_and_the_pass_is_voided"],
                     [mock.patch.object(R, "_park_in_place", lambda *a, **k: None)])

    def test_m3_the_wall_never_ends_in_a_503(self):
        self._killed(RetreatTest, ["test_the_wall_that_stands_after_n_retreats_answers_503_for_that_request_only",
                                   "test_an_impossible_remaining_need_ends_at_the_first_retreat_only_with_the_opt_in_switch"],
                     [mock.patch.object(R, "verdict", lambda **k: (R.RETRY, ""))])

    def test_m4_every_retreat_is_a_503(self):
        self._killed(RetreatTest, ["test_the_continuation_is_parked_in_place_and_the_pass_is_voided",
                                   "test_something_in_flight_keeps_the_request_waiting"],
                     [mock.patch.object(R, "verdict", lambda **k: (R.ABORT, "mutant"))])

    def test_m5_the_follower_retreats_too(self):
        self._killed(RetreatTest, ["test_out_of_scope_seats_re_raise"],
                     [mock.patch.object(R, "applies", lambda *a, **k: (True, ""))])

    def test_m6_a_fresh_request_is_not_given_back(self):
        self._killed(RetreatTest, ["test_fresh_requests_go_back_by_the_retract_flow"],
                     [mock.patch.object(R, "_give_back_fresh", lambda *a, **k: None)])

    def test_m7_the_walk_forgets_the_demoted_children(self):
        from test_pdflip_prefill_oom_repro_1013 import WalkFollowsThePeelTest

        self._killed(WalkFollowsThePeelTest, ["test_6193_is_not_payable_when_the_children_it_needs_gone_are_host_locked"],
                     [mock.patch.object(PR, "_remnant_clearable", lambda node, state: True)])

    def test_m9_the_carrierless_form_retreats_too(self):
        self._killed(RetreatTest, ["test_carrierless_the_typed_exception_propagates_and_nothing_of_the_pass_is_touched",
                                   "test_out_of_scope_seats_re_raise"],
                     [mock.patch.object(R, "followers_execute_row", lambda sched: True)])

    def test_m10_the_carrierless_pp0_caps_alone(self):
        self._killed(OwnWalkTest, ["test_carrierless_pp0_publishes_no_cap_of_its_own"],
                     [mock.patch.object(R, "followers_execute_row", lambda sched: True)])

    def test_m11_the_follower_or_carrierless_form_retreats_the_load_back(self):
        self._killed(LoadBackRetreatTest, ["test_out_of_scope_seats_get_none_and_nothing_is_touched"],
                     [mock.patch.object(R, "applies", lambda *a, **k: (True, ""))])

    def test_m12_the_load_back_wall_never_ends_in_a_503(self):
        self._killed(LoadBackRetreatTest, ["test_the_wall_that_stands_after_n_passes_answers_503_for_that_request_only",
                                           "test_an_impossible_need_with_nothing_in_flight_ends_at_the_first_pass_only_with_the_opt_in_switch"],
                     [mock.patch.object(R, "verdict", lambda **k: (R.RETRY, ""))])

    def test_m13_every_load_back_retreat_is_a_503(self):
        self._killed(LoadBackRetreatTest, ["test_the_request_is_skipped_for_the_pass_and_stays_queued"],
                     [mock.patch.object(R, "verdict", lambda **k: (R.ABORT, "mutant"))])

    def test_m14_the_hold_is_ignored_while_a_write_through_is_in_flight(self):
        self._killed(LoadBackRetreatTest, ["test_a_write_through_in_flight_holds_the_503_back_until_the_hold_has_passed"],
                     [mock.patch.object(R, "_wt_inflight", lambda sched: False)])

    def test_m15_the_given_back_requests_go_to_the_tail(self):
        self._killed(RetreatTest, ["test_requests_given_back_return_to_the_head_in_the_order_the_pass_admitted_them"],
                     [mock.patch.object(R, "_requeue_front", lambda *a, **k: None)])

    def test_m16_the_impossible_shortcut_is_on_by_default(self):
        self._killed(RetreatTest, ["test_without_the_switch_an_impossible_need_waits_for_the_n_retreats_as_ordered"],
                     [mock.patch.object(R, "fast_abort_enabled", lambda env=None: True)])

    def test_m8_the_guard_keeps_counting_unsent_offers(self):
        self._killed(GuardTest, ["test_retreats_do_not_escalate_the_request"],
                     [mock.patch("flliper.srt.managers.pp_admission_congruence.PPAdmissionCongruenceGuard.forget_offer",
                                 lambda self, rid: None)])


def _mutated_retreat_module(old, new):
    """The retreat module's own source with ONE planted change, executed as a separate module object."""
    src = pathlib.Path(R.__file__).read_text()
    assert old in src, f"mutation anchor gone: {old!r}"
    mod = types.ModuleType("prefill_oom_retreat_mutant")
    mod.__file__ = R.__file__
    exec(compile(src.replace(old, new, 1), R.__file__, "exec"), mod.__dict__)
    return mod


class SourceMutantsTest(unittest.TestCase):
    """Inline danger directions: the planted source change must turn a test of this file red."""

    def _killed(self, old, new, names):
        import sys

        mut = _mutated_retreat_module(old, new)
        here = sys.modules[__name__]
        self.assertFalse(_fails(RetreatTest, names, []), "the unmutated tests must pass")
        self.assertTrue(_fails(RetreatTest, names, [mock.patch.object(here, "R", mut)]), "the mutant survived")

    def test_ma_the_inflight_count_of_the_unlaunched_chunk_is_not_given_back(self):
        self._killed("carried.inflight_middle_chunks -= 1  # counted for a chunk that never launched", "pass",
                     ["test_the_continuation_is_parked_in_place_and_the_pass_is_voided"])

    def test_mh_the_last_chunk_is_taken_for_a_fresh_request(self):
        self._killed("cont = carried_before if carried_before is not None else (chunked if new_chunked is None else None)",
                     "cont = chunked if new_chunked is None else None",
                     ["test_the_last_chunk_of_a_continuation_stays_the_continuation",
                      "test_the_last_chunk_of_a_continuation_that_stands_at_the_wall_answers_503_through_the_chunked_abort"])

    def test_mi_the_last_chunk_gives_back_a_count_nobody_added(self):
        self._killed("counted = carried is not None and chunked is carried",
                     "counted = carried is not None",
                     ["test_the_last_chunk_of_a_continuation_stays_the_continuation"])

    def test_mj_the_last_chunk_is_not_restored_as_the_continuation(self):
        self._killed("restore_chunked = carried is not None and chunked is None",
                     "restore_chunked = False",
                     ["test_the_last_chunk_of_a_continuation_stays_the_continuation",
                      "test_a_pending_abort_on_a_last_chunk_is_taken_back_without_a_second_answer"])

    def test_mk_the_retreat_ignores_the_form(self):
        self._killed("    elif not followers_execute_row(sched) and not frame_ok:\n        return False, \"no_row_carrier\"\n", "",
                     ["test_carrierless_the_typed_exception_propagates_and_nothing_of_the_pass_is_touched"])

    def test_mb_the_aborted_continuation_is_not_marked_pending(self):
        self._killed("carried._pdflip_oom_abort_pending = True", "pass",
                     ["test_an_aborted_continuation_is_not_parked_rows_back_and_is_never_answered_twice"])

    def test_mc_the_fresh_new_chunked_request_stays_the_chunked_request(self):
        self._killed("sched.chunked_req = None  # the fresh request that became the chunked request goes back below",
                     "pass", ["test_a_fresh_new_chunked_request_is_no_longer_the_chunked_request"])

    def test_md_the_load_back_wire_of_the_voided_pass_stays(self):
        self._killed("    sched._pp_load_back_wire = None", "    pass",
                     ["test_the_continuation_is_parked_in_place_and_the_pass_is_voided"])

    def test_me_the_aborted_continuation_is_offered_to_the_rows_back_park(self):
        self._killed("_park_in_place(carried, flag=not (v == ABORT and blamed is carried) and not frame)", "_park_in_place(carried)",
                     ["test_an_aborted_continuation_is_not_parked_rows_back_and_is_never_answered_twice"])

    def test_mf_a_pending_abort_is_answered_again(self):
        self._killed("if carried is not None and str(carried.rid) in ledger.abort_pending:", "if False:",
                     ["test_an_aborted_continuation_is_not_parked_rows_back_and_is_never_answered_twice"])

    def test_mg_the_head_of_step_park_ignores_a_pending_oom_abort(self):
        import inspect

        from flliper.srt.managers import scheduler as sched_mod
        from flliper.srt.managers.scheduler import Scheduler

        src = textwrap.dedent(inspect.getsource(Scheduler.process_pending_pdflip_park))
        guard = 'if getattr(req, "_pdflip_oom_abort_pending", False):'
        self.assertIn(guard, src)
        ns = {}
        exec(compile(src.replace(guard, "if False:"), "<mutant>", "exec"), sched_mod.__dict__, ns)
        name = "test_an_aborted_continuation_is_not_parked_rows_back_and_is_never_answered_twice"
        self.assertFalse(_fails(RetreatTest, [name], []))
        self.assertTrue(_fails(RetreatTest, [name],
                               [mock.patch.object(Scheduler, "process_pending_pdflip_park", ns["process_pending_pdflip_park"])]),
                        "the mutant survived")


if __name__ == "__main__":
    logging.basicConfig()
    unittest.main()
