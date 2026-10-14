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


def _sched(*, pp_rank=0, tp_size=1, floor=True, relay=True):
    tree = types.SimpleNamespace(token_to_kv_pool_allocator=_Alloc(), ongoing_write_through={}, ongoing_load_back={})
    setattr(tree, SF.FLOOR_LOCAL_PP_ATTR, floor)
    sent = []
    s = types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_rank=pp_rank, tp_size=tp_size, pp_size=3),
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
        self._env = {k: os.environ.get(k) for k in (R.ENV, R.ENV_RETRIES, R.GROUP_ENV, "FLLIPER_PDFLIP_PARK")}
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

    def test_an_impossible_remaining_need_ends_at_the_first_retreat(self):
        """The wall will not move (nothing in flight, remaining 128901 > free+paid): no N passes into it."""
        s = _sched()
        req = _Req("pdflip-587-1256", prefix=98240, total=148965)
        s.chunked_req = req
        self.assertTrue(R.retreat(s, _exc(), None, [req], _adder()))
        self.assertEqual(len(s._pdflip_intake_origin_aborts.take()), 1)

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
        }
        for name, c in cases.items():
            with self.subTest(name):
                self.assertFalse(R.retreat(c["sched"], exc, None, [req], _adder()))
                self.assertIsNone(req.extend_range)
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
    def _pp0(self, root):
        tree = types.SimpleNamespace(
            root_node=root, ongoing_write_through={},
            cache_controller=types.SimpleNamespace(write_policy="write_back", mem_pool_host=None),
            uniform_avail_floor=None, evictable_size=lambda: 150336,
        )
        tree.token_to_kv_pool_allocator = types.SimpleNamespace(available_size=lambda: 13568)
        setattr(tree, SF.FLOOR_LOCAL_PP_ATTR, True)
        s = types.SimpleNamespace(
            ps=types.SimpleNamespace(pp_size=3, pp_rank=0, attn_tp_rank=0, attn_cp_rank=0),
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
                                   "test_an_impossible_remaining_need_ends_at_the_first_retreat"],
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
        self._killed("_park_in_place(carried, flag=not (v == ABORT and blamed is carried))", "_park_in_place(carried)",
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
