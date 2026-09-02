"""#1153: a follower's refusal of a forwarded schedule is a group STOP, not a void.

THE SPECIMEN, boot_855_weg1b2_8bf12cfd44_0902_213438.log:

    65000  PP1  #791 PP-ADMISSION forwarded schedule REFUSED on rank 1:
                ... the decision names 2 request(s) and this rank's
                admission loop reached only 1 ...
    65001  PP1  #788 ... verdict=DECLINE ... reason=loop_skips(batch_full_break=1)
    65004  PP1  #631 ROW-DELIVER BATCH NULLED slot=0 ... pass_voided=True
    65119  PP0  Prefill batch ... (PP2's NEXT output, merged under this slot)

The handler that answered the refusal (`_pp_refuse_forwarded_schedule`,
deleted) set `_pp_admission_pass_voided`, `_pp_void_own_batch` nulled PP1's
slot, PP1 sent no proxy, PP0's slot stayed set, nothing carried the void
upstream (the #797 return trip has zero call sites since CUT V; #1072
deleted the void relay) -- so PP0's blocking recv consumed the last rank's
next real output under this slot's label and was one output ahead for the
rest of the boot, until the pp_to_tp arm turned the debt into an
unproducible output and #980 / #1071 stopped the group 60-90 s later.

THREE PINS, one per operator decision:

  T1 (F1, the root): the funnel `Scheduler.get_new_batch_prefill` answers a
     `PPScheduleRefused` with a RuntimeError whose message starts with
     `#791 FORWARDED SCHEDULE UNEXECUTABLE STOP` and names rank, slot, told
     and reached rids, the skip census and every count-arm input -- and
     sets NO `_pp_admission_pass_voided`, nulls NO batch.
  T2 (F2, the trigger class): on a forwarded schedule the follower's
     rank-local seat-count veto (`rank_local_count_veto_applies`) is off;
     on PP0 and on a non-PP boot it is the pre-#1153 expression, unchanged.
     Pinned on the helper AND on the shipped loop's source, so restoring the
     veto in either place is red.
  T3 (F3, the sibling): the row-authority `_row_skip_plan` exit applies the
     same named #1020 guard the void path applies, so a slot this rank
     launched and is still owed a result for is never nulled silently.

The harness follows test_pp_refused_pass_keeps_continuation_971: a bare
`SimpleNamespace` with the REAL methods bound via `types.MethodType`, and
`_get_new_batch_prefill_raw` stood in by the refusal it raises.
"""

import inspect
import logging
import re
import types
import unittest
from unittest.mock import patch

from sglang.srt.managers.pp_admission_congruence import (
    FORWARDED_SCHEDULE_STOP_FORMAT,
    FORWARDED_SCHEDULE_STOP_PREFIX,
    PPAdmissionDecision,
    PPAdmissionEntry,
    PPScheduleRefused,
    forwarded_schedule_stop_message,
    rank_local_count_veto_applies,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-b-test-small-1-gpu")

RID_TOLD_A = "told-and-reached-1153"
RID_TOLD_B = "told-but-unreached-1153"
MB_ID = 1


class _Limiter:
    current = 8


class _Pool:
    def available_size(self):
        return 61


class _ParkedSet:
    def admission_headroom(self, running_bs, requested):
        return min(int(requested), 20 - int(running_bs))


class _Req:
    def __init__(self, rid):
        self.rid = rid


def _holder(*, rank=1, told=None):
    told = {RID_TOLD_A: (0, 64), RID_TOLD_B: (0, 64)} if told is None else told
    h = types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_rank=rank, pp_size=3),
        pp_group=types.SimpleNamespace(is_first_rank=rank == 0, is_last_rank=False),
        running_batch=types.SimpleNamespace(reqs=[_Req("resident-1153")]),
        prefill_delayer=None,
        admission_limiter=_Limiter(),
        req_to_token_pool=_Pool(),
        parked_decode_set=_ParkedSet(),
        _pp_admission_incoming_schedule=told,
        _pp_admission_pass_voided=False,
        _pp_admission_incoming_effective=dict(told),
        _pp_live_mb_id=MB_ID,
        _admission_decline_note="loop_skips(batch_full_break=1(first=told-but-unre))",
        _pp_admission_reached_rids=(RID_TOLD_A,),
        _pp_batch_full_setter="count_arm",
        _pp_batch_full_at_loop_entry=False,
        _pp_head_inputs_this_pass=types.SimpleNamespace(admit_limit=1),
        void_calls=[],
    )
    h._pp_admission_amended_to_forward = PPAdmissionDecision(
        mb_id=MB_ID,
        entries=tuple(
            PPAdmissionEntry(rid=rid, prefix_len=0, extend_len=64, admitted=True)
            for rid in told
        ),
    )
    h.get_num_allocatable_reqs = lambda running_bs: 7 - int(running_bs)
    h._parked_carrier_discount = lambda running_bs: 0
    h._trace_pp_admission_verdict = lambda ret: None
    h._pp_void_own_batch = lambda mb_id: h.void_calls.append(mb_id)
    h._pp_scheduled_extents = types.MethodType(Scheduler._pp_scheduled_extents, h)
    h._pp_forwarded_schedule_stop = types.MethodType(
        Scheduler._pp_forwarded_schedule_stop, h
    )
    h.get_new_batch_prefill = types.MethodType(Scheduler.get_new_batch_prefill, h)
    return h


def _refuse(h, message=None):
    """Drive one pass whose stood-in `_get_new_batch_prefill_raw` refuses."""
    message = message or (
        f"#791 FORWARDED SCHEDULE UNEXECUTABLE: the decision names 2 "
        f"request(s) and this rank's admission loop reached only 1; "
        f"missing rid(s)={RID_TOLD_B}."
    )

    def _raw(*, prefill_delayer_single_pass, running_batch):
        raise PPScheduleRefused(message)

    h._get_new_batch_prefill_raw = _raw
    with unittest.TestCase().assertRaises(RuntimeError) as ctx:
        h.get_new_batch_prefill(running_batch=h.running_batch)
    return ctx.exception


class T1ARefusedForwardedScheduleStopsTheGroup(unittest.TestCase):
    """F1: the compensation is gone; the refusal is a named RuntimeError."""

    def test_the_refusal_is_a_runtime_error_with_the_stop_prefix(self):
        exc = _refuse(_holder())
        self.assertTrue(
            str(exc).startswith(FORWARDED_SCHEDULE_STOP_PREFIX),
            f"the Boot-3 acceptance grep anchors on the prefix: {exc}",
        )
        self.assertIsInstance(exc.__cause__, PPScheduleRefused)
        self.assertNotIsInstance(exc, PPScheduleRefused)

    def test_the_stop_line_names_every_term(self):
        with patch(
            "sglang.srt.managers.scheduler.get_server_args",
            return_value=types.SimpleNamespace(pp_max_micro_batch_size=2),
        ):
            msg = str(_refuse(_holder()))
        for term in (
            "rank=1",
            f"slot={MB_ID}",
            f"told=[{RID_TOLD_A},{RID_TOLD_B}]",
            f"reached=[{RID_TOLD_A}]",
            "census=loop_skips(batch_full_break=1(first=told-but-unre))",
            "local=6",
            "limiter=8",
            # #1153 follow-up: the FIRST min() term of get_num_allocatable_reqs
            # (=2 under the flip override on weg1b2), the likeliest trigger.
            "pp_max_mb=2",
            "running_bs=1",
            "parked=0",
            "r2t_avail=61",
            "headroom=6",
            "group_limit=1",
            "batch_full_setter=count_arm",
            "batch_full_at_loop_entry=False",
            "the decision names 2 request(s)",
        ):
            self.assertIn(term, msg)

    def test_no_void_flag_is_set_and_no_batch_is_nulled(self):
        """THE WITHDRAWAL: `_pp_admission_pass_voided` stays False, the
        decision dicts are not emptied, and `_pp_void_own_batch` is never
        reached -- the three things the deleted handler did."""
        h = _holder()
        _refuse(h)
        self.assertFalse(h._pp_admission_pass_voided)
        self.assertEqual(
            h._pp_admission_incoming_schedule,
            {RID_TOLD_A: (0, 64), RID_TOLD_B: (0, 64)},
        )
        self.assertEqual(
            h._pp_admission_incoming_effective, dict(h._pp_admission_incoming_schedule)
        )
        self.assertEqual(h.void_calls, [])

    def test_the_detector_line_is_still_logged_before_the_stop(self):
        with self.assertLogs(
            "sglang.srt.managers.scheduler", level=logging.ERROR
        ) as cm:
            _refuse(_holder())
        self.assertTrue(
            any(
                "#791 PP-ADMISSION forwarded schedule REFUSED on rank 1" in line
                for line in cm.output
            ),
            cm.output,
        )

    def test_an_unreadable_probe_never_masks_the_stop(self):
        """A stand-in with no pools at all still gets the STOP line, with
        `n/a` where the number could not be read."""
        h = _holder()
        del h.req_to_token_pool
        del h.admission_limiter
        del h.parked_decode_set
        with patch(
            "sglang.srt.managers.scheduler.get_server_args",
            side_effect=AttributeError("no server args in this process"),
        ):
            msg = str(_refuse(h))
        self.assertTrue(msg.startswith(FORWARDED_SCHEDULE_STOP_PREFIX))
        self.assertIn("limiter=n/a", msg)
        self.assertIn("pp_max_mb=n/a", msg)
        self.assertIn("r2t_avail=n/a", msg)
        self.assertIn("headroom=n/a", msg)

    def test_pp0_own_refusal_stops_too_with_told_na(self):
        """The `#791 SCHEDULE UNBUILDABLE` raise in
        `build_pp_admission_decision` is PP0's and funnels here too; PP0
        owns its admission truth, so `told` is n/a rather than a map."""
        h = _holder(rank=0)
        msg = str(_refuse(h, "#791 SCHEDULE UNBUILDABLE for rid=x"))
        self.assertIn("rank=0", msg)
        self.assertIn("told=[n/a]", msg)

    def test_the_compensation_is_deleted_from_the_shipped_class(self):
        """THE MATCHED CHECK, as a test: the error class of this edit is 'a
        follower still ends a PP0-launched pass silently'. The handler is
        gone, the funnel raises, and the funnel writes no void flag."""
        self.assertFalse(hasattr(Scheduler, "_pp_refuse_forwarded_schedule"))
        funnel = inspect.getsource(Scheduler.get_new_batch_prefill)
        self.assertIn("except PPScheduleRefused", funnel)
        self.assertIn("raise self._pp_forwarded_schedule_stop(refusal)", funnel)
        self.assertNotIn("_pp_admission_pass_voided = True", funnel)
        stop = inspect.getsource(Scheduler._pp_forwarded_schedule_stop)
        self.assertNotIn("_pp_admission_pass_voided = True", stop)
        self.assertNotIn("void_pp_admission_decision", stop)
        self.assertNotIn("pp_rehome_refused_chunked_req", stop)

    def test_the_format_string_is_the_one_boot_3_greps(self):
        line = forwarded_schedule_stop_message(
            rank=1,
            slot=0,
            told=["b", "a"],
            reached=["a"],
            census="loop=clean",
            local=1,
            limiter=8,
            pp_max_mb=4,
            running_bs=1,
            parked=1,
            r2t_avail=19,
            headroom=8,
            group_limit=None,
            batch_full_setter=None,
            batch_full_at_loop_entry=True,
            refusal="why",
        )
        self.assertEqual(
            line,
            "#791 FORWARDED SCHEDULE UNEXECUTABLE STOP rank=1 slot=0 told=[a,b] "
            "reached=[a] census=loop=clean local=1 limiter=8 pp_max_mb=4 "
            "running_bs=1 parked=1 r2t_avail=19 headroom=8 group_limit=None "
            "batch_full_setter=none batch_full_at_loop_entry=True: why",
        )
        self.assertTrue(
            FORWARDED_SCHEDULE_STOP_FORMAT.startswith(FORWARDED_SCHEDULE_STOP_PREFIX)
        )

    def test_an_in_loop_raise_still_names_what_the_loop_reached(self):
        """#1153 follow-up: the reached rids are recorded AT the in-loop
        `except PPScheduleRefused` (from `adder.can_run_list`, via
        `_pp_record_reached_rids`), not only after the loop, so a refusal
        raised inside `add_one_req` never prints `reached=[]` for a pass
        whose can_run_list held rids."""
        h = _holder()
        h._pp_admission_reached_rids = ()
        h._pp_record_reached_rids = types.MethodType(
            Scheduler._pp_record_reached_rids, h
        )

        def _raw(*, prefill_delayer_single_pass, running_batch):
            # what the loop's except path does at the moment of the raise
            h._pp_record_reached_rids([_Req(RID_TOLD_A)])
            raise PPScheduleRefused("#791 FORWARDED SCHEDULE UNEXECUTABLE: in-loop")

        h._get_new_batch_prefill_raw = _raw
        with self.assertRaises(RuntimeError) as ctx:
            h.get_new_batch_prefill(running_batch=h.running_batch)
        self.assertIn(f"reached=[{RID_TOLD_A}]", str(ctx.exception))
        self.assertEqual(h._pp_admission_reached_rids, (RID_TOLD_A,))

    def test_the_loop_records_reached_at_the_in_loop_except(self):
        """SOURCE PIN (nothing in this tree drives the loop): the record sits
        between the `except PPScheduleRefused as exc:` and its `break`, and
        the post-loop record uses the same helper."""
        src = inspect.getsource(Scheduler._get_new_batch_prefill_raw)
        self.assertIn(
            "self._pp_record_reached_rids(adder.can_run_list)\n"
            "                schedule_refusal = exc\n"
            "                break",
            src,
        )
        self.assertEqual(
            src.count("self._pp_record_reached_rids(adder.can_run_list)"), 2
        )


class T2TheCountArmIsNotAVerdictOnAForwardedSchedule(unittest.TestCase):
    """F2: the follower seats the told rids; its own seat count is not a veto."""

    def test_the_veto_is_off_on_a_forwarded_schedule(self):
        self.assertFalse(rank_local_count_veto_applies({RID_TOLD_A: (0, 64)}))
        self.assertFalse(
            rank_local_count_veto_applies({RID_TOLD_A: (0, 64), RID_TOLD_B: (0, 64)})
        )

    def test_the_veto_is_on_where_the_rank_owns_its_admission(self):
        """PP0 and every non-PP boot: `_pp_scheduled_extents` is None there."""
        self.assertTrue(rank_local_count_veto_applies(None))
        self.assertTrue(rank_local_count_veto_applies({}))

    def test_pp0_and_single_rank_hand_the_helper_none(self):
        h = _holder(rank=0)
        self.assertIsNone(h._pp_scheduled_extents())
        h1 = _holder(rank=1)
        h1.ps.pp_size = 1
        self.assertIsNone(h1._pp_scheduled_extents())
        self.assertIsNotNone(_holder(rank=1)._pp_scheduled_extents())

    def test_the_shipped_loop_gates_the_count_arm_on_the_helper(self):
        """SOURCE PIN, because nothing in this tree drives the ~700-line
        `_get_new_batch_prefill_raw`. Every use of the #823 count arm and
        the `batch_full_break` break must be gated on `_count_veto`, and
        `_count_veto` must come from the helper. Restoring the veto in the
        loop (mutant M2) is red here even if the helper is untouched."""
        src = inspect.getsource(Scheduler._get_new_batch_prefill_raw)
        self.assertIn(
            "_count_veto = rank_local_count_veto_applies(self._pp_scheduled_extents())",
            src,
        )
        arms = re.findall(
            r"if (.*?)len\(\s*adder\.can_run_list\s*\)\s*>=\s*self\._uniform_allocatable_reqs\(",
            src,
            re.S,
        )
        self.assertEqual(len(arms), 1, "exactly one #823 count arm in the loop")
        self.assertIn("_count_veto and", arms[0])
        self.assertIn(
            "if running_batch.batch_is_full and (_count_veto or _disagg_full):",
            src,
        )
        self.assertNotIn("\n            if running_batch.batch_is_full:\n", src)
        # The three pre-loop count gates are the same arithmetic one step
        # earlier and worse (no refusal line at all): gated identically.
        self.assertIn("(running_batch.batch_is_full and _count_veto)", src)
        self.assertIn(
            "self.min_free_slots_delayer is not None\n            and _count_veto",
            src,
        )
        self.assertIn(
            "_count_veto\n            and self.get_num_allocatable_reqs(running_bs) <= 0",
            src,
        )
        # #1153 follow-up: the FOURTH site of the same arithmetic, and the
        # one that ACTUATES (retracts a parked carrier): gated identically.
        self.assertEqual(
            src.count(
                "_count_veto\n            and self.get_num_allocatable_reqs(running_bs) <= 0"
            ),
            2,
            "the parked-carrier yield and the no_allocatable gate",
        )
        self.assertNotIn(
            "        if (\n            self.get_num_allocatable_reqs(running_bs) <= 0\n",
            src,
        )


class _Sentinel(Exception):
    """Stops the real `_get_new_batch_prefill_raw` right after the gates."""


def _loop_holder(*, rank):
    """A stand-in that drives the REAL `_get_new_batch_prefill_raw` from its
    first line to the parked-carrier yield site, with a local seat count of
    0 and one queued request. PP0 declines at the no_allocatable gate; a
    follower on a non-empty told map runs past both gates into
    `policy.calc_priority`, which is stood in by `_Sentinel`."""

    def _stop(*args, **kwargs):
        raise _Sentinel()

    h = types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_rank=rank, pp_size=3),
        pp_group=types.SimpleNamespace(is_first_rank=rank == 0, is_last_rank=False),
        server_args=types.SimpleNamespace(enable_flexkv=False),
        grammar_manager=types.SimpleNamespace(has_waiting_grammars=lambda: False),
        enable_hierarchical_cache=False,
        enable_hicache_storage=False,
        enable_priority_preemption=False,
        is_hybrid_swa=False,
        chunked_req=None,
        min_free_slots_delayer=None,
        waiting_queue=[_Req("queued-1153")],
        running_batch=types.SimpleNamespace(
            reqs=[_Req("carrier-1153")], batch_is_full=False, is_empty=lambda: False
        ),
        _pp_admission_incoming_schedule={RID_TOLD_A: (0, 64)},
        policy=types.SimpleNamespace(calc_priority=_stop),
        yield_calls=[],
    )
    h._drain_prefetch_progress = lambda: {}
    h._take_uniform_head_inputs = lambda: None
    h._rederive_latched_batch_full = lambda running_batch: False
    h.get_num_allocatable_reqs = lambda running_bs: 0

    def _yield(running_batch, running_bs, allocatable):
        h.yield_calls.append((running_bs, allocatable))
        return False

    h._maybe_yield_parked_carrier = _yield
    h._pp_scheduled_extents = types.MethodType(Scheduler._pp_scheduled_extents, h)
    h._get_new_batch_prefill_raw = types.MethodType(
        Scheduler._get_new_batch_prefill_raw, h
    )
    return h


class T4TheParkedCarrierYieldIsGatedLikeTheCountGates(unittest.TestCase):
    """#1153 follow-up: `_maybe_yield_parked_carrier` retracts a parked decode
    carrier on THIS rank's own seat count -- an actuator, a rank-local state
    change the peers do not make. On a forwarded schedule it must not run."""

    def test_pp0_with_no_seat_still_yields_and_then_declines(self):
        h = _loop_holder(rank=0)
        ret, rb = h._get_new_batch_prefill_raw(
            prefill_delayer_single_pass=None, running_batch=h.running_batch
        )
        self.assertIsNone(ret)
        self.assertEqual(h.yield_calls, [(1, 0)])
        self.assertEqual(h._pp_batch_full_setter, "no_allocatable_reqs_gate")
        self.assertTrue(rb.batch_is_full)

    def test_a_follower_on_a_told_map_never_yields_a_carrier_on_its_own(self):
        h = _loop_holder(rank=1)
        with self.assertRaises(_Sentinel):
            h._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None, running_batch=h.running_batch
            )
        self.assertEqual(h.yield_calls, [], "a rank-local retraction below PP0")
        self.assertFalse(h.running_batch.batch_is_full)
        self.assertIsNone(h._pp_batch_full_setter)


class T5TheMembershipLineIsTheOnlyStopAndIsNotVetoGated(unittest.TestCase):
    """#1153 follow-up (reviewer mutant MC): with the count veto off, the
    post-loop `if missing:` is the only thing turning a follower's physical
    inability (add_one_req NO_TOKEN -> a told rid not reached) into a STOP.
    SOURCE PIN, because nothing in this tree drives the loop: the block is
    not gated on `_count_veto`, and the mutant `if missing and _count_veto:`
    is red here."""

    def test_the_missing_check_is_ungated(self):
        src = inspect.getsource(Scheduler._get_new_batch_prefill_raw)
        start = src.index(
            "scheduled_extents = self._pp_scheduled_extents()\n"
            "        if scheduled_extents:"
        )
        seg = src[start : src.index("extra = [rid for rid in admitted_rids", start)]
        self.assertIn("            if missing:\n", seg)
        self.assertNotIn("_count_veto", seg)
        self.assertIn("raise PPScheduleRefused(", seg)
        self.assertIn("missing rid(s)=", seg)


class T3ALaunchedSlotIsNeverNulledSilently(unittest.TestCase):
    """F3: the `_row_skip_plan` exit carries the #1020 guard the void path has."""

    def _h(self, pending):
        h = types.SimpleNamespace(
            mbs=[None, "batch-on-slot-1", None],
            mb_metadata=[None, "meta", None],
            _pp_launched_pending=set(pending),
        )
        h._pp_slot_holds_unconsumed_launch = types.MethodType(
            SchedulerPPMixin._pp_slot_holds_unconsumed_launch, h
        )
        h._pp_null_frameless_slot = types.MethodType(
            SchedulerPPMixin._pp_null_frameless_slot, h
        )
        return h

    def test_an_idle_frameless_slot_is_nulled(self):
        h = self._h(pending=())
        self.assertTrue(h._pp_null_frameless_slot(1))
        self.assertIsNone(h.mbs[1])
        self.assertEqual(getattr(h, "_1020_refused", 0), 0)

    def test_a_launched_frameless_slot_keeps_its_batch_and_says_so(self):
        h = self._h(pending=(1,))
        with self.assertLogs(
            "sglang.srt.managers.scheduler_pp_mixin", level=logging.WARNING
        ) as cm:
            self.assertFalse(h._pp_null_frameless_slot(1))
        self.assertEqual(h.mbs[1], "batch-on-slot-1", "never nulled silently")
        self.assertEqual(h._1020_refused, 1)
        self.assertTrue(
            any(
                "#1020 VOID REFUSED ON A LAUNCHED SLOT" in line
                and "site=row_skip_plan" in line
                for line in cm.output
            ),
            cm.output,
        )

    def test_the_void_path_uses_the_same_guard(self):
        """One predicate, two sites: `_pp_void_own_batch` names its site and
        the row-skip branch of the loop calls the frameless helper."""
        void = inspect.getsource(SchedulerPPMixin._pp_void_own_batch)
        self.assertIn(
            'self._pp_slot_holds_unconsumed_launch(mb_id, "void_own_batch")', void
        )
        loop = inspect.getsource(SchedulerPPMixin._event_loop_pp_body)
        branch = loop[loop.index("if _row_skip_plan:") :]
        branch = branch[: branch.index("elif _pre_proxy is not None:")]
        self.assertIn("self._pp_null_frameless_slot(mb_id)", branch)
        self.assertNotIn("self.mbs[mb_id] = None", branch)


class TheOtherWriterIsProvablyUnreachable(unittest.TestCase):
    """F3, mixin:9331: `_pp_void_pass_without_upstream_launch` is the last
    writer of `_pp_admission_pass_voided = True`, and it returns at its
    first statement because `pp_upstream_void_pending` returns False on
    every path (its final statement is `return False`)."""

    def test_the_predicate_is_constant_false(self):
        from sglang.srt.managers.scheduler_pp_mixin import pp_upstream_void_pending

        src = inspect.getsource(pp_upstream_void_pending)
        body = src[src.index('"""', src.index('"""') + 3) + 3 :]
        self.assertNotIn("return True", body)
        self.assertTrue(body.rstrip().endswith("return False"))
        h = types.SimpleNamespace(
            ps=types.SimpleNamespace(pp_size=3, pp_rank=1),
            pp_group=types.SimpleNamespace(is_first_rank=False),
            _pp_gapped_wire=False,
            _pp_upstream_launched_incoming=False,
            _pp_pass_voided_incoming=True,
        )
        self.assertFalse(pp_upstream_void_pending(h))

    def test_the_streak_writer_returns_before_writing(self):
        h = types.SimpleNamespace(
            ps=types.SimpleNamespace(pp_size=3, pp_rank=1),
            pp_group=types.SimpleNamespace(is_first_rank=False),
            _pp_gapped_wire=False,
            mbs=[None, "batch", None],
            _pp_admission_pass_voided=False,
            _pp_launched_pending={1},
        )
        h._pp_void_pass_without_upstream_launch = types.MethodType(
            SchedulerPPMixin._pp_void_pass_without_upstream_launch, h
        )
        self.assertFalse(h._pp_void_pass_without_upstream_launch(1))
        self.assertFalse(h._pp_admission_pass_voided)
        self.assertEqual(h.mbs[1], "batch")


if __name__ == "__main__":
    unittest.main()
