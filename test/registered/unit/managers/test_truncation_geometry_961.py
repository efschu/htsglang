"""#961: the truncation nulls a geometry whose re-derivation it does not own.

WHAT DIED (window-958-boot, boot 1, pin 78d030ec20,
boot_943bx_78d030ec20_0828_041213.log, 04:16:27Z, ~25 s under the chunked
acceptance load):

    File ".../managers/scheduler_pp_mixin.py", line 2161, in _event_loop_pp_body
      plan = self.get_next_batch_to_run(...)
    File ".../managers/scheduler.py", line 7010, in get_next_batch_to_run
      if self.chunked_req.extend_range.end > len(self.chunked_req.prefix_indices):
    AttributeError: 'NoneType' object has no attribute 'end'

The line immediately above the traceback is `#946 PREMISE RECOMPUTE
rid=f086476313a2...`, i.e. the dead-premise terminator, which ends in
`Req.truncate_prefix_to(0)` and nulls `extend_range`.

THE ATTRIBUTED BRANCH IS NOT THE ONE THAT FIRED, and this file exists partly to
record that. window-958's closeout attributed the null to the #906 seam-refusal
branch (`scheduler.py:8916`), which keeps `self.chunked_req` without calling the
adder. On metal that branch was never entered: the full-phrase counter
`[#906] SEAM CHUNK REFUSED rid=` is **0** in both boots of that window, and
`_note_seam_chunk_refused` logs its first three occurrences unconditionally
(`scheduler.py:5434-5446`), so the zero is a measurement and not a rate limit.
Other bracketed INFO tags from the same process (`[#760]`, `[#690]`, `[#839]`,
`[#959]`, `[#928 cow]`) are present in the same file, so the sink is not the
explanation either. A fix placed at that branch alone would not have touched
this crash.

THE PRODUCER THAT DID FIRE is two statements above the reader, in the traceback
itself. `scheduler_pp_mixin.py:2159` calls `pp_apply_dead_premise_anywhere(self)`
-- #948's relocated actuator, armed for that boot by `SGLANG_946_ACT_AT_RING=1`
(TICKET_958_WINDOW.md, boot recipe) -- and `:2161` calls
`get_next_batch_to_run`. `pp_request_locations` (`scheduler_pp_mixin.py:1350`)
lists `chunked_req` among the four places it sweeps, so the terminator truncates
the RESIDENT continuation, and nothing between the two statements re-derives
anything.

WHY THAT IS THE ROOT AND NOT AN OVERSIGHT. #946 placed the act inside
`_get_new_batch_prefill_raw`'s chunked block and justified it there:
"`add_chunked_req` below derives everything from `len(req.prefix_indices)` and
only THEN calls `set_extend_range`, so nothing about the next chunk is committed
yet" (`scheduler.py:8888-8896`). #948 then MOVED the act, for a measured reason
recorded at `scheduler_pp_mixin.py:2100-2109`: the old site "was entered ~6 times
while 9471 passes voided". The move took the actuator to a site that RUNS and
left the re-derivation at the site that did not. The legality argument was a
property of the old neighbourhood; it does not travel with the call.

THE FIX IS AT THE WRITER, ONE PLACE, AND IT IS NOT A READER GUARD.
`Req.truncate_prefix_to` now writes a zero-length range at the new prefix
(`Range(told, told)`) instead of `None`. That satisfies `_executed_extent`'s
invariant `extend_range.start == len(prefix_indices)` (pp_admission_congruence.py
:719-757) by construction, at the only place that can break it, and it closes
every producer in one cut rather than one branch per boot:

  * `scheduler_pp_mixin.py:2159`  the ring actuator -- the crash above;
  * `scheduler.py:8916`           the #906 seam refusal -- latent, never fired;
  * `schedule_policy.py:1396`     `add_chunked_req`'s hybrid-SWA zero-budget
                                  `return req`, which unlike the #679 park two
                                  lines below it (`:1434-1436`) returns without
                                  `set_extend_range`;
  * `scheduler.py:9087` / `:9107` the #791 clamp sites, where an
                                  `AddReqResult.NO_TOKEN` break leaves the
                                  request in the waiting queue un-re-derived.

#958'S ARGUMENT IS NOT REVERSED, IT IS HONOURED. `schedule_batch.py`'s
"NONE, NOT A RECOMPUTED RANGE" paragraph refuses `Range(told, old_end)`, because
keeping the old `end` "would INVENT a pass: the discarded tokens would have to be
computed in this chunk". `Range(told, told)` invents nothing -- it is zero rows,
and zero rows is exactly what a request whose premise was just dropped will run
this pass. It is also not a new state: `_park_chunked_prefill_chunk` writes
`Range(start, start)` for the same purpose (`scheduler_pp_mixin.py:592`), the
#679 park writes it (`schedule_policy.py:1434-1436`), and `_executed_extent`'s
own docstring declares zero-length ranges first-class ("A ZERO-LENGTH RANGE IS
REPORTED, NOT SUPPRESSED").

AND THE OFFER STILL MOVES, which was #958's whole point. `_executed_extent` now
returns `(0, 0)` instead of `None`, so PP0 offers `told=0` -- the value
`reconcile_pp_admission_decision` admits UNCONDITIONALLY -- instead of producing
an unreadable geometry that the refusal downstream never got to see. The
commit's designed net (`PPScheduleRefused`, `require_executed_geometry`) fired
**0** times on metal while the unguarded dereference killed the process, because
the net iterates `can_run_list` and the resident continuation is not in it. The
net is not made reachable by this fix; it is made unnecessary for this producer,
which is the honest claim and is asserted below rather than argued.

HARNESS PROVENANCE. `_Req` borrows the REAL `Req.truncate_prefix_to`, following
`test_offer_delivery_958.py`; a hand-copied actuator can only prove things about
the copy (the #946 suite's mistake). The `:7010` reader is driven through the
REAL `Scheduler.get_next_batch_to_run` on an uninitialised instance carrying only
the five attributes that line needs, so the assertion breaks on the production
line and not on a re-spelling of it.
"""

import types
import unittest

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25)

RID = "f086476313a24b6d8fe076a0e914e275"
PREFIX = 8192
CHUNK = 512
TOTAL = 20000


class _Req:
    """A mid-chunked-prefill continuation carrying the REAL acting half."""

    truncate_prefix_to = Req.truncate_prefix_to

    def __init__(self, prefix_len=PREFIX, chunk=CHUNK, total=TOTAL):
        self.rid = RID
        self.prefix_indices = list(range(prefix_len))
        self.extend_range = Range(prefix_len, prefix_len + chunk)
        self.full_untruncated_fill_ids = list(range(total))
        self.origin_input_ids = list(range(total))
        self.cache_protected_len = prefix_len


def _holder(req):
    """A stand-in the REAL `pp_request_locations` resolves to `chunked_req`.

    No `_prefetch_kvcache`, which is the metal case: all three recomputes in
    boot 1 reported "no re-fetch could be issued", so the terminator is what
    runs.
    """
    return types.SimpleNamespace(
        waiting_queue=[],
        chunked_req=req,
        running_batch=None,
        _pp_chunked_req_before_by_slot={},
        ps=types.SimpleNamespace(pp_rank=0, pp_size=3),
    )


def _run_the_ring_actuator(holder):
    """The REAL #948 sweep, as `scheduler_pp_mixin.py:2159` calls it."""
    from sglang.srt.managers.scheduler_pp_mixin import (
        pp_apply_dead_premise_anywhere,
        pp_mark_premise_dead,
    )

    pp_mark_premise_dead(holder.chunked_req)
    return pp_apply_dead_premise_anywhere(holder)


def _scheduler_carrying(req):
    """The REAL `Scheduler`, with only what `:7010` needs on the way in.

    Five attributes, each one demanded by a line between the top of
    `get_next_batch_to_run` and the dereference: an uninitialised instance
    reaches the production line and raises the production error on it.
    """
    s = object.__new__(Scheduler)
    s.enable_fpm = False
    s.dllm_config = None
    s.waiting_queue = []
    s._pending_chunked_abort_req = None
    s.enable_hierarchical_cache = False
    s.chunked_req = req
    return s


def _empty_running_batch():
    rb = object.__new__(ScheduleBatch)
    rb.reqs = []
    return rb


def _read_at_7010(req):
    """Drive the REAL `scheduler.py:7010`.

    Returns the exception it raised, or None when the line was crossed. The
    call continues past `:7010` into code this stand-in does not satisfy, so a
    LATER `AttributeError` naming something other than `end` means the reader
    under test was passed -- that distinction is made explicitly rather than by
    swallowing every AttributeError.
    """
    s = _scheduler_carrying(req)
    try:
        s.get_next_batch_to_run(running_batch=_empty_running_batch(), last_batch=None)
    except Exception as exc:  # noqa: BLE001 - the exception IS the measurement
        return exc
    return None


def _crashed_at_the_geometry(exc) -> bool:
    return isinstance(exc, AttributeError) and "has no attribute 'end'" in str(exc)


class TheProducerIsTheRingActuatorNotTheSeam(unittest.TestCase):
    """Reachability first. A suite that cannot enter the path proves nothing."""

    def test_the_ring_actuator_truncates_the_RESIDENT_continuation(self):
        req = _Req()
        holder = _holder(req)
        out = _run_the_ring_actuator(holder)
        self.assertEqual(
            out.get(RID),
            "recompute",
            "the #948 sweep must reach the terminator for the request that is "
            "`chunked_req`; if it does not, every assertion below is about a "
            "path this test never entered",
        )
        self.assertEqual(len(req.prefix_indices), 0, "the terminator's WRITE happened")
        self.assertIs(
            holder.chunked_req,
            req,
            "the continuation stays RESIDENT across the terminator -- dropping "
            "it here would be the re-prefill the standing law forbids, and it "
            "is what makes the geometry a cross-round obligation",
        )

    def test_nothing_between_the_actuator_and_the_reader_re_derives(self):
        """The junction pin: `:2159` and `:2161` are adjacent statements.

        The invariant therefore has to hold when the actuator RETURNS -- there
        is no later step to lean on. This is the assertion the whole fix is
        placed to satisfy.
        """
        req = _Req()
        holder = _holder(req)
        _run_the_ring_actuator(holder)
        self.assertIsNotNone(
            getattr(req, "extend_range", None),
            "a resident continuation left the dead-premise actuator with no "
            "geometry, and the very next statement in `_event_loop_pp_body` is "
            "`get_next_batch_to_run`",
        )
        self.assertEqual(
            req.extend_range.start,
            len(req.prefix_indices),
            "`_executed_extent`'s invariant: `extend_range.start` == "
            "`len(prefix_indices)`, one quantity with one expression",
        )
        self.assertEqual(
            req.extend_range.length,
            0,
            "zero rows, because zero rows is what this pass will run for it -- "
            "a non-zero length here would name a pass no rank runs (instr21)",
        )


class TheRealReadersAfterTheTruncation(unittest.TestCase):
    """Each assertion breaks on a production line, not on a re-spelling."""

    def _truncated(self):
        req = _Req()
        _run_the_ring_actuator(_holder(req))
        return req

    def test_READER_scheduler_7010_get_next_batch_to_run(self):
        exc = _read_at_7010(self._truncated())
        self.assertFalse(
            _crashed_at_the_geometry(exc),
            "scheduler.py:7010 dereferenced the geometry the terminator "
            "invalidated -- this is the window-958 boot-1 death, reproduced "
            "on the real line: {!r}".format(exc),
        )

    def test_READER_schedule_batch_2620_next_prompt_token(self):
        from sglang.srt.managers.schedule_batch import (
            _compute_chunked_req_next_prompt_token,
        )

        req = self._truncated()
        self.assertEqual(
            _compute_chunked_req_next_prompt_token(req, vocab_size=TOTAL + 1),
            0,
            "the fill boundary is now 0, so the next real prompt token is the "
            "request's first one",
        )

    def test_READER_pp_chunked_local_match_reports_a_measured_zero(self):
        from sglang.srt.managers.scheduler_pp_mixin import pp_chunked_local_match

        self.assertEqual(
            pp_chunked_local_match(self._truncated()),
            0,
            "after the premise is dropped this rank has KV for zero leading "
            "tokens, and that is a MEASUREMENT. `None` would be reported as "
            "'nothing known' and re-enter the #944 unresolved path -- a lookup "
            "miss dressed as the answer",
        )

    def test_READER_executed_extent_is_readable_and_the_offer_MOVES(self):
        """The #958 delivery line, now via a geometry instead of a refusal."""
        from sglang.srt.managers.pp_admission_congruence import _executed_extent

        before = _Req()
        self.assertEqual(_executed_extent(before), (PREFIX, CHUNK))
        after = self._truncated()
        self.assertEqual(
            _executed_extent(after),
            (0, 0),
            "the terminator discarded the whole prefix; the production offer "
            "must read 0 -- the value `reconcile_pp_admission_decision` admits "
            "unconditionally, which is what ends the retraction storm",
        )

    def test_the_PRODUCER_puts_a_zero_offer_on_the_wire(self):
        """Read off `build_pp_admission_decision`, never re-derived here."""
        from sglang.srt.managers.pp_admission_congruence import (
            PPAdmissionCongruenceGuard,
            build_pp_admission_decision,
        )

        req = self._truncated()
        decision = build_pp_admission_decision(
            0,
            [req],
            pp_size=3,
            guard=PPAdmissionCongruenceGuard(unresolved_defer_cap=3),
            require_executed_geometry=True,
        )
        self.assertEqual(decision.entries[0].prefix_len, 0)


class TheInvariantIsLoadBearingAtEveryReader(unittest.TestCase):
    """CANFAIL. The mutant is the pre-fix writer: geometry nulled, request kept.

    These assertions must hold BEFORE and AFTER the fix. They are what makes
    the suite above non-decorative: without them, a green run could mean the
    readers never cared.
    """

    def _mutant(self):
        req = _Req()
        req.prefix_indices = []
        req.cache_protected_len = 0
        req.extend_range = None  # exactly what truncate_prefix_to used to leave
        return req

    def test_MUTANT_kills_reader_scheduler_7010(self):
        exc = _read_at_7010(self._mutant())
        self.assertTrue(
            _crashed_at_the_geometry(exc),
            "the pre-fix shape must still kill this reader; if it does not, "
            "the reader stopped depending on the invariant and this suite is "
            "measuring nothing: {!r}".format(exc),
        )

    def test_MUTANT_kills_reader_schedule_batch_2620(self):
        from sglang.srt.managers.schedule_batch import (
            _compute_chunked_req_next_prompt_token,
        )

        with self.assertRaises(AttributeError):
            _compute_chunked_req_next_prompt_token(self._mutant(), vocab_size=TOTAL + 1)

    def test_MUTANT_makes_the_offer_UNREADABLE(self):
        from sglang.srt.managers.pp_admission_congruence import _executed_extent

        self.assertIsNone(_executed_extent(self._mutant()))

    def test_MUTANT_makes_local_match_report_UNKNOWN_instead_of_zero(self):
        from sglang.srt.managers.scheduler_pp_mixin import pp_chunked_local_match

        self.assertIsNone(pp_chunked_local_match(self._mutant()))

    def test_the_designed_net_never_sees_this_request(self):
        """Why the commit's own safety net fired 0 times on metal.

        `build_pp_admission_decision` refuses an unreadable geometry -- but it
        iterates the list it is handed, and the resident continuation is not in
        `can_run_list` on a pass where the adder did not add it. The net sits
        downstream of the reader that dies. This asserts the structural fact,
        so "the net is unnecessary here" is measured rather than assumed.
        """
        from sglang.srt.managers.pp_admission_congruence import (
            PPAdmissionCongruenceGuard,
            build_pp_admission_decision,
        )

        resident = self._mutant()
        decision = build_pp_admission_decision(
            0,
            [],  # the pass built nothing: this is `can_run_list`
            pp_size=3,
            guard=PPAdmissionCongruenceGuard(unresolved_defer_cap=3),
            require_executed_geometry=True,
        )
        self.assertEqual(
            len(decision.entries),
            0,
            "the net cannot refuse what it is never handed -- the request that "
            "kills the next pass is {}".format(resident.rid),
        )


class TheTruncationMayNotOVERREACH(unittest.TestCase):
    """The other danger direction: repairing more than was broken."""

    def test_a_NO_OP_truncation_leaves_a_healthy_geometry_untouched(self):
        req = _Req()
        before = req.extend_range
        req.truncate_prefix_to(PREFIX + 1000)  # told >= len(prefix_indices)
        self.assertIs(
            req.extend_range,
            before,
            "a truncation that moves nothing may not rewrite a valid range; "
            "voiding healthy passes for nothing is the opposite defect",
        )

    def test_a_PARTIAL_truncation_anchors_at_the_new_prefix(self):
        req = _Req()
        req.truncate_prefix_to(4096)
        self.assertEqual(req.extend_range, Range(4096, 4096))
        self.assertEqual(len(req.prefix_indices), 4096)
        self.assertEqual(req.cache_protected_len, 4096)

    def test_the_old_END_is_never_carried_forward(self):
        """#958's refusal, kept: `Range(told, old_end)` would invent a pass."""
        req = _Req()
        req.truncate_prefix_to(0)
        self.assertEqual(
            req.extend_range.end,
            0,
            "carrying the old end would demand the discarded tokens be "
            "computed in this chunk, past the budget the adder decided",
        )


if __name__ == "__main__":
    unittest.main()
