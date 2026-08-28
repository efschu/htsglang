"""#958: the terminator writes one quantity and the offer is read from another.

THE MEASURED FACT THIS FILE EXISTS FOR (window-955-boot, pin 27bcb4884f,
boot_943bx_27bcb4884f_0828_024615.log). The recompute terminator fired three
times, on three distinct rids, exactly once each -- #955 works. And the offer
did not move. For rid `a6132c5de55f4e4db70f2ac3f6131d6e` the log carries
`told=8192` NINE times before its `#946 PREMISE RECOMPUTE` and THREE HUNDRED
AND THIRTY-SIX times after it. Not one genuine `told=0` offer exists in the
whole boot: the three `told=0` matches a bare grep finds are the literal
string inside the UNRESOLVABLE message's own prose ("it now offers told=0,
which is honourable without a measurement") -- the same grep trap the
`#801-spin` counter already taught this family, in a second colour.

THE ROOT, and it is NOT that the clamp accessor is on the wrong branch.

  * The acting half is `Req.truncate_prefix_to` (schedule_batch.py:1611). It
    writes `prefix_indices` and `cache_protected_len`. It does NOT touch
    `extend_range`.
  * The production offer is `_executed_extent`
    (pp_admission_congruence.py:719-757), which reads `extend_range.start`
    and nothing else.

So the terminator's effect is invisible to the only quantity that carries the
offer. `_executed_extent`'s own docstring states the invariant it depends on --
"`extend_range.start` == `len(prefix_indices)`" -- and closes with "Two
expressions that must agree is the defect one level up, so there is exactly one
expression". The terminator breaks precisely that invariant, and nothing
asserts it, so the stale `start` is reported as though it were a fresh
measurement.

WHY THE LIVELOCK FOLLOWS. `reconcile_pp_admission_decision` admits `told <= 0`
UNCONDITIONALLY (pp_admission_congruence.py:1003-1025) -- "A ZERO OFFER DEMANDS
NOTHING, so no lookup result -- not even a failed one -- can make it
unhonourable." A real `told=0` therefore ends the retraction storm whatever the
downstream lookup does. The escape was built to reach exactly that branch and
never delivers it, so PP1 retracts every pass, the pass voids, and PP2 voids
512 CONSECUTIVE passes into the `#801-spin` refusal.

WHY THE DESK WAS GREEN WHILE METAL WAS RED, recorded because it is the reusable
lesson. `test_chunked_continuation_clamp_946.py`'s terminator test asserts
`len(req.prefix_indices) == 0` and calls that "the terminator applied". That
measures the WRITE. Delivery is whether the next OFFER moves, and the same
stand-in leaves `extend_range` at 8192 exactly as production does -- so the
assertion passes on a request whose next offer is unchanged. An indicator is
not a finding until it is shown to measure what it claims (INDIKATOR-GESETZ).
Every test here asserts the offer the PRODUCER emits, never a field the
actuator happens to write.

THE FIX MUST NOT BE A CLAMP ON THE EXECUTED BRANCH, and this is a refusal with
a file:line, not a preference. `build_pp_admission_decision`'s executed branch
(:866-889) REPORTS the geometry the rank actually ran; rewriting it names a
pass no rank ran. That is the instr21 defect verbatim (:794-816: a 512-row
chunk on the wire against a batch of 16983 tokens, dead in 37 s), and three
separate places in the module forbid it -- :468-472, :625-626, :873-879. The
offer has to move by the request's geometry genuinely being re-derived, not by
the report lying about what ran.
"""

import types
import unittest

from sglang.srt.managers.schedule_batch import Req
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20)

RID = "rid-mid-chunked-prefill"

# The metal geometry, so a regression reads against the numbers in the log.
PREFIX = 8192
EXTEND = 4096
CAP = 3  # UNRESOLVED_DEFER_CAP


class _Range:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class _Req:
    """A mid-chunked-prefill continuation carrying the REAL acting half.

    `truncate_prefix_to` is BORROWED from `Req` rather than re-spelled, and
    that is the point of this file rather than a convenience. The #946 suite
    hand-copied the method into its own stand-in, asserted
    `len(prefix_indices) == 0` afterwards, and called the terminator delivered
    -- while production offered the same `told=8192` 336 more times. A copied
    actuator can only ever prove things about the copy.

    Everything else here is data (a rid, a prefix, a geometry, the full token
    list), which is what a stand-in is legitimately for.
    """

    truncate_prefix_to = Req.truncate_prefix_to

    def __init__(self, rid, prefix_len, extend_len):
        self.rid = rid
        self.prefix_indices = list(range(prefix_len))
        self.extend_range = _Range(prefix_len, prefix_len + extend_len)
        self.full_untruncated_fill_ids = list(range(prefix_len + extend_len))
        self.cache_protected_len = prefix_len


def _holder(**kw):
    h = types.SimpleNamespace(
        waiting_queue=[],
        chunked_req=None,
        running_batch=None,
        ps=types.SimpleNamespace(pp_rank=0, pp_size=3),
    )
    for k, v in kw.items():
        setattr(h, k, v)
    return h


def _guard():
    from sglang.srt.managers.pp_admission_congruence import (
        PPAdmissionCongruenceGuard,
    )

    return PPAdmissionCongruenceGuard(unresolved_defer_cap=CAP)


def _offer(guard, req):
    """What PP0 actually puts on the wire for `req` this pass.

    THE REAL PRODUCER, not a re-derivation. Anything that recomputed the offer
    here would be a second expression for the quantity under test, which is the
    very defect class this file measures.
    """
    from sglang.srt.managers.pp_admission_congruence import (
        build_pp_admission_decision,
    )

    decision = build_pp_admission_decision(
        0,
        [req],
        pp_size=3,
        guard=guard,
        require_executed_geometry=True,
    )
    return decision.entries[0].prefix_len


def _run_terminator(holder, req):
    from sglang.srt.managers.scheduler_pp_mixin import (
        pp_apply_dead_premise_at_chunk_boundary,
        pp_mark_premise_dead,
    )

    pp_mark_premise_dead(req)
    # No `_prefetch_kvcache` on the holder, so the re-fetch cannot be issued
    # and the terminator is what runs -- the metal case, where all three
    # recomputes reported "no re-fetch could be issued".
    return pp_apply_dead_premise_at_chunk_boundary(holder, req)


def _adder_rederives(req, chunk):
    """What `add_chunked_req` does to a continuation on the next pass.

    It "derives everything from `len(req.prefix_indices)` and only THEN calls
    `set_extend_range`" (schedule_policy.py:1348-1421, quoted by the actuator's
    own legality argument). Modelled rather than called because the real one
    needs a `PrefillAdder` with a token pool; what matters here is the ORDER --
    the geometry is a function of the prefix AT ADD TIME -- and that is exactly
    what is reproduced.
    """
    start = len(req.prefix_indices)
    end = min(start + chunk, len(req.full_untruncated_fill_ids))
    req.extend_range = _Range(start, end)


class TheTerminatorMustMoveTheOfferItWasSpentOn(unittest.TestCase):
    """The delivery half. Every assertion reads the PRODUCER's output."""

    def _stuck(self):
        req = _Req(RID, PREFIX, EXTEND)
        guard = _guard()
        holder = _holder(chunked_req=req, _pp_admission_guard=guard)
        # Drive the real defect path: the same geometry re-offered until the
        # lap-free bound escalates. CAP+1 offers is what the log shows ("for 4
        # consecutive passes" at UNRESOLVED_DEFER_CAP=3).
        for _ in range(CAP + 1):
            self.assertEqual(_offer(guard, req), PREFIX)
        return guard, holder, req

    def test_the_escalation_really_arms_on_this_path(self):
        """Reachability first: without this the rest would be vacuous.

        window-951 recorded 0/0 on a line no batch ever reached and read it as
        a pass. A test that never arms the bound proves nothing about a fix to
        the bound.
        """
        guard, _holder_, req = self._stuck()
        self.assertTrue(
            guard.is_escalated(RID),
            "the bound must actually arm on the production (executed) branch, "
            "or every assertion below is about a path this test never entered",
        )
        self.assertEqual(guard.offer_streak(RID), CAP + 1)

    def test_the_terminator_is_actually_spent(self):
        """The half that already worked (#955), pinned so a fix cannot lose it."""
        guard, holder, req = self._stuck()
        self.assertEqual(_run_terminator(holder, req), "recompute")
        self.assertEqual(
            guard.terminator_spent(RID),
            PREFIX,
            "#955: the exit is recorded at the offer it was spent on",
        )
        self.assertEqual(len(req.prefix_indices), 0, "the WRITE happened")

    # ---- EXIT 1: the premise heals and the request continues -------------

    def test_EXIT_1_after_the_adder_rederives_the_offer_finally_MOVES(self):
        """THE DELIVERY. This is the whole ticket.

        Measured on metal: 336 further `told=8192` offers after the recompute.
        The terminator spent a request's entire prefix reuse -- a named double
        prefill, permitted once -- and bought nothing, because the quantity it
        wrote was not the quantity the offer is read from.
        """
        guard, holder, req = self._stuck()
        self.assertEqual(_run_terminator(holder, req), "recompute")
        _adder_rederives(req, EXTEND)
        after = _offer(guard, req)
        self.assertNotEqual(
            after,
            PREFIX,
            "the terminator discarded {} tokens and the next offer is STILL "
            "{} -- the escape cannot reach the offer it exists to move. This "
            "is the 336-offer livelock in one assertion.".format(PREFIX, after),
        )

    def test_EXIT_1_the_offer_reaches_the_UNCONDITIONAL_branch(self):
        """`told <= 0` is admitted whatever the downstream lookup says.

        That branch (pp_admission_congruence.py:1003-1025) is the only exit
        that survives a lookup miss, and a lookup miss is what every retraction
        in the specimen reports (`local=UNKNOWN`). An offer that merely got
        SMALLER would not end the storm; it has to reach zero.
        """
        guard, holder, req = self._stuck()
        self.assertEqual(_run_terminator(holder, req), "recompute")
        _adder_rederives(req, EXTEND)
        self.assertLessEqual(
            _offer(guard, req),
            0,
            "the recompute threw the prefix away, so the honest report is a "
            "zero-length reused prefix -- the one offer "
            "`reconcile_pp_admission_decision` admits without a measurement",
        )

    def test_EXIT_1_the_reported_pair_stays_congruent(self):
        """THE MUTANT GUARD, and it points straight at the forbidden fix.

        A fix that simply rewrote the executed branch's `prefix_len` to 0 would
        satisfy both assertions above and reintroduce instr21: the report would
        name a prefix the rank did not use, against an `extend_len` still sized
        for the old one. `prepare_for_extend` sizes the cross-stage tensor off
        this pair, so an incongruent pair is a SHAPE disagreement on the wire
        -- the #631 crash (512 rows for a batch of 16983 tokens).

        The pair must describe one real pass: the reported prefix is the prefix
        the request actually holds, and the extend is the chunk actually
        scheduled from it.
        """
        from sglang.srt.managers.pp_admission_congruence import (
            build_pp_admission_decision,
        )

        guard, holder, req = self._stuck()
        self.assertEqual(_run_terminator(holder, req), "recompute")
        _adder_rederives(req, EXTEND)
        entry = build_pp_admission_decision(
            0,
            [req],
            pp_size=3,
            guard=guard,
            require_executed_geometry=True,
        ).entries[0]
        self.assertEqual(
            entry.prefix_len,
            len(req.prefix_indices),
            "the reported prefix must be the prefix the request actually "
            "holds -- reporting anything else names a pass no rank ran "
            "(instr21)",
        )
        self.assertEqual(
            entry.prefix_len,
            req.extend_range.start,
            "the two expressions the module calls ONE quantity must agree at "
            "the point the report is made",
        )
        self.assertEqual(
            entry.extend_len,
            req.extend_range.end - req.extend_range.start,
            "the extend must be the chunk the adder scheduled, not a remainder "
            "this pass will not run",
        )

    # ---- EXIT 3: the named fail, when the geometry was NOT re-derived ----

    def test_EXIT_3_a_stale_geometry_can_no_longer_be_REPORTED(self):
        """The pathological ordering gets a loud, rid-naming refusal.

        If the adder did not re-derive the geometry, the request is in the
        defect state itself. Before #958 that state produced a stale `told=8192`
        indistinguishable from a fresh measurement, forever. Now the existing
        `require_executed_geometry` refusal names the rid on the FIRST pass.

        This is exit 3, not a regression: one loud line beats 336 silent ones,
        and the refusal is the mechanism `reset_for_retract` already relies on.
        """
        from sglang.srt.managers.pp_admission_congruence import PPScheduleRefused

        guard, holder, req = self._stuck()
        self.assertEqual(_run_terminator(holder, req), "recompute")
        with self.assertRaises(PPScheduleRefused) as caught:
            _offer(guard, req)
        self.assertIn(RID, str(caught.exception), "the refusal must NAME the request")

    def test_the_FOURTH_exit_is_gone(self):
        """Silent unbounded re-offer of a geometry the request no longer has.

        The one outcome that must be unreachable. Whatever happens after the
        terminator, `PREFIX` must never be offered again: either the adder
        re-derives (exit 1, offer 0) or the refusal fires (exit 3). There is no
        third way to reach the producer.
        """
        guard, holder, req = self._stuck()
        self.assertEqual(_run_terminator(holder, req), "recompute")
        try:
            after = _offer(guard, req)
        except Exception:
            return  # exit 3, loud and named -- asserted in detail above
        self.assertNotEqual(
            after, PREFIX, "the stale offer came back -- this is the livelock"
        )


class TheInvariantTheActingHalfBreaks(unittest.TestCase):
    """THE CLASS, not just this instance.

    `_executed_extent`'s docstring rests on `extend_range.start ==
    len(prefix_indices)` and calls it "exactly one expression". Any operation
    that writes one side without the other turns the report into a stale
    reading. The terminator is the instance that cost a boot; the invariant is
    the thing that must hold for every such operation, including ones written
    after this ticket.
    """

    def _reported_start(self, req):
        """What the producer would put on the wire for `req` right now."""
        from sglang.srt.managers.pp_admission_congruence import _executed_extent

        extent = _executed_extent(req)
        return None if extent is None else extent[0]

    def test_no_stale_start_survives_a_truncation(self):
        """THE INVARIANT, stated so it outlives this particular fix.

        Either the geometry agrees with the prefix it was derived from, or
        there is no geometry. What must never exist is a geometry that
        disagrees, because `_executed_extent` cannot tell a stale reading from
        a fresh measurement and reports it as the latter.
        """
        for told in (0, 1, PREFIX // 2, PREFIX - 1):
            with self.subTest(told=told):
                req = _Req(RID, PREFIX, EXTEND)
                self.assertEqual(self._reported_start(req), PREFIX)
                req.truncate_prefix_to(told)
                start = self._reported_start(req)
                self.assertTrue(
                    start is None or start == len(req.prefix_indices),
                    "after truncating to {} the producer still reports "
                    "start={} against a prefix of {} -- a stale geometry is "
                    "readable as a measurement".format(
                        told, start, len(req.prefix_indices)
                    ),
                )

    def test_the_pre_truncation_length_is_never_reported_again(self):
        """The specific shape that cost the boot: the OLD value coming back."""
        req = _Req(RID, PREFIX, EXTEND)
        req.truncate_prefix_to(0)
        self.assertNotEqual(
            self._reported_start(req),
            PREFIX,
            "the discarded prefix length is still what the offer would carry",
        )

    def test_CANFAIL_a_NO_OP_truncation_keeps_a_healthy_geometry(self):
        """The other direction, so the fix cannot pay with healthy passes.

        `told >= len(prefix_indices)` changes nothing, so the geometry is still
        an accurate report and must survive. Clearing it here would void good
        passes and turn a livelock fix into a throughput defect.
        """
        req = _Req(RID, PREFIX, EXTEND)
        req.truncate_prefix_to(PREFIX)
        self.assertIsNotNone(
            req.extend_range,
            "a truncation that moved nothing must not invalidate anything",
        )
        self.assertEqual(self._reported_start(req), PREFIX)


if __name__ == "__main__":
    unittest.main()
