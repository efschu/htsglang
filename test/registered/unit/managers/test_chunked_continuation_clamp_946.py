"""#946: the legal clamp point for a chunked continuation, and the fourth
instance of the compensator-reachability class.

THE STRUCTURAL FACT THAT EXPLAINS TWO DEFECTS AT ONCE. `scheduler.py`'s #943b
re-issue built its candidate set as `{r.rid: r for r in self.waiting_queue}`.
A chunked continuation is NOT in the waiting queue -- #797b deliberately
RESTORES it as `self.chunked_req`, because that state outlives the round and
putting it back is what stopped boot instr19 dying on `extend_range.end`. So:

  * the re-issue could never nominate the request that was stuck
    (measured: `PREFETCH RE-ISSUED` 0 across six boots), and
  * the request never re-entered through the waiting-queue path where
    `prefix_len_for` already applies, which is why the offer froze at
    `told=8192` for thousands of passes.

ONE fact, both symptoms. That is the fourth instance of the class this ticket
family is about: a compensator sitting off the path that produces the defect.

WHERE THE CLAMP IS LEGAL, and why it is not where the decision is made.
`add_chunked_req` (`schedule_policy.py`) derives everything from
`len(req.prefix_indices)` and only THEN calls `set_extend_range`. Before that
call no geometry for the next chunk exists, so changing the premise there
cannot contradict a report -- the report has not been made. `#906 / #890 hole
2` already gates that exact site and its own comment establishes the safety:
the request stays `self.chunked_req`, is not dropped, and resumes mid-chunk
unharmed.

The VOID is where the group already agrees (`pp_pass_should_void` ORs the
incoming flag and never clears it), but it fires AFTER the pass was committed
and launched -- changing the premise there rewrites a pass in flight, which is
the instr20 crash. So: DECIDE at the void, ACT at the chunk boundary.

THE MARK IS STAMPED WITH THE BINDING GENERATION and is valid only on a match.
That makes it survive the #797b chunked restore (same generation -- the whole
point) and invalidate ITSELF on a retract or cutover (the generation moves, the
reader sees a stale stamp and re-derives). No `reset_for_retract` special case,
no stale state after a premise change -- the same rule #911 uses for completion
routing.

KEIN DOPPEL-PREFILL. `told=0` on the observed rid discards 8192 tokens -- two
full `--chunked-prefill-size 4096` chunks -- against a user law that allows at
most ONE chunk of loss. So the escape is a RE-FETCH under the current
generation wherever one can be issued, and `told=0` is a bounded last resort
that must name the number of tokens it throws away. A law breach may be
necessary; it may never be silent.
"""

import logging
import types
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20)

RID_CHUNKED = "rid-chunked-continuation"
RID_WAITING = "rid-in-the-waiting-queue"
RID_RUNNING = "rid-in-the-running-batch"
RID_SLOT = "rid-in-the-slots-chunked-req"


class _Range:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class _Req:
    def __init__(self, rid, prefix_len=0, extend_len=0):
        self.rid = rid
        self.prefix_indices = list(range(prefix_len))
        self.extend_range = _Range(prefix_len, prefix_len + extend_len)
        self.full_untruncated_fill_ids = list(range(prefix_len + extend_len))
        self.kv_committed_len = prefix_len
        self.cache_protected_len = prefix_len

    def truncate_prefix_to(self, told):
        self.prefix_indices = self.prefix_indices[:told]
        self.cache_protected_len = min(self.cache_protected_len, told)


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


class TheReissueCandidateSetMustSeeAChunkedContinuation(unittest.TestCase):
    """ARM 1 -- THE REACHABILITY ARM, and the one that must be red first.

    This is the #946 twin of the #944c ratchet: it asks whether the compensator
    can even SEE the request that is stuck, rather than whether it behaves
    correctly once handed one.
    """

    def _locations(self, holder):
        from sglang.srt.managers.scheduler_pp_mixin import pp_request_locations

        return pp_request_locations(holder)

    def test_a_chunked_continuation_is_a_candidate(self):
        req = _Req(RID_CHUNKED, prefix_len=8192, extend_len=4096)
        got = self._locations(_holder(chunked_req=req))
        self.assertIn(
            RID_CHUNKED,
            got,
            "THE #946 DEFECT: the re-issue's candidate set was "
            "`{r.rid: r for r in self.waiting_queue}`, and a chunked "
            "continuation is never in the waiting queue (#797b restores it as "
            "`self.chunked_req`). So the one request that was stuck could "
            "never be nominated -- measured as PREFETCH RE-ISSUED 0 over six "
            "boots.",
        )
        self.assertIs(got[RID_CHUNKED], req)

    def test_all_four_places_a_request_can_live_are_covered(self):
        """The SAME enumeration #944 built for the reconcile lookup. Sharing
        the list is the point: a fifth place must not have to be discovered
        twice, once per consumer."""
        waiting = _Req(RID_WAITING)
        chunked = _Req(RID_CHUNKED)
        slot = _Req(RID_SLOT)
        running = _Req(RID_RUNNING)
        holder = _holder(
            waiting_queue=[waiting],
            chunked_req=chunked,
            running_batch=types.SimpleNamespace(reqs=[running]),
        )
        holder._pp_chunked_req_before_by_slot = {0: slot}
        got = self._locations(holder)
        for rid in (RID_WAITING, RID_CHUNKED, RID_SLOT, RID_RUNNING):
            self.assertIn(rid, got, f"{rid} is one of the four places")

    def test_an_empty_scheduler_yields_an_empty_set_and_never_raises(self):
        """A holder with nothing anywhere is the idle pass, and the candidate
        set feeds a COLLECTIVE (`take_agreed_reissue`) -- a raise here would
        take the vote down with it."""
        self.assertEqual(self._locations(_holder()), {})

    def test_it_tolerates_a_holder_that_carries_none_of_the_fields(self):
        """Every lookup is a getattr for the #787 reason: a stand-in holder, a
        rank with no slot array, and `pp_size <= 1` must all read as 'nothing
        known' rather than AttributeError inside a collective's candidate
        construction."""
        self.assertEqual(self._locations(types.SimpleNamespace()), {})


class TheEscalationMarkIsStampedWithTheBindingGeneration(unittest.TestCase):
    """ARM 2 -- DECIDE AT THE VOID, and make the decision self-invalidating."""

    def _mark(self, req):
        from sglang.srt.managers.scheduler_pp_mixin import (
            pp_mark_premise_dead,
        )

        return pp_mark_premise_dead(req)

    def _is_live(self, req):
        from sglang.srt.managers.scheduler_pp_mixin import (
            pp_premise_is_dead,
        )

        return pp_premise_is_dead(req)

    def test_a_fresh_request_has_no_dead_premise(self):
        self.assertFalse(self._is_live(_Req(RID_CHUNKED)))

    def test_the_mark_survives_a_chunked_restore_within_one_generation(self):
        """#797b restores the SAME object, so the mark rides with it. That is
        the whole reason it lives on the request rather than in a guard-side
        dict keyed by rid: the dict would be consulted by a rank that never
        saw the void."""
        req = _Req(RID_CHUNKED, prefix_len=8192)
        self._mark(req)
        self.assertTrue(self._is_live(req))

    def test_the_mark_invalidates_ITSELF_when_the_generation_moves(self):
        """A retract or a cutover moves the binding generation, and a premise
        recorded under the old one is exactly the stale state that must not
        outlive it. No `reset_for_retract` special case -- the stamp does it.
        Same rule #911 uses for completion routing."""
        from sglang.srt.mem_cache import hicache_phase_binding as hpb

        req = _Req(RID_CHUNKED, prefix_len=8192)
        self._mark(req)
        self.assertTrue(self._is_live(req))
        real = hpb.current_generation
        try:
            hpb.current_generation = lambda: real() + 1
            self.assertFalse(
                self._is_live(req),
                "a premise marked under a superseded binding must read as "
                "absent, not as live -- that is the difference between a "
                "self-invalidating stamp and a leak",
            )
        finally:
            hpb.current_generation = real


class TheEscapePrefersARefetchOverARecompute(unittest.TestCase):
    """ARM 3 -- THE LAW ARM. Kein-Doppel-Prefill is a user law, and `told=0`
    on an 8192-token prefix breaks it by two full chunks."""

    def _apply(self, holder, req):
        from sglang.srt.managers.scheduler_pp_mixin import (
            pp_apply_dead_premise_at_chunk_boundary,
        )

        return pp_apply_dead_premise_at_chunk_boundary(holder, req)

    def _marked(self, req):
        from sglang.srt.managers.scheduler_pp_mixin import pp_mark_premise_dead

        pp_mark_premise_dead(req)
        return req

    def test_an_unmarked_request_is_left_completely_alone(self):
        """THE DEFAULT PATH IS UNTOUCHED, and this is the assertion that says
        so: same object, same prefix, no action taken."""
        req = _Req(RID_CHUNKED, prefix_len=8192, extend_len=4096)
        holder = _holder(chunked_req=req)
        self.assertEqual(self._apply(holder, req), "none")
        self.assertEqual(len(req.prefix_indices), 8192)

    def test_a_refetch_is_preferred_and_the_prefix_is_KEPT(self):
        """The store still holds the pages; re-fetching under the current
        generation costs a fetch, whereas `told=0` costs a full recompute of
        8192 tokens. The law allows one chunk of loss, so the fetch is not
        merely nicer -- it is the only compliant answer when it is available."""
        req = self._marked(_Req(RID_CHUNKED, prefix_len=8192, extend_len=4096))
        called = []
        holder = _holder(chunked_req=req)
        holder._prefetch_kvcache = lambda r: called.append(r.rid)
        self.assertEqual(self._apply(holder, req), "refetch")
        self.assertEqual(called, [RID_CHUNKED])
        self.assertEqual(
            len(req.prefix_indices),
            8192,
            "a re-fetch keeps the premise -- discarding it as well would pay "
            "the recompute AND the fetch",
        )

    def test_told_zero_is_the_LAST_RESORT_and_names_what_it_discards(self):
        """When no re-fetch can be issued the terminator still applies, and it
        must say how many tokens it threw away. A law breach may be necessary;
        it may never be silent."""
        req = self._marked(_Req(RID_CHUNKED, prefix_len=8192, extend_len=4096))
        holder = _holder(chunked_req=req)  # no _prefetch_kvcache available
        catcher = _Catcher(logging.WARNING)
        log = logging.getLogger("sglang.srt.managers.scheduler_pp_mixin")
        log.addHandler(catcher)
        try:
            self.assertEqual(self._apply(holder, req), "recompute")
        finally:
            log.removeHandler(catcher)
        self.assertEqual(len(req.prefix_indices), 0, "the terminator applied")
        self.assertTrue(catcher.records, "the discard must be logged")
        msg = catcher.messages[0]
        self.assertIn("8192", msg, f"the discarded token count must be named: {msg}")
        self.assertIn("#946", msg)

    def test_the_mark_is_consumed_so_the_escape_fires_once(self):
        req = self._marked(_Req(RID_CHUNKED, prefix_len=8192, extend_len=4096))
        holder = _holder(chunked_req=req)
        holder._prefetch_kvcache = lambda r: None
        self.assertEqual(self._apply(holder, req), "refetch")
        self.assertEqual(
            self._apply(holder, req),
            "none",
            "a consumed premise must not re-fire every chunk -- that would be "
            "the 2106-line log in a new colour",
        )


class _Catcher(logging.Handler):
    def __init__(self, level):
        super().__init__(level=level)
        self.records = []

    def emit(self, record):
        self.records.append(record)

    @property
    def messages(self):
        return [r.getMessage() for r in self.records]


if __name__ == "__main__":
    unittest.main()


class TheRingInstrumentMustNotLIE(unittest.TestCase):
    """#947 THE INDIKATOR-GESETZ ARM. An indicator is not a finding until it is
    shown to measure what it claims.

    This family has now spent five boots discovering that a compensator sat on
    a path the failure starves. The answer was to MEASURE the ring instead of
    reasoning about it -- and a measurement that cannot itself be falsified
    would just move the same mistake one level up. So: the markers must fire
    under a constructed void cycle, and must NOT report voids during a healthy
    run.
    """

    def _note(self, holder, site, voided):
        from sglang.srt.managers.scheduler_pp_mixin import pp_ring_note

        return pp_ring_note(holder, site, voided)

    def test_a_void_cycle_is_counted_as_a_RUN_not_as_a_rate(self):
        """The window-946 evidence could only give a quotient (9471/6). A
        quotient cannot tell a steady 1578 from one burst of 9000 -- and the
        fix differs between those shapes, so the instrument records RUNS."""
        h = _holder()
        for _ in range(7):
            self._note(h, "ring:pre_plan", True)
        self._note(h, "ring:pre_plan", False)
        for _ in range(3):
            self._note(h, "ring:pre_plan", True)
        self._note(h, "ring:pre_plan", False)
        self.assertEqual(
            h._pp_ring_void_runs,
            [7, 3],
            "two void runs of 7 and 3 -- not a mean of 5, which is what a rate "
            "would have reported and what would have hidden the burst",
        )

    def test_a_healthy_run_reports_NO_voids(self):
        """THE CAN-LIE DIRECTION. If the instrument counted iterations rather
        than voids it would look identical on a healthy boot, and every future
        reading would be worthless."""
        h = _holder()
        for _ in range(20):
            self._note(h, "ring:pre_plan", False)
        self.assertEqual(getattr(h, "_pp_ring_void_runs", []), [])
        self.assertEqual(h._pp_ring_voids_run, 0)
        self.assertEqual(h._pp_ring_admissions, 20)

    def test_a_site_that_never_runs_is_ABSENT_from_the_map(self):
        """The load-bearing claim: 'a site absent from that map does not run
        under this failure'. It only holds if absence is real absence."""
        h = _holder()
        for _ in range(5):
            self._note(h, "ring:pre_plan", True)
        self.assertIn("ring:pre_plan", h._pp_ring_site_counts)
        self.assertNotIn(
            "prefill:chunked_block",
            h._pp_ring_site_counts,
            "a site that was never reached must not appear at all -- a zero "
            "entry and an absent entry must not be confusable",
        )

    def test_the_census_is_rate_limited_and_never_one_line_per_pass(self):
        """9471 log lines is the defect this is investigating. The census may
        not become it."""
        import os

        h = _holder()
        catcher = _Catcher(logging.WARNING)
        log = logging.getLogger("sglang.srt.managers.scheduler_pp_mixin")
        log.addHandler(catcher)
        old = os.environ.get("SGLANG_947_RING_EVERY")
        os.environ["SGLANG_947_RING_EVERY"] = "25"
        try:
            for _ in range(60):
                self._note(h, "ring:pre_plan", True)
                self._note(h, "ring:pre_plan", False)
        finally:
            log.removeHandler(catcher)
            if old is None:
                os.environ.pop("SGLANG_947_RING_EVERY", None)
            else:
                os.environ["SGLANG_947_RING_EVERY"] = old
        self.assertEqual(
            len(catcher.records),
            2,
            f"60 admissions at every=25 must emit exactly 2 lines, got "
            f"{len(catcher.records)}",
        )
        self.assertIn("#947 VOID-RING CENSUS", catcher.messages[0])

    def test_the_relocated_actuator_is_OFF_unless_the_env_says_otherwise(self):
        """DEFAULT-OFF IS THE DISCIPLINE, not a convenience. Five placements
        were shipped armed and each cost a boot; this one measures first."""
        import os

        from sglang.srt.managers.scheduler_pp_mixin import pp_act_at_ring_enabled

        old = os.environ.get("SGLANG_946_ACT_AT_RING")
        try:
            os.environ.pop("SGLANG_946_ACT_AT_RING", None)
            self.assertFalse(pp_act_at_ring_enabled(), "must ship inert")
            os.environ["SGLANG_946_ACT_AT_RING"] = "1"
            self.assertTrue(pp_act_at_ring_enabled(), "and must be flippable")
            os.environ["SGLANG_946_ACT_AT_RING"] = "0"
            self.assertFalse(pp_act_at_ring_enabled())
        finally:
            if old is None:
                os.environ.pop("SGLANG_946_ACT_AT_RING", None)
            else:
                os.environ["SGLANG_946_ACT_AT_RING"] = old
