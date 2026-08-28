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
        # #971 THE MASKING FIXTURE, corrected. This was `{0: slot}` -- a dict,
        # the one container `pp_request_locations`' `.values()` lookup could
        # read. Production builds this ring as a LIST (`init_pp_loop_state`:
        # `[None] * pp_loop_size`), on which that lookup raised AttributeError
        # and was swallowed as "a stand-in that is not a mapping". So the
        # fourth place this test claims to cover was, in production, empty for
        # every consumer -- and the fixture was the reason nobody could see
        # it. The test now drives the shape the product has.
        holder._pp_chunked_req_before_by_slot = [slot]
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
        # CONTRACT INVERTED 2026-08-27 (#949), reasoning recorded at the site
        # like the four inversions before it. This stub used to return None --
        # `called.append(...)` evaluates to None -- and the escape reported
        # "refetch" for it. That is EXACTLY the defect window-946fix-0828
        # measured on metal: the escape fired 8885 times, reported success
        # every time, and issued nothing. The test was green while the shipped
        # code was blind, because the test supplied the same empty answer the
        # production path did. A stub must now say what really happened.
        req = self._marked(_Req(RID_CHUNKED, prefix_len=8192, extend_len=4096))
        called = []

        def _issue(r):
            called.append(r.rid)
            return "issued"

        holder = _holder(chunked_req=req)
        holder._prefetch_kvcache = _issue
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

    def test_the_mark_is_consumed_by_an_ISSUED_refetch_so_it_fires_once(self):
        # CONTRACT INVERTED 2026-08-27 (#949). The old spelling used a stub
        # returning None and asserted the mark was consumed anyway. Under the
        # corrected contract the mark is spent by an OUTCOME, not by an
        # attempt: an issued re-fetch consumes it, a decline keeps it. The
        # anti-re-fire guarantee this test exists for is unchanged and is still
        # asserted -- only the thing that earns it moved.
        req = self._marked(_Req(RID_CHUNKED, prefix_len=8192, extend_len=4096))
        holder = _holder(chunked_req=req)
        holder._prefetch_kvcache = lambda r: "issued"
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


class TheThreeCandidatesMustBeSEPARABLE(unittest.TestCase):
    """#948: window-947 left three explanations standing and could not choose.
    Each gets ONE counter, and each counter gets a CAN-FAIL arm -- a constructed
    case that makes exactly that counter non-zero and the other two absent.

    Without the can-fail arms these counters would be three more indicators
    nobody had shown could move, which is the failure mode this whole family is
    a list of.
    """

    def _probe(self, holder, kind, **f):
        from sglang.srt.managers.scheduler_pp_mixin import pp_premise_probe

        return pp_premise_probe(holder, kind, **f)

    def _counts(self, holder):
        return getattr(holder, "_pp_premise_probe_counts", {})

    def test_ABSENT_is_not_zero(self):
        """The load-bearing semantics: a kind that never fired must not appear.
        'Missing' and 'measured zero' must never be confusable, or a null
        reading proves nothing."""
        h = _holder()
        self._probe(h, "mark_hit", rid="r")
        self.assertIn("mark_hit", self._counts(h))
        self.assertNotIn("mark_miss", self._counts(h))
        self.assertNotIn("act_rid_mismatch", self._counts(h))
        self.assertNotIn("gen_mismatch", self._counts(h))

    def test_canfail_a_mark_miss_moves_only_its_own_counter(self):
        h = _holder()
        self._probe(h, "mark_miss", rid="r", list_size=0)
        self.assertEqual(self._counts(h), {"mark_miss": 1})

    def test_canfail_b_act_rid_mismatch_moves_only_its_own_counter(self):
        h = _holder()
        self._probe(h, "act_rid_mismatch", chunked_rid=None, marked_rids=["r"])
        self.assertEqual(self._counts(h), {"act_rid_mismatch": 1})

    def test_canfail_c_gen_mismatch_carries_BOTH_values(self):
        """'They differ' without the numbers cannot tell a cutover from an
        unreadable stamp, so the sample must carry both."""
        h = _holder()
        self._probe(h, "gen_mismatch", rid="r", stamped=7, now=9)
        self.assertEqual(self._counts(h), {"gen_mismatch": 1})
        s = h._pp_premise_probe_samples["gen_mismatch"]
        self.assertEqual((s["stamped"], s["now"]), (7, 9))

    def test_the_first_sample_is_kept_and_not_overwritten(self):
        h = _holder()
        self._probe(h, "gen_mismatch", rid="first", stamped=1, now=2)
        self._probe(h, "gen_mismatch", rid="second", stamped=3, now=4)
        self.assertEqual(self._counts(h)["gen_mismatch"], 2)
        self.assertEqual(h._pp_premise_probe_samples["gen_mismatch"]["rid"], "first")

    def test_the_gen_mismatch_counter_fires_from_the_REAL_act_path(self):
        """Not a hand-called probe: drive the shipped actuator with a stamp
        from a superseded generation and require counter (c) to move."""
        from sglang.srt.managers.scheduler_pp_mixin import (
            _PREMISE_DEAD_STAMP,
            pp_apply_dead_premise_at_chunk_boundary,
        )
        from sglang.srt.mem_cache import hicache_phase_binding as hpb

        req = _Req(RID_CHUNKED, prefix_len=8192, extend_len=4096)
        setattr(req, _PREMISE_DEAD_STAMP, int(hpb.current_generation()) - 1)
        h = _holder(chunked_req=req)
        self.assertEqual(pp_apply_dead_premise_at_chunk_boundary(h, req), "none")
        self.assertIn("gen_mismatch", self._counts(h))

    def test_THE_SEPARATION_a_marked_request_that_is_not_the_chunked_req(self):
        """THE LIVE SHAPE, reproduced hermetically.

        The mark is written over `can_run_list`; the act reads ONE field
        (`self.chunked_req`). A request marked anywhere else is marked forever
        and inspected never -- which is candidate (b), and it needs no metal to
        demonstrate: it is a property of the two call sites.
        """
        from sglang.srt.managers.scheduler_pp_mixin import (
            pp_apply_dead_premise_at_chunk_boundary,
            pp_mark_premise_dead,
            pp_request_locations,
        )

        stuck = _Req(RID_RUNNING, prefix_len=8192, extend_len=4096)
        other = _Req(RID_CHUNKED, prefix_len=16, extend_len=16)
        h = _holder(
            chunked_req=other,
            running_batch=types.SimpleNamespace(reqs=[stuck]),
        )
        pp_mark_premise_dead(stuck)

        self.assertIn(
            RID_RUNNING,
            pp_request_locations(h),
            "the marked request IS findable -- the four-place enumeration sees "
            "it, so this is not an absence",
        )
        self.assertEqual(
            pp_apply_dead_premise_at_chunk_boundary(h, h.chunked_req),
            "none",
            "...and the act still does nothing, because it inspects one field "
            "instead of the places the request can live. THAT is why the "
            "armed actuator never fired on metal.",
        )
        self.assertEqual(
            pp_apply_dead_premise_at_chunk_boundary(h, stuck),
            "recompute",
            "handed the request it was marked on, the same actuator acts -- so "
            "the actuator is correct and its ARGUMENT was wrong",
        )


class TheActMustSweepTheFourPlacesNotOneField(unittest.TestCase):
    """#948 THE FIX, red first.

    Desk verdict from the separation above: the actuator is correct and its
    ARGUMENT was wrong. The mark is written over `can_run_list`; the act read
    `self.chunked_req`. A request marked anywhere else was marked forever and
    inspected never -- which is exactly why the armed actuator produced
    `PREMISE RECOMPUTE` 0 on metal while `UNRESOLVABLE` fired on the right rid.

    The enumeration to sweep already exists: `pp_request_locations`, built for
    #946's candidate set. The fix is to use it here too -- the same list, a
    second consumer, which is the whole reason it was factored out rather than
    inlined.
    """

    def _sweep(self, holder):
        from sglang.srt.managers.scheduler_pp_mixin import (
            pp_apply_dead_premise_anywhere,
        )

        return pp_apply_dead_premise_anywhere(holder)

    def test_it_acts_on_a_marked_request_in_the_RUNNING_BATCH(self):
        from sglang.srt.managers.scheduler_pp_mixin import pp_mark_premise_dead

        stuck = _Req(RID_RUNNING, prefix_len=8192, extend_len=4096)
        h = _holder(
            chunked_req=_Req(RID_CHUNKED, prefix_len=16),
            running_batch=types.SimpleNamespace(reqs=[stuck]),
        )
        pp_mark_premise_dead(stuck)
        self.assertEqual(self._sweep(h), {RID_RUNNING: "recompute"})
        self.assertEqual(len(stuck.prefix_indices), 0, "the terminator applied")

    def test_it_acts_on_a_marked_request_in_the_WAITING_QUEUE(self):
        from sglang.srt.managers.scheduler_pp_mixin import pp_mark_premise_dead

        # CONTRACT INVERTED 2026-08-27 (#949): the stub must now report what
        # actually happened. `lambda r: None` used to read as a successful
        # re-fetch, which is the defect this ticket closes.
        stuck = _Req(RID_WAITING, prefix_len=8192, extend_len=4096)
        h = _holder(waiting_queue=[stuck])
        h._prefetch_kvcache = lambda r: "issued"
        pp_mark_premise_dead(stuck)
        self.assertEqual(self._sweep(h), {RID_WAITING: "refetch"})

    def test_it_still_prefers_the_refetch_and_keeps_the_prefix(self):
        from sglang.srt.managers.scheduler_pp_mixin import pp_mark_premise_dead

        stuck = _Req(RID_RUNNING, prefix_len=8192, extend_len=4096)
        seen = []
        h = _holder(running_batch=types.SimpleNamespace(reqs=[stuck]))

        def _issue(r):
            seen.append(r.rid)
            return "issued"  # #949: report the outcome, never nothing

        h._prefetch_kvcache = _issue
        pp_mark_premise_dead(stuck)
        self.assertEqual(self._sweep(h), {RID_RUNNING: "refetch"})
        self.assertEqual(seen, [RID_RUNNING])
        self.assertEqual(
            len(stuck.prefix_indices),
            8192,
            "Kein-Doppel-Prefill: a re-fetch must KEEP the premise",
        )

    def test_an_unmarked_scheduler_is_a_complete_no_op(self):
        """THE DEFAULT PATH. Nothing marked anywhere -> empty result, nothing
        touched, and no cost beyond one walk of the four places."""
        h = _holder(
            waiting_queue=[_Req("a", 8)],
            chunked_req=_Req("b", 8),
            running_batch=types.SimpleNamespace(reqs=[_Req("c", 8)]),
        )
        self.assertEqual(self._sweep(h), {})

    def test_each_marked_request_is_acted_on_ONCE(self):
        from sglang.srt.managers.scheduler_pp_mixin import pp_mark_premise_dead

        stuck = _Req(RID_RUNNING, prefix_len=8192, extend_len=4096)
        h = _holder(running_batch=types.SimpleNamespace(reqs=[stuck]))
        pp_mark_premise_dead(stuck)
        self.assertEqual(self._sweep(h), {RID_RUNNING: "recompute"})
        self.assertEqual(
            self._sweep(h), {}, "the mark is consumed -- no re-firing per pass"
        )


class _FakeTreeCache:
    """A tree cache whose ONLY honest signal is the ongoing set.

    Deliberately minimal: the verdict under test must be derived from the
    EFFECT (did an operation get registered?) and not from anything this stub
    says about itself.
    """

    def __init__(self, register: bool = True, gate_reason=None):
        self.ongoing_prefetch = {}
        self._register = register
        self._gate_reason = gate_reason
        self.calls = []
        self.hicache_storage_pass_prefix_keys = False
        self.root_node = object()

    def prefetch_from_storage(self, req_id, *a, **kw):
        self.calls.append(req_id)
        if self._gate_reason is not None:
            from sglang.srt.mem_cache.match_refusal_census import note_prefetch_gate

            note_prefetch_gate(self._gate_reason, 4096)
        if self._register:
            self.ongoing_prefetch[req_id] = object()


class TheRefetchVerdictMustBeAnOBSERVATIONNotAPromise(unittest.TestCase):
    """#949 ARM 1 -- THE ONE WINDOW-946FIX PAID FOR.

    The measured failure: `_prefetch_kvcache` returned None on all six of its
    silent exits, the escape read that as success, and the boot showed 8885
    escapes reporting "refetch" while `PREFETCH RE-ISSUED` stayed 0, `cached>0`
    stayed 0 and the rid stayed frozen at told=8192. The compensator was, for
    the first time in this family, provably ON the defect's path -- and its
    SUCCESS VALUE WAS NOT EVIDENCE THAT IT ACTED.

    THIS ARM IS DELIBERATELY NOT BEHIND `SGLANG_946_ACT_AT_RING`. My own
    pre-boot finding this window was that the #948 discriminator counters sat
    inside the actuator's env gate, so the stage that was supposed to read them
    could not. An indicator must never sit behind the gate whose state it
    measures, and this class is written to that lesson: it drives
    `_prefetch_kvcache` and the actuator directly, with no env flag involved.
    """

    def _scheduler_with(self, tree_cache, enable=True):
        from sglang.srt.managers.scheduler import Scheduler

        sched = types.SimpleNamespace()
        sched.enable_hicache_storage = enable
        sched.tree_cache = tree_cache
        sched._prefetch_kvcache = types.MethodType(Scheduler._prefetch_kvcache, sched)
        return sched

    def _req(self):
        req = _Req(RID_CHUNKED, prefix_len=8192, extend_len=4096)
        req.init_next_round_input = lambda *a, **kw: None
        req.last_host_node = types.SimpleNamespace(
            backuped=True,
            get_last_hash_value=lambda: "h",
            get_prefix_hash_values=lambda parent: None,
            parent=None,
        )
        req.host_hit_length = 0
        req._compute_max_prefix_len = lambda n: n
        return req

    def test_an_ISSUED_refetch_is_proven_by_the_ONGOING_SET_not_the_return(self):
        """DELIVERY PROOF. "issued" is only allowed to be returned when the
        operation actually appears in `ongoing_prefetch` -- the registration at
        unified_radix_cache.py:2996 that is the one event meaning a prefetch
        exists. A return value alone is exactly what failed on metal."""
        tc = _FakeTreeCache(register=True)
        sched = self._scheduler_with(tc)
        req = self._req()
        self.assertEqual(sched._prefetch_kvcache(req), "issued")
        self.assertIn(
            req.rid,
            tc.ongoing_prefetch,
            "the verdict must be backed by a real registration, not asserted",
        )

    def test_CANFAIL_a_call_that_registers_NOTHING_is_never_issued(self):
        """THE CAN-FAIL ARM, and it is the mutant that shipped. A
        `prefetch_from_storage` that runs, raises nothing and registers nothing
        is the shape of every one of the six silent exits. The old code called
        this success."""
        tc = _FakeTreeCache(register=False)
        sched = self._scheduler_with(tc)
        verdict = sched._prefetch_kvcache(self._req())
        self.assertTrue(
            verdict.startswith("declined:"),
            f"a call that registered nothing must decline, got {verdict!r}",
        )
        self.assertTrue(tc.calls, "and it must genuinely have reached the call")

    def test_each_silent_exit_declines_WITH_ITS_OWN_REASON(self):
        """#767's lesson -- NAME THE EATEN BRANCH. A decline that cannot say
        which term produced it sends the next reader to re-argue it, which is
        how #915's three verdicts wore one boolean for twelve days."""
        sched = self._scheduler_with(_FakeTreeCache(), enable=False)
        self.assertEqual(
            sched._prefetch_kvcache(self._req()), "declined:storage_disabled"
        )

        tc = _FakeTreeCache(register=False)
        sched = self._scheduler_with(tc)
        req = self._req()
        req.last_host_node.backuped = False
        self.assertEqual(sched._prefetch_kvcache(req), "declined:anchor_no_vote")
        self.assertEqual(tc.calls, [], "an anchor decline must not reach the call")

    def test_the_915_RATE_LIMITED_reason_reaches_the_verdict(self):
        """CANDIDATE (iii) MADE READABLE. The coordinator's structural
        suspicion -- the host-tier capacity asymmetry across the flip, where
        `prefetch_capacity_limit` is `0.5 * mem_pool_host.size`
        (cache_controller.py:841) and #905 measured 703472 PP rows against
        30518 TP rows, a 23x gap -- can now ARRIVE at the escape as a named
        reason instead of being re-derived from a comment."""
        from sglang.srt.mem_cache import match_refusal_census as mrc

        before = dict(mrc.PREFETCH_GATE_COUNTS)
        try:
            tc = _FakeTreeCache(register=False, gate_reason="rate_limited")
            sched = self._scheduler_with(tc)
            self.assertEqual(
                sched._prefetch_kvcache(self._req()), "declined:rate_limited"
            )
        finally:
            mrc.PREFETCH_GATE_COUNTS.clear()
            mrc.PREFETCH_GATE_COUNTS.update(before)


class ADeclinedRefetchKeepsTheMarkAndIsBounded(unittest.TestCase):
    """#949 ARM 2 -- THE CONSEQUENCE, and it is where the old code leaked.

    The mark used to be cleared BEFORE the attempt, so a declined re-fetch
    spent it for nothing and the request never saw the escape again. Invisible,
    because the decline reported success. The corrected rule: an outcome spends
    the mark, an attempt does not.
    """

    def _apply(self, holder, req):
        from sglang.srt.managers.scheduler_pp_mixin import (
            pp_apply_dead_premise_at_chunk_boundary,
        )

        return pp_apply_dead_premise_at_chunk_boundary(holder, req)

    def _marked(self, req):
        from sglang.srt.managers.scheduler_pp_mixin import pp_mark_premise_dead

        pp_mark_premise_dead(req)
        return req

    def test_a_decline_KEEPS_the_mark_so_the_escape_can_retry(self):
        req = self._marked(_Req(RID_CHUNKED, prefix_len=8192, extend_len=4096))
        holder = _holder(chunked_req=req)
        holder._prefetch_kvcache = lambda r: "declined:rate_limited"
        self.assertEqual(self._apply(holder, req), "refetch-declined")
        from sglang.srt.managers.scheduler_pp_mixin import pp_premise_is_dead

        self.assertTrue(
            pp_premise_is_dead(req),
            "a declined attempt must not spend the mark -- that is the leak "
            "that made the escape a one-shot that never shot",
        )
        self.assertEqual(len(req.prefix_indices), 8192, "and the prefix is kept")

    def test_the_decline_names_its_REASON_in_the_log(self):
        req = self._marked(_Req(RID_CHUNKED, prefix_len=8192, extend_len=4096))
        holder = _holder(chunked_req=req)
        holder._prefetch_kvcache = lambda r: "declined:rate_limited"
        catcher = _Catcher(logging.WARNING)
        log = logging.getLogger("sglang.srt.managers.scheduler_pp_mixin")
        log.addHandler(catcher)
        try:
            self._apply(holder, req)
        finally:
            log.removeHandler(catcher)
        msg = "\n".join(catcher.messages)
        self.assertIn("REFETCH DECLINED", msg)
        self.assertIn("rate_limited", msg)
        self.assertIn(
            "cache_controller.py:841",
            msg,
            "a persistent rate_limited decline points at the #915 capacity "
            "asymmetry, and the line must say so rather than leave it to be "
            "rediscovered",
        )

    def test_the_retry_is_BOUNDED_and_ends_in_the_named_terminator(self):
        """UNBOUNDED RETRY IS THE #858 LIVELOCK SHAPE. The cap is what makes
        keeping the mark safe, and the terminator still names its discard so a
        Kein-Doppel-Prefill breach can never be silent."""
        from sglang.srt.managers.scheduler_pp_mixin import REFETCH_DECLINE_CAP

        req = self._marked(_Req(RID_CHUNKED, prefix_len=8192, extend_len=4096))
        holder = _holder(chunked_req=req)
        holder._prefetch_kvcache = lambda r: "declined:rate_limited"
        catcher = _Catcher(logging.WARNING)
        log = logging.getLogger("sglang.srt.managers.scheduler_pp_mixin")
        log.addHandler(catcher)
        try:
            outcomes = [self._apply(holder, req) for _ in range(REFETCH_DECLINE_CAP)]
        finally:
            log.removeHandler(catcher)
        self.assertEqual(
            outcomes[:-1], ["refetch-declined"] * (REFETCH_DECLINE_CAP - 1)
        )
        self.assertEqual(outcomes[-1], "recompute", "the cap must terminate")
        self.assertEqual(len(req.prefix_indices), 0)
        self.assertIn("8192", "\n".join(catcher.messages))

    def test_CANFAIL_a_None_returning_prefetch_is_a_DECLINE_not_a_success(self):
        """THE REGRESSION PIN FOR THE WHOLE TICKET. A tree cache that predates
        the honest verdict returns None. Reading that as success is precisely
        what produced 8885 silent no-ops on metal, so it must read as a
        decline and the mark must survive."""
        req = self._marked(_Req(RID_CHUNKED, prefix_len=8192, extend_len=4096))
        holder = _holder(chunked_req=req)
        holder._prefetch_kvcache = lambda r: None
        self.assertEqual(self._apply(holder, req), "refetch-declined")
        from sglang.srt.managers.scheduler_pp_mixin import pp_premise_is_dead

        self.assertTrue(pp_premise_is_dead(req))


class The915GateLineMustBeWIREDNotMerelyPresent(unittest.TestCase):
    """#949 ARM 3 -- THE THREE-STATE DELIVERY RULE, applied to #915.

    #915 was PRESENT-BUT-UNWIRED for twelve days: `note_prefetch_gate` recorded
    the decline reason on every attempt, and `format_prefetch_gate` had ZERO
    callers anywhere in the tree, so it was never printed. Every boot script
    printed that state and it was read as a note rather than as the missing
    half. This arm pins the wiring so it cannot silently regress.
    """

    def test_the_formatter_has_an_importer_OUTSIDE_its_own_module(self):
        import sglang.srt.mem_cache.unified_radix_cache as urc

        self.assertTrue(
            hasattr(urc, "_format_prefetch_gate"),
            "PRESENT-BUT-UNWIRED is the middle state and the expensive one: "
            "the reason was recorded all along and could not be read",
        )

    def test_the_gate_line_is_rate_limited_and_shares_the_census_knob(self):
        from sglang.srt.mem_cache import match_refusal_census as mrc

        self.assertEqual(
            mrc.prefetch_gate_due.__module__,
            "sglang.srt.mem_cache.match_refusal_census",
        )

    def test_ABSENT_is_not_zero_in_the_gate_line(self):
        from sglang.srt.mem_cache import match_refusal_census as mrc

        before = dict(mrc.PREFETCH_GATE_COUNTS)
        try:
            mrc.PREFETCH_GATE_COUNTS.clear()
            self.assertIn("no observation", mrc.format_prefetch_gate())
            mrc.note_prefetch_gate("rate_limited", 4096)
            line = mrc.format_prefetch_gate()
            self.assertIn("rate_limited=1", line)
            self.assertNotIn("anchor=", line, "a term that never fired is ABSENT")
        finally:
            mrc.PREFETCH_GATE_COUNTS.clear()
            mrc.PREFETCH_GATE_COUNTS.update(before)


class TheTerminatorMustSurviveATensorPrefix(unittest.TestCase):
    """#796 CLASS, THIRD INSTANCE -- and the first one a boot actually reached.

    Window-946rf-0828 killed all three ranks here:

        RuntimeError: Boolean value of Tensor with more than one value is
        ambiguous
          scheduler_pp_mixin.py:1523
          discarded = len(getattr(req, "prefix_indices", None) or ())

    `Req.prefix_indices` is a torch tensor. `x or ()` asks `bool(x)`, which
    torch refuses from two elements up -- fine for an empty prefix, fine for a
    ONE-token prefix, fatal for a real one.

    THE LINE IS PRE-EXISTING (#946) AND WAS UNREACHABLE UNTIL #949. The escape
    always took the silent `return "refetch"`, so the terminator below it had
    never run in production. Making the terminator DELIVERABLE is what exposed
    it -- the bug and its discovery have the same cause.

    IT IS ALSO THE THIRD COPY OF A SPELLING THAT ALREADY HAD ONE CANONICAL
    DEFINITION. `phase_flip_draft_bootstrap.prefix_len` exists precisely
    because this crash happened before (W37-B, 2026-08-25) and its docstring
    opens "ONE DEFINITION, because two of them is what put the boot on the
    floor". `scheduler.py:8108` carries the same warning in a comment. #946
    wrote a fourth spelling anyway, so the fix is to USE the canonical helper,
    not to write a fifth.
    """

    def _apply(self, holder, req):
        from sglang.srt.managers.scheduler_pp_mixin import (
            pp_apply_dead_premise_at_chunk_boundary,
        )

        return pp_apply_dead_premise_at_chunk_boundary(holder, req)

    def _tensor_req(self, n):
        import torch

        req = _Req(RID_CHUNKED, prefix_len=0, extend_len=4096)
        req.prefix_indices = torch.arange(n)

        def _truncate(told):
            req.prefix_indices = req.prefix_indices[:told]

        req.truncate_prefix_to = _truncate
        from sglang.srt.managers.scheduler_pp_mixin import pp_mark_premise_dead

        pp_mark_premise_dead(req)
        return req

    def test_the_terminator_survives_a_MULTI_ELEMENT_tensor_prefix(self):
        """The exact metal crash: 8192 cached slot ids as a tensor."""
        req = self._tensor_req(8192)
        holder = _holder(chunked_req=req)  # no _prefetch_kvcache -> terminator
        catcher = _Catcher(logging.WARNING)
        log = logging.getLogger("sglang.srt.managers.scheduler_pp_mixin")
        log.addHandler(catcher)
        try:
            self.assertEqual(self._apply(holder, req), "recompute")
        finally:
            log.removeHandler(catcher)
        self.assertIn(
            "8192",
            "\n".join(catcher.messages),
            "the discarded token count must still be NAMED -- a law breach may "
            "be necessary, it may never be silent",
        )

    def test_a_ZERO_D_tensor_prefix_reads_as_one_token_not_an_exception(self):
        """`len()` RAISES on a 0-d tensor while `numel()` answers 1. The
        canonical helper asks numel first for exactly this reason."""
        import torch

        req = self._tensor_req(0)
        req.prefix_indices = torch.tensor(7)  # 0-d
        req.truncate_prefix_to = lambda told: None
        holder = _holder(chunked_req=req)
        self.assertEqual(self._apply(holder, req), "recompute")

    def test_it_uses_the_CANONICAL_helper_and_does_not_spell_a_fifth_copy(self):
        """ONE DEFINITION. Two of them put the boot on the floor once already;
        three of them did it again in window-946rf."""
        import inspect

        from sglang.srt.managers import scheduler_pp_mixin as m

        src = inspect.getsource(m.pp_apply_dead_premise_at_chunk_boundary)
        # CODE ONLY. The first version of this assertion grepped raw source and
        # went red on the fix's own COMMENT, which quotes the landmine in order
        # to explain it. An assertion about code must not be satisfiable or
        # breakable by prose -- that is the #915 guard-comment lesson applied to
        # a test.
        code = "\n".join(
            line.split("#", 1)[0] for line in src.splitlines() if line.strip()
        )
        self.assertNotIn(
            "or ()",
            code,
            "the `x or ()` spelling is the landmine itself -- use prefix_len",
        )
        self.assertIn("prefix_len", code)


class _StoreTreeCache(_FakeTreeCache):
    """Tree cache whose controller answers a CONTENT-KEY presence query."""

    def __init__(self, pages_present: int, **kw):
        super().__init__(**kw)
        self.root_node = object()
        self.presence_calls = []
        outer = self

        class _CC:
            def store_presence_pages(self, token_ids, last_hash, prefix_keys=None):
                outer.presence_calls.append(len(token_ids or ()))
                return pages_present

        self.cache_controller = _CC()


class TheEscapeMustIssueWithoutALocalAnchor(unittest.TestCase):
    """#950 ARM 1 -- THE ROOT WINDOW-946RF PROVED, ANSWERED.

    Measured on metal: `reason=anchor_no_vote` 5 of 5, `[#915 prefetch-gate] no
    observation`. The escape's precondition is `last_host_node.backuped` --
    "the full KV is ALREADY in this rank's host pool" -- demanded as the entry
    price for an operation whose whole purpose is to obtain what is NOT
    resident. It is anti-correlated with the situation the escape exists for.

    The replacement is presence BY CONTENT KEY. Under #706 the key is a
    function of the tokens, not of any rank's layout or residency, so the fetch
    lands in the CURRENT layout -- which is also why this is the practical route
    to `cached>0` after a flip.
    """

    def _sched(self, tc, enable=True):
        from sglang.srt.managers.scheduler import Scheduler

        s = types.SimpleNamespace()
        s.enable_hicache_storage = enable
        s.tree_cache = tc
        s._prefetch_kvcache = types.MethodType(Scheduler._prefetch_kvcache, s)
        return s

    def _anchorless_req(self):
        """A dead-premise request with NO local anchor -- the metal shape."""
        req = _Req(RID_CHUNKED, prefix_len=8192, extend_len=4096)
        req.init_next_round_input = lambda *a, **kw: None
        req.last_host_node = types.SimpleNamespace(
            backuped=False,  # <-- the anchor is ABSENT, as on metal
            get_last_hash_value=lambda: None,
            get_prefix_hash_values=lambda parent: None,
            parent=None,
        )
        req.host_hit_length = 0
        req._compute_max_prefix_len = lambda n: n
        return req

    def test_no_anchor_but_the_STORE_HOLDS_THE_PAGES_so_the_fetch_is_ISSUED(self):
        tc = _StoreTreeCache(pages_present=4, register=True)
        sched = self._sched(tc)
        req = self._anchorless_req()
        self.assertEqual(sched._prefetch_kvcache(req), "issued")
        self.assertIn(
            req.rid,
            tc.ongoing_prefetch,
            "delivery is proven by the EFFECT, never by the return value",
        )
        self.assertTrue(tc.presence_calls, "the presence check must actually run")

    def test_the_store_is_asked_with_the_REAL_span_not_an_empty_one(self):
        """The pre-#950 code blanked the token list on the ineligible branch. A
        presence query on an empty span would answer about nothing."""
        tc = _StoreTreeCache(pages_present=4, register=True)
        self._sched(tc)._prefetch_kvcache(self._anchorless_req())
        self.assertTrue(
            tc.presence_calls and tc.presence_calls[0] > 0,
            f"presence asked about an empty span: {tc.presence_calls}",
        )

    def test_CANFAIL_store_does_NOT_hold_the_pages_declines_with_store_absent(self):
        """THE COUNTER-ARM. Presence must be able to say NO, with a reason --
        otherwise it is not a check, it is a rubber stamp."""
        tc = _StoreTreeCache(pages_present=0, register=False)
        verdict = self._sched(tc)._prefetch_kvcache(self._anchorless_req())
        self.assertEqual(verdict, "declined:store_absent")
        self.assertEqual(tc.calls, [], "an absent store must not enter the fetch")

    def test_a_LOCALLY_ANCHORED_request_never_pays_for_a_presence_query(self):
        """The default path is untouched and costs nothing extra: `backuped`
        still short-circuits, so the normal admission path adds no round-trip."""
        tc = _StoreTreeCache(pages_present=4, register=True)
        req = self._anchorless_req()
        req.last_host_node.backuped = True
        self.assertEqual(self._sched(tc)._prefetch_kvcache(req), "issued")
        self.assertEqual(
            tc.presence_calls, [], "an anchored request must not query the store"
        )

    def test_the_presence_verdict_feeds_the_580_VOTE_not_a_new_early_return(self):
        """GROUP UNIFORMITY. `prefetch_from_storage` is a collective under
        `symmetric`; a rank-local presence answer must ride the EXISTING vote as
        `locally_eligible`, never become a new early return and never a new
        collective -- that would be the #580 desync again."""
        seen = {}
        tc = _StoreTreeCache(pages_present=4, register=True)

        def _pfs(req_id, node, tokens, last_hash=None, prefix_keys=None, **kw):
            seen.update(kw)
            tc.ongoing_prefetch[req_id] = object()

        tc.prefetch_from_storage = _pfs
        tc.prefetch_participation_is_collective = lambda: True
        self._sched(tc)._prefetch_kvcache(self._anchorless_req())
        self.assertIn("locally_eligible", seen)
        self.assertTrue(
            seen["locally_eligible"],
            "the presence verdict must arrive AS the vote's local term",
        )
