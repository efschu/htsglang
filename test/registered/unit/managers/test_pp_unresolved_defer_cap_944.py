"""#944: a lookup miss is not a measurement, and the defer it causes is BOUNDED.

THE CLASS, THIRD INSTANCE. `_pp_reconcile_incoming_admission` resolves a rid
through a chain of lookups and, on a total miss, wrote `0` into
`local_match_lens`. `reconcile_pp_admission_decision` then read that 0 as a
MEASUREMENT and voided the pass. #797c patched the `chunked_req` miss, #798
patched the wrong-slot miss; each added a lookup and left the miss answering
with a number, which is why each looked like a fresh defect instead of the same
one a third time. #944 is the `running_batch` miss -- measured under real
agent-shaped load as 2106 `unhonourable prefix` events, 2107 voided passes and
a three-rank hang 35 s after health.

WHAT THIS FILE PINS, AND WHY EACH PART IS LOAD-BEARING.

1. `told == 0` IS HONOURABLE REGARDLESS OF THE LOOKUP. A zero-length prefix
   asks the receiving rank for nothing at all, so there is nothing a failed
   lookup could make unhonourable. Before the sentinel this fell out of the
   arithmetic for free (`local=0 >= told=0`); the sentinel is -1, so it stops
   falling out and has to be stated. It is not cosmetic: it is what makes the
   cap's escape hatch (below) actually terminate, and without it the FIRST,
   congruent round of every request retracts -- which is strictly worse than
   the defect being fixed.

2. THE TWO POPULATIONS ARE SEPARABLE ON THE WIRE, not merely in the log.
   `PPAdmissionEntry.unresolved` says "no rank could locate this request";
   `observed_local` says "this rank measured N and N < told". Encoding the
   first as a special value of the second is the class itself, one level up --
   so it gets its own field, and the field crosses the wire because the rank
   that must act on it (PP0) is not the rank that observes it.

3. THE DEFER IS PP0-ANCHORED, AND A DEFER ONLY ONE RANK TAKES IS THE NEXT
   DIVERGENCE. Each rank computes its own `local`, so each would decide
   locally and they would decide differently. Downstream ranks therefore only
   REPORT `unresolved`; the whole pass is voided by the existing #797
   mechanism (group-uniform: `pp_pass_should_void` ORs the incoming flag and
   never clears it), and the decision to defer AGAIN versus escalate is taken
   once, by PP0, from a count that rode the wire.

4. THE RE-OFFER IS BOUNDED. `_learned_floor` is what damped the re-offer
   before, and it is fed from `observed_local` -- which a lookup miss must not
   set, because no number was measured. So the sentinel ALONE makes the
   2106-loop worse than the 0 did: the 0 at least clamped. The cap is what
   replaces that clamp with a bound that terminates for a stated reason:
   after `UNRESOLVED_DEFER_CAP` unresolved rounds PP0 emits a LOUD refusal
   naming the rid and all four lookup locations, and pins the next offer to
   `told=0`, which (1) makes unconditionally honourable. Never unbounded --
   that is the #858 livelock shape.

WHY told=0 HERE WHEN `PPAdmissionCongruenceGuard`'s own docstring REJECTS a
told=0 pin. It rejects it as the GENERAL policy for an ordinary shortfall,
where learning the real observed value costs the same and keeps the degrade
rare. That reasoning does not reach here, because here there IS no observed
value to learn -- the lookup failed. told=0 is the only offer that is
honourable without a measurement, so it is the only available terminator.
"""

import logging
import unittest

from sglang.srt.managers.pp_admission_congruence import (
    UNKNOWN_MATCH,
    UNRESOLVED_DEFER_CAP,
    PPAdmissionCongruenceGuard,
    PPAdmissionDecision,
    PPAdmissionEntry,
    reconcile_pp_admission_decision,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20)

WORLD = 3
RANK = 1
RID = "rid-944"


def _decision(told, *, rid=RID, extend=64, mb_id=0, **kw):
    return PPAdmissionDecision(
        mb_id=mb_id,
        entries=(PPAdmissionEntry(rid=rid, prefix_len=told, extend_len=extend, **kw),),
    )


def _reconcile(told, local, **kw):
    """One entry through the SHIPPED pure reconcile. `local=None` means the
    rid is absent from `local_match_lens` entirely -- the real miss shape,
    not a hand-written sentinel."""
    lens = {} if local is None else {RID: local}
    return reconcile_pp_admission_decision(
        _decision(told, **kw), lens, rank=RANK, pp_size=WORLD
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


class TheZeroOfferIsHonourableRegardlessOfTheLookup(unittest.TestCase):
    """Part 1. This is the regression the sentinel introduced and the
    precondition the cap's escape depends on."""

    def test_an_unresolved_rid_offered_told_zero_is_admitted(self):
        effective, amended = _reconcile(0, None)
        self.assertEqual(
            effective,
            {RID: 0},
            "told=0 demands no prefix reuse, so a failed lookup cannot make "
            "it unhonourable -- and retracting it voids the FIRST, congruent "
            "round of every request",
        )
        entry = amended.entries[0]
        self.assertFalse(entry.retracted)
        self.assertFalse(entry.unresolved)
        self.assertTrue(entry.admitted)

    def test_a_known_rank_at_told_zero_is_unchanged(self):
        """The hoist must be a no-op for every rank that DID resolve: local
        is non-negative, so `local >= told == 0` already held."""
        for local in (0, 1, 4096):
            with self.subTest(local=local):
                effective, amended = _reconcile(0, local)
                self.assertEqual(effective, {RID: 0})
                self.assertFalse(amended.entries[0].retracted)

    def test_a_positive_offer_to_an_unresolved_rid_still_defers(self):
        """The hoist must not become a blanket amnesty: told>0 on a rid
        nobody could find is still not admissible this pass."""
        effective, amended = _reconcile(512, None)
        self.assertEqual(effective, {})
        self.assertTrue(amended.entries[0].retracted)
        self.assertTrue(amended.entries[0].unresolved)


class TheTwoPopulationsAreSeparableOnTheWire(unittest.TestCase):
    """Part 2. The whole reason #797c and #798 each read as a fresh defect."""

    def test_a_lookup_miss_is_unresolved_and_teaches_nothing(self):
        _, amended = _reconcile(512, None)
        entry = amended.entries[0]
        self.assertTrue(entry.unresolved, "the miss must say it is a miss")
        self.assertIsNone(
            entry.observed_local,
            "a miss measured nothing, so it must put no number on the wire -- "
            "`observed_local` feeds `_learned_floor`, which clamps the next "
            "offer, and a floor from an unmeasured number is the class again",
        )
        self.assertEqual(entry.retracted_by_rank, RANK)

    def test_a_genuine_shortfall_is_measured_and_is_not_unresolved(self):
        _, amended = _reconcile(512, 128)
        entry = amended.entries[0]
        self.assertFalse(
            entry.unresolved,
            "a rank that resolved the rid and found 128 tokens measured "
            "something; it must never be pooled with the ranks that found "
            "nothing to measure",
        )
        self.assertEqual(entry.observed_local, 128)

    def test_the_explicit_sentinel_and_the_absent_key_agree(self):
        """`local_match_lens` may spell the miss either way (the mixin writes
        the sentinel; a caller that never wrote the key at all is the same
        fact). Both must land in the same population."""
        by_absence = _reconcile(512, None)[1].entries[0]
        by_sentinel = _reconcile(512, UNKNOWN_MATCH)[1].entries[0]
        self.assertEqual(by_absence.unresolved, by_sentinel.unresolved)
        self.assertEqual(by_absence.observed_local, by_sentinel.observed_local)
        self.assertTrue(by_sentinel.unresolved)

    def test_the_flag_survives_the_wire_codec(self):
        from sglang.srt.managers.scheduler_pp_mixin import (
            pp_admission_decision_from_wire,
            pp_admission_decision_to_wire,
        )

        _, amended = _reconcile(512, None)
        back = pp_admission_decision_from_wire(pp_admission_decision_to_wire(amended))
        self.assertTrue(
            back.entries[0].unresolved,
            "PP0 is not the rank that observes the miss, so the flag is "
            "useless unless it crosses the wire",
        )
        self.assertIsNone(back.entries[0].observed_local)

    def test_an_already_retracted_entry_passes_through_verbatim(self):
        """#791's exactly-once contract, re-pinned against the new field: a
        later rank must not overwrite an earlier rank's verdict."""
        pre = PPAdmissionDecision(
            mb_id=0,
            entries=(
                PPAdmissionEntry(
                    rid=RID,
                    prefix_len=512,
                    extend_len=64,
                    admitted=False,
                    retracted=True,
                    retracted_by_rank=0,
                    observed_local=7,
                ),
            ),
        )
        _, amended = reconcile_pp_admission_decision(pre, {}, rank=RANK, pp_size=WORLD)
        self.assertEqual(amended.entries[0], pre.entries[0])


class TheUnresolvedDeferIsCapped(unittest.TestCase):
    """Parts 3 and 4. PP0 owns the count and the escalation."""

    def _guard(self, **kw):
        return PPAdmissionCongruenceGuard(**kw)

    def _defer_once(self, guard, rid=RID):
        """A LAP that arrives -- i.e. the healthy case. Still exercised,
        because when the ring turns this is how the population gets its NAME
        (`unresolved_rounds`). It is no longer how the bound is reached."""
        guard.record_return_trip(
            _decision(
                512,
                rid=rid,
                admitted=False,
                retracted=True,
                retracted_by_rank=RANK,
                observed_local=None,
                unresolved=True,
            )
        )

    def _reoffer(self, guard, n, rid=RID, candidate=4096):
        """#944b DRIVE THE BOUND THE WAY PRODUCTION DOES: PP0 re-offers the
        same rid the same length, pass after pass, and NOTHING comes back.

        These sites previously reached the cap by calling `record_return_trip`
        -- i.e. by handing the guard the very delivery whose absence is the
        defect. The live boot of 2026-08-27 showed that delivery never happens
        on a voided pass (the void blocks the ring), so the old trigger was
        unreachable in production while these tests were green. INVERTED
        DELIBERATELY, reasoning recorded, nothing deleted: the behaviour under
        test (a non-progressing offer must be bounded) is unchanged; only what
        drives it moved off the wire and onto PP0's own local observation.
        """
        return [guard.prefix_len_for(rid, candidate) for _ in range(n)]

    def test_below_the_cap_the_offer_is_not_clamped(self):
        g = self._guard()
        for _ in range(UNRESOLVED_DEFER_CAP - 1):
            self._defer_once(g)
        self.assertEqual(
            g.prefix_len_for(RID, 4096),
            4096,
            "a rid that has not exhausted its defers keeps its full reuse -- "
            "the miss taught nothing, so there is nothing to clamp from",
        )

    def test_at_the_cap_the_offer_is_pinned_to_zero(self):
        g = self._guard()
        offers = self._reoffer(g, UNRESOLVED_DEFER_CAP)
        self.assertEqual(offers, [4096] * UNRESOLVED_DEFER_CAP, "not yet capped")
        self.assertEqual(
            g.prefix_len_for(RID, 4096),
            0,
            "told=0 is the only offer honourable WITHOUT a measurement, so it "
            "is the only terminator available to a defer nobody could resolve",
        )

    def test_the_escalation_is_loud_and_names_the_four_lookup_locations(self):
        g = self._guard()
        self._reoffer(g, UNRESOLVED_DEFER_CAP)
        catcher = _Catcher(logging.ERROR)
        log = logging.getLogger("sglang.srt.managers.pp_admission_congruence")
        log.addHandler(catcher)
        try:
            g.prefix_len_for(RID, 4096)
        finally:
            log.removeHandler(catcher)
        self.assertEqual(
            len(catcher.records), 1, f"expected one ERROR, got {catcher.messages}"
        )
        msg = catcher.messages[0]
        self.assertIn(RID, msg)
        self.assertIn("#944", msg)
        for place in ("waiting queue", "chunked_req", "slot", "running batch"):
            self.assertIn(
                place,
                msg,
                "the refusal must name every place that was searched, or the "
                "next reader has to rediscover the lookup chain from the code",
            )

    def test_the_escalation_fires_once_not_once_per_offer(self):
        g = self._guard()
        self._reoffer(g, UNRESOLVED_DEFER_CAP)
        catcher = _Catcher(logging.ERROR)
        log = logging.getLogger("sglang.srt.managers.pp_admission_congruence")
        log.addHandler(catcher)
        try:
            for _ in range(5):
                self.assertEqual(g.prefix_len_for(RID, 4096), 0)
        finally:
            log.removeHandler(catcher)
        self.assertEqual(
            len(catcher.records),
            1,
            "a bounded refusal that re-logs every pass is the 2106-line log "
            "the ticket is about, in a different colour",
        )

    def test_a_served_rid_forgets_both_the_floor_and_the_count(self):
        g = self._guard()
        self._reoffer(g, UNRESOLVED_DEFER_CAP)
        self.assertEqual(g.prefix_len_for(RID, 4096), 0)
        g.record_return_trip(_decision(0, admitted=True))
        self.assertEqual(
            g.prefix_len_for(RID, 4096),
            4096,
            "the count clears only on a pass that actually SERVED the rid -- "
            "and it must clear, or one bad minute caps the request forever",
        )
        self.assertEqual(g.unresolved_rounds(RID), 0)

    def test_a_deferral_must_not_clear_the_count_that_bounds_it(self):
        """#552's own lesson, re-pinned: a defer that resets its own counter
        makes the bound unreachable, which is the bug wearing a fix."""
        g = self._guard()
        for i in range(UNRESOLVED_DEFER_CAP + 2):
            self._defer_once(g)
            self.assertEqual(g.unresolved_rounds(RID), i + 1)

    def test_a_measured_shortfall_does_not_count_toward_the_defer_cap(self):
        g = self._guard()
        for _ in range(UNRESOLVED_DEFER_CAP + 3):
            g.record_return_trip(
                _decision(
                    512,
                    admitted=False,
                    retracted=True,
                    retracted_by_rank=RANK,
                    observed_local=128,
                )
            )
        self.assertEqual(
            g.unresolved_rounds(RID),
            0,
            "the #791 shortfall path has its own terminating argument (a "
            "strictly decreasing learned floor); pooling it into the #944 "
            "cap would be the two populations read as one again",
        )
        self.assertEqual(g.prefix_len_for(RID, 4096), 128)

    def test_the_cap_is_disableable_and_that_restores_the_pre_944_shape(self):
        g = self._guard(unresolved_defer_cap=0)
        self._reoffer(g, UNRESOLVED_DEFER_CAP + 5)
        self.assertEqual(
            g.prefix_len_for(RID, 4096),
            4096,
            "<=0 disables the bound entirely, the same escape hatch #552's "
            "defer_limit carries -- so the cap can be neutered on its own in "
            "a can-fail proof without taking the sentinel down with it",
        )

    def test_the_count_is_per_rid(self):
        g = self._guard()
        self._reoffer(g, UNRESOLVED_DEFER_CAP, rid="rid-a")
        self.assertEqual(g.prefix_len_for("rid-a", 4096), 0)
        self.assertEqual(g.prefix_len_for("rid-b", 4096), 4096)


class TheBoundActuallyTerminates(unittest.TestCase):
    """The cap is only worth anything if the loop it bounds ENDS. Drives the
    real PP0 guard against the real downstream reconcile, with the lookup
    permanently blinded, and requires the pass to run."""

    def _round(self, guard, candidate, deliver=True):
        """One pass. `deliver=False` models THE BROKEN RING, which is not a
        hypothetical: the void that must be counted is carried home by the
        output lap, and the same void parks a middle rank in
        `_pp_drain_voided_proxy`, so on the live rig the lap does not arrive
        (measured 2026-08-27: 4010 UNRESOLVED, 0 escalations, told frozen at
        8192 for 8023 lines).

        The first version of this test always delivered, which is why it
        passed while the shipped bound was dead code -- A TEST THAT SUPPLIES
        THE DELIVERY UNDER QUESTION CANNOT TEST THAT DELIVERY.
        """
        told = guard.prefix_len_for(RID, candidate)
        effective, amended = _reconcile(told, None)
        if deliver:
            guard.record_return_trip(amended)
        return told, effective

    def test_the_bound_holds_with_the_RING_BROKEN(self):
        """#944b THE REGRESSION THE LIVE BOOT FOUND, as a desk test.

        No lap is ever delivered -- `record_return_trip` is never called, which
        is the measured live condition. The bound must still terminate, because
        a bound that needs the ring cannot end a failure that stops the ring.
        """
        g = PPAdmissionCongruenceGuard()
        tolds, served_at = [], None
        for i in range(UNRESOLVED_DEFER_CAP + 5):
            told, effective = self._round(g, 4096, deliver=False)
            tolds.append(told)
            if RID in effective:
                served_at = i
                break
        self.assertIsNotNone(
            served_at,
            f"THE 2026-08-27 LIVE FAILURE, reproduced: told never moved and "
            f"nothing was ever served -- {tolds}. A lap-fed counter reads 0 "
            f"exactly when the ring is broken, which is when it must not.",
        )
        self.assertEqual(tolds[-1], 0, f"the terminator is told=0: {tolds}")
        self.assertEqual(
            g.unresolved_rounds(RID),
            0,
            "and it terminated with the LAP-FED counter still at zero -- proof "
            "the bound did not lean on the return trip",
        )
        self.assertGreater(g.offer_streak(RID), 0)

    def test_the_630_SHORTFALL_loop_is_bounded_too_when_the_ring_is_broken(self):
        """SIBLING SWEEP RESULT (#630, the THIRD instance of the class).

        `_learned_floor` is the other thing `record_return_trip` writes, and it
        is written on the LEARN side -- so #630's termination argument ("every
        new retraction sets the floor strictly below the told that just
        failed") holds only while laps arrive. With the ring broken, a genuine
        MEASURED shortfall re-offers the identical told for ever, exactly like
        the unresolved one did. Same class, different population.

        It needs no separate fix, and that is the point of counting the OFFER
        rather than the reported round: the streak does not care WHY the offer
        stopped moving. This test is the proof that the class fix covers the
        sibling, not just the instance that was reported.
        """
        g = PPAdmissionCongruenceGuard()
        tolds = []
        for _ in range(UNRESOLVED_DEFER_CAP + 3):
            told = g.prefix_len_for(RID, 4096)
            tolds.append(told)
            # A real, MEASURED shortfall (not a miss): local=128 < told. No lap
            # is delivered, so the floor is never learned.
            _reconcile(told, 128)
            if told == 0:
                break
        self.assertEqual(
            tolds[-1],
            0,
            f"a measured shortfall whose lesson never gets home must be "
            f"bounded by the same lap-free streak: {tolds}",
        )
        self.assertIsNone(
            g.learned_floor(RID),
            "and it terminated with NO floor learned -- proving the bound and "
            "the #630 floor are independent mechanisms",
        )

    def test_a_permanently_unresolvable_rid_is_served_within_the_cap(self):
        g = PPAdmissionCongruenceGuard()
        tolds, served_at = [], None
        for i in range(UNRESOLVED_DEFER_CAP + 5):
            told, effective = self._round(g, 4096)
            tolds.append(told)
            if RID in effective:
                served_at = i
                break
        self.assertIsNotNone(
            served_at,
            f"the re-offer never terminated: told sequence {tolds} -- this is "
            f"the 2106-void loop, and an unbounded defer is the #858 shape",
        )
        self.assertLessEqual(
            served_at,
            UNRESOLVED_DEFER_CAP,
            f"the bound must be the cap, not 'eventually': {tolds}",
        )
        self.assertEqual(tolds[-1], 0, f"the terminator is told=0: {tolds}")

    def test_without_the_cap_the_same_drive_never_terminates(self):
        """CAN-FAIL PROOF. Neuter only the cap; the sentinel, the wire flag
        and the told=0 hoist all still run. If this converges anyway, the cap
        is not what is doing the terminating and this file is measuring
        nothing."""
        g = PPAdmissionCongruenceGuard(unresolved_defer_cap=0)
        for _ in range(UNRESOLVED_DEFER_CAP + 20):
            told, effective = self._round(g, 4096)
            self.assertEqual(told, 4096, "nothing else damps the re-offer")
            self.assertEqual(effective, {})


class _CanRunListReq:
    """A request in the shape PRODUCTION re-offers, which is the shape none of
    the earlier tests ever built.

    `PrefillAdder` writes `extend_range` via `set_extend_range` on every path
    that appends to `can_run_list`, so a mid-chunked-prefill request always has
    one -- and `_executed_extent` therefore returns a value for it, which sends
    `build_pp_admission_decision` down its EXECUTED branch. That branch is the
    one the guard is not on.
    """

    class _Range:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    def __init__(self, rid, prefix_len, extend_len):
        self.rid = rid
        self.prefix_indices = list(range(prefix_len))
        self.extend_range = self._Range(prefix_len, prefix_len + extend_len)
        self.full_untruncated_fill_ids = list(range(prefix_len + extend_len))


class TheCompensatorMustBeONTheDefectsPATH(unittest.TestCase):
    """#944c THE REACHABILITY RATCHET -- one test per instance of the class.

    THE CLASS, three instances, all the same shape: a compensator that does the
    right thing when called, sitting off the path that produces the defect.

      #939  the prefetch re-issue, behind the refusal it compensates
      #944  the defer cap, behind the return trip the void breaks
      #944b the offer streak, behind a `prefix_len_for` the re-offer skips

    Every one of them passed a test asking "does this do the right thing when
    called". None of them had a test asking "IS THIS ON THE PATH THE DEFECT
    TAKES". That second question is what these are, and instance 4 must fail
    HERE rather than on a boot -- two windows were spent learning that.
    """

    def _build(self, guard, req, pp_size=WORLD):
        from sglang.srt.managers.pp_admission_congruence import (
            build_pp_admission_decision,
        )

        return build_pp_admission_decision(
            0,
            [req],
            pp_size=pp_size,
            guard=guard,
            # THE PRODUCTION SETTING. scheduler.py:9076 is the one production
            # call site and it passes True. Every earlier test in this file
            # left it False, which is why they all exercised the fallback
            # branch -- the one the guard IS on -- and never the real one.
            require_executed_geometry=True,
        )

    def test_944b_the_streak_sees_the_EXECUTED_reoffer_path(self):
        """#944b REACHABILITY. Measured on the rig 2026-08-27: two rids,
        `told=8192` on every line, `UNRESOLVABLE` 0 against 3506 UNRESOLVED --
        because `build_pp_admission_decision`'s executed branch returns before
        the guard block, so `prefix_len_for` is never called for an
        already-admitted request. This is that failure as a desk test."""
        g = PPAdmissionCongruenceGuard()
        req = _CanRunListReq(RID, prefix_len=8192, extend_len=512)
        tolds = []
        for _ in range(UNRESOLVED_DEFER_CAP + 3):
            d = self._build(g, req)
            tolds.append(d.entries[0].prefix_len)
        self.assertGreater(
            g.offer_streak(RID),
            0,
            "THE #944b LIVE FAILURE: the guard never saw the offer at all, so "
            "its streak stayed at 0 while production re-offered the same "
            "length thousands of times. A bound the defect's own path does "
            "not traverse is not a bound.",
        )
        self.assertGreater(
            g.offer_streak(RID),
            UNRESOLVED_DEFER_CAP,
            f"and the streak must actually cross the cap: {tolds}",
        )
        # THE REPORT IS NEVER REWRITTEN ON THIS BRANCH, and that is not a
        # shortfall of the fix -- it is the contract. `prefix_len` here is what
        # the rank EXECUTED (`extend_range.start`); reporting anything else
        # names a pass no rank ran (instr21). So the executed branch COUNTS and
        # refuses loudly; the acting half belongs upstream, where the prefix is
        # still choosable. See #946.
        self.assertEqual(
            tolds,
            [8192] * len(tolds),
            "the executed geometry must be reported truthfully, always",
        )

    def test_the_loud_refusal_FIRES_on_the_executed_path(self):
        """CRITERION (1) OF THE WINDOW, as a desk test. Two boots measured
        `UNRESOLVABLE` = 0 while the rig re-offered a frozen `told` thousands
        of times. The escalation must now fire from the branch production
        actually takes, and fire exactly once."""
        g = PPAdmissionCongruenceGuard()
        req = _CanRunListReq(RID, prefix_len=8192, extend_len=512)
        catcher = _Catcher(logging.ERROR)
        log = logging.getLogger("sglang.srt.managers.pp_admission_congruence")
        log.addHandler(catcher)
        try:
            for _ in range(UNRESOLVED_DEFER_CAP + 4):
                self._build(g, req)
        finally:
            log.removeHandler(catcher)
        self.assertEqual(
            len(catcher.records),
            1,
            f"expected exactly one UNRESOLVABLE from the executed path, got "
            f"{catcher.messages}",
        )
        self.assertIn("UNRESOLVABLE", catcher.messages[0])
        self.assertIn(RID, catcher.messages[0])

    def test_946_the_learned_floor_is_INERT_on_the_reoffer_path_ON_PURPOSE(self):
        """#946 DECIDED, NOT INFERRED.

        The question filed as #946 is whether `_learned_floor` is inert on the
        re-offer path. IT IS, and this test pins that as INTENTIONAL rather
        than leaving it as a suspicion: the executed branch reports
        `extend_range.start` -- the prefix the rank actually used -- and
        applying a floor there would report a prefix the rank did NOT use,
        which is the instr21 defect the builder's own docstring forbids.

        So #944c does NOT cover #946 by making the floor apply here, and it
        must not. What it covers is the OBSERVABILITY half: the offer is now
        counted on this path, so a floor that cannot bite still produces a
        loud, named refusal instead of silence. The remaining half of #946 --
        applying an escape to a chunked continuation -- lives upstream, where
        the prefix is still choosable, and is not this function's to take.
        """
        g = PPAdmissionCongruenceGuard()
        g.record_return_trip(
            _decision(
                8192,
                admitted=False,
                retracted=True,
                retracted_by_rank=RANK,
                observed_local=128,
            )
        )
        self.assertEqual(g.learned_floor(RID), 128, "a floor IS outstanding")
        req = _CanRunListReq(RID, prefix_len=8192, extend_len=512)
        d = self._build(g, req)
        self.assertEqual(
            d.entries[0].prefix_len,
            8192,
            "#946: the floor is deliberately NOT applied to an executed "
            "geometry -- the report must state what ran, not what a floor "
            "would have preferred. Changing this to 128 would be the instr21 "
            "defect, not a fix.",
        )
        self.assertEqual(
            g.offer_streak(RID),
            1,
            "but the offer IS counted, which is the half #944c does deliver: "
            "an inert floor now produces a named refusal instead of silence",
        )

    def test_the_offer_construction_sites_are_all_covered(self):
        """THE RATCHET ITSELF. `prefix_len` is written into a
        `PPAdmissionEntry` at exactly N sites inside the builder; every one of
        them is an OFFER, and the streak must see all of them.

        Pinned as a count so a NEW branch that emits an offer turns this red
        and forces the same question to be asked again. That is the whole
        mechanism: instance 4 fails here, not on a boot.
        """
        import inspect

        from sglang.srt.managers import pp_admission_congruence as pac

        src = inspect.getsource(pac.build_pp_admission_decision)
        sites = src.count("PPAdmissionEntry(")
        self.assertEqual(
            sites,
            2,
            "The builder emits offers from a different number of sites than "
            "this pin expects. Each one is a place production can propose a "
            "`told`, and the #944b bound is only real if EVERY one of them is "
            "counted. Sweep the new site, route it through the same counter, "
            "then update this pin.",
        )

    def test_939_the_prefetch_reissue_is_WIRED_not_merely_present(self):
        """#939 REACHABILITY -- the FIRST instance of the class, kept here so
        all three sit together and instance 4 has a pattern to match.

        Measured across six boots: `PREFETCH RE-ISSUED` 0 and `RETIRED
        PREFETCH REAPED` 0 -- the compensator existed, was correct, and was
        never reached, because it sat behind the refusal it was built to
        compensate. Presence at file:line was never the question.

        THREE STATES, NEVER TWO (the delivery rule): ABSENT /
        PRESENT-BUT-UNWIRED / PRESENT-AND-WIRED. The middle one is the
        expensive mistake in both directions, so it gets its own assertion:
        an importer OUTSIDE the defining module is what separates it from the
        third state.
        """
        import pathlib

        import sglang

        root = pathlib.Path(sglang.__file__).resolve().parent
        home = root / "srt/mem_cache/unified_radix_cache.py"
        self.assertTrue(home.exists())
        text = home.read_text(encoding="utf-8", errors="replace")
        self.assertIn("def drain_retired_prefetch", text, "ABSENT would be state 1")

        outside = [
            p
            for p in root.rglob("*.py")
            if p != home
            and "drain_retired_prefetch"
            in p.read_text(encoding="utf-8", errors="replace")
        ]
        self.assertTrue(
            outside,
            "PRESENT-BUT-UNWIRED: `drain_retired_prefetch` has no caller "
            "outside its own file, so it cannot run in production no matter "
            "how correct it is. That is the #939 shape and it costs a boot to "
            "discover.",
        )

    def test_944_the_cap_does_not_depend_on_a_lap(self):
        """#944 REACHABILITY, kept as its own arm. The first bound was fed by
        `record_return_trip`, i.e. by a ring lap the void itself breaks."""
        g = PPAdmissionCongruenceGuard()
        for _ in range(UNRESOLVED_DEFER_CAP + 2):
            g.prefix_len_for(RID, 4096)
        self.assertEqual(
            g.prefix_len_for(RID, 4096),
            0,
            "the bound must arm with NOTHING ever delivered back",
        )
        self.assertEqual(
            g.unresolved_rounds(RID),
            0,
            "and with the lap-fed counter still at zero, which is the proof "
            "it is not what armed it",
        )


class TheConsumerSweepRatchet(unittest.TestCase):
    """THE ZUKUNFTS-CHECK: what test would have caught instance FOUR?

    Not a fourth lookup -- a fourth CONSUMER. #797c and #798 were each fixed
    by adding a lookup, and neither fix asked who READS the value; that is the
    only reason the same defect survived twice. A value's meaning is a
    property of its readers, so the fix is only complete while the reader set
    is known, and it is only known while something FAILS when it changes.

    This test enumerates every read of the two overloaded names in the shipped
    tree and pins the set. A new reader turns it red, and the red says what to
    do: sweep the new site, decide what it must do with UNKNOWN/None, then
    extend the list here with that decision recorded.
    """

    # Paths are relative to the `python/` root, i.e. exactly what an importer
    # writes. Deliberately a FILE SET, not line numbers: line numbers churn on
    # every edit above them and would make this a nuisance rather than a
    # ratchet.
    EXPECTED_OBSERVED_LOCAL_READERS = {
        "sglang/srt/managers/pp_admission_congruence.py",
        "sglang/srt/managers/scheduler_pp_mixin.py",
    }

    #: The TEST readers, pinned separately and deliberately. A value's meaning
    #: is a property of its readers, and a test that asserts on
    #: `observed_local` is asserting a MEANING -- four of the files below had
    #: to be inverted by hand when #944 changed what a miss puts there, and
    #: they were found by grep, not by a failing test. Now they are found by a
    #: failing test. Adding a reader here is cheap; the point is that it cannot
    #: happen silently.
    EXPECTED_OBSERVED_LOCAL_TEST_READERS = {
        "test_pp_admission_retry_livelock_630.py",
        "test_pp_admission_wiring_791.py",
        "test_pp_dead_peer_is_not_the_wedge_801.py",
        "test_pp_proxy_retracted_pass_mispair_791c.py",
        "test_pp_reconcile_slot_blind_798.py",
        "test_pp_retracted_pass_void_797.py",
        "test_pp_unresolved_defer_cap_944.py",
        # Added because THIS RATCHET CAUGHT IT, on its first run after the
        # gloo falsifiers were written -- which is the can-fail proof this
        # pin would otherwise be missing. Its assertion for the UNRESOLVED
        # case: `observed_local` must be None on every round of the clean
        # drive and 0 on at least one round of the danger mutant.
        "test_pp_unresolved_group_defer_gloo_944.py",
        "test_pp_void_send_contract_801.py",
        # #963: the prefix-scoped floor learns from the same `observed_local`,
        # so it is a reader of this value's MEANING. Its assertion for the
        # UNRESOLVED (None) case is that the miss teaches the PREFIX floor
        # nothing -- the stakes are strictly higher than for the rid-scoped
        # floor, because a floor invented from a miss would cap every request
        # over that prefix rather than one request
        # (`test_an_unresolved_miss_teaches_the_prefix_floor_nothing`).
        "test_pp_prefix_scoped_floor_963.py",
    }

    def _tree(self):
        import pathlib

        import sglang

        return pathlib.Path(sglang.__file__).resolve().parents[1]

    def _readers(self, needle, root=None):
        import re

        root = root or self._tree()
        hits = {}
        for path in sorted(root.rglob("*.py")):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            n = len(re.findall(rf"\b{re.escape(needle)}\b", text))
            if n:
                hits[str(path.relative_to(root))] = n
        return hits

    def test_the_set_of_files_reading_observed_local_is_pinned(self):
        found = set(self._readers("observed_local"))
        self.assertEqual(
            found,
            self.EXPECTED_OBSERVED_LOCAL_READERS,
            "A NEW READER OF `observed_local` APPEARED (or one vanished). "
            "This field carries a MEASUREMENT and is None when the rank "
            "measured nothing -- #944's whole root is a reader that could not "
            "tell those apart. Sweep the new site: decide explicitly what it "
            "must do with None, then update this pin with that decision "
            "recorded at the site.",
        )

    def test_the_set_of_tests_asserting_on_observed_local_is_pinned(self):
        import pathlib

        here = pathlib.Path(__file__).resolve().parent
        found = {p for p in self._readers("observed_local", root=here)}
        self.assertEqual(
            found,
            self.EXPECTED_OBSERVED_LOCAL_TEST_READERS,
            "A NEW TEST ASSERTS ON `observed_local`. That is a reader too, and "
            "the #944 sweep had to invert four of these BY GREP because "
            "nothing failed when the field's meaning changed. Decide what the "
            "new site must assert for the UNRESOLVED (None) case, record the "
            "reasoning at that site, then add it here.",
        )

    def test_the_set_of_files_reading_the_unknown_sentinel_is_pinned(self):
        found = set(self._readers("UNKNOWN_MATCH"))
        self.assertEqual(
            found,
            {
                "sglang/srt/managers/pp_admission_congruence.py",
                "sglang/srt/managers/scheduler_pp_mixin.py",
            },
            "The UNKNOWN sentinel must stay INSIDE the resolution path. It "
            "escaping into a third module is how it would reach a reader that "
            "treats -1 as a length -- the #944 defect with the sign flipped.",
        )


if __name__ == "__main__":
    unittest.main()
