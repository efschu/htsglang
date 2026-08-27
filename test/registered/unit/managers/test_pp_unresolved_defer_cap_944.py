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
        for _ in range(UNRESOLVED_DEFER_CAP):
            self._defer_once(g)
        self.assertEqual(
            g.prefix_len_for(RID, 4096),
            0,
            "told=0 is the only offer honourable WITHOUT a measurement, so it "
            "is the only terminator available to a defer nobody could resolve",
        )

    def test_the_escalation_is_loud_and_names_the_four_lookup_locations(self):
        g = self._guard()
        for _ in range(UNRESOLVED_DEFER_CAP):
            self._defer_once(g)
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
        for _ in range(UNRESOLVED_DEFER_CAP):
            self._defer_once(g)
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
        for _ in range(UNRESOLVED_DEFER_CAP):
            self._defer_once(g)
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
        for _ in range(UNRESOLVED_DEFER_CAP + 5):
            self._defer_once(g)
        self.assertEqual(
            g.prefix_len_for(RID, 4096),
            4096,
            "<=0 disables the bound entirely, the same escape hatch #552's "
            "defer_limit carries -- so the cap can be neutered on its own in "
            "a can-fail proof without taking the sentinel down with it",
        )

    def test_the_count_is_per_rid(self):
        g = self._guard()
        for _ in range(UNRESOLVED_DEFER_CAP):
            self._defer_once(g, rid="rid-a")
        self.assertEqual(g.prefix_len_for("rid-a", 4096), 0)
        self.assertEqual(g.prefix_len_for("rid-b", 4096), 4096)


class TheBoundActuallyTerminates(unittest.TestCase):
    """The cap is only worth anything if the loop it bounds ENDS. Drives the
    real PP0 guard against the real downstream reconcile, with the lookup
    permanently blinded, and requires the pass to run."""

    def _round(self, guard, candidate):
        told = guard.prefix_len_for(RID, candidate)
        effective, amended = _reconcile(told, None)
        guard.record_return_trip(amended)
        return told, effective

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
