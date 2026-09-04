"""#1176 (review r3): THE RELOCATED STOP MUST ACTUALLY ARRIVE.

be3ec1760b traded a loud group STOP for a silent rank split (a follower whose
store witness contradicted withheld `seam_transport_premise_holds` alone, so it
built no prefill batch while PP0 built one). 1634bc3d28 traded that split for a
REPORT that rides the #1175 completion carrier home to PP0, where
`_admit_under_group_completion` raises once, loudly, with the peer named.

Two independent reviewers measured that the report never arrives:

  (B1) `pp_prefetch_completion_facts_from_wire` special-cased only
       PREFETCH_PENDING. Both sentinels are STRINGS by construction, so
       `int(completed)` raised ValueError on CONTRADICTION and the
       `except (TypeError, ValueError): continue` DROPPED the entry -- at PP0's
       absorber and at every relay hop. Measured on the shipped functions:
       wire {KEY: (("r1", CONTRADICTION, 1), ("r1", 4096, 2))} parsed to
       (("r1", 4096, 2),) alone, PP0's table held only rank 2, and
       `peers_reporting_contradiction` returned (). The raise at
       scheduler.py `_admit_under_group_completion` was unreachable on the real
       carrier.

  (B5) Even parsed, the report was WITHDRAWN one lap later. The follower now
       counts the contradiction as `restored` and therefore ADMITS in the same
       pass; the admission pops the record and removes the rid from
       `waiting_queue`, and `pp_prefetch_completion_stamp` withdraws this
       rank's claims for any rid no longer in that queue. Measured: a report
       emitted on pass N became None on pass N+1.

Together those made the commit's own B3 input end in COMPLETE SILENCE plus a
licensed multi-chunk recompute -- worse than either ancestor (#939,
raenge-nie-uneins). These tests are the falsifiers for both halves.
"""

import unittest

from sglang.srt.managers import scheduler_pp_mixin as M
from sglang.srt.managers.pp_prefetch_completion import (
    CONTRADICTION,
    PENDING,
    peers_reporting_contradiction,
)
from sglang.test.test_utils import CustomTestCase

KEY = M._PP_PREFETCH_COMPLETION_KEY

# The weg1b6 specimen shape: the follower measured 100 materialized tokens
# against a 6008-token retract stamp -- 5908 beyond the one-chunk allowance.
B6_RID = "1e95e023"
PEER_COUNT = 4096


class _PS:
    def __init__(self, pp_rank):
        self.pp_rank = pp_rank


class _Tree:
    def __init__(self, table):
        self._table = table

    def completed_prefetch_tokens(self, rid):
        return self._table.get(rid)

    def prefetch_is_ongoing(self, rid):
        return False


class _Req:
    def __init__(self, rid):
        self.rid = rid


class _Holder:
    def __init__(self, pp_rank, queue, table):
        self.ps = _PS(pp_rank)
        self.tree_cache = _Tree(table)
        self.waiting_queue = [_Req(r) for r in queue]


class A_TheWireCarriesTheContradiction(CustomTestCase):
    """(B1) RED-FIRST on 1634bc3d28: the parse dropped the sentinel."""

    def test_a_contradiction_survives_the_wire_parse(self):
        wire = {KEY: ((B6_RID, CONTRADICTION, 1), (B6_RID, PEER_COUNT, 2))}
        facts = M.pp_prefetch_completion_facts_from_wire(wire)
        self.assertEqual(
            facts, ((B6_RID, CONTRADICTION, 1), (B6_RID, PEER_COUNT, 2))
        )

    def test_pending_and_numbers_are_unchanged(self):
        """The two readings that already worked must be byte-identical."""
        wire = {KEY: (("a", PENDING, 1), ("b", 512, 2), ("c", "garbage", 1))}
        self.assertEqual(
            M.pp_prefetch_completion_facts_from_wire(wire),
            (("a", PENDING, 1), ("b", 512, 2)),
        )

    def test_pp0_absorbs_it_and_the_gate_can_see_it(self):
        """The whole point: PP0's table carries the sentinel, so
        `peers_reporting_contradiction` -- the input of the group STOP -- is
        non-empty. On the parent this returned ()."""
        holder = _Holder(0, [], {})
        wire = {KEY: ((B6_RID, CONTRADICTION, 1), (B6_RID, PEER_COUNT, 2))}
        self.assertEqual(M.pp_note_prefetch_completion(holder, wire), 2)
        table = M.pp_prefetch_completion_table(holder)
        self.assertEqual(table[(B6_RID, 1)], CONTRADICTION)
        self.assertEqual(
            peers_reporting_contradiction(table, B6_RID, 3, own_rank=0), (1,)
        )


class B_AContradictionIsNeverWithdrawnAsStale(CustomTestCase):
    """(B5) RED-FIRST on 1634bc3d28: the follower's own admission deleted it.

    The withdrawal rule exists so PP0 never admits on a token COUNT nobody
    stands behind any more. A contradiction promises no capacity; withdrawing it
    converts a measured divergence into silence exactly one lap before the
    authority can act.
    """

    def _stamp(self, holder, incoming):
        out = {}
        M.pp_prefetch_completion_stamp(holder, incoming, out)
        return out.get(KEY)

    def test_the_report_is_emitted_while_the_rid_is_still_queued(self):
        holder = _Holder(1, [B6_RID], {B6_RID: 100})
        # Stand in for the witness: this rank is a follower by construction, so
        # `store_witness` returns the reported state rather than raising. The
        # original is RESTORED, never deleted -- deleting it removes the real
        # module function and poisons every later test in the process (caught by
        # the combined red-first run of this very file).
        original = M._pp_store_witness_contradicts
        M._pp_store_witness_contradicts = lambda h, req: req.rid == B6_RID
        try:
            self.assertEqual(
                self._stamp(holder, {}), ((B6_RID, CONTRADICTION, 1),)
            )
        finally:
            M._pp_store_witness_contradicts = original

    def test_it_survives_the_pass_after_the_follower_admits(self):
        """After the admission the fix itself causes, the rid is gone from
        `waiting_queue` and the record is popped -- the rank produces no fresh
        report. The claim already on the wire must NOT be withdrawn."""
        admitted = _Holder(1, [], {})
        self.assertEqual(
            self._stamp(admitted, {KEY: ((B6_RID, CONTRADICTION, 1),)}),
            ((B6_RID, CONTRADICTION, 1),),
        )

    def test_a_stale_COUNT_from_the_same_rank_is_still_withdrawn(self):
        """The exemption is exactly one value wide. A number is a capacity
        claim and keeps the staleness rule it was built for."""
        admitted = _Holder(1, [], {})
        self.assertIsNone(self._stamp(admitted, {KEY: ((B6_RID, 4096, 1),)}))

    def test_a_stale_PENDING_from_the_same_rank_is_still_withdrawn(self):
        admitted = _Holder(1, [], {})
        self.assertIsNone(self._stamp(admitted, {KEY: ((B6_RID, PENDING, 1),)}))

    def test_another_rank_s_entries_are_untouched_either_way(self):
        """A rank only ever withdraws its OWN claims (the pre-existing rule)."""
        admitted = _Holder(1, [], {})
        got = self._stamp(admitted, {KEY: ((B6_RID, 4096, 2), ("x", CONTRADICTION, 2))})
        self.assertEqual(got, ((B6_RID, 4096, 2), ("x", CONTRADICTION, 2)))


if __name__ == "__main__":
    unittest.main()
