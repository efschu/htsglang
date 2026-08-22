# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#798: the admission reconcile is SLOT-BLIND, and that is the livelock.

THE MEASURED LIVELOCK, boot_798_0822_0829.log. That boot never served a
single request -- 0 decode batches, 0 throughput lines, 8 requests submitted
and 0 answered, health 200 throughout -- while emitting 5219 void lines in 13
minutes. Its shape:

  * TWO rids, not one. rid=f6116ba2 PROGRESSED (told=98304 local=90112, then
    told=100203 local=98304) and left. rid=48abbc0e reported
    ``told=98304 local=0`` 2212 consecutive times and never moved once.
  * Every own-void released "0 of 1 request", 2210 times: the single batch
    member is always KEPT by ``pp_void_keeps_request``, so it is never
    re-queued, so it never re-enters ``waiting_queue`` -- the one place the
    reconcile's radix lookup would have found it.
  * The voids alternate between slots without a single exception:
    2, 0, 2, 0, 2, 0, ...
  * The defensive ``self.chunked_req = None`` branch fired ZERO times, so it
    is not the cause.

THE GAP. ``_pp_reconcile_incoming_admission`` (scheduler_pp_mixin.py:4452,
called at :1300) answers "how much of this rid has THIS rank computed" from
the single scheduler-wide ``self.chunked_req``. The void path restores that
same field PER SLOT out of ``_pp_chunked_req_before_by_slot[mb_id]``
(:4740-4748). With more than one microbatch slot in flight, whatever stands
in ``self.chunked_req`` at reconcile time belongs to whichever slot last
wrote it -- not to the slot whose decision is being reconciled. The mismatch
is then taken as a MEASUREMENT (:4499-4505, ``local_match_lens[rid] = 0``),
the pass is retracted and voided, and the void restores the other slot's
snapshot. That is what makes the alternation self-sustaining.

AND THE SLOT IS ALREADY IN HAND. ``PPAdmissionDecision.mb_id`` exists and is
documented as "one PP microbatch slot" (pp_admission_congruence.py:208-219).
The reconcile receives the decision and ignores the field. This is not a
missing piece of information; it is an unread one.

WHY THIS IS #797c ONE LEVEL UP. #797c fixed the "dropped out of
waiting_queue, lives in chunked_req" miss, which is exactly why the
single-slot rid in the specimen progressed. It did not make the lookup
SLOT-aware, which is exactly why the two-slot rid could not.

Hermetic: no CUDA, no distributed, no spawned ranks. The reconcile is a pure
lookup over holder state, so the defect reproduces on a holder -- and a
harness that cannot even reach a second slot could not have shown this.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

from sglang.srt.managers.pp_admission_congruence import (
    PPAdmissionDecision,
    PPAdmissionEntry,
)
from sglang.srt.managers import scheduler_pp_mixin as spm
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

RING = 3
RANK = 1
PP_SIZE = 3

#: The specimen's numbers, kept verbatim so the test and the boot log can be
#: read against each other.
TOLD = 98_304
SLOT_UNDER_TEST = 2
OTHER_SLOT = 0

RID_STUCK = "48abbc0e"
RID_OTHER = "f6116ba2"


def _chunked_req(rid: str, computed: int):
    """A request mid-chunked-prefill, as the reconcile reads one.

    ``pp_chunked_local_match`` asks only for ``extend_range.end`` -- the
    absolute index this rank has computed up to -- so that is all this needs
    to carry. See that function's docstring for why the range end and not a
    radix re-match is the honest answer here.
    """
    return types.SimpleNamespace(
        rid=rid,
        extend_range=types.SimpleNamespace(start=0, end=computed),
        prefix_indices=[0] * computed,
    )


def _holder(*, chunked_req, by_slot):
    h = types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_size=PP_SIZE, pp_rank=RANK),
        # THE SPECIMEN'S STATE: the rid is in neither queue. It was kept by
        # pp_void_keeps_request rather than re-queued, 2210 times running.
        waiting_queue=[],
        tree_cache=None,
        chunked_req=chunked_req,
        _pp_chunked_req_before_by_slot=list(by_slot),
        running_mbs=[None] * RING,
    )
    h._pp_reconcile_incoming_admission = types.MethodType(
        SchedulerPPMixin._pp_reconcile_incoming_admission, h
    )
    return h


def _decision(mb_id: int, rid: str, told: int = TOLD):
    return PPAdmissionDecision(
        mb_id=mb_id,
        entries=(PPAdmissionEntry(rid=rid, prefix_len=told, extend_len=4096),),
    )


def _entry(decision, rid):
    return decision.by_rid()[rid]


class TestTheReconcileMustReadTheDecisionsSlot(unittest.TestCase):
    def test_a_chunked_request_in_another_slot_is_not_a_local_match_of_zero(self):
        """THE LIVELOCK, reproduced.

        Slot 2 holds a request this rank has computed 98304 tokens of. Slot 0's
        snapshot is what happens to be standing in ``self.chunked_req``,
        because slot 0's void wrote it there last. A decision for SLOT 2
        arrives. The rank has the KV; it must not report that it has none.
        """
        stuck = _chunked_req(RID_STUCK, TOLD)
        holder = _holder(
            chunked_req=_chunked_req(RID_OTHER, 4096),
            by_slot=[_chunked_req(RID_OTHER, 4096), None, stuck],
        )
        decision = _decision(SLOT_UNDER_TEST, RID_STUCK)

        effective, amended = holder._pp_reconcile_incoming_admission(decision)

        entry = _entry(amended, RID_STUCK)
        self.assertFalse(
            entry.retracted,
            "the rank has computed %d tokens of this rid and the decision "
            "names slot %d, where that request is held; retracting here is "
            "the false negative that voided 2212 passes in a row"
            % (TOLD, SLOT_UNDER_TEST),
        )
        self.assertIn(
            RID_STUCK,
            effective,
            "a rid this rank can honour must stay admissible for the pass",
        )

    def test_the_false_measurement_is_reported_as_zero(self):
        """The specimen's exact signature, asserted as a signature.

        ``observed_local`` documents itself as "the retracting rank's ACTUAL
        local match length ... never a guess". Under the slot-blind lookup it
        carries 0 for a rank holding 98304 tokens, and #630's congruence guard
        LEARNS from that number -- so the defect does not merely void one
        pass, it teaches PP0 a false shortfall.
        """
        stuck = _chunked_req(RID_STUCK, TOLD)
        holder = _holder(
            chunked_req=None,
            by_slot=[None, None, stuck],
        )
        decision = _decision(SLOT_UNDER_TEST, RID_STUCK)

        _effective, amended = holder._pp_reconcile_incoming_admission(decision)

        entry = _entry(amended, RID_STUCK)
        self.assertNotEqual(
            entry.observed_local,
            0,
            "observed_local=0 for a rank holding %d tokens is precisely the "
            "'told=%d local=0' line the boot repeated 2212 times" % (TOLD, TOLD),
        )


class TestTheGuardMustStillBeAbleToFire(unittest.TestCase):
    """A fix that simply stopped retracting would pass the tests above and
    reintroduce the mispair #791 exists to prevent. Each of these is a
    dying mutant for that fix."""

    def test_a_rid_no_slot_holds_is_still_retracted(self):
        """Genuinely absent everywhere -> the retraction is CORRECT."""
        holder = _holder(chunked_req=None, by_slot=[None, None, None])
        decision = _decision(SLOT_UNDER_TEST, RID_STUCK)

        effective, amended = holder._pp_reconcile_incoming_admission(decision)

        entry = _entry(amended, RID_STUCK)
        self.assertTrue(
            entry.retracted,
            "a rank that really has nothing for this rid MUST retract, or the "
            "pass it runs is a strict subset of the one already launched",
        )
        self.assertEqual(entry.observed_local, 0)
        self.assertNotIn(RID_STUCK, effective)

    def test_a_short_local_match_is_still_retracted(self):
        """Present in the right slot, but SHORT. Still unhonourable."""
        short = _chunked_req(RID_STUCK, TOLD - 4096)
        holder = _holder(chunked_req=None, by_slot=[None, None, short])
        decision = _decision(SLOT_UNDER_TEST, RID_STUCK)

        _effective, amended = holder._pp_reconcile_incoming_admission(decision)

        entry = _entry(amended, RID_STUCK)
        self.assertTrue(
            entry.retracted,
            "slot-awareness must make the measurement TRUE, not permissive",
        )
        self.assertEqual(
            entry.observed_local,
            TOLD - 4096,
            "and the shortfall #630 learns must be the real one",
        )

    def test_the_wrong_slot_is_not_consulted_as_a_fallback(self):
        """Reading ANY slot that happens to hold the rid would be the same
        class of bug with a wider blast radius: it would answer a question
        about slot 2 with slot 0's progress. The decision names its slot;
        only that slot may answer for it."""
        stale = _chunked_req(RID_STUCK, TOLD)
        holder = _holder(chunked_req=None, by_slot=[stale, None, None])
        decision = _decision(SLOT_UNDER_TEST, RID_STUCK)

        _effective, amended = holder._pp_reconcile_incoming_admission(decision)

        entry = _entry(amended, RID_STUCK)
        self.assertTrue(
            entry.retracted,
            "slot %d holds nothing for this rid; slot %d's progress is not an "
            "answer about slot %d" % (SLOT_UNDER_TEST, OTHER_SLOT, SLOT_UNDER_TEST),
        )


class TestTheShippedPathsAreUnchanged(unittest.TestCase):
    def test_the_single_slot_chunked_request_still_matches(self):
        """#797c's own case: ``self.chunked_req`` IS the request. This is the
        path the specimen's OTHER rid took successfully, and it must keep
        working whether or not the slot ring agrees."""
        stuck = _chunked_req(RID_STUCK, TOLD)
        holder = _holder(chunked_req=stuck, by_slot=[None, None, None])
        decision = _decision(SLOT_UNDER_TEST, RID_STUCK)

        _effective, amended = holder._pp_reconcile_incoming_admission(decision)

        self.assertFalse(
            _entry(amended, RID_STUCK).retracted,
            "#797c's lookup must not be regressed by adding the slot lookup",
        )

    def test_a_waiting_queue_request_still_uses_its_radix_match(self):
        """The ordinary path: in ``waiting_queue``, matched against the tree.

        ``init_next_round_input`` is stubbed to publish a full match, which is
        the only thing the reconcile reads off it (``len(prefix_indices)``).
        """
        req = types.SimpleNamespace(rid=RID_STUCK, prefix_indices=[])

        def _match(tree_cache):
            req.prefix_indices = [0] * TOLD

        req.init_next_round_input = _match
        holder = _holder(chunked_req=None, by_slot=[None, None, None])
        holder.waiting_queue = [req]
        decision = _decision(SLOT_UNDER_TEST, RID_STUCK)

        _effective, amended = holder._pp_reconcile_incoming_admission(decision)

        self.assertFalse(
            _entry(amended, RID_STUCK).retracted,
            "the radix path must keep precedence for a queued request",
        )

    def test_pp_size_one_is_untouched(self):
        holder = _holder(chunked_req=None, by_slot=[None, None, None])
        holder.ps = types.SimpleNamespace(pp_size=1, pp_rank=0)
        decision = _decision(SLOT_UNDER_TEST, RID_STUCK)

        effective, amended = holder._pp_reconcile_incoming_admission(decision)

        self.assertEqual(effective, {})
        self.assertIs(amended, decision)

    def test_a_missing_slot_ring_does_not_raise(self):
        """A holder predating the ring (or a stand-in) must degrade to the
        shipped behaviour, not to an AttributeError on the admission path."""
        holder = _holder(chunked_req=None, by_slot=[None, None, None])
        del holder._pp_chunked_req_before_by_slot
        decision = _decision(SLOT_UNDER_TEST, RID_STUCK)

        _effective, amended = holder._pp_reconcile_incoming_admission(decision)

        self.assertTrue(_entry(amended, RID_STUCK).retracted)

    def test_an_out_of_range_mb_id_does_not_raise(self):
        holder = _holder(chunked_req=None, by_slot=[None, None, None])
        decision = _decision(RING + 5, RID_STUCK)

        _effective, amended = holder._pp_reconcile_incoming_admission(decision)

        self.assertTrue(_entry(amended, RID_STUCK).retracted)


class TestNeuteringTheSlotLookupRestoresTheDefect(unittest.TestCase):
    """THE DYING MUTANT for the one call edge this change adds.

    Blinding `_pp_chunked_req_for_slot` to None is exactly the lookup that
    shipped before #798. If the defect does NOT come back under that neuter,
    then the passing tests above are passing for some other reason and this
    fix is not the thing doing the work.
    """

    def test_blinding_the_slot_lookup_brings_the_livelock_back(self):
        stuck = _chunked_req(RID_STUCK, TOLD)
        holder = _holder(chunked_req=None, by_slot=[None, None, stuck])

        with mock.patch.object(
            spm, "pp_chunked_req_for_slot", lambda holder, mb_id: None
        ):
            _effective, amended = holder._pp_reconcile_incoming_admission(
                _decision(SLOT_UNDER_TEST, RID_STUCK)
            )

        entry = _entry(amended, RID_STUCK)
        self.assertTrue(
            entry.retracted,
            "with the slot lookup blinded the rank must fall back to the "
            "false negative -- if it does not, this test file is not "
            "measuring the fix it claims to measure",
        )
        self.assertEqual(
            entry.observed_local,
            0,
            "and the false measurement must be the specimen's exact 0",
        )

    def test_the_shipped_chunked_lookup_is_a_separate_edge(self):
        """#797c's lookup must survive the neuter, or a single proof could
        not tell the two lookups apart."""
        stuck = _chunked_req(RID_STUCK, TOLD)
        holder = _holder(chunked_req=stuck, by_slot=[None, None, None])

        with mock.patch.object(
            spm, "pp_chunked_req_for_slot", lambda holder, mb_id: None
        ):
            _effective, amended = holder._pp_reconcile_incoming_admission(
                _decision(SLOT_UNDER_TEST, RID_STUCK)
            )

        self.assertFalse(
            _entry(amended, RID_STUCK).retracted,
            "#797c's edge is independent and must not be taken down with it",
        )


if __name__ == "__main__":
    unittest.main()
