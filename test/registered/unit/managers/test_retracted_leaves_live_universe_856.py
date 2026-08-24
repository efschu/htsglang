"""#856 W27 ROOT: a retracted request must LEAVE the live universe.

THE SPECIMEN. W27 (pin 3111539b3c, boot_w27_0824_1510.log) died on all three
ranks one second after the retraction ran correctly:

    15:14:44  PHASE-FLIP RESIDENTS RELEASED for pp_to_tp: 1 request(s)
              retracted and the prefix tree dropped, in that order (#856)
    15:14:45  resident_mamba_slots (gdn_flip_mover.py:620)
              KvReshardError: PHASE-FLIP-GDN live request
                da65cfe401934d8d847466d39ab6cf27 has no mamba slot
                -- refusing to flip past unmoved linear state

THE GUARD WAS RIGHT. The request really had no mamba slot -- the retraction
had just freed it -- and refusing to flip past unmoved linear state is exactly
what it should do. What was wrong is that the request was still being OFFERED
to it.

`retract_all` frees rows, the mamba slot and the tree lock ref. It does not
touch the scheduler's batch structures, and `_live_reqs` -- the one authority
for "who is resident" -- reads exactly four places: every `running_mbs` slot,
`running_batch`, `last_batch`, and the out-of-batch `chunked_req`. So after a
retraction every seam consumer still sees a live request whose resources are
gone.

SAME SHAPE AS #731's FIX: there the carry had to CONSUME the queue entry
instead of leaving one request counted twice. Freeing a resource and retiring
the reference to it are different jobs; doing only the first leaves a live
object every reader has to special-case.

FIXED AT THE AUTHORITY, NOT THE CONSUMERS. `resident_mamba_slots` is the first
reader to hit this and explicitly not expected to be the last -- teaching each
reader to skip freed requests would be the same defect once per reader, and
the next reader added reintroduces it. So the assertions below are about
`_live_reqs`, with the W27 falsifier pinned directly on top of it.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import types
import unittest

from sglang.srt.managers.phase_flip_runtime import (
    _live_reqs,
    consume_retracted_from_live_universe,
)
from sglang.test.test_utils import CustomTestCase

W27_RID = "da65cfe401934d8d847466d39ab6cf27"


class _Req:
    def __init__(self, rid, mamba=1):
        self.rid = rid
        self.mamba_pool_idx = mamba
        self.seqlen = 8
        self.req_pool_idx = 0

    def free_mamba(self):
        """What retraction does to it."""
        self.mamba_pool_idx = None


class _Batch:
    """Minimal stand-in with the one method the fix is required to use.

    `filter_batch(keep_indices=...)` is the real mechanism, because a batch
    carries per-request tensors alongside the list and a raw `.reqs` edit
    desynchronises them. Modelled here so the test fails if the fix ever
    reverts to mutating the list directly.
    """

    def __init__(self, reqs):
        self.reqs = list(reqs)
        self.filtered_with = None

    def filter_batch(self, chunked_req_to_exclude=None, keep_indices=None):
        self.filtered_with = list(keep_indices or [])
        self.reqs = [self.reqs[i] for i in self.filtered_with]


def _sched(*, mbs=None, running=None, last=None, chunked=None):
    return types.SimpleNamespace(
        running_mbs=mbs, running_batch=running, last_batch=last, chunked_req=chunked
    )


class TestTheW27SpecimenIsReproduced(CustomTestCase):
    """Before the fix, the retracted request is still live. That IS the bug."""

    def test_a_freed_request_is_still_enumerated_without_the_fix(self):
        req = _Req(W27_RID)
        sched = _sched(running=_Batch([req]))
        req.free_mamba()  # what retract_all does
        live = _live_reqs(sched)
        self.assertIn(req, live)
        self.assertIsNone(live[0].mamba_pool_idx)

    def test_that_is_exactly_what_the_gdn_guard_refuses(self):
        # The guard's own condition, reproduced: a live request with no mamba
        # slot. Asserting the CONDITION rather than importing the raiser keeps
        # this test about the live universe, which is where the defect is.
        req = _Req(W27_RID)
        req.free_mamba()
        sched = _sched(running=_Batch([req]))
        offending = [r for r in _live_reqs(sched) if r.mamba_pool_idx is None]
        self.assertEqual([r.rid for r in offending], [W27_RID])


class TestRetractedRequestsLeaveEveryLiveUniverse(CustomTestCase):
    """All four places `_live_reqs` reads. Missing one is missing the bug."""

    def test_running_batch(self):
        req, keep = _Req("gone"), _Req("stays")
        sched = _sched(running=_Batch([req, keep]))
        n = consume_retracted_from_live_universe(sched, [req])
        self.assertEqual(n, 1)
        self.assertEqual([r.rid for r in _live_reqs(sched)], ["stays"])

    def test_last_batch(self):
        req = _Req("gone")
        sched = _sched(last=_Batch([req]))
        consume_retracted_from_live_universe(sched, [req])
        self.assertEqual(_live_reqs(sched), [])

    def test_every_running_mbs_slot(self):
        # THE #631 defect-J SHAPE: `running_batch` names ONE microbatch slot
        # under event_loop_pp. A fix that cleaned only that one would leave the
        # request live in every other slot -- silently.
        req = _Req("gone")
        keep = _Req("stays")
        sched = _sched(mbs=[_Batch([keep]), _Batch([req]), _Batch([req, keep])])
        consume_retracted_from_live_universe(sched, [req])
        self.assertEqual(sorted(r.rid for r in _live_reqs(sched)), ["stays"])

    def test_the_out_of_batch_chunked_req(self):
        # #631 defect O: the chunked prefill is resident and in NO batch, which
        # is why `_live_reqs` enumerates it separately -- so it must be cleared
        # separately, or it stays live by that route alone.
        req = _Req("gone")
        sched = _sched(chunked=req)
        n = consume_retracted_from_live_universe(sched, [req])
        self.assertEqual(n, 1)
        self.assertIsNone(sched.chunked_req)
        self.assertEqual(_live_reqs(sched), [])

    def test_the_w27_specimen_end_to_end(self):
        # The whole point: after consuming, nothing live lacks a mamba slot,
        # so the guard that killed W27 has nothing to refuse.
        req = _Req(W27_RID)
        keep = _Req("healthy")
        sched = _sched(mbs=[_Batch([req, keep])], running=_Batch([req]), chunked=None)
        req.free_mamba()
        consume_retracted_from_live_universe(sched, [req])
        live = _live_reqs(sched)
        self.assertEqual([r.rid for r in live], ["healthy"])
        self.assertEqual([r for r in live if r.mamba_pool_idx is None], [])


class TestItUsesTheRealBatchMechanism(CustomTestCase):
    def test_filter_batch_is_used_not_a_raw_list_edit(self):
        # A batch carries per-request tensors beside the list; editing `.reqs`
        # directly desynchronises them. Pinned because the raw edit is the
        # tempting shortcut and it fails silently, later.
        req, keep = _Req("gone"), _Req("stays")
        b = _Batch([req, keep])
        consume_retracted_from_live_universe(_sched(running=b), [req])
        self.assertEqual(b.filtered_with, [1])


class TestItIsSafeAtTheSeam(CustomTestCase):
    """It runs with requests already parked; it may never abort a flip."""

    def test_no_targets_is_a_no_op(self):
        b = _Batch([_Req("a")])
        self.assertEqual(consume_retracted_from_live_universe(_sched(running=b), []), 0)
        self.assertIsNone(b.filtered_with)

    def test_nothing_matching_leaves_the_batch_untouched(self):
        # THE CAN-FAIL DIRECTION: an implementation that filtered
        # unconditionally would pass every removal test above while quietly
        # dropping requests it was never asked to drop.
        b = _Batch([_Req("a"), _Req("b")])
        consume_retracted_from_live_universe(_sched(running=b), [_Req("other")])
        self.assertIsNone(b.filtered_with)
        self.assertEqual(len(b.reqs), 2)

    def test_a_refusing_filter_batch_does_not_raise(self):
        class _Angry(_Batch):
            def filter_batch(self, chunked_req_to_exclude=None, keep_indices=None):
                raise RuntimeError("batch refused")

        req = _Req("gone")
        consume_retracted_from_live_universe(_sched(running=_Angry([req])), [req])

    def test_a_scheduler_missing_every_attribute_is_survivable(self):
        self.assertEqual(
            consume_retracted_from_live_universe(types.SimpleNamespace(), [_Req("x")]),
            0,
        )


if __name__ == "__main__":
    unittest.main()


class TestTheGdnMoverRetiresByConstruction(CustomTestCase):
    """#856 step 2: the GDN retirement is DERIVED, never a standalone edit.

    W27 refused a retry that would have dropped `GdnFlipMover.move()` from the
    flip path, because doing so with live linear state still present trades a
    LOUD CRASH for SILENT LINEAR-STATE LOSS.

    Downstream of the finding-2 fix that trade disappears. After the seam
    consumes the retracted requests AND drops the prefix tree, both halves of
    `flip_mamba_slots` -- resident requests' slots UNION the tree's
    checkpoints -- are empty, so the mover has nothing to move and retires BY
    CONSTRUCTION. No deletion, exactly as the KV mover was retired by emptying
    its input.

    THE ORDER IS LOAD-BEARING and that is the whole point: retiring the mover
    while requests are still live is the silent-loss trade; retiring it after
    the live universe is empty is a no-op. This asserts against the REAL
    `resident_mamba_slots`, the function that killed W27.
    """

    def test_the_real_guard_no_longer_refuses_after_the_consume(self):
        from sglang.srt.managers.gdn_flip_mover import resident_mamba_slots

        req = _Req(W27_RID)
        sched = _sched(running=_Batch([req]))
        req.free_mamba()
        consume_retracted_from_live_universe(sched, [req])
        slots = resident_mamba_slots(sched)
        self.assertEqual(list(slots), [], "nothing live, so nothing to move")

    def test_the_real_guard_STILL_refuses_without_the_consume(self):
        # THE CAN-FAIL DIRECTION, and it is the one that matters. If this ever
        # stops raising, the guard has been weakened rather than satisfied --
        # which is precisely the silent-loss trade W27's no-retry refused.
        from sglang.srt.layers.dcp.reshard_plan import KvReshardError
        from sglang.srt.managers.gdn_flip_mover import resident_mamba_slots

        req = _Req(W27_RID)
        sched = _sched(running=_Batch([req]))
        req.free_mamba()
        with self.assertRaises(KvReshardError) as caught:
            resident_mamba_slots(sched)
        self.assertIn("has no mamba slot", str(caught.exception))

    def test_a_live_request_WITH_its_slot_is_still_moved(self):
        # The guard must keep working for the case it exists for: a genuinely
        # resident request still yields its slot. An implementation that
        # returned empty unconditionally would pass both tests above while
        # losing every real linear state.
        from sglang.srt.managers.gdn_flip_mover import resident_mamba_slots

        sched = _sched(running=_Batch([_Req("healthy", mamba=7)]))
        self.assertEqual(list(resident_mamba_slots(sched)), [7])
