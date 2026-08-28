"""#968b RED-FIRST, written against the PRE-FIX API only.

WHY THIS FILE EXISTS BESIDE `test_pp_continuation_cross_slot_rehome_968b.py`.
That suite is the fix's own suite and it is a good one -- but it was written
WITH the fix, in one commit, so its red-first proof is the weak kind: run it
against the pre-fix tree and 28 of its 31 arms die with `AttributeError:
module 'sglang.srt.managers.scheduler_pp_mixin' has no attribute
'pp_rehome_displaced_chunked_req'`. A suite that red-fails on a MISSING
SYMBOL proves only that it needs the symbol. It would go green against a fix
that defined the three helpers as `return False` and changed no behaviour at
all, which is precisely the shape the indicator law exists to catch.

THIS FILE NAMES NO NEW SYMBOL. Every name it touches predates the fix:
`_pp_absorb_void_output`, `_pp_void_own_batch`,
`_pp_note_chunked_req_before_admission`, `pp_request_locations` (#946's
four-place reader) and `_PP_VOID_OUTPUT_KEY`. It therefore imports, collects
and RUNS on both trees, and the difference between them is a difference in
BEHAVIOUR:

    pre-fix  b07029698d : RED  -- the continuation is in none of the four
                                 places after the second slot's restore
    post-fix 4b204740a0 : GREEN -- it is reachable again

MEASURED, both trees, CUDA_VISIBLE_DEVICES="" and PYTHONPATH set to the tree
under test. See the register entry [test-agent flip-0828] of 2026-08-28.

THE MECHANISM, which is the whole of #968b in six lines. A retraction voids
EVERY slot back to back, and both void sites restore `self.chunked_req` from
their OWN slot's snapshot:

    carried_slots = getattr(self, "_pp_chunked_req_before_by_slot", None)
    chunked_before = carried_slots[mb_id] ...
    if getattr(self, "chunked_req", None) is not chunked_before:
        self.chunked_req = chunked_before

`self.chunked_req` is ONE field and the loop runs once per slot, so the last
slot wins. A continuation that was slot 0's carried chunk is overwritten by
slot 1's `None`, and it is not in the waiting queue by construction (the
void's disposal keeps it out, :798-802, and `pp_rehome_refused_chunked_req`
says so in words: "the one request that is never in the waiting queue, which
is exactly why it is the one that goes missing"). One pass later the slot
ring is rewritten and the last reference is gone.
"""

import types
import unittest
from array import array

import torch

from sglang.srt.managers import scheduler_pp_mixin as m
from sglang.srt.managers.scheduler_pp_mixin import (
    _PP_VOID_OUTPUT_KEY,
    SchedulerPPMixin,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

WORLD = 3
SLOT_A, SLOT_B = 0, 1

# The boot-4 specimen's rid and geometry.
RID_CONT = "1fa3f808000000000000000000000000"
RID_OTHER = "4077b704000000000000000000000000"
EXECUTED = 4096
TOTAL = 8422
# Boot 5's own numbers, from the `#968 MINT none` line that named the
# occupant this re-home refused: end=7939 against prefix=4096.
UNPARKED_END = 7939


def _req(rid, *, prefix_len, extend_len):
    """A real `Req` in the shape the adder leaves it in."""
    req = Req.__new__(Req)
    req.rid = rid
    fill = list(range(TOTAL))
    req.origin_input_ids = array("q", fill)
    req.output_ids = array("q")
    req.full_untruncated_fill_ids = array("q", fill)
    req.prefix_indices = torch.arange(prefix_len, dtype=torch.int64)
    req.extend_range = Range(prefix_len, prefix_len + extend_len)
    req.req_pool_idx = 0
    req.inflight_middle_chunks = 1
    req.cache_protected_len = prefix_len
    req.is_retracted = False
    req.finished_reason = None
    return req


def _parked(rid=RID_CONT, executed=EXECUTED):
    """The #971 park shape: the range ENDS where the prefix ends."""
    return _req(rid, prefix_len=executed, extend_len=0)


def _unparked(rid=RID_CONT, executed=EXECUTED, end=UNPARKED_END):
    """The boot-5 shape: a MID-PLAN occupant, prepared but never run.

    Not parked, so `extend_range.end` (7939) is past the prefix (4096). This
    is what an occupant looks like when it became `chunked_req` through the
    ordinary mint after a BUILT batch, rather than through a park.
    """
    return _req(rid, prefix_len=executed, extend_len=end - executed)


class _StubPool:
    """`req_to_token_pool` as far as the park actuator reads it."""

    def __init__(self, rows=1, cols=TOTAL):
        self.req_to_token = torch.arange(rows * cols, dtype=torch.int64).view(
            rows, cols
        )
        self.freed_req = []

    def free(self, req):
        self.freed_req.append(req)


class _StubAllocator:
    def __init__(self):
        self.freed = []

    def free(self, indices):
        self.freed.append(indices.clone())


def _scheduler():
    """A holder carrying exactly what the two void sites read.

    `batch.reqs` is EMPTY in every arm below, which is what keeps this
    hermetic: the disposal loop is the only caller of
    `_release_voided_request`, so with no requests to dispose there is no
    tree cache, no token pool and no allocator on the path at all.
    """
    h = types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_rank=0, pp_size=WORLD),
        pp_group=types.SimpleNamespace(is_last_rank=False, is_first_rank=True),
        chunked_req=None,
        waiting_queue=[],
        running_mbs=[None] * WORLD,
        mbs=[None] * WORLD,
        mb_metadata=[None] * WORLD,
        pp_loop_size=WORLD,
        _pp_admission_guard=None,
        _pp_chunked_req_before_by_slot=[None] * WORLD,
        _pp_launched_chain_by_slot=[None] * WORLD,
        _pp_idle_void_suppress_log=False,
        _pp_void_forward_payload=None,
        req_to_token_pool=_StubPool(),
        token_to_kv_pool_allocator=_StubAllocator(),
        tree_cache=None,
    )
    for name in (
        "_pp_absorb_void_output",
        "_pp_void_own_batch",
        "_pp_note_chunked_req_before_admission",
    ):
        setattr(h, name, types.MethodType(getattr(SchedulerPPMixin, name), h))
    return h


def _snapshot(h, slot, req):
    """One pass's top-of-pass snapshot for `slot`, through the shipped writer."""
    h.chunked_req = req
    h._pp_note_chunked_req_before_admission(slot)


def _empty_batch():
    return types.SimpleNamespace(reqs=[])


def _absorb(h, slot):
    return h._pp_absorb_void_output(
        slot, {_PP_VOID_OUTPUT_KEY: True}, h.mbs, h.mb_metadata
    )


def _located(h):
    """The #946 four-place reader: every rid this scheduler can still find."""
    return set(m.pp_request_locations(h))


class CrossSlotDisplacementIsARealLoss(unittest.TestCase):
    """The defect, in the shipped functions, named by pre-fix API only."""

    def _two_slot_retraction(self, drive):
        """Slot A carries the continuation; slot B carries nothing.

        Then ONE retraction voids both, in the order production voids them,
        and the next pass rewrites slot A's snapshot. `drive` is the void
        site under test, so the same specimen runs against the void-output
        site and against its own-void twin.
        """
        h = _scheduler()
        cont = _parked()

        _snapshot(h, SLOT_A, cont)
        _snapshot(h, SLOT_B, None)
        self.assertIn(
            RID_CONT,
            _located(h),
            "precondition: before the retraction the continuation IS "
            "reachable -- it is slot A's carried chunk",
        )

        h.mbs[SLOT_A] = _empty_batch()
        h.mbs[SLOT_B] = _empty_batch()
        drive(h, SLOT_A)
        drive(h, SLOT_B)

        # The next pass that visits slot A overwrites its snapshot with
        # whatever `chunked_req` now is. Production does this every pass,
        # immediately before `get_next_batch_to_run`, so a request whose only
        # remaining reference is a stale ring entry has at most `pp_size`
        # passes to live. Asserting before this line would credit the ring
        # with a permanence it does not have.
        h._pp_note_chunked_req_before_admission(SLOT_A)
        return h, cont

    def test_the_void_output_site_must_not_lose_the_continuation(self):
        """RED pre-fix: slot B's restore overwrites slot A's continuation."""
        h, _cont = self._two_slot_retraction(_absorb)

        self.assertIn(
            RID_CONT,
            _located(h),
            "THE #968b DEFECT: `self.chunked_req` is one field and the void "
            "loop writes it once per slot, so slot B's `None` displaced slot "
            "A's continuation. It is in no queue (the void's disposal keeps "
            "it out) and the ring entry that held it has been rewritten, so "
            "it is now in NONE of the four places `pp_request_locations` "
            "enumerates -- the 407-extra/107-missing signature of boot 4. "
            "This assertion is the fix, stated as behaviour and naming no "
            "symbol the fix introduces.",
        )

    def test_the_own_void_twin_site_must_not_lose_it_either(self):
        """The twin carries the identical eight lines, so it loses it too."""

        def drive(h, slot):
            return h._pp_void_own_batch(slot)

        h, _cont = self._two_slot_retraction(drive)

        self.assertIn(
            RID_CONT,
            _located(h),
            "the own-void twin restores from its own slot with the same "
            "eight lines, so it displaces the same way -- a fix applied to "
            "only one of the two sites leaves this arm red",
        )

    def test_the_executed_prefix_is_never_discarded_by_the_repair(self):
        """Whatever the repair does, it must not throw away executed tokens.

        The #963 floor and the no-double-prefill rule: 4096 tokens were
        computed and cached, and a repair that re-queued the request with an
        empty prefix would buy reachability with a second prefill. Green on
        BOTH trees when it holds -- it is the guard rail on the fix, not a
        statement about the defect.
        """
        h, cont = self._two_slot_retraction(_absorb)
        found = m.pp_request_locations(h).get(RID_CONT)
        if found is None:
            self.skipTest(
                "the continuation was lost -- that is the defect, and "
                "`test_the_void_output_site_must_not_lose_the_continuation` "
                "is the arm that reports it; this arm has nothing to measure"
            )
        self.assertIs(found, cont, "the SAME request object, not a rebuild")
        self.assertEqual(
            len(found.prefix_indices),
            EXECUTED,
            "the executed prefix must survive the repair intact",
        )
        self.assertFalse(
            getattr(found, "is_retracted", False),
            "a re-homed continuation must not be handed on in the "
            "reset_for_retract shape the next pass cannot read",
        )


class TheRehomeMustNotASSUMETheParkedShape(unittest.TestCase):
    """#968b-2, boot 5's third exit -- and it is the SAME class as #968b.

    The fix that closed the cross-slot displacement carried the defect class
    it was closing. `pp_rehome_displaced_chunked_req` gated its re-home on
    `extend_range.end == len(prefix_indices)` -- the PARKED shape -- and
    returned None for anything else. Boot 5's occupant was not parked: it
    became `chunked_req` through the ordinary mint after a BUILT batch, so
    end=7939 against prefix=4096, named verbatim by its own `#968 MINT none`
    line. The re-home refused it and the caller's next statement nulled the
    live occupant. A DROP, by the code written to prevent drops.

    THE CLASS, since it is the point: a predicate was ASSUMED where a
    predicate already existed to be ASKED. `pp_chunked_req_is_reachable`
    sits four lines below and TESTS reachability; the equality above GUESSED
    that only a parked request was worth keeping. Same shape as the act
    docstring that cited two line ranges it had never checked.

    RED at 4b204740a0, GREEN at 646f41087d, same file, same fixture.

    THE SPECIMEN HAD TO BE THE RIGHT ONE, and the first attempt was not --
    recorded because the wrong one is GREEN ON BOTH TREES and would have
    been reported as a passing red-first proof. Driving the two-slot
    cross-slot displacement of `CrossSlotDisplacementIsARealLoss` with an
    un-parked occupant does NOT reach the refusing gate: the voided slot's
    own `_park_chunked_prefill_chunk` runs first (the `#791b` line prints
    `chunk parked=True`), so by the time the SECOND slot's restore asks the
    re-home, the request is already in the parked shape and the equality
    holds even before the fix.

    The shape that reaches the gate un-parked is the one boot 5 had: the
    slot carried NOTHING at the top of the pass, the pass then minted a
    mid-plan occupant into `self.chunked_req` the ordinary way, and a single
    void on that slot asks the re-home to keep it. `incoming` is the slot's
    empty snapshot, `current` is the un-parked occupant, nothing has parked
    it, and pre-fix the gate refuses and the next statement nulls it.
    """

    def _minted_then_voided(self, drive):
        """Boot 5's shape: empty snapshot, ordinary mint, then one void."""
        h = _scheduler()
        _snapshot(h, SLOT_A, None)
        cont = _unparked()
        # What `new_chunked_req` / `add_chunked_req` leave behind on a pass
        # that BUILT a batch: an occupant that no park has touched.
        h.chunked_req = cont
        self.assertIn(RID_CONT, _located(h), "precondition: reachable before")

        h.mbs[SLOT_A] = _empty_batch()
        drive(h, SLOT_A)
        h._pp_note_chunked_req_before_admission(SLOT_A)
        return h, cont

    def test_an_unparked_mid_plan_occupant_is_not_dropped_by_the_void_output(self):
        h, _cont = self._minted_then_voided(_absorb)
        self.assertIn(
            RID_CONT,
            _located(h),
            "the occupant was MID-PLAN (end=7939, prefix=4096), not parked. "
            "A re-home that keeps only the parked shape drops it -- boot 5's "
            "third exit, and the drop is performed by the anti-drop code "
            "itself. Pre-fix the four-place reader returns set()",
        )

    def test_the_own_void_route_refuses_it_the_same_way(self):
        """Both call sites pass `route=`; both asked the same bad question."""

        def drive(h, slot):
            return h._pp_void_own_batch(slot)

        h, _cont = self._minted_then_voided(drive)
        self.assertIn(
            RID_CONT,
            _located(h),
            "own-void-cross-slot refuses identically -- a fix at only the "
            "void-output site leaves this red",
        )

    def test_the_giveback_happens_before_the_queueing(self):
        """It may be kept only AFTER its never-run tail is given back.

        Queueing a mid-plan occupant while `extend_range` still points past
        the prefix would leave the next round's unconditional stash caching a
        chunk no rank ran -- the double-prefill the park exists to prevent.
        So reachability alone is not the whole assertion: the request must
        arrive in the parked shape.
        """
        h, cont = self._minted_then_voided(_absorb)
        found = m.pp_request_locations(h).get(RID_CONT)
        if found is None:
            self.skipTest(
                "dropped -- `test_an_unparked_mid_plan_occupant_is_not_"
                "dropped_by_the_void_output` is the arm that reports that"
            )
        self.assertIs(found, cont, "the same object, not a rebuild")
        self.assertEqual(
            found.extend_range.end,
            len(found.prefix_indices),
            "parked on the way through: the never-run tail [4096:7939) is "
            "given back, so the next round's stash is a no-op on it",
        )
        self.assertEqual(
            len(found.prefix_indices),
            EXECUTED,
            "and the 4096 EXECUTED tokens are untouched -- the give-back "
            "frees only what this round allocated, never the radix tree's",
        )
        self.assertFalse(
            getattr(found, "is_retracted", False),
            "parked, not reset: a reset request cannot be read by the next "
            "pass's get_next_batch_to_run",
        )


class ASingleSlotVoidWasNeverBroken(unittest.TestCase):
    """The legacy path, pinned on both trees: no displacement, no change.

    If this arm ever goes red, the repair changed behaviour on the path a
    healthy boot actually takes, which no part of #968b is allowed to do.
    """

    def test_one_slot_carrying_the_chunk_keeps_it(self):
        h = _scheduler()
        cont = _parked()
        _snapshot(h, SLOT_A, cont)
        h.mbs[SLOT_A] = _empty_batch()

        _absorb(h, SLOT_A)

        self.assertIs(
            h.chunked_req,
            cont,
            "a single-slot void restores its own carried chunk and there is "
            "no second slot to displace it -- unchanged before and after",
        )
        self.assertIn(RID_CONT, _located(h))

    def test_a_void_with_no_carried_chunk_queues_nothing(self):
        h = _scheduler()
        _snapshot(h, SLOT_A, None)
        h.mbs[SLOT_A] = _empty_batch()

        _absorb(h, SLOT_A)

        self.assertIsNone(h.chunked_req)
        self.assertEqual(
            h.waiting_queue,
            [],
            "nothing was displaced, so nothing may be re-homed -- a repair "
            "that queues on the empty case would inject a phantom request",
        )


if __name__ == "__main__":
    unittest.main()
