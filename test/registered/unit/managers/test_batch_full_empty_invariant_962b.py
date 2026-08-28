# SPDX-License-Identifier: Apache-2.0
"""#962b: ``batch_is_full=True`` on an EMPTY running batch is a contradiction.

THE DEFECT, measured on boot 2 of window-flip-0828 (11:21:37Z, specimen
SPECIMEN-2026-08-28T1135Z-flip0828-boot2-RING-WAIT-WEDGE.txt). Three latched
declines, all of them this line and nothing else::

    gate=batch_full_or_empty_queue(batch_is_full=1,queue=4)   with running=0

``batch_is_full`` is a PASS-scoped memo. It is written at scheduler.py:9081
while the prefill adder walks the waiting queue ("I stopped admitting because
the seats ran out on THIS pass") -- legitimate for that pass -- onto
``running_batch``, an object whose lifetime is the whole SCHEDULER. Two
lifetimes on one field is the defect; the gate at :8639 reads it a round later
as if it still meant something.

WHY IT NEVER CLEARS AT running=0, which is what makes it self-sustaining
rather than merely stale. Every clear site the flag has is unreachable in
exactly the state that needs it:

  * :7043    a FINISH path -- needs a shrinking ``extend`` ``last_batch``,
             which the void destroys BEFORE this could run;
  * :9838    behind ``if not running_batch.is_empty()`` (caller :7636);
  * :10008   behind the same guard (caller :7665);
  * :12701   the retract-RPC, which DOES clear -- and is the in-tree
             PRECEDENT this fix follows;
  * the #888b relief at :8636 -- positionally on the path, but its producer
    ``_note_parked_carriers`` (:4119 via :7675) sits behind the SAME emptiness
    guard, so it contributes nothing at running=0. That is registered as
    #962b and is pinned open by ``test_the_888b_relief_is_still_inert_...``
    below rather than quietly fixed by accident.

So at running=0 -- precisely the state the requeue-without-finish family
(#797/#798/#791b/#971) leaves behind, since those paths release residents into
``waiting_queue`` without ever finishing them -- not one clear site can run,
and the memo latches until an external kill.

THE FIX IS ONE JUNCTION, NOT AN EIGHTH CLEAR SITE. At the gate read the memo
is only credible when the running batch is non-empty; ``batch_is_full=1`` AND
``is_empty()`` is not a state, it is a contradiction, and the gate re-derives
from the seat test instead of trusting it. That covers every requeue-without-
finish path, including ones not yet written, without any of them needing to
know the flag exists.

RED-FIRST, AND THE FALSIFIER DRIVES THE REAL DEFECT PATH. Not a hand-set
flag on a bare object: the flag is latched in the :9081 shape, the batch is
then destroyed through the void family (``_pp_void_own_batch``: releases +
``waiting_queue.append``, ``running_batch`` deliberately untouched), and the
REAL ``Scheduler._get_new_batch_prefill_raw`` is then called for the next
pass. The whole #797 void test family carries ZERO assertions on this flag
(measured: one occurrence of the name in test_pp_retracted_pass_void_797.py,
as a stub field, never asserted) -- which is how the defect shipped past it.
That hole is closed here.
"""

import types
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20)


# The gate's decline note, verbatim from the boot 2 log line this module
# exists to make impossible. Matched as a substring so the queue depth and
# flag value in the note stay free to change.
BOOT2_DECLINE = "gate=batch_full_or_empty_queue"


class _ProceededPastTheGate(Exception):
    """Raised by the stand-in delayer, which sits immediately BELOW the gate.

    A sentinel rather than a full pass: reaching the delayer is exactly the
    proposition under test ("the gate did not decline"), and stopping there
    keeps this module off the prefill adder, the tree cache and the pools --
    none of which the invariant touches.
    """


class _Req:
    def __init__(self, rid):
        self.rid = rid
        self.req_pool_idx = None
        self.extend_range = None
        self.retracted = False

    def reset_for_retract(self):
        self.retracted = True


class _Batch:
    """A ``running_batch``-shaped stand-in: the fields the gate junction reads.

    ``is_empty`` is the real semantics (schedule_batch.py:2969,
    ``len(self.reqs) == 0``) rather than a constant, because the invariant
    under test is a statement ABOUT that predicate.
    """

    def __init__(self, reqs=(), batch_is_full=False):
        self.reqs = list(reqs)
        self.batch_is_full = batch_is_full

    def is_empty(self):
        return len(self.reqs) == 0


class _StubPool:
    def free(self, *a, **k):
        pass


class _StubAllocator:
    def free(self, *a, **k):
        pass


def _void_holder(mb_id=1, size=3):
    """The #797d own-void holder (the 630/757/795 pattern, as in the 797 file)."""
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    h = types.SimpleNamespace(
        mbs=[None] * size,
        mb_metadata=[None] * size,
        chunked_req=None,
        waiting_queue=[],
        running_mbs=[None] * size,
        req_to_token_pool=_StubPool(),
        token_to_kv_pool_allocator=_StubAllocator(),
        _pp_chunked_req_before_by_slot=[None] * size,
    )
    h._pp_void_own_batch = types.MethodType(SchedulerPPMixin._pp_void_own_batch, h)
    return h


def _gate_holder(waiting_queue, allocatable, relief_calls=None):
    """A holder carrying exactly what the REAL gate junction reads.

    Everything above the gate in ``_get_new_batch_prefill_raw`` is made inert
    (no grammar queue, no hierarchical cache, no preemption, no hybrid SWA),
    so the only live decision in the call is the one under test. The #888b
    relief is bound as the REAL method, not stubbed: whether it fires at
    running=0 is itself an assertion of this module.
    """
    from sglang.srt.managers.scheduler import Scheduler

    h = types.SimpleNamespace(
        _admission_decline_note=None,
        grammar_manager=types.SimpleNamespace(has_waiting_grammars=lambda: False),
        enable_hierarchical_cache=False,
        server_args=types.SimpleNamespace(enable_flexkv=False),
        enable_priority_preemption=False,
        is_hybrid_swa=False,
        chunked_req=None,
        waiting_queue=waiting_queue,
        # #888b's two inputs, in the state a zero-resident pass leaves them:
        # no purity verdict was ever recorded, because the decode branch that
        # records one never ran.
        _parked_decode_verdict=(None, False),
        phase_flip_active_stack=None,
        parked_decode_set=types.SimpleNamespace(enabled=False),
        min_free_slots_delayer=types.SimpleNamespace(
            should_delay=lambda **kw: (_ for _ in ()).throw(_ProceededPastTheGate())
        ),
    )
    h._drain_prefetch_progress = lambda: {}
    h._take_uniform_head_inputs = lambda: None
    h.get_num_allocatable_reqs = lambda running_bs: allocatable
    if relief_calls is not None:
        h._note_parked_carriers = lambda rb, blocked: relief_calls.append(1)
    for name in (
        "_get_new_batch_prefill_raw",
        "_rederive_latched_batch_full",
        "_decode_forbidden_this_phase",
    ):
        setattr(h, name, types.MethodType(getattr(Scheduler, name), h))
    return h


def _run_gate(holder, running_batch):
    """Call the real gate. Returns ('declined', note) or ('admitted', note)."""
    try:
        batch, _ = holder._get_new_batch_prefill_raw(None, running_batch)
    except _ProceededPastTheGate:
        return "admitted", holder._admission_decline_note
    return "declined", holder._admission_decline_note


class BatchFullEmptyInvariant962b(unittest.TestCase):
    # ---------------------------------------------------------------- arm 1
    def test_the_boot2_state_is_not_a_decline(self):
        """ARM 1, THE INVARIANT. Boot 2's exact constellation, asserted.

        Empty running batch, flag latched, queue non-empty, seats available.
        Before the fix the gate returns at the flag -- ABOVE the seat test --
        and names itself ``batch_full_or_empty_queue``, which is the log line
        the specimen carries three times. After it, the memo is re-derived
        against the seat test and the pass proceeds.
        """
        rb = _Batch(reqs=[], batch_is_full=True)
        holder = _gate_holder(waiting_queue=[_Req("q0"), _Req("q1")], allocatable=4)

        verdict, note = _run_gate(holder, rb)

        self.assertEqual(
            verdict,
            "admitted",
            f"the gate declined at running=0 with seats free; note={note!r}",
        )
        self.assertNotIn(
            BOOT2_DECLINE,
            str(note),
            "boot 2's decline line survived the fix",
        )
        self.assertFalse(
            rb.batch_is_full,
            "the contradiction must be resolved on the batch, not merely "
            "stepped over -- the next reader reads the same field",
        )

    def test_the_seat_test_and_not_the_memo_is_the_authority(self):
        """Re-derive, do not clear. With NO seats free the gate must still
        decline -- the fix restores the question, it does not admit anything.

        This is the half that keeps the invariant from being a licence to
        over-admit: same latched-and-empty contradiction as arm 1, opposite
        seat verdict, opposite outcome.
        """
        rb = _Batch(reqs=[], batch_is_full=True)
        holder = _gate_holder(waiting_queue=[_Req("q0")], allocatable=0)

        verdict, note = _run_gate(holder, rb)

        self.assertEqual(verdict, "declined", f"note={note!r}")
        self.assertIn(BOOT2_DECLINE, str(note))
        self.assertTrue(
            rb.batch_is_full,
            "re-derived to True: the seats really are gone, so the memo's "
            "value was right even though its credibility was not",
        )

    def test_an_empty_queue_still_declines_for_its_own_reason(self):
        """The gate's OTHER disjunct is untouched. A pass with nothing waiting
        declines whatever the flag says, and must keep saying so."""
        rb = _Batch(reqs=[], batch_is_full=True)
        holder = _gate_holder(waiting_queue=[], allocatable=8)

        verdict, note = _run_gate(holder, rb)

        self.assertEqual(verdict, "declined")
        self.assertIn(BOOT2_DECLINE, str(note))

    # ---------------------------------------------------------------- arm 2
    def test_the_888b_relief_is_still_inert_at_zero_residents(self):
        """ARM 2, THE #962b PIN. The relief did NOT get fixed by accident.

        ``_rederive_latched_batch_full`` (#888b, :8636) sits one line above
        the junction this module fixes, and covers a DIFFERENT case (a
        non-empty seat table of carriers the phase forbids to run). Its
        producer ``_note_parked_carriers`` is reached only from :7675, behind
        the same ``not running_batch.is_empty()`` guard, so at running=0 it is
        never called and the relief reports False.

        Asserted, not assumed: if a later change makes the relief fire here,
        #962b has changed shape and this register entry must be re-read
        rather than silently closed.
        """
        rb = _Batch(reqs=[], batch_is_full=True)
        calls = []
        holder = _gate_holder(
            waiting_queue=[_Req("q0")], allocatable=4, relief_calls=calls
        )

        self.assertFalse(
            holder._rederive_latched_batch_full(rb),
            "the #888b relief claimed the latched flag at running=0 -- it "
            "cannot: its producer is unreachable in this state",
        )
        self.assertTrue(
            rb.batch_is_full,
            "the relief must not have cleared the flag; if it did, arm 1 "
            "would be passing for the wrong reason",
        )

        _run_gate(holder, rb)
        self.assertEqual(
            calls,
            [],
            "_note_parked_carriers ran on a zero-resident pass -- the "
            "emptiness guard at :7636/:7665 has moved",
        )

    # ---------------------------------------------------------------- arm 3
    def test_the_void_family_hole_the_requeue_without_finish_path(self):
        """ARM 3, THE REAL DEFECT PATH, end to end and in order.

        1. The adder latches the memo in the :9081 shape (seats ran out for
           THIS pass, on a batch that really did hold requests).
        2. The void family destroys the pass: ``_pp_void_own_batch`` releases
           the requests and appends them to ``waiting_queue``. No finish path
           runs -- that is what "requeue without finish" means, and it is why
           :7043 cannot clear the flag afterwards.
        3. The released requests are no longer resident, so the running batch
           the next pass sees holds zero requests while still carrying the
           memo. That is boot 2's state, reached the way boot 2 reached it.
        4. The next gate call must admit.

        The #797 family has no assertion on this flag anywhere; this is it.
        """
        reqs = [_Req("r0"), _Req("r1")]

        # (1) the writer's shape: a pass that admitted, then ran out of seats.
        running_batch = _Batch(reqs=list(reqs), batch_is_full=True)

        # (2) the void.
        void = _void_holder(mb_id=1)
        void.mbs[1] = types.SimpleNamespace(reqs=list(reqs))
        void._pp_admission_pass_voided = True
        void._pp_void_own_batch(1)

        self.assertEqual(
            [r.rid for r in void.waiting_queue],
            ["r0", "r1"],
            "the void must re-queue what it released -- if it did not, this "
            "test is not driving the defect path",
        )
        self.assertIsNone(void.mbs[1])

        # (3) released == no longer resident. No finish path ran, so the memo
        # is still standing on a batch that now holds nothing.
        requeued = {r.rid for r in void.waiting_queue}
        running_batch.reqs = [r for r in running_batch.reqs if r.rid not in requeued]
        self.assertTrue(running_batch.is_empty())
        self.assertTrue(
            running_batch.batch_is_full,
            "nothing on the void path clears the memo -- that is the defect, "
            "and this line is the assertion the 797 family never made",
        )

        # (4) the next pass.
        holder = _gate_holder(waiting_queue=void.waiting_queue, allocatable=4)
        verdict, note = _run_gate(holder, running_batch)

        self.assertEqual(
            verdict,
            "admitted",
            f"the requeued work was locked out by the memo the void left "
            f"behind; note={note!r}",
        )

    # ---------------------------------------------------------------- arm 4
    def test_green_guard_a_genuinely_full_seat_table_still_declines(self):
        """ARM 4. The #888b case must keep declining, on BOTH sides of the fix.

        Eight residents and no allocatable seats: the flag is credible here
        because the batch is not empty, so the invariant does not apply and
        the gate returns at the flag exactly as it always has. This is the
        direction that must stay byte-identical -- a fix that admitted here
        would be over-admission wearing a bug fix's name.
        """
        residents = [_Req(f"resident-{i}") for i in range(8)]
        rb = _Batch(reqs=residents, batch_is_full=True)
        holder = _gate_holder(waiting_queue=[_Req("q0")], allocatable=0)

        verdict, note = _run_gate(holder, rb)

        self.assertEqual(verdict, "declined")
        self.assertIn(BOOT2_DECLINE, str(note))
        self.assertTrue(
            rb.batch_is_full,
            "a credible memo on a non-empty batch must not be re-derived",
        )

    def test_green_guard_a_healthy_pass_never_reaches_the_invariant(self):
        """The ordinary case: flag not latched at all. Unchanged by construction,
        pinned so that "unchanged" is a measurement."""
        rb = _Batch(reqs=[_Req("resident")], batch_is_full=False)
        holder = _gate_holder(waiting_queue=[_Req("q0")], allocatable=4)

        verdict, _ = _run_gate(holder, rb)

        self.assertEqual(verdict, "admitted")
        self.assertFalse(rb.batch_is_full)

    # ---------------------------------------------------------------- arm 5
    def test_can_fail_neutering_the_emptiness_clause_restores_the_defect(self):
        """ARM 5, THE MUTANT. The emptiness clause is what does the work.

        The invariant is ``batch_is_full AND is_empty()``. Make ``is_empty``
        lie -- report False on a batch that holds nothing, which is precisely
        the pre-fix reading in which the memo is always credible -- and arms 1
        and 3 must come back red. If they stay green, they are green for some
        other reason and this module proves nothing.
        """

        class _NeverEmpty(_Batch):
            def is_empty(self):
                return False

        rb = _NeverEmpty(reqs=[], batch_is_full=True)
        holder = _gate_holder(waiting_queue=[_Req("q0"), _Req("q1")], allocatable=4)

        verdict, note = _run_gate(holder, rb)

        self.assertEqual(
            verdict,
            "declined",
            "with the emptiness clause neutered the gate must decline again "
            "-- if it admits, the fix is not the thing arm 1 measures",
        )
        self.assertIn(BOOT2_DECLINE, str(note))


if __name__ == "__main__":
    unittest.main()
