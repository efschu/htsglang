"""#631 defect R: the resident-carry leak, driven through the REAL PP loop.

THE DEFECT IN ONE SENTENCE
--------------------------
``last_mbs[slot]`` must name the batch that slot ran in its PREVIOUS
iteration; nesting its assignment under "did this slot run something"
turned it into "the last non-empty batch this slot EVER ran", and under
strict phase purity -- where a resident PP slot legitimately runs nothing
-- that stale entry is re-merged into the resident batch once per cycle,
forever. Measured on metal twice: the claimed resident count walked
5 -> 13338 -> ... -> 868447 at +1 per scheduler round while the same round
reported ``#running-req: 1``.

WHY THIS DRIVES THE SHIPPING LOOP AND NOT A MODEL OF IT
--------------------------------------------------------
``_event_loop_pp_body`` is taken unbound off the mixin, exactly as
``test_pp_flip_slot_hold_631`` does, so the statement ORDER under test --
read ``last_mbs[mb_id]``, plan, write ``mbs[mb_id]``, publish
``last_mbs[next_mb_id]`` -- is the real one. A model of that order would
have been circular: the order IS the defect. Only the two collaborators
that make the stale entry reachable and harmful are stubbed, and both are
stubbed to their production rule:

  * PURITY -- ``get_next_batch_to_run`` returns ``batch_to_run = None`` for
    a resident decode batch in the PP layout (``scheduler.py``,
    ``phase_decode_blocked_here`` -> ``ret = None``); and
  * MERGE -- a non-empty EXTEND ``last_batch`` distinct from
    ``running_batch`` reaches ``running_batch.merge_batch(last_batch)``
    (``scheduler.py`` :4435), which extends ``reqs`` IN PLACE.

AND SINCE #791, THE LOOP ALSO RUNS AN ADMISSION PASS (#815). This rank is
last-but-not-first, so it really receives a decision each round. Everything
in that pass is the SHIPPING code, taken unbound like the loop itself; the
harness supplies only the wire, as an EMPTY decision -- the value production
builds when a rank has nothing to report. Empty entries make the
reconciliation provably inert rather than accidentally inert, which is the
condition the resident-carry leak was measured under. Nine of those methods
run 100 times each per fixture; that was checked by counting calls, not
assumed, and methods that never fired were not left in as decoration.

THE CAN-FAIL LEVER
------------------
``_LeakyRank`` overrides ONE method -- ``_pp_record_slot_last_batch`` --
back to the pre-fix conditional, and nothing else. Same loop, same stubs,
one line of semantics swapped, so the contrast isolates the fix rather
than the harness. Keeping it is what stops an innocent refactor from
re-nesting the assignment and finding every test still green.
"""

import unittest
from collections import deque

from sglang.srt.managers.pp_admission_congruence import PPAdmissionDecision
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin


class _StopLoop(Exception):
    """Ends the deliberately infinite loop at a pass boundary."""


class _Batch:
    """A duck-typed ScheduleBatch: identity, a reqs list, a forward mode.

    ``merge_batch`` extends in place, as the real one does. That
    non-idempotence is the defect's amplifier, so the stub keeps it rather
    than quietly deduplicating and hiding the growth it exists to measure.
    """

    def __init__(self, reqs=(), extend=False):
        self.reqs = list(reqs)
        self.extend = extend

    def is_empty(self):
        return not self.reqs

    def merge_batch(self, other):
        self.reqs.extend(other.reqs)


class _Plan:
    def __init__(self, batch_to_run, running_batch):
        self.batch_to_run = batch_to_run
        self.running_batch = running_batch


class _PS:
    pp_size = 2
    # #815: rank 1 OF 2, which is what `_Group` below already says -- last but
    # not first. The two are one statement about one machine, so they are
    # written next to each other rather than left to agree by accident.
    pp_rank = 1


class _Group:
    # #815: `is_first_rank` is FALSE, and it has to be. f31fd5e43e [#791] gave
    # the loop a `not self.pp_group.is_first_rank` read, and with pp_size=2 a
    # rank that is last is not also first. Setting True to dodge the branch
    # would model a rank that is simultaneously the head and the tail of a
    # two-stage pipeline -- a machine that does not exist, and precisely the
    # convenient-instead-of-faithful stub this class of repair exists to avoid.
    #
    # Being faithful means the loop now really enters the #791 admission
    # receive, so the collaborators below are stubbed the same way this file
    # already stubs its other two: to their production rule.
    is_first_rank = False
    is_last_rank = True


class _ServerArgs:
    pp_async_batch_depth = 0
    enable_phase_flip = True


class _Receiver:
    def recv_requests(self):
        return []


class _Event:
    def synchronize(self):
        pass


class _Rank:
    """A rank whose LOOP CONTROL is the shipping code."""

    _event_loop_pp_body = SchedulerPPMixin._event_loop_pp_body
    _pp_flip_hold_slot = SchedulerPPMixin._pp_flip_hold_slot
    _pp_record_slot_last_batch = SchedulerPPMixin._pp_record_slot_last_batch
    # #815: taken unbound for the same reason the three above are -- the #791
    # admission pass is now on this rank's path, and a model of it would be
    # circular in exactly the way the docstring warns about. These are the
    # SHIPPING methods, not stand-ins; the harness supplies only the wire.
    _pp_reconcile_incoming_admission = SchedulerPPMixin._pp_reconcile_incoming_admission
    _pp_void_retracted_pass = SchedulerPPMixin._pp_void_retracted_pass
    _pp_forwarded_schedule_from = SchedulerPPMixin._pp_forwarded_schedule_from
    _pp_note_output_expectation = SchedulerPPMixin._pp_note_output_expectation
    _pp_note_chunked_req_before_admission = (
        SchedulerPPMixin._pp_note_chunked_req_before_admission
    )
    _pp_void_pass_without_upstream_launch = (
        SchedulerPPMixin._pp_void_pass_without_upstream_launch
    )
    _pp_send_admission_decision = SchedulerPPMixin._pp_send_admission_decision
    _pp_commit_admission_send_work = SchedulerPPMixin._pp_commit_admission_send_work
    _pp_drain_voided_proxy = SchedulerPPMixin._pp_drain_voided_proxy

    PP_LOOP_SIZE = 4
    CYCLES = 25

    def __init__(self):
        self.pp_loop_size = self.PP_LOOP_SIZE
        self.ps = _PS()
        self.pp_group = _Group()
        self.server_args = _ServerArgs()
        self.request_receiver = _Receiver()
        self.mbs = [None] * self.pp_loop_size
        self.last_mbs = [None] * self.pp_loop_size
        self.running_mbs = [_Batch() for _ in range(self.pp_loop_size)]
        self.mb_metadata = [None] * self.pp_loop_size
        self.send_proxy_work = []
        self.pp_outputs = None
        self.last_rank_comm_queue = None
        self.chunked_req = None
        self.launch_event = None
        self.cur_batch_for_debug = None
        self.running_batch = None
        self.last_batch = None
        self.armed = False
        self.passes = 0
        # #815: the state the #791 admission pass reads on the quiet path. All
        # three are the values a rank holds when nothing has been admitted and
        # no crossing wire is configured -- the condition under which the
        # resident-carry leak was measured, so they are the scenario's own
        # values rather than whatever silences the loop.
        self.waiting_queue = []
        self._pp_gapped_wire = False
        self._pp_admission_pending_sends = deque()

        # THE SEEDED STATE, and it is the state a completed prefill really
        # leaves in a purity-idle slot. The prefill's request is ALREADY
        # resident in running_mbs[0] -- that is what the first, correct
        # merge did -- while a DISTINCT extend batch still holds it. That
        # shape is what defeats both existing defences at once: the
        # self-merge guard compares ``last_batch is running_batch``, and
        # ``harvest_resident_batches`` dedupes by ``id(batch)``, so neither
        # can see a duplicate that is neither the same object nor a second
        # batch of different requests.
        self.req = object()
        self.running_mbs[0] = _Batch([self.req])
        self.last_mbs[0] = _Batch([self.req], extend=True)

    # -- what the loop calls -------------------------------------------

    def _pp_flip_pass_tick(self, mb_id):
        if mb_id == 0:
            self.passes += 1
            if self.passes > self.CYCLES:
                raise _StopLoop

    def _pp_forward_and_process_input_requests(self, recv_reqs):
        pass

    def get_next_batch_to_run(self, running_batch, last_batch):
        """The two production rules this defect rides on."""
        if last_batch is not None and last_batch.extend and last_batch.reqs:
            if last_batch is not running_batch:
                running_batch.merge_batch(last_batch)
        # STRICT PURITY: decode is forbidden in the PP layout, so a resident
        # decode batch yields no work. Returning None is the intended
        # signal, not a degenerate case -- and it is what leaves the slot's
        # ``mbs`` entry None on exactly the rounds that must still refresh
        # its ``last_mbs`` entry.
        return _Plan(None, running_batch)

    def _pp_recv_admission_decision(self):
        """THE ONLY NEW STUB, and it is stubbed to a value production builds.

        The real method reads a decision off the gloo wire, which this harness
        has none of. What it returns on the quiet path is an EMPTY decision --
        `PPAdmissionDecision(mb_id=..., entries=())` is what the shipping code
        itself constructs when a rank has nothing to report
        (scheduler_pp_mixin.py:1473 and :1556), not a shape invented here.

        Empty entries make the reconciliation below provably inert rather than
        incidentally inert: `reconcile_pp_admission_decision` loops over
        `decision.entries`, so with none it returns the decision unchanged. That
        is what keeps this file about the resident-carry leak: the #791 pass is
        present and real, and it narrows nothing, which is the condition under
        which the leak was measured on metal.
        """
        self._pp_output_expected_incoming = False
        self._pp_pass_voided_incoming = False
        self._pp_upstream_launched_incoming = False
        return PPAdmissionDecision(mb_id=0, entries=())

    def _pp_recv_proxy_tensors(self, mb_id):
        return None

    def _pp_commit_send_output_work_and_preprocess_output_tensors(self, a, b):
        return None, None, _Event()

    def _pp_commit_comm_work(self, work):
        pass

    def _pp_launch_batch(self, *args, **kwargs):
        return object(), None

    def _pp_process_batch_result(self, batch, result):
        pass

    def on_idle(self):
        pass

    def pp_phase_flip_armed(self):
        return False

    def _phase_flip_on_round(self, require_armed_and_parked=False):
        pass

    def run(self):
        try:
            self._event_loop_pp_body()
        except _StopLoop:
            pass
        return self

    @property
    def resident(self):
        return self.running_mbs[0].reqs


class _LeakyRank(_Rank):
    """The pre-fix rule, and nothing else changed. The can-fail lever."""

    def _pp_record_slot_last_batch(self, slot):
        if self.mbs[slot] is not None:
            self.last_mbs[slot] = self.mbs[slot]


class TestPPSlotLastBatch(unittest.TestCase):
    def test_can_fail_the_prefix_rule_leaks_once_per_cycle(self):
        rank = _LeakyRank().run()

        # The metal law, reproduced: +1 resident entry per cycle, unbounded.
        self.assertGreater(
            len(rank.resident),
            _Rank.CYCLES,
            "the pre-fix rule must reproduce the leak, or this harness has "
            "drifted from the loop it drives and the green arm proves nothing",
        )
        # It is DUPLICATION, not admission: still exactly one request.
        self.assertEqual(len({id(r) for r in rank.resident}), 1)
        self.assertIsNotNone(
            rank.last_mbs[0],
            "the stale extend batch is what gets re-merged; it must still be "
            "reachable under the pre-fix rule",
        )

    def test_the_shipped_rule_consumes_the_stale_batch_exactly_once(self):
        rank = _Rank().run()

        self.assertEqual(
            len(rank.resident),
            2,
            "the seeded entry was already published when the leak was "
            "seeded, so it may be consumed EXACTLY ONCE and never again",
        )
        self.assertIsNone(
            rank.last_mbs[0],
            "a slot that ran nothing must report 'nothing' as its previous "
            "batch rather than keep the previous answer",
        )

    def test_every_slot_reports_its_own_previous_round(self):
        """The rule is per slot, not only for the seeded one."""
        rank = _Rank().run()
        for slot in range(_Rank.PP_LOOP_SIZE):
            self.assertIsNone(rank.last_mbs[slot], f"slot {slot} kept a stale entry")


class TestResidentSetRepair(unittest.TestCase):
    """The containment half: a refusal must not outlive its condition."""

    class _FilterableBatch(_Batch):
        def filter_batch(self, keep_indices):
            self.reqs = [self.reqs[i] for i in keep_indices]

    def _scheduler_with(self, batch):
        class _S:
            pass

        sched = _S()
        sched.running_mbs = [batch, _Batch()]
        sched.running_batch = batch
        sched.mbs = [None, None]
        sched.last_mbs = [None, None]
        return sched

    def test_repair_removes_intra_batch_duplicates(self):
        from sglang.srt.managers.phase_flip_resident_carry import (
            repair_duplicate_resident_reqs,
        )

        req = object()
        batch = self._FilterableBatch([req] * 9)
        removed = repair_duplicate_resident_reqs(self._scheduler_with(batch))

        self.assertEqual(removed, 8)
        self.assertEqual(batch.reqs, [req])

    def test_repair_is_inert_on_a_merely_large_set(self):
        """A big set is not a corrupt one; only duplication is repairable."""
        from sglang.srt.managers.phase_flip_resident_carry import (
            repair_duplicate_resident_reqs,
        )

        class _NoFilter(_Batch):
            def filter_batch(self, keep_indices):  # pragma: no cover
                raise AssertionError("must not filter a duplicate-free batch")

        reqs = [object() for _ in range(9)]
        batch = _NoFilter(reqs)

        self.assertEqual(repair_duplicate_resident_reqs(self._scheduler_with(batch)), 0)
        self.assertEqual(batch.reqs, reqs)

    def test_describe_resident_slots_names_the_duplication(self):
        """The forensic line must distinguish duplication from admission."""
        from sglang.srt.managers.phase_flip_resident_carry import (
            describe_resident_slots,
        )

        req = object()
        line = describe_resident_slots(self._scheduler_with(_Batch([req] * 5)))

        self.assertIn("reqs=5", line)
        self.assertIn("uniq=1", line)


if __name__ == "__main__":
    unittest.main()
