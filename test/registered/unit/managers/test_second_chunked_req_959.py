"""#959: two chunked continuations at once, on the post-flip TP path.

WHAT DIED. window-955-boot's second boot (pin 27bcb4884f,
boot_943bx_27bcb4884f_0828_025835.log) took `assert self.chunked_req is None`
(scheduler.py, `_get_new_batch_prefill_raw`) on ALL THREE ranks at 03:03:29 --
three seconds after `PHASE-FLIP DONE pp_to_tp (epoch 1) in 9709.5 ms`, the
first clean cutover this family has ever produced, with 9 `phase=tp` batches
already run. The line immediately before the crash, on every rank:
`PHASE-FLIP armed (tp_to_pp) but NOT QUIESCENT: a chunked prefill is
incomplete` -- so a continuation was demonstrably resident when a second one
was minted.

WHY THE GUARD THAT EXISTS DID NOT COVER IT. `phase_flip_runtime.py:1853-1856`
does clear `scheduler.chunked_req` at the seam, but only for a request whose
`id()` is in the RETRACTED target set. A continuation that survives the flip
un-retracted -- which is the designed behaviour, see `chunk_blocks_quiescence`
-- is not in that set and is not cleared.

WHY #951 IS NOT THE FIX AND SAID SO ITSELF. Its own comment at the assert
records that the invariant is held by ARITHMETIC, not by a check: a surviving
continuation has normally spent all of `rem_chunk_tokens`, so the fresh-request
chunk branch computes `trunc_len <= 0` and never mints. It then records that
this is BREAKABLE, with witnesses, and that #951 closes only the PP instance (a
#798-voided pass no longer runs the function) -- "It does NOT close the general
case, which needs its own posten and its own danger-direction analysis." This
file is that posten.

AND window-951's GREEN ON THIS LINE WAS VACUOUS. It measured 0/0 on a path no
batch entered: every batch in that window was `phase=pp`. Boot 2 is the first
evidence the TP side of this line is reachable at all. So the first test here
proves REACHABILITY before anything else is asserted -- a suite that cannot
enter the path cannot report on it (INDIKATOR-GESETZ).

THE DANGER DIRECTION, and it decides WHERE the fix goes. There are two ways to
restore the invariant and they are not symmetric:

  * refuse the FRESH chunked admission -- nothing of that request has been
    committed, no KV is held for it, no chunk of it has run. It waits one pass.
    Nothing is lost.
  * drop the RESIDENT continuation (clear `chunked_req` at the cutover) -- that
    re-prefills a request mid-flight. That is the double prefill the standing
    law forbids outright, and the #858 wedge shape besides.

So the resident continuation is never the one to give way. The fix is a guard
at the two fresh-request mint sites in `add_one_req`, which is also where the
precedent already lives: `_add_scheduled_req`'s `carried_chunk` flag refuses to
announce a new chunked req for exactly this reason, and names this assert while
doing it.

THE ASSERT STAYS. It is the honest watcher and it has now named its own
reachability twice. Nothing here weakens it; the point is that it stops being
reachable.

HARNESS PROVENANCE: this drives the REAL `PrefillAdder` through the real
`add_chunked_req` / `preempt_to_schedule` / `add_one_req` sequence, adapted
from /spinning/evidence-665-f1/witness_951/witness_941_a.py -- the witness #951
used to establish that the arithmetic is breakable. Reusing it rather than
inventing a second harness is deliberate: a fresh stand-in would be a second
expression of the state under test.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefResult, IncLockRefResult
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20)

AVAIL = 10000
CHUNK = 5000


def _tree_cache():
    tc = MagicMock()
    tc.full_evictable_size.return_value = 0
    tc.swa_evictable_size.return_value = 0
    tc.evictable_size.return_value = 0
    tc.disable = False
    tc.inc_lock_ref.return_value = IncLockRefResult()
    tc.dec_lock_ref.return_value = DecLockRefResult()
    return tc


def _allocator(avail):
    al = MagicMock()
    al.full_available_size.return_value = avail
    al.swa_available_size.return_value = avail
    al.available_size.return_value = avail
    return al


def _running_req(rid, max_new_tokens):
    r = MagicMock(spec=Req)
    r.rid = str(rid)
    r.priority = 0
    r.prefix_indices = []
    r.full_untruncated_fill_ids = []
    r.output_ids = []
    r.sampling_params = SimpleNamespace(max_new_tokens=max_new_tokens)
    r.time_stats = SimpleNamespace(wait_queue_entry_time=0)
    r.retracted_stain = False
    r.finished.return_value = False
    r.needs_host_load_back.return_value = False
    return r


def _fresh_req(rid, priority, n_tokens, max_new_tokens=8):
    r = MagicMock(spec=Req)
    r.rid = str(rid)
    r.priority = priority
    r.prefix_indices = []
    r.full_untruncated_fill_ids = list(range(n_tokens))
    r.output_ids = []
    r.host_hit_length = 0
    r.swa_host_hit_length = 0
    r.sampling_params = SimpleNamespace(max_new_tokens=max_new_tokens, ignore_eos=False)
    r.time_stats = SimpleNamespace(wait_queue_entry_time=0)
    r.retracted_stain = False
    r.born_spilled = False
    r.born_spilled_deep = False
    r.last_node = None
    r.mamba_pool_idx = None
    r.finished.return_value = False
    r.needs_host_load_back.return_value = False
    r.set_extend_range = MagicMock(
        side_effect=lambda start, end: setattr(r, "extend_range", Range(start, end))
    )
    return r


def _server_args():
    sa = MagicMock()
    sa.schedule_low_priority_values_first = False
    sa.enable_fast_lane = False
    sa.fast_lane_reserved_heavy_slots = 0
    return sa


def _running_batch(reqs):
    b = MagicMock()
    b.reqs = list(reqs)
    b.release_req.return_value = None
    b.filter_batch.return_value = None
    return b


class _Bench:
    """The real adder, driven to the state the assert guards.

    CLIP_MAX_NEW_TOKENS caps one running request's contribution to
    `rem_total_token_offset`, so TWO running requests are needed to push the
    initial offset above `rem_chunk_tokens` and force `add_chunked_req` to be
    bound by `rem_total_tokens` -- which is witness (A)'s whole mechanism, and
    the state in which a surviving continuation leaves `rem_chunk_tokens`
    positive.
    """

    def __init__(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        self.adder = PrefillAdder(
            page_size=1,
            tree_cache=_tree_cache(),
            token_to_kv_pool_allocator=_allocator(AVAIL),
            running_batch=_running_batch(
                [
                    _running_req("run1", max_new_tokens=4096),
                    _running_req("run2", max_new_tokens=4096),
                ]
            ),
            new_token_ratio=1.0,
            rem_input_tokens=1_000_000,
            rem_chunk_tokens=CHUNK,
            num_mixed_decode_tokens=0,
            priority_scheduling_preemption_threshold=0,
        )
        self.chunk_req = _fresh_req("chunked", priority=0, n_tokens=CHUNK)
        # The carried continuation survives, truncated by rem_total_tokens.
        self.carried = self.adder.add_chunked_req(self.chunk_req)
        self.new_req = _fresh_req("new1", priority=5, n_tokens=5000)

    def stamp(self, resident):
        """What scheduler.py does right after `add_chunked_req` settles."""
        self.adder.chunked_req_outstanding = bool(resident)

    def admit_fresh(self):
        return self.adder.add_one_req(self.new_req, truncation_align_size=None)


class TheAssertPreconditionMustBeReachableAtAll(unittest.TestCase):
    """Reachability first. window-951 read 0/0 on this line as a pass."""

    def test_a_continuation_really_survives_with_budget_left(self):
        b = _Bench()
        self.assertIsNotNone(
            b.carried,
            "the carried continuation must SURVIVE this pass, or the state the "
            "assert guards is never entered and nothing below means anything",
        )
        self.assertGreater(
            b.adder.rem_chunk_tokens,
            0,
            "and it must leave chunk budget behind -- that is witness (A)'s "
            "mechanism and the reason the emergent arithmetic does not hold",
        )

    def test_CANFAIL_without_the_guard_a_SECOND_continuation_is_minted(self):
        """The defect, reproduced against the real adder.

        This is the precondition of `assert self.chunked_req is None`: a
        carried continuation still resident AND a freshly minted one. With the
        residency fact withheld -- which is exactly the pin's behaviour, since
        nothing stamped it -- the adder mints the second one.

        If this test ever goes green on its own, the guard below is no longer
        measuring anything.
        """
        b = _Bench()
        b.stamp(False)  # the pin: the adder is never told
        b.adder.preempt_to_schedule(b.new_req, _server_args())
        b.admit_fresh()
        self.assertIsNotNone(
            b.adder.new_chunked_req,
            "the unguarded adder must reproduce the crash precondition, "
            "otherwise this harness does not reach the defect",
        )
        self.assertTrue(
            b.carried is not None and b.adder.new_chunked_req is not None,
            "BOTH at once is the assert precondition -- one resident "
            "continuation and one newly minted",
        )


class OnlyOneContinuationMayExistAtATime(unittest.TestCase):
    """The fix, and the shape of the refusal."""

    def _guarded(self):
        b = _Bench()
        b.stamp(True)  # what scheduler.py now stamps
        b.adder.preempt_to_schedule(b.new_req, _server_args())
        return b

    def test_no_second_continuation_is_minted(self):
        b = self._guarded()
        b.admit_fresh()
        self.assertIsNone(
            b.adder.new_chunked_req,
            "a second chunked continuation was minted while one was resident "
            "-- this is the assert that killed all three ranks 3 s after the "
            "first clean pp_to_tp cutover",
        )

    def test_the_fresh_request_is_LEFT_FOR_LATER_not_swallowed(self):
        """Refused, not admitted-and-forgotten.

        `OTHER` is the adder's "leave this request for a later pass", the
        requeue-for-free the admission loop already relies on. Admitting it
        without marking it chunked would be far worse: it would be treated as a
        COMPLETE prefill of a request only partly computed.
        """
        b = self._guarded()
        self.assertEqual(b.admit_fresh(), AddReqResult.OTHER)
        self.assertNotIn(
            b.new_req,
            b.adder.can_run_list,
            "a refused request must not be in the batch",
        )

    def test_the_RESIDENT_continuation_is_untouched(self):
        """KEIN DOPPEL-PREFILL. The fix may not cost the carried request.

        The whole reason the guard sits on the fresh admission rather than at
        the cutover: the resident continuation keeps its geometry and its
        progress, so nothing is re-prefilled.
        """
        b = self._guarded()
        before = (
            b.chunk_req.extend_range.start,
            b.chunk_req.extend_range.end,
        )
        b.admit_fresh()
        self.assertIn(b.chunk_req, b.adder.can_run_list)
        self.assertEqual(
            (b.chunk_req.extend_range.start, b.chunk_req.extend_range.end),
            before,
            "the carried continuation's committed geometry moved -- that is a "
            "re-prefill of work already done",
        )

    def test_the_guard_does_NOT_fire_when_nothing_is_resident(self):
        """The other direction, so the fix cannot cost throughput.

        With no continuation resident the fresh request must still be chunked
        normally. A guard that refused unconditionally would stall every
        chunked admission -- a livelock fix paid for with a throughput defect.
        """
        b = _Bench()
        b.stamp(False)
        b.adder.preempt_to_schedule(b.new_req, _server_args())
        b.admit_fresh()
        self.assertIsNotNone(
            b.adder.new_chunked_req,
            "with nothing resident the ordinary chunked admission must still happen",
        )

    def test_the_default_is_permissive_so_untouched_callers_are_unchanged(self):
        """A caller that never stamps behaves exactly as before the ticket."""
        b = _Bench()
        self.assertFalse(
            b.adder.chunked_req_outstanding,
            "the residency fact must default to False, or every adder built "
            "outside the scheduler silently stops chunking",
        )


if __name__ == "__main__":
    unittest.main()
