"""#798: a void-output slot must not carry a RETRACTED request as the
scheduler's chunked request.

WHAT KILLED THE BOOT. boot_798_0822_0646.log, commit 9478e774b6, is the first
run on this rig in which the phase flip ever COMMITTED a tp_to_pp cutover --
before #796 the seam declined 114 consecutive times and the cutover never
happened, so everything downstream of it had never once executed. Four seconds
after the cutover, rank 0 logged three `#791b PP-ADMISSION void output`
messages back to back on slots 2, 0 and 1 (releasing 1 of 2, 1 of 2, then 0 of
1 -- the last one kept precisely because it was that slot's chunked request),
and the next line was:

    File "sglang/srt/managers/scheduler.py", line 5675, in get_next_batch_to_run
      if self.chunked_req.extend_range.end > len(self.chunked_req.prefix_indices)
    AttributeError: 'NoneType' object has no attribute 'end'

PP1 and PP2 then died on gloo `Connection closed by peer`, a cascade rather
than three independent faults.

THE MECHANISM, and it is specific to void-OUTPUT rather than to voids in
general. `_pp_absorb_void_output` fires ONCE PER SLOT and several slots are
absorbed in a row with no `get_next_batch_to_run` between them. Each call
restores ITS OWN slot's `chunked_before` into `self.chunked_req` and then
resets every batch member that is not kept FOR THAT SLOT. A request that is
slot B's carried chunk but only an ordinary member of slot A's batch is
therefore reset by slot A's disposal loop and then reinstated, already reset,
as `self.chunked_req` by slot B's restore. `pp_void_keeps_request` cannot
prevent this: it is asked per slot, and per slot it answers correctly.

WHY THE TWIN SITE WAS ALREADY SAFE. The own-void path (`#797d own-void`) has
carried this exact guard for some time, and its comment argues the state is
unreachable there because `chunked_before` is a snapshot from the top of the
same pass, so `get_next_batch_to_run` would have raised earlier in that pass.
That argument holds for a single-call site and does NOT hold across a run of
per-slot absorbs. The fix is the twin guard, and the omission was that only one
of two identical sites ever received it.

WHY THE CONDITION TESTS `is_retracted` AND NOT ONLY `extend_range`.
`extend_range` is merely the field the next reader touches first;
`reset_for_retract` clears seventeen more in the same breath. `is_retracted` is
the marker of the shape itself -- set by `reset_for_retract`, cleared only by
`prepare_for_extend` -- so it is True over exactly the window in which the
request must not be carried. `test_the_fake_cannot_drift_from_the_real_req`
pins that claim against the real `Req` so this file cannot quietly stop
describing production.
"""

import types
import unittest

from sglang.test.ci.ci_register import register_cpu_ci


class _Range:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    @property
    def length(self):
        return self.end - self.start


class _FakeReq:
    """The subset of `Req` this path touches, with a faithful retract."""

    def __init__(self, rid, end=64):
        self.rid = rid
        self.extend_range = _Range(0, end)
        self.prefix_indices = []
        self.is_retracted = False
        self.retracted_stain = False
        self.last_node = object()
        self.mamba_pool_idx = 3
        self.retraction_count = 0

    def reset_for_retract(self):
        self.retraction_count += 1
        self.prefix_indices = []
        self.last_node = None
        self.mamba_pool_idx = None
        self.extend_range = None
        self.is_retracted = True
        self.retracted_stain = True


def _reads_chunked_req_like_the_scheduler(holder):
    """Exactly what scheduler.py:5675 does, and it has no guard of its own."""
    if holder.chunked_req is not None:
        return holder.chunked_req.extend_range.end > len(
            holder.chunked_req.prefix_indices
        )
    return False


def _holder(chunked_by_slot, batches):
    from sglang.srt.managers import scheduler_pp_mixin as m

    return types.SimpleNamespace(
        chunked_req=None,
        waiting_queue=[],
        running_mbs=[None] * len(batches),
        _pp_chunked_req_before_by_slot=list(chunked_by_slot),
        _pp_void_forward_payload=None,
        _absorb=m.SchedulerPPMixin._pp_absorb_void_output,
    )


class PPVoidChunkedRetracted798(unittest.TestCase):
    """The metal shape: two absorbs in a row, one shared request."""

    def _run_two_absorbs(self, monkeypatched=True):
        from sglang.srt.managers import scheduler_pp_mixin as m

        shared = _FakeReq("shared-rid")
        # Slot 0 carries nothing and has `shared` as an ordinary member.
        # Slot 1 carries `shared` as ITS chunked request.
        batches = [
            types.SimpleNamespace(reqs=[shared]),
            types.SimpleNamespace(reqs=[]),
        ]
        holder = _holder([None, shared], batches)
        mbs = list(batches)
        mb_metadata = [None, None]

        saved = {
            name: getattr(m, name)
            for name in (
                "pp_void_forward_payload",
                "pp_absorb_admission_return",
                "_park_chunked_prefill_chunk",
                "_release_dynamic_chunk_probe",
            )
        }
        m.pp_void_forward_payload = lambda *a, **k: None
        m.pp_absorb_admission_return = lambda *a, **k: None
        # The real park treats `extend_range is None` as "nothing to give
        # back" and leaves the request untouched -- that is exactly why it
        # cannot repair this state, so the stub must NOT repair it either.
        m._park_chunked_prefill_chunk = lambda scheduler, req: False
        m._release_dynamic_chunk_probe = lambda scheduler, req: None
        try:
            for mb_id in (0, 1):
                holder._absorb(
                    holder,
                    mb_id,
                    {m._PP_VOID_OUTPUT_KEY: True},
                    mbs,
                    mb_metadata,
                )
        finally:
            for name, fn in saved.items():
                setattr(m, name, fn)
        return holder, shared

    def test_slot_a_disposal_reaches_slot_b_chunked_request(self):
        """The precondition. Without it the rest proves nothing."""
        _holder_, shared = self._run_two_absorbs()
        self.assertTrue(
            shared.is_retracted,
            "slot 0's disposal loop must have reset the request that slot 1 "
            "carries -- if it did not, this test no longer reproduces the "
            "boot and its green result is meaningless",
        )
        self.assertIsNone(shared.extend_range)

    def test_the_retracted_chunk_is_not_carried(self):
        """THE FIX. Fails with AttributeError before it, passes after."""
        holder, _shared = self._run_two_absorbs()
        self.assertIsNone(
            holder.chunked_req,
            "a request in the reset_for_retract shape was left in "
            "self.chunked_req; the next pass's get_next_batch_to_run "
            "dereferences it with no guard of its own",
        )

    def test_the_next_pass_read_does_not_raise(self):
        """The boot's actual failure, reproduced at its own call shape."""
        holder, _shared = self._run_two_absorbs()
        try:
            _reads_chunked_req_like_the_scheduler(holder)
        except AttributeError as exc:  # pragma: no cover - the red state
            self.fail(
                "scheduler.py's chunked_req read raised the boot's exception: "
                f"{exc}"
            )

    def test_a_healthy_carried_chunk_is_still_carried(self):
        """The guard must not eat the ordinary case it shares a site with."""
        from sglang.srt.managers import scheduler_pp_mixin as m

        healthy = _FakeReq("healthy-rid")
        batches = [types.SimpleNamespace(reqs=[])]
        holder = _holder([healthy], batches)

        saved = {
            name: getattr(m, name)
            for name in (
                "pp_void_forward_payload",
                "pp_absorb_admission_return",
                "_park_chunked_prefill_chunk",
                "_release_dynamic_chunk_probe",
            )
        }
        m.pp_void_forward_payload = lambda *a, **k: None
        m.pp_absorb_admission_return = lambda *a, **k: None
        m._park_chunked_prefill_chunk = lambda scheduler, req: True
        m._release_dynamic_chunk_probe = lambda scheduler, req: None
        try:
            holder._absorb(
                holder, 0, {m._PP_VOID_OUTPUT_KEY: True}, list(batches), [None]
            )
        finally:
            for name, fn in saved.items():
                setattr(m, name, fn)

        self.assertIs(
            holder.chunked_req,
            healthy,
            "a chunk that was never retracted must survive the void -- "
            "clearing it would discard the tree handles of every chunk "
            "already stashed",
        )

    def test_a_scheduler_that_never_set_chunked_req_survives(self):
        """The guard must not ASSUME the attribute exists.

        The restore eight lines above assigns `self.chunked_req` only when the
        carried value DIFFERS from the current one, so a scheduler that never
        set the attribute and carries None still does not have it by the time
        the guard runs. A plain `self.chunked_req` read raises there, and it
        raised for real: `test_ring_survives_the_retraction_with_the_fix` in
        test_pp_output_ring_retraction_wedge_791b.py went red with
        `AttributeError: 'types.SimpleNamespace' object has no attribute
        'chunked_req'` on the first version of this fix. That suite is 4-green
        alone with the guard off and was 1-red with it on -- causal, not
        order-dependent, and the full 31-suite sweep is what surfaced it.
        """
        from sglang.srt.managers import scheduler_pp_mixin as m

        batches = [types.SimpleNamespace(reqs=[])]
        holder = _holder([None], batches)
        del holder.chunked_req  # the shape the ring's worker actually builds
        self.assertFalse(hasattr(holder, "chunked_req"))

        saved = {
            name: getattr(m, name)
            for name in (
                "pp_void_forward_payload",
                "pp_absorb_admission_return",
                "_park_chunked_prefill_chunk",
                "_release_dynamic_chunk_probe",
            )
        }
        m.pp_void_forward_payload = lambda *a, **k: None
        m.pp_absorb_admission_return = lambda *a, **k: None
        m._park_chunked_prefill_chunk = lambda scheduler, req: False
        m._release_dynamic_chunk_probe = lambda scheduler, req: None
        try:
            holder._absorb(
                holder, 0, {m._PP_VOID_OUTPUT_KEY: True}, list(batches), [None]
            )
        except AttributeError as exc:  # pragma: no cover - the red state
            self.fail(f"the guard assumed an attribute that need not exist: {exc}")
        finally:
            for name, fn in saved.items():
                setattr(m, name, fn)

    def test_the_fake_cannot_drift_from_the_real_req(self):
        """`_FakeReq.reset_for_retract` must keep describing production.

        The guard keys on `is_retracted`, so this file is only evidence for as
        long as the real `reset_for_retract` still sets it and still clears
        `extend_range`. Pin both against the real source rather than trusting
        the fake.
        """
        import inspect

        from sglang.srt.managers.schedule_batch import Req

        src = inspect.getsource(Req.reset_for_retract)
        self.assertIn("self.extend_range = None", src)
        self.assertIn("self.is_retracted = True", src)


register_cpu_ci(__file__)

if __name__ == "__main__":
    unittest.main()
