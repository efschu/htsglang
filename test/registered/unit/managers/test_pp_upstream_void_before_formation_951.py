"""#951: the #798 void must be decided BEFORE the pass builds anything.

THE SPECIMEN. boot_943bx_dc4895e1dc_0828_000240.log and ..._001113.log, pin
dc4895e1dc, two boots, both dead at exactly 7 batches with no CUDA error
involved:

    File "sglang/srt/managers/scheduler.py", line 9286, in _get_new_batch_prefill_raw
      assert self.chunked_req is None
    AssertionError

and immediately above it, on PP1:

    #798 PP-ADMISSION pass voided on slot N ...
    #797d PP-ADMISSION own pass voided on slot N ... chunk parked=True

THE ASYMMETRY, and it is the whole defect. There are two void paths and they
are explicitly written as mirrors of each other -- `_event_loop_pp_body`
labels the second "AND THE MIRROR OF IT". They are not mirrored in the one
respect that matters:

  * #797 (this rank retracted) sets `_pp_admission_pass_voided` in
    `_pp_void_retracted_pass`, which runs BEFORE `get_next_batch_to_run`. So
    scheduler.py's guard sees the flag and the pass builds nothing at all --
    which is what that guard's own comment demands: "The retraction voids the
    pass, so the pass must build nothing at all."

  * #798 (this rank's UPSTREAM did not launch) sets the same flag in
    `_pp_void_pass_without_upstream_launch`, which runs AFTER
    `get_next_batch_to_run`. So its pass builds EVERYTHING -- it runs the full
    `_get_new_batch_prefill_raw`, advances `chunked_req` through
    `add_chunked_req`, may adopt a fresh `adder.new_chunked_req`, takes lock
    refs, removes requests from `waiting_queue` -- and is then unwound
    retroactively by `_pp_void_own_batch`.

A retroactive unwind is a compensator, and this one is reached only after the
damage is done. `_pp_void_own_batch` restores what it knows about
(`mbs[mb_id]`, `mb_metadata[mb_id]`, `chunked_req` from the per-slot snapshot,
and it parks the chunk). It cannot restore what it was never told about: the
`inflight_middle_chunks += 1` that scheduler.py applies immediately after
adopting `adder.new_chunked_req` lands on the NEWLY adopted request, while
`_park_chunked_prefill_chunk` gives the decrement back to the PRE-PASS one.

WHY THE FIX IS THE GUARD AND NOT A BIGGER UNWIND. Completing the unwind
treats the effect: it must enumerate every field the ungated pass touched and
keep that list correct forever. The condition the guard needs is already
available before the call -- `_pp_upstream_launched_incoming` is written in
`_pp_recv_admission_decision` (scheduler_pp_mixin.py), whose own docstring
records that it is "Positioned in `_event_loop_pp_body` strictly BEFORE
`get_next_batch_to_run`", and the call site confirms it. So the pass can be
refused at the same place the #797 pass already is, and then there is no state
to unwind, this time or after any future edit.

WHAT THIS FILE PINS, and the danger direction it pins in both directions:
the guard must fire on a rank whose upstream did not launch, and must be
INERT on the first rank, on `pp_size <= 1`, on a gapped wire, and whenever the
upstream did launch. A guard that over-fires voids every pass on PP0 and
serves nothing at all, which is a worse failure than the one it fixes.
"""

import types
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20)

WORLD = 3


class _ReachedPrefillFormation(Exception):
    """Raised by the sentinel that stands where the prefill formation begins.

    The assertion is on the SHIPPED guard's control flow: everything up to and
    including `scheduler.py`'s void guard is the real method. Only what lies
    strictly downstream of the guard is replaced, so a green result cannot be
    produced by stubbing out the thing under test.
    """


class _StubPool:
    def __init__(self):
        self.size = 1 << 20

    def available_size(self):
        return self.size


class _StubAllocator:
    def available_size(self):
        return 1 << 20


class _Range:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    @property
    def length(self):
        return self.end - self.start


class _ChunkReq:
    """Enough of a Req for the prologue's chunked read, and no more."""

    def __init__(self, rid="chunk-rid", prefix=4096, end=8192):
        self.rid = rid
        self.prefix_indices = list(range(prefix))
        self.extend_range = _Range(prefix, end)
        self.inflight_middle_chunks = 0
        self.is_retracted = False

    def init_next_round_input(self, *a, **k):
        return None


def _holder(
    *,
    pp_rank=1,
    is_first_rank=False,
    upstream_launched=False,
    gapped=False,
    pp_size=WORLD,
    chunked_req=None,
):
    """A scheduler stand-in that runs the SHIPPED `get_next_batch_to_run`."""
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    h = types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_rank=pp_rank, pp_size=pp_size),
        pp_group=types.SimpleNamespace(is_first_rank=is_first_rank, is_last_rank=False),
        chunked_req=chunked_req,
        waiting_queue=[],
        running_mbs=[None] * WORLD,
        req_to_token_pool=_StubPool(),
        token_to_kv_pool_allocator=_StubAllocator(),
        _pp_admission_guard=None,
        _pp_chunked_req_before_by_slot=[None] * WORLD,
        # -- the two facts this guard turns on
        _pp_admission_pass_voided=False,
        _pp_upstream_launched_incoming=upstream_launched,
        _pp_gapped_wire=gapped,
        _pp_upstream_idle_void_streak=0,
        _pp_upstream_idle_voids=0,
        # -- prologue reads, all inert
        enable_hierarchical_cache=False,
        enable_fpm=False,
        enable_hisparse=False,
        enable_lora=False,
        dllm_config=None,
        kv_session_offload=None,
        _regime_observer_mode="off",
        regime_observer=None,
        kv_capacity_runtime=None,
        congruent_prefill_lane=None,
        _round_built_nothing=True,
        stashed=[],
        server_args=types.SimpleNamespace(
            kv_reshard_vectors=None,
            enable_phase_flip=False,
            enable_vram_dial=False,
            kv_pressure_ladder=None,
            gdn_resident_state_slots=None,
        ),
    )
    h.process_pending_chunked_abort = lambda: None
    h._abort_on_waiting_timeout = lambda: None
    h._abort_on_running_timeout = lambda rb: None
    h._update_uniform_pool_budget = lambda: None
    h._census_tick = lambda: None
    h._corridor_trace_tick = lambda: None
    h._flight_serving_tick = lambda: None
    h.stash_chunked_request = lambda req: h.stashed.append(req)
    h._pp_note_chunked_req_before_admission = types.MethodType(
        SchedulerPPMixin._pp_note_chunked_req_before_admission, h
    )

    def _sentinel(running_batch):
        raise _ReachedPrefillFormation(
            "get_next_batch_to_run entered the prefill formation on a pass "
            "whose upstream reported launched=False"
        )

    h.get_new_batch_prefill = _sentinel
    h.get_next_batch_to_run = types.MethodType(Scheduler.get_next_batch_to_run, h)
    return h


def _run_one_pass(h, mb_id=1):
    """One pass, arranged exactly as `_event_loop_pp_body` arranges it:
    record the pre-admission chunked request, then plan."""
    h._pp_note_chunked_req_before_admission(mb_id)
    running = types.SimpleNamespace(
        is_prefill_only=False, reqs=[], batch_is_full=False, is_empty=lambda: True
    )
    return h.get_next_batch_to_run(running_batch=running, last_batch=None)


def _patch_prefill_blocked(value=False):
    """`phase_prefill_blocked_here` is #631's strict-purity gate and sits
    between the void guard and the formation. It is not what this file is
    about, so it is pinned open -- pinning it SHUT would hide the defect."""
    from sglang.srt.managers import scheduler as sched_mod

    saved = sched_mod.phase_prefill_blocked_here
    sched_mod.phase_prefill_blocked_here = lambda *a, **k: value
    return sched_mod, saved


class PPUpstreamVoidBeforeFormation951(unittest.TestCase):
    """THE ROOT: a #798-voided pass must not reach the prefill formation."""

    def test_a_pass_whose_upstream_did_not_launch_builds_nothing(self):
        """RED before the fix, GREEN after.

        Before: control reaches the sentinel, i.e. the shipped method runs the
        whole prefill formation for a pass that is already known to be void.
        After: the guard withholds the round, exactly as it does for #797.
        """
        mod, saved = _patch_prefill_blocked(False)
        try:
            h = _holder(upstream_launched=False, chunked_req=_ChunkReq())
            try:
                plan = _run_one_pass(h)
            except _ReachedPrefillFormation as exc:
                self.fail(
                    "#951: the pass reached the prefill formation although its "
                    "upstream reported launched=False. Everything it does from "
                    "here is unwound retroactively by _pp_void_own_batch, which "
                    "is the compensator the #798 specimens died on: " + str(exc)
                )
        finally:
            mod.phase_prefill_blocked_here = saved

        self.assertIsNone(
            plan.batch_to_run,
            "a pass whose upstream did not launch must build no batch at all",
        )

    def test_the_chunked_request_is_not_touched_by_a_voided_pass(self):
        """The invariant, stated as state rather than as control flow.

        `chunked_req` is scheduler state that outlives the round. A pass that
        is refused before it starts cannot have moved it, so no unwind is
        needed and none can be forgotten.
        """
        chunk = _ChunkReq()
        before = (chunk.extend_range.start, chunk.extend_range.end)
        before_inflight = chunk.inflight_middle_chunks

        mod, saved = _patch_prefill_blocked(False)
        try:
            h = _holder(upstream_launched=False, chunked_req=chunk)
            try:
                _run_one_pass(h)
            except _ReachedPrefillFormation:
                self.fail(
                    "#951: reached the formation; the state assertions below "
                    "would be measuring the compensator, not the guard"
                )
        finally:
            mod.phase_prefill_blocked_here = saved

        self.assertIs(h.chunked_req, chunk, "the chunked request must survive")
        self.assertEqual(
            (chunk.extend_range.start, chunk.extend_range.end),
            before,
            "a refused pass may not advance the chunked request's extend range",
        )
        self.assertEqual(
            chunk.inflight_middle_chunks,
            before_inflight,
            "a refused pass may not leak an inflight_middle_chunks increment -- "
            "its matching decrement lives in process_batch_result_prefill, "
            "which never runs for a voided pass",
        )


class PPUpstreamVoidGuardIsInert951(unittest.TestCase):
    """THE DANGER DIRECTION. A guard that over-fires serves nothing at all.

    Each arm here is a shape in which the #798 void itself is a documented
    no-op (`_pp_void_pass_without_upstream_launch`: "NO-OP ON EVERY PATH THAT
    CANNOT HAVE THE PROBLEM: the first rank (no upstream), pp_size <= 1, a
    gapped set ... an upstream that did launch"). The new guard must honour
    exactly the same exclusions, or PP0 -- which never receives an admission
    decision and therefore never sets `_pp_upstream_launched_incoming` -- would
    void every pass it ever takes.
    """

    def _reaches_formation(self, **kw):
        mod, saved = _patch_prefill_blocked(False)
        try:
            h = _holder(**kw)
            try:
                _run_one_pass(h)
            except _ReachedPrefillFormation:
                return True
            return False
        finally:
            mod.phase_prefill_blocked_here = saved

    def test_the_first_rank_is_never_voided_by_this_guard(self):
        """PP0 has no upstream, so its `_pp_upstream_launched_incoming` is
        False on EVERY pass (reset each pass, written only by the receive it
        does not make). This is the catastrophic direction."""
        self.assertTrue(
            self._reaches_formation(
                pp_rank=0, is_first_rank=True, upstream_launched=False
            ),
            "the first rank must keep building batches -- it has no upstream "
            "whose launch could be missing",
        )

    def test_a_single_stage_boot_is_never_voided_by_this_guard(self):
        self.assertTrue(
            self._reaches_formation(
                pp_rank=0, is_first_rank=True, pp_size=1, upstream_launched=False
            ),
            "pp_size <= 1 has no upstream hop at all",
        )

    def test_a_gapped_wire_is_never_voided_by_this_guard(self):
        self.assertTrue(
            self._reaches_formation(gapped=True, upstream_launched=False),
            "a gapped set has no stage-boundary proxy, so the receive this "
            "void prevents is never made in the first place",
        )

    def test_a_healthy_pass_still_builds_its_batch(self):
        """The ordinary case this guard shares a site with."""
        self.assertTrue(
            self._reaches_formation(upstream_launched=True),
            "an upstream that DID launch must be paired with, not voided",
        )

    def test_a_healthy_chunked_continuation_still_reaches_the_formation(self):
        """The gegenrichtung the assert at scheduler.py:9286 exists for: a
        multi-part prefill with no void anywhere must be entirely unaffected."""
        self.assertTrue(
            self._reaches_formation(upstream_launched=True, chunked_req=_ChunkReq()),
            "a chunked continuation on a healthy pass must still be admitted",
        )


class PPUpstreamVoidGuardMatchesTheVoid951(unittest.TestCase):
    """The two decisions must not be able to drift apart.

    The guard and `_pp_void_pass_without_upstream_launch` answer the same
    question at two different moments. If one is edited and the other is not,
    the pass is either built and unwound again (the defect returns) or voided
    without being forwarded (a new one). They are held to one predicate.
    """

    def test_both_sites_read_the_same_predicate(self):
        import inspect

        from sglang.srt.managers import scheduler as sched_mod
        from sglang.srt.managers import scheduler_pp_mixin as mixin_mod

        name = "pp_upstream_void_pending"
        self.assertTrue(
            hasattr(mixin_mod, name),
            "the shared predicate must exist as one named function, not as "
            "two copies of the same four conditions",
        )
        guard_src = inspect.getsource(sched_mod.Scheduler.get_next_batch_to_run)
        void_src = inspect.getsource(
            mixin_mod.SchedulerPPMixin._pp_void_pass_without_upstream_launch
        )
        self.assertIn(
            name,
            guard_src,
            "scheduler.py's void guard must consult the shared predicate",
        )
        self.assertIn(
            name,
            void_src,
            "the #798 void must consult the same shared predicate",
        )


if __name__ == "__main__":
    unittest.main()


class _VoidHolder:
    """The #798 void site's own reads, and only those.

    Driven with the SHIPPED `_pp_void_pass_without_upstream_launch` and the
    SHIPPED `_pp_void_own_batch`, because the two facts under test here --
    whether the streak survives an emptied slot, and whether the suppress flag
    is consumed before the early exit -- are properties of those bodies.
    """

    def __init__(self, *, withheld_work, batch, streak=0):
        import types as _t

        self.ps = _t.SimpleNamespace(pp_rank=1, pp_size=WORLD)
        self.pp_group = _t.SimpleNamespace(is_first_rank=False, is_last_rank=False)
        self._pp_gapped_wire = False
        self._pp_upstream_launched_incoming = False
        self._pp_upstream_void_withheld_work = withheld_work
        self._pp_upstream_idle_void_streak = streak
        self._pp_upstream_idle_voids = 0
        self._pp_idle_void_suppress_log = False
        self._pp_admission_pass_voided = False
        self._pp_admission_incoming_effective = None
        self._pp_admission_amended_to_forward = None
        self._pp_admission_incoming_schedule = None
        self.mbs = [batch] * WORLD if batch is not None else [None] * WORLD
        self.mb_metadata = [None] * WORLD
        self.running_mbs = [None] * WORLD
        self.chunked_req = None
        self.waiting_queue = []
        self._pp_chunked_req_before_by_slot = [None] * WORLD

    def bind(self):
        import types as _t

        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        self._pp_void_pass_without_upstream_launch = _t.MethodType(
            SchedulerPPMixin._pp_void_pass_without_upstream_launch, self
        )
        self._pp_void_own_batch = _t.MethodType(
            SchedulerPPMixin._pp_void_own_batch, self
        )
        return self


class PPUpstreamVoidKeepsTheSpinDetector951(unittest.TestCase):
    """#801-spin must survive #951, and #951 must not silence #797d.

    Both are regressions this change could introduce and neither would show up
    as a failure anywhere else: a livelock detector that stops counting looks
    exactly like a livelock that stopped happening, and a suppressed log looks
    exactly like a void that did not occur.
    """

    def test_the_streak_still_counts_when_the_guard_emptied_the_slot(self):
        """The slot is empty BECAUSE the guard refused the pass, and this rank
        held work. That is the same state the streak has always counted."""
        h = _VoidHolder(withheld_work=True, batch=None, streak=4).bind()
        h._pp_void_pass_without_upstream_launch(0)
        self.assertEqual(
            h._pp_upstream_idle_void_streak,
            5,
            "#951 empties the slot on exactly the passes #798 used to find "
            "non-empty; reading emptiness alone would retire the #801-spin "
            "detector at the moment it started mattering",
        )

    def test_an_idle_rank_still_clears_the_streak(self):
        """The other half, and the one that keeps the bound honest: a rank
        with no work of its own is not in a no-progress spin, it is simply
        idle, and must not be counted towards a RuntimeError."""
        h = _VoidHolder(withheld_work=False, batch=None, streak=4).bind()
        h._pp_void_pass_without_upstream_launch(0)
        self.assertEqual(
            h._pp_upstream_idle_void_streak,
            0,
            "an idle rank must clear the streak exactly as it did before",
        )

    def test_the_suppress_flag_is_consumed_even_when_the_slot_is_empty(self):
        """The flag's own contract is that it 'can never outlive this pass'.

        It is set by the #798 site and consumed by `_pp_void_own_batch`. Once
        #951 makes that method return early on an empty slot, a consume placed
        after the return would leak a True into the NEXT void -- an ordinary
        #797 retraction -- and silence its record.
        """
        h = _VoidHolder(withheld_work=True, batch=None).bind()
        h._pp_idle_void_suppress_log = True
        h._pp_void_own_batch(0)
        self.assertFalse(
            h._pp_idle_void_suppress_log,
            "the suppress flag survived a void it did not belong to; the next "
            "#797 own-void would be silently unrecorded",
        )
