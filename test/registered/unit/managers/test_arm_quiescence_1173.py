"""#1173: the drained premise, the arm precondition, and the launched-pass STOP.

THE SPECIMEN (boot_855_weg1b4_f58a71bde0_0903_080200.log, 08:06:49Z). PP0
admitted the last queued request into microbatch slot 1 (`#969N ADMIT slot=1
fwd_ct=81 bs=1 extend=1912 rids=[b64dc1cb]`, log 82089), posted its proxy one
line later (82144) -- and on the NEXT line armed `pp_to_tp` on a DRAINED
verdict reading "0 tok remaining" (82145) while the ring still carried the
launched pass. The group then died in the #1153 starvation shape.

Three properties are pinned here, one per decision:

D1  every pp-exit arm reads a pending term that INCLUDES launched-but-
    unreturned extend work. `extend_range.end` is advanced when a pass is
    PREPARED, so the pre-#1173 resident term priced the in-flight request at
    zero; the honest quantity is `total - extend_range.start`.
D2a quiescence is a PRECONDITION of the arm: with a launched pass
    outstanding PP0 defers the arm by name instead of taking it and holding.
D2b a follower never freezes its microbatch ring while a frame for a
    launched pass sits in the typed inbox, and a frame it cannot execute
    under the arm STOPS the group through the launcher instead of spinning.
"""

import time
import types
import unittest

from sglang.srt.managers import scheduler_pp_mixin as ppm
from sglang.srt.managers.scheduler import Scheduler


class _Range:
    def __init__(self, start, end):
        self.start = start
        self.end = end


def _req(rid, total, start, end):
    return types.SimpleNamespace(
        rid=rid,
        origin_input_ids=list(range(total)),
        extend_range=_Range(start, end),
    )


class _Mb:
    def __init__(self, reqs):
        self.reqs = reqs
        self.forward_mode = types.SimpleNamespace(is_extend=lambda: True)

    def is_empty(self):
        return not self.reqs


class TestTheDrainedPremiseCountsLaunchedWork(unittest.TestCase):
    """D1: the pending term the arm reads must see the launched pass."""

    def _double(self, mbs, chunked=None, running=()):
        return types.SimpleNamespace(
            waiting_queue=[],
            mbs=mbs,
            chunked_req=chunked,
            running_batch=types.SimpleNamespace(reqs=list(running)),
        )

    def test_a_launched_pass_with_an_empty_queue_is_not_drained(self):
        # The weg1b4 numbers verbatim: slot 1 carries b64dc1cb, whose pass was
        # prepared over [4096, 6008) of a 6008-token prompt.
        req = _req("b64dc1cb", 6008, 4096, 6008)
        sched = self._double([None, _Mb([req])])
        pending = Scheduler._pending_prefill_tokens(sched, None, include_health=False)
        self.assertEqual(pending, 1912)
        self.assertIn("inflight=1912", sched._pending_prefill_terms)
        self.assertIn(
            "producer Scheduler._pending_prefill_tokens", sched._pending_prefill_terms
        )

    def test_the_in_flight_request_is_billed_once_not_twice(self):
        # Same request resident AND in flight: the resident term must yield.
        #
        # #1173 review (N9): `extend_range.end < total` ON PURPOSE. The first
        # draft used end == total, so the resident term was 0 whatever the
        # dedup guard did and deleting the guard left the test green -- the
        # case was vacuous on the very axis it named. Here the resident term
        # would contribute 6008 - 5000 = 1008 if the guard were removed, so
        # the assertion measures the double-billing it claims to measure:
        # 1912 (in-flight, from `start`) and NOT 2920.
        req = _req("b64dc1cb", 6008, 4096, 5000)
        sched = self._double([_Mb([req])], running=[req])
        self.assertEqual(
            Scheduler._pending_prefill_tokens(sched, None, include_health=False), 1912
        )

    def test_a_resident_request_the_ring_never_launched_is_still_billed(self):
        # The other side of the same guard: with no in-flight entry the
        # resident term is the one that pays, so the dedup above cannot be
        # "skip the resident term always".
        req = _req("resident", 6008, 4096, 5000)
        sched = self._double([None], running=[req])
        self.assertEqual(
            Scheduler._pending_prefill_tokens(sched, None, include_health=False), 1008
        )

    def test_a_decode_microbatch_owes_no_prefill(self):
        req = _req("d", 6008, 4096, 6008)
        mb = _Mb([req])
        mb.forward_mode = types.SimpleNamespace(is_extend=lambda: False)
        self.assertEqual(
            Scheduler._pending_prefill_tokens(
                self._double([mb]), None, include_health=False
            ),
            0,
        )


class TestTheHoldReleasesForAStashedFrame(unittest.TestCase):
    """D2b: an armed follower must reach the slot a launched frame names."""

    def _sched(self, mbs, queue):
        # Bound at CALL time, not at class-body time: on a tree without the
        # #1173 helper this must fail as a red TEST, not as a collection error
        # that hides every other case in the module.
        double = type(
            "_Double",
            (),
            {
                "_pp_flip_stashed_frame_forces_advance": (
                    ppm.SchedulerPPMixin._pp_flip_stashed_frame_forces_advance
                ),
                # #1173 review: the hold's early returns clear the window
                # through this helper, so the double must carry it too.
                "_1173_forget_stashed_frame": (
                    ppm.SchedulerPPMixin._1173_forget_stashed_frame
                ),
            },
        )
        sched = double()
        sched.server_args = types.SimpleNamespace(enable_phase_flip=True)
        sched.pp_phase_flip_armed = lambda: True
        sched.chunked_req = None
        sched.mbs = mbs
        sched.pp_group = object()
        sched.ps = types.SimpleNamespace(pp_rank=1)
        sched.forward_ct = 81
        # `_pp_flip_pass_tick` publishes this at the top of every real armed
        # iteration, before any enable test, so the helper can always read the
        # slot the loop is on. The budget counts ADVANCES of it (#1173 review
        # blocker 1), so a faithful double has to move it the way the loop
        # does -- see `_spin`.
        sched._pp_live_mb_id = 0
        sched._pp_flip_epoch = lambda: 2
        self._queue = queue
        return sched

    @staticmethod
    def _spin(sched, turns):
        """Run `turns` armed loop iterations, advancing the slot like the loop.

        The ring only advances on an iteration the hold RELEASED, which is
        exactly what `_pp_flip_hold_slot` returning False means -- so the
        double reproduces the production coupling instead of asserting on a
        counter the real loop would never reach.
        """
        held = []
        ring = max(1, len(sched.mbs))
        for _ in range(turns):
            hold = ppm.SchedulerPPMixin._pp_flip_hold_slot(sched)
            held.append(hold)
            if not hold:
                sched._pp_live_mb_id = (sched._pp_live_mb_id + 1) % ring
        return held

    def _patch_inbox(self, queue):
        self._orig_src = ppm.resolve_src
        self._orig_inbox = ppm.typed_inbox
        ppm.resolve_src = lambda group, x: 0
        ppm.typed_inbox = lambda group: {(0, "proxy"): queue}
        self.addCleanup(self._restore)

    def _restore(self):
        ppm.resolve_src = self._orig_src
        ppm.typed_inbox = self._orig_inbox

    def test_an_empty_ring_with_an_empty_inbox_holds(self):
        self._patch_inbox([])
        sched = self._sched([None, None], [])
        self.assertTrue(ppm.SchedulerPPMixin._pp_flip_hold_slot(sched))

    def test_a_stashed_frame_releases_the_hold_so_the_ring_advances(self):
        stamp = (1, 6, 1912, 2, 82, ("b64dc1cb", 4096, 6008))
        self._patch_inbox([{"__stamp__": stamp}])
        sched = self._sched([None, None], [])
        self.assertFalse(ppm.SchedulerPPMixin._pp_flip_hold_slot(sched))

    def test_a_frame_that_never_leaves_stops_the_group_by_name(self):
        stamp = (1, 6, 1912, 2, 82, ("b64dc1cb", 4096, 6008))
        self._patch_inbox([{"__stamp__": stamp}])
        sched = self._sched([None, None], [])
        with self.assertRaises(RuntimeError) as caught:
            self._spin(sched, 64)
        msg = str(caught.exception)
        for term in (
            "#1173 LAUNCHED PASS UNEXECUTED UNDER ARM STOP",
            "rank=1",
            "slot=1",
            "fwd_ct=82",
            "rid=b64dc1cb",
            "arm_epoch=2",
            "reason=",
        ):
            self.assertIn(term, msg)

    def test_a_frame_consumed_within_the_bound_clears_the_counter(self):
        queue = [{"__stamp__": (1, 6, 1912, 2, 82, ("b64dc1cb", 4096, 6008))}]
        self._patch_inbox(queue)
        sched = self._sched([None, None], [])
        self.assertEqual(self._spin(sched, 4), [False] * 4)
        queue.clear()
        self.assertTrue(ppm.SchedulerPPMixin._pp_flip_hold_slot(sched))
        queue.append({"__stamp__": (0, 7, 8, 2, 83, ("aa", 0, 8))})
        # Counter restarted: a fresh frame gets its own full bound.
        self.assertEqual(self._spin(sched, 6), [False] * 6)

    def test_back_to_back_frames_each_get_their_own_budget(self):
        """#1173 review, blocker 1: the reviewer's MEASURED false STOP.

        Three frames, each consumed after 3 advances, against a ring of 3
        (bound 8). The first draft counted CONSECUTIVE VISITS WITH ANY FRAME
        AT THE HEAD and reset only on an empty queue, so this drained
        follower -- every frame consumed well inside the bound -- tripped the
        group STOP after 9 visits. The budget is now keyed to the frame, so
        no frame can spend another frame's allowance.
        """
        queue = [
            {"__stamp__": (0, 6, 10, 2, 90, ("aaaaaaaa", 0, 10))},
            {"__stamp__": (1, 6, 11, 2, 91, ("bbbbbbbb", 0, 11))},
            {"__stamp__": (2, 6, 12, 2, 92, ("cccccccc", 0, 12))},
        ]
        self._patch_inbox(queue)
        sched = self._sched([None, None, None], [])
        for _ in range(3):
            self.assertEqual(self._spin(sched, 3), [False] * 3)
            queue.pop(0)
        self.assertTrue(ppm.SchedulerPPMixin._pp_flip_hold_slot(sched))

    def test_back_to_back_frames_on_one_slot_each_get_their_own_budget(self):
        """The sharpest form of blocker 1, on the ring where the bound is 4.

        In the TP phase `pp_loop_size` is 1, so EVERY frame names slot 0 and
        every lap is an arrival at it. Five frames, each consumed after a
        single lap, are five healthy consumptions -- but against a budget
        keyed to the armed window alone they sum to 5 and trip the group STOP
        at the fourth. Only a budget keyed to the FRAME can tell "one frame
        the ring never executed" from "five frames the ring executed".
        """
        queue = [
            {"__stamp__": (0, 6, 10 + i, 2, 90 + i, ("f%d" % i, 0, 10))}
            for i in range(5)
        ]
        self._patch_inbox(queue)
        sched = self._sched([None], [])
        for _ in range(5):
            self.assertFalse(ppm.SchedulerPPMixin._pp_flip_hold_slot(sched))
            queue.pop(0)
        self.assertTrue(ppm.SchedulerPPMixin._pp_flip_hold_slot(sched))

    def test_a_frozen_ring_does_not_burn_the_budget(self):
        """#1173 review, blocker 1(b): visits are not chances.

        The frame names slot 1 while some other guard freezes the loop on
        slot 0. The loop can spin far past the bound without ever ARRIVING at
        the slot the frame names, so the receive never got a chance and no
        STOP is owed -- the budget must not move.
        """
        self._patch_inbox([{"__stamp__": (1, 6, 1912, 2, 82, ("b64dc1cb", 0, 8))}])
        sched = self._sched([None, None], [])
        for _ in range(64):
            self.assertFalse(ppm.SchedulerPPMixin._pp_flip_hold_slot(sched))
        self.assertEqual(getattr(sched, "_1173_held_frame_visits", 0), 0)

    def test_the_bound_is_reachable_on_a_one_slot_ring(self):
        """The TP phase has `pp_loop_size` 1, and the guard must work there.

        `mb_id` is 0 on every iteration of a one-slot ring and never CHANGES
        value, so a change-detector would make this STOP structurally
        unfireable in exactly the phase whose bound is smallest (4). Counting
        ARRIVALS at the named slot keeps it reachable.
        """
        self._patch_inbox([{"__stamp__": (0, 6, 10, 2, 90, ("aaaaaaaa", 0, 10))}])
        sched = self._sched([None], [])
        with self.assertRaises(RuntimeError) as caught:
            self._spin(sched, 32)
        self.assertIn(
            "#1173 LAUNCHED PASS UNEXECUTED UNDER ARM STOP", str(caught.exception)
        )

    def test_a_disarm_does_not_leave_a_count_for_the_next_window(self):
        """#1173 review, blocker 2: no early return may leave a count standing.

        In the TP phase `pp_loop_size` is 1, so the bound is 4 and a single
        inherited visit changes the verdict -- in the group-killing direction.
        """
        queue = [{"__stamp__": (0, 6, 10, 2, 90, ("aaaaaaaa", 0, 10))}]
        self._patch_inbox(queue)
        sched = self._sched([None], [])
        self.assertEqual(self._spin(sched, 3), [False] * 3)
        # Disarm: every early return of the hold must forget the window.
        sched.pp_phase_flip_armed = lambda: False
        self.assertFalse(ppm.SchedulerPPMixin._pp_flip_hold_slot(sched))
        sched.pp_phase_flip_armed = lambda: True
        # A fresh armed window with the SAME frame gets the full bound again.
        self.assertEqual(self._spin(sched, 4), [False] * 4)


class TestQuiescenceIsAnArmPrecondition(unittest.TestCase):
    """D2a: PP0 defers the arm while it holds a launched pass."""

    def _runtime(self, launched, stall_s=120.0):
        import torch

        from sglang.srt.managers.kv_reshard import KvPoolView
        from sglang.srt.managers.phase_flip_runtime import (
            PHASE_PP,
            PhaseFlipRuntime,
        )

        def _view(n):
            return KvPoolView(
                [torch.zeros(4, 8) for _ in range(n)],
                [torch.zeros(4, 8) for _ in range(n)],
            )

        return PhaseFlipRuntime(
            n_ranks=2,
            rank=0,
            layer_map=((0,), (1,)),
            n_layers=2,
            tp_vector=(7, 9),
            boot_phase=PHASE_PP,
            consensus_interval=1,
            # A one-rank echo: this test never crosses a rank boundary (the
            # precondition is PP0-only by construction), so the identity
            # reduction is the faithful stand-in for a group of one voter.
            collective_min=lambda payload, timeout_s=None: list(payload),
            exchange=lambda *a, **k: None,
            pp_pool_view=_view(1),
            tp_pool_view=_view(2),
            live_slots_fn=lambda: [],
            ready_fn=lambda: True,
            cutover_fn=lambda d: None,
            launched_passes_fn=launched,
            launched_pass_stall_s=stall_s,
        )

    def test_a_launched_pass_defers_the_arm_by_name(self):
        rt = self._runtime(lambda: ([1], 81))
        armed, msg = rt.arm("pp_to_tp", "test")
        self.assertFalse(armed)
        self.assertIn("#1173 ARM DEFERRED", msg)
        self.assertIn("slots=[1]", msg)
        self.assertIn("fwd_ct=81", msg)
        self.assertEqual(rt.arm_deferred_launched, 1)
        self.assertIsNone(rt.pending)

    def test_a_quiescent_ring_arms(self):
        rt = self._runtime(lambda: ([], 82))
        armed, msg = rt.arm("pp_to_tp", "test")
        self.assertTrue(armed, msg)
        self.assertEqual(rt.arm_deferred_launched, 0)

    def test_an_unreadable_probe_never_defers(self):
        def _boom():
            raise RuntimeError("probe is broken")

        rt = self._runtime(_boom)
        armed, msg = rt.arm("pp_to_tp", "test")
        self.assertTrue(armed, msg)

    def test_a_frozen_ring_escalates_the_deferral_to_the_named_stop(self):
        """#1173 review, blocker 4 / N2: the deferral is BOUNDED.

        Without a bound a launched pass that never returns converts what used
        to be a crash into a silent hang in PP that only this counter
        distinguishes -- forbidden by the #1153 contract ("no rank ever ends a
        PP0-launched pass silently"). With the stall budget set to zero
        seconds the very next deferral on an unchanged (slots, fwd_ct) must
        raise the SAME named STOP the follower raises.
        """
        rt = self._runtime(lambda: ([1], 81), stall_s=0.001)
        armed, msg = rt.arm("pp_to_tp", "test")
        self.assertFalse(armed)
        self.assertIn("#1173 ARM DEFERRED", msg)
        time.sleep(0.01)
        with self.assertRaises(RuntimeError) as caught:
            rt.arm("pp_to_tp", "test")
        text = str(caught.exception)
        self.assertIn("#1173 LAUNCHED PASS UNEXECUTED UNDER ARM STOP", text)
        self.assertIn("slots=[1]", text)
        self.assertIn("direction=pp_to_tp", text)

    def test_ring_progress_restarts_the_stall_budget(self):
        """Forward progress is the discriminator, not the deferral count.

        `arm` is driven from the receive poll, so N deferrals is a function of
        client traffic, not of time or of the ring being stuck. A ring whose
        forward count advances between two deferrals is working; only a
        FROZEN (slots, fwd_ct) may ever reach the bound.
        """
        counts = iter([81, 82, 83, 84])
        rt = self._runtime(lambda: ([1], next(counts)), stall_s=0.001)
        for _ in range(4):
            time.sleep(0.005)
            armed, msg = rt.arm("pp_to_tp", "test")
            self.assertFalse(armed)
            self.assertIn("#1173 ARM DEFERRED", msg)
        self.assertEqual(rt.arm_deferred_launched, 4)


class TestTheArmPrintsItsTermAndItsProducer(unittest.TestCase):
    """The small #1173 item: an arm line that can be read back."""

    def test_the_pending_breakdown_is_reported_or_named_absent(self):
        from sglang.srt.managers.phase_policy import PhasePolicyInputs, _pending_terms

        self.assertIn("UNREPORTED", _pending_terms(types.SimpleNamespace()))
        self.assertEqual(
            _pending_terms(
                types.SimpleNamespace(pending_prefill_terms="inflight=1912")
            ),
            "inflight=1912",
        )
        # Both #1173 fields exist on the dataclass the policy actually reads.
        inp = PhasePolicyInputs.__dataclass_fields__
        self.assertIn("pending_prefill_terms", inp)
        self.assertIn("seam_witness_states", inp)

    def test_the_witness_census_names_the_state_not_just_the_verdict(self):
        from sglang.srt.managers import phase_purity as pp

        orig_c = pp.seam_readmit_candidates
        orig_w = pp.store_witness
        try:
            reqs = [object(), object(), object()]
            states = {
                id(reqs[0]): "hit",
                id(reqs[1]): "bounded",
                id(reqs[2]): "bounded",
            }
            pp.seam_readmit_candidates = lambda sched: list(reqs)
            pp.store_witness = lambda sched, req: states[id(req)]
            self.assertEqual(pp.store_witness_census(object()), "bounded=2 hit=1")
            pp.seam_readmit_candidates = lambda sched: []
            self.assertIn("none", pp.store_witness_census(object()))
        finally:
            pp.seam_readmit_candidates = orig_c
            pp.store_witness = orig_w


class TestTheRingRebuildForgetsLaunchedSlots(unittest.TestCase):
    """#1173 review, blocker 2/4: no launched slot number outlives its ring.

    `_pp_launched_pending` holds RING-SCOPED SLOT NUMBERS and had no
    cutover-scoped clear. Before #1173 a survivor was near-benign (only the
    #1020 void guard read it); with the arm precondition reading the raw set,
    ONE survivor defers every future arm in BOTH directions for ever -- the
    TP-sticky class, visible only as a growing ARM DEFERRED line. The entry
    CAN survive: the sole discard site sits in the `else` branch after
    `_pp_process_batch_result`, and a launched batch that became EMPTY reads
    quiescent to `build_flip_quiescence_fn`, so the cutover commits with the
    entry still set.
    """

    def test_the_designated_authority_clears_the_launched_set(self):
        holder = types.SimpleNamespace(
            _pp_flip_arm_mb_id=2,
            _pp_flip_arm_epoch=5,
            _pp_flip_resume_slot=1,
            _pp_launched_pending={2},
        )
        ppm.pp_flip_forget_ring_scoped_slots(holder)
        self.assertEqual(holder._pp_launched_pending, set())
        # The three fields #829 already cleared must still be cleared.
        self.assertIsNone(holder._pp_flip_arm_mb_id)
        self.assertIsNone(holder._pp_flip_arm_epoch)
        self.assertIsNone(holder._pp_flip_resume_slot)

    def test_a_slot_from_the_old_ring_cannot_defer_the_next_arm(self):
        """The end-to-end shape: set carries slot 2, ring rebuilds to size 1.

        Slot 2 is not merely stale on the TP ring, it is OUT OF RANGE -- and
        the probe would hand it to `arm()` as an outstanding pass for ever.
        """
        holder = types.SimpleNamespace(
            _pp_flip_arm_mb_id=None,
            _pp_flip_arm_epoch=None,
            _pp_flip_resume_slot=None,
            _pp_launched_pending={2},
            forward_ct=81,
        )
        probe = ppm_build()(holder)
        self.assertEqual(probe(), ([2], 81))
        ppm.pp_flip_forget_ring_scoped_slots(holder)
        self.assertEqual(probe(), ([], 81))


class TestTheLaunchedPassProbe(unittest.TestCase):
    """D2a: the probe the arm precondition reads."""

    def test_it_reports_the_outstanding_slots_and_the_forward_count(self):
        sched = types.SimpleNamespace(_pp_launched_pending={1, 0}, forward_ct=81)
        fn = ppm_build()(sched)
        self.assertEqual(fn(), ([0, 1], 81))

    def test_no_launched_pass_reports_empty(self):
        sched = types.SimpleNamespace(_pp_launched_pending=set(), forward_ct=5)
        self.assertEqual(ppm_build()(sched)(), ([], 5))


def ppm_build():
    from sglang.srt.managers.phase_flip_runtime import build_launched_passes_fn

    return build_launched_passes_fn


if __name__ == "__main__":
    unittest.main()
