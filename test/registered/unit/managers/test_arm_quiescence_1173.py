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
        req = _req("b64dc1cb", 6008, 4096, 6008)
        sched = self._double([_Mb([req])], running=[req])
        self.assertEqual(
            Scheduler._pending_prefill_tokens(sched, None, include_health=False), 1912
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
                )
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
        self._queue = queue
        return sched

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
            for _ in range(64):
                ppm.SchedulerPPMixin._pp_flip_hold_slot(sched)
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
        for _ in range(4):
            self.assertFalse(ppm.SchedulerPPMixin._pp_flip_hold_slot(sched))
        queue.clear()
        self.assertTrue(ppm.SchedulerPPMixin._pp_flip_hold_slot(sched))
        queue.append({"__stamp__": (0, 7, 8, 2, 83, ("aa", 0, 8))})
        # Counter restarted: a fresh frame gets its own full bound.
        for _ in range(6):
            self.assertFalse(ppm.SchedulerPPMixin._pp_flip_hold_slot(sched))


class TestQuiescenceIsAnArmPrecondition(unittest.TestCase):
    """D2a: PP0 defers the arm while it holds a launched pass."""

    def _runtime(self, launched):
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


class TestTheArmPrintsItsTermAndItsProducer(unittest.TestCase):
    """The small #1173 item: an arm line that can be read back."""

    def test_the_pending_breakdown_is_reported_or_named_absent(self):
        from sglang.srt.managers.phase_policy import PhasePolicyInputs, _pending_terms

        self.assertIn("UNREPORTED", _pending_terms(types.SimpleNamespace()))
        self.assertEqual(
            _pending_terms(types.SimpleNamespace(pending_prefill_terms="inflight=1912")),
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
            states = {id(reqs[0]): "hit", id(reqs[1]): "bounded", id(reqs[2]): "bounded"}
            pp.seam_readmit_candidates = lambda sched: list(reqs)
            pp.store_witness = lambda sched, req: states[id(req)]
            self.assertEqual(pp.store_witness_census(object()), "bounded=2 hit=1")
            pp.seam_readmit_candidates = lambda sched: []
            self.assertIn("none", pp.store_witness_census(object()))
        finally:
            pp.seam_readmit_candidates = orig_c
            pp.store_witness = orig_w


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
