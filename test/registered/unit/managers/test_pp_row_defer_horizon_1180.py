# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""#1180: the #631 row-probe's defer must be BOUNDED and end in a named stop.

THE SPECIMEN (boot weg1b7, /spinning/evidence-665-f1/
boot_855_weg1b7_261551467d_0904_034226.log, 03:49:20 -> 03:50:24Z):

    PP0  #631 PROXY-SEND t40 stamp=(2, 40, 2, 4, 157, ('772368da', ...))
    PP0  #888b yielding a parked carrier's request seat (binder=req_slot)
    PP0  #969AD RETRACT site=_retract_decode_and_requeue rid=c25108f3 fwd_ct=157
    PP1  #631 ROW-PROBE DEFER slot=2: ... (first=c25108f3) (occurrence=1..3)
    PP0  PP-RECV-OBJ step-expired site=pp:0/recv_object[src=2,tag=0] waited=60.0s
    PP1/PP2  PP-CHAIN-RECV closed ring ... drain turn 1, turn 2
    PP1  Scheduler hit an exception  (60 s chain stall bound -> group dead)

PP0 sealed a row naming ``c25108f3`` and then retracted that rid one
scheduling iteration later. The row can never be satisfied again, so PP1's
peek defers for ever -- and PP0 is already parked on the OUTPUT of the pass
whose frame PP1 refuses to execute. The defer therefore closes the ring by
construction; it is not a latency hedge once the missing set stops moving.

WHAT THESE TESTS PIN
  * the defer still happens (the boot 631row14 protection is untouched, and
    the frame is never consumed early);
  * a CHANGING missing set is progress and restarts the clock (the boot
    631row15 hot-defer world stays free);
  * an UNCHANGING missing set past the horizon raises a NAMED stop that
    carries the slot, the rid and the remedy knob;
  * the bound is operator-disableable back to today's exact behaviour.

RED ON THE PARENT (261551467d): the horizon does not exist there, so the
probe answers False for ever and `test_an_unchanging_missing_set_past_the_
horizon_stops_by_name` fails on the missing raise -- a behavioural red, not
an import error. The module-level import of `pp_row_defer_horizon` is
deliberately LAZY inside the tests that need it for exactly that reason.
"""

from __future__ import annotations

import types
import unittest

from sglang.srt.managers import scheduler_pp_mixin as ppm


def _wire_row(rid: str, *, admitted: bool = True, retracted: bool = False):
    """One admission-decision wire row, in the codec's own tuple shape."""
    return (rid, 0, 1, admitted, retracted, None, None, False, None, (), None)


def _frame(slot: int, epoch: int, rids):
    """A stamped proxy frame whose row names ``rids``."""
    return {
        "__stamp__": (slot, 40, 2, epoch, 157, ("772368da", 6011, 6012)),
        ppm._ADMISSION_DECISION_PAYLOAD_KEY: (
            slot,
            tuple(_wire_row(r) for r in rids),
        ),
    }


class RowDeferHorizonTest(unittest.TestCase):
    """Drive `_pp_proxy_frame_pending` through a stand-in holder."""

    SLOT = 2
    EPOCH = 2

    def setUp(self):
        self._orig_src = ppm.resolve_src
        self._orig_inbox = ppm.typed_inbox
        self.addCleanup(self._restore)

    def _restore(self):
        ppm.resolve_src = self._orig_src
        ppm.typed_inbox = self._orig_inbox

    def _sched(self, queue):
        ppm.resolve_src = lambda group, x: 0
        ppm.typed_inbox = lambda group: {(0, "proxy"): queue}
        sched = types.SimpleNamespace()
        sched.pp_group = object()
        sched.ps = types.SimpleNamespace(pp_rank=1, pp_size=3)
        # The four containers the peek consults for `_known`. All empty, so
        # every rid the row names is "not locatable here" -- the specimen.
        sched.waiting_queue = []
        sched.chunked_req = None
        sched.running_batch = None
        sched._pp_flip_epoch = lambda: self.EPOCH
        return sched

    def _probe(self, sched, mb_id=None):
        return ppm.SchedulerPPMixin._pp_proxy_frame_pending(
            sched, self.SLOT if mb_id is None else mb_id
        )

    # ---------------------------------------------------------------- protection

    def test_the_first_defer_leaves_the_frame_in_the_inbox(self):
        """The boot 631row14 protection is untouched: no early consume."""
        queue = [_frame(self.SLOT, self.EPOCH, ["c25108f3"])]
        sched = self._sched(queue)
        self.assertIs(self._probe(sched), False)
        self.assertEqual(len(queue), 1, "the deferred frame must stay queued")

    # ---------------------------------------------------------------- the bound

    def test_the_defer_is_not_unbounded(self):
        """THE BEHAVIOURAL RED, and it imports nothing new on purpose.

        On the parent this probe answers False for ever -- 4096 consecutive
        defers of the SAME unsatisfiable row, no exception, exactly the
        weg1b7 livelock. The assertion below is what fails there, so the red
        is a statement about BEHAVIOUR and not about a missing module.
        """
        queue = [_frame(self.SLOT, self.EPOCH, ["c25108f3"])]
        sched = self._sched(queue)
        clock = [1000.0]
        sched._pp_row_defer_now = lambda: clock[0]

        for _ in range(4096):
            clock[0] += 1.0
            try:
                self._probe(sched)
            except RuntimeError as exc:
                self.assertIn("#1180", str(exc))
                return
        self.fail(
            "the row probe deferred the same unsatisfiable row 4096 times "
            "over an hour of its own clock without ever stopping -- that is "
            "the weg1b7 ring, and the sender is meanwhile parked on the "
            "output of the pass this frame belongs to"
        )

    def test_an_unchanging_missing_set_past_the_horizon_stops_by_name(self):
        """THE #1180 SPECIMEN. Same missing rid, held past the bound -> STOP.

        RED ON THE PARENT: there the probe answers False for ever and no
        exception is raised, so `assertRaises` fails.
        """
        from sglang.srt.managers.pp_row_defer_horizon import (
            HORIZON_ENV,
            PpRowDeferHorizonLapsed,
        )

        queue = [_frame(self.SLOT, self.EPOCH, ["c25108f3"])]
        sched = self._sched(queue)
        clock = [1000.0]
        sched._pp_row_defer_now = lambda: clock[0]

        self.assertIs(self._probe(sched), False)  # starts the clock
        clock[0] += 1.0
        self.assertIs(self._probe(sched), False)  # still inside the horizon

        clock[0] += 3600.0
        with self.assertRaises(PpRowDeferHorizonLapsed) as caught:
            self._probe(sched)

        msg = str(caught.exception)
        self.assertIn("#1180", msg)
        self.assertIn("slot 2", msg)
        self.assertIn("c25108f3", msg)
        self.assertIn(HORIZON_ENV, msg)
        # It names the two measured shapes, so the next reader does not have
        # to re-derive them from the boot log.
        self.assertIn("RETRACTED", msg)
        self.assertIn("re-admitted resident", msg)

    def test_a_changing_missing_set_restarts_the_clock(self):
        """Progress is observable: a landing hop changes the missing set.

        The boot 631row15 world (43k hot defers on a TURNING ring) must never
        reach the bound, and this is the mechanism that keeps it free.
        """
        from sglang.srt.managers.pp_row_defer_horizon import (
            PpRowDeferHorizonLapsed,
        )

        queue = [_frame(self.SLOT, self.EPOCH, ["aaaaaaaa", "bbbbbbbb"])]
        sched = self._sched(queue)
        clock = [1000.0]
        sched._pp_row_defer_now = lambda: clock[0]

        self.assertIs(self._probe(sched), False)
        # A hop lands: one rid becomes locatable, so the missing set shrinks.
        clock[0] += 3600.0
        sched.waiting_queue = [types.SimpleNamespace(rid="aaaaaaaa")]
        queue[:] = [_frame(self.SLOT, self.EPOCH, ["aaaaaaaa", "bbbbbbbb"])]
        self.assertIs(self._probe(sched), False)
        # ...and the clock restarted, so the very next probe must not stop.
        clock[0] += 1.0
        try:
            self.assertIs(self._probe(sched), False)
        except PpRowDeferHorizonLapsed as exc:  # pragma: no cover - failure path
            self.fail(f"progress must restart the clock, got: {exc}")

    def test_a_new_frame_does_not_inherit_its_predecessors_age(self):
        """A slot whose head alternates must not age across frames.

        The probe only observes a slot while ITS frame is at the head. If the
        ring hands the head to another slot for a while and then returns a
        FRESH frame for this one -- same unlocatable rid, new pass -- that new
        frame is a new wait, not the continuation of an old one. Without the
        frame token in the clock's identity this test stops a healthy ring.
        """
        from sglang.srt.managers.pp_row_defer_horizon import (
            PpRowDeferHorizonLapsed,
        )

        queue = [_frame(self.SLOT, self.EPOCH, ["c25108f3"])]
        sched = self._sched(queue)
        clock = [1000.0]
        sched._pp_row_defer_now = lambda: clock[0]

        self.assertIs(self._probe(sched), False)
        # A whole horizon passes while this slot's frame is NOT at the head,
        # then a new pass ships a new frame naming the same absent rid.
        clock[0] += 3600.0
        queue[:] = [
            {
                "__stamp__": (self.SLOT, 41, 2, self.EPOCH, 158, ("772368da", 0, 1)),
                ppm._ADMISSION_DECISION_PAYLOAD_KEY: (
                    self.SLOT,
                    (_wire_row("c25108f3"),),
                ),
            }
        ]
        try:
            self.assertIs(self._probe(sched), False)
        except PpRowDeferHorizonLapsed as exc:  # pragma: no cover - failure path
            self.fail(f"a fresh frame must start its own clock, got: {exc}")

    def test_zero_disables_the_bound_and_restores_todays_behaviour(self):
        """`SGLANG_PP_ROW_DEFER_HORIZON_S=0` = defer for ever, exactly."""
        import os

        from sglang.srt.managers.pp_row_defer_horizon import (
            HORIZON_ENV,
            PpRowDeferHorizonLapsed,
        )

        queue = [_frame(self.SLOT, self.EPOCH, ["c25108f3"])]
        sched = self._sched(queue)
        clock = [1000.0]
        sched._pp_row_defer_now = lambda: clock[0]

        prior = os.environ.get(HORIZON_ENV)
        os.environ[HORIZON_ENV] = "0"
        try:
            for _ in range(4):
                clock[0] += 3600.0
                try:
                    self.assertIs(self._probe(sched), False)
                except PpRowDeferHorizonLapsed as exc:  # pragma: no cover
                    self.fail(f"0 must disable the bound, got: {exc}")
        finally:
            if prior is None:
                os.environ.pop(HORIZON_ENV, None)
            else:
                os.environ[HORIZON_ENV] = prior

    def test_a_locatable_row_neither_defers_nor_arms_a_clock(self):
        """The healthy pass: every named rid is here, so the slot proceeds."""
        queue = [_frame(self.SLOT, self.EPOCH, ["c25108f3"])]
        sched = self._sched(queue)
        sched.waiting_queue = [types.SimpleNamespace(rid="c25108f3")]
        clock = [1000.0]
        sched._pp_row_defer_now = lambda: clock[0]

        self.assertIs(self._probe(sched), True)
        self.assertIsNone(
            getattr(sched, "_pp_row_defer_horizon", None),
            "a row that was never missing must not even construct a clock",
        )

    def test_a_satisfied_row_clears_the_clock_its_own_defer_armed(self):
        """The RECOVERY path, and the only test that can see the clear.

        The previous test cannot: with nothing ever missing there is no
        horizon object to inspect, so its assertion is vacuous about the
        clear. Here the SAME slot defers first (arming the clock), the hop
        then lands, and the satisfied exit must forget the clock -- otherwise
        a later, unrelated defer on this slot inherits an arbitrarily old
        start time and stops a healthy ring at its first probe.
        """
        queue = [_frame(self.SLOT, self.EPOCH, ["c25108f3"])]
        sched = self._sched(queue)
        clock = [1000.0]
        sched._pp_row_defer_now = lambda: clock[0]

        self.assertIs(self._probe(sched), False)  # arms the clock
        horizon = getattr(sched, "_pp_row_defer_horizon", None)
        self.assertIsNotNone(horizon, "the defer must have armed a clock")
        self.assertIn(self.SLOT, horizon._since)

        # The hop lands and the very same frame now names a locatable rid.
        sched.waiting_queue = [types.SimpleNamespace(rid="c25108f3")]
        clock[0] += 1.0
        self.assertIs(self._probe(sched), True)
        self.assertEqual(
            horizon._since,
            {},
            "a satisfied row must leave no clock behind for the next frame",
        )


class RowDeferHorizonUnitTest(unittest.TestCase):
    """The pure decision object, without the probe around it."""

    def test_the_first_sighting_always_defers(self):
        from sglang.srt.managers.pp_row_defer_horizon import RowDeferHorizon

        h = RowDeferHorizon()
        v = h.observe(0, ["x"], now=100.0, bound_s=5.0)
        self.assertTrue(v.defer)
        self.assertEqual(v.occurrence, 1)
        self.assertEqual(v.waited_s, 0.0)
        self.assertIsNone(v.message)

    def test_the_bound_is_exclusive_at_its_own_value(self):
        """`waited == bound` still defers; only strictly past it stops."""
        from sglang.srt.managers.pp_row_defer_horizon import RowDeferHorizon

        h = RowDeferHorizon()
        h.observe(0, ["x"], now=100.0, bound_s=5.0)
        self.assertTrue(h.observe(0, ["x"], now=105.0, bound_s=5.0).defer)
        self.assertFalse(h.observe(0, ["x"], now=105.01, bound_s=5.0).defer)

    def test_clear_forgets_one_slot_only(self):
        from sglang.srt.managers.pp_row_defer_horizon import RowDeferHorizon

        h = RowDeferHorizon()
        h.observe(0, ["x"], now=100.0, bound_s=5.0)
        h.observe(1, ["y"], now=100.0, bound_s=5.0)
        h.clear(0)
        self.assertTrue(h.observe(0, ["x"], now=200.0, bound_s=5.0).defer)
        self.assertFalse(h.observe(1, ["y"], now=200.0, bound_s=5.0).defer)

    def test_a_different_token_restarts_the_clock(self):
        from sglang.srt.managers.pp_row_defer_horizon import RowDeferHorizon

        h = RowDeferHorizon()
        h.observe(0, ["x"], token=("a",), now=100.0, bound_s=5.0)
        v = h.observe(0, ["x"], token=("b",), now=200.0, bound_s=5.0)
        self.assertTrue(v.defer, "a new frame is a new wait")
        self.assertEqual(v.occurrence, 1)

    def test_an_unhashable_token_degrades_instead_of_raising(self):
        """A stamp this module cannot key on must never kill a boot."""
        from sglang.srt.managers.pp_row_defer_horizon import RowDeferHorizon

        h = RowDeferHorizon()
        h.observe(0, ["x"], token=["unhashable"], now=100.0, bound_s=5.0)
        v = h.observe(0, ["x"], token=["unhashable"], now=200.0, bound_s=5.0)
        self.assertFalse(v.defer, "it still ages on the missing set alone")

    def test_slots_do_not_share_a_clock(self):
        from sglang.srt.managers.pp_row_defer_horizon import RowDeferHorizon

        h = RowDeferHorizon()
        h.observe(0, ["x"], now=100.0, bound_s=5.0)
        v = h.observe(1, ["x"], now=200.0, bound_s=5.0)
        self.assertTrue(v.defer, "slot 1's first sighting is its own")


if __name__ == "__main__":
    unittest.main()
