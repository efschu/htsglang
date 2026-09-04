# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""#1180: the #631 row-probe's defer must be BOUNDED and end in a named stop.

THE SPECIMEN (boot weg1b7, /spinning/evidence-665-f1/
boot_855_weg1b7_261551467d_0904_034226.log, 03:49:20 -> 03:50:24Z):

    PP0  #631 PROXY-SEND t40 stamp=(2, 40, 2, 4, 157, ('772368da', ...))
    PP0  #888b yielding a parked carrier's request seat (binder=req_slot)
    PP0  #969AD RETRACT site=_retract_decode_and_requeue rid=c25108f3 fwd_ct=157
    PP1  #631 ROW-PROBE DEFER slot=2: ... (first=c25108f3)  x3, same second
    PP0  PP-RECV-OBJ step-expired site=pp:0/recv_object[src=2,tag=0] waited=60.0s
    PP1/PP2  PP-CHAIN-RECV closed ring ... drain turn 1, turn 2
    PP1  Scheduler hit an exception  (60 s chain stall bound -> group dead)

PP0 sealed a row naming ``c25108f3`` and then retracted that rid one
scheduling iteration later. The row can never be satisfied again, so PP1's
peek defers for ever -- and PP0 is already parked on the OUTPUT of the pass
whose frame PP1 refuses to execute. The defer therefore closes the ring by
construction; it is not a latency hedge once the missing set stops moving.

WHAT THESE TESTS PIN, AND WHY THE UNITS ARE LAPS
------------------------------------------------
The bound is counted in RING LAPS, not wall-clock seconds, because a clock
sampled only by this probe is sampled only by a loop the defect halts. Two
pins carry that:

  * `test_a_repeat_defer_does_not_rearm_the_chain_hedge` -- the reachability
    pin. Every defer used to set `_pp_row_chain_owed`, and the next loop
    iteration turns that flag into a BLOCKING chain receive with no bound
    (`SGLANG_PP_CHAIN_RECV_STALL_S` defaults to "0"). Arming it only on the
    FIRST sighting is what lets this rank come back to the probe on its own.
  * `test_the_defer_is_not_unbounded` -- the behavioural pin, and it imports
    NOTHING new on purpose, so its red is a statement about behaviour rather
    than about a missing module.

and the rest:

  * the defer still happens and the frame is never consumed -- on the first
    defer AND at the raise (the boot 631row14 protection);
  * a CHANGING missing set is progress and restarts the count (the boot
    631row15 hot-defer world stays free);
  * an UNCHANGING set past the cap raises a NAMED stop naming slot, rid and
    the disarm knob;
  * the cap is ONE object with the admission layer's `UNRESOLVED_DEFER_CAP`
    -- same question, same unit, one authority.

RED ON THE PARENT (8f45927d14, the wall-clock horizon): the horizon defaults
to 20 s and 4096 tight iterations take milliseconds, so the probe answers
False for ever -- and the horizon re-arms the hedge on every defer. Both
pins above fail there.
"""

from __future__ import annotations

import types
import unittest

from sglang.srt.managers import scheduler_pp_mixin as ppm


def _wire_row(rid: str, *, admitted: bool = True, retracted: bool = False):
    """One admission-decision wire row, in the codec's own tuple shape."""
    return (rid, 0, 1, admitted, retracted, None, None, False, None, (), None)


def _frame(slot: int, epoch: int, rids, *, pass_ct: int = 157):
    """A stamped proxy frame whose row names ``rids``."""
    return {
        "__stamp__": (slot, 40, 2, epoch, pass_ct, ("772368da", 6011, 6012)),
        ppm._ADMISSION_DECISION_PAYLOAD_KEY: (
            slot,
            tuple(_wire_row(r) for r in rids),
        ),
    }


class RowDeferCapProbeTest(unittest.TestCase):
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
        sched._pp_row_chain_owed = False
        sched._pp_flip_epoch = lambda: self.EPOCH
        return sched

    def _probe(self, sched, mb_id=None):
        return ppm.SchedulerPPMixin._pp_proxy_frame_pending(
            sched, self.SLOT if mb_id is None else mb_id
        )

    # ------------------------------------------------------------- protection

    def test_the_first_defer_leaves_the_frame_in_the_inbox(self):
        """The boot 631row14 protection is untouched: no early consume."""
        queue = [_frame(self.SLOT, self.EPOCH, ["c25108f3"])]
        sched = self._sched(queue)
        self.assertIs(self._probe(sched), False)
        self.assertEqual(len(queue), 1, "the deferred frame must stay queued")

    # --------------------------------------------------------- reachability

    def test_a_repeat_defer_does_not_rearm_the_chain_hedge(self):
        """THE REACHABILITY PIN (#1180 B1).

        `_pp_row_chain_owed` opens an UNBOUNDED blocking chain receive on the
        next loop iteration. Arming it on the first sighting is the hedge;
        arming it AGAIN, after a receive already completed without delivering
        the rid, parks this rank on a peer that is not sending and the lap
        count can never advance. So: armed once, never renewed while the
        missing set stands still.
        """
        queue = [_frame(self.SLOT, self.EPOCH, ["c25108f3"])]
        sched = self._sched(queue)

        self.assertIs(self._probe(sched), False)
        self.assertTrue(
            sched._pp_row_chain_owed,
            "the FIRST defer must still arm the hedge (631row15 stays free)",
        )

        # The loop consumes the one-shot flag by entering the receive.
        sched._pp_row_chain_owed = False

        self.assertIs(self._probe(sched), False)
        self.assertFalse(
            sched._pp_row_chain_owed,
            "a repeat defer of the SAME missing set must NOT re-arm the "
            "unbounded chain receive -- that is what makes the lap cap "
            "reachable from this rank alone",
        )

    def test_a_changed_missing_set_arms_the_hedge_again(self):
        """A hop landed: the inference is fresh again, so the hedge is fresh.

        This is the boot 631row15 world (43k hot defers on a turning ring):
        the missing set keeps changing, every defer is a first sighting, and
        the chain receive must keep being entered.
        """
        queue = [_frame(self.SLOT, self.EPOCH, ["c25108f3", "772368da"])]
        sched = self._sched(queue)
        self.assertIs(self._probe(sched), False)
        sched._pp_row_chain_owed = False

        # One of the two rids landed -> a different missing set.
        queue[:] = [_frame(self.SLOT, self.EPOCH, ["c25108f3"])]
        self.assertIs(self._probe(sched), False)
        self.assertTrue(
            sched._pp_row_chain_owed,
            "a CHANGED missing set is a landing hop: arm the hedge again",
        )

    # ----------------------------------------------------------- the bound

    def test_the_defer_is_not_unbounded(self):
        """THE BEHAVIOURAL RED, and it imports nothing new on purpose.

        On the parent this probe answers False for ever -- 4096 consecutive
        defers of the SAME unsatisfiable row, no exception, exactly the
        weg1b7 livelock. The assertion below is what fails there, so the red
        is a statement about BEHAVIOUR and not about a missing module.
        """
        queue = [_frame(self.SLOT, self.EPOCH, ["c25108f3"])]
        sched = self._sched(queue)

        for _ in range(4096):
            try:
                verdict = self._probe(sched)
            except Exception:  # noqa: BLE001 - any named stop passes here
                return
            self.assertIs(verdict, False)
        self.fail(
            "the row probe deferred the same unsatisfiable row 4096 "
            "consecutive times without ever stopping"
        )

    def test_an_unchanging_missing_set_past_the_cap_stops_by_name(self):
        from sglang.srt.managers.pp_row_defer_cap import (
            ROW_DEFER_LAP_CAP,
            PpRowDeferCapExceeded,
        )

        queue = [_frame(self.SLOT, self.EPOCH, ["c25108f3"])]
        sched = self._sched(queue)

        for _ in range(ROW_DEFER_LAP_CAP):
            self.assertIs(self._probe(sched), False)  # still inside the cap

        with self.assertRaises(PpRowDeferCapExceeded) as caught:
            self._probe(sched)

        msg = str(caught.exception)
        self.assertIn("#1180", msg)
        self.assertIn("slot 2", msg)
        self.assertIn("c25108f3", msg)
        self.assertIn("SGLANG_PP_ROW_AUTHORITY=0", msg)
        self.assertIn("UNRESOLVED_DEFER_CAP", msg)

    def test_the_frame_is_still_queued_when_the_cap_lapses(self):
        """The stop must NOT consume the frame.

        Consuming it would trade this deadlock for boot 631row14's: the plan
        finds nothing and the ring dies upstream-waiting. The stop's only
        product is a verdict and a message.
        """
        from sglang.srt.managers.pp_row_defer_cap import (
            ROW_DEFER_LAP_CAP,
            PpRowDeferCapExceeded,
        )

        frame = _frame(self.SLOT, self.EPOCH, ["c25108f3"])
        queue = [frame]
        sched = self._sched(queue)

        for _ in range(ROW_DEFER_LAP_CAP):
            self._probe(sched)
        with self.assertRaises(PpRowDeferCapExceeded):
            self._probe(sched)

        self.assertEqual(len(queue), 1, "the stop must not consume the frame")
        self.assertIs(queue[0], frame, "and must not replace it either")

    def test_the_count_survives_a_swallowed_stop(self):
        """If any future caller ever swallows the stop, the NEXT probe of the
        same identity must raise again at once -- never degrade into a
        periodic raise-and-retry that keeps the ring closed between raises.
        """
        from sglang.srt.managers.pp_row_defer_cap import (
            ROW_DEFER_LAP_CAP,
            PpRowDeferCapExceeded,
        )

        queue = [_frame(self.SLOT, self.EPOCH, ["c25108f3"])]
        sched = self._sched(queue)
        for _ in range(ROW_DEFER_LAP_CAP):
            self._probe(sched)
        with self.assertRaises(PpRowDeferCapExceeded):
            self._probe(sched)
        with self.assertRaises(PpRowDeferCapExceeded):
            self._probe(sched)

    def test_a_changing_missing_set_never_reaches_the_cap(self):
        from sglang.srt.managers.pp_row_defer_cap import PpRowDeferCapExceeded

        queue = [_frame(self.SLOT, self.EPOCH, ["c25108f3"])]
        sched = self._sched(queue)
        try:
            for i in range(64):
                # Every pass the row names a DIFFERENT rid: hops are landing.
                queue[:] = [_frame(self.SLOT, self.EPOCH, [f"rid{i:04d}"])]
                self.assertIs(self._probe(sched), False)
        except PpRowDeferCapExceeded as exc:  # pragma: no cover - failure path
            self.fail(f"a moving missing set must never lapse: {exc}")

    def test_a_new_frame_does_not_inherit_its_predecessors_count(self):
        from sglang.srt.managers.pp_row_defer_cap import (
            ROW_DEFER_LAP_CAP,
            PpRowDeferCapExceeded,
        )

        queue = [_frame(self.SLOT, self.EPOCH, ["c25108f3"])]
        sched = self._sched(queue)
        for _ in range(ROW_DEFER_LAP_CAP):
            self._probe(sched)

        # A NEW frame for the same slot (different pass counter in the stamp)
        # with the same missing set: the count starts from zero.
        queue[:] = [_frame(self.SLOT, self.EPOCH, ["c25108f3"], pass_ct=158)]
        try:
            for _ in range(ROW_DEFER_LAP_CAP):
                self.assertIs(self._probe(sched), False)
        except PpRowDeferCapExceeded as exc:  # pragma: no cover
            self.fail(f"a new frame inherited its predecessor's count: {exc}")

    # ------------------------------------------------------------- clearing

    def test_a_locatable_row_neither_defers_nor_arms_a_count(self):
        queue = [_frame(self.SLOT, self.EPOCH, ["c25108f3"])]
        sched = self._sched(queue)
        sched.running_batch = types.SimpleNamespace(
            reqs=[types.SimpleNamespace(rid="c25108f3")]
        )
        self.assertIs(self._probe(sched), True)
        self.assertIsNone(
            getattr(sched, "_pp_row_defer_cap", None),
            "a satisfied row must not construct a lap counter at all",
        )
        self.assertFalse(sched._pp_row_chain_owed)

    def test_a_satisfied_row_clears_the_count_its_own_defer_armed(self):
        """The clear is only meaningful once a count EXISTS.

        The locatable-row test above cannot see the clear (there is no
        counter to inspect), so this one arms the count first and then
        satisfies the row.
        """
        queue = [_frame(self.SLOT, self.EPOCH, ["c25108f3"])]
        sched = self._sched(queue)
        self.assertIs(self._probe(sched), False)

        cap = getattr(sched, "_pp_row_defer_cap", None)
        self.assertIsNotNone(cap, "the defer must have armed a lap count")
        self.assertIn(self.SLOT, cap._laps)

        sched.running_batch = types.SimpleNamespace(
            reqs=[types.SimpleNamespace(rid="c25108f3")]
        )
        self.assertIs(self._probe(sched), True)
        self.assertNotIn(
            self.SLOT,
            cap._laps,
            "a satisfied row must forget the slot's lap count",
        )


class RowDeferCapUnitTest(unittest.TestCase):
    """The pure decision object, without the probe around it."""

    def test_the_cap_is_the_admission_layers_cap_object(self):
        """ONE authority for both unresolvable-rid defers, not two numbers.

        Asserted STRUCTURALLY, on the binding in the source. An `assertIs`
        on the values is vacuous here: both are the small int 3 and CPython
        interns it, so `ROW_DEFER_LAP_CAP = 3` would pass an identity check
        while forking the number -- a test that cannot fail on the very
        mutation it exists to catch. The equality below is kept as a second,
        weaker statement; the AST check is the one that bites.
        """
        import ast
        import inspect

        from sglang.srt.managers import pp_row_defer_cap as mod
        from sglang.srt.managers.pp_admission_congruence import (
            UNRESOLVED_DEFER_CAP,
        )

        self.assertEqual(mod.ROW_DEFER_LAP_CAP, UNRESOLVED_DEFER_CAP)

        tree = ast.parse(inspect.getsource(mod))
        bound_to = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "ROW_DEFER_LAP_CAP"
                for t in node.targets
            ):
                bound_to = node.value
        self.assertIsNotNone(bound_to, "ROW_DEFER_LAP_CAP must be bound")
        self.assertIsInstance(
            bound_to,
            ast.Name,
            "the cap must be bound to the admission layer's NAME, never to "
            "a literal -- a literal is a second accounting of the same "
            "question in the same unit",
        )
        self.assertEqual(bound_to.id, "UNRESOLVED_DEFER_CAP")

        imported = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "sglang.srt.managers.pp_admission_congruence"
            for alias in node.names
        }
        self.assertIn("UNRESOLVED_DEFER_CAP", imported)

    def test_the_first_sighting_always_defers(self):
        from sglang.srt.managers.pp_row_defer_cap import RowDeferCap

        c = RowDeferCap()
        v = c.observe(0, ["a"], token=("t",))
        self.assertTrue(v.defer)
        self.assertEqual(v.occurrence, 1)
        self.assertIsNone(v.message)

    def test_the_cap_is_inclusive_at_its_own_value(self):
        """`cap` consecutive observations still defer; `cap`+1 stops."""
        from sglang.srt.managers.pp_row_defer_cap import RowDeferCap

        c = RowDeferCap()
        for i in range(3):
            self.assertTrue(c.observe(0, ["a"], token=("t",), cap=3).defer, i)
        self.assertFalse(c.observe(0, ["a"], token=("t",), cap=3).defer)

    def test_zero_disables_the_bound_and_restores_todays_behaviour(self):
        from sglang.srt.managers.pp_row_defer_cap import RowDeferCap

        c = RowDeferCap()
        for _ in range(1000):
            self.assertTrue(c.observe(0, ["a"], token=("t",), cap=0).defer)

    def test_clear_forgets_one_slot_only(self):
        from sglang.srt.managers.pp_row_defer_cap import RowDeferCap

        c = RowDeferCap()
        c.observe(0, ["a"], token=("t",))
        c.observe(1, ["a"], token=("t",))
        c.clear(0)
        self.assertEqual(c.observe(0, ["a"], token=("t",)).occurrence, 1)
        self.assertEqual(c.observe(1, ["a"], token=("t",)).occurrence, 2)

    def test_a_different_token_restarts_the_count(self):
        from sglang.srt.managers.pp_row_defer_cap import RowDeferCap

        c = RowDeferCap()
        c.observe(0, ["a"], token=("t1",))
        c.observe(0, ["a"], token=("t1",))
        self.assertEqual(c.observe(0, ["a"], token=("t2",)).occurrence, 1)

    def test_an_unhashable_token_degrades_instead_of_raising(self):
        from sglang.srt.managers.pp_row_defer_cap import RowDeferCap

        c = RowDeferCap()
        v = c.observe(0, ["a"], token=["unhashable"])
        self.assertTrue(v.defer)
        self.assertEqual(c.observe(0, ["a"], token=["other"]).occurrence, 2)

    def test_slots_do_not_share_a_count(self):
        from sglang.srt.managers.pp_row_defer_cap import RowDeferCap

        c = RowDeferCap()
        for _ in range(8):
            c.observe(0, ["a"], token=("t",), cap=0)
        self.assertEqual(c.observe(1, ["a"], token=("t",)).occurrence, 1)

    def test_the_message_names_every_term(self):
        from sglang.srt.managers.pp_row_defer_cap import RowDeferCap

        c = RowDeferCap()
        v = None
        for _ in range(4):
            v = c.observe(7, ["deadbeefcafe"], token=("t",), cap=3)
        self.assertFalse(v.defer)
        for term in (
            "#1180",
            "slot 7",
            "deadbeef",
            "LAPS, NOT SECONDS",
            "UNRESOLVED_DEFER_CAP",
            "SGLANG_PP_ROW_AUTHORITY=0",
        ):
            self.assertIn(term, v.message)


if __name__ == "__main__":
    unittest.main()
