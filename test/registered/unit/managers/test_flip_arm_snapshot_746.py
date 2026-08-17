"""#746: the parked extent is a SNAPSHOT TAKEN AT ARM, not a memory of the
last enumeration that happened to see requests.

WHAT #744/#748 LEFT INEXACT (TICKET_746). ``_flip_pending()`` answered from a
sticky attribute on the live-slot enumeration -- the last enumeration that saw
requests -- consulted only while a flip is armed. Conservative, but wrong in
two named ways:

1. a flip that arms before ANY enumeration saw requests has no sticky value,
   the probe answers ``(-1, -1)`` = UNKNOWN, and #748's ``_parked_ceiling``
   turns that into the one remaining WHOLESALE refusal (``-2``) for the whole
   flip, on no evidence;
2. the resident set can change between the last enumeration and the arm, so
   the remembered extent is stale in either direction.

The correct answer is "the rows this flip will pack", and that is fixed at
ARM: the requests quiesce AFTER arming, so the arm instant is the last moment
the resident set is both enumerable and final. #746 measures it exactly there,
stores it on the controller, and the rung reads that snapshot.

THE TRAP, named in the filing: the snapshot must clear on BOTH exits -- commit
AND abandon (all four abandon paths). A snapshot that outlives its flip pins
the rung permanently, which is the M5 failure mode #744's mutation matrix
refuses ("gate ALWAYS on -- rung dead outside flips"). Two independent
defences are pinned below: every exit site clears the attribute, and the
``parked_extent()`` accessor refuses to answer while no flip is pending -- so
even a missed clear cannot pin the rung.

NOT TOUCHED, per the filing's DO-NOT: #748's exclusion-ceiling semantics and
its UNKNOWN-while-armed refusal stand. #746 only makes UNKNOWN rare (the
snapshot exists from arm to exit unless the arm-time measurement itself
failed), it does not remove the refusal.
"""

import ast
import inspect
import textwrap
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# The #744 specimen's numbers, reused so the suites talk about the same flip.
PARKED_ROWS = 127_182
PARKED_TOP = 183_998


def _runtime(pending=None, snapshot=None, live_split=None, live_raises=False):
    """A PhaseFlipRuntime carrying only what the tested methods read
    (the #717 stub idiom, as used by test_evict_rung_flip_park_744)."""
    from sglang.srt.managers import phase_flip_runtime as m

    rt = m.PhaseFlipRuntime.__new__(m.PhaseFlipRuntime)
    rt._pending = pending
    rt._parked_extent = snapshot

    if live_raises:

        def _live():
            raise RuntimeError("enumeration unavailable")

    else:

        def _live():
            return None

        _live.last_split = live_split
    rt._live_slots_fn = _live
    return rt


def _armable_runtime(live_split):
    """A stub complete enough to drive the REAL ``arm()`` end to end."""
    from sglang.srt.managers import phase_flip_runtime as m

    rt = _runtime(pending=None, snapshot=None, live_split=live_split)
    rt.blocking_guards = ()
    rt._phase = m.PHASE_PP  # makes PP_TO_TP the legal direction
    rt._park_deadline_s = 30.0
    rt._clock = lambda: 0.0
    rt._prearm_floor_relief = lambda direction: (True, "")
    rt._pool_census = lambda label, direction: None
    return rt


class TestTheSnapshotIsMeasuredAtArm(CustomTestCase):
    """Acceptance 1 of TICKET_746: a flip that arms with no prior enumeration
    reports the REAL extent instead of UNKNOWN. Driven through the real
    ``arm()``, not through a paraphrase of it."""

    def test_arm_captures_the_extent_from_a_fresh_enumeration(self):
        from sglang.srt.managers.phase_flip_runtime import PP_TO_TP

        rt = _armable_runtime(
            {"req_rows": PARKED_ROWS, "req_max": PARKED_TOP, "tree_rows": 0}
        )
        ok, _msg = rt.arm(PP_TO_TP, source="test")
        self.assertTrue(ok)
        self.assertEqual(rt.parked_extent(), (PARKED_ROWS, PARKED_TOP))

    def test_an_idle_arm_snapshots_exactly_nothing(self):
        """Case 1 turns from "blocks" into "blocks exactly what it should":
        zero resident rows at arm is the EXACT answer (0, -1), never
        UNKNOWN."""
        from sglang.srt.managers.phase_flip_runtime import PP_TO_TP

        rt = _armable_runtime({"req_rows": 0, "req_max": -1, "tree_rows": 5})
        ok, _msg = rt.arm(PP_TO_TP, source="test")
        self.assertTrue(ok)
        self.assertEqual(rt.parked_extent(), (0, -1))

    def test_a_failing_measurement_stays_unknown_not_empty(self):
        """The #744 axiom carries over: UNKNOWN IS NOT EMPTY. If the arm-time
        enumeration cannot be read, the snapshot is None and the rung's
        UNKNOWN path (block while armed) takes over -- never (0, -1)."""
        rt = _runtime(live_raises=True)
        self.assertIsNone(rt._snapshot_parked_extent())

    def test_a_missing_split_is_unknown_too(self):
        rt = _runtime(live_split=None)
        self.assertIsNone(rt._snapshot_parked_extent())


class TestTheSnapshotCannotOutliveItsFlip(CustomTestCase):
    """THE TRAP (M5 analog). Both defences, pinned separately so a mutation
    that removes either one turns red."""

    def test_accessor_refuses_while_no_flip_is_pending(self):
        """Defence 2: even a snapshot that somehow survived an exit is
        unreadable once ``_pending`` is None -- the accessor is gated the
        same way the rung's consult is gated on armed."""
        rt = _runtime(pending=None, snapshot=(PARKED_ROWS, PARKED_TOP))
        self.assertIsNone(
            rt.parked_extent(),
            "a snapshot outliving its flip would pin the rung permanently "
            "(the M5 failure mode); the accessor must refuse",
        )

    def test_accessor_answers_while_pending(self):
        from sglang.srt.managers.phase_flip_runtime import PP_TO_TP

        rt = _runtime(pending=PP_TO_TP, snapshot=(PARKED_ROWS, PARKED_TOP))
        self.assertEqual(rt.parked_extent(), (PARKED_ROWS, PARKED_TOP))

    def test_every_disarm_site_clears_the_snapshot(self):
        """Defence 1, structural: EVERY function that disarms (assigns
        ``self._pending = None``) must also clear ``self._parked_extent`` --
        commit and all four abandon paths alike, parsed from the source so a
        future exit path cannot forget the clear without turning this red."""
        from sglang.srt.managers import phase_flip_runtime as m

        tree = ast.parse(
            textwrap.dedent(inspect.getsource(m.PhaseFlipRuntime))
        )

        def _count_assigns(fn, attr):
            n = 0
            for node in ast.walk(fn):
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if (
                            isinstance(t, ast.Attribute)
                            and t.attr == attr
                            and isinstance(t.value, ast.Name)
                            and t.value.id == "self"
                            and isinstance(node.value, ast.Constant)
                            and node.value.value is None
                        ):
                            n += 1
            return n

        disarm_fns = [
            fn
            for fn in ast.walk(tree)
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
            and _count_assigns(fn, "_pending") > 0
        ]
        # Five SITES across at least four FUNCTIONS: three dedicated abandon
        # paths, plus the fit/frame abandon and the commit, which share the
        # execute path. The site count is the inventory this suite audits.
        total_sites = sum(_count_assigns(fn, "_pending") for fn in disarm_fns)
        self.assertGreaterEqual(
            total_sites,
            5,
            "expected the commit site plus four abandon sites; the exit "
            "inventory changed -- re-audit the snapshot lifecycle",
        )
        for fn in disarm_fns:
            with self.subTest(exit_site=fn.name):
                self.assertGreaterEqual(
                    _count_assigns(fn, "_parked_extent"),
                    _count_assigns(fn, "_pending"),
                    f"{fn.name} disarms without clearing the arm-time "
                    "snapshot at every site -- a snapshot would outlive "
                    "its flip there",
                )

    def test_abandon_paths_clear_behaviorally(self):
        """The three abandon paths that are callable hermetically, driven for
        real rather than only parsed."""
        import logging

        from sglang.srt.managers import phase_flip_runtime as m

        logging.getLogger(m.__name__).setLevel(logging.CRITICAL)

        def _fresh():
            rt = _runtime(pending=m.PP_TO_TP, snapshot=(PARKED_ROWS, PARKED_TOP))
            rt._armed_at = 0.0
            rt._last_hold_reason = None
            rt._clock = lambda: 1.0
            rt.presence_timeouts = 0
            rt.join_deadline_aborts = 0
            rt.park_deadline_aborts = 0
            rt._phase = m.PHASE_PP
            rt._park_deadline_s = 30.0
            rt._presence_deadline_s = 30.0
            return rt

        cases = (
            ("no_quorum", lambda rt: rt._abandon_no_quorum(1, ["r1"], 5.0)),
            ("unjoined", lambda rt: rt._abandon_unjoined_flip("test why")),
            ("parked", lambda rt: rt._abandon_parked_flip(0)),
        )
        for name, call in cases:
            with self.subTest(abandon=name):
                rt = _fresh()
                call(rt)
                self.assertIsNone(rt._parked_extent)
                self.assertIsNone(rt.parked_extent())


class TestTheRungReadsTheSnapshot(CustomTestCase):
    """The wiring half: ``kv_backing_provider``'s ``_flip_pending`` closure
    must read the controller snapshot, and the sticky last-enumeration channel
    must be GONE -- removed at writer and reader both, not half-retired."""

    def test_the_factory_reads_parked_extent(self):
        from sglang.srt.managers import kv_backing_relief as m

        src = inspect.getsource(m.kv_backing_provider)
        self.assertIn("parked_extent", src)

    def test_the_sticky_channel_is_gone_from_the_reader(self):
        from sglang.srt.managers import kv_backing_relief as m

        src = inspect.getsource(m.kv_backing_provider)
        self.assertNotIn(
            "last_req_extent",
            src,
            "the reader still consults the last-seen-enumeration value; "
            "TICKET_746 replaces it with the arm-time snapshot",
        )

    def test_the_sticky_channel_is_gone_from_the_writer(self):
        from sglang.srt.managers import phase_flip_runtime as m

        src = inspect.getsource(m.build_flip_live_slots_fn)
        self.assertNotIn(
            "last_req_extent",
            src,
            "the writer still maintains the sticky value nothing reads",
        )

    def test_the_reader_answers_unknown_not_empty_without_a_snapshot(self):
        """AST pin on the production closure itself: while armed with no
        readable snapshot it must return ``(-1, -1)`` -- UNKNOWN IS NOT
        EMPTY, the #744 axiom. A source-string pin cannot see this (the
        function would still mention ``parked_extent``), so the return
        tuples are parsed."""
        from sglang.srt.managers import kv_backing_relief as m

        tree = ast.parse(textwrap.dedent(inspect.getsource(m.kv_backing_provider)))
        closure = next(
            (
                fn
                for fn in ast.walk(tree)
                if isinstance(fn, ast.FunctionDef) and fn.name == "_flip_pending"
            ),
            None,
        )
        self.assertIsNotNone(closure, "the _flip_pending closure is gone")

        def _tuple_returns(fn):
            out = []
            for node in ast.walk(fn):
                if isinstance(node, ast.Return) and isinstance(
                    node.value, ast.Tuple
                ):
                    vals = []
                    for e in node.value.elts:
                        if isinstance(e, ast.Constant):
                            vals.append(e.value)
                        elif (
                            isinstance(e, ast.UnaryOp)
                            and isinstance(e.op, ast.USub)
                            and isinstance(e.operand, ast.Constant)
                        ):
                            vals.append(-e.operand.value)
                        else:
                            vals.append(None)
                    out.append(tuple(vals))
            return out

        self.assertIn(
            (-1, -1),
            _tuple_returns(closure),
            "the closure no longer has an UNKNOWN return; a missing "
            "snapshot while armed would read as empty and re-open the "
            "#744 specimen",
        )

    def test_pending_probe_semantics_through_a_runtime(self):
        """The closure's contract, driven against a real runtime object:
        exact while armed with a snapshot, UNKNOWN while armed without one,
        inert outside a flip -- including with a STALE snapshot, which is the
        M5-analog at the reader."""
        from sglang.srt.managers import phase_flip_runtime as pfr

        class _Sched:
            phase_flip_runtime = None

        sched = _Sched()

        def _armed():
            rt = sched.phase_flip_runtime
            return rt is not None and rt._pending is not None

        def _pending():
            # Production shape (kv_backing_provider): armed -> snapshot,
            # no snapshot -> UNKNOWN, not armed -> nothing parked.
            if not _armed():
                return (0, -1)
            rt = sched.phase_flip_runtime
            snap = rt.parked_extent() if rt is not None else None
            if snap is None:
                return (-1, -1)
            return (int(snap[0]), int(snap[1]))

        with self.subTest(state="armed with snapshot -> exact"):
            sched.phase_flip_runtime = _runtime(
                pending=pfr.PP_TO_TP, snapshot=(PARKED_ROWS, PARKED_TOP)
            )
            self.assertEqual(_pending(), (PARKED_ROWS, PARKED_TOP))
        with self.subTest(state="armed without snapshot -> UNKNOWN"):
            sched.phase_flip_runtime = _runtime(pending=pfr.PP_TO_TP, snapshot=None)
            self.assertEqual(_pending(), (-1, -1))
        with self.subTest(state="disarmed stale snapshot -> inert"):
            sched.phase_flip_runtime = _runtime(
                pending=None, snapshot=(999, 999)
            )
            self.assertEqual(_pending(), (0, -1))


class TestTheExclusionCeilingStands(CustomTestCase):
    """The DO-NOT: #746 is no licence to drop #748's semantics. The rung's
    ``_parked_ceiling`` still refuses wholesale (-2) on UNKNOWN-while-armed
    and still pins the ceiling on a known extent -- fed by the snapshot now,
    which this test wires through a real runtime end to end."""

    def _rung(self, runtime):
        from sglang.srt.managers import kv_backing_relief as m

        r = m.KvBackingRelief.__new__(m.KvBackingRelief)

        def _armed():
            return runtime._pending is not None

        def _pending():
            if not _armed():
                return (0, -1)
            snap = runtime.parked_extent()
            if snap is None:
                return (-1, -1)
            return (int(snap[0]), int(snap[1]))

        r._flip_armed_fn = _armed
        r._flip_pending_fn = _pending
        return r

    def test_snapshot_feeds_the_ceiling(self):
        from sglang.srt.managers import phase_flip_runtime as pfr

        rt = _runtime(pending=pfr.PP_TO_TP, snapshot=(PARKED_ROWS, PARKED_TOP))
        self.assertEqual(self._rung(rt)._parked_ceiling(), PARKED_TOP)

    def test_unknown_snapshot_still_refuses_wholesale(self):
        from sglang.srt.managers import phase_flip_runtime as pfr

        rt = _runtime(pending=pfr.PP_TO_TP, snapshot=None)
        self.assertEqual(self._rung(rt)._parked_ceiling(), -2)

    def test_after_the_flip_the_rung_is_fully_live(self):
        """test_THE_RUNG_IS_NOT_DEAD_OUTSIDE_FLIPS, restated through the
        snapshot lifecycle: after every exit the ceiling reads 'nothing
        parked'."""
        rt = _runtime(pending=None, snapshot=None)
        self.assertEqual(self._rung(rt)._parked_ceiling(), -1)


if __name__ == "__main__":
    unittest.main()
