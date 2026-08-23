# SPDX-License-Identifier: Apache-2.0
"""#834: the seam shrink -- quiesce before arming, and grow outside the window.

TWO MOVES, ONE FLAG, AND THE FLAG IS OFF BY DEFAULT. Both halves change WHEN
work happens relative to a collective, and the dangerous failure modes of the
second are invisible to a hermetic suite (a three-rank abort inside
``store_kvcache``'s bounds assert; #814's permanent pool shrink). So the default
path stays the measured, booted one and this file proves what a desk CAN prove:
that the moves do what they claim, that they are gated, and that the two
directions in which they can go wrong are REFUSED rather than hung.

WHAT ANALYSE_830 F1 ASKED FOR, verbatim, because half of this file is its
answer:

    "F1 -- Get the HiCache stream drain out of the seam's critical path.
    phase_flip_runtime.py:8150-8152 / :3871-3898. The drain waits for copies its
    own comment says 'outlive their Python call by seconds under load', while
    requests are parked and the ring is being rebuilt. Candidate shapes: quiesce
    BEFORE arming (so the wait happens while the pipeline still runs), or refuse
    to arm while device-tier I/O is in flight instead of arming and then
    waiting."

Both candidates are built, because they are one measurement read twice: the
pre-arm quiesce measures the drain, and the refusal is what happens when that
measurement comes back over #830 F4's budget.

THE DRAIN IS NOT REMOVED, and that is pinned here as well as in
test_seam_window_830.py. It exists because of two live SIGSEGVs on 2026-08-19,
each three seconds after a cutover, a HiCache copy reaching pool memory the seam
had released. Under ``layer_first`` the stale binding is SHAPE-IDENTICAL to the
live one, so no Python-side predicate can catch it. The seam's quiesce stays
exactly where it is; what changes is that by the time it runs there is nothing
left for it to wait for.

WHAT THE TIMING MODEL IS AND IS NOT. It is a hermetic model of the ORDERING
claim: device-tier work accumulates while the tier is armed, a quiesce drains
what has accumulated, and the question is which side of the no-return point that
draining lands on. It is NOT a latency measurement and does not pretend to be
one -- no wall clock is asserted anywhere in this file. The millisecond figures
are a token the model passes around so that "the drain happened HERE and not
THERE" is a countable statement instead of a narrative one.
"""

import inspect
import unittest
import unittest.mock
from types import SimpleNamespace

from sglang.srt.layers.dcp.phase_flip_plan import PP_TO_TP, TP_TO_PP
from sglang.srt.managers import phase_flip_runtime as _rt
from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

#: How much device-tier work one round of serving hands to the controller. The
#: value is arbitrary and the model only ever compares sums of it, so nothing
#: here depends on it being realistic -- only on it being non-zero, which is
#: what makes "the drain landed in the seam" distinguishable from "there was no
#: drain to land anywhere".
ROUND_MS = 250.0


class _FakeController:
    """The HiCache controller's device tier, reduced to what the seam sees.

    Two behaviours matter and both are real:

      * ``quiesce_device_io`` waits for what is ALREADY enqueued and returns
        how long that took. It does NOT stop new work arriving -- the real one
        is ``stream.synchronize()`` per stream, which has no opinion about
        anything enqueued after it. That is why the guard, not the drain, is
        what closes the race, and why the order guard-then-drain is load
        bearing rather than stylistic.
      * work only arrives while the tier is ARMED. In production that is the
        #718/#760 phase guard refusing device-tier I/O whenever the runtime
        reports a seam; here it is the same predicate read from the same
        attribute.
    """

    def __init__(self, armed_predicate):
        self._armed = armed_predicate
        self.inflight_ms = 0.0
        self.drains = []
        self.writes_accepted = 0
        self.writes_refused = 0

    def serve_one_round(self) -> bool:
        """One round of ordinary serving, which wants to write to the tier."""
        if not self._armed():
            self.writes_refused += 1
            return False
        self.writes_accepted += 1
        self.inflight_ms += ROUND_MS
        return True

    def quiesce_device_io(self, reason: str) -> float:
        drained = self.inflight_ms
        self.inflight_ms = 0.0
        self.drains.append((reason, drained))
        return drained / 1000.0


def _runtime(controller=None, phase=None):
    """A PhaseFlipRuntime with exactly the state ``arm`` touches.

    ``__new__`` rather than a constructed instance, following
    test_seam_window_830.py's recorders: ``__init__`` needs a live group, and
    every method under test here is deliberately one that does not.
    """
    rt = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
    rt._arm_seq = 0
    rt.blocking_guards = ()
    rt._pending = None
    rt._phase = phase if phase is not None else _rt.PHASE_PP
    rt._round = 0
    rt._entry_round = 0
    rt._presence_wait_stamp = None
    rt._armed_at = None
    rt._park_deadline_s = 30.0
    rt.hicache_seam_active = False
    rt._prearm_hold_direction = None
    rt._prearm_drain_ms = None
    rt._prearm_drain_defers = 0
    rt._seam_drain_ms = None
    rt._deferred_grow_pending = False
    rt._deferred_grow_rows = 0
    rt._deferred_grow_round = None
    rt._deferred_grow_level = None
    rt._clock = lambda: 0.0
    rt._prearm_floor_relief = lambda direction: None
    rt._snapshot_parked_extent = lambda: None
    rt._pool_census = lambda when, direction: None
    rt._arming_condition_persists = lambda: False
    tree = SimpleNamespace(cache_controller=controller)
    rt._census_scheduler = SimpleNamespace(tree_cache=tree)
    return rt


def _round_hook_finally(rt) -> None:
    """The two lines ``on_round``'s ``finally`` runs, and nothing else.

    REPLICATED, NOT CALLED, and the reason is stated rather than glossed:
    ``on_round`` needs a live consensus channel to reach its own ``finally``,
    which is exactly the thing a hermetic test cannot have. So the ordering
    claim is proved here against a replica, and
    ``test_can_fail_the_round_hook_finally_honours_the_hold`` pins the replica
    to the real source. Neither half is worth anything without the other: the
    model without the source pin proves a fiction, and the source pin without
    the model proves a string.
    """
    if not rt._prearm_quiesce_held():
        rt.hicache_seam_active = False


def _drive_a_flip(rt, ctrl, direction, rounds_armed=4, rounds_before=3):
    """Serve, arm, wait for the group, then enter the seam. Returns the seam's
    own drain in ms -- the number that is inside the no-return window."""
    for _ in range(rounds_before):
        ctrl.serve_one_round()
    ok, msg = rt.arm(direction, "test")
    if not ok:
        return None, msg
    for _ in range(rounds_armed):
        # The armed window: the group is not yet quiescent, so ``on_round``
        # returns without flipping and its ``finally`` runs anyway. This is the
        # great majority of rounds in any real boot and is where an
        # unconditional clear would drop the pre-arm guard.
        ctrl.serve_one_round()
        _round_hook_finally(rt)
    # THE NO-RETURN POINT, in the order ``_execute`` uses it.
    rt.hicache_seam_active = True
    seam_ms = rt._quiesce_hicache(direction)
    return seam_ms, msg


class TestPrearmQuiesceMovesTheDrainOutOfTheSeam(CustomTestCase):
    """#834 A / ANALYSE_830 F1, as a model of where the wait happens."""

    def test_red_first_today_the_drain_is_paid_inside_the_seam(self):
        """THE STATE THE FIX IS AGAINST, asserted rather than described.

        With the gate off, every millisecond of device-tier work accumulated
        since the last flip is waited for at the no-return point, with the
        requests parked. This is not a claim about the current tree's absolute
        cost -- #830 F1 measured the drain at 0.0 ms in boot_window2 and said
        so -- it is a claim about WHERE the cost lands when there is one."""
        rt = _runtime()
        ctrl = _FakeController(lambda: not rt.hicache_seam_active)
        rt._census_scheduler.tree_cache.cache_controller = ctrl

        seam_ms, _ = _drive_a_flip(rt, ctrl, PP_TO_TP)

        self.assertGreater(
            seam_ms,
            0.0,
            "with the shrink off the seam must still pay the drain; a zero "
            "here means the model never enqueued anything and proves nothing",
        )
        self.assertEqual(
            [], [d for d in ctrl.drains if "prearm" in d[0]],
            "the gate is off, so nothing may drain before the arm",
        )
        # Everything the tier accepted was waited for in the window.
        self.assertAlmostEqual(ctrl.writes_accepted * ROUND_MS, seam_ms, places=3)

    def test_green_with_the_gate_on_the_seam_finds_nothing_left_to_drain(self):
        """THE FIX. The same work, the same rounds, drained on the other side
        of the arm -- while the pipeline is still serving."""
        rt = _runtime()
        ctrl = _FakeController(lambda: not rt.hicache_seam_active)
        rt._census_scheduler.tree_cache.cache_controller = ctrl

        with unittest.mock.patch.object(
            _rt, "seam_shrink_prearm_quiesce_enabled", lambda: True
        ):
            seam_ms, _ = _drive_a_flip(rt, ctrl, PP_TO_TP)

        prearm = [d for d in ctrl.drains if "prearm" in d[0]]
        self.assertEqual(1, len(prearm), "the arm must drain exactly once")
        self.assertGreater(
            prearm[0][1], 0.0, "the pre-arm drain must have had work to do"
        )
        self.assertEqual(
            0.0,
            seam_ms,
            "the seam's own quiesce must find an empty tier; it is now a "
            "confirmation, not a wait",
        )

    def test_green_the_correctness_pin_no_write_slips_through_the_armed_window(self):
        """THE PIN THAT MATTERS MOST, and the one a latency change is most
        likely to break: nothing may be enqueued between the quiesce's
        confirmation and the cutover.

        The seam's drain closes a race whose two halves are 'no NEW I/O' and
        'no OLD I/O in flight'. Pulling the drain forward is only sound if the
        guard is pulled forward WITH it and stays up -- including across every
        ``on_round`` that returns without flipping, of which a real armed
        window has many."""
        rt = _runtime()
        ctrl = _FakeController(lambda: not rt.hicache_seam_active)
        rt._census_scheduler.tree_cache.cache_controller = ctrl

        with unittest.mock.patch.object(
            _rt, "seam_shrink_prearm_quiesce_enabled", lambda: True
        ):
            for _ in range(3):
                ctrl.serve_one_round()
            accepted_before_arm = ctrl.writes_accepted
            rt.arm(PP_TO_TP, "test")
            for _ in range(6):
                ctrl.serve_one_round()
                _round_hook_finally(rt)
            self.assertEqual(
                accepted_before_arm,
                ctrl.writes_accepted,
                "a device-tier write was accepted during the armed window; "
                "the drain taken at arm no longer covers the seam",
            )
            self.assertEqual(6, ctrl.writes_refused)
            self.assertTrue(
                rt.hicache_seam_active,
                "the guard must survive on_round's insurance clear while the "
                "flip is pending",
            )

    def test_can_fail_the_hold_is_released_when_nothing_is_pending(self):
        """THE OTHER DIRECTION, and it is the #742 inert-state class: a
        capability silently dead for the life of the process. A hold that
        outlives its flip disables the device tier for ever, which would be a
        worse bug than the latency it was fixing."""
        rt = _runtime()
        ctrl = _FakeController(lambda: not rt.hicache_seam_active)
        rt._census_scheduler.tree_cache.cache_controller = ctrl

        with unittest.mock.patch.object(
            _rt, "seam_shrink_prearm_quiesce_enabled", lambda: True
        ):
            rt.arm(PP_TO_TP, "test")
            self.assertTrue(rt.hicache_seam_active)
            # The flip resolves -- committed, abandoned, it makes no difference
            # to this: nothing is pending any more.
            rt._pending = None
            rt._release_prearm_quiesce("test")
            _round_hook_finally(rt)

        self.assertFalse(rt.hicache_seam_active)
        self.assertTrue(ctrl.serve_one_round(), "the device tier must come back")

    def test_can_fail_an_over_budget_drain_refuses_the_arm_by_name(self):
        """ANALYSE_830 F1's second candidate: "refuse to arm while device-tier
        I/O is in flight instead of arming and then waiting".

        AND THE REFUSAL MUST NOT LEAVE THE TIER DOWN. A refused arm leaves
        nothing pending, so nothing would ever clear a guard left up here."""
        rt = _runtime()
        ctrl = _FakeController(lambda: not rt.hicache_seam_active)
        rt._census_scheduler.tree_cache.cache_controller = ctrl
        # Far past the 1094 ms #830 F4 budget.
        ctrl.inflight_ms = 99_000.0

        with unittest.mock.patch.object(
            _rt, "seam_shrink_prearm_quiesce_enabled", lambda: True
        ):
            ok, msg = rt.arm(PP_TO_TP, "test")

        self.assertFalse(ok)
        self.assertIn(_rt.PREARM_DRAIN_REFUSED, msg)
        self.assertIsNone(rt._pending, "a refused arm must arm nothing")
        self.assertFalse(
            rt.hicache_seam_active,
            "a refused arm must re-arm the device tier; nothing is pending, so "
            "nothing would ever clear it",
        )

    def test_can_fail_the_measured_drain_feeds_the_830_budget(self):
        """The pre-arm drain narrows #830 F4's stated limitation instead of
        arguing it away. F4 projects from the LAST flip's drain because
        "nothing at arm time can enumerate" the backlog. With the drain pulled
        forward, arm time just did."""
        rt = _runtime()
        ctrl = _FakeController(lambda: not rt.hicache_seam_active)
        rt._census_scheduler.tree_cache.cache_controller = ctrl
        ctrl.inflight_ms = 400.0

        with unittest.mock.patch.object(
            _rt, "seam_shrink_prearm_quiesce_enabled", lambda: True
        ):
            rt.arm(PP_TO_TP, "test")

        self.assertAlmostEqual(400.0, rt._prearm_drain_ms, places=3)
        self.assertAlmostEqual(
            400.0,
            rt._seam_drain_ms,
            places=3,
            msg="the seam budget's projector must read THIS arm's measurement",
        )

    def test_can_fail_the_gate_off_path_touches_nothing(self):
        """DEFAULT-OFF MEANS BYTE-IDENTICAL. With the gate off, ``arm`` must
        not touch the seam guard, must not drain, and must not book a hold."""
        rt = _runtime()
        ctrl = _FakeController(lambda: not rt.hicache_seam_active)
        rt._census_scheduler.tree_cache.cache_controller = ctrl

        ok, _ = rt.arm(PP_TO_TP, "test")

        self.assertTrue(ok)
        self.assertFalse(rt.hicache_seam_active)
        self.assertEqual([], ctrl.drains)
        self.assertIsNone(rt._prearm_hold_direction)
        self.assertFalse(rt._prearm_quiesce_held())

    def test_can_fail_the_round_hook_finally_honours_the_hold(self):
        """SOURCE PIN for the replica in ``_round_hook_finally``.

        ``on_round``'s ``finally`` is #760's insurance and it clears the guard
        unconditionally today. If it stays unconditional, the pre-arm hold is
        dropped on the very next round and every behavioural test above is
        modelling code that does not exist."""
        src = inspect.getsource(PhaseFlipRuntime.on_round)
        # THE ASSERTION IS ON THE STATEMENT, NOT ON THE NAME, and mutant M2
        # is why. The first version of this pin searched for the bare token
        # ``_prearm_quiesce_held`` -- which also appears in the COMMENT above
        # the branch, so deleting the branch and keeping the comment passed
        # cleanly. An indicator that matches its own documentation instead of
        # its subject is the #830 F5 failure exactly, one file over.
        self.assertIn(
            "if not prearm_quiesce_held(self):",
            src,
            "on_round's insurance clear must be CONDITIONED on the pre-arm "
            "hold; unconditional, it drops the guard on the next round and "
            "the drain taken at arm covers nothing",
        )
        clear = "self.hicache_seam_active = False"
        self.assertIn(clear, src, "the insurance clear itself must survive")
        guard_at = src.index("if not prearm_quiesce_held(self):")
        self.assertLess(
            guard_at,
            src.index(clear, guard_at),
            "the guard must be READ BEFORE the clear it conditions",
        )

    def test_can_fail_the_seam_still_quiesces_at_the_no_return_point(self):
        """#830 M11, restated because this change is the exact temptation it
        was written against: the seam's drain now measures 0.0 ms with the gate
        on, which makes it look free to delete. It is not. Two SIGSEGVs."""
        src = inspect.getsource(PhaseFlipRuntime._execute)
        self.assertIn("_quiesce_hicache", src)
        self.assertLess(
            src.index("hicache_seam_active = True"),
            src.index("_quiesce_hicache"),
            "the seam guard must still be up before the seam's own drain",
        )


class TestDeferredGrowLeavesTheNoReturnWindow(CustomTestCase):
    """#834 B, the runtime half: booking, paying, and the two refusals."""

    def test_can_fail_the_gate_off_path_calls_the_shipped_function(self):
        """DEFAULT-OFF MEANS BYTE-IDENTICAL, for B as for A: one call to
        ``recover_kv_backing`` and no booking anywhere."""
        rt = _runtime()
        sched = SimpleNamespace(phase_flip_runtime=rt)
        with unittest.mock.patch(
            "sglang.srt.managers.phase_flip_spill.recover_kv_backing"
        ) as whole:
            with unittest.mock.patch(
                "sglang.srt.managers.phase_flip_spill.level_kv_backing_to_group"
            ) as half:
                _rt.seam_kv_recover(sched, lambda v: v, TP_TO_PP)
        self.assertEqual(1, whole.call_count)
        self.assertEqual(0, half.call_count)
        self.assertFalse(rt._deferred_grow_pending)

    def test_green_the_gate_on_path_levels_in_the_seam_and_books_the_grow(self):
        """THE SPLIT'S DIRECTION, which is the load-bearing judgement: the
        COLLECTIVE stays inside the no-return window, the RANK-LOCAL grow
        leaves. Reversing that is the 2026-08-08 boots 9/10 PP wedge."""
        rt = _runtime()
        sched = SimpleNamespace(phase_flip_runtime=rt)
        with unittest.mock.patch.object(
            _rt, "seam_shrink_defer_grow_enabled", lambda: True
        ):
            with unittest.mock.patch(
                "sglang.srt.managers.phase_flip_spill.recover_kv_backing"
            ) as whole:
                with unittest.mock.patch(
                    "sglang.srt.managers.phase_flip_spill.level_kv_backing_to_group",
                    return_value=2048,
                ) as half:
                    _rt.seam_kv_recover(sched, lambda v: v, TP_TO_PP)
        self.assertEqual(
            0,
            whole.call_count,
            "the whole recovery must not run in the seam once the grow is "
            "deferred, or the grow never left the window",
        )
        self.assertEqual(1, half.call_count, "the levelling must stay in the seam")
        self.assertTrue(rt._deferred_grow_pending)
        self.assertEqual(2048, rt._deferred_grow_level)

    def test_can_fail_no_runtime_means_no_deferral(self):
        """A grow deferred to nobody is #814's ratchet with extra steps: the
        memory is released and nothing ever grows it back."""
        sched = SimpleNamespace(phase_flip_runtime=None)
        with unittest.mock.patch.object(
            _rt, "seam_shrink_defer_grow_enabled", lambda: True
        ):
            with unittest.mock.patch(
                "sglang.srt.managers.phase_flip_spill.recover_kv_backing"
            ) as whole:
                _rt.seam_kv_recover(sched, lambda v: v, TP_TO_PP)
        self.assertEqual(1, whole.call_count)

    def test_green_the_paid_grow_clamps_to_the_agreed_level(self):
        """B step 2. The grow and the clamp are one operation with two calls,
        never separated by a return: ``recover()`` ends by raising exposure to
        this rank's OWN backing, which is the divergence if nothing follows."""
        rt = _runtime()
        rt._deferred_grow_pending = True
        rt._deferred_grow_level = 2048
        relief = SimpleNamespace(backed_rows=lambda: 4096, exposed_rows=lambda: 2048)
        rt._grow_relief = lambda: relief
        setattr(rt._census_scheduler, "kv_backing_relief", relief)

        order = []
        with unittest.mock.patch(
            "sglang.srt.managers.phase_flip_spill.grow_kv_backing_local",
            side_effect=lambda s: (order.append("grow"), 2048)[1],
        ):
            with unittest.mock.patch(
                "sglang.srt.managers.phase_flip_spill.clamp_kv_exposure_to_level",
                side_effect=lambda s, lv: (order.append(("clamp", lv)), 0)[1],
            ):
                rt._pay_deferred_grow()

        self.assertEqual(["grow", ("clamp", 2048)], order)
        self.assertFalse(rt._deferred_grow_pending)
        self.assertEqual(
            2048,
            rt._deferred_grow_rows,
            "4096 backed against an agreed level of 2048 is 2048 rows of debt",
        )

    def test_can_fail_a_grow_is_not_paid_while_a_flip_is_pending(self):
        """The seam prices its own affordability at the gate. A grow landing
        between that pricing and the seam spends memory the gate has already
        promised to the staging fund."""
        rt = _runtime()
        rt._deferred_grow_pending = True
        rt._deferred_grow_level = 2048
        rt._pending = TP_TO_PP
        with unittest.mock.patch(
            "sglang.srt.managers.phase_flip_spill.grow_kv_backing_local"
        ) as grow:
            rt._pay_deferred_grow()
        self.assertEqual(0, grow.call_count)
        self.assertTrue(rt._deferred_grow_pending, "the booking must survive")

    def test_can_fail_hazard_one_an_unlevelled_exposure_refuses_the_flip(self):
        """HAZARD DIRECTION 1, and the one the design note calls the reason the
        levelling cannot move: a rank levelling on a LOCAL cadence comes back
        with an id space its peers do not share. An id one rank exposes and a
        peer cannot map aborts ALL THREE inside ``store_kvcache``'s bounds
        assert.

        THE REQUIRED OUTCOME IS A NAMED REFUSAL, NOT A HANG. Blocking until the
        group agrees would be a collective entered at a local cadence, which is
        the wedge. Refusing costs a flip and no ranks."""
        rt = _runtime()
        rt._deferred_grow_level = 2048
        rt._deferred_grow_rows = 2048
        # The divergent state: this rank hands out ids above the agreed level.
        rt._grow_relief = lambda: SimpleNamespace(exposed_rows=lambda: 4096)

        with unittest.mock.patch.object(
            _rt, "seam_shrink_defer_grow_enabled", lambda: True
        ):
            refusal = rt._unlevelled_exposure_refusal()

        self.assertIsNotNone(refusal, "an unlevelled exposure must refuse")
        self.assertIn(_rt.UNLEVELLED_EXPOSURE_REFUSED, refusal)
        self.assertIn("4096", refusal)
        self.assertIn("2048", refusal)

    def test_can_fail_a_levelled_exposure_refuses_nothing(self):
        """The refusal must DEPEND on the divergence. A guard that refuses
        whichever way the numbers fall stops every flip and is worse than none:
        the flip is how this instance alternates prefill and decode."""
        rt = _runtime()
        rt._deferred_grow_level = 2048
        rt._deferred_grow_rows = 2048
        rt._grow_relief = lambda: SimpleNamespace(exposed_rows=lambda: 2048)
        with unittest.mock.patch.object(
            _rt, "seam_shrink_defer_grow_enabled", lambda: True
        ):
            self.assertIsNone(rt._unlevelled_exposure_refusal())

    def test_can_fail_an_unreadable_exposure_never_refuses(self):
        """#721's rule: a refusal is the thing with a service cost, so it must
        never rest on a number we do not have."""
        rt = _runtime()
        rt._deferred_grow_level = 2048
        rt._deferred_grow_rows = 2048

        def _blind():
            raise RuntimeError("no allocator")

        rt._grow_relief = lambda: SimpleNamespace(exposed_rows=_blind)
        with unittest.mock.patch.object(
            _rt, "seam_shrink_defer_grow_enabled", lambda: True
        ):
            self.assertIsNone(rt._unlevelled_exposure_refusal())

    def test_can_fail_hazard_two_an_unpaid_grow_debt_is_shouted_naming_814(self):
        """HAZARD DIRECTION 2: the levelling never comes, so rows stay backed
        and unexposed for ever. That is #814's ratchet -- pool at 26.8% of its
        id space for the life of the process, a user served overloaded_error
        against a pool sized 3.7x larger -- and the whole reason a deferral has
        to be a DEBT rather than a skip.

        AND IT IS AN ALARM, NEVER AN ACTUATOR. The tempting repair is to expose
        the rows once the wait gets embarrassing, which is hazard 1."""
        rt = _runtime()
        rt._deferred_grow_level = 2048
        rt._deferred_grow_rows = 2048
        rt._deferred_grow_round = 0
        rt._round = 0
        relief = SimpleNamespace(backed_rows=lambda: 4096)
        rt._grow_relief = lambda: relief

        with unittest.mock.patch.object(
            _rt, "seam_shrink_grow_debt_rounds", lambda: 8
        ):
            with unittest.mock.patch.object(_rt.logger, "error") as err:
                rt._round = 5
                rt._deferred_grow_debt_check()
                self.assertEqual(0, err.call_count, "not yet out of patience")
                rt._round = 9
                rt._deferred_grow_debt_check()

        self.assertEqual(1, err.call_count)
        line = err.call_args.args[0] % tuple(err.call_args.args[1:])
        self.assertIn(_rt.GROW_DEBT_UNPAID, line)
        self.assertIn("#814", line)
        # The debt must still be there: shouting is not paying.
        self.assertEqual(2048, rt._deferred_grow_rows)

    def test_can_fail_a_paid_debt_stops_shouting(self):
        """A latched alarm for a condition that has gone is how a real one gets
        ignored. The check RE-READS the allocator instead of trusting its own
        booking, so a later collective that levelled the group up clears it."""
        rt = _runtime()
        rt._deferred_grow_level = 2048
        rt._deferred_grow_rows = 2048
        rt._deferred_grow_round = 0
        rt._round = 99
        # A later collective raised the group's level to this rank's backing.
        rt._deferred_grow_level = 4096
        rt._grow_relief = lambda: SimpleNamespace(backed_rows=lambda: 4096)

        with unittest.mock.patch.object(
            _rt, "seam_shrink_grow_debt_rounds", lambda: 8
        ):
            with unittest.mock.patch.object(_rt.logger, "error") as err:
                rt._deferred_grow_debt_check()

        self.assertEqual(0, err.call_count)
        self.assertEqual(0, rt._deferred_grow_rows)
        self.assertIsNone(rt._deferred_grow_round)

    def test_can_fail_the_deferred_grow_is_wired_into_the_round_hook(self):
        """RED ARM against a booking nothing ever pays. A debt with no payer is
        #814 exactly, and it would be invisible: every seam would look faster
        and the pool would simply never come back."""
        src = inspect.getsource(PhaseFlipRuntime.on_round)
        self.assertIn("pay_deferred_grow(self)", src)

    def test_can_fail_the_refusal_is_wired_into_the_seam_gate(self):
        """RED ARM. A verdict nothing consults refuses nothing. It must vote
        through ``too_small``, which is what makes the whole GROUP decline
        together rather than this rank declining alone."""
        src = inspect.getsource(PhaseFlipRuntime._execute)
        self.assertIn("_unlevelled_exposure_refusal", src)
        block = src[src.index("_unlevelled_exposure_refusal") :]
        self.assertIn("too_small.append(refusal)", block)


class TestSharedPathsTolerateAHolderWithoutTheFeature(CustomTestCase):
    """THE REGRESSION THIS TICKET ACTUALLY CAUSED, pinned so it cannot return.

    The first cut of #834 called ``self._release_prearm_quiesce(...)`` directly
    from the five sites that clear ``_pending``. Three of those are abandon and
    disarm paths, and those paths are driven throughout ``unit/managers`` by
    minimal duck-typed stubs -- a bare object carrying the handful of attributes
    the path under test touches, because that is the only way to test an abandon
    without a live three-rank group. The result was 11 red tests across
    test_phase_policy.py and test_pp_presence_withholding_deadlock_800.py, every
    one of them a pin on "this rank abandons instead of wedging", all failing
    with AttributeError raised from INSIDE the abandon path.

    AN ABANDON PATH IS THE WORST PLACE IN THIS FILE TO ADD A HARD DEPENDENCY.
    It exists to survive trouble. #795 already established the convention -- the
    arm epoch is read through ``pp_flip_epoch_of`` so "a holder with no accessor
    keeps pre-#795 behaviour" -- and these three helpers are that rule applied
    to the seam shrink.
    """

    def test_green_a_bare_holder_survives_all_three_helpers(self):
        bare = SimpleNamespace()
        _rt.release_prearm_quiesce(bare, "no feature here")
        _rt.pay_deferred_grow(bare)
        self.assertFalse(_rt.prearm_quiesce_held(bare))

    def test_green_a_real_runtime_is_still_driven_by_them(self):
        """The tolerance must not become inertness: on a holder that DOES have
        the feature, each helper must reach it. A guard that no-ops on
        everything is the #742 inert-state class wearing a compatibility
        shim."""
        rt = _runtime()
        ctrl = _FakeController(lambda: not rt.hicache_seam_active)
        rt._census_scheduler.tree_cache.cache_controller = ctrl

        with unittest.mock.patch.object(
            _rt, "seam_shrink_prearm_quiesce_enabled", lambda: True
        ):
            rt.arm(PP_TO_TP, "test")
            self.assertTrue(_rt.prearm_quiesce_held(rt))
            rt._pending = None
            _rt.release_prearm_quiesce(rt, "test")
        self.assertFalse(rt.hicache_seam_active)
        self.assertIsNone(rt._prearm_hold_direction)

        rt._deferred_grow_pending = True
        rt._deferred_grow_level = 2048
        rt._grow_relief = lambda: SimpleNamespace(backed_rows=lambda: 2048)
        with unittest.mock.patch(
            "sglang.srt.managers.phase_flip_spill.grow_kv_backing_local",
            return_value=0,
        ):
            with unittest.mock.patch(
                "sglang.srt.managers.phase_flip_spill.clamp_kv_exposure_to_level",
                return_value=0,
            ):
                _rt.pay_deferred_grow(rt)
        self.assertFalse(rt._deferred_grow_pending, "the helper must reach it")

    def test_can_fail_a_raising_predicate_does_not_break_the_insurance(self):
        """``prearm_quiesce_held`` gates #760's insurance clear. If it can
        raise, a broken predicate takes out the clear that stops the device
        tier being disabled for the life of the process -- trading a small
        latency feature for the exact failure class it was built beside."""

        def _boom():
            raise RuntimeError("predicate is broken")

        holder = SimpleNamespace(_prearm_quiesce_held=_boom)
        self.assertFalse(_rt.prearm_quiesce_held(holder))


class TestTheGateIsOffByDefault(CustomTestCase):
    """Nothing in this ticket may change a boot that did not ask for it."""

    def test_green_every_half_is_off_by_default(self):
        self.assertFalse(_rt._seam_shrink_master())
        self.assertFalse(_rt.seam_shrink_prearm_quiesce_enabled())
        self.assertFalse(_rt.seam_shrink_defer_grow_enabled())

    def test_can_fail_a_half_override_can_cut_the_two_apart(self):
        """The overrides exist so a GPU window can attribute a result to ONE
        half. A window that moves both at once cannot say which moved the
        number, and this family has twice paid for reading adjacency as
        attribution."""
        import os

        env = "SGLANG_SEAM_SHRINK_PREARM_QUIESCE"
        old = os.environ.get(env)
        try:
            os.environ[env] = "1"
            self.assertTrue(_rt.seam_shrink_prearm_quiesce_enabled())
            self.assertFalse(
                _rt.seam_shrink_defer_grow_enabled(),
                "one half on must NOT turn the other on",
            )
            os.environ[env] = "0"
            self.assertFalse(_rt.seam_shrink_prearm_quiesce_enabled())
        finally:
            if old is None:
                os.environ.pop(env, None)
            else:
                os.environ[env] = old


register_cpu_ci(__file__)

if __name__ == "__main__":
    unittest.main()
