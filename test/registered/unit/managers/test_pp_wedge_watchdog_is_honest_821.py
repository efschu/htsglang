"""#821: the scheduler watchdog must be able to SEE a rank parked in a
blocking PP receive -- and must still not fire on a healthy idle one.

THE BLINDNESS. `create_scheduler_watchdog` (invariant_checker.py) fires when
`get_counter` (`scheduler.forward_ct`) stops advancing while `is_active()` is
true. Its activity predicate read exactly two things: `is_initializing`, and
`cur_batch_for_debug is not None`. A PP rank blocked in a mandatory dict
receive shows NEITHER -- `forward_ct` cannot advance because no forward is
running, and `cur_batch_for_debug` still holds the PREVIOUS pass's value,
which is None whenever that pass was idle. Every assignment to it on the PP
path (scheduler_pp_mixin.py:1567 / :1799 / :1950) happens AFTER the receive,
so no ordering exists in which the old predicate could have seen this state.
The watchdog therefore stayed inactive through the one failure it was best
placed to diagnose. That is why the #816 specimen ends with no watchdog line:
no SIGQUIT, no py-spy dump, no evidence -- just the launcher, 119.7 s later.

WHAT MAKES THIS FIX DANGEROUS IF DONE NAIVELY, and why half this file is
about the case that must NOT fire. A healthy PP rank is blocked in that
receive on nearly every pass; an IDLE server is blocked in it almost
continuously, because the ring is paced by those receives. So the obvious
form -- a boolean "am I in a receive" -- would make `is_active` permanently
true on a healthy idle boot, where `forward_ct` is frozen by definition, and
the watchdog would SIGQUIT a server whose only fault is having no work. The
marker is therefore a TIMESTAMP and the reader applies the watchdog's own
timeout to it. `TheIdleBootMustNotBeKilled` below is the arm that pins that,
and it matters more than the arm that pins the detection.

AND THE SECOND TRAP, which is why `dump_info` is tested here at all.
`dump_info` used to dereference `cur_batch_for_debug` unconditionally. That
was safe only while `is_active` could not be true with it None -- exactly the
combination the new arm makes the COMMON firing case. An AttributeError
raised there is not a spoiled dump: `WatchdogRaw._watchdog_thread` catches it,
logs "watchdog thread crashed" and RETURNS, so the thread dies, the SIGQUIT is
never sent, and the process loses its watchdog permanently. Arming the new arm
without guarding `dump_info` would have traded a silent wedge for a silent
wedge with no watchdog left. Tested as its own case, not assumed.

SCOPE, STATED. This does not un-wedge anything. It converts a silent,
evidence-free stall into the watchdog's normal loud death: a py-spy dump of
every scheduler plus SIGQUIT. That dump is precisely the instrument
COORD-strand16e-801-ring.md says the investigation lacks ("Ein py-spy auf die
Ueberlebenden waehrend des naechsten Vorfalls entscheidet es in einem Zug").
The wedge's root cause remains open; this makes the next occurrence
self-documenting instead of mute.
"""

import time
import types
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

WATCHDOG_TIMEOUT_S = 30.0


class _Pools:
    def get_pool_stats(self):
        return {}


class _InvariantChecker:
    def _check_all_pools(self, _stats):
        return True, ["pool line A", "pool line B"]


class _Batch:
    def batch_size(self):
        return 7

    @property
    def reqs(self):
        return ["req-a"]


def _scheduler(*, cur_batch=None, blocked_since=None, initializing=False):
    """The narrowest stand-in that `create_scheduler_watchdog` actually reads.

    Deliberately not a real Scheduler: this pins the PREDICATE, and a real
    scheduler would drag a model and a device in to prove nothing extra.
    """
    return types.SimpleNamespace(
        is_initializing=initializing,
        cur_batch_for_debug=cur_batch,
        forward_ct=0,
        invariant_checker=_InvariantChecker(),
        pool_stats_observer=_Pools(),
        _pp_blocked_recv_since=blocked_since,
    )


def _watchdog_for(scheduler):
    """Build the real watchdog WITHOUT letting it start its thread.

    `WatchdogRaw.__init__` starts a daemon thread that would SIGQUIT this test
    runner's parent on expiry, so the constructor is stubbed out and only the
    two callables under test are captured.
    """
    from sglang.srt.managers.scheduler_components import invariant_checker as ic

    captured = {}

    class _Fake:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    real = ic.WatchdogRaw
    ic.WatchdogRaw = _Fake
    try:
        ic.create_scheduler_watchdog(
            scheduler, watchdog_timeout=WATCHDOG_TIMEOUT_S, soft=True
        )
    finally:
        ic.WatchdogRaw = real
    return captured


class TheWedgedRankBecomesVisible(unittest.TestCase):
    """RED on the unchanged tree: `is_active` had no third arm."""

    def test_a_rank_overdue_in_a_pp_receive_reads_as_active(self):
        s = _scheduler(
            cur_batch=None,
            blocked_since=time.monotonic() - (WATCHDOG_TIMEOUT_S + 5.0),
        )
        self.assertTrue(
            _watchdog_for(s)["is_active"](),
            "a rank parked in a blocking PP receive for longer than the whole "
            "watchdog budget, with no cur_batch, is the #816 wedge -- the "
            "watchdog must consider it active or it can never fire",
        )

    def test_the_wedged_rank_is_exactly_the_shape_the_old_predicate_missed(self):
        """The two old inputs must BOTH read 'nothing to see' in that state,
        or this file is testing a state the old predicate already caught."""
        s = _scheduler(
            cur_batch=None,
            blocked_since=time.monotonic() - (WATCHDOG_TIMEOUT_S + 5.0),
        )
        self.assertFalse(s.is_initializing)
        self.assertIsNone(s.cur_batch_for_debug)


class TheIdleBootMustNotBeKilled(unittest.TestCase):
    """The arm that matters most: a healthy idle PP server sits in these
    receives almost continuously, with forward_ct frozen. If this ever goes
    red, the fix SIGQUITs healthy servers."""

    def test_a_brief_block_does_not_read_as_active(self):
        s = _scheduler(cur_batch=None, blocked_since=time.monotonic())
        self.assertFalse(
            _watchdog_for(s)["is_active"](),
            "an ordinary per-pass receive must never arm the watchdog",
        )

    def test_a_block_just_under_the_budget_does_not_read_as_active(self):
        s = _scheduler(
            cur_batch=None,
            blocked_since=time.monotonic() - (WATCHDOG_TIMEOUT_S - 2.0),
        )
        self.assertFalse(
            _watchdog_for(s)["is_active"](),
            "the threshold is the watchdog's own budget; just under it is not "
            "over it",
        )

    def test_a_rank_not_in_a_receive_at_all_does_not_read_as_active(self):
        self.assertFalse(_watchdog_for(_scheduler())["is_active"]())

    def test_a_scheduler_predating_the_marker_does_not_read_as_active(self):
        """The attribute is read with getattr(..., None): a holder or an older
        object that never sets it must not become permanently active."""
        s = _scheduler()
        del s._pp_blocked_recv_since
        self.assertFalse(_watchdog_for(s)["is_active"]())


class TheOldArmsStillWork(unittest.TestCase):
    def test_initializing_still_reads_active(self):
        self.assertTrue(_watchdog_for(_scheduler(initializing=True))["is_active"]())

    def test_a_running_batch_still_reads_active(self):
        self.assertTrue(_watchdog_for(_scheduler(cur_batch=_Batch()))["is_active"]())


class TheDumpMustSurviveTheNewFiringCase(unittest.TestCase):
    """RED on the unchanged tree: AttributeError on None.batch_size()."""

    def test_dump_info_does_not_raise_without_a_cur_batch(self):
        s = _scheduler(
            cur_batch=None,
            blocked_since=time.monotonic() - (WATCHDOG_TIMEOUT_S + 5.0),
        )
        info = _watchdog_for(s)["dump_info"]()
        # #824 W5(b) CHANGED THIS WORDING ON PURPOSE. It used to read
        # "parked in a blocking PP dict receive" unconditionally, because
        # _pp_recv_typed_dict was the only site that set the marker. On
        # boot_827 the two ranks that actually wedged were in the
        # request-relay chain receive, so this line named the wrong channel
        # in the one report an operator reads first. The dump now carries
        # the arm the marker recorded instead of asserting a channel.
        self.assertIn("parked in a blocking PP receive on arm=", info)

    def test_the_dump_names_how_long_the_receive_has_been_blocked(self):
        s = _scheduler(
            cur_batch=None,
            blocked_since=time.monotonic() - (WATCHDOG_TIMEOUT_S + 5.0),
        )
        info = _watchdog_for(s)["dump_info"]()
        self.assertIn("waited=", info)
        self.assertIn("pool line A", info, "the pool report must survive too")

    def test_dump_info_still_reports_a_real_batch(self):
        info = _watchdog_for(_scheduler(cur_batch=_Batch()))["dump_info"]()
        self.assertIn("batch_size()=7", info)
        self.assertIn("pool line B", info)

    def test_dump_info_is_still_empty_while_initializing(self):
        self.assertEqual(
            _watchdog_for(_scheduler(initializing=True))["dump_info"](), ""
        )


class TheMarkerIsActuallySetByTheReceive(unittest.TestCase):
    """A predicate reading an attribute nobody writes is the #182 shape. This
    drives the shipped `_pp_recv_typed_dict` and watches the marker."""

    def _holder(self, wire):
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        h = types.SimpleNamespace(pp_group=wire)
        h._pp_boundary_stats = lambda: None
        h._pp_flip_bump_consumed = lambda chan: None
        h._pp_recv_typed_dict = types.MethodType(
            SchedulerPPMixin._pp_recv_typed_dict, h
        )
        return h

    def test_the_marker_is_set_during_the_receive_and_cleared_after(self):
        seen = {}

        class _Wire:
            rank_in_group = 1
            world_size = 2

            def recv_tensor_dict(self, src=None, all_gather_group=None):
                seen["during"] = getattr(holder, "_pp_blocked_recv_since", None)
                return {"__msg_type__": "proxy"}

        holder = self._holder(_Wire())
        holder._pp_recv_typed_dict(expected_kind="proxy")
        self.assertIsNotNone(
            seen["during"], "the marker must be set while the receive blocks"
        )
        self.assertIsNone(
            holder._pp_blocked_recv_since,
            "the marker must be cleared once the receive returns",
        )

    def test_a_raising_receive_still_clears_the_marker(self):
        """A dead gloo peer makes this call RAISE (see
        test_pp_dead_peer_is_not_the_wedge_801.py). A stale timestamp left
        behind by that path would later read as a wedge that is not there."""

        class _Wire:
            rank_in_group = 1
            world_size = 2

            def recv_tensor_dict(self, src=None, all_gather_group=None):
                raise RuntimeError("Connection closed by peer")

        holder = self._holder(_Wire())
        with self.assertRaises(RuntimeError):
            holder._pp_recv_typed_dict(expected_kind="proxy")
        self.assertIsNone(holder._pp_blocked_recv_since)

    def test_the_marker_would_be_noticed_if_it_stopped_being_set(self):
        """Can-fail arm: the assertion above is only worth its line count if a
        receive that never set the marker would be caught."""
        holder = types.SimpleNamespace()
        self.assertIsNone(getattr(holder, "_pp_blocked_recv_since", None))


if __name__ == "__main__":
    unittest.main()
