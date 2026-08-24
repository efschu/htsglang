"""#850: the gate stops waiting for a consumer it is itself excluding.

THE PURE HALF is ``test_pp_presence_disposition_850``: which withhold reasons an
armed service turn can clear. THIS file is the WIRING, and the distinction is
the #699 lesson -- a classification that never reaches the gate would pass every
test in the pure file while changing nothing on metal. So the gate here is the
SHIPPED ``_await_group_presence``, bound to a duck-typed rank in the same shape
``test_pp_presence_withholding_deadlock_800`` uses, and the reasons are the real
strings ``pp_flip_channels_empty`` produces.

THREE DIRECTIONS, because a bound that only ever fires is not a guard:

1. A futile reason (``request-chain inbox holds ...``) abandons at the short
   bound instead of the 60 s presence deadline, counts itself, and says DEFECT.
2. A self-clearing reason (``send_output_work is not reaped`` -- 609 of the 823
   withholds measured across 291 boot logs) is NOT shortened. This is the
   mutant that matters: a shortening applied to every withhold would turn the
   commonest healthy transient into a refused flip.
3. ``SGLANG_PP_PRESENCE_FUTILE_S=0`` restores the full deadline for the futile
   reason too, so the guard is provable in both directions.

Hermetic: no scheduler, no process group, no CUDA.
"""

import unittest


class _Presence:
    """Minimal presence book: this rank alone, never reaching quorum.

    Quorum is what the futile rank cannot form -- the whole point is what
    happens while it does not -- so a single-rank book that reports a missing
    peer reproduces the gate's waiting state exactly.
    """

    def __init__(self):
        self.announced = set()
        self.withdrawn = set()

    def announce(self, epoch, note=None, round_=0):
        self.announced.add((epoch, round_))

    def observe(self, epoch, round_=0):
        return {0} if (epoch, round_) in self.announced else set()

    def quorum(self, epoch, round_=0):
        return False

    def may_withdraw(self, epoch, round_=0):
        return True

    def declare_withdrawn(self, epoch, round_=0):
        self.withdrawn.add((epoch, round_))

    def missing(self, epoch, round_=0):
        return [1]


class _Clock:
    """A hand-cranked monotonic clock, so a 60 s deadline costs no seconds."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _gate(reason, clock, futile_s=2.0, deadline=60.0):
    """The SHIPPED gate, wired to a probe that returns a real reason string."""
    from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime
    from sglang.srt.managers.phase_policy import PHASE_PP

    class R:
        pass

    r = R()
    r._presence = _Presence()
    r._pump_fn = None
    r._drain_fn = None
    r._owes_send_fn = lambda: False
    r._service_fn = None
    r._channels_empty_fn = lambda: reason
    r._flush_pending_sends_fn = None
    r.presence_withheld_rounds = 0
    r.presence_withheld_channels = 0
    r.entry_channel_violations = 0
    r._last_withhold_log = None
    r._last_not_ready_log = None
    r._log_not_ready = lambda: None
    r._entry_round = 0
    r._presence_wait_stamp = None
    r._presence_deadline_s = deadline
    r._presence_wait_started = None
    r._gate_open_epoch = None
    r._epoch = 1
    r._pending = "pp_to_tp"
    r._armed_at = 0.0
    r._last_hold_reason = None
    r._phase = PHASE_PP
    r.presence_timeouts = 0
    r._clock = clock
    r._sleep = lambda _s: None
    r._presence_poll_interval_s = 0.0
    r._last_presence_withhold_reason = None
    # #850 state, as `__init__` sets it on the shipped object.
    r.presence_futile_rounds = 0
    r.presence_futile_detected = 0
    r.presence_futile_abandons = 0
    r._presence_futile_since = None
    r._presence_futile_key = None
    r._presence_futile_alarmed = None
    r._presence_futile_s = futile_s
    r.abandoned = []
    r._abandon_no_quorum = lambda epoch, missing, waited: r.abandoned.append(waited)
    for name in ("_await_group_presence", "_commit_to_entering"):
        setattr(r, name, getattr(PhaseFlipRuntime, name).__get__(r, R))
    return r


#: The real clause, verbatim from `pp_flip_channels_empty`. Measured 11 times
#: across 291 boot logs under /spinning/evidence-665-f1.
FUTILE = "request-chain inbox holds 1 unhandled message(s)"

#: The commonest healthy withhold: 609 of 823.
SELF_CLEARING = "send_output_work is not reaped"


def _run(gate, clock, until, step=0.5):
    """Crank the gate until it abandons or `until` seconds have passed."""
    while clock.t <= until:
        gate._await_group_presence()
        if gate.abandoned:
            return clock.t
        clock.t += step
    return None


class TestFutileWaitIsCutShort(unittest.TestCase):
    def test_futile_reason_abandons_at_the_short_bound(self):
        clock = _Clock()
        gate = _gate(FUTILE, clock, futile_s=2.0, deadline=60.0)

        at = _run(gate, clock, until=30.0)

        self.assertIsNotNone(at, "the futile withhold never abandoned")
        self.assertLess(
            at,
            10.0,
            f"abandoned at {at}s -- the short bound did not reach the gate",
        )
        self.assertGreater(gate.presence_futile_rounds, 0)
        self.assertEqual(gate.presence_futile_detected, 1)
        self.assertEqual(gate.presence_futile_abandons, 1)

    def test_self_clearing_reason_is_not_shortened(self):
        """THE MUTANT THAT MATTERS. Shortening every withhold would refuse
        the commonest healthy flip: an unreaped send IS reaped by the service
        turn a round or two later."""
        clock = _Clock()
        gate = _gate(SELF_CLEARING, clock, futile_s=2.0, deadline=60.0)

        at = _run(gate, clock, until=30.0)

        self.assertIsNone(
            at, f"a self-clearing withhold was abandoned early at {at}s"
        )
        self.assertEqual(gate.presence_futile_rounds, 0)
        self.assertEqual(gate.presence_futile_abandons, 0)

    def test_the_bound_is_disableable_but_detection_continues(self):
        """The off-switch, and the counter split it forced.

        SGLANG_PP_PRESENCE_FUTILE_S=0 disables the ACTUATOR while leaving the
        detector intact -- the same bargain #800's escape clock strikes. So the
        defect is still counted and still named, and nothing is abandoned
        early. The first version of this fix incremented one counter in the
        detection path and called it `abandons`, which reported an action it
        had not taken; this assertion is what caught that.
        """
        clock = _Clock()
        gate = _gate(FUTILE, clock, futile_s=0.0, deadline=60.0)

        at = _run(gate, clock, until=30.0)

        self.assertIsNone(at, f"disabled bound still abandoned early at {at}s")
        self.assertEqual(gate.presence_futile_abandons, 0)
        self.assertEqual(gate.presence_futile_detected, 1)
        self.assertGreater(gate.presence_futile_rounds, 0)

    def test_the_full_deadline_still_abandons(self):
        """Disabling the shortening must not disable ABANDONING."""
        clock = _Clock()
        gate = _gate(FUTILE, clock, futile_s=0.0, deadline=10.0)

        at = _run(gate, clock, until=30.0)

        self.assertIsNotNone(at)
        self.assertGreaterEqual(at, 10.0)

    def test_mixed_reason_keeps_waiting(self):
        """One clause a service turn can still fix suppresses the verdict."""
        clock = _Clock()
        gate = _gate(f"{FUTILE}; {SELF_CLEARING}", clock, futile_s=2.0)

        at = _run(gate, clock, until=30.0)

        self.assertIsNone(at, f"a mixed withhold was abandoned early at {at}s")
        self.assertEqual(gate.presence_futile_rounds, 0)


class TestTheClockIsPerEpisode(unittest.TestCase):
    """A later arm must not inherit an earlier arm's futility age.

    THE BUG THIS CAUGHT, found by re-reading #800's own rule rather than by a
    failure: a futile withhold that ends in an abandon leaves the clock set.
    The NEXT arm then measures its bound from the PREVIOUS episode and expires
    in its first round -- before the service turn has had a single chance to
    clear anything. #800 states the rule for its own escape clock: reset when
    the key empties, "rather than inheriting a stranger's age".
    """

    def test_a_second_arm_gets_its_own_full_bound(self):
        clock = _Clock()
        gate = _gate(FUTILE, clock, futile_s=2.0, deadline=60.0)

        first = _run(gate, clock, until=30.0)
        self.assertIsNotNone(first)

        # A new arm: new epoch, and the wait restarts. The stale clock would
        # make this abandon immediately instead of after its own bound.
        gate.abandoned.clear()
        gate._epoch = 2
        gate._entry_round = 0
        gate._presence_wait_started = None
        started = clock.t
        second = _run(gate, clock, until=started + 30.0)

        self.assertIsNotNone(second, "the second arm never abandoned")
        self.assertGreaterEqual(
            second - started,
            2.0,
            "the second arm inherited the first arm's futility clock and "
            "expired before its own bound",
        )


class TestTheShippedConstructorSetsTheState(unittest.TestCase):
    """The REAL ``__init__``, because every other test here builds a stub.

    THE BUG THIS EXISTS FOR, found by ruff and not by any test: the first
    version of the knob read a module-global ``envs`` that this file does not
    import (every other ``envs`` read in it is a local import inside the
    function). That is a ``NameError`` on EVERY ``PhaseFlipRuntime``
    construction -- serving would not have booted at all -- and all 19 unit
    tests stayed green, because a duck-typed stub never runs ``__init__``.
    A stubbed gate cannot vouch for the shipped object; this can.
    """

    def test_init_sets_the_futile_state(self):
        from types import SimpleNamespace

        from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

        noop = lambda *a, **k: None  # noqa: E731
        runtime = PhaseFlipRuntime(
            n_ranks=3,
            rank=1,
            layer_map=[list(range(0, 12)), list(range(12, 22)), list(range(22, 32))],
            n_layers=32,
            tp_vector=[32, 16, 16],
            collective_min=lambda v: v,
            exchange=noop,
            pp_pool_view=SimpleNamespace(num_layers=10),
            tp_pool_view=SimpleNamespace(num_layers=32),
            live_slots_fn=lambda: [],
            ready_fn=lambda: True,
            cutover_fn=noop,
        )

        self.assertIsInstance(runtime._presence_futile_s, float)
        self.assertEqual(runtime.presence_futile_rounds, 0)
        self.assertEqual(runtime.presence_futile_detected, 0)
        self.assertEqual(runtime.presence_futile_abandons, 0)
        self.assertIsNone(runtime._presence_futile_since)
        self.assertIsNone(runtime._presence_futile_alarmed)


class TestTheGrepKeyIsStable(unittest.TestCase):
    """The alarm is a BOOT-ACCEPTANCE CRITERION, so its literal is pinned here.

    Same contract #838 states for ``ALARM_CONFORMANCE``: monitoring greps for
    this exact text, so a reword that looks harmless in review silently
    switches the detector off everywhere outside the python tree. Pinning it in
    a test is what makes the reword fail loudly instead.
    """

    def test_the_emitted_line_carries_the_literal(self):
        from sglang.srt.managers.pp_presence_disposition import (
            ALARM_PRESENCE_FUTILE,
        )

        self.assertEqual(ALARM_PRESENCE_FUTILE, "DEFECT PRESENCE-WAIT-FUTILE (#850)")

        clock = _Clock()
        gate = _gate(FUTILE, clock, futile_s=2.0)
        with self.assertLogs(
            "sglang.srt.managers.phase_flip_runtime", level="ERROR"
        ) as caught:
            _run(gate, clock, until=30.0)

        emitted = "\n".join(caught.output)
        self.assertIn(ALARM_PRESENCE_FUTILE, emitted)
        # The line must also NAME the real consumer -- a defect line that says
        # only "something is stuck" is the silence #800 paid five minutes for.
        self.assertIn("_pull_raw_reqs", emitted)


if __name__ == "__main__":
    unittest.main()
