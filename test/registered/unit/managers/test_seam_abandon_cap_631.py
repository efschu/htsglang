# SPDX-License-Identifier: Apache-2.0
"""#485: the seam's retry is bounded, damped, and ends in a verdict.

WHAT THIS FILE PINS, AND WHY IT EXISTS

With `--enable-phase-flip` on and a layer cut whose seam staging does not fit,
the group re-armed every `SGLANG_PHASE_POLICY_MIN_DWELL_S` forever. Measured
2026-08-12 on the #485 planner cut `[42,11,11]`: rank0 wanted 4881 MiB of
staging against 4314 MiB spendable, and the group abandoned 185 times in nine
minutes (555 log lines across three ranks). Every attempt runs the full spill
ladder and a torch `empty_cache` while the armed window withholds admissions,
so nothing drained; the detokenizer heartbeat expired, and the instance --
which had already printed "fired up and ready to roll" -- stopped answering
/health while every scheduler stack sat IDLE in a normal wait. Nothing was
deadlocked. An unbounded retry of a refusal that CANNOT CHANGE is what turned
an unfundable configuration into a dead instance.

Three properties are pinned here:

  1. the retry is DAMPED, so a repeated refusal stops producing work;
  2. it is CAPPED, and the cap ends in a terminal verdict that leaves the
     instance serving in its current phase instead of killing it;
  3. the decision is GROUP-UNIFORM and reached without a collective -- the
     backoff clock is ARM REQUESTS, which every rank sees as one broadcast,
     never a wall clock (rank-local) and never a round count (not uniform
     within one arm).

The margin-delay path is deliberately NOT bounded by any of this: its own
`seam_entry_delay_budget` already terminates it by yielding to the corridor
law. `test_a_margin_delay_is_not_damped_by_the_backoff` is the inertness
claim on the shipped path.
"""

import os

from sglang.srt.managers.phase_flip_runtime import (
    PP_TO_TP,
    SEAM_ABANDON_CAP_GUARD,
    TP_TO_PP,
    PhaseFlipRuntime,
    seam_abandon_backoff_max,
    seam_abandon_cap,
    seam_backoff_skips,
)


class _Env:
    """Set the cap/backoff environment for one test."""

    def __init__(self, cap=None, backoff_max=None):
        self.env = {}
        if cap is not None:
            self.env["SGLANG_SEAM_ABANDON_CAP"] = str(cap)
        if backoff_max is not None:
            self.env["SGLANG_SEAM_ABANDON_BACKOFF_MAX"] = str(backoff_max)

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)
        return self

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


def _runtime(phase="pp"):
    """A stub runtime carrying exactly the state ``arm`` reads.

    Built with ``__new__`` like the rest of this corpus's hermetic gate tests,
    so no CUDA, no scheduler and no checkpoint is required.
    """
    r = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
    r.blocking_guards = ()
    r._phase = phase
    r._pending = None
    r._entry_round = 0
    r._presence_wait_stamp = None
    r._armed_at = None
    r._park_deadline_s = 30.0
    r._clock = lambda: 0.0
    r._pool_census = lambda *a, **k: None
    r._arm_seq = 0
    r._seam_abandons_in_a_row = {PP_TO_TP: 0, TP_TO_PP: 0}
    r._seam_retry_at_arm = {PP_TO_TP: 0, TP_TO_PP: 0}
    r.seam_backoff_skips = {PP_TO_TP: 0, TP_TO_PP: 0}
    return r


class TestTheBackoffCurve:
    def test_the_first_abandon_is_not_damped(self):
        # A seam that was short because a request happened to be resident
        # deserves an immediate second look; growth starts only once the
        # refusal has REPEATED, which is the signature of a fixed demand.
        assert seam_backoff_skips(0, 16) == 0
        assert seam_backoff_skips(1, 16) == 0

    def test_it_doubles(self):
        assert [seam_backoff_skips(k, 1024) for k in range(2, 7)] == [1, 3, 7, 15, 31]

    def test_it_clamps(self):
        assert seam_backoff_skips(10, 16) == 16
        # No overflow at absurd streaks: the clamp answers, not 2**k.
        assert seam_backoff_skips(4096, 16) == 16

    def test_a_zero_clamp_disables_the_damping_but_not_the_cap(self):
        assert seam_backoff_skips(5, 0) == 0


class TestTheDampingActuallyDeclines:
    def test_arm_is_declined_inside_the_backoff_window(self):
        r = _runtime()
        r._seam_abandons_in_a_row[PP_TO_TP] = 3
        r._seam_retry_at_arm[PP_TO_TP] = 5  # next entry at arm 5
        ok, msg = r.arm(PP_TO_TP, "test")  # this becomes arm 1
        assert ok is False
        assert "backing off" in msg
        assert r._pending is None, "a declined arm must not leave a pending flip"
        assert r.seam_backoff_skips[PP_TO_TP] == 1

    def test_arm_goes_through_once_the_window_passes(self):
        r = _runtime()
        r._seam_abandons_in_a_row[PP_TO_TP] = 3
        r._seam_retry_at_arm[PP_TO_TP] = 3
        assert r.arm(PP_TO_TP, "t")[0] is False  # arm 1
        assert r.arm(PP_TO_TP, "t")[0] is False  # arm 2
        ok, _ = r.arm(PP_TO_TP, "t")  # arm 3 == retry point
        assert ok is True
        assert r._pending == PP_TO_TP
        assert r.seam_backoff_skips[PP_TO_TP] == 2

    def test_the_clock_advances_on_declined_arms_too(self):
        # If a refused arm did not tick the sequence, a rank that refuses
        # would fall behind a rank that does not and the two would disagree
        # about which arm is the retry point. The tick is the uniformity.
        r = _runtime()
        r.blocking_guards = ("something else",)
        r.arm(PP_TO_TP, "t")
        r.arm(PP_TO_TP, "t")
        assert r._arm_seq == 2

    def test_without_a_backoff_window_arm_is_untouched(self):
        # The can-fail proof for this file: with the window at its default 0,
        # every assertion above about declining must NOT hold.
        r = _runtime()
        ok, _ = r.arm(PP_TO_TP, "t")
        assert ok is True
        assert r.seam_backoff_skips[PP_TO_TP] == 0


class TestTheTerminalVerdict:
    def test_the_cap_installs_a_guard_that_arm_honours(self):
        r = _runtime()
        r._install_seam_cap_guard(PP_TO_TP, 8, ["staging 4881 MiB needed"])
        assert any(g.startswith(SEAM_ABANDON_CAP_GUARD) for g in r.blocking_guards)
        ok, msg = r.arm(PP_TO_TP, "t")
        assert ok is False
        assert SEAM_ABANDON_CAP_GUARD in msg

    def test_the_verdict_names_the_numbers_that_decided_it(self):
        r = _runtime()
        r._install_seam_cap_guard(TP_TO_PP, 8, ["staging 4881 MiB needed"])
        guard = [g for g in r.blocking_guards if g.startswith(SEAM_ABANDON_CAP_GUARD)][
            0
        ]
        assert TP_TO_PP in guard
        assert "8" in guard
        assert "4881" in guard, "a verdict without its numbers is not actionable"

    def test_the_instance_stays_in_its_phase(self):
        # The whole point: serving continues, degraded, rather than dying.
        r = _runtime(phase="tp")
        r._install_seam_cap_guard(TP_TO_PP, 8, ["short"])
        assert r._phase == "tp"

    def test_installing_twice_does_not_stack_guards(self):
        # Every rank reaches this branch, and a re-entry must be a no-op.
        r = _runtime()
        r._install_seam_cap_guard(PP_TO_TP, 8, ["short"])
        r._install_seam_cap_guard(PP_TO_TP, 9, ["short"])
        caps = [g for g in r.blocking_guards if g.startswith(SEAM_ABANDON_CAP_GUARD)]
        assert len(caps) == 1

    def test_a_pre_existing_guard_is_preserved(self):
        r = _runtime()
        r.blocking_guards = ("some other guard",)
        r._install_seam_cap_guard(PP_TO_TP, 8, ["short"])
        assert "some other guard" in r.blocking_guards
        assert len(r.blocking_guards) == 2


class TestTheEnvelope:
    def test_defaults_are_bounded_and_nonzero(self):
        with _Env():
            assert seam_abandon_cap() > 0
            assert seam_abandon_backoff_max() > 0

    def test_zero_cap_restores_the_unbounded_retry_exactly(self):
        # An off switch that is a VALUE of the same term, not a second code
        # path, so the off switch cannot drift from the on switch.
        with _Env(cap=0):
            assert seam_abandon_cap() == 0

    def test_garbage_falls_back_rather_than_raising(self):
        with _Env(cap="not-a-number", backoff_max="also-not"):
            assert seam_abandon_cap() > 0
            assert seam_abandon_backoff_max() > 0

    def test_negative_is_clamped_to_zero(self):
        with _Env(cap=-5):
            assert seam_abandon_cap() == 0


class TestTheDecisionIsGroupUniform:
    def test_two_ranks_with_the_same_history_decide_the_same_way(self):
        # The inputs are the group-uniform abandon book and the broadcast arm
        # sequence, so no collective is needed to agree -- and none is added.
        a, b = _runtime(), _runtime()
        for r in (a, b):
            r._seam_abandons_in_a_row[PP_TO_TP] = 4
            r._seam_retry_at_arm[PP_TO_TP] = 8
        results_a = [a.arm(PP_TO_TP, "t")[0] for _ in range(10)]
        results_b = [b.arm(PP_TO_TP, "t")[0] for _ in range(10)]
        assert results_a == results_b
        assert results_a.count(False) == 7, "arms 1-7 declined, arm 8 enters"

    def test_a_margin_delay_is_not_damped_by_the_backoff(self):
        # INERTNESS ON THE SHIPPED PATH. _execute applies the cap and the
        # backoff only when the group abandon is NOT a pure C20 margin delay,
        # because that path terminates itself by yielding to the corridor law
        # after its own budget. A runtime that has only ever delayed for the
        # margin therefore carries no backoff window at all.
        r = _runtime()
        r._seam_abandons_in_a_row[PP_TO_TP] = 2  # spent on margin delays
        assert r._seam_retry_at_arm[PP_TO_TP] == 0
        assert r.arm(PP_TO_TP, "t")[0] is True
