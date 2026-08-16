"""#699: liveness from progress, with the 16:23 specimen as the can-fail case.

The specimen: `/health` returned 200 for 52+ minutes while nothing advanced and
work was pending. Two blind spots produced it, and the tests below pin both.

  1. `/health` answers "is the process up", not "is work moving".
  2. The shipped watchdog arms only while a batch EXISTS
     (`invariant_checker.py:536-540`: `is_active = ... cur_batch_for_debug is
     not None`). An admission wedge is exactly the state where no batch exists
     while work is pending, so the timer never starts.

Hermetic: pure arithmetic, no CUDA, no server, no clock.
"""

import pytest
from sglang.srt.managers.progress_liveness import (
    ACTION_ALARM,
    ACTION_NONE,
    ACTION_RESTART,
    HEALTHY,
    IDLE,
    INHIBITED,
    UNKNOWN,
    WEDGED,
    LivenessPolicy,
    ProgressLivenessError,
    ProgressSample,
    assess,
    build_liveness_is_active,
)


def _trace(n=6, dt=10.0, **rates):
    """Build a trace where each named counter advances by its rate per sample."""
    out = []
    acc = {"completions": 0, "decode_steps": 0, "prefill_chunks": 0}
    for i in range(n):
        for k in acc:
            acc[k] += rates.get(k, 0)
        out.append(
            ProgressSample(
                t_s=i * dt,
                completions=acc["completions"],
                decode_steps=acc["decode_steps"],
                prefill_chunks=acc["prefill_chunks"],
                pending_requests=rates.get("pending_requests", 0),
                pending_tokens=rates.get("pending_tokens", 0),
            )
        )
    return out


def test_the_16_23_specimen_alarms():
    """CAN-FAIL PROOF against the real wedge shape.

    health 200 + zero progress + nonzero pending. Every progress counter is
    frozen while requests wait. This MUST alarm; a health-200 check calls it
    fine, which is the whole defect.
    """
    trace = _trace(
        n=6,
        completions=0,
        decode_steps=0,
        prefill_chunks=0,
        pending_requests=7,
        pending_tokens=180_000,
    )
    r = assess(trace, LivenessPolicy(confirmations=1))
    assert r.verdict == WEDGED
    assert r.action == ACTION_ALARM
    assert not r.progressing
    assert r.pending_requests == 7
    assert "health" in r.detail.lower()


def test_an_idle_box_is_not_a_wedge():
    """The refusal that keeps the alarm worth listening to."""
    trace = _trace(n=6, pending_requests=0, pending_tokens=0)
    r = assess(trace, LivenessPolicy(confirmations=1))
    assert r.verdict == IDLE
    assert r.action == ACTION_NONE
    # Same zero progress as the wedge: ONLY pending work separates them.
    assert not r.progressing


def test_pure_prefill_is_healthy_though_nothing_completes():
    """The trap a completions-only watchdog falls into.

    A 640-chunk prompt finishes nothing for minutes and runs no decode step.
    Only prefill_chunks moves, and that must be enough.
    """
    trace = _trace(n=6, prefill_chunks=40, pending_requests=1, pending_tokens=300_000)
    r = assess(trace, LivenessPolicy(confirmations=1))
    assert r.verdict == HEALTHY
    assert r.deltas["completions"] == 0
    assert r.deltas["decode_steps"] == 0
    assert r.deltas["prefill_chunks"] > 0


def test_pure_decode_is_healthy_though_no_chunk_is_admitted():
    trace = _trace(n=6, decode_steps=120, pending_requests=0, pending_tokens=0)
    r = assess(trace, LivenessPolicy(confirmations=1))
    assert r.verdict == HEALTHY
    assert r.deltas["prefill_chunks"] == 0


def test_completions_alone_also_count_as_progress():
    trace = _trace(n=6, completions=3, pending_requests=2, pending_tokens=500)
    assert assess(trace, LivenessPolicy(confirmations=1)).verdict == HEALTHY


def test_a_deliberate_pause_is_not_a_wedge():
    """A flip is 2-4.2 s of legitimate silence; maintenance holds are longer."""
    trace = _trace(n=6, pending_requests=4, pending_tokens=9000)
    paused = list(trace)
    paused[3] = ProgressSample(
        t_s=paused[3].t_s,
        completions=0,
        decode_steps=0,
        prefill_chunks=0,
        pending_requests=4,
        pending_tokens=9000,
        inhibited=True,
        inhibit_reason="phase flip in progress",
    )
    r = assess(paused, LivenessPolicy(confirmations=1))
    assert r.verdict == INHIBITED
    assert r.action == ACTION_NONE
    assert "flip" in r.detail


def test_a_cold_start_is_not_judged():
    r = assess(_trace(n=1, pending_requests=5), LivenessPolicy())
    assert r.verdict == UNKNOWN
    assert r.action == ACTION_NONE


def test_a_counter_reset_is_not_read_as_a_wedge():
    """After a restart the counters go back to zero.

    A negative delta means "restarted", not "stalled". Reading it as a wedge
    would make every restart trigger another restart.
    """
    trace = _trace(n=4, prefill_chunks=10, pending_requests=3, pending_tokens=100)
    trace.append(
        ProgressSample(
            t_s=trace[-1].t_s + 10.0,
            completions=0,
            decode_steps=0,
            prefill_chunks=0,
            pending_requests=3,
            pending_tokens=100,
        )
    )
    r = assess(trace, LivenessPolicy(confirmations=1))
    assert r.verdict == UNKNOWN
    assert r.action == ACTION_NONE
    assert "backwards" in r.detail


def test_confirmations_prevent_a_single_slow_window_from_alarming():
    trace = _trace(n=6, pending_requests=2, pending_tokens=50)
    first = assess(trace, LivenessPolicy(confirmations=2), consecutive_wedges=0)
    assert first.verdict == WEDGED and first.action == ACTION_NONE
    second = assess(trace, LivenessPolicy(confirmations=2), consecutive_wedges=1)
    assert second.action == ACTION_ALARM


def test_the_restart_policy_fires_and_then_respects_its_cooldown():
    trace = _trace(n=6, pending_requests=2, pending_tokens=50)
    pol = LivenessPolicy(
        confirmations=1, restart_after_alarms=3, restart_cooldown_s=300.0
    )
    assert assess(trace, pol, alarms_raised=0).action == ACTION_ALARM
    hot = assess(trace, pol, alarms_raised=2, since_last_restart_s=10.0)
    assert hot.action == ACTION_ALARM, "cooldown must withhold the restart"
    assert "restart loop" in hot.detail
    cold = assess(trace, pol, alarms_raised=2, since_last_restart_s=9999.0)
    assert cold.action == ACTION_RESTART


def test_restart_can_be_disabled_entirely():
    trace = _trace(n=6, pending_requests=2, pending_tokens=50)
    pol = LivenessPolicy(confirmations=1, restart_after_alarms=None)
    r = assess(trace, pol, alarms_raised=99, since_last_restart_s=9999.0)
    assert r.action == ACTION_ALARM


def test_pending_tokens_alone_are_enough_to_mean_work():
    """A queued request with no admitted req_pool slot still counts."""
    trace = _trace(n=6, pending_requests=0, pending_tokens=4096)
    r = assess(trace, LivenessPolicy(confirmations=1))
    assert r.verdict == WEDGED


def test_the_replacement_gate_arms_during_an_admission_wedge():
    """THE fix, stated against the shipped gate's own failure mode.

    The shipped gate arms only while a batch exists, so it is False for the
    whole wedge. The replacement arms whenever work is pending.
    """

    class _Wedged:
        is_initializing = False
        cur_batch_for_debug = None  # no batch -- the shipped gate goes silent
        waiting_queue = (object(), object())

    class _Idle:
        is_initializing = False
        cur_batch_for_debug = None
        waiting_queue = ()

    shipped_wedged = _Wedged.is_initializing or _Wedged.cur_batch_for_debug is not None
    assert shipped_wedged is False, "the shipped gate is off during the wedge"
    assert build_liveness_is_active(_Wedged())() is True
    # And it stays off when there is genuinely nothing to do.
    assert build_liveness_is_active(_Idle())() is False


def test_the_monitoring_dict_exposes_deltas_not_just_a_verdict():
    """An operator seeing only 'wedged' cannot tell which wedge it is."""
    trace = _trace(n=6, pending_requests=7, pending_tokens=180_000)
    d = assess(trace, LivenessPolicy(confirmations=1)).to_monitoring_dict()
    assert d["liveness_verdict"] == WEDGED
    assert d["liveness_pending_requests"] == 7
    assert d["liveness_progressing"] == 0
    for f in ("completions", "decode_steps", "prefill_chunks"):
        assert f"liveness_delta_{f}" in d


def test_the_window_slides():
    """Old samples must not keep a dead server looking alive."""
    trace = _trace(
        n=6, dt=10.0, prefill_chunks=5, pending_requests=1, pending_tokens=10
    )
    frozen = trace[-1]
    for i in range(1, 7):
        trace.append(
            ProgressSample(
                t_s=frozen.t_s + i * 10.0,
                completions=frozen.completions,
                decode_steps=frozen.decode_steps,
                prefill_chunks=frozen.prefill_chunks,
                pending_requests=1,
                pending_tokens=10,
            )
        )
    r = assess(trace, LivenessPolicy(window_s=45.0, confirmations=1))
    assert r.verdict == WEDGED, "the stale early progress must have slid out"


def test_malformed_policies_are_refused():
    with pytest.raises(ProgressLivenessError, match="window_s"):
        LivenessPolicy(window_s=0)
    with pytest.raises(ProgressLivenessError, match="two samples"):
        LivenessPolicy(min_samples=1)
    with pytest.raises(ProgressLivenessError, match="confirmations"):
        LivenessPolicy(confirmations=0)
