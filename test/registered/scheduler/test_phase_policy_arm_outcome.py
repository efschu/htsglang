# SPDX-License-Identifier: Apache-2.0
"""#656: the arm OUTCOME has to reach the policy, or a refused flip spins forever.

THE SIGNATURE THIS PINS (boot E, /spinning/evidence-631/kvuniverse-r1/boot_e.log).
The runtime refused every ``tp_to_pp`` seam -- ``staging 464 MiB needed but only
444 MiB is spendable`` -- and after eight consecutive group abandons its own
``SEAM_ABANDON_CAP`` installed the ``seam unfundable`` blocking guard, which is
the runtime saying "stop asking". The policy never heard it: ``note_flip_armed``
committed the dwell clock the moment ``decide`` WANTED a flip, and
``handle_phase_flip`` dropped ``(ok, msg)`` on the floor for internal requests.
So the policy re-armed on the bare ``min_dwell`` forever -- 179 arms, 0 completed
cutovers, 336 ``seam unfundable`` refusals -- while strict purity kept prefill out
of TP. /health answered 200 and the instance emitted no tokens.

The runtime's backoff could not help: it damps the SEAM's retries, and the seam
was being re-armed from outside. A bound on retries is only a bound if the thing
that retries can see it.

Three properties, each of which the pre-fix code fails:
  1. a REFUSED arm is not a flip, so it must not commit the dwell clock;
  2. consecutive refusals must back off, so the arm rate collapses;
  3. a refusal that keeps repeating must DEGRADE LOUDLY -- named once, counted --
     rather than spin silently at the dwell interval.
"""

import pytest

from sglang.srt.managers.phase_policy import (
    PHASE_PP,
    PHASE_TP,
    TP_TO_PP,
    PhasePolicyConfig,
    PhasePolicyInputs,
    PhasePolicyState,
    decide,
    note_flip_armed,
    note_flip_completed,
    note_flip_outcome,
)

#: Verbatim from boot_e.log, so the test fails for the reason the rig failed.
BOOT_E_REFUSAL = (
    "seam unfundable: tp_to_pp abandoned 8 times consecutively; staging 464 "
    "MiB needed but only 444 MiB is spendable"
)


def _cfg(**extra):
    """The ship policy config, under strict purity (prefill cannot run in TP)."""
    base = dict(
        enabled=True,
        flip_tokens=7004,
        min_dwell_s=3.0,
        idle_dwell_s=30.0,
        prefill_runs_in_tp=False,
    )
    base.update(extra)
    return PhasePolicyConfig(**base)


def _pending_prefill_in_tp(now: float) -> PhasePolicyInputs:
    """Boot E's standing input: sitting in TP with a prefill that only PP can run."""
    return PhasePolicyInputs(
        phase=PHASE_TP, pending_prefill_tokens=1, running_bs=0, now=now
    )


def _arm_cycle(cfg, state, now, ok, message=BOOT_E_REFUSAL):
    """One scheduler tick, all the way to the flip's fate.

    ``ok=False`` models the boot-E shape exactly: the arm itself is ACCEPTED
    (the runtime only refuses at arm once its own cap has installed a guard),
    and the seam abandons rounds later. Both report through
    ``note_flip_outcome``; only a real cutover reports completion.

    Returns the direction armed, or None if the policy held.
    """
    d = decide(cfg, state, _pending_prefill_in_tp(now))
    if not d.wants_flip:
        return None
    note_flip_armed(state, d, now)
    if ok:
        note_flip_outcome(cfg, state, d.direction, True, "", now)
        note_flip_completed(cfg, state, d.direction, now)
    else:
        note_flip_outcome(cfg, state, d.direction, False, message, now)
    return d.direction


# ---------------------------------------------------------------------------
# 1. A refused arm is not a flip.
# ---------------------------------------------------------------------------


def test_refused_arm_does_not_commit_the_dwell_clock():
    cfg, state = _cfg(), PhasePolicyState()
    before = state.last_flip_at

    assert _arm_cycle(cfg, state, now=100.0, ok=False) == TP_TO_PP
    assert state.last_flip_at == before, (
        "a refused arm moved no request and changed no layout; recording it as "
        "'the last flip' is what made min_dwell the retry interval"
    )


def test_a_successful_arm_does_commit_the_dwell_clock():
    """The success path is unchanged -- this is the control arm of the fix."""
    cfg, state = _cfg(), PhasePolicyState()

    assert _arm_cycle(cfg, state, now=100.0, ok=True, message="") == TP_TO_PP
    assert state.last_flip_at == 100.0
    assert state.arm_refusals.get(TP_TO_PP, 0) == 0
    assert state.flips_completed == 1


# ---------------------------------------------------------------------------
# 2. Consecutive refusals back off.
# ---------------------------------------------------------------------------


def test_consecutive_refusals_grow_the_hold():
    cfg, state = _cfg(), PhasePolicyState()

    # First refusal at t=100 -> the next arm may not happen at t=103 (min dwell).
    assert _arm_cycle(cfg, state, now=100.0, ok=False) == TP_TO_PP
    assert _arm_cycle(cfg, state, now=103.0, ok=False) is None, (
        "the second arm landed one min_dwell after the first: this is the "
        "boot-E retry rate exactly"
    )
    d = decide(cfg, state, _pending_prefill_in_tp(103.0))
    assert not d.wants_flip and "refus" in d.reason.lower()

    # The hold doubles: 6s, then 12s, then 24s ...
    assert _arm_cycle(cfg, state, now=106.0, ok=False) == TP_TO_PP
    assert _arm_cycle(cfg, state, now=112.0, ok=False) is None
    assert _arm_cycle(cfg, state, now=118.0, ok=False) == TP_TO_PP


def test_the_boot_e_arm_storm_collapses():
    """Nine minutes of the boot-E condition, ticked at the round cadence.

    Measured on metal before the fix: 179 arms and 0 cutovers over the run.
    The bound here is deliberately loose -- the property is 'the arm rate is
    no longer the dwell rate', not a particular schedule.
    """
    cfg, state = _cfg(), PhasePolicyState()
    arms = 0
    t = 100.0
    while t < 100.0 + 9 * 60:
        if _arm_cycle(cfg, state, now=t, ok=False) is not None:
            arms += 1
        t += 0.5

    assert arms < 20, f"{arms} arms in 9 minutes is still a storm (metal saw 179)"
    assert state.arm_refusals[TP_TO_PP] == arms


# ---------------------------------------------------------------------------
# 3. It degrades loudly rather than spinning silently.
# ---------------------------------------------------------------------------


def test_repeated_refusal_degrades_and_names_the_reason(caplog):
    cfg, state = _cfg(), PhasePolicyState()
    t = 100.0
    with caplog.at_level("ERROR"):
        for _ in range(cfg.refusal_degrade_after + 4):
            while _arm_cycle(cfg, state, now=t, ok=False) is None:
                t += 0.5
                if t > 100.0 + 3600:
                    pytest.fail("the policy stopped arming entirely")
            t += 0.5

    assert state.arm_degraded.get(TP_TO_PP), "no degraded verdict was recorded"
    assert "464" in state.arm_degraded[TP_TO_PP], (
        "the degraded verdict must carry the runtime's numbers, not a generic "
        "'flip failed'"
    )
    assert state.arm_degrade_events == 1, (
        "the degradation must be announced exactly once per episode -- a "
        "12765-line log flood has already self-killed this feature once"
    )
    loud = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(loud) == 1 and "464" in loud[0].getMessage()


def test_degradation_still_re_probes_so_a_transient_can_heal():
    """Degraded is a LOUD hold, not a latch: the condition may change."""
    cfg, state = _cfg(), PhasePolicyState()
    t = 100.0
    for _ in range(cfg.refusal_degrade_after):
        while _arm_cycle(cfg, state, now=t, ok=False) is None:
            t += 0.5
        t += 0.5
    assert state.arm_degraded.get(TP_TO_PP)

    # Far enough ahead that the capped hold has expired, and this time the
    # runtime funds the seam.
    t += cfg.refusal_backoff_cap_s + 1.0
    assert _arm_cycle(cfg, state, now=t, ok=True, message="") == TP_TO_PP
    assert not state.arm_degraded.get(TP_TO_PP)
    assert state.arm_refusals.get(TP_TO_PP, 0) == 0
    assert state.last_flip_at == t
    assert state.flips_completed == 1


def test_an_accepted_arm_that_abandons_is_still_a_refusal():
    """The boot-E window the obvious fix misses.

    ``arm`` returns True for the first SEAM_ABANDON_CAP attempts of an
    unfundable configuration -- the guard is only installed after them. If
    'accepted' retired the attempt, those attempts would still run at the
    dwell rate, which is where boot E spent its arms.
    """
    cfg, state = _cfg(), PhasePolicyState()
    d = decide(cfg, state, _pending_prefill_in_tp(100.0))
    note_flip_armed(state, d, 100.0)
    note_flip_outcome(cfg, state, d.direction, True, "armed", 100.0)
    assert state.pending_arm is not None, "an accepted arm is still outstanding"

    # ... and the seam abandons three rounds later.
    note_flip_outcome(cfg, state, d.direction, False, BOOT_E_REFUSAL, 101.5)
    assert state.last_flip_at == 0.0
    assert state.arm_refusals[TP_TO_PP] == 1
    assert not decide(cfg, state, _pending_prefill_in_tp(103.0)).wants_flip


def test_a_refusal_in_one_direction_does_not_gag_the_other():
    """pp_to_tp and tp_to_pp are funded by different legs and fail apart."""
    cfg, state = _cfg(), PhasePolicyState()
    for _ in range(cfg.refusal_degrade_after):
        state.arm_refusals[TP_TO_PP] = state.arm_refusals.get(TP_TO_PP, 0)
        note_flip_outcome(cfg, state, TP_TO_PP, False, BOOT_E_REFUSAL, 100.0)
    assert state.arm_degraded.get(TP_TO_PP)

    # Sitting in PP with a decode backlog and nothing to prefill: the policy
    # must still be free to go to TP.
    d = decide(
        cfg,
        state,
        PhasePolicyInputs(
            phase=PHASE_PP, pending_prefill_tokens=0, running_bs=4, now=1000.0
        ),
    )
    assert d.wants_flip and d.direction != TP_TO_PP
