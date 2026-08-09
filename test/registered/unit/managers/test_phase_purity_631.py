"""#631 STRICT PHASE PURITY -- the invariant, pinned.

The rule under test (user, 2026-08-09, hard): no decode step executes in
the PP prefill layout, and not a single token is prefilled in the TP decode
layout. Work for the other layout is deferred and executed batched after
the flip.

WHAT THESE TESTS PIN, and why each one exists
---------------------------------------------
The defect these replace was not a wrong value, it was an INTERLEAVING:
87 decode batch records executed in the PP layout at 35 tok/s with no CUDA
graphs while the policy simultaneously refused to leave PP because prefill
was pending (metal, 21:15:25-21:16:15Z). So the tests assert the GATE
DECISIONS across the full cross-product of (layout, mode, load), not a
single happy path -- an interleaving defect hides in the combinations, not
in the common case.
"""

from __future__ import annotations

import pytest

from sglang.srt.managers.phase_policy import (
    PHASE_PP,
    PHASE_TP,
    PP_TO_TP,
    TP_TO_PP,
    PhasePolicyConfig,
    PhasePolicyInputs,
    PhasePolicyState,
    decide,
    observe_idle,
)
from sglang.srt.managers.phase_purity import (
    MODE_OFF,
    MODE_STRICT,
    MODE_THRESHOLD,
    PhasePurity,
    PhasePurityError,
    decode_blocked_here,
    parse_purity,
    prefill_blocked_here,
    validate_purity_policy_pair,
)


# -- parsing ------------------------------------------------------------


def test_default_is_strict():
    """The DEFAULT is the strict rule -- not 'off'. A default of 'off'
    would make every deployment that does not name the flag reproduce the
    defect."""
    assert parse_purity(None).mode == MODE_STRICT
    assert parse_purity("").mode == MODE_STRICT
    assert parse_purity("strict").strict


def test_threshold_parses_and_zero_collapses_to_strict():
    p = parse_purity("threshold:3")
    assert p.mode == MODE_THRESHOLD
    assert p.decode_in_pp_threshold == 3
    # n=0 means "no decode in PP", which IS strict; one representation for
    # one behaviour keeps the gate logic single-branch.
    assert parse_purity("threshold:0").strict


@pytest.mark.parametrize(
    "raw",
    ["threshold", "threshold:", "threshold:x", "threshold:-1", "loose", "1"],
)
def test_invalid_purity_is_loud(raw):
    """A silently-ignored purity would serve decode from the prefill layout
    while the operator believes it cannot happen."""
    with pytest.raises(PhasePurityError):
        parse_purity(raw)


# -- the gate decisions -------------------------------------------------


class _Sched:
    """The two fields the gates read, and nothing else."""

    def __init__(self, phase, purity, enabled=True):
        self.server_args = type(
            "A", (), {"enable_phase_flip": enabled, "phase_flip_purity": None}
        )()
        self.phase_flip_active_stack = phase
        self._phase_purity = purity


def test_strict_forbids_decode_in_pp_at_every_batch_size():
    """THE headline invariant. Parameterised over batch size because the
    defect ran with running_bs 2 -- a rule that only held for large batches
    would have missed it."""
    sched = _Sched(PHASE_PP, PhasePurity(mode=MODE_STRICT))
    for bs in (1, 2, 3, 8, 64):
        assert decode_blocked_here(sched, bs) is True


def test_strict_forbids_prefill_in_tp():
    sched = _Sched(PHASE_TP, PhasePurity(mode=MODE_STRICT))
    assert prefill_blocked_here(sched) is True


def test_purity_gates_are_layout_directional():
    """Each prohibition binds in ONE layout. A gate that fired in both
    would stall the server completely -- so this pins that decode is free
    in TP and prefill is free in PP."""
    tp = _Sched(PHASE_TP, PhasePurity(mode=MODE_STRICT))
    pp = _Sched(PHASE_PP, PhasePurity(mode=MODE_STRICT))
    assert decode_blocked_here(tp, 4) is False  # decode belongs in TP
    assert prefill_blocked_here(pp) is False  # prefill belongs in PP


def test_threshold_escape_allows_small_decode_in_pp_only():
    sched = _Sched(PHASE_PP, PhasePurity(mode=MODE_THRESHOLD, decode_in_pp_threshold=2))
    assert decode_blocked_here(sched, 1) is False
    assert decode_blocked_here(sched, 2) is False
    assert decode_blocked_here(sched, 3) is True  # above the escape


def test_threshold_never_relaxes_prefill_in_tp():
    """The escape hatch is asymmetric ON PURPOSE: TP prefill is a 4.3x
    throughput loss with no latency argument on the other side."""
    sched = _Sched(PHASE_TP, PhasePurity(mode=MODE_THRESHOLD, decode_in_pp_threshold=8))
    assert prefill_blocked_here(sched) is True


def test_off_lifts_both_prohibitions():
    assert decode_blocked_here(_Sched(PHASE_PP, PhasePurity(mode=MODE_OFF)), 4) is False
    assert prefill_blocked_here(_Sched(PHASE_TP, PhasePurity(mode=MODE_OFF))) is False


def test_flip_disabled_gates_nothing():
    """Byte-identical default path for every instance without the flip."""
    sched = _Sched(PHASE_PP, PhasePurity(mode=MODE_STRICT), enabled=False)
    assert decode_blocked_here(sched, 4) is False
    assert prefill_blocked_here(sched) is False


# -- the deadlock guard -------------------------------------------------


def _cfg(**kw):
    base = dict(enabled=True, flip_tokens=7004, min_dwell_s=3.0)
    base.update(kw)
    return PhasePolicyConfig(**base)


def test_purity_without_a_pp_window_is_refused_at_boot():
    """Purity makes a PP phase reachable that may not decode and cannot
    admit prefill. The bounded window is its ONLY exit, so the combination
    without one is refused rather than deployed."""
    with pytest.raises(PhasePurityError):
        validate_purity_policy_pair(
            PhasePurity(mode=MODE_STRICT), _cfg(pp_window_s=0.0)
        )


def test_purity_off_does_not_require_a_window():
    validate_purity_policy_pair(PhasePurity(mode=MODE_OFF), _cfg(pp_window_s=0.0))


# -- the policy windows that make purity progress ------------------------


def _drive(cfg, state, phase, pending, bs, now):
    inp = PhasePolicyInputs(
        phase=phase, pending_prefill_tokens=pending, running_bs=bs, now=now
    )
    observe_idle(state, inp)
    return decide(cfg, state, inp)


def test_sustained_backlog_still_leaves_pp_via_the_window():
    """THE STARVATION REGRESSION. Reproduces the metal condition: pending
    prefill far above N forever, decode work waiting. Before the window
    this returned 'holding in pp' on every call, without end."""
    cfg = _cfg(pp_window_s=15.0)
    state = PhasePolicyState()
    # t=0 enters PP; the backlog never drops below N.
    assert _drive(cfg, state, PHASE_PP, 302757, 2, 0.0).direction is None
    assert _drive(cfg, state, PHASE_PP, 302757, 2, 14.0).direction is None
    out = _drive(cfg, state, PHASE_PP, 302757, 2, 15.0)
    assert out.direction == PP_TO_TP
    assert "pp window" in out.reason


def test_the_window_waits_for_decode_work_to_exist():
    """With nothing decoding there is nothing being starved, so a PP phase
    chewing through a backlog is left alone."""
    cfg = _cfg(pp_window_s=15.0)
    state = PhasePolicyState()
    assert _drive(cfg, state, PHASE_PP, 302757, 0, 0.0).direction is None
    assert _drive(cfg, state, PHASE_PP, 302757, 0, 99.0).direction is None


def test_decode_floor_stops_the_mirror_starvation():
    """Under purity the backlog is always above N, so without a floor the
    TP phase would last exactly one min_dwell and decode would starve in
    the mirror image of the defect."""
    cfg = _cfg(min_dwell_s=3.0, tp_decode_floor_s=10.0)
    state = PhasePolicyState()
    assert _drive(cfg, state, PHASE_TP, 302757, 2, 0.0).direction is None
    held = _drive(cfg, state, PHASE_TP, 302757, 2, 9.0)
    assert held.direction is None
    assert "decode floor" in held.reason
    assert _drive(cfg, state, PHASE_TP, 302757, 2, 10.0).direction == TP_TO_PP


def test_decode_floor_does_not_delay_an_idle_decode_layout():
    """A long prompt arriving at a decode-idle server must reach the PP
    layout inside its TTFT -- the floor protects decode, and there is none."""
    cfg = _cfg(min_dwell_s=0.0, tp_decode_floor_s=10.0)
    state = PhasePolicyState()
    assert _drive(cfg, state, PHASE_TP, 302757, 0, 0.5).direction == TP_TO_PP


def test_both_queues_drain_in_bounded_alternation():
    """THE END-TO-END PROPERTY the green criterion asks for: under a load
    that always has prefill pending AND always has decode waiting, the
    policy must visit BOTH layouts within a bounded time -- never park in
    either. Drives 300 s of a saturated arrival pattern and asserts both
    directions actually occur, repeatedly."""
    cfg = _cfg(min_dwell_s=3.0, pp_window_s=15.0, tp_decode_floor_s=10.0)
    state = PhasePolicyState()
    phase = PHASE_PP
    flips = {PP_TO_TP: 0, TP_TO_PP: 0}
    t = 0.0
    while t < 300.0:
        out = _drive(cfg, state, phase, 302757, 2, t)
        if out.wants_flip:
            flips[out.direction] += 1
            # The flip commits: the phase the policy asked for becomes the
            # observed one, and the caller stamps the dwell clock.
            phase = PHASE_TP if out.direction == PP_TO_TP else PHASE_PP
            state.last_flip_at = t
        t += 1.0
    assert flips[PP_TO_TP] >= 5, flips
    assert flips[TP_TO_PP] >= 5, flips
    # Neither side may dominate: a cycle is one window plus one floor, so
    # the two counts stay within one of each other.
    assert abs(flips[PP_TO_TP] - flips[TP_TO_PP]) <= 1, flips


def test_phase_clock_follows_an_unannounced_layout_change():
    """A manual POST /phase_flip changes the layout without the policy
    arming it. The window clock must restart on the OBSERVED phase change,
    or the first decision after a manual flip would measure a window that
    never happened.

    Note the exact semantics this pins, which is the only honest one: the
    clock starts when the policy FIRST OBSERVES the new layout, not when
    the flip physically happened. The policy is not told about a flip it
    did not arm, so an unobserved interval cannot count -- the effect is a
    window that may run slightly long after a manual flip, never one that
    fires early on time the server never spent in the phase."""
    cfg = _cfg(pp_window_s=15.0)
    state = PhasePolicyState()
    _drive(cfg, state, PHASE_TP, 0, 1, 90.0)
    # Someone flips to PP by hand; the policy first sees PP at t=100.
    assert _drive(cfg, state, PHASE_PP, 302757, 2, 100.0).direction is None
    assert state.phase_since == 100.0
    assert _drive(cfg, state, PHASE_PP, 302757, 2, 110.0).direction is None
    assert _drive(cfg, state, PHASE_PP, 302757, 2, 115.0).direction == PP_TO_TP
