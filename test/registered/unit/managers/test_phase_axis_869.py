# SPDX-License-Identifier: Apache-2.0
"""#869: a predicate whose governing input only one phase can produce.

THE DEFECT, in one sentence: ``bundle_is_mid_flight()`` asks "has this phase's
decode bundle had a fair share of steps yet", reads ``decode_steps_this_phase``
to answer, and is evaluated in the PP layout -- where strict purity forbids
decode, so that counter is 0 for the whole phase BY CONSTRUCTION and the
question can only ever be answered "no, it is young".

THE CONSEQUENCE, traced through three call sites:

  bundle_is_mid_flight()  -> True in PP whenever running_bs > 0, permanently
  demand_prefill_tokens() -> short-circuits to 0 (phase_policy.py, "#861f")
  _strict_holds_pp        -> False, so the DRAINED arm is unguarded

``_strict_holds_pp`` is #861d-2's guard, and its own comment states the
property it is there to provide: "the SAME single read as the tp->pp arm, so
the two directions cannot disagree about whether work exists". A guard that is
structurally disarmed in one of the two phases does not provide that property;
it provides it in TP and asserts it in PP.

THE CLASS THIS FILE PINS, which is bigger than the instance:

    A predicate evaluated in BOTH phases whose governing input can only be
    PRODUCED in one of them is not a measurement in the other. It is a
    constant wearing a measurement's name, and every guard keyed on it is
    silently disarmed (or silently pinned) in exactly that phase.

Lineage, because the repetition is the point: #861e counted retracted requests
as decode work and deadlocked W37-E; #861f replaced the measure with a step
count; #861i found the step count had no producer and wired one. Three fixes to
one term, each closing an axis, none of them asking in WHICH PHASE the producer
can run. That is the axis here.

Hermetic: pure dataclass and pure-function tests, no scheduler, no device, no
network.
"""

import pytest

from sglang.srt.managers import phase_policy as pp
from sglang.srt.managers.phase_policy import (
    MIN_DECODE_STEPS_PER_PHASE,
    PP_TO_TP,
    PhasePolicyInputs,
)


def mk(**kw):
    """A snapshot. Defaults match the pre-#869 field defaults on purpose."""
    base = dict(phase="tp", pending_prefill_tokens=0, running_bs=0, now=0.0)
    base.update(kw)
    return PhasePolicyInputs(**base)


def _cfg(**kw):
    base = dict(
        enabled=True,
        flip_tokens=7004,
        flip_cost_s=3.0,
        drain_mode=True,
        pp_exit_tokens=4096,
        pp_prefill_tok_s=1000.0,
        min_dwell_s=0.0,
        tp_decode_floor_s=0.0,
        idle_dwell_s=1.0,
        # STRICT PURITY, which is the whole precondition: prefill may not run
        # in TP, so PP is the only layout that can serve a queued prompt.
        prefill_runs_in_tp=False,
    )
    base.update(kw)
    return pp.PhasePolicyConfig(**base)


# --------------------------------------------------------------- THE INSTANCE


def test_a_bundle_is_not_mid_flight_in_a_layout_that_cannot_fly_it():
    """PP + strict purity: residents exist, decode is forbidden, steps are 0.

    Pre-#869 this returned True -- and would have returned True at every step
    count PP can ever reach, because PP can only ever reach 0.
    """
    inp = mk(
        phase="pp",
        running_bs=3,
        decode_steps_this_phase=0,
        decode_runs_in_this_phase=False,
    )
    assert inp.bundle_is_mid_flight() is False, (
        "PP forbids decode, so the step counter is 0 by construction and there "
        "is no bundle in flight to chop"
    )


def test_the_pp_read_cannot_be_rescued_by_waiting():
    """The tell that it is a constant, not a slow measurement.

    A real anti-chop floor comes out the other way once the work is done. This
    one cannot: every step count PP can reach is 0, so the verdict is the same
    at every batch size. Sweeping the axis is what makes it a CONSTANT rather
    than a value that happens to be low right now.
    """
    verdicts = {
        mk(
            phase="pp",
            running_bs=bs,
            decode_steps_this_phase=0,
            decode_runs_in_this_phase=False,
        ).bundle_is_mid_flight()
        for bs in (1, 2, 4, 8, 64)
    }
    assert verdicts == {False}, (
        "post-#869 the PP answer is uniformly False; pre-#869 it was uniformly "
        "True, and 'uniform' either way is the fingerprint of the class"
    )


def test_the_demand_term_is_no_longer_suppressed_in_pp():
    """The consumer, one hop down: work exists and must be visible as demand."""
    inp = mk(
        phase="pp",
        pending_prefill_tokens=0,  # fully cached prompts: economics reads 0
        admissible_prefill_tokens=8192,  # ... but a pass is still owed
        running_bs=3,
        decode_steps_this_phase=0,
        decode_runs_in_this_phase=False,
    )
    assert inp.demand_prefill_tokens() == 8192, (
        "the mid-flight short-circuit must not fire in a layout that cannot "
        "decode, or the existence term is invisible exactly where it matters"
    )


def test_the_drained_arm_is_guarded_again_in_pp():
    """The guard #861d-2 installed, re-armed. This is the acceptance.

    Pre-#869: pending 0 <= one chunk, running_bs > 0, `_strict_holds_pp` False
    because the demand term was suppressed -> pp_to_tp armed while queued
    prefill work existed that ONLY PP can serve. Post-#869 the arm is refused.
    """
    cfg = _cfg()
    state = pp.PhasePolicyState()
    inp = mk(
        phase="pp",
        pending_prefill_tokens=0,
        admissible_prefill_tokens=8192,
        running_bs=3,
        decode_steps_this_phase=0,
        decode_runs_in_this_phase=False,
        now=100.0,
    )
    d = pp.decide(cfg, state, inp)
    assert d.direction != PP_TO_TP, (
        "PP must not hand the layout back while it holds the only work class "
        f"it can serve; got {d.direction!r} because {d.reason!r}"
    )


# ------------------------------------------------- WHAT MUST NOT HAVE CHANGED


def test_the_tp_anti_chop_floor_is_untouched():
    """d4 thrash, the specimen #861f exists for. TP decodes, so the floor is a
    real measurement there and must still hold."""
    inp = mk(
        phase="tp",
        running_bs=7,
        decode_steps_this_phase=2,
        decode_runs_in_this_phase=True,
    )
    assert inp.bundle_is_mid_flight() is True
    assert inp.demand_prefill_tokens() == 0, "a live TP bundle still suppresses"


def test_the_tp_floor_still_releases_when_the_bundle_has_had_its_steps():
    inp = mk(
        phase="tp",
        running_bs=7,
        pending_prefill_tokens=999,
        decode_steps_this_phase=MIN_DECODE_STEPS_PER_PHASE,
        decode_runs_in_this_phase=True,
    )
    assert inp.bundle_is_mid_flight() is False
    assert inp.demand_prefill_tokens() == 999


def test_a_relaxed_purity_pp_phase_keeps_the_floor():
    """The fix is keyed on CAN THIS LAYOUT DECODE, not on the phase NAME.

    Under a purity mode that permits decode in PP the counter can advance
    there, so the floor is a measurement again and must behave exactly as it
    does in TP. Keying on `phase == "pp"` instead would have broken this.
    """
    inp = mk(
        phase="pp",
        running_bs=2,
        decode_steps_this_phase=1,
        decode_runs_in_this_phase=True,
    )
    assert inp.bundle_is_mid_flight() is True


def test_the_default_reproduces_the_pre_869_behaviour():
    """Every stand-in and every non-purity deployment is unaffected.

    The field defaults True, so a caller that does not supply it gets exactly
    the old predicate -- which is also the CORRECT one there, since without
    purity decode runs in whatever layout is up.
    """
    inp = mk(phase="pp", running_bs=3, decode_steps_this_phase=0)
    assert inp.decode_runs_in_this_phase is True
    assert inp.bundle_is_mid_flight() is True


# ------------------------------------------------------------- THE CLASS GATE


@pytest.mark.parametrize(
    "phase,can_decode,steps,expected",
    [
        # A layout that cannot decode is never mid-flight, at any step count.
        ("pp", False, 0, False),
        ("pp", False, MIN_DECODE_STEPS_PER_PHASE, False),
        # A layout that can decode answers from the counter, in either phase.
        ("pp", True, 0, True),
        ("pp", True, MIN_DECODE_STEPS_PER_PHASE, False),
        ("tp", True, 0, True),
        ("tp", True, MIN_DECODE_STEPS_PER_PHASE, False),
    ],
)
def test_the_predicate_is_a_function_of_capability_not_of_phase_name(
    phase, can_decode, steps, expected
):
    """The full truth table, so a future edit cannot re-introduce the phase
    name as the discriminator without turning a row red."""
    inp = mk(
        phase=phase,
        running_bs=4,
        decode_steps_this_phase=steps,
        decode_runs_in_this_phase=can_decode,
    )
    assert inp.bundle_is_mid_flight() is expected
