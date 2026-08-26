# SPDX-License-Identifier: Apache-2.0
"""#869c: the economics subtraction rested on a premise nobody checked.

THE DEFECT. Two subtractions, one justification, one of them verified:

  scheduler.py  `_pending_now -= _seam_transport_now`   UNCONDITIONAL
  scheduler.py  `_seam_serviceable_now = ...`           gated on `_in_tp_now`
                                                        AND on
                                                        `seam_transport_premise_holds`

Both rest on the same claim -- a seam re-admission is cheap flip transport,
because "their prefixes are served by read-through from the canonical store".
#861j verified that claim for the EXISTENCE term and never backported the
verification to the ECONOMICS term beside it.

WHY IT MATTERS, and the blast radius is deliberately narrow. `demand_prefill_
tokens()` takes `max(pending, admissible - serviceable)` and
`admissible_prefill_tokens` is NOT reduced by seam transport, so with #869
landed `_strict_holds_pp` still sees the true backlog. What stays exposed is the
one arm still reading RAW pending: the #677(a) BLOCKED-ADMISSION STALL ESCAPE
(`phase_policy.py`, `inp.pending_prefill_tokens > cfg.pp_exit_tokens`). A
deflated pending holds a genuine stall BELOW its own escape threshold -- the
wedge that escape exists to end. Delayed escape, not a wrong answer.

THE FALSIFIER, as specified before the fix was written: PP phase, seam-transport
tokens present, premise FALSE, raw pending ABOVE `pp_exit_tokens` before the
subtraction and BELOW it after. Assert the stall escape still arms. It must FAIL
on the pre-#869c tree and pass with the fix.

Hermetic: pure functions and a policy stand-in. No scheduler, no device, no
network.
"""

import pytest

from sglang.srt.managers import phase_policy as pp
from sglang.srt.managers.phase_policy import PhasePolicyInputs
from sglang.srt.managers.phase_purity import seam_transport_deduction

#: The live shape this is drawn from: one chunk, and a backlog just above it.
PP_EXIT_TOKENS = 4096
SEAM_TOKENS = 3000
RAW_PENDING = 5000  # above the chunk...
# ...and 5000 - 3000 = 2000, below it. That gap IS the defect.


def _cfg(**kw):
    base = dict(
        enabled=True,
        flip_tokens=7004,
        flip_cost_s=3.0,
        drain_mode=True,
        pp_exit_tokens=PP_EXIT_TOKENS,
        pp_prefill_tok_s=1000.0,
        min_dwell_s=0.0,
        tp_decode_floor_s=0.0,
        idle_dwell_s=1.0,
        prefill_runs_in_tp=False,
        # ISOLATE THE #677(a) ESCAPE: the hand-set pp_window stopwatch is a
        # SECOND arm that fires on the same specimen for a different reason,
        # and leaving it on would let this test pass without the escape ever
        # being consulted. 0 disables it (phase_policy.py:1506).
        pp_window_s=0.0,
        decode_stall_slo_s=0.0,
    )
    base.update(kw)
    return pp.PhasePolicyConfig(**base)


def _pending_after_deduction(*, in_tp, premise_holds):
    """The build site's arithmetic, isolated."""
    return max(
        0,
        RAW_PENDING
        - seam_transport_deduction(
            SEAM_TOKENS, in_tp=in_tp, premise_holds=premise_holds
        ),
    )


# ------------------------------------------------------------- THE RULE ITSELF


def test_an_unverified_premise_buys_no_deduction():
    """THE FIX. Premise false -> nothing is deducted, whatever the phase."""
    assert seam_transport_deduction(SEAM_TOKENS, in_tp=True, premise_holds=False) == 0
    assert seam_transport_deduction(SEAM_TOKENS, in_tp=False, premise_holds=False) == 0


def test_the_deduction_is_also_phase_gated_like_its_twin():
    """A stamp buys nothing in the layout that cannot admit it."""
    assert seam_transport_deduction(SEAM_TOKENS, in_tp=False, premise_holds=True) == 0


def test_a_verified_premise_in_tp_still_deducts():
    """NOT A DISARM. The #861j/W32 behaviour is preserved where it was earned."""
    assert (
        seam_transport_deduction(SEAM_TOKENS, in_tp=True, premise_holds=True)
        == SEAM_TOKENS
    )


def test_the_deduction_never_manufactures_negative_pending():
    assert seam_transport_deduction(-5, in_tp=True, premise_holds=True) == 0
    assert seam_transport_deduction(None, in_tp=True, premise_holds=True) == 0


@pytest.mark.parametrize(
    "in_tp,premise,expected",
    [
        (True, True, SEAM_TOKENS),
        (True, False, 0),
        (False, True, 0),
        (False, False, 0),
    ],
)
def test_the_deduction_truth_table(in_tp, premise, expected):
    """Both gates, every combination, so neither can be dropped silently."""
    assert (
        seam_transport_deduction(SEAM_TOKENS, in_tp=in_tp, premise_holds=premise)
        == expected
    )


# ------------------------------------ THE SPECIFIED FALSIFIER: THE STALL ESCAPE


def _stall_decision(pending_tokens):
    """Drive the #677(a) blocked-admission escape with a stalled PP phase."""
    cfg = _cfg()
    now = 10_000.0
    state = pp.PhasePolicyState()
    state.phase_since = now - 600.0
    # Prefill has made no progress for far longer than the stall window.
    state.last_prefill_progress_at = now - 600.0
    inp = PhasePolicyInputs(
        phase="pp",
        pending_prefill_tokens=pending_tokens,
        admissible_prefill_tokens=pending_tokens,
        running_bs=4,
        decode_steps_this_phase=0,
        decode_runs_in_this_phase=False,  # #869: strict purity, PP cannot decode
        now=now,
        seam_transport_tokens=SEAM_TOKENS,
        seam_serviceable_tokens=0,
    )
    return pp.decide(cfg, state, inp)


def test_the_stall_escape_arms_on_the_true_backlog():
    """PP phase, premise FALSE, raw pending above one chunk.

    With the premise unverified nothing is deducted, pending stays at 5000 >
    4096, and the blocked-admission escape can still see the stall.
    """
    pending = _pending_after_deduction(in_tp=False, premise_holds=False)
    assert pending == RAW_PENDING, "an unverified premise must not deflate pending"
    assert pending > PP_EXIT_TOKENS
    d = _stall_decision(pending)
    assert d.wants_flip, f"the stall escape must still arm; got {d.reason!r}"


def test_the_pre_869c_deduction_hid_the_stall():
    """THE DEFECT, PINNED AS ARITHMETIC.

    Deducting on an unverified premise -- the pre-#869c behaviour, reproduced
    here by asking for the deduction unconditionally -- takes pending from 5000
    to 2000, under the 4096 chunk. The escape's own threshold can then never be
    crossed, however long the stall runs.
    """
    deflated = max(0, RAW_PENDING - SEAM_TOKENS)  # the unconditional subtraction
    assert deflated < PP_EXIT_TOKENS, "the specimen must actually cross the chunk"
    d = _stall_decision(deflated)
    # Asserted on the VERDICT, not on a substring of the reason: an earlier
    # draft of this test matched "stall" and hit
    # SGLANG_PHASE_POLICY_DECODE_STALL_SLO_S in an unrelated suggestion line.
    assert not d.wants_flip, (
        "with pending deflated below one chunk nothing should arm -- the "
        f"escape's threshold is unreachable; got {d.direction!r} "
        f"because {d.reason!r}"
    )


def test_the_two_subtractions_are_now_one_rule():
    """The economics and existence terms agree about the same population.

    Whatever the deduction claims, the serviceable credit claims the same -- so
    a token cannot be both 'not PP workload' and 'not serviceable here'. That
    double-classification is the W37-F oscillation #861j was written to end, and
    leaving one side unverified reopened it from the other direction.
    """
    for in_tp in (True, False):
        for premise in (True, False):
            deduction = seam_transport_deduction(
                SEAM_TOKENS, in_tp=in_tp, premise_holds=premise
            )
            serviceable = SEAM_TOKENS if (in_tp and premise) else 0
            assert deduction == serviceable, (in_tp, premise)
