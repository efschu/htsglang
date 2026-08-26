"""#874: an UNPRICED seam is a config state, so it is checkable at boot.

WHAT WAS MEASURED, and it is the whole reason this file exists. Boot
w40_857strict (2026-08-25 23:42 -> 2026-08-26 04:47, PP=3 + phase flip,
``--phase-flip-purity strict``) answered a two-token health-check ping in
15.6-16.6 s, three times over four and a half hours. The ping is 13 tokens.
The 16 s are not compute:

    04:46:18  PHASE-POLICY arming tp_to_pp: pending prefill 13 tok > 0
              (purity: prefill cannot run in tp, nothing decoding)
    04:46:24  PHASE-FLIP DONE tp_to_pp in 6204.1 ms
    04:46:27  PHASE-POLICY arming pp_to_tp: DRAINED: 0 tok remaining
    04:46:34  PHASE-FLIP DONE pp_to_tp in 6061.4 ms

Two cutovers per request, ~6.2 s each. Every one of the boot's 611 flips
carried ``sent 0 cells / 0.00 MiB``: the seam moved no KV at all. The seam
census names where the seconds went instead --

    worst 'refill_highwater->weights_refill' 4868.4 ms (77% of the walk)
    REFILL tp_to_pp took 4.868 s for 16362.7 MiB (3361 MiB/s)

-- the weights arena, 16.4 GiB per rank per flip, PCIe bound. That term is
occupancy-INDEPENDENT by construction (``phase_flip_runtime`` says so at the
``movers_ms`` bracket: "the weights arena refill is the same bytes whatever
the KV live set is"), so a payload-free cutover is not the cheap case. It is
the same case.

THE POLICY IS NOT DEFECTIVE HERE, and that is the finding worth pinning.
Under strict purity ``effective_flip_threshold`` returns 0 BY DESIGN
(``if not cfg.prefill_runs_in_tp: return 0``), because a sub-N prompt cannot
run in TP at all and a surcharge that resurrected a threshold would leave
short prompts unrunnable -- the wedge recorded at the scheduler's own wiring
site, "a one-token health check wedged an otherwise idle server". The arm
then fires on ANY pending token, deliberately, against the user's law
"Break-even ist NICHT der Trigger".

SO THE DEFECT IS A CLASS, NOT AN INSTANCE: a mechanism with a large fixed
cost, whose price gate has been switched off by a mode flag, is free to fire
at whatever frequency the arrival pattern dictates -- and nothing in the
instance says so. The cost stays invisible until a user times a ping.

That state is decidable from config alone: ``flip_cost_s > 0`` (the seam
costs something) AND ``effective_flip_threshold(...) == 0`` (nothing is
required to justify paying it). This file pins the predicate that names it,
and pins that it does NOT fire on a priced seam -- a guard that cannot say
"no" is not a guard.

WHAT THIS FILE DELIBERATELY DOES NOT CLAIM: that strict purity is wrong.
It is a supported mode with a documented reason. The predicate reports a
cost; it never refuses a boot.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import unittest

from sglang.srt.managers import phase_policy as pp
from sglang.test.test_utils import CustomTestCase

NOW = 1000.0

#: The measured ping, verbatim from the boot log's arming lines.
PING_TOKENS = 13

#: The measured seam, verbatim from the boot's DONE lines (~6.2 s one way).
MEASURED_FLIP_COST_S = 6.2


def _cfg(prefill_runs_in_tp: bool, flip_cost_s: float = MEASURED_FLIP_COST_S):
    """The booted shape, reduced to the terms this decision reads.

    ``flip_tokens`` is the deployed break-even rung. It is set far above the
    ping on purpose: a threshold at or below 13 would make both purity modes
    answer alike and the comparison below would prove nothing.
    """
    return pp.PhasePolicyConfig(
        enabled=True,
        drain_mode=True,
        drain_mode_strict=True,
        prefill_runs_in_tp=prefill_runs_in_tp,
        flip_tokens=7004,
        flip_cost_s=flip_cost_s,
        decode_strand_weight=0.0,
        decode_contention=0.0,
        pp_window_s=15.0,
        min_dwell_s=0.0,
        tp_decode_floor_s=0.0,
        idle_dwell_s=0.0,
        pp_exit_tokens=4096,
    )


def _state():
    return pp.PhasePolicyState(
        last_flip_at=0.0,
        phase_since=0.0,
        last_phase=pp.PHASE_TP,
        bundle_at_phase_entry=0,
        last_bundle_progress_at=NOW,
        last_prefill_progress_at=NOW,
    )


def _inp(pending: int):
    """The ping's own arrival shape: idle TP, nothing decoding, 13 tokens."""
    return pp.PhasePolicyInputs(
        phase=pp.PHASE_TP,
        pending_prefill_tokens=pending,
        running_bs=0,
        now=NOW,
        nothing_can_run=False,
        target_can_admit=True,
        ready_carriers=0,
        queue_nonempty=True,
        kv_available_tokens=100000,
    )


class TestTheComparisonIsRealAtAll(CustomTestCase):
    """Guards the two decision tests against going vacuous."""

    def test_the_ping_is_far_below_the_deployed_rung(self):
        priced = _cfg(prefill_runs_in_tp=True)
        self.assertGreater(pp.effective_flip_threshold(priced, 0), PING_TOKENS)

    def test_purity_is_what_collapses_the_rung_to_zero(self):
        self.assertEqual(pp.effective_flip_threshold(_cfg(False), 0), 0)


class TestTheMeasuredDecision(CustomTestCase):
    """What the booted config does with the ping, and what the alternative does.

    Both directions are asserted so neither can pass by accident.
    """

    def test_strict_purity_flips_for_thirteen_tokens(self):
        # The booted behaviour, reproduced without a GPU. This is the arming
        # line from 04:46:18, and it is CORRECT for the mode.
        d = pp.decide(_cfg(prefill_runs_in_tp=False), _state(), _inp(PING_TOKENS))
        self.assertEqual(d.direction, pp.TP_TO_PP)
        self.assertIn("purity", (d.reason or "").lower())

    def test_the_priced_mode_serves_the_ping_where_it_stands(self):
        # THE CAN-FAIL PARTNER. Same arrival, same numbers, the price gate on:
        # 13 tokens do not buy a 6.2 s seam, so nothing arms.
        d = pp.decide(_cfg(prefill_runs_in_tp=True), _state(), _inp(PING_TOKENS))
        self.assertIsNone(d.direction)


class TestTheUnpricedSeamIsNamed(CustomTestCase):
    """The class check: a fixed cost with its price gate off, reported at boot."""

    def test_it_fires_on_the_booted_shape(self):
        note = pp.unpriced_seam_note(_cfg(prefill_runs_in_tp=False), running_bs=0)
        self.assertIsNotNone(note)
        # The figure an operator acts on is the ROUND TRIP, because that is
        # what one request costs: in and back out again.
        self.assertIn("12.4", note)

    def test_it_stays_silent_when_the_seam_is_priced(self):
        # CAN-FAIL IN THE OTHER DIRECTION. A predicate that answers "yes" to
        # every config reports nothing.
        self.assertIsNone(
            pp.unpriced_seam_note(_cfg(prefill_runs_in_tp=True), running_bs=0)
        )

    def test_it_stays_silent_when_the_seam_costs_nothing(self):
        # An unmeasured or free seam has no cost to warn about, and warning
        # there would train the reader to ignore the line.
        self.assertIsNone(
            pp.unpriced_seam_note(
                _cfg(prefill_runs_in_tp=False, flip_cost_s=0.0), running_bs=0
            )
        )


if __name__ == "__main__":
    unittest.main()
