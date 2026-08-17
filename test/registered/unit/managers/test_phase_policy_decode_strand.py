"""Flipping to PP must also price the decodes it strands there.

THE GAP THIS CLOSES
-------------------
`break_even_tokens` answers "is there enough pending prefill to repay the
seam?" and nothing else. But a `tp_to_pp` cutover does not only cost the seam:
every request currently decoding is PAUSED and carried into the PP layout,
where decode is forbidden, so it waits out the whole PP window before it
produces another token. That cost is real, scales with how many requests are
decoding, and was simply not in the comparison.

The asymmetry that makes it worth pricing (user, 2026-08-14):

  * misplacing PREFILL is a ONE-OFF, bounded by prompt length:
        P x (1/r_tp - 1/r_pp)
  * stranding DECODE is RECURRING and unbounded in generation length, and in
    PP it also forfeits CUDA graphs and speculation (measured on this rig:
    accept length 3.24, 74.8%).

So the flip decision becomes

    gain = P x (1/r_tp - 1/r_pp)
    cost = seam_roundtrip + running_decodes x pp_window x weight
    flip only if gain > cost

Expressed against the existing threshold, that is a SURCHARGE on N rather
than a new mechanism -- the seconds simply go into the same C:

    N_eff = flip_tokens x (1 + weight x running_bs x pp_window_s / flip_cost_s)

BACKWARD COMPATIBILITY IS THE DEFAULT
-------------------------------------
`flip_cost_s` defaults to 0.0, meaning "the seam was never measured here", and
with it the surcharge is disabled and the threshold is exactly `flip_tokens`
as before. A deployment that does not supply a measured seam cost therefore
behaves byte-identically to the previous policy -- the surcharge is opt-in by
MEASUREMENT, not by flag, which is the right gate for a term whose whole
justification is that it was measured.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.managers.phase_policy import (
    PHASE_TP,
    TP_TO_PP,
    PhasePolicyConfig,
    PhasePolicyInputs,
    PhasePolicyState,
    decide,
    effective_flip_threshold,
    observe_idle,
)
from sglang.test.test_utils import CustomTestCase


def _cfg(**kw):
    base = dict(
        enabled=True,
        flip_tokens=7004,
        min_dwell_s=0.0,
        pp_window_s=15.0,
        tp_decode_floor_s=0.0,
        prefill_runs_in_tp=True,
    )
    base.update(kw)
    return PhasePolicyConfig(**base)


class TestTheSurchargeArithmetic(CustomTestCase):
    def test_no_measured_seam_means_no_surcharge(self):
        """The compatibility default: behaves exactly as before."""
        cfg = _cfg(flip_cost_s=0.0)
        for bs in (0, 1, 8):
            with self.subTest(bs=bs):
                self.assertEqual(effective_flip_threshold(cfg, bs), 7004)

    def test_nothing_decoding_means_no_surcharge(self):
        cfg = _cfg(flip_cost_s=4.8)
        self.assertEqual(effective_flip_threshold(cfg, 0), 7004)

    def test_one_decode_raises_the_bar_by_the_window_ratio(self):
        """15 s window against a 4.8 s seam -> 1 + 15/4.8 = 4.125x."""
        cfg = _cfg(flip_cost_s=4.8, decode_strand_weight=1.0)
        self.assertEqual(effective_flip_threshold(cfg, 1), int(round(7004 * 4.125)))

    def test_the_surcharge_scales_with_the_number_stranded(self):
        cfg = _cfg(flip_cost_s=4.8, decode_strand_weight=1.0)
        one = effective_flip_threshold(cfg, 1)
        four = effective_flip_threshold(cfg, 4)
        self.assertGreater(four, one)
        self.assertEqual(four, int(round(7004 * (1 + 4 * 15.0 / 4.8))))

    def test_weight_zero_disables_it_without_disabling_the_seam_cost(self):
        cfg = _cfg(flip_cost_s=4.8, decode_strand_weight=0.0)
        self.assertEqual(effective_flip_threshold(cfg, 8), 7004)

    def test_purity_strict_still_collapses_to_zero(self):
        """`strict` means a sub-N prompt would never run; the surcharge must
        not resurrect a threshold there."""
        cfg = _cfg(flip_cost_s=4.8, prefill_runs_in_tp=False)
        self.assertEqual(effective_flip_threshold(cfg, 4), 0)


class TestItChangesTheDecision(CustomTestCase):
    def _drive(self, cfg, pending, bs):
        state = PhasePolicyState()
        inp = PhasePolicyInputs(
            phase=PHASE_TP, pending_prefill_tokens=pending, running_bs=bs, now=1000.0
        )
        observe_idle(state, inp)
        return decide(cfg, state, inp)

    def test_a_mid_size_prefill_flips_when_nothing_is_decoding(self):
        cfg = _cfg(flip_cost_s=4.8)
        self.assertEqual(self._drive(cfg, 10_000, 0).direction, TP_TO_PP)

    def test_the_same_prefill_does_not_strand_four_decodes(self):
        """10k tokens repays the seam, but not the seam PLUS four paused
        generations waiting out a 15 s PP window."""
        cfg = _cfg(flip_cost_s=4.8)
        d = self._drive(cfg, 10_000, 4)
        self.assertIsNone(d.direction)
        self.assertIn("strand", d.reason.lower())
        self.assertIn("4 req decoding", d.reason)

    def test_a_large_enough_prefill_still_wins_against_decodes(self):
        cfg = _cfg(flip_cost_s=4.8)
        self.assertEqual(self._drive(cfg, 400_000, 4).direction, TP_TO_PP)


if __name__ == "__main__":
    unittest.main()


class TestTheRestingLayout(CustomTestCase):
    """CHANGED 2026-08-14 (user): rest in TP, not PP.

    Every request decodes and decode in PP is forbidden, so resting in PP
    made every request pay a `pp_to_tp` cutover and then idle back to PP to
    pay it again -- the 882-flip thrash cycle. Only large-prefill requests
    want PP, and above N they can afford the flip by construction.
    """

    def test_default_rest_phase_is_tp(self):
        from sglang.srt.managers.phase_policy import PHASE_TP, REST_DECODE

        cfg = PhasePolicyConfig(enabled=True, flip_tokens=7004)
        self.assertEqual(cfg.rest_state, REST_DECODE)
        self.assertEqual(cfg.rest_phase, PHASE_TP)

    def test_it_stays_configurable_for_large_prompt_traffic(self):
        from sglang.srt.managers.phase_policy import PHASE_PP, REST_PREFILL

        cfg = PhasePolicyConfig(enabled=True, flip_tokens=7004, rest_state=REST_PREFILL)
        self.assertEqual(cfg.rest_phase, PHASE_PP)
