"""Calibration of the decode-knee guard against measurement (task #216).

The guard in ``PerfCostModel.decode_knee_detail`` decides whether an MLP
concentration vector may be proposed by ``--rank-tp-ratio auto-performance``.
Its shipped model asked a single question:

    is rank r's share of the streamed weight bytes <= its share of the rig's
    PROBED PEAK memory bandwidth?

On the reference rig (RTX 5090 + 2x RTX 3080, Qwen3.6-27B-FP8, uneven TP=3)
that model certified ``6,1,1`` as "knee OK" and its companion round-time proxy
predicted the vector would make decode ~22 % FASTER. An interleaved four-boot
A/B with the KV ownership vector pinned measured the opposite: ``6,1,1`` costs
**+16.5 %** of the decode step (ctx 400: 32.546 -> 37.989 ms; ctx 12000:
33.104 -> 38.551 ms), against a boot-to-boot noise floor of 0.10-0.51 %.

Peak bandwidth is not achieved bandwidth. At batch size 1 a rank's shard is
executed as many small quantized GEMV kernels, and how close that comes to the
card's streaming peak differs per architecture and per quant path -- on this
rig the sm_120 native-FP8 lane and the sm_86 Marlin upconvert lane do not sit
at the same fraction of peak. Treating the 7:3:3 peak ratio as if it were the
ratio of achieved decode bandwidth puts the guard's ceiling for the fast rank
far too high, so it waves through exactly the concentration that makes that
rank the lockstep pacer.

These tests pin both halves of the correction:
  1. the ceiling is taken from EFFECTIVE, not peak, bandwidth, so the
     measured-bad vector is no longer certified;
  2. the cost is QUANTIFIED rather than silently passed -- a rejected or
     accepted candidate alike carries a predicted decode delta whose sign is
     right, because a silent guard is what let this through the first time.
"""

import os
import unittest

from sglang.srt.uneven_perf import PerfCostModel, PlanInputs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_CACHE = os.environ.get("HTSGLANG_TEST_MODEL_DIR", "")
_MODEL = os.path.join(_CACHE, "Qwen3.6-27B-FP8") if _CACHE else ""

#: Probed peak membw of the reference rig, rank order 5090, 3080, 3080.
_MEMBW = [1664.1, 718.2, 718.2]

#: The budget vector the #216 campaign booted under. The guard's verdict
#: depends on it (it sets the base plan the attention/GDN families follow),
#: and THIS vector is the one that reproduces the shipped guard's documented
#: misjudgement: 6,1,1 certified, 8,1,1 and 10,1,1 rejected.
_BUDGETS_216 = [26107, 18280, 18280]

#: The budget vector the follow-up campaign booted under, and therefore the
#: only one whose predicted decode COSTS can be checked against measured
#: numbers. Measured ms per speculative step, natural-text prompts, two
#: context depths, interleaved cold boots, KV ownership vector pinned:
#: base 30.10, 3,1,1 31.90 (+6.0 %), 4,1,1 34.12 (+13.4 %), 6,1,1 34.76
#: (+15.5 %); boot-to-boot floor 0.6-0.8 %.
_BUDGETS_MEASURED = [28447, 16320, 16320]
_MEASURED_COST_PCT = {(3, 1, 1): 6.0, (4, 1, 1): 13.4, (6, 1, 1): 15.5}


def _model(budgets):
    pi = PlanInputs(
        tp_size=3,
        model_path=_MODEL,
        kv_cache_dtype="fp8_e5m2",
        speculative_algorithm="NEXTN",
        speculative_num_draft_tokens=4,
        rank_gpu_id=[0, 1, 2],
        effective_vram_mib=list(budgets),
        rank_tp_ratio=list(budgets),
    )
    return PerfCostModel(pi, list(budgets), list(budgets))


@unittest.skipUnless(
    _MODEL and os.path.isdir(_MODEL),
    "HTSGLANG_TEST_MODEL_DIR/Qwen3.6-27B-FP8 not present",
)
class TestDecodeKneeCalibration(CustomTestCase):
    def test_measured_bad_vector_is_not_certified(self):
        """6,1,1 measured +16.5 % decode; the guard must not call it OK."""
        m = _model(_BUDGETS_216)
        ok, reason = m.decode_knee_detail([6, 1, 1], _MEMBW)
        self.assertFalse(
            ok,
            "6,1,1 is certified knee-OK but measures +16.5 % decode step "
            "on the reference rig",
        )
        self.assertTrue(reason)

    def test_the_still_worse_vectors_stay_rejected(self):
        """The correction must not work by simply rejecting everything --
        the vectors the shipped guard already caught stay caught."""
        m = _model(_BUDGETS_216)
        for vec in ([8, 1, 1], [10, 1, 1]):
            with self.subTest(vec=vec):
                self.assertFalse(m.decode_knee_detail(vec, _MEMBW)[0])

    def test_the_base_plan_itself_is_always_certified(self):
        """The VRAM-auto split is the reference point, never a violation."""
        m = _model(_BUDGETS_216)
        self.assertTrue(m.decode_knee_detail(_BUDGETS_216, _MEMBW)[0])

    def test_decode_cost_is_quantified_with_the_right_sign(self):
        """A guard that only says yes/no is how +16.5 % passed silently.
        Concentrating onto the fast rank must be reported as a COST."""
        m = _model(_BUDGETS_MEASURED)
        for vec in _MEASURED_COST_PCT:
            with self.subTest(vec=vec):
                self.assertGreater(
                    m.decode_cost_percent(list(vec), _MEMBW),
                    0.0,
                    "predicted decode delta is not a cost; the shipped "
                    "peak-bandwidth proxy predicted -10 to -22 % for these "
                    "vectors where +6 to +16 % was measured",
                )

    def test_predicted_cost_tracks_the_measured_cost(self):
        """Within 5 points of the four measured vectors. Loose on purpose:
        the point of the number is to be right about the size of the effect,
        not to pretend the parse-time model is a benchmark."""
        m = _model(_BUDGETS_MEASURED)
        for vec, measured in _MEASURED_COST_PCT.items():
            with self.subTest(vec=vec):
                self.assertAlmostEqual(
                    m.decode_cost_percent(list(vec), _MEMBW),
                    measured,
                    delta=5.0,
                )

    def test_the_base_plan_costs_nothing_against_itself(self):
        m = _model(_BUDGETS_MEASURED)
        self.assertAlmostEqual(
            m.decode_cost_percent(_BUDGETS_MEASURED, _MEMBW), 0.0, delta=1e-6
        )

    def test_cost_is_monotone_in_concentration_past_the_balance_point(self):
        """Past the effective-bandwidth balance point, deeper concentration
        on the fast rank cannot be cheaper. (Below it the curve is U-shaped
        -- the base split may itself be under-loaded on the fast card -- so
        monotonicity is only asserted where it is physically required, and
        where the measurement actually sits.)"""
        m = _model(_BUDGETS_MEASURED)
        costs = [m.decode_cost_percent(list(v), _MEMBW)
                 for v in ([3, 1, 1], [4, 1, 1], [6, 1, 1], [8, 1, 1])]
        self.assertEqual(costs, sorted(costs), f"non-monotone: {costs}")

    def test_effective_bandwidth_is_compressed_against_peak(self):
        """The whole correction in one assertion: the fast card's share of
        effective decode bandwidth is smaller than its share of probed peak,
        which is why its ceiling drops."""
        m = _model(_BUDGETS_MEASURED)
        eff = m.effective_decode_bw(_MEMBW)
        self.assertLess(
            eff[0] / sum(eff),
            _MEMBW[0] / sum(_MEMBW),
            "effective bandwidth share is not compressed against peak",
        )


if __name__ == "__main__":
    unittest.main()
