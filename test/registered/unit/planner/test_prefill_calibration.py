"""Calibration of the prefill model against measurement (task #230).

``PerfCostModel.prefill_time_model`` used to book ONLY shard-proportional
time: per-token GEMM FLOPs over the probe rate, plus the per-layer
all-reduce. The measured #216 campaign showed a step that does not work that
way -- the boots ran with ``prefill.backend='disabled'`` (no captured prefill
graph), so every layer's kernels are launched eagerly and that overhead, plus
the non-GEMM ops, is split-INVARIANT. Predicted concentration gains were
inflated by the missing term:

    vector    predicted (shipped)    measured slope gain
    3,1,1          +9.2 %                 +6.4 %
    4,1,1         +15.1 %                 +9.0 %
    6,1,1         +21.6 %                +13.0 %

an over-prediction of x1.4-1.7 with the current probe inputs (campaign-time
inputs read 23.7 % vs 13.0 %, x1.8). The corrected model charges
``_PREDICT_PREFILL_INVARIANT_FRACTION`` (0.35, fitted on these three points)
of the BASE plan's step as a candidate-independent term: ranking is
untouched, reported gains land within 0.6 points of measurement.

The measured slope gains: 6,1,1 is the directly measured +13.0 % (prefill
slope, unique random input ids, ``#cached-token: 0`` proven). 3,1,1 / 4,1,1
follow from the same campaign's measured per-prompt-token savings (0.0473 /
0.0649 ms) against the base slope those two numbers pin (0.0906 ms saved =
13.0 % => base slope 0.7875 ms/token): +6.4 % / +9.0 %.

These tests also pin the refit seam (task #230): the four fitted/assumed
scalars are parameters (``PerfCalibration`` / ``SGLANG_PERF_*`` env), so a
foreign-hardware refit is a value change, not a code edit.
"""

import os
import unittest
from unittest import mock

from sglang.srt import uneven_perf
from sglang.srt.environ import envs
from sglang.srt.uneven_perf import PerfCalibration, PerfCostModel, PlanInputs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_CACHE = os.environ.get("HTSGLANG_TEST_MODEL_DIR", "")
_MODEL = os.path.join(_CACHE, "Qwen3.6-27B-FP8") if _CACHE else ""

#: Probed GEMM rate of the reference rig, rank order 5090, 3080, 3080
#: (cached stage-0 profile, the same inputs apply_auto_performance feeds).
_GEMM = [233.91, 63.17, 61.24]
#: Narrowest pairwise link of the same rig (GB/s).
_MIN_LINK = 5.11
#: The #216 follow-up campaign's base plan and budgets.
_BASE_PLAN = [63, 37, 36]
_BUDGETS = [28447, 16320, 16320]
#: Probed peak membw + decode GEMV of the same rig (for the decode-side
#: refit-seam tests; identical to test_decode_knee_calibration's fixture).
_MEMBW = [1664.1, 718.2, 718.2]
_MEMBW_GEMV = [1532.3, 717.4, 717.4]

#: Measured prefill SLOPE gains (see module docstring for the derivation of
#: the first two from the measured per-token savings).
_MEASURED_PREFILL_GAIN_PCT = {(3, 1, 1): 6.4, (4, 1, 1): 9.0, (6, 1, 1): 13.0}


def _model(calibration=None):
    pi = PlanInputs(
        tp_size=3,
        model_path=_MODEL,
        kv_cache_dtype="fp8_e5m2",
        speculative_algorithm="NEXTN",
        speculative_num_draft_tokens=4,
        rank_gpu_id=[0, 1, 2],
        effective_vram_mib=list(_BUDGETS),
        rank_tp_ratio=list(_BASE_PLAN),
    )
    return PerfCostModel(
        pi, list(_BASE_PLAN), list(_BUDGETS), calibration=calibration
    )


def _gain_pct(m, vec):
    t_base = m.prefill_time_model(list(_BASE_PLAN), _GEMM, _MIN_LINK)
    t_cand = m.prefill_time_model(list(vec), _GEMM, _MIN_LINK)
    return (t_base / t_cand - 1.0) * 100.0


@unittest.skipUnless(
    _MODEL and os.path.isdir(_MODEL),
    "HTSGLANG_TEST_MODEL_DIR/Qwen3.6-27B-FP8 not present",
)
class TestPrefillCalibration(CustomTestCase):
    def test_predicted_gain_tracks_the_measured_slope_gain(self):
        """Within 1.6 points of the three measured vectors. The shipped
        shard-proportional-only model missed them by +2.8 to +8.6 points --
        always high, because the part of the step that does not move with the
        split was not in the model."""
        m = _model()
        for vec, measured in _MEASURED_PREFILL_GAIN_PCT.items():
            with self.subTest(vec=vec):
                self.assertAlmostEqual(_gain_pct(m, vec), measured, delta=1.6)

    def test_the_invariant_share_is_what_closes_the_gap(self):
        """With the fraction forced to 0 the model reverts to the pure
        shard-proportional reading and over-predicts every point again."""
        m = _model()
        with mock.patch.object(
            uneven_perf, "_PREDICT_PREFILL_INVARIANT_FRACTION", 0.0
        ):
            for vec, measured in _MEASURED_PREFILL_GAIN_PCT.items():
                with self.subTest(vec=vec):
                    self.assertGreater(
                        _gain_pct(m, vec),
                        measured + 1.6,
                        "the uncorrected model no longer over-predicts; the "
                        "invariant fraction would then be double-charging",
                    )

    def test_candidate_ranking_is_untouched(self):
        """The invariant term is constant across candidates, so it deflates
        the reported gains without moving the optimizer's choice."""
        cands = [(3, 1, 1), (4, 1, 1), (6, 1, 1), (2, 1, 1), (5, 1, 1)]
        m_fit = _model()
        m_raw = _model(calibration=PerfCalibration(prefill_invariant_fraction=0.0))

        def order(m):
            return sorted(
                cands,
                key=lambda v: m.prefill_time_model(list(v), _GEMM, _MIN_LINK),
            )

        self.assertEqual(order(m_fit), order(m_raw))

    def test_gain_sign_is_preserved(self):
        """A candidate faster than base stays a gain, one slower stays a
        loss: the selection gate (gain > 0) sees the same sign either way."""
        m_fit = _model()
        m_raw = _model(calibration=PerfCalibration(prefill_invariant_fraction=0.0))
        for vec in [(3, 1, 1), (6, 1, 1), (1, 3, 3)]:
            with self.subTest(vec=vec):
                g_fit, g_raw = _gain_pct(m_fit, vec), _gain_pct(m_raw, vec)
                self.assertEqual(g_fit > 0, g_raw > 0)
                self.assertEqual(g_fit < 0, g_raw < 0)


@unittest.skipUnless(
    _MODEL and os.path.isdir(_MODEL),
    "HTSGLANG_TEST_MODEL_DIR/Qwen3.6-27B-FP8 not present",
)
class TestRefitSeam(CustomTestCase):
    """The fitted/assumed scalars are parameters, not buried constants.

    MEASURED per rig (probe): GEMM, streaming membw, decode GEMV, links.
    FITTED/ASSUMED (reference rig only): the two exponents, the decode
    non-weight fraction, the prefill invariant fraction. A foreign-hardware
    refit sets a ``PerfCalibration`` or an ``SGLANG_PERF_*`` env var.
    """

    def test_explicit_calibration_object_wins(self):
        raw = _model(calibration=PerfCalibration(prefill_invariant_fraction=0.0))
        fit = _model(calibration=PerfCalibration(prefill_invariant_fraction=0.35))
        self.assertGreater(
            _gain_pct(raw, (6, 1, 1)), _gain_pct(fit, (6, 1, 1))
        )

    def test_prefill_fraction_env_override(self):
        with envs.SGLANG_PERF_PREFILL_INVARIANT_FRACTION.override(0.0):
            m_env = _model()
            self.assertEqual(
                m_env.calibration.overridden_fields(),
                ["prefill_invariant_fraction"],
            )
            g_env = _gain_pct(m_env, (6, 1, 1))
        g_raw = _gain_pct(
            _model(calibration=PerfCalibration(prefill_invariant_fraction=0.0)),
            (6, 1, 1),
        )
        self.assertAlmostEqual(g_env, g_raw, delta=1e-9)

    def test_decode_gemv_exponent_env_override(self):
        """Same behavior the constant-patch test pins, through the seam: at
        exponent 1.0 the unexponentiated GEMV divisor flips the predicted
        decode cost to a gain."""
        with envs.SGLANG_PERF_DECODE_GEMV_RESIDUAL_EXP.override(1.0):
            m = _model()
            _, beta, _ = m.decode_bw_basis(_MEMBW, _MEMBW_GEMV)
            self.assertEqual(beta, 1.0)
            self.assertLess(
                m.decode_cost_percent([6, 1, 1], _MEMBW, _MEMBW_GEMV), 0.0
            )

    def test_decode_peak_exponent_env_override(self):
        with envs.SGLANG_PERF_DECODE_PEAK_COMPRESSION_EXP.override(1.0):
            m = _model()
            _, beta, basis = m.decode_bw_basis(_MEMBW, None)
            self.assertEqual(beta, 1.0)
            self.assertIn("streaming peak", basis)

    def test_defaults_are_the_shipped_reference_fit(self):
        cal = _model().calibration
        self.assertEqual(cal.overridden_fields(), [])
        self.assertEqual(cal.gemv_residual, uneven_perf._PREDICT_DECODE_GEMV_RESIDUAL)
        self.assertEqual(
            cal.peak_compression, uneven_perf._PREDICT_DECODE_BW_COMPRESSION
        )
        self.assertEqual(
            cal.nonweight_fraction,
            uneven_perf._PREDICT_DECODE_NONWEIGHT_FRACTION,
        )
        self.assertEqual(
            cal.prefill_invariant,
            uneven_perf._PREDICT_PREFILL_INVARIANT_FRACTION,
        )

    def test_a_bad_refit_fraction_is_clamped_not_divided_by_zero(self):
        cal = PerfCalibration(prefill_invariant_fraction=1.0)
        self.assertLessEqual(cal.prefill_invariant, 0.95)
        m = _model(calibration=cal)
        self.assertTrue(
            m.prefill_time_model(list(_BASE_PLAN), _GEMM, _MIN_LINK) > 0
        )


if __name__ == "__main__":
    unittest.main()
