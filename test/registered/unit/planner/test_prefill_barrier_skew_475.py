"""The prefill lockstep max is taken per barrier, not once per step (#475).

The defect. ``PerfCostModel._prefill_sharded_time`` priced a prefill round as
``max_rank sum_family`` GEMM time plus one all-reduce term. Its own comment
says the round contains "two all-reduces of H bf16 per layer" -- so the group
does not synchronise once at the end of the step, it synchronises after the
attention block and again after the MLP block of every layer. The lockstep max
therefore applies PER BARRIER. By Jensen ``sum max >= max sum``, with equality
exactly when one rank is slowest in every family, so the old form was a lower
bound that is tight on a rig whose weak card is weak everywhere and loose
precisely where ``--rank-perf-tune phase-prefill`` operates: moving the MLP
family onto the rank that is NOT the slow rank for attention.

What it cost. On the reference rig's INT8-W8A8 checkpoint the solver chose
``8,1,1``, predicted +8.4 % prefill, and four measured points came back the
other way. The CollectiveClock lines of the two boots say where the win went
(per 1000 prompt tokens, ``docs/dev/NOTE_475_phase_prefill_prediction.md``):

    layout                       critical-rank compute   collective
    VRAM-auto split (#424)              117.9 ms          409.5 ms
    solved 8,1,1     (#433)              80.7 ms          437.4 ms

The concentration did what it was asked -- the critical rank's own compute
fell 31.6 % -- and the round got no shorter, because the collective share rose
by 27.9 ms/1000 tokens. This arithmetic predicts 27.6 ms for that vector with
no fitted parameter, which is the anchor below.

Same rig, same day, FP8 checkpoint: the 3080s run fp8 through weight-only
Marlin at ~1/10 the 5090's native rate, so they pace every barrier at every
candidate vector, the skew is exactly zero, and the same concentration
measured +15.2 % on the prefill window. Both facts have to come out of one
model, and the per-barrier max is what makes them.
"""

import os
import unittest

from sglang.srt.uneven_perf import PerfCostModel, PlanInputs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_CACHE = os.environ.get("HTSGLANG_TEST_MODEL_DIR", "")
_INT8 = os.path.join(_CACHE, "Qwen3.6-27B-INT8-W8A8") if _CACHE else ""
_FP8 = os.path.join(_CACHE, "Qwen3.6-27B-FP8") if _CACHE else ""

#: Both boots' own derived budgets, and the base plan they imply
#: (``2026-08-02_433_int8_prefill/plan_int8_prefill_solved.txt``: "derived
#: memory budgets [28447, 16320, 16320] MiB").
_BUDGETS = [28447, 16320, 16320]
#: The rates the same plan log prints for the INT8-W8A8 checkpoint: both
#: 3080s on the NATIVE int8 lane, 3.7:1 against the 5090.
_GEMM_INT8 = [681.4, 187.6, 183.8]
#: ... and the FP8 checkpoint's, from the FP8 boot of the neighbouring
#: battery (``2026-08-02_424_phase_record_bench/raw/server_fp8_decode.log``):
#: the 3080s on fp8 Marlin, 9.8:1.
_GEMM_FP8 = [563.1, 57.6, 60.8]
#: Narrowest measured pairwise link on that rig (GB/s), same logs.
_MIN_LINK = 5.1

#: Measured growth of the prefill window's collective share between the two
#: INT8 boots, in seconds per token. #424 int8_decode 409.5 ms and #433
#: int8_prefill_solved 437.4 ms per 1000 prompt tokens, aggregated over the
#: 52 and 45 full 2048-token prefill batches of the two s=1 probe windows.
#: This is the ONLY variable that changed between those boots: same commit
#: family, same checkpoint, same probes, same barlink BAR1 transport, same
#: DCP token vector [31, 17, 16].
_MEASURED_SKEW_S_PER_TOKEN = (437.4 - 409.5) * 1e-6


def _model(model_path):
    pi = PlanInputs(
        tp_size=3,
        model_path=model_path,
        kv_cache_dtype="fp8_e4m3",
        speculative_algorithm="NEXTN",
        speculative_num_draft_tokens=4,
        rank_gpu_id=[0, 1, 2],
        effective_vram_mib=list(_BUDGETS),
        rank_tp_ratio=list(_BUDGETS),
    )
    return PerfCostModel(pi, list(_BUDGETS), list(_BUDGETS))


def _old_lockstep(m, vec, gemm):
    """The pre-#475 term: one max at the end of the step."""
    return max(m.per_rank_prefill_compute_times(list(vec), gemm), default=0.0)


@unittest.skipUnless(
    _INT8 and os.path.isdir(_INT8) and _FP8 and os.path.isdir(_FP8),
    "HTSGLANG_TEST_MODEL_DIR/Qwen3.6-27B-{INT8-W8A8,FP8} not present",
)
class TestBarrierSkewIsTheMissingTerm(CustomTestCase):
    def test_the_skew_reproduces_the_measured_collective_growth(self):
        """The anchor: 8,1,1 on INT8, predicted against measured.

        FALSIFIER: the pre-#475 term predicts exactly 0 s of extra lockstep
        time for this vector, against 27.9 ms per 1000 tokens on the card.
        """
        m = _model(_INT8)
        skew_cand = m.prefill_barrier_skew([8, 1, 1], _GEMM_INT8)
        skew_base = m.prefill_barrier_skew(list(_BUDGETS), _GEMM_INT8)
        predicted = skew_cand - skew_base
        self.assertAlmostEqual(
            predicted, _MEASURED_SKEW_S_PER_TOKEN, delta=5e-6,
            msg=f"predicted {predicted * 1e6:.1f} us/token against a measured "
                f"{_MEASURED_SKEW_S_PER_TOKEN * 1e6:.1f}",
        )
        # The falsifier half, as a claim about what the two models REPORT:
        # the old term makes 8,1,1 a reportable win (over the 3.5 % A-vs-A
        # prefill floor the #433 boot measured on itself), the new one does
        # not. This is the assertion that fails on the pre-#475 tree.
        base_old = _old_lockstep(m, list(_BUDGETS), _GEMM_INT8)
        cand_old = _old_lockstep(m, [8, 1, 1], _GEMM_INT8)
        inv = m.calibration.prefill_invariant
        t_ar = self._all_reduce_seconds(m)
        k = inv / (1.0 - inv)
        gain_old = (
            (base_old + t_ar) * (1.0 + k) / ((cand_old + t_ar) + k * (base_old + t_ar))
            - 1.0
        ) * 100.0
        self.assertGreater(gain_old, 3.5, "the pre-#475 claim was reportable")
        base_new = m.prefill_time_model(list(_BUDGETS), _GEMM_INT8, _MIN_LINK)
        cand_new = m.prefill_time_model([8, 1, 1], _GEMM_INT8, _MIN_LINK)
        self.assertLess((base_new / cand_new - 1.0) * 100.0, 3.5)

    @staticmethod
    def _all_reduce_seconds(m):
        """The model's own comm term, per token: two all-reduces of H bf16
        per layer over the narrowest link, ring factor 2(N-1)/N."""
        return m.n_layers * 2 * m.hidden * 2 * 2 * 2 / 3 / (_MIN_LINK * 1e9)

    def test_the_int8_prediction_lands_in_the_measured_band(self):
        """The #433 arm measured +3.3 % prefill against its own 3.5 % A-vs-A
        floor, i.e. "not resolvable"; the GPU window read -1.8 % to +1.8 %
        across the two INT8 concentration arms. The pre-#475 model reported
        +6.2 % for 8,1,1 and +5.8 % for 10,1,1 -- reportable wins that were
        never measurable. Both must now come out small."""
        m = _model(_INT8)
        base = m.prefill_time_model(list(_BUDGETS), _GEMM_INT8, _MIN_LINK)
        for vec in ([8, 1, 1], [10, 1, 1]):
            with self.subTest(vec=vec):
                cand = m.prefill_time_model(list(vec), _GEMM_INT8, _MIN_LINK)
                gain = (base / cand - 1.0) * 100.0
                self.assertLess(gain, 3.5, "above the boot's own prefill floor")
                self.assertGreater(gain, 0.0)

    def test_fp8_is_untouched_because_one_rank_paces_every_barrier(self):
        """Generality guard, and the reason the #216/#230 calibration anchor
        did not have to move: on the fp8 Marlin lanes the two 3080s are the
        slowest rank in every family at every candidate, so ``sum max`` and
        ``max sum`` are the same number and #475 is a no-op. The measured
        +15.2 % prefill window of that arm therefore keeps its prediction."""
        m = _model(_FP8)
        for vec in ([10, 1, 1], [8, 1, 1], [6, 1, 1], [4, 1, 1], [3, 1, 1]):
            with self.subTest(vec=vec):
                self.assertEqual(m.prefill_barrier_skew(list(vec), _GEMM_FP8), 0.0)
                self.assertEqual(
                    m.prefill_lockstep_compute_time(list(vec), _GEMM_FP8),
                    _old_lockstep(m, vec, _GEMM_FP8),
                )

    def test_a_symmetric_rig_has_no_skew_at_any_vector(self):
        """Nothing here is rig-fitted: with equal cards every family's max is
        the same rank's, so the term vanishes whatever the split."""
        m = _model(_INT8)
        equal = [200.0, 200.0, 200.0]
        for vec in ([1, 1, 1], [8, 1, 1], [16, 2, 3], list(_BUDGETS)):
            with self.subTest(vec=vec):
                self.assertAlmostEqual(
                    m.prefill_barrier_skew(list(vec), equal), 0.0, places=12
                )

    def test_the_skew_is_never_negative(self):
        """Jensen, as an invariant rather than as a comment."""
        m = _model(_INT8)
        for vec in ([1, 1, 1], [2, 1, 1], [4, 1, 1], [8, 1, 1], [16, 1, 1],
                    [1, 4, 1], [1, 1, 9], [16, 2, 3], list(_BUDGETS)):
            for gemm in (_GEMM_INT8, _GEMM_FP8):
                with self.subTest(vec=vec, gemm=gemm[0]):
                    self.assertGreaterEqual(
                        m.prefill_barrier_skew(list(vec), gemm), -1e-15
                    )

    def test_the_plan_log_states_the_skew_it_charged(self):
        """A discount nobody can see is a discount nobody can refute: the
        per-candidate line names the skew as a share of the base step."""
        from sglang.srt.uneven_perf import PerfCostModel as _P

        m = _model(_INT8)
        skew = m.prefill_barrier_skew([10, 1, 1], _GEMM_INT8)
        base = m.prefill_time_model(list(_BUDGETS), _GEMM_INT8, _MIN_LINK)
        self.assertGreater(skew / base * 100.0, 1.0)
        # and the two APIs agree by construction
        self.assertAlmostEqual(
            _P.prefill_lockstep_compute_time(m, [10, 1, 1], _GEMM_INT8),
            _old_lockstep(m, [10, 1, 1], _GEMM_INT8) + skew,
            places=15,
        )


if __name__ == "__main__":
    unittest.main()
