"""The prefill phase is cut per FAMILY, not per MLP vector (#485 slice 1).

The law this implements (CLAUDE.md, "PER-FAMILY x PER-PHASE OPTIMA"): every
weight family has its own optimum per phase, so a single-family arm is a
diagnostic, never a phase verdict.

Why the machinery had to change. #475 replaced the prefill compute term with
``sum_family max_rank`` -- the lockstep max applies PER BARRIER, because a
prefill step is two all-reduces per layer. That form is SEPARABLE over
families: each family's contribution is minimized independently by equalizing
its own per-rank time, and there is nothing to gain by compensating one
family's imbalance with another family's vector. Compensating is exactly what
a single-family solve is forced to do, and it is what manufactures the
barrier skew #475 measured at 27.9 ms per 1000 prompt tokens.

Before this change the attention and GDN families followed the base plan and
nothing else could be asked of them: ``_shard_fractions`` read
``self.base_plan`` unconditionally, so a solver that wanted to price a
different attention split had no way to express one. #299 concluded the
attention lever was worth 0.01 % unconstrained -- under a cost model in which
aligning two families' pacers is worth exactly zero BY CONSTRUCTION, because
it took one max at the end of the step. That verdict is not transferable to
the per-barrier model and is re-derived here.

What must not move: every measured point. The four 2026-08-02 arms of #475
carry no attention vector, so the joint machinery must return their numbers
bit-for-bit.
"""

import os
import unittest

from sglang.srt.uneven_perf import (
    PerfCostModel,
    PlanInputs,
    _attn_candidates,
    _attn_lane_bracket,
    _cand_label,
    _cand_vectors,
    _mlp_candidates,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

_CACHE = os.environ.get("HTSGLANG_TEST_MODEL_DIR", "")
_INT8 = os.path.join(_CACHE, "Qwen3.6-27B-INT8-W8A8") if _CACHE else ""
_FP8 = os.path.join(_CACHE, "Qwen3.6-27B-FP8") if _CACHE else ""

#: The #475 fixtures, unchanged: both boots' derived budgets (which are also
#: the base plan) and the per-rank GEMM lane rates their plan logs print.
_BUDGETS = [28447, 16320, 16320]
_GEMM_INT8 = [681.4, 187.6, 183.8]
_GEMM_FP8 = [563.1, 57.6, 60.8]
_MIN_LINK = 5.1
#: Measured collective growth of the #424 -> #433 INT8 pair, s/token.
_MEASURED_SKEW_S_PER_TOKEN = (437.4 - 409.5) * 1e-6
#: Measured GEMV rates of the same three cards (#231 probe group "membw").
_GEMV = [900.0, 420.0, 420.0]


def _model(model_path, base_plan=None):
    plan = list(base_plan or _BUDGETS)
    pi = PlanInputs(
        tp_size=3,
        model_path=model_path,
        kv_cache_dtype="fp8_e4m3",
        speculative_algorithm="NEXTN",
        speculative_num_draft_tokens=4,
        rank_gpu_id=[0, 1, 2],
        effective_vram_mib=list(_BUDGETS),
        rank_tp_ratio=plan,
    )
    return PerfCostModel(pi, plan, list(_BUDGETS))


@unittest.skipUnless(
    _INT8 and os.path.isdir(_INT8) and _FP8 and os.path.isdir(_FP8),
    "HTSGLANG_TEST_MODEL_DIR/Qwen3.6-27B-{INT8-W8A8,FP8} not present",
)
class TestTheJointCutDoesNotMoveTheMeasuredPoints(CustomTestCase):
    """Section 1 of the backtest, as assertions."""

    def test_the_475_anchor_is_unmoved(self):
        """27.6 us/token predicted against 27.9 measured, with the joint
        machinery in the call path and no attention vector passed."""
        m = _model(_INT8)
        predicted = m.prefill_barrier_skew(
            [8, 1, 1], _GEMM_INT8
        ) - m.prefill_barrier_skew(list(_BUDGETS), _GEMM_INT8)
        self.assertAlmostEqual(
            predicted, _MEASURED_SKEW_S_PER_TOKEN, delta=5e-6,
            msg=f"predicted {predicted * 1e6:.1f} us/token",
        )

    def test_passing_no_attention_vector_is_the_pre_485_arithmetic(self):
        """Not "close": the SAME float. An explicit ``None`` and an omitted
        argument must take the identical branch, or the four measured arms
        are being re-priced by a term that was not there when they were
        measured."""
        for path, gemm in ((_INT8, _GEMM_INT8), (_FP8, _GEMM_FP8)):
            m = _model(path)
            for vec in ([1, 1, 1], [4, 1, 1], [8, 1, 1], [10, 1, 1], _BUDGETS):
                with self.subTest(path=os.path.basename(path), vec=vec):
                    self.assertEqual(
                        m.prefill_time_model(list(vec), gemm, _MIN_LINK),
                        m.prefill_time_model(
                            list(vec), gemm, _MIN_LINK, None, None
                        ),
                    )
                    self.assertEqual(
                        m.prefill_lockstep_compute_time(list(vec), gemm),
                        m.prefill_lockstep_compute_time(
                            list(vec), gemm, None, None
                        ),
                    )
                    self.assertEqual(
                        m.per_rank_weight_bytes(list(vec)),
                        m.per_rank_weight_bytes(list(vec), None),
                    )
                    self.assertEqual(
                        m.predict_capacity(list(vec))["ctx"],
                        m.predict_capacity(list(vec), None)["ctx"],
                    )

    def test_fp8_still_has_zero_skew_without_an_attention_vector(self):
        """The #216/#230 calibration anchor's guard, kept."""
        m = _model(_FP8)
        for vec in ([10, 1, 1], [8, 1, 1], [6, 1, 1], [4, 1, 1], [3, 1, 1]):
            with self.subTest(vec=vec):
                self.assertEqual(m.prefill_barrier_skew(list(vec), _GEMM_FP8), 0.0)


@unittest.skipUnless(
    _INT8 and os.path.isdir(_INT8),
    "HTSGLANG_TEST_MODEL_DIR/Qwen3.6-27B-INT8-W8A8 not present",
)
class TestTheAttentionVectorIsTheBasePlanActuator(CustomTestCase):
    """The attention/GDN vector has no runtime family plan of its own: only
    "mlp" is a named family (``distributed/utils._TP_PARTITION_FAMILIES``),
    so ``--rank-tp-ratio`` is what installs it. The model must therefore
    price an attention vector as exactly that -- a different base plan --
    or the solve would name a layout no flag can produce."""

    def test_an_attention_vector_equals_rebuilding_on_that_base_plan(self):
        m = _model(_INT8)
        for attn in ([3, 1, 1], [10, 3, 3], [1, 1, 1], [1, 2, 2]):
            with self.subTest(attn=attn):
                alt = _model(_INT8, base_plan=attn)
                for shard in ("attn", "gdn", "gdn_base"):
                    self.assertEqual(
                        m._shard_fractions(shard, [8, 1, 1], attn),
                        alt._shard_fractions(shard, [8, 1, 1]),
                        f"{shard} does not follow the attention vector",
                    )
                # ... and the MLP family is NOT touched by it.
                self.assertEqual(
                    m._shard_fractions("mlp", [8, 1, 1], attn),
                    m._shard_fractions("mlp", [8, 1, 1]),
                )

    def test_the_ssm_pool_and_the_weight_bytes_follow_it(self):
        """ANALYSE_299's constraint, as arithmetic: the GDN state pool sticks
        to the rank that owns the units, so a joint cut that concentrates GDN
        also concentrates the pool -- and the capacity gate has to see that,
        or the solve prices a layout that does not boot."""
        m = _model(_INT8)
        base_pool = m.mamba_pool_bytes_for(None)
        cand_pool = m.mamba_pool_bytes_for([3, 1, 1])
        self.assertEqual(m.gdn_unit_partition(), [8, 4, 4])
        self.assertEqual(m.gdn_unit_partition([3, 1, 1]), [10, 3, 3])
        self.assertGreater(cand_pool[0], base_pool[0])
        self.assertLess(cand_pool[2], base_pool[2])
        base_w = m.per_rank_weight_bytes([8, 1, 1])
        cand_w = m.per_rank_weight_bytes([8, 1, 1], [3, 1, 1])
        self.assertGreater(cand_w[0], base_w[0])
        # Re-partitioning conserves the total: nothing is created or lost.
        self.assertAlmostEqual(sum(cand_w) / sum(base_w), 1.0, places=9)


@unittest.skipUnless(
    _INT8 and os.path.isdir(_INT8) and _FP8 and os.path.isdir(_FP8),
    "HTSGLANG_TEST_MODEL_DIR/Qwen3.6-27B-{INT8-W8A8,FP8} not present",
)
class TestSeparabilityAndTheCandidateSpace(CustomTestCase):
    def test_the_candidates_keep_one_unit_per_rank_on_both_grids(self):
        """#62/#116: every rank keeps at least one unit on the kv-head grid
        AND on the GDN k-head grid. Delegated to ``partition_units``, asserted
        here because a candidate that violates it is not a layout, it is a
        crash at layer construction."""
        for path, gemm in ((_INT8, _GEMM_INT8), (_FP8, _GEMM_FP8)):
            m = _model(path)
            for cand in _attn_candidates(m, gemm, _BUDGETS):
                with self.subTest(path=os.path.basename(path), cand=cand):
                    gdn = m.gdn_unit_partition(list(cand))
                    self.assertEqual(sum(gdn), m.gdn_units)
                    self.assertTrue(all(u >= 1 for u in gdn), gdn)
                    attn = m._shard_fractions("attn", _BUDGETS, list(cand))
                    self.assertTrue(all(f > 0.0 for f in attn), attn)
                    self.assertAlmostEqual(sum(attn), 1.0, places=9)

    def test_the_candidate_set_is_deduplicated_on_the_materialized_layout(self):
        """The attention grid on this checkpoint is 4 kv-head units wide, so
        most of the ladder collapses to the same partition there and only the
        16-unit GDN grid resolves it. Two wish vectors that materialize
        identically are one candidate."""
        m = _model(_INT8)
        seen = set()
        for cand in _attn_candidates(m, _GEMM_INT8, _BUDGETS):
            key = tuple(m.gdn_unit_partition(list(cand)))
            self.assertNotIn(key, seen, f"{cand} duplicates {key}")
            seen.add(key)
        self.assertNotIn(tuple(m.gdn_unit_partition()), seen, "base is excluded")

    def test_balancing_a_family_on_its_own_lane_shrinks_that_barrier(self):
        """Separability, as the claim the whole slice rests on: under
        ``sum_family max_rank`` the GDN barrier is minimized by GDN's own
        rate-proportional split, INDEPENDENTLY of what the MLP vector does.
        The same assertion holds at two very different MLP vectors precisely
        because the families do not interact through the max."""
        m = _model(_INT8)
        rate_prop = [max(1, round(16 * s / max(_GEMM_INT8))) for s in _GEMM_INT8]
        for mlp in ([8, 1, 1], [4, 1, 1], _BUDGETS):
            with self.subTest(mlp=mlp):
                base = m.per_family_prefill_compute_times(list(mlp), _GEMM_INT8)
                cand = m.per_family_prefill_compute_times(
                    list(mlp), _GEMM_INT8, None, rate_prop
                )
                self.assertLess(max(cand["gdn"]), max(base["gdn"]))
                # The MLP barrier is untouched by the attention vector: that
                # independence IS the separability.
                self.assertEqual(cand["mlp"], base["mlp"])

    def test_the_ladder_proposes_nothing_on_a_symmetric_rate_profile(self):
        """Generality: nothing here is rig-fitted. The ladder is built from
        the profile's own per-rank rates, so equal rates produce only the
        uniform vector, which is excluded -- the solve proposes no attention
        candidate at all rather than proposing one and rejecting it."""
        m = _model(_INT8)
        self.assertEqual(_attn_candidates(m, [200.0, 200.0, 200.0], _BUDGETS), [])
        self.assertEqual(_attn_candidates(m, [1.0, 1.0, 1.0], [4, 4, 4]), [])

    def test_the_lever_is_a_property_of_base_plan_vs_lane_rates(self):
        """And NOT of card heterogeneity, which is the generalization #299's
        rig-specific verdict does not survive.

        Equal cards behind an UNEQUAL base plan (the VRAM-auto split of a rig
        whose cards differ in memory but not in speed) leave the attention
        family split 0.50/0.25/0.25 across three equally fast ranks: rank 0
        paces a barrier it has no reason to pace. Rebalancing it to even is a
        real gain that a single-family MLP solve cannot express, because
        moving MLP units cannot change who paces the ATTENTION barrier.

        The mirror case is the guard above: equal rates AND an equal base
        plan leave nothing to win.
        """
        m = _model(_INT8)
        equal = [200.0, 200.0, 200.0]
        base = m.prefill_time_model(list(_BUDGETS), equal, _MIN_LINK)
        even = m.prefill_time_model(
            list(_BUDGETS), equal, _MIN_LINK, None, [1, 1, 1]
        )
        self.assertGreater(base / even - 1.0, 0.01)
        sym = _model(_INT8, base_plan=[1, 1, 1])
        self.assertAlmostEqual(
            sym.prefill_time_model([1, 1, 1], equal, _MIN_LINK)
            / sym.prefill_time_model([1, 1, 1], equal, _MIN_LINK, None, [1, 1, 1]),
            1.0,
            places=12,
        )


@unittest.skipUnless(
    _INT8 and os.path.isdir(_INT8) and _FP8 and os.path.isdir(_FP8),
    "HTSGLANG_TEST_MODEL_DIR/Qwen3.6-27B-{INT8-W8A8,FP8} not present",
)
class TestTheFalsifiers(CustomTestCase):
    def test_a_detuned_attention_vector_prices_worse_than_the_aligned_one(self):
        """CAN-FAIL ARM. Pair the optimal MLP vector with the aligned
        attention vector REVERSED -- same grid, same unit count, attention
        mass pushed onto the ranks the lane rates call slowest. If the
        objective is reading the attention family at all, this must price
        strictly worse. An objective that ignored the second half of the pair
        would return the identical number."""
        for path, gemm in ((_INT8, _GEMM_INT8), (_FP8, _GEMM_FP8)):
            m = _model(path)
            mlp = [list(c) for c in _mlp_candidates(m, gemm, _BUDGETS)]
            attn = [list(c) for c in _attn_candidates(m, gemm, _BUDGETS)]
            ref = m.prefill_time_model(list(_BUDGETS), gemm, _MIN_LINK)
            best = max(
                (
                    (ref / m.prefill_time_model(v, gemm, _MIN_LINK, None, a) - 1, v, a)
                    for v in mlp
                    for a in attn
                ),
                key=lambda x: x[0],
            )
            g_ok, v_ok, a_ok = best
            detuned = list(reversed(a_ok))
            g_bad = (
                ref / m.prefill_time_model(v_ok, gemm, _MIN_LINK, None, detuned) - 1
            )
            with self.subTest(path=os.path.basename(path)):
                self.assertLess(
                    g_bad, g_ok,
                    f"aligned {a_ok} {g_ok:+.4f} vs detuned {detuned} "
                    f"{g_bad:+.4f} -- the attention half is not priced",
                )

    def test_the_lane_bracket_varies_only_the_ratio_never_the_mass(self):
        """The bracket's honesty condition. The flash/scan core's per-token
        mass is not modelled (it depends on the context length, which this
        parse-time model does not carry), so the bandwidth endpoint must not
        smuggle a mass in through the back door: at the BASE plan the
        attention and GDN families must cost the same total time under both
        endpoints, and only their inter-rank ratio may differ."""
        for path, gemm in ((_INT8, _GEMM_INT8), (_FP8, _GEMM_FP8)):
            m = _model(path)
            bracket = _attn_lane_bracket(m, None, gemm, _GEMV)
            self.assertIsNotNone(bracket)
            gemm_lane, bw_lane = bracket
            for name, fam in m.families.items():
                if fam.params <= 0:
                    continue
                with self.subTest(path=os.path.basename(path), family=name):
                    fr = m._shard_fractions(fam.shard, list(_BUDGETS))
                    t_gemm = sum(
                        f / r for f, r in zip(fr, gemm_lane[name]) if r > 0
                    )
                    t_bw = sum(f / r for f, r in zip(fr, bw_lane[name]) if r > 0)
                    self.assertAlmostEqual(t_gemm / t_bw, 1.0, places=9)
                    if fam.shard in ("attn", "gdn", "gdn_base"):
                        self.assertNotEqual(gemm_lane[name], bw_lane[name])
                    else:
                        self.assertEqual(gemm_lane[name], bw_lane[name])

    def test_a_bracket_endpoint_can_reverse_the_verdict(self):
        """And it does, which is why the solve reports LANE-SENSITIVE rather
        than a point estimate: on this rig the attention/GDN lane spread is
        3.7:1 in int8 GEMM and 2.1:1 in measured bandwidth, so the two
        endpoints disagree about whether the attention family has a lever at
        all. A bracket whose endpoints always agreed would be decoration."""
        m = _model(_INT8)
        bracket = _attn_lane_bracket(m, None, _GEMM_INT8, _GEMV)
        gemm_lane, bw_lane = bracket
        attn = [3, 1, 1]
        deltas = []
        for rates in (gemm_lane, bw_lane):
            ref = m.prefill_time_model(list(_BUDGETS), _GEMM_INT8, _MIN_LINK, rates)
            joint = (
                ref
                / m.prefill_time_model(
                    [4, 1, 1], _GEMM_INT8, _MIN_LINK, rates, attn
                )
                - 1.0
            )
            single = (
                ref
                / m.prefill_time_model([4, 1, 1], _GEMM_INT8, _MIN_LINK, rates)
                - 1.0
            )
            deltas.append(joint - single)
        self.assertGreater(deltas[0], 0.0, "GEMM endpoint: the pair should win")
        self.assertLess(deltas[1], deltas[0], "the endpoints must disagree")


class TestTheCandidateLabelReadsBothShapes(CustomTestCase):
    """No checkpoint needed: pure plumbing, and the plumbing is what the
    refusal path (``_no_lever_lines``) formats a candidate with."""

    def test_plain_vectors_and_pairs_both_format(self):
        self.assertEqual(_cand_vectors((8, 1, 1)), ([8, 1, 1], None))
        self.assertEqual(
            _cand_vectors(((8, 1, 1), (3, 1, 1))), ([8, 1, 1], [3, 1, 1])
        )
        self.assertEqual(_cand_vectors(((8, 1, 1), None)), ([8, 1, 1], None))
        self.assertEqual(_cand_label((8, 1, 1)), "8,1,1")
        self.assertEqual(
            _cand_label(((8, 1, 1), (3, 1, 1))), "8,1,1 + attn/GDN 3,1,1"
        )


if __name__ == "__main__":
    unittest.main()
