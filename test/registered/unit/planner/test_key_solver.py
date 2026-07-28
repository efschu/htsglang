"""The key solver (#272): the closed form, the goals, and the regression store.

What is pinned here, in the order the module derives it:

1. THE OPTIMIZER. ``water_fill`` really solves the min-max it claims to
   solve — verified against brute force on the integer grid, not against
   itself — and honours the role bounds that make the 0-100 % continuum
   representable.
2. THE GEOMETRY. A hybrid checkpoint is counted over its FULL-ATTENTION
   layers, and a layer WINDOW by intersection. The #201 slice-2 case (a
   14/10 split holding 3 and 3) is the one that says why.
3. THE REGRESSION STORE — the four measured points the model was built from
   plus the multi-instance trade, re-derived and compared. A model that
   cannot reproduce a measurement it was fitted against is wrong, and these
   are where that would show.
4. THE ADDITIVE MECHANIC. One rule for every family that adds a serving
   source: sum the sources that jointly fit, and refuse to sum the ones that
   do not. Shared weight bytes (rank reuse) count once.

Everything that needs the real Qwen3.6-27B checkpoints or the rig's card
probe skips cleanly when they are absent, so the file is a CPU unit test on
any machine and a rig-anchored regression test on this one.
"""

import json
import math
import os
import random
import tempfile
import unittest
from itertools import product
from unittest import mock

from sglang.srt.planner import key_solver as ks
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=90, suite="base-a-test-cpu")


#: Local model-zoo root; the checkpoint-backed cases skip without it.
_CACHE = os.environ.get(
    "HTSGLANG_TEST_MODEL_DIR", "/spinning/llm_stuff/club-3090/models-cache"
)
_FP8 = os.path.join(_CACHE, "Qwen3.6-27B-FP8")
_Q3 = os.path.join(_CACHE, "Qwen3.6-27B-MTP-Q3_K_M-GGUF", "Qwen3.6-27B-Q3_K_M.gguf")
_SMALL = os.path.join(_CACHE, "Qwen3.5-4B")

#: The reference rig's probe, as the card probe records it. Inlined rather
#: than read from ~/.cache so the regression numbers cannot silently change
#: under the test when somebody re-probes the machine.
_PROBE = {
    "cards": [
        {
            "uuid": "GPU-5090",
            "name": "NVIDIA GeForce RTX 5090",
            "cuda_index": 0,
            "total_mib": 32607,
            "gemm_bf16_tflops": 231.97,
            "gemm_fp8_tflops": 566.88,
            "membw_read_gbs": 1660.4,
            "membw_gemv_gbs": 1533.8,
            "h2d_gbs": 14.41,
            "d2h_gbs": 14.26,
        },
        {
            "uuid": "GPU-3080a",
            "name": "NVIDIA GeForce RTX 3080",
            "cuda_index": 1,
            "total_mib": 20480,
            "gemm_bf16_tflops": 65.57,
            "gemm_fp8_tflops": None,
            "membw_read_gbs": 717.0,
            "membw_gemv_gbs": 717.4,
            "h2d_gbs": 6.47,
            "d2h_gbs": 6.58,
        },
        {
            "uuid": "GPU-3080b",
            "name": "NVIDIA GeForce RTX 3080",
            "cuda_index": 2,
            "total_mib": 20480,
            "gemm_bf16_tflops": 65.59,
            "gemm_fp8_tflops": None,
            "membw_read_gbs": 717.1,
            "membw_gemv_gbs": 717.8,
            "h2d_gbs": 13.4,
            "d2h_gbs": 13.16,
        },
    ],
    "pairs": [
        {
            "src_uuid": a,
            "dst_uuid": b,
            "bandwidth_gbs": bw,
            "latency_us": lat,
            "transport": "host staging (pinned)",
            "peer_access": False,
        }
        for a, b, bw, lat in (
            ("GPU-5090", "GPU-3080a", 4.44, 22.4),
            ("GPU-5090", "GPU-3080b", 6.91, 19.8),
            ("GPU-3080a", "GPU-5090", 4.52, 22.1),
            ("GPU-3080a", "GPU-3080b", 4.41, 21.5),
            ("GPU-3080b", "GPU-5090", 6.88, 19.5),
            ("GPU-3080b", "GPU-3080a", 4.32, 21.6),
        )
    ],
}


def _have(path: str) -> bool:
    return os.path.exists(path)


class _Bf16StateEnv(CustomTestCase):
    """Base class for the cases that reproduce a booted arm.

    Every measured arm below ran with ``SGLANG_MAMBA_SSM_DTYPE=bfloat16``,
    which halves the recurrent-state pool and therefore every capacity number
    in this file. The variable is read from the environment at call time, so
    it has to be set -- but it is scoped to these classes on purpose: setting
    it at module import changed the state geometry for every OTHER planner
    test in the same session (``test_mrr_balance`` pins the fp32 figures and
    went red). A test file that reconfigures its neighbours is a
    test-pollution bug, and this project has paid for that one before.
    """

    _env: "mock._patch_dict"

    @classmethod
    def setUpClass(cls):
        cls._env = mock.patch.dict(os.environ, {"SGLANG_MAMBA_SSM_DTYPE": "bfloat16"})
        cls._env.start()

    @classmethod
    def tearDownClass(cls):
        cls._env.stop()


# ---------------------------------------------------------------------------
# 1. The optimizer
# ---------------------------------------------------------------------------


class TestWaterFill(CustomTestCase):
    """``water_fill`` is the whole optimizer; if it is not the true optimum,
    nothing above it means anything."""

    def test_matches_brute_force_on_the_integer_grid(self):
        # Small enough to enumerate every partition exhaustively, random
        # enough that a lucky coincidence is not what is being tested.
        rng = random.Random(20260728)
        for _ in range(40):
            n = rng.choice((2, 3, 4))
            total = rng.choice((8, 12, 16))
            a = [rng.uniform(0.0, 5.0) for _ in range(n)]
            b = [rng.uniform(0.5, 4.0) for _ in range(n)]
            lo = [0] * n
            hi = [total] * n
            cont = ks.water_fill(
                a, b, float(total), [float(x) for x in lo], [float(x) for x in hi]
            )
            self.assertAlmostEqual(sum(cont), total, places=6)
            vec = ks.project_to_grid(cont, total, lo, hi)
            vec = ks._repair(
                vec,
                lambda u: max(a[i] + b[i] * u[i] for i in range(n)),
                lo,
                hi,
            )
            got = max(a[i] + b[i] * vec[i] for i in range(n))

            best = math.inf
            for cand in product(range(total + 1), repeat=n):
                if sum(cand) != total:
                    continue
                best = min(best, max(a[i] + b[i] * cand[i] for i in range(n)))
            self.assertLessEqual(
                got,
                best + 1e-9,
                f"water_fill+repair gave {got:.6f} where brute force found "
                f"{best:.6f} (a={a}, b={b}, total={total})",
            )

    def test_continuous_optimum_equalizes_the_interior_ranks(self):
        # The defining property of the water level: every rank that is not
        # clamped sits at the SAME cost. If two interior ranks differ, the
        # allocation is not optimal.
        a = [1.0, 4.0, 2.5]
        b = [1.0, 2.0, 0.5]
        u = ks.water_fill(a, b, 20.0, [0.0] * 3, [20.0] * 3)
        levels = [a[i] + b[i] * u[i] for i in range(3)]
        interior = [i for i in range(3) if 1e-9 < u[i] < 20.0 - 1e-9]
        self.assertGreaterEqual(len(interior), 2)
        for i in interior[1:]:
            self.assertAlmostEqual(levels[i], levels[interior[0]], places=6)

    def test_role_bounds_are_honoured(self):
        # A kv_donor takes zero units; the rest of the model still has to be
        # placed. This is the 0 % end of the continuum, and it is a bound,
        # not a special case.
        u = ks.water_fill(
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            30.0,
            [0.0, 0.0, 0.0],
            [30.0, 0.0, 30.0],
        )
        self.assertAlmostEqual(u[1], 0.0, places=9)
        self.assertAlmostEqual(sum(u), 30.0, places=6)

    def test_nesting_bounds_are_a_ceiling_per_shared_card(self):
        # The dual-group case: an inner lane reusing an outer lane's resident
        # weights may hold at most what the outer lane holds, card by card.
        # A card the outer lane does not use is unbounded -- it carries the
        # complement, the only genuinely new bytes in the rig.
        b = ks.nesting_bounds([118, 18], [0, 1, None])
        self.assertEqual(b, [(None, 118), (None, 18), (None, None)])

    def test_nesting_and_role_bounds_intersect(self):
        # A nesting ceiling must never widen a role, and a role must never
        # widen a nesting ceiling: the two are intersected, not overridden.
        lo, hi, _ = ks._role_bounds(
            ["shard", "kv_donor", "shard"], 136, [(None, 100), (None, 50), (10, None)]
        )
        self.assertEqual(hi[0], 100)  # nesting tightens the shard
        self.assertEqual(hi[1], 0)  # role tightens below the nesting ceiling
        self.assertEqual(lo[2], 10)  # an explicit floor is honoured

    def test_nesting_box_is_respected_by_the_optimizer(self):
        u = ks.water_fill(
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            60.0,
            [0.0, 0.0, 0.0],
            [20.0, 10.0, 60.0],
        )
        self.assertLessEqual(u[0], 20.0 + 1e-9)
        self.assertLessEqual(u[1], 10.0 + 1e-9)
        self.assertAlmostEqual(sum(u), 60.0, places=6)

    def test_infeasible_bounds_raise_instead_of_lying(self):
        # Two donors and one shard that cannot hold everything: there is no
        # allocation, and the solver has to say so rather than return one.
        with self.assertRaises(ValueError):
            ks.water_fill([1.0, 1.0], [1.0, 1.0], 30.0, [0.0, 0.0], [10.0, 0.0])

    def test_projection_preserves_the_total_exactly(self):
        rng = random.Random(7)
        for _ in range(50):
            n = rng.choice((2, 3, 5))
            total = rng.randint(5, 200)
            u = [rng.uniform(0, total) for _ in range(n)]
            s = sum(u) or 1.0
            u = [x * total / s for x in u]
            vec = ks.project_to_grid(u, total, [0] * n, [total] * n)
            self.assertEqual(sum(vec), total)
            self.assertTrue(all(v >= 0 for v in vec))


# ---------------------------------------------------------------------------
# 2. Hybrid geometry — regression gate 4
# ---------------------------------------------------------------------------


class TestHybridGeometry(CustomTestCase):
    """Gate 4: ``num_hidden_layers`` sizes a hybrid wrong, in both
    directions at once."""

    def test_full_attention_count_of_qwen36_27b(self):
        layer_types = [
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
            for i in range(64)
        ]
        self.assertEqual(ks.full_attention_layers(layer_types), 16)
        self.assertNotEqual(ks.full_attention_layers(layer_types), 64)

    def test_201_slice2_window_split_is_3_and_3_not_14_and_10(self):
        # The #201 slice-2 finding, verbatim: a 14/10 layer split of a
        # period-4 hybrid puts THREE full-attention layers on each stage.
        # A proportional reading (14:10) would size the two KV pools 40 %
        # apart when they are in fact equal.
        layer_types = [
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
            for i in range(24)
        ]
        self.assertEqual(ks.full_attention_layers(layer_types), 6)
        first = ks.full_attention_layers_in_window(layer_types, 0, 14)
        second = ks.full_attention_layers_in_window(layer_types, 14, 24)
        self.assertEqual((first, second), (3, 3))
        self.assertEqual(first + second, ks.full_attention_layers(layer_types))
        # And the wrong answer the count alone would have given:
        self.assertNotEqual((first, second), (14, 10))

    def test_window_helper_is_total_preserving_at_every_cut(self):
        layer_types = [
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
            for i in range(64)
        ]
        total = ks.full_attention_layers(layer_types)
        for cut in range(65):
            lo = ks.full_attention_layers_in_window(layer_types, 0, cut)
            hi = ks.full_attention_layers_in_window(layer_types, cut, 64)
            self.assertEqual(lo + hi, total, f"cut at {cut}")

    def test_pure_attention_model_reports_none(self):
        # No layer_types -> the caller's num_hidden_layers is already right,
        # and the helper says so instead of inventing a zero.
        self.assertIsNone(ks.full_attention_layers(None))
        self.assertIsNone(ks.full_attention_layers([]))
        self.assertIsNone(ks.full_attention_layers_in_window(None, 0, 10))


# ---------------------------------------------------------------------------
# 3. Rates
# ---------------------------------------------------------------------------


class TestRates(CustomTestCase):
    def test_pair_matrix_takes_the_narrowest_and_the_worst(self):
        r = ks.rates_from_probe(_PROBE, [0, 1, 2])
        self.assertAlmostEqual(r.link_bw_gbs, 4.32, places=2)
        self.assertAlmostEqual(r.link_latency_us, 22.4, places=1)
        self.assertEqual(r.absent, [])

    def test_pairs_outside_the_used_cards_are_ignored(self):
        # Only ranks 0 and 2 talk; the 4.32 GB/s pair between the two 3080s
        # is not on this group's critical path and must not set its floor.
        r = ks.rates_from_probe(_PROBE, [0, 2])
        self.assertAlmostEqual(r.link_bw_gbs, 6.88, places=2)

    def test_single_rank_has_no_link_and_no_absence(self):
        r = ks.rates_from_probe(_PROBE, [0])
        self.assertIsNone(r.link_bw_gbs)
        self.assertEqual(r.absent, [])

    def test_missing_pair_matrix_is_absent_not_invented(self):
        probe = {"cards": _PROBE["cards"], "pairs": []}
        r = ks.rates_from_probe(probe, [0, 1, 2])
        self.assertIsNone(r.link_bw_gbs)
        self.assertTrue(any("pair matrix" in a for a in r.absent))

    def test_unknown_gpu_index_fails_loudly(self):
        with self.assertRaises(ValueError):
            ks.rates_from_probe(_PROBE, [0, 7])

    def test_gemm_dtype_resolution_picks_per_card(self):
        r = ks.rates_from_probe(_PROBE, [0, 1, 2])
        fp8 = r.resolve_gemm_dtype("fp8")
        # The 5090 has an fp8 path; the 3080s do not and fall back to bf16.
        self.assertAlmostEqual(fp8.gemm_tflops[0], 566.88, places=2)
        self.assertAlmostEqual(fp8.gemm_tflops[1], 65.57, places=2)
        bf = r.resolve_gemm_dtype("bf16")
        self.assertAlmostEqual(bf.gemm_tflops[0], 231.97, places=2)

    def test_gguf_is_planned_on_bf16_not_fp8(self):
        # A K-quant is dequantized into a bf16 GEMM. Reading the 5090's fp8
        # rate for it over-states that rank by 2.4x, which is a mixed-rig-
        # only error -- exactly where the planner is used.
        with tempfile.TemporaryDirectory() as d:
            gguf = os.path.join(d, "model.gguf")
            open(gguf, "w").close()
            self.assertEqual(ks.gemm_dtype_for_checkpoint(gguf), "bf16")

    def test_fp8_checkpoint_is_planned_on_fp8(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump(
                    {"quantization_config": {"quant_method": "fp8", "fmt": "e4m3"}},
                    f,
                )
            self.assertEqual(ks.gemm_dtype_for_checkpoint(d), "fp8")

    def test_unreadable_config_falls_back_to_bf16(self):
        self.assertEqual(
            ks.gemm_dtype_for_checkpoint("/nonexistent/checkpoint"), "bf16"
        )


# ---------------------------------------------------------------------------
# 4. The regression store — gates 1, 2 and 3
# ---------------------------------------------------------------------------


@unittest.skipUnless(_have(_FP8), f"needs the 27B-FP8 checkpoint at {_FP8}")
class TestRegressionAnchors(_Bf16StateEnv):
    """The measured points the model was built from, re-derived.

    These are not self-consistency checks: every ``measured`` number below
    comes from a booted arm recorded elsewhere in the project, and the
    tolerances are argued in ``REGRESSION_ANCHORS[*].tolerance_reason``.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rows = {r["key"]: r for r in ks.check_regressions(_FP8, _PROBE)}

    def test_gate1_611_is_net_negative(self):
        # #264: concentrating to 6,1,1 buys +8.2 % prefill and pays -13.7 %
        # decode. The verdict "net negative" is asserted exactly; the
        # magnitudes carry the noise-floor tolerance.
        row = self.rows["264_611_net_negative"]
        self.assertTrue(row["signs_match"], row)
        self.assertGreater(row["predicted_pct"]["enc"], 0.0)
        self.assertLess(row["predicted_pct"]["dec"], 0.0)
        # Net verdict: the decode loss outweighs the prefill gain.
        self.assertGreater(
            abs(row["predicted_pct"]["dec"]),
            row["predicted_pct"]["enc"],
            "6,1,1 must come out NET NEGATIVE; the model made it a win",
        )
        self.assertTrue(row["within_tolerance"], row)

    def test_gate2_252_is_prefill_only(self):
        # The phase-dual falsifier: a real prefill gain in the ~5 % class,
        # and a decode delta the measurement could not resolve above its
        # noise floor. The requirement on decode is that it stays SMALL.
        row = self.rows["phasedual_252_prefill_only"]
        self.assertGreater(row["predicted_pct"]["enc"], 0.0)
        self.assertTrue(
            2.0 <= row["predicted_pct"]["enc"] <= 8.0,
            f"prefill gain {row['predicted_pct']['enc']:.1f} % is outside the "
            "measured 3.4-5.7 % band's neighbourhood",
        )
        self.assertLess(
            abs(row["predicted_pct"]["dec"]),
            6.0,
            "the 2,5,2 decode cost was measured below the noise floor; a "
            "model that predicts a large one has the wrong shape",
        )
        self.assertTrue(row["within_tolerance"], row)

    def test_gate1_is_worse_than_gate2_on_both_axes_of_the_trade(self):
        # The two anchors are the same lever at two strengths. The model has
        # to order them: the extreme vector costs more decode AND (being
        # past the knee) is the one the measurements rejected.
        a = self.rows["264_611_net_negative"]["predicted_pct"]
        b = self.rows["phasedual_252_prefill_only"]["predicted_pct"]
        self.assertLess(a["dec"], b["dec"])
        self.assertGreater(a["enc"], b["enc"])

    def test_gate3_27b_fp8_does_not_boot_solo_on_the_5090(self):
        # Three OOM boots with an identical, context-independent failure:
        # ~28.5 GiB of weights + draft + pools leave no room for KV on the
        # 5090. The solver must call this unbootable, at any key -- TP=1 has
        # only one, so "any" is the whole space.
        from sglang.srt.uneven_perf import PlanInputs

        budget = 32607 - 3000
        inputs = PlanInputs(
            tp_size=1,
            model_path=_FP8,
            kv_cache_dtype="fp8_e4m3",
            speculative_algorithm="NEXTN",
            speculative_num_draft_tokens=4,
            max_running_requests=16,
            rank_gpu_id=[0],
            effective_vram_mib=[budget],
        )
        rates = ks.rates_from_probe(_PROBE, [0])
        model = ks.build_cost_model(inputs, [1], [budget], rates)
        cap = model.capacity([model.units], [0.0])
        self.assertFalse(
            cap["feasible"],
            "27B-FP8 solo on the 5090 must be reported unbootable; the "
            "ledger post is ~28.5 GiB against a ~28.9 GiB budget",
        )
        self.assertGreater(model.weight_bytes([model.units])[0] / 2**30, 27.0)

    def test_gate3_the_same_model_does_boot_at_tp3(self):
        # The other half of the gate: an unbootability verdict that fires on
        # everything is not a verdict.
        from sglang.srt.uneven_perf import PlanInputs

        budgets = [32607 - 3000, 20480 - 2700, 20480 - 2700]
        inputs = PlanInputs(
            tp_size=3,
            model_path=_FP8,
            kv_cache_dtype="fp8_e4m3",
            speculative_algorithm="NEXTN",
            speculative_num_draft_tokens=4,
            max_running_requests=16,
            rank_gpu_id=[0, 1, 2],
            effective_vram_mib=budgets,
        )
        rates = ks.rates_from_probe(_PROBE, [0, 1, 2])
        model = ks.build_cost_model(inputs, budgets, budgets, rates)
        units = model.perf.mlp_unit_partition([2, 1, 1])
        cap = model.capacity(units, [0.0] * 3)
        self.assertTrue(cap["feasible"])
        # The measured boot reported 502528 max_total_num_tokens for this
        # vector; the capacity model is an ESTIMATE, so only the class is
        # asserted.
        self.assertTrue(
            4.0e5 <= cap["ctx"] <= 6.0e5,
            f"predicted {cap['ctx']:.0f} tokens against a measured 502528",
        )


# ---------------------------------------------------------------------------
# 5. The affine decomposition the closed form depends on
# ---------------------------------------------------------------------------


@unittest.skipUnless(_have(_FP8), f"needs the 27B-FP8 checkpoint at {_FP8}")
class TestAffineModel(_Bf16StateEnv):
    """The optimizer's closed form is only exact if the byte model really is
    affine in the unit vector. That is checked against the family model
    itself, not assumed."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from sglang.srt.uneven_perf import PlanInputs

        cls.budgets = [32607 - 3000, 20480 - 2700, 20480 - 2700]
        inputs = PlanInputs(
            tp_size=3,
            model_path=_FP8,
            kv_cache_dtype="fp8_e4m3",
            speculative_algorithm="NEXTN",
            speculative_num_draft_tokens=4,
            max_running_requests=16,
            rank_gpu_id=[0, 1, 2],
            effective_vram_mib=cls.budgets,
        )
        rates = ks.rates_from_probe(_PROBE, [0, 1, 2])
        cls.model = ks.build_cost_model(inputs, cls.budgets, cls.budgets, rates)

    def test_weight_bytes_match_the_family_model(self):
        for vec in ([2, 1, 1], [6, 1, 1], [1, 1, 1], [5, 3, 3], [1, 0, 0]):
            units = self.model.perf.mlp_unit_partition(vec)
            mine = self.model.weight_bytes(units)
            theirs = self.model.perf.streamed_bytes(vec)
            for r in range(3):
                self.assertAlmostEqual(
                    mine[r] / theirs[r],
                    1.0,
                    places=6,
                    msg=f"vector {vec}, rank {r}: affine model "
                    f"{mine[r]:.0f} vs family model {theirs[r]:.0f}",
                )

    def test_free_bytes_intercept_reproduces_the_capacity_model(self):
        # free_r(u) = free_r(0) - unit_bytes * u_r must reproduce
        # predict_capacity at every representable vector, because they are
        # the same arithmetic seen from two ends.
        free0 = self.model.free_bytes_at_zero()
        cell = float(self.model.perf.kv_cell_bytes)
        for vec in ([2, 1, 1], [6, 1, 1], [3, 2, 2], [5, 3, 3]):
            units = self.model.perf.mlp_unit_partition(vec)
            p = self.model.perf.predict_capacity(vec)["p"]
            for r in range(3):
                predicted = (free0[r] - self.model.unit_bytes * units[r]) / cell
                self.assertAlmostEqual(predicted, p[r], delta=1.0)

    def test_total_kv_is_invariant_under_the_key(self):
        # The identity the maxkv goal turns on: sum_r u_r is fixed, so
        # sum_r free_r is fixed and the MLP key MOVES KV between cards, it
        # does not create it. Stated as a result in the docstring; pinned
        # here so a future change to the byte model cannot break it silently.
        totals = []
        for vec in ([2, 1, 1], [6, 1, 1], [1, 1, 1], [5, 3, 3], [3, 4, 4]):
            units = self.model.perf.mlp_unit_partition(vec)
            totals.append(sum(self.model.capacity(units, [0.0] * 3)["p"]))
        for t in totals[1:]:
            self.assertAlmostEqual(t / totals[0], 1.0, places=6)

    def test_collective_term_comes_from_the_pair_matrix(self):
        # 2 all-reduces per layer of hidden*2 bytes, ring factor 2(R-1)/R,
        # narrowest ordered pair. Recomputed by hand here so a change to the
        # formula has to be deliberate.
        coll = self.model.collective_decode_s()
        n_layers = self.model.perf.n_layers
        hidden = self.model.perf.hidden
        expect = 2 * n_layers * (2 * 22.4e-6 + (2 * 2 / 3) * hidden * 2 / (4.32e9))
        self.assertAlmostEqual(coll, expect, places=9)
        self.assertGreater(coll * 1e3, 1.0)  # milliseconds, not microseconds

    def test_collective_is_absent_without_a_pair_matrix(self):
        from sglang.srt.uneven_perf import PlanInputs

        inputs = PlanInputs(
            tp_size=3,
            model_path=_FP8,
            kv_cache_dtype="fp8_e4m3",
            max_running_requests=16,
            rank_gpu_id=[0, 1, 2],
            effective_vram_mib=self.budgets,
        )
        rates = ks.rates_from_probe({"cards": _PROBE["cards"], "pairs": []}, [0, 1, 2])
        model = ks.build_cost_model(inputs, self.budgets, self.budgets, rates)
        self.assertIsNone(model.collective_decode_s())
        self.assertIsNone(model.prefill_seconds([46, 45, 45], 4096))


# ---------------------------------------------------------------------------
# 6. The goals end to end
# ---------------------------------------------------------------------------


@unittest.skipUnless(_have(_FP8), f"needs the 27B-FP8 checkpoint at {_FP8}")
class TestSolve(_Bf16StateEnv):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from sglang.srt.uneven_perf import PlanInputs

        cls.budgets = [32607 - 3000, 20480 - 2700, 20480 - 2700]
        cls.inputs = PlanInputs(
            tp_size=3,
            model_path=_FP8,
            kv_cache_dtype="fp8_e4m3",
            speculative_algorithm="NEXTN",
            speculative_num_draft_tokens=4,
            max_running_requests=16,
            rank_gpu_id=[0, 1, 2],
            effective_vram_mib=cls.budgets,
        )
        cls.rates = ks.rates_from_probe(_PROBE, [0, 1, 2])

    def _solve(self, **kw):
        return ks.solve(self.inputs, self.budgets, self.budgets, self.rates, **kw)

    def test_every_goal_returns_a_usable_key(self):
        for goal in ks.GOALS:
            ans = self._solve(goal=goal)
            self.assertTrue(ans.ok, goal)
            self.assertEqual(len(ans.candidates), 1)
            cand = ans.candidates[0]
            self.assertEqual(sum(cand.units), ans.model_units)
            self.assertTrue(cand.feasible, f"{goal}: {cand.reasons}")
            # Every candidate carries EVERY goal's value, sacrificed ones
            # included -- that is the whole point of the trade-off line.
            for other in ks.GOALS:
                self.assertIn(other, cand.predictions)
            self.assertEqual(cand.launch_flags[0], "--rank-mlp-ratio")

    def test_the_decode_optimum_is_not_beaten_by_the_measured_vectors(self):
        # The solver's own claim: it finds the decode optimum. Two vectors
        # that were actually booted on this rig are the adversaries.
        ans = self._solve(goal="dec")
        best = ans.candidates[0]
        model = ks.build_cost_model(self.inputs, self.budgets, self.budgets, self.rates)
        mine = model.decode_weight_time(best.units)
        for rival in ([2, 1, 1], [6, 1, 1], [5, 3, 3], [1, 1, 1]):
            theirs = model.decode_weight_time(model.perf.mlp_unit_partition(rival))
            self.assertLessEqual(
                mine,
                theirs * (1 + 1e-9),
                f"the booted vector {rival} beats the solver's decode optimum",
            )

    def test_the_prefill_optimum_is_not_beaten_either(self):
        ans = self._solve(goal="enc")
        best = ans.candidates[0]
        model = ks.build_cost_model(self.inputs, self.budgets, self.budgets, self.rates)
        mine = model.prefill_compute_time(best.units)
        for rival in ([2, 1, 1], [6, 1, 1], [5, 3, 3], [1, 1, 1]):
            theirs = model.prefill_compute_time(model.perf.mlp_unit_partition(rival))
            self.assertLessEqual(mine, theirs * (1 + 1e-9), str(rival))

    def test_prefill_and_decode_optima_pull_apart(self):
        # If they did not, there would be nothing to trade and no reason for
        # a front. On a mixed rig they must differ.
        enc = self._solve(goal="enc").candidates[0]
        dec = self._solve(goal="dec").candidates[0]
        self.assertNotEqual(enc.units, dec.units)
        self.assertGreater(enc.units[0], dec.units[0])

    def test_pareto_front_has_endpoints_and_a_knee(self):
        ans = self._solve(goal="dec", goal_b="enc", front_size=5)
        self.assertEqual(ans.mode, "pareto")
        self.assertGreaterEqual(len(ans.candidates), 3)
        self.assertLessEqual(len(ans.candidates), 5)
        labels = [c.label for c in ans.candidates]
        self.assertIn("knee", labels)
        # A front is ordered and non-dominated: as decode rises, prefill falls.
        dec = [c.predictions["dec"]["value"] for c in ans.candidates]
        enc = [c.predictions["enc"]["value"] for c in ans.candidates]
        self.assertEqual(dec, sorted(dec))
        self.assertEqual(enc, sorted(enc, reverse=True))
        for cand in ans.candidates:
            self.assertTrue(cand.tradeoff["line"])
            self.assertEqual(cand.remeasure["path"], "/api/split_probe")
            self.assertEqual(
                cand.remeasure["body"]["candidate"],
                ",".join(str(v) for v in cand.mlp_ratio),
            )

    def test_the_knee_sits_between_the_endpoints(self):
        ans = self._solve(goal="dec", goal_b="enc", front_size=5)
        knee = next(c for c in ans.candidates if c.label == "knee")
        dec = [c.predictions["dec"]["value"] for c in ans.candidates]
        self.assertLess(min(dec), knee.predictions["dec"]["value"])
        self.assertGreater(max(dec), knee.predictions["dec"]["value"])

    def test_constraint_form_respects_the_threshold(self):
        free = self._solve(goal="enc").candidates[0]
        floor = free.predictions["dec"]["value"] + 10.0
        ans = self._solve(goal="enc", constraints={"dec": floor})
        self.assertEqual(ans.mode, "constraint")
        cand = ans.candidates[0]
        self.assertGreaterEqual(cand.predictions["dec"]["value"], floor)
        # And it costs prefill, which is the trade the caller asked for.
        self.assertLess(
            cand.predictions["enc"]["value"], free.predictions["enc"]["value"]
        )

    def test_impossible_constraint_says_so_and_still_answers(self):
        ans = self._solve(goal="enc", constraints={"dec": 1.0e6})
        self.assertTrue(ans.ok)
        self.assertTrue(ans.reasons)
        self.assertEqual(len(ans.candidates), 1)
        self.assertIn("not satisfiable", ans.candidates[0].label)

    def test_unknown_goal_is_rejected(self):
        with self.assertRaises(ValueError):
            self._solve(goal="ttft_at_n")
        self.assertIn("ttft_at_n", ks.GOAL_INTERFACES)
        self.assertIn("session_max", ks.GOAL_INTERFACES)

    def test_pinned_roles_are_honoured(self):
        # rank 1 as a pure KV donor: the 0 % end of the continuum. It must
        # end up with no MLP units, and the model must still be placed.
        ans = self._solve(goal="maxkv", roles=["shard", "kv_donor", "shard"])
        cand = ans.candidates[0]
        self.assertEqual(cand.units[1], 0)
        self.assertEqual(sum(cand.units), ans.model_units)
        self.assertEqual(cand.roles, ["shard", "kv_donor", "shard"])

    def test_json_shape_is_renderable(self):
        out = self._solve(goal="dec", goal_b="maxkv").to_json()
        self.assertTrue(out["ok"])
        for key in (
            "goals",
            "goal_interfaces",
            "roles",
            "candidates",
            "reference",
            "caveats",
            "rates_basis",
            "anchor",
        ):
            self.assertIn(key, out)
        for cand in out["candidates"]:
            for goal in ks.GOALS:
                cell = cand["predictions"][goal]
                self.assertIn(cell["provenance"], ("measured", "estimate", "absent"))
                if cell["provenance"] == "absent":
                    self.assertIsNone(cell["value"])
                self.assertTrue(cell["basis"])


# ---------------------------------------------------------------------------
# 7. The additive mechanic — regression gate 5
# ---------------------------------------------------------------------------


class TestCoexistenceBracket(CustomTestCase):
    """The bracket arithmetic on hand-built estimates: no checkpoint needed,
    so the rule itself is pinned even where the model zoo is not."""

    @staticmethod
    def _est(key, weights, other, share=None):
        return ks.InstanceEstimate(
            key=key,
            kind="serving",
            local=True,
            feasible=True,
            reasons=[],
            prefill_tok_s=1000.0,
            decode_tok_s=50.0,
            max_kv_tokens=10000.0,
            weights_mib=dict(weights),
            other_mib=dict(other),
            posts_mib={},
            share_group=share,
        )

    def test_naive_duplication_sums_the_weights(self):
        a = self._est("a", {0: 6000}, {0: 2000})
        b = self._est("b", {0: 6000}, {0: 2000})
        out = ks.coexistence([a, b], {0: 20000})
        row = out["per_gpu"][0]
        # 6000 + 6000 weights + 2000 + 2000 other + one process post.
        self.assertAlmostEqual(row["weights_mib"], 12000.0)
        self.assertAlmostEqual(row["shared_weight_saving_mib"], 0.0)
        self.assertAlmostEqual(row["claimed_mib"], 16000.0 + ks.FIXED_PROCESS_POST_MIB)

    def test_rank_reuse_counts_shared_weights_once(self):
        # The dual-group runtime: a nested complementary shard and the shared
        # tp1 shard are the SAME bytes, so the union is the larger of the
        # two, not their sum. Pools and the process post stay duplicated,
        # because those really are.
        a = self._est("group", {0: 6000}, {0: 2000}, share="dual")
        b = self._est("lane", {0: 13000}, {0: 2000}, share="dual")
        out = ks.coexistence([a, b], {0: 20000})
        row = out["per_gpu"][0]
        self.assertAlmostEqual(row["weights_mib"], 13000.0)
        self.assertAlmostEqual(row["shared_weight_saving_mib"], 6000.0)
        self.assertAlmostEqual(row["claimed_mib"], 17000.0 + ks.FIXED_PROCESS_POST_MIB)
        self.assertIn("counted once", row["note"])

    def test_different_share_groups_do_not_share(self):
        a = self._est("a", {0: 6000}, {0: 1000}, share="x")
        b = self._est("b", {0: 6000}, {0: 1000}, share="y")
        row = ks.coexistence([a, b], {0: 20000})["per_gpu"][0]
        self.assertAlmostEqual(row["weights_mib"], 12000.0)

    def test_overflow_names_the_gpu_and_the_mib(self):
        a = self._est("a", {0: 15000}, {0: 4000})
        b = self._est("b", {1: 1000}, {1: 1000})
        out = ks.coexistence([a, b], {0: 16000, 1: 20000})
        self.assertFalse(out["fits"])
        bad = [r for r in out["per_gpu"] if not r["fits"]]
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad[0]["gpu"], 0)
        self.assertIn("over by", bad[0]["note"])
        self.assertLess(bad[0]["headroom_mib"], 0)

    def test_a_remote_instance_claims_nothing_local(self):
        # A prefill satellite on another rig adds throughput and no VRAM.
        remote = ks.InstanceEstimate(
            key="satellite",
            kind="satellite",
            local=False,
            feasible=True,
            reasons=[],
            prefill_tok_s=400.0,
            decode_tok_s=None,
            max_kv_tokens=None,
            weights_mib={},
            other_mib={},
            posts_mib={},
        )
        local = self._est("main", {0: 6000}, {0: 2000})
        out = ks.coexistence([remote, local], {0: 20000})
        self.assertTrue(out["fits"])
        self.assertEqual(out["per_gpu"][0]["instances"], ["main"])

    def test_reserve_is_subtracted_before_the_verdict(self):
        a = self._est("a", {0: 15000}, {0: 2000})
        self.assertTrue(ks.coexistence([a], {0: 20000})["fits"])
        self.assertFalse(ks.coexistence([a], {0: 20000}, reserve_mib={0: 4000})["fits"])


@unittest.skipUnless(
    _have(_FP8) and _have(_Q3), "needs the 27B-FP8 and 27B-Q3_K_M checkpoints"
)
class TestAdditiveRegression(_Bf16StateEnv):
    """Gate 5: the Q3_K_M trade, the coexistence verdict and the sum."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.out = ks.check_additive_regression(
            _Q3,
            _FP8,
            _PROBE,
            small_model_path=(_SMALL if _have(_SMALL) else None),
        )

    def test_a_trade_direction_is_exact(self):
        # No tolerance on the direction: solo prefills faster, the group
        # holds more KV. A model that gets this backwards is not usable.
        a = self.out["a_trade"]
        self.assertTrue(a["direction_ok"], a)
        self.assertGreater(a["prefill_ratio_solo_over_group"], 1.0)
        self.assertGreater(a["kv_ratio_group_over_solo"], 1.0)

    def test_a_absolute_rates_reproduce_three_measured_arms(self):
        a = self.out["a_trade"]
        self.assertEqual(len(a["arms"]), 3)
        for arm in a["arms"]:
            self.assertIsNotNone(arm["predicted_tok_s"], arm)
            self.assertLessEqual(
                abs(arm["deviation"]),
                self.out["rate_tolerance"],
                f"{arm['arm']}: predicted {arm['predicted_tok_s']} against "
                f"measured {arm['measured_tok_s']} "
                f"({arm['deviation'] * 100:+.1f} %)",
            )
        self.assertTrue(a["rates_within_tolerance"])

    def test_a_ratios_reproduce_the_recorded_trade(self):
        # 2.94x prefill bought with 4.21x of KV given up.
        a = self.out["a_trade"]
        self.assertAlmostEqual(a["measured_prefill_ratio"], 2.94, places=2)
        self.assertAlmostEqual(a["measured_kv_ratio"], 4.21, places=2)
        self.assertTrue(a["ratios_within_tolerance"], a)

    def test_a_error_is_one_sided_in_the_predicted_direction(self):
        # The un-fitted collective term is documented as UNDER-stating a
        # group collective (the pair matrix measured one ordered pair at a
        # time). So the collective-bearing arms must come out too FAST and
        # the solo arm too slow. If that ever flips, the explanation is
        # wrong even when the tolerance still passes.
        by_arm = {a["arm"]: a for a in self.out["a_trade"]["arms"]}
        solo = next(v for k, v in by_arm.items() if "solo" in k)
        groups = [v for k, v in by_arm.items() if "solo" not in k]
        self.assertLess(solo["deviation"], 0.0)
        for g in groups:
            self.assertGreater(g["deviation"], 0.0)

    def test_b_naive_duplication_does_not_fit(self):
        # 27B-Q3 beside 27B-Q3 with the weights held twice: the 5090 runs
        # out, and the aggregate is NOT reported.
        b = self.out["b_coexistence"]
        self.assertFalse(b["naive"]["fits"])
        self.assertFalse(b["naive"]["aggregate_reported"])
        bad = [r for r in b["naive"]["coexistence"]["per_gpu"] if not r["fits"]]
        self.assertTrue(bad)
        self.assertEqual(bad[0]["gpu"], 0)

    def test_b_rank_reuse_makes_the_same_pair_fit(self):
        # The dual-group runtime: 12.9 GiB of weights on the 5090 ONCE.
        b = self.out["b_coexistence"]
        self.assertTrue(b["rank_reuse"]["fits"])
        self.assertTrue(b["verdict_ok"])
        gpu0 = b["rank_reuse"]["coexistence"]["per_gpu"][0]
        self.assertEqual(gpu0["gpu"], 0)
        self.assertGreater(gpu0["shared_weight_saving_mib"], 1000.0)
        self.assertGreater(gpu0["headroom_mib"], 0.0)

    def test_b_reuse_aggregate_prefill_is_about_four_times_the_group(self):
        # Measured: (3202.8 + 1089.4) / 1089.4 = 3.94x. The prediction
        # carries the ratio tolerance, for the reason stated in the anchor.
        b = self.out["b_coexistence"]["rank_reuse"]
        self.assertIsNotNone(b["aggregate_prefill"]["value"])
        self.assertEqual(b["aggregate_prefill"]["provenance"], "estimate")
        self.assertAlmostEqual(b["measured_aggregate_over_group"], 3.94, places=2)
        self.assertGreater(b["aggregate_over_group"], 2.5)
        self.assertTrue(b["ratio_within_tolerance"], b)

    @unittest.skipUnless(_have(_SMALL), f"needs a small checkpoint at {_SMALL}")
    def test_c_a_fitting_second_instance_gives_an_additive_aggregate(self):
        c = self.out["c_additive"]
        self.assertTrue(c["ran"])
        self.assertTrue(c["fits"], c["coexistence"])
        self.assertTrue(c["sum_matches"], c)
        self.assertEqual(c["aggregate_prefill"]["provenance"], "estimate")
        total = c["aggregate_prefill"]["value"]
        parts = list(c["per_instance_prefill_tok_s"].values())
        self.assertAlmostEqual(total, sum(parts), delta=0.5)
        self.assertGreater(total, max(parts))


# ---------------------------------------------------------------------------
# 7b. Co-resident KV capacity — the aggregate() over-count
# ---------------------------------------------------------------------------


class TestSharedProcessBracket(CustomTestCase):
    """``shared_process`` is a property of the RUNTIME, so it is a parameter.
    Two lanes of one engine process share the CUDA context, the graph pool
    and the activation scratch; charging a second post would invent memory
    nobody allocates."""

    @staticmethod
    def _est(key, weights, other, share=None):
        return ks.InstanceEstimate(
            key=key,
            kind="serving",
            local=True,
            feasible=True,
            reasons=[],
            prefill_tok_s=1000.0,
            decode_tok_s=50.0,
            max_kv_tokens=10000.0,
            weights_mib=dict(weights),
            other_mib=dict(other),
            posts_mib={},
            share_group=share,
        )

    def test_one_process_drops_exactly_one_post_per_shared_card(self):
        a = self._est("a", {0: 6000, 1: 3000}, {0: 2000, 1: 1000}, share="d")
        b = self._est("b", {0: 6000}, {0: 2000}, share="d")
        two = ks.coexistence([a, b], {0: 20000, 1: 20000})
        one = ks.coexistence([a, b], {0: 20000, 1: 20000}, shared_process=True)
        h2 = {r["gpu"]: r["headroom_mib"] for r in two["per_gpu"]}
        h1 = {r["gpu"]: r["headroom_mib"] for r in one["per_gpu"]}
        # GPU 0 carries both lanes -> one post saved. GPU 1 carries one lane
        # -> nothing to save, and nothing invented either.
        self.assertAlmostEqual(h1[0] - h2[0], ks.FIXED_PROCESS_POST_MIB, places=6)
        self.assertAlmostEqual(h1[1], h2[1], places=6)
        self.assertTrue(one["per_gpu"][0]["shared_process"])
        self.assertFalse(two["per_gpu"][0]["shared_process"])

    def test_lane_count_is_reported_per_card(self):
        a = self._est("a", {0: 1000}, {0: 500})
        b = self._est("b", {0: 1000, 1: 500}, {0: 500, 1: 200})
        rows = {
            r["gpu"]: r for r in ks.coexistence([a, b], {0: 20000, 1: 20000})["per_gpu"]
        }
        self.assertEqual(rows[0]["lanes"], 2)
        self.assertEqual(rows[1]["lanes"], 1)


@unittest.skipUnless(_have(_FP8), f"needs the 27B-FP8 checkpoint at {_FP8}")
class TestCoresidentCapacity(_Bf16StateEnv):
    """The defect the FP8 tp2-in-tp3 evaluation found, and its fix.

    ``estimate_instance`` sizes a lane against the budget it was given, i.e.
    as if that lane owned its cards. Summing two such figures for lanes that
    SHARE a card counts the same free bytes twice. The two candidates of the
    evaluation are the regression: their corrected joint capacities are 240k
    and 343k tokens, where the old code reported 1.14M for both.
    """

    #: The evaluation's rig, in the resolved device order (the x8-attached
    #: 3080 is cuda_index 2, not 1 -- both profile signals agree by ~2x).
    PD_G, MAIN_G = [0, 2], [0, 2, 1]
    PD_B, MAIN_B = [29607, 17780], [29607, 17780, 17780]
    TOTAL = {0: 32607, 1: 20480, 2: 20480}
    RESERVE = {0: 3000, 1: 2700, 2: 2700}
    #: What the broken sum produced for BOTH candidates.
    OLD_WRONG_SUM = 1143619.0

    def _spec(self, key, tp, gpus, budgets, vec, mrr, kind="serving"):
        return ks.InstanceSpec(
            key=key,
            model_path=_FP8,
            tp_size=tp,
            rank_gpu_id=list(gpus),
            budgets_mib=list(budgets),
            base_plan=list(budgets),
            mlp_vector=list(vec),
            kv_cache_dtype="fp8_e4m3",
            speculative_algorithm="NEXTN",
            speculative_num_draft_tokens=4,
            max_running_requests=mrr,
            kind=kind,
            share_group="dual",
        )

    def _pair(self, main_vec):
        return [
            self._spec("main", 3, self.MAIN_G, self.MAIN_B, main_vec, 4),
            self._spec("pd", 2, self.PD_G, self.PD_B, [59, 9], 1, "prefill_lane"),
        ]

    def _agg(self, main_vec, **kw):
        return ks.aggregate(
            self._pair(main_vec),
            _PROBE,
            self.TOTAL,
            reserve_mib=self.RESERVE,
            prefill_tokens=20000,
            **kw,
        )

    def test_candidate_a_joint_capacity(self):
        # Candidate A of the evaluation: PD 59,9 @mrr 1 + main 69,18,49 @mrr 4.
        out = self._agg([69, 18, 49])
        self.assertTrue(out["fits"])
        self.assertTrue(out["shares_a_card"])
        kv = out["aggregate"]["max_kv_tokens"]
        self.assertEqual(kv["provenance"], "estimate")
        self.assertAlmostEqual(kv["value"], 240361.0, delta=2000.0)
        rows = {i["key"]: i for i in out["instances"]}
        self.assertAlmostEqual(
            rows["main"]["coresident_kv_tokens"], 203646.0, delta=2000.0
        )
        self.assertAlmostEqual(
            rows["pd"]["coresident_kv_tokens"], 36715.0, delta=1000.0
        )

    def test_candidate_c_joint_capacity(self):
        out = self._agg([59, 4, 5])
        kv = out["aggregate"]["max_kv_tokens"]
        self.assertAlmostEqual(kv["value"], 342942.0, delta=3000.0)

    def test_the_old_sum_is_not_reproduced(self):
        # The non-regression, stated as such: whatever the model says, it must
        # not be the number that assumed each lane owned the whole card.
        for vec in ([69, 18, 49], [59, 4, 5]):
            out = self._agg(vec)
            kv = out["aggregate"]["max_kv_tokens"]["value"]
            self.assertLess(kv, self.OLD_WRONG_SUM * 0.5, vec)
            solo = sum(i["max_kv_tokens"] for i in out["instances"])
            self.assertAlmostEqual(solo, self.OLD_WRONG_SUM, delta=5000.0)
            self.assertGreater(solo / kv, 3.0)

    def test_the_basis_names_the_over_count_it_removed(self):
        out = self._agg([69, 18, 49])
        basis = out["aggregate"]["max_kv_tokens"]["basis"]
        self.assertIn("co-resident capacity", basis)
        self.assertIn("over-count", basis)
        self.assertIn("#260", basis)

    def test_solo_capacities_are_still_reported_per_lane(self):
        # The re-sizing must not destroy the standalone figure; a caller
        # comparing "this lane alone" against "this lane beside the other"
        # needs both, and that comparison IS the cost of co-residence.
        out = self._agg([69, 18, 49])
        for row in out["instances"]:
            self.assertIsNotNone(row["max_kv_tokens"])
            self.assertLess(row["coresident_kv_tokens"], row["max_kv_tokens"])

    def test_shared_process_raises_capacity_by_the_freed_posts(self):
        two = self._agg([69, 18, 49])
        one = self._agg([69, 18, 49], shared_process=True)
        self.assertGreater(
            one["aggregate"]["max_kv_tokens"]["value"],
            two["aggregate"]["max_kv_tokens"]["value"],
        )
        h2 = {r["gpu"]: r["headroom_mib"] for r in two["coexistence"]["per_gpu"]}
        h1 = {r["gpu"]: r["headroom_mib"] for r in one["coexistence"]["per_gpu"]}
        # The two shared cards each give back exactly one post; the third,
        # which carries only the main lane, is untouched.
        for g in (0, 2):
            self.assertAlmostEqual(h1[g] - h2[g], ks.FIXED_PROCESS_POST_MIB, places=4)
        self.assertAlmostEqual(h1[1], h2[1], places=4)

    def test_disjoint_lanes_are_not_resized(self):
        # No shared card -> nothing to divide, and the plain sum is correct.
        # Re-sizing here would be as wrong as summing was in the shared case.
        specs = [
            self._spec("a", 2, [0, 2], self.PD_B, [59, 9], 2),
            ks.InstanceSpec(
                key="remote",
                model_path=_FP8,
                tp_size=2,
                rank_gpu_id=[0, 2],
                budgets_mib=self.PD_B,
                base_plan=self.PD_B,
                mlp_vector=[59, 9],
                kv_cache_dtype="fp8_e4m3",
                max_running_requests=2,
                local=False,
            ),
        ]
        out = ks.aggregate(specs, _PROBE, self.TOTAL, reserve_mib=self.RESERVE)
        self.assertFalse(out["shares_a_card"])
        rows = {i["key"]: i for i in out["instances"]}
        self.assertIsNone(rows["a"]["coresident_budget_mib"])
        self.assertEqual(rows["a"]["coresident_kv_tokens"], rows["a"]["max_kv_tokens"])

    def test_budget_mapping_refuses_an_overflowing_card(self):
        # A negative leftover has no honest division, so the mapping returns
        # None and the caller reports absent instead of inventing a split.
        specs = self._pair([69, 18, 49])
        estimates = [
            ks.estimate_instance(s, _PROBE, prefill_tokens=20000) for s in specs
        ]
        self.assertIsNone(
            ks.coresident_budgets(
                specs,
                estimates,
                {0: 8000, 1: 8000, 2: 8000},
                reserve_mib=self.RESERVE,
            )
        )

    def test_budget_mapping_conserves_the_cards(self):
        # Conservation, with the sign the mapping really has:
        #
        #   sum_i budget_i(g) = available(g) - posts + shared_weight_saving
        #
        # The shared weight bytes are handed to BOTH lanes on purpose --
        # each lane's capacity model subtracts its own weight view, so each
        # has to be given it. The physical accounting stays right because
        # the leftover those budgets are built from charged the shared bytes
        # only once. Reading this line as "available minus the saving" is the
        # natural mistake, so it is pinned with the correct sign.
        specs = self._pair([69, 18, 49])
        estimates = [
            ks.estimate_instance(s, _PROBE, prefill_tokens=20000) for s in specs
        ]
        budgets = ks.coresident_budgets(
            specs, estimates, self.TOTAL, reserve_mib=self.RESERVE
        )
        self.assertIsNotNone(budgets)
        by_key = {e.key: e for e in estimates}
        for gpu in (0, 2):  # the shared cards
            handed = 0.0
            for spec in specs:
                for r, g in enumerate(spec.rank_gpu_id):
                    if int(g) == gpu:
                        handed += budgets[spec.key][r]
            avail = self.TOTAL[gpu] - self.RESERVE[gpu]
            saved = sum(by_key[s.key].weights_mib.get(gpu, 0.0) for s in specs) - max(
                by_key[s.key].weights_mib.get(gpu, 0.0) for s in specs
            )
            self.assertAlmostEqual(
                handed, avail - ks.FIXED_PROCESS_POST_MIB + saved, delta=3.0
            )
            self.assertGreater(saved, 0.0)  # the cards really are shared


# ---------------------------------------------------------------------------
# 8. The JSON surface
# ---------------------------------------------------------------------------


@unittest.skipUnless(_have(_FP8), f"needs the 27B-FP8 checkpoint at {_FP8}")
class TestSolverApi(_Bf16StateEnv):
    def setUp(self):
        from sglang.srt.planner import solver_api

        self.api = solver_api
        self._real_probe = solver_api.cached_card_probe
        solver_api.cached_card_probe = lambda: _PROBE

    def tearDown(self):
        self.api.cached_card_probe = self._real_probe

    def _body(self, **kw):
        base = dict(
            model_path=_FP8,
            tp_size=3,
            rank_gpu_id=[0, 1, 2],
            rank_gpu_memory_mib=[32607 - 3000, 20480 - 2700, 20480 - 2700],
            kv_cache_dtype="fp8_e4m3",
            speculative_algorithm="NEXTN",
            speculative_num_draft_tokens=4,
            max_running_requests=16,
            target_context=8192,
        )
        base.update(kw)
        return base

    def test_payload_is_json_serializable(self):
        out = self.api.key_solver_payload(self._body(goal="dec", goal_b="enc"))
        self.assertTrue(out["ok"])
        json.dumps(out)  # must not raise

    def test_missing_model_path_is_a_reason_not_an_exception(self):
        out = self.api.key_solver_payload({})
        self.assertFalse(out["ok"])
        self.assertTrue(out["reasons"])

    def test_rank_gpu_id_length_mismatch_is_rejected(self):
        out = self.api.key_solver_payload(self._body(rank_gpu_id=[0, 1], tp_size=3))
        self.assertFalse(out["ok"])
        self.assertIn("rank_gpu_id", out["reasons"][0])

    def test_unknown_constraint_goal_is_rejected(self):
        out = self.api.key_solver_payload(
            self._body(goal="dec", constraints={"ttft_at_n": 1.0})
        )
        self.assertFalse(out["ok"])

    def test_missing_budget_is_rejected_with_a_remedy(self):
        body = self._body()
        body.pop("rank_gpu_memory_mib")
        out = self.api.key_solver_payload(body)
        self.assertFalse(out["ok"])
        self.assertIn("rank_gpu_memory_mib", out["reasons"][0])

    def test_no_card_probe_offers_the_instrument(self):
        self.api.cached_card_probe = lambda: None
        out = self.api.key_solver_payload(self._body())
        self.assertFalse(out["ok"])
        self.assertEqual(out["remeasure"]["path"], "/api/card_probe")

    def test_model_payload_reports_predicted_against_measured(self):
        out = self.api.key_solver_model_payload({"model_path": _FP8})
        self.assertTrue(out["ok"])
        self.assertTrue(out["all_signs_match"])
        self.assertTrue(out["all_within_tolerance"])
        for row in out["anchors"]:
            self.assertIn("predicted_pct", row)
            self.assertIn("measured_pct", row)
            self.assertTrue(row["tolerance_reason"])
        json.dumps(out)

    def test_aggregate_payload_needs_the_totals(self):
        out = self.api.key_solver_aggregate_payload(
            {
                "instances": [
                    {
                        "model_path": _FP8,
                        "rank_gpu_id": [0],
                        "rank_gpu_memory_mib": [20000],
                    }
                ]
            }
        )
        self.assertFalse(out["ok"])
        self.assertIn("gpu_total_mib", out["reasons"][0])

    def test_aggregate_payload_round_trips(self):
        out = self.api.key_solver_aggregate_payload(
            {
                "gpu_total_mib": {"0": 32607, "1": 20480, "2": 20480},
                "instances": [
                    {
                        "key": "main",
                        "model_path": _FP8,
                        "tp_size": 3,
                        "rank_gpu_id": [0, 1, 2],
                        "rank_gpu_memory_mib": [29607, 17780, 17780],
                        "kv_cache_dtype": "fp8_e4m3",
                        "max_running_requests": 16,
                    }
                ],
            }
        )
        self.assertTrue(out["ok"])
        self.assertIn("coexistence", out)
        self.assertIn("aggregate", out)
        json.dumps(out)


if __name__ == "__main__":
    unittest.main()
