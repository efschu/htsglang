"""CPU unit tests for the offline config planner (design #97, stage S1).

No GPU, no NVML, no server boot, no weight loads: the model is a synthetic
Qwen3.6-27B-shaped checkpoint dir (config.json + a sparse .safetensors file
whose SIZE anchors the byte model), the hardware is a hand-declared
HardwareSpec of the reference rig (1x RTX 5090 32 GiB + 2x RTX 3080 20 GiB).
"""

import dataclasses
import json
import os
import tempfile
import unittest

from sglang.srt.planner import advantage as advantage_mod
from sglang.srt.planner import capacity as capacity_mod
from sglang.srt.planner import feasibility as feasibility_mod
from sglang.srt.planner import hardware as hardware_mod
from sglang.srt.planner import plan as plan_mod
from sglang.srt.planner.feasibility import PlanRejected, plan, validate_plan_inputs
from sglang.srt.planner.hardware import hardware_from_manual, parse_manual_gpu
from sglang.srt.planner.model import resolve_model_ref
from sglang.srt.planner.plan import derive_auto_plan
from sglang.srt.uneven_perf import PlanInputs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

# Qwen3.6-27B-shaped dense hybrid-GDN geometry (4 KV heads -> even TP=3 is
# geometrically impossible for stock; the fork's showcase case).
_CONFIG = {
    "architectures": ["Qwen3NextForCausalLM"],
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_hidden_layers": 48,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "vocab_size": 151936,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 12,
    "quantization_config": {"group_size": 32},
}

#: The reference rig (hand-declared; enumeration order 5090 first).
RIG = ("RTX 5090:32607", "RTX 3080:20480", "RTX 3080:20480")


def _make_checkpoint(tmpdir, config=_CONFIG, ckpt_gib=14.0):
    path = os.path.join(tmpdir, "model")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)
    # Sparse file: reports the full size (the byte-model anchor) but uses
    # no disk.
    with open(os.path.join(path, "model-00001.safetensors"), "wb") as f:
        f.truncate(int(ckpt_gib * 2**30))
    return path


class PlannerFixture(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.model = _make_checkpoint(cls._tmp.name)
        cls.hw = hardware_from_manual(RIG)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()


class TestHardwareSpec(PlannerFixture):
    def test_manual_parse(self):
        g = parse_manual_gpu("RTX 3080:20g", 1)
        self.assertEqual((g.name, g.total_mib, g.index), ("RTX 3080", 20480, 1))
        self.assertIsNone(g.free_mib)  # manual specs know no live free

    def test_manual_parse_rejects_garbage(self):
        for bad in ("RTX 3080", "x:", ":123", "a:12q"):
            with self.assertRaises(ValueError):
                parse_manual_gpu(bad, 0)

    def test_unknown_gpu_named_in_error(self):
        with self.assertRaisesRegex(ValueError, r"GPU 7.*RTX 5090"):
            self.hw.gpu(7)


class TestDeriveAutoPlan(PlannerFixture):
    def _inputs(self, rank_gpu_id, **kw):
        return PlanInputs(
            tp_size=len(rank_gpu_id),
            model_path=self.model,
            rank_gpu_id=list(rank_gpu_id),
            **kw,
        )

    def test_heterogeneous_budgets_and_ratio(self):
        auto = derive_auto_plan(self._inputs([0, 1, 2]), self.hw)
        # 5090 rank gets the bigger budget; ratio is non-uniform, gcd-reduced.
        self.assertGreater(auto.budgets_mib[0], auto.budgets_mib[1])
        self.assertEqual(auto.budgets_mib[1], auto.budgets_mib[2])
        self.assertIsNotNone(auto.rank_tp_ratio)
        import math

        self.assertEqual(math.gcd(*auto.rank_tp_ratio), 1)
        # Budget = total - reserve (no live free term for manual specs).
        for r, gid in enumerate([0, 1, 2]):
            self.assertEqual(
                auto.budgets_mib[r],
                self.hw.gpu(gid).total_mib - auto.auto_reserve_per_gpu[gid],
            )

    def test_uniform_budgets_collapse_to_even(self):
        hw = hardware_from_manual(["A:20480", "B:20480"])
        auto = derive_auto_plan(self._inputs([0, 1]), hw)
        self.assertIsNone(auto.rank_tp_ratio)  # classic even split
        self.assertEqual(auto.budgets_mib[0], auto.budgets_mib[1])

    def test_colocation_divides_the_card(self):
        solo = derive_auto_plan(self._inputs([0, 1, 2]), self.hw)
        colo = derive_auto_plan(self._inputs([0, 0, 1, 2]), self.hw)
        # Two ranks on the 5090: each gets less than half the solo budget
        # (the capture reserve also scales with co-located ranks, #68).
        self.assertLess(colo.budgets_mib[0], solo.budgets_mib[0] // 2 + 1)
        self.assertEqual(colo.budgets_mib[0], colo.budgets_mib[1])

    def test_user_free_reserve_shrinks_the_budget(self):
        base = derive_auto_plan(self._inputs([0, 1, 2]), self.hw)
        carved = derive_auto_plan(
            self._inputs([0, 1, 2]), self.hw, user_free_reserve_mib=[2048, 0, 0]
        )
        self.assertEqual(
            carved.budgets_mib[0], base.budgets_mib[0] - 2048
        )
        self.assertEqual(carved.budgets_mib[1:], base.budgets_mib[1:])

    def test_overcarved_reserve_fails_loud_naming_the_card(self):
        with self.assertRaisesRegex(ValueError, r"GPU 1.*RTX 3080"):
            derive_auto_plan(
                self._inputs([0, 1, 2]),
                self.hw,
                user_free_reserve_mib=[0, 40000, 0],
            )

    def test_reserve_length_must_match_cards(self):
        with self.assertRaisesRegex(ValueError, "plan-free-reserve"):
            derive_auto_plan(
                self._inputs([0, 1, 2]), self.hw, user_free_reserve_mib=[1024]
            )


class TestValidation(PlannerFixture):
    """Manual-edit validation (design §2.6): every invalid layout is
    rejected WITH a reason, through the same path as auto plans."""

    def _inputs(self, **kw):
        base = dict(
            tp_size=3,
            model_path=self.model,
            rank_gpu_id=[0, 1, 2],
            effective_vram_mib=[20000, 15000, 15000],
            rank_tp_ratio=[4, 3, 3],
        )
        base.update(kw)
        return PlanInputs(**base)

    def test_valid_plan_has_no_errors(self):
        self.assertEqual(validate_plan_inputs(self._inputs(), self.hw), [])

    def test_ratio_length_mismatch(self):
        errs = validate_plan_inputs(
            self._inputs(rank_tp_ratio=[4, 3]), self.hw
        )
        self.assertTrue(any("--rank-tp-ratio length" in e for e in errs))

    def test_budget_list_requires_ratio(self):
        errs = validate_plan_inputs(self._inputs(rank_tp_ratio=None), self.hw)
        self.assertTrue(any("requires" in e for e in errs), errs)

    def test_unknown_gpu_id(self):
        errs = validate_plan_inputs(
            self._inputs(rank_gpu_id=[0, 1, 9]), self.hw
        )
        self.assertTrue(any("GPU 9" in e for e in errs), errs)

    def test_physical_impossibility_names_card_ranks_sum_total(self):
        errs = validate_plan_inputs(
            self._inputs(
                rank_gpu_id=[0, 0, 1],
                effective_vram_mib=[20000, 20000, 15000],
            ),
            self.hw,
        )
        msg = "\n".join(errs)
        self.assertIn("Physical impossibility", msg)
        self.assertIn("GPU 0", msg)
        self.assertIn("[0, 1]", msg)
        self.assertIn("40000", msg)
        self.assertIn("32607", msg)

    def test_kv_token_vector_length(self):
        errs = validate_plan_inputs(
            self._inputs(kv_token_vector=[3, 1]), self.hw
        )
        self.assertTrue(any("KV token vector" in e for e in errs))

    def test_dcp_size_range(self):
        errs = validate_plan_inputs(self._inputs(dcp_size=5), self.hw)
        self.assertTrue(any("--dcp-size" in e for e in errs))


class TestPlanEndToEnd(PlannerFixture):
    def test_auto_plan_fits_on_the_rig(self):
        result = plan(self.model, self.hw, tp_size=3, with_advantage=False)
        self.assertTrue(result.fits, result.infeasible_reasons)
        cap = result.capacity
        self.assertTrue(cap.feasible)
        self.assertGreater(cap.max_context_tokens, 0)
        self.assertEqual(len(cap.per_rank), 3)
        # The 5090 rank must hold more weight bytes than a 3080 rank.
        self.assertGreater(cap.per_rank[0].weight_gib, cap.per_rank[1].weight_gib)
        # Launch flags round-trip the exact budgets (design §2.5).
        flags = " ".join(result.launch_flags)
        self.assertIn("--rank-gpu-id 0,1,2", flags)
        self.assertIn("--rank-gpu-memory-mib", flags)
        self.assertIn("--rank-tp-ratio", flags)

    def test_manual_edit_reruns_same_validation(self):
        # Same entry point, one knob edited to an invalid value -> rejected
        # with the reason (never silently accepted).
        with self.assertRaises(PlanRejected) as ctx:
            plan(
                self.model,
                self.hw,
                rank_gpu_id=[0, 1, 2],
                rank_gpu_memory_mib=[20000, 15000, 15000],
                rank_tp_ratio=[4, 3],  # wrong length
            )
        self.assertTrue(
            any("--rank-tp-ratio length" in r for r in ctx.exception.reasons)
        )

    def test_infeasible_names_the_specific_card(self):
        # A geometry whose weights exceed the whole rig's aggregate budget —
        # not even uneven-TP redistribution can absorb that (a bigger
        # checkpoint alone is not enough: the byte model clamps bytes/param,
        # so the PARAM COUNT must be big).
        big_cfg = dict(
            _CONFIG,
            hidden_size=8192,
            intermediate_size=29568,
            num_hidden_layers=96,
            num_attention_heads=64,
            num_key_value_heads=8,
            head_dim=128,
            layer_types=None,
        )
        big_cfg.pop("layer_types")
        big = _make_checkpoint(
            self._tmp.name + "/big-model-tmp", big_cfg, ckpt_gib=120.0
        )
        try:
            result = plan(big, self.hw, tp_size=3, with_advantage=False)
            self.assertFalse(result.fits)
            msg = "\n".join(result.infeasible_reasons)
            self.assertIn("RTX", msg)
            self.assertIn("GiB", msg)
        finally:
            import shutil

            shutil.rmtree(os.path.dirname(big), ignore_errors=True)

    def test_free_reserve_lowers_capacity_monotonically(self):
        base = plan(self.model, self.hw, tp_size=3, with_advantage=False)
        carved = plan(
            self.model,
            self.hw,
            tp_size=3,
            user_free_reserve_mib=[4096, 2048, 2048],
            with_advantage=False,
        )
        self.assertLess(
            carved.capacity.max_context_tokens,
            base.capacity.max_context_tokens,
        )

    def test_colocation_via_rank_gpu_id_duplicates(self):
        result = plan(
            self.model,
            self.hw,
            rank_gpu_id=[0, 0, 1, 2],
            with_advantage=False,
        )
        self.assertEqual(result.inputs.tp_size, 4)
        self.assertEqual(
            result.capacity.per_rank[0].budget_mib,
            result.capacity.per_rank[1].budget_mib,
        )

    def test_tp_above_card_count_requires_explicit_map(self):
        with self.assertRaisesRegex(PlanRejected, "co-location"):
            plan(self.model, self.hw, tp_size=4)


class TestAdvantage(PlannerFixture):
    def test_stock_cannot_shard_4_kv_heads_across_3(self):
        result = plan(self.model, self.hw, tp_size=3)
        adv = result.advantage
        self.assertFalse(adv.stock.runs)
        self.assertTrue(any("KV heads" in r for r in adv.stock.reasons))
        # Advantage is feasibility; no capacity % against a non-running stock.
        self.assertIsNone(adv.capacity_pct_range)

    def test_stock_runs_on_even_geometry_and_gain_is_a_range(self):
        cfg = dict(_CONFIG, num_key_value_heads=8, num_attention_heads=32)
        model = _make_checkpoint(self._tmp.name + "/even-tmp", cfg, ckpt_gib=10.0)
        try:
            result = plan(
                model, self.hw, tp_size=2, rank_gpu_id=[0, 1]
            )  # 5090 + one 3080
            adv = result.advantage
            self.assertTrue(adv.stock.runs, adv.stock.reasons)
            self.assertIsNotNone(adv.capacity_pct_range)
            lo, hi = adv.capacity_pct_range
            self.assertLess(lo, hi)  # a band, never a point
            self.assertGreater(hi, 0)  # uneven recovers the 5090 surplus
        finally:
            import shutil

            shutil.rmtree(os.path.dirname(model), ignore_errors=True)

    def test_measured_is_absent_without_a_cached_profile(self):
        # Manual hardware + no NVML in the loop: measured fields are ABSENT
        # (None), never a fallback estimate.
        result = plan(self.model, self.hw, tp_size=3)
        self.assertIsNone(result.advantage.measured)
        self.assertIsNone(result.advantage.decode_knee_ok)


class TestHonestyStructure(CustomTestCase):
    """Design §3.4: emitting an estimated-throughput number must be
    STRUCTURALLY impossible — no planner dataclass may even carry a field
    that could hold one."""

    FORBIDDEN = (
        "tokps",
        "tok_s",
        "tok_per",
        "tps",
        "throughput",
        "latency",
        "speed",
        "perf",
    )

    def test_no_planner_type_has_a_throughput_shaped_field(self):
        offenders = []
        for mod in (
            feasibility_mod,
            capacity_mod,
            advantage_mod,
            plan_mod,
            hardware_mod,
        ):
            for name in dir(mod):
                obj = getattr(mod, name)
                if not (isinstance(obj, type) and dataclasses.is_dataclass(obj)):
                    continue
                for f in dataclasses.fields(obj):
                    lowered = f.name.lower()
                    if any(bad in lowered for bad in self.FORBIDDEN):
                        offenders.append(f"{mod.__name__}.{name}.{f.name}")
        self.assertEqual(offenders, [])

    def test_advantage_exposes_only_the_three_honest_fields(self):
        names = {f.name for f in dataclasses.fields(advantage_mod.Advantage)}
        self.assertEqual(
            names,
            {"stock", "capacity_pct_range", "measured", "decode_knee_ok"},
        )

    def test_measured_card_score_is_measured_only(self):
        # The only perf-ish numbers in the package are the MEASURED probe
        # scores echoed from the cache — assert the type says so.
        doc = advantage_mod.MeasuredCardScore.__doc__ or ""
        self.assertIn("MEASURED", doc)


class TestPlanInputsFromServerArgs(CustomTestCase):
    """The single-source-of-truth contract: the boot path builds the same
    PlanInputs the planner builds (design §2.1/§5)."""

    def test_mapping(self):
        from sglang.srt.server_args import ServerArgs

        sa = ServerArgs(model_path="dummy")  # short-circuits __post_init__
        sa.tp_size = 3
        sa.rank_gpu_id = [0, 1, 1]
        sa.rank_gpu_memory_mib = [20000, 15000, 15000]
        sa.rank_tp_ratio = [4, 3, 3]
        sa.kv_cache_dtype = "fp8_e4m3"
        pi = PlanInputs.from_server_args(sa)
        self.assertEqual(pi.tp_size, 3)
        self.assertEqual(pi.rank_gpu_id, [0, 1, 1])
        self.assertEqual(pi.effective_vram_mib, [20000, 15000, 15000])
        self.assertEqual(pi.rank_gpu_memory_mib, [20000, 15000, 15000])
        self.assertEqual(pi.rank_tp_ratio, [4, 3, 3])
        self.assertEqual(pi.kv_cache_dtype, "fp8_e4m3")

    def test_scalar_budget_expands_per_rank(self):
        from sglang.srt.server_args import ServerArgs

        sa = ServerArgs(model_path="dummy")
        sa.tp_size = 2
        sa.rank_gpu_memory_mib = 16000
        pi = PlanInputs.from_server_args(sa)
        self.assertEqual(pi.effective_vram_mib, [16000, 16000])


class TestModelResolution(CustomTestCase):
    def test_dir_with_config(self):
        with tempfile.TemporaryDirectory() as d:
            path = _make_checkpoint(d)
            self.assertEqual(resolve_model_ref(path), path)

    def test_gguf_dir_resolves_to_file(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "m.gguf"), "wb").close()
            open(os.path.join(d, "imatrix_unsloth.gguf_file"), "wb").close()
            self.assertEqual(resolve_model_ref(d), os.path.join(d, "m.gguf"))

    def test_empty_dir_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "neither"):
                resolve_model_ref(d)


if __name__ == "__main__":
    unittest.main()
