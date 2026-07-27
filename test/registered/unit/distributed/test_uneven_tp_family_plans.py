"""CPU unit tests for the NAMED family shard plans of uneven TP
(--rank-mlp-ratio / SGLANG_UNEVEN_MLP_VECTOR).

Covers the registry semantics in sglang.srt.distributed.utils (family
fallback to the base vector, isolation between the base plan and family
plans, validation) and the loader consistency: layers constructed with
tp_family="mlp" must partition their shapes AND their checkpoint offsets
with the family vector while base-plan layers stay untouched — verified
by a full roundtrip (concatenation of all rank shards reconstructs the
full tensor) with a family vector that DIFFERS from the base vector.

No GPU, no distributed init: layer classes accept explicit
tp_rank/tp_size; `sgl_kernel` is stubbed out before the sglang imports.
"""

import importlib.util
import sys
import types
import unittest


def _install_sgl_kernel_stub():
    if importlib.util.find_spec("sgl_kernel") is not None:
        # The real package is importable; stubbing it here would leave a
        # process-wide empty-__path__ package that breaks every later
        # ``import sgl_kernel.<submodule>`` in the same pytest run.
        return
    def _make(name, pkg=False):
        mod = types.ModuleType(name)
        if pkg:
            mod.__path__ = []

        def _getattr(attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            return lambda *a, **k: None

        mod.__getattr__ = _getattr
        sys.modules.setdefault(name, mod)

    _make("sgl_kernel", pkg=True)
    _make("sgl_kernel.quantization")


_install_sgl_kernel_stub()

import torch  # noqa: E402

from sglang.srt.distributed.utils import (  # noqa: E402
    get_tp_partition_ratios,
    set_tp_partition_ratios,
    tp_loaded_shard_start,
    tp_partition_offset,
    tp_partition_sizes,
    tp_plan_active,
)
from sglang.srt.layers.linear import (  # noqa: E402
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

FP = torch.float32
BASE = [2, 1, 1]  # base plan (attention/KV etc.)
MLP = [3, 2, 2]  # deviating mlp family vector
TP = 3


def _full(rows, cols, offset=0):
    return torch.arange(rows * cols, dtype=FP).reshape(rows, cols) + offset


class FamilyPlanTestCase(CustomTestCase):
    def setUp(self):
        set_tp_partition_ratios(None)

    def tearDown(self):
        set_tp_partition_ratios(None)


class TestFamilyRegistry(FamilyPlanTestCase):
    def test_family_falls_back_to_base(self):
        # A family without its own vector uses the base plan, so passing
        # tp_family is always safe.
        set_tp_partition_ratios(BASE)
        self.assertEqual(get_tp_partition_ratios("mlp"), BASE)
        self.assertEqual(
            tp_partition_sizes(28, TP, 28, "mlp"),
            tp_partition_sizes(28, TP, 28),
        )
        self.assertTrue(tp_plan_active(TP, "mlp"))

    def test_family_overrides_base(self):
        set_tp_partition_ratios(BASE, families={"mlp": MLP})
        self.assertEqual(get_tp_partition_ratios(), BASE)
        self.assertEqual(get_tp_partition_ratios("mlp"), MLP)
        # 28 units: base (2,1,1) -> [14,7,7]; mlp (3,2,2) -> [12,8,8].
        self.assertEqual(tp_partition_sizes(28, TP, 28), [14, 7, 7])
        self.assertEqual(tp_partition_sizes(28, TP, 28, "mlp"), [12, 8, 8])

    def test_families_are_isolated(self):
        # An installed mlp vector must not leak into other families or
        # the base plan.
        set_tp_partition_ratios(BASE, families={"mlp": MLP})
        self.assertEqual(get_tp_partition_ratios("attn"), BASE)
        self.assertEqual(tp_partition_sizes(28, TP, 28, "attn"), [14, 7, 7])
        self.assertEqual(
            tp_partition_offset(28, TP, 1, 28), 14
        )  # base prefix sum
        self.assertEqual(
            tp_partition_offset(28, TP, 1, 28, "mlp"), 12
        )  # family prefix sum

    def test_no_plan_family_is_even_split(self):
        # Default path: no plan installed, family or not -> divide().
        self.assertIsNone(get_tp_partition_ratios("mlp"))
        self.assertEqual(tp_partition_sizes(30, TP, None, "mlp"), [10, 10, 10])
        self.assertFalse(tp_plan_active(TP, "mlp"))

    def test_family_requires_base_plan(self):
        with self.assertRaisesRegex(ValueError, "base"):
            set_tp_partition_ratios(None, families={"mlp": MLP})

    def test_family_length_must_match_base(self):
        with self.assertRaisesRegex(ValueError, "entries"):
            set_tp_partition_ratios(BASE, families={"mlp": [1, 2]})

    def test_family_entries_must_be_positive_ints(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            set_tp_partition_ratios(BASE, families={"mlp": [3, 0, 2]})
        with self.assertRaisesRegex(ValueError, "positive"):
            set_tp_partition_ratios(BASE, families={"mlp": [3, -1, 2]})

    def test_empty_family_vector_ignored(self):
        set_tp_partition_ratios(BASE, families={"mlp": None})
        self.assertEqual(get_tp_partition_ratios("mlp"), BASE)
        set_tp_partition_ratios(BASE, families={"mlp": []})
        self.assertEqual(get_tp_partition_ratios("mlp"), BASE)

    def test_reinstall_replaces_families(self):
        # Every set_tp_partition_ratios call replaces the COMPLETE plan.
        set_tp_partition_ratios(BASE, families={"mlp": MLP})
        set_tp_partition_ratios(BASE)
        self.assertEqual(get_tp_partition_ratios("mlp"), BASE)
        set_tp_partition_ratios(None)
        self.assertIsNone(get_tp_partition_ratios("mlp"))

    def test_family_mismatched_tp_size_falls_back_to_even(self):
        # Layers of a different tp_size keep the classic even split even
        # when a family vector is installed.
        set_tp_partition_ratios(BASE, families={"mlp": MLP})
        self.assertEqual(tp_partition_sizes(28, 1, None, "mlp"), [28])
        self.assertFalse(tp_plan_active(1, "mlp"))

    def test_loaded_shard_start_uses_family(self):
        set_tp_partition_ratios(BASE, families={"mlp": MLP})
        # 28 elements: base sizes [14,7,7], mlp sizes [12,8,8].
        self.assertEqual(tp_loaded_shard_start(28, TP, 1, 7, 28), 14)
        self.assertEqual(tp_loaded_shard_start(28, TP, 1, 8, 28, "mlp"), 12)
        # Family shard must match the FAMILY partition, not the base one.
        with self.assertRaisesRegex(ValueError, "shard mismatch"):
            tp_loaded_shard_start(28, TP, 1, 7, 28, "mlp")


class TestFamilyWeightLoading(FamilyPlanTestCase):
    """Loader consistency: shapes and checkpoint offsets of tp_family
    layers both follow the family vector (roundtrip with a family vector
    deviating from the base plan)."""

    INTERMEDIATE = 28  # mlp sizes [12, 8, 8]; base sizes [14, 7, 7]
    HIDDEN = 8

    def setUp(self):
        super().setUp()
        set_tp_partition_ratios(BASE, families={"mlp": MLP})

    def _mlp_sizes(self):
        return tp_partition_sizes(self.INTERMEDIATE, TP, self.INTERMEDIATE, "mlp")

    def test_merged_column_gate_up_roundtrip(self):
        # Qwen2MoeMLP.gate_up_proj: two packed outputs, family "mlp".
        gate = _full(self.INTERMEDIATE, self.HIDDEN)
        up = _full(self.INTERMEDIATE, self.HIDDEN, offset=1000)
        sizes = self._mlp_sizes()
        rebuilt_gate, rebuilt_up = [], []
        for rank in range(TP):
            layer = MergedColumnParallelLinear(
                self.HIDDEN,
                [self.INTERMEDIATE] * 2,
                bias=False,
                params_dtype=FP,
                tp_rank=rank,
                tp_size=TP,
                tp_units=self.INTERMEDIATE,
                tp_family="mlp",
            )
            # Shapes follow the FAMILY partition, not the base plan.
            self.assertEqual(layer.output_partition_sizes, [sizes[rank]] * 2)
            layer.weight_loader(layer.weight, gate, 0)
            layer.weight_loader(layer.weight, up, 1)
            rebuilt_gate.append(layer.weight.data[: sizes[rank]])
            rebuilt_up.append(layer.weight.data[sizes[rank] :])
        self.assertTrue(torch.equal(torch.cat(rebuilt_gate), gate))
        self.assertTrue(torch.equal(torch.cat(rebuilt_up), up))

    def test_row_parallel_down_roundtrip(self):
        # Qwen2MoeMLP.down_proj: input dimension follows the family.
        full = _full(self.HIDDEN, self.INTERMEDIATE)
        sizes = self._mlp_sizes()
        shards = []
        for rank in range(TP):
            layer = RowParallelLinear(
                self.INTERMEDIATE,
                self.HIDDEN,
                bias=False,
                params_dtype=FP,
                tp_rank=rank,
                tp_size=TP,
                tp_units=self.INTERMEDIATE,
                tp_family="mlp",
            )
            self.assertEqual(layer.input_size_per_partition, sizes[rank])
            layer.weight_loader(layer.weight, full)
            shards.append(layer.weight.data)
        self.assertTrue(torch.equal(torch.cat(shards, dim=1), full))

    def test_family_attr_propagated_to_parameters(self):
        layer = MergedColumnParallelLinear(
            self.HIDDEN,
            [self.INTERMEDIATE] * 2,
            bias=False,
            params_dtype=FP,
            tp_rank=1,
            tp_size=TP,
            tp_units=self.INTERMEDIATE,
            tp_family="mlp",
        )
        self.assertEqual(getattr(layer.weight, "tp_family", None), "mlp")
        self.assertEqual(
            getattr(layer.weight, "tp_units", None), self.INTERMEDIATE
        )

    def test_base_layer_unaffected_by_family_vector(self):
        # A layer WITHOUT tp_family keeps the base partition even while
        # the mlp vector is installed (isolation at the layer level).
        full = _full(self.INTERMEDIATE, self.HIDDEN)
        base_sizes = tp_partition_sizes(self.INTERMEDIATE, TP, self.INTERMEDIATE)
        self.assertEqual(base_sizes, [14, 7, 7])
        shards = []
        for rank in range(TP):
            layer = ColumnParallelLinear(
                self.HIDDEN,
                self.INTERMEDIATE,
                bias=False,
                params_dtype=FP,
                tp_rank=rank,
                tp_size=TP,
                tp_units=self.INTERMEDIATE,
            )
            self.assertEqual(layer.output_size_per_partition, base_sizes[rank])
            layer.weight_loader(layer.weight, full)
            shards.append(layer.weight.data)
        self.assertTrue(torch.equal(torch.cat(shards), full))

    def test_mixed_model_roundtrip(self):
        # Base-plan layer and family layer coexist in one process: each
        # reconstructs its full tensor under its OWN partition.
        base_full = _full(self.INTERMEDIATE, self.HIDDEN)
        mlp_full = _full(self.HIDDEN, self.INTERMEDIATE, offset=5000)
        base_shards, mlp_shards = [], []
        for rank in range(TP):
            base_layer = ColumnParallelLinear(
                self.HIDDEN,
                self.INTERMEDIATE,
                bias=False,
                params_dtype=FP,
                tp_rank=rank,
                tp_size=TP,
                tp_units=self.INTERMEDIATE,
            )
            mlp_layer = RowParallelLinear(
                self.INTERMEDIATE,
                self.HIDDEN,
                bias=False,
                params_dtype=FP,
                tp_rank=rank,
                tp_size=TP,
                tp_units=self.INTERMEDIATE,
                tp_family="mlp",
            )
            base_layer.weight_loader(base_layer.weight, base_full)
            mlp_layer.weight_loader(mlp_layer.weight, mlp_full)
            base_shards.append(base_layer.weight.data)
            mlp_shards.append(mlp_layer.weight.data)
        self.assertTrue(torch.equal(torch.cat(base_shards), base_full))
        self.assertTrue(torch.equal(torch.cat(mlp_shards, dim=1), mlp_full))

    def test_moe_src_start_uses_moe_family(self):
        # FusedMoE loads expert shards under the "moe" family: with an
        # installed moe vector the source offset follows it; without one
        # it falls back to the base plan; without any plan it is the
        # classic rank * shard_size.
        from types import SimpleNamespace

        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        fake = SimpleNamespace(
            moe_tp_size=TP, moe_tp_units=self.INTERMEDIATE, moe_tp_family="moe"
        )
        set_tp_partition_ratios(BASE, families={"moe": MLP})
        # moe (3,2,2) of 28 -> sizes [12,8,8], rank-1 offset 12.
        self.assertEqual(FusedMoE._moe_src_start(fake, 28, 8, 1), 12)
        set_tp_partition_ratios(BASE)
        # Fallback to base (2,1,1): sizes [14,7,7], rank-1 offset 14.
        self.assertEqual(FusedMoE._moe_src_start(fake, 28, 7, 1), 14)
        set_tp_partition_ratios(None)
        self.assertEqual(FusedMoE._moe_src_start(fake, 30, 10, 1), 10)

    def test_default_path_regression_with_family_kwarg(self):
        # No plan installed: tp_family layers behave exactly like the
        # classic even split (byte-identical default path).
        set_tp_partition_ratios(None)
        full = _full(30, self.HIDDEN)
        for rank in range(TP):
            layer = ColumnParallelLinear(
                self.HIDDEN,
                30,
                bias=False,
                params_dtype=FP,
                tp_rank=rank,
                tp_size=TP,
                tp_family="mlp",
            )
            self.assertEqual(layer.output_size_per_partition, 10)
            layer.weight_loader(layer.weight, full)
            self.assertTrue(
                torch.equal(layer.weight.data, full[rank * 10 : (rank + 1) * 10])
            )
            # No plan -> no weight attrs.
            self.assertIsNone(getattr(layer.weight, "tp_family", None))


if __name__ == "__main__":
    unittest.main()
