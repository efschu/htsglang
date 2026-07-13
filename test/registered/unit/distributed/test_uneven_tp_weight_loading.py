"""CPU unit tests for uneven-TP weight loading (phase 2).

Drives the REAL weight_loader methods of ColumnParallelLinear,
MergedColumnParallelLinear, QKVParallelLinear and RowParallelLinear (plus
the v2 loaders in layers/parameter.py, the mamba conv loader, the vocab
embedding padding and model_config.get_num_kv_heads) with dummy CPU
tensors. Core property: for a 3-rank shard plan every rank loads its
shard from one full tensor and the concatenation of the shards must
reconstruct the full tensor exactly (roundtrip). The default path (no
plan installed) must behave exactly like the classic rank * shard_size
split.

No GPU, no distributed init: the layer classes accept explicit
tp_rank/tp_size, and `sgl_kernel` (not installed in the CPU test env) is
stubbed out before the sglang imports.
"""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch


def _install_sgl_kernel_stub():
    """The quantization package imports sgl_kernel at module scope; the
    CPU test environment does not have it. Provide inert stubs."""

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
    set_tp_partition_ratios,
    tp_loaded_shard_start,
    tp_partition_sizes,
)
from sglang.srt.layers.attention.mamba.mamba import (  # noqa: E402
    mamba_v2_sharded_weight_loader,
)
from sglang.srt.layers.linear import (  # noqa: E402
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.parameter import ModelWeightParameter  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

FP = torch.float32
PLAN = [2, 1, 1]  # 3 ranks, rank 0 twice as large
TP = len(PLAN)


def _full(rows, cols, offset=0):
    return torch.arange(rows * cols, dtype=FP).reshape(rows, cols) + offset


class UnevenTPTestCase(CustomTestCase):
    def setUp(self):
        set_tp_partition_ratios(None)

    def tearDown(self):
        set_tp_partition_ratios(None)


class TestTpLoadedShardStart(UnevenTPTestCase):
    def test_no_plan_is_classic_formula(self):
        self.assertEqual(tp_loaded_shard_start(12, 3, 2, 4), 8)
        self.assertEqual(tp_loaded_shard_start(12, None, 1, 4), 4)

    def test_plan_size_mismatch_falls_back(self):
        set_tp_partition_ratios(PLAN)
        # Layer runs with tp_size=2 (plan has 3 entries) -> classic.
        self.assertEqual(tp_loaded_shard_start(12, 2, 1, 6), 6)

    def test_plan_prefix_sums(self):
        set_tp_partition_ratios(PLAN)
        # 16 elements over (2,1,1): sizes [8,4,4], offsets [0,8,12].
        self.assertEqual(tp_loaded_shard_start(16, 3, 0, 8), 0)
        self.assertEqual(tp_loaded_shard_start(16, 3, 1, 4), 8)
        self.assertEqual(tp_loaded_shard_start(16, 3, 2, 4), 12)

    def test_replicated_component_offset_zero(self):
        set_tp_partition_ratios(PLAN)
        self.assertEqual(tp_loaded_shard_start(16, 3, 2, 16), 0)

    def test_shard_size_mismatch_raises(self):
        set_tp_partition_ratios(PLAN)
        with self.assertRaisesRegex(ValueError, "shard mismatch"):
            tp_loaded_shard_start(16, 3, 1, 5)


class TestColumnParallelLinear(UnevenTPTestCase):
    def _make(self, rank, out=16, inp=8, **kw):
        return ColumnParallelLinear(
            inp, out, bias=False, params_dtype=FP, tp_rank=rank, tp_size=TP, **kw
        )

    def test_default_path_regression(self):
        # No plan: shapes and loaded slices exactly rank * shard_size.
        full = _full(15, 8)
        for rank in range(3):
            layer = self._make(rank, out=15)
            self.assertEqual(layer.output_size_per_partition, 5)
            layer.weight_loader(layer.weight, full)
            self.assertTrue(
                torch.equal(layer.weight.data, full[rank * 5 : (rank + 1) * 5])
            )

    def test_uneven_roundtrip(self):
        set_tp_partition_ratios(PLAN)
        full = _full(16, 8)
        shards = []
        for rank in range(3):
            layer = self._make(rank)
            layer.weight_loader(layer.weight, full)
            shards.append(layer.weight.data.clone())
        self.assertEqual([s.shape[0] for s in shards], [8, 4, 4])
        self.assertTrue(torch.equal(torch.cat(shards, dim=0), full))

    def test_uneven_indivisible_dim_raises_with_dimension_and_weights(self):
        set_tp_partition_ratios(PLAN)
        with self.assertRaisesRegex(ValueError, r"size 15.*\[2, 1, 1\]"):
            self._make(0, out=15)

    def test_uneven_bias_roundtrip(self):
        set_tp_partition_ratios(PLAN)
        full_bias = torch.arange(16, dtype=FP)
        parts = []
        for rank in range(3):
            layer = ColumnParallelLinear(
                8, 16, bias=True, params_dtype=FP, tp_rank=rank, tp_size=TP
            )
            layer.weight_loader(layer.bias, full_bias)
            parts.append(layer.bias.data.clone())
        self.assertTrue(torch.equal(torch.cat(parts), full_bias))


class TestMergedColumnParallelLinear(UnevenTPTestCase):
    OUT_SIZES = [12, 24]

    def _make(self, rank, **kw):
        return MergedColumnParallelLinear(
            8,
            list(self.OUT_SIZES),
            bias=False,
            params_dtype=FP,
            tp_rank=rank,
            tp_size=TP,
            **kw,
        )

    def _roundtrip(self):
        full0 = _full(self.OUT_SIZES[0], 8)
        full1 = _full(self.OUT_SIZES[1], 8, offset=1000)
        parts0, parts1 = [], []
        for rank in range(3):
            layer = self._make(rank)
            layer.weight_loader(layer.weight, full0, 0)
            layer.weight_loader(layer.weight, full1, 1)
            split = layer.output_partition_sizes
            parts0.append(layer.weight.data[: split[0]].clone())
            parts1.append(layer.weight.data[split[0] :].clone())
        self.assertTrue(torch.equal(torch.cat(parts0), full0))
        self.assertTrue(torch.equal(torch.cat(parts1), full1))

    def test_default_path_regression(self):
        # Even split: per-part shard == out // 3 at offset rank * shard.
        full0 = _full(self.OUT_SIZES[0], 8)
        layer = self._make(1)
        layer.weight_loader(layer.weight, full0, 0)
        self.assertTrue(torch.equal(layer.weight.data[:4], full0[4:8]))

    def test_uneven_roundtrip_per_component(self):
        set_tp_partition_ratios(PLAN)
        self._roundtrip()

    def test_uneven_fused_on_disk_roundtrip(self):
        # loaded_shard_id=None: fused checkpoint split into components.
        set_tp_partition_ratios(PLAN)
        full0 = _full(self.OUT_SIZES[0], 8)
        full1 = _full(self.OUT_SIZES[1], 8, offset=1000)
        fused = torch.cat([full0, full1], dim=0)
        parts0, parts1 = [], []
        for rank in range(3):
            layer = self._make(rank)
            layer.weight_loader(layer.weight, fused, None)
            split = layer.output_partition_sizes
            parts0.append(layer.weight.data[: split[0]].clone())
            parts1.append(layer.weight.data[split[0] :].clone())
        self.assertTrue(torch.equal(torch.cat(parts0), full0))
        self.assertTrue(torch.equal(torch.cat(parts1), full1))

    def test_uneven_indivisible_component_raises(self):
        set_tp_partition_ratios(PLAN)
        with self.assertRaisesRegex(ValueError, r"size 10.*\[2, 1, 1\]"):
            MergedColumnParallelLinear(
                8, [12, 10], bias=False, params_dtype=FP, tp_rank=0, tp_size=TP
            )

    def test_uneven_units_partition(self):
        # tp_units=4 makes the otherwise indivisible [12, 12] with plan
        # (3,2,2) partitionable in whole units of 3 rows.
        set_tp_partition_ratios([3, 2, 2])
        full0 = _full(12, 8)
        full1 = _full(12, 8, offset=1000)
        parts0, parts1 = [], []
        for rank in range(3):
            layer = MergedColumnParallelLinear(
                8,
                [12, 12],
                bias=False,
                params_dtype=FP,
                tp_rank=rank,
                tp_size=TP,
                tp_units=4,
            )
            layer.weight_loader(layer.weight, full0, 0)
            layer.weight_loader(layer.weight, full1, 1)
            split = layer.output_partition_sizes
            parts0.append(layer.weight.data[: split[0]].clone())
            parts1.append(layer.weight.data[split[0] :].clone())
        self.assertEqual([p.shape[0] for p in parts0], [6, 3, 3])
        self.assertTrue(torch.equal(torch.cat(parts0), full0))
        self.assertTrue(torch.equal(torch.cat(parts1), full1))


class TestQKVParallelLinear(UnevenTPTestCase):
    HIDDEN = 16
    HEAD = 4
    Q_HEADS = 8
    KV_HEADS = 4

    def _make(self, rank, **kw):
        return QKVParallelLinear(
            self.HIDDEN,
            self.HEAD,
            self.Q_HEADS,
            self.KV_HEADS,
            bias=False,
            params_dtype=FP,
            tp_rank=rank,
            tp_size=TP,
            **kw,
        )

    def _roundtrip(self, expect_heads):
        full_q = _full(self.Q_HEADS * self.HEAD, self.HIDDEN)
        full_k = _full(self.KV_HEADS * self.HEAD, self.HIDDEN, offset=10_000)
        full_v = _full(self.KV_HEADS * self.HEAD, self.HIDDEN, offset=20_000)
        qs, ks, vs, heads = [], [], [], []
        for rank in range(3):
            layer = self._make(rank)
            heads.append((layer.num_heads, layer.num_kv_heads))
            layer.weight_loader(layer.weight, full_q, "q")
            layer.weight_loader(layer.weight, full_k, "k")
            layer.weight_loader(layer.weight, full_v, "v")
            d = layer.weight.data
            q_end = layer.num_heads * self.HEAD
            k_end = q_end + layer.num_kv_heads * self.HEAD
            qs.append(d[:q_end].clone())
            ks.append(d[q_end:k_end].clone())
            vs.append(d[k_end:].clone())
        self.assertEqual(heads, expect_heads)
        self.assertTrue(torch.equal(torch.cat(qs), full_q))
        self.assertTrue(torch.equal(torch.cat(ks), full_k))
        self.assertTrue(torch.equal(torch.cat(vs), full_v))

    def test_uneven_gqa_roundtrip_exact_plan(self):
        set_tp_partition_ratios(PLAN)
        # kv units (2,1,1) of 4 -> [2,1,1]; q heads follow -> [4,2,2].
        self._roundtrip([(4, 2), (2, 1), (2, 1)])

    def test_uneven_gqa_roundtrip_unit_rounding(self):
        # sum(ratios)=7 does NOT divide any dim: kv heads are rounded as
        # whole units, q heads follow the same unit distribution.
        set_tp_partition_ratios([3, 2, 2])
        self._roundtrip([(4, 2), (2, 1), (2, 1)])

    def test_uneven_bias_roundtrip(self):
        set_tp_partition_ratios(PLAN)
        full_bq = torch.arange(self.Q_HEADS * self.HEAD, dtype=FP)
        full_bk = torch.arange(self.KV_HEADS * self.HEAD, dtype=FP) + 500
        full_bv = torch.arange(self.KV_HEADS * self.HEAD, dtype=FP) + 900
        qs, ks, vs = [], [], []
        for rank in range(3):
            layer = QKVParallelLinear(
                self.HIDDEN,
                self.HEAD,
                self.Q_HEADS,
                self.KV_HEADS,
                bias=True,
                params_dtype=FP,
                tp_rank=rank,
                tp_size=TP,
            )
            layer.weight_loader(layer.bias, full_bq, "q")
            layer.weight_loader(layer.bias, full_bk, "k")
            layer.weight_loader(layer.bias, full_bv, "v")
            b = layer.bias.data
            q_end = layer.num_heads * self.HEAD
            k_end = q_end + layer.num_kv_heads * self.HEAD
            qs.append(b[:q_end].clone())
            ks.append(b[q_end:k_end].clone())
            vs.append(b[k_end:].clone())
        self.assertTrue(torch.equal(torch.cat(qs), full_bq))
        self.assertTrue(torch.equal(torch.cat(ks), full_bk))
        self.assertTrue(torch.equal(torch.cat(vs), full_bv))

    def test_uneven_rejects_kv_replication(self):
        # tp=3 > 2 kv heads: replication is refused under a plan.
        set_tp_partition_ratios(PLAN)
        with self.assertRaisesRegex(ValueError, "replication is not supported"):
            QKVParallelLinear(
                self.HIDDEN,
                self.HEAD,
                6,
                2,
                bias=False,
                params_dtype=FP,
                tp_rank=0,
                tp_size=TP,
            )

    def test_uneven_rejects_non_integral_gqa_groups(self):
        set_tp_partition_ratios(PLAN)
        with self.assertRaisesRegex(ValueError, "multiple of total_num_kv_heads"):
            QKVParallelLinear(
                self.HIDDEN,
                self.HEAD,
                10,
                4,
                bias=False,
                params_dtype=FP,
                tp_rank=0,
                tp_size=TP,
            )

    def test_default_path_regression_gqa(self):
        # No plan: kv 4 < tp 3 is not the case here (4 >= 3 fails the
        # divide) -> use tp-divisible totals and verify classic offsets.
        full_k = _full(self.KV_HEADS * self.HEAD, self.HIDDEN)
        layer = QKVParallelLinear(
            self.HIDDEN,
            self.HEAD,
            8,
            4,
            bias=False,
            params_dtype=FP,
            tp_rank=1,
            tp_size=4,
        )
        self.assertEqual((layer.num_heads, layer.num_kv_heads), (2, 1))
        layer.weight_loader(layer.weight, full_k, "k")
        q_end = layer.num_heads * self.HEAD
        k_end = q_end + layer.num_kv_heads * self.HEAD
        self.assertTrue(
            torch.equal(layer.weight.data[q_end:k_end], full_k[4:8])
        )

    def test_default_path_regression_kv_replication(self):
        # No plan: tp 4 > kv 2 -> replicas; ranks 0/1 share kv shard 0.
        full_k = _full(2 * self.HEAD, self.HIDDEN)
        datas = []
        for rank in range(4):
            layer = QKVParallelLinear(
                self.HIDDEN,
                self.HEAD,
                8,
                2,
                bias=False,
                params_dtype=FP,
                tp_rank=rank,
                tp_size=4,
            )
            self.assertEqual(layer.num_kv_head_replicas, 2)
            layer.weight_loader(layer.weight, full_k, "k")
            q_end = layer.num_heads * self.HEAD
            k_end = q_end + layer.num_kv_heads * self.HEAD
            datas.append(layer.weight.data[q_end:k_end].clone())
        self.assertTrue(torch.equal(datas[0], datas[1]))
        self.assertTrue(torch.equal(datas[2], datas[3]))
        self.assertTrue(torch.equal(datas[0], full_k[:4]))
        self.assertTrue(torch.equal(datas[2], full_k[4:]))


class TestRowParallelLinear(UnevenTPTestCase):
    def test_default_path_regression(self):
        full = _full(8, 12)
        layer = RowParallelLinear(
            12, 8, bias=False, params_dtype=FP, tp_rank=2, tp_size=TP
        )
        layer.weight_loader(layer.weight, full)
        self.assertTrue(torch.equal(layer.weight.data, full[:, 8:12]))

    def test_uneven_roundtrip(self):
        set_tp_partition_ratios(PLAN)
        full = _full(8, 12)
        parts = []
        for rank in range(3):
            layer = RowParallelLinear(
                12, 8, bias=False, params_dtype=FP, tp_rank=rank, tp_size=TP
            )
            layer.weight_loader(layer.weight, full)
            parts.append(layer.weight.data.clone())
        self.assertEqual([p.shape[1] for p in parts], [6, 3, 3])
        self.assertTrue(torch.equal(torch.cat(parts, dim=1), full))

    def test_uneven_roundtrip_with_units(self):
        # o_proj-like: input dim = 8 q heads * 4, kv-head units = 4.
        set_tp_partition_ratios([3, 2, 2])
        full = _full(8, 32)
        parts = []
        for rank in range(3):
            layer = RowParallelLinear(
                32, 8, bias=False, params_dtype=FP, tp_rank=rank, tp_size=TP,
                tp_units=4,
            )
            layer.weight_loader(layer.weight, full)
            parts.append(layer.weight.data.clone())
        self.assertEqual([p.shape[1] for p in parts], [16, 8, 8])
        self.assertTrue(torch.equal(torch.cat(parts, dim=1), full))

    def test_uneven_indivisible_raises(self):
        set_tp_partition_ratios(PLAN)
        with self.assertRaisesRegex(ValueError, r"size 10.*\[2, 1, 1\]"):
            RowParallelLinear(
                10, 8, bias=False, params_dtype=FP, tp_rank=0, tp_size=TP
            )


class TestParameterV2Loaders(UnevenTPTestCase):
    """Drive the parameter-class (weight_loader_v2) loaders directly."""

    def _param(self, rows, cols, tp_units=None):
        p = ModelWeightParameter(
            data=torch.empty(rows, cols, dtype=FP),
            input_dim=1,
            output_dim=0,
            weight_loader=lambda *a, **k: None,
        )
        if tp_units is not None:
            p.tp_units = tp_units
        return p

    def test_column_parallel_v2_roundtrip(self):
        set_tp_partition_ratios(PLAN)
        full = _full(16, 8)
        sizes = tp_partition_sizes(16, TP)
        parts = []
        for rank in range(3):
            p = self._param(sizes[rank], 8)
            p.load_column_parallel_weight(full, tp_rank=rank)
            parts.append(p.data.clone())
        self.assertTrue(torch.equal(torch.cat(parts), full))

    def test_row_parallel_v2_roundtrip(self):
        set_tp_partition_ratios(PLAN)
        full = _full(8, 16)
        sizes = tp_partition_sizes(16, TP)
        parts = []
        for rank in range(3):
            p = ModelWeightParameter(
                data=torch.empty(8, sizes[rank], dtype=FP),
                input_dim=1,
                output_dim=0,
                weight_loader=lambda *a, **k: None,
            )
            p.load_row_parallel_weight(full, tp_rank=rank)
            parts.append(p.data.clone())
        self.assertTrue(torch.equal(torch.cat(parts, dim=1), full))

    def test_merged_column_v2_roundtrip(self):
        set_tp_partition_ratios(PLAN)
        full0 = _full(12, 8)
        full1 = _full(20, 8, offset=1000)
        sizes0 = tp_partition_sizes(12, TP)
        sizes1 = tp_partition_sizes(20, TP)
        parts0, parts1 = [], []
        for rank in range(3):
            p = self._param(sizes0[rank] + sizes1[rank], 8)
            p.load_merged_column_weight(
                full0, shard_offset=0, shard_size=sizes0[rank], tp_rank=rank,
                use_presharded_weights=False,
            )
            p.load_merged_column_weight(
                full1, shard_offset=sizes0[rank], shard_size=sizes1[rank],
                tp_rank=rank, use_presharded_weights=False,
            )
            parts0.append(p.data[: sizes0[rank]].clone())
            parts1.append(p.data[sizes0[rank] :].clone())
        self.assertTrue(torch.equal(torch.cat(parts0), full0))
        self.assertTrue(torch.equal(torch.cat(parts1), full1))

    def test_qkv_v2_roundtrip_with_units(self):
        # k component of a GQA layer: 4 kv heads * head 4, units=4.
        set_tp_partition_ratios([3, 2, 2])
        full_k = _full(16, 8)
        sizes = tp_partition_sizes(16, TP, units=4)
        parts = []
        for rank in range(3):
            p = self._param(sizes[rank], 8, tp_units=4)
            p.load_qkv_weight(
                full_k, tp_rank=rank, shard_offset=0, shard_size=sizes[rank],
                shard_id="k", num_heads=1,
            )
            parts.append(p.data.clone())
        self.assertEqual([p.shape[0] for p in parts], [8, 4, 4])
        self.assertTrue(torch.equal(torch.cat(parts), full_k))

    def test_default_v2_regression(self):
        full = _full(15, 8)
        p = self._param(5, 8)
        p.load_column_parallel_weight(full, tp_rank=2)
        self.assertTrue(torch.equal(p.data, full[10:15]))


class TestMambaV2ShardedLoader(UnevenTPTestCase):
    """GDN conv1d loader: groups (K, K, V) with k-head units."""

    K_HEADS = 4
    HEAD_K = 3
    HEAD_V = 6
    V_HEADS = 8  # 2 v heads per k head

    def _roundtrip(self, plan, expect_k_rows):
        key_dim = self.K_HEADS * self.HEAD_K  # 12
        value_dim = self.V_HEADS * self.HEAD_V  # 48
        set_tp_partition_ratios(plan)
        full = torch.arange(2 * key_dim + value_dim, dtype=FP).unsqueeze(1)
        spec = [(key_dim, 0, False), (key_dim, 0, False), (value_dim, 0, False)]
        k1s, k2s, vs = [], [], []
        for rank in range(3):
            sizes_k = tp_partition_sizes(key_dim, TP, units=self.K_HEADS)
            sizes_v = tp_partition_sizes(value_dim, TP, units=self.K_HEADS)
            loader = mamba_v2_sharded_weight_loader(
                spec, TP, rank, tp_units=self.K_HEADS
            )
            param = torch.nn.Parameter(
                torch.empty(sizes_k[rank] * 2 + sizes_v[rank], 1, dtype=FP)
            )
            loader(param, full)
            k1s.append(param.data[: sizes_k[rank]].clone())
            k2s.append(param.data[sizes_k[rank] : 2 * sizes_k[rank]].clone())
            vs.append(param.data[2 * sizes_k[rank] :].clone())
        self.assertEqual([t.shape[0] for t in k1s], expect_k_rows)
        rec = torch.cat(
            [torch.cat(k1s), torch.cat(k2s), torch.cat(vs)], dim=0
        )
        self.assertTrue(torch.equal(rec, full))

    def test_uneven_roundtrip(self):
        self._roundtrip(PLAN, [6, 3, 3])

    def test_uneven_roundtrip_unit_rounding(self):
        self._roundtrip([3, 2, 2], [6, 3, 3])

    def test_default_regression(self):
        # No plan: classic full_dim // tp split, bit-identical.
        key_dim = 12
        value_dim = 48
        full = torch.arange(2 * key_dim + value_dim, dtype=FP).unsqueeze(1)
        spec = [(key_dim, 0, False), (key_dim, 0, False), (value_dim, 0, False)]
        loader = mamba_v2_sharded_weight_loader(spec, TP, 1, tp_units=self.K_HEADS)
        param = torch.nn.Parameter(torch.empty(24, 1, dtype=FP))
        loader(param, full)
        expected = torch.cat(
            [full[4:8], full[12 + 4 : 12 + 8], full[24 + 16 : 24 + 32]], dim=0
        )
        self.assertTrue(torch.equal(param.data, expected))


class TestShardedWeightLoader(UnevenTPTestCase):
    """weight_utils.sharded_weight_loader (dt_bias / A_log path)."""

    def _run(self, plan, tp_units, rank, param_rows, full_rows):
        import sglang.srt.model_loader.weight_utils as wu

        set_tp_partition_ratios(plan)
        full = torch.arange(full_rows, dtype=FP)
        param = torch.nn.Parameter(torch.empty(param_rows, dtype=FP))
        if tp_units is not None:
            param.tp_units = tp_units
        fake_parallel = SimpleNamespace(attn_tp_rank=rank, attn_tp_size=TP, tp_size=TP)
        with patch.object(wu, "get_parallel", return_value=fake_parallel):
            wu.sharded_weight_loader(0)(param, full)
        return param.data.clone(), full

    def test_uneven_roundtrip(self):
        # 8 v heads in k-head units of 4 over (3,2,2) -> [4,2,2].
        parts = []
        for rank, rows in enumerate([4, 2, 2]):
            data, full = self._run([3, 2, 2], 4, rank, rows, 8)
            parts.append(data)
        self.assertTrue(torch.equal(torch.cat(parts), full))

    def test_default_regression(self):
        data, full = self._run(None, 4, 2, 2, 6)
        self.assertTrue(torch.equal(data, full[4:6]))


class TestVocabParallelEmbedding(UnevenTPTestCase):
    def _make(self, rank, vocab=1000, dim=8, plan=PLAN, tp_size=TP):
        import sglang.srt.layers.vocab_parallel_embedding as vpe

        set_tp_partition_ratios(plan)
        fake_parallel = SimpleNamespace(tp_rank=rank, tp_size=tp_size)
        with patch.object(vpe, "get_parallel", return_value=fake_parallel):
            return vpe.VocabParallelEmbedding(vocab, dim, params_dtype=FP)

    def test_uneven_pads_to_lcm_and_stays_even(self):
        # 1000 padded to a multiple of lcm(64, 3) = 192 -> 1152; the
        # vocab dimension deliberately keeps the EVEN split.
        layers = [self._make(rank) for rank in range(3)]
        self.assertEqual(layers[0].num_embeddings_padded, 1152)
        for layer in layers:
            self.assertEqual(layer.num_embeddings_per_partition, 384)

    def test_uneven_weight_roundtrip(self):
        full = _full(1000, 8)
        org_parts = []
        for rank in range(3):
            layer = self._make(rank)
            layer.weight_loader(layer.weight, full)
            n_org = (
                layer.shard_indices.org_vocab_end_index
                - layer.shard_indices.org_vocab_start_index
            )
            org_parts.append(layer.weight.data[:n_org].clone())
        self.assertTrue(torch.equal(torch.cat(org_parts), full))

    def test_default_padding_unchanged(self):
        # No plan (tp=4 divides the padded vocab): padding stays 64 ->
        # 1024 (regression guard: no lcm padding on the default path).
        layer = self._make(0, plan=None, tp_size=4)
        self.assertEqual(layer.padding_size, 64)
        self.assertEqual(layer.num_embeddings_padded, 1024)
        self.assertEqual(layer.num_embeddings_per_partition, 256)


class TestModelConfigNumKvHeads(UnevenTPTestCase):
    def _mc(self, total_kv=8):
        from sglang.srt.configs.model_config import ModelConfig

        mc = ModelConfig.__new__(ModelConfig)
        mc.hf_config = SimpleNamespace(model_type="llama")
        mc.hf_text_config = SimpleNamespace(num_key_value_heads=total_kv)
        return mc

    def test_default_unchanged(self):
        mc = self._mc()
        self.assertEqual(mc.get_num_kv_heads(4), 2)
        self.assertEqual(mc.get_num_kv_heads(16), 1)  # replication floor

    def test_uneven_rank_aware(self):
        set_tp_partition_ratios(PLAN)
        mc = self._mc()
        self.assertEqual([mc.get_num_kv_heads(TP, rank=r) for r in range(3)], [4, 2, 2])
        # Without a rank: the smallest share (conservative basis).
        self.assertEqual(mc.get_num_kv_heads(TP), 2)

    def test_uneven_unit_rounding(self):
        set_tp_partition_ratios([3, 2, 2])
        mc = self._mc()
        self.assertEqual([mc.get_num_kv_heads(TP, rank=r) for r in range(3)], [4, 2, 2])

    def test_plan_of_other_tp_size_ignored(self):
        set_tp_partition_ratios(PLAN)
        mc = self._mc()
        self.assertEqual(mc.get_num_kv_heads(4), 2)

    def test_swa_uneven_rank_aware(self):
        set_tp_partition_ratios(PLAN)
        mc = self._mc()
        mc.hf_text_config.swa_num_key_value_heads = 4
        self.assertEqual(
            [mc.get_swa_num_kv_heads(TP, rank=r) for r in range(3)], [2, 1, 1]
        )


if __name__ == "__main__":
    unittest.main()
