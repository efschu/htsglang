"""Repro + inertness proof for the GPTQ-Marlin small-output-layer fix (#316).

Behind the #300 unit-count fix, the same GPTQ arm of the #274 battery
(Qwen3.6-27B-...-Native-MTP-Preserved-GPTQ-Int4, TP=3, --rank-tp-ratio
auto-performance -> [29607, 17780, 17780]) loaded all six shards and then died
in `process_weights_after_loading`:

    gptq_marlin_repack.cuh:309: size_n = 30 is not divisible by tile_n_size=64  (TP1)
    gptq_marlin_repack.cuh:309: size_n = 24 is not divisible by tile_n_size=64  (TP2)

The module is the gated-delta-net's merged b/a projection,
`linear_attn.in_proj_ba` = 5120 -> 2 x `linear_num_value_heads` = 2 x 48 = 96
outputs, partitioned in whole k-head units (`gdn_tp_units` = 16). The unit
split [7, 5, 4] scales by 48/16 = 3 elements per part, i.e. [21, 15, 12] per
part and [42, 30, 24] per rank -- the two numbers in the traceback, with rank 0
at 42 (equally invalid; the log only caught two ranks before the sigquit).

This is NOT the #289/#300 failure mode even though it surfaced from the same
battery run. Those were split-granularity bugs, curable by coarsening the unit
family to the quant block. Here nothing is curable by splitting differently:
96 is not a multiple of Marlin's `min_thread_n` = 64, so no shard of it is
either -- not the uneven [42, 30, 24], not the even TP=3 32, not replication
(96) and not TP=1 (96). The layer is simply not Marlin-packable, which is why
every quantizer leaves it dense: the AWQ (auto-round) and FP8 siblings of the
same base model list `linear_attn.in_proj_a` / `in_proj_b` (FP8 also the fused
`in_proj_ba`) in `modules_to_not_convert`, and the fork's GGUF loader carves it
out as F32. GPTQModel writes no ignore list at all, so sglang built a Marlin
layer whose `qweight` no checkpoint tensor ever reaches and whose repack then
aborts.

The fix gives `GPTQMarlinConfig.get_quant_method` the shape guard
`AWQMarlinConfig` has always had, judged on the UNSHARDED geometry so a bad
SHARD still fails loudly, and resolving to `UnquantizedLinearMethod` rather
than to the exllama kernel (which would build a `qweight` for a dense-on-disk
module -- silent garbage instead of a hard abort).

Plan-time arithmetic plus real layer construction; no GPU, no server.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest
from unittest import mock

import torch

from sglang.srt.distributed.utils import (
    partition_sizes,
    scoped_tp_partition_ratios,
    tp_partition_sizes,
)
from sglang.srt.layers.linear import (
    LinearBase,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.gptq.gptq import (
    GPTQMarlinConfig,
    GPTQMarlinLinearMethod,
    gptq_marlin_unpackable_reason,
)
from sglang.srt.layers.quantization.gptq.schemes.gptq_marlin import (
    GPTQMarlinLinearScheme,
)
from sglang.srt.layers.quantization.marlin_utils import (
    GPTQ_MARLIN_MIN_THREAD_K,
    GPTQ_MARLIN_MIN_THREAD_N,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.test.test_utils import CustomTestCase

# Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4.
HIDDEN = 5120
GROUP_SIZE = 128
INTERMEDIATE = 17408
# GDN geometry (config.text_config).
V_HEADS, K_HEADS = 48, 16
KEY_DIM, VALUE_DIM = 2048, 6144
GDN_UNITS = K_HEADS
# Full attention: 24 q heads x 256 x 2 (attn_output_gate), 4 kv heads x 256.
Q_DIM, KV_DIM, Q_HEADS, KV_HEADS = 12288, 1024, 24, 4
# The weight vector the failing boot installed.
R7B_WEIGHTS = [29607, 17780, 17780]
# The per-rank output shards of in_proj_ba the traceback reported.
BA_SHARDS = [42, 30, 24]
# #77's validated GPTQ MoE checkpoints: every dense module is a negative
# `dynamic` match, so only the expert weights are quantized.
MOE_DYNAMIC = {
    "lm_head": {},
    "model.language_model.embed_tokens": {},
    "-:.*attn.*": {},
    "-:.*shared_expert.*": {},
    "-:.*mtp.*": {},
    "-:.*visual.*": {},
}


def _marlin_config(group_size: int = GROUP_SIZE, dynamic=None) -> GPTQMarlinConfig:
    return GPTQMarlinConfig(
        weight_bits=4,
        group_size=group_size,
        desc_act=False,
        is_sym=True,
        lm_head_quantized=False,
        dynamic=dynamic or {},
        full_config={},
    )


def _stub_linear(input_size: int, output_size: int) -> LinearBase:
    """A LinearBase carrying only the two sizes `get_quant_method` may read.

    Built without `__init__` on purpose: at the point the real
    `LinearBase.__init__` asks the config for a method, exactly these two
    attributes are set and nothing else is (`output_size_per_partition` is
    computed by `ColumnParallelLinear` only after the super() call returns).
    """
    layer = LinearBase.__new__(LinearBase)
    layer.input_size = input_size
    layer.output_size = output_size
    return layer


# Every quantized module of the checkpoint, as (name, in, out). Row-parallel
# modules list their full input dim; the guard reads unsharded sizes only.
QUANTIZED_MODULES = (
    ("linear_attn.in_proj_qkvz", HIDDEN, 2 * KEY_DIM + 2 * VALUE_DIM),
    ("linear_attn.out_proj", VALUE_DIM, HIDDEN),
    ("self_attn.qkv_proj", HIDDEN, Q_DIM + 2 * KV_DIM),
    ("self_attn.o_proj", Q_HEADS * 256, HIDDEN),
    ("mlp.gate_up_proj", HIDDEN, 2 * INTERMEDIATE),
    ("mlp.down_proj", INTERMEDIATE, HIDDEN),
)


class TestInProjBaGeometry(CustomTestCase):
    """The arithmetic that produced 24 and 30, and why no split repairs it."""

    def test_the_reported_shards_are_reproduced_exactly(self):
        # in_proj_ba is a MergedColumnParallelLinear of two num_v_heads-wide
        # parts, each partitioned in whole k-head units.
        per_part = partition_sizes(V_HEADS, R7B_WEIGHTS, GDN_UNITS)
        self.assertEqual(per_part, [21, 15, 12])
        self.assertEqual([2 * p for p in per_part], BA_SHARDS)
        # The traceback's two values, with the rank they were raised on.
        self.assertEqual(BA_SHARDS[1], 30)
        self.assertEqual(BA_SHARDS[2], 24)

    def test_the_split_is_the_same_one_every_other_gdn_tensor_uses(self):
        # 16 k-head units is what qkv/z, conv1d and the state caches use; the
        # b/a scalars must stay on that split or the per-rank v heads and
        # their decay/gate scalars drift apart.
        self.assertEqual(partition_sizes(KEY_DIM, R7B_WEIGHTS, GDN_UNITS)[1], 640)
        self.assertEqual(partition_sizes(VALUE_DIM, R7B_WEIGHTS, GDN_UNITS)[1], 1920)
        self.assertEqual(partition_sizes(V_HEADS, R7B_WEIGHTS, GDN_UNITS)[1], 15)

    def test_no_shard_of_96_can_ever_meet_the_marlin_tile(self):
        # Uneven, even, replicated and single-rank -- all of them miss it, so
        # neither a coarser unit family (#289/#300) nor replication (#62) is a
        # candidate fix here.
        candidates = list(BA_SHARDS) + [2 * V_HEADS // 3, 2 * V_HEADS, V_HEADS]
        for size in candidates:
            self.assertNotEqual(size % GPTQ_MARLIN_MIN_THREAD_N, 0, f"{size=}")

    def test_coarsening_the_unit_family_cannot_help(self):
        # Any coarsening divides 48 into fewer, larger units; the LARGEST
        # possible unit is the whole part, and 2 x 48 is still not 64-aligned.
        for units in (u for u in range(1, V_HEADS + 1) if V_HEADS % u == 0):
            whole = 2 * V_HEADS
            self.assertNotEqual(whole % GPTQ_MARLIN_MIN_THREAD_N, 0)
            if units >= len(R7B_WEIGHTS):
                shards = partition_sizes(V_HEADS, R7B_WEIGHTS, units)
                self.assertTrue(
                    all(2 * s % GPTQ_MARLIN_MIN_THREAD_N != 0 for s in shards),
                    f"{units=} {shards=}",
                )


class TestUnpackableReason(CustomTestCase):
    """The guard itself: fires on in_proj_ba, on nothing else."""

    def test_in_proj_ba_is_reported_unpackable(self):
        reason = gptq_marlin_unpackable_reason(
            _stub_linear(HIDDEN, 2 * V_HEADS), GROUP_SIZE
        )
        self.assertIsNotNone(reason)
        self.assertIn("96", reason)
        self.assertIn(str(GPTQ_MARLIN_MIN_THREAD_N), reason)

    def test_every_other_quantized_module_passes(self):
        for name, in_size, out_size in QUANTIZED_MODULES:
            with self.subTest(module=name):
                self.assertIsNone(
                    gptq_marlin_unpackable_reason(
                        _stub_linear(in_size, out_size), GROUP_SIZE
                    ),
                    name,
                )

    def test_a_bad_shard_is_not_hidden_by_the_guard(self):
        # The guard reads UNSHARDED sizes. A layer whose full geometry is fine
        # but whose SHARD misses the tile must stay a loud error, so the guard
        # must not fire on it -- that case belongs to the shard plan (#300).
        self.assertIsNone(
            gptq_marlin_unpackable_reason(
                _stub_linear(HIDDEN, INTERMEDIATE), GROUP_SIZE
            )
        )
        bad_shard = 7904  # the pre-#300 K split of the same checkpoint
        self.assertNotEqual(bad_shard % GPTQ_MARLIN_MIN_THREAD_K, 0)

    def test_non_linear_modules_are_not_judged(self):
        self.assertIsNone(gptq_marlin_unpackable_reason(torch.nn.Module(), GROUP_SIZE))

    def test_input_side_is_covered_too(self):
        # 4304 (the vision MLP width) is neither group- nor min_thread_k
        # aligned; a row-parallel module of that input width has no Marlin
        # form either.
        self.assertIsNotNone(
            gptq_marlin_unpackable_reason(_stub_linear(4304, HIDDEN), GROUP_SIZE)
        )


class TestQuantMethodSelection(CustomTestCase):
    """What `GPTQMarlinConfig` hands each module of the checkpoint."""

    def test_in_proj_ba_gets_the_unquantized_method(self):
        config = _marlin_config()
        method = config.get_quant_method(
            _stub_linear(HIDDEN, 2 * V_HEADS),
            prefix="model.layers.0.linear_attn.in_proj_ba",
        )
        self.assertIsInstance(method, UnquantizedLinearMethod)

    def test_every_other_module_still_gets_marlin(self):
        config = _marlin_config()
        for name, in_size, out_size in QUANTIZED_MODULES:
            with self.subTest(module=name):
                method = config.get_quant_method(
                    _stub_linear(in_size, out_size), prefix=f"model.layers.0.{name}"
                )
                self.assertIsInstance(method, GPTQMarlinLinearMethod)

    def test_the_dynamic_skip_list_still_wins(self):
        # #77's MoE checkpoints skip every dense module via `dynamic`; that
        # path must reach `get_linear_quant_method` unchanged.
        config = _marlin_config(dynamic=MOE_DYNAMIC)
        method = config.get_quant_method(
            _stub_linear(HIDDEN, Q_DIM), prefix="model.layers.0.self_attn.q_proj"
        )
        self.assertIsInstance(method, UnquantizedLinearMethod)
        method = config.get_quant_method(
            _stub_linear(HIDDEN, 2 * 512), prefix="model.layers.0.mlp.experts.0.up_proj"
        )
        self.assertIsInstance(method, GPTQMarlinLinearMethod)

    def test_selection_does_not_depend_on_the_shard_plan(self):
        # Same verdict with and without an installed uneven plan, on every
        # rank: the guard is a property of the checkpoint, not of TP.
        config = _marlin_config()
        cases = [("in_proj_ba", HIDDEN, 2 * V_HEADS, UnquantizedLinearMethod)] + [
            (name, i, o, GPTQMarlinLinearMethod) for name, i, o in QUANTIZED_MODULES
        ]
        for name, in_size, out_size, expected in cases:
            for weights in (None, R7B_WEIGHTS):
                with self.subTest(module=name, weights=weights):
                    with scoped_tp_partition_ratios(weights):
                        method = config.get_quant_method(
                            _stub_linear(in_size, out_size), prefix=name
                        )
                    self.assertIsInstance(method, expected)


class TestRealLayerConstruction(CustomTestCase):
    """The real `MergedColumnParallelLinear` the model builds for in_proj_ba."""

    def _build(self, tp_rank: int, tp_size: int, weights):
        with scoped_tp_partition_ratios(weights):
            return MergedColumnParallelLinear(
                input_size=HIDDEN,
                output_sizes=[V_HEADS, V_HEADS],
                bias=False,
                quant_config=_marlin_config(),
                prefix="model.layers.0.linear_attn.in_proj_ba",
                tp_rank=tp_rank,
                tp_size=tp_size,
                tp_units=GDN_UNITS,
            )

    def test_uneven_ranks_build_dense_weights_of_the_reported_widths(self):
        for rank, width in enumerate(BA_SHARDS):
            with self.subTest(rank=rank):
                layer = self._build(rank, 3, R7B_WEIGHTS)
                self.assertIsInstance(layer.quant_method, UnquantizedLinearMethod)
                # Dense bf16/float weight, not a packed qweight: exactly what
                # the checkpoint's BF16 in_proj_a / in_proj_b carry.
                self.assertFalse(hasattr(layer, "qweight"))
                self.assertEqual(tuple(layer.weight.shape), (width, HIDDEN))
                self.assertEqual(layer.output_partition_sizes, [width // 2] * 2)
                # The k-head unit family survives (an unquantized layer carries
                # no quant block, so nothing coarsens it).
                self.assertEqual(layer.tp_units, GDN_UNITS)

    def test_even_tp_and_tp1_take_the_same_route(self):
        # 96/3 = 32 and 96/1 = 96 miss the tile as well, so this is not an
        # uneven-TP-only repair.
        for tp_size, width in ((1, 96), (3, 32)):
            with self.subTest(tp_size=tp_size):
                layer = self._build(0, tp_size, None)
                self.assertIsInstance(layer.quant_method, UnquantizedLinearMethod)
                self.assertEqual(tuple(layer.weight.shape), (width, HIDDEN))

    def test_a_marlin_shaped_sibling_is_untouched(self):
        # A row-parallel module of the same layer: still Marlin, still packed.
        # Only the scheme's two GPU touch points are stubbed (the kernel
        # support probe reads the device capability, `_init_kernel` imports
        # the CUDA backend); neither takes part in weight creation.
        with (
            scoped_tp_partition_ratios(R7B_WEIGHTS),
            mock.patch(
                "sglang.srt.layers.quantization.gptq.schemes.gptq_marlin.verify_marlin_supported"
            ),
            mock.patch.object(
                GPTQMarlinLinearScheme, "_init_kernel", return_value=mock.MagicMock()
            ),
        ):
            layer = RowParallelLinear(
                input_size=VALUE_DIM,
                output_size=HIDDEN,
                bias=False,
                quant_config=_marlin_config(),
                prefix="model.layers.0.linear_attn.out_proj",
                tp_rank=1,
                tp_size=3,
                tp_units=GDN_UNITS,
            )
        self.assertIsInstance(layer.quant_method, GPTQMarlinLinearMethod)
        self.assertTrue(hasattr(layer, "qweight"))
        # 1920 input channels on rank 1: group- and min_thread_k-aligned.
        self.assertEqual(layer.input_size_per_partition, 1920)
        self.assertEqual(layer.input_size_per_partition % GPTQ_MARLIN_MIN_THREAD_K, 0)


class TestNoRepackWouldBeAttempted(CustomTestCase):
    """The end the traceback came from: `size_n` never reaches the kernel."""

    def test_the_partition_widths_that_reached_repack_are_gone(self):
        # Before the fix these three widths were `partition_weight_shape[1]`
        # of a GPTQMarlinLinearScheme and went straight into
        # gptq_marlin_repack(size_n=...). Now no Marlin scheme exists for the
        # module at all.
        for rank, width in enumerate(BA_SHARDS):
            layer = self._layer(rank)
            self.assertIsInstance(layer.quant_method, UnquantizedLinearMethod)
            self.assertIsNone(getattr(layer, "scheme", None))
            self.assertEqual(layer.weight.shape[0], width)

    def _layer(self, rank: int):
        with scoped_tp_partition_ratios(R7B_WEIGHTS):
            return MergedColumnParallelLinear(
                input_size=HIDDEN,
                output_sizes=[V_HEADS, V_HEADS],
                bias=False,
                quant_config=_marlin_config(),
                prefix="model.layers.0.linear_attn.in_proj_ba",
                tp_rank=rank,
                tp_size=3,
                tp_units=GDN_UNITS,
            )

    def test_the_mlp_geometry_of_300_is_unchanged(self):
        # The #300 pin: 136 group rows split as 62/37/37 -> 7936/4736/4736.
        with scoped_tp_partition_ratios(R7B_WEIGHTS):
            shards = tp_partition_sizes(INTERMEDIATE, 3, 136)
        self.assertEqual(shards, [7936, 4736, 4736])
        for shard in shards:
            self.assertEqual(shard % GPTQ_MARLIN_MIN_THREAD_K, 0)


if __name__ == "__main__":
    unittest.main()
