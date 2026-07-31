"""NVFP4 lane defects: #291-S3 plus the three latent bugs of #323.

All four come out of ``docs/dev/ANALYSE_321_nvfp4_asymmetry.md``:

* **#291-S3** (§9.1) -- ``CompressedTensorsW4A4Fp4`` declared
  ``get_min_capability() == 100`` with no fallback branch, so the only NVFP4
  variant that is simultaneously VRAM-positive (-7736 MiB), context-positive
  (1.57x) and decode-positive (-27 %) -- all-Linear NVFP4, compressed-tensors
  ``nvfp4-pack-quantized`` -- could not boot on a rig with any pre-Blackwell
  rank. The sm_86 lane it was refusing is ``gptq_marlin`` with in-kernel E2M1
  dequant at 0.93-0.96x of those cards' own dense bf16: their best quantised
  lane, not a compatibility minimum.
* **#323a** (§3.5) -- ``ModelOptFp4Config`` exposed no ``weight_block_size``,
  so the uneven-TP coarsening never fired for the NVFP4 family that *does*
  boot on sm_86. Fifth member of the #37 / #86 / #289 / #300 alignment family.
* **#323b** (§3.5) -- the #268 expert-offload guard is an exclusion list and
  NVFP4 MoE walked straight through it, installing an offload cache with no
  offload half.
* **#323c** (§3.3) -- ``fp4_utils``' ``auto`` dispatch could never select the
  fork's own sm_120a JIT CUTLASS kernel.

Pure functions and mocked device capabilities; no GPU, no server, no
checkpoint.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import contextlib
import logging
import math
import unittest
from unittest import mock

import torch

from sglang.srt.distributed.utils import (
    ACTIVATION_VEC_ELEMS,
    set_tp_partition_ratios,
    tp_partition_sizes,
)
from sglang.srt.layers.linear import _quant_block_aligned_units
from sglang.srt.layers.moe.expert_offload import (
    _OFFLOAD_UNSUPPORTED_QUANT_METHOD_NAMES,
    assert_expert_offload_quant_supported,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import moe_uneven_tp_units
from sglang.srt.layers.quantization import fp4_utils
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A4Fp4,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w4a4_nvfp4 as ct_nvfp4,
)
from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend
from sglang.srt.layers.quantization.marlin_utils import (
    GPTQ_MARLIN_MIN_THREAD_K,
    GPTQ_MARLIN_MIN_THREAD_N,
    verify_marlin_supports_shape,
)
from sglang.srt.layers.quantization.modelopt_quant import (
    FP4_GEMM_ALIGNMENT,
    ModelOptFp4Config,
    modelopt_fp4_uneven_tp_block,
)
from sglang.test.test_utils import CustomTestCase

# Qwen3.6-27B geometry, the checkpoint family this whole task is about.
INTERMEDIATE = 17408
HIDDEN = 5120
NVFP4_GROUP = 16
# The exact uneven weight vector the #289/#300 batteries installed:
# --rank-tp-ratio auto on 5090 + 2x3080.
S03_WEIGHTS = [29607, 17780, 17780]
# The #82 three-on-one co-location layout.
TP5_WEIGHTS = [9869, 9869, 9869, 18280, 18280]

# Head-granular families that must NOT be coarsened.
GDN_VALUE_DIM, GDN_K_HEADS = 6144, 16
Q_DIM, KV_HEADS = 6144, 4
VISION_INTERMEDIATE = 4304


def _nvfp4_ct_config(group_size: int = NVFP4_GROUP) -> CompressedTensorsConfig:
    """A real ``nvfp4-pack-quantized`` compressed-tensors config.

    Shaped like ``ocicek/Qwen3.6-27B-NVFP4`` -- all-Linear NVFP4, tensor_group
    strategy, group 16, symmetric, 4-bit weights AND activations.
    """
    args = {
        "num_bits": 4,
        "type": "float",
        "symmetric": True,
        "strategy": "tensor_group",
        "group_size": group_size,
        "dynamic": False,
    }
    return CompressedTensorsConfig.from_config(
        {
            "format": "nvfp4-pack-quantized",
            "quant_method": "compressed-tensors",
            "ignore": ["lm_head"],
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": dict(args),
                    "input_activations": dict(args, dynamic=True),
                }
            },
        }
    )


def _linear_scheme(config: CompressedTensorsConfig):
    """Run the real dispatch for the config's ``Linear`` target.

    ``get_linear_scheme`` additionally needs a live module to match a target
    name against; the branch under test is the capability gate inside
    ``_get_scheme_from_parts``, so it is called with the parsed args directly.
    """
    scheme_args = config.target_scheme_map["Linear"]
    return config._get_scheme_from_parts(
        scheme_args["weights"], scheme_args["input_activations"]
    )


@contextlib.contextmanager
def _fp4_backend(backend: Fp4GemmRunnerBackend):
    """Pin the per-rank FP4 GEMM backend for the duration of a block."""
    previous = fp4_utils.FP4_GEMM_RUNNER_BACKEND
    fp4_utils.FP4_GEMM_RUNNER_BACKEND = backend
    try:
        yield
    finally:
        fp4_utils.FP4_GEMM_RUNNER_BACKEND = previous


@contextlib.contextmanager
def _mock_card(major: int, minor: int):
    """Pretend the local device reports this NVIDIA compute capability."""
    with mock.patch.object(
        torch.cuda, "get_device_capability", return_value=(major, minor)
    ):
        yield


class _FakeNvfp4Layer(torch.nn.Module):
    """The parameter set ``CompressedTensorsW4A4Fp4.create_weights`` leaves."""

    def __init__(self, n: int = 256, k: int = 512, global_scale: float = 8.0):
        super().__init__()
        self.input_size_per_partition = k
        self.output_size_per_partition = n
        self.logical_widths = [n]
        self.params_dtype = torch.bfloat16
        self.weight_packed = torch.nn.Parameter(
            torch.zeros(n, k // 2, dtype=torch.uint8), requires_grad=False
        )
        self.weight_scale = torch.nn.Parameter(
            torch.zeros(n, k // NVFP4_GROUP, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.weight_global_scale = torch.nn.Parameter(
            torch.tensor(global_scale), requires_grad=False
        )
        self.input_global_scale = torch.nn.Parameter(
            torch.tensor(4.0), requires_grad=False
        )


# ===========================================================================
# #291-S3: the compressed-tensors NVFP4 scheme reaches sm_86 through Marlin
# ===========================================================================


class TestCtNvfp4MinCapability(CustomTestCase):
    def test_floor_is_marlin_not_blackwell(self):
        self.assertEqual(CompressedTensorsW4A4Fp4.get_min_capability(), 80)

    def test_sm86_is_no_longer_refused(self):
        """The scissors of §2.1: this is the whole point of #291-S3."""
        config = _nvfp4_ct_config()
        with _mock_card(8, 6):
            self.assertTrue(config._check_scheme_supported(80, error=False))
            scheme = _linear_scheme(config)
        self.assertIsInstance(scheme, CompressedTensorsW4A4Fp4)

    def test_blackwell_still_selects_the_same_scheme(self):
        config = _nvfp4_ct_config()
        for capability in ((10, 0), (12, 0)):
            with _mock_card(*capability):
                scheme = _linear_scheme(config)
            self.assertIsInstance(scheme, CompressedTensorsW4A4Fp4)

    def test_below_the_marlin_floor_still_refuses_by_name(self):
        # Turing: no NVFP4 kernel of any kind in this tree.
        config = _nvfp4_ct_config()
        with _mock_card(7, 5):
            with self.assertRaisesRegex(NotImplementedError, "80"):
                _linear_scheme(config)


class TestCtNvfp4LaneSelection(CustomTestCase):
    """Which lane a rank takes follows that rank's resolved FP4 backend."""

    def test_marlin_backend_selects_the_marlin_lane(self):
        scheme = CompressedTensorsW4A4Fp4()
        with _fp4_backend(Fp4GemmRunnerBackend.MARLIN):
            self.assertTrue(scheme._use_marlin())

    def test_native_backends_do_not(self):
        scheme = CompressedTensorsW4A4Fp4()
        for backend in (
            Fp4GemmRunnerBackend.CUTLASS,
            Fp4GemmRunnerBackend.FLASHINFER_CUTLASS,
            Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL,
            Fp4GemmRunnerBackend.FLASHINFER_TRTLLM,
        ):
            with _fp4_backend(backend):
                self.assertFalse(scheme._use_marlin())

    def test_a_mixed_arch_rig_resolves_both_lanes(self):
        """TP0 = 5090, TP1/TP2 = 3080: one process each, one backend each."""
        args = mock.Mock(fp4_gemm_runner_backend="auto")
        resolved = []
        for is_sm100, is_sm120, capability in (
            (False, True, (12, 0)),
            (False, False, (8, 6)),
            (False, False, (8, 6)),
        ):
            with (
                mock.patch.object(
                    fp4_utils, "is_sm100_supported", return_value=is_sm100
                ),
                mock.patch.object(
                    fp4_utils, "is_sm120_supported", return_value=is_sm120
                ),
                mock.patch.object(fp4_utils, "is_cuda", return_value=True),
                mock.patch.object(
                    fp4_utils, "get_device_capability", return_value=capability
                ),
                mock.patch.object(
                    fp4_utils, "has_fork_nvfp4_cutlass_kernel", return_value=True
                ),
                _fp4_backend(None),
            ):
                fp4_utils.initialize_fp4_gemm_config(args)
                resolved.append(fp4_utils.get_fp4_gemm_runner_backend())
        self.assertEqual(
            resolved,
            [
                Fp4GemmRunnerBackend.CUTLASS,
                Fp4GemmRunnerBackend.MARLIN,
                Fp4GemmRunnerBackend.MARLIN,
            ],
        )


class TestCtNvfp4MarlinWeightHandover(CustomTestCase):
    """The two layout gaps between the CT and ModelOpt tensor conventions."""

    def test_global_scale_is_inverted(self):
        """The load-bearing detail, and a silent one when wrong.

        compressed-tensors stores the MULTIPLY direction; the Marlin helpers
        (written against ModelOpt) want its reciprocal. vLLM's reference
        implementation does the same inversion ("CT stores as divisors").
        Getting it wrong scales every output by global_scale**2 -- no crash.
        """
        layer = _FakeNvfp4Layer(global_scale=8.0)
        scheme = CompressedTensorsW4A4Fp4()
        with mock.patch.object(ct_nvfp4, "prepare_nvfp4_layer_for_marlin") as prepare:
            scheme._process_weights_for_marlin(layer)
        prepare.assert_called_once_with(layer)
        self.assertAlmostEqual(float(layer.weight_global_scale), 1.0 / 8.0, places=6)

    def test_weight_packed_is_renamed_to_weight(self):
        layer = _FakeNvfp4Layer()
        packed = layer.weight_packed.data
        scheme = CompressedTensorsW4A4Fp4()
        with mock.patch.object(ct_nvfp4, "prepare_nvfp4_layer_for_marlin"):
            scheme._process_weights_for_marlin(layer)
        self.assertFalse(hasattr(layer, "weight_packed"))
        self.assertEqual(layer.weight.data_ptr(), packed.data_ptr())

    def test_scales_reach_marlin_unswizzled(self):
        """Marlin does its own permutation and must see the raw scales."""
        layer = _FakeNvfp4Layer()
        raw = layer.weight_scale.data
        scheme = CompressedTensorsW4A4Fp4()
        with mock.patch.object(ct_nvfp4, "prepare_nvfp4_layer_for_marlin"):
            with mock.patch.object(ct_nvfp4, "swizzle_blockscale") as swizzle:
                scheme._process_weights_for_marlin(layer)
        swizzle.assert_not_called()
        self.assertEqual(layer.weight_scale.data_ptr(), raw.data_ptr())

    def test_non_positive_global_scale_is_a_named_error(self):
        layer = _FakeNvfp4Layer(global_scale=0.0)
        scheme = CompressedTensorsW4A4Fp4()
        with self.assertRaisesRegex(ValueError, "weight_global_scale"):
            scheme._process_weights_for_marlin(layer)

    def test_process_weights_routes_to_marlin_on_the_marlin_backend(self):
        layer = _FakeNvfp4Layer()
        scheme = CompressedTensorsW4A4Fp4()
        with _fp4_backend(Fp4GemmRunnerBackend.MARLIN):
            with mock.patch.object(
                ct_nvfp4, "prepare_nvfp4_layer_for_marlin"
            ) as prepare:
                scheme.process_weights_after_loading(layer)
        prepare.assert_called_once()

    def test_native_path_on_a_non_blackwell_card_names_the_flag(self):
        layer = _FakeNvfp4Layer()
        scheme = CompressedTensorsW4A4Fp4()
        with _fp4_backend(Fp4GemmRunnerBackend.FLASHINFER_CUTLASS):
            with mock.patch.object(
                ct_nvfp4, "is_blackwell_supported", return_value=False
            ):
                with self.assertRaisesRegex(ValueError, "--fp4-gemm-backend marlin"):
                    scheme.process_weights_after_loading(layer)

    def test_apply_uses_the_marlin_gemm(self):
        layer = _FakeNvfp4Layer()
        layer.weight = torch.nn.Parameter(
            torch.zeros(4, 4, dtype=torch.int32), requires_grad=False
        )
        layer.workspace = torch.zeros(4, dtype=torch.int32)
        x = torch.zeros(3, layer.input_size_per_partition, dtype=torch.bfloat16)
        scheme = CompressedTensorsW4A4Fp4()
        with _fp4_backend(Fp4GemmRunnerBackend.MARLIN):
            with mock.patch.object(
                ct_nvfp4, "apply_fp4_marlin_linear", return_value=torch.zeros(1)
            ) as apply:
                scheme.apply_weights(layer, x, bias=None)
        kwargs = apply.call_args.kwargs
        self.assertEqual(kwargs["size_n"], layer.output_size_per_partition)
        self.assertEqual(kwargs["size_k"], layer.input_size_per_partition)
        self.assertIs(kwargs["weight_global_scale"], layer.weight_global_scale)

    def test_create_weights_rejects_a_tile_invalid_shard_early(self):
        """A late Marlin abort mid-first-forward becomes a load-time error."""
        scheme = CompressedTensorsW4A4Fp4()
        layer = torch.nn.Module()
        with _fp4_backend(Fp4GemmRunnerBackend.MARLIN):
            with self.assertRaisesRegex(ValueError, "9504"):
                scheme.create_weights(
                    layer=layer,
                    output_partition_sizes=[9504],
                    input_size_per_partition=HIDDEN,
                    params_dtype=torch.bfloat16,
                    weight_loader=lambda *a, **k: None,
                )


# ===========================================================================
# #323a: ModelOptFp4Config joins the uneven-TP alignment family
# ===========================================================================


def _mlp_units(intermediate: int, quant_config) -> int:
    """Mirrors the derivation in sglang.srt.models.qwen2_moe.Qwen2MoeMLP."""
    units = intermediate // math.gcd(intermediate, ACTIVATION_VEC_ELEMS)
    return _quant_block_aligned_units(intermediate, units, quant_config, 1)


def _modelopt_fp4_config(group_size: int = NVFP4_GROUP) -> ModelOptFp4Config:
    return ModelOptFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo="auto",
        group_size=group_size,
        exclude_modules=[],
    )


class TestModelOptFp4Block(CustomTestCase):
    def test_block_folds_group_and_both_kernel_tiles(self):
        block = modelopt_fp4_uneven_tp_block(NVFP4_GROUP)
        self.assertEqual(block, [128, 128])
        self.assertEqual(block[0] % NVFP4_GROUP, 0)
        self.assertEqual(block[0] % FP4_GEMM_ALIGNMENT, 0)  # native FP4 CUTLASS
        self.assertEqual(block[0] % GPTQ_MARLIN_MIN_THREAD_K, 0)  # Marlin E2M1
        self.assertEqual(block[0] % GPTQ_MARLIN_MIN_THREAD_N, 0)

    def test_config_exposes_it(self):
        self.assertEqual(_modelopt_fp4_config().weight_block_size, [128, 128])

    def test_missing_group_size_falls_back_to_the_kernel_tiles(self):
        self.assertEqual(
            modelopt_fp4_uneven_tp_block(None),
            [math.lcm(FP4_GEMM_ALIGNMENT, GPTQ_MARLIN_MIN_THREAD_K)] * 2,
        )

    def test_the_block_is_architecture_invariant(self):
        """The property that makes a mixed-arch rig safe.

        The shard plan is derived independently on every rank. On this rig the
        ranks resolve DIFFERENT FP4 backends (native FP4 on the 5090, Marlin on
        the 3080s), so a block derived from the local backend would hand rank 0
        a 32-aligned split and ranks 1/2 a 128-aligned one, and the model would
        be silently mis-sharded. The block must depend on the checkpoint only.
        """
        config = _modelopt_fp4_config()
        blocks = []
        for backend in (
            Fp4GemmRunnerBackend.MARLIN,
            Fp4GemmRunnerBackend.CUTLASS,
            Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL,
            Fp4GemmRunnerBackend.AUTO,
        ):
            with _fp4_backend(backend):
                blocks.append(config.weight_block_size)
        self.assertEqual(blocks, [[128, 128]] * 4)

    def test_it_agrees_with_the_compressed_tensors_sibling(self):
        """§3.5 counter-test: the CT path already returns [128, 128]."""
        self.assertEqual(
            _nvfp4_ct_config().weight_block_size,
            _modelopt_fp4_config().weight_block_size,
        )


class TestModelOptFp4UnevenSplit(CustomTestCase):
    def setUp(self):
        set_tp_partition_ratios(S03_WEIGHTS)

    def tearDown(self):
        set_tp_partition_ratios(None)

    def test_pre_change_reproduces_the_tile_abort(self):
        units = _mlp_units(INTERMEDIATE, None)
        self.assertEqual(units, INTERMEDIATE // ACTIVATION_VEC_ELEMS)
        sizes = tp_partition_sizes(INTERMEDIATE, 3, units=units, family="mlp")
        self.assertEqual(sizes, [7904, 4752, 4752])
        merged = [2 * s for s in sizes]  # gate_up is fused
        with self.assertRaisesRegex(ValueError, "9504.*min_thread_n = 64"):
            verify_marlin_supports_shape(
                output_size_per_partition=merged[1],
                input_size_per_partition=HIDDEN,
                input_size=HIDDEN,
                group_size=NVFP4_GROUP,
            )
        # And none of them ends a whole NVFP4 group of 16 on the native lane
        # either -- 4752 % 16 == 0 holds, but the CUTLASS K alignment does not.
        self.assertTrue(any(s % FP4_GEMM_ALIGNMENT for s in sizes))

    def test_post_change_is_group_and_tile_clean(self):
        units = _mlp_units(INTERMEDIATE, _modelopt_fp4_config())
        self.assertEqual(units, INTERMEDIATE // 128)  # 136
        sizes = tp_partition_sizes(INTERMEDIATE, 3, units=units, family="mlp")
        self.assertEqual(sizes, [7936, 4736, 4736])
        self.assertEqual(sum(sizes), INTERMEDIATE)
        for s in sizes:
            verify_marlin_supports_shape(
                output_size_per_partition=2 * s,
                input_size_per_partition=HIDDEN,
                input_size=HIDDEN,
                group_size=NVFP4_GROUP,
            )
            verify_marlin_supports_shape(
                output_size_per_partition=HIDDEN,
                input_size_per_partition=s,
                input_size=INTERMEDIATE,
                group_size=NVFP4_GROUP,
            )
            self.assertEqual(s % NVFP4_GROUP, 0)
            self.assertEqual(s % FP4_GEMM_ALIGNMENT, 0)
            self.assertEqual(s % ACTIVATION_VEC_ELEMS, 0)

    def test_gate_up_and_down_coarsen_identically(self):
        config = _modelopt_fp4_config()
        units = _mlp_units(INTERMEDIATE, config)
        self.assertEqual(
            _quant_block_aligned_units(INTERMEDIATE, units, config, 0), units
        )
        self.assertEqual(
            _quant_block_aligned_units(INTERMEDIATE, units, config, 1), units
        )

    def test_the_ratio_survives_the_coarsening(self):
        units = _mlp_units(INTERMEDIATE, _modelopt_fp4_config())
        sizes = tp_partition_sizes(INTERMEDIATE, 3, units=units, family="mlp")
        total = sum(S03_WEIGHTS)
        for size, weight in zip(sizes, S03_WEIGHTS):
            self.assertLess(abs(size / INTERMEDIATE - weight / total), 0.01)
        self.assertGreater(sizes[0], sizes[1])


class TestModelOptFp4HeadGranularFamilies(CustomTestCase):
    def setUp(self):
        set_tp_partition_ratios(S03_WEIGHTS)

    def tearDown(self):
        set_tp_partition_ratios(None)

    def test_gdn_k_head_units_unchanged(self):
        self.assertEqual(
            _quant_block_aligned_units(
                GDN_VALUE_DIM, GDN_K_HEADS, _modelopt_fp4_config(), 1
            ),
            GDN_K_HEADS,
        )

    def test_qkv_q_block_units_unchanged(self):
        self.assertEqual(
            _quant_block_aligned_units(Q_DIM, KV_HEADS, _modelopt_fp4_config(), 0),
            KV_HEADS,
        )

    def test_non_block_multiple_dimension_unchanged(self):
        self.assertEqual(
            _quant_block_aligned_units(
                VISION_INTERMEDIATE, VISION_INTERMEDIATE, _modelopt_fp4_config(), 0
            ),
            VISION_INTERMEDIATE,
        )


class TestModelOptFp4InertOnTheDefaultPath(CustomTestCase):
    def setUp(self):
        set_tp_partition_ratios(None)

    def tearDown(self):
        set_tp_partition_ratios(None)

    def test_even_tp_is_identical_with_and_without_the_block(self):
        pre = _mlp_units(INTERMEDIATE, None)
        post = _mlp_units(INTERMEDIATE, _modelopt_fp4_config())
        for tp in (1, 2, 4, 8):
            classic = [INTERMEDIATE // tp] * tp
            self.assertEqual(
                tp_partition_sizes(INTERMEDIATE, tp, units=pre, family="mlp"), classic
            )
            self.assertEqual(
                tp_partition_sizes(INTERMEDIATE, tp, units=post, family="mlp"), classic
            )

    def test_moe_expert_grain_unchanged(self):
        """The dense block must not hijack the expert intermediate grain.

        NVFP4's group is 16, below moe_uneven_tp_units' 32 floor, so the
        expert dimension stays element-granular exactly as it was before this
        config exposed anything.
        """
        config = _modelopt_fp4_config()
        for intermediate in (512, 704, 1024):
            self.assertEqual(
                moe_uneven_tp_units(intermediate, config),
                moe_uneven_tp_units(intermediate, None),
            )


class TestModelOptFp4TP5CoLocation(CustomTestCase):
    def setUp(self):
        set_tp_partition_ratios(TP5_WEIGHTS)

    def tearDown(self):
        set_tp_partition_ratios(None)

    def test_five_ranks_stay_tile_clean(self):
        units = _mlp_units(INTERMEDIATE, _modelopt_fp4_config())
        sizes = tp_partition_sizes(INTERMEDIATE, 5, units=units, family="mlp")
        self.assertEqual(sum(sizes), INTERMEDIATE)
        for s in sizes:
            self.assertEqual(s % 128, 0)
            self.assertEqual((2 * s) % GPTQ_MARLIN_MIN_THREAD_N, 0)


# ===========================================================================
# #323b: the expert-offload guard knows about NVFP4 MoE
# ===========================================================================


def _named(cls_name: str):
    return type(cls_name, (), {})()


class TestExpertOffloadNvfp4Guard(CustomTestCase):
    NVFP4_MOE_METHODS = (
        "ModelOptNvFp4FusedMoEMethod",
        "ModelOptNvFp4OnlineFusedMoEMethod",
    )

    def test_every_nvfp4_moe_method_is_denied_by_name(self):
        for name in self.NVFP4_MOE_METHODS:
            with self.assertRaises(RuntimeError) as ctx:
                assert_expert_offload_quant_supported(_named(name), layer_id=7)
            message = str(ctx.exception)
            self.assertIn(name, message)
            self.assertIn("layer_id=7", message)
            self.assertIn("NVFP4", message)

    def test_the_compressed_tensors_scheme_is_reached_through_the_wrapper(self):
        """A CT layer's quant_method is always the same delegating wrapper.

        Checking only the wrapper would either miss NVFP4 entirely or deny
        every compressed-tensors checkpoint; the scheme is the class that owns
        the tensor layout.
        """
        wrapper = _named("CompressedTensorsFusedMoEMethod")
        # Wrapper alone: allowed (fp8/int8 CT MoE keep working).
        assert_expert_offload_quant_supported(wrapper)
        with self.assertRaisesRegex(RuntimeError, "CompressedTensorsW4A4Nvfp4MoE"):
            assert_expert_offload_quant_supported(
                wrapper, layer_id=3, scheme=_named("CompressedTensorsW4A4Nvfp4MoE")
            )

    def test_the_error_is_hard_not_a_silent_downgrade(self):
        # The guard raises; it must never return a "degraded" sentinel.
        with self.assertRaises(RuntimeError):
            assert_expert_offload_quant_supported(_named("ModelOptNvFp4FusedMoEMethod"))

    def test_supported_offload_paths_are_untouched(self):
        """Regression pins for the validated #77 / #256 offload arms."""
        for name in (
            "Fp8MoEMethod",
            "GPTQMarlinMoEMethod",
            "AWQMoEMethod",
            "AWQMarlinMoEMethod",
            "UnquantizedFusedMoEMethod",
            "CompressedTensorsFusedMoEMethod",
        ):
            assert_expert_offload_quant_supported(_named(name), layer_id=0)
            assert_expert_offload_quant_supported(
                _named("CompressedTensorsFusedMoEMethod"), scheme=_named(name)
            )

    def test_the_original_268_denials_still_hold(self):
        for name in ("GGUFMoEMethod", "GGUFMoEAscendMethod", "MoeWNA16Method"):
            with self.assertRaises(RuntimeError):
                assert_expert_offload_quant_supported(_named(name))

    def test_none_scheme_is_tolerated(self):
        # Non-CT layers pass scheme=None.
        assert_expert_offload_quant_supported(_named("Fp8MoEMethod"), scheme=None)

    def test_the_deny_list_names_all_three_nvfp4_classes(self):
        for name in (
            *self.NVFP4_MOE_METHODS,
            "CompressedTensorsW4A4Nvfp4MoE",
        ):
            self.assertIn(name, _OFFLOAD_UNSUPPORTED_QUANT_METHOD_NAMES)


# ===========================================================================
# #323c: `auto` can finally select the fork's own sm_120a kernel
# ===========================================================================


class TestFp4AutoBackendRouting(CustomTestCase):
    def _resolve(self, *, is_sm100, is_sm120, capability, fork_kernel, backend="auto"):
        args = mock.Mock(fp4_gemm_runner_backend=backend)
        with (
            mock.patch.object(fp4_utils, "is_sm100_supported", return_value=is_sm100),
            mock.patch.object(fp4_utils, "is_sm120_supported", return_value=is_sm120),
            mock.patch.object(fp4_utils, "is_cuda", return_value=True),
            mock.patch.object(
                fp4_utils, "get_device_capability", return_value=capability
            ),
            mock.patch.object(
                fp4_utils, "has_fork_nvfp4_cutlass_kernel", return_value=fork_kernel
            ),
            _fp4_backend(None),
        ):
            fp4_utils.initialize_fp4_gemm_config(args)
            return fp4_utils.get_fp4_gemm_runner_backend()

    def test_sm120_selects_the_fork_kernel_when_it_exists(self):
        self.assertEqual(
            self._resolve(
                is_sm100=False, is_sm120=True, capability=(12, 0), fork_kernel=True
            ),
            Fp4GemmRunnerBackend.CUTLASS,
        )

    def test_sm120_degrades_by_name_when_it_does_not(self):
        with self.assertLogs(fp4_utils.logger, level=logging.WARNING) as logs:
            resolved = self._resolve(
                is_sm100=False, is_sm120=True, capability=(12, 0), fork_kernel=False
            )
        self.assertEqual(resolved, Fp4GemmRunnerBackend.FLASHINFER_CUTLASS)
        joined = "\n".join(logs.output)
        self.assertIn("sm_120a", joined)
        self.assertIn("cutlass_scaled_fp4_mm", joined)

    def test_sm100_is_unchanged(self):
        self.assertEqual(
            self._resolve(
                is_sm100=True, is_sm120=False, capability=(10, 0), fork_kernel=True
            ),
            Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL,
        )

    def test_sm80_to_sm89_is_unchanged(self):
        for capability in ((8, 0), (8, 6), (8, 9)):
            self.assertEqual(
                self._resolve(
                    is_sm100=False,
                    is_sm120=False,
                    capability=capability,
                    fork_kernel=False,
                ),
                Fp4GemmRunnerBackend.MARLIN,
            )

    def test_an_explicit_backend_is_never_overridden(self):
        for name in ("marlin", "cutlass", "flashinfer_trtllm"):
            self.assertEqual(
                self._resolve(
                    is_sm100=False,
                    is_sm120=True,
                    capability=(12, 0),
                    fork_kernel=True,
                    backend=name,
                ),
                Fp4GemmRunnerBackend(name),
            )

    def test_kernel_availability_needs_both_arch_and_import(self):
        """#304: never claim a kernel from a foreign cache without an image."""
        with mock.patch.object(fp4_utils, "is_sm100_supported", return_value=False):
            with mock.patch.object(fp4_utils, "is_sm120_supported", return_value=False):
                self.assertFalse(fp4_utils.has_fork_nvfp4_cutlass_kernel())


if __name__ == "__main__":
    unittest.main()
