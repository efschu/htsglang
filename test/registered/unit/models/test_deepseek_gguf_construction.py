# SPDX-License-Identifier: Apache-2.0
"""#401: the DeepSeek V2/V3/V4 class can be BUILT under GGUF.

THE DEFECT, as it stood
``DeepseekV2MoE.__init__`` classified its shared expert by reading
``self.shared_experts.gate_up_proj.weight.dtype``. A GGUF linear has no
``weight`` -- it registers ``qweight`` -- so construction died with

    AttributeError: 'MergedColumnParallelLinear' object has no attribute
    'weight'. Did you mean: 'qweight'?

before a single byte of the checkpoint was read (DeepSeek-V4-Flash GGUF boot
attempt 5, 2026-08-01). ``n_shared_experts=1`` is in the V4-Flash config and
cannot be switched off, so the site was unavoidable.

The same shape of bug sat at four more places, each of them a *fast-path
selector* that read a dtype off a tensor that only dense/int8/fp8 checkpoints
have. They are pinned individually below, because fixing the first one only
moves the failure:

  * ``DeepseekV2MoE.__init__``          -- int8 / fp8 shared-expert flags
  * ``DeepseekV2MLP.forward``           -- uint8 zero-allocator and fp8
                                           swiglu fast paths
  * ``DeepseekV2AttentionMLA.__init__`` -- ``use_min_latency_fused_a_gemm``
  * ``init_mla_fused_rope_cpu_forward`` -- ``qkv_proj_with_rope_is_int8/fp8``
                                           (this one is NOT in the original
                                           report; it runs from the MLA
                                           constructor on every device and
                                           was the next stopper after :760)
  * ``determine_num_fused_shared_experts`` -- the fused layout has no slot a
                                           GGUF shared expert could occupy

The fix routes every dtype gate through ``dense_weight_dtype``, which answers
``None`` when there is no dense weight, so a packed checkpoint takes the
general path instead of raising. NEUTRALITY, pinned below: whenever a dense
``weight`` exists, ``dense_weight_dtype`` returns exactly what the old direct
attribute access returned, so fp8 / int8 / awq / gptq verdicts are unchanged.

Where the general path does not exist, the code refuses BY NAME rather than
running something wrong -- also pinned here.
"""

import os

# The DSV4 attention module captures SGLANG_OPT_FP8_WO_A_GEMM at import time.
# It is auto-disabled on every CUDA device below sm100, but this test must be
# deterministic on the CI host too. Set it before sglang is imported; the
# fp8-on-GGUF refusal is exercised separately by patching the module global.
os.environ.setdefault("SGLANG_OPT_FP8_WO_A_GEMM", "0")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from sglang.srt.layers.quantization.gguf import GGUFConfig
from sglang.srt.runtime_context import get_context, get_parallel, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _ensure_dist_initialized() -> None:
    """Minimal single-rank gloo world plus TP=1 model-parallel groups.

    Every DeepSeek submodule reads the parallel context inside ``__init__``
    to size its shards, so the groups must exist before any construction.
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29641")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    if not torch.distributed.is_initialized():
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend="gloo",
        )


# Scaled-down V4-Flash: the shapes that matter for this bug are
# n_shared_experts=1 (the site), swiglu_limit (the MLP fp8 gate) and a MoE
# layer that is not the first_k_dense_replace one.
_V4_CONFIG = dict(
    architectures=["DeepseekV4ForCausalLM"],
    quantization_config=None,
    num_hidden_layers=2,
    first_k_dense_replace=0,
    num_hash_layers=1,
    n_routed_experts=8,
    num_experts_per_tok=2,
    n_group=1,
    topk_group=1,
    hidden_size=256,
    intermediate_size=128,
    moe_intermediate_size=64,
    num_attention_heads=4,
    index_n_heads=4,
    index_head_dim=32,
    index_topk=32,
    q_lora_rank=64,
    o_lora_rank=64,
    o_groups=2,
    head_dim=64,
    kv_lora_rank=32,
    qk_nope_head_dim=32,
    qk_rope_head_dim=16,
    v_head_dim=32,
    vocab_size=512,
    max_position_embeddings=1024,
    compress_ratios=[0, 0],
    swiglu_limit=10.0,
    n_shared_experts=1,
    hc_mult=2,
    rope_scaling={
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 2,
        "original_max_position_embeddings": 512,
        "type": "yarn",
    },
)

# Scaled-down V3, i.e. the DeepseekV2 base class that mistral_large_3 and the
# NextN drafts also come through.
_V3_CONFIG = dict(
    architectures=["DeepseekV3ForCausalLM"],
    model_type="deepseek_v3",
    vocab_size=512,
    hidden_size=256,
    intermediate_size=128,
    moe_intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_shared_experts=1,
    n_routed_experts=8,
    num_experts_per_tok=2,
    n_group=1,
    topk_group=1,
    routed_scaling_factor=1.0,
    norm_topk_prob=True,
    scoring_func="sigmoid",
    topk_method="noaux_tc",
    first_k_dense_replace=1,
    moe_layer_freq=1,
    hidden_act="silu",
    rms_norm_eps=1e-6,
    q_lora_rank=64,
    kv_lora_rank=32,
    qk_nope_head_dim=32,
    qk_rope_head_dim=16,
    v_head_dim=32,
    rope_theta=10000,
    max_position_embeddings=1024,
    tie_word_embeddings=False,
    pad_token_id=0,
    rope_scaling={
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 2,
        "original_max_position_embeddings": 512,
        "type": "yarn",
    },
)


def _moe_config(quant_method: str) -> SimpleNamespace:
    """Minimal DeepseekV2MoE config with one shared expert.

    The hidden/intermediate sizes are multiples of 128 so the fp8 block-quant
    layout is legal, which is what lets the fp8 neutrality case below build.
    """
    return SimpleNamespace(
        hidden_size=256,
        moe_intermediate_size=128,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        hidden_act="silu",
        routed_scaling_factor=1.0,
        vocab_size=128,
        num_hash_layers=0,
        quantization_config={"quant_method": quant_method},
    )


class _ServerArgsFixture(CustomTestCase):
    """Publishes a CPU ServerArgs and restores whatever was there before."""

    def setUp(self):
        _ensure_dist_initialized()
        self._saved_server_args = get_context()._server_args
        server_args = ServerArgs(model_path="dummy", device="cpu")
        get_context().set_server_args(server_args)
        self.server_args = server_args

    def tearDown(self):
        if self._saved_server_args is None:
            reset_context()
        else:
            get_context().set_server_args(self._saved_server_args)


class TestConstructionUnderGGUF(_ServerArgsFixture):
    """The falsifier: pre-fix each of these raises AttributeError."""

    def test_deepseek_v2_moe_builds_its_shared_expert(self):
        # The exact site of the boot failure: deepseek_v2.py DeepseekV2MoE
        # __init__, reached because n_shared_experts=1 and the fused layout is
        # off. Pre-fix: AttributeError on `.weight`.
        from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

        self.server_args.disable_shared_experts_fusion = True
        with torch.device("meta"):
            moe = DeepseekV2MoE(
                config=_moe_config("gguf"),
                layer_id=0,
                quant_config=GGUFConfig(),
                prefix="model.layers.0.mlp",
            )

        # The shared expert exists (it is not disableable for V4-Flash) and
        # is packed, so neither dense fast path is selected.
        self.assertTrue(hasattr(moe, "shared_experts"))
        self.assertFalse(hasattr(moe.shared_experts.gate_up_proj, "weight"))
        self.assertTrue(hasattr(moe.shared_experts.gate_up_proj, "qweight"))
        self.assertFalse(moe.shared_experts_is_int8)
        self.assertFalse(moe.shared_experts_is_fp8)
        self.assertIsNone(moe.shared_experts_weight_block_size)

    def test_deepseek_v3_causal_lm_builds(self):
        # The V2 base class end to end: MLA attention (fused qkv_a gate and
        # the CPU fused-rope gate) plus the MoE layer.
        from transformers import DeepseekV3Config

        from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM

        config = DeepseekV3Config(**_V3_CONFIG)
        with torch.device("meta"), get_parallel().override(
            attn_dp_size=1, attn_dp_rank=0
        ):
            model = DeepseekV2ForCausalLM(config=config, quant_config=GGUFConfig())

        attn = model.model.layers[1].self_attn
        self.assertTrue(attn.has_fused_proj)
        # No dense weight -> the fused-a GEMM fast path is off, and the CPU
        # fused-rope int8/fp8 selectors resolve to False instead of raising.
        self.assertFalse(attn.use_min_latency_fused_a_gemm)
        self.assertFalse(attn.qkv_proj_with_rope_is_int8)
        self.assertFalse(attn.qkv_proj_with_rope_is_fp8)
        self.assertFalse(model.model.layers[1].mlp.shared_experts_is_fp8)

    def test_deepseek_v3_nextn_draft_builds(self):
        # The other entry point into the same decoder layer. V4-Flash GGUF
        # ships no NextN, but other DSV4 exports may, and the draft class
        # instantiates the very modules that were broken.
        from transformers import DeepseekV3Config

        from sglang.srt.models.deepseek_nextn import DeepseekV3ForCausalLMNextN

        config = DeepseekV3Config(num_nextn_predict_layers=1, **_V3_CONFIG)
        with torch.device("meta"), get_parallel().override(
            attn_dp_size=1, attn_dp_rank=0
        ):
            model = DeepseekV3ForCausalLMNextN(config=config, quant_config=GGUFConfig())

        # ...and the vocab tables it would want to share with its target are
        # packed by default, which the deepseek NextN class cannot do (it has
        # no module-level sharing). eagle_worker_v2 refuses that combination
        # by name before calling get_embed_and_head; this pins the premise of
        # that refusal, so adding module-level sharing here is a deliberate
        # act rather than an accident.
        self.assertTrue(hasattr(model.lm_head, "qweight"))
        self.assertFalse(hasattr(model.lm_head, "weight"))
        self.assertFalse(hasattr(model, "set_embed_and_head_modules"))

    def test_deepseek_v4_causal_lm_builds(self):
        # The boot target. DeepseekV4DecoderLayer delegates its MoE to
        # deepseek_v2.DeepseekV2MoE, which is where the boot died.
        from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        config = DeepSeekV4Config(**_V4_CONFIG)
        with torch.device("meta"):
            model = DeepseekV4ForCausalLM(config=config, quant_config=GGUFConfig())

        self.assertEqual(model.num_fused_shared_experts, 0)
        mlp = model.model.layers[1].mlp
        self.assertTrue(hasattr(mlp, "shared_experts"))
        self.assertFalse(mlp.shared_experts_is_int8)
        self.assertFalse(mlp.shared_experts_is_fp8)


class TestNonGGUFVerdictsUnchanged(_ServerArgsFixture):
    """The dense quants still take their fast paths.

    ``dense_weight_dtype`` is only a safe substitution if the int8 and fp8
    verdicts still come out TRUE for checkpoints that carry a dense weight.
    These build the real ``DeepseekV2MoE`` against the real fp8 and int8
    linear methods and read the flags back."""

    def _build(self, quant_method, quant_config):
        from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

        self.server_args.disable_shared_experts_fusion = True
        with torch.device("meta"):
            return DeepseekV2MoE(
                config=_moe_config(quant_method),
                layer_id=0,
                quant_config=quant_config,
                prefix="model.layers.0.mlp",
            )

    def test_unquantized_selects_neither_fast_path(self):
        moe = self._build("none", None)
        self.assertTrue(hasattr(moe.shared_experts.gate_up_proj, "weight"))
        self.assertFalse(moe.shared_experts_is_int8)
        self.assertFalse(moe.shared_experts_is_fp8)

    def test_block_fp8_still_selects_the_fp8_path(self):
        from sglang.srt.layers.quantization.fp8 import Fp8Config

        moe = self._build(
            "fp8",
            Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]),
        )
        self.assertEqual(
            moe.shared_experts.gate_up_proj.weight.dtype, torch.float8_e4m3fn
        )
        self.assertTrue(moe.shared_experts_is_fp8)
        self.assertFalse(moe.shared_experts_is_int8)
        # The block size is still picked up, i.e. the branch behind the gate
        # ran exactly as before.
        self.assertEqual(moe.shared_experts_weight_block_size, [128, 128])

    def test_w8a8_int8_still_selects_the_int8_path(self):
        from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config

        moe = self._build("w8a8_int8", W8A8Int8Config())
        self.assertEqual(moe.shared_experts.gate_up_proj.weight.dtype, torch.int8)
        self.assertTrue(moe.shared_experts_is_int8)
        self.assertFalse(moe.shared_experts_is_fp8)


class TestDtypeGateNeutrality(CustomTestCase):
    """The helper import lives inside the methods, not at module scope, so
    that the construction tests above still RUN (and fail at the real site)
    against an unfixed tree instead of erroring at collection.

    Every rewritten site replaced ``x.weight.dtype == D`` with
    ``dense_weight_dtype(x) == D``. That is behavior-preserving exactly when
    ``dense_weight_dtype`` agrees with ``.weight.dtype`` wherever a dense
    weight exists -- which is what these pin, dtype by dtype."""

    def _dense_layer(self, dtype):
        layer = nn.Module()
        layer.weight = nn.Parameter(
            torch.zeros(2, 2, dtype=torch.float32).to(dtype), requires_grad=False
        )
        return layer

    def test_dense_dtypes_are_reported_unchanged(self):
        from sglang.srt.models.deepseek_common.utils import dense_weight_dtype

        for dtype in (
            torch.bfloat16,
            torch.float16,
            torch.int8,
            torch.uint8,
            torch.float8_e4m3fn,
        ):
            with self.subTest(dtype=dtype):
                layer = self._dense_layer(dtype)
                self.assertEqual(dense_weight_dtype(layer), layer.weight.dtype)
                self.assertEqual(dense_weight_dtype(layer), dtype)

    def test_packed_layers_report_none_where_the_old_code_raised(self):
        from sglang.srt.models.deepseek_common.utils import dense_weight_dtype

        # AWQ / GPTQ / GGUF shape: qweight, no weight.
        layer = nn.Module()
        layer.qweight = nn.Parameter(
            torch.zeros(2, 2, dtype=torch.int32), requires_grad=False
        )
        self.assertIsNone(dense_weight_dtype(layer))
        with self.assertRaises(AttributeError):
            _ = layer.weight.dtype

    def test_none_never_equals_a_real_dtype(self):
        from sglang.srt.models.deepseek_common.utils import dense_weight_dtype

        # The gates are all `== <dtype>`, so None must not collide with any
        # of them; this is what makes "packed" mean "general path".
        layer = nn.Module()
        for dtype in (torch.int8, torch.uint8, torch.float8_e4m3fn, torch.bfloat16):
            self.assertNotEqual(dense_weight_dtype(layer), dtype)

    def test_quant_method_name_reads_the_layer_not_the_model(self):
        from sglang.srt.models.deepseek_common.utils import layer_quant_method_name

        layer = nn.Module()
        self.assertIsNone(layer_quant_method_name(layer))
        layer.quant_method = SimpleNamespace(
            quant_config=SimpleNamespace(get_name=lambda: "fp8")
        )
        self.assertEqual(layer_quant_method_name(layer), "fp8")

    def test_is_gguf_quant_config_only_matches_gguf(self):
        from sglang.srt.models.deepseek_common.utils import is_gguf_quant_config

        self.assertFalse(is_gguf_quant_config(None))
        for name in ("fp8", "w8a8_int8", "awq", "awq_marlin", "gptq", "moe_wna16"):
            with self.subTest(name=name):
                self.assertFalse(
                    is_gguf_quant_config(SimpleNamespace(get_name=lambda n=name: n))
                )
        self.assertTrue(is_gguf_quant_config(SimpleNamespace(get_name=lambda: "gguf")))


class TestSharedExpertFusionPolicy(_ServerArgsFixture):
    """The fused layout remaps ``mlp.shared_experts.*`` onto an extra routed
    expert slot. GGUF ships the shared expert as its own packed linears, so
    the slot would be filled with nothing -- a silent mis-load. It is refused
    for GGUF and left exactly as it was for every other quant."""

    def _v2_model(self, quant_name):
        return SimpleNamespace(
            config=SimpleNamespace(
                architectures=["DeepseekV3ForCausalLM"],
                n_routed_experts=256,
                n_shared_experts=1,
            ),
            quant_config=(
                None
                if quant_name is None
                else SimpleNamespace(get_name=lambda: quant_name)
            ),
        )

    def _v4_model(self, quant_name):
        return SimpleNamespace(
            config=SimpleNamespace(n_shared_experts=1),
            quant_config=(
                None
                if quant_name is None
                else SimpleNamespace(get_name=lambda: quant_name)
            ),
        )

    def test_v4_gguf_disables_fusion_and_says_why(self):
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        self.server_args.enforce_shared_experts_fusion = False
        model = self._v4_model("gguf")
        DeepseekV4ForCausalLM.determine_num_fused_shared_experts(model)
        self.assertEqual(model.num_fused_shared_experts, 0)
        self.assertTrue(self.server_args.disable_shared_experts_fusion)

    def test_v4_gguf_refuses_enforced_fusion_by_name(self):
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        self.server_args.enforce_shared_experts_fusion = True
        model = self._v4_model("gguf")
        with self.assertRaises(NotImplementedError) as ctx:
            DeepseekV4ForCausalLM.determine_num_fused_shared_experts(model)
        self.assertIn("gguf", str(ctx.exception))
        self.assertIn("enforce-shared-experts-fusion", str(ctx.exception))

    def test_v4_non_gguf_enforced_fusion_is_unchanged(self):
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        for quant_name in (None, "fp8", "w8a8_int8", "awq", "gptq"):
            with self.subTest(quant=quant_name):
                self.server_args.enforce_shared_experts_fusion = True
                self.server_args.disable_shared_experts_fusion = False
                model = self._v4_model(quant_name)
                DeepseekV4ForCausalLM.determine_num_fused_shared_experts(model)
                self.assertEqual(model.num_fused_shared_experts, 1)
                self.assertFalse(self.server_args.disable_shared_experts_fusion)

    def test_v2_gguf_disables_fusion_even_when_enforced_is_off(self):
        from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM

        self.server_args.enforce_shared_experts_fusion = False
        model = self._v2_model("gguf")
        DeepseekV2ForCausalLM.determine_num_fused_shared_experts(model)
        self.assertEqual(model.num_fused_shared_experts, 0)
        self.assertTrue(self.server_args.disable_shared_experts_fusion)

    def test_v2_gguf_refuses_enforced_fusion_by_name(self):
        from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM

        self.server_args.enforce_shared_experts_fusion = True
        model = self._v2_model("gguf")
        with self.assertRaises(NotImplementedError) as ctx:
            DeepseekV2ForCausalLM.determine_num_fused_shared_experts(model)
        self.assertIn("gguf", str(ctx.exception))

    def test_v2_non_gguf_enforced_fusion_is_unchanged(self):
        from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM

        for quant_name in (None, "fp8", "w8a8_int8", "awq", "gptq"):
            with self.subTest(quant=quant_name):
                self.server_args.enforce_shared_experts_fusion = True
                self.server_args.disable_shared_experts_fusion = False
                model = self._v2_model(quant_name)
                DeepseekV2ForCausalLM.determine_num_fused_shared_experts(model)
                self.assertEqual(model.num_fused_shared_experts, 1)
                self.assertFalse(self.server_args.disable_shared_experts_fusion)


class TestNamedRefusals(CustomTestCase):
    """Where no general path exists, the code names the feature and the quant
    method instead of running something wrong."""

    def test_mla_absorption_refuses_gguf_kv_b_proj(self):
        # w_kc / w_vc are split out of a DENSE kv_b_proj. A GGUF kv_b_proj
        # carries qweight like an AWQ one, so without the refusal it would
        # fall into the AWQ dequantizer and die there on a missing `scales`.
        from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
            DeepseekV2WeightLoaderMixin,
        )

        kv_b_proj = nn.Module()
        kv_b_proj.qweight = nn.Parameter(
            torch.zeros(2, 2, dtype=torch.int32), requires_grad=False
        )
        kv_b_proj.quant_method = SimpleNamespace(
            quant_config=SimpleNamespace(get_name=lambda: "gguf")
        )
        self_attn = SimpleNamespace(kv_b_proj=kv_b_proj)
        loader = SimpleNamespace(
            model=SimpleNamespace(start_layer=0, end_layer=1, layers=[self_attn]),
            config=SimpleNamespace(num_hidden_layers=1),
            quant_config=SimpleNamespace(get_name=lambda: "gguf"),
        )
        loader.model.layers = [SimpleNamespace(self_attn=self_attn)]

        with self.assertRaises(NotImplementedError) as ctx:
            DeepseekV2WeightLoaderMixin.post_load_weights(loader)
        message = str(ctx.exception)
        self.assertIn("gguf", message)
        self.assertIn("kv_b_proj", message)
        self.assertIn("V4", message)

    def test_fp8_wo_a_gemm_refuses_gguf(self):
        # DSV4's fp8 wo_a einsum reads a dense weight plus block scales.
        from sglang.srt.models import deepseek_v4

        self.assertFalse(
            deepseek_v4._FP8_WO_A_GEMM,
            "the module constant must be off for the rest of this file",
        )
        with mock.patch.object(deepseek_v4, "_FP8_WO_A_GEMM", True):
            _ensure_dist_initialized()
            saved = get_context()._server_args
            get_context().set_server_args(ServerArgs(model_path="dummy", device="cpu"))
            try:
                from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config

                config = DeepSeekV4Config(**_V4_CONFIG)
                with self.assertRaises(NotImplementedError) as ctx:
                    with torch.device("meta"):
                        deepseek_v4.DeepseekV4ForCausalLM(
                            config=config, quant_config=GGUFConfig()
                        )
            finally:
                if saved is None:
                    reset_context()
                else:
                    get_context().set_server_args(saved)
        self.assertIn("gguf", str(ctx.exception))
        self.assertIn("SGLANG_OPT_FP8_WO_A_GEMM", str(ctx.exception))

    def test_cpu_amx_shared_expert_kernel_refuses_packed_weights(self):
        # forward_cpu hands the two dense shared-expert tensors straight to
        # shared_expert_cpu; there is no packed variant of that kernel.
        from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

        gate_up_proj = nn.Module()
        gate_up_proj.qweight = nn.Parameter(
            torch.zeros(2, 2, dtype=torch.int32), requires_grad=False
        )
        gate_up_proj.quant_method = SimpleNamespace(
            quant_config=SimpleNamespace(get_name=lambda: "gguf")
        )
        moe = SimpleNamespace(shared_experts=SimpleNamespace(gate_up_proj=gate_up_proj))

        with self.assertRaises(NotImplementedError) as ctx:
            DeepseekV2MoE.forward_cpu(moe, torch.zeros(1, 2))
        self.assertIn("gguf", str(ctx.exception))
        self.assertIn("shared_expert_cpu", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
