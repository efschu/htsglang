# SPDX-License-Identifier: Apache-2.0
"""#443: DeepSeek classifies packed layers by what was built, not by name.

THE DEFECT, as it stood
Two sites decided "is this layer packed?" by matching the quantization
method's name against a fixed set::

    {"awq", "awq_marlin", "moe_wna16"}

  * ``DeepseekV2MoE.__init__``          -- the int8 / fp8 shared-expert gates
  * ``DeepseekV2AttentionMLA.__init__`` -- ``self.is_packed_weight``, a public
                                           attribute that ``kimi_k25_eagle3``
                                           also writes and that the CPU
                                           fused-rope selector reads

The set is an enumeration, so every packed format it does not name is
classified as *dense*: ``gptq``, ``auto-round``, ``gguf`` and the packed
``compressed-tensors`` schemes all register ``qweight`` and no ``weight``, and
all of them answered False. #401 had already made the two consumers inside this
tree read their dtype through ``dense_weight_dtype``, so the misclassification
no longer reaches an ``AttributeError`` here -- but ``is_packed_weight`` is a
published attribute that still answered the wrong thing about a packed layer,
which is exactly the trap the guard exists to prevent.

THE RULE (ours, from #33265): packed-ness is ``getattr(layer, "weight", None)
is None``. It is a property of the layer that was built, it cannot go stale
when a new packed format is added, and it needs no list to maintain.

WHAT FAILS WITHOUT THE FIX
``TestAttentionPackedFlagIsStructural`` -- a GGUF and a GPTQ DeepSeek-V3 both
report ``self_attn.is_packed_weight is False`` on the unfixed tree although the
layer carries ``qweight`` and no ``weight``.

WHAT MUST NOT MOVE
``TestNameListVerdictsPreserved`` and ``TestSharedExpertDtypeGatesUnchanged``
pin that the AWQ family and every dense checkpoint keep the verdict and the
fast path they had before.
"""

import os

# The DSV4 attention module captures SGLANG_OPT_FP8_WO_A_GEMM at import time.
os.environ.setdefault("SGLANG_OPT_FP8_WO_A_GEMM", "0")

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from sglang.srt.runtime_context import get_context, get_parallel, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

# The enumeration the two sites used to carry, kept here so the tests can
# state what changed instead of describing it.
_RETIRED_PACKED_QUANT_NAMES = {"awq", "awq_marlin", "moe_wna16"}


def _retired_name_verdict(layer) -> bool:
    """The classification this task replaced, reproduced verbatim."""
    quant_method = getattr(layer, "quant_method", None)
    return (
        hasattr(quant_method, "quant_config")
        and quant_method.quant_config.get_name() in _RETIRED_PACKED_QUANT_NAMES
    )


def _packed_layer(quant_name, packed_attr="qweight"):
    """A layer as a packed quantization method leaves it: no ``weight``."""
    layer = nn.Module()
    setattr(
        layer,
        packed_attr,
        nn.Parameter(torch.zeros(2, 2, dtype=torch.int32), requires_grad=False),
    )
    if quant_name is not None:
        layer.quant_method = SimpleNamespace(
            quant_config=SimpleNamespace(get_name=lambda: quant_name)
        )
    return layer


def _dense_layer(quant_name, dtype=torch.bfloat16):
    layer = nn.Module()
    layer.weight = nn.Parameter(
        torch.zeros(2, 2, dtype=torch.float32).to(dtype), requires_grad=False
    )
    if quant_name is not None:
        layer.quant_method = SimpleNamespace(
            quant_config=SimpleNamespace(get_name=lambda: quant_name)
        )
    return layer


def _ensure_dist_initialized() -> None:
    """Minimal single-rank gloo world plus TP=1 model-parallel groups.

    Every DeepSeek submodule reads the parallel context inside ``__init__``
    to size its shards, so the groups must exist before any construction.
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29643")
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


# Scaled-down V3. Every linear dimension the attention module cuts is a
# multiple of 32 so a real group-wise GPTQ checkpoint is legal at group_size
# 32; that is what lets the GPTQ case below build the actual packed layers
# instead of a mock.
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
    """Minimal DeepseekV2MoE config with one shared expert."""
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


class TestIsPackedLayerIsStructural(CustomTestCase):
    """The predicate itself: it reads the layer, never a quantization name."""

    def test_dense_weights_of_every_dtype_are_not_packed(self):
        from sglang.srt.models.deepseek_common.utils import is_packed_layer

        for dtype in (
            torch.bfloat16,
            torch.float16,
            torch.int8,
            torch.uint8,
            torch.float8_e4m3fn,
        ):
            with self.subTest(dtype=dtype):
                self.assertFalse(is_packed_layer(_dense_layer(None, dtype)))

    def test_a_qweight_only_layer_is_packed(self):
        from sglang.srt.models.deepseek_common.utils import is_packed_layer

        layer = _packed_layer(None)
        self.assertTrue(is_packed_layer(layer))
        with self.assertRaises(AttributeError):
            _ = layer.weight

    def test_a_weight_packed_only_layer_is_packed(self):
        # compressed-tensors registers `weight_packed`; DeepseekV2MLP aliases
        # it onto `weight` afterwards, but the predicate must be right for the
        # window before that and for every other caller.
        from sglang.srt.models.deepseek_common.utils import is_packed_layer

        self.assertTrue(is_packed_layer(_packed_layer(None, "weight_packed")))

    def test_a_layer_with_no_weights_at_all_is_packed(self):
        from sglang.srt.models.deepseek_common.utils import is_packed_layer

        self.assertTrue(is_packed_layer(nn.Module()))

    def test_it_agrees_with_dense_weight_dtype(self):
        # The two helpers must not be able to disagree; several sites use one
        # and several the other on the same layer.
        from sglang.srt.models.deepseek_common.utils import (
            dense_weight_dtype,
            is_packed_layer,
        )

        for layer in (
            _dense_layer(None, torch.bfloat16),
            _dense_layer(None, torch.int8),
            _packed_layer(None),
            _packed_layer(None, "weight_packed"),
            nn.Module(),
        ):
            self.assertEqual(is_packed_layer(layer), dense_weight_dtype(layer) is None)


class TestNameListBlindSpots(CustomTestCase):
    """What the retired enumeration got wrong, format by format."""

    def test_packed_formats_the_name_list_missed_are_now_packed(self):
        from sglang.srt.models.deepseek_common.utils import is_packed_layer

        for quant_name in ("gptq", "gptq_marlin", "auto-round", "gguf", "bitsandbytes"):
            with self.subTest(quant=quant_name):
                layer = _packed_layer(quant_name)
                self.assertFalse(_retired_name_verdict(layer))
                self.assertTrue(is_packed_layer(layer))

    def test_compressed_tensors_packed_scheme_is_now_packed(self):
        from sglang.srt.models.deepseek_common.utils import is_packed_layer

        layer = _packed_layer("compressed-tensors", "weight_packed")
        self.assertFalse(_retired_name_verdict(layer))
        self.assertTrue(is_packed_layer(layer))


class TestNameListVerdictsPreserved(CustomTestCase):
    """Everything the enumeration did classify keeps its answer."""

    def test_the_awq_family_is_still_packed(self):
        from sglang.srt.models.deepseek_common.utils import is_packed_layer

        for quant_name in sorted(_RETIRED_PACKED_QUANT_NAMES):
            with self.subTest(quant=quant_name):
                layer = _packed_layer(quant_name)
                self.assertTrue(_retired_name_verdict(layer))
                self.assertTrue(is_packed_layer(layer))

    def test_dense_quantizations_are_still_dense(self):
        from sglang.srt.models.deepseek_common.utils import is_packed_layer

        for quant_name, dtype in (
            (None, torch.bfloat16),
            ("fp8", torch.float8_e4m3fn),
            ("w8a8_int8", torch.int8),
            ("compressed-tensors", torch.float8_e4m3fn),
        ):
            with self.subTest(quant=quant_name):
                layer = _dense_layer(quant_name, dtype)
                self.assertFalse(_retired_name_verdict(layer))
                self.assertFalse(is_packed_layer(layer))

    def test_an_unquantized_layer_inside_an_awq_checkpoint_stays_dense(self):
        # Modules on the ignore list get UnquantizedLinearMethod, which has no
        # `quant_config`; both the old and the new rule must call them dense.
        from sglang.srt.models.deepseek_common.utils import is_packed_layer

        layer = _dense_layer(None)
        layer.quant_method = SimpleNamespace()
        self.assertFalse(_retired_name_verdict(layer))
        self.assertFalse(is_packed_layer(layer))


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


class TestAttentionPackedFlagIsStructural(_ServerArgsFixture):
    """The falsifier: on the unfixed tree these report False.

    ``DeepseekV2AttentionMLA.is_packed_weight`` is read by the CPU fused-rope
    selector and written by ``kimi_k25_eagle3``, so it has to be the truth
    about the layer rather than the truth about a name list.
    """

    def _build_v3(self, quant_config, **config_overrides):
        from transformers import DeepseekV3Config

        from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM

        config = DeepseekV3Config(**{**_V3_CONFIG, **config_overrides})
        with torch.device("meta"), get_parallel().override(
            attn_dp_size=1, attn_dp_rank=0
        ):
            return DeepseekV2ForCausalLM(config=config, quant_config=quant_config)

    def test_gguf_attention_reports_packed(self):
        from sglang.srt.layers.quantization.gguf import GGUFConfig

        model = self._build_v3(GGUFConfig())
        attn = model.model.layers[1].self_attn

        self.assertTrue(attn.has_fused_proj)
        self.assertTrue(hasattr(attn.fused_qkv_a_proj_with_mqa, "qweight"))
        self.assertFalse(hasattr(attn.fused_qkv_a_proj_with_mqa, "weight"))
        # The verdict itself -- False before this fix.
        self.assertTrue(attn.is_packed_weight)
        # ...and the fast paths it gates stay off, as they already did.
        self.assertFalse(attn.use_min_latency_fused_a_gemm)
        self.assertFalse(attn.qkv_proj_with_rope_is_int8)
        self.assertFalse(attn.qkv_proj_with_rope_is_fp8)

    def test_gptq_attention_reports_packed(self):
        # A real GPTQ config, not a mock: this is the format the name list
        # missed. GPTQ has no MoE method ("please use gptq_marlin"), so the
        # model is built with dense MLPs -- the attention module, which is the
        # site under test, is identical either way.
        from sglang.srt.layers.quantization.gptq.gptq import GPTQConfig

        model = self._build_v3(
            GPTQConfig(
                weight_bits=4,
                group_size=32,
                desc_act=False,
                lm_head_quantized=False,
                dynamic={},
            ),
            first_k_dense_replace=_V3_CONFIG["num_hidden_layers"],
        )
        attn = model.model.layers[1].self_attn

        self.assertTrue(attn.has_fused_proj)
        self.assertTrue(hasattr(attn.fused_qkv_a_proj_with_mqa, "qweight"))
        self.assertFalse(hasattr(attn.fused_qkv_a_proj_with_mqa, "weight"))
        # The verdict itself -- False before this fix.
        self.assertTrue(attn.is_packed_weight)
        self.assertFalse(attn.use_min_latency_fused_a_gemm)
        self.assertFalse(attn.qkv_proj_with_rope_is_int8)
        self.assertFalse(attn.qkv_proj_with_rope_is_fp8)

    def test_unquantized_attention_reports_dense(self):
        # Neutrality: a dense checkpoint answers exactly what it answered
        # before, and the fused-a fast path is still decided by dtype/shape.
        model = self._build_v3(None)
        attn = model.model.layers[1].self_attn

        self.assertTrue(attn.has_fused_proj)
        self.assertTrue(hasattr(attn.fused_qkv_a_proj_with_mqa, "weight"))
        self.assertFalse(attn.is_packed_weight)


class TestEagle3ReplacementLayerUsesTheSameRule(_ServerArgsFixture):
    """``kimi_k25_eagle3`` swaps in a wider fused QKV-down projection and then
    recomputes ``is_packed_weight`` for it. That second writer carried its own
    copy of the name list, so it has to move with the class it writes into."""

    def _build_layer(self, quant_config):
        from transformers import DeepseekV3Config

        from sglang.srt.models.kimi_k25_eagle3 import Eagle3MLADecoderLayer

        config = DeepseekV3Config(**_V3_CONFIG)
        with torch.device("meta"), get_parallel().override(
            attn_dp_size=1, attn_dp_rank=0
        ):
            return Eagle3MLADecoderLayer(
                config=config, layer_id=0, quant_config=quant_config, prefix="draft"
            )

    def test_gguf_replacement_projection_reports_packed(self):
        from sglang.srt.layers.quantization.gguf import GGUFConfig

        attn = self._build_layer(GGUFConfig()).self_attn
        self.assertTrue(hasattr(attn.fused_qkv_a_proj_with_mqa, "qweight"))
        self.assertFalse(hasattr(attn.fused_qkv_a_proj_with_mqa, "weight"))
        # False before this fix.
        self.assertTrue(attn.is_packed_weight)
        self.assertFalse(attn.use_min_latency_fused_a_gemm)

    def test_unquantized_replacement_projection_reports_dense(self):
        attn = self._build_layer(None).self_attn
        self.assertTrue(hasattr(attn.fused_qkv_a_proj_with_mqa, "weight"))
        self.assertFalse(attn.is_packed_weight)
        self.assertFalse(attn.use_min_latency_fused_a_gemm)


class TestSharedExpertDtypeGatesUnchanged(_ServerArgsFixture):
    """The MoE site dropped the ``not is_packed_weight`` conjunct.

    That is only safe because ``dense_weight_dtype`` already answers None for
    every packed layer, so the conjunct could never change an outcome. These
    pin both directions on the real ``DeepseekV2MoE``.
    """

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

    def test_gguf_selects_neither_fast_path(self):
        from sglang.srt.layers.quantization.gguf import GGUFConfig

        moe = self._build("gguf", GGUFConfig())
        self.assertTrue(hasattr(moe.shared_experts.gate_up_proj, "qweight"))
        self.assertFalse(moe.shared_experts_is_int8)
        self.assertFalse(moe.shared_experts_is_fp8)
        self.assertIsNone(moe.shared_experts_weight_block_size)

    def test_unquantized_selects_neither_fast_path(self):
        moe = self._build("none", None)
        self.assertFalse(moe.shared_experts_is_int8)
        self.assertFalse(moe.shared_experts_is_fp8)

    def test_block_fp8_still_selects_the_fp8_path(self):
        from sglang.srt.layers.quantization.fp8 import Fp8Config

        moe = self._build(
            "fp8",
            Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]),
        )
        self.assertTrue(moe.shared_experts_is_fp8)
        self.assertFalse(moe.shared_experts_is_int8)
        self.assertEqual(moe.shared_experts_weight_block_size, [128, 128])

    def test_w8a8_int8_still_selects_the_int8_path(self):
        from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config

        moe = self._build("w8a8_int8", W8A8Int8Config())
        self.assertTrue(moe.shared_experts_is_int8)
        self.assertFalse(moe.shared_experts_is_fp8)


if __name__ == "__main__":
    unittest.main()
