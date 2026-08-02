# SPDX-License-Identifier: Apache-2.0
"""#446: the three siblings of the #443 ``is_packed_weight`` family.

#443 replaced a quantization-name enumeration with a structural test in the
DeepSeek files. The same enumeration, and the same class of mistake, sat in
three more places. This file pins all three.

(a) ``Glm4MoeSparseMoeBlock.__init__`` carried the ``{"awq", "awq_marlin",
    "moe_wna16"}`` list next to an unguarded ``.weight.dtype`` read. A GGUF or
    GPTQ-family GLM-4-MoE checkpoint is packed but unnamed by the list, so the
    guard said "dense" and the dtype read raised ``AttributeError:
    'MergedColumnParallelLinear' object has no attribute 'weight'`` while the
    block was still being constructed -- the checkpoint could not be loaded at
    all. ``TestGlm4MoeSharedExpertConstruction`` is the falsifier.

(b) ``Glm4MoeLiteSparseMoeBlock.__init__`` carried a degenerate variant:
    ``hasattr(gate_up_proj.quant_method, "quant_config")``, with no name test
    at all. That is true for every linear method that keeps its config --
    ``Fp8LinearMethod`` included -- so ``shared_experts_is_fp8`` could never
    become True. No crash, just an answer that was always the same one.
    ``TestGlm4MoeLiteSharedExpertDtypeGates`` is the falsifier.

(c) The ``cat_dim`` choice when fusing ``q_a_proj`` and ``kv_a_proj_with_mqa``
    into ``fused_qkv_a_proj_with_mqa`` used the same three-name list to decide
    which axis carries the output features. GPTQ's ``qweight`` is
    ``[in // pack_factor, out]`` -- output on axis 1, exactly like AWQ -- but
    is not in the list, so it was concatenated along the input axis. GPTQ also
    registers ``g_idx``, one entry per INPUT channel, which is not a
    concatenation in either direction. ``TestFuseQKvAProjLayout`` and
    ``TestFuseQKvAProjRefusesUnfusableGIdx`` are the falsifiers; see
    ``docs/dev/NOTE_446_gptq_cat_dim.md`` for the layout analysis.

WHAT MUST NOT MOVE
``TestGlm4MoeSharedExpertConstruction`` and
``TestGlm4MoeLiteSharedExpertDtypeGates`` also pin the fp8/int8/dense verdicts
that were already right, and ``TestFuseQKvAProjLayout`` pins axis 0 for every
format that had it (dense, fp8, GGUF, compressed-tensors) and axis 1 for the
AWQ family.
"""

import os

os.environ.setdefault("SGLANG_OPT_FP8_WO_A_GEMM", "0")

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from sglang.srt.runtime_context import get_context, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# The enumeration both retired sites carried, reproduced so the tests can state
# what changed instead of describing it.
_RETIRED_AXIS1_QUANT_NAMES = {"awq", "awq_marlin", "moe_wna16"}


def _retired_cat_dim(quant_config) -> int:
    """The cat_dim choice this task replaced, reproduced verbatim."""
    if (
        quant_config is not None
        and quant_config.get_name() in _RETIRED_AXIS1_QUANT_NAMES
    ):
        return 1
    return 0


def _retired_lite_packed_verdict(layer) -> bool:
    """The glm4_moe_lite predicate this task replaced, reproduced verbatim."""
    return hasattr(getattr(layer, "quant_method", None), "quant_config")


def _ensure_dist_initialized() -> None:
    """Minimal single-rank gloo world plus TP=1 model-parallel groups."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29646")
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


def _gptq_config():
    from sglang.srt.layers.quantization.gptq.gptq import GPTQConfig

    return GPTQConfig(
        weight_bits=4,
        group_size=_GROUP_SIZE,
        desc_act=False,
        lm_head_quantized=False,
        dynamic={},
    )


def _awq_config():
    from sglang.srt.layers.quantization.awq.awq import AWQConfig

    return AWQConfig(weight_bits=4, group_size=_GROUP_SIZE, zero_point=True)


def _fp8_block_config():
    from sglang.srt.layers.quantization.fp8 import Fp8Config

    return Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])


def _int8_config():
    from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config

    return W8A8Int8Config()


def _gguf_config():
    from sglang.srt.layers.quantization.gguf import GGUFConfig

    return GGUFConfig()


def _moe_config(quant_method: str) -> SimpleNamespace:
    """Minimal GLM-4-MoE sparse-block config with one shared expert."""
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
        num_nextn_predict_layers=0,
        quantization_config={"quant_method": quant_method},
    )


class _ServerArgsFixture(CustomTestCase):
    """Publishes a CPU ServerArgs and restores whatever was there before."""

    def setUp(self):
        _ensure_dist_initialized()
        self._saved_server_args = get_context()._server_args
        server_args = ServerArgs(model_path="dummy", device="cpu")
        server_args.disable_shared_experts_fusion = True
        get_context().set_server_args(server_args)
        self.server_args = server_args

    def tearDown(self):
        if self._saved_server_args is None:
            reset_context()
        else:
            get_context().set_server_args(self._saved_server_args)

    def _build_block(self, cls, quant_method, quant_config):
        with torch.device("meta"):
            return cls(
                config=_moe_config(quant_method),
                layer_id=0,
                quant_config=quant_config,
                prefix="model.layers.0.mlp",
            )


class TestGlm4MoeSharedExpertConstruction(_ServerArgsFixture):
    """(a) The falsifier: on the unfixed tree the GGUF case never returns."""

    def _build(self, quant_method, quant_config):
        from sglang.srt.models.glm4_moe import Glm4MoeSparseMoeBlock

        return self._build_block(Glm4MoeSparseMoeBlock, quant_method, quant_config)

    def test_gguf_block_constructs_and_selects_neither_fast_path(self):
        moe = self._build("gguf", _gguf_config())
        gate_up = moe.shared_experts.gate_up_proj

        # The layer really is packed: qweight, and no weight to read a dtype
        # from. Before this fix, reaching this line was impossible -- __init__
        # raised AttributeError on `gate_up_proj.weight.dtype`.
        self.assertTrue(hasattr(gate_up, "qweight"))
        self.assertFalse(hasattr(gate_up, "weight"))
        self.assertFalse(moe.shared_experts_is_int8)
        self.assertFalse(moe.shared_experts_is_fp8)
        self.assertIsNone(moe.shared_experts_weight_block_size)

    def test_block_fp8_still_selects_the_fp8_path(self):
        moe = self._build("fp8", _fp8_block_config())
        self.assertTrue(moe.shared_experts_is_fp8)
        self.assertFalse(moe.shared_experts_is_int8)
        self.assertEqual(moe.shared_experts_weight_block_size, [128, 128])

    def test_w8a8_int8_still_selects_the_int8_path(self):
        moe = self._build("w8a8_int8", _int8_config())
        self.assertTrue(moe.shared_experts_is_int8)
        self.assertFalse(moe.shared_experts_is_fp8)

    def test_unquantized_selects_neither_fast_path(self):
        moe = self._build("none", None)
        self.assertFalse(moe.shared_experts_is_int8)
        self.assertFalse(moe.shared_experts_is_fp8)


class TestGlm4MoeLiteSharedExpertDtypeGates(_ServerArgsFixture):
    """(b) The falsifier: on the unfixed tree fp8 answers False here."""

    def _build(self, quant_method, quant_config):
        from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteSparseMoeBlock

        return self._build_block(Glm4MoeLiteSparseMoeBlock, quant_method, quant_config)

    def test_block_fp8_selects_the_fp8_path(self):
        moe = self._build("fp8", _fp8_block_config())
        gate_up = moe.shared_experts.gate_up_proj

        # The retired predicate called this dense-fp8 layer "packed" -- it only
        # asked whether the linear method kept its config around.
        self.assertTrue(_retired_lite_packed_verdict(gate_up))
        self.assertEqual(gate_up.weight.dtype, torch.float8_e4m3fn)
        # The verdict itself -- False before this fix.
        self.assertTrue(moe.shared_experts_is_fp8)
        self.assertFalse(moe.shared_experts_is_int8)

    def test_w8a8_int8_still_selects_the_int8_path(self):
        # The one quantization the retired predicate happened to get right:
        # W8A8Int8LinearMethod does not keep a `quant_config` attribute.
        moe = self._build("w8a8_int8", _int8_config())
        self.assertFalse(_retired_lite_packed_verdict(moe.shared_experts.gate_up_proj))
        self.assertTrue(moe.shared_experts_is_int8)
        self.assertFalse(moe.shared_experts_is_fp8)

    def test_gguf_selects_neither_fast_path(self):
        moe = self._build("gguf", _gguf_config())
        self.assertFalse(hasattr(moe.shared_experts.gate_up_proj, "weight"))
        self.assertFalse(moe.shared_experts_is_int8)
        self.assertFalse(moe.shared_experts_is_fp8)

    def test_unquantized_selects_neither_fast_path(self):
        moe = self._build("none", None)
        self.assertFalse(moe.shared_experts_is_int8)
        self.assertFalse(moe.shared_experts_is_fp8)


# MLA a-projection geometry: hidden 256 in, 256 + 128 = 384 out. Every output
# split is a multiple of the 4-bit pack factor 8 and of the 128-wide fp8 block,
# and the input is a multiple of the group size, so every real quantization
# scheme accepts these shapes and every parameter it registers splits cleanly
# back into the two checkpoint tensors it was assembled from.
_HIDDEN = 256
_Q_LORA_RANK = 256
_KV_A_OUT = 128
_FUSED_OUT = _Q_LORA_RANK + _KV_A_OUT
_GROUP_SIZE = 32


class _FusedAProjFixture(CustomTestCase):
    """Builds the real fused layer for a quantization and hands out its params."""

    def setUp(self):
        _ensure_dist_initialized()
        self._saved_server_args = get_context()._server_args
        get_context().set_server_args(ServerArgs(model_path="dummy", device="cpu"))

    def tearDown(self):
        if self._saved_server_args is None:
            reset_context()
        else:
            get_context().set_server_args(self._saved_server_args)

    def _fused_params(self, quant_config):
        """``{name: param}`` of a real ``fused_qkv_a_proj_with_mqa``."""
        from sglang.srt.layers.linear import ReplicatedLinear

        with torch.device("meta"):
            layer = ReplicatedLinear(
                _HIDDEN,
                _FUSED_OUT,
                bias=False,
                quant_config=quant_config,
                prefix="model.layers.0.self_attn.fused_qkv_a_proj_with_mqa",
            )
        return dict(layer.named_parameters())

    def _source_pair(self, param):
        """The two checkpoint tensors this fused param is assembled from.

        Split the fused parameter's own shape back along its output axis, which
        is what a real checkpoint stores as ``q_a_proj.<suffix>`` and
        ``kv_a_proj_with_mqa.<suffix>``.
        """
        output_dim = getattr(param, "output_dim", None)
        shape = list(param.shape)
        if output_dim is None:
            # No output axis: both sources carry the identical vector.
            data = torch.arange(shape[0] if shape else 1, dtype=param.dtype)
            return data, data.clone()
        ratio = _FUSED_OUT // shape[output_dim]
        q_shape, kv_shape = list(shape), list(shape)
        q_shape[output_dim] = _Q_LORA_RANK // ratio
        kv_shape[output_dim] = _KV_A_OUT // ratio
        return (
            torch.zeros(q_shape, dtype=param.dtype),
            torch.ones(kv_shape, dtype=param.dtype),
        )


class TestFuseQKvAProjLayout(_FusedAProjFixture):
    """(c) The fused tensor must come out with the destination's shape."""

    def _assert_round_trip(self, quant_config, expected_axes):
        from sglang.srt.models.deepseek_common.utils import fuse_q_kv_a_proj

        seen = {}
        for name, param in self._fused_params(quant_config).items():
            q_src, kv_src = self._source_pair(param)
            fused = fuse_q_kv_a_proj(param, name, q_src, kv_src)
            self.assertEqual(
                tuple(fused.shape),
                tuple(param.shape),
                f"{name}: fused {tuple(fused.shape)} != destination "
                f"{tuple(param.shape)}",
            )
            seen[name] = getattr(param, "output_dim", None)
        self.assertEqual(seen, expected_axes)

    def test_gptq_fuses_along_the_output_axis(self):
        # The falsifier. GPTQ's qweight is [in // pack_factor, out]: output on
        # axis 1, the same as AWQ. The retired list did not name gptq, so the
        # concatenation ran along axis 0 and could not produce the
        # destination's shape at all.
        quant_config = _gptq_config()
        self.assertEqual(_retired_cat_dim(quant_config), 0)
        self._assert_round_trip(
            quant_config,
            {"qweight": 1, "g_idx": None, "qzeros": 1, "scales": 1},
        )

    def test_the_retired_list_could_not_have_fused_gptq(self):
        # Explicitly: axis 0 is not merely a different answer, it is an
        # impossible one -- the two source shards differ on the other axis.
        param = self._fused_params(_gptq_config())["qweight"]
        q_src, kv_src = self._source_pair(param)
        with self.assertRaises(RuntimeError):
            torch.cat([q_src, kv_src], dim=_retired_cat_dim(_gptq_config()))

    def test_awq_keeps_the_output_axis_it_had(self):
        quant_config = _awq_config()
        self.assertEqual(_retired_cat_dim(quant_config), 1)
        self._assert_round_trip(quant_config, {"qweight": 1, "qzeros": 1, "scales": 1})

    def test_unquantized_keeps_axis_zero(self):
        self._assert_round_trip(None, {"weight": 0})

    def test_block_fp8_keeps_axis_zero(self):
        self._assert_round_trip(
            _fp8_block_config(), {"weight": 0, "weight_scale_inv": 0}
        )

    def test_w8a8_int8_keeps_axis_zero(self):
        params = self._fused_params(_int8_config())
        self.assertEqual(getattr(params["weight"], "output_dim", None), 0)
        self._assert_round_trip(
            _int8_config(),
            {name: 0 for name in params},
        )


class TestFuseQKvAProjScalarMarkers(_FusedAProjFixture):
    """A 0-d checkpoint tensor is a marker, not a shard: one value survives."""

    def test_a_scalar_pair_yields_a_single_scalar(self):
        from sglang.srt.models.deepseek_common.utils import fuse_q_kv_a_proj

        marker = torch.tensor(14, dtype=torch.uint8)
        fused = fuse_q_kv_a_proj(
            nn.Parameter(torch.zeros(1), requires_grad=False),
            "fused_qkv_a_proj_with_mqa.qweight_type",
            marker,
            marker.clone(),
        )
        self.assertEqual(fused.shape, torch.Size([]))
        self.assertEqual(fused.item(), 14)


class TestFuseQKvAProjRefusesUnfusableGIdx(_FusedAProjFixture):
    """(c) ``g_idx`` is per-input-channel: one copy, or a named refusal."""

    def _g_idx_param(self):
        return self._fused_params(_gptq_config())["g_idx"]

    def test_matching_g_idx_is_passed_through_not_concatenated(self):
        from sglang.srt.models.deepseek_common.utils import fuse_q_kv_a_proj

        param = self._g_idx_param()
        self.assertIsNone(getattr(param, "output_dim", None))
        self.assertEqual(param.input_dim, 0)

        g_idx = torch.arange(_HIDDEN, dtype=torch.int32) // _GROUP_SIZE
        fused = fuse_q_kv_a_proj(param, "…g_idx", g_idx, g_idx.clone())

        self.assertEqual(tuple(fused.shape), tuple(param.shape))
        self.assertTrue(torch.equal(fused, g_idx))
        # Concatenating would have described twice as many input channels as
        # the layer has.
        self.assertEqual(torch.cat([g_idx, g_idx]).shape[0], 2 * param.shape[0])

    def test_differing_g_idx_is_refused_by_name(self):
        from sglang.srt.models.deepseek_common.utils import (
            UnfusableAProjParameter,
            fuse_q_kv_a_proj,
        )

        param = self._g_idx_param()
        g_idx = torch.arange(_HIDDEN, dtype=torch.int32) // _GROUP_SIZE
        reordered = g_idx.flip(0).contiguous()

        with self.assertRaises(UnfusableAProjParameter) as caught:
            fuse_q_kv_a_proj(
                param, "…fused_qkv_a_proj_with_mqa.g_idx", g_idx, reordered
            )

        message = str(caught.exception)
        self.assertIn("fused_qkv_a_proj_with_mqa.g_idx", message)
        self.assertIn("desc_act", message)

    def test_the_refusal_can_fire_only_when_the_two_differ(self):
        # The can-fail proof for the guard: identical input never refuses.
        from sglang.srt.models.deepseek_common.utils import fuse_q_kv_a_proj

        param = self._g_idx_param()
        g_idx = torch.zeros(_HIDDEN, dtype=torch.int32)
        fuse_q_kv_a_proj(param, "…g_idx", g_idx, g_idx.clone())


class TestEveryFusionSiteUsesTheHelper(CustomTestCase):
    """The enumeration existed in four copies; none may be left behind."""

    _SITES = (
        "python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py",
        "python/sglang/srt/models/longcat_flash.py",
        "python/sglang/srt/models/longcat_flash_nextn.py",
        "python/sglang/srt/models/bailing_moe_linear.py",
    )

    def _repo_root(self):
        import sglang

        return os.path.abspath(
            os.path.join(os.path.dirname(sglang.__file__), os.pardir, os.pardir)
        )

    def test_no_site_still_carries_the_quant_name_enumeration(self):
        root = self._repo_root()
        for site in self._SITES:
            path = os.path.join(root, site)
            with self.subTest(site=site):
                source = open(path, encoding="utf-8").read()
                self.assertIn("fuse_q_kv_a_proj", source)
                self.assertNotIn("cat_dim", source)


if __name__ == "__main__":
    unittest.main()
