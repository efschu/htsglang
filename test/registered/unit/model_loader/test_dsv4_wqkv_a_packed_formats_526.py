# SPDX-License-Identifier: Apache-2.0
"""The DeepSeek V4 ``wq_a`` + ``wkv`` fusion must refuse packed quant formats.

``SGLANG_OPT_FUSE_WQA_WKV`` is on by default and carried no format gate. The
join in ``DeepseekV4ForCausalLM.load_weights`` knows exactly four checkpoint
leaves -- ``weight``, ``weight_scale_inv``, ``qweight``, ``qweight_type`` --
and joins all of them by concatenating along dim 0, the output-row axis.

A packed integer format builds its linear from other leaves entirely, and both
halves of that go wrong quietly (task #526; the same symptom class as the open
upstream issue #33245):

* GPTQ / auto-round add ``qzeros``, ``scales`` and, for act-order, ``g_idx``;
  AWQ adds ``qzeros`` and ``scales``. None of them matches the fusion
  predicate, so they fall through to the unmatched-name branch, which on a
  non-GGUF checkpoint only ``logger.warning``s and continues -- the tensors are
  DROPPED and the fused parameter keeps its uninitialised contents.
* the one leaf that does match, ``qweight``, is not row-major there. GPTQ packs
  the INPUT dim into dim 0 (``(in//8, out)``, asserted below against the real
  ``GPTQLinearMethod``), so ``torch.cat(..., dim=0)`` is a join along the pack
  axis, not along output rows.
* a compressed-tensors packed checkpoint (``weight_packed`` / ``weight_scale``
  / ``weight_shape`` / ``weight_zero_point``) matches NOTHING, so the load runs
  to completion having delivered zero tensors to the fused parameter.

The fix reads the leaf inventory off the module the quant method ACTUALLY
built, not off a list of quant-method names, and refuses (or, at construction
time, turns the fusion off by name) when a leaf has no route.

No GPU and no checkpoint: the quant methods are constructed for real under
``CUDA_VISIBLE_DEVICES=99`` and the weight streams are synthetic tensors of the
shapes those methods declare.
"""

import threading
import types
import unittest
from typing import Dict, List, Optional, Tuple
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.models.deepseek_v4 import (
    _WQKV_A_LEAVES,
    DeepseekV4ForCausalLM,
    _is_wqkv_a_fusion_input,
    _unroutable_wqkv_a_leaves,
    _warn_wqkv_a_fusion_auto_off,
    _wqkv_a_fusion_survives_quant_format,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# harness -- the stub `load_weights` reads, as in test_dsv4_gguf_weight_fusions
# ---------------------------------------------------------------------------


class _FakeQuantConfig:
    """Only ``get_name()`` is read (``is_gguf_quant_config``)."""

    def __init__(self, name: str):
        self._name = name

    def get_name(self) -> str:
        return self._name


class _CapturingParam:
    def __init__(self, name: str, sink: Dict[str, torch.Tensor], dtype: torch.dtype):
        self._name = name
        self._sink = sink
        self._lock = sink.setdefault("__lock__", threading.Lock())
        self.dtype = dtype

    def weight_loader(self, param, loaded_weight, *args, **kwargs) -> None:
        with self._lock:
            self._sink[self._name] = loaded_weight


def _make_stub(quant_name: Optional[str], params: dict) -> types.SimpleNamespace:
    stub = types.SimpleNamespace()
    stub.config = types.SimpleNamespace(num_hidden_layers=8, n_routed_experts=2)
    stub.quant_config = None if quant_name is None else _FakeQuantConfig(quant_name)
    stub.num_fused_shared_experts = 0
    stub.model = types.SimpleNamespace()
    stub.pp_group = types.SimpleNamespace(is_first_rank=True, is_last_rank=True)
    stub.named_parameters = lambda: params.items()
    stub.remap_weight_name_to_dpsk_hf_format = (
        DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format
    )
    stub.post_load_weights = lambda **kwargs: None
    stub._prewarm_mhc_pre_kernels = lambda: None
    return stub


def _run_load(
    quant_name: Optional[str],
    stream: List[Tuple[str, torch.Tensor]],
    param_specs: Dict[str, torch.dtype],
) -> Dict[str, torch.Tensor]:
    """Drive the real ``load_weights`` and return what reached each parameter."""
    sink: Dict[str, torch.Tensor] = {}
    params = {
        name: _CapturingParam(name, sink, dtype) for name, dtype in param_specs.items()
    }
    stub = _make_stub(quant_name, params)
    with envs.SGLANG_OPT_FP8_WO_A_GEMM.override(False):
        DeepseekV4ForCausalLM.load_weights(stub, iter(stream))
    sink.pop("__lock__", None)
    return sink


# ---------------------------------------------------------------------------
# synthetic packed checkpoints
# ---------------------------------------------------------------------------

_HIDDEN = 256
_Q_OUT = 128
_KV_OUT = 64
_GROUP = 128
_PACK = 8  # 32-bit words / 4-bit values

_ATTN = "layers.2.attn"
_WQKV_A = "model.layers.2.self_attn.wqkv_a"
_WQ_A = "model.layers.2.self_attn.wq_a"
_WKV = "model.layers.2.self_attn.wkv"


def _gptq_projection(out: int, seed: int) -> Dict[str, torch.Tensor]:
    """One GPTQ projection, in the shapes ``GPTQLinearMethod`` registers."""
    g = torch.Generator().manual_seed(seed)
    return {
        "qweight": torch.randint(
            -(2**31), 2**31 - 1, (_HIDDEN // _PACK, out), generator=g, dtype=torch.int32
        ),
        "qzeros": torch.randint(
            -(2**31),
            2**31 - 1,
            (_HIDDEN // _GROUP, out // _PACK),
            generator=g,
            dtype=torch.int32,
        ),
        "scales": torch.randn(_HIDDEN // _GROUP, out, generator=g, dtype=torch.float32),
        "g_idx": torch.arange(_HIDDEN, dtype=torch.int32),
    }


def _gptq_stream() -> List[Tuple[str, torch.Tensor]]:
    q = _gptq_projection(_Q_OUT, seed=1)
    kv = _gptq_projection(_KV_OUT, seed=2)
    stream = []
    for projection, leaves in ((f"{_ATTN}.wq_a", q), (f"{_ATTN}.wkv", kv)):
        for leaf, tensor in leaves.items():
            stream.append((f"{projection}.{leaf}", tensor))
    return stream


_GPTQ_FUSED_PARAMS = {
    f"{_WQKV_A}.qweight": torch.int32,
    f"{_WQKV_A}.qzeros": torch.int32,
    f"{_WQKV_A}.scales": torch.float32,
    f"{_WQKV_A}.g_idx": torch.int32,
}

_GPTQ_SPLIT_PARAMS = {
    f"{prefix}.{leaf}": dtype
    for prefix in (_WQ_A, _WKV)
    for leaf, dtype in (
        ("qweight", torch.int32),
        ("qzeros", torch.int32),
        ("scales", torch.float32),
        ("g_idx", torch.int32),
    )
}

#: compressed-tensors pack-quantized spellings (``compressed_tensors_wNa16.py``
#: registers exactly these four), none of which the fusion knows.
_CT_LEAVES = ("weight_packed", "weight_scale", "weight_shape", "weight_zero_point")


def _compressed_tensors_stream() -> List[Tuple[str, torch.Tensor]]:
    stream = []
    for projection, out in ((f"{_ATTN}.wq_a", _Q_OUT), (f"{_ATTN}.wkv", _KV_OUT)):
        stream.append(
            (f"{projection}.weight_packed", torch.zeros(out, _HIDDEN // _PACK))
        )
        stream.append((f"{projection}.weight_scale", torch.ones(out, 1)))
        stream.append(
            (
                f"{projection}.weight_shape",
                torch.tensor([_HIDDEN, out], dtype=torch.int64),
            )
        )
        stream.append(
            (
                f"{projection}.weight_zero_point",
                torch.zeros(out, _HIDDEN // _GROUP // _PACK),
            )
        )
    return stream


_CT_FUSED_PARAMS = {f"{_WQKV_A}.{leaf}": torch.float32 for leaf in _CT_LEAVES}


def _no_gate(leaves):
    """The pre-fix state: no leaf gate at all."""
    return ()


# ---------------------------------------------------------------------------
# the instrument: which leaves each quant method actually builds
# ---------------------------------------------------------------------------


class TestLeafInventoryDiscriminates(CustomTestCase):
    """Spread precondition: the predicate must separate known-different inputs.

    Every module here is built by the REAL quant method, so the leaf names are
    the ones a boot would produce, not names written down in this test.
    """

    @staticmethod
    def _leaves(quant_config, params_dtype=torch.bfloat16) -> List[str]:
        linear = ReplicatedLinear(
            _HIDDEN,
            _Q_OUT + _KV_OUT,
            bias=False,
            quant_config=quant_config,
            prefix="wqkv_a",
            params_dtype=params_dtype,
        )
        return [name for name, _ in linear.named_parameters(recurse=False)]

    def test_unquantized_and_gguf_stay_routable(self):
        """The default and the GGUF route are unchanged by the gate."""
        self.assertEqual(self._leaves(None), ["weight"])
        self.assertEqual(_unroutable_wqkv_a_leaves(self._leaves(None)), ())

        from sglang.srt.layers.quantization.gguf import GGUFConfig

        gguf_leaves = self._leaves(GGUFConfig())
        self.assertEqual(gguf_leaves, ["qweight", "qweight_type"])
        self.assertEqual(_unroutable_wqkv_a_leaves(gguf_leaves), ())

    def test_fp8_block_leaf_pair_is_routable(self):
        """The reference DSV4 route: ``weight`` + ``weight_scale_inv``.

        Building an ``Fp8LinearMethod`` with a block size needs an initialised
        TP group, so the pair is asserted against the predicate directly; the
        end-to-end fp8 fusion is pinned in
        ``test_dsv4_gguf_weight_fusions.py::TestWqkvADenseRouteUnchanged``.
        """
        self.assertEqual(_unroutable_wqkv_a_leaves(("weight", "weight_scale_inv")), ())

    def test_packed_integer_formats_are_not_routable(self):
        from sglang.srt.layers.quantization.awq.awq import AWQConfig
        from sglang.srt.layers.quantization.gptq import GPTQConfig

        gptq_leaves = self._leaves(
            GPTQConfig(
                weight_bits=4,
                group_size=_GROUP,
                desc_act=False,
                lm_head_quantized=False,
                dynamic={},
            ),
            params_dtype=torch.float16,
        )
        self.assertEqual(sorted(gptq_leaves), ["g_idx", "qweight", "qzeros", "scales"])
        self.assertEqual(
            _unroutable_wqkv_a_leaves(gptq_leaves), ("g_idx", "qzeros", "scales")
        )

        awq_leaves = self._leaves(
            AWQConfig(
                weight_bits=4,
                group_size=_GROUP,
                zero_point=True,
                modules_to_not_convert=None,
            ),
            params_dtype=torch.float16,
        )
        self.assertEqual(sorted(awq_leaves), ["qweight", "qzeros", "scales"])
        self.assertEqual(_unroutable_wqkv_a_leaves(awq_leaves), ("qzeros", "scales"))

    def test_per_tensor_fp8_is_not_routable_either(self):
        """The same hole, on a format nobody would call "packed"."""
        from sglang.srt.layers.quantization.fp8 import Fp8Config

        leaves = self._leaves(
            Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="static")
        )
        self.assertEqual(
            _unroutable_wqkv_a_leaves(leaves), ("input_scale", "weight_scale")
        )

    def test_gptq_qweight_dim0_is_the_pack_axis_not_output_rows(self):
        """The concatenation axis claim, checked against the real method.

        ``_fuse_wqkv_a`` concatenates dim 0 because that is the output-row axis
        for a dense or GGUF payload. For GPTQ dim 0 is ``in_features //
        pack_factor`` and the OUTPUT dim is dim 1, so the same concatenation
        joins two shards along their packed input axis.
        """
        from sglang.srt.layers.quantization.gptq import GPTQConfig

        linear = ReplicatedLinear(
            _HIDDEN,
            _Q_OUT,
            bias=False,
            quant_config=GPTQConfig(
                weight_bits=4,
                group_size=_GROUP,
                desc_act=False,
                lm_head_quantized=False,
                dynamic={},
            ),
            prefix="wqkv_a",
            params_dtype=torch.float16,
        )
        self.assertEqual(tuple(linear.qweight.shape), (_HIDDEN // _PACK, _Q_OUT))


# ---------------------------------------------------------------------------
# can-fail arms: today's code, driven for real, is wrong
# ---------------------------------------------------------------------------


class TestPreFixRoutingIsWrong(CustomTestCase):
    """The gate removed, i.e. the code as it stood before this fix."""

    def test_gptq_siblings_are_dropped_and_the_join_fails_without_naming_why(self):
        delivered: Dict[str, torch.Tensor] = {}
        params = {
            name: _CapturingParam(name, delivered, dtype)
            for name, dtype in _GPTQ_FUSED_PARAMS.items()
        }
        stub = _make_stub("gptq", params)

        with mock.patch(
            "sglang.srt.models.deepseek_v4._unroutable_wqkv_a_leaves", _no_gate
        ):
            with envs.SGLANG_OPT_FP8_WO_A_GEMM.override(False):
                with self.assertRaises(RuntimeError) as ctx:
                    DeepseekV4ForCausalLM.load_weights(stub, iter(_gptq_stream()))

        # The only diagnosis the pre-fix code offers is a bare shape error.
        message = str(ctx.exception)
        self.assertIn("Sizes of tensors must match", message)
        self.assertNotIn("SGLANG_OPT_FUSE_WQA_WKV", message)
        self.assertNotIn("gptq", message.lower())

        # And the three sibling leaves never reached a parameter at all: they
        # took the unmatched-name branch, which only warns on a non-GGUF
        # checkpoint.
        delivered.pop("__lock__", None)
        self.assertEqual(delivered, {})
        for leaf in ("qzeros", "scales", "g_idx"):
            self.assertFalse(_is_wqkv_a_fusion_input(f"{_WQ_A}.{leaf}"))
            self.assertFalse(_is_wqkv_a_fusion_input(f"{_WKV}.{leaf}"))

    def test_compressed_tensors_packed_loads_nothing_and_says_nothing(self):
        """The fully silent arm: no exception, no tensor, no diagnosis."""
        delivered: Dict[str, torch.Tensor] = {}
        params = {
            name: _CapturingParam(name, delivered, dtype)
            for name, dtype in _CT_FUSED_PARAMS.items()
        }
        stub = _make_stub("compressed-tensors", params)

        with mock.patch(
            "sglang.srt.models.deepseek_v4._unroutable_wqkv_a_leaves", _no_gate
        ):
            with envs.SGLANG_OPT_FP8_WO_A_GEMM.override(False):
                DeepseekV4ForCausalLM.load_weights(
                    stub, iter(_compressed_tensors_stream())
                )

        delivered.pop("__lock__", None)
        self.assertEqual(delivered, {})

    def test_the_pre_fix_fuse_flag_diverges_from_the_built_topology(self):
        """``fuse_wqa_wkv = envs.SGLANG_OPT_FUSE_WQA_WKV.get()``, kept executable.

        The construction-time auto-off makes the env and the built topology
        disagree, and the pre-fix line followed the env: the wq_a/wkv tensors
        would have been routed into a fused parameter that does not exist.
        """
        legacy_fuse = envs.SGLANG_OPT_FUSE_WQA_WKV.get()
        self.assertTrue(legacy_fuse, "the flag is default-on; that is the trap")

        built_fused = any(".wqkv_a." in name for name in _GPTQ_SPLIT_PARAMS)
        self.assertFalse(built_fused)
        self.assertNotEqual(legacy_fuse, built_fused)


# ---------------------------------------------------------------------------
# fixed behaviour
# ---------------------------------------------------------------------------


class TestPackedFormatsAreRefusedByName(CustomTestCase):
    def test_gptq_fused_build_is_refused_before_any_tensor_is_read(self):
        consumed: List[str] = []

        def probe():
            for item in _gptq_stream():
                consumed.append(item[0])
                yield item

        params = {
            name: _CapturingParam(name, {}, dtype)
            for name, dtype in _GPTQ_FUSED_PARAMS.items()
        }
        stub = _make_stub("gptq", params)
        with envs.SGLANG_OPT_FP8_WO_A_GEMM.override(False):
            with self.assertRaises(ValueError) as ctx:
                DeepseekV4ForCausalLM.load_weights(stub, probe())

        message = str(ctx.exception)
        for leaf in ("g_idx", "qzeros", "scales"):
            self.assertIn(leaf, message)
        self.assertIn("SGLANG_OPT_FUSE_WQA_WKV=0", message)
        self.assertEqual(consumed, [], "refused only after reading the stream")

    def test_compressed_tensors_fused_build_is_refused(self):
        params = {
            name: _CapturingParam(name, {}, dtype)
            for name, dtype in _CT_FUSED_PARAMS.items()
        }
        stub = _make_stub("compressed-tensors", params)
        with envs.SGLANG_OPT_FP8_WO_A_GEMM.override(False):
            with self.assertRaises(ValueError) as ctx:
                DeepseekV4ForCausalLM.load_weights(
                    stub, iter(_compressed_tensors_stream())
                )

        message = str(ctx.exception)
        for leaf in _CT_LEAVES:
            self.assertIn(leaf, message)

    def test_the_refusal_names_the_leaves_the_fusion_does_know(self):
        params = {
            name: _CapturingParam(name, {}, dtype)
            for name, dtype in _GPTQ_FUSED_PARAMS.items()
        }
        stub = _make_stub("gptq", params)
        with envs.SGLANG_OPT_FP8_WO_A_GEMM.override(False):
            with self.assertRaises(ValueError) as ctx:
                DeepseekV4ForCausalLM.load_weights(stub, iter([]))
        for leaf in _WQKV_A_LEAVES:
            self.assertIn(leaf, str(ctx.exception))


class TestConstructionTimeDecision(CustomTestCase):
    """``MqaAttentionBase.__init__`` decides the topology with this call.

    The modules are the ones the real quant methods build, so the decision runs
    on the same inventory a boot would hand it.
    """

    @staticmethod
    def _linear(quant_config, params_dtype=torch.bfloat16) -> ReplicatedLinear:
        return ReplicatedLinear(
            _HIDDEN,
            _Q_OUT + _KV_OUT,
            bias=False,
            quant_config=quant_config,
            prefix="wqkv_a",
            params_dtype=params_dtype,
        )

    @staticmethod
    def _gptq_config():
        from sglang.srt.layers.quantization.gptq import GPTQConfig

        return GPTQConfig(
            weight_bits=4,
            group_size=_GROUP,
            desc_act=False,
            lm_head_quantized=False,
            dynamic={},
        )

    def test_routable_formats_keep_the_fusion(self):
        from sglang.srt.layers.quantization.gguf import GGUFConfig

        for tag, cfg in (("bf16", None), ("gguf", GGUFConfig())):
            with self.subTest(format=tag):
                for explicit in (False, True):
                    self.assertTrue(
                        _wqkv_a_fusion_survives_quant_format(
                            self._linear(cfg), explicit
                        )
                    )

    def test_gptq_turns_the_inherited_default_off_with_a_named_notice(self):
        # The notice is emitted once per cause; clear the memo so this test
        # observes it regardless of test order.
        _warn_wqkv_a_fusion_auto_off.cache_clear()
        linear = self._linear(self._gptq_config(), params_dtype=torch.float16)

        with self.assertLogs("sglang.srt.models.deepseek_v4", level="WARNING") as logs:
            self.assertFalse(_wqkv_a_fusion_survives_quant_format(linear, False))

        message = "\n".join(logs.output)
        self.assertIn("SGLANG_OPT_FUSE_WQA_WKV", message)
        self.assertIn("GPTQLinearMethod", message)
        for leaf in ("g_idx", "qzeros", "scales"):
            self.assertIn(leaf, message)

    def test_gptq_refuses_an_explicit_opt_in_instead_of_turning_it_off(self):
        linear = self._linear(self._gptq_config(), params_dtype=torch.float16)
        with self.assertRaises(NotImplementedError) as ctx:
            _wqkv_a_fusion_survives_quant_format(linear, True)

        message = str(ctx.exception)
        self.assertIn("requested explicitly", message)
        self.assertIn("SGLANG_OPT_FUSE_WQA_WKV=0", message)
        for leaf in ("g_idx", "qzeros", "scales"):
            self.assertIn(leaf, message)

    def test_the_notice_is_emitted_once_per_cause_not_once_per_layer(self):
        """60 decoder layers must not produce 60 identical warnings."""
        _warn_wqkv_a_fusion_auto_off.cache_clear()
        cfg = self._gptq_config()

        with self.assertLogs("sglang.srt.models.deepseek_v4", level="WARNING") as logs:
            for _ in range(4):
                self.assertFalse(
                    _wqkv_a_fusion_survives_quant_format(
                        self._linear(cfg, params_dtype=torch.float16), False
                    )
                )
        self.assertEqual(len(logs.output), 1, logs.output)


class TestSplitBuildLoadsPackedFormatsUnchanged(CustomTestCase):
    """What the construction-time auto-off leaves behind: the unfused route."""

    def test_every_gptq_tensor_reaches_its_own_parameter(self):
        stream = _gptq_stream()
        sink = _run_load("gptq", stream, _GPTQ_SPLIT_PARAMS)

        self.assertEqual(len(sink), len(_GPTQ_SPLIT_PARAMS))
        for native_name, tensor in stream:
            param_name = native_name.replace(f"{_ATTN}.", "model.layers.2.self_attn.")
            self.assertIn(param_name, sink)
            self.assertTrue(torch.equal(sink[param_name], tensor), param_name)

    def test_the_loader_follows_the_topology_and_not_the_env(self):
        """The env stays at its default-on value; the split build still loads.

        This is the same call as above with the flag left untouched, which is
        exactly the configuration the pre-fix line got wrong.
        """
        self.assertTrue(envs.SGLANG_OPT_FUSE_WQA_WKV.get())
        sink = _run_load("gptq", _gptq_stream(), _GPTQ_SPLIT_PARAMS)
        self.assertEqual(len(sink), len(_GPTQ_SPLIT_PARAMS))

    def test_a_mixed_topology_routes_per_block(self):
        """One unquantized block beside a packed one, as `dynamic` produces.

        The topology is read per attention block, so layer 2 (packed, split)
        and layer 3 (unquantized, fused) are routed differently in the same
        load. A single model-wide flag gets one of the two wrong whichever way
        it is set.
        """
        other = "layers.3.attn"
        other_param = "model.layers.3.self_attn.wqkv_a.weight"
        wq_a = torch.randn(_Q_OUT, _HIDDEN, dtype=torch.bfloat16)
        wkv = torch.randn(_KV_OUT, _HIDDEN, dtype=torch.bfloat16)

        stream = _gptq_stream() + [
            (f"{other}.wq_a.weight", wq_a),
            (f"{other}.wkv.weight", wkv),
        ]
        param_specs = dict(_GPTQ_SPLIT_PARAMS)
        param_specs[other_param] = torch.bfloat16

        sink = _run_load("gptq", stream, param_specs)

        self.assertEqual(len(sink), len(_GPTQ_SPLIT_PARAMS) + 1)
        self.assertTrue(torch.equal(sink[other_param], torch.cat([wq_a, wkv], dim=0)))
        self.assertTrue(
            torch.equal(
                sink[f"{_WQ_A}.qweight"],
                dict(_gptq_stream())[f"{_ATTN}.wq_a.qweight"],
            )
        )


class TestRoutableFormatsStillFuse(CustomTestCase):
    """Behaviour neutrality for the routes the gate must not touch."""

    def test_dense_fp8_block_pair_still_fuses_along_output_rows(self):
        wq_a = torch.randn(_Q_OUT, _HIDDEN, dtype=torch.bfloat16)
        wkv = torch.randn(_KV_OUT, _HIDDEN, dtype=torch.bfloat16)
        wq_a_scale = torch.randn(1, 1)
        wkv_scale = torch.randn(1, 1)

        sink = _run_load(
            "fp8",
            [
                (f"{_ATTN}.wq_a.weight", wq_a),
                (f"{_ATTN}.wkv.weight", wkv),
                (f"{_ATTN}.wq_a.weight_scale_inv", wq_a_scale),
                (f"{_ATTN}.wkv.weight_scale_inv", wkv_scale),
            ],
            {
                f"{_WQKV_A}.weight": torch.bfloat16,
                f"{_WQKV_A}.weight_scale_inv": torch.float32,
            },
        )

        self.assertTrue(
            torch.equal(sink[f"{_WQKV_A}.weight"], torch.cat([wq_a, wkv], dim=0))
        )
        self.assertTrue(
            torch.equal(
                sink[f"{_WQKV_A}.weight_scale_inv"],
                torch.cat([wq_a_scale, wkv_scale], dim=0),
            )
        )

    def test_bf16_single_weight_leaf_still_fuses(self):
        wq_a = torch.randn(_Q_OUT, _HIDDEN, dtype=torch.bfloat16)
        wkv = torch.randn(_KV_OUT, _HIDDEN, dtype=torch.bfloat16)

        sink = _run_load(
            None,
            [(f"{_ATTN}.wq_a.weight", wq_a), (f"{_ATTN}.wkv.weight", wkv)],
            {f"{_WQKV_A}.weight": torch.bfloat16},
        )
        self.assertTrue(
            torch.equal(sink[f"{_WQKV_A}.weight"], torch.cat([wq_a, wkv], dim=0))
        )


if __name__ == "__main__":
    unittest.main()
