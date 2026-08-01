# SPDX-License-Identifier: Apache-2.0
"""Both DeepSeek V4 weight-fusion sites must classify on the name SEGMENT (#391).

``DeepseekV4ForCausalLM.load_weights`` joins two projection pairs before
handing them to a parameter:

* ``compressor.wkv`` + ``compressor.wgate`` -> ``compressor.wkv_gate``
* ``wq_a`` + ``wkv`` -> ``wqkv_a`` (under ``SGLANG_OPT_FUSE_WQA_WKV``)

Both sites used to select their inputs by a ``.weight`` SUFFIX while the
compressor branch was GUARDED by the ``.compressor.w`` substring. On a GGUF
checkpoint the two disagree: a quantized projection arrives as ``.qweight``
plus a 0-dim ``.qweight_type`` marker, so

* every compressor tensor entered the branch and answered False to both
  suffix tests, tripping ``assert is_kv != is_wgate`` on the first one
  (boot 7, ``model.layers.2.self_attn.compressor.wgate.qweight_type``);
* every ``wq_a``/``wkv`` tensor missed the fusion predicate entirely and fell
  through to a parameter lookup for a module the fused build does not have,
  which only warns -- the fused ``wqkv_a`` stayed unfilled.

The two targets need opposite treatment, because their module graphs differ:

* ``Compressor.wkv_gate`` is a ``ReplicatedLinear`` built with
  ``quant_config=None`` and a bfloat16 ``params_dtype``, and
  ``compute_kv_score`` feeds ``wkv_gate.weight`` straight into
  ``linear_bf16_fp32``. It is dense by construction and there is no unfused
  path, so a GGUF payload has to be UNPACKED here.
* ``wqkv_a`` is a ``ReplicatedLinear`` built with the model's
  ``quant_config``, so under GGUF it is packed and the shards are joined as
  BYTES -- which is well defined only if both carry the same ggml type.

No GPU and no checkpoint: the payloads are real gguf-py quantizations of small
random matrices, the model is a stub carrying only the attributes
``load_weights`` reads.
"""

import threading
import types
import unittest
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.models.deepseek_v4 import (
    DeepseekV4ForCausalLM,
    _is_wqkv_a_fusion_input,
    _split_compressor_weight_name,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


_HIDDEN = 64
_KV_ROWS = 8
_GATE_ROWS = 8
_Q_A_ROWS = 12

#: The tensor the boot-7 assert named, in its post-remap spelling.
_BOOT7_TENSOR = "model.layers.2.self_attn.compressor.wgate.qweight_type"


def _ggml(name: str):
    import gguf

    return gguf.GGMLQuantizationType[name]


def _quantize(rows: int, seed: int, qtype_name: str = "Q8_0"):
    """A real gguf-py payload plus the values it decodes to.

    Returns ``(packed_uint8_tensor, type_marker_tensor, dequantized_float32)``.
    """
    from gguf.quants import dequantize, quantize

    rng = np.random.default_rng(seed)
    source = rng.standard_normal((rows, _HIDDEN), dtype=np.float32)
    qtype = _ggml(qtype_name)
    packed = quantize(source, qtype)
    return (
        torch.from_numpy(packed.copy()),
        torch.tensor(int(qtype)),
        torch.from_numpy(dequantize(packed, qtype).copy()),
    )


class _FakeQuantConfig:
    """Only ``get_name()`` is read (``is_gguf_quant_config``)."""

    def __init__(self, name: str):
        self._name = name

    def get_name(self) -> str:
        return self._name


class _CapturingParam:
    """Stands in for a parameter: a dtype plus a recording ``weight_loader``."""

    def __init__(self, name: str, sink: Dict[str, torch.Tensor], dtype: torch.dtype):
        self._name = name
        self._sink = sink
        self._lock = sink.setdefault("__lock__", threading.Lock())
        self.dtype = dtype

    def weight_loader(self, param, loaded_weight, *args, **kwargs) -> None:
        with self._lock:
            self._sink[self._name] = loaded_weight


def _make_stub(quant_name: Optional[str], params: dict) -> types.SimpleNamespace:
    """A stub carrying exactly the attributes ``load_weights`` reads."""
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
    fuse_wqa_wkv: bool = True,
) -> Dict[str, torch.Tensor]:
    """Drive the real ``load_weights`` and return what reached each parameter."""
    sink: Dict[str, torch.Tensor] = {}
    params = {
        name: _CapturingParam(name, sink, dtype) for name, dtype in param_specs.items()
    }
    stub = _make_stub(quant_name, params)
    with envs.SGLANG_OPT_FP8_WO_A_GEMM.override(False):
        with envs.SGLANG_OPT_FUSE_WQA_WKV.override(fuse_wqa_wkv):
            DeepseekV4ForCausalLM.load_weights(stub, iter(stream))
    sink.pop("__lock__", None)
    return sink


# ---------------------------------------------------------------------------
# compressor -> wkv_gate
# ---------------------------------------------------------------------------

#: Native (pre-remap) prefix the GGUF adapter's name map produces.
_COMPRESSOR = "layers.2.attn.compressor"
_COMPRESSOR_PARAM = "model.layers.2.self_attn.compressor.wkv_gate.weight"


def _gguf_compressor_stream(kv, gate) -> List[Tuple[str, torch.Tensor]]:
    """The iterator's two passes: every type marker first, then the payloads."""
    kv_packed, kv_type, _ = kv
    gate_packed, gate_type, _ = gate
    return [
        (f"{_COMPRESSOR}.wkv.qweight_type", kv_type),
        (f"{_COMPRESSOR}.wgate.qweight_type", gate_type),
        (f"{_COMPRESSOR}.wkv.qweight", kv_packed),
        (f"{_COMPRESSOR}.wgate.qweight", gate_packed),
    ]


class TestCompressorFusionOnGguf(CustomTestCase):
    def test_gguf_pair_fuses_into_the_dense_wkv_gate(self):
        """(b) the fused tensor equals a hand-built reference, bit for bit."""
        kv = _quantize(_KV_ROWS, seed=1)
        gate = _quantize(_GATE_ROWS, seed=2)

        sink = _run_load(
            "gguf",
            _gguf_compressor_stream(kv, gate),
            {_COMPRESSOR_PARAM: torch.bfloat16},
        )

        self.assertEqual(list(sink), [_COMPRESSOR_PARAM])
        fused = sink[_COMPRESSOR_PARAM]
        reference = torch.cat([kv[2], gate[2]], dim=0).to(torch.bfloat16)
        self.assertEqual(fused.dtype, torch.bfloat16)
        self.assertEqual(tuple(fused.shape), (_KV_ROWS + _GATE_ROWS, _HIDDEN))
        self.assertTrue(torch.equal(fused, reference))

    def test_order_is_kv_then_wgate(self):
        """The fusion order is load-bearing and is the pre-fix order."""
        kv = _quantize(_KV_ROWS, seed=3)
        gate = _quantize(_GATE_ROWS, seed=4)
        fused = _run_load(
            "gguf",
            _gguf_compressor_stream(kv, gate),
            {_COMPRESSOR_PARAM: torch.bfloat16},
        )[_COMPRESSOR_PARAM]

        self.assertTrue(
            torch.equal(fused[:_KV_ROWS], kv[2].to(torch.bfloat16)),
        )
        self.assertTrue(
            torch.equal(fused[_KV_ROWS:], gate[2].to(torch.bfloat16)),
        )

    def test_an_incomplete_pair_never_reaches_the_parameter(self):
        """Both markers land before any payload; fusing then would be wrong.

        Truncating after the two markers and the first payload leaves the pair
        incomplete: nothing may be loaded, and the leftover cache must trip the
        end-of-load audit rather than pass silently.
        """
        kv = _quantize(_KV_ROWS, seed=5)
        gate = _quantize(_GATE_ROWS, seed=6)
        stream = _gguf_compressor_stream(kv, gate)

        sink: Dict[str, torch.Tensor] = {}
        params = {
            _COMPRESSOR_PARAM: _CapturingParam(_COMPRESSOR_PARAM, sink, torch.bfloat16)
        }
        stub = _make_stub("gguf", params)
        with envs.SGLANG_OPT_FP8_WO_A_GEMM.override(False):
            with self.assertRaises(AssertionError):
                DeepseekV4ForCausalLM.load_weights(stub, iter(stream[:3]))

        sink.pop("__lock__", None)
        self.assertEqual(sink, {})

    def test_mixed_ggml_types_still_fuse_correctly(self):
        """Unpacking per projection means the two need not share a type.

        The ``wqkv_a`` site cannot do this -- its target keeps ONE marker --
        which is why only that site carries a type-equality precondition.
        """
        kv = _quantize(_KV_ROWS, seed=7, qtype_name="Q8_0")
        gate = _quantize(_GATE_ROWS, seed=8, qtype_name="Q5_0")

        fused = _run_load(
            "gguf",
            _gguf_compressor_stream(kv, gate),
            {_COMPRESSOR_PARAM: torch.bfloat16},
        )[_COMPRESSOR_PARAM]

        reference = torch.cat([kv[2], gate[2]], dim=0).to(torch.bfloat16)
        self.assertTrue(torch.equal(fused, reference))

    def test_can_fail_arm_the_suffix_classifier_rejects_these_names(self):
        """(a) the pre-fix classifier, kept executable, on the boot-7 tensor.

        This is the exact pair of predicates that stood at deepseek_v4.py:2944
        and the exact assert that fired. If it ever stops raising, the segment
        classifier below has stopped being a fix.
        """

        def legacy_classify(name: str) -> bool:
            is_kv = name.endswith(".wkv.weight")
            is_wgate = name.endswith(".wgate.weight")
            assert is_kv != is_wgate
            return is_kv

        for name in (
            _BOOT7_TENSOR,
            "model.layers.2.self_attn.compressor.wgate.qweight",
            "model.layers.2.self_attn.compressor.wkv.qweight",
            "model.layers.2.self_attn.compressor.wkv.qweight_type",
        ):
            with self.subTest(name=name):
                with self.assertRaises(AssertionError):
                    legacy_classify(name)
                self.assertIsNotNone(
                    _split_compressor_weight_name(name),
                    "the segment classifier must recognize what the suffix "
                    "classifier could not",
                )

        # ... and the dense spelling both agree on, so the fix is a widening.
        dense = "model.layers.2.self_attn.compressor.wkv.weight"
        self.assertTrue(legacy_classify(dense))
        self.assertEqual(
            _split_compressor_weight_name(dense),
            ("model.layers.2.self_attn.compressor", "wkv", "weight"),
        )

    def test_unknown_compressor_leaf_is_refused_by_name(self):
        """A compressor tensor the fusion does not know still fails loudly."""
        with self.assertRaises(AssertionError) as ctx:
            _run_load(
                "gguf",
                [(f"{_COMPRESSOR}.wkv.weight_scale_inv", torch.zeros(2))],
                {_COMPRESSOR_PARAM: torch.bfloat16},
            )
        self.assertIn("compressor.wkv.weight_scale_inv", str(ctx.exception))


class TestCompressorDenseRouteUnchanged(CustomTestCase):
    """(d) the safetensors route must be untouched, pinned on an fp8 stream."""

    def test_dense_pair_fuses_exactly_as_before(self):
        kv = torch.randn(_KV_ROWS, _HIDDEN, dtype=torch.bfloat16)
        gate = torch.randn(_GATE_ROWS, _HIDDEN, dtype=torch.bfloat16)

        sink = _run_load(
            "fp8",
            [
                (f"{_COMPRESSOR}.wkv.weight", kv),
                (f"{_COMPRESSOR}.wgate.weight", gate),
            ],
            {_COMPRESSOR_PARAM: torch.bfloat16},
        )

        self.assertTrue(
            torch.equal(sink[_COMPRESSOR_PARAM], torch.cat([kv, gate], dim=0))
        )

    def test_dense_pair_fuses_in_either_arrival_order(self):
        kv = torch.randn(_KV_ROWS, _HIDDEN, dtype=torch.bfloat16)
        gate = torch.randn(_GATE_ROWS, _HIDDEN, dtype=torch.bfloat16)
        sink = _run_load(
            "fp8",
            [
                (f"{_COMPRESSOR}.wgate.weight", gate),
                (f"{_COMPRESSOR}.wkv.weight", kv),
            ],
            {_COMPRESSOR_PARAM: torch.bfloat16},
        )
        self.assertTrue(
            torch.equal(sink[_COMPRESSOR_PARAM], torch.cat([kv, gate], dim=0))
        )


# ---------------------------------------------------------------------------
# wq_a + wkv -> wqkv_a
# ---------------------------------------------------------------------------

_ATTN = "layers.2.attn"
_WQKV_A = "model.layers.2.self_attn.wqkv_a"


def _gguf_wqkv_a_stream(q, kv) -> List[Tuple[str, torch.Tensor]]:
    q_packed, q_type, _ = q
    kv_packed, kv_type, _ = kv
    return [
        (f"{_ATTN}.wq_a.qweight_type", q_type),
        (f"{_ATTN}.wkv.qweight_type", kv_type),
        (f"{_ATTN}.wq_a.qweight", q_packed),
        (f"{_ATTN}.wkv.qweight", kv_packed),
    ]


class TestWqkvAFusionOnGguf(CustomTestCase):
    def test_packed_shards_join_as_bytes_and_share_one_marker(self):
        """(b) the packed join decodes to the same values as unpacking apart.

        Q8_0 blocks run along the INPUT dimension, so concatenating output rows
        keeps every block intact. That is the whole precondition for joining
        bytes instead of values, and it is asserted here rather than asserted
        in a comment.
        """
        from gguf.quants import dequantize

        q = _quantize(_Q_A_ROWS, seed=11)
        kv = _quantize(_KV_ROWS, seed=12)

        sink = _run_load(
            "gguf",
            _gguf_wqkv_a_stream(q, kv),
            {
                f"{_WQKV_A}.qweight": torch.uint8,
                f"{_WQKV_A}.qweight_type": torch.uint8,
            },
        )

        self.assertEqual(
            sorted(sink), [f"{_WQKV_A}.qweight", f"{_WQKV_A}.qweight_type"]
        )
        fused_bytes = sink[f"{_WQKV_A}.qweight"]
        self.assertTrue(torch.equal(fused_bytes, torch.cat([q[0], kv[0]], dim=0)))
        self.assertEqual(
            int(sink[f"{_WQKV_A}.qweight_type"].item()), int(_ggml("Q8_0"))
        )

        decoded = dequantize(fused_bytes.numpy(), _ggml("Q8_0"))
        self.assertTrue(
            torch.equal(
                torch.from_numpy(decoded.copy()), torch.cat([q[2], kv[2]], dim=0)
            )
        )

    def test_mismatched_qweight_type_pair_is_refused(self):
        """(c) two ggml types cannot share the single fused marker."""
        q = _quantize(_Q_A_ROWS, seed=13, qtype_name="Q8_0")
        kv = _quantize(_KV_ROWS, seed=14, qtype_name="Q5_0")

        with self.assertRaises(ValueError) as ctx:
            _run_load(
                "gguf",
                _gguf_wqkv_a_stream(q, kv),
                {
                    f"{_WQKV_A}.qweight": torch.uint8,
                    f"{_WQKV_A}.qweight_type": torch.uint8,
                },
            )

        message = str(ctx.exception)
        self.assertIn(f"{_WQKV_A}.qweight_type", message)
        self.assertIn("Q8_0", message)
        self.assertIn("Q5_0", message)
        self.assertIn("SGLANG_OPT_FUSE_WQA_WKV=0", message)

    def test_refusal_fires_on_the_markers_before_any_payload_is_read(self):
        """Both markers arrive in the iterator's first pass; refuse there."""
        q = _quantize(_Q_A_ROWS, seed=15, qtype_name="Q8_0")
        kv = _quantize(_KV_ROWS, seed=16, qtype_name="Q5_0")
        consumed: List[str] = []

        def probe():
            for item in _gguf_wqkv_a_stream(q, kv):
                consumed.append(item[0])
                yield item

        params = {
            name: _CapturingParam(name, {}, torch.uint8)
            for name in (f"{_WQKV_A}.qweight", f"{_WQKV_A}.qweight_type")
        }
        stub = _make_stub("gguf", params)
        with envs.SGLANG_OPT_FP8_WO_A_GEMM.override(False):
            with self.assertRaises(ValueError):
                DeepseekV4ForCausalLM.load_weights(stub, probe())

        self.assertEqual(len(consumed), 2, f"read past the markers: {consumed}")

    def test_can_fail_arm_the_suffix_predicate_misses_gguf_names(self):
        """(a) the pre-fix predicate, kept executable."""

        def legacy_predicate(name: str) -> bool:
            return (
                name.endswith(".wq_a.weight")
                or name.endswith(".wq_a.weight_scale_inv")
                or name.endswith(".wkv.weight")
                or name.endswith(".wkv.weight_scale_inv")
            )

        for name in (
            "model.layers.2.self_attn.wq_a.qweight",
            "model.layers.2.self_attn.wq_a.qweight_type",
            "model.layers.2.self_attn.wkv.qweight",
            "model.layers.2.self_attn.wkv.qweight_type",
        ):
            with self.subTest(name=name):
                self.assertFalse(legacy_predicate(name))
                self.assertTrue(_is_wqkv_a_fusion_input(name))

        for name in (
            "model.layers.2.self_attn.wq_a.weight",
            "model.layers.2.self_attn.wq_a.weight_scale_inv",
            "model.layers.2.self_attn.wkv.weight",
            "model.layers.2.self_attn.wkv.weight_scale_inv",
        ):
            with self.subTest(name=name):
                self.assertTrue(legacy_predicate(name))
                self.assertTrue(_is_wqkv_a_fusion_input(name))

        # Neighbours that must stay out of the fusion.
        for name in (
            "model.layers.2.self_attn.wq_b.qweight",
            "model.layers.2.self_attn.wo_a.qweight",
            "model.layers.2.self_attn.indexer.wq_b.qweight",
        ):
            with self.subTest(name=name):
                self.assertFalse(_is_wqkv_a_fusion_input(name))


class TestWqkvADenseRouteUnchanged(CustomTestCase):
    """(d) the safetensors route, pinned on an fp8-marked stream."""

    def test_dense_and_block_scale_pairs_fuse_exactly_as_before(self):
        wq_a = torch.randn(_Q_A_ROWS, _HIDDEN, dtype=torch.bfloat16)
        wkv = torch.randn(_KV_ROWS, _HIDDEN, dtype=torch.bfloat16)
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

    def test_fusion_off_loads_the_two_projections_separately(self):
        """The escape hatch the mismatch error names must actually exist."""
        q = _quantize(_Q_A_ROWS, seed=17)
        kv = _quantize(_KV_ROWS, seed=18)

        sink = _run_load(
            "gguf",
            _gguf_wqkv_a_stream(q, kv),
            {
                "model.layers.2.self_attn.wq_a.qweight": torch.uint8,
                "model.layers.2.self_attn.wq_a.qweight_type": torch.uint8,
                "model.layers.2.self_attn.wkv.qweight": torch.uint8,
                "model.layers.2.self_attn.wkv.qweight_type": torch.uint8,
            },
            fuse_wqa_wkv=False,
        )

        self.assertEqual(len(sink), 4)
        self.assertTrue(
            torch.equal(sink["model.layers.2.self_attn.wq_a.qweight"], q[0])
        )
        self.assertTrue(
            torch.equal(sink["model.layers.2.self_attn.wkv.qweight"], kv[0])
        )


if __name__ == "__main__":
    unittest.main()
