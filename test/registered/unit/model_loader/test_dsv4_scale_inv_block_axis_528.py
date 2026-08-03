# SPDX-License-Identifier: Apache-2.0
"""The DeepSeek V4 ``wq_a`` + ``wkv`` fusion must not cut a scale BLOCK in half.

``_fuse_wqkv_a`` joins every leaf it knows with ``torch.cat(..., dim=0)``. For
``weight`` / ``qweight`` that is the output-row axis, which is what the join
means. For ``weight_scale_inv`` of a block-quantized checkpoint it is NOT a row
axis: one scale row governs ``weight_block_size[0]`` weight rows. The two agree
only while ``wq_a``'s width is a whole number of blocks (task #528, found while
reviewing the #526 gate, which deliberately left this byte-identical).

When it is not, fused scale block ``q_lora_rank // b`` spans wq_a's tail AND
wkv's first rows under wq_a's scale, and every later block is shifted -- with
no shape error to show for it whenever the row counts still add up, which for
the DSV4 shape (``kv_rows = head_dim``, a multiple of 128) they always do.

REACHABILITY, the reason this ships as a guard rather than as a repair: every
DSV4 geometry that exists today is aligned -- ``q_lora_rank`` 1024 against the
fp8 ``weight_block_size`` [128, 128] of the DSpark export, and 1 against the
MXFP8 [1, 32] OCP layout (the #444b eighth alignment sibling, which crosses this
fusion and is provably immune because its dim-0 block IS a single row). The
tests below pin that assumption at the code, so a future geometry that breaks
it turns red here instead of loading silently wrong numbers.

The fix direction is REFUSAL, not a block-aware concatenation: fused block
``q_lora_rank // b`` is fed by two tensors that were quantized independently,
so no single scale row describes it -- repairing the join would mean
dequantizing and requantizing the seam, i.e. changing the checkpoint's numbers.

No GPU and no checkpoint: the fused linears are built by the REAL
``Fp8LinearMethod`` under ``CUDA_VISIBLE_DEVICES=99`` (the TP world size is the
only thing it needs, and ``get_parallel().override`` supplies it), and the
weight streams are synthetic tensors of the shapes that method declares.
"""

import dataclasses
import json
import os
import threading
import types
import unittest
from typing import Dict, List, Optional
from unittest import mock

import torch

from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.environ import envs
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.models.deepseek_v4 import (
    DeepseekV4ForCausalLM,
    _misaligned_scale_block_axis,
    _warn_wqkv_a_fusion_auto_off,
    _wqkv_a_scale_block_height,
    _wqkv_a_scale_block_misalignment,
    _wqkv_a_fusion_survives_quant_format,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

_HIDDEN = 256  # a whole number of block_k columns; irrelevant to this axis
_BLOCK = [128, 128]  # what the DSpark DSV4 export declares
_KV_ROWS = 512  # config.head_dim, the wkv width
_ALIGNED_Q = 1024  # config.q_lora_rank, 8 whole scale blocks
_MISALIGNED_Q = 1000  # 7 whole blocks + 104 rows

_ATTN = "layers.2.attn"
_WQKV_A = "model.layers.2.self_attn.wqkv_a"

#: The published DSpark export, if this rig has it. Optional on purpose: the
#: assumption is pinned against the config CLASS below, which travels with the
#: source; this arm only confirms the class defaults describe the real file.
_DSPARK_CONFIG = (
    "/spinning/llm_stuff/club-3090/models-cache/"
    "DeepSeek-V4-Flash-0731-dspark-head/config.json"
)


def _fp8_block_config() -> Fp8Config:
    return Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=list(_BLOCK),
    )


def _mxfp8_config() -> Fp8Config:
    """MXFP8 pins ``weight_block_size`` to the OCP [1, 32] (fp8.py:297-300)."""
    return Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        use_mxfp8=True,
    )


def _fused_linear(quant_config, out_rows: int, **kwargs) -> ReplicatedLinear:
    """A fused ``wqkv_a`` built by the real quant method, off GPU.

    ``Fp8LinearMethod.create_weights`` reads the TP world size before it decides
    anything; overriding it is the whole distributed setup this needs.
    """
    with get_parallel().override(tp_size=1):
        return ReplicatedLinear(
            _HIDDEN,
            out_rows,
            bias=False,
            quant_config=quant_config,
            prefix="wqkv_a",
            **kwargs,
        )


# ---------------------------------------------------------------------------
# harness -- the stub `load_weights` reads (shape shared with
# test_dsv4_gguf_weight_fusions / test_dsv4_wqkv_a_packed_formats_526)
# ---------------------------------------------------------------------------


class _FakeQuantConfig:
    """``get_name()`` and, for the load-time backstop, ``weight_block_size``."""

    def __init__(self, name: str, weight_block_size: Optional[List[int]] = None):
        self._name = name
        if weight_block_size is not None:
            self.weight_block_size = weight_block_size

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


def _make_stub(
    quant_config, params: dict, q_lora_rank: Optional[int]
) -> types.SimpleNamespace:
    config = types.SimpleNamespace(num_hidden_layers=8, n_routed_experts=2)
    if q_lora_rank is not None:
        config.q_lora_rank = q_lora_rank
    stub = types.SimpleNamespace()
    stub.config = config
    stub.quant_config = quant_config
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


def _fused_param_specs() -> Dict[str, torch.dtype]:
    return {
        f"{_WQKV_A}.weight": torch.float32,
        f"{_WQKV_A}.weight_scale_inv": torch.float32,
    }


def _fp8_block_stream(q_rows: int, seed: int = 7):
    """One layer's wq_a/wkv pair plus their block scales, in checkpoint order."""
    g = torch.Generator().manual_seed(seed)
    scale_cols = -(-_HIDDEN // _BLOCK[1])
    tensors = {
        "wq_a.weight": torch.randn(q_rows, _HIDDEN, generator=g),
        "wkv.weight": torch.randn(_KV_ROWS, _HIDDEN, generator=g),
        # Deliberately far apart per row, so a swapped block cannot pass a
        # numeric comparison by accident (spread precondition).
        "wq_a.weight_scale_inv": 1.0
        + torch.arange(
            -(-q_rows // _BLOCK[0]) * scale_cols, dtype=torch.float32
        ).reshape(-1, scale_cols),
        "wkv.weight_scale_inv": 1000.0
        + torch.arange(
            -(-_KV_ROWS // _BLOCK[0]) * scale_cols, dtype=torch.float32
        ).reshape(-1, scale_cols),
    }
    stream = [(f"{_ATTN}.{leaf}", tensor) for leaf, tensor in tensors.items()]
    return stream, tensors


def _mxfp8_stream(q_rows: int, seed: int = 11):
    """The same pair under the OCP [1, 32] block: one scale row per weight row."""
    g = torch.Generator().manual_seed(seed)
    cols = -(-_HIDDEN // 32)
    tensors = {
        "wq_a.weight": torch.randn(q_rows, _HIDDEN, generator=g),
        "wkv.weight": torch.randn(_KV_ROWS, _HIDDEN, generator=g),
        "wq_a.weight_scale_inv": 1.0
        + torch.arange(q_rows * cols, dtype=torch.float32).reshape(q_rows, cols),
        "wkv.weight_scale_inv": 1000.0
        + torch.arange(_KV_ROWS * cols, dtype=torch.float32).reshape(_KV_ROWS, cols),
    }
    return [(f"{_ATTN}.{leaf}", tensor) for leaf, tensor in tensors.items()], tensors


def _run_load(stub, stream) -> None:
    with envs.SGLANG_OPT_FP8_WO_A_GEMM.override(False):
        DeepseekV4ForCausalLM.load_weights(stub, iter(stream))


def _load_fused(q_rows: int, quant_config, stream) -> Dict[str, torch.Tensor]:
    sink: Dict[str, torch.Tensor] = {}
    params = {
        name: _CapturingParam(name, sink, dtype)
        for name, dtype in _fused_param_specs().items()
    }
    _run_load(_make_stub(quant_config, params, q_rows), stream)
    sink.pop("__lock__", None)
    return sink


def _block_dequantize(
    weight: torch.Tensor, scale: torch.Tensor, block=_BLOCK
) -> torch.Tensor:
    """The dense tensor a block-scaled pair stands for, at full precision."""
    rows, cols = weight.shape
    expanded = scale.repeat_interleave(block[0], dim=0)[:rows]
    expanded = expanded.repeat_interleave(block[1], dim=1)[:, :cols]
    return weight * expanded


# ---------------------------------------------------------------------------
# 1. reachability: every geometry that exists today is aligned
# ---------------------------------------------------------------------------


class TestShippedGeometryIsAligned(CustomTestCase):
    """The pin. This class is what fails when a new checkpoint breaks it."""

    @staticmethod
    def _default(field: str) -> int:
        return next(
            f.default for f in dataclasses.fields(DeepSeekV4Config) if f.name == field
        )

    def test_config_class_defaults_end_on_a_block_boundary(self):
        """Read off the config, not written down here."""
        q_lora_rank = self._default("q_lora_rank")
        head_dim = self._default("qk_nope_head_dim") + self._default("qk_rope_head_dim")
        self.assertEqual((q_lora_rank, head_dim), (1024, 512))
        self.assertEqual(
            q_lora_rank % _BLOCK[0],
            0,
            "q_lora_rank no longer ends on a scale block: the wq_a/wkv fusion "
            "now refuses instead of fusing -- see #528. Re-price the fusion for "
            "this geometry rather than relaxing this assertion.",
        )
        # kv side, for the record: it never decides the verdict (the cut sits
        # at q_lora_rank), but a misaligned wkv is what hides the defect,
        # because then the two scale row counts stop adding up and the copy
        # raises a shape error instead of loading wrong numbers.
        self.assertEqual(head_dim % _BLOCK[0], 0)

    def test_the_published_dspark_export_agrees_with_the_class(self):
        if not os.path.exists(_DSPARK_CONFIG):
            self.skipTest("DSpark DSV4 export not present on this host")
        with open(_DSPARK_CONFIG) as handle:
            config = json.load(handle)
        block = config["quantization_config"]["weight_block_size"]
        self.assertEqual(block, _BLOCK)
        self.assertEqual(config["q_lora_rank"] % block[0], 0)
        self.assertEqual(config["head_dim"] % block[0], 0)

    def test_the_shipped_geometry_is_fused_exactly_as_before(self):
        """Behaviour neutrality: the guard does not fire at 1024 rows."""
        linear = _fused_linear(_fp8_block_config(), _ALIGNED_Q + _KV_ROWS)
        self.assertIsNone(_wqkv_a_scale_block_misalignment(linear, _ALIGNED_Q))
        self.assertTrue(
            _wqkv_a_fusion_survives_quant_format(linear, False, q_rows=_ALIGNED_Q)
        )
        self.assertTrue(
            _wqkv_a_fusion_survives_quant_format(linear, True, q_rows=_ALIGNED_Q)
        )


# ---------------------------------------------------------------------------
# 2. the instrument: what the predicate reads, on real built modules
# ---------------------------------------------------------------------------


class TestBlockHeightIsReadOffTheBuiltModule(CustomTestCase):
    def test_fp8_block_declares_a_128_row_scale_axis(self):
        linear = _fused_linear(_fp8_block_config(), _ALIGNED_Q + _KV_ROWS)
        self.assertEqual(_wqkv_a_scale_block_height(linear), _BLOCK[0])
        # ... and that height is a fact about the parameter, not a label:
        self.assertEqual(
            tuple(linear.weight_scale_inv.shape),
            ((_ALIGNED_Q + _KV_ROWS) // _BLOCK[0], _HIDDEN // _BLOCK[1]),
        )

    def test_mxfp8_block_axis_is_a_single_row_so_it_cannot_be_cut(self):
        """#444b, the eighth alignment sibling, crossed with this fusion.

        MXFP8 exposes ``weight_block_size = [1, 32]``: dim 0 is an ELEMENT
        axis, so ``torch.cat`` on the scale is the same axis as ``torch.cat``
        on the weight for ANY split -- including a misaligned one. The leaf
        names are ``weight`` + ``weight_scale_inv``, i.e. the #526 gate lets
        MXFP8 fuse; this is what makes the crossing live rather than academic.
        """
        config = _mxfp8_config()
        self.assertEqual(config.weight_block_size, [1, 32])
        linear = _fused_linear(config, _MISALIGNED_Q + _KV_ROWS)
        self.assertEqual(
            sorted(name for name, _ in linear.named_parameters(recurse=False)),
            ["weight", "weight_scale_inv"],
        )
        self.assertEqual(_wqkv_a_scale_block_height(linear), 1)
        self.assertIsNone(_wqkv_a_scale_block_misalignment(linear, _MISALIGNED_Q))
        self.assertEqual(linear.weight_scale_inv.shape[0], _MISALIGNED_Q + _KV_ROWS)

    def test_a_module_without_a_block_scale_has_no_block_axis(self):
        linear = _fused_linear(None, _MISALIGNED_Q + _KV_ROWS)
        self.assertEqual(
            [name for name, _ in linear.named_parameters(recurse=False)], ["weight"]
        )
        self.assertIsNone(_wqkv_a_scale_block_height(linear))
        self.assertIsNone(_wqkv_a_scale_block_misalignment(linear, _MISALIGNED_Q))

    def test_the_predicate_discriminates(self):
        """Spread precondition: it must reject something and accept something."""
        self.assertIsNone(_misaligned_scale_block_axis(128, 1024))
        self.assertIsNone(_misaligned_scale_block_axis(1, 1000))
        self.assertIsNone(_misaligned_scale_block_axis(None, 1000))
        self.assertIsNone(_misaligned_scale_block_axis(128, None))
        self.assertEqual(_misaligned_scale_block_axis(128, 1000), (128, 104))
        self.assertEqual(_misaligned_scale_block_axis(64, 1000), (64, 40))

    def test_the_height_falls_back_to_the_shape_ratio(self):
        """A block-scaled format that declares no ``weight_block_size``."""
        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(torch.empty(1536, _HIDDEN))
        module.weight_scale_inv = torch.nn.Parameter(torch.empty(12, 2))
        self.assertEqual(_wqkv_a_scale_block_height(module), 128)
        self.assertIsNone(_wqkv_a_scale_block_misalignment(module, 1024))
        self.assertEqual(_wqkv_a_scale_block_misalignment(module, 1000), (128, 104))


# ---------------------------------------------------------------------------
# 3. can-fail arm: the pre-fix join, driven for real, is silently wrong
# ---------------------------------------------------------------------------


def _no_guard(block_height, q_rows):
    """The pre-fix state: no block-axis check in either consumer."""
    return None


class TestPreFixJoinIsSilentlyWrong(CustomTestCase):
    """Both guards removed, i.e. the code exactly as it stood before #528."""

    def test_the_row_counts_still_add_up_so_nothing_raises(self):
        """Why this could never be caught by a shape error on DSV4.

        ``ceil(a/b) + ceil(c/b) == ceil((a+c)/b)`` whenever ``c % b == 0``, and
        ``c`` here is ``head_dim``. The concatenated scale therefore has exactly
        the row count the fused parameter declares, misaligned or not.
        """
        linear = _fused_linear(_fp8_block_config(), _MISALIGNED_Q + _KV_ROWS)
        self.assertEqual(
            linear.weight_scale_inv.shape[0],
            -(-_MISALIGNED_Q // _BLOCK[0]) + _KV_ROWS // _BLOCK[0],
        )

    def test_the_fused_scales_describe_the_wrong_weight_rows(self):
        stream, tensors = _fp8_block_stream(_MISALIGNED_Q)

        with mock.patch(
            "sglang.srt.models.deepseek_v4._misaligned_scale_block_axis", _no_guard
        ):
            sink = _load_fused(
                _MISALIGNED_Q,
                _FakeQuantConfig("fp8", weight_block_size=list(_BLOCK)),
                stream,
            )

        # No exception, no missing tensor: the load "succeeded".
        self.assertEqual(sorted(sink), sorted(_fused_param_specs()))
        fused_weight = sink[f"{_WQKV_A}.weight"]
        fused_scale = sink[f"{_WQKV_A}.weight_scale_inv"]
        self.assertTrue(
            torch.equal(
                fused_scale,
                torch.cat(
                    [tensors["wq_a.weight_scale_inv"], tensors["wkv.weight_scale_inv"]],
                    dim=0,
                ),
            ),
            "pre-fix behaviour is the naive concatenation",
        )

        # The reference: what the two projections mean unfused.
        reference = torch.cat(
            [
                _block_dequantize(
                    tensors["wq_a.weight"], tensors["wq_a.weight_scale_inv"]
                ),
                _block_dequantize(
                    tensors["wkv.weight"], tensors["wkv.weight_scale_inv"]
                ),
            ],
            dim=0,
        )
        fused = _block_dequantize(fused_weight, fused_scale)

        self.assertFalse(torch.allclose(reference, fused))
        wrong_rows = (reference != fused).any(dim=1).nonzero().flatten()

        # Which rows must be wrong, derived rather than guessed: wkv row `j`
        # sits in fused row `q + j`, hence in fused scale block
        # `(q + j) // b`, and the concatenation makes that block wkv's
        # `(q + j) // b - ceil(q / b)` -- which is wkv's own `j // b` only
        # while `q % b == 0`. Every wq_a row keeps its own scale, so the damage
        # is entirely on the wkv side: the `b - (q % b)` rows at the head of
        # each shifted block, four blocks' worth here.
        blocks_q = -(-_MISALIGNED_Q // _BLOCK[0])
        expected_wrong = [
            _MISALIGNED_Q + j
            for j in range(_KV_ROWS)
            if (_MISALIGNED_Q + j) // _BLOCK[0] - blocks_q != j // _BLOCK[0]
        ]
        self.assertEqual(wrong_rows.tolist(), expected_wrong)
        self.assertEqual(
            len(expected_wrong), 4 * (_BLOCK[0] - _MISALIGNED_Q % _BLOCK[0])
        )
        self.assertEqual(int(wrong_rows[0]), _MISALIGNED_Q)

    def test_the_aligned_geometry_is_unaffected_by_the_pre_fix_code(self):
        """The other half of the can-discriminate proof."""
        stream, tensors = _fp8_block_stream(_ALIGNED_Q)

        with mock.patch(
            "sglang.srt.models.deepseek_v4._misaligned_scale_block_axis", _no_guard
        ):
            sink = _load_fused(
                _ALIGNED_Q,
                _FakeQuantConfig("fp8", weight_block_size=list(_BLOCK)),
                stream,
            )

        reference = torch.cat(
            [
                _block_dequantize(
                    tensors["wq_a.weight"], tensors["wq_a.weight_scale_inv"]
                ),
                _block_dequantize(
                    tensors["wkv.weight"], tensors["wkv.weight_scale_inv"]
                ),
            ],
            dim=0,
        )
        fused = _block_dequantize(
            sink[f"{_WQKV_A}.weight"], sink[f"{_WQKV_A}.weight_scale_inv"]
        )
        self.assertTrue(torch.equal(reference, fused))


# ---------------------------------------------------------------------------
# 4. the fix: refusal, at construction and at load
# ---------------------------------------------------------------------------


class TestConstructionTimeRefusal(CustomTestCase):
    def test_an_inherited_default_on_fusion_is_turned_off_by_name(self):
        _warn_wqkv_a_fusion_auto_off.cache_clear()
        linear = _fused_linear(_fp8_block_config(), _MISALIGNED_Q + _KV_ROWS)

        with self.assertLogs("sglang.srt.models.deepseek_v4", level="WARNING") as logs:
            self.assertFalse(
                _wqkv_a_fusion_survives_quant_format(
                    linear, False, q_rows=_MISALIGNED_Q
                )
            )

        message = "\n".join(logs.output)
        self.assertIn("SGLANG_OPT_FUSE_WQA_WKV", message)
        self.assertIn("128-row BLOCK axis", message)
        self.assertIn(str(_MISALIGNED_Q), message)
        self.assertIn("104", message)

    def test_an_explicit_opt_in_is_refused_instead(self):
        linear = _fused_linear(_fp8_block_config(), _MISALIGNED_Q + _KV_ROWS)
        with self.assertRaises(NotImplementedError) as ctx:
            _wqkv_a_fusion_survives_quant_format(linear, True, q_rows=_MISALIGNED_Q)

        message = str(ctx.exception)
        self.assertIn("requested explicitly", message)
        self.assertIn("SGLANG_OPT_FUSE_WQA_WKV=0", message)
        self.assertIn("BLOCK axis", message)

    def test_a_caller_that_does_not_know_the_cut_keeps_the_old_behaviour(self):
        """``q_rows`` is optional, so the #526 call shape stays valid."""
        linear = _fused_linear(_fp8_block_config(), _MISALIGNED_Q + _KV_ROWS)
        self.assertTrue(_wqkv_a_fusion_survives_quant_format(linear, False))

    def test_the_attention_block_passes_its_own_q_lora_rank(self):
        """Binds-proof for the parameter: the gate is called WITH the cut.

        Constructing an ``MqaAttentionBase`` needs a full config plus rope
        tables; the wiring is what matters here, so it is read off the source of
        the one call site instead.
        """
        import inspect

        from sglang.srt.models.deepseek_v4 import MqaAttentionBase

        source = inspect.getsource(MqaAttentionBase.__init__)
        self.assertIn("_wqkv_a_fusion_survives_quant_format(", source)
        self.assertIn("q_rows=self.q_lora_rank", source)


class TestLoadTimeBackstop(CustomTestCase):
    """A fused build the construction-time gate did not stop must not load."""

    def test_a_misaligned_fused_build_is_refused_before_the_stream_is_read(self):
        params = {
            name: _CapturingParam(name, {}, dtype)
            for name, dtype in _fused_param_specs().items()
        }
        stub = _make_stub(
            _FakeQuantConfig("fp8", weight_block_size=list(_BLOCK)),
            params,
            _MISALIGNED_Q,
        )

        def explode():
            raise AssertionError("the stream was read before the refusal")
            yield  # pragma: no cover

        with self.assertRaises(ValueError) as ctx:
            _run_load(stub, explode())

        message = str(ctx.exception)
        self.assertIn("cannot load the fused wqkv_a", message)
        self.assertIn("128-row BLOCK axis", message)
        self.assertIn("SGLANG_OPT_FUSE_WQA_WKV=0", message)

    def test_an_unfused_build_is_never_refused(self):
        """Nothing to misalign when the two projections stayed split."""
        split = {
            f"model.layers.2.self_attn.{projection}.{leaf}": torch.float32
            for projection in ("wq_a", "wkv")
            for leaf in ("weight", "weight_scale_inv")
        }
        sink: Dict[str, torch.Tensor] = {}
        params = {
            name: _CapturingParam(name, sink, dtype) for name, dtype in split.items()
        }
        stub = _make_stub(
            _FakeQuantConfig("fp8", weight_block_size=list(_BLOCK)),
            params,
            _MISALIGNED_Q,
        )
        stream, tensors = _fp8_block_stream(_MISALIGNED_Q)
        _run_load(stub, stream)
        sink.pop("__lock__", None)

        self.assertEqual(sorted(sink), sorted(split))
        for leaf, tensor in tensors.items():
            self.assertTrue(
                torch.equal(sink[f"model.layers.2.self_attn.{leaf}"], tensor)
            )

    def test_a_checkpoint_without_a_block_size_is_untouched(self):
        """bf16 / GGUF / per-tensor fp8: no block axis, no verdict, no cost."""
        stream, tensors = _fp8_block_stream(_MISALIGNED_Q)
        sink = _load_fused(_MISALIGNED_Q, _FakeQuantConfig("fp8"), stream)
        self.assertEqual(sorted(sink), sorted(_fused_param_specs()))


class TestAlignedRouteStaysByteIdentical(CustomTestCase):
    """The shipped geometry, end to end, against the pre-#528 expectation."""

    def test_weight_and_scale_are_the_plain_concatenation(self):
        stream, tensors = _fp8_block_stream(_ALIGNED_Q)
        sink = _load_fused(
            _ALIGNED_Q,
            _FakeQuantConfig("fp8", weight_block_size=list(_BLOCK)),
            stream,
        )
        self.assertTrue(
            torch.equal(
                sink[f"{_WQKV_A}.weight"],
                torch.cat([tensors["wq_a.weight"], tensors["wkv.weight"]], dim=0),
            )
        )
        self.assertTrue(
            torch.equal(
                sink[f"{_WQKV_A}.weight_scale_inv"],
                torch.cat(
                    [tensors["wq_a.weight_scale_inv"], tensors["wkv.weight_scale_inv"]],
                    dim=0,
                ),
            )
        )

    def test_the_fused_scales_describe_the_right_weight_rows(self):
        stream, tensors = _fp8_block_stream(_ALIGNED_Q)
        sink = _load_fused(
            _ALIGNED_Q,
            _FakeQuantConfig("fp8", weight_block_size=list(_BLOCK)),
            stream,
        )
        reference = torch.cat(
            [
                _block_dequantize(
                    tensors["wq_a.weight"], tensors["wq_a.weight_scale_inv"]
                ),
                _block_dequantize(
                    tensors["wkv.weight"], tensors["wkv.weight_scale_inv"]
                ),
            ],
            dim=0,
        )
        fused = _block_dequantize(
            sink[f"{_WQKV_A}.weight"], sink[f"{_WQKV_A}.weight_scale_inv"]
        )
        self.assertTrue(torch.equal(reference, fused))

    def test_mxfp8_fuses_correctly_even_at_a_misaligned_split(self):
        """The [1, 32] layout keeps the join exact where [128, 128] cannot.

        Same misaligned cut that corrupts the fp8-block route above, run on the
        MXFP8 block: one scale row per weight row, so ``torch.cat`` on the scale
        is the identical axis to ``torch.cat`` on the weight.
        """
        stream, tensors = _mxfp8_stream(_MISALIGNED_Q)
        sink = _load_fused(
            _MISALIGNED_Q,
            _FakeQuantConfig("mxfp8", weight_block_size=[1, 32]),
            stream,
        )
        reference = torch.cat(
            [
                _block_dequantize(
                    tensors["wq_a.weight"],
                    tensors["wq_a.weight_scale_inv"],
                    block=[1, 32],
                ),
                _block_dequantize(
                    tensors["wkv.weight"],
                    tensors["wkv.weight_scale_inv"],
                    block=[1, 32],
                ),
            ],
            dim=0,
        )
        fused = _block_dequantize(
            sink[f"{_WQKV_A}.weight"],
            sink[f"{_WQKV_A}.weight_scale_inv"],
            block=[1, 32],
        )
        self.assertTrue(torch.equal(reference, fused))


if __name__ == "__main__":
    unittest.main()
