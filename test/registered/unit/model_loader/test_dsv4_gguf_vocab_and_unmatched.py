# SPDX-License-Identifier: Apache-2.0
"""#391 walls 10-12, 15 and 16: the silent holes on the DeepSeek V4 GGUF load path.

Boot 7 came up with 24 ``not found in params_dict`` warnings and no error. Four
of the five defects below were exactly that shape -- a name that reached the
final parameter lookup, matched nothing, logged one line and left a parameter
at its uninitialized values:

* wall 10 (embed): ``token_embd`` is Q8_0, so the stream ships
  ``embed.qweight``, while ``model.embed_tokens`` is a plain dense
  VocabParallelEmbedding. The embedding never loaded. Fixed in the ADAPTER, by
  dequantizing to a dense ``embed.weight`` -- the same repair gemma4 makes for
  the same reason.
* wall 11 (head): ``output`` is Q6_K, so the stream ships ``head.qweight``,
  while the remap only knew ``head.weight``. ``lm_head.qweight`` /
  ``lm_head.qweight_type`` DO exist (a quantized-resident vocab head is this
  stack's GGUF default), so this is a pure naming miss and the head stays
  packed.
* wall 12 (wo_a): ``attn_output_a`` is Q8_0, but ``wo_a`` is built with
  ``quant_config=None`` below sm100 and ``_compute_o`` reads
  ``self.wo_a.weight``. The packed payload is unpacked at load time.
* wall 15 (structural): under GGUF, an unmatched tensor is now a hard error.
  Every GGUF tensor is mapped by an explicit table, so nothing legitimately
  falls through except the NEXTN (``mtp``) weights the draft loads itself.
* wall 16 (latent, weight_utils): the iterator derived ``qweight`` names with
  ``name.replace("weight", ...)``, which rewrote EVERY occurrence, so
  ``...indexer.weights_proj.weight`` would have become
  ``...indexer.qweights_proj.qweight``. That tensor is F32 in the published
  export, which is the only reason the defect was invisible rather than fatal.

Wall 16's family has a second member, closed here as well: a name with NO
``.weight`` leaf gets no rename at all, so a quantized payload and its 0-dim
type marker would arrive under one name and overwrite each other. The routing
table already had a case for that; every other bare parameter (attention
sinks, compressor APEs, hyper-connection triples) is F32 today and F32 emits
no marker, so the general form is now refused by name rather than left to a
future re-export.

No GPU and no checkpoint. Payloads are real gguf-py quantizations wherever the
load path actually decodes them; the ggml TYPES used are the ones measured on
unsloth/DeepSeek-V4-Flash-0731-GGUF UD-Q3_K_XL.
"""

from __future__ import annotations

import threading
import types
import unittest
from typing import Dict, List, Optional, Set, Tuple
from unittest import mock

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.model_loader.gguf_deepseek4 import Deepseek4GGUFAdapter
from sglang.srt.model_loader.weight_utils import gguf_quantized_name
from sglang.srt.models import deepseek_v4
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# Harness (same shape as test_dsv4_gguf_weight_fusions.py's)
# ---------------------------------------------------------------------------

_HIDDEN = 64


def _ggml(name: str):
    import gguf

    return gguf.GGMLQuantizationType[name]


def _quantize(rows: int, seed: int, qtype_name: str = "Q8_0", cols: int = _HIDDEN):
    """A real gguf-py payload plus the values it decodes to."""
    from gguf.quants import dequantize, quantize

    rng = np.random.default_rng(seed)
    source = rng.standard_normal((rows, cols), dtype=np.float32)
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


def _make_stub(
    quant_name: Optional[str],
    params: dict,
    num_hidden_layers: int = 8,
    n_routed_experts: int = 2,
    is_last_rank: bool = True,
) -> types.SimpleNamespace:
    stub = types.SimpleNamespace()
    stub.config = types.SimpleNamespace(
        num_hidden_layers=num_hidden_layers, n_routed_experts=n_routed_experts
    )
    stub.quant_config = None if quant_name is None else _FakeQuantConfig(quant_name)
    stub.num_fused_shared_experts = 0
    stub.model = types.SimpleNamespace()
    stub.pp_group = types.SimpleNamespace(is_first_rank=True, is_last_rank=is_last_rank)
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
    **stub_kwargs,
) -> Dict[str, torch.Tensor]:
    """Drive the real ``load_weights`` and return what reached each parameter."""
    sink: Dict[str, torch.Tensor] = {}
    params = {
        name: _CapturingParam(name, sink, dtype) for name, dtype in param_specs.items()
    }
    stub = _make_stub(quant_name, params, **stub_kwargs)
    with envs.SGLANG_OPT_FP8_WO_A_GEMM.override(False):
        with envs.SGLANG_OPT_FUSE_WQA_WKV.override(fuse_wqa_wkv):
            DeepseekV4ForCausalLM.load_weights(stub, iter(stream))
    sink.pop("__lock__", None)
    return sink


#: The real remap, aliased at module level: as a TestCase class attribute a
#: plain function would bind `self` as its first argument.
_REMAP = DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format


def _opaque(qtype_name: str, nbytes: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """A payload the load path never decodes, plus its ggml type marker.

    gguf-py can dequantize every K-quant but can only QUANTIZE a few types, so
    a tensor that stays packed end to end (the vocab head) is represented by
    its marker plus opaque bytes rather than a real Q6_K encoding.
    """
    return (
        torch.arange(nbytes, dtype=torch.uint8),
        torch.tensor(int(_ggml(qtype_name))),
    )


class _Cfg:
    """Only the fields the adapter reads."""

    model_type = "deepseek_v4"

    def __init__(self, num_hidden_layers: int, torch_dtype: str = "bfloat16"):
        self.num_hidden_layers = num_hidden_layers
        self.torch_dtype = torch_dtype


def _adapter(num_hidden_layers: int, tensor_names: Optional[Set[str]] = None):
    adapter = Deepseek4GGUFAdapter(
        _Cfg(num_hidden_layers), "/nonexistent/deepseek4.gguf"
    )
    if tensor_names is not None:
        adapter._file_tensor_names = tensor_names
    return adapter


# ---------------------------------------------------------------------------
# wall 16 -- weight_utils: rename the LAST segment only
# ---------------------------------------------------------------------------


class TestQuantizedNameRewritesTheLeafOnly(CustomTestCase):
    _WEIGHTS_PROJ = "model.layers.2.self_attn.indexer.weights_proj.weight"

    def test_weights_proj_keeps_its_module_name(self):
        self.assertEqual(
            gguf_quantized_name(self._WEIGHTS_PROJ, "qweight"),
            "model.layers.2.self_attn.indexer.weights_proj.qweight",
        )
        self.assertEqual(
            gguf_quantized_name(self._WEIGHTS_PROJ, "qweight_type"),
            "model.layers.2.self_attn.indexer.weights_proj.qweight_type",
        )

    def test_can_fail_arm_the_substring_rename_corrupts_the_module_name(self):
        """(a) the pre-fix expression, kept executable, on the same name."""

        def legacy(name: str, leaf: str) -> str:
            return name.replace("weight", leaf)

        self.assertEqual(
            legacy(self._WEIGHTS_PROJ, "qweight"),
            "model.layers.2.self_attn.indexer.qweights_proj.qweight",
        )
        self.assertNotEqual(
            legacy(self._WEIGHTS_PROJ, "qweight"),
            gguf_quantized_name(self._WEIGHTS_PROJ, "qweight"),
        )

    def test_ordinary_names_are_unchanged_by_the_narrowing(self):
        """Every other mapped spelling must rename exactly as it did before."""
        for name in (
            "embed.weight",
            "head.weight",
            "model.layers.3.self_attn.wo_a.weight",
            "model.layers.3.mlp.shared_experts.gate_proj.weight",
            "weight",
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    gguf_quantized_name(name, "qweight"),
                    name.replace("weight", "qweight"),
                )

    def test_bare_parameters_have_no_leaf_to_rename(self):
        """Names with no ``.weight`` leaf pass through, as they did before."""
        for name in (
            "layers.3.attn.attn_sink",
            "layers.3.hc_ffn_fn",
            "layers.3.ffn.gate.tid2eid",
        ):
            with self.subTest(name=name):
                self.assertEqual(gguf_quantized_name(name, "qweight"), name)
                self.assertEqual(name.replace("weight", "qweight"), name)


class TestWeightsProjSurvivesEitherArrival(CustomTestCase):
    """The indexer projection must land whether it is F32 or quantized.

    F32 is what the published export stores, and the only reason wall 16 was
    latent rather than live. A quantized arrival is the hypothetical the fix
    has to cover: the name survives (wall 16) and the payload is unpacked into
    the dense module the C4Indexer actually builds (wall 12's mechanism, whose
    table lists ``weights_proj`` for exactly this case).
    """

    _PARAM = "model.layers.2.self_attn.indexer.weights_proj.weight"
    _NATIVE = "layers.2.attn.indexer.weights_proj"

    def test_f32_arrival_loads_dense(self):
        dense = torch.randn(4, _HIDDEN, dtype=torch.bfloat16)
        sink = _run_load(
            "gguf", [(f"{self._NATIVE}.weight", dense)], {self._PARAM: torch.bfloat16}
        )
        self.assertTrue(torch.equal(sink[self._PARAM], dense))

    def test_quantized_arrival_is_unpacked_into_the_dense_parameter(self):
        packed, qtype, reference = _quantize(4, seed=41)
        sink = _run_load(
            "gguf",
            [
                (gguf_quantized_name(f"{self._NATIVE}.weight", "qweight_type"), qtype),
                (gguf_quantized_name(f"{self._NATIVE}.weight", "qweight"), packed),
            ],
            {self._PARAM: torch.bfloat16},
        )
        self.assertTrue(torch.equal(sink[self._PARAM], reference.to(torch.bfloat16)))


# ---------------------------------------------------------------------------
# wall 10 -- embed
# ---------------------------------------------------------------------------


class TestEmbedIsDequantizedByTheAdapter(CustomTestCase):
    _PARAM = "model.embed_tokens.weight"

    def _stream(self, packed, qtype):
        return [("embed.qweight_type", qtype), ("embed.qweight", packed)]

    def test_embed_arrives_dense_and_equals_the_reference_dequant(self):
        packed, qtype, reference = _quantize(16, seed=21)
        out = list(_adapter(4).transform_stream(self._stream(packed, qtype)))

        self.assertEqual([name for name, _ in out], ["embed.weight"])
        dense = out[0][1]
        self.assertEqual(dense.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(dense, reference.to(torch.bfloat16)))

    def test_the_dense_embedding_parameter_is_actually_filled(self):
        packed, qtype, reference = _quantize(16, seed=22)
        stream = list(_adapter(4).transform_stream(self._stream(packed, qtype)))
        sink = _run_load("gguf", stream, {self._PARAM: torch.bfloat16})
        self.assertTrue(torch.equal(sink[self._PARAM], reference.to(torch.bfloat16)))

    def test_can_fail_arm_without_the_dequant_the_embedding_stays_empty(self):
        """(a) the pre-fix stream, unchanged by the adapter, into load_weights.

        Boot 7 logged ``embed.qweight_type not found in params_dict`` and came
        up on an uninitialized embedding. With wall 15 in place the same stream
        now fails loudly instead.
        """
        packed, qtype, _ = _quantize(16, seed=23)
        with self.assertRaises(KeyError) as ctx:
            _run_load(
                "gguf", self._stream(packed, qtype), {self._PARAM: torch.bfloat16}
            )
        self.assertIn("model.embed_tokens.qweight_type", str(ctx.exception))

    def test_an_f32_export_passes_through_untouched(self):
        dense = torch.randn(8, _HIDDEN)
        out = list(_adapter(4).transform_stream([("embed.weight", dense)]))
        self.assertEqual(len(out), 1)
        self.assertTrue(torch.equal(out[0][1], dense))


class TestBareParameterNameCollisionIsRefused(CustomTestCase):
    """Wall 16's family, generalized: a name with no ``.weight`` leaf gets no
    rename, so a quantized payload and its 0-dim marker would arrive under one
    name and overwrite each other. Every such tensor is F32 today (attention
    sinks, compressor APEs, hyper-connection triples) and F32 emits no marker,
    so this is a re-export guard, not a live path.
    """

    def test_f32_bare_parameters_still_pass_through(self):
        sink = torch.randn(64)
        out = list(
            _adapter(4).transform_stream(
                [
                    ("layers.3.attn.attn_sink", sink),
                    ("layers.3.hc_ffn_fn", sink),
                    ("hc_head_scale", sink),
                ]
            )
        )
        self.assertEqual(
            [name for name, _ in out],
            ["layers.3.attn.attn_sink", "layers.3.hc_ffn_fn", "hc_head_scale"],
        )

    def test_the_integer_routing_table_keeps_its_own_case(self):
        table = torch.zeros(4, 6, dtype=torch.int32)
        out = list(
            _adapter(4).transform_stream(
                [
                    ("layers.0.ffn.gate.tid2eid", torch.tensor(int(_ggml("I32")))),
                    ("layers.0.ffn.gate.tid2eid", table),
                ]
            )
        )
        self.assertEqual(len(out), 1)
        self.assertTrue(torch.equal(out[0][1], table))

    def test_a_quantized_bare_parameter_is_refused_by_name(self):
        marker = torch.tensor(int(_ggml("Q8_0")))
        with self.assertRaises(RuntimeError) as ctx:
            list(_adapter(4).transform_stream([("layers.3.attn.attn_sink", marker)]))
        message = str(ctx.exception)
        self.assertIn("layers.3.attn.attn_sink", message)
        self.assertIn("bare parameter", message)

    def test_the_head_is_left_packed(self):
        """tie_word_embeddings is false here, so head is a separate decision."""
        packed, qtype = _opaque("Q6_K")
        out = list(
            _adapter(4).transform_stream(
                [("head.qweight_type", qtype), ("head.qweight", packed)]
            )
        )
        self.assertEqual(
            [name for name, _ in out], ["head.qweight_type", "head.qweight"]
        )


# ---------------------------------------------------------------------------
# wall 11 -- head
# ---------------------------------------------------------------------------


class TestHeadRemapKeepsItsLeaf(CustomTestCase):
    def test_every_leaf_maps_to_lm_head(self):
        for leaf in ("weight", "qweight", "qweight_type"):
            with self.subTest(leaf=leaf):
                self.assertEqual(_REMAP(f"head.{leaf}"), f"lm_head.{leaf}")
                self.assertEqual(_REMAP(f"embed.{leaf}"), f"model.embed_tokens.{leaf}")

    def test_hyper_connection_head_names_are_not_swallowed(self):
        self.assertEqual(_REMAP("hc_head_base"), "model.hc_head_base")
        self.assertEqual(_REMAP("hc_head_scale"), "model.hc_head_scale")

    def test_the_packed_head_lands_with_its_marker(self):
        packed, qtype = _opaque("Q6_K")
        sink = _run_load(
            "gguf",
            [("head.qweight_type", qtype), ("head.qweight", packed)],
            {"lm_head.qweight": torch.uint8, "lm_head.qweight_type": torch.uint8},
        )
        self.assertEqual(sorted(sink), ["lm_head.qweight", "lm_head.qweight_type"])
        self.assertTrue(torch.equal(sink["lm_head.qweight"], packed))
        self.assertEqual(int(sink["lm_head.qweight_type"].item()), int(_ggml("Q6_K")))

    def test_can_fail_arm_the_exact_match_remap_leaves_both_tensors_homeless(self):
        """(a) the pre-fix remap, kept executable, on the boot-7 names."""

        def legacy(name: str) -> str:
            if name == "head.weight":
                return "lm_head.weight"
            return name

        for leaf in ("qweight", "qweight_type"):
            with self.subTest(leaf=leaf):
                self.assertEqual(legacy(f"head.{leaf}"), f"head.{leaf}")
                self.assertEqual(_REMAP(f"head.{leaf}"), f"lm_head.{leaf}")

    def test_a_non_last_pp_rank_skips_the_whole_head(self):
        packed, qtype = _opaque("Q6_K")
        sink = _run_load(
            "gguf",
            [("head.qweight_type", qtype), ("head.qweight", packed)],
            {},
            is_last_rank=False,
        )
        self.assertEqual(sink, {})


# ---------------------------------------------------------------------------
# wall 12 -- wo_a
# ---------------------------------------------------------------------------

_WO_A_NATIVE = "layers.0.attn.wo_a"
_WO_A_PARAM = "model.layers.0.self_attn.wo_a.weight"
#: The boot-7 warning name, verbatim (boot7.log, 8 distinct names x 3 ranks).
_BOOT7_WO_A = "model.layers.0.self_attn.wo_a.qweight_type not found in params_dict."


class TestWoAIsUnpackedIntoItsDenseParameter(CustomTestCase):
    def _stream(self, packed, qtype):
        return [
            (f"{_WO_A_NATIVE}.qweight_type", qtype),
            (f"{_WO_A_NATIVE}.qweight", packed),
        ]

    def test_wo_a_equals_the_reference_dequant(self):
        packed, qtype, reference = _quantize(8, seed=51)
        sink = _run_load(
            "gguf", self._stream(packed, qtype), {_WO_A_PARAM: torch.bfloat16}
        )
        self.assertEqual(list(sink), [_WO_A_PARAM])
        self.assertEqual(sink[_WO_A_PARAM].dtype, torch.bfloat16)
        self.assertTrue(torch.equal(sink[_WO_A_PARAM], reference.to(torch.bfloat16)))

    def test_an_incomplete_pair_never_reaches_the_parameter(self):
        packed, qtype, _ = _quantize(8, seed=52)
        sink: Dict[str, torch.Tensor] = {}
        params = {_WO_A_PARAM: _CapturingParam(_WO_A_PARAM, sink, torch.bfloat16)}
        stub = _make_stub("gguf", params)
        with envs.SGLANG_OPT_FP8_WO_A_GEMM.override(False):
            with self.assertRaises(AssertionError):
                DeepseekV4ForCausalLM.load_weights(
                    stub, iter(self._stream(packed, qtype)[:1])
                )
        sink.pop("__lock__", None)
        self.assertEqual(sink, {})

    def test_a_dense_wo_a_still_takes_the_ordinary_path(self):
        """(d) the safetensors route is untouched: only packed leaves match."""
        dense = torch.randn(8, _HIDDEN, dtype=torch.bfloat16)
        sink = _run_load(
            "fp8", [(f"{_WO_A_NATIVE}.weight", dense)], {_WO_A_PARAM: torch.bfloat16}
        )
        self.assertTrue(torch.equal(sink[_WO_A_PARAM], dense))

    def test_can_fail_arm_boot7_warning_reproduced_verbatim(self):
        """(a) with the unpack site removed, boot 7's log line comes back.

        Pinned on a NON-GGUF quant config so wall 15 does not preempt the
        warning this arm exists to reproduce; the GGUF arm below is the same
        stream under the hard-fail rule.
        """
        packed, qtype, _ = _quantize(8, seed=53)
        with mock.patch.object(deepseek_v4, "_GGUF_DENSE_PROJECTIONS", ()):
            with self.assertLogs(deepseek_v4.logger, level="WARNING") as logs:
                sink = _run_load(
                    "fp8", self._stream(packed, qtype), {_WO_A_PARAM: torch.bfloat16}
                )
        self.assertIn(_BOOT7_WO_A, [record.getMessage() for record in logs.records])
        self.assertEqual(sink, {}, "wo_a must have stayed unfilled in this arm")

    def test_can_fail_arm_the_same_stream_is_now_fatal_under_gguf(self):
        packed, qtype, _ = _quantize(8, seed=54)
        with mock.patch.object(deepseek_v4, "_GGUF_DENSE_PROJECTIONS", ()):
            with self.assertRaises(KeyError) as ctx:
                _run_load(
                    "gguf", self._stream(packed, qtype), {_WO_A_PARAM: torch.bfloat16}
                )
        self.assertIn("wo_a.qweight_type", str(ctx.exception))

    def test_wo_b_is_not_in_the_dense_set(self):
        """wo_b keeps the model's quant_config, so it stays packed."""
        self.assertIsNone(
            deepseek_v4._gguf_dense_projection_base(
                "model.layers.0.self_attn.wo_b.qweight"
            )
        )
        for name in (
            "model.layers.0.self_attn.wq_b.qweight",
            "model.layers.0.self_attn.indexer.wq_b.qweight",
            "model.layers.0.mlp.shared_experts.down_proj.qweight",
        ):
            with self.subTest(name=name):
                self.assertIsNone(deepseek_v4._gguf_dense_projection_base(name))


# ---------------------------------------------------------------------------
# wall 15 -- unmatched is fatal under GGUF
# ---------------------------------------------------------------------------


class TestUnmatchedTensorIsFatalUnderGguf(CustomTestCase):
    def test_a_typo_names_the_tensor_and_stops_the_load(self):
        """(the can-fail: plant a name no parameter has)."""
        dense = torch.randn(4, _HIDDEN, dtype=torch.bfloat16)
        with self.assertRaises(KeyError) as ctx:
            _run_load(
                "gguf",
                [("layers.0.attn.q_nrom.weight", dense)],
                {"model.layers.0.self_attn.q_norm.weight": torch.bfloat16},
            )
        message = str(ctx.exception)
        self.assertIn("model.layers.0.self_attn.q_nrom.weight", message)
        self.assertIn("matched no parameter", message)

    def test_the_clean_stream_still_loads(self):
        dense = torch.randn(4, _HIDDEN, dtype=torch.bfloat16)
        sink = _run_load(
            "gguf",
            [("layers.0.attn.q_norm.weight", dense)],
            {"model.layers.0.self_attn.q_norm.weight": torch.bfloat16},
        )
        self.assertTrue(
            torch.equal(sink["model.layers.0.self_attn.q_norm.weight"], dense)
        )

    def test_the_dense_route_still_only_warns(self):
        """(d) backward compatibility: safetensors keeps its warning."""
        dense = torch.randn(4, _HIDDEN, dtype=torch.bfloat16)
        with self.assertLogs(deepseek_v4.logger, level="WARNING") as logs:
            _run_load(
                "fp8",
                [("layers.0.attn.q_nrom.weight", dense)],
                {"model.layers.0.self_attn.q_norm.weight": torch.bfloat16},
            )
        self.assertIn(
            "model.layers.0.self_attn.q_nrom.weight not found in params_dict.",
            [record.getMessage() for record in logs.records],
        )

    def test_nextn_weights_are_still_skipped_silently(self):
        """The one named-ignorable class: the draft loads them in its own pass."""
        dense = torch.randn(4, _HIDDEN, dtype=torch.bfloat16)
        sink = _run_load("gguf", [("mtp.norm.weight", dense)], {})
        self.assertEqual(sink, {})


# ---------------------------------------------------------------------------
# full-stream integration -- the boot-8 gate
# ---------------------------------------------------------------------------

NUM_LAYERS = 43
COMPRESSOR_LAYERS = range(2, 43)
INDEXER_LAYERS = range(2, 43, 2)
HASH_LAYERS = (0, 1, 2)
BIAS_LAYERS = range(3, 43)
#: Reduced expert width. The per-expert split is one hardcoded name template in
#: the generic iterator, exercised identically at 4 experts and at 256; the
#: layer/tensor STRUCTURE below is the published export's, unreduced.
NUM_EXPERTS = 4

#: ggml type per gguf tensor suffix, as measured on
#: unsloth/DeepSeek-V4-Flash-0731-GGUF UD-Q3_K_XL (1328 tensors, 4 shards).
#: The published ``ffn_*_exps`` mix of IQ3_XXS and MXFP4 is written here as the
#: post-repack Q5_0 that the iterator actually emits (gguf_mxfp4_repack, whose
#: own test pins the conversion) -- the expert payloads are opaque bytes on
#: this path, so only the marker's presence matters.
_TYPES: Dict[str, str] = {
    "token_embd.weight": "Q8_0",
    "output.weight": "Q6_K",
    "output_norm.weight": "F32",
    "output_hc_base.weight": "F32",
    "output_hc_fn.weight": "F32",
    "output_hc_scale.weight": "F32",
    "attn_norm.weight": "F32",
    "ffn_norm.weight": "F32",
    "attn_q_a.weight": "Q8_0",
    "attn_q_b.weight": "Q8_0",
    "attn_kv.weight": "Q8_0",
    "attn_output_a.weight": "Q8_0",
    "attn_output_b.weight": "Q8_0",
    "attn_q_a_norm.weight": "F32",
    "attn_kv_a_norm.weight": "F32",
    "attn_sinks.weight": "F32",
    "ffn_gate_inp.weight": "BF16",
    "exp_probs_b.bias": "F32",
    "ffn_gate_tid2eid.weight": "I32",
    "ffn_gate_shexp.weight": "Q8_0",
    "ffn_down_shexp.weight": "Q8_0",
    "ffn_up_shexp.weight": "Q8_0",
    "ffn_gate_exps.weight": "Q5_0",
    "ffn_down_exps.weight": "Q5_0",
    "ffn_up_exps.weight": "Q5_0",
    "attn_compressor_ape.weight": "F32",
    "attn_compressor_gate.weight": "Q8_0",
    "attn_compressor_kv.weight": "Q8_0",
    "attn_compressor_norm.weight": "F32",
    "indexer.attn_q_b.weight": "Q8_0",
    "indexer.proj.weight": "F32",
    "indexer_compressor_ape.weight": "F32",
    "indexer_compressor_gate.weight": "Q8_0",
    "indexer_compressor_kv.weight": "Q8_0",
    "indexer_compressor_norm.weight": "F32",
    "hc_attn_base.weight": "F32",
    "hc_attn_fn.weight": "F32",
    "hc_attn_scale.weight": "F32",
    "hc_ffn_base.weight": "F32",
    "hc_ffn_fn.weight": "F32",
    "hc_ffn_scale.weight": "F32",
}

#: Native leaves the load path DEQUANTIZES, so their payloads must be real.
#: Everything else travels as opaque bytes.
_REALLY_DECODED = (
    "token_embd.weight",
    "attn_output_a.weight",
    "attn_compressor_gate.weight",
    "attn_compressor_kv.weight",
    "indexer_compressor_gate.weight",
    "indexer_compressor_kv.weight",
)

_MOE_STACKED = {
    "ffn_gate_exps.weight": "gate_proj",
    "ffn_down_exps.weight": "down_proj",
    "ffn_up_exps.weight": "up_proj",
}


def _file_tensor_names() -> Set[str]:
    """The tensor set of the published export, by structure.

    Mirrors ``test_gguf_deepseek4_name_map.py``'s fixture; the 1328 assertion
    below is what keeps the two from drifting apart silently.
    """
    names = {
        "token_embd.weight",
        "output.weight",
        "output_norm.weight",
        "output_hc_base.weight",
        "output_hc_fn.weight",
        "output_hc_scale.weight",
    }
    per_layer = [
        "attn_norm.weight",
        "ffn_norm.weight",
        "attn_q_a.weight",
        "attn_q_b.weight",
        "attn_kv.weight",
        "attn_output_a.weight",
        "attn_output_b.weight",
        "attn_q_a_norm.weight",
        "attn_kv_a_norm.weight",
        "attn_sinks.weight",
        "ffn_gate_inp.weight",
        "ffn_gate_shexp.weight",
        "ffn_down_shexp.weight",
        "ffn_up_shexp.weight",
        "ffn_gate_exps.weight",
        "ffn_down_exps.weight",
        "ffn_up_exps.weight",
        "hc_attn_base.weight",
        "hc_attn_fn.weight",
        "hc_attn_scale.weight",
        "hc_ffn_base.weight",
        "hc_ffn_fn.weight",
        "hc_ffn_scale.weight",
    ]
    for i in range(NUM_LAYERS):
        for suffix in per_layer:
            names.add(f"blk.{i}.{suffix}")
        if i in COMPRESSOR_LAYERS:
            for suffix in (
                "attn_compressor_ape.weight",
                "attn_compressor_gate.weight",
                "attn_compressor_kv.weight",
                "attn_compressor_norm.weight",
            ):
                names.add(f"blk.{i}.{suffix}")
        if i in INDEXER_LAYERS:
            for suffix in (
                "indexer.attn_q_b.weight",
                "indexer.proj.weight",
                "indexer_compressor_ape.weight",
                "indexer_compressor_gate.weight",
                "indexer_compressor_kv.weight",
                "indexer_compressor_norm.weight",
            ):
                names.add(f"blk.{i}.{suffix}")
        if i in HASH_LAYERS:
            names.add(f"blk.{i}.ffn_gate_tid2eid.weight")
        if i in BIAS_LAYERS:
            names.add(f"blk.{i}.exp_probs_b.bias")
    return names


def _suffix_of(gguf_name: str) -> str:
    head, _, rest = gguf_name.partition(".")
    if head != "blk":
        return gguf_name
    return rest.split(".", 1)[1]


def _payload(gguf_name: str, qtype_name: str, seed: int) -> torch.Tensor:
    """One tensor's payload, real where the load path decodes it."""
    if _suffix_of(gguf_name) in _REALLY_DECODED:
        return _quantize(8, seed=seed, qtype_name=qtype_name)[0]
    if qtype_name == "F32":
        return torch.randn(4, _HIDDEN)
    if qtype_name == "I32":
        return torch.zeros(4, 6, dtype=torch.int32)
    if qtype_name == "BF16":
        # gguf-py hands BF16 back as uint8 with the last dim doubled.
        return torch.randn(4, _HIDDEN, dtype=torch.bfloat16).view(torch.uint8)
    return torch.zeros(64, dtype=torch.uint8)


def _emit_like_the_iterator(name_map: Dict[str, str]) -> List[Tuple[str, torch.Tensor]]:
    """The two passes ``gguf_quant_weights_iterator`` makes over the file.

    Pass 1 is every ggml type marker, pass 2 every payload -- that ordering is
    load-bearing for the fusion caches. The leaf rename goes through the real
    ``gguf_quantized_name``, so the naming rule under test is not re-stated
    here.
    """
    markers: List[Tuple[str, torch.Tensor]] = []
    payloads: List[Tuple[str, torch.Tensor]] = []
    for seed, gguf_name in enumerate(sorted(name_map)):
        native = name_map[gguf_name]
        suffix = _suffix_of(gguf_name)
        qtype_name = _TYPES[suffix]
        quantized = qtype_name != "F32"

        if suffix in _MOE_STACKED:
            layer = int(gguf_name.split(".")[1])
            proj = _MOE_STACKED[suffix]
            for expert in range(NUM_EXPERTS):
                base = f"model.layers.{layer}.mlp.experts.{expert}.{proj}"
                markers.append(
                    (f"{base}.qweight_type", torch.tensor(int(_ggml(qtype_name))))
                )
                payloads.append((f"{base}.qweight", torch.zeros(64, dtype=torch.uint8)))
            continue

        payload = _payload(gguf_name, qtype_name, seed)
        if quantized:
            markers.append(
                (
                    gguf_quantized_name(native, "qweight_type"),
                    torch.tensor(int(_ggml(qtype_name))),
                )
            )
            payloads.append((gguf_quantized_name(native, "qweight"), payload))
        else:
            payloads.append((native, payload))
    return markers + payloads


def _expected_params() -> Dict[str, torch.dtype]:
    """Every parameter DeepseekV4ForCausalLM holds under a GGUF quant_config.

    Written from the module CONSTRUCTORS, not from the stream: a projection
    built with the model's quant_config keeps ``.qweight`` + ``.qweight_type``,
    one built with ``quant_config=None`` (wo_a, weights_proj, the fused
    compressor wkv_gate) keeps a dense ``.weight``, and the vocab embedding is
    a plain VocabParallelEmbedding while the lm_head is a quantized
    ParallelLMHead.
    """
    packed = torch.uint8
    dense = torch.bfloat16
    params: Dict[str, torch.dtype] = {
        "model.embed_tokens.weight": dense,
        "lm_head.qweight": packed,
        "lm_head.qweight_type": packed,
        "model.norm.weight": dense,
        "model.hc_head_base": dense,
        "model.hc_head_fn": dense,
        "model.hc_head_scale": dense,
    }
    for i in range(NUM_LAYERS):
        layer = f"model.layers.{i}"
        attn = f"{layer}.self_attn"
        params[f"{layer}.input_layernorm.weight"] = dense
        params[f"{layer}.post_attention_layernorm.weight"] = dense
        for leaf in ("base", "fn", "scale"):
            params[f"{layer}.hc_attn_{leaf}"] = dense
            params[f"{layer}.hc_ffn_{leaf}"] = dense
        for projection in ("wqkv_a", "wq_b", "wo_b"):
            params[f"{attn}.{projection}.qweight"] = packed
            params[f"{attn}.{projection}.qweight_type"] = packed
        params[f"{attn}.wo_a.weight"] = dense
        params[f"{attn}.q_norm.weight"] = dense
        params[f"{attn}.kv_norm.weight"] = dense
        params[f"{attn}.attn_sink"] = dense
        if i in COMPRESSOR_LAYERS:
            params[f"{attn}.compressor.wkv_gate.weight"] = dense
            params[f"{attn}.compressor.ape"] = dense
            params[f"{attn}.compressor.norm.weight"] = dense
        if i in INDEXER_LAYERS:
            params[f"{attn}.indexer.wq_b.qweight"] = packed
            params[f"{attn}.indexer.wq_b.qweight_type"] = packed
            params[f"{attn}.indexer.weights_proj.weight"] = dense
            params[f"{attn}.indexer.compressor.wkv_gate.weight"] = dense
            params[f"{attn}.indexer.compressor.ape"] = dense
            params[f"{attn}.indexer.compressor.norm.weight"] = dense
        params[f"{layer}.mlp.gate.weight"] = dense
        if i in HASH_LAYERS:
            params[f"{layer}.mlp.topk.tid2eid"] = torch.int32
        if i in BIAS_LAYERS:
            params[f"{layer}.mlp.gate.e_score_correction_bias"] = dense
        # The shared expert is a DeepseekV2MLP: gate/up are one merged module.
        for projection in ("gate_up_proj", "down_proj"):
            params[f"{layer}.mlp.shared_experts.{projection}.qweight"] = packed
            params[f"{layer}.mlp.shared_experts.{projection}.qweight_type"] = packed
        for stacked in ("w13", "w2"):
            params[f"{layer}.mlp.experts.{stacked}_qweight"] = packed
            params[f"{layer}.mlp.experts.{stacked}_qweight_type"] = packed
    return params


class TestFullStreamLoadsClean(CustomTestCase):
    """The boot-8 gate: the whole published tensor set, zero warnings.

    Zero warnings covers both directions at once -- no tensor of the stream
    reaches the unmatched check (which is now fatal anyway), and no parameter
    of the model is left uninitialized, which is the other warning
    ``load_weights`` ends on.
    """

    @classmethod
    def setUpClass(cls):
        names = _file_tensor_names()
        cls.name_map = _adapter(NUM_LAYERS, names)._build_name_map_unchecked()
        cls.stream = list(
            _adapter(NUM_LAYERS).transform_stream(_emit_like_the_iterator(cls.name_map))
        )
        cls.params = _expected_params()

    def test_the_fixture_is_the_published_tensor_set(self):
        self.assertEqual(len(self.name_map), 1328)

    def test_zero_warnings_and_every_parameter_filled(self):
        with mock.patch.object(deepseek_v4.logger, "warning") as warn:
            sink = _run_load(
                "gguf",
                self.stream,
                self.params,
                num_hidden_layers=NUM_LAYERS,
                n_routed_experts=NUM_EXPERTS,
            )
        self.assertEqual(
            warn.call_args_list, [], f"load_weights warned: {warn.call_args_list}"
        )
        self.assertEqual(
            set(self.params) - set(sink),
            set(),
            "parameters no tensor of the stream reached",
        )

    def test_a_single_planted_typo_fails_the_whole_load(self):
        """The gate has to be able to fail: rename one tensor of 1328."""
        planted = [
            (
                (
                    "layers.9.attn.wo_bb.qweight"
                    if name.endswith("layers.9.attn.wo_b.qweight")
                    else name
                ),
                tensor,
            )
            for name, tensor in self.stream
        ]
        with self.assertRaises(KeyError) as ctx:
            _run_load(
                "gguf",
                planted,
                self.params,
                num_hidden_layers=NUM_LAYERS,
                n_routed_experts=NUM_EXPERTS,
            )
        self.assertIn("wo_bb", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
