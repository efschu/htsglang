# SPDX-License-Identifier: Apache-2.0
"""NF-GGUF G14 (2026-10-12): a BF16 tensor must reach a GGUF linear as bf16, not as raw bytes.

Metal (G13 stage A, report ``done/gguf-nf-metall-g13-1012.md`` 3.4): the D boot died in the first CUDA-graph
capture on layer 3, the first layer with a QSA indexer::

    layers/quantization/gguf.py:1338 fused_mul_mat_gguf: x @ qweight.T
    RuntimeError: mat1 and mat2 shapes cannot be multiplied (24x2560 and 5120x640)

Root (read in the loader, not guessed). gguf-py hands an unquantized BF16 tensor back as ``uint8`` with the last
dimension DOUBLED (numpy has no bfloat16). ``gguf_quant_weights_iterator`` renames every non-F32 tensor that is not a
known dense module (``GGUF_DENSE_PARAM_SUFFIXES``) to ``.qweight`` and used to hand that payload over untouched. For the
quantized types that is right (block bytes). For the UNQUANTIZED types the consumer is a different one:
``fused_mul_mat_gguf`` (``qweight_type in UNQUANTIZED_TYPES`` -> ``x @ qweight.T``), ``apply_gguf_embedding``
(``torch.embedding``), ``GGUFLinearMethod._cast_dense_qweight`` / ``_create_flat_weight_param`` (both gated on
``is_floating_point()``) -- all of them expect a real-valued tensor of the element shape. F16 happens to arrive that way
(gguf-py returns ``float16``), BF16 does not: ``[640, 2560]`` BF16 arrived as ``uint8 [640, 5120]`` (5120 = 2560 x 2
bytes). The #647 fix (``gguf_dense_payload``) re-viewed the payload only for the two dense-module suffixes; the
indexer ``index_qk_proj`` is a quantized-resident ``ReplicatedLinear`` with the BF16 type, so it kept the raw bytes.
The existing G1 test pinned the bytes (``np.array_equal(fused.numpy(), bf16_bytes(...))``) and never ran ``apply``.

These tests drive the REAL chain: GGUFWriter file -> ``gguf_quant_weights_iterator`` -> ``Qwen4ExpGGUFAdapter`` stream ->
``ReplicatedLinear`` with ``GGUFConfig`` (``weight_loader`` -> ``process_weights_after_loading`` -> ``apply``), CPU.
"""

from __future__ import annotations

import os

# BEFORE the first ``import torch`` (the wrapper sets it empty anyway; a bare pytest run does not)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import torch

import gguf
from gguf.quants import dequantize, quantize

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization import gguf as G
from sglang.srt.model_loader import gguf_qwen4exp as Q4
from sglang.srt.model_loader import weight_utils as WU
from sglang.srt.model_loader.weight_utils import gguf_quant_weights_iterator

import test_gguf_qwen4exp_g1_1009 as G1T  # the tiny qwen4exp file with all 47 roles

Q = gguf.GGMLQuantizationType

# the real geometry of the QSA indexer (config.json: hidden 2560, 4 index heads + 1 kv head, head dim 128)
REAL_H = 2560
REAL_NQ = 4 * 128
REAL_NK = 1 * 128
IDX = "model.layers.3.self_attn.indexer.index_qk_proj"


@pytest.fixture(autouse=True)
def _no_cuda_probe(monkeypatch):
    # ``fused_mul_mat_gguf`` sizes its MMVQ threshold from the current CUDA device BEFORE the unquantized branch; the
    # override makes the CPU run reach the branch under test (it does not touch the unquantized path itself).
    monkeypatch.setenv("SGLANG_GGUF_MMVQ_SAFE", "2")


def _bf16_ref(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16)


def _write_indexer_file(path, w_q: torch.Tensor, w_k: torch.Tensor) -> None:
    """A GGUF with just the two BF16 indexer halves, as the unsloth export stores them."""
    writer = gguf.GGUFWriter(str(path), "qwen4exp")
    for name, w in (("blk.3.indexer.q_proj.weight", w_q), ("blk.3.indexer.k_proj.weight", w_k)):
        writer.add_tensor(
            name,
            quantize(w.float().numpy(), Q.BF16),
            raw_dtype=Q.BF16,
            raw_shape=(w.shape[0], w.shape[1] * 2),
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _adapter_for(path, hidden=REAL_H):
    cfg = G1T.make_config(hidden_size=hidden)
    return Q4.Qwen4ExpGGUFAdapter(cfg, str(path))


def _stream_halves(path, adapter=None):
    """The indexer tensors as the loader's stream yields them (iterator + qwen4exp pre-stage)."""
    adapter = adapter or _adapter_for(path)
    nm = {
        "blk.3.indexer.q_proj.weight": "model.layers.3.self_attn.indexer.index_qk_proj.__q.weight",
        "blk.3.indexer.k_proj.weight": "model.layers.3.self_attn.indexer.index_qk_proj.__k.weight",
    }
    out = {}
    for name, w in adapter._pre_stream(gguf_quant_weights_iterator(str(path), nm)):
        out[name.removesuffix(Q4._FINAL)] = w
    return out


def _linear_loaded(stream, n_out, n_in, params_dtype=torch.bfloat16):
    """ReplicatedLinear + GGUF method, loaded the way ``load_weights`` does (type marker first)."""
    layer = ReplicatedLinear(
        n_in, n_out, bias=False, quant_config=G.GGUFConfig([]), prefix=IDX, params_dtype=params_dtype
    )
    params = dict(layer.named_parameters())
    for leaf in ("qweight_type", "qweight"):
        p = params[leaf]
        p.weight_loader(p, stream[f"{IDX}.{leaf}"])
    layer.quant_method.process_weights_after_loading(layer)
    return layer


@pytest.fixture
def real_shape_file(tmp_path):
    g = torch.Generator().manual_seed(14)
    # values that bf16 represents exactly are NOT needed: the reference is the bf16 rounding of the same floats
    w_q = torch.randn(REAL_NQ, REAL_H, generator=g)
    w_k = torch.randn(REAL_NK, REAL_H, generator=g)
    path = tmp_path / "indexer-real-shape.gguf"
    _write_indexer_file(path, w_q, w_k)
    return path, w_q, w_k


# ---------------------------------------------------------------------------
# the chain that died on the metal
# ---------------------------------------------------------------------------


def test_the_indexer_arrives_as_bf16_with_the_element_shape(real_shape_file):
    """[640, 2560] bf16 -- not uint8 [640, 5120]."""
    path, w_q, w_k = real_shape_file
    out = _stream_halves(path)
    fused = out[f"{IDX}.qweight"]
    assert int(out[f"{IDX}.qweight_type"].item()) == int(Q.BF16)
    assert fused.dtype == torch.bfloat16, f"raw payload handed over: {fused.dtype} {tuple(fused.shape)}"
    assert tuple(fused.shape) == (REAL_NQ + REAL_NK, REAL_H)
    want = torch.cat([_bf16_ref(w_q), _bf16_ref(w_k)], dim=0)  # q rows first, then k rows
    assert torch.equal(fused, want)


@pytest.mark.parametrize("m", [1, 2, 3, 6, 8, 24, 192])
def test_index_qk_proj_apply_matches_the_torch_reference(real_shape_file, m):
    """The metal failure: M = 24 (bs 6 x MTP verify 4) -> 'mat1 and mat2 shapes cannot be multiplied (24x2560 and
    5120x640)'. Now apply() equals the torch bf16 reference for every M the graph capture / prefill uses."""
    path, w_q, w_k = real_shape_file
    stream = _stream_halves(path)
    layer = _linear_loaded(stream, REAL_NQ + REAL_NK, REAL_H)
    g = torch.Generator().manual_seed(100 + m)
    x = torch.randn(m, REAL_H, generator=g).to(torch.bfloat16)
    got = layer.quant_method.apply(layer, x)
    w = torch.cat([_bf16_ref(w_q), _bf16_ref(w_k)], dim=0)
    want = x @ w.T
    assert got.shape == (m, REAL_NQ + REAL_NK) and got.dtype == torch.bfloat16
    assert torch.equal(got, want)  # same bf16 matmul on the same operands: bit-equal
    # the halves are not interchangeable: swapped rows give a different result
    swapped = x @ torch.cat([w[REAL_NQ:], w[:REAL_NQ]], dim=0).T
    assert not torch.equal(got, swapped)


def test_the_error_of_the_metal_run_is_gone_by_construction(real_shape_file):
    """The loaded parameter itself has the element shape and a real dtype (what ``x @ qweight.T`` needs)."""
    path, _, _ = real_shape_file
    layer = _linear_loaded(_stream_halves(path), REAL_NQ + REAL_NK, REAL_H)
    assert layer.qweight.dtype == torch.bfloat16
    assert tuple(layer.qweight.shape) == (REAL_NQ + REAL_NK, REAL_H)
    x = torch.zeros(24, REAL_H, dtype=torch.bfloat16)
    G.fused_mul_mat_gguf(x, layer.qweight, layer.qweight_type.weight_type)  # must not raise


def test_fp16_params_get_the_bf16_indexer_cast_once(real_shape_file):
    """A float16 model with a BF16 tensor: the load-time normalisation casts it once (``_cast_dense_qweight``), which
    is only reachable for a real-valued qweight."""
    path, w_q, w_k = real_shape_file
    layer = _linear_loaded(_stream_halves(path), REAL_NQ + REAL_NK, REAL_H, params_dtype=torch.float16)
    assert layer.qweight.dtype == torch.float16
    assert layer.qweight_type.weight_type == int(G._weight_type_for_dtype(torch.float16))
    x = torch.randn(3, REAL_H).to(torch.float16)
    got = layer.quant_method.apply(layer, x)
    w = torch.cat([w_q, w_k], dim=0).to(torch.bfloat16).to(torch.float16)
    torch.testing.assert_close(got, x @ w.T)


# ---------------------------------------------------------------------------
# the whole tiny qwen4exp file (all 47 roles) through the real adapter
# ---------------------------------------------------------------------------


def test_tiny_pipeline_indexer_is_bf16_values_not_bytes(tmp_path):
    path = str(tmp_path / "m-00001-of-00001.gguf")
    ref = G1T.build_tiny(path)
    _, _, out = G1T.run_pipeline(path)
    fused = out[f"{IDX}.qweight"]
    n_q = G1T.IDX_HEADS * G1T.IDX_DIM
    assert fused.dtype == torch.bfloat16
    assert tuple(fused.shape) == (n_q + G1T.IDX_DIM, G1T.H)
    assert torch.equal(fused, _bf16_ref(ref.t["l3.idx"]))
    # and it runs through a GGUF linear
    layer = _linear_loaded(out, n_q + G1T.IDX_DIM, G1T.H)
    x = torch.randn(5, G1T.H).to(torch.bfloat16)
    assert torch.equal(layer.quant_method.apply(layer, x), x @ _bf16_ref(ref.t["l3.idx"]).T)


# ---------------------------------------------------------------------------
# the class: every unquantized type that reaches a quantized-resident module is a typed tensor
# ---------------------------------------------------------------------------


def _write_generic(path, tensors):
    writer = gguf.GGUFWriter(str(path), "qwen4exp")
    for name, (arr, qtype) in tensors.items():
        if qtype == Q.BF16:
            writer.add_tensor(
                name, quantize(arr, Q.BF16), raw_dtype=Q.BF16, raw_shape=(arr.shape[0], arr.shape[1] * 2)
            )
        else:
            writer.add_tensor(name, arr)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_iterator_hands_over_f32_f16_bf16_with_their_own_dtype(tmp_path):
    rng = np.random.default_rng(14)
    f32 = rng.standard_normal((4, 8)).astype(np.float32)
    f16 = rng.standard_normal((4, 8)).astype(np.float16)
    b16 = rng.standard_normal((4, 8)).astype(np.float32)
    path = tmp_path / "unq.gguf"
    _write_generic(
        path,
        {
            "blk.0.attn_q.weight": (f16, Q.F16),
            "blk.0.attn_k.weight": (b16, Q.BF16),
            "blk.0.attn_v.weight": (f32, Q.F32),
        },
    )
    nm = {
        "blk.0.attn_q.weight": "model.layers.0.self_attn.q_proj.weight",
        "blk.0.attn_k.weight": "model.layers.0.self_attn.k_proj.weight",
        "blk.0.attn_v.weight": "model.layers.0.self_attn.v_proj.weight",
    }
    got = dict(gguf_quant_weights_iterator(str(path), nm))
    q = got["model.layers.0.self_attn.q_proj.qweight"]  # F16 (never was a problem)
    k = got["model.layers.0.self_attn.k_proj.qweight"]  # BF16 (the class)
    v = got["model.layers.0.self_attn.v_proj.weight"]  # F32 stays a plain .weight
    assert (q.dtype, tuple(q.shape)) == (torch.float16, (4, 8))
    assert (k.dtype, tuple(k.shape)) == (torch.bfloat16, (4, 8))
    assert torch.equal(k, torch.from_numpy(b16).to(torch.bfloat16))
    assert (v.dtype, tuple(v.shape)) == (torch.float32, (4, 8))
    # type markers are unchanged
    assert int(got["model.layers.0.self_attn.k_proj.qweight_type"].item()) == int(Q.BF16)


def test_the_stream_names_the_conversion_once_for_the_metal_log(tmp_path, caplog):
    """The metal evidence line: ``G14 GGUF-DENSE-PAYLOAD: N unquantized F16/BF16 tensor(s) ...`` -- once per stream, with the
    count, and absent when nothing was converted."""
    import logging

    rng = np.random.default_rng(17)
    path = tmp_path / "log.gguf"
    _write_generic(
        path,
        {
            "blk.0.attn_k.weight": (rng.standard_normal((4, 8)).astype(np.float32), Q.BF16),
            "blk.0.attn_q.weight": (rng.standard_normal((4, 8)).astype(np.float16), Q.F16),
            "blk.0.attn_v.weight": (rng.standard_normal((4, 8)).astype(np.float32), Q.F32),
        },
    )
    nm = {
        "blk.0.attn_k.weight": "model.layers.0.self_attn.k_proj.weight",
        "blk.0.attn_q.weight": "model.layers.0.self_attn.q_proj.weight",
        "blk.0.attn_v.weight": "model.layers.0.self_attn.v_proj.weight",
    }
    with caplog.at_level(logging.INFO, logger=WU.logger.name):
        list(gguf_quant_weights_iterator(str(path), nm))
    lines = [r.getMessage() for r in caplog.records if "G14 GGUF-DENSE-PAYLOAD" in r.getMessage()]
    assert len(lines) == 1 and lines[0].startswith("G14 GGUF-DENSE-PAYLOAD: 2 unquantized F16/BF16 tensor(s)"), lines
    # nothing converted -> no line
    caplog.clear()
    path2 = tmp_path / "log2.gguf"
    _write_generic(path2, {"blk.0.attn_v.weight": (rng.standard_normal((4, 8)).astype(np.float32), Q.F32)})
    with caplog.at_level(logging.INFO, logger=WU.logger.name):
        list(gguf_quant_weights_iterator(str(path2), {"blk.0.attn_v.weight": nm["blk.0.attn_v.weight"]}))
    assert not [r for r in caplog.records if "G14 GGUF-DENSE-PAYLOAD" in r.getMessage()]


def test_a_quantized_tensor_still_arrives_as_raw_block_bytes(tmp_path):
    """Q8_0 must NOT be re-viewed: its payload is block bytes (34 B per 32 elements)."""
    rng = np.random.default_rng(15)
    w = rng.standard_normal((4, 64)).astype(np.float32)
    path = tmp_path / "q8.gguf"
    writer = gguf.GGUFWriter(str(path), "qwen4exp")
    raw = quantize(w, Q.Q8_0)
    writer.add_tensor("blk.0.attn_q.weight", raw, raw_dtype=Q.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    got = dict(
        gguf_quant_weights_iterator(
            str(path), {"blk.0.attn_q.weight": "model.layers.0.self_attn.q_proj.weight"}
        )
    )
    q = got["model.layers.0.self_attn.q_proj.qweight"]
    assert q.dtype == torch.uint8 and tuple(q.shape) == (4, 2 * 34)
    assert np.array_equal(q.numpy(), raw)


def test_merged_bf16_shard_beside_a_q8_0_sibling_is_normalised_and_applies(tmp_path):
    """The merged-module case of the same class (task #64 'mixed-dtype qkvz'): a BF16 shard next to a Q8_0 one is cast
    once into the flat container and ``apply`` concatenates both. With raw bytes the BF16 shard stayed uint8."""
    rng = np.random.default_rng(16)
    h = 64
    w_a = rng.standard_normal((12, h)).astype(np.float32)  # BF16 shard
    w_b = rng.standard_normal((6, h)).astype(np.float32)  # Q8_0 shard
    path = tmp_path / "merged.gguf"
    writer = gguf.GGUFWriter(str(path), "qwen4exp")
    writer.add_tensor("a.weight", quantize(w_a, Q.BF16), raw_dtype=Q.BF16, raw_shape=(12, h * 2))
    writer.add_tensor("b.weight", quantize(w_b, Q.Q8_0), raw_dtype=Q.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    stream = dict(
        gguf_quant_weights_iterator(
            str(path), {"a.weight": "gate_up.0.weight", "b.weight": "gate_up.1.weight"}
        )
    )
    a = stream["gate_up.0.qweight"]
    b = stream["gate_up.1.qweight"]
    assert a.dtype == torch.bfloat16 and tuple(a.shape) == (12, h)
    # The TP-group dependent weight_loader is not needed for what is under test (load-time normalisation + apply):
    # fill the shard container exactly as MergedColumnParallelLinear.weight_loader's GGUF branch does
    # (linear.py: shard_id.append / shard_id_map / data_container.append / shard_weight_type).
    layer = torch.nn.Module()
    layer.prefix = "gate_up"
    method = G.GGUFLinearMethod(G.GGUFConfig([]))
    method.create_weights(layer, h, [12, 6], h, 18, torch.bfloat16)
    layer.quant_method = method
    for shard, key in ((0, "gate_up.0"), (1, "gate_up.1")):
        layer.qweight.shard_id.append(shard)
        layer.qweight.shard_id_map[shard] = len(layer.qweight.data_container)
        layer.qweight.data_container.append(stream[key + ".qweight"])
        layer.qweight_type.shard_weight_type[shard] = int(stream[key + ".qweight_type"].item())
    method.process_weights_after_loading(layer)
    view_a = layer._gguf_shard_views[0]
    assert view_a.dtype == torch.bfloat16 and tuple(view_a.shape) == (12, h)
    assert torch.equal(view_a, torch.from_numpy(w_a).to(torch.bfloat16))
    assert layer.qweight_type.shard_weight_type[0] == int(Q.BF16)
    # the Q8_0 sibling needs the CUDA ggml kernel; the BF16 shard goes through the same dispatch ``apply`` uses
    x = torch.randn(3, h).to(torch.bfloat16)
    got = G.fused_mul_mat_gguf(x, view_a, layer.qweight_type.shard_weight_type[0])
    assert torch.equal(got, x @ torch.from_numpy(w_a).to(torch.bfloat16).T)
    # the Q8_0 sibling's shard stays packed block bytes in the same flat container
    assert layer._gguf_shard_views[1].dtype == torch.uint8
    assert b.dtype == torch.uint8


# ---------------------------------------------------------------------------
# row-parallel uneven-TP sharding reads the input dim of a typed tensor in elements
# ---------------------------------------------------------------------------


def test_row_parallel_input_units_are_elements_for_a_typed_unquantized_qweight():
    from sglang.srt.layers.linear import _gguf_input_dim_units

    # BF16 as VALUES: one element per unit (GGML_QUANT_SIZES[BF16] is (1, 2): bytes -> would halve the width)
    assert _gguf_input_dim_units(Q.BF16, torch.zeros(4, 8, dtype=torch.bfloat16)) == (1, 1)
    assert _gguf_input_dim_units(Q.F16, torch.zeros(4, 8, dtype=torch.float16)) == (1, 1)
    # raw bytes of a quantized type keep the block geometry
    assert _gguf_input_dim_units(Q.Q8_0, torch.zeros(4, 68, dtype=torch.uint8)) == tuple(
        gguf.GGML_QUANT_SIZES[Q.Q8_0]
    )
    # a (legacy) raw-byte BF16 payload is still read as bytes
    assert _gguf_input_dim_units(Q.BF16, torch.zeros(4, 16, dtype=torch.uint8)) == tuple(
        gguf.GGML_QUANT_SIZES[Q.BF16]
    )


# ---------------------------------------------------------------------------
# sweep over the PRODUCTION GGUF header (unquantized tensors only; no expert / quantized payload is read)
# ---------------------------------------------------------------------------

REAL_DIR = (
    "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-Flash-Next-GGUF-unsloth/UD-IQ4_XS"
)
REAL_PART1 = os.path.join(REAL_DIR, "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf")
REAL_MTP = (
    "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-Flash-Next-GGUF-unsloth/MTP/"
    "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
)
REAL_CFG = (
    "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-Flash-Next-GGUF-unsloth-sibling/config.json"
)


def _real_unquantized_stream(path, *, draft):
    """The real name map + iterator + qwen4exp stream over ONLY the F32/F16/BF16 tensors of ``path``.

    The reader is narrowed to the unquantized tensors (their payload is small), so the 87 GiB quantized bulk is not
    touched; the split-count guard is lifted for the same reason."""
    import json
    import types

    from sglang.srt.model_loader import gguf_shards as GS

    cfgd = json.load(open(REAL_CFG))
    td = dict(cfgd["text_config"])
    if draft:
        td["num_hidden_layers"] = 1
    text = types.SimpleNamespace(**td)
    cfg = types.SimpleNamespace(
        model_type="qwen4_exp",
        architectures=["Qwen4ExpForCausalLMMTP"] if draft else ["Qwen4ExpForConditionalGeneration"],
        text_config=text,
        vision_config=object(),
    )
    cfg.get_text_config = lambda: text
    adapter = Q4.Qwen4ExpGGUFAdapter(cfg, path)
    nm = adapter.build_name_map()
    unq = {Q.F32, Q.F16, Q.BF16}
    real_reader = gguf.GGUFReader

    class _Narrow(real_reader):
        @property
        def tensors(self):
            return self._narrow

        @tensors.setter
        def tensors(self, v):
            self._narrow = [t for t in v if t.tensor_type in unq]

    saved_reader, saved_count = gguf.GGUFReader, GS.declared_tensor_count
    gguf.GGUFReader = _Narrow
    GS.declared_tensor_count = lambda *a, **k: None
    try:
        stream = adapter.transform_stream(gguf_quant_weights_iterator(path, adapter.stream_name_map(nm)))
        return [(n, w) for n, w in stream if "ngram_embedding.gguf_table" not in n]
    finally:
        gguf.GGUFReader, GS.declared_tensor_count = saved_reader, saved_count


@pytest.mark.skipif(not os.path.exists(REAL_PART1), reason="production GGUF not on this box")
@pytest.mark.parametrize("draft", [False, True], ids=["main", "mtp-draft"])
def test_production_gguf_no_unquantized_tensor_arrives_as_raw_bytes(draft):
    """Sweep: of every tensor the production file stores UNQUANTIZED (557 F32 + 24 BF16 in the main file, 11 + 2 in the
    shared MTP head), none may reach the model as a ``uint8`` payload; every ``.qweight`` among them is bf16 with the
    element shape."""
    path = REAL_MTP if draft else REAL_PART1
    if not os.path.exists(path):
        pytest.skip("MTP head not on this box")
    items = _real_unquantized_stream(path, draft=draft)
    assert items, "the sweep produced nothing"
    raw = [(n, tuple(w.shape)) for n, w in items if w.dtype == torch.uint8]
    assert raw == [], f"unquantized tensors that still arrive as raw bytes: {raw[:5]}"
    q = {n: w for n, w in items if n.endswith(".qweight")}
    n_idx = 1 if draft else 12  # one QSA indexer per full-attention layer (12 in the main file, 1 in the MTP head)
    assert len(q) == n_idx, sorted(q)
    for n, w in q.items():
        assert n.endswith("indexer.index_qk_proj.qweight")
        assert w.dtype == torch.bfloat16 and tuple(w.shape) == (640, 2560), (n, w.dtype, tuple(w.shape))
    # F32 stays a plain ``.weight`` of a floating dtype; the PLE hash constants are int64 buffers
    for n, w in items:
        if n.endswith(".weight"):
            assert w.is_floating_point(), (n, w.dtype)
