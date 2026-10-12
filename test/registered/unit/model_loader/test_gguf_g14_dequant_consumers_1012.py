# SPDX-License-Identifier: Apache-2.0
"""NF-GGUF G14 fix round 1 (2026-10-12): the CONSUMERS of a ``.qweight`` that dequantize through numpy.

G14 made ``gguf_quant_weights_iterator`` hand an unquantized F16/BF16 ``.qweight`` over as a typed tensor
(``weight_utils.py`` ``gguf_dense_payload``). The first G14 head fixed the producer; review finding 1: consumers that
still did ``dequantize(weight.numpy(), qtype)`` raise ``TypeError: Got unsupported ScalarType BFloat16`` on such a
tensor (torch bfloat16 has no numpy view) where they used to work from the raw uint8 bytes:

* ``gguf_qwen35.Qwen35GGUFAdapter.transform_stream``: the MTP draft's ``mtp.fc`` (``blk.N.nextn.eh_proj``) and, under
  ``SGLANG_GGUF_DENSE_VOCAB=1``, ``model.embed_tokens`` / ``lm_head`` (Qwen3.8-27B-UD-Q8_K_XL.gguf stores BF16
  ``blk.N.nextn.eh_proj.weight`` and BF16 ``output.weight``),
* ``deepseek_v4._gguf_dequantize`` (GGUF compressor projections).

Fix: ``weight_utils.gguf_dequantize_payload`` (unquantized type = the tensor of its own dtype, numpy only for
quantized types). The tests run the REAL producer (GGUFWriter file -> iterator) into the real consumer.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import types

import numpy as np
import pytest
import torch

import gguf
from gguf.quants import dequantize, quantize

from sglang.srt.model_loader import gguf_qwen35 as Q35
from sglang.srt.model_loader import weight_utils as WU
from sglang.srt.model_loader.weight_utils import (
    gguf_dequantize_payload,
    gguf_quant_weights_iterator,
)

Q = gguf.GGMLQuantizationType


def _write(path, tensors):
    """tensors: name -> (float32 array [rows, cols], ggml type)."""
    writer = gguf.GGUFWriter(str(path), "qwen35")
    for name, (w, qt) in tensors.items():
        data = quantize(w, qt)
        if qt in (Q.BF16, Q.F16):
            raw_shape = (w.shape[0], data.shape[-1])
            writer.add_tensor(name, data, raw_dtype=qt, raw_shape=raw_shape)
        else:
            writer.add_tensor(name, data, raw_dtype=qt)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _bf16_ref(w: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(w).to(torch.bfloat16)


def _adapter(is_draft: bool):
    """A qwen35 adapter shell: transform_stream's vocab/draft path needs only these attributes."""
    ad = object.__new__(Q35.Qwen35GGUFAdapter)
    ad.is_draft = is_draft
    ad.config = types.SimpleNamespace(torch_dtype=torch.bfloat16)
    ad.num_k = ad.num_v = ad.head_k_dim = ad.head_v_dim = 0
    ad.mmproj_path = None
    ad.vision_config = None
    return ad


W = np.random.default_rng(7).standard_normal((4, 64)).astype(np.float32)


# --------------------------------------------------------------------------- the helper itself


@pytest.mark.parametrize("qt", [Q.BF16, Q.F16, Q.F32])
def test_helper_unquantized_types_are_their_own_values(tmp_path, qt):
    p = tmp_path / "t.gguf"
    _write(p, {"blk.0.nextn.eh_proj.weight": (W, qt)})
    out = dict(gguf_quant_weights_iterator(str(p), {"blk.0.nextn.eh_proj.weight": "mtp.fc.weight"}))
    if qt == Q.F32:
        got = out["mtp.fc.weight"]
    else:
        got = gguf_dequantize_payload(out["mtp.fc.qweight"], out["mtp.fc.qweight_type"].item())
    assert got.shape == W.shape
    ref = {Q.BF16: torch.from_numpy(W).bfloat16(), Q.F16: torch.from_numpy(W).half(), Q.F32: torch.from_numpy(W)}[qt]
    assert torch.equal(got.float(), ref.float())


def test_helper_accepts_raw_uint8_bytes_of_an_older_producer():
    ref = _bf16_ref(W)
    raw = ref.contiguous().view(torch.uint8)  # [4, 128]
    got = gguf_dequantize_payload(raw, int(Q.BF16))
    assert got.dtype == torch.bfloat16 and torch.equal(got, ref)
    raw32 = torch.from_numpy(W).contiguous().view(torch.uint8)
    assert torch.equal(gguf_dequantize_payload(raw32, int(Q.F32)), torch.from_numpy(W))


def test_helper_quantized_type_still_goes_through_gguf_py(tmp_path):
    w = np.random.default_rng(1).standard_normal((4, 64)).astype(np.float32)
    p = tmp_path / "q.gguf"
    _write(p, {"blk.0.nextn.eh_proj.weight": (w, Q.Q8_0)})
    out = dict(gguf_quant_weights_iterator(str(p), {"blk.0.nextn.eh_proj.weight": "mtp.fc.weight"}))
    q = out["mtp.fc.qweight"]
    assert q.dtype == torch.uint8
    got = gguf_dequantize_payload(q, int(Q.Q8_0))
    assert np.array_equal(got.numpy(), dequantize(q.numpy(), Q.Q8_0))
    assert np.allclose(got.numpy(), w, atol=0.05)


# --------------------------------------------------------------------------- qwen35 consumer


def test_qwen35_draft_mtp_fc_bf16_dequants_from_a_typed_stream(tmp_path):
    p = tmp_path / "d.gguf"
    _write(p, {"blk.0.nextn.eh_proj.weight": (W, Q.BF16)})
    stream = gguf_quant_weights_iterator(str(p), {"blk.0.nextn.eh_proj.weight": "mtp.fc.weight"})
    got = dict(_adapter(True).transform_stream(stream))
    assert set(got) == {"mtp.fc.weight"}
    assert got["mtp.fc.weight"].dtype == torch.bfloat16
    assert torch.equal(got["mtp.fc.weight"], _bf16_ref(W))


def test_qwen35_dense_vocab_bf16_output_and_embed(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_GGUF_DENSE_VOCAB", "1")
    p = tmp_path / "v.gguf"
    emb = np.random.default_rng(3).standard_normal((8, 64)).astype(np.float32)
    _write(p, {"output.weight": (W, Q.BF16), "token_embd.weight": (emb, Q.F16)})
    stream = gguf_quant_weights_iterator(
        str(p), {"output.weight": "lm_head.weight", "token_embd.weight": "model.embed_tokens.weight"}
    )
    got = dict(_adapter(False).transform_stream(stream))
    assert set(got) == {"lm_head.weight", "model.embed_tokens.weight"}
    assert torch.equal(got["lm_head.weight"], _bf16_ref(W))
    assert torch.equal(got["model.embed_tokens.weight"], torch.from_numpy(emb).half().to(torch.bfloat16))


def test_qwen35_draft_mtp_fc_q8_0_unchanged(tmp_path):
    w = np.random.default_rng(1).standard_normal((4, 64)).astype(np.float32)
    p = tmp_path / "q.gguf"
    _write(p, {"blk.0.nextn.eh_proj.weight": (w, Q.Q8_0)})
    stream = gguf_quant_weights_iterator(str(p), {"blk.0.nextn.eh_proj.weight": "mtp.fc.weight"})
    got = dict(_adapter(True).transform_stream(stream))["mtp.fc.weight"]
    assert got.dtype == torch.bfloat16 and np.allclose(got.float().numpy(), w, atol=0.05)


# --------------------------------------------------------------------------- deepseek_v4 consumer


@pytest.mark.parametrize("qt", [Q.BF16, Q.F16, Q.Q8_0])
def test_deepseek_v4_gguf_dequantize(tmp_path, qt):
    dsv4 = pytest.importorskip("sglang.srt.models.deepseek_v4")
    p = tmp_path / "ds.gguf"
    _write(p, {"blk.0.attn_compressor_kv.weight": (W, qt)})
    out = dict(gguf_quant_weights_iterator(str(p), {"blk.0.attn_compressor_kv.weight": "c.wkv.weight"}))
    got = dsv4._gguf_dequantize(out["c.wkv.qweight"], out["c.wkv.qweight_type"], torch.bfloat16)
    assert got.shape == W.shape and got.dtype == torch.bfloat16
    atol = 0.0 if qt == Q.BF16 else 0.05
    ref = torch.from_numpy(W).bfloat16() if qt == Q.BF16 else torch.from_numpy(W)
    assert torch.allclose(got.float(), ref.float(), atol=max(atol, 0.01) if qt != Q.BF16 else 0.0)
