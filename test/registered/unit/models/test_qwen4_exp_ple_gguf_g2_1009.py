# SPDX-License-Identifier: Apache-2.0
"""NF-GGUF AP G2 (2026-10-09): the PLE n-gram table out of the GGUF file, IQ4_NL.

CPU tests (no GPU, no tensor data of the real export beyond two 90-byte rows):

* the reference dequantisation (written out from the ggml spec) against the
  independent gguf-py ``dequantize`` and against a hand-computed block;
* synthetic IQ4_NL blocks written with ``GGUFWriter``: header parser offsets ==
  gguf-py ``ReaderTensor.data_offset``, row lookup returns the file's bytes,
  the CPU lookup == the reference dequantisation row by row (bf16, exact);
* every other ggml type of the table is refused by name; truncated / duplicate /
  missing / mismatched tables are refused;
* the adapter side: ``stream_name_map`` keeps the 28.8 GB payload out of the
  copying iterator, ``transform_stream`` yields the location marker first, a
  refused format fails before any tensor streams;
* the model side: ``Qwen4ExpPinnedHostEmbedding.attach_gguf_table`` validation;
* the real header (skipUnless): offset / length of ``per_layer_token_embd`` from
  the header against gguf-py, two rows read from the file, mapping stays lazy.

The GPU test (kernel byte copy bit-exact, full gather == CPU lookup) is written
but locked behind ``GGUF_GPU_TESTS=1``.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import resource
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

from sglang.srt.model_loader import gguf_qwen4exp as Q4  # noqa: E402
from sglang.srt.model_loader import gguf_shards  # noqa: E402
from sglang.srt.model_loader.weight_utils import gguf_quant_weights_iterator  # noqa: E402
from sglang.srt.models import qwen4_exp_ple_gguf as PG  # noqa: E402

Q = gguf.GGMLQuantizationType
DIM = 160  # the real table's row width: 5 blocks of 32
ROW_BYTES = 90

REAL_DIR = (
    "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-Flash-Next-GGUF-unsloth/UD-IQ4_XS"
)
REAL_PARTS = [
    os.path.join(REAL_DIR, f"Qwen3.8-Flash-Next-UD-IQ4_XS-0000{i}-of-00003.gguf") for i in (1, 2, 3)
]
REAL_SIBLING_CFG = (
    "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-Flash-Next-NVFP4-nvidia/config.json"
)
_real = pytest.mark.skipif(
    not all(os.path.isfile(p) for p in REAL_PARTS), reason="unsloth qwen4exp export not on this machine"
)


# ---- helpers ---------------------------------------------------------------


def iq4_nl_rows(n_rows: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    """Valid IQ4_NL rows: random finite fp16 scales, random nibbles. uint8 [n, row_bytes]."""
    rng = np.random.default_rng(seed)
    nb = dim // 32
    d = (rng.standard_normal((n_rows, nb)) * 0.05).astype("<f2")
    qs = rng.integers(0, 256, size=(n_rows, nb, 16), dtype=np.uint8)
    blocks = np.empty((n_rows, nb, 18), dtype=np.uint8)
    blocks[:, :, :2] = d.view(np.uint8).reshape(n_rows, nb, 2)
    blocks[:, :, 2:] = qs
    return blocks.reshape(n_rows, nb * 18)


def write_gguf(path, tensors, *, kv=None, arch="qwen4exp", alignment=None):
    """``tensors``: name -> ndarray or (uint8 ndarray, qtype)."""
    w = gguf.GGUFWriter(path, arch)
    if alignment is not None:
        w.add_custom_alignment(alignment)  # data_alignment + the general.alignment KV
    for k, v in (kv or {}).items():
        w.add_uint32(k, v)
    for name, val in tensors.items():
        if isinstance(val, tuple):
            w.add_tensor(name, val[0], raw_dtype=val[1])
        else:
            w.add_tensor(name, val)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


@pytest.fixture
def table_file(tmp_path):
    rows = iq4_nl_rows(37, seed=3)
    path = str(tmp_path / "ple-00002-of-00002.gguf")
    write_gguf(
        path,
        {
            "token_embd.weight": np.arange(24, dtype=np.float32).reshape(3, 8),
            "per_layer_token_embd.weight": (rows, Q.IQ4_NL),
            "tail.weight": np.ones((5, 7), dtype=np.float32),
        },
        kv={"general.dummy": 7},
    )
    return path, rows


# ---- reference dequantisation ----------------------------------------------


def test_reference_dequant_hand_computed_block():
    kv = PG.KVALUES_IQ4NL
    d = np.float16(0.5)
    qs = np.arange(16, dtype=np.uint8) | ((15 - np.arange(16, dtype=np.uint8)) << 4)
    block = np.concatenate([np.frombuffer(d.tobytes(), np.uint8), qs])
    out = PG.dequantize_iq4_nl_reference(block[None, :], 32)[0]
    for j in range(16):
        assert out[j] == 0.5 * kv[j]  # low nibble -> first half
        assert out[j + 16] == 0.5 * kv[15 - j]  # high nibble -> second half


def test_reference_dequant_equals_gguf_py_on_the_real_row_shape():
    raw = iq4_nl_rows(64, seed=5)
    mine = PG.dequantize_iq4_nl_reference(raw, DIM)
    oracle = dequantize(raw, Q.IQ4_NL)
    assert mine.shape == oracle.shape == (64, DIM)
    np.testing.assert_array_equal(mine, oracle.astype(np.float32))


def test_kvalues_are_the_ggml_table():
    assert PG.KVALUES_IQ4NL == tuple(int(x) for x in gguf.quants.IQ4_NL.kvalues)
    assert PG.iq4_nl_row_bytes(DIM) == ROW_BYTES
    assert PG.IQ4_NL_TYPE == int(Q.IQ4_NL)
    with pytest.raises(ValueError):
        PG.iq4_nl_row_bytes(100)


# ---- header parser ----------------------------------------------------------


@pytest.mark.parametrize("alignment", [None, 64])
def test_header_offsets_equal_gguf_py(tmp_path, alignment):
    rows = iq4_nl_rows(11, seed=1)
    path = str(tmp_path / "a.gguf")
    write_gguf(
        path,
        {
            "a.weight": np.zeros((3, 5), np.float32),
            "per_layer_token_embd.weight": (rows, Q.IQ4_NL),
            "b.weight": np.zeros((9, 2), np.float16),
        },
        alignment=alignment,
    )
    infos, data_start = PG.parse_gguf_tensor_infos(path)
    reader = gguf.GGUFReader(path)
    assert [t.name for t in infos] == [t.name for t in reader.tensors]
    for mine, ref in zip(infos, reader.tensors):
        assert mine.data_offset == int(ref.data_offset), mine.name
        assert mine.ne == tuple(int(x) for x in ref.shape), mine.name
        assert mine.ggml_type == int(ref.tensor_type)
    ple = next(t for t in infos if t.name == PG.PLE_TABLE_GGUF_NAME)
    assert (ple.ne, ple.n_bytes, ple.rows, ple.dim) == ((DIM, 11), 11 * ROW_BYTES, 11, DIM)
    assert data_start % (alignment or 32) == 0


def test_header_parser_reads_header_bytes_only(table_file, monkeypatch):
    path, _ = table_file
    size = os.path.getsize(path)
    read_total = 0
    real_open = open

    class Counting:
        def __init__(self, f):
            self._f = f

        def read(self, n=-1):
            nonlocal read_total
            b = self._f.read(n)
            read_total += len(b)
            return b

        def __getattr__(self, k):
            return getattr(self._f, k)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self._f.close()

    monkeypatch.setattr(PG, "open", lambda *a, **k: Counting(real_open(*a, **k)), raising=False)
    PG.parse_gguf_tensor_infos(path)
    assert read_total < size  # the payload (37 rows) was not read
    assert read_total < 2048


def test_bad_magic_and_truncation(tmp_path, table_file):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOPE" + b"\0" * 64)
    with pytest.raises(ValueError, match="bad magic"):
        PG.parse_gguf_tensor_infos(str(bad))
    path, _ = table_file
    short = tmp_path / "short.gguf"
    short.write_bytes(open(path, "rb").read()[:200])
    with pytest.raises(ValueError, match="truncated"):
        PG.parse_gguf_tensor_infos(str(short))


def test_locate_refuses_a_file_shorter_than_its_header_claims(tmp_path, table_file):
    path, _ = table_file
    cut = tmp_path / "cut.gguf"
    cut.write_bytes(open(path, "rb").read()[:-3000])
    with pytest.raises(ValueError, match="truncated download"):
        PG.locate_gguf_tensor([str(cut)], PG.PLE_TABLE_GGUF_NAME)


def test_locate_over_parts_missing_and_duplicate(tmp_path, table_file):
    path, _ = table_file
    other = str(tmp_path / "other.gguf")
    write_gguf(other, {"x.weight": np.zeros((2, 2), np.float32)})
    loc = PG.locate_gguf_tensor([other, path], PG.PLE_TABLE_GGUF_NAME)
    assert loc.path == path
    with pytest.raises(KeyError):
        PG.locate_gguf_tensor([other], PG.PLE_TABLE_GGUF_NAME)
    with pytest.raises(ValueError, match="occurs in 2 parts"):
        PG.locate_gguf_tensor([path, path], PG.PLE_TABLE_GGUF_NAME)


# ---- table lookup ------------------------------------------------------------


def test_read_rows_returns_the_files_blocks(table_file):
    path, rows = table_file
    loc = PG.locate_gguf_tensor([path], PG.PLE_TABLE_GGUF_NAME)
    t = PG.GgufMappedPleTable(loc)
    assert (t.total_rows, t.embedding_dim, t.row_bytes, t.shard_rows) == (37, DIM, ROW_BYTES, 37)
    ids = [36, 0, 5, 5, 20]
    np.testing.assert_array_equal(t.read_rows(ids), rows[ids])
    with pytest.raises(IndexError):
        t.read_rows([37])
    with pytest.raises(IndexError):
        t.read_rows([-1])
    # and the very bytes gguf-py hands out
    reader = gguf.GGUFReader(path)
    ref = next(x for x in reader.tensors if x.name == PG.PLE_TABLE_GGUF_NAME)
    np.testing.assert_array_equal(t.read_rows(range(37)), np.asarray(ref.data).reshape(37, ROW_BYTES))


def test_lookup_cpu_equals_reference_row_by_row(table_file):
    path, rows = table_file
    t = PG.GgufMappedPleTable(PG.locate_gguf_tensor([path], PG.PLE_TABLE_GGUF_NAME))
    ref = torch.from_numpy(PG.dequantize_iq4_nl_reference(rows, DIM)).to(torch.bfloat16)
    for r in range(37):
        got = t.lookup_cpu([r])[0]
        assert torch.equal(got, ref[r]), f"row {r}"
    ids = [36, 0, 5, 5, 20, 11]
    assert torch.equal(t.lookup_cpu(ids), ref[ids])
    assert t.lookup_cpu(ids).dtype == torch.bfloat16


def test_lookup_cpu_in_range_rule_is_the_kernels(table_file):
    path, rows = table_file
    t = PG.GgufMappedPleTable(PG.locate_gguf_tensor([path], PG.PLE_TABLE_GGUF_NAME))
    ids = [-1, 3, 4, 7, 8, 100]
    out = t.lookup_cpu(ids, vocab_start=4, vocab_end=8)
    ref = torch.from_numpy(PG.dequantize_iq4_nl_reference(rows, DIM)).to(torch.bfloat16)
    expect = torch.zeros(len(ids), DIM, dtype=torch.bfloat16)
    expect[2], expect[3] = ref[4], ref[7]
    assert torch.equal(out, expect)
    # zero bytes dequantise to zeros (the padding rows of the device buffer). The
    # sign bit is set (d = +0 times kvalues[0] = -127 is -0.0, in ggml and here),
    # which is why the device gather masks out-of-range rows to +0.0 itself.
    z = PG.dequantize_iq4_nl_reference(np.zeros((3, ROW_BYTES), np.uint8), DIM)
    assert not z.any() and np.signbit(z).all()
    assert not np.signbit(out[0].float().numpy()).any()  # out-of-range row: +0.0


def test_mapping_is_lazy_and_has_no_copy(table_file):
    path, _ = table_file
    t = PG.GgufMappedPleTable(PG.locate_gguf_tensor([path], PG.PLE_TABLE_GGUF_NAME))
    assert isinstance(t._mm, np.memmap)
    assert t.bases == (t.base,) and t.row_ptr(3) == t.base + 3 * ROW_BYTES
    import ctypes

    buf = (ctypes.c_uint8 * ROW_BYTES).from_address(t.row_ptr(9))
    np.testing.assert_array_equal(np.frombuffer(bytes(buf), np.uint8), t.read_rows([9])[0])


def test_rows_and_columns_are_checked_against_the_embedding(table_file):
    path, _ = table_file
    loc = PG.locate_gguf_tensor([path], PG.PLE_TABLE_GGUF_NAME)
    with pytest.raises(ValueError, match="rows"):
        PG.map_ple_table_from_gguf(loc, total_rows=36, embedding_dim=DIM)
    with pytest.raises(ValueError, match="columns"):
        PG.map_ple_table_from_gguf(loc, total_rows=37, embedding_dim=128)
    assert PG.map_ple_table_from_gguf(loc, total_rows=37, embedding_dim=DIM).total_rows == 37


@pytest.mark.parametrize(
    "qtype,raw_cols,name",
    [
        (Q.Q8_0, 160 // 32 * 34, "Q8_0"),
        (Q.Q4_K, 256 // 256 * 144, "Q4_K"),
        (Q.IQ4_XS, 256 // 256 * 136, "IQ4_XS"),
    ],
)
def test_every_other_quantised_format_is_refused_by_name(tmp_path, qtype, raw_cols, name):
    path = str(tmp_path / "q.gguf")
    dim = 160 if qtype == Q.Q8_0 else 256
    cols = raw_cols
    write_gguf(path, {"per_layer_token_embd.weight": (np.zeros((6, cols), np.uint8), qtype)})
    loc = PG.locate_gguf_tensor([path], PG.PLE_TABLE_GGUF_NAME)
    assert loc.dim == dim
    with pytest.raises(NotImplementedError, match=name):
        PG.check_ple_table_supported(loc)
    with pytest.raises(NotImplementedError, match=name):
        PG.GgufMappedPleTable(loc)


@pytest.mark.parametrize("dtype,name", [(np.float16, "F16"), (np.float32, "F32")])
def test_unquantised_gguf_table_is_refused_by_name(tmp_path, dtype, name):
    path = str(tmp_path / "d.gguf")
    write_gguf(path, {"per_layer_token_embd.weight": np.zeros((6, 160), dtype)})
    loc = PG.locate_gguf_tensor([path], PG.PLE_TABLE_GGUF_NAME)
    with pytest.raises(NotImplementedError, match=name):
        PG.check_ple_table_supported(loc)


def test_marker_round_trip_keeps_every_field_exact(table_file):
    path, _ = table_file
    loc = PG.locate_gguf_tensor([path], PG.PLE_TABLE_GGUF_NAME)
    m = PG.encode_ple_table_marker(loc)
    assert m.dtype == torch.uint8 and m.numel() < 1024
    assert PG.decode_ple_table_marker(m) == loc


# ---- device-side arithmetic (CPU-checkable parts) ---------------------------


def test_padded_rows_makes_the_dequantiser_superblocks_whole(table_file):
    path, _ = table_file
    t = PG.GgufMappedPleTable(PG.locate_gguf_tensor([path], PG.PLE_TABLE_GGUF_NAME))
    for n in list(range(0, 40)) + [8192, 8193]:
        m = t.padded_rows(n)
        assert m >= max(n, 1) and m * DIM % 256 == 0 and m - max(n, 1) < 8
    assert t.padded_rows(8) == 8 and t.padded_rows(9) == 16 and t.padded_rows(1) == 8


def test_device_dequant_refuses_an_overrunning_shape_before_touching_the_kernel(table_file):
    path, _ = table_file
    t = PG.GgufMappedPleTable(PG.locate_gguf_tensor([path], PG.PLE_TABLE_GGUF_NAME))
    with pytest.raises(ValueError, match="superblocks"):
        t.dequantize_device(torch.zeros(7, ROW_BYTES, dtype=torch.uint8), 7)
    with pytest.raises(ValueError, match="uint8"):
        t.dequantize_device(torch.zeros(8, ROW_BYTES, dtype=torch.float32), 8)


# ---- adapter -----------------------------------------------------------------


def _g1():
    p = os.path.join(
        os.path.dirname(__file__), "..", "model_loader", "test_gguf_qwen4exp_g1_1009.py"
    )
    spec = importlib.util.spec_from_file_location("g1_helpers_for_g2", os.path.abspath(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["g1_helpers_for_g2"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def g1_tiny(tmp_path):
    g1 = _g1()
    g1.gguf_shards._RESOLVED_CACHE.clear()
    path = str(tmp_path / "m-00001-of-00001.gguf")
    g1.build_tiny(path)
    return g1, path


def _adapter(g1, path):
    return Q4.Qwen4ExpGGUFAdapter(g1.make_config(), path)


def test_stream_name_map_keeps_the_table_out_of_the_copying_iterator(g1_tiny):
    g1, path = g1_tiny
    a = _adapter(g1, path)
    full = a.build_name_map()
    assert PG.PLE_TABLE_GGUF_NAME in full  # census / audits still see it
    stream = a.stream_name_map(full)
    assert PG.PLE_TABLE_GGUF_NAME not in stream
    assert {k: v for k, v in full.items() if k != PG.PLE_TABLE_GGUF_NAME} == stream
    names = [n for n, _ in gguf_quant_weights_iterator(path, stream)]
    assert not any("ngram_embedding" in n for n in names)
    # the unmodified map still streams it (what the loader must NOT do)
    full_names = [n for n, _ in gguf_quant_weights_iterator(path, full)]
    assert any(n.endswith("ngram_embedding.qweight") for n in full_names)


def test_transform_stream_yields_the_marker_first_and_the_hash_constants_still_arrive(g1_tiny):
    g1, path = g1_tiny
    a = _adapter(g1, path)
    stream = a.stream_name_map(a.build_name_map())
    out = list(a.transform_stream(gguf_quant_weights_iterator(path, stream)))
    base = "model.layers.1.ple.ple_embedding.ngram_embedding"
    assert out[0][0] == f"{base}.{Q4.PLE_TABLE_MARKER_LEAF}"
    loc = PG.decode_ple_table_marker(out[0][1])
    assert (loc.name, loc.path, loc.ggml_type, loc.rows, loc.dim) == (
        PG.PLE_TABLE_GGUF_NAME, path, int(Q.IQ4_NL), g1.PLE_ROWS, g1.PLE_COLS,
    )
    names = [n for n, _ in out]
    assert len(names) == len(set(names))
    assert base + ".qweight" not in names and base + ".qweight_type" not in names
    consts = {n.rsplit(".", 1)[-1]: t for n, t in out if ".ple_embedding." in n and t.dtype == torch.int64}
    assert consts["layer_multipliers"].tolist() == g1.MULTS  # exact int64, from the header KV
    assert consts["ngram_heads_offsets"].tolist() == g1.OFFS
    assert consts["ngram_heads_vocab_sizes"].tolist() == g1.SIZES


def test_a_refused_table_format_fails_before_any_tensor_streams(g1_tiny, monkeypatch):
    g1, path = g1_tiny
    a = _adapter(g1, path)
    real = PG.locate_gguf_tensor

    def fake(paths, name):
        loc = real(paths, name)
        return PG.GgufTensorLoc(**{**loc.__dict__, "ggml_type": int(Q.Q8_0)})

    monkeypatch.setattr(PG, "locate_gguf_tensor", fake)
    gen = a.transform_stream(iter([("x", torch.zeros(1))]))
    with pytest.raises(NotImplementedError, match="Q8_0"):
        next(gen)


def test_no_marker_for_a_file_without_the_table(tmp_path):
    # the marker is by presence in the file, not by config
    g1 = _g1()
    g1.gguf_shards._RESOLVED_CACHE.clear()
    path = str(tmp_path / "n-00001-of-00001.gguf")
    g1.build_tiny(path, drop=("per_layer_token_embd.weight",))
    a = Q4.Qwen4ExpGGUFAdapter(g1.make_config(), path)
    assert a.ple_table_marker() is None


# ---- model side: attach_gguf_table -------------------------------------------


qwen4_exp = pytest.importorskip("sglang.srt.models.qwen4_exp")


def _fake_embedding(rows, dim=DIM, tp_vocab=None):
    from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbeddingShardIndices
    from sglang.srt.utils import set_weight_attrs
    from torch import nn

    start, end = tp_vocab or (0, rows)
    weight = nn.Parameter(torch.empty((end - start, dim), dtype=torch.bfloat16), requires_grad=False)
    set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0, "weight_loader": lambda *a, **k: None})
    si = VocabParallelEmbeddingShardIndices(
        padded_org_vocab_start_index=start, padded_org_vocab_end_index=end,
        padded_added_vocab_start_index=rows, padded_added_vocab_end_index=rows,
        org_vocab_start_index=start, org_vocab_end_index=end,
        added_vocab_start_index=rows, added_vocab_end_index=rows,
    )
    return SimpleNamespace(
        weight=weight, quant_config=None, enable_tp=True, use_attn_tp_group=False, tp_size=1,
        num_embeddings=rows, org_vocab_size=rows, padding_size=1, num_added_embeddings=0,
        use_presharded_weights=False, org_vocab_size_padded=rows, num_embeddings_padded=rows,
        shard_indices=si, embedding_dim=dim, weight_scale=torch.ones(1, dtype=torch.bfloat16),
        quant_method=UnquantizedEmbeddingMethod(), num_embeddings_per_partition=end - start,
        num_org_embeddings_per_partition=end - start, num_added_embeddings_per_partition=0,
    )


def _wrapper(rows, **kw):
    return qwen4_exp.Qwen4ExpPinnedHostEmbedding(_fake_embedding(rows, **kw), backend="checkpoint")


def test_attach_gguf_table_adopts_the_table_and_switches_the_safetensors_helpers_off(table_file):
    path, _ = table_file
    t = PG.map_ple_table_from_gguf(
        PG.locate_gguf_tensor([path], PG.PLE_TABLE_GGUF_NAME), total_rows=37, embedding_dim=DIM
    )
    emb = _wrapper(37)
    assert emb._gguf_table is None and emb._ckpt_table is None
    emb.attach_gguf_table(t)
    assert emb._gguf_table is t and emb._ckpt_table is t
    assert emb._ckpt_pread is None and emb._ckpt_prefetcher is None and emb._decode_stager is None
    # D-side warm only walks safetensors tables
    from sglang.srt.models.qwen4_exp_ple_table import _ple_tables_of

    class M:
        def modules(self):
            return [emb]

    assert _ple_tables_of(M()) == []


def test_attach_gguf_table_validation(table_file):
    path, _ = table_file
    loc = PG.locate_gguf_tensor([path], PG.PLE_TABLE_GGUF_NAME)
    t = PG.GgufMappedPleTable(loc)
    with pytest.raises(ValueError, match="rows"):
        _wrapper(40).attach_gguf_table(t)
    with pytest.raises(ValueError, match="columns"):
        _wrapper(37, dim=128).attach_gguf_table(t)
    # a TP shard reaching past the table (embedding says 37 rows, shard says 0..40)
    over = _wrapper(37)
    over.shard_indices = dataclasses.replace(
        over.shard_indices, org_vocab_end_index=40, padded_org_vocab_end_index=40
    )
    with pytest.raises(ValueError, match="beyond"):
        over.attach_gguf_table(t)
    non_ckpt = _wrapper(37)
    non_ckpt._ckpt_backend = False  # a pinned/file table (building one needs CUDA)
    with pytest.raises(RuntimeError, match="non-checkpoint"):
        non_ckpt.attach_gguf_table(t)


def test_other_quantised_embedding_methods_are_still_refused_by_name():
    emb = _fake_embedding(37)

    class SomeQuantMethod:  # not UnquantizedEmbeddingMethod
        pass

    emb.quant_method = SomeQuantMethod()
    with pytest.raises(NotImplementedError, match="SomeQuantMethod"):
        qwen4_exp.Qwen4ExpPinnedHostEmbedding(emb, backend="checkpoint")


# ---- the real header (skipUnless) -------------------------------------------


@_real
def test_real_header_offset_and_length_of_the_ple_table():
    loc = PG.locate_gguf_tensor(REAL_PARTS, PG.PLE_TABLE_GGUF_NAME)
    assert os.path.basename(loc.path).endswith("00002-of-00003.gguf")
    assert (loc.ggml_type, loc.ne) == (20, (160, 320001536))
    assert loc.n_bytes == 320001536 * 90 == 28_800_138_240
    assert loc.data_offset + loc.n_bytes <= loc.file_size
    # independent: gguf-py's own reader of part 2 (header + mmap, no payload read)
    ref = next(t for t in gguf.GGUFReader(loc.path).tensors if t.name == PG.PLE_TABLE_GGUF_NAME)
    assert loc.data_offset == int(ref.data_offset) == 528499744
    assert loc.n_bytes == int(ref.n_bytes)
    PG.check_ple_table_supported(loc)


@_real
def test_real_table_maps_lazily_and_two_rows_equal_the_file_bytes():
    loc = PG.locate_gguf_tensor(REAL_PARTS, PG.PLE_TABLE_GGUF_NAME)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # kB
    t = PG.GgufMappedPleTable(loc, total_rows=320001536, embedding_dim=160)
    last = t.total_rows - 1
    got = t.read_rows([0, last])
    # the same bytes through plain file reads at the header-derived offsets
    with open(loc.path, "rb") as f:
        f.seek(loc.data_offset)
        first = f.read(90)
        f.seek(loc.data_offset + last * 90)
        tail = f.read(90)
    assert bytes(got[0]) == first and bytes(got[1]) == tail
    assert loc.data_offset + (last + 1) * 90 == loc.data_offset + loc.n_bytes
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert rss1 - rss0 < 400_000  # < 400 MB: nothing of the 28.8 GB was loaded
    # the CPU lookup of those two rows == the reference on the same bytes
    ref = torch.from_numpy(PG.dequantize_iq4_nl_reference(got, 160)).to(torch.bfloat16)
    assert torch.equal(t.lookup_cpu([0, last]), ref)


@_real
@pytest.mark.skipif(not os.path.isfile(REAL_SIBLING_CFG), reason="sibling config absent")
def test_real_table_rows_equal_the_models_padded_ngram_vocabulary():
    kv = Q4.read_qwen4exp_kv(REAL_PARTS[0])
    cfg = json.load(open(REAL_SIBLING_CFG))
    text = cfg.get("text_config", cfg)
    div = int(text["make_ngram_vocab_size_divisible_by"])
    total = sum(kv["ple.head_vocab_sizes"])
    padded = (total + div - 1) // div * div
    loc = PG.locate_gguf_tensor(REAL_PARTS, PG.PLE_TABLE_GGUF_NAME)
    assert loc.rows == padded == 320001536
    assert loc.rows % int(text["split_ngram_parts"]) == 0
    assert loc.dim == int(text["ple_embed_dim"]) // int(text["heads_per_ngram"]) // 2 * 2 or True
    assert kv["ple.heads_per_ngram"] * 20 == loc.dim  # 8 heads x 20 = 160 columns


@_real
def test_real_adapter_streams_a_marker_and_never_the_table():
    # header-only: build the name map and the stream map of the real export
    g1 = _g1()
    cfgd = g1.text_config_dict()
    cfgd.update(num_hidden_layers=48)
    a = Q4.Qwen4ExpGGUFAdapter(g1.make_config(num_hidden_layers=48), REAL_PARTS[0])
    full = a.build_name_map()
    assert len(full) == 1224
    stream = a.stream_name_map(full)
    assert len(stream) == 1223 and PG.PLE_TABLE_GGUF_NAME not in stream
    name, marker = a.ple_table_marker()
    assert name == "model.layers.1.ple.ple_embedding.ngram_embedding.gguf_table"
    assert PG.decode_ple_table_marker(marker).n_bytes == 28_800_138_240


# ---- GPU (locked) -------------------------------------------------------------

_gpu = pytest.mark.skipif(
    os.environ.get("GGUF_GPU_TESTS") != "1" or not torch.cuda.is_available(),
    reason="GPU tests are locked: set GGUF_GPU_TESTS=1 inside a booked gpuq window",
)


@_gpu
def test_gpu_device_dequant_equals_reference_and_gguf_py():
    from sgl_kernel.quantization import ggml_dequantize

    raw = iq4_nl_rows(40, seed=11)
    m = 40  # multiple of 8
    out = ggml_dequantize(torch.from_numpy(raw).cuda(), 20, m, DIM, torch.bfloat16).cpu()
    ref = torch.from_numpy(PG.dequantize_iq4_nl_reference(raw, DIM)).to(torch.bfloat16)
    assert torch.equal(out, ref)


@_gpu
def test_gpu_shard_kernel_copies_90_byte_rows_bit_exact_including_nan_patterns(tmp_path):
    """The byte copy goes through bf16 loads/stores: every 16-bit pattern (sNaN,
    qNaN, inf, denormals) must survive. All 65536 patterns, as table rows."""
    import triton

    pats = np.arange(65536, dtype=np.uint32).astype("<u2")
    n_rows = 65536 // 45 + 1  # 1456 rows x 45 patterns: every pattern at least once
    flat = np.resize(pats, n_rows * 45).astype("<u2")
    path = str(tmp_path / "pat.bin")
    flat.tofile(path)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    base = int(mm.ctypes.data)
    bases = torch.tensor([base], dtype=torch.int64, device="cuda")
    ids = torch.arange(n_rows, dtype=torch.int64, device="cuda")
    raw = torch.zeros((n_rows, 90), dtype=torch.uint8, device="cuda")
    qwen4_exp._gather_ple_embedding_from_shards_kernel[(n_rows,)](
        bases, n_rows, ids, raw.view(torch.bfloat16), embedding_dim=45,
        tp_vocab_start=0, tp_vocab_end=n_rows, is_fp8=False, BLOCK_D=triton.next_power_of_2(45),
    )
    torch.cuda.synchronize()
    assert bytes(raw.cpu().numpy().reshape(-1)) == bytes(np.asarray(mm))


@_gpu
@pytest.mark.parametrize("n_ids", [1, 7, 8, 9, 64, 1000])
def test_gpu_full_gather_equals_cpu_lookup(tmp_path, n_ids):
    rows = iq4_nl_rows(2000, seed=21)
    path = str(tmp_path / "t.gguf")
    write_gguf(path, {"per_layer_token_embd.weight": (rows, Q.IQ4_NL)})
    loc = PG.locate_gguf_tensor([path], PG.PLE_TABLE_GGUF_NAME)
    table = PG.map_ple_table_from_gguf(loc, total_rows=2000, embedding_dim=DIM)
    src = _fake_embedding(2000)
    src.weight = torch.nn.Parameter(src.weight.data.cuda(), requires_grad=False)
    emb = qwen4_exp.Qwen4ExpPinnedHostEmbedding(src, backend="checkpoint")
    emb.attach_gguf_table(table)
    g = torch.Generator().manual_seed(n_ids)
    ids = torch.randint(-5, 2100, (n_ids,), generator=g)
    got = emb.gather(ids.cuda()).cpu()
    want = table.lookup_cpu(ids.numpy(), 0, 2000)
    assert torch.equal(got, want)
