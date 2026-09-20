"""Task #53 Sitz 5 (20.09.): the fused QSA rows resolve + owner compaction
(qsa/rows_resolve.py) against the torch chain it replaces
(_logical_to_physical_graph -> _local_rows -> compact_owned_rows).

The kernel runs on CPU through the Triton interpreter in a subprocess
(TRITON_INTERPRET must be set before triton is imported; the collecting
pytest process may already hold a compiled triton), so the suite stays
hermetic (CUDA_VISIBLE_DEVICES="").
"""

import os
import subprocess
import sys
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "python"))

CHECK = r'''
import os, sys, types, torch
os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, %r)
from sglang.srt.layers.attention.qsa.rows_resolve import (
    MODE_EVEN, MODE_NONE, MODE_WEIGHTED, qsa_rows_resolve,
)
from sglang.srt.layers.attention.qsa.sparse_attn import compact_owned_rows
from sglang.srt.layers.attention import qwen_sparse_attn_backend as qb

torch.manual_seed(7)
R, C, B, K = 6, 300, 4, 40
req_to_token = torch.randint(0, 5000, (R, C), dtype=torch.int32)
seq_lens = torch.tensor([300, 17, 1, 250])
req_pool = torch.tensor([5, 0, 3, 1])
seq_ids = torch.tensor([0, 0, 1, 2, 3, 3, 3])
Tq = seq_ids.numel()
logical = torch.randint(-5, 320, (Tq, K), dtype=torch.int32)
logical[2, :3] = torch.tensor([0, 16, 17])   # len 17: 17 invalid
logical[3] = 0                                 # len 1: only position 0 valid
meta = types.SimpleNamespace(
    is_cuda_graph=True, token_to_batch_idx=seq_ids, sequence_lengths=seq_lens,
    row_req_pool_indices=req_pool,
)

def reference(mode, **kw):
    stub = types.SimpleNamespace(req_to_token=req_to_token)
    slots = qb.QwenSparseAttnBackend._logical_to_physical_graph(stub, logical, meta)
    if mode == MODE_NONE:
        stub.dcp_size = 1
    elif mode == MODE_EVEN:
        stub.dcp_size, stub.dcp_rank, stub.uneven_dcp_weighted = kw["world"], kw["rank"], False
    else:
        stub.dcp_size, stub.uneven_dcp_weighted = 3, True
        stub.cp_S, stub.cp_lo, stub.cp_hi, stub.cp_ratio = kw["cp_S"], kw["cp_lo"], kw["cp_hi"], kw["cp_ratio"]
    rows = qb.QwenSparseAttnBackend._local_rows(stub, slots)
    return compact_owned_rows(rows)

cases = [
    (MODE_NONE, {}),
    (MODE_EVEN, dict(world=3, rank=1)),
    (MODE_WEIGHTED, dict(cp_S=7, cp_lo=2, cp_hi=5, cp_ratio=3)),
    (MODE_WEIGHTED, dict(cp_S=7, cp_lo=0, cp_hi=2, cp_ratio=2)),
]
for mode, kw in cases:
    rows, counts = qsa_rows_resolve(logical, seq_ids, seq_lens, req_pool, req_to_token, mode=mode, **kw)
    ref_rows, ref_counts = reference(mode, **kw)
    assert rows.dtype == torch.int32 and counts.dtype == torch.int32
    assert counts.tolist() == ref_counts.tolist(), (mode, counts.tolist(), ref_counts.tolist())
    for i in range(Tq):
        n = int(counts[i])
        assert (rows[i, :n] >= 0).all() and (rows[i, n:] == -1).all(), (mode, i, rows[i].tolist())
        assert sorted(rows[i, :n].tolist()) == sorted(ref_rows[i, :n].tolist()), (mode, i)
    assert int(counts[2]) <= K - 1  # lane 17 of the length-17 row is invalid
print("ROWS_RESOLVE_OK", [int(c) for c in counts])
'''


def _run_interpreted():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TRITON_INTERPRET="1", PYTHONPATH=SRC)
    return subprocess.run(
        [sys.executable, "-c", CHECK % SRC], env=env, capture_output=True, text=True, timeout=600
    )


def test_fused_rows_match_the_torch_chain_under_the_interpreter():
    pytest.importorskip("triton")
    r = _run_interpreted()
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    assert "ROWS_RESOLVE_OK" in r.stdout


def test_backend_switch_and_wiring(monkeypatch):
    from sglang.srt.layers.attention import qwen_sparse_attn_backend as qb

    for raw, want in (("", True), ("0", False), ("1", True)):
        qb._QSA_ROWS_FUSED["on"] = None
        if raw == "":
            monkeypatch.delenv("SGLANG_QSA_ROWS_FUSED", raising=False)
        else:
            monkeypatch.setenv("SGLANG_QSA_ROWS_FUSED", raw)
        assert qb._qsa_rows_fused_on() is want, raw
    qb._QSA_ROWS_FUSED["on"] = None
    import inspect

    src = inspect.getsource(qb.QwenSparseAttnBackend)
    # the three rows routes hand the fused counts to the attention
    assert src.count("rows, counts = self._rows_and_counts(topk_indices, metadata)") == 3
    assert "def _attend_rows(\n        self, q: torch.Tensor, layer, rows: torch.Tensor, row_counts=None" in src
    # the eager path (no graph metadata) keeps the torch chain
    meta = types.SimpleNamespace(is_cuda_graph=False)
    stub = types.SimpleNamespace(
        _topk_rows=lambda t, m: "chain", req_to_token=None, dcp_size=1
    )
    rows, counts = qb.QwenSparseAttnBackend._rows_and_counts(stub, torch.zeros(1, 2, dtype=torch.int32), meta)
    assert rows == "chain" and counts is None


def test_fused_kernel_is_bounded_by_lane_and_writes_every_lane_once():
    import inspect

    from sglang.srt.layers.attention.qsa import rows_resolve as rr

    src = inspect.getsource(rr)
    assert "dest = tl.where(keep, incl - 1, count + (offs - incl))" in src
    assert "tl.store(rows_ptr + q * K + dest, row, mask=lane)" in src
    assert "valid = lane & (pos >= 0) & (pos < seq_len)" in src
