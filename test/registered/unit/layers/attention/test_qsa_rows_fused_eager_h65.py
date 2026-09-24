"""fnFL2 H65: SGLANG_WEG2_QSA_ROWS_FUSED_EAGER -- the P prefix chunks resolve
their top-k rows through the fused Triton launch (qsa/rows_resolve.py) the
graph path already uses, instead of the eager torch chain
(_logical_to_physical -> _local_rows), whose int64 top-k copy, gather,
full_like and where hold ~0.57 GB per full-attention layer of a 16k chunk.

CPU: the route (default off, graph path unchanged), the wiring of
_rows_and_counts, and -- in a subprocess under Triton's interpreter, like
tests/moe_offload/test_qsa_rows_resolve_fused_0920.py -- that the fused
kernel returns exactly the eager chain's rows for QSA's valid-first top-k.
Metal (GPU window): the counts-bounded kernel loop gives the bit-identical
attention output.
"""

import os
import subprocess
import sys
import types
import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention import qwen_sparse_attn_backend as qb
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(qb.__file__))))
SRC = os.path.dirname(os.path.dirname(SRC))  # .../python


class _Cuda:
    is_cuda = True


class RouteTest(unittest.TestCase):
    def setUp(self):
        self._old = qb._QSA_ROWS_FUSED["on"]
        qb._QSA_ROWS_FUSED["on"] = True
        self.addCleanup(qb._QSA_ROWS_FUSED.__setitem__, "on", self._old)

    def test_graph_always_eager_only_with_the_switch(self):
        graph = types.SimpleNamespace(is_cuda_graph=True, row_req_pool_indices=object())
        eager = types.SimpleNamespace(is_cuda_graph=False, row_req_pool_indices=object())
        r2t = object()
        self.assertTrue(qb._qsa_rows_fused_route(graph, _Cuda(), r2t, eager=False))
        self.assertFalse(qb._qsa_rows_fused_route(eager, _Cuda(), r2t, eager=False))
        self.assertTrue(qb._qsa_rows_fused_route(eager, _Cuda(), r2t, eager=True))
        # never on CPU tensors, never without the tables
        self.assertFalse(qb._qsa_rows_fused_route(eager, torch.zeros(1), r2t, eager=True))
        self.assertFalse(qb._qsa_rows_fused_route(eager, _Cuda(), None, eager=True))
        none_rows = types.SimpleNamespace(is_cuda_graph=False, row_req_pool_indices=None)
        self.assertFalse(qb._qsa_rows_fused_route(none_rows, _Cuda(), r2t, eager=True))
        # SGLANG_QSA_ROWS_FUSED=0 still turns every fused resolve off
        qb._QSA_ROWS_FUSED["on"] = False
        self.assertFalse(qb._qsa_rows_fused_route(graph, _Cuda(), r2t, eager=True))

    def test_default_is_off(self):
        self.assertIs(envs.SGLANG_WEG2_QSA_ROWS_FUSED_EAGER.get(), False)


class WiringTest(unittest.TestCase):
    def _stub(self):
        return types.SimpleNamespace(dcp_size=1, req_to_token=torch.zeros(2, 8, dtype=torch.int32))

    def _meta(self):
        return types.SimpleNamespace(
            is_cuda_graph=False,
            token_to_batch_idx=torch.zeros(3, dtype=torch.int32),
            sequence_lengths=torch.tensor([8], dtype=torch.int32),
            row_req_pool_indices=torch.tensor([1]),
            token_slot_table=torch.arange(8, dtype=torch.int32).view(1, 8),
        )

    def test_switch_on_calls_the_fused_resolve_with_the_eager_tables(self):
        from sglang.srt.layers.attention.qsa import rows_resolve

        stub, meta = self._stub(), self._meta()
        topk = torch.zeros(3, 5, dtype=torch.int32)
        seen = {}

        def fake(logical, seq_ids, seq_lens, req_pool, r2t, *, mode, **kw):
            seen.update(logical=logical, seq_ids=seq_ids, seq_lens=seq_lens, req_pool=req_pool,
                        r2t=r2t, mode=mode, kw=kw)
            return "rows", "counts"

        with envs.SGLANG_WEG2_QSA_ROWS_FUSED_EAGER.override(True), \
                mock.patch.object(qb, "_qsa_rows_fused_route", lambda m, t, r, eager: eager), \
                mock.patch.object(rows_resolve, "qsa_rows_resolve", fake), \
                self.assertLogs(qb.logger, level="INFO") as logs:
            out = qb.QwenSparseAttnBackend._rows_and_counts(stub, topk, meta)
            qb.QwenSparseAttnBackend._rows_and_counts(stub, topk, meta)
        self.assertEqual(out, ("rows", "counts"))
        self.assertEqual(seen["mode"], rows_resolve.MODE_NONE)
        self.assertIs(seen["seq_ids"], meta.token_to_batch_idx)
        self.assertIs(seen["seq_lens"], meta.sequence_lengths)
        self.assertIs(seen["req_pool"], meta.row_req_pool_indices)
        self.assertIs(seen["r2t"], stub.req_to_token)
        eager_lines = [r for r in logs.records if "EAGER forwards resolve" in r.getMessage()]
        self.assertEqual(len(eager_lines), 1)

    def test_switch_off_keeps_the_torch_chain(self):
        stub, meta = self._stub(), self._meta()
        stub._local_rows = lambda slots: slots
        stub._logical_to_physical = qb.QwenSparseAttnBackend._logical_to_physical
        stub._topk_rows = lambda t, m: qb.QwenSparseAttnBackend._topk_rows(stub, t, m)
        topk = torch.tensor([[0, 1, -1], [2, -1, -1], [7, 3, -1]], dtype=torch.int32)
        with envs.SGLANG_WEG2_QSA_ROWS_FUSED_EAGER.override(False), \
                mock.patch.object(qb, "_qsa_rows_fused_route", lambda m, t, r, eager: eager):
            rows, counts = qb.QwenSparseAttnBackend._rows_and_counts(stub, topk, meta)
        self.assertIsNone(counts)
        self.assertEqual(rows.tolist(), [[0, 1, -1], [2, -1, -1], [7, 3, -1]])


CHECK = r'''
import os, sys, types, torch
os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, %r)
from sglang.srt.layers.attention.qsa.rows_resolve import MODE_NONE, qsa_rows_resolve
from sglang.srt.layers.attention.qsa.kernel import torch_expand_qsa_block_indices
from sglang.srt.layers.attention import qwen_sparse_attn_backend as qb

torch.manual_seed(65)
R, C, ratio, token_topk = 5, 640, 4, 48
block_topk = token_topk // ratio
req_to_token = torch.randint(1, 50000, (R, C), dtype=torch.int32)
seq_lens = torch.tensor([600, 131, 7], dtype=torch.int32)
req_pool = torch.tensor([3, 0, 4])
ext = [40, 25, 7]                                   # extend rows per request (prefix = len - ext)
seq_ids = torch.cat([torch.full((e,), b, dtype=torch.int32) for b, e in enumerate(ext)])
pos = torch.cat([torch.arange(int(L) - e, int(L)) for L, e in zip(seq_lens, ext)])
blocks = torch.full((pos.numel(), block_topk), -1, dtype=torch.int32)
for i, p in enumerate(pos.tolist()):
    vis = (p + 1) // ratio
    if vis:
        pick = torch.randperm(vis)[:block_topk]
        blocks[i, : pick.numel()] = pick.to(torch.int32)
logical = torch_expand_qsa_block_indices(blocks, pos, seq_lens[seq_ids.long()], ratio, token_topk)
max_len = int(seq_lens.max())
meta = types.SimpleNamespace(
    is_cuda_graph=False, token_to_batch_idx=seq_ids, sequence_lengths=seq_lens,
    row_req_pool_indices=req_pool, token_slot_table=req_to_token[req_pool.long(), :max_len],
)
chain = qb.QwenSparseAttnBackend._logical_to_physical(logical, meta)
rows, counts = qsa_rows_resolve(logical, seq_ids, seq_lens, req_pool, req_to_token, mode=MODE_NONE)
assert torch.equal(rows, chain), (rows[:2], chain[:2])
assert torch.equal(counts, (chain >= 0).sum(1).to(torch.int32))
valid = chain >= 0
assert bool((valid[:, 1:] <= valid[:, :-1]).all()), "QSA hands over valid-first rows"
print("FUSED-EAGER-OK", int(valid.sum()), tuple(rows.shape))
'''


class InterpretedEquivalenceTest(unittest.TestCase):
    def test_fused_rows_equal_the_eager_chain(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TRITON_INTERPRET="1")
        r = subprocess.run([sys.executable, "-c", CHECK % SRC], capture_output=True, text=True, env=env, timeout=600)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn("FUSED-EAGER-OK", r.stdout)


@unittest.skipUnless(torch.cuda.is_available(), "metal: run in a GPU window")
class MetalTest(unittest.TestCase):
    def test_counts_bounded_loop_is_bit_identical(self):
        from sglang.srt.layers.attention.qsa.sparse_attn import sparse_attn_rows_triton

        g = torch.Generator(device="cuda").manual_seed(66)
        tq, n, topk = 900, 8192, 2051
        q = torch.randn(tq, 24, 256, device="cuda", generator=g).bfloat16()
        k = (torch.randn(n, 2, 256, device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)
        v = (torch.randn(n, 2, 256, device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)
        rows = torch.randint(0, n, (tq, topk), device="cuda", generator=g, dtype=torch.int32)
        valid_n = torch.randint(1, topk + 1, (tq, 1), device="cuda", generator=g)
        rows = torch.where(torch.arange(topk, device="cuda").unsqueeze(0) < valid_n, rows, -1)
        counts = (rows >= 0).sum(1).to(torch.int32)
        a, la = sparse_attn_rows_triton(q, k, v, rows, 0.0625)
        b, lb = sparse_attn_rows_triton(q, k, v, rows, 0.0625, row_counts=counts)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(la, lb))


if __name__ == "__main__":
    unittest.main()
