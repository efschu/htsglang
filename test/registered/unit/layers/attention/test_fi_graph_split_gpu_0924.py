"""KV split inside the full prefill graph -- NEEDS CUDA, operator window only.

Run (seconds; loads cached modules, compiles nothing -- the pre-check skips
instead, see fi_jit_cache_check.py):
  TEST27B_GPU=1 CUDA_VISIBLE_DEVICES=GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d \
  PATH=/usr/local/cuda/bin:/usr/bin:/bin /root/.claude/jobs/1ab4cd30/tmp/test27b.sh \
  env PYTHONPATH=<tree>/python /spinning/htsglang-gpu/.venv/bin/python -m pytest -q -s \
  -p no:cacheprovider test/registered/unit/layers/attention/test_fi_graph_split_gpu_0924.py

What it proves (27B P geometry: 512 new tokens causal over prefix + chunk,
24 q / 4 KV heads, head_dim 256, page_size 1):
(A) our work-item arrays are flashinfer's: the C++ planner's own fixed-split
    arrays (read back from the wrapper's pinned int buffer) equal
    fi_graph_split.split_arrays for the same chunk;
(B) a CUDA graph captured ONCE with the split grid, replayed at prefixes 4k /
    36k / 98k with per-replay arrays, matches eager flashinfer WITHOUT split
    within one bf16 ulp of the output magnitude (not bit-identical by
    construction in 0.6.14: split partials are bf16, #4356), and uses > 1 chunk
    at depth; replay times split vs a stock-plan graph are printed.
"""

import math
import unittest

import pytest
import torch

if not torch.cuda.is_available():  # pragma: no cover - desk
    pytest.skip("needs CUDA", allow_module_level=True)

from sglang.srt.layers.attention import fi_jit_cache_check as J  # noqa: E402

_OK, _LINES = J.check_prefill_modules(("bf16",))  # also mirrors the server's arch flags
if not _OK:  # pragma: no cover - window hygiene
    pytest.skip("flashinfer module would compile here: " + " | ".join(_LINES), allow_module_level=True)

import flashinfer  # noqa: E402

from sglang.srt.layers.attention import fi_graph_split as G  # noqa: E402
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cuda_ci(est_time=60, suite="nightly-1-gpu")

QO, HQ, HKV, HD = 512, 24, 4, 256
KV_MAX = 98304 + QO
WS = 384 * 1024 * 1024
DEV = "cuda"
I32 = dict(dtype=torch.int32, device=DEV)


def _kv(kv_len, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    k = (torch.randn(kv_len, 1, HKV, HD, device=DEV, generator=g) * 0.5).to(torch.bfloat16)
    v = (torch.randn(kv_len, 1, HKV, HD, device=DEV, generator=g) * 0.5).to(torch.bfloat16)
    q = torch.randn(QO, HQ, HD, device=DEV, generator=g).to(torch.bfloat16)
    return q, k, v


def _plan(w, kv_len, **kw):
    w.plan(
        torch.tensor([0, QO], **I32), torch.tensor([0, kv_len], **I32),
        torch.arange(kv_len, **I32), torch.tensor([1], **I32),
        HQ, HKV, HD, 1, causal=True, q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16, **kw)


def _eager_stock(q, k, v, kv_len):
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.zeros(WS, dtype=torch.uint8, device=DEV), "NHD")
    _plan(w, kv_len)
    return w.run(q, (k[:kv_len], v[:kv_len])).float()


class TestPlannerParity(CustomTestCase):
    def test_python_arrays_equal_the_cpp_planner(self):
        for kv_len, n in ((36000 + QO, 5), (98304 + QO, 7), (2048 + QO, 4)):
            chunk = math.ceil(kv_len / n)
            w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.zeros(WS, dtype=torch.uint8, device=DEV), "NHD")
            _plan(w, kv_len, fixed_split_size=chunk)
            torch.cuda.synchronize()
            info = [int(x) for x in w._plan_info]
            pin = w._pin_memory_int_workspace_buffer
            want = G.split_arrays([QO], [kv_len], gqa=HQ // HKV, cta_tile_q=info[3], kv_chunk=chunk)
            items = info[0]
            self.assertEqual(items, len(want["request_indices"]))

            def read(off, count):
                return pin[off: off + 4 * count].view(torch.int32).tolist()

            self.assertEqual(read(info[4], items), want["request_indices"])
            self.assertEqual(read(info[5], items), want["qo_tile_indices"])
            self.assertEqual(read(info[6], items), want["kv_tile_indices"])
            self.assertEqual(read(info[8], 2), want["o_indptr"])
            self.assertEqual(read(info[9], 1), want["kv_chunk_size"])
            self.assertEqual(read(info[7], QO + 1), want["merge_indptr"])


class TestGraphSplit(CustomTestCase):
    def _graph(self, split: bool):
        ws = torch.zeros(WS, dtype=torch.uint8, device=DEV)
        bufs = dict(
            qo_indptr_buf=torch.zeros(2, **I32),
            paged_kv_indptr_buf=torch.zeros(2, **I32),
            paged_kv_indices_buf=torch.zeros(KV_MAX + 256, **I32),
            paged_kv_last_page_len_buf=torch.ones(1, **I32),
        )
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", use_cuda_graph=True, **bufs)
        k = torch.zeros(KV_MAX, 1, HKV, HD, dtype=torch.bfloat16, device=DEV)
        v = torch.zeros_like(k)
        q = torch.zeros(QO, HQ, HD, dtype=torch.bfloat16, device=DEV)
        out = torch.zeros(QO, HQ, HD, dtype=torch.bfloat16, device=DEV)
        _plan(w, 4096 + QO)
        st = None
        if split:
            ok, why = G.flashinfer_contract_ok(w)
            self.assertTrue(ok, why)
            lay, why = G.layout_from_stock(
                [int(x) for x in w._plan_info], slots=1, max_chunks=7, num_qo_heads=HQ,
                num_kv_heads=HKV, head_dim_vo=HD, out_bytes=2,
                int_workspace_bytes=w._int_workspace_buffer.numel(),
                float_workspace_bytes=ws.numel())
            self.assertIsNotNone(lay, why)
            st = G.GraphSplitState(lay, [int(x) for x in w._plan_info])
            G.apply(w, st, [QO], [4096 + QO], num_kv_heads=HKV, num_sm=self.num_sm)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            w.run(q, (k, v), out=out)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            w.run(q, (k, v), out=out)
        return w, st, g, q, k, v, out

    @classmethod
    def setUpClass(cls):
        cls.num_sm = torch.cuda.get_device_properties(0).multi_processor_count

    def _replay(self, w, st, g, q, k, v, prefix, seed):
        kv_len = prefix + QO
        q2, k2, v2 = _kv(kv_len, seed)
        q.copy_(q2)
        k[:kv_len].copy_(k2)
        v[:kv_len].copy_(v2)
        _plan(w, kv_len)
        chunk = None
        if st is not None:
            chunk, _choice = G.apply(w, st, [QO], [kv_len], num_kv_heads=HKV, num_sm=self.num_sm)
        g.replay()
        torch.cuda.synchronize()
        return (q2, k2, v2, kv_len, chunk)

    def test_split_graph_matches_eager_stock(self):
        wS, stS, gS, qS, kS, vS, outS = self._graph(split=True)
        w0, _st0, g0, q0, k0, v0, out0 = self._graph(split=False)
        for prefix in (4096, 36000, 98304):
            q2, k2, v2, kv_len, chunk = self._replay(wS, stS, gS, qS, kS, vS, prefix, seed=prefix)
            got = outS.float().clone()
            ref = _eager_stock(q2, k2, v2, kv_len)
            diff = (got - ref).abs()
            scale = ref.abs().amax().item()
            ulp = scale * 2.0 ** -7
            # timing: split graph vs stock graph at the same depth
            self._replay(w0, None, g0, q0, k0, v0, prefix, seed=prefix)
            t = {}
            for name, gg in (("split", gS), ("stock", g0)):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(10):
                    gg.replay()
                e.record()
                torch.cuda.synchronize()
                t[name] = s.elapsed_time(e) / 10
            print(
                "prefix %d: chunks %d (kv_chunk %s), max|diff| %.3e = %.2f bf16 ulp, differing %.4f, "
                "graph replay split %.3f ms vs stock %.3f ms (x%.2f)"
                % (prefix, stS.last_chunks, chunk, diff.max().item(), diff.max().item() / ulp,
                   (diff > 0).float().mean().item(), t["split"], t["stock"], t["stock"] / t["split"]))
            self.assertLessEqual(diff.max().item(), ulp)
            if prefix >= 36000:
                self.assertGreater(stS.last_chunks, 1)


if __name__ == "__main__":
    unittest.main()
