"""KV split inside the full prefill graph -- NEEDS CUDA, operator window only (METAL TEST).

Run (seconds; loads cached modules, compiles nothing -- the pre-check skips
instead, see fi_jit_cache_check.py):
  TEST27B_GPU=1 CUDA_VISIBLE_DEVICES=GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d \
  PATH=/usr/local/cuda/bin:/usr/bin:/bin /root/.claude/jobs/1ab4cd30/tmp/test27b.sh \
  env PYTHONPATH=<tree>/python /spinning/htsglang-gpu/.venv/bin/python -m pytest -q -s \
  -p no:cacheprovider test/registered/unit/layers/attention/test_fi_graph_split_gpu_0924.py
(the parity alone, first failure stops: append ``::TestPlannerParity -x``)

flashinfer 0.6.14 AND 0.7.0 (2026-10-04 port; the layout comes from
``fi_graph_split.installed_version_key()``, any other version fails the
contract assertion instead of running a layout it does not know). The desk half
-- the same arrays against the C++ planner header of 0.6.14, 0.7.0@2f3bc5ac and
the PyPI 0.7.0 -- is test_fi_graph_split_cpp_parity_0924.py; this file is what
the desk cannot run: the kernel reading OUR arrays.

What it proves (27B P geometry: 512 new tokens causal over prefix + chunk,
24 q / 4 KV heads, head_dim 256, page_size 1):
(A) our work-item arrays are flashinfer's: the JIT module's own fixed-split
    arrays (read back from the wrapper's pinned int buffer) equal
    fi_graph_split.split_arrays for the same chunk. The reference is an EAGER
    plan, and 0.6.14 reserves its split partials per work item x CTA_TILE_Q x
    sizeof(float) (#5177): 506 MiB at 98k / 7 chunks, so the workspace is sized
    by stock_eager_split_float_bytes -- with 384 MiB the planner refused, which
    was the first window run's failure (2026-09-24 20:23Z; found at the desk by
    test_fi_graph_split_cpp_parity_0924.py, which compiles the same planner
    host-only and compares the same arrays over ~700 shapes without a GPU).
    A refusal or a difference fails with the case, the plan vector and each
    array's first differing index;
(B) a CUDA graph captured ONCE with the split grid, replayed at prefixes 4k /
    36k / 98k with per-replay arrays, matches eager flashinfer WITHOUT split
    within one bf16 ulp of the output magnitude (not bit-identical by
    construction: split partials are bf16, #4356), and uses > 1 chunk at depth;
    replay times split vs a stock-plan graph are printed (prefix 4k / 16k / 36k /
    98k; the metal marker lines are ``GRAPH-SPLIT-PROBE ...``). Run for bf16 KV
    and for fp8 (e4m3) KV, the server's own KV dtype (kUseRepack, 1 CTA/SM): the
    fp8 class skips -- never compiles -- when its JIT module is not cached.
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
from sglang.srt.layers.attention.fi_prefill_wave_split import fa2_cta_tile_q  # noqa: E402
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=60, suite="nightly-1-gpu")

QO, HQ, HKV, HD = 512, 24, 4, 256
KV_MAX = 98304 + QO
WS = 384 * 1024 * 1024
DEV = "cuda"
I32 = dict(dtype=torch.int32, device=DEV)


def _kv(kv_len, seed, kv_dtype=torch.bfloat16):
    g = torch.Generator(device=DEV).manual_seed(seed)
    k = (torch.randn(kv_len, 1, HKV, HD, device=DEV, generator=g) * 0.5).to(kv_dtype)
    v = (torch.randn(kv_len, 1, HKV, HD, device=DEV, generator=g) * 0.5).to(kv_dtype)
    q = torch.randn(QO, HQ, HD, device=DEV, generator=g).to(torch.bfloat16)
    return q, k, v


def _plan(w, kv_len, kv_dtype=torch.bfloat16, **kw):
    w.plan(
        torch.tensor([0, QO], **I32), torch.tensor([0, kv_len], **I32),
        torch.arange(kv_len, **I32), torch.tensor([1], **I32),
        HQ, HKV, HD, 1, causal=True, q_data_type=torch.bfloat16,
        kv_data_type=kv_dtype, **kw)


def _eager_stock(q, k, v, kv_len, kv_dtype=torch.bfloat16):
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.zeros(WS, dtype=torch.uint8, device=DEV), "NHD", backend="fa2")
    _plan(w, kv_len, kv_dtype)
    return w.run(q, (k[:kv_len], v[:kv_len])).float()


def _first_diff(got, want):
    """Index of the first difference (or where the shorter one ends), None if equal."""
    for i, (a, b) in enumerate(zip(got, want)):
        if a != b:
            return i
    return None if len(got) == len(want) else min(len(got), len(want))


def _around(xs, i):
    return xs[max(0, i - 2): i + 3]


# Plain unittest.TestCase ON PURPOSE (both classes): sglang's CustomTestCase
# runs every test inside retry(), and the first window run showed nothing but
# "retry() exceed maximum number of retries" -- the cause stayed in the chained
# exception the excerpt did not reach.
class TestPlannerParity(unittest.TestCase):
    def test_python_arrays_equal_the_cpp_planner(self):
        major = torch.cuda.get_device_capability()[0]
        gqa = HQ // HKV
        for kv_len, n in ((36000 + QO, 5), (98304 + QO, 7), (2048 + QO, 4)):
            chunk = math.ceil(kv_len / n)
            tile = fa2_cta_tile_q(QO * gqa, HD, major)  # the eager plan's own tile rule
            items = math.ceil(QO * gqa / tile) * math.ceil(kv_len / chunk)
            fws = G.stock_eager_split_float_bytes(HQ, items, tile, HD)
            case = "kv=%d n=%d chunk=%d float_ws=%d B" % (kv_len, n, chunk, fws)
            ws = torch.empty(fws, dtype=torch.uint8, device=DEV)
            w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
            try:
                _plan(w, kv_len, fixed_split_size=chunk)
            except Exception as e:  # noqa: BLE001 - name the refusal itself
                self.fail("PARITY %s: the C++ planner refused: %s: %s" % (case, type(e).__name__, e))
            torch.cuda.synchronize()
            info = [int(x) for x in w._plan_info]
            pin = w._pin_memory_int_workspace_buffer

            def read(off, count, width=4):
                raw = pin[off: off + width * count]
                return (raw.view(torch.int32) if width == 4 else raw).tolist()

            want = G.split_arrays([QO], [kv_len], gqa=gqa, cta_tile_q=info[3], kv_chunk=chunk)
            got = {
                "request_indices": read(info[4], info[0]),
                "qo_tile_indices": read(info[5], info[0]),
                "kv_tile_indices": read(info[6], info[0]),
                "o_indptr": read(info[8], 2),
                "kv_chunk_size": read(info[9], 1),
                "merge_indptr": read(info[7], QO + 1) if info[14] else [],
            }
            diffs = []
            if info[0] != len(want["request_indices"]):
                diffs.append("work items cpp %d vs ours %d" % (info[0], len(want["request_indices"])))
            if not info[14]:
                diffs.append("the C++ plan did not split (split_kv=0)")
            # no block_valid_mask read here: an EAGER plan carries one only in 0.6.14 / PyPI 0.7.0 (split plans),
            # not after flashinfer #5176 (2f3bc5ac: every GRAPH plan) -- the mask is compared by the C++ parity
            # test and exercised end to end by TestGraphSplit below
            for name, g in got.items():
                i = _first_diff(g, want[name])
                if i is not None:
                    diffs.append(
                        "%s[%d] (len cpp %d, ours %d): cpp %s vs ours %s"
                        % (name, i, len(g), len(want[name]), _around(g, i), _around(want[name], i)))
            if diffs:
                self.fail("PARITY %s plan_info=%s -- first differences: %s" % (case, info, " | ".join(diffs)))
            print("PARITY %s: %d work items, tile %d, arrays identical" % (case, info[0], info[3]), flush=True)
            del w, ws


class TestGraphSplit(unittest.TestCase):
    KV = torch.bfloat16  # the fp8 subclass below flips this
    KV_NAME = "bf16"

    def _graph(self, split: bool):
        ws = torch.zeros(WS, dtype=torch.uint8, device=DEV)
        bufs = dict(
            qo_indptr_buf=torch.zeros(2, **I32),
            paged_kv_indptr_buf=torch.zeros(2, **I32),
            paged_kv_indices_buf=torch.zeros(KV_MAX + 256, **I32),
            paged_kv_last_page_len_buf=torch.ones(1, **I32),
        )
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", use_cuda_graph=True, backend="fa2", **bufs)
        k = torch.zeros(KV_MAX, 1, HKV, HD, dtype=self.KV, device=DEV)
        v = torch.zeros_like(k)
        q = torch.zeros(QO, HQ, HD, dtype=torch.bfloat16, device=DEV)
        out = torch.zeros(QO, HQ, HD, dtype=torch.bfloat16, device=DEV)
        _plan(w, 4096 + QO, self.KV)
        st = None
        if split:
            ok, why = G.flashinfer_contract_ok(w)
            self.assertTrue(ok, why)
            fi_version = G.installed_version_key()
            self.assertIsNotNone(fi_version, "flashinfer %s is not mirrored by fi_graph_split" % flashinfer.__version__)
            lay, why = G.layout_from_stock(
                [int(x) for x in w._plan_info], slots=1, max_chunks=7, num_qo_heads=HQ,
                num_kv_heads=HKV, head_dim_vo=HD, out_bytes=2,
                int_workspace_bytes=w._int_workspace_buffer.numel(),
                float_workspace_bytes=ws.numel(), fi_version=fi_version)
            self.assertIsNotNone(lay, why)
            print("GRAPH-SPLIT-PROBE armed flashinfer=%s kv=%s N=%d grid=%d (%d q tiles x %d chunks) partials=%.1f MB"
                  % (fi_version, self.KV_NAME, lay.max_chunks, lay.padded, lay.padded // lay.max_chunks,
                     lay.max_chunks, (lay.offsets["float_end"] - lay.offsets["v"]) / 1e6), flush=True)
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
        q2, k2, v2 = _kv(kv_len, seed, self.KV)
        q.copy_(q2)
        k[:kv_len].copy_(k2)
        v[:kv_len].copy_(v2)
        _plan(w, kv_len, self.KV)
        chunk = None
        if st is not None:
            chunk, _choice = G.apply(w, st, [QO], [kv_len], num_kv_heads=HKV, num_sm=self.num_sm)
        g.replay()
        torch.cuda.synchronize()
        return (q2, k2, v2, kv_len, chunk)

    def test_split_graph_matches_eager_stock(self):
        wS, stS, gS, qS, kS, vS, outS = self._graph(split=True)
        w0, _st0, g0, q0, k0, v0, out0 = self._graph(split=False)
        for prefix in (4096, 16384, 36000, 98304):
            q2, k2, v2, kv_len, chunk = self._replay(wS, stS, gS, qS, kS, vS, prefix, seed=prefix)
            got = outS.float().clone()
            ref = _eager_stock(q2, k2, v2, kv_len, self.KV)
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
                "GRAPH-SPLIT-PROBE kv=%s prefix=%d chunks=%d kv_chunk=%s max_diff=%.3e (%.2f bf16 ulp) "
                "differing=%.4f split_ms=%.3f stock_ms=%.3f speedup=x%.2f"
                % (self.KV_NAME, prefix, stS.last_chunks, chunk, diff.max().item(), diff.max().item() / ulp,
                   (diff > 0).float().mean().item(), t["split"], t["stock"], t["stock"] / t["split"]),
                flush=True)
            self.assertLessEqual(diff.max().item(), ulp)
            if prefix >= 36000:
                self.assertGreater(stS.last_chunks, 1)


class TestGraphSplitE4M3(TestGraphSplit):
    """The server's own KV dtype (--kv-cache-dtype fp8): the kernel whose
    real occupancy is ONE CTA per SM. Skips -- never compiles -- when that
    JIT module is not in the cache."""

    KV = torch.float8_e4m3fn
    KV_NAME = "e4m3"

    @classmethod
    def setUpClass(cls):
        ok, lines = J.check_prefill_modules(("e4m3",))
        if not ok:  # pragma: no cover - window hygiene
            raise unittest.SkipTest("fp8 flashinfer module would compile here: " + " | ".join(lines))
        super().setUpClass()


if __name__ == "__main__":
    unittest.main()
