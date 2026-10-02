"""D's handback extend prefix read, stock vs forced KV split -- NEEDS CUDA.

N6d (ec4d492f58): D's first P>D extend (5 new tokens on ~98k cached, DCP=3,
TP0 owns ~43k prefix rows) spends its extra ~80-140 ms in the 15 full-
attention layers. The DCP extend reads the owned prefix with
prefill_wrapper_paged.forward_return_lse(q_full, causal=False): 5 queries x
24 q heads (GQA 6 -> 30 packed rows, one 64-row q tile per KV head, 4 KV
heads). Whether flashinfer's stock plan splits that KV (its search allows up
to 2*num_sm/num_kv_heads work items) decides whether the paged kernel or the
DCP collectives carry the cost. This check answers it in seconds: the stock
plan's split flag and time, a forced split's time, and the output/LSE match
(one bf16 ulp; flashinfer 0.6.14 writes split partials in the output dtype).

Run in a GPU window (same recipe as test_fi_prefill_wave_split_gpu_0924.py):
  TEST27B_GPU=1 CUDA_VISIBLE_DEVICES=<the 5090 UUID> PATH=/usr/local/cuda/bin:/usr/bin:/bin \
  /root/.claude/jobs/1ab4cd30/tmp/test27b.sh env PYTHONPATH=<tree>/python \
  /spinning/htsglang-gpu/.venv/bin/python -m pytest -q -s -p no:cacheprovider \
  test/registered/unit/layers/attention/test_fi_small_q_prefix_split_gpu_1002.py
"""

import math
import unittest

import pytest
import torch

if not torch.cuda.is_available():  # pragma: no cover - desk
    pytest.skip("needs CUDA", allow_module_level=True)

from sglang.srt.layers.attention import fi_jit_cache_check as _jit  # noqa: E402

_JIT_OK, _JIT_LINES = _jit.check_prefill_modules(("bf16", "e4m3"))
if not _JIT_OK:  # pragma: no cover
    pytest.skip("flashinfer module would compile here: " + " | ".join(_JIT_LINES), allow_module_level=True)

flashinfer = pytest.importorskip("flashinfer")

from sglang.test.test_utils import CustomTestCase  # noqa: E402

QO, HQ, HKV, HD = 5, 24, 4, 256
WS_BYTES = 384 * 1024 * 1024


def _run(kv_len: int, fixed_split_size, kv_dtype, reps: int = 20):
    torch.manual_seed(7 + kv_len)
    dev = "cuda"
    q = torch.randn(QO, HQ, HD, dtype=torch.bfloat16, device=dev)
    k = (torch.randn(kv_len, 1, HKV, HD, device=dev) * 0.5).to(kv_dtype)
    v = (torch.randn(kv_len, 1, HKV, HD, device=dev) * 0.5).to(kv_dtype)
    ws = torch.zeros(WS_BYTES, dtype=torch.uint8, device=dev)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
    i32 = dict(dtype=torch.int32, device=dev)
    w.plan(torch.tensor([0, QO], **i32), torch.tensor([0, kv_len], **i32),
           torch.arange(kv_len, **i32), torch.tensor([1], **i32),
           HQ, HKV, HD, 1, causal=False, q_data_type=torch.bfloat16,
           kv_data_type=kv_dtype, fixed_split_size=fixed_split_size)
    out, lse = w.run_return_lse(q, (k, v)) if hasattr(w, "run_return_lse") else w.run(q, (k, v), return_lse=True)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        w.run(q, (k, v))
    b.record()
    torch.cuda.synchronize()
    split_flag = int(w._plan_info[14]) if hasattr(w, "_plan_info") else -1
    return out.float(), lse.float(), split_flag, a.elapsed_time(b) / reps


class TestHandbackPrefixRead(CustomTestCase):
    def _compare(self, kv_len, kv_dtype):
        num_sm = torch.cuda.get_device_properties(0).multi_processor_count
        stock_o, stock_l, s0, t0 = _run(kv_len, None, kv_dtype)
        chunk = max(128, math.ceil(kv_len / max(1, (2 * num_sm) // HKV)))
        split_o, split_l, s1, t1 = _run(kv_len, chunk, kv_dtype)
        diff = (split_o - stock_o).abs().max().item()
        ulp = stock_o.abs().amax().item() * 2.0 ** -7
        print("kv %d %s: stock split_flag=%d %.3f ms | forced chunk=%d split_flag=%d %.3f ms | "
              "max|do| %.3e (%.2f ulp) max|dlse| %.3e"
              % (kv_len, kv_dtype, s0, t0, chunk, s1, t1, diff, diff / ulp if ulp else 0.0,
                 (split_l - stock_l).abs().max().item()))
        self.assertLessEqual(diff, ulp)
        self.assertTrue(math.isfinite(diff))

    def test_owned_prefix_rows_fp8(self):
        for kv_len in (27808, 43140):
            try:
                self._compare(kv_len, torch.float8_e4m3fn)
            except (RuntimeError, TypeError) as exc:  # pragma: no cover
                self.skipTest(f"fp8 KV prefill not built: {exc}")

    def test_owned_prefix_rows_bf16(self):
        self._compare(43140, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
