"""Split vs unsplit flashinfer prefill at the 27B P geometry -- NEEDS CUDA.

Operator order 24.09.: check the wave-aware split bit-identical, or within
tolerance, against eager without the split, before a boot arms it. flashinfer
0.6.14 writes split partials as the OUTPUT dtype (tmp_v is DTypeO; upstream
#4356 open, #5166 argues for fp32 partials), so bit-identity is NOT expected:
each partial is rounded to bf16 once before the fp32 merge. The bound pinned
here is one bf16 ulp of the output magnitude, and the share of differing
elements is printed. Geometry: 512 new tokens, causal over prefix + chunk (the
P plan with use_ragged False), 24 q / 4 KV heads, head_dim 256, page_size 1,
prefixes where the chooser splits on the 5090. Skipped without CUDA; on a
3080 the chooser declines, so the split is FORCED at the 5090's choice there
(the kernel is the same).

Run in a GPU window (seconds, < 1 GiB):
  CUDA_VISIBLE_DEVICES=<5090> PYTHONPATH=<tree>/python python -m pytest -q -s \
    test/registered/unit/layers/attention/test_fi_prefill_wave_split_gpu_0924.py
"""

import math
import unittest

import pytest
import torch

if not torch.cuda.is_available():  # pragma: no cover - desk
    pytest.skip("needs CUDA", allow_module_level=True)

flashinfer = pytest.importorskip("flashinfer")

from sglang.srt.layers.attention import fi_prefill_wave_split as W  # noqa: E402
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cuda_ci(est_time=30, suite="nightly-1-gpu")

QO, HQ, HKV, HD = 512, 24, 4, 256
WS_BYTES = 384 * 1024 * 1024


def _run(prefix: int, fixed_split_size, kv_dtype=torch.bfloat16):
    torch.manual_seed(1234 + prefix)
    kv_len = prefix + QO
    dev = "cuda"
    q = torch.randn(QO, HQ, HD, dtype=torch.bfloat16, device=dev)
    k = (torch.randn(kv_len, 1, HKV, HD, device=dev) * 0.5).to(kv_dtype)
    v = (torch.randn(kv_len, 1, HKV, HD, device=dev) * 0.5).to(kv_dtype)
    ws = torch.zeros(WS_BYTES, dtype=torch.uint8, device=dev)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
    i32 = dict(dtype=torch.int32, device=dev)
    w.plan(
        torch.tensor([0, QO], **i32),
        torch.tensor([0, kv_len], **i32),
        torch.arange(kv_len, **i32),
        torch.tensor([1], **i32),
        HQ,
        HKV,
        HD,
        1,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=kv_dtype,
        fixed_split_size=fixed_split_size,
    )
    out = w.run(q, (k, v))
    torch.cuda.synchronize()
    split_flag = int(w._plan_info[14]) if hasattr(w, "_plan_info") else -1
    return out.float(), split_flag


class TestSplitMatchesStock(CustomTestCase):
    def _choice(self, prefix):
        c = W.choose_prefill_kv_split(
            [QO], [prefix + QO], num_qo_heads=HQ, num_kv_heads=HKV, head_dim=HD,
            num_sm=170, float_workspace_bytes=WS_BYTES)
        self.assertEqual(c.reason, "split")
        return c

    def _compare(self, prefix, kv_dtype=torch.bfloat16):
        c = self._choice(prefix)
        stock, s0 = _run(prefix, None, kv_dtype)
        split, s1 = _run(prefix, c.fixed_split_size, kv_dtype)
        self.assertEqual(s0, 0, "stock plan must not split (the measured 192-CTA form)")
        self.assertEqual(s1, 1, "fixed_split_size must engage split-KV")
        diff = (split - stock).abs()
        scale = stock.abs().amax().item()
        ulp = scale * 2.0 ** -7  # one bf16 ulp at the output's magnitude
        share = (diff > 0).float().mean().item()
        print(
            "prefix %d chunks %d fixed_split_size %d: max|diff| %.3e (%.2f bf16 ulp at "
            "max|o| %.3e), differing share %.4f, bit-identical %s"
            % (prefix, c.chunks, c.fixed_split_size, diff.max().item(),
               diff.max().item() / ulp, scale, share, share == 0.0)
        )
        self.assertLessEqual(diff.max().item(), ulp)
        self.assertTrue(math.isfinite(diff.max().item()))

    def test_bf16_kv_at_the_split_depths(self):
        for prefix in (10240, 36000, 98304):
            self._compare(prefix)

    def test_fp8_kv_like_group_p(self):
        try:
            self._compare(36000, torch.float8_e4m3fn)
        except (RuntimeError, TypeError) as exc:  # pragma: no cover - build without fp8 prefill
            self.skipTest(f"fp8 KV prefill not built: {exc}")


if __name__ == "__main__":
    unittest.main()
