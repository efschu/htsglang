#!/usr/bin/env python3
"""Non-GEMM components of one 27B P layer at a 512-token chunk (Backlog #38, N4B).

Answers "what is the 3080 stage's 'other'" per kernel, on whichever card it runs
(sm_86 or sm_120), with Qwen3.8-27B dims:
  gdn        sglang fla chunk_gated_delta_rule, T=512, 48 v-heads x 128/128,
             qk-l2norm in kernel, fp32 state (the triton linear-attn backend call)
  attn_pre   full attention of the 512-token chunk against a prefix of P tokens
             (24 q-heads, 4 kv-heads, head_dim 256), torch SDPA as the FA2 proxy
             (flashinfer's own prefill is the serving kernel; SDPA flash is the
             same algorithm class on both cards) -- non-causal prefix part
  attn_diag  the causal 512x512 part
  rmsnorm    [512, 5120] (sgl_kernel)
  silu_mul   [512, 2*17408] -> [512, 17408] (sgl_kernel)
  q8_h/q8_i  per_token_quant_int8 of [512,5120] / [512,17408]
Timing as in bench_5090_nvfp4.py (CUDA graph where possible, else events).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

for _v in ("TVM_FFI_CACHE_DIR", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
    _p = os.environ.get(_v, "")
    if not _p or _p.startswith("/root/.cache") or _p.startswith(os.path.expanduser("~/.cache")):
        sys.exit(f"refuse: {_v}={_p!r} must be set and must not point under ~/.cache")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from bench_5090_nvfp4 import timed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--prefixes", default="1024,4096,8192,16384")
    a = ap.parse_args()
    dev = torch.cuda.current_device()
    meta = {"device": torch.cuda.get_device_name(dev), "cap": torch.cuda.get_device_capability(dev),
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(json.dumps(meta), flush=True)
    rows = []

    def rec(name, us, mode, **kw):
        r = {"comp": name, "us": round(us, 1), "mode": mode, **kw}
        rows.append(r)
        print(json.dumps(r), flush=True)

    def err(name, e, **kw):
        r = {"comp": name, "error": f"{type(e).__name__}: {str(e)[:300]}", **kw}
        rows.append(r)
        print(json.dumps(r), flush=True)

    T, Hv, Dk, Dv = 512, 48, 128, 128
    bf = torch.bfloat16
    # --- GDN chunk kernel ---
    try:
        from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule

        q = torch.randn(1, T, Hv, Dk, device="cuda", dtype=bf)
        k = torch.randn(1, T, Hv, Dk, device="cuda", dtype=bf)
        v = torch.randn(1, T, Hv, Dv, device="cuda", dtype=bf)
        g = F.logsigmoid(torch.rand(1, T, Hv, device="cuda", dtype=torch.float32))
        beta = torch.rand(1, T, Hv, device="cuda", dtype=bf).sigmoid()
        state = torch.zeros(4, Hv, Dv, Dk, device="cuda", dtype=torch.float32)
        idx = torch.tensor([1], device="cuda", dtype=torch.int32)
        cu = torch.tensor([0, T], device="cuda", dtype=torch.long)

        def f_gdn():
            return chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, initial_state=state,
                                          initial_state_indices=idx, cu_seqlens=cu, head_first=False,
                                          use_qk_l2norm_in_kernel=True)
        us, mode = timed(f_gdn, a.reps)
        rec("gdn", us, mode, T=T, heads=Hv)
    except Exception as e:  # noqa: BLE001
        err("gdn", e)
    # --- attention (SDPA proxy) ---
    Hq, Hk, D = 24, 4, 256
    qa = torch.randn(1, Hq, T, D, device="cuda", dtype=bf)
    try:
        kd = torch.randn(1, Hq, T, D, device="cuda", dtype=bf)
        vd = torch.randn(1, Hq, T, D, device="cuda", dtype=bf)
        us, mode = timed(lambda: F.scaled_dot_product_attention(qa, kd, vd, is_causal=True), a.reps)
        rec("attn_diag", us, mode, flops_g=round(4 * Hq * D * T * T / 2 / 1e9, 2))
    except Exception as e:  # noqa: BLE001
        err("attn_diag", e)
    for p in [int(x) for x in a.prefixes.split(",") if x]:
        try:
            kp = torch.randn(1, Hk, p, D, device="cuda", dtype=bf).repeat_interleave(Hq // Hk, dim=1)
            vp = torch.randn(1, Hk, p, D, device="cuda", dtype=bf).repeat_interleave(Hq // Hk, dim=1)
            fl = 4.0 * Hq * D * T * p
            us, mode = timed(lambda: F.scaled_dot_product_attention(qa, kp, vp), a.reps)
            rec("attn_pre", us, mode, prefix=p, tflops=round(fl / us / 1e6, 1))
            del kp, vp
        except Exception as e:  # noqa: BLE001
            err("attn_pre", e, prefix=p)
    # --- elementwise ---
    try:
        from sgl_kernel import rmsnorm, silu_and_mul

        x = torch.randn(T, 5120, device="cuda", dtype=bf)
        w = torch.ones(5120, device="cuda", dtype=bf)
        us, mode = timed(lambda: rmsnorm(x, w, 1e-6), a.reps)
        rec("rmsnorm", us, mode)
        gu = torch.randn(T, 2 * 17408, device="cuda", dtype=bf)
        us, mode = timed(lambda: silu_and_mul(gu), a.reps)
        rec("silu_mul", us, mode)
    except Exception as e:  # noqa: BLE001
        err("elementwise", e)
    try:
        from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8

        xh = torch.randn(T, 5120, device="cuda", dtype=bf)
        xi = torch.randn(T, 17408, device="cuda", dtype=bf)
        us, mode = timed(lambda: per_token_quant_int8(xh), a.reps)
        rec("q8_h", us, mode)
        us, mode = timed(lambda: per_token_quant_int8(xi), a.reps)
        rec("q8_i", us, mode)
    except Exception as e:  # noqa: BLE001
        err("q8", e)
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(a.out, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
