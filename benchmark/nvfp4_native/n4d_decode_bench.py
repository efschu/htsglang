#!/usr/bin/env python3
"""N4D: W4A8 decode GEMV (native NVFP4 layout) vs Marlin W4A16 vs N4A's W4A8 GEMM on ONE RTX 3080.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<nvml 3080> python n4d_decode_bench.py <out.json> [--quick]
      [--shapes D.gate_up,D.down,...] [--ms 1,4,8,16,32,48,512] [--sweep]

Per 27B shape (3080 D shards + P layers + D lm_head) and M:
  * correctness on one copy: new kernel, N4A, Marlin against
      (a) the fp64 W4A8 emulation (same int8 activations: s_x * sum_b s_b * sum(2*e2m1 * q8) / 2 * gs), and
      (b) the fp32 W4A16 reference  x_bf16 @ dequant(W)^T  (the order's "fp32-Dequant-Referenz");
  * time: CUDA graph over R weight copies (R * bytes >= 48 MB, i.e. rotated far beyond the 5 MB L2), us per call,
    effective GB/s (weight + scale bytes) and TOPS (2*M*N*K). "full" = activation quant + GEMM (what apply runs),
    "gemm" = GEMM alone on pre-quantised activations.
Random weights (E2M1 bytes uniform, E4M3 scale codes 0x28..0x47 as in n4c_micro), gaussian activations.
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch  # noqa: E402

from sglang.jit_kernel import nvfp4_w4a8_decode as dec  # noqa: E402
from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack  # noqa: E402
from sglang.jit_kernel.nvfp4_w4a8 import (  # noqa: E402
    nvfp4_w4a8_gemm,
    nvfp4_w4a8_linear,
    nvfp4_w4a8_quantize_activation,
)
from sglang.srt.layers.quantization import nvfp4_native_mixed as nm  # noqa: E402
from sglang.srt.layers.quantization.marlin_utils import (  # noqa: E402
    marlin_make_workspace,
    marlin_permute_scales,
)
from sglang.srt.layers.quantization.marlin_utils_fp4 import (  # noqa: E402
    apply_fp4_marlin_linear,
    nvfp4_marlin_process_global_scale,
    nvfp4_marlin_process_scales,
)

dev = torch.device("cuda:0")
GS = 0.0025
SHAPES = {
    "D.gate_up": (8192, 5120),
    "D.down": (5120, 4096),
    "D.lm_head": (82816, 5120),
    "P.gate_up": (34816, 5120),
    "P.down": (5120, 17408),
}
E2M1X2 = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)


def sync():
    torch.cuda.synchronize()


def graph_us(fns, iters=20):
    """fns: list of callables (one per weight copy); returns us per call of one graph replay of all of them."""
    for f in fns:
        f()
    sync()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns:
            f()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    for _ in range(3):
        g.replay()
    sync()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    res = []
    for _ in range(3):
        a.record()
        for _ in range(iters):
            g.replay()
        b.record()
        sync()
        res.append(a.elapsed_time(b) * 1000.0 / (iters * len(fns)))
    del g
    return sorted(res)[1]


def dequant_rows(w, raw_sc):
    """uint8 [N, K/2] + raw (unswizzled) e4m3 codes [N, K/16] -> fp64 weight WITHOUT gs, value = e2m1*e4m3."""
    lut = torch.tensor(E2M1X2, dtype=torch.float64, device=w.device) * 0.5
    codes = torch.stack([w & 0xF, w >> 4], dim=-1).reshape(w.shape[0], -1).long()
    sc = raw_sc.view(torch.float8_e4m3fn).to(torch.float64)
    return (lut[codes].view(w.shape[0], -1, 16) * sc.unsqueeze(-1)).view(w.shape[0], -1)


def errs(y, ref):
    d = y.double() - ref
    return {
        "rel_fro": float(d.norm() / ref.norm().clamp_min(1e-30)),
        "max_abs_over_absmax": float(d.abs().max() / ref.abs().max().clamp_min(1e-30)),
    }


def make_marlin(w, raw, n, k):
    qw = gptq_marlin_repack(w.view(torch.int32).T.contiguous(), torch.empty(0, dtype=torch.int, device=dev), k, n, 4)
    sc = raw.view(torch.float8_e4m3fn).T.contiguous().to(torch.bfloat16)
    sc = nvfp4_marlin_process_scales(marlin_permute_scales(s=sc, size_k=k, size_n=n, group_size=16))
    return qw, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--ms", default="1,4,8,16,32,48,512")
    ap.add_argument("--sweep", action="store_true", help="also time every (mode, kw) of the new kernel")
    ap.add_argument("--no-marlin", action="store_true")
    ap.add_argument("--no-n4a", action="store_true")
    ap.add_argument("--min-mb", type=float, default=48.0)
    args = ap.parse_args()
    res = {
        "device": torch.cuda.get_device_name(0),
        "cap": list(torch.cuda.get_device_capability(0)),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": [],
    }

    def dump():
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)

    g = torch.Generator(device=dev).manual_seed(4)
    gs_t = torch.tensor([GS], dtype=torch.float32, device=dev)
    gsc_m = nvfp4_marlin_process_global_scale(torch.tensor(GS, device=dev).to(torch.bfloat16)).reshape(1)
    ws_m = marlin_make_workspace(dev)
    ms = [int(v) for v in args.ms.split(",")]
    for name in args.shapes.split(","):
        n, k = SHAPES[name]
        nbytes = n * k // 2 + n * k // 16
        R = max(1, -(-int(args.min_mb * 1e6) // nbytes))
        copies = []
        for _ in range(R):
            w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev, generator=g)
            raw = torch.randint(0x28, 0x48, (n, k // 16), dtype=torch.uint8, device=dev, generator=g)
            sw = nm.swizzle_128x4(raw).contiguous().view(torch.float8_e4m3fn)
            copies.append([w, raw, sw, None])
        if not args.no_marlin:
            for c in copies:
                c[3] = make_marlin(c[0], c[1], n, k)
        w0, raw0, sw0, _ = copies[0]

        def ref_mm(a):  # a fp64 [M, K] -> a @ dequant(W0)^T * GS in fp64, 4096-row chunks (memory share)
            outs = []
            for r0 in range(0, n, 4096):
                outs.append(a @ dequant_rows(w0[r0:r0 + 4096], raw0[r0:r0 + 4096]).T)
            return torch.cat(outs, dim=1) * GS

        for M in ms:
            if name.endswith("lm_head") and M > 48:
                continue
            row = {"shape": name, "N": n, "K": k, "M": M, "R": R, "bytes": nbytes}
            try:
                x = torch.randn((M, k), dtype=torch.bfloat16, device=dev, generator=g)
                ref16 = ref_mm(x.double())  # W4A16 fp64 reference (== fp32 dequant ref up to fp32 rounding)
                xq, xs = nvfp4_w4a8_quantize_activation(x)
                ref8 = ref_mm(xq.double() * xs.double().unsqueeze(1))  # exact W4A8 emulation
                row["ref_absmax"] = float(ref16.abs().max())
                use_dec = M <= dec.DECODE_MAX_M and dec.eligible(k, w0)
                if use_dec:
                    y = dec.nvfp4_w4a8_decode_linear(x, w0, sw0, gs_t, n)
                    row["new_vs_w4a8_exact"] = errs(y, ref8)
                    row["new_vs_fp32_w4a16"] = errs(y, ref16)
                    cfg = dec.config_for(M, n, k)
                    row["new_cfg"] = list(cfg)
                    # the other mode as well, where both apply
                    if M <= dec.DIAG_MAX_M:
                        y2 = dec.nvfp4_w4a8_decode_linear(x, w0, sw0, gs_t, n, cfg=(1 - cfg[0], 2, 1, 1))
                        row["new_othermode_vs_w4a8_exact"] = errs(y2, ref8)
                if not args.no_n4a:
                    y = nvfp4_w4a8_linear(x, w0, sw0, gs_t, n)
                    row["n4a_vs_w4a8_exact"] = errs(y, ref8)
                    row["n4a_vs_fp32_w4a16"] = errs(y, ref16)
                if not args.no_marlin:
                    qw, sc = copies[0][3]
                    y = apply_fp4_marlin_linear(input=x, weight=qw, weight_scale=sc, weight_global_scale=gsc_m,
                                                workspace=ws_m, size_n=n, size_k=k, bias=None)
                    row["marlin_vs_fp32_w4a16"] = errs(y, ref16)
                sync()
                flops = 2.0 * M * n * k

                def rec(key, us):
                    row[key] = {"us": round(us, 2), "GBs": round(nbytes / us / 1e3, 1),
                                "TOPS": round(flops / us / 1e6, 2)}

                if use_dec:
                    cfg = dec.config_for(M, n, k)
                    mode = cfg[0]
                    rec("new_full", graph_us([lambda c=c: dec.nvfp4_w4a8_decode_linear(x, c[0], c[2], gs_t, n)
                                              for c in copies]))
                    xqd, xsd = dec.quantize_activation(x, permuted=(mode == 1))
                    outb = torch.empty((M, n), dtype=torch.bfloat16, device=dev)
                    rec("new_gemm", graph_us([lambda c=c: dec.decode_gemm(xqd, xsd, c[0], c[2], gs_t, n, cfg,
                                                                            out=outb) for c in copies]))
                    rec("new_quant", graph_us([lambda: dec.quantize_activation(x, permuted=(mode == 1))] * 4))
                if not args.no_n4a:
                    rec("n4a_full", graph_us([lambda c=c: nvfp4_w4a8_linear(x, c[0], c[2], gs_t, n) for c in copies]))
                    xq4, xs4 = nvfp4_w4a8_quantize_activation(x)
                    rec("n4a_gemm", graph_us([lambda c=c: nvfp4_w4a8_gemm(xq4, xs4, c[0], c[2], gs_t, n)
                                              for c in copies]))
                if not args.no_marlin:
                    rec("marlin", graph_us([lambda c=c: apply_fp4_marlin_linear(
                        input=x, weight=c[3][0], weight_scale=c[3][1], weight_global_scale=gsc_m, workspace=ws_m,
                        size_n=n, size_k=k, bias=None) for c in copies]))
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {e}"
            res["rows"].append(row)
            print(json.dumps(row), flush=True)
            dump()
        del copies
        torch.cuda.empty_cache()
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dump()


if __name__ == "__main__":
    main()
