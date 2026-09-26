#!/usr/bin/env python3
"""N4D config sweep of the W4A8 decode GEMV on one RTX 3080: every (mode, kw, rw, u) per 27B shape and M.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<nvml 3080> python n4d_sweep.py <out.json> [--shapes ..] [--ms ..]

GEMM-only time (CUDA graph over weight copies rotated beyond L2), plus a correctness gate per config against the
fp64 W4A8 emulation (max |err| / max|ref| must stay below 1e-2 -- bf16 output rounding is ~4e-3).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import n4d_decode_bench as B  # noqa: E402
from sglang.jit_kernel import nvfp4_w4a8_decode as dec  # noqa: E402
from sglang.jit_kernel.nvfp4_w4a8 import nvfp4_w4a8_quantize_activation  # noqa: E402
from sglang.srt.layers.quantization import nvfp4_native_mixed as nm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--shapes", default="D.gate_up,D.down,P.down,D.lm_head,P.gate_up")
    ap.add_argument("--ms", default="1,4,8,16,32,48")
    ap.add_argument("--min-mb", type=float, default=48.0)
    args = ap.parse_args()
    dev = B.dev
    res = {"device": torch.cuda.get_device_name(0), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "rows": []}
    g = torch.Generator(device=dev).manual_seed(7)
    gs_t = torch.tensor([B.GS], dtype=torch.float32, device=dev)
    pairs = [(kw, rw) for kw in (1, 2, 4, 8) for rw in (1, 2, 4, 8) if kw * rw <= 8]
    for name in args.shapes.split(","):
        n, k = B.SHAPES[name]
        nbytes = n * k // 2 + n * k // 16
        R = max(1, -(-int(args.min_mb * 1e6) // nbytes))
        copies = []
        for _ in range(R):
            w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev, generator=g)
            raw = torch.randint(0x28, 0x48, (n, k // 16), dtype=torch.uint8, device=dev, generator=g)
            copies.append((w, raw, nm.swizzle_128x4(raw).contiguous().view(torch.float8_e4m3fn)))
        w0, raw0, sw0 = copies[0]
        for M in (int(v) for v in args.ms.split(",")):
            x = torch.randn((M, k), dtype=torch.bfloat16, device=dev, generator=g)
            xq, xs = nvfp4_w4a8_quantize_activation(x)
            ref8 = torch.cat([(xq.double() * xs.double().unsqueeze(1)) @ B.dequant_rows(w0[r:r + 4096], raw0[r:r + 4096]).T
                              for r in range(0, n, 4096)], dim=1) * B.GS
            outb = torch.empty((M, n), dtype=torch.bfloat16, device=dev)
            row = {"shape": name, "M": M, "bytes": nbytes, "cfg_us": {}, "bad": []}
            modes = (0, 1) if M <= 8 else (1,)
            for md in modes:
                xqd, xsd = dec.quantize_activation(x, permuted=(md == 1))
                for (kw, rw), u in itertools.product(pairs, (1, 2, 4)):
                    cfg = (md, kw, rw, u)
                    key = ",".join(map(str, cfg))
                    try:
                        y = dec.decode_gemm(xqd, xsd, w0, sw0, gs_t, n, cfg)
                        e = float((y.double() - ref8).abs().max() / ref8.abs().max())
                        if not e < 1e-2:
                            row["bad"].append([key, e])
                            continue
                        us = B.graph_us([lambda c=c, cfg=cfg: dec.decode_gemm(xqd, xsd, c[0], c[2], gs_t, n, cfg, out=outb)
                                         for c in copies], iters=10)
                        row["cfg_us"][key] = round(us, 2)
                    except Exception as ex:  # noqa: BLE001
                        row["bad"].append([key, f"{type(ex).__name__}: {ex}"[:200]])
            best = sorted(row["cfg_us"].items(), key=lambda kv: kv[1])[:5]
            row["best"] = best
            row["default_cfg"] = ",".join(map(str, dec.config_for(M, n, k)))
            row["default_us"] = row["cfg_us"].get(row["default_cfg"])
            res["rows"].append(row)
            print(json.dumps({k2: row[k2] for k2 in ("shape", "M", "best", "default_cfg", "default_us")}),
                  "bad:", len(row["bad"]), flush=True)
            with open(args.out, "w") as f:
                json.dump(res, f, indent=1)
        del copies
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
