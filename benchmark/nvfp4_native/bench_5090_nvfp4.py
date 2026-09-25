#!/usr/bin/env python3
"""5090 NVFP4 native micro-bench (Backlog #38, N4B, 25.09.).

Measures on ONE sm_120 card, per 27B NVFP4 linear shape (RadixArk
Qwen3.8-27B-NVFP4, modelopt MIXED_PRECISION: NVFP4 only on mlp gate/up/down and
lm_head) and M in {1, 8, 16, 48, 512, 4096}:

  q_fi       flashinfer fp4_quantize (the call ModelOptFp4LinearMethod.apply makes)
  q_sgl      fork JIT scaled_fp4_quant (sglang.jit_kernel.nvfp4)
  mm_sgl     fork CUTLASS cutlass_scaled_fp4_mm (what `auto` resolves on sm_120)
  mm_fi_*    flashinfer mm_fp4 backends cutlass / cudnn / b12x (whichever load)
  apply_nat  ModelOptFp4LinearMethod.apply end to end, backend=cutlass (quant+pad+gemm)
  apply_mar  ModelOptFp4LinearMethod.apply end to end, backend=marlin (W4A16)
  i8_q       per_token_quant_int8
  i8_mm      sgl_kernel int8_scaled_mm (W8A8, same N,K)
  bf16_mm    torch.mm reference (skipped when the bf16 weight is > 1.2 GB)

Timing: CUDA graph of R back-to-back calls, replayed; median of 5 replays / R.
Falls back to eager events when a graph cannot be captured (reported).
Weights are rotated over C copies so the rotated set exceeds the 96 MB L2 --
decode numbers are then DRAM-bound as in serving, not L2-flattered.

JIT: every cache is redirected by the caller (TVM_FFI_CACHE_DIR,
FLASHINFER_WORKSPACE_BASE, TRITON_CACHE_DIR, CUTE_DSL_CACHE_DIR, TMPDIR). The
script refuses to run if any of them still points under /root/.cache.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback

for _v in ("TVM_FFI_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE", "TRITON_CACHE_DIR",
           "CUTE_DSL_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
    _p = os.environ.get(_v, "")
    if not _p or _p.startswith("/root/.cache") or _p.startswith(os.path.expanduser("~/.cache")):
        sys.exit(f"refuse: {_v}={_p!r} must be set and must not point under ~/.cache")

import torch  # noqa: E402

L2_BYTES = 96 * 1024 * 1024
ROT_TARGET = 3 * L2_BYTES

# (name, N, K). P = full layer on one PP stage (the 5090 carries whole layers
# in P), D = the 5090's TP shard under the D ratio 58/25/25 over 136 units of
# 128 (intermediate 17408): 73 units -> 9344.
SHAPES = [
    ("P.gate_up", 2 * 17408, 5120),
    ("P.down", 5120, 17408),
    ("P.lm_head", 248320, 5120),
    ("D.gate_up", 2 * 9344, 5120),
    ("D.down", 5120, 9344),
]
MS = [1, 8, 16, 48, 512, 4096]
# FP8 per-tensor linears of the same checkpoint (attention q|k|v and o on 16
# layers, GDN in_proj_qkvz and out_proj on 48), full width (P form).
FP8_SHAPES = [
    ("F.qkv", [12288, 1024, 1024], 5120),
    ("F.qkvz", [2048, 2048, 6144, 6144], 5120),
    ("F.o", [5120], 6144),
]


def timed(fn, reps: int, use_graph: bool = True):
    """Return (us_per_call, mode)."""
    torch.cuda.synchronize()
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    if use_graph:
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    fn()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(reps):
                    fn()
            g.replay()
            torch.cuda.synchronize()
            ts = []
            for _ in range(5):
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                g.replay()
                b.record()
                b.synchronize()
                ts.append(a.elapsed_time(b) * 1e3 / reps)
            ts.sort()
            del g
            return ts[len(ts) // 2], "graph"
        except Exception as e:  # noqa: BLE001
            torch.cuda.synchronize()
            mode = f"eager({type(e).__name__})"
    else:
        mode = "eager"
    ts = []
    for _ in range(5):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / reps)
    ts.sort()
    return ts[len(ts) // 2], mode


class Rot:
    """Round-robin over C prepared operand sets."""

    def __init__(self, items):
        self.items = items
        self.i = 0

    def next(self):
        it = self.items[self.i]
        self.i = (self.i + 1) % len(self.items)
        return it


def make_nvfp4_layer(method, N, K, backend_name, fused=False):
    from sglang.srt.layers.quantization import fp4_utils

    fp4_utils.FP4_GEMM_RUNNER_BACKEND = fp4_utils.Fp4GemmRunnerBackend(backend_name)
    layer = torch.nn.Module()
    with torch.device("cuda"):
        method.create_weights(layer, K, [N // 2, N // 2] if fused else [N],
                              K, N, torch.bfloat16, weight_loader=None)
    # random but valid E2M1 bytes / E4M3 scales / scalars
    layer.weight.data.copy_(torch.randint(0, 256, layer.weight.shape, dtype=torch.uint8, device="cuda"))
    ws = (torch.rand(layer.weight_scale.shape, device="cuda") * 2 + 0.25).to(torch.float8_e4m3fn)
    layer.weight_scale.data.copy_(ws)
    layer.input_scale.data.fill_(0.01)
    layer.weight_scale_2.data.fill_(0.001)
    method.process_weights_after_loading(layer)
    return layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--shapes", default=",".join(s[0] for s in SHAPES))
    ap.add_argument("--ms", default=",".join(map(str, MS)))
    ap.add_argument("--fi-backends", default="cutlass,cudnn,b12x")
    ap.add_argument("--fp8-shapes", default=",".join(s[0] for s in FP8_SHAPES))
    ap.add_argument("--only-fp8", action="store_true")
    args = ap.parse_args()
    shapes = [] if args.only_fp8 else [s for s in SHAPES if s[0] in args.shapes.split(",")]
    ms = [int(x) for x in args.ms.split(",")]

    dev = torch.cuda.current_device()
    cap = torch.cuda.get_device_capability(dev)
    name = torch.cuda.get_device_name(dev)
    assert cap[0] == 12, f"this bench is for sm_120, got {cap} {name}"
    meta = {"device": name, "cap": cap, "torch": torch.__version__,
            "clock_note": "clocks/power as set on the rig (5090 400 W limit per operator)",
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        import flashinfer
        meta["flashinfer"] = flashinfer.__version__
    except Exception:  # noqa: BLE001
        meta["flashinfer"] = None
    print(json.dumps(meta), flush=True)

    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptFp4Config,
        ModelOptFp4LinearMethod,
    )
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize
    from sglang.jit_kernel.nvfp4 import cutlass_scaled_fp4_mm, scaled_fp4_quant
    from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8
    try:
        from sgl_kernel import int8_scaled_mm
    except Exception:  # noqa: BLE001
        int8_scaled_mm = None
    try:
        from flashinfer import mm_fp4
    except Exception:  # noqa: BLE001
        mm_fp4 = None

    cfg = ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=16)
    method = ModelOptFp4LinearMethod(cfg)
    rows = []

    def rec(shape, M, N, K, kern, us, mode, extra=None):
        flops = 2.0 * M * N * K
        r = {"shape": shape, "M": M, "N": N, "K": K, "kernel": kern,
             "us": round(us, 2), "tflops": round(flops / us / 1e6, 1), "mode": mode}
        if extra:
            r.update(extra)
        rows.append(r)
        print(json.dumps(r), flush=True)

    def rec_err(shape, M, kern, e):
        r = {"shape": shape, "M": M, "kernel": kern, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        rows.append(r)
        print(json.dumps(r), flush=True)

    all_ms = ms
    for shape, N, K in shapes:
        # lm_head only ever sees the sampled rows; M=4096 would be a 2 GB logits tensor.
        ms = [m for m in all_ms if m <= 512] if shape == "P.lm_head" else all_ms
        fp4_bytes = N * K // 2 + N * K // 16
        C = max(1, math.ceil(ROT_TARGET / fp4_bytes)) if shape != "P.lm_head" else 1
        C = min(C, 8)
        # --- native layers (backend cutlass) ---
        try:
            nat = [make_nvfp4_layer(method, N, K, "cutlass", fused=shape.endswith("gate_up")) for _ in range(C)]
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            rec_err(shape, 0, "make_native", e)
            nat = []
        for M in ms:
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            if nat:
                L0 = nat[0]
                # quant alone
                try:
                    us, mode = timed(lambda: fp4_quantize(x, L0.input_scale_inv), args.reps)
                    rec(shape, M, N, K, "q_fi", us, mode, {"bytes_in": M * K * 2})
                except Exception as e:  # noqa: BLE001
                    rec_err(shape, M, "q_fi", e)
                try:
                    us, mode = timed(lambda: scaled_fp4_quant(x, L0.input_scale_inv), args.reps)
                    rec(shape, M, N, K, "q_sgl", us, mode)
                except Exception as e:  # noqa: BLE001
                    rec_err(shape, M, "q_sgl", e)
                # gemm alone (fork CUTLASS)
                try:
                    xq, xs = scaled_fp4_quant(x, L0.input_scale_inv)
                    rot = Rot(nat)

                    def f_mm():
                        L = rot.next()
                        return cutlass_scaled_fp4_mm(xq, L.weight, xs, L.weight_scale_interleaved,
                                                     L.alpha, torch.bfloat16)
                    ref = f_mm()
                    us, mode = timed(f_mm, args.reps)
                    rec(shape, M, N, K, "mm_sgl", us, mode,
                        {"GBps_w": round(fp4_bytes / us / 1e3, 0), "C": C})
                except Exception as e:  # noqa: BLE001
                    rec_err(shape, M, "mm_sgl", e)
                    ref = None
                # flashinfer mm_fp4 backends
                if mm_fp4 is not None:
                    xq_fi, xs_fi = fp4_quantize(x, L0.input_scale_inv)
                    for be in [b for b in args.fi_backends.split(",") if b]:
                        try:
                            rot = Rot(nat)

                            def f_fi(be=be):
                                L = rot.next()
                                return mm_fp4(xq_fi, L.weight.T, xs_fi, L.weight_scale_interleaved.T,
                                              L.alpha, torch.bfloat16, backend=be)
                            o = f_fi()
                            agree = None
                            if ref is not None:
                                L = nat[0]
                                o0 = mm_fp4(xq_fi, L.weight.T, xs_fi, L.weight_scale_interleaved.T,
                                            L.alpha, torch.bfloat16, backend=be)
                                r0 = cutlass_scaled_fp4_mm(xq, L.weight, xs, L.weight_scale_interleaved,
                                                           L.alpha, torch.bfloat16)
                                agree = float((o0.float() - r0.float()).abs().max() /
                                              (r0.float().abs().max() + 1e-9))
                            us, mode = timed(f_fi, args.reps)
                            rec(shape, M, N, K, f"mm_fi_{be}", us, mode,
                                {"GBps_w": round(fp4_bytes / us / 1e3, 0), "rel_maxdiff_vs_sgl": agree})
                        except Exception as e:  # noqa: BLE001
                            rec_err(shape, M, f"mm_fi_{be}", e)
                # end to end native apply
                try:
                    from sglang.srt.layers.quantization import fp4_utils
                    fp4_utils.FP4_GEMM_RUNNER_BACKEND = fp4_utils.Fp4GemmRunnerBackend("cutlass")
                    rot = Rot(nat)
                    us, mode = timed(lambda: method.apply(rot.next(), x), args.reps)
                    rec(shape, M, N, K, "apply_nat", us, mode)
                except Exception as e:  # noqa: BLE001
                    rec_err(shape, M, "apply_nat", e)
            del x
        del nat
        torch.cuda.empty_cache()
        # --- marlin layers ---
        try:
            mar = [make_nvfp4_layer(method, N, K, "marlin", fused=shape.endswith("gate_up")) for _ in range(C)]
            from sglang.srt.layers.quantization import fp4_utils
            for M in ms:
                x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                fp4_utils.FP4_GEMM_RUNNER_BACKEND = fp4_utils.Fp4GemmRunnerBackend("marlin")
                rot = Rot(mar)
                try:
                    us, mode = timed(lambda: method.apply(rot.next(), x), args.reps)
                    rec(shape, M, N, K, "apply_mar", us, mode,
                        {"GBps_w": round(fp4_bytes / us / 1e3, 0)})
                except Exception as e:  # noqa: BLE001
                    rec_err(shape, M, "apply_mar", e)
            del mar
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            rec_err(shape, 0, "make_marlin", e)
        torch.cuda.empty_cache()
        # --- int8 W8A8 ---
        if int8_scaled_mm is not None:
            i8_bytes = N * K
            Ci = min(8, max(1, math.ceil(ROT_TARGET / i8_bytes))) if shape != "P.lm_head" else 1
            try:
                ws = [torch.randint(-127, 127, (N, K), dtype=torch.int8, device="cuda").t() for _ in range(Ci)]
                wsc = torch.rand(N, 1, device="cuda", dtype=torch.float32) * 0.01
                for M in ms:
                    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                    try:
                        us, mode = timed(lambda: per_token_quant_int8(x), args.reps)
                        rec(shape, M, N, K, "i8_q", us, mode)
                    except Exception as e:  # noqa: BLE001
                        rec_err(shape, M, "i8_q", e)
                    try:
                        xq8, xs8 = per_token_quant_int8(x)
                        rot = Rot(ws)
                        us, mode = timed(lambda: int8_scaled_mm(xq8, rot.next(), xs8, wsc,
                                                                out_dtype=torch.bfloat16), args.reps)
                        rec(shape, M, N, K, "i8_mm", us, mode, {"GBps_w": round(i8_bytes / us / 1e3, 0)})
                    except Exception as e:  # noqa: BLE001
                        rec_err(shape, M, "i8_mm", e)
                del ws
            except Exception as e:  # noqa: BLE001
                rec_err(shape, 0, "make_int8", e)
            torch.cuda.empty_cache()
        # --- bf16 reference ---
        if N * K * 2 <= 1.2e9:
            Cb = min(4, max(1, math.ceil(ROT_TARGET / (N * K * 2))))
            wb = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) for _ in range(Cb)]
            for M in ms:
                x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                rot = Rot(wb)
                try:
                    us, mode = timed(lambda: torch.mm(x, rot.next().t()), args.reps)
                    rec(shape, M, N, K, "bf16_mm", us, mode)
                except Exception as e:  # noqa: BLE001
                    rec_err(shape, M, "bf16_mm", e)
            del wb
            torch.cuda.empty_cache()

    # ---------------- FP8 per-tensor (ModelOptFp8LinearMethod) ----------------
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptFp8Config,
        ModelOptFp8LinearMethod,
    )
    fcfg = ModelOptFp8Config(is_checkpoint_fp8_serialized=True)
    ms = all_ms
    for shape, parts, K in [s for s in FP8_SHAPES if s[0] in args.fp8_shapes.split(",")]:
        N = sum(parts)
        wbytes = N * K
        C8 = min(8, max(1, math.ceil(ROT_TARGET / wbytes)))
        for variant, use_marlin in (("f8_nat", False), ("f8_mar", True)):
            try:
                m8 = ModelOptFp8LinearMethod(fcfg)
                m8.use_marlin = use_marlin
                lays = []
                for _ in range(C8):
                    L = torch.nn.Module()
                    with torch.device("cuda"):
                        m8.create_weights(L, K, list(parts), K, N, torch.bfloat16, weight_loader=None)
                    L.weight.data.copy_((torch.randn(N, K, device="cuda") * 0.5).to(torch.float8_e4m3fn))
                    L.weight_scale.data.fill_(0.01)
                    L.input_scale.data.fill_(0.05)
                    m8.process_weights_after_loading(L)
                    lays.append(L)
                for M in ms:
                    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                    rot = Rot(lays)
                    try:
                        us, mode = timed(lambda: m8.apply(rot.next(), x), args.reps)
                        rec(shape, M, N, K, variant, us, mode, {"GBps_w": round(wbytes / us / 1e3, 0)})
                    except Exception as e:  # noqa: BLE001
                        rec_err(shape, M, variant, e)
                del lays
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                rec_err(shape, 0, variant, e)
            torch.cuda.empty_cache()
        if int8_scaled_mm is not None:
            try:
                ws = [torch.randint(-127, 127, (N, K), dtype=torch.int8, device="cuda").t() for _ in range(C8)]
                wsc = torch.rand(N, 1, device="cuda", dtype=torch.float32) * 0.01
                for M in ms:
                    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                    rot = Rot(ws)

                    def f_i8():
                        xq8, xs8 = per_token_quant_int8(x)
                        return int8_scaled_mm(xq8, rot.next(), xs8, wsc, out_dtype=torch.bfloat16)
                    try:
                        us, mode = timed(f_i8, args.reps)
                        rec(shape, M, N, K, "i8_q+mm", us, mode, {"GBps_w": round(wbytes / us / 1e3, 0)})
                    except Exception as e:  # noqa: BLE001
                        rec_err(shape, M, "i8_q+mm", e)
                del ws
            except Exception as e:  # noqa: BLE001
                rec_err(shape, 0, "i8_q+mm", e)
            torch.cuda.empty_cache()

    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["max_mem_alloc_mib"] = int(torch.cuda.max_memory_allocated() / 2**20)
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=1)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
