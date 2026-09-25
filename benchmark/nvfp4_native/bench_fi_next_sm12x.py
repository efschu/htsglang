#!/usr/bin/env python3
"""sm_12x NVFP4 kernel choice micro-bench (flashinfer 0.7.0 @ 2f3bc5ac, 25.09.).

One 5090, the 27B NVFP4 linear shapes (RadixArk Qwen3.8-27B-NVFP4: NVFP4 only
on mlp gate/up/down and lm_head), M in {1, 8, 16, 32, 48, 512, 4096}. Every lane
reads the SAME native layout the fork builds for sm_120 (weight uint8 [N, K/2],
weight_scale_interleaved e4m3 128x4-swizzled, scalars), except Marlin, which
repacks:

  w4a16_native  flashinfer.mm_bf16_fp4(backend="cute-dsl-native")  (#5242), BF16 x
  w4a4_sgl      ModelOptFp4LinearMethod.apply, backend=cutlass (fork JIT CUTLASS,
                what sm_120 runs today; includes the FP4 activation quant)
  w4a4_fi_cutlass  flashinfer fp4_quantize + mm_fp4(backend="cutlass")
  w4a4_fi_b12x     flashinfer fp4_quantize + mm_fp4(backend="b12x")
  w4a16_marlin  ModelOptFp4LinearMethod.apply, backend=marlin

Timing: CUDA graph of R back-to-back calls, median of 5 replays / R. Weights are
rotated over C copies (> 3x the 96 MB L2) so decode numbers are DRAM-bound as in
serving. flashinfer lanes are autotuned once per (shape, M) under
flashinfer.autotuner.autotune() before timing (serving tunes at warmup).

Numerics: at the first M of each shape every lane is compared against an fp32
reference built from the checkpoint-form bytes (E2M1 low nibble = even element,
row-major e4m3 block scale, weight_scale_2) on the first 2048 output rows.

All JIT caches must be redirected by the caller (never ~/.cache).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback

for _v in ("TVM_FFI_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE", "TRITON_CACHE_DIR", "CUTE_DSL_CACHE_DIR"):
    _p = os.environ.get(_v, "")
    if not _p or _p.startswith("/root/.cache") or _p.startswith(os.path.expanduser("~/.cache")):
        sys.exit(f"refuse: {_v}={_p!r} must be set and must not point under ~/.cache")

import torch  # noqa: E402

L2_BYTES = 96 * 1024 * 1024
ROT_TARGET = 3 * L2_BYTES
REF_ROWS = 2048

# (name, N, K). P = full layer (the 5090 carries whole layers in P); D = the
# 5090's TP shard under the D ratio 58/25/25 over 136 units of 128 -> 9344.
SHAPES = [
    ("P.gate_up", 2 * 17408, 5120),
    ("P.down", 5120, 17408),
    ("P.lm_head", 248320, 5120),
    ("D.gate_up", 2 * 9344, 5120),
    ("D.down", 5120, 9344),
]
MS = [1, 8, 16, 32, 48, 512, 4096]
LANES = ["w4a16_native", "w4a4_sgl", "w4a4_fi_cutlass", "w4a4_fi_b12x", "w4a16_marlin"]
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def timed(fn, reps: int):
    torch.cuda.synchronize()
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
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
        del g
        mode = "graph"
    except Exception as e:  # noqa: BLE001
        torch.cuda.synchronize()
        mode = f"eager({type(e).__name__}: {str(e)[:80]})"
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
    def __init__(self, items):
        self.items, self.i = items, 0

    def next(self):
        it = self.items[self.i]
        self.i = (self.i + 1) % len(self.items)
        return it


def make_layer(method, N, K, backend, fused):
    from sglang.srt.layers.quantization import fp4_utils

    fp4_utils.FP4_GEMM_RUNNER_BACKEND = fp4_utils.Fp4GemmRunnerBackend(backend)
    layer = torch.nn.Module()
    with torch.device("cuda"):
        method.create_weights(layer, K, [N // 2, N // 2] if fused else [N], K, N, torch.bfloat16, weight_loader=None)
    g = torch.Generator(device="cuda").manual_seed(1234)
    layer.weight.data.copy_(torch.randint(0, 256, layer.weight.shape, dtype=torch.uint8, device="cuda", generator=g))
    ws = (torch.rand(layer.weight_scale.shape, device="cuda", generator=g) * 2 + 0.25).to(torch.float8_e4m3fn)
    layer.weight_scale.data.copy_(ws)
    layer.input_scale.data.fill_(0.01)
    layer.weight_scale_2.data.fill_(0.001)
    raw = (layer.weight.data[:REF_ROWS].clone(), ws[:REF_ROWS].clone())
    method.process_weights_after_loading(layer)
    return layer, raw


def reference(x, raw, wscale2=0.001):
    wq, ws = raw
    lut = torch.tensor(E2M1, device=wq.device, dtype=torch.float32)
    lo = lut[(wq & 0x0F).long()]
    hi = lut[(wq >> 4).long()]
    w = torch.stack((lo, hi), dim=-1).reshape(wq.shape[0], wq.shape[1] * 2)
    w = w * ws.float().repeat_interleave(16, dim=1) * wscale2
    return x.float() @ w.t()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--shapes", default=",".join(s[0] for s in SHAPES))
    ap.add_argument("--ms", default=",".join(map(str, MS)))
    ap.add_argument("--lanes", default=",".join(LANES))
    ap.add_argument("--deadline", type=float, default=0.0, help="epoch seconds; stop starting new lanes after it")
    ap.add_argument("--no-tune", action="store_true")
    args = ap.parse_args()
    lanes = [l for l in args.lanes.split(",") if l]
    shapes = [s for s in SHAPES if s[0] in args.shapes.split(",")]
    all_ms = [int(m) for m in args.ms.split(",")]

    import flashinfer
    from flashinfer import mm_bf16_fp4, mm_fp4
    from flashinfer.autotuner import autotune

    from sglang.srt.layers.quantization import fp4_utils
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config, ModelOptFp4LinearMethod

    dev = torch.cuda.current_device()
    meta = {
        "device": torch.cuda.get_device_name(dev),
        "cap": torch.cuda.get_device_capability(dev),
        "torch": torch.__version__,
        "flashinfer": flashinfer.__version__,
        "flashinfer_commit": getattr(__import__("flashinfer._build_meta", fromlist=["x"]), "__git_commit__", None),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        from importlib.metadata import version

        meta["cutlass_dsl"] = version("nvidia-cutlass-dsl")
    except Exception:  # noqa: BLE001
        pass
    print(json.dumps(meta), flush=True)
    assert meta["cap"][0] == 12, f"sm_12x only, got {meta['cap']}"
    cap_mib = int(os.environ.get("BENCH_VRAM_MIB", "0") or 0)
    if cap_mib:  # stay inside the booked share of the card
        total = torch.cuda.get_device_properties(dev).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap_mib * 2**20 / total), dev)
        meta["vram_cap_mib"] = cap_mib
    fh = open(args.out, "a")
    fh.write(json.dumps({"meta": meta}) + "\n")

    method = ModelOptFp4LinearMethod(ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=16))

    def rec(row):
        print(json.dumps(row), flush=True)
        fh.write(json.dumps(row) + "\n")
        fh.flush()

    def late():
        return args.deadline and time.time() > args.deadline

    for shape, N, K in shapes:
        ms = [m for m in all_ms if m <= 512] if shape == "P.lm_head" else all_ms
        fp4_bytes = N * K // 2 + N * K // 16
        C = 1 if shape == "P.lm_head" else min(8, max(1, math.ceil(ROT_TARGET / fp4_bytes)))
        fused = shape.endswith("gate_up")
        nat = []
        if any(l != "w4a16_marlin" for l in lanes):
            nat = [make_layer(method, N, K, "cutlass", fused) for _ in range(C)]
        mar, raw_mar = [], None
        if "w4a16_marlin" in lanes:
            made = [make_layer(method, N, K, "marlin", fused) for _ in range(C)]
            mar, raw_mar = [m[0] for m in made], made[0][1]
        for L, _ in nat:
            L.alpha_w4a16 = L.weight_scale_2.max().to(torch.float32).reshape(1)
        for mi, M in enumerate(ms):
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            ref = reference(x, (nat[0][1] if nat else raw_mar)) if mi == 0 else None
            fns = {}
            if nat:
                rn = Rot([l for l, _ in nat])

                def f_nat(rn=rn):
                    L = rn.next()
                    return mm_bf16_fp4(x, L.weight, L.weight_scale_interleaved, L.alpha_w4a16, backend="cute-dsl-native")

                def f_sgl(rn=rn):
                    fp4_utils.FP4_GEMM_RUNNER_BACKEND = fp4_utils.Fp4GemmRunnerBackend("cutlass")
                    return method.apply(rn.next(), x)

                def mk_fi(be, rn=rn):
                    def f():
                        L = rn.next()
                        xq, xs = fp4_quantize(x, L.input_scale_inv)
                        return mm_fp4(xq, L.weight.T, xs, L.weight_scale_interleaved.T, L.alpha, torch.bfloat16, backend=be)

                    return f

                fns.update(
                    w4a16_native=f_nat, w4a4_sgl=f_sgl, w4a4_fi_cutlass=mk_fi("cutlass"), w4a4_fi_b12x=mk_fi("b12x")
                )
            if mar:
                rm = Rot(mar)

                def f_mar(rm=rm):
                    fp4_utils.FP4_GEMM_RUNNER_BACKEND = fp4_utils.Fp4GemmRunnerBackend("marlin")
                    return method.apply(rm.next(), x)

                fns["w4a16_marlin"] = f_mar
            for lane in lanes:
                if lane not in fns:
                    continue
                if late():
                    rec({"shape": shape, "M": M, "lane": lane, "skipped": "deadline"})
                    continue
                fn = fns[lane]
                row = {"shape": shape, "M": M, "N": N, "K": K, "lane": lane, "C": C}
                try:
                    t0 = time.time()
                    if lane.startswith(("w4a16_native", "w4a4_fi")) and not args.no_tune:
                        with autotune(True):
                            fn()
                        torch.cuda.synchronize()
                    row["first_call_s"] = round(time.time() - t0, 2)
                    if ref is not None:
                        # every copy (and the Marlin copies) holds the same seeded bytes
                        o = fn().float()[:, :REF_ROWS]
                        row["rel_err_vs_ref"] = float((o - ref).norm() / (ref.norm() + 1e-9))
                    us, mode = timed(fn, args.reps)
                    row.update(us=round(us, 2), tflops=round(2.0 * M * N * K / us / 1e6, 2),
                               GBps_w=round(fp4_bytes / us / 1e3, 0), mode=mode)
                except Exception as e:  # noqa: BLE001
                    row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                    traceback.print_exc()
                rec(row)
            del x
        del nat, mar
        torch.cuda.empty_cache()
    rec({"done": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


if __name__ == "__main__":
    main()
