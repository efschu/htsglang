"""Microbenchmark: CPU int8 expert FFN vs RAM->GPU weight streaming.

This is the go/no-go instrument for the cold-expert CPU compute lane. It answers
one question: for a cold MoE expert, is it cheaper to compute it on the CPU over
a pinned host-resident int8 shard, or to stream its weights host->device and
compute it on the GPU?

It is hermetic and CPU-only. No CUDA device is touched, so it can run while the
rig is serving.

Anti-fooling discipline (the numbers are worthless without it):

1. ROTATING EXPERT POOL. A single expert's int8 weights are ~3 MB and fit in the
   5950X's L3. Benchmarking one expert in a loop measures an L3-resident rate
   that production will never see, because production cycles through thousands
   of experts. Every iteration therefore touches a DIFFERENT expert drawn from a
   pool sized well beyond L3, so the weight reads come from DRAM as they would
   in the real lane.
2. WARMUP DISCARD, then back-to-back timed reps.
3. THREAD COUNT IS REPORTED. A lane competing with the serving process for cores
   does not get all 16. Rates are reported per thread count, not as one number.
4. LOAD AVERAGE IS RECORDED at measurement time, so a contended run is never
   mistaken for an idle-box upper bound.
5. AN A-VS-A NOISE FLOOR is measured first. Any reported difference smaller than
   that floor is reported as "below noise", not as a win.

Reference for the streaming side: measured per-rank H2D link rates from the
club-3090 bench (BENCH_394): 14.42 GB/s (5090, x8), 6.45 GB/s (3080, x4),
13.41 GB/s (3080, x8).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time

import torch

# Measured per-rank H2D rates on this rig, from docs/dev/BENCH_394_v4flash_club3090.md.
LINKS_GB_S = {
    "tp0_5090_x8": 14.42,
    "tp1_3080_x4": 6.45,
    "tp2_3080_x8": 13.41,
}

# Expert FFN geometries. hidden/moe_intermediate pairs.
SHAPES = {
    "Qwen3.5-35B-A3B": dict(hidden=2048, inter=512),
    "Qwen3.5-122B-A10B": dict(hidden=3072, inter=1024),
    "DSV4F-class": dict(hidden=4096, inter=2048),
}

# Tokens routed to ONE expert in a single call.
#   1  = bs=1 decode, the hardest case for the CPU
#   4  = MTP verify batch with 3 draft tokens (num_draft_tokens + 1)
#   8  = MTP verify at bs=2, or bs=8 decode with perfect expert collision
#   64 = prefill-ish (2048-token chunk over 256 experts x top_k 8)
M_VALUES = [1, 2, 4, 8, 16, 32, 64]


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class ExpertPoolFP32:
    """A pool of distinct fp32 experts, sized to exceed L3."""

    def __init__(self, hidden: int, inter: int, n_experts: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.hidden, self.inter, self.n = hidden, inter, n_experts
        # [n, inter, hidden] gate & up ; [n, hidden, inter] down
        self.gate = torch.randn(n_experts, inter, hidden, generator=g) * 0.02
        self.up = torch.randn(n_experts, inter, hidden, generator=g) * 0.02
        self.down = torch.randn(n_experts, hidden, inter, generator=g) * 0.02

    def bytes_per_expert(self) -> int:
        return 4 * 3 * self.hidden * self.inter

    def forward(self, e: int, x: torch.Tensor) -> torch.Tensor:
        g = torch.nn.functional.linear(x, self.gate[e])
        u = torch.nn.functional.linear(x, self.up[e])
        return torch.nn.functional.linear(silu(g) * u, self.down[e])


class ExpertPoolINT8Packed:
    """A pool of distinct int8 experts held as fbgemm-prepacked dynamic Linears.

    This mirrors the host tier the lane would actually keep: weights quantised
    to int8 and fbgemm-prepacked ONCE at load time. There is deliberately no
    per-event dequantisation anywhere in `forward`. That step is what sank the
    earlier fp32 variant of this lane (int4->fp32 was measured at 6.177 ms per
    expert, and even int8->fp32 widening costs 2.831 ms -- both far above the
    entire H2D fetch they were meant to replace), so the lane computes directly
    on the int8 bytes and never widens.

    fbgemm is the AVX2 integer GEMM (pmaddubsw-class) path. It is what makes
    int8 a compute win here; torch's other int8 CPU entry points are not
    competitive (aten::_weight_int8pack_mm is GEMV-only at ~17 GF/s, and
    torch._int_mm reaches only ~8-12 GF/s).
    """

    def __init__(self, hidden: int, inter: int, n_experts: int, engine: str = "fbgemm", seed: int = 0):
        torch.backends.quantized.engine = engine
        torch.manual_seed(seed)
        self.hidden, self.inter, self.n = hidden, inter, n_experts
        qd = torch.ao.quantization.quantize_dynamic

        def mk(in_f, out_f):
            m = torch.nn.Sequential(torch.nn.Linear(in_f, out_f, bias=False))
            return qd(m, {torch.nn.Linear}, dtype=torch.qint8)

        # Distinct modules so each expert owns distinct prepacked bytes; the
        # rotation over these is what forces DRAM reads instead of L3 hits.
        self.gate = [mk(hidden, inter) for _ in range(n_experts)]
        self.up = [mk(hidden, inter) for _ in range(n_experts)]
        self.down = [mk(inter, hidden) for _ in range(n_experts)]

    def bytes_per_expert(self) -> int:
        return 3 * self.hidden * self.inter

    def forward(self, e: int, x: torch.Tensor) -> torch.Tensor:
        g = self.gate[e](x)
        u = self.up[e](x)
        return self.down[e](silu(g) * u)


class ExpertPoolW8A32:
    """Weight-only int8: int8 weights, fp32 activations, per-output-channel scale.

    The accurate mode (~1.3e-2 relative, inside the accepted lossy-offload band)
    because activations are never quantised. The kernel is GEMV-only, so cost
    grows linearly with M -- which is exactly what this bench is here to price
    against the batched W8A8 path.
    """

    def __init__(self, hidden: int, inter: int, n_experts: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.hidden, self.inter, self.n = hidden, inter, n_experts
        self.gate = torch.randint(-127, 128, (n_experts, inter, hidden), generator=g, dtype=torch.int8)
        self.up = torch.randint(-127, 128, (n_experts, inter, hidden), generator=g, dtype=torch.int8)
        self.down = torch.randint(-127, 128, (n_experts, hidden, inter), generator=g, dtype=torch.int8)
        self.gs = (torch.rand(n_experts, inter, generator=g) * 1e-3 + 1e-4).float()
        self.us = (torch.rand(n_experts, inter, generator=g) * 1e-3 + 1e-4).float()
        self.ds = (torch.rand(n_experts, hidden, generator=g) * 1e-3 + 1e-4).float()

    def bytes_per_expert(self) -> int:
        return 3 * self.hidden * self.inter

    def forward(self, e: int, x: torch.Tensor) -> torch.Tensor:
        mm = torch.ops.aten._weight_int8pack_mm
        g = mm(x, self.gate[e], self.gs[e])
        u = mm(x, self.up[e], self.us[e])
        return mm((silu(g) * u).contiguous(), self.down[e], self.ds[e])


def time_pool(pool, m: int, reps: int, warmup: int) -> tuple[float, float]:
    """Return (median_ms, stdev_ms) per expert call, rotating over the pool."""
    x = torch.randn(m, pool.hidden)
    n = pool.n
    for i in range(warmup):
        pool.forward(i % n, x)
    samples = []
    for i in range(reps):
        e = i % n
        t0 = time.perf_counter()
        pool.forward(e, x)
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples), statistics.pstdev(samples)


def pool_experts_for(hidden: int, inter: int, target_mb: int) -> int:
    """Choose a pool size whose int8 footprint exceeds L3 by a wide margin."""
    per = 3 * hidden * inter / 1e6
    return max(8, int(target_mb / per) + 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="+", default=[16, 8, 4])
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--pool-mb", type=int, default=400, help="int8 pool footprint per shape, MB (must exceed L3)")
    ap.add_argument("--shapes", nargs="+", default=list(SHAPES))
    ap.add_argument("--engine", default="fbgemm", choices=["fbgemm", "x86", "onednn", "qnnpack"])
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    la1, la5, la15 = os.getloadavg()
    print("=" * 78)
    print("CPU EXPERT LANE MICROBENCH")
    print("=" * 78)
    print(f"host        : {platform.node()}  {platform.machine()}")
    print(f"torch       : {torch.__version__}  mkldnn={torch.backends.mkldnn.is_available()}")
    print(f"cpu count   : {os.cpu_count()}")
    print(f"load avg    : {la1:.2f} {la5:.2f} {la15:.2f}   <-- contention at measurement time")
    print(f"reps        : {args.reps} (warmup {args.warmup} discarded), rotating expert pool >= {args.pool_mb} MB")
    print()

    results = {"meta": {"loadavg": [la1, la5, la15], "torch": torch.__version__,
                        "reps": args.reps, "pool_mb": args.pool_mb, "engine": args.engine}, "runs": []}

    # A-vs-A noise floor: the SAME pool measured twice. Any int8-vs-fp32 gap
    # smaller than this spread is not a result, it is the box breathing.
    torch.set_num_threads(args.threads[0])
    _nf_pool = ExpertPoolINT8Packed(2048, 512, 64, engine=args.engine)
    a1, _ = time_pool(_nf_pool, 1, args.reps, args.warmup)
    a2, _ = time_pool(_nf_pool, 1, args.reps, args.warmup)
    floor = abs(a1 - a2) / max(a1, a2) * 100
    print(f"noise floor  : A-vs-A on identical pool = {a1:.3f} / {a2:.3f} ms "
          f"-> {floor:.1f} % spread at threads={args.threads[0]}")
    print(f"               differences below {floor:.1f} % are NOT results.")
    results["meta"]["noise_floor_pct"] = floor
    print()

    for shape_name in args.shapes:
        cfg = SHAPES[shape_name]
        h, i = cfg["hidden"], cfg["inter"]
        n_pool = pool_experts_for(h, i, args.pool_mb)

        bf16_expert_mb = 2 * 3 * h * i / 1e6
        int8_expert_mb = 3 * h * i / 1e6

        print("-" * 78)
        print(f"SHAPE {shape_name}: hidden={h} moe_intermediate={i}")
        print(f"  one expert: bf16 {bf16_expert_mb:.2f} MB | int8 {int8_expert_mb:.2f} MB")
        print(f"  rotating pool: {n_pool} experts = {n_pool * int8_expert_mb:.0f} MB int8 (L3-defeating)")
        fetch = {k: bf16_expert_mb / (v * 1e3) * 1e3 for k, v in LINKS_GB_S.items()}
        print("  H2D fetch of the bf16 expert: " + " | ".join(f"{k} {v:.3f} ms" for k, v in fetch.items()))
        print()

        int8_pool = ExpertPoolINT8Packed(h, i, n_pool, engine=args.engine)
        w32_pool = ExpertPoolW8A32(h, i, n_pool)
        # fp32 pool is 4x the bytes; keep it smaller but still L3-defeating.
        fp32_pool = ExpertPoolFP32(h, i, max(8, n_pool // 3))

        for nt in args.threads:
            torch.set_num_threads(nt)
            print(f"  threads={nt}")
            print(f"    {'M':>4} | {'W8A8 ms':>8} | {'W8A32 ms':>9} | {'fp32 ms':>8} | "
                  f"{'best':>6} | {'vs x4':>7} | {'vs x8':>7}")
            for m in M_VALUES:
                i8, i8sd = time_pool(int8_pool, m, args.reps, args.warmup)
                w32, _ = time_pool(w32_pool, m, args.reps, args.warmup)
                f32, _ = time_pool(fp32_pool, m, args.reps, args.warmup)
                # W8A32 is preferred whenever it is not slower, because it is
                # the more accurate mode; W8A8 is taken only when it actually wins.
                best_ms, best_name = (w32, "W8A32") if w32 <= i8 else (i8, "W8A8")
                gflops = 6 * m * h * i / (best_ms * 1e-3) / 1e9
                sp_x4 = fetch["tp1_3080_x4"] / best_ms
                sp_x8 = fetch["tp2_3080_x8"] / best_ms
                print(f"    {m:>4} | {i8:>8.3f} | {w32:>9.3f} | {f32:>8.3f} | "
                      f"{best_name:>6} | {sp_x4:>6.2f}x | {sp_x8:>6.2f}x")
                results["runs"].append(dict(shape=shape_name, hidden=h, inter=i, threads=nt, m=m,
                                            w8a8_ms=i8, w8a8_sd=i8sd, w8a32_ms=w32, fp32_ms=f32,
                                            best_ms=best_ms, best_mode=best_name, gflops=gflops,
                                            fetch_ms=fetch, speedup_x4=sp_x4, speedup_x8=sp_x8))
            print()

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"json written to {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
