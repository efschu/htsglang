# SPDX-License-Identifier: Apache-2.0
"""#355: what the masked KV writer's in-kernel bound check costs.

``masked_set_kv_buffer_kernel`` runs on every target-side DCP write, i.e. once
per full-attention layer per decode step. A guard on that path has to be paid
for out of the decode budget, so it gets measured rather than asserted about.

The A/B is exact: ONE kernel source, launched with ``debug=True`` (the
``tl.device_assert`` is lowered) and ``debug=False`` (it lowers to nothing, so
the compiled kernel is what the pre-#355 code was, modulo an unused scalar
argument). Nothing else differs -- same buffers, same grid, same launch.

Method:
  * CUDA-event timing around a loop of ``--iters`` launches, so the per-call
    number carries launch overhead exactly as production does;
  * ``--reps`` repeats of that loop, reported as median and inter-quartile
    range, and the arms are INTERLEAVED (A B A B ...) so clock drift lands on
    both;
  * an A-vs-A arm first (the same ``debug=False`` kernel against itself) to
    establish the noise floor. Any A-vs-B delta below that floor is not
    reportable.

Default shape is the Qwen3.6-27B TP=3 decode row: 4 replicated KV heads x
head_dim 256, fp8 KV cache.

    python benchmark/kernels/bench_masked_kv_bound_check.py
    python benchmark/kernels/bench_masked_kv_bound_check.py --dtype bf16 --n 1,4,16,64,256
"""

import argparse
import statistics

import torch

from sglang.srt.mem_cache.memory_pool import masked_set_kv_buffer_kernel

DTYPES = {
    "fp8": torch.float8_e4m3fn,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def _launch(k, v, kbuf, vbuf, loc, mask, bound, n, h, d, debug):
    masked_set_kv_buffer_kernel[(n,)](
        k,
        v,
        kbuf,
        vbuf,
        loc,
        mask,
        bound,
        n,
        h,
        d,
        128,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        debug=debug,
    )


def time_arm(bufs, n, h, d, debug, iters):
    k, v, kbuf, vbuf, loc, mask, bound = bufs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        _launch(k, v, kbuf, vbuf, loc, mask, bound, n, h, d, debug)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e6 / iters  # ns per call


def make_bufs(n, h, d, dtype, rows, device):
    k = torch.zeros(n, h, d, dtype=dtype, device=device)
    v = torch.zeros(n, h, d, dtype=dtype, device=device)
    kbuf = torch.zeros(rows, h, d, dtype=dtype, device=device)
    vbuf = torch.zeros(rows, h, d, dtype=dtype, device=device)
    loc = torch.arange(n, dtype=torch.int64, device=device) % rows
    mask = torch.ones(n, dtype=torch.bool, device=device)
    return k, v, kbuf, vbuf, loc, mask, rows


def summarize(samples):
    s = sorted(samples)
    q1 = s[len(s) // 4]
    q3 = s[(3 * len(s)) // 4]
    return statistics.median(s), q3 - q1


def run_shape(n, h, d, dtype, rows, iters, reps, device):
    bufs = make_bufs(n, h, d, dtype, rows, device)
    # Warm both compilations and the allocator out of the measurement.
    for debug in (False, True):
        for _ in range(20):
            _launch(*bufs[:6], bufs[6], n, h, d, debug)
    torch.cuda.synchronize()

    a0, a1, b = [], [], []
    for _ in range(reps):
        # A-vs-A noise floor and A-vs-B, all interleaved.
        a0.append(time_arm(bufs, n, h, d, False, iters))
        b.append(time_arm(bufs, n, h, d, True, iters))
        a1.append(time_arm(bufs, n, h, d, False, iters))
    return a0, a1, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="1,4,16,64,256", help="tokens per launch")
    ap.add_argument("--heads", type=int, default=4, help="replicated KV heads")
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--dtype", default="fp8", choices=sorted(DTYPES))
    ap.add_argument("--rows", type=int, default=65536, help="KV buffer rows")
    ap.add_argument("--iters", type=int, default=200, help="launches per timed loop")
    ap.add_argument("--reps", type=int, default=100, help="timed loops per arm")
    args = ap.parse_args()

    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    dtype = DTYPES[args.dtype]
    print(f"card: {name}   dtype: {args.dtype}   H={args.heads} D={args.head_dim}")
    print(f"iters/loop={args.iters}  reps={args.reps}  rows={args.rows}")
    print(
        f"{'N':>6} {'off ns':>10} {'on ns':>10} {'delta ns':>10} "
        f"{'delta %':>8} {'noise ns':>9} {'verdict':>12}"
    )

    for n in [int(x) for x in args.n.split(",")]:
        a0, a1, b = run_shape(
            n, args.heads, args.head_dim, dtype, args.rows, args.iters, args.reps, dev
        )
        m_a0, _ = summarize(a0)
        m_a1, _ = summarize(a1)
        m_b, _ = summarize(b)
        off = (m_a0 + m_a1) / 2
        noise = abs(m_a1 - m_a0)  # A-vs-A: the floor this run can resolve
        delta = m_b - off
        verdict = "below noise" if abs(delta) <= noise else "MEASURABLE"
        print(
            f"{n:>6} {off:>10.1f} {m_b:>10.1f} {delta:>+10.1f} "
            f"{100 * delta / off:>+7.2f}% {noise:>9.1f} {verdict:>12}"
        )


if __name__ == "__main__":
    main()
