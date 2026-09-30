#!/usr/bin/env python3
"""DUAL-TP3PP3 risk 1: can a decode load and a prefill load really run at the
same time on ONE card?

Two load shapes, each a stand-in for one layout of the dual layout:

* decode  -- the D rank's step: a CUDA-graph replay of 64 "layers" of skinny
             GEMMs (M = --rows, bf16, ~1.7 GB of weights read per step).
             Memory-bandwidth bound, latency sensitive.
* prefill -- the P stage's chunk: large bf16 GEMMs (M = --prefill-m).
             Tensor-core bound.

Arrangements (who shares the card):

* solo      -- one role alone (the floor for its share)
* inproc    -- one process, two threads, two streams (decode on the
               HIGH-priority stream) -- the DESIGN_121 (a) arrangement
* 2proc     -- two processes (P and D as separate processes, as weg2 runs them
               today); without MPS the driver time-slices them
* 2proc-mps -- the same two processes as clients of a private MPS daemon

The figure of merit is card equivalents E = share_decode + share_prefill with
share = rate(shared) / rate(solo): E ~ 1.0 means time-sharing (no gain over a
flip), E > 1 is real overlap.  Decode step latency p50/p99 is reported next to
it because a time slice shows up there first.

Usage (one role, synchronised start):
  pd_overlap_bench.py --role decode  --start-at <epoch> --dur 8 --out d.json
  pd_overlap_bench.py --role prefill --start-at <epoch> --dur 8 --out p.json
  pd_overlap_bench.py --role both    --start-at <epoch> --dur 8 --out b.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time

import torch


def build_decode(rows: int, layers: int, hidden: int, inter: int, stream):
    ws = []
    for _ in range(layers):
        ws.append(torch.randn(hidden, inter, dtype=torch.bfloat16, device="cuda") * 0.01)
        ws.append(torch.randn(inter, hidden, dtype=torch.bfloat16, device="cuda") * 0.01)
    x = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(x)

    def step():
        h = x
        for i in range(0, len(ws), 2):
            h = (h @ ws[i]) @ ws[i + 1]
        out.copy_(h)

    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    stream.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        step()
    stream.synchronize()
    nbytes = sum(w.numel() * w.element_size() for w in ws)
    return g, nbytes


def build_prefill(m: int, hidden: int, inter: int, stream):
    a = torch.randn(m, hidden, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(hidden, inter, dtype=torch.bfloat16, device="cuda") * 0.01
    c = torch.randn(inter, hidden, dtype=torch.bfloat16, device="cuda") * 0.01

    def step():
        (a @ b) @ c

    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    stream.synchronize()
    flops = 2 * m * hidden * inter * 2
    return step, flops


def run_decode(g, stream, t_end, res):
    lat = []
    n = 0
    while True:
        t0 = time.perf_counter()
        if t0 >= t_end:
            break
        with torch.cuda.stream(stream):
            g.replay()  # a graph replays into the CURRENT stream
        stream.synchronize()
        lat.append((time.perf_counter() - t0) * 1e3)
        n += 1
    res["decode_steps"] = n
    res["decode_lat_ms"] = lat


def run_prefill(step, stream, t_end, res, flops):
    n = 0
    t_first = time.perf_counter()
    with torch.cuda.stream(stream):
        while time.perf_counter() < t_end:
            step()
            stream.synchronize()
            n += 1
    res["prefill_iters"] = n
    res["prefill_flops"] = n * flops


def summarize(res, dur):
    out = {"dur_s": dur}
    if "decode_steps" in res:
        lat = sorted(res["decode_lat_ms"]) or [float("nan")]
        out.update(
            decode_steps_per_s=res["decode_steps"] / dur,
            decode_p50_ms=statistics.median(lat),
            decode_p99_ms=lat[min(len(lat) - 1, int(0.99 * len(lat)))],
            decode_max_ms=lat[-1],
            decode_weight_gb=res.get("decode_weight_gb"),
        )
    if "prefill_iters" in res:
        out.update(
            prefill_tflops=res["prefill_flops"] / dur / 1e12,
            prefill_iters=res["prefill_iters"],
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["decode", "prefill", "both"], required=True)
    ap.add_argument("--start-at", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=8.0)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--inter", type=int, default=2560)
    ap.add_argument("--prefill-m", type=int, default=4096)
    ap.add_argument("--prefill-inter", type=int, default=8192)
    ap.add_argument("--decode-prio", type=int, default=-1,
                    help="stream priority of decode (lower = higher prio); 0 = equal")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.cuda.init()
    lo, hi = torch.cuda.Stream.priority_range() if hasattr(torch.cuda.Stream, "priority_range") else (0, -1)
    res = {}
    threads = []
    if a.role in ("decode", "both"):
        ds = torch.cuda.Stream(priority=a.decode_prio)
        g, nb = build_decode(a.rows, a.layers, a.hidden, a.inter, ds)
        res["decode_weight_gb"] = nb / 1e9
    if a.role in ("prefill", "both"):
        ps = torch.cuda.Stream(priority=0)
        pstep, flops = build_prefill(a.prefill_m, a.hidden, a.prefill_inter, ps)
    torch.cuda.synchronize()

    now = time.time()
    if a.start_at > now:
        time.sleep(a.start_at - now)
    t_end = time.perf_counter() + a.dur
    if a.role in ("decode", "both"):
        threads.append(threading.Thread(target=run_decode, args=(g, ds, t_end, res)))
    if a.role in ("prefill", "both"):
        threads.append(threading.Thread(target=run_prefill, args=(pstep, ps, t_end, res, flops)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    out = summarize(res, a.dur)
    out.update(role=a.role, rows=a.rows, prefill_m=a.prefill_m,
               device=torch.cuda.get_device_name(0),
               mem_reserved_gb=torch.cuda.memory_reserved() / 1e9,
               wall_start=now)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
