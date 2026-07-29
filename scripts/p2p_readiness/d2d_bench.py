#!/usr/bin/env python3
"""Directed D2D micro-bench: direct peer copy vs host staging, per pair.

Size ladder 64 KiB .. 1 GiB. The ladder ALWAYS brackets the 256-MiB
small-BAR boundary (window-1MiB / window / window+1MiB) so windowed-aperture
behaviour on the 3080 targets -- a knee, chunking cost, window switching --
is visible instead of falling between two doubling points. Per point:
time-bounded repeats (default 2 s), median + p95.

Pressure arms (the #278 methodology):
  * bidir: both directions of a pair simultaneously (two threads, own streams)
  * dual-window: ONE source writing into BOTH 3080s at once, each through its
    own small BAR window -- simultaneous pressure on both apertures. Arm is
    generated for every (src, dst_a, dst_b) with two windowed targets, so it
    needs no hardcoded card roles.

Copies above a pair's effective aperture may fail: recorded per point as a
result row with status=failed, the run continues.

Cards are PCI-identified; torch indices are joined via PCI bus id (the
device-order trap). Sets CUDA_DEVICE_ORDER=PCI_BUS_ID before torch import
for stable in-process order.

Usage:
    python d2d_bench.py --out results/<date>/d2d_bench.json \
        [--seconds-per-point 2.0] [--max-mib 1024] [--dry-run]

Runtime: ~2 s x ~14 sizes x (6 directed pairs x 2 modes + arms) -- a few
minutes. No sglang, no model, no server.
"""

import argparse
import os
import statistics
import sys
import threading
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

from p2p_common import (
    GIB,
    KIB,
    MIB,
    classify_bar,
    result_envelope,
    size_ladder,
    write_json,
)


def bench_copy(torch, src, dst, size, seconds, staged, stream=None):
    """Median/p95 seconds for one directed copy of `size` bytes."""
    a = torch.empty(size, dtype=torch.uint8, device=f"cuda:{src}")
    a.fill_(1)
    b = torch.empty(size, dtype=torch.uint8, device=f"cuda:{dst}")
    host = torch.empty(size, dtype=torch.uint8, pin_memory=True) if staged else None

    def one():
        if staged:
            host.copy_(a, non_blocking=False)
            b.copy_(host, non_blocking=False)
        else:
            b.copy_(a, non_blocking=False)
        torch.cuda.synchronize(src)
        torch.cuda.synchronize(dst)

    one()  # warmup
    times = []
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end or len(times) < 3:
        t0 = time.perf_counter()
        one()
        times.append(time.perf_counter() - t0)
        if len(times) >= 2000:
            break
    times.sort()
    return {
        "n": len(times),
        "median_s": statistics.median(times),
        "p95_s": times[int(0.95 * (len(times) - 1))],
        "gib_per_s": (size / GIB) / statistics.median(times),
    }


def run_point(torch, src, dst, size, seconds, staged):
    try:
        r = bench_copy(torch, src, dst, size, seconds, staged)
        r["status"] = "ok"
        return r
    except RuntimeError as e:
        # above the effective aperture / mapping failure: a RESULT, not an abort
        torch.cuda.empty_cache()
        return {"status": "failed", "error": str(e)}


def parallel_arm(torch, legs, size, seconds):
    """Run several directed copies simultaneously; per-leg medians under
    mutual pressure. legs: [(src, dst), ...]"""
    results = [None] * len(legs)
    barrier = threading.Barrier(len(legs))

    def worker(i, src, dst):
        try:
            barrier.wait(timeout=30)
            results[i] = bench_copy(torch, src, dst, size, seconds, staged=False)
            results[i]["status"] = "ok"
        except Exception as e:  # noqa: BLE001 -- recorded per leg
            results[i] = {"status": "failed", "error": str(e)}

    threads = [
        threading.Thread(target=worker, args=(i, s, d)) for i, (s, d) in enumerate(legs)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [
        {"src": s, "dst": d, **(r or {"status": "failed", "error": "no result"})}
        for (s, d), r in zip(legs, results)
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="d2d_bench.json")
    ap.add_argument("--seconds-per-point", type=float, default=2.0)
    ap.add_argument("--max-mib", type=int, default=1024)
    ap.add_argument(
        "--arm-size-mib",
        type=int,
        default=192,
        help="size for the pressure arms (below the nominal window)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ladder = size_ladder(64 * KIB, args.max_mib * MIB)
    if args.dry_run:
        print(
            f"ladder ({len(ladder)} points): "
            + ", ".join(
                f"{s // MIB}MiB" if s >= MIB else f"{s // KIB}KiB" for s in ladder
            )
        )
        print("modes: direct peer vs host-staged, per directed pair")
        print("arms: bidir per pair; dual-window (1 src -> both windowed dsts)")
        print(f"per point: {args.seconds_per_point}s; out: {args.out}")
        return 0

    import torch  # after CUDA_DEVICE_ORDER

    from p2p_common import cuda_pci_bus_id, nvml_devices

    devs = nvml_devices()
    by_pci = {d.pci_bus_id: d for d in devs}
    n = torch.cuda.device_count()
    idx_pci = {i: cuda_pci_bus_id(i) for i in range(n)}

    def windowed(i):
        d = by_pci.get(idx_pci[i])
        return d and classify_bar(d.bar1_total_bytes, d.vram_total_bytes) == "windowed"

    pairs = [(s, d) for s in range(n) for d in range(n) if s != d]
    results = {"pairs": [], "arms": []}

    for src, dst in pairs:
        for staged in (False, True):
            row = {
                "src": src,
                "dst": dst,
                "src_pci": idx_pci[src],
                "dst_pci": idx_pci[dst],
                "mode": "staged" if staged else "direct",
                "points": [],
            }
            for size in ladder:
                r = run_point(torch, src, dst, size, args.seconds_per_point, staged)
                r["size_bytes"] = size
                row["points"].append(r)
            results["pairs"].append(row)

    arm_size = args.arm_size_mib * MIB
    for src, dst in pairs:
        results["arms"].append(
            {
                "kind": "bidir",
                "legs": parallel_arm(
                    torch, [(src, dst), (dst, src)], arm_size, args.seconds_per_point
                ),
                "size_bytes": arm_size,
            }
        )
    windowed_idx = [i for i in range(n) if windowed(i)]
    for src in range(n):
        tgts = [i for i in windowed_idx if i != src]
        if len(tgts) >= 2:
            results["arms"].append(
                {
                    "kind": "dual-window",
                    "note": "one source into BOTH windowed targets at once "
                    "(simultaneous small-BAR aperture pressure)",
                    "legs": parallel_arm(
                        torch,
                        [(src, t) for t in tgts[:2]],
                        arm_size,
                        args.seconds_per_point,
                    ),
                    "size_bytes": arm_size,
                }
            )

    payload = result_envelope("d2d_bench")
    payload.update(
        {
            "config": vars(args),
            "cuda_index_to_pci": idx_pci,
            "devices": devs,
            **results,
        }
    )
    write_json(args.out, payload)
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
