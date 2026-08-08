#!/usr/bin/env python
"""#651 guard battery: repeated GPU sanity verdicts across load and idle phases.

A single guard pass is not evidence. The gfx1103 defect family on this laptop
has repeatedly looked clean in one probe and wrong in the next, so the question
is not "does the guard pass" but "does it keep passing across the state
transitions that were ever suspected of poisoning the GPU".

Each cycle runs:

    guard -> sustained load -> guard -> idle window -> guard

so that a defect appearing only after load, or only after idle, is separated
from a defect that is simply always there. The verdict is the whole matrix; any
single FAIL makes the battery fail.

The guard itself (gpu_sanity_guard.py) is the discriminator: Q5_K dequantize
determinism over 8 runs plus Q4_K correctness against the numpy oracle.
"""

import argparse
import subprocess
import sys
import time

GUARD = "/root/651-p2/scripts/gpu_sanity_guard.py"


def run_guard(tag: str) -> bool:
    t0 = time.perf_counter()
    p = subprocess.run(
        [sys.executable, GUARD], capture_output=True, text=True, timeout=300
    )
    dt = time.perf_counter() - t0
    ok = p.returncode == 0
    detail = " | ".join(
        line.strip()
        for line in p.stdout.splitlines()
        if line.startswith("GUARD:") and "sane" not in line
    )
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag:22s} ({dt:5.1f}s)  {detail}")
    if not ok:
        print(f"    stdout: {p.stdout.strip()[-500:]}")
        print(f"    stderr: {p.stderr.strip()[-500:]}")
    return ok


def sustained_load(seconds: float):
    """Keep the GFX core continuously fed, as the keep-busy daemon does."""
    import torch

    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < seconds:
        a = (a @ a).clamp(-1, 1)
        n += 1
    torch.cuda.synchronize()
    del a
    torch.cuda.empty_cache()
    print(f"  ... sustained load {seconds:.0f}s ({n} matmuls)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--load-s", type=float, default=15.0)
    ap.add_argument("--idle-s", type=float, default=30.0)
    args = ap.parse_args()

    results = []
    print(f"=== guard battery: {args.cycles} cycles "
          f"(load {args.load_s:.0f}s / idle {args.idle_s:.0f}s) ===")
    for c in range(1, args.cycles + 1):
        print(f"cycle {c}/{args.cycles}")
        results.append((c, "baseline", run_guard(f"c{c} baseline")))
        sustained_load(args.load_s)
        results.append((c, "after_load", run_guard(f"c{c} after load")))
        print(f"  ... idle {args.idle_s:.0f}s")
        time.sleep(args.idle_s)
        results.append((c, "after_idle", run_guard(f"c{c} after idle")))

    passed = sum(1 for _, _, ok in results if ok)
    total = len(results)
    print()
    for phase in ("baseline", "after_load", "after_idle"):
        sub = [ok for _, p, ok in results if p == phase]
        print(f"  {phase:11s}: {sum(sub)}/{len(sub)} pass")
    print(f"\nBATTERY: {passed}/{total} guard passes")
    verdict = passed == total
    print("VERDICT:", "GPU STABLE ACROSS LOAD AND IDLE" if verdict else "UNSTABLE")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
