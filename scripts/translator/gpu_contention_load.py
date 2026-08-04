# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""A synthetic SM load on one card, as a PROXY for co-tenancy.

WHAT THIS IS NOT. It is not the 27B on port 30030. It does not reproduce that
server's memory traffic, its kernel mix, or its scheduling behaviour, and a
number measured against it must never be reported as "the cost of sharing the
card with the 27B". The real co-tenancy arm needs a window on 30030 and is
kept separate for that reason.

WHAT IT IS FOR. The talker's per-step cost is dominated by host and launch
overhead rather than by arithmetic (measured: 22.6 ms for a 28-layer, hidden
1024 decode step that is ~0.4 ms memory-bound on this card). Whether such a
step degrades gracefully or catastrophically when another process is
occupying the SMs is a property of the MECHANISM, and that can be established
with any competing kernel stream. This provides one, at a controllable
intensity, so the sensitivity is measured rather than assumed.

The load is a steady stream of medium GEMMs -- large enough to occupy the
device, small enough that the queue stays deep rather than bursty.

    CUDA_VISIBLE_DEVICES=<uuid> python gpu_contention_load.py --seconds 200
"""

from __future__ import annotations

import argparse
import json
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=200.0)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--sleep-ms", type=float, default=0.0,
                        help="idle between iterations, to dial intensity down")
    args = parser.parse_args()

    import torch

    device = "cuda:0"
    dtype = getattr(torch, args.dtype)
    a = torch.randn(args.size, args.size, device=device, dtype=dtype)
    b = torch.randn(args.size, args.size, device=device, dtype=dtype)
    torch.cuda.synchronize()
    print(json.dumps({"event": "contention_started", "size": args.size}))
    started = time.time()
    iterations = 0
    try:
        while time.time() - started < args.seconds:
            for _ in range(20):
                a = torch.mm(a, b)
                # Keep the values bounded so this cannot turn into inf/NaN
                # work with a different cost profile than real GEMMs.
                a = a * 0.001
            iterations += 20
            if args.sleep_ms:
                torch.cuda.synchronize()
                time.sleep(args.sleep_ms / 1000.0)
    finally:
        torch.cuda.synchronize()
    elapsed = time.time() - started
    print(
        json.dumps(
            {
                "event": "contention_stopped",
                "seconds": round(elapsed, 1),
                "iterations": iterations,
                "gemm_per_s": round(iterations / elapsed, 1),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
