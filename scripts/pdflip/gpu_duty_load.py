#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""A foreign load on ONE card, duty-cycled: the 23.10. desktop-load stand-in.

barlink-race-1023: the WRONG RESULT of 23.10. 10:50Z (pair 0,2) appeared while
card 2 carried the user's game (NVML 38 % utilization, 84-211 W, 1.2 GB). A
time-sliced card delays its own rank's host-visible completions by up to a
slice, so the two ranks of a bench see the same collective at different
offsets. This script makes that condition on demand, so a repro does not wait
for somebody to start a game:

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \\
        python3 scripts/pdflip/gpu_duty_load.py --on-ms 6 --off-ms 9 --mib 1200

It runs back-to-back fp16 matmuls for ``--on-ms``, then sleeps ``--off-ms``
(6/9 ~ 40 % busy, the specimen's figure), holding ``--mib`` of VRAM, until
``--secs`` elapse or it is killed. It touches only the card it is given; it is
a load, not a measurement, and it prints one line per 10 s so a log shows it
was alive. GPU access still goes through the gpuq window of the run it serves.
"""
from __future__ import annotations

import argparse
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--on-ms", type=float, default=6.0)
    ap.add_argument("--off-ms", type=float, default=9.0)
    ap.add_argument("--mib", type=int, default=1200, help="VRAM to hold")
    ap.add_argument("--n", type=int, default=2048, help="matmul edge")
    ap.add_argument("--secs", type=float, default=900.0)
    a = ap.parse_args()

    import torch

    dev = torch.device("cuda", 0)
    hold = torch.empty(max(0, a.mib) << 20, dtype=torch.uint8, device=dev)
    x = torch.randn(a.n, a.n, dtype=torch.float16, device=dev)
    y = torch.randn(a.n, a.n, dtype=torch.float16, device=dev)
    end = time.monotonic() + a.secs
    last = time.monotonic()
    busy = idle = 0.0
    while time.monotonic() < end:
        t0 = time.monotonic()
        while (time.monotonic() - t0) * 1e3 < a.on_ms:
            x = (x @ y).clamp_(-1, 1)
            torch.cuda.synchronize(dev)
        t1 = time.monotonic()
        time.sleep(a.off_ms / 1e3)
        t2 = time.monotonic()
        busy += t1 - t0
        idle += t2 - t1
        if t2 - last >= 10.0:
            print(f"# duty load: {100 * busy / max(1e-9, busy + idle):.0f} % busy, "
                  f"{hold.numel() >> 20} MiB held", flush=True)
            last, busy, idle = t2, 0.0, 0.0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
