#!/usr/bin/env python3
"""Sample the scheduler thread's leaf frame N times and rank the leaves.

The record battery reported single py-spy dumps; task #600 reported ratios
("11/14 samples in check_aborted"), which is the instrument this window needs:
one dump cannot separate a hot leaf from a lucky one. So: N dumps, take the
LEAF (top-most) frame of the thread that owns the scheduler event loop, and
count. The denominator is the number of dumps in which that thread was found,
not the number of dumps attempted -- a dump that failed to attach is not a
sample of anything.
"""

from __future__ import annotations

import argparse
import collections
import re
import subprocess
import sys
import time

# Frames that identify the thread doing the serving work. A TP worker runs the
# scheduler event loop; anything else in the process (watchdog, tokenizer,
# metrics) is not what a decode round is spent in.
LOOP_MARKERS = (
    "event_loop_overlap",
    "event_loop_normal",
    "run_scheduler_process",
    "forward_batch_generation",
)


def dump(pyspy: str, pid: int, timeout: int = 25) -> str:
    try:
        r = subprocess.run(
            [pyspy, "dump", "--pid", str(pid)],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout
    except Exception:
        return ""


def leaf_of_loop_thread(text: str):
    """Return (leaf_frame, whole_stack) for the scheduler thread, or None."""
    blocks = re.split(r"\nThread ", text)
    for b in blocks:
        if not any(m in b for m in LOOP_MARKERS):
            continue
        frames = [ln.strip() for ln in b.splitlines() if ln.strip().startswith(("_", ))
                  or re.match(r"^\s+\S+ \(.*:\d+\)", ln)]
        frames = [ln.strip() for ln in b.splitlines() if re.match(r"^\s+\S+ \(.*:\d+\)$", ln)]
        if frames:
            return frames[0], b
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyspy", default="/spinning/htsglang-gpu/.venv/bin/py-spy")
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--interval", type=float, default=0.35)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    leaves = collections.Counter()
    stacks = {}
    found = 0
    for _ in range(args.samples):
        txt = dump(args.pyspy, args.pid)
        got = leaf_of_loop_thread(txt)
        if got:
            found += 1
            leaves[got[0]] += 1
            stacks.setdefault(got[0], got[1])
        time.sleep(args.interval)

    print(f"=== pyspy leaves {args.label} pid={args.pid} "
          f"samples_with_loop_thread={found}/{args.samples} ===")
    for leaf, n in leaves.most_common(8):
        print(f"  {n}/{found}  {leaf}")
    print("--- full stack of the top leaf ---")
    if leaves:
        top = leaves.most_common(1)[0][0]
        print(stacks.get(top, "")[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
