#!/usr/bin/env python3
"""#631 Route A prefill ladder with a quiet gate, for the TP phase.

Same draw methodology as ``route_a_631_prefill_ladder.py`` (FINAL_631
section 1): random ``input_ids`` per draw so prefix caching cannot turn a
repeat into a cache hit, ``max_new_tokens=1`` so the number is prefill and
not decode, one warm-up per rung discarded, several kept draws, median
reported with the spread visible.

The difference is the QUIET GATE. This ladder is meant to run against the
live production server, which also carries real agent traffic. A prefill
draw that shares the scheduler with somebody else's decode does not measure
this server's prefill throughput -- it measures a contended mixture. So each
draw is:

  1. gated on the scheduler log going quiet (no batch line for
     ``--quiet-seconds``), and
  2. checked afterwards against the log region it produced: if any batch
     line during the draw reports another request in flight, the draw is
     marked CONTAMINATED and re-drawn (up to ``--max-retries``).

That makes the measurement honest without stopping anyone's traffic, which
matters because the agents driving this server retry rather than fail.

Stdlib only, on purpose -- it has to run against a live production server
without pulling anything into its environment.
"""

import argparse
import json
import os
import random
import re
import statistics
import time
import urllib.request

# "#running-req: 2" in a Prefill/Decode batch line. The draw's own request
# counts as one, so anything above one is foreign traffic.
RUNNING_RE = re.compile(r"#running-req:\s*(\d+)")
BATCH_RE = re.compile(r"(Prefill batch|Decode batch)")


def log_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def read_from(path: str, offset: int) -> str:
    try:
        with open(path, "r", errors="replace") as fh:
            fh.seek(offset)
            return fh.read()
    except OSError:
        return ""


def wait_quiet(path: str, quiet_seconds: float, timeout: float) -> bool:
    """Block until the scheduler log has been silent for quiet_seconds."""
    deadline = time.time() + timeout
    last_size = log_size(path)
    last_change = time.time()
    while time.time() < deadline:
        time.sleep(0.25)
        size = log_size(path)
        if size != last_size:
            last_size = size
            last_change = time.time()
        elif time.time() - last_change >= quiet_seconds:
            return True
    return False


def max_concurrency(text: str) -> int:
    """Highest #running-req seen in batch lines of this log region."""
    peak = 0
    for line in text.splitlines():
        if BATCH_RE.search(line):
            m = RUNNING_RE.search(line)
            if m:
                peak = max(peak, int(m.group(1)))
    return peak


def draw(port: int, n_tokens: int, vocab: int, timeout: float) -> float:
    """One prefill draw. Returns wall seconds for an n_tokens prefill."""
    # Random ids, fresh per draw: an uncached prefix is the whole point.
    ids = [random.randint(1000, vocab) for _ in range(n_tokens)]
    body = json.dumps(
        {
            "input_ids": ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 1,
                "ignore_eos": True,
            },
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()
    return time.perf_counter() - t0


def clean_draw(args, rung: int):
    """A quiet-gated draw. Returns (ms, peak_concurrency, contaminated)."""
    wait_quiet(args.log, args.quiet_seconds, args.quiet_timeout)
    off = log_size(args.log)
    ms = draw(args.port, rung, args.vocab, args.timeout) * 1000.0
    region = read_from(args.log, off)
    peak = max_concurrency(region)
    # peak <= 1 means only our own request was ever in flight.
    return ms, peak, peak > 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30030)
    ap.add_argument("--log", default="/spinning/serving-30030.boot.log")
    ap.add_argument("--rungs", default="2048,8192,32768")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--vocab", type=int, default=150000)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--quiet-seconds", type=float, default=2.0)
    ap.add_argument("--quiet-timeout", type=float, default=180.0)
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--phase", default="tp", help="layout label for the record")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = {
        "phase": args.phase,
        "method": (
            "random ids, max_new_tokens=1, warmup discarded, "
            "quiet-gated, contaminated draws rejected"
        ),
        "rungs": [],
    }
    print(f"phase={args.phase} port={args.port}")
    print(f"{'input tok':>10} {'median ms':>10} {'tok/s':>8}   draws (ms)   [rejected]")

    for rung in [int(x) for x in args.rungs.split(",")]:
        clean_draw(args, rung)  # warm-up, discarded
        times = []
        rejected = 0
        while len(times) < args.draws and rejected <= args.max_retries:
            ms, peak, dirty = clean_draw(args, rung)
            if dirty:
                rejected += 1
                print(f"  rejected {rung}-tok draw: peak #running-req {peak}")
                continue
            times.append(ms)
        if not times:
            print(f"  {rung}: no clean draw obtained")
            continue
        med = statistics.median(times)
        spread = (max(times) - min(times)) / med * 100.0 if len(times) > 1 else 0.0
        rec = {
            "input_tokens": rung,
            "median_ms": round(med, 1),
            "tok_s": round(rung / (med / 1000.0), 1),
            "draws_ms": [round(t, 1) for t in times],
            "spread_percent": round(spread, 2),
            "rejected_draws": rejected,
        }
        out["rungs"].append(rec)
        print(
            f"{rung:>10} {med:>10.1f} {rec['tok_s']:>8.1f}   "
            + " / ".join(f"{t:.1f}" for t in times)
            + f"   [{rejected}] spread {spread:.2f}%"
        )

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
