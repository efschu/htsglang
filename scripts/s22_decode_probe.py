#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#631/#656 decode decomposition: IN-PHASE tok/s vs wall-clock tok/s.

The open question is why aggregate decode throughput looks poor against
plain TP3 when the TP decode phase itself may be fine. Those are different
numbers and the flip setup is the only configuration where they can
diverge: a request's decode only advances inside TP windows, so
wall-clock throughput is in-phase throughput multiplied by the TP duty
cycle. Reporting only the first blames the flip for a scheduling property;
reporting only the second hides a genuine decode regression.

This driver measures both from the SAME requests:

  wall tok/s      = completion_tokens / (end - start)
  in-phase tok/s  = completion_tokens / (time actually spent in TP)

The TP occupancy comes from the server log's own phase transitions, so it
is the server's account of where the work ran, not an inference from
timing. Run it against a live instance; it does not reconfigure anything.

WHY LONG GENERATIONS ARE MANDATORY: decode batch lines are emitted every
`decode_log_interval` iterations (default 40). A driver asking for 8 tokens
produces ZERO decode evidence, which is why earlier runs in this chain
showed an empty decode set and it was mistaken for a decode defect.
"""
import argparse
import json
import re
import statistics
import subprocess
import threading
import time
import urllib.request

MIB = 1024 * 1024


def post(port, payload, timeout):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    return t0, time.time(), body


def tp_seconds(log, t_start, t_end):
    """Seconds inside TP windows between two wall instants, from the log."""
    # The prefix carries a RANK TAG between the time and the bracket:
    #   [2026-08-10 10:47:30 PP2] PHASE-FLIP DONE pp_to_tp
    # The original pattern required "]" immediately after the time, so it
    # matched NOTHING, `events` stayed empty, `in_tp` stayed False, and this
    # function returned 0.0 -- reported as "0% TP duty cycle" on a run whose
    # log held 1026 of these lines. A duty cycle of exactly zero is
    # impossible under strict purity, where decode may ONLY run in TP, so
    # the instrument was contradicting a boot-enforced invariant.
    out = subprocess.run(
        ["grep", "-oE",
         r"[0-9-]+ [0-9:]+ [A-Za-z0-9_]+\] PHASE-FLIP DONE (pp_to_tp|tp_to_pp)",
         log],
        capture_output=True, text=True, timeout=120,
    ).stdout.splitlines()
    # ONE FLIP IS LOGGED ONCE PER RANK, so the raw lines over-count by the
    # rank count and each duplicate would close and reopen the same window.
    # Collapsing on (timestamp, direction) is NOT enough: the log has
    # one-second resolution and the ranks straddle second boundaries, so on
    # a real 1026-line log that yields 560 "distinct" events against ~342
    # real flips. Filter to a SINGLE rank instead -- exact by construction.
    parsed = []
    for line in out:
        m = re.match(
            r"([0-9-]+ [0-9:]+) ([A-Za-z0-9_]+)\] PHASE-FLIP DONE (\w+)", line
        )
        if not m:
            continue
        ts = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
        parsed.append((ts, m.group(2), m.group(3)))
    if not parsed:
        return 0.0
    keep = sorted({rank for _, rank, _ in parsed})[0]
    events = sorted((ts, d) for ts, rank, d in parsed if rank == keep)
    total, in_tp, cur = 0.0, False, t_start
    for ts, direction in events:
        if ts < t_start or ts > t_end:
            if ts <= t_start:
                in_tp = direction == "pp_to_tp"
            continue
        if in_tp:
            total += ts - cur
        cur = ts
        in_tp = direction == "pp_to_tp"
    if in_tp:
        total += t_end - cur
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30030)
    ap.add_argument("--log", default="/spinning/serving-30030.boot.log")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=600,
                    help=">= 40 x a few, so decode_log_interval is reached")
    ap.add_argument("--prompt-tokens", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    prompt = "Explain, in careful detail and without repeating yourself, " + (
        "the trade-offs of pipeline versus tensor parallelism. " * (args.prompt_tokens // 12)
    )
    results = []
    lock = threading.Lock()

    def worker(i):
        for _ in range(args.rounds):
            try:
                t0, t1, body = post(args.port, {
                    "model": "Qwen3.6-27B",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": args.max_new,
                    "temperature": 0.0,
                    "chat_template_kwargs": {"enable_thinking": False},
                }, args.timeout)
                usage = body.get("usage", {})
                with lock:
                    results.append({
                        "t0": t0, "t1": t1,
                        "completion": usage.get("completion_tokens", 0),
                        "prompt": usage.get("prompt_tokens", 0),
                    })
            except Exception as e:  # noqa: BLE001 - one failure must not end the probe
                with lock:
                    results.append({"error": repr(e)[:200]})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.concurrency)]
    run0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    run1 = time.time()

    ok = [r for r in results if "error" not in r]
    bad = [r for r in results if "error" in r]
    print(f"requests: {len(ok)} ok, {len(bad)} failed, concurrency {args.concurrency}")
    for b in bad[:3]:
        print(f"  error: {b['error']}")
    if not ok:
        return 1
    total_out = sum(r["completion"] for r in ok)
    wall = run1 - run0
    tp = tp_seconds(args.log, run0, run1)
    print(f"window            : {wall:.1f} s wall, {tp:.1f} s inside TP "
          f"({100*tp/wall:.0f}% TP duty cycle)")
    print(f"output tokens     : {total_out}")
    print(f"WALL tok/s        : {total_out/wall:.1f}   <- what a client sees")
    if tp > 0:
        print(f"IN-PHASE tok/s    : {total_out/tp:.1f}   <- decode speed when decode is running")
        print("  A gap between these two is the flip's DUTY CYCLE, not a decode defect.")
        print("  Only the in-phase number is comparable against plain TP3.")
    per = [r["completion"]/(r["t1"]-r["t0"]) for r in ok if r["t1"] > r["t0"]]
    if per:
        print(f"per-request tok/s : median {statistics.median(per):.1f}, "
              f"min {min(per):.1f}, max {max(per):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
