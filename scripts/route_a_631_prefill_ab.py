#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#631: PP-layout vs TP-layout PREFILL throughput, on ONE boot.

THE WHOLE POINT IS THE SAME-BOOT DISCIPLINE. A prefill number from one
boot compared against a number from another boot is not a comparison:
pool sizes, the resident prefix cache, clocks and power state all move
between boots. So this measures BOTH layouts against the same weights, the
same pools and the same clocks, minutes apart, and it takes an A-vs-A
NOISE FLOOR in the first layout BEFORE it flips, so the A-vs-B gap can be
read against the run-to-run spread rather than against zero.

The request shape is copied verbatim from the acceptance script's
post-idle probe (`input_ids` of N random tokens, `max_new_tokens=1`,
temperature 0, ignore_eos) so the two are comparable.

Run against a POLICY=manual boot: with the automatic policy running, the
layout would move underneath the measurement -- which is exactly what it
is supposed to do, and exactly what makes it useless as a measurement
harness.

    python3 scripts/route_a_631_prefill_ab.py --port 30030 --tokens 32768
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import urllib.request
from typing import List, Optional, Tuple


def post(port: int, path: str, payload: dict, timeout: float):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def one_prefill(port: int, tokens: int, vocab: int, timeout: float) -> float:
    """Seconds for a `tokens`-long prompt with a single generated token."""
    ids = [random.randint(1000, vocab) for _ in range(tokens)]
    t0 = time.perf_counter()
    post(
        port,
        "/generate",
        {
            "input_ids": ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 1,
                "ignore_eos": True,
            },
        },
        timeout,
    )
    return time.perf_counter() - t0


def flush(port: int, timeout: float) -> None:
    """A cold prefix cache before every rep.

    Without this the second rep of a rung hits the radix cache and
    measures the cache, not the layout -- the fastest way to manufacture a
    flattering number here.
    """
    try:
        urllib.request.urlopen(
            f"http://127.0.0.1:{port}/flush_cache?timeout=60", timeout=timeout
        ).read()
    except Exception:  # noqa: BLE001
        pass


def flip(port: int, direction: str, timeout: float) -> Tuple[bool, str]:
    try:
        out = post(port, "/phase_flip", {"direction": direction}, timeout)
        return bool(out.get("success")), str(out.get("message", ""))[:160]
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)


def rung(port: int, tokens: int, vocab: int, reps: int, timeout: float) -> List[float]:
    out: List[float] = []
    for _ in range(reps):
        flush(port, timeout)
        out.append(one_prefill(port, tokens, vocab, timeout))
    return out


def report(label: str, secs: List[float], tokens: int) -> None:
    rates = [tokens / s for s in secs]
    print(
        f"  {label:<22} {statistics.median(rates):8.1f} tok/s median  "
        f"(min {min(rates):.1f}, max {max(rates):.1f}, n={len(rates)}; "
        f"{statistics.median(secs)*1000:.0f} ms median)"
    )


def wait_committed(port: int, log: str, direction: str, deadline: float) -> bool:
    """A flip returns 200 for ARMED. Wait for the COMMIT in the log."""
    try:
        with open(log, "rb") as fh:
            fh.seek(0, 2)
            start = fh.tell()
    except OSError:
        return False
    t0 = time.time()
    while time.time() - t0 < deadline:
        time.sleep(1.0)
        try:
            with open(log, "rb") as fh:
                fh.seek(start)
                if f"PHASE-FLIP DONE {direction}" in fh.read().decode(
                    "utf-8", "replace"
                ):
                    return True
        except OSError:
            return False
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30030)
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--vocab", type=int, default=150000)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--log", default="/spinning/serving-30030.boot.log")
    args = ap.parse_args()

    print(f"#631 prefill A/B, {args.tokens} tok, {args.reps} reps per arm")
    print("boot layout is PP (Route A); the policy must be MANUAL for this")

    # A-vs-A FIRST: the noise floor, in the layout we start in. Without it
    # the A-vs-B number below has nothing to be significant against.
    print("\nPP layout (boot):")
    a1 = rung(args.port, args.tokens, args.vocab, args.reps, args.timeout)
    report("A (noise floor 1)", a1, args.tokens)
    a2 = rung(args.port, args.tokens, args.vocab, args.reps, args.timeout)
    report("A (noise floor 2)", a2, args.tokens)

    ok, msg = flip(args.port, "pp_to_tp", args.timeout)
    print(f"\nflip pp_to_tp armed={ok}: {msg}")
    if not ok:
        print("FAILED: could not arm the flip; no B arm was measured")
        return 1
    if not wait_committed(args.port, args.log, "pp_to_tp", 90.0):
        print("FAILED: flip never COMMITTED (armed != committed); aborting")
        return 1
    print("flip committed")

    print("\nTP layout:")
    b = rung(args.port, args.tokens, args.vocab, args.reps, args.timeout)
    report("B (tp)", b, args.tokens)

    ok, msg = flip(args.port, "tp_to_pp", args.timeout)
    committed = ok and wait_committed(args.port, args.log, "tp_to_pp", 90.0)
    print(f"\nflip back to pp armed={ok} committed={committed}")
    if committed:
        print("\nPP layout (after the return trip):")
        a3 = rung(args.port, args.tokens, args.vocab, args.reps, args.timeout)
        report("A (return trip)", a3, args.tokens)

    pp_rate = statistics.median([args.tokens / s for s in a1 + a2])
    tp_rate = statistics.median([args.tokens / s for s in b])
    floor = abs(
        statistics.median([args.tokens / s for s in a1])
        - statistics.median([args.tokens / s for s in a2])
    )
    print("\n" + "=" * 62)
    print(f"PP prefill   {pp_rate:8.1f} tok/s")
    print(f"TP prefill   {tp_rate:8.1f} tok/s")
    print(f"A-vs-A floor {floor:8.1f} tok/s  (run-to-run spread in PP)")
    if tp_rate > 0:
        print(f"PP / TP      {pp_rate / tp_rate:8.2f}x")
    print(
        "significant" if abs(pp_rate - tp_rate) > 3 * max(floor, 1e-9)
        else "NOT significant against the noise floor"
    )
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
