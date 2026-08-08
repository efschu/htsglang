#!/usr/bin/env python3
"""#631 Route A prefill ladder, measured in the PP=3 phase.

Methodology is FINAL_631 section 1 verbatim, so the numbers are directly
comparable to the acceptance table:

  * random ``input_ids`` per draw -- prefix caching cannot contaminate a
    repeat, which a fixed prompt would silently do (the second draw would
    measure a cache hit, not prefill),
  * ``max_new_tokens=1`` so the measurement is prefill and not decode,
  * one warm-up draw per rung, discarded,
  * 3 kept draws, median reported, spread printed so the reader can see
    the noise floor rather than trust a single number.

Prefill MUST be taken in the PP phase: a run that reports prefill and
decode from the same phase has not measured Route A.

Stdlib only, on purpose -- this has to run against a live production
server without pulling anything into its environment.
"""

import argparse
import json
import random
import statistics
import time
import urllib.request


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30030)
    ap.add_argument("--rungs", default="2048,8192,32768")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--vocab", type=int, default=150000)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = {"rungs": [], "method": "random ids, max_new_tokens=1, warmup discarded"}
    print(f"{'input tok':>10} {'median ms':>10} {'tok/s':>8}   draws (ms)")
    for rung in [int(x) for x in args.rungs.split(",")]:
        draw(args.port, rung, args.vocab, args.timeout)  # warm-up, discarded
        times = [
            draw(args.port, rung, args.vocab, args.timeout) * 1000.0
            for _ in range(args.draws)
        ]
        med = statistics.median(times)
        rec = {
            "input_tokens": rung,
            "median_ms": round(med, 1),
            "tok_s": round(rung / (med / 1000.0), 1),
            "draws_ms": [round(t, 1) for t in times],
        }
        out["rungs"].append(rec)
        print(
            f"{rung:>10} {med:>10.1f} {rec['tok_s']:>8.1f}   "
            + " / ".join(f"{t:.1f}" for t in times)
        )

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
