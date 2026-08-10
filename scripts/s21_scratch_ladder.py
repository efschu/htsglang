#!/usr/bin/env python3
"""#631/#656 successor 21: measure the SCRATCH HIGH-WATER against prefill length.

WHY. Every capacity argument in this chain has compared a corridor minimum
against a KV pool size, and the two have never tracked each other: pool 190000
settled at 1037 MiB free and pool 260000 at 1007 MiB, a 30 MiB difference for a
pool step that should have cost about 1.1 GiB on the largest card. That is the
signature of a term that is NOT the pool. Boot-time sizing leaves 7.8/5.8/5.0
GiB free per card; under load the free memory decays by 4-6.5 GiB. Nobody has
named that decay, and because the allocator's high-water is STICKY it never
comes back, so it behaves like resident mass while being invisible to every
boot-time itemisation.

The hypothesis this script tests: the decay is driven by PREFILL LENGTH, not by
elapsed time or request count. If so it saturates as soon as the longest
sequence the deployment will ever see has been prefilled once, and it can be
measured in minutes instead of discovered after an hour of soaking.

METHOD. Send ONE request per rung, with a prompt of a known token length and a
short generation, and read the NVML free floor before and after each rung from
a corridor CSV that a separate sampler is already writing at 100 ms. The
marginal cost of a rung is the DROP in the running floor that the rung causes.
Because the allocator never returns the segments, the floor is monotone and the
rung costs are additive -- which is exactly why the number is dangerous and
exactly why it must be measured to the longest length the deployment allows.

The ladder deliberately runs from short to long. Running it long-to-short would
allocate the worst case first and report zero marginal cost for every later
rung, which is a true statement about that ordering and a useless one.

Usage:
  s21_scratch_ladder.py --corridor <csv> [--port 30030]
                        [--rungs 4096,8192,16384,32768,65536,131072]
                        [--max-new 8] [--settle 6]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from typing import List, Optional, Tuple

# One line of filler that tokenises densely and predictably for this model.
FILLER = (
    "Distributed inference schedulers interleave prefill and decode work "
    "across pipeline stages while speculative drafting proposes tokens ahead. "
)


def corridor_tail(path: str, seconds: float) -> Optional[List[int]]:
    """Minimum free MiB per card over the last `seconds` of the sampler CSV."""
    try:
        with open(path, "r", errors="replace") as fh:
            lines = fh.readlines()[-4000:]
    except OSError:
        return None
    now = time.time()
    mins: List[int] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("ts"):
            continue
        parts = [p for p in line.split(",") if p != ""]
        if len(parts) < 2:
            continue
        try:
            t = float(parts[0])
            vals = [int(float(v)) for v in parts[1:]]
        except ValueError:
            continue
        if now - t > seconds:
            continue
        if not mins:
            mins = vals
        else:
            mins = [min(a, b) for a, b in zip(mins, vals[: len(mins)])]
    return mins or None


def nvml_free() -> List[int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    ).stdout
    return [int(x.strip()) for x in out.split() if x.strip().isdigit()]


def post(url: str, payload: dict, timeout: float) -> Tuple[int, dict]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:400]}
    except Exception as e:  # noqa: BLE001 - a failed rung must not end the ladder
        return 0, {"error": repr(e)[:400]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corridor", required=True)
    ap.add_argument("--port", type=int, default=30030)
    ap.add_argument("--rungs", default="4096,8192,16384,32768,65536,131072")
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--settle", type=float, default=6.0)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument(
        "--flush-between",
        action="store_true",
        help="Ask the allocator to return cached segments before each rung. "
             "Run the ladder BOTH ways: without this flag the floor measures "
             "the ACCUMULATED residue (what the deployment actually suffers "
             "today), with it the floor measures the CONCURRENT peak of a "
             "single prefill (what it would suffer if the cold segments were "
             "returned at every phase boundary). The difference between the "
             "two floors is exactly what a release-at-cutover buys, and it "
             "cannot be inferred from either run alone.",
    )
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    rungs = [int(x) for x in args.rungs.split(",") if x.strip()]

    print(f"scratch ladder: rungs={rungs} max_new={args.max_new} "
          f"settle={args.settle}s corridor={args.corridor}", flush=True)
    print("The floor is MONOTONE by construction (sticky allocator); a rung's "
          "cost is the drop it causes, and costs are additive.", flush=True)
    print(flush=True)

    floor = corridor_tail(args.corridor, 30.0) or nvml_free()
    print(f"{'rung_tok':>9} {'prompt_tok':>10} {'http':>5} {'wall_s':>7} "
          f"{'free_after (MiB)':>26} {'drop_vs_prev (MiB)':>22}", flush=True)
    print(f"{'baseline':>9} {'-':>10} {'-':>5} {'-':>7} "
          f"{','.join(str(x) for x in floor):>26} {'-':>22}", flush=True)

    reps = max(1, int(rungs[-1] / max(1, len(FILLER) // 5)) + 8)
    big = FILLER * reps

    for rung in rungs:
        if args.flush_between:
            post(f"{base}/flush_cache", {}, 120.0)
            time.sleep(3.0)
            pre = corridor_tail(args.corridor, 2.5) or nvml_free()
        else:
            pre = None
        # Characters-per-token for this filler is ~4.6; overshoot then let the
        # server report the true prompt token count rather than guessing.
        approx_chars = int(rung * 4.4)
        text = big[:approx_chars]
        t0 = time.time()
        code, resp = post(
            f"{base}/generate",
            {
                "text": text,
                "sampling_params": {
                    "max_new_tokens": args.max_new,
                    "temperature": 0.0,
                },
            },
            args.timeout,
        )
        wall = time.time() - t0
        ptok = "?"
        if isinstance(resp, dict):
            meta = resp.get("meta_info") or {}
            ptok = meta.get("prompt_tokens", "?")
        # The in-flight minimum spans the request itself; with --flush-between
        # this is the CONCURRENT peak of that one prefill, uncontaminated by
        # every earlier rung's residue.
        inflight = corridor_tail(args.corridor, wall + 1.0) or nvml_free()
        time.sleep(args.settle)
        after = corridor_tail(args.corridor, args.settle + 4.0) or nvml_free()
        n = min(len(after), len(floor))
        drop = [floor[i] - after[i] for i in range(n)]
        floor = [min(floor[i], after[i]) for i in range(n)]
        note = ""
        if pre is not None:
            m = min(len(pre), len(inflight))
            note = ("  inflight_dip=" +
                    ",".join(str(pre[i] - inflight[i]) for i in range(m)))
        print(f"{rung:>9} {str(ptok):>10} {code:>5} {wall:>7.1f} "
              f"{','.join(str(x) for x in after):>26} "
              f"{','.join(str(x) for x in drop):>22}{note}", flush=True)
        if code != 200:
            print(f"    rung failed: {str(resp)[:200]}", flush=True)

    print(flush=True)
    print("FINAL FLOOR (min over the whole ladder), MiB per card: "
          f"{','.join(str(x) for x in floor)}", flush=True)
    print("Read this as the scratch high-water the deployment must survive at "
          "the longest prefill it admits. It is additive with the KV pool, and "
          "it is what a corridor floor actually has to be sized against.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
