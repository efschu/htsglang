#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#656 spec item 12: drive occupancy until the KV rung has to fund the seam.

WHY A SEPARATE LEG, AND WHY IT IS NOT CHEATING
-----------------------------------------------
The KV backing rung is the tier BELOW the allocator cache. Measured twice now
(s32's acceptance and this one's first 15 minutes), the gate arms at the
corridor law and the cache covers the whole deficit every time, so the rung
proposes nothing and spec item 12 goes unexercised. That is the tier law
working -- free money before KV capacity -- and it is also why the rung can
sit unproven forever at a fill level where the cheap tier is always enough.

The deficit at the seam is ``floor + delta + staging - free``, and the STAGING
term scales with the live slot count. So the way to make the rung necessary is
not to weaken the tier above it -- that would prove nothing about the real
ladder -- but to raise OCCUPANCY until the seam's demand exceeds what an
allocator cache can hold. That is the same regime spec item 15 asks for
("fill to the limit, then spill"), so this leg drives the acceptance INTO the
state the order describes rather than around it.

Concurrency, not one giant prompt: ``max_running_requests`` is 4 and a single
session cannot exceed the context, so several mid-sized sessions reach a
higher resident row count than one maximal one.

The corridor gate is the backstop. If this pushes the instance past what the
ladder can fund, the correct outcome is a REFUSED flip and an abandoned
cutover -- both of which are recorded and neither of which is a breach. A leg
that could only be run safely by turning the safety off would not be evidence.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import urllib.request

BASE_DEFAULT = "http://127.0.0.1:30030"


def _words(n: int, seed: int):
    rng = random.Random(seed)
    vocabulary = [
        f"{a}{i}"
        for a in ("alpha", "beta", "gamma", "delta", "kappa", "sigma", "omega")
        for i in range(1000)
    ]
    return [rng.choice(vocabulary) for _ in range(n)]


def post(url: str, payload: dict, timeout: float):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def count_tokens(base: str, text: str, timeout: float) -> int:
    try:
        res = post(
            f"{base}/v1/messages/count_tokens",
            {"model": "Qwen3.6-27B", "messages": [{"role": "user", "content": text}]},
            timeout,
        )
        return int(res.get("input_tokens", 0))
    except Exception:  # noqa: BLE001
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_DEFAULT)
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=130000)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    probe = _words(2000, 1)
    counted = count_tokens(args.base, " ".join(probe), 120.0)
    tpw = (counted / len(probe)) if counted > 0 else 3.9
    print(f"[occupancy] {tpw:.3f} tokens/word", flush=True)
    need_words = int(args.tokens / max(0.1, tpw))

    results = []
    lock = threading.Lock()

    def one(idx: int, rnd: int):
        prompt = " ".join(_words(need_words, 1000 * rnd + idx))
        t0 = time.time()
        try:
            res = post(
                f"{args.base}/v1/completions",
                {
                    "model": "Qwen3.6-27B",
                    "prompt": prompt,
                    "max_tokens": args.max_tokens,
                    "temperature": 0.0,
                    "stream": False,
                },
                args.timeout,
            )
            u = res.get("usage") or {}
            rec = {
                "ok": True,
                "session": idx,
                "round": rnd,
                "prompt_tokens": int(u.get("prompt_tokens", 0)),
                "completion_tokens": int(u.get("completion_tokens", 0)),
                "seconds": round(time.time() - t0, 1),
            }
        except Exception as e:  # noqa: BLE001
            rec = {
                "ok": False,
                "session": idx,
                "round": rnd,
                "error": str(e)[:200],
                "seconds": round(time.time() - t0, 1),
            }
        with lock:
            results.append(rec)
        print(f"[occupancy] {json.dumps(rec)}", flush=True)

    for rnd in range(args.rounds):
        threads = [
            threading.Thread(target=one, args=(i, rnd), daemon=True)
            for i in range(args.sessions)
        ]
        print(
            f"[occupancy] round {rnd + 1}/{args.rounds}: {args.sessions} "
            f"concurrent sessions x ~{args.tokens} tokens",
            flush=True,
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    ok = sum(1 for r in results if r.get("ok"))
    resident = sum(r.get("prompt_tokens", 0) for r in results if r.get("ok"))
    verdict = {
        "sessions": args.sessions,
        "rounds": args.rounds,
        "ok": ok,
        "failed": len(results) - ok,
        "peak_concurrent_prompt_tokens": resident // max(1, args.rounds),
        "results": results,
    }
    print(f"[occupancy] VERDICT {json.dumps(verdict)}", flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(verdict, fh, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
