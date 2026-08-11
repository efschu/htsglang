#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#656 spec items 3+4: the bs1 leg, which is ONE leg and not two.

WHY THE TWO SPEC ITEMS ARE THE SAME RUN
----------------------------------------
Item 4 wants a bs=1 session decoding ABOVE the 262144 standard context with
YaRN RoPE. Item 3 wants that same bs=1 regime to have the MAXIMUM KV
available, with the bs2-4 reserves (idle mamba slots, KV headroom) spilled
while only one request is live.

A single session whose context passes 262144 is the strongest form of both:
it is the long-context proof by construction, and it is simultaneously the
largest KV residency a single request can demand -- which is exactly the
pressure that makes the relief ladder reach past its cheapest tier. Running
them as two legs would prove less and cost twice.

WHAT IT ASSERTS, AND WHAT IT ONLY RECORDS
------------------------------------------
ASSERTS: ``prompt_tokens > 262144`` as reported by the SERVER's own usage
accounting, and that decode produced tokens at that depth. A prompt this
side of the boundary is not the leg; a prompt that merely LOOKS long is not
evidence, because the client's idea of the token count is not the server's.

RECORDS (for the judged extract, not asserted here): the corridor during the
leg, whether the KV rung fired, and whether the restored session came back on
CUDA graphs. Those live in the serving log and are judged from it -- a load
script that graded the axes it is itself driving would be marking its own
work.

The prompt is built from a deterministic pseudo-random word stream rather
than a repeated block, on purpose: a repeated block is a prefix-cache and
radix-tree special case, and a leg meant to occupy KV must not be quietly
deduplicated into a fraction of the rows it claims to hold.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request

BOUNDARY = 262144


def build_prompt(target_tokens: int, seed: int = 656) -> str:
    """A non-repeating word stream of roughly ``target_tokens`` tokens.

    ~0.75 words per token is the conservative direction for this tokenizer:
    it overshoots, and the leg re-measures against the server's own count
    rather than trusting this estimate.
    """
    rng = random.Random(seed)
    vocabulary = [
        f"{a}{n}"
        for a in ("alpha", "beta", "gamma", "delta", "kappa", "sigma", "omega")
        for n in range(1000)
    ]
    words = [rng.choice(vocabulary) for _ in range(int(target_tokens * 1.05))]
    return " ".join(words)


def post(url: str, payload: dict, timeout: float):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30030")
    ap.add_argument("--tokens", type=int, default=272000)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--timeout", type=float, default=2400.0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    prompt = build_prompt(args.tokens)
    results = []
    for attempt in range(args.repeat):
        t0 = time.time()
        print(
            f"[yarn-bs1] leg {attempt + 1}/{args.repeat}: posting a "
            f"~{args.tokens} token prompt ({len(prompt)} chars)",
            flush=True,
        )
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
        except Exception as e:  # noqa: BLE001 - a failed leg is a RESULT
            print(f"[yarn-bs1] FAILED after {time.time() - t0:.0f}s: {e}", flush=True)
            results.append({"ok": False, "error": str(e), "seconds": time.time() - t0})
            continue
        usage = res.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        text = (res.get("choices") or [{}])[0].get("text", "")
        rec = {
            "ok": prompt_tokens > BOUNDARY and completion_tokens > 0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "above_262144": prompt_tokens > BOUNDARY,
            "seconds": round(time.time() - t0, 1),
            "sample": text[:160],
        }
        results.append(rec)
        print(f"[yarn-bs1] {json.dumps(rec)}", flush=True)

    ok = [r for r in results if r.get("ok")]
    verdict = {
        "legs": len(results),
        "ok": len(ok),
        "max_prompt_tokens": max((r.get("prompt_tokens", 0) for r in results), default=0),
        "boundary": BOUNDARY,
        "results": results,
    }
    print(f"[yarn-bs1] VERDICT {json.dumps(verdict)}", flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(verdict, fh, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
