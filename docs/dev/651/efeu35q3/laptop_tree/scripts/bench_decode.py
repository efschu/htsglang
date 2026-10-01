#!/usr/bin/env python
"""Decode-rate benchmark for the #651 laptop GGUF bring-up.

Companion to bench_prefill.py. Prefill is measured by TTFT with max_tokens=1;
decode is a different regime and needs its own number: the steady-state
inter-token interval once the first token is out.

Method, matching the standing measurement rules of this project:

* **Streaming.** The first token's arrival separates prefill from decode. With
  a non-streaming request the two are fused into one latency and cannot be
  told apart.
* **The first inter-token gap is discarded.** It carries the tail of prefill
  and the first decode step's one-off costs.
* **Every prompt is unique** (nonce) so no prefix cache can serve it, and the
  prompt is deliberately short: this measures decode, not prefill.
* **Time-bounded, not count-bounded** (>= min_seconds of decoding per point),
  because short runs on this hardware are dominated by fixed costs.
* **A-vs-A first.** Two identical runs establish the noise floor. A later
  difference smaller than that floor is not a result.
* **Per-token interval statistics are reported** (median / p90 / max), not just
  the mean rate: the ms-per-round distribution is what a pipeline-parallel
  split has to reason about, and a fat tail is invisible in an average.
"""

import argparse
import json
import random
import statistics
import string
import sys
import time
import urllib.request


def _nonce(n: int = 24) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def make_prompt() -> str:
    # Short and unique: decode rate must not be polluted by prefill work.
    return (
        f"Session {_nonce()}. Write a detailed explanation of how a river "
        "forms, from source to delta. Be thorough and continue at length."
    )


def stream_once(base: str, model: str, max_tokens: int, timeout: float):
    """Return (ttft_s, [inter-token gaps in s], n_tokens)."""
    body = json.dumps(
        {
            "model": model,
            "prompt": make_prompt(),
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    gaps = []
    t0 = time.perf_counter()
    ttft = None
    last = None
    n = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices or not choices[0].get("text"):
                continue
            now = time.perf_counter()
            n += 1
            if ttft is None:
                ttft = now - t0
            else:
                gaps.append(now - last)
            last = now
    return ttft, gaps, n


def measure(base, model, max_tokens, min_seconds, timeout, tag):
    all_gaps = []
    ttfts = []
    t_start = time.perf_counter()
    reqs = 0
    while time.perf_counter() - t_start < min_seconds:
        ttft, gaps, n = stream_once(base, model, max_tokens, timeout)
        reqs += 1
        if ttft is not None:
            ttfts.append(ttft)
        # Drop the first gap: it still carries prefill tail and first-step cost.
        all_gaps.extend(gaps[1:])
        if reqs >= 50:
            break
    if not all_gaps:
        return None
    med = statistics.median(all_gaps)
    srt = sorted(all_gaps)
    p90 = srt[int(0.9 * (len(srt) - 1))]
    res = {
        "tag": tag,
        "requests": reqs,
        "tokens_timed": len(all_gaps),
        "ttft_median_s": statistics.median(ttfts) if ttfts else None,
        "gap_median_ms": med * 1000,
        "gap_p90_ms": p90 * 1000,
        "gap_max_ms": max(all_gaps) * 1000,
        "decode_tok_per_s": 1.0 / med if med else 0.0,
    }
    print(
        f"  [{tag}] {res['decode_tok_per_s']:7.2f} tok/s   "
        f"per-token median {res['gap_median_ms']:7.2f} ms  "
        f"p90 {res['gap_p90_ms']:7.2f} ms  max {res['gap_max_ms']:8.2f} ms  "
        f"(reqs={reqs}, timed tokens={len(all_gaps)}, "
        f"ttft_med={res['ttft_median_s']:.3f} s)"
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=31651)
    ap.add_argument("--model", default="Qwen3.6-35B-A3B-Q4")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--warmup-s", type=float, default=8.0)
    ap.add_argument("--label", default="gpu")
    ap.add_argument("--out")
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    print(f"warmup ({args.warmup_s}s, discarded) ...")
    measure(base, args.model, args.max_tokens, args.warmup_s, args.timeout, "warmup")

    print("run A ...")
    a = measure(base, args.model, args.max_tokens, args.seconds, args.timeout, "A")
    print("run A' (noise floor) ...")
    a2 = measure(base, args.model, args.max_tokens, args.seconds, args.timeout, "A'")
    if a is None or a2 is None:
        print("no tokens timed; nothing to report")
        return 2

    r1, r2 = a["decode_tok_per_s"], a2["decode_tok_per_s"]
    floor = abs(r1 - r2) / max(r1, r2) * 100
    print()
    print(f"A-vs-A decode noise floor: {r1:.2f} vs {r2:.2f} tok/s -> {floor:.2f} %")
    print("Any later decode claim must clear that spread to be a result.")

    out = {
        "label": args.label,
        "port": args.port,
        "runA": a,
        "runA_prime": a2,
        "noise_floor_pct": floor,
        "decode_tok_per_s": max(r1, r2),
    }
    path = args.out or f"decode_{args.label}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
