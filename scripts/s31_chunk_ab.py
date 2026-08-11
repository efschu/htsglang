#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#656 item 8, CHUNK A/B: what does ``chunked_prefill_size`` actually buy?

WHY THIS EXISTS. The ship config runs `chunked_prefill_size=2048` -- the bare
default, not even present in argv -- and nobody has measured it on this rig.
Observed live on 2026-08-11: one agent request carried `#pending-token 178784`
and was fed 2048 tokens at a time, i.e. **~87 sequential prefill chunks for a
single request**, against 4841 prefill batches vs 144 decode batches over the
window. Whatever the right chunk is, 2048 was never chosen; it was inherited.

WHAT IS MEASURED, and what is deliberately NOT
-----------------------------------------------
PREFILL is the axis the chunk moves, so the probe is built to isolate it:

* ``max_tokens`` is tiny, so decode contributes almost nothing to the wall
  clock and cannot smuggle its own variance into the result. Throughput on
  this rig is known to follow the OUTPUT content (r=0.90), and that effect is
  exactly what must be kept out of a prefill measurement.
* TTFT is the number, and prefill tok/s is derived as
  ``prompt_tokens / TTFT``. TTFT ends at the first streamed token, which is
  the last event the prefill is responsible for.
* Every prompt is UNIQUE BY CONSTRUCTION (a random preamble). A repeated
  prompt would hit the prefix cache and measure the cache, not the chunk.
  This is the one place where defeating the cache is correct: the question is
  "how fast does this rig prefill N cold tokens", and a cache hit answers a
  different question.

A-VS-A FIRST, ALWAYS
--------------------
The first thing this prints is a noise floor: the same arm run twice,
back-to-back, on ONE boot. Without it there is no way to say whether a 6%
difference between two chunk sizes is a result or a mood. Warmup requests are
discarded rather than averaged in, because the first request after a boot
pays for lazy allocation and graph state that no later one does.

**THE FLOOR ON THIS RIG IS LARGE, AND THAT IS THE MEASUREMENT'S MAIN
DIFFICULTY.** Measured at n=1: 20.7% between two identical passes. The cause
is structural rather than sloppy -- the instance flips PP<->TP continuously
(318 flips in ~40 min), and under strict purity a prefill cannot start until
the PP phase comes round, so each request's TTFT carries a wait whose length
depends on where in the flip cycle it arrived. That wait is real and belongs
in the number, but it is NOISE with respect to the chunk size.

Two defences, and both are needed:
* enough requests per pass to average over flip phase (n=1 is meaningless
  here, whatever the numbers look like);
* a prompt long enough that the CHUNK COUNT dominates the flip wait. At a
  30k-token prompt, chunk 2048 is ~15 chunks and chunk 16384 is 2, which is a
  large enough separation to clear the floor. A short prompt cannot beat the
  noise no matter how many samples are taken.

USAGE

    # noise floor + one arm, on the running instance
    python3 scripts/s31_chunk_ab.py --label chunk2048 --out /tmp/ab

    # then reboot with a different --chunked-prefill-size and repeat
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import string
import sys
import time
import urllib.request

DEFAULT_URL = "http://127.0.0.1:30030/v1/completions"


def _unique_prompt(approx_tokens: int, rng: random.Random) -> str:
    """A prompt of roughly ``approx_tokens`` tokens that cannot be cached.

    Built from random words rather than repeated filler: a long run of one
    token compresses in ways real text does not, and the radix cache would
    also find structure in it.

    ``approx_tokens`` IS NOMINAL AND RUNS ~2.5x LOW. Random letter sequences
    are not words and the tokenizer shreds them: measured on this model,
    3000 generated "words" charged 10098 prompt tokens, i.e. ~3.4 tokens each
    rather than the ~1.33 an English ratio would predict. The harness always
    reports the count the SERVER charged, so no result depends on this
    estimate -- but the knob would mislead anyone reading the invocation, so
    the factor is named here rather than left as a surprise.
    """
    words = int(approx_tokens * 0.75)
    out = []
    for _ in range(words):
        n = rng.randint(3, 9)
        out.append("".join(rng.choice(string.ascii_lowercase) for _ in range(n)))
    return " ".join(out)


def _one(url: str, prompt: str, max_tokens: int, timeout: float):
    """Send one streamed completion; return (ttft_s, total_s, prompt_tokens)."""
    body = json.dumps(
        {
            "model": "Qwen3.6-27B",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    ttft = None
    prompt_tokens = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = obj.get("usage")
            if usage and usage.get("prompt_tokens"):
                prompt_tokens = int(usage["prompt_tokens"])
            choices = obj.get("choices") or []
            if ttft is None and choices and choices[0].get("text"):
                ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    return ttft, total, prompt_tokens


def _run(url, n, approx_tokens, max_tokens, timeout, seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        p = _unique_prompt(approx_tokens, rng)
        try:
            ttft, total, ptok = _one(url, p, max_tokens, timeout)
        except Exception as e:
            print(f"   request failed: {e}", file=sys.stderr)
            continue
        if ttft is None:
            continue
        rows.append({"ttft_s": ttft, "total_s": total, "prompt_tokens": ptok})
    return rows


def _summary(rows):
    if not rows:
        return None
    ttfts = [r["ttft_s"] for r in rows]
    ptoks = [r["prompt_tokens"] for r in rows if r["prompt_tokens"]]
    med = statistics.median(ttfts)
    tok = statistics.median(ptoks) if ptoks else 0
    return {
        "n": len(rows),
        "ttft_median_s": med,
        "ttft_min_s": min(ttfts),
        "ttft_max_s": max(ttfts),
        "prompt_tokens_median": tok,
        "prefill_tok_s": (tok / med) if (tok and med) else 0.0,
    }


def _show(name, s):
    if not s:
        print(f"   {name}: NO DATA")
        return
    print(
        f"   {name}: TTFT median {s['ttft_median_s']:.2f}s "
        f"(min {s['ttft_min_s']:.2f} / max {s['ttft_max_s']:.2f}), "
        f"{s['prompt_tokens_median']} prompt tokens, "
        f"prefill {s['prefill_tok_s']:.0f} tok/s  [n={s['n']}]"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="arm name, e.g. chunk2048")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--tokens", type=int, default=32000,
                    help="approximate PROMPT size; the server's own count is "
                         "what gets reported")
    ap.add_argument("--n", type=int, default=3, help="measured requests per pass")
    ap.add_argument("--warmup", type=int, default=1,
                    help="discarded requests before measuring")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    print(f"===== CHUNK A/B arm '{a.label}'  ~{a.tokens} prompt tokens, "
          f"max_tokens={a.max_tokens}")

    if a.warmup:
        print(f"-- warmup ({a.warmup} discarded)")
        _run(a.url, a.warmup, a.tokens, a.max_tokens, a.timeout, seed=1)

    # A-VS-A FIRST. Two identical passes, different random prompts, so the
    # spread between them is the noise floor every later comparison is judged
    # against.
    print("-- A-vs-A noise floor (two identical passes, back to back)")
    a1 = _summary(_run(a.url, a.n, a.tokens, a.max_tokens, a.timeout, seed=11))
    a2 = _summary(_run(a.url, a.n, a.tokens, a.max_tokens, a.timeout, seed=22))
    _show("pass A", a1)
    _show("pass A'", a2)
    floor = None
    if a1 and a2 and a1["prefill_tok_s"] and a2["prefill_tok_s"]:
        lo, hi = sorted((a1["prefill_tok_s"], a2["prefill_tok_s"]))
        floor = (hi - lo) / hi * 100.0
        print(f"   NOISE FLOOR: {floor:.1f}% between two identical passes. "
              f"A difference smaller than this is not a result.")

    both = []
    for s in (a1, a2):
        if s:
            both.append(s["prefill_tok_s"])
    arm_tok_s = statistics.median(both) if both else 0.0
    print(f"-- ARM RESULT '{a.label}': prefill {arm_tok_s:.0f} tok/s "
          f"(median of the two passes)")

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        path = os.path.join(a.out, f"{a.label}.json")
        with open(path, "w") as fh:
            json.dump(
                {"label": a.label, "pass_a": a1, "pass_a2": a2,
                 "noise_floor_pct": floor, "arm_prefill_tok_s": arm_tok_s,
                 "approx_prompt_tokens": a.tokens,
                 "max_tokens": a.max_tokens}, fh, indent=2)
        print(f"written: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
