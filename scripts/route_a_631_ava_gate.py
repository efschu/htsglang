#!/usr/bin/env python3
"""#631 A-vs-A regression gate client: same-boot noise floor first, then the
cross-commit number.

THE RULE THIS ENFORCES. A cross-commit delta may only be quoted against a
spread measured INSIDE each boot, on the same content axis, with the warmup
discarded and the reps taken back to back. So every rung here runs

    1 warmup (discarded)  +  2 * REPS_PER_BLOCK measured reps

and the measured reps are split into two blocks, A1 and A2, which are the
SAME configuration measured twice. |A1 - A2| within a boot is the noise
floor; a cross-commit delta smaller than that floor is not a finding.

CONTENT AXIS. Both rungs replay one fixed prompt string, so the input token
count and the work per rep are identical across reps and across boots.
``/flush_cache`` runs before every prefill rep, otherwise rep 2 onward would
be served from the radix cache and measure nothing. The decode rung is
greedy with ``ignore_eos``, so its output length is fixed by construction;
the output hash is recorded per rep because throughput on this rig tracks
output CONTENT, and a content divergence between boots would invalidate a
decode comparison that ignored it.

RUN LENGTH. Both rungs are sized to keep a single rep above the 10 s floor
the measurement canon requires, so a rep spans many scheduler rounds rather
than one lucky one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

# One fixed paragraph, repeated. Deterministic bytes -> deterministic
# tokenization -> the same prefill work in every rep and in both boots.
_PARAGRAPH = (
    "The scheduler advances one round at a time, and every rank in the "
    "pipeline pays the same clock. A rank that finishes its shard early "
    "does not go faster; it waits at the next collective until the slowest "
    "rank arrives, which is why the round is the unit of measurement and "
    "the wall time of a single rank is not. "
)


def build_prompt(words: int) -> str:
    unit = _PARAGRAPH.split()
    out = []
    while len(out) < words:
        out.extend(unit)
    return " ".join(out[:words])


def post(url: str, payload: dict | None, timeout: float) -> dict:
    data = json.dumps(payload if payload is not None else {}).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode()
    if not body.strip():
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        # /flush_cache answers in plain prose, not JSON. Only /generate's
        # body is parsed for numbers, so a non-JSON ack is not an error.
        return {"_raw": body}


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def flush(base: str) -> None:
    """Flush the radix cache, and REFUSE to measure if it did not flush.

    The endpoint answers 400 ("pending requests") for a moment after a
    request completes, even at #queue-req 0 / #running-req 0 -- the
    bookkeeping trails the completion. It takes a ``timeout`` query
    parameter and will wait, so the retry is a wait, not a poll-and-hope.
    A rep that ran against a warm cache measures the radix tree, not the
    prefill, so an unflushed cache aborts the run instead of quietly
    producing a fast number.
    """
    last = ""
    for _ in range(6):
        try:
            post(f"{base}/flush_cache?timeout=30", None, 60)
            return
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
            time.sleep(2.0)
    raise RuntimeError(f"flush_cache never succeeded ({last})")


def one_rep(base: str, prompt: str, max_new: int, timeout: float) -> dict:
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_new,
            "temperature": 0.0,
            "ignore_eos": max_new > 1,
        },
    }
    t0 = time.perf_counter()
    out = post(f"{base}/generate", payload, timeout)
    wall = time.perf_counter() - t0
    meta = out.get("meta_info", {}) or {}
    text = out.get("text", "") or ""
    return {
        "wall_s": wall,
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": meta.get("completion_tokens"),
        "cached_tokens": meta.get("cached_tokens"),
        "e2e_latency": meta.get("e2e_latency"),
        "spec_accept_length": meta.get("spec_accept_length"),
        "out_sha1_12": hashlib.sha1(text.encode()).hexdigest()[:12],
        "out_head": text[:60].replace("\n", " "),
    }


def summarize(reps: list[dict], key: str, block: int) -> dict:
    vals = [r[key] for r in reps]
    a1 = vals[:block]
    a2 = vals[block : 2 * block]
    mean_all = statistics.fmean(vals)
    out = {
        "n": len(vals),
        "values": [round(v, 4) for v in vals],
        "mean": mean_all,
        "min": min(vals),
        "max": max(vals),
        "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "block_A1_mean": statistics.fmean(a1) if a1 else None,
        "block_A2_mean": statistics.fmean(a2) if a2 else None,
    }
    if a1 and a2:
        d = out["block_A2_mean"] - out["block_A1_mean"]
        out["AvsA_abs"] = d
        out["AvsA_pct"] = 100.0 * d / out["block_A1_mean"]
        out["spread_pct"] = 100.0 * (out["max"] - out["min"]) / mean_all
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30031")
    ap.add_argument("--label", required=True, help="boot label, e.g. flip-7ed8abfddc")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefill-words", type=int, default=36000)
    ap.add_argument("--decode-words", type=int, default=380)
    ap.add_argument("--decode-tokens", type=int, default=512)
    ap.add_argument("--block", type=int, default=3, help="reps per A block")
    args = ap.parse_args()

    prefill_prompt = build_prompt(args.prefill_words)
    decode_prompt = build_prompt(args.decode_words)
    result: dict = {
        "label": args.label,
        "base": args.base,
        "started": now_iso(),
        "block": args.block,
        "rungs": {},
    }

    # ---- rung 1: PREFILL (max_new_tokens 1, cache flushed every rep) -------
    print(f"[{now_iso()}] rung prefill: warmup", flush=True)
    flush(args.base)
    one_rep(args.base, prefill_prompt, 1, 900)
    reps = []
    t_start = now_iso()
    for i in range(2 * args.block):
        flush(args.base)
        time.sleep(1.0)
        r = one_rep(args.base, prefill_prompt, 1, 900)
        r["tok_s"] = (r["prompt_tokens"] or 0) / r["wall_s"]
        reps.append(r)
        print(
            f"[{now_iso()}] prefill rep {i}: {r['wall_s']:.2f}s "
            f"{r['prompt_tokens']} tok -> {r['tok_s']:.1f} tok/s",
            flush=True,
        )
    result["rungs"]["prefill"] = {
        "window": [t_start, now_iso()],
        "reps": reps,
        "tok_s": summarize(reps, "tok_s", args.block),
        "wall_s": summarize(reps, "wall_s", args.block),
    }

    # ---- rung 2: DECODE (fixed greedy length, bs=1) ------------------------
    print(f"[{now_iso()}] rung decode: warmup", flush=True)
    flush(args.base)
    one_rep(args.base, decode_prompt, 64, 900)
    reps = []
    t_start = now_iso()
    for i in range(2 * args.block):
        flush(args.base)
        time.sleep(1.0)
        r = one_rep(args.base, decode_prompt, args.decode_tokens, 1800)
        # ms per decode ROUND at bs=1: one round emits one accepted-token
        # group, and completion_tokens counts the tokens actually returned.
        n = r["completion_tokens"] or args.decode_tokens
        r["tok_s"] = n / r["wall_s"]
        r["ms_per_token"] = 1000.0 * r["wall_s"] / n
        reps.append(r)
        print(
            f"[{now_iso()}] decode rep {i}: {r['wall_s']:.2f}s {n} tok -> "
            f"{r['tok_s']:.2f} tok/s  sha {r['out_sha1_12']}",
            flush=True,
        )
    result["rungs"]["decode"] = {
        "window": [t_start, now_iso()],
        "reps": reps,
        "tok_s": summarize(reps, "tok_s", args.block),
        "ms_per_token": summarize(reps, "ms_per_token", args.block),
        "out_hashes": sorted({r["out_sha1_12"] for r in reps}),
    }

    result["finished"] = now_iso()
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[{now_iso()}] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.URLError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        sys.exit(2)
