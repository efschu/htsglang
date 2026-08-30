#!/usr/bin/env python3
"""#855 — the two depth ladders, measured compute-honestly.

WHY THIS EXISTS. The per-batch ``input throughput`` lines the server logs are
WALL-CONFOUNDED on the flip shape: they include idle gaps between batches, so
arm A's median of ~18 tok/s is a measure of how often work arrived, not of how
fast a prefill runs. Bursts on both arms reach ~32k. Neither number is a
verdict. And the #855 microbench proved a 1.39-1.46x gain on the GDN LINEARS
only -- the collective floor sits underneath the whole system and can eat it.
So the e2e result here is REPORTED OPEN: "flat" is a valid outcome.

PREFILL LADDER. One request at a time against an idle server, so queue time is
~0 and TTFT is the prefill window. Each prompt carries a UNIQUE LEADING token
(all nine rig drivers do; a trailing unique token leaves the prefix cacheable
and the measurement is then a cache-hit, not a prefill). cached_tokens is READ
BACK from the response and a non-zero value FAILS the point rather than being
quietly averaged in.

DECODE DEPTH. One bs=1 generation carried to >=10k tokens, timed per token from
the stream and reported in depth buckets. This is the unmeasured risk zone: the
GDN recurrent state and the speculative drafter both evolve with depth, and the
whole #855 change is in the GDN projections. spec_accept_length is read per
bucket from meta_info.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request

URL = "http://127.0.0.1:30030"


def _post(path: str, payload: dict, timeout: float, stream: bool = False):
    req = urllib.request.Request(
        f"{URL}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


class NoProgress(Exception):
    """Raised when an arm stops producing tokens but the socket stays open.

    WHY THIS EXISTS (2026-08-30, user caught it): a livelocked server does not
    close the connection and does not answer either, so a streaming read just
    sits there. Three arms of this harness hung silently that way and the
    OPERATOR noticed before the harness did. A bench arm must therefore carry
    its own progress bound: no new token within STALL_S, or the whole arm past
    ARM_BUDGET_S, aborts LOUDLY and lets the next arm run. Never a silent wait.
    """


STALL_S = 90.0
ARM_BUDGET_S = 900.0


def make_prompt(approx_tokens: int, seed: int) -> str:
    """Unique-LEADING-token prompt of roughly `approx_tokens` tokens."""
    rnd = random.Random(seed)
    head = f"UNIQUE-{seed}-{rnd.getrandbits(64):016x} "
    # ~0.75 tokens per word for this vocab; overshoot slightly and let the
    # measured prompt_tokens be the number we report.
    words = [f"{rnd.choice(WORDS)}{rnd.randint(0,999)}" for _ in range(int(approx_tokens * 0.9))]
    return head + " ".join(words)


WORDS = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "kernel", "tensor", "matrix", "vector", "buffer", "segment", "region",
    "scheduler", "allocator", "gradient", "quantise", "channel", "token",
]


def prefill_ladder(depths, timeout: float) -> None:
    print(f"{'target':>8} {'prompt_tok':>11} {'cached':>7} {'TTFT_s':>9} {'tok/s':>10}  status")
    for i, d in enumerate(depths):
        payload = {
            "text": make_prompt(d, 1000 + i),
            "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            "stream": True,
        }
        t0 = time.time()
        ttft = None
        last = None
        last_evt = t0
        try:
            resp = _post("/generate", payload, min(timeout, ARM_BUDGET_S), stream=True)
            for raw in resp:
                now = time.time()
                if now - last_evt > STALL_S or now - t0 > ARM_BUDGET_S:
                    raise NoProgress(
                        f"no stream event for {now-last_evt:.0f}s / arm ran {now-t0:.0f}s"
                    )
                last_evt = now
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                if ttft is None:
                    ttft = time.time() - t0
                last = json.loads(body)
        except Exception as e:  # noqa: BLE001
            print(f"{d:>8} {'-':>11} {'-':>7} {'-':>9} {'-':>10}  FAILED: {type(e).__name__}: {e}")
            continue
        mi = (last or {}).get("meta_info", {})
        ptok = mi.get("prompt_tokens")
        cached = mi.get("cached_tokens")
        ok = "ok" if cached == 0 else f"REJECTED cached={cached}"
        rate = (ptok / ttft) if (ptok and ttft) else 0.0
        print(f"{d:>8} {ptok:>11} {cached:>7} {ttft:>9.3f} {rate:>10.1f}  {ok}")


def decode_depth(total_tokens: int, timeout: float) -> None:
    payload = {
        "text": make_prompt(200, 77) + "\n\nWrite an extremely long, detailed technical manual. Do not stop.\n",
        "sampling_params": {"temperature": 0.7, "max_new_tokens": total_tokens, "ignore_eos": True},
        "stream": True,
    }
    buckets = [(0, 2000), (2000, 5000), (5000, 10**9)]
    stamps = []
    t0 = time.time()
    last = None
    resp = _post("/generate", payload, min(timeout, ARM_BUDGET_S), stream=True)
    prev_n = 0
    last_evt = t0
    for raw in resp:
        now_ = time.time()
        if now_ - last_evt > STALL_S or now_ - t0 > ARM_BUDGET_S:
            print(f"  ABORT: no new token for {now_-last_evt:.0f}s (arm {now_-t0:.0f}s) -- "
                  f"server is not progressing; reporting partial and moving on")
            break
        last_evt = now_
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            break
        last = json.loads(body)
        n = last.get("meta_info", {}).get("completion_tokens") or 0
        if n > prev_n:
            now = time.time()
            for _ in range(n - prev_n):
                stamps.append(now)
            prev_n = n
    total = len(stamps)
    print(f"\ndecode depth: {total} tokens in {time.time()-t0:.1f}s")
    for lo, hi in buckets:
        seg = [s for i, s in enumerate(stamps) if lo <= i < hi]
        if len(seg) < 2:
            continue
        span = seg[-1] - seg[0]
        label = f"{lo}-{hi if hi < 10**9 else ''}"
        print(f"  depth {label:>10}: {len(seg):>6} tok  {len(seg)/span:>7.2f} tok/s")
    mi = (last or {}).get("meta_info", {})
    print(f"  spec_accept_length (whole run): {mi.get('spec_accept_length')}")
    print(f"  spec_accept_rate               : {mi.get('spec_accept_rate')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prefill", "decode"], required=True)
    ap.add_argument("--depths", default="12000,32000,50000,100000")
    ap.add_argument("--decode-tokens", type=int, default=10000)
    ap.add_argument("--timeout", type=float, default=1800)
    a = ap.parse_args()
    if a.mode == "prefill":
        prefill_ladder([int(x) for x in a.depths.split(",")], a.timeout)
    else:
        decode_depth(a.decode_tokens, a.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
