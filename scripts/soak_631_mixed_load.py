#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#631 mixed-load soak driver -- and the corpse-I crash recipe.

THE SHAPE THIS REPRODUCES. The 2026-08-09 20:31:48Z death needed three
things at once, which is why 69 acceptance flips missed it:

  1. requests RESIDENT AND DECODING across a cutover (bs >= 2),
  2. a prefill ARRIVING so the policy arms tp_to_pp
     (pending prefill > N=7004 tok),
  3. the two alternating fast enough that epochs land back to back.

So this driver keeps N_DECODE long generations permanently in flight (they
supply the carried batches) while injecting large prompts on a cadence
(they supply the arming pressure and the post-cutover extend batches). It
does NOT sleep between phases: the point is the interleaving.

Bounded by wall time, never by request count, and every request carries a
client-side timeout, so the driver cannot outlive its window or wedge on a
dead server -- it reports the death instead, which is the observation the
soak exists to make.

Usage:
    python scripts/soak_631_mixed_load.py --minutes 60
    python scripts/soak_631_mixed_load.py --minutes 3 --recipe   # crash try
"""

import argparse
import json
import threading
import time
import urllib.error
import urllib.request

PORT = 30030
URL = f"http://127.0.0.1:{PORT}/v1/completions"

STOP = threading.Event()
LOCK = threading.Lock()
STATS = {"ok": 0, "err": 0, "decode_tokens": 0, "prefill_tokens": 0}
ERRORS: list = []


def post(prompt, max_tokens, timeout):
    body = json.dumps(
        {
            "model": "Qwen3.6-27B",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def note_ok(kind, n):
    with LOCK:
        STATS["ok"] += 1
        STATS[f"{kind}_tokens"] += n


def note_err(exc):
    with LOCK:
        STATS["err"] += 1
        if len(ERRORS) < 20:
            ERRORS.append(f"{type(exc).__name__}: {str(exc)[:160]}")


def decode_worker(idx):
    """A long generation: this is what must be RESIDENT across a cutover."""
    while not STOP.is_set():
        try:
            out = post(
                f"Write a detailed technical explanation, part {idx}. "
                "Cover the topic thoroughly and continue at length.",
                max_tokens=512,
                timeout=600,
            )
            note_ok("decode", out.get("usage", {}).get("completion_tokens", 0))
        except Exception as exc:  # noqa: BLE001 - the soak records, never dies
            note_err(exc)
            time.sleep(2)


def prefill_worker(tokens_per_prompt, period_s):
    """Large prompts on a cadence: the tp_to_pp arming pressure.

    ~4 characters per token is close enough for Qwen; the exact count does
    not matter, only that it lands well above the policy's N=7004.

    THE PREFIX CACHE DEFEATS A NAIVE DRIVER, and it did: the first version
    reused one filler string, so --enable-prefix-caching served every
    repeat from cache and the prefill shrank from ~13000 tokens to
    "#new-token: 1063". Pending prefill then never crossed N=7004 and
    tp_to_pp was never armed for the reason the soak needs it armed. Each
    prompt therefore carries a UNIQUE HIGH-ENTROPY PREFIX, which makes the
    whole body a cache miss.
    """
    import random

    while not STOP.is_set():
        t0 = time.time()
        nonce = f"{random.getrandbits(64):016x}-{time.time_ns()}"
        words = ("alpha bravo charlie delta echo foxtrot golf hotel "
                 "india juliet kilo lima mike november oscar papa").split()
        body = " ".join(
            random.choice(words) for _ in range(int(tokens_per_prompt * 0.9))
        )
        filler = f"Document {nonce}.\n{body}\n"
        try:
            out = post(
                filler + "\n\nSummarise the passage above in one sentence.",
                max_tokens=48,
                timeout=600,
            )
            note_ok("prefill", out.get("usage", {}).get("prompt_tokens", 0))
        except Exception as exc:  # noqa: BLE001
            note_err(exc)
        slept = time.time() - t0
        if slept < period_s and not STOP.is_set():
            STOP.wait(period_s - slept)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--decode-streams", type=int, default=2)
    ap.add_argument("--prefill-tokens", type=int, default=12000)
    ap.add_argument("--prefill-period", type=float, default=8.0)
    ap.add_argument(
        "--recipe",
        action="store_true",
        help="crash recipe: hammer prefill with no gap, maximum flip churn",
    )
    args = ap.parse_args()

    if args.recipe:
        args.prefill_period = 0.0

    deadline = time.time() + args.minutes * 60
    threads = [
        threading.Thread(target=decode_worker, args=(i,), daemon=True)
        for i in range(args.decode_streams)
    ]
    threads.append(
        threading.Thread(
            target=prefill_worker,
            args=(args.prefill_tokens, args.prefill_period),
            daemon=True,
        )
    )
    for t in threads:
        t.start()

    print(
        f"soak start {time.strftime('%H:%M:%SZ', time.gmtime())} "
        f"minutes={args.minutes} decode_streams={args.decode_streams} "
        f"prefill_tokens={args.prefill_tokens} period={args.prefill_period}",
        flush=True,
    )
    last = 0.0
    while time.time() < deadline:
        time.sleep(5)
        now = time.time()
        if now - last >= 60:
            last = now
            with LOCK:
                s = dict(STATS)
            print(
                f"  {time.strftime('%H:%M:%SZ', time.gmtime())} "
                f"ok={s['ok']} err={s['err']} "
                f"decode_tok={s['decode_tokens']} "
                f"prefill_tok={s['prefill_tokens']}",
                flush=True,
            )
    STOP.set()
    for t in threads:
        t.join(timeout=15)

    with LOCK:
        s = dict(STATS)
    print(f"soak end {time.strftime('%H:%M:%SZ', time.gmtime())} {s}", flush=True)
    for e in ERRORS:
        print(f"  ERR {e}", flush=True)
    return 1 if s["err"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
