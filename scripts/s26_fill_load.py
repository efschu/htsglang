#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#631/#656 HIGH-OCCUPANCY load: fill the KV pool and keep it filled.

WHY THIS EXISTS, and it is the recurring failure of this whole chain.
Every capacity step so far has been refused as evidence for the same
reason: occupancy. Successor 25's 500000 boot reached 26%, the 65-minute
acceptance run reached 38.1%, and the 430000 step under the 2.1b seam
reached 30.6%. The seam's staging term scales with the LIVE SET, so a
load that never fills the pool cannot exercise the term that decides the
ceiling, and `s25_step_verdict.py` correctly refuses to draw a capacity
conclusion from one.

THE CAUSE was in the load, not the pool. `soak_631_mixed_load.py` drives
occupancy with its PREFILL worker, whose requests retire almost as fast
as they arrive, while its DECODE workers -- the ones that stay resident
across a cutover, which is what they are for -- carry a one-sentence
prompt and 512 output tokens. So the resident set was a few thousand
slots plus whatever prefill happened to be in flight, and no cadence
tuning on the prefill side could fix that: a request that retires cannot
hold a slot.

WHAT THIS DOES INSTEAD: K concurrent streams, each carrying a LONG unique
prompt and then decoding slowly, so each one holds its whole context
resident for the length of the run. Occupancy is then
`K * context_tokens`, chosen directly rather than hoped for. With the
rig's `max_running_requests=4`, four streams of ~105000 tokens hold
~420000 slots, which is ~98% of a 430000 pool.

UNIQUE HIGH-ENTROPY PREFIX per prompt, for the reason
`soak_631_mixed_load.py` documents: `--enable-prefix-caching` serves a
repeated filler from cache, the prefill collapses to a handful of new
tokens, and the pool never fills. Each prompt here is unique.

Usage:
  s26_fill_load.py --minutes 16 --streams 4 --context-tokens 105000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import string
import threading
import time
import urllib.error
import urllib.request

PORT = os.environ.get("FILL_PORT", "30030")
URL = f"http://127.0.0.1:{PORT}/v1/completions"
MODEL = os.environ.get("FILL_MODEL", "Qwen3.6-27B")

STOP = threading.Event()
LOCK = threading.Lock()
STATS = {"ok": 0, "err": 0, "decode_tokens": 0, "prefill_tokens": 0}
ERRORS: list = []


def _unique_prompt(tokens: int) -> str:
    """~4 characters per token is close enough for Qwen.

    The leading random block is what makes the whole body a prefix-cache
    miss; without it the prefill collapses and the pool never fills.
    """
    head = "".join(random.choices(string.ascii_letters + string.digits, k=64))
    filler = (
        "The following is a long technical corpus about distributed "
        "inference, tensor parallelism and key-value cache management. "
    )
    need = max(1, tokens * 4 - len(head))
    body = (filler * (need // len(filler) + 1))[:need]
    return f"[{head}] {body}"


def post(prompt, max_tokens, timeout):
    body = json.dumps(
        {
            "model": MODEL,
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


def note_ok(kind, decode_tokens, prefill_tokens):
    with LOCK:
        STATS["ok"] += 1
        STATS["decode_tokens"] += int(decode_tokens)
        STATS["prefill_tokens"] += int(prefill_tokens)


def note_err(exc):
    with LOCK:
        STATS["err"] += 1
        if len(ERRORS) < 20:
            ERRORS.append(repr(exc)[:300])


def fill_worker(idx, context_tokens, max_tokens, timeout):
    """One resident stream: long context in, slow decode out.

    The point is the RESIDENCY, not the throughput. Each iteration holds
    ``context_tokens`` slots from the end of its prefill until its last
    decoded token, and a flip that lands in that window is a flip at
    representative occupancy -- which is the only kind that tests the
    ceiling.
    """
    while not STOP.is_set():
        try:
            out = post(_unique_prompt(context_tokens), max_tokens, timeout)
            usage = out.get("usage", {})
            note_ok(
                "fill",
                usage.get("completion_tokens", 0),
                usage.get("prompt_tokens", 0),
            )
        except Exception as exc:  # noqa: BLE001 - the load records, never dies
            note_err(exc)
            time.sleep(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=16.0)
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--context-tokens", type=int, default=105000)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=float, default=1200.0)
    args = ap.parse_args()

    print(
        f"fill load start {time.strftime('%H:%M:%SZ', time.gmtime())} "
        f"minutes={args.minutes} streams={args.streams} "
        f"context_tokens={args.context_tokens} "
        f"target_occupancy={args.streams * args.context_tokens} slots",
        flush=True,
    )
    threads = [
        threading.Thread(
            target=fill_worker,
            args=(i, args.context_tokens, args.max_tokens, args.timeout),
            daemon=True,
        )
        for i in range(args.streams)
    ]
    for t in threads:
        t.start()

    deadline = time.time() + args.minutes * 60
    while time.time() < deadline:
        time.sleep(60)
        with LOCK:
            snap = dict(STATS)
        print(
            f"  {time.strftime('%H:%M:%SZ', time.gmtime())} "
            f"ok={snap['ok']} err={snap['err']} "
            f"decode_tok={snap['decode_tokens']} "
            f"prefill_tok={snap['prefill_tokens']}",
            flush=True,
        )
    STOP.set()
    with LOCK:
        snap = dict(STATS)
    print(
        f"fill load end {time.strftime('%H:%M:%SZ', time.gmtime())} {snap}",
        flush=True,
    )
    for e in ERRORS:
        print(f"  ERR {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
