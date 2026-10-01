#!/usr/bin/env python
"""#651: bisect the prefill length at which the gfx1103 serving path dies with
`HIP error: unspecified launch failure`.

OBSERVATION 2026-08-08 14:40:25: a server that had just answered six short
coherence prompts correctly and deterministically died two seconds into the
first ~2048-token prefill of the throughput bench. That reframes the crash: it
tracks PREFILL LENGTH, not idle time. The chunked-prefill size is 1024, so a
2048-token prompt is the first request that spans more than one prefill chunk.

This script walks prompt length upward against a live server and reports the
largest length that survives and the first that kills it. It deliberately does
NOT restart the server: the first failure ends the run, because after
`unspecified launch failure` the HIP context is dead and every later number
would be meaningless.

Each length is probed with several UNIQUE prompts (nonce, so no prefix cache
can serve them) before being declared safe -- a length that crashes on the
third attempt but not the first is still a crashing length.
"""

import argparse
import json
import random
import string
import sys
import time
import urllib.error
import urllib.request


def _nonce(n: int = 24) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def make_prompt(approx_tokens: int) -> str:
    words = (
        "the system processes each request through several distinct stages "
        "before the final response is produced and returned to the caller "
        "which makes the behaviour easier to reason about in practice "
    ).split()
    out = [f"session {_nonce()} begins."]
    need = int(approx_tokens * 0.75)
    while len(" ".join(out).split()) < need:
        out.append(" ".join(random.choices(words, k=12)))
    return " ".join(out)


def attempt(base: str, model: str, approx_tokens: int, timeout: float):
    """Return (ok, prompt_tokens, detail)."""
    body = json.dumps(
        {
            "model": model,
            "prompt": make_prompt(approx_tokens),
            "max_tokens": 1,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        ptok = (data.get("usage") or {}).get("prompt_tokens")
        return True, ptok, "ok"
    except urllib.error.HTTPError as exc:
        # A 400 is the server refusing the length, not a crash: it is a clean
        # rejection and must not be reported as a survival or a death.
        return None, None, f"HTTP {exc.code}: {exc.read()[:200].decode('utf-8', 'replace')}"
    except Exception as exc:  # noqa: BLE001
        return False, None, f"{type(exc).__name__}: {exc}"


def alive(base: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=10) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=31651)
    ap.add_argument("--model", default="Qwen3.6-35B-A3B-Q4")
    ap.add_argument(
        "--lengths",
        default="128,256,512,768,1024,1152,1408,1792,2048,2560",
        help="ascending prompt lengths to try",
    )
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    if not alive(base):
        print("server is not answering /health; nothing to bisect")
        return 2

    survived, killed_at, rejected = [], None, []
    for L in [int(x) for x in args.lengths.split(",")]:
        print(f"length ~{L} tokens:", flush=True)
        ok_all = True
        for i in range(args.repeats):
            t0 = time.perf_counter()
            ok, ptok, detail = attempt(base, args.model, L, args.timeout)
            dt = time.perf_counter() - t0
            if ok is None:
                print(f"  [{i+1}] REJECTED ({dt:.1f}s) {detail}")
                rejected.append(L)
                ok_all = False
                break
            if ok:
                print(f"  [{i+1}] ok ({dt:6.2f}s, prompt_tokens={ptok})", flush=True)
                continue
            print(f"  [{i+1}] FAILED ({dt:.1f}s) {detail}")
            print(f"  server alive after failure: {alive(base)}")
            killed_at = L
            ok_all = False
            break
        if killed_at is not None:
            break
        if ok_all:
            survived.append(L)

    print()
    print(f"survived lengths : {survived}")
    print(f"rejected (HTTP 400, not a crash): {sorted(set(rejected))}")
    print(f"killed at        : {killed_at}")
    if killed_at is not None and survived:
        print(
            f"CRASH THRESHOLD between ~{survived[-1]} and ~{killed_at} tokens "
            "(chunked-prefill-size is the boundary to compare against)"
        )
    elif killed_at is None:
        print("NO CRASH in the probed range")
    return 0


if __name__ == "__main__":
    sys.exit(main())
