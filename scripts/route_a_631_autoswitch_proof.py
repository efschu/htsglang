#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#631: PROVE the autoswitch, with numbers, in one unmanned run.

The claim under test, stated the way the operator states it:

    prefill happens in the PP=3 layout, decode happens in the TP=3
    layout, and the instance moves between them BY ITSELF.

So this measures both halves SEPARATELY and stamps each with the layout
that was active while it ran:

  * PREFILL: a long prompt with max_new_tokens=1, so the measured time is
    prefill and essentially nothing else. Cache flushed first, or the
    second rep measures the radix cache instead of the layout.
  * DECODE: a short prompt with many generated tokens, timed from the
    FIRST token to the last (time-to-first-token is prefill and is
    excluded), so the rate is decode and essentially nothing else.

THE LAYOUT STAMP IS THE POINT, and it is read from the serving log's
cutover line -- the single writer of that fact. Utilisation cannot
substitute: under PP the three stages are PIPELINED, so a long chunked
prefill saturates all three cards exactly as TP does.

This script issues NO flip call. Every layout change it reports came from
the policy. Run it against a POLICY=auto boot.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
import urllib.request
from typing import List, Optional, Tuple

CUTOVER_RE = re.compile(r"cutover complete: active stack (\w+)")


def read_phase(log: str) -> str:
    """The active layout, from the cutover line's single writer."""
    try:
        with open(log, "rb") as fh:
            try:
                fh.seek(-4_000_000, 2)
            except OSError:
                fh.seek(0)
            text = fh.read().decode("utf-8", "replace")
    except OSError:
        return "?"
    hits = CUTOVER_RE.findall(text)
    return hits[-1] if hits else "pp"  # no flip yet -> the Route A boot layout


def count_flips(log: str) -> Tuple[int, int]:
    try:
        with open(log, "rb") as fh:
            try:
                fh.seek(-4_000_000, 2)
            except OSError:
                fh.seek(0)
            text = fh.read().decode("utf-8", "replace")
    except OSError:
        return 0, 0
    return (
        text.count("PHASE-FLIP DONE pp_to_tp"),
        text.count("PHASE-FLIP DONE tp_to_pp"),
    )


def post(port: int, payload: dict, timeout: float):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def flush(port: int) -> None:
    try:
        urllib.request.urlopen(
            f"http://127.0.0.1:{port}/flush_cache?timeout=60", timeout=90
        ).read()
    except Exception:  # noqa: BLE001
        pass


def measure_prefill(port, tokens, vocab, timeout, log) -> Tuple[float, str]:
    flush(port)
    ids = [random.randint(1000, vocab) for _ in range(tokens)]
    t0 = time.perf_counter()
    post(
        port,
        {
            "input_ids": ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 1,
                "ignore_eos": True,
            },
        },
        timeout,
    )
    dt = time.perf_counter() - t0
    return tokens / dt, read_phase(log)


def measure_decode(port, prompt_tokens, new_tokens, vocab, timeout, log):
    """Rate from FIRST token to last: prefill is excluded by construction."""
    ids = [random.randint(1000, vocab) for _ in range(prompt_tokens)]
    payload = {
        "input_ids": ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": new_tokens,
            "ignore_eos": True,
        },
        "stream": True,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    first_at: Optional[float] = None
    last_at = 0.0
    n = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload_s = line[5:].strip()
            if payload_s == "[DONE]":
                break
            try:
                obj = json.loads(payload_s)
            except json.JSONDecodeError:
                continue
            meta = obj.get("meta_info") or {}
            got = int(meta.get("completion_tokens") or 0)
            if got <= 0:
                continue
            now = time.perf_counter()
            if first_at is None:
                first_at = now
                n = got
                continue
            last_at = now
            n = got
    if first_at is None or last_at <= first_at or n <= 1:
        return 0.0, read_phase(log), n
    return (n - 1) / (last_at - first_at), read_phase(log), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30030)
    ap.add_argument("--log", default="/spinning/serving-30030.boot.log")
    ap.add_argument("--prefill-tokens", type=int, default=32768)
    ap.add_argument("--decode-prompt", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--vocab", type=int, default=150000)
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    f0 = count_flips(args.log)
    print("#631 AUTOSWITCH PROOF -- this script issues NO flip call.")
    print(f"start layout: {read_phase(args.log)}   flips so far: {f0}")
    print()
    print(f"{'cycle':<6}{'phase':<7}{'prefill tok/s':>15}   "
          f"{'phase':<7}{'decode tok/s':>14}{'gen':>6}")
    print("-" * 62)

    pre_rates: List[float] = []
    dec_rates: List[float] = []
    pre_phases: List[str] = []
    dec_phases: List[str] = []

    for c in range(1, args.cycles + 1):
        pr, pph = measure_prefill(
            args.port, args.prefill_tokens, args.vocab, args.timeout, args.log
        )
        dr, dph, n = measure_decode(
            args.port,
            args.decode_prompt,
            args.decode_tokens,
            args.vocab,
            args.timeout,
            args.log,
        )
        pre_rates.append(pr)
        dec_rates.append(dr)
        pre_phases.append(pph)
        dec_phases.append(dph)
        print(f"{c:<6}{pph:<7}{pr:>15.1f}   {dph:<7}{dr:>14.1f}{n:>6}")

    f1 = count_flips(args.log)
    print("-" * 62)
    print(f"prefill median {statistics.median(pre_rates):8.1f} tok/s   "
          f"layouts seen: {sorted(set(pre_phases))}")
    print(f"decode  median {statistics.median(dec_rates):8.1f} tok/s   "
          f"layouts seen: {sorted(set(dec_phases))}")
    print(f"automatic flips during this run: pp_to_tp {f1[0]-f0[0]}, "
          f"tp_to_pp {f1[1]-f0[1]}  (zero manual)")
    switched = (f1[0] - f0[0]) + (f1[1] - f0[1]) > 0
    print()
    print(f"AUTOSWITCH OBSERVED: {switched}")
    return 0 if switched else 1


if __name__ == "__main__":
    sys.exit(main())
