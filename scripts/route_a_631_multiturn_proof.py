#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#631: AGENTIC MULTI-TURN proof of the automatic PP<->TP switch.

WHY THIS EXISTS AND THE SINGLE-SHOT PROOF DOES NOT SUFFICE. The earlier
harness measured a prefill and a decode as two SEPARATE requests, which
answers "can it switch" but not "does it switch the way real traffic makes
it switch". A real agent turn is one request that does BOTH: a large
prefill (the conversation so far, plus whatever document it is working
on), immediately followed by a long decode -- and then the next turn does
it again with a longer context.

That is the pattern #631 exists for, and it is the one that stresses the
switch hardest: every turn demands the PP layout and then the TP layout,
back to back, with no idle gap to hide the transition in.

WHAT IS MEASURED PER TURN, each stamped with the layout active at the time
(read from the serving log's cutover line -- the single writer of that
fact; utilisation cannot answer it, because a pipelined PP prefill
saturates all three cards exactly as TP does):

  * PREFILL: time to the FIRST token, over the prompt tokens that were not
    already cached, and the layout when it landed.
  * DECODE: tokens after the first, over the time after the first, and the
    layout when the turn ended.

This script issues NO flip call. Every transition it reports came from the
policy.
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
from typing import Dict, List, Optional, Tuple

CUTOVER_RE = re.compile(r"cutover complete: active stack (\w+)")


def _tail(log: str, nbytes: int = 4_000_000) -> str:
    try:
        with open(log, "rb") as fh:
            try:
                fh.seek(-nbytes, 2)
            except OSError:
                fh.seek(0)
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def read_phase(log: str) -> str:
    hits = CUTOVER_RE.findall(_tail(log))
    return hits[-1] if hits else "pp"


def count_flips(log: str) -> Tuple[int, int]:
    text = _tail(log)
    return (
        text.count("PHASE-FLIP DONE pp_to_tp"),
        text.count("PHASE-FLIP DONE tp_to_pp"),
    )


def turn(
    port: int,
    messages: List[Dict[str, str]],
    max_tokens: int,
    timeout: float,
    log: str,
) -> Dict:
    """One agent turn: prefill then decode, measured separately."""
    payload = {
        "model": "Qwen3.6-27B",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first_at: Optional[float] = None
    last_at = t0
    text = ""
    n_out = 0
    prompt_tokens = 0
    cached_tokens = 0
    phase_at_first = "?"
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
            usage = obj.get("usage") or {}
            if usage:
                prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                details = usage.get("prompt_tokens_details") or {}
                cached_tokens = int(
                    details.get("cached_tokens") or cached_tokens
                )
            choices = obj.get("choices") or []
            if not choices:
                continue
            piece = (choices[0].get("delta") or {}).get("content") or ""
            if not piece:
                continue
            now = time.perf_counter()
            if first_at is None:
                first_at = now
                phase_at_first = read_phase(log)
            last_at = now
            text += piece
            n_out += 1
    if first_at is None:
        return {"ok": False, "text": "", "n_out": 0}
    fresh = max(prompt_tokens - cached_tokens, 0)
    prefill_s = first_at - t0
    decode_s = max(last_at - first_at, 1e-9)
    return {
        "ok": True,
        "text": text,
        "n_out": n_out,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "fresh_tokens": fresh,
        "prefill_s": prefill_s,
        "prefill_rate": (fresh / prefill_s) if prefill_s > 0 else 0.0,
        "phase_prefill": phase_at_first,
        "decode_rate": (n_out - 1) / decode_s if n_out > 1 else 0.0,
        "phase_decode": read_phase(log),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30030)
    ap.add_argument("--log", default="/spinning/serving-30030.boot.log")
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--doc-words", type=int, default=9000,
                    help="size of the document the agent carries in turn 1")
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    # A document big enough that turn 1 is REAL prefill work, built from a
    # fixed vocabulary so it is deterministic across runs.
    random.seed(20260809)
    words = [f"item{random.randint(0, 99999)}" for _ in range(args.doc_words)]
    doc = " ".join(words)

    messages: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": (
                "Here is a list of inventory tokens:\n\n" + doc +
                "\n\nAnswer briefly, in one short sentence: how would you "
                "describe the structure of this list?"
            ),
        }
    ]
    follow_ups = [
        "Now, in one short sentence: what would a good index for it be?",
        "In one short sentence: how would you detect duplicates?",
        "In one short sentence: how would you shard it across three nodes?",
        "In one short sentence: what would you cache first?",
        "In one short sentence: how would you compress it?",
        "In one short sentence: what would you monitor about it?",
    ]

    f0 = count_flips(args.log)
    print("#631 AGENTIC MULTI-TURN PROOF -- this script issues NO flip call.")
    print(f"start layout: {read_phase(args.log)}")
    print()
    print(f"{'turn':<5}{'prompt':>8}{'fresh':>8}  {'pfx phase':<10}"
          f"{'prefill tok/s':>14}  {'dec phase':<10}{'decode tok/s':>13}"
          f"{'out':>6}")
    print("-" * 80)

    rows: List[Dict] = []
    for t in range(1, args.turns + 1):
        r = turn(args.port, messages, args.max_tokens, args.timeout, args.log)
        if not r["ok"]:
            print(f"{t:<5}  TURN FAILED (no tokens)")
            return 1
        rows.append(r)
        print(
            f"{t:<5}{r['prompt_tokens']:>8}{r['fresh_tokens']:>8}  "
            f"{r['phase_prefill']:<10}{r['prefill_rate']:>14.1f}  "
            f"{r['phase_decode']:<10}{r['decode_rate']:>13.1f}{r['n_out']:>6}"
        )
        messages.append({"role": "assistant", "content": r["text"]})
        messages.append(
            {"role": "user", "content": follow_ups[(t - 1) % len(follow_ups)]}
        )

    f1 = count_flips(args.log)
    pp_to_tp = f1[0] - f0[0]
    tp_to_pp = f1[1] - f0[1]
    print("-" * 80)
    big = [r for r in rows if r["fresh_tokens"] >= 4096]
    if big:
        print(
            f"turns with REAL prefill work (>=4096 fresh tok): {len(big)}; "
            f"median {statistics.median([r['prefill_rate'] for r in big]):.1f}"
            f" tok/s, layouts {sorted({r['phase_prefill'] for r in big})}"
        )
    print(
        f"decode median "
        f"{statistics.median([r['decode_rate'] for r in rows]):.1f} tok/s, "
        f"layouts {sorted({r['phase_decode'] for r in rows})}"
    )
    print(f"automatic flips: pp_to_tp {pp_to_tp}, tp_to_pp {tp_to_pp} "
          f"(zero manual)")

    decode_in_tp = all(r["phase_decode"] == "tp" for r in rows)
    prefill_in_pp = all(r["phase_prefill"] == "pp" for r in big) if big else False
    print()
    print(f"decode ran in TP on every turn:      {decode_in_tp}")
    print(f"real prefill ran in PP on every turn: {prefill_in_pp}")
    print(f"switched automatically:               {pp_to_tp + tp_to_pp > 0}")
    ok = decode_in_tp and (prefill_in_pp or not big) and (pp_to_tp + tp_to_pp) > 0
    print(f"VERDICT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
