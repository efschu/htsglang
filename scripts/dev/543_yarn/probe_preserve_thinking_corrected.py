#!/usr/bin/env python3
"""Corrected preserve_thinking A/B (#544 follow-up).

Two fixes against
/spinning/gpu-battery-results/2026-08-04_541_thinking_ab/probe_preserve_thinking.py:

1. The original control arm sent NO ``chat_template_kwargs`` at all
   (``if preserve: body[...] = {...}``). That was a valid control only while the
   server had no default. Now that the server boots with
   ``--chat-template-default-kwargs '{"preserve_thinking": true}'``, omission
   INHERITS true, so both arms ran preserve_thinking=true and the reported
   77.2 % vs 6.4 % spread cannot be attributed to the kwarg. This version sends
   the value explicitly in both arms.
2. The original read global ``cached_tokens_total`` / ``prompt_tokens_total``
   counters, which any concurrent traffic on the shared server pollutes. This
   version reads the per-response usage block.

Each arm gets its own nonce, and the arms alternate order across repetitions so
an order or eviction effect cannot masquerade as a kwarg effect.
"""

import json
import time
import urllib.request

BASE = "http://127.0.0.1:30030"


def post(payload: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE}/v1/messages",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as fh:
        return json.loads(fh.read().decode())


def variant(nonce: str, preserve: bool) -> dict:
    user1 = (
        f"Case {nonce}. A scheduler has three queues with weights 5, 3 and 2 and "
        "must dispatch 40 items proportionally, but queue two only has 7 items. "
        "Work out the final per-queue counts and state them."
    )
    turn1 = post(
        {
            "model": "Qwen3.6-27B",
            "max_tokens": 700,
            "thinking": {"type": "adaptive"},
            "messages": [{"role": "user", "content": user1}],
        }
    )
    blocks = turn1["content"]
    thinking_blocks = [b for b in blocks if b.get("type") == "thinking"]

    turn2 = post(
        {
            "model": "Qwen3.6-27B",
            "max_tokens": 120,
            "thinking": {"type": "adaptive"},
            # The whole point: sent explicitly in BOTH arms.
            "chat_template_kwargs": {"preserve_thinking": preserve},
            "messages": [
                {"role": "user", "content": user1},
                {"role": "assistant", "content": blocks},
                {"role": "user", "content": "Now state only the count for queue three."},
            ],
        }
    )
    usage = turn2.get("usage", {}) or {}
    inp = usage.get("input_tokens", 0)
    cached = usage.get("cache_read_input_tokens", 0)
    return {
        "nonce": nonce,
        "preserve_thinking": preserve,
        "turn1_thinking_blocks": len(thinking_blocks),
        "turn1_thinking_chars": sum(len(b.get("thinking", "")) for b in thinking_blocks),
        "turn2_input_tokens": inp,
        "turn2_cache_read": cached,
        "reuse_pct": round(100.0 * cached / inp, 1) if inp else None,
        "usage_keys": sorted(usage),
    }


if __name__ == "__main__":
    stamp = str(int(time.time()))
    rows = []
    # Alternate order so an order effect cannot look like a kwarg effect.
    for rep, order in enumerate(((False, True), (True, False))):
        for preserve in order:
            tag = f"{'T' if preserve else 'F'}{rep}{stamp}"
            rows.append(variant(tag, preserve))
            print(json.dumps(rows[-1]))
    off = [r for r in rows if not r["preserve_thinking"]]
    on = [r for r in rows if r["preserve_thinking"]]
    print(
        "\nSUMMARY  preserve=False input/cached:",
        [(r["turn2_input_tokens"], r["turn2_cache_read"]) for r in off],
        "\n         preserve=True  input/cached:",
        [(r["turn2_input_tokens"], r["turn2_cache_read"]) for r in on],
    )
