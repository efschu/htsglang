# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""A representative decode load on the 27B, so TTS can be measured beside it.

The translator's talker shares a card with the 27B on port 30030. The field
session of 2026-08-04 carries a 61 s synthesis stall under exactly that
sharing, so the co-tenancy term in the latency budget is large and has to be
MEASURED rather than assumed away.

This puts load on the 27B only through its ordinary OpenAI-compatible
endpoint. It does not restart, reconfigure or otherwise touch that server --
the point is to reproduce what a working rig does to the talker, not to
construct a worst case that never happens.

Streaming is used so the load is a sustained DECODE, which is the phase that
competes with the talker for SM time. A non-streaming request of the same
size would spend part of its life in prefill and understate the contention.

    python scripts/translator/tts_cotenancy_load.py --concurrency 2 --seconds 240
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request

PROMPT = (
    "Explain, in careful and complete prose, how a speculative decoding "
    "scheme verifies draft tokens against a target model, and why the "
    "acceptance length matters for throughput. Cover the tree-structured "
    "variant as well."
)


def one_stream(base_url: str, model: str, max_tokens: int, stop: threading.Event,
               counter: dict, index: int) -> None:
    while not stop.is_set():
        body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": True,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode()
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                for raw in response:
                    if stop.is_set():
                        break
                    if raw.startswith(b"data: "):
                        counter[index] = counter.get(index, 0) + 1
        except Exception as exc:  # keep the load alive across a hiccup
            counter[f"err{index}"] = str(exc)
            time.sleep(1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30030/v1")
    parser.add_argument("--model", default="default")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seconds", type=float, default=240.0)
    args = parser.parse_args()

    stop = threading.Event()
    counter: dict = {}
    threads = [
        threading.Thread(
            target=one_stream,
            args=(args.base_url, args.model, args.max_tokens, stop, counter, i),
            daemon=True,
        )
        for i in range(args.concurrency)
    ]
    started = time.time()
    for thread in threads:
        thread.start()
    print(json.dumps({"event": "load_started", "concurrency": args.concurrency}))
    try:
        while time.time() - started < args.seconds:
            time.sleep(1.0)
    finally:
        stop.set()
    elapsed = time.time() - started
    chunks = sum(v for k, v in counter.items() if isinstance(k, int))
    print(
        json.dumps(
            {
                "event": "load_stopped",
                "seconds": round(elapsed, 1),
                "stream_chunks": chunks,
                "chunks_per_s": round(chunks / elapsed, 1) if elapsed else None,
                "errors": {k: v for k, v in counter.items() if not isinstance(k, int)},
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
