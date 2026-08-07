#!/usr/bin/env python3
"""#616g KV-PRESSURE generator.

The 4-way verify-stress harness never reproduced the wedge in the A/B windows,
and the reason is measurable: full token usage stayed at a median of 0.20 and
peaked at 0.66. The defect under test needs the opposite regime -- the pool at
capacity, so `evict_from_tree_cache` actually fires and load-back actually
fails for want of device space. Those are the two rank-local triggers whose
divergence is the wedge.

So this generator does what verify-stress deliberately does not: it sends
UNIQUE long contexts, so nothing is served from the prefix cache and every
request must allocate. The pool is 460288 tokens globally; a handful of
concurrent 30K-token prompts fills it and then keeps it pinned at the eviction
boundary.

Prefill-dominated on purpose (max_tokens small): it is the extend path whose
token count diverged in the specimen.
"""

import argparse
import json
import random
import string
import sys
import threading
import time
import urllib.error
import urllib.request

WORDS = [
    "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 9)))
    for _ in range(4000)
]

stats = {"sent": 0, "ok": 0, "err": 0, "http_err": 0}
lock = threading.Lock()
stop = threading.Event()


def unique_prompt(approx_tokens: int, rng: random.Random) -> str:
    # ~0.75 words per token for this tokenizer class; overshoot slightly.
    n_words = int(approx_tokens * 0.8)
    head = f"doc-{rng.getrandbits(64):016x} "
    return head + " ".join(rng.choice(WORDS) for _ in range(n_words))


def worker(url: str, model: str, lo: int, hi: int, seed: int) -> None:
    rng = random.Random(seed)
    while not stop.is_set():
        size = rng.randint(lo, hi)
        body = json.dumps(
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": unique_prompt(size, rng)
                        + "\n\nReply with one word: ok",
                    }
                ],
                "max_tokens": 8,
                "temperature": 0.0,
            }
        ).encode()
        req = urllib.request.Request(
            url + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with lock:
            stats["sent"] += 1
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                r.read()
            with lock:
                stats["ok"] += 1
        except urllib.error.HTTPError:
            with lock:
                stats["http_err"] += 1
        except Exception:
            with lock:
                stats["err"] += 1
            time.sleep(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30030")
    ap.add_argument("--model", default="Qwen3.6-27B")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--min-tokens", type=int, default=20000)
    ap.add_argument("--max-tokens", type=int, default=45000)
    ap.add_argument("--minutes", type=float, default=30.0)
    a = ap.parse_args()

    threads = [
        threading.Thread(
            target=worker,
            args=(a.url, a.model, a.min_tokens, a.max_tokens, 1000 + i),
            daemon=True,
        )
        for i in range(a.concurrency)
    ]
    for t in threads:
        t.start()

    end = time.time() + a.minutes * 60
    while time.time() < end:
        time.sleep(20)
        with lock:
            print(
                f"{time.strftime('%H:%M:%S')} sent={stats['sent']} ok={stats['ok']} "
                f"http_err={stats['http_err']} err={stats['err']}",
                flush=True,
            )
    stop.set()
    print("pressure window over", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
