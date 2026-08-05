"""Prefill throughput probe: unique long prompts, max_tokens=1, TTFT per request.

max_tokens=1 keeps decode out of the measurement, so the wall time is
prompt processing plus one sampled token. Prompts are made unique per run so
that neither the radix cache nor the hierarchical cache can serve a prefix
and silently turn the measurement into a cache-hit benchmark.
"""

import argparse
import json
import random
import statistics
import time
import urllib.request

WORDS = (
    "cache token buffer kernel stream tensor matrix vector pointer segment "
    "replay capture bucket padding latency throughput scheduler allocator "
    "gradient parameter attention residual embedding quantise dispatch"
).split()


def make_prompt(rng: random.Random, approx_tokens: int) -> str:
    # ~1 token per short word is close enough; the exact count is reported by
    # the server anyway via usage.prompt_tokens.
    return " ".join(rng.choice(WORDS) for _ in range(approx_tokens))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tokens", type=int, default=1900)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Distinct seed space per run keeps prompts unique ACROSS arms too.
    rng = random.Random(args.seed)

    def one(prompt: str) -> dict:
        payload = json.dumps(
            {
                "model": "default",
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": False,
            }
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{args.port}/v1/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read().decode())
        dt = time.perf_counter() - t0
        return {
            "s": dt,
            "prompt_tokens": body.get("usage", {}).get("prompt_tokens"),
            "cached": body.get("usage", {}).get("cached_tokens", 0) or 0,
        }

    prompts = [make_prompt(rng, args.tokens) for _ in range(args.warmup + args.n)]
    if args.concurrency <= 1:
        results = [one(p) for p in prompts]
        # Warmups ran first and serially, so the measured wall span is simply
        # the sum of the kept requests.
        wall = None
    else:
        # Concurrent arrivals so the scheduler forms multi-request prefill
        # batches -- that is the regime the bs>1 captured shapes exist for.
        from concurrent.futures import ThreadPoolExecutor

        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(one, prompts))
        wall = time.perf_counter() - t_start
    # discard warmup: the first calls carry autotune and lazy init
    samples = results[args.warmup :]

    durations = [s["s"] for s in samples]
    toks = [s["prompt_tokens"] for s in samples]
    cached_total = sum(s["cached"] for s in samples)
    # Per-request rate. Under concurrency this UNDERSTATES throughput, because
    # each request's wall time includes time queued behind the others -- read
    # aggregate_tok_s instead in that case.
    rates = [t / d for t, d in zip(toks, durations)]
    result = {
        "n": len(samples),
        "concurrency": args.concurrency,
        "prompt_tokens_median": statistics.median(toks),
        "cached_tokens_total": cached_total,
        "seconds_median": statistics.median(durations),
        "seconds_min": min(durations),
        "prefill_tok_s_median": statistics.median(rates),
        "prefill_tok_s_max": max(rates),
        "prefill_tok_s_all": rates,
        "seconds_all": durations,
        # Aggregate: total prompt tokens over the whole concurrent span,
        # including the warmups that shared the span. This is the number that
        # means "throughput" when concurrency > 1.
        "wall_seconds": wall,
        "aggregate_tok_s": (
            sum(s["prompt_tokens"] for s in results) / wall if wall else None
        ),
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(
        f"n={result['n']} prompt_tokens~{result['prompt_tokens_median']} "
        f"cached_total={cached_total} "
        f"median={result['seconds_median'] * 1000:.1f} ms "
        f"prefill={result['prefill_tok_s_median']:.1f} tok/s (median)"
    )
    if cached_total:
        print("WARNING: cache hits present -- prefill numbers are contaminated")


if __name__ == "__main__":
    main()
