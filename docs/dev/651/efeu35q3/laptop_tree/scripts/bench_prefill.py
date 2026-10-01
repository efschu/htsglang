#!/usr/bin/env python
"""Prefill-throughput benchmark for the #651 laptop CPU+GPU PP split.

Measures the three numbers the layer-split decision needs:

  1. what the GPU achieves in prefill   (boot GPU-only, run this)
  2. what the CPU achieves in prefill   (boot CPU-only, run this)
  3. the compute-weighted layer split that follows from (1) and (2)

Usage:
    python bench_prefill.py --port 30040 --label gpu
    python bench_prefill.py --port 30041 --label cpu
    python bench_prefill.py --split-from gpu.json cpu.json

METHOD, and why each part is there (this project has been burned by all of
these):

* **Prefill is timed by TTFT with max_tokens=1.** Everything after the first
  token is decode, which is a different regime and would pollute the number.
* **Every prompt is UNIQUE** (a random nonce is embedded). Prefix caching would
  otherwise serve a cached prefill and report an absurd rate. The script also
  refuses to report if the server says any prompt was cached.
* **A-vs-A noise floor FIRST.** Two identical runs are compared before any
  cross-device claim is made. A difference smaller than the A-vs-A spread is
  not a result. This is a standing rule here and it has repeatedly turned an
  apparent win into noise.
* **Warmup is discarded.** The first run pays JIT, autotune and page-cache
  costs that never recur.
* **Runs are time-bounded, not count-bounded** (>= ~10 s of work per point),
  because short runs on this class of hardware are dominated by fixed costs.
* **Several prompt lengths.** Prefill is compute-bound only above some length;
  below it you are measuring overhead. The sweep shows where the regime starts,
  which is exactly the region the PP split cares about.

The output JSON feeds --split-from, which does the balance arithmetic.
"""

import argparse
import json
import random
import statistics
import string
import sys
import time
import urllib.request

# Layer geometry of Qwen3.6-35B-A3B (verified against the GGUF header and
# config.json; see docs/dev/HANDOFF_651_laptop.md).
N_DECODER_LAYERS = 40


def _nonce(n: int = 24) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def make_prompt(approx_tokens: int) -> str:
    """A prompt of roughly the requested token count, unique every call.

    Filler is prose-like rather than repeated tokens: a repeated single token
    can hit degenerate attention/routing paths and is not representative of
    real prefill work.
    """
    words = (
        "the system processes each request through several distinct stages "
        "before the final response is produced and returned to the caller "
        "which makes the behaviour easier to reason about in practice "
    ).split()
    out = [f"session {_nonce()} begins."]
    # ~0.75 words per token is a reasonable English approximation.
    need = int(approx_tokens * 0.75)
    while len(" ".join(out).split()) < need:
        out.append(" ".join(random.choices(words, k=12)))
    return " ".join(out)


def one_request(base: str, model: str, prompt: str, timeout: float):
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    dt = time.perf_counter() - t0
    usage = data.get("usage", {}) or {}
    return dt, usage.get("prompt_tokens"), usage.get("cached_tokens", 0)


def measure_point(base, model, approx_tokens, min_seconds, timeout):
    """Repeat until at least min_seconds of work has been done."""
    lat, toks, cached_total = [], [], 0
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < min_seconds:
        dt, ptok, cached = one_request(base, model, make_prompt(approx_tokens), timeout)
        lat.append(dt)
        toks.append(ptok or approx_tokens)
        cached_total += cached or 0
        if len(lat) >= 200:
            break
    ptok = statistics.median(toks)
    med = statistics.median(lat)
    return {
        "approx_tokens": approx_tokens,
        "prompt_tokens": ptok,
        "n": len(lat),
        "median_s": med,
        "min_s": min(lat),
        "tok_per_s": ptok / med if med else 0.0,
        "cached_tokens_total": cached_total,
    }


def run(args):
    base = f"http://127.0.0.1:{args.port}"
    lengths = [int(x) for x in args.lengths.split(",")]

    print(f"warmup ({args.warmup_s}s, discarded) ...")
    measure_point(base, args.model, lengths[-1], args.warmup_s, args.timeout)

    def sweep(tag):
        rows = []
        for L in lengths:
            r = measure_point(base, args.model, L, args.seconds, args.timeout)
            rows.append(r)
            print(
                f"  [{tag}] ~{L:6d} tok -> prompt_tokens={r['prompt_tokens']:6} "
                f"median {r['median_s']*1000:8.1f} ms  {r['tok_per_s']:9.1f} tok/s "
                f"(n={r['n']}, cached={r['cached_tokens_total']})"
            )
        return rows

    print("run A ...")
    a = sweep("A")
    print("run A' (noise floor) ...")
    a2 = sweep("A'")

    cached = sum(r["cached_tokens_total"] for r in a + a2)
    if cached:
        print(
            f"\nREFUSING to report: {cached} cached prompt tokens seen. "
            "Prefix caching is serving these prefills, so the rate is not a "
            "prefill measurement. Boot with --disable-radix-cache (or the "
            "equivalent) and re-run."
        )
        return 2

    print("\nA-vs-A noise floor (this is the bar any later claim must clear):")
    best = None
    for r1, r2 in zip(a, a2):
        s1, s2 = r1["tok_per_s"], r2["tok_per_s"]
        spread = abs(s1 - s2) / max(s1, s2) * 100 if max(s1, s2) else 0
        print(
            f"  ~{r1['approx_tokens']:6d} tok: {s1:9.1f} vs {s2:9.1f} tok/s "
            f"-> floor {spread:5.2f} %"
        )
        # the peak sustained rate is what the layer split should be weighted by
        cand = max(s1, s2)
        if best is None or cand > best["tok_per_s"]:
            best = {"tok_per_s": cand, "approx_tokens": r1["approx_tokens"]}

    out = {
        "label": args.label,
        "port": args.port,
        "peak_tok_per_s": best["tok_per_s"],
        "peak_at_tokens": best["approx_tokens"],
        "runA": a,
        "runA_prime": a2,
    }
    path = args.out or f"{args.label}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\npeak sustained prefill: {best['tok_per_s']:.1f} tok/s "
          f"(at ~{best['approx_tokens']} tokens)   -> {path}")
    return 0


def split_from(paths):
    data = []
    for p in paths:
        with open(p) as fh:
            data.append(json.load(fh))
    by = {d["label"]: d for d in data}
    if "gpu" not in by or "cpu" not in by:
        print("need one file labelled 'gpu' and one labelled 'cpu'")
        return 2
    rg = by["gpu"]["peak_tok_per_s"]
    rc = by["cpu"]["peak_tok_per_s"]
    if rc <= 0:
        print("cpu rate is zero; cannot weight a split")
        return 2
    ratio = rg / rc
    lg = N_DECODER_LAYERS * rg / (rg + rc)
    lc = N_DECODER_LAYERS - lg
    print(f"GPU prefill : {rg:10.1f} tok/s")
    print(f"CPU prefill : {rc:10.1f} tok/s")
    print(f"ratio       : {ratio:10.1f}x")
    print()
    print("Compute-weighted balance (equal stage time, L_gpu/L_cpu = R_gpu/R_cpu):")
    print(f"  GPU {lg:5.1f} layers   CPU {lc:5.1f} layers   of {N_DECODER_LAYERS}")
    print(f"  rounded: GPU {round(lg)}  CPU {N_DECODER_LAYERS - round(lg)}")
    print()
    print("Read this against the MEMORY-feasible split before choosing:")
    print("  each decoder layer is ~506 MiB of weights, so the number of layers")
    print("  the GPU can hold is (GPU memory for weights) / 506 MiB. If memory")
    print("  forces MORE layers onto the CPU than the balance above, the CPU")
    print("  stage is the bottleneck and prefill runs at roughly CPU speed --")
    print("  the split is then a memory fallback, not a speedup.")
    print()
    print("  Also: PP pipelines MICROBATCHES; it does not speed up a single")
    print("  prefill (stages run in sequence). The gain needs several chunked-")
    print("  prefill chunks or concurrent requests in flight.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30040)
    ap.add_argument("--model", default="Qwen3.6-35B-A3B-Q4")
    ap.add_argument("--label", default="gpu", help="gpu | cpu")
    ap.add_argument("--lengths", default="512,2048,8192")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--warmup-s", type=float, default=8.0)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--out")
    ap.add_argument("--split-from", nargs=2, metavar=("GPU_JSON", "CPU_JSON"))
    args = ap.parse_args()
    if args.split_from:
        return split_from(args.split_from)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
