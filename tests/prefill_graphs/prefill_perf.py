"""Prefill throughput probe: sustained continuous load, with clock evidence.

Measurement discipline (user directive 2026-08-05, the #483 clock-ramp defect):

* A single ~1 s draw on idle cards measures the clock ramp, not the code. Every
  point here is therefore a SUSTAINED window of back-to-back draws with no
  gaps, and the reported number is the aggregate over that whole window --
  never the median of per-draw rates.
* ~5 s of sustained measured load is the STANDARD, not a floor to exceed:
  long enough for the cards to clock up and the ramp to vanish into noise,
  short enough that this stays a probe and not a battery.
* A warmup window runs first and is discarded.
* SM clock and P-state are sampled DURING the measured window and reported, so
  the clock state is evidence rather than an assumption.

Prompts are unique per draw so neither the radix cache nor the hierarchical
cache can turn the measurement into a cache-hit benchmark.
"""

import argparse
import json
import random
import statistics
import subprocess
import threading
import time
import urllib.request

WORDS = (
    "cache token buffer kernel stream tensor matrix vector pointer segment "
    "replay capture bucket padding latency throughput scheduler allocator "
    "gradient parameter attention residual embedding quantise dispatch"
).split()


def make_prompt(rng: random.Random, approx_tokens: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(approx_tokens))


class ClockSampler:
    """Samples SM clock + P-state per GPU while a measured window runs."""

    def __init__(self, interval_ms: int = 200):
        self.cmd = [
            "nvidia-smi",
            "--query-gpu=index,clocks.sm,pstate",
            "--format=csv,noheader,nounits",
            f"-lms={interval_ms}",
        ]
        self.proc = None
        self.rows = []
        self._thread = None

    def _reader(self):
        for line in self.proc.stdout:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                self.rows.append((int(parts[0]), int(parts[1]), parts[2]))
            except ValueError:
                continue

    def __enter__(self):
        try:
            self.proc = subprocess.Popen(
                self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()
        except Exception:
            self.proc = None
        return self

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()

    def summary(self) -> dict:
        by_gpu = {}
        states = {}
        for idx, sm, pstate in self.rows:
            by_gpu.setdefault(idx, []).append(sm)
            states.setdefault(idx, set()).add(pstate)
        return {
            str(i): {
                "sm_min": min(v),
                "sm_median": int(statistics.median(v)),
                "sm_max": max(v),
                "samples": len(v),
                "pstates": sorted(states[i]),
            }
            for i, v in sorted(by_gpu.items())
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tokens", type=int, default=1900)
    ap.add_argument("--seconds", type=float, default=5.0, help="measured window")
    ap.add_argument("--warmup-seconds", type=float, default=2.0)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rng_lock = threading.Lock()

    def next_prompt() -> str:
        with rng_lock:
            return make_prompt(rng, args.tokens)

    def one() -> dict:
        payload = json.dumps(
            {
                "model": "default",
                "prompt": next_prompt(),
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
        return {
            "s": time.perf_counter() - t0,
            "prompt_tokens": body.get("usage", {}).get("prompt_tokens"),
            "cached": body.get("usage", {}).get("cached_tokens", 0) or 0,
        }

    def run_until(duration_s: float) -> list:
        """Back-to-back draws, no gaps, until the deadline is crossed.

        The draw that crosses the deadline is kept: truncating it would drop
        work that was really performed and inflate the rate.
        """
        got = []
        glock = threading.Lock()
        stop_at = time.perf_counter() + duration_s

        def worker():
            while time.perf_counter() < stop_at:
                r = one()
                with glock:
                    got.append(r)

        if args.concurrency <= 1:
            worker()
        else:
            threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        return got

    # Warmup window -- discarded. Its only job is to get the cards clocked up
    # and the lazy init / autotune out of the way before anything is recorded.
    run_until(args.warmup_seconds)

    with ClockSampler() as clocks:
        t_start = time.perf_counter()
        samples = run_until(args.seconds)
        wall = time.perf_counter() - t_start

    total_tokens = sum(s["prompt_tokens"] for s in samples)
    cached_total = sum(s["cached"] for s in samples)
    per_draw = [s["s"] for s in samples]
    result = {
        "tokens_arg": args.tokens,
        "concurrency": args.concurrency,
        "window_seconds": wall,
        "draws": len(samples),
        "prompt_tokens_median": statistics.median(
            [s["prompt_tokens"] for s in samples]
        ),
        "total_prompt_tokens": total_tokens,
        "cached_tokens_total": cached_total,
        # THE number: aggregate over the sustained window.
        "aggregate_tok_s": total_tokens / wall,
        "draw_seconds_median": statistics.median(per_draw),
        "draw_seconds_min": min(per_draw),
        "draw_seconds_max": max(per_draw),
        "clocks": clocks.summary(),
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1)

    band = " ".join(
        f"gpu{i}:{v['sm_min']}-{v['sm_max']}MHz/{'|'.join(v['pstates'])}"
        for i, v in result["clocks"].items()
    )
    print(
        f"tokens~{args.tokens} conc={args.concurrency} "
        f"window={wall:.2f}s draws={len(samples)} "
        f"AGGREGATE={result['aggregate_tok_s']:.1f} tok/s  clocks[{band}]"
    )
    if cached_total:
        print("WARNING: cache hits present -- prefill numbers are contaminated")
    if wall < args.seconds:
        print(f"WARNING: measured window {wall:.2f}s < requested {args.seconds}s")


if __name__ == "__main__":
    main()
