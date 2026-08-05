"""Prefill throughput probe: sustained load, completion-span scoring, clock evidence.

Measurement discipline (user directive 2026-08-05):

* A single short draw on idle cards measures the clock ramp, not the code.
  Every point is a SUSTAINED window of back-to-back draws with no gaps, and a
  discarded warmup runs first.
* ~5 s of sustained measured load is the STANDARD, not a floor to exceed.
* SM clock and P-state are sampled DURING the measured span and reported, so
  clock state is evidence rather than an assumption.

Scoring -- ms per FIXED unit of work:

  The reference metric is milliseconds per prefill over an IDENTICAL prompt
  set: same prompts, same count, same order in every arm, generated from a
  fixed seed. Identical work is what makes the arms comparable, and it is
  valid here precisely because the power limit is identical across all runs
  (200/400/200 W). Tokens/s is reported too, but it is the same number
  rescaled -- the work is fixed.

  Clock and power draw are sampled together and reported as DIAGNOSTIC
  ANNOTATION ONLY. At a fixed power limit a lower clock usually means HIGHER
  load, not a disadvantage: a power-limited card drops clocks when it is doing
  more work per cycle, especially with high power draw at the same time. Low
  clock with LOW power is the light-load case. So clock alone proves nothing
  in either direction, and nothing here is ever normalised by clock.

Superseded scoring -- why not tokens/wall:

  The first version divided total tokens by the whole wall span. That charges
  the run for its DRAIN: after the deadline the in-flight draws still have to
  finish, and the length of that tail varies run to run. It cost a real
  measurement -- window 3's 256c4 point produced an eager-vs-eager floor of
  +9.1 % from two runs that had done byte-identical work (32 draws, 8523
  tokens); the whole gap was 0.471 s of extra drain in one of them.

  A fixed 5.000 s denominator does not fix it either: with ~1.15 s draws only
  four complete inside the window, so dropping the partial fifth quantises the
  estimate by more than 20 %.

  This version scores the COMPLETION SPAN. With completions at c1 < ... < cN,
  throughput is the work completed in (c1, cN] divided by the measured span
  (cN - c1). Both boundaries are real completion instants, so ramp before c1
  and drain after cN are excluded, the denominator is measured rather than
  assumed, and nothing is quantised away.

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
    """Samples SM clock, P-state and POWER DRAW per GPU, timestamped so the
    summary can be restricted to the span actually being scored.

    Clock and power are only meaningful together: at a fixed power limit, low
    clock + high power is a heavily loaded card, while low clock + low power is
    an idle one. Both are recorded so the reader can tell those apart; neither
    is a validity criterion.
    """

    def __init__(self, interval_ms: int = 200):
        self.cmd = [
            "nvidia-smi",
            "--query-gpu=index,clocks.sm,pstate,power.draw",
            "--format=csv,noheader,nounits",
            f"-lms={interval_ms}",
        ]
        self.proc = None
        self.rows = []  # (t, gpu, sm_mhz, pstate, watts)

    def _reader(self):
        for line in self.proc.stdout:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                continue
            try:
                self.rows.append(
                    (
                        time.perf_counter(),
                        int(parts[0]),
                        int(parts[1]),
                        parts[2],
                        float(parts[3]),
                    )
                )
            except ValueError:
                continue

    def __enter__(self):
        try:
            self.proc = subprocess.Popen(
                self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
            threading.Thread(target=self._reader, daemon=True).start()
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

    def summary(self, t_lo=None, t_hi=None) -> dict:
        by_gpu, states, watts = {}, {}, {}
        for t, gpu, sm, pstate, w in self.rows:
            if t_lo is not None and not (t_lo <= t <= t_hi):
                continue
            by_gpu.setdefault(gpu, []).append(sm)
            states.setdefault(gpu, set()).add(pstate)
            watts.setdefault(gpu, []).append(w)
        return {
            str(g): {
                "sm_min": min(v),
                "sm_median": int(statistics.median(v)),
                "sm_max": max(v),
                "watt_median": round(statistics.median(watts[g]), 1),
                "watt_max": round(max(watts[g]), 1),
                "samples": len(v),
                "pstates": sorted(states[g]),
            }
            for g, v in sorted(by_gpu.items())
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tokens", type=int, default=1900)
    ap.add_argument("--seconds", type=float, default=5.0, help="measured span")
    ap.add_argument("--warmup-seconds", type=float, default=2.0)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--prompts", type=int, default=8, help="prompts per pass (the fixed work unit)"
    )
    ap.add_argument(
        "--passes",
        type=int,
        default=1,
        help="passes over the set; raise until wall >= --seconds",
    )
    ap.add_argument("--warmup-draws", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

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
        t1 = time.perf_counter()
        usage = body.get("usage", {})
        return {
            "start": t0,
            "done": t1,
            "s": t1 - t0,
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached": usage.get("cached_tokens", 0) or 0,
        }

    def run_set(prompts):
        """Run an identical, fixed list of prompts to completion.

        Wall time spans the whole list, drain included -- that is legitimate
        here because the WORK is fixed: every arm runs the same prompts, the
        same number of them, in the same order. Whatever the tail costs, both
        arms pay it.
        """
        idx = {"i": 0}
        ilock = threading.Lock()
        out = []
        olock = threading.Lock()

        def worker():
            while True:
                with ilock:
                    i = idx["i"]
                    if i >= len(prompts):
                        return
                    idx["i"] = i + 1
                r = one(prompts[i])
                with olock:
                    out.append(r)

        threads = [
            threading.Thread(target=worker) for _ in range(max(1, args.concurrency))
        ]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return out, t0, time.perf_counter()

    # Prompt lists are drawn from a seeded RNG in a fixed order, so every arm
    # sees byte-identical work. They are all distinct within a run, so no
    # prefix cache can serve them.
    warmup_prompts = [make_prompt(rng, args.tokens) for _ in range(args.warmup_draws)]
    n_measured = args.prompts * args.passes
    measured_prompts = [make_prompt(rng, args.tokens) for _ in range(n_measured)]

    run_set(warmup_prompts)  # discarded: clock ramp, lazy init, autotune

    with ClockSampler() as clocks:
        draws, t_start, t_end = run_set(measured_prompts)

    wall = t_end - t_start
    tokens = sum(d["prompt_tokens"] for d in draws)
    cached_total = sum(d["cached"] for d in draws)
    per_draw = [d["s"] for d in draws]

    result = {
        "tokens_arg": args.tokens,
        "concurrency": args.concurrency,
        "prompts_per_pass": args.prompts,
        "passes": args.passes,
        # The fixed unit of work. Identical across arms by construction.
        "prefills": len(draws),
        "wall_seconds": wall,
        # THE reference metric.
        "ms_per_prefill": wall * 1000.0 / len(draws),
        "total_prompt_tokens": tokens,
        "prompt_tokens_median": statistics.median([d["prompt_tokens"] for d in draws]),
        "cached_tokens_total": cached_total,
        # Same measurement rescaled; kept for continuity with earlier windows.
        "aggregate_tok_s": tokens / wall,
        "draw_seconds_median": statistics.median(per_draw),
        "draw_seconds_min": min(per_draw),
        "draw_seconds_max": max(per_draw),
        # DIAGNOSTIC ANNOTATION ONLY -- never a validity criterion, never a
        # normaliser. Read clock and power together or not at all.
        "clocks": clocks.summary(t_start, t_end),
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1)

    band = " ".join(
        f"g{i}:{v['sm_median']}MHz/{v['watt_median']}W"
        for i, v in result["clocks"].items()
    )
    print(
        f"tokens~{args.tokens} conc={args.concurrency} "
        f"prefills={len(draws)} wall={wall:.2f}s "
        f"ms/prefill={result['ms_per_prefill']:.1f} "
        f"({result['aggregate_tok_s']:.1f} tok/s)  diag[{band}]"
    )
    if cached_total:
        print("WARNING: cache hits present -- prefill numbers are contaminated")
    if wall < args.seconds:
        print(
            f"WARNING: measured wall {wall:.2f}s < {args.seconds}s sustained "
            f"standard -- raise --passes for this point"
        )


if __name__ == "__main__":
    main()
